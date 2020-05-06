from pathlib import Path
from sys import stderr, exit

import click
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter
from vtk.util.numpy_support import numpy_to_vtk


def load_vtk(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def _make_linspace(vmin, vmax, npts):
    if vmin == vmax:
        return [vmin]
    return np.linspace(vmin, vmax, npts)


def _extract_and_cartesify(data, bbox, shape):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    # Create one-dimensional grids
    xvalues = _make_linspace(xmin, xmax, shape[0])
    yvalues = _make_linspace(ymin, ymax, shape[1])
    zvalues = _make_linspace(zmin, zmax, shape[2])
    final_shape = tuple(len(k) for k in (xvalues, yvalues, zvalues) if len(k) > 1)

    # Create a cartesian grid
    x, y, z = np.meshgrid(xvalues, yvalues, zvalues, indexing='ij')
    pointsdata = np.stack([x.flat, y.flat, z.flat], axis=1)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(pointsdata))

    # Create a vtkStructuredGrid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(len(xvalues), len(yvalues), len(zvalues))
    grid.SetPoints(points)

    # Create a vtkProbeFilter for interpolating
    probefilter = vtk.vtkProbeFilter()
    probefilter.SetSourceData(data)
    probefilter.SetInputData(grid)
    probefilter.Update()

    # Grab data from interpolated output
    struct = probefilter.GetStructuredGridOutput()
    struct = dataset_adapter.WrapDataObject(struct)

    # Create numpy datatype
    spec = []
    for key in struct.PointData.keys():
        array = struct.PointData[key]
        shape = () if array.ndim == 1 else (array.shape[-1],)
        spec.append((key, array.dtype, shape))
    datatype = np.dtype(spec)

    # Finalize output array
    output = np.zeros(final_shape, dtype=datatype)
    for key in struct.PointData.keys():
        output[key] = struct.PointData[key].reshape(output[key].shape)
    return output


def cartesify(data, shape):
    return _extract_and_cartesify(data, data.GetBounds(), shape)


def cutplane(data, point, normal, shape):
    xmin, xmax, ymin, ymax, zmin, zmax = data.GetBounds()
    if normal == 'x':
        xmin = xmax = point
    elif normal == 'y':
        ymin = ymax = point
    elif normal == 'z':
        zmin = zmax = point
    return _extract_and_cartesify(data, (xmin, xmax, ymin, ymax, zmin, zmax), shape)


@click.group()
def main():
    """VTK command-line utilities."""
    pass


@main.command('cartesify')
@click.option('--shape', default=(20, 20, 20), nargs=3, type=int, help='Shape of resulting array')
@click.argument('filenames', type=click.Path(exists=True, dir_okay=False, readable=True), nargs=-1)
def _cartesify(shape, filenames):
    """Convert data to cartesian grid and output as npy."""
    for filename in filenames:
        grid = load_vtk(filename)
        output = cartesify(grid, shape)
        np.save(Path(filename).with_suffix('.npy'), output)


@main.command('cutplane')
@click.option('--normal', '-n', type=click.Choice(['x', 'y', 'z'], case_sensitive=False), help='Normal plane direction')
@click.option('--point', '-p', default=0.0, type=float, help='A coordinate in the normal direction')
@click.option('--shape', default=(20, 20, 20), nargs=3, type=int, help='Shape of resulting array')
@click.argument('filenames', type=click.Path(exists=True, dir_okay=False, readable=True), nargs=-1)
def _cutplane(point, normal, shape, filenames):
    """Extract a cutting plane from 3D data."""
    for filename in filenames:
        grid = load_vtk(filename)
        output = cutplane(grid, point, normal, shape)
        np.save(Path(filename).with_suffix('.npy'), output)
