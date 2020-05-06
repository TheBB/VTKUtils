from pathlib import Path
from sys import stderr, exit

import click
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter
from vtk.util.numpy_support import numpy_to_vtk


def load_vtk(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName('Re10_000_vorticity_zNormal.vtk')
    reader.Update()
    return reader


def _make_linspace(vmin, vmax, npts):
    if vmin == vmax:
        return [vmin]
    return np.linspace(vmin, vmax, npts)


def _cartesify(filename, reader, xpts, ypts, zpts):
    xmin, xmax, ymin, ymax, zmin, zmax = reader.GetOutput().GetBounds()

    xvalues = _make_linspace(xmin, xmax, xpts)
    yvalues = _make_linspace(ymin, ymax, ypts)
    zvalues = _make_linspace(zmin, zmax, zpts)
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
    probefilter.SetSourceConnection(reader.GetOutputPort())
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

    np.save(filename, output)


@click.group()
def main():
    """VTK command-line utilities"""
    pass


@main.command('cartesify')
@click.option('--xpts', '-x', default=20, help='Number of points in x-direction')
@click.option('--ypts', '-y', default=20, help='Number of points in y-direction')
@click.option('--zpts', '-z', default=20, help='Number of points in z-direction')
@click.argument('filenames', type=click.Path(exists=True, dir_okay=False, readable=True), nargs=-1)
def cartesify(xpts, ypts, zpts, filenames):
    """Convert data to cartesian grid and output as npy."""
    for filename in filenames:
        grid = load_vtk(filename)
        _cartesify(Path(filename).with_suffix('.npy'), grid, xpts, ypts, zpts)
