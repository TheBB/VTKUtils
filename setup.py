#!/usr/bin/env python

from pathlib import Path
from setuptools import setup
from distutils.extension import Extension

setup(
    name='VTKUtils',
    version='1.0.3',
    description='Various CLI tools for manipulating VTK files.',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=['vtkutils'],
    install_requires=['click', 'numpy', 'vtk'],
    entry_points={
        'console_scripts': [
            'vtkutils=vtkutils.__main__:main',
        ],
    },
)
