#!/usr/bin/env python

from setuptools import setup
from setuptools.extension import Extension

import numpy as np

try:
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False

extensions = [
    Extension('*', ['pyunicorn/%s/*.%s' % (pkg, 'pyx' if CYTHON else 'c')],
              include_dirs=[np.get_include()])
    for pkg in ['core']]

if CYTHON:
    extensions = cythonize(extensions, compiler_directives={
        'language_level': 2, 'embedsignature': True,
        'boundscheck': False, 'wraparound': False, 'initializedcheck': False,
        'nonecheck': False})

setup(
    name='pyunicorn',
    version='0.5.1',
    description="Unified complex network and recurrence analysis toolbox",
    long_description="Advanced statistical analysis and modeling of \
general and spatially embedded complex networks with applications to \
multivariate nonlinear time series analysis",
    keywords='complex networks statistics modeling time series analysis \
nonlinear climate recurrence plot surrogates spatial model',
    author='Jonathan F. Donges',
    author_email='donges@pik-potsdam.de',
    url='http://www.pik-potsdam.de/~donges/pyunicorn/',
    platforms=['all'],
    packages=['pyunicorn', 'pyunicorn.core', 'pyunicorn.climate',
              'pyunicorn.timeseries', 'pyunicorn.funcnet',
              'pyunicorn.utils', 'pyunicorn.utils.progressbar'],
    ext_modules=extensions,
    requires=['numpy (>=1.8)', 'scipy (>=0.14)', 'weave (>=0.15)',
              'pythonigraph (>=0.7)'],
    provides=['pyunicorn'],
    scripts=[],
    license='BSD',
)
