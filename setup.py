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
    for pkg in ['core', 'timeseries']]
extensions += [
    Extension('pyunicorn.timeseries.numerics',
    sources=['pyunicorn/timeseries/numerics.pyx',
            'pyunicorn/timeseries/_ext/src_fast_surrogate.c'],
    include_dirs=[np.get_include()], extra_compile_args=['-O3', '-std=c99'])]

if CYTHON:
    extensions = cythonize(extensions, compiler_directives={
        'language_level': 2, 'embedsignature': True,
        'boundscheck': False, 'wraparound': False, 'initializedcheck': False,
        'nonecheck': False})

setup(
    name='pyunicorn',
    version='0.5.2',
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
              'pyunicorn.timeseries', 'pyunicorn.timeseries._ext', 'pyunicorn.funcnet',
              'pyunicorn.utils', 'pyunicorn.utils.progressbar'],
    ext_modules=extensions,
    install_requires=open('requirements.txt', 'r').read().split('\n'),
    provides=['pyunicorn'],
    scripts=[],
    license='BSD',
)
