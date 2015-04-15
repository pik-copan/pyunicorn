#!/usr/bin/env python

from distutils.core import setup

setup(
    name='pyunicorn',
    version='0.4.1',
    # metadata for upload to PyPI
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
    requires=['numpy (>=1.8)', 'scipy (>=0.14)', 'weave (>=0.15)',
              'pythonigraph (>=0.7)'],
    provides=['pyunicorn'],
    license='BSD',
)
