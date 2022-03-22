# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [Extension(
            f'pyunicorn.{pkg}._ext.numerics',
            sources=[f'pyunicorn/{pkg}/_ext/numerics.pyx'],
            # sources=[f'src/pyunicorn/{pkg}/_ext/numerics.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O3', '-std=c99', '-D_GNU_SOURCE'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
         for pkg in ['climate', 'core', 'funcnet', 'timeseries']],
        compiler_directives={
            'language_level': 3, 'embedsignature': True,
            'boundscheck': False, 'wraparound': False,
            'initializedcheck': False, 'nonecheck': False},
        nthreads=4))
