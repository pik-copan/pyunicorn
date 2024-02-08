# This file is part of pyunicorn.
# Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
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

from platform import system
from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np


# ==============================================================================


win = system() == 'Windows'
c_args = {
    'include_dirs': [np.get_include()],
    'extra_compile_args': ['/O2'] if win else ['-O3', '-std=c99', '-Wall'],
    'define_macros': [('_GNU_SOURCE', None),
                      ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]}
cy_args = {
    'language_level': '3str', 'embedsignature': True,
    'boundscheck': True, 'wraparound': False,
    'initializedcheck': True, 'nonecheck': True,
    'warn.unused': True, 'warn.unused_arg': False, 'warn.unused_result': False}


# ==============================================================================


setup(
    ext_modules=cythonize(
        [Extension(
            f'pyunicorn.{pkg}._ext.numerics',
            sources=[f'src/pyunicorn/{pkg}/_ext/numerics.pyx'],
            **c_args)
         for pkg in ['climate', 'core', 'funcnet', 'timeseries']],
        compiler_directives=cy_args,
        nthreads=4))
