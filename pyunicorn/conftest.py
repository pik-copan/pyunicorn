#! /usr/bin/env python2

# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Configure `py.test` fixtures.
"""

import numpy as np
import pytest
from pyunicorn import legacy


legacy.set_legacy_mode()


def _i():
    """
    Match preceding stdout (e.g. from `weave`) on '#'.
    Otherwise default doctest.ELLIPSIS = '...' is indistinguishable from line
    continuation in doctest.
    """
    import doctest
    doctest.ELLIPSIS_MARKER = '#'


def r(obj, decimals=4):
    """
    Round numbers, arrays or iterables thereof for doctests.
    """
    if isinstance(obj, (np.ndarray, np.matrix)):
        if obj.dtype.kind == 'f':
            rounded = np.around(obj.astype(np.float128),
                                decimals=decimals).astype(np.float)
        elif obj.dtype.kind == 'i':
            rounded = obj.astype(np.int)
    elif isinstance(obj, list):
        rounded = map(r, obj)
    elif isinstance(obj, tuple):
        rounded = tuple(map(r, obj))
    elif isinstance(obj, (float, np.float32, np.float64, np.float128)):
        rounded = np.float(np.around(np.float128(obj), decimals=decimals))
    elif isinstance(obj, (int, np.int8, np.int16)):
        rounded = int(obj)
    else:
        rounded = obj
    return rounded


def rr(obj, decimals=4):
    """
    Force arrays in stubborn scientific notation into a few digits.
    """
    print(np.vectorize('%.4g'.__mod__)(r(obj, decimals=decimals)))


@pytest.fixture(autouse=True)
def add_round(doctest_namespace):
    """
    Inject rounding helpers into doctest namespace.
    """
    doctest_namespace['_i'] = _i
    doctest_namespace['r'] = r
    doctest_namespace['rr'] = rr
