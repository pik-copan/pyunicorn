#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Simple tests for the core Network class.
"""

import numpy as np
import scipy.sparse as sp

from pyunicorn import Network


def testIntOverflow():
    """
    Avoid integer overflow in scipy.sparse representation.
    """
    for n in [10, 200, 2000, 33000]:
        adj = sp.lil_matrix((n, n), dtype=np.int8)
        adj[0, 1:] = 1
        deg = Network(adjacency=adj).degree()
        assert (deg.min(), deg.max()) == (0, n - 1)
