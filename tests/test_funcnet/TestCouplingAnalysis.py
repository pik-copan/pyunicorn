#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Simple tests for the funcnet CouplingAnalysis class.
"""
import numpy as np

from pyunicorn.core.data import Data
from pyunicorn.funcnet import CouplingAnalysis

from numpy.testing import assert_array_almost_equal

def create_test_data():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    n_index, n_times = tdata.shape
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:,None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:,None]
    return tdata

def testSymmetrizeByAbsmax():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    coup_ana = CouplingAnalysis(tdata)
    similarity_matrix = np.random.rand(n_index, n_times).astype('float32')
    lag_matrix = np.random.rand(n_index, n_times).astype(np.int8)
    sm_new = coup_ana.symmetrize_by_absmax(similarity_matrix, lag_matrix)[0]
    for i in range(n_index):
        for j in range(n_times):
            assert sm_new[i,j] >= similarity_matrix[i,j]
