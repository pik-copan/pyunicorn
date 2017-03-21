#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Simple tests for the timeseries class.
"""

import numpy as np

from pyunicorn.timeseries import Surrogates
from pyunicorn.core.data import Data

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


# -----------------------------------------------------------------------------
# surrogates
# -----------------------------------------------------------------------------

# turn off for weave compilation & error detection
parallel = False

def test_TestPearsonCorrelation():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    n_index, n_times = tdata.shape
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:,None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:,None]

    norm = 1.0 / float(n_times)

    c = Surrogates.test_pearson_correlation(tdata, tdata, fast=True)
    corrcoef = np.corrcoef(tdata, tdata)[n_index:,:n_index]*norm
    for i in xrange(n_index):
        corrcoef[i,i]=0.0

    assert c.shape == (n_index, n_index)
    assert_array_almost_equal(c, corrcoef, decimal=5)
