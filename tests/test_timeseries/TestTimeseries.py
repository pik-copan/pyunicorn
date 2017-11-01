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

from pyunicorn.timeseries import CrossRecurrencePlot
from pyunicorn.timeseries import VisibilityGraph
from pyunicorn.timeseries import Surrogates
from pyunicorn.core.data import Data

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

# turn off for weave compilation & error detection
parallel = False

def create_test_data():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    n_index, n_times = tdata.shape
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:,None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:,None]
    return tdata


# -----------------------------------------------------------------------------
# cross_recurrence_plot
# -----------------------------------------------------------------------------

def testManhattanDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    c = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    manh_dist = c.manhattan_distance_matrix(tdata.T, tdata.T)

def testEuclideanDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    c = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    eucl_dist = c.euclidean_distance_matrix(tdata.T, tdata.T)

def testSupremumDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    c = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    supr_dist = c.supremum_distance_matrix(tdata.T, tdata.T)


# -----------------------------------------------------------------------------
# surrogates
# -----------------------------------------------------------------------------

def testTwinSurrogates():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    s = Surrogates(tdata)
    tsurro = s.twin_surrogates(tdata, 1, 0, 0.2)
    corrcoef = np.corrcoef(tdata, tsurro)[n_index:,:n_index]
    for i in xrange(n_index):
        corrcoef[i,i]=0.0
    assert (corrcoef>=-1.0).all() and (corrcoef<=1.0).all()


def test_TestPearsonCorrelation():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    norm = 1.0 / float(n_times)
    c = Surrogates.test_pearson_correlation(tdata, tdata)
    corrcoef = np.corrcoef(tdata, tdata)[n_index:,:n_index]*norm
    for i in xrange(n_index):
        corrcoef[i,i]=0.0

    assert c.shape == (n_index, n_index)
    assert_array_almost_equal(c, corrcoef, decimal=5)


def test_TestMutualInformation():
    tdata = create_test_data()
    n_bins=32
    test_mi = Surrogates.test_mutual_information(tdata[:1], tdata[:1],
                                                 n_bins=n_bins)
    assert (test_mi>=-1.0).all() and (test_mi<=1.0).all()
