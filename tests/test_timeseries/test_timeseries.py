#!/usr/bin/python
# -*- coding: utf-8 -*-
#
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

"""
Simple tests for the timeseries class.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from pyunicorn.timeseries import CrossRecurrencePlot, VisibilityGraph, \
    Surrogates
from pyunicorn.core.data import Data


# turn off for weave compilation & error detection
parallel = False

def create_test_data():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    n_index, n_times = tdata.shape
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:, None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:, None]
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
    corrcoef = np.corrcoef(tdata, tsurro)[n_index:, :n_index]
    for i in range(n_index):
        corrcoef[i, i] = 0.0
    assert (corrcoef >= -1.0).all() and (corrcoef <= 1.0).all()


def test_TestPearsonCorrelation():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    norm = 1.0 / float(n_times)
    c = Surrogates.test_pearson_correlation(tdata, tdata)
    corrcoef = np.corrcoef(tdata, tdata)[n_index:, :n_index]*norm
    for i in range(n_index):
        corrcoef[i, i] = 0.0

    assert c.shape == (n_index, n_index)
    assert_array_almost_equal(c, corrcoef, decimal=5)


def test_TestMutualInformation():
    tdata = create_test_data()
    test_mi = Surrogates.test_mutual_information(tdata[:1], tdata[-1:],
                                                 n_bins=32)
    assert (test_mi >= -1.0).all() and (test_mi <= 1.0).all()

# -----------------------------------------------------------------------------
# visibility_graph
# -----------------------------------------------------------------------------

def create_test_timeseries():
    """
    Return test data set of 6 time series with 10 sampling points each.

    :rtype: Data instance
    :return: a Data instance for testing purposes.
    """
    #  Create time series
    ts = np.zeros((2, 10))

    for i in range(2):
        ts[i, :] = np.random.rand(10)

    ts[0, :].sort()

    return ts

#def testVisibility():
#    tdata = create_test_timeseries()
#    n_times = tdata.shape[1]
#    vg = VisibilityGraph(tdata[1], timings=tdata[0])
#    # Choose two different, not neighbouring random nodes i, j
#    node1, node2 = 0, 0
#    while (abs(node2-node1)<=1):
#        node1 = np.int(np.floor(np.random.rand()*n_times))
#        node2 = np.int(np.floor(np.random.rand()*n_times))
#    time, val = tdata
#    i, j = min(node1, node2), max(node1, node2)
#    testfun = lambda k: np.less((val[k]-val[i])/(time[k]-time[i]),
#                                (val[j]-val[i])/(time[j]-time[i]))
#    test = np.bool(np.sum(~np.array(map(testfun, range(i+1,j)))))
#    assert np.invert(test) == vg.visibility(node1, node2)

def testVisibilityHorizontal():
    tdata = create_test_timeseries()
    n_times = tdata.shape[1]
    vg = VisibilityGraph(tdata[1], timings=tdata[0])
    # Choose two different, not neighbouring random nodes i, j
    node1, node2 = 0, 0
    while abs(node2-node1) <= 0:
        node1 = np.int(np.floor(np.random.rand()*n_times))
        node2 = np.int(np.floor(np.random.rand()*n_times))

    val = tdata[1]
    i, j = min(node1, node2), max(node1, node2)
    if np.sum(~(val[i+1:j] < min(val[i], val[j]))):
        test = False
    else:
        test = True

    assert test == vg.visibility_horizontal(node1, node2)
