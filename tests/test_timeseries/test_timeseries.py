#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2023 Jonathan F. Donges and pyunicorn authors
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


def testCrossRecurrencePlot():
    tdata = create_test_data()
    CrossRecurrencePlot(x=tdata, y=tdata, threshold=0.2)


def testDistanceMatrix():
    tdata = create_test_data()
    crp = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    crp.distance_matrix(tdata.T, tdata.T, metric='manhattan')


def testManhattanDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    crp = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    manh_dist_1 = crp.manhattan_distance_matrix(tdata.T, tdata.T)
    manh_dist_2 = crp.manhattan_distance_matrix(tdata.T, tdata.T)
    assert np.allclose(manh_dist_1, manh_dist_2, atol=1e-04)


def testEuclideanDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    crp = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    eucl_dist_1 = crp.euclidean_distance_matrix(tdata.T, tdata.T)
    eucl_dist_2 = crp.euclidean_distance_matrix(tdata.T, tdata.T)
    assert np.allclose(eucl_dist_1, eucl_dist_2, atol=1e-04)


def testSupremumDistanceMatrix():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    crp = CrossRecurrencePlot(x=tdata, y=tdata, threshold=1.0)
    supr_dist_1 = crp.supremum_distance_matrix(tdata.T, tdata.T)
    supr_dist_2 = crp.supremum_distance_matrix(tdata.T, tdata.T)
    assert np.allclose(supr_dist_1, supr_dist_2, atol=1e-04)


# -----------------------------------------------------------------------------
# surrogates
# -----------------------------------------------------------------------------


def testNormalizeTimeSeriesArray():
    ts = Surrogates.SmallTestData().original_data
    Surrogates.SmallTestData().normalize_time_series_array(ts)
    res = ts.mean(axis=1)
    exp = np.array([0., 0., 0., 0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = ts.std(axis=1)
    exp = np.array([1., 1., 1., 1., 1., 1.])
    assert np.allclose(res, exp, atol=1e-04)


def testEmbedTimeSeriesArray():
    ts = Surrogates.SmallTestData().original_data
    res = Surrogates.SmallTestData().embed_time_series_array(
        time_series_array=ts, dimension=3, delay=2)[0, :6, :]
    exp = np.array([[0., 0.61464833, 1.14988147],
                    [0.31244015, 0.89680225, 1.3660254],
                    [0.61464833, 1.14988147, 1.53884177],
                    [0.89680225, 1.3660254, 1.6636525],
                    [1.14988147, 1.53884177, 1.73766672],
                    [1.3660254, 1.6636525, 1.76007351]])
    assert np.allclose(res, exp, atol=1e-04)


def testWhiteNoiseSurrogates():
    ts = Surrogates.SmallTestData().original_data
    surrogates = Surrogates.SmallTestData().white_noise_surrogates(ts)

    assert(np.allclose(np.histogram(ts[0, :])[0],
                       np.histogram(surrogates[0, :])[0]))


def testCorrelatedNoiseSurrogates():
    ts = Surrogates.SmallTestData().original_data
    surrogates = Surrogates.SmallTestData().correlated_noise_surrogates(ts)
    assert np.allclose(np.abs(np.fft.fft(ts, axis=1))[0, 1:10],
                       np.abs(np.fft.fft(surrogates, axis=1))[0, 1:10])


def testTwinSurrogates():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    s = Surrogates(tdata)
    tsurro = s.twin_surrogates(tdata, 1, 0, 0.2)
    corrcoef = np.corrcoef(tdata, tsurro)[n_index:, :n_index]
    for i in range(n_index):
        corrcoef[i, i] = 0.0
    assert (corrcoef >= -1.0).all() and (corrcoef <= 1.0).all()


def testPearsonCorrelation():
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    norm = 1.0 / float(n_times)
    c = Surrogates.test_pearson_correlation(tdata, tdata)
    corrcoef = np.corrcoef(tdata, tdata)[n_index:, :n_index]*norm
    for i in range(n_index):
        corrcoef[i, i] = 0.0

    assert c.shape == (n_index, n_index)
    assert_array_almost_equal(c, corrcoef, decimal=5)


def testMutualInformation():
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

# def testVisibility():
#     tdata = create_test_timeseries()
#     n_times = tdata.shape[1]
#     vg = VisibilityGraph(tdata[1], timings=tdata[0])
#     # Choose two different, not neighbouring random nodes i, j
#     node1, node2 = 0, 0
#     while (abs(node2-node1)<=1):
#         node1 = np.int(np.floor(np.random.rand()*n_times))
#         node2 = np.int(np.floor(np.random.rand()*n_times))
#     time, val = tdata
#     i, j = min(node1, node2), max(node1, node2)
#     testfun = lambda k: np.less((val[k]-val[i])/(time[k]-time[i]),
#                                 (val[j]-val[i])/(time[j]-time[i]))
#     test = np.bool(np.sum(~np.array(map(testfun, range(i+1,j)))))
#     assert np.invert(test) == vg.visibility(node1, node2)


def testVisibilityHorizontal():
    tdata = create_test_timeseries()
    n_times = tdata.shape[1]
    vg = VisibilityGraph(tdata[1], timings=tdata[0])
    # Choose two different, not neighbouring random nodes i, j
    node1, node2 = 0, 0
    while abs(node2-node1) <= 0:
        node1 = np.int32(np.floor(np.random.rand()*n_times))
        node2 = np.int32(np.floor(np.random.rand()*n_times))

    val = tdata[1]
    i, j = min(node1, node2), max(node1, node2)
    if np.sum(~(val[i+1:j] < min(val[i], val[j]))):
        test = False
    else:
        test = True

    assert test == vg.visibility_horizontal(node1, node2)
