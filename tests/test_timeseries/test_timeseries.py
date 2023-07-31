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

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyunicorn.timeseries import CrossRecurrencePlot, \
    InterSystemRecurrenceNetwork, Surrogates, VisibilityGraph
from pyunicorn.core.data import Data
from pyunicorn.core._ext.types import DFIELD


# turn off for weave compilation & error detection
parallel = False


def create_test_data():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:, None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:, None]
    return tdata


# -----------------------------------------------------------------------------
# cross_recurrence_plot
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("met", ["supremum", "euclidean", "manhattan"])
@pytest.mark.parametrize(
    "thresh, rr", [(.2, None)])  # add (None, .2) when #188 solved
def testCrossRecurrencePlot(thresh, rr, met):
    # create two instances of the same test dataset
    tdata1 = create_test_data()
    x1 = tdata1[:, 0]
    y1 = tdata1[:, 1]
    tdata2 = create_test_data()
    x2 = tdata2[:, 0]
    y2 = tdata2[:, 1]
    # create CrossRecurrencePlot for both
    crp1 = CrossRecurrencePlot(
            x1, y1, threshold=thresh, recurrence_rate=rr, metric=met)
    crp2 = CrossRecurrencePlot(
            x2, y2, threshold=thresh, recurrence_rate=rr, metric=met)
    # get respective distance matrices
    dist_1 = crp1.distance_matrix(crp1.x_embedded, crp1.y_embedded, met)
    dist_2 = crp2.distance_matrix(crp2.x_embedded, crp2.y_embedded, met)
    # get respective recurrence matrices
    CR1 = crp1.recurrence_matrix()
    CR2 = crp2.recurrence_matrix()

    assert np.allclose(dist_1, dist_2, atol=1e-04)
    assert CR1.shape == CR2.shape
    assert CR1.shape == (len(x1), len(y1))
    assert CR1.dtype == CR2.dtype
    assert CR1.dtype == np.int8


# -----------------------------------------------------------------------------
# inter_system_recurrence_network
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("met", ["supremum", "euclidean", "manhattan"])
@pytest.mark.parametrize(
    "thresh, rr", [((.2, .3, .2), None)]
    )  # add (None, (.2, .3, .2) when #188 solved
def testInterSystemRecurrenceNetwork(thresh, rr, met):
    # create two instances of the same test dataset
    tdata1 = create_test_data()
    x1 = tdata1[:, 0]
    y1 = tdata1[:, 1]
    tdata2 = create_test_data()
    x2 = tdata2[:, 0]
    y2 = tdata2[:, 1]
    # create InterSystemRecurrenceNetwork for both
    isrn1 = InterSystemRecurrenceNetwork(
            x1, y1, threshold=thresh, recurrence_rate=rr, metric=met)
    isrn2 = InterSystemRecurrenceNetwork(
            x2, y2, threshold=thresh, recurrence_rate=rr, metric=met)
    # get respective adjacency matrices
    A1 = isrn1.adjacency
    A2 = isrn2.adjacency

    assert np.array_equal(A1, A2)
    assert A1.shape == A2.shape
    assert A1.shape == (len(x1)*2, len(y1)*2)
    assert A1.dtype == A2.dtype
    assert A1.dtype == np.int16


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

    assert np.allclose(np.histogram(ts[0, :])[0],
                       np.histogram(surrogates[0, :])[0])


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
    Return test time series including random values and timings
    with 10 sampling points.

    :rtype: 2D array
    :return: a timeseries for testing purposes
    """
    #  Create time series
    ts = np.zeros((2, 10))

    for i in range(2):
        ts[i, :] = np.random.rand(10)

    ts[1, :].sort()

    return ts


# def testVisibility():
#     x, t = create_test_timeseries()
#     n_times = len(t)
#     vg = VisibilityGraph(x, timings=t)
#     # Choose two different, not neighbouring random nodes i, j
#     node1, node2 = 0, 0
#     while (abs(node2-node1)<=1):
#         node1 = np.int32(np.floor(np.random.rand()*n_times))
#         node2 = np.int32(np.floor(np.random.rand()*n_times))

#     # the following returns incorrect results, assertion therefore failing
#     i, j = min(node1, node2), max(node1, node2)
#     test = np.zeros((j-(i+1)))
#     for k in range(i+1,j):
#         test[k-(i+1)] = np.less((x[k]-x[i]) / (t[k]-t[i]),
#                                 (x[j]-x[i]) / (t[j]-t[i]))

#     assert np.invert(bool(np.sum(test))) == vg.visibility(node1, node2)


def testVisibility():
    x, t = create_test_timeseries()
    vg = VisibilityGraph(x, timings=t)
    A = vg.adjacency

    assert A.shape == (len(x), len(x))
    assert A.dtype == np.int16


def testVisibilityHorizontal():
    x, t = create_test_timeseries()
    vg = VisibilityGraph(x, timings=t, horizontal=True)
    A = vg.adjacency

    assert A.shape == (len(x), len(x))
    assert A.dtype == np.int16


def testRetardedLocalClustering():
    x, t = create_test_timeseries()
    vg = VisibilityGraph(x, timings=t)
    C_ret = vg.retarded_local_clustering()

    assert C_ret.shape == x.shape
    assert C_ret.dtype == DFIELD


def testAdvancedLocalClustering():
    x, t = create_test_timeseries()
    vg = VisibilityGraph(x, timings=t)
    C_adv = vg.advanced_local_clustering()

    assert C_adv.shape == x.shape
    assert C_adv.dtype == DFIELD
