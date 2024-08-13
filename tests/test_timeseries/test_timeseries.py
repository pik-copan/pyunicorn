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

"""
Simple tests for the timeseries class.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyunicorn.timeseries import RecurrencePlot, CrossRecurrencePlot, \
    RecurrenceNetwork, JointRecurrenceNetwork, InterSystemRecurrenceNetwork, \
    Surrogates, VisibilityGraph
from pyunicorn.core.data import Data
from pyunicorn.core._ext.types import DFIELD


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


@pytest.mark.parametrize("thresh, rr", [(.2, None), (None, .2)])
def testCrossRecurrencePlot(thresh, rr, metric: str):
    # create two instances of the same test dataset
    tdata1 = create_test_data()
    x1 = tdata1[:, 0]
    y1 = tdata1[:, 1]
    tdata2 = create_test_data()
    x2 = tdata2[:, 0]
    y2 = tdata2[:, 1]
    # create CrossRecurrencePlot for both
    crp1 = CrossRecurrencePlot(
            x1, y1, threshold=thresh, recurrence_rate=rr, metric=metric)
    crp2 = CrossRecurrencePlot(
            x2, y2, threshold=thresh, recurrence_rate=rr, metric=metric)
    # get respective distance matrices
    dist_1 = crp1.distance_matrix(metric)
    dist_2 = crp2.distance_matrix(metric)
    # get respective recurrence matrices
    CR1 = crp1.recurrence_matrix()
    CR2 = crp2.recurrence_matrix()

    assert np.allclose(dist_1, dist_2, atol=1e-04)
    assert CR1.shape == CR2.shape
    assert CR1.shape == (len(x1), len(y1))
    assert CR1.dtype == CR2.dtype
    assert CR1.dtype == np.int8


# -----------------------------------------------------------------------------
# recurrence_network
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("thresh, rr",
                         [(.2, None), (None, .2)], ids=str)
def testRecurrenceNetwork(thresh, rr, metric: str):
    # create two instances of the same test dataset
    tdata1 = create_test_data()
    tdata2 = create_test_data()
    # create RecurrenceNetwork for both
    rn1 = RecurrenceNetwork(
            tdata1, threshold=thresh, recurrence_rate=rr, metric=metric)
    rn2 = RecurrenceNetwork(
            tdata2, threshold=thresh, recurrence_rate=rr, metric=metric)
    # get respective adjacency matrices
    A1 = rn1.adjacency
    A2 = rn2.adjacency

    assert np.array_equal(A1, A2)
    assert A1.shape == A2.shape
    assert A1.shape == (len(tdata1), len(tdata1))
    assert A1.dtype == A2.dtype
    assert A1.dtype == np.int16


def testRecurrenceNetwork_setters():
    tdata = create_test_data()
    rn = RecurrenceNetwork(tdata, threshold=.2)
    # recalculate with different fixed threshold
    rn.set_fixed_threshold(.3)
    assert rn.adjacency.shape == (len(tdata), len(tdata))
    # recalculate with fixed threshold in units of the ts' std
    rn.set_fixed_threshold_std(.3)
    assert rn.adjacency.shape == (len(tdata), len(tdata))
    # recalculate with fixed recurrence rate
    rn.set_fixed_recurrence_rate(.2)
    assert rn.adjacency.shape == (len(tdata), len(tdata))
    # recalculate with fixed local recurrence rate
    rn.set_fixed_local_recurrence_rate(.2)
    assert rn.adjacency.shape == (len(tdata), len(tdata))


# -----------------------------------------------------------------------------
# joint_recurrence_network
# -----------------------------------------------------------------------------


def testJointRecurrenceNetwork(metric: str):
    tdata = create_test_data()
    x = tdata[:, 0]
    y = tdata[:, 1]
    n = len(tdata)
    jrp = JointRecurrenceNetwork(x, y, threshold=(.1, .1),
                                 metric=(metric, metric))
    dist = {}
    for i in "xy":
        jrp.embedding = getattr(jrp, f"{i}_embedded")
        dist[i] = jrp.distance_matrix(metric=metric)
    assert all(d.shape == (n, n) for d in dist.values())
    assert jrp.recurrence_matrix().shape == (n, n)


# -----------------------------------------------------------------------------
# inter_system_recurrence_network
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("thresh, rr",
                         [((.2, .3, .2), None), (None, (.2, .3, .2))], ids=str)
def testInterSystemRecurrenceNetwork(thresh, rr, metric: str):
    # create two instances of the same test dataset
    tdata1 = create_test_data()
    x1 = tdata1[:, 0]
    y1 = tdata1[:, 1]
    tdata2 = create_test_data()
    x2 = tdata2[:, 0]
    y2 = tdata2[:, 1]
    # create InterSystemRecurrenceNetwork for both
    isrn1 = InterSystemRecurrenceNetwork(
            x1, y1, threshold=thresh, recurrence_rate=rr, metric=metric)
    isrn2 = InterSystemRecurrenceNetwork(
            x2, y2, threshold=thresh, recurrence_rate=rr, metric=metric)
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


def testNormalizeOriginalData():
    ts = Surrogates.SmallTestData()
    ts.normalize_original_data()

    res = ts.original_data.mean(axis=1)
    exp = np.array([0., 0., 0., 0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = ts.original_data.std(axis=1)
    exp = np.array([1., 1., 1., 1., 1., 1.])
    assert np.allclose(res, exp, atol=1e-04)


def testEmbedTimeSeriesArray():
    ts = Surrogates.SmallTestData()
    res = Surrogates.embed_time_series_array(
        time_series_array=ts.original_data,
        dimension=3, delay=2)[0, :6, :]
    exp = np.array([[0., 0.61464833, 1.14988147],
                    [0.31244015, 0.89680225, 1.3660254],
                    [0.61464833, 1.14988147, 1.53884177],
                    [0.89680225, 1.3660254, 1.6636525],
                    [1.14988147, 1.53884177, 1.73766672],
                    [1.3660254, 1.6636525, 1.76007351]])
    assert np.allclose(res, exp, atol=1e-04)


def testSurrogatesRecurrencePlot():
    thresh = .2
    dim = 3
    tau = 2
    # calculate Surrogates.recurrence_plot()
    ts = Surrogates.SmallTestData()
    embedding = Surrogates.\
        embed_time_series_array(ts.original_data, dimension=dim, delay=tau)
    rp1 = Surrogates.recurrence_plot(embedding[0], threshold=thresh)
    # compare to timeseries.RecurrencePlot
    rp2 = RecurrencePlot(
        ts.original_data[0], threshold=thresh, dim=dim, tau=tau).R
    assert np.array_equal(rp1, rp2)


def testWhiteNoiseSurrogates():
    ts = Surrogates.SmallTestData()
    surrogates = ts.white_noise_surrogates()

    assert np.allclose(np.histogram(ts.original_data[0, :])[0],
                       np.histogram(surrogates[0, :])[0])


def testCorrelatedNoiseSurrogates():
    ts = Surrogates.SmallTestData()
    surrogates = ts.correlated_noise_surrogates()
    assert np.allclose(np.abs(np.fft.fft(ts.original_data, axis=1))[0, 1:10],
                       np.abs(np.fft.fft(surrogates, axis=1))[0, 1:10])


def testTwinSurrogates():
    tdata = create_test_data()
    ts = Surrogates(tdata)
    tsurro = ts.twin_surrogates(1, 0, 0.2)
    corrcoef = np.corrcoef(tdata, tsurro)[ts.N:, :ts.N]
    for i in range(ts.N):
        corrcoef[i, i] = 0.0
    assert (corrcoef >= -1.0).all() and (corrcoef <= 1.0).all()


def testAAFTSurrogates():
    ts = Surrogates.SmallTestData()
    # also covers Surrogates.AAFT_surrogates(), which is used as starting point
    surr_R, surr_s = ts.refined_AAFT_surrogates(n_iterations=3, output="both")
    # assert conserved amplitude distribution
    assert all(np.histogram(ts.original_data[0, :])[0] ==
               np.histogram(surr_R[0, :])[0])
    # assert conserved power spectrum
    assert np.allclose(np.abs(np.fft.fft(ts.original_data, axis=1))[0, 1:10],
                       np.abs(np.fft.fft(surr_s, axis=1))[0, 1:10])


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


def testOriginalDistribution():
    nbins = 10
    ts = Surrogates.SmallTestData()
    hist, lbb = ts.original_distribution(
        Surrogates.test_mutual_information, n_bins=nbins)

    assert np.isclose(hist.sum(), 1) and len(lbb) == nbins


def testThresholdSignificance():
    nbins = 10
    ts = Surrogates.SmallTestData()
    density_estimate, lbb = ts.test_threshold_significance(
        Surrogates.white_noise_surrogates,
        Surrogates.test_mutual_information,
        realizations=5,
        interval=[0, 2],
        n_bins=nbins)

    assert np.isclose(density_estimate.sum(), 1) and len(lbb) == nbins


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
