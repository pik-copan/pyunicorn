# This file is part of pyunicorn.
# Copyright (C) 2008--2026 Jonathan F. Donges and pyunicorn authors
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
Tests for the EventSeries class.
"""
import numpy as np
import pytest
from pyunicorn.eventseries import EventSeries


def create_test_data():
    N = 1000
    tstamps = np.linspace(0.0, (N-1)*0.1, N)
    data = np.zeros((N, 2))
    for i in range(N):
        data[i] = np.array([np.sin(tstamps[i]*np.pi),
                            np.sin((tstamps[i]-0.2)*np.pi)])
    return data, tstamps


def test_EventSeries_init():
    data = np.random.randint(2, size=(1000, 3))
    test_object1 = EventSeries(data)

    assert np.all(test_object1.get_event_matrix() == data)


def test_make_event_matrix():
    data = create_test_data()[0]

    # Test 'value' method for first variable > 0.99 and second variable < 0.0
    thresholds1 = np.array([0.99, 0.0])
    test_object1 = EventSeries(data, threshold_method='value',
                               threshold_values=thresholds1,
                               threshold_types=np.array(['above', 'below']))
    eventmatrix = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        if data[i][0] > thresholds1[0]:
            eventmatrix[i][0] = 1
        else:
            eventmatrix[i][0] = 0

        if data[i][1] < thresholds1[1]:
            eventmatrix[i][1] = 1
        else:
            eventmatrix[i][1] = 0

    assert np.all(test_object1.get_event_matrix() == eventmatrix)

    # Test 'quantile' method for first variable < 0.3 quantile and
    # second variable > 0.8 quantile
    threshold_values2 = np.array([0.3, 0.8])
    test_object2 = EventSeries(data, threshold_method='quantile',
                               threshold_values=threshold_values2,
                               threshold_types=['below', 'above'])
    eventmatrix = np.zeros((data.shape[0], data.shape[1]))
    thresholds2 = [np.quantile(data[:, 0], threshold_values2[0]),
                   np.quantile(data[:, 1], threshold_values2[1])]
    for i in range(data.shape[0]):
        if data[i][0] < thresholds2[0]:
            eventmatrix[i][0] = 1
        else:
            eventmatrix[i][0] = 0

        if data[i][1] > thresholds2[1]:
            eventmatrix[i][1] = 1
        else:
            eventmatrix[i][1] = 0

    assert np.all(test_object2.get_event_matrix() == eventmatrix)


def eca_second_implementaion(eventseriesx, eventseriesy, *, ts1=None,
                             ts2=None, deltaT=3, lag=0.0):
    """
    Test implementation of event coincidence analysis

    :type eventseriesx: 1D numpy array
    :arg eventseriesx: Event series containing '0' and '1'
    :type eventseriesy: 1D numpy array
    :arg eventseriesy: Event series containing '0' and '1'
    :type ts1: 1D numpy array
    :arg ts1: time stamps of events in eventseriesx, not obligatory
    :type ts2: 1D numpy array
    :arg ts2: time stamps of events in eventseriesy, not obligatory
    :type deltaT: float
    :arg deltaT: time window defining if two events are considered precursor
                 or trigger of the other
    :type lag: float
    :arg lag: delay between the two time series, eventseriesy is shifted
    :rtype: list
    :return:
    """

    # Get time indices
    if ts1 is None:
        e1 = np.where(eventseriesx)[0]
    else:
        e1 = ts1[eventseriesx == 1]
    if ts2 is None:
        e2 = np.where(eventseriesy)[0]
    else:
        e2 = ts2[eventseriesy == 1]

    # Number of events
    l1 = len(e1)
    l2 = len(e2)

    # Find number of events which cannot be trigger or precursor events
    s1p = len(e1[e1 <= (e1[0] + lag + deltaT)])
    s2t = len(e2[e2 >= (e2[-1] - lag - deltaT)])
    s1t = len(e1[e1 >= (e1[-1] - lag - deltaT)])
    s2p = len(e2[e2 <= (e2[0] + lag + deltaT)])

    precursorxy = 0.0
    precursoryx = 0.0
    triggerxy = 0.0
    triggeryx = 0.0

    for i1 in range(s1p, l1):
        precursordummy = False
        for j2 in range(l2):
            if e1[i1] - lag - e2[j2] >= 0.0 and \
                    e1[i1] - lag - e2[j2] <= deltaT:
                precursordummy = True
        if precursordummy:
            precursorxy = precursorxy + 1.0

    for j2 in range(l2-s2t):
        triggerdummy = False
        for i1 in range(l1):
            if e1[i1] - lag - e2[j2] >= 0.0 and \
                    e1[i1] - lag - e2[j2] <= deltaT:
                triggerdummy = True
        if triggerdummy:
            triggerxy = triggerxy + 1.0

    for i2 in range(s2p, l2):
        precursordummy = False
        for j1 in range(l1):
            if e2[i2] - lag - e1[j1] >= 0.0 and \
                    e2[i2] - lag - e1[j1] <= deltaT:
                precursordummy = True
        if precursordummy:
            precursoryx = precursoryx + 1.0

    for j1 in range(l1-s1t):
        triggerdummy = False
        for i2 in range(l2):
            if e2[i2] - lag - e1[j1] >= 0.0 and \
                    e2[i2] - lag - e1[j1] <= deltaT:
                triggerdummy = True
        if triggerdummy:
            triggeryx = triggeryx + 1.0

    return (precursorxy/(l1-s1p), triggerxy/(l2-s2t),
            precursoryx/(l2-s2p), triggeryx/(l1-s1t))


def test_eca():
    es1 = np.random.randint(2, size=100)
    es2 = np.random.randint(2, size=100)
    for tau in [0.0, 1.5]:
        for window in [2, 5]:
            for ts1, ts2 in [[None, None],
                             [np.random.uniform(size=len(es1)).cumsum(),
                              np.random.uniform(size=len(es1)).cumsum()]]:
                eca_orig = \
                    EventSeries.event_coincidence_analysis(es1, es2, window,
                                                           ts1=ts1,
                                                           ts2=ts2,
                                                           lag=tau)
                eca_check = \
                    eca_second_implementaion(es1, es2, ts1=ts1, ts2=ts2,
                                             deltaT=window, lag=tau)
                assert np.allclose(np.array(eca_orig), np.array(eca_check),
                                   atol=1e-04)


def eca_implementation_for_symmetric_window(es1, es2, *, ts1=None, ts2=None,
                                            taumax=3.0, lag=0.0):
    # Get time indices
    if ts1 is None:
        e1 = np.where(es1)[0]
    else:
        e1 = ts1[es1 == 1]
    if ts2 is None:
        e2 = np.where(es2)[0]
    else:
        e2 = ts2[es2 == 1]

    # Number of events
    l1 = len(e1)
    l2 = len(e2)

    # Find number of events which cannot be trigger or precursor events
    if not (lag == 0 and taumax == 0):
        s1start = len(e1[e1 <= (e1[0] + lag + taumax)])
        s1end = len(e1[e1 >= (e1[-1] - lag - taumax)])
        s2end = len(e2[e2 >= (e2[-1] - lag - taumax)])
        s2start = len(e2[e2 <= (e2[0] + lag + taumax)])
    else:
        s1start, s1end, s2start, s2end = 0, 0, 0, 0

    coincidence12, coincidence21 = 0.0, 0.0

    for i in range(s1start, l1 - s1end):
        for j in range(l2):
            if np.abs(e1[i] - lag - e2[j]) <= taumax:
                coincidence12 = coincidence12 + 1.0
                break

    for j in range(s2start, l2 - s2end):
        for i in range(l1):
            if np.abs(e2[j] - lag - e1[i]) <= taumax:
                coincidence21 = coincidence21 + 1.0
                break

    return (coincidence12 / (l1 - s1start - s1end),
            coincidence21 / (l2 - s2start - s2end))


def es_second_implementation(es1, es2, *, ts1=None, ts2=None, taumax=np.inf,
                             lag=0.0):
    """
    Calculates the directed event synchronization from two event series X
    and Y.
    :type es1: 1D Numpy array
    :arg es1: Event series containing '0's and '1's
    :type es2: 1D Numpy array
    :arg es2: Event series containing '0's and '1's
    :type ts1: 1D Numpy array
    :arg ts1: Event time array containing time points when events of event
               series 1 occur, not obligatory
    :type ts2: 1D Numpy array
    :arg ts2: Event time array containing time points when events of event
               series 2 occur, not obligatory
    :type taumax: float
    :arg taumax: maximum distance of two events to be counted as
                 synchronous
    :type lag: float
    :arg lag: delay between the two event series, the second event series
                  is shifted by the value of lag
    :rtype: list
    :return: [Event synchronization XY, Event synchronization YX]
    """

    # Get time indices
    if ts1 is None:
        ex = np.where(es1)[0]
    else:
        ex = ts1[es1 == 1]
    if ts2 is None:
        ey = np.where(es2)[0]
    else:
        ey = ts2[es2 == 1]

    lx = len(ex)
    ly = len(ey)

    ey = ey + lag

    Axy = np.zeros((lx, ly), dtype=bool)
    Ayx = np.zeros((lx, ly), dtype=bool)
    eqtime = np.zeros((lx, ly), dtype=bool)
    for m in range(1, lx - 1):
        for n in range(1, ly - 1):
            dstxy = ex[m] - ey[n]

            if abs(dstxy) > taumax:
                continue
            # finding the dynamical delay lag
            tau = min([ex[m + 1] - ex[m], ex[m] - ex[m - 1],
                       ey[n + 1] - ey[n], ey[n] - ey[n - 1]]) / 2

            if dstxy > 0:
                if dstxy <= tau:
                    Axy[m, n] = True
            if dstxy < 0:
                if dstxy >= -tau:
                    Ayx[m, n] = True
            elif dstxy == 0:
                eqtime[m, n] = True

    # Loop over coincidences and determine number of double counts
    # by checking at least one event of the pair is also coincided
    # in other direction
    countxydouble = countyxdouble = 0

    for i, j in np.transpose(np.where(Axy)):
        countxydouble += np.any(Ayx[i, :]) or np.any(Ayx[:, j])
    for i, j in np.transpose(np.where(Ayx)):
        countyxdouble += np.any(Axy[i, :]) or np.any(Axy[:, j])

    countxy = np.sum(Axy) + 0.5 * np.sum(eqtime) - 0.5 * countxydouble
    countyx = np.sum(Ayx) + 0.5 * np.sum(eqtime) - 0.5 * countyxdouble
    norm = np.sqrt((lx - 2) * (ly - 2))
    return countxy / norm, countyx / norm


def test_es():
    es1 = np.random.binomial(1, 0.5, 100)
    es2 = np.random.binomial(1, 0.5, 100)

    for tau in [0.0, 1.5]:
        for taumax in [np.inf, 5]:
            for ts1, ts2 in [[None, None],
                             [np.random.uniform(size=len(es1)).cumsum(),
                              np.random.uniform(size=len(es1)).cumsum()]]:
                es_orig = \
                    EventSeries.event_synchronization(es1, es2, ts1=ts1,
                                                      ts2=ts2, taumax=taumax,
                                                      lag=tau)
                es_check = \
                    es_second_implementation(es1, es2, ts1=ts1, ts2=ts2,
                                             taumax=taumax, lag=tau)

                assert np.allclose(np.array(es_orig), np.array(es_check),
                                   atol=1e-04)


def eca_matrix_second_implementation(eventmatrix, taumax,
                                     eca_type):

    if eca_type in ['advanced', 'retarded']:
        N = eventmatrix.shape[1]
        res_p, res_t = np.zeros((N, N)), np.zeros((N, N))

        for i in np.arange(0, N):
            for j in np.arange(i + 1, N):
                res_p[i, j], res_t[i, j], res_p[j, i], res_t[j, i] = \
                    eca_second_implementaion(eventmatrix[:, i],
                                             eventmatrix[:, j],
                                             deltaT=taumax)

        if eca_type == 'advanced':
            return res_p

        elif eca_type == 'retarded':
            return res_t

        else:
            return None

    elif eca_type == 'symmetric':
        N = eventmatrix.shape[1]
        res_s = np.zeros((N, N))
        for i in np.arange(0, N):
            for j in np.arange(i + 1, N):
                res_s[i, j], res_s[j, i] = \
                    eca_implementation_for_symmetric_window(eventmatrix[:, i],
                                                            eventmatrix[:, j],
                                                            taumax=taumax)
        return res_s

    return None


def test_vectorized_eca():
    # Create an event matrix of length 1000 with 3 variables
    test_event_matrix = np.random.randint(2, size=(100, 3))

    for eca_type in ['advanced', 'retarded', 'symmetric']:
        for taumax in [1, 5, 16]:
            eso = EventSeries(test_event_matrix, taumax=taumax)
            directedeca_matrix = \
                eca_matrix_second_implementation(test_event_matrix,
                                                 taumax, eca_type)
            # directed, no symmetry
            assert \
                np.allclose(directedeca_matrix,
                            eso.event_series_analysis(
                                method='ECA', symmetrization='directed',
                                window_type=eca_type),
                            atol=1e-04)
            # Test for the various symmetry options
            # mean
            assert \
                np.allclose(np.mean([directedeca_matrix, directedeca_matrix.T],
                                    axis=0),
                            eso.event_series_analysis(method='ECA',
                                                      symmetrization='mean',
                                                      window_type=eca_type),
                            atol=1e-04)
            # max
            assert \
                np.allclose(np.max([directedeca_matrix, directedeca_matrix.T],
                                   axis=0),
                            eso.event_series_analysis(method='ECA',
                                                      symmetrization='max',
                                                      window_type=eca_type),
                            atol=1e-04)

            # min
            assert \
                np.allclose(np.min([directedeca_matrix, directedeca_matrix.T],
                                   axis=0),
                            eso.event_series_analysis(method='ECA',
                                                      symmetrization='min',
                                                      window_type=eca_type),
                            atol=1e-04, equal_nan=True)


def es_matrix_second_implementation(eventmatrix, taumax):
    N = eventmatrix.shape[1]
    res = np.zeros((N, N))

    for i in np.arange(0, N):
        for j in np.arange(i + 1, N):
            res[i, j], res[j, i] = \
                es_second_implementation(eventmatrix[:, i],
                                         eventmatrix[:, j], taumax=taumax)
    return res


def test_vectorized_es():
    # Test if the vectorized implementation coincides with the straight forward
    # one.

    for taumax in [1, 5, 16, np.inf]:
        length, N, eventprop = 1000, 2, 0.2
        # equal event counts (normalization requirement)
        eventcount = int(length * eventprop)
        eventmatrix = np.zeros((length, N), dtype=int)
        for v in range(N):
            fills = np.random.choice(np.arange(length), eventcount,
                                     replace=False)
            eventmatrix[fills, v] = 1

        esob = EventSeries(eventmatrix, taumax=taumax)
        directedes_matrix = es_matrix_second_implementation(eventmatrix,
                                                            taumax)
        # directed, no symmetry
        assert \
            np.allclose(directedes_matrix,
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='directed'),
                        atol=1e-04)

        # Test for all symmetry options
        # symmetric
        assert \
            np.allclose(directedes_matrix + directedes_matrix.T,
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='symmetric'),
                        atol=1e-04)

        # antisymmetric
        assert \
            np.allclose(directedes_matrix - directedes_matrix.T,
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='antisym'),
                        atol=1e-04, equal_nan=True)

        # mean
        assert \
            np.allclose(np.mean([directedes_matrix, directedes_matrix.T],
                                axis=0),
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='mean'),
                        atol=1e-04)

        # max
        assert \
            np.allclose(np.maximum(directedes_matrix, directedes_matrix.T),
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='max'),
                        atol=1e-04)

        # min
        assert \
            np.allclose(np.minimum(directedes_matrix, directedes_matrix.T),
                        esob.event_series_analysis(method='ES',
                                                   symmetrization='min'),
                        atol=1e-04)


def test_significance():
    n_surrs = 100
    data, tstamps = create_test_data()

    # Testing without symmetry
    esob = EventSeries(data, timestamps=tstamps, threshold_method='value',
                       threshold_values=np.array([0.98, 0.98]),
                       threshold_types=np.array(['above', 'above']),
                       taumax=np.inf, lag=0.0)

    # Test Event Synchronization significance via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ES',
                                                     n_surr=n_surrs),
                    np.array([[0., 0.], [1., 0.]]), atol=1e-04)
    # Test ECA
    esob = EventSeries(data, timestamps=tstamps, threshold_method='value',
                       threshold_values=np.array([0.98, 0.98]),
                       threshold_types=np.array(['above', 'above']),
                       taumax=1.0, lag=0.0)

    # Test precursor coincidence via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='shuffle',
                                                     n_surr=n_surrs,
                                                     symmetrization='directed',
                                                     window_type='advanced'),
                    np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-04)

    # Test trigger coincidence via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='shuffle',
                                                     n_surr=n_surrs,
                                                     symmetrization='directed',
                                                     window_type='retarded'),
                    np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-04)
    # Test trigger coincidence via binomial distribution
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='analytic',
                                                     symmetrization='directed',
                                                     window_type='retarded'),
                    np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-04)

    # Test precursor coincidence via binomial distribution
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='analytic',
                                                     symmetrization='directed',
                                                     window_type='advanced'),
                    np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-04)

    # Testing for symmetrization=mean
    # Testing without symmetry
    esob = EventSeries(data, timestamps=tstamps, threshold_method='value',
                       threshold_values=np.array([0.98, 0.98]),
                       threshold_types=np.array(['above', 'above']),
                       taumax=np.inf, lag=0.0)

    # Test Event Synchronization significance via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ES',
                                                     n_surr=n_surrs,
                                                     symmetrization='mean'),
                    np.array([[0., 1.0], [1.0, 0.]]), atol=1e-04)
    # Test ECA
    esob = EventSeries(data, timestamps=tstamps, threshold_method='value',
                       threshold_values=np.array([0.98, 0.98]),
                       threshold_types=np.array(['above', 'above']),
                       taumax=1.0, lag=0.0)

    # Test precursor coincidence via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='shuffle',
                                                     n_surr=n_surrs,
                                                     symmetrization='mean',
                                                     window_type='advanced'),
                    np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-04)

    # Test trigger coincidence via surrogates
    assert \
        np.allclose(esob.event_analysis_significance(method='ECA',
                                                     surrogate='shuffle',
                                                     n_surr=n_surrs,
                                                     symmetrization='mean',
                                                     window_type='retarded'),
                    np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-04)


# ==========================================================================
# Additional tests covering the optimized kernels, new getters, fallback
# paths, optional joblib parallelization, NaN handling, and input validation.
# These tests are intentionally small and deterministic (fixed RNG seeds)
# so they remain stable and fast in CI.
# ==========================================================================


def _make_binary_matrix(T, N, p=0.2, seed=0):
    """Utility: reproducible binary event matrix with roughly p-fraction
    events per variable."""
    rng = np.random.default_rng(seed)
    return (rng.random((T, N)) < p).astype(int)


def test_getters():
    """Cover the public getter methods and the __str__ representation."""
    data = _make_binary_matrix(120, 4, seed=1)
    ts = np.linspace(0.0, 11.9, 120)
    es = EventSeries(data, timestamps=ts, taumax=2.5, lag=0.5)

    assert es.get_T() == 120
    assert es.get_N() == 4
    assert np.allclose(es.get_timestamps(), ts)
    assert es.get_taumax() == 2.5
    assert es.get_lag() == 0.5
    # __str__ is covered via repr() to hit that branch too
    s = str(es)
    assert "EventSeries" in s and "4 variables" in s and "120 timesteps" in s


def test_es_sparse_with_nonzero_lag():
    """Finite taumax with nonzero lag routes through the sparse
    binary-search kernel (`_compute_es_matrix_sparse_lag`).
    Verify matrix-level result matches the pairwise reference."""
    eventmatrix = _make_binary_matrix(200, 3, p=0.2, seed=2)
    taumax = 4
    lag = 1.5

    esob = EventSeries(eventmatrix, taumax=taumax, lag=lag)
    fast = esob.event_series_analysis(method='ES',
                                      symmetrization='directed')

    # Reference: pairwise public API per (i, j) pair
    N = eventmatrix.shape[1]
    ref = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ref[i, j], ref[j, i] = \
                EventSeries.event_synchronization(eventmatrix[:, i],
                                                  eventmatrix[:, j],
                                                  taumax=taumax, lag=lag)
    assert np.allclose(fast, ref, atol=1e-04)


def test_es_pairwise_fallback():
    """taumax=inf combined with lag != 0 should route through
    `_ndim_event_synchronization_pairwise`. We assert agreement with
    the pairwise public API on the same inputs."""
    eventmatrix = _make_binary_matrix(200, 3, p=0.2, seed=3)
    lag = 0.75
    esob = EventSeries(eventmatrix, taumax=np.inf, lag=lag)
    fast = esob.event_series_analysis(method='ES',
                                      symmetrization='directed')

    N = eventmatrix.shape[1]
    ref = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ref[i, j], ref[j, i] = \
                EventSeries.event_synchronization(eventmatrix[:, i],
                                                  eventmatrix[:, j],
                                                  taumax=np.inf, lag=lag)
    assert np.allclose(fast, ref, atol=1e-04)


def test_eca_vectorized_nonzero_lag():
    """Integer lag != 0 with uniform integer timestamps triggers the
    vectorized cumsum + BLAS kernel. Compare against the pairwise
    reference implementation for each of the three window_types."""
    eventmatrix = _make_binary_matrix(120, 3, p=0.25, seed=4)
    taumax = 3
    lag = 2  # integer so the vectorized path is used

    for window_type in ['advanced', 'retarded', 'symmetric']:
        esob = EventSeries(eventmatrix, taumax=taumax, lag=lag)
        fast = esob.event_series_analysis(method='ECA',
                                          symmetrization='directed',
                                          window_type=window_type)

        N = eventmatrix.shape[1]
        ref = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if window_type == 'symmetric':
                    ref[i, j], ref[j, i] = \
                        eca_implementation_for_symmetric_window(
                            eventmatrix[:, i], eventmatrix[:, j],
                            taumax=taumax, lag=lag)
                else:
                    p, t, p2, t2 = \
                        eca_second_implementaion(eventmatrix[:, i],
                                                 eventmatrix[:, j],
                                                 deltaT=taumax, lag=lag)
                    if window_type == 'advanced':
                        ref[i, j], ref[j, i] = p, p2
                    else:  # 'retarded'
                        ref[i, j], ref[j, i] = t, t2
        assert np.allclose(fast, ref, atol=1e-04), \
            f"vectorized ECA mismatch for window_type={window_type}"


def test_eca_pairwise_symmetric_nonuniform():
    """Non-uniform timestamps force the pairwise ECA fallback
    (`_compute_eca_pairwise`). Verify against the reference symmetric
    implementation."""
    eventmatrix = _make_binary_matrix(60, 3, p=0.3, seed=5)
    rng = np.random.default_rng(6)
    ts = np.sort(rng.uniform(0.0, 60.0, size=60))  # non-uniform spacing

    taumax = 2.5
    esob = EventSeries(eventmatrix, timestamps=ts, taumax=taumax, lag=0.0)
    fast = esob.event_series_analysis(method='ECA',
                                      symmetrization='directed',
                                      window_type='symmetric')

    N = eventmatrix.shape[1]
    ref = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ref[i, j], ref[j, i] = \
                eca_implementation_for_symmetric_window(eventmatrix[:, i],
                                                        eventmatrix[:, j],
                                                        ts1=ts, ts2=ts,
                                                        taumax=taumax,
                                                        lag=0.0)
    assert np.allclose(fast, ref, atol=1e-04)


def test_joblib_parallel_equivalence():
    """If joblib is available, n_jobs>1 must produce bit-for-bit identical
    matrices to n_jobs=1 for both the blocked-dense and sparse kernels."""
    # pylint: disable=protected-access
    pytest.importorskip("joblib")
    eventmatrix = _make_binary_matrix(150, 4, p=0.2, seed=7)

    # Blocked dense kernel path (taumax=inf, lag=0)
    es_block = EventSeries(eventmatrix, taumax=np.inf, lag=0.0)
    node_data = es_block._precompute_es_node_data()
    serial = es_block._compute_es_matrix_blocked(node_data, n_jobs=1)
    parallel = es_block._compute_es_matrix_blocked(node_data, n_jobs=2)
    assert np.allclose(serial, parallel, atol=1e-10)

    # Sparse kernel path (finite taumax)
    es_sparse = EventSeries(eventmatrix, taumax=5, lag=1.0)
    node_data_s = es_sparse._precompute_es_node_data()
    serial_s = es_sparse._compute_es_matrix_sparse_lag(node_data_s, n_jobs=1)
    parallel_s = es_sparse._compute_es_matrix_sparse_lag(node_data_s,
                                                         n_jobs=2)
    assert np.allclose(serial_s, parallel_s, atol=1e-10)


def test_nan_handling_quantile():
    """Modification 2: make_event_matrix must use np.nanquantile so NaNs
    in continuous input don't propagate to the thresholds."""
    rng = np.random.default_rng(8)
    data = rng.normal(size=(200, 2))
    # Inject NaNs into both variables
    data[5, 0] = np.nan
    data[17, 1] = np.nan
    data[42, 0] = np.nan

    es = EventSeries(data,
                     threshold_method='quantile',
                     threshold_values=np.array([0.9, 0.1]),
                     threshold_types=np.array(['above', 'below']))
    evmat = es.get_event_matrix()
    # Thresholds must be finite even though the raw data contains NaNs
    # (i.e. at least one event should have been detected per variable)
    assert evmat.sum() > 0
    # NaN input positions must not themselves be marked as events
    assert evmat[5, 0] == 0
    assert evmat[17, 1] == 0
    assert evmat[42, 0] == 0


def test_edge_cases_few_events():
    """Nodes with 0, 1, 2 events are handled without crashing: they
    should produce zero scores (no inner events → _precompute_es_node_data
    returns empty arrays)."""
    T, N = 50, 4
    eventmatrix = np.zeros((T, N), dtype=int)
    # Column 0: 0 events. Column 1: 1 event. Column 2: 2 events.
    # Column 3: several events.
    eventmatrix[10, 1] = 1
    eventmatrix[[5, 30], 2] = 1
    eventmatrix[[3, 11, 19, 27, 35], 3] = 1

    # Finite taumax → sparse kernel must accept empty/tiny event arrays
    esob = EventSeries(eventmatrix, taumax=3, lag=0.0)
    out = esob.event_series_analysis(method='ES', symmetrization='directed')
    assert out.shape == (N, N)
    assert np.all(np.isfinite(out))
    # Zero events vs anything is 0
    assert np.all(out[0, :] == 0) and np.all(out[:, 0] == 0)
    # 1-event node also produces zeros (no inner events)
    assert np.all(out[1, :] == 0) and np.all(out[:, 1] == 0)

    # Blocked-dense kernel path (taumax=inf, lag=0) exercises the
    # empty-inner-events branch of _compute_es_block
    esob_inf = EventSeries(eventmatrix, taumax=np.inf, lag=0.0)
    out_inf = esob_inf.event_series_analysis(method='ES',
                                             symmetrization='directed')
    assert out_inf.shape == (N, N) and np.all(np.isfinite(out_inf))


def test_invalid_inputs():
    """Exercise the IOError branches in the constructor, make_event_matrix
    and the public analysis/significance methods."""
    # Non-binary data without threshold_method
    with pytest.raises(IOError):
        EventSeries(np.array([[0, 2], [1, 3]]))

    # Non-ndarray data with threshold_method set
    with pytest.raises(IOError):
        EventSeries([[0.1, 0.5], [0.3, 0.7]], threshold_method='quantile',
                    threshold_values=np.array([0.5, 0.5]),
                    threshold_types=np.array(['above', 'above']))

    # Mismatched timestamps length
    with pytest.raises(IOError):
        EventSeries(np.random.randint(2, size=(20, 2)),
                    timestamps=np.arange(10))

    # Invalid threshold_method
    rng = np.random.default_rng(9)
    cont = rng.normal(size=(40, 2))
    with pytest.raises(IOError):
        EventSeries(cont, threshold_method='nonsense',
                    threshold_values=np.array([0.5, 0.5]),
                    threshold_types=np.array(['above', 'above']))

    # Invalid threshold_types
    with pytest.raises(IOError):
        EventSeries(cont, threshold_method='quantile',
                    threshold_values=np.array([0.5, 0.5]),
                    threshold_types=np.array(['above', 'sideways']))

    # Valid event matrix — now exercise analysis-level validations
    evmat = np.random.randint(2, size=(40, 2))
    es = EventSeries(evmat, taumax=2.0)
    with pytest.raises(IOError):
        es.event_series_analysis(method='XYZ')
    with pytest.raises(IOError):
        es.event_series_analysis(method='ES', symmetrization='bogus')
    with pytest.raises(IOError):
        es.event_series_analysis(method='ECA', symmetrization='antisym',
                                 window_type='symmetric')
    with pytest.raises(IOError):
        es.event_series_analysis(method='ECA', symmetrization='directed',
                                 window_type='diagonal')

    # ECA with taumax=inf must raise ValueError
    es_inf = EventSeries(evmat, taumax=np.inf)
    with pytest.raises(ValueError):
        es_inf.event_series_analysis(method='ECA', symmetrization='directed',
                                     window_type='symmetric')

    # Significance-path validations
    with pytest.raises(IOError):
        es.event_analysis_significance(method='XYZ')
    with pytest.raises(IOError):
        es.event_analysis_significance(method='ECA', surrogate='nonsense')
    with pytest.raises(IOError):
        es.event_analysis_significance(method='ES', surrogate='analytic')
    with pytest.raises(IOError):
        es.event_analysis_significance(method='ECA', surrogate='analytic',
                                       symmetrization='mean',
                                       window_type='advanced')
    with pytest.raises(IOError):
        es.event_analysis_significance(method='ECA', surrogate='analytic',
                                       symmetrization='directed',
                                       window_type='symmetric')
