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
Simple tests for the RecurrencePlot class.
"""

from itertools import chain, product

import pytest
import numpy as np

from pyunicorn.core import Data
from pyunicorn.core._ext.types import NODE, ADJ, DFIELD
from pyunicorn.timeseries import RecurrencePlot


# -----------------------------------------------------------------------------
# test RecurrencePlot instantiation
# -----------------------------------------------------------------------------

# test non-default metrics

def test_RP_euclidean():
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x[:, :-3], threshold=1.2, metric='euclidean')
    res = RP.recurrence_matrix()
    assert res.dtype == ADJ
    exp = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    ])
    assert np.array_equal(res, exp)


def test_RP_manhattan():
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x, threshold=3.5, metric='manhattan')
    res = RP.recurrence_matrix()
    assert res.dtype == ADJ
    exp = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    assert np.array_equal(res, exp)


# test thresholding variations

def test_RP_threshold_std():
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x, threshold_std=.8)
    res = RP.recurrence_matrix()
    assert res.dtype == ADJ
    exp = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    assert np.array_equal(res, exp)


def test_RP_recurrence_rate():
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x, recurrence_rate=.4)
    res = RP.recurrence_matrix()
    assert res.dtype == ADJ
    exp = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    assert np.array_equal(res, exp)


def test_RP_local_recurrence_rate():
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x, local_recurrence_rate=.6)
    res = RP.recurrence_matrix()
    assert res.dtype == ADJ
    exp = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ])
    assert np.array_equal(res, exp)


# -----------------------------------------------------------------------------
# prepare fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module", name="recurrence_crit", ids=str,
                params=list(chain(product(np.arange(0, 1.7, .8), [None]),
                                  product([None], np.arange(0, 1.1, .4)))))
def recurrence_crit_fixture(request):
    threshold, rate = request.param
    assert np.sum([threshold is None, rate is None]) == 1
    return request.param


@pytest.fixture(scope="module", name="small_RP")
def small_RP_fixture(metric, recurrence_crit):
    """
    RP fixture, parametrized to cover various settings.
    """
    x = Data.SmallTestData().observable()
    threshold, rate = recurrence_crit
    return RecurrencePlot(
        x, threshold=threshold, recurrence_rate=rate, metric=metric)


@pytest.fixture(scope="module", name="small_RP_basic", params=[False, True])
def small_RP_basic_fixture(request):
    """
    RP fixture with single basic setting to test numerical results.
    """
    sparse = request.param
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x, threshold=.8, metric='supremum', sparse_rqa=sparse)
    if not sparse:
        res = RP.recurrence_matrix()
        assert res.dtype == ADJ
        exp = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        ])
        assert np.array_equal(res, exp)
    return RP


# -----------------------------------------------------------------------------
# test RecurrencePlot RQA
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("measure", ["diag", "vert", "white_vert"])
def test_line_dist(measure: str, small_RP):
    res = getattr(small_RP, f"{measure}line_dist")()
    assert res.dtype == NODE
    assert res.shape[0] == small_RP.N
    assert (0 <= res).all() and (res <= small_RP.N).all()


@pytest.mark.parametrize(
    "measure, exp",
    [("diag", [4, 0, 0, 0, 0, 0, 0, 2, 2, 0]),
     ("vert", [0, 0, 1, 2, 5, 2, 0, 0, 0, 0]),
     ("white_vert", [2, 1, 2, 2, 3, 2, 1, 0, 0, 0])])
def test_line_dist_numeric(measure: str, small_RP_basic, exp):
    if small_RP_basic.sparse_rqa and measure == "white_vert":
        return
    res = getattr(small_RP_basic, f"{measure}line_dist")()
    assert res.dtype == NODE
    assert res.shape[0] == small_RP_basic.N
    assert np.array_equal(res, exp)


@pytest.mark.parametrize("sparse", [False, True])
def test_line_dist_edgecases(sparse):
    x = Data.SmallTestData().observable()

    RP = RecurrencePlot(x, metric="supremum", threshold=0., sparse_rqa=sparse)
    assert RP.max_diaglength() == 0
    assert RP.max_vertlength() == 0
    if not sparse:
        assert RP.max_white_vertlength() == RP.N

    RP = RecurrencePlot(x, metric="supremum", threshold=2., sparse_rqa=sparse)
    assert RP.max_diaglength() == (RP.N - 1)
    assert RP.max_vertlength() == RP.N
    if not sparse:
        assert RP.max_white_vertlength() == 0


def test_rqa_summary(small_RP):
    res = small_RP.rqa_summary()
    measures = ['RR', 'DET', 'L', 'LAM']
    assert all(res[m].dtype == DFIELD for m in measures)


def test_rqa_summary_numeric(small_RP_basic):
    res = small_RP_basic.rqa_summary()
    exp = {'RR': 0.48, 'DET': 0.8947, 'L': 8.4999, 'LAM': 0.9999}
    assert all(np.isclose(res[m], val, atol=1e-04)
               for m, val in exp.items())


@pytest.mark.parametrize('measure', ['diagline', 'vertline'])
@pytest.mark.parametrize('M', np.arange(5, 90, 40).tolist())
def test_resample_line_dist(measure: str, M: int, small_RP):
    res = getattr(small_RP, f"resample_{measure}_dist")(M)
    assert res.dtype == NODE
    assert res.shape[0] == small_RP.N
    assert (0 <= res).all()


@pytest.mark.parametrize(
    "var, exp", [("trapping", 4.7999), ("mean_recurrence", 3.9999)])
def test_time(small_RP_basic, var, exp):
    if small_RP_basic.sparse_rqa and var == "mean_recurrence":
        return
    res = getattr(small_RP_basic, f"{var}_time")()
    assert np.isclose(res, exp, atol=1e-04)


# test entropy

@pytest.mark.parametrize(
    "ts, measure, value",
    [(np.array([4, 7, 9, 10, 6, 11, 3]), m, v)
     for m, v in [("permutation", 0.5888), ("complexity", 0.29)]]
    + [(np.arange(20), "complexity", 0.0)])
def test_entropy(ts: np.ndarray, measure: str, value: float):
    rp = RecurrencePlot(ts[:, np.newaxis], threshold=1, dim=3, tau=1)
    assert np.isclose(getattr(rp, f"{measure}_entropy")(), value, atol=1e-04)


@pytest.mark.parametrize(
    'measure, exp',
    [('diag', 0.6931), ('vert', 1.2206), ('white_vert', 1.8848)])
def test_line_dist_entropy(measure: str, exp: float, small_RP_basic):
    if small_RP_basic.sparse_rqa and measure == "white_vert":
        return
    res = getattr(small_RP_basic, f"{measure}_entropy")()
    assert np.isclose(res, exp, atol=1e-04)
