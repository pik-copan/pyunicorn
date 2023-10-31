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
Simple tests for the RecurrencePlot class.
"""
import pytest
import numpy as np

from pyunicorn.timeseries import RecurrencePlot
from pyunicorn.core import Data
from pyunicorn.core._ext.types import NODE, DFIELD


@pytest.fixture(
    scope='session',
    params=[
        (0, None, "supremum"), (.5, None, "supremum"),
        (1.5, None, "supremum"), (None, .2, "supremum"),
        (0, None, "euclidean"), (.5, None, "euclidean"),
        (1.5, None, "euclidean"), (None, .2, "euclidean"),
        (0, None, "manhattan"), (.5, None, "manhattan"),
        (1.5, None, "manhattan"), (None, .2, "manhattan")],
    ids=[
        '0-None-supremum', '.5-None-supremum"',
        '1.5-None-supremum', 'None-.2-supremum',
        '0-None-euclidean', '.5-None-euclidean-',
        '1.5-None-euclidean', 'None-.2-euclidean"',
        '0-None-manhattan', '.5-None-manhattan',
        '1.5-None-manhattan', 'None-.2-manhattan'],
    name="test_RP")
def test_RP_fixture(request):
    x = Data.SmallTestData().observable()
    RP = RecurrencePlot(x,
                        threshold=request.param[0],
                        recurrence_rate=request.param[1],
                        metric=request.param[2])
    return RP


def test_d_dist(test_RP):
    d_dist = test_RP.diagline_dist()
    assert d_dist.dtype == NODE
    assert d_dist.shape[0] == test_RP.N


def test_vertline_dist(test_RP):
    v_dist = test_RP.vertline_dist()
    assert v_dist.dtype == NODE
    assert v_dist.shape[0] == test_RP.N


def test_white_vertline_dist(test_RP):
    wv_dist = test_RP.white_vertline_dist()
    assert wv_dist.dtype == NODE
    assert wv_dist.shape[0] == test_RP.N


def test_rqa_summary(test_RP):
    res = test_RP.rqa_summary()
    measures = ['RR', 'DET', 'L', 'LAM']
    assert all(res[m].dtype == DFIELD for m in measures)


def test_permutation_entropy():
    ts = np.array([[4], [7], [9], [10], [6], [11], [3]])
    rp = RecurrencePlot(ts, threshold=1, dim=3, tau=1)

    res = rp.permutation_entropy()
    exp = 0.5888
    assert np.isclose(res, exp, atol=1e-04)


def test_complexity_entropy():
    ts = np.array([[4], [7], [9], [10], [6], [11], [3]])
    rp = RecurrencePlot(ts, threshold=1, dim=3, tau=1)

    res = rp.complexity_entropy()
    exp = 0.29
    assert np.isclose(res, exp, atol=1e-04)

    ts = np.array([[1], [2], [3], [4], [5], [6], [7]])
    rp = RecurrencePlot(ts, threshold=1, dim=3, tau=1)

    res = rp.complexity_entropy()
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)
