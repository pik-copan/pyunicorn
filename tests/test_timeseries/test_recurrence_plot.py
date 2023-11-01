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

from itertools import chain, product

import pytest
import numpy as np

from pyunicorn.core import Data
from pyunicorn.core._ext.types import NODE, DFIELD
from pyunicorn.timeseries import RecurrencePlot


@pytest.fixture(scope="module", name="recurrence_crit", ids=str,
                params=list(chain(product(np.arange(0, 2.1, .5), [None]),
                                  product([None], np.arange(0, 1.1, .2)))))
def recurrence_crit_fixture(request):
    threshold, rate = request.param
    assert np.sum([threshold is None, rate is None]) == 1
    return request.param


@pytest.fixture(scope="module", name="small_RP")
def small_RP_fixture(metric, recurrence_crit):
    x = Data.SmallTestData().observable()
    threshold, rate = recurrence_crit
    return RecurrencePlot(
        x, threshold=threshold, recurrence_rate=rate, metric=metric)


@pytest.mark.parametrize("measure", ["diagline", "vertline", "white_vertline"])
def test_dist(measure: str, small_RP):
    res = getattr(small_RP, f"{measure}_dist")()
    assert res.dtype == NODE
    assert res.shape[0] == small_RP.N
    assert (0 <= res).all() and (res <= small_RP.N).all()


def test_rqa_summary(small_RP):
    res = small_RP.rqa_summary()
    measures = ['RR', 'DET', 'L', 'LAM']
    assert all(res[m].dtype == DFIELD for m in measures)


@pytest.mark.parametrize(
    "ts, measure, value",
    [(np.array([4, 7, 9, 10, 6, 11, 3]), m, v)
     for m, v in [("permutation", 0.5888), ("complexity", 0.29)]]
    + [(np.arange(20), "complexity", 0.0)])
def test_entropy(ts: np.ndarray, measure: str, value: float):
    rp = RecurrencePlot(ts[:, np.newaxis], threshold=1, dim=3, tau=1)
    assert np.isclose(getattr(rp, f"{measure}_entropy")(), value, atol=1e-04)
