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
Simple tests for the JointRecurrencePlot class.
"""

import numpy as np

import pytest

from pyunicorn.timeseries import JointRecurrencePlot
from pyunicorn.funcnet import CouplingAnalysis


@pytest.mark.parametrize("n", [2, 10, 50])
def test_recurrence(metric: str, n: int):
    ts = CouplingAnalysis.test_data()[:n, 0]
    jrp = JointRecurrencePlot(ts, ts, threshold=(.1, .1),
                              metric=(metric, metric))
    dist = {}
    for i in "xy":
        jrp.embedding = getattr(jrp, f"{i}_embedded")
        dist[i] = jrp.distance_matrix(metric=metric)
    assert all(d.shape == (n, n) for d in dist.values())
    assert np.allclose(dist["x"], dist["y"])
    assert jrp.recurrence_matrix().shape == (n, n)


def test_exceptions():
    ts1 = CouplingAnalysis.test_data()[:10, 0]
    ts2 = CouplingAnalysis.test_data()[:12, 0]
    # provoke error for missing thresholding parameter
    with pytest.raises(NameError):
        JointRecurrencePlot(ts1, ts1)
    # provoke error for mismatched timeseries lengths
    with pytest.raises(ValueError):
        JointRecurrencePlot(ts1, ts2)
    # provoke error for given lag value too large
    with pytest.raises(ValueError):
        JointRecurrencePlot(ts1, ts1, lag=14)
