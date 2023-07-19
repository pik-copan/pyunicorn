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
import numpy as np

from pyunicorn.timeseries.recurrence_plot import RecurrencePlot


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
