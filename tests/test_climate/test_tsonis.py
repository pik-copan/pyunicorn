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
Tests for the TsonisClimateNetwork class.
"""
import numpy as np

from pyunicorn.climate.climate_data import ClimateData
from pyunicorn.climate.tsonis import TsonisClimateNetwork


def test_str(capsys):
    print(TsonisClimateNetwork.SmallTestNetwork())
    out = capsys.readouterr()[0]
    out_ref = "TsonisClimateNetwork:\n" + \
              "ClimateNetwork:\n" + \
              "GeoNetwork:\n" + \
              "SpatialNetwork:\n" + \
              "Network: undirected, 6 nodes, 6 links, link density 0.400." + \
              "\nGeographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00    2.50\n" + \
              "   max    9.0   25.00   15.00\n" + \
              "Threshold: 0.5\n" + \
              "Local connections filtered out: False\n" + \
              "Use only data points from winter months: False\n"
    assert out == out_ref


def test_SmallTestNetwork():
    res = TsonisClimateNetwork.SmallTestNetwork().adjacency
    exp = np.array([[0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]])
    assert np.allclose(res, exp, atol=1e-04)


def test_calculate_similarity_measure():
    res = TsonisClimateNetwork.SmallTestNetwork().calculate_similarity_measure(
        anomaly=ClimateData.SmallTestData().anomaly())
    exp = np.array([[1., -0.2538, -1., 0.2538, 1., -0.2538],
                    [-0.2538, 1., 0.2538, -1., -0.2538, 1.],
                    [-1., 0.2538, 1., -0.2538, -1., 0.2538],
                    [0.2538, -1., -0.2538, 1., 0.2538, -1.],
                    [1., -0.2538, -1., 0.2538, 1., -0.2538],
                    [-0.2538, 1., 0.2538, -1., -0.2538, 1.]],
                   dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_correlation():
    res = TsonisClimateNetwork.SmallTestNetwork().correlation()
    exp = np.array([[1., 0.25377226, 1., 0.25377226, 1., 0.25377226],
                    [0.25377226, 1., 0.25377226, 1., 0.25377226, 1.],
                    [1., 0.25377226, 1., 0.25377226, 1., 0.25377226],
                    [0.25377226, 1., 0.25377226, 1., 0.25377226, 1.],
                    [1., 0.25377226, 1., 0.25377226, 1., 0.25377226],
                    [0.25377226, 1., 0.25377226, 1., 0.25377226, 1.]],
                   dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_winter_only():
    res = TsonisClimateNetwork.SmallTestNetwork().winter_only()
    exp = False
    assert res == exp


def test_set_winter_only():
    net = TsonisClimateNetwork.SmallTestNetwork()
    net.set_winter_only(winter_only=False)

    res = net.n_links
    exp = 6
    assert res == exp


def test_correlation_weighted_average_path_length():
    res = TsonisClimateNetwork.SmallTestNetwork().\
            correlation_weighted_average_path_length()
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)


def test_correlation_weighted_closeness():
    res = TsonisClimateNetwork.SmallTestNetwork().\
                correlation_weighted_closeness()
    exp = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    assert np.allclose(res, exp, atol=1e-04)


def test_local_correlation_weighted_vulnerability():
    res = TsonisClimateNetwork.SmallTestNetwork().\
                local_correlation_weighted_vulnerability()
    exp = np.array([0., 0., 0., 0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)
