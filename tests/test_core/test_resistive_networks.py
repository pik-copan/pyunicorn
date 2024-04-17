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
Simple tests for the ResNetwork class.
"""
import numpy as np

from pyunicorn import ResNetwork

# -----------------------------------------------------------------------------
# Class member tests with TestNetwork
# -----------------------------------------------------------------------------


def test_init(capsys):
    print(ResNetwork.SmallTestNetwork())
    out = capsys.readouterr()[0]
    out_ref = "ResNetwork:\nGeoNetwork:\nSpatialNetwork:\n" + \
              "Network: undirected, 5 nodes, 5 links, link density 0.500." + \
              "\nGeographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00 -180.00\n" + \
              "   max    9.0   90.00  180.00\n" + \
              "Average resistance: 2.4\n"
    assert out == out_ref


def test_SmallTestNetwork():
    assert isinstance(ResNetwork.SmallTestNetwork(), ResNetwork)


def test_SmallComplexNetwork():
    net = ResNetwork.SmallComplexNetwork()
    assert net.flagComplex

    adm = net.get_admittance()

    res = adm.real
    exp = [[0., 0.1, 0., 0., 0.],
           [0.1, 0., 0.0625, 0.25, 0.],
           [0., 0.0625, 0., 0.0625, 0.],
           [0., 0.25, 0.0625, 0., 0.05],
           [0., 0., 0., 0.05, 0.]]
    assert np.allclose(res, exp, atol=1e-04)

    res = adm.imag
    exp = [[0., -0.2, 0., 0., 0.],
           [-0.2, 0., -0.0625, -0.25, 0.],
           [0., -0.0625, 0., -0.0625, 0.],
           [0., -0.25, -0.0625, 0., -0.05],
           [0., 0., 0., -0.05, 0.]]
    assert np.allclose(res, exp, atol=1e-04)


def test_update_resistances():
    net = ResNetwork.SmallTestNetwork()
    net.update_resistances(net.adjacency)

    res = net.get_admittance()
    exp = [[0., 1., 0., 0., 0.],
           [1., 0., 1., 1., 0.],
           [0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 1.],
           [0., 0., 0., 1., 0.]]
    assert (res == exp).all()

    res = net.admittance_lapacian()
    exp = [[1., -1., 0., 0., 0.],
           [-1., 3., -1., -1., 0.],
           [0., -1., 2., -1., 0.],
           [0., -1., -1., 3., -1.],
           [0., 0., 0., -1., 1.]]
    assert (res == exp).all()


def test_update_admittance():
    net = ResNetwork.SmallTestNetwork()
    net.update_admittance()
    assert True


def test_get_admittance():
    net = ResNetwork.SmallTestNetwork()

    res = net.get_admittance()
    exp = [[0., 0.5, 0., 0., 0.],
           [0.5, 0., 0.125, 0.5, 0.],
           [0., 0.125, 0., 0.125, 0.],
           [0., 0.5, 0.125, 0., 0.1],
           [0., 0., 0., 0.1, 0.]]
    assert np.allclose(res, exp, atol=1e-04)


def test_update_R():
    net = ResNetwork.SmallTestNetwork()
    net.update_R()
    assert True


def test_get_R():
    res = ResNetwork.SmallTestNetwork().get_R()
    exp = [[2.28444444, 0.68444444, -0.56, -0.20444444, -2.20444444],
           [0.68444444, 1.08444444, -0.16, 0.19555556, -1.80444444],
           [-0.56, -0.16, 3.04, -0.16, -2.16],
           [-0.20444444, 0.19555556, -0.16, 1.08444444, -0.91555556],
           [-2.20444444, -1.80444444, -2.16, -0.91555556, 7.08444444]]
    assert np.allclose(res, exp, atol=1e-04)


def test_admittance_laplacian():
    res = ResNetwork.SmallTestNetwork().admittance_lapacian()
    exp = [[0.5, -0.5, 0., 0., 0.],
           [-0.5, 1.125, -0.125, -0.5, 0.],
           [0., -0.125, 0.25, -0.125, 0.],
           [0., -0.5, -0.125, 0.725, -0.1],
           [0., 0., 0., -0.1, 0.1]]
    assert np.allclose(res, exp, atol=1e-04)


def test_admittive_degree():
    res = ResNetwork.SmallTestNetwork().admittive_degree()
    exp = [0.5, 1.125, 0.25, 0.725, 0.1]
    assert np.allclose(res, exp, atol=1e-04)


def test_average_neighbors_admittive_degree():
    res = ResNetwork.SmallTestNetwork().average_neighbors_admittive_degree()
    exp = [2.25, 1.31111111, 7.4, 2.03448276, 7.25]
    assert np.allclose(res, exp, atol=1e-04)


def test_local_admittive_clustering():
    res = ResNetwork.SmallTestNetwork().local_admittive_clustering()
    exp = [0., 0.00694444, 0.0625, 0.01077586, 0.]
    assert np.allclose(res, exp, atol=1e-04)


def test_global_admittive_clustering():
    res = ResNetwork.SmallTestNetwork().global_admittive_clustering()
    exp = 0.016
    assert np.isclose(res, exp, atol=1e-04)


def test_effective_resistance():
    net = ResNetwork.SmallTestNetwork()

    res = net.effective_resistance(1, 1)
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.effective_resistance(1, 2)
    exp = 4.4444
    assert np.isclose(res, exp, atol=1e-04)


def test_average_effective_resistance():
    res = ResNetwork.SmallTestNetwork().average_effective_resistance()
    exp = 7.2889
    assert np.isclose(res, exp, atol=1e-04)


def test_diameter_effective_resistance():
    res = ResNetwork.SmallTestNetwork().diameter_effective_resistance()
    exp = 14.4444
    assert np.isclose(res, exp, atol=1e-04)


def test_effective_resistance_closeness_centrality():
    net = ResNetwork.SmallTestNetwork()

    res = net.effective_resistance_closeness_centrality(0)
    exp = 0.1538
    assert np.isclose(res, exp, atol=1e-04)

    res = net.effective_resistance_closeness_centrality(4)
    exp = 0.08
    assert np.isclose(res, exp, atol=1e-04)


def test_vertex_current_flow_betweenness():
    net = ResNetwork.SmallTestNetwork()

    res = net.vertex_current_flow_betweenness(1)
    exp = 0.3889
    assert np.isclose(res, exp, atol=1e-04)


def test_edge_current_flow_betweenness():
    net = ResNetwork.SmallTestNetwork()

    res = net.edge_current_flow_betweenness()
    exp = [[0., 0.4, 0., 0., 0.],
           [0.4, 0., 0.2444, 0.5333, 0.],
           [0., 0.2444, 0., 0.2444, 0.],
           [0., 0.5333, 0.2444, 0., 0.4],
           [0., 0., 0., 0.4, 0.]]
    assert np.allclose(res, exp, atol=1e-04)

    net.update_resistances(net.adjacency)
    res = net.edge_current_flow_betweenness()
    exp = [[0., 0.4, 0., 0., 0.],
           [0.4, 0., 0.3333, 0.4, 0.],
           [0., 0.3333, 0., 0.3333, 0.],
           [0., 0.4, 0.3333, 0., 0.4],
           [0., 0., 0., 0.4, 0.]]
    assert np.allclose(res, exp, atol=1e-04)
