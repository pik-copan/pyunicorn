#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
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
Simple tests for the ClimateNetwork class.
"""
import numpy as np

from pyunicorn.climate.climate_network import ClimateNetwork

def test_str(capsys):
    print(ClimateNetwork.SmallTestNetwork())
    out, err = capsys.readouterr()
    out_ref = "ClimateNetwork:\n" + \
              "GeoNetwork:\n" + \
              "Network: undirected, 6 nodes, 7 links, link density 0.467." + \
              "\nGeographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00    2.50\n" + \
              "   max    9.0   25.00   15.00\n" + \
              "Threshold: 0.5\n" + \
              "Local connections filtered out: False\n"
    assert out == out_ref

def test_SmallTestNetwork():
    res = ClimateNetwork.SmallTestNetwork().adjacency
    exp = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    assert np.allclose(res, exp, atol=1e-04)

def test_link_density_function():
    res = ClimateNetwork.SmallTestNetwork().link_density_function(5)
    exp = (np.array([0., 0.2778, 0.4444, 0.6111, 0.7222]),
           np.array([0.1, 0.28, 0.46, 0.64, 0.82, 1.]))

    assert np.allclose(res[0], exp[0], atol=1e-04)
    assert np.allclose(res[1], exp[1], atol=1e-04)

def test_threshold_from_link_density():
    res = ClimateNetwork.SmallTestNetwork().\
        threshold_from_link_density(link_density=0.5)
    exp = 0.4
    assert np.isclose(res, exp, atol=1e-04)

def test_similarity_measure():
    res = ClimateNetwork.SmallTestNetwork().similarity_measure()[0, :]
    exp = np.array([1., 0.1, 0.2, 0.6, 0.7, 0.55])
    assert np.allclose(res, exp, atol=1e-04)

def test_non_local():
    res = ClimateNetwork.SmallTestNetwork().non_local()
    exp = False
    assert res == exp

def test_set_non_local():
    net = ClimateNetwork.SmallTestNetwork()
    net.set_non_local(non_local=True)

    res = net.adjacency
    exp = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    assert np.allclose(res, exp, atol=1e-04)

def test_threshold():
    res = ClimateNetwork.SmallTestNetwork().threshold()
    exp = 0.5
    assert np.isclose(res, exp, atol=1e-04)

def test_set_threshold():
    net = ClimateNetwork.SmallTestNetwork()

    res = net.n_links
    exp = 7
    assert res == exp

    net.set_threshold(threshold=0.7)
    res = net.n_links
    exp = 3
    assert res == exp

def test_set_link_density():
    net = ClimateNetwork.SmallTestNetwork()

    res = net.link_density
    exp = 0.4667
    assert np.isclose(res, exp, atol=1e-04)

    net.set_link_density(link_density=0.7)
    res = net.link_density
    exp = 0.6667
    assert np.isclose(res, exp, atol=1e-04)

def test_correlation_distance():
    res = ClimateNetwork.SmallTestNetwork().correlation_distance().round(2)
    exp = np.array([[0., 0.01, 0.04, 0.18, 0.27, 0.27],
                    [0.01, 0., 0.05, 0.18, 0.29, 0.12],
                    [0.04, 0.05, 0., 0.02, 0.16, 0.03],
                    [0.18, 0.18, 0.01, 0., 0.01, 0.06],
                    [0.27, 0.29, 0.16, 0.01, 0., 0.04],
                    [0.27, 0.12, 0.03, 0.06, 0.04, 0.]])
    assert np.allclose(res, exp, atol=1e-04)

def test_correlation_distance_weighted_closeness():
    res = ClimateNetwork.SmallTestNetwork().\
        correlation_distance_weighted_closeness()
    exp = np.array([0.1646, 0.1351, 0.0894, 0.1096, 0.1659, 0.1102])
    assert np.allclose(res, exp, atol=1e-04)

def test_local_correlation_distance_weighted_vulnerability():
    res = ClimateNetwork.SmallTestNetwork().\
        local_correlation_distance_weighted_vulnerability()
    exp = np.array([0.4037, 0.035, -0.1731, -0.081, 0.3121, -0.0533])
    assert np.allclose(res, exp, atol=1e-04)
