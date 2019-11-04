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
Simple tests for the SpatialNetwork class.
"""
import numpy as np

from pyunicorn.core.geo_network import SpatialNetwork

def test_randomly_rewire_geomodel_I():
    net = SpatialNetwork.SmallTestNetwork()
    net.randomly_rewire_geomodel_I(distance_matrix=net.grid.distance(),
                                   iterations=100, inaccuracy=100)
    res = net.degree()
    exp = np.array([3, 3, 2, 2, 3, 1])
    assert (res == exp).all()

def test_set_random_links_by_distance():
    net = SpatialNetwork.SmallTestNetwork()
    while net.n_links != 5:
        net.set_random_links_by_distance(a=0., b=-0.04)
    res = net.n_links
    exp = 5
    assert res == exp

def test_link_distance_distribution():
    net = SpatialNetwork.SmallTestNetwork()

    res = net.link_distance_distribution(n_bins=4, geometry_corrected=False)[0]
    exp = np.array([0.14285714, 0.28571429, 0.28571429, 0.28571429])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.link_distance_distribution(n_bins=4, geometry_corrected=True)[0]
    exp = np.array([0.09836066, 0.24590164, 0.32786885, 0.32786885])
    assert np.allclose(res, exp, atol=1e-04)
