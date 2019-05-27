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
Simple tests for the GeoNetwork class.
"""
import numpy as np

from pyunicorn.core.geo_network import GeoNetwork
from pyunicorn.core.grid import Grid

def test_ErdosRenyi(capsys):
    print(GeoNetwork.ErdosRenyi(grid=Grid.SmallTestGrid(),
                                n_nodes=6, n_links=5))
    out, err = capsys.readouterr()
    out_ref = "Generating Erdos-Renyi random graph with 6 nodes and 5 " + \
              "links...\nSetting area weights according to type surface " + \
              "...\nGeoNetwork:\n" + \
              "Network: undirected, 6 nodes, 5 links, " + \
              "link density 0.333.\nGeographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00    2.50\n" + \
              "   max    9.0   25.00   15.00\n"
    assert out == out_ref

def test_ConfigurationModel():
    n = 0
    while n != 7:
        net = GeoNetwork.ConfigurationModel(
            grid=Grid.SmallTestGrid(),
            degrees=GeoNetwork.SmallTestNetwork().degree(),
            silence_level=2)
        n = net.n_links
    res = net.link_density
    exp = 0.46666667
    assert np.allclose(res, exp, atol=1e-04)

def test_randomly_rewire_geomodel_I():
    net = GeoNetwork.SmallTestNetwork()
    net.randomly_rewire_geomodel_I(distance_matrix=net.grid.angular_distance(),
                                   iterations=100, inaccuracy=1.0)
    res = net.degree()
    exp = np.array([3, 3, 2, 2, 3, 1])
    assert (res == exp).all()

def test_set_random_links_by_distance():
    net = GeoNetwork.SmallTestNetwork()
    while net.n_links != 5:
        net.set_random_links_by_distance(a=0., b=-4.)
    res = net.n_links
    exp = 5
    assert res == exp

def test_geographical_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.geographical_distribution(sequence=net.degree(), n_bins=3)[0]
    exp = np.array([0.15645071, 0.33674395, 0.50680541])
    assert np.allclose(res, exp, atol=1e-04)

def test_geographical_cumulative_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.geographical_cumulative_distribution(sequence=net.degree(),
                                                   n_bins=3)[0]
    exp = np.array([1.00000007, 0.84354936, 0.50680541])
    assert np.allclose(res, exp, atol=1e-04)

def test_link_distance_distribution():
    net = GeoNetwork.SmallTestNetwork()

    res = net.link_distance_distribution(n_bins=4, geometry_corrected=False)[0]
    exp = np.array([0.14285714, 0.28571429, 0.28571429, 0.28571429])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.link_distance_distribution(n_bins=4, geometry_corrected=True)[0]
    exp = np.array([0.09836066, 0.24590164, 0.32786885, 0.32786885])
    assert np.allclose(res, exp, atol=1e-04)

def test_area_weighted_connectivity():
    res = GeoNetwork.SmallTestNetwork().area_weighted_connectivity()
    exp = np.array([0.48540673, 0.4989577, 0.33418113, 0.34459165, 0.51459336,
                    0.17262428], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

def test_inarea_weighted_connectivity():
    res = GeoNetwork.SmallTestNetwork().inarea_weighted_connectivity()
    exp = np.array([0.48540673, 0.4989577, 0.33418113, 0.34459165, 0.51459336,
                    0.17262428], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

def test_outarea_weighted_connectivity():
    res = GeoNetwork.SmallTestNetwork().outarea_weighted_connectivity()
    exp = np.array([0.4854067, 0.4989577, 0.33418113, 0.34459165, 0.51459336,
                    0.17262428], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

def test_area_weighted_connectivity_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.area_weighted_connectivity_distribution(n_bins=4)[0]
    exp = np.array([0.15645071, 0.33674395, 0.34459165, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_inarea_weighted_connectivity_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.inarea_weighted_connectivity_distribution(n_bins=4)[0]
    exp = np.array([0.15645071, 0.33674395, 0.34459165, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_outarea_weighted_connectivity_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.outarea_weighted_connectivity_distribution(n_bins=4)[0]
    exp = np.array([0.15645071, 0.33674395, 0.34459165, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_area_weighted_connectivity_cumulative_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.area_weighted_connectivity_cumulative_distribution(n_bins=4)[0]
    exp = np.array([1.00000007, 0.84354936, 0.50680541, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_inarea_weighted_connectivity_cumulative_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.inarea_weighted_connectivity_cumulative_distribution(n_bins=4)[0]
    exp = np.array([1.00000007, 0.84354936, 0.50680541, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_outarea_weighted_connectivity_cumulative_distribution():
    net = GeoNetwork.SmallTestNetwork()
    res = net.outarea_weighted_connectivity_cumulative_distribution(
        n_bins=4)[0]
    exp = np.array([1.00000007, 0.84354936, 0.50680541, 0.16221375])
    assert np.allclose(res, exp, atol=1e-04)

def test_average_neighbor_area_weighted_connectivity():
    net = GeoNetwork.SmallTestNetwork()
    res = net.average_neighbor_area_weighted_connectivity()
    exp = np.array([0.3439364, 0.39778873, 0.5067755, 0.4921822, 0.4395152,
                    0.48540673], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

def test_max_neighbor_area_weighted_connectivity():
    net = GeoNetwork.SmallTestNetwork()
    res = net.max_neighbor_area_weighted_connectivity()
    exp = np.array([0.51459336, 0.51459336, 0.51459336, 0.49895769, 0.49895769,
                    0.48540673])
    assert np.allclose(res, exp, atol=1e-04)

def test_average_link_distance():
    net = GeoNetwork.SmallTestNetwork()

    res = net.average_link_distance(geometry_corrected=False)
    exp = np.array([0.3884563, 0.19434186, 0.14557932, 0.24325928, 0.29118925,
                    0.48467883])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.average_link_distance(geometry_corrected=True)[:-1]
    exp = np.array([1.5987561, 1.0920637, 1.00011046, 1.67081917, 1.63841055])
    assert np.allclose(res, exp, atol=1e-04)

def test_inaverage_link_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.inaverage_link_distance(geometry_corrected=False)
    exp = np.array([0.3884563, 0.19434186, 0.14557932, 0.24325928, 0.29118925,
                    0.48467883])
    assert np.allclose(res, exp, atol=1e-04)

def test_outaverage_link_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.outaverage_link_distance(geometry_corrected=False)
    exp = np.array([0.3884563, 0.19434186, 0.14557932, 0.24325928, 0.29118925,
                    0.48467883])
    assert np.allclose(res, exp, atol=1e-04)

def test_total_link_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.total_link_distance(geometry_corrected=False)
    exp = np.array([0.1885593, 0.09696837, 0.04864986, 0.08382512, 0.14984406,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_intotal_link_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.intotal_link_distance(geometry_corrected=False)
    exp = np.array([0.1885593, 0.09696837, 0.04864986, 0.08382512, 0.14984406,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_outtotal_link_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.outtotal_link_distance(geometry_corrected=False)
    exp = np.array([0.18855929, 0.09696837, 0.04864986, 0.08382512, 0.14984406,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_connectivity_weighted_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.connectivity_weighted_distance()
    exp = np.array([0.0625227, 0.03207134, 0.02408994, 0.04192858, 0.0500332,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_inconnectivity_weighted_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.inconnectivity_weighted_distance()
    exp = np.array([0.0625227, 0.03207134, 0.02408994, 0.04192858, 0.0500332,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_outconnectivity_weighted_distance():
    net = GeoNetwork.SmallTestNetwork()
    res = net.outconnectivity_weighted_distance()
    exp = np.array([0.0625227, 0.03207134, 0.02408994, 0.04192858, 0.0500332,
                    0.08366733])
    assert np.allclose(res, exp, atol=1e-04)

def test_max_link_distance():
    res = GeoNetwork.SmallTestNetwork().max_link_distance()
    exp = np.array([0.48467883, 0.2911402, 0.19376467, 0.2920272, 0.38866287,
                    0.48467883], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

def test_average_distance_weighted_path_length():
    res = GeoNetwork.SmallTestNetwork().average_distance_weighted_path_length()
    exp = 0.49846491
    assert np.allclose(res, exp, atol=1e-04)

def test_distance_weighted_closeness():
    res = GeoNetwork.SmallTestNetwork().distance_weighted_closeness()
    exp = np.array([2.23782229, 2.45008978, 2.23956348, 2.45008978, 2.2396005,
                    1.19817005])
    assert np.allclose(res, exp, atol=1e-04)

def test_local_distance_weighted_vulnerability():
    res = GeoNetwork.SmallTestNetwork().local_distance_weighted_vulnerability()
    exp = np.array([0.03252519, 0.3136603, 0.20562718, 0.02799094, -0.02828809,
                    -0.28798557])
    assert np.allclose(res, exp, atol=1e-04)

def test_local_geographical_clustering():
    res = GeoNetwork.SmallTestNetwork().local_geographical_clustering()
    exp = np.array([0., 0.0998, 0.1489, 0., 0.2842, 0.])
    assert np.allclose(res, exp, atol=1e-04)
