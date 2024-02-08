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


def test_average_link_distance():
    net = SpatialNetwork.SmallTestNetwork()

    res = net.average_link_distance(geometry_corrected=False)
    exp = np.array([22.36067963, 11.18033981, 8.38525486, 13.97542477,
                    16.77050908, 27.95084953])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.average_link_distance(geometry_corrected=True)[:-1]
    exp = np.array([1.6, 1.09090909, 1., 1.66666667, 1.63636357])
    assert np.allclose(res, exp, atol=1e-04)


def test_inaverage_link_distance():
    net = SpatialNetwork.SmallTestNetwork()
    res = net.inaverage_link_distance(geometry_corrected=False)
    exp = np.array([22.36067963, 11.18033981, 8.38525486, 13.97542477,
                    16.77050908, 27.95084953])
    assert np.allclose(res, exp, atol=1e-04)


def test_outaverage_link_distance():
    net = SpatialNetwork.SmallTestNetwork()
    res = net.outaverage_link_distance(geometry_corrected=False)
    exp = np.array([22.36067963, 11.18033981, 8.38525486, 13.97542477,
                    16.77050908, 27.95084953])
    assert np.allclose(res, exp, atol=1e-04)


def test_max_link_distance():
    res = SpatialNetwork.SmallTestNetwork().max_link_distance()
    exp = np.array([27.95085, 16.77051, 11.18034, 16.77051, 22.36068,
                    27.95085], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_average_distance_weighted_path_length():
    res = SpatialNetwork.SmallTestNetwork(). \
            average_distance_weighted_path_length()
    exp = 28.69620552062988
    assert np.allclose(res, exp, atol=1e-04)


def test_distance_weighted_closeness():
    res = SpatialNetwork.SmallTestNetwork().distance_weighted_closeness()
    exp = np.array([0.03888814, 0.04259177, 0.03888814, 0.04259177, 0.03888814,
                    0.02080063])
    assert np.allclose(res, exp, atol=1e-04)


def test_local_distance_weighted_vulnerability():
    res = SpatialNetwork.SmallTestNetwork(). \
            local_distance_weighted_vulnerability()
    exp = np.array([0.03233506, 0.31442454, 0.20580213, 0.02843829,
                    -0.02929477, -0.2883446])
    assert np.allclose(res, exp, atol=1e-04)
