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
Simple tests for the GeoGrid class.
"""
import numpy as np

from pyunicorn.core.geo_grid import GeoGrid


def test_RegularGrid():
    res = GeoGrid.RegularGrid(time_seq=np.arange(2),
                              space_grid=(np.array([0., 5.]),
                                          np.array([1., 2.])),
                              silence_level=2).lat_sequence()
    exp = np.array([0., 0., 5., 5.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

    res = GeoGrid.RegularGrid(time_seq=np.arange(2),
                              space_grid=(np.array([0., 5.]),
                                          np.array([1., 2.])),
                              silence_level=2).lon_sequence()
    exp = np.array([1., 2., 1., 2.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_coord_sequence_from_rect_grid():
    res = GeoGrid.coord_sequence_from_rect_grid(lat_grid=np.array([0., 5.]),
                                                lon_grid=np.array([1., 2.]))
    exp = (np.array([0., 0., 5., 5.]), np.array([1., 2., 1., 2.]))
    assert np.allclose(res, exp, atol=1e-04)


def test_lat_sequence():
    res = GeoGrid.SmallTestGrid().lat_sequence()
    exp = np.array([0., 5., 10., 15., 20., 25.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_lon_sequence():
    res = GeoGrid.SmallTestGrid().lon_sequence()
    exp = np.array([2.5, 5., 7.5, 10., 12.5, 15.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_convert_lon_coordinates():
    res = GeoGrid.SmallTestGrid().convert_lon_coordinates(
        np.array([10., 350., 20., 340., 170., 190.]))
    exp = np.array([10., -10., 20., -20., 170., -170.])
    assert np.allclose(res, exp, atol=1e-04)


def test_node_number2d():
    res = GeoGrid.SmallTestGrid().node_number(lat_node=14., lon_node=9.)
    exp = 3
    assert res == exp


def test_cos_lat():
    res = GeoGrid.SmallTestGrid().cos_lat()[:2]
    exp = np.array([1., 0.9962])
    assert np.allclose(res, exp, atol=1e-04)


def test_sin_lat():
    res = GeoGrid.SmallTestGrid().sin_lat()[:2]
    exp = np.array([0., 0.0872])
    assert np.allclose(res, exp, atol=1e-04)


def test_cos_lon():
    res = GeoGrid.SmallTestGrid().cos_lon()[:2]
    exp = np.array([0.999, 0.9962])
    assert np.allclose(res, exp, atol=1e-04)


def test_sin_lon():
    res = GeoGrid.SmallTestGrid().sin_lon()[:2]
    exp = np.array([0.0436, 0.0872])
    assert np.allclose(res, exp, atol=1e-04)


def test_angular_distance():
    res = GeoGrid.SmallTestGrid().angular_distance().round(2)
    exp = np.array([[0., 0.1, 0.19, 0.29, 0.39, 0.48],
                    [0.1, 0., 0.1, 0.19, 0.29, 0.39],
                    [0.19, 0.1, 0., 0.1, 0.19, 0.29],
                    [0.29, 0.19, 0.1, 0., 0.1, 0.19],
                    [0.39, 0.29, 0.19, 0.1, 0., 0.1],
                    [0.48, 0.39, 0.29, 0.19, 0.1, 0.]], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_boundaries():
    res = GeoGrid.SmallTestGrid().print_boundaries()
    exp = "         time     lat     lon\n   min    0.0    0.00    2.50\n" + \
          "   max    9.0   25.00   15.00"
    assert res == exp


def test_geometric_distance_distribution():
    res = GeoGrid.SmallTestGrid().geometric_distance_distribution(3)[0]
    exp = np.array([0.3333, 0.4667, 0.2])
    assert np.allclose(res, exp, atol=1e-04)

    res = GeoGrid.SmallTestGrid().geometric_distance_distribution(3)[1]
    exp = np.array([0., 0.1616, 0.3231, 0.4847])
    assert np.allclose(res, exp, atol=1e-04)


def test_region_indices():
    res = GeoGrid.SmallTestGrid().region_indices(
        np.array([0., 0., 0., 11., 11., 11., 11., 0.])).astype(int)
    exp = np.array([0, 1, 1, 0, 0, 0])
    assert np.allclose(res, exp, atol=1e-04)
