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
Simple tests for the Grid class.
"""
import numpy as np

from pyunicorn.core.grid import Grid


def test_RegularGrid():
    res = Grid.RegularGrid(time_seq=np.arange(2),
                           space_grid=np.array([[0., 5.], [1., 2.]]),
                           silence_level=2).sequence(0)
    exp = np.array([0., 0., 5., 5.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

    res = Grid.RegularGrid(time_seq=np.arange(2),
                           space_grid=np.array([[0., 5.], [1., 2.]]),
                           silence_level=2).sequence(1)
    exp = np.array([1., 2., 1., 2.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_coord_sequence_from_rect_grid():
    res = Grid.coord_sequence_from_rect_grid(space_grid=np.array([[0., 5.],
                                                                  [1., 2.]]))
    exp = (np.array([0., 0., 5., 5.]), np.array([1., 2., 1., 2.]))
    assert np.allclose(res, exp, atol=1e-04)


def test_sequence():
    res = Grid.SmallTestGrid().sequence(0)
    exp = np.array([0., 5., 10., 15., 20., 25.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_node_number():
    res = Grid.SmallTestGrid().node_number(x=(14., 9.))
    exp = 3
    assert res == exp


def test_node_coordinates():
    res = Grid.SmallTestGrid().node_coordinates(3)
    exp = (15.0, 10.0)
    assert np.allclose(res, exp, atol=1e-04)


def test_grid():
    res = Grid.SmallTestGrid().grid()["space"][0]
    exp = np.array([0., 5., 10., 15., 20., 25.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

    res = Grid.SmallTestGrid().grid()["space"][1]
    exp = np.array([2.5, 5., 7.5, 10., 12.5, 15.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)


def test_grid_size():
    res = Grid.SmallTestGrid().print_grid_size()
    exp = '     space    time\n         6      10'
    assert res == exp


def test_geometric_distance_distribution():
    res = Grid.SmallTestGrid().geometric_distance_distribution(3)[0]
    exp = np.array([0.3333, 0.4667, 0.2])
    assert np.allclose(res, exp, atol=1e-04)

    res = Grid.SmallTestGrid().geometric_distance_distribution(3)[1]
    exp = np.array([0., 9.317, 18.6339, 27.9509])
    assert np.allclose(res, exp, atol=1e-04)


def test_euclidean_distance():
    res = Grid.SmallTestGrid().euclidean_distance().round(2)
    exp = np.array([[0., 5.59, 11.18, 16.77, 22.36, 27.95],
                    [5.59, 0., 5.59, 11.18, 16.77, 22.36],
                    [11.18, 5.59, 0., 5.59, 11.18, 16.77],
                    [16.77, 11.18, 5.59, 0., 5.59, 11.18],
                    [22.36, 16.77, 11.18, 5.59, 0., 5.59],
                    [27.95, 22.36, 16.77, 11.18, 5.59, 0.]], dtype=np.float32)
    assert (res == exp).all()
