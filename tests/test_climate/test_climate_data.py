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
Simple tests for the ClimateData class.
"""
import numpy as np

from pyunicorn.core.data import Data
from pyunicorn.climate.climate_data import ClimateData

# -----------------------------------------------------------------------------
# Class member tests
# -----------------------------------------------------------------------------
def test_SmallTestData():
    res = Data.SmallTestData().observable()
    exp = np.array([[0., 1., 0., -1., -0., 1.],
                    [0.309, 0.9511, -0.309, -0.9511, 0.309, 0.9511],
                    [0.5878, 0.809, -0.5878, -0.809, 0.5878, 0.809],
                    [0.809, 0.5878, -0.809, -0.5878, 0.809, 0.5878],
                    [0.9511, 0.309, -0.9511, -0.309, 0.9511, 0.309],
                    [1., 0., -1., -0., 1., 0.],
                    [0.9511, -0.309, -0.9511, 0.309, 0.9511, -0.309],
                    [0.809, -0.5878, -0.809, 0.5878, 0.809, -0.5878],
                    [0.5878, -0.809, -0.5878, 0.809, 0.5878, -0.809],
                    [0.309, -0.9511, -0.309, 0.9511, 0.309, -0.9511]])
    assert np.allclose(res, exp, atol=1e-04)

def test_phase_indices():
    res = ClimateData.SmallTestData().phase_indices()
    exp = np.array([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
    assert (res == exp).all()

def test_indices_selected_phases():
    res = ClimateData.SmallTestData().indices_selected_phases([0, 1, 4])
    exp = np.array([0, 1, 4, 5, 6, 9])
    assert (res == exp).all()

def test_phase_mean():
    res = ClimateData.SmallTestData().phase_mean()
    exp = np.array([[0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                    [0.63, 0.321, -0.63, -0.321, 0.63, 0.321],
                    [0.6984, 0.1106, -0.6984, -0.1106, 0.6984, 0.1106],
                    [0.6984, -0.1106, -0.6984, 0.1106, 0.6984, -0.1106],
                    [0.63, -0.321, -0.63, 0.321, 0.63, -0.321]])
    assert np.allclose(res, exp, atol=1e-04)

def test_anomaly():
    res = ClimateData.SmallTestData().anomaly()[:, 0]
    exp = np.array([-0.5, -0.321, -0.1106, 0.1106, 0.321,
                    0.5, 0.321, 0.1106, -0.1106, -0.321])
    assert np.allclose(res, exp, atol=1e-04)

def test_shuffled_anomaly():
    res = ClimateData.SmallTestData().anomaly().std(axis=0)
    exp = np.array([0.31, 0.6355, 0.31, 0.6355, 0.31, 0.6355])
    assert np.allclose(res, exp, atol=1e-04)

    res = ClimateData.SmallTestData().shuffled_anomaly().std(axis=0)
    exp = np.array([0.31, 0.6355, 0.31, 0.6355, 0.31, 0.6355])
    assert np.allclose(res, exp, atol=1e-04)

def test_window():
    data = ClimateData.SmallTestData()
    data.set_window(window={"time_min": 0., "time_max": 0.,
                            "lat_min": 10., "lat_max": 20.,
                            "lon_min": 5., "lon_max": 10.})
    res = data.anomaly()
    exp = np.array([[0.5, -0.5], [0.321, -0.63], [0.1106, -0.6984],
                    [-0.1106, -0.6984], [-0.321, -0.63], [-0.5, 0.5],
                    [-0.321, 0.63], [-0.1106, 0.6984], [0.1106, 0.6984],
                    [0.321, 0.63]])
    assert np.allclose(res, exp, atol=1e-04)

def test_set_global_window():
    data = ClimateData.SmallTestData()
    data.set_window(window={"time_min": 0., "time_max": 4.,
                            "lat_min": 10., "lat_max": 20.,
                            "lon_min": 5., "lon_max": 10.})
    res = data.grid.grid()["lat"]
    exp = np.array([10., 15.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)

    data.set_global_window()
    res = data.grid.grid()["lat"]
    exp = np.array([0., 5., 10., 15., 20., 25.], dtype=np.float32)
    assert np.allclose(res, exp, atol=1e-04)
