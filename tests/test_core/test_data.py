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
Simple tests for the Data class.
"""
import numpy as np

from pyunicorn.core.data import Data


def test_observable():
    obs = Data.SmallTestData().observable()
    obs_ref = np.array([[0.00000000e+00, 1.00000000e+00, 1.22464680e-16,
                         -1.00000000e+00, -2.44929360e-16, 1.00000000e+00],
                        [3.09016994e-01, 9.51056516e-01, -3.09016994e-01,
                         -9.51056516e-01, 3.09016994e-01, 9.51056516e-01],
                        [5.87785252e-01, 8.09016994e-01, -5.87785252e-01,
                         -8.09016994e-01, 5.87785252e-01, 8.09016994e-01],
                        [8.09016994e-01, 5.87785252e-01, -8.09016994e-01,
                         -5.87785252e-01, 8.09016994e-01, 5.87785252e-01],
                        [9.51056516e-01, 3.09016994e-01, -9.51056516e-01,
                         -3.09016994e-01, 9.51056516e-01, 3.09016994e-01],
                        [1.00000000e+00, 1.22464680e-16, -1.00000000e+00,
                         -2.44929360e-16, 1.00000000e+00, 3.67394040e-16],
                        [9.51056516e-01, -3.09016994e-01, -9.51056516e-01,
                         3.09016994e-01, 9.51056516e-01, -3.09016994e-01],
                        [8.09016994e-01, -5.87785252e-01, -8.09016994e-01,
                         5.87785252e-01, 8.09016994e-01, -5.87785252e-01],
                        [5.87785252e-01, -8.09016994e-01, -5.87785252e-01,
                         8.09016994e-01, 5.87785252e-01, -8.09016994e-01],
                        [3.09016994e-01, -9.51056516e-01, -3.09016994e-01,
                         9.51056516e-01, 3.09016994e-01, -9.51056516e-01]])

    isequal = np.allclose(obs, obs_ref)
    assert isequal


def test_window():
    lon_min = Data.SmallTestData().window()["lon_min"]
    lon_max = Data.SmallTestData().window()["lon_max"]
    lon_min_ref = 2.5
    lon_max_ref = 15.0
    isequal = np.allclose(lon_min, lon_min_ref) and \
        np.allclose(lon_max, lon_max_ref)
    assert isequal


def test_set_window():
    data = Data.SmallTestData()
    data.set_window(window={"time_min": 0., "time_max": 4., "lat_min": 10.,
                            "lat_max": 20., "lon_min": 5., "lon_max": 10.})
    obs = data.observable()
    obs_ref = np.array([[1.22464680e-16, -1.00000000e+00],
                        [-3.09016994e-01, -9.51056516e-01],
                        [-5.87785252e-01, -8.09016994e-01],
                        [-8.09016994e-01, -5.87785252e-01],
                        [-9.51056516e-01, -3.09016994e-01]])

    isequal = np.allclose(obs, obs_ref)
    assert isequal


def test_set_global_window():
    data = Data.SmallTestData()
    data.set_window(window={"time_min": 0., "time_max": 4.,
                            "lat_min": 10., "lat_max": 20., "lon_min": 5.,
                            "lon_max": 10.})

    lat1 = data.grid.grid()["lat"]
    data.set_global_window()
    lat2 = data.grid.grid()["lat"]

    lat1_ref = np.array([10., 15.], dtype=np.float32)
    lat2_ref = np.array([0., 5., 10., 15., 20., 25.], dtype=np.float32)

    isequal = np.allclose(lat1, lat1_ref) and np.allclose(lat2, lat2_ref)
    assert isequal


def test_normalize_time_series_array():
    ts = np.arange(16).reshape(4, 4).astype("float")
    Data.normalize_time_series_array(ts)

    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    ts = ts[:, 0]

    mean_ref = np.array([0., 0., 0., 0.])
    std_ref = np.array([1., 1., 1., 1.])
    ts_ref = np.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079])

    isequal = np.allclose(mean, mean_ref) and np.allclose(std, std_ref) and \
        np.allclose(ts, ts_ref)
    assert isequal


def test_next_power_2():
    isequal = np.allclose(Data.next_power_2(253), 256)
    assert isequal


def test_zero_pad_data():
    ts = np.arange(20).reshape(5, 4)
    zpd = Data.zero_pad_data(ts)

    zpd_ref = np.array([[0., 0., 0., 0.], [0., 1., 2., 3.],
                        [4., 5., 6., 7.], [8., 9., 10., 11.],
                        [12., 13., 14., 15.], [16., 17., 18., 19.],
                        [0., 0., 0., 0.], [0., 0., 0., 0.]])

    isequal = np.allclose(zpd, zpd_ref)
    assert isequal


def test_cos_window():
    ts = np.arange(24).reshape(12, 2)
    cw = Data.cos_window(data=ts, gamma=0.75)
    cw_ref = np.array([[0., 0.], [0.14644661, 0.14644661],
                       [0.5, 0.5], [0.85355339, 0.85355339],
                       [1., 1.], [1., 1.],
                       [1., 1.], [1., 1.],
                       [0.85355339, 0.85355339], [0.5, 0.5],
                       [0.14644661, 0.14644661], [0., 0.]])

    isequal = np.allclose(cw, cw_ref)
    assert isequal
