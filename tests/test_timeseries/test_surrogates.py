#!/usr/bin/python
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
Simple tests for the Surrogates class.
"""
import numpy as np

from pyunicorn.timeseries import Surrogates


def test_normalize_time_series_array():
    ts = Surrogates.SmallTestData().original_data
    Surrogates.SmallTestData().normalize_time_series_array(ts)
    res = ts.mean(axis=1)
    exp = np.array([0., 0., 0., 0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = ts.std(axis=1)
    exp = np.array([1., 1., 1., 1., 1., 1.])
    assert np.allclose(res, exp, atol=1e-04)

def test_embed_time_series_array():
    ts = Surrogates.SmallTestData().original_data
    res = Surrogates.SmallTestData().embed_time_series_array(
        time_series_array=ts, dimension=3, delay=2)[0, :6, :]
    exp = np.array([[0., 0.61464833, 1.14988147],
                    [0.31244015, 0.89680225, 1.3660254],
                    [0.61464833, 1.14988147, 1.53884177],
                    [0.89680225, 1.3660254, 1.6636525],
                    [1.14988147, 1.53884177, 1.73766672],
                    [1.3660254, 1.6636525, 1.76007351]])
    assert np.allclose(res, exp, atol=1e-04)

def test_white_noise_surrogates():
    ts = Surrogates.SmallTestData().original_data
    surrogates = Surrogates.SmallTestData().white_noise_surrogates(ts)

    assert(np.allclose(np.histogram(ts[0, :])[0],
                       np.histogram(surrogates[0, :])[0]))

def test_correlated_noise_surrogates():
    ts = Surrogates.SmallTestData().original_data
    surrogates = Surrogates.SmallTestData().correlated_noise_surrogates(ts)
    assert np.allclose(np.abs(np.fft.fft(ts, axis=1))[0, 1:10],
                       np.abs(np.fft.fft(surrogates, axis=1))[0, 1:10])
