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
Tests for the EventSynchronizationClimateNetworkClimateData class.
"""
import numpy as np

from pyunicorn.core.data import Data
from pyunicorn.climate.eventsynchronization_climatenetwork import\
 EventSynchronizationClimateNetwork


def test_str(capsys):
    data = EventSynchronizationClimateNetwork.SmallTestData()
    print(EventSynchronizationClimateNetwork(data, 0.8, 16))
    out, err = capsys.readouterr()
    out_ref = "Extracting network adjacency matrix by thresholding...\n" + \
              "Setting area weights according to type surface ...\n" + \
              "Setting area weights according to type surface ...\n" + \
              "EventSynchronizationClimateNetwork:\n" + \
              "EventSynchronization: 6 variables, 10 timesteps, taumax: 16" + \
              "\nClimateNetwork:\n" + \
              "GeoNetwork:\n" + \
              "Network: directed, 6 nodes, 0 links, link density 0.000.\n" + \
              "Geographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00    2.50\n" + \
              "   max    9.0   25.00   15.00\n" + \
              "Threshold: 0\n" + \
              "Local connections filtered out: False\n" + \
              "Type of event synchronization to construct " + \
              "the network: directedES\n"
    assert out == out_ref

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
