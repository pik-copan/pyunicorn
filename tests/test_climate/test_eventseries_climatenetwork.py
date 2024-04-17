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
Tests for the EventSeriesClimateNetwork class.
"""
import numpy as np

from pyunicorn.core.data import Data
from pyunicorn.climate.eventseries_climatenetwork import\
 EventSeriesClimateNetwork


def test_str(capsys):
    data = EventSeriesClimateNetwork.SmallTestData()
    print(EventSeriesClimateNetwork(data, method='ES',
                                    threshold_method='quantile',
                                    threshold_values=0.8, taumax=16,
                                    threshold_types='above'))
    out = capsys.readouterr()[0]
    out_ref = "Extracting network adjacency matrix by thresholding...\n" + \
              "Setting area weights according to type surface ...\n" + \
              "Setting area weights according to type surface ...\n" + \
              "EventSeriesClimateNetwork:\n" + \
              "EventSeries: 6 variables, 10 timesteps, taumax: 16.0, " \
              "lag: 0.0" + \
              "\nClimateNetwork:\n" + \
              "GeoNetwork:\n" + \
              "SpatialNetwork:\n" + \
              "Network: directed, 6 nodes, 0 links, link density 0.000.\n" + \
              "Geographical boundaries:\n" + \
              "         time     lat     lon\n" + \
              "   min    0.0    0.00    2.50\n" + \
              "   max    9.0   25.00   15.00\n" + \
              "Threshold: 0\n" + \
              "Local connections filtered out: False\n" + \
              "Type of event series measure to construct " + \
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
