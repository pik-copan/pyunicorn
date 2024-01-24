# This file is part of pyunicorn.
# Copyright (C) 2008--2023 Jonathan F. Donges and pyunicorn authors
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
Simple test for the MapPlot class.
"""

import matplotlib.pyplot as plt

from pyunicorn.climate.climate_data import ClimateData
from pyunicorn.climate.tsonis import TsonisClimateNetwork
from pyunicorn.climate.map_plot import MapPlot


def test_map_plot():
    """
    Simple test for the MapPlot class.

    No sanity checks, only testing if it runs without errors.
    """
    # prepare ClimateNetwork fixture
    file = 'notebooks/air.mon.mean.nc'
    # select subset of data to speed up loading and calculation
    window = {
        "time_min": 0., "time_max": 0.,
        "lat_min": 30, "lon_min": 0,
        "lat_max": 50, "lon_max": 30}
    data = ClimateData.Load(
        file_name=file, observable_name="air",
        file_type="NetCDF", window=window, time_cycle=12)

    # create MapPlot
    map_plot = MapPlot(data.grid, "ncep_ncar_reanalysis")
    assert map_plot.title == "ncep_ncar_reanalysis"

    net = TsonisClimateNetwork(data, threshold=.05, winter_only=False)
    degree = net.degree()
    # plot, with suppressed display
    plt.ioff()
    map_plot.plot(degree, "Degree")
    plt.close()
