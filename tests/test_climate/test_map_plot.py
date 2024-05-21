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

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from pyunicorn.climate.climate_data import ClimateData
from pyunicorn.climate.tsonis import TsonisClimateNetwork
from pyunicorn.climate.map_plot import MapPlot

matplotlib.use('Agg')


# pylint: disable=too-few-public-methods
class TestMapPlot:
    """
    Simple tests for the `MapPlot` class.
    """

    @staticmethod
    def test_plot(reanalysis_data: Path):
        """
        Check error-free execution.
        """
        # prepare ClimateNetwork fixture
        # (select subset of data to speed up loading and calculation)
        title = "ncep_ncar_reanalysis"
        window = {
            "time_min": 0., "time_max": 0.,
            "lat_min": 30, "lon_min": 0,
            "lat_max": 50, "lon_max": 30}
        data = ClimateData.Load(
            file_name=str(reanalysis_data), observable_name="air",
            file_type="NetCDF", window=window, time_cycle=12)
        net = TsonisClimateNetwork(data, threshold=.05, winter_only=False)

        # create MapPlot
        map_plot = MapPlot(data.grid, title)
        assert map_plot.title == title

        # plot
        map_plot.plot(net.degree(), "Degree")
        assert plt.gca().get_title() == title
        plt.close()
