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
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cf
except ImportError:
    print("climate: Package cartopy could not be loaded. Some functionality "
          "in class MapPlot might not be available!")

from ..core import Grid

#
#  Define class MapPlot
#


# pylint: disable=too-few-public-methods
class MapPlot:
    """
    Encapsulates map plotting functions via Cartopy and Matplotlib.
    """

    def __init__(self, grid: Grid, title: str):
        """
        :arg grid: The `Grid` object describing the map data to be plotted.
        :arg str title: The title describing the map data.
        """
        self.grid: Grid = grid
        self.title: str = title

        #
        #  Adjust Cartopy settings, fine tuning can be done externally
        #

        # Specify Coordinate Refference System for Map Projection
        # pylint: disable-next=abstract-class-instantiated
        self.projection = ccrs.PlateCarree()

        # Specify CRS (where data should be plotted)
        # pylint: disable-next=abstract-class-instantiated
        self.crs = ccrs.PlateCarree()

        # get spatial dims
        self.lon = self.grid.convert_lon_coordinates(self.grid.lon_sequence())
        self.lat = self.grid.lat_sequence()
        self.gridsize_lon = len(self.lon)
        self.gridsize_lat = len(self.lat)
        self.lon_min = self.grid.boundaries()["lon_min"]
        self.lon_max = self.grid.boundaries()["lon_max"]
        self.lat_min = self.grid.boundaries()["lat_min"]
        self.lat_max = self.grid.boundaries()["lat_max"]

        # extent of data will also give extent of world map
        self.data_extent = [self.lon_min, self.lon_max,
                            self.lat_min, self.lat_max]

    def plot(self, data: np.ndarray, label: str):
        """
        Plot dataset onto ``self.grid``. A simple setup to get a quick view of
        your data.

        The plot can be customized by calling additional Matplotlib or Cartopy
        methods afterwards. It can then be saved via ``plt.savefig()``.

        :arg ndarray data: The dataset to be plotted on the Grid.
        :arg str label: A name for the dataset to print as label.
        """

        #  Generate figure
        plt.figure()

        # create GeoAxes object
        gax = plt.axes(projection=self.projection)

        # create some standards of plotting that can be adjusted
        # before calling ``generate_cartopy_plot()``
        # adjust size and plot coastlines and borders
        gax.set_extent(self.data_extent, crs=self.crs)
        # ax.set_global()
        gax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
        gax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2)

        # Draw gridlines in degrees over map
        gl = gax.gridlines(
            crs=self.crs, draw_labels=True,
            linewidth=.6, color='gray',
            alpha=0.5, linestyle='-.'
        )
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        # plot data upon map
        plt.tricontourf(self.lon, self.lat, data,
                        extent=self.data_extent, transform=self.crs)
        cbar = plt.colorbar(shrink=0.5)
        cbar.set_label(label, rotation=270)

        # add title
        plt.title(self.title)
