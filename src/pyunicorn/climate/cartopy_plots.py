#!/usr/bin/python
# -*- coding: utf-8 -*-
#
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
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.
"""

import os
import glob
# time handling
# import datetime

import numpy as np
import matplotlib.pyplot as plt

#  Import Ngl support functions for plotting, map projections etc.
try:
    import cartopy
except ImportError:
    print("climate: Package cartopy could not be loaded. Some functionality "
          "in class MapPlots might not be available!")

import cartopy.crs as ccrs
import cartopy.feature as cf

#
#  Define class CartopyPlots
#


class CartopyPlots:

    """
    Encapsulates map plotting functions via cartopy and matplotlib.

    """

    def __init__(self, grid, title):
        """
        Initialize an instance of MapPlots.

        Plotting of maps is powered by cartopy and matplotlib.

        :type grid: :class:`.Grid`
        :arg grid: The Grid object describing the map data to be plotted.
        :arg str title: The title describing the map data.
        """
        self.grid = grid
        """(Grid) - The Grid object describing the map data to be plotted."""
        self.title = title
        """(string) - The title describing the map data."""

        #  Initialize list to store data sets and titles
        self.map_data = []
        """(list) - The list storing map data and titles."""
        #  Also for multiple maps
        self.map_mult_data = []
        """(list) - The list storing map data and titles for multiple maps."""

        #
        #  Adjust cartopy settings, fine tuning can be done externally
        #

        # Specify Coordinate Refference System for Map Projection
        self.projection = ccrs.PlateCarree()

        # Specify CRS (where data should be plotted)
        self.crs = ccrs.PlateCarree()

        # get spatial dims
        self.lon = self.grid.convert_lon_coordinates(self.grid.lon_sequence())
        # self.lon = self.grid.lon_sequence()
        self.lat = self.grid.lat_sequence()
        self.gridsize_lon = len(self.lon)
        self.gridsize_lat = len(self.lat)
        self.lon_min = self.grid.boundaries()["lon_min"]
        self.lon_max = self.grid.boundaries()["lon_max"]
        self.lat_min = self.grid.boundaries()["lat_min"]
        self.lat_max = self.grid.boundaries()["lat_max"]
        
        # extent of data will also give extent of world map
        self.data_extent = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]
       
        print("Created plot class.")

    def add_dataset(self, title, data):
        """
        Add a map data set for plotting.

        Data sets are stored as dictionaries in the :attr:`map_data` list.

        :arg str title: The string describing the data set.
        :type data: 1D array [index]
        :arg data: The numpy array containing the map to be drawn
        """
        self.map_data.append({"title": title, "data": data})

    def generate_plots(self, file_name, title_on=True, labels_on=True):
        """
        Generate and map plots.

        Store the plots in the file indicated by ``file_name`` in the current
        directory.

        Map plots are stored in a PDF file, with each map occupying its own
        page.

        :arg str file_name: The name for the PDF file containing map plots.
        :arg bool title_on: Determines, whether main title is plotted.
        :arg bool labels_on: Determines whether individual map titles are
            plotted.
        """

        for dataset in self.map_data: 
            
            #  Generate figue
            fig = plt.figure()
            
            # create map plot
            ax = plt.axes(projection=self.projection)
        
            # make it a class feature, as to work with it from outside
            # self.ax = ax
            
            # create some standards of plotting that can be adjusted
            # before calling generate_cartopy_plot    
            # adjust size and plot coastlines and borders
            ax.set_extent(self.data_extent, crs=self.crs)
            # ax.set_global()
            ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
            ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2)       
            
            # Draw gridlines in degrees over map
            gl = ax.gridlines(crs=self.crs, draw_labels=True,
                          linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}                
            # plot data upon map
            ax = plt.tricontourf(self.lon, self.lat, dataset["data"], extent = self.data_extent, transform = self.crs)
                     
            ax = plt.colorbar(shrink=0.5)

            # plot main title
            if title_on:
                plt.suptitle(self.title)
            
            # plot subtitles
            if labels_on:
                plt.title(dataset["title"])
            
            # save figures at current dir
            file_extension = dataset["title"] + ".png"
            file_extension = file_extension.replace(" ", "")
            
            fig.savefig(file_name + "_" + file_extension)
            
            plt.close()            
        
        print("Created and saved plots @ current directory.")
            


