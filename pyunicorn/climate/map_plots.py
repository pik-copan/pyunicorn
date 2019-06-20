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
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.
"""

import os
import glob
# time handling
# import datetime

import numpy as np

#  Import Ngl support functions for plotting, map projections etc.
try:
    import Ngl
except ImportError:
    print("climate: Package Ngl could not be loaded. Some functionality "
          "in class MapPlots might not be available!")

#
#  Define class MapPlots
#


class MapPlots:

    """
    Encapsulates map plotting functions.

    Provides functionality to easily bundle multiple geo-datasets
    into a single file.
    """

    def __init__(self, grid, title):
        """
        Initialize an instance of MapPlots.

        Plotting of maps is powered by PyNGL.

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
        #  Adjust PyNGL settings, fine tuning can be done externally
        #

        #  Set PyNGL resources
        resources = Ngl.Resources()

        #  Define grid
        resources.sfXArray = self.grid.lon_sequence()
        resources.sfYArray = self.grid.lat_sequence()

        #  Change the map projection
        resources.mpProjection = "Robinson"

        #  Rotate the projection
        #  Center in the middle of lonMin and lonMax
        # resources.mpRelativeCenterLon = "true"

        #  Set plot limits
        resources.mpLimitMode = "LatLon"
        resources.mpMinLonF = self.grid.boundaries()["lon_min"]
        resources.mpMaxLonF = self.grid.boundaries()["lon_max"]
        resources.mpMinLatF = self.grid.boundaries()["lat_min"]
        resources.mpMaxLatF = self.grid.boundaries()["lat_max"]

        #  Change the color map
        resources.wkColorMap = "wh-bl-gr-ye-re"

        #  Change thickness of geophysical lines
        resources.mpGeophysicalLineThicknessF = 2.0

        #  Configure the legend
        resources.lbAutoManage = False
        resources.lbOrientation = "Horizontal"
        resources.lbLabelFont = "Helvetica"
        resources.lbLabelFontHeightF = 0.0075
        resources.lbTitleFontHeightF = 0.01

        #  Larger font for regional networks
        # resources.lbLabelFontHeightF = 0.014
        # resources.lbTitleFontHeightF = 0.02

        #  Configure the contour plots
        resources.cnFillOn = True
        resources.cnLinesOn = False
        resources.cnLineLabelsOn = False
        resources.cnInfoLabelOn = False
        resources.cnMaxLevelCount = 22

        resources.cnFillMode = "RasterFill"

        #  Make resources object accessible from outside
        self.resources = resources
        """The PyNGL resources allow fine tuning of plotting options."""

    def add_dataset(self, title, data):
        """
        Add a map data set for plotting.

        Data sets are stored as dictionaries in the :attr:`map_data` list.

        :arg str title: The string describing the data set.
        :type data: 1D array [index]
        :arg data: The numpy array containing the map to be drawn
        """
        self.map_data.append({"title": title, "data": data})

    def generate_map_plots(self, file_name, title_on=True, labels_on=True):
        """
        Generate and save map plots.

        Store the plots in the file indicated by ``file_name`` in the current
        directory.

        Map plots are stored in a PDF file, with each map occupying its own
        page.

        :arg str file_name: The name for the PDF file containing map plots.
        :arg bool title_on: Determines, whether main title is plotted.
        :arg bool labels_on: Determines whether individual map titles are
            plotted.
        """
        #  Set resources
        resources = self.resources

        #  Set plot title
        if title_on:
            resources.tiMainString = self.title

        #  Open a workstation, display in X11 window
        #  Alternatively wks_type = "ps", wks_type = "pdf" or wks_type = "x11"
        wks_type = "pdf"
        wks = Ngl.open_wks(wks_type, file_name, resources)

        #
        #  Generate map plots
        #
        for dataset in self.map_data:
            #  Set title
            if labels_on:
                resources.lbTitleString = dataset["title"]

            #  Generate map plot
            cmap = Ngl.contour_map(wks, dataset["data"], resources)

        #  Clean up
        del cmap
        del resources

        Ngl.end()

    #  FIXME: Clean this up (Jakob)
    def add_multiple_datasets(self, map_number, title, data):
        """
        Add a map-dataset consisting of a title and the dataset itself
        to the :attr:`map_data` list of dictionaries (pure dictionaries have no
        order) and reshapes data array for plotting.

        INPUT: title a string describing the dataset
               data a numpy array containing the map to be drawn
        """
        if map_number > len(self.map_mult_data) - 1:
            self.map_mult_data.append([])
        self.map_mult_data[map_number].append({"title": title, "data": data})

    #  FIXME: Clean this up (Jakob)
    def generate_multiple_map_plots(self, map_names, map_scales, title_on=True,
                                    labels_on=True):
        """
        Generate map plots from the datasets stored in the :attr:`map_data`
        list of dictionaries. Stores the plots in the file indicated by
        filename in the current directory.
        """
        for k in range(len(self.map_mult_data)):
            #  Set resources
            resources = self.resources

            #  Set plot title
            if title_on:
                resources.tiMainString = self.title

            #  Open a workstation for every map, only wks_type = "ps" allows
            #  multiple workstations

            # Define own levels
            resources.cnLevelSelectionMode = "ExplicitLevels"
            resources.cnLevels = map_scales[k]

            wks_type = "pdf"
            wks = Ngl.open_wks(wks_type, map_names[k], resources)

            #
            #  Generate map plots
            #

            for dataset in self.map_mult_data[k]:
                #  Set title
                if labels_on:
                    resources.lbTitleString = dataset["title"]

                #  Reshape for visualization on the sphere
                dataset["data"].shape = (self.grid.grid_size()["lat"],
                                         self.grid.grid_size()["lon"])

                #  Generate map plot
                cmap = Ngl.contour_map(wks, dataset["data"], resources)

            # Clear map
            del cmap
            Ngl.destroy(wks)

        #  Clean up
        del resources

        Ngl.end()

    #  FIXME: Clean this up (Jakob)
    def add_multiple_datasets_npy(self, map_number, title, data):
        """
        Method for very large data sets (RAM issues) and useful for PARALLEL
        code. Data is copied to npy files (titles still in the list) that
        can be loaded afterwards.

        INPUT: title a string describing the data set
               data a Numpy array containing the map to be drawn
        """
        if map_number > len(self.map_mult_data) - 1:
            self.map_mult_data.append([])
        self.map_mult_data[map_number].append(title)

        np.save(str(map_number) + "_" + title, data)

    #  FIXME: Clean this up (Jakob)
    def generate_multiple_map_plots_npy(self, map_names, map_scales,
                                        title_on=True, labels_on=True):
        """
        Method for very large datasets (RAM issues) and useful for PARALLEL
        code. Generates map plots from the datasets stored in the npy files
        and the list of titles. The data is sorted as parallel computation
        mixes it up. Stores the plots in the file indicated by filename in the
        current directory.
        """
        #  Set resources
        resources = self.resources

        #  Set plot title
        if title_on:
            resources.tiMainString = self.title

        for k in range(len(self.map_mult_data)):
            #  Open a workstation for every map, only wks_type = "ps" allows
            #  multiple workstation

            #  Sort dataset, as parallel code will mix it
            self.map_mult_data[k].sort()

            # Define own levels
            resources.cnLevelSelectionMode = "ExplicitLevels"
            resources.cnLevels = map_scales[k]

            wks_type = "pdf"
            wks = Ngl.open_wks(wks_type, map_names[k], resources)

            #
            #  Generate map plots
            #
            for ititle in self.map_mult_data[k]:
                #  Set title
                if labels_on:
                    resources.lbTitleString = ititle

                data = np.load(str(k) + "_" + ititle + ".npy")
                #  Reshape for visualization on the sphere
                data.shape = (self.grid.grid_size()["lat"],
                              self.grid.grid_size()["lon"])

                #  Generate map plot
                cmap = Ngl.contour_map(wks, data, resources)

            # Clear map
            del cmap
            Ngl.destroy(wks)

        #  Clean up
        for file_name in glob.glob('*.npy'):
            os.remove(file_name)
        del resources

        Ngl.end()

    #  FIXME: Possibly bogus method?
    def save_ps_map(self, title, data, labels_on=True):
        """
        Directly create a PS file of data with filename=title.
        Assumes normalized data between 0 and 1.

        INPUT: title a string describing the dataset data a numpy array
               containing the map to be drawn
        """
        #  Change the levels of contouring
        resources = self.resources
        resources.cnLevelSelectionMode = "ExplicitLevels"  # Define own levels.
        resources.cnLevels = np.arange(0., 1., 0.05)

        wks_type = "ps"
        wks = Ngl.open_wks(wks_type, title, resources)

        if labels_on:
            resources.lbTitleString = title

        #  Reshape for visualization on the sphere
        data.shape = (self.grid.grid_size()["lat"],
                      self.grid.grid_size()["lon"])

        #  Generate map plot
        cmap = Ngl.contour_map(wks, data, resources)

        # Clear map
        del cmap
        Ngl.destroy(wks)

        #  Clean up
        del resources

        Ngl.end()
