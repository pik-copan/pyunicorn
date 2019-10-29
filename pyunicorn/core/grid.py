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
Provides class for spatio-temporal grids.
"""

#
#  Import essential packages
#

#  array object and fast numerics
import numpy as np


#
#  Define class Grid
#


class Grid:

    """
    Encapsulates a spatio-temporal grid.

    The spatial grid points can be arbitrarily distributed, which is useful
    for representing station data or geodesic grids.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, time_seq, space_seq, silence_level=0):
        """
        Initialize an instance of Grid.

        :type time_seq: 1D Numpy array [time]
        :arg time_seq: The increasing sequence of temporal sampling points.

        :type lat_seq: 2D Numpy array [dim, index]
        :arg lat_seq: The sequences of spatial sampling points.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.
        """

        #  Set basic dictionaries
        self._grid = {"time": time_seq.astype("float32"),
                      "space": space_seq.astype("float32")}
        self._grid_size = {"time": len(time_seq),
                           "space": space_seq.shape[1]}

        #  Defines the number of spatial grid points / nodes at one instant
        #  of time
        self.N = self._grid_size["space"]
        """(number (int)) - The number of spatial grid points / nodes."""

        #  Set silence level
        self.silence_level = silence_level
        """(number (int)) - The inverse level of verbosity of the object."""

        #  Defines the total number of data points / grid points / samples of
        #  the corresponding data set.
        self.n_grid_points = self._grid_size["time"] * self.N
        """(number (int)) - The total number of data points / samples."""

    def __str__(self):
        """
        Return a string representation of the Grid object.
        """
        return 'Grid: %i grid points, %i timesteps.' % (
            self._grid_size['space'], self._grid_size['time'])

    @staticmethod
    def SmallTestGrid():
        """
        Return test grid of 6 spatial grid points with 10 temporal sampling
        points each.

        :rtype: Grid2D instance
        :return: a Grid2D instance for testing purposes.
        """
        return Grid(time_seq=np.arange(10),
                    space_seq=np.array([[0, 5, 10, 15, 20, 25],
                                        [2.5, 5., 7.5, 10., 12.5, 15.]]),
                    silence_level=2)
