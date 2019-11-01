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
Provides class for analyzing spatially embedded complex networks.
"""

from .network import Network
from .grid2d import Grid


#
#  Define class SpatialNetwork
#

class SpatialNetwork(Network):
    """
    Encapsulates a spatially embedded network.

    Adds more network measures and statistics based on the
    spatial embedding.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, grid, adjacency=None, edge_list=None, directed=False,
                 silence_level=0):
        """
        Initialize an instance of SpatialNetwork.

        :type grid: :class:`.Grid`
        :arg grid: The Grid object describing the network's spatial embedding.
        :type adjacency: 2D array (int8) [index, index]
        :arg adjacency: The network's adjacency matrix.
        :type edge_list: array-like list of lists
        :arg  edge_list: Edge list of the new network.
                         Entries [i,0], [i,1] contain the end-nodes of an edge.
        :arg bool directed: Determines, whether the network is treated as
            directed.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        self.grid = grid
        """(Grid) - Grid object describing the network's spatial embedding"""

        #  Call constructor of parent class Network
        Network.__init__(self, adjacency=adjacency, edge_list=edge_list,
                         directed=directed, silence_level=silence_level)

    def __str__(self):
        """
        Return a string representation of the SpatialNetwork object.
        """
        return f'SpatialNetwork:\n{Network.__str__(self)}'

    def clear_cache(self):
        """
        Clean up cache.

        Is reversible, since all cached information can be recalculated from
        basic data.
        """
        Network.clear_cache(self)
        self.grid.clear_cache()

    @staticmethod
    def SmallTestNetwork():
        """
        Return a 6-node undirected geographically embedded test network.

        The test network consists of the SmallTestNetwork of the Network class
        with node coordinates given by the SmallTestGrid of the Grid class.

        The network looks like this::

                3 - 1
                |   | \\
            5 - 0 - 4 - 2

        :rtype: GeoNetwork instance
        :return: an instance of GeoNetwork for testing purposes.
        """
        return SpatialNetwork(adjacency=Network.SmallTestNetwork().adjacency,
                              grid=Grid.SmallTestGrid(),
                              directed=False, silence_level=2)
