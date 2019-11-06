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

# array object and fast numerics
import numpy as np
# random number generation
from numpy import random

from ._ext.numerics import _randomly_rewire_geomodel_I, \
        _randomly_rewire_geomodel_II, _randomly_rewire_geomodel_III

from .network import Network
from .grid import Grid


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

    @staticmethod
    def Model(network_model, grid, **kwargs):
        """
        Return a new model graph generated with the specified network model
        and embedded on the specified spatial grid
        """
        A = getattr(SpatialNetwork, network_model)(**kwargs)
        return SpatialNetwork(grid, A)

    #
    #  Graph randomization methods
    #

    #  TODO: Experimental code!
    def randomly_rewire_geomodel_I(self, distance_matrix, iterations,
                                   inaccuracy):
        """
        Randomly rewire the current network in place using geographical
        model I.

        Geographical model I preserves the degree sequence (exactly) and the
        link distance distribution :math:`p(l)` (approximately).

        A higher ``inaccuracy`` in the conservation of :math:`p(l)` will lead
        to

          - less deterministic links in the network and, hence,
          - more degrees of freedom for the random graph and
          - a shorter runtime of the algorithm, since more pairs of nodes
            eligible for rewiring can be found.

        **Example** (The degree sequence should be the same after rewiring):

        >>> _i()
        >>> net = SpatialNetwork.SmallTestNetwork()
        >>> net.randomly_rewire_geomodel_I(
        ...     distance_matrix=net.grid.distance(),
        ...     iterations=100, inaccuracy=100)
        #
        >>> net.degree()
        array([3, 3, 2, 2, 3, 1])

        :type distance_matrix: 2D Numpy array [index, index]
        :arg distance_matrix: Suitable distance matrix between nodes.

        :type iterations: number (int)
        :arg iterations: The number of rewirings to be performed.

        :type inaccuracy: number (float)
        :arg inaccuracy: The inaccuracy with which to conserve :math:`p(l)`.
        """
        if self.silence_level <= 1:
            print("Randomly rewiring given graph, preserving the degree "
                  "sequence and link distance distribution...")
        #  Get number of nodes
        N = self.N
        #  Get number of links
        E = self.n_links
        #  Collect adjacency and distance matrices
        A = self.adjacency.copy(order='c')
        D = distance_matrix.astype("float32").copy(order='c')
        #  Get degree sequence
        # degree = self.degree()

        #  Define for brevity
        eps = float(inaccuracy)

        # iterations = int(iterations)

        #  Get edge list
        edges = np.array(self.graph.get_edgelist()).copy(order='c')

        #  Initialize list of neighbors
        # neighbors = np.zeros((N, degree.max()))

        _randomly_rewire_geomodel_I(iterations, eps, A, D, E, N, edges)

        #  Update all other properties of GeoNetwork
        self.adjacency = A

    #  TODO: Experimental code!
    def randomly_rewire_geomodel_II(self, distance_matrix,
                                    iterations, inaccuracy):
        """
        Randomly rewire the current network in place using geographical
        model II.

        Geographical model II preserves the degree sequence :math:`k_v`
        (exactly), the link distance distribution :math:`p(l)` (approximately),
        and the average link distance sequence :math:`<l>_v` (approximately).

        A higher ``inaccuracy`` in the conservation of :math:`p(l)` and
        :math:`<l>_v` will lead to:

          - less deterministic links in the network and, hence,
          - more degrees of freedom for the random graph and
          - a shorter runtime of the algorithm, since more pairs of nodes
            eligible for rewiring can be found.

        :type distance_matrix: 2D Numpy array [index, index]
        :arg distance_matrix: Suitable distance matrix between nodes.

        :type iterations: number (int)
        :arg iterations: The number of rewirings to be performed.

        :type inaccuracy: number (float)
        :arg inaccuracy: The inaccuracy with which to conserve :math:`p(l)`.
        """
        #  FIXME: Add example
        if self.silence_level <= 1:
            print("Randomly rewiring given graph, preserving the degree "
                  "sequence, link distance distribution and average link "
                  "distance sequence...")

        #  Get number of nodes
        N = int(self.N)
        #  Get number of links
        E = int(self.n_links)
        #  Collect adjacency and distance matrices
        A = self.adjacency.copy(order='c')
        D = distance_matrix.astype("float32").copy(order='c')

        #  Define for brevity
        eps = float(inaccuracy)

        #  Get edge list
        edges = np.array(self.graph.get_edgelist()).copy(order='c')

        _randomly_rewire_geomodel_II(iterations, eps, A, D, E, N, edges)

        #  Set new adjacency matrix
        self.adjacency = A

    #  TODO: Experimental code!
    def randomly_rewire_geomodel_III(self, distance_matrix,
                                     iterations, inaccuracy):
        """
        Randomly rewire the current network in place using geographical
        model III.

        Geographical model III preserves the degree sequence :math:`k_v`
        (exactly), the link distance distribution :math:`p(l)` (approximately),
        and the average link distance sequence :math:`<l>_v` (approximately).
        Moreover, degree-degree correlations are also conserved exactly.

        A higher ``inaccuracy`` in the conservation of :math:`p(l)` and
        :math:`<l>_v` will lead to:

          - less deterministic links in the network and, hence,
          - more degrees of freedom for the random graph and
          - a shorter runtime of the algorithm, since more pairs of nodes
            eligible for rewiring can be found.

        :type distance_matrix: 2D Numpy array [index, index]
        :arg distance_matrix: Suitable distance matrix between nodes.

        :type iterations: number (int)
        :arg iterations: The number of rewirings to be performed.

        :type inaccuracy: number (float)
        :arg inaccuracy: The inaccuracy with which to conserve :math:`p(l)`.
        """
        #  FIXME: Add example
        if self.silence_level <= 1:
            print("Randomly rewiring given graph, preserving the degree "
                  "sequence, degree-degree correlations, link distance "
                  "distribution and average link distance sequence...")

        #  Get number of nodes
        N = int(self.N)
        #  Get number of links
        E = int(self.n_links)
        #  Collect adjacency and distance matrices
        A = self.adjacency.copy(order='c')
        D = distance_matrix.astype("float32").copy(order='c')
        #  Get degree sequence
        degree = self.degree().copy(order='c')

        #  Define for brevity
        eps = float(inaccuracy)

        #  Get edge list
        edges = np.array(self.graph.get_edgelist()).copy(order='c')

        _randomly_rewire_geomodel_III(iterations, eps, A, D, E, N, edges,
                                      degree)

        #  Set new adjacency matrix
        self.adjacency = A

    def set_random_links_by_distance(self, a, b):
        """
        Reassign links independently with
        link probability = :math:`exp(a + b*angular distance)`.

        .. note::
           Modifies network in place, creates an undirected network!

        **Example** (Repeat until a network with 5 links is created):

        >>> net = GeoNetwork.SmallTestNetwork()
        >>> while (net.n_links != 5):
        ...     net.set_random_links_by_distance(a=0., b=-4.)
        >>> net.n_links
        5

        :type a: number (float)
        :arg a: The a parameter.

        :type b: number (float)
        :arg b: The b parameter.
        """
        #  Get angular distance matrix
        D = self.grid.distance()
        #  Calculate link probabilities
        p = np.exp(a + b * D)

        #  Generate random numbers
        P = random.random(D.shape)
        #  Symmetrize
        P = 0.5 * (P + P.transpose())

        #  Create new adjacency matrix
        A = (p >= P).astype("int8")
        #  Set diagonal to zero - no self-loops!
        np.fill_diagonal(A, 0)

        #  Set new adjacency matrix
        self.adjacency = A

    #
    #  Get link distance distribution
    #

    def link_distance_distribution(self, n_bins, grid_type="euclidean",
                                   geometry_corrected=False):
        """
        Return the normalized link distance distribution.

        Correct for the geometry of the embedding space by default.

        **Examples:**

        >>> GeoNetwork.SmallTestNetwork().link_distance_distribution(
        ...     n_bins=4, geometry_corrected=False)[0]
        array([ 0.14285714,  0.28571429,  0.28571429,  0.28571429])
        >>> GeoNetwork.SmallTestNetwork().link_distance_distribution(
        ...     n_bins=4, geometry_corrected=True)[0]
        array([ 0.09836066,  0.24590164,  0.32786885,  0.32786885])

        :arg int n_bins: The number of bins for histogram.
        :arg str grid_type: Type of grid, used for distance calculation, can
            take values "euclidean" and "spherical" (only for GeoNetwork).
        :arg bool geometry_corrected: Toggles correction for grid geometry.
        :rtype: tuple of three 1D arrays [bin]
        :return: the link distance distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating link distance distribution...")

        #  Collect matrices
        A = self.adjacency
        if grid_type == "spherical":
            if self.grid.__class__.__name__ != "GeoGrid":
                raise NotImplementedError("Spherical coordinates are only "
                                          "supported for GeoGrid!")
            D = self.grid.angular_distance()
        elif grid_type == "euclidean":
            D = self.grid.euclidean_distance()
        else:
            raise ValueError("Grid type unknown!")

        #  Determine range for link distance histograms
        interval = (0, D.max())

        #  Get link distance distribution
        (dist, error, lbb) = self._histogram(D[A == 1], n_bins=n_bins,
                                             interval=interval)

        if geometry_corrected:
            geometric_ld_dist = \
                self.grid.geometric_distance_distribution(n_bins)[0]
            # Divide out the geometrical factor of the distribution
            dist /= geometric_ld_dist

        #  Normalize the distribution
        dist /= dist.sum()

        return (dist, error, lbb)
