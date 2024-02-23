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
Provides classes for generating and analyzing complex coupled climate networks.
"""

import numpy as np

from ..core import InteractingNetworks, GeoNetwork, GeoGrid
from .climate_network import ClimateNetwork


class CoupledClimateNetwork(InteractingNetworks, ClimateNetwork):
    """
    Encapsulates a coupled similarity network embedded on a spherical surface.

    Particularly provides functionality to generate a complex network from the
    matrix of a similarity measure of time series from two different
    observables (temperature, pressure), vertical levels etc.

    So far, most methods only give meaningful results for undirected networks!

    The idea of coupled climate networks is based on the concept of coupled
    patterns, for a review refer to [Bretherton1992]_.

    .. note::
       The two observables (layers) need to have the same time grid \
       (temporal sampling points).
    """
    #
    #  Definitions of internal methods
    #

    def __init__(self, grid_1, grid_2, similarity_measure, threshold=None,
                 link_density=None, non_local=False, directed=False,
                 node_weight_type="surface", silence_level=0):
        """
        Initialize an instance of CoupledClimateNetwork.

        .. note::
           Either threshold **OR** link_density have to be given!

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos**2 lat)

        :type grid_1: :class:`.GeoGrid`
        :arg  grid_1: The GeoGrid object describing the first layer's spatial
            embedding.
        :type grid_2: :class:`.GeoGrid`
        :arg  grid_2: The GeoGrid object describing the second layer's spatial
            embedding.
        :type similarity_measure: 2D array [index, index]
        :arg similarity_measure: The similarity measure for all pairs of nodes.
        :arg float threshold: The threshold of similarity measure, above which
            two nodes are linked in the network.
        :arg float link_density: The networks's desired link density.
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg bool directed: Determines, whether the network is treated as
            directed.
        :arg strnode_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        #  Store single grids
        self.grid_1 = grid_1
        """(Grid) - The GeoGrid object describing the first layer's spatial
                    embedding."""
        self.grid_2 = grid_2
        """(Grid) - The GeoGrid object describing the second layer's spatial
                    embedding."""

        #  Construct grid object describing both layers
        time_1 = grid_1.grid()["time"]
        lat_1 = grid_1.grid()["lat"]
        lon_1 = grid_1.grid()["lon"]

        time_2 = grid_2.grid()["time"]
        lat_2 = grid_2.grid()["lat"]
        lon_2 = grid_2.grid()["lon"]

        if len(time_1) == len(time_2):
            grid = GeoGrid(time_1, np.concatenate((lat_1, lat_2)),
                           np.concatenate((lon_1, lon_2)))
            #  Set total number of nodes
            self.N = grid.N
            """(number (int)) - The total number of nodes in both layers."""

            #  Set number of nodes for both layers
            self.N_1 = len(lat_1)
            """(number (int)) - The number of nodes in the first layer."""
            self.N_2 = len(lat_2)
            """(number (int)) - The number of nodes in the second layer."""

            #  Create lists of node indices for both layers
            self.nodes_1 = list(range(self.N_1))
            """(list (int)) - List of node indices for first layer"""
            self.nodes_2 = list(range(self.N_1, self.N))
            """(list (int)) - List of node indices for second layer"""

            #  Call the constructor of the parent class ClimateNetwork
            ClimateNetwork.__init__(self, grid=grid,
                                    similarity_measure=similarity_measure,
                                    threshold=threshold,
                                    link_density=link_density,
                                    non_local=non_local,
                                    directed=directed,
                                    node_weight_type=node_weight_type,
                                    silence_level=silence_level)
            InteractingNetworks.__init__(self, self.adjacency)
        else:
            print("The two observables (layers) have to have the same number "
                  "of temporal sampling points!")

    def __str__(self):
        """
        Return a string representation of CoupledClimateNetwork object.
        """
        return (f'CoupledClimateNetwork:\n{ClimateNetwork.__str__(self)}\n'
                f'N1: {self.N_1}\nN2: self.N_2')

    #
    #  Define methods for handling the coupled network
    #

    def network_1(self):
        """
        Return network consisting of layer 1 nodes and their internal links.

        This can be used to conveniently analyze the layer 1 separately, e.g.,
        for calculation network measures solely for layer 1.

        :rtype: GeoNetwork
        :return: the network consisting of layer 1 nodes and their internal
                 links.
        """
        return GeoNetwork(adjacency=self.adjacency_1(), grid=self.grid_1,
                          directed=self.directed,
                          node_weight_type=self.node_weight_type,
                          silence_level=self.silence_level)

    def network_2(self):
        """
        Return network consisting of layer 2 nodes and their internal links.

        This can be used to conveniently analyze the layer 2 separately, e.g.,
        for calculation network measures solely for layer 2.

        :rtype: GeoNetwork
        :return: the network consisting of layer 2 nodes and their internal
                 links.
        """
        return GeoNetwork(adjacency=self.adjacency_2(), grid=self.grid_2,
                          directed=self.directed,
                          node_weight_type=self.node_weight_type,
                          silence_level=self.silence_level)

    def similarity_measure_1(self):
        """
        Return internal similarity measure matrix of first layer.

        :rtype: 2D Numpy array [index_1, index_1]
        :return: the internal similarity measure matrix of first layer.
        """
        return self.similarity_measure()[:self.N_1, :self.N_1]

    def similarity_measure_2(self):
        """
        Return internal similarity measure matrix of second layer.

        :rtype: 2D Numpy array [index_2, index_2]
        :return: the internal similarity measure matrix of first layer.
        """
        return self.similarity_measure()[self.N_1:, self.N_1:]

    def cross_similarity_measure(self):
        """
        Return cross similarity measure matrix.

        .. note::
           Cross similarity measure matrix is NEITHER square NOR \
           symmetric in general!

        :rtype: 2D Numpy array [index_1, index_2]
        :return: the cross similarity measure matrix.
        """
        return self.similarity_measure()[:self.N_1, self.N_1:]

    def adjacency_1(self):
        """
        Return internal adjacency matrix of first layer.

        :rtype: 2D Numpy array [index_1, index_1]
        :return: the internal adjacency matrix of first layer.
        """
        return self.internal_adjacency(self.nodes_1)

    def adjacency_2(self):
        """
        Return internal adjacency matrix of second layer.

        :rtype: 2D Numpy array [index_2, index_2]
        :return: the internal adjacency matrix of second layer.
        """
        return self.internal_adjacency(self.nodes_2)

    def cross_layer_adjacency(self):
        """
        Return cross adjacency matrix of the coupled network.

        The cross adjacency matrix entry :math:`CA_{ij} = 1` describes that
        node :math:`i` in the first layer is linked to node :math:`j` in the
        second layer.  Vice versa, :math:`CA_{ji} = 1` indicates that node
        :math:`j` in the first layer is linked to node :math:`i` in the second
        layer.

        .. note::
           Cross adjacency matrix is **NEITHER** square **NOR** symmetric in
           general!

        :rtype: 2D Numpy array [index_1, index_2]
        :return: the cross adjacency matrix.
        """
        return self.cross_adjacency(node_list1=self.nodes_1,
                                    node_list2=self.nodes_2)

    def path_lengths_1(self, link_attribute=None):
        """
        Return internal path length matrix of first layer.

        Contains the paths length between all pairs of nodes within layer 1.
        However, the paths themselves will generally contain nodes from both
        layers. To avoid this and only consider paths lying within layer 1,
        do the following::

            net_1 = coupled_network.network_1()
            path_lengths_1 = net_1.path_lengths(link_attribute)

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: 2D array [index_1, index_1]
        :return: the internal path length matrix of first layer.
        """
        return self.internal_path_lengths(node_list=self.nodes_1,
                                          link_attribute=link_attribute)

    def path_lengths_2(self, link_attribute=None):
        """
        Return internal path length matrix of second layer.

        Contains the path lengths between all pairs of nodes within layer 2.
        However, the paths themselves will generally contain nodes from both
        layers. To avoid this and only consider paths lying within layer 2,
        do the following::

            net_2 = coupled_network.network_2()
            path_lengths_2 = net_2.path_lengths(link_attribute)

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: 2D array [index_2, index_2]
        :return: the internal path length matrix of second layer.
        """
        return self.internal_path_lengths(node_list=self.nodes_2,
                                          link_attribute=link_attribute)

    def cross_path_lengths(self, link_attribute=None):
        """
        Return cross path length matrix.

        Contains the path length between nodes from different layers. The paths
        contain nodes from both layers.

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: 2D array [index_1, index_2]
        :return: the cross path length matrix.
        """
        return InteractingNetworks.\
            cross_path_lengths(self, node_list1=self.nodes_1,
                               node_list2=self.nodes_2,
                               link_attribute=link_attribute)

    def cross_link_distance(self):
        """
        Return cross link distance matrix.

        Contains the distance between nodes from different layers.

        :rtype: 2D array [index_1, index_2]
        :return: the cross link distance matrix.
        """
        return self.distance()[self.nodes_1, :][:, self.nodes_2]

    #
    #  Define scalar coupled network statistics
    #

    def number_cross_layer_links(self):
        """
        Return the number of links between the two layers.

        :return int: the number of links between nodes from different layers.
        """
        return self.number_cross_links(node_list1=self.nodes_1,
                                       node_list2=self.nodes_2)

    def number_internal_links(self):
        """
        Return the number of links within each layer.

        :rtype: (int, int)
        :return: the number of links within each layer.
        """
        n_links_1 = InteractingNetworks.number_internal_links(
            self, self.nodes_1)
        n_links_2 = InteractingNetworks.number_internal_links(
            self, self.nodes_2)

        return (n_links_1, n_links_2)

    def cross_link_density(self):
        """
        Return the density of links between the two layers.

        :return float: the density of links between the two layers.
        """
        return InteractingNetworks.cross_link_density(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)

    def internal_link_density(self):
        """
        Return the density of links within the two layers.

        :rtype: (float, float)
        :return: the density of links within the two layers.
        """
        density_1 = InteractingNetworks.\
            internal_link_density(self, self.nodes_1)
        density_2 = InteractingNetworks.\
            internal_link_density(self, self.nodes_2)

        return (density_1, density_2)

    def internal_global_clustering(self):
        """
        Return global clustering coefficients for each layer separately.

        Internal global clustering coefficients are calculated as mean values
        from the local clustering sequence of the whole coupled network. This
        implies that triangles spanning both layers will generally contribute
        to the internal clustering coefficients.

        To avoid this and consider only triangles lying within each layer::

            net_1 = coupled_network.network_1()
            clustering_1 = net_1.global_clustering()
            net_2 = coupled_network.network_2()
            clustering_2 = net_2.global_clustering()

        :rtype: (float, float)
        :return: the internal global clustering coefficients.
        """
        clustering_1 = InteractingNetworks.\
            internal_global_clustering(self, self.nodes_1)
        clustering_2 = InteractingNetworks.\
            internal_global_clustering(self, self.nodes_2)

        return (clustering_1, clustering_2)

    def cross_global_clustering(self):
        """
        Return global cross clustering for coupled network.

        The global cross clustering coefficient C_v gives the average
        probability, that two randomly drawn neighbors in layer 2 of node v in
        layer 1 are also neighbors and vice versa. It counts triangles having
        one vertex in layer 1 and two vertices in layer 2 and vice versa.

        :rtype: (float, float)
        :return: the cross global clustering coefficients.
        """
        cc_12 = InteractingNetworks.cross_global_clustering(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)
        cc_21 = InteractingNetworks.cross_global_clustering(
            self, node_list1=self.nodes_2, node_list2=self.nodes_1)

        return (cc_12, cc_21)

    def cross_transitivity(self):
        """
        Return cross transitivity for coupled network.

        The cross transitivity is the probability, that
        two randomly drawn neighbors in layer 2 of node v in layer 1 are also
        neighbors and vice versa. It counts triangles having one vertex in
        layer 1 and two vertices in layer 2 and vice versa. Cross transitivity
        tends to weight low cross degree vertices less strongly when compared
        to the global cross clustering coefficient (see [Newman2003]_).

        :rtype: (float, float)
        :return: the cross transitivities.
        """
        ct_12 = InteractingNetworks.cross_transitivity(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)
        ct_21 = InteractingNetworks.cross_transitivity(
            self, node_list1=self.nodes_2, node_list2=self.nodes_1)
        return (ct_12, ct_21)

    def cross_average_link_distance(self, reverse=False):
        """
        Return the cross average link distance

        The cross average link distance is the average link distance of each
        node of the first subnetwork to the nodes of the second subnetwork
        it is connected to. If reverse is set to True, the method calculates
        the average link distance of each node of the second subnetwork to the
        nodes of the first subnetwork.

        :arg bool reverse: Replace the subnetworks.

        :rtype: 1D Numpy array
        :return: the cross average link distances
        """
        if reverse:
            ax = 0
        else:
            ax = 1

        adj = self.cross_layer_adjacency()
        cld = self.cross_link_distance()
        return np.sum(adj*cld, axis=ax) / np.sum(adj, axis=ax)

    def cross_average_path_length(self, link_attribute=None):
        """
        Return cross average path length.

        Return the average (weighted) shortest path length between all pairs
        of nodes from different layers only.

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :return float: the cross average path length.
        """
        return InteractingNetworks.cross_average_path_length(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2,
            link_attribute=link_attribute)

    def internal_average_path_length(self, link_attribute=None):
        """
        Return internal average path length.

        Return the average (weighted) shortest path length between all pairs
        of nodes within each layer separately for which a path exists. Paths
        between nodes from different layers are not included in the averages!

        However, even if the end points lie within the same layer, the paths
        themselves will generally contain nodes from both layers. To avoid
        this and only consider paths lying within layer i, do the following::

            net_i = coupled_network.network_i()
            path_lengths_i = net_i.path_lengths(link_attribute)

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: (float, float)
        :return: the internal average path length.
        """
        apl_1 = InteractingNetworks.internal_average_path_length(
            self, node_list=self.nodes_1, link_attribute=link_attribute)
        apl_2 = InteractingNetworks.internal_average_path_length(
            self, node_list=self.nodes_2, link_attribute=link_attribute)

        return (apl_1, apl_2)

    #
    #  Define local coupled network measures
    #

    def cross_degree(self):
        """
        Return the cross degree sequences.

        Gives the number of links from a specific node in one layer to the
        other layer.

        :rtype: tuple of two 1D arrays [index]
        :return: the cross degree sequences.
        """
        cross_degree_1 = InteractingNetworks.cross_degree(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)
        cross_degree_2 = InteractingNetworks.cross_degree(
            self, node_list1=self.nodes_2, node_list2=self.nodes_1)
        return (cross_degree_1, cross_degree_2)

    def internal_degree(self):
        """
        Return the internal degree sequences.

        Gives the number of links from a specific node to other nodes within
        the same layer.

        :rtype: tuple of two 1D arrays [index]
        :return: the internal degree sequences.
        """
        degree_1 = InteractingNetworks.internal_degree(
            self, node_list=self.nodes_1)
        degree_2 = InteractingNetworks.internal_degree(
            self, node_list=self.nodes_2)
        return (degree_1, degree_2)

    def cross_local_clustering(self):
        """
        Return local cross clustering for coupled network.

        The local cross clustering coefficient C_v gives the probability, that
        two randomly drawn neighbors in layer 2 of node v in layer 1 are also
        neighbors and vice versa. It counts triangles having one vertex in
        layer 1 and two vertices in layer 2 and vice versa.

        :rtype: tuple of two 1D arrays [index]
        :return: the cross local clustering coefficients.
        """
        cc_12 = InteractingNetworks.cross_local_clustering(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)
        cc_21 = InteractingNetworks.cross_local_clustering(
            self, node_list1=self.nodes_2, node_list2=self.nodes_1)
        return (cc_12, cc_21)

    def cross_closeness(self, link_attribute=None):
        """
        Return cross closeness sequence.

        Gives the inverse average geodesic distance from a node in one layer
        to all nodes in the other layer.

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: tuple of two 1D arrays [index]
        :return: the cross closeness sequence.
        """
        cc_12 = InteractingNetworks.cross_closeness(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2,
            link_attribute=link_attribute)
        cc_21 = InteractingNetworks.cross_closeness(
            self, node_list1=self.nodes_2, node_list2=self.nodes_1,
            link_attribute=link_attribute)
        return (cc_12, cc_21)

    def internal_closeness(self, link_attribute=None):
        """
        Return internal closeness sequence.

        Gives the inverse average geodesic distance from a node to all other
        nodes in the same layer.

        However, the included paths will generally contain nodes from both
        layers. To avoid this, do the following::

            net_i = coupled_network.network_i()
            closeness_i = net_i.closeness(link_attribute)

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: tuple of two 1D arrays [index]
        :return: the internal closeness sequence.
        """
        closeness_1 = InteractingNetworks.internal_closeness(
            self, node_list=self.nodes_1, link_attribute=link_attribute)
        closeness_2 = InteractingNetworks.internal_closeness(
            self, node_list=self.nodes_2, link_attribute=link_attribute)
        return (closeness_1, closeness_2)

    def cross_betweenness(self):
        """
        Return the cross betweenness sequence.

        Gives the normalized number of shortest paths only between nodes from
        **different** layers, in which a node :math:`i` is contained. This is
        equivalent to the inter-regional / inter-group betweenness with respect
        to layer 1 and layer 2.

        :rtype: tuple of two 1D arrays [index]
        :return: the cross betweenness sequence.
        """
        cb = InteractingNetworks.cross_betweenness(
            self, node_list1=self.nodes_1, node_list2=self.nodes_2)
        return (cb[self.nodes_1], cb[self.nodes_2])

    def internal_betweenness_1(self):
        """
        Return the internal betweenness sequences for layer 1.

        Gives the normalized number of shortest paths only between nodes from
        layer 1, in which a node :math:`i` is contained. :math:`i` can be
        member of any of the two layers. This is equivalent to the
        inter-regional / inter-group betweenness with respect to layer 1 and
        layer 1.

        :rtype: tuple of two 1D arrays [index]
        :return: the internal betweenness sequence for layer 1.
        """
        ib = self.internal_betweenness(self.nodes_1)

        return (ib[self.nodes_1], ib[self.nodes_2])

    def internal_betweenness_2(self):
        """
        Return the internal betweenness sequences for layer 2.

        Gives the normalized number of shortest paths only between nodes from
        layer 2, in which a node :math:`i` is contained. :math:`i` can be
        member of any of the two layers. This is equivalent to the
        inter-regional / inter-group betweenness with respect to layer 2 and
        layer 2.

        :rtype: tuple of two 1D arrays [index]
        :return: the internal betweenness sequence for layer 2.
        """
        ib = self.internal_betweenness(self.nodes_2)

        return (ib[self.nodes_1], ib[self.nodes_2])
