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

# array object and fast numerics
import numpy as np
# random number generation
from numpy import random
# high performance graph theory tools written in pure ANSI-C
import igraph

from ._ext.numerics import _randomly_rewire_geomodel_I, \
        _randomly_rewire_geomodel_II, _randomly_rewire_geomodel_III

from .network import Network, cached_const
from .grid import Grid


#
#  Define class ClimateNetwork
#

class GeoNetwork(Network):

    """
    Encapsulates a network embedded on a spherical surface.

    Particularly adds more network measures and statistics based on the
    spatial embedding.

    :ivar node_weight_type: (string) - The type of geographical node weight to
                            be used.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, grid, adjacency=None, edge_list=None, directed=False,
                 node_weight_type="surface", silence_level=0):
        """
        Initialize an instance of GeoNetwork.

        :type grid: :class:`.Grid`
        :arg grid: The Grid object describing the network's spatial embedding.
        :type adjacency: 2D array (int8) [index, index]
        :arg adjacency: The network's adjacency matrix.
        :type edge_list: array-like list of lists
        :arg  edge_list: Edge list of the new network.
                         Entries [i,0], [i,1] contain the end-nodes of an edge.
        :arg bool directed: Determines, whether the network is treated as
            directed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos² lat)
        """
        self.grid = grid
        """(Grid) - Grid object describing the network's spatial embedding"""

        #  Call constructor of parent class Network
        Network.__init__(self, adjacency=adjacency, edge_list=edge_list,
                         directed=directed, silence_level=silence_level)

        #  Set area weights
        self.set_node_weight_type(node_weight_type)

        #  cartesian coordinates of nodes
        self.cartesian = None
        self.grid_neighbours = None
        self.grid_neighbours_set = None

    def __str__(self):
        """
        Return a string representation of the GeoNetwork object.
        """
        return (f'GeoNetwork:\n{Network.__str__(self)}\n'
                f'Geographical boundaries:\n{self.grid.print_boundaries()}')

    def clear_cache(self):
        """
        Clean up cache.

        Is reversible, since all cached information can be recalculated from
        basic data.
        """
        Network.clear_cache(self)
        self.grid.clear_cache()

    def set_node_weight_type(self, node_weight_type):
        """
        Set node weights for calculation of n.s.i. measures according to
        requested type.

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos² lat)

        :arg str node_weight_type: The type of geographical node weight to be
            used.
        """
        if self.silence_level <= 1:
            print("Setting area weights according to type "
                  f"{node_weight_type} ...")

        #  Set instance variable accordingly
        self.node_weight_type = node_weight_type

        if node_weight_type == "surface":
            self.node_weights = self.grid.cos_lat()
        elif node_weight_type == "irrigation":
            self.node_weights = np.square(self.grid.cos_lat())
        #  If None or invalid choice:
        else:
            self.node_weights = None

    #
    #  Load and save GeoNetwork object
    #

    def save(self, filename_network, filename_grid=None, fileformat=None,
             *args, **kwds):
        """
        Save the GeoNetwork object to files.

        Unified writing function for graphs. Relies on and partially extends
        the corresponding igraph function. Refer to igraph documentation for
        further details on the various writer methods for different formats.

        This method tries to identify the format of the graph given in
        the first parameter (based on extension) and calls the corresponding
        writer method.

        Existing node and link attributes/weights are also stored depending
        on the chosen file format. E.g., the formats GraphML and gzipped
        GraphML are able to store both node and link weights.

        The grid is not stored if the corresponding filename is None.

        The remaining arguments are passed to the writer method without
        any changes.

        :arg str filename_network:  The name of the file where the Network
            object is to be stored.
        :arg str filename_grid:  The name of the file where the Grid object is
            to be stored (including ending).
        :arg str fileformat: the format of the file (if one wants to override
            the format determined from the filename extension, or the filename
            itself is a stream). ``None`` means auto-detection.  Possible
            values are: ``"ncol"`` (NCOL format), ``"lgl"`` (LGL format),
            ``"graphml"``, ``"graphmlz"`` (GraphML and gzipped GraphML format),
            ``"gml"`` (GML format), ``"dot"``, ``"graphviz"`` (DOT format, used
            by GraphViz), ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"``
            (DIMACS format), ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge
            list), ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python
            pickled format), ``"svg"`` (Scalable Vector Graphics).
        """
        #  Store network
        Network.save(self, filename=filename_network, fileformat=fileformat,
                     *args, **kwds)

        #  Store grid
        if filename_grid is not None:
            self.grid.save(filename=filename_grid)

    def save_for_cgv(self, filename, fileformat="graphml"):
        """
        Save the GeoNetwork and its attributes for the CGV visualization
        software.

        The node coordinates are stored as node attributes by default, likewise
        angular link distances are stored as edge attributes by default. All
        additional node and link properties are also stored for visualization.

        This format is intended for being used by the spatial graph
        visualization software CGV developed in Rostock (contact Thomas Nocke,
        nocke@pik-potsdam.de). By default, the file includes the latitude and
        longitude vectors as node properties, as well as the geodesic angular
        distance as an link property.

        :arg str file_name: The file name should end with ".dot" or ".gml".
        :arg str fileformat: The file format: "graphml"  - GraphML format
            "graphmlz" - gzipped GraphML format
            "graphviz" - GraphViz format
        """
        #  Save node coordinates as node attribute
        self.set_node_attribute("lat", self.grid.lat_sequence())
        self.set_node_attribute("lon", self.grid.lon_sequence())

        #  Save geodesic angular distances on the sphere as link attribute
        self.set_link_attribute("ang_dist", self.grid.angular_distance())

        #  Save network, independent of filename!
        if fileformat in ["graphml", "graphmlz", "graphviz"]:
            self.graph.save(filename, format=fileformat)
        else:
            print("ERROR: the chosen format is not supported by save_for_cgv "
                  "for use with the CGV software.")

    @staticmethod
    def Load(filename_network, filename_grid, fileformat=None,
             silence_level=0, *args, **kwds):
        """
        Return a GeoNetwork object stored in files.

        Unified reading function for graphs. Relies on and partially extends
        the corresponding igraph function. Refer to igraph documentation for
        further details on the various reader methods for different formats.

        This method tries to identify the format of the graph given in
        the first parameter and calls the corresponding reader method.

        Existing node and link attributes/weights are also restored depending
        on the chosen file format. E.g., the formats GraphML and gzipped
        GraphML are able to store both node and link weights.

        The remaining arguments are passed to the reader method without
        any changes.

        :arg str filename_network:  The name of the file where the Network
            object is to be stored.
        :arg str filename_grid:  The name of the file where the Grid object is
            to be stored (including ending).
        :arg str fileformat: the format of the file (if known in advance)
          ``None`` means auto-detection. Possible values are: ``"ncol"`` (NCOL
          format), ``"lgl"`` (LGL format), ``"graphml"``, ``"graphmlz"``
          (GraphML and gzipped GraphML format), ``"gml"`` (GML format),
          ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"`` (DIMACS format),
          ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge list),
          ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python pickled
          format).
        :arg int silence_level: The inverse level of verbosity of the object.
        :rtype: GeoNetwork object
        :return: :class:`GeoNetwork` instance.
        """
        #  Load Grid object
        grid = Grid.Load(filename_grid)

        #  Load to igraph Graph object
        graph = igraph.Graph.Read(f=filename_network, format=fileformat,
                                  *args, **kwds)

        #  Extract adjacency matrix
        A = np.array(graph.get_adjacency(type=2).data)

        #  Create GeoNetwork instance
        net = GeoNetwork(adjacency=A, grid=grid, directed=graph.is_directed(),
                         node_weight_type=None)

        #  Extract node weights
        if "node_weight_nsi" in graph.vs.attribute_names():
            node_weights = \
                np.array(graph.vs.get_attribute_values("node_weight_nsi"))
            net.node_weights = node_weights

        #  Overwrite igraph Graph object in Network instance to restore link
        #  attributes/weights
        net.graph = graph
        #  Restore link attributes/weights
        net.clear_paths_cache()

        return net

    #
    #  Graph generation methods
    #

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
        return GeoNetwork(adjacency=Network.SmallTestNetwork().adjacency,
                          grid=Grid.SmallTestGrid(),
                          directed=False, node_weight_type="surface",
                          silence_level=2)

    @staticmethod
    def ErdosRenyi(grid, n_nodes, link_probability=None, n_links=None,
                   node_weight_type="surface", silence_level=0):
        """
        Generates an undirected and spatially embedded Erdos-Renyi random graph

        Any pair of nodes is connected with probability :math:`p`.

        **Example:**

        >>> print(GeoNetwork.ErdosRenyi(
        ...     grid=Grid.SmallTestGrid(), n_nodes=6, n_links=5))
        Generating Erdos-Renyi random graph with 6 nodes and 5 links...
        Setting area weights according to type surface...
        GeoNetwork:
        Network: undirected, 6 nodes, 5 links, link density 0.333.
        Geographical boundaries:
                 time     lat     lon
           min    0.0    0.00    2.50
           max    9.0   25.00   15.00

        :type grid: :class:`.Grid` object
        :arg grid: The :class:`.Grid` object describing the network's spatial
            embedding.
        :type n_nodes: number > 0 (int)
        :arg  n_nodes: Number of nodes.
        :type link_probability: number from 0 to 1 (float), or None
        :arg  link_probability: If not None, each pair of nodes is
            independently linked with this probability.  (Default: None)
        :type n_links: number > 0 (int), or None
        :arg  n_links: If not None, this many links are assigned at random.
            Must be None if link_probability is not None.  (Default: None)
        :arg str node_weight_type: The type of geographical node weight to be
            used (see :meth:`set_node_weight_type`).
        :arg int silence_level: The inverse level of verbosity of the object.
        :rtype: :class:`GeoNetwork`
        :return: the Erdos-Renyi random graph.
        """
        if link_probability is not None and n_links is None:
            if silence_level <= 1:
                print(f"Generating Erdos-Renyi random graph with {n_nodes} "
                      f"nodes and probability {link_probability}...")
            graph = igraph.Graph.Erdos_Renyi(n=n_nodes, p=link_probability)
            # type=2 corresponds to returning the full adjacency matrix
            A = np.array(graph.get_adjacency(type=2).data)

        elif link_probability is None and n_links is not None:
            if silence_level <= 1:
                print(f"Generating Erdos-Renyi random graph with {n_nodes} "
                      f"nodes and {n_links} links...")
            graph = igraph.Graph.Erdos_Renyi(n=n_nodes, m=n_links)
            # type=2 corresponds to returning the full adjacency matrix
            A = np.array(graph.get_adjacency(type=2).data)

        else:
            return None

        return GeoNetwork(adjacency=A, grid=grid, directed=False,
                          node_weight_type=node_weight_type,
                          silence_level=silence_level)

    @staticmethod
    def BarabasiAlbert(n_nodes, n_links, grid, node_weight_type="surface",
                       silence_level=0):
        """
        Generates an undirected and spatially embedded Barabasi-Albert network.

        :arg int n_nodes: The number of nodes.
        :arg int n_links: The number of links of the node that is added at each
            step of the growth process.
        :type grid: Grid object
        :arg grid: The Grid object describing the network's spatial embedding.
        :arg str node_weight_type: The type of geographical node weight to be
            used (see :meth:`set_node_weight_type`).
        :arg int silence_level: The inverse level of verbosity of the object.
        :rtype: GeoNetwork
        :return: the Barabasi-Albert network.
        """
        #  FIXME: Add example

        if silence_level <= 1:
            print("Generating Barabasi-Albert random graph "
                  f"(n = {n_nodes}, m = {n_links})...")

        graph = igraph.Graph.Barabasi(n_nodes, n_links)

        #  Remove self-loops and multiple links, this does of course change the
        #  actual degree sequence of the generated graph, but just slightly
        graph.simplify()

        #  type=2 corresponds to returning the full adjacency matrix
        A = np.array(graph.get_adjacency(type=2).data)

        network = GeoNetwork(adjacency=A, grid=grid, directed=False,
                             node_weight_type=node_weight_type,
                             silence_level=silence_level)

        return network

    @staticmethod
    def ConfigurationModel(grid, degrees, node_weight_type="surface",
                           silence_level=0):
        """
        Generates an undirected and spatially embedded configuration model
        graph.

        The configuration model gives a fully random graph with a given degree
        sequence `degrees`.

        .. note::
           The configuration model network is simplified to eliminate
           self-loops and multiple edges. This results in a model
           degree sequence differing (slightly) from the original one.
           To fully conserve the degree sequence, distribution, link density
           etc., random rewiring should be used
           (:meth:`.Network.randomly_rewire`).

        **Example** (Repeat creation of configuration model network from
        SmallTestNetwork until the number of links is the same as in the
        original network):

        >>> n = 0
        >>> while n != 7:
        ...     net = GeoNetwork.ConfigurationModel(
        ...         grid=Grid.SmallTestGrid(),
        ...         degrees=GeoNetwork.SmallTestNetwork().degree(),
        ...         silence_level=2)
        ...     n = net.n_links
        >>> print(net.link_density)
        0.4666666666666667

        :type degrees: 1D array [index]
        :arg degrees: The original degree sequence.
        :type grid: Grid object
        :arg grid: The Grid object describing the network's spatial embedding.
        :arg str node_weight_type: The type of geographical node weight to be
            used (see :meth:`set_node_weight_type`).
        :arg int silence_level: The inverse level of verbosity of the object.
        :rtype: GeoNetwork
        :return: the configuration model network.
        """
        if silence_level <= 1:
            print("Generating configuration model random graph from degree "
                  "sequence...")

        graph = igraph.Graph.Degree_Sequence(list(degrees))

        #  Remove self-loops and multiple links, this does of course change the
        #  actual degree sequence of the generated graph, but just slightly
        graph.simplify()

        #  type=2 corresponds to returning the full adjacency matrix
        A = np.array(graph.get_adjacency(type=2).data)

        network = GeoNetwork(adjacency=A, grid=grid, directed=False,
                             node_weight_type=node_weight_type,
                             silence_level=silence_level)

        return network

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
        >>> net = GeoNetwork.SmallTestNetwork()
        >>> net.randomly_rewire_geomodel_I(
        ...     distance_matrix=net.grid.angular_distance(),
        ...     iterations=100, inaccuracy=1.0)
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

        #  Update all other properties of GeoNetwork
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

        #  Update all other properties of GeoNetwork
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
        D = self.grid.angular_distance()
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

    #  FIXME: Check this method and implement in C++ via Cython for speed.
    #  FIXME: Also improve documentation.
    #  FIXME: Add example
    def shuffled_by_distance_copy(self):
        """
        Return a copy of the network where all links in each node-distance
        class have been randomly re-assigned.

        In other words, the result is a random network in which the link
        probability only depends on the nodes' distance and is the same as in
        the original network.

        :rtype: GeoNetwork
        :return: the distance shuffled copy.
        """
        N = self.N
        A = self.adjacency
        D = self.grid.angular_distance()

        #  Count pairs and links by distance
        n_pairs_by_dist = {}
        n_links_by_dist = {}
        for j in range(0, N):
            print(j)
            for i in range(0, j):
                d = D[i, j]
                try:
                    n_pairs_by_dist[d] += 1
                except KeyError:
                    n_pairs_by_dist[d] = 1
                if A[i, j]:
                    try:
                        n_links_by_dist[d] += 1
                    except KeyError:
                        n_links_by_dist[d] = 1

        #  Determine link probabilities
        p_by_dist = {}
        for d in n_pairs_by_dist:
            try:
                p_by_dist[d] = n_links_by_dist[d] * 1.0 / n_pairs_by_dist[d]
            except KeyError:
                p_by_dist[d] = 0.0
            print(d, p_by_dist[d])
        del n_links_by_dist, n_pairs_by_dist

        #  Link new pairs with respective probability
        A_new = np.zeros((N, N))
        for j in range(0, N):
            print("new ", j)
            for i in range(0, j):
                d = D[i, j]
                if p_by_dist[d] >= np.random.random():
                    A_new[i, j] = A_new[j, i] = 1
                    print(i, j, d, p_by_dist[d])

        #  Create new GeoNetwork object based on A_new
        net = GeoNetwork(adjacency=A_new, grid=self.grid,
                         directed=self.directed,
                         node_weight_type=self.node_weight_type,
                         silence_level=self.silence_level)

        return net

    #
    #  Generate a geographical distribution
    #

    #  FIXME: Derive this method from a generalized variant based on n.s.i.
    #  distributions.
    def geographical_distribution(self, sequence, n_bins):
        """
        Return a normalized geographical frequency distribution.

        Also return the estimated statistical error and lower bin boundaries.

        This function counts which percentage of total surface area falls into
        a bin and NOT which number of nodes does so.

        .. note::
           Be aware that this method only returns meaningful results
           for regular rectangular grids, where the representative area of each
           node is proportional to the cosine of its latitude.

        **Example:**

        >>> net = GeoNetwork.SmallTestNetwork()
        >>> r(net.geographical_distribution(
        ...     sequence=net.degree(), n_bins=3)[0])
        array([ 0.1565, 0.3367, 0.5068])

        :type sequence: 1D Numpy array [index]
        :arg sequence: The input sequence (e.g., some local network measure).

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the geographical distribution, statistical error, and lower
                 bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating geographical frequency distribution...")

        #  Initializations
        hist = np.zeros(n_bins)

        #  Get sequence of cosines of latitude
        cos_lat = self.grid.cos_lat()
        #  Calculate total dimensionless area of the sphere
        norm = cos_lat.sum()

        #  Get range for histogram
        range_min = float(sequence.min())
        range_max = float(sequence.max())

        #  Calculate scaling factor for histogram
        scaling = 1. / (range_max - range_min)

        #  Get array of symbols corresponding to sequence
        symbolic = \
            ((n_bins - 1) * scaling * (sequence - range_min)).astype("int")

        #  Calculate histogram
        for i in range(len(sequence)):
            hist[symbolic[i]] += cos_lat[i]

        #  Normalize histogram by the total dimensionless area
        hist /= norm

        #  Construct lower bin boundaries
        lbb = np.linspace(range_min, range_max, n_bins + 1)[:-1]

        #  Calculate statistical error given by 1/n_i per bin i,
        #  where n_i is the number of samples per bin.
        error = np.zeros(n_bins)
        error[hist != 0] = 1 / np.sqrt(hist[hist != 0])

        return (hist, error, lbb)

    def geographical_cumulative_distribution(self, sequence, n_bins):
        """
        Return a normalized geographical cumulative distribution.

        Also return estimated statistical error and the lower bin boundaries.

        This function counts which percentage of total surface area has a value
        of sequence larger or equal than the one bounded by a specific bin and
        NOT which number of nodes does so.

        .. note::
           Be aware that this method only returns meaningful results
           for regular rectangular grids, where the representative area of each
           node is proportional to the cosine of its latitude.

        **Example:**

        >>> net = GeoNetwork.SmallTestNetwork()
        >>> r(net.geographical_cumulative_distribution(
        ...     sequence=net.degree(), n_bins=3)[0])
        array([ 1. , 0.8435, 0.5068])

        :type sequence: 1D Numpy array [index]
        :arg sequence: The input sequence (e.g., some local network measure).

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the cumulative geographical distribution, statistical error,
                 and lower bin boundaries.
        """
        (dist, error, lbb) = self.geographical_distribution(sequence, n_bins)
        cumu_dist = np.zeros(n_bins)
        for i in range(n_bins):
            cumu_dist[i] = dist[i:].sum()
        return (cumu_dist, error, lbb)

    #
    #  Get link distance distribution
    #

    def link_distance_distribution(self, n_bins, grid_type="spherical",
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
            take values "euclidean" and "spherical".
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
            D = self.grid.angular_distance()
        elif grid_type == "euclidean":
            D = self.grid.euclidean_distance()

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

    #
    #  Area weighted connectivity (AWC) related measures
    #

    def area_weighted_connectivity(self):
        """
        Return area weighted connectivity (:math:`AWC`).

        It gives the fractional area of the network, a node is connected to.
        :math:`AWC` is closely related to node splitting invariant degree
        :meth:`.Network.nsi_degree` with area as node weight.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().area_weighted_connectivity())
        array([ 0.4854, 0.499 , 0.3342, 0.3446, 0.5146, 0.1726])

        :rtype: 1D Numpy array [index]
        :return: the area weighted connectivity sequence.
        """
        if self.silence_level <= 1:
            print("Calculating area weighted connectivity...")

        if self.directed:
            return (self.inarea_weighted_connectivity()
                    + self.outarea_weighted_connectivity())
        else:
            return self.inarea_weighted_connectivity()

    def inarea_weighted_connectivity(self):
        """
        Return in-area weighted connectivity.

        It gives the fractional area of the netwerk that connects to a given
        node. For undirected networks, it calculates total area weighted
        connectivity.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().inarea_weighted_connectivity())
        array([ 0.4854, 0.499 , 0.3342, 0.3446, 0.5146, 0.1726])

        :rtype: 1D Numpy array [index]
        :return: the in-area weighted connectivity sequence.
        """
        if self.silence_level <= 1:
            print("Calculating in - area weighted connectivity...")

        #  Calculate the sequence of cosine of latitude for all nodes
        cos_lat = self.grid.cos_lat()

        #  Calculate total dimensionless area of the sphere
        norm = cos_lat.sum()

        #  Normalize area weighted connectivity by the total dimensionless area
        inawc = cos_lat.dot(self.adjacency) / norm

        return inawc

    def outarea_weighted_connectivity(self):
        """
        Return out-area weighted connectivity.

        It gives the fractional area of the netwerk that a given node connects
        to. For undirected networks, it calculates total area weighted
        connectivity.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outarea_weighted_connectivity())
        array([ 0.4854, 0.499 , 0.3342, 0.3446, 0.5146, 0.1726])

        :rtype: 1D Numpy array [index]
        :return: the out-area weighted connectivity sequence.
        """
        if self.silence_level <= 1:
            print("Calculating out - area weighted connectivity...")

        #  Calculate the sequence of cosine of latitude for all nodes
        cos_lat = self.grid.cos_lat()

        #  Calculate total dimensionless area of the sphere
        norm = cos_lat.sum()

        #  Normalize area weighted connectivity by the total dimensionless area
        outawc = np.dot(self.adjacency, cos_lat) / norm

        return outawc

    def area_weighted_connectivity_distribution(self, n_bins):
        """
        Return the area weighted connectivity frequency distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                area_weighted_connectivity_distribution(n_bins=4)[0])
        array([ 0.1565, 0.3367, 0.3446, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the :math:`AWC` distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating AWC frequency distribution...")

        awc = self.area_weighted_connectivity()

        return self.geographical_distribution(awc, n_bins)

    def inarea_weighted_connectivity_distribution(self, n_bins):
        """
        Return the in-area weighted connectivity frequency distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                inarea_weighted_connectivity_distribution(n_bins=4)[0])
        array([ 0.1565, 0.3367, 0.3446, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the in-:math:`AWC` distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating in-AWC frequency distribution...")

        in_awc = self.inarea_weighted_connectivity()

        return self.geographical_distribution(in_awc, n_bins)

    def outarea_weighted_connectivity_distribution(self, n_bins):
        """
        Return the out-area weighted connectivity frequency distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outarea_weighted_connectivity_distribution(n_bins=4)[0])
        array([ 0.1565, 0.3367, 0.3446, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the out-:math:`AWC` distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating out-AWC frequency distribution...")

        out_awc = self.outarea_weighted_connectivity()

        return self.geographical_distribution(out_awc, n_bins)

    def area_weighted_connectivity_cumulative_distribution(self, n_bins):
        """
        Return the cumulative area weighted connectivity distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                area_weighted_connectivity_cumulative_distribution(
        ...         n_bins=4)[0])
        array([ 1. , 0.8435, 0.5068, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the cumulative :math:`AWC` distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating cumulative AWC distribution...")

        awc = self.area_weighted_connectivity()

        return self.geographical_cumulative_distribution(awc, n_bins)

    def inarea_weighted_connectivity_cumulative_distribution(self, n_bins):
        """
        Return the cumulative in-area weighted connectivity distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                inarea_weighted_connectivity_cumulative_distribution(
        ...         n_bins=4)[0])
        array([ 1. , 0.8435, 0.5068, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the cumulative in-:math:`AWC` distribution, statistical error,
                 and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating cumulative in-AWC distribution...")

        in_awc = self.inarea_weighted_connectivity()

        return self.geographical_cumulative_distribution(in_awc, n_bins)

    def outarea_weighted_connectivity_cumulative_distribution(self, n_bins):
        """
        Return the cumulative out-area weighted connectivity distribution.

        Also return estimated statistical error and lower bin boundaries.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outarea_weighted_connectivity_cumulative_distribution(
        ...         n_bins=4)[0])
        array([ 1. , 0.8435, 0.5068, 0.1622])

        :type n_bins: number (int)
        :arg n_bins: The number of bins for histogram.

        :rtype: tuple of three 1D Numpy arrays [bin]
        :return: the cumulative out-:math:`AWC` distribution, statistical
                 error, and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Calculating cumulative out-AWC distribution...")

        out_awc = self.outarea_weighted_connectivity()

        return self.geographical_cumulative_distribution(out_awc, n_bins)

    def average_neighbor_area_weighted_connectivity(self):
        """
        Return average neighbor area weighted connectivity.

        .. note::
           Does not use directionality information.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                average_neighbor_area_weighted_connectivity())
        array([ 0.3439, 0.3978, 0.5068, 0.4922, 0.4395, 0.4854])

        :rtype: 1D Numpy array [index]
        :return: the average neighbor area weighted connectivity sequence.
        """
        if self.silence_level <= 1:
            print("Calculating average neighbour AWC...")

        A = self.undirected_adjacency()
        degree = self.degree()
        awc = self.area_weighted_connectivity()
        average_neighbor_awc = A * awc

        #  Normalize by node degree
        average_neighbor_awc[degree != 0] /= degree[degree != 0]
        return average_neighbor_awc

    def max_neighbor_area_weighted_connectivity(self):
        """
        Return maximum neighbor area weighted connectivity.

        .. note::
           Does not use directionality information.

        >>> r(GeoNetwork.SmallTestNetwork().\
                max_neighbor_area_weighted_connectivity())
        array([ 0.5146, 0.5146, 0.5146, 0.499 , 0.499 , 0.4854])

        :rtype: 1D Numpy array [index]
        :return: the maximum neighbor area weighted connectivity sequence.
        """
        if self.silence_level <= 1:
            print("Calculating maximum neighbour AWC...")

        A = self.undirected_adjacency().A
        awc = self.area_weighted_connectivity()
        max_neighbor_awc = np.zeros(self.N)

        for i in range(self.N):
            max_neighbor_awc[i] = awc[A[i, :] == 1].max()

        return max_neighbor_awc

    #
    #  Distance related measures
    #

    #  (Un)directed average link distances

    #  TODO: Discuss geometry correction with Jobst.
    def _calculate_general_average_link_distance(self, adjacency, degrees,
                                                 geometry_corrected=False):
        """
        Return general average link distances (:math:`ALD`).

        This general method is called to calculate undirected average link
        distance, average in-link distance and average out-link distance.

        The resulting sequence can optionally be corrected for biases in
        average link distance arising due to the grid geometry. E.g., for
        regional networks, nodes on the boundaries may have a bias towards
        larger values of :math:`ALD`, while nodes in the center have a bias
        towards smaller values of :math:`ALD`.

        :type adjacency: 2D array [index, index]
        :arg adjacency: The adjacency matrix.
        :type degrees: 1D array [index]
        :arg degrees: The degree sequence.
        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the general average link distance sequence.
        """
        D = self.grid.angular_distance()
        k = self.degree()

        average_link_distance = np.zeros(self.N)

        #  Normalize by degree, not by number of nodes!!!
        average_link_distance[k != 0] = \
            (D * adjacency).sum(axis=1)[k != 0] / k[k != 0]

        if geometry_corrected:
            #  Calculate the average link distance for a fully connected
            #  network to correct for geometrical biases, particularly in
            #  regional networks.
            ald_correction = D.mean(axis=1)
            # aldCorrection = angularDistance.max(axis=1)

            #  Correct average link distance
            average_link_distance /= ald_correction

        return average_link_distance

    def average_link_distance(self, geometry_corrected=False):
        """
        Return average link distances (undirected).

        .. note::
           Does not use directionality information.

        **Examples:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                average_link_distance(geometry_corrected=False))
        array([ 0.3885, 0.1943, 0.1456, 0.2433, 0.2912, 0.4847])
        >>> r(GeoNetwork.SmallTestNetwork().\
                average_link_distance(geometry_corrected=True))[:-1]
        array([ 1.5988, 1.0921, 1.0001, 1.6708, 1.6384])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the average link distance sequence (undirected).
        """
        if self.silence_level <= 1:
            print("Calculating average link distance...")

        A = self.undirected_adjacency().A
        degree = self.degree()

        return self._calculate_general_average_link_distance(
            A, degree, geometry_corrected=geometry_corrected)

    def inaverage_link_distance(self, geometry_corrected=False):
        """
        Return in-average link distances.

        Return regular average link distance for undirected networks.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                inaverage_link_distance(geometry_corrected=False))
        array([ 0.3885, 0.1943, 0.1456, 0.2433, 0.2912, 0.4847])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the in-average link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating in-average link distance...")

        A = self.adjacency.T
        in_degree = self.indegree()

        return self._calculate_general_average_link_distance(
            A, in_degree, geometry_corrected=geometry_corrected)

    def outaverage_link_distance(self, geometry_corrected=False):
        """
        Return out-average link distances.

        Return regular average link distance for undirected networks.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outaverage_link_distance(geometry_corrected=False))
        array([ 0.3885, 0.1943, 0.1456, 0.2433, 0.2912, 0.4847])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the out-average link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating out-average link distance...")

        A = self.adjacency
        out_degree = self.outdegree()

        return self._calculate_general_average_link_distance(
            A, out_degree, geometry_corrected=geometry_corrected)

    #  (Un)directed total link distances

    def total_link_distance(self, geometry_corrected=False):
        """
        Return the sequence of total link distances for all nodes.

        .. note::
           Does not use directionality information.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                total_link_distance(geometry_corrected=False))
        array([ 0.1886, 0.097 , 0.0486, 0.0838, 0.1498, 0.0837])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the total link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating total link distance...")

        ald = self.average_link_distance(geometry_corrected)
        awc = self.area_weighted_connectivity()

        return ald * awc

    def intotal_link_distance(self, geometry_corrected=False):
        """
        Return the sequence of in-total link distances for all nodes.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                intotal_link_distance(geometry_corrected=False))
        array([ 0.1886, 0.097 , 0.0486, 0.0838, 0.1498, 0.0837])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the in-total link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating in-total link distance...")

        in_ald = self.inaverage_link_distance(geometry_corrected)
        in_awc = self.inarea_weighted_connectivity()

        return in_ald * in_awc

    def outtotal_link_distance(self, geometry_corrected=False):
        """
        Return the sequence of out-total link distances for all nodes.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outtotal_link_distance(geometry_corrected=False))
        array([ 0.1886, 0.097 , 0.0486, 0.0838, 0.1498, 0.0837])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the out-total link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating out-total link distance...")

        out_ald = self.outaverage_link_distance(geometry_corrected)
        out_awc = self.outarea_weighted_connectivity()

        return out_ald * out_awc

    #  (Un)directed connectivity weighted link distances

    def _calculate_general_connectivity_weighted_distance(self, adjacency,
                                                          degrees):
        """
        Return general connectivity weighted link distances (CWD).

        This method is called to calculate undirected CWD, in-CWD
        and out-CWD.

        :type adjacency: 2D array [index, index]
        :arg adjacency: The adjacency matrix.
        :type degrees: 1D array [index]
        :arg degrees: The degree sequence.
        :rtype: 1D array [index]
        :return: the general connectivity weighted distance sequence.
        """
        D = self.grid.angular_distance()
        connectivity_weighted_distance = np.zeros(self.N)

        cos_lat = self.grid.cos_lat()
        norm = cos_lat.sum()

        for i in range(self.N):
            connectivity_weighted_distance[i] = \
                (adjacency[i, :] * cos_lat * D[i, :]).sum()

        #  Normalize by node degree and total dimensionless area
        connectivity_weighted_distance[degrees != 0] /= \
            degrees[degrees != 0] * norm

        return connectivity_weighted_distance

    def connectivity_weighted_distance(self):
        """
        Return undirected connectivity weighted link distances (CWD).

        .. note::
           Does not use directionality information.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                connectivity_weighted_distance())
        array([ 0.0625, 0.0321, 0.0241, 0.0419, 0.05 , 0.0837])

        :rtype: 1D Numpy array [index]
        :return: the undirected connectivity weighted distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating connectivity weighted link distance...")

        A = self.undirected_adjacency().A
        degree = self.degree()
        return self._calculate_general_connectivity_weighted_distance(
            A, degree)

    def inconnectivity_weighted_distance(self):
        """
        Return in-connectivity weighted link distances (CWD).

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                inconnectivity_weighted_distance())
        array([ 0.0625, 0.0321, 0.0241, 0.0419, 0.05 , 0.0837])

        :rtype: 1D Numpy array [index]
        :return: the in-connectivity weighted distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating in-connectivity weighted link distance...")

        A = self.adjacency.transpose()
        indegree = self.indegree()

        return self._calculate_general_connectivity_weighted_distance(A,
                                                                      indegree)

    def outconnectivity_weighted_distance(self):
        """
        Return out-connectivity weighted link distances (CWD).

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                outconnectivity_weighted_distance())
        array([ 0.0625, 0.0321, 0.0241, 0.0419, 0.05 , 0.0837])

        :rtype: 1D Numpy array [index]
        :return: the out-connectivity weighted distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating out-connectivity weighted link distance...")

        A = self.adjacency
        outdegree = self.outdegree()

        return self._calculate_general_connectivity_weighted_distance(
            A, outdegree)

    def max_link_distance(self):
        """
        Return maximum angular geodesic link distances.

        .. note::
           Does not use directionality information.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().max_link_distance())
        array([ 0.4847, 0.2911, 0.1938, 0.292 , 0.3887, 0.4847])

        :rtype: 1D Numpy array [index]
        :return: the maximum link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating maximum link distance...")

        A = self.undirected_adjacency().A
        D = self.grid.angular_distance()

        maximum_link_distance = (D * A).max(axis=1)
        return maximum_link_distance

    #
    #  Link weighted network measures
    #

    @cached_const('base', 'angular_distance')
    def angular_distance(self):
        """
        Return the angular great circle distance matrix.
        """
        ad = self.grid.angular_distance()
        self.set_link_attribute('angular_distance', ad)
        return ad

    def average_distance_weighted_path_length(self):
        """
        Return average distance weighted path length.

        Returns the average path length link-weighted by the angular
        great circle distance between nodes.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                average_distance_weighted_path_length())
        0.4985

        :rtype: number (float)
        :return: the average distance weighted path length.
        """
        self.angular_distance()
        return self.average_path_length('angular_distance')

    def distance_weighted_closeness(self):
        """
        Return distance weighted closeness.

        Returns the sequence of closeness centralities link-weighted by the
        angular great circle distance between nodes.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                distance_weighted_closeness())
        array([ 2.2378, 2.4501, 2.2396, 2.4501, 2.2396, 1.1982])

        :rtype: 1D Numpy array [index]
        :return: the distance weighted closeness sequence.
        """
        self.angular_distance()
        return self.closeness('angular_distance')

    def local_distance_weighted_vulnerability(self):
        """
        Return local distance weighted vulnerability.

        Return the sequence of vulnerabilities link-weighted by
        the angular great circle distance between nodes.

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().\
                local_distance_weighted_vulnerability())
        array([ 0.0325, 0.3137, 0.2056, 0.028 , -0.0283, -0.288 ])

        :rtype: 1D Numpy array [index]
        :return: the local distance weighted vulnerability sequence.
        """
        self.angular_distance()
        return self.local_vulnerability('angular_distance')

    #
    #  Clustering coefficients including geographical information
    #

    #  TODO: Maybe implement this one day...
    def local_tsonis_clustering(self):
        """
        Return local Tsonis clustering.

        This measure of local clustering was introduced in [Tsonis2008a]_.

        :rtype: 1D Numpy array (index)
        :return: the local Tsonis clustering sequence.
        """
        if self.silence_level <= 1:
            print("Calculating local Tsonis clustering coefficients...")

        tsonis_clustering = np.zeros(self.N)
        return tsonis_clustering

    def local_geographical_clustering(self):
        """
        Return local geographical clustering.

        Returns the sequence of local clustering coefficients weighted by the
        inverse angular great circle distance between nodes. This guarantees,
        that short links between spatially neighboring nodes in a triangle are
        weighted higher than long links between nodes that are spatially
        far away.

        Uses a definition of weighted clustering coefficient introduced in
        [Holme2007]_.

        .. note::
           Experimental measure!

        **Example:**

        >>> r(GeoNetwork.SmallTestNetwork().local_geographical_clustering())
        Calculating local weighted clustering coefficient...
        array([ 0. , 0.0998, 0.1489, 0. , 0.2842, 0. ])

        :rtype: 1D Numpy array (index)
        :return: the local geographical clustering sequence.
        """
        ad = self.grid.angular_distance().copy()
        ad[ad == 0] = np.inf
        distance_weighted_adjacency = self.adjacency / ad
        return self.weighted_local_clustering(distance_weighted_adjacency)

    #  TODO: Experimental code!
    #  TODO: Improve documentation (Jobst).
    def nsi_connected_hamming_cluster_tree(self, lon_closed=True,
                                           lat_closed=False, alpha=0.01):
        """
        Perform NSI agglomerative clustering.

        Minimize in each step the Hamming distance between the original and
        the clustered network, but only joins connected clusters.

        Return c,h where c[i,j] = i iff node j is in cluster no. i,
        and 0 otherwise, and h is the corresponding list of total resulting
        relative Hamming distance between 0 and 1. The cluster numbers for all
        nodes and a k clusters solution is then c[:2*N-k,:].max(axis=0)

        :arg bool lon_closed: TODO
        :arg bool lat_closed: TODO
        :arg float alpha: TODO

        :rtype: TODO
        :return: TODO
        """
        N = self.N
        B = np.zeros((N, N)).astype("int")
        width = self.grid.grid()["lon"].size

        for i in range(0, N):
            if i % width > 0:
                B[i, i - 1] = 1
            elif lon_closed:
                B[i, i - 1 + width] = 1
            if i % width < width - 1:
                B[i, i + 1] = 1
            elif lon_closed:
                B[i, i + 1 - width] = 1
            if i >= width:
                B[i, i - width] = 1
            elif lat_closed:
                B[i, i - width + N] = 1
            if i < N - width:
                B[i, i + width] = 1
            elif lat_closed:
                B[i, i + width - N] = 1

        return self.do_nsi_hamming_clustering(admissible_joins=B, alpha=alpha)

    @staticmethod
    def cartesian2latlon(pos):
        return np.arcsin(pos[2]) * 180 / np.pi, \
            np.arctan2(pos[0], pos[1]) * 180 / np.pi

    @staticmethod
    def latlon2cartesian(lat, lon):
        lat *= np.pi/180
        lon *= np.pi/180
        coslat = np.cos(lat)
        return [coslat * np.sin(lon), coslat * np.cos(lon), np.sin(lat)]

    def boundary(self, nodes, geodesic=True, gap=0.0):
        """
        Return a list of ordered lists of nodes on the connected parts of the
        boundary of a subset of nodes and a list of ordered lists of (lat,lon)
        coordinates of the corresponding polygons

        * EXPERIMENTAL! *
        """
        #  Optional import for this experimental method
        try:
            import stripack  # @UnresolvedImport
            # tries to import stripack.so which must have been compiled with
            # f2py -c -m stripack stripack.f90
        except ImportError:
            raise RuntimeError("NOTE: stripack.so not available, boundary() \
                               won't work.")

        N = self.N
        nodes_set = set(nodes)
        if len(nodes_set) >= N:
            return [], [], [], [(0.0, 0.0)]
        # find grid neighbours:
        if geodesic:
            if self.cartesian is not None:
                pos = self.cartesian
            else:
                # find cartesian coordinates of nodes,
                # assuming a perfect unit radius sphere:
                lat = self.grid.lat_sequence() * np.pi / 180
                lon = self.grid.lon_sequence() * np.pi / 180
                pos = self.cartesian = np.zeros((N, 3))
                coslat = np.cos(lat)
                self.cartesian[:, 0] = coslat * np.sin(lon)
                self.cartesian[:, 1] = coslat * np.cos(lon)
                self.cartesian[:, 2] = np.sin(lat)

                # find neighbours of each node in Delaunay triangulation,
                # sorted in counter-clockwise order, using stripack fortran
                # library:
                #  will contain 1-based node indices
                list_ = np.zeros(6*(N-2)).astype("int32")
                #  will contain 1-based list_ indices
                lptr = np.zeros(6*(N-2)).astype("int32")
                #  will contain 1-based list_ indices
                lend = np.zeros(N).astype("int32")
                lnew = 0
                near = np.zeros(N).astype("int32")
                foll = np.zeros(N).astype("int32")
                dist = np.zeros(N)
                ier = 0
                stripack.trmesh(self.cartesian[:, 0],
                                self.cartesian[:, 1],
                                self.cartesian[:, 2],
                                list_, lptr, lend, lnew,  # output vars
                                near, foll, dist,
                                ier)  # output var
                self.grid_neighbours = [None for i in range(N)]
                self.grid_neighbours_set = [None for i in range(N)]
                rN = range(N)
                for i in rN:
                    nbsi = []
                    ptr0 = ptr = lend[i]-1
                    for j in rN:
                        nbsi.append(list_[ptr]-1)
                        ptr = lptr[ptr]-1
                        if ptr == ptr0:
                            break
                    self.grid_neighbours[i] = nbsi
                    self.grid_neighbours_set[i] = set(nbsi)
        else:
            raise NotImplementedError("Not yet implemented for \
                                      lat-lon-regular grids!")

        remaining = nodes_set.copy()
        boundary = []
        shape = []
        fullshape = []
        representative = []
        # find a node on the boundary and an outer neighbour:
        lam = 0.5 + gap/2
        lam1 = 1-lam
        while remaining:
            i = list(remaining)[0]
            this_remove = [i]
            cont = False
            while self.grid_neighbours_set[i] <= nodes_set:
                i = self.grid_neighbours[i][int(np.floor(
                    len(self.grid_neighbours[i])*random.uniform()))]
                if i not in remaining:  # we had this earlier
                    cont = True
                    break
                this_remove.append(i)
            remaining -= set(this_remove)
            # if len(nodes_set)==151: print(i,this_remove,remaining,cont)
            if cont:
                continue
            o = list(self.grid_neighbours_set[i] - nodes_set)[0]

            # traverse boundary:
            partial_boundary = [i]
            partial_shape = [lam*pos[i] + lam1*pos[o]]
            partial_fullshape = [0.49*pos[i] + 0.51*pos[o]]
            steps = [(i, o)]
            for it in range(N):  # at most this many steps we need
                nbi = self.grid_neighbours[i]
                j = nbi[0]
                try:
                    j = nbi[(nbi.index(o)-1) % len(nbi)]
                except IndexError:
                    print("O!", i, o, j, nbi, self.grid_neighbours[o], steps)
                    raise
                if j in nodes_set:
                    i = j
                    partial_boundary.append(i)
                    try:
                        remaining.remove(i)
                    except KeyError:
                        pass
                else:
                    partial_fullshape.append(
                        0.32*pos[i]+0.34*pos[o]+0.34*pos[j])
                    o = j
                partial_shape.append(lam*pos[i] + lam1*pos[o])
                partial_fullshape.append(0.49*pos[i] + 0.51*pos[o])
                if (i, o) in steps:
                    break
                steps.append((i, o))

            mind2 = np.inf
            latlon_shape = []
            latlon_fullshape = []
            length = len(partial_shape)-1
            off = length/2
            for it in range(length):
                pos1 = partial_shape[it]
                pos2 = partial_shape[(it+off) % length]
                latlon_shape.append(self.cartesian2latlon(pos1))
                d2 = ((pos2-pos1)**2).sum()
                if d2 < mind2:
                    rep = self.cartesian2latlon((pos1+pos2)/2)
                    mind2 = d2
            latlon_shape.append(self.cartesian2latlon(partial_shape[-1]))
            for it, _ in enumerate(partial_fullshape):
                pos1 = partial_fullshape[it]
                latlon_fullshape.append(self.cartesian2latlon(pos1))

            boundary.append(partial_boundary)
            shape.append(latlon_shape)
            fullshape.append(latlon_fullshape)
            representative.append(rep)

        # TODO: sort sub-regions by descending size!
        return boundary, shape, fullshape, representative
