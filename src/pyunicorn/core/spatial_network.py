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
Provides class for analyzing spatially embedded complex networks.
"""

from typing import Tuple
from collections.abc import Hashable

import numpy as np
from numpy import random
import igraph

from ._ext.types import to_cy, ADJ, NODE, FIELD, DEGREE
from ._ext.numerics import _randomly_rewire_geomodel_I, \
    _randomly_rewire_geomodel_II, _randomly_rewire_geomodel_III

from .network import Network
from .grid import Grid


class SpatialNetwork(Network):
    """
    Encapsulates a spatially embedded network.

    Adds more network measures and statistics based on the
    spatial embedding.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, grid: Grid, adjacency=None, edge_list=None,
                 directed=False, silence_level=0):
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
        assert isinstance(grid, Grid)
        self.grid: Grid = grid
        """(Grid) - Grid object describing the network's spatial embedding"""

        #  Call constructor of parent class Network
        Network.__init__(self, adjacency=adjacency, edge_list=edge_list,
                         directed=directed, silence_level=silence_level)

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return Network.__cache_state__(self) + (self.grid,)

    def __str__(self):
        """
        Return a string representation of the SpatialNetwork object.
        """
        return f'SpatialNetwork:\n{Network.__str__(self)}'

    #
    #  Load and save GeoNetwork object
    #

    # pylint: disable=keyword-arg-before-vararg
    def save(self, filename, fileformat=None, *args, **kwds):
        """
        Save the SpatialNetwork object to files.

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

        :arg tuple/list filename: Tuple or list of two strings, namely
            the paths to the files where the Network object and the
            GeoGrid object are to be stored (filename_network, filename_grid)
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
        try:
            (filename_network, filename_grid) = filename
        except ValueError as e:
            raise ValueError("'filename' must be a tuple or list of two "
                             "items: filename_network, filename_grid") from e
        #  Store network
        Network.save(self, filename=filename_network, fileformat=fileformat,
                     *args, **kwds)

        #  Store grid
        if filename_grid is not None:
            self.grid.save(filename=filename_grid)

    # pylint: disable=keyword-arg-before-vararg
    @staticmethod
    def Load(filename, fileformat=None, silence_level=0, *args, **kwds):
        """
        Return a SpatialNetwork object stored in files.

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

        :arg tuple/list filename: Tuple or list of two strings, namely
            the paths to the files containing the Network object and the
            Grid object (filename_network, filename_grid)
        :arg str fileformat: the format of the file (if known in advance)
          ``None`` means auto-detection. Possible values are: ``"ncol"`` (NCOL
          format), ``"lgl"`` (LGL format), ``"graphml"``, ``"graphmlz"``
          (GraphML and gzipped GraphML format), ``"gml"`` (GML format),
          ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"`` (DIMACS format),
          ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge list),
          ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python pickled
          format).
        :arg int silence_level: The inverse level of verbosity of the object.
        :rtype: SpatialNetwork object
        :return: :class:`SpatialNetwork` instance.
        """
        try:
            (filename_network, filename_grid) = filename
        except ValueError as e:
            raise ValueError("'filename' must be a tuple or list of two "
                             "items: filename_network, filename_grid") from e

        #  Load Grid object
        grid = Grid.Load(filename_grid)

        #  Load to igraph Graph object
        graph = igraph.Graph.Read(f=filename_network, format=fileformat,
                                  *args, **kwds)

        #  Extract adjacency matrix
        A = np.array(graph.get_adjacency(type=2).data)

        #  Create GeoNetwork instance
        net = SpatialNetwork(grid=grid, adjacency=A,
                             directed=graph.is_directed(),
                             silence_level=silence_level)

        #  Extract node weights
        if "node_weight_nsi" in graph.vs.attribute_names():
            node_weights = \
                np.array(graph.vs.get_attribute_values("node_weight_nsi"))
            net.node_weights = node_weights

        #  Overwrite igraph Graph object in Network instance to restore link
        #  attributes/weights
        net.graph = graph
        #  invalidate cache
        net._mut_la += 1
        return net

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

        :rtype: SpatialNetwork object
        :return: :class:`SpatialNetwork` instance for testing purposes.
        """
        return SpatialNetwork(grid=Grid.SmallTestGrid(),
                              adjacency=Network.SmallTestNetwork().adjacency,
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
        #  Get number of links
        E = self.n_links
        #  Collect adjacency and distance matrices
        A = to_cy(self.adjacency, ADJ)
        D = to_cy(distance_matrix, FIELD)
        #  Get degree sequence
        # degree = self.degree()

        #  Define for brevity
        eps = float(inaccuracy)

        # iterations = int(iterations)

        #  Get edge list
        edges = to_cy(np.array(self.graph.get_edgelist()), NODE)

        _randomly_rewire_geomodel_I(iterations, eps, A, D, E, edges)

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

        #  Get number of links
        E = int(self.n_links)
        #  Collect adjacency and distance matrices
        A = to_cy(self.adjacency, ADJ)
        D = to_cy(distance_matrix, FIELD)

        #  Define for brevity
        eps = float(inaccuracy)

        #  Get edge list
        edges = to_cy(np.array(self.graph.get_edgelist()), NODE)

        _randomly_rewire_geomodel_II(iterations, eps, A, D, E, edges)

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
        #  Get number of links
        E = int(self.n_links)
        #  Collect adjacency and distance matrices
        A = to_cy(self.adjacency, ADJ)
        D = to_cy(distance_matrix, FIELD)
        #  Get degree sequence
        degree = to_cy(self.degree(), DEGREE)

        #  Define for brevity
        eps = float(inaccuracy)

        #  Get edge list
        edges = to_cy(np.array(self.graph.get_edgelist()), NODE)

        _randomly_rewire_geomodel_III(iterations, eps, A, D, E, edges, degree)

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
        A = (p >= P).astype(ADJ)
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

    #
    #  Distance related measures
    #

    #  (Un)directed average link distances

    def _calculate_general_average_link_distance(self, adjacency, degree,
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
        :type degree: 1D array [index]
        :arg degree: The degree sequence.
        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the general average link distance sequence.
        """
        D = self.grid.distance()

        average_link_distance = np.zeros(self.N)

        #  Normalize by degree, not by number of nodes
        average_link_distance[degree != 0] = \
            (D * adjacency).sum(axis=1)[degree != 0] / degree[degree != 0]

        if geometry_corrected:
            #  Calculate the average link distance for a fully connected
            #  network to correct for geometrical biases, particularly in
            #  regional networks.
            ald_correction = D.mean(axis=1)

            #  Correct average link distance
            average_link_distance /= ald_correction

        return average_link_distance

    def average_link_distance(self, geometry_corrected=False):
        """
        Return average link distances (undirected).

        .. note::
           Does not use directionality information.

        **Examples:**

        >>> SpatialNetwork.SmallTestNetwork().\
                average_link_distance(geometry_corrected=False)
        array([22.36067963, 11.18033981,  8.38525486, 13.97542477, 16.77050908,
               27.95084953])
        >>> SpatialNetwork.SmallTestNetwork().\
                average_link_distance(geometry_corrected=True)[:-1]
        array([1.6       , 1.09090909, 1.        , 1.66666667, 1.63636357])

        :arg bool geometry_corrected: Toggles geometry correction.
        :rtype: 1D array [index]
        :return: the average link distance sequence (undirected).
        """
        if self.silence_level <= 1:
            print("Calculating average link distance...")

        A = self.undirected_adjacency().toarray()
        degree = self.degree()

        return self._calculate_general_average_link_distance(
            A, degree, geometry_corrected=geometry_corrected)

    def inaverage_link_distance(self, geometry_corrected=False):
        """
        Return in-average link distances.

        Return regular average link distance for undirected networks.

        **Example:**

        >>> SpatialNetwork.SmallTestNetwork().\
                inaverage_link_distance(geometry_corrected=False)
        array([22.36067963, 11.18033981,  8.38525486, 13.97542477, 16.77050908,
               27.95084953])

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

        >>> SpatialNetwork.SmallTestNetwork().
                outaverage_link_distance(geometry_corrected=False)
        array([22.36067963, 11.18033981,  8.38525486, 13.97542477, 16.77050908,
               27.95084953])

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

    def max_link_distance(self):
        """
        Return maximum angular geodesic link distances.

        .. note::
           Does not use directionality information.

        **Example:**

        >>> SpatialNetwork.SmallTestNetwork().max_link_distance()
        array([27.95085, 16.77051, 11.18034, 16.77051, 22.36068, 27.95085],
              dtype=float32)

        :rtype: 1D Numpy array [index]
        :return: the maximum link distance sequence.
        """
        if self.silence_level <= 1:
            print("Calculating maximum link distance...")

        A = self.undirected_adjacency().toarray()
        D = self.grid.distance()

        maximum_link_distance = (D * A).max(axis=1)
        return maximum_link_distance

    #
    #  Link weighted network measures
    #

    def distance(self):
        """
        Return the distance matrix.
        """
        dist = self.grid.distance()
        if not self.find_link_attribute('distance'):
            self.set_link_attribute('distance', dist)
        return dist

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
        self.distance()
        return self.average_path_length('distance')

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
        self.distance()
        return self.closeness('distance')

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
        self.distance()
        return self.local_vulnerability('distance')
