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
Provides classes for generating and analyzing complex climate networks.
"""

from typing import Tuple
from collections.abc import Hashable, Callable

import numpy as np
import igraph

from ..core.cache import Cached
from ..core import GeoNetwork, GeoGrid


class ClimateNetwork(GeoNetwork):

    """
    Encapsulates a similarity network embedded on a spherical surface.

    Particularly provides functionality to generate a complex network from the
    matrix of a similarity measure of time series.

    The analysis of climate time series based on similarity networks was first
    introduced in [Tsonis2004]_.
    """
    #
    #  Definitions of internal methods
    #

    def __init__(self, grid: GeoGrid, similarity_measure: np.ndarray,
                 threshold=None, link_density=None, non_local=False,
                 directed=False, node_weight_type="surface", silence_level=0):
        """
        Initialize an instance of :class:`ClimateNetwork`.

        .. note::
           Either threshold **OR** link_density have to be given!

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos**2 lat)

        :type grid: :class:`.GeoGrid`
        :arg  grid: The GeoGrid object describing the network's spatial
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
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        #  Initialize
        assert isinstance(grid, GeoGrid)
        self.grid: GeoGrid = grid
        self.directed = directed
        self.silence_level = silence_level

        # mutation count
        if not hasattr(self, "_mut_clim"):
            self._mut_clim: int = 0
        else:
            self._mut_clim += 1

        #  FIXME: Is taking the absolute value by default OK?
        self._similarity_measure = np.abs(similarity_measure.astype("float32"))
        self._non_local = non_local
        self.N = grid.N
        self.node_weight_type = node_weight_type

        #  Sets the threshold and generates the network by thresholding and
        #  calling the "constructor" of parent class GeoNetwork.
        if threshold is not None:
            self.set_threshold(threshold)
        elif link_density is not None:
            self.set_link_density(link_density)
        else:
            print("Either threshold or link_density have to be prescribed "
                  "for network construction!")
        GeoNetwork.__init__(self, adjacency=self.adjacency, grid=self.grid,
                            directed=self.directed,
                            node_weight_type=self.node_weight_type,
                            silence_level=self.silence_level)

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return GeoNetwork.__cache_state__(self) + (self._mut_clim,)

    def __str__(self):
        """
        Return a string representation of the ClimateNetwork object.

        **Example:**

        >>> print(ClimateNetwork.SmallTestNetwork())
        ClimateNetwork:
        GeoNetwork:
        Network: undirected, 6 nodes, 7 links, link density 0.467.
        Geographical boundaries:
                 time     lat     lon
           min    0.0    0.00    2.50
           max    9.0   25.00   15.00
        Threshold: 0.5
        Local connections filtered out: False
        """
        return (f'ClimateNetwork:\n{GeoNetwork.__str__(self)}\n' +
                f'Threshold: {self.threshold()}\n' +
                f'Local connections filtered out: {self.non_local()}')

    def _regenerate_network(self):
        """
        Regenerate the current climate network according to a new similarity
        measure.
        """
        ClimateNetwork.__init__(self, grid=self.grid,
                                similarity_measure=self._similarity_measure,
                                threshold=self._threshold,
                                link_density=self.link_density,
                                non_local=self._non_local,
                                directed=self.directed,
                                node_weight_type=self.node_weight_type,
                                silence_level=self.silence_level)

    #
    #  Load and save ClimateNetwork object
    #

    # pylint: disable=keyword-arg-before-vararg
    def save(self, filename, fileformat=None, *args, **kwds):
        """
        Save the ClimateNetwork object to files.

        Unified writing function for graphs. Relies on and partially extends
        the corresponding igraph function. Refer to igraph documentation for
        further details on the various writer methods for different formats.

        This method tries to identify the format of the graph given in
        the first parameter (based on extension) and calls the corresponding
        writer method.

        Existing node and link attributes/weights are also stored depending
        on the chosen file format. E.g., the formats GraphML and gzipped
        GraphML are able to store both node and link weights.

        .. note::
           The similarity measure matrix and grid are not stored if
           the corresponding filenames are None.

        The remaining arguments are passed to the writer method without
        any changes.

        :arg tuple/list filename: Tuple or list of three strings, namely
            the paths to the files where the Network object, the
            GeoGrid object and the similarity measure matrix are to be stored.
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
        :arg str filename_similarity_measure:  The name of the file where the
            similarity measure matrix is to be stored.
        """
        try:
            (filename_network, filename_grid,
             filename_similarity_measure) = filename
        except ValueError as e:
            raise ValueError("'filename' must be a tuple or list of three "
                             "items: filename_network, filename_grid, "
                             "filename_similarity_measure") from e

        #  Store GeoNetwork
        GeoNetwork.save(self, filename=(filename_network, filename_grid),
                        fileformat=fileformat,
                        *args, **kwds)

        #  Store similarity measure
        if filename_similarity_measure is not None:
            similarity_measure = self.similarity_measure()
            similarity_measure.dump(filename_similarity_measure)

    # pylint: disable=keyword-arg-before-vararg
    @staticmethod
    def Load(filename, fileformat=None, silence_level=0, *args, **kwds):
        """
        Return a ClimateNetwork object stored in files.

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

        :arg tuple/list filename: Tuple or list of three strings, namely
            the paths to the files containing the Network object, the
            GeoGrid object and the similarity measure matrix.
            (filename_network, filename_grid, filename_similarity_measure)
        :arg str fileformat: the format of the file (if known in advance)
            ``None`` means auto-detection. Possible values are: ``"ncol"``
            (NCOL format), ``"lgl"`` (LGL format), ``"graphml"``,
            ``"graphmlz"`` (GraphML and gzipped GraphML format), ``"gml"`` (GML
            format), ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"``
            (DIMACS format), ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge
            list), ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python
            pickled format).
        :return: :class:`ClimateNetwork` instance.
        """
        try:
            (filename_network, filename_grid,
             filename_similarity_measure) = filename
        except ValueError as e:
            raise ValueError("'filename' must be a tuple or list of three "
                             "items: filename_network, filename_grid, "
                             "filename_similarity_measure") from e

        #  Load GeoGrid object
        grid = GeoGrid.Load(filename_grid)

        #  Load similarity measure
        similarity_measure = np.load(filename_similarity_measure)

        #  Load to igraph Graph object
        graph = igraph.Graph.Read(f=filename_network, format=fileformat,
                                  *args, **kwds)

        #  Extract adjacency matrix
        A = np.array(graph.get_adjacency(type=2).data)

        #  Extract node weights
        if "node_weight_nsi" in graph.vs.attribute_names():
            node_weights = np.array(
                graph.vs.get_attribute_values("node_weight_nsi"))
        else:
            node_weights = None

        #  Create ClimateNetwork instance
        net = ClimateNetwork(grid=grid, similarity_measure=similarity_measure,
                             directed=graph.is_directed(),
                             silence_level=silence_level)
        net.adjacency = A
        net.node_weights = node_weights

        #  Overwrite igraph Graph object in Network instance to restore link
        #  attributes/weights
        net.graph = graph
        #  invalidate cache
        net._mut_la += 1
        return net

    #
    #  Methods for testing purposes
    #

    @staticmethod
    def SmallTestNetwork():
        """
        Return a 6-node undirected test climate network from a similarity
        matrix.

        The network looks like this::

                3 - 1
                |   | \\
            5 - 0 - 4 - 2

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().adjacency)
        array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])

        :rtype: :class:`.Network` instance
        """
        return ClimateNetwork(grid=GeoGrid.SmallTestGrid(),
                              similarity_measure=np.array(
                                  [[1.0, 0.1, 0.2, 0.6, 0.7, 0.55],
                                   [0.1, 1.0, 0.55, 0.9, 1.0, 0.3],
                                   [0.2, 0.55, 1.0, 0.2, 0.8, 0.1],
                                   [0.6, 0.9, 0.1, 1.0, 0.1, 0.3],
                                   [0.7, 1.0, 0.8, 0.1, 1.0, 0.4],
                                   [0.55, 0.3, 0.1, 0.3, 0.4, 1.0]]),
                              threshold=0.5,
                              directed=False,
                              silence_level=2)

    #
    #  Methods related to fixing link density
    #

    def link_density_function(self, n_bins):
        """
        Return the network's link density as a function of the threshold.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().\
                link_density_function(3)[0])
        array([ 0. , 0.3889, 0.6667])
        >>> r(ClimateNetwork.SmallTestNetwork().\
                link_density_function(3)[1])
        array([ 0.1, 0.4, 0.7, 1. ])

        :type n_bins: number (int)
        :arg n_bins: The number of bins.

        :rtype: tuple of two 1D Numpy arrays [bin]
        :return: the network's link density in dependence on threshold.
        """
        if self.silence_level <= 1:
            print("Calculate the link density function...")

        #  Get the histogram of the correlation measure
        (hist, threshold) = np.histogram(self.similarity_measure(),
                                         bins=n_bins)

        #  Normalize histogram
        hist = hist.astype("float64")
        hist /= hist.sum()

        link_density_function = np.empty(n_bins)

        #  Calculate the link density function
        for i in range(n_bins):
            link_density_function[i] = hist[:i].sum()

        return (link_density_function, threshold)

    def threshold_from_link_density(self, link_density):
        """
        Return the threshold for network construction given link density.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().\
                threshold_from_link_density(link_density=0.5))
        0.4

        :type link_density: number (float)
        :arg link_density: The networks's desired link density.

        :rtype: number (float)
        :return: The threshold of similarity measure, above which
                 two nodes are linked in the network.
        """
        #  Flatten and sort correlation measure matrix
        flat_corr = self.similarity_measure().copy()
        flat_corr = flat_corr.flatten()
        flat_corr.sort()

        #  Get threshold, exclude the entries on the main diagonal here,
        #  since they will not be included in the network anyways!
        threshold = flat_corr[int((1-link_density) * (len(flat_corr)-self.N))]

        #  Clean up
        del flat_corr

        return threshold

    #
    #  Generate adjacency matrix from correlation measure
    #

    def _calculate_threshold_adjacency(self, similarity_measure, threshold):
        """
        Extract the network's adjacency matrix by thresholding.

        The resulting network is a simple graph, i.e., self-loops and
        multiple links are not allowed.

        **Example** (Threshold zero should yield a fully connected network
        given the test similarity matrix):

        >>> net = ClimateNetwork.SmallTestNetwork()
        >>> net._calculate_threshold_adjacency(
        ...     similarity_measure=net.similarity_measure(), threshold=0.0)
        array([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1],
               [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1],
               [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0]], dtype=int8)

        :type similarity_measure: 2D Numpy array [index, index]
        :arg  similarity_measure: The similarity measure for all pairs of
                                  nodes.

        :type threshold: number (float)
        :arg  threshold: The threshold of similarity measure, above which
                         two nodes are linked in the network.

        :rtype:  2D Numpy array (int8) [index, index]
        :return: the network's adjacency matrix.
        """
        if self.silence_level <= 1:
            print("Extracting network adjacency matrix by thresholding...")

        N = similarity_measure.shape[0]
        A = np.zeros((N, N), dtype="int8")
        A[similarity_measure > threshold] = 1

        #  Set the diagonal of the adjacency matrix to zero -> no self loops
        #  allowed.
        A.flat[::N+1] = 0

        return A

    def _calculate_non_local_adjacency(self, similarity_measure, threshold,
                                       a=20, d_min=0.05):
        """
        Return the adjacency matrix with suppressed spatially local links.

        Physically trivial links between geographically close nodes
        are removed.

        For large a, :math:`d_min` corresponds to the minimum distance for
        which links are allowed to exist.

        **Example:**

        >>> net = ClimateNetwork.SmallTestNetwork()
        >>> net._calculate_non_local_adjacency(
        ...     similarity_measure=net.similarity_measure(),
        ...     threshold=0.5, a=30, d_min=0.20)
        array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=int8)

        :type similarity_measure: 2D Numpy array [index, index]
        :arg  similarity_measure: The similarity measure for all pairs of
                                  nodes.

        :type threshold: number (float)
        :arg  threshold: The threshold of similarity measure, above which
                        two nodes are linked in the network.

        :type a: number (float)
        :arg  a: The steepness parameter of the distance weighting function
                 in the transition region from not including any links
                 (weight=0) to including all links (weight=1).

        :type d_min: number (float)
        :arg  d_min: The parameter controlling the minimum distance, above
                     which links can be included in the network
                     (unit radians).

        :rtype:  2D Numpy array (int8) [index, index]
        :return: the network's adjacency matrix.
        """
        if self.silence_level <= 1:
            print("Extracting network adjacency matrix removing local "
                  "connections...")

        weighted_similarity = similarity_measure * \
            (0.5 * (np.tanh(a * (self.grid.angular_distance() - d_min)) + 1))
        # The above line is a function that provides a smooth
        # transition of distance weight, centered around distance d_min.
        # Other sigmoidal type functions could be used as well.

        return self._calculate_threshold_adjacency(weighted_similarity,
                                                   threshold)

    def similarity_measure(self):
        """
        Return the similarity measure used for network construction.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().similarity_measure()[0,:])
        array([ 1. , 0.1 , 0.2 , 0.6 , 0.7 , 0.55])

        :rtype: 2D Numpy array [index, index]
        :return: The similarity measure for all pairs of nodes.
        """
        try:
            return self._similarity_measure
        except AttributeError as e:
            raise AttributeError("Similarity matrix was deleted "
                                 "earlier and cannot be retrieved.") from e

    def non_local(self):
        """
        Indicate if links between spatially close nodes were suppressed.

        **Example:**

        >>> ClimateNetwork.SmallTestNetwork().non_local()
        False

        :return bool: Determines, whether links between spatially close nodes
            should be suppressed.
        """
        return self._non_local

    def set_non_local(self, non_local):
        """
        Toggle suppression of links between spatially close nodes.

        **Example:**

        >>> net = ClimateNetwork.SmallTestNetwork()
        >>> net.set_non_local(non_local=True)
        >>> r(net.adjacency)
        array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])

        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        """
        #  Only change the network if there is a real change in non_local
        if self.non_local() != non_local:
            self._non_local = non_local
            #  Regenerate the climate network using the new setting
            self.set_threshold(self.threshold())

    def threshold(self):
        """
        Return the threshold used to generate the current climate network.

        **Example:**

        >>> ClimateNetwork.SmallTestNetwork().threshold()
        0.5

        :rtype: number (float)
        :return: the threshold used to generate the current climate network.
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        Generate climate network by thresholding similarity matrix.

        **Example** (Number of links decreases as threshold increases):

        >>> net = ClimateNetwork.SmallTestNetwork()
        >>> net.n_links
        7
        >>> net.set_threshold(threshold=0.7)
        >>> net.n_links
        3

        :type threshold: number (float)
        :arg threshold: the threshold used to generate the current climate
                          network.
        """
        #  Set class variable _threshold
        self._threshold = threshold

        similarity = self.similarity_measure()

        if self.non_local():
            A = self._calculate_non_local_adjacency(similarity, threshold)
        else:
            A = self._calculate_threshold_adjacency(similarity, threshold)

        #  Call constructor of parent class GeoNetwork
        GeoNetwork.__init__(self, adjacency=A, grid=self.grid,
                            directed=self.directed,
                            node_weight_type=self.node_weight_type,
                            silence_level=self.silence_level)

    def set_link_density(self, link_density):
        """
        Generate climate network by thresholding with prescribed link density.

        .. note::
           The desired link density can only be achieved approximately in most
           cases.

        **Example:**

        >>> net = ClimateNetwork.SmallTestNetwork()
        >>> r(net.link_density)
        0.4667
        >>> net.set_link_density(link_density=0.7)
        >>> r(net.link_density)
        0.6667

        :type link_density: number (float)
        :arg link_density: The networks's desired link density.
        """
        threshold = self.threshold_from_link_density(link_density)
        self.set_threshold(threshold)

    @Cached.method()
    def correlation_distance(self):
        """
        Return correlation weighted distances between nodes.

        Defined as the elementwise product of the correlation measure and
        angular great circle distance matrices.

        This is a useful measure of the relative importance of links,
        since links with high geographical distance and high correlation
        (teleconnections) get the highest weight. Trivial correlations with
        small geographical distance and high correlation get a lower weight.

        Correlation distance appears to be the simplest functional form of
        combining geographical distance and correlation measure that yields
        meaningful results.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().correlation_distance(), 2)
        array([[ 0.  , 0.01, 0.04, 0.18, 0.27, 0.27],
               [ 0.01, 0.  , 0.05, 0.18, 0.29, 0.12],
               [ 0.04, 0.05, 0.  , 0.02, 0.16, 0.03],
               [ 0.18, 0.18, 0.01, 0.  , 0.01, 0.06],
               [ 0.27, 0.29, 0.16, 0.01, 0.  , 0.04],
               [ 0.27, 0.12, 0.03, 0.06, 0.04, 0.  ]])

        :rtype: 2D matrix [index, index]
        :return: the correlation distance matrix.
        """
        return self.similarity_measure() * self.grid.angular_distance()

    @Cached.method()
    def inv_correlation_distance(self):
        """
        Return correlation weighted distances between nodes.

        :rtype: 2D matrix [index, index]
        """
        m = self.correlation_distance()
        np.fill_diagonal(m, np.inf)
        self.set_link_attribute('inv_correlation_distance', 1 / m)
        return 1 / m

    #
    #  Link weighted network measures
    #

    def correlation_distance_weighted_closeness(self):
        """
        Return correlation distance weighted closeness.

        Calculates the sequence of closeness centralities link-weighted by the
        inverse of correlation distance between nodes. For closeness
        centrality calculation, the inverse of correlation distance is used,
        because high values of this measure should correspond to short
        distances in the graph and vice versa when weighted shortest paths are
        calculated.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().\
                correlation_distance_weighted_closeness())
        array([ 0.1646, 0.1351, 0.0894, 0.1096, 0.1659, 0.1102])

        :rtype: 1D Numpy array [index]
        :return: the correlation distance weighted closeness sequence.
        """
        self.inv_correlation_distance()
        return self.closeness('inv_correlation_distance')

    def local_correlation_distance_weighted_vulnerability(self):
        """
        Return local correlation distance weighted vulnerability.

        Calculates the sequence of vulnerabilities link-weighted by the
        inverse of correlation distance between nodes. For vulnerability
        calculation, the inverse of correlation distance is used, because
        high values of this measure should correspond to short distances in
        the graph and vice versa when weighted shortest paths are calculated.

        **Example:**

        >>> r(ClimateNetwork.SmallTestNetwork().\
                local_correlation_distance_weighted_vulnerability())
        array([ 0.4037, 0.035 , -0.1731, -0.081 , 0.3121, -0.0533])

        :rtype: 1D Numpy array
        :return: the local correlation distance weighted vulnerability
                 sequence.
        """
        self.inv_correlation_distance()
        return self.local_vulnerability('inv_correlation_distance')

    def _weighted_metric(self, attr: str, calc: Callable, metric: str):
        if not self.find_link_attribute(attr):
            self.set_link_attribute(attr, calc())
        return getattr(self, metric)(attr)
