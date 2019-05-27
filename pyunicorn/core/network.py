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

# general TODO:
# - find segfault problem in a.w. shortest path betweenness
# - rename aw... to nsi... (node splitting invariant)
# - implement "corrected" node splitting invariant measures named cnsi...
#   (see paper)
# - implement Newman modularity and iterative division
# - treat type-related ambiguities more thoroughly
#   (flatten(), list(...), astype(...) etc.)

#
#  Import essential packages
#


import sys                          # performance testing
import time
from functools import wraps         # helper function for decorators

import numpy as np                  # array object and fast numerics
from numpy import random
from scipy import linalg            # solvers
from scipy.linalg import matfuncs
from scipy import sparse as sp      # fast sparse matrices
from scipy.sparse.linalg import eigsh, inv, splu

import igraph                       # high performance graph theory tools

from ..utils import mpi             # parallelized computations

from ._ext.numerics import _local_cliquishness_4thorder, \
    _local_cliquishness_5thorder, _cy_mpi_nsi_newman_betweenness, \
    _cy_mpi_newman_betweenness, _nsi_betweenness, \
    _newman_betweenness_badly_cython, _do_nsi_clustering_I, \
    _do_nsi_clustering_II, _do_nsi_hamming_clustering

from ..utils import progressbar     # easy progress bar handling


def nz_coords(matrix):
    """
    Find coordinates of all non-zero entries in a sparse matrix.

    :return: list of coordinates [row,col]
    :rtype:  array([[int>=0,int>=0]])
    """
    return np.array(matrix.nonzero()).T


def cache_helper(self, cat, key, msg, func, *args, **kwargs):
    """
    Cache result of a function in a subdict of :attr:`self.cache`.

    :arg str cat: cache category
    :arg str key: cache key
    :arg str msg: message to be displayed during first calculation
    :arg func func: function to be cached
    """
    # categories can be added on the fly?!?!
    self.cache.setdefault(cat, {})

    if self.cache[cat].setdefault(key) is None:
        if msg is not None and self.silence_level <= 1:
            print('Calculating ' + msg + '...')
        self.cache[cat][key] = func(self, *args, **kwargs)
    return self.cache[cat][key]


def cached_const(cat, key, msg=None):
    """
    Cache result of decorated method in a fixed subdict of :attr:`self.cache`.
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return cache_helper(self, cat, key, msg, func, *args, **kwargs)
        return wrapped
    return wrapper


def cached_var(cat, msg=None):
    """
    Cache result of decorated method in a variable subdict of
    :attr:`self.cache`, specified as first argument to the decorated method.
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, key=None, **kwargs):
            return cache_helper(self, cat, key, msg, func, key, **kwargs)
        return wrapped
    return wrapper


class NetworkError(Exception):
    """
    Used for all exceptions raised by Network.
    """
    def __init__(self, value):
        Exception.__init__(self)
        self.value = value

    def __str__(self):
        return repr(self.value)


#
#  Define class Network
#

class Network:
    """
    A Network is a simple, undirected or directed graph with optional node
    and/or link weights. This class encapsulates data structures and methods to
    represent, generate and analyze such structures.

    Network relies on the package igraph for many of its features, but also
    implements new functionality. Highlights include weighted and directed
    statistical network measures, measures based on random walks, and
    node splitting invariant network measures.

    **Examples:**

    Create an undirected network given the adjacency matrix:

    >>> net = Network(adjacency=[[0,1,0,0,0,0], [1,0,1,0,0,1],
    ...                          [0,1,0,1,1,0], [0,0,1,0,1,0],
    ...                          [0,0,1,1,0,1], [0,1,0,0,1,0]])

    Create an Erdos-Renyi random graph:

    >>> net = Network.ErdosRenyi(n_nodes=100, link_probability=0.05)
    Generating Erdos-Renyi random graph with 100 nodes and probability 0.05...
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, adjacency=None, n_nodes=None, edge_list=None,
                 directed=False, node_weights=None, silence_level=0):
        """
        Return a new directed or undirected Network object
        with given adjacency matrix and optional node weights.

        :type adjacency: square array-like [node,node], or pysparse matrix of
            0s and 1s
        :arg  adjacency: Adjacency matrix of the new network.  Entry [i,j]
            indicates whether node i links to node j.  Its diagonal must be
            zero.  Must be symmetric if directed=False.
        :type n_nodes: int
        :arg  n_nodes: Number of nodes, optional argument when using edge_list
        :type edge_list: array-like list of lists
        :arg  edge_list: Edge list of the new network.  Entries [i,0], [i,1]
            contain the end-nodes of an edge.
        :arg bool directed: Indicates whether the network shall be considered
            as directed. If False, adjacency must be symmetric.
        :type node_weights: 1d numpy array or list [node] of floats >= 0
        :arg  node_weights: Optional array or list of node weights to be used
            for node splitting invariant network measures.  Entry [i] is the
            weight of node i.  (Default: list of ones)
        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        :rtype: :class:`Network` instance
        :return: The new network.
        """

        self.directed = directed
        """(bool) Indicates whether the network is directed."""
        self.silence_level = silence_level
        """(int>=0) higher -> less progress info"""

        if n_nodes is None:
            self.N = 0
            """(int>0) number of nodes"""
        else:
            self.N = n_nodes

        self.n_links = 0
        """(int>0) number of links"""
        self.link_density = 0
        """(0<float<1) proportion of linked node pairs"""

        self.sp_A = None
        """(sparse.csc_matrix([[int,int]]) with entries 0,1)
        Adjacency matrix. A[i,j]=1 indicates a link i -> j. Symmetric if the
        network is undirected."""
        self.sp_dtype = None

        self.graph = None
        """(igraph.Graph) Embedded graph object providing some standard network
        measures."""

        self._node_weights = None
        """(array([int>=0])) array of node weights"""
        self.mean_node_weight = 0
        """mean node weight"""
        self.total_node_weight = 0
        """total node weight"""

        self.cache = {'base': {}, 'nsi': {}, 'paths': {}}
        """(dict) cache of re-usable computation results"""

        if adjacency is not None:
            self._set_adjacency(adjacency)
        elif edge_list is not None:
            self.set_edge_list(edge_list, n_nodes)
        else:
            raise NetworkError("An adjacency matrix or edge list has to be \
                               given to initialize an instance of Network.")

        self._set_node_weights(node_weights)
        self.degree()

    def __str__(self):
        """
        Return a short summary of the network.

        **Example:**

        >>> print(Network.SmallTestNetwork())
        Network: undirected, 6 nodes, 7 links, link density 0.467.

        :rtype: string
        """
        return ('Network: %sdirected, %i nodes, %i links, '
                'link density %.3f.') % ('' if self.directed else 'un', self.N,
                                         self.n_links, self.link_density)

    def __len__(self):
        """
        Return the number of nodes as the 'length'.

        **Example:**

        >>> len(Network.SmallTestNetwork())
        6

        :rtype: int > 0
        """
        return self.N

    def clear_cache(self):
        """
        Clear cache of information that can be recalculated from basic data.
        """
        self.cache['base'] = {}
        self.clear_nsi_cache()
        self.clear_paths_cache()

    def clear_nsi_cache(self):
        """
        Clear cache of information that can be recalculated from basic data
        and depends on the node weights.
        """
        self.cache['nsi'] = {}

    def clear_paths_cache(self):
        """
        Clear cache of path legths for link attributes.
        """
        for attr in self.cache['paths']:
            self.clear_link_attribute(attr)
        self.cache['paths'] = {}

    def copy(self):
        """
        Return a copy of the network.
        """
        return Network(adjacency=self.sp_A, directed=self.directed,
                       node_weights=self.node_weights,
                       silence_level=self.silence_level)

    def undirected_copy(self):
        """
        Return an undirected copy of the network.

        Nodes i and j are linked in the copy if, in the current network, i
        links to j or j links to i or both.

        **Example:**

        >>> net = Network(adjacency=[[0,1],[0,0]], directed=True); print(net)
        Network: directed, 2 nodes, 1 links, link density 0.500.
        >>> print(net.undirected_copy())
        Network: undirected, 2 nodes, 1 links, link density 1.000.

        :rtype: :class:`Network` instance
        """
        return Network(adjacency=self.undirected_adjacency(),
                       directed=False, node_weights=self.node_weights,
                       silence_level=self.silence_level)

    def permuted_copy(self, permutation):
        """
        Return a copy of the network with node numbers rearranged. This
        operation should not change topological information and network
        measures.

        :type permutation: array-like [int]
        :arg permutation: desired permutation of nodes
        :rtype: :class:`Network` instance
        """
        idx = np.array(permutation)
        if (sorted(idx) != np.arange(self.N)).any():
            raise NetworkError("Incorrect permutation indices!")

        return Network(adjacency=self.sp_A[idx][:, idx],
                       node_weights=self.node_weights[idx],
                       directed=self.directed,
                       silence_level=self.silence_level)

    def splitted_copy(self, node=-1, proportion=0.5):
        """
        Return a copy of the network with one node splitted.

        The specified node is split in two interlinked nodes
        which are linked to the same nodes as the original node,
        and the weight is splitted according to the given proportion.

        (This method is useful for testing the node splitting invariance
        of measures since a n.s.i. measure will be the same before and after
        the split.)

        **Example:**

        >>> net = Network.SmallTestNetwork(); print(net)
        Network: undirected, 6 nodes, 7 links, link density 0.467.
        >>> net2 = net.splitted_copy(node=5, proportion=0.2); print(net2)
        Network: undirected, 7 nodes, 9 links, link density 0.429.
        >>> print(net.node_weights); print(net2.node_weights)
        [ 1.5  1.7  1.9  2.1  2.3  2.5]
        [ 1.5  1.7  1.9  2.1  2.3  2.  0.5]

        :type node: int
        :arg  node: The index of the node to be splitted. If negative,
                    N + index is used. The new node gets index N. (Default: -1)

        :type proportion: float from 0 to 1
        :arg  proportion: The splitted node gets a new weight of
                          (1-proportion) * (weight of splitted node),
                          and the new node gets a weight of
                          proportion * (weight of splitted node).
                          (Default: 0.5)

        :rtype: :class:`Network`
        """
        N, A, w = self.N, self.sp_A, self.node_weights
        if node < 0:
            node += N

        new_A = sp.lil_matrix((N+1, N+1))
        new_w = np.zeros(N+1)
        new_A[:N, :N] = A
        # add last row and column
        new_A[:N, N] = A[:, node]
        new_A[N, :N] = A[node, :]
        # connect new node with original
        new_A[node, N] = new_A[N, node] = 1
        # copy and adjust weights
        new_w[:N] = w[:N]
        new_w[N] = proportion * w[node]
        new_w[node] = (1.0 - proportion) * w[node]

        new_NW = Network(adjacency=new_A, directed=self.directed,
                         node_weights=new_w, silence_level=self.silence_level)
        # -- Copy link attributes
        for a in self.graph.es.attributes():
            W = self.link_attribute(a)
            new_W = np.zeros((N+1, N+1))
            new_W[:N, :N] = W
            # add last row and column
            new_W[:N, N] = W[:, node]
            new_W[N, :N] = W[node, :]
            # assign weight between new node and original and for self loop
            new_W[node, N] = new_W[N, node] = new_W[N, N] = W[node, node]
            new_NW.set_link_attribute(a, new_W)
        # --
        return new_NW

    @property
    def adjacency(self):
        """
        Return the (possibly non-symmetric) adjacency matrix as a dense matrix.

        **Example:**

        >>> r(Network.SmallTestNetwork().adjacency)
        array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])

        :rtype: square numpy array [node,node] of 0s and 1s
        """
        return self.sp_A.A

    def _set_adjacency(self, adjacency):
        """
        Set a new adjacency matrix.

        **Example:**

        >>> net = Network.SmallTestNetwork(); print(net)
        Network: undirected, 6 nodes, 7 links, link density 0.467.
        >>> net.adjacency = [[0,1],[1,0]]; print(net)
        Network: undirected, 2 nodes, 1 links, link density 1.000.

        :type adjacency: square array-like [[0|1]]
        :arg  adjacency: Entry [i,j] indicates whether node i links to node j.
            Its diagonal must be zero. Symmetric if the network is undirected.
        """
        # convert to sparse matrix
        self.sp_A = None
        if not sp.issparse(adjacency):
            adjacency = sp.csc_matrix(np.array(adjacency))

        # ensure square matrix
        M, N = adjacency.shape
        if M != N:
            raise NetworkError("Adjacency must be square!")
        self.N = N
        if N < 32767:
            self.sp_dtype = np.int16
        else:
            self.sp_dtype = np.int32
        self.sp_A = adjacency.tocsc().astype(self.sp_dtype)

        # calculate graph attributes
        edges = nz_coords(adjacency)
        self.n_links = edges.shape[0]
        self.link_density = 1.0 * self.n_links / N / (N - 1)
        if not self.directed:
            self.n_links //= 2

        # create graph object
        self.graph = igraph.Graph(n=N, edges=list(edges),
                                  directed=self.directed)
        self.graph.simplify()
        Network.clear_cache(self)

    @adjacency.setter
    def adjacency(self, adjacency):
        self._set_adjacency(adjacency)

    def set_edge_list(self, edge_list, n_nodes=None):
        """
        Reset network from an edge list representation.

        .. note::
           Assumes that nodes are numbered by natural numbers from 0 to N-1
           without gaps!

        **Example:**

        :type edge_list: array-like [[int>=0,int>=0]]
        :arg  edge_list: [[i,j]] for edges i -> j
        """
        #  Convert to Numpy array and get number of nodes
        edges = np.array(edge_list)

        if n_nodes is None:
            N = edges.max() + 1
        else:
            N = n_nodes

        #  Symmetrize if undirected network
        if not self.directed:
            edges = np.append(edges, edges[:, [1, 0]], axis=0)

        #  Create sparse adjacency matrix from edge list
        sp_A = sp.coo_matrix(
            (np.ones_like(edges.T[0]), tuple(edges.T)), shape=(N, N))

        #  Set sparse adjacency matrix
        self.adjacency = sp_A

    @property
    def node_weights(self):
        """(array([int>=0])) array of node weights"""
        return self._node_weights

    def _set_node_weights(self, weights):
        """
        Set the node weights to be used for node splitting invariant network
        measures.

        **Example:**

        >>> net = Network.SmallTestNetwork(); print(net.node_weights)
        [ 1.5  1.7  1.9  2.1  2.3  2.5]
        >>> net.node_weights = [1,1,1,1,1,1]; print(net.node_weights)
        [ 1.  1.  1.  1.  1.  1.]

        :type weights: array-like [float>=0]
        :arg  weights: array-like [node] of weights (default: [1...1])
        """
        N = self.N
        self.clear_nsi_cache()

        if weights is None:
            w = np.ones(N, dtype=np.float)
        elif len(weights) != N:
            raise NetworkError("Incorrect number of node weights!")
        else:
            w = np.array(weights, dtype=np.float)

        self._node_weights = w
        self.mean_node_weight = w.mean()
        self.total_node_weight = w.sum()

    @node_weights.setter
    def node_weights(self, node_weights):
        self._set_node_weights(node_weights)

    def sp_Aplus(self):
        """A^+ = A + Id. matrix used in n.s.i. measures"""
        return self.sp_A + sp.identity(self.N, dtype=self.sp_dtype)

    def sp_diag_w(self):
        """Sparse diagonal matrix of node weights"""
        return sp.diags([self.node_weights], [0],
                        shape=(self.N, self.N), format='csc')

    def sp_diag_w_inv(self):
        """Sparse diagonal matrix of inverse node weights"""
        return sp.diags([1 / self.node_weights], [0],
                        shape=(self.N, self.N), format='csc')

    def sp_diag_sqrt_w(self):
        """Sparse diagonal matrix of square roots of node weights"""
        return sp.diags([np.sqrt(self.node_weights)], [0],
                        shape=(self.N, self.N), format='csc')

    #
    #  Load and save Network object
    #

    def save(self, filename, fileformat=None, *args, **kwds):
        """
        Save the Network object to a file.

        Unified writing function for graphs. Relies on and partially extends
        the corresponding igraph function. Refer to igraph documentation for
        further details on the various writer methods for different formats.

        This method tries to identify the format of the graph given in
        the first parameter (based on extension) and calls the corresponding
        writer method.

        Existing node and link attributes/weights are also stored depending
        on the chosen file format. E.g., the formats GraphML and gzipped
        GraphML are able to store both node and link weights.

        The remaining arguments are passed to the writer method without
        any changes.

        :arg str filename: The name of the file where the Network object is to
            be stored.
        :arg str fileformat: the format of the file (if one wants to override
            the format determined from the filename extension, or the filename
            itself is a stream). ``None`` means auto-detection. Possible values
            are: ``"ncol"`` (NCOL format), ``"lgl"`` (LGL format),
            ``"graphml"``, ``"graphmlz"`` (GraphML and gzipped GraphML format),
            ``"gml"`` (GML format), ``"dot"``, ``"graphviz"`` (DOT format, used
            by GraphViz), ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"``
            (DIMACS format), ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge
            list), ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python
            pickled format), ``"svg"`` (Scalable Vector Graphics).
        """
        #  Store node weights as an igraph vertex attribute for saving
        #  Link attributes/weights are stored automatically if they exist
        if self.node_weights is not None:
            self.graph.vs.set_attribute_values(
                "node_weight_nsi", list(self.node_weights))

        self.graph.write(f=filename, format=fileformat, *args, **kwds)

    @staticmethod
    def Load(filename, fileformat=None, silence_level=0, *args, **kwds):
        """
        Return a Network object stored in a file.

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

        :arg str filename: The name of the file containing the Network object.
        :arg str fileformat: the format of the file (if known in advance).
          ``None`` means auto-detection. Possible values are: ``"ncol"`` (NCOL
          format), ``"lgl"`` (LGL format), ``"graphml"``, ``"graphmlz"``
          (GraphML and gzipped GraphML format), ``"gml"`` (GML format),
          ``"net"``, ``"pajek"`` (Pajek format), ``"dimacs"`` (DIMACS format),
          ``"edgelist"``, ``"edges"`` or ``"edge"`` (edge list),
          ``"adjacency"`` (adjacency matrix), ``"pickle"`` (Python pickled
          format).
        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        :rtype: Network object
        :return: :class:`Network` instance.
        """
        #  Load to igraph Graph object
        graph = igraph.Graph.Read(f=filename, format=fileformat, *args, **kwds)
        return Network.FromIGraph(graph=graph, silence_level=silence_level)

    #
    #  Graph generation methods
    #

    @staticmethod
    def FromIGraph(graph, silence_level=0):
        """
        Return a :class:`Network` object given an igraph Graph object.

        :type graph: igraph Graph object
        :arg graph: The igraph Graph object to be converted.

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.

        :rtype: :class:`Network` instance
        :return: :class:`Network` object.
        """
        #  Get number of nodes
        N = len(graph.vs)

        #  Get directedness
        directed = graph.is_directed()

        #  Extract edge list
        edges = np.array(graph.get_edgelist())

        #  Symmetrize if undirected network
        if not directed:
            edges = np.append(edges, edges[:, [1, 0]], axis=0)

        #  Create sparse adjacency matrix from edge list
        sp_A = sp.coo_matrix(
            (np.ones_like(edges.T[0]), tuple(edges.T)), shape=(N, N))

        #  Extract node weights
        if "node_weight_nsi" in graph.vs.attribute_names():
            node_weights = np.array(
                graph.vs.get_attribute_values("node_weight_nsi"))
        else:
            node_weights = None

        net = Network(adjacency=sp_A, directed=directed,
                      node_weights=node_weights, silence_level=silence_level)

        #  Overwrite igraph Graph object in Network instance to restore link
        #  attributes/weights
        net.graph = graph
        net.clear_paths_cache()

        return net

    @staticmethod
    def SmallTestNetwork():
        """
        Return a 6-node undirected test network with node weights.

        The network looks like this::

                3 - 1
                |   | \\
            5 - 0 - 4 - 2

        The node weights are [1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
        a typical node weight for corrected n.s.i. measures would be 2.0.

        :rtype: Network instance
        """
        nw = Network(adjacency=[[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0],
                                [0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                     directed=False,
                     node_weights=[1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
                     silence_level=1)
        link_weights = np.array([[0, 0, 0, 1.3, 2.5, 1.1],
                                 [0, 0, 2.3, 2.9, 2.7, 0],
                                 [0, 2.3, 0, 0, 1.5, 0],
                                 [1.3, 2.9, 0, 0, 0, 0],
                                 [2.5, 2.7, 1.5, 0, 0, 0],
                                 [1.1, 0, 0, 0, 0, 0]])
        nw.set_link_attribute("link_weights", link_weights)
        return nw

    @staticmethod
    def SmallDirectedTestNetwork():
        """
        Return a 6-node directed test network with node and edge weights.

        The node weights are [1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
        a typical node weight for corrected n.s.i. measures would be 2.0.

        :rtype: Network instance
        """
        nw = Network(adjacency=[[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                [1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                     directed=True,
                     node_weights=[1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
                     silence_level=1)
        nw.set_link_attribute("link_weights", np.array([[0, 1.3, 0, 2.5, 0, 0],
                                                        [0, 0, 1.9, 0, 1.0, 0],
                                                        [0, 0, 0, 0, 0, 0],
                                                        [0, 3.0, 0, 0, 0, 0],
                                                        [2.1, 0, 2.7, 0, 0, 0],
                                                        [1.5, 0, 0, 0, 0, 0]]))
        return nw

    @staticmethod
    def ErdosRenyi(n_nodes=100, link_probability=None, n_links=None,
                   silence_level=0):
        """
        Return a new undirected Erdos-Renyi random graph
        with a given number of nodes and linking probability.

        The expected link density equals this probability.

        **Example:**

        >>> print(Network.ErdosRenyi(n_nodes=10, n_links=18))
        Generating Erdos-Renyi random graph with 10 nodes and 18 links...
        Network: undirected, 10 nodes, 18 links, link density 0.400.

        :type n_nodes: int > 0
        :arg  n_nodes: Number of nodes. (Default: 100)

        :type link_probability: float from 0 to 1, or None
        :arg  link_probability: If not None, each pair of nodes is
                                independently linked with this probability.
                                (Default: None)

        :type n_links: int > 0, or None
        :arg  n_links: If not None, this many links are assigned at random.
                       Must be None if link_probability is not None.
                       (Default: None)

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.

        :rtype: :class:`Network` instance
        """
        if link_probability is not None and n_links is None:
            if silence_level < 1:
                print(f"Generating Erdos-Renyi random graph with "
                      f"{n_nodes} nodes and probability {link_probability}...")

            graph = igraph.Graph.Erdos_Renyi(n=n_nodes, p=link_probability)
            #  Get edge list
            edge_list = graph.get_edgelist()

        elif link_probability is None and n_links is not None:
            if silence_level < 1:
                print(f"Generating Erdos-Renyi random graph with "
                      f"{n_nodes} nodes and {n_links} links...")

            graph = igraph.Graph.Erdos_Renyi(n=n_nodes, m=n_links)
            #  Get edge list
            edge_list = graph.get_edgelist()

        else:
            return None

        return Network(n_nodes=n_nodes, edge_list=edge_list, directed=False,
                       silence_level=silence_level)

    @staticmethod
    def BarabasiAlbert_igraph(n_nodes=100, n_links_each=5, silence_level=0):
        """
        Return a new undirected Barabasi-Albert random graph generated by
        igraph.

        CAUTION: actual no. of new links can be smaller than n_links_each
        because neighbours are drawn with replacement and graph is then
        simplified.

        The given number of nodes are added in turn to the initially empty node
        set, and each new node is linked to the given number of existing nodes.
        The resulting link density is approx. 2 * ``n_links_each``/``n_nodes``.

        **Example:** Generating a random tree:

        >>> net = Network.BarabasiAlbert_igraph(n_nodes=100, n_links_each=1)
        >>> print(net.link_density)
        0.02

        :type n_nodes: int > 0
        :arg  n_nodes: Number of nodes. (Default: 100)
        :type n_links_each: int > 0
        :arg  n_links_each: Number of links to existing nodes each new node
                            gets during construction. (Default: 5)
        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        :rtype: :class:`Network` instance
        """
        graph = igraph.Graph.Barabasi(n=n_nodes, m=n_links_each)

        # Remove self-loops and multiple links, this does of course change the
        # actual degree sequence of the generated graph, but just slightly
        graph.simplify()
        edge_list = graph.get_edgelist()

        return Network(edge_list=edge_list, directed=False,
                       silence_level=silence_level)

    @staticmethod
    def BarabasiAlbert(n_nodes=100, n_links_each=5, silence_level=0):
        """
        Return a new undirected Barabasi-Albert random graph
        with exactly n_links_each * (n_nodes-n_links_each) links.

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        """
        # start with 1+m nodes of which the first is linked to the rest
        N, m = n_nodes, n_links_each
        A = sp.lil_matrix((N, N), dtype=np.int8)
        A[0, 1:1+m] = 1
        A[1:1+m, 0] = 1

        # inverse cum. degree distribution
        targets, last_child = np.zeros(2*m*(N-m), dtype=np.int8), np.zeros(N)
        targets[m:2*m] = range(1, 1+m)
        n_targets = 2*m
        for j in range(1+m, N):
            for it in range(m):
                while True:
                    i = targets[int(random.uniform(low=0, high=n_targets))]
                    if last_child[i] != j:
                        break
                A[i, j] = A[j, i] = 1
                targets[n_targets + it] = i
                last_child[i] = j
            targets[n_targets + m: n_targets + 2*m] = j
            n_targets += 2*m

        return Network(A, silence_level=silence_level)

    @staticmethod
    def GrowPreferentially_old(n_nodes=100, m=2, silence_level=0):
        """
        EXPERIMENTAL: Return a random network grown with preferential weight
        increase and preferential attachment.

        Return a random network grown as follows: Starting with a clique
        of m+1 unit weight nodes, iteratively add a unit weight node and then
        m times increase the weight of an existing node by one unit, for
        n=m+2...N. Choose the growing node with probabilities proportional to
        the node's weight. After each node addition or weight increase, add one
        link from the respective node to another node, chosen with probability
        proportional to that node's n.s.i. degree.

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        """
        N = n_nodes
        w, A = np.zeros(N, int), sp.lil_matrix((N, N))
        nbs = [[] for i in range(N)]

        # start with m+1 fully connected nodes
        w[:m+1] = 1

        # total weight now and in the end
        last_W = m+1
        # this is also approx. the total no. of links in the end!
        W = (m+1) * (N-m)
        inc_target = np.zeros(W, "int")  # inverse cum. w distribution
        inc_target[:m+1] = range(m+1)

        # max. possible w before step n: 1 + m(n-m-2),
        # so the addition in step n increases total n.s.i. degree by at most
        # 3 + m(n-m-2) <= nm,
        # each of the m weight increases increases it by at most n, totalling
        # mn,
        # and each of the m additional links increases it by at most
        # 2 * (1 + m(n-m-2) + m), totalling <= 2nm, all totalling <= 2nm^2
        # total n.s.i. degree now and max. in the end:
        # last_Kstar = (m+1)**2
        last_Kstar = (m+1)*m
        max_Kstar = N**2 * (m+1)**2
        # inverse cum. k* distribution
        link_target = np.zeros(max_Kstar, "int")

        for i in range(m+1):
            for j in range(i):
                A[i, j] = A[j, i] = 1
            nbs[i] = range(m+1)
            nbs[i].remove(i)
            link_target[(m+1)*i:(m+1)*(i+1)] = i

        for n in range(m+2, N+1):
            # add node n-1 with unit weight:
            w[n-1] = 1
            inc_target[last_W] = n-1
            last_W += 1
            # link it to some i:
            i = int(link_target[int(random.uniform(last_Kstar))])
            print("n", n, "i", i)
            A[i, n-1] = A[n-1, i] = 1
            nbs[n-1] = [i]
            nbs[i].append(n-1)
            link_target[last_Kstar] = i
            # link_target[last_Kstar+1:last_Kstar+2+w[i]] = n-1
            # last_Kstar += 2+w[i]
            link_target[last_Kstar+1] = n-1
            last_Kstar += 2

            for jj in range(m):
                # increase weight of some j not already linked to all:
                j = int(inc_target[int(random.uniform(last_W))])
                while len(nbs[j]) == n-1:
                    print(" not j", j)
                    j = int(inc_target[int(random.uniform(last_W))])
                w[j] += 1
                print(" jj", jj, "j", j, "w[j]", w[j])
                inc_target[last_W] = j
                last_W += 1
                # link_target[last_Kstar] = j
                # last_Kstar += 1
                # for i in nbs[j]:
                #     print("  i", i)
                #     link_target[last_Kstar] = i
                #     last_Kstar += 1

                # link it to some i not already linked to it:
                i = int(link_target[int(random.uniform(last_Kstar))])
                while i == j or A[i, j] == 1:
                    # print("  not i",i)
                    i = int(link_target[int(random.uniform(last_Kstar))])
                A[i, j] = A[j, i] = 1
                nbs[j].append(i)
                nbs[i].append(j)
                # print("  i",i,"nbs[i]",nbs[i],"nbs[j]",nbs[j])
                # link_target[last_Kstar:last_Kstar+w[j]] = i
                # last_Kstar += w[j]
                # link_target[last_Kstar:last_Kstar+w[i]] = j
                # last_Kstar += w[i]
                link_target[last_Kstar] = i
                link_target[last_Kstar+1] = j
                last_Kstar += 2

        del nbs, link_target, inc_target
        return Network(A, node_weights=w, silence_level=silence_level)

    @staticmethod
    def GrowPreferentially(n_nodes=100, n_growths=1, n_links_new=1,
                           n_links_old=1, nsi=True, preferential_exponent=1,
                           n_initials=1, silence_level=0):
        """
        EXPERIMENTAL: Return a random network grown with preferential weight
        increase and n.s.i. preferential attachment.

        Return a random network grown as follows: Starting with a clique
        of 2*n_links_new+1 unit weight nodes, iteratively add a unit weight
        node, connect it with n_links_new different existing nodes chosen
        with probabilities proportional to their current n.s.i. degree, then
        increase the weights of n_growths nodes chosen with probabilities
        proportional to their current weight (with replacement), then add
        n_links_old new links between pairs of nodes chosen with probabilities
        proportional to their current weight.

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.
        """
        N = n_nodes
        w, A = np.zeros(N, dtype=int), sp.lil_matrix((N, N))
        nbs = [[] for i in range(N)]
        inc_target = list(range(n_initials))

        if nsi:
            kstar = np.zeros(N)
            link_prob = np.zeros(N)  # w * kstar

            w[:n_initials] = 1
            link_prob[:n_initials] = 1
            total_link_prob = link_prob.sum()

            def _link_target():
                thd = random.uniform(low=0, high=total_link_prob)
                i = 0
                cum = link_prob[0]
                while cum < thd:
                    i += 1
                    cum += link_prob[i]
                return i

            progress = progressbar.ProgressBar(maxval=N).start()
            for j in range(n_initials, N):
                # add node j with unit weight:
                link_prob[j] = kstar[j] = w[j] = 1
                total_link_prob += 1
                inc_target.append(j)
                # link it to some i's:
                for _ in range(n_links_new):
                    i = _link_target()
                    # print(j,i)
                    while i == j:
                        # print("not i",i)
                        i = _link_target()
                    if A[i, j]:
                        continue
                    # print("j", j, "i", i)
                    A[i, j] = A[j, i] = 1
                    nbs[i].append(j)
                    nbs[j] = [i]
                    total_link_prob -= link_prob[i] + link_prob[j]
                    kstar[i] += w[j]
                    kstar[j] += w[i]
                    link_prob[i] = w[i] * kstar[i]**preferential_exponent
                    link_prob[j] = w[j] * kstar[j]**preferential_exponent
                    total_link_prob += link_prob[i] + link_prob[j]
                # print(total_link_prob, link_prob.sum())

                for _ in range(n_growths):
                    # increase weight of some i:
                    i = inc_target[int(
                        random.uniform(low=0, high=len(inc_target)))]
                    # print(i,inc_target)
                    total_link_prob -= link_prob[nbs[i]].sum() + link_prob[i]
                    w[i] += 1
                    inc_target.append(i)
                    kstar[i] += 1
                    kstar[nbs[i]] += 1
                    link_prob[i] = w[i] * kstar[i]**preferential_exponent
                    link_prob[nbs[i]] = \
                        w[nbs[i]] * kstar[nbs[i]]**preferential_exponent
                    total_link_prob += link_prob[nbs[i]].sum() + link_prob[i]
                    # print(" ii",ii,"i",i,"w[i]",w[i])
                # print(total_link_prob, link_prob.sum())
                for ii in range(n_links_old):
                    # j2 = _link_target()
                    j2 = inc_target[int(
                        random.uniform(low=0, high=len(inc_target)))]
                    i = _link_target()
                    while i == j2:
                        i = _link_target()
                    if A[i, j2]:
                        continue
                    A[i, j2] = A[j2, i] = 1
                    nbs[i].append(j2)
                    nbs[j2].append(i)
                    total_link_prob -= link_prob[i] + link_prob[j2]
                    kstar[i] += w[j2]
                    kstar[j2] += w[i]
                    link_prob[i] = w[i] * kstar[i]**preferential_exponent
                    link_prob[j2] = w[j2] * kstar[j2]**preferential_exponent
                    total_link_prob += link_prob[i] + link_prob[j2]
                # print(total_link_prob, link_prob.sum())

                if j % 10:
                    progress.update(j)

            progress.finish()

        else:
            link_target = []

            # start with (2*n_links_new+1) fully connected nodes:
            n_initials = n_links_new+2*n_links_old+2
            # max(n_links_new+2*n_links_old+2, n_growths-1)
            w[:n_initials] = 1

            for i in range(n_initials):
                for j in range(i):
                    if min(i-j, j+n_initials-i) <= \
                            np.ceil((n_links_new + n_links_old)/2.0):
                        A[i, j] = A[j, i] = 1
                        nbs[i].append(j)
                        nbs[j].append(i)
                link_target += [i for _ in range(n_links_new + n_links_old)]

            # last_grown = np.zeros(N)
            for j in range(n_initials, N):
                # add node j with unit weight:
                w[j] = 1
                inc_target.append(j)
                link_target.append(j)
                # link it to some i's:
                for ii in range(n_links_new):
                    i = int(link_target[int(
                        random.uniform(low=0, high=len(link_target)))])
                    while i == j or A[i, j] == 1:
                        # print("not i",i)
                        i = int(link_target[int(
                            random.uniform(low=0, high=len(link_target)))])
                    # print("j", j, "i", i)
                    A[i, j] = A[j, i] = 1
                    nbs[j] = [i]
                    nbs[i].append(j)
                    link_target += [j for iii in range(w[i])] + [i]
                for ii in range(n_growths):
                    # increase weight of some i:
                    i = int(inc_target[int(
                        random.uniform(low=0, high=len(inc_target)))])
                    # while last_grown[i] == j:
                    #    i = int(inc_target[int(
                    #        random.uniform(len(inc_target)))])
                    # last_grown[i] = j
                    w[i] += 1
                    # print(" ii",ii,"i",i,"w[i]",w[i])
                    inc_target.append(i)
                    link_target += nbs[i] + [i]
                for ii in range(n_links_old):
                    # j2 = int(inc_target[int(
                    #      random.uniform(len(inc_target)))])
                    j2 = int(link_target[int(
                        random.uniform(low=0, high=len(link_target)))])
                    # i = int(inc_target[int(
                    #     random.uniform(len(inc_target)))])
                    i = int(link_target[int(
                        random.uniform(low=0, high=len(link_target)))])
                    while i == j2 or A[i, j2] == 1:
                        # i = int(inc_target[int(
                        #     random.uniform(len(inc_target)))])
                        i = int(link_target[int(
                            random.uniform(low=0, high=len(link_target)))])
                    A[i, j2] = A[j2, i] = 1
                    nbs[j2].append(i)
                    nbs[i].append(j2)
                    link_target += [j2 for iii in range(w[i])] + \
                        [i for iii in range(w[j2])]

            del link_target

        del nbs, inc_target
        return Network(A, node_weights=w, silence_level=silence_level)

    @staticmethod
    def GrowWeights(n_nodes=100, n_initials=1, exponent=1,
                    mode="exp",
                    split_prob=.01,  # for exponential model
                    split_weight=100,  # for reciprocal model
                    beta=1.0, n_increases=1e100):
        """
        EXPERIMENTAL
        """
        N = n_nodes
        w = np.zeros(N)
        inc_prob = np.zeros(N)
        w[:n_initials] = 1
        inc_prob[:n_initials] = 1
        total_inc_prob = inc_prob.sum()
        hold_prob = 1 - split_prob

        def _inc_target():
            thd = random.uniform(low=0, high=total_inc_prob)
            i = 0
            cum = inc_prob[0]
            while cum < thd:
                i += 1
                cum += inc_prob[i]
            return i

        this_N = n_initials
        progress = progressbar.ProgressBar(maxval=N).start()
        it = 0
        while this_N < N and it < n_increases:
            it += 1
            i = _inc_target()
            total_inc_prob -= inc_prob[i]
            w[i] += 1
            inc_prob[i] = w[i]**exponent
            total_inc_prob += inc_prob[i]
            if (mode == "exp" and random.uniform() > hold_prob**w[i]) or \
                    (mode == "rec" and random.uniform()
                     < w[i]*1.0/(split_weight+w[i])):  # reciprocal
                # split i into i,this_N:
                total_inc_prob -= inc_prob[i]
                w[this_N] = w[i]*random.beta(beta, beta)
                w[i] -= w[this_N]
                inc_prob[this_N] = w[this_N]**exponent
                inc_prob[i] = w[i]**exponent
                total_inc_prob += inc_prob[this_N] + inc_prob[i]
                this_N += 1
            if this_N % 10:
                progress.update(this_N)

        progress.finish()
        return w

    @staticmethod
    def ConfigurationModel(degrees, silence_level=0):
        """
        Return a new configuration model random graph
        with a given degree sequence.

        **Example:** Generate a network of 1000 nodes with degree 3 each:

        >>> net = Network.ConfigurationModel([3 for _ in range(0,1000)])
        Generating configuration model random graph
        from given degree sequence...
        >>> print(int(round(net.degree().mean())))
        3

        :type degrees: 1d numpy array or list [node]
        :arg  degrees: Array or list of degrees wanted.

        :type silence_level: int >= 0
        :arg  silence_level: The higher, the less progress info is output.

        :rtype: :class:`Network` instance
        """
        print("Generating configuration model random graph\n"
              + "from given degree sequence...")

        graph = igraph.Graph.Degree_Sequence(out=list(degrees))

        #  Remove self-loops and multiple links, this does of course change the
        #  actual degree sequence of the generated graph, but just slightly
        graph.simplify()

        #  Extract edge list
        edge_list = graph.get_edgelist()

        network = Network(edge_list=edge_list, directed=False,
                          silence_level=silence_level)

        return network

    @staticmethod
    def WattsStrogatzGraph(N, k, p):
        """
        Return a Watt-Strogatz random graph.

        Reference: [Watts1998]_

        :type N: int > 0
        :arg N: Number of nodes.

        :type k: int > 0
        :arg k: Each node is connected to k nearest neighbors in ring topology.

        :type p: float > 0
        :arg p: Probability of rewiring each edge.
        """
        raise NetworkError("Not implemented yet...")

    def randomly_rewire(self, iterations):
        """
        Randomly rewire the network, preserving the degree sequence.

        **Example:** Generate a network of 100 nodes with degree 5 each:

        >>> net = Network.SmallTestNetwork(); print(net)
        Network: undirected, 6 nodes, 7 links, link density 0.467.
        >>> net.randomly_rewire(iterations=10); print(net)
        Randomly rewiring the network,preserving the degree sequence...
        Network: undirected, 6 nodes, 7 links, link density 0.467.

        :type iterations: int > 0
        :arg iterations: Number of iterations. In each iteration, two randomly
            chosen links a--b and c--d for which {a,c} and {b,d} are not
            linked, are replaced by the links a--c and b--d.
        """
        # TODO: verify that it is indeed as described above.
        if self.silence_level <= 1:
            print("Randomly rewiring the network,"
                  + "preserving the degree sequence...")

        # rewire embedded igraph.Graph:
        self.graph.rewire(iterations)

        # update all data that depends on rewired edge list:
        self.set_edge_list(self.graph.get_edgelist())

    def edge_list(self):
        """
        Return the network's edge list.

        **Example:**

        >>> print(Network.SmallTestNetwork().edge_list()[:8])
        [[0 3] [0 4] [0 5] [1 2] [1 3] [1 4] [2 1] [2 4]]

        :rtype: array-like (numpy matrix or list of lists/tuples)
        """
        return nz_coords(self.sp_A)

    # TODO: deprecate this and rather use undirected_copy()
    def undirected_adjacency(self):
        """
        Return the adjacency matrix of the undirected version of the network
        as a dense numpy array.
        Entry [i,j] is 1 if i links to j or j links to i.

        **Example:**

        >>> net = Network(adjacency=[[0,1],[0,0]], directed=True)
        >>> print(net.undirected_adjacency().A)
        [[0 1] [1 0]]

        :rtype: array([[0|1]])
        """
        return self.sp_A.maximum(self.sp_A.T)

    def laplacian(self, direction="out", link_attribute=None):
        """
        Return the (possibly non-symmetric) dense Laplacian matrix.

        **Example:**

        >>> r(Network.SmallTestNetwork().laplacian())
        array([[ 3,  0,  0, -1, -1, -1], [ 0,  3, -1, -1, -1,  0],
               [ 0, -1,  2,  0, -1,  0], [-1, -1,  0,  2,  0,  0],
               [-1, -1, -1,  0,  3,  0], [-1,  0,  0,  0,  0,  1]])

        :arg str direction: This argument is ignored for undirected graphs.
            "out" - out-degree on diagonal of laplacian
            "in"  - in-degree on diagonal of laplacian
        :arg str link_attribute: name of link attribute to be used
        :rtype: square array [node,node] of ints
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        if link_attribute is None:
            if self.directed:
                if direction == "out":
                    diagonal = self.outdegree()
                elif direction == "in":
                    diagonal = self.indegree()
                else:
                    print("ERROR: argument direction of Network.laplacian "
                          "can only take values <<in>> or <<out>>.")
            else:
                diagonal = self.degree()

            return np.diag(diagonal, 0) - self.adjacency
        else:
            raise NotImplementedError("Only implemented for link_attribute \
                                      =None.")

    def nsi_laplacian(self):
        """
        Return the n.s.i. Laplacian matrix (undirected networks only!).

        **Example:**

        >>> Network.SmallTestNetwork().nsi_laplacian()
        Calculating n.s.i. degree...
        array([[ 6.9,  0. ,  0. , -2.1, -2.3, -2.5],
               [ 0. ,  6.3, -1.9, -2.1, -2.3,  0. ],
               [ 0. , -1.7,  4. ,  0. , -2.3,  0. ],
               [-1.5, -1.7,  0. ,  3.2,  0. ,  0. ],
               [-1.5, -1.7, -1.9,  0. ,  5.1,  0. ],
               [-1.5,  0. ,  0. ,  0. ,  0. ,  1.5]])

        :rtype: square array([[float]])
        """
        return (self.sp_nsi_diag_k() - self.sp_Aplus() * self.sp_diag_w()).A

    #
    #  Calculate frequency and cumulative distributions
    #

    # TODO: add sensible default for n_bins depending on len(values)
    @staticmethod
    def _histogram(values, n_bins, interval=None):
        """
        Return a normalized histogram of a list of values,
        its statistical error, and the lower bin boundaries.

        **Example:** Get the relative frequencies only:

        >>> r(Network._histogram(
        ...     values=[1,2,13], n_bins=3, interval=(0,30))[0])
        array([ 0.6667,  0.3333,  0. ])

        :type values: 1d array or list of floats
        :arg  values: The values whose distribution is wanted.

        :type n_bins: int > 0
        :arg  n_bins: Number of bins to be used for the histogram.

        :type interval: tuple (float,float), or None
        :arg  interval: Optional interval to use. If None, the minimum and
                        maximum values are used. (Default: None)

        :rtype:  tuple (list,list,list)
        :return: A list of relative bin frequencies, a list of estimated
                 statistical errors, and a list of lower bin boundaries.
        """
        hist = np.histogram(values, bins=n_bins, range=interval, density=False)
        frequencies = hist[0].astype('float64')
        bin_starts = hist[1][:-1]

        # Calculate statistical error given by 1/n_i per bin i,
        # where n_i is the number of samples per bin
        error = np.zeros(n_bins)
        error[frequencies != 0] = 1 / np.sqrt(frequencies[frequencies != 0])
        # FIXME: this seems not correct. If the true probability for the bin
        # is p_i, the variance of  n_i / N  is  p_i * (1 - p_i) / N
        # which can be estimated from n_i by  n_i * (N - n_i) / N**3

        #  Normalize frequencies and error
        rel_freqs = frequencies / frequencies.sum()
        error /= frequencies.sum()

        return (rel_freqs, error, bin_starts)

    @staticmethod
    def _cum_histogram(values, n_bins, interval=None):
        """
        Return a normalized cumulative histogram of a list of values,
        and the lower bin boundaries.

        **Example:** Get the relative frequencies only:

        >>> r(Network._cum_histogram(
        ...     values=[1,2,13], n_bins=3, interval=(0,30))[0])
        array([ 1. ,  0.3333,  0. ])

        :type values: 1d array or list of floats
        :arg  values: The values whose distribution is wanted.

        :type n_bins: int > 0
        :arg  n_bins: Number of bins to be used for the histogram.

        :type interval: tuple (float,float), or None
        :arg  interval: Optional range to use. If None, the minimum and maximum
                        values are used. (Default: None)

        :rtype:  tuple (list,list)
        :return: A list of cumulative relative bin frequencies
                 (entry [i] is the sum of the frequencies of all bins j >= i),
                 and a list of lower bin boundaries.
        """
        (rel_freqs, _, bin_starts) = \
            Network._histogram(values=values, n_bins=n_bins, interval=interval)
        cum_rel_freqs = rel_freqs[::-1].cumsum()[::-1]
        return (cum_rel_freqs, bin_starts)

    #
    #  Methods working with node attributes
    #

    def set_node_attribute(self, attribute_name, values):
        """
        Add a node attribute.

        Examples for node attributes/weights are degree or betweenness.

        :arg str attribute_name: The name of the node attribute.

        :type values: 1D Numpy array [node]
        :arg values: The node attribute sequence.
        """
        # TODO: add example

        #  Test whether the data vector has the same length as the number of
        #  nodes in the graph.
        if len(values) == self.N:
            #  Add node property to igraph Graph object
            self.graph.vs.set_attribute_values(attrname=attribute_name,
                                               values=values)
        else:
            print("Error! Vertex attribute data array", attribute_name,
                  "has to have the same length as the number of nodes "
                  "in the graph.")

    def node_attribute(self, attribute_name):
        """
        Return a node attribute.

        Examples for node attributes/weights are degree or betweenness.

        :arg str attribute_name: The name of the node attribute.

        :rtype: 1D Numpy array [node]
        :return: The node attribute sequence.
        """
        # TODO: add example
        return np.array(self.graph.vs.get_attribute_values(attribute_name))

    def del_node_attribute(self, attribute_name):
        """
        Delete a node attribute.

        :arg str attribute_name: Name of node attribute to be deleted.
        """
        # TODO: add example
        self.graph.es.__delattr__(attribute_name)

    #
    #  Methods working with link attributes
    #

    # TODO: verify whether return types are list or numpy array

    def average_link_attribute(self, attribute_name):
        """
        For each node, return the average of a link attribute
        over all links of that node.

        :arg str attribute_name: Name of link attribute to be used.

        :rtype: 1d numpy array [node] of floats
        """
        # TODO: add example
        return self.link_attribute(attribute_name).mean(axis=1)

    def link_attribute(self, attribute_name):
        """
        Return the values of a link attribute.

        :arg str attribute_name: Name of link attribute to be used.

        :rtype:  square numpy array [node,node]
        :return: Entry [i,j] is the attribute of the link from i to j.
        """
        # TODO: add example
        # TODO: test this for directed graphs
        #  Initialize weights array
        weights = np.zeros((self.N, self.N))

        if self.directed:
            for e in self.graph.es:
                weights[e.tuple] = e[attribute_name]
        #  Symmetrize if graph is undirected
        else:
            for e in self.graph.es:
                weights[e.tuple] = e[attribute_name]
                weights[e.tuple[1], e.tuple[0]] = e[attribute_name]

        return weights

    def clear_link_attribute(self, attribute_name):
        """
        Clear cache of a link attribute.

        :arg str attribute_name: name of link attribute
        """
        if attribute_name in self.cache['paths']:
            del self.cache['paths'][attribute_name]

    def del_link_attribute(self, attribute_name):
        """
        Delete a link attribute.

        :arg str attribute_name: name of link attribute to be deleted
        """
        # TODO: add example
        if attribute_name in self.cache['paths']:
            self.clear_link_attribute(attribute_name)
            self.graph.es.__delattr__(attribute_name)
        else:
            print("WARNING: Link attribute", attribute_name, "not found!")

    def set_link_attribute(self, attribute_name, values):
        """
        Set the values of some link attribute.

        These can be used as weights in measures requiring link weights.

        .. note::
           The attribute/weight matrix should be symmetric for undirected
           networks.

        :arg str attribute_name: name of link attribute to be set

        :type values: square numpy array [node,node]
        :arg  values: Entry [i,j] is the attribute of the link from i to j.
        """
        # TODO: add example and sparse version
        # TODO: test this for directed graphs
        #  Set link attribute in igraph
        for e in self.graph.es:
            e[attribute_name] = values[e.tuple]

        #  Set Network specific attributes
        self.clear_link_attribute(attribute_name)

    #
    #  Degree related measures
    #

    # @cached_const('base', 'degree')
    @cached_var('degree')
    def degree(self, key=None):
        """
        Return list of degrees.

        If a link attribute key is specified, return the associated strength

        **Example:**

        >>> Network.SmallTestNetwork().degree()
        array([3, 3, 2, 2, 3, 1])

        :arg str key: link attribute key [optional]
        :rtype: array([int>=0])
        """
        if self.directed:
            return self.indegree(key) + self.outdegree(key)
        else:
            return self.outdegree(key)

    # TODO: use directed example here and elsewhere
    @cached_var('indegree')
    def indegree(self, key=None):
        """
        Return list of in-degrees.

        If a link attribute key is specified, return the associated in strength

        **Example:**

        >>> Network.SmallDirectedTestNetwork().indegree()
        array([2, 2, 2, 1, 1, 0])

        :arg str key: link attribute key [optional]
        :rtype: array([int>=0])
        """
        if key is None:
            return self.sp_A.sum(axis=0).A.squeeze().astype(int)
        else:
            return self.link_attribute(key).sum(axis=0).T

    @cached_var('outdegree')
    def outdegree(self, key=None):
        """
        Return list of out-degrees.

        If a link attribute key is specified, return the associated out
        strength

        **Example:**

        >>> Network.SmallDirectedTestNetwork().outdegree()
        array([2, 2, 0, 1, 2, 1])

        :arg str key: link attribute key [optional]
        :rtype: array([int>=0])
        """
        if key is None:
            return self.sp_A.sum(axis=1).T.A.squeeze().astype(int)
        else:
            return self.link_attribute(key).sum(axis=1).T

    @cached_var('bildegree')
    def bildegree(self, key=None):
        """
        Return list of bilateral degrees, i.e. the number of simultaneously in-
        and out-going edges.

        If a link attribute key is specified, return the associated bilateral
        strength

        **Exmaple:**

        >>> Network.SmallDirectedTestNetwork().bildegree()
        array([0, 0, 0, 0, 0, 0], dtype=int16)
        >>> net = Network.SmallTestNetwork()
        >>> (net.bildegree() == net.degree()).all()
        True
        """
        if key is None:
            return (self.sp_A * self.sp_A).diagonal()
        else:
            w = self.link_attribute(key)
            return (w @ w).diagonal()

    @cached_var('nsi_degree', 'n.s.i. degree')
    def nsi_degree_uncorr(self, key=None):
        """
        For each node, return its uncorrected n.s.i. degree.

        If a link attribute key is specified, return the associated nsi
        strength

        :arg str key: link attribute key [optional]
        :rtype: array([float])
        """
        if self.directed:
            return self.nsi_indegree(key) + self.nsi_outdegree(key)
        else:
            if key is None:
                return self.sp_Aplus() * self.node_weights
            else:
                w = self.link_attribute(key)
                return (self.node_weights @ w).squeeze()

    def sp_nsi_diag_k(self):
        """Sparse diagonal matrix of n.s.i. degrees"""
        return sp.diags([self.nsi_degree_uncorr()], [0],
                        shape=(self.N, self.N), format='csc')

    def sp_nsi_diag_k_inv(self):
        """Sparse diagonal matrix of inverse n.s.i. degrees"""
        return sp.diags([np.power(self.nsi_degree_uncorr(), -1)], [0],
                        shape=(self.N, self.N), format='csc')

    def nsi_degree(self, typical_weight=None, key=None):
        """
        For each node, return its uncorrected or corrected n.s.i. degree.

        If a link attribute key is specified, return the associated nsi
        strength


        **Examples:**

        >>> net = Network.SmallTestNetwork()
        >>> net.nsi_degree()
        Calculating n.s.i. degree...
        array([ 8.4,  8. ,  5.9,  5.3,  7.4,  4. ])
        >>> net.splitted_copy().nsi_degree()
        Calculating n.s.i. degree...
        array([ 8.4,  8. ,  5.9,  5.3,  7.4,  4. ,  4. ])
        >>> net.nsi_degree(typical_weight=2.0)
        array([ 3.2 ,  3.  ,  1.95,  1.65,  2.7 ,  1.  ])
        >>> net.splitted_copy().nsi_degree(typical_weight=2.0)
        Calculating n.s.i. degree...
        array([ 3.2 ,  3.  ,  1.95,  1.65,  2.7 ,  1.  ,  1.  ])

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.degree())
        array([3, 3, 2, 2, 3, 1])
        >>> r(net.splitted_copy().degree())
        array([4, 3, 2, 2, 3, 2, 2])

        :type typical_weight: float > 0
        :arg  typical_weight: Optional typical node weight to be used for
                              correction. If None, the uncorrected measure is
                              returned. (Default: None)
        :arg str key: link attribute key (optional)
        :rtype: array([float])
        """
        if typical_weight is None:
            return self.nsi_degree_uncorr(key)
        else:
            return self.nsi_degree_uncorr(key)/typical_weight - 1.0

    @cached_var('nsi_indegree')
    def nsi_indegree(self, key=None):
        """
        For each node, return its n.s.i. indegree

        If a link attribute key is specified, return the associated nsi in
        strength

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> net.nsi_indegree()
        array([ 6.3,  5.3,  5.9,  3.6,  4. ,  2.5])
        >>> net.splitted_copy().nsi_indegree()
        array([ 6.3,  5.3,  5.9,  3.6,  4. ,  2.5,  2.5])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> net.indegree()
        array([2, 2, 2, 1, 1, 0])
        >>> net.splitted_copy().indegree()
        array([3, 2, 2, 1, 1, 1, 1])

        :arg str key: link attribute key [optional]
        """
        if key is None:
            return self.node_weights * self.sp_Aplus()
        else:
            w = self.link_attribute(key)
            return (self.node_weights @ w).squeeze()

    @cached_var('nsi_outdegree')
    def nsi_outdegree(self, key=None):
        """
        For each node, return its n.s.i.outdegree

        If a link attribute key is specified, return the associated nsi out
        strength

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> net.nsi_outdegree()
        array([ 5.3,  5.9,  1.9,  3.8,  5.7,  4. ])
        >>> net.splitted_copy().nsi_outdegree()
        array([ 5.3,  5.9,  1.9,  3.8,  5.7,  4. ,  4. ])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> net.outdegree()
        array([2, 2, 0, 1, 2, 1])
        >>> net.splitted_copy().outdegree()
        array([2, 2, 0, 1, 2, 2, 2])

        :arg str key: link attribute key [optional]
        """
        if key is None:
            return self.sp_Aplus() * self.node_weights
        else:
            w = self.link_attribute(key)
            return (w @ self.node_weights.transpose()).transpose().squeeze()

    @cached_const('base', 'degree df', 'the degree frequency distribution')
    def degree_distribution(self):
        """
        Return the degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().degree_distribution())
        Calculating the degree frequency distribution...
        array([ 0.1667, 0.3333, 0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having degree k.
        """
        k = self.degree()
        return self._histogram(values=k, n_bins=k.max())[0]

    @cached_const('base', 'indegree df', 'in-degree frequency distribution')
    def indegree_distribution(self):
        """
        Return the in-degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().indegree_distribution())
        Calculating in-degree frequency distribution...
        array([ 0.1667, 0.3333, 0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having in-degree k.
        """
        ki = self.indegree()
        return self._histogram(values=ki, n_bins=ki.max())[0]

    @cached_const('base', 'outdegree df', 'out-degree frequency distribution')
    def outdegree_distribution(self):
        """
        Return the out-degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().outdegree_distribution())
        Calculating out-degree frequency distribution...
        array([ 0.1667, 0. , 0.3333, 0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having out-degree k.
        """
        ko = self.outdegree()
        return self._histogram(values=ko, n_bins=ko.max()+1)[0]

    @cached_const('base', 'degree cdf', 'the cumulative degree distribution')
    def degree_cdf(self):
        """
        Return the cumulative degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().degree_cdf())
        Calculating the cumulative degree distribution...
        array([ 1. , 0.8333,  0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having degree k or more.
        """
        k = self.degree()
        return self._cum_histogram(values=k, n_bins=k.max())[0]

    @cached_const('base', 'indegree cdf',
                  'the cumulative in-degree distribution')
    def indegree_cdf(self):
        """
        Return the cumulative in-degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().indegree_cdf())
        Calculating the cumulative in-degree distribution...
        array([ 1. , 0.8333, 0.8333, 0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having in-degree k or more.
        """
        ki = self.indegree()
        return self._cum_histogram(values=ki, n_bins=ki.max() + 1)[0]

    @cached_const('base', 'outdegree cdf',
                  'the cumulative out-degree distribution')
    def outdegree_cdf(self):
        """
        Return the cumulative out-degree frequency distribution.

        **Example:**

        >>> r(Network.SmallTestNetwork().outdegree_cdf())
        Calculating the cumulative out-degree distribution...
        array([ 1. , 0.8333, 0.8333, 0.5 ])

        :rtype:  1d numpy array [k] of ints >= 0
        :return: Entry [k] is the number of nodes having out-degree k or more.
        """
        ko = self.outdegree()
        return self._cum_histogram(values=ko, n_bins=ko.max() + 1)[0]

    # FIXME: should rather return the weighted distribution!
    @cached_const('nsi', 'degree hist', 'a n.s.i. degree frequency histogram')
    def nsi_degree_histogram(self):
        """
        Return a frequency (!) histogram of n.s.i. degree.

        **Example:**

        >>> r(Network.SmallTestNetwork().nsi_degree_histogram())
        Calculating a n.s.i. degree frequency histogram...
        Calculating n.s.i. degree...
        (array([ 0.3333, 0.1667, 0.5 ]), array([ 0.1179, 0.1667, 0.0962]),
         array([ 4. , 5.4667, 6.9333]))

        :rtype:  tuple (list,list)
        :return: List of frequencies and list of lower bin bounds.
        """
        nsi_k = self.nsi_degree()
        return self._histogram(values=nsi_k,
                               n_bins=int(nsi_k.max()/nsi_k.min()) + 1)

    # FIXME: should rather return the weighted distribution!
    @cached_const('nsi', 'degree hist',
                  'a cumulative n.s.i. degree frequency histogram')
    def nsi_degree_cumulative_histogram(self):
        """
        Return a cumulative frequency (!) histogram of n.s.i. degree.

        **Example:**

        >>> r(Network.SmallTestNetwork().nsi_degree_cumulative_histogram())
        Calculating a cumulative n.s.i. degree frequency histogram...
        Calculating n.s.i. degree...
        (array([ 1. , 0.6667, 0.5 ]), array([ 4. , 5.4667, 6.9333]))

        :rtype:  tuple (list,list)
        :return: List of cumulative frequencies and list of lower bin bounds.
        """
        nsi_k = self.nsi_degree()
        return self._cum_histogram(values=nsi_k,
                                   n_bins=int(nsi_k.max()/nsi_k.min()) + 1)

    @cached_const('base', 'avg nbr degree', "average neighbours' degrees")
    def average_neighbors_degree(self):
        """
        For each node, return the average degree of its neighbors.

        (Does not use directionality information.)

        **Example:**

        >>> r(Network.SmallTestNetwork().average_neighbors_degree())
        Calculating average neighbours' degrees...
        array([ 2. ,  2.3333,  3. , 3. ,  2.6667,  3. ])

        :rtype: 1d numpy array [node] of floats >= 0
        """
        k = self.degree() * 1.0
        return self.undirected_adjacency() * k / k[k != 0]

    @cached_const('base', 'max nbr degree', "maximum neighbours' degree")
    def max_neighbors_degree(self):
        """
        For each node, return the maximal degree of its neighbors.

        (Does not use directionality information.)

        **Example:**

        >>> Network.SmallTestNetwork().max_neighbors_degree()
        Calculating maximum neighbours' degree...
        array([3, 3, 3, 3, 3, 3])

        :rtype: 1d numpy array [node] of ints >= 0
        """
        nbks = self.undirected_adjacency().multiply(self.degree())
        return nbks.max(axis=1).T.A.squeeze()

    @cached_const('nsi', 'avg nbr degree', "n.s.i. average neighbours' degree")
    def nsi_average_neighbors_degree(self):
        """
        For each node, return the average n.s.i. degree of its neighbors.

        (not yet implemented for directed networks.)

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_average_neighbors_degree())
        Calculating n.s.i. average neighbours' degree...
        Calculating n.s.i. degree...
        array([ 6.0417, 6.62 , 7.0898, 7.0434, 7.3554, 5.65 ])
        >>> r(net.splitted_copy().nsi_average_neighbors_degree())
        Calculating n.s.i. average neighbours' degree...
        Calculating n.s.i. degree...
        array([ 6.0417, 6.62 , 7.0898, 7.0434, 7.3554, 5.65 , 5.65 ])

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.average_neighbors_degree())
        Calculating average neighbours' degrees...
        array([ 2. , 2.3333, 3. , 3. , 2.6667, 3. ])
        >>> r(net.splitted_copy().average_neighbors_degree())
        Calculating average neighbours' degrees...
        array([ 2.25 , 2.3333, 3. , 3.5 , 3. , 3. , 3. ])

        :rtype: 1d numpy array [node] of floats >= 0
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        # A+ * (Dw * k) is faster than (A+ * Dw) * k
        nsi_k = self.nsi_degree()
        return self.sp_Aplus() * (self.sp_diag_w() * nsi_k) / nsi_k

    @cached_const('nsi', 'max nbr degree', "n.s.i. maximum neighbour degree")
    def nsi_max_neighbors_degree(self):
        """
        For each node, return the maximal n.s.i. degree of its neighbors.

        (not yet implemented for directed networks.)

        **Example:**

        >>> Network.SmallTestNetwork().nsi_max_neighbors_degree()
        Calculating n.s.i. maximum neighbour degree...
        Calculating n.s.i. degree...
        array([ 8.4,  8. ,  8. ,  8.4,  8.4,  8.4])

        as compared to the unweighted version:

        >>> print(Network.SmallTestNetwork().max_neighbors_degree())
        Calculating maximum neighbours' degree...
        [3 3 3 3 3 3]

        :rtype: 1d numpy array [node] of floats >= 0
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        self.nsi_degree()
        # matrix with the degrees of nodes' neighbours as rows
        return (self.sp_Aplus() * self.sp_nsi_diag_k()).max(axis=1).T.A[0]

    #
    #   Measures of clustering, transitivity and cliquishness
    #

    @cached_const('base', 'local clustering', 'local clustering coefficients')
    def local_clustering(self):
        """
        For each node, return its (Watts-Strogatz) clustering coefficient.

        This is the proportion of all pairs of its neighbors which are
        themselves interlinked.

        (Uses directionality information, if available)

        **Example:**

        >>> r(Network.SmallTestNetwork().local_clustering())
        Calculating local clustering coefficients...
        array([ 0. , 0.3333, 1. , 0. , 0.3333, 0. ])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        C = np.array(self.graph.transitivity_local_undirected())
        C[np.isnan(C)] = 0
        return C

    @cached_const('base', 'global clustering',
                  'global clustering coefficient (C_2)')
    def global_clustering(self):
        """
        Return the global (Watts-Strogatz) clustering coefficient.

        This is the mean of the local clustering coefficients. [Newman2003]_
        refers to this measure as C_2.

        **Example:**

        >>> r(Network.SmallTestNetwork().global_clustering())
        Calculating global clustering coefficient (C_2)...
        Calculating local clustering coefficients...
        0.2778

        :rtype: float between 0 and 1
        """
        return self.local_clustering().mean()

    def _motif_clustering_helper(self, t_func, T, key=None, nsi=False):
        """
        Helper function to compute the local motif clustering coefficients.
        For each node, returns a specific clustering coefficient, depending
        on the input arguments.

        :arg function t_func: multiplication of adjacency-type matrices
        :arg 1d numpy array [node]: denominator made out of (in/out/bil)degrees
        :arg str key: link attribute key (optional)
        :arg bool nsi: flag for nsi calculation (default: False)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        if nsi:
            nodew = sp.csc_matrix(np.eye(self.N) * self.node_weights)
        if key is None:
            A = self.sp_Aplus() * nodew if nsi else self.sp_A
            AT = self.sp_Aplus().T * nodew if nsi else A.T
        else:
            M = sp.csc_matrix(self.link_attribute(key)**(1/3.))
            A = M * nodew if nsi else M
            AT = M.T * nodew if nsi else M.T

        t = t_func(A, AT).diagonal()
        T = T.astype(float)
        T[T == 0] = np.nan
        C = t / (self.node_weights * T) if nsi else t / T
        C[np.isnan(C)] = 0
        return C

    @cached_var('local cyclemotif', 'local cycle motif clustering coefficient')
    def local_cyclemotif_clustering(self, key=None):
        """
        For each node, return the clustering coefficient with respect to the
        cycle motif.

        If a link attribute key is specified, return the associated link
        weighted version

        **Example:**

        >>> r(Network.SmallDirectedTestNetwork().local_cyclemotif_clustering())
        Calculating local cycle motif clustering coefficient...
        array([ 0.25,  0.25,  0.  ,  0.  ,  0.5 ,  0.  ])

        :arg str key: link attribute key (optional)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        def t_func(x, xT):
            return x * x * x
        T = self.indegree() * self.outdegree() - self.bildegree()
        return self._motif_clustering_helper(t_func, T, key=key)

    @cached_var('local midmotif', 'local mid. motif clustering coefficient')
    def local_midmotif_clustering(self, key=None):
        """
        For each node, return the clustering coefficient with respect to the
        mid. motif.

        If a link attribute key is specified, return the associated link
        weighted version

        **Example:**

        >>> r(Network.SmallDirectedTestNetwork().local_midmotif_clustering())
        Calculating local mid. motif clustering coefficient...
        array([ 0. ,  0. ,  0. ,  1. ,  0.5,  0. ])

        :arg str key: link attribute key (optional)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        def t_func(x, xT):
            return x * xT * x
        T = self.indegree() * self.outdegree() - self.bildegree()
        return self._motif_clustering_helper(t_func, T, key=key)

    @cached_var('local inmotif', 'local in motif clustering coefficient')
    def local_inmotif_clustering(self, key=None):
        """
        For each node, return the clustering coefficient with respect to the
        in motif.

        If a link attribute key is specified, return the associated link
        weighted version

        **Example:**

        >>> r(Network.SmallDirectedTestNetwork().local_inmotif_clustering())
        Calculating local in motif clustering coefficient...
        array([ 0. ,  0.5,  0.5,  0. ,  0. ,  0. ])

        :arg str key: link attribute key (optional)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        def t_func(x, xT):
            return xT * x * x
        T = self.indegree() * (self.indegree() - 1)
        return self._motif_clustering_helper(t_func, T, key=key)

    @cached_var('local outmotif', 'local out motif clustering coefficient')
    def local_outmotif_clustering(self, key=None):
        """
        For each node, return the clustering coefficient with respect to the
        out motif.

        If a link attribute key is specified, return the associated link
        weighted version

        **Example:**

        >>> r(Network.SmallDirectedTestNetwork().local_outmotif_clustering())
        Calculating local out motif clustering coefficient...
        array([ 0.5,  0.5,  0. ,  0. ,  0. ,  0. ])

        :arg str key: link attribute key (optional)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        def t_func(x, xT):
            return x * x * xT
        T = self.outdegree() * (self.outdegree() - 1)
        return self._motif_clustering_helper(t_func, T, key=key)

    @cached_var('nsi local cyclemotif',
                'local nsi cycle motif clustering coefficient')
    def nsi_local_cyclemotif_clustering(self, key=None):
        """
        For each node, return the nsi clustering coefficient with respect to
        the cycle motif.

        If a link attribute key is specified, return the associated link
        weighted version

        Reference: [Zemp2014]_

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.nsi_local_cyclemotif_clustering())
        Calculating local nsi cycle motif clustering coefficient...
        array([ 0.1845,  0.2028,  0.322 ,  0.3224,  0.3439,  0.625 ])
        >>> r(net.splitted_copy(node=1).nsi_local_cyclemotif_clustering())
        Calculating local nsi cycle motif clustering coefficient...
        array([ 0.1845,  0.2028,  0.322 ,  0.3224,  0.3439,  0.625 ,  0.2028])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.local_cyclemotif_clustering())
        Calculating local cycle motif clustering coefficient...
        array([ 0.25,  0.25,  0.  ,  0.  ,  0.5 ,  0.  ])
        >>> r(net.splitted_copy(node=1).local_cyclemotif_clustering())
        Calculating local cycle motif clustering coefficient...
        array([ 0.3333,  0.125 ,  0.    ,  0.    ,  0.5   ,  0.    ,  0.125 ])

        :arg str key: link attribute key (optional)
        """
        def t_func(x, xT):
            return x * x * x
        T = self.nsi_indegree() * self.nsi_outdegree()
        return self._motif_clustering_helper(t_func, T, key=key, nsi=True)

    @cached_var('nsi local midemotif',
                'local nsi mid. motif clustering coefficient')
    def nsi_local_midmotif_clustering(self, key=None):
        """
        For each node, return the nsi clustering coefficient with respect to
        the mid motif.

        If a link attribute key is specified, return the associated link
        weighted version

        Reference: [Zemp2014]_

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.nsi_local_midmotif_clustering())
        Calculating local nsi mid. motif clustering coefficient...
        array([ 0.4537,  0.5165,  1.    ,  1.    ,  0.8882,  1.    ])
        >>> r(net.splitted_copy(node=4).nsi_local_midmotif_clustering())
        Calculating local nsi mid. motif clustering coefficient...
        array([ 0.4537,  0.5165,  1.    ,  1.    ,  0.8882,  1.    ,  0.8882])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.local_midmotif_clustering())
        Calculating local mid. motif clustering coefficient...
        array([ 0. ,  0. ,  0. ,  1. ,  0.5,  0. ])
        >>> r(net.splitted_copy(node=4).local_midmotif_clustering())
        Calculating local mid. motif clustering coefficient...
        array([ 0. ,  0. ,  0. ,  1. ,  0.8,  0. ,  0.8])

        :arg str key: link attribute key (optional)
        """
        def t_func(x, xT):
            return x * xT * x
        T = self.nsi_indegree() * self.nsi_outdegree()
        return self._motif_clustering_helper(t_func, T, key=key, nsi=True)

    @cached_var('nsi local inemotif',
                'local nsi in motif clustering coefficient')
    def nsi_local_inmotif_clustering(self, key=None):
        """
        For each node, return the nsi clustering coefficient with respect to
        the in motif.

        If a link attribute key is specified, return the associated link
        weighted version

        Reference: [Zemp2014]_

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.nsi_local_inmotif_clustering())
        Calculating local nsi in motif clustering coefficient...
        array([ 0.5288,  0.67  ,  0.6693,  0.7569,  0.7556,  1.    ])
        >>> r(net.splitted_copy(node=1).nsi_local_inmotif_clustering())
        Calculating local nsi in motif clustering coefficient...
        array([ 0.5288,  0.67  ,  0.6693,  0.7569,  0.7556,  1.    ,  0.67  ])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.local_inmotif_clustering())
        Calculating local in motif clustering coefficient...
        array([ 0. ,  0.5,  0.5,  0. ,  0. ,  0. ])
        >>> r(net.splitted_copy(node=1).local_inmotif_clustering())
        Calculating local in motif clustering coefficient...
        array([ 0.    ,  0.5   ,  0.6667,  0.    ,  1.    ,  0.    ,  0.5   ])


        :arg str key: link attribute key (optional)
        """
        def t_func(x, xT):
            return xT * x * x
        T = self.nsi_indegree()**2
        return self._motif_clustering_helper(t_func, T, key=key, nsi=True)

    @cached_var('nsi local outemotif',
                'local nsi out motif clustering coefficient')
    def nsi_local_outmotif_clustering(self, key=None):
        """
        For each node, return the nsi clustering coefficient with respect to
        the out motif.

        If a link attribute key is specified, return the associated link
        weighted version

        Reference: [Zemp2014]_

        **Examples:**

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.nsi_local_outmotif_clustering())
        Calculating local nsi out motif clustering coefficient...
        array([ 0.67  ,  0.6693,  1.    ,  0.7528,  0.5839,  0.7656])
        >>> r(net.splitted_copy(node=0).nsi_local_outmotif_clustering())
        Calculating local nsi out motif clustering coefficient...
        array([ 0.67  ,  0.6693,  1.    ,  0.7528,  0.5839,  0.7656,  0.67  ])

        as compared to the unweighted version:

        >>> net = Network.SmallDirectedTestNetwork()
        >>> r(net.local_outmotif_clustering())
        Calculating local out motif clustering coefficient...
        array([ 0.5,  0.5,  0. ,  0. ,  0. ,  0. ])
        >>> r(net.splitted_copy(node=0).local_outmotif_clustering())
        Calculating local out motif clustering coefficient...
        array([ 0.5   ,  0.5   ,  0.    ,  0.    ,  0.3333,  1.    ,  0.5   ])

        :arg str key: link attribute key (optional)
        """
        def t_func(x, xT):
            return x * x * xT
        T = self.nsi_outdegree()**2
        return self._motif_clustering_helper(t_func, T, key=key, nsi=True)

    @cached_const('base', 'transitivity', 'transitivity coefficient (C_1)')
    def transitivity(self):
        """
        Return the transitivity (coefficient).

        This is the ratio of three times the number of triangles to the number
        of connected triples of vertices. [Newman2003]_ refers to this measure
        as C_1.

        **Example:**

        >>> r(Network.SmallTestNetwork().transitivity())
        Calculating transitivity coefficient (C_1)...
        0.2727

        :rtype: float between 0 and 1
        """
        return self.graph.transitivity_undirected()

    def higher_order_transitivity(self, order, estimate=False):
        """
        Return transitivity of a certain order.

        The transitivity of order n is defined as:
         - (n x Number of cliques of n nodes) / (Number of stars of n nodes)

        It is a generalization of the standard network transitivity, which is
        included as a special case for n = 3.

        :arg int order: The order (number of nodes) of cliques to be
            considered.
        :arg bool estimate: Toggles random sampling for estimating higher order
            transitivity (much faster than exact calculation).
        :rtype: number (float) between 0 and 1
        """
        if self.silence_level <= 1:
            print("Calculating transitivity of order", order, "...")

        if order in [0, 1, 2]:
            raise NetworkError("Higher order transitivity is not defined for \
                               orders 0, 1 and 2.")
        if order == 3:
            return self.transitivity()

        if order == 4:
            #  Gathering
            # N = self.N
            # A = self.adjacency
            # T = _higher_order_transitivity4(N, A)
            # return T

            if estimate:
                motif_counts = self.graph.motifs_randesu(
                    size=4, cut_prob=[0.5, 0.5, 0.5, 0.5])
            else:
                motif_counts = self.graph.motifs_randesu(size=4)

            #  Sum over all motifs that contain a star
            n_stars = motif_counts[4] + motif_counts[7] + \
                2 * motif_counts[9] + 4 * motif_counts[10]
            n_cliques = motif_counts[10]

            # print(motif_counts)

            if n_stars != 0:
                return 4 * n_cliques / float(n_stars)
            else:
                return 0.

        if order > 4:
            raise NotImplementedError("Higher order transitivity is not yet \
                                      implemented for orders larger than 4.")

        raise ValueError("Order has to be a positive integer.")

    def local_cliquishness(self, order):
        """
        Return local cliquishness of a certain order.

        The local cliquishness measures the relative number of cliques (fully
        connected subgraphs) of a certain order that a node participates in.

        Local cliquishness is not defined for orders 1 and 2. For order 3,
        it is equivalent to the local clustering coefficient
        :meth:`local_clustering`, since cliques of order 3 are triangles.

        Local cliquishness is always bounded by 0 and 1 and set to zero for
        nodes with degree smaller than order - 1.

        :type order: number (int)
        :arg order: The order (number of nodes) of cliques to be considered.

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        if self.directed:
            raise NetworkError("Not implemented yet...")

        if self.silence_level <= 1:
            print("Calculating local cliquishness of order", order, "...")

        if order in [0, 1, 2]:
            raise NetworkError("Local cliquishness is not defined for orders \
                               0, 1 and 2.")

        if order == 3:
            return self.local_clustering()

        if order == 4:
            return _local_cliquishness_4thorder(self.N,
                                                self.adjacency.astype(int),
                                                self.degree())
        if order == 5:
            return _local_cliquishness_5thorder(self.N,
                                                self.adjacency.astype(int),
                                                self.degree())
        if order > 5:
            raise NotImplementedError("Local cliquishness is not yet \
                                      implemented for orders larger than 5.")

        raise ValueError("Order has to be a positive integer.")

    @staticmethod
    def weighted_local_clustering(weighted_A):
        """
        For each node, return its weighted clustering coefficient,
        given a weighted adjacency matrix.

        This follows [Holme2007]_.

        **Example:**

        >>> print(r(Network.weighted_local_clustering(weighted_A=[
        ...     [ 0.  , 0.  , 0.  , 0.55, 0.65, 0.75],
        ...     [ 0.  , 0.  , 0.63, 0.77, 0.91, 0.  ],
        ...     [ 0.  , 0.63, 0.  , 0.  , 1.17, 0.  ],
        ...     [ 0.55, 0.77, 0.  , 0.  , 0.  , 0.  ],
        ...     [ 0.65, 0.91, 1.17, 0.  , 0.  , 0.  ],
        ...     [ 0.75, 0.  , 0.  , 0.  , 0.  , 0.  ]])))
        Calculating local weighted clustering coefficient...
        [ 0.  0.2149  0.3539  0.  0.1538  0. ]

        as compared to the unweighted version:

        >>> print(r(Network.SmallTestNetwork().local_clustering()))
        Calculating local clustering coefficients...
        [ 0.  0.3333  1.  0.  0.3333  0. ]

        :type weighted_A: square numpy array [node,node] of floats >= 0
        :arg  weighted_A: Entry [i,j] is the link weight from i to j.
                          A value of 0 means there is no link.

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        # TODO: must be symmetric? directed version?
        print("Calculating local weighted clustering coefficient...")

        wA = np.array(weighted_A)
        max_w = np.ones_like(wA).dot(wA.max())
        return (np.linalg.matrix_power(wA, 3).diagonal()
                / (wA.dot(max_w).dot(wA)).diagonal())

    def nsi_twinness(self):
        """
        For each pair of nodes, return an n.s.i. measure of 'twinness'.

        This varies from 0.0 for unlinked nodes to 1.0 for linked nodes having
        exactly the same neighbors (called twins).

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> print(r(net.nsi_twinness()))
        Calculating n.s.i. degree...
        [[ 1.      0.      0.      0.4286  0.4524  0.4762]
         [ 0.      1.      0.7375  0.475   0.7375  0.    ]
         [ 0.      0.7375  1.      0.      0.7973  0.    ]
         [ 0.4286  0.475   0.      1.      0.      0.    ]
         [ 0.4524  0.7375  0.7973  0.      1.      0.    ]
         [ 0.4762  0.      0.      0.      0.      1.    ]]
        >>> print(r(net.splitted_copy().nsi_twinness()))
        Calculating n.s.i. degree...
        [[ 1.      0.      0.      0.4286  0.4524  0.4762  0.4762]
         [ 0.      1.      0.7375  0.475   0.7375  0.      0.    ]
         [ 0.      0.7375  1.      0.      0.7973  0.      0.    ]
         [ 0.4286  0.475   0.      1.      0.      0.      0.    ]
         [ 0.4524  0.7375  0.7973  0.      1.      0.      0.    ]
         [ 0.4762  0.      0.      0.      0.      1.      1.    ]
         [ 0.4762  0.      0.      0.      0.      1.      1.    ]]

        :rtype: square array [node,node] of floats between 0 and 1
        """
        # TODO: implement other versions as weĺl
        N, k, Ap = self.N, self.nsi_degree(), self.sp_Aplus()
        commons = Ap * self.sp_diag_w() * Ap
        kk = np.repeat([k], N, axis=0)
        return Ap.A * commons.A / np.maximum(kk, kk.T)

    #
    #  Measure Assortativity coefficient
    #

    def assortativity(self):
        """
        Return the assortativity coefficient.

        This follows [Newman2002]_.

        **Example:**

        >>> r(Network.SmallTestNetwork().assortativity())
        -0.4737

        :rtype: float between 0 and 1
        """
        degrees = self.graph.degree()
        degrees_sq = [deg**2 for deg in degrees]

        m = float(self.graph.ecount())
        num1, num2, den1 = 0, 0, 0

        for source, target in self.graph.get_edgelist():
            num1 += degrees[source] * degrees[target]
            num2 += degrees[source] + degrees[target]
            den1 += degrees_sq[source] + degrees_sq[target]

        num1 /= m
        den1 /= 2*m
        num2 = (num2 / (2 * m)) ** 2
        return (num1 - num2) / (den1 - num2)

    @cached_const('nsi', 'local clustering')
    def nsi_local_clustering_uncorr(self):
        """
        For each node, return its uncorrected n.s.i. clustering coefficient
        (between 0 and 1).

        (not yet implemented for directed networks)

        :rtype: array([float])
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        w, k = self.node_weights, self.nsi_degree()
        A_Dw = self.sp_A * self.sp_diag_w()
        numerator = (A_Dw * self.sp_Aplus() * A_Dw.T).diagonal()
        return (numerator + 2*k*w - w**2) / k**2

    def nsi_local_clustering(self, typical_weight=None):
        """
        For each node, return its uncorrected (between 0 and 1) or corrected
        (at most 1 / negative / NaN) n.s.i. clustering coefficient.

        (not yet implemented for directed networks)

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_local_clustering())
        Calculating n.s.i. degree...
        array([ 0.5513, 0.7244, 1. , 0.8184, 0.8028, 1. ])
        >>> r(net.splitted_copy().nsi_local_clustering())
        Calculating n.s.i. degree...
        array([ 0.5513, 0.7244, 1. , 0.8184, 0.8028, 1. , 1. ])

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.local_clustering())
        Calculating local clustering coefficients...
        array([ 0. , 0.3333, 1. , 0. , 0.3333, 0. ])
        >>> r(net.splitted_copy().local_clustering())
        Calculating local clustering coefficients...
        array([ 0.1667, 0.3333, 1. ,  0. , 0.3333, 1. , 1. ])

        :type typical_weight: float > 0
        :arg  typical_weight: Optional typical node weight to be used for
                              correction. If None, the uncorrected measure is
                              returned. (Default: None)

        :rtype: array([float])
        """
        if typical_weight is None:
            return self.nsi_local_clustering_uncorr()
        else:
            k = self.nsi_degree(typical_weight=typical_weight)
            if self.silence_level <= 1:
                print("Calculating corrected n.s.i. "
                      "local clustering coefficients...")

            Ap = self.sp_Aplus()
            Ap_Dw = Ap * self.sp_diag_w()
            numerator = (Ap_Dw * Ap_Dw * Ap).diagonal()
            return (numerator/typical_weight**2 - 3.0*k - 1.0) / (k * (k-1.0))

    @cached_const('nsi', 'global clustering',
                  'n.s.i. global topological clustering coefficient')
    def nsi_global_clustering(self):
        """
        Return the n.s.i. global clustering coefficient.

        (not yet implemented for directed networks.)

        **Example:**

        >>> r(Network.SmallTestNetwork().nsi_global_clustering())
        Calculating n.s.i. global topological clustering coefficient...
        Calculating n.s.i. degree...
        0.8353

        as compared to the unweighted version:

        >>> r(Network.SmallTestNetwork().global_clustering())
        Calculating global clustering coefficient (C_2)...
        Calculating local clustering coefficients...
        0.2778

        :rtype: float between 0 and 1
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        return (self.nsi_local_clustering().dot(self.node_weights)
                / self.total_node_weight)

    @cached_const('nsi', 'transitivity', 'n.s.i. transitivity')
    def nsi_transitivity(self):
        """
        Return the n.s.i. transitivity.

        .. warning::
           Not yet implemented!

        :rtype: float between 0 and 1
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        A = self.sp_Aplus()
        A_Dw = A * self.sp_diag_w()
        num = (A_Dw * A_Dw * A_Dw).diagonal().sum()
        denum = (self.sp_diag_w() * A_Dw * A_Dw).sum()

        return num / denum

    @cached_const('nsi', 'soffer clustering',
                  'n.s.i. local Soffer clustering coefficients')
    def nsi_local_soffer_clustering(self):
        """
        For each node, return its n.s.i. clustering coefficient
        with bias-reduction following [Soffer2005]_.

        (not yet implemented for directed networks.)

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_local_soffer_clustering())
        Calculating n.s.i. local Soffer clustering coefficients...
        Calculating n.s.i. degree...
        array([ 0.7665, 0.8754, 1. , 0.8184, 0.8469, 1. ])
        >>> r(net.splitted_copy().nsi_local_soffer_clustering())
        Calculating n.s.i. local Soffer clustering coefficients...
        Calculating n.s.i. degree...
        array([ 0.7665, 0.8754, 1. , 0.8184, 0.8469, 1. , 1. ])

        as compared to the version without bias-reduction:

        >>> r(Network.SmallTestNetwork().nsi_local_clustering())
        Calculating n.s.i. degree...
        array([ 0.5513, 0.7244, 1. , 0.8184, 0.8028, 1. ])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        if self.directed:
            raise NotImplementedError("Not implemented for directed networks.")

        # numerator is determined as above
        Ap = self.sp_Aplus()
        Ap_Dw = Ap * self.sp_diag_w()
        numerator = (Ap_Dw * Ap_Dw * Ap).diagonal()

        # denominator depends on degrees of neighbours
        N, k = self.N, self.nsi_degree()
        mink = np.array([[min(k[i], k[j]) for j in range(N)]
                         for i in range(N)])
        denominator = (mink * (self.sp_diag_w() * Ap)).diagonal()
        return numerator / denominator

    #
    #  Measure path lengths
    #

    @cached_var('paths')
    def path_lengths(self, link_attribute=None):
        """
        For each pair of nodes i,j, return the (weighted) shortest path length
        from i to j (also called the distance from i to j).

        This is the shortest length of a path from i to j along links,
        or infinity if there is no such path.

        The length of links can be specified by an optional link attribute.

        **Example:**

        >>> print(Network.SmallTestNetwork().path_lengths())
        Calculating all shortest path lengths...
        [[ 0.  2.  2.  1.  1.  1.]
         [ 2.  0.  1.  1.  1.  3.]
         [ 2.  1.  0.  2.  1.  3.]
         [ 1.  1.  2.  0.  2.  2.]
         [ 1.  1.  1.  2.  0.  2.]
         [ 1.  3.  3.  2.  2.  0.]]

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: square array [[float]]
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        if link_attribute is None:
            if self.silence_level <= 1:
                print("Calculating all shortest path lengths...")

            # fixed negative numbers to infinity!
            pl = np.array(self.graph.shortest_paths(), dtype=float)
            pl[pl < 0] = np.inf
            return pl
        else:
            if self.silence_level <= 1:
                print("Calculating weighted shortest path lengths...")

            return np.array(
                self.graph.shortest_paths(weights=link_attribute, mode=1))

    def average_path_length(self, link_attribute=None):
        """
        Return the average (weighted) shortest path length between all pairs
        of nodes for which a path exists.

        **Example:**

        >>> print(r(Network.SmallTestNetwork().average_path_length()))
        Calculating average (weighted) shortest path length...
        1.6667

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: float
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        if self.silence_level <= 1:
            print("Calculating average (weighted) shortest path length...")

        if link_attribute is None:
            return self.graph.average_path_length()
        else:
            path_lengths = self.path_lengths(link_attribute)

            #  Identify unconnected pairs and save in binary array isinf
            unconnected_pairs = np.isinf(path_lengths)
            #  Count the number of unconnected pairs
            n_unconnected_pairs = unconnected_pairs.sum()
            #  Set infinite entries corresponding to unconnected pairs to zero
            path_lengths[unconnected_pairs] = 0

            #  Take average of shortest geographical path length matrix
            #  excluding the diagonal, since it is always zero, and all
            #  unconnected pairs.  The diagonal should never contain
            #  infinities, so that should not be a problem.
            average_path_length = (path_lengths.sum() / float(
                self.N * (self.N - 1) - n_unconnected_pairs))

            #  Reverse changes to path_lengths
            path_lengths[unconnected_pairs] = np.inf

            return average_path_length

    @cached_const('nsi', 'avg path length',
                  'n.s.i. average shortest path length')
    def nsi_average_path_length(self):
        """
        Return the n.s.i. average shortest path length between all pairs of
        nodes for which a path exists.

        The path length from a node to itself is considered to be 1 to achieve
        node splitting invariance.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_average_path_length())
        Calculating n.s.i. average shortest path length...
        Calculating all shortest path lengths...
        1.6003
        >>> r(net.splitted_copy().nsi_average_path_length())
        Calculating n.s.i. average shortest path length...
        Calculating all shortest path lengths...
        1.6003

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.average_path_length())
        Calculating average (weighted) shortest path length...
        1.6667
        >>> r(net.splitted_copy().average_path_length())
        Calculating average (weighted) shortest path length...
        1.7619

        :rtype: float
        """
        w = self.node_weights
        # set diagonal to 1 (nodes get unit distance to themselves)
        nsi_distances = self.path_lengths() + np.identity(self.N)
        weight_products = np.outer(w, w)

        #  Set infinite entries corresponding to unconnected pairs to zero
        unconnected_pairs = np.isinf(nsi_distances)
        nsi_distances[unconnected_pairs] = 0
        weight_products[unconnected_pairs] = 0

        # nsi_distances is not sparse, so we use matrix product
        return w.dot(nsi_distances.dot(w)) / weight_products.sum()

    def diameter(self, directed=True, only_connected=True):
        """
        Return the diameter (largest shortest path length between any nodes).

        **Example:**

        >>> print(Network.SmallTestNetwork().diameter())
        3

        :arg bool directed: Indicates whether to respect link directions if the
            network is directed. (Default: True)
        :arg bool only_connected: Indicates whether to use only pairs of nodes
            with a connecting path. If False and the network is unconnected,
            the number of all nodes is returned.  (Default: True)
        :rtype: int >= 0
        """
        return self.graph.diameter(directed=directed, unconn=only_connected)

    #
    #  Link valued measures
    #

    @cached_const('base', 'matching idx', 'matching index matrix')
    def matching_index(self):
        """
        For each pair of nodes, return their matching index.

        This is the ratio of the number of common neighbors and the number of
        nodes linked to at least one of the two nodes.

        **Example:**

        >>> print(r(Network.SmallTestNetwork().matching_index()))
        Calculating matching index matrix...
        [[ 1.    0.5   0.25    0.      0.      0.    ]
         [ 0.5   1.    0.25    0.      0.2     0.    ]
         [ 0.25  0.25  1.      0.3333  0.25    0.    ]
         [ 0.    0.    0.3333  1.      0.6667  0.5   ]
         [ 0.    0.2   0.25    0.6667  1.      0.3333]
         [ 0.    0.    0.      0.5     0.3333  1.    ]]

        :rtype: array([[0<=float<=1,0<=float<=1]])
        """
        commons = (self.sp_A * self.sp_A).astype(np.float).A
        kk = np.repeat([self.degree()], self.N, axis=0)
        return commons / (kk + kk.T - commons)

    @cached_const('base', 'link btw', 'link betweenness')
    def link_betweenness(self):
        """
        For each link, return its betweenness.

        This measures on how likely the link is on a randomly chosen shortest
        path in the network.

        (Does not respect directionality of links.)

        **Example:**

        >>> print(Network.SmallTestNetwork().link_betweenness())
        Calculating link betweenness...
        [[ 0.   0.   0.   3.5  5.5  5. ] [ 0.   0.   2.   3.5  2.5  0. ]
         [ 0.   2.   0.   0.   3.   0. ] [ 3.5  3.5  0.   0.   0.   0. ]
         [ 5.5  2.5  3.   0.   0.   0. ] [ 5.   0.   0.   0.   0.   0. ]]

        :rtype:  square numpy array [node,node] of floats between 0 and 1
        :return: Entry [i,j] is the betweenness of the link between i and j,
                 or 0 if i is not linked to j.
        """
        #  Calculate link betweenness
        link_betweenness = self.graph.edge_betweenness()

        #  Initialize
        result, ecount = np.zeros((self.N, self.N)), 0

        #  Get graph adjacency list
        A_list = self.graph.get_adjlist()

        #  Write link betweenness values to matrix
        for i, Ai in enumerate(A_list):
            for j in Ai:
                #  Only visit links once
                if i < j:
                    result[i, j] = result[j, i] = link_betweenness[ecount]
                    ecount += 1
        return result

    def edge_betweenness(self):
        """
        For each link, return its betweenness.

        Alias to :meth:`link_betweenness`. This measures on how likely the
        link is on a randomly chosen shortest path in the network.

        (Does not respect directionality of links.)

        **Example:**

        >>> print(Network.SmallTestNetwork().edge_betweenness())
        Calculating link betweenness...
        [[ 0.   0.   0.   3.5  5.5  5. ] [ 0.   0.   2.   3.5  2.5  0. ]
         [ 0.   2.   0.   0.   3.   0. ] [ 3.5  3.5  0.   0.   0.   0. ]
         [ 5.5  2.5  3.   0.   0.   0. ] [ 5.   0.   0.   0.   0.   0. ]]

        :rtype:  square numpy array [node,node] of floats between 0 and 1
        :return: Entry [i,j] is the betweenness of the link between i and j,
                 or 0 if i is not linked to j.
        """
        return self.link_betweenness()

    #
    #  Node valued centrality measures
    #

    @cached_const('base', 'btw', 'node betweenness')
    def betweenness(self, no_big_int=True):
        """
        For each node, return its betweenness.

        This measures roughly how many shortest paths pass through the node.

        **Example:**

        >>> Network.SmallTestNetwork().betweenness()
        Calculating node betweenness...
        array([ 4.5,  1.5,  0. ,  1. ,  3. ,  0. ])

        :arg bool no_big_int: Toggles use of big integer calculation (slow if
            False).
        :rtype: 1d numpy array [node] of floats >= 0
        """
        #  Return the absolute value of normed tbc, since a bug sometimes
        #  results in negative signs
        #  The measure is normed by the maximum betweenness centrality achieved
        #  only by the star (Freeman 1978): (n**2-3*n+2)/2
        #  This restricts TBC to 0 <= TBC <= 1
        # maxTBC =  ( self.N**2 - 3 * self.N + 2 ) / 2

        return np.abs(np.array(self.graph.betweenness(nobigint=no_big_int)))

    # @cached_const('base', 'inter btw', 'interregional betweenness')
    def interregional_betweenness(self, sources=None, targets=None):
        """
        For each node, return its interregional betweenness for given sets
        of source and target nodes.

        This measures roughly how many shortest paths from one of the sources
        to one of the targets pass through the node.

        **Examples:**

        >>> Network.SmallTestNetwork().interregional_betweenness(
        ...     sources=[2], targets=[3,5])
        Calculating interregional betweenness...
        array([ 1.,  1.,  0.,  0.,  1.,  0.])
        >>> Network.SmallTestNetwork().interregional_betweenness(
        ...     sources=range(0,6), targets=range(0,6))
        Calculating interregional betweenness...
        array([ 9.,  3.,  0.,  2.,  6.,  0.])

        as compared to

        >>> Network.SmallTestNetwork().betweenness()
        Calculating node betweenness...
        array([ 4.5,  1.5,  0. ,  1. ,  3. ,  0. ])

        :type sources: 1d numpy array or list of ints from 0 to n_nodes-1
        :arg  sources: Set of source node indices.

        :type targets: 1d numpy array or list of ints from 0 to n_nodes-1
        :arg  targets: Set of target node indices.

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        return self.nsi_betweenness(sources=sources, targets=targets,
                                    aw=0, silent=1)

    # @cached_const('nsi', 'inter btw', 'n.s.i. interregional betweenness')
    def nsi_interregional_betweenness(self, sources, targets):
        """
        For each node, return its n.s.i. interregional betweenness for given
        sets of source and target nodes.

        This measures roughly how many shortest paths from one of the sources
        to one of the targets pass through the node, taking node weights into
        account.

        **Example:**

        >>> r(Network.SmallTestNetwork().nsi_interregional_betweenness(
        ...     sources=[2], targets=[3,5]))
        Calculating n.s.i. interregional betweenness...
        array([ 3.1667, 2.3471, 0. , 0. , 2.0652, 0. ])

        as compared to the unweighted version:

        >>> Network.SmallTestNetwork().interregional_betweenness(
        ...     sources=[2], targets=[3,5])
        Calculating interregional betweenness...
        array([ 1.,  1.,  0.,  0.,  1.,  0.])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        return self.nsi_betweenness(sources=sources, targets=targets, silent=1)

    def nsi_betweenness(self, **kwargs):
        """
        For each node, return its n.s.i. betweenness.

        This measures roughly how many shortest paths pass through the node,
        taking node weights into account.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_betweenness())
        Calculating n.s.i. betweenness...
        array([ 29.6854, 7.7129, 0. , 3.0909, 9.6996, 0. ])
        >>> r(net.splitted_copy().nsi_betweenness())
        Calculating n.s.i. betweenness...
        array([ 29.6854, 7.7129, 0. , 3.0909, 9.6996, 0. , 0. ])

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> net.betweenness()
        Calculating node betweenness...
        array([ 4.5,  1.5,  0. ,  1. ,  3. ,  0. ])
        >>> net.splitted_copy().betweenness()
        Calculating node betweenness...
        array([ 8.5,  1.5,  0. ,  1.5,  4.5,  0. ,  0. ])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        if self.silence_level <= 1:
            if "silent" not in kwargs:
                print("Calculating n.s.i. betweenness...")

        w = self.node_weights
        if "aw" in kwargs:
            if kwargs["aw"] == 0:
                w = 0.0*w + 1.0

        N, k = self.N, self.degree()
        rN = range(0, N)
        zn = np.zeros(N, dtype=np.float)
        betweenness_times_w = zn.copy()

        # initialize node lists:
        is_source = zn.copy()
        if "sources" in kwargs and kwargs["sources"] is not None:
            for i in kwargs["sources"]:
                is_source[i] = 1
        else:
            for i in rN:
                is_source[i] = 1
        if "targets" in kwargs and kwargs["targets"] is not None:
            targets = kwargs["targets"]
        else:
            targets = rN

        # node offsets for flat arrays:
        offsets = np.zeros(N)
        for i in range(1, N):
            offsets[i] = offsets[i-1] + k[i-1]
        # Note: We don't use k.cumsum() since that uses to much memory!

        # sort links by node indices (contains each link twice!):
        links = nz_coords(self.sp_A)

        # neighbours of each node:
        flat_neighbors = np.array(links)[:, 1].astype(int)
        E = len(flat_neighbors)

        # this main loop might be parallelized:
        for j0 in targets:
            j = int(j0)

            betweenness_to_j = w.copy().astype(float)
            excess_to_j = w.copy().astype(float)
            flat_predecessors = list(np.zeros(E, dtype=int))
            # Note: this cannot be transferred as numpy array since if too
            # large we get an glibc error...
            _nsi_betweenness(N, E, w, k, j, betweenness_to_j,
                             excess_to_j, offsets.astype(int),
                             flat_neighbors,
                             is_source, np.array(flat_predecessors))
            del flat_predecessors
            betweenness_times_w += w[j] * (betweenness_to_j - excess_to_j)

        return betweenness_times_w / w

    def _eigenvector_centrality_slow(self, link_attribute=None):
        """
        For each node, return its (weighted) eigenvector centrality.

        This is the load on this node from the eigenvector corresponding to the
        largest eigenvalue of the (weighted) adjacency matrix, normalized to a
        maximum of 1.

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' weight. If None, links have weight 1. (Default: None)
        :rtype: 1d numpy array [node] of floats
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        if link_attribute is None:
            if self.silence_level <= 1:
                print("Calculating topological eigenvector centrality...")

            return np.array(self.graph.eigenvector_centrality(weights=None))
        else:
            if self.silence_level <= 1:
                print("Calculating weighted eigenvector centrality...")

            return np.array(self.graph.eigenvector_centrality(
                weights=link_attribute))

    # faster version of the above:
    @cached_const('base', 'ev centrality', 'eigenvector centrality')
    def eigenvector_centrality(self):
        """
        For each node, return its eigenvector centrality.

        This is the load on this node from the eigenvector corresponding to the
        largest eigenvalue of the adjacency matrix, normalized to a
        maximum of 1.

        **Example:**

        >>> r(Network.SmallTestNetwork().eigenvector_centrality())
        Calculating eigenvector centrality...
        array([ 0.7895, 0.973 , 0.7769, 0.6941, 1. , 0.3109])

        :rtype: 1d numpy array [node] of floats
        """
        # TODO: allow for weights
        _, evecs = eigsh(self.sp_A.astype(float), k=1, sigma=self.N**2,
                         maxiter=100, tol=1e-8)
        ec = evecs.T[0]
        ec *= np.sign(ec[0])
        return ec / ec.max()

    @cached_const('nsi', 'ev centrality', 'n.s.i. eigenvector centrality')
    def nsi_eigenvector_centrality(self):
        """
        For each node, return its n.s.i. eigenvector centrality.

        This is the load on this node from the eigenvector corresponding to the
        largest eigenvalue of the n.s.i. adjacency matrix, divided by
        sqrt(node weight) and normalized to a maximum of 1.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_eigenvector_centrality())
        Calculating n.s.i. eigenvector centrality...
        array([ 0.8045, 1. , 0.8093, 0.6179, 0.9867, 0.2804])
        >>> r(net.splitted_copy().nsi_eigenvector_centrality())
        Calculating n.s.i. eigenvector centrality...
        array([ 0.8045, 1. , 0.8093, 0.6179, 0.9867, 0.2804, 0.2804])

        as compared to the unweighted version:

        >>> r(net.eigenvector_centrality())
        Calculating eigenvector centrality...
        array([ 0.7895, 0.973 , 0.7769, 0.6941, 1. , 0.3109])
        >>> r(net.splitted_copy().eigenvector_centrality())
        Calculating eigenvector centrality...
        array([ 1. , 0.8008, 0.6226, 0.6625, 0.8916, 0.582 , 0.582 ])

        :rtype: 1d numpy array [node] of floats
        """
        DwR = self.sp_diag_sqrt_w()
        sp_Astar = DwR * self.sp_Aplus() * DwR
        _, evecs = eigsh(sp_Astar, k=1, sigma=self.total_node_weight**2,
                         maxiter=100, tol=1e-8)
        ec = evecs.T[0] / np.sqrt(self.node_weights)
        ec *= np.sign(ec[0])
        return ec / ec.max()

    def pagerank(self, link_attribute=None, use_directed=True):
        """
        For each node, return its (weighted) PageRank.

        This is the load on this node from the eigenvector corresponding to the
        largest eigenvalue of a modified adjacency matrix, normalized to a
        maximum of 1.

        **Example:**

        >>> r(Network.SmallTestNetwork().pagerank())
        Calculating PageRank...
        array([ 0.2184, 0.2044, 0.1409, 0.1448, 0.2047, 0.0869])

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' weight. If None, links have weight 1. (Default: None)
        :rtype: 1d numpy array [node] of
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None
        if link_attribute is None:
            if self.silence_level <= 1:
                print("Calculating PageRank...")
            return np.array(self.graph.personalized_pagerank(
                directed=use_directed, weights=None))
        else:
            if self.silence_level <= 1:
                print("Calculating weighted PageRank...")
            return np.array(self.graph.personalized_pagerank(
                directed=use_directed, weights=link_attribute))

    def closeness(self, link_attribute=None):
        """
        For each node, return its (weighted) closeness.

        This is the inverse of the mean shortest path length from the node to
        all other nodes.

        **Example:**

        >>> r(Network.SmallTestNetwork().closeness())
        Calculating closeness...
        array([ 0.7143, 0.625 , 0.5556, 0.625 , 0.7143, 0.4545])

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        # TODO: check and describe behaviour for unconnected networks.
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        if link_attribute is None:
            if self.silence_level <= 1:
                print("Calculating closeness...")

            #  Return the absolute value of tcc, since a bug sometimes results
            #  in negative signs
            return np.abs(np.array(self.graph.closeness()))

        else:
            CC = np.zeros(self.N)
            path_lengths = self.path_lengths(link_attribute)

            if self.silence_level <= 1:
                print("Calculating weighted closeness...")

            #  Identify unconnected pairs and save in binary array isinf
            unconnected_pairs = np.isinf(path_lengths)
            #  Set infinite entries corresponding to unconnected pairs to
            #  number of vertices
            path_lengths[unconnected_pairs] = self.N

            #  Some polar nodes have an assigned distance of zero to all their
            #  neighbors. These nodes get zero geographical closeness
            #  centrality.
            path_length_sum = path_lengths.sum(axis=1)
            CC[path_length_sum != 0] = \
                (self.N - 1) / path_length_sum[path_length_sum != 0]

            #  Reverse changes to weightedPathLengths
            path_lengths[unconnected_pairs] = np.inf

            return CC

    @cached_const('nsi', 'closeness', 'n.s.i. closeness')
    def nsi_closeness(self):
        """
        For each node, return its n.s.i. closeness.

        This is the inverse of the mean shortest path length from the node to
        all other nodes. If the network is not connected, the result is 0.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_closeness())
        Calculating n.s.i. closeness...
        Calculating all shortest path lengths...
        array([ 0.7692, 0.6486, 0.5825, 0.6417, 0.7229, 0.5085])
        >>> r(net.splitted_copy().nsi_closeness())
        Calculating n.s.i. closeness...
        Calculating all shortest path lengths...
        array([ 0.7692, 0.6486, 0.5825, 0.6417, 0.7229, 0.5085, 0.5085])

        as compared to the unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.closeness())
        Calculating closeness...
        array([ 0.7143, 0.625 , 0.5556, 0.625 , 0.7143, 0.4545])
        >>> r(net.splitted_copy().closeness())
        Calculating closeness...
        array([ 0.75 , 0.5455, 0.5 , 0.6 , 0.6667, 0.5 , 0.5 ])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        # similar to nsi_average_path_length:
        nsi_distances = self.path_lengths() + np.identity(self.N)
        return (self.total_node_weight
                / np.dot(nsi_distances, self.node_weights))

    @cached_const('nsi', 'harm closeness', 'n.s.i. harmonic closeness')
    def nsi_harmonic_closeness(self):
        """
        For each node, return its n.s.i. harmonic closeness.

        This is the inverse of the harmonic mean shortest path length from the
        node to all other nodes. If the network is not connected, the result is
        not necessarily 0.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_harmonic_closeness())
        Calculating n.s.i. harmonic closeness...
        Calculating all shortest path lengths...
        array([ 0.85 , 0.7986, 0.7111, 0.7208, 0.8083, 0.6167])
        >>> r(net.splitted_copy().nsi_harmonic_closeness())
        Calculating n.s.i. harmonic closeness...
        Calculating all shortest path lengths...
        array([ 0.85 , 0.7986, 0.7111, 0.7208, 0.8083, 0.6167, 0.6167])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        # similar to nsi_average_path_length:
        nsi_distances = self.path_lengths() + np.identity(self.N)
        return (np.dot(1.0 / nsi_distances, self.node_weights)
                / self.total_node_weight)

    @cached_const('nsi', 'exp closeness',
                  'n.s.i. exponential closeness centrality')
    def nsi_exponential_closeness(self):
        """
        For each node, return its n.s.i. exponential harmonic closeness.

        This is the mean of  2**(- shortest path length)  from the
        node to all other nodes. If the network is not connected, the result is
        not necessarily 0.

        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_exponential_closeness())
        Calculating n.s.i. exponential closeness centrality...
        Calculating all shortest path lengths...
        array([ 0.425 , 0.3906, 0.3469, 0.3604, 0.4042, 0.2958])
        >>> r(net.splitted_copy().nsi_exponential_closeness())
        Calculating n.s.i. exponential closeness centrality...
        Calculating all shortest path lengths...
        array([ 0.425 , 0.3906, 0.3469, 0.3604, 0.4042, 0.2958, 0.2958])

        :rtype: 1d numpy array [node] of floats between 0 and 1
        """
        # similar to nsi_average_path_length:
        nsi_distances = self.path_lengths() + np.identity(self.N)
        return (np.dot(2.0**(-nsi_distances), self.node_weights)
                / self.total_node_weight)

    @cached_const('base', 'arenas btw', 'Arenas-type random walk betweenness')
    def arenas_betweenness(self):
        """
        For each node, return its Arenas-type random walk betweenness.

        This measures how often a random walk search for a random target node
        from a random source node is expected to pass this node.  (see
        [Arenas2003]_)

        **Example:**

        >>> r(Network.SmallTestNetwork().arenas_betweenness())
        Calculating Arenas-type random walk betweenness...
           (giant component size: 6 (1.0))
        array([ 50.1818, 50.1818, 33.4545, 33.4545, 50.1818, 16.7273])

        :rtype: 1d numpy array [node] of floats >= 0
        """
        t0 = time.time()

        #  Initialize the array to hold random walk betweenness
        arenas_betweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately Therefore get different components of the graph first
        components = self.graph.clusters()

        #  Print giant component size
        if self.silence_level <= 1:
            print("   (giant component size: "
                  + str(components.giant().vcount()) + " ("
                  + str(components.giant().vcount()
                        / float(self.graph.vcount())) + "))")

        for c, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            if len(comp) == 1:
                arenas_betweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph corresponding to component i
                subgraph = components.subgraph(c)

                #  Get the subgraph A matrix
                A = np.array(subgraph.get_adjacency(type=2).data)

                #  Generate a Network object representing the subgraph
                subnetwork = Network(adjacency=A, directed=False)

                #  Get the number of nodes of the subgraph (the component size)
                N = subnetwork.N

                #  Initialize the RWB array
                component_betweenness = np.zeros(N)

                #  Get the subnetworks degree sequence
                k = subnetwork.degree().astype('float64')

                #  Clean up
                del subgraph, subnetwork

                #  Get the P that is modified and inverted by the C++ code
                P = np.dot(np.diag(1 / k), A)

                for i in range(N):
                    #  Store the kth row of the P
                    row_i = np.copy(P[i, :])

                    #  Set the i-th row of the P to zero to account for the
                    #  absorption of random walkers at their destination
                    P[i, :] = 0

                    #  Calculate the b^i matrix
                    B = np.dot(np.linalg.inv(np.identity(N) - P), P)

                    #  Perform the summation over source node c
                    component_betweenness += B.sum(axis=0)

                    #  Restore the P
                    P[i, :] = row_i

                #  Normalize RWB by component size
                # component_betweenness *= N

                #  Get the list of vertex numbers in the subgraph
                nodes = comp

                #  Copy results into randomWalkBetweennessArray at the correct
                #  positions
                for j, node in enumerate(nodes):
                    arenas_betweenness[node] = component_betweenness[j]

        if self.silence_level <= 0:
            print("...took", time.time()-t0, "seconds")

        return arenas_betweenness

    # TODO: remove this slow version after regression test:
    def _arenas_betweenness_slow(self):
        print("WARNING: _arenas_betweenness_slow() is deprecated!")

        t0 = time.time()

        #  Initialize the array to hold random walk betweenness
        awRandomWalkBetweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately. Therefore get different components of the graph first
        components = self.graph.clusters()

        #  Print giant component size
        if self.silence_level <= 1:
            print("   (giant component size: "
                  + str(components.giant().vcount()) + " ("
                  + str(components.giant().vcount()
                        / float(self.graph.vcount())) + "))")

        for i, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            if len(comp) == 1:
                awRandomWalkBetweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph corresponding to component i
                subgraph = components.subgraph(i)

                #  Get the subgraph adjacency matrix
                adjacency = np.array(subgraph.get_adjacency(type=2).data)

                #  Get the list of vertex numbers in the subgraph
                vertexList = comp

                # Extract corresponding area weight vector:
                aw = np.zeros(len(vertexList))
                for j, vs in enumerate(vertexList):
                    aw[j] = self.node_weights[vs]

                #  Generate a Network object representing the subgraph
                subnetwork = Network(adjacency, directed=False)

                #  Get the number of nodes of the subgraph (the component size)
                nNodes = subnetwork.N

                #  Initialize the RWB array
                rwb = np.zeros(nNodes)

                #  Get the subnetworks degree sequence
                awDegreeSequence = subnetwork.nsi_degree()

                #  Clean up
                del subgraph, subnetwork

                #  Get the pMatrix that is modified and inverted
                Identity = np.identity(nNodes)
                Ap = adjacency + Identity
                pMatrix = np.diag(1/awDegreeSequence).dot(Ap).dot(np.diag(aw))

                for k in range(nNodes):
                    #  For k and each neighbour of it, set the corresponding
                    #  row of the pMatrix to zero to account for the absorption
                    #  of random walkers at their destination
                    mask = 1-Ap[k, :]
                    pMk = pMatrix*(mask.reshape((nNodes, 1)))

                    #  Calculate the b^k matrix
                    bMatrix = np.dot(np.linalg.inv(Identity-pMk), pMk)

                    #  Perform the summation over source node i
                    rwb += aw[k] * np.dot(aw.reshape((1, self.N)),
                                          bMatrix).flatten() * mask

                rwb /= aw

                #  Copy results into randomWalkBetweennessArray at the correct
                #  positions
                for j, vs in enumerate(vertexList):
                    awRandomWalkBetweenness[vs] = rwb[j]

        if self.silence_level <= 1:
            print("...took", time.time()-t0, "seconds")

        return awRandomWalkBetweenness

    # parallelized main loop
    @staticmethod
    def _mpi_nsi_arenas_betweenness(
            N, sp_P, this_Aplus, w, this_w, start_i, end_i,
            exclude_neighbors, stopping_mode, this_twinness):
        error_message, result = '', None
        try:
            component_betweenness = np.zeros(N)
            for i in range(start_i, end_i):
                # For i and each neighbour of it, modify the corresponding row
                # of P to account for the absorption of random walkers at their
                # destination
                sp_Pi = sp_P.copy()
                Aplus_i = this_Aplus[i-start_i, :]
                update_keys = [k for k in sp_Pi.keys() if Aplus_i[k[0]] == 1]
                if stopping_mode == "twinness":
                    twinness_i = this_twinness[i-start_i, :]
                    update_vals = [sp_Pi[k] * (1.0 - twinness_i[k[0]])
                                   for k in update_keys]
                else:  # "neighbors"
                    update_vals = np.zeros(len(update_keys))
                update_rows, update_cols = zip(*update_keys)
                sp_Pi[update_rows, update_cols] = update_vals
                sp_Pi = sp_Pi.tocsc()
                sp_Pi.eliminate_zeros()

                # solve (1 - sp_Pi) * V = sp_Pi
                V = splu(sp.identity(N, format='csc') - sp_Pi).solve(sp_Pi.A)

                if exclude_neighbors:
                    # for the result, we use only those targets i which are not
                    # neighboured to our node of interest j
                    B_sum = w.dot((V.T * (1 - Aplus_i)).T) * (1 - Aplus_i)
                else:
                    B_sum = w.dot(V)
                component_betweenness += this_w[i-start_i] * B_sum

            result = component_betweenness, start_i, end_i
        except RuntimeError:
            e = sys.exc_info()
            error_message = (str(e[0]) + '\n' + str(e[1]))

        return error_message, result

    # TODO: settle for some suitable defaults
    def nsi_arenas_betweenness(self, exclude_neighbors=True,
                               stopping_mode="neighbors"):
        """
        For each node, return its n.s.i. Arenas-type random walk betweenness.

        This measures how often a random walk search for a random target node
        from a random source node is expected to pass this node. (see
        [Arenas2003]_)

        **Examples:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_arenas_betweenness())
        Calculating n.s.i. Arenas-type random walk betweenness...
           (giant component size: 6 (1.0))
        Calculating n.s.i. degree...
        array([ 20.5814, 29.2103, 27.0075, 19.5434, 25.2849, 24.8483])
        >>> r(net.splitted_copy().nsi_arenas_betweenness())
        Calculating n.s.i. Arenas-type random walk betweenness...
           (giant component size: 7 (1.0))
        Calculating n.s.i. degree...
        array([ 20.5814, 29.2103, 27.0075, 19.5434, 25.2849, 24.8483, 24.8483])
        >>> r(net.nsi_arenas_betweenness(exclude_neighbors=False))
        Calculating n.s.i. Arenas-type random walk betweenness...
           (giant component size: 6 (1.0))
        Calculating n.s.i. degree...
        array([ 44.5351, 37.4058, 27.0075, 21.7736, 31.3256, 24.8483])
        >>> r(net.nsi_arenas_betweenness(stopping_mode="twinness"))
        Calculating n.s.i. Arenas-type random walk betweenness...
           (giant component size: 6 (1.0))
        Calculating n.s.i. degree...
        Calculating n.s.i. degree...
        array([ 22.6153, 41.2314, 38.6411, 28.6195, 38.5824, 30.2994])

        as compared to its unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.arenas_betweenness())
        Calculating Arenas-type random walk betweenness...
           (giant component size: 6 (1.0))
        array([ 50.1818, 50.1818, 33.4545, 33.4545, 50.1818, 16.7273])
        >>> r(net.splitted_copy().arenas_betweenness())
        Calculating Arenas-type random walk betweenness...
           (giant component size: 7 (1.0))
        array([ 90.4242, 67.8182, 45.2121, 45.2121, 67.8182, 45.2121, 45.2121])

        :arg bool exclude_neighbors: Indicates whether to use only source and
            target nodes that are not linked to the node of interest.
            (Default: True)
        :arg str stopping_mode: Specifies when the random walk is stopped. If
            "neighbors", the walk stops as soon as it reaches a neighbor of the
            target node. If "twinnness", the stopping probability at each step
            is the twinnness of the current and target nodes as given by
            :meth:`nsi_twinness()`. (Default: "neighbors")
        :rtype: 1d numpy array [node] of floats >= 0
        """
        if self.silence_level <= 1:
            print("Calculating n.s.i. Arenas-type random walk betweenness...")

        t0 = time.time()

        #  Initialize the array to hold random walk betweenness
        nsi_arenas_betweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately. Therefore get different components of the graph first
        components = self.graph.clusters()

        #  Print giant component size
        if self.silence_level <= 1:
            print("   (giant component size: "
                  + str(components.giant().vcount()) + " ("
                  + str(components.giant().vcount()
                        / float(self.graph.vcount())) + "))")

        for c, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            if len(comp) == 1:
                nsi_arenas_betweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph corresponding to component i
                subgraph = components.subgraph(c)
                A = np.array(subgraph.get_adjacency(type=2).data)
                del subgraph

                #  Get the list of vertex numbers in the subgraph
                nodes = comp

                # Extract corresponding area weight vector
                w = np.zeros(len(nodes))
                for j, node in enumerate(nodes):
                    w[j] = self.node_weights[node]

                #  Generate a Network object representing the subgraph
                subnet = Network(adjacency=A, directed=False, node_weights=w)
                N = subnet.N

                #  Calculate the subnetworks degree sequence
                subnet.nsi_degree()
                Aplus = (A + np.identity(N)).astype(int)
                if stopping_mode == "twinness":
                    twinness = self.nsi_twinness()

                #  Get the sparse P matrix that gets modified and inverted
                sp_P = (subnet.sp_nsi_diag_k_inv() * subnet.sp_Aplus()
                        * subnet.sp_diag_w()).todok()

                if mpi.available:
                    parts = max(1, int(np.ceil(
                        min((mpi.size-1) * 10.0, 0.1 * N))))
                    step = int(np.ceil(1.0 * N / (1.0 * parts)))
                    if self.silence_level <= 0:
                        print(f"   parallelizing on {mpi.size-1}"
                              f" slaves into {parts} parts with "
                              f"{step} nodes each...")

                    for index in range(parts):
                        start_i = index * step
                        end_i = min((index + 1) * step, N)
                        if start_i >= end_i:
                            break
                        this_Aplus = Aplus[start_i:end_i, :]
                        this_w = w[start_i:end_i]
                        if stopping_mode == "twinness":
                            this_twinness = twinness[start_i:end_i, :]
                        else:
                            this_twinness = None
                        if self.silence_level <= 0:
                            print("   submitting", index)
                            mpi.submit_call(
                                "Network._mpi_nsi_arenas_betweenness",
                                (N, sp_P, this_Aplus, w, this_w,
                                 start_i, end_i, exclude_neighbors,
                                 stopping_mode, this_twinness),
                                module="pyunicorn", id=index)

                    # Retrieve results of all submited jobs
                    component_betweenness = np.zeros(N)
                    for index in range(parts):
                        start_i = index * step
                        if self.silence_level <= 0:
                            print("   retrieving results from", index)
                        error_message, result = mpi.get_result(index)
                        if error_message != '':
                            print(error_message)
                            sys.exit()
                        this_betweenness, start_i, end_i = result
                        component_betweenness += this_betweenness
                else:
                    component_betweenness = np.zeros(N)
                    if stopping_mode == "twinness":
                        this_twinness = twinness
                    else:
                        this_twinness = None
                    error_message, result = \
                        Network._mpi_nsi_arenas_betweenness(
                            N, sp_P, Aplus, w, w, 0, N,
                            exclude_neighbors, stopping_mode, this_twinness)
                    if error_message != '':
                        print(error_message)
                        sys.exit()
                    this_betweenness, start_i, end_i = result
                    component_betweenness += this_betweenness

                component_betweenness /= w

                # here I tried several ways to correct for the fact that k is
                # not neighboured to j (see above):
                # component_betweenness *= 1-w/nsi_k
                # component_betweenness += subnet.total_node_weight*nsi_k
                # component_betweenness -= subnet.total_node_weight*nsi_k
                # is this an improvement???

                #  Clean up
                del subnet

                #  Copy results into randomWalkBetweennessArray at the correct
                #  positions
                for j, node in enumerate(nodes):
                    nsi_arenas_betweenness[node] = component_betweenness[j]

        if self.silence_level <= 0:
            print("...took", time.time()-t0, "seconds")

        return nsi_arenas_betweenness

    # deactivated and replaced by corrected and faster version (see below):
    # TODO: remove after regression test
    def _newman_betweenness_badly(self, link_attribute=None):
        print("WARNING: _newman_betweenness_badly() is deprecated!")

        #  Initialize the array to hold random walk betweenness
        randomWalkBetweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately. Therefore get different components of the graph first
        components = self.graph.clusters()

        for i, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            if len(comp) == 1:
                randomWalkBetweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph corresponding to component i
                subgraph = components.subgraph(i)

                #  Get the subgraph adjacency matrix
                adjacency = np.array(subgraph.get_adjacency(type=2).data)

                #  Generate a Network object representing the subgraph
                subnetwork = Network(adjacency, directed=True)

                #  Get the number of nodes of the subgraph (the component size)
                nNodes = subnetwork.N

                #  Initialize the RWB array
                rwb = np.zeros(nNodes)

                # Heitzig FIXME: laplacian is faulty for undirected
                # networks since row-sum != 0: Get the (directed and weighted)
                # graph Laplacian
                laplacian = \
                    subnetwork.laplacian(link_attribute).astype('float64')

                #  Remove the last row and column to make the matrix invertible
                laplacian = laplacian[:-1, :-1]

                #  Invert the reduced laplacian matrix and
                T = np.linalg.inv(laplacian)

                #  Clean up
                del subgraph, laplacian

                #  add row and column of zeros to T at the position they were
                #  removed from the laplacian matrix (the last row and column)
                T = np.vstack((T, np.zeros(nNodes - 1)))
                T = np.ascontiguousarray(np.hstack((T, np.zeros((nNodes, 1)))))

                nNodes = float(nNodes)

                #  Calculate the random walk betweenness in C++ using Cython
                # added -w since numerous warnings of type "Warnung: veraltete
                # Konvertierung von Zeichenkettenkonstante in »char*«"
                # occurred:
                rwb = _newman_betweenness_badly_cython(adjacency.astype(int),
                                                       T, rwb, self.N)

                #  Normalize RWB by component size
                rwb *= nNodes

                #  Get the list of vertex numbers in the subgraph
                vertexList = comp

                #  Copy results into randomWalkBetweennessArray at the correct
                #  positions
                for j, vs in enumerate(vertexList):
                    randomWalkBetweenness[vs] = rwb[j]

        return randomWalkBetweenness

    # much faster (and corrected) version of the preceding:
    @cached_const('base', 'newman btw', "Newman's random walk betweenness")
    def newman_betweenness(self):
        """
        For each node, return Newman's random walk betweenness.

        This measures how often a random walk search for a random target node
        from a random source node is expected to pass this node, not counting
        when the walk returns along a link it took before to leave the node.
        (see [Newman2005]_)

        **Example:**

        >>> r(Network.SmallTestNetwork().newman_betweenness())
        Calculating Newman's random walk betweenness...
           (giant component size: 6 (1.0))
        array([ 4.1818, 3.4182, 2.5091, 3.0182, 3.6 , 2. ])

        :rtype: 1d numpy array [node] of floats >= 0
        """
        t0 = time.time()

        #  Initialize the array to hold random walk betweenness
        newman_betweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately. Therefore get different components of the graph first
        components = self.graph.clusters()

        #  Print giant component size
        if self.silence_level <= 1:
            print("   (giant component size: "
                  + str(components.giant().vcount()) + " ("
                  + str(components.giant().vcount()
                        / float(self.graph.vcount())) + "))")

        for c, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            if len(comp) < 2:
                newman_betweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph A matrix corresponding to component c
                subgraph = components.subgraph(c)
                A = np.array(subgraph.get_adjacency(type=2).data,
                             dtype=np.int8)

                #  Generate a Network object representing the subgraph
                subnetwork = Network(adjacency=A, directed=False)
                N, sp_A = subnetwork.N, subnetwork.sp_A

                # Kirchhoff matrix
                sp_M = sp.diags([subnetwork.indegree()], [0],
                                shape=(N, N), format='csc') - sp_A

                # invert it without last row/col
                # FIXME: in rare cases (when there is an exact twin to the last
                # node), this might not be invertible and a different row/col
                # would need to be removed!
                V = sp.lil_matrix((N, N))
                V[:-1, :-1] = inv(sp_M[:-1, :-1])
                V = V.A
                del subgraph, subnetwork, sp_A, sp_M

                #  Calculate the random walk betweenness in C++ using Weave
                component_betweenness = np.zeros(N)
                if mpi.available:
                    # determine in how many parts we split the outer loop:
                    parts = max(1, int(np.ceil(min((mpi.size-1) * 10.0,
                                                   0.1 * N))))
                    # corresponding step size for c index of outer loop:
                    step = int(np.ceil(1.0 * N / (1.0 * parts)))
                    if self.silence_level <= 0:
                        print("   parallelizing on " + str((mpi.size-1))
                              + " slaves into " + str(parts) + " parts with "
                              + str(step) + " nodes each...")

                    # now submit the jobs:
                    for index in range(parts):
                        start_i = index * step
                        end_i = min((index + 1) * step, N)
                        if start_i >= end_i:
                            break
                        this_A = A[start_i:end_i, :]
                        # submit the job and add it to the list of jobs, so
                        # that later the results can be retrieved:
                        if self.silence_level <= 0:
                            print("submitting part from", start_i, "to", end_i)
                        mpi.submit_call("_cy_mpi_newman_betweenness",
                                        (this_A.astype(int), V.astype(float),
                                         N, start_i, end_i),
                                        module="pyunicorn", id=index,
                                        time_est=this_A.sum())

                    # Retrieve results of all submitted jobs:
                    component_betweenness = np.zeros(N)
                    for index in range(parts):
                        # the following call connects to the submitted job,
                        # waits until it finishes, and retrieves the result:
                        if self.silence_level <= 0:
                            print("retrieving results from ", index)
                        this_betweenness, start_i, end_i = \
                            mpi.get_result(index)
                        component_betweenness[start_i:end_i] = this_betweenness
                else:
                    component_betweenness, start_i, end_i =\
                        _cy_mpi_newman_betweenness(A.astype(int),
                                                   V.astype(float), N, 0, N)

                component_betweenness += 2 * (N - 1)
                component_betweenness /= (N - 1.0)  # TODO: why is this?

                # sort results into correct positions
                nodes = comp
                for j, node in enumerate(nodes):
                    newman_betweenness[node] = component_betweenness[j]

        if self.silence_level <= 0:
            print("...took", time.time()-t0, "seconds")

        return newman_betweenness

    def nsi_newman_betweenness(self, add_local_ends=False):
        """
        For each node, return its n.s.i. Newman-type random walk betweenness.

        This measures how often a random walk search for a random target node
        from a random source node is expected to pass this node, not counting
        when the walk returns along a link it took before to leave the node.
        (see [Newman2005]_)

        In this n.s.i. version, node weights are taken into account, and only
        random walks are used that do not start or end in neighbors of the
        node.


        **Example:**

        >>> net = Network.SmallTestNetwork()
        >>> r(net.nsi_newman_betweenness())
        Calculating n.s.i. Newman-type random walk betweenness...
           (giant component size: 6 (1.0))
        Calculating n.s.i. degree...
        array([ 0.4048, 0. , 0.8521, 3.3357, 1.3662, 0. ])
        >>> r(net.splitted_copy().nsi_newman_betweenness())
        Calculating n.s.i. Newman-type random walk betweenness...
           (giant component size: 7 (1.0))
        Calculating n.s.i. degree...
        array([ 0.4048, 0. , 0.8521, 3.3357, 1.3662, 0. , 0. ])
        >>> r(net.nsi_newman_betweenness(add_local_ends=True))
        Calculating n.s.i. Newman-type random walk betweenness...
           (giant component size: 6 (1.0))
        Calculating n.s.i. degree...
        array([ 131.4448, 128. , 107.6421, 102.4457, 124.2062, 80. ])
        >>> r(net.splitted_copy().nsi_newman_betweenness(
        ...     add_local_ends=True))
        Calculating n.s.i. Newman-type random walk betweenness...
           (giant component size: 7 (1.0))
        Calculating n.s.i. degree...
        array([ 131.4448, 128. , 107.6421, 102.4457, 124.2062, 80. , 80. ])

        as compared to its unweighted version:

        >>> net = Network.SmallTestNetwork()
        >>> r(net.newman_betweenness())
        Calculating Newman's random walk betweenness...
           (giant component size: 6 (1.0))
        array([ 4.1818, 3.4182, 2.5091, 3.0182, 3.6 , 2. ])
        >>> r(net.splitted_copy().newman_betweenness())
        Calculating Newman's random walk betweenness...
           (giant component size: 7 (1.0))
        array([ 5.2626, 3.5152, 2.5455, 3.2121, 3.8182, 2.5556, 2.5556])

        :arg bool add_local_ends: Indicates whether to add a correction for the
            fact that walks starting or ending in neighbors are not used.
            (Default: false)
        :rtype: array [float>=0]
        """
        if self.silence_level <= 1:
            print("Calculating n.s.i. Newman-type random walk betweenness...")

        t0 = time.time()

        #  Initialize the array to hold random walk betweenness
        nsi_newman_betweenness = np.zeros(self.N)

        #  Random walk betweenness has to be calculated for each component
        #  separately. Therefore get different components of the graph first
        components = self.graph.clusters()

        #  Print giant component size
        if self.silence_level <= 1:
            print("   (giant component size: "
                  + str(components.giant().vcount()) + " ("
                  + str(components.giant().vcount()
                        / float(self.graph.vcount())) + "))")

        for c, comp in enumerate(components):
            #  If the component has size 1, set random walk betweenness to zero
            # FIXME: check why there was a problem with ==1
            if len(comp) < 2:
                nsi_newman_betweenness[comp[0]] = 0
            #  For larger components, continue with the calculation
            else:
                #  Get the subgraph corresponding to component i
                subgraph = components.subgraph(c)

                #  Get the subgraph A matrix
                A = np.array(subgraph.get_adjacency(type=2).data,
                             dtype=np.int8)

                #  Get the list of vertex numbers in the subgraph
                nodes = comp

                # Extract corresponding area weight vector:
                w = np.zeros(len(nodes))
                for j, node in enumerate(nodes):
                    w[j] = self.node_weights[node]

                #  Generate a Network object representing the subgraph
                subnet = Network(adjacency=A, directed=False, node_weights=w)
                N = subnet.N

                #  Initialize the RWB array
                component_betweenness = np.zeros(N)

                # sp_M = area-weighted Kirchhoff matrix * diag(w)^(-1)
                Ap = subnet.sp_Aplus()
                Dw, DwI = subnet.sp_diag_w(), subnet.sp_diag_w_inv()
                Dk, DkI = subnet.sp_nsi_diag_k(), subnet.sp_nsi_diag_k_inv()
                sp_M = Dw * (Dk - Ap * Dw) * DwI

                # invert sp_M without last row/col (see above)
                sp_M_inv = sp.lil_matrix((N, N))
                sp_M_inv[:-1, :-1] = inv(sp_M[:-1, :-1])

                # Note: sp_M_inv is not necessarily sparse, so the order is
                # important for performance
                V = ((DkI * Ap) * sp_M_inv).T.astype('float32').A
                del subgraph, Ap, Dw, DwI, Dk, DkI, sp_M, sp_M_inv

                # TODO: verify that this was indeed wrong
                # w = self.node_weights

                # indicator matrix that i,j are not neighboured or equal
                not_adjacent_or_equal = (1 - A - np.identity(N)).astype('int8')

                if mpi.available:
                    parts = max(1, int(np.ceil(min((mpi.size-1) * 10.0,
                                                   0.1 * N))))
                    step = int(np.ceil(1.0*N/(1.0*parts)))
                    if self.silence_level <= 0:
                        print("   parallelizing on " + str((mpi.size-1))
                              + " slaves into " + str(parts) + " parts with "
                              + str(step) + " nodes each...")

                    for idx in range(parts):
                        start_i = idx * step
                        end_i = min((idx+1)*step, N)
                        if start_i >= end_i:
                            break
                        this_A = A[start_i:end_i, :]
                        this_not_adjacent_or_equal = \
                            not_adjacent_or_equal[start_i:end_i, :]

                        mpi.submit_call(
                            "_cy_mpi_nsi_newman_betweenness",
                            (this_A.astype(int), V.astype(float), N,
                             w.astype(float),
                             this_not_adjacent_or_equal.astype(int), start_i,
                             end_i),
                            module="pyunicorn", id=idx)

                    # Retrieve results of all submited jobs
                    component_betweenness = np.zeros(N)
                    for idx in range(parts):
                        this_betweenness, start_i, end_i = mpi.get_result(idx)
                        component_betweenness[start_i:end_i] = this_betweenness

                else:
                    component_betweenness, start_i, end_i = \
                        _cy_mpi_nsi_newman_betweenness(
                            A.astype(int), V.astype(float), N, w.astype(float),
                            not_adjacent_or_equal.astype(int), 0, N)

                #  Correction for the fact that we used only s,t not
                #  neighboured to i
                if add_local_ends:
                    nsi_k = subnet.nsi_degree()
                    component_betweenness += (2.0 * w.sum() - nsi_k) * nsi_k

                #  Copy results into randomWalkBetweennessArray at the correct
                #  positions
                for j, node in enumerate(nodes):
                    nsi_newman_betweenness[node] = component_betweenness[j]

        if self.silence_level <= 0:
            print("...took", time.time()-t0, "seconds")

        return nsi_newman_betweenness

    #
    #  Efficiency measures
    #

    def global_efficiency(self, link_attribute=None):
        """
        Return the global (weighted) efficiency. (see [Costa2007]_)

        **Example:**

        >>> r(Network.SmallTestNetwork().global_efficiency())
        Calculating all shortest path lengths...
        Calculating global (weighted) efficiency...
        0.7111

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: float
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        path_lengths = self.path_lengths(link_attribute)

        if self.silence_level <= 1:
            print("Calculating global (weighted) efficiency...")

        #  Set path lengths on diagonal to infinity to avoid summing over those
        #  entries when calculating efficiency
        np.fill_diagonal(path_lengths, np.inf)

        #  Calculate global efficiency
        efficiency = (1/float(self.N * (self.N-1)) * (1/path_lengths).sum())

        #  Restore path lengths on diagonal to zero
        np.fill_diagonal(path_lengths, 0)

        return efficiency

    @cached_const('nsi', 'global eff', 'n.s.i. global efficiency')
    def nsi_global_efficiency(self):
        """
        Return the n.s.i. global efficiency.

        **Example:**

        >>> r(Network.SmallTestNetwork().nsi_global_efficiency())
        Calculating n.s.i. global efficiency...
        Calculating all shortest path lengths...
        0.7415

        :rtype: float
        """
        # TODO: check results of examples!
        w = self.node_weights
        #  Set path lengths on diagonal to 1
        nsi_dist = self.path_lengths() + np.identity(self.N)
        return w.dot((1/nsi_dist).dot(w)) / self.total_node_weight**2

    def distance_based_measures(self, replace_inf_by=None):
        """
        Return a dictionary of local and global measures that are based on
        shortest path lengths.

        This is useful for large graphs for which the matrix of all shortest
        path lengths cannot be stored.

        EXPERIMENTAL!

        :type replace_inf_by: float/inf/None
        :arg replace_inf_by: If None, the number of nodes is used.
            (Default: None)
        :rtype: dictionary with keys "closeness", "harmonic_closeness",
            "exponential_closeness", "average_path_length",
            "global_efficiency", "nsi_closeness", "nsi_harmonic_closeness",
            "nsi_exponential_closeness", "nsi_average_path_length",
            "nsi_global_efficiency"
        """
        N, w, W = self.N, self.node_weights, self.total_node_weight

        if replace_inf_by is None:
            replace_inf_by = N

        closeness = np.zeros(N)
        harmonic_closeness = np.zeros(N)
        exponential_closeness = np.zeros(N)
        average_path_length = 0
        nsi_closeness = np.zeros(N)
        nsi_harmonic_closeness = np.zeros(N)
        nsi_exponential_closeness = np.zeros(N)
        nsi_average_path_length = 0

        for i in range(N):
            if self.silence_level == 0:
                print(i)
            di = np.array(self.graph.shortest_paths(i), dtype=float).flatten()
            di[np.where(di == np.inf)] = replace_inf_by

            closeness[i] = 1.0 / di.sum()
            average_path_length += di.sum()

            di[i] = np.inf
            harmonic_closeness[i] = (1.0/di).sum()
            exponential_closeness[i] = (0.5**di).sum()

            di[i] = 1
            nsi_closeness[i] = 1.0 / (w*di).sum()
            nsi_average_path_length += w[i] * (w*di).sum()
            nsi_harmonic_closeness[i] = (w/di).sum()
            nsi_exponential_closeness[i] = (w * 0.5**di).sum()

        return {
            "closeness": closeness * (N-1),
            "harmonic_closeness": harmonic_closeness / (N-1),
            "exponential_closeness": exponential_closeness / (N-1),
            "average_path_length": average_path_length / N*(N-1),
            "global_efficiency": harmonic_closeness.mean() / (N-1),
            "nsi_closeness": nsi_closeness * W,
            "nsi_harmonic_closeness": nsi_harmonic_closeness / W,
            "nsi_exponential_closeness": nsi_exponential_closeness / W,
            "nsi_average_path_length": nsi_average_path_length / W**2,
            "nsi_global_efficiency": w.dot(nsi_harmonic_closeness) / W**2
        }

    #
    #  Vulnerability measures
    #

    def local_vulnerability(self, link_attribute=None):
        """
        For each node, return its vulnerability. (see [Costa2007]_)

        **Example:**

        >>> r(Network.SmallTestNetwork().local_vulnerability())
        Calculating all shortest path lengths...
        Calculating global (weighted) efficiency...
        Calculating (weighted) node vulnerabilities...
        array([ 0.2969, 0.0625, -0.0313, -0.0078, 0.0977, -0.125 ])

        :arg str link_attribute: Optional name of the link attribute to be used
            as the links' length. If None, links have length 1. (Default: None)
        :rtype: 1d numpy array [node] of floats
        """
        if link_attribute == "topological":
            print("WARNING: link_attribute='topological' is deprecated.\n"
                  + "Use link_attribute=None instead.")
            link_attribute = None

        vulnerability = np.zeros(self.N)

        #  Calculate global efficiency of complete network E
        global_efficiency = self.global_efficiency(link_attribute)

        if self.silence_level <= 1:
            print("Calculating (weighted) node vulnerabilities...")

        #  Initialize progress bar
        if self.silence_level <= 1:
            progress = progressbar.ProgressBar(maxval=self.N).start()

        for i in range(self.N):
            # Update progress bar every 10 steps
            if self.silence_level <= 1:
                if (i % 10) == 0:
                    progress.update(i)

            #  Remove vertex i from graph
            graph = self.graph - i

            #  Generate Network object from this reduced graph
            network = Network.FromIGraph(graph, 2)

            #  Calculate global topological efficiency E_i after removal of
            #  vertex i
            node_efficiency = network.global_efficiency(link_attribute)

            #  Calculate local topological vulnerability of vertex i
            vulnerability[i] = ((global_efficiency - node_efficiency)
                                / global_efficiency)

            #  Clean up
            del graph, network

        #  Terminate progress bar
        if self.silence_level <= 1:
            progress.finish()

        return vulnerability

    #
    #  Community measures
    #

    @cached_const('base', 'coreness', 'coreness')
    def coreness(self):
        """
        For each node, return its coreness.

        The k-core of a network is a maximal subnetwork in which each node has
        at least degree k. (Degree here means the degree in the subnetwork of
        course). The coreness of a node is k if it is a member of the k-core
        but not a member of the (k+1)-core.

        **Example:**

        >>> Network.SmallTestNetwork().coreness()
        Calculating coreness...
        array([2, 2, 2, 2, 2, 1])

        :rtype: 1d numpy array [node] of floats
        """
        return np.array(self.graph.coreness())

    #
    #  Synchronizability measures
    #

    @cached_const('base', 'msf sync',
                  'master stability function synchronizability')
    def msf_synchronizability(self):
        """
        Return the synchronizability in the master stability function
        framework.

        This is equal to the largest eigenvalue of the graph Laplacian divided
        by the smallest non-zero eigenvalue. A smaller value indicates higher
        synchronizability and vice versa. This function makes sense for
        undirected climate networks (with symmetric laplacian matrix).
        For directed networks, the undirected laplacian matrix is used.

        (see [Pecora1998]_)

        .. note::
           Only defined for undirected networks.

        **Example:**

        >>> r(Network.SmallTestNetwork().msf_synchronizability())
        Calculating master stability function synchronizability...
        6.7784

        :rtype: float
        """
        # TODO: use sparse version to speed up!
        #  Get undirected graph laplacian
        laplacian = self.laplacian()

        #  Get eigenvalues of laplacian
        eigenvalues = np.real(linalg.eigvals(laplacian))

        #  Sort eigenvalues in ascending order
        eigenvalues.sort()

        #  Get smallest non-zero eigenvalue (Fiedler value)
        i = 0
        fiedler_value = 0

        #  The limited accuracy of eigenvalue calculation forces the use of
        #  some threshold, below which eigenvalues are considered to be zero
        accuracy = 10**(-10)

        while (eigenvalues[i] <= accuracy) and (i < self.N - 1):
            fiedler_value = eigenvalues[i+1]
            i += 1

        #  Calculate synchronizability R
        R = eigenvalues[-1] / fiedler_value

        return R

    #
    #  Distance measures between two graphs
    #

    def hamming_distance_from(self, other_network):
        """
        Return the normalized hamming distance between this and another
        network.

        This is the percentage of links that have to be changed to transform
        this network into the other. Hamming distance is only defined for
        networks with an equal number of nodes.

        :rtype: float between 0 and 1
        """
        #  Get own adjacency matrix
        A = self.adjacency
        #  Get the other graph's adjacency matrix
        B = other_network.adjacency

        #  Check if the graphs have the same number of vertices
        if self.N == other_network.N:
            #  Calculate the hamming distance
            hamming = (A != B).sum()

            #  Return the normalized hamming distance
            return hamming / float(self.N * (self.N - 1))
        else:
            raise NetworkError("Only defined for networks with same number of \
                               nodes.")

    def spreading(self, alpha=None):
        """
        For each node, return its "spreading" value.

        .. note::
           This is still EXPERIMENTAL!

        :rtype: 1d numpy array [node] of floats
        """
        if alpha is None:
            alpha = 1.0 / self.degree().mean()
        return matfuncs.expm(
            np.log(2.0) * (alpha * self.adjacency
                           - np.identity(self.N))).sum(axis=0).flatten()

    def nsi_spreading(self, alpha=None):
        """
        For each node, return its n.s.i. "spreading" value.

        .. note::
           This is still EXPERIMENTAL!

        :rtype: 1d numpy array [node] of floats
        """
        N, Aplus = self.N, self.sp_Aplus().A
        w, k = self.node_weights, self.nsi_degree()
        if alpha is None:
            alpha = self.total_node_weight / k.dot(w)
        # print(alpha)
        return (
            matfuncs.expm(
                np.log(2.0)*(Aplus * alpha * w - sp.identity(N))).dot(Aplus)
            * w.reshape((N, 1))).sum(axis=0)

    def do_nsi_pca_clustering(self, max_n_clusters=None):
        """
        Perform a clustering of the nodes using principal components analysis.

        Perform a PCA for the columns of the adjacency matrix, extract the
        largest eigenvalues, and assign each node to that eigenvalue whose
        eigenvector explains the largest amount of the node's column's
        variance, i.e. the one that maximizes the value of eigenvalue *
        corresponding factor load on that node's column.

        .. note::
           This is still EXPERIMENTAL!

        :type max_n_clusters: int >= 1
        :arg  max_n_clusters: Number of clusters to find at most.
                              (Default: ceil(sqrt(N)))

        :rtype: tuple (list[node], list[node], list[cluster], 2d numpy array)
        :return: A list of cluster indices for each node,
                 a list with the fraction of the node's column's variance
                 explained by chosen eigenvector, for each node,
                 a list of eigenvalues corresponding to each cluster,
                 and an array whose columns are the corresponding eigenvectors
        """
        # TODO: works only for undirected graphs so far. For directed, A
        # stacked with its transpose would have to be used!

        # CSS (corrected sum of squares); proportional to covariance matrix
        DwR = self.sp_diag_sqrt_w()
        DAD = DwR * self.sp_Aplus() * DwR
        corr = self.nsi_degree() * np.sqrt(self.node_weights)
        CSS = DAD * DAD - np.outer(corr, corr) / self.total_node_weight

        # extract max_n_clusters largest eigenvalues and eigenvectors from CSS
        N = self.N
        if max_n_clusters is None:
            max_n_clusters = int(np.ceil(np.sqrt(N)))
        # total variance is proportional to trace of CSS
        var = CSS.diagonal().A[0]

        # target eigenvalue (known upper bound) -> largest eigenvalues
        tau = sum(var)
        evals, evecs = eigsh(CSS, k=max_n_clusters, sigma=tau,
                             maxiter=100*max_n_clusters, tol=1e-8)

        # fraction of node's variance explained by each eigenvector
        explained_var = np.power(evecs, 2.0) * evals.reshape((1, evals.size))

        # assign each node to cluster 2*i or 2*i+1
        # for that eigenvector i which explains the largest part of the node's
        # variance.  assign node to cluster 2*i if eigenvector positive at the
        # node, otherwise to cluster 2*i+1:
        cluster_index = 2 * np.argmax(explained_var, axis=1)
        for i in range(0, N):
            if evecs[i, cluster_index[i]/2] < 0.0:
                cluster_index[i] += 1

        cluster_explained_var = np.max(explained_var, axis=1)
        cluster_index_set = set(cluster_index)
        cluster_sizes = np.zeros(max(cluster_index_set)+1)
        for i in range(0, N):
            cluster_sizes[cluster_index[i]] += self.node_weights[i]
        cluster_sizes = cluster_sizes[list(cluster_index_set)]
        cluster_fit = cluster_explained_var / var
        if self.silence_level <= 1:
            print("max_n_clusters was", max_n_clusters)
            print(f"found {len(evals)} eigenvalues and "
                  f"{len(cluster_index_set)} clusters")
            print(f"cluster sizes range from {cluster_sizes.min()} to "
                  f"{cluster_sizes.max()} with median "
                  f"{np.median(cluster_sizes)}: {cluster_sizes}")
            print(f"max and min found eigenvalues are {max(evals)} and "
                  f"{min(evals)} (average of all was {tau/N})")
            print(f"pca and clusters explain {sum(evals)/tau} and "
                  f"{sum(cluster_explained_var)/tau} of total variance.")

        return (cluster_index,  # cluster_index for each node
                cluster_fit,    # fraction of node's variance explained by
                                # chosen eigenvector, for each node
                evals,          # eigenvalues
                evecs)          # matrix with columns=eigenvectors

    def do_nsi_clustering(self, d0=None, tree_dotfile=None,
                          distances=None, candidates=None):
        """
        Perform agglomerative clustering based on representation accuracy.

        This minimizes in each step the mean squared error of representing the
        pairwise node distances by their cluster averages.

        .. note::
           This is still EXPERIMENTAL!

        See the code for arguments and return value.

        Clusters 0...n-1 are the singletons (cluster i containing just node i).
        Clusters n...2n-2 are numbered in the order in which clusters are
        joined (a cluster with id c is a union of two earlier clusters with
        ids c1,c2 < c). In particular, cluster 2n-2 is the full set of nodes.

        :rtype:  dictionary
        :return: A dictionary containing the following keys:

           - "min_clusters": int > 0. Smallest number of clusters generated.
           - "error": array(n+1). Entry [k] is the representation error for the
             solution with k clusters.
           - "node2cluster": array(n,n+1). Entry [i,k] is the id of the cluster
             that contains node i in the solution with k clusters.
           - "cluster_weight": array(2n-1). Entry [c] is the total weight of
             cluster c.
           - "cluster2rank": array(2n-1,n+1). Entry [c,k] is the descending
             order rank of cluster c in the k-cluster solution, i.e., the
             number of larger clusters in that solution. Use this to convert
             cluster ids in 0...2n-1 to cluster ids in 0...k-1.
           - "node_in_cluster": array(n,2n-1). Entry [i,c] indicates whether
             node i is in the cluster with id c.
           - "children": array(2n-1,2). Entries [c,0] and [c,1] are the ids of
             the two clusters that were joined to give cluster c.
           - "sibling": array(2n-2). Entry [c] is the id of the cluster with
             which cluster c is joined.
           - "parent": array(2n-2). Entry [c] is the id of the cluster that
             results from joining cluster c with its sibling.
        """
        N = self.N
        N2 = 2*N - 1
        rN = range(N)
        w = self.node_weights.copy()
        k = self.nsi_degree()  # TODO: link weight

        # init result structures:
        error = np.zeros(N+1) + np.inf
        error[-1] = 0.0
        node2cluster = np.zeros((N, N+1), dtype=np.int16)
        node2cluster[:, 0] = -1
        node2cluster[:, N] = rN
        cluster_weight = np.zeros(N2)
        cluster_weight[0:N] = w
        cluster2rank = np.zeros((N2, N+1), dtype=np.int16) - 1
        cluster2rank[0, 1] = 0
        node_in_cluster = np.zeros((N, N2), dtype=np.int8)  # or bool?
        children = np.zeros((N2, 2), dtype=np.int16)
        children[:N] = -1
        sibling = np.zeros(N2-1, dtype=np.int16) - 1
        parent = np.zeros(N2-1, dtype=np.int16) - 1
        clid = np.arange(N)

        # a dynamic doubly linked list of distance matrix entries:
        #  D_firstpos[cl] = pos. of first nb. of cl.
        #  D_lastpos[cl] = pos. of last nb. of cl.
        #  D_nextpos[pos] = pos. of next nb. of the same cl.
        #  D_prevpos[pos] = pos. of previous nb. of the samle cl.
        #  D_cluster[pos] = cluster index of the neighbour at this pos.
        #  D_invpos[pos] = pos. of cl. in nbs. list of nbs.
        # all needed link attributes are stored with the same pos.
        # when cls are joined, the resp. lists are concatenated and duplicates
        # are unlinked (but their pos. not reused), so we need 2M many
        # pos., 1...2M, where pos 0 remains empty:
        if distances is None:
            # contains each link twice!
            distance_keys = nz_coords(self.sp_A)
        else:
            try:
                distance_keys = distances.keys()
            except AttributeError:
                distance_keys = [(i, j) for i in range(N) for j in range(N)]
        M = len(distance_keys)
        rM = range(M)
        rpos = range(1, M+1)
        # if M < 65535:
        #     postype = "int16"
        postype = "int32"
        D_firstpos = np.zeros(N, postype)  # pos. of first nb. of cl.
        D_lastpos = np.zeros(N, postype)  # pos. of last nb. of cl.
        # pos. of next nb. of the same cl.
        D_nextpos = np.zeros(2*M+1, postype)
        # pos. of previous nb. of the samle cl.
        D_prevpos = np.zeros(2*M+1, postype)
        # pos. of cl. in nbs. list of nbs.
        D_invpos = np.zeros(2*M+1, postype)
        # cluster index of the neighbour at this pos.
        D_cluster = np.zeros(2*M+1, "int16")

        # compute average distance of unconnected pairs,
        # which will be used as an estimate for them:
        n_pairs = N * (N-1) / 2
        if d0 is None:
            t0 = time.time()
            if n_pairs > M:
                d0 = (self.average_path_length()*1.0 * n_pairs - M) /\
                     (n_pairs - M)  # TODO: link weight
            else:
                d0 = 1.0 * N
            print(f"calculated {d0} as average non-linked distance, "
                  f"needed {time.time()-t0} sec.")

        ftype = "float32"
        dict_D = {}  # weighted sum of distances between clusters
        dict_Delta = {}  # error increase upon join, only i<j

        # init the list:
        t0 = time.time()
        posj = 0
        posi = M
        for i0, j0 in distance_keys:
            if i0 == j0:
                dict_D[(N+1)*i0] = w[i0] * distances[i0, i0]
                continue
            if i0 < j0:
                i, j = i0, j0
            else:
                i, j = j0, i0
            ij = i*N+j
            if ij in dict_D:
                continue
            posj = posj + 1
            if D_firstpos[i] == 0:
                D_firstpos[i] = D_lastpos[i] = posj
            else:
                D_prevpos[posj] = lpos = D_lastpos[i]
                D_nextpos[lpos] = D_lastpos[i] = posj
            D_cluster[posj] = j
            if distances is None:
                # i.e., use dist 1 if linked, d0 otherwise
                Dij = dict_D[ij] = dict_D[j*N+i] = w[i] * w[j]
            else:
                Dij = dict_D[ij] = dict_D[j*N+i] = \
                    w[i] * w[j] * distances[i0, j0]
            D_invpos[posj] = posi = posi + 1
            if D_firstpos[j] == 0:
                D_firstpos[j] = D_lastpos[j] = posi
            else:
                D_prevpos[posi] = lpos = D_lastpos[j]
                D_nextpos[lpos] = D_lastpos[j] = posi
            D_cluster[posi] = i
            D_invpos[posi] = posj
        del distance_keys
        print("initialization of distances needed", time.time()-t0, "sec.")

        # init candidates:
        t0 = time.time()
        if candidates is None:
            candidates = nz_coords(self.sp_A)
        for i0, j0 in candidates:
            if i0 < j0:
                i, j = i0, j0
            else:
                i, j = j0, i0
            ij = i*N+j
            if ij in dict_Delta:
                continue
            wi = w[i]
            wj = w[j]
            wc = wi + wj
            wiwj = wi * wj
            if ij in dict_D:
                Dcc_wc2 = 2 * dict_D.get(ij, wiwj*d0) / wc**2
            else:
                dict_D[ij] = dict_D[j*N+i] = wiwjd0 = wiwj*d0
                Dcc_wc2 = 2 * wiwjd0 / wc**2
            dict_Delta[ij] = (wi**2 + wj**2) * (Dcc_wc2)**2 + \
                2 * wiwj * (Dcc_wc2-1)**2
        print("initialization of candidates needed", time.time()-t0, "sec.")

        t0 = time.time()
        cands = dict_Delta.keys()
        n_cands = len(cands)

        dict_Delta = _do_nsi_clustering_I(n_cands, cands, D_cluster, w, d0,
                                          D_firstpos, D_nextpos, N, dict_D,
                                          dict_Delta)

        print(f"initialization of error increments needed"
              f"{time.time()-t0} sec.")

        # successively join the best pair:
        sumt1 = sumt2 = sumt3 = 0.0
        actives = range(N)
        min_clusters = 1
        for n_clusters in range(N-1, 0, -1):

            # find best pair a<b:
            t0 = time.time()
            vals = dict_Delta.values()
            if not vals:
                min_clusters = n_clusters + 1
                break
            minpos = np.argmin(vals)
            ab = dict_Delta.keys()[minpos]
            del dict_Delta[ab]
            a = ab / N
            b = ab % N
            this_error = vals[minpos]
            sumt1 += time.time()-t0

            # remove duplicates in D and rewire nbs c1 of b to point to a:
            delkeys = [(b, b)]
            t0 = time.time()
            lpos = D_lastpos[a]
            D_nextpos[lpos] = posc1 = D_firstpos[b]
            D_prevpos[posc1] = lpos
            D_lastpos[a] = D_lastpos[b]
            while posc1 != 0:
                c1 = D_cluster[posc1]
                delkeys += [(c1, b)]
                if c1 < a:
                    c1akey = c1*N+a
                else:
                    c1akey = a*N+c1
                if c1 < b:
                    c1bkey = c1*N+b
                else:
                    c1bkey = b*N+c1
                if c1bkey in dict_Delta:  # rewire cand. c1-b to c1-a:
                    del dict_Delta[c1bkey]
                    dict_Delta[c1akey] = 0.0  # will later be recomputed!
                if c1 == a or c1akey in dict_D:
                    iposc1 = D_invpos[posc1]
                    ippos = D_prevpos[iposc1]
                    inpos = D_nextpos[iposc1]
                    if ippos > 0:
                        D_nextpos[ippos] = inpos
                    else:
                        D_firstpos[c1] = inpos
                    if inpos > 0:
                        D_prevpos[inpos] = ippos
                    else:
                        D_lastpos[c1] = ippos
                    ppos = D_prevpos[posc1]
                    posc1 = D_nextpos[posc1]
                    if ppos > 0:
                        D_nextpos[ppos] = posc1
                    else:
                        D_firstpos[a] = posc1
                    if posc1 > 0:
                        D_prevpos[posc1] = ppos
                    else:
                        D_lastpos[a] = ppos
                else:
                    D_cluster[D_invpos[posc1]] = a
                    posc1 = D_nextpos[posc1]
            D_firstpos[b] = D_lastpos[b] = 0
            sumt2 += time.time()-t0

            # TODO: this is the bottleneck, so speed it up:
            # first update Delta[a1,b1] for each pair a1,b1 with a1 linked to c
            # and b1 != c, and compute Delta[a1,c] for each a1 linked to c:
            wa = w[a]
            wb = w[b]
            wc = wa + wb
            wad0 = wa * d0
            wbd0 = wb * d0

            t0 = time.time()

            dict_Delta = _do_nsi_clustering_II(a, b, D_cluster, w, d0,
                                               D_firstpos, D_nextpos, N,
                                               dict_D, dict_Delta)

            sumt3 = time.time()-t0

            # finally update D:
            Daa = dict_D.get(a*(N+1), 0.0)
            Dbb = dict_D.get(b*(N+1), 0.0)
            dict_D[a*(N+1)] = Daa + Dbb + 2*dict_D[a*N+b]
            posc1 = D_firstpos[a]
            while posc1 > 0:
                c1 = D_cluster[posc1]
                Dac1 = dict_D.get(a*N+c1, w[c1]*wad0)
                Dbc1 = dict_D.get(b*N+c1, w[c1]*wbd0)
                dict_D[c1*N+a] = dict_D[a*N+c1] = Dac1 + Dbc1
                posc1 = D_nextpos[posc1]

            # update result structures:
            c = N2 - n_clusters
            error[n_clusters] = error[n_clusters+1] + this_error
            # TODO: node2cluster
            cluster_weight[c] = wc
            # TODO: cluster2rank
            # TODO: node_in_cluster
            children[c, 0] = ca = clid[a]
            children[c, 1] = sibling[ca] = cb = clid[b]
            sibling[cb] = ca
            parent[ca] = parent[cb] = c
            parent[c] = N2 - 1  # initially, until joined.

            # remove b and replace a by c:
            for k1, k2 in delkeys:
                try:
                    del dict_D[k1*N+k2], dict_D[k2*N+k1]
                except KeyError:
                    pass
            actives.remove(b)
            clid[a] = c
            w[a] = wc

            print(n_clusters, ": joining", ca, cb, "to", c, "at", this_error)
            if n_clusters < 10:
                print("D", dict_D)
                print("Delta", dict_Delta)

        print("part 1 needed", sumt1, "sec.")
        print("part 2 needed", sumt2, "sec.")
        print("part 3 needed", sumt3, "sec.")

        if tree_dotfile is not None:
            # use penwidth and len!
            edges = [(int(n), int(parent[n])) for n in range(N2-1)]
            minlen = [int(parent[n]-max(n, N-1)) for n in range(N2-1)]
            # TODO: eps + error difference
            edgelen = np.array(
                [max(0.0, error[N2-parent[n]]) for n in range(N)]
                + [max(0.0, error[N2-parent[n]]-error[N2-n])
                   for n in range(N, N2-1)])  # minlen
            # TODO: 1/(eps + error difference)
            # [1.0 for i in range(N2-1)]
            penwidth = 30.0 / (1.0 + 29.0*edgelen/edgelen.max())
            tree = igraph.Graph(edges, directed=True)
            tree.es.set_attribute_values("minlen", minlen)
            tree.es.set_attribute_values("len", edgelen)
            tree.es.set_attribute_values("penwidth", penwidth)
            tree["rankdir"] = "BT"
            tree.write_dot(tree_dotfile)
            del tree

        return {
            "min_clusters": min_clusters, "node2cluster": node2cluster,
            "cluster2rank": cluster2rank, "cluster_weight": cluster_weight,
            "node_in_cluster": node_in_cluster, "error": error,
            "children": children, "sibling": sibling, "parent": parent
        }

    def do_nsi_hamming_clustering(self, admissible_joins=None, alpha=0.01,
                                  tree_dotfile=None):
        """
        Perform agglomerative clustering based on Hamming distances.

        This minimizes in each step the Hamming distance between the original
        and the "clustered" network.

        .. note::
           This is still EXPERIMENTAL!

        See the code for arguments and return value.

        Clusters 0...n-1 are the singletons (cluster i containing just node i).
        Clusters n...2n-2 are numbered in the order in which clusters are
        joined (a cluster with id c is a union of two earlier clusters with
        ids c1,c2 < c). In particular, cluster 2n-2 is the full set of nodes.

        :rtype:  dictionary
        :return: A dictionary containing the following keys:

           - "error": array(n+1). Entry [k] is the representation error for the
             solution with k clusters.
           - "node2cluster": array(n,n+1). Entry [i,k] is the id of the cluster
             that contains node i in the solution with k clusters.
           - "cluster_weight": array(2n-1). Entry [c] is the total weight of
             cluster c.
           - "cluster2rank": array(2n-1,n+1). Entry [c,k] is the descending
             order rank of cluster c in the k-cluster solution, i.e., the
             number of larger clusters in that solution. Use this to convert
             cluster ids in 0...2n-1 to cluster ids in 0...k-1.
           - "node_in_cluster": array(n,2n-1). Entry [i,c] indicates whether
             node i is in the cluster with id c.
           - "children": array(2n-1,2). Entries [c,0] and [c,1] are the ids of
             the two clusters that were joined to give cluster c.
           - "sibling": array(2n-2). Entry [c] is the id of the cluster with
             which cluster c is joined.
           - "parent": array(2n-2). Entry [c] is the id of the cluster that
             results from joining cluster c with its sibling.
        """
        # took about 15h on Zuse for HadCM3 globe
        # ?takes about 90*(N/800)^4 seconds on a 1.67 GHz i686,
        # which makes about 10 days for N=8000 (e.g. a HadCM3 globe)

        t0 = time.time()

        # initialize data structures:

        n = self.N
        n2 = 2*n-1
        w = self.node_weights
        WW = self.total_node_weight**2

        # join admissibility matrix:
        if admissible_joins is None:
            print("all joins admissible")
            mayJoin = np.zeros((n2, n2), dtype=int) + 1
        else:
            print("only some joins admissible")
            mayJoin = np.zeros((n2, n2), dtype=int)
            mayJoin[0:n, 0:n] = admissible_joins
        # cluster membership indicators:
        clusterMembers = np.zeros((n2, n), dtype=int)
        clusterMembers[0:n, 0:n] = np.identity(n)
        # cluster weights:
        clusterWeights = np.zeros(n2)
        clusterWeights[0:n] = w
        # weight products:
        weightProducts = np.zeros((n2, n2))
        weightProducts[0:n, 0:n] = np.dot(w.reshape((n, 1)), w.reshape((1, n)))
        # linked weights:
        A, Aplus = self.adjacency, self.sp_Aplus().A
        linkedWeights = np.zeros((n2, n2))
        linkedWeights[0:n, 0:n] = \
            self.node_weights.reshape((n, 1)) * Aplus * \
            self.node_weights.reshape((1, n))
        # error contributions of cluster pairs
        # (sum up to total error = 2*Hamming distance):
        errors = np.zeros((n2, n2))
        # distance = increase of Hamming distance:
        # and find first pair to join:
        distances = np.zeros((n2, n2))

        # list of active cluster indices:
        activeIndices = range(0, n)

        # final Hamming distances:
        hamming = np.zeros(n2)

        # list of parents and siblings:
        sibling = np.zeros(n2-1, dtype=int)
        parent = np.zeros(n2-1, dtype=int)

        # list of parts:
        parts = np.zeros((n2, 2), dtype=int)
        parts[:n] = -1

        node2cluster = np.zeros((n, n+1), dtype=int)
        node2cluster[:, 0] = -1
        node2cluster[:, n] = range(n)
        cluster2rank = np.zeros((n2, n+1), dtype=int) - 1
        cluster2rank[0, 1] = 0

        lastunited = part1 = part2 = -1

        # iteratively join those two clusters which increase Hamming distance
        # the least:
        for united in range(n, n2):

            # find next optimal pair:

            nActiveIndices = len(activeIndices)
            theActiveIndices = np.sort(activeIndices)
            mind0 = float(np.power(1.0*self.total_node_weight, 3.0))
            minwp0 = float(2.0*weightProducts.max())
            result = np.zeros(3)

            results = _do_nsi_hamming_clustering(
                n2, nActiveIndices, mind0, minwp0, lastunited, part1,
                part2, distances.copy(mode='c'),
                theActiveIndices.copy(mode='c'),
                linkedWeights.copy(mode='c'),
                weightProducts.copy(mode='c'),
                errors.copy(mode='c'), result.copy(mode='c'),
                mayJoin.copy(mode='c'))

            mind = result[0]
            part1 = int(result[1])
            part2 = int(result[2])
            if mind < 0:
                print(united, mind, part1, part2)
                raise Exception

            cluster2rank[np.array(activeIndices)[
                (-clusterWeights[activeIndices]).argsort()], n2+1-united] = \
                range(n2+1-united)

            hamming[united] = hamming[united-1] + 2.0 * mind

            if united < n + 100 or united % (1 + n2/100) == 0 or \
                    united >= n2 - 100:
                print(f"for {n2-united} clusters with error "
                      f"{hamming[united]/WW} we join clusters "
                      f"{part1} and {part2} to get cluster {united}")
                sys.stdout.flush()

            # unite parts:

            parent[part1] = parent[part2] = united
            parts[united, 0] = sibling[part2] = part1
            parts[united, 1] = sibling[part1] = part2
            clusterMembers[united, :] = \
                clusterMembers[part1, :] + clusterMembers[part2, :]
            node2cluster[:, n2-united] = \
                node2cluster[:, 1+n2-united] * (1-clusterMembers[united, :]) +\
                united*clusterMembers[united, :]
            activeIndices.remove(part1)
            activeIndices.remove(part2)
            activeIndices.append(united)

            # compute new entries in clusterWeights, weightProducts,
            # linkedWeights, errors, mayJoin:
            clusterWeights[united] = \
                clusterWeights[part1] + clusterWeights[part2]
            weightProducts[united, 0:united] = \
                weightProducts[part1, 0:united] + \
                weightProducts[part2, 0:united]
            weightProducts[0:united, united] = \
                weightProducts[united, 0:united].flatten()
            weightProducts[united, united] = \
                np.power(clusterWeights[united], 2.0)
            linkedWeights[united, 0:united] = \
                linkedWeights[part1, 0:united] + linkedWeights[part2, 0:united]
            linkedWeights[0:united, united] = \
                linkedWeights[united, 0:united].flatten()
            linkedWeights[united, united] = \
                linkedWeights[part1, part1] + linkedWeights[part2, part2] + \
                2.0 * linkedWeights[part1, part2]
            mayJoin[united, 0:united] = \
                mayJoin[part1, 0:united] + mayJoin[part2, 0:united]
            mayJoin[0:united, united] = mayJoin[united, 0:united].flatten()
            for c in range(0, united):
                lw = linkedWeights[united, c]
                errors[united, c] = errors[c, united] = \
                    min(lw, weightProducts[united, c] - lw)
            errors[united, united] = \
                weightProducts[united, united] - linkedWeights[united, united]
            if errors.min() < -1e-10:
                print(errors)
                raise Exception
            lastunited = united

        print(time.time()-t0, "seconds")

        # node2cluster = np.array(range(0, n2)).reshape((n2, 1))*clusterMembers

        node_in_cluster = clusterMembers.T.astype(int)
        error = np.zeros(n+1)
        error[0] = np.inf
        error[-1-np.arange(n)] = hamming[-n:] / WW

        if tree_dotfile is not None:
            edges = [(int(i), int(parent[i])) for i in range(n2-1)]
            minlen = [int(parent[i]-max(i, n-1)) for i in range(n2-1)]
            tree = igraph.Graph(edges, directed=True)
            tree.es.set_attribute_values("minlen", minlen)
            tree["rankdir"] = "LR"
            tree.write_dot(tree_dotfile)
            del tree

        return {
            "node2cluster": node2cluster, "cluster2rank": cluster2rank,
            "cluster_weight": clusterWeights,
            "node_in_cluster": node_in_cluster, "error": error,
            "children": parts, "sibling": sibling, "parent": parent
        }
