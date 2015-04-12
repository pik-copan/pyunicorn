#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# # Copyright (C) 2014 Paul Schultz and the SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
#   Paul Schultz pschultz@pik-potsdam.de>
#
#
# References: [Schultz2014]_, [Schultz2014a]_,
#
# ToDo:
#  - included Schultz' Predictor and growth algorithm (among others)


"""
Module contains class ResNetwork.

Provides function for computing resistance based networks. It is subclassed
from GeoNetwork and provides most GeoNetwork's functions/properties.

The class has the following instance variables::

    (bool) flagDebug     : flag for debugging mode
    (bool) flagWeave     : flag for switching between python/C code parts
    (bool) flagComplex   : flag for complex input
    (ndarray) resistances: array of resistances (complex or real)

Overriden inherited methods::

    (str) __str__          : extended description
    (ndarray) get_adjacency: returns complex adjacency if needed
"""

# Import things we inherit from

from . import GeoNetwork
from . import Grid

# a network Error (use uncertain)
# from . import NetworkError

#  Import NumPy for the array object and fast numerics
import numpy as np

# Import handler for sparse matrices
from scipy import sparse

#  Import iGraph for high performance graph theory tools written in pure ANSI-C
import igraph

# Weave for inline C
try:
    from scipy import weave
    flagWeave = True

except ImportError:
    try:
        import weave
        flagWeave = True

    except ImportError:
        print "Could not import weave. Using fallback Python code instead of C"
        print "Consider installing weave for a *significant* speed-up"
        flagWeave = False


class ResNetwork(GeoNetwork):
    """ A resistive network class

    ResNetwork, provides methods for an extended analysis of
    resistive/resistance-based networks.

    **Examples:**

    >>> print GeoNetwork.SmallTestNetwork()
    Undirected network, 6 nodes, 7 links, link_density 0.466666666667
    Geographical network boundaries:
             time     lat     lon
       min    0.0    0.00    2.50
       max    9.0   25.00   15.00
    """
###############################################################################
# ##                       MAGIC FUNCTIONS                                 ## #
###############################################################################

    def __init__(self, resistances, grid=None, adjacency=None, edge_list=None,
                 directed=False, node_weight_type=None, silence_level=2):

        """Initialize an instance of ResNetwork.

        :type resistances: 2D NumPy array
        :arg resistances: A matrix with the resistances

        :type grid: Grid object
        :arg grid: The Grid object describing the network's spatial embedding.

        :type adjacency: 2D NumPy array (int8) [index, index]
        :arg adjacency: The network's adjacency matrix.

        :type edge_list: array-like list of lists
        :arg  edge_list: Edge list of the new network.
                         Entries [i,0], [i,1] contain the end-nodes of an edge.

        :type directed: boolean
        :arg directed: Determines, whether the network is treated as directed.

        :type node_weight_type: string
        :arg node_weight_type: The type of geographical node weight to be used.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.

        **Examples:**

        >>> print ResNetwork.SmallTestNetwork()
        Undirected network, 5 nodes, 5 links, link_density 0.5
        Geographical network boundaries:
                 time     lat      lon
           min    0.0    0.00  -180.00
           max    9.0   90.00   180.00
        Average resistance: 2.4
        """

        # 1) prepare params so we can init() the parent

        # 1a) an adjacency matrix
        #     by default, if only resistance are given,
        #     these define the adjacency
        if adjacency is None:
            if silence_level < 2:
                print "Using adjacency as definded by resistances"

            adjacency = np.zeros(resistances.shape)
            adjacency[resistances != 0] = 1

        # 1b) a Grid object
        #     an actual grid might not exist, so we fake one
        if grid is None:
            if silence_level < 2:
                print "Using dummy grid"
            grid = Grid(time_seq=np.arange(10), lat_seq=np.absolute(
                np.linspace(-90, 90, adjacency.shape[0])),
                lon_seq=np.linspace(-180, 180, adjacency.shape[0]),
                silence_level=0)

        # 2) init parent
        GeoNetwork.__init__(self, grid, adjacency=adjacency,
                            node_weight_type=node_weight_type,
                            silence_level=silence_level)

        # 3) add ResNetwork specific parts

        # 3a) set the resitance values
        #     this sets the property and forces the
        #     updating of the admittance and R
        self.update_resistances(resistances)

        # 3b) switch weave support internally as well
        self.flagWeave = flagWeave

    def __str__(self):
        """Return a short summary of the resistive network.

        :return: A short summary of the network
        :rtype: string


        **Examples:**

        >>> res =  ResNetwork.SmallTestNetwork(); print res
        Undirected network, 5 nodes, 5 links, link_density 0.5
        Geographical network boundaries:
                 time     lat     lon
           min    0.0    0.00 -180.00
           max    9.0   90.00  180.00
        Average resistance: 2.4
        >>> type(res.__str__())
        <type 'str'>
        """

        text = GeoNetwork.__str__(self)
        text += "\nAverage resistance: " + str(self.resistances.mean())

        return text

###############################################################################
# ##                       PUBLIC FUNCTIONS                                ## #
###############################################################################
    @staticmethod
    def SmallTestNetwork():
        """
        Create a small test network with unit resistances of the following
        topology::

            0------1--------3------4
                    \      /
                     \    /
                      \  /
                       \/
                       2

        :rtype: Resistive Network instance
        :return: an ResNetwork instance for testing purposes.

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> isinstance(res,ResNetwork)
        True
        """
        adjacency = np.array([[0, 1, 0, 0, 0],
                              [1, 0, 1, 1, 0],
                              [0, 1, 0, 1, 0],
                              [0, 1, 1, 0, 1],
                              [0, 0, 0, 1, 0]], dtype='int8')

        # sample symmetric resistances w/ rational reciprocals
        # resistances = np.array([2,2,8,2,8,8,2,8,10,10])

        # # the resistances should be a full matrix
        resistances = np.array([[0, 2, 0, 0, 0],
                                [2, 0, 8, 2, 0],
                                [0, 8, 0, 8, 0],
                                [0, 2, 8, 0, 10],
                                [0, 0, 0, 10, 0]])
        # a grid
        grid = Grid(time_seq=np.arange(10), lat_seq=np.absolute(
            np.linspace(-90, 90, adjacency.shape[0])),
            lon_seq=np.linspace(-180, 180, adjacency.shape[0]),
            silence_level=0)

        return ResNetwork(resistances, grid=grid, adjacency=adjacency)

    @staticmethod
    def SmallComplexNetwork():
        """
        A test network with complex resistances analogue to
        SmallTestNetwork()

        :rtype: Resistive Network instance
        :return: an ResNetwork instance with complex resistances

        **Examples:**

        >>> res = ResNetwork.SmallComplexNetwork()
        >>> isinstance(res,ResNetwork)
        True
        >>> res.flagComplex
        True
        >>> adm = res.get_admittance()
        >>> print adm.real
        [[ 0.      0.1     0.      0.      0.    ]
         [ 0.1     0.      0.0625  0.25    0.    ]
         [ 0.      0.0625  0.      0.0625  0.    ]
         [ 0.      0.25    0.0625  0.      0.05  ]
         [ 0.      0.      0.      0.05    0.    ]]

        >>> print adm.imag
        [[ 0.     -0.2     0.      0.      0.    ]
         [-0.2     0.     -0.0625 -0.25    0.    ]
         [ 0.     -0.0625  0.     -0.0625  0.    ]
         [ 0.     -0.25   -0.0625  0.     -0.05  ]
         [ 0.      0.      0.     -0.05    0.    ]]

        """

        resistances = np.zeros((5, 5), dtype=complex)
        resistances.real = [[0, 2, 0, 0, 0],
                            [2, 0, 8, 2, 0],
                            [0, 8, 0, 8, 0],
                            [0, 2, 8, 0, 10],
                            [0, 0, 0, 10, 0]]
        resistances.imag = [[0, 4, 0, 0, 0],
                            [4, 0, 8, 2, 0],
                            [0, 8, 0, 8, 0],
                            [0, 2, 8, 0, 10],
                            [0, 0, 0, 10, 0]]

        return ResNetwork(resistances)

    def update_resistances(self, resistances):
        """ Update the resistance matrix

        This function is called to changed the resistance matrix. It sets the
        property and the calls the :meth:`update_admittance` and
        :meth:`update_R` functions.

        :rtype: None

        **Examples:**

        >>> # test network with given resistances
        >>> res = ResNetwork.SmallTestNetwork()
        >>> print res.resistances
        [[ 0  2  0  0  0]
         [ 2  0  8  2  0]
         [ 0  8  0  8  0]
         [ 0  2  8  0 10]
         [ 0  0  0 10  0]]
        >>> # print admittance and admittance Laplacian
        >>> print res.get_admittance()
        [[ 0.     0.5    0.     0.     0.   ]
         [ 0.5    0.     0.125  0.5    0.   ]
         [ 0.     0.125  0.     0.125  0.   ]
         [ 0.     0.5    0.125  0.     0.1  ]
         [ 0.     0.     0.     0.1    0.   ]]
        >>> print res.admittance_lapacian()
        [[ 0.5   -0.5    0.     0.     0.   ]
         [-0.5    1.125 -0.125 -0.5    0.   ]
         [ 0.    -0.125  0.25  -0.125  0.   ]
         [ 0.    -0.5   -0.125  0.725 -0.1  ]
         [ 0.     0.     0.    -0.1    0.1  ]]
        >>> # now update to unit resistance
        >>> res.update_resistances(res.adjacency)
        >>> # and check new admittance/admittance Laplacian
        >>> print res.get_admittance()
        [[ 0.  1.  0.  0.  0.]
         [ 1.  0.  1.  1.  0.]
         [ 0.  1.  0.  1.  0.]
         [ 0.  1.  1.  0.  1.]
         [ 0.  0.  0.  1.  0.]]
        >>> print res.admittance_lapacian()
        [[ 1. -1.  0.  0.  0.]
         [-1.  3. -1. -1.  0.]
         [ 0. -1.  2. -1.  0.]
         [ 0. -1. -1.  3. -1.]
         [ 0.  0.  0. -1.  1.]]
        """

        # ensure ndarray
        if not isinstance(resistances, np.ndarray):
            resistances = np.array(resistances)

        # check complex/real
        if np.iscomplexobj(resistances):
            self.flagComplex = True
        else:
            self.flagComplex = False

        # set property
        self.resistances = resistances

        # update the admittance
        self.update_admittance()

        # and update R
        self.update_R()

    def update_admittance(self):
        """Updates admittance matrix which is inverse the resistances

        :rtype: none

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork();print res.get_admittance()
        [[ 0.     0.5    0.     0.     0.   ]
         [ 0.5    0.     0.125  0.5    0.   ]
         [ 0.     0.125  0.     0.125  0.   ]
         [ 0.     0.5    0.125  0.     0.1  ]
         [ 0.     0.     0.     0.1    0.   ]]
        >>> print type(res.get_admittance())
        <type 'numpy.ndarray'>

        """
        # a sparse matrix for the admittance values
        # we start w/ a lil_matrix, maybe convert that
        # to csr_matrix ( Compressed Sparse Row) or
        # to csc_matrix ( Compressed Sparse Column) later

        # complex number support
        if self.flagComplex:
            dtype = complex
        else:
            dtype = float

        self.sparse_Adm = sparse.lil_matrix((self.N, self.N), dtype=dtype)

        # get the edges
        edgeList = list(self.edge_list())

        # populate array
        for edge in edgeList:
            # print "setting %d %d to %f" % (
            #     edge[0], edge[1], 1./self.resistances[edge[0], edge[1]])
            self.sparse_Adm[edge[0], edge[1]] = \
                1./self.resistances[edge[0], edge[1]]

        # Similar to GeoNetwork, we embed an iGraph instance for
        # the admittance matrix
        self.adm_graph = igraph.Graph(n=self.N, edges=edgeList,
                                      directed=self.directed)
        self.graph.simplify()

    def get_admittance(self):
        """Return the (possibly non-symmetric) dense admittance matrix

        :rtype: square NumPy matrix [node,node] of ints

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork();print res.get_admittance()
        [[ 0.     0.5    0.     0.     0.   ]
         [ 0.5    0.     0.125  0.5    0.   ]
         [ 0.     0.125  0.     0.125  0.   ]
         [ 0.     0.5    0.125  0.     0.1  ]
         [ 0.     0.     0.     0.1    0.   ]]
        >>> print type( res.get_admittance() )
        <type 'numpy.ndarray'>
        """
        return np.array(self.sparse_Adm.todense())

    def update_R(self):
        """Updates R, the pseudo inverse of the admittance Laplacian

        This function is run, whenever the admittance is changed.

        :rtype: none

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork();print res.get_admittance()
        [[ 0.     0.5    0.     0.     0.   ]
         [ 0.5    0.     0.125  0.5    0.   ]
         [ 0.     0.125  0.     0.125  0.   ]
         [ 0.     0.5    0.125  0.     0.1  ]
         [ 0.     0.     0.     0.1    0.   ]]
        >>> print type( res.get_admittance() )
        <type 'numpy.ndarray'>

        """
        # a sparse matrix for the admittance values
        self.sparse_R = sparse.lil_matrix(
            np.linalg.pinv(self.admittance_lapacian()))

    def get_R(self):
        """Return the pseudo inverse of of the admittance Laplacian

        The pseudoinverse is used of the novel betweeness  measures such as
        :meth:`vertex_current_flow_betweenness` and
        :meth:`edge_current_flow_betweenness` It is computed on instantiation
        and on change of the resistances/admittance

        :returns: the pseudoinverse of the admittange Laplacian
        :rtype: ndarray (float)

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork();print res.get_R()
        [[ 2.28444444  0.68444444 -0.56       -0.20444444 -2.20444444]
         [ 0.68444444  1.08444444 -0.16        0.19555556 -1.80444444]
         [-0.56       -0.16        3.04       -0.16       -2.16      ]
         [-0.20444444  0.19555556 -0.16        1.08444444 -0.91555556]
         [-2.20444444 -1.80444444 -2.16       -0.91555556  7.08444444]]
        """
        return np.array(self.sparse_R.todense())

    def admittance_lapacian(self):
        """
        Return the (possibly non-symmetric) dense Laplacian matrix of the
        admittance.

        :rtype: square NumPy matrix [node,node] of

        **Examples:**

        >>> print ResNetwork.SmallTestNetwork().admittance_lapacian()
        [[ 0.5   -0.5    0.     0.     0.   ]
         [-0.5    1.125 -0.125 -0.5    0.   ]
         [ 0.    -0.125  0.25  -0.125  0.   ]
         [ 0.    -0.5   -0.125  0.725 -0.1  ]
         [ 0.     0.     0.    -0.1    0.1  ]]
        >>> print type( ResNetwork.SmallTestNetwork().admittance_lapacian() )
        <type 'numpy.ndarray'>

         """

        return (np.diag(sum(self.get_admittance())) - self.get_admittance())

    def admittive_degree(self):
        """admittive degree of the network

        The admittive (or effective) degree of the resistive network,
        which is the counterpart to the traditional degree.

        :rtype: 1D NumPy array

        **Examples:**

        >>> print ResNetwork.SmallTestNetwork().admittive_degree()
        [ 0.5    1.125  0.25   0.725  0.1  ]
        >>> print type( ResNetwork.SmallTestNetwork().admittive_degree())
        <type 'numpy.ndarray'>
        """
        return np.sum(self.get_admittance(), axis=0)

    def average_neighbors_admittive_degree(self):
        """ Average neighbour effective degree

        :rtype: 1D NumPy array

        **Examples:**

        >>> print ResNetwork.SmallTestNetwork().\
                average_neighbors_admittive_degree()
        [ 2.25  1.31111111  7.4  2.03448276  7.25  ]
        >>> print type(ResNetwork.SmallTestNetwork().admittive_degree())
        <type 'numpy.ndarray'>

        """

        # get the admittive degree (row sum)
        # and adjacency matrix
        ad = self.admittive_degree()
        adj = self.adjacency

        # in case of complex resistances we also use the dot product,
        # but of complex-valued arrays
        # This is NOT right, BUT:
        # np.dot treats complex numbers wrongly and computes the
        # dot product of the real and the imag part seperately
        # which in our case is exactly what we want
        if self.flagComplex:
            adj = np.array(adj, dtype=complex)
            adj.imag = adj.real

        # dot product of adjacency and degree
        # normalised by the row sum (admittive degree)
        return np.dot(adj, ad) / ad

        # to sweave later on:
        # N = self.N
        # adjacency = self.adjacency
        # ANED = np.zeros(N)
        # ED = self.admittive_degree()
        # for i in xrange(N):
        #     sum = 0
        #     for j in xrange(N):
        #         sum += adjacency[i][j]*ED[j]
        #     ANED[i] = sum/ED[i]
        # # print ANED

    def local_admittive_clustering(self):
        r"""
        Return node wise admittive clustering coefficient (AC).

        The AC is the electrical analogue of the clustering coefficient for
        regular network (see :meth:`.get_admittive_ws_clustering` and
        :meth:`.get_local_clustering` and sometimes called Effective Clustering
        (EC))

        The admittive clustering (:math:`ac`) of node :math:`i` is defined as:

        .. math::
            \text{ac}_i = \frac
                {\sum_{j,k}^N\alpha_{i,j},\alpha_{i,k},\alpha_{j,k}}
                {\text{ad}_i(\text{d}_i-1)}

        where
            - :math:`\alpha` is the admittance matrix
            - :math:`ad_i` is the admittive degree of the node :math:`i`
            - :math:`d_i` is the degree of the node :math:`i`

        :rtype: 1d NumPy array (float)

        **Examples:**

        >>> res =  ResNetwork.SmallTestNetwork()
        >>> print res.local_admittive_clustering()
        [ 0.  0.00694444  0.0625  0.01077586  0. ]
        >>> print type(res.local_admittive_clustering())
        <type 'numpy.ndarray'>
        """

        # needed vals: admittance matrix and admittiv_degree
        # are already complex/real as needed
        admittance = self.get_admittance()
        ad = self.admittive_degree()

        # output and the degree have to be switched
        if self.flagComplex:
            d = np.array(self.degree(), dtype=complex)
            ac = np.zeros(self.N, dtype=complex)

        else:
            d = self.degree()
            ac = np.zeros(self.N)

        # TODO: Sweave me!
        for i in xrange(self.N):
            dummy = 0
            for j in xrange(self.N):
                for k in xrange(self.N):
                    dummy += admittance[i][j]*admittance[i][k]*admittance[j][k]
            if d[i] == 1:
                ac[i] = 0
            else:
                ac[i] = dummy/(ad[i] * (d[i]-1))

        # return
        return ac

    def global_admittive_clustering(self):
        """
        Return node wise admittive clustering coefficient.

        :rtype: NumPy float

        **Examples:**

        >>> res =  ResNetwork.SmallTestNetwork()
        >>> print "%.3f" % res.global_admittive_clustering()
        0.016
        >>> print type(res.global_admittive_clustering())
        <type 'numpy.float64'>
        """

        return self.local_admittive_clustering().mean()

    def effective_resistance(self, a, b):
        """
        Return the effective resistance (ER) between two nodes
        a and b. The ER is the electrical analogue to the shortest path
        where a is considered as "source" and b as the "sink"

        :type a: int
        :arg a: index of the "source" node

        :type b: int
        :arg b: index of the "sink" node

        :rtype: NumPy float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print res.effective_resistance(1,1)
        0.0
        >>> print type( res.effective_resistance(1,1) )
        <type 'float'>
        >>> print "%.3f" % res.effective_resistance(1,2)
        4.444
        >>> print type( res.effective_resistance(1,1) )
        <type 'float'>

        """
        # return def for self-loop
        if a == b:
            if self.flagComplex:
                return complex(0.0)
            else:
                return float(0.0)

        # Get pseudoinverse of the Laplacian
        R = self.get_R()

        # return looked-up values
        return R[a, a] - R[a, b] - R[b, a] + R[b, b]

    def average_effective_resistance(self):
        """
        Return the average effective resistance (<ER>) of the resistive
        network, the average resistances for all "paths" (connections)

        :rtype: float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print "%.5f" % res.average_effective_resistance()
        7.28889
        >>> print type( res.average_effective_resistance() )
        <type 'numpy.float64'>

        """

        # since the NW is symmetric, we can only
        # sum over upper triangle, excluding zeros
        # but multiply by 2 later on

        # we also store a hidden, quick access var
        self._effective_resistances = np.array([])

        for i in xrange(self.N):
            for j in xrange(i):
                self._effective_resistances = np.append(
                    self._effective_resistances,
                    self.effective_resistance(i, j))

        return 2*np.sum(self._effective_resistances) / (self.N*(self.N-1))

    def diameter_effective_resistance(self):
        """
        Return the diameter (the highest resistance path between any nodes).

        :rtype: float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print "%.3f" % res.diameter_effective_resistance()
        Re-computing all effective resistances
        14.444
        >>> print type( res.diameter_effective_resistance() )
        <type 'numpy.float64'>

        >>> res = ResNetwork.SmallTestNetwork()
        >>> x = res.average_effective_resistance()
        >>> print "%.3f" % res.diameter_effective_resistance()
        14.444

        """

        # try to use pre-computed values
        try:
            diameter = np.max(self._effective_resistances)

        except AttributeError:
            print "Re-computing all effective resistances"
            self.average_effective_resistance()
            diameter = np.max(self._effective_resistances)

        return diameter

    def effective_resistance_closeness_centrality(self, a):

        """
        The effective resistance closeness centrality (ERCC) of node a

        :type a: int
        :arg a: index of the "source" node

        :rtype: NumPy float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print "%.3f" % res.effective_resistance_closeness_centrality(0)
        0.154
        >>> print "%.3f" % res.effective_resistance_closeness_centrality(4)
        0.080
        """

        # alloc
        ERCC = np.float(0.0)

        # compute
        for i in range(self.N):
            ERCC += self.effective_resistance(a, i)
        # ERCC /=  np.square( self.N - 1 )
        ERCC = (self.N - 1) / ERCC

        # return
        return ERCC

    def vertex_current_flow_betweenness(self, i):
        r"""
        Vertex Current Flow Betweeness (VCFB) of a node i.

        The electrial version of Newmann's node betweeness is here
        defined as the Vertex Current Flow Betweeness (VCGB) of a node

        .. math::
            VCFB_i := \frac{ 2 }{ n \left( n-1 \right)} \sum_{s<t} I_i^{st}

        where

        .. math::
            I_i^{st} &= \frac{1}{2}\sum_{j} \Gamma_{i,j} | V_i - V_j |\\
                     &= \frac{1}{2}\sum_{j} \Gamma_{i,j}
                     | I_s(R_{i,s}-R_{j,s}) + I_t(R_{j,t}-R_{i,t}) |

        and further:
            - :math:`I_{s}^{st} := I_{s}`
            - :math:`I_{t}^{st}  := I_{t}`
            - :math:`\Gamma` is the admittance matrix
            - :math:`R` is the pseudoinverse of the admittance Laplacian

        :arg int a: index of the "source" node
        :rtype: NumPy float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print "%.3f" % res.vertex_current_flow_betweenness(1)
        0.389
        >>> print "%.3f" % res.vertex_current_flow_betweenness(2)
        0.044
        """
        # switch the implementation according to weave support
        if self.flagWeave and not self.flagComplex:
            return self._vertex_current_flow_betweenness_weave(i)
        else:
            return self._vertex_current_flow_betweenness_python(i)

    def edge_current_flow_betweenness(self):
        """The electrial version of Newmann's edge betweeness

        :rtype: NumPy float

        **Examples:**

        >>> res = ResNetwork.SmallTestNetwork()
        >>> print res.edge_current_flow_betweenness()
        [[ 0.          0.4         0.          0.          0.        ]
         [ 0.4         0.          0.24444444  0.53333333  0.        ]
         [ 0.          0.24444444  0.          0.24444444  0.        ]
         [ 0.          0.53333333  0.24444444  0.          0.4       ]
         [ 0.          0.          0.          0.4         0.        ]]
        >>> #update to unit resistances
        >>> res.update_resistances(res.adjacency)
        >>> print res.edge_current_flow_betweenness()
        [[ 0.          0.4         0.          0.          0.        ]
         [ 0.4         0.          0.33333333  0.4         0.        ]
         [ 0.          0.33333333  0.          0.33333333  0.        ]
         [ 0.          0.4         0.33333333  0.          0.4       ]
         [ 0.          0.          0.          0.4         0.        ]]
        """
        # switch the implementation according to weave support
        if self.flagWeave and not self.flagComplex:
            return self._edge_current_flow_betweenness_weave()
        else:
            return self._edge_current_flow_betweenness_python()

###############################################################################
# ##                       PRIVATE FUNCTIONS                               ## #
###############################################################################
    def _vertex_current_flow_betweenness_python(self, i):
        """Python version of VCFB
        """
        # get required matrices
        admittance = self.get_admittance()
        R = self.get_R()

        # set params
        Is = It = np.float(1.0)

        # alloc output
        VCFB = np.float(0)

        for t in xrange(self.N):
            for s in xrange(t):
                I = 0.0
                if i == t or i == s:
                    pass
                else:
                    for j in xrange(self.N):
                        I += admittance[i][j] * np.abs(
                            Is*(R[i][s]-R[j][s]) + It*(R[j][t]-R[i][t]))/2.
                VCFB += 2.*I/(self.N*(self.N-1))

        return VCFB

    def _vertex_current_flow_betweenness_weave(self, i):
        """C Version of VCFB
        """
        # get required matrices
        admittance = self.get_admittance()
        R = self.get_R()

        # set params
        Is = It = np.float(1.0)

        # alloc output
        VCFB = np.float(0.0)
        N = np.int(self.N)

        code = """
            int t=0;
            int s=0;
            int j=0;
            double I=0;
            N = double(N);

            for(t=0;t<N;t++){
                for(s=0; s<t; s++){
                    I = 0.0;
                    if(i == t || i == s){
                        continue;
                    }
                    else{
                        for(j=0;j<N;j++){
                            I += ADMITTANCE2(i,j)*\
                            fabs( Is*(R2(i,s)-R2(j,s))+\
                                  It*(R2(j,t)-R2(i,t)) \
                                ) / 2.0;
                        } // for  j
                    }
                    VCFB += 2.0*I/(N*(N-1));
                } // for s
            } // for t

            return_val = VCFB;
        """
        VCFB = weave.inline(code,
                            ['N', 'Is', 'It', 'admittance', 'R', 'i', 'VCFB'],
                            compiler="gcc", headers=["<math.h>"])
        return VCFB

    def _edge_current_flow_betweenness_python(self):
        """
        Python version of ECFB
        """
        # set currents
        Is = It = np.float(1)

        # alloc output
        if self.flagComplex:
            dtype = complex
        else:
            dtype = float

        ECFB = np.zeros([self.N, self.N], dtype=dtype)

        # the usual
        admittance = self.get_admittance()
        R = self.get_R()

        for i in xrange(self.N):
            for j in xrange(self.N):
                I = 0
                for t in xrange(self.N):
                    for s in xrange(t):
                        I += admittance[i][j] * np.abs(
                            Is*(R[i][s]-R[j][s])+It*(R[j][t]-R[i][t]))

                # Lets try to compute the in
                ECFB[i][j] = 2*I/(self.N*(self.N-1))

        return ECFB

    def _edge_current_flow_betweenness_weave(self):
        """
        Weave/C version of ECFB
        """
        # set currents
        Is = It = np.float(1)

        # alloc output
        ECFB = np.zeros([self.N, self.N])

        # the usual
        admittance = self.get_admittance()
        R = self.get_R()

        N = np.int(self.N)
        code = """
            int i=0;
            int j=0;
            int t=0;
            int s=0;
            double I = 0.0;

            N = double(N);
            for(i=0; i<N; i++){
                for(j=0;j<N;j++){
                    I = 0.0;
                    for(t=0;t<N;t++){
                        for(s=0; s<t; s++){
                            I += ADMITTANCE2(i,j)*\
                                 fabs(  Is*(R2(i,s)-R2(j,s))+\
                                        It*(R2(j,t)-R2(i,t)) \
                                    );
                        } //for s
                    } // for t
                    ECFB2(i,j) += 2.*I/(N*(N-1));
                } // for j
            } // for i

            return_val = ECFB;
        """
        weave.inline(code, ['N', 'Is', 'It', 'admittance', 'R', 'ECFB'],
                     compiler="gcc", headers=["<math.h>"])

        return ECFB

###############################################################################
# ##                       FUNCTIONS ATTIC                                 ## #
###############################################################################

    # These functions are no longer needed as the computation can be broken
    # down to some indexing but they are kept as implementatin references
    # since all git logs will be lost when adding resnw to pyunicorn.
    #
    #
    # def _effective_resistance_python(self,a,b):
    #     """Python version of the effective resistance
    #     """
    #     # Get pseudoinverse of the Laplacian
    #     R = self.get_R()

    #     t = R[a,a] - R[a,b] - R[b,a] + R[b,b]
    #     print "the t is %f + %fi " % (t.real,t.imag)
    #     # construct a vector that is all zero except for
    #     # the source (a) and the sink (b)
    #     # if self.flagComplex:
    #     #     base = np.zeros(self.N,dtype=complex)
    #     # else:
    #     base = np.zeros(self.N)

    #     base[a] = 1;
    #     base[b] = -1;

    #     # multiply every row of R with the above vector
    #     # and sum across rows => np.sum(R * base, axis=1) and then
    #     # build the scalar product of the result and the base vector
    #     ER = np.dot( np.sum(R * base, axis=1), base )

    #     return ER

    # def _effective_resistance_weave(self,a,b):
    #     """ C version of effective resistance
    #     """
    #     # Get pseudoinverse of the Laplacian
    #     R = self.get_R()
    #     N = np.float( self.N)

    #     code = \
    #     """
    #     int i=0;
    #     int j=0;
    #     double ER = 0.0;

    #     // vector for temp values
    #     double tmp[int(N)];

    #     // setup the base vector
    #     double base[int(N)];

    #     base[a] = 1;
    #     base[b] = -1;

    #     for(i=0;i<N; i++){
    #         for (j=0; j<N; j++){
    #             tmp[i] += R2(i,j) * base[j];
    #         } //for j
    #         ER += base[i] * tmp[i];
    #     } // for i

    #     return_val = ER;
    #     """
    #     variables = ['a','b','N','R'];
    #     ER = weave.inline(code,variables,compiler = "gcc")
    #     return ER

    # def _effective_resistance_weave_complex(self,a,b):
    #     """ C version of effective resistance
    #     """
    #     # Get pseudoinverse of the Laplacian
    #     R = self.get_R()
    #     N = np.float( self.N)

    #     code = \
    #     """
    #     int i=0;
    #     int j=0;
    #     std::complex<double> ER = 0.0;

    #     // vector for base and values
    #     std::complex<double> base[int(N)];
    #     base[a].real() = 1;
    #     base[b].real() = -1;

    #     // vector for temp values
    #     std::complex<double> tmp[int(N)];

    #     /*
    #     tmp = (double*) calloc(N,sizeof(double));

    #    /// handling of complex numbers via struct definition
    #     // struct describing complex no.
    #      typedef struct {
    #         double real;
    #         double imag;
    #     } complex_def;

    #     complex_def *base;

    #     base = calloc(N * sizeof(*array));

    #     */

    #     for(i=0;i<N; i++){
    #         for (j=0; j<N; j++){
    #             tmp[i] += R2(i,j) * base[j];
    #         } //for j
    #         ER += base[i] * tmp[i];
    #     } // for i

    #     return_val = ER;
    #     """
    #     variables = ['a','b','N','R'];
    #     ER = weave.inline(code,variables,compiler = "gcc",
    #         headers=["<complex.h>"])
    #     return ER
