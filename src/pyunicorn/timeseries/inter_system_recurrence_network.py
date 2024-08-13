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
Provides classes for the analysis of dynamical systems and time series based
on recurrence plots, including measures of recurrence quantification
analysis (RQA) and recurrence network analysis.
"""

from typing import Tuple
from collections.abc import Hashable

# array object and fast numerics
import numpy as np

from ..core import InteractingNetworks
from .recurrence_plot import RecurrencePlot
from .cross_recurrence_plot import CrossRecurrencePlot


#
#  Class definitions
#

class InterSystemRecurrenceNetwork(InteractingNetworks):

    """
    Generating and quantitatively analyzing inter-system recurrence networks.

    For a inter-system recurrence network, time series x and y do not need to
    have the same length! Formally, nodes are identified with state vectors in
    the common phase space of both time series. Hence, the time series need to
    have the same number of dimensions and identical physical units.
    Undirected links are added to describe recurrences within x and y as well
    as cross-recurrences between x and y. Self-loops are excluded in this
    undirected network representation.

    More information on the theory and applications of inter system recurrence
    networks can be found in [Feldhoff2012]_.

    **Examples:**

     - Create an instance of InterSystemRecurrenceNetwork with fixed
       recurrence thresholds and without embedding::

           InterSystemRecurrenceNetwork(x, y, threshold=(0.1, 0.2, 0.1))

     - Create an instance of InterSystemRecurrenceNetwork at a fixed
       recurrence rate and using time delay embedding::

           InterSystemRecurrenceNetwork(
               x, y, dim=3, tau=(2, 1), recurrence_rate=(0.05, 0.05, 0.02))
    """

    #
    #  Internal methods
    #

    def __init__(self, x, y, metric="supremum",
                 normalize=False, silence_level=0, **kwds):
        """
        Initialize an instance of InterSystemRecurrenceNetwork (ISRN).

        .. note::
           For an inter system recurrence network, time series x and y need to
           have the same number of dimensions!

        Creates an embedding of the given time series x and y, calculates a
        inter system recurrence matrix from the embedding and then creates
        an InteractingNetwork object from this matrix, interpreting the inter
        system recurrence matrix as the adjacency matrix of an undirected
        complex network.

        Either recurrence thresholds ``threshold`` or
        recurrence rates ``recurrence_rate`` have to be given as keyword
        arguments.

        Embedding is only supported for scalar time series. If embedding
        dimension ``dim`` and delay ``tau`` are **both** given as keyword
        arguments, embedding is applied. Multidimensional time series are
        processed as is by default.

        :type x: 2D Numpy array (time, dimension)
        :arg x: The time series x to be analyzed, can be scalar or
                multi-dimensional.
        :type y: 2D Numpy array (time, dimension)
        :arg y: The time series y to be analyzed, can be scalar or
                multi-dimensional.
        :type metric: tuple of string
        :arg metric: The metric for measuring distances in phase space
                     ("manhattan", "euclidean", "supremum").
        :arg bool normalize: Decide whether to normalize the time series to
                             zero mean and unit standard deviation.
        :arg int silence_level: The inverse level of verbosity of the object.
        :arg kwds: Additional options.
        :type threshold: tuple of number (three numbers)
        :keyword threshold: The recurrence threshold keyword for generating
                            the recurrence plot using fixed thresholds. Give
                            for each time series and the cross recurrence plot
                            separately.
        :type recurrence_rate: tuple of number (three numbers)
        :keyword recurrence_rate: The recurrence rate keyword for generating
                                  the recurrence plot using a fixed recurrence
                                  rate. Give separately for each time series.
        :keyword int dim: The embedding dimension. Must be the same for both
                          time series.
        :type tau: tuple of int
        :keyword tau: The embedding delay. Give separately for each time
                      series.
        """
        #  Store time series
        self.x = x.copy().astype("float32")
        """The time series x."""
        self.y = y.copy().astype("float32")
        """The time series y."""

        #  Reshape time series
        self.x.shape = (self.x.shape[0], -1)
        self.y.shape = (self.y.shape[0], -1)

        #  Get embedding dimension and delay from **kwds
        dim = kwds.get("dim")
        tau = kwds.get("tau")

        #  Check for consistency
        if self.x.shape[1] == self.y.shape[1]:
            #  Set silence_level
            self.silence_level = silence_level
            """The inverse level of verbosity of the object."""

            #  Get number of nodes in subnetwork x
            self.N_x = self.x.shape[0]
            """Number of nodes in subnetwork x."""

            #  Get number of nodes in subnetwork y
            self.N_y = self.y.shape[0]
            """Number of nodes in subnetwork y."""

            #  Get total number of nodes of ISRN
            self.N = self.N_x + self.N_y
            """Total number of nodes of ISRN."""

            #  Store type of metric
            self.metric = metric
            """The metric used for measuring distances in phase space."""

            #  Normalize time series
            if normalize:
                RecurrencePlot.normalize_time_series(self.x)
                RecurrencePlot.normalize_time_series(self.y)

            #  Embed time series if required
            self.dim = dim
            if dim is not None and tau is not None and self.x.shape[1] == 1:
                self.x_embedded = \
                    RecurrencePlot.embed_time_series(self.x, dim, tau[0])
                """The embedded time series x."""
                self.y_embedded = \
                    RecurrencePlot.embed_time_series(self.y, dim, tau[1])
                """The embedded time series y."""
            else:
                self.x_embedded = self.x
                self.y_embedded = self.y

            #  Get threshold or recurrence rate from **kwds, construct
            #  ISRN accordingly
            threshold = kwds.get("threshold")
            recurrence_rate = kwds.get("recurrence_rate")
            self.threshold = threshold

            if threshold is not None:
                #  Calculate the ISRN using the radius of neighborhood
                #  threshold
                ISRM = self.set_fixed_threshold(threshold)
            elif recurrence_rate is not None:
                #  Calculate the ISRN using a fixed recurrence rate
                ISRM = self.set_fixed_recurrence_rate(recurrence_rate)
            else:
                raise NameError("Please give either threshold or \
                                recurrence_rate to construct the joint \
                                recurrence plot!")

            InteractingNetworks.__init__(self, adjacency=ISRM, directed=False,
                                         silence_level=self.silence_level)
            #  No treatment of missing values yet!
            self.missing_values = False

        else:
            raise ValueError("Both time series x and y need to have the same \
                             dimension!")

    def __str__(self):
        """
        Returns a string representation.
        """
        return ("InterSystemRecurrenceNetwork: "
                f"time series shapes {self.x.shape}, {self.y.shape}.\n"
                f"Embedding dimension {self.dim if self.dim else 0}\n"
                f"Threshold {self.threshold}, {self.metric} metric.\n"
                f"{InteractingNetworks.__str__(self)}")

    #
    #  Service methods
    #

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return (self.rp_x, self.rp_x, self.crp_xy,)

    #
    #  Methods to handle inter system recurrence networks
    #

    def inter_system_recurrence_matrix(self):
        """
        Return the current inter system recurrence matrix :math:`ISRM`.

        :rtype: 2D square Numpy array
        :return: the current inter system recurrence matrix :math:`ISRM`.
        """
        #  Shortcuts
        N = self.N
        N_x = self.N_x

        #  Init
        ISRM = np.zeros((N, N))

        #  Combine to inter system recurrence matrix
        ISRM[:N_x, :N_x] = self.rp_x.recurrence_matrix()
        ISRM[:N_x, N_x:N] = self.crp_xy.recurrence_matrix()
        ISRM[N_x:N, :N_x] = self.crp_xy.recurrence_matrix().transpose()
        ISRM[N_x:N, N_x:N] = self.rp_y.recurrence_matrix()

        return ISRM

    def set_fixed_threshold(self, threshold):
        """
        Create a inter system recurrence network at fixed thresholds.

        :type threshold: tuple of number (three numbers)
        :arg threshold: The three threshold parameters. Give for each
                        time series and the cross recurrence plot separately.
        """
        #  Compute recurrence matrices of x and y
        self.rp_x = RecurrencePlot(time_series=self.x_embedded,
                                   threshold=threshold[0],
                                   metric=self.metric,
                                   silence_level=self.silence_level)
        self.rp_y = RecurrencePlot(time_series=self.y_embedded,
                                   threshold=threshold[1],
                                   metric=self.metric,
                                   silence_level=self.silence_level)

        #  Compute cross-recurrence matrix of x and y
        self.crp_xy = CrossRecurrencePlot(x=self.x_embedded, y=self.y_embedded,
                                          threshold=threshold[2],
                                          metric=self.metric,
                                          silence_level=self.silence_level)

        #  Get combined ISRM
        ISRM = self.inter_system_recurrence_matrix()

        #  Set diagonal of ISRM to zero to avoid self-loops
        ISRM.flat[::self.N + 1] = 0
        return ISRM

    def set_fixed_recurrence_rate(self, density):
        """
        Create a inter system recurrence network at fixed link densities (
        recurrence rates).

        :type density: tuple of number (three numbers)
        :arg density: The three recurrence rate parameters. Give for each
                        time series and the cross recurrence plot separately.
        """
        #  Compute recurrence matrices of x and y
        self.rp_x = RecurrencePlot(time_series=self.x_embedded,
                                   recurrence_rate=density[0],
                                   metric=self.metric,
                                   silence_level=self.silence_level)
        self.rp_y = RecurrencePlot(time_series=self.y_embedded,
                                   recurrence_rate=density[1],
                                   metric=self.metric,
                                   silence_level=self.silence_level)

        #  Compute cross-recurrence matrix of x and y
        self.crp_xy = CrossRecurrencePlot(x=self.x_embedded, y=self.y_embedded,
                                          recurrence_rate=density[2],
                                          metric=self.metric,
                                          silence_level=self.silence_level)

        #  Get combined ISRM
        ISRM = self.inter_system_recurrence_matrix()

        #  Set diagonal of ISRM to zero to avoid self-loops
        ISRM.flat[::self.N + 1] = 0
        return ISRM

    #
    #  Methods to quantify inter system recurrence networks
    #

    def internal_recurrence_rates(self):
        """
        Return internal recurrence rates of subnetworks x and y.

        :rtype: tuple of number (float)
        :return: the internal recurrence rates of subnetworks x and y.
        """
        return (self.rp_x.recurrence_rate(),
                self.rp_y.recurrence_rate())

    def cross_recurrence_rate(self):
        """
        Return cross recurrence rate between subnetworks x and y.

        :rtype: number (float)
        :return: the cross recurrence rate between subnetworks x and y.
        """
        return self.crp_xy.cross_recurrence_rate()

    def cross_global_clustering_xy(self):
        """
        Return cross global clustering of x with respect to y.

        See [Feldhoff2012]_ for definition, further explanation and
        applications.

        :rtype: number (float)
        :return: the cross global clustering of x with respect to y.
        """
        return self.cross_global_clustering(np.arange(self.N_x),
                                            np.arange(self.N_x, self.N))

    def cross_global_clustering_yx(self):
        """
        Return cross global clustering of y with respect to x.

        See [Feldhoff2012]_ for definition, further explanation and
        applications.

        :rtype: number (float)
        :return: the cross global clustering of y with respect to x.
        """
        return self.cross_global_clustering(np.arange(self.N_x, self.N),
                                            np.arange(self.N_x))

    def cross_transitivity_xy(self):
        """
        Return cross transitivity of x with respect to y.

        See [Feldhoff2012]_ for definition, further explanation and
        applications.

        :rtype: number (float)
        :return: the cross transitivity of x with respect to y.
        """
        return self.cross_transitivity(np.arange(self.N_x),
                                       np.arange(self.N_x, self.N))

    def cross_transitivity_yx(self):
        """
        Return cross transitivity of y with respect to x.

        See [Feldhoff2012]_ for definition, further explanation and
        applications.

        :rtype: number (float)
        :return: the cross transitivity of y with respect to x.
        """
        return self.cross_transitivity(np.arange(self.N_x, self.N),
                                       np.arange(self.N_x))
