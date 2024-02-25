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

import numpy as np

from ..core.cache import Cached
from .recurrence_plot import RecurrencePlot
from ..core._ext.types import to_cy, DFIELD
from ._ext.numerics import _manhattan_distance_matrix_crp, \
    _euclidean_distance_matrix_crp, _supremum_distance_matrix_crp


class CrossRecurrencePlot(RecurrencePlot):
    """
    Class CrossRecurrencePlot for generating and quantitatively analyzing
    :index:`cross recurrence plots <single: cross recurrence plot>`.

    The CrossRecurrencePlot class supports the construction of cross recurrence
    plots from two multi-dimensional time series, optionally using embedding.
    Currently, manhattan, euclidean and supremum norms are provided for
    measuring distances in phase space.

    Methods for calculating commonly used measures of :index:`recurrence
    quantification analysis <pair: RQA; cross recurrence plot>` (RQA) are
    provided, e.g., determinism, maximum diagonal line length and laminarity.
    The definitions of these measures together with a review of the theory and
    applications of cross recurrence plots are given in [Marwan2007]_.

    **Examples:**

     - Create an instance of CrossRecurrencePlot from time series x and y with
       a fixed recurrence threshold and without embedding::

           CrossRecurrencePlot(x, y, threshold=0.1)

     - Create an instance of CrossRecurrencePlot at a fixed recurrence rate and
       using time delay embedding::

           CrossRecurrencePlot(x, y, dim=3, tau=2,
                               recurrence_rate=0.05).recurrence_rate()
    """

    #
    #  Internal methods
    #

    def __init__(self, x, y, metric="supremum", normalize=False,
                 sparse_rqa=False, silence_level=0, **kwds):
        """
        Initialize an instance of CrossRecurrencePlot.

        .. note::
           For a cross recurrence plot, time series x and y generally do
           **not** need to have the same length!

        Either recurrence threshold ``threshold`` or recurrence rate
        ``recurrence_rate`` have to be given as keyword arguments.

        Embedding is only supported for scalar time series. If embedding
        dimension ``dim`` and delay ``tau`` are **both** given as keyword
        arguments, embedding is applied. Multidimensional time series are
        processed as is by default. The same delay embedding is applied to both
        time series x and y.

        :type x: 2D array (time, dimension)
        :arg x: One of the time series to be analyzed, can be scalar or
            multi-dimensional.
        :type y: 2D array (time, dimension)
        :arg y: One of the time series to be analyzed, can be scalar or
            multi-dimensional.
        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :arg bool normalize: Decide whether to normalize both time series to
            zero mean and unit standard deviation.
        :arg number silence_level: Inverse level of verbosity of the object.
        :arg number threshold: The recurrence threshold keyword for generating
            the cross recurrence plot using a fixed threshold.
        :arg number recurrence_rate: The recurrence rate keyword for generating
            the cross recurrence plot using a fixed recurrence rate.
        :arg number dim: The embedding dimension.
        :arg number tau: The embedding delay.
        """
        threshold = kwds.get("threshold")
        recurrence_rate = kwds.get("recurrence_rate")

        RecurrencePlot.__init__(
            self, np.empty((2, 0)), metric=metric, normalize=normalize,
            sparse_rqa=sparse_rqa, silence_level=silence_level,
            skip_recurrence=True)

        self.CR = None
        """The cross recurrence matrix."""
        self.N = 0
        """The length of the embedded time series x."""
        self.M = 0
        """The length of the embedded time series y."""

        #  Store time series
        self.x = x.copy()
        """The time series x."""
        self.y = y.copy()
        """The time series y."""

        #  Reshape time series
        self.x.shape = (self.x.shape[0], -1)
        self.y.shape = (self.y.shape[0], -1)

        #  Normalize time series
        if normalize:
            self.normalize_time_series(self.x)
            self.normalize_time_series(self.y)

        #  Get embedding dimension and delay from **kwds
        dim = kwds.get("dim")
        tau = kwds.get("tau")

        self._mut_embedding: int = 0
        if (dim is not None) and (tau is not None):
            #  Embed the time series
            self.x_embedded = self.embed_time_series(self.x, dim, tau)
            self.y_embedded = self.embed_time_series(self.y, dim, tau)
        else:
            self.x_embedded = self.x
            self.y_embedded = self.y

        #  construct recurrence plot accordingly to threshold / recurrence rate
        if threshold is not None:
            #  Calculate the recurrence matrix R using the radius of
            #  neighborhood threshold
            CrossRecurrencePlot.set_fixed_threshold(self, threshold)
        elif recurrence_rate is not None:
            #  Calculate the recurrence matrix R using a fixed recurrence rate
            CrossRecurrencePlot.\
                set_fixed_recurrence_rate(self, recurrence_rate)
        else:
            raise NameError("Please give either threshold or recurrence_rate \
                            to construct the cross recurrence plot!")

    def __str__(self):
        """
        Returns a string representation.
        """
        return ("CrossRecurrencePlot: "
                f"time series shapes {self.x.shape}, {self.y.shape}.\n"
                f"Embedding dimension {self.dim if self.dim else 0}\n"
                f"Threshold {self.threshold}, {self.metric} metric")

    @property
    def x_embedded(self) -> np.ndarray:
        """
        The embedded time series x.
        """
        return self._x_embedded

    @x_embedded.setter
    def x_embedded(self, embedding: np.ndarray):
        self._x_embedded = to_cy(embedding, DFIELD)
        self._mut_embedding += 1

    @property
    def y_embedded(self) -> np.ndarray:
        """
        The embedded time series y.
        """
        return self._y_embedded

    @y_embedded.setter
    def y_embedded(self, embedding: np.ndarray):
        self._y_embedded = to_cy(embedding, DFIELD)
        self._mut_embedding += 1

    #
    #  Service methods
    #

    def recurrence_matrix(self):
        """
        Return the current cross recurrence matrix :math:`CR`.

        :rtype: 2D square Numpy array
        :return: the current cross recurrence matrix :math:`CR`.
        """
        return self.CR

    def distance_matrix(self, metric):
        """
        Return phase space cross distance matrix :math:`D` according to the
        chosen metric.

        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :rtype: 2D square array
        :return: the phase space cross distance matrix :math:`D`
        """
        assert metric in self._known_metrics, f"unknown metric: {metric}"
        return getattr(self, f"{metric}_distance_matrix")()

    #
    #  Calculate recurrence plot
    #

    @Cached.method(name="the manhattan distance matrix")
    def manhattan_distance_matrix(self):
        """
        Return the manhattan distance matrix from two (embedded) time series.

        :type x_embedded: 2D Numpy array (time, embedding dimension)
        :arg x_embedded: The phase space trajectory x.
        :type y_embedded: 2D Numpy array (time, embedding dimension)
        :arg y_embedded: The phase space trajectory y.
        :rtype: 2D rectangular Numpy array
        :return: the manhattan distance matrix.
        """
        ntime_x = self.x_embedded.shape[0]
        ntime_y = self.y_embedded.shape[0]
        dim = self.x_embedded.shape[1]
        return _manhattan_distance_matrix_crp(ntime_x, ntime_y, dim,
                                              self.x_embedded, self.y_embedded)

    @Cached.method(name="the euclidean distance matrix")
    def euclidean_distance_matrix(self):
        """
        Return the euclidean distance matrix from two (embedded) time series.

        :rtype: 2D rectangular Numpy array
        :return: the euclidean distance matrix.
        """
        ntime_x = self.x_embedded.shape[0]
        ntime_y = self.y_embedded.shape[0]
        dim = self.x_embedded.shape[1]
        return _euclidean_distance_matrix_crp(ntime_x, ntime_y, dim,
                                              self.x_embedded, self.y_embedded)

    @Cached.method(name="the supremum distance matrix")
    def supremum_distance_matrix(self):
        """
        Return the supremum distance matrix from two (embedded) time series.

        :rtype: 2D rectangular Numpy array
        :return: the supremum distance matrix.
        """
        ntime_x = self.x_embedded.shape[0]
        ntime_y = self.y_embedded.shape[0]
        dim = self.x_embedded.shape[1]
        return _supremum_distance_matrix_crp(ntime_x, ntime_y, dim,
                                             self.x_embedded, self.y_embedded)

    def set_fixed_threshold(self, threshold):
        """
        Set the cross recurrence plot to a fixed threshold.

        Modifies / sets the class variables :attr:`CR`, :attr:`N` and :attr:`M`
        accordingly.

        :arg number threshold: The recurrence threshold.
        """
        if self.silence_level <= 1:
            print("Calculating cross recurrence plot at fixed threshold...")

        distance = self.distance_matrix(self.metric)
        (N, M) = distance.shape
        recurrence = np.zeros((N, M), dtype="int8")
        recurrence[distance < threshold] = 1
        self.CR = recurrence
        self.N = N
        self.M = M

    def set_fixed_recurrence_rate(self, recurrence_rate):
        """
        Set the cross recurrence plot to a fixed recurrence rate.

        Modifies / sets the class variables :attr:`CR`, :attr:`N` and :attr:`M`
        accordingly.

        :arg number recurrence_rate: The recurrence rate.
        """
        if self.silence_level <= 1:
            print("Calculating cross recurrence plot at "
                  "fixed recurrence rate...")

        distance = self.distance_matrix(self.metric)
        (N, M) = distance.shape
        threshold = self.threshold_from_recurrence_rate(distance,
                                                        recurrence_rate)
        recurrence = np.zeros((N, M), dtype="int8")
        recurrence[distance < threshold] = 1
        self.CR = recurrence
        self.N = N
        self.M = M

    #
    #  Extended RQA measures
    #

    def recurrence_rate(self):
        """
        Return cross recurrence rate.

        Alias to :meth:`cross_recurrence_rate`, since
        :meth:`RecurrencePlot.recurrence_rate` would give incorrect results
        here.

        :rtype: number (float)
        :return: the cross recurrence rate.
        """
        return self.cross_recurrence_rate()

    def cross_recurrence_rate(self):
        """
        Return cross recurrence rate.

        :rtype: number (float)
        :return: the cross recurrence rate.
        """
        return float(self.CR.sum()) / (self.N * self.M)

    def balance(self):
        """
        Return balance of the cross recurrence plot.

        Might be useful for detecting the direction of coupling between systems
        using cross recurrence analysis.
        """
        #  Get cross recurrence matrix
        CR = self.recurrence_matrix()

        #  Get sum of upper triangle of cross recurrence matrix, excluding the
        #  main diagonal
        upper_triangle_sum = np.triu(CR, k=1).sum()

        #  Get sum of lower triangle of cross recurrence matrix, excluding the
        #  main diagonal
        lower_triangle_sum = np.tril(CR, k=-1).sum()

        #  Return balance
        return (upper_triangle_sum - lower_triangle_sum) / \
            float(upper_triangle_sum + lower_triangle_sum)

    def diagline_dist(self):
        """Not implemented yet"""
        raise NotImplementedError(
            "Line distributions are not yet "
            "available for cross-recurrence plots")

    def vertline_dist(self):
        """Not implemented yet"""
        raise NotImplementedError(
            "Line distributions are not yet "
            "available for cross-recurrence plots")

    def white_vertline_dist(self):
        """Not implemented yet"""
        raise NotImplementedError(
            "Line distributions are not yet "
            "available for cross-recurrence plots")
