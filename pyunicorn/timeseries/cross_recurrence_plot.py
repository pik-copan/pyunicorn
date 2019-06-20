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
Provides classes for the analysis of dynamical systems and time series based
on recurrence plots, including measures of recurrence quantification
analysis (RQA) and recurrence network analysis.
"""

# array object and fast numerics
import numpy as np

from ._ext.numerics import _manhattan_distance_matrix_crp, \
    _euclidean_distance_matrix_crp, _supremum_distance_matrix_crp

from .recurrence_plot import RecurrencePlot

#
#  Class definitions
#


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
            sparse_rqa=sparse_rqa, threshold=threshold,
            recurrence_rate=recurrence_rate, silence_level=silence_level)

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

        if dim is not None and tau is not None:
            #  Embed the time series
            self.x_embedded = self.embed_time_series(self.x, dim, tau)
            """The embedded time series x."""
            self.y_embedded = self.embed_time_series(self.y, dim, tau)
            """The embedded time series y."""
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
        return ('CrossRecurrencePlot: time series shapes %s, %s.\n'
                'Embedding dimension %i\nThreshold %s, %s metric') % (
                    self.x.shape, self.y.shape, self.dim if self.dim else 0,
                    self.threshold, self.metric)

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

    def distance_matrix(self, x_embedded, y_embedded, metric):
        """
        Return phase space cross distance matrix :math:`D` according to the
        chosen metric.

        :type x_embedded: 2D array (time, embedding dimension)
        :arg x_embedded: The phase space trajectory x.
        :type y_embedded: 2D array (time, embedding dimension)
        :arg y_embedded: The phase space trajectory y.
        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :rtype: 2D square array
        :return: the phase space cross distance matrix :math:`D`
        """
        #  Return distance matrix according to chosen metric:
        if metric == "manhattan":
            return self.manhattan_distance_matrix(x_embedded, y_embedded)
        elif metric == "euclidean":
            return self.euclidean_distance_matrix(x_embedded, y_embedded)
        elif metric == "supremum":
            return self.supremum_distance_matrix(x_embedded, y_embedded)
        else:
            return None

    #
    #  Calculate recurrence plot
    #

    def manhattan_distance_matrix(self, x_embedded, y_embedded):
        """
        Return the manhattan distance matrix from two (embedded) time series.

        :type x_embedded: 2D Numpy array (time, embedding dimension)
        :arg x_embedded: The phase space trajectory x.
        :type y_embedded: 2D Numpy array (time, embedding dimension)
        :arg y_embedded: The phase space trajectory y.
        :rtype: 2D rectangular Numpy array ("float32")
        :return: the manhattan distance matrix.
        """
        if self.silence_level <= 1:
            print("Calculating the manhattan distance matrix...")

        ntime_x = x_embedded.shape[0]
        ntime_y = y_embedded.shape[0]
        dim = x_embedded.shape[1]
        x_embedded = x_embedded.astype('double').copy(order='c')
        y_embedded = y_embedded.astype('double').copy(order='c')
        return _manhattan_distance_matrix_crp(ntime_x, ntime_y, dim,
                                              x_embedded, y_embedded)

    def euclidean_distance_matrix(self, x_embedded, y_embedded):
        """
        Return the euclidean distance matrix from two (embedded) time series.

        :type x_embedded: 2D Numpy array (time, embedding dimension)
        :arg x_embedded: The phase space trajectory x.
        :type y_embedded: 2D Numpy array (time, embedding dimension)
        :arg y_embedded: The phase space trajectory y.
        :rtype: 2D rectangular Numpy array ("float32")
        :return: the euclidean distance matrix.
        """
        if self.silence_level <= 1:
            print("Calculating the euclidean distance matrix...")

        ntime_x = x_embedded.shape[0]
        ntime_y = y_embedded.shape[0]
        dim = x_embedded.shape[1]
        x_embedded = x_embedded.astype('double').copy(order='c')
        y_embedded = y_embedded.astype('double').copy(order='c')
        return _euclidean_distance_matrix_crp(ntime_x, ntime_y, dim,
                                              x_embedded, y_embedded)

    def supremum_distance_matrix(self, x_embedded, y_embedded):
        """
        Return the supremum distance matrix from two (embedded) time series.

        :type x_embedded: 2D Numpy array (time, embedding dimension)
        :arg x_embedded: The phase space trajectory x.
        :type y_embedded: 2D Numpy array (time, embedding dimension)
        :arg y_embedded: The phase space trajectory y.
        :rtype: 2D rectangular Numpy array ("float32")
        :return: the supremum distance matrix.
        """
        if self.silence_level <= 1:
            print("Calculating the supremum distance matrix...")

        ntime_x = x_embedded.shape[0]
        ntime_y = y_embedded.shape[0]
        dim = x_embedded.shape[1]
        x_embedded = x_embedded.astype('float32').copy(order='c')
        y_embedded = y_embedded.astype('float32').copy(order='c')
        return _supremum_distance_matrix_crp(ntime_x, ntime_y, dim, x_embedded,
                                             y_embedded)

    def set_fixed_threshold(self, threshold):
        """
        Set the cross recurrence plot to a fixed threshold.

        Modifies / sets the class variables :attr:`CR`, :attr:`N` and :attr:`M`
        accordingly.

        :arg number threshold: The recurrence threshold.
        """
        if self.silence_level <= 1:
            print("Calculating cross recurrence plot at fixed threshold...")

        #  Get distance matrix, according to self.metric
        distance = self.distance_matrix(self.x_embedded, self.y_embedded,
                                        self.metric)
        print(distance)
        #  Get length of time series x and y
        (N, M) = distance.shape

        #  Initialize recurrence matrix
        recurrence = np.zeros((N, M), dtype="int8")

        #  Thresholding the distance matrix
        recurrence[distance < threshold] = 1

        #  Clean up
        del distance

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

        #  Get distance matrix, according to self.metric
        distance = self.distance_matrix(self.x_embedded, self.y_embedded,
                                        self.metric)

        #  Get length of time series x and y
        (N, M) = distance.shape

        #  Get threshold to obtain fixed recurrence rate
        threshold = self.threshold_from_recurrence_rate(distance,
                                                        recurrence_rate)

        #  Initialize recurrence matrix
        recurrence = np.zeros((N, M), dtype="int8")

        #  Thresholding the distance matrix
        recurrence[distance < threshold] = 1

        #  Clean up
        del distance

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
