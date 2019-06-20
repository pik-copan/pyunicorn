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

#
#  Import essential packages
#

# array object and fast numerics
import numpy as np

from .recurrence_plot import RecurrencePlot


#
#  Class definitions
#

class JointRecurrencePlot(RecurrencePlot):

    """
    Class JointRecurrencePlot for generating and quantitatively analyzing
    :index:`joint recurrence plots <single: joint recurrence plot>`.

    The JointRecurrencePlot class supports the construction of joint recurrence
    plots from two multi-dimensional time series, optionally using embedding.
    Currently, manhattan, euclidean and supremum norms are provided for
    measuring distances in phase space.

    Methods for calculating commonly used measures of :index:`recurrence
    quantification analysis <pair: RQA; joint recurrence plot>` (RQA) are
    provided, e.g., determinism, maximum diagonal line length and laminarity.
    The definitions of these measures together with a review of the theory and
    applications of joint recurrence plots are given in [Marwan2007]_.

    **Examples:**

     - Create an instance of JointRecurrencePlot with a fixed recurrence
       threshold and without embedding::

           JointRecurrencePlot(x, y, threshold=(0.1,0.2))

     - Create an instance of JointRecurrencePlot with a fixed recurrence
       threshold in units of STD and without embedding::

           JointRecurrencePlot(x, y, threshold_std=(0.03,0.05))

     - Create an instance of JointRecurrencePlot at a fixed recurrence rate and
       using time delay embedding::

           JointRecurrencePlot(
               x, y, dim=(3,5), tau=(2,1),
               recurrence_rate=(0.05,0.04)).recurrence_rate()
    """

    #
    #  Internal methods
    #

    def __init__(self, x, y, metric=("supremum", "supremum"),
                 normalize=False, lag=0, silence_level=0, **kwds):
        """
        Initialize an instance of JointRecurrencePlot.

        .. note::
           For a joint recurrence plot, time series x and y need to have the
           same length!

        Either recurrence thresholds ``threshold``/``threshold_std`` or
        recurrence rates ``recurrence_rate`` have to be given as keyword
        arguments.

        Embedding is only supported for scalar time series. If embedding
        dimension ``dim`` and delay ``tau`` are **both** given as keyword
        arguments, embedding is applied. Multidimensional time series are
        processed as is by default.

        :type x: 2D array (time, dimension)
        :arg x: Time series x to be analyzed (scalar/multi-dimensional).
        :type y: 2D array (time, dimension)
        :arg y: Time series y to be analyzed (scalar/multi-dimensional).
        :type metric: tuple of string
        :arg metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum"). Give separately for each
            time series.
        :type normalize: tuple of bool
        :arg normalize: Decide whether to normalize the time series to zero
            mean and unit standard deviation. Give separately for each time
            series.
        :arg number lag: To create a delayed version of the JRP.
        :arg number silence_level: Inverse level of verbosity of the object.
        :type threshold: tuple of number
        :keyword threshold: The recurrence threshold keyword for generating the
            recurrence plot using a fixed threshold.  Give separately for each
            time series.
        :type threshold_std: tuple of number
        :keyword threshold_std: The recurrence threshold keyword for generating
            the recurrence plot using a fixed threshold in units of the time
            series' STD. Give separately for each time series.
        :type recurrence_rate: tuple of number
        :keyword recurrence_rate: The recurrence rate keyword for generating
            the recurrence plot using a fixed recurrence rate. Give separately
            for each time series.
        :type dim: tuple of number
        :keyword dim: Embedding dimension. Give separately for each time
            series.
        :type tau: tuple of number
        :keyword tau: Embedding delay. Give separately for each time series.
        """
        threshold = kwds.get("threshold")
        threshold_std = kwds.get("threshold_std")
        recurrence_rate = kwds.get("recurrence_rate")

        RecurrencePlot.__init__(
            self, np.empty((2, 0)), metric=metric[0], normalize=normalize,
            threshold=threshold[0] if threshold else 0,
            recurrence_rate=recurrence_rate, silence_level=silence_level)

        self._distance_matrix_cached = False
        self.JR = None
        """The joint recurrence matrix."""
        self.N = 0
        """The length of both embedded time series x and y."""

        #  Check for consistency: x and y need to have the same length
        if x.shape[0] == y.shape[0]:

            #  Store time series
            self.x = x.copy().astype("float32")
            """The time series x."""
            self.y = y.copy().astype("float32")
            """The time series y."""

            #  Reshape time series
            self.x.shape = (self.x.shape[0], -1)
            self.y.shape = (self.y.shape[0], -1)

            #  Normalize time series
            if normalize:
                self.normalize_time_series(self.x)
                self.normalize_time_series(self.y)

            #  Store lag
            self.lag = lag
            """Used to create a delayed JRP."""

            #  Get embedding dimension and delay from **kwds
            dim = kwds.get("dim")
            tau = kwds.get("tau")

            if dim is not None and tau is not None:
                #  Embed the time series
                self.x_embedded = \
                    self.embed_time_series(self.x, dim[0], tau[0])
                """The embedded time series x."""
                self.y_embedded = \
                    self.embed_time_series(self.y, dim[1], tau[1])
                """The embedded time series y."""
            else:
                self.x_embedded = self.x
                self.y_embedded = self.y

            #  Prune embedded time series to same length
            #  (number of state vectors)
            min_N = min(self.x_embedded.shape[0], self.y_embedded.shape[0])
            self.x_embedded = self.x_embedded[:min_N, :]
            self.y_embedded = self.y_embedded[:min_N, :]

            #  construct recurrence plot accordingly to
            #  threshold / recurrence rate
            if np.abs(lag) > x.shape[0]:
                #  Lag must be smaller than size of recurrence plot
                raise ValueError("Delay value (lag) must not exceed length of \
                                 time series!")
            if threshold is not None:
                #  Calculate the recurrence matrix R using the radius of
                #  neighborhood threshold
                JointRecurrencePlot.set_fixed_threshold(self, threshold)
            elif threshold_std is not None:
                #  Calculate the recurrence matrix R using the radius of
                #  neighborhood threshold in units of the time series' STD
                JointRecurrencePlot.\
                    set_fixed_threshold_std(self, threshold_std)
            elif recurrence_rate is not None:
                #  Calculate the recurrence matrix R using a fixed recurrence
                #  rate
                JointRecurrencePlot.\
                    set_fixed_recurrence_rate(self, recurrence_rate)
            else:
                raise NameError("Please give either threshold or \
                                recurrence_rate to construct the joint \
                                recurrence plot!")

            #  No treatment of missing values yet!
            self.missing_values = False

        else:
            raise ValueError("Both time series x and y need to have the same \
                             length!")

    def __str__(self):
        """
        Returns a string representation.
        """
        return ('JointRecurrencePlot: time series shapes %s.\n'
                'Embedding dimension %i\nThreshold %s, %s metric') % (
                    self.x.shape, self.dim if self.dim else 0,
                    self.threshold, self.metric)

    #
    #  Service methods
    #

    def recurrence_matrix(self):
        """
        Return the current joint recurrence matrix :math:`JR`.

        :rtype: 2D square Numpy array
        :return: the current joint recurrence matrix :math:`JR`.
        """
        return self.JR

    #
    #  Calculate recurrence plot
    #

    def set_fixed_threshold(self, threshold):
        """
        Set the joint recurrence plot to fixed thresholds.

        Modifies / sets the class variables :attr:`JR` and :attr:`N`
        accordingly.

        :type threshold: tuple of number
        :arg threshold: The recurrence threshold. Give for both time series
            separately.
        """
        if self.silence_level <= 1:
            print("Calculating joint recurrence plot at fixed threshold...")

        #  Disable caching of distances in RecurrencePlot class
        self._distance_matrix_cached = False
        #  Get distance matrix for the first time series
        distance = self.distance_matrix(self.x_embedded, self.metric[0])

        #  Get length of time series
        N = distance.shape[0]

        #  Initialize first recurrence matrix
        recurrence_x = np.zeros((N, N), dtype="int8")

        #  Thresholding the first distance matrix
        recurrence_x[distance < threshold[0]] = 1

        #  Clean up
        del distance

        #  Disable caching of distances in RecurrencePlot class
        self._distance_matrix_cached = False
        #  Get distance matrix for the second time series
        distance = self.distance_matrix(self.y_embedded, self.metric[1])

        #  Initialize second recurrence matrix
        recurrence_y = np.zeros((N, N), dtype="int8")

        #  Thresholding the second distance matrix
        recurrence_y[distance < threshold[1]] = 1

        if self.lag >= 0:
            self.JR = recurrence_x[:N-self.lag, :N-self.lag] * \
                recurrence_y[self.lag:N, self.lag:N]
        else:
            # self.JR = recurrence_x[-self.lag:N, -self.lag:N] * \
            #     recurrence_y[:N+self.lag, :N+self.lag]
            self.JR = recurrence_y[:N+self.lag, :N+self.lag] * \
                recurrence_x[-self.lag:N, -self.lag:N]
        self.N = N

        #  Clean up
        del distance, recurrence_x, recurrence_y

    def set_fixed_threshold_std(self, threshold_std):
        """
        Set the joint recurrence plot to fixed thresholds in units of the
        standard deviation of the time series.

        Calculates the absolute thresholds and calls
        :meth:`set_fixed_threshold`.

        :type threshold_std: tuple of number
        :arg threshold_std: The recurrence threshold in units of the standard
            deviation of the time series. Give for both time series separately.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot at fixed threshold "
                  "in units of time series STD...")

        #  Get absolute threshold
        threshold_x = threshold_std[0] * self.x.std()
        threshold_y = threshold_std[1] * self.y.std()

        #  Call set fixed threshold method
        JointRecurrencePlot.\
            set_fixed_threshold(self, (threshold_x, threshold_y))

    def set_fixed_recurrence_rate(self, recurrence_rate):
        """
        Set the joint recurrence plot to fixed recurrence rates.

        Modifies / sets the class variables :attr:`JR` and :attr:`N`
        accordingly.

        :type recurrence_rate: tuple of number
        :arg recurrence_rate: The recurrence rate. Give for both time series
            separately.
        """
        if self.silence_level <= 1:
            print("Calculating joint recurrence plot at "
                  "fixed recurrence rate...")

        #  Disable caching of distances in RecurrencePlot class
        self._distance_matrix_cached = False
        #  Get distance matrix for the first time series
        distance = self.distance_matrix(self.x_embedded, self.metric[0])

        #  Get length of time series
        N = distance.shape[0]

        #  Get first threshold to obtain fixed recurrence rate
        threshold_x = self.\
            threshold_from_recurrence_rate(distance, recurrence_rate[0])

        #  Initialize recurrence matrix
        recurrence_x = np.zeros((N, N), dtype="int8")

        #  Thresholding the distance matrix
        recurrence_x[distance < threshold_x] = 1

        #  Clean up
        del distance

        #  Disable caching of distances in RecurrencePlot class
        self._distance_matrix_cached = False
        #  Get distance matrix for the second time series
        distance = self.distance_matrix(self.y_embedded, self.metric[1])

        #  Get first threshold to obtain fixed recurrence rate
        threshold_y = self.\
            threshold_from_recurrence_rate(distance, recurrence_rate[1])

        #  Initialize recurrence matrix
        recurrence_y = np.zeros((N, N), dtype="int8")

        #  Thresholding the distance matrix
        recurrence_y[distance < threshold_y] = 1

        if self.lag >= 0:
            self.JR = recurrence_x[:N-self.lag, :N-self.lag] * \
                recurrence_y[self.lag:N, self.lag:N]
        else:
            # self.JR = recurrence_x[-self.lag:N, -self.lag:N] * \
            #     recurrence_y[:N+self.lag, :N+self.lag]
            self.JR = recurrence_y[:N+self.lag, :N+self.lag] * \
                recurrence_x[-self.lag:N, -self.lag:N]
        self.N = N

        #  Clean up
        del distance, recurrence_x, recurrence_y
