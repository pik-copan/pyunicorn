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

from ..core import Network
from .joint_recurrence_plot import JointRecurrencePlot


#
#  Class definitions
#

class JointRecurrenceNetwork(JointRecurrencePlot, Network):

    """
    Class JointRecurrenceNetwork for generating and quantitatively analyzing
    joint recurrence networks.

    For a joint recurrence network, time series x and y need to have the same
    length! Formally, nodes are identified with sampling points in time, while
    an undirected link (i,j) is introduced if x at time i is recurrent to x at
    time j and also y at time i is recurrent to y at time j. Self-loops are
    excluded in this undirected network representation.

    More information on the theory and applications of joint recurrence
    networks can be found in [Feldhoff2013]_.

    **Examples:**

     - Create an instance of JointRecurrenceNetwork with a fixed recurrence
       threshold and without embedding::

           JointRecurrenceNetwork(x, y, threshold=(0.1,0.2))

     - Create an instance of JointRecurrenceNetwork with a fixed recurrence
       threshold in units of STD and without embedding::

           JointRecurrenceNetwork(x, y, threshold_std=(0.03,0.05))

     - Create an instance of JointRecurrenceNetwork at a fixed recurrence rate
       and using time delay embedding::

           JointRecurrenceNetwork(
               x, y, dim=(3,5), tau=(2,1),
               recurrence_rate=(0.05,0.04)).recurrence_rate()
    """

    #
    #  Internal methods
    #

    def __init__(self, x, y, metric=("supremum", "supremum"),
                 normalize=False, lag=0, silence_level=0, **kwds):
        """
        Initialize an instance of JointRecurrenceNetwork.

        .. note::
           For a joint recurrence network, time series x and y need to have the
           same length!

        Creates an embedding of the given time series x and y, calculates a
        joint recurrence plot from the embedding and then creates a Network
        object from the joint recurrence plot, interpreting the joint
        recurrence matrix as the adjacency matrix of an undirected complex
        network.

        Either recurrence thresholds ``threshold``/``threshold_std`` or
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
            ("manhattan", "euclidean", "supremum").  Give separately for each
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
        :keyword dim: The embedding dimension. Give separately for each time
            series.
        :type tau: tuple of number
        :keyword tau: The embedding delay. Give separately for each time
            series.
        """
        #  Check for consistency
        if np.abs(lag) < x.shape[0]:
            if x.shape[0] == y.shape[0]:
                #  Initialize the underlying RecurrencePlot object
                JointRecurrencePlot.__init__(self, x, y, metric, normalize,
                                             lag, silence_level, **kwds)

                #  Set diagonal of JR to zero to avoid self-loops in the joint
                #  recurrence network
                A = self.JR - np.eye((self.N-np.abs(lag)), dtype="int8")

                #  Create a Network object interpreting the recurrence matrix
                #  as the graph adjacency matrix. Joint recurrence networks
                #  are undirected by definition.
                Network.__init__(self, A, directed=False,
                                 silence_level=silence_level)
            else:
                raise ValueError("Both time series x and y need to have the \
                                 same length!")
        else:
            raise ValueError("Delay value (lag) must not exceed length of \
                             time series!")

    def __str__(self):
        """
        Returns a string representation.
        """
        return 'JointRecurrenceNetwork:\n%s\n%s' % (
            JointRecurrencePlot.__str__(self), Network.__str__(self))

    def clear_cache(self):
        """
        Clean up memory by deleting information that can be recalculated from
        basic data.

        Extends the clean up methods of the parent classes.
        """
        #  Call clean up of RecurrencePlot
        JointRecurrencePlot.clear_cache(self)
        #  Call clean up of Network
        Network.clear_cache(self)

    #
    #  Methods to handle recurrence networks
    #
    def set_fixed_threshold(self, threshold):
        """
        Create a joint recurrence network at fixed thresholds.

        :type threshold: tuple of number
        :arg threshold: The threshold. Give for each time series separately.
        """
        #  Set fixed threshold on recurrence plot level
        JointRecurrencePlot.set_fixed_threshold(self, threshold)

        #  Set diagonal of JR to zero to avoid self-loops in the joint
        #  recurrence network
        A = self.JR.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Joint recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def set_fixed_threshold_std(self, threshold_std):
        """
        Create a joint recurrence network at fixed thresholds in units of the
        standard deviation of the time series.

        :type threshold_std: tuple of number
        :arg threshold_std: The threshold in units of standard deviation. Give
            for each time series separately.
        """
        #  Set fixed threshold on recurrence plot level
        JointRecurrencePlot.set_fixed_threshold_std(self, threshold_std)

        #  Set diagonal of JR to zero to avoid self-loops in the joint
        #  recurrence network
        A = self.JR.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Joint recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def set_fixed_recurrence_rate(self, density):
        """
        Create a joint recurrence network at fixed link densities (recurrence
        rates).

        :type density: tuple of number
        :arg density: The link density / recurrence rate. Give for each time
            series separately.
        """
        #  Set fixed recurrence rate on recurrence plot level
        JointRecurrencePlot.set_fixed_recurrence_rate(self, density)

        #  Set diagonal of JR to zero to avoid self-loops in the joint
        #  recurrence network
        A = self.JR.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Joint recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)
