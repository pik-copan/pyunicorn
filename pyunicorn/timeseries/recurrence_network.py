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
from .recurrence_plot import RecurrencePlot


#
#  Class definitions
#

class RecurrenceNetwork(RecurrencePlot, Network):

    """
    Class RecurrenceNetwork for generating and quantitatively analyzing
    recurrence networks.

    More information on the theory and applications of recurrence networks can
    be found in [Marwan2009]_, [Donner2010b]_.

    Examples:

     - Create an instance of RecurrenceNetwork with a fixed recurrence
       threshold and without embedding::

           RecurrenceNetwork(time_series, threshold=0.1)

     - Create an instance of RecurrenceNetwork at a fixed (global) recurrence
       rate and using time delay embedding::

           RecurrenceNetwork(time_series, dim=3, tau=2,
                             recurrence_rate=0.05).recurrence_rate()
    """

    #
    #  Internal methods
    #

    def __init__(self, time_series, metric="supremum", normalize=False,
                 missing_values=False, silence_level=0, **kwds):
        """
        Initialize an instance of RecurrenceNetwork.

        Creates an embedding of the given time series, calculates a recurrence
        plot from the embedding and then creates a Network object from the
        recurrence plot, interpreting the recurrence matrix as the adjacency
        matrix of a complex network.

        Either recurrence threshold ``threshold``/``threshold_std``, recurrence
        rate ``recurrence_rate`` or local recurrence rate
        ``local_recurrence_rate`` have to be given as keyword arguments.

        Embedding is only supported for scalar time series. If embedding
        dimension ``dim`` and delay ``tau`` are **both** given as keyword
        arguments, embedding is applied. Multidimensional time series are
        processed as is by default.

        :type time_series: 2D array (time, dimension)
        :arg time_series: The time series to be analyzed, can be scalar or
            multi-dimensional.
        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :arg bool normalize: Decide whether to normalize the time series to
            zero mean and unit standard deviation.
        :arg bool missing_values: Toggle special treatment of missing values in
            :attr:`.RecurrencePlot.time_series`.
        :arg number silence_level: Inverse level of verbosity of the object.
        :arg number threshold: The recurrence threshold keyword for generating
            the recurrence network using a fixed threshold.
        :arg number threshold_std: The recurrence threshold keyword for
            generating the recurrence plot using a fixed threshold in units of
            the time series' STD.
        :arg number recurrence_rate: The recurrence rate keyword for generating
            the recurrence network using a fixed recurrence rate.
        :arg number local_recurrence_rate: The local recurrence rate keyword
            for generating the recurrence plot using a fixed local recurrence
            rate (same number of recurrences for each state vector).
        :arg number adaptive_neighborhood_size: The adaptive neighborhood size
            parameter for generating recurrence plots based on the algorithm in
            [Xu2008]_.
        :arg number dim: The embedding dimension.
        :arg number tau: The embedding delay.
        :type node_weights: 1D array (time)
        :arg node_weights: The sequence of weights associated with each
            node for calculating n.s.i. network measures.
        """
        #  Initialize the underlying RecurrencePlot object
        RecurrencePlot.__init__(self, time_series=time_series, metric=metric,
                                normalize=normalize,
                                missing_values=missing_values,
                                silence_level=silence_level, **kwds)

        #  Set diagonal of R to zero to avoid self-loops in the recurrence
        #  network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  Get keywords
        local_recurrence_rate = kwds.get("local_recurrence_rate")
        node_weights = kwds.get("node_weights")

        #  Assign to each embedded state vector the weight of the earliest
        #  sample contained in it (too simple!)
        if node_weights is not None:
            node_weights = node_weights[:self.N]

        #  Remove state vectors containing missing values
        if missing_values:
            A = np.delete(A, np.where(self.missing_value_indices), axis=0)
            A = np.delete(A, np.where(self.missing_value_indices), axis=1)

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix.
        if local_recurrence_rate is not None:
            #  A recurrence network with fixed local recurrence rate (Eckmann
            #  definition of a recurrence plot) is directed by definition.
            Network.__init__(self, A, directed=True, node_weights=node_weights,
                             silence_level=silence_level)
        else:
            #  Create a Network object interpreting the recurrence matrix as
            #  the graph adjacency matrix. Recurrence networks are undirected
            #  by definition.
            Network.__init__(self, A, directed=False,
                             node_weights=node_weights,
                             silence_level=silence_level)

    def __str__(self):
        """
        Returns a string representation.
        """
        return 'RecurrenceNetwork:\n%s\n%s' % (
            RecurrencePlot.__str__(self), Network.__str__(self))

    def clear_cache(self):
        """
        Clean up memory by deleting information that can be recalculated from
        basic data.

        Extends the clean up methods of the parent classes.
        """
        #  Call clean up of RecurrencePlot
        RecurrencePlot.clear_cache(self)
        #  Call clean up of Network
        Network.clear_cache(self)

    #
    #  Methods to handle recurrence networks
    #
    def set_fixed_threshold(self, threshold):
        """
        Create a recurrence network at a fixed threshold.

        :arg number threshold: The threshold.
        """
        #  Set fixed threshold on recurrence plot level
        RecurrencePlot.set_fixed_threshold(self, threshold)

        #  Set diagonal of R to zero to avoid self-loops in the recurrence
        #  network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def set_fixed_threshold_std(self, threshold_std):
        """
        Set the recurrence network to a fixed threshold in units of the
        standard deviation of the time series.

        :arg number threshold_std: The recurrence threshold in units of the
            standard deviation of the time series.
        """
        #  Set fixed threshold in units of STD on recurrence plot level
        RecurrencePlot.set_fixed_threshold_std(self, threshold_std)

        #  Set diagonal of R to zero to avoid self-loops in the recurrence
        #  network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def set_fixed_recurrence_rate(self, recurrence_rate):
        """
        Create a recurrence network at a fixed link density (recurrence rate).

        :arg number recurrence_rate: The link density / recurrence rate.
        """
        #  Set fixed recurrence rate on recurrence plot level
        RecurrencePlot.set_fixed_recurrence_rate(self, recurrence_rate)

        #  Set diagonal of R to zero to avoid self-loops in the recurrence
        #  network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Recurrence networks are undirected by
        #  definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def set_fixed_local_recurrence_rate(self, local_recurrence_rate):
        """
        Create a recurrence network at a fixed local recurrence rate.

        This leads to a directed recurrence network with identical out-degree
        :math:`int(N * local_recurrence_rate)`, but variable in-degree. The
        associated recurrence plot coincides with the original Eckmann
        definition.

        :arg number local_recurrence_rate: The local recurrence rate.
        """
        #  Set fixed local recurrence rate on recurrence plot level
        RecurrencePlot.\
            set_fixed_local_recurrence_rate(self, local_recurrence_rate)

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Set diagonal of R to zero to avoid
        #  self-loops in the recurrence network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  A recurrence network with fixed local recurrence rate (Eckmann
        #  definition of a recurrence plot) is directed by definition.
        Network.__init__(self, A, directed=True,
                         silence_level=self.silence_level)

    def set_adaptive_neighborhood_size(self, adaptive_neighborhood_size,
                                       order=None):
        """
        Create a recurrence network using the :index:`adaptive neighborhood
        size <single: adaptive neighborhood size; recurrence network>`
        algorithm used in [Xu2008]_.

        The exact algorithm was deduced from private correspondence with the
        authors.  It leads to an undirected network with mean degree
        :math:`<k> = 2 * m`, where m is the adaptive_neighborhood_size.
        The degree :math:`k_v` of single nodes may vary, but :math:`k_v >= m`
        holds!

        :arg number adaptive_neighborhood_size: The number of adaptive nearest
            neighbors (recurrences) assigned to each state vector.
        :type order: 1D array (int32)
        :arg order: The indices of state vectors in the order desired for
            processing by the algorithm.
        """
        #  Set adaptive neighborhood size on recurrence plot level
        RecurrencePlot.set_adaptive_neighborhood_size(
            self, adaptive_neighborhood_size, order)

        #  Create a Network object interpreting the recurrence matrix as the
        #  graph adjacency matrix. Set diagonal of R to zero to avoid
        #  self-loops in the recurrence network
        A = self.R.copy()
        A.flat[::self.N+1] = 0

        #  A recurrence network with fixed local recurrence rate (Eckmann
        #  definition of a recurrence plot) is directed by definition.
        Network.__init__(self, A, directed=False,
                         silence_level=self.silence_level)

    def transitivity_dim_single_scale(self):
        """
        Return transitivity dimension for a single scale.

        The single scale transitivity dimension can be interpreted as a global
        measure of the dimensionality of the set of points underlying the
        recurrence network ([Donner2011b]_.). The scale is determined by the
        chosen recurrence threshold. Note that the maxima and minima of the
        single scale transitivity dimension when varying the scale give a
        more meaningful measure of dimensionality as is explained in
        [Donner2011b]_.

        **Attention:** currently only works correctly for supremum norm.

        :rtype: float
        :return: the single scale transitivity dimension.
        """
        return np.log(self.transitivity()) / np.log(3. / 4.)

    def local_clustering_dim_single_scale(self):
        """
        Return local clustering dimension for a single scale.

        The single scale local clustering dimension can be interpreted as a
        local measureof the dimensionality of the set of points underlying the
        recurrence network ([Donner2011b]_.). The scale is determined by the
        chosen recurrence threshold. Note that the maxima and minima of the
        single scale local clustering dimension when varying the scale give a
        more meaningful measure of dimensionality as is explained in
        [Donner2011b]_.

        **Attention:**
        currently only works correctly for supremum norm.

        :rtype: 1d numpy array [node] of float
        :return: the single scale transitivity dimension.
        """
        return np.log(self.local_clustering()) / np.log(3. / 4.)
