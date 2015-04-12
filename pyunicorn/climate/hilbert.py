#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides classes for generating and analyzing complex climate networks.
"""

#
#  Import essential packages
#

#  Import NumPy for the array object and fast numerics
import numpy as np

#  Import scipy.signal for signal processing
try:
    import scipy.signal
except:
    print "climate: Package scipy.signal could not be loaded. Some \
functionality in class HilbertClimateNetwork might not be available!"

#  Import cnNetwork for Network base class
from .climate_network import ClimateNetwork


#
#  Define class HilbertClimateNetwork
#

class HilbertClimateNetwork(ClimateNetwork):

    """
    Encapsulates a Hilbert climate network.

    The associated similarity matrix is based on Hilbert coherence between
    time series.

    Hilbert climate networks can be directed and undirected. Optional
    directionality is based on the average phase difference between time
    series.

    A preliminary study of Hilbert climate networks is presented in
    [Donges2009c]_.

    Derived from the class ClimateNetwork.
    """

    #
    #  Defines internal methods
    #

    def __init__(self, data, threshold=None, link_density=None,
                 non_local=False, directed=True, node_weight_type="surface",
                 silence_level=0):
        """
        Initialize an instance of HilbertClimateNetwork.

        .. note::
           Either threshold **OR** link_density have to be given!

        Possible choices for ``node_weight_type``:
          - None (constant unit weights)
          - "surface" (cos lat)
          - "irrigation" (cos**2 lat)

        :type data: :class:`.ClimateData`
        :arg data: The climate data used for network construction.
        :arg float threshold: The threshold of similarity measure, above which
            two nodes are linked in the network.
        :arg float link_density: The networks's desired link density.
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg bool directed: Determines, whether the network is constructed as
            directed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print "Generating a Hilbert climate network..."

        #  Set instance variables
        self.data = data
        """(ClimateData) - The climate data used for network construction."""

        self.N = data.grid.N
        self._threshold = threshold
        self._prescribed_link_density = link_density
        self._non_local = non_local
        self.node_weight_type = node_weight_type
        self.silence_level = silence_level

        #  The constructor of ClimateNetwork is called within this method
        self.set_directed(directed)

    def __str__(self):
        """Return a string representation."""
        text = "Hilbert climate network: \n"
        text += ClimateNetwork.__str__(self)

        return text

    def clear_cache(self, irreversible=False):
        """
        Clean up cache.

        If irreversible=True, the network cannot be recalculated using a
        different threshold, or link density.

        :arg bool irreversible: The irreversibility of clearing the cache.
        """
        ClimateNetwork.clear_cache(self, irreversible)

        if irreversible:
            try:
                del self._coherence_phase
            except:
                pass

    #
    #  Defines methods to calculate Hilbert correlation measures
    #

    def set_directed(self, directed):
        """
        Switch between directed and undirected Hilbert climate network.

        Also performs the complete network generation.

        :arg bool directed: Determines, whether the network is constructed as
            directed.
        """
        self.directed = directed

        #  Calculate coherence and phase from data.
        #  The phase is only used for directed Hilbert networks.
        results = self._calculate_hilbert_correlation(self.data.anomaly())
        similarity_measure = results[0]
        self._coherence_phase = results[1]

        #  Call constructor of parent class ClimateNetwork
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=similarity_measure,
                                threshold=self.threshold(),
                                link_density=self._prescribed_link_density,
                                non_local=self.non_local(),
                                directed=directed,
                                node_weight_type=self.node_weight_type,
                                silence_level=self.silence_level)

        if directed:
            #  Calculate the directed adjacency matrix using coherence phase
            #  information and reconstruct the network using this information.
            directed_adjacency = \
                self.adjacency * (self.phase_shift() > 0)
            self.adjacency = directed_adjacency

    def _calculate_hilbert_correlation(self, anomaly):
        """
        Calculate Hilbert coherence and phase matrices.

        Output corresponds to modulus and argument of the complex correlation
        coefficients between all pairs of analytic signals calculated from
        anomaly data, as described in [Bergner2008]_.

        :type anomaly: 2D Numpy array [time, index]
        :arg anomaly: The anomaly data for network construction.

        :rtype: tuple of two 2D Numpy matrices [index, index]
        :return: the Hilbert coherence and phase matrices.
        """
        if self.silence_level <= 1:
            print "Calculating Hilbert transform correlation measures \
following [Bergner2008]_..."

        #  Calculate the analytic signals associated with the anomaly time
        #  series.
        analytic_signals = np.apply_along_axis(scipy.signal.hilbert, 0,
                                               anomaly)

        #  Normalize analytic signal time series to zero mean and unit variance
        self.data.normalize_time_series_array(analytic_signals)

        #  Get length of time series
        n_time = anomaly.shape[0]

        #  Calculate matrix of complex correlation coefficients
        complex_corr = np.dot(analytic_signals.transpose(),
                              analytic_signals.conjugate()) / float(n_time - 1)

        #  Clean up
        del analytic_signals

        #  Calculate the coherence, i.e. the modulus of the complex
        #  correlation coefficient
        coherence = np.abs(complex_corr)

        #  Calculate the average phase between signals
        phase = np.angle(complex_corr)

        return (coherence, phase)

    def coherence(self):
        """
        Return the Hilbert coherence matrix.

        :rtype: 2D Numpy array [index, index]
        :return: the Hilbert coherence matrix.
        """
        return self.similarity_measure()

    def phase_shift(self):
        """
        Return the average phase shift matrix.

        :rtype: 2D Numpy array [index, index]
        :return: the average phase shift matrix.
        """
        return self._coherence_phase
