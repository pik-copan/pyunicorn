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
except ImportError:
    print("climate: Package scipy.signal could not be loaded. Some "
          "functionality in class HilbertClimateNetwork might not be "
          "available!")

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
            print("Generating a Hilbert climate network...")
        self.silence_level = silence_level

        #  Set instance variables
        self._coherence_phase = None
        self.data = data
        """(ClimateData) - The climate data used for network construction."""
        self.N = data.grid.N
        self._threshold = threshold
        self._prescribed_link_density = link_density

        self._set_directed(directed, calculate_coherence=True)
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=self._similarity_measure,
                                threshold=threshold,
                                link_density=link_density,
                                non_local=non_local,
                                directed=directed,
                                node_weight_type=node_weight_type,
                                silence_level=silence_level)
        self._set_directed(directed, calculate_coherence=False)

    def __str__(self):
        """
        Return a string representation.
        """
        return 'HilbertClimateNetwork:\n' + ClimateNetwork.__str__(self)

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
            except AttributeError:
                pass

    #
    #  Defines methods to calculate Hilbert correlation measures
    #

    def _set_directed(self, directed, calculate_coherence=True):
        """
        Switch between directed and undirected Hilbert climate network.

        :arg bool directed: Determines whether the network is constructed as
            directed.
        :arg bool calculate_coherence: Determines whether coherence and phase
            are calculated from data or the directed adjacency matrix is
            constructed from coherence and phase information.
        """
        if calculate_coherence:
            results = self._calculate_hilbert_correlation(self.data.anomaly())
            self._similarity_measure = results[0]
            self._coherence_phase = results[1]
            self.directed = directed
        else:
            # The phase is only used for directed Hilbert networks.
            if directed:
                self.adjacency = self.adjacency * (self.phase_shift() > 0)

    def set_directed(self, directed):
        """
        Switch between directed and undirected Hilbert climate network.

        Also performs the complete network generation.

        :arg bool directed: Determines whether the network is constructed as
            directed.
        """
        self._set_directed(directed, calculate_coherence=True)
        self._regenerate_network()
        self._set_directed(directed, calculate_coherence=False)

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
            print("Calculating Hilbert transform correlation measures "
                  "following [Bergner2008]_...")

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
