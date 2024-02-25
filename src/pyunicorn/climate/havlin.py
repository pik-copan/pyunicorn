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
Provides classes for generating and analyzing complex climate networks.
"""

from typing import Tuple
from collections.abc import Hashable

import numpy as np
from tqdm import trange

from .climate_data import ClimateData
from .climate_network import ClimateNetwork


class HavlinClimateNetwork(ClimateNetwork):

    """
    Encapsulates a Havlin climate network.

    The similarity matrix associated with a Havlin climate network is the
    maximum-lag correlation matrix with each entry normalized by the
    cross-correlation function's standard deviation.

    Havlin climate networks are undirected so far.

    Havlin climate networks were studied for daily data in [Yamasaki2008]_,
    [Gozolchiani2008]_, [Yamasaki2009]_.

    .. note::
       So far, the cross-correlation functions are estimated using \
       convolution in Fourier space (FFT). This may not be reliable \
       for larger delays.
    """

    #
    #  Defines internal methods
    #

    def __init__(self, data, max_delay, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface", silence_level=0):
        """
        Initialize an instance of HavlinClimateNetwork.

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
        :param int max_delay: Maximum delay for cross-correlation functions.
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print("Generating a Havlin climate network...")
        self.silence_level = silence_level

        #  Set instance variables
        self._max_delay = 0
        self._correlation_lag = None
        assert isinstance(data, ClimateData)
        self.data: ClimateData = data
        """The climate data used for network construction."""
        self.N = data.grid.N
        self._prescribed_link_density = link_density

        self._mut_clim: int = 0
        self._set_max_delay(max_delay)
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=self._similarity_measure,
                                threshold=threshold,
                                link_density=link_density,
                                non_local=non_local,
                                directed=False,
                                node_weight_type=node_weight_type,
                                silence_level=silence_level)

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return ClimateNetwork.__cache_state__(self) + (self.data,)

    def __str__(self):
        """
        Return a string version of the instance of HavlinClimateNetwork.
        """
        return f'HavlinClimateNetwork:\n \
                {ClimateNetwork.__str__(self)}\n \
                Maximum delay used for correlation strength estimation: \
                {self.get_max_delay()}'

    def clear_cache(self):
        """
        Clean up cache.
        """
        try:
            del self._correlation_lag
        except AttributeError:
            pass

    #
    #  Defines methods to calculate correlation strength and lags
    #

    #  TODO: Implement an algorithm, which is NOT based on FFT!
    def _calculate_correlation_strength(self, anomaly, max_delay, gamma=0.2):
        """
        Calculate correlation strength and maximum lag matrices.

        Follows the method described in [Yamasaki2008]_.

        Also returns the time lag at maximum correlation for each link.

        :type anomaly: 2D array [time, index]
        :arg anomaly: The anomaly data for network construction.
        :arg int max_delay: The maximum delay for cross-correlation functions.
        :arg float gamma: The width of decay region in cosine shaped window
            used for FFT cross-correlation estimation.
        :rtype: tuple of two 2D arrays [index, index]
        :return: the correlation strength and maximum lag matrices.
        """
        N = self.N
        self.data.normalize_time_series_array(anomaly)
        anomaly *= self.data.cos_window(anomaly, gamma)
        #  Zero pad windowed data to set the length of each time series to
        #  a power of two
        anomaly = self.data.zero_pad_data(anomaly)
        correlation_strength = np.empty((N, N))
        max_lag_matrix = np.empty((N, N))

        #  Calculate the inverse Fourier transform of all time series
        ifft = np.fft.ifft(anomaly, axis=0)

        for i in trange(N, disable=self.silence_level > 1):
            #  Calculate the cross correlation function of node i to all other
            #  nodes which is not normalized yet.
            #  The real part has to be taken to get rid of small imaginary
            #  parts due to rounding errors.
            cc_one_to_all = np.conjugate(np.tile(ifft[:, i],
                                                 (N, 1)).transpose()) * ifft
            cc_one_to_all = np.real(np.fft.fft(cc_one_to_all, axis=0))

            #  Consider only the cross correlations between -max_delay and
            #  +max_delay.
            #  Correlation values at negative lag are now stored left of the
            #  correlation values at positive lag.
            cc_one_to_all = np.concatenate((cc_one_to_all[-max_delay:-1, :],
                                            cc_one_to_all[0:max_delay, :]))

            #  Consider only absolute values
            cc_one_to_all = np.abs(cc_one_to_all)

            #  Store correlation strengths
            correlation_strength[i, :] = \
                cc_one_to_all.max(axis=0) / cc_one_to_all.std(axis=0)

            #  Store time delays at maximum cross correlation
            max_lag_matrix[i, :] = cc_one_to_all.argmax(axis=0) - max_delay

        return (correlation_strength, max_lag_matrix)

    def get_max_delay(self):
        """
        Return the maximum delay used for cross-correlation estimation.

        :return float: the maximum delay used for cross-correlation estimation.
        """
        return self._max_delay

    def _set_max_delay(self, max_delay):
        """
        Set the maximum lag time used for cross-correlation estimation.

        :arg int max_delay: The maximum delay for cross-correlation functions.
        """
        self._mut_clim += 1
        self._max_delay = max_delay
        results = self._calculate_correlation_strength(self.data.anomaly(),
                                                       max_delay)
        self._similarity_measure = results[0]
        self._correlation_lag = results[1]

    def set_max_delay(self, max_delay):
        """
        Set the maximum lag time used for cross-correlation estimation.

        (Re)generates the current Havlin climate network accordingly.

        :arg int max_delay: The maximum delay for cross-correlation functions.
        """
        self._set_max_delay(max_delay)
        self._regenerate_network()

    def correlation_strength(self):
        """
        Return the correlation strength matrix.

        :rtype: 2D array [index, index]
        :return: the correlation strength matrix.
        """
        return self._similarity_measure

    def correlation_lag(self):
        """
        Return the lag at maximum cross-correlation matrix.

        :rtype: 2D array [index, index]
        :return: the lag at maximum cross-correlation matrix.
        """
        return self._correlation_lag

    #
    #  Methods to calculate weighted network measures
    #

    def correlation_strength_weighted_average_path_length(self):
        """
        Return correlation strength weighted average path length.

        :return float: the correlation strength weighted average path length.
        """
        return self._weighted_metric(
            "correlation_strength",
            lambda: np.abs(self.correlation_strength()),
            "average_path_length")

    def correlation_strength_weighted_closeness(self):
        """
        Return correlation strength weighted closeness.

        :rtype: 1D array [index]
        :return: the correlation strength weighted closeness sequence.
        """
        return self._weighted_metric(
            "correlation_strength",
            lambda: np.abs(self.correlation_strength()),
            "closeness")

    def correlation_lag_weighted_average_path_length(self):
        """
        Return correlation lag weighted average path length.

        :return float: the correlation lag weighted average path length.
        """
        return self._weighted_metric(
            "correlation_lag",
            lambda: np.abs(self.correlation_lag()),
            "average_path_length")

    def correlation_lag_weighted_closeness(self):
        """
        Return correlation lag weighted closeness.

        :rtype: 1D array [index]
        :return: the correlation lag weighted closeness sequence.
        """
        return self._weighted_metric(
            "correlation_lag",
            lambda: np.abs(self.correlation_lag()),
            "closeness")

    def local_correlation_strength_weighted_vulnerability(self):
        """
        Return correlation strength weighted vulnerability.

        :rtype: 1D array [index]
        :return: the correlation strength weighted vulnerability sequence.
        """
        return self._weighted_metric(
            "correlation_strength",
            lambda: np.abs(self.correlation_strength()),
            "local_vulnerability")

    def local_correlation_lag_weighted_vulnerability(self):
        """
        Return correlation lag weighted vulnerability.

        :rtype: 1D array [index]
        :return: the correlation lag weighted vulnerability sequence.
        """
        return self._weighted_metric(
            "correlation_lag",
            lambda: np.abs(self.correlation_lag()),
            "local_vulnerability")
