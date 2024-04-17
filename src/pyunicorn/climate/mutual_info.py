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

from ..core._ext.types import to_cy, FIELD
from ._ext.numerics import mutual_information
from .climate_data import ClimateData
from .climate_network import ClimateNetwork


class MutualInfoClimateNetwork(ClimateNetwork):
    """
    Represents a mutual information climate network.

    Constructs a static climate network based on mutual information at zero
    lag, as in [Ueoka2008]_.

    Mutual information climate networks are undirected, since mutual
    information is a symmetrical measure. In contrast to Pearson correlation
    used in :class:`.TsonisClimateNetwork`, mutual information has the
    potential to detect nonlinear statistical interdependencies.
    """

    #
    #  Defines internal methods
    #

    def __init__(self, data, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface", winter_only=True,
                 silence_level=0):
        """
        Initialize an instance of MutualInfoClimateNework.

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
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg bool winter_only: Determines, whether only data points from the
            winter months (December, January and February) should be used for
            analysis. Possibly, this further suppresses the annual cycle in the
            time series.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print("Generating a mutual information climate network...")
        self.silence_level = silence_level

        #  Set instance variables
        assert isinstance(data, ClimateData)
        self.data: ClimateData = data
        """The climate data used for network construction."""
        self.N = self.data.grid.N
        self._prescribed_link_density = link_density
        self._winter_only = winter_only

        #  Class specific settings
        self.mi_file = "mutual_information_" + data.data_source + "_" \
            + data.observable_name + ".data"
        """(string) - The name of the file for storing the mutual information
                      matrix."""

        self._set_winter_only(winter_only)
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=self._similarity_measure,
                                threshold=threshold,
                                non_local=non_local,
                                directed=False,
                                node_weight_type=node_weight_type,
                                silence_level=silence_level)

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return ClimateNetwork.__cache_state__(self) + (self.data,)

    def __str__(self):
        """
        Return a string representation of MutualInfoClimateNetwork.
        """
        return 'MutualInfoClimateNetwork:\n' + ClimateNetwork.__str__(self)

    def _cython_calculate_mutual_information(self, anomaly, n_bins=32):
        """
        Calculate the mutual information matrix at zero lag.

        The cython code is adopted from the Tisean 3.0.1 mutual.c module.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :arg int n_bins: The number of bins for estimating probability
            distributions.
        :arg bool fast: Indicates, whether fast or slow algorithm should be
            used.
        :rtype: 2D array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        if self.silence_level <= 1:
            print("Calculating mutual information matrix at zero lag from "
                  "anomaly values using cython...")

        #  Normalize anomaly time series to zero mean and unit variance
        self.data.normalize_time_series_array(anomaly)

        #  Create local transposed copy of anomaly
        anomaly = anomaly.T.copy()

        (N, n_samples) = anomaly.shape

        #  Get common range for all histograms
        range_min = float(anomaly.min())
        range_max = float(anomaly.max())

        #  Rescale all time series to the interval [0,1],
        #  using the maximum range of the whole dataset.
        scaling = 1./(range_max - range_min)

        mi = mutual_information(
            to_cy(anomaly, FIELD), n_samples, N, n_bins, scaling, range_min)

        if self.silence_level <= 1:
            print("Done!")

        return mi

    def calculate_similarity_measure(self, anomaly):
        """
        Calculate the mutual information matrix.

        Encapsulates calculation of mutual information with standard
        parameters.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.

        :rtype: 2D Numpy array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        return self._cython_calculate_mutual_information(anomaly)

    def mutual_information(self, anomaly=None, dump=True):
        """
        Return mutual information matrix at zero lag.

        Check if mutual information matrix (MI) was already calculated before:
          - If yes, return MI from a data file.
          - If not, return MI from calculation and store in file.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: The anomaly time series.
        :arg bool dump: Store MI in data file.

        :rtype: 2D Numpy array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        try:
            #  Try to load MI from file
            if self.silence_level <= 1:
                print("Loading mutual information matrix from "
                      f"{self.mi_file}...")

            with open(self.mi_file, 'r', encoding="utf-8") as f:
                mi = np.load(f)
                #  Check if the dimensions of mutual_information correspond to
                #  the grid.
                if mi.shape != (self.N, self.N):
                    print(f"{self.mi_file} in current directory has "
                          "incorrect dimensions!")
                    raise RuntimeError

        except (IOError, RuntimeError):
            if self.silence_level <= 1:
                print("An error occured while loading data from "
                      f"{self.mi_file}.")
                print("Recalculating mutual information.")

            mi = self._cython_calculate_mutual_information(anomaly)
            if dump:
                with open(self.mi_file, 'w', encoding="utf-8") as f:
                    if self.silence_level <= 1:
                        print("Storing in", self.mi_file)
                    mi.dump(f)

        return mi

    def winter_only(self):
        """
        Indicate, if only winter months were used for network generation.

        :return bool: whether only winter months were used for network
            generation.
        """
        return self._winter_only

    def _set_winter_only(self, winter_only, dump=False):
        """
        Toggle use of exclusively winter data points for network generation.

        :arg bool winter_only: Indicates whether only winter months were used
            for network generation.
        :arg bool dump: Store MI in data file.
        """
        self._winter_only = winter_only
        if winter_only:
            winter_anomaly = self.data.anomaly_selected_months([0, 1, 11])
            mi = self.mutual_information(winter_anomaly, dump=dump)
        else:
            mi = self.mutual_information(self.data.anomaly(), dump=dump)
        self._similarity_measure = mi

    def set_winter_only(self, winter_only, dump=True):
        """
        Toggle use of exclusively winter data points for network generation.

        Also explicitly regenerates the instance of MutualInfoClimateNetwork.

        :arg bool winter_only: Indicates whether only winter months were used
            for network generation.
        :arg bool dump: Store MI in data file.
        """
        self._set_winter_only(winter_only, dump=dump)
        self._regenerate_network()

    #
    #  Defines methods to calculate  weighted network measures
    #

    def mutual_information_weighted_average_path_length(self):
        """
        Return mutual information weighted average path length.

        :return float: the mutual information weighted average path length.
        """
        return self._weighted_metric(
            "mutual_information",
            lambda: np.abs(self.mutual_information()),
            "average_path_length")

    def mutual_information_weighted_closeness(self):
        """
        Return mutual information weighted closeness.

        :rtype: 1D Numpy array [index]
        :return: the mutual information weighted closeness sequence.
        """
        return self._weighted_metric(
            "mutual_information",
            lambda: np.abs(self.mutual_information()),
            "closeness")

    def local_mutual_information_weighted_vulnerability(self):
        """
        Return mutual information weighted vulnerability.

        :rtype: 1D Numpy array [index]
        :return: the mutual information weighted vulnerability sequence.
        """
        return self._weighted_metric(
            "mutual_information",
            lambda: np.abs(self.mutual_information()),
            "local_vulnerability")
