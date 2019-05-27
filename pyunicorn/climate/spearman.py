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

#  Import cnTsonisClimateNetwork for TsonisClimateNetwork class
from .tsonis import TsonisClimateNetwork


#
#  Define class TsonisClimateNetwork
#

class SpearmanClimateNetwork(TsonisClimateNetwork):

    """
    Encapsulate a Spearman climate network.

    The Spearman climate network is constructed from the Spearman rank order
    correlation matrix (Spearman's rho). Spearman's rho is more robust with
    respect to ouliers and non-gaussian data distributions than the Pearson
    correlation coefficient used in :class:`TsonisClimateNetwork`.

    Hence, Spearman climate networks are undirected due to the symmetry of the
    Spearman's rho matrix.
    """

    #
    #  Defines internal methods
    #

    def __init__(self, data, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface", winter_only=True,
                 silence_level=0):
        """
        Initialize an instance of :class:`SpearmanClimateNetwork`.

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
            print("Generating a Spearman climate network...")

        #  Call constructor of parent class TsonisClimateNetwork
        TsonisClimateNetwork.__init__(self, data=data, threshold=threshold,
                                      link_density=link_density,
                                      non_local=non_local,
                                      node_weight_type=node_weight_type,
                                      winter_only=winter_only,
                                      silence_level=silence_level)

    def __str__(self):
        """
        Returns a string representation of SpearmanClimateNetwork.
        """
        return ('SpearmanClimateNetwork:\n'
                f'{TsonisClimateNetwork.__str__(self)}')

    #
    #  Defines methods to calculate the correlation matrix
    #

    @staticmethod
    def rank_time_series(anomaly):
        """
        Return rank time series.

        Ranks are generated individually for each time series.

        :type anomaly: 2D Numpy array [time, index]
        :arg anomaly: The anomaly time series to be converted into ranks.

        :rtype: 2D Numpy array [time, index]
        :return: the rank time series.
        """
        #  Obtain rank time series
        rank_time_series = anomaly.argsort(axis=0).argsort(axis=0)

        return rank_time_series

    def _calculate_correlation(self, anomaly):
        """
        Return Spearman's rho matrix at zero lag.

        :type anomaly: 2D Numpy array (time, index)
        :arg anomaly: the anomaly time series from to calculate the correlation
                     matrix at zero lag.

        :rtype: 2D Numpy array (index, index)
        :return: the Spearman's rho matrix at zero lag.
        """
        if self.silence_level <= 1:
            print("Calculating Spearman Rho matrix at zero lag from anomaly "
                  "values...")

        #  Convert anomaly time series to time series of ranks
        ranks = self.rank_time_series(anomaly)

        #  Cast to float32 type to save memory since correlation coefficients
        #  are not needed in high floating point precision.
        spearman_rho = np.corrcoef(ranks.transpose()).astype("float32")

        return spearman_rho
