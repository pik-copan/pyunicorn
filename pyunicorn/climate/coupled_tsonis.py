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
Provides classes for generating and analyzing complex coupled climate networks.
"""

#
#  Import essential packages
#

#  Import NumPy for the array object and fast numerics
import numpy as np

#  Import climate_network for CoupledClimateNetwork class
from .coupled_climate_network import CoupledClimateNetwork


#
#  Define class CoupledClimateNetwork
#

class CoupledTsonisClimateNetwork(CoupledClimateNetwork):

    """
    Encapsulates a coupled similarity network embedded on a spherical surface.

    Particularly provides functionality to generate a complex network from the
    Pearson correlation matrix of time series from two different
    observables (temperature, pressure), vertical levels etc.

    Construct a static climate network following Tsonis et al. from the
    Pearson correlation matrix at zero lag [Tsonis2004]_.

    Hence, coupled Tsonis climate networks are undirected due to the symmetry
    of the correlation matrix.

    The idea of coupled climate networks is based on the concept of coupled
    patterns, for a review refer to [Bretherton1992]_.

    .. note::
       The two observables (layers) need to have the same time grid \
       (temporal sampling points).
    """
    #
    #  Definitions of internal methods
    #

    def __init__(self, data_1, data_2, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface",
                 selected_months=None, silence_level=0):
        """
        Initialize an instance of :class:`CoupledTsonisClimateNetwork`.

        .. note::
           Either threshold **OR** link_density have to be given!

        Possible choices for ``node_weight_type``:
          - ``None`` (constant unit weights)
          - ``"surface"`` (cos lat)
          - ``"irrigation"`` (cos**2 lat)

        :type data_1: :class:`.ClimateData`
        :arg data_1: The climate data for the first layer.
        :type data_2: :class:`.ClimateData`
        :arg data_2: The climate data for the second layer.
        :arg float threshold: The threshold of similarity measure, above which
            two nodes are linked in the network.
        :arg float link_density: The networks's desired link density.
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg [int ]selected_months: The months for which to calculate the
            correlation matrix. The full time series are used for default value
            None.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        self.silence_level = silence_level
        """(string) - The inverse level of verbosity of the object."""

        n_time_1 = data_1.grid.grid_size()["time"]
        n_time_2 = data_2.grid.grid_size()["time"]

        #  The time series from both observables have to have the same length
        if n_time_1 == n_time_2:
            #  Prepare
            if selected_months is None:
                anomaly_1 = data_1.anomaly()
                anomaly_2 = data_2.anomaly()
            else:
                anomaly_1 = data_1.anomaly_selected_months(selected_months)
                anomaly_2 = data_2.anomaly_selected_months(selected_months)

            correlation = self._calculate_correlation(anomaly_1=anomaly_1,
                                                      anomaly_2=anomaly_2)

            #  Call the constructor of the parent class ClimateNetwork
            CoupledClimateNetwork.__init__(self, grid_1=data_1.grid,
                                           grid_2=data_2.grid,
                                           similarity_measure=correlation,
                                           threshold=threshold,
                                           link_density=link_density,
                                           non_local=non_local,
                                           directed=False,
                                           node_weight_type=node_weight_type,
                                           silence_level=silence_level)
        else:
            print("\nThe two observables (layers) have to have the same "
                  "number of temporal sampling points!\n")

    def __str__(self):
        """
        Return a string representation of CoupledClimateNetwork object.
        """
        return ('CoupledTsonisClimateNetwork:\n'
                f'{CoupledClimateNetwork.__str__(self)}')

    #
    #  Defines methods to calculate the correlation matrix
    #

    def _calculate_correlation(self, anomaly_1, anomaly_2):
        """
        Return the correlation matrix at zero lag.

        :type anomaly_1: 2D Numpy array (time, index_1)
        :arg anomaly_1: the first set of anomaly time series from which to
                        calculate the correlation matrix at zero lag.

        :type anomaly_2: 2D Numpy array (time, index_2)
        :arg anomaly_2: the second set of anomaly time series from which to
                        calculate the correlation matrix at zero lag.

        :rtype: 2D Numpy array (index, index)
        :return: the correlation matrix at zero lag.
        """
        if self.silence_level <= 1:
            print("Calculating correlation matrix at zero lag from anomaly "
                  "values...")

        anomaly = np.concatenate((anomaly_1, anomaly_2), axis=1)

        #  Cast to float32 type to save memory since correlation coefficients
        #  are not needed in high floating point precision.
        correlation = np.corrcoef(anomaly.transpose()).astype("float32")

        return correlation

    def calculate_similarity_measure(self, anomaly_1, anomaly_2):
        """
        Encapsulate the calculation of the correlation matrix at zero lag.

        :type anomaly_1: 2D Numpy array (time, index_1)
        :arg anomaly_1: the first set of anomaly time series from which to
                        calculate the correlation matrix at zero lag.

        :type anomaly_2: 2D Numpy array (time, index_2)
        :arg anomaly_2: the second set of anomaly time series from which to
                        calculate the correlation matrix at zero lag.

        :rtype: 2D Numpy array (index, index)
        :return: the correlation matrix at zero lag.
        """
        return self._calculate_correlation(anomaly_1, anomaly_2)

    def correlation(self):
        """
        Return the coupled correlation matrix at zero lag.

        :rtype: 2D Numpy array (index, index)
        :return: the correlation matrix at zero lag.
        """
        return self.similarity_measure()
