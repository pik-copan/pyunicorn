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
Provides class for the analysis of dynamical systems and time series based
on event synchronization
"""

# array object and fast numerics
import numpy as np

from ..funcnet import EventSynchronization
from .climate_network import ClimateNetwork
from .climate_data import ClimateData
from ..core import Data


#
#  Class definitions
#

class EventSynchronizationClimateNetwork(EventSynchronization, ClimateNetwork):
    """
    Class EventSynchronizationClimateNetwork for generating and quantitatively
    analyzing event synchronization networks.

    References: [Boers2014]_.
    """

    #
    #  Internal methods
    #

    def __init__(self, data, quantile_threshold, taumax,
                 eventsynctype="directedES", non_local=False,
                 node_weight_type="surface", silence_level=0):
        """
        Initialize an instance of EventSynchronizationClimateNetwork.

        For other applications of event synchronization networks please use
        the event synchronization class directly.

        :type data: :classL`..climate.ClimateData`
        :arg data: The climate data used for network construction.
        :type quantile_threshold: float between 0 and 1
        :arg quantile_threshold: values above will be treated as events
        :arg int taumax: Maximum dynamical delay
        :type eventsynctype: str
        :arg eventsynctype: one of "directedES", "symmetricES" or
            "antisymmetricES" [default: "directed"]
        :arg bool non_local: Determines, whether links between spatially close
            nodes should be suppressed.
        :arg str node_weight_type: The type of geographical node weight to be
            used.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        etypes = ["directedES", "symmetricES", "antisymmetricES"]
        if eventsynctype not in etypes:
            raise IOError("wrong eventsynctype...\n \
                          Available options: '%s', '%s' or '%s'" %
                          (etypes[0], etypes[1], etypes[2]))

        self.__eventsynctype = eventsynctype
        self.directed = self.__eventsynctype != "symmetricES"

        eventmatrix = data.observable() > np.percentile(data.observable(),
                                                        quantile_threshold*100,
                                                        axis=0)
        EventSynchronization.__init__(self, eventmatrix.astype(int), taumax)

        eventsyncmatrix = getattr(self, self.__eventsynctype)()
        ClimateNetwork.__init__(self, grid=data.grid,
                                similarity_measure=eventsyncmatrix,
                                threshold=0,
                                non_local=non_local,
                                directed=self.directed,
                                node_weight_type=node_weight_type,
                                silence_level=silence_level)

    def __str__(self):
        """
        Return a string representation of TsonisClimateNetwork.

        **Example:**

        >>> data = EventSynchronizationClimateNetwork.SmallTestData()
        >>> print(EventSynchronizationClimateNetwork(data, 0.8, 16))
        Extracting network adjacency matrix by thresholding...
        Setting area weights according to type surface...
        Setting area weights according to type surface...
        EventSynchronizationClimateNetwork:
        EventSynchronization: 6 variables, 10 timesteps, taumax: 16
        ClimateNetwork:
        GeoNetwork:
        Network: directed, 6 nodes, 0 links, link density 0.000.
        Geographical boundaries:
                 time     lat     lon
           min    0.0    0.00    2.50
           max    9.0   25.00   15.00
        Threshold: 0
        Local connections filtered out: False
        Type of event synchronization to construct the network: directedES
        """
        text = ("EventSynchronizationClimateNetwork:\n%s\n%s\n"
                "Type of event synchronization to construct the network: %s")
        return text % (EventSynchronization.__str__(self),
                       ClimateNetwork.__str__(self), self.__eventsynctype)

    @staticmethod
    def SmallTestData():
        """
        Return test data set of 6 time series with 10 sampling points each.

        **Example:**

        >>> r(Data.SmallTestData().observable())
        array([[ 0.    ,  1.    ,  0.    , -1.    , -0.    ,  1.    ],
               [ 0.309 ,  0.9511, -0.309 , -0.9511,  0.309 ,  0.9511],
               [ 0.5878,  0.809 , -0.5878, -0.809 ,  0.5878,  0.809 ],
               [ 0.809 ,  0.5878, -0.809 , -0.5878,  0.809 ,  0.5878],
               [ 0.9511,  0.309 , -0.9511, -0.309 ,  0.9511,  0.309 ],
               [ 1.    ,  0.    , -1.    , -0.    ,  1.    ,  0.    ],
               [ 0.9511, -0.309 , -0.9511,  0.309 ,  0.9511, -0.309 ],
               [ 0.809 , -0.5878, -0.809 ,  0.5878,  0.809 , -0.5878],
               [ 0.5878, -0.809 , -0.5878,  0.809 ,  0.5878, -0.809 ],
               [ 0.309 , -0.9511, -0.309 ,  0.9511,  0.309 , -0.9511]])

        :rtype: ClimateData instance
        :return: a ClimateData instance for testing purposes.
        """
        data = Data.SmallTestData()

        return ClimateData(observable=data.observable(), grid=data.grid,
                           time_cycle=5, silence_level=2)
