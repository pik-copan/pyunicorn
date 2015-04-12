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

#  array object and fast numerics
import numpy as np
#  C++ inline code
import weave

#  Import cnTsonisClimateNetwork for TsonisClimateNetwork class
from .climate_network import ClimateNetwork


#
#  Define class RainfallClimateNetwork
#

class RainfallClimateNetwork(ClimateNetwork):

    """
    Encapsulate a Rainfall climate network.

    The Rainfall climate network is constructed from the Spearman rank order
    correlation matrix (Spearman's rho) but without considering "zeros" in the
    dataset, which represent the time at which there is no rainfall.
    Spearman's rho is more robust with respect to outliers and non-gaussian
    data distributions than the Pearson correlation coefficient.

    Rainfall climate networks are undirected due to the symmetry of the
    Spearman's rho matrix.

    Derived from the class ClimateNetwork.

    Class RainfallClimateNetwork was created by `Marc Wiedermann
    <marcw@physik.hu-berlin.de>`__ during an internship
    at PIK in March 2010.
    """

    #
    # Defines internal methods
    #

    def __init__(self, data, threshold=None, link_density=None,
                 non_local=False, node_weight_type="surface",
                 event_threshold=[0, 1], scale_fac=37265, offset=10**(-7),
                 silence_level=0):
        """
        Initialize an instance of RainfallClimateNetwork.

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
        :type event_threshold: list of two numbers between 0 and 1.
        :arg event_threshold: The quantiles of the rainfall distribution at
            each location between which rainfall events should be considered
            for calculating correlations.
        :arg float scale_fac: Scale factor for rescaling data.
        :arg float offset: Offset for rescaling data.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print "Generating a Rainfall climate network..."

        #  Set instance variables
        self.data = data
        """(ClimateData) - The climate data used for network construction."""

        self.N = self.data.grid.N
        self._threshold = threshold
        self._prescribed_link_density = link_density
        self._non_local = non_local
        self.node_weight_type = node_weight_type
        self.silence_level = silence_level
        self.time_cycle = self.data.time_cycle

        #  Calculate correlation measure
        correlation = self._calculate_correlation(event_threshold,
                                                  scale_fac, offset,
                                                  time_cycle=self.time_cycle)

        #  Call the constructor of the parent class ClimateNetwork
        ClimateNetwork.__init__(self, grid=self.data.grid,
                                similarity_measure=correlation,
                                threshold=self.threshold(),
                                link_density=self._prescribed_link_density,
                                non_local=self.non_local(),
                                directed=False,
                                node_weight_type=self.node_weight_type,
                                silence_level=self.silence_level)

    #
    # Defines methods to calculate the correlation matrix
    #

    def _calculate_correlation(self, event_threshold, scale_fac, offset,
                               time_cycle):
        """
        Returns the Spearman Rho correlation matrix.

        An event_threshold can be given to extract a percentage of the given
        dataset, i.e. [0.9,1] extracts the ten percent of heaviest rainfall
        events. [0,1] selects the whole dataset.

        :type event_threshold: list of two numbers between 0 and 1.
        :arg event_threshold: The quantiles of the rainfall distribution at
                              each location between which rainfall events
                              should be considered for calculating
                              correlations.

        :type scale_fac: number (float)
        :arg scale_fac: Scale factor for rescaling data.

        :type offset: number (float)
        :arg offset: Offset for rescaling data.

        :type time_cycle: number (int)
        :arg time_cycle: Length of annual cycle in given data (monthly: 12,
                         daily: 365 etc.)

        :rtype: 2D Numpy array (index, index)
        :return: the Spearman's rho matrix at zero lag.
        """
        # !!!Notice the all calculations are done with the transposed dataset!!
        observable = self.data.observable()
        observable = observable.transpose()

        # Calculate the real rainfall from observable
        rainfall = self.calculate_rainfall(observable, scale_fac, offset)

        if self.silence_level <= 1:
            print "Calculating Rainfall-Anomaly using Weave..."

        # Calculate the anomaly for the rainfall dataset
        # anomaly = self.calculate_rainfall_anomaly(rainfall, time_cycle)
        anomaly = self.calculate_rainfall(self.data.anomaly(), scale_fac,
                                          offset)

        # Correct anomaly for offset due to rescaling
        anomaly -= scale_fac * offset

        # Create a mask, which filters out zeros and events outside the
        # event_threshold
        final_mask = self.calculate_top_events(rainfall, event_threshold)

        if self.silence_level <= 1:
            print "Calculating Spearman-Rho-Matrix using Weave..."

        # Return the correlation matrix
        return self.calculate_corr(final_mask, anomaly)

    def calculate_rainfall(self, observable, scale_fac, offset):
        """
        Returns the rainfall in mm on each measuring point.

        :type observable: 2D Numpy array (time, index)
        :arg observable: The observable time series from the data source.

        :type scale_fac: number (float)
        :arg scale_fac: Scale factor for rescaling data.

        :type offset: number (float)
        :arg offset: Offset for rescaling data.

        :rtype: 2D Numpy array (time, index)
        :return: the rainfall for each time and location
        """
        # Multiply the observable with the scaling factor after adding the
        # offset
        rainfall = (observable + offset) * scale_fac

        return rainfall

    def calculate_top_events(self, rainfall, event_threshold):
        """
        Returns a mask with boolean values. The entries are false, when the
        rainfall of one day is zero, or when the rainfall is not inside the
        event_treshold

        :type rainfall: 2D Numpy array (index, time)
        :arg rainfall: the rainfall time series for each measuring point

        :type event_threshold: list of two numbers between 0 and 1.
        :arg event_threshold: The quantiles of the rainfall distribution at
                              each location between which rainfall events
                              should be considered for calculating
                              correlations.

        :rtype: 2D Numpy array (index, time)
        :return: A bool array with False for every value in the rainfall
                 data, which are zero or outside the top_event Interval.
        """
        rainfall_copy = rainfall.copy()

        m = len(rainfall) * len(rainfall.transpose())

        onelist = rainfall.reshape(m)

        onelist = onelist[onelist.sort()][0]

        downlimit = m * event_threshold[0] // 1

        uplimit = m * event_threshold[1] // 1

        rainfall = rainfall_copy

        down_mask = rainfall >= onelist[downlimit]

        up_mask = rainfall <= onelist[uplimit-1]

        no_rain_mask = (rainfall != 0)

        final_mask = down_mask & up_mask & no_rain_mask

        return final_mask

    def rank_time_series(self, anomaly):
        """
        Return rank time series.

        :type anomaly: 2D Numpy array (index, time)
        :arg anomaly: the rainfall anomaly time series for each measuring point

        :rtype: 2D Numpy array (index, time)
        :return: The ranked time series for each gridpoint
        """
        rank_time_series = anomaly.argsort(axis=1).argsort(axis=1) + 1.0
        return rank_time_series

    def calculate_corr(self, final_mask, anomaly):
        """
        Return the Spearman Correlation Matrix at zero lag.

        :type final_mask: 2D Numpy array (index, time)
        :arg final_mask: A bool array with False for every value in the
                         rainfall data, which are zero or outside the top_event
                         interval.

        :type anomaly: 2D Numpy array (index, time)
        :arg anomaly: The rainfall anomaly time series for each measuring
                      point.

        :rtype: 2D Numpy array (index, index)
        :return: the Spearman correlation matrix.
        """
        # Get rank time series
        time_series_ranked = self.rank_time_series(anomaly)

        m = len(anomaly)

        tmax = len(anomaly.transpose())

        spearman_rho = np.zeros((m, m))

        m = int(m)

        tmax = int(tmax)

        # Calculate the correlation matrix using Weave
        code = """
        double cov = 0;
        double sigmai = 0;
        double sigmaj = 0;
        double meani = 0;
        double meanj = 0;
        int zerocount = 0;
        double rankedi[tmax];
        double rankedj[tmax];
        double normalizedi[tmax];
        double normalizedj[tmax];

        for (int i=0; i<m; i++) {
            for (int j=i; j<m; j++) {

                for (int t=0; t<tmax; t++) {
                    if ((final_mask(i,t) | final_mask(j,t)) == false)
                        zerocount = zerocount+1;
                }

                for (int t=0; t<tmax; t++) {
                    rankedi[t] = time_series_ranked(i,t) - zerocount;
                    rankedj[t] = time_series_ranked(j,t) - zerocount;
                }

                for (int t=0; t<tmax; t++) {
                    if (rankedi[t]>=0)
                        meani = meani + rankedi[t];
                    if (rankedj[t]>=0)
                        meanj = meanj + rankedj[t];
                }

                meani = meani/(tmax-zerocount);
                meanj = meanj/(tmax-zerocount);

                for (int t=0; t<tmax; t++) {
                    if ((final_mask(i,t) | final_mask(j,t)) == true) {
                        normalizedi[t] = rankedi[t] - meani;
                        normalizedj[t] = rankedj[t] - meanj;
                        cov = cov + normalizedi[t]*normalizedj[t];
                        sigmai = sigmai + normalizedi[t]*normalizedi[t];
                        sigmaj = sigmaj + normalizedj[t]*normalizedj[t];
                    }
                }
                spearman_rho(i,j)=spearman_rho(j,i) = cov/sqrt(sigmai*sigmaj);
                meani = 0;
                meanj = 0;
                cov = 0;
                sigmai = 0;
                sigmaj = 0;
                zerocount = 0;

            }
        }
        """
        args = ['m', 'tmax', 'spearman_rho', 'final_mask',
                'time_series_ranked']
        weave.inline(code, arg_names=args,
                     type_converters=weave.converters.blitz, compiler='gcc',
                     extra_compile_args=["-O3"])
        return spearman_rho
