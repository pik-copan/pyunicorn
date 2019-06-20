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
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.

Written by Jakob Runge.
CMSI Method Reference: [Pompe2011]_
"""

# array object and fast numerics
import numpy


#
#  Define class CouplingAnalysisPurePython
#

class CouplingAnalysisPurePython:

    """
    Contains methods to calculate coupling matrices from large arrays
    of scalar time series.

    Comprises linear and information theoretic measures, lagged
    and directed (causal) couplings.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, dataarray, only_tri=False, silence_level=0):
        """
        Initialize an instance of CouplingAnalysisPurePython.

        Possible choices for only_tri:
          - "True" will calculate only the upper triangle of the coupling
            matrix, excluding the diagonal, assuming symmetry (not for directed
            measures)
          - "False" will calculate the whole matrix (asymmetry somes from
             different integration ranges)

        :type dataarray: 4D, 3D or 2D Numpy array [time, index, index] or
                         [time, index]
        :arg dataarray: The time series array with time in first dimension
        :arg bool only_tri: Symmetric/asymmetric assumption on coupling matrix.
        :arg int silence_level: The inverse level of verbosity of the object.
        """

        #  only_tri will calculate the upper triangle excluding the diagonal
        #  only. This assumes stationarity on the time series
        self.only_tri = only_tri

        #  Set silence level
        self.silence_level = silence_level

        #  Flatten observable anomaly array along lon/lat dimension to allow
        #  for more convinient indexing and transpose the whole array as this
        #  is faster in loops
        if numpy.ndim(dataarray) == 4:
            (self.total_time, n_lev, n_lat, n_lon) = dataarray.shape
            self.N = n_lev * n_lat * n_lon
            self.dataarray = numpy.\
                fastCopyAndTranspose(dataarray.reshape(-1, self.N))
        if numpy.ndim(dataarray) == 3:
            (self.total_time, n_lat, n_lon) = dataarray.shape
            self.N = n_lat * n_lon
            self.dataarray = numpy.\
                fastCopyAndTranspose(dataarray.reshape(-1, self.N))

        elif numpy.ndim(dataarray) == 2:
            (self.total_time, self.N) = dataarray.shape
            self.dataarray = numpy.fastCopyAndTranspose(dataarray)

        else:
            print("irregular array shape...")
            self.dataarray = numpy.fastCopyAndTranspose(dataarray)

        #  factorials below 10 in a list for permutation patterns
        self.factorial = \
            numpy.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])
        self.patternized = False
        self.has_fft = False
        self.originalFFT = None

        #  lag_mode dict
        self.lag_modi = {"all": 0, "sum": 1, "max": 2}

    def __str__(self):
        """
        Return a string representation of the CouplingAnalysisPurePython
        object.
        """
        shape = self.dataarray.shape
        return 'CouplingAnalysisPurePython: %i variables, %i timesteps.' % (
            shape[0], shape[1])

    #
    #  Define methods to calculate correlation strength and lags
    #

    #
    #  Routines for calculating Cross Correlation
    #

    def cross_correlation(self, tau_max=0, lag_mode='all'):
        """
        Returns the normalized cross correlation from all pairs of nodes from
        a range of time lags.

        The calculation ranges are shown below::

            (-------------------------total_time--------------------------)
            (---tau_max---)(---------corr_range------------)(---tau_max---)

        CC is calculated about corr_range and with the other time series
        shifted by tau

        Possible choices for lag_mode:

        - "all" will return the full function for all lags, possible large
          memory need if only_tri is True, only the upper triangle contains the
          values, the lower one is zeros
        - "sum" will return the sum over positive and negative lags seperatly,
          each inclunding tau=0 corrmat[0] is the positive sum, corrmat[1] the
          negative sum
        - "max" will return only the maximum coupling (in corrmat[0]) and its
          lag (in corrmat[1])

        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: the output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """
        #  Normalize anomaly time series to zero mean and unit variance for all
        #  lags, array contains normalizations for all lags
        corr_range = self.total_time - 2*tau_max
        normalized_array = numpy.empty((2*tau_max + 1, self.N, corr_range),
                                       dtype="float32")

        for t in range(2*tau_max + 1):
            #  Remove mean value from time series at each vertex (grid point)
            normalized_array[t] = self.dataarray[:, t:t+corr_range] - \
                self.dataarray[:, t:t+corr_range].\
                mean(axis=1).reshape(self.N, 1)

            #  Normalize the variance of anomalies to one
            normalized_array[t] /= normalized_array[t].\
                std(axis=1).reshape(self.N, 1)

            #  Correct for grid points with zero variance in their time series
            normalized_array[t][numpy.isnan(normalized_array[t])] = 0

        return self._calculate_cc(normalized_array, corr_range=corr_range,
                                  tau_max=tau_max, lag_mode=lag_mode)

    def shuffled_surrogate_for_cc(self, fourier=False, tau_max=1,
                                  lag_mode='all'):
        """
        Returns a correlation matrix calculated with an independently shuffled
        surrogate of the dataarray of length corr_range for all taus.

        :arg int corr_range: length of sample
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """
        corr_range = self.total_time - 2*tau_max

        # Shuffle a copy of dataarray separatly for each node
        array = numpy.copy(self.dataarray)
        if fourier:
            array = self.correlatedNoiseSurrogates(array)
        else:
            for i in range(self.N):
                numpy.random.shuffle(array[i])

        sample_array = numpy.zeros((1, self.N, corr_range), dtype="float32")

        sample_array[0] = array[:, :corr_range]
        sample_array[0] -= sample_array[0].mean(axis=1).reshape(self.N, 1)
        sample_array[0] /= sample_array[0].std(axis=1).reshape(self.N, 1)
        sample_array[0, numpy.isnan(sample_array[0])] = 0

        res = self._calculate_cc(sample_array, corr_range=corr_range,
                                 tau_max=0, lag_mode='all')

        if lag_mode == 'all':
            corrmat = numpy.repeat(res, 2*tau_max + 1, axis=0)
        elif lag_mode == 'sum':
            corrmat = numpy.array([abs(res[0]), abs(res[0])]) * (tau_max+1.)
        elif lag_mode == 'max':
            corrmat = numpy.array([abs(res[0]),
                                   numpy.random.randint(-tau_max, tau_max+1,
                                                        (self.N, self.N))])

        return corrmat

    def time_surrogate_for_cc(self, sample_range=100, tau_max=1,
                              lag_mode='all'):
        """
        Returns a joint shuffled surrogate of the full dataarray of length
        sample_range for all taus.

        Used for time evolution analysis. First one initializes the
        CouplingAnalysis class with the full dataarray and then this function
        is called for every single surrogate.

        :arg int sample_range: length of sample
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """

        perm = numpy.random.permutation(
            range(tau_max, self.total_time - tau_max))[:sample_range]

        sample_array = numpy.empty((2*tau_max + 1, self.N, sample_range),
                                   dtype="float32")

        for t in range(2 * tau_max + 1):
            tau = t - tau_max
            sample_array[t] = self.dataarray[:, perm + tau]
            sample_array[t] -= sample_array[t].mean(axis=1).reshape(self.N, 1)
            sample_array[t] /= sample_array[t].std(axis=1).reshape(self.N, 1)
            sample_array[t][numpy.isnan(sample_array[t])] = 0

        return self._calculate_cc(sample_array, corr_range=sample_range,
                                  tau_max=tau_max, lag_mode=lag_mode)

    def _calculate_cc(self, array, corr_range, tau_max, lag_mode):
        """
        Returns the CC matrix.

        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices

        ## lag_mode dict
        mode = self.lag_modi[lag_mode]
        """

        # lag_mode dict
        mode = self.lag_modi[lag_mode]
        only_tri = int(self.only_tri)

        if lag_mode == 'all':
            corrmat = numpy.zeros((2*tau_max + 1, self.N, self.N),
                                  dtype='float32')
        elif lag_mode == 'sum':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')
        elif lag_mode == 'max':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')

        # loop over all node pairs, NOT symmetric due to time shifts!
        for i in range(self.N-only_tri):
            for j in range((i+1)*only_tri, self.N):

                if mode == 2:
                    maxcross = 0.0
                    argmax = 0

                # loop over taus INCLUDING the last tau value
                for t in range(2*tau_max+1):

                    # here the actual cross correlation is calculated
                    crossij = (array[tau_max, i, :] * array[t, j, :]).mean()

                    # fill in values in matrix depending on lag_mode
                    if mode == 0:
                        corrmat[t, i, j] = crossij

                    elif mode == 1:
                        if t <= tau_max:
                            corrmat[1, i, j] += numpy.abs(crossij)
                        if t >= tau_max:
                            corrmat[0, i, j] += numpy.abs(crossij)

                    elif mode == 2:
                        # calculate max and argmax by comparing to previous
                        # value and storing max
                        if numpy.abs(crossij) > maxcross:
                            maxcross = numpy.abs(crossij)
                            argmax = t

                if mode == 2:
                    corrmat[0, i, j] = maxcross
                    corrmat[1, i, j] = argmax - tau_max

        if self.only_tri:
            if lag_mode == 'all':
                corrmat = corrmat + corrmat.transpose(0, 2, 1)[::-1]
            elif lag_mode == 'sum':
                corrmat[0] += corrmat[1].transpose()
                corrmat[1] = corrmat[0].transpose()
            elif lag_mode == 'max':
                corrmat[0] += corrmat[0].transpose()
                corrmat[1] -= corrmat[1].transpose()

        return corrmat

    #
    #  Routines for calculating Mutual Information with adaptive bins
    #

    def mutual_information(self, bins=16, tau_max=0, lag_mode='all'):
        """
        Returns the normalized mutual information from all pairs of nodes from
        a range of time lags.

        MI = H_x + H_y - H_xy

        Uses adaptive bins, where each marginal bin contains the same number of
        samples. Then the marginal entropies have equal probable distributions
        H_x = H_y = log(bins)

        The calculation ranges are shown below::

            (-------------------------total_time--------------------------)
            (---tau_max---)(---------corr_range------------)(---tau_max---)

        MI is calculated about corr_range and with the other time series
        shifted by tau

        Possible choices for lag_mode:

        - "all" will return the full function for all lags, possible large
          memory need if only_tri is True, only the upper triangle contains the
          values, the lower one is zeros
        - "sum" will return the sum over positive and negative lags seperatly,
          each inclunding tau=0 corrmat[0] is the positive sum, corrmat[1] the
          negative sum
        - "max" will return only the maximum coupling (in corrmat[0]) and its
          lag (in corrmat[1])

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """
        if bins < 255:
            dtype = 'uint8'
        else:
            dtype = 'int16'

        # Normalize anomaly time series to zero mean and unit variance for all
        # lags, array contains normalizations for all lags
        corr_range = self.total_time - 2*tau_max

        # get the bin quantile steps
        bin_edge = numpy.ceil(corr_range/float(bins))

        symbolic_array = numpy.empty((2*tau_max + 1, self.N, corr_range),
                                     dtype=dtype)

        for t in range(2*tau_max + 1):

            array = self.dataarray[:, t:t+corr_range]

            # get the lower edges of the bins for every time series
            edges = numpy.sort(array, axis=1)[:, ::bin_edge]
            bins = edges.shape[1]

            # This gives the symbolic time series
            symbolic_array[t] = \
                (array.reshape(self.N, corr_range, 1)
                 >= edges.reshape(self.N, 1, bins)).sum(axis=2) - 1

        return self._calculate_mi(symbolic_array, corr_range=corr_range,
                                  bins=bins, tau_max=tau_max,
                                  lag_mode=lag_mode)

    def mutual_information_edges(self, bins=16, tau=0, lag_mode='all'):
        """
        Returns the normalized mutual information from all pairs of nodes from
        a range of time lags.

        MI = H_x + H_y - H_xy

        Uses adaptive bins, where each marginal bin contains the same number of
        samples. Then the marginal entropies have equal probable distributions
        H_x = H_y = log(bins)

        The calculation ranges are shown below::

            (-------------------------total_time--------------------------)
            (---tau_max---)(---------corr_range------------)(---tau_max---)

        MI is calculated about corr_range and with the other time series
        shifted by tau

        Possible choices for lag_mode:

        - "all" will return the full function for all lags, possible large
          memory need if only_tri is True, only the upper triangle contains the
          values, the lower one is zeros
        - "sum" will return the sum over positive and negative lags seperatly,
          each inclunding tau=0 corrmat[0] is the positive sum, corrmat[1] the
          negative sum
        - "max" will return only the maximum coupling (in corrmat[0]) and its
          lag (in corrmat[1])

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 2D numpy array (float) [index, index]
        :return: bin edges for zero lag
        """

        # get the bin quantile steps
        bin_edge = numpy.ceil(self.total_time/float(bins))

        array = self.dataarray[:, :]
        array[:-tau, 1] = array[tau, 1]

        # get the lower edges of the bins for every time series
        edges = numpy.sort(array, axis=1)[:, ::bin_edge]
        bins = edges.shape[1]

        return edges

    def shuffled_surrogate_for_mi(self, fourier=False, bins=16, tau_max=0,
                                  lag_mode='all'):
        """
        Returns a shuffled surrogate of normalized mutual information from all
        pairs of nodes from a range of time lags.

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """
        if bins < 255:
            dtype = 'uint8'
        else:
            dtype = 'int16'

        #  Normalize anomaly time series to zero mean and unit variance for all
        #  lags, array contains normalizations for all lags
        corr_range = self.total_time - 2*tau_max

        # Shuffle a copy of dataarray seperatly for each node
        array = numpy.copy(self.dataarray)
        if fourier:
            array = self.correlatedNoiseSurrogates(array)
        else:
            for i in range(self.N):
                numpy.random.shuffle(array[i])

        # get the bin quantile steps
        bin_edge = numpy.ceil(corr_range/float(bins))

        symbolic_array = numpy.empty((1, self.N, corr_range), dtype=dtype)

        array = array[:, :corr_range]

        # get the lower edges of the bins for every time series
        edges = numpy.sort(array, axis=1)[:, ::bin_edge]
        bins = edges.shape[1]

        # This gives the symbolic time series
        symbolic_array[0] = \
            (array.reshape(self.N, corr_range, 1)
             >= edges.reshape(self.N, 1, bins)).sum(axis=2) - 1

        res = self._calculate_mi(symbolic_array, corr_range=corr_range,
                                 bins=bins, tau_max=0, lag_mode='all')

        if lag_mode == 'all':
            corrmat = numpy.repeat(res, 2*tau_max + 1, axis=0)
        elif lag_mode == 'sum':
            corrmat = numpy.array([res[0], res[0]]) * (tau_max+1.)
        elif lag_mode == 'max':
            corrmat = numpy.array(
                [res[0], numpy.random.randint(-tau_max, tau_max+1,
                                              (self.N, self.N))])

        return corrmat

    def time_surrogate_for_mi(self, bins=16, sample_range=100, tau_max=1,
                              lag_mode='all'):
        """
        Returns a joint shuffled surrogate of the full dataarray of length
        sample_range for all taus.

        Used for time evolution analysis. First one initializes the
        CouplingAnalysis class with the full dataarray and then this function
        is called for every single surrogate.

        :arg int sample_range: length of sample
        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """

        if bins < 255:
            dtype = 'uint8'
        else:
            dtype = 'int16'

        perm = numpy.random.permutation(
            range(tau_max, self.total_time - tau_max))[:sample_range]

        # get the bin quantile steps
        bin_edge = numpy.ceil(sample_range/float(bins))

        symbolic_array = numpy.empty((2*tau_max + 1, self.N, sample_range),
                                     dtype=dtype)

        for t in range(2*tau_max + 1):
            tau = t - tau_max

            array = self.dataarray[:, perm + tau]

            # get the lower edges of the bins for every time series
            edges = numpy.sort(array, axis=1)[:, ::bin_edge]
            bins = edges.shape[1]

            # This gives the symbolic time series
            symbolic_array[t] = \
                (array.reshape(self.N, sample_range, 1)
                 >= edges.reshape(self.N, 1, bins)).sum(axis=2) - 1

        return self._calculate_mi(symbolic_array, corr_range=sample_range,
                                  bins=bins, tau_max=tau_max,
                                  lag_mode=lag_mode)

    def _calculate_mi(self, array, corr_range, bins, tau_max, lag_mode):
        """
        Returns the mi matrix.

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float) [index, index, index]
        :return: correlation matrix with different lag_mode choices
        """

        # lag_mode dict
        mode = self.lag_modi[lag_mode]
        only_tri = int(self.only_tri)

        # Initialize
        hist2D = numpy.zeros((bins, bins), dtype="int32")
        if lag_mode == 'all':
            corrmat = numpy.zeros((2*tau_max + 1, self.N, self.N),
                                  dtype='float32')
        elif lag_mode == 'sum':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')
        elif lag_mode == 'max':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')

        # Precalculation of the log
        gfunc = numpy.zeros(corr_range+1)
        for t in range(1, corr_range + 1):
            gfunc[t] = t*numpy.log(t)

        # loop over all node pairs, NOT symmetric due to time shifts!
        for i in range(self.N-only_tri):
            for j in range((i+1)*only_tri, self.N):

                if mode == 2:
                    maxcross = 0.0
                    argmax = 0

                # loop over taus from -tau_max to tau_max INCLUDING the last
                # tau value
                for t in range(2*tau_max + 1):
                    tau = t - tau_max

                    # here the joint probability distribution is calculated
                    for k in range(corr_range):
                        indexi = array[tau_max, i, k]
                        indexj = array[t, j, k]
                        hist2D[indexi, indexj] += 1

                    # here the joint entropy is calculated by summing over all
                    # pattern combinations
                    jointent = 0.0
                    for l in range(bins):
                        for m in range(bins):
                            jointent -= gfunc[hist2D[l, m]]
                            hist2D[l, m] = 0

                    jointent /= float(corr_range)
                    jointent += numpy.log(float(corr_range))

                    # Mutual Information is...
                    mi = 0.0
                    mi = 2. * numpy.log(bins) - jointent

                    # norm the mi
                    mi /= numpy.log(bins)

                    # fill in values in matrix depending on lag_mode
                    if mode == 0:
                        corrmat[t, i, j] = mi

                    elif mode == 1:
                        if t <= tau_max:
                            corrmat[1, i, j] += mi
                        if t >= tau_max:
                            corrmat[0, i, j] += mi

                    elif mode == 2:
                        # calculate max and argmax by comparing to previous
                        # value and storing max
                        if mi > maxcross:
                            maxcross = mi
                            argmax = tau

                if mode == 2:
                    corrmat[0, i, j] = maxcross
                    corrmat[1, i, j] = argmax

        if self.only_tri:
            if lag_mode == 'all':
                corrmat = corrmat + corrmat.transpose(0, 2, 1)[::-1]
            if lag_mode == 'sum':
                corrmat[0] += corrmat[1].transpose()
                corrmat[1] = corrmat[0].transpose()
            elif lag_mode == 'max':
                corrmat[0] += corrmat[0].transpose()
                corrmat[1] -= corrmat[1].transpose()

        return corrmat

    #
    # A subroutine for fourier surrogates (from J Donges)
    #

    def correlatedNoiseSurrogates(self, original):
        """
        Generates surrogates by Fourier transforming the original time series,
        randomizing the phases and then applying an inverse Fourier transform.
        Correlated noise surrogates share their power spectrum and
        autocorrelation function with the original time series.

        :type original: 2D array
        :arg original: dim. 0 is index of time series, dim. 1 is time
        :return: surrogate time series (same dimensions as original)
        """

        #  Calculate FFT of original time series
        #  The FFT of the original data has to be calculated only once, so it
        #  is stored in self.originalFFT
        if self.has_fft:
            surrogates = self.originalFFT
        else:
            surrogates = numpy.fft.fft(original, axis=1)
            self.originalFFT = surrogates
            self.has_fft = True

        (nNodes, ntime) = original.shape

        if (ntime % 2) == 0:
            lenPhase = (ntime - 2) / 2
        else:
            lenPhase = (ntime - 1) / 2

        #  Generate random phases uniformly distributed in the interval
        #  [0, 2*Pi]. Guarantee that the phases for positive and negative
        #  frquencies are the same to obtain real surrogates in the end!
        phases = numpy.random.uniform(low=0, high=2 * numpy.pi,
                                      size=(nNodes, lenPhase))

        #  Add random phases uniformly distributed in the interval [0, 2*Pi]
        surrogates[:, 1:lenPhase+1] *= numpy.exp(1j * phases)

        #  Discriminate between even and uneven number of samples
        #  Note that the output of fft has the following form:
        #  - Even sample number: (mean, pos. freq, nyquist freq, neg. freq)
        #  - Odd sample number: (mean, pos. freq, neg. freq)
        if (ntime % 2) == 0:
            surrogates[:, lenPhase+2:ntime] = \
                numpy.flipud(surrogates[:, 1:lenPhase+1].conjugate())
        else:
            surrogates[:, lenPhase+1:ntime] = \
                numpy.flipud(surrogates[:, 1:lenPhase+1].conjugate())

        #  Calculate IFFT and take the real part, the remaining imaginary part
        #  is due to numerical errors
        return numpy.ascontiguousarray(numpy.real(
            numpy.fft.ifft(surrogates, axis=1)))
