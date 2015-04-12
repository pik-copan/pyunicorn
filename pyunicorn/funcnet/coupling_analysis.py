#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.

Written by Jakob Runge.
MSI Method Reference: [Pompe2011]_
"""

# array object and fast numerics
import numpy
# C++ inline code
import weave


#
#  Define class Coupling Analysis
#

class CouplingAnalysis(object):

    """
    Contains methods to calculate coupling matrices from large arrays
    of scalar time series.

    Comprises linear and Information theoretic measures, lagged
    and directed (causal) couplings.

    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, dataarray, only_tri=False, silence_level=0):
        """
        Initialize an instance of CouplingAnalysis.

        Possible choices for only_tri:
            - "True" will calculate only the upper triangle of the coupling
              matrix, excluding the diagonal, assuming symmetry (not for
              directed measures)
            - "False" will calculate the whole matrix (asymmetry somes from
              different integration ranges)

        :type dataarray: 4D/3D/2D array [time, index, index] or [time, index]
        :arg dataarray: The time series array with time in first dimension
        :arg bool only_tri: Symmetric/asymmetric assumption on coupling matrix.
        :arg int silence_level: The inverse level of verbosity of the object.
        """

        #  only_tri will calculate the upper triangle excluding the diagonal
        #  only, This assumes stationarity on the time series
        self.only_tri = only_tri

        #  Set silence level
        self.silence_level = silence_level

        #  Flatten observable anomaly array along lon/lat dimension to allow
        #  for more convinient indexing and transpose the whole array as this
        #  is faster in loops
        if numpy.rank(dataarray) == 4:
            (self.total_time, n_lev, n_lat, n_lon) = dataarray.shape
            self.N = n_lev * n_lat * n_lon
            self.dataarray = numpy.fastCopyAndTranspose(
                dataarray.reshape(-1, self.N))
        if numpy.rank(dataarray) == 3:
            (self.total_time, n_lat, n_lon) = dataarray.shape
            self.N = n_lat * n_lon
            self.dataarray = numpy.fastCopyAndTranspose(
                dataarray.reshape(-1, self.N))

        elif numpy.rank(dataarray) == 2:
            (self.total_time, self.N) = dataarray.shape
            self.dataarray = numpy.fastCopyAndTranspose(dataarray)

        else:
            print "irregular array shape..."
            self.dataarray = numpy.fastCopyAndTranspose(dataarray)

        #  factorials below 10 in a list for permutation patterns
        self.factorial = \
            numpy.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])

        #  lag_mode dict
        self.lag_modi = {"all": 0, "sum": 1, "max": 2}

    def __str__(self):
        """Return a string representation of the CouplingAnalysis object."""
        text = CouplingAnalysis.__str__(self)

        return text

    #
    #  Define methods to calculate correlation strength and lags
    #

    def cross_correlation(self, tau_max=0, lag_mode='all'):
        """
        Returns the normalized cross correlation from all pairs of nodes from a
        range of time lags.

        The calculation ranges are shown below::

            (-------------------------total_time--------------------------)
            (---tau_max---)(---------corr_range------------)(---tau_max---)

        CC is calculated about corr_range and with the other time series
        shifted by tau

        Possible choices for lag_mode:
            - "all" will return the full function for all lags, possible large
              memory need if only_tri is True, only the upper triangle contains
              the values, the lower one is zeros
            - "sum" will return the sum over positive and negative lags
              seperatly, each inclunding tau=0 corrmat[0] is the positive sum,
              corrmat[1] the negative sum
            - "max" will return only the maximum coupling (in corrmat[0]) and
              its lag (in corrmat[1])

        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: the output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
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
            normalized_array[t] /= \
                normalized_array[t].std(axis=1).reshape(self.N, 1)

            #  Correct for grid points with zero variance in their time series
            normalized_array[t][numpy.isnan(normalized_array[t])] = 0

        return self._calculate_cc(normalized_array, corr_range=corr_range,
                                  tau_max=tau_max, lag_mode=lag_mode)

    def surrogate_for_cc(self, sample_range=100, tau_max=1, lag_mode='all'):
        """
        Returns a joint shuffled surrogate of the full dataarray of length
        sample_range for all taus.

        Used for time evolution analysis. First one initializes the
        CouplingAnalysis class with the full dataarray and then this function
        is called for every single surrogate.

        :arg int sample_range: length of sample
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: the output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """

        # lag_mode dict
        mode = self.lag_modi[lag_mode]

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
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """

        # lag_mode dict
        mode = self.lag_modi[lag_mode]

        N = self.N

        if lag_mode == 'all':
            corrmat = numpy.zeros((2*tau_max + 1, self.N, self.N),
                                  dtype='float32')
        elif lag_mode == 'sum':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')
        elif lag_mode == 'max':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')

        code = r"""
        int i,j,t,k, argmax;
        double crossij, max;

        // loop over all node pairs, NOT symmetric due to time shifts!
        for (i = 0; i < N - only_tri; i++) {
            for (j = (i+1)*only_tri; j < N; j++) {

                if(mode == 2) {
                    max = 0.0;
                    argmax = 0;
                }

                // loop over taus INCLUDING the last tau value
                for( t = 0; t < 2*tau_max + 1; t++) {

                    crossij = 0;
                    // here the actual cross correlation is calculated
                    for ( k = 0; k < corr_range; k++) {
                        crossij += array(tau_max, i, k) * array(t, j, k);
                    }

                    // fill in values in matrix depending on lag_mode
                    if(mode == 0) {
                        corrmat(t,i,j) = crossij/(float)(corr_range);
                    }
                    else if(mode == 1) {
                        if( t <= tau_max) {
                            corrmat(1,i,j) +=
                                fabs(crossij)/(float)(corr_range);
                        }
                        if( t >= tau_max) {
                            corrmat(0,i,j) +=
                                fabs(crossij)/(float)(corr_range);
                        }
                    }
                    else if(mode == 2) {
                        // calculate max and argmax by comparing to
                        // previous value and storing max
                        if (fabs(crossij) > max) {
                            max = fabs(crossij);
                            argmax = t;
                        }
                    }
                }
                if(mode == 2) {
                    corrmat(0,i,j) = max/(float)(corr_range);
                    corrmat(1,i,j) = argmax - tau_max;
                }
            }
        }
        """
        only_tri = int(self.only_tri)
        args = ['array', 'corrmat', 'N', 'tau_max', 'corr_range', 'mode',
                'only_tri']
        weave.inline(code, arg_names=args,
                     type_converters=weave.converters.blitz, compiler='gcc',
                     extra_compile_args=["-O3"])

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
            memory need if only_tri is True, only the upper triangle contains
            the values, the lower one is zeros
          - "sum" will return the sum over positive and negative lags
            seperatly, each inclunding tau=0 corrmat[0] is the positive sum,
            corrmat[1] the negative sum
          - "max" will return only the maximum coupling (in corrmat[0]) and its
            lag (in corrmat[1])

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: the output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """
        if bins < 255:
            dtype = 'uint8'
        else:
            dtype = 'int16'

        #  Normalize anomaly time series to zero mean and unit variance for all
        #  lags, array contains normalizations for all lags
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
                (array.reshape(self.N, corr_range, 1) >=
                 edges.reshape(self.N, 1, bins)).sum(axis=2) - 1

        return self._calculate_mi(symbolic_array, corr_range=corr_range,
                                  bins=bins, tau_max=tau_max,
                                  lag_mode=lag_mode)

    def surrogate_for_mi(self, bins=16, sample_range=100, tau_max=1,
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
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
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
                (array.reshape(self.N, sample_range, 1) >=
                 edges.reshape(self.N, 1, bins)).sum(axis=2) - 1

        return self._calculate_mi(symbolic_array, corr_range=sample_range,
                                  bins=bins, tau_max=tau_max,
                                  lag_mode=lag_mode)

    def _calculate_mi(self, array, corr_range, bins, tau_max, lag_mode):
        """
        Returns the mi matrix.

        :arg int bins: number of bins for estimating MI
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """

        N = self.N
        # lag_mode dict
        mode = self.lag_modi[lag_mode]

        # Initialize for weave
        hist2D = numpy.zeros((bins, bins), dtype="int32")
        if lag_mode == 'all':
            corrmat = numpy.zeros((2*tau_max + 1, self.N, self.N),
                                  dtype='float32')
        elif lag_mode == 'sum':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')
        elif lag_mode == 'max':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')

        code = r"""
        int i,j,k,l,m,t, argmax, tau, indexi, indexj;
        double max, jointent, mi, gfunc[corr_range];

        // Precalculation of the log
        gfunc[0] = 0.;
        for(t = 1; t < corr_range; t++) {
            gfunc[t] = t*log(t);
        }


        // loop over all node pairs, NOT symmetric due to time shifts!
        for (i = 0; i < N - only_tri; i++) {
            for (j = (i+1)*only_tri; j < N; j++) {

                if(mode == 2) {
                    max = 0.0;
                    argmax = 0;
                }

                // loop over taus from -tau_max to tau_max
                // INCLUDING the last tau value
                for( t = 0; t < (2*tau_max + 1); t++) {
                    tau = t - tau_max;

                    // here the joint probability distribution is calculated
                    for( k = 0; k < corr_range; k++) {
                        indexi = array(tau_max, i, k);
                        indexj = array(t, j, k);
                        hist2D(indexi, indexj) += 1;
                    }

                    // here the joint entropy is calculated by summing
                    // over all pattern combinations
                    jointent = 0.0;
                    for(l = 0; l < bins; l++) {
                        for(m = 0; m < bins; m++) {
                            jointent -= gfunc[hist2D(l,m)];
                            hist2D(l,m) = 0;
                        }
                    }
                    jointent /= (float)(corr_range);
                    jointent += log((float)(corr_range));

                    // Mutual Information is...
                    mi = 0.0;
                    mi = 2. * log(bins) - jointent;

                    // norm the mi
                    mi /= log(bins);

                    // fill in values in matrix depending on lag_mode
                    if(mode == 0) {
                        corrmat(t,i,j) = mi;
                    }
                    else if(mode == 1) {
                        if( t <= tau_max) {
                            corrmat(1,i,j) += mi;}
                        if( t >= tau_max) {
                            corrmat(0,i,j) += mi;}
                    }
                    else if(mode == 2) {
                        // calculate max and argmax by comparing to
                        // previous value and storing max
                        if (mi > max) {
                            max = mi;
                            argmax = tau;
                        }
                    }
                }
                if(mode == 2) {
                    corrmat(0,i,j) = max;
                    corrmat(1,i,j) = argmax;
                }
            }
        }
        """
        only_tri = int(self.only_tri)
        args = ['array', 'corrmat', 'hist2D', 'N', 'tau_max', 'corr_range',
                'bins', 'mode', 'only_tri']
        weave.inline(code, arg_names=args,
                     type_converters=weave.converters.blitz, compiler='gcc',
                     extra_compile_args=["-O3"])

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

    def msi(self, patt_dim=2, patt_step=1, patt_sort=1, tau_max=1,
            lag_mode='all'):
        """
        Returns the mutual sorting information from all pairs of nodes from a
        range of time lags.

        MSI = H_m,n+d  +  H_m+d,n  -  H_m+d,n+d  -  H_m,n   where H are the
        joint entropies of patterns and m,n are the patternseries of X,Y, d is
        the number of sorted points ahead.

        Further reference in [Pompe2011]_.

        The calculation ranges are shown below::

            (-------------------------total_time--------------------------)
            (---tau_max---)(---------corr_range------------)(---tau_max---)

        MSI is calculated about corr_range and with the other time series
        shifted by tau

        Possible choices for lag_mode:
         - "all" will return the full function for all lags, possible large
            memory need if only_tri is True, only the upper triangle contains
            the values, the lower one is zeros
         - "sum" will return the sum over positive and negative lags seperatly,
           each inclunding tau=0 corrmat[0] is the positive sum, corrmat[1] the
           negative sum
         - "max" will return only the maximum coupling (in corrmat[0]) and its
           lag (in corrmat[1])

        :arg int patt_dim: dimension of the pattern rank vector
        :arg int patt_step: step size of the pattern rank vector
        :arg int patt_sort: dimension of the sorted pattern rank vector among
            the previous patt_dims
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """
        if int(self.factorial[patt_dim + patt_sort]) < 256:
            dtype = 'uint8'
        elif int(self.factorial[patt_dim + patt_sort]) < 2**15:
            dtype = 'int16'
        else:
            dtype = 'int32'

        # Both arrays will be of the same length
        patt_time = self.total_time - (patt_dim + patt_sort) * patt_step
        corr_range = patt_time - 2*tau_max

        # Calculate the pattern time series for both dimension patt_dim and
        # patt_dim + patt_sort
        (patt, pattplus) = self.\
            _calculate_patt_pattplus(patt_time=patt_time, patt_dim=patt_dim,
                                     patt_step=patt_step, patt_sort=patt_sort)

        patt_taus = numpy.empty((2*tau_max+1, self.N, corr_range), dtype=dtype)
        pattplus_taus = numpy.empty((2*tau_max+1, self.N, corr_range),
                                    dtype=dtype)

        for t in range(2*tau_max + 1):

            patt_taus[t], pattplus_taus[t] = \
                patt[:, t: t + corr_range], pattplus[:, t: t + corr_range]

        return self._calculate_msi(patt=patt_taus, pattplus=pattplus_taus,
                                   corr_range=corr_range, tau_max=tau_max,
                                   patt_dim=patt_dim, patt_sort=patt_sort,
                                   lag_mode=lag_mode)

    def surrogate_for_msi(self, sample_range=100, patt_dim=2, patt_step=1,
                          patt_sort=1, tau_max=1, lag_mode='all'):
        """
        Returns a joint shuffled surrogate of the full dataarray of length
        sample_range for all taus.

        Used for time evolution analysis. First one initializes the
        CouplingAnalysis class with the full dataarray and then this function
        is called for every single surrogate.

        :arg int sample_range: length of sample
        :arg int patt_dim: dimension of the pattern rank vector
        :arg int patt_step: step size of the pattern rank vector
        :arg int patt_sort: dimension of the sorted pattern rank vector among
            the previous patt_dims
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """

        if int(self.factorial[patt_dim + patt_sort]) < 256:
            dtype = 'uint8'
        elif int(self.factorial[patt_dim + patt_sort]) < 2**15:
            dtype = 'int16'
        else:
            dtype = 'int32'

        # Both arrays will be of the same length
        patt_time = self.total_time - (patt_dim + patt_sort) * patt_step
        corr_range = patt_time - 2*tau_max

        # Calculate the pattern time series for both dimension patt_dim and
        # patt_dim + patt_sort
        (patt, pattplus) = self.\
            _calculate_patt_pattplus(patt_time=patt_time, patt_dim=patt_dim,
                                     patt_step=patt_step, patt_sort=patt_sort)

        perm = numpy.random.permutation(
            range(tau_max, patt_time - tau_max))[:sample_range]

        patt_taus = numpy.empty((2*tau_max + 1, self.N, sample_range),
                                dtype=dtype)
        pattplus_taus = numpy.empty((2*tau_max + 1, self.N, sample_range),
                                    dtype=dtype)

        for t in range(2*tau_max + 1):
            tau = t - tau_max

            patt_taus[t], pattplus_taus[t] = \
                patt[:, perm + tau], pattplus[:, perm + tau]

        return self._calculate_msi(patt=patt_taus, pattplus=pattplus_taus,
                                   corr_range=sample_range, tau_max=tau_max,
                                   patt_dim=patt_dim, patt_sort=patt_sort,
                                   lag_mode=lag_mode)

    def _calculate_patt_pattplus(self, patt_time, patt_dim, patt_step,
                                 patt_sort):
        """
        Returns the pattern series for patt_dim and patt_dim + patt_sort.

        Each pattern is assigned a unique integer.
        Further reference in [Pompe2011]_.

        :arg int patt_time: length of pattern series arrays for patt and
            pattplus
        :arg int patt_dim: dimension of the pattern rank vector
        :arg int patt_step: step size of the pattern rank vector
        :arg int patt_sort: dimension of the sorted pattern rank vector among
            the previous patt_dims
        :rtype: tuple of 2D int arrays ([nodes,time], [nodes,time])
        :return: patt and pattplus arrays ([N,patt_time], [N,patt_time])
        """
        if int(self.factorial[patt_dim + patt_sort]) < 256:
            dtype = 'uint8'
        elif int(self.factorial[patt_dim + patt_sort]) < 2**15:
            dtype = 'int16'
        else:
            dtype = 'int32'

        # Convert data to two arrays of patterns for patt_dim and
        # patt_dim + patt_sort
        patt = numpy.zeros((self.N, patt_time), dtype=dtype)
        pattplus = numpy.zeros((self.N, patt_time), dtype=dtype)

        for i in xrange(patt_time):
            pattern = self.dataarray[:, i: i + patt_dim * patt_step:
                                     patt_step].argsort(axis=1).argsort(axis=1)
            patternplus = self.\
                dataarray[:, i: i + (patt_dim + patt_sort) * patt_step:
                          patt_step].argsort(axis=1).argsort(axis=1)
            for k in xrange(1, patt_dim):
                patt[:, i] += pattern[:, : k + 1].argsort(axis=1).\
                    argsort(axis=1)[:, k] * self.factorial[k]
            for k in xrange(1, patt_dim + patt_sort):
                pattplus[:, i] += patternplus[:, : k + 1].argsort(axis=1).\
                    argsort(axis=1)[:, k] * self.factorial[k]

        return (patt, pattplus)

    def _calculate_msi(self, patt, pattplus, corr_range, tau_max,
                       patt_dim, patt_sort, lag_mode):
        """
        Returns the msi matrix.

        :arg int corr_range: integration range
        :arg int patt_dim: dimension of the pattern rank vector
        :arg int patt_sort: dimension of the sorted pattern rank vector among
            the previous patt_dims
        :arg int tau_max: maximum lag in both directions, including last lag
        :arg str lag_mode: the output mode
        :rtype: 3D numpy array (float)
                [lags, nodes, nodes] for lag_mode 'all',
                [2, nodes, nodes] for lag_mode 'sum',
                [2, nodes, nodes] for lag_mode 'max'
        :return: correlation matrix with different lag_mode choices
        """

        # lag_mode dict
        mode = self.lag_modi[lag_mode]

        #  Set up arrays for weave
        bins = int(self.factorial[patt_dim])
        binsplus = int(self.factorial[patt_dim + patt_sort])
        N = self.N

        # Allocate histograms
        histmn = numpy.zeros((bins, bins), dtype="int32")
        histmnplus = numpy.zeros((bins, binsplus), dtype="int32")
        histmplusn = numpy.zeros((binsplus, bins), dtype="int32")
        histmplusnplus = numpy.zeros((binsplus, binsplus), dtype="int32")
        if lag_mode == 'all':
            corrmat = numpy.zeros((2*tau_max + 1, self.N, self.N),
                                  dtype='float32')
        elif lag_mode == 'sum':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')
        elif lag_mode == 'max':
            corrmat = numpy.zeros((2, self.N, self.N), dtype='float32')

        code = r"""
        int i,j,k,m,l,t, argmax, tau, indexi, indexj, indexiplus, indexjplus;
        double max, entmn, entmplusn, entmnplus, entmplusnplus;
        double msi, gamma = .5772156649015328606065, gfunc[corr_range];

        // This calculates the summands of Grassberger's best
        // analytic estimator for Shannon's Entropy
        gfunc[0] = 0.;
        gfunc[1] = - gamma - log(2.);
        for(t = 2; t < corr_range; t++) {
            if( t % 2 == 0) {
                gfunc[t] = t*(gfunc[t-1]/(t-1) + 2./(t-1));
            }
            else {
                gfunc[t] = t*gfunc[t-1]/(t-1);
            }
        }

        // Other choice is using the naive estimator
//      gfunc[0] = 0.;
//      for(t = 1; t < corr_range; t++) {
//          gfunc[t] = t*log(t);
//      }

        // loop over all node pairs, NOT symmetric due to time shifts!
        for (i = 0; i < N - only_tri; i++) {
            for (j = (i+1)*only_tri; j < N; j++) {

                if(mode == 2) {
                    max = 0.0;
                    argmax = 0;
                }

                // loop over taus from -tau_max to tau_max
                // INCLUDING the last tau value
                for(t = 0; t < (2*tau_max + 1); t++) {
                    tau = t - tau_max;

                    // here the joint probability distributions are calculated
                    for( k = 0; k < corr_range; k++) {
                        indexi = patt(tau_max, i, k);
                        indexj = patt(t, j, k);
                        indexiplus = pattplus(tau_max, i, k);
                        indexjplus = pattplus(t, j, k);
                        histmn(indexi, indexj) += 1;
                        histmplusn(indexiplus, indexj) += 1;
                        histmnplus(indexi, indexjplus) += 1;
                        histmplusnplus(indexiplus, indexjplus) += 1;
                    }

                    // here the joint entropies are calculated
                    entmn = 0.0;
                    for(m = 0; m < bins; m++) {
                        for(l = 0; l < bins; l++) {
                            entmn -= gfunc[histmn(m,l)];
                            histmn(m,l) = 0;
                        }
                    }
                    entmn /= (float)(corr_range);
                    entmn += log((float)(corr_range));

                    entmplusn = 0.0;
                    for(m = 0; m < binsplus; m++) {
                        for(l = 0; l < bins; l++) {
                            entmplusn -= gfunc[histmplusn(m,l)];
                            histmplusn(m,l) = 0;
                        }
                    }
                    entmplusn /= (float)(corr_range);
                    entmplusn += log((float)(corr_range));

                    entmnplus = 0.0;
                    for(m = 0; m < bins; m++) {
                        for(l = 0; l < binsplus; l++) {
                            entmnplus -= gfunc[histmnplus(m,l)];
                            histmnplus(m,l) = 0;
                        }
                    }
                    entmnplus /= (float)(corr_range);
                    entmnplus += log((float)(corr_range));

                    entmplusnplus = 0.0;
                    for(m = 0; m < binsplus; m++) {
                        for(l = 0; l < binsplus; l++) {
                            entmplusnplus -= gfunc[histmplusnplus(m,l)];
                            histmplusnplus(m,l) = 0;
                        }
                    }
                    entmplusnplus /= (float)(corr_range);
                    entmplusnplus += log((float)(corr_range));

                    // Now get Mutual Sorting Information
                    msi = 0.0;
                    msi = entmnplus + entmplusn - entmplusnplus - entmn;

                    // norm the msi by the smaller entropy

                    // fill in values in matrix depending on lag_mode
                    if(mode == 0) {
                        corrmat(t,i,j) = msi;
                    }
                    else if(mode == 1) {
                        if( t <= tau_max) {
                            corrmat(1,i,j) += msi;}
                        if( t >= tau_max) {
                            corrmat(0,i,j) += msi;}
                    }
                    else if(mode == 2) {
                        // calculate max and argmax by comparing to
                        // previous value and storing max
                        if (msi > max) {
                            max = msi;
                            argmax = tau;
                        }
                    }
                }

                if(mode == 2) {
                    corrmat(0,i,j) = max;
                    corrmat(1,i,j) = argmax;
                }
            }
        }
        """
        only_tri = int(self.only_tri)
        args = ['patt', 'pattplus', 'corrmat', 'histmn', 'histmplusn',
                'histmnplus', 'histmplusnplus', 'N', 'tau_max', 'corr_range',
                'bins', 'binsplus', 'mode', 'only_tri']
        weave.inline(code, arg_names=args,
                     type_converters=weave.converters.blitz, compiler='gcc',
                     extra_compile_args=["-O3"])

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
