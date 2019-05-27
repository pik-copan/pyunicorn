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
"""

# array object and fast numerics
import numpy as np
from numpy import random

from ._ext.numerics import _embed_time_series_array, _recurrence_plot, \
    _twins_s, _twin_surrogates, _test_pearson_correlation, \
    _test_mutual_information

# easy progress bar handling
from ..utils import progressbar


#
#  Define class Surrogates
#

class Surrogates:

    """
    Encapsulates structures and methods related to surrogate time series.

    Provides data structures and methods to generate surrogate data sets from a
    set of time series and to evaluate the significance of various correlation
    measures using these surrogates.

    More information on time series surrogates can be found in [Schreiber2000]_
    and [Kantz2006]_.
    """

    #
    #  Define internal methods
    #
    def __init__(self, original_data, silence_level=1):
        """
        Initialize an instance of Surrogates.

        .. note::
           The order of array dimensions is different from the standard of
           ``core``. Here it is [index, time] for reasons of computational
           speed!

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series for surrogate generation.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print("Generated an instance of the Surrogates class.")

        #  Set class variables
        self.original_data = original_data
        """The original time series for surrogate generation."""
        self.silence_level = silence_level
        """(string) - The inverse level of verbosity of the object."""

        #  Set flags
        self._normalized = False
        self._fft_cached = False
        self._twins_cached = False

        #  Cache
        self._twins = None
        self._original_data_fft = None

    def __str__(self):
        """
        Returns a string representation.
        """
        return 'Surrogates: time series shape %s.' % (
            self.original_data.shape,)

    def clear_cache(self):
        """Clean up cache."""
        try:
            del self._original_data_fft
            del self._twins
        except AttributeError:
            pass

    #
    #  Methods for testing purposes
    #

    @staticmethod
    def SmallTestData():
        """
        Return Surrogates instance representing test a data set of 6 time
        series.

        :rtype: Surrogates instance
        :return: a Surrogates instance for testing purposes.
        """
        #  Create time series
        ts = np.zeros((6, 200))

        for i in range(6):
            ts[i, :] = np.sin(np.arange(200)*np.pi/15. + i*np.pi/2.) + \
                np.sin(np.arange(200) * np.pi / 30.)

        return Surrogates(original_data=ts, silence_level=2)

    #
    #  Define methods to normalize and analyze the data
    #

    @staticmethod
    def normalize_time_series_array(time_series_array):
        """
        :index:`Normalize <pair: normalize; time series array>` an array of
        time series to zero mean and unit variance individually for each
        individual time series.

        **Modifies the given array in place!**

        **Examples:**

        >>> ts = Surrogates.SmallTestData().original_data
        >>> Surrogates.SmallTestData().normalize_time_series_array(ts)
        >>> r(ts.mean(axis=1))
        array([ 0., 0., 0., 0., 0., 0.])
        >>> r(ts.std(axis=1))
        array([ 1., 1., 1., 1., 1., 1.])

        :type time_series_array: 2D array [index, time]
        :arg time_series_array: The time series array to be normalized.
        """
        mean = time_series_array.mean(axis=1)
        std = time_series_array.std(axis=1)

        for i in range(time_series_array.shape[0]):
            #  Remove mean value from time series at each node (grid point)
            time_series_array[i, :] -= mean[i]
            #  Normalize the standard deviation of anomalies to one
            if std[i] != 0:
                time_series_array[i, :] /= std[i]

    def embed_time_series_array(self, time_series_array, dimension, delay):
        """
        Return a :index:`delay embedding` of all time series.

        .. note::
           Only works for scalar time series!

        **Example:**

        >>> ts = Surrogates.SmallTestData().original_data
        >>> Surrogates.SmallTestData().embed_time_series_array(
        ...     time_series_array=ts, dimension=3, delay=2)[0,:6,:]
        array([[ 0.        ,  0.61464833,  1.14988147],
               [ 0.31244015,  0.89680225,  1.3660254 ],
               [ 0.61464833,  1.14988147,  1.53884177],
               [ 0.89680225,  1.3660254 ,  1.6636525 ],
               [ 1.14988147,  1.53884177,  1.73766672],
               [ 1.3660254 ,  1.6636525 ,  1.76007351]])

        :type time_series_array: 2D array [index, time]
        :arg time_series_array: The time series array to be normalized.
        :arg int dimension: The embedding dimension.
        :arg int delay: The embedding delay.
        :rtype: 3D array [index, time, dimension]
        :return: the embedded time series.
        """
        if self.silence_level <= 1:
            print(f"Embedding all time series in dimension {dimension} "
                  f"and with lag {delay} ...")
        (N, n_time) = time_series_array.shape

        embedding = np.empty((N, n_time - (dimension - 1)*delay, dimension))

        _embed_time_series_array(N, n_time, dimension, delay,
                                 time_series_array, embedding)
        return embedding

    # FIXME: I(wb) included the line
    # dimension = embedding.shape[1]
    # whose missing caused an error. I can't guarantee if it is correct.
    def recurrence_plot(self, embedding, threshold):
        """
        Return the :index:`recurrence plot <pair: recurrence plot; time
        series>` from an embedding of a time series.

        Uses supremum norm.

        :type embedding: 2D array [time, dimension]
        :arg embedding: The embedded time series.
        :arg float threshold: The recurrence threshold.
        :rtype: 2D array [time, time]
        :return: the recurrence matrix.
        """
        if self.silence_level <= 1:
            print("Calculating the recurrence plot...")

        n_time = embedding.shape[0]
        dimension = embedding.shape[1]
        R = np.ones((n_time, n_time), dtype="int8")

        _recurrence_plot(n_time, dimension, threshold,
                         embedding.copy(order='c'),
                         R.copy(order='c'))
        return R

    # FIXME: I(wb) included the line
    # dimension = embedding_array.shape[2]
    # whose missing caused an error. I can't guarantee if it is correct.
    def twins(self, embedding_array, threshold, min_dist=7):
        """
        Return list of the :index:`twins <pair: twins; surrogates>` of each
        state vector for all time series.

        Two state vectors are said to be twins if they share the same
        recurrences, i.e., if the corresponding rows or columns in the
        recurrence plot are identical.

        References: [Thiel2006]_, [Marwan2007]_.

        :type embedding_array: 3D array [index, time, dimension]
        :arg  embedding_array: The embedded time series array.
        :arg float threshold: The recurrence threshold.
        :arg number min_dist: The minimum temporal distance for twins.
        :rtype: [[number]]
        :return: the list of twins for each state vector in the time series.
        """
        if self.silence_level <= 1:
            print("Finding twins...")

        N = embedding_array.shape[0]
        n_time = embedding_array.shape[1]
        dimension = embedding_array.shape[2]
        twins = []

        #  Initialize the R matrix with ones
        R = np.empty((n_time, n_time))
        #  Initialize array to store the number of neighbors for each sample
        nR = np.empty(n_time)

        _twins_s(N, n_time, dimension, threshold, min_dist,
                 embedding_array.copy(order='c'), R.copy(order='c'),
                 nR.copy(order='c'), twins)

        return twins

    #
    #  Define methods to generate sets of surrogate time series
    #

    def white_noise_surrogates(self, original_data):
        """
        Return a shuffled copy of a time series array.

        Each time series is shuffled individually. The surrogates correspond to
        realizations of white noise consistent with the :attr:`original_data`
        time series' amplitude distribution.

        **Example** (Distributions of white noise surrogates should the same as
        for the original data):

        >>> ts = Surrogates.SmallTestData().original_data
        >>> surrogates = Surrogates.\
                SmallTestData().white_noise_surrogates(ts)
        >>> np.allclose(np.histogram(ts[0,:])[0],
        ...             np.histogram(surrogates[0,:])[0])
        True

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        if self.silence_level <= 1:
            print("Generating white noise surrogates by random shuffling...")

        #  Generate reference to shuffle function
        shuffle = random.shuffle

        surrogates = original_data.copy()

        for i in range(surrogates.shape[0]):
            shuffle(surrogates[i, :])

        return surrogates

    def correlated_noise_surrogates(self, original_data):
        """
        Return Fourier surrogates.

        Generate surrogates by Fourier transforming the :attr:`original_data`
        time series (assumed to be real valued), randomizing the phases and
        then applying an inverse Fourier transform. Correlated noise surrogates
        share their power spectrum and autocorrelation function with the
        original_data time series.

        The Fast Fourier transforms of all time series are cached to facilitate
        a faster generation of several surrogates for each time series. Hence,
        :meth:`clear_cache` has to be called before generating surrogates from
        a different set of time series!

        .. note::
           The amplitudes are not adjusted here, i.e., the
           individual amplitude distributions are not conserved!

        **Examples:**

        The power spectrum is conserved up to small numerical deviations:

        >>> ts = Surrogates.SmallTestData().original_data
        >>> surrogates = Surrogates.\
                SmallTestData().correlated_noise_surrogates(ts)
        >>> all(r(np.abs(np.fft.fft(ts,         axis=1))[0,1:10]) == \
                r(np.abs(np.fft.fft(surrogates, axis=1))[0,1:10]))
        True

        However, the time series amplitude distributions differ:

        >>> all(np.histogram(ts[0,:])[0] == np.histogram(surrogates[0,:])[0])
        False

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        if self.silence_level <= 1:
            print("Generating correlated noise surrogates...")

        #  Calculate FFT of original_data time series
        #  The FFT of the original_data data has to be calculated only once,
        #  so it is stored in self._original_data_fft.
        if self._fft_cached:
            surrogates = self._original_data_fft
        else:
            surrogates = np.fft.rfft(original_data, axis=1)
            self._original_data_fft = surrogates
            self._fft_cached = True

        #  Get shapes
        (N, n_time) = original_data.shape
        len_phase = surrogates.shape[1]

        #  Generate random phases uniformly distributed in the
        #  interval [0, 2*Pi]
        phases = random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

        #  Add random phases uniformly distributed in the interval [0, 2*Pi]
        surrogates *= np.exp(1j * phases)

        #  Calculate IFFT and take the real part, the remaining imaginary part
        #  is due to numerical errors.
        return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                         axis=1)))

    def AAFT_surrogates(self, original_data):
        """
        Return surrogates using the amplitude adjusted Fourier transform
        method.

        Reference: [Schreiber2000]_

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        #  Create sorted Gaussian reference series
        gaussian = random.randn(original_data.shape[0], original_data.shape[1])
        gaussian.sort(axis=1)

        #  Rescale data to Gaussian distribution
        ranks = original_data.argsort(axis=1).argsort(axis=1)
        rescaled_data = np.zeros(original_data.shape)

        for i in range(original_data.shape[0]):
            rescaled_data[i, :] = gaussian[i, ranks[i, :]]

        #  Phase randomize rescaled data
        phase_randomized_data = \
            self.correlated_noise_surrogates(rescaled_data)

        #  Rescale back to amplitude distribution of original data
        sorted_original = original_data.copy()
        sorted_original.sort(axis=1)

        ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

        for i in range(original_data.shape[0]):
            rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

        return rescaled_data

    def refined_AAFT_surrogates(self, original_data, n_iterations,
                                output="true_amplitudes"):
        """
        Return surrogates using the iteratively refined amplitude adjusted
        Fourier transform method.

        A set of AAFT surrogates (:meth:`AAFT_surrogates`) is iteratively
        refined to produce a closer match of both amplitude distribution and
        power spectrum of surrogate and original data.

        Reference: [Schreiber2000]_

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :arg int n_iterations: Number of iterations / refinement steps
        :arg str output: Type of surrogate to return. "true_amplitudes":
            surrogates with correct amplitude distribution, "true_spectrum":
            surrogates with correct power spectrum, "both": return both outputs
            of the algorithm.
        :rtype: 2D array [index, time]
        :return: The surrogate time series.
        """
        #  Get size of dimensions
        n_time = original_data.shape[1]

        #  Get Fourier transform of original data with caching
        if self._fft_cached:
            fourier_transform = self._original_data_fft
        else:
            fourier_transform = np.fft.rfft(original_data, axis=1)
            self._original_data_fft = fourier_transform
            self._fft_cached = True

        #  Get Fourier amplitudes
        original_fourier_amps = np.abs(fourier_transform)

        #  Get sorted copy of original data
        sorted_original = original_data.copy()
        sorted_original.sort(axis=1)

        #  Get starting point / initial conditions for R surrogates
        # (see [Schreiber2000]_)
        R = self.AAFT_surrogates(original_data)

        #  Start iteration
        for i in range(n_iterations):
            #  Get Fourier phases of R surrogate
            r_fft = np.fft.rfft(R, axis=1)
            r_phases = r_fft / np.abs(r_fft)

            #  Transform back, replacing the actual amplitudes by the desired
            #  ones, but keeping the phases exp(iψ(i)
            s = np.fft.irfft(original_fourier_amps * r_phases, n=n_time,
                             axis=1)

            #  Rescale to desired amplitude distribution
            ranks = s.argsort(axis=1).argsort(axis=1)

            for j in range(original_data.shape[0]):
                R[j, :] = sorted_original[j, ranks[j, :]]

        if output == "true_amplitudes":
            return R
        elif output == "true_spectrum":
            return s
        elif output == "both":
            return (R, s)
        else:
            return (R, s)

    def twin_surrogates(self, original_data, dimension, delay, threshold,
                        min_dist=7):
        """
        Return surrogates using the twin surrogate method.

        Scalar twin surrogates are created by isolating the first component
        (dimension) of the twin surrogate trajectories.

        Twin surrogates share linear and nonlinear properties with the original
        time series, since they correspond to realizations of trajectories of
        the same dynamical systems with different initial conditions.

        References: [Thiel2006]_ [*], [Marwan2007]_.

        The twin lists of all time series are cached to facilitate a faster
        generation of several surrogates for each time series. Hence,
        :meth:`clear_cache` has to be called before generating twin surrogates
        from a different set of time series!

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :arg int dimension: The embedding dimension.
        :arg int delay: The embedding delay.
        :arg float threshold: The recurrence threshold.
        :arg number min_dist: The minimum temporal distance for twins.
        :rtype: 2D array [index, time]
        :return: the twin surrogates.
        """
        #  The algorithm proceeds in several steps:
        #  1. Embed the original_data time series, using time delay embedding
        #     for simplicity. Use the same dimension and time delay delay for
        #     all time series for simplicity. Determine delay using time
        #     delayed mutual information and d using false nearest neighbors
        #     methods.
        #  2. Use the algorithm proposed in [*] to find twins
        #  3. Reconstruct one-dimensional twin surrogate time series

        (N, n_time) = original_data.shape
        n_time = n_time - (dimension-1)*delay

        #  Make sure that twins are calculated only once
        if self._twins_cached:
            twins = self._twins
        else:
            embedding = self.embed_time_series_array(original_data,
                                                     dimension, delay)
            twins = self.twins(embedding, threshold, min_dist)
            self._twins = twins
            self._twins_cached = True

        return _twin_surrogates(N, n_time, twins,
                                original_data.copy(order='c'))

    #
    #  Defines methods to generate correlation measure matrices based on
    #  original_data and surrogate data for significance testing.
    #

    @staticmethod
    def eval_fast_code(function, original_data, surrogates):
        """
        Evaluate performance of fast and slow versions of algorithms.

        Designed for evaluating fast and dirty C code against cleaner code
        using Blitz arrays. Does some profiling and returns the total error
        between the results.

        :type function: Python function
        :arg function: The function to be evaluated.
        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :type surrogates: 2D array [index, time]
        :arg surrogates: The surrogate time series.
        :return float: The total squared difference between resulting matrices.
        """
        #  Some profiling
        # profile.run("fastResult = function(original_data, surrogates,
        #             fast=True)")
        # profile.run("slowResult = function(original_data, surrogates,
        #             fast=False)")

        fast_result = function(original_data, surrogates, fast=True)
        slow_result = function(original_data, surrogates, fast=False)

        #  Return error
        return np.sqrt(((fast_result - slow_result)**2).sum())

    @staticmethod
    def test_pearson_correlation(original_data, surrogates):
        """
        Return a test matrix of the Pearson correlation coefficient (zero lag).

        The test matrix's entry :math:`(i,j)` contains the Pearson correlation
        coefficient between original time series i and surrogate time series j
        at lag zero. The resulting matrix is useful for significance tests
        based on the Pearson correlation matrix of the original data.

        .. note::
           Assumes, that original_data and surrogates are already normalized.

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :type surrogates: 2D array [index, time]
        :arg surrogates: The surrogate time series.
        :rtype: 2D array [index, index]
        :return: the Pearson correlation test matrix.
        """
        (N, n_time) = original_data.shape
        norm = 1. / float(n_time)
        return _test_pearson_correlation(original_data.copy(order='c'),
                                         surrogates.copy(order='c'),
                                         N, n_time)

    @staticmethod
    def test_mutual_information(original_data, surrogates, n_bins=32):
        """
        Return a test matrix of mutual information (zero lag).

        The test matrix's entry :math:`(i,j)` contains the mutual information
        between original time series i and surrogate time series j at zero lag.
        The resulting matrix is useful for significance tests based on the
        mutual information matrix of the original data.

        .. note::
           Assumes, that original_data and surrogates are already normalized.

        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :type surrogates: 2D Numpy array [index, time]
        :arg surrogates: The surrogate time series.
        :arg int n_bins: Number of bins for estimating prob. distributions.
        :rtype: 2D array [index, index]
        :return: the mutual information test matrix.
        """
        (N, n_time) = original_data.shape
        #  Calculate symbolic time series and histograms
        #  Calculate 2D histograms and mutual information
        #  mi[i,j] gives the mutual information between the ith original_data
        #  time series and the jth surrogate time series.
        return _test_mutual_information(original_data.copy(order='c'),
                                        surrogates.copy(order='c'), N, n_time,
                                        n_bins)

    #
    #  Define methods to perform significance tests on correlation measures
    #  based on surrogates.
    #

    def original_distribution(self, test_function, original_data, n_bins=100):
        """
        Return a normalized histogram of a similarity measure matrix.

        The absolute value of the similarity measure is used, since only the
        degree of similarity was of interest originally.

        :type test_function: Python function
        :arg test_function: The function implementing the similarity measure.
        :type original_data: 2D array [index, time]
        :arg original_data: The original time series.
        :arg int n_bins: The number of bins for estimating prob. distributions.
        :rtype: tuple of 1D arrays ([bins],[bins])
        :return: the similarity measure histogram and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print("Estimating probability density distribution of "
                  "original_data ...")

        #  Normalize original_data time series to zero mean and unit variance
        if not self._normalized:
            self.normalize_time_series_array(original_data)
            self._normalized = True

        correlation_measure = np.abs(test_function(original_data,
                                                   original_data))
        (hist, lbb) = np.histogram(correlation_measure, n_bins, density=True)
        #  Normalize
        hist /= hist.sum()

        lbb = lbb[:-1]

        return (hist, lbb)

    def test_threshold_significance(self, surrogate_function, test_function,
                                    realizations=1, n_bins=100,
                                    interval=(-1, 1)):
        """
        Return a test distribution for a similarity measure.

        Perform a significance test on the values of a correlation measure
        based on original_data time series and surrogate data. Returns a
        density estimate (histogram) of the absolute value of the correlation
        measure over all realizations.

        The resulting distribution of the values of similarity measure from
        original and surrogate time series is of use for testing the
        statistical significance of a selected threshold value for climate
        network generation.

        :type surrogate_function: Python function
        :arg surrogate_function: The function implementing the surrogates.
        :type test_function: Python function
        :arg test_function: The function implementing the similarity measure.
        :arg int realizations: The number of surrogates to be created for each
            time series.
        :arg int n_bins: The number of bins for estimating probability
            distribution of test similarity measure.
        :type interval: (float, float)
        :arg interval: The range over which to estimate similarity measure
            distribution.
        :rtype: tuple of 1D arrays ([bins],[bins])
        :return: similarity measure test histogram and lower bin boundaries.
        """
        if self.silence_level <= 1:
            print(f"Starting significance test based on {realizations} "
                  "realizations of surrogates...")

        original_data = self.original_data
        self._fft_cached = False
        self._twins_cached = False

        #  Create reference to np.histogram function
        numpy_hist = np.histogram

        #  Normalize original_data time series to zero mean and unit variance
        if not self._normalized:
            self.normalize_time_series_array(original_data)
            self._normalized = True

        #  Initialize density estimate
        density_estimate = np.zeros(n_bins)

        #  Initialize progress bar
        if self.silence_level <= 2:
            progress = progressbar.ProgressBar(maxval=realizations).start()

        for i in range(realizations):
            #  Update progress bar
            if self.silence_level <= 2:
                progress.update(i)

            #  Get the surrogate
            #  Mean and variance are conserved by all surrogates
            surrogates = surrogate_function(original_data)

            #  Get the correlation measure test matrix
            correlation_measure_test = np.abs(test_function(original_data,
                                                            surrogates))

            #  Test if correlation measure values are outside range
            if correlation_measure_test.min() < interval[0]:
                print("Warning! Correlation measure value left of range.")
            if correlation_measure_test.max() > interval[1]:
                print("Warning! Correlation measure value right of range.")

            #  Estimate density of current realization
            (hist, lbb) = numpy_hist(correlation_measure_test, n_bins,
                                     interval, normed=True)

            #  Add to density estimate over all realizations
            density_estimate += hist

            #  Clean up (should be done automatically by Python,
            #  but you never know...)
            del surrogates, correlation_measure_test

        if self.silence_level <= 2:
            progress.finish()

        #  Normalize density estimate
        density_estimate /= density_estimate.sum()

        lbb = lbb[:-1]

        return (density_estimate, lbb)
