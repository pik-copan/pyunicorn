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
Provides classes for the analysis of dynamical systems and time series based
on recurrence plots, including measures of recurrence quantification
analysis (RQA) and recurrence network analysis.
"""

from math import factorial
from typing import Tuple
from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray

from ..core.cache import Cached
from ..core._ext.types import to_cy, NODE, LAG, FIELD, DFIELD
from ._ext.numerics import _embed_time_series, _manhattan_distance_matrix_rp, \
    _euclidean_distance_matrix_rp, _supremum_distance_matrix_rp, \
    _set_adaptive_neighborhood_size, _bootstrap_distance_matrix_manhattan, \
    _bootstrap_distance_matrix_euclidean, \
    _bootstrap_distance_matrix_supremum, \
    _diagline_dist_missingvalues, _diagline_dist, \
    _diagline_dist_sequential_missingvalues, _diagline_dist_sequential, \
    _vertline_dist_missingvalues, _vertline_dist, \
    _vertline_dist_sequential_missingvalues, _vertline_dist_sequential, \
    _rejection_sampling, _white_vertline_dist, _twins_r, _twin_surrogates_r


class RecurrencePlot(Cached):
    """
    Class RecurrencePlot for generating and quantitatively analyzing
    :index:`recurrence plots <single: recurrence plot>`.

    The RecurrencePlot class supports the construction of recurrence plots
    from multi-dimensional time series, optionally using embedding. Currently,
    manhattan, euclidean and supremum norms are provided for measuring
    distances in phase space.

    Methods for calculating commonly used measures of :index:`recurrence
    quantification analysis <pair: RQA; recurrence plot>` (RQA) are provided,
    e.g., determinism, maximum diagonal line length and laminarity. The
    definitions of these measures together with a review of the theory and
    applications of recurrence plots are given in [Marwan2007]_.

    **Examples:**

     - Create an instance of RecurrencePlot with a fixed recurrence threshold
       and without embedding::

           RecurrencePlot(time_series, threshold=0.1)

     - Create an instance of RecurrencePlot with a fixed recurrence threshold
       in units of STD and without embedding::

           RecurrencePlot(time_series, threshold_std=0.03)

     - Create an instance of RecurrencePlot at a fixed (global) recurrence rate
       and using time delay embedding::

           RecurrencePlot(time_series, dim=3, tau=2,
                          recurrence_rate=0.05).recurrence_rate()
    """

    #
    #  Internal methods
    #

    def __init__(self, time_series: NDArray, metric: str = "supremum",
                 normalize: bool = False, missing_values: bool = False,
                 sparse_rqa: bool = False, silence_level: int = 0,
                 **kwargs):
        """
        Initialize an instance of RecurrencePlot.

        Either recurrence threshold ``threshold``/``threshold_std``, recurrence
        rate ``recurrence_rate`` or local recurrence rate
        ``local_recurrence_rate`` have to be given as keyword arguments.

        Embedding is only supported for scalar time series. If embedding
        dimension ``dim`` and delay ``tau`` are **both** given as keyword
        arguments, embedding is applied. Multidimensional time series are
        processed as is by default.

        :attention: The sparse_rqa feature is experimental and currently only
                    works for fixed threshold and the supremum metric.

        :type time_series: 2D array (time, dimension)
        :arg time_series: The time series to be analyzed, can be scalar or
            multi-dimensional.
        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :arg bool normalize: Decide whether to normalize the time series to
            zero mean and unit standard deviation.
        :arg bool missing_values: Toggle special treatment of missing values in
            :attr:`.RecurrencePlot.time_series`.
        :arg bool sparse_rqa: Toggles sequential RQA computation using less
            memory for use with long time series.
        :arg bool skip_recurrence: Skip calculation of recurrence matrix within
            RP class (e.g. when overloading respective methods in child class)
        :arg int silence_level: Inverse level of verbosity of the object.
        :arg number threshold: The recurrence threshold keyword for generating
            the recurrence plot using a fixed threshold.
        :arg number threshold_std: The recurrence threshold keyword for
            generating the recurrence plot using a fixed threshold in units of
            the time series' STD.
        :arg number recurrence_rate: The recurrence rate keyword for generating
            the recurrence plot using a fixed recurrence rate.
        :arg number local_recurrence_rate: The local recurrence rate keyword
            for generating the recurrence plot using a fixed local recurrence
            rate (same number of recurrences for each state vector).
        :arg number adaptive_neighborhood_size: The adaptive neighborhood size
            parameter for generating recurrence plots based on the algorithm in
            [Xu2008]_.
        :arg number dim: The embedding dimension.
        :arg number tau: The embedding delay.
        """
        #  Set silence_level
        self.silence_level = silence_level
        """The inverse level of verbosity of the object."""

        #  Set missing_values flag
        self.missing_values = missing_values
        """Controls special treatment of missing values in
           :attr:`.RecurrencePlot.time_series`."""

        #  Set sparse RQA flag
        self.sparse_rqa = sparse_rqa
        """Controls sequential calculation of RQA measures."""

        #  Store time series
        self.time_series = to_cy(time_series, FIELD)
        """The time series from which the recurrence plot is constructed."""

        #  Reshape time series
        self.time_series.shape = (self.time_series.shape[0], -1)

        #  Store type of metric
        self._known_metrics = ("manhattan", "euclidean", "supremum")
        assert metric in self._known_metrics
        self.metric = metric
        """The metric used for measuring distances in phase space."""

        # minimal denominator for numerical stability
        self._epsilon = 1e-08

        #  Normalize time series
        if normalize:
            self.normalize_time_series(self.time_series)

        #  Get embedding dimension and delay from **kwargs
        self.dim = kwargs.get("dim")
        self.tau = kwargs.get("tau")

        self.N: int = 0
        """The number of state vectors (number of lines and rows) of the RP."""
        self.R = None
        """The recurrence matrix."""

        self._mut_embedding: int = 0
        if (self.dim is not None) and (self.tau is not None):
            #  Embed the time series
            self.embedding = self.embed_time_series(
                self.time_series, self.dim, self.tau)
        else:
            self.embedding = self.time_series

        #  Get missing value indices
        if self.missing_values:
            self.missing_value_indices = \
                np.isnan(self.embedding).sum(axis=1) != 0

        #  Get threshold or recurrence rate from **kwargs, construct recurrence
        #  plot accordingly
        self.threshold = kwargs.get("threshold")
        self.threshold_std = kwargs.get("threshold_std")
        #  Make sure not to overwrite the method recurrence_rate()
        recurrence_rate = kwargs.get("recurrence_rate")
        self.local_recurrence_rate = kwargs.get("local_recurrence_rate")
        self.adaptive_neighborhood_size = \
            kwargs.get("adaptive_neighborhood_size")

        #  Precompute recurrence matrix only if sequential RQA is switched off,
        #  and not calling from child class with respective overriding methods.
        skip_recurrence = kwargs.get("skip_recurrence")

        if not sparse_rqa and not skip_recurrence:
            if self.threshold is not None:
                #  Calculate the recurrence matrix R using the radius of
                #  neighborhood threshold
                RecurrencePlot.set_fixed_threshold(self, self.threshold)
            elif self.threshold_std is not None:
                #  Calculate the recurrence matrix R using the radius of
                #  neighborhood threshold in units of the time series' STD
                RecurrencePlot.set_fixed_threshold_std(
                    self, self.threshold_std)
            elif recurrence_rate is not None:
                #  Calculate the recurrence matrix R using a fixed recurrence
                #  rate
                RecurrencePlot.set_fixed_recurrence_rate(
                    self, recurrence_rate)
            elif self.local_recurrence_rate is not None:
                #  Calculate the recurrence matrix R using a fixed local
                #  recurrence rate
                RecurrencePlot.set_fixed_local_recurrence_rate(
                    self, self.local_recurrence_rate)
            elif self.adaptive_neighborhood_size is not None:
                #  Calculate the recurrence matrix R using the adaptive
                #  neighborhood size algorithm in [Xu2008]_
                RecurrencePlot.set_adaptive_neighborhood_size(
                    self, self.adaptive_neighborhood_size)
            else:
                raise NameError("Please give either threshold or \
                                recurrence_rate to construct the recurrence \
                                plot!")

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        return (self._mut_embedding,)

    def __str__(self):
        """
        Returns a string representation.
        """
        return ("RecurrencePlot: "
                f"time series shape {self.time_series.shape}.\n"
                f"Embedding dimension {self.dim if self.dim else 0}\n"
                f"Threshold {self.threshold}, {self.metric} metric")

    @property
    def embedding(self) -> np.ndarray:
        """
        The embedded time series / phase space trajectory
        (time, embedding dimension).
        """
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: np.ndarray):
        self._embedding = to_cy(embedding, DFIELD)
        self.N = self._embedding.shape[0]
        self._mut_embedding += 1

    #
    #  Service methods
    #

    def recurrence_matrix(self):
        """
        Return the current recurrence matrix :math:`R`.

        :rtype: 2D square Numpy array
        :return: the current recurrence matrix :math:`R`.
        """
        if not self.sparse_rqa:
            return self.R
        else:
            print("Exception: Sequential RQA mode is enabled. "
                  "Recurrence matrix is not stored in memory.")
            return None

    def distance_matrix(self, metric: str):
        """
        Return phase space distance matrix :math:`D` according to the chosen
        metric.

        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :rtype: 2D square array
        :return: the phase space distance matrix :math:`D`
        """
        assert metric in self._known_metrics, f"unknown metric: {metric}"
        return getattr(RecurrencePlot, f"{metric}_distance_matrix")(self)

    #
    #  Time series handling methods
    #

    @staticmethod
    def normalize_time_series(time_series):
        """
        :index:`Normalize <pair: normalize; time series>` each component of a
        time series **in place**.

        Works also for complex valued time series.

        .. note::
           Modifies the given array in place!

        :type time_series: 2D array (time, dimension)
        :arg time_series: The time series to be normalized.
        """
        #  Get number of components of time_series
        dim = time_series.shape[1]

        mean = time_series.mean(axis=0)
        std = time_series.std(axis=0)

        #  Normalize all components separately
        for i in range(dim):
            time_series[:, i] -= mean[i]
            if std[i] != 0:
                time_series[:, i] /= std[i]

    # Heitzig:
    @staticmethod
    def legendre_coordinates(x, dim=3, t=None, p=None, tau_w="est"):
        """
        Return a phase space trajectory reconstructed using orthogonal
        polynomial filters.

        The reconstructed state vector components are the zero-th to (dim-1)-th
        derivatives of the (possibly irregularly spaced) time series x
        as estimated by folding with the orthogonal polynomial filters that
        correspond to the sequence of measurement time points t.

        This is a generalization for irregularly spaced time series
        of the "Legendre coordinates" introduced in Gibson et al. (1992).

        :arg array-like x: Time series values
        :arg int dim: Dimension > 0 of reconstructed phase space. Default: 3
        :type t: array-like or None
        :arg t: Optional array of measurement time points corresponding to the
            values in x. Default: [0,...,x.size-1]
        :type p: int > 0 or None
        :arg p: No. of past and future time points to use for the estimation.
            Default: dim or determined by tau_w if given
        :type tau_w: float > 0 or "est" or None
        :arg tau_w: Optional (average) window width to use in determining p
            when p = None. Following Gibson et al. (1992), this should be about
            sqrt(3/< x**2 >) * std(x), or about a quarter period.  If "est",
            this is estimated iteratively, starting with
            4 * (max(t)-min(t)) / (N-1) and estimating x' from that.
        :rtype:  2D array [observation index, dimension index]
        :return: Estimated derivatives. Rows are reconstructed state vectors.
        """
        x = np.array(x).flatten()
        N = x.size

        # time points:
        if t is None:
            t = np.arange(N)

        if p is None:
            if tau_w == "est":
                tau_w = 4 * (t.max() - t.min()) / (N-1)
                for i in range(5):
                    y0 = RecurrencePlot.legendre_coordinates(x, dim=2, t=t,
                                                             tau_w=tau_w)
                    tau_w = np.sqrt(3*x.var()/(y0[:, 1]**2).mean())
                print("tau_w set to", tau_w)
            if tau_w is None:
                p = dim
            else:
                p = 1
                while (t[2*p+1:] - t[:-(2*p+1)]).mean() < tau_w and p < N/4:
                    p += 1
                print("p set to", p)

        m = 2*p + 1
        N1 = N - m + 1

        # time differences:
        dt = np.zeros((N1, m))
        for i in range(N1):
            dt[i, :] = t[i:i+m] - t[i+p]

        # filter weights
        # = recursively computed values of orthogonal polynomials:
        r = np.zeros((N1, dim, m))
        for j in range(dim):
            r[:, j, :] = dt**j - (
                r[:, :j, :] * ((dt**j).reshape((N1, 1, m)) * r[:, :j, :]).sum(
                    axis=2).reshape((N1, j, 1))).sum(axis=1)
            r[:, j, :] /= np.sqrt((r[:, j, :]**2).sum(axis=1)).reshape((N1, 1))
        for j in range(dim):
            r[:, j, :] *= factorial(j) / \
                (r[:, j, :] * dt**j).sum(axis=1).reshape((-1, 1))

        # reconstructed state vectors = filtered time series values:
        y = np.zeros((N1, dim))
        for i in range(N1):
            y[i, :] = (r[i, :, :]*x[i:i+m].reshape((1, m))).sum(axis=1)

        return y

    @staticmethod
    def embed_time_series(time_series, dim, tau):
        """
        Return a time series' delay embedding.

        Returns a Numpy array containing a delay embedding of the time series
        using embedding dimension dim and time delay tau.

        :type time_series: 1D array
        :arg time_series: The scalar time series to be embedded.
        :arg int dim: The embedding dimension.
        :arg int tau: The embedding delay.
        :rtype: 2D array (time, dimension)
        :return: the embedded phase space trajectory.
        """
        #  Make sure that dim and tau are Python integers
        dim = int(dim)
        tau = int(tau)
        time_series = to_cy(time_series, FIELD)
        if time_series.ndim > 1:
            time_series = time_series.squeeze(axis=-1)
        assert time_series.ndim == 1
        n_time = time_series.shape[0]

        embedding = np.empty((n_time - (dim - 1) * tau, dim), dtype=FIELD)
        _embed_time_series(n_time, dim, tau, time_series, embedding)
        return embedding

    def permutation_entropy(self, normalize=True):
        """
        Returns the permutation entropy for an embedded time series.
        An embedding of 3 <= embedding dimension <= 7 is recommended for this
        method.

        Reference: [Bandt2002]_

        :rtype: double
        :return: Permutation entropy of the embedded time series
        """
        if self.dim is None or self.tau is None:
            raise ValueError("Permutation Entropy only works for "
                             "one-dimensional embedded time series!")

        # Calculate order from embedding
        temp = self.embedding.argsort(axis=1)
        ranks = temp.argsort(axis=1)

        # Calculate probability distribution
        P = np.unique(ranks, return_counts=True, axis=0)[1]
        P = P / ranks.shape[0]

        entropy = abs(-(P*np.log2(P)).sum())
        if normalize:
            entropy = entropy / np.log2(factorial(self.dim))

        return entropy

    def complexity_entropy(self):
        """
        Returns the complexity entropy for each dimension of the time series.

        Reference: [Ribeiro2011]_

        :rtype: double
        :return: Complexity entropy of the embedded time series
        """

        if self.dim is None or self.tau is None:
            raise ValueError("Complexity Entropy only works for "
                             "one-dimensional embedded time series!")

        # Calculate ranks from embedding
        temp = self.embedding.argsort(axis=1)
        ranks = temp.argsort(axis=1)

        # Calculate probability distribution
        P = np.unique(ranks, return_counts=True, axis=0)[1]
        P = P / ranks.shape[0]

        # Calculate factorial of embedding dimension
        d_fac = factorial(self.dim)

        # Calculate permutation entropy
        S_P = abs(-(P*np.log2(P)).sum())

        # Initialize unit probability distribution
        P_e = np.ones(d_fac) / d_fac

        # Calculate permutation entropy of unit probability distribution
        S_Pe = abs(-(P_e*np.log2(P_e)).sum())

        # Calculate combined probability distribution
        P_comb = np.zeros(d_fac)
        P_comb[:len(P)] = P
        P_comb = (P_comb + P_e) / 2

        # Calculate permutation entropy of combinded probability distribution
        S_PPe = abs(-(P_comb*np.log2(P_comb)).sum())

        # Calculation of maximum Q split into several steps for
        # readability
        Q_max = (d_fac + 1) / d_fac * np.log2(d_fac + 1)
        Q_max = Q_max - 2*np.log2(2*d_fac) + np.log2(d_fac)
        Q_max = - Q_max / 2

        Q = (S_PPe - S_P/2 - S_Pe/2) / Q_max

        complexity_entropy = Q * S_P / np.log2(d_fac)

        return complexity_entropy

    #
    #  Calculate recurrence plot
    #

    @Cached.method(name="the manhattan distance matrix")
    def manhattan_distance_matrix(self):
        """
        Return the manhattan distance matrix from an embedding of a time
        series.

        :rtype: 2D square array
        :return: the manhattan distance matrix.
        """
        (n_time, dim) = self.embedding.shape
        return _manhattan_distance_matrix_rp(n_time, dim, self.embedding)

    @Cached.method(name="the euclidean distance matrix")
    def euclidean_distance_matrix(self):
        """
        Return the euclidean distance matrix from an embedding of a time
        series.

        :rtype: 2D square array
        :return: the euclidean distance matrix.
        """
        (n_time, dim) = self.embedding.shape
        return _euclidean_distance_matrix_rp(n_time, dim, self.embedding)

    @Cached.method(name="the supremum distance matrix")
    def supremum_distance_matrix(self):
        """
        Return the supremum distance matrix from an embedding of a time series.

        :rtype: 2D square Numpy array
        :return: the supremum distance matrix.
        """
        (n_time, dim) = self.embedding.shape
        return _supremum_distance_matrix_rp(n_time, dim, self.embedding)

    def set_fixed_threshold(self, threshold):
        """
        Set the recurrence plot to a fixed threshold.

        Modifies / sets the class variables :attr:`R` and :attr:`N`
        accordingly.

        :arg number threshold: The recurrence threshold.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot at fixed threshold...")

        distance = RecurrencePlot.distance_matrix(self, self.metric)
        n_time = distance.shape[0]
        recurrence = np.zeros((n_time, n_time), dtype="int8")
        recurrence[distance < threshold] = 1
        if self.missing_values:
            #  Write missing value lines and rows to recurrence matrix
            #  NaN flag is not supported by int8 data format -> use 0 here
            recurrence[self.missing_value_indices, :] = 0
            recurrence[:, self.missing_value_indices] = 0

        self.R = recurrence

    def set_fixed_threshold_std(self, threshold_std):
        """
        Set the recurrence plot to a fixed threshold in units of the standard
        deviation of the time series.

        Calculates the absolute threshold and calls
        :meth:`set_fixed_threshold`.

        :arg number threshold_std: The recurrence threshold in units of the
            standard deviation of the time series.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot at fixed threshold in units of "
                  "time series STD...")

        threshold = threshold_std * self.time_series.std()
        RecurrencePlot.set_fixed_threshold(self, threshold)

    def set_fixed_recurrence_rate(self, recurrence_rate):
        """
        Set the recurrence plot to a fixed recurrence rate.

        Modifies / sets the class variables :attr:`R` and :attr:`N`
        accordingly.

        :arg number recurrence_rate: The recurrence rate.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot at fixed recurrence rate...")

        distance = RecurrencePlot.distance_matrix(self, self.metric)
        n_time = distance.shape[0]
        threshold = self.threshold_from_recurrence_rate(distance,
                                                        recurrence_rate)
        recurrence = np.zeros((n_time, n_time), dtype="int8")
        recurrence[distance < threshold] = 1
        self.R = recurrence

    def set_fixed_local_recurrence_rate(self, local_recurrence_rate):
        """
        Set the recurrence plot to a fixed local recurrence rate.

        This results in a fixed number of recurrences for each state vector,
        i.e., all state vectors have the same number of recurrences.  Modifies
        / sets the class variables :attr:`R` and :attr:`N` accordingly.

        .. note::
           The resulting recurrence matrix :math:`R` is generally asymmetric!

        :arg number local_recurrence_rate: The local recurrence rate.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot at fixed "
                  "local recurrence rate...")

        distance = RecurrencePlot.distance_matrix(self, self.metric)
        n_time = distance.shape[0]
        recurrence = np.zeros((n_time, n_time), dtype="int8")
        for i in range(n_time):
            #  Get threshold for state vector i to obtain fixed local
            #  recurrence rate
            local_threshold = self.threshold_from_recurrence_rate(
                distance[i, :], local_recurrence_rate)
            #  Thresholding the distance matrix for column i
            recurrence[i, distance[i, :] < local_threshold] = 1
        self.R = recurrence

    def set_adaptive_neighborhood_size(self, adaptive_neighborhood_size,
                                       order=None):
        """
        Construct recurrence plot using the :index:`adaptive neighborhood
        size <single: adaptive neighborhood size; recurrence plot>` algorithm
        introduced in [Xu2008]_.

        The exact algorithm was deduced from private correspondence with the
        authors, as the description given in the above mentioned is not correct
        or at least ambiguous.

        Modifies / sets the class variables :attr:`R` and :attr:`N`
        accordingly.

        :arg number adaptive_neighborhood_size: The number of adaptive nearest
            neighbors (recurrences) assigned to each state vector.
        :type order: 1D array of int32
        :arg order: The indices of state vectors in the order desired for
            processing by the algorithm. The standard order is :math:`1,...,N`.
        """
        if self.silence_level <= 1:
            print("Calculating recurrence plot using the "
                  "adaptive neighborhood size algorithm...")

        distance = RecurrencePlot.distance_matrix(self, self.metric)

        #  Get indices that would sort the distance matrix.
        #  sorted_neighbors[i,j] contains the index of the jth nearest neighbor
        #  of i. Sorting order is very important here!
        sorted_neighbors = to_cy(distance.argsort(axis=1), NODE)

        n_time = distance.shape[0]
        recurrence = np.zeros((n_time, n_time), dtype=LAG)

        #  Set processing order of state vectors
        if order is None:
            order = np.arange(n_time, dtype=NODE)
        else:
            order = to_cy(order, NODE)

        _set_adaptive_neighborhood_size(n_time, adaptive_neighborhood_size,
                                        sorted_neighbors, order, recurrence)
        self.R = recurrence

    @staticmethod
    def threshold_from_recurrence_rate(distance, recurrence_rate: float):
        """
        Return the threshold for recurrence plot construction given the
        recurrence rate.

        Be aware, that the returned threshold can only approximately give the
        desired recurrence rate. The accuracy depends on the distribution of
        values in the given distance matrix :math:`D`.

        :type distance: 2D square array.
        :arg distance: The phase space distance matrix :math:`D`.
        :arg number recurrence_rate: The desired recurrence rate.
        :return number: the recurrence threshold corresponding to the desired
            recurrence rate.
        """
        #  Flatten and sort distance matrix
        flat_distance = distance.flatten()
        flat_distance.sort()

        #  Get threshold
        assert 0 <= recurrence_rate <= 1
        N = len(flat_distance)
        threshold = flat_distance[int(recurrence_rate * (N - 1))]

        #  Clean up
        del flat_distance

        return threshold

    @staticmethod
    def threshold_from_recurrence_rate_fast(distance, recurrence_rate,
                                            rr_precision=0.001):
        """
        Return the threshold for recurrence plot construction given the
        recurrence rate.

        The threshold yielding a given recurrence_rate is approximated using a
        randomly selected rr_precision percent of the distance matrix' entries.
        Hence, the expected accuracy is lower than that achieved by using
        :index:`threshold_from_recurrence_rate`.

        :type distance: 2D square array.
        :arg distance: The phase space distance matrix :math:`D`.
        :arg number recurrence_rate: The desired recurrence rate.
        :arg number rr_precision: The desired precision of recurrence rate
            estimation.
        :return number: the recurrence threshold corresponding to the desired
            recurrence rate.
        """
        #  Get number of distances to be randomly chosen
        n_samples = int(rr_precision * distance.size)

        #  Get number of phase space points
        n_time = distance.shape[0]

        # vectorized version
        i = np.random.randint(n_time, size=n_samples)
        j = np.random.randint(n_time, size=n_samples)
        samples = distance[i, j]

        #  Sort and get threshold
        samples.sort()
        threshold = samples[int(recurrence_rate * n_samples)]
        return threshold

    @staticmethod
    def bootstrap_distance_matrix(embedding, metric, M):
        """
        Return bootstrap samples from distance matrix.

        :type embedding: 2D array (time, embedding dimension)
        :arg embedding: The phase space trajectory.
        :arg str metric: The metric for measuring distances in phase space
            ("manhattan", "euclidean", "supremum").
        :arg int M: Number of bootstrap samples
        :rtype: 1D array ("float32")
        :return: the bootstrap samples from distance matrix.
        """
        #  Prepare
        M = int(M)
        embedding = to_cy(embedding, DFIELD)
        distances = np.zeros(M, dtype=DFIELD)
        (n_time, dim) = embedding.shape

        if metric == "manhattan":
            _bootstrap_distance_matrix_manhattan(n_time, dim, embedding,
                                                 distances, M)

        elif metric == "euclidean":
            _bootstrap_distance_matrix_euclidean(n_time, dim, embedding,
                                                 distances, M)

        elif metric == "supremum":
            _bootstrap_distance_matrix_supremum(n_time, dim, embedding,
                                                distances, M)
        return distances

    #
    #  Recurrence quantification analysis (RQA)
    #

    def rqa_summary(self, l_min=2, v_min=2):
        """
        Return a selection of RQA measures.

        The selection consists of the recurrence rate :math:`RR`, the
        determinism :math:`DET`, the average diagonal line length :math:`L` and
        the laminarity :math:`LAM`.

        :arg int l_min: The minimum diagonal line length.
        :arg int v_min: The minimum vertical line length.
        :rtype: Python dictionary
        :return: a selection of RQA measures.
        """
        RR = self.recurrence_rate()
        DET = self.determinism(l_min)
        L = self.average_diaglength(l_min)
        LAM = self.laminarity(v_min)

        return {"RR": RR, "DET": DET, "L": L, "LAM": LAM}

    def recurrence_rate(self):
        """
        Return the :index:`recurrence rate` :math:`RR`.

        RR gives the percentage of black dots in the recurrence plot.

        :return number: the recurrence rate :math:`RR`.
        """
        N = self.N
        if not self.sparse_rqa:
            R = self.recurrence_matrix()
            RR = R.sum() / N ** 2
        elif self.metric == "supremum":
            RR = (self.vertline_dist() * np.arange(1, N + 1)).sum() / N ** 2
        else:
            raise NotImplementedError(
                "Sequential RQA is currently only available for "
                "fixed threshold and the supremum metric.")
        return RR

    def recurrence_probability(self, lag=0):
        """
        Return the recurrence probability. This is the probability, that
        the trajectory is recurrent after 'lag' time steps.

        Contributed by Jan H. Feldhoff.

        :return number: the recurrence probability
        """
        R = self.recurrence_matrix()
        N = self.N
        SUM = np.sum(np.diag(R, lag))

        return SUM / float(N-lag)

    #
    #  RQA measures based on black diagonal lines
    #

    @Cached.method(attrs=(
        "metric", "threshold", "missing_values", "sparse_rqa"))
    def diagline_dist(self):
        """
        Return the :index:`frequency distribution of diagonal line lengths
        <triple: frequency distribution; diagonal; line length>`
        :math:`P(l-1)`.

        Note that entry :math:`P(l-1)` contains the number of
        :index:`diagonal lines <pair: diagonal; lines>` of length :math:`l`.
        Thus, :math:`P(0)` counts lines of length :math:`1`,
        :math:`P(1)` counts lines of length :math:`2`, asf.
        The main diagonal is not counted,
        hence :math:`P(N)` will always be :math:`0`.

        .. note::
           Experimental handling of missing values. Diagonal lines
           touching lines and blocks of missing entries in the
           recurrence matrix are not counted.

        :rtype: 1D array (int32)
        :return: the frequency distribution of diagonal line lengths
            :math:`P(l-1)`.
        """
        #  Prepare
        n_time = self.N
        diagline = np.zeros(n_time, dtype=NODE)

        if not self.sparse_rqa:
            #  Get recurrence matrix
            recmat = self.recurrence_matrix()

            if self.missing_values:
                mv_indices = self.missing_value_indices
                _diagline_dist_missingvalues(
                    n_time, diagline, recmat, mv_indices)
            else:
                _diagline_dist(n_time, diagline, recmat)

        #  Calculations for sequential RQA
        elif self.metric == "supremum" and self.threshold is not None:
            #  Get embedding
            embedding = self.embedding
            #  Get time series dimension
            dim = embedding.shape[1]
            #  Get threshold
            eps = float(self.threshold)

            if self.missing_values:
                mv_indices = self.missing_value_indices
                _diagline_dist_sequential_missingvalues(
                    n_time, diagline, mv_indices, embedding, eps, dim)
            else:
                _diagline_dist_sequential(
                    n_time, diagline, embedding, eps, dim)

        else:
            raise NotImplementedError(
                "Sequential RQA is currently only available for "
                "fixed threshold and the supremum metric.")

        #  Function just runs over the upper triangular matrix
        return 2 * diagline

    @staticmethod
    def rejection_sampling(dist, M):
        """
        Rejection sampling of discrete frequency distribution.

        Use simple rejection sampling algorithm for computing a resampled
        version of a given frequency distribution with discrete support.

        :type dist: 1D array (integer)
        :arg dist: discrete frequency distribution
        :arg int M: number of resamplings
        :rtype: 1D array (integer)
        :return: the resampled frequency distribution.
        """
        #  Get number of support points
        N = len(dist)

        #  Prepare
        resampled_dist = np.zeros(N, dtype=NODE)

        #  Prescribed distribution
        dist = to_cy(dist, DFIELD)
        #  Normalize distribution
        dist /= dist.sum()

        _rejection_sampling(dist, resampled_dist, N, M)
        return resampled_dist

    def resample_diagline_dist(self, M):
        """
        Return resampled frequency distribution of diagonal lines.

        The resampled frequency distribution can be used for obtaining
        confidence bounds on diagonal line based RQA measures. This is
        described in detail in [Schinkel2009]_.

        Concerning the choice of the number of resamplings, Schinkel et al.
        write: "The number of resamplings is not generally agreed upon but
        common guidelines suggest values between 800 and 1500."

        :arg int M: number of resamplings
        :rtype: 1D array (integer)
        :return: the resampled frequency distribution of diagonal lines.
        """
        #  Get original distribution of diagonal lines
        diagline = self.diagline_dist()

        #  Get maximal diagonal line length
        L_max = self.max_diaglength()

        #  Get resampled distribution
        if L_max == 0:
            resampled_dist = diagline
        else:
            resampled_dist = np.zeros(len(diagline), dtype=NODE)
            resampled_dist[:L_max] = RecurrencePlot.\
                rejection_sampling(diagline[:L_max], M)

        return resampled_dist

    def max_diaglength(self):
        """
        Return diagonal line-based RQA measure :index:`maximum diagonal line
        length <triple: maximum; diagonal; line length>` :math:`L_max`.

        :math:`L_max` is defined as the maximal length of a diagonal line in
        the recurrence matrix.

        :return number: the maximal diagonal line length :math:`L_max`.
        """
        return 1 + np.nonzero(self.diagline_dist())[0].max(initial=-1)

    def determinism(self, l_min=2, resampled_dist=None):
        """
        Return diagonal line-based RQA measure :index:`determinism <pair: RQA;
        determinism>` :math:`DET`.

        :math:`DET` is defined as the ratio of recurrence points that form
        diagonal structures (of at least length :math:`l_min`) to all
        recurrence points.

        :arg number l_min: The minimum diagonal line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of diagonal lines
        :return number: the determinism :math:`DET`.
        """
        #  Use resampled distribution of diagonal lines if provided
        diagline = (self.diagline_dist() if resampled_dist is None
                    else resampled_dist)
        n_time = self.N

        #  Number of recurrence points that form diagonal structures (of at
        #  least length l_min)
        partial_sum = np.arange(l_min, n_time+1) @ diagline[l_min-1:]

        #  Number of all recurrence points that form diagonal lines (except
        #  the main diagonal)
        full_sum = np.arange(1, n_time+1) @ diagline

        return partial_sum / float(full_sum + self._epsilon)

    def average_diaglength(self, l_min=2, resampled_dist=None):
        """
        Return diagonal line-based RQA measure :index:`average diagonal line
        length <triple: average; diagonal; line length>` :math:`L`.

        :math:`L` is defined as the average length of diagonal lines (of at
        least length :math:`l_min`).

        :arg number l_min: The minimum diagonal line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of diagonal lines
        :return number: the average diagonal line length :math:`L`.
        """
        #  Use resampled distribution of diagonal lines if provided
        diagline = (self.diagline_dist() if resampled_dist is None
                    else resampled_dist)
        n_time = self.N

        #  Number of recurrence points that form diagonal structures (of at
        #  least length l_min)
        partial_sum = np.arange(l_min, n_time+1) @ diagline[l_min-1:]

        #  Total number of diagonal lines of at least length l_min
        number_diagline = diagline[l_min-1:].sum()

        return partial_sum / float(number_diagline + self._epsilon)

    def diag_entropy(self, l_min=2, resampled_dist=None):
        """
        Return diagonal line-based RQA measure :index:`diagonal line entropy
        <pair: diagonal; line entropy>` :math:`ENTR`.

        :math:`ENTR` is defined as the entropy of the probability to find a
        diagonal line of exactly length l in the RP - reflects the complexity
        of the RP with respect to diagonal lines.

        :arg number l_min: The minimal diagonal line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of diagonal lines
        :return number: the diagonal line-based entropy :math:`ENTR`.
        """
        #  Use resampled distribution of diagonal lines if provided
        diagline = (self.diagline_dist() if resampled_dist is None
                    else resampled_dist)

        #  Creates a reduced array of the values (not 0) of the diagonal line
        #  length (langer than l_min)
        diagline = diagline[l_min-1:]
        diagline = np.extract(diagline != 0, diagline)

        #  Normalized array of the number of all diagonal lines = probability
        #  of diagonal line length
        diagnorm = diagline / float(diagline.sum() + self._epsilon)

        return -(diagnorm * np.log(diagnorm)).sum()

    #
    #  RQA measures based on black vertical lines
    #

    @Cached.method(attrs=(
        "metric", "threshold", "missing_values", "sparse_rqa"))
    def vertline_dist(self):
        """
        Return the :index:`frequency distribution of vertical line lengths
        <triple: frequency distribution; vertical; line length>`
        :math:`P(v-1)`.

        Note that entry :math:`P(v-1)` contains the number of
        :index:`vertical lines <pair: vertical; lines>` of length :math:`v`.
        Thus, :math:`P(0)` counts lines of length :math:`1`,
        :math:`P(1)` counts lines of length :math:`2`, asf.

        :rtype: 1D array (int32)
        :return: the frequency distribution of vertical line lengths
            :math:`P(v-1)`.
        """
        #  Prepare
        n_time = self.N
        vertline = np.zeros(n_time, dtype=NODE)

        if not self.sparse_rqa:
            #  Get recurrence matrix
            recmat = self.recurrence_matrix()

            if self.missing_values:
                mv_indices = self.missing_value_indices
                _vertline_dist_missingvalues(
                    n_time, vertline, recmat, mv_indices)
            else:
                _vertline_dist(n_time, vertline, recmat)

        #  Calculations for sequential RQA
        elif self.metric == "supremum" and self.threshold is not None:
            #  Get embedding
            embedding = self.embedding
            #  Get time series dimension
            dim = embedding.shape[1]
            #  Get threshold
            eps = float(self.threshold)

            if self.missing_values:
                mv_indices = self.missing_value_indices
                _vertline_dist_sequential_missingvalues(
                    n_time, vertline, mv_indices, embedding, eps, dim)

            else:
                _vertline_dist_sequential(
                    n_time, vertline, embedding, eps, dim)

        else:
            raise NotImplementedError(
                "Sequential RQA is currently only available for "
                "fixed threshold and the supremum metric.")

        #  Function covers the whole recurrence matrix
        return vertline

    def resample_vertline_dist(self, M):
        """
        Return resampled frequency distribution of vertical lines.

        The resampled frequency distribution can be used for obtaining
        confidence bounds on vertical line based RQA measures. This is
        described in detail in [Schinkel2009]_.

        Concerning the choice of the number of resamplings, Schinkel et al.
        write: "The number of resamplings is not generally agreed upon but
        common guidelines suggest values between 800 and 1500."

        :arg int M: number of resamplings
        :rtype: 1D array (integer)
        :return: the resampled frequency distribution of vertical lines.
        """
        #  Get original distribution of vertical lines
        vertline = self.vertline_dist()

        #  Get maximal vertical line length
        L_max = self.max_vertlength()

        #  Get resampled distribution
        if L_max == 0:
            resampled_dist = vertline
        else:
            resampled_dist = np.zeros(len(vertline), dtype=NODE)
            resampled_dist[:L_max] = RecurrencePlot.\
                rejection_sampling(vertline[:L_max], M)

        return resampled_dist

    def max_vertlength(self):
        """
        Return vertical line-based RQA measure :index:`maximal vertical line
        length <triple: maximum; vertical; line length>` :math:`V_max`.

        :math:`V_max` is defined as the maximal length of a vertical line of
        the recurrence matrix.

        :return number: the maximal vertical line length :math:`V_max`.
        """
        return 1 + np.nonzero(self.vertline_dist())[0].max(initial=-1)

    def laminarity(self, v_min=2, resampled_dist=None):
        """
        Return vertical line-based RQA measure :index:`laminarity` :math:`LAM`.

        :math:`LAM` is defined as the ratio of recurrence points that form
        vertical structures (of at least length :math:`v_min`) to all
        recurrence points.

        :arg number v_min: The minimal vertical line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of vertical lines
        :return number: the laminarity :math:`LAM`.
        """
        #  Use resampled distribution of vertical lines if provided
        vertline = (self.vertline_dist() if resampled_dist is None
                    else resampled_dist)
        n_time = self.N

        #  Number of recurrence points that form vertical structures (of at
        #  least length v_min)
        partial_sum = np.arange(v_min, n_time+1) @ vertline[v_min-1:]

        #  Number of all recurrence points that form vertical lines
        full_sum = np.arange(1, n_time+1) @ vertline

        return partial_sum / float(full_sum + self._epsilon)

    def average_vertlength(self, v_min=2, resampled_dist=None):
        """
        Return vertical line-based RQA measure :index:`average vertical line
        length <triple: average; vertical; line length>` :math:`TT`.

        :math:`TT` is defined as the average vertical line length (of at least
        length :math:`v_min`) and is also called :index:`trapping time`
        :math:`TT`.

        :arg number v_min: The minimal vertical line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of vertical lines
        :return number: the trapping time :math:`TT`.
        """
        #  Use resampled distribution of vertical lines if provided
        vertline = (self.vertline_dist() if resampled_dist is None
                    else resampled_dist)
        n_time = self.N

        #  Number of recurrence points that form vertical structures (of at
        #  least length v_min)
        partial_sum = np.arange(v_min, n_time+1) @ vertline[v_min-1:]

        #  Total number of vertical lines of at least length v_min
        number_vertline = vertline[v_min-1:].sum()

        return partial_sum / (float(number_vertline) + self._epsilon)

    def trapping_time(self, v_min=2, resampled_dist=None):
        """
        Alias for :meth:`average_vertlength` (see description there).
        """
        return self.average_vertlength(v_min, resampled_dist)

    def vert_entropy(self, v_min=2, resampled_dist=None):
        """
        Return vertical line-based RQA measure :index:`vertical line entropy
        <pair: vertical; line entropy>`.

        It is defined as the entropy of the probability to find a vertical line
        of exactly length l in the RP - reflects the complexity of the RP with
        respect to vertical lines.

        :arg int v_min: The minimal vertical line length.
        :type resampled_dist: 1D array (integer)
        :arg resampled_dist: resampled frequency distribution of vertical lines
        :return number: the vertical line-based entropy.
        """
        #  Use resampled distribution of vertical lines if provided
        if resampled_dist is None:
            vertline = self.vertline_dist()
        else:
            vertline = resampled_dist

        #  Creates a reduced array of the values (not 0) of the vertical line
        #  length (langer than v_min)
        vertline = vertline[v_min-1:]
        vertline = np.extract(vertline != 0, vertline)

        #  Normalized array of the number of all vertical lines = probability
        #  of vertical line length
        vertline_normed = vertline / float(vertline.sum() + self._epsilon)

        return -(vertline_normed * np.log(vertline_normed)).sum()

    #
    #  RQA measures based on white vertical lines
    #

    def white_vertline_dist(self):
        """
        Return the :index:`frequency distribution of white vertical line
        lengths <triple: frequency distribution; white vertical; line length>`
        :math:`P(w-1)`.

        Note that entry :math:`P(w-1)` contains the number of
        :index:`white vertical lines <pair: white vertical; lines>` of length
        :math:`w`. Thus, :math:`P(0)` counts lines of length :math:`1`,
        :math:`P(1)` counts lines of length :math:`2`, asf.

        The length of a white vertical line in a recurrence plot corresponds to
        the time the system takes to return close to an earlier state.

        :rtype: 1D array (int32)
        :return: the frequency distribution of white vertical line lengths
            :math:`P(w-1)`.
        """
        R = self.recurrence_matrix()
        n_time = self.N
        white_vertline = np.zeros(n_time, dtype=NODE)
        _white_vertline_dist(n_time, white_vertline, R)

        #  Function covers the whole recurrence matrix
        return white_vertline

    def max_white_vertlength(self):
        """
        Return white vertical line-based RQA measure :index:`maximal white
        vertical line length <triple: maximum; white vertical; line length>`.

        It is defined as the maximal length of a white vertical line of
        the recurrence matrix and corresponds to the maximum recurrence time
        occuring in the time series.

        :return number: the maximal white vertical line length.
        """
        return 1 + np.nonzero(self.white_vertline_dist())[0].max(initial=-1)

    def average_white_vertlength(self, w_min=1):
        """
        Return white vertical line-based RQA measure :index:`average white
        vertical line length <triple: average; white vertical; line length>`.

        It is defined as the average white vertical line length (of at least
        length :math:`w_min`) and is also called :index:`mean recurrence time`.

        Reference: [Ngamga2007]_.

        :arg number w_min: The minimal white vertical line length.
        :return number: the mean recurrence time.
        """
        white_vertline = self.white_vertline_dist()
        n_time = self.N

        #  Number of recurrence points that form white vertical structures
        #  (of at least length w_min)
        partial_sum = np.arange(w_min, n_time+1) @ white_vertline[w_min-1:]

        #  Total number of white vertical lines of at least length v_min
        number_white_vertline = white_vertline[w_min-1:].sum()

        return partial_sum / float(number_white_vertline + self._epsilon)

    def mean_recurrence_time(self, w_min=1):
        """
        Alias for :meth:`average_white_vertlength` (see description there).
        """
        return self.average_white_vertlength(w_min)

    def white_vert_entropy(self, w_min=1):
        """
        Return white vertical line-based RQA measure :index:`white vertical
        line entropy <pair: white vertical; line entropy>`.

        It is defined as the entropy of the probability to find a white
        vertical line of exactly length l in the RP - reflects the complexity
        of the RP with respect to white vertical lines (recurrence times).

        :arg int w_min: Minimal white vertical line length (recurrence time).
        :return number: the white vertical line-based entropy.
        """
        #  Creates a reduced array of the values (not 0) of the vertical line
        #  length (langer than v_min)
        white_vertline = self.white_vertline_dist()
        white_vertline = white_vertline[w_min-1:]
        white_vertline = np.extract(white_vertline != 0, white_vertline)

        #  Normalized array of the number of all vertical lines = probability
        #  of vertical line length
        white_vertline_normed = white_vertline / float(
            white_vertline.sum() + self._epsilon)

        return -(white_vertline_normed * np.log(white_vertline_normed)).sum()

    #
    #  Methods for recurrence-based surrogates
    #

    def twins(self, min_dist=7):
        """
        Return list of the :index:`twins <pair: twins; recurrence plot>` of
        each state vector based on the recurrence matrix.

        Two state vectors are said to be twins if they share the same
        recurrences, i.e., if the corresponding rows or columns in the
        recurrence plot are identical.

        References: [Thiel2006]_, [Marwan2007]_.

        :arg number min_dist: The minimum temporal distance for twins.
        :return [[number]]: the list of twins for each state vector in the time
            series.
        """
        if self.silence_level <= 1:
            print("Finding twins based on recurrence matrix...")

        #  Initialize
        twins = []
        N = self.N

        #  Get current recurrence matrix
        R = self.recurrence_matrix()
        #  Get number of neighbors for each state vector
        nR = R.sum(axis=0)

        _twins_r(min_dist, N, R, nR, twins)
        return twins

    def twin_surrogates(self, n_surrogates=1, min_dist=7):
        """
        Generate surrogates based on the current (embedded) time series
        :attr:`embedding` using the :index:`twin surrogate` method.

        The twins surrogates have the same dimensionality as the (embedded)
        trajectory used for constructing the recurrence plot. If scalar
        surrogate time series are desired, any component of the twin surrogate
        trajectory may be isolated.

        Twin surrogates share linear and nonlinear properties with the original
        time series, since they correspond to realizations of trajectories of
        the same dynamical systems with different initial conditions.

        References: [Thiel2006]_ [*], [Marwan2007]_.

        :arg number min_dist: The minimum temporal distance for twins.
        :arg int n_surrogates: The number of twin surrogate trajectories to be
            returned.
        :rtype: 3D array (surrogate number, time, dimension)
        :return: the twin surrogate trajectories.
        """
        #  The algorithm proceeds in two steps:
        #  1. Use the algorithm proposed in [*] to find twins
        #  2. Reconstruct one-dimensional twin surrogate time series
        if self.silence_level <= 1:
            print("Generating twin surrogates...")

        #  Collect
        N = self.N
        embedding = self.embedding
        dim = embedding.shape[1]
        twins = self.twins(min_dist)

        return _twin_surrogates_r(n_surrogates, N, dim, twins, embedding)
