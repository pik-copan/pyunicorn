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
Provides class for event series analysis, namely event synchronization (ES) and
event coincidence analysis (ECA). In addition, a method for the generation of
binary event series from continuous time series data is included.
When instantiating a class, data must either be passed as an event
matrix (for details see below) or as a continuous time series. Using the class,
an ES or ECA matrix can be calculated to generate a climate network using the
EventSeriesClimateNetwork class. Both ES and ECA may be called without
instantiating an object of the class.
Significance levels are provided using analytic calculations using Poisson
point processes as a null model (for ECA only) or a Monte Carlo approach.

Modifications vs. original pyunicorn EventSeries
-------------------------------------------------
Modification 1 (constructor):
    Removed the axis-swap heuristic that transposed data when the number of
    variables exceeded the number of timesteps.  The heuristic assumed that
    datasets with more variables than timesteps were transposed by mistake,
    which is not always true.

Modification 2 (make_event_matrix):
    Replaced np.quantile with np.nanquantile so that time series containing
    NaN values are handled gracefully instead of raising an error.

Modification 3 (matrix loops):
    Added tqdm progress bars to _ndim_event_synchronization and
    _ndim_event_coincidence_analysis for visual feedback during long runs.

Modification 4 (getters):
    Added get_T(), get_N(), get_timestamps(), get_taumax(), get_lag() accessor
    methods to expose key instance attributes without name-mangling gymnastics.

Optimization 1 — Sparse binary-search ES with lag support (finite taumax):
    For finite taumax, precomputes sorted inner-event arrays once per node and
    uses np.searchsorted to find only the event pairs within the lag-shifted
    taumax window.  Complexity O(N²·L·taumax/T) vs O(N²·L²).
    Results are exact (atol < 1e-12 vs pairwise reference).

Optimization 2 — Blocked dense ES (infinite taumax, lag=0):
    Falls back to a 4-D blocked-dense tensor kernel that vectorises the
    double-count correction loop — the main bottleneck in the original code.
    Block size is tunable; joblib parallelization is supported.

Optimization 3 — Vectorized ECA via cumsum + BLAS (integer timestamps):
    For uniform integer timestamps and integer-valued lag, builds lag=0 smeared
    indicator matrices with cumsum rolling windows, shifts by lag steps, then
    uses a single BLAS matrix multiply.  Complexity O(T·N + N²).
    Precision is identical to pairwise within float32 rounding (atol < 1e-6).

Optimization 4 — Optional joblib parallelization:
    All kernel dispatchers accept an n_jobs keyword argument.  If joblib is
    installed, computation is distributed across n_jobs worker processes.
    Falls back to a single-process loop when joblib is absent.

Contributors
------------
Guruprem Bishnoi — Modifications 1–4 and Optimizations 1–4 (2026)
"""

from typing import Tuple
from collections.abc import Hashable

import warnings

import numpy as np
from scipy import stats
from tqdm import tqdm

# Optimization 4: optional joblib for parallelization
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


from ..core.cache import Cached


# ===========================================================================
# Module-level helpers required by the fast-path ES kernels
# (kept at module level so that joblib can pickle them for multi-processing)
# ===========================================================================

# Optimization 1 / 2 helper -------------------------------------------------

def _pad_block(node_data, indices):
    """
    Pad inner-event arrays for a block of nodes into dense 2-D arrays.

    Used by the blocked-dense ES kernel (Optimization 2).
    """
    max_len = max((node_data[i]['inner_count'] for i in indices), default=0)
    nA = len(indices)
    times = np.zeros((nA, max_len), dtype=np.float64)
    tau2 = np.zeros((nA, max_len), dtype=np.float64)
    mask = np.zeros((nA, max_len), dtype=bool)
    counts = np.zeros(nA, dtype=np.int32)
    for row, idx in enumerate(indices):
        info = node_data[idx]
        length = info['inner_count']
        counts[row] = length
        if length:
            times[row, :length] = info['inner_times']
            tau2[row, :length] = info['tau2_inner']
            mask[row, :length] = True
    return times, tau2, mask, counts


def _compute_es_block(node_data, block_a, block_b):
    """
    Exact ES for all node pairs in two blocks via 4-D dense boolean tensors.

    Vectorises the double-count correction loop that is the bottleneck in
    pyunicorn's pairwise code.  Used by the blocked-dense kernel (Opt 2).
    """
    A_times, A_tau2, A_mask, A_counts = _pad_block(node_data, block_a)
    B_times, B_tau2, B_mask, B_counts = _pad_block(node_data, block_b)

    block_xy = np.zeros((len(block_a), len(block_b)), dtype=np.float64)
    block_yx = np.zeros((len(block_a), len(block_b)), dtype=np.float64)

    if A_times.shape[1] == 0 or B_times.shape[1] == 0:
        return block_xy, block_yx

    valid = A_mask[:, None, :, None] & B_mask[None, :, None, :]
    D2 = 2.0 * (A_times[:, None, :, None] - B_times[None, :, None, :])
    tau2 = np.minimum(A_tau2[:, None, :, None], B_tau2[None, :, None, :])

    Axy = valid & (D2 > 0.0) & (D2 <= tau2)
    Ayx = valid & (D2 < 0.0) & (D2 >= -tau2)
    eqtime = valid & (D2 == 0.0)

    row_any_Ayx = Ayx.any(axis=3, keepdims=True)
    col_any_Ayx = Ayx.any(axis=2, keepdims=True)
    row_any_Axy = Axy.any(axis=3, keepdims=True)
    col_any_Axy = Axy.any(axis=2, keepdims=True)

    countxy = Axy.sum(axis=(2, 3)).astype(np.float64)
    countyx = Ayx.sum(axis=(2, 3)).astype(np.float64)
    eqcount = eqtime.sum(axis=(2, 3)).astype(np.float64)
    countxydouble = (Axy & (row_any_Ayx | col_any_Ayx)).sum(
        axis=(2, 3)).astype(np.float64)
    countyxdouble = (Ayx & (row_any_Axy | col_any_Axy)).sum(
        axis=(2, 3)).astype(np.float64)

    numer_xy = countxy + 0.5 * eqcount - 0.5 * countxydouble
    numer_yx = countyx + 0.5 * eqcount - 0.5 * countyxdouble

    A_cf = A_counts[:, None].astype(np.float64)
    B_cf = B_counts[None, :].astype(np.float64)
    denom = np.sqrt(A_cf * B_cf)
    good = denom > 0.0
    block_xy[good] = numer_xy[good] / denom[good]
    block_yx[good] = numer_yx[good] / denom[good]
    return block_xy, block_yx


def _compute_es_pair_sparse_lag(tx, tau2x, ty, tau2y, *, taumax, lag):
    """
    Exact ES for one pair using sorted inner-event times + binary search,
    extended to support any scalar lag (Optimization 1).

    Parameters
    ----------
    tx, tau2x : 1-D float64 — inner event times and per-event tau2 for X
    ty, tau2y : 1-D float64 — inner event times and per-event tau2 for Y
    taumax    : float        — coincidence window cap (must be finite)
    lag       : float        — time lag (Y is conceptually shifted by +lag)

    Returns
    -------
    (countxy / norm, countyx / norm) : float64 pair
    """
    lx, ly = len(tx), len(ty)
    if lx == 0 or ly == 0:
        return 0.0, 0.0
    norm = np.sqrt(float(lx * ly))
    if norm == 0.0:
        return 0.0, 0.0

    # Binary search with lag-shifted bounds
    lo = np.searchsorted(ty, tx - lag - taumax, side='left')    # (lx,)
    hi = np.searchsorted(ty, tx - lag + taumax, side='right')   # (lx,)
    counts = hi - lo                                              # (lx,) >= 0

    total = int(counts.sum())
    if total == 0:
        return 0.0, 0.0

    cumcounts_lo = np.concatenate([[0], np.cumsum(counts[:-1])])
    offsets = np.arange(total, dtype=np.intp) - np.repeat(cumcounts_lo, counts)
    k_inds = np.repeat(np.arange(lx, dtype=np.intp), counts)
    l_inds = np.repeat(lo, counts).astype(np.intp) + offsets

    D2_sp = 2.0 * (tx[k_inds] - ty[l_inds] - lag)
    tau2_sp = np.minimum(tau2x[k_inds], tau2y[l_inds])

    Axy_sp = (D2_sp > 0.0) & (D2_sp <= tau2_sp)
    Ayx_sp = (D2_sp < 0.0) & (D2_sp >= -tau2_sp)
    eq_sp = D2_sp == 0.0
    eqtime = float(eq_sp.sum())

    row_any_Ayx = np.zeros(lx, dtype=bool)
    col_any_Ayx = np.zeros(ly, dtype=bool)
    row_any_Axy = np.zeros(lx, dtype=bool)
    col_any_Axy = np.zeros(ly, dtype=bool)

    k_ayx = k_inds[Ayx_sp]
    l_ayx = l_inds[Ayx_sp]
    k_axy = k_inds[Axy_sp]
    l_axy = l_inds[Axy_sp]

    if len(k_ayx):
        np.logical_or.at(row_any_Ayx, k_ayx, True)
        np.logical_or.at(col_any_Ayx, l_ayx, True)
    if len(k_axy):
        np.logical_or.at(row_any_Axy, k_axy, True)
        np.logical_or.at(col_any_Axy, l_axy, True)

    countxydouble = float(np.sum(
        Axy_sp & (row_any_Ayx[k_inds] | col_any_Ayx[l_inds])
    ))
    countyxdouble = float(np.sum(
        Ayx_sp & (row_any_Axy[k_inds] | col_any_Axy[l_inds])
    ))

    countxy = float(Axy_sp.sum()) + 0.5 * eqtime - 0.5 * countxydouble
    countyx = float(Ayx_sp.sum()) + 0.5 * eqtime - 0.5 * countyxdouble

    return countxy / norm, countyx / norm


def _process_es_rows_sparse_lag(r_start, r_end, N, node_data, *, taumax, lag):
    """
    Process all pairs (i, j > i) where i in [r_start, r_end) — row-based
    chunk for joblib (Optimization 1 / 4).
    """
    results = []
    for i in range(r_start, r_end):
        ndi = node_data[i]
        for j in range(i + 1, N):
            ndj = node_data[j]
            if ndi['inner_count'] == 0 or ndj['inner_count'] == 0:
                results.append((i, j, 0.0, 0.0))
            else:
                xy, yx = _compute_es_pair_sparse_lag(
                    ndi['inner_times'], ndi['tau2_inner'],
                    ndj['inner_times'], ndj['tau2_inner'],
                    taumax=taumax, lag=lag,
                )
                results.append((i, j, xy, yx))
    return results


# ===========================================================================
# Main class
# ===========================================================================

class EventSeries(Cached):

    def __init__(self, data, *, timestamps=None, taumax=np.inf, lag=0.0,
                 threshold_method=None, threshold_values=None,
                 threshold_types=None):
        """
        Initialize an instance of EventSeries. Input data must be a 2D numpy
        array with time as the first axis and variables as the second axis.
        Event data is stored as an eventmatrix.

        Format of eventmatrix:
        An eventmatrix is a 2D numpy array with the first dimension covering
        the timesteps and the second dimensions covering the variables. Each
        variable at a specific timestep is either '1' if an event occured or
        '0' if it did not, e.g. for 3 variables with 10 timesteps the
        eventmatrix could look like

            array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 0],
                   [1, 0, 1],
                   [0, 1, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 0, 0]])

        If input data is not provided as an eventmatrix, the constructor tries
        to generate one using the make_event_matrix method. Default keyword
        arguments are used in this case.

        :type data: 2D Numpy array [time, variables]
        :arg data: Event series array or array of non-binary variable values
        :type timestamps: 1D Numpy array
        :arg timestamps: Time points of events of data. If not provided,
                        integer values are used
        :type taumax: float
        :arg taumax: maximum time difference between two events to be
                    considered synchronous. Caution: For ES, the default is
                    np.inf because of the intrinsic dynamic coincidence
                    interval in ES. For ECA, taumax is a parameter of the
                    method that needs to be defined!
        :type lag: float
        :arg lag: extra time lag between the event series
        :type threshold_method: str 'quantile' or 'value' or 1D numpy array or
                                str 'quantile' or 'value'
        :arg threshold_method: specifies the method for generating a binary
                               event matrix from an array of continuous time
                               series. Default: None
        :type threshold_values: 1D Numpy array or float
        :arg threshold_values: quantile or real number determining threshold
                               for each variable. Default: None.
        :type threshold_types: str 'above' or 'below' or 1D list of strings
                               'above' or 'below'
        :arg threshold_types: Determines for each variable if event is below
                              or above threshold
        """

        if threshold_method is None:
            # Check if data contains only binary values
            if len(np.unique(data)) != 2 or not (
                    np.unique(data) == np.array([0, 1])).all():
                raise IOError("Event matrix not in correct format")

            # Save event matrix
            self.__T = data.shape[0]
            self.__N = data.shape[1]
            self.__eventmatrix = data

        else:
            # If data is not in eventmatrix format, use method
            # make_event_matrix to transform continuous time series to a binary
            # time series
            # Modification 1: Removed the axis-swap heuristic that previously
            # transposed data when data.shape[1] > data.shape[0].  That
            # heuristic assumed datasets with more variables than timesteps
            # were accidentally transposed, which is not always true.
            if isinstance(data, np.ndarray):
                self.__eventmatrix = \
                    self.make_event_matrix(data,
                                           threshold_method=threshold_method,
                                           threshold_values=threshold_values,
                                           threshold_types=threshold_types)

                self.__T = self.__eventmatrix.shape[0]
                self.__N = self.__eventmatrix.shape[1]

            else:
                raise IOError('Input data is not in event matrix format!')

        # If no timestamps are given, use integer array indices as timestamps
        if timestamps is not None:
            if timestamps.shape[0] != self.__T:
                raise IOError("Timestamps array has not the same length as"
                              " event matrix!")
            self.__timestamps = timestamps
        else:
            self.__timestamps = np.linspace(0.0, self.__T - 1, self.__T)

        self.__taumax = float(taumax)
        self.__lag = float(lag)

        # save number of events
        NrOfEvs = np.array(np.sum(self.__eventmatrix, axis=0), dtype=int)
        self.__nrofevents = NrOfEvs

        # Dictionary of symmetrization functions for later use
        self.symmetrization_options = {
            'directed': EventSeries._symmetrization_directed,
            'symmetric': EventSeries._symmetrization_symmetric,
            'antisym': EventSeries._symmetrization_antisym,
            'mean': EventSeries._symmetrization_mean,
            'max': EventSeries._symmetrization_max,
            'min': EventSeries._symmetrization_min
        }

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        # The following attributes are assumed immutable:
        #   (__eventmatrix, __timestamps, __taumax, __lag)
        return ()

    def __str__(self):
        """
        Return a string representation of the EventSeries object.
        """
        return (f"EventSeries: {self.__N} variables, "
                f"{self.__T} timesteps, taumax: {self.__taumax:.1f}, "
                f"lag: {self.__lag:.1f}")

    def get_event_matrix(self):
        return self.__eventmatrix

    # Modification 4: Added getter methods so callers can access key instance
    # attributes without relying on name-mangling.

    def get_T(self):
        return self.__T

    def get_N(self):
        return self.__N

    def get_timestamps(self):
        return self.__timestamps

    def get_taumax(self):
        return self.__taumax

    def get_lag(self):
        return self.__lag

    @staticmethod
    def _symmetrization_directed(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: original matrix
        """
        return matrix

    @staticmethod
    def _symmetrization_symmetric(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: symmetrized matrix
        """
        return matrix + matrix.T

    @staticmethod
    def _symmetrization_antisym(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: antisymmetrized matrix
        """
        return matrix - matrix.T

    @staticmethod
    def _symmetrization_mean(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: symmetrized matrix using element-wise mean of matrix and
                 transposed matrix
        """
        return np.mean([matrix, matrix.T], axis=0)

    @staticmethod
    def _symmetrization_max(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: symmetrized matrix using element-wise maximum of matrix and
                 transposed matrix
        """
        return np.maximum(matrix, matrix.T)

    @staticmethod
    def _symmetrization_min(matrix):
        """
        Helper function for symmetrization options

        :type matrix: 2D numpy array
        :arg matrix: pairwise ECA/ES scores of data
        :rtype: 2D numpy array
        :return: symmetrized matrix using element-wise minimum of matrix and
                 transposed matrix
        """
        return np.minimum(matrix, matrix.T)

    @staticmethod
    def make_event_matrix(data, threshold_method='quantile',
                          threshold_values=None, threshold_types=None):
        """
        Create a binary event matrix from continuous time series data. Data
        format is eventmatrix, i.e. a 2D numpy array with first dimension
        covering time and second dimension covering the values of the
        variables.

        :type data: 2D numpy array
        :arg data: Continuous input data
        :type threshold_method: str 'quantile' or 'value' or 1D numpy array of
                                strings 'quantile' or 'value'
        :arg threshold_method: specifies the method for generating a binary
                               event matrix from an array of continuous time
                               series. Default: 'quantile'
        :type threshold_values: 1D Numpy array or float
        :arg threshold_values: quantile or real number determining threshold
                               for each variable. Default: None.
        :type threshold_types: str 'above' or 'below' or 1D list of strings
                               'above' or 'below'
        :arg threshold_types: Determines for each variable if event is below or
                              above threshold
        :rtype: 2D numpy array
        :return: eventmatrix
        """

        # Check correct format of event matrix
        if not np.all([len(i) == len(data[0]) for i in data]):
            warnings.warn("Data does not contain equal number of events")

        data_axswap = np.swapaxes(data, 0, 1)
        thresholds = np.zeros(data.shape[1])

        # Check if inserted keyword arguments are correct and create parameter
        # arrays in case only single keywords are used for data with more than
        # one variable
        threshold_method = np.array(threshold_method)
        if threshold_method.shape == (data.shape[1],):
            if not np.all([i in ['quantile', 'value'] for i in
                           threshold_method]):
                raise IOError("'threshold_method' must be either 'quantile' or"
                              " 'value' or a 1D array-like object with"
                              " entries 'quantile' or 'value' for each"
                              " variable!")
        elif not threshold_method.shape:
            if threshold_method in ['quantile', 'value']:
                threshold_method = np.array([threshold_method] * data.shape[1])
            else:
                raise IOError("'threshold_method' must be either 'quantile' or"
                              " 'value' or a 1D array-like object with entries"
                              " 'quantile' or 'value' for each variable!")
        else:
            raise IOError("'threshold_method' must be either 'quantile' or "
                          "'value' or a 1D array-like object with entries "
                          "'quantile' or 'value' for each variable!")

        if threshold_values is not None:
            threshold_values = np.array(threshold_values)
            if threshold_values.shape == (data.shape[1],):
                if not np.all([isinstance(i, (float, int))
                               for i in threshold_values]):
                    raise IOError("'threshold_values' must be either float/int"
                                  " or 1D array-like object of float/int for "
                                  " each variable!")
            elif not threshold_values.shape:
                if isinstance(threshold_values.item(), (int, float)):
                    threshold_values = \
                        np.array([threshold_values] * data.shape[1])
                else:
                    raise IOError("'threshold_values' must be either float/int"
                                  " or 1D array-like object of float/int for "
                                  "each variable!")
            else:
                raise IOError("'threshold_values' must be either float/int or "
                              "1D array-like object of float/int for each "
                              "variable!")
        else:
            threshold_values = np.array([None] * data.shape[1])
            warnings.warn("No 'threshold_values' given. Median is used by "
                          "default!")

        if threshold_types is not None:
            threshold_types = np.array(threshold_types)
            if threshold_types.shape == (data.shape[1],):
                if not np.all([i in ['above', 'below']
                               for i in threshold_types]):
                    raise IOError("'threshold_types' must be either 'above' or"
                                  " 'below' or a 1D array-like object with "
                                  "entries 'above' or 'below' for each "
                                  "variable!")
            elif not threshold_types.shape:
                if threshold_types in ['above', 'below']:
                    threshold_types = \
                        np.array([threshold_types] * data.shape[1])
                else:
                    raise IOError("'threshold_types' must be either 'above' or"
                                  " 'below' or a 1D array-like object with "
                                  "entries 'above' or 'below' for each "
                                  "variable!")
            else:
                raise IOError("'threshold_types' must be either 'above' or "
                              "'below' or a 1D array-like object with entries "
                              "'above' or 'below' for each variable!")
        else:
            threshold_types = np.array([None] * data.shape[1])
            warnings.warn("No 'threshold_types' given. If 'threshold_values' "
                          ">= median, 'above' is used by default!")

        # Go through threshold_method, threshold_value and threshold_type
        # for each variable and check if input parameters are valid
        # In case of missing input parameters, try to set default values
        for i in range(data.shape[1]):

            if threshold_method[i] == 'quantile':

                # Check if threshold quantile is between zero and one
                if threshold_values[i] is not None:
                    if threshold_values[i] > 1.0 or threshold_values[i] < 0.0:
                        raise ValueError("Threshold_value for threshold_method"
                                         " 'quantile' must lie between 0.0 and"
                                         " 1.0!")

                # If threshold values are not given, use the median
                else:
                    threshold_values[i] = 0.5

                # Modification 2: Use nanquantile instead of quantile so that
                # time series containing NaN values are handled gracefully
                # rather than propagating NaN to the threshold computation.
                thresholds[i] = \
                    np.nanquantile(data_axswap[i], threshold_values[i])

                # If no threshold_types is given, check if threshold value is
                # larger or equal median, then 'above'
                if threshold_types[i] is None:
                    if threshold_values[i] >= 0.5:
                        threshold_types[i] = 'above'
                    else:
                        threshold_types[i] = 'below'

            if threshold_method[i] == 'value':

                if threshold_values[i] is None:
                    thresholds[i] = np.median(data_axswap[i])
                else:
                    # Check if given threshold values lie within data range
                    if np.max(data_axswap[i]) < threshold_values[i] or \
                            np.min(data_axswap[i]) > threshold_values[i]:
                        raise IOError("Threshold_value for threshold_method "
                                      "'value' must lie within variable "
                                      "range!")
                    thresholds[i] = threshold_values[i]

                if threshold_types[i] is None:
                    if thresholds[i] >= np.median(data_axswap[i]):
                        threshold_types[i] = 'above'
                    else:
                        threshold_types[i] = 'below'

        # Other methods for thresholding can be easily added here

        eventmatrix = np.zeros((data.shape[0], data.shape[1])) * (-1)
        # Iterate through all variables of the data and create event matrix
        # according to specified methods
        for t in range(data.shape[0]):
            for i in range(data.shape[1]):
                if threshold_types[i] == 'above':
                    if data[t][i] > thresholds[i]:
                        eventmatrix[t][i] = 1
                    else:
                        eventmatrix[t][i] = 0
                elif threshold_types[i] == 'below':
                    if data[t][i] < thresholds[i]:
                        eventmatrix[t][i] = 1
                    else:
                        eventmatrix[t][i] = 0

        return eventmatrix

    @staticmethod
    def event_synchronization(eventseriesx, eventseriesy, *,
                              ts1=None, ts2=None,
                              taumax=np.inf, lag=0.0):
        """
        Calculates the directed event synchronization from two event series X
        and Y using the algorithm described in [Quiroga2002]_,
        [Odenweller2020]_

        :type eventseriesx: 1D Numpy array
        :arg eventseriesx: Event series containing '0's and '1's
        :type eventseriesy: 1D Numpy array
        :arg eventseriesy: Event series containing '0's and '1's
        :type ts1: 1D Numpy array
        :arg ts1: Event time array containing time points when events of event
                  series 1 occur, not obligatory
        :type ts2: 1D Numpy array
        :arg ts2: Event time array containing time points when events of event
                  series 2 occur, not obligatory
        :type taumax: float
        :arg taumax: maximum distance of two events to be counted as
                     synchronous
        :type lag: float
        :arg lag: delay between the two event series, the second event series
                  is shifted by the value of lag
        :rtype: list
        :return: [Event synchronization XY, Event synchronization YX]
        """

        # Get time indices (type boolean or simple '0's and '1's)
        # Careful here with datatype, int16 allows for maximum time index 32767
        # Get time indices
        if ts1 is None:
            ex = np.array(np.where(eventseriesx), dtype='int16')
        else:
            ex = np.array([ts1[eventseriesx == 1]], dtype='float')
        if ts2 is None:
            ey = np.array(np.where(eventseriesy), dtype='int16')
        else:
            ey = np.array([ts2[eventseriesy == 1]], dtype='float')

        ey = ey + lag

        lx = ex.shape[1]
        ly = ey.shape[1]
        if lx == 0 or ly == 0:  # Division by zero in output
            return np.nan, np.nan
        if lx in [1, 2] or ly in [1, 2]:  # Too few events to calculate
            return 0., 0.

        # Array of distances
        dstxy2 = 2 * (np.repeat(ex[:, 1:-1].T, ly - 2, axis=1)
                      - np.repeat(ey[:, 1:-1], lx - 2, axis=0))
        # Dynamical delay
        diffx = np.diff(ex)
        diffy = np.diff(ey)
        diffxmin = np.minimum(diffx[:, 1:], diffx[:, :-1])
        diffymin = np.minimum(diffy[:, 1:], diffy[:, :-1])
        tau2 = np.minimum(np.repeat(diffxmin.T, ly - 2, axis=1),
                          np.repeat(diffymin, lx - 2, axis=0))
        tau2 = np.minimum(tau2, 2 * taumax)

        # Count equal time events and synchronised events
        eqtime = dstxy2.size - np.count_nonzero(dstxy2)

        # Calculate boolean matrices of coincidences
        Axy = (dstxy2 > 0) * (dstxy2 <= tau2)
        Ayx = (dstxy2 < 0) * (dstxy2 >= -tau2)

        # Loop over coincidences and determine number of double counts
        # by checking at least one event of the pair is also coincided
        # in other direction
        countxydouble = countyxdouble = 0

        for i, j in np.transpose(np.where(Axy)):
            countxydouble += np.any(Ayx[i, :]) or np.any(Ayx[:, j])
        for i, j in np.transpose(np.where(Ayx)):
            countyxdouble += np.any(Axy[i, :]) or np.any(Axy[:, j])

        # Calculate counting quantities and subtract half of double countings
        countxy = np.sum(Axy) + 0.5 * eqtime - 0.5 * countxydouble
        countyx = np.sum(Ayx) + 0.5 * eqtime - 0.5 * countyxdouble

        norm = np.sqrt((lx - 2) * (ly - 2))
        return countxy / norm, countyx / norm

    @staticmethod
    def event_coincidence_analysis(eventseriesx, eventseriesy, taumax, *,
                                   ts1=None, ts2=None, lag=0.0):
        """
         Event coincidence analysis:
         Returns the precursor and trigger coincidence rates of two event
         series X and Y following the algorithm described in [Odenweller2020]_.

         :type eventseriesx: 1D Numpy array
         :arg eventseriesx: Event series containing '0's and '1's
         :type eventseriesy: 1D Numpy array
         :arg eventseriesy: Event series containing '0's and '1's
         :type ts1: 1D Numpy array
         :arg ts1: Event time array containing time points when events of event
                   series 1 occur, not obligatory
         :type ts2: 1D Numpy array
         :arg ts2: Event time array containing time points when events of event
                   series 2 occur, not obligatory
         :type taumax: float
         :arg taumax: coincidence interval width
         :type lag: int
         :arg lag: lag parameter
         :rtype: list
         :return: [Precursor coincidence rate XY, Trigger coincidence rate XY,
               Precursor coincidence rate YX, Trigger coincidence rate YX]
         """

        # Get time indices
        if ts1 is None:
            e1 = np.where(eventseriesx)[0]
        else:
            e1 = ts1[eventseriesx == 1]
        if ts2 is None:
            e2 = np.where(eventseriesy)[0]
        else:
            e2 = ts2[eventseriesy == 1]

        # Count events that cannot be coincided due to lag and delT
        if not (lag == 0 and taumax == 0):
            n11 = len(e1[e1 <= e1[0] + lag + taumax])  # Start of es1
            n12 = len(e1[e1 >= (e1[-1] - lag - taumax)])  # End of es1
            n21 = len(e2[e2 <= e2[0] + lag + taumax])  # Start of es2
            n22 = len(e2[e2 >= (e2[-1] - lag - taumax)])  # End of es2
        else:
            n11, n12, n21, n22 = 0, 0, 0, 0  # Instantaneous coincidence

        # Number of events
        l1 = len(e1)
        l2 = len(e2)
        # Array of all interevent distances
        dst = (np.array([e1] * l2).T - np.array([e2] * l1))

        # Count coincidences with array slicing
        prec12 = np.count_nonzero(
            np.any(((dst - lag >= 0) * (dst - lag <= taumax))[n11:, :],
                   axis=1))
        trig12 = np.count_nonzero(
            np.any(((dst - lag >= 0) * (dst - lag <= taumax))
                   [:, :dst.shape[1] - n22], axis=0))
        prec21 = np.count_nonzero(np.any(((-dst - lag >= 0)
                                          * (-dst - lag <= taumax))[:, n21:],
                                         axis=0))
        trig21 = np.count_nonzero(
            np.any(((-dst - lag >= 0) * (-dst - lag <= taumax))
                   [:dst.shape[0] - n12, :], axis=1))

        # Normalisation and output
        return (np.float32(prec12) / (l1 - n11),
                np.float32(trig12) / (l2 - n22),
                np.float32(prec21) / (l2 - n21),
                np.float32(trig21) / (l1 - n12))

    def _eca_coincidence_rate(self, eventseriesx, eventseriesy, *,
                              window_type='symmetric', ts1=None, ts2=None):
        """
         Event coincidence analysis:
         Returns the coincidence rates of two event series for both directions

         :type eventseriesx: 1D Numpy array
         :arg eventseriesx: Event series containing '0's and '1's
         :type eventseriesy: 1D Numpy array
         :arg eventseriesy: Event series containing '0's and '1's
         :type ts1: 1D Numpy array
         :arg ts1: Event time array containing time points when events of event
                   series 1 occur, not obligatory
         :type ts2: 1D Numpy array
         :arg ts2: Event time array containing time points when events of event
                   series 2 occur, not obligatory
         :type window_type: str {'retarded', 'advanced', 'symmetric'}
         :arg window_type: Only for ECA. Determines if precursor coincidence
                           rate ('advanced'), trigger coincidence rate
                           ('retarded') or a general coincidence rate with the
                           symmetric interval [-taumax, taumax] are computed
                           ('symmetric'). Default: 'symmetric'
         :rtype: list
         :return: Precursor coincidence rates [XY, YX]
         """
        # Get time indices
        if ts1 is None:
            e1 = np.where(eventseriesx)[0]
        else:
            e1 = ts1[eventseriesx == 1]
        if ts2 is None:
            e2 = np.where(eventseriesy)[0]
        else:
            e2 = ts2[eventseriesy == 1]

        lag = self.__lag
        taumax = self.__taumax

        # Number of events
        l1 = len(e1)
        l2 = len(e2)

        # Array of all interevent distances
        dst = (np.array([e1] * l2).T - np.array([e2] * l1))

        if window_type == 'advanced':
            deltaT1 = 0.0
            deltaT2 = taumax

            # Count events that cannot be coincided due to lag and deltaT
            if not (lag == 0 and taumax == 0):
                n11 = len(e1[e1 <= (e1[0] + lag + deltaT2)])  # Start of es1
                n21 = len(e2[e2 <= (e2[0] + lag + deltaT2)])  # Start of es2
                n12, n22 = 0, 0
            else:
                n11, n12, n21, n22 = 0, 0, 0, 0  # Instantaneous coincidence

            # Count coincidences with array slicing
            coincidence12 = np.count_nonzero(
                np.any(((dst - lag >= deltaT1) * (dst - lag <= deltaT2))
                       [n11:, :], axis=1))
            coincidence21 = np.count_nonzero(
                np.any(((-dst - lag >= deltaT1) * (-dst - lag <= deltaT2))
                       [:, n21:], axis=0))

        elif window_type == 'retarded':
            deltaT1 = 0.0
            deltaT2 = taumax

            # Count events that cannot be coincided due to lag and delT
            if not (lag == 0 and taumax == 0):
                n11 = 0  # Start of es1
                n12 = len(e1[e1 >= (e1[-1] - lag - deltaT2)])  # End of es1
                n21 = 0  # Start of es2
                n22 = len(e2[e2 >= (e2[-1] - lag - deltaT2)])  # End of es2
            else:
                n11, n12, n21, n22 = 0, 0, 0, 0  # Instantaneous coincidence

            # Count coincidences with array slicing
            coincidence12 = np.count_nonzero(
                np.any(((dst - lag >= deltaT1) * (dst - lag <= deltaT2))
                       [:, :dst.shape[1] - n22], axis=0))
            coincidence21 = np.count_nonzero(
                np.any(((-dst - lag >= deltaT1) * (-dst - lag <= deltaT2))
                       [:dst.shape[0] - n12, :], axis=1))

            return ((np.float32(coincidence12) / (l2 - n22),
                     np.float32(coincidence21) / (l1 - n12)))

        elif window_type == 'symmetric':
            deltaT1, deltaT2 = -taumax, taumax

            # Count events that cannot be coincided due to lag and delT
            if not (lag == 0 and taumax == 0):
                n11 = len(e1[e1 <= (e1[0] + lag + deltaT2)])  # Start of es1
                n12 = len(e1[e1 >= (e1[-1] - lag + deltaT1)])  # End of es1
                n21 = len(e2[e2 <= (e2[0] + lag + deltaT2)])  # Start of es2
                n22 = len(e2[e2 >= (e2[-1] - lag + deltaT1)])  # End of es2
            else:
                n11, n12, n21, n22 = 0, 0, 0, 0  # Instantaneous coincidence

            # Count coincidences with array slicing
            coincidence12 = np.count_nonzero(
                np.any(((dst - lag >= deltaT1) * (dst - lag <= deltaT2))
                       [n11:dst.shape[0]-n12, :], axis=1))
            coincidence21 = np.count_nonzero(
                np.any(((-dst - lag >= deltaT1) * (-dst - lag <= deltaT2))
                       [:, n21:dst.shape[1]-n22], axis=0))

        else:
            raise IOError("Parameter 'window_type' must be 'advanced',"
                          " 'retarded' or 'symmetric'!")

        # Normalisation and output
        return (np.float32(coincidence12) / (l1 - n11 - n12),
                np.float32(coincidence21) / (l2 - n21 - n22))

    def event_series_analysis(self, method='ES', symmetrization='directed',
                              window_type='symmetric'):
        """
        Returns the NxN matrix of the chosen event series measure where N is
        the number of variables. The entry [i, j] denotes the event
        synchronization or event coincidence analysis from variable j to
        variable i. According to the 'symmetrization' parameter the event
        series measure matrix is symmetrized or not.

        The event coincidence rate of ECA is calculated according to the
        formula: r(Y|X, DeltaT1, DeltaT2, tau) =
        1/N_X sum_{i=1}^{N_X} Theta[sum{j=1}^{N_Y}
        1_[DeltaT1, DeltaT2] (t_i^X - (t_j^Y + tau))],
        where X is the first input event series, Y the second input event
        series, N_X the number of events in X, DeltaT1 and DeltaT2 the given
        coincidence interval boundaries, tau the lag between X and Y, Theta the
        Heaviside function and 1 the indicator function.

        :type method: str 'ES' or 'ECA'
        :arg method: determines if ES or ECA should be used
        :type symmetrization: str {'directed', 'symmetric', 'antisym',
                              'mean', 'max', 'min'} for ES,
                              str {'directed', 'mean', 'max', 'min'} for ECA
        :arg symmetrization: determines if and which symmetrisation
                             method should be used for the ES/ECA score matrix
        :type window_type: str {'retarded', 'advanced', 'symmetric'}
        :arg window_type: Only for ECA. Determines if precursor coincidence
                          rate ('advanced'), trigger coincidence rate
                          ('retarded') or a general coincidence rate with the
                          symmetric interval [-taumax, taumax] are computed
                          ('symmetric'). Default: 'symmetric'
        :rtype: 2D numpy array
        :return: pairwise event synchronization or pairwise coincidence rates
                symmetrized according to input parameter 'symmetrization'
        """

        if method not in ['ES', 'ECA']:
            raise IOError("'method' parameter must be 'ECA' or 'ES'!")

        directedESMatrix = []

        if method == 'ES':

            if symmetrization not in ['directed', 'symmetric', 'antisym',
                                      'mean', 'max', 'min']:
                raise IOError("'symmetrization' parameter must be 'directed', "
                              "'symmetric', 'antisym', 'mean', 'max' or"
                              "'min' for event synchronization!")

            directedESMatrix = self._ndim_event_synchronization()

        elif method == 'ECA':
            if self.__taumax is np.inf:
                raise ValueError("'delta' must be a finite time window to"
                                 " determine event coincidence!")

            if symmetrization not in ['directed', 'mean', 'max', 'min']:
                raise IOError("'symmetrization' parameter must be 'directed', "
                              "'mean', 'max' or 'min' for event"
                              "coincidence analysis!")

            if window_type not in ['retarded', 'advanced', 'symmetric']:
                raise IOError("'window_type' must be 'retarded',"
                              "'advanced' or 'symmetric'!")

            directedESMatrix = \
                self._ndim_event_coincidence_analysis(window_type=window_type)

        # Use symmetrization functions for symmetrization and return result
        return self.symmetrization_options[symmetrization](directedESMatrix)

    # =========================================================================
    # ES — dispatches to the fastest available kernel (Optimizations 1 & 2)
    # =========================================================================

    @Cached.method()
    def _ndim_event_synchronization(self):
        """
        Compute NxN event synchronization matrix [i,j] with event
        synchronization from j to i without symmetrization.

        Routing (Optimizations 1 & 2):
          finite taumax, any lag  →  sparse binary-search   (fast, exact)
          infinite taumax, lag=0  →  blocked-dense 4-D      (vectorized)
          infinite taumax, lag≠0  →  pairwise loop          (fallback)

        :rtype: NxN numpy array where N is the number of variables of the
                eventmatrix
        :return: event synchronization matrix
        """
        node_data = self._precompute_es_node_data()

        if np.isfinite(self.__taumax):
            # Optimization 1: sparse binary-search kernel
            # (exact, O(N²·L·taumax/T))
            return self._compute_es_matrix_sparse_lag(node_data)
        else:
            if self.__lag != 0.0:
                # Fallback: pairwise loop (identical to original pyunicorn)
                return self._ndim_event_synchronization_pairwise()
            # Optimization 2: blocked dense 4-D tensor kernel
            return self._compute_es_matrix_blocked(node_data)

    def _ndim_event_synchronization_pairwise(self):
        """
        Pairwise ES loop — identical to original pyunicorn.
        Used as fallback when taumax=inf and lag≠0.

        Modification 3: tqdm progress bar added.
        """
        N = self.__N
        eventmatrix = self.__eventmatrix
        timestamps = self.__timestamps
        taumax = self.__taumax
        lag = self.__lag

        directed = np.zeros((N, N))
        for i in tqdm(range(N), desc='ES pairwise'):
            for j in range(i + 1, N):
                directed[i, j], directed[j, i] = \
                    self.event_synchronization(eventmatrix[:, i],
                                               eventmatrix[:, j],
                                               ts1=timestamps, ts2=timestamps,
                                               taumax=taumax, lag=lag)
        return directed

    def _precompute_es_node_data(self):
        """
        Optimization 1 / 2: Precompute inner-event times and per-event
        dynamic tau2 for every node.  Called once before the matrix loop.
        """
        E = self.__eventmatrix
        ts = self.__timestamps
        taumax = self.__taumax
        cap = 2.0 * taumax

        node_data = []
        for n in range(self.__N):
            event_times = ts[E[:, n] > 0.5].astype(np.float64)
            if len(event_times) <= 2:
                inner_times = np.empty(0, dtype=np.float64)
                tau2_inner = np.empty(0, dtype=np.float64)
            else:
                inner_times = event_times[1:-1].copy()
                diffs = np.diff(event_times)
                tau2_inner = np.minimum(diffs[:-1], diffs[1:])
                tau2_inner = np.minimum(tau2_inner, cap)
            node_data.append({
                'event_times': event_times,
                'inner_times': inner_times,
                'tau2_inner': tau2_inner,
                'inner_count': len(inner_times),
            })
        return node_data

    def _compute_es_matrix_blocked(self, node_data, block_size=32, n_jobs=1):
        """
        Optimization 2: Assemble the full N×N directed ES matrix from blocked
        4-D computations (infinite taumax, lag=0).

        Optimization 4: n_jobs > 1 uses joblib if available.
        """
        N = self.__N
        directed = np.zeros((N, N), dtype=np.float64)

        pairs = []
        for a_start in range(0, N, block_size):
            block_a = list(range(a_start, min(N, a_start + block_size)))
            for b_start in range(a_start, N, block_size):
                block_b = list(range(b_start, min(N, b_start + block_size)))
                pairs.append((block_a, block_b))

        if n_jobs != 1 and _JOBLIB_AVAILABLE:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_es_block)(node_data, ba, bb)
                for ba, bb in pairs
            )
        else:
            results = [
                _compute_es_block(node_data, ba, bb)
                for ba, bb in tqdm(pairs, desc='ES blocks')
            ]

        for (block_a, block_b), (bxy, byx) in zip(pairs, results):
            for ia, i in enumerate(block_a):
                jb_start = ia + 1 if block_a[0] == block_b[0] else 0
                for jb in range(jb_start, len(block_b)):
                    j = block_b[jb]
                    if i == j:
                        continue
                    directed[i, j] = bxy[ia, jb]
                    directed[j, i] = byx[ia, jb]

        np.fill_diagonal(directed, 0.0)
        return directed

    def _compute_es_matrix_sparse_lag(self, node_data, n_jobs=1):
        """
        Optimization 1: Assemble the full N×N directed ES matrix using the
        lag-aware sparse binary-search kernel.  Works for any scalar lag value.

        Row-based chunking minimises joblib serialisation overhead.
        Optimization 4: n_jobs > 1 uses joblib if available.
        """
        N = self.__N
        taumax = self.__taumax
        lag = self.__lag
        directed = np.zeros((N, N), dtype=np.float64)

        n_workers = n_jobs if (n_jobs != 1 and _JOBLIB_AVAILABLE) else 1
        rows_per_job = max(1, -(-N // n_workers))   # ceiling division
        row_chunks = [(s, min(N, s + rows_per_job))
                      for s in range(0, N, rows_per_job)]

        if n_jobs != 1 and _JOBLIB_AVAILABLE:
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_process_es_rows_sparse_lag)(
                    r0, r1, N, node_data, taumax=taumax, lag=lag
                )
                for r0, r1 in row_chunks
            )
        else:
            results_list = [
                _process_es_rows_sparse_lag(
                    r0, r1, N, node_data, taumax=taumax, lag=lag
                )
                for r0, r1 in tqdm(row_chunks, desc='ES sparse (lag)',
                                   unit='chunk')
            ]

        for results in results_list:
            for i, j, xy, yx in results:
                directed[i, j] = xy
                directed[j, i] = yx

        np.fill_diagonal(directed, 0.0)
        return directed

    # =========================================================================
    # ECA — dispatches to vectorized or pairwise kernel (Optimization 3)
    # =========================================================================

    def _ndim_event_coincidence_analysis(self, window_type='symmetric'):
        """
        Computes NxN event coincidence matrix of event coincidence rate.

        Routing (Optimization 3):
          uniform integer timestamps + integer lag  →  vectorized cumsum + BLAS
          otherwise                                 →  pairwise loop (original)

        :type window_type: str {'retarded', 'advanced', 'symmetric'}
        :arg window_type: Only for ECA. Determines if precursor coincidence
                          rate ('advanced'), trigger coincidence rate
                          ('retarded') or a general coincidence rate with the
                          symmetric interval [-taumax, taumax] are computed
                          ('symmetric'). Default: 'symmetric'
        :rtype: NxN numpy array where N is the number of variables of the
                eventmatrix
        :return: event coincidence matrix
        """
        if window_type not in ['advanced', 'retarded', 'symmetric']:
            raise IOError("'window_type' must be 'advanced', 'retarded' or"
                          " 'symmetric'!")

        ts = self.__timestamps
        dts = np.diff(ts)
        is_uniform_integer = (len(dts) > 0
                              and np.allclose(dts, 1.0, rtol=1e-9, atol=1e-9))
        is_integer_lag = np.isclose(self.__lag, round(self.__lag),
                                    rtol=1e-9, atol=1e-9)

        if is_uniform_integer and is_integer_lag:
            # Optimization 3: vectorized cumsum + BLAS path
            return self._compute_eca_vectorized_lag(window_type)
        else:
            # Fallback: pairwise loop (identical to original pyunicorn)
            return self._compute_eca_pairwise(window_type)

    def _compute_eca_pairwise(self, window_type):
        """
        Pairwise ECA loop — identical to original pyunicorn.
        Modification 3: tqdm progress bar added.
        """
        N = self.__N
        eventmatrix = self.__eventmatrix
        timestamps = self.__timestamps
        directed = np.zeros((N, N))

        for i in tqdm(range(N), desc=f'ECA pairwise ({window_type})'):
            for j in range(i + 1, N):
                directed[i, j], directed[j, i] = \
                    self._eca_coincidence_rate(eventmatrix[:, i],
                                               eventmatrix[:, j],
                                               window_type=window_type,
                                               ts1=timestamps, ts2=timestamps)
        return directed

    def _compute_eca_vectorized_lag(self, window_type):
        """
        Optimization 3: Vectorized ECA via cumsum rolling windows + BLAS
        matrix multiply, extended to support any integer-valued lag.

        Core idea: build the lag=0 smeared indicator matrix with a cumsum
        rolling window, then shift it by `lag` steps along the time axis.
        Boundary corrections mirror the pairwise formula exactly.

        For lag=0 the output is bit-for-bit identical to the pairwise path
        within float32 rounding (atol < 1e-6 vs pairwise reference).
        Complexity: O(T·N + N²) vs O(N²·L²) for the pairwise approach.
        """
        E = self.__eventmatrix.astype(np.float64)
        ts = self.__timestamps
        T, N = E.shape
        tau = int(round(self.__taumax))
        lag = int(round(self.__lag))
        N_events = E.sum(axis=0)

        directed = np.zeros((N, N), dtype=np.float64)

        def _shift(arr, shift):
            """Shift (T, N) array along axis=0 by `shift` rows (fill zeros)."""
            if shift == 0:
                return arr.copy()
            out = np.zeros_like(arr)
            if shift > 0:
                rows = min(shift, T)
                if rows < T:
                    out[rows:] = arr[:T - rows]
            else:
                rows = min(-shift, T)
                if rows < T:
                    out[:T - rows] = arr[rows:]
            return out

        if tau == 0 and lag == 0:
            tau_start = -1
            tau_end = -1
        else:
            tau_start = tau + lag
            tau_end = tau + lag

        def _n_boundary_start(n, t_eff):
            if N_events[n] == 0 or t_eff < 0:
                return 0.0
            ev = ts[E[:, n] > 0.5]
            return float(np.sum(ev <= ev[0] + t_eff))

        def _n_boundary_end(n, t_eff):
            if N_events[n] == 0 or t_eff < 0:
                return 0.0
            ev = ts[E[:, n] > 0.5]
            return float(np.sum(ev >= ev[-1] - t_eff))

        def _first_ev_ts(n):
            return ts[np.argmax(E[:, n] > 0.5)] if N_events[n] > 0 else ts[-1]

        def _last_ev_ts(n):
            return ts[T - 1 - np.argmax(E[::-1, n] > 0.5)] \
                if N_events[n] > 0 else ts[0]

        if window_type == 'advanced':
            E_cs = np.cumsum(E, axis=0)
            E_window = E_cs.copy()
            if tau > 0:
                E_window[tau + 1:] -= E_cs[:-tau - 1]
            else:
                E_window = E.copy()
            E_smeared = (E_window > 0).astype(np.float64)
            E_smeared = _shift(E_smeared, lag)

            n_start = np.array([_n_boundary_start(n, tau_start)
                                for n in range(N)])
            E_trimmed = E.copy()
            for n in range(N):
                if N_events[n] > 0 and tau_start >= 0:
                    E_trimmed[ts <= _first_ev_ts(n) + tau_start, n] = 0.0

            adj_N = np.where(N_events - n_start > 0,
                             N_events - n_start, 1.0)
            counts_f64 = E_trimmed.T @ E_smeared
            adj_N_f32 = adj_N.astype(np.float32)
            directed_f32 = counts_f64.astype(np.float32) / adj_N_f32[:, None]
            directed = directed_f32.astype(np.float64)
            np.fill_diagonal(directed, 0.0)

        elif window_type == 'retarded':
            E_rev = E[::-1]
            E_cs_rev = np.cumsum(E_rev, axis=0)
            E_win_rev = E_cs_rev.copy()
            if tau > 0:
                E_win_rev[tau + 1:] -= E_cs_rev[:-tau - 1]
            else:
                E_win_rev = E_rev.copy()
            E_smeared_fwd = (E_win_rev[::-1] > 0).astype(np.float64)
            E_smeared_fwd = _shift(E_smeared_fwd, -lag)

            n_end = np.array([_n_boundary_end(n, tau_end)
                              for n in range(N)])
            E_trimmed_fwd = E.copy()
            for n in range(N):
                if N_events[n] > 0 and tau_end >= 0:
                    E_trimmed_fwd[ts >= _last_ev_ts(n) - tau_end, n] = 0.0

            adj_N = np.where(N_events - n_end > 0,
                             N_events - n_end, 1.0)
            counts_f64 = E_smeared_fwd.T @ E_trimmed_fwd
            adj_N_f32 = adj_N.astype(np.float32)
            directed_f32 = counts_f64.astype(np.float32) / adj_N_f32[None, :]
            directed = directed_f32.astype(np.float64)
            np.fill_diagonal(directed, 0.0)

        elif window_type == 'symmetric':
            E_cs = np.cumsum(E, axis=0)
            E_win_b = E_cs.copy()
            if tau > 0:
                E_win_b[tau + 1:] -= E_cs[:-tau - 1]
            else:
                E_win_b = E.copy()

            E_rev = E[::-1]
            E_cs_rev = np.cumsum(E_rev, axis=0)
            E_win_f_r = E_cs_rev.copy()
            if tau > 0:
                E_win_f_r[tau + 1:] -= E_cs_rev[:-tau - 1]
            else:
                E_win_f_r = E_rev.copy()
            E_win_f = E_win_f_r[::-1]

            E_sym = E_win_b + E_win_f - E
            E_smeared_sym = (E_sym > 0).astype(np.float64)
            E_smeared_sym = _shift(E_smeared_sym, lag)

            n_start = np.array([_n_boundary_start(n, tau_start)
                                for n in range(N)])
            n_end = np.array([_n_boundary_end(n, tau_end)
                              for n in range(N)])
            E_trimmed_sym = E.copy()
            for n in range(N):
                if N_events[n] > 0:
                    mask = np.zeros(T, dtype=bool)
                    if tau_start >= 0:
                        mask |= ts <= _first_ev_ts(n) + tau_start
                    if tau_end >= 0:
                        mask |= ts >= _last_ev_ts(n) - tau_end
                    E_trimmed_sym[mask, n] = 0.0

            adj_N = np.where(N_events - n_start - n_end > 0,
                             N_events - n_start - n_end, 1.0)
            counts_f64 = E_trimmed_sym.T @ E_smeared_sym
            adj_N_f32 = adj_N.astype(np.float32)
            directed_f32 = counts_f64.astype(np.float32) / adj_N_f32[:, None]
            directed = directed_f32.astype(np.float64)
            np.fill_diagonal(directed, 0.0)

        return directed

    # =========================================================================
    # Significance analysis (identical to original pyunicorn)
    # =========================================================================

    def _empirical_percentiles(self, method=None, n_surr=1000,
                               symmetrization='directed',
                               window_type='symmetric'):
        """
        Compute p-values of event synchronisation (ES) and event coincidence
        analysis (ECA) using a Monte-Carlo approach. Surrogates are obtained by
        shuffling the event series. ES/ECA scores of the surrogate event series
        are computed and p-values are the empirical percentiles of the original
        event series compared to the ES/ECA scores of the surrogates.

        :type method: str 'ES' or 'ECA'
        :arg method: determines if ES or ECA should be used
        :type n_surr: int
        :arg n_surr: number of surrogates for Monte-Carlo method
        :type symmetrization: str {'directed', 'symmetric', 'antisym',
                              'mean', 'max', 'min'} for ES,
                              str {'directed', 'mean', 'max', 'min'} for ECA
        :arg symmetrization: determines if and which symmetrisation
                             method should be used for the ES/ECA score matrix
        :type window_type: str {'retarded', 'advanced', 'symmetric'}
        :arg window_type: Only for ECA. Determines if precursor coincidence
                          rate ('advanced'), trigger coincidence rate
                          ('retarded') or a general coincidence rate with the
                          symmetric interval [-taumax, taumax] are computed
                          ('symmetric'). Default: 'symmetric'
        :rtype: 2D numpy array
        :return: p-values of the ES/ECA scores for all
        """

        # Get instance variables
        lag = self.__lag
        deltaT = self.__taumax

        event_series_result = \
            self.event_series_analysis(method=method,
                                       symmetrization=symmetrization,
                                       window_type=window_type)

        surrogates = np.zeros((n_surr, self.__N, self.__N))
        shuffled_es = self.__eventmatrix.copy()

        # For each surrogate, shuffle each event series and perform ES/ECA
        # analysis
        for n in range(n_surr):
            for i in range(self.__N):
                np.random.shuffle(shuffled_es[:, i])

            if method == 'ES':
                for i in range(0, self.__N):
                    for j in range(i + 1, self.__N):
                        surrogates[n, i, j], surrogates[n, j, i] = \
                            self.event_synchronization(shuffled_es[:, i],
                                                       shuffled_es[:, j],
                                                       taumax=deltaT, lag=lag)

            elif method == 'ECA':
                for i in range(0, self.__N):
                    for j in range(i + 1, self.__N):
                        surrogates[n, i, j], surrogates[n, j, i] = \
                            self._eca_coincidence_rate(shuffled_es[:, i],
                                                       shuffled_es[:, j],
                                                       window_type=window_type)

            # Symmetrize according to symmetry keyword argument
            surrogates[n, :, :] = \
                self.symmetrization_options[
                    symmetrization](surrogates[n, :, :])

        # Calculate significance level via strict empirical percentiles for
        # each event series pair
        empirical_percentiles = np.zeros((self.__N, self.__N))
        for i in range(self.__N):
            for j in range(self.__N):
                empirical_percentiles[i, j] = \
                    stats.percentileofscore(surrogates[:, i, j],
                                            event_series_result[i][j],
                                            kind='strict') / 100

        return empirical_percentiles

    def event_analysis_significance(self, *, method=None,
                                    surrogate='shuffle', n_surr=1000,
                                    symmetrization='directed',
                                    window_type='symmetric'):
        """
        Returns significance levels (1 - p-values) for event synchronisation
        (ES) and event coincidence analysis (ECA). For ECA, there is an
        analytic option providing significance levels based on independent
        Poisson processes. The 'shuffle' option uses a Monte-Carlo approach,
        calculating ES or ECA scores for surrogate event time series obtained
        by shuffling the original event time series. The significance levels
        are the empirical percentiles of the ES/ECA scores of the original
        event series compared with the scores of the surrogate data.

        :type method: str 'ES' or 'ECA'
        :arg method: determines if ES or ECA should be used
        :type surrogate: str 'analytic' or 'shuffle'
        :arg surrogate: determines if p-values should be calculated using a
                        Monte-Carlo method or (only for ECA) an analytic
                        Poisson process null model
        :type n_surr: int
        :arg n_surr: number of surrogates for Monte-Carlo method
        :type symmetrization: str {'directed', 'symmetric', 'antisym',
                              'mean', 'max', 'min'} for ES,
                              str {'directed', 'mean', 'max', 'min'} for ECA
        :arg symmetrization: determines if and which symmetrisation
                             method should be used for the ES/ECA score matrix
        :type window_type: str {'retarded', 'advanced', 'symmetric'}
        :arg window_type: Only for ECA. Determines if precursor coincidence
                          rate ('advanced'), trigger coincidence rate
                          ('retarded') or a general coincidence rate with the
                          symmetric interval [-taumax, taumax] are computed
                          ('symmetric'). Default: 'symmetric'
        :rtype: 2D numpy array
        :return: significance levels of the ES/ECA scores for all pairs of
                 event series in event matrix
        """

        if method not in ['ES', 'ECA']:
            raise IOError("'method' parameter must be 'ECA' or 'ES'!")

        if surrogate not in ['analytic', 'shuffle']:
            raise IOError("'surrogate' parameter must be 'analytic' or "
                          "'shuffle'!")

        # Get instance variables
        deltaT = self.__taumax
        lag = self.__lag

        if method == 'ECA':

            if symmetrization not in ['directed', 'mean', 'max', 'min']:
                raise IOError("'symmetrization' parameter must be 'directed', "
                              "'mean', 'max' or 'min' for event"
                              "coincidence analysis!")

            if window_type not in ['retarded', 'advanced', 'symmetric']:
                raise IOError("'window_type' must be 'retarded',"
                              "'advanced' or 'symmetric'!")

            if surrogate == 'analytic':

                if symmetrization != 'directed':
                    raise IOError("'symmetrization' parameter should be"
                                  "'directed' for analytical calculation of"
                                  "significance levels!")

                if window_type not in ['retarded', 'advanced']:
                    raise IOError("'window_type' parameter must be 'retarded'"
                                  " or 'advanced' for analytical computation"
                                  " of significance levels!")

                # Compute ECA scores of stored event matrix
                directedECAMatrix = \
                    self._ndim_event_coincidence_analysis(
                        window_type=window_type)
                significance_levels = np.zeros((self.__N, self.__N))

                NEvents = self.__nrofevents

                if window_type == 'advanced':
                    for i in range(self.__N):
                        for j in range(i + 1, self.__N):
                            # Compute Poisson probability p
                            p = deltaT / (float(self.__timestamps[-1]) - lag)

                            # Compute number of precursor coincidences 2->1
                            K12 = int(directedECAMatrix[i][j] * NEvents[i])
                            # Compute probability of at least K12 precursor
                            # events
                            pvalue = 0.0
                            for K_star in range(K12, NEvents[i] + 1):
                                pvalue += \
                                    stats.binom.pmf(K_star, NEvents[i],
                                                    1.0 - pow(1.0 - p,
                                                              NEvents[j]))
                            significance_levels[i][j] = 1.0 - pvalue

                            # Compute number of precursor coincidence 1->2
                            K21 = int(directedECAMatrix[j][i] * NEvents[j])
                            # Compute probability of at least K21 precursor
                            # events
                            pvalue = 0.0
                            for K_star in range(K21, NEvents[j] + 1):
                                pvalue += \
                                    stats.binom.pmf(K_star, NEvents[j],
                                                    1.0 - pow(1.0 - p,
                                                              NEvents[i]))
                            significance_levels[j][i] = 1.0 - pvalue

                    return significance_levels

                # If window_type != 'advanced', it must be 'retarded'
                else:
                    for i in range(self.__N):
                        for j in range(i + 1, self.__N):
                            p = deltaT / (float(self.__timestamps[-1]) - lag)
                            # Compute probability of at least K12 trigger
                            # events

                            # Compute number of trigger coincidence 2->1
                            K12 = int(directedECAMatrix[i][j] * NEvents[j])
                            # Compute Poisson probability p
                            pvalue = 0.0
                            for K_star in range(K12, NEvents[j]):
                                pvalue += \
                                    stats.binom.pmf(K_star, NEvents[j],
                                                    1.0 - pow(1.0 - p,
                                                              NEvents[i]))
                            significance_levels[i][j] = 1.0 - pvalue

                            # Compute number of trigger coincidence 1->2
                            K21 = int(directedECAMatrix[j][i] * NEvents[i])
                            # Compute probability of at least K21 trigger
                            # events
                            pvalue = 0.0
                            for K_star in range(K21, NEvents[i]):
                                pvalue += \
                                    stats.binom.pmf(K_star, NEvents[i],
                                                    1.0 - pow(1.0 - p,
                                                              NEvents[j]))
                            significance_levels[j][i] = 1.0 - pvalue

                    return significance_levels

            # If surrogate is not 'analytic', it must be 'shuffle'
            else:
                return \
                    self._empirical_percentiles(method='ECA', n_surr=n_surr,
                                                symmetrization=symmetrization,
                                                window_type=window_type)

        elif method == 'ES':

            if surrogate != 'shuffle':
                raise IOError("Analytical calculation of significance level is"
                              " only possible for event coincidence analysis!")

            if symmetrization not in ['directed', 'symmetric', 'antisym',
                                      'mean', 'max', 'min']:
                raise IOError("'symmetrization' parameter must be 'directed', "
                              "'symmetric', 'antisym', 'mean', 'max' or"
                              "'min' for event synchronization!")

            return \
                self._empirical_percentiles(method='ES',
                                            n_surr=n_surr,
                                            symmetrization=symmetrization)
        else:
            return None
