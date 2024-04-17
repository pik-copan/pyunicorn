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
"""

from typing import Tuple
from collections.abc import Hashable

import warnings

import numpy as np
from scipy import stats

from ..core.cache import Cached


class EventSeries(Cached):

    def __init__(self, data, timestamps=None, taumax=np.inf, lag=0.0,
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
            # Allow for wrong axis, i.e. first axis variables and second axis
            # time if time series have the same length
            if isinstance(data, np.ndarray):
                if data.shape[1] > data.shape[0]:
                    data = np.swapaxes(data, 0, 1)

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

                # Compute threshold value according to quantile
                thresholds[i] = \
                    np.quantile(data_axswap[i], threshold_values[i])

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
    def event_synchronization(eventseriesx, eventseriesy, ts1=None, ts2=None,
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
    def event_coincidence_analysis(eventseriesx, eventseriesy, taumax,
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

    def _eca_coincidence_rate(self, eventseriesx, eventseriesy,
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

    @Cached.method()
    def _ndim_event_synchronization(self):
        """
        Compute NxN event synchronization matrix [i,j] with event
        synchronization from j to i without symmetrization.

        :rtype: NxN numpy array where N is the number of variables of the
                eventmatrix
        :return: event synchronization matrix
        """
        # Get instance variables
        eventmatrix = self.__eventmatrix
        timestamps = self.__timestamps
        lag = self.__lag
        taumax = self.__taumax

        directed = np.zeros((self.__N, self.__N))
        for i in range(0, self.__N):
            for j in range(i + 1, self.__N):
                directed[i, j], directed[j, i] = \
                    self.event_synchronization(eventmatrix[:, i],
                                               eventmatrix[:, j],
                                               ts1=timestamps, ts2=timestamps,
                                               taumax=taumax, lag=lag)
        return directed

    def _ndim_event_coincidence_analysis(self, window_type='symmetric'):
        """
        Computes NxN event coincidence matrix of event coincidence rate

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

        eventmatrix = self.__eventmatrix
        directed = np.zeros((self.__N, self.__N))
        timestamps = self.__timestamps

        for i in range(0, self.__N):
            for j in range(i + 1, self.__N):
                directed[i, j], directed[j, i] = \
                    self._eca_coincidence_rate(eventmatrix[:, i],
                                               eventmatrix[:, j],
                                               window_type=window_type,
                                               ts1=timestamps, ts2=timestamps)

        return directed

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

    def event_analysis_significance(self, method=None,
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
