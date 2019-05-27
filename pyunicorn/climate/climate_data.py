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
from numpy import random

from ..core import Data


#
#  Define class ClimateData
#
class ClimateData(Data):

    """
    Encapsulates spatio-temporal climate data.

    Provides methods to manipulate this data, i.e. calculate daily (monthly)
    mean values and anomaly values.

    @ivar data_source: (string) - The name of the data source
                                  (model, reanalysis, station)
    """

    #
    #  Defines internal methods
    #

    def __init__(self, observable, grid, time_cycle, anomalies=False,
                 observable_name="", observable_long_name=None, window=None,
                 silence_level=0):
        """
        Initialize an instance of ClimateData.

        The spatio-temporal window is described by the following
        dictionary::

            window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                      "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        :type observable: 2D array [time, index]
        :arg observable: The array of time series to be represented by the
            :class:`.Data` instance.
        :type grid: :class:`.Grid` instance
        :arg grid: The Grid representing the spatial coordinates associated to
            the time series and their temporal sampling.
        :arg int time_cycle: The annual cycle length of the data (units of
            samples).
        :arg bool anomalies: Indicates whether the data are climatological
            anomaly values.
        :arg str observable_name: A short name for the observable.
        :arg str observable_long_name: A long name for the observable.
        :arg dict window: Spatio-temporal window to select a view on the data.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        Data.__init__(self, observable=observable, grid=grid,
                      observable_name=observable_name,
                      observable_long_name=observable_long_name,
                      window=window, silence_level=silence_level)

        #  Set class variables
        self.time_cycle = time_cycle
        """(number (int)) - The annual cycle length of the data
                            (units of samples)."""

        #  Set flags
        self._flag_phase_mean = False
        self._phase_mean = None

        self.data_source = ""

        # If data are anomalies skip automatic calculation of anomalies
        if anomalies:
            self._flag_anomaly = True
            self._anomaly = observable
        else:
            self._flag_anomaly = False

    def __str__(self):
        """
        Returns a string representation.
        """
        return 'ClimateData:\n' + Data.__str__(self)

    def clear_cache(self):
        """
        Clean up cache.

        Is reversible, since all cached information can be recalculated from
        basic data.
        """
        Data.clear_cache(self)

        if self._flag_phase_mean:
            del self._phase_mean
            self._flag_phase_mean = False

        if self._flag_anomaly:
            del self._anomaly
            self._flag_anomaly = False

    #
    #  Define alternative constructors
    #

    @classmethod
    def Load(cls, file_name, observable_name, time_cycle,
             time_name="time", latitude_name="lat", longitude_name="lon",
             data_source=None,
             file_type="NetCDF", window=None,
             vertical_level=None, silence_level=0):
        """
        Initialize an instance of ClimateData.

        Supported file types ``file_type`` are:
          - "NetCDF" for regular (rectangular) grids
          - "iNetCDF" for irregular (e.g. geodesic) grids or station data.

        The :index:`spatio-temporal window` is described by the following
        dictionary::

            window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                      "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        :arg str file_name: The name of the data file.
        :arg str observable_name: The short name of the observable within data
            file (particularly relevant for NetCDF).
        :arg int time_cycle: The annual cycle length of the data (units of
            samples).
        :arg str time_name: The name of the time variable within data file.
        :arg str latitude_name: The name of the latitude variable within data
            file.
        :arg str longitude_name: The name of longitude variable within data
            file.
        :arg str data_source: The name of the data source (model, reanalysis,
            station).
        :arg str file_type: The format of the data file.
        :arg dict window: Spatio-temporal window to select a view on the data.
        :arg int vertical_level: The vertical level to be extracted from the
            data file. Is ignored for horizontal data sets. If None, the first
            level in the data file is chosen.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        dimension_names = {"time": time_name, "lat": latitude_name,
                           "lon": longitude_name}

        #  Load data using _load_data method from parent class
        res = cls._load_data(file_name=file_name, file_type=file_type,
                             dimension_names=dimension_names,
                             observable_name=observable_name,
                             vertical_level=vertical_level)

        #  Create instance of ClimateData
        data = cls(observable=res["observable"], grid=res["grid"],
                   time_cycle=time_cycle,
                   observable_name=res["observable_name"],
                   observable_long_name=res["observable_long_name"],
                   window=window, silence_level=silence_level)

        #  Set class variables
        data.file_name = file_name
        data.file_type = file_type
        data.vertical_level = vertical_level
        data.data_source = data_source

        return data

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

    #
    #  Define methods to work with (climatological) anomaly data
    #

    def phase_indices(self):
        """
        Return time indices associated to all phases in the annual cycle.

        In other words, provides all time indices falling into a particular
        day, month etc. of the year.

        Just includes measurements from years for which complete data exists.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        **Example:**

        >>> ClimateData.SmallTestData().phase_indices()
        array([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])

        :rtype: 2D Numpy array (int) [phase index, year]
        :return: the time indices associated to all phases of the annual cycle.
        """
        range_years = int(self.grid.grid_size()["time"]
                          / self.time_cycle)

        phase_indices = np.zeros((self.time_cycle, range_years), dtype=int)

        for i in range(self.time_cycle):
            phase_indices[i, :] = np.arange(i, range_years * self.time_cycle,
                                            self.time_cycle)

        return phase_indices

    def indices_selected_phases(self, selected_phases):
        """
        Return sorted time indices associated to certain phase indices.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        **Example:**

        >>> ClimateData.SmallTestData().indices_selected_phases([0,1,4])
        array([0, 1, 4, 5, 6, 9])

        :arg [int] selected_phases: The selected phase indices.
        :rtype: 1D array (int)
        :return: the sorted time indices corresponding to chosen phase indices.
        """
        #  Get all
        phase_indices = self.phase_indices()

        #  Select time indices corresponding to chosen phase indices
        selected_indices = phase_indices[selected_phases, :]

        #  Flatten and sort selected time indices
        selected_indices = selected_indices.flatten()
        selected_indices.sort()

        return selected_indices

    def indices_selected_months(self, selected_months):
        """
        Return sorted time indices associated to certain months.

        Currently, only cycle lengths of 12 (monthly data) and 360
        (standardized daily data) are supported.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        :arg [number] selected_months: The selected months.
        :rtype: 1D array (int)
        :return: the sorted time indices corresponding to chosen months.
        """
        if self.time_cycle == 12:
            return self.indices_selected_phases(selected_months)
        elif self.time_cycle == 360:
            selected_days = []
            for month in selected_months:
                for day in range(30):
                    selected_days.append(month * 30 + day)

            return self.indices_selected_phases(selected_days)
        else:
            raise NotImplementedError("Currently only time cycles 12 and 360 \
                                      are supported")

    def _calculate_phase_mean(self):
        """
        Calculate mean values of observable for each phase of the annual cycle.

        This is also commonly referred to as climatological mean, e.g., the
        mean temperature for all Januaries in the data set for monthly time
        resolution (time_cycle=12).

        .. note::
           Only the currently selected spatio-temporal window is considered.

        :rtype: 2D Numpy array [cycle index, node index]
        :return: the mean values of observable for each phase of the annual
                 cycle.
        """
        if self.silence_level <= 1:
            print("Calculating climatological mean values...")

        #  Get raw data
        observable = self.observable()
        #  Get time cycle
        time_cycle = self.time_cycle

        #  Get number of time series
        N = observable.shape[1]

        #  Initialize
        phase_mean = np.zeros((time_cycle, N))

        #  Calculate mean value for each day (month) on each node
        for i in range(time_cycle):
            phase_mean[i, :] = observable[i::time_cycle, :].mean(axis=0)

        return phase_mean

    def phase_mean(self):
        """
        Return mean values of observable for each phase of the annual cycle.

        For further comments, see :meth:`_calculate_phase_mean`.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        **Example:**

        >>> r(ClimateData.SmallTestData().phase_mean())
        array([[ 0.5   ,  0.5   , -0.5   , -0.5   ,  0.5   ,  0.5   ],
               [ 0.63  ,  0.321 , -0.63  , -0.321 ,  0.63  ,  0.321 ],
               [ 0.6984,  0.1106, -0.6984, -0.1106,  0.6984,  0.1106],
               [ 0.6984, -0.1106, -0.6984,  0.1106,  0.6984, -0.1106],
               [ 0.63  , -0.321 , -0.63  ,  0.321 ,  0.63  , -0.321 ]])

        :rtype: 2D Numpy array [cycle index, node index]
        :return: the mean values of observable for each phase of the annual
                 cycle.
        """
        if not self._flag_phase_mean:
            self._phase_mean = self._calculate_phase_mean()
            self._flag_phase_mean = True

        return self._phase_mean

    def _calculate_anomaly(self):
        """
        Calculate anomaly time series from observable.

        To obtain climatological anomaly time series, the climatological means
        are subtracted from each sample in the original time series. This
        procedure is also known as phase averaging.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        :rtype: 2D Numpy array [time, node index]
        :return: the anomalized time series.
        """
        if self.silence_level <= 1:
            print("Calculating daily (monthly) anomaly values...")

        #  Get raw data
        observable = self.observable()
        #  Get time cycle
        time_cycle = self.time_cycle
        #  Initialize array
        anomaly = np.zeros(observable.shape)

        #  Thanks to Jakob Runge
        for i in range(time_cycle):
            sample = observable[i::time_cycle, :]
            anomaly[i::time_cycle, :] = sample - sample.mean(axis=0)

        return anomaly

    def anomaly(self):
        """
        Return anomaly time series from observable.

        For further comments, see :meth:`_calculate_anomaly`.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        **Example:**

        >>> r(ClimateData.SmallTestData().anomaly()[:,0])
        array([-0.5 , -0.321 , -0.1106,  0.1106,  0.321 ,
                0.5 ,  0.321 ,  0.1106, -0.1106, -0.321 ])

        :rtype: 2D Numpy array [time, node index]
        :return: the anomalized time series.
        """
        if not self._flag_anomaly:
            self._anomaly = self._calculate_anomaly()
            self._flag_anomaly = True

        return self._anomaly

    def anomaly_selected_months(self, selected_months):
        """
        Return anomaly time series from observable for selected months.

        For further comments, see :meth:`_calculate_anomaly`.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        :arg [number] selected_months: The selected months.
        :rtype:  2D array [time, node index]
        :return: the anomalized time series for selected months.
        """
        selected_indices = self.indices_selected_months(selected_months)
        print(selected_indices)
        return self.anomaly()[selected_indices, :]

    def shuffled_anomaly(self):
        """
        Return the randomly shuffled anomaly time series.

        Each anomaly time series is shuffled individually.

        .. note::
           Only the currently selected spatio-temporal window is considered.

        **Example** (Anomaly with and without temporal shuffling should have
        the same standard deviation along time axis):

        >>> r(ClimateData.SmallTestData().anomaly().std(axis=0))
        array([ 0.31 , 0.6355, 0.31 , 0.6355, 0.31 , 0.6355])
        >>> r(ClimateData.SmallTestData().shuffled_anomaly().std(axis=0))
        array([ 0.31 , 0.6355, 0.31 , 0.6355, 0.31 , 0.6355])

        :rtype: 2D Numpy array [time, node index]
        :return: the anomalized and shuffled time series.
        """
        if self.silence_level <= 1:
            print("Shuffling anomaly time series for significance tests...")

        N = self.grid.grid_size()["space"]
        shuffled_anomaly = np.empty(self.anomaly().shape)

        for i in range(N):
            temp = self.anomaly()[:, i].copy()
            random.shuffle(temp)
            shuffled_anomaly[:, i] = temp

        return shuffled_anomaly

    def set_window(self, window):
        """
        Set spatio-temporal window.

        Calls set_window method of parent class Data and additionally sets
        flags, so that measures derived from data (mean, anomaly) will be
        recalculated for new window.

        The spatio-temporal window is described by the following dictionary::

           window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                     "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        If the temporal boundaries are equal, the data's full time range is
        selected. If any of the two corresponding spatial boundaries are
        equal, the data's full spatial extension is included.

        For more information see :meth:`pyunicorn.Data.set_window`.

        **Example:**

        >>> data = ClimateData.SmallTestData()
        >>> data.set_window(window={"time_min": 0., "time_max": 0.,
        ...                 "lat_min": 10., "lat_max": 20.,
        ...                 "lon_min": 5.,  "lon_max": 10.})
        >>> r(data.anomaly())
        array([[ 0.5   , -0.5   ], [ 0.321 , -0.63  ], [ 0.1106, -0.6984],
               [-0.1106, -0.6984], [-0.321 , -0.63  ], [-0.5   ,  0.5   ],
               [-0.321 ,  0.63  ], [-0.1106,  0.6984], [ 0.1106,  0.6984],
               [ 0.321 ,  0.63  ]])

        :type window: dictionary
        :arg window: The spatio-temporal window to select a view on the data.
        """
        Data.set_window(self, window)

        self._flag_phase_mean = False
        self._flag_anomaly = False

    def set_global_window(self):
        """
        Set the view on the whole data set.

        Select the full data set and creates a data array as well as
        a corresponding Grid object to access this window from outside.

        **Example** (Set smaller window and subsequently restore global
        window):

        >>> data = ClimateData.SmallTestData()
        >>> data.set_window(window={"time_min": 0., "time_max": 4.,
        ...                 "lat_min": 10., "lat_max": 20.,
        ...                 "lon_min": 5.,  "lon_max": 10.})
        >>> data.grid.grid()["lat"]
        array([ 10.,  15.], dtype=float32)
        >>> data.set_global_window()
        >>> data.grid.grid()["lat"]
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)
        """
        Data.set_global_window(self)

        self._flag_phase_mean = False
        self._flag_anomaly = False
