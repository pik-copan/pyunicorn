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
Provides classes for analyzing spatially embedded complex networks, handling
multivariate data and generating time series surrogates.
"""

from typing import Optional

import numpy as np
try:
    from h5netcdf.legacyapi import Dataset
except ImportError:
    try:
        from netCDF4 import Dataset
    except ImportError:
        print("pyunicorn: Packages netCDF4 or h5netcdf could not be loaded. "
              "Some functionality in class Data might not be available!")

from .geo_grid import GeoGrid


class Data:

    """
    Encapsulates general spatio-temporal data.

    Also contains methods to load data from various file formats
    (currently NetCDF and ASCII).

    Mainly an abstract class.
    """

    #
    #  Define internal methods
    #

    def __init__(self, observable: np.ndarray, grid: GeoGrid,
                 observable_name: str = None, observable_long_name: str = None,
                 window: Optional[dict] = None, silence_level: int = 0):
        """
        Initialize an instance of Data.

        The spatio-temporal window is described by the following dictionary::

            window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                      "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        :type observable: 2D array [time, index]
        :arg observable: The array of time series to be represented by the
            :class:`Data` instance.
        :type grid: :class:`.GeoGrid` instance
        :arg grid: The GeoGrid representing the spatial coordinates associated
            to the time series and their temporal sampling.
        :arg str observable_name: A short name for the observable.
        :arg str observable_long_name: A long name for the observable.
        :arg dict window: Spatio-temporal window to select a view on the data.
        :arg int silence_level: The inverse level of verbosity of the object.
        """

        self.silence_level = silence_level
        """(int) - The inverse level of verbosity of the object."""
        self._full_observable = observable

        assert isinstance(grid, GeoGrid)
        self._full_grid = grid
        self.grid = None
        """The :class:`.GeoGrid` object associated with the data."""

        self.observable_name = observable_name
        """(str) - The short name of the observable within
                      data file (particularly relevant for NetCDF)."""

        self.observable_long_name = observable_long_name
        """(str) - The long name of the observable within data file."""

        self._observable = None
        """Current spatio-temporal view on the data."""

        self.file_name = ""
        self.file_type = ""
        self.vertical_level = None

        #  Select a spatio-temporal window to look at the data, can later be
        #  changed in run time without having to reload data from a file.
        #  Force the calling of the Data.setWindow method, since child classes
        #  may overwrite this method.
        if window is None:
            Data.set_global_window(self)
        else:
            Data.set_window(self, window)

    def __str__(self):
        """Return a string representation of the object."""
        if self.file_name:
            self.print_data_info()

        return (f"Data: {self.grid.N} grid points, "
                f"{self.grid.n_grid_points} measurements.\n"
                f"Geographical boundaries:\n{self.grid.print_boundaries()}")

    def set_silence_level(self, silence_level):
        """
        Set the silence level.

        Includes dependent objects such as :attr:`grid`.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.
        """
        self.silence_level = silence_level
        self.grid.silence_level = silence_level
        self._full_grid.silence_level = silence_level

    #
    #  Methods for creating Data objects and alternative constructors
    #

    @classmethod
    def Load(cls, file_name, observable_name, file_type, dimension_names=None,
             window=None, vertical_level=None, silence_level=0):
        """
        Initialize an instance of Data.

        Supported file types ``file_type`` are:
          - "NetCDF" for regular (rectangular) grids
          - "iNetCDF" for irregular (e.g. geodesic) grids or station data.

        The :index:`spatio-temporal window` is described by the following
        dictionary::

            window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                      "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        .. note::
            It is assumed that the NetCDF file to be loaded uses the following
            dimension names: lat, lon, time (e.g., as is the case for
            `NCEP/NCAR reanalysis 1 data <http://www.esrl.noaa.
            gov/psd/data/gridded/data.ncep.reanalysis.html>`_). These standard
            dimension names can be modified using the dimension_names argument.
            Alternatively, the standard class constructor :meth:`__init__`
            needs to be used after loading the data manually, e.g., employing
            netcdf4-python or scipy.io.netcdf functionality.

        :arg str file_name: The name of the data file.
        :arg str observable_name: The short name of the observable within data
            file (particularly relevant for NetCDF).
        :arg str file_type: The type of the data file.
        :arg dict dimension_names: The names of the dimensions as used in the
            NetCDF file. Default: {"lat": "lat", "lon": "lon", "time": "time"}
        :arg dict window: Spatio-temporal window to select a view on the data.
        :arg int vertical_level: The vertical level to be extracted from the
            data file. Is ignored for horizontal data sets. If None, the first
            level in the data file is chosen.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if dimension_names is None:
            dimension_names = {"lat": "lat", "lon": "lon", "time": "time"}

        # Import data from given file
        res = cls._load_data(file_name, file_type, observable_name,
                             dimension_names, vertical_level)

        # Create instance of Data
        data = cls(observable=res["observable"], grid=res["grid"],
                   observable_name=res["observable_name"],
                   observable_long_name=res["observable_long_name"],
                   window=window, silence_level=silence_level)

        # Set some variables
        data.file_name = file_name
        data.file_type = file_type
        data.vertical_level = vertical_level

        return data

    @staticmethod
    def SmallTestData():
        """
        Return test data set of 6 time series with 10 sampling points each.

        **Example:**

        >>> Data.SmallTestData().observable()
        array([[  0.00000000e+00,   1.00000000e+00,   1.22464680e-16,
                 -1.00000000e+00,  -2.44929360e-16,   1.00000000e+00],
               [  3.09016994e-01,   9.51056516e-01,  -3.09016994e-01,
                 -9.51056516e-01,   3.09016994e-01,   9.51056516e-01],
               [  5.87785252e-01,   8.09016994e-01,  -5.87785252e-01,
                 -8.09016994e-01,   5.87785252e-01,   8.09016994e-01],
               [  8.09016994e-01,   5.87785252e-01,  -8.09016994e-01,
                 -5.87785252e-01,   8.09016994e-01,   5.87785252e-01],
               [  9.51056516e-01,   3.09016994e-01,  -9.51056516e-01,
                 -3.09016994e-01,   9.51056516e-01,   3.09016994e-01],
               [  1.00000000e+00,   1.22464680e-16,  -1.00000000e+00,
                 -2.44929360e-16,   1.00000000e+00,   3.67394040e-16],
               [  9.51056516e-01,  -3.09016994e-01,  -9.51056516e-01,
                  3.09016994e-01,   9.51056516e-01,  -3.09016994e-01],
               [  8.09016994e-01,  -5.87785252e-01,  -8.09016994e-01,
                  5.87785252e-01,   8.09016994e-01,  -5.87785252e-01],
               [  5.87785252e-01,  -8.09016994e-01,  -5.87785252e-01,
                  8.09016994e-01,   5.87785252e-01,  -8.09016994e-01],
               [  3.09016994e-01,  -9.51056516e-01,  -3.09016994e-01,
                  9.51056516e-01,   3.09016994e-01,  -9.51056516e-01]])

        :rtype: Data instance
        :return: a Data instance for testing purposes.
        """
        #  Create time series
        ts = np.zeros((10, 6))

        for i in range(6):
            ts[:, i] = np.sin(np.arange(10) * np.pi / 10. + i * np.pi / 2.)

        return Data(observable=ts, grid=GeoGrid.SmallTestGrid(),
                    silence_level=2)

    #
    #  Defines methods to load data from files and display related information
    #

    @classmethod
    def _get_netcdf_data(cls, file_name, file_type, observable_name,
                         dimension_names, vertical_level=None,
                         silence_level=0):
        """
        Import data from a NetCDF file with a regular and rectangular grid.

        Supported file types ``file_type`` are:
          - "NetCDF" for regular (rectangular) grids
          - "iNetCDF" for irregular (e.g. geodesic) grids or station data

        :arg str file_name: The name of the data file.
        :arg str file_type: The format of the data file.
        :arg str observable_name: The short name of the observable within data
            file (particularly relevant for NetCDF).
        :arg dict dimension_names: The names of the dimensions as used in the
            NetCDF file. E.g., dimension_names = {"lat": "lat", "lon": "lon",
            "time": "time"}.
        :arg int vertical_level: The vertical level to be extracted from the
            data file. Is ignored for horizontal data sets. If None, the first
            level in the data file is chosen.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if silence_level <= 1:
            print("Reading NetCDF File and converting data to NumPy array...")

        # Initialize dictionary of results
        res = {}

        # Open netCDF4 file
        f = Dataset(file_name, "r")

        # Create reference to observable
        observable = f.variables[observable_name][:].astype("float32")

        # Get time axis from NetCDF file
        time = f.variables[dimension_names["time"]][:].astype("float32")

        # Get number of dimensions of data
        n_dim = observable.ndim

        # Distinguish between regular and irregular grids
        if file_type == "NetCDF":
            # Create GeoGrid instance
            lat_grid = f.variables[dimension_names["lat"]][:].astype("float32")
            lon_grid = f.variables[dimension_names["lon"]][:].astype("float32")
            res["grid"] = GeoGrid.RegularGrid(time, (lat_grid, lon_grid),
                                              silence_level)

            # If 3D data set (time, lat, lon), select whole data set
            if n_dim == 3:
                res["observable"] = observable.copy()
            # If 4D data set (time, level, lat, lon), select certain vertical
            # level.
            elif n_dim == 4:
                # Handle selected vertical level
                if vertical_level is None:
                    level = 0
                else:
                    level = vertical_level

                res["observable"] = observable[:, level, :, :].copy()
            else:
                print("Regular NetCDF data sets with dimensions other than "
                      "3 (time, lat, lon) or 4 (time, level, lat, lon) are "
                      "not supported by Data class!")

        elif file_type == "iNetCDF":
            # Create GeoGrid instance
            lat_seq = f.variables["grid_center_lat"][:].astype("float32")
            lon_seq = f.variables["grid_center_lon"][:].astype("float32")
            res["grid"] = GeoGrid(time, lat_seq, lon_seq, silence_level)

            # If 2D data set (time, index), select whole data set
            if n_dim == 2:
                res["observable"] = observable.copy()
            # If 3D data set (time, level, index), select certain vertical
            # level.
            elif n_dim == 3:
                # Handle selected vertical level
                if vertical_level is None:
                    level = 0
                else:
                    level = vertical_level

                res["observable"] = observable[:, level, :].copy()
            else:
                print("Irregular NetCDF data sets with dimensions other than "
                      "2 (time, index) or 3 (time, level, index) are not "
                      "supported by Data class!")

        # Get length of raw data time axis
        n_time = res["observable"].shape[0]
        # Reshape observable to comply with the standard shape (time, index)
        res["observable"].shape = (n_time, -1)

        # Get long name of observable
        res["observable_long_name"] = f.variables[observable_name].long_name

        # Store name of observable
        res["observable_name"] = observable_name

        f.close()
        return res

    @classmethod
    def _load_data(cls, file_name, file_type, observable_name,
                   dimension_names, vertical_level=None, silence_level=0):
        """
        Load data into a Numpy array and create a corresponding GeoGrid object.

        Supported file types ``file_type`` are:
          - "NetCDF" for regular (rectangular) grids
          - "iNetCDF" for irregular (e.g. geodesic) grids or station data

        :arg str file_name: The name of the data file.
        :arg str file_type: The format of the data file.
        :arg str observable_name: The short name of the observable within data
            file (particularly relevant for NetCDF).
        :arg dict dimension_names: The names of the dimensions as used in the
            NetCDF file. E.g., dimension_names = {"lat": "lat", "lon": "lon",
            "time": "time"}.
        :arg int vertical_level: The vertical level to be extracted from the
            data file. Is ignored for horizontal data sets. If None, the first
            level in the data file is chosen.
        :arg int silence_level: The inverse level of verbosity of the object.
        """
        if file_type in ["NetCDF", "iNetCDF"]:
            return cls._get_netcdf_data(file_name, file_type, observable_name,
                                        dimension_names, vertical_level,
                                        silence_level)
        else:
            if silence_level <= 1:
                print("This file type can currently not be read "
                      "by pyunicorn.")
            return None

    def print_data_info(self):
        """Print information on the data encapsulated by the Data object."""
        # Open netCDF4 file
        f = Dataset(self.file_name, "r")
        print("Global attributes:")
        for name in f.ncattrs():
            print(name + ":", getattr(f, name))
        print("Variables (size):")
        for name, obj in f.variables.items():
            print(f"{name} ({len(obj)})")
        f.close()

    def observable(self):
        """
        Return the current spatio-temporal view on the data.

        **Example:**

        >>> Data.SmallTestData().observable()[0,:]
        array([  0.00000000e+00,   1.00000000e+00,   1.22464680e-16,
                -1.00000000e+00,  -2.44929360e-16,   1.00000000e+00])

        :rtype: 2D Numpy array [time, space]
        :return: the current spatio-temporal view on the data.
        """
        return self._observable

    #
    #  Defines methods for windowing the data
    #

    def window(self):
        """
        Return the current spatio-temporal window.

        **Examples:**

        >>> Data.SmallTestData().window()["lon_min"]
        2.5

        >>> Data.SmallTestData().window()["lon_max"]
        15.0

        :rtype: dictionary
        :return: the current spatio-temporal window.
        """
        return self.grid.boundaries()

    def set_window(self, window):
        """
        Select a rectangular spatio-temporal region from the data set.

        Create a data array as well as a corresponding GeoGrid object to access
        this window.

        The time axis of the underlying raw data is assumed to be ordered and
        increasing. The latitude and longitude sequences can be arbitrarily
        chosen, i.e., no ordering and no regular grid is required.

        The spatio-temporal window is described by the following dictionary::

           window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                     "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        If the temporal boundaries are equal, the data's full time range is
        selected. If any of the two corresponding spatial boundaries are
        equal, the data's full spatial extension is included.

        **Example:**

        >>> data = Data.SmallTestData()
        >>> data.set_window(window={
        ...     "time_min": 0., "time_max": 4., "lat_min": 10.,
        ...     "lat_max": 20., "lon_min": 5., "lon_max": 10.})
        >>> data.observable()
        array([[  1.22464680e-16,  -1.00000000e+00],
               [ -3.09016994e-01,  -9.51056516e-01],
               [ -5.87785252e-01,  -8.09016994e-01],
               [ -8.09016994e-01,  -5.87785252e-01],
               [ -9.51056516e-01,  -3.09016994e-01]])

        :type window: dictionary
        :arg window: A spatio-temporal window to select a view on the data.
        """
        # Collect arrays
        full_time = self._full_grid.grid()["time"]
        full_lat_seq = self._full_grid.grid()["lat"]
        full_lon_seq = self._full_grid.grid()["lon"]

        # Get time indices for temporal window boundaries
        if window["time_min"] == window["time_max"]:
            # If boundaries time are equal, use all available time points
            time_indices = np.repeat(True,
                                     self._full_grid.grid_size()["time"])
        else:
            # Get indices for chosen time boundaries
            time_indices = (full_time >= window["time_min"]) & \
                           (full_time <= window["time_max"])

        # Get indices of nodes lying within the prescribed spatial
        # window boundaries
        if (window["lat_min"] == window["lat_max"]) \
           or (window["lon_min"] == window["lon_max"]):
            # If boundaries in latitude or longitude are equal, use all nodes
            # from the full spatial grid.
            # space_indices is an array of bool indicating whether a node
            # lies within the window or not.
            space_indices = np.repeat(True,
                                      self._full_grid.grid_size()["space"])
        else:
            # space_indices is an array of bool indicating whether a node
            # lies within the window or not.
            space_indices = (full_lat_seq >= window["lat_min"]) & \
                            (full_lat_seq <= window["lat_max"]) & \
                            (full_lon_seq >= window["lon_min"]) & \
                            (full_lon_seq <= window["lon_max"])

        # Set windowed observable and grid object
        time = full_time[time_indices]
        lat_seq = full_lat_seq[space_indices]
        lon_seq = full_lon_seq[space_indices]

        self._observable = \
            self._full_observable[time_indices, :][:, space_indices]
        self.grid = GeoGrid(time, lat_seq, lon_seq, self.silence_level)

    def set_global_window(self):
        """
        Set the view on the whole data set.

        Select the full data set and creates a data array as well as
        a corresponding GeoGrid object to access this window from outside.

        **Example** (Set smaller window and subsequently restore global
        window):

        >>> data = Data.SmallTestData()
        >>> data.set_window(window={"time_min": 0., "time_max": 4.,
        ...                 "lat_min": 10., "lat_max": 20., "lon_min": 5.,
        ...                 "lon_max": 10.})
        >>> data.grid.grid()["lat"]
        array([ 10.,  15.], dtype=float32)
        >>> data.set_global_window()
        >>> data.grid.grid()["lat"]
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)
        """
        global_window = {"time_min": 0., "time_max": 0., "lat_min": 0.,
                         "lat_max": 0., "lon_min": 0., "lon_max": 0.}

        self.set_window(global_window)

    #
    # Define methods for reformatting data
    #

    # TODO: Documentation.
    @staticmethod
    def rescale(array, var_type):
        """
        Rescale an array to a given data type.

        Returns the tuple
        (scaled_array, scale_factor, add_offset, actual_range).
        Allows flexible handling of final amount of
        used storage volume for the file.

        :type array:
        :arg array:

        :arg str var_type: Determines the desired final data type of the array.
        """
        # TODO: Add example

        ar_max = array.max()
        ar_min = array.min()
        actual_range = np.array([ar_min, ar_max])

        if var_type == 'float64':
            scaled_array = array.astype('float64')
            scale_factor = 1.
            add_offset = 0.
        elif var_type == 'float32':
            scaled_array = array.astype('float32')
            scale_factor = 1.
            add_offset = 0.
        elif var_type == 'int32':
            scale_factor = (ar_max - ar_min) / (2. * 2. ** 31 - 2.)
            add_offset = (ar_max + ar_min) / 2.
            array -= add_offset
            array /= scale_factor
            scaled_array = array.astype("int32")
        elif var_type == 'int16':
            scale_factor = (ar_max - ar_min) / (2. * 2. ** 15 - 2.)
            add_offset = (ar_max + ar_min) / 2.
            array -= add_offset
            array /= scale_factor
            scaled_array = array.astype('int16')
        elif var_type == 'uint8':
            scale_factor = (ar_max - ar_min) / (2. ** 8 - 1.)
            add_offset = ar_min
            array -= add_offset
            array /= scale_factor
            scaled_array = array.astype('uint8')
        else:
            raise ValueError(f"Data type {var_type} not supported.")

        return (scaled_array, scale_factor, add_offset, actual_range)

    #
    #  Define methods to prepare data for similarity measure calculation
    #

    @staticmethod
    def normalize_time_series_array(time_series_array):
        """
        :index:`Normalize <pair: normalize; time series array>` an array of
        time series to zero mean and unit variance individually for each
        individual time series.

        Works also for complex valued time series.

        **Modifies the given array in place!**

        **Example:**

        >>> ts = np.arange(16).reshape(4,4).astype("float")
        >>> Data.normalize_time_series_array(ts)
        >>> ts.mean(axis=0)
        array([ 0.,  0.,  0.,  0.])
        >>> ts.std(axis=0)
        array([ 1.,  1.,  1.,  1.])
        >>> ts[:,0]
        array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])

        :type time_series_array: 2D Numpy array [time, index]
        :arg time_series_array: The time series array to be normalized.
        """
        #  Remove mean value from time series at each node (grid point)
        time_series_array -= time_series_array.mean(axis=0)

        #  Normalize the variance of anomalies to one
        time_series_array /= np.sqrt(
            (time_series_array * time_series_array.conjugate()).mean(axis=0))

        #  Correct for grid points with zero variance in their time series
        time_series_array[np.isnan(time_series_array)] = 0

    @staticmethod
    def next_power_2(i):
        """
        Return the power of two 2^n, that is greater or equal than i.

        **Example:**

        >>> Data.next_power_2(253)
        256

        :type i: number (float)
        :arg i: Some real number.

        :rtype: number (float)
        :return: the power of two greater of equal than a given value.
        """
        n = 2
        while n < i:
            n = n * 2

        return n

    @staticmethod
    def zero_pad_data(data):
        """
        Return :index:`zero padded data`, such that the length of individual
        time series is a power of 2.

        **Example:**

        >>> ts = np.arange(20).reshape(5,4)
        >>> Data.zero_pad_data(ts)
        array([[  0.,   0.,   0.,   0.], [  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.], [  8.,   9.,  10.,  11.],
               [ 12.,  13.,  14.,  15.], [ 16.,  17.,  18.,  19.],
               [  0.,   0.,   0.,   0.], [  0.,   0.,   0.,   0.]])

        :type data: 2D Numpy array [time, index]
        :arg data: The data array to be zero padded.

        :rtype: 2D Numpy array [time, index]
        :return: the zero padded data array.
        """
        (n_time, n_nodes) = data.shape

        #  Get the power of n that is larger or equal than the length of
        #  individual time series.
        n = Data.next_power_2(n_time)

        zeros_before = (n - n_time) // 2
        zeros_after = (n - n_time) - (n - n_time) // 2

        before = np.zeros((zeros_before, n_nodes))
        after = np.zeros((zeros_after, n_nodes))

        return np.concatenate((before, data, after), axis=0)

    @staticmethod
    def cos_window(data, gamma):
        """
        Return a cosine window fitting the shape of the data argument.

        The window is one for most of the time and goes to zero at the
        boundaries of each time series in the data array.

        The width of the cosine shaped decay region is controlled by the shape
        parameter gamma:

          - Gamma=1 means, that each of the two decay regions extends over
            half of the time series.
          - Gamma=0 means, that the decay regions vanish and the window
            transformation becomes the identity.

        **Example:**

        >>> ts = np.arange(24).reshape(12,2)
        >>> Data.cos_window(data=ts, gamma=0.75)
        array([[ 0.        ,  0.        ], [ 0.14644661,  0.14644661],
               [ 0.5       ,  0.5       ], [ 0.85355339,  0.85355339],
               [ 1.        ,  1.        ], [ 1.        ,  1.        ],
               [ 1.        ,  1.        ], [ 1.        ,  1.        ],
               [ 0.85355339,  0.85355339], [ 0.5       ,  0.5       ],
               [ 0.14644661,  0.14644661], [ 0.        ,  0.        ]])

        :type data: 2D Numpy array [time, index]
        :arg data: The data array to be fitted by cosine window.

        :type gamma: number (float)
        :arg gamma: The cosine window shape parameter.

        :rtype: 2D Numpy array [time, index]
        :return: the cosine window fitting data array.
        """
        (n_time, n_nodes) = data.shape

        #  Calculate length of decay regions
        decay_length = int(gamma * n_time / 2)

        #  Calculate decay and growth regions
        growth_region = 0.5 * (1 + np.cos(
            np.arange(decay_length) * np.pi / float(decay_length) + np.pi))
        growth_region = np.tile(growth_region, (n_nodes, 1))
        growth_region = growth_region.transpose()

        decay_region = growth_region[::-1, :].copy()

        return np.concatenate((growth_region,
                               np.ones((n_time - 2 * decay_length, n_nodes)),
                               decay_region), axis=0)
