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

#
#  Import essential packages
#

#  Import pickle for loading and saving Python objects
import pickle

#  array object and fast numerics
import numpy as np

# Import package to calculate points inside a polygon
try:
    from matplotlib import path
except ImportError:
    print("An error occurred when importing matplotlib.path! "
          "Some functionality in Grid class might not be available.")

#  Cythonized functions
from ._ext.numerics import _cy_calculate_angular_distance, _euclidean_distance

#
#  Define class Grid
#


class Grid:

    """
    Encapsulates a horizontal spatio-temporal grid on the sphere.

    The spatial grid points can be arbitrarily distributed, which is useful
    for representing station data or geodesic grids.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, time_seq, lat_seq, lon_seq, silence_level=0):
        """
        Initialize an instance of Grid.

        :type time_seq: 1D Numpy array [time]
        :arg time_seq: The increasing sequence of temporal sampling points.

        :type lat_seq: 1D Numpy array [index]
        :arg lat_seq: The sequence of latitudinal sampling points.

        :type lon_seq: 1D Numpy array [index]
        :arg lon_seq: The sequence of longitudinal sampling points.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.
        """
        #  Set basic dictionaries
        self._grid = {"time": time_seq.astype("float32"),
                      "lat": lat_seq.astype("float32"),
                      "lon": lon_seq.astype("float32")}
        self._grid_size = {"time": len(time_seq),
                           "space": len(lat_seq)}
        self._boundaries = {"time_min": time_seq.min(),
                            "time_max": time_seq.max(),
                            "lat_min": lat_seq.min(),
                            "lat_max": lat_seq.max(),
                            "lon_min": lon_seq.min(),
                            "lon_max": lon_seq.max()}

        #  Set silence level
        self.silence_level = silence_level
        """(number (int)) - The inverse level of verbosity of the object."""

        #  Defines the number of spatial grid points / nodes at one instant
        #  of time
        self.N = self._grid_size["space"]
        """(number (int)) - The number of spatial grid points / nodes."""

        #  Defines the total number of data points / grid points / samples of
        #  the corresponding data set.
        self.n_grid_points = self._grid_size["time"] * self.N
        """(number (int)) - The total number of data points / samples."""

        #  Cache
        self._angular_distance = None
        self._angular_distance_cached = False

    def __str__(self):
        """
        Return a string representation of the Grid object.
        """
        return 'Grid: %i grid points, %i timesteps.' % (
            self._grid_size['space'], self._grid_size['time'])

    def clear_cache(self):
        """
        Clean up cache.

        Is reversible, since all cached information can be recalculated from
        basic data.
        """
        if self._angular_distance_cached:
            del self._angular_distance
            self._angular_distance_cached = False

    #
    #  Functions for loading and saving the Grid object
    #

    def save(self, filename):
        """
        Save the Grid object to a pickle file.

        :arg str filename: The name of the file where Grid object is stored
            (including ending).
        """
        try:
            f = open(filename, 'w')
            pickle.dump(self, f)
            f.close()
        except IOError:
            print("An error occurred while saving Grid instance to "
                  f"pickle file {filename}")

    def save_txt(self, filename):
        """
        Save the Grid object to text files.

        The latitude, longitude and time sequences are stored in three separate
        text files.

        :arg str filename: The name of the files where Grid object is stored
            (excluding ending).
        """
        #  Gather sequences
        lat_seq = self.lat_sequence()
        lon_seq = self.lon_sequence()
        time_seq = self.grid()["time"]

        #  Store as text files
        try:
            np.savetxt(filename + "_lat.txt", lat_seq)
            np.savetxt(filename + "_lon.txt", lon_seq)
            np.savetxt(filename + "_time.txt", time_seq)
        except IOError:
            print("An error occurred while saving Grid instance to "
                  f"text files {filename}")

    @staticmethod
    def Load(filename):
        """
        Return a Grid object stored in a pickle file.

        :arg str filename: The name of the file where Grid object is stored
            (including ending).
        :rtype: Grid object
        :return: :class:`Grid` instance.
        """
        try:
            f = open(filename, 'r')
            grid = pickle.load(f)
            f.close()

            return grid
        except IOError:
            print("An error occurred while loading Grid instance from "
                  f"pickle file {filename}")

    @staticmethod
    def LoadTXT(filename):
        """
        Return a Grid object stored in text files.

        The latitude, longitude and time sequences are loaded from three
        separate text files.

        :arg str filename: The name of the files where Grid object is stored
            (excluding endings).
        :rtype: Grid object
        :return: :class:`Grid` instance.
        """
        try:
            lat_seq = np.loadtxt(filename + "_lat.txt")
            lon_seq = np.loadtxt(filename + "_lon.txt")
            time_seq = np.loadtxt(filename + "_time.txt")
        except IOError:
            print("An error occurred while loading Grid instance from "
                  f"text files {filename}")

        return Grid(time_seq, lat_seq, lon_seq)

    #
    #  Alternative constructors and Grid generation methods
    #

    @staticmethod
    def SmallTestGrid():
        """
        Return test grid of 6 spatial grid points with 10 temporal sampling
        points each.

        :rtype: Grid instance
        :return: a Grid instance for testing purposes.
        """
        return Grid(time_seq=np.arange(10),
                    lat_seq=np.array([0, 5, 10, 15, 20, 25]),
                    lon_seq=np.array([2.5, 5., 7.5, 10., 12.5, 15.]),
                    silence_level=2)

    @staticmethod
    def RegularGrid(time_seq, lat_grid, lon_grid, silence_level=0):
        """
        Initialize an instance of a regular grid.

        **Examples:**

        >>> Grid.RegularGrid(
        ...     time_seq=np.arange(2), lat_grid=np.array([0.,5.]),
        ...     lon_grid=np.array([1.,2.]), silence_level=2).lat_sequence()
        array([ 0.,  0.,  5.,  5.], dtype=float32)
        >>> Grid.RegularGrid(
        ...     time_seq=np.arange(2), lat_grid=np.array([0.,5.]),
        ...     lon_grid=np.array([1.,2.]), silence_level=2).lon_sequence()
        array([ 1.,  2.,  1.,  2.], dtype=float32)

        :type time_seq: 1D Numpy array [time]
        :arg time_seq: The increasing sequence of temporal sampling points.

        :type lat_grid: 1D Numpy array [n_lat]
        :arg lat_grid: The latitudinal grid.

        :type lon_grid: 1D Numpy array [n_lon]
        :arg lon_grid: The longitudinal grid.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.

        :rtype: Grid object
        :return: :class:`Grid` instance.
        """
        #  Generate sequence of latitudes and longitudes for all nodes
        lat_seq, lon_seq = Grid.coord_sequence_from_rect_grid(lat_grid,
                                                              lon_grid)

        #  Return instance of Grid
        return Grid(time_seq, lat_seq, lon_seq, silence_level)

    #
    #  Definitions of grid related functions
    #

    @staticmethod
    def coord_sequence_from_rect_grid(lat_grid, lon_grid):
        """
        Return the sequences of latitude and longitude for a regular and
        rectangular grid.

        **Example:**

        >>> Grid.coord_sequence_from_rect_grid(
        ...     lat_grid=np.array([0.,5.]), lon_grid=np.array([1.,2.]))
        (array([ 0.,  0.,  5.,  5.]), array([ 1.,  2.,  1.,  2.]))

        :type lat_grid: 1D Numpy array [lat]
        :arg lat_grid: The grid's latitudinal sampling points.

        :type lon_grid: 1D Numpy array [lon]
        :arg lon_grid: The grid's longitudinal sampling points.

        :rtype: tuple of two 1D Numpy arrays [index]
        :return: the coordinates of all nodes in the grid.
        """
        #  Get grid size
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)
        n = n_lat * n_lon

        #  Initialize sequences
        lat_seq = np.empty(n)
        lon_seq = np.empty(n)

        for i in range(n_lat):
            lat_seq[i * n_lon:(i + 1) * n_lon] = lat_grid[i] * np.ones(n_lon)
            lon_seq[i * n_lon:(i + 1) * n_lon] = lon_grid

        #  Return results as a tuple
        return (lat_seq, lon_seq)

    def lat_sequence(self):
        """
        Return the sequence of latitudes for all nodes.

        **Example:**

        >>> Grid.SmallTestGrid().lat_sequence()
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)

        :rtype: 1D Numpy array [index]
        :return: the sequence of latitudes for all nodes.
        """
        return self._grid["lat"]

    def lon_sequence(self):
        """
        Return the sequence of longitudes for all nodes.

        **Example:**

        >>> Grid.SmallTestGrid().lon_sequence()
        array([  2.5,   5. ,   7.5,  10. ,  12.5,  15. ], dtype=float32)

        :rtype: 1D Numpy array [index]
        :return: the sequence of longitudes for all nodes.
        """
        return self._grid["lon"]

    def convert_lon_coordinates(self, lon_seq):
        """
        Return longitude coordinates in the system
        -180 deg W <= lon <= +180 deg O for all nodes.

        Accepts longitude coordinates in the system 0 deg <= lon <= 360 deg.
        0 deg corresponds to Greenwich, England.

        **Example:**

        >>> Grid.SmallTestGrid().convert_lon_coordinates(
        ...     np.array([10.,350.,20.,340.,170.,190.]))
        array([  10.,  -10.,   20.,  -20.,  170., -170.])

        :type lon_seq: 1D Numpy array [index]
        :arg lon_seq: Sequence of longitude coordinates.

        :rtype: 1D Numpy array [index]
        :return: the converted longitude coordinates for all nodes.
        """
        new_lon_grid = np.empty(self.N)

        for i in range(self.N):
            if lon_seq[i] > 180.:
                new_lon_grid[i] = lon_seq[i] - 360.
            else:
                new_lon_grid[i] = lon_seq[i]

        return new_lon_grid

    def node_coordinates(self, index):
        """
        Return the geographical latitude and longitude of node ``index``.

        **Example:**

        >>> Grid.SmallTestGrid().node_coordinates(3)
        (15.0, 10.0)

        :type index: number (int)
        :arg index: The node index as used in node sequences.

        :rtype: tuple of number (float)
        :return: the node's latitude and longitude coordinates.
        """
        lat_node = self.lat_sequence()[index]
        lon_node = self.lon_sequence()[index]
        return (lat_node, lon_node)

    def node_number(self, lat_node, lon_node):
        """
        Return the index of the closest node given geographical coordinates.

        **Example:**

        >>> Grid.SmallTestGrid().node_number(lat_node=14., lon_node=9.)
        3

        :type lat_node: number (float)
        :arg lat_node: The latitude coordinate.

        :type lon_node: number (float)
        :arg lon_node: The longitude coordinate.

        :rtype: number (int)
        :return: the closest node's index.
        """
        #  Get sequences of cosLat, sinLat, cosLon and sinLon for all nodes
        cos_lat = self.cos_lat()
        sin_lat = self.sin_lat()
        cos_lon = self.cos_lon()
        sin_lon = self.sin_lon()

        sin_lat_v = np.sin(lat_node * np.pi / 180)
        cos_lat_v = np.cos(lat_node * np.pi / 180)
        sin_lon_v = np.sin(lon_node * np.pi / 180)
        cos_lon_v = np.cos(lon_node * np.pi / 180)

        #  Calculate angular distance from the given coordinate to all
        #  other nodes
        expr = sin_lat*sin_lat_v + cos_lat*cos_lat_v * (sin_lon*sin_lon_v
                                                        + cos_lon*cos_lon_v)

        #  Correct for rounding errors
        expr[expr < -1.] = -1.
        expr[expr > 1.] = 1.

        angdist = np.arccos(expr)

        #  Get index of closest node
        n_node = angdist.argmin()

        return n_node

    def cos_lat(self):
        """
        Return the sequence of cosines of latitude for all nodes.

        **Example:**

        >>> r(Grid.SmallTestGrid().cos_lat()[:2])
        array([ 1. , 0.9962])

        :rtype: 1D Numpy array [index]
        :return: the cosine of latitudes for all nodes.
        """
        return np.cos(self.lat_sequence() * np.pi / 180)

    def sin_lat(self):
        """
        Return the sequence of sines of latitude for all nodes.

        **Example:**

        >>> r(Grid.SmallTestGrid().sin_lat()[:2])
        array([ 0. , 0.0872])

        :rtype: 1D Numpy array [index]
        :return: the sine of latitudes for all nodes.
        """
        return np.sin(self.lat_sequence() * np.pi / 180)

    def cos_lon(self):
        """
        Return the sequence of cosines of longitude for all nodes.

        **Example:**

        >>> r(Grid.SmallTestGrid().cos_lon()[:2])
        array([ 0.999 , 0.9962])

        :rtype: 1D Numpy array [index]
        :return: the cosine of longitudes for all nodes.
        """
        return np.cos(self.lon_sequence() * np.pi / 180)

    def sin_lon(self):
        """
        Return the sequence of sines of longitude for all nodes.

        **Example:**

        >>> r(Grid.SmallTestGrid().sin_lon()[:2])
        array([ 0.0436, 0.0872])

        :rtype: 1D Numpy array [index]
        :return: the sine of longitudes for all nodes.
        """
        return np.sin(self.lon_sequence() * np.pi / 180)

    def _calculate_angular_distance(self):
        """
        Calculate and return the angular great circle distance matrix.

        **No normalization applied anymore!** Return values are in the range
        0 to Pi.

        :rtype: 2D Numpy array [index, index]
        :return: the angular great circle distance matrix (unit radians).
        """
        if self.silence_level <= 1:
            print("Calculating angular great circle distance using Cython...")

        #  Get number of nodes
        N = self.N

        #  Get sequences of cosLat, sinLat, cosLon and sinLon for all nodes
        cos_lat = self.cos_lat()
        sin_lat = self.sin_lat()
        cos_lon = self.cos_lon()
        sin_lon = self.sin_lon()

        #  Initialize cython cof of angular distance matrix
        cosangdist = np.zeros((N, N), dtype="float32")

        _cy_calculate_angular_distance(cos_lat, sin_lat, cos_lon, sin_lon,
                                       cosangdist, N)
        return np.arccos(cosangdist)

    def angular_distance(self):
        """
        Return the angular great circle distance matrix.

        **No normalization applied anymore!** Return values are in the range
        0 to Pi.

        **Example:**

        >>> rr(Grid.SmallTestGrid().angular_distance(), 2)
        [['0'    '0.1'  '0.19' '0.29' '0.39' '0.48']
         ['0.1'  '0'    '0.1'  '0.19' '0.29' '0.39']
         ['0.19' '0.1'  '0'    '0.1'  '0.19' '0.29']
         ['0.29' '0.19' '0.1'  '0'    '0.1'  '0.19']
         ['0.39' '0.29' '0.19' '0.1'  '0'    '0.1']
         ['0.48' '0.39' '0.29' '0.19' '0.1'  '0']]

        :rtype: 2D Numpy array [index, index]
        :return: the angular great circle distance matrix.
        """
        if not self._angular_distance_cached:
            self._angular_distance = self._calculate_angular_distance()
            self._angular_distance_cached = True

        return self._angular_distance

    def euclidean_distance(self):
        """
        Return the euclidean distance matrix between grid points.

        So far assumes that the given latitude and longitude coordinates are
        planar, euclidean coordinates. Approximates great circle distance well
        for points with a separation much smaller than the Earth's radius.

        :rtype: 2D Numpy array [index, index]
        :return: the euclidean distance matrix.
        """
        #  Get number of nodes
        N = self.N

        #  Get sequences of coordinates
        x = self.lon_sequence()
        y = self.lat_sequence()

        distance = np.zeros((N, N), dtype="float32")
        _euclidean_distance(x, y, distance, N)

        return distance

    def boundaries(self):
        """
        Return the spatio-temporal grid boundaries.

        Structure of the returned dictionary:
          - self._boundaries = {"time_min": time_seq.min(),
                                "time_max": time_seq.max(),
                                "lat_min": lat_seq.min(),
                                "lat_max": lat_seq.max(),
                                "lon_min": lon_seq.min(),
                                "lon_max": lon_seq.max()}

        **Example:**

        >>> print(Grid.SmallTestGrid().print_boundaries())
                 time     lat     lon
           min    0.0    0.00    2.50
           max    9.0   25.00   15.00

        :rtype: dictionary
        :return: the spatio-temporal grid boundaries.
        """
        return self._boundaries

    def print_boundaries(self):
        """
        Pretty print the spatio-temporal grid boundaries.
        """
        return (
            "         time     lat     lon"
            "\n   min {time_min:6.1f} {lat_min: 7.2f} {lon_min: 7.2f}"
            "\n   max {time_max:6.1f} {lat_max: 7.2f} {lon_max: 7.2f}"
        ).format(**self.boundaries())

    def grid(self):
        """
        Return the grid's spatio-temporal sampling points.

        Structure of the returned dictionary:
          - self._grid = {"time": time_seq.astype("float32"),
                          "lat": lat_seq.astype("float32"),
                          "lon": lon_seq.astype("float32")}

        **Examples:**

        >>> Grid.SmallTestGrid().grid()["lat"]
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)
        >>> Grid.SmallTestGrid().grid()["lon"][5]
        15.0

        :rtype: dictionary
        :return: the grid's spatio-temporal sampling points.
        """
        return self._grid

    def grid_size(self):
        """
        Return the sizes of the grid's spatial and temporal dimensions.

        Structure of the returned dictionary:
          - self._grid_size = {"time": len(time_seq),
                               "space": len(lat_seq)}

        **Example:**

        >>> print(Grid.SmallTestGrid().print_grid_size())
           space    time
               6      10

        :rtype: dictionary
        :return: the sizes of the grid's spatial and temporal dimensions.
        """
        return self._grid_size

    def print_grid_size(self):
        """
        Pretty print the sizes of the grid's spatial and temporal dimensions.
        """
        return "     space    time\n   {space:7} {time:7}".format(
            **self.grid_size())

    def geometric_distance_distribution(self, n_bins):
        """
        Return the distribution of angular great circle distances between all
        pairs of grid points.

        **Examples:**

        >>> r(Grid.SmallTestGrid().geometric_distance_distribution(3)[0])
        array([ 0.3333, 0.4667, 0.2 ])
        >>> r(Grid.SmallTestGrid().geometric_distance_distribution(3)[1])
        array([ 0. , 0.1616, 0.3231, 0.4847])

        :type n_bins: number (int)
        :arg n_bins: The number of histogram bins.

        :rtype: tuple of two 1D Numpy arrays [bin]
        :return: the normalized histogram and lower bin boundaries of angular
                 great circle distances.
        """
        if self.silence_level <= 1:
            print("Calculating the geometric distance distribution of the "
                  "grid...")

        #  Get angular distance matrix
        D = self.angular_distance()

        #  Determine range for link distance histograms
        max_range = D.max()
        interval = (0, max_range)

        #  Calculate geometry related factor of distributions to divide it out
        (dist, lbb) = np.histogram(a=D, bins=n_bins, range=interval)
        #  Subtract self.N from first bin because of spurious links with zero
        #  distance on the diagonal of the angular distance matrix
        dist[0] -= self.N

        #  Normalize distribution
        dist = dist.astype("float")
        dist /= dist.sum()

        return (dist, lbb)

    #
    #  Methods for selecting regions
    #

    def region_indices(self, region):
        """
        Returns a boolean array of nodes with True values when the node
        is inside the region.

        **Example:**

        >>> Grid.SmallTestGrid().region_indices(
        ...     np.array([0.,0.,0.,11.,11.,11.,11.,0.])).astype(int)
        array([0, 1, 1, 0, 0, 0])

        :type region: 1D Numpy array [n_polygon_nodes]
        :arg region: array of lon, lat, lon, lat, ...
                     [-80.2, 5., -82.4, 5.3, ...] as copied from Google Earth
                     Polygon file
        :rtype: 1D bool array [index]
        :return: bool array with True for nodes inside region
        """
        # Reshape Google Earth array  into (n,2) array
        remapped_region = region.reshape(len(region)//2, 2)
        # Remap from East-West to 360 degree map if the longitudes are [0, 360]
        if self._grid["lon"].min() >= 0:
            remapped_region[remapped_region[:, 0] < 0, 0] = \
                360 + remapped_region[remapped_region[:, 0] < 0, 0]

        lat_lon_map = np.column_stack((self._grid["lon"], self._grid["lat"]))

        return path.Path(remapped_region).contains_points(lat_lon_map)

    @staticmethod
    def region(name):
        """Return some standard regions."""
        if name == 'ENSO':
            return np.array([-79.28273150749884, -10.49311965331937,
                             -79.29849791038734, 10.12527300655218,
                             -174.9221853596061, 10.07293121423917,
                             -174.8362810586096, -10.46407198776264,
                             -80.13229308153623, -10.36724072894785,
                             -79.28273150749884, -10.49311965331937])
        elif name == 'NINO34':
            return np.array([-118.6402427933005, 7.019906838300821,
                             -171.0067408177714, 6.215022481004243,
                             -171.0364908514962, -5.768616252424354,
                             -119.245702264066, -5.836385150138187,
                             -118.6402427933005, 7.019906838300821])
        else:
            return None
