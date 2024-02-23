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
Provides class for spatio-temporal grids.
"""

from typing import Tuple
from collections.abc import Hashable
import pickle

import numpy as np

from .cache import Cached
from ._ext.types import to_cy, FIELD
from ._ext.numerics import _calculate_euclidean_distance


class Grid(Cached):
    """
    Encapsulates a spatio-temporal grid.

    The spatial grid points can be arbitrarily distributed, which is useful
    for representing station data or geodesic grids.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, time_seq: np.ndarray, space_seq: np.ndarray,
                 silence_level: int = 0):
        """
        Initialize an instance of Grid.

        :type time_seq: 1D Numpy array [time]
        :arg time_seq: The increasing sequence of temporal sampling points.

        :type space_seq: 2D Numpy array [dim, index]
        :arg space_seq: The sequences of spatial sampling points.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.
        """

        #  Set basic dictionaries
        self._grid = {"time": time_seq.astype("float32"),
                      "space": space_seq.astype("float32")}
        self._grid_size = {"time": len(time_seq),
                           "space": space_seq.shape[1]}

        self._boundaries = {"time_min": time_seq.min(),
                            "time_max": time_seq.max(),
                            "space_min": np.amin(space_seq, axis=1),
                            "space_max": np.amax(space_seq, axis=1)}

        #  Defines the number of spatial grid points / nodes at one instant
        #  of time
        self.N = self._grid_size["space"]
        """(number (int)) - The number of spatial grid points / nodes."""

        #  Set silence level
        self.silence_level = silence_level
        """(number (int)) - The inverse level of verbosity of the object."""

        #  Defines the total number of data points / grid points / samples of
        #  the corresponding data set.
        self.n_grid_points = self._grid_size["time"] * self.N
        """(number (int)) - The total number of data points / samples."""

    def __cache_state__(self) -> Tuple[Hashable, ...]:
        # The following attributes are assumed immutable:
        #   (N, _grid)
        return ()

    def __str__(self):
        """
        Return a string representation of the Grid object.
        """
        return (f"Grid: {self._grid_size['space']} grid points, "
                f"{self._grid_size['time']} timesteps.")

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
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except IOError:
            print("An error occurred while saving Grid instance to "
                  f"pickle file {filename}")

    @staticmethod
    def Load(filename):
        """
        Return a Grid object stored in a pickle file.

        :arg str filename: The name of the file where Grid object is stored
            (including ending).
        :rtype: Grid object
        :return: :class:`Grid` instance.
        """
        with open(filename, 'rb') as f:
            grid = pickle.load(f)

        return grid

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
                    space_seq=np.array([[0, 5, 10, 15, 20, 25],
                                        [2.5, 5., 7.5, 10., 12.5, 15.]]),
                    silence_level=2)

    @staticmethod
    def RegularGrid(time_seq, space_grid, silence_level=0):
        """
        Initialize an instance of a regular grid.

        **Examples:**

        >>> Grid.RegularGrid(
        ...      time_seq=np.arange(2),
                 space_grid=[np.array([0.,5.]), np.array([1.,2.])],
                 silence_level=2).sequence(0)
        array([ 0.,  0.,  5.,  5.], dtype=float32)
        >>> Grid.RegularGrid(
        ...     time_seq=np.arange(2),
                space_grid=[np.array([0.,5.]), np.array([1.,2.])],
                silence_level=2).sequence(1)
        array([ 1.,  2.,  1.,  2.], dtype=float32)

        :type time_seq: 1D Numpy array [time]
        :arg time_seq: The increasing sequence of temporal sampling points.

        :type space_grids: list of 1D Numpy arrays [dim, n]
        :arg space_grids: The spatial grid.

        :type silence_level: number (int)
        :arg silence_level: The inverse level of verbosity of the object.

        :rtype: Grid object
        :return: :class:`Grid` instance.
        """
        #  Generate sequences of positions for all nodes
        space_seq = Grid.coord_sequence_from_rect_grid(space_grid)

        #  Return instance of Grid
        return Grid(time_seq, space_seq, silence_level)

    #
    #  Definitions of grid related functions
    #

    @staticmethod
    def coord_sequence_from_rect_grid(space_grid):
        """
        Return the sequences of coordinates for a regular and
        rectangular grid.

        **Example:**

        >>> Grid.coord_sequence_from_rect_grid(
        ...     space_grid=[np.array([0.,5.]), np.array([1.,2.])]
        [array([ 0.,  0.,  5.,  5.]), array([ 1.,  2.,  1.,  2.])]

        :type space_grid: list of 1D Numpy arrays [dim, n]
        :arg space_grid: The grid's sampling points.

        :rtype: list of 1D Numpy arrays [index]
        :return: the coordinates of all nodes in the grid.
        """
        space_seq = np.meshgrid(*space_grid)
        space_seq = np.array([dim_seq.flatten('F') for dim_seq in space_seq])

        return space_seq

    def sequence(self, dimension):
        """
        Return the positional sequence for all nodes for the specified
        dimension.

        **Example:**

        >>> Grid.SmallTestGrid().sequence(0)
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)

        :type dimension: integer
        :arg dimension: The number of the dimension

        :rtype: 1D Numpy array [index]
        :return: the sequence of positions in the specified dimension for all
            nodes.
        """
        return self._grid["space"][dimension]

    def node_number(self, x):
        """
        Return the index of the closest node given euclidean coordinates.

        **Example:**

        >>> Grid.SmallTestGrid().node_number(x=(14., 9.))
        3

        :type x: number (float)
        :arg x: The x coordinate.

        :type y: number (float)
        :arg y: The y coordinate.

        :rtype: number (int)
        :return: the closest node's index.
        """
        x = np.array(x)

        # Get the differences of the coordinates of all nodes to the given
        # coordinates
        diff = self._grid["space"].T - x

        # Get sequences of cosLat, sinLat, cosLon and sinLon for all nodes
        dist = np.sqrt(np.sum(diff**2, axis=1))

        #  Get index of closest node
        n_node = dist.argmin()

        return n_node

    def node_coordinates(self, index):
        """
        Return the position of node ``index``.

        **Example:**

        >>> Grid.SmallTestGrid().node_coordinates(3)
        [15.0, 10.0]

        :type index: number (int)
        :arg index: The node index as used in node sequences.

        :rtype: tuple of number (float)
        :return: the node's coordinates.
        """
        return tuple(self._grid["space"][:, index])

    def distance(self):
        """
        Calculate and return the standard distance matrix of the corresponding
        grid type

        :rtype: 2D Numpy array [index, index]
        :return: the distance matrix.
        """
        return self.euclidean_distance()

    @Cached.method()
    def euclidean_distance(self):
        """
        Return the euclidean distance matrix between grid points.

        **Example:**

        >>> Grid.SmallTestGrid().euclidean_distance().round(2)
        [[ 0.    5.59 11.18 16.77 22.36 27.95]
         [ 5.59  0.    5.59 11.18 16.77 22.36]
         [11.18  5.59  0.    5.59 11.18 16.77]
         [16.77 11.18  5.59  0.    5.59 11.18]
         [22.36 16.77 11.18  5.59  0.    5.59]
         [27.95 22.36 16.77 11.18  5.59  0.  ]]

        :rtype: 2D Numpy array [index, index]
        :return: the euclidean distance matrix.
        """
        #  Get number of nodes
        N_nodes = self.N

        #  Get sequences of coordinates
        sequences = to_cy(self._grid["space"], FIELD)

        #  Get number of dimensions
        N_dim = sequences.shape[0]

        distance = np.zeros((N_nodes, N_nodes), dtype=FIELD)
        _calculate_euclidean_distance(sequences, distance, N_dim, N_nodes)
        return distance

    def boundaries(self):
        """
        Return the spatio-temporal grid boundaries.

        Structure of the returned dictionary:
          - self._boundaries = {"time_min": time_seq.min(),
                                "time_max": time_seq.max(),
                                "space_min": np.amax(space_seq, axis=1),
                                "space_max": np.amin(space_seq, axis=1)}

        :rtype: dictionary
        :return: the spatio-temporal grid boundaries.
        """
        return self._boundaries

    def grid(self):
        """
        Return the grid's spatio-temporal sampling points.

        Structure of the returned dictionary:
          - self._grid = {"time": time_seq.astype("float32"),
                          "space": space_seq.astype("float32")}

        **Examples:**

        >>> Grid.SmallTestGrid().grid()["space"][0]
        array([  0.,   5.,  10.,  15.,  20.,  25.], dtype=float32)
        >>> Grid.SmallTestGrid().grid()["space"][0][5]
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
                               "space": space_seq.shape[1]}

        **Example:**

        >>> print(Grid2D.SmallTestGrid().print_grid_size())
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
        Return the distribution of distances between all pairs of grid points.

        **Examples:**

        >>> Grid.SmallTestGrid().geometric_distance_distribution(3)[0].round(2)
        array([0.33, 0.47, 0.2 ])
        >>> Grid.SmallTestGrid().geometric_distance_distribution(3)[1].round(2)
        array([ 0.  ,  9.32, 18.63, 27.95], dtype=float32)

        :type n_bins: number (int)
        :arg n_bins: The number of histogram bins.

        :rtype: tuple of two 1D Numpy arrays [bin]
        :return: the normalized histogram and lower bin boundaries of
            distances.
        """
        if self.silence_level <= 1:
            print("Calculating the geometric distance distribution of the "
                  "grid...")

        #  Get angular distance matrix
        D = self.distance()

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
