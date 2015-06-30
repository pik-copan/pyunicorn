#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides class for event synchronization measures.

Written by Wolfram Barfuss.
"""

# array object and fast numerics
import numpy as np

# warnings
import warnings


class EventSynchronization(object):

    """
    Contains methods to calculate event synchronization matrices from event
    series. The entries of these matrices represent some variant of the event
    synchronization between two of the varaibles.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, eventmatrix, taumax):
        """
        Initialize an instance of EventSynchronization.

        Format of eventmatrix:
        An eventmatrix is a 2D numpy array with the first dimension covering
        the the timesteps and the second dimensions covering the variables.
        Each variable at a specific timestep is either 1 if an event occured or
        0 if it did not, e.g. for 3 variables with 10 timesteps the evenmatrix
        could look like

            array([[0, 1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 1],
                   [0, 1, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])

        Important note!!!
        Due to normalization issues the event synchronization matrices should
        only be used if the number of events (i.e the number 1s) in each
        variable is identical.

        :type eventmatrix: 2D Numpy array [time, variables]
        :arg evenmatrix: Event series array
        :type taumax: int
        :arg taumax: Maximum dynamical delay
        """

        self.__T = eventmatrix.shape[0]
        self.__N = eventmatrix.shape[1]
        self.__eventmatrix = eventmatrix
        self.taumax = taumax

        # Check for right input format
        if len(np.unique(eventmatrix)) != 2 or not (np.unique(eventmatrix) ==
                                                    np.array([0, 1])).all():
            raise ValueError("Eventmatrix not in correct format")

        # Print warning if number of events is not identical for all variables
        NrOfEvs = np.sum(eventmatrix, axis=0)
        if not (NrOfEvs == NrOfEvs[0]).all():
            warnings.warn("Data does not contain equal number of events")

    def __str__(self):
        """
        Return a string representation of the EventSynchronization object.
        """
        text = EventSynchronization.__str__(self)

        return text

    #
    #  Definitions of event synchronization measures
    #

    def directed(self):
        """
        Returns the NxN matrix of the directed event synchronization measure.
        The entry [i, j] denotes the directed event synchrnoization from
        variable j to variable i.
        """
        eventmatrix = self.__eventmatrix
        res = np.ones((self.__N, self.__N)) * np.inf

        for i in np.arange(0, self.__N):
            for j in np.arange(i+1, self.__N):
                res[i, j], res[j, i] = self._EventSync(eventmatrix[:, i],
                                                       eventmatrix[:, j])
        return res

    def symmetric(self):
        """
        Returns the NxN matrix of the undirected or symmetrix event
        synchronization measure. It is obtained by the sum of both directed
        versions.
        """
        directed = self.directed()
        return directed + directed.T

    def antisymmetric(self):
        """
        Returns the NxN matrix of the antisymmetric synchronization measure.
        It is obtained by the difference of both directed versions.
        """
        directed = self.directed()
        return directed - directed.T

    def _EventSync(self, EventSeriesX, EventSeriesY):
        """
        Calculates the directed event synchronization from two event series X
        and Y.

        :type EventSeriesX: 1D Numpy array
        :arg EventSeriesX: Event series containing '0's and '1's
        :type EventSeriesY: 1D Numpy array
        :arg EventSeriesY: Event series containing '0's and '1's
        :rtype: tuple
        :return: (Event synchronization XY, Event synchronization YX)

        """

        # Change format: (list of timesteps of events)
        ex = np.arange(len(EventSeriesX))[EventSeriesX == 1]
        ey = np.arange(len(EventSeriesY))[EventSeriesY == 1]
        # Number of events
        lx = len(ex)
        ly = len(ey)

        # Vectorized calculation
        EX = np.reshape(np.repeat(ex, ly), (lx, ly), 'C')
        EY = np.reshape(np.repeat(ey, lx), (lx, ly), 'F')

        DSTxy = EX[1:-1, 1:-1] - EY[1:-1, 1:-1]
        DSTyx = -DSTxy

        # Dynamical delay
        tauX = EX[1:, 1:-1] - EX[:-1, 1:-1]
        tauY = EY[1:-1, 1:] - EY[1:-1, :-1]
        TAU = np.min((tauX[1:, :], tauX[:-1, :],
                      tauY[:, 1:], tauY[:, :-1]), axis=0) / 2

        # count number of sync events
        countXY = (np.sum((DSTxy > 0) * (DSTxy <=
                   np.minimum(TAU, self.taumax))) + 0.5*np.sum(DSTxy == 0))
        countYX = (np.sum((DSTyx > 0) * (DSTyx <=
                   np.minimum(TAU, self.taumax))) + 0.5*np.sum(DSTyx == 0))

        # normalization
        return countXY / np.sqrt(lx * ly), countYX / np.sqrt(lx * ly)
