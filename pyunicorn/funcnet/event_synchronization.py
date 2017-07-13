#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Provides class for event synchronization measures.
"""
# warnings
import warnings

# array object and fast numerics
import numpy as np

from .. import cached_const


class EventSynchronization(object):

    """
    Contains methods to calculate event synchronization matrices from event
    series. The entries of these matrices represent some variant of the event
    synchronization between two of the variables.

    References: [Quiroga2002]_, [Boers2014]_.
    """

    #
    #  Definitions of internal methods
    #

    def __init__(self, eventmatrix, taumax):
        """
        Initialize an instance of EventSynchronization.

        Format of eventmatrix:
        An eventmatrix is a 2D numpy array with the first dimension covering
        the timesteps and the second dimensions covering the variables. Each
        variable at a specific timestep is either '1' if an event occured or
        '0' if it did not, e.g. for 3 variables with 10 timesteps the
        eventmatrix could look like

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
        :arg int taumax: Maximum dynamical delay
        """

        self.__T = eventmatrix.shape[0]
        self.__N = eventmatrix.shape[1]
        self.__eventmatrix = eventmatrix
        self.taumax = taumax

        # Dictionary for chached constants
        self.cache = {'base': {}}
        """(dict) cache of re-usable computation results"""

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
        return ('EventSynchronization: %i variables, %i timesteps, taumax: %i'
                % (self.__N, self.__T, self.taumax))

    #
    #  Definitions of event synchronization measures
    #

    @cached_const('base', 'directedES')
    def directedES(self):
        """
        Returns the NxN matrix of the directed event synchronization measure.
        The entry [i, j] denotes the directed event synchrnoization from
        variable j to variable i.
        """
        eventmatrix = self.__eventmatrix
        res = np.ones((self.__N, self.__N)) * np.inf

        for i in xrange(0, self.__N):
            for j in xrange(i+1, self.__N):
                res[i, j], res[j, i] = self._EventSync(eventmatrix[:, i],
                                                       eventmatrix[:, j])
        return res

    def symmetricES(self):
        """
        Returns the NxN matrix of the undirected or symmetrix event
        synchronization measure. It is obtained by the sum of both directed
        versions.
        """
        directed = self.directedES()
        return directed + directed.T

    def antisymmetricES(self):
        """
        Returns the NxN matrix of the antisymmetric synchronization measure.
        It is obtained by the difference of both directed versions.
        """
        directed = self.directedES()
        return directed - directed.T

    def ECA(self, EventSeriesX, EventSeriesY, delT, tau=0, ts1=None, ts2=None):
        """
        Event coincidence analysis:
        Returns the precursor and trigger coincidence rates of two event series
        X and Y.

        :type EventSeriesX: 1D Numpy array
        :arg EventSeriesX: Event series containing '0's and '1's
        :type EventSeriesY: 1D Numpy array
        :arg EventSeriesY: Event series containing '0's and '1's
        :arg delT: coincidence interval width
        :arg int tau: lag parameter
        :rtype: list
        :return: [Precursor coincidence rate XY, Trigger coincidence rate XY,
                  Precursor coincidence rate YX, Trigger coincidence rate YX]
        """

        # Count events that cannot be coincided due to tau and delT
        if not (tau == 0 and delT == 0):
            # Start of EventSeriesX
            n11 = np.count_nonzero(EventSeriesX[:tau+delT])
            # End of EventSeriesX
            n12 = np.count_nonzero(EventSeriesX[-(tau+delT):])
            # Start of EventSeriesY
            n21 = np.count_nonzero(EventSeriesY[:tau+delT])
            # End of EventSeriesY
            n22 = np.count_nonzero(EventSeriesY[-(tau+delT):])
        else:
            # Instantaneous coincidence
            n11, n12, n21, n22 = 0, 0, 0, 0
        # Get time indices
        if ts1 is None:
            e1 = np.where(EventSeriesX)[0]
        else:
            e1 = ts1[EventSeriesX]
        if ts2 is None:
            e2 = np.where(EventSeriesY)[0]
        else:
            e2 = ts2[EventSeriesY]
        del EventSeriesX, EventSeriesY, ts1, ts2
        # Number of events
        l1 = len(e1)
        l2 = len(e2)
        # Array of all interevent distances
        dst = (np.array([e1]*l2).T - np.array([e2]*l1))
        # Count coincidences with array slicing
        prec12 = np.count_nonzero(np.any(((dst - tau >= 0) *
                                          (dst - tau <= delT))[n11:, :],
                                         axis=1))
        trig12 = np.count_nonzero(np.any(((dst - tau >= 0) *
                                          (dst - tau <= delT))
                                          [:, :dst.shape[1]-n22],
                                          axis=0))
        prec21 = np.count_nonzero(np.any(((-dst - tau >= 0) *
                                          (-dst - tau <= delT))[:, n21:],
                                          axis=0))
        trig21 = np.count_nonzero(np.any(((-dst - tau >= 0) *
                                          (-dst - tau <= delT))
                                          [:dst.shape[0]-n12, :],
                                          axis=1))
        # Normalisation and output
        return (np.float32(prec12)/(l1-n11), np.float32(trig12)/(l2-n22),
                np.float32(prec21)/(l2-n21), np.float32(trig21)/(l1-n12))

    def _EventSync(self, EventSeriesX, EventSeriesY):
        """
        Calculates the directed event synchronization from two event series X
        and Y.

        :type EventSeriesX: 1D Numpy array
        :arg EventSeriesX: Event series containing '0's and '1's
        :type EventSeriesY: 1D Numpy array
        :arg EventSeriesY: Event series containing '0's and '1's
        :rtype: list
        :return: [Event synchronization XY, Event synchronization YX]

        """

        # Get time indices (type boolean or simple '0's and '1's)
        ex = np.array(np.where(EventSeriesX), dtype=np.int8)
        ey = np.array(np.where(EventSeriesY), dtype=np.int8)
        # Number of events
        lx = ex.shape[1]
        ly = ey.shape[1]
        if lx == 0 or ly == 0:              # Division by zero in output
            return np.nan, np.nan
        if lx in [1, 2] or ly in [1, 2]:    # Too few events to calculate
            return 0., 0.
        # Array of distances
        dstxy2 = 2 * (np.repeat(ex[:, 1:-1].T, ly-2, axis=1) -
                      np.repeat(ey[:, 1:-1], lx-2, axis=0))
        # Dynamical delay
        diffx = np.diff(ex)
        diffy = np.diff(ey)
        diffxmin = np.minimum(diffx[:, 1:], diffx[:, :-1])
        diffymin = np.minimum(diffy[:, 1:], diffy[:, :-1])
        tau2 = np.minimum(np.repeat(diffxmin.T, ly-2, axis=1),
                          np.repeat(diffymin, lx-2, axis=0))
        tau2 = np.minimum(tau2, 2 * self.taumax)
        # Count equal time events and synchronised events
        eqtime = 0.5 * (dstxy2.size - np.count_nonzero(dstxy2))
        countxy = np.sum((dstxy2 > 0) * (dstxy2 <= tau2)) + eqtime
        countyx = np.sum((dstxy2 < 0) * (dstxy2 >= -tau2)) + eqtime
        norm = np.sqrt((lx-2) * (ly-2))
        return countxy / norm, countyx / norm
