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
Simple tests for the funcnet EventSynchronization class.
"""
import numpy as np
from pyunicorn.funcnet import EventSynchronization


def NonVecEventSync(es1, es2, taumax):
    """
    Compute non-vectorized event synchronization

    :type es1: 1D Numpy array
    :arg es1: Event series containing '0's and '1's
    :type es2: 1D Numpy array
    :arg es2: Event series containing '0's and '1's
    :rtype: list
    :return: [Event synchronization XY, Event synchronization YX]
    """
    ex = np.arange(len(es1))[es1 == 1]
    ey = np.arange(len(es2))[es2 == 1]
    lx = len(ex)
    ly = len(ey)

    Axy = np.zeros((lx, ly), dtype=bool)
    Ayx = np.zeros((lx, ly), dtype=bool)
    eqtime = np.zeros((lx, ly), dtype=bool)
    for m in range(1, lx-1):
        for n in range(1, ly-1):
            dstxy = ex[m] - ey[n]

            if abs(dstxy) > taumax:
                continue
            # finding the dynamical delay tau
            tau = min([ex[m+1] - ex[m], ex[m] - ex[m-1],
                       ey[n+1] - ey[n], ey[n] - ey[n-1]]) / 2

            if dstxy > 0:
                if dstxy <= tau:
                    Axy[m, n] = True
            if dstxy < 0:
                if dstxy >= -tau:
                    Ayx[m, n] = True
            elif dstxy == 0:
                eqtime[m, n] = True

    # Loop over coincidences and determine number of double counts
    # by checking at least one event of the pair is also coincided
    #in other direction
    countxydouble = countyxdouble = 0

    for i, j in np.transpose(np.where(Axy)):
        countxydouble += np.any(Ayx[i, :]) or np.any(Ayx[:, j])
    for i, j in np.transpose(np.where(Ayx)):
        countyxdouble += np.any(Axy[i, :]) or np.any(Axy[:, j])

    countxy = np.sum(Axy) + 0.5*np.sum(eqtime) - 0.5*countxydouble
    countyx = np.sum(Ayx) + 0.5*np.sum(eqtime) - 0.5*countyxdouble
    norm = np.sqrt((lx-2) * (ly-2))
    return countxy / norm, countyx / norm


def UndirectedESyncMat(eventmatrix, taumax):
    N = eventmatrix.shape[1]
    res = np.ones((N, N)) * np.inf

    for i in np.arange(0, N):
        for j in np.arange(i+1, N):
            res[i, j], res[j, i] = NonVecEventSync(eventmatrix[:, i],
                                                   eventmatrix[:, j],
                                                   taumax)
    return res


def testVectorization():
    """
    Test if the vectorized implementation coincides with the straight forward
    one.
    """
    for taumax in [1, 5, 16]:
        length, N, eventprop = 100, 50, 0.2
        # equal event counts (normalization requirement)
        eventcount = int(length * eventprop)
        eventmatrix = np.zeros((length, N), dtype=int)
        for v in range(N):
            fills = np.random.choice(np.arange(length), eventcount,
                                     replace=False)
            eventmatrix[fills, v] = 1
        ES = EventSynchronization(eventmatrix, taumax)
        assert np.allclose(UndirectedESyncMat(eventmatrix, taumax),
                           ES.directedES())
