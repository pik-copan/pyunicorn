#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

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
    :float return: Event synchronization es2 to es1
    """
    ex = np.arange(len(es1))[es1 == 1]
    ey = np.arange(len(es2))[es2 == 1]
    lx = len(ex)
    ly = len(ey)

    count = 0
    for m in range(1, lx-1):
        for n in range(1, ly-1):
            dst = ex[m] - ey[n]

            if abs(dst) > taumax:
                continue
            elif dst == 0:
                count += 0.5
                continue

            # finding the dynamical delay tau
            tmp = ex[m+1] - ex[m]
            if tmp > ex[m] - ex[m-1]:
                tmp = ex[m] - ex[m-1]
            tau = ey[n+1] - ey[n]
            if tau > ey[n] - ey[n-1]:
                tau = ey[n] - ey[n-1]
            if tau > tmp:
                tau = tmp
            tau = tau / 2

            if dst > 0 and dst <= tau:
                count += 1
#                print "dst, tau: %i, %i"%(dst, tau)

    return count / np.sqrt(lx * ly)  # np.sqrt((lx-2)/2. * (ly-2)/2.)


def UndirectedESyncMat(eventmatrix, taumax):
    N = eventmatrix.shape[1]
    res = np.ones((N, N)) * np.inf

    for i in np.arange(0, N):
        for j in np.arange(0, N):
            if i == j:
                continue
            res[i, j] = NonVecEventSync(eventmatrix[:, i], eventmatrix[:, j],
                                        taumax)
    return res


def testVectorization():
    """
    Test if the vectorized implementation coincides with the straight forward
    one.
    """
    for taumax in [1, 5, 16]:
        length, N, eventprop = 100, 50, 0.2
        eventmatrix = 1-(np.random.rand(length, N) > eventprop).astype(int)
        ES = EventSynchronization(eventmatrix, taumax)
        assert np.allclose(UndirectedESyncMat(eventmatrix, taumax),
                           ES.directedES())
