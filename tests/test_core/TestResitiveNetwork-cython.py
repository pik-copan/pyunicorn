#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
"""
Weave tests to check that python and weave implementations give the same
results
"""

from pyunicorn import ResNetwork
import numpy as np


res = ResNetwork.SmallTestNetwork()
resC = ResNetwork.SmallComplexNetwork()


def testVCFB():
    for i in range(5):
        admittance = res.get_admittance()
        R = res.get_R()
        # set params
        Is = It = np.float(1.0)
        # alloc output
        vcfbPython = np.float(0)
        for t in xrange(res.N):
            for s in xrange(t):
                I = 0.0
                if i == t or i == s:
                    pass
                else:
                    for j in xrange(res.N):
                        I += admittance[i][j] * np.abs(
                            Is*(R[i][s]-R[j][s]) + It*(R[j][t]-R[i][t]))/2.
                vcfbPython += 2.*I/(res.N*(res.N-1))

        vcfbCython = res.vertex_current_flow_betweenness(i)
        assert round(vcfbPython, 4) == round(vcfbCython, 4)


def testECFB():
    # set currents
    Is = It = np.float(1)
    # alloc output
    if res.flagComplex:
        dtype = complex
    else:
        dtype = float
    ecfbPython = np.zeros([res.N, res.N], dtype=dtype)
    # the usual
    admittance = res.get_admittance()
    R = res.get_R()
    for i in xrange(res.N):
        for j in xrange(res.N):
            I = 0
            for t in xrange(res.N):
                for s in xrange(t):
                    I += admittance[i][j] * np.abs(
                        Is*(R[i][s]-R[j][s])+It*(R[j][t]-R[i][t]))
            # Lets try to compute the in
            ecfbPython[i][j] = 2*I/(res.N*(res.N-1))

    ecfbCython = res.edge_current_flow_betweenness()
    l = len(ecfbPython)
    for i in range(l):
        for j in range(l):
            assert round(ecfbPython[i][j], 4) == round(ecfbCython[i][j], 4)
