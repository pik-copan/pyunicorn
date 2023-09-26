# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
"""
Weave tests to check that python and weave implementations give the same
results
"""

import numpy as np
from pyunicorn import ResNetwork


res = ResNetwork.SmallTestNetwork()
resC = ResNetwork.SmallComplexNetwork()


def testVCFB():
    for i in range(5):
        admittance = res.get_admittance()
        R = res.get_R()
        # set params
        Is = It = np.float32(1.0)
        # alloc output
        vcfbPython = np.float32(0)
        for t in range(res.N):
            for s in range(t):
                K = 0.0
                if i in (t, s):
                    pass
                else:
                    for j in range(res.N):
                        K += admittance[i][j] * np.abs(
                            Is*(R[i][s]-R[j][s]) + It*(R[j][t]-R[i][t]))/2.
                vcfbPython += 2.*K/(res.N*(res.N-1))

        vcfbCython = res.vertex_current_flow_betweenness(i)
        assert round(vcfbPython, 4) == round(vcfbCython, 4)


def testECFB():
    # set currents
    Is = It = np.float32(1)
    # alloc output
    if res.flagComplex:
        dtype = complex
    else:
        dtype = float
    ecfbPython = np.zeros([res.N, res.N], dtype=dtype)
    # the usual
    admittance = res.get_admittance()
    R = res.get_R()
    for i in range(res.N):
        for j in range(res.N):
            K = 0
            for t in range(res.N):
                for s in range(t):
                    K += admittance[i][j] * np.abs(
                        Is*(R[i][s]-R[j][s])+It*(R[j][t]-R[i][t]))
            # Lets try to compute the in
            ecfbPython[i][j] = 2*K/(res.N*(res.N-1))

    ecfbCython = res.edge_current_flow_betweenness()
    L = len(ecfbPython)
    for i in range(L):
        for j in range(L):
            assert round((ecfbPython[i][j].astype('float32')), 4) == \
                   round((ecfbCython[i][j].astype('float32')), 4)
