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


res = ResNetwork.SmallTestNetwork()
resC = ResNetwork.SmallComplexNetwork()


def testVCFB():
    for i in range(5):
        res.flagWeave = False
        vcfbPython = res.vertex_current_flow_betweenness(i)
        res.flagWeave = True
        vcfbWeave = res.vertex_current_flow_betweenness(i)
        assert vcfbPython == vcfbWeave


def testECFB():
    res.flagWeave = False
    ecfbPython = res.edge_current_flow_betweenness()
    res.flagWeave = True
    ecfbWeave = res.edge_current_flow_betweenness()
    l = len(ecfbPython)
    for i in range(l):
        for j in range(l):
            assert ecfbPython[i][j] == ecfbWeave[i][j]
