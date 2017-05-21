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
        vcfbPython = res._vertex_current_flow_betweenness_python(i)
        vcfbCython = res.vertex_current_flow_betweenness(i)
        assert round(vcfbPython, 4) == round(vcfbCython, 4)


def testECFB():
    ecfbPython = res._edge_current_flow_betweenness_python()
    ecfbCython = res.edge_current_flow_betweenness()
    l = len(ecfbPython)
    for i in range(l):
        for j in range(l):
            assert round(ecfbPython[i][j], 4) == round(ecfbCython[i][j], 4)
