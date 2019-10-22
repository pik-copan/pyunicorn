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
Simple tests for the CrossRecurrencePlot class.
"""
import numpy as np

from pyunicorn.timeseries import CrossRecurrencePlot

def test_cross_recurrence_plot():
    ts1 = np.array([0, 0, 1])
    ts2 = np.array([1, 0, 1])
    CrossRecurrencePlot(x=ts1, y=ts2, threshold=0.2)
    assert True
    
def test_manhattan_distance_matrix():
    """Checks consistency of two subsequent calculations of supremum distance 
    matrix"""
    ts1 = np.array([0, 0, 1])
    ts2 = np.array([1, 0, 1])
    crp = CrossRecurrencePlot(x=ts1, y=ts2, threshold=0.2, metric="manhattan")
    dist_mat_1 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="manhattan")
    dist_mat_2 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="manhattan")
    assert np.allclose(dist_mat_1, dist_mat_2, atol=1e-04)
    
def test_euclidean_distance_matrix():
    """Checks consistency of two subsequent calculations of euclidean distance 
    matrix"""
    ts1 = np.array([0, 0, 1])
    ts2 = np.array([1, 0, 1])
    crp = CrossRecurrencePlot(x=ts1, y=ts2, threshold=0.2, metric="euclidean")
    dist_mat_1 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="euclidean")
    dist_mat_2 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="euclidean")
    assert np.allclose(dist_mat_1, dist_mat_2, atol=1e-04)

def test_supremum_distance_matrix():
    """Checks consistency of two subsequent calculations of supremum distance 
    matrix"""
    ts1 = np.array([0, 0, 1])
    ts2 = np.array([1, 0, 1])
    crp = CrossRecurrencePlot(x=ts1, y=ts2, threshold=0.2, metric="supremum")
    dist_mat_1 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="supremum")
    dist_mat_2 = crp.distance_matrix(crp.x_embedded, crp.y_embedded,
                                     metric="supremum")
    assert np.allclose(dist_mat_1, dist_mat_2, atol=1e-04)