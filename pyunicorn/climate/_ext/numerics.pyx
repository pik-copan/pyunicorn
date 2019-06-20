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

cimport cython
from cpython cimport bool


import numpy as np
cimport numpy as np


BOOLTYPE = np.uint8
INTTYPE = np.int
INT8TYPE = np.int8
INT16TYPE = np.int16
INT32TYPE = np.int32
INT64TYPE = np.int64
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
FLOAT64TYPE = np.float64
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.int64_t INT64TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t
ctypedef np.float64_t FLOAT64TYPE_t


cdef extern from "src_numerics.c":
    void _cython_calculate_mutual_information(
            float *anomaly, int n_samples, int N, int n_bins, double scaling,
            double range_min, long *symbolic, long *hist, long *hist2d,
            float *mi)
    void _calculate_corr_fast(int m, int tmax, int *final_mask,
            float *time_series_ranked, float *spearman_rho)


# mutual_info =================================================================

def _calculate_mutual_information_cython(
    np.ndarray[float, ndim=2, mode='c'] anomaly not None,
    int n_samples, int N, int n_bins, double scaling, double range_min):

    cdef:
        np.ndarray[long, ndim=2, mode='c'] symbolic = \
            np.zeros((N, n_samples), dtype='int64')
        np.ndarray[long, ndim=2, mode='c'] hist = \
            np.zeros((N, n_bins), dtype='int64')
        np.ndarray[long, ndim=2, mode='c'] hist2d = \
            np.zeros((n_bins, n_bins), dtype='int64')
        np.ndarray[float, ndim=2, mode='c'] mi = \
            np.zeros((N, N), dtype='float32')

    _cython_calculate_mutual_information(
        <float*> np.PyArray_DATA(anomaly), n_samples, N, n_bins, scaling,
        range_min, <long*> np.PyArray_DATA(symbolic),
        <long*> np.PyArray_DATA(hist), <long*> np.PyArray_DATA(hist2d),
        <float*> np.PyArray_DATA(mi))

    return mi


# rainfall ====================================================================

def _calculate_corr(int m, int tmax,
    np.ndarray[int, ndim=2, mode='c'] final_mask not None,
    np.ndarray[float, ndim=2, mode='c'] time_series_ranked not None):

    cdef np.ndarray[float, ndim=2, mode='c'] spearman_rho = \
            np.zeros((m, m), dtype='float32')

    _calculate_corr_fast(m, tmax,
            <int*> np.PyArray_DATA(final_mask),
            <float*> np.PyArray_DATA(time_series_ranked),
            <float*> np.PyArray_DATA(spearman_rho))

    return spearman_rho
