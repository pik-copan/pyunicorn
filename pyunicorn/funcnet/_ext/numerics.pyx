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
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
FLOAT64TYPE = np.float64
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t
ctypedef np.float64_t FLOAT64TYPE_t


cdef extern from "src_numerics.c":
    void _symmetrize_by_absmax_fast(float *similarity_matrix,
            signed char *lag_matrix, int N)
    void _cross_correlation_max_fast(float *array, float *similarity_matrix,
            signed char *lag_matrix, int N, int tau_max, int corr_range)
    void _cross_correlation_all_fast(float *array, float *lagfuncs, int N,
            int tau_max, int corr_range)
    void _get_nearest_neighbors_fast(float *array, int T, int dim_x, int dim_y,
            int k, int dim, int *k_xy, int *k_yz, int *k_z)


# coupling_analysis ===========================================================

def _symmetrize_by_absmax(
    np.ndarray[float, ndim=2, mode='c'] similarity_matrix not None,
    np.ndarray[INT8TYPE_t, ndim=2, mode='c'] lag_matrix not None, int N):

    _symmetrize_by_absmax_fast(
        <float*> np.PyArray_DATA(similarity_matrix),
        <signed char*> np.PyArray_DATA(lag_matrix), N)

    return similarity_matrix, lag_matrix


def _cross_correlation_max(
    np.ndarray[FLOAT32TYPE_t, ndim=3, mode='c'] array not None,
    int N, int tau_max, int corr_range):

    cdef np.ndarray[FLOAT32TYPE_t, ndim=2, mode='c'] similarity_matrix = \
        np.ones((N, N), dtype="float32")
    cdef np.ndarray[INT8TYPE_t, ndim=2, mode='c'] lag_matrix = \
        np.zeros((N, N), dtype='int8')

    cdef:
        double crossij, max
        int i,j,tau,k, argmax

    # loop over all node pairs, NOT symmetric due to time shifts!
    for i in range(N):
        for j in range(N):
            if i != j:
                max = 0.0
                argmax = 0
                # loop over taus INCLUDING the last tau value
                for tau in range(tau_max + 1):
                    crossij = 0
                    # here the actual cross correlation is calculated
                    # assuming standardized arrays
                    for k in range(corr_range):
                        crossij += array[tau,i,k] * array[tau_max,j,k]
                    # calculate max and argmax by comparing to
                    # previous value and storing max
                    if abs(crossij) > abs(max):
                        max = crossij
                        argmax = tau
                similarity_matrix[i,j] = max/(float)(corr_range)
                lag_matrix[i,j] = tau_max - argmax

    return similarity_matrix, lag_matrix

def _cross_correlation_all(
        np.ndarray[FLOAT32TYPE_t, ndim=3, mode='c'] array not None,
        int N, int tau_max, int corr_range):

    """
    lagfuncs = np.zeros((N, N, tau_max+1), dtype="float32")
    """
    cdef np.ndarray[FLOAT32TYPE_t, ndim=3, mode='c'] lagfuncs = \
        np.zeros((N, N, tau_max+1), dtype="float32")

    cdef:
        int i, j, tau, k
        double crossij
    # loop over all node pairs, NOT symmetric due to time shifts!
    for i in range(N):
        for j in range(N):
            # loop over taus INCLUDING the last tau value
            for tau in range(tau_max):
                crossij = 0
                # here the actual cross correlation is calculated
                # assuming standardized arrays
                for k in range(corr_range):
                    crossij += array[tau,i,k] * array[tau_max,j,k]

                lagfuncs[i,j,tau_max-tau] = crossij/(float)(corr_range)

    return lagfuncs


def _get_nearest_neighbors_cython(
        np.ndarray[float, ndim=1, mode='c'] array not None,
        int T, int dim_x, int dim_y, int k, int dim):

    # Initialize
    cdef np.ndarray[INT32TYPE_t, ndim=1, mode='c'] k_xz = \
            np.zeros((T), dtype='int32')
    cdef np.ndarray[INT32TYPE_t, ndim=1, mode='c'] k_yz = \
            np.zeros((T), dtype='int32')
    cdef np.ndarray[INT32TYPE_t, ndim=1, mode='c'] k_z = \
            np.zeros((T), dtype='int32')

    _get_nearest_neighbors_fast(
        <float*> np.PyArray_DATA(array), T, dim_x, dim_y, k, dim,
        <int*> np.PyArray_DATA(k_xz), <int*> np.PyArray_DATA(k_yz),
        <int*> np.PyArray_DATA(k_z))

    return k_xz, k_yz, k_z
