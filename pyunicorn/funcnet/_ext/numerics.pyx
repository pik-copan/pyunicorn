# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

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
            float *lag_matrix, int N)


# coupling_analysis ===========================================================

def _symmetrize_by_absmax(
    np.ndarray[float, ndim=2, mode='c'] similarity_matrix not None,
    np.ndarray[int, ndim=2, mode='c'] lag_matrix not None, int N):

    _symmetrize_by_absmax_fast(
        <float*> np.PyArray_DATA(similarity_matrix),
        <int*> np.PyArray_DATA(lag_matrix), N)

    return similarity_matrix, lag_matrix
