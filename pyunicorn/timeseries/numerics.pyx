# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)


cimport cython
from cpython cimport bool
from libc.math cimport sqrt, floor
from libc.stdlib cimport rand, RAND_MAX


import numpy as np
cimport numpy as np
import numpy.random as rd
import random

randint = rd.randint



BOOLTYPE = np.uint8
INTTYPE = np.int
INT8TYPE = np.int8
INT16TYPE = np.int16
INT32TYPE = np.int32
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t

cdef extern from "stdlib.h":
    double drand48()

cdef extern from "stdlib.h":
    double srand48()

cdef extern from "time.h":
    double time()


# visibitly graph =============================================================

def _visibility_relations_missingvalues(
    np.ndarray[FLOAT32TYPE_t, ndim=1] x, np.ndarray[FLOAT32TYPE_t, ndim=1] t,
    int N, np.ndarray[INT8TYPE_t, ndim=2] A,
    np.ndarray[BOOLTYPE_t, ndim=1, cast=True] mv_indices):
    """
    >>> 42 == 42
    True
    """

    cdef:
        int i, j, k
        float test

    for i in xrange(N-2):
        for j in xrange(i+2, N):
            k = i + 1

            test = (x[j] - x[i]) / (t[j] - t[i])

            while not mv_indices[k] and\
                (x[k] - x[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in xrange(N-1):
        if not mv_indices[i] and not mv_indices[i+1]:
            A[i, i+1] = A[i+1, i] = 1


def _visibility_relations_no_missingvalues(
    np.ndarray[FLOAT32TYPE_t, ndim=1] x, np.ndarray[FLOAT32TYPE_t, ndim=1] t,
    int N, np.ndarray[INT8TYPE_t, ndim=2] A):

    cdef:
        int i, j, k
        float test

    for i in xrange(N-2):
        for j in xrange(i+2, N):
            k = i + 1

            test = (x[j] - x[i]) / (t[j] - t[i])

            while (x[k] - x[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in xrange(N-1):
        A[i, i+1] = A[i+1, i] = 1


def _visibility_relations_horizontal(
    np.ndarray[FLOAT32TYPE_t, ndim=1] x, np.ndarray[FLOAT32TYPE_t, ndim=1] t,
    int N, np.ndarray[INT8TYPE_t, ndim=2] A):

    cdef:
        int i, j, k
        float minimum

    for i in xrange(N-2):
        for j in xrange(i+2, N):
            k = i + 1
            minimum = min(x[i], x[j])

            while x[k] < minimum and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in xrange(N-1):
        A[i, i+1] = A[i+1, i] = 1

def _retarded_local_clustering(
    int N, np.ndarray[INT16TYPE_t, ndim=2] A,
    np.ndarray[FLOATTYPE_t, ndim=1] norm,
    np.ndarray[FLOATTYPE_t, ndim=1] retarded_clustering):

    cdef:
        int i, j, k
        long counter

    # Loop over all nodes
    for i in xrange(N):
        # Check if i has right degree larger than 1
        if norm[i] != 0:
            # Reset counter
            counter = 0

            # Loop over unique pairs of nodes in the past of i
            for j in xrange(i):
                for k in xrange(j):
                    if A[i, j] == 1 and A[j, k] == 1 and A[k, i] == 1:
                        counter += 1

            retarded_clustering[i] = counter / norm[i]

def _advanced_local_clustering(
    int N, np.ndarray[INT16TYPE_t, ndim=2] A,
    np.ndarray[FLOATTYPE_t, ndim=1] norm,
    np.ndarray[FLOATTYPE_t, ndim=1] advanced_clustering):

    cdef:
        int i, j, k
        long counter

    # Loop over all nodes
    for i in xrange(N-2):
        # Check if i has right degree larger than 1
        if norm[i] != 0:
            # Reset counter
            counter = 0

            # Loop over unique pairs of nodes in the future of i
            for j in xrange(i+1, N):
                for k in xrange(i+1, j):
                    if A[i, j] == 1 and A[j, k] == 1 and A[k, i] == 1:
                        counter += 1

            advanced_clustering[i] = counter / norm[i]
