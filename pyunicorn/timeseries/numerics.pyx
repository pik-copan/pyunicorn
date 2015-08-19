# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)


cimport cython
cimport numpy as np

import numpy as np
import numpy.random as rd

randint = rd.randint

INTTYPE = np.int
INT32TYPE = np.int32
INT8TYPE = np.int8
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
ctypedef np.int_t INTTYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t


# recurrence plot==============================================================

def _embed_time_series(
    int n_time, int dim, int tau,
    np.ndarray[FLOAT32TYPE_t, ndim=1] time_series,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding):
    """
    >>> 42 == 42
    True
    """

    cdef:
        int i,j, max_delay, len_embedded, index

    # Calculate the maximum delay
    max_delay = (dim - 1) * tau
    # Calculate the length of the embedded time series
    len_embedded = n_time - max_delay

    for j in xrange(dim):
        index = j * tau
        for k in xrange(len_embedded):
            embedding[k, j] = time_series[index]
            index += 1


def _manhatten_distance_matrix(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=2] distance):

    cdef:
        int j, k, l
        float sum

    # Calculate the manhatten distance matrix
    for j in xrange(n_time):
        # Ignore the main diagonal, since every samle is neighbor of itself
        for k in xrange(j):
            sum = 0
            for l in xrange(dim):
                # use manhattan norm
                sum += abs(embedding[j, l] - embedding[k, l])

            distance[j, k] = distance[k, j] = sum

def _euclidean_distance_matrix(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=2] distance):

    cdef:
        int j, k, l
        float sum, diff

    # Calculate the eucliadean distance matrix
    for j in xrange(n_time):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in xrange(j):
            sum = 0
            for l in xrange(dim):
                # Use euclidean norm
                diff = abs(embedding[j, l] - embedding[k, l])
                sum += diff * diff
            distance[j, k] = distance[k, j] = sum

def _supremum_distance_matrix(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=2] distance):

    cdef:
        int j, k, l
        float temp_diff, diff


    # Calculate the eucliadean distance matrix
    for j in xrange(n_time):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in xrange(j):
            temp_diff = diff = 0
            for l in xrange(dim):
                # Use supremum norm
                temp_diff = abs(embedding[j, l] - embedding[k, l])
                if temp_diff > diff:
                    diff = temp_diff

            distance[j, k] = distance[k, j] = diff


def _set_adaptive_neighborhood_size(
    int n_time, int adaptive_neighborhood_size,
    np.ndarray[INT32TYPE_t, ndim=2] sorted_neighbors,
    np.ndarray[INTTYPE_t, ndim=1] order,
    np.ndarray[INT8TYPE_t, ndim=2] recurrence):

    cdef:
        int i, j, k, l

    for i in xrange(adaptive_neighborhood_size):
        for j in xrange(n_time):
            # Get the node index to be processed
            l = order[j]

            # Start with k = i + 1, since state vectors closer than the (i+1)th
            # nearest neighbor are already connected to j at this stage
            k = i + 1
            while recurrence[l, sorted_neighbors[l, k]] == 1 and k < n_time:
                k += 1
            # add a "new" nearest neighbor of l to the recurrence plot
            recurrence[l, sorted_neighbors[l, k]] = \
                recurrence[sorted_neighbors[l, k], l] = 1
