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
INT32TYPE = np.int32
INT8TYPE = np.int8
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t

cdef extern from "stdlib.h":
    double drand48()

cdef extern from "stdlib.h":
    double srand48()

cdef extern from "time.h":
    double time()


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


def _bootstrap_distance_matrix_manhatten(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[INTTYPE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float sum, diff

    for i in xrange(M):
        #Compute their distance
        sum = 0
        for l in xrange(dim):
            # Use manhatten norm
            sum += abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])

        distances[i] = sum


def _bootstrap_distance_matrix_euclidean(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[INTTYPE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float sum, diff

    for i in xrange(M):
        #Compute their distance
        sum = 0
        for l in xrange(dim):
            # Use manhatten norm
            diff = abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])
            sum += diff * diff

        distances[i] = sqrt(sum)


def _bootstrap_distance_matrix_supremum(
    int n_time, int dim, np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOAT32TYPE_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[INTTYPE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float temp_diff, diff

    for i in xrange(M):
        #Compute their distance
        temp_diff = diff = 0
        for l in xrange(dim):
            # Use supremum norm
            temp_diff = abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])

            if temp_diff > diff:
                diff = temp_diff

            distances[i] = diff


def _diagline_dist_norqa_missingvalues(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] diagline,
    np.ndarray[INT8TYPE_t, ndim=2] recmat,
    np.ndarray[BOOLTYPE_t, ndim=1, cast=True] mv_indices):

    cdef:
        int i, j, k = 0
        BOOLTYPE_t missing_flag = False

    for i in xrange(n_time):
        if k != 0 and not missing_flag:
            diagline[k] += 1
            k = 0

        missing_flag = False

        for j in xrange(i+1):
            # Check if curren tpoint in RP belongs to a mising value
            if mv_indices[n_time-1-i+j] or mv_indices[j]:
                missing_flag = True
                k = 0
            elif recmat[n_time-1-i+j, j] == 0 and missing_flag:
                missing_flag = False

            if not missing_flag:
                # Only incease k if some previous point in diagonal was not a
                # missing value!
                if recmat[n_time-1-i+j, j] == 1:
                    k += 1
                # Only count diagonal lines that are not followed by a missing
                # point in the recurrence plot
                elif k != 0:
                    diagline[k] += 1
                    k = 0


def _diagline_dist_norqa(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] diagline,
    np.ndarray[INT8TYPE_t, ndim=2] recmat):

    cdef:
        int i, j, k = 0

    for i in xrange(n_time):
        if k != 0:
            diagline[k] += 1
            k = 0
        for j in xrange(i+1):
            if recmat[n_time-1-i+j, j] == 1:
                k += 1
            elif k != 0:
                diagline[k] += 1
                k = 0


def _diagline_dist_rqa_missingvalues(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] diagline,
    np.ndarray[BOOLTYPE_t, ndim=1, cast=True] mv_indices,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        float temp_diff, diff
        BOOLTYPE_t missing_flag = False

    for i in xrange(n_time):
        if k != 0 and not missing_flag:
            diagline[k] += 1
            k = 0

        missing_flag = False

        for j in xrange(i+1):
            # Compute supreumum distance between state vectors
            temp_diff = diff = 0
            for l in xrange(dim):
                # Use supremum norm
                temp_diff = abs(embedding[j, l] - embedding[n_time-1-i+j, l])
                if temp_diff > diff:
                    diff = temp_diff

            # Check if curren tpoint in RP belongs to a missing value
            if mv_indices[n_time-1-i+j] or mv_indices[j]:
                missing_flag = True
                k = 0
            elif diff > eps and missing_flag:
                missing_flag = False

            if not missing_flag:
                # Only increase k if some previous point in diagonal was not a
                # missig value!
                if diff < eps:
                    k += 1

                # Only count diagonal lines that are not followed by a missing
                # value point in the recurrenc plot
                elif k != 0:
                    diagline[k] += 1
                    k = 0


def _diagline_dist_rqa(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] diagline,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        float temp_diff, diff

    for i in xrange(n_time):
        if k != 0:
            diagline[k] += 1
            k = 0

        for j in xrange(i+1):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in xrange(dim):
                # Use supremum norm
                temp_diff = abs(embedding[j, l] - embedding[n_time-1-i+j, l])
                if temp_diff > diff:
                    diff = temp_diff

            # check if R(j, n_time-q-i+j) == 1 -> recurrence
            if diff < eps:
                k += 1
            elif k != 0:
                diagline[k] += 1
                k = 0

def _rejection_sampling(
    np.ndarray[FLOATTYPE_t, ndim=1] dist,
    np.ndarray[FLOATTYPE_t, ndim=1] resampled_dist, int N, int M):

    cdef:
        int i = 0, x

    while i < M:
        x = int(floor(drand48() * N))

        if (drand48() < dist[x]):
            resampled_dist[x] += 1
            i += 1


def _white_vertline_dist(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] white_vertline,
    np.ndarray[INT8TYPE_t, ndim=2] R):

    cdef int i, j, k = 0

    for i in xrange(n_time):
        if k != 0:
            white_vertline[k] += 1
            k = 0

        for j in xrange(n_time):
            if R[i, j] == 0:
                k += 1
            elif k != 0:
                white_vertline[k] += 1
                k = 0

def _twins(
    int min_dist, int N, np.ndarray[INT8TYPE_t, ndim=2] R,
    np.ndarray[INTTYPE_t, ndim=1] nR, twins):

    cdef int j, k, l

    twins.append([])

    for j in xrange(N):
        twins.append([])
        twins_j = twins[j]

        # Respect a minimal temporal spacing between twins to avoid false
        # twins du to th higher sample density in phase space along the
        # trajectory
        for k in xrange(j - min_dist):
            # Continue only if both samples have the same number of
            # neighbors and more than just one neighbor (themselves)
            if nR[j] == nR[k] and nR[j] != 1:
                l = 0

                while R[j, l] == R[k, l]:
                    l = l + 1

                    # If l is equal to the length of the time series at
                    # this point, j and k are twins
                    if l == N:
                        # And the twins to the twin list
                        twins_k = twins[k]

                        twins_j.append(k)
                        twins_k.append(j)

                        break


def _twin_surrogates(
    int n_surrogates, int N, int dim, twins,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding,
    np.ndarray[FLOATTYPE_t, ndim=3] surrogates):

    cdef int i, j, k, l, new_k, n_twins, rand

    # Initialize random number generator
    # srand48(time(0)) -> does not work in cython somehow ?!?!?

    for i in xrange(n_surrogates):
        # Randomly choose a starting point in the original trajectory
        k = int(floor(drand48() * N))

        j = 0

        while j < N:
            # Assign state vector of surrogate trajectory
            for l in xrange(dim):
                surrogates[i, j, l] = embedding[k, l]

            # Get the list of twins of state vector k in the original time
            # series
            twins_k = twins[k]

            # Get the number of twins of k
            n_twins = len(twins_k)

            # If k has no twins, go to the next sample k+1, If k has twins at
            # m, choose among m+1 and k+1 with equal probability
            if n_twins == 0:
                k += 1
            else:
                # Generate a random integer between 0 and n_twins
                rand = int(floor(drand48() * (n_twins + 1)))

                # If rand = n_twings go to smple k+1, otherwise jump to the
                # future of one of the twins
                if rand == n_twins:
                    k += 1
                else:
                    k = twins_k[rand]
                    k += 1

            # If the new k >= n_time, choose a new random starting point in the
            # original time series
            if k >= N:
                while True:
                    new_k = int(floor(drand48() * N))
                    if new_k != k:
                        break

                k = new_k

            j += 1
