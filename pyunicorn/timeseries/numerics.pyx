# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
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
FLOAT64TYPE = np.float64
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t
ctypedef np.float64_t FLOAT64TYPE_t

cdef extern from "stdlib.h":
    double drand48()

cdef extern from "stdlib.h":
    double srand48()

cdef extern from "time.h":
    double time()


cdef extern from "./_ext/src_fast_numerics.h":
  void _test_pearson_correlation_fast(double *original_data,
    double *surrogates, float *correlation, int n_time, int N, double norm)
  void _test_pearson_correlation_slow(double *original_data,
    double *surrogates, float *correlation, int n_time, int N, double norm)

# surrogates ==================================================================


def _embed_time_series_array(
    int N, int n_time, int dimension, int delay,
    np.ndarray[FLOATTYPE_t, ndim=2] time_series_array,
    np.ndarray[FLOAT64TYPE_t, ndim=3] embedding):
    """
    >>> 42 == 42
    True
    """

    cdef int i, j, k, max_delay, len_embedded, index

    # Calculate the maximum delay
    max_delay = (dimension - 1) * delay
    # Calculate the length of the embedded time series
    len_embedded = n_time - max_delay

    for i in xrange(N):
        for j in xrange(dimension):
            index = j*delay
            for k in xrange(len_embedded):
                embedding[i, k, j] = time_series_array[i, index]
                index += 1


def _recurrence_plot(
    int n_time, int dimension, float threshold,
    np.ndarray[FLOATTYPE_t, ndim=2] embedding,
    np.ndarray[INT8TYPE_t, ndim=2] R):

    cdef:
        int j, k, l
        double diff

    for j in xrange(n_time):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in xrange(j):
            for l in xrange(dimension):
                # Use supremum norm
                diff = embedding[j, l] - embedding[k, l]

                if abs(diff) > threshold:
                    # j and k are not neighbors
                    R[j, k] = R[k, j] = 0

                    # Leave the loop
                    break


def _twins(
    int N, int n_time, int dimension, float threshold, int min_dist,
    np.ndarray[FLOATTYPE_t, ndim=3] embedding_array,
    np.ndarray[FLOATTYPE_t, ndim=2] R, np.ndarray[FLOATTYPE_t, ndim=1] nR,
    twins):

    cdef:
        int i, j, k, l
        double diff

    for i in xrange(N):
        # Initialize the recurrence matrix R and nR

        for j in xrange(n_time):
            for k in xrange(j+1):
                R[j, k] = R[k, j] = 1
            nR[j] = n_time

        # Calculate the recurrence matrix for time series i

        for j in xrange(n_time):
            # Ignore main diagonal, since every sample is neighbor of itself
            for k in xrange(j):
                for l in xrange(dimension):
                    # Use maximum norm
                    diff = embedding_array[i, j, l] - embedding_array[i, k, l]

                    if abs(diff) > threshold:
                        # j and k are not neighbors
                        R[j, k] = R[k, j] = 0

                        # Reduce neighbor count of j and k by one
                        nR[j] -= 1
                        nR[k] -= 1

                        # Leave the for loop
                        break

        # Add list for twins in time series i
        twins.append([])

        # Find all twins in the recurrence matrix

        for j in xrange(n_time):
            twins_i = twins[i]
            twins_i.append([])
            twins_ij = twins_i[j]

            # Respect a minimal temporal spacing between twins to avoid false
            # twins due to the higher
            # sample density in phase space along the trajectory
            for k in xrange(j-min_dist):
                # Continue only if both samples have the same number of
                # neighbors and more than jsut one neighbor (themselves)
                if nR[j] == nR[k] and nR[j] != 1:
                    l = 0

                    while R[j, l] == R[k, l]:
                        l += 1

                        # If l is equal to the length of the time series at
                        # this point, j and k are twins
                        if l == n_time:
                            # Add the twins to the twin list
                            twins_ik = twins_i[k]

                            twins_ij.append(k)
                            twins_ik.append(j)

                            # Leave the while loop
                            break

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


def _vertline_dist_norqa_missingvalues(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] vertline,
    np.ndarray[INT8TYPE_t, ndim=2] recmat,
    np.ndarray[BOOLTYPE_t, ndim=1, cast=True] mv_indices):

    cdef:
        int i, j, k = 0
        BOOLTYPE_t missing_flag = False

    for i in xrange(n_time):
        if (k != 0 and not missing_flag):
            vertline[k] += 1
            k = 0

        missing_flag = False

        for j in xrange(n_time):
            # check if current point in RP belongs to a missing value
            if mv_indices[i] or mv_indices[j]:
                missing_flag = True
                k = 0
            elif recmat[i, j] == 0 and missing_flag:
                missing_flag = False

            if not missing_flag:
                if recmat[i, j] != 0:
                    k += 1
                elif k != 0:
                    vertline[k] += 1
                    k = 0

def _vertline_dist_norqa(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] vertline,
    np.ndarray[INT8TYPE_t, ndim=2] recmat):

    cdef int i, j, k = 0

    for i in xrange(n_time):
        if k != 0:
            vertline[k] += 1
            k = 0

        for j in xrange(n_time):
            if recmat[i, j] != 0:
                k += 1
            elif k != 0:
                vertline[k] += 1
                k = 0


def _vertline_dist_rqa_missingvalues(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] vertline,
    np.ndarray[BOOLTYPE_t, ndim=1, cast=True] mv_indices,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        float temp_diff, diff
        BOOLTYPE_t missing_flag = False

    for i in xrange(n_time):
        if k != 0 and not missing_flag:
            vertline[k] += 1
            k = 0

        missing_flag = False

        for j in xrange(n_time):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in xrange(dim):
                # Use supremum norm
                temp_diff = abs(embedding[i, l] - embedding[j, l])

                if temp_diff > diff:
                    diff = temp_diff

            # Check if current point in RP belongs to a missing values
            if mv_indices[i] or mv_indices[j]:
                missing_flag = True
                k = 0
            elif diff > eps and missing_flag:
                missing_flag = True

            if not missing_flag:
                # Check if recurrent point has been reached
                if diff < eps:
                    k += 1
                elif k != 0:
                    vertline[k] += 1
                    k = 0

def _vertline_dist_rqa(
    int n_time, np.ndarray[INT32TYPE_t, ndim=1] vertline,
    np.ndarray[FLOAT32TYPE_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        float temp_diff, diff

    for i in xrange(n_time):
        if k != 0:
            vertline[k] += 1
            k = 0

        for j in xrange(n_time):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in xrange(dim):
                # Use supremum norm
                temp_diff = abs(embedding[i, l] - embedding[j, l])

                if temp_diff > diff:
                    diff = temp_diff

            # Check if recurrent point has been reached
            if diff < eps:
                k += 1
            elif k != 0:
                vertline[k] += 1
                k = 0


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


def _test_pearson_correlation(
    np.ndarray[double, ndim=2, mode='c'] original_data not None,
    np.ndarray[double, ndim=2, mode='c'] surrogates not None, 
    int N, int n_time, BOOLTYPE_t fast):

    cdef double norm = 1.0 / float(n_time)

    #  Initialize Pearson correlation matrix
    cdef np.ndarray[float, ndim=2, mode='c'] correlation = np.zeros((N, N), 
            dtype="float32")
    
    if (fast==True):
        _test_pearson_correlation_fast(
            <double*> np.PyArray_DATA(original_data),
            <double*> np.PyArray_DATA(surrogates),
            <float*> np.PyArray_DATA(correlation),
            n_time, N, norm)
    else:
        _test_pearson_correlation_slow(
            <double*> np.PyArray_DATA(original_data),
            <double*> np.PyArray_DATA(surrogates),
            <float*> np.PyArray_DATA(correlation),
            n_time, N, norm)

    return correlation


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
