# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2022 Jonathan F. Donges and pyunicorn authors
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

from libc.math cimport sqrt, floor
from libc.stdlib cimport rand, RAND_MAX

from datetime import datetime
import random

cimport numpy as np
import numpy as np
import numpy.random as rd
randint = rd.randint

from ...core._ext.types import NODE, FIELD, DFIELD
from ...core._ext.types cimport \
    ADJ_t, MASK_t, NODE_t, DEGREE_t, LAG_t, FIELD_t, DFIELD_t

cdef extern from "src_numerics.c":
    void _manhattan_distance_matrix_fast(int ntime_x, int ntime_y, int dim,
        double *x_embedded, double *y_embedded, float *distance)
    void _euclidean_distance_matrix_fast(int ntime_x, int ntime_y, int dim,
        double *x_embedded, double *y_embedded, float *distance)
    void _supremum_distance_matrix_fast(int ntime_x, int ntime_y, int dim,
        float *x_embedded, float *y_embedded, float *distance)
    void _test_pearson_correlation_fast(double *original_data,
        double *surrogates, float *correlation, int n_time, int N, double norm)
    void _test_mutual_information_fast(int N, int n_time, int n_bins,
        double scaling, double range_min, double *original_data,
        double *surrogates, int *symbolic_original, int *symbolic_surrogates,
        int *hist_original, int *hist_surrogates, int * hist2d, float *mi)


# cross_recurrence_plot =======================================================


def _manhattan_distance_matrix_crp(
    int ntime_x, int ntime_y, int dim,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] x_embedded not None,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] y_embedded not None):

    cdef np.ndarray[DFIELD_t, ndim=2, mode='c'] distance = \
        np.zeros((ntime_x, ntime_y), dtype=DFIELD)

    _manhattan_distance_matrix_fast(
        ntime_x, ntime_y, dim,
        <double*> np.PyArray_DATA(x_embedded),
        <double*> np.PyArray_DATA(y_embedded),
        <float*> np.PyArray_DATA(distance))

    return distance


def _euclidean_distance_matrix_crp(
    int ntime_x, int ntime_y, int dim,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] x_embedded not None,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] y_embedded not None):

    cdef np.ndarray[DFIELD_t, ndim=2, mode='c'] distance = \
        np.zeros((ntime_x, ntime_y), dtype=DFIELD)

    _euclidean_distance_matrix_fast(
        ntime_x, ntime_y, dim,
        <double*> np.PyArray_DATA(x_embedded),
        <double*> np.PyArray_DATA(y_embedded),
        <float*> np.PyArray_DATA(distance))

    return distance


def _supremum_distance_matrix_crp(
    int ntime_x, int ntime_y, int dim,
    np.ndarray[FIELD_t, ndim=2, mode='c'] x_embedded not None,
    np.ndarray[FIELD_t, ndim=2, mode='c'] y_embedded not None):

    cdef np.ndarray[FIELD_t, ndim=2, mode='c'] distance = \
        np.zeros((ntime_x, ntime_y), dtype=FIELD)

    _supremum_distance_matrix_fast(
        ntime_x, ntime_y, dim,
        <float*> np.PyArray_DATA(x_embedded),
        <float*> np.PyArray_DATA(y_embedded),
        <float*> np.PyArray_DATA(distance))

    return distance


# surrogates ==================================================================


def _embed_time_series_array(
    int n, int n_time, int dimension, int delay,
    np.ndarray[DFIELD_t, ndim=2] time_series_array,
    np.ndarray[DFIELD_t, ndim=3] embedding):

    cdef:
        unsigned int i, j, k, max_delay, len_embedded, index
        unsigned int N = n, D = dimension

    # Calculate the maximum delay
    max_delay = (dimension - 1) * delay
    # Calculate the length of the embedded time series
    len_embedded = n_time - max_delay

    for i in range(N):
        for j in range(D):
            index = j*delay
            for k in range(len_embedded):
                embedding[i, k, j] = time_series_array[i, index]
                index += 1


def _recurrence_plot(
    int n_time, int dimension, float threshold,
    np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[ADJ_t, ndim=2] R):

    cdef:
        unsigned int j, k, l, T = n_time, D = dimension
        FIELD_t diff

    for j in range(T):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in range(j):
            for l in range(D):
                # Use supremum norm
                diff = embedding[j, l] - embedding[k, l]

                if abs(diff) > threshold:
                    # j and k are not neighbors
                    R[j, k] = R[k, j] = 0

                    # Leave the loop
                    break


def _twins_s(
    int N, int n_time, int dimension, float threshold, int min_dist,
    np.ndarray[DFIELD_t, ndim=3] embedding_array,
    np.ndarray[ADJ_t, ndim=2] R, np.ndarray[DEGREE_t, ndim=1] nR,
    twins):

    cdef:
        int i, j, k, l
        DFIELD_t diff

    for i in range(N):
        # Initialize the recurrence matrix R and nR
        for j in range(n_time):
            for k in range(j+1):
                R[j, k] = R[k, j] = 1
            nR[j] = n_time

        # Calculate the recurrence matrix for time series i
        for j in range(n_time):
            # Ignore main diagonal, since every sample is neighbor of itself
            for k in range(j):
                for l in range(dimension):
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
        for j in range(n_time):
            twins_i = twins[i]
            twins_i.append([])
            twins_ij = twins_i[j]

            # Respect a minimal temporal spacing between twins to avoid false
            # twins due to the higher
            # sample density in phase space along the trajectory
            for k in range(j - min_dist):
                # Continue only if both samples have the same number of
                # neighbors and more than just one neighbor (themselves)
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


def _twin_surrogates_s(int n_surrogates, int N, twins,
                       np.ndarray[DFIELD_t, ndim=2] original_data):
    cdef:
        int i, j, k, l, new_k, n_twins, rand
        np.ndarray[DFIELD_t, ndim=2] surrogates = np.empty(
            (n_surrogates, N), dtype=DFIELD)

    for i in range(n_surrogates):
        # Get the twin list for time series i
        twins_i = twins[i]
        # Randomly choose a starting point in the original trajectory
        k = int(floor(random.random() * N))

        j = 0
        while j < N:
            # Assign state vector of surrogate trajectory
            surrogates[i,j] = original_data[i,k]
            # Get the list of twins of state vector k in the original time
            # series
            twins_ik = twins_i[k]

            # Get the number of twins of k
            n_twins = len(twins_ik)
            # If k has no twins, go to the next sample k+1, If k has twins at
            # m, choose among m+1 and k+1 with equal probability
            if n_twins == 0:
                k += 1
            else:
                # Generate a random integer between 0 and n_twins
                rand = int(floor(random.random() * (n_twins + 1)))

                # If rand = n_twings go to smple k+1, otherwise jump to the
                # future of one of the twins
                if rand == n_twins:
                    k += 1
                else:
                    k = twins_ik[rand]
                    k += 1

            # If the new k >= n_time, choose a new random starting point in the
            # original time series
            if k >= N:
                while True:
                    new_k = int(floor(random.random() * N))
                    if new_k != k:
                        break
                k = new_k

            j += 1

    return surrogates


def _test_pearson_correlation(
    np.ndarray[DFIELD_t, ndim=2, mode='c'] original_data not None,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] surrogates not None,
    int N, int n_time):

    cdef:
        DFIELD_t norm = 1.0 / float(n_time)
        #  Initialize Pearson correlation matrix
        np.ndarray[FIELD_t, ndim=2, mode='c'] correlation = np.zeros(
            (N, N), dtype=FIELD)

    _test_pearson_correlation_fast(
        <double*> np.PyArray_DATA(original_data),
        <double*> np.PyArray_DATA(surrogates),
        <float*> np.PyArray_DATA(correlation),
        n_time, N, norm)

    return correlation


def _test_mutual_information(
    np.ndarray[DFIELD_t, ndim=2, mode='c'] original_data not None,
    np.ndarray[DFIELD_t, ndim=2, mode='c'] surrogates not None,
    int N, int n_time, int n_bins):

    cdef:
        #  Get common range for all histograms
        DFIELD_t range_min = np.min((original_data.min(), surrogates.min()))
        DFIELD_t range_max = np.max((original_data.max(), surrogates.max()))
        #  Rescale all time series to the interval [0,1], using the maximum
        #  range of the whole dataset
        DFIELD_t scaling = 1. / (range_max - range_min)
        #  Create arrays to hold symbolic trajectories
        np.ndarray[NODE_t, ndim=2, mode='c'] symbolic_original = \
            np.empty((N, n_time), dtype=NODE)
        np.ndarray[NODE_t, ndim=2, mode='c'] symbolic_surrogates = \
            np.empty((N, n_time), dtype=NODE)
        #  Initialize array to hold 1d-histograms of individual time series
        np.ndarray[NODE_t, ndim=2, mode='c'] hist_original = \
            np.zeros((N, n_bins), dtype=NODE)
        np.ndarray[NODE_t, ndim=2, mode='c'] hist_surrogates = \
            np.zeros((N, n_bins), dtype=NODE)
        #  Initialize array to hold 2d-histogram for one pair of time series
        np.ndarray[NODE_t, ndim=2, mode='c'] hist2d = \
            np.zeros((n_bins, n_bins), dtype=NODE)
        #  Initialize mutual information array
        np.ndarray[FIELD_t, ndim=2, mode='c'] mi = np.zeros((N, N), dtype=FIELD)

    _test_mutual_information_fast(
            N, n_time, n_bins, scaling, range_min,
            <double*> np.PyArray_DATA(original_data),
            <double*> np.PyArray_DATA(surrogates),
            <int*> np.PyArray_DATA(symbolic_original),
            <int*> np.PyArray_DATA(symbolic_surrogates),
            <int*> np.PyArray_DATA(hist_original),
            <int*> np.PyArray_DATA(hist_surrogates),
            <int*> np.PyArray_DATA(hist2d),
            <float*> np.PyArray_DATA(mi))

    return mi


# recurrence plot==============================================================


def _embed_time_series(
    int n_time, int dim, int tau,
    np.ndarray[FIELD_t, ndim=2] time_series,
    np.ndarray[FIELD_t, ndim=2] embedding):

    cdef:
        unsigned int j, max_delay, index, D = dim
        int k, len_embedded

    # Calculate the maximum delay
    max_delay = (dim - 1) * tau
    # Calculate the length of the embedded time series
    len_embedded = n_time - max_delay

    for j in range(D):
        index = j * tau
        for k in range(len_embedded):
            embedding[k, j] = time_series[index]
            index += 1


def _manhattan_distance_matrix_rp(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=2] distance):

    cdef:
        unsigned int j, k, l, T = n_time, D = dim
        float sum

    # Calculate the manhattan distance matrix
    for j in range(T):
        # Ignore the main diagonal, since every samle is neighbor of itself
        for k in range(j):
            sum = 0
            for l in range(D):
                # use manhattan norm
                sum += abs(embedding[j, l] - embedding[k, l])

            distance[j, k] = distance[k, j] = sum


def _euclidean_distance_matrix_rp(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=2] distance):

    cdef:
        unsigned int j, k, l, T = n_time, D = dim
        float sum, diff

    # Calculate the eucliadean distance matrix
    for j in range(T):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in range(j):
            sum = 0
            for l in range(D):
                # Use euclidean norm
                diff = abs(embedding[j, l] - embedding[k, l])
                sum += diff * diff
            distance[j, k] = distance[k, j] = sum


def _supremum_distance_matrix_rp(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=2] distance):

    cdef:
        unsigned int j, k, l, T = n_time, D = dim
        float temp_diff, diff

    # Calculate the eucliadean distance matrix
    for j in range(T):
        # Ignore the main diagonal, since every sample is neighbor of itself
        for k in range(j):
            temp_diff = diff = 0
            for l in range(D):
                # Use supremum norm
                temp_diff = abs(embedding[j, l] - embedding[k, l])
                if temp_diff > diff:
                    diff = temp_diff

            distance[j, k] = distance[k, j] = diff


def _set_adaptive_neighborhood_size(
    int n_time, int adaptive_neighborhood_size,
    np.ndarray[NODE_t, ndim=2] sorted_neighbors,
    np.ndarray[NODE_t, ndim=1] order,
    np.ndarray[LAG_t, ndim=2] recurrence):

    cdef:
        int i, j, k, l

    for i in range(adaptive_neighborhood_size):
        for j in range(n_time):
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


def _bootstrap_distance_matrix_manhattan(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[NODE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float sum, diff

    for i in range(M):
        #Compute their distance
        sum = 0
        for l in range(dim):
            # Use Manhattan norm
            sum += abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])

        distances[i] = sum


def _bootstrap_distance_matrix_euclidean(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[NODE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float sum, diff

    for i in range(M):
        #Compute their distance
        sum = 0
        for l in range(dim):
            # Use Manhattan norm
            diff = abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])
            sum += diff * diff

        distances[i] = sqrt(sum)


def _bootstrap_distance_matrix_supremum(
    int n_time, int dim, np.ndarray[FIELD_t, ndim=2] embedding,
    np.ndarray[FIELD_t, ndim=1] distances, int M):

    cdef:
        int i, l
        np.ndarray[NODE_t, ndim=2] jk = rd.randint(n_time, size=(2,M))
        float temp_diff, diff

    for i in range(M):
        #Compute their distance
        temp_diff = diff = 0
        for l in range(dim):
            # Use supremum norm
            temp_diff = abs(embedding[jk[0, i], l] - embedding[jk[1, i], l])

            if temp_diff > diff:
                diff = temp_diff

            distances[i] = diff


def _diagline_dist_norqa_missingvalues(
    int n_time, np.ndarray[NODE_t, ndim=1] diagline,
    np.ndarray[LAG_t, ndim=2] recmat,
    np.ndarray[MASK_t, ndim=1, cast=True] mv_indices):

    cdef:
        int i, j, k = 0
        MASK_t missing_flag = False

    for i in range(n_time):
        if k != 0 and not missing_flag:
            diagline[k] += 1
            k = 0

        missing_flag = False

        for j in range(i+1):
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
    int n_time, np.ndarray[NODE_t, ndim=1] diagline,
    np.ndarray[LAG_t, ndim=2] recmat):

    cdef:
        int i, j, k = 0

    for i in range(n_time):
        if k != 0:
            diagline[k] += 1
            k = 0
        for j in range(i+1):
            if recmat[n_time-1-i+j, j] == 1:
                k += 1
            elif k != 0:
                diagline[k] += 1
                k = 0


def _diagline_dist_rqa_missingvalues(
    int n_time, np.ndarray[NODE_t, ndim=1] diagline,
    np.ndarray[MASK_t, ndim=1, cast=True] mv_indices,
    np.ndarray[FIELD_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        FIELD_t temp_diff, diff
        MASK_t missing_flag = False

    for i in range(n_time):
        if k != 0 and not missing_flag:
            diagline[k] += 1
            k = 0

        missing_flag = False

        for j in range(i+1):
            # Compute supreumum distance between state vectors
            temp_diff = diff = 0
            for l in range(dim):
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
    int n_time, np.ndarray[NODE_t, ndim=1] diagline,
    np.ndarray[FIELD_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        FIELD_t temp_diff, diff

    for i in range(n_time):
        if k != 0:
            diagline[k] += 1
            k = 0

        for j in range(i+1):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in range(dim):
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
    np.ndarray[FIELD_t, ndim=1] dist,
    np.ndarray[FIELD_t, ndim=1] resampled_dist, int N, int M):

    cdef:
        int i = 0, x

    while i < M:
        x = int(floor(random.random() * N))

        if (random.random() < dist[x]):
            resampled_dist[x] += 1
            i += 1


def _vertline_dist_norqa_missingvalues(
    int n_time, np.ndarray[NODE_t, ndim=1] vertline,
    np.ndarray[LAG_t, ndim=2] recmat,
    np.ndarray[MASK_t, ndim=1, cast=True] mv_indices):

    cdef:
        int i, j, k = 0
        MASK_t missing_flag = False

    for i in range(n_time):
        if (k != 0 and not missing_flag):
            vertline[k] += 1
            k = 0

        missing_flag = False

        for j in range(n_time):
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
    int n_time, np.ndarray[NODE_t, ndim=1] vertline,
    np.ndarray[LAG_t, ndim=2] recmat):

    cdef int i, j, k = 0

    for i in range(n_time):
        if k != 0:
            vertline[k] += 1
            k = 0

        for j in range(n_time):
            if recmat[i, j] != 0:
                k += 1
            elif k != 0:
                vertline[k] += 1
                k = 0


def _vertline_dist_rqa_missingvalues(
    int n_time, np.ndarray[NODE_t, ndim=1] vertline,
    np.ndarray[MASK_t, ndim=1, cast=True] mv_indices,
    np.ndarray[FIELD_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        FIELD_t temp_diff, diff
        MASK_t missing_flag = False

    for i in range(n_time):
        if k != 0 and not missing_flag:
            vertline[k] += 1
            k = 0

        missing_flag = False

        for j in range(n_time):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in range(dim):
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
    int n_time, np.ndarray[NODE_t, ndim=1] vertline,
    np.ndarray[FIELD_t, ndim=2] embedding, float eps, int dim):

    cdef:
        int i, j, k = 0, l
        FIELD_t temp_diff, diff

    for i in range(n_time):
        if k != 0:
            vertline[k] += 1
            k = 0

        for j in range(n_time):
            # Compute supremum distance between state vectors
            temp_diff = diff = 0
            for l in range(dim):
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
    int n_time, np.ndarray[NODE_t, ndim=1] white_vertline,
    np.ndarray[LAG_t, ndim=2] R):

    cdef int i, j, k = 0

    for i in range(n_time):
        if k != 0:
            white_vertline[k] += 1
            k = 0

        for j in range(n_time):
            if R[i, j] == 0:
                k += 1
            elif k != 0:
                white_vertline[k] += 1
                k = 0


def _twins_r(
    int min_dist, int N, np.ndarray[LAG_t, ndim=2] R,
    np.ndarray[NODE_t, ndim=1] nR, twins):

    cdef int j, k, l

    twins.append([])

    for j in range(N):
        twins.append([])
        twins_j = twins[j]

        # Respect a minimal temporal spacing between twins to avoid false
        # twins du to th higher sample density in phase space along the
        # trajectory
        for k in range(j - min_dist):
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


def _twin_surrogates_r(int n_surrogates, int N, int dim, twins,
                       np.ndarray[DFIELD_t, ndim=2] embedding):
    cdef:
        int i, j, k, new_k, n_twins, rand
        np.ndarray[DFIELD_t, ndim=2] surrogates = np.empty(
            (n_surrogates, N, dim), dtype=DFIELD)

    # Initialize random number generator
    random.seed(datetime.now())

    for i in range(n_surrogates):
        # Randomly choose a starting point in the original trajectory
        k = int(floor(random.random() * N))

        j = 0
        while j < N:
            # Assign state vector of surrogate trajectory
            surrogates[i, j, :] = embedding[k, :]
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
                rand = int(floor(random.random() * (n_twins + 1)))

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
                    new_k = int(floor(random.random() * N))
                    if new_k != k:
                        break
                k = new_k

            j += 1

    return surrogates


# visibitly graph =============================================================


def _visibility_relations_missingvalues(
    np.ndarray[FIELD_t, ndim=1] x, np.ndarray[FIELD_t, ndim=1] t,
    int N, np.ndarray[MASK_t, ndim=2] A,
    np.ndarray[MASK_t, ndim=1, cast=True] mv_indices):

    cdef:
        int i, j, k
        FIELD_t test

    for i in range(N-2):
        for j in range(i+2, N):
            k = i + 1

            test = (x[j] - x[i]) / (t[j] - t[i])

            while not mv_indices[k] and\
                (x[k] - x[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in range(N-1):
        if not mv_indices[i] and not mv_indices[i+1]:
            A[i, i+1] = A[i+1, i] = 1


def _visibility_relations_no_missingvalues(
    np.ndarray[FIELD_t, ndim=1] x, np.ndarray[FIELD_t, ndim=1] t,
    int N, np.ndarray[MASK_t, ndim=2] A):

    cdef:
        int i, j, k
        FIELD_t test

    for i in range(N-2):
        for j in range(i+2, N):
            k = i + 1

            test = (x[j] - x[i]) / (t[j] - t[i])

            while (x[k] - x[i]) / (t[k] - t[i]) < test and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in range(N-1):
        A[i, i+1] = A[i+1, i] = 1


def _visibility_relations_horizontal(
    np.ndarray[FIELD_t, ndim=1] x, np.ndarray[FIELD_t, ndim=1] t,
    int N, np.ndarray[MASK_t, ndim=2] A):

    cdef:
        int i, j, k
        FIELD_t minimum

    for i in range(N-2):
        for j in range(i+2, N):
            k = i + 1
            minimum = min(x[i], x[j])

            while x[k] < minimum and k < j:
                k += 1

            if k == j:
                A[i, j] = A[j, i] = 1

    # Add trivial connections of subsequent observations in time series
    for i in range(N-1):
        A[i, i+1] = A[i+1, i] = 1


def _visibility(
        np.ndarray[FIELD_t, ndim=1] time,
        np.ndarray[FIELD_t, ndim=1] val, int node1, int node2):

    cdef:
        int i, j, k
        np.ndarray[MASK_t, ndim=1] test

    i = min(node1,node2)
    j = max(node1,node2)

    test = np.zeros((j-(i+1)), dtype=np.uint8)
    for k in range(i+1,j):
        test[k-(i+1)] = np.less((val[k]-val[i]) / (time[k]-time[i]),
                                (val[j]-val[i]) / (time[j]-time[i]))
    return np.invert(np.bool(np.sum(test)))


def _retarded_local_clustering(
    int N, np.ndarray[ADJ_t, ndim=2] A,
    np.ndarray[FIELD_t, ndim=1] norm,
    np.ndarray[FIELD_t, ndim=1] retarded_clustering):

    cdef:
        int i, j, k
        long counter

    # Loop over all nodes
    for i in range(N):
        # Check if i has right degree larger than 1
        if norm[i] != 0:
            # Reset counter
            counter = 0

            # Loop over unique pairs of nodes in the past of i
            for j in range(i):
                for k in range(j):
                    if A[i, j] == 1 and A[j, k] == 1 and A[k, i] == 1:
                        counter += 1

            retarded_clustering[i] = counter / norm[i]


def _advanced_local_clustering(
    int N, np.ndarray[ADJ_t, ndim=2] A,
    np.ndarray[FIELD_t, ndim=1] norm,
    np.ndarray[FIELD_t, ndim=1] advanced_clustering):

    cdef:
        int i, j, k
        long counter

    # Loop over all nodes
    for i in range(N-2):
        # Check if i has right degree larger than 1
        if norm[i] != 0:
            # Reset counter
            counter = 0

            # Loop over unique pairs of nodes in the future of i
            for j in range(i+1, N):
                for k in range(i+1, j):
                    if A[i, j] == 1 and A[j, k] == 1 and A[k, i] == 1:
                        counter += 1

            advanced_clustering[i] = counter / norm[i]