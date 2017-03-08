# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)


cimport cython
cimport numpy as np

import numpy as np
import numpy.random as rd

randint = rd.randint

INTTYPE = np.int
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
ctypedef np.int_t INTTYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t


# interacting_networks ========================================================

cdef void overwriteAdjacency(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[INTTYPE_t, ndim=2] cross_A,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    int m, int n):
    """
    Overwrite the adjacency matrix of the full interacting network with the
    randomly rewired cross edges of the two considered subnetworks.
    """
    cdef:
        unsigned int i, j
        INTTYPE_t n1, n2

    for i in xrange(m):
        for j in xrange(n):
            n1, n2 = nodes1[i], nodes2[j]
            A[n1, n2] = A[n2, n1] = cross_A[i, j]


def _randomlySetCrossLinks(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[INTTYPE_t, ndim=2] cross_A,
    int number_cross_links,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    int m, int n):
    """
    >>> A = np.eye(2, dtype=np.int)
    >>> _randomlySetCrossLinks(A, np.array([[0]]), 1,
    ...                        np.array([0]), np.array([1]), 1, 1)
    >>> np.all(A == np.ones(2))
    True
    """
    cdef:
        unsigned int i, j

    # create random cross links
    for _ in xrange(number_cross_links):
        while True:
            i, j = randint(m), randint(n)
            if not cross_A[i, j]:
                break
        cross_A[i, j] = 1
    overwriteAdjacency(A, cross_A, nodes1, nodes2, m, n)


def _randomlyRewireCrossLinks(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[INTTYPE_t, ndim=2] cross_A,
    np.ndarray[INTTYPE_t, ndim=2] cross_links,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    int number_cross_links, int number_swaps):

    cdef:
        int m = len(nodes1), n = len(nodes2)
        INTTYPE_t e1, e2, a, b, c, d

    # implement permutations
    for _ in xrange(number_swaps):
        while True:
            # choose two random edges
            e1, e2 = randint(number_cross_links), randint(number_cross_links)
            a, b = cross_links[e1]
            c, d = cross_links[e2]
            # repeat the procedure in case there already exists a link between
            # starting point of e1 and ending point of e2 or vice versa
            if not (cross_A[a, d] or cross_A[c, b]):
                break
        # delete initial edges within the cross adjacency matrix
        cross_A[a, b] = cross_A[c, d] = 0
        # create new edges within the cross adjacency matrix by swapping the
        # ending points of e1 and e2
        cross_A[a, d] = cross_A[c, b] = 1
        # likewise, adjust cross_links
        b                  = cross_links[e1, 1]
        cross_links[e1, 1] = cross_links[e2, 1]
        cross_links[e2, 1] = b
    overwriteAdjacency(A, cross_A, nodes1, nodes2, m, n)


def _cross_transitivity(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[INTTYPE_t, ndim=1] nodes1,
    np.ndarray[INTTYPE_t, ndim=1] nodes2):

    cdef:
        int m = len(nodes1), n = len(nodes2)
        unsigned int i, j, k
        INTTYPE_t n1, n2, n3
        long triangles = 0, triples = 0

    for i in xrange(m):
        n1 = nodes1[i]
        # loop over unique pairs of nodes in subnetwork 2
        for j in xrange(n):
            n2 = nodes2[j]
            if A[n1, n2]:
                for k in xrange(j):
                    n3 = nodes2[k]
                    if A[n1, n3]:
                        triples += 1
                    if A[n2, n3] and A[n3, n1]:
                        triangles += 1
    if triples:
        return triangles / float(triples)
    else:
        return 0.0


def _nsi_cross_transitivity(
    np.ndarray[INTTYPE_t, ndim=2] A,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    np.ndarray[FLOATTYPE_t, ndim=1] node_weights):

    cdef:
        int m = len(nodes1), n = len(nodes2)
        unsigned int v, p, q
        INTTYPE_t node_v, node_p, node_q
        FLOATTYPE_t weight_v, weight_p, ppv, pqv, T1 = 0, T2 = 0

    for v in xrange(m):
        node_v = nodes1[v]
        weight_v = node_weights[node_v]
        for p in xrange(n):
            node_p = nodes2[p]
            if A[node_v, node_p]:
                weight_p = node_weights[node_p]
                ppv = weight_p * weight_p * weight_v
                T1 += ppv
                T2 += ppv
                for q in xrange(p + 1, n):
                    node_q = nodes2[q]
                    if A[node_v, node_q]:
                        pqv = 2 * weight_p * node_weights[node_q] * weight_v
                        T2 += pqv
                        if A[node_p, node_q]:
                            T1 += pqv
    return T1 / T2


def _cross_local_clustering(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[FLOATTYPE_t, ndim=1] norm,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    np.ndarray[FLOATTYPE_t, ndim=1] cross_clustering):

    cdef:
        int m = len(nodes1), n = len(nodes2)
        unsigned int i, j, k
        INTTYPE_t n1, n2, n3
        long counter

    for i in xrange(m):
        n1 = nodes1[i]
        # check if node1[i] has cross degree larger than 1
        if norm[i]:
            counter = 0
            # loop over unique pairs of nodes in subnetwork 2
            for j in xrange(n):
                n2 = nodes2[j]
                if A[n1, n2]:
                    for k in xrange(j):
                        n3 = nodes2[k]
                        if A[n2, n3] and A[n3, n1]:
                            counter += 1
            cross_clustering[i] = counter / norm[i]


def _nsi_cross_local_clustering(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[FLOATTYPE_t, ndim=1] nsi_cc,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    np.ndarray[FLOATTYPE_t, ndim=1] node_weights):

    cdef:
        int m = len(nodes1), n = len(nodes2)
        unsigned int v, p, q
        INTTYPE_t node_v, node_p, node_q
        FLOATTYPE_t weight_p

    for v in xrange(m):
        node_v = nodes1[v]
        for p in xrange(n):
            node_p = nodes2[p]
            if A[node_v, node_p]:
                weight_p = node_weights[node_p]
                nsi_cc[v] += weight_p * weight_p
                for q in xrange(p + 1, n):
                    node_q = nodes2[q]
                    if A[node_p, node_q] and A[node_q, node_v]:
                        nsi_cc[v] += 2 * weight_p * node_weights[node_q]

# network =====================================================================

def _local_cliquishness_4thorder(
    int N, np.ndarray[INTTYPE_t, ndim=2] A,
    np.ndarray[INTTYPE_t, ndim=1] degree):

    cdef:
        unsigned int order = 4, index
        INTTYPE_t node1, node2, node3, degree_i
        long counter
        np.ndarray[INTTYPE_t, ndim=1] neighbors = np.zeros(N, dtype=INTTYPE)
        np.ndarray[FLOATTYPE_t, ndim=1] local_cliquishness = \
            np.zeros(N, dtype=FLOATTYPE)

    # Iterate over all nodes
    for i in xrange(N):
        # If degree is smaller than order - 1, set local cliquishness to 0
        degree_i = degree[i]
        if degree_i >= order - 1:
            # Get neighbors of node i
            index = 0
            for j in xrange(N):
                if A[i, j] == 1:
                    neighbors[index] = j
                    index += 1
            counter = 0
            # Iterate over possibly existing edges between 3 neighbors of i
            for j in xrange(degree_i):
                node1 = neighbors[j]
                for k in xrange(degree_i):
                    node2 = neighbors[k]
                    if A[node1, node2] == 1:
                        for l in xrange(degree_i):
                            node3 = neighbors[l]
                            if A[node2, node3] == 1 and A[node3, node1] == 1:
                                counter += 1
            local_cliquishness[i] = float(counter) / degree_i /\
                (degree_i - 1) / (degree_i - 2)
    return local_cliquishness


def _local_cliquishness_5thorder(
    int N, np.ndarray[INTTYPE_t, ndim=2] A,
    np.ndarray[INTTYPE_t, ndim=1] degree):

    cdef:
        unsigned int index, order = 5
        INTTYPE_t j, node1, node2, node3, node4, degree_i
        long counter
        np.ndarray[INTTYPE_t, ndim=1] neighbors = np.zeros(N, dtype=INTTYPE)
        np.ndarray[FLOATTYPE_t, ndim=1] local_cliquishness = \
            np.zeros(N, dtype=FLOATTYPE)

    # Iterate over all nodes
    for i in xrange(N):
        # If degree is smaller than order - 1, set local cliquishness to 0
        degree_i = degree[i]
        if degree_i >= order - 1:
            # Get neighbors of node i
            index = 0
            for j in xrange(N):
                if A[i, j] == 1:
                    neighbors[index] = j
                    index += 1
            counter = 0
            # Iterate over possibly existing edges between 4 neighbors of i
            for j in xrange(degree_i):
                node1 = neighbors[j]
                for k in xrange(degree_i):
                    node2 = neighbors[k]
                    if A[node1, node2] == 1:
                        for l in xrange(degree_i):
                            node3 = neighbors[l]
                            if A[node1, node3] == 1 and A[node2, node3] == 1:
                                for m in xrange(degree_i):
                                    node4 = neighbors[m]
                                    if (A[node1, node4] == 1 and
                                        A[node2, node4] == 1 and
                                        A[node3, node4] == 1):
                                        counter += 1
            local_cliquishness[i] = float(counter) / degree_i /\
                (degree_i - 1) / (degree_i - 2) / (degree_i -3)
    return local_cliquishness


def _nsi_betweenness(
    int N, int E, np.ndarray[FLOATTYPE_t, ndim=1] w,
    np.ndarray[INTTYPE_t, ndim=1] k, int j ,
    np.ndarray[FLOATTYPE_t, ndim=1] betweenness_to_j,
    np.ndarray[FLOATTYPE_t, ndim=1] excess_to_j,
    np.ndarray[INTTYPE_t, ndim=1] offsets,
    np.ndarray[INTTYPE_t, ndim=1] flat_neighbors,
    np.ndarray[FLOATTYPE_t, ndim=1] is_source,
    np.ndarray[INTTYPE_t, ndim=1] flat_predecessors):

    cdef:
        unsigned int qi, oi, queue_len, l_index, ql, fi
        INTTYPE_t l, i, next_d, dl, ol
        float base_factor
        np.ndarray[INTTYPE_t, ndim=1] distances_to_j =\
            2 * N * np.ones(N, dtype=INTTYPE)
        np.ndarray[INTTYPE_t, ndim=1] n_predecessors =\
            np.zeros(N, dtype=INTTYPE)
        np.ndarray[INTTYPE_t, ndim=1] queue =\
          np.zeros(N, dtype=INTTYPE)
        np.ndarray[FLOATTYPE_t, ndim=1] multiplicity_to_j =\
            np.zeros(N, dtype=FLOATTYPE)

    # init distances to j and queue of nodes by distance from j
    for l in xrange(N):
        # distances_to_j[l] = 2 * N
        # n_predecessors[l] = 0
        # multiplicity_to_j[l] = 0.0
        # initialize contribution of paths ending in j to the betweenness of l
        excess_to_j[l] = betweenness_to_j[l] = is_source[l] * w[l]

    distances_to_j[j] = 0
    queue[0] = j
    queue_len = 1
    multiplicity_to_j[j] = w[j]

    # process the queue forward and grow it on the way: (this is the standard
    # breadth-first search giving all the shortest paths to j)
    qi = 0
    while qi < queue_len:
    #for qi in xrange(queue_len):
        i = queue[qi]
        if i == -1:
            # this should never happen ...
            print "Opps: %d,%d,%d\n" % qi, queue_len, i
            break
        next_d = distances_to_j[i] + 1
        #iterate through all neighbors l of i
        oi = offsets[i]
        for l_index in xrange(oi, oi+k[i]):
            # if on a shortes j-l-path, register i as predecessor of l
            l = flat_neighbors[l_index]
            dl = distances_to_j[l]
            if dl >= next_d:
                fi = offsets[l] + n_predecessors[l]
                n_predecessors[l] += 1
                flat_predecessors[fi] = i
                multiplicity_to_j[l] += w[l] * multiplicity_to_j[i]
                if dl > next_d:
                    distances_to_j[l] = next_d
                    queue[queue_len] = l
                    queue_len += 1
        qi += 1

    # process the queue again backward: (this is Newman's 2nd part where
    # the contribution of paths ending in j to the betweenness of all nodes
    # is computed recursively by traversing the shortest paths backwards)
    for ql in xrange(queue_len-1, -1, -1):
        l = queue[ql]
        if l == -1:
            print "Opps: %d,%d,%d\n" % ql, queue_len, l
            break
        if l == j:
            # set betweenness and excess to zero
            betweenness_to_j[l] = excess_to_j[l] = 0
        else:
            # otherwise, iterate through all predecessors i of l:
            base_factor = w[l] / multiplicity_to_j[l]
            ol = offsets[l]
            for fi in xrange(ol, ol+n_predecessors[l]):
                # add betweenness to predecessor
                i = flat_predecessors[fi]
                betweenness_to_j[i] += betweenness_to_j[l] * base_factor * \
                    multiplicity_to_j[i]


def _cy_mpi_newman_betweenness(
    np.ndarray[INTTYPE_t, ndim=2] this_A, np.ndarray[FLOATTYPE_t, ndim=2] V,
    int N, int start_i, int end_i):
    """
    This function does the outer loop for a certain range start_i-end_i of
    c's.  it gets the full V matrix but only the needed rows of the A matrix.
    Each parallel job will consist of a call to this function:
    """

    cdef:
        unsigned int i_rel, j, s, t, i_abs
        float sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        np.ndarray[FLOATTYPE_t, ndim=1] this_betweenness =\
            np.zeros(this_N, dtype=FLOATTYPE)

    for i_rel in xrange(this_N):
        # correct i index for V matrix
        i_abs = i_rel + start_i
        for j in xrange(N):
             if this_A[i_rel, j]:
                sum_j = 0.0
                for s in xrange(N):
                    if i_abs != s:
                        Vis_minus_Vjs = V[i_abs, s] - V[j, s]
                        sum_s = 0.0
                        for t in xrange(s):
                            if i_abs != t:
                                sum_s += abs(Vis_minus_Vjs - V[i_abs, t] +
                                             V[j, t])
                        sum_j += sum_s
                this_betweenness[i_rel] += sum_j

    return this_betweenness, start_i, end_i


def _cy_mpi_nsi_newman_betweenness(
    np.ndarray[INTTYPE_t, ndim=2] this_A, np.ndarray[FLOATTYPE_t, ndim=2] V,
    int N, np.ndarray[FLOATTYPE_t, ndim=1] w,
    np.ndarray[INTTYPE_t, ndim=2] this_not_adj_or_equal, int start_i,
    int end_i):

    cdef:
        unsigned int i_rel, j, s, t, i_abs
        float sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        np.ndarray[FLOATTYPE_t, ndim=1] this_betweenness =\
            np.zeros(this_N, dtype=FLOATTYPE)

    for i_rel in xrange(this_N):
        i_abs = i_rel + start_i
        for j in xrange(N):
             if this_A[i_rel, j]:
                sum_j = 0.0
                for s in xrange(N):
                    if this_not_adj_or_equal[i_rel, s]:
                        Vis_minus_Vjs = V[i_abs, s] - V[j, s]
                        sum_s = 0.0
                        for t in xrange(s):
                            if this_not_adj_or_equal[i_rel, t]:
                                sum_s += w[t] *\
                                    abs(Vis_minus_Vjs - V[i_abs, t] + V[j, t])
                        sum_j += w[s] * sum_s
                this_betweenness[i_rel] += w[j] * sum_j

    return this_betweenness, start_i, end_i


# grid ========================================================================

def _cy_calculate_angular_distance(
    np.ndarray[FLOAT32TYPE_t, ndim=1] cos_lat,
    np.ndarray[FLOAT32TYPE_t, ndim=1] sin_lat,
    np.ndarray[FLOAT32TYPE_t, ndim=1] cos_lon,
    np.ndarray[FLOAT32TYPE_t, ndim=1] sin_lon,
    np.ndarray[FLOAT32TYPE_t, ndim=2] cosangdist, int N):

    cdef:
        FLOAT32TYPE_t expr
        unsigned int i,j

    for i in xrange(N):
        for j in xrange(i+1):
            expr = sin_lat[i]*sin_lat[j] + cos_lat[i]*cos_lat[j] * \
                (sin_lon[i]*sin_lon[j] + cos_lon[i]*cos_lon[j])

            if expr > 1:
                expr = 1
            elif expr < -1:
                expr = -1

            cosangdist[i, j] = cosangdist[j, i] = expr


def _euclidiean_distance(
    np.ndarray[FLOAT32TYPE_t, ndim=1] x, np.ndarray[FLOAT32TYPE_t, ndim=1] y,
    np.ndarray[FLOAT32TYPE_t, ndim=2] distance, int N):

    cdef:
        unsigned int i,j
        FLOAT32TYPE_t expr

    for i in xrange(N):
        for j in xrange(i+1):
            expr = (x[i]-x[j])**2 + (y[i]-y[j])**2
            distance[i, j] = distance[j, i] = expr**(0.5)
