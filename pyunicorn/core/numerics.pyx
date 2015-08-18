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
