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
cimport numpy as np

import numpy as np
import numpy.random as rd

randint = rd.randint

INTTYPE = np.int
INT16TYPE = np.int16
INT32TYPE = np.int32
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
FLOAT64TYPE = np.float64
ctypedef np.int_t INTTYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t
ctypedef np.float64_t FLOAT64TYPE_t

cdef extern from "stdlib.h":
    double drand48()
    double srand48()

cdef extern from "time.h":
    double time()

cdef extern from "src_numerics.c":
    void _randomly_rewire_geomodel_I_fast(int iterations, float eps, short *A,
        float *D, int E, int N, int *edges)
    void _randomly_rewire_geomodel_II_fast(int iterations, float eps, short *A,
        float *D, int E, int N, int *edges)
    void _randomly_rewire_geomodel_III_fast(int iterations, float eps,
        short *A, float *D, int E, int N, int *edges, int *degree)
    double _higher_order_transitivity4_fast(int N, short* A)
    void _do_nsi_hamming_clustering_fast(int n2, int nActiveIndices,
        float mind0, float minwp0, int lastunited, int part1, int part2,
        float *distances, int *theActiveIndices, float *linkedWeights,
        float *weightProducts, float *errors, float *result, int *mayJoin)
    double _vertex_current_flow_betweenness_fast(int N, double Is, double It,
        float *admittance, float *R, int i)
    void _edge_current_flow_betweenness_fast(int N, double Is, double It,
        float *admittance, float *R, float *ECFB)


# geo_network =================================================================

def _randomly_rewire_geomodel_I(int iterations, float eps,
    np.ndarray[short, ndim=2] A, np.ndarray[float, ndim=2] D, int E, int N,
    np.ndarray[INTTYPE_t, ndim=2] edges):

    cdef:
        int i, j, s, t, k, l, edge1, edge2, count
        int neighbor_s_index, neighbor_t_index
        int neighbor_k_index, neighbor_l_index

    #  Create list of neighbors
    #for (int i = 0; i < N; i++) {
    #
    #    count = 0;
    #
    #    for (int j = 0; j < N; j++) {
    #        if (A(i,j) == 1) {
    #            neighbors(i,count) = j;
    #            count++;
    #        }
    #    }
    #}

    #  Initialize random number generator
    #srand48(time(0))

    i = 0
    count = 0
    while (i < iterations):
        # Randomly choose 2 edges
        edge1 = np.floor(rd.random() * E)
        edge2 = np.floor(rd.random() * E)

        s = edges[edge1,0]
        t = edges[edge1,1]

        k = edges[edge2,0]
        l = edges[edge2,1]

        #  Randomly choose 2 nodes
        #s = floor(drand48() * N);
        #k = floor(drand48() * N);

        #  Randomly choose 1 neighbor of each
        #neighbor_s_index = floor(drand48() * degree(s));
        #neighbor_k_index = floor(drand48() * degree(k));
        #t = neighbors(s,neighbor_s_index);
        #l = neighbors(k,neighbor_k_index);

        count+=1

        #  Proceed only if s != k, s != l, t != k, t != l
        if (s != k and s != l and t != k and t != l):
            # Proceed only if the new links {s,l} and {t,k}
            # do NOT already exist
            if (A[s,l] == 0 and A[t,k] == 0):
                # Proceed only if the link lengths fulfill condition C1
                if ((np.abs(D[s,t] - D[k,t]) < eps and
                        np.abs(D[k,l] - D[s,l]) < eps ) or
                            (np.abs(D[s,t] - D[s,l]) < eps and
                                np.abs(D[k,l] - D[k,t]) < eps )):
                    # Now rewire the links symmetrically
                    # and increase i by 1
                    A[s,t] = 0
                    A[t,s] = 0
                    A[k,l] = 0
                    A[l,k] = 0
                    A[s,l] = 1
                    A[l,s] = 1
                    A[t,k] = 1
                    A[k,t] = 1

                    edges[edge1,0] = s
                    edges[edge1,1] = l
                    edges[edge2,0] = k
                    edges[edge2,1] = t

                    #  Update neighbor lists of all 4 involved nodes
                    #neighbors(s,neighbor_s_index) = l;
                    #neighbors(k,neighbor_k_index) = t;

                    #neighbor_t_index = 0;
                    #while (neighbors(t,neighbor_t_index) != s) {
                    #    neighbor_t_index++;
                    #}
                    #neighbors(t,neighbor_t_index) = k;

                    #neighbor_l_index = 0;
                    #while (neighbors(l,neighbor_l_index) != k) {
                    #    neighbor_l_index++;
                    #}
                    #neighbors(l,neighbor_l_index) = s;

                    i+=1

    print("Trials %d, Rewirings %d", (count, iterations))
    """
    _randomly_rewire_geomodel_I_fast(iterations, eps,
            <short*> np.PyArray_DATA(A), <float*> np.PyArray_DATA(D), E, N,
            <int*> np.PyArray_DATA(edges))
    """

def _randomly_rewire_geomodel_II(int iterations, float eps,
    np.ndarray[short, ndim=2] A, np.ndarray[float, ndim=2] D, int E, int N,
    np.ndarray[INTTYPE_t, ndim=2] edges):

    _randomly_rewire_geomodel_II_fast(iterations, eps,
            <short*> np.PyArray_DATA(A), <float*> np.PyArray_DATA(D), E, N,
            <int*> np.PyArray_DATA(edges))

def _randomly_rewire_geomodel_III(int iterations, float eps,
    np.ndarray[short, ndim=2] A, np.ndarray[float, ndim=2] D, int E, int N,
    np.ndarray[INTTYPE_t, ndim=2] edges, np.ndarray[INTTYPE_t, ndim=1] degree):

    _randomly_rewire_geomodel_III_fast(iterations, eps,
            <short*> np.PyArray_DATA(A), <float*> np.PyArray_DATA(D), E, N,
            <int*> np.PyArray_DATA(edges), <int*> np.PyArray_DATA(degree))


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
        int i, j
        INTTYPE_t n1, n2

    for i in range(m):
        for j in range(n):
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
    for _ in range(number_cross_links):
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
    for _ in range(number_swaps):
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
        unsigned int m = len(nodes1), n = len(nodes2)
        unsigned int i, j, k
        INTTYPE_t n1, n2, n3
        long triangles = 0, triples = 0

    for i in range(m):
        n1 = nodes1[i]
        # loop over unique pairs of nodes in subnetwork 2
        for j in range(n):
            n2 = nodes2[j]
            if A[n1, n2]:
                for k in range(j):
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
        unsigned int m = len(nodes1), n = len(nodes2)
        unsigned int v, p, q
        INTTYPE_t node_v, node_p, node_q
        FLOATTYPE_t weight_v, weight_p, ppv, pqv, T1 = 0, T2 = 0

    for v in range(m):
        node_v = nodes1[v]
        weight_v = node_weights[node_v]
        for p in range(n):
            node_p = nodes2[p]
            if A[node_v, node_p]:
                weight_p = node_weights[node_p]
                ppv = weight_p * weight_p * weight_v
                T1 += ppv
                T2 += ppv
                for q in range(p + 1, n):
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
        unsigned int m = len(nodes1), n = len(nodes2)
        unsigned int i, j, k
        INTTYPE_t n1, n2, n3
        long counter

    for i in range(m):
        n1 = nodes1[i]
        # check if node1[i] has cross degree larger than 1
        if norm[i]:
            counter = 0
            # loop over unique pairs of nodes in subnetwork 2
            for j in range(n):
                n2 = nodes2[j]
                if A[n1, n2]:
                    for k in range(j):
                        n3 = nodes2[k]
                        if A[n2, n3] and A[n3, n1]:
                            counter += 1
            cross_clustering[i] = counter / norm[i]


def _nsi_cross_local_clustering(
    np.ndarray[INTTYPE_t, ndim=2] A, np.ndarray[FLOATTYPE_t, ndim=1] nsi_cc,
    np.ndarray[INTTYPE_t, ndim=1] nodes1, np.ndarray[INTTYPE_t, ndim=1] nodes2,
    np.ndarray[FLOATTYPE_t, ndim=1] node_weights):

    cdef:
        unsigned int m = len(nodes1), n = len(nodes2)
        unsigned int v, p, q
        INTTYPE_t node_v, node_p, node_q
        FLOATTYPE_t weight_p

    for v in range(m):
        node_v = nodes1[v]
        for p in range(n):
            node_p = nodes2[p]
            if A[node_v, node_p]:
                weight_p = node_weights[node_p]
                nsi_cc[v] += weight_p * weight_p
                for q in range(p + 1, n):
                    node_q = nodes2[q]
                    if A[node_p, node_q] and A[node_q, node_v]:
                        nsi_cc[v] += 2 * weight_p * node_weights[node_q]


# network =====================================================================

def _higher_order_transitivity4(int N, np.ndarray[short, ndim=2] A):
    return _higher_order_transitivity4_fast(N, <short*> np.PyArray_DATA(A))


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
    for i in range(N):
        # If degree is smaller than order - 1, set local cliquishness to 0
        degree_i = degree[i]
        if degree_i >= order - 1:
            # Get neighbors of node i
            index = 0
            for j in range(N):
                if A[i, j] == 1:
                    neighbors[index] = j
                    index += 1
            counter = 0
            # Iterate over possibly existing edges between 3 neighbors of i
            for j in range(degree_i):
                node1 = neighbors[j]
                for k in range(degree_i):
                    node2 = neighbors[k]
                    if A[node1, node2] == 1:
                        for l in range(degree_i):
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
    for i in range(N):
        # If degree is smaller than order - 1, set local cliquishness to 0
        degree_i = degree[i]
        if degree_i >= order - 1:
            # Get neighbors of node i
            index = 0
            for j in range(N):
                if A[i, j] == 1:
                    neighbors[index] = j
                    index += 1
            counter = 0
            # Iterate over possibly existing edges between 4 neighbors of i
            for j in range(degree_i):
                node1 = neighbors[j]
                for k in range(degree_i):
                    node2 = neighbors[k]
                    if A[node1, node2] == 1:
                        for l in range(degree_i):
                            node3 = neighbors[l]
                            if A[node1, node3] == 1 and A[node2, node3] == 1:
                                for m in range(degree_i):
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
    for l in range(N):
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
    #for qi in range(queue_len):
        i = queue[qi]
        if i == -1:
            # this should never happen ...
            print("Opps: %d,%d,%d\n" % qi, queue_len, i)
            break
        next_d = distances_to_j[i] + 1
        #iterate through all neighbors l of i
        oi = offsets[i]
        for l_index in range(oi, oi+k[i]):
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
    for ql in range(queue_len-1, -1, -1):
        l = queue[ql]
        if l == -1:
            print("Opps: %d,%d,%d\n" % ql, queue_len, l)
            break
        if l == j:
            # set betweenness and excess to zero
            betweenness_to_j[l] = excess_to_j[l] = 0
        else:
            # otherwise, iterate through all predecessors i of l:
            base_factor = w[l] / multiplicity_to_j[l]
            ol = offsets[l]
            for fi in range(ol, ol+n_predecessors[l]):
                # add betweenness to predecessor
                i = flat_predecessors[fi]
                betweenness_to_j[i] += betweenness_to_j[l] * base_factor * \
                    multiplicity_to_j[i]


def _newman_betweenness_badly_cython(np.ndarray[INTTYPE_t, ndim=2] adjacency,
    np.ndarray[FLOAT64TYPE_t, ndim=2] T, np.ndarray[FLOAT64TYPE_t, ndim=2] rwb,
    N):

    cdef:
        int i, j, s, t
        double norm, sum, Tis, Tit, Tjs, Tjt

    norm = 2 / (N * (N - 1))

    for i in range(N):
        for s in range(N):
            for t in range(s):
                if (i == s or i == t):
                    rwb[i] += 1
                else:
                    sum = 0
                    Tis = T[i,s]
                    Tit = T[i,t]
                    for j in range(N):
                        if (adjacency[i,j] == 1):
                            Tjs = T(j,s)
                            Tjt = T(j,t)
                            sum += np.abs(Tis - Tit - Tjs + Tjt)
                    rwb[i] += 0.5 * sum;
        rwb[i] *= norm
    return rwb


def _cy_mpi_newman_betweenness(
    np.ndarray[INTTYPE_t, ndim=2] this_A, np.ndarray[FLOATTYPE_t, ndim=2] V,
    int N, int start_i, int end_i):
    """
    This function does the outer loop for a certain range start_i-end_i of
    c's.  it gets the full V matrix but only the needed rows of the A matrix.
    Each parallel job will consist of a call to this function:
    """

    cdef:
        int i_rel, j, s, t, i_abs
        float sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        np.ndarray[FLOATTYPE_t, ndim=1] this_betweenness =\
            np.zeros(this_N, dtype=FLOATTYPE)

    for i_rel in range(this_N):
        # correct i index for V matrix
        i_abs = i_rel + start_i
        for j in range(N):
             if this_A[i_rel, j]:
                sum_j = 0.0
                for s in range(N):
                    if i_abs != s:
                        Vis_minus_Vjs = V[i_abs, s] - V[j, s]
                        sum_s = 0.0
                        for t in range(s):
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
        int i_rel, j, s, t, i_abs
        float sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        np.ndarray[FLOATTYPE_t, ndim=1] this_betweenness =\
            np.zeros(this_N, dtype=FLOATTYPE)

    for i_rel in range(this_N):
        i_abs = i_rel + start_i
        for j in range(N):
             if this_A[i_rel, j]:
                sum_j = 0.0
                for s in range(N):
                    if this_not_adj_or_equal[i_rel, s]:
                        Vis_minus_Vjs = V[i_abs, s] - V[j, s]
                        sum_s = 0.0
                        for t in range(s):
                            if this_not_adj_or_equal[i_rel, t]:
                                sum_s += w[t] *\
                                    abs(Vis_minus_Vjs - V[i_abs, t] + V[j, t])
                        sum_j += w[s] * sum_s
                this_betweenness[i_rel] += w[j] * sum_j

    return this_betweenness, start_i, end_i


def _do_nsi_clustering_I(
    int n_cands, np.ndarray[INTTYPE_t, ndim=1] cands,
    np.ndarray[INT16TYPE_t, ndim=1] D_cluster,
    np.ndarray[FLOATTYPE_t, ndim=1] w, double d0,
    np.ndarray[INT32TYPE_t, ndim=1] D_firstpos,
    np.ndarray[INT32TYPE_t, ndim=1] D_nextpos, int N, dict dict_D,
    dict dict_Delta):

    cdef:
        int ca, ij, i, j, posh, h, ih, jh
        float wi, wj, wjd0, Delta_inc, wh, Dih, Djh, Dch_wc, Dch_wc_Dih_wi, \
            Dch_wc_Dih_wj, whd0, Dch_wc_whd0

    # loop thru candidates:
    for ca in range(n_cands):
        ij = cands[ca]
        i = int(ij/N)
        j = ij%N
        wi = w[i]
        wj = w[j]
        wc = wi + wj
        wjd0 = wj * d0
        Delta_inc = 0.0
        # loop thru all nbs of i other than j:
        posh = D_firstpos[i]
        while (posh > 0):
            h = D_cluster[posh]
            if (h != j):
                ih = N*i+h
                jh = N*j+h
                wh = w[h]
                Dih = dict_D[ih]
                if (dict_D.has_key(jh)):
                    Djh = dict_D[jh]
                else:
                    Djh = wh * wjd0

                Dch_wc = (Dih + Djh) / wc
                Dch_wc_Dih_wi = Dch_wc - Dih/wi
                Dch_wc_Djh_wj = Dch_wc - Djh/wj

                Delta_inc += (wi * Dch_wc_Dih_wi*Dch_wc_Dih_wi +
                              wj * Dch_wc_Djh_wj*Dch_wc_Djh_wj) / wh

            posh = D_nextpos[posh]

        # loop thru all nbs of j other than i which are not nbs of i:
        posh = D_firstpos[j]
        while (posh > 0):
            h = D_cluster[posh]
            ih = N*i+h
            wh = w[h]
            whd0 = wh * d0
            if (h != i and not dict_D.has_key(ih)):
                jh = N*j+h
                Djh = dict_D[jh]
                Dch_wc = (wi*whd0 + Djh) / wc
                Dch_wc_whd0 = Dch_wc - whd0
                Dch_wc_Djh_wj = Dch_wc - Djh/wj

                Delta_inc += (wi * Dch_wc_whd0*Dch_wc_whd0 +
                              wj * Dch_wc_Djh_wj*Dch_wc_Djh_wj) / wh

            posh = D_nextpos[posh]

        dict_Delta[ij] = float(dict_Delta[ij]) + Delta_inc

    return dict_Delta


def _do_nsi_clustering_II(int a, int b,
    np.ndarray[INT16TYPE_t, ndim=1] D_cluster,
    np.ndarray[FLOATTYPE_t, ndim=1] w, double d0,
    np.ndarray[INT32TYPE_t, ndim=1] D_firstpos,
    np.ndarray[INT32TYPE_t, ndim=1] D_nextpos, int N, dict dict_D,
    dict dict_Delta):

    cdef:
        float wa = w[a], wb = w[b], wc = wa+wb, wad0 = wa*d0, wbd0 = wb*d0
        float wa1, wa1sq, wa1d0, Da1a1, Da1a, Da1b, Da1c, wb1, wb1d0, wb1sq, \
                wa1b1, wc1, Db1b1, Da1b1, Dc1c1_wc1sq, \
                Dc1c1_wc1sq_Da1a1_wa1sq, Dc1c1_wc1sq_Da1a1_wb1sq, \
                Dc1c1_wc1sq_Da1b1_wa1b1, Delta_new, wc2, Da1c2, Db1c2, \
                Dc1c2_wc1, Dc1c2_wc1_Da1c2_wa1, Dc1c2_wc1_Db1c2_wb1, \
                Dc1c2_wc1_wc2_d0, Db1a, Db1b, Db1c, Dc1a_wc1, Dc1b_wc1, \
                Dc1c_wc1, Dc1c_wc1_Da1c_wa1, Dc1a_wc1_Da1a_wa1, \
                Dc1b_wc1_Da1b_wa1, Dc1c_wc1_Db1c_wb1, Dc1a_wc1_Db1a_wb1, \
                Dc1b_wc1_Db1b_wb1
        int N1 = N+1, posa1 = D_firstpos[a] # a meaning c!
        int a1, a1N, a1a1, a1a, a1b, posb1, b1, b1N, posc2, b1c2, b1a, b1b, \
                a1b1key, b1b1


    while (posa1 > 0):
        a1 = D_cluster[posa1]
        a1N = a1*N
        a1a1 = a1*N1
        a1a = a1N+a
        a1b = a1N+b
        wa1 = w[a1]
        wa1sq = wa1*wa1
        wa1d0 = wa1*d0
        if (dict_D.has_key(a1a1)):
            Da1a1 = dict_D[a1a1]
        else:
            Da1a1 = 0.0
        if (dict_D.has_key(a1a)):
            Da1a = dict_D[a1a]
        else:
            Da1a = wa1*wad0
        if (dict_D.has_key(a1b)):
            Da1b = dict_D[a1b]
        else:
            Da1b = wa1*wbd0

        Da1c = Da1a + Da1b
        posb1 = D_firstpos[a1]

        while (posb1 > 0):
            b1 = D_cluster[posb1]
            b1N = b1*N
            b1b1 = b1*N1

            if (a1 < b1):
                a1b1key = a1N+b1
            else:
                a1b1key = b1N+a1
            if (dict_Delta.has_key(a1b1key)):
                wb1 = w[b1]
                wb1d0 = wb1 * d0
                wb1sq = wb1*wb1
                wa1b1 = wa1*wb1
                wc1 = wa1+wb1
                if (dict_D.has_key(b1b1)):
                    Db1b1 = dict_D[b1b1]
                else:
                    Db1b1 = 0.0
                if (dict_D.has_key(a1b1key)):
                    Da1b1 = dict_D[a1b1key]
                else:
                    Da1b1 = wb1 * wa1d0
                Dc1c1_wc1sq = (Da1a1+Db1b1+2.0*Da1b1) / (wc1*wc1)
                if (b1 == a): # a meaning c!
                    Dc1c1_wc1sq_Da1a1_wa1sq = Dc1c1_wc1sq - Da1a1/wa1sq
                    Dc1c1_wc1sq_Db1b1_wb1sq = Dc1c1_wc1sq - Db1b1/wb1sq
                    Dc1c1_wc1sq_Da1b1_wa1b1 = Dc1c1_wc1sq - Da1b1/wa1b1
                    Delta_new = wa1sq * Dc1c1_wc1sq_Da1a1_wa1sq * \
                                Dc1c1_wc1sq_Da1a1_wa1sq + \
                                wb1sq * Dc1c1_wc1sq_Db1b1_wb1sq * \
                                Dc1c1_wc1sq_Db1b1_wb1sq + \
                                2.0 * wa1b1 * Dc1c1_wc1sq_Da1b1_wa1b1 * \
                                Dc1c1_wc1sq_Da1b1_wa1b1
                    # loop thru all nbs c2 of a1 other than b1:
                    posc2 = D_firstpos[a1], c2
                    while (posc2 > 0):
                        c2 = D_cluster[posc2]
                        if (c2 != b1):
                            b1c2 = b1N+c2
                            wc2 = w[c2]
                            Da1c2 = dict_D[a1N+c2]
                            if (dict_D.has_key(b1c2)):
                                Db1c2 = dict_D[b1c2]
                            else:
                                Db1c2 = wc2*wb1d0
                            Dc1c2_wc1 = (Da1c2 + Db1c2) / wc1
                            Dc1c2_wc1_Da1c2_wa1 = Dc1c2_wc1 - Da1c2/wa1
                            Dc1c2_wc1_Db1c2_wb1 = Dc1c2_wc1 - Db1c2/wb1
                            Delta_new += (wa1 * Dc1c2_wc1_Da1c2_wa1 * \
                                          Dc1c2_wc1_Da1c2_wa1 + \
                                          wb1 * Dc1c2_wc1_Db1c2_wb1 * \
                                          Dc1c2_wc1_Db1c2_wb1) / wc2

                        posc2 = D_nextpos[posc2]

                    #  loop thru all nbs of b1 other than a1
                    #  which are not nbs of a1:
                    posc2 = D_firstpos[b1]
                    while (posc2 > 0):
                        c2 = D_cluster[posc2]
                        if (c2 != a1):
                            if not (dict_D.has_key(a1N+c2)):
                                b1c2 = b1N+c2
                                wc2 = w[c2]

                            if (dict_D.has_key(b1c2)):
                                Db1c2 = dict_D[b1c2]
                            else:
                                Db1c2 = wc2*wb1d0
                            Dc1c2_wc1 = (wc2*wa1d0 + Db1c2) / wc1
                            Dc1c2_wc1_wc2_d0 = Dc1c2_wc1 - wc2*d0
                            Dc1c2_wc1_Db1c2_wb1 = Dc1c2_wc1 - Db1c2/wb1
                            Delta_new += (wa1 * Dc1c2_wc1_wc2_d0 * \
                                          Dc1c2_wc1_wc2_d0 + \
                                          wb1 * Dc1c2_wc1_Db1c2_wb1 * \
                                          Dc1c2_wc1_Db1c2_wb1) / wc2

                        posc2 = D_nextpos[posc2]
                    dict_Delta[a1b1key] = Delta_new

                else:
                    b1a = b1N+a
                    b1b = b1N+b
                    if (dict_D.has_key(b1a)):
                        Db1a = dict_D[b1a]
                    else:
                        Db1a = wb1*wad0
                    if (dict_D.has_key(b1b)):
                        Db1b = dict_D[b1b]
                    else:
                        Db1b = wb1*wbd0;
                    Db1c = Db1a + Db1b
                    wc1 = wa1 + wb1
                    Dc1a_wc1 = (Da1a + Db1a) / wc1
                    Dc1b_wc1 = (Da1b + Db1b) / wc1
                    Dc1c_wc1 = (Da1c + Db1c) / wc1
                    Dc1c_wc1_Da1c_wa1 = Dc1c_wc1 - Da1c/wa1
                    Dc1a_wc1_Da1a_wa1 = Dc1a_wc1 - Da1a/wa1
                    Dc1b_wc1_Da1b_wa1 = Dc1b_wc1 - Da1b/wa1
                    Dc1c_wc1_Db1c_wb1 = Dc1c_wc1 - Db1c/wb1
                    Dc1a_wc1_Db1a_wb1 = Dc1a_wc1 - Db1a/wb1
                    Dc1b_wc1_Db1b_wb1 = Dc1b_wc1 - Db1b/wb1
                    Delta_inc = 2 * (Dc1c_wc1_Da1c_wa1*Dc1c_wc1_Da1c_wa1/wc - \
                                     Dc1a_wc1_Da1a_wa1*Dc1a_wc1_Da1a_wa1/wa - \
                                     Dc1b_wc1_Da1b_wa1*Dc1b_wc1_Da1b_wa1/wb)/ \
                                wa1 + \
                                2 * (Dc1c_wc1_Db1c_wb1*Dc1c_wc1_Db1c_wb1/wc - \
                                     Dc1a_wc1_Db1a_wb1*Dc1a_wc1_Db1a_wb1/wa - \
                                     Dc1b_wc1_Db1b_wb1*Dc1b_wc1_Db1b_wb1/wb)/ \
                                wb1
                    dict_Delta[a1b1key] = float(dict_Delta[a1b1key])+Delta_inc

            posb1 = D_nextpos[posb1]

        posa1 = D_nextpos[posa1]

    return dict_Delta


def _do_nsi_hamming_clustering(int n2, int nActiveIndices, float mind0,
    float minwp0, int lastunited, int part1, int part2,
    np.ndarray[FLOATTYPE_t, ndim=2] distances,
    np.ndarray[INTTYPE_t, ndim=1] theActiveIndices,
    np.ndarray[FLOATTYPE_t, ndim=2] linkedWeights,
    np.ndarray[FLOATTYPE_t, ndim=2] weightProducts,
    np.ndarray[FLOATTYPE_t, ndim=2] errors,
    np.ndarray[FLOATTYPE_t, ndim=1] results,
    np.ndarray[INTTYPE_t, ndim=2] mayJoin):

    return _do_nsi_hamming_clustering_fast(n2, nActiveIndices, mind0, minwp0,
        lastunited, part1, part2,
        <float*> np.PyArray_DATA(distances),
        <int*> np.PyArray_DATA(theActiveIndices),
        <float*> np.PyArray_DATA(linkedWeights),
        <float*> np.PyArray_DATA(weightProducts),
        <float*> np.PyArray_DATA(errors),
        <float*> np.PyArray_DATA(results),
        <int*> np.PyArray_DATA(mayJoin))


# grid ========================================================================

def _cy_calculate_angular_distance(
    np.ndarray[FLOAT32TYPE_t, ndim=1] cos_lat,
    np.ndarray[FLOAT32TYPE_t, ndim=1] sin_lat,
    np.ndarray[FLOAT32TYPE_t, ndim=1] cos_lon,
    np.ndarray[FLOAT32TYPE_t, ndim=1] sin_lon,
    np.ndarray[FLOAT32TYPE_t, ndim=2] cosangdist, unsigned int N):

    cdef:
        FLOAT32TYPE_t expr
        unsigned int i,j

    for i in range(N):
        for j in range(i+1):
            expr = sin_lat[i]*sin_lat[j] + cos_lat[i]*cos_lat[j] * \
                (sin_lon[i]*sin_lon[j] + cos_lon[i]*cos_lon[j])

            if expr > 1:
                expr = 1
            elif expr < -1:
                expr = -1

            cosangdist[i, j] = cosangdist[j, i] = expr


def _euclidean_distance(
    np.ndarray[FLOAT32TYPE_t, ndim=1] x, np.ndarray[FLOAT32TYPE_t, ndim=1] y,
    np.ndarray[FLOAT32TYPE_t, ndim=2] distance, unsigned int N):

    cdef:
        unsigned int i,j
        FLOAT32TYPE_t expr

    for i in range(N):
        for j in range(i+1):
            expr = (x[i]-x[j])**2 + (y[i]-y[j])**2
            distance[i, j] = distance[j, i] = expr**(0.5)


# resistive_network ===========================================================

def _vertex_current_flow_betweenness(int N, double Is, double It,
    np.ndarray[FLOAT32TYPE_t, ndim=2] admittance,
    np.ndarray[FLOAT32TYPE_t, ndim=2] R, int i):

    return _vertex_current_flow_betweenness_fast(N, Is, It,
        <float*> np.PyArray_DATA(admittance),
        <float*> np.PyArray_DATA(R), i)

def _edge_current_flow_betweenness(int N, double Is, double It,
    np.ndarray[FLOAT32TYPE_t, ndim=2] admittance,
    np.ndarray[FLOAT32TYPE_t, ndim=2] R,):

    # alloc output
    cdef np.ndarray[FLOAT32TYPE_t, ndim=2, mode='c'] ECFB = \
            np.zeros((N, N), dtype='float32')

    _edge_current_flow_betweenness_fast(N, Is, It,
        <float*> np.PyArray_DATA(admittance),
        <float*> np.PyArray_DATA(R),
        <float*> np.PyArray_DATA(ECFB))

    return ECFB
