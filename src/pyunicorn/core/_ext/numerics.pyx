# This file is part of pyunicorn.
# Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
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

import random

import numpy as np
cimport numpy as cnp
from numpy cimport ndarray, abs
import numpy.random as rd
randint = rd.randint

from ...core._ext.types import NODE, DEGREE, FIELD, DFIELD
from ...core._ext.types cimport \
    ADJ_t, MASK_t, NODE_t, DEGREE_t, WEIGHT_t, DWEIGHT_t, FIELD_t, DFIELD_t

cdef extern from "src_numerics.c":
    double _vertex_current_flow_betweenness_fast(int N, double Is, double It,
        float *admittance, float *R, int i)
    void _edge_current_flow_betweenness_fast(int N, double Is, double It,
        float *admittance, float *R, float *ECFB)


# geo_network =================================================================


# parameters for `_randomly_rewire_geomodel()`
ctypedef bint (*rewire_cond_len)(FIELD_t[:,:], float, int, int, int, int)
ctypedef bint (*rewire_cond_deg)(DEGREE_t[:], int, int, int, int)

cdef:
    # condition C1
    inline bint cond_len_c1(
        FIELD_t[:,:] D, float eps, int s, int t, int k, int l):
        return (
            (abs(D[s,t] - D[k,t]) < eps and abs(D[k,l] - D[s,l]) < eps) or
            (abs(D[s,t] - D[s,l]) < eps and abs(D[k,l] - D[k,t]) < eps))
    # condition C2
    inline bint cond_len_c2(
        FIELD_t[:,:] D, float eps, int s, int t, int k, int l):
        return (
            abs(D[s,t] - D[s,l]) < eps and abs(D[t,s] - D[t,k]) < eps and
            abs(D[k,l] - D[k,t]) < eps and abs(D[l,k] - D[l,s]) < eps)
    # invariance of degree-degree correlations
    inline bint cond_deg_corr(DEGREE_t[:] degree, int s, int t, int k, int l):
        return (degree[s] == degree[k] and degree[t] == degree[l])
    # tautology
    rewire_cond_deg cond_deg_true = NULL


cdef void _randomly_rewire_geomodel(int iterations, float eps,
    ndarray[ADJ_t, ndim=2] A, ndarray[FIELD_t, ndim=2] D, int E,
    ndarray[NODE_t, ndim=2] edges, ndarray[DEGREE_t, ndim=1] degree,
    rewire_cond_len cond_len, rewire_cond_deg cond_deg):

    cdef int i = 0, s, t, k, l, edge1, edge2

    while (i < iterations):
        # Randomly choose 2 edges
        edge1 = np.floor(rd.random() * E)
        edge2 = np.floor(rd.random() * E)
        s, t = edges[edge1,[0,1]]
        k, l = edges[edge2,[0,1]]

        # Proceed only if old links are disjoint
        if ((s != k and s != l and t != k and t != l) and
            # Proceed only if new links do NOT already exist
            (A[s,l] == 0 and A[t,k] == 0) and
            # Proceed only if link conditions are fulfilled
            (cond_deg is NULL or cond_deg(degree, s, t, k, l)) and
            cond_len(D, eps, s, t, k, l)):

                # Now rewire the links symmetrically & increment i
                A[s,t] = A[t,s] = 0
                A[k,l] = A[l,k] = 0
                A[s,l] = A[l,s] = 1
                A[t,k] = A[k,t] = 1
                edges[edge1,[0,1]] = s, l
                edges[edge2,[0,1]] = k, t

                i+=1


def _randomly_rewire_geomodel_I(int iterations, float eps,
    ndarray[ADJ_t, ndim=2] A, ndarray[FIELD_t, ndim=2] D, int E,
    ndarray[NODE_t, ndim=2] edges):
    cdef:
        ndarray[DEGREE_t, ndim=1] null = np.array([], dtype=DEGREE)
    _randomly_rewire_geomodel(iterations, eps, A, D, E, edges, null,
                              cond_len_c1, cond_deg_true)

def _randomly_rewire_geomodel_II(int iterations, float eps,
    ndarray[ADJ_t, ndim=2] A, ndarray[FIELD_t, ndim=2] D, int E,
    ndarray[NODE_t, ndim=2] edges):
    cdef:
        ndarray[DEGREE_t, ndim=1] null = np.array([], dtype=DEGREE)
    _randomly_rewire_geomodel(iterations, eps, A, D, E, edges, null,
                              cond_len_c2, cond_deg_true)

def _randomly_rewire_geomodel_III(int iterations, float eps,
    ndarray[ADJ_t, ndim=2] A, ndarray[FIELD_t, ndim=2] D, int E,
    ndarray[NODE_t, ndim=2] edges, ndarray[DEGREE_t, ndim=1] degree):
    _randomly_rewire_geomodel(iterations, eps, A, D, E, edges, degree,
                              cond_len_c2, cond_deg_corr)


# interacting_networks ========================================================


cdef void overwriteAdjacency(
    ndarray[ADJ_t, ndim=2] A, ndarray[ADJ_t, ndim=2] cross_A,
    ndarray[NODE_t, ndim=1] nodes1, ndarray[NODE_t, ndim=1] nodes2,
    int m, int n):
    """
    Overwrite the adjacency matrix of the full interacting network with the
    randomly rewired cross edges of the two considered subnetworks.
    """
    cdef:
        int i, j
        NODE_t n1, n2

    for i in range(m):
        for j in range(n):
            n1, n2 = nodes1[i], nodes2[j]
            A[n1, n2] = A[n2, n1] = cross_A[i, j]


def _randomlySetCrossLinks(
    ndarray[ADJ_t, ndim=2] A, ndarray[ADJ_t, ndim=2] cross_A,
    int number_cross_links,
    ndarray[NODE_t, ndim=1] nodes1, ndarray[NODE_t, ndim=1] nodes2,
    int m, int n):
    """
    >>> A = np.eye(2, dtype=np.int)
    >>> _randomlySetCrossLinks(A, np.array([[0]]), 1,
    ...                        np.array([0]), np.array([1]), 1, 1)
    >>> np.all(A == np.ones(2))
    True
    """
    cdef:
        int i, j

    # create random cross links
    for _ in range(number_cross_links):
        while True:
            i, j = randint(m), randint(n)
            if not cross_A[i, j]:
                break
        cross_A[i, j] = 1
    overwriteAdjacency(A, cross_A, nodes1, nodes2, m, n)


def _randomlyRewireCrossLinks(
    ndarray[ADJ_t, ndim=2] A,
    ndarray[ADJ_t, ndim=2] cross_A,
    ndarray[NODE_t, ndim=2] cross_links,
    ndarray[NODE_t, ndim=1] nodes1,
    ndarray[NODE_t, ndim=1] nodes2,
    int number_cross_links, int number_swaps):

    cdef:
        int m = int(len(nodes1)), n = int(len(nodes2))
        NODE_t e1, e2, a, b, c, d

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
    ndarray[ADJ_t, ndim=2] A,
    ndarray[NODE_t, ndim=1] nodes1, ndarray[NODE_t, ndim=1] nodes2):

    cdef:
        int m = int(len(nodes1)), n = int(len(nodes2))
        int i, j, k
        NODE_t n1, n2, n3
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
    ndarray[ADJ_t, ndim=2] A,
    ndarray[NODE_t, ndim=1] nodes1,
    ndarray[NODE_t, ndim=1] nodes2,
    ndarray[DWEIGHT_t, ndim=1] node_weights):

    cdef:
        int m = int(len(nodes1)), n = int(len(nodes2))
        int v, p, q
        NODE_t node_v, node_p, node_q
        DWEIGHT_t weight_v, weight_p, ppv, pqv, T1 = 0, T2 = 0

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
    ndarray[ADJ_t, ndim=2] A,
    ndarray[DFIELD_t, ndim=1] norm,
    ndarray[NODE_t, ndim=1] nodes1, ndarray[NODE_t, ndim=1] nodes2,
    ndarray[DFIELD_t, ndim=1] cross_clustering):

    cdef:
        int m = int(len(nodes1)), n = int(len(nodes2))
        int i, j, k
        NODE_t n1, n2, n3
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
    ndarray[ADJ_t, ndim=2] A,
    ndarray[DFIELD_t, ndim=1] nsi_cc,
    ndarray[NODE_t, ndim=1] nodes1, ndarray[NODE_t, ndim=1] nodes2,
    ndarray[DWEIGHT_t, ndim=1] node_weights):

    cdef:
        int m = int(len(nodes1)), n = int(len(nodes2))
        int v, p, q
        NODE_t node_v, node_p, node_q
        DWEIGHT_t weight_p

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
                        nsi_cc[v] += <WEIGHT_t>2 * weight_p * node_weights[node_q]


# network =====================================================================


def _local_cliquishness_4thorder(
    int N, ndarray[ADJ_t, ndim=2] A, ndarray[DEGREE_t, ndim=1] degree):

    cdef:
        int i, j, k, l
        int index
        NODE_t node1, node2, node3, degree_i, order = 4
        long counter
        ndarray[NODE_t, ndim=1] neighbors = np.zeros(N, dtype=NODE)
        ndarray[DFIELD_t, ndim=1] local_cliquishness = \
            np.zeros(N, dtype=DFIELD)

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
            local_cliquishness[i] = counter /\
                (degree_i * (degree_i - 1) * (degree_i - 2))
    return local_cliquishness


def _local_cliquishness_5thorder(
    int N, ndarray[ADJ_t, ndim=2] A, ndarray[DEGREE_t, ndim=1] degree):

    cdef:
        int i, index
        NODE_t j, k, l, m, node1, node2, node3, node4, degree_i, order = 5
        long counter
        ndarray[NODE_t, ndim=1] neighbors = np.zeros(N, dtype=NODE)
        ndarray[DFIELD_t, ndim=1] local_cliquishness = \
            np.zeros(N, dtype=DFIELD)

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
            local_cliquishness[i] = counter /\
                (degree_i * (degree_i - 1) * (degree_i - 2) * (degree_i -3))
    return local_cliquishness


def _nsi_betweenness(
    int N, ndarray[DWEIGHT_t, ndim=1] w,
    ndarray[DEGREE_t, ndim=1] k,
    ndarray[NODE_t, ndim=1] flat_neighbors,
    ndarray[MASK_t, ndim=1] is_source,
    ndarray[NODE_t, ndim=1] targets):
    """
    Performs Newman's algorithm. [Newman2001]_
    """

    cdef:
        long int E = len(flat_neighbors)
        int j, qi, oi, queue_len, l_index, ql
        NODE_t l, i, next_d, dl, ol, fi
        DFIELD_t base_factor
        ndarray[NODE_t, ndim=1] offsets = np.zeros(N, dtype=NODE)
        ndarray[NODE_t, ndim=1] distances_to_j = np.ones(N, dtype=NODE)
        ndarray[NODE_t, ndim=1] n_predecessors = np.zeros(N, dtype=NODE)
        ndarray[NODE_t, ndim=1] flat_predecessors = np.zeros(E, dtype=NODE)
        ndarray[NODE_t, ndim=1] queue = np.zeros(N, dtype=NODE)
        ndarray[DFIELD_t, ndim=1] multiplicity_to_j = np.zeros(N, dtype=DFIELD)
        ndarray[DFIELD_t, ndim=1] betweenness_to_j = np.zeros(N, dtype=DFIELD)
        ndarray[DFIELD_t, ndim=1] excess_to_j = np.zeros(N, dtype=DFIELD)
        ndarray[DFIELD_t, ndim=1] betweenness_times_w = np.zeros(N, dtype=DFIELD)

    # init node offsets
    # NOTE: We don't use k.cumsum() since that uses too much memory!
    for i in range(1, N):
        offsets[i] = offsets[i-1] + k[i-1]

    for j in targets:
        # init distances to j and queue of nodes by distance from j
        distances_to_j.fill(2 * N)
        n_predecessors.fill(0)
        flat_predecessors.fill(0)
        queue.fill(0)
        multiplicity_to_j.fill(0)

        # init contribution of paths ending in j to the betweenness of l
        for l in range(N):
            excess_to_j[l] = betweenness_to_j[l] = is_source[l] * w[l]

        distances_to_j[j] = 0
        queue[0] = j
        queue_len = 1
        multiplicity_to_j[j] = w[j]

        # process the queue forward and grow it on the way: (this is the
        # standard breadth-first search giving all the shortest paths to j)
        qi = 0
        while qi < queue_len:
            i = queue[qi]
            if i == -1:
                # this should never happen ...
                print("Opps: %d,%d,%d\n" % qi, queue_len, i)
                break
            next_d = distances_to_j[i] + 1
            # iterate through all neighbors l of i
            oi = offsets[i]
            for l_index in range(oi, oi+k[i]):
                # if on a shortest j-l-path, register i as predecessor of l
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

        # process the queue again backward: (this is Newman's 2nd part where the
        # contribution of paths ending in j to the betweenness of all nodes is
        # computed recursively by traversing the shortest paths backwards)
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

        betweenness_times_w += w[j] * (betweenness_to_j - excess_to_j)
    return betweenness_times_w


def _mpi_newman_betweenness(
    ndarray[ADJ_t, ndim=2] this_A, ndarray[DFIELD_t, ndim=2] V,
    int N, int start_i, int end_i):
    """
    This function does the outer loop for a certain range start_i-end_i of
    c's.  it gets the full V matrix but only the needed rows of the A matrix.
    Each parallel job will consist of a call to this function:
    """

    cdef:
        int i_rel, j, s, t, i_abs
        double sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        ndarray[DFIELD_t, ndim=1] this_betweenness = \
            np.zeros(this_N, dtype=DFIELD)

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


def _mpi_nsi_newman_betweenness(
    ndarray[ADJ_t, ndim=2] this_A,
    ndarray[DFIELD_t, ndim=2] V, int N, ndarray[DWEIGHT_t, ndim=1] w,
    ndarray[MASK_t, ndim=2] this_not_adj_or_equal, int start_i, int end_i):

    cdef:
        int i_rel, j, s, t, i_abs
        double sum_s, sum_j, Vis_minus_Vjs

        int this_N = end_i - start_i
        ndarray[DFIELD_t, ndim=1] this_betweenness =\
            np.zeros(this_N, dtype=DFIELD)

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


# grid ========================================================================


def _calculate_angular_distance(
    ndarray[FIELD_t, ndim=1] cos_lat,
    ndarray[FIELD_t, ndim=1] sin_lat,
    ndarray[FIELD_t, ndim=1] cos_lon,
    ndarray[FIELD_t, ndim=1] sin_lon,
    ndarray[FIELD_t, ndim=2] cosangdist, int N):

    cdef:
        FIELD_t expr
        int i,j

    for i in range(N):
        for j in range(i+1):
            expr = sin_lat[i]*sin_lat[j] + cos_lat[i]*cos_lat[j] * \
                (sin_lon[i]*sin_lon[j] + cos_lon[i]*cos_lon[j])

            if expr > 1:
                expr = 1
            elif expr < -1:
                expr = -1

            cosangdist[i, j] = cosangdist[j, i] = expr


def _calculate_euclidean_distance(
    ndarray[FIELD_t, ndim=2] x,
    ndarray[FIELD_t, ndim=2] distance,
    int N_dim, int N_nodes):

    cdef:
        int i,j,k
        FIELD_t expr

    for i in range(N_nodes):
        for j in range(i+1):
            expr = 0
            for k in range(N_dim):
                expr += (x[k, i]-x[k, j])**2
            distance[i, j] = distance[j, i] = expr**(<FIELD_t> 0.5)


# resistive_network ===========================================================


def _vertex_current_flow_betweenness(int N, double Is, double It,
    ndarray[FIELD_t, ndim=2] admittance, ndarray[FIELD_t, ndim=2] R,
    int i):

    return _vertex_current_flow_betweenness_fast(N, Is, It,
        <FIELD_t*> cnp.PyArray_DATA(admittance),
        <FIELD_t*> cnp.PyArray_DATA(R), i)


def _edge_current_flow_betweenness(int N, double Is, double It,
    ndarray[FIELD_t, ndim=2] admittance,
    ndarray[FIELD_t, ndim=2] R,):

    # alloc output
    cdef ndarray[FIELD_t, ndim=2, mode='c'] ECFB = \
            np.zeros((N, N), dtype=FIELD)

    _edge_current_flow_betweenness_fast(N, Is, It,
        <FIELD_t*> cnp.PyArray_DATA(admittance),
        <FIELD_t*> cnp.PyArray_DATA(R),
        <FIELD_t*> cnp.PyArray_DATA(ECFB))

    return ECFB
