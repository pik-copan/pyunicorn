#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

"""
Simple tests for the core Network class.
"""

from functools import partial
from itertools import islice, product, repeat
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.sparse as sp

from pyunicorn import Network


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------

# turn off for weave compilation & error detection
parallel = False


def compareResults(desired, actual, rev_perm=None):
    assert type(desired) is type(actual)
    if isinstance(desired, dict):
        _ = [compareResults(*rs, rev_perm=rev_perm)
             for rs in zip(desired.values(), actual.values())]
    else:
        if sp.issparse(desired):
            desired, actual = desired.A, actual.A
        if isinstance(desired, np.ndarray):
            actual = actual[rev_perm]
            if len(actual.shape) == 2:
                actual = actual[:, rev_perm]
        assert np.allclose(desired, actual, atol=1e-06)


def compareMeasures(orig, pnets, rev_perms, tasks):
    for (measure, i) in tasks:
        method, args = (measure, ()) if isinstance(measure, str) else measure
        compareResults(getattr(orig, method)(*args),
                       getattr(pnets[i], method)(*args), rev_perms[i])


def comparePermutations(net, permutations, measures):
    pnets, rev_perms = zip(
        *((net.permuted_copy(p), p.argsort()) for p in
          map(np.random.permutation, repeat(net.N, permutations))))
    tasks = list(product(measures, range(permutations)))
    if not parallel:
        compareMeasures(net, pnets, rev_perms, tasks)
    else:
        pool, cores = Pool(), cpu_count()
        pool.map(partial(compareMeasures, net, pnets, rev_perms),
                 (list(islice(tasks, c, None, cores)) for c in range(cores)))
        pool.close()
        pool.join()


def compareNSI(net, nsi_measures):
    net_copies = [net.splitted_copy(node=i) for i in range(net.N)]
    for nsi_measure in nsi_measures:
        if isinstance(nsi_measure, tuple):
            kwargs = nsi_measure[1]
            nsi_measure = nsi_measure[0]
        else:
            kwargs = {}
        print(nsi_measure)
        for i, netc in enumerate(net_copies):
            # test for invariance of old nodes
            assert np.allclose(getattr(netc, nsi_measure)(**kwargs)[0:net.N],
                               getattr(net, nsi_measure)(**kwargs))
            # test for invariance of origianl and new splitted node
            assert np.allclose(getattr(netc, nsi_measure)(**kwargs)[i],
                               getattr(netc, nsi_measure)(**kwargs)[-1])


# -----------------------------------------------------------------------------
# stability
# -----------------------------------------------------------------------------

def testIntOverflow():
    """
    Avoid integer overflow in scipy.sparse representation.
    """
    for n in [10, 200, 2000, 33000]:
        adj = sp.lil_matrix((n, n), dtype=np.int8)
        adj[0, 1:] = 1
        deg = Network(adjacency=adj).degree()
        assert (deg.min(), deg.max()) == (0, n - 1)


# -----------------------------------------------------------------------------
# consistency
# -----------------------------------------------------------------------------

def testPermutations():
    """
    Permutation invariance of topological information.
    """
    comparePermutations(
        Network.SmallTestNetwork(), 3, [
            "degree", "indegree", "outdegree", "nsi_degree", "nsi_indegree",
            "nsi_outdegree", "nsi_average_neighbors_degree",
            "nsi_max_neighbors_degree", "undirected_adjacency", "laplacian",
            "nsi_laplacian", "local_clustering", "global_clustering",
            "transitivity", ("higher_order_transitivity", [4]),
            ("local_cliquishness", [4]), ("local_cliquishness", [5]),
            "nsi_twinness", "assortativity", "nsi_local_clustering",
            "nsi_global_clustering", "nsi_transitivity",
            "nsi_local_soffer_clustering", "path_lengths",
            "average_path_length", "nsi_average_path_length", "diameter",
            "matching_index", "link_betweenness", "betweenness",
            "eigenvector_centrality", "nsi_eigenvector_centrality", "pagerank",
            "closeness", "nsi_closeness", "nsi_harmonic_closeness",
            "nsi_exponential_closeness", "arenas_betweenness",
            "nsi_arenas_betweenness", "newman_betweenness",
            "nsi_newman_betweenness", "global_efficiency",
            "nsi_global_efficiency", "distance_based_measures",
            "local_vulnerability", "coreness", "msf_synchronizability",
            "spreading", "nsi_spreading"
        ])


def testNSI():
    """
    Consistency of nsi measures with splitted network copies
    """
    dnw = Network.SmallDirectedTestNetwork()
    nw = Network.SmallTestNetwork()

    nsi_measures = ["nsi_degree", "nsi_indegree", "nsi_outdegree",
                    "nsi_closeness", "nsi_harmonic_closeness",
                    "nsi_exponential_closeness", "nsi_arenas_betweenness",
                    "nsi_spreading",
                    "nsi_local_cyclemotif_clustering",
                    "nsi_local_midmotif_clustering",
                    "nsi_local_inmotif_clustering",
                    "nsi_local_outmotif_clustering",
                    ("nsi_degree", {"key": "link_weights"}),
                    ("nsi_indegree", {"key": "link_weights"}),
                    ("nsi_outdegree", {"key": "link_weights"}),
                    ("nsi_local_cyclemotif_clustering",
                     {"key": "link_weights"}),
                    ("nsi_local_midmotif_clustering",
                     {"key": "link_weights"}),
                    ("nsi_local_inmotif_clustering",
                     {"key": "link_weights"}),
                    ("nsi_local_outmotif_clustering",
                     {"key": "link_weights"})]

    nsi_undirected_measures = ["nsi_local_clustering",
                               "nsi_average_neighbors_degree",
                               "nsi_max_neighbors_degree",
                               "nsi_eigenvector_centrality",
                               "nsi_local_clustering",
                               "nsi_local_soffer_clustering",
                               "nsi_newman_betweenness"]

    compareNSI(dnw, nsi_measures)
    compareNSI(nw, nsi_measures + nsi_undirected_measures)
