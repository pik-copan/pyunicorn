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

"""
Simple tests for the Network class.
"""

from functools import partial
from itertools import islice, product, repeat
from multiprocessing import get_context, cpu_count

import pytest
import numpy as np
import scipy.sparse as sp

from pyunicorn import Network
from pyunicorn.core.network import r


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------


def compare_results(desired, actual, rev_perm=None):
    assert type(desired) is type(actual)
    if isinstance(desired, dict):
        _ = [compare_results(*rs, rev_perm=rev_perm)
             for rs in zip(desired.values(), actual.values())]
    else:
        if sp.issparse(desired):
            desired, actual = desired.toarray(), actual.toarray()
        if isinstance(desired, np.ndarray):
            actual = actual[rev_perm]
            if len(actual.shape) == 2:
                actual = actual[:, rev_perm]
        assert np.allclose(desired, actual, atol=1e-06)


def compare_measures(orig, pnets, rev_perms, tasks):
    for (measure, i) in tasks:
        method, args = (measure, ()) if isinstance(measure, str) else measure
        compare_results(getattr(orig, method)(*args),
                        getattr(pnets[i], method)(*args), rev_perms[i])


def compare_permutations(net, permutations, measures):
    pnets, rev_perms = zip(
        *((net.permuted_copy(p), p.argsort()) for p in
          map(np.random.permutation, repeat(net.N, permutations))))
    tasks = list(product(measures, range(permutations)))
    cores = cpu_count()
    with get_context("spawn").Pool() as pool:
        pool.map(partial(compare_measures, net, pnets, rev_perms),
                 (list(islice(tasks, c, None, cores)) for c in range(cores)))
        pool.close()
        pool.join()


def compare_nsi(net, nsi_measures):
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


def test_int_overflow():
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


def test_permutations():
    """
    Permutation invariance of topological information.
    """
    compare_permutations(
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


def test_nsi():
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

    compare_nsi(dnw, nsi_measures)
    compare_nsi(nw, nsi_measures + nsi_undirected_measures)


# -----------------------------------------------------------------------------
# test doctest helpers
# -----------------------------------------------------------------------------


def test_r():
    arr = np.random.rand(3, 3)
    assert r(arr).dtype == np.float64


def test_r_type_error():
    arr = np.array(['one', 'two', 'three'])
    with pytest.raises(TypeError, match='obj is of unsupported dtype kind.'):
        r(arr)


# -----------------------------------------------------------------------------
# Class member tests with TestNetwork
# -----------------------------------------------------------------------------


def test_init():
    Network(adjacency=[[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1],
                       [0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 0],
                       [0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]])
    assert True


def test_str(capsys):
    print(Network.SmallTestNetwork())
    out = capsys.readouterr()[0]
    out_ref = "Network: undirected, 6 nodes, 7 links, link density 0.467.\n"
    assert out == out_ref


def test_len():
    assert np.allclose(len(Network.SmallTestNetwork()), 6)


def test_undirected_copy(capsys):
    net = Network(adjacency=[[0, 1], [0, 0]], directed=True)

    print(net)
    out1 = capsys.readouterr()[0]
    out1_ref = "Network: directed, 2 nodes, 1 links, link density 0.500.\n"
    assert out1 == out1_ref

    print(net.undirected_copy())
    out2 = capsys.readouterr()[0]
    out2_ref = "Network: undirected, 2 nodes, 1 links, link density 1.000.\n"
    assert out2 == out2_ref


def test_splitted_copy(capsys):
    net = Network.SmallTestNetwork()
    net2 = net.splitted_copy(node=5, proportion=0.2)
    print(net2)
    out = capsys.readouterr()[0]
    out_ref = "Network: undirected, 7 nodes, 9 links, link density 0.429.\n"
    assert out == out_ref

    nw1_ref = [1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
    nw2_ref = [1.5, 1.7, 1.9, 2.1, 2.3, 2., 0.5]

    assert np.allclose(net.node_weights, nw1_ref)
    assert np.allclose(net2.node_weights, nw2_ref)


def test_adjacency():
    adj_ref = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0],
                        [0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    assert np.array_equal(Network.SmallTestNetwork().adjacency, adj_ref)


def test_set_adjacency(capsys):
    net = Network.SmallTestNetwork()
    net.adjacency = [[0, 1], [1, 0]]
    print(net)
    out = capsys.readouterr()[0]
    out_ref = "Network: undirected, 2 nodes, 1 links, link density 1.000.\n"
    assert out == out_ref


def test_set_node_weights():
    net = Network.SmallTestNetwork()
    nw_ref = [1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
    assert np.allclose(net.node_weights, nw_ref)

    net.node_weights = [1, 1, 1, 1, 1, 1]
    nw_ref = [1., 1., 1., 1., 1., 1.]
    assert np.allclose(net.node_weights, nw_ref)


def test_ErdosRenyi(capsys):
    print(Network.Model("ErdosRenyi", n_nodes=10, n_links=18))
    out = capsys.readouterr()[0]
    out_ref = "Generating Erdos-Renyi random graph with 10 " + \
              "nodes and 18 links...\n" + \
              "Network: undirected, 10 nodes, 18 links, link density 0.400.\n"
    assert out == out_ref


def test_BarabasiAlbert_igraph():
    net = Network.Model("BarabasiAlbert_igraph", n_nodes=100, n_links_each=1)
    assert np.allclose(net.link_density, 0.02)


def test_ConfigurationModel():
    net = Network.Model("Configuration", degree=[3 for _ in range(0, 1000)])
    assert int(round(net.degree().mean())) == 3


def test_WattsStrogatz():
    net = Network.Model("WattsStrogatz", N=100, k=2, p=0.1)
    assert int(round(net.degree().mean())) == 4


def test_GrowWeights():
    n_nodes = 10
    weights = Network.GrowWeights(n_nodes=n_nodes)
    assert len(weights) == n_nodes


def test_randomly_rewire(capsys):
    net = Network.SmallTestNetwork()
    net.randomly_rewire(iterations=10)
    out = capsys.readouterr()[0]
    out_ref = "Randomly rewiring the network,preserving " + \
              "the degree sequence...\n"
    assert out == out_ref
    print(net)
    out = capsys.readouterr()[0]
    out_ref = "Network: undirected, 6 nodes, 7 links, link density 0.467.\n"
    assert out == out_ref


def test_edge_list():
    edges = Network.SmallTestNetwork().edge_list()[:8]
    edges_ref = [[0, 3], [0, 4], [0, 5], [1, 2],
                 [1, 3], [1, 4], [2, 1], [2, 4]]
    assert np.array_equal(edges, edges_ref)


def test_undirected_adjacency():
    net = Network(adjacency=[[0, 1], [0, 0]], directed=True)
    adj_ref = [[0, 1], [1, 0]]
    assert np.array_equal(net.undirected_adjacency().toarray(), adj_ref)


def test_laplacian():
    lap_ref = np.array([[3, 0, 0, -1, -1, -1], [0, 3, -1, -1, -1, 0],
                        [0, -1, 2, 0, -1, 0], [-1, -1, 0, 2, 0, 0],
                        [-1, -1, -1, 0, 3, 0], [-1, 0, 0, 0, 0, 1]])
    assert np.allclose(Network.SmallTestNetwork().laplacian(), lap_ref)


def test_laplacian_value_error():
    with pytest.raises(ValueError, match='direction must be "in" or "out".'):
        Network.SmallDirectedTestNetwork().laplacian(direction='some_other')


def test_nsi_laplacian():
    nsi_lap_ref = np.array([[6.9, 0., 0., -2.1, -2.3, -2.5],
                            [0., 6.3, -1.9, -2.1, -2.3, 0.],
                            [0., -1.7, 4., 0., -2.3, 0.],
                            [-1.5, -1.7, 0., 3.2, 0., 0.],
                            [-1.5, -1.7, -1.9, 0., 5.1, 0.],
                            [-1.5, 0., 0., 0., 0., 1.5]])
    assert np.allclose(Network.SmallTestNetwork().nsi_laplacian(), nsi_lap_ref)


def test_degree():
    deg = Network.SmallTestNetwork().degree()
    deg_ref = np.array([3, 3, 2, 2, 3, 1])
    assert (deg == deg_ref).all()


def test_indegree():
    deg = Network.SmallDirectedTestNetwork().indegree()
    deg_ref = np.array([2, 2, 2, 1, 1, 0])
    assert (deg == deg_ref).all()


def test_outdegree():
    deg = Network.SmallDirectedTestNetwork().outdegree()
    deg_ref = np.array([2, 2, 0, 1, 2, 1])
    assert (deg == deg_ref).all()


def test_bildegree():
    deg = Network.SmallDirectedTestNetwork().bildegree()
    deg_ref = np.array([0, 0, 0, 0, 0, 0], dtype=np.int16)
    assert (deg == deg_ref).all()

    net = Network.SmallTestNetwork()
    assert (net.bildegree() == net.degree()).all()


def test_nsi_degree():
    net = Network.SmallTestNetwork()

    deg_ref = np.array([8.4, 8., 5.9, 5.3, 7.4, 4.])
    assert np.allclose(net.nsi_degree(), deg_ref)

    deg_ref = np.array([8.4, 8., 5.9, 5.3, 7.4, 4., 4.])
    assert np.allclose(net.splitted_copy().nsi_degree(), deg_ref)

    deg_ref = np.array([3.2, 3., 1.95, 1.65, 2.7, 1.])
    assert np.allclose(net.nsi_degree(typical_weight=2.0), deg_ref)

    deg_ref = np.array([3.2, 3., 1.95, 1.65, 2.7, 1., 1.])
    assert np.allclose(net.splitted_copy().nsi_degree(typical_weight=2.0),
                       deg_ref)


@pytest.mark.parametrize("tw, exp, exp_split", [
    (None,
     np.array([6.3, 5.3, 5.9, 3.6, 4., 2.5]),
     np.array([6.3, 5.3, 5.9, 3.6, 4., 2.5, 2.5])),
    (2.,
     np.array([2.15, 1.65, 1.95, 0.8, 1., 0.25]),
     np.array([2.15, 1.65, 1.95, 0.8, 1., 0.25, 0.25]))
    ])
def test_nsi_indegree(tw, exp, exp_split):
    net = Network.SmallDirectedTestNetwork()
    assert np.allclose(net.nsi_indegree(typical_weight=tw), exp)
    assert np.allclose(
        net.splitted_copy().nsi_indegree(typical_weight=tw), exp_split)


@pytest.mark.parametrize("tw, exp, exp_split", [
    (None,
     np.array([5.3, 5.9, 1.9, 3.8, 5.7, 4.]),
     np.array([5.3, 5.9, 1.9, 3.8, 5.7, 4., 4.])),
    (2.,
     np.array([1.65, 1.95, -0.05, 0.9, 1.85, 1.]),
     np.array([1.65, 1.95, -0.05, 0.9, 1.85, 1., 1.]))
    ])
def test_nsi_outdegree(tw, exp, exp_split):
    net = Network.SmallDirectedTestNetwork()
    assert np.allclose(net.nsi_outdegree(typical_weight=tw), exp)
    assert np.allclose(net.splitted_copy().nsi_outdegree(typical_weight=tw),
                       exp_split)


@pytest.mark.parametrize("tw, exp, exp_split", [
    (None,
     np.array([1.5, 1.7, 1.9, 2.1, 2.3, 2.5]),
     np.array([1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.5])),
    (2.,
     np.array([-0.25, -0.15, -0.05,  0.05,  0.15,  0.25]),
     np.array([-0.25, -0.15, -0.05,  0.05,  0.15,  0.25,  0.25]))
    ])
def test_nsi_bildegree(tw, exp, exp_split):
    net = Network.SmallDirectedTestNetwork()
    assert np.allclose(net.nsi_bildegree(typical_weight=tw), exp)
    assert np.allclose(net.splitted_copy().nsi_bildegree(typical_weight=tw),
                       exp_split)


def test_degree_distribution():
    dist = Network.SmallTestNetwork().degree_distribution()
    dist_ref = np.array([0.16666667, 0.33333333, 0.5])
    assert np.allclose(dist, dist_ref)


def test_indegree_distribution():
    dist = Network.SmallTestNetwork().indegree_distribution()
    dist_ref = np.array([0.16666667, 0.33333333, 0.5])
    assert np.allclose(dist, dist_ref)


def test_outdegree_distribution():
    dist = Network.SmallTestNetwork().outdegree_distribution()
    dist_ref = np.array([0.16666667, 0., 0.33333333, 0.5])
    assert np.allclose(dist, dist_ref)


def test_degree_cdf():
    cdf_ref = np.array([1., 0.83333333, 0.5])
    assert np.allclose(Network.SmallTestNetwork().degree_cdf(), cdf_ref)


def test_indegree_cdf():
    cdf_ref = np.array([1., 0.83333333, 0.83333333, 0.5])
    assert np.allclose(Network.SmallTestNetwork().indegree_cdf(), cdf_ref)


def test_outdegree_cdf():
    cdf_ref = np.array([1., 0.83333333, 0.83333333, 0.5])
    assert np.allclose(Network.SmallTestNetwork().outdegree_cdf(), cdf_ref)


def test_nsi_degree_histogram():
    hist = Network.SmallTestNetwork().nsi_degree_histogram()
    hist_ref = (np.array([0.33333333, 0.16666667, 0.5]),
                np.array([0.11785113, 0.16666667, 0.09622504]),
                np.array([4., 5.46666667, 6.93333333]))
    assert np.allclose(hist, hist_ref)


def test_nsi_degree_cumulative_histogram():
    res = Network.SmallTestNetwork().nsi_degree_cumulative_histogram()
    exp = (np.array([1., 0.66666667, 0.5]),
           np.array([4., 5.46666667, 6.93333333]))
    assert np.allclose(res, exp)


def test_average_neighbors_degree():
    res = Network.SmallTestNetwork().average_neighbors_degree()
    exp = np.array([2., 2.33333333, 3., 3., 2.66666667, 3.])
    assert np.allclose(res, exp)


def test_max_neighbors_degree():
    res = Network.SmallTestNetwork().max_neighbors_degree()
    exp = np.array([3, 3, 3, 3, 3, 3])
    assert (res == exp).all()


def test_nsi_average_neighbors_degree():
    net = Network.SmallTestNetwork()

    res = net.nsi_average_neighbors_degree()
    exp = np.array([6.0417, 6.62, 7.0898, 7.0434, 7.3554, 5.65])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_average_neighbors_degree()
    exp = np.array([6.0417, 6.62, 7.0898, 7.0434, 7.3554, 5.65, 5.65])
    assert np.allclose(res, exp)


def test_nsi_max_neighbors_degree():
    res = Network.SmallTestNetwork().nsi_max_neighbors_degree()
    exp = np.array([8.4, 8., 8., 8.4, 8.4, 8.4])
    assert np.allclose(res, exp)


def test_local_clustering():
    res = Network.SmallTestNetwork().local_clustering()
    exp = np.array([0., 0.33333333, 1., 0., 0.33333333, 0.])
    assert np.allclose(res, exp)


def test_global_clustering():
    res = Network.SmallTestNetwork().global_clustering()
    exp = 0.27777777
    assert np.allclose(res, exp)


def test_local_cyclemotif_clustering():
    res = Network.SmallDirectedTestNetwork().local_cyclemotif_clustering()
    exp = np.array([0.25, 0.25, 0., 0., 0.5, 0.])
    assert np.allclose(res, exp)


def test_local_midmotif_clustering():
    res = Network.SmallDirectedTestNetwork().local_midmotif_clustering()
    exp = np.array([0., 0., 0., 1., 0.5, 0.])
    assert np.allclose(res, exp)


def test_local_inmotif_clustering():
    res = Network.SmallDirectedTestNetwork().local_inmotif_clustering()
    exp = np.array([0., 0.5, 0.5, 0., 0., 0.])
    assert np.allclose(res, exp)


def test_local_outmotif_clustering():
    res = Network.SmallDirectedTestNetwork().local_outmotif_clustering()
    exp = np.array([0.5, 0.5, 0., 0., 0., 0.])
    assert np.allclose(res, exp)


@pytest.mark.parametrize("tw, exp, exp_split", [
    (None,
     np.array([0.18448637, 0.20275024, 0.3220339,
               0.32236842, 0.34385965, 0.625]),
     np.array([0.18448637, 0.20275024, 0.3220339,
               0.32236842, 0.34385965, 0.625, 0.20275024])),
    (2.,
     np.array([0.3309814,  0.29011913,  0.05236908,
               -0.0260989,  0.22417582, -0.13636364]),
     np.array([0.3309814,  0.29011913,  0.05236908,
               -0.0260989,  0.22417582, -0.13636364,  0.29011913]))
    ])
def test_nsi_local_cyclemotif_clustering(tw, exp, exp_split):
    net = Network.SmallDirectedTestNetwork()
    assert np.allclose(net.nsi_local_cyclemotif_clustering(typical_weight=tw),
                       exp)
    assert np.allclose(
        net.splitted_copy(node=1).nsi_local_cyclemotif_clustering(
            typical_weight=tw),
        exp_split)


def test_nsi_local_midmotif_clustering():
    net = Network.SmallDirectedTestNetwork()

    res = net.nsi_local_midmotif_clustering()
    exp = np.array([0.45372866, 0.51646946, 1., 1., 0.88815789, 1.])
    assert np.allclose(res, exp)

    res = net.splitted_copy(node=4).local_midmotif_clustering()
    exp = np.array([0., 0., 0., 1., 0.8, 0., 0.8])
    assert np.allclose(res, exp)


def test_nsi_local_inmotif_clustering():
    net = Network.SmallDirectedTestNetwork()

    res = net.nsi_local_inmotif_clustering()
    exp = np.array([0.52884858, 0.66998932, 0.66934789, 0.75694444, 0.755625,
                    1.])
    assert np.allclose(res, exp)

    res = net.splitted_copy(node=1).nsi_local_inmotif_clustering()
    exp = np.array([0.52884858, 0.66998932, 0.66934789, 0.75694444, 0.755625,
                    1., 0.66998932])
    assert np.allclose(res, exp)


def test_nsi_local_outmotif_clustering():
    net = Network.SmallDirectedTestNetwork()

    res = net.nsi_local_outmotif_clustering()
    exp = np.array([0.66998932, 0.66934789, 1., 0.75277008, 0.58387196,
                    0.765625])
    assert np.allclose(res, exp)

    res = net.splitted_copy(node=1).nsi_local_outmotif_clustering()
    exp = np.array([0.66998932, 0.66934789, 1., 0.75277008, 0.58387196,
                    0.765625, 0.66934789])
    assert np.allclose(res, exp)


def test_transitivity():
    res = Network.SmallTestNetwork().transitivity()
    exp = 0.27272727
    assert np.allclose(res, exp)


def test_weighted_local_clustering():
    res = Network.weighted_local_clustering(
        weighted_A=[[0., 0., 0., 0.55, 0.65, 0.75],
                    [0., 0., 0.63, 0.77, 0.91, 0.],
                    [0., 0.63, 0., 0., 1.17, 0.],
                    [0.55, 0.77, 0., 0., 0., 0.],
                    [0.65, 0.91, 1.17, 0., 0., 0.],
                    [0.75, 0., 0., 0., 0., 0.]])
    exp = np.array([0., 0.21487603, 0.35388889, 0., 0.15384615, 0.])
    assert np.allclose(res, exp)


def test_nsi_twinness():
    net = Network.SmallDirectedTestNetwork()

    res = net.nsi_twinness()
    exp = np.array([[0.12931034, 0.45689655, 0., 0.31034483, 0., 0.],
                    [0., 0.15178571, 0.52678571, 0., 0.35714286, 0.],
                    [0., 0., 0.24358974, 0., 0., 0.],
                    [0., 0.33928571, 0., 0.28378378, 0., 0.],
                    [0.32758621, 0., 0.43298969, 0., 0.2371134, 0.],
                    [0.34482759, 0., 0., 0., 0., 0.38461538]])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_twinness()
    exp = np.array([[0.12931034, 0.45689655, 0., 0.31034483, 0., 0., 0.],
                    [0., 0.15178571, 0.52678571, 0., 0.35714286, 0., 0.],
                    [0., 0., 0.24358974, 0., 0., 0., 0.],
                    [0., 0.33928571, 0., 0.28378378, 0., 0., 0.],
                    [0.32758621, 0., 0.43298969, 0., 0.2371134, 0., 0.],
                    [0.34482759, 0., 0., 0., 0., 0.38461538, 0.38461538],
                    [0.34482759, 0., 0., 0., 0., 0.38461538, 0.38461538]])
    assert np.allclose(res, exp)


def test_assortativity():
    res = Network.SmallTestNetwork().assortativity()
    exp = -0.47368421
    assert np.allclose(res, exp)


@pytest.mark.parametrize("tw, exp, exp_split", [
    (None,
     np.array([0.55130385, 0.724375, 1., 0.81844073, 0.80277575, 1.]),
     np.array([0.55130385, 0.724375, 1., 0.81844073, 0.80277575, 1., 1.])),
    (3.,
     np.array([-1.44290123, -0.764, 1., 4.16770186, -0.75324675, 1.]),
     np.array([-1.44290123, -0.764, 1., 4.16770186, -0.75324675, 1., 1.]))
    ])
def test_nsi_local_clustering(tw, exp, exp_split):
    net = Network.SmallTestNetwork()

    assert np.allclose(net.nsi_local_clustering(typical_weight=tw), exp)
    assert np.allclose(
        net.splitted_copy().nsi_local_clustering(typical_weight=tw),
        exp_split)


def test_nsi_global_clustering():
    res = Network.SmallTestNetwork().nsi_global_clustering()
    exp = 0.83529192
    assert np.allclose(res, exp)


def test_nsi_local_soffer_clustering():
    net = Network.SmallTestNetwork()

    res = net.nsi_local_soffer_clustering()
    exp = np.array([0.76650246, 0.87537764, 1., 0.81844073, 0.84685032, 1.])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_local_soffer_clustering()
    exp = np.array([0.76650246, 0.87537764, 1., 0.81844073, 0.84685032, 1.,
                    1.])
    assert np.allclose(res, exp)


def test_path_lengths():
    res = Network.SmallTestNetwork().path_lengths()
    exp = np.array([[0., 2., 2., 1., 1., 1.],
                    [2., 0., 1., 1., 1., 3.],
                    [2., 1., 0., 2., 1., 3.],
                    [1., 1., 2., 0., 2., 2.],
                    [1., 1., 1., 2., 0., 2.],
                    [1., 3., 3., 2., 2., 0.]])
    assert np.allclose(res, exp)


def test_average_path_length():
    res = Network.SmallTestNetwork().average_path_length()
    exp = 1.66666667
    assert np.allclose(res, exp)


def test_nsi_average_path_length():
    net = Network.SmallTestNetwork()

    res = net.nsi_average_path_length()
    exp = 1.60027778
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_average_path_length()
    exp = 1.60027778
    assert np.allclose(res, exp)


def test_diameter():
    res = Network.SmallTestNetwork().diameter()
    exp = 3
    assert np.allclose(res, exp)


def test_matching_index():
    res = Network.SmallTestNetwork().matching_index()
    exp = np.array([[1., 0.5, 0.25, 0., 0., 0.],
                    [0.5, 1., 0.25, 0., 0.2, 0.],
                    [0.25, 0.25, 1., 0.33333333, 0.25, 0.],
                    [0., 0., 0.33333333, 1., 0.66666667, 0.5],
                    [0., 0.2, 0.25, 0.66666667, 1., 0.33333333],
                    [0., 0., 0., 0.5, 0.33333333, 1.]])
    assert np.allclose(res, exp)


def test_link_betweenness():
    res = Network.SmallTestNetwork().link_betweenness()
    exp = np.array([[0., 0., 0., 3.5, 5.5, 5.],
                    [0., 0., 2., 3.5, 2.5, 0.],
                    [0., 2., 0., 0., 3., 0.],
                    [3.5, 3.5, 0., 0., 0., 0.],
                    [5.5, 2.5, 3., 0., 0., 0.],
                    [5., 0., 0., 0., 0., 0.]])
    assert np.allclose(res, exp)


def test_edge_betweenness():
    res = Network.SmallTestNetwork().edge_betweenness()
    exp = np.array([[0., 0., 0., 3.5, 5.5, 5.],
                    [0., 0., 2., 3.5, 2.5, 0.],
                    [0., 2., 0., 0., 3., 0.],
                    [3.5, 3.5, 0., 0., 0., 0.],
                    [5.5, 2.5, 3., 0., 0., 0.],
                    [5., 0., 0., 0., 0., 0.]])
    assert np.allclose(res, exp)


def test_betweenness():
    res = Network.SmallTestNetwork().betweenness()
    exp = np.array([4.5, 1.5, 0., 1., 3., 0.])
    assert np.allclose(res, exp)


def test_interregional_betweenness():
    net = Network.SmallTestNetwork()
    res = net.interregional_betweenness(sources=[2], targets=[3, 5])
    exp = np.array([1., 1., 0., 0., 1., 0.])
    assert np.allclose(res, exp)

    res = net.interregional_betweenness(sources=range(0, 6),
                                        targets=range(0, 6))
    exp = np.array([9., 3., 0., 2., 6., 0.])
    assert np.allclose(res, exp)


def test_nsi_interregional_betweenness():
    res = Network.SmallTestNetwork().nsi_interregional_betweenness(
        sources=[2], targets=[3, 5])
    exp = np.array([3.16666689, 2.34705893, 0., 0., 2.06521743, 0.])
    assert np.allclose(res, exp)


@pytest.mark.parametrize("parallelize", [False, True])
def test_nsi_betweenness(parallelize):
    net = Network.SmallTestNetwork()

    res = net.nsi_betweenness(parallelize=parallelize)
    exp = np.array([29.68541738, 7.7128677, 0., 3.09090906, 9.69960462, 0.])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_betweenness(parallelize=parallelize)
    exp = np.append(exp, [0.])
    assert np.allclose(res, exp)


def test_eigenvector_centrality():
    res = Network.SmallTestNetwork().eigenvector_centrality()
    exp = np.array([0.7895106, 0.97303126, 0.77694188, 0.69405519, 1.,
                    0.31089413])
    assert np.allclose(res, exp)


def test_nsi_eigenvector_centrality():
    net = Network.SmallTestNetwork()

    res = net.nsi_eigenvector_centrality()
    exp = np.array([0.80454492, 1., 0.80931481, 0.61787145, 0.98666885,
                    0.28035747])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_eigenvector_centrality()
    exp = np.array([0.80454492, 1., 0.80931481, 0.61787145, 0.98666885,
                    0.28035747, 0.28035747])
    assert np.allclose(res, exp)


def test_pagerank():
    res = Network.SmallTestNetwork().pagerank()
    exp = np.array([0.21836231, 0.20440819, 0.14090543, 0.14478497, 0.20466978,
                    0.08686932])
    assert np.allclose(res, exp)


def test_closeness():
    res = Network.SmallTestNetwork().closeness()
    exp = np.array([0.71428571, 0.625, 0.55555556, 0.625, 0.71428571,
                    0.45454545])
    assert np.allclose(res, exp)


def test_nsi_closeness():
    net = Network.SmallTestNetwork()

    res = net.nsi_closeness()
    exp = np.array([0.76923077, 0.64864865, 0.58252427, 0.64171123, 0.72289157,
                    0.50847458])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_closeness()
    exp = np.array([0.76923077, 0.64864865, 0.58252427, 0.64171123, 0.72289157,
                    0.50847458, 0.50847458])
    assert np.allclose(res, exp)


def test_nsi_harmonic_closeness():
    net = Network.SmallTestNetwork()

    res = net.nsi_harmonic_closeness()
    exp = np.array([0.85, 0.79861111, 0.71111111, 0.72083333, 0.80833333,
                    0.61666667])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_harmonic_closeness()
    exp = np.array([0.85, 0.79861111, 0.71111111, 0.72083333, 0.80833333,
                    0.61666667, 0.61666667])
    assert np.allclose(res, exp)


def test_nsi_exponential_closeness():
    net = Network.SmallTestNetwork()

    res = net.nsi_exponential_closeness()
    exp = np.array([0.425, 0.390625, 0.346875, 0.36041667, 0.40416667,
                    0.29583333])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_exponential_closeness()
    exp = np.array([0.425, 0.390625, 0.346875, 0.36041667, 0.40416667,
                    0.29583333, 0.29583333])
    assert np.allclose(res, exp)


def test_arenas_betweenness():
    res = Network.SmallTestNetwork().arenas_betweenness()
    exp = np.array([50.18181818, 50.18181818, 33.45454545, 33.45454545,
                    50.18181818, 16.72727273])
    assert np.allclose(res, exp)


def test_nsi_arenas_betweenness():
    net = Network.SmallTestNetwork()

    res = net.nsi_arenas_betweenness()
    exp = np.array([20.58135241, 29.21033898, 27.00747741, 19.5433536,
                    25.28490117, 24.84826305])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_arenas_betweenness()
    exp = np.array([20.58135241, 29.21033898, 27.00747741, 19.5433536,
                    25.28490117, 24.84826305, 24.84826305])
    assert np.allclose(res, exp)

    res = net.nsi_arenas_betweenness(exclude_neighbors=False)
    exp = np.array([44.53512333, 37.40578733, 27.00747741, 21.77360559,
                    31.32557606, 24.84826305])
    assert np.allclose(res, exp)

    res = net.nsi_arenas_betweenness(stopping_mode="twinness")
    exp = np.array([22.61533156, 41.23139296, 38.64105931, 28.61953314,
                    38.58242175, 30.29941829])
    assert np.allclose(res, exp)


def test_newman_betweenness():
    res = Network.SmallTestNetwork().newman_betweenness()
    exp = np.array([4.1818182, 3.41818185, 2.5090909, 3.0181818, 3.60000002,
                    2.])
    assert np.allclose(res, exp)


def test_nsi_newman_betweenness():
    net = Network.SmallTestNetwork()

    res = net.nsi_newman_betweenness()
    exp = np.array([0.40476082, 0., 0.85212808, 3.33573728, 1.36618345, 0.])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_newman_betweenness()
    exp = np.array([0.40476082, 0., 0.85212808, 3.33573728, 1.36618345,
                    0., 0.])
    assert np.allclose(res, exp)

    res = net.nsi_newman_betweenness(add_local_ends=True)
    exp = np.array([131.44476082, 128., 107.64212808, 102.44573728,
                    124.20618345, 80.])
    assert np.allclose(res, exp)

    res = net.splitted_copy().nsi_newman_betweenness(add_local_ends=True)
    exp = np.array([131.44476082, 128., 107.64212808, 102.44573728,
                    124.20618345, 80., 80.])
    assert np.allclose(res, exp)


def test_global_efficiency():
    res = Network.SmallTestNetwork().global_efficiency()
    exp = 0.71111111
    assert np.allclose(res, exp)


def test_nsi_global_efficiency():
    res = Network.SmallTestNetwork().nsi_global_efficiency()
    exp = 0.74152777
    assert np.allclose(res, exp)


def test_local_vulnerability():
    res = Network.SmallTestNetwork().local_vulnerability()
    exp = np.array([0.296875, 0.0625, -0.03125, -0.0078125, 0.09765625,
                    -0.125])
    assert np.allclose(res, exp)


def test_coreness():
    res = Network.SmallTestNetwork().coreness()
    exp = np.array([2, 2, 2, 2, 2, 1])
    assert (res == exp).all()


def test_msf_synchronizability():
    res = Network.SmallTestNetwork().msf_synchronizability()
    exp = 6.77842586
    assert np.allclose(res, exp)
