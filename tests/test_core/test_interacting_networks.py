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
Simple tests for the InteractingNetworks class.
"""
import numpy as np

from pyunicorn.core.interacting_networks import InteractingNetworks


def test_RandomlyRewireCrossLinks():
    net = InteractingNetworks.SmallTestNetwork()
    rewired_net = net.RandomlyRewireCrossLinks(network=net,
                                               node_list1=[0, 3, 5],
                                               node_list2=[1, 2, 4], swaps=10.)

    res = rewired_net.degree()
    exp = [3, 3, 2, 2, 3, 1]
    assert (res == exp).all()

    res = rewired_net.cross_degree(node_list1=[0, 3, 5], node_list2=[1, 2, 4])
    exp = [1, 1, 0]
    assert (res == exp).all()


def test_internal_adjacency():
    net = InteractingNetworks.SmallTestNetwork()
    res = net.internal_adjacency([0, 3, 5])
    exp = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int8)
    assert (res == exp).all()

    res = net.internal_adjacency([1, 2, 4])
    exp = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
    assert (res == exp).all()


def test_cross_adjacency():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_adjacency([1, 2, 4], [0, 3, 5])
    exp = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    assert (res == exp).all()

    res = net.cross_adjacency([1, 2, 3, 4], [0, 5])
    exp = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
    assert (res == exp).all()


def test_cross_adjacency_sparse():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_adjacency_sparse([1, 2, 4], [0, 3, 5])
    exp = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
    assert (res == exp).all()


def test_internal_link_attribute():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_link_attribute("link_weights", [1, 2, 3])
    exp = np.array([[0., 2.3, 2.9],
                    [2.3, 0., 0.],
                    [2.9, 0., 0.]])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_link_attribute():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_link_attribute("link_weights", [1, 2, 3], [0, 4])
    exp = np.array([[0., 2.7],
                    [0., 1.5],
                    [1.3, 0.]])
    assert np.allclose(res, exp, atol=1e-04)


def test_internal_path_lengths():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_path_lengths([0, 3, 5], None)
    exp = np.array([[0., 1., 1.], [1., 0., 2.], [1., 2., 0.]])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.internal_path_lengths([1, 2, 4], None)
    exp = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_path_lengths():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_path_lengths([0, 3, 5], [1, 2, 4], None)
    exp = np.array([[2., 2., 1.], [1., 2., 2.], [3., 3., 2.]])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_path_lengths([0, 5], [1, 2, 3, 4], None)
    exp = np.array([[2., 2., 1., 1.], [3., 3., 2., 2.]])
    assert np.allclose(res, exp, atol=1e-04)


def test_number_cross_links():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.number_cross_links([0, 3, 5], [1, 2, 4])
    exp = 2
    assert res == exp

    res = net.number_cross_links([0, 5], [1, 2, 3, 4])
    exp = 2
    assert res == exp


def test_total_cross_degree():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.total_cross_degree([0, 3, 5], [1, 2, 4])
    exp = 0.6667
    assert np.isclose(res, exp, atol=1e-04)

    res = net.total_cross_degree([0, 5], [1, 2, 3, 4])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)


def test_number_internal_links():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.number_internal_links([0, 3, 5])
    exp = 2
    assert res == exp

    res = net.number_internal_links([1, 2, 4])
    exp = 3
    assert res == exp


def test_cross_degree_density():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_degree_density([0, 3, 5], [1, 2, 4])
    exp = np.array([0.33333333, 0.33333333, 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_link_density():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_link_density([0, 3, 5], [1, 2, 4])
    exp = 0.2222
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_link_density([0, 5], [1, 2, 3, 4])
    exp = 0.25
    assert np.isclose(res, exp, atol=1e-04)


def test_internal_link_density():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_link_density([0, 3, 5])
    exp = 0.6667
    assert np.isclose(res, exp, atol=1e-04)

    res = net.internal_link_density([1, 2, 3, 4])
    exp = 0.6667
    assert np.isclose(res, exp, atol=1e-04)


def test_internal_global_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_global_clustering([0, 3, 5])
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.internal_global_clustering([1, 2, 4])
    exp = 0.5556
    assert np.isclose(res, exp, atol=1e-04)


def test_cross_global_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_global_clustering([0, 3, 5], [1, 2, 4])
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_global_clustering([2], [1, 3, 4])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_global_clustering([3, 4], [1, 2])
    exp = 0.5
    assert np.isclose(res, exp, atol=1e-04)


def test_cross_global_clustering_sparse():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_global_clustering_sparse([0, 3, 5], [1, 2, 4])
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_global_clustering_sparse([2], [1, 3, 4])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_global_clustering_sparse([3, 4], [1, 2])
    exp = 0.5
    assert np.isclose(res, exp, atol=1e-04)


def test_cross_transitivity():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_transitivity([0, 3, 5], [1, 2, 4])
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_transitivity([2], [1, 3, 4])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_transitivity([3, 4], [1, 2])
    exp = 1.0


def test_cross_transitivity_sparse():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_transitivity_sparse([0, 3, 5], [1, 2, 4])
    exp = 0.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_transitivity_sparse([2], [1, 3, 4])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_transitivity_sparse([3, 4], [1, 2])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)


def test_cross_average_path_length():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_average_path_length([0, 3, 5], [1, 2, 4], None)
    exp = 2.0
    assert np.isclose(res, exp, atol=1e-04)

    res = net.cross_average_path_length([0, 5], [1, 2, 3, 4], None)
    exp = 2.0
    assert np.isclose(res, exp, atol=1e-04)


def test_internal_average_path_length():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_average_path_length([0, 3, 5], None)
    exp = 1.3333
    assert np.isclose(res, exp, atol=1e-04)

    res = net.internal_average_path_length([1, 2, 4], None)
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)


def test_average_cross_closeness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.average_cross_closeness([0, 5], [1, 2, 3, 4])
    exp = 0.5333
    assert np.isclose(res, exp, atol=1e-04)


def test_global_efficiency():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.global_efficiency([0, 5], [1, 2, 3, 4])
    exp = 1.7143
    assert np.isclose(res, exp, atol=1e-04)


def test_cross_degree():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_degree([0, 3, 5], [1, 2, 4])
    exp = np.array([1, 1, 0])
    assert (res == exp).all()

    res = net.cross_degree([1, 2, 4], [0, 3, 5])
    exp = np.array([1, 0, 1])
    assert (res == exp).all()

    res = net.cross_degree([1, 2, 3, 4], [0, 5])
    exp = np.array([0, 0, 1, 1])
    assert (res == exp).all()


def test_cross_indegree():
    net = InteractingNetworks.SmallDirectedTestNetwork()

    res = net.cross_indegree([1, 2], [0, 3, 4])
    exp = np.array([2, 1])
    assert (res == exp).all()


def test_cross_outdegree():
    net = InteractingNetworks.SmallDirectedTestNetwork()

    res = net.cross_outdegree([1, 2], [0, 3, 4])
    exp = np.array([1, 0])
    assert (res == exp).all()


def test_internal_degree():
    net = InteractingNetworks.SmallDirectedTestNetwork()

    res = net.internal_degree([0, 3, 5])
    exp = np.array([2, 1, 1])
    assert (res == exp).all()

    res = net.internal_degree([1, 2, 4])
    exp = np.array([2, 2, 2])
    assert (res == exp).all()


def test_internal_indegree():
    net = InteractingNetworks.SmallDirectedTestNetwork()

    res = net.internal_indegree([0, 1, 3])
    exp = np.array([0, 2, 1])
    assert (res == exp).all()


def test_internal_outdegree():
    net = InteractingNetworks.SmallDirectedTestNetwork()

    res = net.internal_outdegree([0, 1, 3])
    exp = np.array([2, 0, 1])
    assert (res == exp).all()


def test_cross_local_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_local_clustering([0, 3, 5], [1, 2, 4])
    exp = np.array([0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_local_clustering([2], [1, 3, 4])
    exp = np.array([1.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_local_clustering([3, 4], [1, 2])
    exp = np.array([0., 1.])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_local_clustering_sparse():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_local_clustering_sparse([0, 3, 5], [1, 2, 4])
    exp = np.array([0., 0., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_local_clustering_sparse([2], [1, 3, 4])
    exp = np.array([1.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_local_clustering_sparse([3, 4], [1, 2])
    exp = np.array([0., 1.])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_closeness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_closeness([0, 3, 5], [1, 2, 4], None)
    exp = np.array([0.6, 0.6, 0.375])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_closeness([0, 5], [1, 2, 3, 4], None)
    exp = np.array([0.66666667, 0.4])
    assert np.allclose(res, exp, atol=1e-04)


def test_internal_closeness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_closeness([0, 3, 5], None)
    exp = np.array([1., 0.66666667, 0.66666667])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.internal_closeness([1, 2, 4], None)
    exp = np.array([1., 1., 1.])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_betweenness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.cross_betweenness([2], [3, 5])
    exp = np.array([1., 1., 0., 0., 1., 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.cross_betweenness(range(0, 6), range(0, 6))
    exp = np.array([9., 3., 0., 2., 6., 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_internal_betweenness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.internal_betweenness(range(0, 6))
    exp = np.array([9., 3., 0., 2., 6., 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_local_efficiency():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.local_efficiency([0, 5], [1, 2, 3, 4])
    exp = np.array([0.75, 0.41666667])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_degree():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_degree([0, 1, 2], [3, 4, 5])
    exp = np.array([4.2, 2.6, 1.4])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_cross_degree([0, 2, 5], [1, 4])
    exp = np.array([1.4, 2.2, 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_mean_degree():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_mean_degree([0, 1, 2], [3, 4, 5])
    exp = 2.5
    assert np.isclose(res, exp, atol=1e-04)

    res = net.nsi_cross_mean_degree([0, 2, 5], [1, 4])
    exp = 0.95
    assert np.isclose(res, exp, atol=1e-04)


def test_nsi_internal_degree():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_internal_degree([0, 3, 5])
    exp = np.array([3.4, 1.8, 2.2])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_internal_degree([0, 1, 3, 5])
    exp = np.array([3.4, 2., 2.6, 2.2])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_local_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_local_clustering([0, 1, 2], [3, 4, 5])
    exp = np.array([0.33786848, 0.50295858, 1.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_cross_local_clustering([0, 2, 5], [1, 4])
    exp = np.array([1., 1., 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_closeness_centrality():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_closeness_centrality([0, 1, 2], [3, 4, 5])
    exp = np.array([1., 0.56756757, 0.48837209])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_cross_closeness_centrality([0, 2, 5], [1, 4])
    exp = np.array([0.73333333, 1., 0.42307692])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_internal_closeness_centrality():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_internal_closeness_centrality([0, 3, 5])
    exp = np.array([1., 0.68, 0.73913043])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_internal_closeness_centrality([0, 1, 3, 5])
    exp = np.array([0.84, 0.525, 0.72413793, 0.6])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_global_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_global_clustering([0, 1, 2], [3, 4, 5])
    exp = 0.6688
    assert np.isclose(res, exp, atol=1e-04)


def test_nsi_internal_local_clustering():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_internal_local_clustering([1, 2, 3, 5])
    exp = np.array([0.73333333, 1., 1., 1.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_internal_local_clustering([0, 2, 4])
    exp = np.array([1., 1., 0.86666667])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_betweenness():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_betweenness([0, 4, 5], [1, 3])
    exp = np.array([6.5333, 1.2, 0., 0.6769, 0.6769, 0.])
    assert np.allclose(res, exp, atol=1e-04)

    res = net.nsi_cross_betweenness([0, 1], [2, 3, 4, 5])
    exp = np.array([2.1333, 0., 0., 0.4923, 0.9209, 0.])
    assert np.allclose(res, exp, atol=1e-04)


def test_nsi_cross_edge_density():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_edge_density([1, 2, 3], [0, 5])
    exp = 0.1091
    assert np.isclose(res, exp, atol=1e-04)

    res = net.nsi_cross_edge_density([0], [1, 4, 5])
    exp = 0.7895
    assert np.isclose(res, exp, atol=1e-04)


def test_nsi_cross_transitivity():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_transitivity([1, 2], [0, 3, 4, 5])
    exp = 0.6352
    assert np.isclose(res, exp, atol=1e-04)

    res = net.nsi_cross_transitivity([0, 2, 3], [1])
    exp = 1.0
    assert np.isclose(res, exp, atol=1e-04)


def test_nsi_cross_average_path_length():
    net = InteractingNetworks.SmallTestNetwork()

    res = net.nsi_cross_average_path_length([0, 5], [1, 2, 4])
    exp = 3.3306
    assert np.isclose(res, exp, atol=1e-04)

    res = net.nsi_cross_average_path_length([1, 3, 4, 5], [2])
    exp = 0.376
    assert np.isclose(res, exp, atol=1e-04)
