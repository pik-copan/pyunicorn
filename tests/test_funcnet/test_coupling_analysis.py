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
Simple tests for the funcnet CouplingAnalysis class.
"""
import numpy as np
import pytest

from pyunicorn.core.data import Data
from pyunicorn.funcnet import CouplingAnalysis

from pyunicorn.core._ext.types import LAG, FIELD


def create_test_data():
    # Create test time series
    tdata = Data.SmallTestData().observable()
    # subtract means form the input data
    tdata -= np.mean(tdata, axis=1)[:, None]
    # normalize the data
    tdata /= np.sqrt(np.sum(tdata*tdata, axis=1))[:, None]
    return tdata


def test_symmetrize_by_absmax():
    # Test example
    ca = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = ca.cross_correlation(tau_max=2)

    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[1., 0.698, 0.7788, 0.7535],
                     [0.4848, 1., 0.4507, 0.52],
                     [0.6219, 0.5704, 1., 0.5996],
                     [0.4833, 0.5503, 0.5002, 1.]]),
           np.array([[0, 2, 1, 2], [0, 0, 0, 0],
                     [0, 2, 0, 1], [0, 2, 0, 0]]))
    assert np.allclose(res, exp, atol=1e-04)

    res = ca.symmetrize_by_absmax(similarity_matrix, lag_matrix)
    exp = (np.array([[1., 0.698, 0.7788, 0.7535],
                     [0.698, 1., 0.5704, 0.5503],
                     [0.7788, 0.5704, 1., 0.5996],
                     [0.7535, 0.5503, 0.5996, 1.]]),
           np.array([[0, 2, 1, 2], [-2, 0, -2, -2],
                     [-1, 2, 0, 1], [-2, 2, -1, 0]]))
    assert np.allclose(res, exp, atol=1e-04)

    # Random consistency test
    tdata = create_test_data()
    n_index, n_times = tdata.shape
    coup_ana = CouplingAnalysis(tdata)
    similarity_matrix = np.random.rand(n_index, n_times).astype(FIELD)
    lag_matrix = np.random.rand(n_index, n_times).astype(np.int8)
    sm_new = coup_ana.symmetrize_by_absmax(similarity_matrix, lag_matrix)[0]
    for i in range(n_index):
        for j in range(n_times):
            assert sm_new[i, j] >= similarity_matrix[i, j]


def test_cross_correlation_max():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.cross_correlation(
        tau_max=5, lag_mode='max')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[1., 0.7570, 0.7790, 0.7536],
                     [0.4847, 1., 0.4502, 0.5197],
                     [0.6219, 0.5844, 1., 0.5992],
                     [0.4827, 0.5509, 0.4996, 1.]]),
           np.array([[0, 4, 1, 2], [0, 0, 0, 0], [0, 3, 0, 1], [0, 2, 0, 0]]))
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_correlation_all():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    res = coup_ana.cross_correlation(tau_max=1, lag_mode='all')
    exp = np.array(
        [[[1., 0.8173], [0.4849, 0.5804], [0.6214, 0.7786], [0.4831, 0.6042]],
         [[0.4849, 0.4101], [1., 0.9362], [0.4503, 0.3780], [0.5199, 0.4286]],
         [[0.6214, 0.5178], [0.4503, 0.5376], [1., 0.4962], [0.5004, 0.5996]],
         [[0.4831, 0.3762], [0.5199, 0.5404], [0.5004, 0.4092], [1., 0.4380]]])
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_knn():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.mutual_information(
        tau_max=5, knn=10, estimator='knn')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[4.6505, 0.4387, 0.4652, 0.4126],
                     [0.147, 4.6505, 0.1065, 0.1639],
                     [0.2483, 0.2126, 4.6505, 0.2204],
                     [0.1209, 0.199, 0.1453, 4.6505]], dtype=FIELD),
           np.array([[0, 4, 1, 2],
                     [0, 0, 0, 0],
                     [0, 2, 0, 1],
                     [0, 2, 0, 0]], dtype=LAG))
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_binning():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.mutual_information(
        tau_max=5, bins=6, estimator='binning')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[1.7828, 0.3765, 0.3551, 0.3288],
                     [0.1326, 1.7828, 0.1140, 0.1498],
                     [0.1951, 0.1784, 1.7828, 0.1918],
                     [0.1139, 0.1681, 0.1266, 1.7828]], dtype=FIELD),
           np.array([[0, 4, 1, 2],
                     [0, 0, 0, 0],
                     [0, 3, 0, 1],
                     [0, 2, 0, 0]], dtype=LAG))
    assert np.allclose(res, exp, atol=1e-04)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
def test_mutual_information_gauss():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.mutual_information(
        tau_max=5, estimator='gauss')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[np.inf, 0.4255, 0.4668, 0.4196],
                     [0.1339, np.inf, 0.1133, 0.1574],
                     [0.2445, 0.2089, np.inf, 0.2224],
                     [0.1326, 0.1808, 0.1436, np.inf]], dtype=FIELD),
           np.array([[0, 4, 1, 2],
                     [0, 0, 0, 0],
                     [0, 3, 0, 1],
                     [0, 2, 0, 0]], dtype=LAG))
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_value_error():
    with pytest.raises(ValueError,
                       match='estimator must be "knn", "binning" or "gauss".'):
        CouplingAnalysis(CouplingAnalysis.test_data()) \
            .mutual_information(estimator='some_other')


def test_information_transfer_knn():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.information_transfer(
        tau_max=5, estimator='knn', knn=10)
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[0., 0.1544, 0.3261, 0.3047],
                     [0.0218, 0., 0.0394, 0.0976],
                     [0.0134, 0.0663, 0., 0.1502],
                     [0.0066, 0.0694, 0.0401, 0.]]),
           np.array([[0, 2, 1, 2], [5, 0, 0, 0], [5, 1, 0, 1], [5, 0, 0, 0]]))
    assert np.allclose(res, exp, atol=1e-04)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
def test_information_transfer_gauss():
    coup_ana = CouplingAnalysis(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.information_transfer(
        tau_max=5, estimator='gauss')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[0., 0.1732, 0.3256, 0.3148],
                     [0.0006, 0., 0.0324, 0.0755],
                     [0.0012, 0.0754, 0., 0.1365],
                     [0.0008, 0.0753, 0.0442, 0.]], dtype=FIELD),
           np.array([[0, 2, 1, 2],
                     [5, 0, 0, 0],
                     [3, 1, 0, 1],
                     [1, 0, 0, 0]], dtype=LAG))
    assert np.allclose(res, exp, atol=1e-04)


def test_information_transfer_value_error():
    with pytest.raises(ValueError,
                       match='estimator must be "knn", "binning" or "gauss".'):
        CouplingAnalysis(CouplingAnalysis.test_data()) \
            .information_transfer(estimator='some_other')
