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
Simple tests for the funcnet CouplingAnalysisPurePython class.
"""
import numpy as np

from pyunicorn.funcnet import CouplingAnalysis, CouplingAnalysisPurePython


def test_cross_correlation_all():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    res = coup_ana.cross_correlation(tau_max=1, lag_mode='all')
    exp = np.array([[[0.8181, 0.4121, 0.5183, 0.3773],
                     [0.5802, 0.9359, 0.5377, 0.5395],
                     [0.7786, 0.3763, 0.4961, 0.4081],
                     [0.6041, 0.4265, 0.6000, 0.4368]],
                    [[1., 0.4871, 0.6233, 0.4856],
                     [0.4871, 1., 0.4486, 0.5180],
                     [0.6233, 0.4486, 1., 0.4988],
                     [0.4856, 0.5180, 0.4988, 1.]],
                    [[0.8173, 0.5804, 0.7787, 0.6041],
                     [0.4100, 0.9361, 0.3783, 0.4286],
                     [0.5179, 0.5376, 0.4961, 0.5995],
                     [0.3765, 0.5408, 0.4086, 0.4378]]])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_correlation_sum():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    res = coup_ana.cross_correlation(
        tau_max=1, lag_mode='sum')
    exp = np.array([[[1.8173, 1.0676, 1.4021, 1.0898],
                     [0.8972, 1.9361, 0.8270, 0.9466],
                     [1.1413, 0.9863, 1.4961, 1.0984],
                     [0.8622, 1.0588, 0.9075, 1.4378]],
                    [[1.8181, 0.8993, 1.1417, 0.8630],
                     [1.0674, 1.9359, 0.9864, 1.0575],
                     [1.4020, 0.8250, 1.4961, 0.9069],
                     [1.0898, 0.9445, 1.0988, 1.4368]]])
    assert np.allclose(res, exp, atol=1e-04)


def test_cross_correlation_max():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.cross_correlation(
        tau_max=5, lag_mode='max')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[1., 0.7559, 0.7803, 0.7520],
                     [0.7569, 1., 0.5820, 0.5521],
                     [0.7807, 0.5808, 1., 0.5999],
                     [0.7521, 0.5501, 0.6005, 1.]]),
           np.array([[0, 4, 1, 2], [-4, 0, -3, -2],
                     [-1, 3, 0, 1], [-2, 2, -1, 0]]))
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_all():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    res = coup_ana.mutual_information(
        tau_max=1, lag_mode='all')
    exp = np.array([[[0.2128, 0.0767, 0.1049, 0.0706],
                     [0.1153, 0.3755, 0.1001, 0.1007],
                     [0.1952, 0.0735, 0.0910, 0.0722],
                     [0.1166, 0.0804, 0.1236, 0.0807]],
                    [[1.0003, 0.0974, 0.1309, 0.0858],
                     [0.0974, 1.0003, 0.0782, 0.1042],
                     [0.1309, 0.0782, 1.0003, 0.0944],
                     [0.0858, 0.1042, 0.0944, 1.0003]],
                    [[0.2129, 0.1158, 0.1959, 0.1162],
                     [0.0766, 0.3756, 0.0760, 0.0806],
                     [0.1047, 0.1002, 0.0906, 0.1237],
                     [0.0686, 0.1019, 0.0716, 0.0812]]])
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_sum():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    res = coup_ana.mutual_information(
        tau_max=5, lag_mode='sum')
    exp = np.array([[[1.6146, 0.9180, 0.7181, 0.6732],
                     [0.4027, 2.1516, 0.3861, 0.4253],
                     [0.4756, 0.6212, 1.3498, 0.5132],
                     [0.3866, 0.5948, 0.3683, 1.3113]],
                    [[1.6117, 0.3971, 0.4783, 0.3916],
                     [0.9158, 2.1477, 0.6197, 0.6002],
                     [0.7116, 0.3791, 1.3447, 0.3708],
                     [0.6632, 0.4233, 0.5123, 1.3074]]])
    assert np.allclose(res, exp, atol=1e-04)


def test_mutual_information_max():
    coup_ana = CouplingAnalysisPurePython(CouplingAnalysis.test_data())
    similarity_matrix, lag_matrix = coup_ana.mutual_information(
        tau_max=5, lag_mode='max')
    res = (similarity_matrix, lag_matrix)
    exp = (np.array([[1., 0.1959, 0.1956, 0.1768],
                     [0.2008, 1., 0.1169, 0.1093],
                     [0.1972, 0.1170, 1., 0.1232],
                     [0.1796, 0.1069, 0.1245, 1.]]),
           np.array([[0, 4, 1, 2], [-4, 0, -2, -2],
                     [-1, 2, 0, 1], [-2, 2, -1, 0]]))
    assert np.allclose(res, exp, atol=1e-04)
