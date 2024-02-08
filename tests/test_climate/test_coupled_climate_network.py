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
Simple tests for the CoupledClimateNetwork class.
"""

from pyunicorn.climate import ClimateData, CoupledTsonisClimateNetwork


def test_internal_link_density():
    cd = ClimateData.SmallTestData()
    tsonis_ccn = CoupledTsonisClimateNetwork(cd, cd, threshold=.2)
    res = tsonis_ccn.internal_link_density()
    assert isinstance(res, tuple) and all(isinstance(d, float) for d in res)
    assert all(0 <= d <= 1 for d in res)
