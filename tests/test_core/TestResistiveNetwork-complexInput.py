# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
"""
Provides Simple test for the treatment of complex admittances
in resistive networks.
"""

import numpy as np

from pyunicorn import ResNetwork


res = ResNetwork.SmallTestNetwork()
resC = ResNetwork.SmallComplexNetwork()
# a symmetric one too
r = np.zeros((5, 5), dtype=complex)
r.real = [[0, 2, 0, 0, 0],
          [2, 0, 8, 2, 0],
          [0, 8, 0, 8, 0],
          [0, 2, 8, 0, 10],
          [0, 0, 0, 10, 0]]
r.imag = r.real
resCsym = ResNetwork(r)


def testComplexAdmittance():
    # Ensure that the sum of real and imaginary
    # admittance values in resc is zeros

    sumAdm = resCsym.get_admittance().real + resCsym.get_admittance().imag
    assert np.all(sumAdm == 0)


def testPinvSymmetry():
    # test that the real and imag part of "R" are identical for
    # the test network

    assert np.allclose(resCsym.get_R().real, resCsym.get_R().imag)


def testAdmittiveDegree():

    sumAD = resCsym.admittive_degree().real + resCsym.admittive_degree().imag
    assert np.all(sumAD == 0)
