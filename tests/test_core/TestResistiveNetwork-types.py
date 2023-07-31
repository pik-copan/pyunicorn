# Copyright (C) 2014 SWIPO Project
#
# Authors (this file):
#   Stefan Schinkel <stefan.schinkel@gmail.com>
"""
Provides simple type check for resistive networks
"""

import numpy as np

from pyunicorn import ResNetwork


res = ResNetwork.SmallTestNetwork()


def testAdmittiveDegreeType():
    print("testing types")
    assert isinstance(res.admittive_degree(), np.ndarray)
