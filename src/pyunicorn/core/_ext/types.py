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

import numpy as np


BOOLTYPE = np.int8
INT8TYPE = np.int8
INT16TYPE = np.int16
INT32TYPE = np.int32
INT64TYPE = np.int64
FLOAT32TYPE = np.float32
FLOAT64TYPE = np.float64

ADJ = BOOLTYPE
MASK = BOOLTYPE
LAG = INT8TYPE
DEGREE = INT16TYPE
NODE = INT32TYPE
WEIGHT = FLOAT32TYPE
DWEIGHT = FLOAT64TYPE
FIELD = FLOAT32TYPE
DFIELD = FLOAT64TYPE


def to_cy(arr, ty):
    return arr.astype(dtype=ty, copy=True, order='c', casting='same_kind')
