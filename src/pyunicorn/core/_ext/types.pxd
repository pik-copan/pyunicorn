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
cimport numpy as cnp

cnp.import_array()


ctypedef cnp.int8_t BOOLTYPE_t
ctypedef cnp.int8_t INT8TYPE_t
ctypedef cnp.int16_t INT16TYPE_t
ctypedef cnp.int32_t INT32TYPE_t
ctypedef cnp.int64_t INT64TYPE_t
ctypedef cnp.float32_t FLOAT32TYPE_t
ctypedef cnp.float64_t FLOAT64TYPE_t

ctypedef BOOLTYPE_t ADJ_t
ctypedef BOOLTYPE_t MASK_t
ctypedef INT8TYPE_t LAG_t
ctypedef INT16TYPE_t DEGREE_t
ctypedef INT32TYPE_t NODE_t
ctypedef FLOAT32TYPE_t WEIGHT_t
ctypedef FLOAT64TYPE_t DWEIGHT_t
ctypedef FLOAT32TYPE_t FIELD_t
ctypedef FLOAT64TYPE_t DFIELD_t
