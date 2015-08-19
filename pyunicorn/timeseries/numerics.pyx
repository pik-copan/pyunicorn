# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2015 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)


cimport cython
cimport numpy as np

import numpy as np
import numpy.random as rd

randint = rd.randint

INTTYPE = np.int
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
ctypedef np.int_t INTTYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t


# recurrence plot==============================================================

def _embed_time_series(int n_time, int dim, int tau,
                       np.ndarray[FLOATTYPE_t, ndim=1] time_series,
                       np.ndarray[FLOAT32TYPE_t, ndim=2] embedding):
    """
    >>> 42 == 42
    True
    """

    cdef:
        int i,j, max_delay, len_embedded, index

    # Calculate the maximum delay
    max_delay = (dim - 1) * tau
    # Calculate the length of the embedded time series
    len_embedded = n_time - max_delay

    for j in xrange(dim):
        index = j * tau
        for k in xrange(len_embedded):
            embedding[k, j] = time_series[index]
            index += 1