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

import numpy as np
cimport numpy as cnp
from numpy cimport ndarray

from ...core._ext.types import FIELD, INT64TYPE
from ...core._ext.types cimport MASK_t, FIELD_t, INT64TYPE_t

cdef extern from "src_numerics.c":
    void _mutual_information(
            float *anomaly, int n_samples, int N, int n_bins, double scaling,
            double range_min, long *symbolic, long *hist, long *hist2d,
            float *mi)
    void _spearman_corr(int m, int tmax, bint *final_mask,
            float *time_series_ranked, float *spearman_rho)


# mutual_info =================================================================


def mutual_information(
    ndarray[FIELD_t, ndim=2, mode='c'] anomaly not None,
    int n_samples, int N, int n_bins, float scaling, float range_min):

    cdef:
        ndarray[INT64TYPE_t, ndim=2, mode='c'] symbolic = np.zeros(
            (N, n_samples), dtype=INT64TYPE)
        ndarray[INT64TYPE_t, ndim=2, mode='c'] hist = np.zeros(
            (N, n_bins), dtype=INT64TYPE)
        ndarray[INT64TYPE_t, ndim=2, mode='c'] hist2d = np.zeros(
            (n_bins, n_bins), dtype=INT64TYPE)
        ndarray[FIELD_t, ndim=2, mode='c'] mi = np.zeros(
            (N, N), dtype=FIELD)

    _mutual_information(
        <FIELD_t*> cnp.PyArray_DATA(anomaly), n_samples, N, n_bins, scaling,
        range_min, <long*> cnp.PyArray_DATA(symbolic),
        <long*> cnp.PyArray_DATA(hist), <long*> cnp.PyArray_DATA(hist2d),
        <FIELD_t*> cnp.PyArray_DATA(mi))

    return mi


# rainfall ====================================================================


def spearman_corr(int m, int tmax,
    ndarray[MASK_t, ndim=2, mode='c'] final_mask not None,
    ndarray[FIELD_t, ndim=2, mode='c'] time_series_ranked not None):

    cdef ndarray[FIELD_t, ndim=2, mode='c'] spearman_rho = np.zeros(
        (m, m), dtype=FIELD)

    _spearman_corr(m, tmax,
            <bint*> cnp.PyArray_DATA(final_mask),
            <FIELD_t*> cnp.PyArray_DATA(time_series_ranked),
            <FIELD_t*> cnp.PyArray_DATA(spearman_rho))

    return spearman_rho
