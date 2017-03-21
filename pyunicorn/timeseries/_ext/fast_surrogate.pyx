# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# This file is part of pyunicorn.
# Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)

import numpy as np
cimport numpy as np

cdef extern from "src_fast_surrogate.h":
  void _test_pearson_correlation_fast(double *original_data,
    double *surrogates, float *correlation, int n_time, int N, double norm)
  void _test_pearson_correlation_slow(double *original_data,
    double *surrogates, float *correlation, int n_time, int N, double norm)


def test_pearson_correlation_fast(
    np.ndarray[double, ndim=2, mode='c'] original_data not None,
    np.ndarray[double, ndim=2, mode='c'] surrogates not None, 
    int N, int n_time):
    
    cdef double norm = 1.0 / float(n_time)

    #  Initialize Pearson correlation matrix
    cdef np.ndarray[float, ndim=2, mode='c'] correlation = np.zeros((N, N), dtype="float32")

    _test_pearson_correlation_fast(
        <double*> np.PyArray_DATA(original_data),
        <double*> np.PyArray_DATA(surrogates),
        <float*> np.PyArray_DATA(correlation),
        n_time, N, norm)

    return correlation


def test_pearson_correlation_slow(
    np.ndarray[double, ndim=2, mode='c'] original_data not None,
    np.ndarray[double, ndim=2, mode='c'] surrogates not None, 
    int N, int n_time):
    
    cdef double norm = 1.0 / float(n_time)

    #  Initialize Pearson correlation matrix
    cdef np.ndarray[float, ndim=2, mode='c'] correlation = np.zeros((N, N), dtype="float32")

    _test_pearson_correlation_slow(
        <double*> np.PyArray_DATA(original_data),
        <double*> np.PyArray_DATA(surrogates),
        <float*> np.PyArray_DATA(correlation),
        n_time, N, norm)

    return correlation
