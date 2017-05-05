/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

#ifndef PYUNICORN
#define PYUNICORN

void _test_pearson_correlation_fast(double *original_data, double *surrogates, 
    float *correlation, int n_time, int N, double norm);
void _test_pearson_correlation_slow(double *original_data, double *surrogates, 
    float *correlation, int n_time, int N, double norm);

#endif
