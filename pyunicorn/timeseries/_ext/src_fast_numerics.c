/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

void _test_pearson_correlation_fast(double *original_data, double *surrogates, 
    float *correlation, int n_time, int N, double norm)  {

    float *p_correlation;
    double *p_original, *p_surrogates;

    for (int i = 0; i < N; i++) {
        //  Set pointer to correlation(i,0)
        p_correlation = correlation + i*N;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                //  Set pointer to original_data(i,0)
                p_original = original_data + i*n_time;
                //  Set pointer to surrogates(j,0)
                p_surrogates = surrogates + j*n_time;

                for (int k = 0; k < n_time; k++) {
                    *p_correlation += (*p_original) * (*p_surrogates);
                    //  Set pointer to original_data(i,k+1)
                    p_original++;
                    //  Set pointer to surrogates(j,k+1)
                    p_surrogates++;
                }
                *p_correlation *= norm;
            }
            p_correlation++;
        }
    }
}


void _test_pearson_correlation_slow(double *original_data, double *surrogates,
    float *correlation, int n_time, int N, double norm)  {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                for (int k = 0; k < n_time; k++) {
                    correlation[i*N+j] += original_data[i*N+k] *
                      surrogates[j*N+k];
                }
                correlation[i*N+j] *= norm;
            }
        }
    }
}
