/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/


// coupling_analysis ==========================================================

void _symmetrize_by_absmax_fast(double *similarity_matrix, double *lag_matrix,
    int N)  {

    int i,j;
    // loop over all node pairs
    for (i = 0; i < N; i++) {
        for (j = i+1; j < N; j++) {
            // calculate max and argmax by comparing to
            // previous value and storing max
            if (fabs(similarity_matrix[i*N+j]) >
                    fabs(similarity_matrix[j*N+i])) {
                similarity_matrix[j*N+i] = similarity_matrix[i*N+j];
                lag_matrix[j*N+i] = -lag_matrix[i*N+j];
            }
            else {
                similarity_matrix[i*N+j] = similarity_matrix[j*N+i];
                lag_matrix[i*N+j] = -lag_matrix[j*N+i];
            }
        }
    }
}


void _cross_correlation_fast(float *array, float *similarity_matrix,
    int *lag_matrix, int N, int tau_max, int corr_range)  {

    int i,j,tau,k, argmax;
    double crossij, max;
    // loop over all node pairs, NOT symmetric due to time shifts!
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if(i != j){
                max = 0.0;
                argmax = 0;
                // loop over taus INCLUDING the last tau value
                for(tau = 0; tau < tau_max + 1; tau++) {
                    crossij = 0;
                    // here the actual cross correlation is calculated
                    // assuming standardized arrays
                    for (k = 0; k < corr_range; k++) {
                        crossij += array[tau*(tau_max+1) + i*N + k] *
                                   array[tau_max*(tau_max+1) + j*N + k];
                    }
                    // calculate max and argmax by comparing to
                    // previous value and storing max
                    if (fabs(crossij) > fabs(max)) {
                        max = crossij;
                        argmax = tau;
                    }
                }
                similarity_matrix[i*N+j] = max/((float) corr_range);
                lag_matrix[i*N+j] = tau_max - argmax;
            }
        }
    }
}
