/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

#include <math.h>

// coupling_analysis ==========================================================

void _symmetrize_by_absmax_fast(float *similarity_matrix,
    signed char *lag_matrix, int N)  {

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


void _cross_correlation_max_fast(float *array, float *similarity_matrix,
    signed char *lag_matrix, int N, int tau_max, int corr_range)  {

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
                similarity_matrix[i*N+j] = max/(float) corr_range;
                lag_matrix[i*N+j] = tau_max - argmax;
            }
        }
    }
}


void _cross_correlation_all_fast(float *array, float *lagfuncs, int N,
    int tau_max, int corr_range)  {

    int i,j,tau,k;
    double crossij;
    // loop over all node pairs, NOT symmetric due to time shifts!
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            // loop over taus INCLUDING the last tau value
            for( tau = 0; tau < tau_max + 1; tau++) {
                crossij = 0;
                // here the actual cross correlation is calculated
                // assuming standardized arrays
                for ( k = 0; k < corr_range; k++) {
                    crossij += array[tau*(tau_max+1)+i*N+k] *
                               array[tau_max*(tau_max+1)+j*N+k];
                }
                lagfuncs[i*N+j*N+(tau_max-tau)] =
                    crossij/(float)(corr_range);
            }
        }
    }
}


void _get_nearest_neighbors_fast(float *array, int T, int dim_x, int dim_y,
        int k, int dim, int *k_xz, int *k_yz, int *k_z)  {

    int i, j, index=0, t, m, n, d, kxz, kyz, kz, indexfound[T];
    double  dz=0., dxyz=0., dx=0., dy=0., eps, epsmax;
    double dist[T*dim], dxyzarray[k+1];

    // Loop over time
    for(i = 0; i < T; i++){

        // Growing cube algorithm: Test if n = #(points in epsilon-
        // environment of reference point i) > k
        // Start with epsilon for which 95% of points are inside the cube
        // for a multivariate Gaussian
        // eps increased by 2 later, also the initial eps
        eps = 1.*pow((((float) k)/(float) T), (1./dim));

        // n counts the number of neighbors
        n = 0;
        while(n <= k){
            // Increase cube size
            eps *= 2.;
            // Start with zero again
            n = 0;
            // Loop through all points
            for(t = 0; t < T; t++){
                d = 0;
                while(fabs(array[d*T + i] - array[d*T + t] ) < eps
                        && d < dim){
                        d += 1;
                }
                // If all distances are within eps, the point t lies
                // within eps and n is incremented
                if(d == dim){
                    indexfound[n] = t;
                    n += 1;
                }
            }
        }

        // Calculate distance to points only within epsilon environment
        // according to maximum metric
        for(j = 0; j < n; j++){
            index = indexfound[j];

            dxyz = 0.;
            for(d = 0; d < dim; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + index]);
                dxyz = fmax( dist[d*T + j], dxyz);
            }

            // Use insertion sort
            dxyzarray[j] = dxyz;
            if ( j > 0 ){
                // only list of smallest k+1 distances need to be kept!
                m = fmin(k, j-1);
                while ( m >= 0 && dxyzarray[m] > dxyz ){
                    dxyzarray[m+1] = dxyzarray[m];
                    m -= 1;
                }
                dxyzarray[m+1] = dxyz;
            }

        }

        // Epsilon of k-th nearest neighbor in joint space
        epsmax = dxyzarray[k];

        // Count neighbors within epsmax in subspaces, since the reference
        // point is included, all neighbors are at least 1
        kz = 0;
        kxz = 0;
        kyz = 0;
        for(j = 0; j < T; j++){

            // X-subspace
            dx = fabs(array[0*T + i] - array[0*T + j]);
            for(d = 1; d < dim_x; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dx = fmax( dist[d*T + j], dx);
            }

            // Y-subspace
            dy = fabs(array[dim_x*T + i] - array[dim_x*T + j]);
            for(d = dim_x; d < dim_x+dim_y; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dy = fmax( dist[d*T + j], dy);
            }

            // Z-subspace, if empty, dz stays 0
            dz = 0.;
            for(d = dim_x+dim_y; d < dim ; d++){
                dist[d*T + j] = fabs(array[d*T + i] - array[d*T + j]);
                dz = fmax( dist[d*T + j], dz);
            }

            // For no conditions, kz is counted up to T
            if (dz < epsmax){
                kz += 1;
                if (dx < epsmax){
                    kxz += 1;
                }
                if (dy < epsmax){
                    kyz += 1;
                }
            }
        }
        // Write to numpy arrays
        k_xz[i] = kxz;
        k_yz[i] = kyz;
        k_z[i] = kz;

    }
}
