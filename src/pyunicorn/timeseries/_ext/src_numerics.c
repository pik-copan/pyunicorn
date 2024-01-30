/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

#include <math.h>

// surrogates =================================================================

void _test_pearson_correlation_fast(double *original_data, double *surrogates, 
    float *correlation, int n_time, int N, double norm)  {

    float *p_correlation;
    double *p_original, *p_surrogates, corr;

    for (int i = 0; i < N; i++) {
        //  Set pointer to correlation(i,0)
        p_correlation = correlation + i*N;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                //  Set pointer to original_data(i,0)
                p_original = original_data + i*n_time;
                //  Set pointer to surrogates(j,0)
                p_surrogates = surrogates + j*n_time;

                corr = 0;
                for (int k = 0; k < n_time; k++) {
                    corr += (*p_original) * (*p_surrogates);
                    //  Set pointer to original_data(i,k+1)
                    p_original++;
                    //  Set pointer to surrogates(j,k+1)
                    p_surrogates++;
                }
                *p_correlation = (float) (corr * norm);
            }
            p_correlation++;
        }
    }
}


void _test_mutual_information_fast(int N, int n_time, int n_bins,
    double scaling, double range_min, double *original_data,
    double *surrogates, int *symbolic_original, int *symbolic_surrogates,
    int *hist_original, int *hist_surrogates, int * hist2d, float *mi)  {

    long i, j, k, l, m, in_bins, jn_bins, in_time, jn_time;
    double norm, rescaled, hpl, hpm, plm;

    double *p_original, *p_surrogates;
    float *p_mi;
    int *p_symbolic_original, *p_symbolic_surrogates, *p_hist_original,
         *p_hist_surrogates, *p_hist2d;

    //  Calculate histogram norm
    norm = 1.0 / n_time;

    //  Initialize in_bins, in_time
    in_time = in_bins = 0;

    for (i = 0; i < N; i++) {

        //  Set pointer to original_data(i,0)
        p_original = original_data + in_time;
        //  Set pointer to surrogates(i,0)
        p_surrogates = surrogates + in_time;
        //  Set pointer to symbolic_original(i,0)
        p_symbolic_original = symbolic_original + in_time;
        //  Set pointer to symbolic_surrogates(i,0)
        p_symbolic_surrogates = symbolic_surrogates + in_time;

        for (k = 0; k < n_time; k++) {

            //  Rescale sample into interval [0,1]
            rescaled = scaling * (*p_original - range_min);

            //  Calculate symbolic trajectories for each time series,
            //  where the symbols are bin numbers.
            if (rescaled < 1.0)
                *p_symbolic_original = (int) (rescaled * n_bins);
            else
                *p_symbolic_original = n_bins - 1;

            //  Calculate 1d-histograms for single time series
            //  Set pointer to hist_original(i, *p_symbolic_original)
            p_hist_original = hist_original + in_bins
                              + *p_symbolic_original;
            (*p_hist_original)++;

            //  Rescale sample into interval [0,1]
            rescaled = scaling * (*p_surrogates - range_min);

            //  Calculate symbolic trajectories for each time series,
            //  where the symbols are bin numbers.
            if (rescaled < 1.0)
                *p_symbolic_surrogates = (int) (rescaled * n_bins);
            else
                *p_symbolic_surrogates = n_bins - 1;

            //  Calculate 1d-histograms for single time series
            //  Set pointer to hist_surrogates(i, *p_symbolic_surrogates)
            p_hist_surrogates = hist_surrogates + in_bins
                                + *p_symbolic_surrogates;
            (*p_hist_surrogates)++;

            //  Set pointer to original_data(i,k+1)
            p_original++;
            //  Set pointer to surrogates(i,k+1)
            p_surrogates++;
            //  Set pointer to symbolic_original(i,k+1)
            p_symbolic_original++;
            //  Set pointer to symbolic_surrogates(i,k+1)
            p_symbolic_surrogates++;
        }
        in_bins += n_bins;
        in_time += n_time;
    }

    //  Initialize in_time, in_bins
    in_time = in_bins = 0;

    for (i = 0; i < N; i++) {

        //  Set pointer to mi(i,0)
        p_mi = mi + i*N;

        //  Initialize jn_time = 0;
        jn_time = jn_bins = 0;

        for (j = 0; j < N; j++) {

            //  Don't do anything if i = j, this case is not of
            //  interest here!
            if (i != j) {

                //  Set pointer to symbolic_original(i,0)
                p_symbolic_original = symbolic_original + in_time;
                //  Set pointer to symbolic_surrogates(j,0)
                p_symbolic_surrogates = symbolic_surrogates + jn_time;

                //  Calculate 2d-histogram for one pair of time series
                //  (i,j).
                for (k = 0; k < n_time; k++) {

                    //  Set pointer to hist2d(*p_symbolic_original,
                    //                        *p_symbolic_surrogates)
                    p_hist2d = hist2d + (*p_symbolic_original)*n_bins
                               + *p_symbolic_surrogates;

                    (*p_hist2d)++;

                    //  Set pointer to symbolic_original(i,k+1)
                    p_symbolic_original++;
                    //  Set pointer to symbolic_surrogates(j,k+1)
                    p_symbolic_surrogates++;
                }

                //  Calculate mutual information for one pair of time
                //  series (i,j)

                //  Set pointer to hist_original(i,0)
                p_hist_original = hist_original + in_bins;

                for (l = 0; l < n_bins; l++) {

                    //  Set pointer to hist_surrogates(j,0)
                    p_hist_surrogates = hist_surrogates + jn_bins;
                    //  Set pointer to hist2d(l,0)
                    p_hist2d = hist2d + l*n_bins;

                    hpl = (*p_hist_original) * norm;

                    if (hpl > 0.0) {
                        for (m = 0; m < n_bins; m++) {

                            hpm = (*p_hist_surrogates) * norm;

                            if (hpm > 0.0) {
                                plm = (*p_hist2d) * norm;
                                if (plm > 0.0)
                                    *p_mi += (float) (plm * log(plm/hpm/hpl));
                            }

                            //  Set pointer to hist_surrogates(j,m+1)
                            p_hist_surrogates++;
                            //  Set pointer to hist2d(l,m+1)
                            p_hist2d++;
                        }
                    }
                    //  Set pointer to hist_original(i,l+1)
                    p_hist_original++;
                }

                //  Reset hist2d to zero in all bins
                for (l = 0; l < n_bins; l++) {

                    //  Set pointer to hist2d(l,0)
                    p_hist2d = hist2d + l*n_bins;

                    for (m = 0; m < n_bins; m++) {
                        *p_hist2d = 0;

                        //  Set pointer to hist2d(l,m+1)
                        p_hist2d++;
                    }
                }
            }
            //  Set pointer to mi(i,j+1)
            p_mi++;

            jn_time += n_time;
            jn_bins += n_bins;
        }
        in_time += n_time;
        in_bins += n_bins;
    }
}
