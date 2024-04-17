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
#ifdef _MSC_VER
    #include <malloc.h>
    #define ALLOCA(sz) _alloca(sz)
#else
    #include <alloca.h>
    #define ALLOCA(sz) alloca(sz)
#endif

// mutual_info ================================================================

void _mutual_information(float *anomaly, int n_samples,
    int N, int n_bins, float scaling, float range_min, long *symbolic,
    long *hist, long *hist2d, float *mi) {

    long i, j, k, l, m, in_bins, jn_bins, ln_bins, in_samples, jn_samples,
         in_nodes;
    double norm, rescaled, hpl, hpm, plm;

    float *p_anomaly;
    float *p_mi, *p_mi2;
    long *p_symbolic, *p_symbolic1, *p_symbolic2, *p_hist, *p_hist1,
         *p_hist2, *p_hist2d;

    //  Calculate histogram norm
    norm = 1.0 / n_samples;

    //  Initialize in_samples, in_bins
    in_samples = in_bins = 0;

    for (i = 0; i < N; i++) {

        //  Set pointer to anomaly(i,0)
        p_anomaly = anomaly + in_samples;
        //  Set pointer to symbolic(i,0)
        p_symbolic = symbolic + in_samples;

        for (k = 0; k < n_samples; k++) {

            //  Rescale sample into interval [0,1]
            rescaled = scaling * (*p_anomaly - range_min);

            //  Calculate symbolic trajectories for each time series,
            //  where the symbols are bin numbers.
            if (rescaled < 1.0) {
                *p_symbolic = (long) (rescaled * n_bins);
            }
            else {
                *p_symbolic = n_bins - 1;
            }

            //  Calculate 1d-histograms for single time series
            //  Set pointer to hist(i, *p_symbolic)
            p_hist = hist + in_bins + *p_symbolic;
            (*p_hist)++;

            //  Set pointer to anomaly(k+1,i)
            p_anomaly++;
            //  Set pointer to symbolic(k+1,i)
            p_symbolic++;
        }
        in_samples += n_samples;
        in_bins += n_bins;
    }

    //  Initialize in_samples, in_bins, in_nodes
    in_samples = in_bins = in_nodes = 0;

    for (i = 0; i < N; i++) {

        //  Set pointer to mi(i,0)
        p_mi = mi + in_nodes;
        //  Set pointer to mi(0,i)
        p_mi2 = mi + i;

        //  Initialize jn_samples, jn_bins
        jn_samples = jn_bins = 0;

        for (j = 0; j <= i; j++) {

            //  Don't do anything for i = j, this case is not of
            //  interest here!
            if (i != j) {

                //  Set pointer to symbolic(i,0)
                p_symbolic1 = symbolic + in_samples;
                //  Set pointer to symbolic(j,0)
                p_symbolic2 = symbolic + jn_samples;

                //  Calculate 2d-histogram for one pair of time series
                //  (i,j).
                for (k = 0; k < n_samples; k++) {

                    //  Set pointer to hist2d(*p_symbolic1, *p_symbolic2)
                    p_hist2d = hist2d + (*p_symbolic1)*n_bins
                               + *p_symbolic2;

                    (*p_hist2d)++;

                    //  Set pointer to symbolic(i,k+1)
                    p_symbolic1++;
                    //  Set pointer to symbolic(j,k+1)
                    p_symbolic2++;
                }

                //  Calculate mutual information for one pair of time
                //  series (i,j).

                //  Set pointer to hist(i,0)
                p_hist1 = hist + in_bins;

                //  Initialize ln_bins
                ln_bins = 0;

                for (l = 0; l < n_bins; l++) {

                    //  Set pointer to hist(j,0)
                    p_hist2 = hist + jn_bins;
                    //  Set pointer to hist2d(l,0)
                    p_hist2d = hist2d + ln_bins;

                    hpl = (double) (*p_hist1) * norm;

                    if (hpl > 0.0) {
                        for (m = 0; m < n_bins; m++) {

                            hpm = (double) (*p_hist2) * norm;

                            if (hpm > 0.0) {
                                plm = (double) (*p_hist2d) * norm;
                                if (plm > 0.0) {
                                    *p_mi += (float) (plm * log(plm/hpm/hpl));
                                }
                            }

                            //  Set pointer to hist(j,m+1)
                            p_hist2++;
                            //  Set pointer to hist2d(l,m+1)
                            p_hist2d++;
                        }
                    }
                    //  Set pointer to hist(i,l+1)
                    p_hist1++;

                    ln_bins += n_bins;
                }

                //  Symmetrize MI
                *p_mi2 = *p_mi;

                //  Initialize ln_bins
                ln_bins = 0;

                //  Reset hist2d to zero in all bins
                for (l = 0; l < n_bins; l++) {

                    //  Set pointer to hist2d(l,0)
                    p_hist2d = hist2d + ln_bins;

                    for (m = 0; m < n_bins; m++) {
                        *p_hist2d = 0;

                        //  Set pointer to hist2d(l,m+1)
                        p_hist2d++;
                    }
                    ln_bins += n_bins;
                }
            }
            //  Set pointer to mi(i,j+1)
            p_mi++;
            //  Set pointer to mi(j+1,i)
            p_mi2 += N;

            jn_samples += n_samples;
            jn_bins += n_bins;
        }
        in_samples += n_samples;
        in_bins += n_bins;
        in_nodes += N;
    }
}


// rainfall ===================================================================


void _spearman_corr(int m, int tmax, int *final_mask,
    float *time_series_ranked, float *spearman_rho)  {

    double cov = 0, sigmai = 0, sigmaj = 0, meani = 0, meanj = 0;
    int zerocount = 0;
    unsigned int T = (unsigned int) tmax;
    double *rankedi = ALLOCA(T * sizeof(double)),
           *rankedj = ALLOCA(T * sizeof(double)),
           *normalizedi = ALLOCA(T * sizeof(double)),
           *normalizedj = ALLOCA(T * sizeof(double));

    for (int i=0; i<m; i++) {
        for (int j=i; j<m; j++) {
            for (int t=0; t<tmax; t++) {
                if ((final_mask[i*m+t] | final_mask[j*m+t]) == 0)
                    zerocount = zerocount+1;
            }

            for (int t=0; t<tmax; t++) {
                rankedi[t] = time_series_ranked[i*m+t] - (float) zerocount;
                rankedj[t] = time_series_ranked[j*m+t] - (float) zerocount;
            }

            for (int t=0; t<tmax; t++) {
                if (rankedi[t]>=0)
                    meani = meani + rankedi[t];
                if (rankedj[t]>=0)
                    meanj = meanj + rankedj[t];
            }

            meani = meani/(tmax-zerocount);
            meanj = meanj/(tmax-zerocount);

            for (int t=0; t<tmax; t++) {
                if ((final_mask[i*m+t] | final_mask[j*m+t]) != 0) {
                    normalizedi[t] = rankedi[t] - meani;
                    normalizedj[t] = rankedj[t] - meanj;
                    cov = cov + normalizedi[t]*normalizedj[t];
                    sigmai = sigmai + normalizedi[t]*normalizedi[t];
                    sigmaj = sigmaj + normalizedj[t]*normalizedj[t];
                }
            }
            spearman_rho[i*m+j] = spearman_rho[j*m+i] =
                (float) (cov/sqrt(sigmai*sigmaj));
            meani = 0;
            meanj = 0;
            cov = 0;
            sigmai = 0;
            sigmaj = 0;
            zerocount = 0;
        }
    }
}
