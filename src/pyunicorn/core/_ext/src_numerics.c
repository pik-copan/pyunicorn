/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2023 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

#include <math.h>
#include <stdlib.h>


// network ====================================================================


void _do_nsi_hamming_clustering_fast(int n2, int nActiveIndices, float mind0,
    float minwp0, int lastunited, int part1, int part2, double *distances,
    int *theActiveIndices, double *linkedWeights, double *weightProducts,
    double *errors, double *result, int *mayJoin)  {

    
    int i1, i2, i3, c3;
    int newpart1=0;
    int newpart2=0;
    double d, lw, mind=mind0, minwp=minwp0;
    for (i1=0; i1<nActiveIndices; i1++) {
        int c1 = theActiveIndices[i1];
        if ((lastunited==-1) || (c1==lastunited)) {
            for (i2=0; i2<i1; i2++) {
                int c2 = theActiveIndices[i2];
                if (mayJoin[c1*n2+c2]>0) {
                    d = 0.0;
                    for (i3=0; i3<i2; i3++) {
                        c3 = theActiveIndices[i3];
                        lw = linkedWeights[c1*n2+c3]
                                + linkedWeights[c2*n2+c3];
                        d += fmin(lw,weightProducts[c1*n2+c3]
                                + weightProducts[c2*n2+c3]-lw)
                                - errors[c1*n2+c3] - errors[c2*n2+c3];
                    }
                    for (i3=i2+1; i3<i1; i3++) {
                        c3 = theActiveIndices[i3];
                        lw = linkedWeights[c1*n2+c3]
                                + linkedWeights[c2*n2+c3];
                        d += fmin(lw,weightProducts[c1*n2+c3]
                                + weightProducts[c2*n2+c3]-lw)
                                - errors[c1*n2+c3] - errors[c2*n2+c3];
                    }
                    for (i3=i1+1; i3<nActiveIndices; i3++) {
                        c3 = theActiveIndices[i3];
                        lw = linkedWeights[c1*n2+c3]
                                + linkedWeights[c2*n2+c3];
                        d += fmin(lw,weightProducts[c1*n2+c3]
                                + weightProducts[c2*n2+c3]-lw)
                                - errors[c1*n2+c3] - errors[c2*n2+c3];
                    }
                    double e = weightProducts[c1*n2+c2]
                                - 2.0*linkedWeights[c1*n2+c2];
                    if (e>0.0) d += e;
                    distances[c1*n2+c2] = d;
                    if ((d<mind) ||
                            ((d==mind) &&
                                (weightProducts[c1*n2+c2]<minwp))) {
                        mind = d;
                        minwp = weightProducts[c1*n2+c2];
                        newpart1 = c1;
                        newpart2 = c2;
                    }
                }
            }
        } else {
            for (i2=0; i2<i1; i2++) {
                int c2 = theActiveIndices[i2];
                if (mayJoin[c1*n2+c2]>0) {
                    double lw_united = linkedWeights[c1*n2+lastunited]
                                       + linkedWeights[c2*n2+lastunited],
                            lw_part1 = linkedWeights[c1*n2+part1]
                                       + linkedWeights[c2*n2+part1],
                            lw_part2 = linkedWeights[c1*n2+part2]
                                       + linkedWeights[c2*n2+part2];
                    distances[c1*n2+c2] += (
                        (fmin(lw_united, weightProducts[c1*n2+lastunited]
                              + weightProducts[c2*n2+lastunited]
                              - lw_united)
                           - errors[c1*n2+lastunited]
                           - errors[c2*n2+lastunited])
                        - (fmin(lw_part1,weightProducts[c1*n2+part1]
                                + weightProducts[c2*n2+part1] - lw_part1)
                           - errors[c1*n2+part1] - errors[c2*n2+part1])
                        - (fmin(lw_part2,weightProducts[c1*n2+part2]
                                + weightProducts[c2*n2+part2] -lw_part2)
                           - errors[c1*n2+part2] - errors[c2*n2+part2]));
                    d = distances[c1*n2+c2];
                    if ((d<mind) ||
                            ((d==mind) &&
                                (weightProducts[c1*n2+c2]<minwp))) {
                        mind = d;
                        minwp = weightProducts[c1*n2+c2];
                        newpart1 = c1;
                        newpart2 = c2;
                    }
                }
            }
        }
    }
    result[0] = mind;
    result[1] = newpart1;
    result[2] = newpart2;
}


double _vertex_current_flow_betweenness_fast(int N, double Is, double It,
    float *admittance, float *R, int i) {

    double VCFB=0.0;
    int t=0;
    int s=0;
    int j=0;
    double I=0;

    for(t=0;t<N;t++){
        for(s=0; s<t; s++){
            I = 0.0;
            if(i == t || i == s){
                continue;
            }
            else{
                for(j=0;j<N;j++){
                    I += admittance[i*N+j]*
                    fabs( Is*(R[i*N+s]-R[j*N+s])+
                          It*(R[j*N+t]-R[i*N+t])
                        ) / 2.0;
                } // for  j
            }
            VCFB += 2.0*I/(N*(N-1));
        } // for s
    } // for t

    return VCFB;
}


void _edge_current_flow_betweenness_fast(int N, double Is, double It, 
    float *admittance, float *R, float *ECFB) {

    int i=0;
    int j=0;
    int t=0;
    int s=0;
    double I = 0.0;

    for(i=0; i<N; i++){
        for(j=0;j<N;j++){
            I = 0.0;
            for(t=0;t<N;t++){
                for(s=0; s<t; s++){
                    I += admittance[i*N+j]*\
                         fabs(Is*(R[i*N+s]-R[j*N+s])+
                              It*(R[j*N+t]-R[i*N+t]));
                } //for s
            } // for t
            ECFB[i*N+j] += (float) (2.* I/(N*(N-1)));
        } // for j
    } // for i
}
