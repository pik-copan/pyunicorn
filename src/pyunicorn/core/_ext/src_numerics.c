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
#include <stdlib.h>


// network ====================================================================


double _vertex_current_flow_betweenness_fast(int N, double Is, double It,
    float *admittance, float *R, int i) {

    double VCFB=0.0;
    int t=0;
    int s=0;
    int j=0;
    double J=0;

    for(t=0;t<N;t++){
        for(s=0; s<t; s++){
            J = 0.0;
            if(i == t || i == s){
                continue;
            }
            else{
                for(j=0;j<N;j++){
                    J += admittance[i*N+j]*
                    fabs( Is*(R[i*N+s]-R[j*N+s])+
                          It*(R[j*N+t]-R[i*N+t])
                        ) / 2.0;
                } // for  j
            }
            VCFB += 2.0*J/(N*(N-1));
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
    double J = 0.0;

    for(i=0; i<N; i++){
        for(j=0;j<N;j++){
            J = 0.0;
            for(t=0;t<N;t++){
                for(s=0; s<t; s++){
                    J += admittance[i*N+j]*\
                         fabs(Is*(R[i*N+s]-R[j*N+s])+
                              It*(R[j*N+t]-R[i*N+t]));
                } //for s
            } // for t
            ECFB[i*N+j] += (float) (2.* J/(N*(N-1)));
        } // for j
    } // for i
}
