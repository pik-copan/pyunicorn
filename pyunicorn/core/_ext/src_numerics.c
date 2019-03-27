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

// geo_network ================================================================

void _randomly_rewire_geomodel_I_fast(int iterations, float eps, short *A,
    float *D, int E, int N, int *edges)  {

    int i, s, t, k, l, edge1, edge2, count;
    //int j, neighbor_s_index, neighbor_t_index;
    //int neighbor_k_index, neighbor_l_index;

    //  Create list of neighbors
    //for (int i = 0; i < N; i++) {
    //
    //    count = 0;
    //
    //    for (int j = 0; j < N; j++) {
    //        if (A(i,j) == 1) {
    //            neighbors(i,count) = j;
    //            count++;
    //        }
    //    }
    //}

    //  Initialize random number generator
    srand48(time(0));

    i = 0;
    count = 0;
    while (i < iterations) {
        //  Randomly choose 2 edges
        edge1 = floor(drand48() * E);
        edge2 = floor(drand48() * E);

        s = edges[edge1*N+0];
        t = edges[edge1*N+1];

        k = edges[edge2*N+0];
        l = edges[edge2*N+1];

        //  Randomly choose 2 nodes
        //s = floor(drand48() * N);
        //k = floor(drand48() * N);

        //  Randomly choose 1 neighbor of each
        //neighbor_s_index = floor(drand48() * degree(s));
        //neighbor_k_index = floor(drand48() * degree(k));
        //t = neighbors(s,neighbor_s_index);
        //l = neighbors(k,neighbor_k_index);

        count++;

        //  Proceed only if s != k, s != l, t != k, t != l
        if (s != k && s != l && t != k && t != l) {
            // Proceed only if the new links {s,l} and {t,k}
            // do NOT already exist
            if (A[s*N+l] == 0 && A[t*N+k] == 0) {
                // Proceed only if the link lengths fulfill condition C1
                if ((fabs(D[s*N+t] - D[k*N+t]) < eps &&
                        fabs(D[k*N+l] - D[s*N+l]) < eps ) ||
                            (fabs(D[s*N+t] - D[s*N+l]) < eps &&
                                fabs(D[k*N+l] - D[k*N+t]) < eps )) {
                    // Now rewire the links symmetrically
                    // and increase i by 1
                    A[s*N+t] = A[t*N+s] = 0;
                    A[k*N+l] = A[l*N+k] = 0;
                    A[s*N+l] = A[l*N+s] = 1;
                    A[t*N+k] = A[k*N+t] = 1;

                    edges[edge1*N+0] = s;
                    edges[edge1*N+1] = l;
                    edges[edge2*N+0] = k;
                    edges[edge2*N+1] = t;

                    //  Update neighbor lists of all 4 involved nodes
                    //neighbors(s,neighbor_s_index) = l;
                    //neighbors(k,neighbor_k_index) = t;

                    //neighbor_t_index = 0;
                    //while (neighbors(t,neighbor_t_index) != s) {
                    //    neighbor_t_index++;
                    //}
                    //neighbors(t,neighbor_t_index) = k;

                    //neighbor_l_index = 0;
                    //while (neighbors(l,neighbor_l_index) != k) {
                    //    neighbor_l_index++;
                    //}
                    //neighbors(l,neighbor_l_index) = s;

                    i++;
                }
            }
        }
    }
    printf("Trials %d, Rewirings %d", count, iterations);
}


void _randomly_rewire_geomodel_II_fast(int iterations, float eps, short *A,
    float *D, int E, int N, int *edges)  {

    int i, s, t, k, l, edge1, edge2;

    //  Initialize random number generator
    srand48(time(0));

    i = 0;
    while (i < iterations) {
        //  Randomly choose 2 edges
        edge1 = floor(drand48() * E);
        edge2 = floor(drand48() * E);

        s = edges[edge1*N+0];
        t = edges[edge1*N+1];

        k = edges[edge2*N+0];
        l = edges[edge2*N+1];

        //  Proceed only if s != k, s != l, t != k, t != l
        if (s != k && s != l && t != k && t != l) {
            // Proceed only if the new links {s,l} and {t,k}
            // do NOT already exist
            if (A[s*N+l] == 0 && A[t*N+k] == 0) {

                // Proceed only if the link lengths fulfill condition C2
                if (fabs(D[s*N+t] - D[s*N+l]) < eps &&
                        fabs(D[t*N+s] - D[t*N+k]) < eps &&
                            fabs(D[k*N+l] - D[k*N+t]) < eps &&
                                fabs(D[l*N+k] - D[l*N+s]) < eps ) {
                    // Now rewire the links symmetrically
                    // and increase i by 1
                    A[s*N+t] = A[t*N+s] = 0;
                    A[k*N+l] = A[l*N+k] = 0;
                    A[s*N+l] = A[l*N+s] = 1;
                    A[t*N+k] = A[k*N+t] = 1;

                    edges[edge1*N+0] = s;
                    edges[edge1*N+1] = l;
                    edges[edge2*N+0] = k;
                    edges[edge2*N+1] = t;

                    i++;
                }
            }
        }
    }
}
  
void _randomly_rewire_geomodel_III_fast(int iterations, float eps, short *A,
    float *D, int E, int N, int *edges, int *degree)  {

    int i, s, t, k, l, edge1, edge2;

    //  Initialize random number generator
    srand48(time(0));

    i = 0;
    while (i < iterations) {
        //  Randomly choose 2 edges
        edge1 = floor(drand48() * E);
        edge2 = floor(drand48() * E);

        s = edges[edge1*N+0];
        t = edges[edge1*N+1];

        k = edges[edge2*N+0];
        l = edges[edge2*N+1];

        //  Proceed only if s != k, s != l, t != k, t != l
        if (s != k && s != l && t != k && t != l) {
            // Proceed only if the new links {s,l} and {t,k}
            // do NOT already exist
            if (A[s*N+l] == 0 && A[t*N+k] == 0) {
                // Proceed only if degree-degree correlations
                // will not be changed
                if (degree[s] == degree[k] && degree[t] == degree[l]) {
                    // Proceed only if the link lengths
                    // fulfill condition C2
                    if (fabs(D[s*N+t] - D[s*N+l]) < eps &&
                            fabs(D[t*N+s] - D[t*N+k]) < eps &&
                                fabs(D[k*N+l] - D[k*N+t]) < eps &&
                                    fabs(D[l*N+k] - D[l*N+s]) < eps ) {
                        // Now rewire the links
                        // symmetrically and increase i by 1
                        A[s*N+t] = A[t*N+s] = 0;
                        A[k*N+l] = A[l*N+k] = 0;
                        A[s*N+l] = A[l*N+s] = 1;
                        A[t*N+k] = A[k*N+t] = 1;

                        edges[edge1*N+0] = s;
                        edges[edge1*N+1] = l;
                        edges[edge2*N+0] = k;
                        edges[edge2*N+1] = t;

                        i++;
                    }
                }
            }
        }
    }
}


double _higher_order_transitivity4_fast(int N, short *A)  {
    long cliques, stars;

    //  Initialize
    cliques = 0;
    stars = 0;
    //  Iterate over all nodes
    for (int v = 0; v < N; v++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    if (A[v*N+i] == 1 && A[v*N+j] == 1 &&
                        A[v*N+k] == 1) {
                        stars++;
                        if (A[i*N+j] == 1 && A[i*N+k] == 1 &&
                            A[j*N+k] == 1)
                            cliques++;
                    }
                }
            }
        }
    }
    double T = ((double) cliques) / ((double) stars);
    return T;
}


// network ====================================================================

void _do_nsi_hamming_clustering_fast(int n2, int nActiveIndices, float mind0,
    float minwp0, int lastunited, int part1, int part2, float *distances,
    int *theActiveIndices, float *linkedWeights, float *weightProducts,
    float *errors, float *result, int *mayJoin)  {

    
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
                    distances[c1*n2+c2] +=
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
                           - errors[c1*n2+part2] - errors[c2*n2+part2]);
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
            ECFB[i*N+j] += 2.*I/(N*(N-1));
        } // for j
    } // for i
}
