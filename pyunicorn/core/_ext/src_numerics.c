/*
* !/usr/bin/python
* -*- coding: utf-8 -*-
*
* This file is part of pyunicorn.
* Copyright (C) 2008--2017 Jonathan F. Donges and pyunicorn authors
* URL: <http://www.pik-potsdam.de/members/donges/software>
* License: BSD (3-clause)
*/

// geo_network ================================================================

void _randomly_rewire_geomodel_I_fast(int iterations, float eps, short *A,
    float *D, int E, int N, int *edges)  {

    int i, j, s, t, k, l, edge1, edge2, count;
    int neighbor_s_index, neighbor_t_index;
    int neighbor_k_index, neighbor_l_index;

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
