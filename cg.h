#ifndef CG_H_
#define CG_H_

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include "util.h"
#include "parameters.h"
#include <cblas.h>
#include <mpi.h>

void cgsolver( double *A, double *b, double *x, int m, int n );
double * init_source_term(int n, double h);
void smvm(int m, const double* val, const int* col, const int* row, const double* x, double* y);
void cgsolver_sparse( double *Aval, int *row, int *col, double *b, double *x, int n);
void cgsolver_mpi( double *A_local, double *b_local, double *x, int local_m, int n, int rank, int size );

#endif /*CG_H_*/


