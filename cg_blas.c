#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <mpi.h>


#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "cg.h"
#include "second.h"


/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

	double * A;
	double * x;
	double * b;
	double * b_local;
	double * A_local;

	double t1,t2;

// Arrays and parameters to read and store the sparse matrix
	double * val = NULL;
	int * row = NULL;
	int * col = NULL;
	int N;
	int nz;
	const char * element_type ="d"; 
	int symmetrize=1;

	struct size_m sA;
	double h;
	int m,n;

	int prank, psize;
	int m_local;
	
	// Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);	

	if(prank == 0){
	// Process 0: read the matrix from file

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	else    
	{ 
		A = read_mat(argv[1]);
		sA = get_size(argv[1]);
	}

	if (loadMMSparseMatrix(argv[1], *element_type, true, &N, &N, &nz, &val, &row, &col, symmetrize)){
		fprintf (stderr, "!!!! loadMMSparseMatrix FAILED\n");
		return EXIT_FAILURE;
	} else {
		printf("Matrix loaded from file %s\n",argv[1]);
		printf("N = %d \n",N);
		printf("nz = %d \n",nz);
		printf("val[0] = %f \n",val[0]);
	}

	printf("MPI initialized with %d processes\n", psize);

	m = sA.m;
	n = sA.n;	

	h = 1./(double)n;
	b = init_source_term(n,h);

	}

	// Broadcast matrix size to all processes
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Calculate number of rows for each process (n=m in this case)
	m_local = m / psize;
	if (prank < m % psize) m_local++;
	

	// Allocate memory for local matrix and vector on each process
	A_local = (double*) malloc(m_local * n * sizeof(double));
	b_local = (double*) malloc(m_local * sizeof(double));

	// Matrix A is stored in COLUMN-MAJOR order (index = col*m + row)
	// To send rows, we need to manually pack (or using a datatype) because rows are NOT contiguous	
	int m_i;
	
	// Calculate starting row for each process
	int *row_starts = (int*) malloc(psize * sizeof(int));
	row_starts[0] = 0;
	for (int i = 0; i < psize; i++) {
		m_i = m / psize;
		if (i < m % psize) {
			m_i++;
		} else {
			m_i = m / psize;
		}
		row_starts[i+1] = row_starts[i] + m_i;
		//printf("Process %d: row_starts[%d] = %d\n", prank, i, row_starts[i]);
	}

	
	double t0 = MPI_Wtime();

	// loop over all processes and send the matrix and vector to each process from rank zero 
	if (prank == 0) {
		
		for (int col = 0; col < n; col++) {
            memcpy(&A_local[col * m_local], &A[col * m], m_local * sizeof(double));
        }
		memcpy(b_local, b, m_local * sizeof(double));
		
		// Send to other processes - need to pack rows manually
		for (int i = 1; i < psize; i++) {
			// compute local m_local for process i
			m_i = m / psize;
			if (i < m % psize) m_i++;
			
			//Define a datatype representing the non-contiguous row block
    		MPI_Datatype column_slice_type;         
    		// Count = n (cols), Block = m_i (rows to send), Stride = m (global rows)
    		MPI_Type_vector(n, m_i, m, MPI_DOUBLE, &column_slice_type);
    		MPI_Type_commit(&column_slice_type);
			
			MPI_Send(&A[row_starts[i]], 1, column_slice_type, i, 0, MPI_COMM_WORLD);
			MPI_Send(&b[row_starts[i]], m_i, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
			
			MPI_Type_free(&column_slice_type);
		}
	} else {
		// Receive from process 0 m_local rows, already received in column-major order in A_local
		MPI_Recv(A_local, m_local * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(b_local, m_local, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
	}
	
	free(row_starts);

	// Wait for all communications to complete (only if using Isend/Irecv)
	// MPI_Waitall(k, m_requests, MPI_STATUS_IGNORE);
	
	//TODO: undertand how to use displ in scatterv
	// Do the same with MPI_Scatterv 
	// int displs_A[psize];
	// int displs_b[psize];
	// int sendcounts_A[psize];
	// int sendcounts_b[psize];

	// for (int p = 0; p < psize; p++) {
	// 	// displaced of i blocks of row_type
	// 	displs_A[p] = p ;
	// 	displs_b[p] = p * m_local;
	// 	// in unit of sendtype (so one block of row_type)
	// 	sendcounts_A[p] = 1;
	// 	sendcounts_b[p] = m_local;
	// }

	// MPI_Scatterv(A, sendcounts_A, displs_A, row_type, A_local, m_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
	// MPI_Scatterv(b, sendcounts_b, displs_b, MPI_DOUBLE, b_local, m_local, MPI_DOUBLE, 0, MPI_COMM_WORLD );

	x = (double*) malloc(n * sizeof(double));
	//initialize solution vector to zero
	memset(x, 0., n*sizeof(double));

	if (prank == 0){
	printf("Matrix and vector distributed to all processes.\n");
	printf("Call cgsolver_mpi() on matrix size (%d x %d) on process %d\n",m_local,n,prank);
	}
	t1 = second();
	cgsolver_mpi( A_local, b_local, x, m_local, n, prank, psize);
	t2 = second();
	double t_end = MPI_Wtime();

	if (prank == 0){
	printf("Time for CG (dense MPI solver)  = %f [s]\n",(t2-t1));
	printf("Total time including communication = %f [s]\n",(t_end - t0));
	}
	
	//Re-initialize solution vector to zero
	memset(x, 0., n *sizeof(double));

	if (prank == 0){
	printf("Call cgsolver() on matrix size (%d x %d)\n",m,n);
	t1 = second();
	cgsolver( A, b, x, m, n );
	t2 = second();
	printf("Time for CG (dense solver)  = %f [s]\n",(t2-t1));
	}

	//Re-initialize solution vector to zero
	memset(x, 0., n*sizeof(double));

	if (prank == 0){
	printf("Call cgsolver_sparse() on matrix size (%d x %d)\n",m,n);
	t1 = second();
	cgsolver_sparse( val, row, col, b, x, N );
	t2 = second();
	printf("Time for CG (sparse solver) = %f [s]\n",(t2-t1));
	}

	if (prank == 0){	
	free(A);
	free(b);
	free(val);
	free(row);
	free(col);
	}
	
	free(A_local);
	free(b_local);
	free(x);


	MPI_Finalize();

	return 0;
}


