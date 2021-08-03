/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    @author Raffaele Solca
    @author Azzam Haidar
    @author Stan Tomov

    @precisions normal z -> c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"

/// Documentation at:
/// http://www.netlib.org/lapack/explore-html/d6/dae/zheevd_8f.html
/// Requires A_d to be allocated before this call
/// Does NOT copy the data back to the CPU
/// Output: A_d (will contain eigenvectors of A_h)
/// Output: eigenval (will contain the eigenvalues of A_h)
extern "C" 
void magma_zheevd_(char* jobzPtr, char* uploPtr, int* nPtr,
                magmaDoubleComplex *A_h, int* ldaPtr, double* eigenval, size_t *A_dPtr, int *lddaPtr)
{

        char jobz = *jobzPtr;
        int n = *nPtr;
        char uplo = *uploPtr;
        int lda = *ldaPtr;
        int ldda = *lddaPtr;
	A_h = A_h; //Make compiler happy

        magmaDoubleComplex *A_d = (magmaDoubleComplex*) *A_dPtr;

        //todo remove
        magma_init();                                                         
        if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {                          
                fprintf(stderr, "ERROR: cublasInit failed\n");                     
                magma_finalize();                                                  
                exit(-1);                                                          
        }                                                                      

        magmaDoubleComplex *h_R, *h_work, aux_work[1];
        double *rwork, aux_rwork[1];
        magma_int_t *iwork, aux_iwork[1];
        magma_int_t N, n2, info, lwork, lrwork, liwork;

        N = n;
        n2   = N*N;

        /* Allocate host memory for the matrix */
        h_R = (magmaDoubleComplex*) malloc( N*lda*sizeof(magmaDoubleComplex) );

        /* Query for workspace sizes */
        magma_zheevd_gpu( jobz, uplo,
                        N, A_d, ldda, eigenval,
                        h_R, lda,
                        aux_work,  -1,
                        aux_rwork, -1,
                        aux_iwork, -1,
                        &info );


        if (info != 0){
                printf("magma_zheevd_gpu returned error %d: %s %c.\n", (int) info, magma_strerror( info ), jobz);
                exit(-1);
        }

        lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
        lrwork = (magma_int_t) aux_rwork[0];
        liwork = aux_iwork[0];

        h_work = (magmaDoubleComplex*) malloc( lwork*sizeof(magmaDoubleComplex) );
        rwork = (double*) malloc(sizeof(double) * lrwork );
        iwork = (magma_int_t*) malloc(sizeof(magma_int_t) * liwork );


        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_zheevd_gpu( jobz, uplo,
                        N, A_d, ldda, eigenval,
                        h_R, lda,
                        h_work, lwork,
                        rwork, lrwork,
                        iwork, liwork,
                        &info );

        if (info != 0){
                printf("magma_zheevd_gpu returned error %d: %s %c.\n", (int) info, magma_strerror( info ), jobz);
                exit(-1);
        }

        free( rwork  );
        free( iwork  );

        free(h_work);
        free(h_R);
        magma_finalize();
}
