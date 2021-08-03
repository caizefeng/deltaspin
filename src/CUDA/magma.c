// File: magma.c
// C/Fortran interface to MAGMA.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
// includes cuda headers
#include <cuda_runtime.h>
// includes magma headers
#include "magma.h"
// includes project headers
//#include "cuda_globals.h"

typedef size_t devptr_t;

/******************************************************/
// CUFFT wrapper for MAGMA API errors, in library

// wrapper for MAGMA errors
inline void __magma_error(magma_int_t status, char *file, int line, char *msg)
{
    if(status != cudaSuccess)
    {
        printf("\nMAGMA Error in %s, line %d: %d\n %s\n", file, line, status, msg);
        cudaDeviceReset();
        exit(-1);
    }
}
// macro for MAGMA errors
#define MAGMA_ERROR(status, msg)  __magma_error( status, __FILE__, __LINE__, msg )

/******************************************************/
// C wrappers for MAGMA, in VASP

// magma_init wrapper function
void magma_init_()
{
    MAGMA_ERROR( magma_init(), "Failed to call magma_init!");
}

// magma_finalize wrapper function
void magma_finalize_()
{
    MAGMA_ERROR( magma_finalize(), "Failed to call magma_finalize!");
}

// magma_zpotrf wrapper function
void magma_zpotrf_(char *uplo, int *n, magmaDoubleComplex *a, int *lda, int *info)
{
    MAGMA_ERROR( magma_zpotrf(uplo[0], *n, a, *lda, info), "Failed to call magm_zpotrf!" );
}

// magma_zhegvd wrapper function
void magma_zhegvd_(int *itype, char *jobz, char *uplo, int *n,
     magmaDoubleComplex *a, int *lda, magmaDoubleComplex *b, int *ldb, double *w, int *info)
{
    magmaDoubleComplex *work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,lrwork,liwork;

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zhegvd(*itype,jobz[0],uplo[0],*n,a,*lda,b,*ldb,w,
		 work_query,-1,rwork_query,-1,iwork_query,-1,info),
		 "Failed to call magma_zhegvd!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    lrwork = (magma_int_t)rwork_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_zhegvd
    MAGMA_ERROR( magma_zhegvd(*itype,jobz[0],uplo[0],*n,a,*lda,b,*ldb,w,
		 work,lwork,rwork,lrwork,iwork,liwork,info), "Failed to call magma_zhegvd!" );

    // free workspace arrays
    free(work); free(rwork); free(iwork);
}

// magma_zhegvdx wrapper function
void magma_zhegvdx_(int *itype, char *jobz, char *range, char *uplo, int *n,
     magmaDoubleComplex *a, int *lda, magmaDoubleComplex *b, int *ldb,
     double *vl, double *vu, int *il, int *iu, int *m, double *w, int *info)
{
    magmaDoubleComplex *work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,lrwork,liwork;

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zhegvdx(*itype,jobz[0],range[0],uplo[0],*n,a,*lda,b,*ldb,
		 *vl,*vu,*il,*iu,m,w,work_query,-1,rwork_query,-1,iwork_query,-1,info),
                 "Failed to call magma_zhegvdx!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    lrwork = (magma_int_t)rwork_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_zhegvdx
    MAGMA_ERROR( magma_zhegvdx(*itype,jobz[0],range[0],uplo[0],*n,a,*lda,b,*ldb,
		 *vl,*vu,*il,*iu,m,w,work,lwork,rwork,lrwork,iwork,liwork,info),
		 "Failed to call magma_zhegvdx!" );

    // free workspace arrays
    free(work); free(rwork); free(iwork);
}

// magma_dsygvd wrapper function
void magma_dsygvd_(int *itype, char *jobz, char *uplo, int *n,
     double *a, int *lda, double *b, int *ldb, double *w, int *info)
{
    double *work,work_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,liwork;

    // query for workspace array dimensions
    MAGMA_ERROR( magma_dsygvd(*itype,jobz[0],uplo[0],*n,a,*lda,b,*ldb,w,
                 work_query,-1,iwork_query,-1,info), "Failed to call magma_dsygvd!" );

    // set workspace array dimensions
    lwork = (magma_int_t)work_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    work = (double*)malloc(lwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_dsygvd
    MAGMA_ERROR( magma_dsygvd(*itype,jobz[0],uplo[0],*n,a,*lda,b,*ldb,w,
                 work,lwork,iwork,liwork,info), "Failed to call magma_dsygvd!" );

    // free workspace arrays
    free(work); free(iwork);
}

// magma_zheevd wrapper function
void magma_zheevd_(char *jobz, char *uplo, int *n,
     magmaDoubleComplex *a, int *lda, double *w, int *info)
{
    magmaDoubleComplex *work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,lrwork,liwork;

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zheevd(jobz[0],uplo[0],*n,a,*lda,w,
                 work_query,-1,rwork_query,-1,iwork_query,-1,info),
		 "Failed to call magma_zheevd!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    lrwork = (magma_int_t)rwork_query[0];
    liwork = (magma_int_t)iwork_query[0];    

    // allocate workspace arrays
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_dsygvd
    MAGMA_ERROR( magma_zheevd(jobz[0],uplo[0],*n,a,*lda,w,
                 work,lwork,rwork,lrwork,iwork,liwork,info),
		 "Failed to call magma_zheevd!" );

    // free workspace arrays
    free(work); free(rwork); free(iwork);
}

// magma_zheevd_gpu wrapper function
void magma_zheevd_gpu_(char *jobz, char *uplo, int *n,
     devptr_t *devptr_a, int *lda, double *w, int *info)
{
    magmaDoubleComplex *wa,*work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,lrwork,liwork;

    // device pointers
    magmaDoubleComplex *a = (magmaDoubleComplex*)(*devptr_a);

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zheevd_gpu(jobz[0],uplo[0],*n,a,*lda,w,
                 wa,*lda,work_query,-1,rwork_query,-1,iwork_query,-1,info),
                 "Failed to call magma_zheevd_gpu!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    lrwork = (magma_int_t)rwork_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    wa = (magmaDoubleComplex*)malloc((*n)*(*lda)*sizeof(magmaDoubleComplex));
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_zheevd
    MAGMA_ERROR( magma_zheevd_gpu(jobz[0],uplo[0],*n,a,*lda,w,
                 wa,*lda,work,lwork,rwork,lrwork,iwork,liwork,info),
                 "Failed to call magma_zheevd_gpu!" );

    // free workspace arrays
    free(wa); free(work); free(rwork); free(iwork);
}

// magma_dsyevd_gpu wrapper function
void magma_dsyevd_gpu_(char *jobz, char *uplo, int *n,
     devptr_t *devptr_a, int *lda, double *w, int *info)
{
    double *wa,*work,work_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork,lrwork,liwork;

    // device pointers
    double *a = (double*)(*devptr_a);

    // query for workspace array dimensions
    MAGMA_ERROR( magma_dsyevd_gpu(jobz[0],uplo[0],*n,a,*lda,w,
                 wa,*lda,work_query,-1,iwork_query,-1,info),
                 "Failed to call magma_dsyevd_gpu!" );

    // set workspace array dimensions
    lwork = (magma_int_t)work_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    wa = (double*)malloc((*n)*(*lda)*sizeof(double));
    work = (double*)malloc(lwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_dsyevd
    MAGMA_ERROR( magma_dsyevd_gpu(jobz[0],uplo[0],*n,a,*lda,w,
                 wa,*lda,work,lwork,iwork,liwork,info),
                 "Failed to call magma_dsyevd_gpu!" );

    // free workspace arrays
    free(wa); free(work); free(iwork);
}

// magma_zheevx_gpu wrapper function
void magma_zheevx_gpu_(char *jobz, char *range, char *uplo, int *n,
     devptr_t *devptr_a, int *lda, double *vl, double *vu, int *il, int *iu,
     double *abstol, int *m, double *w, devptr_t *devptr_z, int *ldz, int *ifail, int *info)
{
    magmaDoubleComplex *wa,*wz,*work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork;

    // device pointers
    magmaDoubleComplex *a = (magmaDoubleComplex*)(*devptr_a);
    magmaDoubleComplex *z = (magmaDoubleComplex*)(*devptr_z);

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zheevx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,
		 *vl,*vu,*il,*iu,*abstol,m,w,z,*ldz,wa,*lda,wz,*ldz,
		 work_query,-1,rwork_query,iwork_query,ifail,info),
                 "Failed to call magma_zheevx_gpu!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    // workspace arrays rwork and iwork have dimensions 7N and 5N

    // allocate workspace arrays
    wa = (magmaDoubleComplex*)malloc((*n)*(*lda)*sizeof(magmaDoubleComplex));
    wz = (magmaDoubleComplex*)malloc((*n)*(*ldz)*sizeof(magmaDoubleComplex));
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(7*(*n)*sizeof(double));
    iwork = (magma_int_t*)malloc(5*(*n)*sizeof(magma_int_t));

    // call to magma_zheevd
    MAGMA_ERROR( magma_zheevx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,
                 *vl,*vu,*il,*iu,*abstol,m,w,z,*ldz,wa,*lda,wz,*ldz,
                 work,lwork,rwork,iwork,ifail,info),
                 "Failed to call magma_zheevx_gpu!" );

    // free workspace arrays
    free(wa); free(wz); free(work); free(rwork); free(iwork);
}

// magma_zheevdx_gpu wrapper function
void magma_zheevdx_gpu_(char *jobz, char *range, char *uplo, int *n,
     devptr_t *devptr_a, int *lda, double *vl, double *vu, int *il, int *iu,
     int *m, double *w, int *info)
{
    magmaDoubleComplex *wa,*work,work_query[1];
    double *rwork,rwork_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork, lrwork, liwork;

    // device pointers
    magmaDoubleComplex *a = (magmaDoubleComplex*)(*devptr_a);

    // query for workspace array dimensions
    MAGMA_ERROR( magma_zheevdx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,*vl,*vu,*il,*iu,
		 m,w,wa,*lda,work_query,-1,rwork_query,-1,iwork_query,-1,info),
                 "Failed to call magma_zheevdx_gpu!" );

    // set workspace array dimensions
    lwork = (magma_int_t)MAGMA_Z_REAL(work_query[0]);
    lrwork = (magma_int_t)rwork_query[0];
    liwork = (magma_int_t)iwork_query[0];

    // allocate workspace arrays
    wa = (magmaDoubleComplex*)malloc((*n)*(*lda)*sizeof(magmaDoubleComplex));
    work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_zheevdx_gpu
    MAGMA_ERROR( magma_zheevdx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,*vl,*vu,*il,*iu,
                 m,w,wa,*lda,work,lwork,rwork,lrwork,iwork,liwork,info),
                 "Failed to call magma_zheevdx_gpu!" );

    // free workspace arrays
    free(wa); free(work); free(rwork); free(iwork);
}

// magma_dsyevdx_gpu wrapper function
void magma_dsyevdx_gpu_(char *jobz, char *range, char *uplo, int *n,
     devptr_t *devptr_a, int *lda, double *vl, double *vu, int *il, int *iu,
     int *m, double *w, int *info)
{
    double *wa,*work,work_query[1];
    magma_int_t *iwork, iwork_query[1];
    magma_int_t lwork, liwork;

    // device pointers
    double *a = (double*)(*devptr_a);

    // query for workspace array dimensions
    MAGMA_ERROR( magma_dsyevdx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,
                 *vl,*vu,*il,*iu,m,w,wa,*lda,work_query,-1,iwork_query,-1,info),
                 "Failed to call magma_dsyevdx_gpu!" );

    // set workspace array dimensions
    lwork = (magma_int_t)work_query[0];
    liwork = (magma_int_t)iwork_query[0];
    
    // allocate workspace arrays
    wa = (double*)malloc((*n)*(*lda)*sizeof(double));
    work = (double*)malloc(lwork*sizeof(double));
    iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    // call to magma_dsyevdx_gpu
    MAGMA_ERROR( magma_dsyevdx_gpu(jobz[0],range[0],uplo[0],*n,a,*lda,
                 *vl,*vu,*il,*iu,m,w,wa,*lda,work,lwork,iwork,liwork,info),
                 "Failed to call magma_dsyevdx_gpu!" );

    // free workspace arrays
    free(wa); free(work); free(iwork);
}

/******************************************************/
