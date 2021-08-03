/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/*
 * This is the private header file used by the CUBLAS library internally
 */

#if !defined(CUBLAS_P_H_)
#define CUBLAS_P_H_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined (__GNUC__)
#include <stdint.h>
#endif

#include "cublas.h"
#include "cuComplex.h"
#include "math_constants.h"

/* CUBLAS context */
struct cublasContext {
    int cublasIsInitialized;
    cublasStatus cublasLastError;
};
/*
#ifdef EMULATION

union __cudart_DoubleInthiloCvt0 {
    double     d;
    signed int i[2];
};

__device__ double __hiloint2double0(int a, int b)
{
  volatile union __cudart_DoubleInthiloCvt0 cvt;
  
  cvt.i[0] = b;
  cvt.i[1] = a;
  return cvt.d;
}

__device__ double __hiloint2double(int a, int b)
{
  volatile union __cudart_DoubleInthiloCvt0 cvt;
  
  cvt.i[0] = b;
  cvt.i[1] = a;
  return cvt.d;
}

#endif

static __inline__ __device__ double fetch_double(texture<uint2, 1> t, int i)
{
    uint2 v = tex1Dfetch(t,i);
#ifdef EMULATION
    return __hiloint2double0(v.y, v.x);
#else
    return __hiloint2double(v.y, v.x);
#endif
}
*/

/* the next three macro definitions trigger
 * code generation when tlsHook.h is included
 */ 
#define __tlsHookIdentifier cublasThreadContext
#define __tlsHookType       struct cublasContext
#define __tlsHookExtern
//#include <tlshook.h>

//#define CUBLAS_GET_CTX() __tlsHookInitTlsValueForcublasThreadContext(cublasInitCtx, cublasShutDownCtx)

#define CUBLAS_THREAD_BUNDLE_SIZE        (32)
#define CUBLAS_CTA_MAX_DIM               (65535)
#define CUBLAS_FASTIMUL_F_MAX_DIM        (2000) /* float */
#define CUBLAS_FASTIMUL_D_MAX_DIM        (1410) /* double, complex */
#define CUBLAS_SMALL_SGEMM_MAT_MAX_ELEMS (6400)
#define CUBLAS_WORD_ALIGN                (64)   /* alignment for 32-bit word */
#define CUBLAS_LONG_ALIGN                (128)  /* alignment for 64-bit long */
#define CUBLAS_1DBUF_ALIGN               (256)  /* alignment for 1D buffer */
#define CUBLAS_MAX_1DBUF_SIZE            ((1<<27)-CUBLAS_1DBUF_ALIGN)

#define CUBLAS_SAXPY_CTAS_MIN           (1)
#define CUBLAS_SAXPY_CTAS_MAX           (80)
#define CUBLAS_SAXPY_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SAXPY_THREAD_MAX         (128)

#define CUBLAS_SCOPY_CTAS_MIN           (1)
#define CUBLAS_SCOPY_CTAS_MAX           (80)
#define CUBLAS_SCOPY_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SCOPY_THREAD_MAX         (128)

#define CUBLAS_SSCAL_CTAS_MIN           (1)
#define CUBLAS_SSCAL_CTAS_MAX           (96)
#define CUBLAS_SSCAL_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SSCAL_THREAD_MAX         (128)

#define CUBLAS_SSWAP_CTAS_MIN           (1)
#define CUBLAS_SSWAP_CTAS_MAX           (80)
#define CUBLAS_SSWAP_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SSWAP_THREAD_MAX         (128)

#define CUBLAS_SROT_CTAS_MIN            (1)
#define CUBLAS_SROT_CTAS_MAX            (64)
#define CUBLAS_SROT_THREAD_MIN          (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SROT_THREAD_MAX          (128)

#define CUBLAS_CSROT_CTAS_MIN           (1)
#define CUBLAS_CSROT_CTAS_MAX           (64)
#define CUBLAS_CSROT_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CSROT_THREAD_MAX         (128)

#define CUBLAS_CROT_CTAS_MIN            (1)
#define CUBLAS_CROT_CTAS_MAX            (64)
#define CUBLAS_CROT_THREAD_MIN          (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CROT_THREAD_MAX          (128)

#define CUBLAS_SROTM_CTAS_MIN           (1)
#define CUBLAS_SROTM_CTAS_MAX           (64)
#define CUBLAS_SROTM_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_SROTM_THREAD_MAX         (128)
#define CUBLAS_SROTM_PARAM_VEC_LEN      (5)

#define CUBLAS_SDOT_LOG_THREAD_COUNT    (7)
#define CUBLAS_SDOT_THREAD_COUNT        (1 << CUBLAS_SDOT_LOG_THREAD_COUNT)
#define CUBLAS_SDOT_CTAS                (80)

#define CUBLAS_SASUM_LOG_THREAD_COUNT   (7)
#define CUBLAS_SASUM_THREAD_COUNT       (1 << CUBLAS_SASUM_LOG_THREAD_COUNT)
#define CUBLAS_SASUM_CTAS               (96)

#define CUBLAS_ISAMAX_LOG_THREAD_COUNT  (7)
#define CUBLAS_ISAMAX_THREAD_COUNT      (1 << CUBLAS_ISAMAX_LOG_THREAD_COUNT)
#define CUBLAS_ISAMAX_CTAS              (80)

#define CUBLAS_ICAMAX_LOG_THREAD_COUNT  (7)
#define CUBLAS_ICAMAX_THREAD_COUNT      (1 << CUBLAS_ICAMAX_LOG_THREAD_COUNT)
#define CUBLAS_ICAMAX_CTAS              (80)

#define CUBLAS_ISAMIN_LOG_THREAD_COUNT  (7)
#define CUBLAS_ISAMIN_THREAD_COUNT      (1 << CUBLAS_ISAMIN_LOG_THREAD_COUNT)
#define CUBLAS_ISAMIN_CTAS              (80)

#define CUBLAS_ICAMIN_LOG_THREAD_COUNT  (7)
#define CUBLAS_ICAMIN_THREAD_COUNT      (1 << CUBLAS_ICAMIN_LOG_THREAD_COUNT)
#define CUBLAS_ICAMIN_CTAS              (80)

#define CUBLAS_SNRM2_LOG_THREAD_COUNT   (7)
#define CUBLAS_SNRM2_THREAD_COUNT       (1 << CUBLAS_SNRM2_LOG_THREAD_COUNT)
#define CUBLAS_SNRM2_CTAS               (64)

#define CUBLAS_SCNRM2_LOG_THREAD_COUNT  (7)
#define CUBLAS_SCNRM2_THREAD_COUNT      (1 << CUBLAS_SCNRM2_LOG_THREAD_COUNT)
#define CUBLAS_SCNRM2_CTAS              (64)

#define CUBLAS_SGEMM_LOG_LARGE_THREAD_COUNT   (9)
#define CUBLAS_SGEMM_LOG_SMALL_THREAD_COUNT   (8)
#define CUBLAS_SGEMM_LARGE_THREAD_COUNT (1 << CUBLAS_SGEMM_LOG_LARGE_THREAD_COUNT)
#define CUBLAS_SGEMM_SMALL_THREAD_COUNT (1 << CUBLAS_SGEMM_LOG_SMALL_THREAD_COUNT)
#define CUBLAS_SGEMM_GRIDW_LOG          (2)
#define CUBLAS_SGEMM_GRIDW              (1 << CUBLAS_SGEMM_GRIDW_LOG)
#define CUBLAS_SGEMM_GRIDH_LOG          (2)
#define CUBLAS_SGEMM_GRIDH              (1 << CUBLAS_SGEMM_GRIDH_LOG)

#define CUBLAS_SSYMM_LOG_THREAD_COUNT   (9)
#define CUBLAS_SSYMM_THREAD_COUNT       (1 << CUBLAS_SSYMM_LOG_THREAD_COUNT)
#define CUBLAS_SSYMM_GRIDW_LOG          (2)
#define CUBLAS_SSYMM_GRIDW              (1 << CUBLAS_SSYMM_GRIDW_LOG)
#define CUBLAS_SSYMM_GRIDH_LOG          (2)
#define CUBLAS_SSYMM_GRIDH              (1 << CUBLAS_SSYMM_GRIDH_LOG)

#define CUBLAS_SSYRK_LOG_THREAD_COUNT   (9)
#define CUBLAS_SSYRK_THREAD_COUNT       (1 << CUBLAS_SSYRK_LOG_THREAD_COUNT)
#define CUBLAS_SSYRK_GRIDW_LOG          (2)
#define CUBLAS_SSYRK_GRIDW              (1 << CUBLAS_SSYRK_GRIDW_LOG)
#define CUBLAS_SSYRK_GRIDH_LOG          (2)
#define CUBLAS_SSYRK_GRIDH              (1 << CUBLAS_SSYRK_GRIDH_LOG)

#define CUBLAS_SSYR2K_LOG_THREAD_COUNT  (9)
#define CUBLAS_SSYR2K_THREAD_COUNT      (1 << CUBLAS_SSYR2K_LOG_THREAD_COUNT)
#define CUBLAS_SSYR2K_GRIDW_LOG         (2)
#define CUBLAS_SSYR2K_GRIDW             (1 << CUBLAS_SSYR2K_GRIDW_LOG)
#define CUBLAS_SSYR2K_GRIDH_LOG         (2)
#define CUBLAS_SSYR2K_GRIDH             (1 << CUBLAS_SSYR2K_GRIDH_LOG)

#define CUBLAS_SSYMV_LOG_THREAD_COUNT   (7)
#define CUBLAS_SSYMV_THREAD_COUNT       (1 << CUBLAS_SSYMV_LOG_THREAD_COUNT)
#define CUBLAS_SSYMV_CTAS               (64)

#define CUBLAS_SGBMV_LOG_THREAD_COUNT   (7)
#define CUBLAS_SGBMV_THREAD_COUNT       (1 << CUBLAS_SGBMV_LOG_THREAD_COUNT)
#define CUBLAS_SGBMV_CTAS               (64)

#define CUBLAS_SSBMV_LOG_THREAD_COUNT   (7)
#define CUBLAS_SSBMV_THREAD_COUNT       (1 << CUBLAS_SSBMV_LOG_THREAD_COUNT)
#define CUBLAS_SSBMV_CTAS               (64)

#define CUBLAS_SSPMV_LOG_THREAD_COUNT   (7)
#define CUBLAS_SSPMV_THREAD_COUNT       (1 << CUBLAS_SSPMV_LOG_THREAD_COUNT)
#define CUBLAS_SSPMV_CTAS               (64)

#define CUBLAS_SGEMVN_LOG_THREAD_COUNT  (7)
#define CUBLAS_SGEMVN_THREAD_COUNT      (1 << CUBLAS_SGEMVN_LOG_THREAD_COUNT)
#define CUBLAS_SGEMVN_CTAS              (64)

#define CUBLAS_SGEMVT_LOG_THREAD_COUNT  (7)
#define CUBLAS_SGEMVT_THREAD_COUNT      (1 << CUBLAS_SGEMVT_LOG_THREAD_COUNT)
#define CUBLAS_SGEMVT_CTAS              (64)

#define CUBLAS_STRSM_LOG_THREAD_COUNT   (9)
#define CUBLAS_STRSM_THREAD_COUNT       (1 << CUBLAS_STRSM_LOG_THREAD_COUNT)
#define CUBLAS_STRSM_CTAS               (16)

#define CUBLAS_STRMM_LOG_THREAD_COUNT   (9)
#define CUBLAS_STRMM_THREAD_COUNT       (1 << CUBLAS_STRMM_LOG_THREAD_COUNT)
#define CUBLAS_STRMM_CTAS               (16)

#define CUBLAS_SSYR_LOG_THREAD_COUNT    (9)
#define CUBLAS_SSYR_THREAD_COUNT        (1 << CUBLAS_SSYR_LOG_THREAD_COUNT)
#define CUBLAS_SSYR_GRIDW_LOG           (2)
#define CUBLAS_SSYR_GRIDW               (1 << CUBLAS_SSYR_GRIDW_LOG)
#define CUBLAS_SSYR_GRIDH_LOG           (2)
#define CUBLAS_SSYR_GRIDH               (1 << CUBLAS_SSYR_GRIDH_LOG)

#define CUBLAS_SSPR_LOG_THREAD_COUNT    (9)
#define CUBLAS_SSPR_THREAD_COUNT        (1 << CUBLAS_SSPR_LOG_THREAD_COUNT)
#define CUBLAS_SSPR_GRIDW_LOG           (2)
#define CUBLAS_SSPR_GRIDW               (1 << CUBLAS_SSPR_GRIDW_LOG)
#define CUBLAS_SSPR_GRIDH_LOG           (2)
#define CUBLAS_SSPR_GRIDH               (1 << CUBLAS_SSPR_GRIDH_LOG)

#define CUBLAS_SGER_LOG_THREAD_COUNT    (8)
#define CUBLAS_SGER_THREAD_COUNT        (1 << CUBLAS_SGER_LOG_THREAD_COUNT)
#define CUBLAS_SGER_GRIDW_LOG           (2)
#define CUBLAS_SGER_GRIDW               (1 << CUBLAS_SGER_GRIDW_LOG)
#define CUBLAS_SGER_GRIDH_LOG           (2)
#define CUBLAS_SGER_GRIDH               (1 << CUBLAS_SGER_GRIDH_LOG)

#define CUBLAS_SSYR2_LOG_THREAD_COUNT   (9)
#define CUBLAS_SSYR2_THREAD_COUNT       (1 << CUBLAS_SSYR2_LOG_THREAD_COUNT)
#define CUBLAS_SSYR2_GRIDW_LOG          (2)
#define CUBLAS_SSYR2_GRIDW              (1 << CUBLAS_SSYR2_GRIDW_LOG)
#define CUBLAS_SSYR2_GRIDH_LOG          (2)
#define CUBLAS_SSYR2_GRIDH              (1 << CUBLAS_SSYR2_GRIDH_LOG)

#define CUBLAS_SSPR2_LOG_THREAD_COUNT   (9)
#define CUBLAS_SSPR2_THREAD_COUNT       (1 << CUBLAS_SSPR2_LOG_THREAD_COUNT)
#define CUBLAS_SSPR2_GRIDW_LOG          (2)
#define CUBLAS_SSPR2_GRIDW              (1 << CUBLAS_SSPR2_GRIDW_LOG)
#define CUBLAS_SSPR2_GRIDH_LOG          (2)
#define CUBLAS_SSPR2_GRIDH              (1 << CUBLAS_SSPR2_GRIDH_LOG)

#define CUBLAS_STRSV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STRSV_THREAD_COUNT       (1 << CUBLAS_STRSV_LOG_THREAD_COUNT)
#define CUBLAS_STRSV_CTAS               (1)
#define CUBLAS_STRSV_MAX_DIM            (4070)

#define CUBLAS_STPSV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STPSV_THREAD_COUNT       (1 << CUBLAS_STPSV_LOG_THREAD_COUNT)
#define CUBLAS_STPSV_CTAS               (1)
#define CUBLAS_STPSV_MAX_DIM            (4070)

#define CUBLAS_STBSV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STBSV_THREAD_COUNT       (1 << CUBLAS_STBSV_LOG_THREAD_COUNT)
#define CUBLAS_STBSV_CTAS               (1)
#define CUBLAS_STBSV_MAX_DIM            (4070)

#define CUBLAS_STRMV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STRMV_THREAD_COUNT       (1 << CUBLAS_STRMV_LOG_THREAD_COUNT)
#define CUBLAS_STRMV_CTAS               (1)
#define CUBLAS_STRMV_MAX_DIM            (4070)

#define CUBLAS_STBMV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STBMV_THREAD_COUNT       (1 << CUBLAS_STRMV_LOG_THREAD_COUNT)
#define CUBLAS_STBMV_CTAS               (1)
#define CUBLAS_STBMV_MAX_DIM            (4070)

#define CUBLAS_STPMV_LOG_THREAD_COUNT   (9)
#define CUBLAS_STPMV_THREAD_COUNT       (1 << CUBLAS_STPMV_LOG_THREAD_COUNT)
#define CUBLAS_STPMV_CTAS               (1)
#define CUBLAS_STPMV_MAX_DIM            (4070)

#define CUBLAS_CAXPY_CTAS_MIN           (1)
#define CUBLAS_CAXPY_CTAS_MAX           (80)
#define CUBLAS_CAXPY_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CAXPY_THREAD_MAX         (128)

#define CUBLAS_CCOPY_CTAS_MIN           (1)
#define CUBLAS_CCOPY_CTAS_MAX           (80)
#define CUBLAS_CCOPY_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CCOPY_THREAD_MAX         (128)

#define CUBLAS_CSCAL_CTAS_MIN           (1)
#define CUBLAS_CSCAL_CTAS_MAX           (96)
#define CUBLAS_CSCAL_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CSCAL_THREAD_MAX         (128)

#define CUBLAS_CSSCAL_CTAS_MIN          (1)
#define CUBLAS_CSSCAL_CTAS_MAX          (96)
#define CUBLAS_CSSCAL_THREAD_MIN        (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CSSCAL_THREAD_MAX        (128)

#define CUBLAS_CSWAP_CTAS_MIN           (1)
#define CUBLAS_CSWAP_CTAS_MAX           (80)
#define CUBLAS_CSWAP_THREAD_MIN         (CUBLAS_THREAD_BUNDLE_SIZE)
#define CUBLAS_CSWAP_THREAD_MAX         (128)

#define CUBLAS_CDOTU_LOG_THREAD_COUNT   (7)
#define CUBLAS_CDOTU_THREAD_COUNT       (1 << CUBLAS_CDOTU_LOG_THREAD_COUNT)
#define CUBLAS_CDOTU_CTAS               (64)

#define CUBLAS_CDOTC_LOG_THREAD_COUNT   (7)
#define CUBLAS_CDOTC_THREAD_COUNT       (1 << CUBLAS_CDOTC_LOG_THREAD_COUNT)
#define CUBLAS_CDOTC_CTAS               (64)

#define CUBLAS_SCASUM_LOG_THREAD_COUNT  (7)
#define CUBLAS_SCASUM_THREAD_COUNT      (1 << CUBLAS_SASUM_LOG_THREAD_COUNT)
#define CUBLAS_SCASUM_CTAS              (96)

#define CUBLAS_CGEMM_LOG_THREAD_COUNT   (8)
#define CUBLAS_CGEMM_THREAD_COUNT       (1 << CUBLAS_CGEMM_LOG_THREAD_COUNT)
#define CUBLAS_CGEMM_GRIDW_LOG          (2)
#define CUBLAS_CGEMM_GRIDW              (1 << CUBLAS_CGEMM_GRIDW_LOG)
#define CUBLAS_CGEMM_GRIDH_LOG          (2)
#define CUBLAS_CGEMM_GRIDH              (1 << CUBLAS_CGEMM_GRIDH_LOG)

struct cublasSaxpyParams {
    const float *sx;
    float *sy;
    int   n;
    float sa;
    int   incx;
    int   incy;
    int   texXOfs;
    int   texYOfs;
};

struct cublasScopyParams {
    const float *sx;
    float *sy;
    int   n;
    int   incx;
    int   incy;
    int   texXOfs;
};

struct cublasSswapParams {
    float *sx;
    float *sy;
    int   n;
    int   incx;
    int   incy;
    int   texXOfs;
    int   texYOfs;
};

struct cublasSscalParams {
    float *sx;
    int   n;
    float sa;
    int   incx;
    int   texXOfs;
};

struct cublasSasumParams {
    const float *sx;
    float *result;
    int   n;
    int   incx;
    int   texXOfs;    
};

struct cublasIsamaxParams {
    const float *sx;
    float *resMax;
    int   *resPos;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasIcamaxParams {
    const cuComplex *cx;
    float *resMax;
    int   *resPos;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasIsaminParams {
    const float *sx;
    float *resMin;
    int   *resPos;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasIcaminParams {
    const cuComplex *cx;
    float *resMin;
    int   *resPos;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasDcontributionParams {
    const  cuDoubleComplex *W1_CR;
    const  cuDoubleComplex *W2_CR;
    const  double *W3_CR;
    const  cuDoubleComplex *SVz;
    const  double *SVd;
    cuDoubleComplex *result;
    cuDoubleComplex *resultM;
    int    n;
    int    is_real;
    int    ISPINOR;
    int    ISPINOR_;
    int    CSTE;
    int    texXOfs;
    int    texYOfs;
};

struct cublasSdotParams {
    const float *sx;
    const float *sy;
    float *result;
    int   n;
    int   incx;
    int   incy;
    int   texXOfs;
    int   texYOfs;
};

struct cublasCdotcParams {
    const cuComplex *cx;
    const cuComplex *cy;
    cuComplex *result;
    int   n;
    int   incx;
    int   incy;
    int   texXOfs;
    int   texYOfs;
};

struct cublasCdotuParams {
    const cuComplex *cx;
    const cuComplex *cy;
    cuComplex *result;
    int   n;
    int   incx;
    int   incy;
    int   texXOfs;
    int   texYOfs;
};

struct cublasSnrm2Params {
    const float *sx;
    float *result;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasScnrm2Params {
    const cuComplex *cx;
    float *result;
    int   n;
    int   incx;
};

struct cublasSrotParams {
    float *sx;
    float *sy;
    int   n;
    int   incx;
    int   incy;
    float sc;
    float ss;
    int   texXOfs;
    int   texYOfs;
};

struct cublasCsrotParams {
    cuComplex *cx;
    cuComplex *cy;
    int   n;
    int   incx;
    int   incy;
    float sc;
    float ss;
};

struct cublasCrotParams {
    cuComplex cs;
    cuComplex *cx;
    cuComplex *cy;
    int   n;
    int   incx;
    int   incy;
    float sc;
};

struct cublasSrotmParams {
    float *sx;                 
    float *sy;                  
    int   n;                   
    int   incx;                 
    int   incy;   
    int   texXOfs;
    int   texYOfs;
    float sparams[CUBLAS_SROTM_PARAM_VEC_LEN]; 
};

struct cublasSgemmParams {
    const float *A;
    const float *B;
    float *C;
    unsigned int   transa;
    unsigned int   transb;
    unsigned int   m;
    unsigned int   n;
    unsigned int   k;
    float alpha;
    unsigned int   lda;
    unsigned int   ldb;
    float beta;
    unsigned int   ldc;
    int texAOfs;
    int texBOfs;
};

struct cublasSsymmParams {
    const float *A;
    const float *B;
    float *C;
    unsigned int   lside;
    unsigned int   upper;
    unsigned int   m;
    unsigned int   n;
    unsigned int   k;
    float alpha;
    unsigned int   lda;
    unsigned int   ldb;
    float beta;
    unsigned int   ldc;
};

struct cublasSsyrkParams {
    const float *A;
    const float *B;
    float *C;
    int   upper;
    int   transpose;
    int   n;
    int   k;
    float alpha;
    int   lda;
    int   ldb;
    float beta;
    int   ldc;
};

struct cublasStrsmParams {
    const float *A;
    float *B;
    int   lside;
    int   upper;
    int   notrans;
    int   nounit;
    int   m;
    int   n;
    float alpha;
    int   lda;
    int   ldb;
};

struct cublasStrmmParams {
    const float *A;
    float *B;
    int   lside;
    int   upper;
    int   notrans;
    int   unit;
    int   m;
    int   n;
    float alpha;
    int   lda;
    int   ldb;
};

struct cublasSgemvParams {
    const float *A;
    const float *x;
    float *y;
    int   m;
    int   n;
    float alpha;
    int   lda;
    int   incx;
    float beta;
    int   incy;
};

struct cublasSsymvParams {
    const float *A;
    const float *x;
    float *y;
    int   up;
    int   n;
    float alpha;
    int   lda;
    int   incx;
    float beta;
    int   incy;
};

struct cublasSgbmvParams {
    const float *A;
    const float *x;
    float *y;
    int   trans;
    int   m;
    int   n;
    int   kl;
    int   ku;
    float alpha;
    int   lda;
    int   incx;
    float beta;
    int   incy;
};

struct cublasSspmvParams {
    const float *AP;
    const float *x;
    float *y;
    int   up;
    int   n;
    float alpha;
    int   incx;
    float beta;
    int   incy;
};

struct cublasStpmvParams {
    const float *AP;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   incx;
};

struct cublasSsyrParams {
    const float *x;
    float *A;
    int   up;
    int   n;
    float alpha;
    int   incx;
    int   lda;
};

struct cublasSsprParams {
    const float *x;
    float *AP;
    int   up;
    int   n;
    float alpha;
    int   incx;
};

struct cublasSgerParams {
    const float *x;
    const float *y;
    float *A;
    int   m;
    int   n;
    float alpha;
    int   incx;
    int   incy;
    int   lda;
};

struct cublasSsyr2Params {
    const float *x;
    const float *y;
    float *A;
    int   up;
    int   m;
    int   n;
    float alpha;
    int   incx;
    int   incy;
    int   lda;
};

struct cublasSspr2Params {
    const float *x;
    const float *y;
    float *AP;
    int   up;
    int   m;
    int   n;
    float alpha;
    int   incx;
    int   incy;
};

struct cublasStrsvParams {
    const float *A;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   lda;
    int   incx;
};

struct cublasStpsvParams {
    const float *AP;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   incx;
};

struct cublasStbsvParams {
    const float *A;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   k;
    int   lda;
    int   incx;
};

struct cublasStrmvParams {
    const float *A;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   lda;
    int   incx;
};

struct cublasStbmvParams {
    const float *A;
    float *x;
    int   up;
    int   trans;
    int   unit;
    int   n;
    int   k;
    int   lda;
    int   incx;
};

struct cublasSsbmvParams {
    const float *A;
    const float *x;
    float *y;
    int   up;
    int   n;
    int   k;
    float alpha;
    int   lda;
    int   incx;
    float beta;
    int   incy;
};

struct cublasCcopyParams {
    const cuComplex *cx; 
    cuComplex *cy;
    int   n;
    int   incx;
    int   incy;
};

struct cublasCaxpyParams {
    cuComplex ca;
    const cuComplex *cx; 
    cuComplex *cy;
    int   n;
    int   incx;
    int   incy;
};

struct cublasCscalParams {
    cuComplex ca;
    cuComplex *cx;
    int   n;
    int   incx;
};

struct cublasCsscalParams {
    cuComplex *cx;
    float sa;
    int   n;
    int   incx;
};

struct cublasCswapParams {
    cuComplex *cx;
    cuComplex *cy;
    int   n;
    int   incx;
    int   incy;
};

struct cublasScasumParams {
    const cuComplex *cx;
    float *result;
    int   n;
    int   incx;
    int   texXOfs;
};

struct cublasCgemmParams {
    cuComplex alpha;
    cuComplex beta;
    const cuComplex *A;
    const cuComplex *B;
    cuComplex *C;
    unsigned int   m;
    unsigned int   n;
    unsigned int   k;
    unsigned int   lda;
    unsigned int   ldb;
    unsigned int   ldc;
    int texAOfs;
    int texBOfs;
};

/* CUBLAS internal functions */
int cublasInitialized (const struct cublasContext *ctx);
void cublasShutDownCtx (struct cublasContext *ctx);
//__tlsHookStatus cublasInitCtx (struct cublasContext *ctx, void *_status);
//void cublasSetError (struct cublasContext *ctx, cublasStatus error);
void cublasVectorSplay (int n, int tMin, int tMax, int gridW, int *nbrCtas, 
                        int *elemsPerCta, int *threadsPerCta);
void cublasSmallSgemm (struct cublasContext *ctx, char transa, char transb, 
                       int m, int n, int k, float alpha, const float *A, 
                       int lda, const float *B, int ldb, float beta, float *C, 
                       int ldc);
void cublasFastSgemm (struct cublasContext *ctx, char transa, char transb,
                      int m, int n, int k, float alpha, const float *A, 
                      int lda, const float *B, int ldb, float beta, float *C, 
                      int ldc);
void cublasLargeSgemm (struct cublasContext *ctx, char transa, char transb, 
                       int m, int n, int k, float alpha, const float *A, 
                       int lda, const float *B, int ldb, float beta, float *C, 
                       int ldc);
void cublasFastCgemm (struct cublasContext *ctx, char transa, char transb, 
                      int m, int n, int k, cuComplex alpha, const cuComplex *A,
                      int lda, const cuComplex *B, int ldb, cuComplex beta, 
                      cuComplex *C, int ldc);

static int imax(int x, int y)
{
    return (x > y) ? x : y;
}
static int imin(int x, int y)
{
    return (x < y) ? x : y;
}

//#include "cublasDblP.h"

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(CUBLAS_P_H_) */
