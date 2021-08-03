//Operator.h
//Hacene Mohamed

#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "cuda_runtime.h"
#include "cuda.h"

#if CUDA_VERSION>=5000
//#include "helper_cuda.h"
#define CUDA_SAFE_CALL(val)           check ( (val), #val, __FILE__, __LINE__ )
#else
#include "cutil.h"
#endif

#include "cuComplex.h"

#define MAX_BLOCKS    65535
#define MAX_THREADS   512
#define MIN_THREADS   64
#define REF_THREADS   64


//From the cublasP.h file used by the cublas library
#define ZDOT_CTAS          (80)
#define ZDOT_THREAD_COUNT  (128)
#define Z1DBUF_ALIGN        (256)  /* alignment for 1D buffer */
#define MAX_1DBUF_SIZE     ((1<<27)-Z1DBUF_ALIGN)
#define WORD_ALIGN         64

static int imax(int x, int y)
{
    return (x > y) ? x : y;
}
static int imin(int x, int y)
{
    return (x < y) ? x : y;
}

/******************************************************************/
 
 __device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCmul(x,y);
 }
 
  __device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCadd(x,y);
 }
 
   __device__ inline cuDoubleComplex operator- (const cuDoubleComplex & x,const cuDoubleComplex & y) {
    return cuCsub(x,y);
 }
  
 __device__ inline cuDoubleComplex operator* (const double & a,const cuDoubleComplex & x) {
    return make_cuDoubleComplex (a*cuCreal(x), a*cuCimag(x));
 } 
 
 __device__ inline cuDoubleComplex operator* (const cuDoubleComplex & x,const double & a) {
    return make_cuDoubleComplex (a*cuCreal(x), a*cuCimag(x));
 }
 
 __device__ inline cuDoubleComplex operator+ (const double & a,const cuDoubleComplex & x) {
    return make_cuDoubleComplex (a+cuCreal(x), cuCimag(x));
 }
 
 __device__ inline cuDoubleComplex operator+ (const cuDoubleComplex & x,const double & a) {
    return make_cuDoubleComplex (a+cuCreal(x), cuCimag(x));
 } 
  __device__ inline double Norm_2(const cuDoubleComplex & x) {
    return (cuCreal(x)*cuCreal(x)) + (cuCimag(x)*cuCimag(x));
 }
 
#endif
