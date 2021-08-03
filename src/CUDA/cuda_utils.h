/** \file cuda_utils.h
 * \brief Definitions of persistent cuda pointers. 
 *
 * Each type internally the same, just typed differently.
 * Remember to also include cuda_utils.cu, which 
 * contains manipulation functions.
 */
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_
#include "cuComplex.h"

//Macros for n-dimensional array access (column major format)
#define DIM2(x, y, xdim)                                       ((y * xdim) + x)
#define DIM3(x, y, z, xdim, ydim)                              ((((z * ydim) + y) * xdim) + x)
#define DIM4(x1, x2, x3, x4, x1dim, x2dim, x3dim)              ((((((x4 * x3dim) + x3) * x2dim) + x2) * x1dim) + x2)
#define DIM5(x1, x2, x3, x4, x5, x1dim, x2dim, x3dim, x4dim)   ((((((((x5 * x4dim) + x4) * x3dim) + x3) * x2dim) + x2) * x1dim) + x1)

#define CUDA_CALL(function) {\
cudaError_t err = function; \
if (err != cudaSuccess) \
  fprintf(stderr, "CURROR [%s,%d]: %s \n", \
  __FILE__,  __LINE__, cudaGetErrorString(err)); \
}

#define CUDA_FFT_CALL(function) {\
cufftResult err = function; \
if (err != CUFFT_SUCCESS) \
  fprintf(stderr, "CURROR [%s,%d]: %d \n", \
  __FILE__,  __LINE__, err); \
}


/** N-dimensional double pointer */
typedef struct ndim_double {
  unsigned size;
  unsigned ndims;
  unsigned *dim;     //GPU
  unsigned lead_dim_size;
  double *ptr;      //current position, GPU
  double *start_pt;  //Starting point for access, GPU
} *ndim_double_ptr;

/** N dimensional cuDoubleComplex ptr */
typedef struct ndim_complex {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  cuDoubleComplex *ptr;        
  cuDoubleComplex *start_pt;
} *ndim_complex_ptr;

/**  int ptr */
typedef struct ndim_int {
  unsigned size;
  unsigned ndims;
  unsigned *dim;
  unsigned lead_dim_size;
  int *ptr;
  int *start_pt;
} *ndim_int_ptr;

/** void pointer, used mostly for casting */
typedef struct void_p{
  unsigned int size;
  void* ptr;
} void_p;

/** double pointer */
typedef struct double_p{
  unsigned int size;
  double* ptr;
} double_p;

/** float pointer */
typedef struct float_p{
  unsigned int size;
  float* ptr;
} float_p;

/** int pointer */
typedef struct int_p{
  unsigned int size;
  int* ptr;
} int_p;

/** Double Complex pointer (foo.x is real, foo.y is imag) */
typedef struct cuDoubleComplex_p{
  unsigned int size;
  cuDoubleComplex* ptr;
} cuDoubleComplex_p;

/** Complex pointer (foo.x is real, foo.y is imag) */
typedef struct cuComplex_p{
  unsigned int size;
  cuComplex* ptr;
} cuComplex_p;

#include "cuda_utils.cu"

#endif
