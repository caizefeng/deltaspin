// File: cuda_errors.h
// CUDABLAS and CUFFT error checking

#ifndef _CUDA_ERRORS_
#define _CUDA_ERRORS_

// includes cuda headers
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

char *cublasGetErrorString(cublasStatus_t error);
char *cufftGetErrorString(cufftResult error);
#endif
/******************************************************/
