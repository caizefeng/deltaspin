// File: cuda_globals.h
// Contains variable declarations shared throughout the library.
#ifndef __CUDA_GLOBALS_
#define __CUDA_GLOBALS_

#include "cuda_runtime.h"
#include "cuda.h"
#include "cuComplex.h"

// macros
#define KB (1024.0)    // 1024 bytes
#define MB (1048576.0) // 1024*1024 bytes
#define NTHREADS 128

//#define NUM_STREAMS     64      // number of streams
#define NUM_THREADS     256     // number of threads per block

// 
typedef size_t devptr_t;

// CUFFT plans
extern int NUM_PLANS;
// CUDA streams
extern int NUM_STREAMS;
extern cudaStream_t *stream;

// parallel reduction arrays
extern double *d_reduce, *d_reduce1;
extern cuDoubleComplex *d_zreduce, *d_zreduce1;

// device pointers arrays
extern devptr_t *d_ptrs, *d_ptrs1;

// shared memory arrays, dynamically allocated
extern __shared__ double sdata[];
extern __shared__ cuDoubleComplex szdata[];

/******************************************************/
// macros for error messages, in library

// wrapper for CUDA API errors
inline void __error(const char *from, const char *file, int line, const char *msg)
{
    printf("\n%s Error in %s, line %d: %s\n", from, file, line, msg);
    cudaDeviceReset();
    exit(-1);
}
#define ERROR(from, msg)   __error( from, __FILE__, __LINE__, msg )

/******************************************************/
// macros for CUDA API errors, in library

// wrapper for CUDA API errors
inline void __cuda_error(cudaError_t status, const char *file, int line, const char *msg)
{
    if(status != cudaSuccess)
    {
        printf("\nCUDA Error in %s, line %d: %s\n %s\n", file, line, cudaGetErrorString(status), msg);
        cudaDeviceReset();
        float* foo = NULL;
        float bar = foo[0];
        printf("Creating segfault, %f\n", bar);
        exit(-1);
    }
}
#define CUDA_ERROR(status, msg)   __cuda_error( status, __FILE__, __LINE__, msg )

/******************************************************/
// macros for CUDA streams, in library

inline cudaStream_t __cuda_stream(int sid, const char *file, int line)
{
    // safety check
    if(sid >= NUM_STREAMS) 
    {
	printf("\nCUDA Error in %s, line %d: %s\n", file, line, "Invalid CUDA stream id!");
	cudaDeviceReset();
        exit(-1);
    }

    // return CUDA stream
    cudaStream_t st=0;  // set null stream
    if(sid>=0) st=stream[sid];  // get stream
    return st;
}
#define CUDA_STREAM(sid)  __cuda_stream( sid, __FILE__, __LINE__ )

#endif
/******************************************************/
