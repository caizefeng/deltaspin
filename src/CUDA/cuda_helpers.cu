// File: cuda_helpers.cu
// CUDA helper kernels used in library.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes cuda headers
#include <cuda_runtime.h>
// includes project headers
#include "cuda_globals.h"
#include "Operator.h"
#include "cuda_helpers.h"

/******************************************************/
// CUDA kernels for parallel reduction, in library

// parallel sum reduction, in place
__global__ void cureducesum(double *data, const int N)
{
    //extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp = 0;

    // for each thread,

    // fetch from global memory and add
    for(int i=idx; i<N; i+=blockDim.x*gridDim.x)
    {
	tmp = tmp + data[i];
    }

    // perform sum reduction
    __cureducesum(data, sdata, tmp);
}

// parallel two sum reduction, in place
__global__ void cureducesum2(double *data, double *data1, const int N)
{
    //extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp = 0, tmp1 = 0;

    // for each thread,

    // fetch from global memory and add
    for(int i=idx; i<N; i+=blockDim.x*gridDim.x)
    {
        tmp = tmp + data[i];
	tmp1 = tmp1 + data1[i];
    }

    // perform two sum reduction
    __cureducesum2(data, data1, sdata, tmp, tmp1);
}

// parallel three sum reduction, in place
__global__ void cureducesum3(double *data, double *data1, double *data2, const int N)
{
    //extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp=0, tmp1=0, tmp2=0;

    // for each thread,

    // fetch from global memory and add
    for(int i=idx; i<N; i+=blockDim.x*gridDim.x)
    {
        tmp += data[i];
        tmp1 += data1[i];
	tmp2 += data2[i];
    }

    // perform three sum reduction
    __cureducesum3(data,data1,data2,sdata,sdata+blockDim.x,sdata+2*blockDim.x,tmp,tmp1,tmp2);
}

// parallel sum reduction with single block, in place
__global__ void cureducesum1block(double *data, const int N)
{
    //extern __shared__ cuDoubleComplex sdata[];
    double tmp = 0;

    // for each thread,

    // fetch from global memory
    if(threadIdx.x < N)
        tmp = data[threadIdx.x];
    else
        tmp = 0;

    // perform sum reduction
    __cureducesum(data, sdata, tmp);
}

// parallel sum reduction with single block, in place
__global__ void cureducesum1block(cuDoubleComplex *data, const int N)
{
    //extern __shared__ double sdata[];
    cuDoubleComplex tmp = make_cuDoubleComplex(0.0,0.0);

    // for each thread,

    // fetch from global memory
    if(threadIdx.x < N)
        tmp = data[threadIdx.x];
    else
        tmp = make_cuDoubleComplex(0.0,0.0);

    // perform sum reduction
    __cureducesum(data, szdata, tmp);
}

// parallel three sum reduction with single block, in place
__global__ void cureducesum3_1block(double *data, double *data1, double *data2, const int N)
{
    double tmp=0,tmp1=0,tmp2=0;

    // for each thread,

    // fetch from global memory
    if(threadIdx.x < N)
    {
        tmp = data[threadIdx.x];
	tmp1 = data1[threadIdx.x];
	tmp2 = data2[threadIdx.x];
    }

    // perform three sum reduction
    __cureducesum3(data,data1,data2,sdata,sdata+blockDim.x,sdata+2*blockDim.x,tmp,tmp1,tmp2);
}

/******************************************************/

