// File: cuda_helpers.h
// CUDA helper global and device kernels used in library.

#ifndef _CUDA_HELPERS_
#define _CUDA_HELPERS_
/******************************************************/
// CUDA kernel declarations

// parallel sum reduction
__global__ void cureducesum(double *data, const int N);
__global__ void cureducesum2(double *data, double *data1, const int N);
__global__ void cureducesum3(double *data, double *data1, double *data2, const int N);
__global__ void cureducesum1block(double *data, const int N);
__global__ void cureducesum1block(cuDoubleComplex *data, const int N);
__global__ void cureducesum3_1block(double *data, double *data1, double *data2, const int N);

/******************************************************/
// CUDA device kernels for reduction

// complex exponential function
__forceinline__
__device__ cuDoubleComplex cexp(cuDoubleComplex z)
{
    cuDoubleComplex res;
    double t = exp(z.x);

    // exp(z) = exp(x + iy) = exp(x) * exp (iy) = exp(x) * (cos(y) + i sin(y))
    sincos(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

// parallel sum reduction
__forceinline__
__device__ void __cureducesum(double *data, double *sdata, double inputval)
{
    // for each thread,

    // copy to shared memory and sync
    sdata[threadIdx.x] = inputval;
    __syncthreads();

    // perform sum reduction

#pragma unroll
    for(int s=blockDim.x>>1; s>0; s>>=1)
    {
        if(threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    // store the result
    if(threadIdx.x == 0)
        data[blockIdx.x] = sdata[0];
}

// parallel sum reduction
__forceinline__
__device__ void __cureducesum(cuDoubleComplex *data, cuDoubleComplex *sdata, 
		cuDoubleComplex inputval)
{
    // for each thread,

    // copy to shared memory and sync
    sdata[threadIdx.x] = inputval;
    __syncthreads();

    // perform sum reduction

#pragma unroll
    for(int s=blockDim.x>>1; s>0; s>>=1)
    {
        if(threadIdx.x < s)
            sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        __syncthreads();
    }

    // store the result
    if(threadIdx.x == 0)
        data[blockIdx.x] = sdata[0];
}

// parallel two sum reduction
__forceinline__
__device__ void __cureducesum2(double *data, double *data1, double *sdata, 
		double inputval, double inputval1)
{
    int idx = threadIdx.x+blockDim.x;

    // for each thread,

    // copy to shared memory and sync
    sdata[threadIdx.x] = inputval;
    sdata[idx] = inputval1;
    __syncthreads();

    // perform two sum reduction
#pragma unroll
    for(int s=blockDim.x>>1; s>0; s>>=1)
    {
        if(threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    // store the results
    if(threadIdx.x == 0)
    {
        data[blockIdx.x] = sdata[0];
        data1[blockIdx.x] = sdata[blockDim.x];
    }
}

// parallel three sum reduction
__forceinline__
__device__ void __cureducesum3(double *data, double *data1, double *data2,
		double *sdata, double *sdata1, double *sdata2, double inputval,
		double inputval1, double inputval2)
{
    // for each thread,

    // copy to shared memory and sync
    sdata[threadIdx.x] = inputval;
    sdata1[threadIdx.x] = inputval1;
    sdata2[threadIdx.x] = inputval2;
    __syncthreads();

    // perform sum reduction
    #pragma unroll
    for(int s=blockDim.x>>1; s>0; s>>=1)
    {
        if(threadIdx.x < s)
	{
            sdata[threadIdx.x] +=  sdata[threadIdx.x + s];
	    sdata1[threadIdx.x] += sdata1[threadIdx.x + s];
	    sdata2[threadIdx.x] += sdata2[threadIdx.x + s];
	}
        __syncthreads();
    }

    // store the result
    if(threadIdx.x == 0)
    {
        data[blockIdx.x] =  sdata[0];
	data1[blockIdx.x] = sdata1[0];
	data2[blockIdx.x] = sdata2[0];
    }
}

#endif
/******************************************************/
