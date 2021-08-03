// File: cuda_blas.cu
// C/Fortran interface to CUBLAS amd CUDA C BLAS.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes cuda headers
#include <cuda_runtime.h>
#include <cublas_v2.h>
// includes project headers
#include "cuda_globals.h"
#include "Operator.h"
#include "cuda_helpers.h"
#include "fortran.h"

// cublas handle
cublasHandle_t hcublas;

/******************************************************/
// CUBLAS wrapper for CUBLAS API errors, in library

extern "C" char *cublasGetErrorString(cublasStatus_t error);
// wrapper for CUBLAS API errors
inline void __cublas_error(cublasStatus_t status, const char *file, int line, const char *msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
      float* foo = NULL;
      float bar = foo[0];
      printf("Tried to segfault! %f\n", bar);

        printf("\nCUBLAS Error in %s, line %d: %s\n %s\n", file, line, cublasGetErrorString(status), msg);
        cudaDeviceReset();
        exit(-1);
    }
}
#define CUBLAS_ERROR(status, msg) __cublas_error( status, __FILE__, __LINE__, msg )

/******************************************************/
// CUBLAS wrappers for init, in VAMP

// Initialize CUBLAS
extern "C"
void cublas_init_C(void)
{
    CUBLAS_ERROR( cublasCreate(&hcublas), "Failed to initialze CUBLAS!" );
}

// Destroy CUBLAS
extern "C"
void cublas_destroy_C(void)
{
    CUBLAS_ERROR( cublasDestroy(hcublas), "Failed to destroy CUBLAS" );
}

/******************************************************/
// CUBLAS wrappers for memcpy operations, in VASP

// copy vector from host to device, synchronous to host
extern "C"
void cublas_set_vector_C(int *n, int *elemSize, void *x, int *incx, void **y, int *incy)
{
    //nvp_start_(&NVP_MEMCPY);
    CUBLAS_ERROR( cublasSetVector(*n,*elemSize,x,*incx,*y,*incy),
		  "Failed to execute cublasSetVector!" );
    //nvp_stop_(&NVP_MEMCPY);
}

// copy vector from device to host, synchronous to host
extern "C"
void cublas_get_vector_C(int *n, int *elemSize, void **x, int *incx, void *y, int *incy)
{
    //nvp_start_(&NVP_MEMCPY);
    CUBLAS_ERROR( cublasGetVector(*n,*elemSize,*x,*incx,y,*incy),
		  "Failed to execute cublasGetVector!" );
    //nvp_stop_(&NVP_MEMCPY);
}

// copy matrix from host to device, synchronous to host
extern "C"
void cublas_set_matrix_C(int *rows, int *cols, int *elemSize, void *A, int *lda,
    void **B, int *ldb)
{
    //nvp_start_(&NVP_MEMCPY);
    CUBLAS_ERROR( cublasSetMatrix(*rows,*cols,*elemSize,A,*lda,*B,*ldb),
                  "Failed to execute cublasSetMatrix!" );
    //nvp_stop_(&NVP_MEMCPY);
}

// copy matrix from host to device with given stream, asynchronous to host
extern "C"
void cublas_set_matrix_async_C(int *sid, int *rows, int *cols, int *elemSize,
    void *A, int *lda, void **B, int *ldb)
{
    CUBLAS_ERROR( cublasSetMatrixAsync(*rows,*cols,*elemSize,A,*lda,*B,*ldb,stream[*sid]),
		  "Failed to execute cublasSetMatrixAsync!" );
}

// copy matrix from device to host, synchronous to host
extern "C"
void cublas_get_matrix_C(int *rows, int *cols, int *elemSize, void **A, int *lda,
     void *B, int *ldb)
{
    //nvp_start_(&NVP_MEMCPY);
    CUBLAS_ERROR( cublasGetMatrix(*rows,*cols,*elemSize,*A,*lda,B,*ldb),
		  "Failed to execute cublasGetMatrix!" );
    //nvp_stop_(&NVP_MEMCPY);
}

// copy matrix from device to host with given stream, asynchronous to host
extern "C"
void cublas_get_matrix_async_C(int *sid, int *rows, int *cols, int *elemSize,
     void **A, int *lda, void *B, int *ldb)
{
    CUBLAS_ERROR( cublasGetMatrixAsync(*rows,*cols,*elemSize,*A,*lda,B,*ldb,stream[*sid]),
                  "Failed to execute cublasGetMatrixAsync!" );
}

/******************************************************/
// CUDA C kernels/wrappers for memcpy operations, in VASP

// copy matrix from device to device, asynchronous to host
template <class T>
__global__ void cusetmatrix(int rows, int cols, T *A, int lda, T *B, int ldb)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    // for each thread,
    if(idx<rows && idy<cols)
    {
	// copy matrix A to B
	B[idx+idy*ldb] = A[idx+idy*lda];
    }
}

// copy matrix from device to device, asynchronous to host
extern "C"
void cuda_zsetmatrix_C(int *sid, int *rows, int *cols, cuDoubleComplex **A, int *lda,
     cuDoubleComplex **B, int *ldb)
{
    // grid dimensions
    dim3 block(16,16);
    dim3 grid((*rows+block.x-1)/block.x,(*cols+block.y-1)/block.y);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream
    // copy matrix A to B
    cusetmatrix<cuDoubleComplex><<<grid,block>>>(*rows,*cols,*A,*lda,*B,*ldb);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cusetmatrix!" );
}

/******************************************************/
// CUBLAS wrappers for BLAS Level-1 operations, in VASP

extern "C"
void cublas_ddot_C(int *n, double **x, int *incx, double **y, int *incy, double *result)
{
    CUBLAS_ERROR( cublasDdot(hcublas,*n,*x,*incx,*y,*incy,result),
		   "Failed to execute cublasDdot!" );
}

extern "C"
void cublas_zdotc_C(int *n, cuDoubleComplex **x, int *incx, cuDoubleComplex **y, int *incy,
     cuDoubleComplex *result)
{
    CUBLAS_ERROR( cublasZdotc(hcublas,*n,*x,*incx,*y,*incy,result),
		  "Failed to execute cublasZdotc!" );
}

/******************************************************/
// CUDA C kernels/wrappers for BLAS Level-1 operations, in VASP

//subtraction of vectors a and b , c = a - b
template <class T>
__global__ void cusub(int n, T *c, T *a, T *b)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    // for each thread,

    if(idx < n)
    {
	// compute, c = a - b
        c[idx] = a[idx] - b[idx];
    }
}

// subtraction of vectors a and b, c = a - b
extern "C"
void cuda_zsub_C(int *n, cuDoubleComplex **c, cuDoubleComplex **a, cuDoubleComplex **b)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*n+block.x-1)/block.x);

    // c = a - b
    cusub<cuDoubleComplex><<<grid,block>>>(*n,*c,*a,*b);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cusub!" );
}

// dot product of vectors x and y, result kept on device
__global__ void cuddot(int n, double *x, double *y, double *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double ddot = 0.0;

    // for each thread,

    // loop over all elements
    for(int i=idx; i<n; i+=blockDim.x*gridDim.x)
    {
        // dot product, zdot = x * y
        ddot = ddot + x[i] * y[i];
    }

    // perform reduction
    __cureducesum(result, sdata, ddot);
}

// dot product of vectors x and y, result kept on device
extern "C"
void cuda_ddot_C(int *n, double **x, double **y, double **result)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid( min((*n+block.x-1)/block.x, MAX_THREADS) );
   // size of shared memory buffer
    int ssize = block.x*sizeof(double);

    // partial dot product
    cuddot<<<grid,block,ssize>>>(*n,*x,*y,d_reduce);
    // add remaining
    cureducesum1block<<<1,block,ssize>>>(d_reduce,grid.x);

    // copy from device to device
    CUDA_ERROR( cudaMemcpy(*result,d_reduce,sizeof(double),cudaMemcpyDeviceToDevice),
                "Failed to copy from device to device in cuda_ddot!" );
}

// nblock hermitian dot products of vectors in xptrs and yptrs, blocked algorithm
__global__ void cuzdotcblock(int n, devptr_t *xptrs, devptr_t *yptrs, cuDoubleComplex *result)
{
    cuDoubleComplex dot = make_cuDoubleComplex(0.0,0.0);

    // for each thread,

    // fetch pointer for each block
    cuDoubleComplex *x = (cuDoubleComplex *)xptrs[blockIdx.x];
    cuDoubleComplex *y = (cuDoubleComplex *)yptrs[blockIdx.x];
    if(x==NULL || y==NULL) return;  // return if not valid

    // loop over all elements
    for(int i=threadIdx.x; i<n; i+=blockDim.x)
    {
        // fetch from global memory
        cuDoubleComplex a = x[i];
        cuDoubleComplex b = y[i];
        // compute, dot = (conj of x) * y
        dot.x = dot.x + a.x*b.x + a.y*b.y;
        dot.y = dot.y + a.x*b.y - a.y*b.x;
    }

    // perform sum reduction
    __cureducesum(result, szdata, dot);
}

// nblock hermitian dot products of vectors in xptrs and yptrs, blocked algorithm
extern "C"
void cuda_zdotcblock_C(int *nblock, int *n, devptr_t *xptrs, devptr_t *yptrs, cuDoubleComplex *result)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid(*nblock);
    // size of shared memory buffer
    int ssize = block.x*sizeof(cuDoubleComplex);

    // copy array of pointers x from host to device
    CUDA_ERROR( cudaMemcpy(d_ptrs,xptrs,grid.x*sizeof(devptr_t),cudaMemcpyHostToDevice),
                "Failed to copy from host to device!" );
    // copy array of pointers y from host to device
    CUDA_ERROR( cudaMemcpy(d_ptrs1,yptrs,grid.x*sizeof(devptr_t),cudaMemcpyHostToDevice),
                "Failed to copy from host to device!" );

    // compute, dot = (conj of x) * y
    cuzdotcblock<<<grid,block,ssize>>>(*n,d_ptrs,d_ptrs1,d_zreduce);
    // copy dot products from device to host
    CUDA_ERROR( cudaMemcpy(result,d_zreduce,grid.x*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost), "Failed to copy from device to host!" );
}

// nblock scales of vectors in xptrs with scalars alpha, blocked algorithm
__global__ void cuzdscalblock(int n, double *alpha, devptr_t *xptrs)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // for each thread,

    if(idx < n)
    {
        // fetch pointer for each block
        cuDoubleComplex *x = (cuDoubleComplex *)xptrs[blockIdx.y];
        if(x == NULL) return;  // return if not valid

        // compute, x = alpha * x
        x[idx] = x[idx] * alpha[blockIdx.y];
    }
}

// nblock scales of vectors in xptrs with scalars alpha, blocked algorithm
extern "C"
void cuda_zdscalblock_C(int *nblock, int *n, double *alpha, devptr_t *xptrs)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*n+block.x-1)/block.x,*nblock);

    // copy array of pointers x from host to device
    CUDA_ERROR( cudaMemcpy(d_ptrs,xptrs,(*nblock)*sizeof(devptr_t),cudaMemcpyHostToDevice),
                "Failed to copy from host to device!" );
    // copy alpha from host to device
    CUDA_ERROR( cudaMemcpy(d_reduce,alpha,(*nblock)*sizeof(double),cudaMemcpyHostToDevice),
                "Failed to copy from host to device!" );

    // compute, x = alpha * x
    cuzdscalblock<<<grid,block>>>(*n,d_reduce,d_ptrs);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuzdscalblock!" );
}

/******************************************************/
// CUBLAS wrappers for BLAS Level-2 operations

extern "C"
void cublas_zgemv_C(char *trans, int *m, int *n, cuDoubleComplex *alpha, cuDoubleComplex **A, int *lda,
     cuDoubleComplex **x, int *incx, cuDoubleComplex *beta, cuDoubleComplex **y, int *incy)
{
    cublasOperation_t t;
    if(*trans=='N' || *trans=='n')
	t = CUBLAS_OP_N;
    else if(*trans=='T' || *trans=='t')
	t = CUBLAS_OP_T;
    else if(*trans=='C' || *trans=='c')
	t = CUBLAS_OP_C;
    CUBLAS_ERROR( cublasZgemv(hcublas,t,*m,*n,alpha,*A,*lda,*x,*incx,beta,*y,*incy),
		  "Failed to execute cublasZgemv!" );
}

/******************************************************/
// CUDA C wrappers/kernels for custom GEMM, in RPROMU_GPU

// CUDA C GEMM kernel optimized for small M,N
template<int perThreadX, int perThreadY>
__global__ void gemm_kernel(const int m, const int n, const int k, const double * A, const int lda, const double * B, const int ldb, double * C, const int ldc, const int total_tiles_y, const int total_tiles_x)
{
    __shared__ double scratch[NTHREADS*4];

    double my_result[perThreadY][perThreadX];
#pragma unroll 
    for(int i = 0; i < perThreadY;++i)
#pragma unroll
        for(int j = 0; j < perThreadX;++j)
            my_result[i][j] = 0;
    const int my_tile_x = blockIdx.x % total_tiles_x;
    const int my_tile_y = blockIdx.x / total_tiles_x;
    const double * myA = A + my_tile_y * perThreadY * lda;
    const double * myB = B + my_tile_x * perThreadX * ldb;
    int no_load = 0;
    const int rest_x = n % perThreadX;
    const int rest_y = m % perThreadY;
    if(rest_y != 0 && my_tile_y == total_tiles_y-1)
        no_load += 1;
    if(rest_x != 0 && my_tile_x == total_tiles_x-1)
        no_load += 2;
    double * const myC = C + my_tile_x * ldc * perThreadX + my_tile_y * perThreadY;
    double prefetchAelements[perThreadY];
    double prefetchBelements[perThreadX];

    switch(no_load)
    {
        case 0:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    prefetchAelements[i] = myA[threadIdx.x + i*lda];
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        prefetchAelements[j] = myA[i + j*lda];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        prefetchBelements[j] = myB[i + j*ldb];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
        case 1:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    if(i < rest_y)
                        prefetchAelements[i] = myA[threadIdx.x + i*lda];
                    else
                        prefetchAelements[i] = 0;
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        if(j < rest_y)
                            prefetchAelements[j] = myA[i + j*lda];
                        else
                            prefetchAelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        prefetchBelements[j] = myB[i + j*ldb];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
        case 2:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    prefetchAelements[i] = myA[threadIdx.x + i*lda];
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    if(i < rest_x)
                        prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                    else
                        prefetchBelements[i] = 0;
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        prefetchAelements[j] = myA[i + j*lda];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        if(j < rest_x)
                            prefetchBelements[j] = myB[i + j*ldb];
                        else
                            prefetchBelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }

            break;
        case 3:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    if(i < rest_y)
                        prefetchAelements[i] = myA[threadIdx.x + i*lda];
                    else
                        prefetchAelements[i] = 0;
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    if(i < rest_x)
                        prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                    else
                        prefetchBelements[i] = 0;
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        if(j < rest_y)
                            prefetchAelements[j] = myA[i + j*lda];
                        else
                            prefetchAelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        if(j < rest_x)
                            prefetchBelements[j] = myB[i + j*ldb];
                        else
                            prefetchBelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
    }
#pragma unroll
    for(int step = 0; step < perThreadX*perThreadY; step += 4)
    {
        const int red_1_x = step % perThreadX;
        const int red_1_y = step / perThreadX;
        const int red_2_x = (step+1) % perThreadX;
        const int red_2_y = (step+1) / perThreadX;
        const int red_3_x = (step+2) % perThreadX;
        const int red_3_y = (step+2) / perThreadX;
        const int red_4_x = (step+3) % perThreadX;
        const int red_4_y = (step+3) / perThreadX;
        scratch[threadIdx.x] = my_result[red_1_y][red_1_x];
        if(red_2_y < perThreadY)
        {
            scratch[threadIdx.x + NTHREADS] = my_result[red_2_y][red_2_x];
        }
        if(red_3_y < perThreadY)
        {
            scratch[threadIdx.x + 2*NTHREADS] = my_result[red_3_y][red_3_x];
        }
        if(red_4_y < perThreadY)
        {
            scratch[threadIdx.x + 3*NTHREADS] = my_result[red_4_y][red_4_x];
        }
    __syncthreads();
    const int shift = (threadIdx.x/(NTHREADS/4))*NTHREADS;
    const int tid = threadIdx.x%(NTHREADS/4);
#pragma unroll
    for(int i = 1; i < 4; ++i)
        scratch[tid + shift] += scratch[tid + shift + (NTHREADS/4)*i];
    __syncthreads();
#pragma unroll
    for(int s = NTHREADS/8; s >= 1; s = s >> 1)
    {
        if(tid < s)
        {
            scratch[tid + shift] += scratch[tid + shift + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        switch(no_load)
        {
            case 0:
                myC[red_1_x*ldc + red_1_y] = scratch[0];
                if(red_2_y < perThreadY)
                {
                    myC[red_2_x*ldc + red_2_y] = scratch[NTHREADS];
                }
                if(red_3_y < perThreadY)
                {
                    myC[red_3_x*ldc + red_3_y] = scratch[2*NTHREADS];
                }
                if(red_4_y < perThreadY)
                {
                    myC[red_4_x*ldc + red_4_y] = scratch[3*NTHREADS];
                }
                break;
            case 1:
                if(red_1_y < rest_y)
                    myC[red_1_x*ldc + red_1_y] = scratch[0];
                if(red_2_y < rest_y)
                {
                    myC[red_2_x*ldc + red_2_y] = scratch[NTHREADS];
                }
                if(red_3_y < rest_y)
                {
                    myC[red_3_x*ldc + red_3_y] = scratch[2*NTHREADS];
                }
                if(red_4_y < rest_y)
                {
                    myC[red_4_x*ldc + red_4_y] = scratch[3*NTHREADS];
                }
                break;
            case 2:
                if(red_1_x < rest_x)
                    myC[red_1_x*ldc + red_1_y] = scratch[0];
                if(red_2_y < perThreadY && red_2_x < rest_x)
                {
                    myC[red_2_x*ldc + red_2_y] = scratch[NTHREADS];
                }
                if(red_3_y < perThreadY && red_3_x < rest_x)
                {
                    myC[red_3_x*ldc + red_3_y] = scratch[2*NTHREADS];
                }
                if(red_4_y < perThreadY && red_4_x < rest_x)
                {
                    myC[red_4_x*ldc + red_4_y] = scratch[3*NTHREADS];
                }

                break;
            case 3:
                if(red_1_y < rest_y && red_1_x < rest_x)
                    myC[red_1_x*ldc + red_1_y] = scratch[0];
                if(red_2_y < rest_y && red_2_x < rest_x)
                {
                    myC[red_2_x*ldc + red_2_y] = scratch[NTHREADS];
                }
                if(red_3_y < rest_y && red_3_x < rest_x)
                {
                    myC[red_3_x*ldc + red_3_y] = scratch[2*NTHREADS];
                }
                if(red_4_y < rest_y && red_4_x < rest_x)
                {
                    myC[red_4_x*ldc + red_4_y] = scratch[3*NTHREADS];
                }


        }
    }
    __syncthreads();
    }
}

// CUDA C GEMM kernel optimized for small M,N
template<int perThreadX, int perThreadY>
__global__ void gemm_kernel_ab(const int m, const int n, const int k, const double alpha, const double * A, const int lda, const double * B, const int ldb, const double beta, double * C, const int ldc, const int total_tiles_y, const int total_tiles_x)
{
    __shared__ double scratch[NTHREADS*4];

    double my_result[perThreadY][perThreadX];
#pragma unroll 
    for(int i = 0; i < perThreadY;++i)
#pragma unroll
        for(int j = 0; j < perThreadX;++j)
            my_result[i][j] = 0;
    const int my_tile_x = blockIdx.x % total_tiles_x;
    const int my_tile_y = blockIdx.x / total_tiles_x;
    const double * myA = A + my_tile_y * perThreadY * lda;
    const double * myB = B + my_tile_x * perThreadX * ldb;
    int no_load = 0;
    const int rest_x = n % perThreadX;
    const int rest_y = m % perThreadY;
    if(rest_y != 0 && my_tile_y == total_tiles_y-1)
        no_load += 1;
    if(rest_x != 0 && my_tile_x == total_tiles_x-1)
        no_load += 2;
    double * const myC = C + my_tile_x * ldc * perThreadX + my_tile_y * perThreadY;
    double prefetchAelements[perThreadY];
    double prefetchBelements[perThreadX];

    switch(no_load)
    {
        case 0:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    prefetchAelements[i] = myA[threadIdx.x + i*lda];
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        prefetchAelements[j] = myA[i + j*lda];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        prefetchBelements[j] = myB[i + j*ldb];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
        case 1:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    if(i < rest_y)
                        prefetchAelements[i] = myA[threadIdx.x + i*lda];
                    else
                        prefetchAelements[i] = 0;
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        if(j < rest_y)
                            prefetchAelements[j] = myA[i + j*lda];
                        else
                            prefetchAelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        prefetchBelements[j] = myB[i + j*ldb];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
        case 2:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    prefetchAelements[i] = myA[threadIdx.x + i*lda];
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    if(i < rest_x)
                        prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                    else
                        prefetchBelements[i] = 0;
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        prefetchAelements[j] = myA[i + j*lda];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        if(j < rest_x)
                            prefetchBelements[j] = myB[i + j*ldb];
                        else
                            prefetchBelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }

            break;
        case 3:
            if(threadIdx.x < k)
            {
#pragma unroll
                for(int i = 0; i < perThreadY;++i)
                {
                    if(i < rest_y)
                        prefetchAelements[i] = myA[threadIdx.x + i*lda];
                    else
                        prefetchAelements[i] = 0;
                }
#pragma unroll
                for(int i = 0; i < perThreadX; ++i)
                {
                    if(i < rest_x)
                        prefetchBelements[i] = myB[threadIdx.x + i*ldb];
                    else
                        prefetchBelements[i] = 0;
                }
                for(int i = threadIdx.x + blockDim.x; i < k; i += blockDim.x)
                {
                    double currentAelements[perThreadY];
                    double currentBelements[perThreadX];
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        currentAelements[j] = prefetchAelements[j];
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        currentBelements[j] = prefetchBelements[j];
                    }

#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
                    {
                        if(j < rest_y)
                            prefetchAelements[j] = myA[i + j*lda];
                        else
                            prefetchAelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadX; ++j)
                    {
                        if(j < rest_x)
                            prefetchBelements[j] = myB[i + j*ldb];
                        else
                            prefetchBelements[j] = 0;
                    }
#pragma unroll
                    for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                        for(int k = 0; k < perThreadX; ++k)
                            my_result[j][k] += currentAelements[j]*currentBelements[k];
                }
                double currentAelements[perThreadY];
                double currentBelements[perThreadX];
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
                {
                    currentAelements[j] = prefetchAelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadX; ++j)
                {
                    currentBelements[j] = prefetchBelements[j];
                }
#pragma unroll
                for(int j = 0; j < perThreadY; ++j)
#pragma unroll
                    for(int k = 0; k < perThreadX; ++k)
                        my_result[j][k] += currentAelements[j]*currentBelements[k];
            }
            break;
    }
#pragma unroll
    for(int step = 0; step < perThreadX*perThreadY; step += 4)
    {
        const int red_1_x = step % perThreadX;
        const int red_1_y = step / perThreadX;
        const int red_2_x = (step+1) % perThreadX;
        const int red_2_y = (step+1) / perThreadX;
        const int red_3_x = (step+2) % perThreadX;
        const int red_3_y = (step+2) / perThreadX;
        const int red_4_x = (step+3) % perThreadX;
        const int red_4_y = (step+3) / perThreadX;
        scratch[threadIdx.x] = my_result[red_1_y][red_1_x];
        if(red_2_y < perThreadY)
        {
            scratch[threadIdx.x + NTHREADS] = my_result[red_2_y][red_2_x];
        }
        if(red_3_y < perThreadY)
        {
            scratch[threadIdx.x + 2*NTHREADS] = my_result[red_3_y][red_3_x];
        }
        if(red_4_y < perThreadY)
        {
            scratch[threadIdx.x + 3*NTHREADS] = my_result[red_4_y][red_4_x];
        }
    __syncthreads();
    const int shift = (threadIdx.x/(NTHREADS/4))*NTHREADS;
    const int tid = threadIdx.x%(NTHREADS/4);
#pragma unroll
    for(int i = 1; i < 4; ++i)
        scratch[tid + shift] += scratch[tid + shift + (NTHREADS/4)*i];
    __syncthreads();
#pragma unroll
    for(int s = NTHREADS/8; s >= 1; s = s >> 1)
    {
        if(tid < s)
        {
            scratch[tid + shift] += scratch[tid + shift + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        switch(no_load)
        {
            case 0:
                myC[red_1_x*ldc + red_1_y] = beta * myC[red_1_x*ldc + red_1_y] + alpha * scratch[0];
                if(red_2_y < perThreadY)
                {
                    myC[red_2_x*ldc + red_2_y] = beta * myC[red_2_x*ldc + red_2_y] + alpha * scratch[NTHREADS];
                }
                if(red_3_y < perThreadY)
                {
                    myC[red_3_x*ldc + red_3_y] =beta * myC[red_3_x*ldc + red_3_y] + alpha *  scratch[2*NTHREADS];
                }
                if(red_4_y < perThreadY)
                {
                    myC[red_4_x*ldc + red_4_y] = beta * myC[red_4_x*ldc + red_4_y] + alpha * scratch[3*NTHREADS];
                }
                break;
            case 1:
                if(red_1_y < rest_y)
                    myC[red_1_x*ldc + red_1_y] = beta * myC[red_1_x*ldc + red_1_y] + alpha * scratch[0];
                if(red_2_y < rest_y)
                {
                    myC[red_2_x*ldc + red_2_y] = beta * myC[red_2_x*ldc + red_2_y] + alpha * scratch[NTHREADS];
                }
                if(red_3_y < rest_y)
                {
                    myC[red_3_x*ldc + red_3_y] = beta * myC[red_3_x*ldc + red_3_y] + alpha * scratch[2*NTHREADS];
                }
                if(red_4_y < rest_y)
                {
                    myC[red_4_x*ldc + red_4_y] = beta * myC[red_4_x*ldc + red_4_y] + alpha * scratch[3*NTHREADS];
                }
                break;
            case 2:
                if(red_1_x < rest_x)
                    myC[red_1_x*ldc + red_1_y] = beta * myC[red_1_x*ldc + red_1_y] + alpha * scratch[0];
                if(red_2_y < perThreadY && red_2_x < rest_x)
                {
                    myC[red_2_x*ldc + red_2_y] = beta * myC[red_2_x*ldc + red_2_y] + alpha * scratch[NTHREADS];
                }
                if(red_3_y < perThreadY && red_3_x < rest_x)
                {
                    myC[red_3_x*ldc + red_3_y] = beta * myC[red_3_x*ldc + red_3_y] + alpha * scratch[2*NTHREADS];
                }
                if(red_4_y < perThreadY && red_4_x < rest_x)
                {
                    myC[red_4_x*ldc + red_4_y] = beta * myC[red_4_x*ldc + red_4_y] + alpha * scratch[3*NTHREADS];
                }

                break;
            case 3:
                if(red_1_y < rest_y && red_1_x < rest_x)
                    myC[red_1_x*ldc + red_1_y] = beta * myC[red_1_x*ldc + red_1_y] + alpha * scratch[0];
                if(red_2_y < rest_y && red_2_x < rest_x)
                {
                    myC[red_2_x*ldc + red_2_y] = beta * myC[red_2_x*ldc + red_2_y] + alpha * scratch[NTHREADS];
                }
                if(red_3_y < rest_y && red_3_x < rest_x)
                {
                    myC[red_3_x*ldc + red_3_y] = beta * myC[red_3_x*ldc + red_3_y] + alpha * scratch[2*NTHREADS];
                }
                if(red_4_y < rest_y && red_4_x < rest_x)
                {
                    myC[red_4_x*ldc + red_4_y] = beta * myC[red_4_x*ldc + red_4_y] + alpha * scratch[3*NTHREADS];
                }
        }
    }
    __syncthreads();
    }
}

// CUDA C GEMM kernel optimized for small M,N
extern "C"
void cuda_gemmsmallmn_C(const int *StreamIdx, const char *transa, const char *transb,
     const int *m_p, const int *n_p, const int *k_p, const double *alpha, const devptr_t* A_p,
     const int *shiftA, const int *lda, const devptr_t* B_p, const int *shiftB, const int *ldb,
     const double *beta, const devptr_t *C_p, const int *shiftC, const int *ldc)
{
        const int m = *m_p;
        const int n = *n_p;
        const int k = *k_p;

	//For CUDA versions 5.0 or earlier only use the cublas gemm functions
#if  CUDART_VERSION <= 5000
	CUBLAS_DGEMMSH_ST_C(StreamIdx, transa, transb, m_p, n_p, k_p, alpha, A_p, shiftA, lda, B_p, shiftB, ldb, beta, C_p, shiftC, ldc);
	return ;
#endif
 
//simple heuristics - if m,n are small (3600 is empirical value)  
//and transa='T',transb='N' use our implementation, else - use cublas
    if(m * n <= 3600 && *transa == 'T' && *transb == 'N')
    {
        double *A = (double *)(*A_p) + *shiftA;
        double *B = (double *)(*B_p) + *shiftB;
        double *C = (double *)(*C_p) + *shiftC;
      
        if(*alpha == 1.0 && *beta == 0.0)
        { 
            if(m % 5 == 0)
            {
                if((n) % 5 == 0)
                     gemm_kernel<5,5><<<((n+4)/5)*((m+4)/5), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,A,*lda,B,*ldb,C,*ldc,(m+4)/5,(n+4)/5);
                else
                     gemm_kernel<4,5><<<((n+3)/4)*((m+4)/5), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,A,*lda,B,*ldb,C,*ldc,(m+4)/5,(n+3)/4);
            }
            else
            {
                if((n) % 5 == 0)
                     gemm_kernel<5,4><<<((n+4)/5)*((m+3)/4), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,A,*lda,B,*ldb,C,*ldc,(m+3)/4,(n+4)/5);
                else
                     gemm_kernel<4,4><<<((n+3)/4)*((m+3)/4), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,A,*lda,B,*ldb,C,*ldc,(m+3)/4,(n+3)/4);

            }
        }
        else
        {
            if((m) % 5 == 0)
            {
                if((n) % 5 == 0)
                     gemm_kernel_ab<5,5><<<((n+4)/5)*((m+4)/5), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,*alpha,A,*lda,B,*ldb,*beta,C,*ldc,(m+4)/5,(n+4)/5);
                else
                     gemm_kernel_ab<4,5><<<((n+3)/4)*((m+4)/5), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,*alpha,A,*lda,B,*ldb,*beta,C,*ldc,(m+4)/5,(n+3)/4);
            }
            else
            {
                if((n) % 5 == 0)
                     gemm_kernel_ab<5,4><<<((n+4)/5)*((m+3)/4), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,*alpha,A,*lda,B,*ldb,*beta,C,*ldc,(m+3)/4,(n+4)/5);
                else
                     gemm_kernel_ab<4,4><<<((n+3)/4)*((m+3)/4), NTHREADS,0,stream[*StreamIdx]>>>(m,n,k,*alpha,A,*lda,B,*ldb,*beta,C,*ldc,(m+3)/4,(n+3)/4);

            }

        }
    }
    else
    {
        CUBLAS_DGEMMSH_ST_C(StreamIdx, transa, transb, m_p, n_p, k_p, alpha, A_p, shiftA, lda, B_p, shiftB, ldb, beta, C_p, shiftC, ldc);
    }
}

/******************************************************/
