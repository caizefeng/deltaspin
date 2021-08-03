// File: davidson.cu
// C/Fortran interface to GPU port of davidson.F.

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
// CUDA kernels/wrappers used in EDDAV

__global__ void cucorrectcham(cuDoubleComplex *cham_all, const int nsim, 
		const int offset, const int N, const int set_diagonal_flag, 
		const double* diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int global_idx = idx + offset;
    int global_idy = idy + offset;

    //make sure that each thread operates within the nsim x nsim tile
    if ( idx < nsim && idy < nsim && global_idx < N && global_idy < N)
    {
	if(set_diagonal_flag && idx == idy)
	{
            cham_all[global_idx + global_idy * N].x = diagonal[idx];
            cham_all[global_idx + global_idy * N].y = 0.0;
        }
	else
	{
            cham_all[global_idx + global_idy * N].x = 0.0;
            cham_all[global_idx + global_idy * N].y = 0.0;
	}
    }
}

/// reset a nsim x nsim block of cham_all_gpu to zero
/// cham_all_gpu needs to be a N x N matrix with leading dimension N
extern "C"
void cuda_correctcham_C(size_t *cham_all_gpu_ptr, const int *nsim_ptr, const int *offset_ptr,
     const int *N_ptr, const int *set_diagonal_flag_ptr, const double *diagonal_cpu )
{
    cuDoubleComplex *cham_all_gpu = (cuDoubleComplex *)*cham_all_gpu_ptr;
    const int nsim = *nsim_ptr;
    const int offset = *offset_ptr;
    const int N = *N_ptr;
    const int set_diagonal_flag= *set_diagonal_flag_ptr;

    double *diagonal_gpu;
    if(set_diagonal_flag)
    {
        cudaMalloc( &diagonal_gpu, sizeof(double) * nsim);
        cudaMemcpy( diagonal_gpu, diagonal_cpu, sizeof(double) * nsim, cudaMemcpyHostToDevice);
    }

    const int num_threads = 16;
    int num_blocks = ( nsim + num_threads - 1) / num_threads;

    dim3 block(num_threads,num_threads);
    dim3 grid(num_blocks,num_blocks);

    cucorrectcham<<<grid,block>>>(cham_all_gpu,nsim,offset,N,set_diagonal_flag,
	diagonal_gpu);

    if(set_diagonal_flag) cudaFree(diagonal_gpu);
}

/******************************************************/
