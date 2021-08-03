// File: mpi.cu
// C/Fortran interface to MPI.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes cuda headers
#include <cuda_runtime.h>
// includes project headers
#include "cuda_globals.h"
#ifdef GPUDIRECT
// includes mpi headers
#undef SEEK_SET  // remove compilation errors
#undef SEEK_CUR  // with C++ binding of MPI
#undef SEEK_END
#include <mpi.h>
#include <algorithm>  //namespace std has no member min error otherwise
#endif

/******************************************************/
// CUDA wrappers/kernels for scattering vectors
// to do a blocked  MPI_AlltoAll call

/*
__global__ void curedis_ref(cuDoubleComplex *src, cuDoubleComplex *dst,
                const int NVECTORS, const int NBLOCKS, const int BLOCK_SIZE)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // for each thread,

    if(idx == 0)
    {
	int stridesrc = NBLOCKS * BLOCK_SIZE;
	int stridedst = NVECTORS * BLOCK_SIZE;
        for(int i=0;i<NVECTORS;i++)
        for(int j=0;j<NBLOCKS;j++)
        for(int k=0;k<BLOCK_SIZE;k++)
            dst[j*stridedst+i*BLOCK_SIZE+k] = src[i*stridesrc+j*BLOCK_SIZE+k];
    }
}
*/

/*
template <class T>
__global__ void cuscatter(T *src, T *dst, const int NVECTORS, 
		const int NBLOCKS, const int BLOCK_SIZE)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // for each thread,

    if(idx < BLOCK_SIZE)
    {
	int indexsrc = blockIdx.y*NBLOCKS*BLOCK_SIZE + blockIdx.z*BLOCK_SIZE + idx;
        int indexdst = blockIdx.z*NVECTORS*BLOCK_SIZE + blockIdx.y*BLOCK_SIZE + idx;

	dst[indexdst] = src[indexsrc];
    }
}
*/

/*
// scatters vectors for blocked MPI_Alltoall, type double
extern "C"
void cuda_scatterd_(devptr_t *devptr_src, devptr_t *devptr_dst,
     const int *NVECTORS, const int *NBLOCKS, const int *BLOCK_SIZE)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*BLOCK_SIZE+block.x-1)/block.x,*NVECTORS,*NBLOCKS);

    // device pointers
    double *src = (double *)(*devptr_src);
    double *dst = (double *)(*devptr_dst);

    cuscatter<double><<<grid,block>>>(src,dst,*NVECTORS,*NBLOCKS,*BLOCK_SIZE);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuscatter!" );
}
*/

/*
// scatters vectors for blocked MPI_Alltoall, type cuDoubleComplex
extern "C"
void cuda_scatterz_(devptr_t *devptr_src, devptr_t *devptr_dst,
     const int *NVECTORS, const int *NBLOCKS, const int *BLOCK_SIZE)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*BLOCK_SIZE+block.x-1)/block.x,*NVECTORS,*NBLOCKS);

    // device pointers
    cuDoubleComplex *src = (cuDoubleComplex *)(*devptr_src);
    cuDoubleComplex *dst = (cuDoubleComplex *)(*devptr_dst);

    cuscatter<cuDoubleComplex><<<grid,block>>>(src,dst,*NVECTORS,*NBLOCKS,*BLOCK_SIZE);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuscatter!" );
}
*/

/******************************************************/
// CUDA wrappers/kernels for MPI_Alltoall with CUDA Aware MPI

/*
// MPI_Alltoall
extern "C"
void cuda_mpialltoall_(devptr_t *sendbuf, const int *sendcount, void *sendtype,
     devptr_t *recvbuf, const int *recvcount, void *recvtype, void *comm)
{
#ifdef GPUDIRECT
    MPI_Alltoall((void *)(*sendbuf),*sendcount,*((MPI_Datatype*)sendtype),
                 (void *)(*recvbuf),*recvcount,*((MPI_Datatype*)recvtype),
		 *((MPI_Comm*)comm));
#endif
}
*/


//Code to support GPU-Direct in the VASP MPI methods that perform an
//all-to-all operation. See readme_GPUDirect for more details on how 
//to enable/use it.

void d2d(void *src, void *dst, const int size)
{
    CUDA_ERROR( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice),
		"Failed to copy from device to device!" );
}
void h2d(void *src, void *dst, const int size)
{
    CUDA_ERROR( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice),
		"Failed to copy from host to device!" );
}

void d2h(void *src, void *dst, const int size)
{
    CUDA_ERROR( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost),
		"Failed to copy from device to host!" );
}

// Custom MPI_Alltoall implementation in VASP
void cuda_alltoall_C(double **src_, double **dst_, const int *size_,
     const int *procId_, const int *nProcs_, const int *MAX_)
{
#ifdef GPUDIRECT
	MPI_Status   *status = NULL;
	MPI_Request *request = NULL;

	double *src      = *src_;
	double *dst 	 = *dst_;
	const int size   = *size_;
	const int procId = *procId_;
	const int nProcs = *nProcs_;
	const int MAX    = *MAX_;

	//fprintf(stderr,"src: %p, dst: %p, %d\t%d\t%d\t%d\n",
	//	src, dst, size,procId, nProcs, MAX);

	const int sndCount = size / nProcs;
	int err   	   = 0;
	int reqIdx         = 0;

	int nRequests	   = (sndCount / MAX) + 1; //Number of loops
	    nRequests	  *= 2; 		   //Send and receive 
	    nRequests	  += 16; 	  	   //2 is enough (local copy), 16 is safety

	status  = new MPI_Status [nRequests];
	request = new MPI_Request[nRequests];

	for(int block=0; block < sndCount; block += MAX)
	{
		const int curSndCount = std::min(MAX, sndCount-block);
		const int p	      = block; //Start address offset

		//Startup the Irecvs
		for(int id=1; id < nProcs; id++)
		{
			const int target = (id+procId) % nProcs;

			err = MPI_Irecv(&dst[target*sndCount + p],
				  curSndCount, MPI_DOUBLE, 
				  target, 543, MPI_COMM_WORLD,
				  &request[reqIdx++]);
			if(err != MPI_SUCCESS) fprintf(stderr,"Error in MPI_Irecv \n");
		}

		//Startup the Isnds
		for(int id=1; id < nProcs; id++)
		{
			const int target = (id+procId) % nProcs;
			err = MPI_Isend(&src[target*sndCount + p], 
					curSndCount, MPI_DOUBLE,
					target, 543, MPI_COMM_WORLD,
					&request[reqIdx++]);
		}

		//Do the local copy, reuse MPI functions. Allows universal interface for
		//the fortan side. Use same function if we use GPU pointer or host pointer
		
		//Get the pointer type, if call fails its normal malloc
		//otherwise test if its device or host to determine copy method
		cudaPointerAttributes attrSrc;
		cudaError_t resSrc = cudaPointerGetAttributes(&attrSrc, src);
		cudaPointerAttributes attrDst;
		cudaError_t resDst = cudaPointerGetAttributes(&attrDst, dst);
		cudaGetLastError();

		//fprintf(stderr,"All to all test: src: %d dst: %d \n", resSrc, resDst);


		if(resSrc != cudaSuccess && resDst != cudaSuccess)
		{
			//Use Isend/Irecv since one of the two failed indicating
			//one of the two is non-pinned host memory
			MPI_Isend(&src[procId*sndCount+p], curSndCount, MPI_DOUBLE, 
				       procId, 123, MPI_COMM_WORLD, &request[reqIdx++]);
			MPI_Irecv(&dst[procId*sndCount+p], curSndCount, MPI_DOUBLE, 
				       procId, 123, MPI_COMM_WORLD, &request[reqIdx++]);
		}
		else
		{	
			//Change type such that below functions work nicely
			if(resSrc != cudaSuccess) attrSrc.memoryType = cudaMemoryTypeHost;
			if(resDst != cudaSuccess) attrDst.memoryType = cudaMemoryTypeHost;

			if(attrSrc.memoryType == cudaMemoryTypeDevice && attrDst.memoryType == cudaMemoryTypeDevice)
			{
				d2d(&src[procId*sndCount+p], &dst[procId*sndCount+p], curSndCount*sizeof(double));
			}
			else
			{
    			  if(attrSrc.memoryType == cudaMemoryTypeDevice && attrDst.memoryType == cudaMemoryTypeHost)
			  {

			    //Data is on device and needs to go to the host
			    d2h(&src[procId*sndCount+p], &dst[procId*sndCount+p], curSndCount*sizeof(double));
			  }
			  else if(attrSrc.memoryType == cudaMemoryTypeHost && attrDst.memoryType == cudaMemoryTypeDevice)
			 {
			   //Data is on the host (pinned) and needs to go to the device
			   h2d(&src[procId*sndCount+p], &dst[procId*sndCount+p], curSndCount*sizeof(double));
			 }
			 else
			 { 
			  //Both buffers are on the host
			  for(int i=0; i < curSndCount; i++)
			    dst[procId*sndCount+p+i] = src[procId*sndCount+p+i];
		       	}
		    } //if memType
		} //if resSrc && resDst
		MPI_Waitall(reqIdx,request, status);
		reqIdx = 0;
	}

	delete[] status;
	delete[] request;
#endif
} //end vasp_all_to_all

void cuda_alltoall_host_dev_C(double **src_, double **dst_, const int *size_,
     const int *procId_, const int *nProcs_, const int *MAX_)
{
    cuda_alltoall_C(src_, dst_, size_, procId_, nProcs_, MAX_);
}
void cuda_alltoall_dev_host_C(double **src_, double **dst_, const int *size_,
     const int *procId_, const int *nProcs_, const int *MAX_)
{
    cuda_alltoall_C(src_, dst_, size_, procId_, nProcs_, MAX_);
}
/******************************************************/
