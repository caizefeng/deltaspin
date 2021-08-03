// File: cuda_fft.cu
// C/Fortran interface to memory management.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// include linux headers
#include <unistd.h>  // for getpagesize()
// includes cuda headers
#include <cuda_runtime.h>
// includes project headers
#include "cuda_globals.h"

/******************************************************/
// CUDA C wrappers for pinned malloc, in VASP

//Use these defines to control how pinned memory is allocated:
//NVPINNED, if this is enabled we actually use pinned memory
//if you want 'normal' memory instead disable this define
//NVREGISTERSELF, if this one and NVPINNED is set then we do not
//use cudaFreeHost but instead register and unregister the
//memory ourself. This is advised for less recent MPI installations
//
#define NVPINNED
#define NVREGISTERSELF

extern "C"
int nvpinnedmalloc_C(void **ptr, size_t *size_)
{
    size_t size = *size_;

#ifdef NVPINNED
    #ifdef NVREGISTERSELF
    //Custom implementation using posix_memalign and host register. This method is
    //preferred for older MPI implementations such that it can hook into/notice the malloc
    //and register calls.
    if(posix_memalign(ptr,getpagesize(),size))
    {
        fprintf(stderr,"Failed to allocate aligned host memory, requested %ld\n", size);
        return -1;
    }
    CUDA_ERROR( cudaHostRegister(*ptr,size,cudaHostRegisterMapped),
		"Failed to register pinned memory!" );
    #else
    CUDA_ERROR( cudaMallocHost(ptr,size), "Failed to allocate pinned memory!" );
    #endif
#else
    void *temp = NULL;
    temp = (void*)malloc(sizeof(char)*size);

    if(!temp)
    {
        fprintf(stderr,"Failed to allocate host memory, requested: %ld\n", size);
        return -1;
    }
    *ptr = temp;
#endif
    return 0;
}

//The function to call to unregister and free allocated pinned memory
extern "C"
int nvpinnedfree_C(void **ptr)
{
#ifdef NVPINNED
    #ifdef NVREGISTERSELF
    CUDA_ERROR( cudaHostUnregister(*ptr), "Failed to unregister pinned memory!" );
    free(*ptr);
    #else
    CUDA_ERROR( cudaFreeHost(*ptr), "Failed to free pinned memory!" );
    #endif
#else
    free(*ptr);
#endif
    return 0;
}

/******************************************************/
// CUDA C wrappers for malloc, in VASP

/* Allocate one large piece of memory, this can then be divided over multiple memory buffers using 'allocate_gpu_memory_batched_request'
 * This reduces the amount of calls to cudaMalloc and therefor results in faster memory allocations
 */
extern "C"
int allocate_gpu_memory_batched_init_C(const size_t *nBytes, size_t *offset,
    size_t *remaining, devptr_t *devPtr)
{
    // Allocate a large piece of memory
    #define SAFETY 1.10

    void *dptr;
    size_t request = *nBytes*SAFETY;
    CUDA_ERROR( cudaMalloc(&dptr,request), "Failed to allocate device memory!" );

    // Allocation succeeded, reset offset and store allocated size
    *devPtr = (devptr_t)dptr;
    *offset = 0;
    *remaining = request;
    return 0;
}

//CC 2.X and 3.X ,128 bytes
//#define ALLIGN_IN_BYTES 512
#define ALLIGN_IN_BYTES 32
//Return the number of elements (of type uint) to be padded
//to get to the correct address boundary
static size_t getGlobalMemAllignmentPadding(size_t n)
{
    const size_t allignBoundary = ALLIGN_IN_BYTES*sizeof(char);
    size_t A = n % allignBoundary;

    if(A == 0) return 0;
    return allignBoundary - A;
}

extern "C"
int allocate_gpu_memory_batched_request_C(const size_t *nBytes, size_t *curOffset_,
    size_t *remaining_, devptr_t *bufferPtr, devptr_t *devicePtr)
{
    //Request a piece of memory from our large array
    size_t remaining = *remaining_;
    size_t curOffset = *curOffset_;

    //Make sure our current offset is already alligned
    //assert(curOffset % (ALLIGN_IN_BYTES) == 0);

    void *tPtrA = (void *)(*devicePtr);
    char *tPtrB = (char *)(*bufferPtr);

    //Assign the memory
    tPtrA      = tPtrB+curOffset;
    *devicePtr = (devptr_t)tPtrA;

    //Compute the new offset, and possibly pad it
    curOffset += *nBytes;

    const int padding = getGlobalMemAllignmentPadding(curOffset);
    curOffset += padding;

    remaining -= (curOffset-*curOffset_);

    //assert(remaining > 0);

    //Assign
    *remaining_ = remaining;
    *curOffset_ = curOffset;
    return 0;
}

/*
// request device memory info
extern "C"
void cuda_memgetinfo_(int *rank)
{
    // print memory info
    size_t total,free;
    CUDA_ERROR( cudaMemGetInfo(&free,&total), "Failed to get mem info!" );
    printf("rank %d, Device Memory Info:\n",*rank);
    printf("Total: %.1f MB\nFree: %.1f MB\nUsed: %.1f MB\n\n",total/MB,free/MB,(total-free)/MB);
}
*/

// allocate device memory
extern "C"
void cublas_alloc_safety_C(int *n, size_t *elemSize, void **devPtr)
{
    //nvp_start_(&NVP_MALLOC);
    // allocate device memory
    cudaError_t status = cudaMalloc(devPtr,(*n)*(*elemSize));
    if(status != cudaSuccess)
    {
	// print memory info
        size_t total,free,req;
        cudaMemGetInfo(&free,&total);
        req = (*n)*(*elemSize);
        printf("Device Memory Info:\n");
        printf("Total: %.1f MB\nFree: %.1f MB\nUsed: %.1f MB\nRequested: %.1f MB\n",
                total/MB,free/MB,(total-free)/MB,req/MB);
	// handle error status
        CUDA_ERROR( status, "Failed to allocate device memory!" );
    }
    //nvp_stop_(&NVP_MALLOC);
}

extern "C"
int cublas_free_C(const devptr_t *devicePtr)
{
    void *tPtr = (void*)(*devicePtr);
    //nvp_start_(&NVP_MALLOC);
    //CUBLAS_ERROR( cublasFree(tPtr), "Failed to execute cublasFree!" );
    CUDA_ERROR( cudaFree(tPtr), "Failed to free device memory!" );
    //nvp_stop_(&NVP_MALLOC);
    return 0;
}

/******************************************************/
// CUDA C wrappers for memcpy operations, in VASP

// copy from device to device with given stream, asynchronous to host
extern "C"
void cuda_memcpydtod_C(int *sid, void **dst, void **src, int *n, size_t *size)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    CUDA_ERROR( cudaMemcpyAsync(*dst,*src,(*n)*(*size),cudaMemcpyDeviceToDevice,st),
		"Failed to copy from device to device async!" );
    // synchronize device
    if(*sid < 0) CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device!" );
}

// copy from host to device with given stream, asynchronous to host
extern "C"
void cuda_memcpyhtod_C(int *sid, void **dst, void *src, int *n, size_t *size)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    CUDA_ERROR( cudaMemcpyAsync(*dst,src,(*n)*(*size),cudaMemcpyHostToDevice,st),
                "Failed to copy from host to device async!" );
    // synchronize device
    if(*sid < 0) CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device!" );
}

// copy from device to host with given stream, asynchronous to host
extern "C"
void cuda_memcpydtoh_C(int *sid, void *dst, void **src, int *n, size_t *size)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream
    CUDA_ERROR( cudaMemcpyAsync(dst,*src,(*n)*(*size),cudaMemcpyDeviceToHost,st),
                "Failed to copy from device to host async!" );
    // synchronize device
    if(*sid < 0) CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device!" );
}

// same as above, but with shift arguments
// needed because of device arrays of type cuDoubleCompled,double in fortran source
extern "C"
void cuda_memcpydtohshift_C(int *sid, char *dst, int *shiftdst, char **src, int *shiftsrc,
     int *n, size_t *size)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    size_t shiftd=(*shiftdst)*(*size);
    size_t shifts=(*shiftsrc)*(*size);
    CUDA_ERROR( cudaMemcpyAsync((void*)((char*)dst+shiftd),(void*)((char*)*src+shifts),(*n)*(*size),cudaMemcpyDeviceToHost,st),
		"Failed to copy from device to host async!" );
    // synchronize device
    if(*sid < 0) CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device!" );
}

/******************************************************/
// CUDA C wrappers for memset operations, in VASP

// memset devPtr, synchronous to host
extern "C"
void cuda_memset_C(void **devPtr, int *value, int *n, size_t *size)
{
    CUDA_ERROR( cudaMemset(*devPtr,*value,(*n)*(*size)), "Failed to execute cudaMemset!" );
}

// memset devPtr with given stream, asynchronous to host
extern "C"
void cuda_memsetasync_C(int *sid, void **devPtr, int *value, int *n, size_t *size)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream
    CUDA_ERROR( cudaMemsetAsync(*devPtr,*value,(*n)*(*size),st),
		"Failed to execute cudaMemsetAsync!" );
}

/******************************************************/
