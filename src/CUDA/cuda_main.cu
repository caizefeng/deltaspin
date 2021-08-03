// File: cuda_main.cu
// C/Fortran interface to CUDA C API.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
// includes cuda headers
#include <cuda_runtime.h>
// includes project headers
#include "cuda_globals.h"
#include "Operator.h"

#ifdef __PARA
//#undef SEEK_SET  // remove compilation errors
//#undef SEEK_CUR  // with C++ binding of MPI
//#undef SEEK_END
#include <mpi.h>
#endif



// global variables
int NUM_STREAMS=0;              // number of CUDA streams
cudaStream_t *stream;		// CUDA stream
double *d_reduce, *d_reduce1;	// arrays for parallel reduction
cuDoubleComplex *d_zreduce, *d_zreduce1;  // arrays for parallel reduction
devptr_t *d_ptrs, *d_ptrs1;	// arrays of device pointers
int nPE_, myPE_;


/******************************************************/
// CUDA C wrappers for init, used in VAMP

extern "C"
void cuda_init_C(int *nstreams, int *nsim)
{
    int i;

    /* Get MPI Information */
#ifdef __PARA
    MPI_Comm_size(MPI_COMM_WORLD, &nPE_);
    MPI_Comm_rank(MPI_COMM_WORLD, &myPE_);
#else
    nPE_ = 1; myPE_ = 0;
#endif


    // check number of CUDA streams requested
    if(*nstreams<=0)
    {
        printf("Nstreams is %d\n", *nstreams);
        ERROR( "GPU Library", "Invalid number of CUDA streams:pick a number greater than zero!");
    }
    NUM_STREAMS=*nstreams;
    printf("creating %d CUDA streams...\n",NUM_STREAMS);

    // create CUDA streams
    stream=(cudaStream_t*)malloc(NUM_STREAMS*sizeof(cudaStream_t));
    for(i=0;i<NUM_STREAMS;i++)
	CUDA_ERROR( cudaStreamCreate(&stream[i]), "Failed to create CUDA stream!" );

    // allocate parallel reduction arrays
    CUDA_ERROR( cudaMalloc((void **)&d_zreduce,MAX_THREADS*sizeof(cuDoubleComplex)),
		"Failed to allocate device memory!" );
    CUDA_ERROR( cudaMalloc((void **)&d_zreduce1,MAX_THREADS*sizeof(cuDoubleComplex)),
                "Failed to allocate device memory!" );
    // set parallel reduction arrays
    d_reduce = (double *)d_zreduce;
    d_reduce1 = (double *)d_zreduce1;

    // allocate device pointer arryas
    CUDA_ERROR( cudaMalloc((void **)&d_ptrs,(*nsim)*sizeof(devptr_t)),
		"Failed to allocate device memory!" );
    CUDA_ERROR( cudaMalloc((void **)&d_ptrs1,(*nsim)*sizeof(devptr_t)),
                "Failed to allocate device memory!" );
}

extern "C"
void cuda_destroy_C(void)
{
    int i;
    // destroy CUDA streams
    for(i=0;i<NUM_STREAMS;i++)
        CUDA_ERROR( cudaStreamDestroy(stream[i]), "Failed to destroy CUDA stream!" );
    free(stream);

    // free parallel reduction arrays
    CUDA_ERROR( cudaFree(d_reduce), "Failed to allocate device memory!" );
    CUDA_ERROR( cudaFree(d_reduce1), "Failed to allocate device memory!" );

    // free device pointer arrays
    CUDA_ERROR( cudaFree(d_ptrs), "Failed to allocate device memory!" );
    CUDA_ERROR( cudaFree(d_ptrs1), "Failed to allocate device memory!" );

    // reset device
    CUDA_ERROR( cudaDeviceReset(), "Failed to reset the device!" );
}

// wrapper for MPI API errors
inline void __mpi_error(int status, const char *file, int line)
{
    if(status != MPI_SUCCESS)
    {
        printf("\nMPI Error in %s, line %d: %d\n", file, line, status);
        exit(-1);
    }
}
#define MPI_ERROR(status)   __mpi_error( status, __FILE__, __LINE__ )

int djb2_hash(char *str, int len)
{
    unsigned long hash = 5381;
    for(int i = 0; i < len; i++) {
        hash = hash * 33 ^ str[i];
    }

    //(Open-)MPI wants a non-negative int in MPI_Comm_split
    //we use the lower 31 bits of the long hash
    //INT_MAX does not have its sign bit set, therefore the result will be positive
    int int_hash = hash & INT_MAX;
    assert(int_hash > 0);
    return int_hash;
}

void mpi_get_local_rank(int *local_rank, int *local_size)
{
    // see https://blogs.fau.de/wittmann/2013/02/mpi-node-local-rank-determination/
#ifdef __PARA
    typedef char PName_t[MPI_MAX_PROCESSOR_NAME + 1];//make space for the 0 delimiter

    //step 1: compute hash of local node name
    int pname_len;
    PName_t pname;
    MPI_ERROR(MPI_Get_processor_name(pname, &pname_len));
    //NOTE(ca): we must delimit the processor name because if we only rely on pname_len
    //there is a chance that we have a hash collision where this processor name is a
    //prefix of the other processor name and strncmp with pname_len would falsely return true
    pname[pname_len] = '\0';
    int hash = djb2_hash(pname, pname_len);

    //step 2: split WORLD communicator with hash as "color"
    MPI_Comm collision_comm;
    int collision_size, collision_rank;

    MPI_ERROR(MPI_Comm_split(MPI_COMM_WORLD, hash, myPE_, &collision_comm));
    MPI_ERROR(MPI_Comm_size(collision_comm, &collision_size));
    //collision_rank is 'almost' what we want but may be incorrect in case of collisions
    MPI_ERROR(MPI_Comm_rank(collision_comm, &collision_rank)); 

    //step 3: collect all processor names from local group to detect collisions
    PName_t* local_pnames = new PName_t[collision_size];
    //local_pnames will be sorted by rank, therefore if my_global_rank < other_global_rank then my_local_rank < other_local_rank
    MPI_ERROR(MPI_Allgather(pname, MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR,
        local_pnames, MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR, collision_comm));

    //step 4: cycle through local processor names to make sure the we don't count collisions
    *local_rank = 0;
    *local_size = 0;
    for (int i = 0; i < collision_size; i++) {
        //Note(ca): don't use strncmp here, see above.
        if (strcmp(pname, local_pnames[i]) == 0) {
            if(i < collision_rank) {
                *local_rank += 1;
            }
            *local_size += 1;
        }
    }

    MPI_ERROR(MPI_Comm_free(&collision_comm));

    delete [] local_pnames;
#else
    *local_rank = 0;
    *local_size = 1;
#endif
}

// TODO: replace with init from exact exchange?
extern "C"
void cuda_mpi_init_C(int *CudaDevice)
{
#ifndef EMULATION
    int deviceCount, gpu_rank;
    cudaDeviceProp deviceProp;

    /* Get MPI Information */
#ifdef __PARA
    MPI_Comm_size(MPI_COMM_WORLD, &nPE_);
    MPI_Comm_rank(MPI_COMM_WORLD, &myPE_);
#else
    nPE_ = 1; myPE_ = 0;
#endif

    CUDA_ERROR( cudaGetDeviceCount(&deviceCount), "No CUDA-supporting devices found!" );

    int local_rank, local_size;
    mpi_get_local_rank(&local_rank, &local_size);

    gpu_rank = local_rank * deviceCount / local_size;
    CUDA_ERROR( cudaGetDeviceProperties(&deviceProp, gpu_rank),
		"Device does not support CUDA!" );
    if(deviceProp.major < 1)
    {
        printf( "CUDA ERROR: Devices does not support CUDA!\n");
        cudaDeviceReset();
	exit(1);
    }
    printf("Using device %d (rank %d, local rank %d, local size %d) : %s\n", gpu_rank,*CudaDevice,local_rank, local_size, deviceProp.name);
    CUDA_ERROR(cudaSetDevice(gpu_rank), "Failed to set the device!" );
#endif
}

/******************************************************/
// CUDA C wrappers for thread sync, in VASP

extern "C"
void cuda_device_reset_C(void)
{
    printf("Reseting the CUDA device...\n");
    CUDA_ERROR( cudaDeviceReset(), "Failed to reset the device!" );
}

// synchronze the device
extern "C"
void cuda_devicesynchronize_C(char *msg)
{
    CUDA_ERROR( cudaDeviceSynchronize(), msg );
}

// in fortran source
extern "C"
void threadsynchronize_C(void)
{
    CUDA_ERROR( cudaThreadSynchronize(), "Failed to synchronize the device!" );
}

extern "C"
void cuda_streamsynchronize_C(int *sid)
{
    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream
    CUDA_ERROR( cudaStreamSynchronize(st), "Failed to synchronize the CUDA stream!" );
}

extern "C"
void cuda_all_stream_synchronize_C(void)
{
    CUDA_ERROR( cudaStreamSynchronize(0), "Failed to synchronize all CUDA streams!" );
}

/******************************************************/
