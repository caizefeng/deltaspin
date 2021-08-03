!
!   GPU macros
!


! number of streams
#define NV_NUM_STREAMS	32
! number of batches
#define NV_NUM_BATCHES  32
#define NULL_STREAM	-1


! GPU Direct macros
#ifdef GPUDIRECT

! use standard MPI_Alltoall and custom VASP_Alltoall otherwise
!#define USE_COLLECTIVE_GPU

#ifdef USE_COLLECTIVE_GPU
! blocks MPI_alltoall transfers, good for small data transfers
#define GPUDIRECT_BLOCK
#endif

#endif
