 /* \brief Contains ports of FOCK_ACC and FOCK_FORCE, and supporting routines.
 */

#include <stdio.h>
#include <algorithm>
#include "cufftXt.h"
#include "cuda.h"
#include "cuComplex.h"
#include "cuda_utils.cu"
#include "cublas_v2.h"
#include "Operator.h"
#ifdef __PARA
//#undef SEEK_SET  // remove compilation errors
//#undef SEEK_CUR  // with C++ binding of MPI
//#undef SEEK_END
#include <mpi.h>
#endif
#include <assert.h>

#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

#include <cuda_profiler_api.h>
#include "kernels.h"
#include "cufft_int.h"
#include "cub/block/block_reduce.cuh"

/* Some useful MACROs */
#define MAX(a,b) (((a) < (b)) ?  (b) : (a))
#define MIN(a,b) (((a) > (b)) ?  (b) : (a))

#ifndef NUM_FFT_STREAM
  #define NUM_FFT_STREAM 1
#endif

#ifndef FFT_CACHE_SIZE
  #define FFT_CACHE_SIZE 12
#endif 

#ifndef NUM_ION_STREAM
  #define NUM_ION_STREAM 8
#endif

#ifndef MAX_DGEMM_BATCH
  #define MAX_DGEMM_BATCH 64
#endif

#ifdef __PARA
#define MPI_CALL(function) {\
int err = function; \
if (err != MPI_SUCCESS) \
  fprintf(stderr, "MPI Error [%s,%d]: %d \n", __FILE__,  __LINE__, err); \
}
#endif

//Set a non-zero value to print matrix sizes information
//#define doDbgPrint 1

//Use 0, to disable profiling, use value larger than 0 to enable the 
//cuda-profiler and stop after DO_PROFILE_RUN 'bq' iterations.
//#define DO_PROFILE_RUN  4
//#define PROFILE_FOCK_ACC

//Set this define to force synchronous execution. Note ideally this 'checkError' function is placed
//after each kernel launch to check for launch-errors
//#define DEBUG_KERNELS

//To get profiler information, in a seperate console do:
//nvprof --profile-from-start-off --profile-all-processes -o output.%h.%p 
//and import the result file in the visual profiler

/* This will store some shared memory data in single precision, see forward_project */
//#define MIXED_PREC

// singlefft is broken
//#define SINGLEFFT

/* Some 'terms':
  rwav -- wavefunctions in real space
  kwav -- wavefunctions in k-space (C/CUDA name)
  CW -- wavefunctions in k-space (fortran name)
*/

/** Struct to hold global parameters.
 *
 * This structure holds problem-wide parameters that
 * should not change within an electronic or ionic
 * minimization step. 
 */
typedef struct global_desc{
//  int_p* rotmap_d; //!<         map indexing atoms that are taken into each other when the symmetry operation is applied
  //int_p* lps_d; //!<            sizes for rotation operation
  double* fermi_weights_g; //!< fermi weights of (bands, kpoints) [global]
  int ntype; //!<               number of types of ions
  int* nion; //!<               number of ions of each type
  int nion_tot; //!<            total number of ions
  int nband; //!<               number of bands
  int nkpoint_full; //!<        total number of kpoints
  int nkpoint_red; //!<         number of kpoints in the wavespin struct (outside requires rotation)
  int* equiv; //!<              equivalence relationships between kpoints
  //int lmax; //!<                size of lps
  //int lps_max; //!<             maximum value of lps (and final dimension of rotation_d)
  int lmdim; //!<               leading dimension of arrays like CDIJ"
  int lmmax_aug; //!<           part of dimension of trans_d
  int irmax; //!<               size of a few things
  double rspin; //!<            spin multiplicity
  double *kpoint_weights; //!<  symmetry weight for each kpoint
  double_p* trans_d; //!<       transformation matrix shape(trans) = (lmdim, lmdim, lmmax_aug, ntype)
} global_desc;

/** Struct to hold a wavefunction.
 *
 * Holds an arbitrary number of bands with real and k-space representations and projectors.
 */
typedef struct wavefunction_desc{
  cuDoubleComplex_p* kwav_d; //!< wavefunction in k-space. dimension is (nkwav, nband)
  cuDoubleComplex_p* rwav_d; //!< wavefunction in real space.  dimension is (nrwav, nband)
  cuDoubleComplex_p* proj_d; //!< wavefunction projection.  dimension is (nproj, nband)
  cuDoubleComplex_p* phase_d; //!< k-point phase shift.  dimension is (nkwav)
  int_p* index_d; //!<            indexes to extract points from kwav needed for FFT to form rwav.  dimension is (nindex)
  double* fermi_weights_l; //!<   fermi weights of (bands, kpoints) [local]
  int grid[3]; //!<               dimension of the 3-dimensional grid
  int nkwav; //!<                 size of k-space wavefunction
  int nrwav; //!<                 size of the real-space wavefunction
  int nproj; //!<                 size of the wavefunction projections
  int nindex; //!<                size of the index array
  int ntype; //!<                 number of types of ions
  int* nion; //!<                 number of ions of each type
  int nion_tot; //!<              total number of ions
  int* lmmax; //!<                number of distinct lm-quantum numbers (for each type)
  int linv; //!<                  take the complex conjugate when going k to real?
  int lshift; //!<                take the complex conjugate when going k to real?
  int nband; //!<                 number of bands
  int npos; //!<                  position of first band among all bands
} wavefunction_desc;

/** Struct to hold a projector. 
 *
 * Holds a real-space projector and corresponding phase vector
 */
typedef struct projector_desc{
  double_p* rproj_d; //!<          real-space projectors
  cuDoubleComplex_p* phase_d; //!< Phase vector
  int nproj; //!<                  size = nion * lmmax (for ntype = 1)
  int ntype; //!<                  number of types of ions
  int *nion; //!<                  number of ions of each type
  int nion_tot; //!<               total number of ions
  int *lmmax; //!<                 number of distinct lm-quantum numbers (for each type)
  int npro; //!<                   sum_i ( nion_i * lmmax_i)
  int irmax; //!<                  array size
  double rinpl; //!<               scale factor
  int* nlimax; //!<                number of elements in each index vector (host)
  int_p* nlimax_d; //!<            number of elements in each index vector (device)
  int_p* rproj_offsets_d; //!<     offsets to rproj for batching reverse/forwards project kernels (device)
  int_p* nli_d; //!<               index vector
  // npro_ni(i) is just (lmmax*i) (for ntype = 1)
} projector_desc;

/* cuBLAS context from cuda_main.cu */
extern cublasHandle_t hcublas;

/* MPI info */
static int nproc_k, myproc_k;
static MPI_Comm mycomm_k;

/* Initialized State */
static bool fock_isInitialized = false;

/* Convinient thread and block definitions for kernel executions */
static dim3 threads(128), athreads(256);
static dim3 ablocks(512);

/* GPU work buffers */
/* inputs */
static double_p* rotation_d = NULL;
static double_p* potfak_d = NULL;
static cuDoubleComplex_p* dproj_d = NULL;

/* outputs */
static cuDoubleComplex_p* cxi_d = NULL;
static cuDoubleComplex_p* ckappa_d = NULL;
static double_p* sif_d = NULL;
static float_p* sif2_d = NULL;
static float_p* forhf_d = NULL;

/* catalysts */
static cuDoubleComplex_p* charge_d = NULL;
static cuDoubleComplex_p* crholm_d = NULL;
static cuDoubleComplex_p* cdij_d = NULL;
static cuDoubleComplex_p* cdlm_d = NULL;
static cuDoubleComplex_p* ctmp_d = NULL;
static double_p* rtmp_d = NULL;
static double_p* itmp_d  = NULL;
static double_p* weights_d = NULL;
static double_p* weights2_d = NULL;
static cuComplex_p* ftmp_d = NULL;

/* FFT variables */
/* Streams and Events */
static cudaStream_t streams[NUM_FFT_STREAM] = {0};
static CufftPlanCache<FFT_CACHE_SIZE, NUM_FFT_STREAM> stream_plan_cache;

/* Helper functions */
//#ifdef SINGLEFFT
//static inline void stream_ffts(cuComplex* data, int* size, int dir, int num);
//#else
static inline void stream_ffts(cuDoubleComplex* data, int* size, int dir, int num);
//#endif
static inline void make_fock_context(void);

cudaStream_t ion_streams[NUM_ION_STREAM] = {0};
static double_p* work1_d[NUM_ION_STREAM] = {NULL};
static double_p* work2_d[NUM_ION_STREAM] = {NULL};

double **d_ParamArray[NUM_ION_STREAM] = {NULL};

void checkError(const int line, const char *file)
{
    #ifdef DEBUG_KERNELS
     cudaDeviceSynchronize();
    #endif    
    cudaError_t cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) 
    {
      printf ("CUDA ERROR in %s:%d  Error: %d %s  \n",file, line, cudaStat, cudaGetErrorString(cudaStat));
      exit(cudaStat);
    }    
}    
    
int countSameSizeDgemms(
	const int index, 
	const int maxIndex, 
	const int size, 
	const int* sizes
) {
    int count = 1;
    while(
         (index + count < maxIndex)
      && (count < MAX_DGEMM_BATCH)
      && (size == sizes[index + count])
    ) {
      count++;
    }
    return count;
}

static __global__ void setup_batch_k(
  double** param_d, 
  double* A, 
  double* B,
  double* C, 
  int m, int n, int k, int num){
  if (blockIdx.x == 0){
    if (threadIdx.x < num)
      param_d[threadIdx.x]         = A + threadIdx.x * m * k;
  }else if (blockIdx.x == 1){
    if (threadIdx.x < num)
      param_d[threadIdx.x + num]   = B + threadIdx.x * n * k;
  } else if (blockIdx.x == 2){
    if (threadIdx.x < num)
      param_d[threadIdx.x + 2*num] = C + threadIdx.x * m * n;
  }
}


void batchedDgemm(
	cublasHandle_t handle, 
	cublasOperation_t transa, 
	cublasOperation_t transb, 
	int m, 
	int n, 
	int k, 
	const double *alpha, 
	double *A, 
	int lda, 
	double *B, 
	int ldb, 
	const double *beta, 
	double *C, 
	int ldc,
	int batchSize,
	int ion_stream_id
) {
  cublasSetStream(handle, ion_streams[ion_stream_id]);
  checkError(__LINE__,__FILE__);
  if(batchSize > 1) {
    setup_batch_k<<<3, batchSize, 0, ion_streams[ion_stream_id]>>>(d_ParamArray[ion_stream_id], A, B, C, m, n, k, batchSize);
    cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, 
            (const double**)&d_ParamArray[ion_stream_id][0*batchSize],
            lda, 
            (const double**)&d_ParamArray[ion_stream_id][1*batchSize], 
            ldb, beta, 
            &d_ParamArray[ion_stream_id][2*batchSize], 
            ldc, batchSize);
    checkError(__LINE__,__FILE__);
  } else {
    cublasDgemm(handle, transa, transb, 
	              m, n, k, alpha, 
                A,
                lda, 
                B,
                ldb, beta, 
                C,
                ldc);
    checkError(__LINE__,__FILE__);
  }
}

__global__ void reverse_project_all(
  cuDoubleComplex *charge,
  cuDoubleComplex* crholm,
  double* rproj,
  cuDoubleComplex* phase,
  int* nli,
  int nr,
  int irmax,
  int lmmax,
  int* nlimax_d,
  int* rproj_offsets_d,
  int npro,
  int nband,
  int nion,
  int ion_global)
{
  int index;
  cuDoubleComplex foo;

  for (int ion = blockIdx.y; ion < nion; ion+=gridDim.y){
	int nlimax = nlimax_d[ion+ion_global];
	double *rproj_p = rproj + rproj_offsets_d[ion+ion_global];
    for (int band = blockIdx.x; band < nband; band+=gridDim.x){
      for (int isr = threadIdx.x + blockIdx.z*blockDim.x; isr < nlimax; isr+= blockDim.x * gridDim.z){
        foo.x = 0.; foo.y = 0.;
        for (int lm = 0; lm < lmmax; lm++){
          foo.x += crholm[lm + ion * lmmax + band * npro].x * rproj_p[isr + nlimax*lm];
          foo.y += crholm[lm + ion * lmmax + band * npro].y * rproj_p[isr + nlimax*lm];
        }
        index = band * nr  + nli[isr + ion * irmax] - 1;
        atomicAddDouble(&charge[index].x, phase[isr + ion * irmax].x * foo.x
                         + phase[isr + ion * irmax].y * foo.y);
        atomicAddDouble(&charge[index].y, phase[isr + ion * irmax].x * foo.y
                         - phase[isr + ion * irmax].y * foo.x);
      }
    }
  }
}

template <int blDim>
__global__ void forward_project_all(
  cuDoubleComplex* crholm,
  cuDoubleComplex *charge,
  double* rproj,
  cuDoubleComplex* phase,
  int* nli,
  double scale,
  int nr,
  int irmax,
  int lmmax,
  int* nlimax_d,
  int* rproj_offsets_d,
  int npro,
  int nband,
  int nion,
  int ion_global)
{
  int index;
#ifdef MIXED_PREC
  extern __shared__ cuComplex charge_s[];
  cuComplex single;
#else
  extern __shared__ cuDoubleComplex charge_s[];
  cuDoubleComplex single;
#endif
  cuDoubleComplex foo;
  cuDoubleComplex charge_l, phase_l;

  typedef cub::BlockReduce<double,blDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storageX;
  __shared__ typename BlockReduce::TempStorage temp_storageY;

  for (int ion = blockIdx.y; ion < nion; ion+=gridDim.y){
	int nlimax = nlimax_d[ion+ion_global];
	double *rproj_p = rproj + rproj_offsets_d[ion+ion_global];
    for (int band = blockIdx.x; band < nband; band+=gridDim.x){
      for (int isr = threadIdx.x + threadIdx.y*blockDim.x; isr < nlimax; isr+= blockDim.x*blockDim.y ){
        index = band * nr  + nli[isr + ion * irmax] - 1;
        charge_l = charge[index];
        phase_l = phase[isr + ion * irmax];
        single.x = (charge_l.x * phase_l.x
                   -charge_l.y * phase_l.y);
        single.y = (charge_l.x * phase_l.y
                   +charge_l.y * phase_l.x);
        charge_s[isr] = single;
      }
      __syncthreads();
      for (int lm = threadIdx.y; lm < lmmax; lm+=blockDim.y){
        foo.x = 0.; foo.y = 0.;
        for (int isr = threadIdx.x; isr < nlimax; isr += blockDim.x){
          foo.x += rproj_p[isr + nlimax*lm] * charge_s[isr].x;
          foo.y += rproj_p[isr + nlimax*lm] * charge_s[isr].y;
        }
        double aggregateX = BlockReduce(temp_storageX).Sum(foo.x);
        if ( threadIdx.x == 0 ) {
          crholm[lm + ion * lmmax + band * npro].x += aggregateX*scale;
        }
        double aggregateY = BlockReduce(temp_storageY).Sum(foo.y);
        if ( threadIdx.x == 0 ) {
          crholm[lm + ion * lmmax + band * npro].y += aggregateY*scale;
        }
      }
    }
  }
}

//This returns a launchable thread-config for the calc_dmmll kernel
void get_calc_dmmll_config(const int nbands, const int nions, const int lmmax_wav,
                           dim3 &grid, dim3 &block)
{
  grid.x = min(65535, nbands);
  grid.y = min(65535, nions);
  
  //Maximum number of threads is limited, make sure we do not pass it
  //We keep to the limits of the 2.x architecture
  block.x = min(512, lmmax_wav);
  
  //Maximum number of threads we can have in a block
  const int tmax = 1024 / block.x; 
  block.y = min(max(1,tmax), lmmax_wav);
}



/** Maps select elements from k-space wavefunction into real-space vector before fft.
 *
 * [NON-OPTIMIZED]
 */
static __global__ void inflate_all_k(cufftDoubleComplex *rwav, //!< real-space destination 
                                     cufftDoubleComplex *kwav, //!< k-space source
                                     int *index,               //!< permutation vector for the inflation
                                     int num,                  //!< number of buffers to work on
                                     int cr_size,              //!< size of real-space buffer
                                     int c_size,               //!< size of k-space buffer
                                     int nindex,               //!< size of permutation vector
                                     int conj                  //!< take a complex conjugate?
                                    ){
  int i, j, ind;

  cuDoubleComplex *rwav_p, *kwav_p;
  
  if (conj){    
    /* Loop over bands */
    for (j = blockIdx.x; j < num; j+= gridDim.x){
      /* Create band-specific aliases */
      rwav_p = rwav + j*cr_size; kwav_p = kwav + j*c_size;
      /* Extract relavent values */
      for (i = threadIdx.x; i < nindex; i+= blockDim.x){
        ind = index[i]-1;
        rwav_p[ind].x = kwav_p[i].x;
        rwav_p[ind].y = - kwav_p[i].y;
      }
    }
  }else{
    /* Loop over bands */
    for (j = blockIdx.x; j < num; j+= gridDim.x){
      /* Create band-specific aliases */
      rwav_p = rwav + j*cr_size; kwav_p = kwav + j*c_size;
      /* Extract relavent values */
      for (i = threadIdx.x; i < nindex; i+= blockDim.x){
        ind = index[i]-1;
        rwav_p[ind].x = kwav_p[i].x;
        rwav_p[ind].y = kwav_p[i].y;
      }
    }
  }
  return;
}

/* As above, but with a phase shift */
static __global__ void inflate_all_shift_k(
                                     cufftDoubleComplex *rwav, //!< real-space destination 
                                     cufftDoubleComplex *kwav, //!< k-space source
				     cufftDoubleComplex *phase, //!< phase shift
                                     int *index,               //!< permutation vector for the inflation
                                     int num,                  //!< number of buffers to work on
                                     int cr_size,              //!< size of real-space buffer
                                     int c_size,               //!< size of k-space buffer
                                     int nindex,               //!< size of permutation vector
                                     int conj                 //!< take a complex conjugate?
                                    ){
  int i, j, ind;

  cuDoubleComplex *rwav_p, *kwav_p, *phase_p;
  
  if (conj){    
    /* Loop over bands */
    for (j = blockIdx.x; j < num; j+= gridDim.x){
      /* Create band-specific aliases */
      rwav_p = rwav + j*cr_size; kwav_p = kwav + j*c_size; phase_p = phase;
      /* Extract relavent values */
      for (i = threadIdx.x; i < nindex; i+= blockDim.x){
        ind = index[i]-1;
        rwav_p[ind].x = kwav_p[i].x * phase_p[i].x + kwav_p[i].y * phase_p[i].y;
        rwav_p[ind].y = kwav_p[i].x * phase_p[i].y - kwav_p[i].y * phase_p[i].x;
      }
    }
  }else{
    /* Loop over bands */
    for (j = blockIdx.x; j < num; j+= gridDim.x){
      /* Create band-specific aliases */
      rwav_p = rwav + j*cr_size; kwav_p = kwav + j*c_size; phase_p = phase;
      /* Extract relavent values */
      for (i = threadIdx.x; i < nindex; i+= blockDim.x){
        ind = index[i]-1;
        rwav_p[ind].x = kwav_p[i].x * phase_p[i].x - kwav_p[i].y * phase_p[i].y;
        rwav_p[ind].y = kwav_p[i].x * phase_p[i].y + kwav_p[i].y * phase_p[i].x;
      }
    }
  }
  return;
}



/** Rotates the wavefunction projections based on a symmetry relation.
 *
 * This has been removed in favor of rotating on the CPU.  I'm leaving it
 * here for the future in case someone wants to revive it.
 * [NON_OPTIMIZED]
 */
#if 0
static __global__ void rotate_cproj_k(cuDoubleComplex *cproj_new, cuDoubleComplex *cproj_old, double* rot_matrix,
                               int* rotmap, int* lps, int num_channels, int lmmax, int mmax, int nion){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
    
  int i, j, k, L, m, nprop, ind;
  /* Loop over ions */
  for (i = idx; i < nion; i+=nthreads){
    nprop = lmmax * (rotmap[i]-1);
    ind = 0;
    /* Not quite sure what channels are */
    for (j = 0; j < num_channels; j++){
      L = lps[j];
      /* Matrix-vector multiply, representing a transformation (rotation) */
      for(k = 0; k < 2*L+1; k++){
        cproj_new[i*lmmax + ind+k].x = 0; cproj_new[i*lmmax + ind+k].y = 0;
        for (m = 0; m < 2*L+1; m++){
          cproj_new[i*lmmax + ind+k].x += rot_matrix[L*mmax*mmax + m*mmax + k] * cproj_old[nprop + ind + m].x;
          cproj_new[i*lmmax + ind+k].y += rot_matrix[L*mmax*mmax + m*mmax + k] * cproj_old[nprop + ind + m].y;
        }
      }
      ind += 2*L+1;
    }
  }
}
#endif

/** Computes the augmentation to the charge
 *
 * Dots two wavefunction projections and multiplies by transformation matrix.  Operates on 
 * nbands of the first wavefunction, but only one type at a time.
 * [NON-OPTMIZIED]
 */
static __global__ void aug_charge_trace_k(
                                          const cuDoubleComplex *__restrict__ c1, //!< chunk of wavefunction bands
                                          const cuDoubleComplex *__restrict__ c2, //!< one wavefunction band 
                                          const double *__restrict__ Qij, //!< transformation matrix
                                          cuDoubleComplex *__restrict__ crholm, //!< output
                                          int nion, //!< number of ions of this type 
                                          int lmmax_w, //!< lmmax for wavefunction of this type
                                          int npro_w, //!< size of wavefunction projection
                                          int lmmax_p, //!< lmmax for projectors of this type
                                          int npro_p, //!< size of projectors 
                                          int mat_size, //!< extent of first 2 dims of trans
                                          int nband //!< number of bands in c1
                                         ){
  cuDoubleComplex tmp;

  const cuDoubleComplex *c1_p, *c2_p;
  cuDoubleComplex *crholm_p;
  cuDoubleComplex localSum;
  
  int i, j, k, l, band;
  for (band = blockIdx.y; band < nband; band += gridDim.y){
    c1_p = c1 + band * npro_w; c2_p = c2; crholm_p = crholm + band * npro_p;
    for (i = blockIdx.x; i < nion; i+= gridDim.x){
      for (l = threadIdx.x; l < lmmax_p; l+=blockDim.x){
        localSum.x = 0.0;
        localSum.y = 0.0;
        for (j = threadIdx.y; j < lmmax_w; j+=blockDim.y){
          for (k = threadIdx.z; k < lmmax_w; k+=blockDim.z){ 
            tmp.x = c1_p[i*lmmax_w + j].x * c2_p[i*lmmax_w + k].x + c1_p[i*lmmax_w + j].y * c2_p[i*lmmax_w + k].y;
            tmp.y = c1_p[i*lmmax_w + j].y * c2_p[i*lmmax_w + k].x - c1_p[i*lmmax_w + j].x * c2_p[i*lmmax_w + k].y;
            localSum.x += tmp.x * Qij[l*mat_size*mat_size + j*mat_size + k];
            localSum.y += tmp.y * Qij[l*mat_size*mat_size + j*mat_size + k];
          }
        }
        atomicAddDouble(&(crholm_p[i*lmmax_p + l].x),localSum.x);
        atomicAddDouble(&(crholm_p[i*lmmax_p + l].y),localSum.y);
      }
    }
  }
}

/** Element-wise multiply to convolve in k-space (DP). */
template <class T>
static __global__ void apply_gfac_k(T *c, double* potfak, int n, int nband){
  T *c_p;

  int i, band;
  for (band = blockIdx.x; band < nband; band += gridDim.x){
    c_p = c + band * n;
    for (i = threadIdx.x; i < n; i+= blockDim.x){
      c_p[i].x *= potfak[i];
      c_p[i].y *= potfak[i];
    }
  }
}

/** Element-wise multiply to convolve in k-space (DP).
 * NOTE(sm): This kernel is bandwidth bound, so it pays off not to read potfak nband times
 * but only once. This is only needed as a fall back, if the combined gfac and gfac_der kernel
 * cannot be use (*loverl == false)
 */
template <class T>
static __global__ void apply_gfac_k_localpotfak(T *c, double* potfak, int n, int nband){
  T *c_p = c;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if ( tid < n ) {
    double potfak_s = potfak[tid];
    for (int band = 0; band < nband; ++band){
      T res = c_p[tid + band*n];
      res.x *= potfak_s;
      res.y *= potfak_s;
      c_p[tid + band*n] = res;
    }
  }
}

/** Sum the product of a element-wise product real potential and magnitude of complex vector
 *
 * [OPTMIZIED] 
 */
//Will split up the loop over 'n' over multiple blocks.
template <class T, int blDim>
static __global__ void apply_gfac_der_k(T *c, 
                                        double* potfak, 
                                        double* sif, 
                                        double* weights, int n){
  const int tid         = threadIdx.x;
  double mySum          = 0;
  const int binIdx      = blockIdx.y % 7;
  
  const T *c2           = &c     [blockIdx.x*n];
  const double *potfak2 = &potfak[binIdx*n];
  
  //Compute the offsets for start and end
  const int start  = blockIdx.y / 7;
  const int start2 = (n / (gridDim.y / 7))*start;
  const int end    = min(n, (n / (gridDim.y / 7))*(start+1));
  
  for (int j = start2+tid; j < end; j+= blockDim.x)
    mySum += potfak2[j] * (c2[j].x * c2[j].x + c2[j].y * c2[j].y);
  
  //Perform the reduction
  typedef cub::BlockReduce<double,blDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double aggregate = BlockReduce(temp_storage).Sum(mySum);
  if(tid == 0)
     atomicAddDouble(&sif[binIdx], aggregate/ n * weights[blockIdx.x]);
}

/** Sum the product of a element-wise product real potential and magnitude of complex vector
 *  NOTE(sm): the apply_gfac_der_k kernel traverses the same data like he bandwidth bound
 *  apply_gfac_k kernel. So, it can give the result of the latter (more or less) for free
 *  because it is bandwidth bound itself.
 *
 * [OPTIMIZED]
 */
//Will split up the loop over 'n' over multiple blocks.
template <class T, int blDim>
static __global__ void apply_gfac_k_and_gfac_der_k(T *c,
                                        double* potfak,
                                        double* sif,
                                        double* weights, int n){
  const int tid         = threadIdx.x;
  double mySum          = 0;
  const int binIdx      = blockIdx.y % 7;

  T *c2           = &c     [blockIdx.x*n];
  const double *potfak2 = &potfak[binIdx*n];

  //Compute the offsets for start and end
  const int start  = blockIdx.y / 7;
  const int start2 = (n / (gridDim.y / 7))*start;
  const int end    = min(n, (n / (gridDim.y / 7))*(start+1));

  for (int j = start2+tid; j < end; j+= blockDim.x) {
    mySum += potfak2[j] * (c2[j].x * c2[j].x + c2[j].y * c2[j].y);
    T res = c2[j];
	res.x *= potfak2[j];
	res.y *= potfak2[j];
	c2[j] = res;
  }

  //Perform the reduction
  typedef cub::BlockReduce<double,blDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double aggregate = BlockReduce(temp_storage).Sum(mySum);
  if(tid == 0)
     atomicAddDouble(&sif[binIdx], aggregate/ n * weights[blockIdx.x]);
}

/** Transform a complex vector by a real matrix
 *
 * \todo replace with DGEMM?
 */
static __global__ void calc_dllmm_k(cuDoubleComplex* cdij, cuDoubleComplex* cdlm, double* trans, int nion, int lmmax_wav, int lmmax_aug, int lmdim, int size_cdij, int size_cdlm, int nband)
{
  cuDoubleComplex *cdij_p, *cdlm_p;
  
  cuDoubleComplex cdije;
  
  int i, j, l, k, band;
  for (band = blockIdx.x; band < nband; band += gridDim.x)
  {
    cdij_p = cdij + band * size_cdij;
    cdlm_p = cdlm + band * size_cdlm;
    for (i = blockIdx.y; i < nion; i+= gridDim.y)
    {
      for (l = threadIdx.x; l < lmmax_wav; l+= blockDim.x)
      {
        for (k = threadIdx.y; k < lmmax_wav; k += blockDim.y)
        {
          cdije = cdij_p[l + k * lmdim + i * lmdim * lmdim];
          for (j = 0; j < lmmax_aug; j++)
          {
             cdije.x += cdlm_p[j + i * lmmax_aug].x * trans[l + k * lmdim + j * lmdim * lmdim];
             cdije.y += cdlm_p[j + i * lmmax_aug].y * trans[l + k * lmdim + j * lmdim * lmdim];
          }
          cdij_p[l + k * lmdim + i * lmdim * lmdim] = cdije;
        }//for k
      }//for l
    }//for i
  }//for band
}//func

/** Sum the product of two complex vectors */
static __global__ void overl_k(cuDoubleComplex* res, cuDoubleComplex* cdij, cuDoubleComplex *cproj, int lmdim, int nion, int lmmax, int size_proj, int size_cdij, int nband){
  cuDoubleComplex cdije, cproje, rese;
  cuDoubleComplex *res_p, *cdij_p;
  
  int i, j, k, band;
  for (band = blockIdx.x; band < nband; band += gridDim.x){
    res_p = res + band * size_proj; cdij_p = cdij + band * size_cdij;
    for (i = threadIdx.x; i < nion; i+= blockDim.x){
      for (j = 0; j < lmmax; j++){
        rese = res_p[j + i*lmmax];
        for (k = 0; k < lmmax; k++){
          cdije = cdij_p[k + j*lmdim + i*lmdim*lmdim]; cproje = cproj[k + i*lmmax]; 
          rese.x += cdije.x * cproje.x - cdije.y * cproje.y;
          rese.y += cdije.x * cproje.y + cdije.y * cproje.x;
        }
        res_p[j + i*lmmax] = rese;
      }
    }
  }
}

/** JBNV. combined eccp_nl_fock_sif_k and eccp_nl_fock_forhf_k **/
//TODO(sm): check if it's beneficial to ditch smem since it's used like a local variable
template<int blDim>
static __global__ void eccp_nl_fock_sif_k_forhf_k(cuDoubleComplex* cdij0, cuDoubleComplex* cdij1, cuDoubleComplex *proj2, cuDoubleComplex *dproj,
                                                  cuDoubleComplex *proj1, float* output, double *weight, int dir, int nion, int lmmax, 
                                                  int lmdim, int size_proj, int size_cdij)
{
  //method==1 do : eccp_nl_fock_forhf_k
  //method==0 do : eccp_nl_fock_sif_k
  const int method = (dir <= 3);
  __shared__ volatile float sdata[blDim];
  double dtmp;
  cuDoubleComplex ctmp, dproje, proj1e, cdij0e, cdij1e;  
  ctmp.x = 0; ctmp.y = 0;
  
  /* Shift pointers based on block index */
  cdij0 += size_cdij * blockIdx.x;
  cdij1 += size_cdij * blockIdx.x;
  dproj   += size_proj * blockIdx.x;
  proj1 += size_proj * blockIdx.x;
  
  int i, j, k;
  sdata[threadIdx.x] = 0;
  for (i = threadIdx.x; i < nion; i+= blockDim.x){
    dtmp = 0;
    for (j = 0; j < lmmax; j++){
      dproje = dproj[j + i * lmmax]; proj1e = proj1[j + i * lmmax];
      for (k = 0; k < lmmax; k++){
        cdij0e = cdij0[k + j * lmdim + i * lmdim * lmdim]; cdij1e = cdij1[k + j * lmdim + i * lmdim * lmdim];
        ctmp.x =  cdij0e.x * dproje.x + cdij0e.y * dproje.y;
        ctmp.y =  cdij0e.y * dproje.x - cdij0e.x * dproje.y;
        ctmp.x += cdij1e.x * proj1e.x + cdij1e.y * proj1e.y;
        ctmp.y += cdij1e.y * proj1e.x - cdij1e.x * proj1e.y;        
        dtmp += proj2[k + i * lmmax].x * ctmp.x - proj2[k + i * lmmax].y * ctmp.y;        
      }
    }
    if(method == 1)
    {
      dtmp *= weight[blockIdx.x];
      atomicAdd(output+dir-1+3*i, -dtmp);      
    }
    else
    {   
      sdata[threadIdx.x] += dtmp;
    }
  }
  
  if(method == 1) return;
  
  typedef cub::BlockReduce<double,blDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double aggregate = BlockReduce(temp_storage).Sum(sdata[threadIdx.x]);
  if ( threadIdx.x == 0 )
      atomicAdd(output+dir-3, aggregate * weight[blockIdx.x]);
}


#ifdef SINGLEFFT
/** Convert FP vector from DP to SP */
static __global__ void double_to_float_k(cuComplex *flt, cuDoubleComplex *dbl, int n){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  
  int i;
  for (i = idx; i < n; i += nthreads){
    flt[i].x = (float) (dbl[i].x);
    flt[i].y = (float) (dbl[i].y);
 }
}
#endif

#ifdef SINGLEFFT
/* Convert FP vector from SP to DP */
static __global__ void float_to_double_k(cuDoubleComplex *dbl, cuComplex *flt, int n){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  
  int i;
  for (i = idx; i < n; i += nthreads){
    dbl[i].x = (double) (flt[i].x);
    dbl[i].y = (double) (flt[i].y);
  }
}
#endif

/** Collects global context variables.
 *
 * Called once per FOCK_ACC or FOCK_FORCE.
 * Collects system-wide parameters like the number of ions, their types, and
 * information about the k-points.
 * Params associated with wavefunctions or projectors have their own handles.
 */
extern "C" void setup_context_cu_C(
                  global_desc** glob,     //!< global parameters to populate
                  double* trans,          //!< rotation matrix 
                  double* fermi_weights,  //!< array of weights by k-point and band
                  double* kpoint_weights, //!< weights of each k-point
                  int* ntype,             //!< number of species
                  int* nion,              //!< number of ions of each species
                  int* nband,             //!< number of bands
                  int* nkpoint_full,      //!< number of k-points
                  int* nkpoint_red,       //!< number of symmetry-unique k-points
                  int* equiv_in,          //!< map of symmetry relation betweek k-points
                  int* lmdim,             //!< another size
                  int* lmmax_aug,         //!< part of dimension of trans
                  double* rspin,          //!< something?
                  int* comm
                                 ){

  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

#ifdef __PARA
  mycomm_k = MPI_Comm_f2c(*comm);
  MPI_Comm_rank(mycomm_k, &myproc_k);
  MPI_Comm_size(mycomm_k, &nproc_k);
#else
  nproc_k = 1; myproc_k = 0; mycomm_k = 0;
#endif

  int i;

  /* Make the structures, if needed */
  //NOTE(sm): glob is set to NULL in every call of FOCK_ACC ad FOCK_FORCE
  //          so this get's exectuting everytime...
  if (*glob == NULL){
    *glob = (global_desc*) malloc(sizeof(global_desc));
    (*glob)->trans_d = (double_p*) create_cu();
  }

  /* Assign integer/logical values */
  (*glob)->nband = *nband * nproc_k;
  (*glob)->nkpoint_full = *nkpoint_full;
  (*glob)->nkpoint_red = *nkpoint_red;
  (*glob)->lmdim = *lmdim;
  (*glob)->lmmax_aug = *lmmax_aug;
  (*glob)->rspin = *rspin;
  
  /* For passing arrays, copy is nessesary for robustness in MPI */
  (*glob)->ntype = *ntype;
  (*glob)->nion = (int*) malloc(*ntype * sizeof(int));
  memcpy((*glob)->nion, nion, *ntype * sizeof(int));
  (*glob)->nion_tot = 0;
  for (i = 0; i < *ntype; i++)
    (*glob)->nion_tot += (*glob)->nion[i];

  (*glob)->equiv = (int*) malloc(*nkpoint_full * sizeof(int));
  memcpy((*glob)->equiv, equiv_in, *nkpoint_full * sizeof(int)); 

  (*glob)->fermi_weights_g = (double*) malloc(*nkpoint_red * *nband * nproc_k * sizeof(double));
  memcpy((*glob)->fermi_weights_g, fermi_weights, *nkpoint_red * *nband * nproc_k * sizeof(double)); 

  (*glob)->kpoint_weights = (double*) malloc(*nkpoint_red * sizeof(double));
  memcpy((*glob)->kpoint_weights, kpoint_weights, *nkpoint_red * sizeof(double)); 
  
  /* Copy data to the GPU */
  assign_cu((void_p*)(*glob)->trans_d, trans, (*ntype) * (*lmdim) * (*lmdim) * (*lmmax_aug) * sizeof(double));

}

/** Collects the wavefunctions for a set of bands at a given k-point
 *
 * Includes options to FFT the wavefunction into real-space and to redistribute
 * the local bands among each PE.  In the redistribution case, bands are inter-
 * leaved with the lowest index being myPE_.  This is consistent with existing
 * Fortran code.  Note that even in the redistribute case, fermi_weights_l only
 * holds the weights formerly local bands.  
 */
extern "C" void gather_waves_cu_C(
                  wavefunction_desc** waves, //!< structure to be populated
                  cuDoubleComplex* kwav,     //!< k-space wavefunctions
                  cuDoubleComplex* proj,     //!< projections
                  cuDoubleComplex* phase,    //!< phase shift for lshift
                  int* index,                //!< map from sphere <-> sticks
                  double* fermi_weights,     //!< fermi weights of local bands
                  int* grid,                 //!< size of inflated grid (sticks)
                  int* nkwav,                //!< size in k-space
                  int* nproj,                //!< number of projectors (size of projection)
                  int* nindex,               //!< size of k-space sphere
                  int* ntype,                //!< number of types of ions
                  int* nion,                 //!< number of ions of each type
                  int* lmmax,                //!< max lm-number
                  int* linv,                 //!< inversion flag
		  int* lshift,               //!< phase shift flag
                  int* nband,                //!< number of bands in this structure
                  int* npos,                 //!< position of first band (assumed sequential)
                  int* do_fft,               //!< do the fft?
                  int* do_redis              //!< redisribution among PEs?
                                ){
  /* Check for pesky gammareal definition */
#ifdef gammareal
  fprintf(stderr, "gather_and_fft_cu: Gamma-point not currently supported... \n");
#endif

  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

  int i;

  /* make the structures, if needed */
  if (*waves == NULL){
    *waves = (wavefunction_desc*) malloc(sizeof(wavefunction_desc));
    (*waves)->kwav_d = (cuDoubleComplex_p*) create_cu();
    (*waves)->rwav_d = (cuDoubleComplex_p*) create_cu();
    (*waves)->proj_d = (cuDoubleComplex_p*) create_cu();
    (*waves)->index_d = (int_p*) create_cu();
    (*waves)->phase_d = (cuDoubleComplex_p*) create_cu();
  }
  
  /* assign global context variables */
  (*waves)->fermi_weights_l = (double*) malloc(*nband * sizeof(double));
  memcpy((*waves)->fermi_weights_l, fermi_weights, *nband * sizeof(double));

  (*waves)->grid[0] = grid[0]; (*waves)->grid[1] = grid[1]; (*waves)->grid[2] = grid[2];
  (*waves)->nkwav = *nkwav;
  (*waves)->nrwav = grid[0] * grid[1] * grid[2];
  (*waves)->nproj = *nproj;
  (*waves)->nindex = *nindex;
  (*waves)->linv = *linv;
  (*waves)->lshift = *lshift;
  (*waves)->nband = *nband;
  (*waves)->npos = *npos;
  
  (*waves)->ntype = *ntype;
  (*waves)->nion = (int*) malloc(*ntype * sizeof(int));
  memcpy((*waves)->nion, nion, *ntype * sizeof(int));
  (*waves)->nion_tot = 0;
  for (i = 0; i < *ntype; i++)
    (*waves)->nion_tot += (*waves)->nion[i];
  (*waves)->lmmax = (int*) malloc(*ntype * sizeof(int));
  memcpy((*waves)->lmmax, lmmax, *ntype * sizeof(int));

  /* Copy data */
  assign_cu((void_p*)(*waves)->kwav_d, kwav, (*waves)->nkwav * (*waves)->nband * sizeof(cuDoubleComplex));
  assign_cu((void_p*)(*waves)->index_d, index, (*waves)->nindex * sizeof(int));
  /* vv this is frequently redundant vv */
  if (*lshift) assign_cu((void_p*)(*waves)->phase_d, phase, (*waves)->nkwav * sizeof(cuDoubleComplex));

#ifdef __PARA
  if (nproc_k == 1 || !*do_redis) // only copy proj over if not re-distributing
#endif
    assign_cu((void_p*)(*waves)->proj_d, proj, (*waves)->nproj * (*waves)->nband * sizeof(cuDoubleComplex));

  if (*do_fft){
    /* Make sure the buffers are the right size and zeroed in the appropriate portions */
    resize_cu((void_p*)(*waves)->rwav_d, (*waves)->nband * (*waves)->nrwav * sizeof(cuDoubleComplex));
    cudaMemset((*waves)->rwav_d->ptr, 0, (*waves)->nband * (*waves)->nrwav * sizeof(cuDoubleComplex));

    /* populate rwav with selected points from kwav */
    if (*lshift){
      inflate_all_shift_k<<<ablocks, threads>>>((*waves)->rwav_d->ptr,
                                          (*waves)->kwav_d->ptr, 
					  (*waves)->phase_d->ptr,
                                          (*waves)->index_d->ptr,
                                          (*waves)->nband, (*waves)->nrwav, (*waves)->nkwav, (*waves)->nindex, (*waves)->linv);
    } else {
      inflate_all_k<<<ablocks, threads>>>((*waves)->rwav_d->ptr,
                                          (*waves)->kwav_d->ptr, 
                                          (*waves)->index_d->ptr,
                                          (*waves)->nband, (*waves)->nrwav, (*waves)->nkwav, (*waves)->nindex, (*waves)->linv);
    }

    /* Do the actual FFTs */
#ifdef SINGLEFFT
    resize_cu((void_p*)ftmp_d, (*waves)->nband * (*waves)->nrwav * sizeof(cuComplex));
    double_to_float_k<<<ablocks, athreads>>>(ftmp_d->ptr, (*waves)->rwav_d->ptr, (*waves)->nband * (*waves)->nrwav);
    stream_ffts(ftmp_d->ptr, (*waves)->grid, 1, (*waves)->nband);
    float_to_double_k<<<ablocks, athreads>>>((*waves)->rwav_d->ptr, ftmp_d->ptr, (*waves)->nband * (*waves)->nrwav);
#else
    stream_ffts((*waves)->rwav_d->ptr, (*waves)->grid, 1, (*waves)->nband);
#endif
  }

  if (nproc_k == 1 || !*do_redis) return;

#ifdef __PARA
  /* Copy real-space wavefunctions back to CPU */
  cuDoubleComplex* wave_buffer = (cuDoubleComplex*) malloc(nproc_k * (*waves)->nrwav * (*waves)->nband * sizeof(cuDoubleComplex));
  cuDoubleComplex* proj_buffer = (cuDoubleComplex*) malloc(nproc_k * (*waves)->nproj * (*waves)->nband * sizeof(cuDoubleComplex));
  retrieve_cu(wave_buffer, (void_p*) (*waves)->rwav_d, (*waves)->nrwav * (*waves)->nband * sizeof(cuDoubleComplex));

  /* Re-space wavefunctions and projections */
  for (i = (*waves)->nband - 1; i >= 0; i--){
    memcpy(wave_buffer + (nproc_k * i + myproc_k) * (*waves)->nrwav, wave_buffer + i * (*waves)->nrwav, (*waves)->nrwav * sizeof(cuDoubleComplex));
    memcpy(proj_buffer + (nproc_k * i + myproc_k) * (*waves)->nproj, proj        + i * (*waves)->nproj, (*waves)->nproj * sizeof(cuDoubleComplex));
  }

  /* Call the gather */
  for (i = 0; i < (*waves)->nband * nproc_k; i++){
    MPI_CALL(MPI_Bcast(wave_buffer + i * (*waves)->nrwav, 2 * (*waves)->nrwav, MPI_DOUBLE, i%nproc_k, mycomm_k))
    MPI_CALL(MPI_Bcast(proj_buffer + i * (*waves)->nproj, 2 * (*waves)->nproj, MPI_DOUBLE, i%nproc_k, mycomm_k))
  }

  /* Re-copy to GPU */
  (*waves)->nband *= nproc_k;
  assign_cu((void_p*)(*waves)->rwav_d, wave_buffer, (*waves)->nrwav * (*waves)->nband * sizeof(cuDoubleComplex));
  assign_cu((void_p*)(*waves)->proj_d, proj_buffer, (*waves)->nproj * (*waves)->nband * sizeof(cuDoubleComplex));

  free(wave_buffer); free(proj_buffer);
#endif

  return;
}

/** Collects projector and phase vector 
 *
 * This structure is wave-function independent
 */
extern "C" void gather_projectors_cu_C(
                  projector_desc** projs, //!< structure to be populated
                  double* rproj,          //!< real-space projectors
                  cuDoubleComplex* phase,          //!< phase vector
                  int* nli,               //!< maps from r-space into projection sphere
                  int* nlimax_in,         //!< number of entries in nli per ion 
                  int *nproj_in,          //!< number of projectors
                  int *ntype,             //!< number of types of ions
                  int *nion_in,           //!< number of ions
                  int *lmmax_in,          //!< max lm-number
                  int *irmax_in,          //!< max(nlimax_in)
                  int *gather             //!< flag: sum values from each PE
                                     ){
  /* Check for pesky gammareal definition */
#ifdef gammareal
  fprintf(stderr, "gather_and_fft_cu: Gamma-point not currently supported... \n");
#endif

  int i;

  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

  /* make the structures, if needed */
  if (*projs == NULL){
    *projs = (projector_desc*) malloc(sizeof(projector_desc));
    (*projs)->rproj_d = (double_p*) create_cu();
    (*projs)->phase_d = (cuDoubleComplex_p*) create_cu();
    (*projs)->nli_d = (int_p*) create_cu();
    (*projs)->nlimax_d = (int_p*) create_cu();
    (*projs)->rproj_offsets_d = (int_p*) create_cu();
  }
  
  /* assign global context variables */
  (*projs)->nproj = *nproj_in;

  (*projs)->ntype = *ntype;
  (*projs)->nion = (int*) malloc(*ntype * sizeof(int));
  memcpy((*projs)->nion, nion_in, *ntype * sizeof(int));
  (*projs)->lmmax = (int*) malloc(*ntype * sizeof(int));
  memcpy((*projs)->lmmax, lmmax_in, *ntype * sizeof(int));
  (*projs)->nion_tot = 0;
  (*projs)->npro     = 0;
  for (i = 0; i < *ntype; i++){
    (*projs)->nion_tot += (*projs)->nion[i];
    (*projs)->npro     += (*projs)->nion[i] * (*projs)->lmmax[i];
  }

  (*projs)->irmax = *irmax_in;
  (*projs)->nlimax = (int*) malloc((*projs)->nion_tot * sizeof(int));
  memcpy((*projs)->nlimax, nlimax_in, (*projs)->nion_tot * sizeof(int));

  
  /* Gather from other PEs */
#ifdef __PARA
  double* buffer;
  if (*gather && nproc_k != 1){
    buffer = (double*) malloc((*projs)->nproj * sizeof(double));
    MPI_CALL(MPI_Allreduce(rproj, buffer, (*projs)->nproj, MPI_DOUBLE, MPI_SUM, mycomm_k))
    rproj = buffer;  
  }
#endif

  /* Copy data */
  assign_cu((void_p*)(*projs)->rproj_d, rproj, (*projs)->nproj * sizeof(double));
  assign_cu((void_p*)(*projs)->phase_d, phase, (*projs)->irmax * (*projs)->nion_tot * sizeof(cuDoubleComplex));
  assign_cu((void_p*)(*projs)->nli_d, nli, (*projs)->irmax * (*projs)->nion_tot * sizeof(int));
  assign_cu((void_p*)(*projs)->nlimax_d, nlimax_in, (*projs)->nion_tot * sizeof(int));

  int *prefixsum = (int*) malloc(((*projs)->nion_tot+1) * sizeof(int));
  prefixsum[0] = 0;
  i = 0;
  for (int type = 0; type < *ntype; ++type){
	  for (int ion = 0; ion < nion_in[type]; ++ion){
          prefixsum[i+1] = prefixsum[i] + (*projs)->nlimax[i] * (*projs)->lmmax[type];
		  ++i;
	  }
  }
  assign_cu((void_p*)(*projs)->rproj_offsets_d, prefixsum, (*projs)->nion_tot * sizeof(int));
  free(prefixsum);

#ifdef __PARA
  if (*gather && nproc_k != 1)
    free(buffer);
#endif
}

/** Collects the derivatives of the projectors */
extern "C" void gather_dproj_cu_C(
                               cuDoubleComplex* dproj, //!< important...
                               int *nproj, //!< it's see cproj
                               int *nband, //!< number of local bands
                               int *ndir
                              ){
  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

#ifdef __PARA
  if (nproc_k != 1){
    /* Elements are re-spaced in FOCK_FORCE */

    /* Call the gather */
    int i,j;
  for (j = 0; j < (*ndir); j++){
      for (i = 0; i < (*nband); i++){
        MPI_CALL(MPI_Bcast(dproj + i * (*nproj) + j*(*nband)*(*nproj), 2 * (*nproj), MPI_DOUBLE, i%nproc_k, mycomm_k))
      }
    }
  }
#endif
  assign_cu((void_p*)dproj_d, dproj, (*nproj) * (*nband) * (*ndir) * sizeof(cuDoubleComplex));
}

/** Updates the phase vector of a projector.
 *
 * Because the phase changes more frequently than the projectors
 */
extern "C" void update_phase_cu_C(
                                 projector_desc** projs, //!< struct to be updated 
                                 cuDoubleComplex* phase//!< new phase vector
                                ){
  assign_cu((void_p*)(*projs)->phase_d, phase, (*projs)->irmax * (*projs)->nion_tot * sizeof(cuDoubleComplex));
}

/** Computes the real-space representations of wavefunctions for a set of bands at a given k-point (using fft) */
extern "C" void fft_waves_cu_C(
                              wavefunction_desc** waves, //!< wavefunctions to fft
                              int* start, //!< starting band
                              int* num //!< number of bands
                             ){
  /* Check for pesky gammareal definition */
#ifdef gammareal
  fprintf(stderr, "gather_and_fft_cu: Gamma-point not currently supported... \n");
#endif

  /* Init various things */
  if (!fock_isInitialized) make_fock_context();
  
  /* Compute array index offset */
  long offset = (*start) * (*waves)->nrwav;
  
  /* Make sure the buffers are the right size and zeroed in the appropriate portions */
  resize_cu((void_p*)(*waves)->rwav_d, (*waves)->nband * (*waves)->nrwav * sizeof(cuDoubleComplex));
  cudaMemset((*waves)->rwav_d->ptr + offset, 0, (*num) * (*waves)->nrwav * sizeof(cuDoubleComplex));
  
  /* populate rwav with selected points from kwav */
  inflate_all_k<<<ablocks, threads>>>((*waves)->rwav_d->ptr + offset, (*waves)->kwav_d->ptr + (*start) * (*waves)->nkwav, (*waves)->index_d->ptr, 
                                      (*num), (*waves)->nrwav, (*waves)->nkwav, (*waves)->nindex, (*waves)->linv);

#ifdef SINGLEFFT
  resize_cu((void_p*)ftmp_d, (*num) * (*waves)->nrwav * sizeof(cuComplex));
  double_to_float_k<<<ablocks, athreads>>>(ftmp_d->ptr, (*waves)->rwav_d->ptr + offset, (*num) * (*waves)->nrwav);
  stream_ffts(ftmp_d->ptr, (*waves)->grid, 1, *num);
  float_to_double_k<<<ablocks, athreads>>>((*waves)->rwav_d->ptr + offset, ftmp_d->ptr, (*num) * (*waves)->nrwav);
#else
  stream_ffts((*waves)->rwav_d->ptr + offset, (*waves)->grid, 1, *num);
#endif

}

/** Frees everything.  Woots! */
extern "C" void free_structs_cu_C(global_desc** glob, wavefunction_desc** waves, projector_desc** projs){
  if (*glob != NULL){
//    free_cu((void_p*)(*glob)->rotmap_d);
//    free((*glob)->rotmap_d);
    free_cu((void_p*)(*glob)->trans_d);
    free((*glob)->trans_d);
    free((*glob)->fermi_weights_g);
    free((*glob)->equiv);
    free((*glob)->kpoint_weights);
    free((*glob)->nion);
    free(*glob);
    *glob = NULL;
  }

  if (*waves != NULL){
    free_cu((void_p*)(*waves)->kwav_d);
    free((*waves)->kwav_d);
    free_cu((void_p*)(*waves)->rwav_d);
    free((*waves)->rwav_d);
    free_cu((void_p*)(*waves)->proj_d);
    free((*waves)->proj_d);
    free_cu((void_p*)(*waves)->index_d);
    free((*waves)->index_d);
    free_cu((void_p*)(*waves)->phase_d);
    free((*waves)->phase_d);
    free((*waves)->fermi_weights_l);    
    free((*waves)->nion);
    free((*waves)->lmmax);
    free(*waves);
    *waves = NULL;
  }
  
  if (*projs != NULL){
    free_cu((void_p*)(*projs)->rproj_d);
    free((*projs)->rproj_d);
    free_cu((void_p*)(*projs)->phase_d);
    free((*projs)->phase_d);
    free_cu((void_p*)(*projs)->nli_d);
    free((*projs)->nli_d);
    free_cu((void_p*)(*projs)->nlimax_d);
    free((*projs)->nlimax_d);
    free_cu((void_p*)(*projs)->rproj_offsets_d);
    free((*projs)->rproj_offsets_d);
    free((*projs)->nlimax);
    free((*projs)->nion);
    free((*projs)->lmmax);
    free(*projs);
    *projs = NULL;
  }

  //clean up streams and batched Dgemm
  //NOTE(ca): we do not clean up streams and Dgemm. See comment in setup_context_cu_
#if 0
  if(ion_streams[0] != 0) {
	  for(int i = 0; i < NUM_ION_STREAM; i++)
	  {
	    cudaStreamDestroy(ion_streams[i]);
	    checkError(__LINE__,__FILE__);
	    ion_streams[i] = 0;
	  }
  }
  cublasSetStream(hcublas, NULL);
  checkError(__LINE__,__FILE__);
  for(int i = 0; i < NUM_ION_STREAM; i++) {
  cudaFree(d_ParamAarray[i]);
  checkError(__LINE__,__FILE__);
  d_ParamArray[i] = 0;
  }
#endif
}

/** This is the actual intercept routine, which does a bunch of work from fock_acc */
extern "C" void fock_acc_cu_C(
                  global_desc** glob,         //!< Global parameters
                  wavefunction_desc** waves1, //!< wavefunctions of k-point
                  wavefunction_desc** waves2, //!< wavefunctinos of q-point
                  projector_desc** proj,      //!< projectors
                  double* potfak,             //!< local potential
                  cuDoubleComplex *cxi,       //!< output wavefunctions
                  cuDoubleComplex* ckappa,    //!< output projectors
                  double* exchange,           //!< ACFDT correction
                  int* loverl,                //!< overlap? 
                  int* compute_exchange,      //!< compute exchange?
                  int* nk,                    //!< k-point number
                  int* nq                     //!< q-point number
                            ){
#ifdef gammareal
  fprintf(stderr, "fock_acc_cu: Gamma-point not currently supported... \n");
#endif
 
#ifdef PROFILE_FOCK_ACC
  cudaProfilerStart();
#endif
  
  int bq, ion, ion_global, type;
  size_t shared_size;
  double one = 1., zero = 0.;
  
  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

  int kstripe = (*waves1)->nband;  

  int c1, c2, c3;
  dim3 tGrid, tBlock;

  /* Runs through the 2nd k-point's bands one at a time */  
  resize_cu((void_p*)(*waves2)->rwav_d, (*waves2)->nrwav * sizeof(cuDoubleComplex));
  
  /* Copy over input parameters */
  assign_cu((void_p*)potfak_d, potfak, (*waves1)->nrwav * sizeof(double));
  assign_cu((void_p*)cxi_d,    cxi,    (*waves1)->nrwav * (*waves1)->nband * sizeof(cuDoubleComplex));
  assign_cu((void_p*)ckappa_d, ckappa, (*waves1)->nproj * (*waves1)->nband * sizeof(cuDoubleComplex));
  if (*compute_exchange){
    assign_cu((void_p*)sif_d, exchange, 1 * sizeof(double));
  }

  /* make sure temporary buffers are the right size temporary stuff */
  resize_cu((void_p*)charge_d, kstripe * (*waves1)->nrwav * sizeof(cuDoubleComplex));
  resize_cu((void_p*)crholm_d, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
  resize_cu((void_p*)cdlm_d,   kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
  resize_cu((void_p*)cdij_d,   kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot * sizeof(cuDoubleComplex));
  resize_cu((void_p*)ctmp_d,   (*waves2)->nproj * sizeof(cuDoubleComplex));
#ifdef SINGLEFFT
  resize_cu((void_p*)ftmp_d,   kstripe * (*waves2)->nrwav * sizeof(cuComplex));
#endif
 
  int ion_stream_id = 0;
  
  /* Needed for the character rotation
  if (!*within_wavespin){
    assign_cu((void_p*)rotation_d, rotation, (*mmax) * (*mmax) * ((*glob)->lps_max+1) * sizeof(double));
  }
  */
 
  /* Check some things */
  assert((*waves1)->nrwav == (*waves2)->nrwav);

  if ( *compute_exchange ) {
	  double *weights = (double*) malloc(kstripe * (*waves2)->nband * sizeof(double));
	  double *lweights = weights;
	  for (bq = 0; bq < (*waves2)->nband; bq++){
		  for (int bk = 0; bk < kstripe; bk++){
			  lweights[bk] = -0.5 * (*glob)->rspin * (*glob)->kpoint_weights[(*nk)-1]* (
					  MIN((*waves2)->fermi_weights_l[bq],  (*glob)->fermi_weights_g[bk + (*waves1)->npos*nproc_k + ((*nk)-1) * (*glob)->nband])
					  -     (*waves2)->fermi_weights_l[bq] * (*glob)->fermi_weights_g[bk + (*waves1)->npos*nproc_k + ((*nk)-1) * (*glob)->nband]);
		  }
		  lweights += kstripe;
	  }
      assign_cu((void_p*)weights_d, weights, kstripe * (*waves2)->nband * sizeof(double));
      free(weights);
  }

  /* Loop over the q-point's bands */
  for (bq = 0; bq < (*waves2)->nband; bq++){
    /* See if the band should be skipped */
    if (abs((*waves2)->fermi_weights_l[bq]) <= 1E-10)
      continue;
    
    /* FFT the wavefunction into real space (see fft_waves) */
    ion = 1;
    //fft_waves_cu_(waves2, &bq, &ion);
    
    /* Rotate the character if the q-point lies outside of the WHF struct \see full_kpoints::ROTATE_WAVE_CHARACTER
    if (!*within_wavespin){
      rotate_cproj_k<<<blocks,sthreads>>>(
                    ctmp_d->ptr, 
                    (*waves2)->proj_d->ptr + bq * (*waves2)->nproj,
                    rotation_d->ptr,
                    (*glob)->rotmap_d->ptr + ((*nq)-1) * (*glob)->nion_tot,
                    (*glob)->lps_d->ptr, 
                    (*glob)->lmax, (*waves2)->lmmax[0], *mmax, (*glob)->nion_tot);
    } else{
    */
      cudaMemcpy(ctmp_d->ptr, (*waves2)->proj_d->ptr + bq * (*waves2)->nproj, (*waves2)->nproj * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    //}
    
    /* Loop over the k-point bands  */
    /* Compute the contribution to the charge from the real-space wavefunctions [pw_charge_trace in hamil.F] */
    charge_trace_k<1><<<512, 256>>>(
                  (*waves2)->rwav_d->ptr + bq * (*waves2)->nrwav, 
                  (*waves1)->rwav_d->ptr, 
                  charge_d->ptr,
                  (*waves1)->nrwav, kstripe);

    /* if 'overlap required' (see type wavedes in wave.F) [usually is] */
    if (*loverl){
      /* Compute the contribution to the charge from the character [depsum_two_bands_rholm_trace in fast_aug.F] */
      cudaMemset(crholm_d->ptr, 0, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
      c1 = 0;
      c2 = 0;
      c3 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        dim3 threads(10,10,1), blocks((*glob)->nion[type],kstripe);
        aug_charge_trace_k <<<blocks, threads>>>(
                           (*waves1)->proj_d->ptr + c1, 
                           ctmp_d->ptr + c2,
                           (*glob)->trans_d->ptr + type * (*glob)->lmdim * (*glob)->lmdim * (*glob)->lmmax_aug, 
                           crholm_d->ptr + c3, 
                           (*glob)->nion[type], (*waves1)->lmmax[type], (*waves1)->nproj, (*proj)->lmmax[type], (*proj)->npro, (*glob)->lmdim, kstripe);
        c1 += (*glob)->nion[type] * (*waves1)->lmmax[type];
        c2 += (*glob)->nion[type] * (*waves1)->lmmax[type];
        c3 += (*glob)->nion[type] * (*proj)->lmmax[type];
      }

      /* Multiply crholm and the projectors, add to charge [racc0_hf in nonlr.F] */
      ion_global = 0; c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        int k = (*proj)->lmmax[type];
        int batchSize;
        batchSize = (*glob)->nion[type];
        int maxValueOfNlimax = *std::max_element((*proj)->nlimax + ion_global, (*proj)->nlimax + ion_global + batchSize);
        dim3 blocks(kstripe, batchSize, maxValueOfNlimax/512+1);
        reverse_project_all<<<blocks, 512, 0, ion_streams[ion_stream_id]>>>(
            charge_d->ptr,
            crholm_d->ptr         + c2,
            (*proj)->rproj_d->ptr,
            (*proj)->phase_d->ptr + ion_global * (*proj)->irmax,
            (*proj)->nli_d->ptr   + ion_global * (*proj)->irmax,
            (*waves1)->nrwav,
            (*proj)->irmax,
            (*proj)->lmmax[type],
            (*proj)->nlimax_d->ptr,
            (*proj)->rproj_offsets_d->ptr,
            (*proj)->npro,
            kstripe,
            batchSize,
            ion_global);
          checkError(__LINE__,__FILE__);
          ion_global += batchSize;
          ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
        c2 += (*glob)->nion[type] * k;
      }
    }
    cudaDeviceSynchronize();  
    checkError(__LINE__,__FILE__);
    cublasSetStream(hcublas, NULL);
    ion_stream_id = 0;

#ifdef SINGLEFFT
    double_to_float_k<<<ablocks, athreads>>>(ftmp_d->ptr, charge_d->ptr, kstripe * (*waves1)->nrwav);
    stream_ffts(ftmp_d->ptr, (*waves1)->grid, -1, kstripe);
    apply_gfac_k<cuComplex><<<ablocks, athreads>>>(ftmp_d->ptr, potfak_d->ptr, (*waves2)->nrwav, kstripe);
    stream_ffts(ftmp_d->ptr, (*waves1)->grid, 1, kstripe);
    float_to_double_k<<<ablocks, athreads>>>(charge_d->ptr, ftmp_d->ptr, kstripe * (*waves1)->nrwav);
#else
    /* Transform the charge into k-space */
    stream_ffts(charge_d->ptr, (*waves1)->grid, -1, kstripe);
    /* Multiply charge by potential [apply_gfac in fock.F] */
    if (*compute_exchange) {
      dim3 blockConfig(kstripe, 1,1);    //28 turned out to be optimal for our particular GPU and test. Note it should be multiple of 7
      apply_gfac_k_and_gfac_der_k<cuDoubleComplex,512><<<blockConfig, 512>>>(charge_d->ptr, potfak_d->ptr, sif_d->ptr, weights_d->ptr + bq * kstripe, (*waves2)->nrwav);
      checkError(__LINE__,__FILE__);
    }
    else {
      apply_gfac_k_localpotfak<cuDoubleComplex><<<ablocks, athreads>>>(charge_d->ptr, potfak_d->ptr, (*waves2)->nrwav, kstripe);
      checkError(__LINE__,__FILE__);
    }

    /* Transform back into real space */
    stream_ffts(charge_d->ptr, (*waves1)->grid, 1, kstripe); 
#endif
    
    /* Multiply the charge density and the real-space wavefunction to find cxi [vhamil_trace in hamil.F]*/
    (*proj)->rinpl = (*waves2)->fermi_weights_l[bq]/(*waves2)->nrwav;
    
    tBlock.x = 256; tBlock.y = 1; tBlock.z = 1;
    tGrid.x = (*waves2)->nrwav/tBlock.x+1; tGrid.y = 1; tGrid.z = 1;
    mul_vec_k<cuDoubleComplex><<<tGrid, tBlock>>>(
             cxi_d->ptr,
             charge_d->ptr,
             (*waves2)->rwav_d->ptr + bq * (*waves2)->nrwav, 
             (*proj)->rinpl, (*waves1)->nrwav, kstripe);
    
    /* if overlap again.. */
    if (*loverl){
      cudaMemset(cdlm_d->ptr, 0, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
      /* Multiply charge by projectors, again [rpro1_hf in nonlr.F] */
      ion_global = 0; c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        int m = (*proj)->lmmax[type];
        int n = 2*kstripe;
        int batchSize;

        batchSize = (*glob)->nion[type];
        int maxValueOfNlimax = *std::max_element((*proj)->nlimax + ion_global, (*proj)->nlimax + ion_global + batchSize);
#ifdef MIXED_PREC
          if (maxValueOfNlimax <= 6144 && (*proj)->lmmax[type] <= 32){
            shared_size = MAX(maxValueOfNlimax * sizeof(cuComplex), 32 * (*proj)->lmmax[type] * sizeof(cuDoubleComplex));
#else
          if (maxValueOfNlimax <= 3072 && (*proj)->lmmax[type] <= 32){
            shared_size = MAX(maxValueOfNlimax * sizeof(cuDoubleComplex), 32 * (*proj)->lmmax[type] * sizeof(cuDoubleComplex));
#endif
            dim3 threads(32, MIN(MAX(maxValueOfNlimax / 32 + 1, (*proj)->lmmax[type]),32)), blocks(kstripe, batchSize, 1);
            forward_project_all<32><<<blocks, threads, shared_size, ion_streams[ion_stream_id]>>>(
              cdlm_d->ptr           + c2,
              charge_d->ptr,
              (*proj)->rproj_d->ptr,
              (*proj)->phase_d->ptr + ion_global * (*proj)->irmax,
              (*proj)->nli_d->ptr   + ion_global * (*proj)->irmax,
              (*proj)->rinpl,
              (*waves1)->nrwav,
              (*proj)->irmax,
              (*proj)->lmmax[type],
              (*proj)->nlimax_d->ptr,
              (*proj)->rproj_offsets_d->ptr,
              (*proj)->npro,
              kstripe,
              batchSize,
              ion_global);
        checkError(__LINE__,__FILE__);
        ion_global += batchSize;
        ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
          } else {
        for (ion = 0; ion < (*glob)->nion[type]; ion += batchSize){
          int k = (*proj)->nlimax[ion_global];
          batchSize = countSameSizeDgemms(ion_global, ion_global - ion + (*glob)->nion[type], k, (*proj)->nlimax);
            resize_cu((void_p*)work2_d[ion_stream_id], batchSize *  n * k * sizeof(double));
            resize_cu((void_p*)work1_d[ion_stream_id], batchSize *  n * (*proj)->irmax * sizeof(double));

 	          dim3 threads(256); dim3 blocks(4,kstripe,1);
            crrexp_mul_wave_k_batched<<<blocks, threads, 0, ion_streams[ion_stream_id]>>>(
                            (*proj)->phase_d->ptr + ion_global * (*proj)->irmax, 
                            charge_d->ptr, 
                            work2_d[ion_stream_id]->ptr, 
                            (*proj)->nli_d->ptr + ion_global * (*proj)->irmax, 
                            (*proj)->nlimax_d->ptr + ion_global, 
                            (*proj)->irmax, (*waves1)->nrwav, kstripe, 1,
 			      batchSize, (*proj)->irmax, n*k, (*proj)->irmax, 1);
 	          checkError(__LINE__,__FILE__);
 
   		  	  batchedDgemm(
   					hcublas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 
   					&one, 
                        (*proj)->rproj_d->ptr + c1, k, 
                        work2_d[ion_stream_id]->ptr, k, 
   					&zero, 
                        work1_d[ion_stream_id]->ptr, m,
   					batchSize,
   					ion_stream_id);
 
            dim3 tblocks(1, kstripe, batchSize); dim3 tthreads(128);
            cucalccproj<0,cuDoubleComplex><<<tblocks, tthreads, 0, ion_streams[ion_stream_id]>>>(
                  m,
                  cdlm_d->ptr + c2 + ion * m,
                   work1_d[ion_stream_id]->ptr, 
                   (*proj)->rinpl,
                   m*n, 0, NULL, m, 2, (*proj)->npro);
            checkError(__LINE__,__FILE__);
          c1 += k * m * batchSize;
          ion_global += batchSize;
          ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
          }
        }
        c2 += (*glob)->nion[type] * m;
      }
      cudaDeviceSynchronize();  
      checkError(__LINE__,__FILE__);
      cublasSetStream(hcublas, NULL);
      ion_stream_id = 0;

      /* Transform result somehow [calc_dllmm_trace in fast_aug.F] */
      cudaMemset(cdij_d->ptr , 0, kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot * sizeof(cuDoubleComplex));
      c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        get_calc_dmmll_config(kstripe ,(*glob)->nion[type], (*proj)->lmmax[type], tGrid, tBlock); 
        calc_dllmm_k<<<tGrid,tBlock>>>(
                    cdij_d->ptr + c1, 
                    cdlm_d->ptr + c2, 
                    (*glob)->trans_d->ptr + type * (*glob)->lmdim * (*glob)->lmdim * (*glob)->lmmax_aug, 
                    (*glob)->nion[type], (*waves1)->lmmax[type], (*proj)->lmmax[type], (*glob)->lmdim, (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, (*proj)->npro, kstripe);
        c1 += (*glob)->nion[type] * (*glob)->lmdim * (*glob)->lmdim;
        c2 += (*glob)->nion[type] * (*proj)->lmmax[type];
      }

      /* Multiply result by the q-point's character [overl_fock in fock.F] */ 
      c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        overl_k<<<min(kstripe,ablocks.x),threads>>>(
               ckappa_d->ptr + c1, 
               cdij_d->ptr   + c2,
               ctmp_d->ptr   + c1, 
               (*glob)->lmdim, (*glob)->nion[type], (*waves2)->lmmax[type], (*waves1)->nproj, (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, kstripe);
        c1 += (*glob)->nion[type] * (*waves2)->lmmax[type];
        c2 += (*glob)->nion[type] * (*glob)->lmdim * (*glob)->lmdim;
      }
      checkError(__LINE__,__FILE__);
    }
  }

  /* Copy the relavent values back to the CPU */
  retrieve_cu(cxi,    (void_p*)cxi_d,    (*waves1)->nrwav *  (*waves1)->nband * sizeof(cuDoubleComplex));
  retrieve_cu(ckappa, (void_p*)ckappa_d, (*waves1)->nproj *  (*waves1)->nband * sizeof(cuDoubleComplex));
  if (*compute_exchange){
    retrieve_cu(exchange, (void_p*) sif_d, 1*sizeof(double));
  }

#ifdef PROFILE_FOCK_ACC
  cudaProfilerStop();
#endif


}


/** Another actual intercept routine, which does the work of FOCK_FORCE */
extern "C" void fock_force_cu_C(
                  global_desc** glob,         //!< global parameters
                  wavefunction_desc** waves1, //!< wavefunctions of k-point
                  wavefunction_desc** waves2, //!< wavefunctions of q-point
                  projector_desc** proj,      //!< projectors
                  double* potfak,             //!< array of potentials (7 of them)
                  double* sif,                //!< one output
                  double* sif2,               //!< another output
                  double* forhf,              //!< third output
                  int* loverl,                //!< flag: overlap? 
                  int* nk,                    //!< k-point number
                  int* nq                     //!< q-point number
                              ){
  /* Init various things */
  if (!fock_isInitialized) make_fock_context();

  int bq, i, ion, ion_global, type; 
  size_t shared_size;
  int kstripe = (*waves1)->nband;
  double one = 1.0, zero = 0.0; 
  
  const int nDIR = 10; //JBNV This value comes from original cuda code 

  int c1, c2, c3;
  dim3 tGrid, tBlock;

  /* Runs through the 2nd k-point's bands one at a time */  
  resize_cu((void_p*)(*waves2)->rwav_d, (*waves2)->nrwav * sizeof(cuDoubleComplex));
  
  /* Copy over input parameters */
  assign_cu((void_p*)potfak_d, potfak, 7 * (*waves2)->nrwav * sizeof(double));
  assign_cu((void_p*)sif_d, sif, 7 * sizeof(double));
  float sif2_float[7];
  for (i = 0; i < 7; i++)
    sif2_float[i] = (float) sif2[i];
  float *forhf_float = (float*) malloc(3 * (*glob)->nion_tot * sizeof(float));
  for (i = 0; i < 3 * (*glob)->nion_tot; i++)
    forhf_float[i] = forhf[i];
  assign_cu((void_p*)sif2_d, sif2_float, 7 * sizeof(float));
  assign_cu((void_p*)forhf_d, forhf_float, 3 * (*glob)->nion_tot * sizeof(float));
  
  /* make sure temporary buffers are the right size */
  resize_cu((void_p*)charge_d, kstripe * (*waves1)->nrwav * sizeof(cuDoubleComplex));
  resize_cu((void_p*)crholm_d, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
  resize_cu((void_p*)cdlm_d,   kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
  resize_cu((void_p*)cdij_d,   10 * kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot * sizeof(cuDoubleComplex));
  resize_cu((void_p*)ctmp_d,   (*waves2)->nproj * sizeof(cuDoubleComplex));
#ifdef SINGLEFFT
  resize_cu((void_p*)ftmp_d,   kstripe * (*waves2)->nrwav * sizeof(cuComplex));
#endif
 
  int ion_stream_id = 0;

  int dir;

  double *weights = (double*) malloc(kstripe * (*waves2)->nband * sizeof(double));
  double *lweights = weights;
  for (bq = 0; bq < (*waves2)->nband; bq++){
    for (int bk = 0; bk < kstripe; bk++){
      lweights[bk] = (*glob)->rspin * (*glob)->kpoint_weights[(*nk)-1] * (*waves2)->fermi_weights_l[bq] * (*glob)->fermi_weights_g[bk + (*waves1)->npos*nproc_k + ((*nk)-1) * (*glob)->nband];
    }
    lweights += kstripe;
  }
  assign_cu((void_p*)weights_d, weights, kstripe * (*waves2)->nband * sizeof(double));
  lweights = weights;
  for (bq = 0; bq < (*waves2)->nband; bq++){
    for (int bk = 0; bk < kstripe; bk++){
      lweights[bk] = (*glob)->rspin * (*glob)->kpoint_weights[(*nk)-1] * (*glob)->fermi_weights_g[bk + (*waves1)->npos*nproc_k + ((*nk)-1) * (*glob)->nband];
    }
    lweights += kstripe;
  }
  assign_cu((void_p*)weights2_d, weights, kstripe * (*waves2)->nband * sizeof(double));
  free(weights);

  /* Needed for the character rotation 
  if (!*within_wavespin){
    assign_cu((void_p*)rotation_d, rotation, (*mmax) * (*mmax) * ((*glob)->lps_max+1) * sizeof(double));
  }
  */
  
  /* Loop over the q-point's bands */
  for (bq = 0; bq < (*waves2)->nband; bq++){
    /* See if the band should be skipped */
    if (abs((*waves2)->fermi_weights_l[bq]) <= 1E-10)
      continue;
    
    /* FFT the wavefunction into real space (see fft_waves) */
    ion = 1;
    //fft_waves_cu_(waves2, &bq, &ion);
    
    /* Rotate the character if the q-point lies outside of the WHF struct 
    if (!*within_wavespin){
      rotate_cproj_k<<<blocks,sthreads>>>(
                    ctmp_d->ptr, 
                    (*waves2)->proj_d->ptr + bq * (*waves2)->nproj , 
                    rotation_d->ptr,
                    (*glob)->rotmap_d->ptr + ((*nq)-1) * (*glob)->nion_tot, 
                    (*glob)->lps_d->ptr, 
                    (*glob)->lmax, (*waves2)->lmmax[0], *mmax, (*glob)->nion_tot);
    } else{ */
      cudaMemcpy(ctmp_d->ptr, (*waves2)->proj_d->ptr + bq * (*waves2)->nproj, (*waves2)->nproj * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
   // }
    
    /* Compute the contribution to the charge from the real-space wavefunctions [pw_charge_trace in hamil.F] */
    charge_trace_k<1><<<512, 256>>>(
                  (*waves2)->rwav_d->ptr + bq * (*waves2)->nrwav, 
                  (*waves1)->rwav_d->ptr, 
                  charge_d->ptr, 
                  (*waves1)->nrwav, kstripe);

    /* if 'overlap required' (see type wavedes in wave.F) [usually is] */
    if (*loverl){
      /* Compute the contribution to the charge from the character [depsum_two_bands_rholm_trace in fast_aug.F] */
      cudaMemset(crholm_d->ptr, 0, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
      c1 = 0;
      c2 = 0;
      c3 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        dim3 threads(10,10), blocks((*glob)->nion[type],kstripe);
        aug_charge_trace_k <<<blocks, threads>>>(
                           (*waves1)->proj_d->ptr + c1,        
                           ctmp_d->ptr + c2,
                           (*glob)->trans_d->ptr + type * (*glob)->lmdim * (*glob)->lmdim * (*glob)->lmmax_aug,
                           crholm_d->ptr + c3,
                           (*glob)->nion[type], (*waves1)->lmmax[type], (*waves1)->nproj, (*proj)->lmmax[type], (*proj)->npro, (*glob)->lmdim, kstripe);
        c1 += (*glob)->nion[type] * (*waves1)->lmmax[type];
        c2 += (*glob)->nion[type] * (*waves1)->lmmax[type];
        c3 += (*glob)->nion[type] * (*proj)->lmmax[type];
      }

     /* Multiply crholm and the projectors, add to charge [racc0_hf in nonlr.F] */
     ion_global = 0; c1 = 0; c2 = 0;
     for (type = 0; type < (*glob)->ntype; type++){
        int k = (*proj)->lmmax[type];
        int batchSize;
        batchSize = (*glob)->nion[type];
        int maxValueOfNlimax = *std::max_element((*proj)->nlimax + ion_global, (*proj)->nlimax + ion_global + batchSize);
        dim3 blocks(kstripe, batchSize, maxValueOfNlimax/512+1);
        reverse_project_all<<<blocks, 512, 0, ion_streams[ion_stream_id]>>>(
            charge_d->ptr,
            crholm_d->ptr         + c2,
            (*proj)->rproj_d->ptr,
            (*proj)->phase_d->ptr + ion_global * (*proj)->irmax,
            (*proj)->nli_d->ptr   + ion_global * (*proj)->irmax,
            (*waves1)->nrwav,
            (*proj)->irmax,
            (*proj)->lmmax[type],
            (*proj)->nlimax_d->ptr,
            (*proj)->rproj_offsets_d->ptr,
            (*proj)->npro,
            kstripe,
            batchSize,
            ion_global);
        checkError(__LINE__,__FILE__);
        ion_global += batchSize;
        ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
        c2 += (*glob)->nion[type] * k;
      }//for type
    } //if (*loverl)
    cudaDeviceSynchronize();
    checkError(__LINE__,__FILE__);
    ion_stream_id = 0;

#ifdef SINGLEFFT
    double_to_float_k<<<ablocks, athreads>>>(ftmp_d->ptr, charge_d->ptr, kstripe * (*waves1)->nrwav);
    stream_ffts(ftmp_d->ptr, (*waves1)->grid, -1, kstripe);
    for (int bk = 0; bk < kstripe; bk++){
      weights[bk] = (*glob)->rspin * (*glob)->kpoint_weights[(*nk)-1] * (*waves2)->fermi_weights_l[bq] * (*glob)->fermi_weights_g[bk + (*waves1)->npos*nproc_k + ((*nk)-1) * (*glob)->nband];
    }
    assign_cu((void_p*)weights_d, weights, kstripe * sizeof(double));
    dim3 blockConfig(kstripe, 28,1);    //28 turned out to be optimal for our particular GPU and test. Note it should be multiple of 7
    apply_gfac_der_k<cuComplex, 512><<<blockConfig, 512>>>(ftmp_d->ptr, potfak_d->ptr, sif_d->ptr, weights_d->ptr, (*waves2)->nrwav);
    checkError(__LINE__,__FILE__);
    if (*loverl){
      apply_gfac_k<cuComplex><<<ablocks,athreads>>>(ftmp_d->ptr , potfak_d->ptr, (*waves2)->nrwav, kstripe);
      stream_ffts(ftmp_d->ptr, (*waves1)->grid, 1, kstripe);
      float_to_double_k<<<ablocks, athreads>>>(charge_d->ptr, ftmp_d->ptr, kstripe * (*waves1)->nrwav);
#else
    /* Transform the charge into k-space */
    stream_ffts(charge_d->ptr, (*waves1)->grid, -1, kstripe);
    /* Multiply charge by potential [apply_gfac in fock.F] */
    dim3 blockConfig(kstripe, 1,1);    //28 turned out to be optimal for our particular GPU and test. Note it should be multiple of 7
    if (*loverl){
      apply_gfac_k_and_gfac_der_k<cuDoubleComplex, 256><<<blockConfig, 256>>>(charge_d->ptr, potfak_d->ptr, sif_d->ptr, weights_d->ptr + bq * kstripe, (*waves2)->nrwav);
    }
    else{
      apply_gfac_der_k<cuDoubleComplex, 256><<<blockConfig, 256>>>(charge_d->ptr, potfak_d->ptr, sif_d->ptr, weights_d->ptr + bq * kstripe, (*waves2)->nrwav);
    }
    if (*loverl){
      /* Transform back into real space */
      stream_ffts(charge_d->ptr, (*waves1)->grid, 1, kstripe);
#endif

      (*proj)->rinpl = (*waves2)->fermi_weights_l[bq]/(*waves2)->nrwav;
      cudaMemset(cdlm_d->ptr, 0, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
      /* Multiply charge by projectors, again [rpro1_hf in nonlr.F] */
      ion_global = 0; c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        int m = (*proj)->lmmax[type];
        int n = 2*kstripe;
        int batchSize;
        batchSize = (*glob)->nion[type];
        int maxValueOfNlimax = *std::max_element((*proj)->nlimax + ion_global, (*proj)->nlimax + ion_global + batchSize);
#ifdef MIXED_PREC
          if (maxValueOfNlimax <= 6144 && (*proj)->lmmax[type] <= 32){
            shared_size = MAX(maxValueOfNlimax * sizeof(cuComplex), 32 * (*proj)->lmmax[type] * sizeof(cuDoubleComplex));
#else
          if (maxValueOfNlimax <= 3072 && (*proj)->lmmax[type] <= 32){
            shared_size = MAX(maxValueOfNlimax * sizeof(cuDoubleComplex), 32 * (*proj)->lmmax[type] * sizeof(cuDoubleComplex));
#endif
            dim3 threads(32, MIN(MAX(maxValueOfNlimax / 32 + 1, (*proj)->lmmax[type]),32)), blocks(kstripe, batchSize, 1);
            forward_project_all<32><<<blocks, threads, shared_size, ion_streams[ion_stream_id]>>>(
              cdlm_d->ptr           + c2,
              charge_d->ptr,
              (*proj)->rproj_d->ptr,
              (*proj)->phase_d->ptr + ion_global * (*proj)->irmax,
              (*proj)->nli_d->ptr   + ion_global * (*proj)->irmax,
              (*proj)->rinpl,
              (*waves1)->nrwav,
              (*proj)->irmax,
              (*proj)->lmmax[type],
              (*proj)->nlimax_d->ptr,// + ion_global,
              (*proj)->rproj_offsets_d->ptr,
              (*proj)->npro,
              kstripe,
              batchSize,
              ion_global);
            checkError(__LINE__,__FILE__);
            ion_global += batchSize;
            ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
          } else {
          for (ion = 0; ion < (*glob)->nion[type]; ion += batchSize){
            int k = (*proj)->nlimax[ion_global];
            batchSize = countSameSizeDgemms(ion_global, ion_global - ion + (*glob)->nion[type], k, (*proj)->nlimax);
            resize_cu((void_p*)work2_d[ion_stream_id], batchSize *  n * k * sizeof(double));
            resize_cu((void_p*)work1_d[ion_stream_id], batchSize *  n * (*proj)->irmax * sizeof(double));

 	          dim3 threads(256); dim3 blocks(4,kstripe,1);
            crrexp_mul_wave_k_batched<<<blocks, threads, 0, ion_streams[ion_stream_id]>>>(
                            (*proj)->phase_d->ptr + ion_global * (*proj)->irmax, 
                            charge_d->ptr, 
                            work2_d[ion_stream_id]->ptr, 
                            (*proj)->nli_d->ptr + ion_global * (*proj)->irmax, 
                            (*proj)->nlimax_d->ptr + ion_global, 
                            (*proj)->irmax, (*waves1)->nrwav, kstripe, 1,
 			      batchSize, (*proj)->irmax, n*k, (*proj)->irmax, 1);
 	          checkError(__LINE__,__FILE__);
 
   		  	  batchedDgemm(
   					hcublas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 
   					&one, 
                        (*proj)->rproj_d->ptr + c1, k, 
                        work2_d[ion_stream_id]->ptr, k, 
   					&zero, 
                        work1_d[ion_stream_id]->ptr, m,
   					batchSize,
   					ion_stream_id);
 
            dim3 tblocks(1, kstripe, batchSize); dim3 tthreads(128);
            cucalccproj<0,cuDoubleComplex><<<tblocks, tthreads, 0, ion_streams[ion_stream_id]>>>(
                  m,
                  cdlm_d->ptr + c2 + ion * m,
                   work1_d[ion_stream_id]->ptr, 
                   (*proj)->rinpl,
                   m*n, 0, NULL, m, 2, (*proj)->npro);
            checkError(__LINE__,__FILE__);
          c1 += k * m * batchSize;
          ion_global += batchSize;
          ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
          }
        }
        c2 += (*glob)->nion[type] * m;
      }
    } //not where this belongs
    cudaDeviceSynchronize(); 
    checkError(__LINE__,__FILE__);
    cublasSetStream(hcublas, NULL);
    ion_stream_id = 0;

    /* Transform result somehow [calc_dllmm_trace in fast_aug.F] */
    cudaMemset(cdij_d->ptr, 0, kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot * sizeof(cuDoubleComplex));
    c1 = 0; c2 = 0;
    for (type = 0; type < (*glob)->ntype; type++){
      get_calc_dmmll_config(kstripe ,(*glob)->nion[type], (*proj)->lmmax[type], tGrid, tBlock);
      calc_dllmm_k<<<tGrid,tBlock>>>(
                  cdij_d->ptr + c1,
                  cdlm_d->ptr + c2,
                  (*glob)->trans_d->ptr + type * (*glob)->lmdim * (*glob)->lmdim * (*glob)->lmmax_aug,
                  (*glob)->nion[type], (*waves1)->lmmax[type], (*proj)->lmmax[type], (*glob)->lmdim, (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, (*proj)->npro, kstripe);
      c1 += (*glob)->nion[type] * (*glob)->lmdim * (*glob)->lmdim;
      c2 += (*glob)->nion[type] * (*proj)->lmmax[type];
      }

    /* Loop over the derivatives, saving results in a huge cdij buffer */
    for (dir = 1; dir < nDIR; dir++){
      cudaMemset(cdlm_d->ptr, 0, kstripe * (*proj)->npro * sizeof(cuDoubleComplex));
      /* Multiply charge by projectors, again [rpro1_hf in nonlr.F] */
      ion_global = 0; c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
	int m = (proj[dir])->lmmax[type];
	int n = 2*kstripe;
	int batchSize;


        for (ion = 0; ion < (*glob)->nion[type]; ion += batchSize){
	  int k = (proj[dir])->nlimax[ion_global];
	  batchSize = countSameSizeDgemms(ion_global, ion_global - ion + (*glob)->nion[type], k, (proj[dir])->nlimax);

          // might have to resize work_d buffers we are applying different projectors 
          resize_cu((void_p*)work1_d[ion_stream_id], batchSize *  n * m * sizeof(double));
          resize_cu((void_p*)work2_d[ion_stream_id], batchSize *  n * (proj[dir])->irmax *sizeof(double));

	  dim3 threads(512); dim3 blocks(2,32);
          crrexp_mul_wave_k_batched<<<blocks, threads, 0, ion_streams[ion_stream_id]>>>(
                           (proj[0])->phase_d->ptr + ion_global * (proj[dir])->irmax,
                           charge_d->ptr,
                           work2_d[ion_stream_id]->ptr,
                           (proj[dir])->nli_d->ptr + ion_global * (proj[dir])->irmax,
                           (proj[dir])->nlimax_d->ptr + ion_global, 
                           (proj[dir])->irmax, (*waves1)->nrwav, kstripe, 1,
			   batchSize, (proj[dir])->irmax, n*k, (proj[dir])->irmax, 1);
	  checkError(__LINE__,__FILE__);

          batchedDgemm(
		hcublas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 
		&one, 
                (proj[dir])->rproj_d->ptr + c1, k, 
                work2_d[ion_stream_id]->ptr, k,
                &zero, 
                work1_d[ion_stream_id]->ptr, m,
		batchSize, 
		ion_stream_id);

          dim3 tblocks(1, kstripe, batchSize); dim3 tthreads(128);
          cucalccproj<0,cuDoubleComplex><<<tblocks, tthreads, 0, ion_streams[ion_stream_id]>>>(
                 m,
                 cdlm_d->ptr + c2 + ion * m,
                 work1_d[ion_stream_id]->ptr, 
                 (proj[0])->rinpl, // note zero index
                 m*n, 0, NULL, m, 2, (proj[dir])->npro);
          c1 += k * m * batchSize;
	  ion_global += batchSize;
	  ion_stream_id = (ion_stream_id+1) % NUM_ION_STREAM;
        }//for ion
        c2 += (*glob)->nion[type] * m;
      }//for type
      cudaDeviceSynchronize();
      checkError(__LINE__,__FILE__);
      ion_stream_id = 0;

      /* Transform result somehow [calc_dllmm_trace in fast_aug.F] */
      cudaMemset(cdij_d->ptr + dir * kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, 0, kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot * sizeof(cuDoubleComplex));
      c1 = 0; c2 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        get_calc_dmmll_config(kstripe ,(*glob)->nion[type], (proj[dir])->lmmax[type], tGrid, tBlock);
        calc_dllmm_k<<<tGrid,tBlock>>>(
                    cdij_d->ptr + dir * kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot + c1,
                    cdlm_d->ptr + c2,
                    (*glob)->trans_d->ptr + type * (*glob)->lmdim * (*glob)->lmdim * (*glob)->lmmax_aug,
                    (*glob)->nion[type], (*waves1)->lmmax[type], (proj[dir])->lmmax[type], (*glob)->lmdim, (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, (proj[dir])->npro, kstripe);
        c1 += (*glob)->nion[type] * (*glob)->lmdim * (*glob)->lmdim;
        c2 += (*glob)->nion[type] * (proj[dir])->lmmax[type];
      }//for type
     
      c1 = 0; c2 = 0; c3 = 0;
      for (type = 0; type < (*glob)->ntype; type++){
        eccp_nl_fock_sif_k_forhf_k<256><<<kstripe, 256>>>(
                        cdij_d->ptr + c1,
                        cdij_d->ptr + c1 + dir * kstripe * (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot, 
                        ctmp_d->ptr + c2, 
                        dproj_d->ptr  + c2 + (dir-1) * kstripe * (*waves2)->nproj, 
                        (*waves1)->proj_d->ptr + c2, 
                        (dir <= 3) ? forhf_d->ptr + c3 : sif2_d->ptr,  //Note the IF here
                        weights2_d->ptr + bq * kstripe, dir, (*glob)->nion[type], (*waves1)->lmmax[type], (*glob)->lmdim, (*waves1)->nproj, (*glob)->lmdim * (*glob)->lmdim * (*glob)->nion_tot);
        
  
        c1 += (*glob)->nion[type] * (*glob)->lmdim * (*glob)->lmdim;
        c2 += (*glob)->nion[type] * (*waves1)->lmmax[type];
        c3 += (*glob)->nion[type] * 3;   
      }

      checkError(__LINE__,__FILE__);
    }

  }// for bq
  
  retrieve_cu(sif, (void_p*)sif_d, 7 * sizeof(double));
  retrieve_cu(sif2_float, (void_p*)sif2_d, 7 * sizeof(float));
  for (i = 0; i < 7; i++)
    sif2[i] = (double) sif2_float[i];
  retrieve_cu(forhf_float, (void_p*)forhf_d, 3 * (*glob)->nion_tot * sizeof(float)); 
  for (i = 0; i < 3 * (*glob)->nion_tot; i++)
    forhf[i] = (double) forhf_float[i];

  free(forhf_float);

  /* Destroy some streams */
  cublasSetStream(hcublas, NULL);
  
}//fock_force_cu

/** Batch-process FFTs using streams */
#ifdef SINGLEFFT
/* Helper functions */

__device__ cufftComplex double_to_single_callback(
    void *dataIn, 
    size_t offset, 
    void *callerInfo, 
    void *sharedPointer){
    cufftComplex res; cuDoubleComplex tmp;
    tmp = ((cuDoubleComplex*) dataIn)[offset];
    res.x = (float) tmp.x; res.y = (float) tmp.y;
  return res;
}

__device__ void single_to_double_callback(
    void *dataOut, 
    size_t offset, 
    cufftComplex element, 
    void *callerInfo, 
    void *sharedPointer){
    cuDoubleComplex tmp;
    tmp.x = (double) element.x; tmp.y = (double) element.y;
    ((cuDoubleComplex*) dataOut)[offset] = tmp;
  return;
}

__device__ cufftCallbackLoadC  double_to_single_callback_ = double_to_single_callback;
__device__ cufftCallbackStoreC single_to_double_callback_ = single_to_double_callback;

static inline void stream_ffts(cuDoubleComplex* data, int* size, int dir, int num){

  cufftCallbackLoadC host_copy_load_callback;
  cufftCallbackStoreC host_copy_store_callback;

  cudaMemcpyFromSymbol(&host_copy_load_callback, 
                       double_to_single_callback_, 
                       sizeof(host_copy_load_callback));
  cudaMemcpyFromSymbol(&host_copy_store_callback, 
                       single_to_double_callback_, 
                       sizeof(host_copy_store_callback));
#else
static inline void stream_ffts(cuDoubleComplex* data, int* size, int dir, int num){
#endif

  /* Compute number of transforms per stream */
  int stride = (num-1)/NUM_FFT_STREAM +1;

  /* Check to see if the size of each transform is already right */
  CufftPlanBundle<NUM_FFT_STREAM> &stream_plan_bundle = stream_plan_cache.getBundle(size[2], size[1], size[0], num);
  if( ! stream_plan_bundle.isInitialized() )
  {
    /* Loop over the FFT streams */
    long running_total = 0;
    for(int i = 0; i < NUM_FFT_STREAM; i++)
    {
      long part_size = MIN(num-running_total, stride); // make sure we don't plan too much
      if (part_size < 0)
        part_size = 0; //still create plans because we will call cufftDestroy()

      /* Plan! and set stream */
#ifdef SINGLEFFT    
      CUDA_FFT_CALL(cufftPlanMany(&stream_plan_bundle[i], 3, stream_plan_bundle.getDims(), NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, part_size))
      CUDA_FFT_CALL(cufftXtSetCallback(stream_plan_bundle[i], (void**)&host_copy_load_callback, CUFFT_CB_LD_COMPLEX, NULL))
      CUDA_FFT_CALL(cufftXtSetCallback(stream_plan_bundle[i], (void**)&host_copy_store_callback, CUFFT_CB_ST_COMPLEX, NULL))
#else
      CUDA_FFT_CALL(cufftPlanMany(&stream_plan_bundle[i], 3, stream_plan_bundle.getDims(), NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, part_size))
#endif
      CUDA_FFT_CALL(cufftSetStream(stream_plan_bundle[i], streams[i]))

      running_total += part_size;
    }
    stream_plan_bundle.setInitialized();
  }
  
  int extent = size[0] * size[1] * size[2];
  long running_total = 0;
  /* Run the actual transforms */
  for (int i = 0; i < NUM_FFT_STREAM; i++){
    long part_size = MIN(num-running_total, stride); // make sure we don't plan too much
    if (part_size <= 0)
      break;
#ifdef SINGLEFFT
    CUDA_FFT_CALL(cufftExecC2C(stream_plan_bundle[i], (cuComplex*) (data + running_total * extent), (cuComplex*) (data + running_total * extent), dir))
#else
    CUDA_FFT_CALL(cufftExecZ2Z(stream_plan_bundle[i], data + running_total * extent, data + running_total * extent, dir))
#endif
    running_total += part_size;
  }
  cudaDeviceSynchronize();
  checkError(__LINE__,__FILE__);
}

/** Initialize all the pointers */
static inline void make_fock_context(void){
  if (rotation_d == NULL){
    rotation_d = (double_p*) create_cu();
  }
  if (potfak_d == NULL){
    potfak_d = (double_p*) create_cu();
  }
  if (dproj_d == NULL){
    dproj_d = (cuDoubleComplex_p*) create_cu();
  }  
  if (charge_d == NULL){
    charge_d = (cuDoubleComplex_p*) create_cu();
  }
  if (crholm_d == NULL){
    crholm_d = (cuDoubleComplex_p*) create_cu();
  }
  if (cdij_d == NULL){
    cdij_d = (cuDoubleComplex_p*) create_cu();
  }
  if (cdlm_d == NULL){
    cdlm_d = (cuDoubleComplex_p*) create_cu();
  }
  if (cxi_d == NULL){
    cxi_d = (cuDoubleComplex_p*) create_cu();
  }
  if (ckappa_d == NULL){
    ckappa_d = (cuDoubleComplex_p*) create_cu();
  }
  if (sif_d == NULL){
    sif_d = (double_p*) create_cu();
  } 
  if (sif2_d == NULL){
    sif2_d = (float_p*) create_cu();
  } 
  if (forhf_d == NULL){
    forhf_d = (float_p*) create_cu();
  }   
  if (ctmp_d == NULL){
    ctmp_d = (cuDoubleComplex_p*) create_cu();
  }
  if (rtmp_d == NULL){
    rtmp_d = (double_p*) create_cu();
  }
  if (itmp_d == NULL){
    itmp_d = (double_p*) create_cu();
  }
  if (work2_d[0] == NULL){
    for(int i = 0; i < NUM_ION_STREAM; i++) {
        work2_d[i] = (double_p*) create_cu();
    }
  }
  if (work1_d[0] == NULL){
    for(int i = 0; i < NUM_ION_STREAM; i++) {
        work1_d[i] = (double_p*) create_cu();
    }
  }
  if (weights_d == NULL){
    weights_d = (double_p*) create_cu();
  } 
  if (weights2_d == NULL){
    weights2_d = (double_p*) create_cu();
  }
  if (ftmp_d == NULL){
    ftmp_d = (cuComplex_p*) create_cu();
  }
  if (streams[0] == 0)
  {
    for (int i = 0; i < NUM_FFT_STREAM; i++)
    {
      cudaStreamCreate(&streams[i]);
      checkError(__LINE__,__FILE__);
    }
  }

  //NOTE(ca): we could set up the streams and d_Xarray arrays here
  //and delete them again in the free_structs_cu_ function.
  //However, set_up and free_structs_cu_ are called multiple times in a run,
  //always creating the same things; therefore we only create
  //them the very first time and keep them around indefinitely, which should be fine
  //since they are small.
  //streams and batched dgemm initializaiton
  if(ion_streams[0] == 0) 
  {
    for(int i = 0; i < NUM_ION_STREAM; i++)
    {
      cudaStreamCreate(&ion_streams[i]);
      checkError(__LINE__,__FILE__);
    }
  }
  
  if(d_ParamArray[0] == NULL)
  {
    for(int i = 0; i < NUM_ION_STREAM; i++)
    {
      cudaMalloc((void**)&d_ParamArray[i], 3*MAX_DGEMM_BATCH*sizeof(double*));
      checkError(__LINE__,__FILE__);
    }
  }


  //fprintf(stderr, "Made Context! \n");
  fock_isInitialized = true;
}
