#ifndef _KERNELS_
#define _KERNELS_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes cuda headers
#include <cuda_runtime.h>
#include "cuda_globals.h"
#include "cuda_helpers.h"


template <class NumType> __global__ void mul_vec_k(cuDoubleComplex *res, cuDoubleComplex *v1, NumType *v2, double scale, int n, int nband);


/** Add the complex elemental product of two vectors and a scale factor */
template <class NumType> __global__ void mul_vec_k(cuDoubleComplex *res, cuDoubleComplex *v1, NumType *v2, double scale, int n, int nband){
  cuDoubleComplex *res_p, *v1_p;

  int i, band;
  for (i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
    NumType v2_l = v2[i];
    for (band = blockIdx.y; band < nband; band += gridDim.y){
      res_p = res + band * n; v1_p = v1 + band * n;
      res_p[i] = res_p[i] + (v1_p[i] * v2_l) * scale;
    }
  }
}


/** Split, gather and complex element-wise product
 *
 * [OPTIMIZED] Multiplies complex vector with another gathered complex vector and splits into real/imag components
 */
static __global__ void crrexp_mul_wave_k(cuDoubleComplex *phase,
                                         cuDoubleComplex *u,
                                         double *split,
                                         int *perm,
                                         int *proj_size_ion,
                                         int max_proj_size,
                                         int wave_size,
                                         int nband, 
                                         int nion
                                        ){
  cuDoubleComplex phase_l;
  int ind2;

  for (int ion = blockIdx.z; ion < nion; ion += gridDim.z){


  for(int ind1 = threadIdx.x + blockIdx.x*blockDim.x; ind1 < proj_size_ion[ion]; ind1 += blockDim.x*gridDim.x){
    ind2     = perm [ind1]-1;
    phase_l  = phase[ind1 + ion*max_proj_size];
    for(int band = blockIdx.y; band < nband; band += gridDim.y){

      split[ind1+                    2*band*proj_size_ion[0] + 2*ion*nband*max_proj_size] = 
        phase_l.x * u[ind2 + band*wave_size].x - phase_l.y * u[ind2 + band*wave_size].y;

      split[ind1+ proj_size_ion[0] + 2*band*proj_size_ion[0] + 2*ion*nband*max_proj_size] = 
        phase_l.x * u[ind2 + band*wave_size].y + phase_l.y * u[ind2 + band*wave_size].x;

    }
  }


  }
}

static __global__ void crrexp_mul_wave_k_batched(cuDoubleComplex *phase,
                                         cuDoubleComplex *u,
                                         double *split,
                                         int *perm,
                                         int *proj_size_ion,
                                         int max_proj_size,
                                         int wave_size,
                                         int nband, 
                                         int nion,
					 int numBatches,
					 int phaseStride,
					 int splitStride,
					 int permStride,
					 int proj_size_ionStride
                                        ){
  cuDoubleComplex phase_l;
  int ind2;

  for (int batch = 0; batch < numBatches; batch++)
  {
    for (int ion = blockIdx.z; ion < nion; ion += gridDim.z){
    for(
	int ind1 = threadIdx.x + blockIdx.x*blockDim.x; 
	ind1 < proj_size_ion[ion + batch*proj_size_ionStride]; 
	ind1 += blockDim.x*gridDim.x
    ){
      ind2     = perm [ind1 + batch*permStride]-1;
      phase_l  = phase[ind1 + ion*max_proj_size + batch*phaseStride];
      for(int band = blockIdx.y; band < nband; band += gridDim.y){

        split[ind1+ 2*band*proj_size_ion[0 + batch*proj_size_ionStride] + 2*ion*nband*max_proj_size + batch*splitStride] = 
          phase_l.x * u[ind2 + band*wave_size].x - phase_l.y * u[ind2 + band*wave_size].y;

        split[ind1+ proj_size_ion[0 + batch*proj_size_ionStride] +2*band*proj_size_ion[0 + batch*proj_size_ionStride] + 2*ion*nband*max_proj_size + batch*splitStride] = 
          phase_l.x * u[ind2 + band*wave_size].y + phase_l.y * u[ind2 + band*wave_size].x;

      }
    }
    }
  }
}
/** Compute wave-function's contribution to charge density.
 *
 * Basically just multiple vector-products.  Multiplexing encourages shared-memory cacheing of second wavefunction.
 */
template <int cmplx>
__global__ void charge_trace_k(
                               cuDoubleComplex *uA, //!< test
                               cuDoubleComplex *uB, //!< test2
                               cuDoubleComplex *rhoAB, //!< test3
                               int n, //!< test4
                               int nbands //!< test5
                              ){
  const int idx      = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  cuDoubleComplex uA_l;

  if(cmplx)
  {
    for (int i = idx; i < n; i+= nthreads){
      uA_l = uA[i];
      for (int j = blockIdx.y; j < nbands; j+= gridDim.y){
        cuDoubleComplex res; // needed to combine store operations
        res.x = uB[i+j*n].x * uA_l.x + uB[i+j*n].y * uA_l.y;
        res.y = uB[i+j*n].y * uA_l.x - uB[i+j*n].x * uA_l.y;
        rhoAB[i+j*n] = res;
      }
    }
  }
  else
  {
    double* rhoAB_real = (double*) rhoAB;
    for (int i = idx; i < n; i+= nthreads){
      uA_l = uA[i];
      for (int j = blockIdx.y; j < nbands; j+= gridDim.y){
        rhoAB_real[i+j*n] = uB[i+j*n].x * uA_l.x + uB[i+j*n].y * uA_l.y;
      }
    }  
  }
}

template <int partial, int cmplx>
__global__ void cusplitcproj(int nbatch, int nsim, int lmmaxc,
		double *tmp, int ldtmp, int sizetmp, cuDoubleComplex *cproj, int ldcproj,
		double rinpl, int ndata, int *ldo)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idz = blockIdx.z*blockDim.z+threadIdx.z;

    // for each thread,

    if(idx<lmmaxc && idy<nsim && idz<nbatch)
    {
	int index, index1;
	// calculate indices
	if(partial) index1=idz*lmmaxc+ldo[idy]*ldcproj+idx;
	else	    index1=idz*lmmaxc+idy*ldcproj+idx;

	if(cmplx)
	{
	    // fetch from global memory and store in tmp
	    index=idz*sizetmp+idy*ndata*ldtmp+idx;
	    tmp[index]=cproj[index1].x*rinpl;
	    tmp[index+ldtmp]=cproj[index1].y*rinpl;
	}
	else
	{
	    // fetch from global memory and store in tmp
	    index=idz*sizetmp+idy*ldtmp+idx;
	    tmp[index]=cproj[index1].x*rinpl;
	}
    }
}

__forceinline__ __device__ double atomicAddDouble(double* address, double val)
{
#if __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull=(unsigned long long int*)address;
    unsigned long long int old=*address_as_ull,assumed;
    do {
        assumed=old;
        old=atomicCAS(address_as_ull,assumed,__double_as_longlong(val+__longlong_as_double(assumed)));
    } while(assumed!=old);
    return __longlong_as_double(old);
#else
//starting with sm_60 and up, cuda defines an atomicAdd(double*, double) intrinsic
    return atomicAdd(address, val);
#endif
}

template <int is_partial>
__global__ void cucalccracc_crrexp(int nsim, int irmax, int nlimax,
		cuDoubleComplex *cracc, int ldcracc, double *work, int ldwork, int shiftWork,
		cuDoubleComplex *crrexp, int ndata, int *nli, int *ldo)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    // for each thread,

    if(idx<nlimax && idy<nsim)
    {
        int index=blockIdx.z * shiftWork + idy*ndata*ldwork+idx;
	int shift = blockIdx.z * irmax;
        // fetch from glboal memory
        cuDoubleComplex a=make_cuDoubleComplex(work[index],work[index+ldwork]);
        cuDoubleComplex b=make_cuDoubleComplex(crrexp[idx + shift].x,-crrexp[idx + shift].y);

        // fetch index of CRACC to update
	if(is_partial)	index=ldo[idy]*ldcracc + nli[idx + shift] - 1;
        else		index=idy*ldcracc + nli[idx + shift] - 1;
        // update CRACC with atomics
        atomicAddDouble(&cracc[index].x,a.x*b.x-a.y*b.y);
        atomicAddDouble(&cracc[index].y,a.y*b.x+a.x*b.y);
    }
}

// idx, loop over innermost dimension LMMAXC
// blockIdx.y, loop over block of wavefunctions NSIM
// blockIdx.z, loop over block of ions STREAM_BLOCK_SIZE
template <int is_partial, class NumType>
__global__ void cucalccproj(const int N, NumType *cproj, 
                const double *tmp, const double rinpl,
                const int shiftT1, const int lmbase,
                const int *ldo, const int nlm,
                const int ndata, const int npro_tot)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // for each thread,

    if(idx < N)
    {
	NumType sum;
	// calculate shifts
	int shift1 = blockIdx.y*ndata*nlm + blockIdx.z*shiftT1;
        int shiftcproj;
	if (is_partial) shiftcproj = lmbase + ldo[blockIdx.y]*npro_tot + blockIdx.z*N;
	else            shiftcproj = lmbase + blockIdx.y*npro_tot + blockIdx.z*N;

	// fetch from global memory
	sum.x = tmp[idx+shift1];
	sum.y = tmp[idx+shift1+nlm];

	// calcualte cproj
	cproj[idx+shiftcproj] = sum * rinpl;
    }
}

#endif
