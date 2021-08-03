// File: nonlr.cu
// C/Fortran interface to GPU port of nonlr.F.

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
#include "kernels.h"

/******************************************************/
// CUDA kernels/wrappers used in RPROMU

// TODO: cleanup
template <class NumType>
__global__ void gpu_crrexp_mul_wave_multi_Kernel(int *NArray, NumType **CR, const int shiftCR,
		double *WORK,const int *NLI, const int N1, const NumType *CRREXP,
		const int shiftW1, int step, const int nsim, const int irmax,
		const int ispiral, const int c1, const int c2, const int nib,
		const int end)
{
    register int i = blockIdx.x*blockDim.x+threadIdx.x;
    register int ip;
    register NumType ctmp;
    const int N = NArray[blockIdx.y + nib-1];

    if(i < N)
    {
	ip = (NLI+(blockIdx.y+nib-1)*N1)[i] - 1;
	NumType crrexpval = (CRREXP + (c1*c2*(ispiral-1)+(nib+blockIdx.y-1)*c1))[i];
	int shift1 = shiftW1*blockIdx.y;
	int shift2 = shift1+irmax;
	for(int j=0; j<nsim; ++j)
	{
            const NumType *currentCR = CR[j]+shiftCR;
            if(CR[j] != NULL)
            {
                ctmp = currentCR[ip]*crrexpval;
                WORK[i+shift1] = ctmp.x;
                WORK[i+shift2] = ctmp.y;
                shift1 += step;
                shift2 += step;
            }
	}
    }
}

// TODO: cleanup, remove local arrays from library
cuDoubleComplex ** CRArray_dev;
int CRArray_allocated_size = 0;
int * MArray_dev = NULL;
int MArray_dev_size = 0;

// TODO: cleanup
extern "C"
void gpu_crrexp_mul_wave_st_multi_C(const int *StreamIdx, const int *MArray,
     const int *MArray_size, const int *copy_MArray, const devptr_t *devPtrCR,
     const int *copy_cr_array, const int *shiftCR, const devptr_t *devPtrWORK,
     const int *shiftW1, const int *irmax, const int *ndata, const int *nsim,
     const devptr_t *devPtrNLI, const int *N1, const devptr_t *devPtrCRREXP,
     const int *ispiral, const int *c1, const int *c2, const int *nib,
     const int *end)
{
    int max = 0;
    for(int i = (*nib) -1; i < (*end); i++)
    {
        if(MArray[i] > max)
            max = MArray[i];
    }
    // grid dimensions
    int NbThreads=256;
    int size = (*end) - (*nib) + 1;
    dim3 threads(NbThreads);
    dim3 grid(((max)+NbThreads-1)/threads.x, size);

    // device pointers
    cuDoubleComplex **CR = (cuDoubleComplex **) devPtrCR;
    int copyM = *copy_MArray;
    if(CRArray_allocated_size < (*nsim))
    {
        if(CRArray_allocated_size > 0)
            CUDA_ERROR( cudaFree(CRArray_dev), "Failed to free device memory!" );

        CUDA_ERROR( cudaMalloc((void**)&CRArray_dev, (*nsim) * sizeof(cuDoubleComplex*)),
                    "Failed to allocate device memory!" );
        CRArray_allocated_size = *nsim;
    }
    if(MArray_dev_size < *MArray_size)
    {
        CUDA_ERROR( cudaFree(MArray_dev), "Failed to free device memory!" );
        CUDA_ERROR( cudaMalloc((void**)&MArray_dev, (*MArray_size) * sizeof(int)),
                    "Failed to allocate device memory!" );
        copyM = 1;
        MArray_dev_size = *MArray_size;
    }
    if(*copy_cr_array == 1)
        CUDA_ERROR( cudaMemcpyAsync(CRArray_dev, CR, (*nsim)*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice, 0), "Failed to copy from host to device async!" );
    if(copyM == 1)
        CUDA_ERROR( cudaMemcpyAsync(MArray_dev, MArray,(*MArray_size) * sizeof(int), cudaMemcpyHostToDevice, 0), "Failed to copy from host to device async!" );

    // device pointers
    cuDoubleComplex *CRREXP = (cuDoubleComplex *)(*devPtrCRREXP);
    double *WORK = (double *)(*devPtrWORK);
    int *NLI = (int *)(*devPtrNLI);

#if 0
    crrexp_mul_wave_k<<<grid, threads, 0, 0>>>(
      CRREXP + ((*c1)*(*c2)*(*ispiral-1)+(*nib-1)*(*c1)), 
      (cuDoubleComplex*) (devPtrCR[0]) + *shiftCR, 
      WORK, 
      NLI + (*nib-1)*(*N1), 
      MArray_dev + *nib-1, 
      *irmax, 
      (devPtrCR[1] - devPtrCR[0])/sizeof(cuDoubleComplex), 
      *nsim, 
      size);
#else
    gpu_crrexp_mul_wave_multi_Kernel<cuDoubleComplex><<<grid,threads,0,0>>>(
                            MArray_dev,
                            CRArray_dev,
                            (*shiftCR),
                            WORK,
                            NLI, 
                            *N1,
                            CRREXP,
                            *shiftW1,
                            (*ndata)*(*irmax), 
                            *nsim, 
                            *irmax, *ispiral, *c1, *c2, *nib, *end);
#endif
    CUDA_ERROR( cudaGetLastError(),
		"Failed to execute CUDA kernel gpu_crrexp_mul_wave_multi_Kernel!" );
}

extern "C"
void cuda_calccproj_C(const int *M, const int *block_size,
     const devptr_t *devptr_cproj, const int *lmbase, const int *npro_tot,
     const devptr_t *devptr_tmp, const int *shiftT1,
     const double *rinpl, const int *nsim, const int *nlm,
     const devptr_t *devptr_ldo, const int *ldo_ind_size, const int *ndata)
{
    // grid dimensions
    dim3 threads(128,1,1);  // threads.y and threads.z must be one!
    dim3 grid(((*M)+threads.x-1)/threads.x, *ldo_ind_size, *block_size);

    // device pointers
    cuDoubleComplex *cproj = (cuDoubleComplex *)(*devptr_cproj);
    double *tmp = (double *)(*devptr_tmp);
    const int *ldo = (const int *)(*devptr_ldo);

    // kernel calculates cproj
    cucalccproj<1,cuDoubleComplex><<<grid,threads>>>(*M,cproj,tmp,*rinpl,
	*shiftT1,*lmbase,ldo,*nlm,*ndata,*npro_tot);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cucalccproj!" );
}

#if 0
extern "C"
void copy_rpromu_data_(const int *C1_p, const int *C2_p, const int *C3_p, const int *N1_p,
     const int *N2_p, const int *M_p, const void *rproj, const int *size1, void **gpu_rproj,
     const void *crrexp, const int *size2, void **gpu_crrexp, const void *nli,
     const int *size3, void **gpu_nli)
{
    // device pointers
    const int C1 = *C1_p;
    const int C2 = *C2_p;
    const int C3 = *C3_p;
    const int N1 = *N1_p;
    const int N2 = *N2_p;
    const int M = *M_p;

    // copy from host to device
    cudaMemcpyAsync(*gpu_rproj,rproj,M*(*size1),cudaMemcpyHostToDevice,stream[0]);
    cudaMemcpyAsync(*gpu_crrexp,crrexp,C1*C2*C3*(*size2),cudaMemcpyHostToDevice,stream[0]);
    cudaMemcpyAsync(*gpu_nli,nli,N1*N2*(*size3),cudaMemcpyHostToDevice,stream[0]);
    // error check
    CUDA_ERROR( cudaGetLastError(), "Failed to execute copy_rpromu_data!" );
}
#endif 

/******************************************************/
// CUDA kernels/wrappers used in RACC0MU

template <int is_partial>
__global__ void cucalccracc(int nsim, int irmax, int nlimax,
                cuDoubleComplex *cracc, int ldcracc, double *work, int ldwork,
		int ndata, int *nli, int *ldo)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    // for each thread,

    if(idx<nlimax && idy<nsim)
    {
	int index,index1;
	index1=idy*ndata*ldwork+idx;

        // fetch index of CRACC to update
        if(is_partial)  index=ldo[idy]*ldcracc + nli[idx] - 1;
        else            index=idy*ldcracc + nli[idx] - 1;
        // update CRACC with atomics
        atomicAddDouble(&cracc[index].x,work[index1]);
        atomicAddDouble(&cracc[index].y,work[index1+ldwork]);
    }
}

extern "C"
void cuda_calccracc_crrexp_C(int *sid, int *nsim, int *irmax, int *nlimax,
     devptr_t *devptr_cracc, int *shiftcracc, int *ldcracc,
     devptr_t *devptr_work, int *shiftwork, int *ldwork,
     devptr_t *devptr_crrexp, int *shiftcrrexp,
     int *ndata, devptr_t *devptr_nli, int *shiftnli,
     devptr_t *devptr_ldo, int *partial)
{
    // grid dimensions
    dim3 block(256,2);
    dim3 grid(((*irmax)+block.x-1)/block.x,((*nsim)+block.y-1)/block.y);

    // device pointers
    cuDoubleComplex *cracc = (cuDoubleComplex *)(*devptr_cracc);
    cuDoubleComplex *crrexp = (cuDoubleComplex *)(*devptr_crrexp);
    double *work = (double*)(*devptr_work);
    int *nli = (int *)(*devptr_nli);
    int *ldo = (int *)(*devptr_ldo);

    if(*partial)  // not all wavefunctions operational
    {
    // kernel calculates cracc
    cucalccracc_crrexp<1><<<grid,block,0,stream[*sid]>>>(*nsim,*irmax,*nlimax,
        cracc+(*shiftcracc),*ldcracc, work+(*shiftwork),*ldwork, 0,
        crrexp+(*shiftcrrexp), *ndata, nli+(*shiftnli), ldo);
    }
    else
    {
    // kernel calculates cracc
    cucalccracc_crrexp<0><<<grid,block,0,stream[*sid]>>>(*nsim,*irmax,*nlimax,
        cracc+(*shiftcracc),*ldcracc, work+(*shiftwork),*ldwork, 0,
        crrexp+(*shiftcrrexp), *ndata, nli+(*shiftnli), ldo);
    }
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cucalccracc_crrexp!" );
}

extern "C"
void cuda_calccracc_C(int *sid, int *nsim, int *irmax, int *nlimax,
     devptr_t *devptr_cracc, int *shiftcracc, int *ldcracc,
     devptr_t *devptr_work, int *shiftwork, int *ldwork,
     int *ndata, devptr_t *devptr_nli, int *shiftnli,
     devptr_t *devptr_ldo, int *partial)
{
    // grid dimensions
    dim3 block(256,2);
    dim3 grid(((*irmax)+block.x-1)/block.x,((*nsim)+block.y-1)/block.y);

    // device pointers
    cuDoubleComplex *cracc = (cuDoubleComplex *)(*devptr_cracc);
    double *work = (double*)(*devptr_work);
    int *nli = (int *)(*devptr_nli);
    int *ldo = (int *)(*devptr_ldo);

    if(*partial)  // not all wavefunctions operational
    {
    // kernel calculates cracc
    cucalccracc<1><<<grid,block,0,stream[*sid]>>>(*nsim,*irmax,*nlimax,
        cracc+(*shiftcracc),*ldcracc, work+(*shiftwork),*ldwork, *ndata, nli+(*shiftnli), ldo);
    }
    else
    {
    // kernel calculates cracc
    cucalccracc<0><<<grid,block,0,stream[*sid]>>>(*nsim,*irmax,*nlimax,
        cracc+(*shiftcracc),*ldcracc, work+(*shiftwork),*ldwork, *ndata, nli+(*shiftnli), ldo);
    }
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cucalccracc!" );
}

extern "C"
void cuda_splitcproj_C(int *nbatch, int *nsim, int *lmmaxc,
     devptr_t *devptr_tmp, int *ldtmp, int *sizetmp,
     devptr_t *devptr_cproj, int *shiftcproj, int *ldcproj,
     double *rinpl, int *ndata, devptr_t *devptr_ldo, int *cmplx, int *partial)
{
    // grid dimensions
    dim3 block(16,4,4);
    dim3 grid(((*lmmaxc)+block.x-1)/block.x,
	 ((*nsim)+block.y-1)/block.y,((*nbatch)+block.z-1)/block.z);

    // device pointers
    cuDoubleComplex *cproj = (cuDoubleComplex*)(*devptr_cproj);
    double *tmp = (double*)(*devptr_tmp);
    int *ldo = (int*)(*devptr_ldo);

    if(*partial)  // not all wavefunctions operational
    {
        if(*cmplx)
        cusplitcproj<1,1><<<grid,block>>>(*nbatch,*nsim,*lmmaxc,
	    tmp,*ldtmp,*sizetmp,cproj+(*shiftcproj),*ldcproj,*rinpl,*ndata,ldo);
        else
        cusplitcproj<1,0><<<grid,block>>>(*nbatch,*nsim,*lmmaxc,
            tmp,*ldtmp,*sizetmp,cproj+(*shiftcproj),*ldcproj,*rinpl,*ndata,ldo);
    }
    else
    {
        if(*cmplx)
	cusplitcproj<0,1><<<grid,block>>>(*nbatch,*nsim,*lmmaxc,
            tmp,*ldtmp,*sizetmp,cproj+(*shiftcproj),*ldcproj,*rinpl,*ndata,ldo);
         else
	cusplitcproj<0,0><<<grid,block>>>(*nbatch,*nsim,*lmmaxc,
            tmp,*ldtmp,*sizetmp,cproj+(*shiftcproj),*ldcproj,*rinpl,*ndata,ldo);
    }
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cusplitcproj!" );
}

/******************************************************/
