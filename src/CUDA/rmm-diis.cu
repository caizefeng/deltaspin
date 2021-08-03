// File: rmm-diis.cu
// C/Fortran interface to GPU port of rmm-diis.F.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes c++ headers
#include <vector>  // for local contribution
// includes cuda headers
#include <cuda_runtime.h>
#include "cublas.h"  // for local_contribution
// includes project headers
#include "cuda_globals.h"
#include "Operator.h"
#include "cuda_helpers.h"

// TODO: Re-write these routines by a) following blocking
// methodology in other .cu files and b) using reduction
// kernels from cuda_kernels.cu.

/******************************************************/
// CUDA kernels/wrappers for ECCP-related functions
// used in EDDRMM

struct localContributionParams {
   const  cuDoubleComplex *W1_CR;
    const  cuDoubleComplex *W2_CR;
    const  double *W3_CR;
    const  cuDoubleComplex *SVz;
    const  double *SVd;
    cuDoubleComplex *result;
    cuDoubleComplex *resultM;
    int    n;
    int    is_real;
    int    ISPINOR;
    int    ISPINOR_;
    int    CSTE;
    int    texXOfs;
    int    texYOfs;
    int    normalizeValue;
};

#define LOCAL_CONTRIBV2_THREADS 512
texture<uint2> texX;
texture<uint2> texY;

template<bool, bool>
__global__ void local_contribution_gld_main (struct localContributionParams parms);
template<bool, bool>
__global__ void local_contribution_tex_main (struct localContributionParams parms);
__global__ void local_contribution_main_v2 (struct localContributionParams *parms, cuDoubleComplex *results);

cuDoubleComplex *tx = NULL;
cuDoubleComplex *devPtrT = NULL;
int tx_size = 0;
int devPtrT_size = 0;
struct localContributionParams *paramPtr = NULL;
int paramPtr_size = 0;

//DO M=1,GRID%RL%NP
//   MM =M+ISPINOR *GRID%MPLWV
//   MM_=M+ISPINOR_*GRID%MPLWV
//   CLOCAL=CLOCAL+SV(M,1+ISPINOR_+2*ISPINOR)*W1%CR(MM_)*CONJG(W2%CR(MM))
//ENDDO

extern "C"
__host__ void get_local_contribution_result_C(int *size, cuDoubleComplex *RESULT)
{
    int n = *size;
    int nbrCtas = 0;
    cuDoubleComplex dot = make_cuDoubleComplex(0.0f, 0.0f);
    cublasStatus status;

    if(n < ZDOT_CTAS)
	nbrCtas = n;
    else
	nbrCtas = ZDOT_CTAS;

    /* early out if nothing to do */
    if (n <= 0)
	return;

    /* Sum the results from each CTA */
    status = cublasGetVector (nbrCtas, sizeof(tx[0]), devPtrT, 1, tx, 1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        cublasFree (devPtrT);
        free (tx);
        printf ("CUDA/CUBLAS_STATUS_SUCCESS ERROR IN %s:%d\n",__FILE__,__LINE__);
        exit(-1);
        return;
    }

    for(int i = 0; i < nbrCtas; i++)
        dot = cuCadd(dot,tx[i]);

    *RESULT = dot;
}


std::vector<struct localContributionParams> paramsV2;

/*
This function stores all the calls/parameters such that we can launch
them in one single batch when ready using local_contributionv2_launch_
*/

extern "C"
__host__ void local_contributionv2_setup_C
         (int *size, const devptr_t *W1_CR_t, const devptr_t *W2_CR_t,
          const devptr_t *SV_t, int *shiftSV, int *CSTE,
          int *ISPINOR, int *ISPINOR_, int * SV_is_real,
          int *IDX_p)
{
    int n = *size, incx = 1, incy = 1;
    int sizeX = n * (imax (1, abs(incx)));
    int sizeY = n * (imax (1, abs(incy)));
    int IDX  = *IDX_p;
    size_t texXOfs = 0;
    size_t texYOfs = 0;
    cuDoubleComplex *W1_CR = (cuDoubleComplex *)(devptr_t)(*W1_CR_t);
    cuDoubleComplex *W2_CR = (cuDoubleComplex *)(devptr_t)(*W2_CR_t);
    double          *SV    = (double *)         (devptr_t)(*SV_t);

    /* early out if nothing to do */
    if(n <= 0)
        return;
    struct localContributionParams param;

    memset (&param, 0, sizeof(param));
    param.n        = n;
    param.W1_CR    = W1_CR;
    param.W2_CR    = W2_CR;
    param.is_real  = *SV_is_real;
    if(*SV_is_real)
       param.SVd   = SV + (*shiftSV);
    else
       param.SVz   = (cuDoubleComplex *)(SV) + (*shiftSV);
    param.CSTE     = *CSTE;
    param.ISPINOR  = *ISPINOR;
    param.ISPINOR_ = *ISPINOR_;
    param.normalizeValue   =  IDX; //reuse this parameter
    param.texXOfs  = (int)texXOfs;
    param.texYOfs  = (int)texYOfs;

    paramsV2.push_back(param);

    return;
}

//Copy the configuration arguments to the GPU
extern "C"
__host__ void CUBLASAPI local_contributionv2_arm_C(int *NSIM_p, cuDoubleComplex **GPU_BUFFER_p)
{
    int nbrCtas = paramsV2.size();
    cublasStatus status = CUBLAS_STATUS_SUCCESS;
    if(paramPtr_size < nbrCtas)
    {
	cudaFree(paramPtr);
        status = cublasAlloc(nbrCtas, sizeof(localContributionParams), (void**)&paramPtr);
        paramPtr_size = nbrCtas;
    }
    if(status != CUBLAS_STATUS_SUCCESS)
    {
	printf ("CUDA/CUBLAS_STATUS_SUCCESS ERROR IN %s:%d\n",__FILE__,__LINE__);
        exit(-1);
    }

   cudaMemcpy(paramPtr, &paramsV2[0], paramsV2.size()*sizeof(paramsV2[0]), cudaMemcpyHostToDevice);
}

extern "C"
__host__ void local_contributionv2_launch_C(int *NSIM_p, cuDoubleComplex ** GPU_BUFFER_p)
{
    int threadsPerCta = LOCAL_CONTRIBV2_THREADS;
    int nbrCtas = paramsV2.size();
    cuDoubleComplex * GPU_BUFFER = *GPU_BUFFER_p;
    cudaError_t cudaStat;
    cublasStatus status;

    /* allocate memory to collect results, one per CTA */
    /* allocate small buffer to retrieve the per-CTA results */
    status = CUBLAS_STATUS_SUCCESS;
    if(devPtrT_size < nbrCtas)
    {
        cudaFree(devPtrT);
        status = cublasAlloc (nbrCtas, sizeof(tx[0]), (void**)&devPtrT);
        devPtrT_size = nbrCtas;
    }
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf ("CUDA/CUBLAS_STATUS_SUCCESS ERROR IN %s:%d\n",__FILE__,__LINE__);
        exit(-1);
        return;
    }
    if (!tx)
    {
        cublasFree (devPtrT);
        printf ("CUDA/CUBLAS_STATUS_SUCCESS ERROR IN %s:%d\n",__FILE__,__LINE__);
        exit(-1);
        return;
    }
    if(paramPtr_size < nbrCtas)
    {
        cudaFree(paramPtr);
        status = cublasAlloc(nbrCtas, sizeof(localContributionParams), (void**)&paramPtr);
        paramPtr_size = nbrCtas;
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUDA/CUBLAS_STATUS_SUCCESS ERROR IN %s:%d\n",__FILE__,__LINE__);
        exit(-1);
    }

    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    dim3 blockConfig2(numSMs*2, nbrCtas,1);

    cudaStat = cudaGetLastError(); /* clear error status */
    cudaMemset(GPU_BUFFER, 0, sizeof(cuDoubleComplex)*(*NSIM_p)*2);
    if(nbrCtas > 0)  local_contribution_main_v2<<<blockConfig2,threadsPerCta, 0, stream[0]>>>(paramPtr, GPU_BUFFER);
    paramsV2.clear();

    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasFree (devPtrT);
        free (tx);
        printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
        exit(-1);
        return;
    }

}

/* 

Sept 19. 2013

If assign == true, then we assign other wise we add to previous value
if normalize > 0 we first normalize currently stored value and then
add the new value to the just normalized value
*/

extern "C"
__host__ void local_contribution_C(int *size, const devptr_t *W1_CR_t,
         const devptr_t *W2_CR_t, const devptr_t *SV_t, int *shiftSV, int *CSTE, int *ISPINOR,
         int *ISPINOR_, int * SV_is_real, cuDoubleComplex * CLOCAL, int * assign, int * normalize)
{
    int n = *size, incx = 1, incy = 1;
    struct localContributionParams params;
    cudaError_t cudaStat;
    cublasStatus status;
    int nbrCtas;
    int threadsPerCta;
    int sizeX = n * (imax (1, abs(incx)));
    int sizeY = n * (imax (1, abs(incy)));
    size_t texXOfs = 0;
    size_t texYOfs = 0;
    int useTexture;
    cuDoubleComplex *W1_CR = (cuDoubleComplex *)(devptr_t)(*W1_CR_t);
    cuDoubleComplex *W2_CR = (cuDoubleComplex *)(devptr_t)(*W2_CR_t);
    double          *SV    = (double *)         (devptr_t)(*SV_t);

    if (n < ZDOT_CTAS) {
         nbrCtas = n;
         threadsPerCta = ZDOT_THREAD_COUNT;
    } else {
         nbrCtas = ZDOT_CTAS;
         threadsPerCta = ZDOT_THREAD_COUNT;
    }

    /* early out if nothing to do */
    if (n <= 0) {
        return;
    }
    useTexture = ((sizeX < MAX_1DBUF_SIZE) &&
                  (sizeY < MAX_1DBUF_SIZE));
    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 50000) || /* experimental bound */
        ((sizeX == n) && (sizeY == n) &&
         (!(((devptr_t) W1_CR) % WORD_ALIGN)) &&
         (!(((devptr_t) W2_CR) % WORD_ALIGN)) &&
         (!(((devptr_t) SV) % WORD_ALIGN)))) {
        useTexture = 0;
    }
    //TODO double precision texture
    useTexture = 0;
    if (useTexture) {
        if ((cudaStat=cudaBindTexture (&texXOfs,texX,W1_CR,sizeX*sizeof(W1_CR[0]))) !=
            cudaSuccess) {
            //cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            printf ("CUBLAS_STATUS_MAPPING_ERROR IN LINE %d\n",__LINE__);
	    exit(-1);
        }
        if ((cudaStat=cudaBindTexture (&texYOfs,texY,W2_CR,sizeY*sizeof(W2_CR[0]))) !=
            cudaSuccess) {
            cudaUnbindTexture (texX);
            //cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            printf ("CUBLAS_STATUS_MAPPING_ERROR IN LINE %d\n",__LINE__);
	    exit(-1);
        }
        texXOfs /= sizeof(W1_CR[0]);
        texYOfs /= sizeof(W2_CR[0]);
    }
    /* allocate memory to collect results, one per CTA */
    //printf("nbrCtas = %d\n",nbrCtas);
    /* allocate small buffer to retrieve the per-CTA results */
    status = CUBLAS_STATUS_SUCCESS;
    if(tx_size < nbrCtas)
    {
        free(tx);
        tx = (cuDoubleComplex *) calloc (nbrCtas, sizeof(tx[0]));
        tx_size = nbrCtas;
    }
    if(devPtrT_size < nbrCtas)
    {
        cudaFree(devPtrT);
        status = cublasAlloc (nbrCtas, sizeof(tx[0]), (void**)&devPtrT);
        devPtrT_size = nbrCtas;
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
        exit(-1);
    }
    if (!tx) {
        cublasFree (devPtrT);
        printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
        exit(-1);
    }
    memset (&params, 0, sizeof(params));
    params.n        = n;
    params.W1_CR    = W1_CR;
    params.W2_CR    = W2_CR;
    params.is_real  = *SV_is_real;
    if (*SV_is_real)
       params.SVd   = SV + (*shiftSV);
    else
       params.SVz   = (cuDoubleComplex *)(SV) + (*shiftSV);
    params.CSTE     = *CSTE;
    params.ISPINOR  = *ISPINOR;
    params.ISPINOR_ = *ISPINOR_;
    params.result   = devPtrT;
    params.texXOfs  = (int)texXOfs;
    params.texYOfs  = (int)texYOfs;
    params.normalizeValue = 1;
    cudaStat = cudaGetLastError(); /* clear error status */

    if (useTexture) {
        if(*assign){
                local_contribution_tex_main<true, false><<<nbrCtas,threadsPerCta>>>(params);
        } else{
                if(*normalize > 0) {
                        params.normalizeValue = *normalize;
                        local_contribution_tex_main<false, true><<<nbrCtas,threadsPerCta>>>(params);       
                } else{
                        local_contribution_tex_main<false, false><<<nbrCtas,threadsPerCta>>>(params);            }
        }
    } else {
        if(*assign){
                local_contribution_gld_main<true, false><<<nbrCtas,threadsPerCta>>>(params);
        } else{
                if(*normalize > 0){
                        params.normalizeValue = *normalize;
                        local_contribution_gld_main<false, true><<<nbrCtas,threadsPerCta>>>(params);       
                } else{
                        local_contribution_gld_main<false, false><<<nbrCtas,threadsPerCta>>>(params);            }
        }
    }


    cudaStat = cudaGetLastError(); /* check for launch error */

    if (cudaStat != cudaSuccess) {
        cublasFree (devPtrT);
        free (tx);
        printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
        exit(-1);
    }


    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
            exit(-1);
        }
        if ((cudaStat = cudaUnbindTexture (texY)) != cudaSuccess) {
            printf ("CUDA ERROR IN LINE %d  Error: %d %s\n", __LINE__, cudaStat, cudaGetErrorString(cudaStat));
            exit(-1);
        }
    }

}

__shared__ cuDoubleComplex partialSum[ZDOT_THREAD_COUNT];

template <bool assign, bool normalize>
__global__ void local_contribution_gld_main (struct localContributionParams parms)
{
#undef  USE_TEX
#define USE_TEX 0
#include "local_contribution.h"
}

template <bool assign, bool normalize>
__global__ void local_contribution_tex_main (struct localContributionParams parms)
{
#undef  USE_TEX
#define USE_TEX 1
#include "local_contribution.h"
}



//Double precision atomic add

__device__ double atomicAdd_2(double* address, double val)
{
        unsigned long long int* address_as_ull=(unsigned long long int*)address;
        unsigned long long int old=*address_as_ull,assumed;
        do {
                assumed=old;
                old=atomicCAS(address_as_ull,assumed,__double_as_longlong(val+__longlong_as_double(assumed)));
        } while(assumed!=old);
        return __longlong_as_double(old);
}

//Only used for the batched local contribution launched from local_contributionv2_launch_
__global__ void local_contribution_main_v2(struct localContributionParams *parmsArray,
                                           cuDoubleComplex *results)
{

    unsigned int i, n, tid, totalThreads, ctaStart, MM, MM_, ISPINOR, ISPINOR_, CSTE;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0f, 0.0f);
    const cuDoubleComplex *W1_CR;
    const cuDoubleComplex *W2_CR;
    const double *SV;
    const cuDoubleComplex *SVz;

    __shared__ cuDoubleComplex partialSumV2[LOCAL_CONTRIBV2_THREADS];

    struct localContributionParams parms = parmsArray[blockIdx.y];

    /* wrapper must ensure that parms.n > 0 */
    tid      = threadIdx.x;
    n        = parms.n;
    ISPINOR  = parms.ISPINOR;
    ISPINOR_ = parms.ISPINOR_;
    CSTE     = parms.CSTE;
    W1_CR = parms.W1_CR;
    W2_CR = parms.W2_CR;
    if(parms.is_real)
        SV = parms.SVd;
    else
        SVz = parms.SVz;

    totalThreads = blockDim.x*gridDim.x;
    ctaStart = blockIdx.x*blockDim.x + tid;

//DO M=1,GRID%RL%NP
//   MM =M+ISPINOR *GRID%MPLWV
//   MM_=M+ISPINOR_*GRID%MPLWV
//   CLOCAL=CLOCAL+SV(M,1+ISPINOR_+2*ISPINOR)*W1%CR(MM_)*CONJG(W2%CR(MM))
//ENDDO

    for (i = ctaStart; i < n; i += totalThreads) {
        MM  = i + ISPINOR  * CSTE;
        MM_ = i + ISPINOR_ * CSTE;
        //sum = sum + fetchz(i) * fetchx(MM_) * cuConj(fetchy(MM));
        if(parms.is_real)
            sum = sum + SV[i] * W1_CR[MM_] * cuConj(W2_CR[MM]);
        else
            sum = sum + SVz[i] * W1_CR[MM_] * cuConj(W2_CR[MM]);
    }

    partialSumV2[tid] = sum;
#if (LOCAL_CONTRIBV2_THREADS & (LOCAL_CONTRIBV2_THREADS- 1))
#error code requires LOCAL_CONTRIBV2_THREADS to be a power of 2
#endif
#pragma unroll
    for (i = LOCAL_CONTRIBV2_THREADS >> 1; i > 0; i >>= 1) {
        __syncthreads();
        if (tid < i) {
            partialSumV2[tid] = partialSumV2[tid] + partialSumV2[tid + i];
        }
    }
    if (tid == 0) {
        //Atomic add, make sure memory is 0'd before we run kernel
        atomicAdd_2(&results[parms.normalizeValue].x, partialSumV2[tid].x);
        atomicAdd_2(&results[parms.normalizeValue].y, partialSumV2[tid].y);
    }
}

/******************************************************/
// CUDA kernels/wrappers for BLAS routines used in EDDRMM

double *reduction_scratch = NULL;
int reduction_scratch_size = 0;
#define NTHREADS 128

template<class NumType>
__device__ __forceinline__ NumType gpu_add(NumType x, NumType y);
template<class NumType>
__device__ __forceinline__ NumType gpu_mul(NumType x, NumType y);

template <>
__device__ __forceinline__ double gpu_add(double x, double y)
{
    return x+y;
}

template<>
__device__ __forceinline__ cuDoubleComplex gpu_add(cuDoubleComplex x, cuDoubleComplex y)
{
    return cuCadd(x,y);
}

template<>
__device__ __forceinline__ double gpu_mul(double x, double y)
{
    return x*y;
}

template<>
__device__ __forceinline__ cuDoubleComplex gpu_mul(cuDoubleComplex x, cuDoubleComplex y)
{
    return cuCmul(cuConj(x),y);
}

template <class NumType, bool copy>
__global__ void gpu_dot_Kernel3_2d_first(NumType *scratch, const int nArguments,
		unsigned long *arguments_GPU, NumType zero, NumType * ret)
{
    //Use blockIdx.y to get the right pointers

    const int N = (int)arguments_GPU[blockIdx.y*4+0];
    NumType *w1 = (NumType*)arguments_GPU[blockIdx.y*4+1];
    NumType *w2 = (NumType*)arguments_GPU[blockIdx.y*4+2];

    scratch = &scratch[blockIdx.y*gridDim.x]; //Get our private piece of memory

    __shared__ NumType temp[NTHREADS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    NumType my_result = zero;

    for(int i = tid; i < N; i += blockDim.x * gridDim.x)
    {
        my_result = gpu_add<NumType>(my_result, gpu_mul<NumType>(w1[i],w2[i]));
    }
    temp[threadIdx.x] = my_result;
    __syncthreads();
#pragma unroll
    for(int s = NTHREADS/2; s >= 1; s = s >> 1)
    {
        if(threadIdx.x < s)
        {
            temp[threadIdx.x] = gpu_add<NumType>(temp[threadIdx.x],temp[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        scratch[blockIdx.x] = temp[0];
    }

}

template<class NumType>
__global__ void gpu_dot_Kernel3_2d_second(NumType *scratch, const int size, 
		const int nArguments, unsigned long *arguments_GPU, NumType zero,NumType * ret)
{
    __shared__ NumType temp[NTHREADS];

    scratch = &scratch[blockIdx.x*size]; //Get our private piece of memory

    const int storeIdx     = (int)arguments_GPU     [blockIdx.x*4+3];

    if(threadIdx.x < size)
    {
        temp[threadIdx.x] = scratch[threadIdx.x];
    }
    else
    {
        temp[threadIdx.x] = zero;
    }
    __syncthreads();
#pragma unroll
    for(int s = NTHREADS/2; s >= 1; s = s >> 1)
    {
        if(threadIdx.x < s)
        {
            temp[threadIdx.x] = gpu_add<NumType>(temp[threadIdx.x],temp[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        ret[storeIdx] = temp[0];
    }
}

//Called from rmm-diis.F, compute the complex dot-product and leave the result on the GPU
//Requires two kernels, one for initial reduction and a final one over the seperate thread-blocks
extern "C"
void gpu_zdot_nocopy_2d_C(int *nArguments_p, void ** argumentsCPU_p, void ** argumentsGPU_p,
     cuDoubleComplex ** ret, const int * StreamIdx)
{
    const int nArguments = *nArguments_p;
    unsigned long int *argumentsCPU = (unsigned long int*)argumentsCPU_p;

    if(nArguments == 0) return;

    int maxItems = 0;

    for(int idx =0; idx < nArguments; idx++)
    {
        int temp = (int)argumentsCPU[idx*4+0];
        if(temp > maxItems) maxItems = temp;
    }

    const int N = maxItems;
    int nThreads = NTHREADS;
    const int nBlocks = min((max(N/16,1)+nThreads-1)/nThreads, nThreads);
    if(reduction_scratch_size < nArguments*nBlocks*2)
    {
        cudaFree(reduction_scratch);
        cudaMalloc((void**)&reduction_scratch, nArguments*nBlocks*2*sizeof(double));
        reduction_scratch_size = 2*nBlocks*nArguments;
    }

    dim3 blockConfig(nBlocks, nArguments, 1);

    gpu_dot_Kernel3_2d_first<cuDoubleComplex, false><<<blockConfig,nThreads, 0, 
	stream[*StreamIdx]>>>((cuDoubleComplex *)reduction_scratch, nArguments, 
	(unsigned long int*) *argumentsGPU_p, make_cuDoubleComplex(0,0), *ret);

    gpu_dot_Kernel3_2d_second<cuDoubleComplex><<<nArguments, nThreads, 0, stream[*StreamIdx]>>>
    	((cuDoubleComplex *)reduction_scratch, nBlocks, nArguments, 
	(unsigned long int*) *argumentsGPU_p, make_cuDoubleComplex(0,0), *ret);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute gpu_zdot_nocopy_2d!" );
}

__global__ void gpu_daxpy_kernel2_2d(const int rIdx, const double *alpha, 
		unsigned long int* arguments_GPU)
{
    const uint tid = threadIdx.x;

    const int increase = blockDim.x*gridDim.x;
    const int start = blockIdx.x*blockDim.x + tid;

    const int N = (int)arguments_GPU[blockIdx.y*6+0];
    const double *X = (double*)arguments_GPU[blockIdx.y*6+1];
    double *Y = (double*)arguments_GPU[blockIdx.y*6+2];

    const int N2 = (int)arguments_GPU[blockIdx.y*6+3];
    const double *X2 = (double*)arguments_GPU[blockIdx.y*6+4];
    double *Y2 = (double*)arguments_GPU[blockIdx.y*6+5];

    //We use *4 here since the original data are in cuDoubleComplex, however we only
    //use the real part and therefor can use normal multiplication. So this only extract
    //the real part, but could be used to also use the imaginary part
    const double  a1 = alpha[blockIdx.y*4+0];
    const double  a2 = alpha[blockIdx.y*4+2];

    for(int i=start; i < N; i += increase)
        Y[i] = a1*X[i]+Y[i];

  for(int i=start; i < N2; i += increase)
      Y2[i] = a2*X2[i]+Y2[i];
}

extern "C"
void gpu_daxpy2_2d_C(const int *nArguments_p, void ** argumentsCPU_p, void ** argumentsGPU_p,  
     void **a_p)
{
    const int nArguments = *nArguments_p;
    if(nArguments == 0) return;
    const int yBlocks = nArguments / 6; //6 since kernel handles 2 vectors

    //Launch one block per NSIM, with each block handling two vectors
    //the pointers to the correct arrays are stored in argumentsGPU

    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    const int nThreads = 512;
    dim3 blockConfig(numSMs*4, yBlocks, 1);

    gpu_daxpy_kernel2_2d<<<blockConfig, nThreads>>>(0, (double*) *a_p, 
	(unsigned long int*) *argumentsGPU_p);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute gpu_daxpy2_2d!" );
}

__global__ void gpu_dscal_kernel2_2d(const double *alpha, unsigned long int* arguments_GPU)
{
    const uint tid = threadIdx.x;
    const int increase = blockDim.x*gridDim.x;
    const int start = blockIdx.x*blockDim.x + tid;

    const int N = (int)arguments_GPU[blockIdx.y*4+0];
    double *X = (double*)arguments_GPU[blockIdx.y*4+1];

    const int N2 = (int)arguments_GPU[blockIdx.y*4+2];
    double *X2 = (double*)arguments_GPU[blockIdx.y*4+3];

    //We use *4 here since the original data are in cuDoubleComplex, however we only
    //use the real part and therefor can use normal multiplication. So this only extract
    //the real part, but could be used to also use the imaginary part
    const double  a1 = alpha[blockIdx.y*4+0];
    const double  a2 = alpha[blockIdx.y*4+2];

    for(int i=start; i < N; i += increase)
        X[i] = a1*X[i];

  for(int i=start; i < N2; i += increase)
      X2[i] = a2*X2[i];
}

extern "C"
void gpu_dscal_2d_C(const int *nArguments_p, void ** argumentsCPU_p,
                   void ** argumentsGPU_p,  void **a_p)
{
    const int nArguments = *nArguments_p;
    if(nArguments == 0) return;
    const int yBlocks = nArguments / 4; //4 since kernel handles 2 vectors

    //Launch one block per NSIM, with each block handling two vectors
    //the pointers to the correct arrays are stored in argumentsGPU

    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    const int nThreads = 512;
    dim3 blockConfig(numSMs*4, yBlocks, 1);

    gpu_dscal_kernel2_2d<<<blockConfig, nThreads>>>((double*) *a_p,
	(unsigned long int*) *argumentsGPU_p);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute gpu_dscal_2d!" );
}

/******************************************************/
