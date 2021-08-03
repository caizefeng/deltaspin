// File: hamil.cu
// C/Fortran interface to GPU port of hamil.F.

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
#include <cublas_v2.h>

extern cublasHandle_t hcublas;

/******************************************************/
// CUDA kernels/wrappers used in VHAMIL

template <class NumType>
__global__ void cuvhamil_spinors(const int np, const int mplwv, cuDoubleComplex *CVR, 
		const NumType *SV, const int strideSV, 
		const cuDoubleComplex *CR, const double rinplw)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx < np)
    {
	cuDoubleComplex tmp;

	int index=idx+mplwv;  // index for CVR and CR
	int index2=idx+strideSV;  // index for SV

	// ISPINOR = 0
	tmp=SV[idx]*CR[idx]*rinplw;
	CVR[idx]=tmp+SV[index2]*CR[index]*rinplw;  index2+=strideSV;

	// ISPINOR = 1
	tmp=SV[index2]*CR[idx]*rinplw;  index2+=strideSV;
	CVR[index]=tmp+SV[index2]*CR[index]*rinplw;
    }
}

extern "C"
void cuda_vhamil_C(const int *sid, const int *nrspinors, const int *np, const int *mplwv, 
     const devptr_t *devptrCVR, const devptr_t *devptrSV, const int *shift,
     const devptr_t *devptrCR, const double *rinplw, const int *is_real)
{
    // grid dimensions
    dim3 block(256);
    dim3 grid((*np+block.x-1)/block.x);

    // device pointers
    cuDoubleComplex *CVR = (cuDoubleComplex *)(*devptrCVR);
    cuDoubleComplex *SV = (cuDoubleComplex *)(*devptrSV);
    cuDoubleComplex *CR = (cuDoubleComplex *)(*devptrCR);

    // set CUDA stream
    cudaStream_t st = 0;
    if(*sid>=0)
	st = stream[*sid];

    if(*nrspinors == 1)  // one spinor
    {
	    CUDA_ERROR( cudaMemsetAsync(CVR,0,(*np)*sizeof(cuDoubleComplex),st),
			"Failed to memset device memory async!" );
	if(*is_real){
//	    cuvhamil<double><<<grid,block,0,st>>>(*np,CVR,((double *)SV)+(*shift),CR,*rinplw);
            mul_vec_k<double><<<grid, block, 0, st>>>(CVR, CR, ((double *)SV)+(*shift), *rinplw, *np, 1);
	} else {
//		cuvhamil<cuDoubleComplex><<<grid,block,0,st>>>(*np,CVR,SV+(*shift),CR,*rinplw);
		mul_vec_k<cuDoubleComplex><<<grid, block, 0, st>>>(CVR, CR, SV+(*shift), *rinplw, *np, 1);
        }
	CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuvhamil!" );
    }
    else  // multiple spinors
    {
        if(*is_real)
	{
	    CUDA_ERROR( cudaMemsetAsync(CVR,0,(*mplwv)*2*sizeof(double),st),
			"Failed to memset device memory async!" );
            cuvhamil_spinors<double><<<grid,block,0,st>>>(*np,*mplwv,CVR,((double *)SV)+
		(*shift),(*mplwv)*2,CR,*rinplw);
	}
        else
	{
	    CUDA_ERROR( cudaMemsetAsync(CVR,0,(*mplwv)*2*sizeof(cuDoubleComplex),st),
                        "Failed to memset device memory async!" );
            cuvhamil_spinors<cuDoubleComplex><<<grid,block,0,st>>>(*np,*mplwv,CVR,SV+
		(*shift),*mplwv,CR,*rinplw);
	}
	CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuvhamil_spinors!" );
    }
}

/******************************************************/
// CUDA kernels/wrappers used in KINHAMIL_GPU

__global__ void cukinhamil(cuDoubleComplex *ch, cuDoubleComplex *cw,
		double *datake, double evalue, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // for each thread,

    if(idx < N)
    {
	double de = datake[idx] - evalue;
	// 
	ch[idx] = ch[idx] + cw[idx]*de;
    }
}

extern "C"
void cuda_kinhamil_C(int *sid, int *N, cuDoubleComplex **ch,
     cuDoubleComplex **cw, double **datake, double *evalue)
{
    // grid dimensions
    dim3 block(256);
    dim3 grid((*N+block.x-1)/block.x);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    // compute, ch = ch - cw*evalue + cw*datake
    cukinhamil<<<grid,block,0,st>>>(*ch,*cw,*datake,*evalue,*N);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cukinhamil!" );
}

/******************************************************/
// CUDA kernels/wrappers used in FFTHAMIL_GPU

extern "C"
void cuda_ffthamil_C(int *sid, int *N, cuDoubleComplex **ch, cuDoubleComplex **cw,
     double *evalue)
{
    // grid dimensions
    cuDoubleComplex zvalue;

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream
    // compute, ch = ch - cw*evalue
    // we don't care about zvalue being complex because axpy is bandwidth-bound
    zvalue.x = *evalue; zvalue.y = 0.;
    cublasSetStream(hcublas , st);
    cublasZaxpy(hcublas, *N, &zvalue, *cw, 1, *ch, 1);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA 'kernel' cuffthamil!" );
}

/******************************************************/
// CUDA kernels/wrappers used in SETUP_PRECOND

// calculate ekin for blockDim.x wavefunctions
__global__ void cucalcekin(double *data, devptr_t *cw_ptrs, const double *datake,
		const int ngdim, const int ngvector, const int N)
{
    //extern __shared__ double sdata[];
    double ekin = 0;

    // for each thread,

    // fetch pointer for each block
    cuDoubleComplex *cw = (cuDoubleComplex *)cw_ptrs[blockIdx.x];
    if(cw == NULL) return;  // return if not valid

    // fetch from global memory and add
    for(int i=threadIdx.x; i<N; i+=blockDim.x)
    {
	int j = i / ngvector;
	int k = i % ngvector;
	cuDoubleComplex c = cw[i];
	ekin += (c.x*c.x + c.y*c.y)*datake[j*ngdim+k];
    }

    // perform sum reduction
    __cureducesum(data, sdata, ekin);
}

extern "C"
void cuda_initprecond_C(const int *nsim, const int *N, const devptr_t *devptr_cw,
     const devptr_t *devptr_datake, double *ekin, const int *ngdim,
     const int *ngvector, const int *ialgo)
{
    // copy array of pointers from host to device
    CUDA_ERROR( cudaMemcpyAsync(d_ptrs,devptr_cw,(*nsim)*sizeof(devptr_t),cudaMemcpyHostToDevice,0), "Failed to copy from host to device async!" );


    if((*ialgo)==0 || (*ialgo)==8 || (*ialgo)==6)
    {
	// grid dimensions
	dim3 block(MAX_THREADS);
	dim3 grid(*nsim);

	// size of shared memory buffer
        int ssize = block.x * sizeof(double);

        // device pointers
        double *datake = (double *)(*devptr_datake);

	// calculate ekin for nsim wavefunctions
	cucalcekin<<<grid,block,ssize>>>(d_reduce,d_ptrs,datake,*ngdim,
	    *ngvector,*N);
	// copy ekin from device to host
	CUDA_ERROR( cudaMemcpy(ekin,d_reduce,(*nsim)*sizeof(double),cudaMemcpyDeviceToHost), "Failed to copy from device to host!" );
    }
}

// calculate preconditioner for IALGO = 0, 8, or 6
__global__ void cucalcprecondi086(double *ekin, devptr_t *cw_ptrs, 
		double *datake, double *precond, const int ngvector, 
		const int ngdim, const int nrplwv, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y;

    // for each thread,

    // return if not valid!
    if(cw_ptrs[blockIdx.y] == NULL) return;

    if(idx < N)
    {
        int i = idx / ngvector;
        int j = idx % ngvector;
        int stride = idy*nrplwv;

        double x = datake[i*ngdim+j] * ekin[idy];
        double y = ((8*x+12)*x+18)*x+27;
        double x2 = x*x;
        double x4 = x2*x2;
        precond[stride+idx] = y/(y+16*x4)*2*ekin[idy];
    }
}

// calculate preconditioner for IALGO = 9
__global__ void cucalcprecondi9(double *ekin, devptr_t *cw_ptrs,
                double *datake, double *precond, double slocal, double de_att,
                const int ngvector, const int ngdim, const int nrplwv, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y;

    // for each thread,

    // return if not valid!
    if(cw_ptrs[blockIdx.y] == NULL) return;

    if(idx < N)
    {
        int i = idx / ngvector;
        int j = idx % ngvector;
        int stride = idy*nrplwv;

        double y = de_att;
        double x = datake[i*ngdim+j] - (ekin[idy]-slocal);
        if(x < 0) x = 0;
        // reciprocal of z=x+yi
        precond[stride+idx] = x/(x*x+y*y); // store real term only!
    }
}

// calculate preconditioner for IALGO != 0, 8, 6, and 9
__global__ void cucalcprecondi(devptr_t *cw_ptrs, double *precond, const int nrplwv, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y;

    // for each thread,

    // return if not valid!
    if(cw_ptrs[blockIdx.y] == NULL) return;

    if(idx < N)
    {
        int stride = idy*nrplwv;
        precond[stride+idx] = 1.;
    }
}

extern "C"
void cuda_calcprecond_C(const int *nsim, const int *N, devptr_t *devptr_cw,
     const devptr_t *devptr_datake, const devptr_t *devptr_precon, double *ekin, 
     double *evalue, const double *slocal, const double *de_att, const int *ngdim,
     const int *ngvector, const int *nrplwv, const int *ialgo)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*N+block.x-1)/block.x,*nsim);

    // device pointers
    double *datake = (double *)(*devptr_datake);
    double *precon = (double *)(*devptr_precon);

    if((*ialgo)==0 || (*ialgo)==8 || (*ialgo)==6)
    {
        for(int i=0; i<(*nsim); i++)
        {
	    //NOTE(ca):avoid "NULL used in arithmetic" warning and make intent clear that devptr_t contains a pointer
            if((void*)devptr_cw[i] != NULL)
            {
                double tmp = ekin[i];
                if(tmp < 2)
                    tmp = 2;
                tmp*=1.5;
                ekin[i] = 1./tmp;
            }
        }

	// copy ekin to device memory
	CUDA_ERROR( cudaMemcpyAsync(d_reduce,ekin,(*nsim)*sizeof(double),cudaMemcpyHostToDevice,0), "Failed to copy from host to device async!" );
	// compute preconditioner
	cucalcprecondi086<<<grid,block>>>(d_reduce,d_ptrs,datake,precon,*ngvector,*ngdim,*nrplwv,*N);
    }
    else if((*ialgo)==9)  // not verified! find benchmark to test on.
    {
	// copy evalue to device memory
        CUDA_ERROR( cudaMemcpyAsync(d_reduce,evalue,(*nsim)*sizeof(double),cudaMemcpyHostToDevice,0), "Failed to copy from host to device async!" );
	// compute preconditioner
	cucalcprecondi9<<<grid,block>>>(d_reduce,d_ptrs,datake,precon,*slocal,*de_att,*ngvector,
	    *ngdim,*nrplwv,*N);
    }
    else  // not verified! find benchmark to test on.
    {
	// compute preconditioner
        cucalcprecondi<<<grid,block>>>(d_ptrs,precon,*nrplwv,*N);
    }
}

/******************************************************/
// CUDA kernels/wrappers used in APPLY_PRECOND

__global__ void cuapplyprecond(cuDoubleComplex *cw2, cuDoubleComplex *cw1,
		const double *precon, const double mul, const int shift,
		const int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx < N)
    {
        cuDoubleComplex ret = cw1[idx];
        double p = precon[idx + shift];
	// cw2 = precon * cw2 * mul
        ret.x = ret.x * p * mul;
        ret.y = ret.y * p * mul;
        cw2[idx] = ret;
    }
}

extern "C"
void cuda_applyprecond_C(const int *sid, const int *N, devptr_t *devptr_cw1,
     devptr_t *devptr_cw2, devptr_t *devptr_precon, const double *mul, const int *shift)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*N+block.x-1)/block.x);

    // device pointers
    cuDoubleComplex *cw1 = (cuDoubleComplex *)(*devptr_cw1);
    cuDoubleComplex *cw2 = (cuDoubleComplex *)(*devptr_cw2);
    double *precon = (double *)(*devptr_precon);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    // calculate cw2 = precon * cw2 * mul
    cuapplyprecond<<<grid,block,0,st>>>(cw2,cw1,precon,*mul,*shift,*N);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cuapplyprecond!" );
}

/******************************************************/
// CUDA kernels/wrappers used in TRUNCATE_HIGH_FREQUENCY_W1

// truncate high frequency values in wavefunction
__global__ void cutruncatehighfrequency(cuDoubleComplex *cw,
		const double *datake, const double enini, const int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // for each thread,

    if(idx < N)
    {
	// truncate high frequency values
        if(datake[idx] > enini)
        {
            cw[idx].x = 0;
            cw[idx].y = 0;
        }
    }
}

extern "C"
void cuda_truncatehighfrequency_C(const int *N, const int *ldelaylspiral,
     devptr_t *devptr_cw, devptr_t *devptr_datake, const double *enini)
{
    // exit condition
    if( !(*ldelaylspiral) )
	return;

    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*N+block.x-1)/block.x);

    // device pointers
    cuDoubleComplex *cw = (cuDoubleComplex *)(*devptr_cw);
    double *datake = (double *)(*devptr_datake);

    // compute, need to test this for correct values!
    cutruncatehighfrequency<<<grid,block>>>(cw,datake,*enini,*N);

    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cutruncatehighfrequency!" );
}

/******************************************************/
// CUDA kernels/wrappers used in PW_NOMR_WITH_METRIC

// calculate partial sums for norm
__global__ void cucalcnorm(double *data, const cuDoubleComplex *c, const int N)
{
    //extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double norm = 0;

    // for each thread,

    // fetch from global memory and add
    for(int i=idx; i<N; i+=blockDim.x*gridDim.x)
    {
        double x = c[i].x;
        double y = c[i].y;
        norm += x*x + y*y;
    }

    // perform sum reduction
    __cureducesum(data, sdata, norm);
}

// calculate partial sums for norm and metric
__global__ void cucalcnorm2(double *data, double *data1, const cuDoubleComplex *c,
		double *metric, const int shift, const int N)
{
    //extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double fnorm = 0, fmetric = 0;

    // for each thread,

    // fetch from global memory and add
    for(int i=idx; i<N; i+=blockDim.x*gridDim.x)
    {
        double x = c[i].x;
        double y = c[i].y;
	double cmul = x*x + y*y;
        fnorm += cmul;
	fmetric += cmul * metric[i+shift];
    }

    // perform two sum reduction
    __cureducesum2(data, data1, sdata, fnorm, fmetric);
}

extern "C"
void cuda_normwithmetric_C(int *sid, int *N, devptr_t *devptr_c, devptr_t *devptr_metric,
     double *fnorm, double *fmetric, int *fmetric_present, int *shift)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid( min((*N+block.x-1)/block.x, MAX_THREADS) );
    // size of shared memory buffer
    int ssize = block.x * sizeof(double);

    // device pointers
    cuDoubleComplex *c = (cuDoubleComplex *)(*devptr_c);
    double *metric = (double *)(*devptr_metric);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    if(*fmetric_present != 0)  // norm and metric
    {
        // calculate partial two sums, store in d_reduce
        cucalcnorm2<<<grid,block,2*ssize,st>>>(d_reduce,d_reduce1,c,metric,*shift,*N);
	// calculate two sums
	cureducesum2<<<1,block,2*ssize,st>>>(d_reduce,d_reduce1,grid.x);
	// copy sums from device to host
        CUDA_ERROR( cudaMemcpyAsync(fmetric,d_reduce1,sizeof(double),
		    cudaMemcpyDeviceToHost,st),"Failed to copy from device to host!" );
    }
    else  // norm only
    {
        // calculate partial sums, store in d_reduce
        cucalcnorm<<<grid,block,ssize,st>>>(d_reduce,c,*N);
        // calculate sum
        cureducesum1block<<<1,block,ssize,st>>>(d_reduce,grid.x);
    }
        // copy sums from device to host
        CUDA_ERROR( cudaMemcpyAsync(fnorm,d_reduce,sizeof(double),cudaMemcpyDeviceToHost,st),
            "Failed to copy from device to host!" );
}

// blocked versions of the above...

// calculate partial sums for blockDim.x norm vector
__global__ void cucalcnormblock(double *data, const devptr_t *c_ptrs, const int N)
{
    //extern __shared__ double sdata[];
    double fnorm=0;

    // for each thread,

    // fetch pointer for each block
    cuDoubleComplex *c = (cuDoubleComplex *)c_ptrs[blockIdx.x];
    if(c == NULL) return;  // return if not valid

    // fetch from global memory and add
    for(int i=threadIdx.x; i<N; i+=blockDim.x)
    {
        double x = c[i].x;
        double y = c[i].y;
        fnorm += x*x + y*y;
    }

    // perform sum reduction
    __cureducesum(data, sdata, fnorm);
}

// calculate partial two sum for blockDim.x norm and metric vectors
__global__ void cucalcnorm2block(double *data, double *data1, const devptr_t *c_ptrs,
		double *metric, const int N)
{
    //extern __shared__ double sdata[];
    double fnorm=0, fmetric=0;

    // for each thread,

    // fetch pointer for each block
    cuDoubleComplex *c = (cuDoubleComplex *)c_ptrs[blockIdx.x];
    if(c == NULL) return;  // return if not valid

    int shift = N*blockIdx.x;
    // fetch from global memory and add
    for(int i=threadIdx.x; i<N; i+=blockDim.x)
    {
        double x = c[i].x;
        double y = c[i].y;
	double cmul = x*x + y*y;
        fnorm += cmul;
	fmetric += cmul * metric[i+shift];
    }

    // perform two sum reduction
    __cureducesum2(data, data1, sdata, fnorm, fmetric);
}

extern "C"
void cuda_normwithmetricblock_C(int *nsim, int *N, devptr_t *devptr_c, devptr_t *devptr_metric,
     double *fnorm, double *fmetric, int *fmetric_present)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid(*nsim);
    // size of shared memory buffer
    int ssize = block.x * sizeof(double);

    // device pointers
    double *metric = (double *)(*devptr_metric);

    // copy array of pointers from host to device
    CUDA_ERROR( cudaMemcpy(d_ptrs,devptr_c,(*nsim)*sizeof(devptr_t),cudaMemcpyHostToDevice),
		"Failed to copy from host to device!" );

    if(*fmetric_present != 0)  // norm and metric
    {
	// calculate nsim two sums
	cucalcnorm2block<<<grid,block,2*ssize>>>(d_reduce,d_reduce1,d_ptrs,metric,*N);
	// copy sums from device to host
	CUDA_ERROR( cudaMemcpy(fmetric,d_reduce1,(*nsim)*sizeof(double),
		    cudaMemcpyDeviceToHost), "Failed to copy from device to host!" );
    }
    else  // norm only
    {
	// calculate nsim sums
	cucalcnormblock<<<grid,block,ssize>>>(d_reduce,d_ptrs,*N);
    }
    // copy sums from device to host
    CUDA_ERROR( cudaMemcpy(fnorm,d_reduce,(*nsim)*sizeof(double),cudaMemcpyDeviceToHost),
                "Failed to copy from device to host!" );
}

/******************************************************/
// CUDA kernels/wrappers for PW_CHARGE, used in SOFT_CHARGE

template <int cmplx>
__global__ void cupwcharge(const int np, cuDoubleComplex *charge, const cuDoubleComplex *cr1,
                const cuDoubleComplex *cr2, const double weight)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // for each thread,

    if(idx < np)
    {
        cuDoubleComplex mul = cuConj(cr2[idx])*cr1[idx];

        if(cmplx)
	{
	    cuDoubleComplex *c = charge;
            c[idx] = c[idx] + mul*weight;
	}
        else
	{
	    double *c = (double *)charge;
            c[idx] = c[idx] + mul.x*weight;
	}
    }
}

template <int cmplx>
__global__ void cupwcharge_spinors(const int np, const int mplwv, cuDoubleComplex *charge,
                const int ndim, const cuDoubleComplex *cr1, const cuDoubleComplex *cr2,
                const double weight)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // for each thread,

    if(idx < np)
    {
	for(int i=0; i<2; i++)
	for(int j=0; j<2; j++)
	{
	    cuDoubleComplex mul = cuConj(cr2[idx+j*mplwv])*cr1[idx+i*mplwv];
	    int index = idx+j*ndim+i*2*ndim;

	    if(cmplx)
	    {
	        cuDoubleComplex *c = charge;
	        c[index] = c[index] + mul*weight;
	    }
	    else
	    {
		double *c = (double *)charge;
		 c[index] = c[index] + mul.x*weight;
	    }
	}
    }
}

extern "C"
void cuda_pwcharge_C(const int *sid, const int *nrspinors, const int *np, const int *mplwv,
     const devptr_t *devptr_charge, const int *ndim, const devptr_t *devptr_cr1,
     const devptr_t *devptr_cr2, const double *weight, const int *is_real)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*np+block.x-1)/block.x);

    // device pointers
    cuDoubleComplex *charge = (cuDoubleComplex *)(*devptr_charge);
    cuDoubleComplex *cr1 = (cuDoubleComplex *)(*devptr_cr1);
    cuDoubleComplex *cr2 = (cuDoubleComplex *)(*devptr_cr2);

    // set CUDA stream
    cudaStream_t st = 0;
    if(*sid>=0)
        st = stream[*sid];

    // charge = charge + conj(cr2) * cr1 * weight
    if(*nrspinors == 1)  // one spinor
    {
        if(*is_real)
        cupwcharge<0><<<grid,block,0,st>>>(*np,charge,cr1,cr2,*weight);
        else
        cupwcharge<1><<<grid,block,0,st>>>(*np,charge,cr1,cr2,*weight);
        CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cupwcharge!" );
    }
    // charge = charge + conj(cr2) * cr1 * weight
    else  // multiple spinors
    {
        if(*is_real)
        cupwcharge_spinors<0><<<grid,block,0,st>>>(*np,*mplwv,charge,*ndim,cr1,cr2,*weight);
        else
        cupwcharge_spinors<1><<<grid,block,0,st>>>(*np,*mplwv,charge,*ndim,cr1,cr2,*weight);
        CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cupwcharge_spinors!" );
    }
}

/******************************************************/
// CUDA kernels/wrappers for PW_CHARGE_TRACE, used in ?

template <int cmplx>
__global__ void cupwchargetrace_spinors(int np, int mplwv, cuDoubleComplex *charge,
                cuDoubleComplex *cr1, cuDoubleComplex *cr2)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // for each thread,

    if(idx < np)
    {
	int index = idx + mplwv;
        cuDoubleComplex mul = cuConj(cr2[index])*cr1[index];

        if(cmplx)
        {
            cuDoubleComplex *c = charge;
            c[index] = c[index] + mul;
        }
        else
        {
            double *c = (double *)charge;
            c[index] = c[index] + mul.x;
        }
    }
}

#if 0
extern "C"
void cuda_pwchargetrace_(const int *sid, const int *nrspinors, const int *np, const int *mplwv,
     const devptr_t *devptr_charge, const devptr_t *devptr_cr1, const devptr_t *devptr_cr2,
     const int *is_real)
{
    // grid dimensions
    dim3 block(MAX_THREADS);
    dim3 grid((*np+block.x-1)/block.x);

    // device pointers
    cuDoubleComplex *charge = (cuDoubleComplex *)(*devptr_charge);
    cuDoubleComplex *cr1 = (cuDoubleComplex *)(*devptr_cr1);
    cuDoubleComplex *cr2 = (cuDoubleComplex *)(*devptr_cr2);

    // set CUDA stream
    cudaStream_t st = 0;
    if(*sid>=0)
        st = stream[*sid];

    // charge = conj(cr2) * cr1
    if(*is_real)
        charge_trace_k<0><<<grid, block, 0, st>>>(cr1, cr2, charge, *np, 1);
    else
        charge_trace_k<1><<<grid, block, 0, st>>>(cr1, cr2, charge, *np, 1);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cupwchargetrace!" );

    // charge = charge + conj(cr2) * cr1
    if(*nrspinors==2)  // multiple spinors
    {
    if(*is_real)
        cupwchargetrace_spinors<0><<<grid,block,0,st>>>(*np,*mplwv,charge,cr1,cr2);
    else
        cupwchargetrace_spinors<1><<<grid,block,0,st>>>(*np,*mplwv,charge,cr1,cr2);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cupwchargetrace_spinors!" );
    }
}
#endif

/******************************************************/
// CUDA kernels/wrappers for ..., used in ...

/******************************************************/

