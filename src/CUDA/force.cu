// File: force.cu
// C/Fortran interface to GPU port of force.F.

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
// CUDA kernels/wrappers used in FORLOC and FORHAR

// interpolates the pseudopotential on the grid of reciprocal lattice vectors
__global__ void cuforlocg(int np, int nrow, int npspts, char ngihalf, double tpi, double argsc,
		double psgma2, double zz, double omega, int *i2, int *i3, int *lpctx, int *lpcty,
		int *lpctz, double *lattb, double *psp, double *work)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // for each thread,

    if( idx < np)
    {
        // fetch indices
        int n1 = idx % nrow;
        int nc = idx / nrow;
        int n2 = i2[nc]-1;  // c indexing
        int n3 = i3[nc]-1;

        // fetch loop counters
        double lpx = lpctx[n1];
        double lpy = lpcty[n2];
        double lpz = lpctx[n3];

        // calculate magnitude of reciprocal lattice vector
        double gx = lpx*lattb[0]+lpy*lattb[3]+lpz*lattb[6];
        double gy = lpx*lattb[1]+lpy*lattb[4]+lpz*lattb[7];
        double gz = lpx*lattb[2]+lpy*lattb[5]+lpz*lattb[8];
        double g = sqrt(gx*gx+gy*gy+gz*gz)*tpi;

        if(g!=0 && g<psgma2)
        {
            // convert mag. (g) to a position in charge density array (prho)
	    int i = (int)(g*argsc);  // c indexing
            double rem = g-psp[i];

            // interpolate pseudopotential and its derivative
	    double vpst = psp[i+npspts]+rem*(psp[i+npspts*2]+rem*(psp[i+npspts*3] + rem*psp[i+npspts*4]));
            work[idx]=(vpst+zz/(g*g))/omega;
        }
        else
            work[idx]=0;
    }
}

// interpolates the pseudopotential on the grid of reciprocal lattice vectors
extern "C"
void cuda_forlocg_(int *np, int *nrow, int *npspts, char *ngihalf, double *tpi, double *argsc,
     double *psgma2, double *zz, double *omega, devptr_t *i2, devptr_t *i3, devptr_t *lpctx,
     devptr_t *lpcty, devptr_t *lpctz, devptr_t *devptr_lattb, devptr_t *devptr_psp,
     devptr_t *devptr_work)
{
    // grid dimensions
    int N = *np;
    dim3 block(MAX_THREADS);
    dim3 grid((N+block.x-1)/block.x);

    // device pointers
    double *lattb = (double *)(*devptr_lattb);
    double *psp = (double *)(*devptr_psp);
    double *work = (double *)(*devptr_work);

    // interpolate pseudopotential
    cuforlocg<<<grid,block>>>(*np,*nrow,*npspts,*ngihalf,*tpi,*argsc,*psgma2,*zz,*omega,
	(int*)*i2,(int*)*i3,(int*)*lpctx,(int*)*lpcty,(int*)*lpctz,lattb,psp,work);

    CUDA_ERROR( cudaDeviceSynchronize(), "Failed to execute CUDA kernel cuforlocg!" );
}

// interpolates the pseudopotential on the grid of reciprocal lattice vectors
__global__ void cuforharg(int np, int nrow, char ngihalf, double tpi, double argsc,double psgma2,
		int *i2, int *i3, int *lpctx, int *lpcty, int *lpctz, double *lattb,
		double *prho, double *work)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // for each thread,

    if( idx < np)
    {
        // fetch indices
        int n1 = idx % nrow;
        int nc = idx / nrow;
        int n2 = i2[nc]-1;  // c indexing
        int n3 = i3[nc]-1;

	// fetch loop counters
	double lpx = lpctx[n1];
	double lpy = lpcty[n2];
	double lpz = lpctx[n3];

	// calculate magnitude of reciprocal lattice vector
	double gx = lpx*lattb[0]+lpy*lattb[3]+lpz*lattb[6];
	double gy = lpx*lattb[1]+lpy*lattb[4]+lpz*lattb[7];
	double gz = lpx*lattb[2]+lpy*lattb[5]+lpz*lattb[8];
	double g = sqrt(gx*gx+gy*gy+gz*gz)*tpi;

	if(g!=0 && g<psgma2)
	{
	    // convert mag. (g) to a position in charge density array (prho)
	    double arg = g*argsc+1;
	    int naddr = (arg>2) ? (int)arg : 2;
	    double rem = arg-naddr;
	    naddr-=1;  // c indexing

	    // fetch atomic charge density
	    double v1=prho[naddr-1];
	    double v2=prho[naddr];
	    double v3=prho[naddr+1];
	    double v4=prho[naddr+2];

	    // interpolate atomic charge density
	    double t0=v2;
	    double t1=((6*v3)-(2*v1)-(3*v2)-v4)/6.0;
	    double t2=(v1+v3-(2*v2))/2.0;
	    double t3=(v4-v1+(3*(v2-v3)))/0.6;
	    work[idx]=t0+rem*(t1+rem*(t2+rem*t3));
	}
	else
	    work[idx]=0;
    }
}

// interpolates the pseudopotential on the grid of reciprocal lattice vectors
extern "C"
void cuda_forharg_(const int *np, const int *nrow, char *ngihalf, double *tpi, double *argsc,
     double *psgma2, devptr_t *i2, devptr_t *i3, devptr_t *lpctx, devptr_t *lpcty, devptr_t *lpctz,
     devptr_t *devptr_lattb, devptr_t *devptr_prho, devptr_t *devptr_work)
{
    // grid dimensions
    int N = *np;
    dim3 block(MAX_THREADS);
    dim3 grid((N+block.x-1)/block.x);

    // device pointers
    double *lattb = (double *)(*devptr_lattb);
    double *prho = (double *)(*devptr_prho);
    double *work = (double *)(*devptr_work);

    // interpolate pseudopotential
    cuforharg<<<grid,block>>>(*np,*nrow,*ngihalf,*tpi,*argsc,*psgma2,(int*)*i2,(int*)*i3,
	(int*)*lpctx,(int*)*lpcty,(int*)*lpctz,lattb,prho,work);

    CUDA_ERROR( cudaDeviceSynchronize(), "Failed to execute CUDA kernel cucalcg!" );
}

// calculate the total force on the ions
__global__ void cucalcf(int np, int nrow, char ngihalf, double poisonx, double poisony,
		double poisonz, cuDoubleComplex citpi, double vca, int *i2, int *i3,
		int *lpctx, int *lpcty, int *lpctz, int *lpctx_, int *lpcty_, int *lpctz_,
		cuDoubleComplex *ch, double *work, double *f1, double *f2, double *f3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // for each thread,

    if( idx < np)
    {
	// fetch indices
        int n1 = idx % nrow;
        int nc = idx / nrow;
        int n2 = i2[nc]-1;  // c indexing
        int n3 = i3[nc]-1;

	// calculate phase factor b
	double g = poisonx*lpctx[n1] + poisony*lpctx[n2] + poisonz*lpctx[n3];
	cuDoubleComplex b = cexp(citpi*g)*vca;

	// calculate force contribution for a
	// single reciprocal lattice vector
	cuDoubleComplex a = ch[idx];
	double f0 = work[idx]*(a.x*b.y-a.y*b.x);
	
	// scale force contribution
	if(ngihalf=='z' && n3)
	    f0*=2;
	else if(ngihalf=='x' && n1)
	    f0*=2;

	// add the contribution to the force
	f1[idx]=-lpctx_[n1]*f0;	 // FOR1
	f2[idx]=-lpcty_[n2]*f0;	 // FOR2
	f3[idx]=-lpctz_[n3]*f0;	 // FOR3
    }
}

// calculate the total force on the ions
extern "C"
void cuda_calcf_(const int *np, const int *nrow, char *ngihalf, double *poison,
     cuDoubleComplex *citpi, double *vca, devptr_t *i2, devptr_t *i3,
     devptr_t *lpctx, devptr_t *lpcty, devptr_t *lpctz,
     devptr_t *lpctx_, devptr_t *lpcty_, devptr_t *lpctz_,
     devptr_t *devptr_f1, devptr_t *devptr_f2, devptr_t *devptr_f3,
     devptr_t *devptr_ch, devptr_t *devptr_work, double *force)
{
    // grid dimensions
    int N = *np;
    dim3 block(MAX_THREADS);
    dim3 grid((N+block.x-1)/block.x);
    // size of shared memory buffer
    int ssize = 3*block.x*sizeof(double);

    // device pointers
    cuDoubleComplex *ch = (cuDoubleComplex *)(*devptr_ch);
    double *work = (double *)(*devptr_work);
    double *f1 = (double *)(*devptr_f1);
    double *f2 = (double *)(*devptr_f2);
    double *f3 = (double *)(*devptr_f3);

    // calculate force contributions
    cucalcf<<<grid,block>>>(N,*nrow,*ngihalf,poison[0],poison[1],poison[2],*citpi,*vca,
        (int*)*i2,(int*)*i3,(int*)*lpctx,(int*)*lpcty,(int*)*lpctz,(int*)*lpctx_,
        (int*)*lpcty_,(int*)*lpctz_,ch,work,f1,f2,f3);

    // calculate total force by summing
    // over reciprocal lattice vectors
    cureducesum3<<<grid,block,ssize>>>(f1,f2,f3,N);
    cureducesum3_1block<<<1,block,ssize>>>(f1,f2,f3,grid.x);

    // copy sums from device to host
    CUDA_ERROR( cudaMemcpy(&force[0],f1,sizeof(double),cudaMemcpyDeviceToHost),
                "Failed to copy from device to host in cuda_calcharfor!");
    CUDA_ERROR( cudaMemcpy(&force[1],f2,sizeof(double),cudaMemcpyDeviceToHost),
                "Failed to copy from device to host in cuda_calcharfor!");
    CUDA_ERROR( cudaMemcpy(&force[2],f3,sizeof(double),cudaMemcpyDeviceToHost),
                "Failed to copy from device to host in cuda_calcharfor");
}

/******************************************************/
