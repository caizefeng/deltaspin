// File: cuda_fft.cu
// C/Fortran interface to CUFFT amd CUDA C FFT.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <sstream>
// includes cuda headers
#include <cuda_runtime.h>
#include <cufft.h>
// includes project headers
#include "cuda_globals.h"
#include "Operator.h"

// global variables
int NUM_PLANS=0;	// number of CUFFT plans
typedef struct BatchedCUFFTPlan
{
    cufftHandle *plan;
    int rank;
    int * n;
    int * inembed;
    int istride;
    int idist;
    int * onembed;
    int ostride;
    int odist;
    cufftType type;
    int batch;
} BatchedCUFFTPlan;

cufftHandle nullplan,*plan;

/******************************************************/
// CUFFT wrapper for CUFFT API errors, in library

extern "C" char *cufftGetErrorString(cufftResult error);
// wrapper for CUFFT API errors
inline void __cufft_error(cufftResult status, const char *file, int line, const char *msg)
{
    if(status != CUFFT_SUCCESS)
    {
        printf("\nCUFFT Error in %s, line %d: %s\n %s\n", file, line, cufftGetErrorString(status), msg);
        cudaDeviceReset();
        exit(-1);
    }
}
#define CUFFT_ERROR(status, msg)  __cufft_error( status, __FILE__, __LINE__, msg )

/******************************************************/
// CUFFT wrapper for CUFFT plans, in library

inline cufftHandle __cufft_plan(int sid, const char *file, int line)
{
    // safety check
    if(sid >= NUM_PLANS)
    {
        CUDA_ERROR( cudaDeviceReset(), "Failed to reset the device!" );
        printf("\nCUDA Error in %s, line %d: %s\n", file, line, "Invalid CUFFT plan id!");
        exit(-1);
    }

    // return CUFFT plan
    cufftHandle p;
    if(sid<0)  // null plan
        p=nullplan;
    else  // stream plan
        p=plan[sid];
    return p;
}
#define CUFFT_PLAN(sid)  __cufft_plan( sid, __FILE__, __LINE__ )

/******************************************************/
// CUFFT wrappers for init, in VAMP

extern "C"
void cufft_init_C(int *nplans, int *nx, int *ny, int *nz)
{
    int i;

    // check number of CUDA streams requested
    if(*nplans<=0)
        ERROR( "GPU Library", "Invalid number of CUFFT plans, pick a number greater than zero!" );
    if(*nplans>NUM_STREAMS)
	ERROR( "GPU Library", "Invalid number of CUFFT plans, pick a number smaller than CUDA streams" );
    NUM_PLANS=*nplans;
    printf("creating %d CUFFT plans with grid size %d x %d x %d...\n",NUM_PLANS,*nz,*ny,*nx);

    // create CUFFT plans
    plan=(cufftHandle*)malloc(NUM_PLANS*sizeof(cufftHandle));
    for(i=0;i<NUM_PLANS;i++)
    {
	CUFFT_ERROR( cufftPlan3d(&plan[i],*nz,*ny,*nx, CUFFT_Z2Z),
		     "Failed to create CUFFT plan!" );
	CUFFT_ERROR( cufftSetStream(plan[i],stream[i]),
		     "Failed to set CUDA stream to CUFFT plan!" );
    }
    // plan for null stream
    CUFFT_ERROR( cufftPlan3d(&nullplan,*nz,*ny,*nx,CUFFT_Z2Z),
		 "Failed to create CUFFT plan!" );
}

extern "C"
void cufft_destroy_C(void)
{
    int i;

    // destroy CUFFT plans
    for(i=0;i<NUM_PLANS;i++)
	CUFFT_ERROR( cufftDestroy(plan[i]), "Failed to destroy CUFFT plan!" );
    free(plan);
    // destroy null plan
    CUFFT_ERROR( cufftDestroy(nullplan), "Failed to destry CUFFT plan!" );
}

/******************************************************/
// CUFFT wrappers for JBNV, in ...

//JBNV, std::map to contain on the fly allocated plans
std::map< std::string, cufftHandle> onTheFlyFFTPlans;

//This is incase a plan size changes during execution. In theory we could replace the plans above (plan) with these on the fly generated plans.
cufftHandle getOnTheFlyPlan(const int nx, const int ny, const int nz,
	    const int sid, const char *name)
{
    //const size_t planSize = nz*ny*nx;

    printf("getOnTheFlyPlan...%d x %d x %d\n", nx, ny, nz);
    //Conver the input to a key
    std::stringstream ss;
    //ss << planSize; ss << "_"; ss << sid;
    ss << nx; ss << "_"; ss << ny; ss << "_";  ss << nz; ss << "_";  //Dimensions
    ss << sid;  //associated stream-id
    std::string key = ss.str();

    //Check if the plan already exists
    if (onTheFlyFFTPlans.find(key) == onTheFlyFFTPlans.end())
    {
        //New size/stream combo, allocate plan
        cufftHandle planTemp;  //Create the temporary plan with the right
        CUFFT_ERROR( cufftPlan3d(&planTemp, nx, ny, nz, CUFFT_Z2Z),
		     "Failed to create 3D CUFFT plan!" );

      if(sid < 0)
      {
          //No stream specified
      }
      else if(sid >= 1000)
          CUFFT_ERROR( cufftSetStream(planTemp, stream[sid-1000]),
			"Failed to associate CUDA stream with CUFFT plan!" );
      else
          CUFFT_ERROR( cufftSetStream(planTemp, stream[sid]),
		       "Failed to associate CUDA stream with CUFFT plan!" );

      //Store the plan
      onTheFlyFFTPlans[key] = planTemp;

      fprintf(stderr,"Notice: Created a custom plan # %d for size: %dx%dx%d and stream: %d Source function: %s\n", onTheFlyFFTPlans.size(), nx,ny,nz,sid, name);

      return planTemp;
    }
    else
    {
        //Excisting combo return
        return onTheFlyFFTPlans[key];
    }
}

std::vector<BatchedCUFFTPlan> batchedCUFFTPlans;

bool batchedCUFFTPlanMatches(const int i, const int rank, const int * n, const int batch,
     const int *inembed, const int istride, const int idist, const int *onembed, 
     const int ostride, const int odist, const cufftType type)
{
    BatchedCUFFTPlan &p = batchedCUFFTPlans[i];
    if(p.rank == rank)
    {
        if(p.batch == batch && p.istride == istride && p.idist == idist && p.ostride == ostride && p.odist == odist && p.type == type)
        {
            for(int j = 0; j < rank; j++)
            {
                if(p.n[j] != n[j] || p.inembed[j] != inembed[j] || p.onembed[j] != onembed[j])
                    return false;
            }
            return true;
        }
    }
    return false;
}

BatchedCUFFTPlan createBatchedCUFFTPlan(int rank, int *n, int batch, int *inembed,
		 int istride, int idist, int * onembed, int ostride, int odist, cufftType type)
{
    printf("Number of batched FFT plans: %d\n", batchedCUFFTPlans.size()+1);
    cufftHandle *plan = new cufftHandle;
    // WARNING: this call to cufftPlanMany is crashing with MPI!!!
    cufftPlanMany(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch);
    //CUFFT_ERROR( cufftPlanMany(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,
    //		 type,batch), "Failed to create CUFFT many plan!" );
    BatchedCUFFTPlan p;
    p.plan = plan;
    p.rank = rank;
    p.n = new int[rank];
    p.inembed = new int[rank];
    p.onembed = new int[rank];
    for(int i = 0; i < rank; ++i)
    {
        p.n[i] = n[i];
        p.inembed[i] = inembed[i];
        p.onembed[i] = onembed[i];
    }
    p.batch = batch;
    p.istride = istride;
    p.ostride = ostride;
    p.idist = idist;
    p.odist = odist;
    p.type = type;
    return p;
}

extern "C"
void getbatchedfftplan_C(cufftHandle **plan, int *rank, int *n, int *howmany,
     int *inembed, int *istride, int *idist, int *onembed, int *ostride,
     int *odist, cufftType *type)
{
    for(int i=0; i<batchedCUFFTPlans.size(); ++i)
    {
        if(batchedCUFFTPlanMatches(i,*rank,n,*howmany,inembed,*istride,*idist,onembed,*ostride,*odist,*type))
        {
            *plan = batchedCUFFTPlans[i].plan;
            return;
        }
    }
    BatchedCUFFTPlan p = createBatchedCUFFTPlan(*rank,n,*howmany,inembed,*istride,*idist,onembed,*ostride,*odist,*type);
    batchedCUFFTPlans.push_back(p);
    *plan = p.plan;
}

extern "C"
void cufft_exec_plan_c2c_C(cufftHandle **plan, cuDoubleComplex **in, cuDoubleComplex **out,
     int *direction)
{
    if(*direction == 1)
        cufftExecZ2Z(**plan, *in, *out, CUFFT_INVERSE);
    else
        cufftExecZ2Z(**plan, *in, *out, CUFFT_FORWARD);
}

extern "C"
void cufft_exec_plan_r2c_C(cufftHandle **plan, double **in, cuDoubleComplex **out)
{
    cufftExecD2Z(**plan, *in, *out);
}

extern "C"
void cufft_exec_plan_c2r_C(cufftHandle **plan, cuDoubleComplex **in, double **out)
{
    cufftExecZ2D(**plan, *in, *out);
}

/******************************************************/
// CUFFT wrappers for 3D FFT, in FFT3D_GPU

extern "C"
void cufft_execz2z_C(const int *sid, const int *n1, const int *n2, const int *n3,
     const devptr_t *idevPtr, const int *ishift, devptr_t *odevPtr, const int *oshift,
     const int *direction)
{
    cufftDoubleComplex *idata=(cufftDoubleComplex *)(*idevPtr);
    cufftDoubleComplex *odata=(cufftDoubleComplex *)(*odevPtr);

    cufftHandle p = CUFFT_PLAN(*sid);  // CUFFT plan

    size_t planSize;
    CUFFT_ERROR( cufftGetSize(p, &planSize), "Failed to get CUFFT plan size!" );
    size_t inputSize = (*n1)*(*n2)*(*n3)*sizeof(cufftDoubleComplex);

    if(inputSize != planSize)
        p = getOnTheFlyPlan(*n3, *n2, *n1,*sid,"cufft_");

    if(*direction < 0)
        CUFFT_ERROR( cufftExecZ2Z(p,idata+(*ishift),odata+(*oshift),CUFFT_FORWARD),
                     "Failed to perform forward FFT!" );
    else
        CUFFT_ERROR( cufftExecZ2Z(p,idata+(*ishift),odata+(*oshift),CUFFT_INVERSE),
                     "Failed to perform backward FFT!" );
}

/******************************************************/
// CUDA C kernels/wrappers for FFT, in FFTWAV_GPU

__global__ void cufftwav(int n, cuDoubleComplex *cr, cuDoubleComplex *c, int *nindpw,
	    double *fftsca, int real2cplx)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // for each thread,
    if(idx < n)
    {
	// fill in non zero elements
	if(real2cplx)
	    cr[nindpw[idx]] = c[idx] * fftsca[idx];
	else
	    cr[nindpw[idx]-1] = c[idx];  // c indexing
    }
}

// transforms c to cr in real space, defined within cutoff-sphere
extern "C"
void cuda_fftwav_C(int *sid, int *n, cuDoubleComplex **cr, cuDoubleComplex **c, int **nindpw,
     double **fftsca, int *real2cplx)
{
    // grid dimensions
    dim3 block(256);
    dim3 grid((*n+block.x-1)/block.x);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    cufftwav<<<grid,block,0,st>>>(*n,*cr,*c,*nindpw,*fftsca,*real2cplx);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute cuda_fftwav!" );
}

/******************************************************/
// CUDA C kernels/wrappers for FFT, in FFTEXT_GPU

__global__ void cufftext(int n, cuDoubleComplex *c, cuDoubleComplex *cr, int *nindpw,
		double *fftsca, int ladd, int real2cplx)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // for each thread,
    if(idx < n)
    {
	// fill in non zero elements
        if(ladd && real2cplx)
	    c[idx] = c[idx] + cr[nindpw[idx]-1]*fftsca[idx];
	else if(ladd && !real2cplx)
	    c[idx] = c[idx] + cr[nindpw[idx]-1];
	else if(real2cplx)
	    c[idx] = cr[nindpw[idx]-1]*fftsca[idx];
	else
	    c[idx] = cr[nindpw[idx]-1];  // c indexing
    }
}

// transforms cr to c in reciprocal space, extracts c from FFT-mesh
extern "C"
void cuda_fftext_C(int *sid, int *n, cuDoubleComplex **c, cuDoubleComplex **cr, int **nindpw,
     double **fftsca, int *ladd, int *real2cplx)
{
    // grid dimensions
    dim3 block(256);
    dim3 grid((*n+block.x-1)/block.x);

    cudaStream_t st = CUDA_STREAM(*sid);  // CUDA stream

    cufftext<<<grid,block,0,st>>>(*n,*c,*cr,*nindpw,*fftsca,*ladd,*real2cplx);
    CUDA_ERROR( cudaGetLastError(), "Failed to execute CUDA kernel cufftext!" );
}

/******************************************************/
