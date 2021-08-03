/**********************************
*      Header for cufft routines
***********************************/

#ifndef CUDA_FFT
#define CUDA_FFT


#include <stdio.h>
#include <stdlib.h>
#include "cufft.h"
#include "cuda.h"
#include "cuda_utils.cu"
#include <assert.h>

template <int NUM_STREAMS>
class CufftPlanBundle
{
    /* collects all bundles with the same dimensions for different streams*/
    bool isInitialized_;
    /* to distinguish planMany plans; this is the total batch size,
 * the assumption is that the work that gets assigned to every stream stays
 * the same between invokations */
    int batchSize_;
    int dims_[3];
    cufftHandle handles_[NUM_STREAMS];

public:
    CufftPlanBundle() : isInitialized_(false) { } 
    virtual ~CufftPlanBundle()
    {
	/* destroys plans if necessary */
	setUninitialized();
    }
    int* getDims()
    {
        return dims_;
    }
    bool isInitialized()
    {
        return isInitialized_;
    }
    void setUninitialized()
    {
	if(isInitialized())
        {
            for(int i = 0; i < NUM_STREAMS; i++)
            {
                CUDA_FFT_CALL(cufftDestroy(handles_[i]));
            }
        }
        isInitialized_ = false;
    }
    void setInitialized()
    {
/*
 * because there are different ways to create cufft plans the API for the cufft plan cache
 * is like this:
 * You ask the cache for a cufft plan batch with given dimentions and the cache
 * WILL return you one. The user then has to check if it has been initializes 
 * (i.e., the cufft plans have been created). If not, the user has to create the
 * plans and set initialized to true. Another option would be to use inheritance to
 * create different kinds of cufft plan batches with their own init routines but
 * the few uses we have in the code doesn't seem to warrant that */
        assert( ! isInitialized() );
        isInitialized_ = true;
    }
    bool matches(int nx, int ny, int nz, int batchSize)
    {
        return isInitialized() 
            && dims_[0] == nx && dims_[1] == ny && dims_[2] == nz
            && batchSize_ == batchSize;
    }
    void setDims(int nx, int ny, int nz, int batchSize)
    {
        if( matches(nx, ny, nz, batchSize) )
        {
            return;
        }
	setUninitialized();
        dims_[0] = nx;
        dims_[1] = ny;
        dims_[2] = nz;
        batchSize_ = batchSize;
    }
    cufftHandle& operator[](std::size_t i)
    {
        assert(i < NUM_STREAMS);
        return handles_[i];
    }
};

template <int CACHE_SIZE, int NUM_STREAMS>
class CufftPlanCache
{
/* abstract base class to collect and cache cufftHandles
 * will be subclassed with concrete plan creation implementations
 * e.g., cufftPlanMany etc */

private:
    CufftPlanBundle<NUM_STREAMS> cache_[CACHE_SIZE];

public:
    CufftPlanCache()
    {
        //Calls constructor on CufftPlanBundle elements, sets them to initialized=false
    }
    virtual ~CufftPlanCache() 
    { 
        //will destroy cache[] which will call cufftDestroy in CufftPlanBundle
    }

    CufftPlanBundle<NUM_STREAMS>& getBundle(int nx, int ny, int nz, int batchSize = -1)
    {
        //test all but the last bundle if one matches or is uninitialized
        for(int i = 0; i < CACHE_SIZE - 1; i++)
        {
            CufftPlanBundle<NUM_STREAMS> &bundle = cache_[i];
            if( ! bundle.isInitialized() )
            {
                bundle.setDims(nx, ny, nz, batchSize);
                return bundle;
            } else if(bundle.matches(nx, ny, nz, batchSize)) {
                return bundle;
            }
        }
        //didn't find a matching bundle, use the last bundle
        CufftPlanBundle<NUM_STREAMS> &bundle = cache_[CACHE_SIZE - 1];
        bundle.setDims(nx, ny, nz, batchSize);
        return bundle;
    }
};


extern "C" void fftwav_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal);

/*
extern "C" void fftwav_cu_gpu(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal);
*/

/*
extern "C" void fftwav_w1_cu(int *npl, int *nk, int *ind,
                              cuDoubleComplex *CR, cuDoubleComplex *C,
                              int *grid, int *lreal, int *ntrans);
*/

extern "C" void fftext_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal, int* ladd);

/*
extern "C" void fftext_cu_gpu(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal, int* ladd);
*/

extern "C" void fft_3d_c2c_C(int *nx, int *ny, int *nz,
                            cuDoubleComplex *a_h, cuDoubleComplex *b_h,
                            int *DIR);
#endif
