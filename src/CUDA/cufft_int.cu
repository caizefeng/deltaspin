/** \file cufft_int.cu
 * 
 * Contains wrappers around cufft calls for
 * use in intercepted FFT calls found in
 * cufft3d.F and cufftmpi.F
 */
#include "cufft_int.h"

#define MAX(a,b) (((a) < (b)) ?  (b) : (a))
#define MIN(a,b) (((a) > (b)) ?  (b) : (a))

#ifndef NUM_FFT_STREAM
  #define NUM_FFT_STREAM 1
#endif

#ifndef FFT_CACHE_SIZE
  #define FFT_CACHE_SIZE 12
#endif 

/* FFT variables */
/* Streams and Events */
static cudaStream_t streams[NUM_FFT_STREAM];

static CufftPlanCache<FFT_CACHE_SIZE, NUM_FFT_STREAM> stream_plan_cache;

//static int index_array_id = -1;
static int_p *index_d = NULL;

#ifdef CUFFT_SP
static cuComplex_p *wr_d = NULL, *wk_d = NULL;
static cuComplex_p *ac2c_d = NULL, *bc2c_d = NULL;
static float* buffer = NULL;
static size_t buf_size = 0;
#else
static cuDoubleComplex_p *wr_d = NULL, *wk_d = NULL; // for FFTWAV/FFTEXT
static cuDoubleComplex_p *ac2c_d = NULL, *bc2c_d = NULL; // for fft_3d
#endif

#define FFT_CACHE_SIZE 12
static CufftPlanCache<FFT_CACHE_SIZE, 1> plan_c2c_cache;

inline void check_sizes(int wk_size, int *grid, int ntrans);
#ifdef CUFFT_SP
inline int realloc_buffer(size_t size);
inline int realloc_buffer_gpu(size_t size);
int BUFFER_GPU = 0;
#endif

/** Initialize all the pointers */
static int cufft_init = 0;
static inline void make_context(void){
  if (wr_d == NULL){
#ifdef CUFFT_SP
    wr_d = (cuComplex_p*) create_cu();
#else
    wr_d = (cuDoubleComplex_p*) create_cu();
#endif
  }
  if (wk_d == NULL){
#ifdef CUFFT_SP
    wk_d = (cuComplex_p*) create_cu();
#else
    wk_d = (cuDoubleComplex_p*) create_cu();
#endif
  }
  if (index_d == NULL){
    index_d = (int_p*) create_cu();
  }
#ifdef CUFFT_SP
  ac2c_d = (cuComplex_p*) create_cu();
  bc2c_d = (cuComplex_p*) create_cu();
#else
  ac2c_d = (cuDoubleComplex_p*) create_cu();
  bc2c_d = (cuDoubleComplex_p*) create_cu();
#endif

  for (int i = 0; i < NUM_FFT_STREAM; i++)
    cudaStreamCreate(&streams[i]);

  cufft_init = 1;
}

/** Re-space wavefunction in k-space before FFT */
#ifdef CUFFT_SP
static __global__ void inflate_k(cuComplex *wk, cuComplex *wr, int *ind, int wk_size, int wr_size, int ntrans){
#else
static __global__ void inflate_k(cuDoubleComplex *wk, cuDoubleComplex *wr, int *ind, int wk_size, int wr_size, int ntrans){
#endif
  int i, j, index;
#ifdef CUFFT_SP
  cuComplex *wr_p, *wk_p;
#else
  cuDoubleComplex *wr_p, *wk_p;
#endif

  /* Loop over bands */
  for (j = blockIdx.x; j < ntrans; j+= gridDim.x){
    /* Create band-specific aliases */
    wr_p = wr + j*wr_size; wk_p = wk + j*wk_size;
    /* Extract relavent values */
    for (i = threadIdx.x; i < wk_size; i+= blockDim.x){
      index = ind[i]-1;
      wr_p[index].x = wk_p[i].x;
      wr_p[index].y = wk_p[i].y;
    }
  }

  return;
}


/** Re-space wavefunction in k-space after FFT */
#ifdef CUFFT_SP
static __global__ void deflate_k(cuComplex *wk, cuComplex *wr, int *ind, int wk_size, int wr_size, int ntrans, int ladd){
#else
static __global__ void deflate_k(cuDoubleComplex *wk, cuDoubleComplex *wr, int *ind, int wk_size, int wr_size, int ntrans, int ladd){
#endif
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int i, j;
  if (ladd){
    for (j = 0; j < ntrans; j++){
      for (i = idx; i < wk_size; i+= blockDim.x * gridDim.x){
        wk[i + j*wk_size].x += wr[ind[i]-1 + j*wr_size].x;
        wk[i + j*wk_size].y += wr[ind[i]-1 + j*wr_size].y;
      }
    }

  } else {
    for (j = 0; j < ntrans; j++){
      for (i = idx; i < wk_size; i+= blockDim.x * gridDim.x){
        wk[i + j*wk_size].x = wr[ind[i]-1 + j*wr_size].x;
        wk[i + j*wk_size].y = wr[ind[i]-1 + j*wr_size].y;
      }
    }
  }

  return;
}

/** Batch-process FFTs using streams */
#ifdef CUFFT_SP
static inline void stream_ffts(cuComplex* data, int* size, int dir, int num){
#else
static inline void stream_ffts(cuDoubleComplex* data, int* size, int dir, int num){
#endif

  /* Compute number of transforms per stream */
  int stride = (num-1)/NUM_FFT_STREAM +1;
  
  //assert(size[0] == size[2]);

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
#ifdef CUFFT_SP    
      CUDA_FFT_CALL(cufftPlanMany(&stream_plan_bundle[i], 3, stream_plan_bundle.getDims(), NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, part_size))
#else
      CUDA_FFT_CALL(cufftPlanMany(&stream_plan_bundle[i], 3, stream_plan_bundle.getDims(), NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, part_size))
#endif
      CUDA_FFT_CALL(cufftSetStream(stream_plan_bundle[i], streams[i]));

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
#ifdef CUFFT_SP
    CUDA_FFT_CALL(cufftExecC2C(stream_plan_bundle[i], data + running_total * extent, data + running_total * extent, dir))
#else
    CUDA_FFT_CALL(cufftExecZ2Z(stream_plan_bundle[i], data + running_total * extent, data + running_total * extent, dir))
#endif
    running_total += part_size;
  }
  cudaDeviceSynchronize();

} 

extern "C" void fftwav_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal){
  unsigned int i, wk_size, wr_size;

  if (!cufft_init) make_context(); 

  //fprintf(stderr, "Entering fftwav_cu \n");
  
  //if (*lreal)  fprintf(stderr, "lreal \n");

  wk_size = *npl; // size in compressed k-space
  wr_size = grid[0] * grid[1] * grid[2]; // size in real-space

#ifdef CUFFT_SP
  resize_cu((void_p*)wr_d, wr_size * sizeof(cuComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * sizeof(cuComplex));
  realloc_buffer(wr_size * sizeof(cuComplex));
  for (i = 0; i < wk_size * 2; i++)
    buffer[i] = (float) *(((double*)C) + i);
  assign_cu((void_p*)wk_d, buffer, wk_size * sizeof(cuComplex));
#else
  resize_cu((void_p*)wr_d, wr_size * sizeof(cuDoubleComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * sizeof(cuDoubleComplex));
  assign_cu((void_p*)wk_d, C, wk_size * sizeof(cuDoubleComplex)); 
#endif

  assign_cu((void_p*) index_d, ind, wk_size * sizeof(int)); 
  inflate_k<<<1, 512>>>(wk_d->ptr, wr_d->ptr, index_d->ptr, wk_size, wr_size, 1);

  stream_ffts(wr_d->ptr, grid, 1, 1);

#ifdef CUFFT_SP
  retrieve_cu(buffer, (void_p*)wr_d, wr_size * sizeof(cuComplex));
  if (*lreal)
    for (i = 0; i < wr_size; i++)
      *(((double*)CR) + i) = (double) buffer[2*i];
  else
    for (i = 0; i < wr_size * 2; i++)
      *(((double*)CR) + i) = (double) buffer[i];
#else
  retrieve_cu(CR, (void_p*)wr_d, wr_size * sizeof(cuDoubleComplex));
  if (*lreal)
    for (i = 0; i < wr_size; i++)
      *(((double*)CR) + i) = *(((double*)CR) + 2*i);
#endif

}

/*
//Similar to fftwav_cu_ but assumes the arrays already reside in GPU memory.
extern "C" void fftwav_cu_gpu_(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal){
  unsigned int i, wk_size, wr_size;

  if (!cufft_init) make_context(); 

  //fprintf(stderr, "Entering fftwav_cu \n");
  
  //if (*lreal)  fprintf(stderr, "lreal \n");

  wk_size = *npl; // size in compressed k-space
  wr_size = grid[0] * grid[1] * grid[2]; // size in real-space

#ifdef CUFFT_SP
  resize_cu((void_p*)wr_d, wr_size * sizeof(cuComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * sizeof(cuComplex));
  realloc_buffer_gpu(wr_size * sizeof(cuComplex));
  for (i = 0; i < wk_size * 2; i++)
    buffer[i] = (float) *(((double*)C) + i);
  assign_cu_gpu((void_p*)wk_d, C, wk_size * sizeof(cuComplex));
#else
  resize_cu((void_p*)wr_d, wr_size * sizeof(cuDoubleComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * sizeof(cuDoubleComplex));
  assign_cu_gpu((void_p*)wk_d, C, wk_size * sizeof(cuDoubleComplex));
#endif
  assign_cu((void_p*) index_d, ind, wk_size * sizeof(int)); 
  inflate_k<<<1, 512>>>(wk_d->ptr, wr_d->ptr, index_d->ptr, wk_size, wr_size, 1);

  stream_ffts(wr_d->ptr, grid, 1, 1);

#ifdef CUFFT_SP
  if (*lreal)
    for (i = 0; i < wr_size; i++)
      *(((double*)CR) + i) = (double) buffer[2*i];
  else
      *(((double*)CR) + i) = (double) buffer[i];
#else
  cudaMemcpy((void *)CR, (void *)wr_d, wr_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  if (*lreal)
    for (i = 0; i < wr_size; i++)
      *(((double*)CR) + i) = *(((double*)CR) + 2*i);
#endif
}
*/
/*
extern "C" void fftwav_w1_cu_(int *npl, int *nk, int *ind, 
                              cuDoubleComplex *CR, cuDoubleComplex *C, 
                              int *grid, int *lreal, int *ntrans){
  if (!cufft_init) make_context();

  unsigned int i, j, wk_size, wr_size;

  wk_size = *npl;
  wr_size = grid[0] * grid[1] * grid[2];

  if (*nk != index_array_id){
    assign_cu((void_p*) index_d, ind, wk_size * sizeof(int));
    index_array_id = *nk;
  }

#ifdef CUFFT_SP
  resize_cu((void_p*) wr_d, wr_size * (*ntrans) * sizeof(cuComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * (*ntrans) * sizeof(cuComplex));
  realloc_buffer(wr_size * (*ntrans) * sizeof(cuComplex));
  for (i = 0; i < wk_size * 2 * (*ntrans); i++)
    buffer[i] = (float) *(((double*)C) + i);
  assign_cu((void_p*) wk_d, buffer, wk_size * (*ntrans) * sizeof(cuComplex));
#else
  resize_cu((void_p*) wr_d, wr_size * (*ntrans) * sizeof(cuComplex));
  cudaMemset(wr_d->ptr, 0, wr_size * (*ntrans) * sizeof(cuDoubleComplex));
  assign_cu((void_p*) wk_d, C, wk_size * (*ntrans) * sizeof(cuDoubleComplex)); 
#endif
 
  inflate_k<<<1, 512>>>(wk_d->ptr, wr_d->ptr, index_d->ptr, wk_size, wr_size, *ntrans);
  stream_ffts(wr_d->ptr, grid, 1, *ntrans); 

  // Copy to device, Execute on device, Copy to host
#ifdef CUFFT_SP
  retrieve_cu(buffer, (void_p*)wr_d, wr_size * (*ntrans) * sizeof(cuComplex));

  if (*lreal)
    for (j = 0; j < *ntrans; j++)
      for (i = 0; i < wr_size; i++)
        *(((double*)CR) + i + j * 2 * wr_size) = (double) buffer[2*i + j * wr_size * 2];
  else
    for (i = 0; i < wr_size * 2 * *ntrans; i++)
      *(((double*)CR) + i) = (double) buffer[i];
#else
  retrieve_cu(CR, (void_p*) wr_d, wr_size * (*ntrans) * sizeof(cuDoubleComplex));
  if (*lreal)
    for (j = 0; j < *ntrans; j++)
      for (i = 0; i < wr_size; i++)
        *(((double*)CR) + i + j * 2 * wr_size) = *(((double*)CR) + 2*i + j * 2 * wr_size);

#endif
}
*/

extern "C" void fftext_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal, int* ladd){
  if (!cufft_init) make_context();

  int i, wk_size, wr_size;

  wk_size = *npl;
  wr_size = grid[0] * grid[1] * grid[2];

  assign_cu((void_p*) index_d, ind, wk_size * sizeof(int));

  if (*lreal){
    for (i = wr_size-1; i >= 0; i--){
      *(((double*)CR)+i*2) = *(((double*)CR)+i);
      *(((double*)CR)+i*2+1) = 0;      
    }
  }

#ifdef CUFFT_SP
  realloc_buffer(wr_size * sizeof(cuComplex));
  for (i = 0; i < wr_size * 2; i++)
    buffer[i] = (float) *(((double*)CR) + i);
  assign_cu((void_p*) wr_d, buffer, wr_size * sizeof(cuComplex));
  for (i = 0; i < wk_size * 2; i++)
    buffer[i] = (float) *(((double*)C) + i);
  assign_cu((void_p*)wk_d, buffer, wk_size * sizeof(cuComplex));
#else
  assign_cu((void_p*)wk_d, C,  wk_size * sizeof(cuDoubleComplex)); 
  assign_cu((void_p*)wr_d, CR, wr_size * sizeof(cuDoubleComplex));
#endif
 
  // Copy to device, Execute on device, Copy to host
  stream_ffts(wr_d->ptr, grid, -1, 1);
  resize_cu((void_p*) wk_d, wk_size * sizeof(cuComplex));
  deflate_k<<<1, 512>>>(wk_d->ptr, wr_d->ptr, index_d->ptr, wk_size, wr_size, 1, *ladd);

#ifdef CUFFT_SP
  retrieve_cu(buffer, (void_p*)wk_d, wk_size * sizeof(cuComplex));

  for (i = 0; i < wk_size * 2; i++)
    *(((double*)C) + i) = (double) buffer[i];
#else
  retrieve_cu(C, (void_p*) wk_d, wk_size * sizeof(cuDoubleComplex));
#endif

}

/*
//Similar to fftext_cu_, but already assumes that CR and C reside in GPU memory.
extern "C" void fftext_cu_gpu(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal, int* ladd){
  if (!cufft_init) make_context();

  int i, wk_size, wr_size;

  wk_size = *npl;
  wr_size = grid[0] * grid[1] * grid[2];

  assign_cu((void_p*) index_d, ind, wk_size * sizeof(int));

  if (*lreal){
    for (i = wr_size-1; i >= 0; i--){
      *(((double*)CR)+i*2) = *(((double*)CR)+i);
      *(((double*)CR)+i*2+1) = 0;      
    }
  }

#ifdef CUFFT_SP
  realloc_buffer_gpu(wr_size * sizeof(cuComplex));
  for (i = 0; i < wr_size * 2; i++)
    buffer[i] = (float) *(((double*)CR) + i);
  assign_cu_gpu((void_p*) wr_d, buffer, wr_size * sizeof(cuComplex));
  for (i = 0; i < wk_size * 2; i++)
    buffer[i] = (float) *(((double*)C) + i);
  assign_cu_gpu((void_p*)wk_d, buffer, wk_size * sizeof(cuComplex));
#else
  assign_cu_gpu((void_p*)wk_d, C,  wk_size * sizeof(cuDoubleComplex)); 
  assign_cu_gpu((void_p*)wr_d, CR, wr_size * sizeof(cuDoubleComplex));
#endif
 
  // Copy to device, Execute on device, Copy to host
  stream_ffts(wr_d->ptr, grid, -1, 1);
  resize_cu((void_p*) wk_d, wk_size * sizeof(cuComplex));
  deflate_k<<<1, 512>>>(wk_d->ptr, wr_d->ptr, index_d->ptr, wk_size, wr_size, 1, *ladd);

#ifdef CUFFT_SP
  retrieve_cu_gpu(buffer, (void_p*)wk_d, wk_size * sizeof(cuComplex));

  for (i = 0; i < wk_size * 2; i++)
    *(((double*)C) + i) = (double) buffer[i];
#else
  retrieve_cu_gpu(C, (void_p*) wk_d, wk_size * sizeof(cuDoubleComplex));
#endif

}
*/

/*
extern "C" void fftext_w1_cu_(int *npl, int *nk, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int * lreal, int* ladd){
  if (!cufft_init) make_context();

  int i, wk_size, wr_size;

  wk_size = *npl;
  wr_size = grid[0] * grid[1] * grid[2];

  if (*lreal){
    for (i = wr_size-1; i >= 0; i--){
      *(((double*)CR)+i*2) = *(((double*)CR)+i);
      *(((double*)CR)+i*2+1) = 0;      
    }
  }

  if (*nk != index_array_id){
    if (index_array_size < wk_size * sizeof(int)){
      if (index_array != NULL)
        cudaFree(index_array);
      cudaMalloc((void**)&index_array, wk_size * sizeof(int));
      index_array_size = wk_size * sizeof(int);
    }
    cudaMemcpy(index_array, ind, wk_size * sizeof(int), cudaMemcpyHostToDevice);
    index_array_id = *nk;
  }

  check_sizes(wk_size, grid, 1);

#ifdef CUFFT_SP
  for (i = 0; i < wr_size * 2; i++)
    buffer[i] = (float) *(((double*)CR) + i);
  cudaMemcpy(wr_d, buffer, wr_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
  for (i = 0; i < wk_size * 2; i++)
    buffer[i] = (float) *(((double*)C) + i);
  cudaMemcpy(wk_d, buffer, wk_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
#else
  cudaMemcpy(wk_d, C, wk_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); 
  cudaMemcpy(wr_d, CR, wr_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
#endif
 
  // Copy to device, Execute on device, Copy to host
#ifdef CUFFT_SP
  cufftExecC2C(plan_c2c, wr_d, wr_d, -1);
  deflate_k<<<1, 512>>>(wk_d, wr_d, index_array, wk_size, *ladd);

  cudaMemcpy(buffer, wk_d, wk_size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  for (i = 0; i < wk_size * 2; i++)
    *(((double*)C) + i) = (double) buffer[i];
#else
  cufftExecZ2Z(plan_c2c, wr_d, wr_d, -1);
  deflate_k<<<1, 512>>>(wk_d, wr_d, index_array, wk_size, *ladd);

  cudaMemcpy(C, wk_d, wk_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
#endif
}
*/

extern "C" void fft_3d_c2c_C(int *nx, int *ny, int *nz, 
                            cuDoubleComplex *a_h, cuDoubleComplex *b_h,
                            int *DIR){

  if (!cufft_init) make_context();

  // required memory size
#ifdef CUFFT_SP
  size_t size1 = sizeof(cuComplex)       * (*nx) * (*ny) * (*nz);
#else
  size_t size1 = sizeof(cuDoubleComplex) * (*nx) * (*ny) * (*nz);
#endif 
 
  CufftPlanBundle<1> &plan_c2c_bundle = plan_c2c_cache.getBundle(*nx, *ny, *nz);
  cufftHandle &plan_c2c = plan_c2c_bundle[0];
  if(! plan_c2c_bundle.isInitialized() )
  {
#ifdef CUFFT_SP
    CUDA_FFT_CALL(cufftPlan3d(&plan_c2c, *nz, *ny, *nx, CUFFT_C2C)); 
#else
    CUDA_FFT_CALL(cufftPlan3d(&plan_c2c, *nz, *ny, *nx, CUFFT_Z2Z));
#endif
    plan_c2c_bundle.setInitialized();
  }
  
  // Copy to device, Execute on device, Copy to host
#ifndef CUFFT_SP
  assign_cu((void_p*)ac2c_d, a_h, size1); 
  resize_cu((void_p*)bc2c_d, size1);
  CUDA_FFT_CALL(cufftExecZ2Z(plan_c2c, ac2c_d->ptr, bc2c_d->ptr, *DIR));
  retrieve_cu(b_h, (void_p*)bc2c_d, size1);
#else
  realloc_buffer(size1);  
  int i;
  for (i = 0; i < 2*(*nz)*(*ny)*(*nx); i++)
    buffer[i] = (float) *(((double*)a_h) + i);
  assign_cu((void_p*)ac2c_d, buffer, size1);
  resize_cu((void_p*)bc2c_d, size1);
  CUDA_FFT_CALL(cufftExecC2C(plan_c2c, ac2c_d->ptr, bc2c_d->ptr, *DIR));
  retrieve_cu(buffer, (void_p*) bc2c_d, size1);
  for (i = 0; i < 2*(*nz)*(*ny)*(*nx); i++)
    *(((double*)b_h) + i) = (double) buffer[i];
#endif
}

/* Want pinned memory so we don't use system realloc */
#ifdef CUFFT_SP

//Allocates buffer in GPU memory
int realloc_buffer_gpu(size_t size) {
   if(buf_size < size) {
      if(buffer != NULL) {
         if(BUFFER_GPU)
            cudaFree(buffer);
         else //resides in host mem
            cudaFreeHost(buffer);
      }
      //Allocate on GPU
      cudaMalloc((void**)&buffer, size);
      buf_size = size;
      BUFFER_GPU = 1;
   }
   return EXIT_SUCCESS;
}

int realloc_buffer(size_t size){
  if (buf_size < size){
    if (buffer != NULL) {
      if(BUFFER_GPU) 
         cudaFree(buffer);
      else  //Resides in host mem
         cudaFreeHost(buffer);
    }
    cudaMallocHost((void**)&buffer, size);
    buf_size = size;
    BUFFER_GPU = 0;
  }
  return EXIT_SUCCESS;
}
#endif


