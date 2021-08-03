/** \file cuda_utils.cu
 * \brief Functions for manipulating persistent cuda pointers.
 *
 * Includes inline functions, so should be included to source directly.
 */
#ifndef _CUDA_UTILS_SRC_
#define _CUDA_UTILS_SRC_

#include <stdio.h>
#include <stdlib.h>
#include "cufft.h"
#include "cuda.h"
#include "cuda_utils.h"

#if 0
/** Memory copy from CPU to standard GPU pointer */
static inline void assign_cu_reg(void *dest,
                                 void *src, //CPU
                                 unsigned size) 
{      
//  fprintf(stdout, "Transferring %d bytes of memory to GPU, address %p\n", size, dest);
  CUDA_CALL(cudaMalloc(&dest, size));

  /* Do the mem copy */
  CUDA_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

/** Allocate and initialize a ndim_double ptr in GPU mem */
static inline ndim_double_ptr create_ndim_double(unsigned *dim, unsigned ndims) {

  ndim_double_ptr tmp;
  tmp = (ndim_double_ptr) malloc(sizeof(struct ndim_double));
  
  //init other fields
  tmp->size = 1; tmp->ndims = ndims;
  for(int i = 0; i < ndims; i++)
    tmp->size *= dim[i];

  //move dim to gpu mem
  unsigned *dim_gpu = NULL;
  assign_cu_reg(dim_gpu, dim, sizeof(int) * ndims);

  tmp->dim = dim_gpu;
  CUDA_CALL(cudaMalloc((void**)&tmp->ptr, sizeof(double) * tmp->size));
  tmp->start_pt = tmp->ptr;
  return tmp;
}

/** Allocate and initialize a ndim_complex ptr in GPU mem */
static inline ndim_complex_ptr create_ndim_complex(unsigned *dim, unsigned ndims) {

  ndim_complex_ptr tmp;
  tmp = (ndim_complex_ptr) malloc( sizeof(struct ndim_complex));
  
  //init other fields
  tmp->size = 1; tmp->ndims = ndims;
  for(int i = 0; i < ndims; i++)
    tmp->size *= dim[i];

  //move dim to gpu mem
  unsigned *dim_gpu = NULL;
  assign_cu_reg(dim_gpu, dim, sizeof(int) * ndims);

  tmp->dim = dim_gpu;
  CUDA_CALL(cudaMalloc((void**)&tmp->ptr, sizeof(cuDoubleComplex) * tmp->size));
  tmp->start_pt = tmp->ptr;
  return tmp;
}

/** Allocate and initialize a ndim_int ptr in GPU mem 
static inline ndim_int_ptr create_ndim_int(unsigned *dim, unsigned ndims) {

  ndim_int_ptr tmp;
  tmp = (ndim_int_ptr) malloc(sizeof(struct ndim_int));
  
  //init other fields
  tmp->size = 1; tmp->ndims = ndims;
  for(int i = 0; i < ndims; i++)
    tmp->size *= dim[i];
  
  //move dim to gpu mem
  unsigned *dim_gpu = NULL;
  assign_cu_reg(dim_gpu, dim, sizeof(int) * ndims);

  tmp->dim = dim_gpu;
  CUDA_CALL(cudaMalloc((void**)&tmp->ptr, sizeof(int) * tmp->size));
  tmp->start_pt = tmp->ptr;
  return tmp;
}
*/

static inline unsigned *create_dim_cpu(unsigned d1, 
                                   unsigned d2,
                                   unsigned d3,
                                   unsigned d4,
                                   unsigned d5,
                                   unsigned ndim) 
{
   unsigned *dim_cpu = (unsigned *) malloc(sizeof(int) * ndim);
   switch(ndim) {
      case 1:
         dim_cpu[0] = d1;
         break;   
      case 2:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         break;   
      case 3:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         break;   
      case 4:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         dim_cpu[3] = d4;
         break;   
     case 5:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         dim_cpu[3] = d4;
         dim_cpu[4] = d5;
         break;   
   }
   return dim_cpu;
}

static inline unsigned *create_dim_gpu(unsigned d1, 
                                   unsigned d2,
                                   unsigned d3,
                                   unsigned d4,
                                   unsigned d5,
                                   unsigned ndim) 
{
   unsigned *dim_gpu = NULL;
   unsigned *dim_cpu = (unsigned *) malloc(sizeof(int) * ndim);
   switch(ndim) {
      case 1:
         dim_cpu[0] = d1;
         break;   
      case 2:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         break;   
      case 3:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         break;   
      case 4:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         dim_cpu[3] = d4;
         break;   
     case 5:
         dim_cpu[0] = d1;
         dim_cpu[1] = d2;
         dim_cpu[2] = d3;
         dim_cpu[3] = d4;
         dim_cpu[4] = d5;
         break;   
   }
   assign_cu_reg((void*)dim_gpu, (void*)dim_cpu, sizeof(int) * ndim);
   return dim_gpu;
}

/** Set start point for array access */
static __global__ void ndim_set_ptr(int **ndim_ptr, 
                                    int *start_pt,
                                    unsigned *dim,
                                    unsigned *index, //Must be in GPU mem 
                                    unsigned ndim) 
{
  switch(ndim) {
     //One dimensional
     case 1:
        *ndim_ptr = &start_pt[index[0]];
        break;
     //Two dimensional
     case 2:
        *ndim_ptr = &start_pt[DIM2(index[0], index[1], dim[0])];
        break;
     //Three dimensional
     case 3:
        *ndim_ptr= 
        &start_pt[DIM3(index[0], index[1], index[2], dim[0], dim[1])];
        break;
     //Four dimensional
     case 4:
        *ndim_ptr = 
        &start_pt[DIM4(index[0], index[1], index[2], index[3], dim[0], dim[1], dim[2])];
        break;
     //Five dimensional (ridiculous)
     case 5:
        *ndim_ptr = 
        &start_pt[DIM5(index[0], index[1], index[2], index[3], index[4], 
        dim[0], dim[1], dim[2], dim[3])];
  }
}

/** Set start point for array access */
static __global__ void ndim_set_ptr(double **ndim_ptr, 
                                double *start_pt,
                                unsigned *dim,
                                unsigned *index,  
                                unsigned ndim) 
{
  switch(ndim) {
     //One dimensional
     case 1:
        *ndim_ptr = &start_pt[index[0]];
        break;
     //Two dimensional
     case 2:
        *ndim_ptr = &start_pt[DIM2(index[0], index[1], dim[0])];
        break;
     //Three dimensional
     case 3:
        *ndim_ptr= 
        &start_pt[DIM3(index[0], index[1], index[2], dim[0], dim[1])];
        break;
     //Four dimensional
     case 4:
        *ndim_ptr = 
        &start_pt[DIM4(index[0], index[1], index[2], index[3], dim[0], dim[1], dim[2])];
        break;
     //Five dimensional (ridiculous)
     case 5:
        *ndim_ptr = 
        &start_pt[DIM5(index[0], index[1], index[2], index[3], index[4], 
        dim[0], dim[1], dim[2], dim[3])];
  }
}

/** Set start point for array access */
static __global__ void ndim_set_ptr(cuDoubleComplex **ndim_ptr, 
                                cuDoubleComplex *start_pt,
                                unsigned *dim,
                                unsigned *index,  
                                unsigned ndim) 
{
  switch(ndim) {
     //One dimensional
     case 1:
        *ndim_ptr = &start_pt[index[0]];
        break;
     //Two dimensional
     case 2:
        *ndim_ptr = &start_pt[DIM2(index[0], index[1], dim[0])];
        break;
     //Three dimensional
     case 3:
        *ndim_ptr= 
        &start_pt[DIM3(index[0], index[1], index[2], dim[0], dim[1])];
        break;
     //Four dimensional
     case 4:
        *ndim_ptr = 
        &start_pt[DIM4(index[0], index[1], index[2], index[3], dim[0], dim[1], dim[2])];
        break;
     //Five dimensional (ridiculous)
     case 5:
        *ndim_ptr = 
        &start_pt[DIM5(index[0], index[1], index[2], index[3], index[4], 
        dim[0], dim[1], dim[2], dim[3])];
  }
}


static __global__ void set_2d_ptr(double *start_pt, double **to_set, int dim1, int dim2, int size_dim1) 
{
  *to_set = &(start_pt[DIM2(dim1, dim2, size_dim1)]);
}

static __global__ void set_2d_ptr(cuDoubleComplex *start_pt, cuDoubleComplex **to_set, 
                                                   int dim1, int dim2, int size_dim1) 
{
  *to_set = &(start_pt[DIM2(dim1, dim2, size_dim1)]);
}

static __global__ void set_ptr(cuDoubleComplex **to_set, cuDoubleComplex *start_pt, int dim) {
  *to_set = &(start_pt[dim]);
}


/** Set pointer back to start point */
static __global__ void ndim_set_to_start(int **ndim_ptr, int *ndim_start ) {
   *ndim_ptr = ndim_start;
}

static __global__ void ndim_set_to_start(cuDoubleComplex **ndim_ptr, cuDoubleComplex *ndim_start) {
   *ndim_ptr = ndim_start;
}

static __global__ void ndim_set_to_start(double **ndim_ptr, double *ndim_start) {
   *ndim_ptr = ndim_start;
}


//Free allocation
static inline void ndim_free(ndim_double_ptr ndim_ptr) {
  CUDA_CALL(cudaFree(ndim_ptr->start_pt));
  CUDA_CALL(cudaFree(ndim_ptr->dim));
  free(ndim_ptr);
}

static inline void ndim_free(ndim_complex_ptr ndim_ptr) {
  CUDA_CALL(cudaFree(ndim_ptr->start_pt));
  CUDA_CALL(cudaFree(ndim_ptr->dim));
  free(ndim_ptr);
}

static inline void ndim_free(ndim_int_ptr ndim_ptr) {
  CUDA_CALL(cudaFree(ndim_ptr->start_pt));
  CUDA_CALL(cudaFree(ndim_ptr->dim));
  free(ndim_ptr);
}


#endif

/** Allocate and initialize a persistent cuda pointer */
static inline void_p* create_cu(void){
  void_p* tmp = (void_p*) malloc(sizeof(void_p));
  tmp->ptr = NULL;
  tmp->size = 0;
  return tmp;
}

#if 0
/** Initialize a persistent cuda pointer.  Requires previous alloc */
static inline void init_cu(void_p* var){
  var->ptr = NULL;
  var->size = 0;
}
#endif

/** Assign a chunk of GPU memory to a chunk of CPU memory */
static inline void assign_cu(void_p* dest, //!<     destination (GPU)
                             void* src, //!<        source (CPU)
                             unsigned int size //<! size (in bytes)
                            ){
  /* Do we need to resize ? */
  if (dest->ptr == NULL || dest->size < size){
    if (dest->ptr != NULL)
      CUDA_CALL(cudaFree(dest->ptr));
    CUDA_CALL(cudaMalloc((void**)&dest->ptr, size));
    dest->size = size;
  }
  /* Do the actual copy */
  CUDA_CALL(cudaMemcpy(dest->ptr, src, size, cudaMemcpyHostToDevice));
}

/** Memory copy from GPU to GPU */
static inline void assign_cu_gpu(void_p* dest,
                                 void *src,  //GPU
                                 unsigned size)
{
  /* Need to resize */
  if(dest->ptr == NULL || dest->size < size) {
    if(dest->ptr != NULL) 
       CUDA_CALL(cudaFree(dest->ptr));
    CUDA_CALL(cudaMalloc((void**)&dest->ptr, size));
    dest->size = size;  
  }
  CUDA_CALL(cudaMemcpy(dest->ptr, src, size, cudaMemcpyDeviceToDevice));
}

#if 0
/** Memory copy from CPU to ndim GPU struct */
static inline void assign_cu_ndim(ndim_complex_ptr dest, //ndim struct pointer
                                  void *src)  //CPU
                                  
{
  if(src != NULL) 
    CUDA_CALL(cudaMemcpy((void*)dest->ptr, src, dest->size, cudaMemcpyHostToDevice));
}

static inline void assign_cu_ndim(ndim_double_ptr dest, //ndim struct pointer
                                  void *src)  //CPU
{
  if(src != NULL)
    CUDA_CALL(cudaMemcpy((void*)dest->ptr, src, dest->size, cudaMemcpyHostToDevice));
}

static inline void assign_cu_ndim(ndim_int_ptr dest, //ndim struct pointer
                                  void *src)  //CPU
{
  if(src != NULL)
    CUDA_CALL(cudaMemcpy((void*)dest->ptr, src, dest->size, cudaMemcpyHostToDevice));
}
#endif

/** Retrieve the value of a chunk of GPU memory to the CPU */
static inline void retrieve_cu(void* dest, //!< destination (CPU)
                               void_p* src, //!< source (GPU)
                               unsigned int size //!< size (in bytes)
                              ){
  /* Just copy */
  CUDA_CALL(cudaMemcpy(dest, src->ptr, size, cudaMemcpyDeviceToHost));
}

/** Retrieve the value of a chunk of GPU memory to the GPU */
static inline void retrieve_cu_gpu(void* dest, //!< destination (GPU)
                               void_p* src, //!< source (GPU)
                               unsigned int size //!< size (in bytes)
                              ){
  /* Just copy */
  CUDA_CALL(cudaMemcpy(dest, src->ptr, size, cudaMemcpyDeviceToDevice));
}

#if 0
/** Retrieve value of GPU ndim struct to CPU */
static inline void retrieve_cu_ndim(void *dest, //CPU
                                    ndim_complex_ptr src) //GPU
{
  /* Memcpy */
  CUDA_CALL(cudaMemcpy(dest, src->start_pt, src->size, cudaMemcpyDeviceToHost));

  /* free ndim ptr */
  ndim_free(src);
}

static inline void retrieve_cu_ndim(void *dest, //CPU
                                    ndim_double_ptr src) //GPU
{
  /* Memcpy */
  CUDA_CALL(cudaMemcpy(dest, src->start_pt, src->size, cudaMemcpyDeviceToHost));

  /* free ndim ptr */
  ndim_free(src);
}

static inline void retrieve_cu_ndim(void *dest, //CPU
                                    ndim_int_ptr src) //GPU
{
  /* Memcpy */
  CUDA_CALL(cudaMemcpy(dest, src->start_pt, src->size, cudaMemcpyDeviceToHost));

  /* free ndim ptr */
  ndim_free(src);
}

static inline void retrieve_cu_reg(void *dest,  //CPU
                                  void *src,   //GPU
                                  unsigned int size)
{
  /* Just copy */
  if(src != NULL) {
    CUDA_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(src));
  }
}
#endif

/** Resize GPU allocation */
static inline void resize_cu(void_p* var, unsigned int size){
  /* Do we actually have to resize? */
  if (var->ptr == NULL || var->size < size){
    if (var->ptr != NULL)
	  CUDA_CALL(cudaFree(var->ptr));
	CUDA_CALL(cudaMalloc((void**)&var->ptr, size));
	var->size = size;
  }
}

/** Free GPU memory allocation, set size to zero */
static inline void free_cu(void_p* var){
  CUDA_CALL(cudaFree(var->ptr));
  var->size = 0;
}

#endif
