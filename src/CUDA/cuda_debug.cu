// File: cuda_helpers.cu
// C/Fortran interface to debugging.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes linux headers
#include <sys/stat.h>
// includes cuda headers
#include <cuda_runtime.h>
// includes project headers
#include "cuda_globals.h"

/******************************************************/
// C wrappers for printing, in VASP

// prints device array da to file
extern "C"
void cuda_print_stdout_(char *type, int *size, void **da)
{
    int i,n;
    size_t cusize;
    void *a;
    char ic = *type;

    // get data size
    n=*size;
    if(ic=='i')
        cusize=n*sizeof(int);
    else if(ic=='f')
        cusize=n*sizeof(float);
    else if(ic=='d')
        cusize=n*sizeof(double);
    else if(ic=='c')
        cusize=2*n*sizeof(double);
    else
	ERROR( "GPU Library", "Invalid type passed to cuda_print_stdout_, Argument 1 must be i, f, d, or c" );

    // allocate host array
    a = (void*)malloc(cusize);

    // synchronize the device
    CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device in cuda_print!" );

    // copy from device to host
    CUDA_ERROR( cudaMemcpy(a,*da,cusize,cudaMemcpyDeviceToHost),
		"Failed to copy from host to device!" );

    // print to stdout 
    if(ic=='i')
    {
        for(i=0;i<n;i++)
        printf("%d\t%d\n",i,((int*)a)[i]);
    }
    else if(ic=='f')
    {
        for(i=0;i<n;i++)
        printf("%d\t%e\n",i,((float*)a)[i]);
    }
    else if(ic=='d')
    {
        for(i=0;i<n;i++)
        printf("%d\t%e\n",i,((double*)a)[i]);
    }
    else if(ic=='c')
    {
        for(i=0;i<n;i++)
        printf("%d\t(%e,%e)\n",i,((double*)a)[2*i],((double*)a)[2*i+1]);
    }
    else
	ERROR( "GPU Library", "Invalid type passed to cuda_print_stdout, Argument 1 must be i, f, d, or c" );
// free host array
free(a);
}
extern "C"
void cuda_print_(char *filename, char *type, int *size, void **da)
{
    FILE *fp;
    int i,n;
    size_t cusize;
    void *a;
    char ic = *type;

    // get data size
    n=*size;
    if(ic=='i')
        cusize=n*sizeof(int);
    else if(ic=='f')
        cusize=n*sizeof(float);
    else if(ic=='d')
        cusize=n*sizeof(double);
    else if(ic=='c')
        cusize=2*n*sizeof(double);
    else
	ERROR( "GPU Library", "Invalid type passed to cuda_print, Argument 2 must be i, f, d, or c" );

    // allocate host array
    a = (void*)malloc(cusize);

    // synchronize the device
    CUDA_ERROR( cudaDeviceSynchronize(), "Failed to synchronize the device in cuda_print!" );

    // copy from device to host
    CUDA_ERROR( cudaMemcpy(a,*da,cusize,cudaMemcpyDeviceToHost),
		"Failed to copy from host to device!" );

    // print to file
    fp=fopen(filename,"w");
    if(ic=='i')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%d\n",i,((int*)a)[i]);
    }
    else if(ic=='f')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%e\n",i,((float*)a)[i]);
    }
    else if(ic=='d')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%e\n",i,((double*)a)[i]);
    }
    else if(ic=='c')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t(%e,%e)\n",i,((double*)a)[2*i],((double*)a)[2*i+1]);
    }
    else
	ERROR( "GPU Library", "Invalid type passed to cuda_print, Argument 2 must be i, f, d, or c" );
fclose(fp);

// free host array
free(a);
}

// prints host array a to file
extern "C"
void fortran_print_(char *filename, char *type, int *size, void *a)
{
    FILE *fp;
    int i,n;
    char ic = *type;

    // get data size
    n=*size;

    // print to file
    fp=fopen(filename,"w");
    if(ic=='i')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%d\n",i,((int*)a)[i]);
    }
    else if(ic=='f')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%e\n",i,((float*)a)[i]);
    }
    else if(ic=='d')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t%e\n",i,((double*)a)[i]);
    }
    else if(ic=='c')
    {
        for(i=0;i<n;i++)
        fprintf(fp,"%d\t(%e,%e)\n",i,((double*)a)[2*i],((double*)a)[2*i+1]);
    }
    else
	ERROR( "GPU Library", "Invalid type passed to cuda_print, Argument 2 must be i, f, d, or c" );
fclose(fp);
}

// prints device array da to file, MPI version
extern "C"
void cuda_prints_(int *rankid, int *i, char *basename, char *type, int *size, void **da)
{
    char filename[256];
    sprintf(filename,"%s%d_rank%d.dat",basename,*i,*rankid);
    cuda_print_(filename,type,size,da);
}

// prints host array a to file, MPI version
extern "C"
void fortran_prints_(int *rankid, int *i, char *basename, char *type, int *size, double *a)
{
    char filename[256];
    sprintf(filename,"%s%d_rank%d.dat",basename,*i,*rankid);
    fortran_print_(filename,type,size,a);
}

/******************************************************/
