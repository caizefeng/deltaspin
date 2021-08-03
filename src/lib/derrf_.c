/*****************************************************************
      FORTRAN interface fuer ERRF
*****************************************************************/

#include "math.h" 

double errf_C(x)
double *x;
{   
     return erf(*x);
} 
double errfc_C(x)
double *x;
{   
     return erfc(*x);
} 
