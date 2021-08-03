/*****************************************************************
      FORTRAN interface fuer ERRF
      for CRAY t3d  in  this case we return double precission
      (i.e. 8 digits) but in contrast to other systems
      CRAY requires capital letters
      God how I hate this small details
*****************************************************************/

#include "math.h" 

double ERRF(x)
double *x;
{   
     return erf(*x);
} 
double ERRFC(x)
double *x;
{   
     return erfc(*x);
} 
