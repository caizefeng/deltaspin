/*****************************************************************
      FORTRAN interface fuer ERRF
      for CRAY mainly in  this case we return single precission
      (i.e. 8 digits)
*****************************************************************/

#include "math.h" 

float ERRF(x)
double *x;
{   
     return erf(*x);
} 
float ERRFC(x)
double *x;
{   
     return erfc(*x);
} 
