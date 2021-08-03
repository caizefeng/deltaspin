/*****************************************************************
      FORTRAN interface fuer ERRF
*****************************************************************/

extern double erf(double);
extern double erfc(double);

double errf(x)
double *x;
{
     return erf(*x);
}

double errfc(x)
double *x;
{
     return erfc(*x);
}
