/*****************************************************************
      FORTRAN interface to get user-time
      should work on almost all UNIX systems
*****************************************************************/

#include <time.h>
#include <sys/time.h>
#include <stdio.h>

void vtime(vputim,cputim)
double  *vputim,*cputim;
{

	long lclock;
	lclock=clock();
	*vputim=((double)lclock)/CLOCKS_PER_SEC;
	*cputim=*vputim;
}
