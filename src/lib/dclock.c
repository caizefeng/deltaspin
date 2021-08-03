/*****************************************************************
      FORTRAN interface to get user-time and real time on
      AIX-system might work on other UNIX systems as well
      (must work on all BSDish systems and SYSVish systems
      with BSD-compatibility timing routines ...)!?
*****************************************************************/

#ifndef NULL
#define NULL    ((void *)0)
#endif

#include <sys/times.h>
#include <sys/resource.h>

void vtime(vputim,cputim)
double  *vputim,*cputim;
{
        int    gettimeofday();
        int    getrusage();
        struct rusage ppt;
        struct rusage cpt;
        struct timeval tpu,tps;
        struct timeval tcu,tcs;
        struct timeval now;
        int    ierr;
        ierr = getrusage(RUSAGE_SELF,&ppt);
        tpu  = ppt.ru_utime;
        tps  = ppt.ru_stime;
        ierr = getrusage(RUSAGE_CHILDREN,&cpt);
        tcu  = cpt.ru_utime;
        tcs  = cpt.ru_stime;
        ierr = gettimeofday(&now,NULL);
        *vputim=((double) tpu.tv_sec) + ((double) tpu.tv_usec) / 1e6 +
                ((double) tps.tv_sec) + ((double) tps.tv_usec) / 1e6 +
                ((double) tcu.tv_sec) + ((double) tcu.tv_usec) / 1e6 +
                ((double) tcs.tv_sec) + ((double) tcs.tv_usec) / 1e6;
        *cputim=((double) now.tv_sec) + ((double) now.tv_usec) / 1e6;
}
