/* provide some timing and resource usage data (CPU, wall clock, memory, ...) */

#ifndef NULL
#define NULL    ((void *)0)
#endif

#include <sys/times.h>
#include <sys/resource.h>

void timing_(mode,utime,stime,now,minpgf,majpgf,maxrsize,avsize,swaps,ios,cswitch,ierr)

double *utime,*stime,*now,*maxrsize,*avsize;
int    *mode,*minpgf,*majpgf,*swaps,*ios,*cswitch,*ierr;

{
   struct rusage  rudata;
   struct timeval usertime;
   struct timeval systime;
   struct timeval realtime;
   int            getrusage();
   int            gettimeofday();
   int            dumerr;
   double         intsize,totaltime;

   if ( *mode == 0 ) {*ierr = getrusage(RUSAGE_SELF,&rudata);};
   if ( *mode != 0 ) {*ierr = getrusage(RUSAGE_CHILDREN,&rudata);};
   usertime  = rudata.ru_utime;
   systime   = rudata.ru_stime;

   *utime    = ((double) usertime.tv_sec) + ((double) usertime.tv_usec) / 1000000. / 16. ;
   *stime    = ((double) systime.tv_sec ) + ((double) systime.tv_usec ) / 1000000. / 16. ;

   dumerr    = gettimeofday(&realtime,NULL);
   *now      = ((double) realtime.tv_sec) + ((double) realtime.tv_usec) / 1000000;

   totaltime = *utime + *stime;

   *minpgf   = (int) rudata.ru_minflt;
   *majpgf   = (int) rudata.ru_majflt;

   intsize   = ((double) rudata.ru_ixrss) + ((double) rudata.ru_idrss) + ((double) rudata.ru_isrss);

   *maxrsize = (double) rudata.ru_maxrss;
   *avsize   = intsize / totaltime / 100 ;

   *swaps    = (int) rudata.ru_nswap;
   *ios      = ((int) rudata.ru_inblock) + ((int) rudata.ru_oublock);
   *cswitch  = (int) rudata.ru_nvcsw;
}
