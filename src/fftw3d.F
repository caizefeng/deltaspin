#define use_r2c_fftw
#include "fft3dsimple.F"
#ifndef MINLOOP
#define MINLOOP 1
#endif
#ifndef CACHE_SIZE

#if defined vector
#define CACHE_SIZE 0
#else
#define CACHE_SIZE 32768
#endif

#endif
!*************************************************************************
!*        Fast Fourier Transform for VASP using the FFTW package.        *
!*                                                                       *
!* This routine is just to give a good interface between VASP and FFTW.  *
!* FFTW3D gives an identical interface to the FFTBAS routine, apart from *
!* that the complex array is placed last instead of first. This is just  *
!* to prevent alignment errors. I.e. if the c-array would be too large   *
!* it is simply cut off with this arrangement (the other way around might*
!* cause trouble on the Macintosh.)                                      *
!* The Fortran calls simply makes use of the FFTW library written in     *
!* C/C++ (see also http://www.fftw.org). The result is considerably      *
!* faster then the fft3dfurth/fftbas system for all array sizes I have   *
!* tested. For small arrays (n<32) you get an improvement of several     *
!* 100%, and for larger you still get an improvement in the range of 50%.*
!* Those who like to use the FFTW should get the package from the url    *
!* above, compile it and link with the library. For the Mac the Absoft   *
!* C/C++ compiler makes a good job even if you have to create your own   *
!* your own makefile.                                                    *
!* It should also be noted that FFTW is just as conservative about mem-  *
!* ory use as FFTBAS, possibly even more so.                             *
!*                                                                       *
!*                                   Ph.D. Stefan Mankefors Mars 3 2000  *
!*                                         sem@fy.chalmers.se            *
!*                                                                       *
!* NOTE: When you compile the library you have to configure the Fortran  *
!* conventions used, i.e. 'upper case', 'lowercase' etc. This is done at *
!* the end of the 'config.h' file where you define *one* of the the conv-*
!* ventions used for Fortran. ('config.h' comes with all F77 conventions *
!* undefined.) Please note that this might be machine dependent.         *
!*                                                                       *
!* NOTE2: The real to complex FFT might also be exchanged in a similar   *
!* way. I have not tested this, however, and the gain in time is very    *
!* slim since this type of FFT is used very little by the VASP - at least*
!* as far as the benchmark runs goes (I do not have the experience yet to*
!* claim anything else.) Hence it is a question of an additional library *
!* against a gain of perhaps 1%. I have choosen not to use FFTW for this.*
!* Please observe that this means that fft3dlib still is needed in the   *
!* make process of VASP.                                                 *
!*************************************************************************

!
      subroutine FFTBAS(c,grid,isign)

      use prec
      use mgrid

      implicit none

      include 'fftw3.f'

      TYPE (grid_3d) grid

      Complex(q) c(*), cdummy
      integer isign
      integer(8) :: plan

!$!=======================================================================
!$!  initialise openMP FFT, has to be done here and not in main.F in
!$!  in general, since the FFTs are called outside and inside openMP
!$!  parallel regions.
!$!=======================================================================
       if (isign.le.0) then
        call dfftw_plan_dft_3d(plan,grid%ngx,grid%ngy,grid%ngz,c,c,FFTW_FORWARD,FFTW_ESTIMATE)
       else
        call dfftw_plan_dft_3d(plan,grid%ngx,grid%ngy,grid%ngz,c,c,FFTW_BACKWARD,FFTW_ESTIMATE)
       endif
       call dfftw_execute_dft(plan,c,c)
       call dfftw_destroy_plan(plan)
       return
       end subroutine


!
      SUBROUTINE FFTMAKEPLAN(c,grid)

      USE prec
      USE mgrid
      USE openmp, ONLY : omp_dfftw_init_threads

      IMPLICIT NONE

      include 'fftw3.f'

      TYPE (grid_3d) grid
      INTEGER(8) :: plan
      COMPLEX(q) c(*), cdummy
      INTEGER nx,ny,nz



      nx=grid%ngx
      ny=grid%ngy
      nz=grid%ngz

      IF (.NOT.grid%real2cplx) THEN
         CALL dfftw_plan_dft_3d(plan,nx,ny,nz, &
              c, c, & 
              FFTW_FORWARD, FFTW_MEASURE)
         CALL dfftw_destroy_plan(plan)
         CALL dfftw_plan_dft_3d(plan,nx,ny,nz,&
              c, c, & 
              FFTW_BACKWARD, FFTW_MEASURE)
         CALL dfftw_destroy_plan(plan)
      ELSE
         CALL dfftw_plan_dft_r2c_3d(plan,nx,ny,nz,&
              c,c, &
              FFTW_MEASURE)
         CALL dfftw_destroy_plan(plan)
         CALL dfftw_plan_dft_c2r_3d(plan,nx,ny,nz,&
              c,c, &
              FFTW_MEASURE)
         CALL dfftw_destroy_plan(plan)
      ENDIF

      RETURN
      END SUBROUTINE FFTMAKEPLAN


!
#ifdef use_r2c_fftw
      SUBROUTINE FFTBRC(C,grid,ISIGN)
      USE prec
      USE mgrid
      IMPLICIT NONE

      include 'fftw3.f'
      TYPE (grid_3d) grid
      COMPLEX(q) C(*)
      INTEGER ISIGN
      ! local variables
      INTEGER(8) plan

!$!=======================================================================
!$!  initialise openMP FFT, has to be done here and not in main.F in
!$!  in general, since the FFTs are called outside and inside openMP
!$!  parallel regions.
!$!=======================================================================
      IF (ISIGN.LE.0) THEN
         CALL dfftw_plan_dft_r2c_3d(plan,grid%ngx,grid%ngy,grid%ngz,C,C,FFTW_ESTIMATE)
      ELSE
         CALL dfftw_plan_dft_c2r_3d(plan,grid%ngx,grid%ngy,grid%ngz,C,C,FFTW_ESTIMATE)
      ENDIF
      IF (ISIGN.LE.0) THEN
         CALL dfftw_execute_dft_r2c(plan,C,C)
      ELSE
         CALL dfftw_execute_dft_c2r(plan,C,C)
      ENDIF
      CALL dfftw_destroy_plan(plan)
      RETURN
      END SUBROUTINE FFTBRC
#endif

!=======================================================================
!   this routine returns the next correct setting for the
!   three dimensional FFT
!=======================================================================

      SUBROUTINE FFTCHK(NFFT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION NFFT(3)
      LOGICAL FFTCH1

      DO 100 IND=1,3
  200 CONTINUE
        IF (FFTCH1(NFFT(IND))) GOTO 100
        NFFT(IND)=NFFT(IND)+1
        GOTO 200
  100 CONTINUE
      END

      LOGICAL FUNCTION FFTCH1(NIN)
      USE prec
      LOGICAL, EXTERNAL :: FFTCHK_FURTH

      FFTCH1=FFTCHK_FURTH(NIN)
      END FUNCTION

      MODULE fft_private
      USE prec
      REAL(q),POINTER,SAVE ::  WORK(:)
      END MODULE

#ifndef use_r2c_fftw
!=======================================================================
!   generic   3-d fast fourier transformation
!   written by Juergen Furthmueller
!   performes the 3-d real to complex FFT
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)    complex to real
!     -1  r->q   vq= sum(r) vr exp(-iqr)    real to complex
!
!=======================================================================
      SUBROUTINE FFTBRC(A,grid,ISIGN)
      USE prec
      USE mgrid
      USE fft_private
      USE smart_allocate
      IMPLICIT REAL(q) (A-H,O-Z)
      TYPE (grid_3d) grid

      COMPLEX(q) A(*)
      INTEGER, PARAMETER :: NMAXM=4096 ! propably sufficient forever
      DIMENSION TRIGX(2*NMAXM),TRIGY(2*NMAXM),TRIGZ(2*NMAXM)
      DIMENSION IFAC(19,3)
      SAVE TRIGX,TRIGY,TRIGZ,IFAC,NXO,NYO,NZO
      DATA NXO /0/, NYO /0/, NZO /0/

      NX=grid%ngx
      NY=grid%ngy
      NZ=grid%ngz
      NMAX=MAX(NX,NY)
      NMAX=MAX(NMAX,NZ)

#if CACHE_SIZE==0
#if defined vector
      NALLOC=4*NX*NY*NZ
#else
      NALLOC=4*NX*MAX(NY,NZ)
#endif
#else
      NALLOC=MAX(CACHE_SIZE, 4*MINLOOP*MAX(NX,NY,NZ))
#endif
      CALL SMART_ALLOCATE_REAL(WORK,NALLOC)

      IF (NX>NMAXM) THEN
         WRITE(*,*) ' FFT3DFURTH: Increase NMAXM to ',NMAX
         STOP
      ENDIF
      IF (NY>NMAXM) THEN
         WRITE(*,*) ' FFT3DFURTH: Increase NMAXM to ',NMAX
         STOP
      ENDIF
      IF (NZ>NMAXM) THEN
         WRITE(*,*) ' FFT3DFURTH: Increase NMAXM to ',NMAX
         STOP
      ENDIF
! Initialize FFT if necessary (changes of mesh size, first call)!
      IF ((NX/=NXO).OR.(NY/=NYO).OR.(NZ/=NZO)) THEN
#if defined vector
      IF ((CACHE_SIZE==0).OR.(CACHE_SIZE>=4*NX*NY*NZ)) THEN
         CALL FFTR3V(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,0,IERR)
      ELSE
         CALL FFTCRN(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,0,IERR,CACHE_SIZE)
      ENDIF
#else
         CALL FFTCRN(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,0,IERR,CACHE_SIZE)
#endif
         IF (IERR/=0) THEN
            WRITE(*,*) 'INIT FFT3D: IERR =',IERR
            STOP
         ENDIF
! Remember last mesh sizes!
         NXO=NX
         NYO=NY
         NZO=NZ
      END IF
! Do the transformation!
#if defined vector
      IF ((CACHE_SIZE==0).OR.(CACHE_SIZE>=4*NX*NY*NZ)) THEN
      CALL FFTR3V(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,-ISIGN,IERR)
      ELSE
      CALL FFTCRN(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,-ISIGN,IERR,CACHE_SIZE)
      ENDIF
#else
      CALL FFTCRN(A,NX+2,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,-ISIGN,IERR,CACHE_SIZE)
#endif
      IF (IERR/=0) THEN
         WRITE(*,*) 'FFT3D: IERR =',IERR
         STOP
      ENDIF

      RETURN
      END
#endif
