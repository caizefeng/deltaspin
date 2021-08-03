#include "fft3dsimple_gpu.F"


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

#ifdef fftw2
       subroutine FFTBAS2(c,grid,isign)

       use prec

       implicit none

       include 'fftw_f77.i'

       Complex(q) c(*), cdummy
       integer grid(3), isign
       integer(8) :: plan
       integer i,j,k, idummy

       if (isign.le.0) then
         call fftw3d_f77_create_plan(plan,grid(1),grid(2),grid(3),&
     & FFTW_FORWARD,FFTW_IN_PLACE)
       else
         call fftw3d_f77_create_plan(plan,grid(1),grid(2),grid(3),&
     & FFTW_BACKWARD,FFTW_IN_PLACE)
       endif

       call fftwnd_f77_one(plan,c,cdummy)
       call fftwnd_f77_destroy_plan(plan)
       return

       end
#else
       subroutine FFTBAS(c,grid,isign)

       use prec
#ifdef CUFFT_MIN
       use cuda_interface
#endif

       implicit none

       include 'fftw3.f'

       Complex(q) c(*), cdummy
       integer grid(3), isign
       integer(8) :: plan
       integer i,j,k, idummy

#ifdef CUFFT_MIN
       if (grid(1) .ge. CUFFT_MIN) then
         ! Simply call the wrappers found in cuda_wrappers.cu
         if (isign .le.0) then
           call fft_3d_c2c(grid(1),grid(2),grid(3), c, c, -1)
         else
           call fft_3d_c2c(grid(1),grid(2),grid(3), c, c, +1)
         endif
       else
#endif
       if (isign.le.0) then
        call dfftw_plan_dft_3d(plan,grid(1),grid(2),grid(3),&
                           c, c, & 
                           FFTW_FORWARD, FFTW_ESTIMATE)
       else
        call dfftw_plan_dft_3d(plan,grid(1),grid(2),grid(3),&
                           c, c, & 
                           FFTW_BACKWARD, FFTW_ESTIMATE)
       endif
  
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)

#ifdef CUFFT_MIN
       endif
#endif

       return

       end subroutine
#endif

! Added by Arash
!#########################################################
!#                                                       #
!#        FFTWAVBAS    transferts CPU<->GPU              #
!#                                                       #
!#########################################################

       subroutine FFTMAKEPLAN(c,grid)

       use prec

       implicit none

       include 'fftw3.f'

       integer grid(3)
       integer(8) :: plan
       Complex(q) c(*), cdummy

       call dfftw_plan_dft_3d(plan,grid(1),grid(2),grid(3),&
                           c, c, & 
                           FFTW_FORWARD, FFTW_MEASURE)
       call dfftw_destroy_plan(plan)
       call dfftw_plan_dft_3d(plan,grid(1),grid(2),grid(3),&
                           c, c, & 
                           FFTW_BACKWARD, FFTW_MEASURE)
       call dfftw_destroy_plan(plan)
       end subroutine


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


!=======================================================================
!   generic   3-d fast fourier transformation
!   written by Juergen Furthmueller
!   performes the 3-d real to complex FFT
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)    complex to real
!     -1  r->q   vq= sum(r) vr exp(-iqr)    real to complex
!
!=======================================================================
      SUBROUTINE FFTBRC(A,MF,ISIGN)
      USE prec
      USE fft_private
      USE smart_allocate
      IMPLICIT REAL(q) (A-H,O-Z)

      COMPLEX(q) A(*)
      INTEGER, PARAMETER :: NMAXM=4096 ! propably sufficient forever
      DIMENSION TRIGX(2*NMAXM),TRIGY(2*NMAXM),TRIGZ(2*NMAXM)
      DIMENSION IFAC(19,3),MF(3)
      SAVE TRIGX,TRIGY,TRIGZ,IFAC,NXO,NYO,NZO
      DATA NXO /0/, NYO /0/, NZO /0/

      NX=MF(1)
      NY=MF(2)
      NZ=MF(3)
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

!||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! USE FFTW (not works)
      SUBROUTINE FFTBRC_W(CR,SR,grid,ISN)
      USE prec
      USE mgrid

      Complex(q) CR(*)
      Real(q)    SR(*)
      INTEGER NX,NY,NZ
      integer grid(3)
      SCALE=1.0_q

      NX=grid(1)
      NY=grid(2)
      NZ=grid(3)
      
      !if(ISN.EQ.(-1)) THEN
      !  do ii=1,NX*NY*NZ
      !  SR(ii)=real(CR(ii))
      !  enddo
      !ENDIF  
       if (isign.le.0) then !R2C
         call dfftw_plan_dft_r2c_3d(plan,grid(1),grid(2),grid(3),&
                           SR, CR,FFTW_ESTIMATE)
         call dfftw_execute_dft_r2c(plan,SR,CR)
       else
         call dfftw_plan_dft_c2r_3d(plan,grid(1),grid(2),grid(3),&
                           CR, SR,FFTW_ESTIMATE)
         call dfftw_execute_dft_c2r(plan,CR,SR)
       endif
  
       call dfftw_destroy_plan(plan)

      RETURN
      END
      
      
      SUBROUTINE FFTBRC_WW(CR,grid,ISN)
      USE prec
      USE mgrid

      Complex(q) CR(*)
      INTEGER NX,NY,NZ
      integer grid(3)
      Real(q) SR(grid(1)*grid(2)*grid(3))
      SCALE=1.0_q

      NX=grid(1)
      NY=grid(2)
      NZ=grid(3)
      
      if(ISN.EQ.(1)) THEN
        write(*,*) ISN
        do ii=1,NX*NY*NZ
        write(*,*) CR(ii)
        enddo
      ENDIF
      
      CALL FFTBRC_W(CR,SR,grid,ISN)
      
      if(ISN.EQ.(1)) THEN
        write(*,*) "##################"
        do ii=1,NX*NY*NZ
        write(*,*) CR(ii)
        enddo
        call exit()
      ENDIF
      
      RETURN
      END

#ifdef CUFFT_MIN
    SUBROUTINE FFTWAV(NPL,NINDPW,CR,C,GRID)
      USE prec
      USE cuda_interface
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRID
      COMPLEX(q) :: C(NPL), CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)

      if ((.not. GRID%REAL2CPLX) .and. (GRID%NGPTAR(3) .ge. CUFFT_MIN)) then
        call fftwav_cu( &
                NPL, &
                NINDPW, &
                CR, &
                C, &
                GRID%NGPTAR, &
                merge(1,0,GRID%LREAL))
      else

      ! zero all elements on the grid gK 021206
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,GRID%NGX_rd*GRID%NGY_rd*GRID%NGZ_rd
           CR(M)=(0.0_q,0.0_q)
        ENDDO
      ! old version to zero all elements
!     DO M=1,GRID%NPLWV
!        CR(M)=(0.0_q,0.0_q)
!     ENDDO
      ! now fill in non zero elements from
      ! within the radial cutoff sphere
      IF (GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            CR(NINDPW(M))=C(M)*GRID%FFTSCA(M,2)
         ENDDO
      ELSE
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            CR(NINDPW(M))=C(M)
         ENDDO
      ENDIF
! here you can make what you want - safety first ...
        CALL FFT3D(CR,GRID,1)
! ... or 'fastness' first (but often it is not   so   much faster ...):
!      CALL FFTQ2Q(CR,GRID%NGPTAR(1),1)
      endif

    END SUBROUTINE FFTWAV



    SUBROUTINE FFTEXT(NPL,NINDPW,CR,C,GRID,LADD)
      USE prec
      USE cuda_interface
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      DIMENSION C(NPL),CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)
      LOGICAL   LADD

      if ((.not. GRID%REAL2CPLX) .and. (GRID%NGPTAR(3) .ge. CUFFT_MIN)) then
        call fftext_cu( &
                NPL, &
                NINDPW, &
                CR, &
                C, &
                GRID%NGPTAR, &
                merge(1,0,GRID%LREAL), &
                merge(1,0,LADD))
      else

      CALL FFT3D(CR,GRID,-1)

      IF (LADD .AND. GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            C(M)=C(M)+CR(NINDPW(M))*GRID%FFTSCA(M,1)
         ENDDO
      ELSE IF (LADD .AND. .NOT. GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            C(M)=C(M)+CR(NINDPW(M))
         ENDDO
      ELSE IF (GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
          C(M)=CR(NINDPW(M))*GRID%FFTSCA(M,1)
        ENDDO
     ELSE
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
          C(M)=CR(NINDPW(M))
        ENDDO
      ENDIF


      endif

      RETURN
    END SUBROUTINE FFTEXT

#endif



