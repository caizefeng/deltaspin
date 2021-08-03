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
!-----------------------------------------------------------------------
! RCS:  $Id: fft3dfurth.F,v 1.3 2003/06/27 13:22:16 kresse Exp kresse $
!
!  ADVICE FOR USERS OF "SCALAR MACHINES" (RISC-WORKSTATIONS):
!  ==========================================================
!
!  if you know the cache size please define it on previous line
!  in complex double pression words
!  or somewhere in the preprocessor call
!  if you have no idea at all set to zero
!  in this case each plane will be transformed seperately
!  only for extremely large FFT-grids this might slow the speed
!
!  ADVICE FOR USERS OF LARGE VECTOR MACHINES:
!  ==========================================
!
!  set CACHE_SIZE to 0 whenever possible! (gives maximum performance)
!  only if you have very restrictive memory problems (high costs for
!  big jobs or extremely big problems) set some value -- this decreases
!  usually (maybe significantly) the size of the work-array needed,
!  A good idea is to set it to the value of 4*NGX*NGY*NGZ
!  in this case FFTs for the wavefunctions will be done with
!  maximum speed vector routine
!
! some examples:
! IBM   32768
! T3D    8000
!-----------------------------------------------------------------------
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
!   dispatches either to the vector or to the seriell version
!   seriell version uses less memory
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)    complex to real
!     -1  r->q   vq= sum(r) vr exp(-iqr)    real to complex
!
!=======================================================================

      SUBROUTINE FFTMAKEPLAN(A,GRID)
      USE prec
      USE mgrid
      TYPE (grid_3d) GRID
      REAL(q) A(*)
      RETURN
      END SUBROUTINE


      SUBROUTINE FFTBAS(A,MF,ISIGN)
      USE prec
      USE fft_private
      USE smart_allocate
      IMPLICIT REAL(q) (A-H,O-Z)

      COMPLEX(q) A(*)
      INTEGER, PARAMETER :: NMAXM=4096  ! propably sufficient forever
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
         CALL FFTC3V(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,IERR)
      ELSE
         CALL FFTC3N(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,IERR,CACHE_SIZE)
      ENDIF
#else
         CALL FFTC3N(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &               TRIGX,TRIGY,TRIGZ,0,IERR,CACHE_SIZE)
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
      CALL FFTC3V(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,IERR)
      ELSE
      CALL FFTC3N(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,IERR,CACHE_SIZE)
      ENDIF
#else
      CALL FFTC3N(A,NX,NX,NY,NY,NZ,NZ,WORK,IFAC, &
     &            TRIGX,TRIGY,TRIGZ,ISIGN,IERR,CACHE_SIZE)
#endif
      IF (IERR/=0) THEN
         WRITE(*,*) 'FFT3D: IERR =',IERR
         STOP
      ENDIF

      RETURN
      END


!=======================================================================
!   generic   3-d fast fourier transformation
!   written by Jueregen Furthmueller
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
