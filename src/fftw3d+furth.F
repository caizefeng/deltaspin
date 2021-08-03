#include "fft3dsimple.F"
!-----------------------------------------------------------------------
! This is routine calls either FFTW (http://www.fftw.org) or fft3dlib
! depending on the first dimension of the array to be transformed
! generally the FFTW is slower if the first dimension is a 2^n
! in that case the fft3dlib is called in all other cases 
! the FFTW library is used
! for the configuration of the  cache size etc please
! check the file fft3dfurth.F
!-----------------------------------------------------------------------
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
!=======================================================================
!
!   complex 3-d fast fourier transform using FFTW of FFT3DFURTH
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!=======================================================================

      SUBROUTINE FFTBAS(A,GRID,ISIGN)
      USE prec
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q) A(*)
      TYPE (grid_3d)   GRID
      LOGICAL FFTCH1

      N2=1
!      WRITE(*,*) 'FFTBAS',GRID%NGX,GRID%NGY,GRID%NGZ,GRID%NGX*GRID%NGY*GRID%NGZ
      DO I=1,30
         N2=N2*2
         IF ((GRID%NGX*GRID%NGY*GRID%NGZ)==N2) THEN
            CALL FFTBAS_FURTH(A,GRID,ISIGN)
            RETURN
         ELSE IF (N2>(GRID%NGX*GRID%NGY*GRID%NGZ))THEN
            EXIT
         ENDIF
      ENDDO
      DO IND=1,3
         IF (FFTCH1(GRID%NGPTAR(IND))) THEN
            CYCLE
         ELSE
            GOTO 200
         ENDIF
      ENDDO

      CALL FFTBAS_FFTW(A,GRID,ISIGN)
      RETURN

 200  CONTINUE
      CALL FFTBAS_FURTH(A,GRID,ISIGN)
      RETURN
      

      END SUBROUTINE

!=======================================================================
!
!   complex 3-d fast fourier transform using FFTW
!
!=======================================================================

      subroutine FFTBAS_FFTW(c,grid,isign)

      use prec

      implicit none

      include 'fftw_f77.i'

      COMPLEX(q)       :: c(*), cdummy
      INTEGER          :: grid(3), isign
      INTEGER,SAVE     :: oldgrid(3),forward_plan,backward_plan
      LOGICAL,SAVE     :: INIT=.FALSE.
      INTEGER          :: i,j,k, idummy

      IF ((.NOT.INIT).OR. &
        & (grid(1)/=oldgrid(1)).OR. &
        & (grid(2)/=oldgrid(2)).OR. &
        & (grid(3)/=oldgrid(3))) THEN

        IF(INIT) THEN
          call fftwnd_f77_destroy_plan_(forward_plan)
          call fftwnd_f77_destroy_plan_(backward_plan)
!          WRITE(*,*) 'plan destroyed'
        ENDIF

!        WRITE(*,*) 'plan generated'
        call fftw3d_f77_create_plan_(forward_plan,grid(1),grid(2),grid(3),&
        & FFTW_FORWARD,FFTW_MEASURE+FFTW_IN_PLACE)
        call fftw3d_f77_create_plan_(backward_plan,grid(1),grid(2),grid(3),&
        & FFTW_BACKWARD,FFTW_MEASURE+FFTW_IN_PLACE)

        oldgrid=grid
        INIT=.TRUE.
      ENDIF

      IF (isign.le.0) THEN
        call fftwnd_f77_one_(forward_plan,c,cdummy)
      ELSE
        call fftwnd_f77_one_(backward_plan,c,cdummy)
      ENDIF

      return

      end


!=======================================================================
!
!   complex 3-d fast fourier transform using ESSL-LIB of FFT3DFURTH
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!=======================================================================
!
!      SUBROUTINE FFTBRC(A,MF,ISIGN)
!      USE prec
!      IMPLICIT REAL(q) (A-H,O-Z)
!      COMPLEX(q) A(*)
!      DIMENSION GRID%NGZ
!      LOGICAL FFTCH1
!
!
!      DO IND=1,3
!         IF (FFTCH1(MF(IND))) THEN
!            CYCLE
!         ELSE
!            GOTO 200
!         ENDIF
!      ENDDO
!
!      CALL FFTBRC_ESSL(A,MF,ISIGN)
!      RETURN
!
! 200  CONTINUE
!      CALL FFTBRC_FURTH(A,MF,ISIGN)
!      RETURN
!
!
!      END SUBROUTINE
    

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

      SUBROUTINE FFTBAS_FURTH(A,GRID,ISIGN)
      USE prec
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q) A(*)

      TYPE (grid_3d)   GRID
      REAL(q),ALLOCATABLE ::  WORK(:)
      INTEGER, PARAMETER :: NMAXM=512  ! propably sufficient forever
      DIMENSION TRIGX(2*NMAXM),TRIGY(2*NMAXM),TRIGZ(2*NMAXM)
      DIMENSION IFAC(19,3)
      SAVE TRIGX,TRIGY,TRIGZ,IFAC,NXO,NYO,NZO
      DATA NXO /0/, NYO /0/, NZO /0/

      NX=GRID%NGX
      NY=GRID%NGY
      NZ=GRID%NGZ
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
      ALLOCATE(WORK(NALLOC))

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

      DEALLOCATE(WORK)
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

      SUBROUTINE FFTBRC(A,GRID,ISIGN)
      USE prec
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q) A(*)
      TYPE (grid_3d)   GRID
      REAL(q),ALLOCATABLE ::  WORK(:)
      INTEGER, PARAMETER :: NMAXM=512 ! propably sufficient forever
      DIMENSION TRIGX(2*NMAXM),TRIGY(2*NMAXM),TRIGZ(2*NMAXM)
      DIMENSION IFAC(19,3)
      SAVE TRIGX,TRIGY,TRIGZ,IFAC,NXO,NYO,NZO
      DATA NXO /0/, NYO /0/, NZO /0/

      NX=GRID%NGX
      NY=GRID%NGY
      NZ=GRID%NGZ
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
      ALLOCATE(WORK(NALLOC))
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
      DEALLOCATE(WORK)
      RETURN
      END


!
!  two test routine only internal use
!
      SUBROUTINE RNG(C,NX,NY,NZ)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) C(1)
      SUM=0
      DO N=1,(NX/2+1)*NY*NZ*2
       SUM=SUM+C(N)*C(N)
      ENDDO
      WRITE(*,*)'CNR',SUM/NX/NY/NZ
      RETURN
      END


      SUBROUTINE CNG(C,NX,NY,NZ)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q) C(1)
      SUM=0
      NE=0
      DO NN=0,(NX/2+1)*(NY*NZ-1),NX/2+1
      FACT=1
      DO N=1,(NX/2+1)
       NE=NE+1
       IF (FACT==2) C(NN+N)=0
       SUM=SUM+C(NN+N)*CONJG(C(NN+N))*FACT
       FACT=2
      ENDDO
      ENDDO
      WRITE(*,*)'CNG',SUM
      RETURN
      END


!=======================================================================
!   this routine returns the next correct setting for the
!   three dimensional FFT for the ESSL
!=======================================================================

      SUBROUTINE FFTCHK_(NFFT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION NFFT(3)
      PARAMETER (NFACT=51)
      DIMENSION NESSL(NFACT)

      DATA NESSL /8,10,12,14,16,18,20,24,28,30,32,36,40,42,48,56,60,64, &
     &    70,72,80,84,90,96,112,120,126,128,140,144,160,168,180,192, &
     &    210,224,240,252,256,280,288,320,336,360,384,420,448,480,504, &
     &    512,0/

      DO 100 IND=1,3
      DO 110 N=1,NFACT
        IF (NESSL(N)>=NFFT(IND)) THEN
           NFFT(IND)=NESSL(N)
           GOTO 100
        ENDIF
  110 CONTINUE
  100 CONTINUE

      END

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
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (NFACT=5)
      DIMENSION IFACT(NFACT),NCOUNT(NFACT)
      DATA      IFACT /2,3,5,7,11/
      N=NIN
      DO 100 I=1,NFACT
        NCOUNT(I)=0
  120   NEXT=N/IFACT(I)
        IF (NEXT*IFACT(I)==N) THEN
          N=NEXT
          NCOUNT(I)=NCOUNT(I)+1
          GOTO 120
        ENDIF
  100 CONTINUE
      IF (N==1 .AND. (NCOUNT(1)/=0) &
     &  .AND.(NCOUNT(2)<=2).AND.(NCOUNT(3)<=1).AND.(NCOUNT(4)<=1) &
     &  .AND.(NCOUNT(5)<=1)) &
     &  THEN
        FFTCH1=.TRUE.
      ELSE
        FFTCH1=.FALSE.
      ENDIF
      RETURN
      END

