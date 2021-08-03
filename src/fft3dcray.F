#include "fft3dsimple.F"
!===============================================================================
! RCS:  $Id: fft3dcray.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!   basic complex 3-d fast fourier transform using CRAY FFT3D
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!===============================================================================


      SUBROUTINE FFTBAS(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
      PARAMETER (NMAXM=64)
      DIMENSION TABLE(100+6*NMAXM)
      DIMENSION WORK(512*NMAXM)
      COMMON /FFTWRK/ TABLE,WORK

      NX=N(1)
      NY=N(2)
      NZ=N(3)
      IF (MAX(NX,NY,NZ)>NMAXM) THEN
         WRITE(*,*)'ERROR: FFTBAS: increase NMAXM to ', &
     &       MAX(NX,NY,NZ)
         STOP
      ENDIF

      SCALE=1.0_q
      CALL CCFFT3D(0  ,NX,NY,NZ,SCALE,C,NX,NY,C,NX,NY,TABLE,WORK,0)
      CALL CCFFT3D(ISN,NX,NY,NZ,SCALE,C,NX,NY,C,NX,NY,TABLE,WORK,0)

      RETURN
      END

!===============================================================================
!   basic real to complex 3-d fast fourier transformation routine using CRAY
!   routines
!   C is used as input and output array
!   in real space C is defined as
!    REAL(q)    C(1:N(1)+2  ,1:N(2),1:N(3)
!   in reciprocal space C is defined as
!    COMPLEX(q) C(1:N(1)/2+1,1:N(2),1:N(3)
!   this conforms to the ESSL and to the CRAY routines
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!===============================================================================


      SUBROUTINE FFTBRC(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
      PARAMETER (NMAXM=64)
      DIMENSION TABLE(100+6*NMAXM)
      DIMENSION WORK(512*NMAXM)
      COMMON /FFTWRK/ TABLE,WORK


      NX=N(1)
      NY=N(2)
      NZ=N(3)
      INC3X=(NX/2+1)*NY
      INC2X= NX/2+1
      SCALE=1.0_q
      IF (ISN==1) THEN
        CALL CSFFT3D(0  ,NX,NY,NZ,SCALE, &
     &         C,INC2X,NY,C,INC2X*2,NY,TABLE,WORK,0)
        CALL CSFFT3D(ISN,NX,NY,NZ,SCALE, &
     &         C,INC2X,NY,C,INC2X*2,NY,TABLE,WORK,0)
      ELSE
        CALL SCFFT3D(0  ,NX,NY,NZ,SCALE, &
     &         C,INC2X*2,NY,C,INC2X,NY,TABLE,WORK,0)
        CALL SCFFT3D(ISN,NX,NY,NZ,SCALE, &
     &         C,INC2X*2,NY,C,INC2X,NY,TABLE,WORK,0)
      ENDIF

      RETURN
      END




!=======================================================================
!   this routine returns the next correct setting for the
!   three dimensional FFT
!   I have not checked whether the CRAY supports prime factors 2,3 and 5
!   because I had no documentation on hand when wrote the routine
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
      PARAMETER (NFACT=3)
      DIMENSION IFACT(NFACT),NCOUNT(NFACT)
      DATA      IFACT /2,3,5/
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
      IF (N==1 .AND. (NCOUNT(1)/=0)) &
     &  THEN
        FFTCH1=.TRUE.
      ELSE
        FFTCH1=.FALSE.
      ENDIF
      RETURN
      END
