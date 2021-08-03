#include "fft3dsimple.F"
!===============================================================================
! RCS:  $Id: fft3dfujitsu.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!   basic complex 3-d fast fourier transform using VPFFTLIB
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!===============================================================================

      SUBROUTINE FFTBAS(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      PARAMETER (NTRIG= 1000)

      COMPLEX(q)   T(NTRIG)
      COMPLEX(q)   C(1)
      REAL(q),ALLOCATABLE:: WORK(:)
      DIMENSION N(3)

      NX=N(1)
      NY=N(2)
      NZ=N(3)
      NXYZ=NX*NY*NZ
      ALLOCATE(WORK(NXYZ*2))

      DO M=1,NXYZ
        WORK(M)= REAL( C(M) ,KIND=q)
        WORK(M+NXYZ)=AIMAG(C(M))
      ENDDO

      IF (ISN==1) THEN
      CALL DFTCBM(WORK,WORK(1+NXYZ),3,N,C,T,'P','I',IFAIL)
      ELSE
      CALL DFTCBM(WORK,WORK(1+NXYZ),3,N,C,T,'M','I',IFAIL)
      ENDIF
      AMUL = SQRT(NXYZ*1._q)

      DO M=1,NXYZ
        C(M)= CMPLX( WORK(M)*AMUL  , WORK(M+NXYZ)*AMUL  ,KIND=q)
      ENDDO

      DEALLOCATE(WORK)

      RETURN
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
