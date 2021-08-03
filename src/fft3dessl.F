#include "fft3dsimple.F"
!=======================================================================
! RCS:  $Id: fft3dessl.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!   basic complex 3-d fast fourier transform using ESSL-LIB
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!  we asume that the optimal STRIDE for second dimension is equal NX
!  (propably this is not allways true)
!  the STRIDE INC3Y can be calculated explicitly (see ESSL-Doc for STRIDE)
!  in that case the WORK-array is used and the values must me rearanged
!  it turns out that this is slower on a RISC 6000 Workstation
!  so this option is commented out
!=======================================================================


      SUBROUTINE FFTBAS(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
      PARAMETER (NAUX=65000)
      COMPLEX(q) AUX(NAUX)
      COMMON /FFTWRK/ AUX

      NX=N(1)
      NY=N(2)
      NZ=N(3)
      INC3X=NX*NY

      INC3Y=INC3X

!      CALL STRIDE(NX,INC3X,INC3Y,'C',0)
!-----if optimal STRIDE INC3Y is equal NX*NY call DCFT3
!-----with output-location equal input-location
!      IF (INC3Y==INC3X) THEN
        CALL DCFT3(C,NX,INC3X,C,NX,INC3X,NX,NY,NZ,-ISN,1.0_q,AUX,NAUX)
!      ELSE
!-----in the other case some data-movement is necessary
!        CALL DCFT3(C,NX,NX*NY,WORK,NX,INC3Y,NX,NY,NZ,-ISN,1.0,AUX,NAUX)
!        DO 100 NNZ=0,NZ-1
!        DO 100 NN2=0,INC3X-1
! 100      C(NN2+INC3X*NNZ)=WORK(NN2+INC3Y*NNZ)
!      ENDIF

      RETURN
      END


!=======================================================================
! basic real to complex 3-d fast fourier transformation routine using ESSL
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
!=======================================================================


      SUBROUTINE FFTBRC(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
      PARAMETER (NAUX=65000)
      COMPLEX(q) AUX(NAUX)
      COMMON /FFTWRK/ AUX

      NX=N(1)
      NY=N(2)
      NZ=N(3)
      INC3X=(NX/2+1)*NY
      INC2X= NX/2+1
      IF (ISN==1) THEN
        CALL DCRFT3(C,INC2X,INC3X,C,INC2X*2,INC3X*2, &
     &            NX,NY,NZ,-ISN,1.0_q,AUX,NAUX)
      ELSE
        CALL DRCFT3(C,INC2X*2,INC3X*2,C,INC2X,INC3X, &
     &              NX,NY,NZ,-ISN,1.0_q,AUX,NAUX)
      ENDIF

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
