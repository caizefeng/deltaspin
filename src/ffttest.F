#undef NGXhalf
#include "symbol.inc"
#define LIST
!=======================================================================
!  This routine benchmarks the FFT3D calls
!=======================================================================

      PROGRAM FFTBEN
      USE prec
      IMPLICIT REAL(q) (A,B,D-H,O-Z)
      IMPLICIT COMPLEX(q) (C)

      PARAMETER (NFFT= 64)
      COMPLEX(q) C1((NFFT+1)*NFFT*NFFT),C2((NFFT+1)*NFFT*NFFT)
            
      PARAMETER (NMAX=100)
      DIMENSION NCHECK(NMAX)       
      DIMENSION NLIST(19)
      COMPLEX(q) CTMP
      DIMENSION NGRID(3)                                               
      LOGICAL FFTCH1
      DATA NLIST /28,30,32,36,40,42,48,56,60,64, &
     &    72,80,90,96,100,108,120,128,0/
            
!-----------------------------------------------------------------------
!  setup factors
!-----------------------------------------------------------------------
      TOTV=0
      TOTC=0
      NS=24
      DO 100 NFAK=1,NMAX
#ifdef LIST
        NCHECK(NFAK)=NLIST(NFAK)
#else
 105    NS=NS+1
        IF (.NOT.FFTCH1(NS)) GOTO 105
        NCHECK(NFAK)=NS
#endif
        IF (NCHECK(NFAK).EQ.0.OR.NCHECK(NFAK).GT.NFFT) GOTO 110
  100 CONTINUE

  110 NFAK=NFAK-1
      WRITE(*,*)'Checking ',NFAK,' FFT-grids'

!-----------------------------------------------------------------------
!  Loop over all factors
!-----------------------------------------------------------------------

#if defined OUTPUT || defined INPUT
      OPEN(35,FILE='FFTDAT',FORM='UNFORMATTED')
#endif

      DO 200 N=1,NFAK

      NGR=NCHECK(N)
#ifdef NGXHALF
      IF (MOD(NGR,2).NE.0) GOTO 200
#endif
      NGRID(1)=NGR
      NGRID(2)=NGR
      NGRID(3)=NGR

      DO 300 N1=1,xm(NGR)*NGR*NGR
        C2(N1)=RANE()+(0.,1.)*RANE()
#ifdef NGXhalf
        IF (((MOD(N1-1,xm(NGR))+1).EQ.1).OR. &
     &      ((MOD(N1-1,xm(NGR))+1).EQ.xm(NGR))) &
     &        C2(N1)=REAL(C2(N1))
#endif
  300 CONTINUE

      NCALL=MAX(2000000._q/xm(NGR)/NGR/NGR,1._q)
      DO 410 N2=1,xm(NGR)*NGR*NGR
         C1(N2)=C2(N2)
  410 CONTINUE
#ifdef NGXhalf
        CALL FFTBRC(C1,NGRID,-1)
        CALL FFTBRC(C1,NGRID,+1)
#else
        CALL FFTBAS(C1,NGRID,-1)
        CALL FFTBAS(C1,NGRID,+1)
#endif

      CALL VTIME(TV0,TC0)

      DO 400 N1=1,NCALL

        DO 610 N2=1,xm(NGR)*NGR*NGR
          C1(N2)=C2(N2)
  610   CONTINUE

#ifdef NGXhalf
        CALL FFTBRC(C1,NGRID,-1)
        CALL FFTBRC(C1,NGRID,+1)
#else
        CALL FFTBAS(C1,NGRID,-1)
        CALL FFTBAS(C1,NGRID,+1)
#endif

  400 CONTINUE

      CALL VTIME(TV,TC)
      TOTV=TOTV+TV-TV0
      TOTC=TOTC+TC-TC0

      RINV=1/REAL(NGR)**3.
      ERR=0
      DO 500 N1=1,xm(NGR)*NGR*NGR
        CTMP=(C2(N1)-C1(N1)*RINV)
#ifdef NGXhalf
        IF ((MOD(N1-1,xm(NGR))+1).EQ.xm(NGR)) CTMP=(0.,0.)
#endif
!        write(*,*) C2(N1),C1(N1)*RINV
        ERR=ERR+CTMP*CONJG(CTMP)
  500 CONTINUE
      ERR=SQRT(ERR)

      WRITE(*,10) NGR,NCALL,(TV-TV0)/NCALL,(TC-TC0)/NCALL,ERR
   10 FORMAT(' N=',I4,' called=',I5,'  t1=',F7.4,'  t2=',F7.4, &
     &         ' err=',E14.5)

      IF (NGR.LE.32) THEN
#ifdef OUTPUT
      DO 420 N2=1,xm(NGR)*NGR*NGR
        C1(N2)=C2(N2)
  420 CONTINUE
#ifdef NGXhalf
      CALL FFTBRC(C1,NGRID,-1)
#else
      CALL FFTBAS(C1,NGRID,-1)
#endif
      DO 430 N2=1,xm(NGR)*NGR*NGR
         WRITE(35) C2(N2)
  430 CONTINUE
      DO 440 N2=1,xm(NGR)*NGR*NGR
         WRITE(35) C1(N2)
  440 CONTINUE
#elif defined INPUT
      DO 620 N2=1,xm(NGR)*NGR*NGR
         READ(35) C1(N2)
  620 CONTINUE
#ifdef NGXhalf
      CALL FFTBRC(C1,NGRID,-1)
#else
      CALL FFTBAS(C1,NGRID,-1)
#endif
      DO 630 N2=1,xm(NGR)*NGR*NGR
         READ(35) C2(N2)
  630 CONTINUE
      ERR=0.
      DO 640 N1=1,xm(NGR)*NGR*NGR
         CTMP=C2(N1)-C1(N1)
         ERR=ERR+CTMP*CONJG(CTMP)
  640 CONTINUE
      WRITE(*,'(A,I4,4X,A,E12.4)') 'NGR=',NGR,'ERR1 = ',ERR
#endif
      ENDIF

  200 CONTINUE
      WRITE(*,11) TOTV,TOTC
   11 FORMAT(' total time ffttest =',2F6.1)

#if defined OUTPUT || defined INPUT
      CLOSE(35)
#endif

      END
