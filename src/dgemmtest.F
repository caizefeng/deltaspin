#include "symbol.inc"
!=======================================================================
!  This routine benchmarks the DGEMM respectively  DO-LOOPS
!=======================================================================

      PROGRAM DGEMMT
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!
!   allocate one large work array 
!   this array should be much large than the cache size 
!   so that each call to DGEMM starts with new data and
!   no chache hit between consectutive calls is possible
      PARAMETER(NWORK=1*1000000)
      PARAMETER(NTEST=1000)

      DIMENSION WORK1(NWORK)
      DIMENSION WORK2(NWORK)  
      DIMENSION ND1(NTEST),N1(NTEST)
      DIMENSION ND2(NTEST),N2(NTEST)
      DIMENSION ND3(NTEST),N3(NTEST)
      LOGICAL LDIFF

      NWRITE=1
!-----------------------------------------------------------------------
!   read in tables with dimensions     
!-----------------------------------------------------------------------
      ATIMC1=0
      ATIMC2=0
      ATIMV1=0
      ATIMV2=0
      READ( 5,*) LDIFF,FAK
      ISET=1
 100  CONTINUE
      IF (LDIFF) THEN
        READ( 5,*,ERR=110,END=110) ND1(ISET),N1(ISET),ND2(ISET), &
     &             N2(ISET),ND3(ISET),N3(ISET)
      ELSE
        READ( 5,*,ERR=110,END=110) ND1(ISET),ND2(ISET),ND3(ISET)
        N1(ISET)=ND1(ISET)
        N2(ISET)=ND2(ISET)
        N3(ISET)=ND3(ISET)
      ENDIF
      ISET=ISET+1
      GOTO 100
  110 ISET=ISET-1
                 
!-----------------------------------------------------------------------
!   loop over all table sizes
!-----------------------------------------------------------------------
      DO 300 I=1,ISET

      NW12=ND1(I)*ND2(I)
      NW13=ND1(I)*ND3(I)
      NW23=ND2(I)*ND3(I)

!-----------------------------------------------------------------------
!  init tables randomly
!-----------------------------------------------------------------------
      DO 200 N=1,NWORK
        WORK1(N)=RANE()
        WORK2(N)=WORK1(N)
  200 CONTINUE
                         
!     determine maximum number of independent data sets
      NDATAS=NW12+NW13+NW23
      NDATA =NWORK/NDATAS     
      NOP   =(N3(I)*N1(I)*N2(I))*2
      FNOP  =NOP
      NCALL =MAX(1._q,FAK*100E6_q/(NDATA*FNOP))*NDATA

      IF (NDATA.EQ.0) THEN
        WRITE(*,*)'increase NWORK to ',NDATAS
        STOP
      ENDIF

      CALL VTIME(TV0,TC0)
      DO 310 NC=1,NCALL,NDATA
      DO 310 ND=1,NDATA
        NDATAP=(ND-1)*NDATAS+1
        CALL DGEMMT1(ND1(I),ND2(I),ND3(I), &
     &         N1 (I),N2 (I),N3 (I), &
     &         WORK1(NDATAP),WORK1(NDATAP+NW12),WORK1(NDATAP+NW12+NW23))
 310  CONTINUE
      CALL VTIME(TV,TC)
      ATIM1=TV-TV0
      ATIMV1=ATIMV1+ATIM1
      ATIMC1=ATIMC1+TC-TC0
      AFLOP1=FNOP*NCALL/(TV-TV0)/1E6

      IF (NWRITE.EQ.0)  &
     &WRITE(*,10) N1(I),N2(I),N3(I),NCALL,(TV-TV0)/NCALL,(TC-TC0)/NCALL, &
     &            FNOP*NCALL/(TV-TV0)/1E6

      CALL VTIME(TV0,TC0)
      DO 320 NC=1,NCALL,NDATA
      DO 320 ND=1,NDATA
        NDATAP=(ND-1)*NDATAS+1
        CALL DGEMMTV(ND1(I),ND2(I),ND3(I), &
     &         N1 (I),N2 (I),N3 (I), &
     &         WORK2(NDATAP),WORK2(NDATAP+NW12),WORK2(NDATAP+NW12+NW23))
 320  CONTINUE
      CALL VTIME(TV,TC)
      ATIM2=TV-TV0
      ATIMV2=ATIMV2+ATIM2
      ATIMC2=ATIMC2+TC-TC0
      AFLOP2=FNOP*NCALL/(TV-TV0)/1E6

      ERR=0
      DO 330 ND=1,NDATA
        NDATAP=(ND-1)*NDATAS+1
        CALL SETERR(ND1(I),ND2(I),ND3(I), &
     &         N1 (I),N2 (I),N3 (I), &
     &         WORK1(NDATAP),WORK2(NDATAP),ERR)
 330  CONTINUE


      IF (NWRITE.EQ.0)  &
     &WRITE(*,10) N1(I),N2(I),N3(I),NCALL,ATIM1,ATIM2, &
     &            NOP*NCALL/(TV-TV0)/1E6,ERR

   10 FORMAT(' N=',3I5,' called=',I5,'  t1=',F7.4,'  t2=',F7.4, &
     &       ' mflop=',F6.1,' err=',E14.5)

      IF (NWRITE.EQ.1)  &
     &WRITE(*,20) N1(I),N2(I),N3(I),AFLOP1,AFLOP2,ERR
   20 FORMAT(' N=',3I6,'  dgemm=',F6.1,' dgemmv=',F6.1,' err =',E14.5)

      WRITE(*,80) ATIMV1,ATIMC1,ATIMV2,ATIMC2 

 300  CONTINUE
      
   80 FORMAT(' total time dgemm=',2F6.1/ &
     &       ' total time do   =',2F6.1)

      END

!-----------------------------------------------------------------------
!  first test routine using DGEMM
!-----------------------------------------------------------------------
      SUBROUTINE DGEMMT1(ND1,ND2,ND3,N1,N2,N3, &
     &  WORK1,WORK2,WORK3)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION WORK1(ND1,ND2)
      DIMENSION WORK2(ND3,ND2)
      DIMENSION WORK3(ND3,ND1)
      A=1.0
      B=0.0

      CALL DGEMM( 'T', 'N' , N1, N2, N3, A, &
     &             WORK3, ND3, WORK2, ND3 , B, WORK1, ND1)

      RETURN
      END

!-----------------------------------------------------------------------
!  second test routine using DO-LOOPS
!-----------------------------------------------------------------------
      SUBROUTINE DGEMMT2(ND1,ND2,ND3,N1,N2,N3, &
     &  WORK1,WORK2,WORK3)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION WORK1(ND1,ND2)
      DIMENSION WORK2(ND3,ND2)
      DIMENSION WORK3(ND3,ND1)

      DO 100 I2=1,N2
!
!   unroll N1 loop by 2
!
      DO 110 I1=1,N1-1,2
      SUM1=0
      SUM2=0
      DO 120 I3=1,N3
        SUM1=SUM1+WORK3(I3,I1)  *WORK2(I3,I2)
        SUM2=SUM2+WORK3(I3,I1+1)*WORK2(I3,I2)
  120 CONTINUE
      WORK1(I1,I2)  =SUM1
      WORK1(I1+1,I2)=SUM2
  110 CONTINUE
!
!   take care of remaining part
!
      I1START=I1
      DO 130 I1=I1,N1
      SUM=0
      DO 140 I3=1,N3
        SUM=SUM+WORK3(I3,I1)*WORK2(I3,I2)
  140 CONTINUE
      WORK1(I1,I2)=SUM
  130 CONTINUE
  100 CONTINUE

      RETURN
      END  

!-----------------------------------------------------------------------
!  second test routine using simple DO-LOOPS
!-----------------------------------------------------------------------
      SUBROUTINE DGEMMT3(ND1,ND2,ND3,N1,N2,N3, &
     &  WORK1,WORK2,WORK3)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION WORK1(ND1,ND2)
      DIMENSION WORK2(ND3,ND2)
      DIMENSION WORK3(ND3,ND1)

      DO 100 I2=1,N2
      DO 110 I1=1,N1
      SUM=0
      DO 120 I3=1,N3
        SUM=SUM+WORK3(I3,I1)*WORK2(I3,I2)
  120 CONTINUE
      WORK1(I1,I2)=SUM
  110 CONTINUE
  100 CONTINUE

      RETURN
      END

!------------------------------------------------------------------
!  do test with DGEMV
!-----------------------------------------------------------------------
      SUBROUTINE DGEMMTV(ND1,ND2,ND3,N1,N2,N3, &
     &  WORK1,WORK2,WORK3)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION WORK1(ND1,ND2)
      DIMENSION WORK2(ND3,ND2)
      DIMENSION WORK3(ND3,ND1)
      DIMENSION TMP(10000)

      A=1.0
      B=0.0

      DO 100 I2=1,N2
      CALL DGEMV( 'T' , N3, N1, A , WORK3(1,1) ,  &
     &             ND3, WORK2(1,I2) , 1 , B ,  TMP(1), 1)
      DO 120 I1=1,N1
       WORK1(I1,I2)=TMP(I1)
  120 CONTINUE
  100 CONTINUE

      RETURN
      END
      

!-----------------------------------------------------------------------
!  determine total error
!-----------------------------------------------------------------------

      SUBROUTINE SETERR(ND1,ND2,ND3,N1,N2,N3, &
     &  WORK1A,WORK1B,ERR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION WORK1A(ND1,ND2)
      DIMENSION WORK1B(ND1,ND2)
                                             
      DO 110 I1=1,N1
      DO 110 I2=1,N2
        ERR=ERR+(WORK1A(I1,I2)-WORK1B(I1,I2))* &
     &          (WORK1A(I1,I2)-WORK1B(I1,I2))
  110 CONTINUE
  100 CONTINUE

      RETURN
      END     





