#include "symbol.inc"
!*******************************************************************
! RCS:  $Id: optreal.F,v 1.4 2002/04/16 07:28:49 kresse Exp $
!
!  optimize real-space projector function according to
!  scheme proposed by King-Smith et al.
!  written by Georg Kresse
!
!  most integral in this routine are done with Gauss quadrature
!  this allows very accurate integration with modest grids
!  the matrix and vectors in King-Smith main linear equation
!  are also defined on Gauss quadrature grids
!
!*******************************************************************


      SUBROUTINE OPTREAL(IU,LREAD,LL,NQNL,DELQNL,BETA,BETAQ, &
     &          QGAMIN,QCUTIN,RMAXIN,LAUTO,ROPT)
      USE prec
      USE ini

      IMPLICIT REAL(q) (A-H,O-Z)

!-----potential
      INTEGER  IU                  !
      INTEGER  LREAD               ! number of ln-channels
      INTEGER  LL(LREAD)           ! l-qunatum number for each channel
      INTEGER  NQNL                ! number of grid points
      REAL(q)  DELQNL              ! spacing in reciprocal space
      REAL(q)  BETA(NQNL,5,LREAD)  ! out: opt projectors in real space
      REAL(q)  BETAQ(0:NQNL,LREAD) ! projectors in rec. space
      REAL(q)  RMAXIN              ! cutoff for real space proj
      REAL(q)  QGAMIN,QCUTIN       ! optimization between [QCUTIN,QGAMIN]
      LOGICAL  LAUTO               ! search for QCUT
!-----gaussian integration
      PARAMETER (M=32)
      REAL(q)  WQ(M),QQA(M),WQQ(M),QA(M),RA(M),WR(M)
      EXTERNAL GAUSSI2
!-----
      DIMENSION BETNEW(M),AMAT(M,M),V(M),VP(M),TMP(M), &
     &          IPIV(M)
      REAL(q) STORE(8,20)
      INTEGER ISTORE(20)

!-----spline interpolation
      REAL(q)  WRK(5,NQNL)
      REAL(q)  WRKN(5,NQNL)
      REAL(q) QMESH(NQNL),RMESH(2*NQNL),BETOPT(2*NQNL),BET2(2*NQNL)

      PARAMETER (AUTOA=0.529177249_q,RYTOEV=13.605826_q)

      PI = 3.141592653589793238462643E0_q
!-----------------------------------------------------------------------
! allign QCUTP and QGAM with linear grid
!-----------------------------------------------------------------------
      QCUTP=DELQNL*INT(QCUTIN/DELQNL)
      IF (QCUTP>=QCUTIN) QCUTP=QCUTIN
      QGAM=DELQNL*INT(QGAMIN/DELQNL)
      ITYPE = 0

      DO N=1,NQNL
        QMESH(N)=(N-1)*DELQNL
      ENDDO
!-----------------------------------------------------------------------
! set up the linear reciprocal space-mesh
! and the    linear real space-mesh
!-----------------------------------------------------------------------
      RMAX=RMAXIN

qcut_loop: DO

      DELR=RMAX/NQNL
      DO N=1,2*NQNL
        RMESH(N)=(N-1)*DELR
      ENDDO
      RMESH(1)=1E-10_q
!-----------------------------------------------------------------------
! do optimization between 0 and RMAXD=RMAX-DELR*NZERO [1...NQNL-1]
! the value are brought continously to zero at the end of the interval
!-----------------------------------------------------------------------
      IF (LAUTO) THEN
        NZERO=0
        RMAXD= RMESH(NQNL-NZERO)  ! last NZERO values are used to smooth end
      ELSE
        NZERO=1
        RMAXD= RMESH(NQNL-NZERO)  ! last NZERO values are used to smooth end
      ENDIF

proj_loop: DO I=1,LREAD
!
      DO N=1,NQNL
        BETOPT(N)=BETAQ(N,I)
      ENDDO

      QCUT= DELQNL*INT(ABS(QCUTP+DELQNL/2)/DELQNL)
!-----------------------------------------------------------------------
! caclulate the positions and weigths for QQ (q bar)  und QI (q)
! and for the R-mesh
!-----------------------------------------------------------------------
      MQQMAX=M
      MD=M
      MQMAX =MD

      CALL GAUSSI(GAUSSI2,QCUT,QGAM,ITYPE,MQQMAX,WQQ,QQA,IFAIL)
!-----reverse the order (because of spline-fit)
      DO N=1,MQQMAX/2
        SWAP         =WQQ(N)
        WQQ(N)       =WQQ(MQQMAX+1-N)
        WQQ(MQQMAX+1-N)=SWAP
        SWAP         =QQA(N)
        QQA(N)       =QQA(MQQMAX+1-N)
        QQA(MQQMAX+1-N)=SWAP
      ENDDO
      CALL GAUSSI(GAUSSI2,0.0_q ,QCUT,ITYPE,MQMAX ,WQ ,QA,IFAIL)
      CALL GAUSSI(GAUSSI2,0.0_q,RMAXD,ITYPE,MD    ,WR ,RA,IFAIL)
!-----------------------------------------------------------------------
! spline fit of BETAQ(q)
!-----------------------------------------------------------------------
      CALL SPLCOF2(QMESH,BETAQ(1,I),NQNL,WRK,0.0_q)
!-----------------------------------------------------------------------
! calculate V(QQ)=A(QQ,QI)*X(QI)*F(QI) using gauss-quadratur
!-----------------------------------------------------------------------
      CALL SPLVAL2(QCUT,BETCUT,DUMMY,NQNL,WRK)
      V   =0
      AMAT=0

!-----Loop over all QI
      DO MQQ=1,MQQMAX
      QQ=QQA(MQQ)
      DO MQ =1,MQMAX
        QI =QA(MQ)
        CALL SBESSE2(RMAXD*QQ, BQQ , BQQP, LL(I))
        CALL SBESSE2(RMAXD*QI , BQ  , BQP , LL(I))
        A= QI**2*QQ**2*RMAXD**2/(QI**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*QI)
        CALL SPLVAL2(QI,B,DUMMY,NQNL,WRK)
        V(MQQ)=V(MQQ)+A*B*WQ(MQ)
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! now caclulate the matrix AMAT=A(QQ,QQP)  + PI/2  D(QQ-QQP) QQ*2
!-----------------------------------------------------------------------
      DO MQQ =1,MQQMAX
      QQ =QQA(MQQ)
      DO MQQP=1,MQQMAX
      QQP=QQA(MQQP)

        CALL SBESSE2(RMAXD*QQ, BQQ , BQQP, LL(I))
        CALL SBESSE2(RMAXD*QQP,BQ  , BQP , LL(I))
        IF (QQP/=QQ) THEN
        A= QQP**2*QQ**2*RMAXD**2/(QQP**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*QQP)
        ELSE
        QR=QQ*RMAXD
         A= QQ**3* RMAXD**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL(I)+1)*LL(I)/QR*BQ**2)
        ENDIF

        AMAT(MQQ,MQQP)= AMAT(MQQ,MQQP)-A*WQQ(MQQP)
        IF (MQQ==MQQP) AMAT(MQQ,MQQP)=AMAT(MQQ,MQQP) + PI/2*QQ**2
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
!  calculate the solution of  V(QQ)=AMAT(QQ,QQP) * BETNEW(QQP)
!-----------------------------------------------------------------------
      CALL DGETRF( MQQMAX, MQQMAX, AMAT, M, IPIV, INFO )
      CALL DGETRS('N', MQQMAX, 1, AMAT, M, IPIV, V, M, INFO)
      DO MQ=1,MQQMAX
        BETNEW(MQ)=V(MQ)
      ENDDO
!-----------------------------------------------------------------------
!  intermezzo:
!  interpolate BETNEW (on Gauss-mesh)
!  and reset the array BETOPT (beta(q) on linear-mesh) to the new values
!-----------------------------------------------------------------------
      CALL SPLVAL2(QCUT,B,DUMMY,NQNL,WRK)
      CALL SPLCOF2(QQA,BETNEW,MQQMAX,WRKN,0.0_q)
      DO N=1,NQNL
        IF ((N-1)*DELQNL>QCUT+QCUT/1000) EXIT
      ENDDO

      DO N=N,2*NQNL
        QI= (N-1)*DELQNL
        IF (QI>QGAM+DELQNL) EXIT
        CALL SPLVAL2(QI,BETOPT(N),DUMMY,MQQMAX,WRKN)
      ENDDO

      DO N=N,2*NQNL
        BETOPT(N)=0
      ENDDO
!      WRITE(79,*)'betq'
!      DO N=1,NQNL
!        WRITE(79,'(2F14.7)') (N-1)*DELQNL,BETOPT(N)
!      ENDDO
!========================================================================
! o recalculate the real-space projection operators BETA
!   (on a equally spaced mesh)
! o TMP is set to the real-space projection operator on a gauss-mesh in
!   real space  this allows straight forward integration in real space
! in the integrals we have two parts int[0,Qcut]+int[Qcut,Qgamma]
!========================================================================
      CALL GAUSSI(GAUSSI2,0.0_q ,RMAX,ITYPE,MD    ,WR ,RA,IFAIL)

! first setup TMP(N) on a gauss-quadrature mesh
!
! integral from 0 to Qcut
      DO MQ=1,MQMAX
        QI=QA(MQ)
        CALL SPLVAL2(QI,BBBB,DUMMY,NQNL,WRK)  ! BBBB introduced to work around DEC
                                              ! compiler bug
        B=BBBB
        V(MQ)   = B*QA(MQ)**2*WQ(MQ)
      ENDDO

      DO N=1,MD
        SUM = 0
        DO MQ=1,MQMAX
           CALL SBESSEL(RA(N)*QA(MQ),BJ,LL(I))
           SUM=SUM+BJ*V(MQ)
        ENDDO
        TMP(N)=SUM/2/PI**2
      ENDDO

! integral from Qcut to Qgamma
      DO MQ=1,MQQMAX
        V(MQ)   = BETNEW(MQ)*QQA(MQ)**2*WQQ(MQ)
      ENDDO

      DO N=1,MD
        SUM = 0
        DO MQ=1,MQQMAX
           CALL SBESSEL(RA(N)*QQA(MQ),BJ,LL(I))
           SUM=SUM+BJ*V(MQ)
        ENDDO
        TMP(N)=TMP(N)+SUM/2/PI**2
      ENDDO
!
! now set BETA(R) (equally spaced mesh)
! integral from 0 to Qcut
      DO MQ=1,MQMAX
        QI=QA(MQ)
        CALL SPLVAL2(QI,B,DUMMY,NQNL,WRK)
        V(MQ)   = B*QA(MQ)**2*WQ(MQ)
      ENDDO

      DO N=1,2*NQNL
        SUM = 0
        DO MQ=1,MQMAX
           CALL SBESSEL(RMESH(N)*QA(MQ),BJ,LL(I))
           SUM=SUM+BJ*V(MQ)
        ENDDO
        IF (N <= NQNL) THEN
           BETA(N,2,I)=SUM/2/PI**2
           BETA(N,1,I)=RMESH(N)
        ENDIF
        BET2(N)=SUM/2/PI**2
      ENDDO
! integral from Qcut to Qgamma
      DO MQ=1,MQQMAX
        V(MQ)   = BETNEW(MQ)*QQA(MQ)**2*WQQ(MQ)
      ENDDO

      DO N=1,2*NQNL
        SUM = 0
        DO MQ=1,MQQMAX
           CALL SBESSEL(RMESH(N)*QQA(MQ),BJ,LL(I))
           SUM=SUM+BJ*V(MQ)
        ENDDO
        IF (N <= NQNL) THEN
           BETA(N,2,I)=BETA(N,2,I)+SUM/2/PI**2
        ENDIF
        BET2(N)=BET2(N)+SUM/2/PI**2
      ENDDO

!      WRITE(78,*)'operator in real space'
!      DO N=1,2*NQNL
!         WRITE(78,'(2F14.7)')  RMESH(N),BET2(N)
!      ENDDO

! damp out the last NZERO values
      IF (NZERO /=0 ) THEN
      ! to reduce termination ripples lead smoothly to zero
!         NZER=NZERO
!         NZ=NQNL-NZER
!         DO N=NZ,NQNL
!            DAMP=(COS((N-NZ)*PI/NZER)+1)/2
!            BETA(N,2,I)=BETA(N,2,I)*DAMP
!         ENDDO
!         BETA(NQNL,2,I)=0
!         WRITE(78,*)'operator fin'
!         DO N=1,NQNL
!            WRITE(78,'(3F14.7)')  RMESH(N),BETA(N,2,I)
!         ENDDO
         NZ=NQNL-NZERO
         NFOUND=NQNL
         DO N=NZ,NQNL
            IF (BETA(N,2,I)*BETA(N-1,2,I) < 0 ) THEN
               NFOUND=N
            ENDIF
         ENDDO
         DO N=NFOUND,NQNL
            BETA(N,2,I)=0
         ENDDO
         
      ENDIF
!========================================================================
!  calculate the transformation of the trucated real-space operator
!  and the difference to the original BETOPT
!  also calculate the difference to BETOPT calculated without
!   spline-fit in real space (this gives error of spline-fit)
!  the quantity is equal to W_l(q) (equ 7c) in King-Smith-paper
!========================================================================

      CALL SPLCOF(BETA(1,1,I) ,NQNL,NQNL,0._q)
      DO K=1,MD
        CALL SPLVAL(RA(K),B,DUMMY,BETA(1,1,I),NQNL,NQNL)
        V (K) = B*RA(K)**2*WR(K)
        VP(K) = TMP(K)*RA(K)**2*WR(K)
      ENDDO

      BETMAX=0
      WMAX  =0
      WMAXSP=0
!
! calculate errorfunction W_l(q) using BETA
! on a uniform grid up to QCUT
!
      NTEST=QCUT/DELQNL+1E-5_q
!      WRITE(80,*) 'next'
      DO N=1,NTEST+5
        SUM   =0
        SUMSPL=0
        DO K=1,MD
           CALL SBESSEL(QMESH(N)*RA(K),BJ,LL(I))
           SUM   =SUM   +BJ*V (K)
           SUMSPL=SUMSPL+BJ*VP(K)
        ENDDO
        SUM   =SUM*4*PI
        SUMSPL=SUMSPL*4*PI
        SOLL=0
        IF (N<=2*NQNL) THEN
           SOLL=BETOPT(N)
        ENDIF

        IF (N < NTEST) THEN
           WMAX  =MAX(WMAX  ,ABS(BETOPT(N)-SUM))
           WMAXSP=MAX(WMAXSP,ABS(SUM-SUMSPL))
           BETMAX=MAX(BETMAX,ABS(BETOPT(N)))
        ENDIF
!        WRITE(80,'(3F14.7)') QMESH(N),SUM-SOLL,SOLL
      ENDDO

      WMAX_HIGH=0
      NTEST2=QGAM/DELQNL+1E-5_q
      DO N=NTEST2,NTEST2+NTEST
        SUM   =0
        SUMGAU=0
        DO K=1,MD
           CALL SBESSEL((N-1)*DELQNL*RA(K),BJ,LL(I))
           SUM   =SUM   +BJ*V (K)
           SUMGAU=SUMGAU+BJ*VP(K)
        ENDDO
        SUM   =SUM*4*PI
        SUMGAU=SUMGAU*4*PI
        SOLL=0
        WMAX_HIGH  =MAX(WMAX_HIGH  ,ABS(SOLL-SUM))
!        WRITE(80,'(4F14.7)') (N-1)*DELQNL,SUM,SUMGAU
      ENDDO

      ISTORE(I)=LL(I)
      STORE(1,I) =QCUT
      STORE(2,I) =BETCUT
      STORE(3,I) =BETNEW(1)
      STORE(4,I) =BETNEW(MQQMAX)
      STORE(5,I) =BETMAX
      STORE(6,I) =WMAX/BETMAX
      STORE(7,I) =WMAX_HIGH/BETMAX
      STORE(8,I) =WMAXSP/BETMAX

! check whether required accuracy was reached
      IF ( LAUTO.AND. ABS(WMAX/BETMAX)  > ROPT ) THEN
! only first projector must be very accurate
!         IF ( I == 1) THEN
            RMAX = RMAX*1.03
            CYCLE qcut_loop
!         ELSE IF ( LL(I) /= LL(I-1) ) THEN
            RMAX = RMAX*1.03
            CYCLE qcut_loop
!         ENDIF
      ENDIF
!-----------------------------------------------------------------------
! write some important information
!-----------------------------------------------------------------------
 ENDDO proj_loop
 EXIT  qcut_loop
 ENDDO qcut_loop

      RMAXIN=RMAX

      IF (IU>=0) THEN
      WRITE(IU,1) DELQNL*NQNL,ABS(QCUTP),QGAM, &
     &      QCUTP**2*AUTOA**2,QGAM**2*AUTOA**2,RMAX

   2  FORMAT(I4,1X,5F10.3,3E12.2)

    1 FORMAT(' Optimization of the real space projectors (old method)'// &
     &       ' maximal supplied QI-value         =',F6.2/ &
     &       ' optimisation between [QCUT,QGAM] = ', &
     &            '[',F6.2,',',F6.2,'] = [',F6.2,',',F6.2,'] Ry ' / &
     &       ' Optimized for a Real-space Cutoff  ',F6.2,' Angstroem'// &
     &       '   l      QCUT      X(QCUT)   X(cont)   X(QGAM) ', &
     &       'max X(q)   W(low)/max W(high)/max e(spline) ')

      DO I=1,LREAD
         WRITE(IU,2) ISTORE(I),STORE(:,I)
      ENDDO
      ENDIF

      RETURN
      END


!*******************************************************************
!
!  optimize real-space projectors according to a new scheme
!  written by Georg Kresse
!
!  the projectors are written as a sum of spherical Besselfunctions
!  on the sphere boundary the Besselfunctions are zero
!  the coefficients of the Besselfunctions are optimized to
!  reduce high frequency components 
!  
!*******************************************************************


      SUBROUTINE OPTREAL_NEW(IU,LREAD,LL,NQNL,DELQNL,BETA,BETAQ, &
     &          QGAMIN,QCUTIN,RMAXIN,LAUTO,ROPT)
      USE prec
      USE ini
      USE nonl_high
      IMPLICIT REAL(q) (A-H,O-Z)

!-----potential
      INTEGER  IU                  !
      INTEGER  LREAD               ! number of ln-channels
      INTEGER  LL(LREAD)           ! l-quantum number for each channel
      INTEGER  NQNL                ! number of grid points
      REAL(q)  DELQNL              ! spacing in reciprocal space
      REAL(q)  BETA(NQNL,5,LREAD)  ! out: opt projectors in real space
      REAL(q)  BETAQ(0:NQNL,LREAD) ! projectors in rec. space
      REAL(q)  RMAXIN              ! cutoff for real space proj
      REAL(q)  QGAMIN,QCUTIN       ! optimization between [QCUTIN,QGAMIN]
      LOGICAL  LAUTO               ! search for QCUT

!-----gaussian integration
      PARAMETER (M=64)
      REAL(q)  WQ(M),QA(M),RA(M),WR(M),AGAM(M,M)
      EXTERNAL GAUSSI2
      DIMENSION TMP(M),V(M),VP(M)
!-----
      PARAMETER (MJD=20)
      DIMENSION QX(MJD),XI(M,MJD),AMAT(MJD,MJD),IPIV(MJD),CONSTRAINT(MJD)
      DIMENSION BETNEW(MJD)
      DIMENSION AMAT_(MJD,MJD),BMAT(MJD,MJD)

      REAL(q) STORE(5,20)
      INTEGER ISTORE(2,20)
      LOGICAL, PARAMETER :: LBIAS=.TRUE.    ! exp ( - (2 q/q_c)r^2) r^2 weightting or not
      LOGICAL, PARAMETER :: LHIGH=.TRUE.    ! minimize high frequency part
      LOGICAL, PARAMETER :: LCSTR=.FALSE.   ! constrain 1st derivative to 0
!-----spline interpolation
      REAL(q)  WRK(5,NQNL)
      REAL(q) QMESH(NQNL),RMESH(2*NQNL),BETOLD(NQNL)

      PARAMETER (AUTOA=0.529177249_q,RYTOEV=13.605826_q)

      PI = 3.141592653589793238462643E0_q
!-----------------------------------------------------------------------
! allign QCUTP and QGAM with linear grid
!-----------------------------------------------------------------------
      QCUTP=DELQNL*INT(QCUTIN/DELQNL)
      IF (QCUTP>=QCUTIN) QCUTP=QCUTIN
      QGAM=DELQNL*INT(QGAMIN/DELQNL)
      ITYPE = 0

      DO N=1,NQNL
        QMESH(N)=(N-1)*DELQNL
      ENDDO
!-----------------------------------------------------------------------
! set up the linear reciprocal space-mesh
! and the    linear real space-mesh
!-----------------------------------------------------------------------
      RMAX=RMAXIN

qcut_loop: DO

      DELR=RMAX/NQNL
      DO N=1,2*NQNL
        RMESH(N)=(N-1)*DELR
      ENDDO
      RMESH(1)=1E-10_q
!-----------------------------------------------------------------------
! do optimization between 0 and RMAXD
!-----------------------------------------------------------------------
      RMAXD= RMESH(NQNL)

proj_loop: DO I=1,LREAD
!
      QCUT= DELQNL*INT(ABS(QCUTP+DELQNL/2)/DELQNL)

!-----------------------------------------------------------------------
! search for q so that j(q Rc)=0
! and set the constraint matrix
!-----------------------------------------------------------------------
      MJ=MJD-1
      CALL OPT_BEZERO(QX,LL(I),MJ)
      DO K=1,MJ
         QX(K)=QX(K)/RMAXD
         QR  =QX(K)*RMAXD
         CALL SBESSE2(QR, BJ , BJP, LL(I))
         BJP =BJP *QX(K)
         CONSTRAINT(K)=BJP
         IF (QX(K) > QGAM ) THEN
            MJ=K-1
            EXIT
         ENDIF
      ENDDO

      NTEST=QCUT/DELQNL+1E-5_q
      BETMAX=0
      DO N=1,NTEST
         BETMAX=MAX(BETMAX,ABS(BETAQ(N,I)))
      ENDDO
!-----------------------------------------------------------------------
! high frequency part
!-----------------------------------------------------------------------
      AMAT=0
      BMAT=0
      high: IF (LHIGH) THEN
!-----------------------------------------------------------------------
! calculate high frequency penalty function
! A_nm = \int_Q(gam)^Inf X_n (q) X_m (q)  q^2 dq
!-----------------------------------------------------------------------
! A(r,r') = r^2 r'^2 int_0^QGAM jl(qr) ql(qr')  q^2 dq = '
!           pi/2  D(r-r') r*2 - A(r,r')
      IF (LBIAS) THEN
      MD=M
      CALL GAUSSI(GAUSSI2,0.0_q,RMAXD,ITYPE,MD    ,WR ,RA,IFAIL)

      DO MR =1,MD
      R =RA(MR)
      DO MRP=1,MD
      RP=RA(MRP)

        CALL SBESSE2(QGAM*R, BQQ , BQQP, LL(I))
        CALL SBESSE2(QGAM*RP,BQ  , BQP , LL(I))
        IF (R/=RP) THEN
           A= RP**2*R**2*QGAM**2/(RP**2-R**2)*(BQ*BQQP*R-BQQ*BQP*RP)
        ELSE
           QR=R*QGAM
           A= R**3* QGAM**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL(I)+1)*LL(I)/QR*BQ**2)
        ENDIF

        AGAM(MR,MRP)= -A*WR(MR)*WR(MRP)
        IF (MR==MRP)  AGAM(MR,MRP)=AGAM(MR,MRP) + PI/2*R**2*WR(MR)
      ENDDO
      ENDDO
! get functions X(r) on gauss mesh
      DO K=1,MJ
      QQ=QX(K)
      DO MR=1,MD
        CALL SBESSE2(RA(MR)*QQ,A , BQQP, LL(I))
        XI(MR,K)=A
      ENDDO
      ENDDO
!  A_nm = \int_0^R \int_0^R X_n(r) X_n'(r') AGAM(r,r') dr dr'
      AMAT=0
      DO K=1,MJ
      DO KP=1,MJ
         SUM=0
         DO MR=1,MD
         DO MRP=1,MD
            SUM=SUM+AGAM(MR,MRP)*XI(MR,K)*XI(MRP,KP)
         ENDDO
         ENDDO

         AMAT(K,KP)=SUM*((4*PI)**2)
         BMAT(K,KP)=AMAT(K,KP)
      ENDDO
      ENDDO
      ELSE
!-----------------------------------------------------------------------
! here another method to calculate the same penalty function
! less accurate, but faster and in addition modified weights
! are easier to implement here (i.e. it is possible
! to remove for instance mainly low or high frequency components)
! A_nm = \int_Q(gam)^Q(max) X_n (q) X_m (q)  g(q) q^2 dq
! g(q) = 1/q^2 at the moment
!-----------------------------------------------------------------------
      MQMAX =M
      CALL GAUSSI(GAUSSI2,QGAM ,4*QCUT+QGAM,ITYPE,MQMAX ,WQ ,QA,IFAIL)
      DO K=1,MJ
      QQ=QX(K)
      DO MQ=1,MQMAX
      QQP=QA(MQ)

        CALL SBESSE2(RMAXD*QQ, BQQ , BQQP, LL(I))
        CALL SBESSE2(RMAXD*QQP,BQ  , BQP , LL(I))
        IF (QQP/=QQ) THEN
        A= 4*PI*RMAXD**2/(QQP**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*QQP)
        ELSE
        QR=QQ*RMAXD
         A= 4*PI/QQ* RMAXD**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL(I)+1)*LL(I)/QR*BQ**2)
        ENDIF

        XI(MQ,K)=A

      ENDDO
      ENDDO


      DO K=1,MJ
      DO KP=1,MJ
         SUM=0
         DO MQ=1,MQMAX
            SUM=SUM+WQ(MQ)*XI(MQ,K)*XI(MQ,KP)*QA(MQ)**2
         ENDDO
         AMAT(K,KP)=SUM
         BMAT(K,KP)=SUM
      ENDDO
      ENDDO

      ENDIF

      ENDIF high
!-----------------------------------------------------------------------
! setup the  functions X_n(q) on Gauss mesh in the interval [0,Q(cut)]
!-----------------------------------------------------------------------
      MQMAX =M
      CALL GAUSSI(GAUSSI2,0.0_q ,QCUT,ITYPE,MQMAX ,WQ ,QA,IFAIL)
      DO K=1,MJ
      QQ=QX(K)
      DO MQ=1,MQMAX
      QQP=QA(MQ)

        CALL SBESSE2(RMAXD*QQ, BQQ , BQQP, LL(I))
        CALL SBESSE2(RMAXD*QQP,BQ  , BQP , LL(I))
        IF (QQP/=QQ) THEN
        A= 4*PI*RMAXD**2/(QQP**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*QQP)
        ELSE
        QR=QQ*RMAXD
         A= 4*PI/QQ* RMAXD**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL(I)+1)*LL(I)/QR*BQ**2)
        ENDIF

        XI(MQ,K)=A

      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! calculate the required inproducts \int_0^Rmax = < | >
!  AMAT  = AMAT + < X_n    | X_np >
!  TMP   = < X_soll | X_n  >
!-----------------------------------------------------------------------
      WT=1./10.*QCUT**2
      
      DO K=1,MJ
      DO KP=1,MJ
         SUM=0
         DO MQ=1,MQMAX
            IF (LBIAS) THEN
               BIAS=WT ! *EXP(- (1*QA(MQ)/QCUT)**2)
            ELSE
               BIAS=1
            ENDIF
            SUM=SUM+WQ(MQ)*XI(MQ,K)*XI(MQ,KP)*BIAS
         ENDDO
         AMAT(K,KP)=AMAT(K,KP)+SUM
      ENDDO
      ENDDO

      !
      ! actually nonl.F uses a simple cubic interpolation
      ! this spline yields almost identical interpolation as the cubic inter.
      CALL SPLCOF2(QMESH,BETAQ(1,I),NQNL,WRK,1E30_q)

      DO MQ=1,MQMAX
         CALL SPLVAL2(QA(MQ),F,DUMMY,NQNL,WRK)
         TMP(MQ)=F
      ENDDO

      DO K=1,MJ
         SUM=0
         DO MQ=1,MQMAX
            IF (LBIAS) THEN
               BIAS=WT ! *EXP(- (1*QA(MQ)/QCUT)**2)
            ELSE
               BIAS=1
            ENDIF
            SUM=SUM+WQ(MQ)*XI(MQ,K)*TMP(MQ)*BIAS
         ENDDO
         V(K)=SUM
      ENDDO
      V0=0
      DO MQ=1,MQMAX
         IF (LBIAS) THEN
            BIAS=WT ! *EXP(- (1*QA(MQ)/QCUT)**2)
         ELSE
            BIAS=1
         ENDIF
         V0=V0+WQ(MQ)*TMP(MQ)*TMP(MQ)*BIAS
      ENDDO
!-----------------------------------------------------------------------
!  calculate the solution of  V(QQ)=AMAT * BETNEW
!-----------------------------------------------------------------------
! fill in constraint

      AMAT(MJ+1,1:MJ)=CONSTRAINT(1:MJ)
      AMAT(1:MJ,MJ+1)=CONSTRAINT(1:MJ)
      V(MJ+1)=0

!      WRITE(0,*) 'AMAT'
!      WRITE(0,'(5F10.4)') AMAT(1:MJ+1,1:MJ+1)
!      WRITE(0,*) 'V'
!      WRITE(0,'(16F10.4)') V(1:MJ+1)

      AMAT(MJ+1,MJ+1)=0
      AMAT_=AMAT

      BETNEW(1:MJ+1)=V(1:MJ+1)
      IF (LCSTR) THEN
         INFO=0
         CALL DGETRF( MJ+1, MJ+1, AMAT, MJD, IPIV, INFO )
         CALL DGETRS('N', MJ+1, 1, AMAT, MJD, IPIV, BETNEW, MJD, INFO)
      ELSE
         CALL DGETRF( MJ, MJ, AMAT, MJD, IPIV, INFO )
         CALL DGETRS('N', MJ, 1, AMAT, MJD, IPIV, BETNEW, MJD, INFO)
      ENDIF

!      WRITE(0,*) 'solution'
!      WRITE(0,'(16F8.2)') BETNEW(1:MJ+1)
!      WRITE(0,*) 'lowfit ',(DOT_PRODUCT(BETNEW(1:MJ),MATMUL(AMAT_(1:MJ,1:MJ),BETNEW(1:MJ))) &
!           -2*DOT_PRODUCT(BETNEW(1:MJ),V(1:MJ))+V0)/BETMAX/BETMAX
!      WRITE(0,*) 'highfit ',DOT_PRODUCT(BETNEW(1:MJ),MATMUL(BMAT(1:MJ,1:MJ),BETNEW(1:MJ))) &
!             /BETMAX/BETMAX

!-----------------------------------------------------------------------
!  interpolate BETNEW (on Gauss-mesh)
!  and set the final operator in real space
!-----------------------------------------------------------------------
!      WRITE(77,*) 'real space proj'

      DO N=1,NQNL
        QR= RMESH(N)
        SUM=0
        DO K=1,MJ
           CALL SBESSE2(QR*QX(K), BJ , BJP, LL(I))
           SUM=SUM+BJ*BETNEW(K)
        ENDDO
        BETA(N,1,I)=RMESH(N)
        BETOLD(N)  =BETA(N,2,I)
        BETA(N,2,I)= SUM
!        WRITE(77,'(4F14.7)') QR, BETA(N,2,I)
      ENDDO
!========================================================================
!  calculate the transformation of the real space operator
!  and the difference to the original BETA
!  also calculate the difference to BETA calculated without
!   spline-fit in real space (this gives error of spline-fit)
!========================================================================
      MD=M
      CALL GAUSSI(GAUSSI2,0.0_q,RMAXD,ITYPE,MD    ,WR ,RA,IFAIL)

!    get the right hand boundary condition for BETA (brute force is simplest)
      QR=DELR/1000
      SUM=0
      DO K=1,MJ
         CALL SBESSE2(QR*QX(K), BJ , BJP, LL(I))
         SUM=SUM+BJ*BETNEW(K)
      ENDDO
      BOUNDARY=(SUM-BETA(1,2,I))/QR
! new version: exact derivative at sphere boundary
!    get the left hand boundary condition for BETA (brute force is simplest)
      IF (LL(I)/=1) THEN
         BOUNDARY=0
      ENDIF

      QR=RMESH(NQNL)-DELR/1000
      SUM=0
      DO K=1,MJ
         CALL SBESSE2(QR*QX(K), BJ , BJP, LL(I))
         SUM=SUM+BJ*BETNEW(K)
      ENDDO
      BOUNDARYN=(BETA(NQNL,2,I)-SUM)/(DELR/1000)
      
      CALL SPLCOF_NDER(BETA(1,1,I) ,NQNL,NQNL,BOUNDARY,BOUNDARYN)

#ifdef debug
      DO K=1,(NQNL-1)*10+1
         QR=DELR/10*(K-1)
         CALL SPLVAL(QR,F,DUMMY,BETA(1,1,I),NQNL,NQNL)
         WRITE(77,*) QR,F
      ENDDO
      WRITE(77,*)
#endif

      DO K=1,MD
        CALL SPLVAL(RA(K),F,DUMMY,BETA(1,1,I),NQNL,NQNL)
        V (K) = F*RA(K)**2*WR(K)
        SUM=0
        DO KP=1,MJ
           CALL SBESSE2(RA(K)*QX(KP), BJ , BJP, LL(I))
           SUM=SUM+BJ*BETNEW(KP)
        ENDDO
        VP(K) = SUM*RA(K)**2*WR(K)
      ENDDO

      BETMAX=0
      WMAX  =0
      WMAXSP=0
!
! calculate errorfunction W_l(q) using BETA
! on a uniform grid up to QCUT
!
      NTEST=QCUT/DELQNL+1E-5_q
!     WRITE(80,*) 'next'
      DO N=1,NTEST+5
        SUM   =0
        SUMGAU=0
        DO K=1,MD
           CALL SBESSEL(QMESH(N)*RA(K),BJ,LL(I))
           SUM   =SUM   +BJ*V (K)
           SUMGAU=SUMGAU+BJ*VP(K)
        ENDDO
        SUM   =SUM*4*PI
        SUMGAU=SUMGAU*4*PI
        SOLL=0
        IF (N<=NQNL) THEN
           SOLL=BETAQ(N,I)
        ENDIF

        IF (N < NTEST) THEN
           WMAX  =MAX(WMAX  ,ABS(SOLL-SUM))
           WMAXSP=MAX(WMAXSP,ABS(SUM-SUMGAU))
           BETMAX=MAX(BETMAX,ABS(SOLL))
        ENDIF
!        WRITE(80,'(4F14.7)') QMESH(N),SUM-SOLL,SUMGAU-SOLL
      ENDDO

      WMAX_HIGH=0
      NTEST2=QGAM/DELQNL+1E-5_q
      DO N=NTEST2,NTEST2+NTEST
        SUM   =0
        SUMGAU=0
        DO K=1,MD
           CALL SBESSEL((N-1)*DELQNL*RA(K),BJ,LL(I))
           SUM   =SUM   +BJ*V (K)
           SUMGAU=SUMGAU+BJ*VP(K)
        ENDDO
        SUM   =SUM*4*PI
        SUMGAU=SUMGAU*4*PI
        SOLL=0
        WMAX_HIGH  =MAX(WMAX_HIGH  ,ABS(SOLL-SUM))
!        WRITE(80,'(4F14.7)') (N-1)*DELQNL,SUM,SUMGAU
      ENDDO

      ISTORE(1,I)=LL(I)
      ISTORE(2,I)=MJ

      STORE(1,I) =QCUT
      STORE(2,I) =BETMAX
      STORE(3,I) =WMAX/BETMAX
      STORE(4,I) =WMAX_HIGH/BETMAX
      STORE(5,I) =WMAXSP/BETMAX
! check whether required accuracy was reached
      IF ( LAUTO.AND. ABS(WMAX/BETMAX)  > ABS(ROPT) ) THEN
! only first projector must be very accurate
!         IF ( I == 1) THEN
            RMAX = RMAX*1.03
            CYCLE qcut_loop
!         ELSE IF ( LL(I) /= LL(I-1) ) THEN
            RMAX = RMAX*1.03
            CYCLE qcut_loop
!         ENDIF
      ENDIF
!-----------------------------------------------------------------------
! write some important information
!-----------------------------------------------------------------------
 ENDDO proj_loop
 EXIT  qcut_loop
 ENDDO qcut_loop

      RMAXIN=RMAX

      IF (IU>=0) THEN
      WRITE(IU,1) DELQNL*NQNL,ABS(QCUTP),QGAM, &
     &      QCUTP**2*AUTOA**2,QGAM**2*AUTOA**2,RMAX

   2  FORMAT(I4,1X,I6,2F10.3,3E12.2,F6.2)

    1 FORMAT(' Optimization of the real space projectors (new method)'// &
     &       ' maximal supplied QI-value         =',F6.2/ &
     &       ' optimisation between [QCUT,QGAM] = ', &
     &            '[',F6.2,',',F6.2,'] = [',F6.2,',',F6.2,'] Ry ' / &
     &       ' Optimized for a Real-space Cutoff  ',F6.2,' Angstroem'// &
     &       '   l    n(q)    QCUT    ', &
     &       'max X(q) W(low)/X(q) W(high)/X(q)  e(spline) ')

      DO I=1,LREAD
         WRITE(IU,2) ISTORE(:,I),STORE(:,I)
      ENDDO
      ENDIF

      RETURN
      END

!*******************************************************************
!
!  optimize real-space projectors according to a new scheme
!  written by Georg Kresse
!
!  the projectors are written as a sum of spherical Besselfunctions
!  on the sphere boundary the Besselfunctions are zero
!  the coefficients of the Besselfunctions are optimized to
!  reduce high frequency components 
!  
!*******************************************************************


      SUBROUTINE OPTREAL_AUG(LL, NQNL, DELQNL, QGAMIN_, RMAX, NQ, QRETURN, ARETURN)
      USE prec
      USE ini
      USE nonl_high
      IMPLICIT REAL(q) (A-H,O-Z)

!-----potential
      INTEGER  LL                  ! l-quantum number for each channel
      INTEGER  NQNL                ! number of grid points
      REAL(q)  DELQNL              ! spacing in reciprocal space
      REAL(q)  RMAX                ! cutoff for real space proj
      REAL(q)  QGAMIN_              ! optimization beyond QGAMIN
      REAL(q)  QRETURN(*), ARETURN(*)
!-----gaussian integration
      PARAMETER (M=64)
      REAL(q)  WQ(M),QA(M),RA(M),WR(M),AGAM(M,M)
      EXTERNAL GAUSSI2
      DIMENSION TMP(M),V(M)
!-----
      PARAMETER (MJD=20)
      REAL(q)  BETA(NQNL,5)
      DIMENSION QX(MJD),XI(M,MJD),AMAT(MJD,MJD),IPIV(MJD),CONSTRAINT(MJD)
      DIMENSION BETNEW(MJD)
      DIMENSION AMAT_(MJD,MJD),BMAT(MJD,MJD)

      REAL(q) STORE(5,20)
      INTEGER ISTORE(2,20)
      LOGICAL, PARAMETER :: LCSTR=.FALSE.   ! constrain 1st derivative to 0
!-----spline interpolation
      REAL(q)  WRK(5,NQNL)
      REAL(q)  QMESH(NQNL),RMESH(2*NQNL)

      PARAMETER (AUTOA=0.529177249_q,RYTOEV=13.605826_q)

      PI = 3.141592653589793238462643E0_q
!-----------------------------------------------------------------------
! allign QGAM with linear grid
!-----------------------------------------------------------------------
      QGAMIN=20/RMAX
!      QGAMIN=QGAMIN_

      QGAM=DELQNL*INT(QGAMIN/DELQNL)
      ITYPE = 0

      DO N=1,NQNL
        QMESH(N)=(N-1)*DELQNL
      ENDDO
!-----------------------------------------------------------------------
! set up the linear reciprocal space-mesh
! and the    linear real space-mesh
!-----------------------------------------------------------------------
      DELR=RMAX/NQNL
      DO N=1,2*NQNL
        RMESH(N)=(N-1)*DELR
      ENDDO
      RMESH(1)=1E-10_q
!-----------------------------------------------------------------------
! do optimization between 0 and RMAXD
!-----------------------------------------------------------------------
      RMAXD= RMESH(NQNL)
!-----------------------------------------------------------------------
! search for q so that j(q Rc)=0
! and set the constraint matrix
!-----------------------------------------------------------------------
      MJ=MJD-1
      CALL OPT_BEZERO(QX,LL,MJ)
      DO K=1,MJ
         QX(K)=QX(K)/RMAXD
         QR  =QX(K)*RMAXD
         CALL SBESSE2(QR, BJ , BJP, LL)
         BJP =BJP *QX(K)
         CONSTRAINT(K)=BJP
         IF (QX(K) > QGAM .AND. K>=4) THEN
            MJ=K-1
            EXIT
         ENDIF
      ENDDO
!-----------------------------------------------------------------------
! high frequency part
!-----------------------------------------------------------------------
      AMAT=0
      BMAT=0
!-----------------------------------------------------------------------
! calculate high frequency penalty function
! A_nm = \int_Q(gam)^Inf X_n (q) X_m (q)  q^2 dq
!-----------------------------------------------------------------------
! A(r,r') = r^2 r'^2 int_0^QGAM jl(qr) ql(qr')  q^2 dq = '
!           pi/2  D(r-r') r*2 - A(r,r')
      MD=M
      CALL GAUSSI(GAUSSI2,0.0_q,RMAXD,ITYPE,MD    ,WR ,RA,IFAIL)

      DO MR =1,MD
      R =RA(MR)
      DO MRP=1,MD
      RP=RA(MRP)

        CALL SBESSE2(QGAM*R, BQQ , BQQP, LL)
        CALL SBESSE2(QGAM*RP,BQ  , BQP , LL)
        IF (R/=RP) THEN
           A= RP**2*R**2*QGAM**2/(RP**2-R**2)*(BQ*BQQP*R-BQQ*BQP*RP)
        ELSE
           QR=R*QGAM
           A= R**3* QGAM**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL+1)*LL/QR*BQ**2)
        ENDIF

        AGAM(MR,MRP)= -A*WR(MR)*WR(MRP)
        IF (MR==MRP)  AGAM(MR,MRP)=AGAM(MR,MRP) + PI/2*R**2*WR(MR)
      ENDDO
      ENDDO
! get functions X(r) on gauss mesh
      DO K=1,MJ
      QQ=QX(K)
      DO MR=1,MD
        CALL SBESSE2(RA(MR)*QQ,A , BQQP, LL)
        XI(MR,K)=A
      ENDDO
      ENDDO
!  A_nm = \int_0^R \int_0^R X_n(r) X_n'(r') AGAM(r,r') dr dr'
      AMAT=0
      DO K=1,MJ
      DO KP=1,MJ
         SUM=0
         DO MR=1,MD
         DO MRP=1,MD
            SUM=SUM+AGAM(MR,MRP)*XI(MR,K)*XI(MRP,KP)
         ENDDO
         ENDDO

         AMAT(K,KP)=SUM*((4*PI)**2)
         BMAT(K,KP)=AMAT(K,KP)
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! setup the  functions X_n(q) on Gauss mesh in the interval [0,Q(cut)]
!-----------------------------------------------------------------------
      MQMAX =M
      CALL GAUSSI(GAUSSI2,DELQNL-1E-3 ,DELQNL+1E-3,ITYPE,MQMAX ,WQ ,QA,IFAIL)

      DO K=1,MJ
      QQ=QX(K)
      DO MQ=1,MQMAX
      QQP=QA(MQ)

        CALL SBESSE2(RMAXD*QQ, BQQ , BQQP, LL)
        CALL SBESSE2(RMAXD*QQP,BQ  , BQP , LL)
        IF (QQP/=QQ) THEN
        A= 4*PI*RMAXD**2/(QQP**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*QQP)
        ELSE
        QR=QQ*RMAXD
         A= 4*PI/QQ* RMAXD**2/2*(BQ*BQP+QR*BQP**2+ &
     &                      QR*BQ**2- (LL+1)*LL/QR*BQ**2)
        ENDIF

        XI(MQ,K)=A

      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! calculate the required inproducts \int_0^Rmax = < | >
!  AMAT  = AMAT + < X_n    | X_np >
!  TMP   = < X_soll | X_n  >
!-----------------------------------------------------------------------
      DO K=1,MJ
      DO KP=1,MJ
         SUM=0
         DO MQ=1,MQMAX
            BIAS=1
            SUM=SUM+WQ(MQ)*XI(MQ,K)*XI(MQ,KP)*BIAS
         ENDDO
         AMAT(K,KP)=AMAT(K,KP)+SUM
      ENDDO
      ENDDO

      DO MQ=1,MQMAX
         TMP(MQ)=QA(MQ)**LL
      ENDDO
      DO K=1,MJ
         SUM=0
         DO MQ=1,MQMAX
            BIAS=1
            SUM=SUM+WQ(MQ)*XI(MQ,K)*TMP(MQ)*BIAS
         ENDDO
         V(K)=SUM
      ENDDO
      V0=0
      DO MQ=1,MQMAX
         BIAS=1
         V0=V0+WQ(MQ)*TMP(MQ)*TMP(MQ)*BIAS
      ENDDO
!-----------------------------------------------------------------------
!  calculate the solution of  V(QQ)=AMAT * BETNEW
!-----------------------------------------------------------------------
! fill in constraint

      AMAT(MJ+1,1:MJ)=CONSTRAINT(1:MJ)
      AMAT(1:MJ,MJ+1)=CONSTRAINT(1:MJ)
      V(MJ+1)=0

!      WRITE(0,*) 'AMAT'
!      WRITE(0,'(5F10.4)') AMAT(1:MJ+1,1:MJ+1)
!      WRITE(0,*) 'V'
!      WRITE(0,'(16F10.4)') V(1:MJ+1)

      AMAT(MJ+1,MJ+1)=0
      AMAT_=AMAT

      BETNEW(1:MJ+1)=V(1:MJ+1)
      IF (LCSTR) THEN
         INFO=0
         CALL DGETRF( MJ+1, MJ+1, AMAT, MJD, IPIV, INFO )
         CALL DGETRS('N', MJ+1, 1, AMAT, MJD, IPIV, BETNEW, MJD, INFO)
      ELSE
         CALL DGETRF( MJ, MJ, AMAT, MJD, IPIV, INFO )
         CALL DGETRS('N', MJ, 1, AMAT, MJD, IPIV, BETNEW, MJD, INFO)
      ENDIF

!      WRITE(0,*) 'solution'
!      WRITE(0,'(16F8.4)') BETNEW(1:MJ)/BETNEW(1)
!      WRITE(0,*) 'lowfit ',(DOT_PRODUCT(BETNEW(1:MJ),MATMUL(AMAT_(1:MJ,1:MJ),BETNEW(1:MJ))) &
!           -2*DOT_PRODUCT(BETNEW(1:MJ),V(1:MJ))+V0)
!      WRITE(0,*) 'highfit ',DOT_PRODUCT(BETNEW(1:MJ),MATMUL(BMAT(1:MJ,1:MJ),BETNEW(1:MJ)))

!-----------------------------------------------------------------------
!  interpolate BETNEW (on Gauss-mesh)
!  and set the final operator in real space
!-----------------------------------------------------------------------
      MD=M
      CALL GAUSSI(GAUSSI2,0.0_q,RMAXD,ITYPE,MD    ,WR ,RA,IFAIL)

      ! proper normalization
      SUM=0
      DO K=1,MD
        DO KP=1,MJ
           CALL SBESSE2(RA(K)*QX(KP), BJ , BJP, LL)
           SUM=SUM+BJ*BETNEW(KP)*RA(K)**(2+LL)*WR(K)
        ENDDO
      ENDDO

      BETNEW(1:MJ)=BETNEW(1:MJ)/SUM

!      WRITE(77,*) 'real space proj'

      DO N=1,NQNL
        QR= RMESH(N)
        SUM=0
        DO K=1,MJ
           CALL SBESSE2(QR*QX(K), BJ , BJP, LL)
           SUM=SUM+BJ*BETNEW(K)
        ENDDO
        BETA(N,1)=RMESH(N)
        BETA(N,2)= SUM
!        WRITE(77,'(4F14.7)') QR, BETA(N,2)
      ENDDO
!========================================================================
!  calculate the transformation of the real space operator
!  and the difference to the original BETA
!  also calculate the difference to BETA calculated without
!   spline-fit in real space (this gives error of spline-fit)
!========================================================================

!    get the right hand boundary condition for BETA (brute force is simplest)
      QR=DELR/1000

      SUM=0
      DO K=1,MJ
         CALL SBESSE2(QR*QX(K), BJ , BJP, LL)
         SUM=SUM+BJ*BETNEW(K)
      ENDDO
      BOUNDARY=(SUM-BETA(1,2))/QR

!    get the left hand boundary condition for BETA (brute force is simplest)
      IF (LL/=1) THEN
         BOUNDARY=0
      ENDIF

      QR=RMESH(NQNL)-DELR/1000
      SUM=0
      DO K=1,MJ
         CALL SBESSE2(QR*QX(K), BJ , BJP, LL)
         SUM=SUM+BJ*BETNEW(K)
      ENDDO
      BOUNDARYN=(BETA(NQNL,2)-SUM)/(DELR/1000)
         
      CALL SPLCOF_NDER(BETA(1,1) ,NQNL,NQNL,BOUNDARY,BOUNDARYN)

      QRETURN(1:MJ)=QX(1:MJ)
      ARETURN(1:MJ)=BETNEW(1:MJ)
      NQ=MJ

#ifdef debug
      DO K=1,(NQNL-1)*10+1
         QR=DELR/10*(K-1)
         CALL SPLVAL(QR,F,DUMMY,BETA(1,1),NQNL,NQNL)
         WRITE(77,*) QR,F
      ENDDO
      WRITE(77,*)
#endif
      ! set V to the real space projector calculated directly using Besselfunction
      ! (use Gauss grid)
      DO K=1,MD
        SUM=0
        DO KP=1,MJ
           CALL SBESSE2(RA(K)*QX(KP), BJ , BJP, LL)
           SUM=SUM+BJ*BETNEW(KP)
        ENDDO
        V(K) = SUM*RA(K)**2*WR(K)
      ENDDO

      WMAX  =0
      WMAXSP=0
!
! calculate reciprocal space result
! on a uniform grid up to 2*QGAM
!
      NTEST2=QGAM/DELQNL+1E-5_q

      DO N=1,2*NTEST2
        SUM   =0
        DO K=1,MD
           CALL SBESSEL((N-1)*DELQNL*RA(K),BJ,LL)
           SUM   =SUM   +BJ*V (K)
        ENDDO
        SUM   =SUM*4*PI
!        WRITE(80,'(4F14.7)') (N-1)*DELQNL,SUM
      ENDDO

!-----------------------------------------------------------------------
! write some important information
!-----------------------------------------------------------------------
    END SUBROUTINE OPTREAL_AUG

!*******************************************************************
!  SUBROUTINE BEZERO
!  searches for NQ zeros j(qr)
!  i/o:
!         XNULL(NQ) result
!         L           quantum number l
!  great full spaghetti code (written by gK)
!********************************************************************

    SUBROUTINE OPT_BEZERO(XNULL,L,NQ)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (STEP=.1_q, BREAK= 1E-10_q)
      DIMENSION XNULL(NQ)
! initialization
      X=STEP
      N=0
! entry point for next q_n
  30  CALL SBESSE2(X, BJ1, DUMMY,  L)
! coarse search
  10  X=X+STEP
      CALL SBESSE2(X, BJ2, DUMMY,  L)
! found one point
      IF(BJ1*BJ2 < 0) THEN
        ETA=0.0_q
! interval bisectioning
        SSTEP=STEP
        XX   =X
  20    SSTEP=SSTEP/2
        IF (BJ1*BJ2 < 0) THEN
          XX=XX-SSTEP
        ELSE
          XX=XX+SSTEP
        ENDIF
        CALL SBESSE2(XX, BJ2, DUMMY,  L)
        IF (SSTEP > BREAK) GOTO 20

        N=N+1
        XNULL(N)=XX
        IF (N == NQ) RETURN
        GOTO 30
      ENDIF
      GOTO 10

    END SUBROUTINE

!**************** SUBROUTINE SPLCOF2 ***********************************
!
!  subroutine for calculating spline-coefficients
!  using the routines of the book 'numerical  recipes'
!  on input X must contain x-values
!           F must contain function-values
!  for the first point zero derivative is assumed
!  for point N natural boundary-conditions are used
!  (SPLCOF uses natural boundary-conditions  for both boundaries)
!***********************************************************************

      SUBROUTINE SPLCOF2(X,F,N,P,Y1P)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(5,N),F(N),X(N)
!
!     determination of spline coefficients
!     ------------------------------------
!     f = ((d*dx+c)*dx+b)*dx+a
!         between adjacent x - values
!
!     result
!     P-ARRAY
!     P(1,I) = X(I)
!     P(2,I) = A(I) = F(I)
!     P(3,I) = B(I)
!     P(4,I) = C(I)
!     P(5,I) = D(I)
!
      DO 10 I=1,N
        P(2,I)=F(I)
        P(1,I)=X(I)
   10 CONTINUE

      IF (Y1P> .99E30_q) THEN
        P(4,1)=0.0_q
        P(3,1)=0.0_q
      ELSE
        P(4,1)=-.5_q
        P(3,1)=(3._q/(P(1,2)-P(1,1)))*((P(2,2)-P(2,1))/ &
     &             (P(1,2)-P(1,1))-Y1P)
      ENDIF

      DO 20 I=2,N-1
        S=(P(1,I)-P(1,I-1))/(P(1,I+1)-P(1,I-1))
        R=S*P(4,I-1)+2._q
        P(4,I)=(S-1._q)/R
        P(3,I)=(6*((P(2,I+1)-P(2,I))/(P(1,I+1)-P(1,I))- &
     &          (P(2,I)-P(2,I-1))/(P(1,I)-P(1,I-1)))/ &
     &          (P(1,I+1)-P(1,I-1))-S*P(3,I-1))/R
   20 CONTINUE

      P(4,N)=0.0_q
      P(3,N)=0.0_q
!
      DO 30 I=N-1,1,-1
        P(4,I)=P(4,I)*P(4,I+1)+P(3,I)
  30  CONTINUE
!
      DO 50 I=1,N-1
        S= P(1,I+1)-P(1,I)
        R=(P(4,I+1)-P(4,I))/6
        P(5,I)=R/S
        P(4,I)=P(4,I)/2.0_q
        P(3,I)=(P(2,I+1)-P(2,I))/S-(P(4,I)+R)*S
 50   CONTINUE
      RETURN
      END

      SUBROUTINE SPLVAL2(X,F,DER,N,P)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(5,N)
!     SPLINE INTERPOLATION
!     --------------------
!     INPUT
!     X-VALUE
!     P-ARRAY (RESULTING FROM CALL SPLCOF)
!
!     RESULT
!     F = INTERPOLATED VALUE
!     DER = FIRST DERIVATIVE
!     IF X OUTSIDE, EXTRAPOLATION FROM
!     FIRST OR LAST POINT IS DONE
!
      I=1
      J=N
      IF (X   <P(1,I)) GO TO 60
      IF (X   <P(1,J)) GO TO 70
      K=J-1
      GOTO 90
   60 K=1
      GOTO 90
   70 K=(I+J)/2
      IF(I==K) GOTO 90
      IF (X   <P(1,K)) GO TO 80
      I=K
      GOTO 70
   80 J=K
      GOTO 70
!
   90 DX=X   -P(1,K)
      F   =((P(5,K)*DX+P(4,K))*DX+P(3,K))*DX+P(2,K)
      DER=(3.0_q*P(5,K)*DX+2.0_q*P(4,K))*DX+P(3,K)
      END


!*******************************************************************
!     SBESSEL
!     caculates the spherical besselfunction and its derivate
!     Input/Output
!     X     argument
!     BJ    value of besselfunctions
!     LMAX  quantum number l
!
!      x>DEL: recursion M(k)= -M(k-2)+(2k-1)M(k-1)/x
!                       M(0) = sin(x) , M(1) = cos(x)/x - sin(x)/x/x
!
!      x<DEL: taylor expansion
!
!********************************************************************

      SUBROUTINE SBESSEL( X, BJ, LMAX)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (DEL=1._q)
      REAL(q) X,BJ,BJP

      IF ( X< DEL) THEN
         BJ  = SBESSITER(X,LMAX)
      ELSE
         BJ = SIN(X)/X
         BJP=(-COS(X)+BJ)/X

         DO L=2,LMAX+1
            TEMP= -BJ + (2*L-1)*BJP/ X
            BJ  = BJP
            BJP = TEMP
         ENDDO
      ENDIF
      RETURN
      END

!--------------------------------------------------------------------

      FUNCTION SBESSITER(X,LMAX)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) SBESSITER
      PARAMETER ( ABBRUCH = 1E-16_q)
      PARAMETER ( ITERMAX = 40 )

      X2= X*X/2
      FAK = 1._q
      DO L=1,LMAX
         FAK = FAK*X /(2*L+1)
      ENDDO

      SUM = 1._q
      D   = 1._q
      DO I=1,ITERMAX
         D = -D* X2/(I*(2*(LMAX+I)+1))
         IF ( ABS(D)< ABBRUCH) GOTO 230
         SUM = SUM + D
      ENDDO

      WRITE(*,*) 'ERROR: SBESSELITER : nicht konvergent'
      STOP

 230  SBESSITER= FAK * SUM
      RETURN
      END

      SUBROUTINE SBESSE2( X, BJ, BJP , LMAX)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (DEL=1._q)
      REAL(q) X,BJ,BJP

      IF ( X< DEL) THEN
         BJ  = SBESSITER(X,LMAX)
         BJP = SBESSITER(X,LMAX+1)
         BJP = -BJP+ BJ*LMAX/X
      ELSE
         BJ = SIN(X)/X
         BJP=(-COS(X)+BJ)/X

         DO L=2,LMAX+1
            TEMP= -BJ + (2*L-1)*BJP/ X
            BJ  = BJP
            BJP = TEMP
         ENDDO
         BJP = -BJP+ BJ*LMAX/X

      ENDIF
      RETURN
      END

      SUBROUTINE SBESSE3( X, BJ, BJP, BJPP, L)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) X,BJ,BJP,BJPP

      CALL SBESSE2(X, BJ1,BJP1, L+1)
      CALL SBESSE2(X, BJ, BJP, L)
      BJPP= -BJP1 + BJP*L/X -BJ*L/X/X
      RETURN
      END


!
!  code required for the gaussian integration
!  taken from some very old NAG release
!
      SUBROUTINE GAUSSI2(A, B, ITYPE, NPTS, WEIGHT, ABSCIS, IFAIL)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!     MARK 7 RELEASE. NAG COPYRIGHT 1978.
!
!     RETURNS WEIGHTS AND PIVOTS FOR ONE GAUSS-LEGENDRE FORMULA IF
!     STORED
!     IFAIL = 1 - THE NPTS RULE IS NOT AMONG THOSE STORED
!     ( WEIGHT,ABSCIS EVALUATED FOR LARGEST VALID NPTS LESS THAN
!     REQUESTED VALUE)
!
!     THE WEIGHTS AND ABSCISSAE RETURNED DEPEND ON A AND B.
!     THOSE STORED ARE WEIGHTS AND ABSCISSAE FOR A=-1,B=+1
!     THOSE RETURNED FOR GENERAL (A,B) ARE RELATED TO THOSE STORED
!     BY
!     W(A,B) = 0.5 * (B-A) * W(-1,+1)
!     X(A,B) = 0.5 * (B-A) * X(-1,+1) + 0.5 * (A+B)
!
!     .. SCALAR ARGUMENTS ..
      REAL(q) A, B
      INTEGER IFAIL, ITYPE, NPTS
!     .. ARRAY ARGUMENTS ..
      REAL(q) ABSCIS(NPTS), WEIGHT(NPTS)
!     ..
!     .. LOCAL SCALARS ..
      REAL(q) HFRNGE, PNTMID
      INTEGER I, IIJJ, N, NL, NN, NPTSA
!     .. LOCAL ARRAYS ..
      REAL(q) ABST(136), WTST(136)
      INTEGER NSTOR(16)
!     ..
      DATA WTST( 1),WTST( 2),WTST( 3),WTST( 4),WTST( 5)/ &
     &+0.200000000000000000000000000000E1_q, &
     &+0.100000000000000000000000000000E1_q, &
     &+0.555555555555555555555555555555E0_q, &
     &+0.888888888888888888888888888888E0_q, &
     &+0.347854845137453857373063949221E0_q/
      DATA WTST( 6),WTST( 7),WTST(08),WTST(09),WTST(10)/ &
     &+0.652145154862546142626936050778E0_q, &
     &+0.236926885056189087514264040719E0_q, &
     &+0.478628670499366468041291514835E0_q, &
     &+0.568888888888888888888888888888E0_q, &
     &+0.171324492379170345040296142172E0_q/
      DATA WTST(11),WTST(12),WTST(13),WTST(14),WTST(15)/ &
     &+0.360761573048138607569833513837E0_q, &
     &+0.467913934572691047389870343989E0_q, &
     &+0.101228536290376259152531354309E0_q, &
     &+0.222381034453374470544355994426E0_q, &
     &+0.313706645877887287337962201986E0_q/
      DATA WTST(16),WTST(17),WTST(18),WTST(19),WTST(20)/ &
     &+0.362683783378361982965150449277E0_q, &
     &+0.666713443086881375935688098933E-1_q, &
     &+0.149451349150580593145776339657E0_q, &
     &+0.219086362515982043995534934228E0_q, &
     &+0.269266719309996355091226921569E0_q/
      DATA WTST(21),WTST(22),WTST(23),WTST(24),WTST(25)/ &
     &+0.295524224714752870173892994651E0_q, &
     &+0.471753363865118271946159614850E-1_q, &
     &+0.106939325995318430960254718193E0_q, &
     &+0.160078328543346226334652529543E0_q, &
     &+0.203167426723065921749064455809E0_q/
      DATA WTST(26),WTST(27),WTST(28),WTST(29),WTST(30)/ &
     &+0.233492536538354808760849898924E0_q, &
     &+0.249147045813402785000562436042E0_q, &
     &+0.351194603317518630318328761381E-1_q, &
     &+0.801580871597602098056332770628E-1_q, &
     &+0.121518570687903184689414809072E0_q/
      DATA WTST(31),WTST(32),WTST(33),WTST(34),WTST(35)/ &
     &+0.157203167158193534569601938623E0_q, &
     &+0.185538397477937813741716590125E0_q, &
     &+0.205198463721295603965924065661E0_q, &
     &+0.215263853463157790195876443316E0_q, &
     &+0.271524594117540948517805724560E-1_q/
      DATA WTST(36),WTST(37),WTST(38),WTST(39),WTST(40)/ &
     &+0.622535239386478928628438369943E-1_q, &
     &+0.951585116824927848099251076022E-1_q, &
     &+0.124628971255533872052476282192E0_q, &
     &+0.149595988816576732081501730547E0_q, &
     &+0.169156519395002538189312079030E0_q/
      DATA WTST(41),WTST(42),WTST(43),WTST(44),WTST(45)/ &
     &+0.182603415044923588866763667969E0_q, &
     &+0.189450610455068496285396723208E0_q, &
     &+0.176140071391521183118619623518E-1_q, &
     &+0.406014298003869413310399522749E-1_q, &
     &+0.626720483341090635695065351870E-1_q/
      DATA WTST(46),WTST(47),WTST(48),WTST(49),WTST(50)/ &
     &+0.832767415767047487247581432220E-1_q, &
     &+0.101930119817240435036750135480E0_q, &
     &+0.118194531961518417312377377711E0_q, &
     &+0.131688638449176626898494499748E0_q, &
     &+0.142096109318382051329298325067E0_q/
      DATA WTST(51),WTST(52),WTST(53),WTST(54),WTST(55)/ &
     &+0.149172986472603746787828737001E0_q, &
     &+0.152753387130725850698084331955E0_q, &
     &+0.123412297999871995468056670700E-1_q, &
     &+0.285313886289336631813078159518E-1_q, &
     &+0.442774388174198061686027482113E-1_q/
      DATA WTST(56),WTST(57),WTST(58),WTST(59),WTST(60)/ &
     &+0.592985849154367807463677585001E-1_q, &
     &+0.733464814110803057340336152531E-1_q, &
     &+0.861901615319532759171852029837E-1_q, &
     &+0.976186521041138882698806644642E-1_q, &
     &+0.107444270115965634782577342446E0_q/
      DATA WTST(61),WTST(62),WTST(63),WTST(64),WTST(65)/ &
     &+0.115505668053725601353344483906E0_q, &
     &+0.121670472927803391204463153476E0_q, &
     &+0.125837456346828296121375382511E0_q, &
     &+0.127938195346752156974056165224E0_q, &
     &+0.701861000947009660040706373885E-2_q/
      DATA WTST(66),WTST(67),WTST(68),WTST(69),WTST(70)/ &
     &+0.162743947309056706051705622063E-1_q, &
     &+0.253920653092620594557525897892E-1_q, &
     &+0.342738629130214331026877322523E-1_q, &
     &+0.428358980222266806568786466061E-1_q, &
     &+0.509980592623761761961632446895E-1_q/
      DATA WTST( 71),WTST( 72),WTST( 73),WTST( 74),WTST( 75)/ &
     &+0.586840934785355471452836373001E-1_q, &
     &+0.658222227763618468376500637069E-1_q, &
     &+0.723457941088485062253993564784E-1_q, &
     &+0.781938957870703064717409188283E-1_q, &
     &+0.833119242269467552221990746043E-1_q/
      DATA WTST( 76),WTST( 77),WTST( 78),WTST( 79),WTST( 80)/ &
     &+0.876520930044038111427714627518E-1_q, &
     &+0.911738786957638847128685771116E-1_q, &
     &+0.938443990808045656391802376681E-1_q, &
     &+0.956387200792748594190820022041E-1_q, &
     &+0.965400885147278005667648300635E-1_q/
      DATA WTST( 81),WTST( 82),WTST( 83),WTST( 84),WTST( 85)/ &
     &+0.315334605230583863267731154389E-2_q, &
     &+0.732755390127626210238397962178E-2_q, &
     &+0.114772345792345394895926676090E-1_q, &
     &+0.155793157229438487281769558344E-1_q, &
     &+0.196161604573555278144607196522E-1_q/
      DATA WTST( 86),WTST( 87),WTST( 88),WTST( 89),WTST( 90)/ &
     &+0.235707608393243791405193013784E-1_q, &
     &+0.274265097083569482000738362625E-1_q, &
     &+0.311672278327980889020657568463E-1_q, &
     &+0.347772225647704388925485859638E-1_q, &
     &+0.382413510658307063172172565237E-1_q/
      DATA WTST( 91),WTST( 92),WTST( 93),WTST( 94),WTST( 95)/ &
     &+0.415450829434647492140588223610E-1_q, &
     &+0.446745608566942804194485871258E-1_q, &
     &+0.476166584924904748259066234789E-1_q, &
     &+0.503590355538544749578076190878E-1_q, &
     &+0.528901894851936670955050562646E-1_q/
      DATA WTST( 96),WTST( 97),WTST( 98),WTST( 99),WTST(100)/ &
     &+0.551995036999841628682034951916E-1_q, &
     &+0.572772921004032157051502346847E-1_q, &
     &+0.591148396983956357464748174335E-1_q, &
     &+0.607044391658938800529692320278E-1_q, &
     &+0.620394231598926639041977841375E-1_q/
      DATA WTST(101),WTST(102),WTST(103),WTST(104),WTST(105)/ &
     &+0.631141922862540256571260227502E-1_q, &
     &+0.639242385846481866239062018255E-1_q, &
     &+0.644661644359500822065041936577E-1_q, &
     &+0.647376968126839225030249387365E-1_q, &
     &+0.178328072169643294729607914497E-2_q/
      DATA WTST(106),WTST(107),WTST(108),WTST(109),WTST(110)/ &
     &+0.414703326056246763528753572855E-2_q, &
     &+0.650445796897836285611736039998E-2_q, &
     &+0.884675982636394772303091465973E-2_q, &
     &+0.111681394601311288185904930192E-1_q, &
     &+0.134630478967186425980607666859E-1_q/
      DATA WTST(111),WTST(112),WTST(113),WTST(114),WTST(115)/ &
     &+0.157260304760247193219659952975E-1_q, &
     &+0.179517157756973430850453020011E-1_q, &
     &+0.201348231535302093723403167285E-1_q, &
     &+0.222701738083832541592983303841E-1_q, &
     &+0.243527025687108733381775504090E-1_q/
      DATA WTST(116),WTST(117),WTST(118),WTST(119),WTST(120)/ &
     &+0.263774697150546586716917926252E-1_q, &
     &+0.283396726142594832275113052002E-1_q, &
     &+0.302346570724024788679740598195E-1_q, &
     &+0.320579283548515535854675043478E-1_q, &
     &+0.338051618371416093915654821107E-1_q/
      DATA WTST(121),WTST(122),WTST(123),WTST(124),WTST(125)/ &
     &+0.354722132568823838106931467152E-1_q, &
     &+0.370551285402400460404151018095E-1_q, &
     &+0.385501531786156291289624969468E-1_q, &
     &+0.399537411327203413866569261283E-1_q, &
     &+0.412625632426235286101562974736E-1_q/
      DATA WTST(126),WTST(127),WTST(128),WTST(129),WTST(130)/ &
     &+0.424735151236535890073397679088E-1_q, &
     &+0.435837245293234533768278609737E-1_q, &
     &+0.445905581637565630601347100309E-1_q, &
     &+0.454916279274181444797709969712E-1_q, &
     &+0.462847965813144172959532492322E-1_q/
      DATA WTST(131),WTST(132),WTST(133),WTST(134),WTST(135)/ &
     &+0.469681828162100173253262857545E-1_q, &
     &+0.475401657148303086622822069442E-1_q, &
     &+0.479993885964583077281261798713E-1_q, &
     &+0.483447622348029571697695271580E-1_q, &
     &+0.485754674415034269347990667839E-1_q/
      DATA WTST(136)/ &
     &+0.486909570091397203833653907347E-1_q/
      DATA ABST( 1),ABST( 2),ABST( 3),ABST( 4),ABST( 5)/ &
     &+0.000000000000000000000000000000E0_q, &
     &+0.577350269189625764509148780501E0_q, &
     &+0.774596669241483377035853079956E0_q, &
     &+0.000000000000000000000000000000E0_q, &
     &+0.861136311594052575223946488892E0_q/
      DATA ABST( 6),ABST( 7),ABST( 8),ABST( 9),ABST(10)/ &
     &+0.339981043584856264802665759103E0_q, &
     &+0.906179845938663992797626878299E0_q, &
     &+0.538469310105683091036314420700E0_q, &
     &+0.000000000000000000000000000000E0_q, &
     &+0.932469514203152027812301554493E0_q/
      DATA ABST(11),ABST(12),ABST(13),ABST(14),ABST(15)/ &
     &+0.661209386466264513661399595019E0_q, &
     &+0.238619186083196908630501721680E0_q, &
     &+0.960289856497536231683560868569E0_q, &
     &+0.796666477413626739591553936475E0_q, &
     &+0.525532409916328985817739049189E0_q/
      DATA ABST(16),ABST(17),ABST(18),ABST(19),ABST(20)/ &
     &+0.183434642495649804939476142360E0_q, &
     &+0.973906528517171720077964012084E0_q, &
     &+0.865063366688984510732096688423E0_q, &
     &+0.679409568299024406234327365114E0_q, &
     &+0.433395394129247190799265943165E0_q/
      DATA ABST(21),ABST(22),ABST(23),ABST(24),ABST(25)/ &
     &+0.148874338981631210884826001129E0_q, &
     &+0.981560634246719250690549090149E0_q, &
     &+0.904117256370474856678465866119E0_q, &
     &+0.769902674194304687036893833212E0_q, &
     &+0.587317954286617447296702418940E0_q/
      DATA ABST(26),ABST(27),ABST(28),ABST(29),ABST(30)/ &
     &+0.367831498998180193752691536643E0_q, &
     &+0.125233408511468915472441369463E0_q, &
     &+0.986283808696812338841597266704E0_q, &
     &+0.928434883663573517336391139377E0_q, &
     &+0.827201315069764993189794742650E0_q/
      DATA ABST(31),ABST(32),ABST(33),ABST(34),ABST(35)/ &
     &+0.687292904811685470148019803019E0_q, &
     &+0.515248636358154091965290718551E0_q, &
     &+0.319112368927889760435671824168E0_q, &
     &+0.108054948707343662066244650219E0_q, &
     &+0.989400934991649932596154173450E0_q/
      DATA ABST(36),ABST(37),ABST(38),ABST(39),ABST(40)/ &
     &+0.944575023073232576077988415534E0_q, &
     &+0.865631202387831743880467897712E0_q, &
     &+0.755404408355003033895101194847E0_q, &
     &+0.617876244402643748446671764048E0_q, &
     &+0.458016777657227386342419442983E0_q/
      DATA ABST(41),ABST(42),ABST(43),ABST(44),ABST(45)/ &
     &+0.281603550779258913230460501460E0_q, &
     &+0.950125098376374401853193354249E-1_q, &
     &+0.993128599185094924786122388471E0_q, &
     &+0.963971927277913791267666131197E0_q, &
     &+0.912234428251325905867752441203E0_q/
      DATA ABST(46),ABST(47),ABST(48),ABST(49),ABST(50)/ &
     &+0.839116971822218823394529061701E0_q, &
     &+0.746331906460150792614305070355E0_q, &
     &+0.636053680726515025452836696226E0_q, &
     &+0.510867001950827098004364050955E0_q, &
     &+0.373706088715419560672548177024E0_q/
      DATA ABST(51),ABST(52),ABST(53),ABST(54),ABST(55)/ &
     &+0.227785851141645078080496195368E0_q, &
     &+0.765265211334973337546404093988E-1_q, &
     &+0.995187219997021360179997409700E0_q, &
     &+0.974728555971309498198391993008E0_q, &
     &+0.938274552002732758523649001708E0_q/
      DATA ABST(56),ABST(57),ABST(58),ABST(59),ABST(60)/ &
     &+0.886415527004401034213154341982E0_q, &
     &+0.820001985973902921953949872669E0_q, &
     &+0.740124191578554364243828103099E0_q, &
     &+0.648093651936975569252495786910E0_q, &
     &+0.545421471388839535658375617218E0_q/
      DATA ABST(61),ABST(62),ABST(63),ABST(64),ABST(65)/ &
     &+0.433793507626045138487084231913E0_q, &
     &+0.315042679696163374386793291319E0_q, &
     &+0.191118867473616309158639820757E0_q, &
     &+0.640568928626056260850430826247E-1_q, &
     &+0.997263861849481563544981128665E0_q/
      DATA ABST(66),ABST(67),ABST(68),ABST(69),ABST(70)/ &
     &+0.985611511545268335400175044630E0_q, &
     &+0.964762255587506430773811928118E0_q, &
     &+0.934906075937739689170919134835E0_q, &
     &+0.896321155766052123965307243719E0_q, &
     &+0.849367613732569970133693004967E0_q/
      DATA ABST( 71),ABST( 72),ABST( 73),ABST( 74),ABST( 75)/ &
     &+0.794483795967942406963097298970E0_q, &
     &+0.732182118740289680387426665091E0_q, &
     &+0.663044266930215200975115168663E0_q, &
     &+0.587715757240762329040745476401E0_q, &
     &+0.506899908932229390023747474377E0_q/
      DATA ABST( 76),ABST( 77),ABST( 78),ABST( 79),ABST( 80)/ &
     &+0.421351276130635345364119436172E0_q, &
     &+0.331868602282127649779916805730E0_q, &
     &+0.239287362252137074544603209165E0_q, &
     &+0.144471961582796493485186373598E0_q, &
     &+0.483076656877383162348125704405E-1_q/
      DATA ABST( 81),ABST( 82),ABST( 83),ABST( 84),ABST( 85)/ &
     &+0.998771007252426118600541491563E0_q, &
     &+0.993530172266350757547928750849E0_q, &
     &+0.984124583722826857744583600026E0_q, &
     &+0.970591592546247250461411983800E0_q, &
     &+0.952987703160430860722960666025E0_q/
      DATA ABST( 86),ABST( 87),ABST( 88),ABST( 89),ABST( 90)/ &
     &+0.931386690706554333114174380101E0_q, &
     &+0.905879136715569672822074835671E0_q, &
     &+0.876572020274247885905693554805E0_q, &
     &+0.843588261624393530711089844519E0_q, &
     &+0.807066204029442627082553043024E0_q/
      DATA ABST( 91),ABST( 92),ABST( 93),ABST( 94),ABST( 95)/ &
     &+0.767159032515740339253855437522E0_q, &
     &+0.724034130923814654674482233493E0_q, &
     &+0.677872379632663905211851280675E0_q, &
     &+0.628867396776513623995164933069E0_q, &
     &+0.577224726083972703817809238540E0_q/
      DATA ABST( 96),ABST( 97),ABST( 98),ABST( 99),ABST(100)/ &
     &+0.523160974722233033678225869137E0_q, &
     &+0.466902904750958404544928861650E0_q, &
     &+0.408686481990716729916225495814E0_q, &
     &+0.348755886292160738159817937270E0_q, &
     &+0.287362487355455576735886461316E0_q/
      DATA ABST(101),ABST(102),ABST(103),ABST(104),ABST(105)/ &
     &+0.224763790394689061224865440174E0_q, &
     &+0.161222356068891718056437390783E0_q, &
     &+0.970046992094626989300539558536E-1_q, &
     &+0.323801709628693620333222431521E-1_q, &
     &+0.999305041735772139456905624345E0_q/
      DATA ABST(106),ABST(107),ABST(108),ABST(109),ABST(110)/ &
     &+0.996340116771955279346924500676E0_q, &
     &+0.991013371476744320739382383443E0_q, &
     &+0.983336253884625956931299302156E0_q, &
     &+0.973326827789910963741853507352E0_q, &
     &+0.961008799652053718918614121897E0_q/
      DATA ABST(111),ABST(112),ABST(113),ABST(114),ABST(115)/ &
     &+0.946411374858402816062481491347E0_q, &
     &+0.929569172131939575821490154559E0_q, &
     &+0.910522137078502805756380668008E0_q, &
     &+0.889315445995114105853404038272E0_q, &
     &+0.865999398154092819760783385070E0_q/
      DATA ABST(116),ABST(117),ABST(118),ABST(119),ABST(120)/ &
     &+0.840629296252580362751691544695E0_q, &
     &+0.813265315122797559741923338086E0_q, &
     &+0.783972358943341407610220525213E0_q, &
     &+0.752819907260531896611863774885E0_q, &
     &+0.719881850171610826848940217831E0_q/
      DATA ABST(121),ABST(122),ABST(123),ABST(124),ABST(125)/ &
     &+0.685236313054233242563558371031E0_q, &
     &+0.648965471254657339857761231993E0_q, &
     &+0.611155355172393250248852971018E0_q, &
     &+0.571895646202634034283878116659E0_q, &
     &+0.531279464019894545658013903544E0_q/
      DATA ABST(126),ABST(127),ABST(128),ABST(129),ABST(130)/ &
     &+0.489403145707052957478526307021E0_q, &
     &+0.446366017253464087984947714758E0_q, &
     &+0.402270157963991603695766771260E0_q, &
     &+0.357220158337668115950442615046E0_q, &
     &+0.311322871990210956157512698560E0_q/
      DATA ABST(131),ABST(132),ABST(133),ABST(134),ABST(135)/ &
     &+0.264687162208767416373964172510E0_q, &
     &+0.217423643740007084149648748988E0_q, &
     &+0.169644420423992818037313629748E0_q, &
     &+0.121462819296120554470376463492E0_q, &
     &+0.729931217877990394495429419403E-1_q/
      DATA ABST(136)/ &
     &+0.243502926634244325089558428537E-1_q/
      DATA NSTOR(1), NSTOR(2), NSTOR(3), NSTOR(4) /1,2,3,4/
      DATA NSTOR(5), NSTOR(6), NSTOR(7), NSTOR(8) /5,6,8,10/
      DATA NSTOR(9), NSTOR(10), NSTOR(11), NSTOR(12) /12,14,16,20/
      DATA NSTOR(13), NSTOR(14), NSTOR(15), NSTOR(16) /24,32,48,64/
      DO 20 I=1,NPTS
         WEIGHT(I) = 0.0_q
         ABSCIS(I) = 0.0_q
   20 CONTINUE
      N = 0
      NPTSA = 0
      IFAIL = 0
      DO 60 I=1,16
         IF (NPTS<NSTOR(I)) GO TO 80
         N = N + (NPTSA+1)/2
         NPTSA = NSTOR(I)
         IF (NPTS==NSTOR(I)) GO TO 100
   60 CONTINUE
   80 IFAIL = 1
  100 HFRNGE = 0.5_q*(B-A)
      PNTMID = 0.5_q*(A+B)
      NL = NPTSA/2
      IF (NL<1) GO TO 140
      DO 120 NN=1,NL
         N = N + 1
         IIJJ = NPTSA + 1 - NN
         ABSCIS(NN) = HFRNGE*ABST(N) + PNTMID
         WEIGHT(NN) = HFRNGE*WTST(N)
         ABSCIS(IIJJ) = -HFRNGE*ABST(N) + PNTMID
         WEIGHT(IIJJ) = HFRNGE*WTST(N)
  120 CONTINUE
  140 IF (NPTSA<=(NL+NL)) GO TO 160
      N = N + 1
      ABSCIS(NL+1) = HFRNGE*ABST(N) + PNTMID
      WEIGHT(NL+1) = HFRNGE*WTST(N)
  160 RETURN
      END


      SUBROUTINE GAUSSI(WTFUN, A, B, ITYPE, NPTS, WEIGHT, ABSCIS, &
     & IFAIL)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!     MARK 7 RELEASE. NAG COPYRIGHT 1978.
!
!     RETURNS WEIGHTS AND PIVOTS FOR ONE GAUSS-WTFUN FORMULA IF
!     STORED
!     IFAIL = 1 - THE NPTS RULE IS NOT AMONG THOSE STORED
!     ( WEIGHT,ABSCIS EVALUATED FOR LARGEST VALID NPTS LESS THAN
!     REQUESTED VALUE)
!     IFAIL = 2 - VALUES OF A OR B INVALID
!     ( ALL WEIGHTS AND ABSCISSAE RETURNED AS ZERO)
!     IFAIL = 3 - UNDERFLOW IN EVALUATING LAGUERRE OR HERMITE
!     NORMAL WEIGHTS
!     ( THE UNDERFLOWING WEIGHTS ARE RETURNED AS ZERO)
!
!     THE WEIGHTS AND ABSCISSAE RETURNED DEPEND ON A AND B.
!
!     .. SCALAR ARGUMENTS ..
      REAL(q) A, B
      INTEGER IFAIL, ITYPE, NPTS
!     .. ARRAY ARGUMENTS ..
      REAL(q) ABSCIS(NPTS), WEIGHT(NPTS)
!     .. SUBROUTINE ARGUMENTS ..
!     WTFUN
!     ..
!     .. LOCAL SCALARS ..
!*      DOUBLE PRECISION SRNAME ... HOLLERITH AUF CHARACTER *****
      CHARACTER (8) SRNAME
      INTEGER IERR
!     .. FUNCTION REFERENCES ..
      INTEGER P01AAF
!     ..
      DATA SRNAME /' GAUSSI '/
      CALL WTFUN(A, B, ITYPE, NPTS, WEIGHT, ABSCIS, IERR)
      IF (IERR==0) GO TO 20
!georg      IFAIL = P01AAF(IFAIL,IERR,SRNAME)
      RETURN
   20 IFAIL = 0
      RETURN
      END
