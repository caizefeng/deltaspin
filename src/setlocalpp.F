!*******************************************************************
!     FOURCHECK
!     the subroutine performes a FFT of the local pseudopotential
!     V(q) =
!        4 \pi /q \int sin(qr) V(r) r dr
!
! due to a design fault presently the PAW potentials do not
! contain information about the local pseudopotential in real space.
! this subroutine extracts the required information
! by means of a sine transform of the local pseudopotential
! from reciprocal to real space
!
! The subroutine calculates the charge density rho(r) 
! that corresponds to the local pseudopotential V(q)
! i.e.
!
! rho(r) = 1  /(2 pi^2) \int_0^Infty sin(qr)/(qr) q^2/ ( 4 Pi e^2) V(q) q^2 dq
!        = 1/r/(2 pi^2) \int_0^Infty sin(qr) q^2/ ( 4 Pi e^2) V(q) q dq
! r V(r) = 1/  (2 pi^2) \int_0^Infty sin(qr) V(q) q dq
! 
! To avoid aliasing on the grid, a filter function is applied to rho(q).
!
! The routine also calculates the self-energy of the charge distribution.
!
! on entry: 
! Z valence
! V stores the potential: V(q) = VQ -  8 \pi Z/q^2
!
!*******************************************************************
SUBROUTINE POTTORHO_FILTER( G1, G2, Z, NQL, VQ, DELQL, NFFT, RHOSPL, RCUT, ESELF, IU6 ) 
  USE prec
  USE constant
  USE base
  USE c2f_interface, ONLY : ERRF
  IMPLICIT NONE
  !
  REAL(q) G1,G2         ! cutoff in Fourier space
  REAL(q) Z          ! valence
  INTEGER NQL        ! number of grid points in local potential          
  REAL(q) VQ(NQL)    ! local potential
  REAL(q) DELQL      ! step width for the array VQ
  INTEGER NFFT
  REAL(q) RHOSPL(NFFT/2,5)
  REAL(q) RCUT
  REAL(q) ESELF
  INTEGER :: IU6     ! OUTCAR
  ! local variables
  REAL(q), PARAMETER :: ACC_CUT=1.E-6_q
  REAL(q) RMAXF      ! maximum r values after FFT
  REAL(q) RDEL       ! spacing in real space
  REAL(q) FFT(NFFT+2)! array for FFT
  REAL(q) SPL(NFFT/2,5) ! spline array for potential/charge in real space
  REAL(q) QQ,SUM
  INTEGER I,K
  REAL(q) :: WINDOW


  DO I=1,NFFT
     QQ=DELQL*(I-1)
     IF (QQ>=G1) THEN 
        G1=QQ 
        EXIT
     ENDIF
  ENDDO

  DO I=1,NFFT
     QQ=DELQL*(I-1)
     IF (QQ>=G2) THEN 
        G2=QQ 
        EXIT
     ENDIF
  ENDDO

  IF(IU6>0) WRITE(IU6,'(A,2F10.2)') 'Fourier filtering potential',G1,G2

  RMAXF= 2*PI/DELQL

  ! first set up the values in the reciprocal space array
  DO I=1,NFFT
     QQ=DELQL*(I-1)
     ! fft of charge
     IF (I<=NQL) THEN
        ! the charge is given by
        ! rho(q) = q^2/ ( 4 Pi e^2) V(q) 
        FFT(I)=(VQ(I)*(QQ*QQ/(4*PI*FELECT))-Z)
     ELSE
        FFT(I)=0
     ENDIF
  ENDDO

  ! apply filter in Fourier space
  DO I=1,NFFT
     QQ=DELQL*(I-1)
     FFT(I)=FFT(I)*WINDOW(G1,G2,QQ)
  ENDDO

  SUM=0
  DO I=1,NFFT
     QQ=DELQL*(I-1)
     SUM=SUM+FFT(I)*QQ*QQ
  ENDDO
  SUM=SUM*DELQL/2/PI/PI

  ! Simpson rule
  ESELF=0
  DO I=1,NFFT
     ESELF=ESELF+FFT(I)**2*(2/3._q)
     IF (MOD(I,2)==0) ESELF=ESELF+FFT(I)**2*(2/3._q)
  END DO
  ESELF=ESELF-FFT(1)**2*(1/3._q)
  ESELF=(4*PI)*EDEPS*DELQL*ESELF/TPI**3/2


  DO I=1,NFFT
     QQ=DELQL*(I-1)
     FFT(I)=FFT(I)*QQ
  ENDDO
  CALL REALFT(FFT,NFFT/2,1)

  RDEL = RMAXF/NFFT
  ! f(r) = int sin_0^Infty  f(q) sin qr dq = del q sum_0^NFFT  f(q) sin qr

  DO K=1,NFFT/2
     SPL(K,2)=FFT(K*2)*DELQL
  ENDDO
  ! setup linear r mesh after FFT
  DO K=1,NFFT/2
     SPL(K,1)=(K-1)*RDEL
  ENDDO
  !
  ! inverse transform of f(q) = 4 pi / q       \int sin (qr) f(r) r dr
  ! is                   f(r) = 1 /(2 pi^2) /r \int sin (qr) f(r) q dr 
  ! since 2 / pi \int_0^Infty sin(qr) dr =  delta(q)
  SPL(1,2)=SUM
  DO K=2,NFFT/2
     SPL(K,2)=SPL(K,2)/SPL(K,1)/2/PI/PI
  ENDDO

  ! find cutoff radius: |f(r)| < ACC_CUT for r>RCUT
  DO K=NFFT/2,1,-1
     IF(ABS(SPL(K,2))>=ACC_CUT)EXIT
  END DO
  RCUT=SPL(K+1,1)
  WRITE(*,*) 'RCUTRHO=',RCUT

  ! calc. spline coefficients
  CALL SPLCOF(SPL ,NFFT/2, NFFT/2, 0.0_q)

  RHOSPL=SPL

END SUBROUTINE POTTORHO_FILTER


SUBROUTINE ATORHO_FILTER(G1,G2,NQL,RHO,DELQL,NFFT,RHOSPL,RCUT,IU6) 
  USE prec
  USE constant
  USE base
  IMPLICIT NONE

  REAL(q) G1,G2      ! cutoff in Fourier space
  REAL(q) Z          ! valence
  INTEGER NQL        ! number of grid points
  REAL(q) RHO(NQL)   ! atomic charge
  REAL(q) DELQL      ! k-step width
  INTEGER NFFT       !
  REAL(q) RHOSPL(NFFT/2,5)  ! array to hold interpolating spline coefficients
  REAL(q) RCUT
  INTEGER :: IU6     ! OUTCAR
  ! local variables
  REAL(q), PARAMETER :: ACC_CUT=1.E-6_q
  REAL(q) RMAXF      ! maximum r values after FFT
  REAL(q) RDEL       ! spacing in real space
  REAL(q) FFT(NFFT+2)! array for FFT
  REAL(q) SPL(NFFT/2,5) ! spline array for charge in real space
  REAL(q) QQ
  INTEGER I,K
  REAL(q) :: SUM
  REAL(q) WINDOW,r,v,dum

  DO I=1,NFFT
     QQ=DELQL*(I-1)
     IF (QQ>=G1) THEN 
        G1=QQ 
        EXIT
     ENDIF
  ENDDO

  DO I=1,NFFT
     QQ=DELQL*(I-1)
     IF (QQ>=G2) THEN 
        G2=QQ 
        EXIT
     ENDIF
  ENDDO

  IF(IU6>0) WRITE(IU6,'(A,2F10.2)') 'Fourier filtering atomic charge',G1,G2

  RMAXF= 2*PI/DELQL

  ! apply filter in Fourier space
  DO I=1,NFFT
     QQ=DELQL*(I-1)
     IF (I<=NQL) THEN
        FFT(I)=RHO(I)*WINDOW(G1,G2,QQ)
     ELSE
        FFT(I)=0
     ENDIF
  ENDDO

  ! calculate rho(r=0)
  SUM=0
  DO I=1,NFFT
     QQ=DELQL*(I-1)
     SUM=SUM+FFT(I)*QQ*QQ
  ENDDO
  SUM=SUM*DELQL/2/PI/PI

  DO I=1,NFFT
     QQ=DELQL*(I-1)
     FFT(I)=FFT(I)*QQ
  ENDDO

  CALL REALFT(FFT,NFFT/2,1)

  RDEL = RMAXF/NFFT
  ! f(r) = int sin_0^Infty  f(q) sin qr dq = del q sum_0^NFFT  f(q) sin qr
  DO K=1,NFFT/2
     SPL(K,2)=FFT(K*2)*DELQL
  ENDDO
  ! setup linear r mesh after FFT
  DO K=1,NFFT/2
     SPL(K,1)=(K-1)*RDEL
  ENDDO
  !
  ! inverse transform of f(q) = 4 pi / q       \int sin (qr) f(r) r dr
  ! is                   f(r) = 1 /(2 pi^2) /r \int sin (qr) f(r) q dr 
  ! since 2 / pi \int_0^Infty sin(qr) dr =  delta(q)
  SPL(1,2)=SUM
  DO K=2,NFFT/2
     SPL(K,2)=SPL(K,2)/SPL(K,1)/2/PI/PI
  ENDDO

  ! find cutoff radius: |f(r)| < ACC_CUT for r>RCUT
  DO K=NFFT/2,1,-1
     IF(ABS(SPL(K,2))>=ACC_CUT)EXIT
  END DO
  RCUT=SPL(K+1,1)
  IF (IU6>0) WRITE(IU6,*) 'RCUTATO=',RCUT

  ! calc. spline coefficients
  CALL SPLCOF(SPL ,NFFT/2, NFFT/2, 0.0_q)

  RHOSPL=SPL

END SUBROUTINE ATORHO_FILTER



FUNCTION WINDOW(X1,X2,X)
  USE prec
  USE constant
  USE c2f_interface, ONLY : ERRF
  IMPLICIT NONE
  REAL(q) WINDOW
  REAL(q),intent(in) :: X1,X2,X
  WINDOW=(1-ERRF(10*((X1+X2)/2-X)/(X1-X2)))/2
END FUNCTION WINDOW


!*******************************************************************
!     FOURCHECK
!     the subroutine performes a FFT of the local pseudopotential
!     V(q) =
!        4 \pi /q \int sin(qr) V(r) r dr
!
!     Parameter
!     MD          number of points used for calculating V(q->0)
!     DELQL       step-size for VLQ
!     RMAX        behind RMAX V(r) is set to zero
!
!*******************************************************************

    SUBROUTINE FOURPOT_TO_Q( RDEP_IN, VL, PSP, NQL, DELQL, RGRD, IU6 )
      USE prec
      USE constant
      USE radial
      USE c2f_interface, ONLY : ERRF
      IMPLICIT NONE

      TYPE(rgrid) RGRD
      REAL(q) RDEP_IN       ! outermost matching point 
                            ! potential is set to zero outside this point
      REAL(q) VL(RGRD%NMAX) ! potential in real space
      INTEGER NQL           ! number of grid points in local potential
      REAL(q) PSP(NQL)      ! on entry: old local potential in reciprocal space
                            ! on exit: new local potential in reciprocal space
      REAL(q) DELQL         ! step in reciprocal space
      INTEGER IU6           ! IO unit for output
!     work arrays
      INTEGER NMAX          ! number of grid points
      REAL(q) RMAX          ! outermost point for integration
      REAL(q) RDEP          ! actual matching point 
      INTEGER, PARAMETER :: NFFT=32768
      REAL(q) :: FAKT, R, SUM, RMAXF, D1, D2, DD, QQ, VRMAX, WINDOW
      REAL(q) :: VR(RGRD%NMAX),WRK(RGRD%NMAX,5)
      REAL(q) :: VLQ(NQL)
      REAL(q) :: VQ1(NFFT),VQ2(NFFT),VQ3(NFFT)
      INTEGER :: K, I 

! outermost point equal outermost grid point
      NMAX=RGRD%NMAX
      RMAX=RGRD%R(NMAX)

!
! find potential at matching point (RDEP) if supplied
! 
      VRMAX=0
      DO K=1,NMAX-1
         IF (RDEP_IN>0 .AND.  RGRD%R(K)-RDEP_IN > -5E-3) EXIT 
      END DO
      RDEP =RGRD%R(K)
      VRMAX=VL(K)

      DO K=1,NMAX
         R =RGRD%R(K)
         WRK(K,1)=R
! I think the hard window is better defined
! if the xc functional does not change, the difference
! between the potentials goes to zero smoothly anyhow
! so the hard window is fine and exact in that case.
! 
#define hardwindow
#ifdef hardwindow
         IF (RDEP_IN>0 .AND.  R-RDEP_IN > -5E-3) THEN
            VL(K)=0
            WRK(K,2)=0
         ELSE
            VL(K)=VL(K)-VRMAX
            WRK(K,2)=VL(K)*R
         ENDIF
#else
         ! window falls off to 0.5E-2 at RDEP 
         WINDOW=(1-ERRF(5*(R-RDEP/1.6)/RDEP))/2
         VL(K)=VL(K)*WINDOW
         WRK(K,2)=VL(K)*R
#endif
!         WRITE(77,'(3E15.6)') WRK(K,1),WRK(K,2),WINDOW
      END DO
      CALL SPLCOF(WRK,SIZE(VL),SIZE(VL),1E30_q)

!----------------------------------------------------------------------
! limit q-> 0   using simpson integration
!----------------------------------------------------------------------
      DO K=1,NMAX
         VR(K)=WRK(K,2)*RGRD%R(K)
      ENDDO

      CALL SIMPI(RGRD, VR, SUM)

      VLQ(1)= 4._q*PI*SUM
!----------------------------------------------------------------------
! FFT with  NFFT NFFT/2 und NFFT/4 points
!----------------------------------------------------------------------
      RMAXF = 2._q*PI/DELQL

      CALL WFSINT(RMAXF, RMAX, NFFT/4, WRK, RGRD%NMAX, VQ1)
      CALL WFSINT(RMAXF, RMAX, NFFT/2, WRK, RGRD%NMAX, VQ2)
      CALL WFSINT(RMAXF, RMAX, NFFT/1, WRK, RGRD%NMAX, VQ3)

!-----error estimation
      D1=0
      D2=0
      DO I=1,NQL
         D1=D1+ABS(VQ3(I)-VQ1(I))
         D2=D2+ABS(VQ3(I)-VQ2(I))
      ENDDO

!-----set VLQ (returned array)
      DO I=2,NQL
         QQ=DELQL*(I-1)
         VLQ(I)=VQ3(I)*4._q*PI/QQ
      ENDDO

!      DO I=1,NQL
!         WRITE(77,'(4F14.7)') DELQL*(I-1), VLQ(I), PSP(I), PSP(I)-VLQ(I)
!      ENDDO

      PSP=PSP-VLQ
!----------------------------------------------------------------------
! write out report
!----------------------------------------------------------------------
      IF (IU6>=0) THEN
         WRITE(IU6,1) RDEP,VRMAX,NFFT,RMAXF,RMAXF/NFFT,DELQL

         FAKT = 4*PI
         WRITE(IU6,'(A,E10.3,A)')' difference to FFT with N/2= ',D2/NQL*FAKT
         WRITE(IU6,'(A,E10.3,A)')' difference to FFT with N/4= ',D1/NQL*FAKT
         DD = D2/D1
         WRITE(IU6,'(A,E10.3,A/)')' estimated error in     v(q)= ',D2*DD/(1-DD)/NQL*FAKT,' 1/q'

    1 FORMAT( /' redefinition  of local potential',/ &
     &       ' ---------------------------------------', &
     &       ' cutoff radius for potential RDEP =',F9.2/ &
     &       ' potential at matching point      =',E10.2/ &
     &       ' number of fourier points    NFFT =',I6/ &
     &       ' outmost radius for FFT      RMAXF=',F9.2/ &
     &       ' distance between FFT points      =',F10.3/ &
     &       ' distance between Q-points   DELQL=',F11.4)

      ENDIF
    END SUBROUTINE FOURPOT_TO_Q
    

!*********************************************************************
!
!  WFSINT
!  performes a sin transformation given an work array containing
!  a spline fit
!
!********************************************************************

    SUBROUTINE WFSINT(RMAXF,RMAX,NFFT,WRK, NMAX, VQ)
      USE constant
      IMPLICIT NONE

      REAL(q) :: RMAXF        ! outmost radius for FFT
      REAL(q) :: RMAX         ! outmost radius for data points; for r>RMAX   data are assumed to be 0)
      INTEGER :: NFFT         ! number of FFT-points 
      INTEGER :: NMAX         ! number of grid points in spline
      REAL(q) ::  WRK(NMAX,5) ! spline fit
      REAL(q) ::  VQ(NFFT)    ! fourier transformed

! local
      REAL(q) :: RDEL, R, DUMMY
      INTEGER :: I, K

      RDEL = RMAXF/NFFT

      DO I=1,NFFT
         VQ(I)=0
      ENDDO

      R=0
      DO I=1,NFFT
         IF (R>RMAX) THEN
            VQ(I)=0
         ELSE
            CALL SPLVAL(R, VQ(I), DUMMY, WRK, NMAX, NMAX)
         ENDIF
         R=R+RDEL
      ENDDO

! f(r) = int sin_0^Infty  f(q) sin qr dq = del r sum_0^NFFT  f(q) sin qr
      CALL REALFT(VQ,NFFT/2,1)

      DO K=1,NFFT/2
         VQ(K)=VQ(K*2)*RDEL
      ENDDO

    END SUBROUTINE WFSINT


!*******************************************************************
!
! due to a design fault presently the PAW potentials do not
! contain information about the local pseudopotential in
! real space (fixed in vasp.5.2 compatible potentials)
! this subroutine extracts the required information
! by means of a sine transform of the local pseudopotential
! from reciprocal to real space
!
! the subroutine calculates either the charge density rho(r) 
! or the potential V(r) 
! that corresponds to the local pseudopotential V(q)
! i.e.
!
! rho(r) = 1/r/(2 pi^2) \int_0^Infty sin(qr) q^2/ ( 4 Pi e^2) V(q) q dq
! r V(r) = 1/  (2 pi^2) \int_0^Infty sin(qr) V(q) q dq
! 
! on entry: 
! Z valence
! V stores the potential
!  V(q) = VQ -  8 \pi Z/q^2
!
!*******************************************************************


      SUBROUTINE POTTORHO( Z, NQL, VQ, DELQL, LPOT , NMAX, R, POTPSC ) 
      USE prec
      USE constant
      USE c2f_interface, ONLY : ERRF
      IMPLICIT NONE
      
      REAL(q) Z          ! valence
      INTEGER NQL        ! number of grid points in local potential          
      REAL(q) VQ(NQL)    ! local potential
      REAL(q) DELQL      ! step width for the array VQ
      LOGICAL LPOT       ! transform to get potential
                         ! or charge
      INTEGER NMAX       ! number of grid points
      REAL(q) R(NMAX)    ! r of each grid point
      REAL(q) POTPSC(NMAX)! potential
! local variable
      INTEGER, PARAMETER :: NFFT=32768
      REAL(q) RMAXF      ! maximum r values after FFT
      REAL(q) RDEL       ! spacing in real space
      REAL(q) FFT(NFFT+2)! array for FFT
      REAL(q) SPL(NFFT/2,5)
                         ! spline array for potential/charge in real space
      REAL(q) QQ

      INTEGER K
      REAL(q) :: FAKT,DUMMY,RHO,RR,ALPC,ALP,ALPSQRT
      REAL(q) :: FE=2
      INTEGER I
      
      RMAXF= 2*PI/DELQL

      ALPC=-(NQL*DELQL)**2/4/LOG(1.D-1)
      ALP=1/AUTOA**2

! first set up the values in the reciprocal space array

      DO I=1,NFFT
         QQ=DELQL*(I-1)
         IF (I<=NQL .AND. LPOT) THEN
!            FFT(I)=VQ(I)*QQ
! fft of potential
! 
! rho(q) = q^2/ ( 4 Pi e^2) V(q) 
            IF (I/=1) THEN
              FFT(I)=(VQ(I)-Z*(1-EXP(-QQ*QQ/4/ALP))*4*PI*FELECT/ &
     &              (QQ*QQ))*QQ
            ELSE
              FFT(I)=0
            ENDIF
         ELSE IF (I<=NQL) THEN
!            FFT(I)=VQ(I)*QQ
! fft of charge
! the charge is given by
! rho(q) = q^2/ ( 4 Pi e^2) V(q) 
            FFT(I)=(VQ(I)*(QQ*QQ/(4*PI*FELECT))-Z)*QQ*EXP(-QQ*QQ/4/ALPC)
         ELSE
            FFT(I)=0
         ENDIF
      ENDDO
      
      CALL REALFT(FFT,NFFT/2,1)

      RDEL = RMAXF/NFFT
! f(r) = int sin_0^Infty  f(q) sin qr dq = del q sum_0^NFFT  f(q) sin qr

      DO K=1,NFFT/2
         SPL(K,2)=FFT(K*2)*DELQL
      ENDDO
! setup linear r mesh after FFT
      DO K=1,NFFT/2
         SPL(K,1)=(K-1)*RDEL
      ENDDO
!
! inverse transform of f(q) = 4 pi / q       \int sin (qr) f(r) r dr
! is                   f(r) = 1 /(2 pi^2) /r \int sin (qr) f(r) q dr 
! since 2 / pi \int_0^Infty sin(qr) dr =  delta(q)
      DO K=2,NFFT/2
         SPL(K,2)=SPL(K,2)/SPL(K,1)/2/PI/PI
      ENDDO

      IF (LPOT) THEN
         ALPSQRT = SQRT(ALP)
         DO K=2,NFFT/2
            SPL(K,2)=SPL(K,2)-FELECT*Z*ERRF(ALPSQRT*SPL(K,1))/SPL(K,1)
         ENDDO
         SPL(1,1)=SPL(1,1)-FELECT*Z*2*ALPSQRT/SQRT(PI)
         
      ENDIF
!
! now we need to go from the linear grid to a logarithmic one
!
      CALL SPLCOF(SPL ,NFFT/2, NFFT/2, 1.E30_q)

      POTPSC=0
      DO K=1,NMAX
         IF (R(K)/RDEL>NFFT) EXIT
         CALL SPLVAL(R(K), RHO, DUMMY, SPL, NFFT/2, NFFT/2)
         IF (LPOT) THEN
            POTPSC(K)=RHO
         ELSE
         ! set radially integrated charge
            POTPSC(K)=RHO*4*PI*R(K)**2
         ENDIF
      ENDDO
!      WRITE(95,"(2F14.7)") (R(K), POTPSC(K)*R(K)/RYTOEV/AUTOA,K=1,NMAX)

      END


!*******************************************************************
!
! the following subroutines are from Numerical Recepies
! they perform real to complex transforms
! or sinus transforms
!
!*******************************************************************

      SUBROUTINE SINFT(Y,N)
      USE prec
      IMPLICIT NONE
      REAL(q) WR,WI,WPR,WPI,WTEMP,THETA,Y1,Y2,SUM
      INTEGER N,M,J
      REAL(q) Y(N)
      THETA=3.14159265358979D0/DBLE(N)
      WR=1.0D0
      WI=0.0D0
      WPR=-2.0D0*DSIN(0.5D0*THETA)**2
      WPI=DSIN(THETA)
      Y(1)=0.0
      M=N/2
      DO 11 J=1,M
        WTEMP=WR
        WR=WR*WPR-WI*WPI+WR
        WI=WI*WPR+WTEMP*WPI+WI
        Y1=WI*(Y(J+1)+Y(N-J+1))
        Y2=0.5*(Y(J+1)-Y(N-J+1))
        Y(J+1)=Y1+Y2
        Y(N-J+1)=Y1-Y2
11    CONTINUE
      CALL REALFT(Y,M,+1)
      SUM=0.0
      Y(1)=0.5*Y(1)
      Y(2)=0.0
      DO 12 J=1,N-1,2
        SUM=SUM+Y(J)
        Y(J)=Y(J+1)
        Y(J+1)=SUM
12    CONTINUE
      RETURN
      END



      SUBROUTINE REALFT(DATA,N,ISIGN)
      USE prec
      IMPLICIT NONE
      REAL(q) WR,WI,WPR,WPI,WTEMP,THETA,WRS,WIS
      REAL(q) H2R,H2I,H1R,H1I,C1,C2
      REAL(q) DATA(*)
      INTEGER N
      INTEGER I3,I4,I1,I2,N2P3,ISIGN,I
      THETA=6.28318530717959D0/2.0D0/DBLE(N)
      WR=1.0D0
      WI=0.0D0
      C1=0.5
      IF (ISIGN.EQ.1) THEN
        C2=-0.5
        CALL FOUR1(DATA,N,+1)
        DATA(2*N+1)=DATA(1)
        DATA(2*N+2)=DATA(2)
      ELSE
        C2=0.5
        THETA=-THETA
        DATA(2*N+1)=DATA(2)
        DATA(2*N+2)=0.0
        DATA(2)=0.0
      ENDIF
      WPR=-2.0D0*DSIN(0.5D0*THETA)**2
      WPI=DSIN(THETA)
      N2P3=2*N+3
      DO 11 I=1,N/2+1
        I1=2*I-1
        I2=I1+1
        I3=N2P3-I2
        I4=I3+1
        WRS=SNGL(WR)
        WIS=SNGL(WI)
        H1R=C1*(DATA(I1)+DATA(I3))
        H1I=C1*(DATA(I2)-DATA(I4))
        H2R=-C2*(DATA(I2)+DATA(I4))
        H2I=C2*(DATA(I1)-DATA(I3))
        DATA(I1)=H1R+WRS*H2R-WIS*H2I
        DATA(I2)=H1I+WRS*H2I+WIS*H2R
        DATA(I3)=H1R-WRS*H2R+WIS*H2I
        DATA(I4)=-H1I+WRS*H2I+WIS*H2R
        WTEMP=WR
        WR=WR*WPR-WI*WPI+WR
        WI=WI*WPR+WTEMP*WPI+WI
11    CONTINUE
      IF (ISIGN.EQ.1) THEN
        DATA(2)=DATA(2*N+1)
      ELSE
        CALL FOUR1(DATA,N,-1)
      ENDIF
      RETURN
      END


      SUBROUTINE FOUR1(DATA,NN,ISIGN)
      USE prec
      IMPLICIT NONE
      REAL(q) WR,WI,WPR,WPI,WTEMP,THETA,TEMPI,TEMPR
      REAL(q) DATA(*)
      INTEGER M,ISTEP,MMAX,ISIGN,NN,N,I,J
      N=2*NN
      J=1
      DO 11 I=1,N,2
        IF(J.GT.I)THEN
          TEMPR=DATA(J)
          TEMPI=DATA(J+1)
          DATA(J)=DATA(I)
          DATA(J+1)=DATA(I+1)
          DATA(I)=TEMPR
          DATA(I+1)=TEMPI
        ENDIF
        M=N/2
1       IF ((M.GE.2).AND.(J.GT.M)) THEN
          J=J-M
          M=M/2
        GO TO 1
        ENDIF
        J=J+M
11    CONTINUE
      MMAX=2
2     IF (N.GT.MMAX) THEN
        ISTEP=2*MMAX
        THETA=6.28318530717959D0/(ISIGN*MMAX)
        WPR=-2.D0*DSIN(0.5D0*THETA)**2
        WPI=DSIN(THETA)
        WR=1.D0
        WI=0.D0
        DO 13 M=1,MMAX,2
          DO 12 I=M,N,ISTEP
            J=I+MMAX
            TEMPR=SNGL(WR)*DATA(J)-SNGL(WI)*DATA(J+1)
            TEMPI=SNGL(WR)*DATA(J+1)+SNGL(WI)*DATA(J)
            DATA(J)=DATA(I)-TEMPR
            DATA(J+1)=DATA(I+1)-TEMPI
            DATA(I)=DATA(I)+TEMPR
            DATA(I+1)=DATA(I+1)+TEMPI
12        CONTINUE
          WTEMP=WR
          WR=WR*WPR-WI*WPI+WR
          WI=WI*WPR+WTEMP*WPI+WI
13      CONTINUE
        MMAX=ISTEP
      GO TO 2
      ENDIF
      RETURN
      END


