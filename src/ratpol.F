#include "symbol.inc"

!*********************************************************************
!
! This module implements rational polynomial fits
! more specifically it fits the following function
!
!   N        1
!  sum    ----------- 
!  i=1    a_i + x b_i 
!
! to a set of date points (x_k,f_k)
!
!*********************************************************************
MODULE ratpolfit
  USE prec
  IMPLICIT NONE
  TYPE ratpol
     COMPLEX(q) :: A0
     COMPLEX(q), POINTER :: A(:) !  coefficients a_i
     COMPLEX(q), POINTER :: B(:) !  coefficients b_i
     INTEGER :: N                !  number of poles N
  END TYPE ratpol

  CONTAINS

!**********************************************************************
!
! create a rational polynomial data structure and delete it
!
!**********************************************************************

    SUBROUTINE ALLOCATE_RATPOT(N, R)
      INTEGER N
      TYPE (ratpol) R
      
      ALLOCATE(R%A(N))
      ALLOCATE(R%B(N))
      R%N=N

    END SUBROUTINE ALLOCATE_RATPOT

    SUBROUTINE DEALLOCATE_RATPOT(R)
      INTEGER N
      TYPE (ratpol) R
      
      DEALLOCATE(R%A)
      DEALLOCATE(R%B)

    END SUBROUTINE DEALLOCATE_RATPOT

    SUBROUTINE REALLOCATE_RATPOT(N, R)
      INTEGER N
      TYPE (ratpol) R
      COMPLEX(q) :: A(R%N)
      COMPLEX(q) :: B(R%N)
      INTEGER :: NMAX

      A=R%A
      B=R%B
      NMAX=MIN(R%N,N)
      
      CALL DEALLOCATE_RATPOT(R ) 
      
      ALLOCATE(R%A(N))
      ALLOCATE(R%B(N))
      
      R%A(1:NMAX)=A(1:NMAX)
      R%B(1:NMAX)=B(1:NMAX)

      IF (R%N+2==N) THEN
         R%A(R%N-1)=R%A(R%N-1)*2
         R%A(R%N)  =R%A(R%N)*2
         R%B(R%N-1)=R%B(R%N-1)*2
         R%B(R%N)  =R%B(R%N)*2
         
         R%A(N-1)=R%A(R%N-1)*1.02
         R%A(N)  =R%A(R%N)  *1.02
         R%B(N-1)=R%B(R%N-1)
         R%B(N)  =R%B(R%N)

         R%A(R%N-1)=R%A(R%N-1)/1.02
         R%A(R%N)  =R%A(R%N)/1.02
      ENDIF

      IF (R%N+4==N) THEN
         R%A(R%N-1)=R%A(R%N-1)*3
         R%A(R%N)  =R%A(R%N)*3
         R%B(R%N-1)=R%B(R%N-1)*3
         R%B(R%N)  =R%B(R%N)*3

         R%A(N-3)=R%A(R%N-1)/1.1
         R%A(N-2)=R%A(R%N)/1.1
         R%B(N-3)=R%B(R%N-1)
         R%B(N-2)=R%B(R%N)

         R%A(N-1)=R%A(R%N-1)*1.1
         R%A(N)  =R%A(R%N)*1.1
         R%B(N-1)=R%B(R%N-1)
         R%B(N)  =R%B(R%N)
      ENDIF

      R%N=N

    END SUBROUTINE REALLOCATE_RATPOT


    SUBROUTINE REALLOCATE_RATPOT_IMAG(N, R)
      INTEGER N
      TYPE (ratpol) R
      COMPLEX(q) :: A(R%N)
      COMPLEX(q) :: B(R%N)
      INTEGER :: NMAX

      A=R%A/R%B
      B=1/R%B

      NMAX=MIN(R%N,N)
      
      CALL DEALLOCATE_RATPOT(R ) 
      
      ALLOCATE(R%A(N))
      ALLOCATE(R%B(N))
      
      R%A(1:NMAX)=A(1:NMAX)
      R%B(1:NMAX)=B(1:NMAX)

      IF (R%N+1==N) THEN
         ! pole in a
         ! intensity in b
         R%B(N)  =R%B(R%N)/2
         R%B(R%N)=R%B(R%N)/2

         R%A(N)  =R%A(R%N)*1.1
         R%A(R%N)=R%A(R%N)/1.1

         R%B=1/R%B
         R%A=R%A*R%B
      ENDIF

      IF (R%N+2==N) THEN
         ! pole in a
         R%B(N)    =R%B(R%N)/2
         R%B(N-1)  =R%B(R%N-1)/2
         R%B(R%N)  =R%B(R%N)/2
         R%B(R%N-1)=R%B(R%N-1)/2

         ! intensity in b
         R%A(N)    =R%A(R%N)*1.1
         R%A(N-1)  =R%A(R%N-1)*1.1
         R%A(R%N)  =R%A(R%N)/1.1
         R%A(R%N-1)=R%A(R%N-1)/1.1

         R%B=1/R%B
         R%A=R%A*R%B
      ENDIF

      R%N=N

    END SUBROUTINE REALLOCATE_RATPOT_IMAG


!**********************************************************************
!
! evaluate rational polynomial at a set of data points
! optionally calculate the derivatives with respect to a_i and b_i
! 
! for high performance no run time checking of array dimensions
! is performed
!
!**********************************************************************
   
    SUBROUTINE CALC_RATPOL(R, X, F, FA, FB)
      TYPE (ratpol) R
      REAL(q),INTENT(IN) :: X(:)     ! x
      COMPLEX(q),INTENT(OUT):: F(:)  ! return f(x) 
      COMPLEX(q),OPTIONAL :: FA(:,:) ! derivatives with respect to a 
      COMPLEX(q),OPTIONAL :: FB(:,:) ! derivatives with respect to b
                                     ! first index corresponds to positions
    ! local
      INTEGER :: I, K

      IF (.NOT.PRESENT(FA)) THEN
         DO K=1,SIZE(X)
            F(K)=R%A0
            DO I=1,R%N
               F(K)=F(K)+1/(R%A(I)+X(K)*R%B(I))
            ENDDO
         ENDDO
      ELSE
         IF (SIZE(FA,1)/=SIZE(X) .OR. SIZE(FA,2)/=R%N) THEN
            WRITE(0,*) 'internal error in CALC_RATPOL: FA is not properly dimensioned',SIZE(FA,1),SIZE(X),SIZE(FA,2),R%N
            STOP
         ENDIF
         IF (SIZE(FB,1)/=SIZE(X) .OR. SIZE(FB,2)/=R%N) THEN
            WRITE(0,*) 'internal error in CALC_RATPOL: FB is not properly dimensioned',SIZE(FB,1),SIZE(X),SIZE(FB,2),R%N
            STOP
         ENDIF
            
         DO K=1,SIZE(X)
            F(K)=R%A0
            DO I=1,R%N
               F(K)   =F(K)+1/(R%A(I)+X(K)*R%B(I))
               FA(K,I)=-1   /(R%A(I)+X(K)*R%B(I))**2
               FB(K,I)=-X(K)/(R%A(I)+X(K)*R%B(I))**2
            ENDDO
         ENDDO
      END IF

    END SUBROUTINE CALC_RATPOL


!**********************************************************************
!
! evaluate rational polynomial at a set of imaginary data points
!
!**********************************************************************
   
    SUBROUTINE CALC_RATPOL_IMAG(R, X, F )
      TYPE (ratpol) R
      REAL(q),INTENT(IN) :: X(:)    ! x
      COMPLEX(q),INTENT(OUT):: F(:) ! return f(x) 
    ! local
      INTEGER :: I, K

      DO K=1,SIZE(X)
         F(K)=R%A0
         DO I=1,R%N
            F(K)=F(K)+1/(R%A(I)-CMPLX(0.0,X(K), q)*R%B(I))
         ENDDO
      ENDDO
    END SUBROUTINE CALC_RATPOL_IMAG

!**********************************************************************
!
! evaluate the least square error of a rational polynomial fit
! to a set of data points supplied by the calling routine
! and optionally  return the gradient with respect to the fitting
! coefficients
!
!**********************************************************************

    SUBROUTINE RMS_RATPOL(R, X, F, W, RMS, DERA, DERB )
      TYPE (ratpol) R
      REAL(q),INTENT(IN) :: X(:)     ! coordinates x
      COMPLEX(q),INTENT(IN) :: F(:)  ! function values f
      REAL(q),INTENT(IN) :: W(:)     ! weights
      REAL(q),INTENT(OUT):: RMS      ! mean error
      COMPLEX(q),OPTIONAL :: DERA(:) ! derivatives w.r.t a 
      COMPLEX(q),OPTIONAL :: DERB(:) ! derivatives w.r.t b 
    ! local
      COMPLEX(q) :: FP(SIZE(X))      ! present values of fit
      COMPLEX(q) :: FA(SIZE(X),R%N)  ! derivative of polynomial w.r.t a at each point x
      COMPLEX(q) :: FB(SIZE(X),R%N)  ! derivative of polynomial w.r.t b at each point x
      INTEGER :: I, K

      RMS=0
      IF (PRESENT(DERA)) THEN
         DERA=0
         DERB=0

         CALL CALC_RATPOL(R, X, FP, FA, FB)

         DO K=1,SIZE(X)
            RMS =RMS+(FP(K)-F(K))*CONJG(FP(K)-F(K))*W(K) !/SIZE(X)
            DO I=1,R%N
               DERA(I)=DERA(I)+2*CONJG(FA(K,I))*(FP(K)-F(K))*W(K) !/SIZE(X)
               DERB(I)=DERB(I)+2*CONJG(FB(K,I))*(FP(K)-F(K))*W(K) !/SIZE(X)
            ENDDO
         ENDDO
      ELSE
         CALL CALC_RATPOL(R, X, FP)
         DO K=1,SIZE(X)
            RMS =RMS+(FP(K)-F(K))*CONJG(FP(K)-F(K))*W(K)
         ENDDO
      ENDIF
    END SUBROUTINE RMS_RATPOL
    
!**********************************************************************
!
! determine the rational polynomial fit to a set of data points
! supplied by the calling routine
! the calling routine must supply some sensible initialisation
! for the rational polynomial as well
! probably the simplest algorithm one can come up with.
! it uses a damped second order equation
! I think one needs to set up the inverse of the Hessian 
! (Newton algorithm) on the long run
!
!**********************************************************************

    SUBROUTINE FIT_RATPOL(R, X, F, W)
      USE main_mpi
      TYPE (ratpol) :: R
      REAL(q),INTENT(IN) :: X(:)    ! coordinates x
      COMPLEX(q),INTENT(IN) :: F(:) ! function values f
      REAL(q),INTENT(IN) :: W(:)    ! weights
    ! local
      REAL(q), PARAMETER :: FRICTION=0.04    ! friction term in the damped equation
      ! smaller value for friction result in instabilities
      REAL(q), PARAMETER :: DT_DEFAULT=1E-5  ! time step
      REAL(q), PARAMETER :: BRC=1E-8         ! break if rms does not change by more than BRC
      INTEGER, PARAMETER :: STEPS=1000000
      REAL(q) :: RMS, RMS_LAST, DT
      COMPLEX(q) :: DERA(R%N)       ! derivatives w.r.t a 
      COMPLEX(q) :: DERB(R%N)       ! derivatives w.r.t b 
      COMPLEX(q) :: VELA(R%N), VELB(R%N)
      COMPLEX(q) :: AIN(R%N), BIN(R%N)
      INTEGER :: I, NBREAK, NINCREASE
      
      ! save initial values
      AIN=R%A
      BIN=R%B
      DT=DT_DEFAULT*2

      ! start by restoring initial values
      ! and divide timestep by 2
 100  CONTINUE
      DT=DT/2
      R%A=AIN
      R%B=BIN
      VELA=0
      VELB=0
      
      CALL RMS_RATPOL(R, X, F, W, RMS, DERA, DERB )
      NBREAK=0
      NINCREASE=0

      DO I=1,STEPS
         DERA=-DERA
         DERB=-DERB
         VELA=((1-FRICTION)*VELA+2*DERA*DT)/(1+FRICTION)
         VELB=((1-FRICTION)*VELB+2*DERB*DT)/(1+FRICTION)
         R%A=R%A+VELA
         R%B=R%B+VELB
         RMS_LAST=RMS
         CALL RMS_RATPOL(R, X, F, W, RMS, DERA, DERB )
         IF (ABS(SQRT(RMS)-SQRT(RMS_LAST))<BRC) THEN
            NBREAK=NBREAK+1
            IF (NBREAK>4) EXIT
         ELSE
            NBREAK=0
         ENDIF
         IF (SQRT(RMS)>SQRT(RMS_LAST)) THEN
            NINCREASE=NINCREASE+1
            IF (NINCREASE>20) GOTO 100
         ELSE
            NINCREASE=0
         ENDIF
      ENDDO
      IF (I==STEPS+1) THEN
         WRITE(*,'(A,E10.3,I8,E10.3)') ' warning in FIT_RATPOL: accuracy not reached rms=',DT,I,RMS
      ELSE
         WRITE(*,'(A,E10.3,I8,E10.3)') ' FIT_RATPOL:',DT,I,RMS
      ENDIF
      RETURN

    END SUBROUTINE FIT_RATPOL


!**********************************************************************
!
! Fit screened two electron integral  using a rational polynomial
! function is not assumed to be symmetric along the real axis
!
!**********************************************************************

    SUBROUTINE FIT_SCREENEDTWOE_HALF( EVALUE, X, F, W, XR, FR)
      USE main_mpi
      REAL(q) :: EVALUE
      REAL(q),INTENT(IN) :: X(:)    ! coordinates x
      COMPLEX(q),INTENT(IN) :: F(:) ! function values f
      REAL(q),INTENT(IN) :: W(:)    ! weights
      REAL(q),INTENT(IN) :: XR(:)   ! coordinates x along real axis
      COMPLEX(q) :: FR(:)           ! function values f along real axis
    ! local
      TYPE (ratpol) :: R
      INTEGER :: N=2
      REAL(q) :: X_FIT(SIZE(X)),W_FIT(SIZE(X))
      COMPLEX(q) :: F_FIT(SIZE(X))
      REAL(q) :: HEIGHT, A0, POLE, RMS
      COMPLEX(q) :: FP(SIZE(X))
      COMPLEX(q) :: FRP(SIZE(XR))
      INTEGER :: K, I

      INTEGER, SAVE :: NCALL = 0

      NCALL=NCALL+1
      N=2
      ! occupied state only add one more pole above Fermi-level
      CALL ALLOCATE_RATPOT(N, R)
      
      ! subtract Hartree Fock term
      ! scale function such that it has a height of 1
      ! and make function symmetric 
      K=SIZE(X)
      
      A0=F(K)
      HEIGHT=F(1)-A0
      POLE  =19

      X_FIT= X
      F_FIT=(F-A0)/HEIGHT
      W_FIT= W

      R%A0  =0
!     set position of poles in R%A
!     and amplitude in R%B

      
      IF (EVALUE<0) THEN
         ! occupied state
         R%A(1)= CMPLX(2.0_q, -POLE+EVALUE, q)
         R%A(2)= CMPLX(10.0_q, POLE+EVALUE, q)
         R%B(1)= CMPLX( 0.0_q, -40.0_q, q)
         R%B(2)= CMPLX(-10.0_q, -10.0_q, q)
      ELSE
         ! empty state
         R%A(1)= CMPLX(2.0_q,  POLE+EVALUE, q)
         R%A(2)= CMPLX(10.0_q,-POLE+EVALUE, q)
         R%B(1)= CMPLX(0.0_q,  40.0_q, q)
         R%B(2)= CMPLX(10.0_q, 10.0_q, q)
      ENDIF

      ! initial values
      ! poles at POLE
      ! 1/ (a + b x) = 1/b / (a/b + x ) = b* / (a* +x)
      R%B=1/R%B
      R%A=R%A*R%B

      WRITE(*,'("pole  ",12F8.3)') R%A/R%B
      WRITE(*,'("inten ",12F8.3)') 1/R%B

      CALL FIT_RATPOL(R, X_FIT, F_FIT, W_FIT)
      CALL REALLOCATE_RATPOT_IMAG(4, R)
      CALL FIT_RATPOL(R, X_FIT, F_FIT, W_FIT)

      WRITE(*,'("pole  ",12F8.3)') R%A/R%B
      WRITE(*,'("inten ",12F8.3)') 1/R%B

      CALL CALC_RATPOL(R, X, FP)
      FP=FP*HEIGHT+A0
!      WRITE(70+NCALL,'(5F14.7)') (ABS(X(I)),F(I),FP(I),I=1,SIZE(X))
!      WRITE(70+NCALL,*)

      CALL CALC_RATPOL_IMAG(R, XR, FRP)
      FRP=FRP*HEIGHT+A0

!     test dump function along real axis
!      WRITE(80+NCALL,'(5F14.7)') (XR(I),FR(I),FRP(I),I=1,SIZE(XR))
!      WRITE(80+NCALL,*)
      FR=FRP
      CALL DEALLOCATE_RATPOT(R)
    END SUBROUTINE FIT_SCREENEDTWOE_HALF

END MODULE ratpolfit


