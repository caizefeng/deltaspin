#include "symbol.inc"
!=============================================================================
! This module contains routines calculating the weights and abscissas
! for the Gauss Legendre,Lagurre,Hermite,Chebyshev and Jacobi
! quadratures.
!
! In addition this module capable of calculating the abscissas and weights
! for a given weight function. 
! This is done by using the Chebyshev/Wheeler algorithm, where instead of the
! monomes x^i orthogonal polynomials with known three term recurrence 
! coefficients alpha and beta are used in order to build generalized moments.
!                                                                          mK
!============================================================================
MODULE gauss_quad
 USE prec
 IMPLICIT NONE
 CONTAINS

!-----------------------------------------------------------------------
! Defines the weight function for which orthogonal polynomials should
! be calculated
!-----------------------------------------------------------------------
FUNCTION WEIGHTFUNCTION(X,N,A,B)
 USE prec 
 IMPLICIT NONE
 REAL(q) WEIGHTFUNCTION
 REAL(q) X
 REAL(q) A,B 
 INTEGER N

 IF (N==1) THEN                                           !linear
   WEIGHTFUNCTION=A*X
 ENDIF

 IF (N==2) THEN                                           !cubic linear
    WEIGHTFUNCTION=A*(X+X**3)/2._q
 ENDIF

 IF (N==3) THEN                                           !quadratic linear
    WEIGHTFUNCTION=A*(X+X**2)/2._q      
 ENDIF

 IF (N==4) THEN                                           !ACFDT
  IF ( ABS(A)>1E-6  ) THEN        
    WEIGHTFUNCTION=-LOG((1-B)*(1-X)+B*(1-X)**2)/A
  ELSE
   WEIGHTFUNCTION=-LOG(1-X)
  ENDIF
 ENDIF

 IF (N==40) THEN                                           !ACFDT default 
  IF ( ABS(A)>1E-6  ) THEN        
    WEIGHTFUNCTION=-LOG((1-B)*(1-X)+B*(1-X)**2)/A
  ELSE
   WEIGHTFUNCTION=-LOG(1-X)
  ENDIF
 ENDIF

 IF (N==201) THEN                                           !ACFDT (imag time)
  IF ( ABS(A)>1E-6  ) THEN        
    WEIGHTFUNCTION=-LOG((1-B)*(1-X)+B*(1-X)**2)/A
  ELSE
   WEIGHTFUNCTION=-LOG(1-X)
  ENDIF
 ENDIF
 
 IF (N==41) THEN                                          !simple logarithmic
  IF ( ABS(A)>1E-6  ) THEN        
   WEIGHTFUNCTION=-LOG(1-X)/A
  ELSE
   WEIGHTFUNCTION=-LOG(1-X)
  ENDIF  
 ENDIF

 IF (N==44 .OR. N==55) THEN                                          !logarithmic exponential
  IF ( ABS(A)>1E-6 ) THEN        
     WEIGHTFUNCTION=(-LOG(1-X))
    WEIGHTFUNCTION=((-LOG(1-X))**B)/A
  ELSE
    WEIGHTFUNCTION=(-LOG(1-X))**B
  ENDIF
 ENDIF
END FUNCTION WEIGHTFUNCTION

!*************************GAUSS_LEGENDRE*******************************
! This routine calculates the weights and abscissas of a Gauss Legendre
! integration. 
! This is taken from NR
!**********************************************************************
SUBROUTINE GAUSS_LEGENDRE(XX1,XX2,X,W,N)
 USE prec
 USE constant
 IMPLICIT NONE
 INTEGER N
 REAL(q) XX1,XX2
 REAL(q) X(N),W(N)
 !local variables
 REAL(q) X1,X2
 INTEGER I,J,M
 REAL(q) P,PP,XL,XM,Z,Z1
 
 X1=REAL(XX1,q)
 X2=REAL(XX2,q)

IF (N==0) THEN
 WRITE(*,*)'Gauss Legendre integration of zeroth order is a bad idea'
 STOP
ENDIF

 !roots are symmetric in the interval, we need to find only half of 
 !them
 M=(N+1)/2
 !variable transform
 XM=0.50_q*(X2+X1)
 XL=0.50_q*(X2-X1)

 find0 : DO I=1,M

         !starting guess for root  
         Z=COS(PI*(I-0.25_q)/(N+0.5_q))

1       CONTINUE 
 
        P=PLEG(Z,N)
        PP=PLEGDER(Z,N)
        
        Z1=Z 
        !Newton's root finding method
        Z=Z1-P/PP
     IF (ABS(Z-Z1)>1E-9) GOTO 1 
     X(I)=XM-XL*Z                    !scale the root to interval
     X(N+1-I)=XM+XL*Z                !symmetric counterpart of root         
     W(I)=2._q*XL/((1._q-Z*Z)*PP*PP) !weight
     W(N+1-I)=W(I)                   !antisymmetric counterpart
 ENDDO find0
RETURN

ENDSUBROUTINE GAUSS_LEGENDRE

!*************************GAUSS_LAGUERRE*******************************
! This routine calculates the weights and abscissas of a Gauss Laguerre
! integration of type \alpha 
! \int_0^inf dx x^alpha e^-x f(x) ~ \sum_j=1^N W_jf(x_j)
! Initial guesses for roots taken from Sroud and 
! Secrest: Gaussian Quadrature Formulas, 1966
! This is taken from NR
!**********************************************************************
SUBROUTINE GAUSS_LAGUERRE(X,W,N,ALF)
 USE prec
  INTEGER N,IMAX
  REAL(q) W(N),X(N)
  REAL(q) ALF
  REAL(q) EPSI
  PARAMETER (EPSI=3.E-14_q,IMAX=10) 
  INTEGER I,ITS,J
  REAL(q) AI
  REAL(q) P1,P2,P3,PP,Z,Z1
  
  DO  I=1,N 
  IF(I.EQ.1)THEN 
   Z=(1.+ALF)*(3.+.92*ALF)/(1.+2.4*N+1.8*ALF)
  ELSE IF(I.EQ.2)THEN 
   Z=Z+(15.+6.25*ALF)/(1.+.9*ALF+2.5*N)
  ELSE 
   AI=I-2
   Z=Z+((1.+2.55*AI)/(1.9*AI)+1.26*AI*ALF/&
        (1.+3.5*AI))*(Z-X(I-2))/(1.+.3*ALF)
  ENDIF
 
  DO ITS=1,IMAX
    P1=1._q
    P2=0._q
   
   DO  J=1,N 
    P3=P2 
    P2=P1
    P1=((2*J-1+ALF-Z)*P2-(J-1+ALF)*P3)/J
   ENDDO 
    
    PP=(N*P1-(N+ALF)*P2)/Z
    Z1=Z
    Z=Z1-P1/PP 
   
    IF(ABS(Z-Z1).LE.EPSI)GOTO 1
  ENDDO 

  WRITE(*,*)'internal Error in VASP: TOO MANY ITERATIONS IN GAULAG'
  STOP

1 X(I)=Z 
 W(I)=-EXP(GAMMLN( REAL(ALF+N,q) )-GAMMLN(REAL(N,q)))/(PP*N*P2)
 ENDDO 
 RETURN
 ENDSUBROUTINE GAUSS_LAGUERRE

!*************************GAUSS_HERMITE*******************************
! This routine calcualtes the weights and abscissas of a Gauss Hermite
! integration using PHERM 
! This is taken from NR
!*********************************************************************
SUBROUTINE GAUSS_HERMITE(X,W,N)
 USE prec
 IMPLICIT NONE
 REAL(q) W(N),X(N)
 INTEGER N
 !local variables
 INTEGER I, ITS,J,M,MAXIT
 REAL(q) P,PP,Z,Z1
 PARAMETER(MAXIT=40)
 
 !roots are symmetric
 M=(N+1)/2

 DO I=1,M
     IF(I==1) THEN                     !intial guess for largest root
      Z=SQRT(REAL(2*N+1,q))-1.855750_q*(2*N+1)**(-0.166670_q)
     ELSEIF(I==2) THEN                 !guess for second largest root
            Z=Z-1.140_q*N**0.4260_q/Z
     ELSEIF(I==3) THEN                 !guess for third largest root
            Z=1.860_q*Z-0.860_q*X(1)
     ELSEIF(I==4) THEN                 !guess for forth largest root 
            Z=1.910_q*Z-0.910_q*X(2)        
     ELSEIF(I>4) THEN                  !guess for all other roots 
            Z=2.0_q*Z-X(I-2)  
     ENDIF

 DO ITS=1,MAXIT
   P=PHERM(Z,N,0._q,0._q)
  PP=SQRT(2.0_q*N)*PHERM(Z,N-1,0._q,0._q)
   
   !again Newton method
  Z1=Z
   Z=Z1-P/PP

 IF(ABS(Z-Z1)<1E-9) GOTO 1
ENDDO   

WRITE(*,*) 'Too many iterations in GAUSS_HERMITE'

 !if check Ok, we have the desired root
1      X(I)=Z                 !store the root
   X(N+1-I)=-Z             !store the corresponding antisymmetric root  
       W(I)=2.0_q/(PP*PP)      !store the weight
   W(N+1-I)=W(I)           

ENDDO

RETURN
END SUBROUTINE GAUSS_HERMITE

!*************************GAUSS_JACOBI**********************************
! This routine calcualtes the weights and abscissas for a Gauss Jacobi
! integral quadrature.  
! This is taken from NR
!***********************************************************************
SUBROUTINE GAUSS_JACOBI(XX1,XX2,X,W,N,A,B)
 USE prec
 IMPLICIT NONE
 INTEGER N
 REAL(q) W(N),X(N)
 REAL(q) A,B,XX1,XX2
 !local variables
 INTEGER I,ITS,J,MAXIT
 PARAMETER(MAXIT=40)
 REAL(q) ALFBET, AN, BN, R1, R2, R3
 REAL(q) P,PP,Z,Z1
 REAL(q) ALPHA,BETA,X1,X2

 X1=REAL( (XX2-XX1)/2 ,q)
 X2=REAL( (XX1+XX2)/2 ,q)
 
 
 ALPHA=REAL(A,q)
 BETA=REAL(B,q)

 !loop over roots
 DO I=1,N
   IF(I==1) THEN                                             !initial guess for largest root
     AN=ALPHA/N
     BN=BETA/N
     R1=(1.+ALPHA)*(2.78/(4.+N*N)+0.768*AN/N)
     R2= 1.+1.48*AN+0.96*BN+0.452*AN*AN+0.83*AN*BN
     Z=1.-R1/R2
   ELSEIF(I==2)THEN                                          !initial guess for second largest root  
     R1=(4.1+ALPHA)/((1.+ALPHA)*(1.+.156*ALPHA))
     R2=1.+.06*(N-8.)*(1.+.12*ALPHA)/N
     R3=1.+.012*BETA*(1.+.25*ABS(ALPHA))/N
     Z=Z-(1.-Z)*R1*R2*R3
   ELSEIF(I==3)THEN                                          !initial guess for third largest root
     R1=(1.67+.28*ALPHA)/(1.+.37*ALPHA)
     R2=1.+.22*(N-8.)/N
     R3=1.+8.*BETA/((6.28+BETA)*N*N)
     Z=Z-(X(1)-Z)*R1*R2*R3
   ELSEIF(I==N-1)THEN                                        !initial guess for second smallest root 
     R1=(1.+.235*BETA)/(.766+.119*BETA)
     R2=1./(1.+.639*(N-4.)/(1.+.71*(N-4.)))
     R3=1./(1.+20.*ALPHA/((7.5+ALPHA)*N*N))
     Z=Z+(Z-X(N-3))*R1*R2*R3
   ELSEIF(I==N)THEN                                          !initial guess for smallest root
     R1=(1.+.37*BETA)/(1.67+.28*BETA)
     R2=1./(1.+.22*(N-8.)/N)
     R3=1./(1.+8.*ALPHA/((6.28+ALPHA)*N*N))
     Z=Z+(Z-X(N-2))*R1*R2*R3
   ELSE                                                      !initial guess for other roots 
     Z=3.*X(I-1)-3.*X(I-2)+X(I-3)
 ENDIF
   
 ALFBET=ALPHA+BETA
 
 !using Newton's method
 DO ITS=1,MAXIT  
    P=PJACOBI(Z,N,-1._q,1._q,A,B)
  !derivative of Jacobi polynomial
   PP=PJACOBIDER(Z,N,-1._q,1._q,A,B)


  Z1=Z
  Z =Z1-P/PP

  IF(ABS(Z-Z1)<1E-9) GOTO  1
 ENDDO

  WRITE(*,*)'Too many iterations in GAUSS_JACOBI',I

!save zeros and weights
1    X(I)=Z*X1+X2
     W(I)=EXP(GAMMLN(ALPHA+REAL(N,q))+GAMMLN(BETA+REAL(N,q))-GAMMLN(N+1._q)-&
      GAMMLN(N+ALFBET+1._q))*(2*N+ALFBET)*2.**ALFBET/(PP*PJACOBI(Z,N-1,-1._q,1._q,A,B))*X1
 

ENDDO


 RETURN
END SUBROUTINE GAUSS_JACOBI

SUBROUTINE GAUSSJAC(X,W,N,ALF,BET)
 USE prec

 INTEGER N,MAXIT
 REAL(q) ALF,BET 
 REAL(q) W(N),X(N)
 REAL(q) EPS
 PARAMETER (EPS=3.E-14_q,MAXIT=10)
 INTEGER I,ITS,J
 REAL(q) ALFBET,AN,BN,R1,R2,R3
 REAL(q) A,B,C,P1,P2,P3,PP,TEMP,Z,Z1
 
 DO I=1,N
 
 IF(I.EQ.1)THEN
 
 AN=ALF/N
 BN=BET/N
 R1=(1.+ALF)*(2.78/(4.+N*N)+.768*AN/N)
 R2=1.+1.48*AN+.96*BN+.452*AN*AN+.83*AN*BN
 Z=1.-R1/R2
 ELSE IF(I.EQ.2)THEN
 
 R1=(4.1+ALF)/((1.+ALF)*(1.+.156*ALF))
 R2=1.+.06*(N-8.)*(1.+.12*ALF)/N
 R3=1.+.012*BET*(1.+.25*ABS(ALF))/N
 Z=Z-(1.-Z)*R1*R2*R3
 ELSE IF(I.EQ.3)THEN
 
 R1=(1.67+.28*ALF)/(1.+.37*ALF)
 R2=1.+.22*(N-8.)/N
 R3=1.+8.*BET/((6.28+BET)*N*N)
 Z=Z-(X(1)-Z)*R1*R2*R3
 ELSE IF(I.EQ.N-1)THEN
 
 R1=(1.+.235*BET)/(.766+.119*BET)
 R2=1./(1.+.639*(N-4.)/(1.+.71*(N-4.)))
 R3=1./(1.+20.*ALF/((7.5+ALF)*N*N))
 Z=Z+(Z-X(N-3))*R1*R2*R3
 ELSE IF(I.EQ.N)THEN
 
 R1=(1.+.37*BET)/(1.67+.28*BET)
 R2=1./(1.+.22*(N-8.)/N)
 R3=1./(1.+8.*ALF/((6.28+ALF)*N*N))
 Z=Z+(Z-X(N-2))*R1*R2*R3
 ELSE
 
 Z=3.*X(I-1)-3.*X(I-2)+X(I-3)
 ENDIF
 ALFBET=ALF+BET
 DO ITS=1,MAXIT
 
 TEMP=2._q+ALFBET
 P1=(ALF-BET+TEMP*Z)/2._q
 P2=1._q
 
 DO J=2,N
 
 P3=P2
 P2=P1
 TEMP=2*J+ALFBET
 A=2*J*(J+ALFBET)*(TEMP-2._q)
 B=(TEMP-1._q)*(ALF*ALF-BET*BET+TEMP*(TEMP-2._q)*Z)
 C=2._q*(J-1+ALF)*(J-1+BET)*TEMP
 P1=(B*P2-C*P3)/A
 ENDDO 
 PP=(N*(ALF-BET-TEMP*Z)*P1+2._q*(N+ALF)*(N+BET)*P2)/(TEMP*(1._q-Z*Z))
 
 Z1=Z
 Z=Z1-P1/PP
 
 IF(ABS(Z-Z1).LE.EPS)GOTO 1
 ENDDO 
  
 WRITE(*,*)' TOO MANY ITERATIONS IN GAUJAC'
 STOP
 
1 X(I)=REAL(Z,q)
 
 W(I)=REAL( EXP(GAMMLN( REAL(ALF+N,q) )+GAMMLN( REAL(BET+N,q) )-GAMMLN(REAL(N+1.,q))-GAMMLN(REAL(N+ALFBET+1.,q) ))*TEMP*2.**ALFBET/(PP*P2), q)
 ENDDO 
  
 RETURN
END SUBROUTINE GAUSSJAC

!*************************GAUSS_CHEBYSHEV1*******************************
! This routine calcualtes the weights and abscissas for a Gauss Chebyshev
! integral quadrature.(first kind)
!***********************************************************************
SUBROUTINE GAUSS_CHEBYSHEV1(XX1,XX2,X,W,N)
 USE prec
 USE constant
 IMPLICIT NONE
 INTEGER N
 REAL(q) XX1,XX2
 REAL(q) X(N),W(N)
 !local variables
 REAL(q) X1,X2,XX(N),WW(N)
 INTEGER I,J,M
  
 M=N 
 !change of variables for integration 
 X1=REAL( (XX2-XX1)/2 ,q)
 X2=REAL( (XX1+XX2)/2 ,q)

  DO I=1,M
  XX(I)=COS(REAL((2*I-1),q)/REAL(2*M,q)*PI)
  WW(I)=PI/REAL(M,q)
  ENDDO

  X(:)=X1*XX(:)+X2
  W(:)=WW(:)*X1

 RETURN
ENDSUBROUTINE GAUSS_CHEBYSHEV1

!*************************GAUSS_CHEBYSHEV1*******************************
! This routine calcualtes the weights and abscissas for a Gauss Chebyshev
! integral quadrature.(second kind)
!***********************************************************************
SUBROUTINE GAUSS_CHEBYSHEV2(XX1,XX2,X,W,N)
 USE prec
 USE constant
 IMPLICIT NONE
 INTEGER N
 REAL(q) XX1,XX2
 REAL(q) X(N),W(N)
 !local variables
 REAL(q) X1,X2,XX(N),WW(N)
 INTEGER I,J,M
  
 M=N 
 !change of variables for integration 
 X1=REAL( (XX2-XX1)/2 ,q)
 X2=REAL( (XX1+XX2)/2 ,q)

 DO I=1,M
  XX(I)=COS(I*PI/REAL(M+1,q))
  WW(I)=PI/REAL(M+1,q)*(SIN(REAL(I,q)*PI/REAL(M+1,q)))**2
 ENDDO
  
  X(:)=X1*XX(:)+X2
  W(:)=WW(:)*X1

 RETURN
ENDSUBROUTINE GAUSS_CHEBYSHEV2

!*********************GAUSSC_FROM_RECURRENCE*********************
! This routine calculates the zeros and the corresponding weights
! for a set of orthogonal polynomials p_j given by the coefficients 
!
!       a_j=(xp_j|p_j)/(p_j|p_j),        j=0,1,....
!       b_j=(p_j|p_j)/(p_j-1|p_j-1)      j=1,2,...., b_0=0
!
! and satisfying the recurrence formula:
!
!       p_-1(x)=0; p_0(x)=0
!       p_j+1(x)=(x-a_j)p_j(x)-b_jp_j-1(x) ,  j=0,1,2,.....
!
! In addition the zero moment M=(1|1)  must be passed.
!
! N determines the desired number of zeros and weights, calculated 
! by finding roots of the corresponding polynomial of order N. 
! a_0-(n-1)=A(1:N) and b_0-(n-1)=B(1:N) must be given!  
!**********************************************************************
SUBROUTINE GAUSSC_FROM_RECURRENCE(N,A,B,M,X,W)
  USE prec
  IMPLICIT NONE
  INTEGER N
  REAL(q) M,A(N),B(N),W(N),X(N)
  REAL(q),ALLOCATABLE ::  Z(:,:),DIAG(:),OFFDIAG(:)
  INTEGER I,J,INF
  REAL(q) WORK(2*N-2)
  
  ALLOCATE(Z(1:N,1:N)); Z=0._q
  ALLOCATE(DIAG(1:N),OFFDIAG(1:N-1)); DIAG=0._q; OFFDIAG=0._q

  IF (SIZE(DIAG)/=SIZE(A)) THEN
   WRITE(*,*)'Error in GAUSSC_FROM_RECURRENCE, size of DIAG and A differ:',SIZE(DIAG),SIZE(A)
   STOP
  ENDIF

  IF ((SIZE(OFFDIAG)+1)/=SIZE(B)) THEN
   WRITE(*,*)'Error in GAUSSC_FROM_RECURRENCE, size of OFFDIAG and B incompatible:',SIZE(OFFDIAG),SIZE(B)
   STOP
  ENDIF

  DIAG(:)=A(:)

  DO I=1,N-1
    IF(B(I)<0) THEN
     WRITE(*,*)'Error in GAUSSC_FROM_RECURRENCE, B<0 for I=',I,B(I)
     STOP
    ELSE
     OFFDIAG(I)=SQRT(B(I+1))
    ENDIF
  ENDDO 

   !solve the symmetric tridiagonal eigenvalue problem
   CALL DSTEV('V',N,DIAG(1),OFFDIAG(1),Z(1,1),N,WORK,INF)

   IF(INF/=0)THEN
    WRITE(*,*) 'ERROR in GAUSSC_FROM_RECURRENCE: DSTEV failed',INF 
    STOP
   ENDIF
   
   !store weights and zeros  
   DO I=1,N
     X(I)=DIAG(I)
     W(I)=M*(Z(1,I)**2)  
   ENDDO 

  DEALLOCATE(DIAG,OFFDIAG,Z)

  RETURN
 ENDSUBROUTINE GAUSSC_FROM_RECURRENCE

!**********************CALC_RECURR_COF*********************************
! This routine calculates the recurrence coefficients a_j, b_j (j=0,..N-1
! for the orthogonal polynomials of WEIGHTFUNCTION using Wheeler's alorithm. 
! On input AP(1:2N-1), B(1:2N-1) are the recurrence coefficients 
! a'_j, b'_j of the orthogonal polynomials used for calculating the 
! generalized moments nu_j given in ANU(1:2N). 
! 
!**********************************************************************
 SUBROUTINE CALC_RECURR_COF(N,ANU,AP,BP,A,B)
  USE prec
   INTEGER  N                                !how many coefficients shall we calculate?
   INTEGER  NTOT                             
   REAL(q) AP(1:2*N+1),BP(1:2*N+1)                !known recurrence coefficients
   REAL(q) ANU(1:2*N+1)                        !generalized moments
   REAL(q) A(1:N),B(1:N)                        !calculated coefficients 
   INTEGER K,L,I
   REAL(q),ALLOCATABLE :: SIG(:,:),ALPHA(:),BETA(:),NU(:),AA(:),BB(:)

   ALLOCATE(SIG(-1:N-1,-1:2*N),ALPHA(0:2*N),BETA(0:2*N),NU(0:2*N))
   ALLOCATE(AA(0:N-1),BB(0:N-1)); AA=0._q; BB=0._q  
 

   IF (SIZE(ALPHA)/=SIZE(AP)) THEN
    WRITE(*,*)'ERROR in CALC_RECURR_COF: Arrays AP and ALPHA have different size:',SIZE(AP),SIZE(ALPHA)
    STOP
   ENDIF

   IF (SIZE(BETA)/=SIZE(BP)) THEN
    WRITE(*,*)'ERROR in CALC_RECURR_COF: Arrays BP and BETA have different size:',SIZE(BP),SIZE(BETA)
    STOP
   ENDIF

   IF (SIZE(NU)/=SIZE(ANU)) THEN
    WRITE(*,*)'ERROR in CALC_RECURR_COF: Arrays ANU and NU have different size:',SIZE(ANU),SIZE(NU)
    STOP
   ENDIF

   IF (SIZE(AA)/=SIZE(A)) THEN
    WRITE(*,*)'ERROR in CALC_RECURR_COF: Arrays AA and A have different size:',SIZE(AA),SIZE(A)
    STOP
   ENDIF
 
   IF (SIZE(BB)/=SIZE(B)) THEN
    WRITE(*,*)'ERROR in CALC_RECURR_COF: Arrays BB and B have different size:',SIZE(BB),SIZE(B)
    STOP
   ENDIF

  DO I=0,2*N
    ALPHA(I)=AP(I+1)
    BETA(I)=BP(I+1)
    NU(I)=ANU(I+1)
  ENDDO

!test
!  WRITE(*,*)'OLD'
!    DO I=0,2*N
!      WRITE(*,'(I4,5F30.10)') I,ALPHA(I),BETA(I),NU(I)
!    ENDDO

  !initialize
   SIG(:,-1)=0._q
   SIG(:,0)=0._q

   DO L=1,2*N-2
    SIG(-1,L)=0._q
   ENDDO
   DO L=0,2*N-1
    SIG(0,L)=NU(L)
   ENDDO
    
   AA(0)=ALPHA(0)+NU(1)/NU(0)
   BB(0)=0._q

   DO K=1,N-1
    DO L=K,2*N-K-1
     
     SIG(K,L)=SIG(K-1,L+1)+&
              ( ALPHA(L) - AA(K-1) )*SIG(K-1,L) - BB(K-1)*SIG(K-2,L) +&
              BETA(L)*SIG(K-1,L-1)
    
    ENDDO
     AA(K)=ALPHA(K) - SIG(K-1,K)/SIG(K-1,K-1) + SIG(K,K+1)/SIG(K,K)
     BB(K)=SIG(K,K)/SIG(K-1,K-1)
   ENDDO

!test
!   WRITE(*,*)'SIG'
!      WRITE(*,'(12I20)') (K,K=-1,N-1)
!    DO L=-1,2*N
!      WRITE(*,'(I4,12F20.10)') L,SIG(-1:N-1,L)
!    ENDDO
!      WRITE(*,'(I4,2F30.10)') 

!   WRITE(*,*)'NEW'
!    DO I=0,N-1
!      WRITE(*,'(I4,5F30.10)') I,AA(I),BB(I),NU(I)
!    ENDDO
!      WRITE(*,'(I4,2F30.10)') 

  DO I=1,N
    A(I)=AA(I-1)
    B(I)=BB(I-1)
  ENDDO

  DEALLOCATE(SIG,ALPHA,BETA,NU)  
 
  RETURN
  ENDSUBROUTINE CALC_RECURR_COF

!*******************CALCULATE_WEIGHTS_AND_ZEROS*************************
!
! calculates points and weights for a gaussian quadrature of order N 
! for the chosen  WEIGHTFUNCTION
! 
! The following set of polynomials \pi_i are used for sampling the 
! integration interval [XMIN,XMAX]:
!
! TYP=1		Legendre  polynomials 
! TYP=2		Chebyshev polynomials 
! TYP=3		Laguerre  polynomials 
! TYP=4		Hermite   polynomials 
! 
! Note that the weight function w should be similar to the
! weight function of the \pi_i
!**********************************************************************
  SUBROUTINE CALCULATE_WEIGHTS_AND_ZEROS(GRIDV,N,XMIN,XMAX,WA,WB,X,W)
   USE prec
     INTEGER  GRIDV                                   !grid version
     INTEGER  N                                       !desired order of quadrature 
     REAL(q)     XMIN, XMAX                              !integration interval
     REAL(q) X(N),W(N)                                !evaluation points and corresponding weights
     INTEGER  I,J  
     REAL(q)  WA,WB                                     !coefficients for weight function
     REAL(q), ALLOCATABLE :: ANU(:),AP(:),BP(:),A(:),B(:)    
    !test
     INTEGER  TYP                                     !type of sampling polynomial
     REAL(q) DX
     REAL(q) ALPHA
 
     ALLOCATE( ANU( 1:2*N+1 ) )
     ALLOCATE( AP( 1:2*N+1 ), BP( 1:2*N+1 ) )
     ALLOCATE( A(1:N),B(1:N) )
     ANU=0._q
     AP= 0._q
     BP= 0._q
     A=  0._q
     B=  0._q 
     
    !the weight function defining \pi_n(x) should be similar to
    !the chosen weight function WEIGHTFUNCTION
     IF (GRIDV==40 .OR. GRIDV==150 ) THEN
          TYP=1          !Legendre
     ELSEIF (GRIDV==3) THEN
          TYP=1
     ELSE
          TYP=1
     ENDIF

    !Legendre polynomials
    !---------------------------------------------------------------------------------    
     IF (TYP == 1 ) THEN !(choose this for almost constant WEIGHTFUNCTION in [-a,b])   
    !---------------------------------------------------------------------------------    
     IF ( ABS(XMAX-XMIN)<1E-6 ) THEN
       WRITE(*,'(/A,I4)')"Error in CALCULATE_WEIGHTS_AND_ZEROS: XMIN too small for TYP=",TYP
       WRITE(*,*)" "
       STOP
     ENDIF
     !calculate generalized moments (we need monic polynomials!)

    IF(GRIDV==55) CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDSQRTL_MOMENT0)
    !IF(GRIDV==55)  CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT0)
    IF(GRIDV==44)  CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT0)
    IF(GRIDV==40)  CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT0)
    IF(GRIDV==150) CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT0)
    IF(GRIDV==3 )  CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,TRAPEZ_MOMENT)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT0)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDSQRTL_MOMENT0)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDSQRTU_MOMENT0)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDINF_MOMENT0)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDEXP_MOMENT0)
     !CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDPOINT_MOMENT1_0INF)

     !for quadrature of Nth order we need  2*N+1 recurrence relation coefficients 
     !choosing the convention alpha_0 = AP(1), etc.
        DO I=1,2*N+1
          AP(I)=REAL( (XMAX+XMIN)/2,q )
          BP(I)=REAL( (I-1)**2, q)/( ( (2./(XMAX-XMIN) )**2)*REAL( 4*((I-1)**2)-1, q) )
        ENDDO
   
    !Chebyshev polynomials first kind
    !---------------------------------------------------------------------------------    
     ELSEIF (TYP == 2 ) THEN !(choose this for  WEIGHTFUNCTION ~ 1/sqrt(1-x**2) in [a,b])   
    !---------------------------------------------------------------------------------    

     ! use this for a bounded WEIGHTFUNCTION in [a,b]
    IF(GRIDV==3 )  CALL QSIMP_MOMENT(PTMON, XMIN, XMAX, WA,WB, GRIDV, ANU,2*N,TRAPEZ_MOMENT)
    IF(GRIDV==40) CALL QSIMP_MOMENT(PLEGMON, XMIN, XMAX,WA,WB,GRIDV, ANU,2*N,MIDSQRTL_MOMENT0)

     !for quadrature of Nth order we need  2*N+1 recurrence relation coefficients 
     !choosing the convention alpha_0 = AP(1), etc.
        DO I=1,2*N+1
          AP(I)=REAL((XMAX+XMIN)/2. , q )
         IF(I<=2) BP(I)=REAL( 1./(2*(2./(XMAX-XMIN))**2) , q)
         IF(I>2 ) BP(I)=REAL( 1./(4*(2./(XMAX-XMIN))**2) , q)
        ENDDO

    !Chebyshev polynomials second kind
    !---------------------------------------------------------------------------------    
     ELSEIF (TYP == 3 ) THEN !(choose this for  WEIGHTFUNCTION ~ sqrt(1-x**2) in [a,b])   
    !---------------------------------------------------------------------------------    

     CALL QSIMP_MOMENT(PUMON, XMIN, XMAX, WA, WB, GRIDV,  ANU,2*N,MIDSQRTL_MOMENT0)

     !for quadrature of Nth order we need  2*N+1 recurrence relation coefficients 
     !choosing the convention alpha_0 = AP(1), etc.
        DO I=1,2*N+1
          AP(I)=REAL((XMAX+XMIN)/2. , q )
          BP(I)=REAL( 1./4 , q)
        ENDDO

    !Laguerre polynomials
    !---------------------------------------------------------------------------------    
     ELSEIF (TYP == 4 ) THEN !(choose this for  WEIGHTFUNCTION ~ exp(-x) in [a,b])   
    !---------------------------------------------------------------------------------    
     !calculate generalized moments (we need monic polynomials!)
     ALPHA=1

     ! use this for a bounded WEIGHTFUNCTION in [a,b]
     CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,TRAPEZ_MOMENT,ALPHA)
     
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDPOINT_MOMENT1,ALPHA)
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDSQRTL_MOMENT1,ALPHA)
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDSQRTU_MOMENT1,ALPHA)
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDINF_MOMENT1,ALPHA)
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDEXP_MOMENT1,ALPHA)
     !CALL QSIMP_MOMENT(PLAGMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDPOINT_MOMENT1_0INF,ALPHA)
     !try integrating moments using a Gauss-Laguerre quadrature of order ALPHA
     !CALL GAUSSLAGI_MOMENT(ANU,2*N,ALPHA)
     
     !for quadrature of Nth order we need  2*N+1 recurrence relation coefficients 
     !choosing the convention alpha_0 = AP(1), etc.
        DO I=1,2*N+1
          AP(I)=REAL(2*(I-1)+1+ALPHA, q )
          BP(I)=REAL( (I-1)*(I-1+ALPHA), q)
        ENDDO

    !Hermite polynomials
    !---------------------------------------------------------------------------------    
     ELSEIF (TYP == 5 ) THEN !(choose this for  WEIGHTFUNCTION ~ exp(-x*x) in [a,b])   
    !---------------------------------------------------------------------------------    

     ! use this for a bounded WEIGHTFUNCTION in [a,b]
     CALL QSIMP_MOMENT(PHERMMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,TRAPEZ_MOMENT)
     !CALL QSIMP_MOMENT(PHERMMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDSQRTL_MOMENT0)
     !CALL QSIMP_MOMENT(PHERMMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDINF_MOMENT0)
     !CALL QSIMP_MOMENT(PHERMMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDEXP_MOMENT0)
     !CALL QSIMP_MOMENT(PHERMMON, XMIN, XMAX, WA, WB, GRIDV, ANU,2*N,MIDPOINT_MOMENT1_0INF)

     !for quadrature of Nth order we need  2*N+1 recurrence relation coefficients 
     !choosing the convention alpha_0 = AP(1), etc.
        DO I=1,2*N+1
          AP(I)=REAL((XMAX+XMIN)/2. , q )
          BP(I)=REAL( (I-1)/(2*(2./(XMAX-XMIN))**2) , q)
        ENDDO

    ENDIF

   !at this stage we have the generalized moments and the corresponding
   !recurrence coefficients of the sampling polynomial   

   !now calculate the recurrence coefficients of the quadrature polynomial
   CALL CALC_RECURR_COF(N,ANU,AP,BP,A,B)
   
  ! WRITE(*,*)'NEW'
  !  DO I=1,N
  !    WRITE(*,'(I4,5F30.10)') I,A(I),B(I),ANU(I)
  !  ENDDO
  !    WRITE(*,'(I4,2F30.10)') 

   !finally calculate weights and roots
   CALL GAUSSC_FROM_RECURRENCE(N,A,B,ANU(1),X,W)
    
  DEALLOCATE(ANU,AP,BP,A,B)
 
   RETURN
  ENDSUBROUTINE CALCULATE_WEIGHTS_AND_ZEROS

!*************************GAUSS_INTEGRATION****************************
! This routine performs a Gauss integration for given weights W and 
! evaluation points X
!**********************************************************************
SUBROUTINE GAUSS_INTEGRATION(N,A,B,X,W,R)
 USE prec
 IMPLICIT NONE
 INTEGER  N                                 ! quadrature order
 REAL(q) X(N),W(N)                          ! moments
 REAL(q) R                                  ! result
 !local variables
 INTEGER I,J,K
 REAL(q) A,B
 REAL(q) F,S,G,S2

#define F(x) COS(x)*WEIGHTFUNCTION(x,40,A,B)
#define G(x) COS(x)

 !   DO I=1,N
 !     WRITE(*,'(I4,3F30.10)') I,X(I),W(I)
 !   ENDDO
 !     WRITE(*,'(I4,2F30.10)') 

  S=0._q
  S2=0._q
   DO J=1,N 
   S=S+F(X(J))*W(J)
   S2=S2+G(X(J))*W(J)
   ENDDO

WRITE(*,'(2F20.10)')S,S2

 !save the result 
  R=S

#undef F
#undef G

 RETURN
ENDSUBROUTINE GAUSS_INTEGRATION

!*************************GAUSSLAGI_MOMENT****************************
! This routine performs a Gauss Laguerre integration for for the 
! generalized moments
!**********************************************************************
SUBROUTINE GAUSSLAGI_MOMENT(NU,N,ALF)
 USE prec
 IMPLICIT NONE
 INTEGER  N, GRIDT                                 ! quadrature order
 REAL(q) WA,WB
 REAL(q) NU(N+1)                            ! moments
 REAL(q) ALF                                  ! laguerre moment parameter 
 !local variables
 INTEGER I,J,K,ORDER
 PARAMETER(ORDER=5)
 REAL(q) S,XI(10),WI(10)
 REAL(q), ALLOCATABLE :: MOM(:)

  ALLOCATE(MOM(0:N)); MOM=0._q
  IF (SIZE(MOM)/=SIZE(NU)) THEN
   WRITE(*,*)'Error in GAUSSLAGI_MOMENT, MOM and NU sizes differ:',SIZE(MOM),SIZE(NU)
   STOP
  ENDIF

  MOM=0._q
!--------------------------------------------------------------
!loop over moments, start with zeroth moment
!\nu_0=ANU(1)
 DO I=0,N              
!--------------------------------------------------------------
  !calculate abscissas and weights of a Gauss Laguerre quadrature
     CALL GAUSS_LAGUERRE(XI,WI,ORDER,ALF)
!    DO J=1,ORDER
!      WRITE(*,'(I4,3F30.10)') I,XI(J),WI(J)
!    ENDDO
!      WRITE(*,'(I4,2F30.10)') 
     
  !sum
   DO J=1,ORDER
    MOM(I)=MOM(I)+WI(J)*PLAGMON( XI(J), I, 0._q,0._q,ALF)*EXP( XI(J) )*WEIGHTFUNCTION( XI(J),GRIDT,WA,WB ) 
   ENDDO


!--------------------------------------------------------------
 ENDDO               !loop moments
!--------------------------------------------------------------
  !save the moments 
 DO I=1,N+1
  NU(I)=MOM(I-1)
 ENDDO

 DEALLOCATE(MOM)

 RETURN
ENDSUBROUTINE GAUSSLAGI_MOMENT

!***************************QSIMP_MOMENT********************************
! Returns as S the general moment NU of the function F with respect to 
! the weight function W using the Simpson integration formula.
! 
! nu(N)=\int_A^B F(x,N)W(x) dx
! 
! ESP can be set  to the desired fractional accuracy and JMAX so that 
! 2^(JMAX-1) is the maximum allowed number of 
! steps. 
!
! For the function CHOOSE choose the following subroutines
! MIDPOINT_MOMENT0,1,2....for finite integration intervals and bounded
!                         weight functions (0 for ALPHA & BETA, 1 for 
!                         BETA missing and 2 for ALPHA & present)
! MIDINF_MOMENT0,1,2......for bounded integrands and infinite intervals
!                         where either A->inf & B> 0 or B->-inf & A<0
! MIDSQRTL_MOMENT0,1,2.....for finite integration intervals and integrands
!                         having a 1/g (0<g<1) integrable singularity at 
!                         either the lower integration limit. 
! MIDSQRTU_MOMENT0,1,2.....for finite integration intervals and integrands
!                         having a 1/g (0<g<1) integrable singularity at 
!                         either the upper integration limit
!
! MIDPOINT_MOMENT_0INF....for integration interval (0,inf)
!***********************************************************************
 SUBROUTINE QSIMP_MOMENT(F,AA,BB,WA,WB,GRIDTYP,NU,N,CHOOSE,ALPHA,BETA)
  USE prec
  INTEGER N,GRIDTYP                            ! order of moment
  REAL(q) NU(N+1)
  REAL(q) AA,BB
  REAL(q) WA,WB
  EXTERNAL :: CHOOSE
  REAL(q),INTENT(IN), OPTIONAL :: ALPHA
  REAL(q),INTENT(IN), OPTIONAL :: BETA    ! optional parameter
  REAL(q), EXTERNAL :: F               !sampling polynomial  
  INTEGER JMAX   
  REAL(q) EPSI,S,A,B                   !maximum number of steps
  PARAMETER (EPSI=1.E-6, JMAX=50)      !accuracy  
  INTEGER J,I
  REAL(q) OS,OST,ST
  REAL(q), ALLOCATABLE :: MOM(:)

  ALLOCATE(MOM(0:N)); MOM=0._q
  IF (SIZE(MOM)/=SIZE(NU)) THEN
   WRITE(*,*)'Error in QSIMP_MOMENT, MOM and NU sizes differ:',SIZE(MOM),SIZE(NU)
   STOP
  ENDIF
 
  
  OST=-1.E30
  OS=-1.E30
 
  A=REAL(AA,q)
  B=REAL(BB,q)
  
!--------------------------------------------------------------
!loop over moments, start with zeroth moment
!\nu_0=ANU(1)
 DO I=0,N              
!--------------------------------------------------------------
  IF ((.NOT.PRESENT(ALPHA)) .AND. (.NOT.PRESENT(BETA)) ) THEN
    DO J=1,JMAX 
     CALL CHOOSE(F,A,B,WA,WB,GRIDTYP,ST,J,I)
     S=(4._q*ST-OST)/3._q
     IF (J>5) THEN                       !avoid spurious early convergence       
          IF (ABS(S-OS)<EPSI ) GOTO 10
     ENDIF
     OS=S
     OST=ST
   ENDDO 
  
 
  ELSEIF ((PRESENT(ALPHA)) .AND. (.NOT.PRESENT(BETA)) ) THEN
    DO J=1,JMAX 
     CALL CHOOSE(F,A,B,WA,WB,GRIDTYP,ST,J,I,ALPHA)
     S=(4._q*ST-OST)/3._q
     IF (J>5) THEN                       !avoid spurious early convergence       
          IF (ABS(S-OS)<EPSI ) GOTO 10
     ENDIF
     OS=S
     OST=ST
   ENDDO 
 
  ELSEIF ((PRESENT(ALPHA)) .AND. (PRESENT(BETA)) ) THEN
     DO J=1,JMAX 

     CALL CHOOSE(F,A,B,WA,WB,GRIDTYP,ST,J,I,ALPHA,BETA)
     S=(4._q*ST-OST)/3._q
     IF (J>5) THEN                       !avoid spurious early convergence       
          IF (ABS(S-OS)<EPSI ) GOTO 10
     ENDIF
     OS=S
     OST=ST
   ENDDO 
  ENDIF
 

  WRITE(*,*) ' ERROR in SIMP_MOMENT: Too many steps',J
  STOP 

10 MOM(I)=S                             !moment calculated
!WRITE(*,*)I

!--------------------------------------------------------------
 ENDDO               !loop moments
!--------------------------------------------------------------
 
 DO I=1,N+1
  NU(I)=MOM(I-1)
 ENDDO

  DEALLOCATE(MOM)

  RETURN
 ENDSUBROUTINE QSIMP_MOMENT

!************************TRAPEZ_MOMENT**********************
! This routine calculates the integral of the function F*WEIGHTFUNCTION using the 
! trapezoidal rule. 
! M=1,2,3... improves the accuracy of S by addint 2^M-2 additional 
! interior points. S should not be modified between subsequent calls!
!***********************************************************************
 SUBROUTINE TRAPEZ_MOMENT(F,A,B,WAA,WBB,GRIDT,S,M,N&
           ,ALPHA,BETA &
            )
  USE prec
  IMPLICIT NONE
  INTEGER M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) A,B,S
  REAL(q), EXTERNAL :: F
  INTEGER  N                       !paramter of function F
  REAL(q),INTENT(IN), OPTIONAL :: ALPHA
  REAL(q),INTENT(IN), OPTIONAL :: BETA ! optional parameter for Jacobi  
  INTEGER IT,J
  REAL(q) DEL,S_ADD,TNM,X,G
  REAL(q) XMIN,XMAX

  XMIN=REAL(A)
  XMAX=REAL(B)

 IF ( .NOT. PRESENT(ALPHA) .AND. .NOT. PRESENT(BETA) ) THEN

  IF (M==1) THEN
   S=0.5_q*(B-A)*(F(A,N,XMIN,XMAX)*WEIGHTFUNCTION(A,GRIDT,WAA,WBB)+F(B,N,XMIN,XMAX)*WEIGHTFUNCTION(B,GRIDT,WAA,WBB))
  ELSE
   IT=2._q**(M-2)
   TNM=IT
   DEL=(B-A)/TNM       !the spacing of points to be added
   X=A+0.5_q*DEL
  
   S_ADD=0._q
   DO J=1,IT
    S_ADD=S_ADD+F(X,N,XMIN,XMAX)*WEIGHTFUNCTION(X,GRIDT,WAA,WBB)
    X=X+DEL
   ENDDO 
   S=0.5_q*(S+(B-A)*S_ADD/TNM) !S replaced by the new value
  ENDIF

 ELSEIF ( PRESENT(ALPHA) .AND. .NOT. PRESENT(BETA) ) THEN

  IF (M==1) THEN
   S=0.5_q*(B-A)*(F(A,N,XMIN,XMAX,ALPHA)*WEIGHTFUNCTION(A,GRIDT,WAA,WBB)+F(B,N,XMIN,XMAX,ALPHA)*WEIGHTFUNCTION(B,GRIDT,WAA,WBB))
  ELSE
   IT=2._q**(M-2)
   TNM=IT
   DEL=(B-A)/TNM       !the spacing of points to be added
   X=A+0.5_q*DEL
  
   S_ADD=0._q
   DO J=1,IT
    S_ADD=S_ADD+F(X,N,XMIN,XMAX,ALPHA)*WEIGHTFUNCTION(X,GRIDT,WAA,WBB)
    X=X+DEL
   ENDDO 
   S=0.5_q*(S+(B-A)*S_ADD/TNM) !S replaced by the new value
  ENDIF

ELSEIF ( PRESENT(ALPHA) .AND. PRESENT(BETA) ) THEN

  IF (M==1) THEN
   S=0.5_q*(B-A)*(F(A,N,XMIN,XMAX,ALPHA,BETA)*WEIGHTFUNCTION(A,GRIDT,WAA,WBB)+F(B,N,XMIN,XMAX,ALPHA,BETA)*WEIGHTFUNCTION(B,GRIDT,WAA,WBB))
  ELSE
   IT=2._q**(M-2)
   TNM=IT
   DEL=(B-A)/TNM       !the spacing of points to be added
   X=A+0.5_q*DEL
  
   S_ADD=0._q
   DO J=1,IT
    S_ADD=S_ADD+F(X,N,XMIN,XMAX,ALPHA,BETA)*WEIGHTFUNCTION(X,GRIDT,WAA,WBB)
    X=X+DEL
   ENDDO 
   S=0.5_q*(S+(B-A)*S_ADD/TNM) !S replaced by the new value
  ENDIF
ENDIF

  RETURN
 ENDSUBROUTINE TRAPEZ_MOMENT

!*********************** MIDPOINT_MOMENT *********************
! Similar to TRAPEZ_MOMENT. We need this as a workhorse for 
! the Romber integration on an open interval.
!*************************************************************************
SUBROUTINE MIDPOINT_MOMENT0(F,A,B,WAA,WBB,GRIDT,S,N,M)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) A,B,S
  REAL(q), EXTERNAL :: F
  INTEGER IT,J
  REAL(q) DDEL,DEL,SSUM,TNM,X,G

#define G(x) F(x,M,REAL(A),REAL(B))*WEIGHTFUNCTION(x,GRIDT,WAA,WBB) 
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDPOINT_MOMENT0

SUBROUTINE MIDPOINT_MOMENT1(F,A,B,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) A,B,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  INTEGER IT,J
  REAL(q) DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(x,M,REAL(A),REAL(B),ALPHA)*WEIGHTFUNCTION(x,GRIDT,WAA,WBB) 
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDPOINT_MOMENT1

SUBROUTINE MIDPOINT_MOMENT2(F,A,B,WAA,WBB,GRIDT,S,N,M,ALPHA,BETA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) A,B,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA, BETA   
  INTEGER IT,J
  REAL(q) DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(x,M,REAL(A),REAL(B),ALPHA,BETA)*WEIGHTFUNCTION(x,GRIDT,WAA,WBB) 
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDPOINT_MOMENT2

!***************MIDPOINT_MOMENT1_0INF*************************************
! this routine performs a variable transform x |-> 1/(x+1) so that 
! integrations on the interval (0,inf) is possible 
!*************************************************************************
SUBROUTINE MIDPOINT_MOMENT1_0INF(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  INTEGER IT,J
  REAL(q) DDEL,DEL,SSUM,TNM,X,G,A,B
  
#define G(x) F(1._q/x-1._q-1._q,M,REAL(AA),REAL(BB),ALPHA)*WEIGHTFUNCTION(1._q/x-1._q,GRIDT,WAA,WBB)/(x**2) 
 
 IF (ABS(AA)>1E-6 .OR. ABS(AA-BB)<1E-6 ) THEN
  WRITE(*,*)'Error in MIDPOINT_MOMENT1_0INF: AA/=0  or |AA-BB|=0',AA,BB
  STOP
 ENDIF

 A=REAL(1._q/(1._q + BB), q)
 B=REAL(1._q/(1._q + AA), q)

  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDPOINT_MOMENT1_0INF

!*********************** MIDSQRTL_MOMENT **********************************
! Similar to TRAPEZ_MOMENT. Use this for integrable 
! singularities at the lower integration limit.
!*************************************************************************
SUBROUTINE MIDSQRTL_MOMENT0(F,AA,BB,WAA,WBB,GRIDT,S,N,M)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(AA+x**2,M,REAL(AA),REAL(BB))*WEIGHTFUNCTION(AA+x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTL_MOMENT0

SUBROUTINE MIDSQRTL_MOMENT1(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(AA+x**2,M,REAL(AA),REAL(BB),ALPHA)*WEIGHTFUNCTION(AA+x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTL_MOMENT1

SUBROUTINE MIDSQRTL_MOMENT2(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA,BETA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA, BETA   
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(AA+x**2,M,REAL(AA),REAL(BB),ALPHA,BETA)*WEIGHTFUNCTION(AA+x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTL_MOMENT2

!*********************** MIDSQRTU_MOMENT **********************************
! Similar to TRAPEZ_MOMENT. Use this for integrable 
! singularities at the upper integration limit.
!*************************************************************************
SUBROUTINE MIDSQRTU_MOMENT0(F,AA,BB,WAA,WBB,GRIDT,S,N,M)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(BB-x**2,REAL(AA),REAL(BB),M)*WEIGHTFUNCTION(BB-x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTU_MOMENT0

SUBROUTINE MIDSQRTU_MOMENT1(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(BB-x**2,M,REAL(AA),REAL(BB),ALPHA)*WEIGHTFUNCTION(BB-x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTU_MOMENT1

SUBROUTINE MIDSQRTU_MOMENT2(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA,BETA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA, BETA   
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) 2._q*x*F(BB-x**2,M,REAL(AA),REAL(BB),ALPHA,BETA)*WEIGHTFUNCTION(BB-x**2,GRIDT,WAA,WBB) 

  B=SQRT(BB-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDSQRTU_MOMENT2

!*********************** MIDINF_MOMENT **********************************
! Similar to TRAPEZ_MOMENT. Use this for infinite integration
! intervals. where either A->inf, B>0 or A<0 and B->-inf 
!*************************************************************************
SUBROUTINE MIDINF_MOMENT0(F,AA,BB,WAA,WBB,GRIDT,S,N,M)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(1._q/x,M,REAL(AA),REAL(BB))*WEIGHTFUNCTION(1._q/x,GRIDT,WAA,WBB)/x**2

  !change the integration interval
  
  IF (ABS(AA)<1E-6.OR.ABS(BB)<1E-6.OR.ABS(AA-BB)<1E-6) THEN
   WRITE(*,'(A,3F20.10)')'Error in MIDINF_MOMENT0',AA,BB,ABS(AA-BB)
   STOP
  ENDIF
  

  B=1._q/AA
  A=1._q/BB

  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDINF_MOMENT0

SUBROUTINE MIDINF_MOMENT1(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(1._q/x,REAL(AA),REAL(BB),M,ALPHA)*WEIGHTFUNCTION(1._q/x,GRIDT,WAA,WBB)/x**2

  IF (ABS(AA)<1E-6.OR.ABS(BB)<1E-6.OR.ABS(AA-BB)<1E-6) THEN
   WRITE(*,'(A,3F20.10)')'Error in MIDINF_MOMENT1',AA,BB,ABS(AA-BB)
   STOP
  ENDIF

  !change the integration interval
  B=1._q/AA
  A=1._q/BB

 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDINF_MOMENT1

SUBROUTINE MIDINF_MOMENT2(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA,BETA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) :: ALPHA
  REAL(q) :: BETA   
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(1._q/x,M,REAL(AA),REAL(BB),ALPHA,BETA)*WEIGHTFUNCTION(1._q/x,GRIDT,WAA,WBB)/x**2

  IF (ABS(AA)<1E-6.OR.ABS(BB)<1E-6.OR.ABS(AA-BB)<1E-6) THEN
    WRITE(*,'(A,3F20.10)')'Error in MIDINF_MOMENT2',AA,BB,ABS(AA-BB)
    STOP
  ENDIF
 
  !change the integration interval
  B=1._q/AA
  A=1._q/BB
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDINF_MOMENT2

!*********************** MIDEXP_MOMENT **********************************
! Similar to TRAPEZ_MOMENT. Here the the following change of 
! integration variable is done x->e^-x
! This often works for exponentially damped integrands and infinite
! integration intervals.
!*************************************************************************
SUBROUTINE MIDEXP_MOMENT0(F,AA,BB,WAA,WBB,GRIDT,S,N,M)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) WAA,WBB
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(-LOG(x),M,REAL(AA),REAL(BB))*WEIGHTFUNCTION(-LOG(x),GRIDT,WAA,WBB)/x 

  B=EXP(-AA)
  A=0._q

  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDEXP_MOMENT0

SUBROUTINE MIDEXP_MOMENT1(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) ALPHA
  REAL(q) WAA,WBB
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(-LOG(x),M,REAL(AA),REAL(BB),ALPHA)*WEIGHTFUNCTION(-LOG(x),GRIDT,WAA,WBB)/x 

  B=EXP(-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDEXP_MOMENT1

SUBROUTINE MIDEXP_MOMENT2(F,AA,BB,WAA,WBB,GRIDT,S,N,M,ALPHA,BETA)
 USE prec
  INTEGER N,M,GRIDT
  REAL(q) AA,BB,S
  REAL(q), EXTERNAL :: F
  REAL(q) WAA,WBB
  REAL(q) ALPHA, BETA   
  INTEGER IT,J
  REAL(q) A,B,DDEL,DEL,SSUM,TNM,X,G
  
#define G(x) F(-LOG(x),M,REAL(AA),REAL(BB),ALPHA,BETA)*WEIGHTFUNCTION(-LOG(x),GRIDT,WAA,WBB)/x 

  B=EXP(-AA)
  A=0._q
 
  IF (N == 1) THEN
   S=(B-A)*G(0.5_q*(A+B))
  ELSE
   IT=3**(N-2)
   TNM=IT
   DEL=(B-A)/(3._q*TNM)
   DDEL=DEL+DEL 
   X=A+0.5_q*DEL
   SSUM=0._q
  
  DO J=1,IT
   SSUM=SSUM+G(X) 
   X=X+DDEL
   SSUM=SSUM+G(X) 
   X=X+DEL
  ENDDO 
   S=(S+(B-A)*SSUM/TNM)/3._q    
  ENDIF 

#undef G

  RETURN
 END SUBROUTINE MIDEXP_MOMENT2


!***********************QROMB_MOMENT*****************************
! Returns as S the general moment NU of the function F with respect to 
! the weight function W using the Romberg integration method.
! 
! nu(N)=\int_A^B F(x,N)W(x) dx
! 
! ESP can be set  to the desired fractional accuracy and JMAX so that 
! 2^(JMAX-1) is the maximum allowed number of steps. 
!  
! (is currently not used)
!***********************************************************************
 SUBROUTINE QROMB_MOMENT(F,AA,BB,WA,WB,GRIDT,NU,N,ALPHA,BETA)
  USE prec
  INTEGER N,GRIDT                            ! order of moment
  REAL(q) NU(N+1)
  REAL(q)  AA,BB
  REAL(q)  WA,WB   
  REAL(q),INTENT(IN), OPTIONAL :: ALPHA
  REAL(q),INTENT(IN), OPTIONAL :: BETA ! optional parameter
  REAL(q), EXTERNAL :: F
  INTEGER JMAX,JMAXP,K,KM   
  REAL(q) EPSI, SS                     !maximum number of steps
  PARAMETER (EPSI=1.E-9, JMAX=32, JMAXP=JMAX+1, K=3, KM=K-1)      !accuracy  
  INTEGER J,I,INFO
  REAL(q) DSS,H(JMAXP),S(JMAXP),A,B
  REAL(q), ALLOCATABLE :: MOM(:)

  ALLOCATE(MOM(0:N)); MOM=0._q
  IF (SIZE(MOM)/=SIZE(NU)) THEN
   WRITE(*,*)'Error in QROMB_MOMENT, MOM and NU sizes differ:',SIZE(MOM),SIZE(NU)
   STOP
  ENDIF
  
  
  H(1)=1._q
  A=REAL(AA,q)
  B=REAL(BB,q)
  INFO=-10 


!--------------------------------------------------------------
!loop over moments, start with zeroth moment
!\nu_0=ANU(1)
 DO I=0,N              
!--------------------------------------------------------------

   IF ( (.NOT. PRESENT(ALPHA) ) .AND. (.NOT. PRESENT( BETA)) ) THEN
      DO J=1,JMAX 
       CALL TRAPEZ_MOMENT(F,A,B,WA,WB,GRIDT,S(J),J,I)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=1._q/4*H(J)
      ENDDO 
    
   
    ELSEIF ((PRESENT(ALPHA)) .AND. (.NOT.PRESENT(BETA)) ) THEN
      DO J=1,JMAX 
       CALL TRAPEZ_MOMENT(F,A,B,WA,WB,GRIDT,S(J),J,I,ALPHA)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=0.25*H(J)
      ENDDO 
   
    ELSEIF ((PRESENT(ALPHA)) .AND. (PRESENT(BETA)) ) THEN
      DO J=1,JMAX 
       CALL TRAPEZ_MOMENT(F,A,B,WA,WB,GRIDT,S(J),J,I,ALPHA,BETA)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=0.25*H(J)
      ENDDO 
   ENDIF
  
   WRITE(*,*) ' ERROR in QROMB_MOMENT: Too many steps',J
  
10 IF (INFO==0) THEN 
    WRITE(*,*) ' Error in QROMB_MOMENT, integration not converged'
    WRITE(*,*) ' Possible reasons: Singularity, vanishing Moment, highly oscillatory integral ' 
    STOP 
   ELSE
    MOM(I)=SS                             !moment calculated
    WRITE(*,*)I,J,MOM(I)
   ENDIF

!------------------------------------------------------------------------- 
   ENDDO
!------------------------------------------------------------------------- 
 DO I=1,N+1
  NU(I)=MOM(I-1)
 ENDDO

  DEALLOCATE(MOM)

  RETURN
 ENDSUBROUTINE QROMB_MOMENT

!***********************QROMB_IMPROPER_MOMENT*****************************
! Almost same as QROMB_GENERAL_MOMENT but on an open interval. Uses
! the midpoint rather (in an appropriate version) rather than the
! trapezoidal rule.
!***********************************************************************
 SUBROUTINE QROMB_IMPROPER_MOMENT(F,A,B,WA,WB,GRIDT,NU,N,CHOOSE,ALPHA,BETA)
  USE prec
  INTEGER N,GRIDT                            ! order of moment
  REAL(q) A,B
  REAL(q) WA,WB
  REAL(q) NU(N+1)
  REAL(q),INTENT(IN), OPTIONAL :: ALPHA
  REAL(q),INTENT(IN), OPTIONAL :: BETA ! optional parameter
  REAL(q), EXTERNAL :: F
  EXTERNAL CHOOSE
  INTEGER JMAX,JMAXP,K,KM   
  REAL(q) EPSI, SS                     !maximum number of steps
  PARAMETER (EPSI=1.E-14, JMAX=25, JMAXP=JMAX+1, K=5, KM=K-1)      !accuracy  
  INTEGER J,I,INFO
  REAL(q) DSS,H(JMAXP),S(JMAXP)
  REAL(q), ALLOCATABLE :: MOM(:)

  ALLOCATE(MOM(0:N)); MOM=0._q
  IF (SIZE(MOM)/=SIZE(NU)) THEN
   WRITE(*,*)'Error in QROMB_I_MOMENT, MOM and NU sizes differ:',SIZE(MOM),SIZE(NU)
   STOP
  ENDIF
  
  H(1)=1._q
!--------------------------------------------------------------
!loop over moments, start with zeroth moment
!\nu_0=ANU(1)
 DO I=0,N              
!--------------------------------------------------------------
   IF ( (.NOT. PRESENT(ALPHA) ) .AND. (.NOT. PRESENT( BETA)) ) THEN
      DO J=1,JMAX 
         CALL CHOOSE(F,A,B,WA,WB,GRIDT,S(J),J,I-1)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=H(J)/9._q
      ENDDO 
    
   
    ELSEIF ((PRESENT(ALPHA)) .AND. (.NOT.PRESENT(BETA)) ) THEN
      DO J=1,JMAX 
         CALL CHOOSE(F,A,B,WA,WB,GRIDT,S(J),J,I-1,ALPHA)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=H(J)/9._q
      ENDDO 
   
    ELSEIF ((PRESENT(ALPHA)) .AND. (PRESENT(BETA)) ) THEN
      DO J=1,JMAX 
         CALL CHOOSE(F,A,B,WA,WB,GRIDT,S(J),J,I-1,ALPHA,BETA)
       IF (J>=K) THEN
         CALL POLYNOMIAL_INTERPOLATION(H(J-KM),S(J-KM),K,0._q,SS,DSS,INFO)
            IF (ABS(DSS)< EPSI*ABS(SS) ) GOTO 10
       ENDIF
       S(J+1)=S(J)
       H(J+1)=H(J)/9._q
      ENDDO 
   ENDIF
  
   WRITE(*,*) ' ERROR in QROMB_IMP_MOMENT: Too many steps',J
  
10 IF (INFO==0) THEN 
    WRITE(*,*) ' Error in QROMB_IMP_MOMENT, integration not converged'
    WRITE(*,*) ' Possible reasons: Singularity, vanishing Moment, highly oscillatory integral ' 
    STOP 
   ELSE
    MOM(I)=SS                             !moment calculated
    WRITE(*,*)I,J,MOM(I)
   ENDIF

!------------------------------------------------------------------------- 
   ENDDO
!------------------------------------------------------------------------- 
 DO I=1,N+1
  NU(I)=MOM(I-1)
 ENDDO

  DEALLOCATE(MOM)

  RETURN
 ENDSUBROUTINE QROMB_IMPROPER_MOMENT

!****************** POLYNOMIAL_INTERPOLATION ***************************
! 
! Given arrays xa and ya, each of length n, and given a value x, 
! this routine returns a value y, and an error estimate dy. 
! If P(x) is the polynomial of degree N − 1 such that
! P(xai) = yai; i =1,....,n,  then the returned value y = P(x).
! (is currently not used)
!
!***********************************************************************
SUBROUTINE POLYNOMIAL_INTERPOLATION(XA,YA,N,X,Y,DY,INF)
 USE prec
 INTEGER N,NMAX
 REAL(q) DY,X,Y,XA(N),YA(N)
 PARAMETER (NMAX=10)  ! largest polynomial wanted
 INTEGER I,M,NS,INF
 REAL(q) DEN,DIF,DIFT,HO,HP,W,C(NMAX),D(NMAX)
 !initialize 
 INF=-10

 NS=1
 DIF=ABS(X-XA(1))
 
  DO I=1,N                    
    DIFT=ABS(X-XA(I))
      IF (DIFT < DIF) THEN
         NS=I
         DIF=DIFT
      ENDIF
    C(I)=YA(I) 
    D(I)=YA(I)
  ENDDO

  Y=YA(NS)           
  NS=NS-1
 
  DO  M=1,N-1 
   DO  I=1,N-M 
     HO=XA(I)-X
     HP=XA(I+M)-X
     W=C(I+1)-D(I)
     DEN=HO-HP
    
     IF(DEN<1E-9) THEN
      WRITE(*,*) ' FAILURE IN POLYNOMIAL_INTERPOLATION :'
      WRITE(*,*) ' two or more points are too close to each other!'
      INF=0
     GOTO 10
     ENDIF
      
     DEN=W/DEN
     D(I)=HP*DEN 
     C(I)=HO*DEN
   ENDDO 
   
   IF (2*NS <  N-M) THEN 
     DY=C(NS+1)
   ELSE
    DY=D(NS)
    NS=NS-1
   ENDIF
  
   Y=Y+DY
  ENDDO 

10  CONTINUE

 RETURN
 END SUBROUTINE POLYNOMIAL_INTERPOLATION


!=======================================================================
!
!
!               SPECIAL POLYNOMIALS AND OTHER FUNCTIONS
!
!
!=======================================================================

!-----------------------------------------------------------------------
! Returns the Legendre Polynomial  PLEG_n(x) in the interval [-1,1]
!-----------------------------------------------------------------------
 FUNCTION PLEG(X,N)
  IMPLICIT NONE
  REAL(q) PLEG
  REAL(q) X
  REAL(q) PLEGN(0:N)
  INTEGER N, K

  PLEGN(0) = 1.0_q
  PLEGN(1) = X

  IF (N <= 1) THEN
    PLEG = PLEGN(N)
  ELSE
   DO K=1,N-1
    PLEGN(K+1) = ((2.0_q*K+1.0_q)*X*PLEGN(K) - REAL(K)*PLEGN(K-1))/(REAL(K+1))
   ENDDO
   PLEG = PLEGN(N)
  ENDIF

 RETURN
 ENDFUNCTION PLEG
!-----------------------------------------------------------------------
! Returns the shifted monic Legendre Polynomial  in the interval [-XMIN,XMAX]
!-----------------------------------------------------------------------
 FUNCTION PLEGMON(X,N,XMIN,XMAX)
  USE prec
  IMPLICIT NONE
  REAL(q) PLEGMON
  REAL(q) X,PLN(-1:N)
  INTEGER N,K
  REAL(q) XMIN,XMAX
  REAL(q) X0,S
 
 IF ( ABS( XMAX-XMIN )<1E-6 ) THEN
   WRITE( *,'(/A,2F10.6)') "ERROR in PLEGMON: XMIN and XMAX too close!",XMIN,XMAX
   WRITE( *,*) " "
   STOP
 ENDIF

 X0=(XMIN+XMAX)/2._q
  S= 2._q/(XMAX-XMIN) 

  PLN(-1)=0._q
  PLN(0) = 1._q

  IF (N <= 0) THEN
    PLEGMON = PLN(N)
  ELSE
   DO K=0,N-1
    PLN(K+1) =( X - X0 )*PLN(K)- REAL(K*K, q )/(REAL(4*K*K - 1, q )*(S**2) )*PLN(K-1)
   ENDDO
   PLEGMON = PLN(N)
  ENDIF
 
 RETURN
 ENDFUNCTION PLEGMON
!-----------------------------------------------------------------------
! Returns the derivative of the Legendre Polynomial  PLEG_n(x) 
! for X=1 PLEGDER is set to zero!
!-----------------------------------------------------------------------
 FUNCTION PLEGDER(X,N)
  IMPLICIT NONE
  REAL(q) PLEGDER
  REAL(q) X
  INTEGER N

  IF(ABS(X-1.0_q)>1E-15) THEN
    PLEGDER=((-1.0_q-N)*X*PLEG(X,N)+(1.0_q+N)*PLEG(X,N+1))/(X*X-1.0_q)
  ELSE
   PLEGDER=0.0_q
  ENDIF

 RETURN
 ENDFUNCTION PLEGDER

!-----------------------------------------------------------------------
! Returns the Laguerre polynomial  PLAG_n(x) 
!-----------------------------------------------------------------------
 FUNCTION PLAG(X,N,XMIN,XMAX,ALPHA)
  USE prec
   IMPLICIT NONE
   REAL(q) PLAG
   REAL(q) X
   REAL(q) PLN(0:N)
   REAL(q),INTENT(IN),OPTIONAL :: ALPHA
   REAL(q) XMIN,XMAX
   INTEGER N, K

  PLN(0) = 1._q
  PLN(1) = 1._q-X

  IF (N <= 1) THEN
    PLAG = PLN(N)
  ELSE
  IF (PRESENT(ALPHA) ) THEN
   DO K=1,N-1
    PLN(K+1) = ((2._q*K+1._q + ALPHA -X)*PLN(K) - REAL(K+ALPHA,q)*PLN(K-1))/(REAL(K+1,q))
   ENDDO
  ELSE
   DO K=1,N-1
    PLN(K+1) = ((2._q*K+1._q -X)*PLN(K) - REAL(K,q)*PLN(K-1))/(REAL(K+1,q))
   ENDDO
  ENDIF
   PLAG = PLN(N)
  ENDIF


 RETURN
 ENDFUNCTION PLAG
!-----------------------------------------------------------------------
! Returns the monic general Laguerre polynomial in the interval [0,inf) 
!-----------------------------------------------------------------------
 FUNCTION PLAGMON(X,N,XMIN,XMAX,ALPHA)
  USE prec
  IMPLICIT NONE
  REAL(q) PLAGMON
  REAL(q) X
  REAL(q) PLN(-1:N)
  REAL(q) XMIN,XMAX
  REAL(q), INTENT(IN), OPTIONAL :: ALPHA
  INTEGER N, K

  PLN(-1) = 0._q
  PLN(0) = 1._q

  IF (N <= 0) THEN
    PLAGMON = PLN(N)
  ELSE
 IF(PRESENT(ALPHA))THEN
   DO K=0,N-1
    PLN(K+1) = ( X-REAL( 2*K+1+ALPHA, q) )*PLN(K) - REAL(K*(K+ALPHA),q)*PLN(K-1)
   ENDDO
 ELSE
   DO K=0,N-1
    PLN(K+1) = ( X-REAL( 2*K+1, q) )*PLN(K) - REAL(K*K,q)*PLN(K-1)
   ENDDO
 ENDIF

   PLAGMON = PLN(N)
  ENDIF

 RETURN
 ENDFUNCTION PLAGMON

!-----------------------------------------------------------------------
! Returns the Hermite Polynomial (physicist convention)
!-----------------------------------------------------------------------
FUNCTION PHERM(X,N,XMIN,XMAX)
  USE prec
  IMPLICIT NONE
  REAL(q) PHERM
  REAL(q) X
  REAL(q) PHN(-1:N)
  REAL(q) XMIN,XMAX
  INTEGER N, K

  IF (N<-1) THEN
   WRITE(*,*)'Hermite polynomials not defined for negative order'
   STOP
  ENDIF
 ! set H_-1=0 

  PHN(-1) = 0.0_q
  PHN(0)  = 0.751125544464942482860_q

  IF (N <= 1) THEN
    PHERM = PHN(N)
  ELSE
   DO K=1,N
    PHN(K) = X*SQRT(2.0_q/K)*PHN(K-1) - SQRT(DBLE(K-1)/DBLE(K))*PHN(K-2)
   ENDDO
   PHERM = PHN(N)
  ENDIF

 RETURN
 ENDFUNCTION PHERM
!-----------------------------------------------------------------------
! Returns the monic Hermite Polynomial in the interval [XMIN,XMAX]
!-----------------------------------------------------------------------
FUNCTION PHERMMON(X,N,XMIN,XMAX)
  USE prec
  IMPLICIT NONE
  REAL(q) PHERMMON
  REAL(q) X
  REAL(q) PHN(-1:N),X0,S
  REAL(q) XMIN,XMAX
  INTEGER N, K

  IF (N<-1) THEN
   WRITE(*,*)'Hermite polynomials not defined for negative order'
   STOP
  ENDIF

  IF ( ABS(XMIN-XMAX)<1E-6 ) THEN
   WRITE(*,*)'Error PHERMMON: XMIN and XMAX too close!',XMIN,XMAX
   STOP
  ENDIF

  X0=REAL( (XMIN+XMAX)/2., q )
   S=REAL( 2._q/(XMAX-XMIN), q )
 
  PHN(-1) = 0.0_q
  PHN(0)  = 1.0_q

  IF (N <= 0) THEN
    PHERMMON = PHN(N)
  ELSE
   DO K=0,N-1
    PHN(K+1) = ( X - X0 )*PHN(K) - REAL( K/(2*S**2) ,q)*PHN(K-1)
   ENDDO
   PHERMMON = PHN(N)
  ENDIF

 RETURN
 ENDFUNCTION PHERMMON

!-----------------------------------------------------------------------
! Returns the Chebgyshev polynomial of first kind PT_n(x) 
!-----------------------------------------------------------------------
 FUNCTION PT(X,N,XMIN,XMAX)
  IMPLICIT NONE
  REAL(q) PT
  REAL(q) X
  REAL(q) PTN(0:N)
  REAL(q) XMIN,XMAX
  INTEGER N, K

  PTN(0) = 1.0_q
  PTN(1) = 1.0_q*X

  IF (N <= 1) THEN
    PT = PTN(N)
  ELSE
   DO K=1,N-1
    PTN(K+1) = (2.0_q*X)*PTN(K) - PTN(K-1)
   ENDDO
   PT = PTN(N)
  ENDIF

 RETURN
 ENDFUNCTION PT
!-----------------------------------------------------------------------
! Returns the monic Chebyshev polynomial of first kind PTMON_n(x) in [XMIN,XMAX]
!-----------------------------------------------------------------------
 FUNCTION PTMON(X,N,XMIN,XMAX)
  USE prec
   IMPLICIT NONE
   REAL(q) PTMON
   REAL(q) X
   REAL(q) PLN(-1:N)
   REAL(q) XMIN,XMAX
   REAL(q) X0,S
   INTEGER N, K

   !shift
  IF ( ABS(XMIN-XMAX)<1E-6 ) THEN
    WRITE(*,*)'Error in PTMON: XMIN,XMAX too close!',XMIN,XMAX
    STOP
  ENDIF

    X0=REAL( (XMAX+XMIN)/2., q )
     S=REAL( 2./(XMAX-XMIN), q )
 
    PLN(-1) = 0._q
    PLN(0) = 1._q
  
    IF (N <= 0) THEN
      PTMON = PLN(N)
    ELSE
     DO K=0,N-1
     
      IF (K<=1)  PLN(K+1) = (X-X0)*PLN(K) - REAL( 1./(2*S**2), q)*PLN(K-1)
      IF (K>1 )  PLN(K+1) = (X-X0)*PLN(K) - REAL( 1./(4*S**2), q)*PLN(K-1)
     
     ENDDO
     PTMON = PLN(N)
    ENDIF
  

 RETURN
 ENDFUNCTION PTMON
!-----------------------------------------------------------------------
! Returns the monic Chebyshev polynomial of second kind PUMON_n(x) in [XMIN,XMAX]
!-----------------------------------------------------------------------
 FUNCTION PUMON(X,N,XMIN,XMAX)
  USE prec
   IMPLICIT NONE
   REAL(q) PUMON
   REAL(q) X
   REAL(q) PLN(-1:N)
   REAL(q) XMIN,XMAX
   REAL(q) X0
   INTEGER N, K

   !shift
   IF ( ABS(XMIN-XMAX)<1E-6 ) THEN
     WRITE(*,*)'Error in PUMON: XMIN,XMAX too close!',XMIN,XMAX
     STOP
   ENDIF

    X0=REAL( (XMAX+XMIN)/2., q )
 
    PLN(-1) = 0._q
    PLN(0) = 1._q
  
    IF (N <= 0) THEN
      PUMON = PLN(N)
    ELSE
     DO K=0,N-1
     
        PLN(K+1) = ( X - X0 )*PLN(K) - ( 1._q/4._q )*PLN(K-1)
     
     ENDDO
     PUMON = PLN(N)
    ENDIF
  

 RETURN
 ENDFUNCTION PUMON

!-----------------------------------------------------------------------
! Just a test function with one real valued parameter ALPHA 
! (PT**ALPHA)
!-----------------------------------------------------------------------
 FUNCTION PTEST(X,N,XMIN,XMAX,ALPHA)
  IMPLICIT NONE
  REAL(q) PTEST
  REAL(q) X
  REAL(q) PTN(0:N)
  INTEGER N, K
  REAL(q) ALPHA,XMIN,XMAX

  PTN(0) = 1.0_q
  PTN(1) = 1.0_q*X

  IF (N <= 1) THEN
    PTEST = PTN(N)**ALPHA
  ELSE
   DO K=1,N-1
    PTN(K+1) = (2.0_q*X)*PTN(K) - PTN(K-1)
   ENDDO
   PTEST = PTN(N)**ALPHA
  ENDIF

 RETURN
 ENDFUNCTION PTEST

!-----------------------------------------------------------------------
! Returns the Jacobi polynomial PJACOBIACOBI_n(x) 
!-----------------------------------------------------------------------
 FUNCTION PJACOBI(X,N,XMIN,XMAX,ALPHA,BETA)
  USE prec
  IMPLICIT NONE
  REAL(q) PJACOBI
  REAL(q) X
  REAL(q) TEMP,AB,A,B,C,P1,P2,P3
  INTEGER N, J
  REAL(q) ALPHA, BETA,XMIN,XMAX
 
  AB= REAL(ALPHA+BETA,q)
  TEMP=2._q+AB
  P1=(ALPHA-BETA+TEMP*X)/2._q
  P2=1._q
  
  DO J=2,N
   P3=P2
   P2=P1
   TEMP=2*J+AB
   A=2*J*(J+AB)*(TEMP-2._q)
   B=(TEMP-1._q)*(ALPHA*ALPHA-BETA*BETA+TEMP*(TEMP-2._q)*X)
   C=2._q*(J-1+ALPHA)*(J-1+BETA)*TEMP
   P1=(B*P2-C*P3)/A
  ENDDO
  
  PJACOBI=P1
 
 RETURN
 ENDFUNCTION PJACOBI
!-----------------------------------------------------------------------
! Returns the derivative of the  Jacobi polynomial PJACOBIACOBI_n(x) 
!-----------------------------------------------------------------------
 FUNCTION PJACOBIDER(X,N,XMIN,XMAX,ALPHA,BETA)
  USE prec
  IMPLICIT NONE
  REAL(q) X
  REAL(q) PJACOBIDER
  REAL(q) TEMP,AB,A,B,C,P1,P2,P3
  INTEGER N, J
  REAL(q) ALPHA, BETA, XMIN,XMAX

  AB=REAL( ALPHA+BETA, q)
  TEMP=2._q+AB
  P1=(ALPHA-BETA+TEMP*X)/2._q
  P2=1._q
  
  DO J=2,N
   P3=P2
   P2=P1
   TEMP=2*J+AB
   A=2*J*(J+AB)*(TEMP-2._q)
   B=(TEMP-1._q)*(ALPHA*ALPHA-BETA*BETA+TEMP*(TEMP-2._q)*X)
   C=2._q*(J-1+ALPHA)*(J-1+BETA)*TEMP
   P1=(B*P2-C*P3)/A
  ENDDO
  
  PJACOBIDER=(N*(ALPHA-BETA-TEMP*X)*P1+2._q*(N+ALPHA)*&
      (N+BETA)*P2)/(TEMP*(1._q-X*X))
 
 RETURN
 ENDFUNCTION PJACOBIDER

!-----------------------------------------------------------------------
!Returns the Logarithm of the Gamma function for X>0
!-----------------------------------------------------------------------
FUNCTION GAMMLN(XX)
 USE prec
 IMPLICIT NONE
 REAL(q) GAMMLN,XX
 INTEGER J
 REAL(q) SER,STP,TMP,X,Y,COF(6)
 SAVE COF,STP
 DATA COF,STP/76.18009172947146_q,    -86.50532032941677_q,&
              24.01409824083091_q,     -1.231739572450155_q,&
                .1208650973866179E-2_q, -.5395239384953E-5_q,&
               2.5066282746310005_q/
  X=XX
  Y=X
  TMP=X+5.5_q
  TMP=(X+0.5_q)*LOG(TMP)-TMP
  SER=1.000000000190015_q

 DO J=1,6
  Y=Y+1._q
  SER=SER+COF(J)/Y
 ENDDO 

  GAMMLN=TMP+LOG(STP*SER/X)

  RETURN
END FUNCTION GAMMLN
!-----------------------------------------------------------------------
!Returns the Nth Pochammer Symbol of X 
!-----------------------------------------------------------------------
FUNCTION POCHHAMMER(X,N)
 USE prec
  IMPLICIT NONE
  REAL(q) X,POCHHAMMER,TMP
  INTEGER N,I,J
  
  IF (N==0) THEN
    POCHHAMMER=X
  ELSEIF (ABS(N)<=100) THEN
    TMP=X 
    DO I=1,N-1 
      TMP=TMP*(X+I)
    ENDDO
    POCHHAMMER=TMP
  ENDIF
 
 RETURN
END FUNCTION POCHHAMMER
  

END MODULE gauss_quad
