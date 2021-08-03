#include "symbol.inc"

!***********************************************************************
! RCS:  $Id: dynbr.F,v 1.3 2002/04/30 15:36:32 kresse Exp $
!
!  calling interface to BRZERO
!  this routine extracts all coordinates/cell shape parameters and
!  forces/stresses and calls the Broyden update for the structure
!
!  the present routine is somewhat illconditioned in the following
!  sense:
!  If the lattice vectors have widely different length,
!  forces parallel to short lattice vectors are weighted
!  more strongly in the minimization routine than forces parallel
!  to short lattice vectors 
!  this problem could be solved by supplying a metric to the
!  optimization routine
!
!***********************************************************************
      MODULE brions_inproduct
        USE prec
        IMPLICIT NONE
        REAL(q), ALLOCATABLE, SAVE :: metric(:,:,:)
        LOGICAL :: ini_inproduct=.FALSE.
        CONTAINS

          FUNCTION INPRODUCT(V1,V2,N)
            INTEGER N
            REAL(q) INPRODUCT,V1(3,N),V2(3,N)
            INTEGER NIONS,I,J,K
            NIONS=N/3

            INPRODUCT=0
            DO I=1,NIONS
               DO J=1,3
               DO K=1,3
                  INPRODUCT=INPRODUCT+ &
                       V1(J,I)*metric(J,K,I)*V2(K,I)
               ENDDO
               ENDDO
            ENDDO
          
          END FUNCTION INPRODUCT

        END MODULE brions_inproduct

      SUBROUTINE BRIONS(NIONS,POS,PC,FOR,A,B,SIF,MAXIT,NFREE, &
     &                  IU6,IU0,FACT,FACTSI,DESTEP)
      USE prec
      USE ini
      USE lattice
      USE chain
      USE brions_inproduct
      IMPLICIT REAL(q) (A-H,O-Z)

      PARAMETER(OS=1.5_q)
      REAL(q)   POS(3,NIONS),PC(3,NIONS),FOR(3,NIONS)
      REAL(q)   A(3,3),B(3,3),SIF(3,3),MET(3,3)
! local work arrays
      REAL(q)   TMP(3,3)
      REAL(q)   X(3*NIONS+9)
      REAL(q)   F(3*NIONS+9)
      REAL(q)   G0(3*NIONS+9)
      REAL(q) :: EMAX_ALLOWED

      INTEGER, SAVE :: ITER=0
      REAL(q), SAVE :: STEP=1.0

      IF (.NOT.ini_inproduct) THEN
         ini_inproduct=.TRUE.
         ALLOCATE(metric(3,3,NIONS+3))
         metric=0

         MET=MATMUL(TRANSPOSE(A),A)

         DO NI=1,NIONS
            metric(:,:,NI)=MET
! to yield the old behaviour 
! use the following lines
!            metric(:,:,NI)=0
!            DO I=1,3
!               metric(I,I,NI)=1
!            ENDDO
         ENDDO
         

         DO NI=NIONS+1,NIONS+3
            DO I=1,3
               metric(I,I,NI)=1
            ENDDO
         ENDDO
      ENDIF

      ITER=ITER+1

! first order energy change (using a step width STEP):
      DESTEP=0._q
      IF (FACT/=0) THEN
      DO 20 NI=1,NIONS
         DESTEP=DESTEP - 1._q/FACT* &
              (FOR(1,NI)*FOR(1,NI)+FOR(2,NI)*FOR(2,NI)+FOR(3,NI)*FOR(3,NI))
   20 CONTINUE
      ENDIF
      GNORM1=-DESTEP*STEP
      CALLMPI_C( sum_chain( GNORM1 ))

      IF (FACTSI/=0) THEN
      DO I=1,3
         DO J=1,3
           DESTEP=DESTEP - 1/FACTSI * SIF(I,J)*SIF(I,J)
         ENDDO
      ENDDO
      ENDIF
      DESTEP=DESTEP*STEP
      CALLMPI_C( sum_chain( DESTEP ))

      GNORM2=-DESTEP-GNORM1

      IF (IU6>=0) WRITE( IU6,2) GNORM1,GNORM2
   2  FORMAT(' Quasi-Newton relaxation of ions (Broydens 2nd method)'/ &
     &       '  g(Force)  =',E10.3,'   g(Stress)=',E10.3,/)

      IF (IU0>=0) &
         WRITE( IU0,1,ADVANCE='NO') GNORM1,GNORM2
   1  FORMAT(' BRION: g(F)= ',E10.3,' g(S)= ',E10.3)

! copy all necessary data:
      PC=FOR

      CALL KARDIR(NIONS,PC,B)
      DO K=1,3
      DO I=1,NIONS
! coordinates:
         X(3*I+K-3)=POS(K,I)
! forces:
         F(3*I+K-3)= PC(K,I)
! save old coordinates
         PC(K,I)=POS(K,I)
      ENDDO
      ENDDO

      TMP=0

      DO K=1,3
      DO I=1,3
      DO J=1,3
         TMP(I,K)=TMP(I,K)+SIF(I,J)*A(J,K)
      ENDDO
      ENDDO
      ENDDO

      DO K=1,3
      DO I=1,3
! lattice vectors
         X(3*NIONS+3*I+K-3)=  A(K,I)
! stresses
         F(3*NIONS+3*I+K-3)=TMP(K,I)
      ENDDO
      ENDDO

! initial step:
      G0    =STEP
      NDIM=3*NIONS+9
      
! call Broyden - and hope to have good luck ... :
      MAXIT_=MAXIT             ! usually MAXIT_ = NSW+1
      IF (NFREE>0 ) THEN 
         MAXIT_=NFREE+1        ! however if NFREE is set MAXIT_= NFREE+1
         EMAX_ALLOWED=1000
      ELSE
         EMAX_ALLOWED=8
      ENDIF
      CALL BRZERO(NDIM,X,F,G0,INPRODUCT,MAXIT_,EMAX_ALLOWED,(ITER==1),IU6,IU0)

! update the positions
      DO  K=1,3
      DO I=1,NIONS
         POS(K,I)=X(3*I+K-3)
      ENDDO
      ENDDO
! update lattice vectors
      DO K=1,3
      DO I=1,3
         A(K,I)=X(3*NIONS+3*I+K-3)
      ENDDO
      ENDDO
      RETURN
      END


!***********************************************************************
!
! Find the zero of a function vector F(X) of a vector X using the
! second form of the modified Broyden scheme of Vanderbilt and Louie
! as "described" in D. Johnson s paper (PRB 38, 12807 [Dec. 1988]).
! (do not take this comment too serious, Johnsons paper contains at
!  least 6 mistakes, which we have corrected ...)
!
! details can be found in
! G. Kresse(gK), J. Furthmueller(jF),  
!         Comput. Mat. Sci. 6, 15-50 (1996)
!
! the very first version of the routine was 
! written by  jF with contributions of gK [small corrections only]
! rewritten to use F90, much simpler now, in addition automatic
! removal of iteration history added  (gK)
!
!***********************************************************************

      SUBROUTINE BRZERO(NDIM,X,F,G0,INPRODUCT,MAXIT,EMAX_ALLOWED,INI,IU6,IU0)
      USE prec
      USE chain

      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL,INTENT(IN) :: INI            ! reset routine
      REAL(q),INTENT(IN) :: F(NDIM)        ! gradient
      REAL(q),INTENT(INOUT) ::X(NDIM)      ! on entry: current position
                                           ! on exit:  updated position
      REAL(q),INTENT(IN) :: G0(NDIM)       ! initial approximation for Hessian
      INTEGER,INTENT(IN) :: MAXIT          ! maximum number of steps stored in iteration history
      REAL(q),INTENT(IN) :: EMAX_ALLOWED   ! maximum eigenvalue of hessian matrix
      REAL(q),EXTERNAL :: INPRODUCT
!
! value to customize the routine
! the problem with Pulays approach is that linear dependencies always
! develop in low dimensional problems,
! resulting in very large eigenvalues in the approximation
! of the inverse Hessian matrix (G). If the force points into the direction
! of this component a step  eigenvalue*force will be done,
! screwing up convergence. 
! Currently there are two solutions to this problem implemented:
! ) the number of degrees of freedom can be supplied by the calling routine
!   in the argument MAXIT
! ) EMAX_ALLOWED determines the maximum allowed
!   eigenvalue in the Hessian matrix.
!   information in the iteration history is removed until all eigenvalues
!   drop below this threshold
!   this is usually a reasonable save option, however it does not work
!   always
! local static work arrays
      REAL(q),SAVE,ALLOCATABLE :: GMAT(:,:),SMAT(:,:)
      REAL(q),SAVE,ALLOCATABLE :: FINF(:,:),WI(:)
      REAL(q),ALLOCATABLE,SAVE :: DF_(:,:),Z_(:,:),U_(:,:),T_(:,:)
      REAL(q),ALLOCATABLE,SAVE :: X_LAST(:),F_LAST(:)
! work arrays
      REAL(q) WRK1(NDIM)
      REAL(q) AMAT(MAXIT,MAXIT),BETA(MAXIT,MAXIT),BETAQ(MAXIT,MAXIT)
      REAL(q) AUX(MAXIT,MAX(8,MAXIT)),VV(MAXIT)
      REAL(q) AUXR(MAXIT),AUXI(MAXIT),AUXBET(MAXIT)
      COMPLEX(q) AUXC(MAXIT)
      REAL(q) GP(MAXIT,MAXIT),SP(MAXIT,MAXIT)
      REAL(q) EIGENVAL(MAXIT)

      INTEGER :: INDEX(MAXIT)
      INTEGER,SAVE ::  ITER=0

      IF (INI .AND.  ALLOCATED(DF_)) THEN
         DEALLOCATE( DF_,Z_,U_,T_,X_LAST,F_LAST,GMAT,SMAT,FINF,WI)
         ITER=0
      ENDIF
!=======================================================================
! first call initialize required work arrays
!=======================================================================
      IF (ITER==0) THEN
         ALLOCATE( DF_(NDIM,MAXIT),Z_(NDIM,MAXIT),U_(NDIM,MAXIT), &
                   T_(NDIM,MAXIT),X_LAST(NDIM),F_LAST(NDIM), &
                   GMAT(MAXIT,MAXIT),SMAT(MAXIT,MAXIT), &
                   FINF(MAXIT,MAXIT),WI(MAXIT))
      ENDIF
! increment iteration counter:
      ITER=ITER+1
! number of previous iteration:
      ITERM1=ITER-1
!=======================================================================
! First iteration is a conventional step along |F> using G^(1):
!=======================================================================
      IF (ITER==1) THEN
 2       FORMAT(' Reset! Starting new Quasi-Newton update for ions ')
         IF (IU6>=0) WRITE(IU6,2)
         ! First save some things needed later: |XLAST>, |F^(1)> and |FP^(1)>:
         X_LAST =X
         F_LAST =F

         ! trial step into direction G0* F
         X=X+G0*F
         IF (IU0>=0) WRITE(IU0,*)
         RETURN
      END IF
!=======================================================================
! BROYDEN updating for ITER > 1 ... :
!=======================================================================
! |DF^(iter-1)> = |F^(iter)>-|F^(iter-1)>:
      DF_(:,ITERM1) =F -F_LAST
! Save current |F>
      F_LAST=F
! inverse norm DF^(iter-1)
      SUM_=INPRODUCT(DF_(:,ITERM1),DF_(:,ITERM1),NDIM)
      CALLMPI_C( sum_chain( SUM_ ))

      FNORM=1._q/SQRT(SUM_)
! renormalize
      DF_(:,ITERM1)=DF_(:,ITERM1)*FNORM

! Vector |U^(iter-1)> = G^(1) |Delta F^(iter-1)> + |Delta X^(iter-1)>:
      U_(:,ITERM1)=(X-X_LAST)*FNORM+G0*DF_(:,ITERM1)
! save current X for next interation
      X_LAST=X
! weight all iterations with same weight, i.e. use Pulays approach
      WEIGHT=10000
      WI=WEIGHT
! or use Broydens second method as described by Stefan Bluegel
!      WI=0
!      WI(ITERM1)=WEIGHT

!=======================================================================
! Matrix FINF: we must only add the elements (iter-1,j) and (j,iter-1):
!   FINF(j,iter-1) = <D F(j) | metric |  D F(iter-1) >
!=======================================================================
      DO J=1,ITERM1
         SUM_=INPRODUCT(DF_(:,J),DF_(:,ITERM1),NDIM)
         CALLMPI_C( sum_chain( SUM_ ))

         FINF(J,ITERM1)=SUM_
         FINF(ITERM1,J)=SUM_

        !  SUM1=<D F(j)| G(1) |D F(iter-1)>
        !  SUM2=<D F(j)| U^(iter-1)> = <D F(j)| G(m)-G(1) | D F^(iter-1)>
        !  last equation holds only for Pulays method
         SUM1=SUM(DF_(:,J)*G0*DF_(:,ITERM1))
         SUM2=SUM(DF_(:,J)*U_(:,ITERM1))

         CALLMPI_C( sum_chain( SUM1 ))
         CALLMPI_C( sum_chain( SUM2 ))

         SMAT(J,ITERM1)=SUM1
         SMAT(ITERM1,J)=SUM1
         GMAT(J,ITERM1)=SUM2
      ENDDO

      DO J=1,ITERM1
         ! SUM=<D F(iter-1)| U^(j)>
         SUM_=SUM(DF_(:,ITERM1)*U_(:,J))
         CALLMPI_C( sum_chain( SUM_ ))

         GMAT(ITERM1,J)=-SUM_
      ENDDO

      FINF(ITERM1,ITERM1)=1._q

      ! correct transposed elements
      DO J=1,ITERM1-1
        FINF(ITERM1,J)=FINF(J,ITERM1)
        SMAT(ITERM1,J)=SMAT(J,ITERM1)
      ENDDO
!=======================================================================
!  solve eigenvalue problem
!   G e = lambda G^1 e
!  where G is the approximation of the Hessian matrix
!  works only for Pulay mixing because we assume  G |d F> = | d X>
!=======================================================================
      NORDER=0
      DO ISHIFT=0,ITERM1-1

         NORDER=ITERM1-ISHIFT
         GP=GMAT
         SP=SMAT
         CALL BR_SHIFT_MATRIX(SP,MAXIT,NORDER,ISHIFT)
         CALL BR_SHIFT_MATRIX(GP,MAXIT,NORDER,ISHIFT)

         IDUMP=0
         IF (IDUMP==1) THEN
            WRITE(*,*)'SMAT'
            DO  I=1,NORDER
               WRITE(*,'(10F10.4)') (SP(I,J),J=1,NORDER)
            ENDDO
            WRITE(*,*)'GMAT'
            DO I=1,NORDER
               WRITE(*,'(10F10.4)') (GP(I,J),J=1,NORDER)
            ENDDO
         ENDIF
#ifdef essl
         CALL DGEGV(0,GP,MAXIT,SP,MAXIT,AUXC,AUXBET,AUX,1,NORDER, &
              AUX,MAXIT*MAXIT)
#else
         CALL DGEGV('N','N',NORDER,GP,MAXIT,SP,MAXIT,AUXR,AUXI,AUXBET, &
              AUX,1,AUX,1,AUX,MAXIT*MAX(8,MAXIT),INFO)
         AUXC= CMPLX( AUXR , AUXI ,KIND=q)
#endif
         IF (IDUMP==1) WRITE(*,'(10F8.3)') AUXC(1:NORDER)
         IF (IDUMP==1) WRITE(*,'(10F8.3)') AUXBET(1:NORDER)
         IDUMP=0

         NEIG=NORDER
         DO I=1,NORDER
            ! linear dependencies result in AUXBET=0
            IF (AUXBET(I) /=0 ) THEN
              EIGENVAL(I)= ABS(AUXC(I)/AUXBET(I)+1)
            ELSE
              EIGENVAL(I)=0
            ENDIF
         ENDDO
         OPT=0
         EMAX=0
         DO I=1,NORDER
            IF (AUXBET(I) /=0 ) THEN
              OPT=OPT+ABS(AUXC(I)/AUXBET(I)+1)
              EMAX=MAX(EMAX,ABS(AUXC(I)/AUXBET(I)+1))
            ELSE
              EMAX=EMAX_ALLOWED
            ENDIF
         END DO
         ! if ill-conditioned/linear-dependencies (one eigenvalue larger than EMAX_ALLOWED)
         ! remove the information from iteration history
         ! if this is not done, and if there is
         ! a force-component into this direction, G f will hopelessly overshoot
         ! also remove iteration history if MAXIT is exceeded
         IF (NORDER<MAXIT .AND. EMAX < EMAX_ALLOWED) EXIT
         IF (NORDER.EQ.1) THEN
            IF (IU0>=0) WRITE(IU0,*)'BRIONS problems: POTIM should be increased'
            EXIT
         ENDIF
      ENDDO
      OPT=OPT/NORDER
      IF (IU6>=0) WRITE(IU6,100) NORDER,OPT,EIGENVAL(1:NORDER)

100   FORMAT( ' retain information from N=',I3,' steps',/ &
              ' eigenvalues of (default step * inverse Hessian matrix)' / &
              '  average eigenvalue of G= ',F8.4,/ &
              ' eigenvalue spectrum of G is', &
              (10F8.4))

      IF(IU0>=0) THEN
         WRITE(IU0,200) NORDER,OPT,EIGENVAL(1:MIN(10,NORDER))
 200     FORMAT(' retain N=',I3,' mean eig=',F5.2 / ' eig: ',10F7.3)
      ENDIF

! shift unnecessary data away
      IF (ISHIFT/=0) THEN
         CALL BR_SHIFT_MATRIX(FINF,MAXIT,NORDER,ISHIFT)
         CALL BR_SHIFT_MATRIX(GMAT,MAXIT,NORDER,ISHIFT)
         CALL BR_SHIFT_MATRIX(SMAT,MAXIT,NORDER,ISHIFT)
         CALL BR_SHIFT_VEC(Z_,NDIM,NORDER,ISHIFT)
         CALL BR_SHIFT_VEC(U_,NDIM,NORDER,ISHIFT)
         CALL BR_SHIFT_VEC(DF_,NDIM,NORDER,ISHIFT)
! correct iteration count
         ITER=ITER    -ISHIFT
         ITERM1=ITERM1-ISHIFT
      ENDIF
!=======================================================================
! construct matrix A:
!=======================================================================
      DO J=1,ITERM1
         AMAT(J,J)=1._q+WI(J)*WI(J)*FINF(J,J)
         DO K=1,J-1
            AMAT(J,K)=FINF(J,K)*WI(J)*WI(K)
            AMAT(K,J)=FINF(K,J)*WI(K)*WI(J)
         ENDDO
      ENDDO

      IDUMP=0
      IF (IDUMP==1) THEN
         WRITE(*,*)
         WRITE(*,*)'AMAT'
         DO I=1,ITERM1
            WRITE(*,'(10F10.4)') (AMAT (I,J),J=1,ITERM1)
         ENDDO
         WRITE(*,*)'FINF'
         DO I=1,ITERM1
            WRITE(*,'(10F10.4)') (FINF(I,J),J=1,ITERM1)
         ENDDO
      ENDIF

      CALL INVERS(AMAT,ITERM1,MAXIT,BETA,AUX,INDEX,VV)
!=======================================================================
! Calculate BETAQ used in the update of Z (formula in Johnsons paper is
! wrong if weights are not equal, here is the correct version, gK):
! a few comments here:
!   for Pulays      approach BETAQ is strictly 0 (equ. 103)
!   for Broydens 2. approach BETAQ and BETA have the structure
!            BETAQ     1   0   0      BETA=0 except for
!                      0   1   0      BETA(ITERM1,ITERM1)=1
!                      0   0   1
!                     g_1 g_2  g_3
!            (see equ. 104 and 105 in Comput Mat. Sci.)
!=======================================================================
      BETAQ=0
      DO I=1,ITERM1
         DO IT=1,ITERM1-1
            BETAQ(I,IT)=0._q
            DO J=1,ITERM1
               BETAQ(I,IT)=BETAQ(I,IT)-WI(I)*BETA(I,J)*WI(J)*FINF(J,IT)
            ENDDO
         ENDDO
         BETAQ(I,I)=BETAQ(I,I)+1._q
      ENDDO

      IDUMP=0

      IF (IDUMP==1) THEN
         WRITE(*,*)'BETA'
         DO I=1,ITERM1
            WRITE(*,'(10F10.4)') (BETA (I,J)*WI(I)*WI(J),J=1,ITERM1)
         ENDDO
         WRITE(*,*)'BETAQ'
         DO I=1,ITERM1
            WRITE(*,'(10F10.4)') (BETAQ(I,J),J=1,ITERM1-1)
         ENDDO
      ENDIF
!=======================================================================
! update all the vectors |Z_it^(m-1)>
!=======================================================================
      DO IT=1,ITERM1
         WRK1=0
! SUM__{im=1,iter-2} BETAQ_{it,im} |Z_im^(iter-2)>
         DO IM=1,ITERM1-1
            WRK1=WRK1+BETAQ(IT,IM)*Z_(:,IM)
         ENDDO
! Add SUM__{im=1,iter-1} WI_im WI_it BETA_{it,im} |U^(im)>:
         DO IM=1,ITERM1
            WRK1=WRK1+WI(IT)*WI(IM)*BETA(IT,IM)*U_(:,IM)
         ENDDO
! Save temporarily (we may not yet destroy the old |Z>-vectors!):
         T_(:,IT)=WRK1
      ENDDO
! Finally get the updated |Z_it^(iter-1)> from temporary records 'T':
      DO IT=1,ITERM1
         Z_(:,IT)=T_(:,IT)
      ENDDO
!=======================================================================
! calculate G |F^(iter)>
!=======================================================================
      DO IT=1,ITERM1
! Build <Delta F^(it)|FP^(iter)>:
         SUM_=INPRODUCT(DF_(:,IT),F_LAST,NDIM)
         CALLMPI_C( sum_chain( SUM_ ))
! Subtract |Z_it^(iter-1)><Delta F^(it)|FP^(iter)> from |X^(iter)>:
         X=X-SUM_*Z_(:,IT)
      ENDDO
! Add G^(1) |F^(iter)> to vector |X^(iter)>:
      X=X+G0*F_LAST

      RETURN
      END


!=======================================================================
! removes NSHIFT elements from a matrix AMAT
! NORDER is the final rank of the matrix
!=======================================================================

      SUBROUTINE BR_SHIFT_MATRIX(AMAT,NDIM,NORDER,ISHIFT)
        USE prec
        REAL(q) AMAT(NDIM,NDIM)     ! matrix from which to remove elements
        INTEGER NDIM                ! leading dimension of matrix
        INTEGER NORDER              ! final rank of matrix
        INTEGER ISHIFT              ! magnitude of shift

        DO I=1,NORDER
           DO J=1,NORDER
              AMAT(I,J)=AMAT(I+ISHIFT,J+ISHIFT)
           ENDDO
        ENDDO
      END SUBROUTINE BR_SHIFT_MATRIX

!=======================================================================
! removes ISHIFT elements from a vector set
! NORDER is the final number of vectors
!=======================================================================

      SUBROUTINE BR_SHIFT_VEC(V,NDIM,NORDER,ISHIFT)
        USE prec
        REAL(q) V(NDIM,NORDER+ISHIFT) ! vector to shift
        INTEGER NDIM                ! leading dimension of vector
        INTEGER NORDER              ! final number of vector
        INTEGER ISHIFT              ! magnitude of shift

        DO I=1,NORDER
           V(:,I)=V(:,I+ISHIFT)
        ENDDO
      END SUBROUTINE BR_SHIFT_VEC
