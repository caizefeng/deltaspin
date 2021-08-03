#include "symbol.inc"
!*********************************************************************
! RCS:  $Id: asa.F,v 1.7 2003/06/27 13:22:13 kresse Exp kresse $
!
!  this modul contains code to calculate Clebsch-Gordan coefficients
!  and code to evaluate the integrals of three spherical (real or
!  complex) harmonics
!
!  The part which calculates the Clebsch-Gordan coefficients
!  was kindly supplied by Helmut Nowotny
!  and is taken from an ASW (augmented spherical waves) code
!  written by Williams, Kuebler and Gelat.
!  port to f90, a lot of extensions and code clean up was
!  done by gK (I think I have also written the version for the real
!  spherical harmonics, but I am not shure ;-).
!
!*********************************************************************

  MODULE asa
    USE prec

    INTEGER, PARAMETER :: NCG=13000 ! LMAX= 6 (f electrons)
!   (LMAX=8: NCG=45000, LMAX=10: NCG=123000, LMAX=14: NCG=588000, LMAX=20: NCG=3191000)
    
    REAL(q) :: YLM3(NCG)  ! table which contains the intregral of three
                          ! real spherical harmonics
    REAL(q) :: YLM3I(NCG) ! inverse table
    INTEGER :: JL(NCG)    ! index L for each element in the array YLM3
    INTEGER :: JS(NCG)    ! compound index L,M for each element in the array YLM3
    ! JS =  L*(L+1)+M+1 (where M=-L,...,L)
    INTEGER :: INDCG(NCG) ! index into array YLM3 which gives the starting
                          ! position for one l,m ,lp,mp  quadruplet
    INTEGER :: LMAXCG=-1  ! maximum l
    INTEGER,ALLOCATABLE,SAVE :: YLM3LOOKUP_TABLE(:,:) ! 

  CONTAINS

!
! the organization of the arrays given above is relatively complicated
! YLM3 stores the integrals of   Y_lm Y_l'm' Y_LM
! for each lm l'm' quadruplet only a small number of integrals is nonzero
! the maximum L is given by triangular rule
!             | l- lp | < L < | l + lp |
! and M= m+-m' (real version) or M= m+m' (complex version)
!
! INDCG stores for each l,l',m,m' quadruplet the startpoint where
!       the integrals which are not zero are stored in YLM3
!       (see YLM3LOOKUP)
! JS    stores for each integral stored in YLM3 the
!       the corresponding L and M index
! to transform one l,lp part of an array organized as
! Q((2l+1)+m,(2l'+1)+m') to Q_l,lp (L,M) the following "pseudocode"
! could be used
!

    SUBROUTINE YLM3TRANS(L,LP,QIN,QOUT)
      IMPLICIT NONE
      INTEGER L,LP,M,MP,LMINDX,ISTART,IEND,IC

      REAL(q) QIN(:,:),QOUT(:)

      CALL YLM3LOOKUP(L,LP,LMINDX)
      DO M =1,2*L+1
      DO MP=1,2*LP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            QOUT(JS(IC))= QOUT(JS(IC))+ QOUT(IC)*QIN(L+M-1,LP+MP-1)
         ENDDO
      ENDDO
      ENDDO
    END SUBROUTINE YLM3TRANS

!**************** FUNCTION CLEBGO ************************************
!
! caculate Clebsch-Gordan-coeff. <J1 J2 M1 M2 I J3 M3>
! using racah-formel
! FAC is a user supplied array containing factorials
!
!*********************************************************************

    FUNCTION CLEBGO(FAC,J1,J2,J3,M1,M2,M3)

      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) FAC(40)
      REAL(q) CLEBGO

      IF(M3/=M1+M2) GO TO 2
      K1=J1+J2-J3+1
      K2=J3+J1-J2+1
      K3=J3+J2-J1+1
      K4=J1+J2+J3+2
      T= (2*J3+1)*FAC(K1)*FAC(K2)*FAC(K3)/FAC(K4)
      K1=J1+M1+1
      K2=J1-M1+1
      K3=J2+M2+1
      K4=J2-M2+1
      K5=J3+M3+1
      K6=J3-M3+1
      T=SQRT(T*FAC(K1)*FAC(K2)*FAC(K3)*FAC(K4)*FAC(K5)*FAC(K6))
      N1=MAX0(J2-J3-M1,J1-J3+M2,0)+1
      N2=MIN0(J1+J2-J3,J1-M1,J2+M2)+1
      IF(N1>N2) GO TO 2
      T1=0.0_q
      DO M=N1,N2
         N=M-1
         K1=J1+J2-J3-N+1
         K2=J1-M1-N+1
         K3=J2+M2-N+1
         K4=J3-J2+M1+N+1
         K5=J3-J1-M2+N+1
         T1=T1+ (1+4*(N/2)-2*N)/(FAC(M)*FAC(K1)*FAC(K2)*FAC(K3) &
              &  *FAC(K4)*FAC(K5))
      ENDDO
      CLEBGO=T*T1
      RETURN
! coefficient is zero, drop back
 2    CONTINUE
      CLEBGO=0.0_q
      RETURN

    END FUNCTION CLEBGO

!**************** FUNCTION CLEBG0 ************************************
!
! calculate Clebsch-Gordan-coeff. <L1 L2 0 0 I L3 0>
! using racah-formel
! FAC is a user supplied array containing factorials
!
!*********************************************************************


    FUNCTION CLEBG0(FAC,L1,L2,L3)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER X,P
      REAL(q) FAC(40)
      REAL(q) CLEBG0

      LT=L1+L2+L3
      P=LT/2
      IF(2*P/=LT) GO TO 1
      CLEBG0= SQRT( REAL(2*L3+1,KIND=q)/(LT+1))
      CLEBG0=CLEBG0*FAC(P+1)/SQRT(FAC(2*P+1))
      X=P-L1
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      X=P-L2
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      X=P-L3
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      IF(X>2*(X/2)) CLEBG0=-CLEBG0
      RETURN
! coefficient is zero, drop back
 1    CONTINUE
      CLEBG0=0.0_q
      RETURN
    END FUNCTION CLEBG0

!************************* YLM3ST   **********************************
!
! calculate the integral of the product of three real spherical
! harmonics
! i.e    Y_lm Y_l'm' Y_LM
!
! LMAX     max value for l and lp (maximum L is given by triagular rule
!             | l- lp | < L < | l + lp |
!
!*********************************************************************


    SUBROUTINE YLM3ST(LMAX)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!
      INTEGER S1,S2,S3,T1,T2,T3
      REAL(q) FAC(40)
      REAL(q), PARAMETER :: SRPI =1.772453850905516027_q
!---------------------------------------------------------------------
! function to evaluate (-1)^I
!---------------------------------------------------------------------
      IF (LMAXCG>0) RETURN
      LMAXCG=LMAX
!---------------------------------------------------------------------
! set up table for factorials
!---------------------------------------------------------------------
      IMAX=30
      FAC(1)=1._q
      DO I=1,IMAX
         FAC(I+1)= I*FAC(I)
      ENDDO

      IC=0

      LMIND=0
!---------------------------------------------------------------------
! loop over l,m         m =-l,+l
! loop over lp,mp       mp=-lp,+lp
!---------------------------------------------------------------------
      DO L1=0,LMAX
      DO L2=0,LMAX
      K2=(2*L1+1)*(2*L2+1)

      DO M1=-L1,L1
      DO M2=-L2,L2

         LMIND=LMIND+1
         INDCG(LMIND)=IC+1

         N1=IABS(M1)
         S1=0
         IF(M1<0) S1=1
         T1=0
         IF(M1==0) T1=1

         N2=IABS(M2)
         S2=0
         IF(M2<0) S2=1
         T2=0
         IF(M2==0) T2=1

!---------------------------------------------------------------------
! for integrals of 3 real spherical harmonics
! two M values are possibly nonzero
!---------------------------------------------------------------------
         IF(M1*M2<0) THEN
            M3=-N1-N2
            M3P=-IABS(N1-N2)
            IF(M3P==0) THEN
               NM3=1
            ELSE
               NM3=2
            ENDIF
         ELSE IF (M1*M2==0) THEN
            M3=M1+M2
            M3P=0     ! Dummy initialization, not used for this case
            NM3=1
         ELSE
            M3=N1+N2
            M3P=IABS(N1-N2)
            NM3=2
         ENDIF

 5       N3=IABS(M3)
         S3=0
         IF(M3<0) S3=1
         T3=0
         IF(M3==0) T3=1

!---------------------------------------------------------------------
! loop over L given by triangular rule
!---------------------------------------------------------------------
         Q1= 1/2._q*SQRT( REAL(K2,KIND=q))*FS(N3+(S1+S2+S3)/2)
         Q2= 1/(SQRT(2._q)**(1+T1+T2+T3))

         DO L3=ABS(L1-L2),L1+L2, 2

            IF(N3>L3) CYCLE
            T=0._q
            IF(N1+N2==-N3) T=T+CLEBG0(FAC(1),L1,L2,L3)
            IF(N1+N2==N3 ) &
     &           T=T+CLEBGO(FAC(1),L1,L2,L3, N1, N2, N3)*FS(N3+S3)
            IF(N1-N2==-N3) &
     &           T=T+CLEBGO(FAC(1),L1,L2,L3, N1,-N2,-N3)*FS(N2+S2)
            IF(N1-N2==N3 ) &
     &           T=T+CLEBGO(FAC(1),L1,L2,L3,-N1, N2,-N3)*FS(N1+S1)
            IC=IC+1

            IF (IC>NCG)  THEN
               WRITE(0,*)'ERROR: in YLM3ST IC larger than NCG', &
     &              '       increase NCG'
               STOP
            ENDIF

            T0=CLEBG0(FAC(1),L1,L2,L3)

            YLM3(IC) = Q1*Q2*T*T0/(SRPI* SQRT( REAL(2*L3+1,KIND=q)))
            IF (T0==0) THEN
               YLM3I(IC)=0
            ELSE
               YLM3I(IC)= T*Q2/Q1/T0*(SRPI* SQRT( REAL(2*L3+1,KIND=q)))
            ENDIF
!           WRITE(*,'(6I4,E14.7)')L3*(L3+1)+M3+1,0,L1,L2,M1,M2,YLM3(IC)

            JL(IC)=L3
            JS(IC)=L3*(L3+1)+M3+1
         ENDDO
! if there is a second M value calculate coefficients for this M
         NM3=NM3-1
         M3=M3P
         IF(NM3>0) GO TO 5

      ENDDO
      ENDDO
      ENDDO
      ENDDO

      INDCG(LMIND+1)=IC+1

      ALLOCATE( YLM3LOOKUP_TABLE(0:LMAXCG,0:LMAXCG))

      LMIND=0

      DO L1=0,LMAXCG
      DO L2=0,LMAXCG

         YLM3LOOKUP_TABLE(L1,L2)=LMIND
         LMIND=LMIND+(2*L1+1)*(2*L2+1)

      ENDDO
      ENDDO

      RETURN
      
    CONTAINS

      FUNCTION FS(I)
        INTEGER FS,I
        FS=1-2*MOD(I+20,2)
      END FUNCTION FS

    END SUBROUTINE YLM3ST


!************************* YLM3ST_COMPL ******************************
!
! calculate the integral of the product of three complex
! spherical harmonics
! i.e    Y_lm Y_l'm' Y_LM
!
! LMAX     max value for l and lp (maximum L is given by triagular rule
!             | l- lp | < L < | l + lp |
! YLM3     results (on exit)
!
!*********************************************************************

    SUBROUTINE YLM3ST_COMPL(LMAX)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      REAL(q) FAC(40)
      REAL(q), PARAMETER :: SRPI =1.772453850905516027_q

!---------------------------------------------------------------------
! function to evaluate (-1)^I
!---------------------------------------------------------------------
      IF (LMAXCG>0) RETURN
      LMAXCG=LMAX
!---------------------------------------------------------------------
! set up table for factorials
!---------------------------------------------------------------------
      IMAX=30
      FAC(1)=1._q
      DO I=1,IMAX
         FAC(I+1)= I*FAC(I)
      ENDDO

      IC=0
      LMIND=0
!---------------------------------------------------------------------
! loop over l    ,m     m =-l,+l
! loop over lp<=l,mp    mp=-lp,+lp
!---------------------------------------------------------------------
      DO L1=0,LMAX
      DO L2=0,L1
      K2=(2*L1+1)*(2*L2+1)

      DO M1=-L1,L1
      DO M2=-L2,L2

         LMIND=LMIND+1
         INDCG(LMIND)=IC+1

         M3=M1+M2
!---------------------------------------------------------------------
! loop over L given by triangular rule
!---------------------------------------------------------------------
         Q1= SQRT( REAL(K2,KIND=q)/4 )*FS(M3)

         DO L3=L1-L2,L1+L2, 1

            IF(ABS(M3)>L3) CYCLE

            T =CLEBGO(FAC(1),L1,L2,L3, M1, M2, M3)
            T0=CLEBG0(FAC(1),L1,L2,L3)
            IC=IC+1

            YLM3(IC)=  Q1*T*T0/(SRPI* SQRT( REAL(2*L3+1, KIND=q)))

            IF (T0==0) THEN
               YLM3I(IC)=0
            ELSE
               YLM3I(IC)= T/Q1/T0*(SRPI* SQRT( REAL(2*L3+1, KIND=q)))
            ENDIF

            JL(IC)=L3
            JS(IC)=L3*(L3+1)+M3+1
         ENDDO


      ENDDO
      ENDDO
      ENDDO
      ENDDO

      INDCG(LMIND+1)=IC+1


      ALLOCATE( YLM3LOOKUP_TABLE(0:LMAXCG,0:LMAXCG))

      LMIND=0

      DO L1=0,LMAXCG
      DO L2=0,LMAXCG

         YLM3LOOKUP_TABLE(L1,L2)=LMIND
         LMIND=LMIND+(2*L1+1)*(2*L2+1)

      ENDDO
      ENDDO

      RETURN

    CONTAINS
      FUNCTION FS(I)
        INTEGER FS,I
        FS=1-2*MOD(I+20,2)
      END FUNCTION FS
    END SUBROUTINE YLM3ST_COMPL

!************************* YLM3LOOKUP ********************************
!
! function to look up a the startpoint in the array
! YLM3 for two quantumnumbers l lp
!
!*********************************************************************

    SUBROUTINE YLM3LOOKUP_OLD(L,LP,LMIND)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)

      LMIND=0

      DO L1=0,LMAXCG
      DO L2=0,LMAXCG

         IF (L1==L .AND. L2==LP) RETURN

         LMIND=LMIND+(2*L1+1)*(2*L2+1)
      ENDDO
      ENDDO

      WRITE(0,*)'internal ERROR: YLM3LK: look up of l=',L,' l''=', LP, &
     &  ' was not possible'
      STOP

      RETURN
    END SUBROUTINE YLM3LOOKUP_OLD

!************************* YLM3LOOKUP ********************************
!
! function to look up a the startpoint in the array
! YLM3 for two quantumnumbers l lp
!
!*********************************************************************

    SUBROUTINE YLM3LOOKUP(L,LP,LMIND)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)

      IF (L>LMAXCG .OR. LP>LMAXCG) THEN
         WRITE(*,*) 'internal error in YLM3LOOKUP: L or LP > LMAXCG'
         WRITE(*,*) '  L, LP, LMAXCG',L, LP, LMAXCG
         STOP
      ENDIF
      LMIND=YLM3LOOKUP_TABLE(L,LP)

      RETURN
    END SUBROUTINE YLM3LOOKUP

!************************* SETYLM ************************************
!
! calculate the real spherical harmonics
! for a set of grid points up to LMAX
!
! YLM(:,1:LMAX) is set to Y_lm(hat x)
!
! hat x must be lying on the unit sphere
!
! written by Georg Kresse
!
!*********************************************************************

    SUBROUTINE SETYLM(LYDIM,INDMAX,YLM,X,Y,Z)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX          ! number of points (X,Y,Z)
      REAL(q) YLM(:,:)        ! spherical harmonics
      REAL(q) X(:),Y(:),Z(:)  ! x,y and z coordinates

! local variables
      REAL(q) FAK
      INTEGER IND,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,2) < (LYDIM+1)**2) THEN
         WRITE(0,*)'internal ERROR: SETYLM, insufficient L workspace'
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,1)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(0,*)'internal ERROR: SETYLM, insufficient INDMAX workspace'
         STOP
      ENDIF

      FAK=1/(2._q * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <0) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,1)=FAK
      ENDDO
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <1) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,2)  = (FAK*SQRT(3._q))*Y(IND)
        YLM(IND,3)  = (FAK*SQRT(3._q))*Z(IND)
        YLM(IND,4)  = (FAK*SQRT(3._q))*X(IND)
      ENDDO
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM <2) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,5)= (FAK*SQRT(15._q))  *X(IND)*Y(IND)
        YLM(IND,6)= (FAK*SQRT(15._q))  *Y(IND)*Z(IND)
        YLM(IND,7)= (FAK*SQRT(5._q)/2._q)*(3*Z(IND)*Z(IND)-1)
        YLM(IND,8)= (FAK*SQRT(15._q))  *X(IND)*Z(IND)
        YLM(IND,9)= (FAK*SQRT(15._q)/2._q)*(X(IND)*X(IND)-Y(IND)*Y(IND))
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      IF (LYDIM <3) GOTO 100
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO IND=1,INDMAX
        YLM(IND,LM) = 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO IND=1,INDMAX
                     YLM(IND,LM) = YLM(IND,LM)+ &
                         YLM3I(IC)*YLM(IND,L*L+M)*YLM(IND,LP*LP+MP)
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

 100  CONTINUE

    END SUBROUTINE SETYLM

!************************* SETYLM_GRAD *******************************
!
! calculate spherical harmonics
! and the gradient of the spherical harmonics 
! for a set of grid points up to LMAX 
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
! YLMD(:,1:LMAX,1:3) = d Y_lm(hat x)/d hat x_i
!
! hat x must lie on the unit sphere
!
! to obtain the final gradient one has to use
!
! d Y_lm(x)/ d x_i = sum_j d Y_lm(hat x)/ d hat x_j  d hat x_j/ d x_i
! d Y_lm(x)/ d x_i =
!   sum_j (delta_ij - hat x_i hat x_j)/|x| d Y_lm(hat x)/ d hat x_j 
!
! where hat x = x / ||x||
!
! written by Georg Kresse
!
!*********************************************************************

    SUBROUTINE SETYLM_GRAD(LYDIM,INDMAX,YLM,YLMD,X,Y,Z)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX          ! number of points (X,Y,Z)
      REAL(q) YLM(:,:)        ! spherical harmonics
      REAL(q) YLMD(:,:,:)     ! gradient of spherical harmonics
      REAL(q) X(:),Y(:),Z(:)  ! x,y and z coordinates

! local variables
      REAL(q) FAK
      INTEGER IND,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,2) < (LYDIM+1)**2) THEN
         WRITE(0,*)'internal ERROR: SETYLM, insufficient L workspace'
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,1)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(0,*)'internal ERROR: SETYLM, insufficient INDMAX workspace'
         STOP
      ENDIF

      FAK=1/(2._q * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <0) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,1)  =FAK
        YLMD(IND,1,:)=0
      ENDDO
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <1) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,2)  = (FAK*SQRT(3._q))*Y(IND)
        YLM(IND,3)  = (FAK*SQRT(3._q))*Z(IND)
        YLM(IND,4)  = (FAK*SQRT(3._q))*X(IND)
        ! gradient with respect to x
        YLMD(IND,2,1)= 0
        YLMD(IND,3,1)= 0
        YLMD(IND,4,1)= (FAK*SQRT(3._q))
        ! gradient with respect to y
        YLMD(IND,2,2)= (FAK*SQRT(3._q))
        YLMD(IND,3,2)= 0
        YLMD(IND,4,2)= 0
        ! gradient with respect to z
        YLMD(IND,2,3)= 0
        YLMD(IND,3,3)= (FAK*SQRT(3._q))
        YLMD(IND,4,3)= 0
      ENDDO
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM <2) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,5)= (FAK*SQRT(15._q))  *X(IND)*Y(IND)
        YLM(IND,6)= (FAK*SQRT(15._q))  *Y(IND)*Z(IND)
        YLM(IND,7)= (FAK*SQRT(5._q)/2._q)*(3*Z(IND)*Z(IND)-1)
        YLM(IND,8)= (FAK*SQRT(15._q))  *X(IND)*Z(IND)
        YLM(IND,9)= (FAK*SQRT(15._q)/2._q)*(X(IND)*X(IND)-Y(IND)*Y(IND))
        ! gradient with respect to x
        YLMD(IND,5,1)= (FAK*SQRT(15._q))  *Y(IND)
        YLMD(IND,6,1)= 0
        YLMD(IND,7,1)= 0
        YLMD(IND,8,1)= (FAK*SQRT(15._q))  *Z(IND)
        YLMD(IND,9,1)= (FAK*SQRT(15._q)/2._q)*2*X(IND)
        ! gradient with respect to y
        YLMD(IND,5,2)= (FAK*SQRT(15._q))  *X(IND)
        YLMD(IND,6,2)= (FAK*SQRT(15._q))  *Z(IND)
        YLMD(IND,7,2)= 0
        YLMD(IND,8,2)= 0
        YLMD(IND,9,2)= (FAK*SQRT(15._q)/2._q)*(-2*Y(IND))
        ! gradient with respect to z
        YLMD(IND,5,3)= 0
        YLMD(IND,6,3)= (FAK*SQRT(15._q))  *Y(IND)
        YLMD(IND,7,3)= (FAK*SQRT(5._q)/2._q)*6*Z(IND)
        YLMD(IND,8,3)= (FAK*SQRT(15._q))  *X(IND)
        YLMD(IND,9,3)= 0
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      IF (LYDIM <3) GOTO 100
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO IND=1,INDMAX
        YLM(IND,LM) = 0
        YLMD(IND,LM,1) = 0
        YLMD(IND,LM,2) = 0
        YLMD(IND,LM,3) = 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO IND=1,INDMAX
                     YLM(IND,LM) = YLM(IND,LM)+ &
                         YLM3I(IC)*YLM(IND,L*L+M)*YLM(IND,LP*LP+MP)
                     ! gradient
                     YLMD(IND,LM,1) = YLMD(IND,LM,1)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,1)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,1))
                     YLMD(IND,LM,2) = YLMD(IND,LM,2)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,2)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,2))
                     YLMD(IND,LM,3) = YLMD(IND,LM,3)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,3)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,3))
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

 100  CONTINUE

    END SUBROUTINE SETYLM_GRAD


!************************* SETYLM_GRAD2 ******************************
!
! calculate spherical harmonics
! and the gradient of the spherical harmonics
! for a set of grid points up to LMAX
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
! YLMD(:,1:LMAX,1:3) = 
!       sum_j (delta_ij - hat x_i hat x_j)  d Y_lm(hat x)/ d hat x_j
!
! to obtain the final gradient the YLMD must be divided by the |x|
! (see SETYLM_GRAD)
!
! hat x must be lying on the unit sphere, i.e. hat x = x / |x|
!
! written by Georg Kresse
!
!*********************************************************************

    SUBROUTINE SETYLM_GRAD2(LYDIM,INDMAX,YLM,YLMD,X,Y,Z)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX          ! number of points (X,Y,Z)
      REAL(q) YLM(:,:)        ! spherical harmonics
      REAL(q) YLMD(:,:,:)     ! gradient of spherical harmonics
      REAL(q) X(:),Y(:),Z(:)  ! x,y and z coordinates
    ! local
      INTEGER LM, IND
      REAL(q) YLMX, YLMY, YLMZ

      CALL SETYLM_GRAD(LYDIM,INDMAX,YLM,YLMD,X,Y,Z)

      DO LM=1,(LYDIM+1)*(LYDIM+1)
         DO IND=1,INDMAX
            YLMX=YLMD(IND,LM,1)
            YLMY=YLMD(IND,LM,2)
            YLMZ=YLMD(IND,LM,3)

            YLMD(IND,LM,1) = YLMX-X(IND)*X(IND)*YLMX-X(IND)*Y(IND)*YLMY-X(IND)*Z(IND)*YLMZ
            YLMD(IND,LM,2) = YLMY-Y(IND)*X(IND)*YLMX-Y(IND)*Y(IND)*YLMY-Y(IND)*Z(IND)*YLMZ
            YLMD(IND,LM,3) = YLMZ-Z(IND)*X(IND)*YLMX-Z(IND)*Y(IND)*YLMY-Z(IND)*Z(IND)*YLMZ
         ENDDO
      ENDDO
    END SUBROUTINE SETYLM_GRAD2



!************************* SETYLM_NABLA_YLM *************************
!
! calculate 
!
! YLM_NABLA_YLM(lm,l'm') = |x| < Y_lm | nabla | Y_l'm' >
! YLM_X_YLM(lm,l'm')     =     < Y_lm | x_i/|x| | Y_l'm' >
!
! where Y_lm are the real spherical harmonics
! lm is applied a compound indes: i=m+l*l (l=0,..., m=1,..,2l+1)
! 
! to obtain the required result < Y_lm | nabla | Y_l'm' >, the 
! returned array must be divided by the norm ||x||
!
! presently the routine uses a numerical integration
! although analytical formulas can be readily implemented by hard
! coding L=1 and using Clebsch-Gordan coefficients as above
! for the other quantum numbers
! the numerical integration is however essentially exact as well
!
! written by Georg Kresse
!
!*********************************************************************

    SUBROUTINE SETYLM_NABLA_YLM(LYDIM,YLM_NABLA_YLM,YLM_X_YLM)
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L

!      REAL(q) YLM_NABLA_YLM((LYDIM+1)*(LYDIM+1),(LYDIM+1)*(LYDIM+1),0:3)
      REAL(q) YLM_NABLA_YLM(:,:,0:)
      REAL(q) YLM_X_YLM(:,:,0:)
    ! local
      INTEGER LLMAX, LMMAX, PHPTS, THPTS, NPTS
      INTEGER I, J, NP, IFAIL
      REAL(q) DELTAPHI, SIM_FAKT, TMP(3)
      REAL(q), ALLOCATABLE ::  RADPTS(:,:), XYZPTS(:,:),YLM(:,:),YLMD(:,:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)
      INTEGER LM, LMP
      EXTERNAL GAUSSI2

      IF (SIZE(YLM_NABLA_YLM,1) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_NABLA_YLM,2) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_NABLA_YLM,3) <4 ) THEN
         WRITE(0,*)'internal ERROR: SETYLM_NABLA_YLM, insufficient L workspace',SIZE(YLM_NABLA_YLM,1),SIZE(YLM_NABLA_YLM,2),SIZE(YLM_NABLA_YLM,3)
         STOP
      ENDIF

      IF (SIZE(YLM_X_YLM,1) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_X_YLM,2) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_X_YLM,3) <4 ) THEN
         WRITE(0,*)'internal ERROR: SETYLM_NABLA_YLM, insufficient L workspace',SIZE(YLM_X_YLM,1),SIZE(YLM_X_YLM,2),SIZE(YLM_X_YLM,3)
         STOP
      ENDIF

      LLMAX=LYDIM
      LMMAX=(LLMAX+1)**2
! number of theta and phi pivot points
! the value below perform the integration exactly without error
! less is not possible more is not required
      PHPTS=(LLMAX+1)*2
      THPTS=FLOOR(REAL(LLMAX/2+1,KIND=q))*2
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX),YLMD(NPTS,LMMAX,3),XYZPTS(NPTS,3),RADPTS(NPTS,2))
      ALLOCATE(WEIGHT(THPTS),ABSCIS(THPTS))

      RADPTS=0; WEIGHT=0; ABSCIS=0
      ! set phi positions, equally spaces
      DO I=1,PHPTS
         DO J=1,THPTS
            RADPTS((J-1)*PHPTS+I,2)=(I-1)*DELTAPHI
         ENDDO
      ENDDO
     ! get theta positions (actually get cos(theta)) (Gauss integration)
      CALL GAUSSI(GAUSSI2,-1._q,1._q,0,THPTS,WEIGHT,ABSCIS,IFAIL)
      DO I=1,THPTS
         RADPTS((I-1)*PHPTS+1:I*PHPTS,1)=ABSCIS(I)
      ENDDO
      ! convert radial to cartesian coordinates
      DO I=1,NPTS
         XYZPTS(I,1)=COS(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! x
         XYZPTS(I,2)=SIN(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! y
         XYZPTS(I,3)=RADPTS(I,1)                                 ! z
      ENDDO
    ! get |r| Y_lm on a unit sphere and its derivatives
      YLM=0 ; YLMD=0

      CALL SETYLM_GRAD2(LLMAX,NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

    ! loop over all points in the angular grid

      YLM_NABLA_YLM=0
      YLM_X_YLM=0

      points: DO NP=1,NPTS
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))
         
         DO LM=1,(LYDIM+1)*(LYDIM+1)
            DO LMP=1,(LYDIM+1)*(LYDIM+1)

               TMP=YLMD(NP,LMP,:)

               YLM_NABLA_YLM(LM,LMP,0)=YLM_NABLA_YLM(LM,LMP,0)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)
               YLM_NABLA_YLM(LM,LMP,1)=YLM_NABLA_YLM(LM,LMP,1)+SIM_FAKT*YLM(NP,LM)*TMP(1)
               YLM_NABLA_YLM(LM,LMP,2)=YLM_NABLA_YLM(LM,LMP,2)+SIM_FAKT*YLM(NP,LM)*TMP(2)
               YLM_NABLA_YLM(LM,LMP,3)=YLM_NABLA_YLM(LM,LMP,3)+SIM_FAKT*YLM(NP,LM)*TMP(3)
               YLM_X_YLM(LM,LMP,1)    =YLM_X_YLM(LM,LMP,1)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,1)
               YLM_X_YLM(LM,LMP,2)    =YLM_X_YLM(LM,LMP,2)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,2)
               YLM_X_YLM(LM,LMP,3)    =YLM_X_YLM(LM,LMP,3)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,3)
            ENDDO
         ENDDO
      ENDDO points
!      WRITE(0,*) 'YLM YLM'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(:,:,0)
!      WRITE(0,*) 'YLM YLM1'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,1)
!      WRITE(0,*) 'YLM YLM2'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,2)
!      WRITE(0,*) 'YLM YLM3'
!      WRITE(0,'(16F6.3)') YLM_NABLA_YLM(1:16,:,3)

      DEALLOCATE(YLM,YLMD,XYZPTS,RADPTS)
      DEALLOCATE(WEIGHT,ABSCIS)

    END SUBROUTINE SETYLM_NABLA_YLM

!************************* SETYLM_GRAD3 *******************************
!
! calculate
! the second derivatives of the spherical harmonics
! for a set of grid points up to LMAX
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
! YLMDD(:,1:LMAX,1:6) = d^2 Y_lm(hat x)/d x_i d x_l
!        = (sum_j,k d^2 Y_lm(hat x)/d hat x_j d hat x_k * (delta_ij - hat x_i hat x_j)*(delta_kl - hat x_k hat x_l) )
!          +(sum_j d Y_lm(hat x)/ d hat x_j *( -delta_ij*hat x_l /|x|  -delta_jl*hat x_i /|x| -delta_il*hat x_j /|x| 
!          + 3* hat x_l*hat x_i*hat x_j/|x|))
! to obtain the final second derivative the YLMDD must be divided by the |x|^2
! (see SETYLM_GRAD)
!
! hat x must lie on the unit sphere , i.e. hat x = x / |x|
!
! written by Nicola Seriani
!
!*********************************************************************

    SUBROUTINE SETYLM_GRAD3(LYDIM,INDMAX,YLM,YLMD,YLMDD,X,Y,Z)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX          ! number of points (X,Y,Z)
      REAL(q) YLM(:,:)        ! spherical harmonics
      REAL(q) YLMD(:,:,:)  ! gradient of spherical harmonics
      REAL(q) YLMDD(:,:,:)  ! second derivatives of spherical harmonics
      REAL(q) YLMX, YLMY, YLMZ
      REAL(q) YLMXX, YLMXY, YLMXZ, YLMYY, YLMYZ, YLMZZ
      REAL(q) X(:),Y(:),Z(:)  ! x,y and z coordinates

! local variables
      REAL(q) FAK
      INTEGER IND,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC
      
      FAK=1/(2._q * SQRT(PI))
      
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <0) GOTO 110
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,1)  =FAK
        YLMD(IND,1,:)=0
        YLMDD(IND,1,:)=0
      ENDDO
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <1) GOTO 110
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,2)  = (FAK*SQRT(3._q))*Y(IND)
        YLM(IND,3)  = (FAK*SQRT(3._q))*Z(IND)
        YLM(IND,4)  = (FAK*SQRT(3._q))*X(IND)
        ! gradient with respect to x
        YLMD(IND,2,1)= 0
        YLMD(IND,3,1)= 0
        YLMD(IND,4,1)= (FAK*SQRT(3._q))
        ! gradient with respect to y
        YLMD(IND,2,2)= (FAK*SQRT(3._q))
        YLMD(IND,3,2)= 0
        YLMD(IND,4,2)= 0
        ! gradient with respect to z
        YLMD(IND,2,3)= 0
        YLMD(IND,3,3)= (FAK*SQRT(3._q))
        YLMD(IND,4,3)= 0

        YLMDD(IND,2,:)= 0
        YLMDD(IND,3,:)= 0
        YLMDD(IND,4,:)= 0
      ENDDO
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM <2) GOTO 110
!DIR$ IVDEP
!OCL NOVREC
      DO IND=1,INDMAX
        YLM(IND,5)= (FAK*SQRT(15._q))  *X(IND)*Y(IND)
        YLM(IND,6)= (FAK*SQRT(15._q))  *Y(IND)*Z(IND)
        YLM(IND,7)= (FAK*SQRT(5._q)/2._q)*(3*Z(IND)*Z(IND)-1)
        YLM(IND,8)= (FAK*SQRT(15._q))  *X(IND)*Z(IND)
        YLM(IND,9)= (FAK*SQRT(15._q)/2._q)*(X(IND)*X(IND)-Y(IND)*Y(IND))
        ! gradient with respect to x
        YLMD(IND,5,1)= (FAK*SQRT(15._q))  *Y(IND)
        YLMD(IND,6,1)= 0
        YLMD(IND,7,1)= 0
        YLMD(IND,8,1)= (FAK*SQRT(15._q))  *Z(IND)
        YLMD(IND,9,1)= (FAK*SQRT(15._q)/2._q)*2*X(IND)
        ! gradient with respect to y
        YLMD(IND,5,2)= (FAK*SQRT(15._q))  *X(IND)
        YLMD(IND,6,2)= (FAK*SQRT(15._q))  *Z(IND)
        YLMD(IND,7,2)= 0
        YLMD(IND,8,2)= 0
        YLMD(IND,9,2)= (FAK*SQRT(15._q)/2._q)*(-2*Y(IND))
        ! gradient with respect to z
        YLMD(IND,5,3)= 0
        YLMD(IND,6,3)= (FAK*SQRT(15._q))  *Y(IND)
        YLMD(IND,7,3)= (FAK*SQRT(5._q)/2._q)*6*Z(IND)
        YLMD(IND,8,3)= (FAK*SQRT(15._q))  *X(IND)
        YLMD(IND,9,3)= 0
! Second derivatives:
        ! gradient with respect to x
        YLMDD(IND,5,1)= 0
        YLMDD(IND,5,2)= (FAK*SQRT(15._q))
        YLMDD(IND,5,3)= 0
        YLMDD(IND,6,1)= 0
        YLMDD(IND,6,2)= 0
        YLMDD(IND,6,3)= 0
        YLMDD(IND,7,1)= 0
        YLMDD(IND,7,2)= 0
        YLMDD(IND,7,3)= 0
        YLMDD(IND,8,1)= 0
        YLMDD(IND,8,2)= 0
        YLMDD(IND,8,3)= (FAK*SQRT(15._q))
        YLMDD(IND,9,1)= (FAK*SQRT(15._q)/2._q)*2
        YLMDD(IND,9,2)= 0 
        YLMDD(IND,9,3)= 0
        ! gradient with respect to y
        YLMDD(IND,5,4)= 0
        YLMDD(IND,5,5)= 0
        YLMDD(IND,6,4)= 0
        YLMDD(IND,6,5)= (FAK*SQRT(15._q)) 
        YLMDD(IND,7,4)= 0
        YLMDD(IND,7,5)= 0
        YLMDD(IND,8,4)= 0
        YLMDD(IND,8,5)= 0
        YLMDD(IND,9,4)= (FAK*SQRT(15._q)/2._q)*(-2)
        YLMDD(IND,9,5)= 0
        ! gradient with respect to z
        YLMDD(IND,5,6)= 0
        YLMDD(IND,6,6)= 0
        YLMDD(IND,7,6)= (FAK*SQRT(5._q)/2._q)*6
        YLMDD(IND,8,6)= 0
        YLMDD(IND,9,6)= 0
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      IF (LYDIM <3) GOTO 110
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO IND=1,INDMAX
        YLM(IND,LM) = 0
        YLMD(IND,LM,1) = 0
        YLMD(IND,LM,2) = 0
        YLMD(IND,LM,3) = 0

        YLMDD(IND,LM,:) = 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three real spherical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------

      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)
            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO IND=1,INDMAX
                     YLM(IND,LM) = YLM(IND,LM)+ &
                         YLM3I(IC)*YLM(IND,L*L+M)*YLM(IND,LP*LP+MP)
                     ! gradient
                     YLMD(IND,LM,1) = YLMD(IND,LM,1)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,1)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,1))
                     YLMD(IND,LM,2) = YLMD(IND,LM,2)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,2)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,2))
                     YLMD(IND,LM,3) = YLMD(IND,LM,3)+ &
                         YLM3I(IC)*(YLMD(IND,L*L+M,3)*YLM(IND,LP*LP+MP)+YLM(IND,L*L+M)*YLMD(IND,LP*LP+MP,3))
                     ! second derivatives
                     YLMDD(IND,LM,1) = YLMDD(IND,LM,1)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,1)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,1)*YLMD(IND,LP*LP+MP,1) &
                         +YLMD(IND,L*L+M,1)*YLMD(IND,LP*LP+MP,1)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,1) ) 
                     YLMDD(IND,LM,2) = YLMDD(IND,LM,2)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,2)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,1)*YLMD(IND,LP*LP+MP,2) &
                         +YLMD(IND,L*L+M,2)*YLMD(IND,LP*LP+MP,1)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,2) )
                     YLMDD(IND,LM,3) = YLMDD(IND,LM,3)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,3)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,1)*YLMD(IND,LP*LP+MP,3) &
                         +YLMD(IND,L*L+M,3)*YLMD(IND,LP*LP+MP,1)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,3) )
                     YLMDD(IND,LM,4) = YLMDD(IND,LM,4)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,4)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,2)*YLMD(IND,LP*LP+MP,2) &
                         +YLMD(IND,L*L+M,2)*YLMD(IND,LP*LP+MP,2)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,4) )
                     YLMDD(IND,LM,5) = YLMDD(IND,LM,5)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,5)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,2)*YLMD(IND,LP*LP+MP,3) &
                         +YLMD(IND,L*L+M,3)*YLMD(IND,LP*LP+MP,2)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,5) )
                     YLMDD(IND,LM,6) = YLMDD(IND,LM,6)+ &
                         YLM3I(IC)*(YLMDD(IND,L*L+M,6)*YLM(IND,LP*LP+MP)+ YLMD(IND,L*L+M,3)*YLMD(IND,LP*LP+MP,3) &
                         +YLMD(IND,L*L+M,3)*YLMD(IND,LP*LP+MP,3)+YLM(IND,L*L+M)*YLMDD(IND,LP*LP+MP,6) )

                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
      ENDDO

 110  CONTINUE
      DO LM=1,(LYDIM+1)*(LYDIM+1)
         DO IND=1,INDMAX
            YLMX=YLMD(IND,LM,1)
            YLMY=YLMD(IND,LM,2)
            YLMZ=YLMD(IND,LM,3)
            YLMXX=YLMDD(IND,LM,1)
            YLMXY=YLMDD(IND,LM,2)
            YLMXZ=YLMDD(IND,LM,3)
            YLMYY=YLMDD(IND,LM,4)
            YLMYZ=YLMDD(IND,LM,5)
            YLMZZ=YLMDD(IND,LM,6)

            YLMD(IND,LM,1) = YLMX-X(IND)*X(IND)*YLMX-X(IND)*Y(IND)*YLMY-X(IND)*Z(IND)*YLMZ
            YLMD(IND,LM,2) = YLMY-Y(IND)*X(IND)*YLMX-Y(IND)*Y(IND)*YLMY-Y(IND)*Z(IND)*YLMZ
            YLMD(IND,LM,3) = YLMZ-Z(IND)*X(IND)*YLMX-Z(IND)*Y(IND)*YLMY-Z(IND)*Z(IND)*YLMZ

            YLMDD(IND,LM,1)=YLMXX*(1._q-X(IND)*X(IND))*(1._q-X(IND)*X(IND))- &
                              2._q*YLMXY*(1._q-X(IND)*X(IND))*X(IND)*Y(IND)-&
                              2._q*YLMXZ*(1._q-X(IND)*X(IND))*X(IND)*Z(IND)+&
                              YLMYY*X(IND)*Y(IND)*X(IND)*Y(IND)+&
                              2._q*YLMYZ*X(IND)*Y(IND)*X(IND)*Z(IND)+&
                              YLMZZ*X(IND)*Z(IND)*X(IND)*Z(IND)+&
                              YLMX*3._q*(-X(IND)+X(IND)*X(IND)*X(IND))+&
                              YLMY*(-Y(IND)+3._q*X(IND)*X(IND)*Y(IND))+&
                              YLMZ*(-Z(IND)+3._q*X(IND)*X(IND)*Z(IND))
            YLMDD(IND,LM,2)=-YLMXX*(1._q-X(IND)*X(IND))*X(IND)*Y(IND)+&
                              YLMXY*(1._q-X(IND)*X(IND))*(1._q-Y(IND)*Y(IND))+&
                              YLMXY*X(IND)*Y(IND)*X(IND)*Y(IND)-&
                              YLMXZ*(1._q-X(IND)*X(IND))*Y(IND)*Z(IND)+&
                              YLMXZ*X(IND)*Z(IND)*Y(IND)*X(IND)-&
                              YLMYY*X(IND)*Y(IND)*(1._q-Y(IND)*Y(IND))-&
                              YLMYZ*(1._q-Y(IND)*Y(IND))*X(IND)*Z(IND)+&
                              YLMYZ*Y(IND)*Z(IND)*X(IND)*Y(IND)+&
                              YLMZZ*X(IND)*Z(IND)*Y(IND)*Z(IND)+&
                              YLMX*(-Y(IND)+3._q*X(IND)*X(IND)*Y(IND))+&
                              YLMY*(-X(IND)+3._q*X(IND)*Y(IND)*Y(IND))+&
                              YLMZ*3._q*X(IND)*Y(IND)*Z(IND)
            YLMDD(IND,LM,3)=-YLMXX*(1._q-X(IND)*X(IND))*X(IND)*Z(IND)+&
                              YLMXZ*(1._q-X(IND)*X(IND))*(1._q-Z(IND)*Z(IND))+&
                              YLMXZ*X(IND)*Z(IND)*X(IND)*Z(IND)-&
                              YLMXY*(1._q-X(IND)*X(IND))*Y(IND)*Z(IND)+&
                              YLMXY*X(IND)*Y(IND)*Z(IND)*X(IND)-&
                              YLMZZ*X(IND)*Z(IND)*(1._q-Z(IND)*Z(IND))-&
                              YLMYZ*(1._q-Z(IND)*Z(IND))*X(IND)*Y(IND)+&
                              YLMYZ*X(IND)*Z(IND)*Z(IND)*Y(IND)+&
                              YLMYY*X(IND)*Y(IND)*Z(IND)*Y(IND)+&
                              YLMX*(-Z(IND)+3._q*X(IND)*X(IND)*Z(IND))+&
                              YLMZ*(-X(IND)+3._q*X(IND)*Z(IND)*Z(IND))+&
                              YLMY*3._q*X(IND)*Y(IND)*Z(IND)
            YLMDD(IND,LM,4)=YLMYY*(1._q-Y(IND)*Y(IND))*(1._q-Y(IND)*Y(IND))- &
                              2._q*YLMXY*(1._q-Y(IND)*Y(IND))*X(IND)*Y(IND)-&
                              2._q*YLMYZ*(1._q-Y(IND)*Y(IND))*Y(IND)*Z(IND)+&
                              YLMXX*Y(IND)*X(IND)*Y(IND)*X(IND)+&
                              2._q*YLMXZ*Y(IND)*X(IND)*Y(IND)*Z(IND)+&
                              YLMZZ*Y(IND)*Z(IND)*Y(IND)*Z(IND)+&
                              YLMY*3._q*(-Y(IND)+Y(IND)*Y(IND)*Y(IND))+&
                              YLMX*(-X(IND)+3._q*Y(IND)*Y(IND)*X(IND))+&
                              YLMZ*(-Z(IND)+3._q*Y(IND)*Y(IND)*Z(IND))
            YLMDD(IND,LM,5)=-YLMYY*(1._q-Y(IND)*Y(IND))*Y(IND)*Z(IND)+&
                              YLMYZ*(1._q-Y(IND)*Y(IND))*(1._q-Z(IND)*Z(IND))+&
                              YLMYZ*Y(IND)*Z(IND)*Y(IND)*Z(IND)-&
                              YLMXY*(1._q-Y(IND)*Y(IND))*X(IND)*Z(IND)+&
                              YLMXY*Y(IND)*X(IND)*Z(IND)*Y(IND)-&
                              YLMZZ*Y(IND)*Z(IND)*(1._q-Z(IND)*Z(IND))-&
                              YLMXZ*(1._q-Z(IND)*Z(IND))*Y(IND)*X(IND)+&
                              YLMXZ*Y(IND)*Z(IND)*Z(IND)*X(IND)+&
                              YLMXX*Y(IND)*X(IND)*Z(IND)*X(IND)+&
                              YLMY*(-Z(IND)+3._q*Y(IND)*Y(IND)*Z(IND))+&
                              YLMZ*(-Y(IND)+3._q*Y(IND)*Z(IND)*Z(IND))+&
                              YLMX*3._q*X(IND)*Y(IND)*Z(IND)
            YLMDD(IND,LM,6)=YLMZZ*(1._q-Z(IND)*Z(IND))*(1._q-Z(IND)*Z(IND))- &
                              2._q*YLMXZ*(1._q-Z(IND)*Z(IND))*X(IND)*Z(IND)-&
                              2._q*YLMYZ*(1._q-Z(IND)*Z(IND))*Z(IND)*Y(IND)+&
                              YLMXX*Z(IND)*X(IND)*Z(IND)*X(IND)+&
                              2._q*YLMXY*Z(IND)*X(IND)*Z(IND)*Y(IND)+&
                              YLMYY*Z(IND)*Y(IND)*Z(IND)*Y(IND)+&
                              YLMZ*3._q*(-Z(IND)+Z(IND)*Z(IND)*Z(IND))+&
                              YLMX*(-X(IND)+3._q*Z(IND)*Z(IND)*X(IND))+&
                              YLMY*(-Y(IND)+3._q*Y(IND)*Z(IND)*Z(IND))
         ENDDO
      ENDDO

    END SUBROUTINE SETYLM_GRAD3
 

!************************* SETYLM_XIXJ_YLM ***************************
!
! calculate 
!
!  YLM_XIXJ_YLM(lm,l'm',ij) = < Y_lm | x_i x_j / |x|^2| Y_l'm' >
!
! where Y_lm are the real spherical harmonics
! lm is applied a compound index: i=m+l*l (l=0,..., m=1,..,2l+1)
! 
! presently the routine uses a numerical integration
!
!*********************************************************************

    SUBROUTINE SETYLM_XIXJ_YLM(LYDIM,YLM_XIXJ_YLM)
      USE constant
      IMPLICIT NONE
      INTEGER LYDIM
      REAL(q) YLM_XIXJ_YLM(:,:,:)

      INTEGER LLMAX, LMMAX, PHPTS, THPTS, NPTS
      INTEGER I, J, NP, IFAIL
      REAL(q) DELTAPHI, SIM_FAKT, TMP(6)
      REAL(q), ALLOCATABLE :: RADPTS(:,:), XYZPTS(:,:)
      REAL(q), ALLOCATABLE :: YLM(:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)
      INTEGER LM, LMP
      EXTERNAL GAUSSI2

      ! sanity check
      IF (SIZE(YLM_XIXJ_YLM,1) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_XIXJ_YLM,2) <(LYDIM+1)*(LYDIM+1) .OR. &
          SIZE(YLM_XIXJ_YLM,3) <6 ) THEN
         WRITE(*,*)'SETYLM_LAPLA_YLM: ERROR: SETYLM_XIXJ_YLM, too small',SIZE(YLM_XIXJ_YLM,1),SIZE(YLM_XIXJ_YLM,2),SIZE(YLM_XIXJ_YLM,3)
         STOP
      ENDIF

      LLMAX=LYDIM
      LMMAX=(LLMAX+1)**2
      ! number of theta and phi pivot points
      ! the value below perform the integration exactly without error
      ! less is not possible more is not required
!     PHPTS=(LLMAX+1)*2
      PHPTS=(LLMAX+1)*3
!     THPTS=FLOOR(REAL(LLMAX/2+1,KIND=q))*2
      THPTS=FLOOR(REAL(LLMAX/2+1,KIND=q))*3
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
      ! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX))
      ALLOCATE(XYZPTS(NPTS,3),RADPTS(NPTS,2))
      ALLOCATE(WEIGHT(THPTS),ABSCIS(THPTS))

      RADPTS=0; WEIGHT=0; ABSCIS=0
      ! set phi positions, equally spaces
      DO I=1,PHPTS
         DO J=1,THPTS
            RADPTS((J-1)*PHPTS+I,2)=(I-1)*DELTAPHI
         ENDDO
      ENDDO
      ! get theta positions (actually get cos(theta)) (Gauss integration)
      CALL GAUSSI(GAUSSI2,-1._q,1._q,0,THPTS,WEIGHT,ABSCIS,IFAIL)
      DO I=1,THPTS
         RADPTS((I-1)*PHPTS+1:I*PHPTS,1)=ABSCIS(I)
      ENDDO
      ! convert radial to cartesian coordinates
      DO I=1,NPTS
         XYZPTS(I,1)=COS(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! x
         XYZPTS(I,2)=SIN(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! y
         XYZPTS(I,3)=RADPTS(I,1)                                 ! z
      ENDDO
      ! get |r| Y_lm on a unit sphere
      YLM=0

      CALL SETYLM(LLMAX,NPTS,YLM,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

      ! loop over all points in the angular grid
      YLM_XIXJ_YLM=0

      points: DO NP=1,NPTS
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))
         
         DO LM=1,(LYDIM+1)*(LYDIM+1)
            DO LMP=1,(LYDIM+1)*(LYDIM+1)

               YLM_XIXJ_YLM(LM,LMP,1)=YLM_XIXJ_YLM(LM,LMP,1)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,1)*XYZPTS(NP,1)
               YLM_XIXJ_YLM(LM,LMP,2)=YLM_XIXJ_YLM(LM,LMP,2)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,2)*XYZPTS(NP,2)
               YLM_XIXJ_YLM(LM,LMP,3)=YLM_XIXJ_YLM(LM,LMP,3)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,3)*XYZPTS(NP,3)
               YLM_XIXJ_YLM(LM,LMP,4)=YLM_XIXJ_YLM(LM,LMP,4)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,1)*XYZPTS(NP,2)
               YLM_XIXJ_YLM(LM,LMP,5)=YLM_XIXJ_YLM(LM,LMP,5)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,1)*XYZPTS(NP,3)
               YLM_XIXJ_YLM(LM,LMP,6)=YLM_XIXJ_YLM(LM,LMP,6)+SIM_FAKT*YLM(NP,LM)*YLM(NP,LMP)*XYZPTS(NP,2)*XYZPTS(NP,3)

            ENDDO
         ENDDO
      ENDDO points

      DEALLOCATE(YLM,XYZPTS,RADPTS)
      DEALLOCATE(WEIGHT,ABSCIS)

    END SUBROUTINE SETYLM_XIXJ_YLM

  END MODULE asa


!
! this interface is here because stupid SGI want compile main.F
! if the module asa is used
!

  SUBROUTINE YLM3ST_(LMAX_TABLE)
    USE asa
    CALL YLM3ST(LMAX_TABLE)
  END SUBROUTINE YLM3ST_
