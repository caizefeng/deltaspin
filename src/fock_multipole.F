!#define dotiming
#include "symbol.inc"

!***********************************************************************
!
! this module implements fast multipole corrections to the
! non-local exchange
!
! written by Rene Gaudoin (2010-2112)
! see J.Chem.Phys
!
!***********************************************************************


MODULE fock_multipole
  USE prec
  USE poscar
  USE nonl_high
  USE augfast
  USE wave_high
  USE c2f_interface, ONLY : ERRF
  ! MCALPHA determines whether multipole corrections (open boundary conditions)
  ! are used
  ! MCALPHA = 0 (default) no multipole corrections
  ! MCALPHA < 0 set a suitable default for MCALPHA
  ! MCALPHA > 0 apply that value for MCALPHA
  IMPLICIT NONE
  REAL(q), SAVE :: MCALPHA=0
  LOGICAL, SAVE :: MCALPHA_DEFAULT=.FALSE. 

  ! used for multipole correction
  GDEF, SAVE  :: MQ_MAT(10,10)
  GDEF, SAVE  :: MQ_VEC(10)
  GDEF, SAVE  :: MQ_VAL(10)
  ! setting up the correction:
  GDEF, ALLOCATABLE, SAVE     :: POT_MULTIPOLE_CORR_MQ(:,:)
  GDEF, ALLOCATABLE, SAVE     :: POT_MULTIPOLE_PROJ(:,:)
  COMPLEX(q),ALLOCATABLE,SAVE :: P_PROJ(:,:), P_PROJ2(:,:)
  COMPLEX(q),POINTER,SAVE     :: PROJ_WEIGHT(:)
  COMPLEX(q) :: POTFAK0 
  COMPLEX(q), SAVE  :: A_MAT(10,10),AH_MAT(10,10)
  GDEF, PRIVATE, SAVE  :: MONOPOL, DIPOLE(3), QUAD(3,3)
  COMPLEX(q), SAVE  :: AA_MAT(10,10),AAH_MAT(10,10)
  COMPLEX(q), SAVE :: MQ_MAT_11

  CONTAINS
!**************** SUBROUTINE FOCK_MULTIPOLE_PROJ_SETUP **********************
! 
! determine corrections potentials
! 
!**********************************************************************

  SUBROUTINE FOCK_MULTIPOLE_PROJ_SETUP(LATT_CUR, GRIDHF)
    USE mdipol
    USE lattice
    USE constant
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) :: GRIDHF

!   local variables
    GDEF,ALLOCATABLE :: GTMP(:)

    COMPLEX(q)    :: S_MAT(10,10),SI_MAT(10,10),SR_MAT(10,10),OV_SUM
    COMPLEX(q)    :: U_MAT(10,10),SRI_MAT(10,10),N_MAT(10,10)
    
    INTEGER NC, N1, N2, N3, NG, N, NP
    REAL(q) :: X, Y, Z, R2, TEST_CHARGE, POT, POTD, POTDD, POTE
    REAL(q) :: POTDD2,R2M,NORM
    REAL(q) :: X1, X2, X3, NMAX1, NMAX2, NMAX3, I1, I2, I3
    GDEF    :: M, D1, D2, D3, Q11, Q12, Q13, Q22, Q23, Q33
    REAL(q) :: A(3,3),QF1,QF2,QF3
    REAL(q) :: FACTM,RTMP,TSUM
    
    GDEF    :: TMP_MAT(10,10),PF_MQ(10,10)

    REAL(q) :: XX, XCUTOFF, YCUTOFF, ZCUTOFF  
    REAL(q), PARAMETER :: WIDTH=4
      

    IF (.NOT. ALLOCATED(P_PROJ2)) THEN
       
       ! NMAX(1-3) is the outermost grid point
       NMAX1=GRIDHF%NGX
       NMAX2=GRIDHF%NGY
       NMAX3=GRIDHF%NGZ
       
       ! X(1-3) is the position of the grid point relative to 
       ! which the multipoles are evaluated
       
       X1   =NINT(0.5+DIP%POSCEN(1)*GRIDHF%NGX)+0.5
       X2   =NINT(0.5+DIP%POSCEN(2)*GRIDHF%NGY)+0.5
       X3   =NINT(0.5+DIP%POSCEN(3)*GRIDHF%NGZ)+0.5
       
       A(:,1)=LATT_CUR%A(:,1)
       A(:,2)=LATT_CUR%A(:,2)
       A(:,3)=LATT_CUR%A(:,3)
       
       IF (GRIDHF%RL%NFAST==3) THEN
          ! mpi version:    x-> N2, y-> N3, z-> N1 
          NMAX2=GRIDHF%NGX
          NMAX3=GRIDHF%NGY
          NMAX1=GRIDHF%NGZ
          
!      new: finds closest midpoint between gridpoint, N gridpoints is always even
!      ---> as many points above as below!
          X2   =NINT(0.5+DIP%POSCEN(1)*GRIDHF%NGX)+0.5
          X3   =NINT(0.5+DIP%POSCEN(2)*GRIDHF%NGY)+0.5
          X1   =NINT(0.5+DIP%POSCEN(3)*GRIDHF%NGZ)+0.5
          
          A(:,2)=LATT_CUR%A(:,1)
          A(:,3)=LATT_CUR%A(:,2)
          A(:,1)=LATT_CUR%A(:,3)
       ENDIF

!==========================================================================
! check whether potential corrections have been set up properly yet
! if not yet, set them up
!==========================================================================
       
       
       ALLOCATE(POT_MULTIPOLE_PROJ(GRIDHF%RL%NP,10))
       ALLOCATE(GTMP(m_ GRIDHF%MPLWV)) ! GRIDHF%MPLWV is the number of
                                       ! complex words required for an FFW work array
                                       ! GTMP real values hence m_ -> 2*
       ALLOCATE(P_PROJ(GRIDHF%RC%NP,10))
       ALLOCATE(P_PROJ2(GRIDHF%RC%NP,10))
       ALLOCATE(PROJ_WEIGHT(GRIDHF%RC%NP))
       
       DO NC=1,3
          X= A(1,NC)
          Y= A(2,NC)
          Z= A(3,NC)
          RTMP=(X*X+Y*Y+Z*Z)
          R2=(X*X+Y*Y+Z*Z)*MCALPHA*0.25
          TEST_CHARGE=EXP(-R2)
       ENDDO

       TSUM=0

       NG=0

       DO NC=1,GRIDHF%RL%NCOL
          N2= GRIDHF%RL%I2(NC)
          N3= GRIDHF%RL%I3(NC)
          DO N1=1,GRIDHF%RL%NROW
             NG=NG+1

             I1=(MOD(N1-X1+NMAX1*200.5,NMAX1))*(1.0_q/NMAX1)-0.5
             I2=(MOD(N2-X2+NMAX2*200.5,NMAX2))*(1.0_q/NMAX2)-0.5
             I3=(MOD(N3-X3+NMAX3*200.5,NMAX3))*(1.0_q/NMAX3)-0.5
             
             XX=ABS(ABS(I1*NMAX1)-NMAX1/2)
             XCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                XCUTOFF = 1.0
             ELSE
                XCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I2*NMAX2)-NMAX2/2)
             YCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                YCUTOFF = 1.0
             ELSE
                YCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I3*NMAX3)-NMAX3/2)
             ZCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                ZCUTOFF = 1.0
             ELSE
                ZCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF

             X= I1*A(1,1)+I2*A(1,2)+I3*A(1,3)
             Y= I1*A(2,1)+I2*A(2,2)+I3*A(2,3)
             Z= I1*A(3,1)+I2*A(3,2)+I3*A(3,3)
             

!            projectors in real space
             
             POT_MULTIPOLE_PROJ(NG,1)  =1.0
             POT_MULTIPOLE_PROJ(NG,2)  =X!*XCUTOFF
             POT_MULTIPOLE_PROJ(NG,3)  =Y!*YCUTOFF
             POT_MULTIPOLE_PROJ(NG,4)  =Z!*ZCUTOFF
             POT_MULTIPOLE_PROJ(NG,5)  =X*X
             POT_MULTIPOLE_PROJ(NG,6)  =X*Y
             POT_MULTIPOLE_PROJ(NG,7)  =X*Z
             POT_MULTIPOLE_PROJ(NG,8)  =Y*Y
             POT_MULTIPOLE_PROJ(NG,9)  =Y*Z
             POT_MULTIPOLE_PROJ(NG,10) =Z*Z

         ENDDO

       ENDDO

       NORM=1/( REAL(GRIDHF%RL%NP,q) )

       DO N=1,10
          GTMP(1:GRIDHF%RL%NP)=POT_MULTIPOLE_PROJ(1:GRIDHF%RL%NP,N)

          ! on real space grid GTMP (GDEF) is has real storage layout
          ! for gamma only

          ! in place fft real to reciprocal (GTMP now complex storage layout)
          CALL FFT3D(GTMP, GRIDHF, -1)
          ! multiply by 4 pi e^2/ G^2

          CALL COPY_PROJ(GRIDHF, GTMP, p_proj(:,N))

          P_PROJ2(:,N)=P_PROJ(:,N)*NORM
!          P_PROJ(:,N)=P_PROJ(:,N)*NORM*NORM  ! or just one NORM ??? ; line probably not even needed

       ENDDO
       
       FACTM=1


       DO N=1,10
          DO NG=1,10
             IF (GRIDHF%RC%NFAST==1 .AND. GRIDHF%REAL2CPLX) THEN
                NP=1

                DO NC=1,GRIDHF%RC%NCOL
                   N2= GRIDHF%RC%I2(NC)
                   N3= GRIDHF%RC%I3(NC)
                   DO N1=1,GRIDHF%RC%NROW
                      FACTM=1
                      IF (N1 /= 1 .AND. N1 /= GRIDHF%NGX/2+1) FACTM=2
                      PROJ_WEIGHT(NP)=FACTM

                   ENDDO
                ENDDO
             ELSE IF (.NOT.GRIDHF%REAL2CPLX) THEN

                PROJ_WEIGHT=FACTM
             ELSE 
                WRITE(0,*) 'internal error in FOCK_MULTIPOLE_CORR: currently this storage layout is not supported'
                STOP
             END IF

          ENDDO
       ENDDO
       DEALLOCATE(GTMP,POT_MULTIPOLE_PROJ)
    
    ENDIF
!==========================================================================
! correct potentials
!==========================================================================
    

  END SUBROUTINE FOCK_MULTIPOLE_PROJ_SETUP


 SUBROUTINE FOCK_MULTIPOLE_CORR_SETUP(LATT_CUR, GRIDHF)
    USE mdipol
    USE lattice
    USE constant
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) :: GRIDHF

!   local variables
    GDEF,ALLOCATABLE :: GTMP(:)

    COMPLEX(q)    :: S_MAT(10,10),SI_MAT(10,10),SR_MAT(10,10),OV_SUM
    COMPLEX(q)    :: U_MAT(10,10),SRI_MAT(10,10),N_MAT(10,10)
    
    INTEGER NC, N1, N2, N3, NG, N, NP
    REAL(q) :: X, Y, Z, R2, TEST_CHARGE, POT, POTD, POTDD, POTE
    REAL(q) :: POTDD2,R2M,NORM
    REAL(q) :: X1, X2, X3, NMAX1, NMAX2, NMAX3, I1, I2, I3
    GDEF    :: M, D1, D2, D3, Q11, Q12, Q13, Q22, Q23, Q33
    REAL(q) :: A(3,3),QF1,QF2,QF3
    REAL(q) :: FACTM,RTMP,TSUM
    
    GDEF    :: TMP_MAT(10,10),PF_MQ(10,10)

    REAL(q) :: XX, XCUTOFF, YCUTOFF, ZCUTOFF  
    REAL(q), PARAMETER :: WIDTH=4
      
    REAL(q),   ALLOCATABLE :: POTFAK_LOC(:)        ! 1/(G+dk)**2 (G)

    
    INTEGER NK,NQ
    REAL(q) :: FSG


    IF (.NOT. ALLOCATED(POT_MULTIPOLE_CORR_MQ)) THEN
       

       ! NMAX(1-3) is the outermost grid point
       NMAX1=GRIDHF%NGX
       NMAX2=GRIDHF%NGY
       NMAX3=GRIDHF%NGZ
       
       ! X(1-3) is the position of the grid point relative to 
       ! which the multipoles are evaluated
       
       X1   =NINT(0.5+DIP%POSCEN(1)*GRIDHF%NGX)+0.5
       X2   =NINT(0.5+DIP%POSCEN(2)*GRIDHF%NGY)+0.5
       X3   =NINT(0.5+DIP%POSCEN(3)*GRIDHF%NGZ)+0.5
       
       A(:,1)=LATT_CUR%A(:,1)
       A(:,2)=LATT_CUR%A(:,2)
       A(:,3)=LATT_CUR%A(:,3)
       
       IF (GRIDHF%RL%NFAST==3) THEN
          ! mpi version:    x-> N2, y-> N3, z-> N1 
          NMAX2=GRIDHF%NGX
          NMAX3=GRIDHF%NGY
          NMAX1=GRIDHF%NGZ
          
!      new: finds closest midpoint between gridpoint, N gridpoints is always even
!      ---> as many points above as below!
          X2   =NINT(0.5+DIP%POSCEN(1)*GRIDHF%NGX)+0.5
          X3   =NINT(0.5+DIP%POSCEN(2)*GRIDHF%NGY)+0.5
          X1   =NINT(0.5+DIP%POSCEN(3)*GRIDHF%NGZ)+0.5
          
          A(:,2)=LATT_CUR%A(:,1)
          A(:,3)=LATT_CUR%A(:,2)
          A(:,1)=LATT_CUR%A(:,3)
       ENDIF

!==========================================================================
! check whether potential corrections have been set up properly yet
! if not yet, set them up
!==========================================================================
       ALLOCATE(POT_MULTIPOLE_CORR_MQ(GRIDHF%RL%NP,10))
       
!       ALLOCATE(POT_MULTIPOLE_PROJ(GRIDHF%RL%NP,10))
       ALLOCATE(GTMP(m_ GRIDHF%MPLWV)) ! GRIDHF%MPLWV is the number of
                                       ! complex words required for an FFW work array
                                       ! GTMP real values hence m_ -> 2*
!       ALLOCATE(P_PROJ(GRIDHF%RC%NP,10))
!       ALLOCATE(P_PROJ2(GRIDHF%RC%NP,10))
!       ALLOCATE(PROJ_WEIGHT(GRIDHF%RC%NP))
       
       ALLOCATE(POTFAK_LOC( GRIDHF%MPLWV))

!       CALL SET_GFAC_NOINTER(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK_LOC)

       NK=1
       NQ=1
       FSG=0.0

       CALL SET_GFAC_NOINTER(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK_LOC)

       DO NC=1,3
          X= A(1,NC)
          Y= A(2,NC)
          Z= A(3,NC)
          RTMP=(X*X+Y*Y+Z*Z)
          R2=(X*X+Y*Y+Z*Z)*MCALPHA*0.25
          TEST_CHARGE=EXP(-R2)
       ENDDO

       TSUM=0

       NG=0

       DO NC=1,GRIDHF%RL%NCOL
          N2= GRIDHF%RL%I2(NC)
          N3= GRIDHF%RL%I3(NC)
          DO N1=1,GRIDHF%RL%NROW
             NG=NG+1

             I1=(MOD(N1-X1+NMAX1*200.5,NMAX1))*(1.0_q/NMAX1)-0.5
             I2=(MOD(N2-X2+NMAX2*200.5,NMAX2))*(1.0_q/NMAX2)-0.5
             I3=(MOD(N3-X3+NMAX3*200.5,NMAX3))*(1.0_q/NMAX3)-0.5
             
             XX=ABS(ABS(I1*NMAX1)-NMAX1/2)
             XCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                XCUTOFF = 1.0
             ELSE
                XCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I2*NMAX2)-NMAX2/2)
             YCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                YCUTOFF = 1.0
             ELSE
                YCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I3*NMAX3)-NMAX3/2)
             ZCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                ZCUTOFF = 1.0
             ELSE
                ZCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF

             X= I1*A(1,1)+I2*A(1,2)+I3*A(1,3)
             Y= I1*A(2,1)+I2*A(2,2)+I3*A(2,3)
             Z= I1*A(3,1)+I2*A(3,2)+I3*A(3,3)
             
             R2=(X*X+Y*Y+Z*Z)

             TEST_CHARGE=-EXP(-MCALPHA*R2)*LATT_CUR%OMEGA
             
             POT_MULTIPOLE_CORR_MQ(NG,1)  =TEST_CHARGE
             POT_MULTIPOLE_CORR_MQ(NG,2)  =TEST_CHARGE*X
             POT_MULTIPOLE_CORR_MQ(NG,3)  =TEST_CHARGE*Y
             POT_MULTIPOLE_CORR_MQ(NG,4)  =TEST_CHARGE*Z
             POT_MULTIPOLE_CORR_MQ(NG,5)  =TEST_CHARGE*X*X
             POT_MULTIPOLE_CORR_MQ(NG,6)  =TEST_CHARGE*X*Y
             POT_MULTIPOLE_CORR_MQ(NG,7)  =TEST_CHARGE*X*Z
             POT_MULTIPOLE_CORR_MQ(NG,8)  =TEST_CHARGE*Y*Y
             POT_MULTIPOLE_CORR_MQ(NG,9)  =TEST_CHARGE*Y*Z
             POT_MULTIPOLE_CORR_MQ(NG,10) =TEST_CHARGE*Z*Z

         ENDDO
       ENDDO

       NORM=1/( REAL(GRIDHF%RL%NP,KIND=q) )

       DO N=1,10


          GTMP(1:GRIDHF%RL%NP)=POT_MULTIPOLE_CORR_MQ(1:GRIDHF%RL%NP,N)

          ! on real space grid GTMP (GDEF) is has real storage layout
          ! for gamma only
          
          ! in place fft real to reciprocal (GTMP now complex storage layout)
          CALL FFT3D(GTMP, GRIDHF, -1)
          ! multiply by 4 pi e^2/ G^2
          CALL APPLY_GFAC(GRIDHF, GTMP(1), POTFAK_LOC(1))
          ! in place fft to real space (now GTMP has real storage layout again)
          CALL FFT3D(GTMP, GRIDHF,1)
          POT_MULTIPOLE_CORR_MQ(1:GRIDHF%RL%NP,N)=GTMP(1:GRIDHF%RL%NP)
       ENDDO
       
       NG=0 
       MQ_MAT=0

       DO NC=1,GRIDHF%RL%NCOL
          N2= GRIDHF%RL%I2(NC)
          N3= GRIDHF%RL%I3(NC)
          DO N1=1,GRIDHF%RL%NROW
             NG=NG+1

             I1=(MOD(N1-X1+NMAX1*200.5,NMAX1))*(1.0_q/NMAX1)-0.5
             I2=(MOD(N2-X2+NMAX2*200.5,NMAX2))*(1.0_q/NMAX2)-0.5
             I3=(MOD(N3-X3+NMAX3*200.5,NMAX3))*(1.0_q/NMAX3)-0.5

             X= I1*A(1,1)+I2*A(1,2)+I3*A(1,3)
             Y= I1*A(2,1)+I2*A(2,2)+I3*A(2,3)
             Z= I1*A(3,1)+I2*A(3,2)+I3*A(3,3)

             R2=(X*X+Y*Y+Z*Z)
             ! 4 pi e^2 -> EDEPS 
             ! non-periodic potential
             ! pot=-2Pi/alpha 1/r * int_0^r exp(-alpha t^2) dt
             ! 2Pi/alpha-> edeps/(2alpha)
             R2M=R2*MCALPHA
             
             IF(R2M<=1E-5) THEN
                ! at small r
                POT=-1.0_q-(0.1_q*R2M-1.0_q/3.0_q)*R2M
                POT=POT*EDEPS/MCALPHA/2
                POTD=EDEPS*(1.0_q/3.0_q-0.2_q*R2M)
                POTDD=-EDEPS*MCALPHA*0.4_q
             ELSE
                !at large r
                POT=-EDEPS*ERRF(SQRT(R2M))*SQRT(PI/MCALPHA/R2)/MCALPHA/4
                ! its derivatives ....
                POTE=-EXP(-R2M)
                POTD=(EDEPS*POTE/MCALPHA/2-POT)/R2
                POTDD=-(EDEPS*POTE+3.0_q*POTD)/R2
             ENDIF
              ! make sure derivatives are compatible with derivatives of rho
             POTD=-POTD/2/MCALPHA
             POTDD=POTDD/4/(MCALPHA*MCALPHA)
             POTDD2=(POT-POTD)/2/MCALPHA

             ! subtract periodic from non periodic
             POT_MULTIPOLE_CORR_MQ(NG,1)  =POT_MULTIPOLE_CORR_MQ(NG,1)  -POT
             POT_MULTIPOLE_CORR_MQ(NG,2)  =POT_MULTIPOLE_CORR_MQ(NG,2)  -POTD*X
             POT_MULTIPOLE_CORR_MQ(NG,3)  =POT_MULTIPOLE_CORR_MQ(NG,3)  -POTD*Y
             POT_MULTIPOLE_CORR_MQ(NG,4)  =POT_MULTIPOLE_CORR_MQ(NG,4)  -POTD*Z
             POT_MULTIPOLE_CORR_MQ(NG,5)  =POT_MULTIPOLE_CORR_MQ(NG,5)  -(POTDD*X*X+POTDD2)
             POT_MULTIPOLE_CORR_MQ(NG,6)  =POT_MULTIPOLE_CORR_MQ(NG,6)  -POTDD*X*Y
             POT_MULTIPOLE_CORR_MQ(NG,7)  =POT_MULTIPOLE_CORR_MQ(NG,7)  -POTDD*X*Z
             POT_MULTIPOLE_CORR_MQ(NG,8)  =POT_MULTIPOLE_CORR_MQ(NG,8)  -(POTDD*Y*Y+POTDD2)
             POT_MULTIPOLE_CORR_MQ(NG,9)  =POT_MULTIPOLE_CORR_MQ(NG,9)  -POTDD*Y*Z
             POT_MULTIPOLE_CORR_MQ(NG,10) =POT_MULTIPOLE_CORR_MQ(NG,10) -(POTDD*Z*Z+POTDD2)

             TEST_CHARGE=EXP(-MCALPHA*R2)

! DIPOLES <--> MONOPOLES/QUADRUPOLES don't mix:

             MQ_MAT(2:4,2)= MQ_MAT(2:4,2)+POT_MULTIPOLE_CORR_MQ(NG,2:4)*TEST_CHARGE*X  
             MQ_MAT(2:4,3)= MQ_MAT(2:4,3)+POT_MULTIPOLE_CORR_MQ(NG,2:4)*TEST_CHARGE*Y  
             MQ_MAT(2:4,4)= MQ_MAT(2:4,4)+POT_MULTIPOLE_CORR_MQ(NG,2:4)*TEST_CHARGE*Z 
             MQ_MAT(5:10,6)= MQ_MAT(5:10,6)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*X*Y  
             MQ_MAT(5:10,7)= MQ_MAT(5:10,7)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*X*Z  
             MQ_MAT(5:10,9)= MQ_MAT(5:10,9)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*Y*Z 
             MQ_MAT(5:10,1)= MQ_MAT(5:10,1)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE
             MQ_MAT(5:10,5)= MQ_MAT(5:10,5)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*X*X  
             MQ_MAT(5:10,8)= MQ_MAT(5:10,8)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*Y*Y  
             MQ_MAT(5:10,10)= MQ_MAT(5:10,10)+POT_MULTIPOLE_CORR_MQ(NG,5:10)*TEST_CHARGE*Z*Z 
             MQ_MAT(1,6)= MQ_MAT(1,6)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*X*Y  
             MQ_MAT(1,7)= MQ_MAT(1,7)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*X*Z  
             MQ_MAT(1,9)= MQ_MAT(1,9)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*Y*Z 
             MQ_MAT(1,1)= MQ_MAT(1,1)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE
             MQ_MAT(1,5)= MQ_MAT(1,5)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*X*X  
             MQ_MAT(1,8)= MQ_MAT(1,8)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*Y*Y  
             MQ_MAT(1,10)= MQ_MAT(1,10)+POT_MULTIPOLE_CORR_MQ(NG,1)*TEST_CHARGE*Z*Z 
          ENDDO
       ENDDO

       DEALLOCATE(GTMP,POTFAK_LOC,POT_MULTIPOLE_CORR_MQ)

       MQ_MAT=MQ_MAT*LATT_CUR%OMEGA/(NMAX1*NMAX2*NMAX3)

       QF1=MCALPHA*sqrt(MCALPHA/PI)/PI
       QF2=QF1*MCALPHA
       QF3=QF2*MCALPHA

       PF_MQ=0

       PF_MQ(2,2)=2*QF2
       PF_MQ(3,3)=2*QF2
       PF_MQ(4,4)=2*QF2    
       PF_MQ(6,6)=4*QF3
       PF_MQ(7,7)=4*QF3
       PF_MQ(9,9)=4*QF3
       PF_MQ(1,1)=2.5*QF1
       PF_MQ(5,5)=2*QF3
       PF_MQ(8,8)=2*QF3
       PF_MQ(10,10)=2*QF3
       PF_MQ(1,5)=-QF2
       PF_MQ(1,8)=-QF2
       PF_MQ(1,10)=-QF2
       PF_MQ(5,1)=-QF2
       PF_MQ(8,1)=-QF2
       PF_MQ(10,1)=-QF2

       TMP_MAT=MATMUL(MQ_MAT,PF_MQ)
       MQ_MAT=MATMUL(PF_MQ,TMP_MAT)


       DO N=1,10
          DO NP=N,10
             MQ_MAT(N,NP)=0.5_Q*(MQ_MAT(N,NP)+GCONJG(MQ_MAT(NP,N)))
             MQ_MAT(NP,N)=GCONJG(MQ_MAT(N,NP))
          ENDDO
       ENDDO

       MQ_MAT=MQ_MAT/( REAL(GRIDHF%RL%NP,KIND=q) )

       MQ_MAT_11=MQ_MAT(1,1)



!       FACTM=1

!       POTFAK0=0
!       POTFAK0=MQ_MAT(1,1)*P_PROJ2(1,1)*CONJG(P_PROJ2(1,1))
!       MQ_MAT(1,1)=0


!       DO N=1,10
!          DO NG=1,10
!             IF (GRIDHF%RC%NFAST==1 .AND. GRIDHF%REAL2CPLX) THEN
!                NP=1
!                OV_SUM=0.0
!                DO NC=1,GRIDHF%RC%NCOL
!                   N2= GRIDHF%RC%I2(NC)
!                   N3= GRIDHF%RC%I3(NC)
!                   DO N1=1,GRIDHF%RC%NROW
!                      FACTM=1
!                      IF (N1 /= 1 .AND. N1 /= GRIDHF%NGX/2+1) FACTM=2
!                      PROJ_WEIGHT(NP)=FACTM
!                      IF(NP>1) THEN
!                         OV_SUM=OV_SUM+GREAL(P_PROJ2(NP,N)*CONJG(P_PROJ2(NP,NG))*FACTM)/POTFAK(NP)
!                      ELSE
!                         OV_SUM=OV_SUM+GREAL(P_PROJ2(NP,N)*CONJG(P_PROJ2(NP,NG))*FACTM)/POTFAK0
!                      ENDIF
!                      NP=NP+1
!                   ENDDO
!                ENDDO
!             ELSE IF (.NOT.GRIDHF%REAL2CPLX) THEN
!                OV_SUM=0.0                
!              note here we have set potfak(1)[which was 0] to potfak0 
!               which is admissible as mq_mat(1,1) is now zero
!                OV_SUM=OV_SUM+P_PROJ2(1,N)*CONJG(P_PROJ2(1,NG))/POTFAK0
!                DO NP=2,GRIDHF%RC%NP
!              note here we have set potfak(1)[which was 0] to potfak0 
!               which is admissible as mq_mat(1,1) is now zero
!                   OV_SUM=OV_SUM+P_PROJ2(NP,N)*CONJG(P_PROJ2(NP,NG))/POTFAK(NP)
!                ENDDO
!                PROJ_WEIGHT=FACTM
!             ELSE 
!                WRITE(0,*) 'internal error in FOCK_MULTIPOLE_CORR: currently this storage layout is not supported'
!                STOP
!             END IF
!             S_MAT(N,NG)=OV_SUM
!          ENDDO
!       ENDDO

!       SR_MAT=S_MAT

!       CALL MATRIX_CSQRT(SR_MAT,10,1,0)
!       SRI_MAT=S_MAT
!       CALL MATRIX_CSQRT(SRI_MAT,10,1,1)
!       SI_MAT=S_MAT
!       CALL MATRIX_CSQRT(SI_MAT,10,0,1)

!       U_MAT=0
!       DO N=1,10
!          U_MAT(N,N)=CMPLX(1.0_Q,0.0_Q)
!       ENDDO
!   some test version ?
!       N_MAT=MQ_MAT+SI_MAT
!       CALL MATRIX_CSQRT(N_MAT,10,1,0)
!       N_MAT=N_MAT-SI_MAT

!       N_MAT=MATMUL(MQ_MAT,SR_MAT)
!       N_MAT=MATMUL(SR_MAT,N_MAT)
!       N_MAT=N_MAT+U_MAT
       
!       CALL MATRIX_CSQRT(N_MAT,10,1,0)

!       A_MAT=MATMUL(SRI_MAT,N_MAT)
!       A_MAT=MATMUL(A_MAT,SRI_MAT)

!       A_MAT=A_MAT-SI_MAT
!       AH_MAT=A_MAT
    
    ENDIF
!==========================================================================
! correct potentials
!==========================================================================
    

  END SUBROUTINE FOCK_MULTIPOLE_CORR_SETUP


!**************** SUBROUTINE GW_MULTIPOLE_PROJ_SETUP **********************
! 
! determine corrections projectors (ACFDT)
! 
!**********************************************************************

  SUBROUTINE GW_MULTIPOLE_PROJ_SETUP( WGWQ, LATT_CUR, DATAKE, DATAKEMAX)
    USE mdipol
    USE lattice
    USE constant
    TYPE (latt) LATT_CUR
    TYPE (wavedes1) WGWQ
    REAL(q) :: DATAKE(:)   ! 1/(G+dk)**2 (G)

!    local variables
    COMPLEX(q),ALLOCATABLE :: GTMP(:)

    COMPLEX(q)    :: S_MAT(10,10),SI_MAT(10,10),SR_MAT(10,10),OV_SUM
    COMPLEX(q)    :: U_MAT(10,10),SRI_MAT(10,10),N_MAT(10,10)
    
    INTEGER NC, N1, N2, N3, NG, N, NP, IND
    REAL(q) :: X, Y, Z, R2, TEST_CHARGE, POT, POTD, POTDD, POTE
    REAL(q) :: POTDD2,R2M
    REAL(q) :: X1, X2, X3, NMAX1, NMAX2, NMAX3, I1, I2, I3
    GDEF    :: M, D1, D2, D3, Q11, Q12, Q13, Q22, Q23, Q33
    REAL(q) :: A(3,3),QF1,QF2,QF3
    REAL(q) :: FACTM,RTMP, XX, XCUTOFF, YCUTOFF, ZCUTOFF
    REAL(q) NORM
    COMPLEX(q) :: TSUM,SUMTMP,TMP2_MAT(10,10)
    
    INTEGER T1,T2,T3,NP0
    REAL(q) :: POTFAK0
!      orig = 0.001
    REAL(q), PARAMETER :: DATAKEMIN=0.001
    REAL(q) :: DATAKEMAX
! a smooth cutoff function is used for potentials and charge densities
! around the dividing plane
! WIDTH must be at least 1
! 4 grid points around the step function are usually a good choice
    REAL(q), PARAMETER :: WIDTH=4

!    GDEF    :: TMP_MAT(10,10),PF_MQ(10,10)
!    COMPLEX(q)    :: AMQ_MAT(10,10)



    INTEGER TCOUNT

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) THEN
       NP=NP*2
    ENDIF
       
    ! NMAX(1-3) is the outermost grid point
    NMAX1=WGWQ%GRID%NGX
    NMAX2=WGWQ%GRID%NGY
    NMAX3=WGWQ%GRID%NGZ
    
    ! X(1-3) is the position of the grid point relative to 
    ! which the multipoles are evaluated
       
    X1   =NINT(0.5+DIP%POSCEN(1)*WGWQ%GRID%NGX)+0.5
    X2   =NINT(0.5+DIP%POSCEN(2)*WGWQ%GRID%NGY)+0.5
    X3   =NINT(0.5+DIP%POSCEN(3)*WGWQ%GRID%NGZ)+0.5
    
    A(:,1)=LATT_CUR%A(:,1)
    A(:,2)=LATT_CUR%A(:,2)
    A(:,3)=LATT_CUR%A(:,3)
       
    IF (WGWQ%GRID%RL%NFAST==3) THEN
       ! mpi version:    x-> N2, y-> N3, z-> N1 
       NMAX2=WGWQ%GRID%NGX
       NMAX3=WGWQ%GRID%NGY
       NMAX1=WGWQ%GRID%NGZ
          
!      new: finds closest midpoint between gridpoint, N gridpoints is always even
!      ---> as many points above as below!
       X2   =NINT(0.5+DIP%POSCEN(1)*WGWQ%GRID%NGX)+0.5
       X3   =NINT(0.5+DIP%POSCEN(2)*WGWQ%GRID%NGY)+0.5
       X1   =NINT(0.5+DIP%POSCEN(3)*WGWQ%GRID%NGZ)+0.5
          
       A(:,2)=LATT_CUR%A(:,1)
       A(:,3)=LATT_CUR%A(:,2)
       A(:,1)=LATT_CUR%A(:,3)
    ENDIF
!==========================================================================
! check whether potential corrections have been set up properly yet
! if not yet, set them up
!==========================================================================

    IF (.NOT. ALLOCATED(POT_MULTIPOLE_CORR_MQ)) THEN
       ALLOCATE(POT_MULTIPOLE_CORR_MQ(WGWQ%GRID%RL%NP,10))
    ENDIF
    IF (.NOT. ALLOCATED(POT_MULTIPOLE_PROJ)) THEN
       ALLOCATE(POT_MULTIPOLE_PROJ(WGWQ%GRID%RL%NP,10))
    ENDIF
    IF (.NOT. ALLOCATED(GTMP)) THEN
       ALLOCATE(GTMP(WGWQ%GRID%MPLWV+30*30)) ! WGWQ%GRID%MPLWV is the number of
    ENDIF

! we need new arrays for p_proj since now we operate in a
! cutoff sphere not in the cube, and also data distribution
! is vastly different
! they are complex and have size 
! mind fock routines has already been initialized reusing the global p_rpoj2
! will not work I believe
    IF (.NOT. ALLOCATED(P_PROJ)) THEN
       ALLOCATE(P_PROJ(WGWQ%NRPLWV,10))
    ENDIF
    IF (.NOT. ALLOCATED(P_PROJ2)) THEN
       ALLOCATE(P_PROJ2(WGWQ%NRPLWV,10))
    ENDIF
    
    DO NC=1,3
       X= A(1,NC)
       Y= A(2,NC)
       Z= A(3,NC)
       RTMP=(X*X+Y*Y+Z*Z)
       R2=(X*X+Y*Y+Z*Z)*MCALPHA*0.25
       TEST_CHARGE=EXP(-R2)
    ENDDO


    TSUM=0.0
    
    NG=0
    DO NC=1,WGWQ%GRID%RL%NCOL
       N2= WGWQ%GRID%RL%I2(NC)
       N3= WGWQ%GRID%RL%I3(NC)
       DO N1=1,WGWQ%GRID%RL%NROW
          NG=NG+1
          
          !
          I1=(MOD(N1-X1+NMAX1*200.5,NMAX1))*(1.0_Q/NMAX1)-0.5
          I2=(MOD(N2-X2+NMAX2*200.5,NMAX2))*(1.0_Q/NMAX2)-0.5
          I3=(MOD(N3-X3+NMAX3*200.5,NMAX3))*(1.0_Q/NMAX3)-0.5
          
          XX=ABS(ABS(I1*NMAX1)-NMAX1/2)
          XCUTOFF=1.0
          IF (XX > WIDTH)  THEN
             XCUTOFF = 1.0
          ELSE
             XCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
          ENDIF
          
          XX=ABS(ABS(I2*NMAX2)-NMAX2/2)
          YCUTOFF=1.0
          IF (XX > WIDTH)  THEN
             YCUTOFF = 1.0
          ELSE
             YCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
          ENDIF
          
          XX=ABS(ABS(I3*NMAX3)-NMAX3/2)
          ZCUTOFF=1.0
          IF (XX > WIDTH)  THEN
             ZCUTOFF = 1.0
          ELSE
             ZCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
          ENDIF
          
          X= I1*A(1,1)+I2*A(1,2)+I3*A(1,3)
          Y= I1*A(2,1)+I2*A(2,2)+I3*A(2,3)
          Z= I1*A(3,1)+I2*A(3,2)+I3*A(3,3)
          
          R2=(X*X+Y*Y+Z*Z)
          
          
!          real space projectors                    
          POT_MULTIPOLE_PROJ(NG,1)  =1.0
          POT_MULTIPOLE_PROJ(NG,2)  =X !*XCUTOFF
          POT_MULTIPOLE_PROJ(NG,3)  =Y !*YCUTOFF
          POT_MULTIPOLE_PROJ(NG,4)  =Z !*ZCUTOFF
          POT_MULTIPOLE_PROJ(NG,5)  =X*X
          POT_MULTIPOLE_PROJ(NG,6)  =X*Y
          POT_MULTIPOLE_PROJ(NG,7)  =X*Z
          POT_MULTIPOLE_PROJ(NG,8)  =Y*Y
          POT_MULTIPOLE_PROJ(NG,9)  =Y*Z
          POT_MULTIPOLE_PROJ(NG,10) =Z*Z
       ENDDO
    ENDDO
    
    NORM=1.0_Q/( REAL(WGWQ%GRID%NPLWV,q) )
    
    DO N=1,10
    
       GTMP=0
       GTMP(1:WGWQ%GRID%RL%NP)=POT_MULTIPOLE_PROJ(1:WGWQ%GRID%RL%NP,N)

          ! on real space grid GTMP (GDEF) is has real storage layout
          ! for gamma only
       TSUM=0
          
       CALL FFTEXT(WGWQ%NGVECTOR,WGWQ%NINDPW(1), &
            GTMP(1), &
            P_PROJ(1,N), WGWQ%GRID,.FALSE.)

       P_PROJ2(:,N)=P_PROJ(:,N)*NORM


    ENDDO
    
    DEALLOCATE(GTMP)

    MQ_MAT(1,1) = MQ_MAT_11 
   
    POTFAK0=0
    POTFAK0=MQ_MAT(1,1)*P_PROJ2(1,1)*CONJG(P_PROJ2(1,1))
    MQ_MAT(1,1)=0
    
    FACTM=1
    DO N=1,10
       DO NG=1,10
          OV_SUM=0.0 
          DO NP=1,WGWQ%NGVECTOR
             IND=WGWQ%NINDPW(NP)
             N1= MOD((IND-1),WGWQ%GRID%RC%NROW)+1
             NC= (IND-1)/WGWQ%GRID%RC%NROW+1
             N2= WGWQ%GRID%RC%I2(NC)
             N3= WGWQ%GRID%RC%I3(NC)
             !rG special treatment for G=0 should be placed here
             IF (N1==1 .AND. N2==1 .AND. N3==1) THEN
                !  special G=0 treatment
                OV_SUM=OV_SUM+P_PROJ2(NP,N)*CONJG(P_PROJ2(NP,NG))/POTFAK0
                NP0=NP
             ELSE
                IF (SQRT(DATAKE(NP)*DATAKE(NP))>DATAKEMIN*DATAKEMAX) THEN
                   OV_SUM=OV_SUM+P_PROJ2(NP,N)*CONJG(P_PROJ2(NP,NG))/DATAKE(NP)/DATAKE(NP)
                ENDIF
             ENDIF
          ENDDO
          S_MAT(N,NG)=OV_SUM
       ENDDO
    ENDDO

!   should not really happen:       
    IF(POTFAK0<0.0) THEN
       WRITE(*,*) "internal error in GW_MULTIPOLE_PROJ_SETUP: PF0<0 - NEED TO THINK ABOUT THIS!"
       STOP
    ELSE
! normfactor already in datake( k != np0)
       DATAKE(NP0)=SQRT(POTFAK0)
    ENDIF

    TCOUNT = 0
! and put datake into projectors:
    DO N=1,10
       DO NP=1,WGWQ%NGVECTOR
          IF(DATAKE(NP)>DATAKEMIN*DATAKEMAX) THEN
             P_PROJ2(NP,N)= P_PROJ2(NP,N)/DATAKE(NP)
          ELSE
             P_PROJ2(NP,N)= 0.0
             TCOUNT=TCOUNT+1
          ENDIF
       ENDDO
    ENDDO

!   WRITE(*,*) "TCOUNT:",TCOUNT,10*WGWQ%NGVECTOR

    SR_MAT=S_MAT
    
!    write(*,*) "--------"
!    write(*,*) s_mat
!    write(*,*) "--------"

    CALL MATRIX_CSQRT(SR_MAT,10,1,0)
    SRI_MAT=S_MAT
    CALL MATRIX_CSQRT(SRI_MAT,10,1,1)
    SI_MAT=S_MAT
    CALL MATRIX_CSQRT(SI_MAT,10,0,1)
    
    U_MAT=0
    DO N=1,10
       U_MAT(N,N)=CMPLX(1.0_Q,0.0_Q)
    ENDDO
    
    N_MAT=MATMUL(MQ_MAT,SR_MAT)
    N_MAT=MATMUL(SR_MAT,N_MAT)
    N_MAT=N_MAT+U_MAT
    
    CALL MATRIX_CSQRT(N_MAT,10,1,0)
    AA_MAT=MATMUL(SRI_MAT,N_MAT)
    AA_MAT=MATMUL(AA_MAT,SRI_MAT)
    
    AA_MAT=AA_MAT-SI_MAT
    AAH_MAT=AA_MAT

  END SUBROUTINE GW_MULTIPOLE_PROJ_SETUP
 


!************************ SUBROUTINE CSQRT   **************************
!
! deals with a hermition matrix with possibly negative EVs
! FA=1 --> inverse ; FB=1 --> SQRT ; may be combined
!
!***********************************************************************

  SUBROUTINE MATRIX_CSQRT(CUNI, NBANDS,FA,FB)
    USE prec
    IMPLICIT NONE
    INTEGER NBANDS,FA,FB
    ! local
    COMPLEX(q) :: CUNI(:,:),TMPDAT,NEWVAL(NBANDS)
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    COMPLEX(q), ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)
   
    NDIM = SIZE(CUNI,1)
  
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))
!=======================================================================
! diagononalize the matrix
!=======================================================================

    CALL ZHEEV &
         &         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         &           W,  IFAIL)

    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTINV(C): Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    
    DO N1=1,NBANDS
       IF(FB==1) THEN
          TMPDAT=CMPLX(1.0/HFEIG(N1),0.0_Q)
       ELSE
          TMPDAT=CMPLX(HFEIG(N1),0.0_Q)
       ENDIF
       IF(FA==1) THEN
          TMPDAT=SQRT(TMPDAT)
       ENDIF
       
       NEWVAL(N1)=TMPDAT
    ENDDO

    DO N1=1,NBANDS
       DO N2=1,NBANDS
          CTMP(N1,N2)=CUNI(N1,N2)*NEWVAL(N2)
       ENDDO
    ENDDO

    CALL ZGEMM( 'N','C', NBANDS, NBANDS, NBANDS, (1.0_q, 0.0_q), CTMP, &
         &              NBANDS, CUNI(1,1), NDIM, (0.0_q, 0.0_q), CEIDB, NBANDS)

    CUNI=0 
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE MATRIX_CSQRT


! setup routine for multipole correction used in ump2:
  SUBROUTINE FOCK_MULTIPOLE_SETUP_WGW(WGW, LATT_CUR, GRIDHF)
    USE mdipol
    USE lattice
    USE constant
    TYPE (latt) LATT_CUR
    TYPE (wavedes) WGW
    TYPE (grid_3d) :: GRIDHF

    TYPE (wavedes1) WGWQ

!    REAL(q) :: POTFAK(:)                ! 1/(G+dk)**2 (G)
!    local variables
!    GDEF,ALLOCATABLE :: GTMP(:)
!   I think if WGWQ is used we should always declare this as comples
!   so that FFTEXT works properly
    GDEF,ALLOCATABLE :: GTMP(:)

    COMPLEX(q)    :: OV_MAT(10,10)

    INTEGER NC, N1, N2, N3, NG, N, NP
    REAL(q) :: X, Y, Z, R2, TEST_CHARGE, POT, POTD, POTDD, POTE
    REAL(q) :: POTDD2,R2M,NORM
    REAL(q) :: X1, X2, X3, NMAX1, NMAX2, NMAX3, I1, I2, I3
    GDEF    :: M, D1, D2, D3, Q11, Q12, Q13, Q22, Q23, Q33
    REAL(q) :: A(3,3),QF1,QF2,QF3
    REAL(q) :: FACTM,RTMP, XX, XCUTOFF, YCUTOFF, ZCUTOFF
    COMPLEX(q) :: TSUM

! a smooth cutoff function is used for potentials and charge densities
! around the dividing plane
! WIDTH must be at least 1
! 4 grid points around the step function are usually a good choice

    REAL(q), PARAMETER :: WIDTH=4
    
    GDEF    :: TMP_MAT(10,10),PF_MQ(10,10)


!    IF (MCALPHA_DEFAULT) THEN
!       MCALPHA=(WGW%ENMAX/RYTOEV)/20
!       write(*,*)'wgw',WGW%ENMAX
!    ENDIF

    CALL SETWDES(WGW,WGWQ,1) ! for gamma point only
    NP=WGWQ%NGVECTOR

    IF (.NOT. ALLOCATED(P_PROJ2)) THEN
       
       ! NMAX(1-3) is the outermost grid point
       NMAX1=WGWQ%GRID%NGX
       NMAX2=WGWQ%GRID%NGY
       NMAX3=WGWQ%GRID%NGZ
       
       ! X(1-3) is the position of the grid point relative to 
       ! which the multipoles are evaluated
       
       X1   =NINT(0.5+DIP%POSCEN(1)*WGWQ%GRID%NGX)+0.5
       X2   =NINT(0.5+DIP%POSCEN(2)*WGWQ%GRID%NGY)+0.5
       X3   =NINT(0.5+DIP%POSCEN(3)*WGWQ%GRID%NGZ)+0.5
       
       
       A(:,1)=LATT_CUR%A(:,1)
       A(:,2)=LATT_CUR%A(:,2)
       A(:,3)=LATT_CUR%A(:,3)
       
       IF (WGWQ%GRID%RL%NFAST==3) THEN
          ! mpi version:    x-> N2, y-> N3, z-> N1 
          NMAX2=WGWQ%GRID%NGX
          NMAX3=WGWQ%GRID%NGY
          NMAX1=WGWQ%GRID%NGZ
          

!      new: finds closest midpoint between gridpoint, N gridpoints is always even
!      ---> as many points above as below!
          X2   =NINT(0.5+DIP%POSCEN(1)*WGWQ%GRID%NGX)+0.5
          X3   =NINT(0.5+DIP%POSCEN(2)*WGWQ%GRID%NGY)+0.5
          X1   =NINT(0.5+DIP%POSCEN(3)*WGWQ%GRID%NGZ)+0.5
          
          A(:,2)=LATT_CUR%A(:,1)
          A(:,3)=LATT_CUR%A(:,2)
          A(:,1)=LATT_CUR%A(:,3)
       ENDIF



!==========================================================================
! check whether potential corrections have been set up properly yet
! if not yet, set them up
!==========================================================================
!    IF (.NOT. ALLOCATED(POT_MULTIPOLE_CORR_MQ)) THEN

!       ALLOCATE(POT_MULTIPOLE_CORR_MQ(MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV),10))

       ALLOCATE(POT_MULTIPOLE_PROJ(MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV),10))
       
! eventuel m_
       ALLOCATE(GTMP(MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))) ! GRIDHF%MPLWV is the number of
                                       ! complex words required for an FFW work array
                                       ! GTMP real values hence m_ -> 2*
       ALLOCATE(P_PROJ(NP,10)) !MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV),10))
       ALLOCATE(P_PROJ2(NP,10)) !MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV),10))
       ALLOCATE(PROJ_WEIGHT(NP)) !MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV)))
      
       NG=0
       
       DO NC=1,3
          X= A(1,NC)
          Y= A(2,NC)
          Z= A(3,NC)
          RTMP=(X*X+Y*Y+Z*Z)
          R2=(X*X+Y*Y+Z*Z)*MCALPHA*0.25
          TEST_CHARGE=EXP(-R2)
       ENDDO

       TSUM=0

       DO NC=1,GRIDHF%RL%NCOL  ! WGWQ%GRID%RL%NCOL
          N2=GRIDHF%RL%I2(NC) !WGWQ%GRID%RL%I2(NC)
          N3=GRIDHF%RL%I3(NC)! WGWQ%GRID%RL%I3(NC)
          DO N1=1,WGWQ%GRID%RL%NROW
             NG=NG+1

             I1=(MOD(N1-X1+NMAX1*200.5,NMAX1))*(1.0_q/NMAX1)-0.5
             I2=(MOD(N2-X2+NMAX2*200.5,NMAX2))*(1.0_q/NMAX2)-0.5
             I3=(MOD(N3-X3+NMAX3*200.5,NMAX3))*(1.0_q/NMAX3)-0.5

             XX=ABS(ABS(I1*NMAX1)-NMAX1/2)
             XCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                XCUTOFF = 1.0
             ELSE
                XCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I2*NMAX2)-NMAX2/2)
             YCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                YCUTOFF = 1.0
             ELSE
                YCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF
              
             XX=ABS(ABS(I3*NMAX3)-NMAX3/2)
             ZCUTOFF=1.0
             IF (XX > WIDTH)  THEN
                ZCUTOFF = 1.0
             ELSE
                ZCUTOFF= ABS(SIN(PI*XX/WIDTH/2))
             ENDIF

             X= I1*A(1,1)+I2*A(1,2)+I3*A(1,3)
             Y= I1*A(2,1)+I2*A(2,2)+I3*A(2,3)
             Z= I1*A(3,1)+I2*A(3,2)+I3*A(3,3)
             

!            projectors in real space
             
             POT_MULTIPOLE_PROJ(NG,1)  =1.0             
             POT_MULTIPOLE_PROJ(NG,2)  =X! *XCUTOFF         
             POT_MULTIPOLE_PROJ(NG,3)  =Y! *YCUTOFF
             POT_MULTIPOLE_PROJ(NG,4)  =Z! *ZCUTOFF
             POT_MULTIPOLE_PROJ(NG,5)  =X*X
             POT_MULTIPOLE_PROJ(NG,6)  =X*Y
             POT_MULTIPOLE_PROJ(NG,7)  =X*Z
             POT_MULTIPOLE_PROJ(NG,8)  =Y*Y
             POT_MULTIPOLE_PROJ(NG,9)  =Y*Z
             POT_MULTIPOLE_PROJ(NG,10) =Z*Z
         ENDDO
       ENDDO

       NORM=1/( REAL(GRIDHF%NPLWV,KIND=q) )

       DO N=1,10
          GTMP(:)=POT_MULTIPOLE_PROJ(:,N)

          ! on real space grid GTMP (GDEF) is has real storage layout
          ! for gamma only

          ! in place fft real to reciprocal (GTMP now complex storage layout)
          CALL FFTEXT(WGWQ%NGVECTOR,WGWQ%NINDPW(1), &
               GTMP(1), &
               P_PROJ(1,N), WGWQ%GRID,.FALSE.)

          ! multiply by 4 pi e^2/ G^2
          P_PROJ2(:,N)=P_PROJ(:,N)*NORM
          P_PROJ(:,N)=P_PROJ2(:,N)*NORM   
       ENDDO
 

       FACTM=1

       DO N=1,10
          DO NG=1,10
             
             IF (WGWQ%GRID%RC%NFAST==1 .AND. WGWQ%GRID%REAL2CPLX) THEN
                NP=1
                DO NC=1,WGWQ%GRID%RC%NCOL
                   
                   N2= WGWQ%GRID%RC%I2(NC)
                   N3= WGWQ%GRID%RC%I3(NC)

                   DO N1=1,WGWQ%GRID%RC%NROW
                      FACTM=1
                      IF (N1 /= 1 .AND. N1 /= WGWQ%GRID%NGX/2+1) FACTM=2
                      PROJ_WEIGHT(NP)=FACTM
                      
                      NP=NP+1
                   ENDDO
                ENDDO
             ELSE IF (.NOT.WGWQ%GRID%REAL2CPLX) THEN
                PROJ_WEIGHT=FACTM
             ELSE 
                WRITE(0,*) 'internal error in FOCK_MULTIPOLE_CORR: currently this storage layout is not supported'
                STOP
             END IF
          ENDDO
       ENDDO


       DEALLOCATE(GTMP,POT_MULTIPOLE_PROJ)

!       MQ_MAT=MQ_MAT*LATT_CUR%OMEGA/(NMAX1*NMAX2*NMAX3)
!
!       QF1=MCALPHA*SQRT(MCALPHA/PI)/PI
!       QF2=QF1*MCALPHA
!       QF3=QF2*MCALPHA
!
!  monopoles, dipoles, and quadrupoles
!       PF_MQ=0
!       PF_MQ(2,2)=2*QF2
!       PF_MQ(3,3)=2*QF2
!       PF_MQ(4,4)=2*QF2    
!       PF_MQ(6,6)=4*QF3
!       PF_MQ(7,7)=4*QF3
!       PF_MQ(9,9)=4*QF3
!       PF_MQ(1,1)=2.5*QF1
!       PF_MQ(5,5)=2*QF3
!       PF_MQ(8,8)=2*QF3
!       PF_MQ(10,10)=2*QF3
!       PF_MQ(1,5)=-QF2
!       PF_MQ(1,8)=-QF2
!       PF_MQ(1,10)=-QF2
!       PF_MQ(5,1)=-QF2
!       PF_MQ(8,1)=-QF2
!       PF_MQ(10,1)=-QF2

! for monopole only
!       PF_MQ=0
!       PF_MQ(1,1)=1.0_q*QF1
! monopoles, dipoles: above + 
!       PF_MQ(2,2)=2*QF2
!       PF_MQ(3,3)=2*QF2
!       PF_MQ(4,4)=2*QF2        

!       TMP_MAT=MATMUL(MQ_MAT,PF_MQ)
!       MQ_MAT=MATMUL(PF_MQ,TMP_MAT)

    ENDIF
!==========================================================================
! correct potentials
!==========================================================================
    
  END SUBROUTINE FOCK_MULTIPOLE_SETUP_WGW

END MODULE fock_multipole


!********************** SUBROUTINE APPLY_GFAC_MULTIPOLE   ************
!
!  these subroutine multiplies the charge density by the potential kernel
!  (essentially e^2 epsilon_0 / 4 pi (G +k-q)^2)
!  as well as the multipole corrections
!
!***********************************************************************

   SUBROUTINE APPLY_GFAC_MULTIPOLE(GRID, CWORK, POTFAK)

     USE prec
     USE mgrid
     USE fock_multipole
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP,N
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)

     GDEF       :: TSUM(10)
     GDEF       :: POTCORRECTION(10)

     TSUM=0
     DO N=1,10
        DO NP=1,GRID%RC%NP
           TSUM(N)=TSUM(N)+GREAL(CWORK(NP)*CONJG(P_PROJ2(NP,N)*PROJ_WEIGHT(NP)))
        ENDDO
     ENDDO

     POTCORRECTION=MATMUL(MQ_MAT,TSUM)

     DO NP=1,GRID%RC%NP
        CWORK(NP)=POTFAK(NP)*CWORK(NP)
     ENDDO

     DO N=1,10
        DO NP=1,GRID%RC%NP
           CWORK(NP)=CWORK(NP)+P_PROJ2(NP,N)*POTCORRECTION(N)
        ENDDO
     ENDDO

   END SUBROUTINE APPLY_GFAC_MULTIPOLE

!
! uses "square root" of correction as in acfdt
! essentially for testing acfdt routines
!
   SUBROUTINE APPLY_GFAC_MULTIPOLE_SQRT(GRID, CWORK, POTFAK)

     USE prec
     USE fock_multipole
     USE mgrid
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP,N
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)

     COMPLEX(q)       :: TSUM(10)
     COMPLEX(q) :: POTCORRECTION(10)

     TSUM=0
     DO N=1,10
        DO NP=1,GRID%RC%NP
           TSUM(N)=TSUM(N)+GREAL(CWORK(NP)*CONJG(P_PROJ2(NP,N)*PROJ_WEIGHT(NP)))
        ENDDO
     ENDDO

     POTCORRECTION=(MATMUL(AH_MAT,TSUM))

     CWORK(1)=SQRT(POTFAK0)*CWORK(1)
     DO NP=2,GRID%RC%NP
        CWORK(NP)=SQRT(POTFAK(NP))*CWORK(NP)
     ENDDO

     DO N=1,10
        CWORK(1)=CWORK(1)+P_PROJ2(1,N)*POTCORRECTION(N)/SQRT(POTFAK0)
        DO NP=2,GRID%RC%NP
           CWORK(NP)=CWORK(NP)+P_PROJ2(NP,N)*POTCORRECTION(N)/SQRT(POTFAK(NP))
        ENDDO
     ENDDO

     TSUM=0
     DO N=1,10
        TSUM(N)=TSUM(N)+GREAL(CWORK(1)*CONJG(P_PROJ2(1,N)*PROJ_WEIGHT(1)))/SQRT(POTFAK0)
        DO NP=2,GRID%RC%NP
           TSUM(N)=TSUM(N)+GREAL(CWORK(NP)*CONJG(P_PROJ2(NP,N)*PROJ_WEIGHT(NP)))/SQRT(POTFAK(NP))
        ENDDO
     ENDDO

     POTCORRECTION=(MATMUL(AH_MAT,TSUM))

     CWORK(1)=SQRT(POTFAK0)*CWORK(1)
     DO NP=2,GRID%RC%NP
        CWORK(NP)=SQRT(POTFAK(NP))*CWORK(NP)
     ENDDO

     DO N=1,10
        DO NP=1,GRID%RC%NP
           CWORK(NP)=CWORK(NP)+P_PROJ2(NP,N)*POTCORRECTION(N)
        ENDDO
     ENDDO
   END SUBROUTINE APPLY_GFAC_MULTIPOLE_SQRT

!  ... as used in ump2 with multipole correction
   SUBROUTINE APPLY_GFAC_MULTIPOLE_WAVEFUN(WDESQ, CWORK, POTFAK)

     USE fock_multipole
     USE prec
     USE mgrid
     IMPLICIT NONE

     TYPE (wavedes1) WDESQ
     INTEGER NP,N
     REAL(q)     :: POTFAK(WDESQ%NGVECTOR)
     COMPLEX(q)  :: CWORK(WDESQ%NGVECTOR)

     GDEF     :: TSUM(10)

     COMPLEX(q)     :: POTCORRECTION(10)

     TSUM=0
     DO N=1,10
        DO NP=1,WDESQ%NGVECTOR
!           TSUM(N)=TSUM(N)+GREAL(CWORK(NP)*CONJG(P_PROJ(NP,N)*PROJ_WEIGHT(NP)))
           TSUM(N)=TSUM(N)+GREAL(CWORK(NP)*CONJG(P_PROJ2(NP,N)*PROJ_WEIGHT(NP)))
        ENDDO
     ENDDO
     ! now multiply tsum with square 10x10 matrix
     POTCORRECTION=MATMUL(MQ_MAT,TSUM)
     
     ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
     DO NP=1,WDESQ%NGVECTOR
        CWORK(NP)=POTFAK(NP)*CWORK(NP)+P_PROJ2(NP,1)*POTCORRECTION(1)
     ENDDO

     DO N=2,10
        DO NP=1,WDESQ%NGVECTOR
           CWORK(NP)=CWORK(NP)+P_PROJ2(NP,N)*POTCORRECTION(N)
        ENDDO
     ENDDO
   END SUBROUTINE APPLY_GFAC_MULTIPOLE_WAVEFUN

! copies proj vector in setup of multipole corrections
   SUBROUTINE COPY_PROJ(GRID, CSRC, CDEST)
     
     USE prec
     USE mgrid
     IMPLICIT NONE
     
     TYPE (grid_3d) GRID
     INTEGER NP
     COMPLEX(q)  :: CSRC(m_ GRID%MPLWV)
     COMPLEX(q)  :: CDEST(GRID%MPLWV)
          
     DO NP=1,GRID%RC%NP
        CDEST(NP)=CSRC(NP)
     ENDDO
          
   END SUBROUTINE COPY_PROJ

