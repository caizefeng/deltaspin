#include "symbol.inc"
MODULE ACFDT
  USE chi_base
  USE local_field
  USE fock 
  IMPLICIT NONE

  INTEGER, PARAMETER, PRIVATE :: NE=8
  REAL(q), PARAMETER, PRIVATE :: STEP_BETWEEN_FREQUENCIES=1.05
! alternative spanning the same frequency range but with only 4 points
!  INTEGER, PARAMETER, PRIVATE :: NE=4
!  REAL(q) :: STEP_BETWEEN_FREQUENCIES=1.120576983388_q

!testend
  TYPE correlation 
     INTEGER :: NE                                ! number of energy points
     REAL(q), POINTER ::    ENCUTGW(:)            ! selected cutoff for which correlation energy is calculated
     REAL(q), POINTER ::    ENCUTGWSOFT(:)        ! selected cutoff for which correlation energy is calculated
     COMPLEX(q), POINTER :: CORRELATION(:)        ! total correlation energy
     COMPLEX(q), POINTER :: CORRSOSEX(:)          ! total correlation energy
     COMPLEX(q), POINTER :: CORRELATION_LAMBDA(:,:) ! lambda dependency of correlation energy
                                                  ! total correlation energy (lambda dependency)
     COMPLEX(q), POINTER :: CORRELATION_K(:)      ! correlation energy at specific k-point
     COMPLEX(q), POINTER :: CORRSOSEX_K(:)        ! correlation energy at specific k-point
     COMPLEX(q), POINTER :: CORRMP2DIR(:)         ! second order direct correlation energy
     COMPLEX(q), POINTER :: CORRMP2EX(:)          ! second order exchange correlation energy
     COMPLEX(q), POINTER :: CORRMP2DIR_K(:)       ! total correlation energy at specific k-point
     COMPLEX(q), POINTER :: CORRMP2EX_K(:)        ! total correlation energy at specific k-point
  END TYPE correlation

  REAL(q), PRIVATE :: DATAKEMAX                   ! maximum energy used in kernel
  INTEGER :: NLAMBDA = 0

!
! Fermi-wave vector for HEG kernel
! set some default to avoid crashing 
!
  REAL(q), PRIVATE :: KFERMI=2.0_q

CONTAINS

!********************** SUBROUTINE XI_ACFDT_SETUP   *******************
!
! setup the energy cutoffs at which the correlation energy is 
! calculated 
! 
!**********************************************************************

  SUBROUTINE XI_ACFDT_SETUP( COR, ENCUTGW, ENCUTGWSOFT)
    USE constant
    TYPE (correlation), POINTER :: COR
    REAL(q) :: ENCUTGW, ENCUTGWSOFT
    INTEGER :: I

    ALLOCATE(COR)
    COR%NE=NE
    ALLOCATE(COR%ENCUTGW(NE))
    ALLOCATE(COR%ENCUTGWSOFT(NE))
    ALLOCATE(COR%CORRELATION(NE))
    ALLOCATE(COR%CORRSOSEX(NE))
    ALLOCATE(COR%CORRELATION_K(NE))
    ALLOCATE(COR%CORRSOSEX_K(NE))
    ALLOCATE(COR%CORRMP2DIR(NE))
    ALLOCATE(COR%CORRMP2EX(NE))
    ALLOCATE(COR%CORRMP2DIR_K(NE))
    ALLOCATE(COR%CORRMP2EX_K(NE))
    ALLOCATE(COR%CORRELATION_LAMBDA(NE,0:NLAMBDA))

    COR%ENCUTGW(1)    =ENCUTGW
    COR%ENCUTGWSOFT(1)=ENCUTGWSOFT
    DO I=2,NE
       COR%ENCUTGW(I)    =COR%ENCUTGW(I-1)/STEP_BETWEEN_FREQUENCIES
       COR%ENCUTGWSOFT(I)=COR%ENCUTGWSOFT(I-1)/STEP_BETWEEN_FREQUENCIES
       IF (COR%ENCUTGWSOFT(I)<=0) COR%ENCUTGWSOFT(I)=-1
    ENDDO

    COR%CORRELATION_LAMBDA=0
    COR%CORRELATION=0
    COR%CORRSOSEX=0
    COR%CORRELATION_K=0
    COR%CORRSOSEX_K=0
    COR%CORRMP2DIR=0
    COR%CORRMP2EX=0
    COR%CORRMP2DIR_K=0
    COR%CORRMP2EX_K=0
    
  END SUBROUTINE XI_ACFDT_SETUP


  SUBROUTINE XI_ACFDT_DEALLOCATE( COR )
    TYPE (correlation), POINTER :: COR

    DEALLOCATE(COR%ENCUTGW)
    DEALLOCATE(COR%CORRELATION)
    DEALLOCATE(COR%CORRSOSEX)
    DEALLOCATE(COR%CORRELATION_K)
    DEALLOCATE(COR%CORRSOSEX_K)
    DEALLOCATE(COR%CORRMP2DIR)
    DEALLOCATE(COR%CORRMP2EX)
    DEALLOCATE(COR%CORRMP2DIR_K)
    DEALLOCATE(COR%CORRMP2EX_K)
    DEALLOCATE(COR%CORRELATION_LAMBDA)
    DEALLOCATE(COR)
    NULLIFY(COR)

  END SUBROUTINE XI_ACFDT_DEALLOCATE

!********************** SUBROUTINE XI_ACFDT_ALL_RPA *******************
!
! determine the RPA correlation energy
! this subroutine integrates the entire calculation in one
! routine to conserve memory and speed things up
! 
!**********************************************************************

  SUBROUTINE XI_ACFDT_ALL_RPA( IU0, CHI, WGWQ, LATT_CUR, OMEGAWEIGHT, COR, NQ , IDIR_MAX, & 
      LRSRPA, L_SUM_KINTER, LADDER, LINVXI, TBSE)
    USE constant
    USE wave 
    USE kpoints_change
    USE full_kpoints

    IMPLICIT NONE
    INTEGER :: IU0
    TYPE (responsefunction) CHI
    TYPE (wavedes1) WGWQ
    TYPE (latt) LATT_CUR
    TYPE (correlation) :: COR
    COMPLEX(q) :: CORRELATION_K, CORRMP2DIR_K
    REAL(q) :: OMEGAWEIGHT(:)
    INTEGER :: NQ            ! total number of q-points used
    INTEGER :: IDIR_MAX      ! maximum number of independent directions
    LOGICAL :: LRSRPA        ! range separated RPA (total - short range)
    LOGICAL :: L_SUM_KINTER 
    LOGICAL :: LADDER
    LOGICAL :: LINVXI        ! .TRUE.:  TBSE = X^-1 GG v GG  X^-1, .FALSE.: TBSE = GG v GG
    TYPE (responsefunction) TBSE
  ! local
    GDEF, ALLOCATABLE :: CHI_WORK(:,:)
    INTEGER :: NOMEGA, IDIR, NP, NP_NEW, I, ILAMBDA
    COMPLEX(q) :: SUM, SUMMP2, SUMSOSEX, SUMMP2EX, SUM_sr, SUMMP2_sr
    REAL(q) :: LAMBDA

    ALLOCATE(CHI_WORK(CHI%NP2, CHI%NP2))

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    IF (CHI%LREAL .AND. .NOT. CHI%LREALSTORE) THEN
       WRITE(0,*) 'internal error in XI_ACFDT_ALL_RPA: for the Gamma point version CHI%LREALSTORE must be set'
       STOP
    ENDIF

    DO ILAMBDA=0,NLAMBDA
    LAMBDA=1.0_q*(ILAMBDA+1)/MAX(1,NLAMBDA)

    COR%CORRELATION_K=0
    COR%CORRMP2DIR_K=0
    COR%CORRSOSEX_K=0
    COR%CORRMP2DIR_K=0
    COR%CORRMP2EX_K=0


! loop over frequency grid
    DO NOMEGA=1,CHI%NOMEGA
       CALL GWPROGRESS(IU0, NOMEGA, CHI%NOMEGA, 1, 1)

       IF (CHI%LGAMMA) THEN
          DO I=1,NE
             SUM=0
             SUMMP2=0
             SUMSOSEX=0
             SUMMP2EX=0
             SUM_sr=0
             SUMMP2_sr=0
             DO IDIR=1,IDIR_MAX
                IF (ASSOCIATED(TBSE%RESPONSEFUN)) THEN 
                   CALL BODY_FROM_WING( TBSE, IDIR, 1)

                   ! tricky the f_xc is so non isotropic so that we need to store it
                   ! fully for x, y and z direction in components 2, 3 and 4
                   IF (SIZE(TBSE%RESPONSEFUN,3)==4 .AND. IDIR_MAX==3) THEN
                      TBSE%RESPONSEFUN(:,:,1) =TBSE%RESPONSEFUN (:,:,IDIR+1)
                   ENDIF
                ENDIF

                CALL BODY_FROM_WING( CHI, IDIR, NOMEGA)
                IF (.NOT. LADDER) THEN
                   CALL XI_LOCAL_FIELD_ACFDT( CHI_WORK, CHI, LATT_CUR, & 
                        COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .FALSE.)
                   CALL ROTLN_TRACE( CHI_WORK, NP_NEW, CHI%LREAL, SUM, SUMMP2, LAMBDA, -1)
                ELSE
!                   CALL XI_LOCAL_FIELD_CCD( CHI_WORK, CHI, LATT_CUR, & 
                   CALL XI_LOCAL_FIELD_ACFDT_FX( CHI_WORK, CHI, LATT_CUR, & 
                        COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .FALSE., & 
                        CHI%LREAL, SUM, SUMMP2, SUMSOSEX, SUMMP2EX, LAMBDA, TBSE, LINVXI, -1)
                ENDIF
             ENDDO
             ! if we want range separated RPA (total - short)
             ! we calculate the short range contribution here
             IF (LRSRPA) THEN
                DO IDIR=1,IDIR_MAX
                   CALL BODY_FROM_WING( CHI, IDIR, NOMEGA)
                   CALL XI_LOCAL_FIELD_ACFDT( CHI_WORK, CHI, LATT_CUR, & 
                        COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .TRUE.)
                   CALL ROTLN_TRACE( CHI_WORK, NP_NEW, CHI%LREAL, SUM_sr, SUMMP2_sr, LAMBDA, -1)
                ENDDO
             ENDIF
             SUM=(SUM-SUM_sr)/IDIR_MAX
             SUMMP2=(SUMMP2-SUMMP2_sr)/IDIR_MAX
             SUMSOSEX=SUMSOSEX/IDIR_MAX
             SUMMP2EX=SUMMP2EX/IDIR_MAX
             COR%CORRELATION_K(I)=COR%CORRELATION_K(I)+SUM/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRMP2DIR_K(I)=COR%CORRMP2DIR_K(I)+SUMMP2/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRSOSEX_K(I)=COR%CORRSOSEX_K(I)+SUMSOSEX/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRMP2EX_K(I)=COR%CORRMP2EX_K(I)+SUMMP2EX/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
          ENDDO
       ELSE
          DO I=1,NE
             SUM=0
             SUMMP2=0
             SUMSOSEX=0
             SUMMP2EX=0
             SUM_sr=0
             SUMMP2_sr=0
             IF (.NOT. LADDER) THEN
                CALL XI_LOCAL_FIELD_ACFDT( CHI_WORK, CHI , LATT_CUR, & 
                     COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .FALSE.)
                CALL ROTLN_TRACE( CHI_WORK, NP_NEW, .FALSE., SUM, SUMMP2, LAMBDA, -1)
             ELSE
!                CALL XI_LOCAL_FIELD_CCD( CHI_WORK, CHI , LATT_CUR, & 
                CALL XI_LOCAL_FIELD_ACFDT_FX( CHI_WORK, CHI , LATT_CUR, & 
                     COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .FALSE., &
                     CHI%LREAL, SUM, SUMMP2, SUMSOSEX, SUMMP2EX, LAMBDA, TBSE, LINVXI, -1)
             ENDIF
             ! if we want range separated RPA (total - short)
             ! we calculate the short range contribution here
             IF (LRSRPA) THEN
                CALL XI_LOCAL_FIELD_ACFDT( CHI_WORK, CHI , LATT_CUR, & 
                     COR%ENCUTGW(I), COR%ENCUTGWSOFT(I), NOMEGA , WGWQ , NP_NEW, .TRUE.)
                CALL ROTLN_TRACE( CHI_WORK, NP_NEW, .FALSE., SUM_sr, SUMMP2_sr, LAMBDA, -1)
             ENDIF
             SUM=SUM-SUM_sr
             SUMMP2=SUMMP2-SUMMP2_sr
             COR%CORRELATION_K(I)=COR%CORRELATION_K(I)+SUM/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRMP2DIR_K(I)=COR%CORRMP2DIR_K(I)+SUMMP2/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRSOSEX_K(I)=COR%CORRSOSEX_K(I)+SUMSOSEX/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
             COR%CORRMP2EX_K(I)=COR%CORRMP2EX_K(I)+SUMMP2EX/2/PI*KPOINTS_ORIG%WTKPT(NQ) &
                  *OMEGAWEIGHT(NOMEGA+CHI%NOMEGA_LOW-1)
          ENDDO
       END IF
    ENDDO

    CALLMPI( M_sum_d(WGWQ%COMM_INTER, COR%CORRELATION_K, 2*NE))
    CALLMPI( M_sum_d(WGWQ%COMM_INTER, COR%CORRMP2DIR_K, 2*NE))
    CALLMPI( M_sum_d(WGWQ%COMM_INTER, COR%CORRSOSEX_K, 2*NE))
    CALLMPI( M_sum_d(WGWQ%COMM_INTER, COR%CORRMP2EX_K, 2*NE))

    IF ( L_SUM_KINTER) THEN
      CALLMPI( M_sum_z(WGWQ%COMM_KINTER, COR%CORRELATION_K, NE))
      CALLMPI( M_sum_z(WGWQ%COMM_KINTER, COR%CORRMP2DIR_K, NE))
      CALLMPI( M_sum_z(WGWQ%COMM_KINTER, COR%CORRSOSEX_K, NE))
      CALLMPI( M_sum_z(WGWQ%COMM_KINTER, COR%CORRMP2EX_K, NE))
    ENDIF
!
! ok possibly the k-point mesh is downsampled
! have to take care of that now
    COR%CORRELATION_K=COR%CORRELATION_K   *NKREDX*NKREDY*NKREDZ
    COR%CORRMP2DIR_K =COR%CORRMP2DIR_K*NKREDX*NKREDY*NKREDZ
    COR%CORRSOSEX_K  =COR%CORRSOSEX_K   *NKREDX*NKREDY*NKREDZ
    COR%CORRMP2EX_K  =COR%CORRMP2EX_K*NKREDX*NKREDY*NKREDZ
    IF (ODDONLY .OR. EVENONLY) COR%CORRELATION_K=COR%CORRELATION_K*2
    IF (ODDONLY .OR. EVENONLY) COR%CORRMP2DIR_K =COR%CORRMP2DIR_K*2
    IF (ODDONLY .OR. EVENONLY) COR%CORRSOSEX_K=COR%CORRSOSEX_K*2
    IF (ODDONLY .OR. EVENONLY) COR%CORRMP2EX_K=COR%CORRMP2EX_K*2

    COR%CORRELATION_LAMBDA(:,ILAMBDA)  =COR%CORRELATION_LAMBDA(:,ILAMBDA)   +COR%CORRELATION_K
    ENDDO

    COR%CORRELATION=COR%CORRELATION   +COR%CORRELATION_K
    COR%CORRMP2DIR =COR%CORRMP2DIR    +COR%CORRMP2DIR_K
    COR%CORRSOSEX  =COR%CORRSOSEX     +COR%CORRSOSEX_K
    COR%CORRMP2EX  =COR%CORRMP2EX    +COR%CORRMP2EX_K

    DEALLOCATE(CHI_WORK)

  END SUBROUTINE XI_ACFDT_ALL_RPA


!************************ SUBROUTINE ROTLN_TRACE ***********************
!
! this subroutine diagonalizes v Xi and calculates the trace of 
!  Log(1-e)+e
! where e are the eigenvalues of v Xi
!
! the second order expansion is A*A and should correspond exactly
! to the Coulomb term in the MP2 energy
!
!***********************************************************************

  SUBROUTINE ROTLN_TRACE(CUNI, NBANDS, LREAL, SUM, SUMMP2, LAMBDA, IU6 )
    GDEF       :: CUNI(:,:)
    INTEGER    :: NBANDS
    COMPLEX(q) :: SUM, SUMMP2
    LOGICAL    :: LREAL
    REAL(q)    :: LAMBDA
    INTEGER    :: IU6
  ! local
    INTEGER, PARAMETER :: LWORK=32
    GDEF    ::CWRK(LWORK*NBANDS)
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: IFAIL, NDIM, N
    NDIM = SIZE(CUNI,1)
!=======================================================================
! diagononalize the matrix to get eigenvalues (N=eigenv, V=eigenvectors)
!=======================================================================
    IF (LREAL) THEN
#ifndef gammareal
       WRITE(0,*) 'internal error in ROTLN_TRACE: LREAL is set, but complex code'
       STOP
#endif
       CALL DSYEV &
            ('N','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
            IFAIL)
    ELSE
#ifdef gammareal
       WRITE(0,*) 'internal error in ROTLN_TRACE: LREAL is not set, but gamma only code'
       STOP
#endif
       CALL ZHEEV &
         ('N','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
         W,  IFAIL)
    ENDIF

    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTLN_TRACE: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    IF (IU6>=0) WRITE(IU6,'(10F10.6)') HFEIG
!=======================================================================
! sum over eigenvalues
!=======================================================================
    DO N=1,NBANDS
       IF (1-HFEIG(N)<1E-8) THEN
          WRITE(*,*) 'WARNING in ROTLN_TRACE: eigenvalues are negative or close to zero', HFEIG(N)
       ELSE
          SUM=SUM+((LOG(1-HFEIG(N)*LAMBDA)+HFEIG(N)*LAMBDA))
          SUMMP2=SUMMP2+(-HFEIG(N)*HFEIG(N))/2
       ENDIF
    ENDDO
  END SUBROUTINE ROTLN_TRACE


!************************ SUBROUTINE ROTLN    **************************
!
! this subroutine calculate the Log(1-A)+A
! old version no longer used
! several test cases are implemented as well
! the second order expansion is A*A and should correspond exactly
! to the Coulomb term in the MP2 energy
!
! TODO: the routine should be rewritten to conserve memory
!
!***********************************************************************

  SUBROUTINE ROTLN(CUNI, NBANDS, LREAL, LAMBDA, IMP2, TRACE, IU6 )
    INTEGER NBANDS
    GDEF       :: CUNI(:,:)
    LOGICAL    :: LREAL
    REAL(q)    :: LAMBDA
    INTEGER    :: IMP2   ! IMP2=0 use RPA, IMP2=1 yield second order MP2 term
    COMPLEX(q) :: TRACE
    INTEGER    :: IU6
    ! local
    INTEGER, PARAMETER :: LWORK=32
    COMPLEX(q) ::CWRK(LWORK*NBANDS)
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: IFAIL, NDIM, N1, N2

    GDEF, ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)

    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))

!=======================================================================
! diagononalize the matrix to get eigenvalues (N=eigenv, V=eigenvectors)
!=======================================================================
    IF (LREAL) THEN
#ifndef gammareal
       WRITE(0,*) 'internal error in ROTLN: LREAL is set, but complex code'
       STOP
#endif
       CALL DSYEV &
            ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
            IFAIL)
    ELSE
#ifdef gammareal
       WRITE(0,*) 'internal error in ROTLN: LREAL is not set, but gamma only code'
       STOP
#endif
       CALL ZHEEV &
         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
         W,  IFAIL)
    ENDIF
!=======================================================================
! diagononalize the matrix
!=======================================================================
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTLN: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    IF (IU6>=0) WRITE(IU6,'(10F10.6)') HFEIG
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================

    DO N2=1,NBANDS
       IF (1-HFEIG(N2)<1E-8) THEN
          CTMP(:,N2)=0
          WRITE(*,*) 'WARNING in ROTLN: eigenvalues are negative or close to zero', CTMP
       ELSE
          IF (IMP2==0) THEN
             ! log (1-A) +A
             TRACE=TRACE+((LOG(1-HFEIG(N2)*LAMBDA)+HFEIG(N2)*LAMBDA))
             CTMP(1:NBANDS,N2)=CUNI(1:NBANDS,N2)*(LOG(1-HFEIG(N2)*LAMBDA)+HFEIG(N2)*LAMBDA)/MAX(HFEIG(N2)*HFEIG(N2),1E-32_q)
!             CTMP(1:NBANDS,N2)=CUNI(1:NBANDS,N2)*(-0.5_q)/(1.0_q-HFEIG(N2))
          ELSE
             TRACE=TRACE+(-HFEIG(N2)*HFEIG(N2))/2
             ! A*A
             CTMP(1:NBANDS,N2)=CUNI(1:NBANDS,N2)*(-0.5_q)
          ENDIF
       ENDIF
    ENDDO
    CALL GGEMM( 'N', trans, NBANDS, NBANDS, NBANDS, one, CTMP, &
         &             NBANDS, CUNI(1,1), NDIM, zero, CEIDB, NBANDS)
   
    CUNI=0 
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE ROTLN

!******************** SUBROUTINE XI_LOCAL_FIELD_ACFDT ******************
!
! this subroutine truncates the response function by multiplying
! with a truncated  Coulomb kernel smoothly going from 1/G^2 to zero
! between ENCUTGWSOFT and ENCUTGW
!
!***********************************************************************


  SUBROUTINE XI_LOCAL_FIELD_ACFDT( CHI_WORK, CHI, LATT_CUR, & 
       ENCUT, ENCUTSOFT, NOMEGA, WGWQ , MAXINDEX, LSR)
    USE constant
    USE fock
    USE c2f_interface, ONLY : ERRF
    IMPLICIT NONE
    GDEF :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI
    TYPE (latt) LATT_CUR
    REAL(q) :: ENCUT, ENCUTSOFT
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
    INTEGER :: MAXINDEX              ! number of grid points using truncated Coulomb kernel
    LOGICAL :: LSR
  ! local
    INTEGER    NI, NP, NI_
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE, POTFAK, E
    REAL(q), ALLOCATABLE :: DATAKE(:)
    INTEGER, ALLOCATABLE :: OLD_INDEX(:)
    REAL(q) :: DFUN, SFUN

!  neu

    REAL(q) :: OMEGBK,QC
    REAL(q) :: GAMMASCALE

! multipole correction
    INTEGER    NJ,NK
    COMPLEX(q) :: TSUM(10)
    COMPLEX(q) :: POTCORRECTION(10)
    REAL(q) :: Q1, Q2

    IF (ENCUTSOFT>=0) THEN
       Q1=SQRT(ENCUTSOFT/HSQDTM)
       Q2=SQRT(ENCUT/HSQDTM)
    ENDIF

    ! e^2/ volume 
    SCALE=EDEPS/LATT_CUR%OMEGA
!=======================================================================
! first set up the truncated Coulomb kernel
! smoothly going from 1/G^2 to zero between ENCUTGWSOFT and ENCUTGW
!=======================================================================
    DKX=(WGWQ%VKPT(1))*LATT_CUR%B(1,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(1,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(WGWQ%VKPT(1))*LATT_CUR%B(2,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(2,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(WGWQ%VKPT(1))*LATT_CUR%B(3,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(3,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(3,3)

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2
    ALLOCATE( DATAKE(NP), OLD_INDEX(NP))

    DATAKEMAX=0.0
    MAXINDEX=0
    DO NI=1,NP
       NI_=NI
       IF (WGWQ%LGAMMA) NI_=(NI-1)/2+1
       
       GX=(WGWQ%IGX(NI_)*LATT_CUR%B(1,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WGWQ%IGX(NI_)*LATT_CUR%B(2,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WGWQ%IGX(NI_)*LATT_CUR%B(3,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
       

       IF (ABS(GSQU)<G2ZERO) THEN
          ! head and wing
          POTFAK=SCALE
          ! if HFRCUT is set, a fixed spherical cutoff is used
          ! resulting in a finite values at G=0
          ! since the head stores the q^2, the final contribution is zero
          ! from the head
          IF (HFRCUT/=0) THEN
             POTFAK=0
          ! if LRHFCALC.AND.LRSCOR the Coulomb kernel is replaced by v_{LR}=v*exp(-q^2/(4*mu^2))
          ! for q=0  v_{LR}=1, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{LR}(q=0) is set to average of v_{LR} in a sphere  
          ! with volume corresponding to BZ volume/ # q-points       
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*GAMMASCALE
          ! if LSR the Coulomb kernel is replaced by v_{SR}=v*[1-exp(-q^2/(4*mu^2))]
          ! for q=0  v_{SR}=0, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{SR}(q=0) is set to average of v_{SR} in a sphere
          ! with volume corresponding to BZ volume/ # q-points
          ELSE IF (LSR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*(1-GAMMASCALE)
          ENDIF

          ! switch off standard convergence correction
          IF(MCALPHA/=0) THEN
             POTFAK=0
          ENDIF

       ELSE
          ! the factor 1/(2 pi)^2 is required to obtain proper reciprocal
          ! lattice vector lenght
          POTFAK=SCALE/(GSQU*TPI**2)
          IF (HFRCUT/=0) THEN
             ! spherical cutoff on Coloumb kernel
             ! see for instance C.A. Rozzi, PRB 73, 205119 (2006)
!             POTFAK=POTFAK*(1-COS(SQRT(GSQU)*TPI*HFRCUT)*EXP(-(SQRT(GSQU)*TPI*HFRCUT)**2*HFRCUT_SMOOTH))
!test
             CALL DELSTP(3,SQRT(GSQU)*TPI*HFRCUT/10,DFUN,SFUN)
             POTFAK=POTFAK*(SFUN-0.5)*2
!test
          ELSE IF (LSR) THEN
             ! use the short range part of the Coulomb kernel
             POTFAK=POTFAK*(1._q-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             ! use the long range part of the Coulomb kernel
             POTFAK=POTFAK*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
          ENDIF
       ENDIF

       ! smooth cutoff function between  ENCUTSOFT and ENCUT
       E=HSQDTM*(GSQU*TPI**2)
       IF (ENCUT>=0 .AND. E>ENCUT) THEN
          POTFAK=0
       ELSE

          MAXINDEX=MAXINDEX+1
          OLD_INDEX(MAXINDEX)=NI
          
          IF (ENCUTSOFT>=0 .AND. E>ENCUTSOFT) THEN
             POTFAK=POTFAK*(1+COS((E-ENCUTSOFT)/(ENCUT-ENCUTSOFT)*PI))/2
!            test attenuated kernel of Merzuk
!             POTFAK=POTFAK*ATTENUATE_CUTOFF_SMOOTH_ACFDT(SQRT(GSQU)*TPI, Q1, Q2 )
          ENDIF
       ENDIF

! POTFAK in fock has an additional factor. Since the corrections were originally devised
! for the fock routine this ensures consistency
       IF(MCALPHA/=0) THEN
          POTFAK=POTFAK*(1.0_Q/WGWQ%GRID%NPLWV)
       ENDIF

       DATAKE(NI)=SQRT(POTFAK)
       ! maximum kinetic energy
       IF(DATAKE(NI)>DATAKEMAX) THEN
          DATAKEMAX=DATAKE(NI)
       ENDIF

    ENDDO

    ! setup of multipole corrections
    IF (MCALPHA/=0) THEN
       CALL FOCK_MULTIPOLE_CORR_SETUP(LATT_CUR, GRIDHF)
       CALL GW_MULTIPOLE_PROJ_SETUP( WGWQ, LATT_CUR, DATAKE,DATAKEMAX )
    ENDIF

!=======================================================================
! now multiply the response function with the kernel
! from left and right hand side
!=======================================================================
    IF (CHI%LREALSTORE) THEN
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF

    DO NI=1,NP
       ! get v^ 1/2
       POTFAK=DATAKE(NI)
       CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)*POTFAK
       CHI_WORK(NI,1:NP)=CHI_WORK(NI,1:NP)*POTFAK
    ENDDO

    ! multipole corrections:
    IF(MCALPHA/=0) THEN
       DO NI=1,NP
          
          TSUM=0.0
          DO NK=1,10
             DO NJ=1,NP
                TSUM(NK)=TSUM(NK)+CHI_WORK(NI,NJ)*CONJG(P_PROJ2(NJ,NK))                
             ENDDO
          ENDDO

          POTCORRECTION=(MATMUL(AAH_MAT,TSUM))

          DO NK=1,10
             DO NJ=1,NP
                CHI_WORK(NI,NJ)=CHI_WORK(NI,NJ)+P_PROJ2(NJ,NK)*POTCORRECTION(NK)
             ENDDO
          ENDDO
          
       ENDDO
       
       
       DO NI=1,NP
          
          TSUM=0.0
          DO NK=1,10
             DO NJ=1,NP
                TSUM(NK)=TSUM(NK)+CHI_WORK(NJ,NI)*P_PROJ2(NJ,NK)
             ENDDO
          ENDDO

          POTCORRECTION=(MATMUL(AAH_MAT,TSUM))

          DO NK=1,10
             DO NJ=1,NP
                CHI_WORK(NJ,NI)=CHI_WORK(NJ,NI)+CONJG(P_PROJ2(NJ,NK))*POTCORRECTION(NK)
             ENDDO
          ENDDO
          
       ENDDO
! When using MCALPHA POTFAK had an additional normalisation factor to ensure consistency with fock.F
! acfdt does not need this extra factor so we have to take it out again:
       CHI_WORK=CHI_WORK*(WGWQ%GRID%NPLWV)
    ENDIF

    DO NI=1,MAXINDEX
       ! get v^ 1/2
       CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,OLD_INDEX(NI))
    ENDDO

    DO NI=1,MAXINDEX
       ! get v^ 1/2
       CHI_WORK(NI,1:MAXINDEX)=CHI_WORK(OLD_INDEX(NI),1:MAXINDEX)
    ENDDO

    DEALLOCATE(DATAKE, OLD_INDEX)

  END SUBROUTINE XI_LOCAL_FIELD_ACFDT


!
! increase Coloumb potential and than bring it back to 0 smoothly
! by Merzuk Kaltak
!
  FUNCTION ATTENUATE_CUTOFF_SMOOTH_ACFDT( X, Q1, Q2)
    USE prec
    IMPLICIT NONE

    REAL(q) :: ATTENUATE_CUTOFF_SMOOTH_ACFDT, X, Q1, Q2

    ATTENUATE_CUTOFF_SMOOTH_ACFDT= ((Q1 - Q2)*(X - Q2))/(Q1**2 + X*(-2*Q1 + Q2))**2 *X**2
        
  END FUNCTION ATTENUATE_CUTOFF_SMOOTH_ACFDT


!******************** SUBROUTINE XI_LOCAL_FIELD_ACFDT_FX ***************
!
! this subroutine truncates the response function by multiplying
! with a truncated  Coulomb kernel smoothly going from 1/G^2 to zero
! between ENCUTGWSOFT and ENCUTGW
! and
!
!***********************************************************************


  SUBROUTINE XI_LOCAL_FIELD_ACFDT_FX( CHI_WORK, CHI, LATT_CUR, & 
       ENCUT, ENCUTSOFT, NOMEGA, WGWQ , MAXINDEX, LSR, &
       LREAL, SUM, SUMMP2,  SUMSOSEX, SUMMP2EX, LAMBDA, TBSE, LINVXI, IU6 )
    USE constant
    USE fock
    USE c2f_interface, ONLY : ERRF
    IMPLICIT NONE
    GDEF :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI   ! response function
    TYPE (responsefunction) :: TBSE  ! exchange kernel 
    TYPE (latt) LATT_CUR
    REAL(q) :: ENCUT, ENCUTSOFT
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
    INTEGER :: MAXINDEX              ! number of grid points using truncated Coulomb kernel
    LOGICAL :: LSR
    COMPLEX(q) :: SUM, SUMMP2        ! correlation energy and second order approximation to it
    COMPLEX(q) :: SUMSOSEX, SUMMP2EX ! exchange correlation energy and second order approximation to it
    LOGICAL    :: LREAL              ! real response function
    REAL(q)    :: LAMBDA             ! coupling strenght parameter (usually 1)
    LOGICAL    :: LINVXI             ! .TRUE.:  TBSE = X^-1 GG v GG  X^-1, .FALSE.: TBSE = GG v GG
    INTEGER    :: IU6                ! unit for output
  ! local
    INTEGER    NI, NP, NI_
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE, SCALEFX, POTFAK, E, ETA
    REAL(q), ALLOCATABLE :: VG(:), FXG(:)
    INTEGER, ALLOCATABLE :: OLD_INDEX(:)
    REAL(q) :: DFUN, SFUN
    COMPLEX(q) :: TRACE, TRACEX, TRACEX2
    GDEF, ALLOCATABLE :: CHI2(:,:), CHI3(:,:)
    REAL(q) :: OMEGBK,QC
    REAL(q) :: GAMMASCALE

! multipole correction
    INTEGER    NJ, NK, IMP2
    COMPLEX(q) :: TSUM(10)
    COMPLEX(q) :: POTCORRECTION(10)

    INTEGER :: NOMEGA_GLOBAL


    ! e^2/ volume 
    SCALE=EDEPS/LATT_CUR%OMEGA

    SCALEFX=FELECT/LATT_CUR%OMEGA/2

!=======================================================================
! first set up the truncated Coulomb kernel
! smoothly going from 1/G^2 to zero between ENCUTGWSOFT and ENCUTGW
!=======================================================================
    DKX=(WGWQ%VKPT(1))*LATT_CUR%B(1,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(1,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(WGWQ%VKPT(1))*LATT_CUR%B(2,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(2,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(WGWQ%VKPT(1))*LATT_CUR%B(3,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(3,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(3,3)

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2
    ALLOCATE( VG(NP), OLD_INDEX(NP), FXG(NP))

    DATAKEMAX=0.0
    MAXINDEX=0
    DO NI=1,NP
       NI_=NI
       IF (WGWQ%LGAMMA) NI_=(NI-1)/2+1
       
       GX=(WGWQ%IGX(NI_)*LATT_CUR%B(1,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WGWQ%IGX(NI_)*LATT_CUR%B(2,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WGWQ%IGX(NI_)*LATT_CUR%B(3,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
       

       IF (ABS(GSQU)<G2ZERO) THEN
          ! head and wing
          POTFAK=SCALE
          ! if HFRCUT is set, a fixed spherical cutoff is used
          ! resulting in a finite values at G=0
          ! since the head stores the q^2, the final contribution is zero
          ! from the head
          IF (HFRCUT/=0) THEN
             POTFAK=0
          ! if LRHFCALC.AND.LRSCOR the Coulomb kernel is replaced by v_{LR}=v*exp(-q^2/(4*mu^2))
          ! for q=0  v_{LR}=1, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{LR}(q=0) is set to average of v_{LR} in a sphere  
          ! with volume corresponding to BZ volume/ # q-points       
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*GAMMASCALE
          ! if LSR the Coulomb kernel is replaced by v_{SR}=v*[1-exp(-q^2/(4*mu^2))]
          ! for q=0  v_{SR}=0, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{SR}(q=0) is set to average of v_{SR} in a sphere
          ! with volume corresponding to BZ volume/ # q-points
          ELSE IF (LSR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*(1-GAMMASCALE)
          ENDIF

       ELSE
          ! the factor 1/(2 pi)^2 is required to obtain proper reciprocal
          ! lattice vector lenght
          POTFAK=SCALE/(GSQU*TPI**2)
          IF (HFRCUT/=0) THEN
             ! spherical cutoff on Coloumb kernel
             ! see for instance C.A. Rozzi, PRB 73, 205119 (2006)
!             POTFAK=POTFAK*(1-COS(SQRT(GSQU)*TPI*HFRCUT)*EXP(-(SQRT(GSQU)*TPI*HFRCUT)**2*HFRCUT_SMOOTH))
!test
             CALL DELSTP(3,SQRT(GSQU)*TPI*HFRCUT/10,DFUN,SFUN)
             POTFAK=POTFAK*(SFUN-0.5)*2
!test
          ELSE IF (LSR) THEN
             ! use the short range part of the Coulomb kernel
             POTFAK=POTFAK*(1._q-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             ! use the long range part of the Coulomb kernel
             POTFAK=POTFAK*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
          ENDIF
       ENDIF

       ! smooth cutoff function between  ENCUTSOFT and ENCUT
       E=HSQDTM*(GSQU*TPI**2)
       IF (ENCUT>=0 .AND. E>ENCUT) THEN
          POTFAK=0
       ELSE

          MAXINDEX=MAXINDEX+1
          OLD_INDEX(MAXINDEX)=NI
          
          IF (ENCUTSOFT>=0 .AND. E>ENCUTSOFT) THEN
             POTFAK=POTFAK*(1+COS((E-ENCUTSOFT)/(ENCUT-ENCUTSOFT)*PI))/2
          ENDIF
       ENDIF


       VG(NI)=SQRT(POTFAK)
       ! maximum value in Coloumb kernel
       IF(VG(NI)>DATAKEMAX) THEN
          DATAKEMAX=VG(NI)
       ENDIF

       ETA=SQRT(GSQU)*TPI/(2*KFERMI)
       
       IF (ABS(GSQU)<G2ZERO) THEN
          FXG(NI)=-SCALEFX *9*PI/KFERMI**2
       ELSE
          FXG(NI)=-SCALEFX *3*PI/(5*KFERMI**2)*( &
               (2/ETA-10*ETA)*LOG((1+ETA)/ABS(1-ETA))+ &
               (2*ETA**4-10*ETA**2)*LOG((1+ETA)*ABS(1-ETA)/ETA**2)+ &
               2*ETA**2+11)
       ENDIF
       
    ENDDO
!=======================================================================
! now multiply the response function with the kernel
! from left and right hand side
!=======================================================================

! IMP2=0 calculate RPA+SOSEX contribution, IMP2=1 MP2 contributions
  mp2rpa: DO IMP2=0, 1

    IF (CHI%LREALSTORE) THEN
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF

    DO NI=1,NP
       ! left and right multiply with v^ 1/2
       POTFAK=VG(NI)
       CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)*POTFAK
       CHI_WORK(NI,1:NP)=CHI_WORK(NI,1:NP)*POTFAK
    ENDDO

    ! reduce to relevant indices for this cutoff
    CALL XI_LOCAL_FIELD_COMPRESS(CHI_WORK, MAXINDEX, OLD_INDEX)

    ! calculate (ln (1-x) +x )/ x^2 of the matrix
    TRACE=0
    CALL ROTLN( CHI_WORK, MAXINDEX, LREAL, LAMBDA, IMP2, TRACE, -1)

    ! multiply by v^ 1/2 from left and right
    DO NI=1,MAXINDEX
       POTFAK=VG(OLD_INDEX(NI))
       CHI_WORK(1:MAXINDEX,NI)=CHI_WORK(1:MAXINDEX,NI)*POTFAK
       CHI_WORK(NI,1:MAXINDEX)=CHI_WORK(NI,1:MAXINDEX)*POTFAK
    ENDDO

    ! left an right multiply by response matrix

    ! temporary work space allocation (not nice but safes memory)
    ALLOCATE(CHI2(SIZE(CHI_WORK,1),SIZE(CHI_WORK,2)), CHI3(SIZE(CHI_WORK,1),SIZE(CHI_WORK,2)))
    
    IF (CHI%LREALSTORE) THEN
       CHI2(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI2(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF
    
!    CALL XI_LOCAL_FIELD_COMPRESS(CHI2, MAXINDEX, OLD_INDEX)
    CALL XI_LOCAL_FIELD_UNCOMPRESS(CHI_WORK, MAXINDEX, OLD_INDEX)

    IF (LINVXI) THEN
       ! left multiply with RESPONSE
       CALL GGEMM( 'N', 'N', NP, NP,  NP, one, CHI2, SIZE(CHI2,1), & 
            CHI_WORK(1,1), SIZE(CHI_WORK,1), zero, CHI3, SIZE(CHI3,1))

       ! right multiply with RESPONSE
       CALL GGEMM( 'N', 'N', NP, NP,  NP, one, CHI3, SIZE(CHI3,1), & 
            CHI2(1,1), SIZE(CHI2,1), zero, CHI_WORK, SIZE(CHI_WORK,1))
    ENDIF

    DEALLOCATE(CHI2, CHI3)

! currently TBSE stores all frequency dependent response functions (globally)
! on all cores, need global index

! TODO: this might often work, but is certainly not clean 
    IF (ASSOCIATED(TBSE%RESPONSEFUN) .AND. FREQUENCY_DEPENDENT_TBSE) THEN
       NOMEGA_GLOBAL=NOMEGA+CHI%NOMEGA_LOW-1
    ELSE
       NOMEGA_GLOBAL=1
    ENDIF

!    now trace to obtain correlation energy use entire TBSE
    TRACEX=0
    DO NI=1,NP
       IF (ASSOCIATED(TBSE%RESPONSEFUN)) THEN
          IF (TBSE%LREALSTORE) THEN
             DO NI_=1,NP
                TRACEX=TRACEX+CHI_WORK(NI,NI_)*TBSE%RESPONSER(NI_,NI,NOMEGA_GLOBAL)
             ENDDO
          ELSE
             DO NI_=1,NP
                TRACEX=TRACEX+CHI_WORK(NI,NI_)*TBSE%RESPONSEFUN(NI_,NI,NOMEGA_GLOBAL)
             ENDDO
          ENDIF
       ELSE
       !   TRACEX=TRACEX+CHI_WORK(NI,NI)*FXG(OLD_INDEX(NI))
          TRACEX=TRACEX+CHI_WORK(NI,NI)*FXG(NI)
       ENDIF
    ENDDO

    ! reduce to relevant indices for this cutoff
    CALL XI_LOCAL_FIELD_COMPRESS(CHI_WORK, MAXINDEX, OLD_INDEX)
    
    IF (LINVXI) THEN
       ! this should yield the same RPA/MP2 correlation energy as obtained from ROTLN
       TRACE=0
       DO NI=1,MAXINDEX
          POTFAK=VG(OLD_INDEX(NI))
          TRACE=TRACE+CHI_WORK(NI,NI)*POTFAK*POTFAK
       ENDDO
    ENDIF

!    now trace with reduced cutoff
    TRACEX2=0
    DO NI=1,MAXINDEX
       IF (ASSOCIATED(TBSE%RESPONSEFUN)) THEN
          IF (TBSE%LREALSTORE) THEN
             DO NI_=1,MAXINDEX
                TRACEX2=TRACEX2+CHI_WORK(NI,NI_)*TBSE%RESPONSER(OLD_INDEX(NI_),OLD_INDEX(NI),NOMEGA_GLOBAL)
             ENDDO
          ELSE
             DO NI_=1,MAXINDEX
                TRACEX2=TRACEX2+CHI_WORK(NI,NI_)*TBSE%RESPONSEFUN(OLD_INDEX(NI_),OLD_INDEX(NI),NOMEGA_GLOBAL)
             ENDDO
          ENDIF
       ELSE
          TRACEX2=TRACEX2+CHI_WORK(NI,NI)*FXG(OLD_INDEX(NI))
       ENDIF
    ENDDO
    ! seems TRACE allows more accurate extrapolation to infinite basis set limit than TRACEX
    ! TRACEX=TRACEX2

    IF (IMP2==0) THEN
       SUM=SUM+TRACE
       SUMSOSEX=SUMSOSEX+TRACEX
    ELSE
       SUMMP2=SUMMP2+TRACE
       SUMMP2EX=SUMMP2EX+TRACEX
    ENDIF
  ENDDO mp2rpa

    DEALLOCATE(VG, OLD_INDEX, FXG )

  END SUBROUTINE XI_LOCAL_FIELD_ACFDT_FX


  SUBROUTINE XI_LOCAL_FIELD_COMPRESS(CHI_WORK, MAXINDEX, OLD_INDEX)
    GDEF    :: CHI_WORK(:,:)
    INTEGER :: MAXINDEX
    INTEGER :: OLD_INDEX(:)
  !local
    INTEGER NI

    ! compress to relevant indices
    DO NI=1,MAXINDEX
       CHI_WORK(1:SIZE(CHI_WORK,1),NI)=CHI_WORK(1:SIZE(CHI_WORK,1),OLD_INDEX(NI))
    ENDDO

    DO NI=1,MAXINDEX
       CHI_WORK(NI,1:MAXINDEX)=CHI_WORK(OLD_INDEX(NI),1:MAXINDEX)
    ENDDO

  END SUBROUTINE XI_LOCAL_FIELD_COMPRESS

  SUBROUTINE XI_LOCAL_FIELD_UNCOMPRESS(CHI_WORK, MAXINDEX, OLD_INDEX)
    GDEF    :: CHI_WORK(:,:)
    INTEGER :: MAXINDEX
    INTEGER :: OLD_INDEX(:)
  !local
    INTEGER NI

    CHI_WORK(:,MAXINDEX+1:SIZE(CHI_WORK,2))=0
    CHI_WORK(MAXINDEX+1:SIZE(CHI_WORK,1),:)=0
    CHI_WORK(MAXINDEX+1:SIZE(CHI_WORK,1),MAXINDEX+1:SIZE(CHI_WORK,2))=0

    DO NI=MAXINDEX,1,-1
       CHI_WORK(OLD_INDEX(NI),1:MAXINDEX)=CHI_WORK(NI,1:MAXINDEX)
       IF (NI /= OLD_INDEX(NI)) CHI_WORK(NI,1:MAXINDEX)=0
    ENDDO

    DO NI=MAXINDEX,1,-1
       CHI_WORK(1:SIZE(CHI_WORK,1),OLD_INDEX(NI))=CHI_WORK(1:SIZE(CHI_WORK,1),NI)
       IF (NI /= OLD_INDEX(NI)) CHI_WORK(1:SIZE(CHI_WORK,1),NI)=0
    ENDDO

  END SUBROUTINE XI_LOCAL_FIELD_UNCOMPRESS



!******************** SUBROUTINE XI_LOCAL_FIELD_CCD      ***************
!
! this subroutine solves the CCD like Riccati equation by iteration
! unfortunately this does not yield the RPA energy, but
! can be used to calculate exact second and third order RPA and
! SOSEX energy
! IF YOU WANT TO USE THIS ROUTINE SET
!#define resonant
! in chi_base.F
! 
!  W= v
!  W= v + v X W + W X+ v + W X+ v X W 
! 
! Ecorr = Tr [W X+ v X-]
!
!***********************************************************************


  SUBROUTINE XI_LOCAL_FIELD_CCD( CHI_WORK, CHI, LATT_CUR, & 
       ENCUT, ENCUTSOFT, NOMEGA, WGWQ , MAXINDEX, LSR, &
       LREAL, SUM, SUMMP2,  SUMSOSEX, SUMMP2EX, LAMBDA, TBSE, IU6 )
    USE constant
    USE fock
    USE c2f_interface, ONLY : ERRF
    IMPLICIT NONE
    GDEF :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI   ! response function
    TYPE (responsefunction) :: TBSE  ! exchange kernel 
    TYPE (latt) LATT_CUR
    REAL(q) :: ENCUT, ENCUTSOFT
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
    INTEGER :: MAXINDEX              ! number of grid points using truncated Coulomb kernel
    LOGICAL :: LSR
    COMPLEX(q) :: SUM, SUMMP2        ! correlation energy and second order approximation to it
    COMPLEX(q) :: SUMSOSEX, SUMMP2EX ! exchange correlation energy and second order approximation to it
    LOGICAL    :: LREAL              ! real response function
    REAL(q)    :: LAMBDA             ! coupling strenght parameter (usually 1)
    INTEGER    :: IU6                ! unit for output
  ! local
    INTEGER    NI, NP, NI_
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE, SCALEFX, POTFAK, E, ETA
    REAL(q), ALLOCATABLE :: VG(:)
    INTEGER, ALLOCATABLE :: OLD_INDEX(:)
    REAL(q) :: DFUN, SFUN
    COMPLEX(q) :: TRACE, TRACEX, TRACEX2
    GDEF, ALLOCATABLE :: CHI2(:,:), CHI3(:,:)
    REAL(q) :: OMEGBK,QC
    REAL(q) :: GAMMASCALE

! multipole correction
    INTEGER    NJ, NK, ITER
    COMPLEX(q) :: TSUM(10)
    COMPLEX(q) :: POTCORRECTION(10)

    INTEGER :: NOMEGA_GLOBAL


    ! e^2/ volume 
    SCALE=EDEPS/LATT_CUR%OMEGA

    SCALEFX=FELECT/LATT_CUR%OMEGA/2

!=======================================================================
! first set up the truncated Coulomb kernel
! smoothly going from 1/G^2 to zero between ENCUTGWSOFT and ENCUTGW
!=======================================================================
    DKX=(WGWQ%VKPT(1))*LATT_CUR%B(1,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(1,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(WGWQ%VKPT(1))*LATT_CUR%B(2,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(2,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(WGWQ%VKPT(1))*LATT_CUR%B(3,1)+ &
        (WGWQ%VKPT(2))*LATT_CUR%B(3,2)+ &
        (WGWQ%VKPT(3))*LATT_CUR%B(3,3)

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2
    ALLOCATE( VG(NP), OLD_INDEX(NP))

    DATAKEMAX=0.0
    MAXINDEX=0
    DO NI=1,NP
       NI_=NI
       IF (WGWQ%LGAMMA) NI_=(NI-1)/2+1
       
       GX=(WGWQ%IGX(NI_)*LATT_CUR%B(1,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WGWQ%IGX(NI_)*LATT_CUR%B(2,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WGWQ%IGX(NI_)*LATT_CUR%B(3,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
       

       IF (ABS(GSQU)<G2ZERO) THEN
          ! head and wing
          POTFAK=SCALE
          ! if HFRCUT is set, a fixed spherical cutoff is used
          ! resulting in a finite values at G=0
          ! since the head stores the q^2, the final contribution is zero
          ! from the head
          IF (HFRCUT/=0) THEN
             POTFAK=0
          ! if LRHFCALC.AND.LRSCOR the Coulomb kernel is replaced by v_{LR}=v*exp(-q^2/(4*mu^2))
          ! for q=0  v_{LR}=1, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{LR}(q=0) is set to average of v_{LR} in a sphere  
          ! with volume corresponding to BZ volume/ # q-points       
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*GAMMASCALE
          ! if LSR the Coulomb kernel is replaced by v_{SR}=v*[1-exp(-q^2/(4*mu^2))]
          ! for q=0  v_{SR}=0, this introduces an error for small mu because of the finite q-grid
          ! In order to correct this error, v_{SR}(q=0) is set to average of v_{SR} in a sphere
          ! with volume corresponding to BZ volume/ # q-points
          ELSE IF (LSR) THEN
             CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
             QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
             GAMMASCALE=8*PI*HFSCREEN*HFSCREEN* &
            &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
            &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
             POTFAK=POTFAK*(1-GAMMASCALE)
          ENDIF

       ELSE
          ! the factor 1/(2 pi)^2 is required to obtain proper reciprocal
          ! lattice vector lenght
          POTFAK=SCALE/(GSQU*TPI**2)
          IF (HFRCUT/=0) THEN
             ! spherical cutoff on Coloumb kernel
             ! see for instance C.A. Rozzi, PRB 73, 205119 (2006)
!             POTFAK=POTFAK*(1-COS(SQRT(GSQU)*TPI*HFRCUT)*EXP(-(SQRT(GSQU)*TPI*HFRCUT)**2*HFRCUT_SMOOTH))
!test
             CALL DELSTP(3,SQRT(GSQU)*TPI*HFRCUT/10,DFUN,SFUN)
             POTFAK=POTFAK*(SFUN-0.5)*2
!test
          ELSE IF (LSR) THEN
             ! use the short range part of the Coulomb kernel
             POTFAK=POTFAK*(1._q-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
          ELSE IF (LRHFCALC.AND.LRSCOR) THEN
             ! use the long range part of the Coulomb kernel
             POTFAK=POTFAK*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
          ENDIF
       ENDIF

       ! smooth cutoff function between  ENCUTSOFT and ENCUT
       E=HSQDTM*(GSQU*TPI**2)
       IF (ENCUT>=0 .AND. E>ENCUT) THEN
          POTFAK=0
       ELSE

          MAXINDEX=MAXINDEX+1
          OLD_INDEX(MAXINDEX)=NI
          
          IF (ENCUTSOFT>=0 .AND. E>ENCUTSOFT) THEN
             POTFAK=POTFAK*(1+COS((E-ENCUTSOFT)/(ENCUT-ENCUTSOFT)*PI))/2
          ENDIF
       ENDIF

       VG(NI)=SQRT(POTFAK)
       ! maximum value in Coloumb kernel
       IF(VG(NI)>DATAKEMAX) THEN
          DATAKEMAX=VG(NI)
       ENDIF

       ETA=SQRT(GSQU)*TPI/(2*KFERMI)
    ENDDO
!=======================================================================
! now multiply the response function with the kernel
! from left and right hand side
!=======================================================================


    ! temporary work space allocation (not nice but safes memory)
    ALLOCATE(CHI2(SIZE(CHI_WORK,1),SIZE(CHI_WORK,2)), CHI3(SIZE(CHI_WORK,1),SIZE(CHI_WORK,2)))

    ! starting iteration W=v
    CHI_WORK=0
    DO NI=1,NP
       CHI_WORK(NI,NI)=CHI_WORK(NI,NI)+VG(NI)*VG(NI)
    ENDDO

 DO ITER=1,2

    IF (CHI%LREALSTORE) THEN
    ! left multiply with X W -> CHI2
       CALL GGEMM( 'N', 'N', NP, NP, NP, one, CHI%RESPONSER(1,1,NOMEGA), SIZE(CHI%RESPONSER,1), & 
         CHI_WORK(1,1), SIZE(CHI_WORK,1), zero, CHI2, SIZE(CHI2,1))
    ! right multiply with (X W) X+
       CALL GGEMM( 'N', trans, NP, NP, NP, one, CHI2, SIZE(CHI2,1), & 
         CHI%RESPONSER(1,1,NOMEGA), SIZE(CHI%RESPONSER,1), SIZE(CHI2,1), zero, CHI3, SIZE(CHI3,1))
    ELSE
    ! left multiply with X W -> CHI2
       CALL GGEMM( 'N', 'N', NP, NP, NP, one, CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1), & 
         CHI_WORK(1,1), SIZE(CHI_WORK,1), zero, CHI2, SIZE(CHI2,1))
    ! right multiply with (X W) X+
       CALL GGEMM( 'N', trans, NP, NP, NP, one, CHI2, SIZE(CHI2,1), & 
         CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1), zero, CHI3, SIZE(CHI3,1))
    ENDIF
    
    TRACE=0
    DO NI=1,NP
       TRACE=TRACE+CHI3(NI,NI)*VG(NI)*VG(NI)
    ENDDO
    TRACEX=0

    IF (ITER==1) THEN
       SUMMP2=SUMMP2+TRACE
       SUMMP2EX=SUMMP2EX+TRACEX
    ENDIF
    
    IF (CHI%LREALSTORE) THEN
       CHI2(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI2(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF

    DO NI=1,NP
       CHI2(NI,1:NP)=VG(NI)*VG(NI)*CHI2(NI,1:NP)
    ENDDO
    
    ! quadratic term   W X+ v X W 
    ! left multiply CHI_WORK:   v X W  -> CHI3
    CALL GGEMM( 'N', 'N', NP, NP, NP, one, CHI2, SIZE(CHI2,1), & 
         CHI_WORK(1,1), SIZE(CHI_WORK,1), zero, CHI3, SIZE(CHI3,1))

    ! multiply by X+ (v X W) -> CHI2
    IF (CHI%LREALSTORE) THEN
       CALL GGEMM( trans, 'N', NP, NP, NP, one, CHI%RESPONSER(1,1,NOMEGA), SIZE(CHI%RESPONSER,1), & 
            CHI3, SIZE(CHI3,1), zero, CHI2, SIZE(CHI2,1))
    ELSE
       CALL GGEMM( trans, 'N', NP, NP, NP, one, CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1), & 
            CHI3, SIZE(CHI3,1), zero, CHI2, SIZE(CHI2,1))
    ENDIF

    ! W (X+ v X W) -> CHI3
    ! at the moment only cubic terms are correct, 
    ! quadratic terms are removed since the enter only in fourth order 
    CALL GGEMM( 'N', 'N', NP, NP, NP, one, CHI_WORK, SIZE(CHI_WORK,1), & 
         CHI2, SIZE(CHI2,1), zero, CHI3, SIZE(CHI3,1))
 
! remove quadratic terms (should be commented out)
    CHI3=0

    ! now add linear terms   v X W + W X+ v

    ! restore v X
    IF (CHI%LREALSTORE) THEN
       CHI2(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI2(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF

    DO NI=1,NP
       CHI2(NI,1:NP)=VG(NI)*VG(NI)*CHI2(NI,1:NP)
    ENDDO

    ! left multiply with CHI_WORK:   v X W
    CALL GGEMM( 'N', 'N', NP, NP, NP, one, CHI2, SIZE(CHI2,1), & 
         CHI_WORK(1,1), SIZE(CHI_WORK,1), one, CHI3, SIZE(CHI3,1))

    ! add right multiply: W X+ v
    CALL GGEMM( 'N', trans, NP, NP, NP, one, CHI_WORK, SIZE(CHI_WORK,1), & 
         CHI2(1,1), SIZE(CHI2,1), one, CHI3, SIZE(CHI3,1))

! add Coulomb 
!    DO NI=1,NP
!       CHI3(NI,NI)=CHI3(NI,NI)+VG(NI)*VG(NI)
!    ENDDO
    CHI_WORK=CHI3

 ENDDO

    DEALLOCATE(CHI2, CHI3)

    SUM=SUM+TRACE
    SUMSOSEX=SUMSOSEX+TRACEX
    
    DEALLOCATE(VG, OLD_INDEX )

  END SUBROUTINE XI_LOCAL_FIELD_CCD


!******************** SUBROUTINE LIN_REG *******************************
!
! perform linear regression of correlation energy versus
! 1/energy^(3/2)
!
!***********************************************************************

  SUBROUTINE LIN_REG(COR, INOUT)

  TYPE (correlation), POINTER :: COR
  INTEGER :: INOUT
 
  REAL(q) :: SX, SY, SXY, SX2, SXM, SYM, SXYM, SX2M
  REAL(q) :: AREG, BREG, AREGM, BREGM
  INTEGER :: I, ILAMBDA
  REAL(q) :: LAMBDA

    SX = 0.0   
    SY = 0.0
    SXY = 0.0
    SX2 = 0.0
    SXM = 0.0   
    SYM = 0.0
    SXYM = 0.0
    SX2M = 0.0
  
    AREG = 0.0
    BREG = 0.0
    AREGM = 0.0
    BREGM = 0.0

    IF (INOUT>=0) WRITE(INOUT,'("      cutoff energy     smooth cutoff   RPA   correlation   Hartree contr. to MP2")')
    IF (INOUT>=0) WRITE(INOUT,'("---------------------------------------------------------------------------------")')       

    DO I=1,COR%NE
      IF (INOUT>=0) WRITE(INOUT,'(" ",2F18.3,2F20.10)') COR%ENCUTGW(I),COR%ENCUTGWSOFT(I), &
                 REAL(COR%CORRELATION(I),q), REAL(COR%CORRMP2DIR(I),q) 
      SX = SX+ 1/COR%ENCUTGW(I)**1.5
      SY = SY+ REAL(COR%CORRELATION(I),q)
      SXY = SXY+ 1/COR%ENCUTGW(I)**1.5*REAL(COR%CORRELATION(I),q)
      SX2 = SX2 + (1/COR%ENCUTGW(I)**1.5)**2
      SXM = SXM+ 1/COR%ENCUTGW(I)**1.5
      SYM = SYM+ REAL(COR%CORRMP2DIR(I),q)
      SXYM = SXYM+ 1/COR%ENCUTGW(I)**1.5*REAL(COR%CORRMP2DIR(I),q)
      SX2M = SX2M + (1/COR%ENCUTGW(I)**1.5)**2      
    ENDDO

    BREG =  (SY*SX - COR%NE*SXY)/(SX*SX - COR%NE*SX2) 
    AREG =  1/(COR%NE+0.0)*(SY - BREG*SX)
    BREGM = (SYM*SXM - COR%NE*SXYM)/(SXM*SXM - COR%NE*SX2M) 
    AREGM =  1/(COR%NE+0.0)*(SYM - BREGM*SXM) 

    IF (INOUT >=0) WRITE(INOUT,'("  linear regression    ")')  
    IF (INOUT >=0) WRITE(INOUT,'("  converged value                    ", 2F20.10)') AREG, AREGM
!   IF (INOUT >=0) WRITE(INOUT,'("  slope                              ", 2F20.10)') BREG, BREGM

    IF (NLAMBDA>=1) THEN
       IF (INOUT >=0) WRITE(INOUT,'("  converged value lambda= ",F6.4,"     ", 2F20.10)') 0.0_q,0.0_q
       DO ILAMBDA=0,NLAMBDA
       LAMBDA=1.0_q*(ILAMBDA+1)/MAX(1,NLAMBDA)
       SX = 0.0   
       SY = 0.0
       SXY = 0.0
       SX2 = 0.0

       DO I=1,COR%NE
          SX = SX+ 1/COR%ENCUTGW(I)**1.5
          SY = SY+ REAL(COR%CORRELATION_LAMBDA(I,ILAMBDA),q)
          SXY = SXY+ 1/COR%ENCUTGW(I)**1.5*REAL(COR%CORRELATION_LAMBDA(I,ILAMBDA),q)
          SX2 = SX2 + (1/COR%ENCUTGW(I)**1.5)**2
       ENDDO

       BREG =  (SY*SX - COR%NE*SXY)/(SX*SX - COR%NE*SX2) 
       AREG =  1/(COR%NE+0.0)*(SY - BREG*SX)

       IF (INOUT >=0) WRITE(INOUT,'("  converged value lambda= ",F6.4,"     ", 2F20.10)') LAMBDA,AREG
       ENDDO
    ENDIF


  END SUBROUTINE LIN_REG

  SUBROUTINE LIN_REG_EX(COR, INOUT)

  TYPE (correlation), POINTER :: COR
  INTEGER :: INOUT
 
  REAL(q) :: SX, SY, SXY, SX2, SXM, SYM, SXYM, SX2M
  REAL(q) :: AREG, BREG, AREGM, BREGM
  INTEGER :: I, ILAMBDA
  REAL(q) :: LAMBDA

    SX = 0.0   
    SY = 0.0
    SXY = 0.0
    SX2 = 0.0
    SXM = 0.0   
    SYM = 0.0
    SXYM = 0.0
    SX2M = 0.0
  
    AREG = 0.0
    BREG = 0.0
    AREGM = 0.0
    BREGM = 0.0

    IF (INOUT>=0) WRITE(INOUT,'("      cutoff energy     smooth cutoff   SOSEX correlation   exchangecontr. to MP2")')
    IF (INOUT>=0) WRITE(INOUT,'("---------------------------------------------------------------------------------")')       

    DO I=1,COR%NE
      IF (INOUT>=0) WRITE(INOUT,'(" ",2F18.3,2F20.10)') COR%ENCUTGW(I),COR%ENCUTGWSOFT(I), &
                 REAL(COR%CORRSOSEX(I),q), REAL(COR%CORRMP2EX(I),q) 
      SX = SX+ 1/COR%ENCUTGW(I)**1.5
      SY = SY+ REAL(COR%CORRSOSEX(I),q)
      SXY = SXY+ 1/COR%ENCUTGW(I)**1.5*REAL(COR%CORRSOSEX(I),q)
      SX2 = SX2 + (1/COR%ENCUTGW(I)**1.5)**2
      SXM = SXM+ 1/COR%ENCUTGW(I)**1.5
      SYM = SYM+ REAL(COR%CORRMP2EX(I),q)
      SXYM = SXYM+ 1/COR%ENCUTGW(I)**1.5*REAL(COR%CORRMP2EX(I),q)
      SX2M = SX2M + (1/COR%ENCUTGW(I)**1.5)**2      
    ENDDO

    BREG =  (SY*SX - COR%NE*SXY)/(SX*SX - COR%NE*SX2) 
    AREG =  1/(COR%NE+0.0)*(SY - BREG*SX)
    BREGM = (SYM*SXM - COR%NE*SXYM)/(SXM*SXM - COR%NE*SX2M) 
    AREGM =  1/(COR%NE+0.0)*(SYM - BREGM*SXM) 

    IF (INOUT >=0) WRITE(INOUT,'("  linear regression    ")')  
    IF (INOUT >=0) WRITE(INOUT,'("  converged value                    ", 2F20.10)') AREG, AREGM
!   IF (INOUT >=0) WRITE(INOUT,'("  slope                              ", 2F20.10)') BREG, BREGM


  END SUBROUTINE LIN_REG_EX


!**********************************************************************
!
! set auxiliary entry in wavefunction array to a weight
! which determines the weight of unoccupied orbitals in the calculation
! of the response functions
! the weight is set to one usually
! if a band is not included in the set of empty bands the weight is
! set to 0
!
!**********************************************************************

  SUBROUTINE VIRTUAL_BAND_CUTOFF(W, ENCUTIN, TELESCOPE)
    USE wave_high
    USE full_kpoints
    USE constant
    IMPLICIT NONE
    TYPE (wavespin), TARGET :: W
    REAL(q), OPTIONAL :: ENCUTIN
    INTEGER, OPTIONAL :: TELESCOPE
    !    local
    TYPE (wavedes1)    WDES1
    TYPE (wavedes), POINTER :: WDES
    
    INTEGER ISP, NK, NB
    REAL(q) :: E, ENCUT, ENCUT2
    REAL(q), PARAMETER :: CUTOFF=0.3_q

    WDES=>W%WDES
    IF (PRESENT(ENCUTIN)) THEN
       ENCUT=ENCUTIN
    ELSE
       ENCUT=0
    ENDIF

    IF (.NOT. PRESENT(TELESCOPE)) THEN
!
!
       spin:  DO ISP=1,WDES%ISPIN
          kpoint: DO NK=1,WDES%NKPTS
             DO NB=1,WDES%NB_TOT
                E=W%CELTOT(NB,NK,ISP)
                IF (NB>WDES%NB_TOTK(NK,ISP)) THEN
                   ! no weight for bands that are outside the basis set limit
                   W%AUXTOT(NB,NK,ISP)=0
                ELSE IF (PRESENT(ENCUTIN).AND.(E>ENCUT)) THEN
                   W%AUXTOT(NB,NK,ISP)=0
                ELSE IF (PRESENT(ENCUTIN).AND.(E>ENCUT*(1-CUTOFF))) THEN
                   ! smooth cutoff on weight
                   W%AUXTOT(NB,NK,ISP)=(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2
                ELSE
                   ! remaining ones have full weight
                   W%AUXTOT(NB,NK,ISP)=1
                ENDIF
             ENDDO
             !        WRITE(*,'(10F8.3)') W%AUX(:,NK,ISP)
          END DO kpoint
       END DO spin
    ELSE
!=======================================================================
!
! at low energies all kpoints and bands are included
! at higher energies a k-point downsampling is performed
!
!=======================================================================
       IF (TELESCOPE==1) THEN
       ENCUT=ENCUTIN/2_q**(2._q/3._q)
       DO ISP=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS

             DO NB=1,WDES%NB_TOT
                E=W%CELTOT(NB,NK,ISP)
                IF (NB>WDES%NB_TOTK(NK,ISP)) THEN
                   ! no weight for bands that are outside the basis set limit
                   W%AUXTOT(NB,NK,ISP)=0
                ELSE
                   !
                   ! low energy part
                   !
                   IF (E>ENCUT) THEN
                      ! no weight for bands that are larger than ENCUT
                      W%AUXTOT(NB,NK,ISP)=0
                   ELSE IF (E>ENCUT*(1-CUTOFF)) THEN
                      ! smooth cutoff on weight
                      W%AUXTOT(NB,NK,ISP)=(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2
                   ELSE
                      ! remaining ones have full weight
                      W%AUXTOT(NB,NK,ISP)=1
                   ENDIF
                   !
                   ! high energy part
                   !
                   IF (MOD(FLOOR((W%WDES%VKPT(1,NK)+32)*KPOINTS_FULL%NKPX+.5),2)==0 .AND. &
                       MOD(FLOOR((W%WDES%VKPT(2,NK)+32)*KPOINTS_FULL%NKPY+.5),2)==0 .AND. &
                       MOD(FLOOR((W%WDES%VKPT(3,NK)+32)*KPOINTS_FULL%NKPZ+.5),2)==0) THEN
                      IF (E>ENCUT) THEN
                         ! add weight 8 for bands that are larger than ENCUT
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+8
                      ELSE IF (E>ENCUT*(1-CUTOFF)) THEN
                         ! smooth cutoff on weight
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+8 & 
                              *(1-(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2)
                      ENDIF
                   ENDIF
                ENDIF
             ENDDO
          END DO
       END DO

       ELSE
!=======================================================================
! this divides the kpoints/ bands in three sets
! the low energy between    [0, ENCUT]
! the medium energy between [ENCUT, ENCUT2]
! the high energy between   [ENCUT2, energy cutoff]
!=======================================================================
       ENCUT =ENCUTIN/2_q**(4._q/3._q)
       ENCUT2=ENCUTIN/2_q**(2._q/3._q)

       DO ISP=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS

             DO NB=1,WDES%NB_TOT
                E=W%CELTOT(NB,NK,ISP)
                IF (NB>WDES%NB_TOTK(NK,ISP)) THEN
                   ! no weight for bands that are outside the basis set limit
                   W%AUXTOT(NB,NK,ISP)=0
                ELSE 
                   !
                   ! low energy part
                   !
                   IF (E>ENCUT) THEN
                      ! no weight for bands that are larger than ENCUT
                      W%AUXTOT(NB,NK,ISP)=0 
                   ELSE IF (E>ENCUT*(1-CUTOFF)) THEN
                      ! smooth cutoff on weight
                      W%AUXTOT(NB,NK,ISP)=(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2
                   ELSE
                      ! remaining ones have full weight
                      W%AUXTOT(NB,NK,ISP)=1
                   ENDIF
                   ! WRITE(76,*) E, W%AUXTOT(NB, NK, ISP)
                   !
                   ! medium energy part
                   !
                   IF (ABS(MODULO(NINT(W%WDES%VKPT(1,NK)*KPOINTS_FULL%NKPX+ &
                                   W%WDES%VKPT(2,NK)*KPOINTS_FULL%NKPY+ &
                                   W%WDES%VKPT(3,NK)*KPOINTS_FULL%NKPZ),2))<1E-6) THEN
                      IF (E>ENCUT2) THEN
                         ! not weighted
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+0
                      ELSE IF (E>ENCUT2*(1-CUTOFF)) THEN
                         ! smooth cutoff on weight
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+2* & 
                              (1+COS((E-ENCUT2*(1-CUTOFF))/(ENCUT2*CUTOFF)*PI))/2
                      ELSE IF (E>ENCUT) THEN
                         ! add weight 8 for bands that are larger than ENCUT
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+2
                      ELSE IF (E>ENCUT*(1-CUTOFF)) THEN
                         ! smooth cutoff on weight
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+2* & 
                              (1-(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2)
                      ENDIF
                      ! WRITE(77,*) E, W%AUXTOT(NB, NK, ISP)
                   ENDIF
                   !
                   ! high energy part
                   !
                   IF (MOD(FLOOR((W%WDES%VKPT(1,NK)+32)*KPOINTS_FULL%NKPX+.5),2)==0 .AND. &
                       MOD(FLOOR((W%WDES%VKPT(2,NK)+32)*KPOINTS_FULL%NKPY+.5),2)==0 .AND. &
                       MOD(FLOOR((W%WDES%VKPT(3,NK)+32)*KPOINTS_FULL%NKPZ+.5),2)==0) THEN
                      IF (E>ENCUT2) THEN
                         ! add weight 64 for bands that are larger than ENCUT2
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+8
                      ELSE IF (E>ENCUT2*(1-CUTOFF)) THEN
                         ! smooth cutoff on weight
                         W%AUXTOT(NB,NK,ISP)=W%AUXTOT(NB,NK,ISP)+8 & 
                              *(1-(1+COS((E-ENCUT2*(1-CUTOFF))/(ENCUT2*CUTOFF)*PI))/2)
                      ENDIF
                      ! WRITE(78,*) E, W%AUXTOT(NB, NK, ISP)
                   ENDIF

                ENDIF
             ENDDO
          END DO
       END DO
          
       ENDIF
    ENDIF
  END SUBROUTINE VIRTUAL_BAND_CUTOFF

!************************ SUBROUTINE FOCK_ACFDT ************************
!
! calculate the total HF energy in a way that is fully compatible
! to ACFDT
! compare equation (12) in Harl et al. PRB 81 115126 (2010)
!
! 
!***********************************************************************

  SUBROUTINE FOCK_ACFDT(GRID, LATT_CUR, W, LMDIM, NONLR_S, NONL_S, P  , OMEGAWEIGHT, OMEGA, IU6)
    USE sym_prec
    USE nonl_high
    USE wave
    USE wave_high
    USE mpimy
    USE mgrid
    USE lattice
    USE constant
    USE pseudo
    USE full_kpoints
    USE paw
    IMPLICIT NONE

! passed variables
    TYPE (grid_3d)  GRID
    TYPE (latt)     LATT_CUR
    TYPE (wavespin) W
    INTEGER LMDIM
    TYPE (nonlr_struct)  NONLR_S
    TYPE (nonl_struct)   NONL_S
    TYPE (potcar)        P(NONLR_S%NTYP)
    GDEF ::              GWORK(m_ GRIDHF%MPLWV,W%WDES%NRSPINORS) ! fock pot in real sp
    INTEGER :: IU6
! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1) :: W1, WQ
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    REAL(q) :: WEIGHT
    REAL(q) :: FSG
    INTEGER NK, ISP, NPOS
    INTEGER N, N_, MQ, NGLB, NGLBN, NSTRIP, NSTRIPN, ISP_IRZ, I
    INTEGER NQ
    LOGICAL LSHIFT
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:)    ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! 1/(G+dk)**2 (G)
    REAL(q)                :: E, EHF, EHF_ACFDT, F, DE
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF
    REAL(q)   :: OMEGAWEIGHT(:) ! weights for Gauss integration
    REAL(q)   :: OMEGA(:)      ! complex frequencies

    NULLIFY(ROT_HANDLE)
!==========================================================================
! some MPI variable initialisation
! and other initialisations
!==========================================================================
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK
    CALL CHECK_FULL_KPOINTS

    ! MPI dividable by NB_PAR
    NSTRIP=((WHF%WDES%NSIM*2+WHF%WDES%NB_PAR-1)/WHF%WDES%NB_PAR)
    NGLB  =NSTRIP*WHF%WDES%NB_PAR

       ! allocate all required structures
    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
      CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS), &
      CDLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
      POTFAK(GRIDHF%MPLWV))


! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band

    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ , WDESQ,.TRUE.)
    CALL SETWDES(WHF%WDES,WDESK,0)
    ALLOCATE(WIN(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(WIN(N) , WDESK,.TRUE.)
    ENDDO

    FSG=SET_FSG(GRIDHF, LATT_CUR)
!==========================================================================
! main loop over spin, k-points, and bands (in blocks of NSTRIP)
!==========================================================================
    EHF=0
    EHF_ACFDT=0

    spin:   DO ISP=1,WHF%WDES%ISPIN
    kpoint: DO NK=1,WHF%WDES%NKPTS
#ifdef MPI
       IF (MOD(NK-1,WHF%WDES%COMM_KINTER%NCPU).NE.WHF%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
    CALL SETWDES(WHF%WDES,WDESK,NK)

    band:   DO NPOS=1,LAST_FILLED_XI(W,NK,ISP)/WHF%WDES%NB_PAR,NSTRIP

    NSTRIPN=MIN(WHF%WDES%NBANDS+1-NPOS,NSTRIP)
    NGLBN  =NSTRIPN*WHF%WDES%NB_PAR
!==========================================================================
! fourier transform the bands to be accelerated to real space (CWRN)
! then distribute the CWRN array to all nodes
!==========================================================================
    CALL W1_GATHER( WHF, NPOS, NPOS+NSTRIPN-1, ISP, WIN)
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!  start literal copy
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
       IF( KPOINTS_FULL%WTKPT(NQ)==0 .OR. &
            (HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(WHF%WDES%VKPT(:,NQ))) .OR. &
            (.NOT.HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK)))) CYCLE

       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))
       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF
   ! set POTFAK for this q and k point
       CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,WHF%WDES%NBANDS
          ! this short cut is usually ok, if the groundstate calculations have not been performed 
          ! using a widely different SIGMA
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband

          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.

             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
         ! calculate charge phi_k nk(r) phi*_q nq(r) 
          nband: DO N=1,NGLBN
             N_=(NPOS-1)*WHF%WDES%NB_PAR+N ! global storage index of present band

             CALL FOCK_CHARGE( WIN(N), WQ, GWORK(:,1), CRHOLM)
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1,1),GRIDHF,-1)
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL EXCHANGE_GFAC(GRIDHF, GWORK(1,1), POTFAK(1), E)
             WEIGHT=WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WHF%FERTOT(N_,NK,ISP)* &
                  WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)
             EHF=EHF+E*WEIGHT/GRIDHF%NPLWV

             DE=-(WHF%CELTOT(N_,NK,ISP)-WHF%CELEN(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))
             ! the cutof should be chose such that F goes through 0 or 2 respectively
             ! (test for some systems e.g. CO-Cu, increasing from 3 to 7 does not change adsorption energy)
             IF (DE>OMEGA(1)*3) THEN
               F=0
             ELSE  IF (DE<-OMEGA(1)*3) THEN
               F=2
             ELSE
               F=0
               DO I =1, SIZE(OMEGA)
                  F=F+OMEGAWEIGHT(I)*2*DE/(DE*DE+OMEGA(I)*OMEGA(I))
               ENDDO
               ! add correction from integral of OMEGATL to infinity
               F=F+2*DE/OMEGA(SIZE(OMEGA))
               F=-(F/(PI)-1)
             ENDIF
!            if the integral is done analytically, one obtains the following result
!            WEIGHT=WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*MIN(WHF%FERTOT(N_,NK,ISP),WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))*F
             WEIGHT=WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WHF%FERTOT(N_,NK,ISP)*F
             EHF_ACFDT=EHF_ACFDT+E*WEIGHT/GRIDHF%NPLWV
          ENDDO nband
       ENDDO mband
    ENDDO qpoints

    ENDDO band

    ENDDO kpoint
    ENDDO spin
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
    CALLMPI( M_sum_d(WHF%WDES%COMM_INTER,EHF,1))
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER,EHF,1))

    CALLMPI( M_sum_d(WHF%WDES%COMM_INTER,EHF_ACFDT,1))
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER,EHF_ACFDT,1))

    IF (IU6>=0) WRITE(IU6,'("HF-correction",3F16.7)') -EHF/2, -EHF_ACFDT/2, -EHF_ACFDT/2+EHF/2

    DEALLOCATE(CRHOLM,CDIJ,CDLM,POTFAK)

    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

  END SUBROUTINE FOCK_ACFDT


!************************ SUBROUTINE DETERMINE_HEX *********************
!
! subroutine to determine approximate exchange (and correlation) 
! kernels
! this routine determine k_F such that 
!
!  sum_G rho^*(G) f_HEX(G) rho(G) =  exact exchange energy
!
! only plane wave contributions are taken into account
! for consistency reasons and the density is evaluated using the 
! fast charge augmentation routines
! 
!***********************************************************************

  SUBROUTINE DETERMINE_HEX( &
       P,NONLR_S,NONL_S,W,LATT_CUR, &
       T_INFO,IO,SYMM, &
       LMDIM,CQIJ)

    USE wave_high
    USE hamil
    USE full_kpoints
    USE pseudo
    USE lattice
    USE sym_prec
    USE base
    USE lattice

    IMPLICIT NONE
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W          ! wavefunction
    TYPE (latt)        LATT_CUR
    TYPE (in_struct)   IO
    TYPE (symmetry) :: SYMM
    INTEGER LMDIM
    OVERLAP  CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
 !  local
    REAL (q) EXHF
    COMPLEX(q) :: CHTOT( GRIDHF%MPLWV)
    REAL(q)    :: FXG(GRIDHF%MPLWV)
    REAL(q)    :: KF, KFUP, KFLOW, EX, EH
    REAL(q)    :: VKPT(3)
    INTEGER    :: N
    
    ! first calculate HF energy
    CALL FOCK_ALL( LATT_CUR, NONLR_S, NONL_S, W, &
         &    LMDIM, P, CQIJ, EXHF)

    EXHF = -EXHF

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(/A,/,A,/,A,/,A,F18.8)')  &
            ' Determing k_F from the exact exchange energy ', &
            ' =============================================', &
            ' to reproduce DFT values use LMAXFOCKAE= -1 ; PRECFOCK = M',&
            ' The exact exchange energy is ', EXHF
    ENDIF
   
    CALL TOTAL_CHARGE_FOCK( W, P, T_INFO, LATT_CUR, CHTOT )

    VKPT=0
    CALL SET_GFAC_COULOMB(GRIDHF, LATT_CUR, VKPT, FXG )

    EH=0
    DO N=1,GRIDHF%RC%NP
       EH=EH+CHTOT(N)*CONJG(CHTOT(N))*FXG(N)
    ENDDO
    EH=EH/2

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(A,F18.8)')' The Hartree energy is        ', EH
    ENDIF


    KF = 2.0
    KFUP =10.0
    KFLOW=0.0

!    KF=1.7566
!    KFUP =KF
!    KFLOW=KF

    DO 
       VKPT=0
       CALL SET_GFAC_FX_HEG(GRIDHF, LATT_CUR, VKPT, FXG, KF)

       ! 1/2 n(r) F_x(r,r') n(r') d3r d^3' = 1/2 sum_g |n(G)|^2 F_x(G)
       EX=0
       DO N=1,GRIDHF%RC%NP
          EX=EX+CHTOT(N)*CONJG(CHTOT(N))*FXG(N)
       ENDDO
       EX=EX/2

       IF (EX<EXHF) THEN
          KFLOW=KF
          KF=(KF+KFUP)/2
       ELSE
          KFUP=KF
          KF=(KF+KFLOW)/2
       ENDIF
       IF ( (KFUP-KFLOW) < 1E-8) THEN
          EXIT
       ENDIF
    ENDDO

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(A,F18.8)')' The LF    exchange energy is (eV) ', EX
       WRITE(IO%IU6,'(/A,F14.4,/)')' Fermi wave vector k_F  (A^-1) ', KF
    ENDIF

    KFERMI=KF

  END SUBROUTINE DETERMINE_HEX



!********************* TOTAL_CHARGE_FOCK *******************************
!
! this subroutine constructs the electronic charge density using
! the same FFT grid as the HF routine and FAST_AUG
! it essentially uses the same symmetrization is ISYM = 3
!
!***********************************************************************

  SUBROUTINE TOTAL_CHARGE_FOCK( W, P, T_INFO, LATT_CUR, CHTOT )
    USE wave_high
    USE gridq
    USE hamil
    USE full_kpoints
    USE pseudo
    USE lattice
    USE sym_prec
    IMPLICIT NONE

    TYPE (wavespin)    W
    COMPLEX(q) :: CHTOT( GRIDHF%MPLWV)
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (latt)        LATT_CUR
  ! local
    INTEGER :: ISP, NK, N, ISP_IRZ
    REAL(q) :: WEIGHT
    LOGICAL :: LSHIFT
    TYPE (wavedes1)     WDES1, WDES1_IRZ
    TYPE (wavefun1)     W1
    INTEGER ISPINOR
   ! work arrays
    REAL (q) :: VKPT(3)
    TYPE (wavespin) WHF
    GDEF, ALLOCATABLE :: GWORK(:)   ! charge from one band in real space
    GDEF, ALLOCATABLE :: GHTOT(:)   ! total charge
    GDEF, ALLOCATABLE :: CRHOLM(:)  ! augmentation occupancy matrix
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    CALL CHECK_FULL_KPOINTS
    NULLIFY(ROT_HANDLE)

    CALL SETWDES(WHF%WDES, WDES1, 0)
    CALL NEWWAV(W1 , WDES1,.TRUE.)

    ! MPLWV is the allocation in complex words
    ! hence if GWORK is REAL one needs to double the allocation 
    ALLOCATE(GWORK(m_ GRIDHF%MPLWV), & 
         GHTOT(m_ GRIDHF%MPLWV), &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS))

    ! set correct phase in FAST_AUG structure
    VKPT = 0
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, VKPT)

    GHTOT=0

    spin: DO ISP=1,WHF%WDES%ISPIN
       kpoints: DO NK=1,KPOINTS_FULL%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WHF%WDES%COMM_KINTER%NCPU).NE.WHF%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL SETWDES(WHF%WDES, WDES1, NK)
          CALL SETWDES(WHF%WDES, WDES1_IRZ, KPOINTS_FULL%NEQUIV(NK))
          
          ISP_IRZ=ISP
          IF (KPOINTS_FULL%SPINFLIP(NK)==1) THEN
             ISP_IRZ=3-ISP
          ENDIF
         
          band: DO N=1,WHF%WDES%NBANDS
             WEIGHT=WHF%WDES%RSPIN*KPOINTS_FULL%WTKPT(NK)*WHF%FERWE(N,KPOINTS_FULL%NEQUIV(NK),ISP)

             IF (WEIGHT==0) CYCLE

             IF (NK <= WHF%WDES%NKPTS) THEN
                CALL W1_COPY(ELEMENT(WHF, WDES1, N, ISP), W1)
                CALL FFTWAV_W1(W1)
             ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NK (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
                LSHIFT=.FALSE.
                IF ((ABS(KPOINTS_FULL%TRANS(1,NK)) >TINY) .OR. &
                     (ABS(KPOINTS_FULL%TRANS(2,NK))>TINY) .OR. &
                     (ABS(KPOINTS_FULL%TRANS(3,NK))>TINY)) LSHIFT=.TRUE.
             CALL W1_ROTATE_AND_FFT(W1, ELEMENT(WHF, WDES1_IRZ, N, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
             ENDIF
             CALL FOCK_CHARGE( W1, W1, GWORK(:), CRHOLM)

             GHTOT(1:GRIDHF%RL%NP)=GHTOT(1:GRIDHF%RL%NP)+GWORK(1:GRIDHF%RL%NP)*(WEIGHT*(1.0_q/GRIDHF%NPLWV))
          ENDDO band
       ENDDO kpoints

    ENDDO spin
#ifdef gammareal
    CALLMPI( M_sum_d(WHF%WDES%COMM_INTER, GHTOT(1), GRIDHF%RL%NP))
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER, GHTOT(1), GRIDHF%RL%NP))
#else
    CALLMPI( M_sum_z(WHF%WDES%COMM_INTER, GHTOT(1), GRIDHF%RL%NP))
    CALLMPI( M_sum_z(WHF%WDES%COMM_KINTER, GHTOT(1), GRIDHF%RL%NP))
#endif
    CALL FFT3D(GHTOT(1),GRIDHF,-1)

    CALL RC_ADD(GHTOT(1),1.0_q,GHTOT(1),0.0_q,CHTOT(1),GRIDHF)

    DEALLOCATE(GWORK, GHTOT, CRHOLM)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(W1, .TRUE.)

  END SUBROUTINE TOTAL_CHARGE_FOCK


!********************** SUBROUTINE SET_GFAC_SINC **********************
! 
!  setup a homogenoes electron gas local field exchange factor f(G)
!  in real space this corresponds to 
!   1 - sin (2 k_F r)/(2 k_F r)
! 
!**********************************************************************

    SUBROUTINE SET_GFAC_FX_HEG(GRID, LATT_CUR, VKPT, FXG, KFERMI)
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID          ! grid descriptor
      TYPE (latt) LATT_CUR         ! lattice vectors
      REAL(q) :: VKPT(3)           ! k-point
      REAL(q) :: KFERMI            ! fermi wave vector
      REAL(q) :: FXG(GRID%MPLWV)   ! exchange factor 
   ! local
      INTEGER    NI,NC,N1,N2,N3
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,SCALEFX
      REAL(q) :: ETA

      CALL CHECK_FULL_KPOINTS
! FELECT = (the electronic charge)/(4*pi*the permittivity of free space)
!         in atomic units this is just e^2
! fx is for up-up and down-down correlation only
! density density correlation includes also up-down contribution
! hence divide by two
      SCALEFX=FELECT/LATT_CUR%OMEGA /2

      ! set correct phase in FAST_AUG structure
      CALL PHASER_HF(GRID, LATT_CUR, FAST_AUG_FOCK,VKPT)

      DKX=VKPT(1)*LATT_CUR%B(1,1)+VKPT(2)*LATT_CUR%B(1,2)+VKPT(3)*LATT_CUR%B(1,3)
      DKY=VKPT(1)*LATT_CUR%B(2,1)+VKPT(2)*LATT_CUR%B(2,2)+VKPT(3)*LATT_CUR%B(2,3)
      DKZ=VKPT(1)*LATT_CUR%B(3,1)+VKPT(2)*LATT_CUR%B(3,2)+VKPT(3)*LATT_CUR%B(3,3)

      NI=0
      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         row: DO N1=1,GRID%RC%NROW
            NI=NI+1
            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))
            GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

            ETA=SQRT(GSQU)*TPI/(2*KFERMI)
            
            IF (GSQU<KPOINTS_FULL%VKPTMINDIST2/40) THEN
               FXG(NI)=-SCALEFX *9*PI/KFERMI**2
            ELSE
               FXG(NI)=-SCALEFX *3*PI/(5*KFERMI**2)*( &
                    (2/ETA-10*ETA)*LOG((1+ETA)/ABS(1-ETA))+ &
                    (2*ETA**4-10*ETA**2)*LOG((1+ETA)*ABS(1-ETA)/ETA**2)+ &
                    2*ETA**2+11)
            ENDIF
         ENDDO row
      ENDDO col


    END SUBROUTINE SET_GFAC_FX_HEG


!********************** SUBROUTINE SET_GFAC_COULOMB *******************
! 
!  setup a truncated Coloumb kernel
! 
!**********************************************************************

    SUBROUTINE SET_GFAC_COULOMB(GRID, LATT_CUR, VKPT, FXG )
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      REAL(q) :: VKPT(3)
      REAL(q) :: FXG(GRID%MPLWV)
   ! local
      INTEGER    NI,NC,N1,N2,N3
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,SCALE

      CALL CHECK_FULL_KPOINTS

    ! e^2/ volume 
      SCALE=EDEPS/LATT_CUR%OMEGA

      ! set correct phase in FAST_AUG structure
      CALL PHASER_HF(GRID, LATT_CUR, FAST_AUG_FOCK,VKPT)

      DKX=VKPT(1)*LATT_CUR%B(1,1)+VKPT(2)*LATT_CUR%B(1,2)+VKPT(3)*LATT_CUR%B(1,3)
      DKY=VKPT(1)*LATT_CUR%B(2,1)+VKPT(2)*LATT_CUR%B(2,2)+VKPT(3)*LATT_CUR%B(2,3)
      DKZ=VKPT(1)*LATT_CUR%B(3,1)+VKPT(2)*LATT_CUR%B(3,2)+VKPT(3)*LATT_CUR%B(3,3)

      NI=0
      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         row: DO N1=1,GRID%RC%NROW
            NI=NI+1
            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))
            GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

            IF (GSQU<KPOINTS_FULL%VKPTMINDIST2/40) THEN
               FXG(NI)=0
            ELSE
               FXG(NI)=SCALE/(GSQU*TPI*TPI)
            ENDIF
         ENDDO row
      ENDDO col


    END SUBROUTINE SET_GFAC_COULOMB


END MODULE ACFDT
