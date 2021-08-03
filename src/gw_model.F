!#define dotiming
#include "symbol.inc"

!***********************************************************************
!
! this module implements the model GW
! the module was mostly implemented by gK and Maxim Shiskin
! it is "based" on the implemenation of jF in vasp.4.4
! actually it is a complete rewrite, but some formulas and F90
! lines have been compied from the original files (in particular
! dynamical correlation part)
! The implementation is based on the following paper:
!
!***********************************************************************


MODULE gw_model
  USE prec
  USE fock
  IMPLICIT NONE

  GDEF, ALLOCATABLE, SAVE :: RHOMAT(:,:,:)

    
CONTAINS

!*********************************************************************
!
!  calculate the matrix elements
!  RHOMAT(i,j,k,isp) =  int psi_i,k*(r) rho(r) psi_j,k(r)
!  comments
!
!*********************************************************************

  SUBROUTINE GW_MODEL_SET_RHOMAT(P, W, LATT_CUR, &
          T_INFO, GRIDC, GRIDUS, GRID_SOFT, SOFT_TO_C, C_TO_US, &
          IRDMAX, LMDIM, DENCOR, CHTOT )

    USE base
    USE lattice
    USE pseudo
    USE msymmetry
    USE mgrid
    USE poscar
    USE wave_high
    USE pot
    USE pawm
    USE wave_high
    USE hamil
    USE subrot
    USE setexm
    USE constant
    IMPLICIT NONE

    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    TYPE (grid_3d)     GRIDC                  ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS                 ! grid for augmentation
    TYPE (grid_3d)     GRID_SOFT              ! grid for soft chargedensity
    TYPE (transit)     SOFT_TO_C              ! index table between GRID_SOFT and GRIDC
    TYPE (transit)     C_TO_US                ! index table between GRID_SOFT and GRIDC
    INTEGER  IRDMAX
    INTEGER  LMDIM
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,W%WDES%NCDIJ) ! local charge
  ! local
    INTEGER ISP, NK, N
    COMPLEX(q), ALLOCATABLE :: CWORK(:,:)     ! available for temporary storage
    GDEF, ALLOCATABLE :: MAT(:,:,:,:)
    REAL(q) :: RHOMAT_SUM, FERWE_SUM, RHOEFF, RHOEFF13, QTF2

  ! quick return if MODEL_GW is not used or smaller equal 1
  ! only for MODEL_GW==2 the diagonal screening matrix becomes
  ! state dependent
    IF (MODEL_GW<=0) RETURN
#ifdef MPI
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('GW_MODEL_SET_RHOMAT: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#endif
    IF (.NOT. ALLOCATED( RHOMAT)) THEN
       ALLOCATE( RHOMAT(W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN))
    ENDIF
    ALLOCATE(CWORK(GRIDC%MPLWV,W%WDES%NCDIJ), MAT(W%WDES%NB_TOT,W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN))

    CWORK=CHTOT
    ! change storage convention to up/ down
    CALL POT_FLIP(CWORK,GRIDC,W%WDES%NCDIJ)
    DO ISP=1,W%WDES%NCDIJ
       ! divide by 1/Omega this is required since the charge times volume is stored in CWORK
       CALL RC_ADD(CWORK(1,ISP),1.0/LATT_CUR%OMEGA,CWORK(1,ISP),0.0_q,CWORK(1,ISP),GRIDC)
       ! transform to real space
       CALL FFT3D(CWORK(1,ISP),GRIDC,1)
    ENDDO
    MAT=0
    CALL LOCAL_BETWEEN_STATES( W, LATT_CUR, P, T_INFO, IRDMAX, LMDIM, &
       GRID_SOFT, GRIDC, GRIDUS, SOFT_TO_C, C_TO_US, CWORK, MAT)

    RHOMAT_SUM=0
    FERWE_SUM =0

    DO ISP=1,W%WDES%ISPIN
       DO NK=1,W%WDES%NKPTS
          DO N=1,W%WDES%NB_TOT
             RHOMAT(N, NK, ISP) = MAT(N, N, NK, ISP)
             RHOMAT_SUM=RHOMAT_SUM+RHOMAT(N, NK, ISP)*W%FERTOT(N, NK, ISP)
             FERWE_SUM=FERWE_SUM+W%FERTOT(N, NK, ISP)
          ENDDO
       ENDDO
    ENDDO

    RHOMAT_SUM=RHOMAT_SUM/FERWE_SUM

    RHOEFF=AUTOA*AUTOA*AUTOA*RHOMAT_SUM
    RHOEFF13=EXP(LOG(RHOEFF)*.3333333333333333_q)
    ! q_TF^2 (divided by AUTOA^2 which is the conversion factor for q^2):
    QTF2=(4._q*EXP(LOG(3._q/PI)/3._q))*RHOEFF13*1._q/(AUTOA*AUTOA)

    WRITE(*,*) 'average Thomas-Fermi wavevector',SQRT(QTF2)
    DEALLOCATE(CWORK, MAT)

  END SUBROUTINE GW_MODEL_SET_RHOMAT

!*******************************************************************
!
! Function that returns .TRUE. if model GW is selected
!
!*******************************************************************

  FUNCTION IS_MODEL_GW()
    LOGICAL IS_MODEL_GW

    IF (MODEL_GW>0) THEN
       IS_MODEL_GW=.TRUE.
    ELSE
       IS_MODEL_GW=.FALSE.
    ENDIF
  END FUNCTION IS_MODEL_GW


!*******************************************************************
!
!  calculate the dynamical contribution
!
!*******************************************************************

  SUBROUTINE GWDYNSM1_IJ(W,WDES,KPOINTS,LATT_CUR,T_INFO,INFO,IO,E_DFT)
    USE prec
    USE constant
    USE base
    USE wave
    USE mpimy
    USE poscar
    USE lattice
    USE mkpoints
    USE c2f_interface, ONLY : VTIME

    IMPLICIT NONE

    TYPE (wavespin)       W
    TYPE (wavedes)        WDES
    TYPE (info_struct)    INFO
    TYPE (latt)           LATT_CUR
    TYPE (kpoints_struct) KPOINTS
    TYPE (type_info)      T_INFO
    TYPE (in_struct)      IO

    REAL(q) E_DFT(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
    CHARACTER(LEN=1)  CHARAC
    COMPLEX(q)  CDUM
    LOGICAL FIRST_CALL,LOPEN,LDUM,lwrite
    INTEGER I,ISP,NK,N,IDUM,MAXBAND,MINBAND,MAXT,MINT,IU0,IU6,IU5,IERR,ierror
    INTEGER NXTFRU,IUNIT
    REAL(q) AVGAP,AVGAP_LAST,FAC,TV,TV0,TC,TC0
    REAL(q) FACQ2,RHOEFF,OMEGAP2,RHOEFF13,QTF2
    REAL(q) CC,DQ,QI,Q2,Q4,OQ,OQ2,OQC,F,A,B,BB,S
    REAL(q) E_SIGMA(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
    REAL(q) Z(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)

    SAVE FIRST_CALL,AVGAP
    DATA FIRST_CALL /.TRUE./

  ! quick return if MODEL_GW is not used or smaller equal 1
  ! only for MODEL_GW==2 the diagonal screening matrix becomes
  ! state dependent
    IF (MODEL_GW<=0) RETURN


    CALL VTIME(TV0,TC0)

    IU0=IO%IU0
    IU6=IO%IU6
    IU5=IO%IU5

! conversion factor between q_TF**2 [a.u.] and rho [a.u.]
    FACQ2=4._q*EXP(LOG(3._q/PI)/3._q)

    IF (FIRST_CALL) THEN
! estimate the average gap (assume that we have a real insulator)
       FIRST_CALL=.FALSE.
       AVGAP=0._q
       MINBAND=WDES%NB_TOT
       MAXBAND=0
       DO ISP=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS
             MINT=WDES%NB_TOT
             MAXT=0
             DO N=1,WDES%NB_TOT
                IF (W%FERTOT(N,NK,ISP)>=1.E-6_q) MAXT=MAX(MAXT,N)
                IF (W%FERTOT(N,NK,ISP)< 1.E-6_q) MINT=MIN(MINT,N)
             ENDDO
             IF (MINBAND>MAXBAND) THEN
                MINBAND=MINT
                MAXBAND=MAXT
             ENDIF
             IF ((MAXT/=MAXBAND).OR.(MINT/=MINBAND).OR.(MINBAND/=MAXBAND+1)) THEN
                IF (IU6>=0) WRITE(IU6,*) 'GWDYN: seems to be a metal? Set AVGAP=0'
                IF (IU0>=0) WRITE(IU0,*) 'GWDYN: seems to be a metal? Set AVGAP=0'
                AVGAP=0._q
                GOTO 10
             ENDIF
          ENDDO
       ENDDO
       DO ISP=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS
             AVGAP=AVGAP+KPOINTS%WTKPT(NK)*(W%CELTOT(MINBAND,NK,ISP)- &
                  W%CELTOT(MAXBAND,NK,ISP))
          ENDDO
       ENDDO
       CONTINUE
       AVGAP_LAST=AVGAP
       AVGAP=0._q                   ! I recommend this default value
! if we do not treat bulk it might be more convenient to provide it in INCAR
       LOPEN=IO%LOPEN
!-----------------------------------------------------------------------
! read in required parameters
!-----------------------------------------------------------------------
10     CONTINUE
       CALL RDATAB(LOPEN,INCAR,IU5,'AVGAP','=','#',';','F', &
                  &               IDUM,AVGAP,CDUM,LDUM,CHARAC,N,1,IERR)
       IF (((IERR/=0).AND.(IERR/=3)).OR. &
            &                     ((IERR==0).AND.(N<1))) THEN
          WRITE(IU0,*)'Error reading item ''AVGAP'' from file INCAR.'
          AVGAP=0._q
          IF (IU0>=0) &
               WRITE(IU0,*)'In order to save things I use the default',AVGAP
       ENDIF
       IF (AVGAP<-1.E-3_q) AVGAP=AVGAP_LAST  ! automatic AVGAP for input < 0
       IF (IU6>=0) WRITE(IU6,*) 'GWDYN: Using an average gap of ',AVGAP
       IF (IU0>=0) WRITE(IU0,*) 'GWDYN: Using an average gap of ',AVGAP
    ENDIF
    ! now get Sigma_dyn and the beta_nk
    AVGAP=-0.5_q*AVGAP/RYTOEV

    ! testing and debugging (on serial machines only!)
    ! Has been commented by MS
    !      lwrite=.TRUE.
    !      IF (lwrite) THEN
    !         IUNIT=NXTFRU()
    !         OPEN(UNIT=IUNIT,FILE='DYNSM1MAT',FORM='FORMATTED',STATUS='UNKNOWN')
    !         WRITE(IUNIT,'(4I5)') T_INFO%NIONS,T_INFO%NIONS,1,INFO%ISPIN
    !         WRITE(IUNIT,'(5E15.7)') &
    !     &         1._q,((LATT_CUR%ANORM(I)*1E-10_q),I=1,3),1E-15_q
    !         WRITE(IUNIT,*) 0._q
    !         WRITE(IUNIT,*) ' CAR '
    !         WRITE(IUNIT,*) INFO%SZNAM1
    ! MS commented this line	 
    !      WRITE(IUNIT,'(3I5)') NINT(INFO%NELECT),KPOINTS%NKPTS_GW,WDES%NB_TOT
    !   ENDIF

    ! MS: Not clear at present what are these GW k-points, so the loop is chaged slightly
    !  DO NK=WDES%NKPTS-WDES%NKPTS_GW+1,WDES%NKPTS
 
    DO NK=1,WDES%NKPTS

       ! testing and debugging (on serial machines only!)
       ! Commented by MS
       !       IF (lwrite) THEN
       !         WRITE(IUNIT,*)
       !        WRITE(IUNIT,'(4E15.7)') WDES%VKPT(1,NK),WDES%VKPT(2,NK), &
       !                                 WDES%VKPT(3,NK),WDES%WTKPT(NK)
       !       ENDIF
       DO N=1,WDES%NB_TOT
          DO ISP=1,WDES%ISPIN
             ! first get the model dielectric function parameters
             ! rho_eff in a.u. (I hope it is correct)

             RHOEFF=AUTOA*AUTOA*AUTOA*RHOMAT(N,NK,ISP)

             ! omega_p^2 in a.u.
             OMEGAP2=4._q*PI*RHOEFF
             ! rhoeff^(1/3)
             RHOEFF13=EXP(LOG(RHOEFF)*.3333333333333333_q)
             ! q_TF^2 in a.u.
             QTF2=FACQ2*RHOEFF13
             CC=0._q

             ! the following does not work yet since I do not get correct occupations in
             ! all cases (depends on k-points and smearing width)
             !         IF (W%FERWE(N,NK,ISP)<1.E-10_q) CC=AVGAP
             ! help myself with this ...
             CC=AVGAP
             IF (N<=NINT(0.25_q+INFO%NELECT/2._q)) CC=0.5_q*AVGAP
             IF (N<=NINT(INFO%NELECT/2._q)) CC=0._q
             ! or is this the really perfect stuff?
             !         CC=(1._q-W%FERWE(N,NK,ISP))*AVGAP
             DQ=0.01_q*SQRT(QTF2)
             A=0._q
             B=0._q
             ! q-integral (extremely primitive - just taken from a program of Giancarlo)
             DO I=1,10001
                FAC=1._q
                IF ((I==1).OR.(I==10001)) FAC=0.5_q
                QI=(I-1)*DQ
                Q2=QI*QI
                Q4=Q2*Q2
                ! this is valid for cubic systems only !!

               ! OQ2=OMEGAP2*(MODEL_GW_EPS0/(MODEL_GW_EPS0-1._q)+(Q2/QTF2))+.25_q*Q4
               ! The expression corrected by mS:   
                OQ2=OMEGAP2*(MODEL_GW_EPS0/(MODEL_GW_EPS0-1._q)+MODEL_GW_ALPHA*(Q2/QTF2))+.25_q*Q4
                OQ=SQRT(OQ2)
                OQC=(OQ+0.5_q*Q2+CC)
                F=Q2/(OQ2*OQC)
                A=A+F*FAC
                BB=1._q/(OQ*OQC*OQC)
                B=B+BB*FAC
             ENDDO
             A=A*DQ*OMEGAP2*RYTOEV/PI
             B=B*DQ*OMEGAP2/PI
             S=1._q/(B+1._q)

             ! testing and debugging (on serial machines only!)
             !        IF (lwrite) THEN
             !          WRITE(IUNIT,'(1X,I3,4X,F10.4,2X,F10.4,2X,F10.4)') N,A,B,S
             !         ENDIF
             ! MS changed this line to the new array E_dyn (needs to be redone)
             !       W%CQUASI(N,NK,ISP)=(W%CQUASI(N,NK,ISP)+A)
             E_SIGMA(N,NK,ISP)=W%CELTOT(N,NK,ISP)
             Z(N,NK,ISP)=S

             W%CELTOT(N,NK,ISP)= ((W%CELTOT(N,NK,ISP)-E_DFT(N,NK,ISP))+A)*S+E_DFT(N,NK,ISP)
             !    W%CELTOT(N,NK,ISP)= ((W%CELTOT(N,NK,ISP)-E_DFT(N,NK,ISP))+A)*S
             !         W%CELTOT(N,NK,ISP)=(W%CELTOT(N,NK,ISP)+A)*S
             !   W%CELTOT(N,NK,ISP)=A*S
             !       S = A*S 
          ENDDO
       ENDDO
    ENDDO
    
    WRITE(IO%IU6,*) 
    WRITE(IO%IU6,*) 'Calculated quasiparticle energies'
    WRITE(IO%IU6,*) 

    DO ISP=1,WDES%ISPIN
       IF (WDES%ISPIN==2) WRITE(IO%IU6,'(/A,I1)') ' spin component ',ISP
       DO N=1,KPOINTS%NKPTS
          WRITE(IO%IU6,3) N,WDES%VKPT(1,N),WDES%VKPT(2,N),WDES%VKPT(3,N)
! Attempt to output the calculated dynamical corrections by MS
          DO I=1,WDES%NB_TOT
             WRITE(IO%IU6,4)  &
                  I,E_DFT(I,N,ISP),REAL(W%CELTOT(I,N,ISP)),E_SIGMA(I,N,ISP), & 
                  Z(I,N,ISP),W%FERTOT(I,N,ISP)*W%WDES%RSPIN
          ENDDO
       ENDDO
    ENDDO
3   FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
          &         "  band No. DFT-energies  QP-energies   Sigma(DFT)    Z            occupation"/)
4   FORMAT((3X,I4,3X,7(F10.4,3X)))

    RETURN
  END SUBROUTINE GWDYNSM1_IJ


END MODULE gw_model


!*******************************************************************
!
!  get the extremely oversimplified COH potential on the real space
!  grid (result returned in SIGMA)
!  the routine is called from pot.F
!
!  written by Juergen Furthmueller
!  modified by gK
!
!*******************************************************************

  SUBROUTINE COHSM1_RGRID(ISPIN,CHDENR,SIGMA,DENCOR,GRID,OMEGA,LAECORE)
    USE prec
    USE constant
    USE mgrid
    USE gw_model
    USE setexm

    IMPLICIT NONE

    TYPE (grid_3d) GRID

    ! global
    INTEGER        ISPIN
    LOGICAL        LAECORE
    REAL(q)        OMEGA
    RGRID          CHDENR(GRID%RL%NP)
    RGRID          SIGMA(DIMREAL(GRID%MPLWV),ISPIN)
    RGRID          DENCOR(GRID%RL%NP)
    ! local
    INTEGER        N
    REAL(q)        A3BOM,EPS0M1,FAC1,FAC2,FACQ3BKF2,FACQ2,QTF2,Q3BYKF2
    REAL(q)        DENS,RHO,RHO16,RINPL,FACPAR
    REAL(q)        SUM

    IF (LEXCH/=100) RETURN

    ! use (partial) core charge or not ...
    FACPAR=0._q
    IF (LAECORE) FACPAR=1._q
    ! scaling factor for conversion from CHDENR to true charge density in a.u.
    A3BOM=AUTOA*AUTOA*AUTOA/OMEGA
    EPS0M1=MODEL_GW_EPS0-1._q
    ! e^2/2 sqrt(1-1/eps_0)/sqrt(alpha)  [in eV units; 1/2 absorbed in Hartree - Ry]
    FAC1 = -RYTOEV*SQRT(EPS0M1/(MODEL_GW_EPS0*MODEL_GW_ALPHA))
    ! sqrt(3*eps_0/(eps_0-1))/alpha      [dimensionless]
    FAC2 = SQRT(3._q*MODEL_GW_EPS0/EPS0M1)/MODEL_GW_ALPHA
    ! conversion factor between q_TF**2 [a.u.] and rho [a.u.]
    FACQ2     = 4._q*EXP(LOG(3._q/PI)/3._q)
    ! conversion factor between q_TF**3/k_F  [a.u.] and rho [a.u.] times FAC2)
    FACQ3BKF2 = 8._q*SQRT(3._q/PI)*EXP(-LOG(3._q*PI*PI)/3._q)*FAC2
    SUM=0
!DIR$ IVDEP
!cdir nodep
!$DIR FORCE_VECTOR
!OCL NOVREC
    DO N=1,GRID%RL%NP
       DENS=REAL( CHDENR(N) , KIND=q) + FACPAR*DENCOR(N)
       ! charge density rho [a.u.] (1/OMEGA is due to the special storage mode of VASP)
       RHO    = ABS(DENS)*A3BOM
       ! avoid floating point errors / special handling for zero charge densities
       RHO    = MAX(RHO,1E-60_q)
       ! rho^(1/6), rho^(1/3)
       RHO16  = EXP(LOG(RHO)*0.1666666666666667_q)
       ! q_TF^2
       QTF2   = FACQ2*RHO16*RHO16
       ! FAC2 * q_TF^3/k_F
       Q3BYKF2 = FACQ3BKF2*RHO16
       ! sigma as proposed by Giancarlo (however, formula rewritten by jF)
       SIGMA(N,1) = (FAC1 * QTF2 / SQRT(QTF2 + Q3BYKF2))
       IF (ISPIN==2) THEN
          SIGMA(N,2) = (FAC1 * QTF2 / SQRT(QTF2 + Q3BYKF2))
       ENDIF
          
       SUM=SUM+REAL( CHDENR(N) , KIND=q)*(FAC1 * QTF2 / SQRT(QTF2 + Q3BYKF2))
    ENDDO

!    WRITE(*,*) 'total Coulomb hole energy',SUM/GRID%NPLWV
    RETURN
  END SUBROUTINE COHSM1_RGRID

!*******************************************************************
!
!  calculate the extremely oversimplified COH potential from the
!  supplied charge density
!
!  written by gK (essentially a copy of the previous routine)
!  this routine is called from radial.F via xclib.F
!
!*******************************************************************

  SUBROUTINE COHSM1(CHDEN,SIGMA)
    USE prec
    USE constant
    USE mgrid
    USE fock
    USE setexm

    IMPLICIT NONE
    REAL(q)        CHDEN, SIGMA
    REAL(q)        A3BOM,EPS0M1,FAC1,FAC2,FACQ3BKF2,FACQ2,QTF2,Q3BYKF2,RHO, RHO16

    ! scaling factor for conversion from CHDEN to a.u.
    A3BOM=AUTOA*AUTOA*AUTOA
    EPS0M1=MODEL_GW_EPS0-1._q
    ! e^2/2 sqrt(1-1/eps_0)/sqrt(alpha)  [in eV units; 1/2 absorbed in Hartree - Ry]
    FAC1 = -RYTOEV*SQRT(EPS0M1/(MODEL_GW_EPS0*MODEL_GW_ALPHA))
    ! sqrt(3*eps_0/(eps_0-1))/alpha      [dimensionless]
    FAC2 = SQRT(3._q*MODEL_GW_EPS0/EPS0M1)/MODEL_GW_ALPHA
    ! conversion factor between q_TF**2 [a.u.] and rho [a.u.]
    FACQ2     = 4._q*EXP(LOG(3._q/PI)/3._q)
    ! conversion factor between q_TF**3/k_F  [a.u.] and rho [a.u.] times FAC2)
    FACQ3BKF2 = 8._q*SQRT(3._q/PI)*EXP(-LOG(3._q*PI*PI)/3._q)*FAC2

    ! charge density rho [a.u.] (1/OMEGA is due to the special storage mode of VASP)
    RHO    = ABS(CHDEN)*A3BOM
    ! avoid floating point errors / special handling for zero charge densities
    RHO    = MAX(RHO,1E-60_q)
    ! rho^(1/6), rho^(1/3)
    RHO16  = EXP(LOG(RHO)*0.1666666666666667_q)
    ! q_TF^2
    QTF2   = FACQ2*RHO16*RHO16
    ! FAC2 * q_TF^3/k_F
    Q3BYKF2 = FACQ3BKF2*RHO16
    ! sigma as proposed by Giancarlo (however, formula rewritten by jF)
    SIGMA=(FAC1 * QTF2 / SQRT(QTF2 + Q3BYKF2))

    RETURN
  END SUBROUTINE COHSM1


!*******************************************************************
!
!  calculate the state dependent diagonal screened Coulomb operator
!
!*******************************************************************

  SUBROUTINE MODEL_GW_SET_GFAC(GRID, LATT_CUR, NK, NQ, NQ_IRZ, MK, MQ, ISP, ISP_IRZ, FSG, POTFAK)
    USE prec
    USE constant
    USE mgrid
    USE nonl_high
    USE lattice
    USE full_kpoints
    USE gw_model
    IMPLICIT NONE

    TYPE (grid_3d) GRID
    TYPE (latt) LATT_CUR
    INTEGER NK, NQ       ! k and q point
    INTEGER NQ_IRZ       ! q index in the IRZ
    INTEGER MK, MQ       ! band indices
    INTEGER ISP, ISP_IRZ ! spin component
    REAL(q) :: FSG
    REAL(q) :: POTFAK(GRID%MPLWV)
  ! local
    INTEGER    NI,NC,N1,N2,N3
    REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,GSQUP,GQUAD,SCALE, RHOEFF, & 
         FOMEGAP2, QTF2, RHOEFF13

    ! effective electron density in a.u.^-3
    RHOEFF=AUTOA*AUTOA*AUTOA*SQRT(RHOMAT(MK, NK, ISP))*SQRT(RHOMAT(MQ, NQ_IRZ, ISP_IRZ))
    ! plasma frequency
    FOMEGAP2=16._q*PI*RHOEFF/(AUTOA*AUTOA*AUTOA*AUTOA)

    RHOEFF13=EXP(LOG(RHOEFF)*.3333333333333333_q)
    ! q_TF^2 (divided by AUTOA^2 which is the conversion factor for q^2):
    QTF2=(4._q*EXP(LOG(3._q/PI)/3._q))*RHOEFF13*1._q/(AUTOA*AUTOA)

    SCALE=KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ*EDEPS/LATT_CUR%OMEGA/TPI**2*(1.0_q/GRID%NPLWV)
#ifndef gammareal
    DKX=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(1,1)+ &
         (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(1,2)+ &
         (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(1,3)
    DKY=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(2,1)+ &
         (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(2,2)+ &
         (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(2,3)
    DKZ=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(3,1)+ &
         (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(3,2)+ &
         (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(3,3)
#else
    DKX=0._q; DKY=0._q; DKZ=0._q
#endif
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
          IF ((GRID%LPCTX(N1)==0).AND.(GRID%LPCTY(N2)==0).AND.&
               (GRID%LPCTZ(N3)==0) .AND. (NK==NQ)) THEN
             POTFAK(NI)=FSG*(1.0_q/GRID%NPLWV)
          ELSE
             GSQUP=GSQU*(TPI*TPI)
             GQUAD=GSQUP*GSQUP
             POTFAK(NI)=SCALE/(GSQU)/ &
                  (1+1/(1/(MODEL_GW_EPS0-1)+GSQUP*(MODEL_GW_ALPHA/QTF2)+GQUAD*(1/FOMEGAP2)))
          ENDIF
       ENDDO row
    ENDDO col
  END SUBROUTINE MODEL_GW_SET_GFAC
