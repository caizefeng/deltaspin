#include "symbol.inc"

MODULE mlr_main
  USE prec
  USE vaspxml
  USE lr_helper
  IMPLICIT NONE

  REAL(q),ALLOCATABLE,PRIVATE :: POSION_LAST(:,:)
  REAL(q),ALLOCATABLE,PRIVATE :: BORN_CHARGES_SAVE(:,:,:)
  REAL(q),ALLOCATABLE,PRIVATE :: DIPOLE_TIME(:,:)
CONTAINS

!*********************************************************************
!
! calculate second derivatives using linear response theory
!
! implemented by gK
! the main scheduler is based on the finite difference kernel of 
! Orest Dubay (at least the skeleton is essentially identical)
! it steps through all ions (at least those that are allowed to move)
! and calls the main linear response kerner to calculate the
! linear response of the wavefunction and the second derivatives
! with respect to the ionic positions
!
!*********************************************************************

  SUBROUTINE LR_SKELETON( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR,LATT_INI, &
          T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI,CHAM, &
          IBRION,LMDIM,IRDMAX,NEDOS, &
          TOTENIN,EFERMI,FORCE,LFAST)

    USE ini
    USE base
    USE lattice
    USE finite_differences
    USE charge
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE msymmetry
    USE mpimy
    USE mgrid
    USE mkpoints
    USE constant
    USE poscar
    USE wave
    USE pot
    USE subrot
    USE pawm
    USE rmm_diis
    USE choleski
    USE david
    USE wave_high
    USE mlrf_main
    USE lri_main
    USE subrot_cluster
    USE kpoints_change
    USE full_kpoints
    USE hamil_high
    USE pead
    USE fock
    USE morbitalmag
    USE meta
    USE wannier_interpolation
    USE crpa, ONLY : LCRPAPLOT
    USE rpa_force, ONLY : STORE_RPA_SECOND_DERIV
! solvation__
    USE solvation
! solvation__
#ifdef PROFILING
    USE profiling
#endif
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (tau_handle)  KINEDEN
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W          ! wavefunction
    TYPE (latt)        LATT_CUR
    TYPE (dynamics)    DYN
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (mixing)      MIX
    TYPE (kpoints_struct) KPOINTS
    TYPE (symmetry)    SYMM
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (grid_3d)     GRIDB      ! Broyden grid
    TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE (energy)      E
    TYPE (latt)        LATT_INI

    INTEGER IBRION
    INTEGER LMDIM,IRDMAX,IRDMAA,NEDOS
    REAL(q) TOTENIN,EFERMI

    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q)  CHTOTL(GRIDC%MPLWV,WDES%NCDIJ)! old charge-density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ),RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
!  density of states
    REAL(q)    DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
!  Hamiltonian
    GDEF       CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
    REAL(q)   XCSIF(3,3)
    REAL(q) :: FORCE(3,T_INFO%NIONS)   ! forces in cartesian coordinates
    LOGICAL, OPTIONAL :: LFAST

! local variables related to finite difference code

    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
    INTEGER :: NIONS
    INTEGER :: IU6                ! OUTCAR file
    INTEGER :: IU6K               ! OUTCAR k-points
    INTEGER :: IU0                ! stdout

    TYPE (type_info)   T_INFO_0
    INTEGER :: IUDYNMAT           ! DYNMAT file
    REAL(q) :: X
    REAL(q),ALLOCATABLE      :: INITIAL_FORCE(:,:)
    REAL(q),ALLOCATABLE      :: DISPL_FORCES(:,:,:)
    REAL(q),ALLOCATABLE      :: INT_STRAIN(:,:,:), PIEZO(:,:,:)
    REAL(q),ALLOCATABLE      :: BORN_CHARGES(:,:,:), BORN_CHARGES2(:,:,:)
    REAL(q),ALLOCATABLE      :: SECOND_DERIV(:,:)
    INTEGER                  :: DOF
    INTEGER                  :: PROCESSED_DISPL
    INTEGER                  :: I,J,K,M,N, IDIR, IDISPL_ON_ION
    REAL(q),ALLOCATABLE      :: WORK(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVECTORS(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVALUES(:)
    INTEGER                  :: IERROR
    REAL(q) :: EPSILON(3,3),  ELASTIC(3,3,3,3),  ELASTICP(6,3,3)
! variables for the reevaluation of energy
    REAL(q) DESUM, RMS       ! change
    REAL(q) TOTEN, TOTEN_    ! energy
    REAL(q) EDIFF            ! break condition
    INTEGER ICOUEV
    INTEGER NSIM
! linear response using symmetry
    INTEGER                  :: NKORIG
    REAL(q),ALLOCATABLE      :: D(:,:,:)
    INTEGER,ALLOCATABLE      :: ND(:), IDIRD(:,:)
    REAL(q)                  :: WORKD(3,3,T_INFO%NIONS)
    INTEGER                  :: IWORK(T_INFO%NIONS)
    CHARACTER(3)             :: IDIR_TEXT(3)=(/"x","y","z"/)
    LOGICAL                  :: LDO(T_INFO%NIONS)
    REAL(q)                  :: WDMAT(3,T_INFO%NIONS,3,T_INFO%NIONS),DMAT(3,3,T_INFO%NIONS,T_INFO%NIONS)
    REAL(q)                  :: ST(3,3,3,T_INFO%NIONS), AST(3,3,3,T_INFO%NIONS)
    REAL(q)                  :: FACT, TMP(3,3)
! variables to store G [H,r] phi
    COMPLEX(qs), ALLOCATABLE :: RPHI(:,:,:,:,:)
    GDEFS, ALLOCATABLE ::  RPHI_CPROJ(:,:,:,:,:)
    TYPE (skpoints_trans)   :: KPOINTS_TRANS

! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

    ! NULLIFY to make sure a later "IF(ASSOCIATED())" works correctly.
    NULLIFY(KPOINTS_TRANS%CPHASE)

#if defined(VASP2WANNIER90) && ! defined(gammareal)
    IF (LWANNIERINTERPOL.OR.LCRPAPLOT) THEN
       CALL WANNIER_INTERPOLATE(HAMILTONIAN,KPOINTS,GRID,LATT_CUR,LATT_INI,&
            T_INFO,NONLR_S,NONL_S,W,LMDIM,P,SV,CQIJ,CDIJ,SYMM,INFO,IO)
       ! at this point the wave functions are messed up, so let's call it a day ...
#ifdef PROFILING
       ! started in INIT_PROFILING
       PROFILING_STOP('total_time')
       IF (IO%IU0>=0) THEN
          CALL PRINT_FULL_PROFILE(IO%IU6,PFLAT=.FALSE.)
       ENDIF
#endif
       CALL DUMP_ALLOCATE(IO%IU6)
       CALL DUMP_FINAL_TIMING(IO%IU6)
       CALL STOP_XML
       CALLMPI_C(M_exit())
    ENDIF
#endif

    IF (KINTER<0) THEN
       IF (LHFCALC) THEN
          CALL INTERPOLATE_BAND_STR(HAMILTONIAN, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
            T_INFO, NONLR_S, NONL_S, W, NEDOS, DOS, DOSI, INFO%NELECT, INFO%NUP_DOWN,  &
            LMDIM, P, SV, CQIJ, CDIJ, SYMM, IO%IU0, IO%IU6)
       ELSE
          CALL INTERPOLATE_BANDSTR_GRDSHFT(HAMILTONIAN, E, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
            T_INFO, NONLR_S, NONL_S, W, NEDOS, &
            LMDIM, P, SV, CQIJ, CDIJ, SYMM, INFO, IO, GRIDC, GRIDUS, C_TO_US, IRDMAX)
       ENDIF
       RETURN
    ENDIF

    IF (LHFCALC .AND. AEXX/=0 ) THEN
       CALL VTUTOR('E','LHFLINEARRESPONSE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU6,3)
       CALL VTUTOR('S','LHFLINEARRESPONSE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU0,3)
    ENDIF

    IF (LDO_METAGGA()) THEN
       CALL VTUTOR('E','METAGGARESPONSE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU6,3)
       CALL VTUTOR('S','METAGGARESPONSE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU0,3)
    ENDIF

    IF (INFO%LREAL) THEN
       CALL VTUTOR('W','LINEARRESPONSE LREAL',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU6,3)
       CALL VTUTOR('W','LINEARRESPONSE LREAL',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IO%IU0,3)
    ENDIF

    NIONS = T_INFO%NIONS
    IU6   = IO%IU6
    IU0   = IO%IU0

    IF (IO%NWRITE>=3) THEN
       IU6K  = IO%IU6
    ELSE
       IU6K  = -1
    ENDIF

    IF (IBRION>0) THEN
       IF (IBRION==8) THEN
          ALLOCATE(D(3,3,T_INFO%NIONS), ND(T_INFO%NIONS),IDIRD(3,T_INFO%NIONS))

          CALL FREDOM(SYMM%ROTMAP,ISYMOP,INVMAP,NROTK,NPCELL,D,ND,1,T_INFO%NTYP,T_INFO%NIONS, &
             T_INFO%NITYP,WORKD,IWORK, & 
             LATT_CUR%A(1,1),LATT_CUR%A(1,2),LATT_CUR%A(1,3), &
             LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),IDIRD,.FALSE.)
          DOF=SUM(ND)

          IF (IU6>=0) THEN
             WRITE(IU6,*)
             WRITE(IU6,'(A,I5,A)') ' Found ',DOF,' degrees of freedom:'
             WRITE(IU6,*)' ----------------------------------------------'
             WRITE(IU6,*)
             DO J=1,NIONS
                IF (ND(J).GT.0) THEN
                   WRITE(IU6,'(A,I5,A,3A2)') '     directions for atom ',J,':  ',IDIR_TEXT(IDIRD(ND(J),J))
                ENDIF
             ENDDO
          ENDIF
       ELSE IF (IBRION==7) THEN
          CALL COUNT_DOF(NIONS, T_INFO%LSFOR, T_INFO%LSDYN, DOF)
       ELSE
          DOF=0
       ENDIF
    ELSE
       DOF=0
    ENDIF

    T_INFO_0=T_INFO
    NULLIFY(T_INFO_0%POSION)
    ALLOCATE(T_INFO_0%POSION(3,NIONS))
    ALLOCATE(INITIAL_FORCE(3,NIONS))
    ALLOCATE(DISPL_FORCES(DOF,3,NIONS),BORN_CHARGES(3,3,NIONS),BORN_CHARGES2(3,3,NIONS), & 
      PIEZO(3,3,3),INT_STRAIN(DOF,3,3))

    DISPL_FORCES=0
    BORN_CHARGES=0 ; BORN_CHARGES2=0
    PIEZO=0
    INT_STRAIN=0

    T_INFO_0%POSION             = T_INFO%POSION
    INITIAL_FORCE               = FORCE
    NKORIG                      = WDES%NKPTS
!=======================================================================
! reset the potential and recalculate the ground state wavefunctions
! with very high precision for this setup
!=======================================================================
    IF (IU0>=0) THEN
       WRITE (IU0,*) 'Linear response reoptimize wavefunctions to high precision'
       WRITE ( 17,*) 'Linear response reoptimize wavefunctions'
    ENDIF

    IF (IU6>=0) THEN
       WRITE (IU6,*) 'Linear response reoptimize wavefunctions'
    ENDIF

    IF (INFO%LREAL) THEN
       CALL RSPHER(GRID,NONLR_S,LATT_CUR)
    ENDIF

    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES,  &
         INFO,P,T_INFO,E,LATT_CUR, &
         CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)

!   CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C,  WDES%COMM_INTER, & 
!        LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)
                  
    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

    CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)
    
    CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ) 
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
         E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    !
    ! only if the tolerance is very tight for the occupied states
    ! it is possible to attain a tight tolerance for the perturbed states
    ! with the present DIIS algorithm

    EDIFF = 1E-10
    NSIM=WDES%NSIM*2
#ifdef MPI
    NSIM=((WDES%NSIM*2+WDES%COMM_INTER%NCPU-1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif

    X=INFO%EBREAK

    ! for some non obvious reasons 
    IF (.NOT. LMAGBLOCH) THEN 
       INFO%EBREAK=0.25*EDIFF
    ELSE
       IF (IU0>=0) WRITE(IU0,*) 'WARNING: EBREAK remains at default, EDDAV has a problem for too tight convergence criteria'
       INFO%EBREAK=EDIFF
    ENDIF

    TOTEN=TOTENIN
    DO I=1,3
       CALL EDDAV(HAMILTONIAN,P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
            LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV, E%EXHF, IO%IU6,IO%IU0, .FALSE., .TRUE., .FALSE.)

       E%EBANDSTR=BANDSTRUCTURE_ENERGY(WDES, W)
       TOTEN_=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+Ediel_sol

       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,1000) I, TOTEN_, TOTEN_-TOTEN, DESUM, ICOUEV, RMS
          WRITE(17,1000)     I, TOTEN_, TOTEN_-TOTEN, DESUM, ICOUEV, RMS
       ENDIF
1000   FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)
       IF(ABS(DESUM)<EDIFF.AND.ABS(TOTEN_-TOTEN)<EDIFF) EXIT

       TOTEN=TOTEN_
    ENDDO
    CALL KPAR_SYNC_ALL(WDES,W)

    INFO%EBREAK=X    ! restore the break condition

    IF (IO%IU0>=0 .AND. IO%LOPEN) CALL WFORCE(17)

!=======================================================================
! determine G [H, r] |phi> = d/ dk | phi(k)>
!=======================================================================
    NULLIFY(DEG_CLUSTER)
    CALL FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER)

    IF (LEPSILON .OR. KINTER/=0 .OR. LMAGBLOCH ) THEN

    CALL SET_NABIJ_AUG(P,T_INFO%NTYP)
    ALLOCATE(RPHI(WDES%NRPLWV,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN,3), &
             RPHI_CPROJ(WDES%NPROD,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN,3))

    IF (LUSEPEAD()) THEN
       IF (LMAGBLOCH) THEN
          LPEAD_RETURN_Q_CPROJ=.FALSE.
       ENDIF
       RPHI=0
       RPHI_CPROJ=0
       CALL PEAD_DPSI_DK_ALL(W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)
       LPEAD_RETURN_Q_CPROJ=.TRUE.
    ELSE 
       RPHI=0
       RPHI_CPROJ=0

       DO IDIR=1,3
          IF (IU0>=0) THEN
             WRITE (IU0,*)'Linear response G [H, r] |phi>, progress :'
             WRITE (IU0,'(A,I3)') &
                  '  Direction: ',IDIR
             WRITE (17,*)'Linear response G [H, r] |phi>, progress :'
             WRITE (17,'(A,I3)') &
                  '  Direction: ',IDIR
          END IF

          IF (IU6>=0) THEN
             WRITE (IU6,*)'Linear response G [H, r] |phi>, progress :'
             WRITE (IU6,'(A,I3)') &
                  '  Direction: ',IDIR
          ENDIF

          CALL LRF_RPHI( &
             P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
             T_INFO,INFO,IO,GRID,GRIDC,GRIDUS,C_TO_US,IRDMAX, &
             CDIJ,CQIJ,SV,LMDIM,DEG_CLUSTER, IDIR, RPHI(:,:,:,:,IDIR), RPHI_CPROJ(:,:,:,:,IDIR), & 
             .TRUE.) !KINTER==0)

          IF (IO%LOPEN) CALL WFORCE(IO%IU6)
       ENDDO
    ENDIF

    IF (KINTER>0) THEN
!       CALL INTERPOLATE_BAND_STR(HAMILTONIAN, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
!            T_INFO, NONLR_S, NONL_S, W, NEDOS, DOS, DOSI, INFO%NELECT, INFO%NUP_DOWN, &
!            LMDIM, P, SV, CQIJ, CDIJ, SYMM, IO%IU0, IO%IU6, RPHI)
       CALL INTERPOLATE_BANDSTR_GRDSHFT( HAMILTONIAN, E, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
            T_INFO, NONLR_S, NONL_S, W, NEDOS, &
            LMDIM, P, SV, CQIJ, CDIJ, SYMM, INFO, IO, GRIDC, GRIDUS, C_TO_US, IRDMAX, RPHI)

       ! well this deallocates most of the quantities we have used, but not all
       DEALLOCATE(RPHI, RPHI_CPROJ)
       RETURN
    ENDIF

    IF (LMAGBLOCH) THEN

       CALL BLOCH_CURRENT( W, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, P, LATT_CUR, & 
          HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT, CHTOT, NONLR_S, NONL_S, & 
          RPHI, RPHI_CPROJ, CDIJ, CQIJ, SV, EFERMI, &
          T_INFO, LMDIM, CRHODE, IRDMAX, IO%IU6, IO%IU0)

       ! well this deallocates most of the quantities we have used, but not all
       DEALLOCATE(RPHI, RPHI_CPROJ)
       RETURN
    ENDIF
!=======================================================================
! response to external excluding local field effects
! i.e. in the independent particle approximation
!=======================================================================
    IF (.NOT. PRESENT (LFAST)) THEN
    DO IDIR=1,3

       IF (IU0>=0) THEN
          WRITE (IU0,*)'Linear response to external field (no local field effect), progress :'
          WRITE (IU0,'(A,I3)') &
               '  Direction: ',IDIR
          WRITE (17,*)'Linear response to external field (no local field effect), progress :'
          WRITE (17,'(A,I3)') &
               '  Direction: ',IDIR
       END IF
     
       IF (IU6>=0) THEN
          WRITE (IU6,*)'Linear response to external field (no local field effect), progress :'
          WRITE (IU6,'(A,I3)') &
               '  Direction: ',IDIR
       ENDIF

       IF (SYMM%ISYM>0) THEN
          DYN%VEL=0
          DYN%VEL(IDIR,:)=1.0
          CALL KARDIR(T_INFO%NIONS,DYN%VEL,LATT_CUR%B)
          CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
               T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
               SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
               SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
       ENDIF

       CALL LRF_MAIN( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,RPHI,RPHI_CPROJ, &
          LATT_CUR, &
          T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI, &
          LMDIM,IRDMAX,NEDOS, &
          TOTEN,EFERMI, DEG_CLUSTER, KPOINTS_TRANS, EPSILON(IDIR,:), BORN_CHARGES(IDIR,:,:), PIEZO(IDIR,:,:),  & 
          IDIR, .FALSE.)

          IF (IO%LOPEN) CALL WFORCE(IO%IU6)
    ENDDO
    ! reinitialise symmetry
    IF (SYMM%ISYM>0) THEN
       DYN%VEL=0
       CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
            T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
            SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
            SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
    ENDIF

    IF (SYMM%ISYM>0) CALL TSYM(EPSILON,ISYMOP,NROTK,LATT_CUR%A)
    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,1110) '(INDEPENDENT PARTICLE, excluding Hartree and local field effects)',EPSILON
       WRITE(IO%IU6,130)
       CALL XML_TENSOR("epsilon_rpa",EPSILON)
    ENDIF
    ENDIF
!=======================================================================
! response to external field including local field effects
! if do_kpoints_chage is not set the k-point set remains unchanged
! and symmetrization of the final tensor is performed
! results are not exact but errors are sometimes small (not recommended)
!=======================================================================
#define do_kpoints_change
#ifndef do_kpoints_change
    DO IDIR=1,3

       IF (IU0>=0) THEN
          WRITE (IU0,*)'Linear response to external field, progress :'
          WRITE (IU0,'(A,I3)') &
               '  Direction: ',IDIR
          WRITE (17,*)'Linear response to external field, progress :'
          WRITE (17,'(A,I3)') &
               '  Direction: ',IDIR
       END IF
     
       IF (IU6>=0) THEN
          WRITE (IU6,*)'Linear response to external field, progress :'
          WRITE (IU6,'(A,I3)') &
               '  Direction: ',IDIR
       ENDIF

       IF (SYMM%ISYM>0) THEN
          DYN%VEL=0
          DYN%VEL(IDIR,:)=1.0
          CALL KARDIR(T_INFO%NIONS,DYN%VEL,LATT_CUR%B)
          CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
               T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
               SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
               SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
       ENDIF

       CALL LRF_MAIN( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,RPHI,RPHI_CPROJ, &
          LATT_CUR, &
          T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI, &
          LMDIM,IRDMAX,NEDOS, &
          TOTEN,EFERMI, DEG_CLUSTER, KPOINTS_TRANS, EPSILON(IDIR,:), BORN_CHARGES(IDIR,:,:), PIEZO(IDIR,:,:),  & 
          IDIR, .TRUE.)

          IF (IO%LOPEN) CALL WFORCE(IO%IU6)
    ENDDO
    ! reinitialise symmetry
    IF (SYMM%ISYM>0) THEN
       DYN%VEL=0
       CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
            T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
            SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
            SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
    ENDIF

    IF (SYMM%ISYM>0) CALL TSYM(EPSILON,ISYMOP,NROTK,LATT_CUR%A)
    IF (IO%IU6>=0) THEN
       IF (LRPA) THEN
          WRITE(IO%IU6,1100) '(including local field effects in RPA (Hartree))',EPSILON
       ELSE
          WRITE(IO%IU6,1100) '(including local field effects in DFT)',EPSILON
       ENDIF
       WRITE(IO%IU6,130)
       CALL XML_TENSOR("epsilon",EPSILON)
    ENDIF

!=======================================================================
! response to external field including local field effects
! this version changes the k-point set on the fly
! and symmetrizes the polarisation vector
!=======================================================================
#else
    DO IDIR=1,3
       IF (IU0>=0) THEN
          WRITE (IU0,*)'Linear response to external field, progress :'
          WRITE (IU0,'(A,I3)') &
               '  Direction: ',IDIR
          WRITE (17,*)'Linear response to external field, progress :'
          WRITE (17,'(A,I3)') &
               '  Direction: ',IDIR
       END IF
     
       IF (IU6>=0) THEN
          WRITE (IU6,*)'Linear response to external field, progress :'
          WRITE (IU6,'(A,I3)') &
               '  Direction: ',IDIR
       ENDIF
       !
       ! reinitialise symmetry part for field in direction IDIR
       ! presently this is done by supplying a velocity field to the ions
       !
       IF (SYMM%ISYM>0) THEN
          DYN%VEL=0
          DYN%VEL(IDIR,:)=1.0
          CALL  KARDIR(T_INFO%NIONS,DYN%VEL,LATT_CUR%B)
          CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
               T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
               SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
               SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
#ifdef oldsym
          CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
               SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
               T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#else
          CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
               SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
               T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#endif
          CALL KPAR_SYNC_ALL(WDES,W)
          CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
          CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS)

          ! Loewdin perturbation theory to improve states at added k-points
          CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
          &    LMDIM,CDIJ,CQIJ,4,SV,T_INFO,P,IO%IU0,DESUM, NKSTART=NKORIG+1)
          CALL KPAR_SYNC_ALL(WDES,W)

          CALL FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER)
       ENDIF

       CALL LRF_MAIN( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,RPHI,RPHI_CPROJ, &
          LATT_CUR, &
          T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI, &
          LMDIM,IRDMAX,NEDOS, &
          TOTEN,EFERMI, DEG_CLUSTER, KPOINTS_TRANS, EPSILON(IDIR,:), BORN_CHARGES(IDIR,:,:), PIEZO(IDIR,:,:), & 
          IDIR, .TRUE.)
       IF (IO%LOPEN) CALL WFORCE(IO%IU6)

       IF (SYMM%ISYM>0) CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS)
    ENDDO

    IF (SYMM%ISYM>0) THEN
       DYN%VEL=0
       CALL KARDIR(T_INFO%NIONS,DYN%VEL,LATT_CUR%B)
       CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
            T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
            SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
            SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
#ifdef oldsym
       CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
            SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
            T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#else
       CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
            SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
            T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#endif
       CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
       CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR)

       CALL TSYM(EPSILON,ISYMOP,NROTK,LATT_CUR%A)
    ENDIF

    IF (IO%IU6>=0) THEN
       IF (LRPA) THEN
          WRITE(IO%IU6,1100) '(including local field effects in RPA (Hartree))',EPSILON
       ELSE
          WRITE(IO%IU6,1100) '(including local field effects in DFT)',EPSILON
       ENDIF
       WRITE(IO%IU6,130)
       CALL XML_TENSOR("epsilon",EPSILON)
    ENDIF
#endif
1100   FORMAT(// &
            " MACROSCOPIC STATIC DIELECTRIC TENSOR ",A/, &
            " ------------------------------------------------------"/, &
            &       3(6X,3F13.6/), &
            " ------------------------------------------------------"/)

1110   FORMAT(// &
            " HEAD OF MICROSCOPIC STATIC DIELECTRIC TENSOR ",A/, &
            " ------------------------------------------------------"/, &
            &       3(6X,3F13.6/), &
            " ------------------------------------------------------"/)


    ENDIF
130 FORMAT (5X, //, &
     &'----------------------------------------------------', &
     &'----------------------------------------------------'//)
!=======================================================================
! ionic displacements
!=======================================================================

    IF (IU0>=0 .AND. DOF>0) THEN
       WRITE (IU0,*) 'Linear response DOF=',DOF
       WRITE (17,*) 'Linear response DOF=',DOF
    ENDIF

    IF (IU6>=0 .AND. DOF>0) THEN
       WRITE (IU6,*) 'Linear response:'
       WRITE (IU6,*) '  Degrees of freedom DOF   = ',DOF
    END IF

    DO PROCESSED_DISPL=1,DOF

       IF (IU0>=0) THEN
          WRITE (IU0,*)'Linear response progress:'
          WRITE (IU0,'(A,I3,A,I3)') &
               '  Degree of freedom: ',PROCESSED_DISPL,'/',DOF
          WRITE (17,*)'Linear response progress:'
          WRITE (17,'(A,I3,A,I3)') &
               '  Degree of freedom: ',PROCESSED_DISPL,'/',DOF
       END IF
     
       IF (IU6>=0) THEN
          WRITE (IU6,*)'Linear response progress:'
          WRITE (IU6,'(A,I3,A,I3)') &
               '  Degree of freedom: ',PROCESSED_DISPL,'/',DOF
       ENDIF
       T_INFO%POSION=T_INFO_0%POSION
       IF (IBRION==8) THEN
          CALL FIND_IJ_ID(.FALSE.,NIONS,PROCESSED_DISPL,ND,J,IDISPL_ON_ION)
          IDIR=IDIRD(IDISPL_ON_ION,J)
          IF (ABS(D(IDIR,IDISPL_ON_ION,J)-1.0) >1E-8) THEN
             CALL VTUTOR('E','LINEARRES POSCAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                  IO%iu6,3)
             CALL VTUTOR('S','LINEARRES POSCAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                  IO%IU0,3)
          ENDIF
       ELSE
          CALL FIND_IJ(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,PROCESSED_DISPL,IDIR,J)
       ENDIF

       IF (SYMM%ISYM>0) THEN
          DYN%VEL=0
          DYN%VEL(IDIR,J)=1.0
          CALL  KARDIR(1,DYN%VEL(:,J),LATT_CUR%B)
          CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
               T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
               SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
               SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
#ifdef oldsym
          CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
               SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
               T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#else
          CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
               SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
               T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#endif
          CALL KPAR_SYNC_ALL(WDES,W)
          CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
          CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS)

          ! Loewdin perturbation theory to improve states at added k-points
          CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
          &    LMDIM,CDIJ,CQIJ,4,SV,T_INFO,P,IO%IU0,DESUM, NKSTART=NKORIG+1)
          CALL KPAR_SYNC_ALL(WDES,W)

          ! or alternatively, but not more accurate 3 steps Davidson
          ! problem is that this might yield wrong piezoelectric tensors
!          X=INFO%EBREAK
!          INFO%EBREAK=0.25*EDIFF
!          DO I=1,3
!             CALL EDDAV(HAMILTONIAN,P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
!             LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV, E%EXHF, IO%IU6,IO%IU0, .FALSE., .TRUE., .FALSE.)
!             
!             E%EBANDSTR=BANDSTRUCTURE_ENERGY(WDES, W)
!             TOTEN_=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF
!
!             IF (IO%IU0>=0) THEN
!                WRITE(IO%IU0,1000) I, TOTEN_, TOTEN_-TOTEN, DESUM, ICOUEV, RMS
!                WRITE(17,1000)     I, TOTEN_, TOTEN_-TOTEN, DESUM, ICOUEV, RMS
!             ENDIF
!             IF(ABS(DESUM)<EDIFF.AND.ABS(TOTEN_-TOTEN)<EDIFF) EXIT
!             TOTEN=TOTEN_
!          ENDDO
!          CALL KPAR_SYNC_ALL(WDES,W)
!          INFO%EBREAK=X         ! restore the break condition

          CALL FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER)
       ENDIF

       CALL LR_MAIN( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
          T_INFO,T_INFO_0,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI, &
          LMDIM,IRDMAX,NEDOS, &
          TOTEN,EFERMI,DISPL_FORCES(PROCESSED_DISPL,:,:),INT_STRAIN(PROCESSED_DISPL,:,:), & 
          IDIR, J, BORN_CHARGES2(:,IDIR,J), &
          DEG_CLUSTER, KPOINTS_TRANS, RPHI, RPHI_CPROJ)
       IF (IO%LOPEN) CALL WFORCE(IO%IU6)

       IF (SYMM%ISYM>0) CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS)
    ENDDO

    CALL FREE_DEG_CLUSTERS(WDES,DEG_CLUSTER)

    IF (LEPSILON) THEN
       DEALLOCATE(RPHI, RPHI_CPROJ)
    ENDIF

    ! restore original symmetry
    IF (SYMM%ISYM>0) THEN
       DYN%VEL=0
       CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
            T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
            SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
            SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
#ifdef oldsym
       CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
            SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
            T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#else
       CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
            SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
            T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU6K,IO%IU0)
#endif
       CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
       CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR)
    ENDIF
!=======================================================================
!
! final processing and output 
! Born effective charges, piezoelectric tensors, and internal strain
!
!=======================================================================
    IF (IBRION==8) THEN
       !
       ! version for case symmetry was used to reduce the number of displacments
       !
       CALL PRINT_DYNMAT_ID(.FALSE.,NIONS,DOF,1.0_q,T_INFO%NTYP,T_INFO%NITYP,T_INFO%POMASS,DISPL_FORCES,D,ND,IU6)

       DO PROCESSED_DISPL=1,DOF
          CALL FIND_IJ_ID(.FALSE.,NIONS,PROCESSED_DISPL,ND,J,IDIR)
          DO N=1,T_INFO%NIONS
             DMAT(1:3,IDIR,N,J)=DISPL_FORCES(PROCESSED_DISPL,1:3,N)
          END DO
          ST(1:3,1:3,IDIR,J)=INT_STRAIN  (PROCESSED_DISPL,1:3,1:3)
       END DO

       CALL MKDMAT(SYMM%ROTMAP,ISYMOP,INVMAP,NROTK,NPCELL,D,DMAT,ST,ND,      &
            1,T_INFO%NTYP,T_INFO%NIONS,T_INFO%NITYP,WORKD,WDMAT,AST,IWORK,SYMM%TAU,SYMM%TAUROT,   &
            SYMM%WRKROT, &
            LATT_CUR%A(1,1),LATT_CUR%A(1,2),LATT_CUR%A(1,3), &
            LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3))

       LDO=.TRUE.
       CALL SYDMAT(DMAT,SYMM%ROTMAP,ISYMOP,NROTK,NPCELL,1,T_INFO%NTYP, &
            T_INFO%NIONS,T_INFO%NITYP,WDMAT,LATT_CUR%A(1,1),LATT_CUR%A(1,2),LATT_CUR%A(1,3),LDO)
       CALL STMAT(ST  ,SYMM%ROTMAP,ISYMOP,NROTK,NPCELL,1,T_INFO%NTYP, &
            T_INFO%NIONS,T_INFO%NITYP,AST  ,LATT_CUR%A(1,1),LATT_CUR%A(1,2),LATT_CUR%A(1,3),LDO)

       ! now reset DOF to NIONS*3
       ! and copy results from DMAT back to DISPL_FORCES

       CALL COUNT_DOF(NIONS, T_INFO%LSFOR, T_INFO%LSDYN, DOF)
       DEALLOCATE(DISPL_FORCES)
       ALLOCATE(DISPL_FORCES(DOF,3,NIONS))

       DO PROCESSED_DISPL=1,DOF
          CALL FIND_IJ(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,PROCESSED_DISPL,IDIR,J)
          DO N=1,NIONS
             DISPL_FORCES(PROCESSED_DISPL,1:3,N)=DMAT(1:3,IDIR,N,J)
          ENDDO
       ENDDO
    ELSE
       !
       ! version for case no symmetry was used
       !
       CALL PRINT_DYNMAT(NIONS,DOF,1.0_q,T_INFO%NTYP,T_INFO%NITYP,T_INFO%POMASS,DISPL_FORCES,T_INFO%LSDYN,T_INFO%LSFOR,IU6)

       DO PROCESSED_DISPL=1,DOF
          CALL FIND_IJ(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,PROCESSED_DISPL,IDIR,J)
          ST(1:3,1:3,IDIR,J)=INT_STRAIN  (PROCESSED_DISPL,1:3,1:3)
       END DO
    ENDIF

    IF (IU6>=0) THEN
       WRITE (IU6,130) 
       IF (LEPSILON) THEN
          ! for the sake of having everything at the very end print EPSILON again
          IF (LRPA) THEN
             WRITE(IO%IU6,1100) '(including local field effects in RPA (Hartree))',EPSILON
          ELSE
             WRITE(IO%IU6,1100) '(including local field effects in DFT)',EPSILON
          ENDIF

          FACT=EVTOJ*1E20_q/LATT_CUR%OMEGA
          
170       FORMAT(/ ' PIEZOELECTRIC TENSOR  for field in x, y, z        (e  Angst)',/ &
               10X,'XX', 10X,'YY', 10X,'ZZ',10X,'XY', 10X,'YZ', 10X,'ZX'/ &
               '  ----------------------------------------------------', &
               '----------------------------')

180       FORMAT(/ ' ',A,'  for field in x, y, z        (C/m^2)',/ &
               &        10X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
               &        '  ----------------------------------------------------', &
               &        '----------------------------')
          IF (.NOT.LRPA) THEN
             WRITE (IU6,170)
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(PIEZO(I,J,J),J=1,3), & 
                     PIEZO(I,1,2),PIEZO(I,2,3),PIEZO(I,3,1)
             ENDDO
             
             CALL TSYM3(PIEZO,ISYMOP,NROTK,LATT_CUR%A)
          
             WRITE (IU6,180) 'PIEZOELECTRIC TENSOR'
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(PIEZO(I,J,J)*FACT,J=1,3), & 
                     PIEZO(I,1,2)*FACT,PIEZO(I,2,3)*FACT,PIEZO(I,3,1)*FACT
             ENDDO
140          FORMAT(2X,A1,6F12.5)

! remove any drift from BORN charges
       IF (IO%IU0>0) WRITE(IO%IU0,*) 'Born effective charges drift removed'
       TMP=0
       DO N=1,T_INFO%NIONS
          TMP(:,:)=TMP(:,:)+BORN_CHARGES(:,:,N)/T_INFO%NIONS
       ENDDO
       DO N=1,T_INFO%NIONS
          BORN_CHARGES(:,:,N)=BORN_CHARGES(:,:,N)-TMP(:,:)
       ENDDO
             
          
             WRITE (IU6,*)
             WRITE (IU6,*) 'BORN EFFECTIVE CHARGES (in e, cummulative output)'
             WRITE (IU6,*) '-------------------------------------------------'
             
             DO N=1,T_INFO%NIONS
                WRITE (IU6,'(" ion ",I4)') N
                DO IDIR =1,3
                   WRITE (IU6,'(I5,3F12.5)') IDIR,BORN_CHARGES(IDIR,:,N)
                ENDDO
             ENDDO

             CALL XML_BORN_CHARGES(BORN_CHARGES,T_INFO%NIONS)
             ! now set the Born charges in the global array defined in pead
             IF (.NOT. LBORN) THEN
                LBORN=.TRUE.
                IF (ALLOCATED(BORN_CHARGES_PEAD)) DEALLOCATE(BORN_CHARGES_PEAD)
                ALLOCATE(BORN_CHARGES_PEAD(3,3,T_INFO%NIOND))
             ENDIF
             BORN_CHARGES_PEAD=BORN_CHARGES
          ENDIF
       ENDIF
    ENDIF
    IF (DOF>0 .AND. IU6>=0) THEN
       IF (DYN%ISIF>0) THEN
160       FORMAT(/ ' INTERNAL STRAIN TENSOR FOR ION ',I4,' for displacements in x,y,z  (eV/Angst):',/ &
               10X,'X', 11X,'Y', 11X,'Z', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
               '  ----------------------------------------------------', &
               '----------------------------')
          DO N=1,T_INFO%NIONS
             WRITE(IU6,160) N
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(ST(J,J,I,N),J=1,3),ST(1,2,I,N),ST(2,3,I,N),ST(3,1,I,N)
             ENDDO
          ENDDO
       END IF
       WRITE (IU6,130)
    ENDIF

!=======================================================================
! apply delta impulse in time DFIELD on ions, by adding 
! a force term to FORCE
!=======================================================================
    IF (.NOT. DFIELD_DONE) THEN
       DFIELD_DONE=.TRUE.
       DO N=1,T_INFO%NIONS
          DO IDIR =1,3 ! loop over cartesian direction of external field
             FORCE(:,N)=FORCE(:,N)+BORN_CHARGES(IDIR,:,N)*DFIELD(IDIR)/DYN%POTIM
          ENDDO
       ENDDO
       
       ALLOCATE(POSION_LAST(3,T_INFO%NIONS))
       ALLOCATE(BORN_CHARGES_SAVE(3,3,T_INFO%NIONS))

       ! store initial postions for later use (required for evolution of dipole moment)
       POSION_LAST=T_INFO_0%POSION
    ENDIF

    
    IF (ALLOCATED(BORN_CHARGES_SAVE) .AND. LEPSILON) THEN
       BORN_CHARGES_SAVE=BORN_CHARGES
    ENDIF
!=======================================================================
!
! final processing and output
! vibrational frequencies
!
!=======================================================================
    IF (DOF>0 .AND. IU6>=0 ) THEN

       ALLOCATE(SECOND_DERIV(DOF,DOF))
       
       DO N=1,DOF
          CALL FIND_IJ(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,N,I,J)       
          DO M=1,DOF
             SECOND_DERIV(M,N)=DISPL_FORCES(M,I,J)
          END DO
       END DO
       
       IF (IU6>=0) THEN
          WRITE (IU6,*) 
          WRITE (IU6,*) 'SECOND DERIVATIVES (NOT SYMMETRIZED)'
          WRITE (IU6,*) '------------------------------------'
          CALL PRINT_SECOND_DERIV(NIONS,DOF,SECOND_DERIV,T_INFO%LSFOR,T_INFO%LSDYN,IU6)
       END IF

       ! symmetrize Hessian matrix
       DO N=1,DOF
          DO M=N+1,DOF
             X=0.5_q*(SECOND_DERIV(N,M)+SECOND_DERIV(M,N))
             SECOND_DERIV(N,M)=X
             SECOND_DERIV(M,N)=X
          END DO
       END DO

       ALLOCATE(WORK(DOF,32),EIGENVECTORS(DOF,DOF),EIGENVALUES(DOF))

       EIGENVECTORS=SECOND_DERIV
       N=1
       DO I=1,T_INFO%NTYP
          DO J=1,T_INFO%NITYP(I)
             DO K=1,3
                CALL FIND_DOF_INDEX(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,K,N,M)
                IF (M>0) EIGENVECTORS(:,M)=EIGENVECTORS(:,M)/SQRT(T_INFO%POMASS(I))
                IF (M>0) EIGENVECTORS(M,:)=EIGENVECTORS(M,:)/SQRT(T_INFO%POMASS(I))
             END DO
             N=N+1
          END DO
       END DO

       CALL XML_TAG("dynmat")
       CALL XML_VECARRAY("hessian")
       CALL XML_ARRAY_REAL(EIGENVECTORS)
       CALL XML_CLOSE_TAG

       CALL DSYEV &
            ('V','U',DOF,EIGENVECTORS,DOF, &
            EIGENVALUES,WORK,32*DOF, IERROR)
       IF (IERROR/=0) THEN
          IF (IU6>=0) THEN
             WRITE(IU6,*) "Error while diagonalisation DSYEV INFO=",IERROR
             WRITE(IU6,*) "Some of (or all) eigenvectors and eigenvalues are not correct !"
          END IF
       END IF

       CALL XML_VEC_REAL(EIGENVALUES,"eigenvalues",'(ES16.8)')
       CALL XML_VECARRAY("eigenvectors")
       CALL XML_ARRAY_REAL(EIGENVECTORS,'(ES16.8)')
       CALL XML_CLOSE_TAG
       CALL XML_CLOSE_TAG

       CALL PRINT_EIGENVECTORS(NIONS,DOF,T_INFO_0%POSION,LATT_CUR%A, &
            EIGENVECTORS,EIGENVALUES,       &
            T_INFO%LSFOR,T_INFO%LSDYN,IU6)

       N=1
       DO I=1,T_INFO%NTYP
          DO J=1,T_INFO%NITYP(I)
             DO K=1,3
                CALL FIND_DOF_INDEX(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,K,N,M)
                IF (M>0) EIGENVECTORS(M,:)=EIGENVECTORS(M,:)/SQRT(T_INFO%POMASS(I))
             END DO
             N=N+1
          END DO
       END DO

       IF (IU6>=0 .AND. IO%NWRITE>=3) THEN
          WRITE(IU6,*) "Eigenvectors after division by SQRT(mass)"
          CALL PRINT_EIGENVECTORS(NIONS,DOF,T_INFO_0%POSION,LATT_CUR%A, &
               EIGENVECTORS,EIGENVALUES,       &
               T_INFO%LSFOR,T_INFO%LSDYN,IU6)
       ENDIF
       DEALLOCATE(WORK)

       CALL STORE_RPA_SECOND_DERIV(SECOND_DERIV)

       IF (DYN%ISIF>0 .OR. LEPSILON) THEN
          WRITE(IO%IU6,130)
          ! invert the matrix of the second derivatives
          SECOND_DERIV=-SECOND_DERIV
          CALL INV_SECOND_DERIV(SECOND_DERIV, DOF, IU6 )

          ! ionic contribution to macroscopic dielectric tensor
          IF (LEPSILON .AND. IO%IU6>=0 .AND. .NOT. LRPA) THEN
             CALL EPSILON_ION( T_INFO, DOF, SECOND_DERIV, BORN_CHARGES, EPSILON )

             ! induced polariation -> field
             EPSILON=EPSILON*2*TPI/(LATT_CUR%OMEGA)*FELECT
             WRITE(IO%IU6,1100) 'IONIC CONTRIBUTION',EPSILON

             CALL XML_TENSOR("epsilon_ion",EPSILON)

             CALL EPSILON_ION_OMEGA( T_INFO, DOF, EIGENVECTORS, EIGENVALUES, BORN_CHARGES, LATT_CUR%OMEGA )
          ENDIF

          IF (DYN%ISIF>0.AND.IO%IU6>=0) THEN
100       FORMAT(/ &
            A / &
            ' Direction', &
            4X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
            ' --------------------------------------------------------------------------------'/ &
            ' XX     ',6F12.4/ &
            ' YY     ',6F12.4/ &
            ' ZZ     ',6F12.4/ &
            ' XY     ',6F12.4/ &
            ' YZ     ',6F12.4/ &
            ' ZX     ',6F12.4/ &
            ' --------------------------------------------------------------------------------'/)
             CALL ELASTIC_ION( T_INFO, DOF, SECOND_DERIV, ST, ELASTIC )

             ELASTICP(1,:,:)=ELASTIC(1,1,:,:)
             ELASTICP(2,:,:)=ELASTIC(2,2,:,:)
             ELASTICP(3,:,:)=ELASTIC(3,3,:,:)
             ELASTICP(4,:,:)=ELASTIC(1,2,:,:)
             ELASTICP(5,:,:)=ELASTIC(2,3,:,:)
             ELASTICP(6,:,:)=ELASTIC(3,1,:,:)

             FACT=EVTOJ*1E22_q/LATT_CUR%OMEGA

             WRITE(IU6,100) ' ELASTIC MODULI IONIC CONTR (kBar)', ( &
                  (ELASTICP(J,I,I)*FACT,I=1,3), &
                   ELASTICP(J,1,2)*FACT,ELASTICP(J,2,3)*FACT,ELASTICP(J,3,1)*FACT,J=1,6)
          END IF

          ! ionic contribution to piezoelectric tensor
          IF (LEPSILON .AND. DYN%ISIF>0.AND.IO%IU6>=0 .AND. .NOT. LRPA) THEN

             CALL PIEZO_ION( T_INFO, DOF, SECOND_DERIV, ST, BORN_CHARGES, PIEZO )
             
             WRITE (IU6,180) 'PIEZOELECTRIC TENSOR IONIC CONTR'
             FACT=EVTOJ*1E20_q/LATT_CUR%OMEGA
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(PIEZO(I,J,J)*FACT,J=1,3), & 
                     PIEZO(I,1,2)*FACT,PIEZO(I,2,3)*FACT,PIEZO(I,3,1)*FACT
             ENDDO
          ENDIF
       END IF
       DEALLOCATE(SECOND_DERIV, EIGENVECTORS, EIGENVALUES)
    END IF
    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    DEALLOCATE(T_INFO_0%POSION)
    DEALLOCATE(INITIAL_FORCE)
    DEALLOCATE(DISPL_FORCES,BORN_CHARGES,BORN_CHARGES2, &
      PIEZO,INT_STRAIN)

    IF (IBRION==8) THEN
      DEALLOCATE(D, ND, IDIRD)
    ENDIF


    IF (IU0>=0 .AND. DOF>0) THEN
       WRITE (IU0,*) 'Linear response finished'
       WRITE (17,*) 'Linear response finished'
    ENDIF

    IF (IO%LOPEN) CALL WFORCE(IO%IU6)

  END SUBROUTINE LR_SKELETON

!************************ SUBROUTINE STORE_DIPOLE_AND_AUTOCORRELATE ****
!
! Routine to store the dipole moment as a function of time
!
!***********************************************************************

  SUBROUTINE STORE_DIPOLE_AND_AUTOCORRELATE( T_INFO, LATT_CUR, POTIM, NSW, NSTEP, IO)
    USE constant
    USE lattice 
    USE poscar
    USE base
    USE mlrf_main
    IMPLICIT NONE
    TYPE (type_info)   T_INFO     ! information on positions etc.
    TYPE (latt)        LATT_CUR   ! lattice parameters
    REAL(q)            POTIM
    INTEGER            NSW        ! total number of steps
    INTEGER            NSTEP      ! current step
    TYPE (in_struct)   IO
    REAL(q),PARAMETER :: PLANK=6.626075E-34
  ! local variables
    REAL(q) :: DIPOLE(3), TMP(3), OMEGA_MAX, OMEGA, DAMPING, SCALE
    INTEGER :: N, I, J
    COMPLEX(q), ALLOCATABLE :: EPSILON(:,:)
    COMPLEX(q) :: EXPFAC
    
    ! first allocate if required
    IF (.NOT. ALLOCATED(DIPOLE_TIME)) THEN
       ALLOCATE(DIPOLE_TIME(3, NSW))
       DIPOLE_TIME=0
    ENDIF

    ! first calculate current dipole
    DIPOLE=0
    
    SCALE=1.0_q/SQRT(DFIELD(1)**2+DFIELD(2)**2+DFIELD(3)**2)
    DO N=1,T_INFO%NIONS
       TMP=T_INFO%POSION(:,N)-POSION_LAST(:,N)
       TMP(1)=MOD(TMP(1)+4.5_q,1.0_q)-0.5_q
       TMP(2)=MOD(TMP(2)+4.5_q,1.0_q)-0.5_q
       TMP(3)=MOD(TMP(3)+4.5_q,1.0_q)-0.5_q
       ! convert to cartesian coordinates
       CALL DIRKAR(1, TMP, LATT_CUR%A)
       ! divide by the absolute magnitude of the external field
       ! to obtain response
       DIPOLE(:)=DIPOLE(:)+MATMUL(BORN_CHARGES_SAVE(:,:,N), TMP)*SCALE
    ENDDO
    ! store current positions into last position array
    POSION_LAST=T_INFO%POSION

    IF (NSTEP==1) THEN
       ! initial dipole is assumed to be zero
       DIPOLE_TIME(:,NSTEP)=0
    ELSE
       ! accumulate the change of the dipole over time
       DIPOLE_TIME(:,NSTEP)=DIPOLE(:)+DIPOLE_TIME(:,NSTEP-1)
    ENDIF

    IF (NSTEP==NSW) THEN
       ALLOCATE(EPSILON(3, NSW))
       EPSILON=0

       OMEGA_MAX=PI/POTIM
       DAMPING=4.0_q/(POTIM*NSW)

       IF (IO%IU6>=0) THEN
          DO I=0,NSW-1
             WRITE(77,'(4F14.7)') POTIM*I, DIPOLE_TIME(:,I+1)
          ENDDO
       ENDIF

       DO I=0,NSW-1
          OMEGA=OMEGA_MAX/NSW*I
          ! weight is the derivative of the function above
          DO J=0,NSW-1
             EXPFAC=EXP( ((0._q,1._q)*OMEGA-DAMPING )*POTIM*J) !expfac=exponential-factor
             EPSILON(:,I+1)=EPSILON(:,I+1)-1.0_q*DIPOLE_TIME(:,J+1)*EXPFAC*POTIM
          ENDDO
       ENDDO !ENDDO omega_sum

       ! in CALC_WRITE_DF there is another factor HARTREE**2 / AUTOA**2 
       EPSILON=EPSILON*EDEPS/LATT_CUR%OMEGA

       IF (IO%IU6>=0) THEN
          DO I=0,NSW-1
             ! convert from circular frequency to cm (see finite_diff.F)
             WRITE(78,'(7F14.7)') OMEGA_MAX/NSW*I*1000, -REAL(EPSILON(:,I+1),q)
          ENDDO
          WRITE(78,*)
          DO I=0,NSW-1
             ! convert from circular frequency to cm (see finite_diff.F)
             WRITE(78,'(7F14.7)') OMEGA_MAX/NSW*I*1000, -AIMAG(EPSILON(:,I+1))
          ENDDO
       ENDIF

       DEALLOCATE(EPSILON)

       
    ENDIF

  END SUBROUTINE STORE_DIPOLE_AND_AUTOCORRELATE


!************************ SUBROUTINE INTERPOLATE_BAND_STR **************
!
! this routine interpolates the band structure to a dense k-point
! grid 
! the strategy is fairly simply
! first the first derivative with respect to k is calculated
!    (presently only the subspace of considered orbitals)
! next the k-point grid is partly shifted and a diagonalization
! in the subspace is performed
! very little data is stored, to keep the routine simple
! and concise
! 
!***********************************************************************

    SUBROUTINE INTERPOLATE_BAND_STR(HAMILTONIAN, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
       &    T_INFO, NONLR_S, NONL_S, W, NEDOS, DOS, DOSI, NELECT, NUP_DOWN, &
       &    LMDIM, P, SV, CQIJ, CDIJ, SYMM, IU0, IU6, RPHI)
      USE prec
      USE wave_high
      USE lattice
      USE poscar
      USE mpimy
      USE mgrid
      USE nonl_high
      USE base
      USE pseudo
      USE kpoints_change
      USE constant
      USE choleski
      USE subrot
      USE mlrf_main
      USE hamil_high
      IMPLICIT NONE

      TYPE (ham_handle)  HAMILTONIAN
      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (latt)        LATT_INI
      TYPE (type_info)   T_INFO
      TYPE (nonlr_struct)NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (potcar)      P(:)
      TYPE (wavespin), TARGET :: W
      INTEGER LMDIM
      OVERLAP         CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP         CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      TYPE (symmetry) SYMM
      RGRID           SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
      INTEGER IU0, IU6
      COMPLEX(qs), OPTIONAL :: RPHI(:,:,:,:,:)
      INTEGER    NEDOS
      REAL(q)    DOS(NEDOS,W%WDES%ISPIN),DOSI(NEDOS,W%WDES%ISPIN), NELECT, NUP_DOWN
!    local
      INTEGER NKPTS_ORIG, I, ISP, NK, NPOS, NB
      INTEGER IKX, IKY, IKZ, IKXP, IKYP, IKZP, IWZ
      REAL(q) DISPL(3), DISPL_CART(3), DISPL_FOUND(3), DIST, DIST_FOUND, KSTART
      REAL(q) EFERMI, ENTROPY, PAR(1,1,1,1,W%WDES%NCDIJ),DOSPAR(1,1,1,W%WDES%NCDIJ)
      REAL(q)    DOS_TMP(NEDOS,W%WDES%ISPIN),DOSI_TMP(NEDOS,W%WDES%ISPIN), EFERMI_TMP
      INTEGER :: N
      REAL(q) :: EXHF
!-----------------------------------------------------------------------
! double k-points in new structure
! in the upper part of the wavefunction arrays the wavefunctions 
! corresponding to the new shifted k-points are stored
! in the lower part the original wavefunctions are stored
!-----------------------------------------------------------------------

      IF (KPOINTS%NKPX<1 .OR. KPOINTS%NKPY<1 .OR. KPOINTS%NKPZ<1 )  THEN
         ! here we need some warning
         ! the routine works only using regular gamma centered grids
         WRITE(*,*) ' INTERPOLATE_BAND_STR: requires a regular k-point grid '
         RETURN
      ENDIF

      CALL CHECK_FULL_KPOINTS

      ! original number of k-points
      NKPTS_ORIG=W%WDES%NKPTS
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM, IU6, IU0, W%WDES%VKPT(:,1:NKPTS_ORIG))
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM, IU6, IU0, W%WDES%VKPT(:,1:NKPTS_ORIG))
#endif
      CALL KPAR_SYNC_ALL(W%WDES,W)
      CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, IU6, IU0)
      CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)

      CALL XML_TAG("eigenvalues", comment="interpolated")
!-----------------------------------------------------------------------
! loop over new super k-point grid
! a  kind of Wigner Seitz procedure is used to generate the new supergrid
! this can be bypassed by setting IWZ to 0
! a few notes are in place here
! ) only gamma centered grids do not violate the symmetry
!   and are therefore recommended
! ) such super grids are created by setting KINTER to an odd value
! ) otherwise the super grids might spoil the symmetry
!-----------------------------------------------------------------------
      IWZ=2
      KSTART=-REAL(ABS(KINTER),q)/2+0.5_q
      DOS =0
      DOSI=0
      EFERMI=0

      DO IKX=0,ABS(KINTER)-1
      DO IKY=0,ABS(KINTER)-1
      DO IKZ=0,ABS(KINTER)-1

         ! search the equivalent k-point with the shortest length
         DIST_FOUND=1E6
         DO IKXP=-IWZ,IWZ
         DO IKYP=-IWZ,IWZ
         DO IKZP=-IWZ,IWZ
            DISPL(1)=((IKX+KSTART)/ABS(KINTER)+IKXP)/KPOINTS%NKPX
            DISPL(2)=((IKY+KSTART)/ABS(KINTER)+IKYP)/KPOINTS%NKPY
            DISPL(3)=((IKZ+KSTART)/ABS(KINTER)+IKZP)/KPOINTS%NKPZ
            ! k-point displacement in cartesian coordinates
            DISPL_CART=DISPL
            CALL DIRKAR(1, DISPL_CART(1), LATT_CUR%B)
            DIST=DISPL_CART(1)**2+DISPL_CART(2)**2+DISPL_CART(3)**2
            IF (DIST<DIST_FOUND) THEN
               DISPL_FOUND=DISPL
               DIST_FOUND =DIST
            ENDIF
         ENDDO
         ENDDO
         ENDDO
         
         DISPL=DISPL_FOUND
         DISPL_CART=DISPL
         CALL DIRKAR(1, DISPL_CART(1), LATT_CUR%B)

         DO NK=1,NKPTS_ORIG
            W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)+DISPL
         ENDDO

         IF (PRESENT(RPHI)) THEN
            DO ISP=1,W%WDES%ISPIN
               DO NK=1,NKPTS_ORIG
                  DO N=1,W%WDES%NBANDS
                     W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)+TPI*( &
                          RPHI(:,N,NK,ISP,1)*(0.0_q,1.0_q)*DISPL_CART(1)+ & 
                          RPHI(:,N,NK,ISP,2)*(0.0_q,1.0_q)*DISPL_CART(2)+ &
                          RPHI(:,N,NK,ISP,3)*(0.0_q,1.0_q)*DISPL_CART(3))
                  ENDDO
               ENDDO
            ENDDO
         ELSE
            DO ISP=1,W%WDES%ISPIN
               DO NK=1,NKPTS_ORIG
                  DO N=1,W%WDES%NBANDS
                     W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)
                  ENDDO
               ENDDO
            ENDDO
         ENDIF
         CALL SET_DATAKE(W%WDES, LATT_CUR%B)
         IF (NONLR_S%LREAL) THEN
            CALL RSPHER(GRID,NONLR_S,LATT_CUR)
         ELSE
            CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
         ENDIF
         CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W)

         CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)
         
         CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
              LMDIM,CDIJ,CQIJ,2,SV,T_INFO,P,IU0,EXHF, & 
              NKSTART=NKPTS_ORIG+1)

         IF (KPOINTS%ISMEAR>=0) THEN
         ! well this is not elegant, we now simply set the weights for
         ! lower k-point set to 0, and that for the upper one to the
         ! proper weight
         KPOINTS%WTKPT(NKPTS_ORIG+1:NKPTS_ORIG*2)=KPOINTS%WTKPT(1:NKPTS_ORIG)
         KPOINTS%WTKPT(1:NKPTS_ORIG)=0

         ! sum dos
         CALL DENSTA( IU0, IU6, W%WDES, W, KPOINTS, NELECT, &
              NUP_DOWN, ENTROPY, EFERMI_TMP, KPOINTS%SIGMA, .TRUE., &
              NEDOS, 0, 0, DOS_TMP, DOSI_TMP, PAR, DOSPAR)

         ! restore weights
         KPOINTS%WTKPT(1:NKPTS_ORIG)=KPOINTS%WTKPT(NKPTS_ORIG+1:NKPTS_ORIG*2)
         KPOINTS%WTKPT(NKPTS_ORIG+1:NKPTS_ORIG*2)=0

         DOS =DOS +DOS_TMP *(1_q/REAL(KINTER,q)/REAL(KINTER,q)/REAL(KINTER,q))
         DOSI=DOSI+DOSI_TMP*(1_q/REAL(KINTER,q)/REAL(KINTER,q)/REAL(KINTER,q))
         EFERMI=EFERMI+EFERMI_TMP*(1_q/REAL(KINTER,q)/REAL(KINTER,q)/REAL(KINTER,q))
         ENDIF

         IF (IU6>=0) THEN
            
            DO ISP=1,W%WDES%ISPIN
               WRITE(IU6,'(/" k-point displacement            :",3F10.4 )') DISPL
               WRITE(IU6,'( " k-point displacement (cartesian):",3F10.4 )') DISPL_CART
               IF (W%WDES%ISPIN==2) WRITE(IU6,'(/A,I1)') ' spin component ',ISP
               DO N=NKPTS_ORIG+1,W%WDES%NKPTS
                  WRITE(IU6,2201) N-NKPTS_ORIG, W%WDES%VKPT(1,N),W%WDES%VKPT(2,N),W%WDES%VKPT(3,N), &
                       &             (I,REAL( W%CELTOT(I,N,ISP) ,KIND=q) ,W%FERTOT(I,N,ISP)*W%WDES%RSPIN,I=1,W%WDES%NB_TOT)
               ENDDO
            ENDDO
2201        FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
                 &         '  band No.  band energies     occupation '/ &
                 &           (3X,I4,3X,F10.4,3X,F10.5))

            
         ENDIF

         CALL XML_VEC_REAL( DISPL, "displacement" )
         CALL XML_VEC_REAL( DISPL_CART, "displacement_cart" )

         CALL XML_EIGENVAL_NOHEAD( W%CELTOT(:,NKPTS_ORIG+1:W%WDES%NKPTS,:), W%FERTOT(:,NKPTS_ORIG+1:W%WDES%NKPTS,:), &
              W%WDES%NB_TOT, W%WDES%NKPTS-NKPTS_ORIG, W%WDES%ISPIN)
      
      ENDDO
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! restore old data layout
!-----------------------------------------------------------------------
      CALL XML_CLOSE_TAG
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#endif
      CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, -1, -1)
      CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)

      IF (KPOINTS%ISMEAR>=0) THEN
      CALL XML_DOS(EFERMI, KPOINTS%EMIN, KPOINTS%EMAX, .FALSE., &
           DOS, DOSI, DOSPAR, NEDOS, 1, 1, W%WDES%NCDIJ, comment='interpolated')
      ENDIF


    END SUBROUTINE INTERPOLATE_BAND_STR


!************** SUBROUTINE INTERPOLATE_BANDSTR_GRDSHFT *****************
!************************************************************************

    SUBROUTINE INTERPOLATE_BANDSTR_GRDSHFT(HAMILTONIAN, E, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
       &    T_INFO, NONLR_S, NONL_S, W, NEDOS, &
       &    LMDIM, P, SV, CQIJ, CDIJ, SYMM, INFO, IO, GRIDC, GRIDUS, C_TO_US, IRDMAX, RPHI)
      USE prec
      USE wave_high
      USE hamil_high
      USE lattice
      USE poscar
      USE mpimy
      USE mgrid
      USE nonl_high
      USE base
      USE pseudo
      USE kpoints_change
      USE constant
      USE choleski
      USE subrot
      USE subrot_cluster
      USE mlr_optic
      USE mlrf_main
      USE ini
      USE david
      IMPLICIT NONE

      TYPE (ham_handle)     :: HAMILTONIAN
      TYPE (energy)         :: E
      TYPE (kpoints_struct) :: KPOINTS
      TYPE (grid_3d)        :: GRID
      TYPE (latt)           :: LATT_CUR
      TYPE (latt)           :: LATT_INI
      TYPE (type_info)      :: T_INFO
      TYPE (nonlr_struct)   :: NONLR_S
      TYPE (nonl_struct)    :: NONL_S
      TYPE (potcar)         :: P(:)
      TYPE (grid_3d)        :: GRIDC 
      TYPE (grid_3d)        :: GRIDUS
      TYPE (transit)        :: C_TO_US
      TYPE (info_struct)    :: INFO
      TYPE (in_struct)      :: IO
      TYPE (symmetry)       :: SYMM
      INTEGER               :: LMDIM, NEDOS
      TYPE (wavespin), TARGET :: W
      TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
      OVERLAP :: CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP :: CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      RGRID   :: SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
      REAL(q) :: DOS(NEDOS,W%WDES%ISPIN),DOSI(NEDOS,W%WDES%ISPIN)
      COMPLEX(qs), OPTIONAL :: RPHI(:,:,:,:,:)
      
      TYPE (kpoints_struct) :: KPOINTS_INTER
      TYPE (skpoints_full),SAVE,POINTER :: KPOINTS_INTER_FULL
      TYPE (skpoints_full),SAVE,POINTER :: KPOINTS_FULL_TEMP
      TYPE (wavedes)  :: WDES_INTER
      TYPE (wavespin) :: W_INTER
      TYPE (wavespin) :: WDIFF
      ! eFL: I DO NOT THINK ALLOCW NEEDS THIS, REMOVE?
      TYPE (wavefun)  :: WTMP

      LOGICAL :: LDONE, LWARN, LDO_VEL
      INTEGER :: I, L, K, J, ISP, NK, NPOS, NEDOS_DIEL, KPAR
      INTEGER :: DISPINDEX, IBZINDEX, IDIR, JDIR, IRDMAX
      INTEGER :: IK, N, NB, NB2, NSIM, NELM, ICOUEV, IWINDOW
      REAL(q) :: EXHF, RMS, DESUM1, TOTEN, TOTENINTER, TOTENRDIFF, WO
      REAL(q) :: TOTENORG, OTHERTERMS, EBANDSTR_INTER, DELTAE, EMAX
      REAL(q) :: DER(3)
      INTEGER :: MULTIPL(3)
      REAL(q) :: EFERMI, ENTROPY, PAR(1,1,1,1,W%WDES%NCDIJ)
      REAL(q) :: DOSPAR(1,1,1,W%WDES%NCDIJ)
      REAL(q), ALLOCATABLE :: DISPL(:,:)
      REAL(q) :: DISPLV(3), DISPLV_CART(3)
      REAL(q), ALLOCATABLE :: EDER(:,:,:,:)
      REAL(q), ALLOCATABLE :: EINTERPOL_FULL(:,:,:,:), EINTERPOL(:,:,:,:)
      REAL(q), ALLOCATABLE :: KINTERPOL(:,:)
      GDEFS, ALLOCATABLE :: CHAMD(:,:,:,:,:)
      GDEFS, ALLOCATABLE :: CHAM(:,:,:,:,:)
      COMPLEX(q), ALLOCATABLE  :: EPSDD(:,:,:)
      COMPLEX(q), ALLOCATABLE  :: EPSJJ(:,:,:)
      REAL(q), ALLOCATABLE :: WPLASM(:,:), COND(:,:)
      REAL(q), ALLOCATABLE :: BIMAG(:,:)
      COMPLEX(q), ALLOCATABLE :: CW_OLD(:,:,:,:)
      REAL(q), ALLOCATABLE :: VKPT_OLD(:,:)
      REAL(q) :: S(3,3)

      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
           GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

#ifdef MPI
      IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALL VTUTOR('E','NOKPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
              .TRUE.,1, IO%IU6,3)
         CALL VTUTOR('S','NOKPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
              .TRUE.,1, IO%IU0,3)
         RETURN
      ENDIF
#endif

      IF (KPOINTS%ISMEAR <= -4) THEN
         CALL VTUTOR('E','GRDSHFT_NOTET',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
              .TRUE.,1, IO%IU6,3)
         CALL VTUTOR('S','GRDSHFT_NOTET',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
              .TRUE.,1, IO%IU0,3)
         RETURN
      ENDIF

      LDO_VEL=LVEL.OR.IO%LOPTICS

      TOTENORG = BANDSTRUCTURE_ENERGY(W%WDES,W)+E%DENC+E%XCENC+E%TEWEN+ &
           E%PSCENC+E%EENTROPY+ E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF
      
      CALL GENERATE_GRDSHFT_GRD_AND_DISPL(KPOINTS, W%WDES, LATT_CUR, T_INFO, &
           SYMM, IO, KINTER, KPOINTS_INTER, KPOINTS_INTER_FULL, DISPL)
      
      ALLOCATE(EINTERPOL(W%WDES%NB_TOT,KPOINTS_INTER%NKPTS,W%WDES%ISPIN,4), & 
               KINTERPOL(3,KPOINTS_INTER%NKPTS), VKPT_OLD(3,KPOINTS%NKPTS), &
               CW_OLD(SIZE(W%CW,1),W%WDES%NBANDS,KPOINTS%NKPTS,W%WDES%ISPIN))
      
      EFERMI=0.0
      DOS =0.0
      DOSI=0.0
     
      IF (IO%LOPTICS) THEN
         NEDOS_DIEL=MAX(NEDOS,1000)
         ALLOCATE(EPSDD(NEDOS_DIEL,3,3), EPSJJ(NEDOS_DIEL,3,3), &
              WPLASM(3,3), COND(3,3), BIMAG(3,3), &
              CHAMD(W%WDES%NB_TOT, W%WDES%NB_TOT, W%WDES%NKPTS, &
              W%WDES%ISPIN, 3), &
              CHAM(W%WDES%NB_TOT, W%WDES%NB_TOT, KPOINTS_INTER%NKPTS, &
              W%WDES%ISPIN, 3))
         
         EPSDD=0
         EPSJJ=0
         BIMAG=0
         
         ! eFL: NOT SO SURE ABOUT THIS BUCKET THING FOR EACH DIFFERENT
         !      DISPLACEMENT. I SUSPECT THIS MIGHT CHANGE SLIGHTLY, SAY
         !      WE HIT A K-POINT THAT MODIFIES THIS RANGE WHILE DISPLACING
         !      MAYBE DO THIS FOR EACH DISPLACEMENT...BUT THEN WE HAVE THE
         !      PROBLEM OF A DIFFERENT dENERGY FOR EACH INDEX...HMMM...
         !      SO HERE WE USE THE BUCKETING FROM THE INPUT GRID
         EMAX=MAX_ENERGY_UNOCCUPIED(W%WDES,W)*1.2
         IF (EMAX<=0) THEN
            CALL VTUTOR('E','NOOCC',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                 IO%IU6,3)
            CALL VTUTOR('S','NOOCC',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                 IO%IU0,3)
            RETURN
         ENDIF
         DELTAE=EMAX/(NEDOS_DIEL-1)
         
         CALL DENSTA( IO%IU0, IO%IU6, W%WDES, W, KPOINTS, &
              INFO%NELECT, INFO%NUP_DOWN, ENTROPY, EFERMI, &
              KPOINTS%SIGMA, .TRUE., NEDOS, 0, 0, &
              DOS, DOSI, PAR, DOSPAR)
      ENDIF
      
      IF (LDO_VEL) THEN
         ALLOCATE(EDER(W%WDES%NB_TOT,KPOINTS%NKPTS,W%WDES%ISPIN,3)) 
         CALL ALLOCW(W%WDES,WDIFF,WTMP,WTMP)
      ENDIF

      DO ISP=1,W%WDES%ISPIN
         DO NK=1,KPOINTS%NKPTS
            IF (ISP==1) THEN
               VKPT_OLD(:,NK)=W%WDES%VKPT(:,NK)
            ENDIF
            DO N=1,W%WDES%NBANDS
               CW_OLD(:,N,NK,ISP)=W%CW(:,N,NK,ISP)
            ENDDO
         ENDDO
      ENDDO

      
      EINTERPOL=0.0
      KINTERPOL=0.0
      IBZINDEX=1
      DISPINDEX=1

      WDES_INTER=W%WDES
      WDES_INTER%NKPTS= KPOINTS_INTER%NKPTS
      WDES_INTER%VKPT =>KPOINTS_INTER%VKPT
      WDES_INTER%WTKPT=>KPOINTS_INTER%WTKPT

      CALL ALLOCW_NOPLANEWAVE(WDES_INTER, W_INTER)

      IF (IO%IU0>=0) WRITE(IO%IU0,*) ' '
      IF (IO%IU0>=0) WRITE(IO%IU0,*) &
           '------------------------------------------------'// &
           '--------------'
      IF (IO%IU0>=0) WRITE(IO%IU0,*) &
           'Starting gridshift interpolation of the bandstructure'
      IF (IO%IU0>=0) WRITE(IO%IU0,*) &
           '  original grid sampling:', KPOINTS%NKPX, &
           KPOINTS%NKPY, KPOINTS%NKPZ
      IF (IO%IU0>=0) WRITE(IO%IU0,*) &
           '  new grid grid sampling:', KPOINTS%NKPX*ABS(KINTER), &
           KPOINTS%NKPY*ABS(KINTER), KPOINTS%NKPZ*ABS(KINTER)
      IF (IO%IU0>=0) WRITE(IO%IU0,*) &
           '------------------------------------------------'// &
           '--------------'
            
      DO IK=1,ABS(KINTER*KINTER*KINTER)
         IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,*) 'Displacement', IK
            WRITE(17,*)     'Displacement', IK
         ENDIF

         DISPLV=DISPL(:,IK)

         DISPLV_CART=DISPLV
         CALL DIRKAR(1, DISPLV_CART(1), LATT_CUR%B)

         DO NK=1,KPOINTS%NKPTS
            W%WDES%VKPT(:,NK)=VKPT_OLD(:,NK)+DISPLV
         ENDDO

         IF (KINTER>=0) THEN
            DO ISP=1,W%WDES%ISPIN
               DO NK=1,KPOINTS%NKPTS
                  DO NB=1,W%WDES%NBANDS
                     ! eFL: Here we not use RPHI even if we hit a
                     ! displaced k-point that exists in the input IBZ
                     ! The accuracy would most likely be improve by using
                     ! the orginal CW values at these points.
                     ! For this we need a mapping table between the
                     ! KPOINTS_INTER and the grid created by the
                     ! displacements
                     W%CW(:,NB,NK,ISP)=CW_OLD(:,NB,NK,ISP)+TPI*( &
                          RPHI(:,NB,NK,ISP,1)*(0.0_q,1.0_q)*DISPLV_CART(1)+ & 
                          RPHI(:,NB,NK,ISP,2)*(0.0_q,1.0_q)*DISPLV_CART(2)+ &
                          RPHI(:,NB,NK,ISP,3)*(0.0_q,1.0_q)*DISPLV_CART(3))
                  ENDDO
               ENDDO
            ENDDO
         ELSE
            DO ISP=1,W%WDES%ISPIN
               DO NK=1,KPOINTS%NKPTS
                  DO NB=1,W%WDES%NBANDS
                     W%CW(:,NB,NK,ISP)=CW_OLD(:,NB,NK,ISP)
                  ENDDO
               ENDDO
            ENDDO
         ENDIF
         CALL SET_DATAKE(W%WDES, LATT_CUR%B)
         IF (NONLR_S%LREAL) THEN
            CALL RSPHER(GRID,NONLR_S,LATT_CUR)
         ELSE
            CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
         ENDIF
         CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W)

         CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)        

         IF (.NOT. LINTERFAST) THEN
            NSIM=W%WDES%NSIM*2
#ifdef MPI
            NSIM=((W%WDES%NSIM*2+W%WDES%COMM_INTER%NCPU-1)/ &
                 W%WDES%COMM_INTER%NCPU)*W%WDES%COMM_INTER%NCPU
#endif
            DO NELM=1,10
               CALL EDDAV(HAMILTONIAN,P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S, &
                    W,W%WDES, NSIM, LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV, SV, &
                    E%EXHF, IO%IU6,IO%IU0, .FALSE., .TRUE., .FALSE.)
               E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES, W)

               IF (IO%IU0>=0) WRITE(17, 200)      &
                    NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
               IF (IO%IU0>=0) WRITE(IO%IU0, 200)  &
                    NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS

               TOTEN=E%EBANDSTR
           
200            FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
                    &       I6,'  ',E10.3)
               
               IF (ABS(DESUM1) < ABS(INFO%EDIFF) ) EXIT
            ENDDO
         ELSE IF (LDO_VEL) THEN
            CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES, &
                 SYMM, LMDIM,CDIJ,CQIJ,3,SV,T_INFO,P,IO%IU0,EXHF)
         ELSE
            CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES, &
                 SYMM,LMDIM,CDIJ,CQIJ,2,SV,T_INFO,P,IO%IU0,EXHF)
         ENDIF

         E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES, W)

! eFL: MARTIJN, CAN YOU CHECK IF THESE THINGS ARE OKEY FOR
!      ISPIN=2? I THINK SO...

         IF (LDO_VEL) THEN
            WDIFF%CW   =0
            WDIFF%CPROJ=0

            NULLIFY(DEG_CLUSTER)
            CALL FIND_DEG_CLUSTERS(W%WDES, W, DEG_CLUSTER)

            IF (LUSEPEAD()) THEN
               ALLOCATE(KPOINTS_FULL_TEMP)
               KPOINTS_FULL_TEMP = KPOINTS_FULL
               KPOINTS_FULL = KPOINTS_INTER_FULL
            ENDIF
 
            IF (IO%LOPTICS) THEN
               IF (IO%IU0>=0) WRITE(IO%IU0,*) 'Calculating the energy '// &
                    'derivatives and density matrix the displaced '// &
                    'fine grid:'
            ELSE
               IF (IO%IU0>=0) WRITE(IO%IU0,*) 'Calculating the energy '// &
                    'derivatives on the displaced fine grid:'
            ENDIF

            DO IDIR=1,3
               IF (IO%IU0>=0) WRITE(IO%IU0,'(A,I4)') '  Direction:',IDIR

               IF (LUSEPEAD()) THEN
                  WDIFF%CW=0
                  WDIFF%CPROJ=0
                  WDIFF%CELTOT=0
                  CALL PEAD_DPSI_DK_IDIR(W,KPOINTS,P,CQIJ,LATT_CUR, &
                       T_INFO,IDIR,WDIFF)
               ELSE
                  CALL FOCK_K_DER_ANALYT(KPOINTS, GRID, LATT_CUR, LATT_INI, &
                       T_INFO,  NONLR_S, NONL_S, W, WDIFF, LMDIM, P, CQIJ, &
                       SYMM, IDIR, LDONE, IO%IU0, IO%IU6)

                  CALL LRF_RPHI0( &
                       P,NONLR_S,NONL_S,W,LATT_CUR, &
                       T_INFO,INFO,IO,GRID,GRIDC,GRIDUS,C_TO_US,IRDMAX, &
                       CDIJ,CQIJ,SV,LMDIM,DEG_CLUSTER, IDIR, WDIFF, LDONE)
               ENDIF

               IF (IO%LOPTICS) THEN
                  CALL INPROD_W(WDIFF,W,CHAMD(:,:,:,:,IDIR), &
                       W%WDES%LOVERL,IO%IU0)
               ENDIF
               DO ISP=1,W%WDES%ISPIN
                  DO NK=1,W%WDES%NKPTS
                     DO NB=1,W%WDES%NB_TOT
                        EDER(NB,NK,ISP,IDIR)=REAL(WDIFF%CELTOT(NB,NK,ISP),KIND=q)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO
         ENDIF

         IF(LUSEPEAD()) THEN
            KPOINTS_FULL = KPOINTS_FULL_TEMP            
         ENDIF
         
         DO N=1,KPOINTS%NKPTS
            NK=KPOINT_IN_FULL_GRID_KINTER(W%WDES%VKPT(:,N),KPOINTS_INTER)
            IF (NK/=-1) THEN
               DO IDIR=1,3
                  KINTERPOL(IDIR,NK)=W%WDES%VKPT(IDIR,N)
               ENDDO
               DO ISP=1,W%WDES%ISPIN
                  DO NB=1,W%WDES%NB_TOT
                     EINTERPOL(NB,NK,ISP,1)=REAL(W%CELTOT(NB,N,ISP),KIND=q)
                     W_INTER%CELTOT(NB,NK,ISP)=W%CELTOT(NB,N,ISP)
                     W_INTER%FERTOT(NB,NK,ISP)=W%FERTOT(NB,N,ISP)
                     IF (LDO_VEL) THEN
                        DO IDIR=1,3
                           IF (IO%LOPTICS) THEN
                              DO NB2=1,W%WDES%NB_TOT
                                 CHAM(NB,NB2,NK,ISP,IDIR) = &
                                      CHAMD(NB,NB2,N,ISP,IDIR)
                              ENDDO
                           ENDIF
                           EINTERPOL(NB,NK,ISP,IDIR+1)=EDER(NB,N,ISP,IDIR)
                        ENDDO
                     ENDIF
                  ENDDO
               ENDDO
            ENDIF
         ENDDO
         DISPINDEX=DISPINDEX+1
      ENDDO

      IF (.NOT. LDO_VEL) THEN
         CALL XML_TAG("eigenvalues", comment="interpolated")
         CALL XML_TAG("kpoints")
         CALL XML_KPOINTS_LIST(KINTERPOL, KPOINTS_INTER%WTKPT)
         CALL XML_CLOSE_TAG("kpoints")
         CALL XML_EIGENVALUES_EXT(EINTERPOL,1, W%WDES%NB_TOT, &
              KPOINTS_INTER%NKPTS, W%WDES%ISPIN)
         CALL XML_CLOSE_TAG("eigenvalues")
      ENDIF

      IF (IO%LOPTICS) THEN
         NBANDS_CDER=MIN(LAST_FILLED_OPTICS(W_INTER)*2, W_INTER%WDES%NB_TOT)
         IF (IO%IU0>=0) WRITE(IO%IU0,*) 'Calculating the imaginary ' // &
              'part of the dielectric function.'
         
         DO IDIR=1,3
            DO JDIR=1,3
               CALL EPSILON_IMAG(W_INTER%WDES, W_INTER, CHAM(:,:,:,:,IDIR), &
                    CHAM(:,:,:,:,JDIR), EINTERPOL(:,:,:,IDIR+1), &
                    EINTERPOL(:,:,:,JDIR+1), EFERMI, NEDOS_DIEL, &
                    EPSDD(:,IDIR,JDIR), EPSJJ(:,IDIR,JDIR), DELTAE, &
                    KPOINTS%ISMEAR, KPOINTS%SIGMA, LATT_CUR%OMEGA, &
                    WPLASM(IDIR, JDIR), COND(IDIR, JDIR), &
                    BIMAG(IDIR, JDIR), RTIME, LCSHIFT)
            ENDDO
         ENDDO
         
         IF (KPOINTS%ISMEAR >= -1) THEN
            CALLMPI( M_sum_z(W%WDES%COMM, EPSDD, SIZE(EPSDD)))
            CALLMPI( M_sum_z(W%WDES%COMM, EPSJJ, SIZE(EPSJJ)))
            CALLMPI( M_sum_d(W%WDES%COMM, BIMAG, SIZE(BIMAG)))
         ENDIF
         

         ! eFL: MARTIJN, IS THE ISYMOP AND NROTK OKEY HERE FOR GENERAL CASES?
         DO I=1,NEDOS_DIEL
            IF (SYMM%ISYM>0) THEN
               CALL TSYM_CMPLX(EPSDD(I,:,:),ISYMOP,NROTK,LATT_CUR%A)
            ENDIF
            IF (SYMM%ISYM>0) THEN
               CALL TSYM_CMPLX(EPSJJ(I,:,:),ISYMOP,NROTK,LATT_CUR%A)
            ENDIF
         ENDDO
         
         IF (SYMM%ISYM>0) CALL TSYM(WPLASM(:,:),ISYMOP,NROTK,LATT_CUR%A)
         IF (SYMM%ISYM>0) CALL TSYM(BIMAG(:,:),ISYMOP,NROTK,LATT_CUR%A)
      
      
         
         IWINDOW=0
         LWARN=.FALSE.
         IF (IO%IU0>=0) WRITE(IO%IU0,*) 'Calculating the real ' // &
              'part of the dielectric function.'
         DO IDIR=1,3
            DO JDIR=1,3
               CALL EPSILON_REAL(W%WDES%COMM, NEDOS_DIEL, &
                    EPSDD(:,IDIR,JDIR), LWARN, DELTAE, LCSHIFT, IWINDOW)
               IF (JDIR==IDIR) THEN
                  EPSDD(:,IDIR,IDIR)=EPSDD(:,IDIR,IDIR)+1
               ENDIF
               CALL EPSILON_REAL(W%WDES%COMM, NEDOS_DIEL, &
                    EPSJJ(:,IDIR,JDIR), LWARN, DELTAE, LCSHIFT, IWINDOW)
            ENDDO
         ENDDO
         IF (LWARN) THEN
            CALL VTUTOR('W','SMALL CSHIFT',0.0_q,1, &
                 &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
            CALL VTUTOR('W','SMALL CSHIFT',0.0_q,1, &
                 &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
         ENDIF
         
         DO IDIR=1,3
            DO JDIR=1,3
               IF (IDIR==JDIR) CYCLE
#ifdef timeordered
               DO I=2,NEDOS_DIEL
                  EPSDD(I,IDIR,JDIR)=EPSDD(I,IDIR,JDIR)+ &
                       (0._q,1._q)*BIMAG(IDIR,JDIR)/(DELTAE*(I-1))
               ENDDO
               EPSDD(1,IDIR,JDIR)=EPSDD(2,IDIR,JDIR)
#else
               DO I=1,NEDOS_DIEL
                  EPSDD(I,IDIR,JDIR)=EPSDD(I,IDIR,JDIR)+ &
                       (0._q,1._q)*BIMAG(IDIR,JDIR)/(DELTAE*(I-1)+ &
                       CMPLX(0,LCSHIFT,q))
               ENDDO
#endif
            ENDDO
         ENDDO
         DO IDIR=1,3
            DO JDIR=1,3
               DO I=2,NEDOS_DIEL
                  EPSJJ(I,IDIR,JDIR)=(EPSJJ(I,IDIR,JDIR)- &
                       EPSJJ(1,IDIR,JDIR))/(DELTAE*(I-1))**2
               ENDDO
               IF (JDIR==IDIR) EPSJJ(:,IDIR,IDIR)=EPSJJ(:,IDIR,IDIR)+1
            ENDDO
         ENDDO
         EPSJJ(1,:,:)=EPSJJ(2,:,:)
         IF (WPLASMAI>0) THEN
            DO IDIR=1,3
               DO I=1,NEDOS_DIEL
                  WO=DELTAE*(I-1)
                  EPSDD(I,IDIR,IDIR)=EPSDD(I,IDIR,IDIR)+ &
                       WPLASMON(IDIR,IDIR)/(WO**2+WPLASMAI**2)* &
                       CMPLX(-1.0_q,WPLASMAI/MAX(WO,DELTAE))
                  EPSJJ(I,IDIR,IDIR)=EPSJJ(I,IDIR,IDIR)+ &
                       WPLASMON(IDIR,IDIR)/(WO**2+WPLASMAI**2)* &
                       CMPLX(-1.0_q,WPLASMAI/MAX(WO,DELTAE))
               ENDDO
            ENDDO
         ENDIF
         
         CALL XML_EPSILON_W(DELTAE, REAL(EPSDD,q), AIMAG(EPSDD), NEDOS_DIEL)
         CALL XML_EPSILON_W(DELTAE, REAL(EPSJJ,q), AIMAG(EPSJJ), NEDOS_DIEL)
      ENDIF
      
      IF (LVEL) THEN
         ALLOCATE(EINTERPOL_FULL(4,W%WDES%NB_TOT,KPOINTS_INTER_FULL%NKPTS, &
              W%WDES%ISPIN))
         DO N=1,KPOINTS_INTER_FULL%NKPTS
            NK=KPOINTS_INTER_FULL%NEQUIV(N)
            S=0
            DO L=1,3
               DO K=1,3
                  DO J=1,3
                     DO I=1,3
                        S(L,I)=S(L,I)+LATT_CUR%A(L,K)* &
                             KPOINTS_INTER_FULL%ISYMOP(J,K,N)*LATT_CUR%B(I,J)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO
            DO ISP=1,W%WDES%ISPIN
               DO NB=1,W%WDES%NB_TOT
                  EINTERPOL_FULL(1,NB,N,ISP)=EINTERPOL(NB,NK,ISP,1)
                  DER=0
                  DO IDIR=1,3
                     DER(:)=DER(:)+S(:,IDIR)*EINTERPOL(NB,NK,ISP,IDIR+1)
                  ENDDO
                  DO IDIR=1,3
                     EINTERPOL_FULL(IDIR+1,NB,N,ISP)=DER(IDIR)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         CALL XML_TAG("eigenvelocities", comment="interpolated")
         CALL XML_TAG("kpoints")
         CALL XML_KPOINTS_LIST(KPOINTS_INTER_FULL%VKPT, &
              KPOINTS_INTER_FULL%WTKPT)
         CALL XML_CLOSE_TAG("kpoints")
         CALL XML_EIGENVALUES_EXT(EINTERPOL_FULL, 4, W%WDES%NB_TOT, &
              KPOINTS_INTER_FULL%NKPTS, W%WDES%ISPIN)
         CALL XML_CLOSE_TAG("eigenvelocities")

         DEALLOCATE(EINTERPOL_FULL,KPOINTS_INTER_FULL)
      ENDIF
      
      CALL DENSTA( IO%IU0, IO%IU6, WDES_INTER, W_INTER, KPOINTS_INTER, &
           INFO%NELECT, INFO%NUP_DOWN, ENTROPY, EFERMI, KPOINTS_INTER%SIGMA, &
           .TRUE., NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
      CALL XML_DOS(EFERMI, KPOINTS%EMIN, KPOINTS%EMAX, .FALSE., &
           DOS, DOSI, DOSPAR, NEDOS, 1, 1, W%WDES%NCDIJ, &
           comment='interpolated')

      EBANDSTR_INTER = BANDSTRUCTURE_ENERGY(WDES_INTER, W_INTER)
      OTHERTERMS = E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+ &
           E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF
      TOTENINTER = EBANDSTR_INTER+OTHERTERMS
      TOTENRDIFF = (TOTENORG - TOTENINTER)/TOTENORG

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,1000) TOTENINTER, TOTENRDIFF
         WRITE(17,1000)     TOTENINTER, TOTENRDIFF
      ENDIF
1000  FORMAT('Energy from interpolated bandstructure: ',E20.12/ &
             '                 (relative difference): ',E20.12)
      
      DO ISP=1,W%WDES%ISPIN
         DO NK=1,KPOINTS%NKPTS
            IF (ISP==1) THEN
               W%WDES%VKPT(:,NK)=VKPT_OLD(:,NK)
            ENDIF
            DO N=1,W%WDES%NBANDS
               W%CW(:,N,NK,ISP)=CW_OLD(:,N,NK,ISP)
            ENDDO
         ENDDO
      ENDDO
      CALL SET_DATAKE(W%WDES, LATT_CUR%B)
      IF (NONLR_S%LREAL) THEN
         CALL RSPHER(GRID,NONLR_S,LATT_CUR)
      ELSE
         CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
      ENDIF
      CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W)
! should not be necessary
!     CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)        

      IF (LDO_VEL) THEN
         DEALLOCATE(EDER)
         CALL DEALLOCW(WDIFF)
      ENDIF
      IF (IO%LOPTICS) THEN
         DEALLOCATE(EPSDD, EPSJJ, &
         CHAMD, CHAM, WPLASM, COND, BIMAG)
      ENDIF
      DEALLOCATE(EINTERPOL, KINTERPOL, CW_OLD, VKPT_OLD)
      CALL DEALLOCW_NOPLANEWAVE(W_INTER)

      IF (IO%IU0>=0) WRITE(IO%IU0,*) '------------------------------------------------'// &
           '--------------'
      
    END SUBROUTINE INTERPOLATE_BANDSTR_GRDSHFT


  SUBROUTINE GENERATE_GRDSHFT_GRD_AND_DISPL(KPOINTS, WDES, LATT_CUR, &
       T_INFO, SYMM, IO, KINTER, KPOINTS_INTER, KPOINTS_INTER_FULL, DISPL)
    USE lattice
    USE base
    USE wave
    USE poscar
    USE mkpoints
    USE full_kpoints
    IMPLICIT NONE

    
    TYPE (kpoints_struct)           KPOINTS
    TYPE (kpoints_struct)           KPOINTS_INTER
    TYPE (skpoints_full),POINTER :: KPOINTS_INTER_FULL
    TYPE (skpoints_full),POINTER :: KPOINTS_FULL_TEMP  
    TYPE (latt)                     LATT_CUR
    TYPE (info_struct)              INFO
    TYPE (in_struct)                IO
    TYPE (symmetry)                 SYMM
    TYPE (type_info)                T_INFO
    ! eFL: CAN WE GET RID OF THIS WDES FROM THIS ROUTINE?
    TYPE (wavedes)                  WDES

    INTEGER KINTER, ABSKINTER
    INTEGER, DIMENSION(3) :: MULTIPL
    INTEGER I, N, NK, IWZ, IK, IKX, IKY, IKZ, IKXP, IKYP, IKZP, NUMDISPL
    REAL(q) DIST, DIST_FOUND, KSTART
    REAL(q) DISPLV(3), DISPLV_CART(3)
    REAL(q), ALLOCATABLE :: DISPL(:,:)
    LOGICAL LDONE
    LOGICAL, ALLOCATABLE :: K_FOUND_IN_GRID(:)
    CHARACTER (LEN=8) :: XMLTAG
    
    DISPLV=0.0
    IF (KPOINT_IN_FULL_GRID_KINTER(DISPLV,KPOINTS)==-1)  THEN
       CALL VTUTOR('E','GAMMAC',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU6,3)
       CALL VTUTOR('S','GAMMAC',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU0,3)
       RETURN
    ENDIF
    IF ((MOD(KPOINTS%NKPX,2)==0) .OR. (MOD(KPOINTS%NKPY,2)==0) &
         .OR. (MOD(KPOINTS%NKPZ,2)==0)) THEN
       CALL VTUTOR('E','NOODDGRID',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU6,3)
       CALL VTUTOR('S','NOODDGRID',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU0,3)
       RETURN
    ENDIF
    IF (MOD(ABS(KINTER),2)==0) THEN
       CALL VTUTOR('E','GRDSHFT_K',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU6,3)
       CALL VTUTOR('S','GRDSHFT_K',0.0_q,1,0,1,(0.0_q,0.0_q),1, &
            .TRUE.,1, IO%IU0,3)
       RETURN
    ENDIF

    ABSKINTER = ABS(KINTER)
    NUMDISPL = ABSKINTER * ABSKINTER * ABSKINTER
    KPOINTS_INTER=KPOINTS
    CALL CHECK_FULL_KPOINTS
    MULTIPL=ABSKINTER
#ifdef oldsym
    CALL RD_KPOINTS(KPOINTS_INTER, LATT_CUR, &
         SYMM%ISYM>=0.AND..NOT. WDES%LSORBIT.AND..NOT. WDES%LSPIRAL, &
         SYMM%ISYM<0, -1, -1, MULTIPL = MULTIPL)
#else
    CALL RD_KPOINTS(KPOINTS_INTER, LATT_CUR, &
         SYMM%ISYM>=0.AND..NOT. WDES%LNONCOLLINEAR, SYMM%ISYM<0, &
         -1, -1, MULTIPL = MULTIPL)
#endif

    ALLOCATE(KPOINTS_INTER_FULL)
    CALL IBZKPT_HF(LATT_CUR, KPOINTS_INTER, &
         KPOINTS_INTER_FULL, T_INFO%NIONS, SYMM%ROTMAP, &
         SYMM%MAGROT,SYMM%ISYM, IO%IU6, IO%IU0)
    
    IWZ=2
    KSTART=-REAL(ABSKINTER,q)/2+0.5_q
    IK=0
    ALLOCATE(DISPL(3,NUMDISPL))
    DISPL=0.0
    DO IKX=0,ABSKINTER-1
       DO IKY=0,ABSKINTER-1
          DO IKZ=0,ABSKINTER-1
             IK=IK+1
             DIST_FOUND=1E6
             DO IKXP=-IWZ,IWZ
                DO IKYP=-IWZ,IWZ
                   DO IKZP=-IWZ,IWZ
                      DISPLV(1)=((IKX+KSTART)/ABSKINTER+IKXP)/KPOINTS%NKPX
                      DISPLV(2)=((IKY+KSTART)/ABSKINTER+IKYP)/KPOINTS%NKPY
                      DISPLV(3)=((IKZ+KSTART)/ABSKINTER+IKZP)/KPOINTS%NKPZ
                      DISPLV_CART=DISPLV
                      CALL DIRKAR(1, DISPLV_CART(1), LATT_CUR%B)
                      DIST=DISPLV_CART(1)**2+ &
                           DISPLV_CART(2)**2+ &
                           DISPLV_CART(3)**2
                      IF (DIST<=DIST_FOUND) THEN
                         DISPL(:,IK)=DISPLV
                         DIST_FOUND =DIST
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ALLOCATE(K_FOUND_IN_GRID(KPOINTS_INTER%NKPTS))
    K_FOUND_IN_GRID=.FALSE.
    DO IK=1,NUMDISPL
       DO N=1,KPOINTS%NKPTS
          NK=KPOINT_IN_FULL_GRID_KINTER(KPOINTS%VKPT(:,N)+DISPL(:,IK), &
               KPOINTS_INTER)
          IF (NK/=-1) K_FOUND_IN_GRID(NK)=.TRUE.
       ENDDO
    ENDDO

    LDONE=.FALSE.
    DO IK=1,NUMDISPL
       DO N=1,KPOINTS%NKPTS
          NK=KPOINT_IN_FULL_GRID_KINTER(KPOINTS%VKPT(:,N)+DISPL(:,IK), &
               KPOINTS_INTER)
          IF (NK==-1) THEN
             I=KPOINT_IN_FULL_GRID(KPOINTS%VKPT(:,N)+DISPL(:,IK), &
                  KPOINTS_INTER_FULL)
             NK=KPOINTS_INTER_FULL%NEQUIV(I)
             IF (.NOT. K_FOUND_IN_GRID(NK)) THEN
                KPOINTS_INTER%VKPT(:,NK)=KPOINTS_INTER_FULL%VKPT(:,I)
                K_FOUND_IN_GRID(NK)=.TRUE.
                LDONE=.TRUE.
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    
    DO NK=1, KPOINTS_INTER%NKPTS
       IF (.NOT. K_FOUND_IN_GRID(NK) ) THEN
          IF (IO%IU0>=0) WRITE(IO%IU0,'(I6,3F14.7)') NK, &
               KPOINTS_INTER%VKPT(:,NK)
          CALL VTUTOR('E','GRDSHFT_NOT_IN_IBZ',0.0_q,1,0,1, &
               (0.0_q,0.0_q),1, .TRUE.,1, IO%IU6,3)
          CALL VTUTOR('S','GRDSHFT_NOT_IN_IBZ',0.0_q,1,0,1, &
               (0.0_q,0.0_q),1, .TRUE.,1, IO%IU0,3)
          STOP
       ENDIF
    ENDDO

    IF (LDONE) THEN
       CALL IBZKPT_HF(LATT_CUR, KPOINTS_INTER, &
            KPOINTS_INTER_FULL, T_INFO%NIONS, SYMM%ROTMAP, &
            SYMM%MAGROT,SYMM%ISYM, IO%IU6, IO%IU0)
    ENDIF
    
  END SUBROUTINE GENERATE_GRDSHFT_GRD_AND_DISPL

    
END MODULE mlr_main
