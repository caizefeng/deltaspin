#include "symbol.inc"
!**********************************************************************
! RCS:  $Id: electron.F,v 1.12 2003/06/27 13:22:15 kresse Exp kresse $
!
! subroutine for performing electronic minimization using the
! KLI/ LHF or EXX method
! under development
!
!**********************************************************************

SUBROUTINE ELMIN_OEP( &
     HAMILTONIAN,KINEDEN, &
     P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
     T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
     GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
     CHTOT,CVTOTL,DENCOR,CVTOT,CSTRF, &
     CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,DLM_EXX_LAST, &
     CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
     NSTEP,LMDIM,IRDMAX,NEDOS, &
     TOTEN,EFERMI,LDIMP,LMDIMP)

  USE prec
  USE charge
  USE pseudo
  USE lattice
  USE steep
  USE us
  USE pot
  USE force
  USE fileio
  USE nonl_high
  USE rmm_diis
  USE david
  USE ini
  USE ebs
  USE wave_high
  USE choleski
  USE mwavpre
  USE mwavpre_noio
  USE msphpro
  USE broyden
  USE msymmetry
  USE subrot
  USE melf
  USE base
  USE mpimy
  USE mgrid
  USE mkpoints
  USE constant
  USE setexm
  USE poscar
  USE wave
  USE hamil_high
  USE pawm
  USE cl
  USE vaspxml
  USE mdipol
  USE pawfock
  USE pawfock_inter
  USE Constrained_M_modular
  USE ini
  USE pwkli
  USE pawkli
  USE fock
  USE LDAPLUSU_MODULE
  USE hamil_lrf
  USE subrot_cluster
  USE subrot_lr
  USE rmm_diis_lr
  USE gridq
  USE meta
! solvation__
      USE solvation
! solvation__
  IMPLICIT NONE
  !=======================================================================
  !  structures
  !=======================================================================
  TYPE (ham_handle)  HAMILTONIAN
  TYPE (tau_handle)  KINEDEN
  TYPE (type_info)   T_INFO
  TYPE (potcar)      P(T_INFO%NTYP)
  TYPE (wavedes)     WDES
  TYPE (nonlr_struct) NONLR_S
  TYPE (nonl_struct) NONL_S
  TYPE (wavespin)    W          ! wavefunction
  TYPE (wavespin)    W_F        ! wavefunction for all bands simultaneous
  TYPE (wavespin)    W_G        ! same as above
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
  TYPE (energy)      E_TMP
  TYPE (latt)        LATT_INI

  INTEGER NSTEP,LMDIM,IRDMAX,NEDOS,ITMP
  REAL(q) :: TOTEN,EFERMI

  COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
  RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
  COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
  COMPLEX(q)  CVTOTL(GRIDC%MPLWV,WDES%NCDIJ)! previous potential
  COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor

  !   augmentation related quantities
  OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
       CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
       CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
  !  paw sphere charge density
  INTEGER N_MIX_PAW
  REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ),DLM_EXX_LAST(N_MIX_PAW,WDES%NCDIJ)
  !  charge-density and potential on soft grid
  COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
  RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
  !  density of states
  REAL(q)    DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
  !  Hamiltonian
  GDEF       CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN), &
       CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
  REAL(q) :: XCSIF(3,3), ECONV
  INTEGER  :: NSIM, NELM
  ! local 
  REAL(q) :: TOTENL=0
  REAL(q) :: DESUM1,DESUM(INFO%NELM),DESUM_,EXHF_DUMMY
  INTEGER :: IONODE, NODE_ME
  !  needed temporary for aspherical GGA calculation 
  OVERLAP,ALLOCATABLE ::  CDIJ_TMP(:,:,:,:)
  ! local l-projected wavefunction characters (not really used here)
  REAL(q)    PAR(1,1,1,1,WDES%NCDIJ),DOSPAR(1,1,1,WDES%NCDIJ)

  REAL(q), EXTERNAL :: RHO0
  INTEGER N,ISP,ICONJU,IROT,ICEL,I,II,IRDMAA, &
       IERR,IDUM,IFLAG,ICOUEV,ICOUEV_,NN,NORDER,IERRBR,L,LP, &
       NCOUNT
  REAL(q) BTRIAL,RDUM,RMS,RMS_,ORT,TOTEN2,RMS2,RMST, &
       WEIGHT,BETATO,DESUM2,RMSC,RMSP,X
  REAL(q) RHOAUG(WDES%NCDIJ),RHOTOT(WDES%NCDIJ)
  COMPLEX(q) CDUM
  CHARACTER (LEN=1) CHARAC
  LOGICAL LDELAY
  ! MetaGGA (Robin Hirschl)
  COMPLEX(q),ALLOCATABLE:: TAU(:,:)
  COMPLEX(q),ALLOCATABLE:: TAUW(:,:)      
  ! parameters for FAST_SPHPRO
  INTEGER :: LDIMP,LMDIMP
  ! local exchange potential
  TYPE (GRIDQUANT) :: POT_EXX
  OVERLAP     CDIJ_EXX(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
  OVERLAP     CDIJ_EXX0(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!tst
  REAL(q)     DLM_EXX(N_MIX_PAW,WDES%NCDIJ)
  LOGICAL  :: LKINETIC=.TRUE.
! use a preconditioning based on the inverse of a model dielectric function
  LOGICAL  :: LPOT_FROM_CHARGE=.FALSE.
!  LOGICAL  :: LPOT_FROM_CHARGE=.TRUE.

  OVERLAP     CDIJ1(LMDIM,LMDIM,WDES%NIONS, WDES%NCDIJ)
  RGRID       SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
  ! CHDEN1 could be replaced by CHDEN
  COMPLEX(q)  CHDEN1(GRID_SOFT%MPLWV,WDES%NCDIJ)
  COMPLEX(q)  CVTOT1(GRIDC%MPLWV,WDES%NCDIJ)  ! change in potential
  TYPE (wavespin)     :: WXI           ! stores H(1) - epsilon S(1) | phi_0>
  TYPE (wavespin)     :: W1            ! first order change of wavefunction
  TYPE (wavefun)      :: WTMP          ! temporary

  TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
  REAL(q) EBANDSTR, ENTROPY1, ENTROPY2, TOTEN_LR, TOTENL_LR
  INTEGER :: NLINEAR, IERROR
!test
  REAL(q)  DISPL(3,T_INFO%NIONS)
!test

  ! complex shift not required since no change of overlap
  ! and no change of symmetry
  REAL(q) :: CSHIFT=0.00 
  ! step width for derivative of occupancies
  REAL(q) :: DELTA=1E-4_q
!=======================================================================
! initialisation
!=======================================================================
  IONODE=0
  NODE_ME=0
#ifdef MPI
  IONODE  = WDES%COMM%IONODE
  NODE_ME = WDES%COMM%NODE_ME
  IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
      CALLMPI( M_stop('ELMIN_OEP: KPAR>1 not implemented, sorry.') )
!PK Look for inspiration in electron_all
      STOP
  END IF
#endif

  CALL ALLOCW(WDES,WXI,WTMP,WTMP)
  CALL ALLOCW(WDES,W1,WTMP,WTMP)

  W1%CW=0
  W1%CPROJ=0

  INFO%LCORR=.TRUE.

  NELM=INFO%NELM
  ! to make timing more sensefull syncronize now
  CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror ))
  CALL START_TIMING("LOOP")

  io_begin
  IF (IO%IU0>=0) WRITE(IO%IU0,142)
  WRITE(17,142)
142 FORMAT('       N       E                     dE             ' &
       ,'d eps       ncg     rms          rms(c)')
  io_end

  DESUM1=0
  INFO%LMIX=.FALSE.

130 FORMAT (5X, //, &
       &'----------------------------------------------------', &
       &'----------------------------------------------------'//)


 140  FORMAT (5X, //, &
     &'----------------------------------------- Iteration ', &
     &I4,'(',I4,')  ---------------------------------------'//)
  DWRITE0 'electron entered'

  CALL DIPOL_RESET()

  CALL ALLOCATE_GRID_QUANTITY(POT_EXX,  GRID_SOFT, WDES%NCDIJ)

! same mixing for up and down channel
  MIX%MIXPRE=MOD(MIX%MIXPRE,10)+10
!=======================================================================
! 
! set potential
!
!=======================================================================
  IF (INFO%LPOTOK) THEN
  ! potential and CDIJ were read from file
     CDIJ_EXX=CDIJ
     ! determine CDIJ from CVTOT
     CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
          LMDIM,CDIJ_EXX,CQIJ,CVTOT,IRDMAA,IRDMAX)
     ! CDIJ_EXX should store all one-center contributions:
     ! i.e. from the LHF and from the Hartree terms
     !
     ! subtract the plane wave part from the CDIJ read from the file 
     ! to obtain the one-centre contribution only and store them in CDIJ_EXX
     ! this will only work if the augmentation scheme has NOT changed
     CDIJ_EXX=CDIJ-CDIJ_EXX
  ELSE
     CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
          INFO,P,T_INFO,E,LATT_CUR, &
          CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)

     ! now determine exact exchange potential (POT_EXX and CDIJ_EXX)
     CALL LHF(HAMILTONIAN,GRID,GRID_SOFT, LATT_CUR, SYMM, T_INFO, INFO, &
          NONLR_S,NONL_S,W,WDES,P, &
          LMDIM, INFO%LOVERL, INFO%LREAL, INFO%LCORE, POT_EXX, CRHODE, CDIJ_EXX, &
          .FALSE., IO%IU0, E%EXHF )
!     CDIJ_EXX=0
!     WRITE(*,*) 'forces DIJ to zero'
     ! fourier transform CVTOT back to rec. space and add POT_EXX to get total CVTOT
     DO ISP=1,WDES%NCDIJ
        CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
        CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, POT_EXX%RG(1,ISP), CVTOT(1,ISP))
! test apply cutoff
! it is really questionable whether this makes any sense
! if a cutoff is applied here the pseudo one-centre terms are no longer
! compatible to the pseudo plane wave part
!        CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT(1,ISP))
     ENDDO
     ! now set SV from CVTOT
     CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)
     
     CALL STOP_TIMING("G",IO%IU6,"POTLOK")
     DWRITE0 'potlok is ok'

     ! determine CDIJ from CVTOT
     CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
          LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)
     
     ! add one centre terms (hartree+local) to CDIJ_EXX
     CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
          WDES%NCDIJ, LMDIM, CDIJ_EXX(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
          E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

     ! add to CDIJ
     CDIJ=CDIJ+CDIJ_EXX
     ! CDIJ_EXX contains all one-centre terms from LHF-OEP plus Hartree-terms
     
     CALL STOP_TIMING("G",IO%IU6,"SETDIJ")
     DWRITE0 'setdij is ok'
     
  ENDIF

  ! apply cutoff to the potential
  ! this is also maybe not very wise since the one-centre term
  ! might become non-compatible with the rest
  ! on the other hand this makes the pw part independent of the initial
  ! start guess
  ! pragmatically one could test which option converges faster
  ! with ENCUTGW in the EXX-OEP calculation
!   DO ISP=1,WDES%NCDIJ
!      CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
!      CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT(1,ISP))
!   ENDDO
!   DO ISP=1,WDES%NCDIJ
!      CALL FFT3D(CVTOT(1,ISP),GRIDC,1)
!   ENDDO


  IF (IO%IU6>=0) THEN
     DO I=1,WDES%NCDIJ
        WRITE(IO%IU6,*) 'one center pseudopotential strength for first ion, spin component:',I
        DO LP=1,P(1)%LMMAX
           WRITE(IO%IU6,'(16(F7.3,1X))') &
                &             (CDIJ_EXX(L,LP,1,I),L=1,MIN(8,P(1)%LMMAX))
           !     &             (REAL(CDIJ_EXX(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
        ENDDO
     ENDDO
  ENDIF

  CDIJ_EXX0=CDIJ_EXX
  ! set DLM_EXX to initial values (required by mixer)
  CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .FALSE.)
  CALL SET_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, WDES%NCDIJ, LMDIM, &
       CDIJ_EXX, DLM_EXX)

  CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .TRUE.)
  
  INFO%LPOTOK=.TRUE.

!=======================================================================
  electron: DO N=1,NELM

     CALL XML_TAG("scstep")

!======================================================================
     io_begin
     WRITE(IO%IU6,140) NSTEP,N
     io_end

     CALL WVREAL(WDES,GRID,W) ! only for gamma some action
     CALL START_TIMING("G")

!======================== SUBROUTINE EDDSPX ============================
!
! these subroutines improve the electronic degrees of freedom
! using band by band schemes
! the Harris functional is used for the calculation
! of the total (free) energy so
! E  =  Tr[ H rho ] - d.c. (from input potential)
!
!=======================================================================
     ! tighten break condition
     X=INFO%DEPER
     INFO%DEPER=MIN(INFO%DEPER,0.1_q)

     DESUM1=0
     RMS   =0
     ICOUEV=0

     LDELAY=.FALSE.
     ! if Davidson and RMM are selected, use Davidsons algorithm during
     ! delay phase
     IF (INFO%LRMM  .AND. INFO%LDAVID .AND. (N <= ABS(INFO%NELMDL) .OR. N==1)) LDELAY=.TRUE.
     ! if LDELAY is set, subspace rotation and orthogonalisations can be bypassed
     ! since they are done by the Davidson algorithm

     !
     ! sub space rotation before eigenvalue optimization
     !
     IF (INFO%LPDIAG .AND. .NOT. LDELAY ) THEN

        IF (INFO%LDIAG) THEN
           IFLAG=3    ! exact diagonalization
        ELSE
           IFLAG=4    ! using Loewdin perturbation theory
        ENDIF
        IF (INFO%IALGO==4) THEN
           IFLAG=1
        ENDIF
        IF (N < ABS(INFO%NELMDL)) IFLAG=13

        CALL START_TIMING("G")

        CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
             LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)

        CALL STOP_TIMING("G",IO%IU6,"EDDIAG",XMLTAG="diag")
        DWRITE0 "eddiag is ok"

     ENDIF
     CALL START_TIMING("G")

     select_algo: IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
        !
        ! RMM-DIIS alogrithm
        !
        CALL EDDRMM(HAMILTONIAN,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
             LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV, SV, IO%IU6,IO%IU0, &
             N < ABS(INFO%NELMDL)-ABS(INFO%NELMDL)/4)
        ! previous line selects  special algorithm during delay

        CALL STOP_TIMING("G",IO%IU6,"RMM-DIIS",XMLTAG="diis")
        DWRITE0 "eddrmm is ok"

     ELSE IF (INFO%LDAVID) THEN
        !
        ! blocked Davidson alogrithm,
        !
        NSIM=WDES%NSIM*2
#ifdef MPI
        NSIM=((WDES%NSIM*2+WDES%COMM_INTER%NCPU-1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
        CALL EDDAV(HAMILTONIAN,P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
                LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV, SV,E%EXHF, IO%IU6,IO%IU0, .FALSE., INFO%LDIAG, .FALSE.)
        
        CALL STOP_TIMING("G",IO%IU6,"EDDAV",XMLTAG="dav")
        DWRITE0 "edddav is ok"

     ELSE IF (INFO%IALGO==5 .OR.INFO%IALGO==6 .OR. &
          &         INFO%IALGO==7 .OR.INFO%IALGO==8 .OR. INFO%IALGO==0) THEN select_algo

        !
        ! CG (Teter, Alan, Payne) potential is fixed !!
        !
        CALL EDSTEP(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
             LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV, SV,IO%IU6, IO%IU0)


        CALL STOP_TIMING("G",IO%IU6,"EDSTEP",XMLTAG="cg")
        DWRITE0 "edstep is ok"

     ENDIF select_algo
     !
     ! orthogonalise all bands (necessary only for residuum-minimizer)
     !
     IF (.NOT.INFO%LORTHO .AND. .NOT. LDELAY) THEN
        CALL START_TIMING("G")

        CALL ORTHCH(WDES,W, INFO%LOVERL, LMDIM,CQIJ)

        CALL STOP_TIMING("G",IO%IU6,"ORTHCH",XMLTAG="orth")
        DWRITE0 "ortch is ok"
     ENDIF
     !
     ! sub space rotation after eigen value optimization
     !
     IF (INFO%LCDIAG .AND. .NOT. LDELAY) THEN

        IF (INFO%LDIAG) THEN
           IFLAG=3
        ELSE
           IFLAG=4
        ENDIF

        CALL START_TIMING("G")
        CALL REDIS_PW_OVER_BANDS(WDES, W)
        CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
             LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)

        CALL STOP_TIMING("G",IO%IU6,"EDDIAG",XMLTAG="diag")
     ENDIF

     ! restore break condition
     INFO%DEPER=X
!=======================================================================
! recalculate the broadened density of states and fermi-weights
! recalculate depletion charge size
!=======================================================================
     CALL START_TIMING("G")
     CALL MRG_CEL(WDES,W)

     E%EENTROPY=0
     DOS=0
     DOSI=0
     CALL DENSTA( IO%IU0, IO%IU6, WDES, W, KPOINTS, INFO%NELECT, &
          INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE., &
          NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
     DWRITE0 "densta is ok"
!=======================================================================
! update charge and potential
!=======================================================================

     INFO%LMIX=.FALSE.
     MIX%NEIG=0
     WXI%FERTOT=W%FERTOT
     WXI%CELTOT=W%CELTOT

     IF (.NOT. INFO%LCHCON .AND. N >= ABS(INFO%NELMDL)) THEN
        CALL START_TIMING("G")

        INFO%LPOTOK=.FALSE.

        !
        ! get new charge and new local potential
        ! 
!test (uncomment to keep charge fixed)
        CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
             GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
             LATT_CUR, P, SYMM, T_INFO, &
             CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)
!test
        DWRITE0 "set_charge is ok"

        CALL STOP_TIMING("G",IO%IU6,"CHARGE")

        CALL START_TIMING("G")
        IF (W%OVER_BAND) THEN
           CALL REDIS_PW_OVER_BANDS(WDES, W)
           CALL STOP_TIMING("G",IO%IU6,"REDIS")
        ENDIF

        ! determine local potential (excludes OEP potential)
        CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
             INFO,P,T_INFO,E,LATT_CUR, &
             CHTOT,CSTRF,CVTOT1, DENCOR, SV1, SOFT_TO_C,XCSIF)
        DWRITE0 "potlok is ok"

        CALL STOP_TIMING("G",IO%IU6,"POTLOK")

        ! determine CDIJ1 from CVTOT1
        CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
             LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)

        ! enforce calculation of HF one center contributions
        LHFCALC_FORCE=.TRUE.
        CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
             WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
             E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
        LHFCALC_FORCE=.FALSE.

        CALL STOP_TIMING("G",IO%IU6,"SETDIJ")

        IF (.NOT. LKINETIC)  THEN
        ! determine potential -V_OEP by calculating the difference to current local potential
        ! and ionic and Hartree contribution to local potential stored in CVTOT1, CDIJ1, and SV1
           CDIJ1 =CDIJ1-CDIJ
           CVTOT1=CVTOT1-CVTOT
           SV1   =SV1-SV
        ELSE
        ! since (T + V_OEP+ V_ion + V_H -epsilon0 S) phi -> 
        ! (V^NL_x - V_OEP) phi = ((V^NL_x + T) + V_ion + V_H - epsilon0 S) phi
        ! one could also use the entire Hamiltonian
        ENDIF

!        WRITE(6,'(10F14.7)') CDIJ1(1:5,1:5,1,1)
!        CALL WRT_RL_LINE(6, GRID_SOFT, SV1)

        ! calculate Fock contribution (possibly T phi)
        CALL FOCK_ACC_ALL(GRID, LATT_CUR, NONLR_S, NONL_S, W, WXI, &
             &    LMDIM, LKINETIC, P , CQIJ, E%EXHF)
        CALL STOP_TIMING("G",IO%IU6,"FOCK")
        DWRITE0 "fock_acc_all is ok"

        ! determine V^NL_x - V_OEP_x
        ! the one centre contributions are properly included since CDIJ1 contains
        ! the one centre Fock contribution
        CALL LRF_HAMIL(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W, WXI, WDES, &
             LMDIM, CDIJ1, CQIJ, SV1, RMS_, ICOUEV_)
        DWRITE0 "lrf_hamil is ok"

        ! this works only for LKINETIC=.TRUE., WXI%CELEN is however not used
        CALL MRG_CEL(WDES,WXI)
        WXI%FERTOT=W%FERTOT

        CALL STOP_TIMING("G",IO%IU6,"HAMIL1")

     ENDIF
!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
     E%EBANDSTR=BANDSTRUCTURE_ENERGY(WDES, WXI)
     TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+Ediel_sol
     IF (M_CONSTRAINED()) TOTEN=TOTEN+E_CONSTRAINT()

     DESUM(N)=TOTEN-TOTENL
     ECONV=DESUM(N)
!=======================================================================
! linear response theory with respect to 
! difference in potentials as stored in SV1 and CDIJ1
!=======================================================================
     IF (.NOT. INFO%LPOTOK) THEN
!=======================================================================
!
!  calculate the first order change of the wavefunction
!    H(0) - epsilon S(0) | phi_1> = - |xi>
!  the first order change of the energy is returned in WXI%CELEN
!  the first order change of the norm in WXI%FERWE
!
!=======================================================================
        NULLIFY(DEG_CLUSTER)
        CALL FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER)
        CALL REINIT_DEG_CLUSTERS(WDES,DEG_CLUSTER)
        W1%CW=0
        W1%CPROJ=0

        DWRITE0 "enter linear_response_diis"

        lr: DO NLINEAR=1,6
           TOTENL_LR=TOTEN_LR
           ! consider LRESET more carefully
           CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W1,WXI,W,WDES, &
                LMDIM,CDIJ,CQIJ, RMS_,DESUM_,ICOUEV_, SV, CSHIFT, IO%IU6,IO%IU0, &
                LRESET=NLINEAR==1, IERROR=IERROR)

           IF (IERROR>0 .AND. NLINEAR>=3) THEN
              IF (IO%IU0>=0) THEN
                 WRITE(IO%IU0,'(A,I6)') 'WARNING: LINEAR_RESPONSE_DIIS required accuracy can not reached for some bands',IERROR
              ENDIF
           ELSE IF (IERROR>0) THEN
              IF (IO%IU0>=0) & 
                 WRITE(IO%IU0,*) 'internal ERROR: LINEAR_RESPONSE_DIIS matrix is zero, required accuracy not reached'
              STOP
           ENDIF
           DWRITE0 "linear_response_diis is ok"

           CALL MRG_CEL(WDES,W1)
           CALL STOP_TIMING("G",IO%IU6,"LRDIIS")

           CALL EDDIAG_LR(W1,W,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE., &
                CSHIFT,IO%IU0,DEG_CLUSTER)
           CALL STOP_TIMING("G",IO%IU6,"LRDIAG")

           ! sum f(0) (<phi(1)|xi> + c.c + <phi(1)| H(0)|phi(1)>)
           W1%FERTOT=W%FERTOT  ! set the occupancy matrix W1 to W
           EBANDSTR= BANDSTRUCTURE_ENERGY(WDES, W1)

           ! occupancies (FERWE) use finite differences

           W1%CELTOT=W%CELTOT+WXI%CELTOT*(DELTA/2)
           CALL DENSTA( IO%IU0, IO%IU6, WDES, W1, KPOINTS, INFO%NELECT, &
                INFO%NUP_DOWN, ENTROPY1, EFERMI, KPOINTS%SIGMA, .FALSE.,  &
                NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
           WXI%FERTOT=W1%FERTOT

           W1%CELTOT=W%CELTOT-WXI%CELTOT*(DELTA/2)
           CALL DENSTA( IO%IU0, IO%IU6, WDES, W1, KPOINTS, INFO%NELECT, &
                INFO%NUP_DOWN, ENTROPY2, EFERMI, KPOINTS%SIGMA, .FALSE.,  &
                NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
           ! finished, store the derivative in W1
           ! and set the eigenvalue array (until this point stored in WXI)
           W1%CELTOT=   WXI%CELTOT
           W1%FERTOT  =(WXI%FERTOT-W1%FERTOT)   *(1/DELTA)
           ENTROPY1=(ENTROPY1-ENTROPY2)*(1/DELTA)

           TOTEN_LR =EBANDSTR+ENTROPY1
           IF (IO%IU0>=0) THEN
              WRITE(IO%IU0,400) NLINEAR, TOTEN_LR, (TOTEN_LR-TOTENL_LR), DESUM_, ICOUEV_, RMS_
              WRITE(17,400)     NLINEAR, TOTEN_LR, (TOTEN_LR-TOTENL_LR), DESUM_, ICOUEV_, RMS_
           ENDIF
400        FORMAT('    RMMLR:',I2,'    ',E20.5,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)

        ENDDO lr
        CALL FREE_DEG_CLUSTERS(WDES,DEG_CLUSTER)

        ! at this point CVTOT1 and CDIJ1 is no longer required
        ! use it to store charge related quantities
#ifdef use_charge_from_orbitals
        W1%CW=W%CW
        W1%CPROJ=W%CPROJ
        W1%FERWE=0
#endif
        ! determine first order change of one center occupancy matrix 
        CALL DEPSUM1(W, W1, WDES, LMDIM, CDIJ1, INFO%LOVERL)

        ! change storage convention to (total, magnetization)
        CALL US_FLIP(WDES, LMDIM, CDIJ1, INFO%LOVERL, .FALSE.)

        ! soft pseudo charge
        CALL SOFT_CHARGE1(GRID,GRID_SOFT,W,W1,WDES, CHDEN1)
        ! change storage convention to (total, magnetization)
        CALL RC_FLIP(CHDEN1, GRID_SOFT, WDES%NCDIJ, .FALSE.)

        ! symmetrisation of soft pseudo charge
        IF (SYMM%ISYM ==2) THEN
           IF (WDES%LNONCOLLINEAR) THEN
              CALL RHOSYM(CHDEN1(1,1),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,1)
              IF (.NOT.WDES%LSPIRAL) &
                   CALL SYMFIELD(CHDEN1(1,2),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
           ELSE
              DO ISP=1,WDES%ISPIN
                 CALL RHOSYM(CHDEN1(1,ISP),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
              ENDDO
           ENDIF
        ENDIF

        CVTOT1=0
        ! add first order change of one center occupancy matrix times
        ! augmentation charge distribution rho(1)_ij Q_ij(r)
        CALL DEPLE_ADD(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
             LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
             LMDIM, CDIJ1, CVTOT1, CHDEN1, IRDMAX)

        ! symmetrise total pseudo charge density CHTOT if required (just in case do it always here)
        IF (SYMM%ISYM >=1) THEN
           IF (WDES%LNONCOLLINEAR) THEN
              CALL RHOSYM(CVTOT1(1,1),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,1)
              IF (.NOT.WDES%LSPIRAL) &
                   &   CALL SYMFIELD(CVTOT1(1,2),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
           ELSE
              DO ISP=1,WDES%ISPIN
                 CALL RHOSYM(CVTOT1(1,ISP),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
              ENDDO
           ENDIF
        ENDIF
        
        ! CVTOT1 is actually density times volume, remove the volume factor
        DO ISP=1,WDES%NCDIJ
           CALL RC_ADD(CVTOT1(1,ISP),1.0_q/LATT_CUR%OMEGA,CVTOT1(1,ISP),0.0_q,CVTOT1(1,ISP),GRIDC)
        ENDDO
#ifdef use_charge_from_orbitals
        CVTOT1=CVTOT1/1000
        CDIJ1=CDIJ1/1000
#endif
!=======================================================================
!
! mix the charge
!
!=======================================================================
#ifndef use_charge_from_orbitals
        ! check for "drift" in potential
        IF (ABS(RHO0(GRIDC, CVTOT1(1,1)))>=1E-2) THEN
           WRITE(*,*) 'internal error in electron_OEP: potential shift observed'
           STOP
        ENDIF
        CALL SET_RHO0(GRID_SOFT, CVTOT1(1,1), 0.0_q)
#endif
        DO ISP=1,WDES%NCDIJ
           ! XINV also applies cutoff similar APPLY_CUTOFF
           CALL XINV(GRIDC,LATT_CUR, LPOT_FROM_CHARGE, CVTOT1(1,ISP), & 
                CDIJ1(1,1,1,ISP), LMDIM, WDES%NIONS)
        ENDDO
        ! at this point the one centre charge density related to CDIJ1 is calculated
        ! this defines the change in the one centre potential, and 
        ! and strength parameters are determined from this charge density
!#define use_charge_from_orbitals
#ifdef use_charge_from_orbitals
        CDIJ_EXX=0
#endif
        IF (EXXOEP==3 .OR. EXXOEP==4) THEN
        CALL SET_DD_OEP(WDES, P , T_INFO, INFO%LOVERL, &
             WDES%NCDIJ, LMDIM, CDIJ_EXX(1,1,1,1), CDIJ1(1,1,1,1) , EXXOEP==3)
        ENDIF
#ifdef use_charge_from_orbitals
!        CVTOT1=0
        ! back to spinor representation
        CALL RC_FLIP_POTENTIAL(CVTOT1,GRIDC, WDES%NCDIJ, .TRUE.)

        ! now set SV from CVTOT
        CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT1)

        CDIJ=0
        DISPL=0
        CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ1,CQIJ,CVTOT1,.FALSE., IRDMAA,IRDMAX, DISPL)

        IF (INFO%LOVERL .AND. IO%IU6>=0 ) THEN
           DO I=1,WDES%NCDIJ
              WRITE(IO%IU6,*) 'CDIJ1, spin component:',I
              DO LP=1,P(1)%LMMAX
                 WRITE(IO%IU6,'(16(F7.3,1X))') &
                      &             (REAL(CDIJ1(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
              ENDDO
           ENDDO
           DO I=1,WDES%NCDIJ
              WRITE(IO%IU6,*) 'CDIJ_EXX, spin component:',I
              DO LP=1,P(1)%LMMAX
                 WRITE(IO%IU6,'(16(F7.3,1X))') &
                      &             (REAL(CDIJ_EXX(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
              ENDDO
           ENDDO
        ENDIF
        CDIJ1=CDIJ1+CDIJ_EXX

        IFLAG=12
        CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
             LMDIM,CDIJ1,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)
        STOP
#endif
        ! current local potential to reciprocal space
        DO ISP=1,WDES%NCDIJ
           CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
        ENDDO

        ! go to total/magnetisation respresentation
        CALL RC_FLIP_POTENTIAL(CVTOT,GRIDC, WDES%NCDIJ, .FALSE.)
        CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .FALSE.)

        ! copy CVTOT to CVTOTL, add CVTOT1 (correction to potential) to CVTOT
        DO ISP=1,WDES%NCDIJ
           CALL RC_ADD(CVTOT(1,ISP),1.0_q,CVTOT(1,ISP),0.0_q,CVTOTL(1,ISP),GRIDC)
           CALL RC_ADD(CVTOT(1,ISP),1.0_q,CVTOT1(1,ISP),1.0_q,CVTOT(1,ISP),GRIDC)
        ENDDO

        ! copy DLM_EXX to DLM_EXX_LAST
        DLM_EXX_LAST=DLM_EXX

        ! update DLM_EXX from the current CDIJ_EXX
        CALL SET_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, WDES%NCDIJ, LMDIM, &
             CDIJ_EXX, DLM_EXX)

        IF (MIX%IMIX/=0) THEN
           CALL START_TIMING("G")
           INFO%LMIX=.TRUE.

           IF (MIX%IMIX==4) THEN
              !  broyden mixing ... :

              CALL BRMIX(KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
                   (2*GRIDC%MPLWV),CVTOT,CVTOTL,WDES%NCDIJ,LATT_CUR%B, &
                   LATT_CUR%OMEGA, N_MIX_PAW, DLM_EXX, DLM_EXX_LAST, &
                   RMST,RMSC,RMSP,WEIGHT,.TRUE.,IERRBR)
              MIX%LRESET=.FALSE.
           ELSE

              !  simple mixing ... :
              RMST=0
              CALL MIX_SIMPLE(GRIDC,MIX,WDES%NCDIJ, CVTOT,CVTOTL, &
                   N_MIX_PAW, DLM_EXX, DLM_EXX_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)

           ENDIF
           CALL STOP_TIMING("G",IO%IU6,"MIXING")
           DWRITE0 "mixing is ok"
        ENDIF

        ! back to spinor representation
        CALL RC_FLIP_POTENTIAL(CVTOT,GRIDC, WDES%NCDIJ, .TRUE.)

        CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)

        ! calculate plane wave contribution to CDIJ
        CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
             LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

        ! one centre corrections have also been passed through mixer
        ! determine CDIJ_EXX from the updated DLM_EXX
        CALL RETRIVE_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, LMDIM, CDIJ_EXX, DLM_EXX)
        ! go to spinor respresentation
        CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .TRUE.)

        ! add one center terms to plane wave part of CDIJ
        CDIJ=CDIJ+CDIJ_EXX


        IF (USELDApU()) CALL LDAPLUSU_PRINTOCC(WDES,T_INFO%NIONS,T_INFO%ITYP,IO%IU6)
        CALL STOP_TIMING("G",IO%IU6,"SETDIJ")
        DWRITE0 'setdij is ok'

        INFO%LPOTOK=.TRUE.
     ENDIF
!=======================================================================
!  write energy
!=======================================================================

     io_begin
     CALL WRITE_CONSTRAINED_M(17,.FALSE.)
     !---- write total energy to OSZICAR file and stdout
303  FORMAT('CG : ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)
1303 FORMAT('RMM: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)
10303 FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)

     IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
        WRITE(17,  1303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
        IF (IO%IU0>=0) &
             WRITE(IO%IU0, 1303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
     ELSE IF (INFO%LDAVID) THEN
        WRITE(17, 10303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
        IF (IO%IU0>=0) &
             WRITE(IO%IU0,10303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
     ELSE
        WRITE(17,   303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
        IF (IO%IU0>=0) &
             WRITE(IO%IU0,  303,ADVANCE='NO')  N,TOTEN,DESUM(N),DESUM1,ICOUEV,RMS
     ENDIF
     CALL STOP_TIMING("G",IO%IU6,"DOS")
     io_end

!=======================================================================
!  Test for Break condition
!=======================================================================
     INFO%LABORT=.FALSE.
     !-----conjugated gradient eigenvalue and energy must be converged
     IF(ABS(DESUM(N))<INFO%EDIFF.AND.ABS(DESUM1)<INFO%EDIFF) INFO%LABORT=.TRUE.
     !-----charge-density not constant and in last cycle no change of charge
     IF (.NOT. INFO%LMIX .AND. .NOT. INFO%LCHCON .AND. MIX%IMIX/=0) INFO%LABORT=.FALSE.
     !-----do not stop during the non-selfconsistent startup phase
     IF (N <= ABS(INFO%NELMDL)) INFO%LABORT=.FALSE.
     !-----do not stop before minimum number of iterations is reached
     IF (N < ABS(INFO%NELMIN)) INFO%LABORT=.FALSE.
     !-----but stop after INFO%NELM steps no matter where we are now
     IF (N>=INFO%NELM) INFO%LABORT=.TRUE.

     IF ((IO%LORBIT>=10).AND.(MOD(N,5)==0).AND.WDES%LNONCOLLINEAR) THEN
        CALL SPHPRO_FAST( &
             GRID,LATT_CUR, P,T_INFO,W, WDES, 71,IO%IU6,&
             INFO%LOVERL,LMDIM,CQIJ, LDIMP, LDIMP,LMDIMP,.FALSE.,IO%LORBIT,PAR)
     ENDIF
! ======================================================================
! If the end of the electronic loop is reached
! calculate accurate initial state core level shifts
! if required
! ======================================================================
     IF (INFO%LABORT .AND. ACCURATE_CORE_LEVEL_SHIFTS()) THEN

        ALLOCATE(CDIJ_TMP(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ))
        CDIJ_TMP=CDIJ

        CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
             WDES%NCDIJ, LMDIM, CDIJ_TMP(1,1,1,1), RHOLM, CRHODE, &
             E, LMETA= .FALSE. , LASPH=INFO%LASPH , LCOREL=.TRUE. )
        DEALLOCATE(CDIJ_TMP)
     ENDIF
!=======================================================================
! total time used for this step
!=======================================================================
     CALL SEPERATOR_TIMING(IO%IU6)
     CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')


!=======================================================================
!  important write statements
!=======================================================================

2440 FORMAT(/' eigenvalue-minimisations  :',I6,/ &
          &       ' total energy-change (2. order) :',E14.7,'  (',E14.7,')')
2441 FORMAT(/ &
          &       ' Broyden mixing:'/ &
          &       '  rms(total) =',E12.5,'    rms(broyden)=',E12.5,/ &
          &       '  rms(prec ) =',E12.5/ &
          &       '  weight for this iteration ',F10.2)

2442 FORMAT(/' eigenvalues of (default mixing * dielectric matrix)' / &
          '  average eigenvalue GAMMA= ',F8.4,/ (10F8.4))

200  FORMAT(' number of electron ',F15.7,' magnetization ',3F15.7)
201  FORMAT(' augmentation part  ',F15.7,' magnetization ',3F15.7)


     DO I=1,WDES%NCDIJ
        RHOTOT(I)=RHO0(GRIDC, CHTOT(1,I))
        RHOAUG(I)=RHOTOT(I)-RHO0(GRID_SOFT, CHDEN(1,I))
     END DO

     io_begin

     ! iteration counts
     WRITE(IO%IU6,2440) ICOUEV,DESUM(N),DESUM1

     ! charge density
     WRITE(IO%IU6,200) RHOTOT
     IF (INFO%LOVERL) THEN
        WRITE(IO%IU6,201) RHOAUG
     ENDIF
     ! dipol moment
     IF (DIP%LCOR_DIP) CALL WRITE_DIP(IO%IU6)

     ! mixing
     IF ( INFO%LMIX .AND. MIX%IMIX==4 ) THEN
        IF (IERRBR/=0) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
                'mixing'' now and reset mixing at next step!'
           WRITE(IO%IU6,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
                'mixing'' now and reset mixing at next step!'
        ENDIF

        IF (IO%NWRITE>=2 .OR. NSTEP==1) THEN
           WRITE(IO%IU6,2441) RMST,RMSC,RMSP,WEIGHT
           IF (ABS(RMST-RMSC)/RMST> 0.1_q) THEN
              WRITE(IO%IU6,*) ' WARNING: grid for Broyden might be to small'
           ENDIF
        ENDIF
        IF (IO%IU0>=0) WRITE(IO%IU0,308) RMST
        WRITE(17,308) RMST
308     FORMAT('   ',E10.3)
        IF (MIX%NEIG > 0) THEN
           WRITE(IO%IU6,2442) MIX%AMEAN,MIX%EIGENVAL(1:MIX%NEIG)
        ENDIF
     ELSE IF (INFO%LMIX) THEN
        IF (IO%IU0>=0) &
             WRITE(IO%IU0,308) RMST
        WRITE(17,308) RMST
     ELSE
        IF (IO%IU0>=0) &
             WRITE(IO%IU0,*)
        WRITE(17 ,*)
     ENDIF
     io1: IF (IO%NWRITE>=2 .OR. (NSTEP==1)) THEN
        ! energy
        NORDER=0 ; IF (KPOINTS%ISMEAR>=0) NORDER=KPOINTS%ISMEAR

        WRITE(IO%IU6,7240) E%PSCENC,E%TEWEN,E%DENC,E%EXHF,E%XCENC,E%PAWPS,E%PAWAE, &
             E%EENTROPY,E%EBANDSTR,INFO%EALLAT,TOTEN, &
             TOTEN-E%EENTROPY,TOTEN-E%EENTROPY/(2+NORDER)

7240    FORMAT(/ &
             ' Free energy of the ion-electron system (eV)'/ &
             &        '  ---------------------------------------------------'/ &
             &        '  alpha Z        PSCENC = ',F18.8/ &
             &        '  Ewald energy   TEWEN  = ',F18.8/ &
             &        '  -Hartree energ DENC   = ',F18.8/ &
             &        '  -exchange  EXHF       = ',F18.8/ &
             &        '  -V(xc)+E(xc)   XCENC  = ',F18.8/ &
             &        '  PAW double counting   = ',2F18.8/ &
             &        '  entropy T*S    EENTRO = ',F18.8/ &
             &        '  eigenvalues    EBANDS = ',F18.8/ &
             &        '  atomic energy  EATOM  = ',F18.8/ &
             &        '  ---------------------------------------------------'/ &
             &        '  free energy    TOTEN  = ',F18.8,' eV'// &
             &        '  energy without entropy =',F18.8, &
             &        '  energy(sigma->0) =',F18.8)
     ELSE io1
        WRITE(IO%IU6,7242) TOTEN,TOTEN-E%EENTROPY
7242    FORMAT(/'  free energy = ',E20.12, &
             &        '  energy without entropy= ',E20.12)

     ENDIF io1

     IF (IO%LOPEN) CALL WFORCE(IO%IU6)
     IF (IO%LOPEN) CALL WFORCE(17)
     WRITE(IO%IU6,130)
     io_end
!=======================================================================
!  perform some additional write statments if required
!=======================================================================
     !-----Eigenvalues and weights
     IF (((NSTEP==1 .OR.NSTEP==DYN%NSW).AND.INFO%LABORT).OR. &
          &     (IO%NWRITE>=1 .AND.INFO%LABORT).OR.IO%NWRITE>=3) THEN

        ! calculate the core level shifts
        IF (INFO%LOVERL) THEN        
           CALL CL_SHIFT_PW( GRIDC, LATT_CUR, IRDMAX,  &
                T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG,INFO%ENMAX), IO%IU6)
        ELSE
           do_io WRITE(*,*) " **** core level shifts not calculated ****"
        ENDIF

        io_begin
        CALL RHOAT0(P,T_INFO, BETATO,LATT_CUR%OMEGA)

        WRITE(IO%IU6,2202) EFERMI,REAL( E%CVZERO ,KIND=q) ,E%PSCENC/INFO%NELECT+BETATO
2202    FORMAT(' E-fermi : ', F8.4,'     XC(G=0): ',F8.4, &
             &         '     alpha+bet :',F8.4/)

        CALL WRITE_EIGENVAL( WDES, W, IO%IU6)

        !-----Charge-density along one line
        WRITE(IO%IU6,130)
        DO I=1,WDES%NCDIJ
           WRITE(IO%IU6,*)'soft charge-density along one line, spin component',I
           WRITE(IO%IU6,'(10(6X,I4))') (II,II=0,9)
           CALL WRT_RC_LINE(IO%IU6,GRID_SOFT, CHDEN(1,I))
           IF (INFO%LOVERL) THEN
              WRITE(IO%IU6,*)'total charge-density along one line'
              CALL WRT_RC_LINE(IO%IU6,GRIDC, CHTOT(1,I))
           ENDIF
           WRITE(IO%IU6,*)
        ENDDO
        !-----pseudopotential strength and augmentation charge
        DO I=1,WDES%NCDIJ
           WRITE(IO%IU6,*) 'one center pseudopotential strength for first ion, spin component:',I
           DO LP=1,P(1)%LMMAX
              WRITE(IO%IU6,'(16(F7.3,1X))') &
                   &             (CDIJ_EXX(L,LP,1,I),L=1,MIN(8,P(1)%LMMAX))
              !     &             (REAL(CDIJ_EXX(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
           ENDDO
        ENDDO

        IF (INFO%LOVERL) THEN
           DO I=1,WDES%NCDIJ
              WRITE(IO%IU6,*) 'total augmentation occupancy for first ion, spin component:',I
              DO LP=1,P(1)%LMMAX
                 WRITE(IO%IU6,'(16(F7.3,1X))') &
                      &             (REAL(CRHODE(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
              ENDDO
           ENDDO
        ENDIF
        io_end

     ENDIF
!=======================================================================
!  xml related output
!=======================================================================
     CALL XML_TAG("energy")
     IF (INFO%LABORT .OR. N==1) THEN
        CALL XML_TAG_REAL("alphaZ",E%PSCENC)
        CALL XML_TAG_REAL("ewald", E%TEWEN)
        CALL XML_TAG_REAL("hartreedc",E%DENC)
        CALL XML_TAG_REAL("XCdc",E%XCENC)
        CALL XML_TAG_REAL("pawpsdc",E%PAWPS)
        CALL XML_TAG_REAL("pawaedc",E%PAWAE)
        CALL XML_TAG_REAL("eentropy",E%EENTROPY)
        CALL XML_TAG_REAL("bandstr",E%EBANDSTR)
        CALL XML_TAG_REAL("atom",INFO%EALLAT)
        CALL XML_ENERGY(TOTEN, TOTEN-E%EENTROPY, TOTEN-E%EENTROPY/(2+NORDER))
     ELSE
        CALL XML_ENERGY(TOTEN, TOTEN-E%EENTROPY, TOTEN-E%EENTROPY/(2+NORDER))
     ENDIF
     CALL XML_CLOSE_TAG


     CALL XML_CLOSE_TAG("scstep")
!======================== end of loop ENDLSC ===========================
! This is the end of the selfconsistent calculation loop
!=======================================================================
     IF (INFO%LABORT) THEN
        io_begin
        WRITE(IO%IU6,131)
131     FORMAT (5X, //, &
             &  '------------------------ aborting loop because EDIFF', &
             &  ' is reached ----------------------------------------'//)
        io_end
        EXIT electron
     ENDIF
     INFO%LSOFT=.FALSE.
#ifndef noSTOPCAR 
#ifndef F90_T3D
     CALL RDATAB(IO%LOPEN,'STOPCAR',99,'LABORT','=','#',';','L', &
          &            IDUM,RDUM,CDUM,INFO%LSOFT,CHARAC,NCOUNT,1,IERR)
     ITMP=0 ; IF (INFO%LSOFT) ITMP=1 ; CALLMPI ( M_sum_i(W%WDES%COMM, ITMP, 1))
     IF (ITMP>0) INFO%LSOFT=.TRUE.
#endif
#endif
     IF (INFO%LSOFT) THEN
        io_begin
        IF (IO%IU0>=0) &
             WRITE(IO%IU0,*) 'hard stop encountered!  aborting job ...'
        WRITE(IO%IU6,13131)
13131   FORMAT (5X, //, &
             &  '------------------------ aborting loop because hard', &
             &  ' stop was set ---------------------------------------'//)
        io_end
        EXIT electron
     ENDIF
     TOTENL=TOTEN

  ENDDO electron

  CALL DEALLOCATE_GRID_QUANTITY(POT_EXX)
!=======================================================================
! write the local potential
!=======================================================================
  IF (IO%LVTOT) THEN
     io_begin
     IF (IO%LOPEN) OPEN(IO%IUVTOT,FILE='POT',STATUS='UNKNOWN')
     REWIND IO%IUVTOT
     CALL OUTPOS(IO%IUVTOT,.FALSE.,INFO%SZNAM1,T_INFO,LATT_CUR%SCALE,LATT_CUR%A,.FALSE.,DYN%POSION)
     io_end

     ! at the moment the spin up and down potential is written to the file
     CALL SET_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, WDES%NCDIJ, LMDIM, &
          CDIJ, DLM_EXX)

     CALL OUTPOT(GRIDC, IO%IUVTOT,.TRUE.,CVTOT)
     CALL WRT_RHO_PAW(P, T_INFO, INFO%LOVERL, DLM_EXX(:,1), GRIDC%COMM, IO%IUVTOT)

     DO ISP=2,WDES%NCDIJ
        do_io WRITE( IO%IUVTOT,'(5E20.12)') (T_INFO%ATOMOM(I),I=1,T_INFO%NIONS)
        CALL OUTPOT(GRIDC, IO%IUVTOT,.TRUE.,CVTOT(1,ISP))

        CALL WRT_RHO_PAW(P, T_INFO, INFO%LOVERL, DLM_EXX(:,ISP), GRIDC%COMM, IO%IUVTOT )
     ENDDO
  ENDIF

  DWRITE0 'electron left'

  RETURN
END SUBROUTINE ELMIN_OEP




!***********************************************************************
!
! Determine the action of the Hamiltonian on all bands and
! store the result in a set of wavefunctions WXI
! 
!***********************************************************************


SUBROUTINE W_IN_ACC(W, WXI, WDES)
  USE prec
  USE wave_mpi
  USE wave
  USE lattice
  USE mpimy
  USE mgrid
  USE nonl_high

  IMPLICIT NONE

  TYPE (wavespin)    W
  TYPE (wavespin)    WXI
  TYPE (wavedes)     WDES
  !    local
  TYPE (wavedes1)    WDES1          ! descriptor for one k-point
  TYPE (wavefun1)    W1             ! current wavefunction
  COMPLEX(q) CDCHF
  INTEGER ISP, NK, NPOS, M, MM, ISPINOR
  REAL(q) SUM

  W%CELEN=0

  spin:  DO ISP=1,WDES%ISPIN
     kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)

        DO NPOS=1,WDES%NBANDS
           DO ISPINOR=0,WDES1%NRSPINORS-1
              DO M=1,WDES1%NGVECTOR
                 MM=M+ISPINOR*WDES1%NGVECTOR
                 W%CELEN(NPOS,NK,ISP)=W%CELEN(NPOS,NK,ISP)+WXI%CW(MM,NPOS,NK,ISP)*CONJG(W%CW(MM,NPOS,NK,ISP))
              ENDDO
           ENDDO
        ENDDO
     END DO kpoint
  END DO spin

  CALL MRG_CEL(WDES,W)

  RETURN
END SUBROUTINE W_IN_ACC


!************************ SUBROUTINE XINV   ****************************
!
! this subroutine calculates the inverse of the Lindhard
! dielectric function and multiplies the potential by the inverse
! additionally it imposes a cutoff (ENMAX) on the potential changes
! for the following reasons
! ) these components are essentially screened out by the Greens-function
!    G= (H(0)-e(0) S(0))^-1
!   and therefore almost impossible to converge
! ) they change the energy only by a very small amount
! ) the Broyden mixer uses a straight mixing for them
!   which can lead to divergence if the components are emphasized
!   too much by the inverse of the dielectric function
!
!***********************************************************************

SUBROUTINE XINV(GRIDC,LATT_CUR, LPOT_FROM_CHARGE, CVTOT, CDIJ, LMDIM, NIONS)
  USE prec
  USE mpimy
  USE mgrid
  USE lattice
  USE constant
  IMPLICIT NONE

  TYPE (grid_3d)     GRIDC
  TYPE (latt)        LATT_CUR
  COMPLEX(q) CVTOT(GRIDC%RC%NP)
  INTEGER    LMDIM, NIONS
  OVERLAP    CDIJ(LMDIM,LMDIM,NIONS)
  REAL(q) ENMAX
  LOGICAL LPOT_FROM_CHARGE
! local
  REAL(q) SCALE, ETA, ETAP2, ENERGI
  INTEGER    NC, N1, N2, N3, NI
  REAL(q), PARAMETER  :: CUTOFF=1E-6_q, KFERMI=2.0_q
  REAL(q) :: GX, GY, GZ, GSQU, XIINV, FACTM
  REAL(q), EXTERNAL :: ENCUTGW_IN_CHI

  ENMAX=ENCUTGW_IN_CHI()
  IF (ENMAX<=0) ENMAX=1E10

  ! scale= 1/ [( omega k_F) /(4 pi^2 e^2  * hbar^2/ 2me)]
  ! volume scaling has already been done in ELMIN_OEP
  SCALE=-(EDEPS*PI*HSQDTM)/KFERMI
  ! scale one center terms
  XIINV   = 0.2*SCALE

  CDIJ=CDIJ*XIINV
!=======================================================================
! calculate the hartree potential on the grid of reciprocal lattice
! vectors and the correction to the total energy
!=======================================================================
  NI=0
  col: DO NC=1,GRIDC%RC%NCOL
     N2= GRIDC%RC%I2(NC)
     N3= GRIDC%RC%I3(NC)
     row: DO N1=1,GRIDC%RC%NROW

        NI=NI+1
        SETFACT1

        SETFACT

        GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
        GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
        GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))

        GSQU=GX**2+GY**2+GZ**2

        ENERGI=HSQDTM*(GSQU)*TPI*TPI
!=======================================================================
! XIINV attenuates the high frequency components which have been
! screened by the application of XI
!=======================================================================
        IF (ENERGI> ENMAX) THEN
           CVTOT(NI)=(0.0_q,0.0_q)
        ELSE
           ETA=SQRT(GSQU*TPI)/KFERMI
           ETAP2 = ETA*ETA
           IF (ABS(ETA-2)<=CUTOFF .OR. .NOT. LPOT_FROM_CHARGE .OR. &
               (GRIDC%LPCTX(N1)==0).AND.(GRIDC%LPCTY(N2)==0).AND.(GRIDC%LPCTZ(N3)==0)) THEN
              XIINV   = 0.2*SCALE
           ELSE
              XIINV   = 0.4*SCALE/((4-ETAP2)/4/ETA*LOG(ABS((2+ETA)/(2-ETA)))+1)
           ENDIF

           CVTOT(NI)=CVTOT(NI)*XIINV
        ENDIF
     ENDDO row
  ENDDO col


END SUBROUTINE XINV


!************************ SUBROUTINE APPLY_CUTOFF **********************
!
! apply a cutoff to the input array 
!
!***********************************************************************

SUBROUTINE APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT)
  USE prec
  USE mpimy
  USE mgrid
  USE lattice
  USE constant
  IMPLICIT NONE

  TYPE (grid_3d)     GRIDC
  TYPE (latt)        LATT_CUR
  COMPLEX(q) CVTOT(GRIDC%RC%NP)
  REAL(q) ENMAX
! local
  REAL(q) ENERGI
  INTEGER    NC, N1, N2, N3, NI
  REAL(q) :: GX, GY, GZ, GSQU, FACTM
  REAL(q), EXTERNAL :: ENCUTGW_IN_CHI

  ENMAX=ENCUTGW_IN_CHI()
  IF (ENMAX<=0) ENMAX=1E10

!=======================================================================
! calculate the hartree potential on the grid of reciprocal lattice
! vectors and the correction to the total energy
!=======================================================================
  NI=0
  col: DO NC=1,GRIDC%RC%NCOL
     N2= GRIDC%RC%I2(NC)
     N3= GRIDC%RC%I3(NC)
     row: DO N1=1,GRIDC%RC%NROW

        NI=NI+1
        SETFACT1

        SETFACT

        GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
        GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
        GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))

        GSQU=GX**2+GY**2+GZ**2

        ENERGI=HSQDTM*(GSQU)*TPI*TPI
        IF (ENERGI> ENMAX) THEN
           CVTOT(NI)=(0.0_q,0.0_q)
        ENDIF
     ENDDO row
  ENDDO col
END SUBROUTINE APPLY_CUTOFF

