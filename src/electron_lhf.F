#include "symbol.inc"
!**********************************************************************
! RCS:  $Id: electron.F,v 1.12 2003/06/27 13:22:15 kresse Exp kresse $
!
! subroutine for performing electronic minimization using the LHF method
!
!**********************************************************************

SUBROUTINE ELMIN_LHF( &
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
  USE pawm
  USE cl
  USE vaspxml
  USE mdipol
  USE pawfock
  USE pawfock_inter
  USE Constrained_M_modular
  USE ini
  USE pwkli
  USE LDAPLUSU_MODULE
  USE gridq
  USE hamil_high
  USE fock
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
  TYPE (latt)        LATT_INI

  INTEGER NSTEP,LMDIM,IRDMAX,NEDOS
  REAL(q) :: TOTEN,EFERMI

  COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
  COMPLEX(q)  CVTOTL(GRIDC%MPLWV,WDES%NCDIJ)! old charge-density
  RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
  COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
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
  TYPE (GRIDQUANT) :: POT_EXX
  !  density of states
  REAL(q)    DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
  !  Hamiltonian
  GDEF       CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN), &
       CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
  REAL(q) :: XCSIF(3,3), ECONV
  INTEGER  :: NSIM, NELM
  ! local 
  REAL(q) :: TOTENL=0
  REAL(q) :: DESUM1,DESUM(INFO%NELM), EXHF_DUMMY
  INTEGER :: IONODE, NODE_ME, ITMP
  !  needed temporary for aspherical GGA calculation 
  OVERLAP,ALLOCATABLE ::  CDIJ_TMP(:,:,:,:)
  ! local l-projected wavefunction characters (not really used here)
  REAL(q)    PAR(1,1,1,1,WDES%NCDIJ),DOSPAR(1,1,1,WDES%NCDIJ)

  REAL(q), EXTERNAL :: RHO0
  INTEGER N,ISP,ICONJU,IROT,ICEL,I,II,IRDMAA, &
       IERR,IDUM,IFLAG,ICOUEV,ICOUEV2,NN,NORDER,IERRBR,L,LP, &
       NCOUNT
  REAL(q) BTRIAL,RDUM,RMS,ORT,TOTEN2,RMS2,RMST, &
       WEIGHT,BETATO,DESUM2,RMSC,RMSP
  REAL(q) RHOAUG(WDES%NCDIJ),RHOTOT(WDES%NCDIJ)
  COMPLEX(q) CDUM
  CHARACTER (LEN=1) CHARAC
  ! MetaGGA (Robin Hirschl)
  COMPLEX(q),ALLOCATABLE:: TAU(:,:)
  COMPLEX(q),ALLOCATABLE:: TAUW(:,:)
  ! parameters for FAST_SPHPRO
  INTEGER :: LDIMP,LMDIMP
  ! local exchange potential
  OVERLAP     CDIJ_EXX(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
  REAL(q)     DLM_EXX(N_MIX_PAW,WDES%NCDIJ)

  ! additional arrays for getting exact energy
  COMPLEX(q)  CHDEN1(GRID_SOFT%MPLWV,WDES%NCDIJ)
  COMPLEX(q)  CVTOT1(GRIDC%MPLWV,WDES%NCDIJ)  ! change in potential
  RGRID       SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
  OVERLAP     CDIJ1(LMDIM,LMDIM,WDES%NIONS, WDES%NCDIJ)
  INTEGER, EXTERNAL :: ONE_CENTER_NMAX_FOCKAE
  REAL(q), EXTERNAL :: ENCUTGW_IN_CHI
  REAL(q) :: ENCUTGW
 
  INTEGER IERROR

  IONODE=0
  NODE_ME=0
#ifdef MPI
  IONODE  = WDES%COMM%IONODE
  NODE_ME = WDES%COMM%NODE_ME
#endif

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

140 FORMAT (5X, //, &
       &'----------------------------------------- Iteration ', &
       &I4,'(',I4,')  ---------------------------------------'//)
  DWRITE0 'electron entered'

  CALL DIPOL_RESET()

  CALL ALLOCATE_GRID_QUANTITY(POT_EXX,  GRID_SOFT, WDES%NCDIJ)

! same mixing for up and down channel
  MIX%MIXPRE=MOD(MIX%MIXPRE,10)+10

  ENCUTGW=ENCUTGW_IN_CHI()
  IF (ENCUTGW>0) THEN
      CALL VTUTOR('W','ENCUTGW LHF', &
      &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
      CALL VTUTOR('W','ENCUTGW LHF', &
      &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
  ENDIF
!=======================================================================
! 
! set potential (do this in any case irregardless of INFO%LPOTOK
!
!=======================================================================
  CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
       INFO,P,T_INFO,E,LATT_CUR, &
       CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)

  ! now determine exact exchange potential (POT_EXX and CDIJ_EXX)
  CALL LHF(HAMILTONIAN, GRID, GRID_SOFT, LATT_CUR, SYMM, T_INFO, INFO, &
       NONLR_S,NONL_S,W,WDES,P, &
       LMDIM, INFO%LOVERL, INFO%LREAL, INFO%LCORE, POT_EXX, CRHODE, CDIJ_EXX, &
       .FALSE., IO%IU0, E%EXHF )

  ! fourier transform CVTOT back to rec. space and add POT_EXX to get total CVTOT
  CVTOT1=0
  DO ISP=1,WDES%NCDIJ
     CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
     CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, POT_EXX%RG(1,ISP), CVTOT(1,ISP))
     CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, POT_EXX%RG(1,ISP), CVTOT1(1,ISP))

     CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT(1,ISP))
     CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT1(1,ISP))
  ENDDO
  CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT1)
  ! now set SV from CVTOT
  CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)

  IF (ONE_CENTER_NMAX_FOCKAE()<=0) THEN
     IF (IO%IU0>=0) WRITE(IO%IU0,*) 'internal warning: using old version for LMAXFOCKAE'
     CALL SETDIJ_FOCKAE(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
          LMDIM,CDIJ,CQIJ,CVTOT1,IRDMAA,IRDMAX)
     CDIJ_EXX=CDIJ_EXX+CDIJ
  ENDIF

  CALL STOP_TIMING("G",IO%IU6,"POTLOK")
  DWRITE0 'potlok is ok'

  ! determine CDIJ from CVTOT
  CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
       LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

  ! add one centre correction to CDIJ_EXX
  CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
       WDES%NCDIJ, LMDIM, CDIJ_EXX(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
       E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

  ! add to CDIJ
  CDIJ=CDIJ+CDIJ_EXX

  CALL STOP_TIMING("G",IO%IU6,"SETDIJ")
  DWRITE0 'setdij is ok'

  ! DLM_EXX during initialisation might be required by mixer
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
     DESUM1=0
     RMS   =0
     ICOUEV=0
! this test statment tests the qualtity of the LHF procedure
! without orbital optimization
!     CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
!        LMDIM,CDIJ,CQIJ, 0,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)
!     CALL WRITE_EIGENVAL( WDES, W, IO%IU6)
!     STOP
     !
     ! sub space rotation before eigenvalue optimization
     !
     IF (INFO%LPDIAG) THEN

        IF (INFO%LDIAG) THEN
           IFLAG=3    ! exact diagonalization
        ELSE
           IFLAG=4    ! using Loewdin perturbation theory
        ENDIF
        IF (INFO%IALGO==4) THEN
           IFLAG=1
        ENDIF

        CALL START_TIMING("G")

        CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
             LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)

        CALL STOP_TIMING("G",IO%IU6,"EDDIAG",XMLTAG="diag")
        DWRITE0 "eddiag is ok"

     ENDIF
     CALL START_TIMING("G")

     select_algo: IF (INFO%LRMM) THEN
        !
        ! RMM-DIIS alogrithm
        !
        CALL EDDRMM(HAMILTONIAN,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
             LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV, SV, IO%IU6,IO%IU0, &
             .FALSE.)
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

        CALL EDDAV(HAMILTONIAN, P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
             LMDIM,CDIJ,CQIJ, RMS,DESUM1,ICOUEV,SV, E%EXHF, IO%IU6,IO%IU0, .FALSE., INFO%LDIAG, .FALSE.)

        E%EBANDSTR= BANDSTRUCTURE_ENERGY(WDES, W)

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
     IF (.NOT.INFO%LORTHO) THEN
        CALL START_TIMING("G")

        CALL ORTHCH(WDES,W, INFO%LOVERL, LMDIM,CQIJ)

        CALL STOP_TIMING("G",IO%IU6,"ORTHCH",XMLTAG="orth")
        DWRITE0 "ortch is ok"
     ENDIF
     !
     ! sub space rotation after eigen value optimization
     !
     IF (INFO%LCDIAG) THEN

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

     IF (IO%IU6>=0) WRITE(IO%IU6,2202) EFERMI,REAL( E%CVZERO ,KIND=q) ,E%PSCENC/INFO%NELECT+BETATO

     CALL WRITE_EIGENVAL( WDES, W, IO%IU6)
!=======================================================================
! if charge density is updated
!  ) set  INFO%LPOTOK to .F. this requires a recalculation of the local pot.
!  ) set  INFO%LMIX to .T.
!  ) call subroutine CHSP+ DEPLE to generate the new charge density
!  ) then perform mixing
!=======================================================================
     INFO%LMIX=.FALSE.
     MIX%NEIG=0

     CALL START_TIMING("G")

     INFO%LPOTOK=.FALSE.

     CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
          GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
          LATT_CUR, P, SYMM, T_INFO, &
          CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

     CALL STOP_TIMING("G",IO%IU6,"CHARGE")

     CALL START_TIMING("G")
     IF (W%OVER_BAND) THEN
        CALL REDIS_PW_OVER_BANDS(WDES, W)
        CALL STOP_TIMING("G",IO%IU6,"REDIS")
     ENDIF

     ! save old potential after FFT
     DO ISP=1,WDES%NCDIJ
        CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
        CALL RC_ADD(CVTOT(1,ISP),1.0_q,CVTOT(1,ISP),0.0_q,CVTOTL(1,ISP),GRIDC)
     ENDDO

     CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
          INFO,P,T_INFO,E,LATT_CUR, &
          CHTOT,CSTRF,CVTOT, DENCOR, SV, SOFT_TO_C,XCSIF)

     ! now determine exact exchange potential (POT_EXX, and CDIJ_EXX)
     CALL LHF(HAMILTONIAN,GRID,GRID_SOFT, LATT_CUR, SYMM, T_INFO, INFO, &
          NONLR_S,NONL_S,W,WDES,P, &
          LMDIM, INFO%LOVERL, INFO%LREAL, INFO%LCORE, POT_EXX, CRHODE, CDIJ_EXX, &
          .TRUE., IO%IU0, E%EXHF )

     ! fourier transform CVTOT back to rec. space and add POT_EXX to get total CVTOT
     CVTOT1=0
     DO ISP=1,WDES%NCDIJ
        CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
        CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, POT_EXX%RG(1,ISP), CVTOT(1,ISP))
        CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, POT_EXX%RG(1,ISP), CVTOT1(1,ISP))

        CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT(1,ISP))
        CALL APPLY_CUTOFF(GRIDC, LATT_CUR, CVTOT1(1,ISP))
     ENDDO
     CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT1)
     ! now set SV from CVTOT
     CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)
     ! project CVTOT1 (EXX-OEP potential) onto additional augmentation charges
     IF (ONE_CENTER_NMAX_FOCKAE()<=0) THEN
        CALL SETDIJ_FOCKAE(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
             LMDIM,CDIJ,CQIJ,CVTOT1,IRDMAA,IRDMAX)
        CDIJ_EXX=CDIJ_EXX+CDIJ
     ENDIF

     CALL STOP_TIMING("G",IO%IU6,"POTLOK")

     ! determine CDIJ from CVTOT
     CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
          LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

     ! add one centre correction to CDIJ_EXX
     CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
          WDES%NCDIJ, LMDIM, CDIJ_EXX(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
          E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

     CDIJ=CDIJ+CDIJ_EXX

     CALL STOP_TIMING("G",IO%IU6,"SETDIJ")

!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
! old version
     ! determine eigenvalues for current wavefunctions and potential
     ! double counting does not apply here
!     CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
!          LMDIM,CDIJ,CQIJ, 1,SV,T_INFO,P,IO%IU0,EXHF_DUMMY)
!     CALL STOP_TIMING("G",IO%IU6,"EDDIAG")

     ! improved version: calculate the energy exactly
     ! more timeconsuming, since HF needs to be calculated again
     CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
          INFO,P,T_INFO,E,LATT_CUR, &
          CHTOT,CSTRF,CVTOT1, DENCOR, SV1, SOFT_TO_C,XCSIF)

     CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
               LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)

     LHFCALC_FORCE=.TRUE.

     CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
          WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
          E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
     CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
          LMDIM,CDIJ1,CQIJ, 1,SV1,T_INFO,P,IO%IU0,E%EXHF)

     LHFCALC_FORCE=.FALSE.

     E%EBANDSTR= BANDSTRUCTURE_ENERGY(WDES, W)
     TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+Ediel_sol

     IF (M_CONSTRAINED()) TOTEN=TOTEN+E_CONSTRAINT()
     io_begin
     CALL WRITE_CONSTRAINED_M(17,.FALSE.)
     io_end
     !---- write total energy to OSZICAR file and stdout
     DESUM(N)=TOTEN-TOTENL
     ECONV=DESUM(N)
     io_begin
303  FORMAT('CG : ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)
1303 FORMAT('RMM: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)
10303 FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
          &       I6,'  ',E10.3)

     IF (INFO%LRMM) THEN
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
! mix potential
!=======================================================================
     IF (.NOT. INFO%LPOTOK) THEN
        ! CVTOT to reciprocal space (required for mixer)
        DO ISP=1,WDES%NCDIJ
           CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
        ENDDO

        ! go to total/magnetisation respresentation
        CALL RC_FLIP(CVTOT,GRIDC, WDES%NCDIJ, .FALSE.)
        CALL RC_FLIP(CVTOTL,GRIDC, WDES%NCDIJ, .FALSE.)
        CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .FALSE.)

        DLM_EXX_LAST=DLM_EXX
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
                   RMST,RMSC,RMSP,WEIGHT,.FALSE.,IERRBR)
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
        CALL RC_FLIP(CVTOT,GRIDC, WDES%NCDIJ, .TRUE.)

        CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)

        CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
             LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

        ! one centre corrections are passed through mixer and added now
        CALL RETRIVE_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, LMDIM, CDIJ_EXX, DLM_EXX)
        CALL US_FLIP(WDES, LMDIM, CDIJ_EXX, INFO%LOVERL, .TRUE.)

        CDIJ=CDIJ+CDIJ_EXX

        IF (USELDApU()) CALL LDAPLUSU_PRINTOCC(WDES,T_INFO%NIONS,T_INFO%ITYP,IO%IU6)
        CALL STOP_TIMING("G",IO%IU6,"SETDIJ")
        DWRITE0 'setdij is ok'

        INFO%LPOTOK=.TRUE.
     ENDIF
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
             INFO%LOVERL,LMDIM,CQIJ, LDIMP, LDIMP,LMDIMP,.FALSE., IO%LORBIT,PAR)
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
             &        '  -exchange      EXHF   = ',F18.8/ &
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
           WRITE(IO%IU6,*) 'pseudopotential strength for first ion, spin component:',I
           DO LP=1,P(1)%LMMAX
              WRITE(IO%IU6,'(16(F7.3,1X))') &
                   &             (CDIJ(L,LP,1,I),L=1,MIN(8,P(1)%LMMAX))
              !     &             (REAL(CDIJ(L,LP,1,I),q),L=1,MIN(16,P(1)%LMMAX))
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
END SUBROUTINE ELMIN_LHF
