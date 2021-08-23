#include "symbol.inc"
!**********************************************************************
! RCS:  $Id: electron.F,v 1.12 2003/06/27 13:22:15 kresse Exp kresse $
!
! subroutine for performing electronic minimization in VASP
!
!**********************************************************************

SUBROUTINE ELMIN( &
    HAMILTONIAN, KINEDEN, &
    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
    NSTEP, LMDIM, IRDMAX, NEDOS, &
    TOTEN, EFERMI, LDIMP, LMDIMP)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj
! solvation__
    USE solvation
! solvation__
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A - B, D - H, O - Z)
!=======================================================================
!  structures
!=======================================================================
    TYPE(ham_handle) HAMILTONIAN
    TYPE(tau_handle) KINEDEN
    TYPE(type_info) T_INFO
    TYPE(potcar) P(T_INFO%NTYP)
    TYPE(wavedes) WDES
    TYPE(nonlr_struct) NONLR_S
    TYPE(nonl_struct) NONL_S
    TYPE(wavespin) W          ! wavefunction
    TYPE(wavespin) W_F        ! wavefunction for all bands simultaneous
    TYPE(wavespin) W_G        ! same as above
    TYPE(latt) LATT_CUR
    TYPE(dynamics) DYN
    TYPE(info_struct) INFO
    TYPE(in_struct) IO
    TYPE(mixing) MIX
    TYPE(kpoints_struct) KPOINTS
    TYPE(symmetry) SYMM
    TYPE(grid_3d) GRID       ! grid for wavefunctions
    TYPE(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    TYPE(grid_3d) GRIDC      ! grid for potentials/charge
    TYPE(grid_3d) GRIDUS     ! temporary grid in us.F
    TYPE(grid_3d) GRIDB      ! Broyden grid
    TYPE(transit) B_TO_C     ! index table between GRIDB and GRIDC
    TYPE(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE(energy) E
    TYPE(latt) LATT_INI

    INTEGER NSTEP, LMDIM, IRDMAX, NEDOS
    REAL(q) :: TOTEN, EFERMI

    COMPLEX(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    COMPLEX(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    REAL(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    REAL(q) :: XCSIF(3, 3)

! local
    REAL(q) :: TOTENL = 0
    REAL(q) :: DESUM1, DESUM(INFO%NELM)
    INTEGER :: IONODE, NODE_ME
!  needed temporary for aspherical GGA calculation
    OVERLAP, ALLOCATABLE ::  CDIJ_TMP(:, :, :, :)
! local l-projected wavefunction characters (not really used here)
    REAL(q) PAR(1, 1, 1, 1, WDES%NCDIJ), DOSPAR(1, 1, 1, WDES%NCDIJ)

    REAL(q), EXTERNAL :: RHO0
    INTEGER N, ISP, ICONJU, IROT, ICEL, I, II, IRDMAA, &
        IERR, IDUM, IFLAG, ICOUEV, ICOUEV2, NN, NORDER, IERRBR, L, LP, &
        NCOUNT
    REAL(q) BTRIAL, RDUM, RMS, ORT, TOTEN2, RMS2, RMST, &
        WEIGHT, BETATO, DESUM2, RMSC, RMSP
    REAL(q) RHOAUG(WDES%NCDIJ), RHOTOT(WDES%NCDIJ)
    COMPLEX(q) CDUM
    CHARACTER(LEN=1) CHARAC
    LOGICAL LDELAY, LABORT_WITHOUT_CONV
! parameters for FAST_SPHPRO
    INTEGER :: LDIMP, LMDIMP
    REAL(q) :: TIFOR(3, T_INFO%NIONS)

    IONODE = 0
    NODE_ME = 0
#ifdef MPI
    IONODE = WDES%COMM%IONODE
    NODE_ME = WDES%COMM%NODE_ME
#endif

    NELM = INFO%NELM
    NORDER = 0; IF (KPOINTS%ISMEAR >= 0) NORDER = KPOINTS%ISMEAR
! to make timing more sensefull syncronize now
    CALLMPI(MPI_barrier(WDES%COMM%MPI_COMM, ierror))
    CALL START_TIMING("LOOP")
    CALL START_TIMING("G")

    io_begin
    IF (IO%IU0 >= 0) WRITE (IO%IU0, 142)
    WRITE (17, 142)
142 FORMAT('       N       E                     dE             ' &
           , 'd eps       ncg     rms          rms(c)')
    io_end

    DESUM1 = 0
    INFO%LMIX = .FALSE.

130 FORMAT(5X, //, &
&'----------------------------------------------------', &
&'----------------------------------------------------'//)

140 FORMAT(5X, //, &
&'--------------------------------------- Iteration ', &
&I6, '(', I4, ')  ---------------------------------------'//)
    DWRITE0 'electron entered'

    CALL DIPOL_RESET()

    CALL SPAWN_PP(T_INFO, SYMM, WDES, P, IO)
    CALL INIT_CORE_REL(WDES, CRHODE, IO%IU0, IO%IU5, IO%IU6)

!-----------------------------------------------------------------------

    IF (MIX%IMIX /= 0 .AND. .NOT. INFO%LCHCON .AND. MIX%MIXFIRST) THEN
        INFO%LPOTOK = .FALSE.
        ! copy current density to last density
        DO ISP = 1, WDES%NCDIJ
            CALL RC_ADD(CHTOT(1, ISP), 1.0_q, CHTOT(1, ISP), 0.0_q, CHTOTL(1, ISP), GRIDC)
        END DO

        IF (LDO_METAGGA() .AND. LMIX_TAU()) THEN
            DO ISP = 1, WDES%NCDIJ
                CALL RC_ADD(KINEDEN%TAU(1, ISP), 1.0_q, KINEDEN%TAU(1, ISP), 0.0_q, KINEDEN%TAUL(1, ISP), GRIDC)
            END DO
        END IF

        RHOLM_LAST = RHOLM

        INFO%LPOTOK = .FALSE.

        ! add the available GAMMA file
        IF (MOD(INFO%ICHARG, 10) == 5) THEN
            CALL ADD_GAMMA_FROM_FILE(WDES, W, KPOINTS, INFO%NELECT, INFO%NUP_DOWN, .TRUE., IO)
        END IF

        ! recalculate charge density and  kinetic energy density
        CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
                        GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
                        LATT_CUR, P, SYMM, T_INFO, &
                        CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

        CALL SET_KINEDEN(GRID, GRID_SOFT, GRIDC, SOFT_TO_C, LATT_CUR, SYMM, &
                         T_INFO%NIONS, W, WDES, KINEDEN)

        CALL STOP_TIMING("G", IO%IU6, "CHARGE")

        INFO%LMIX = .TRUE.
! update one center charge density
        IF (LCORREL()) THEN
            CALL SET_RHO_ONE_CENTRE(CRHODE, RHO_ONE_CENTRE)
        END IF

        IF (MIX%IMIX == 4) THEN
!  broyden mixing ... :
            IF (LCORREL()) THEN
                CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                           (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                           LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                           RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR, &
                           NMIX_ONE_CENTRE, RHO_ONE_CENTRE, RHO_ONE_CENTRE_LAST)
                MIX%LRESET = .FALSE.
            ELSE
                CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                           (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                           LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                           RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR)
                MIX%LRESET = .FALSE.
            END IF
        ELSE
!  simple mixing ... :
            RMST = 0
            CALL MIX_SIMPLE(GRIDC, MIX, WDES%NCDIJ, CHTOT, CHTOTL, &
                            N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
! test
            IF (LCORREL()) THEN
                RHO_ONE_CENTRE(:) = RHO_ONE_CENTRE_LAST(:) + 0.2*(RHO_ONE_CENTRE(:) - RHO_ONE_CENTRE_LAST(:))
                RHO_ONE_CENTRE_LAST(:) = RHO_ONE_CENTRE(:)
            END IF
! test
        END IF
        CALL STOP_TIMING("G", IO%IU6, "MIXING")
    END IF
!=======================================================================
    electron: DO N = 1, NELM

        CALL XML_TAG("scstep")

!======================================================================
        io_begin
        WRITE (IO%IU6, 140) NSTEP, N
        io_end
!=======================================================================
! if recalculation of total lokal potential is necessary (INFO%LPOTOK=.F.)
! call POTLOK: the subroutine calculates
! ) the hartree potential from the electronic  charge density
! ) the exchange correlation potential
! ) and the total lokal potential
!  in addition all double counting correction and forces are calculated
! &
! call SETDIJ
! calculates the Integral of the depletion charges * local potential
! and sets CDIJ
!=======================================================================

        CALL WVREAL(WDES, GRID, W) ! only for gamma some action

        IF (.NOT. INFO%LPOTOK) THEN

! core relaxation and repseudization
            IF (LCORREL()) THEN
!        CALL PW_TO_RADIAL(WDES,GRID_SOFT,CHDEN(:,1),LATT_CUR,T_INFO)
                CALL GET_AVERAGEPOT_PW(GRIDC, LATT_CUR, IRDMAX,  &
               &   T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
                CALL CORREL(RHO_ONE_CENTRE)
            END IF
#define no_update_potential
#ifdef no_update_potential
            CALL POTLOK(GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, &
                        INFO, P, T_INFO, E, LATT_CUR, &
                        CHTOT, CSTRF, CVTOT, DENCOR, SV, SOFT_TO_C, XCSIF)

            CALL POTLOK_METAGGA(KINEDEN, &
                                GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, INFO, P, T_INFO, E, LATT_CUR, &
                                CHDEN, CHTOT, DENCOR, CVTOT, SV, HAMILTONIAN%MUTOT, HAMILTONIAN%MU, SOFT_TO_C, XCSIF)

            CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C, WDES%COMM_INTER, &
                           LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)

            CALL STOP_TIMING("G", IO%IU6, "POTLOK")
            DWRITE0 'potlok is ok'

            CALL SETDIJ(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                        LMDIM, CDIJ, CQIJ, CVTOT, IRDMAA, IRDMAX)

            CALL SETDIJ_AVEC(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                             LMDIM, CDIJ, HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)

            CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ(1, 1, 1, 1), RHOLM, CRHODE(1, 1, 1, 1), &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.FALSE.)

            CALL UPDATE_CMBJ(GRIDC, T_INFO, LATT_CUR, IO%IU6)

            CALL STOP_TIMING("G", IO%IU6, "SETDIJ")
            DWRITE0 'setdij is ok'
#else
            CALL UPDATE_POTENTIAL( &
                KINEDEN, HAMILTONIAN, P, WDES, NONLR_S, NONL_S, LATT_CUR, &
                T_INFO, INFO, IO, &
                GRID, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, &
                CHTOT, DENCOR, CVTOT, CSTRF, &
                LMDIM, IRDMAX, CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, CHDEN, SV)
#endif

            IF (USELDApU()) CALL LDAPLUSU_PRINTOCC(WDES, T_INFO%NIONS, T_INFO%ITYP, IO%IU6)
!remove
            INFO%LPOTOK = .TRUE.
        END IF

        IF (LCORREL()) THEN
            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)
            DWRITE0 'orthch is ok'
        END IF

!======================== SUBROUTINE EDDSPX ============================
!
! these subroutines improve the electronic degrees of freedom
! using band by band schemes
! the Harris functional is used for the calculation
! of the total (free) energy so
! E  =  Tr[ H rho ] - d.c. (from input potential)
!
!=======================================================================
        DESUM1 = 0
        RMS = 0
        ICOUEV = 0

        LDELAY = .FALSE.
! if Davidson and RMM are selected, use Davidsons algorithm during
! delay phase
        IF (INFO%LRMM .AND. INFO%LDAVID .AND. (N <= ABS(INFO%NELMDL) .OR. N == 1)) LDELAY = .TRUE.
! if LDELAY is set, subspace rotation and orthogonalisations can be bypassed
! since they are done by the Davidson algorithm

!
! sub space rotation before eigenvalue optimization
!
        IF (INFO%LPDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3    ! exact diagonalization
            ELSE
                IFLAG = 4    ! using Loewdin perturbation theory
            END IF
            IF (INFO%IALGO == 3) THEN
                IFLAG = 0
            END IF
            IF (N < ABS(INFO%NELMDL)) IFLAG = 13

            IF (INFO%IALGO /= 2) THEN
                CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                            LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            END IF

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "eddiag is ok"
        ELSEIF (MOD(INFO%ICHARG, 10) == 5) THEN
            ! for ICHARG=5, the states have been rotated according to the supplied GAMMA
            ! diagonalize back to our current one-electron Hamiltonian
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!-----------------------------------------------------------------------
        select_algo: IF (INFO%LEXACT_DIAG) THEN
            CALL EDDIAG_EXACT(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                              LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, IO%IU6, E%EXHF, E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "EDDIAG_EXACT is ok"

        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
!
! RMM-DIIS algorithm
!
            CALL EDDRMM(HAMILTONIAN, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0, &
                        N < ABS(INFO%NELMDL) - ABS(INFO%NELMDL)/4)
            ! previous line selects  special algorithm during delay

            CALL STOP_TIMING("G", IO%IU6, "RMM-DIIS", XMLTAG="diis")
            DWRITE0 "eddrmm is ok"

        ELSE IF (INFO%LDAVID) THEN
!
! blocked Davidson algorithm,
!
            NSIM = WDES%NSIM*2
#ifdef MPI
            NSIM = ((WDES%NSIM*2 + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            CALL EDDAV(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                       LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                       LDELAY=.FALSE., LSUBROTI=INFO%LDIAG, LEMPTY=.FALSE., LHF=N >= ABS(INFO%NELMDL), &
                       EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDAV", XMLTAG="dav")
            DWRITE0 "edddav is ok"

        ELSE IF (INFO%IHARMONIC > 0) THEN
            NSIM = WDES%NSIM
#ifdef MPI
            NSIM = ((WDES%NSIM + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            IF (INFO%EREF == 0) THEN
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE.)
            ELSE
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE., EREF=INFO%EREF)
            END IF
            CALL STOP_TIMING("G", IO%IU6, "EINNER", XMLTAG="dav")
            ! since we iterate deep it is safer to recalculate projeciton operators
            ! and reorthogonalize
            CALL PROALL(GRID, LATT_CUR, NONLR_S, NONL_S, W)

            ELSE IF (INFO%IALGO == 5 .OR. INFO%IALGO == 6 .OR. &
            &         INFO%IALGO == 7 .OR. INFO%IALGO == 8 .OR. INFO%IALGO == 0) THEN select_algo

!
! CG (Teter, Alan, Payne) potential is fixed !!
!
            CALL EDSTEP(GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0)

            CALL STOP_TIMING("G", IO%IU6, "EDSTEP", XMLTAG="cg")
            DWRITE0 "edstep is ok"

        END IF select_algo
!-----------------------------------------------------------------------
! orthogonalise all bands (necessary only for residuum-minimizer
! or inner eigenvalue problems, since they iterate very deep)
!
        IF (.NOT. INFO%LORTHO .AND. .NOT. LDELAY) THEN

            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)

            CALL STOP_TIMING("G", IO%IU6, "ORTHCH", XMLTAG="orth")
            DWRITE0 "ortch is ok"
        END IF
!
! sub space rotation after eigen value optimization
!
        IF (INFO%LCDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3
            ELSE
                IFLAG = 4
            END IF

            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!=======================================================================
! recalculate the broadened density of states and fermi-weights
! recalculate depletion charge size
!=======================================================================
        CALL MRG_CEL(WDES, W)
        IF (INFO%IALGO /= 3) THEN
            E%EENTROPY = 0
            DOS = 0
            DOSI = 0
            CALL DENSTA(IO%IU0, IO%IU6, WDES, W, KPOINTS, INFO%NELECT, &
                        INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE., &
                        NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
        END IF
        DWRITE0 "densta is ok"
!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
        E%EBANDSTR = BANDSTRUCTURE_ENERGY(WDES, W)
        TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+ECORE()+ Ediel_sol
!-MM- Added to accomodate constrained moment calculations
        IF (M_CONSTRAINED()) TOTEN = TOTEN + E_CONSTRAINT()
!-MM- end of additions
!---- write total energy to OSZICAR file and stdout
        DESUM(N) = TOTEN - TOTENL
        ECONV = DESUM(N)
        io_begin
305     FORMAT('CG : ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                &       I6, '  ', E10.3)
302     FORMAT('NONE ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                &       I6, '  ', E10.3)
303     FORMAT('EIG: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                &       I6, '  ', E10.3)
304     FORMAT('DIA: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                &       I6, '  ', E10.3)
1303    FORMAT('RMM: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                    &       I6, '  ', E10.3)
10303   FORMAT('DAV: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                        &       I6, '  ', E10.3)
20303   FORMAT('JDH: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                        &       I6, '  ', E10.3)
30303   FORMAT('DAVI:', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                        &       I6, '  ', E10.3)

        IF (INFO%LEXACT_DIAG) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
            WRITE (17, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 1) THEN
            WRITE (17, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 2) THEN
            WRITE (17, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LDAVID) THEN
            WRITE (17, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 4) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 3) THEN
            WRITE (17, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 2) THEN
            WRITE (17, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE
            WRITE (17, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        END IF

        CALL STOP_TIMING("G", IO%IU6, "DOS")
        io_end

!=======================================================================
!  Test for Break condition
!=======================================================================

        INFO%LABORT = .FALSE.
        LABORT_WITHOUT_CONV = .FALSE.

!-----eigenvalues and energy must be converged
        IF (ABS(DESUM(N)) < INFO%EDIFF .AND. ABS(DESUM1) < INFO%EDIFF) INFO%LABORT = .TRUE.
!-----charge-density not constant and in last cycle no change of charge
        IF (.NOT. INFO%LMIX .AND. .NOT. INFO%LCHCON .AND. MIX%IMIX /= 0) INFO%LABORT = .FALSE.
!-----do not stop during the non-selfconsistent startup phase
        IF (N <= ABS(INFO%NELMDL)) INFO%LABORT = .FALSE.
!-----do not stop before minimum number of iterations is reached
        IF (N < ABS(INFO%NELMIN)) INFO%LABORT = .FALSE.
!-----but stop after INFO%NELM steps no matter where we are now
        IF (N >= INFO%NELM) THEN
            IF (.NOT. INFO%LABORT) LABORT_WITHOUT_CONV = .TRUE.
            INFO%LABORT = .TRUE.
        END IF

        IF ((IO%LORBIT >= 10) .AND. (MOD(N, 5) == 0) .AND. WDES%LNONCOLLINEAR) THEN
            CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
        END IF
! ======================================================================
! If the end of the electronic loop is reached
! calculate accurate initial state core level shifts
! if required
! ======================================================================
        IF (INFO%LABORT .AND. ACCURATE_CORE_LEVEL_SHIFTS()) THEN

            ALLOCATE (CDIJ_TMP(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ))
            CDIJ_TMP = CDIJ

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ_TMP(1, 1, 1, 1), RHOLM, CRHODE, &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.TRUE.)
            DEALLOCATE (CDIJ_TMP)
        END IF
!========================= subroutine CHSP  ============================
! if charge density is updated
!  ) first copy current charge to CHTOTL
!  ) set  INFO%LPOTOK to .F. this requires a recalculation of the local pot.
!  ) set INFO%LMIX to .T.
!  ) call subroutine SET_CHARGE to generate the new charge density
!  ) then performe mixing
! MIND:
! ) if delay is selected  do not update
! ) if convergence corrections to forces are calculated do not update charge
!   in last iteration
!=======================================================================
        IF (MOD(INFO%ICHARG, 10) == 5) THEN
            IF (IO%LORBIT == 14) CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
            CALL LPRJ_PROALL(W, WDES, GRID, P, CQIJ, LATT_CUR, T_INFO, INFO, IO%IU6, IO%IU0)
            CALL LPRJ_WRITE(IO%IU6, IO%IU0, W)
            CALL LPRJ_LDApU(IO, W)   ! write a LDA+U GAMMA file
            CALL LPRJ_DEALLOC_COVL
        END IF

        INFO%LMIX = .FALSE.
        MIX%NEIG = 0

        IF (.NOT. INFO%LCHCON .AND. .NOT. (INFO%LABORT .AND. INFO%LCORR) &
        &    .AND. N >= ABS(INFO%NELMDL)) THEN

            DO ISP = 1, WDES%NCDIJ
                CALL RC_ADD(CHTOT(1, ISP), 1.0_q, CHTOT(1, ISP), 0.0_q, CHTOTL(1, ISP), GRIDC)
            END DO

            IF (LDO_METAGGA() .AND. LMIX_TAU()) THEN
                DO ISP = 1, WDES%NCDIJ
                    CALL RC_ADD(KINEDEN%TAU(1, ISP), 1.0_q, KINEDEN%TAU(1, ISP), 0.0_q, KINEDEN%TAUL(1, ISP), GRIDC)
                END DO
            END IF

            RHOLM_LAST = RHOLM

            INFO%LPOTOK = .FALSE.

            IF (MOD(INFO%ICHARG, 10) == 5) THEN
                CALL REMOVE_VASP_LOCK(WDES%COMM) ! remove vasp.lock file
                CALL WAIT_VASP_LOCK(WDES%COMM)   ! wait that it is re-created by other process
                CALL ADD_GAMMA_FROM_FILE(WDES, W, KPOINTS, INFO%NELECT, INFO%NUP_DOWN, INFO%LABORT, IO)
            END IF

            CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
                            GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
                            LATT_CUR, P, SYMM, T_INFO, &
                            CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

            CALL SET_KINEDEN(GRID, GRID_SOFT, GRIDC, SOFT_TO_C, LATT_CUR, SYMM, &
                             T_INFO%NIONS, W, WDES, KINEDEN)

            CALL STOP_TIMING("G", IO%IU6, "CHARGE")

!-----------------------------------------------------------------------
            io_begin
            CALL WRITE_CONSTRAINED_M(17, .TRUE.)
            io_end
!-----------------------------------------------------------------------

            IF (MIX%IMIX /= 0) THEN
                INFO%LMIX = .TRUE.
! update one center charge density
                IF (LCORREL()) THEN
                    CALL SET_RHO_ONE_CENTRE(CRHODE, RHO_ONE_CENTRE)
                END IF

                IF (MIX%IMIX == 4) THEN
!  broyden mixing ... :
                    IF (LCORREL()) THEN
                        CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                       &   (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                       &   LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                       &   RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR, &
                       &   NMIX_ONE_CENTRE, RHO_ONE_CENTRE, RHO_ONE_CENTRE_LAST)
                        MIX%LRESET = .FALSE.
                    ELSE
                        CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                       &   (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                       &   LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                       &   RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR)
                        MIX%LRESET = .FALSE.
                    END IF
                ELSE
!  simple mixing ... :
                    RMST = 0
                    CALL MIX_SIMPLE(GRIDC, MIX, WDES%NCDIJ, CHTOT, CHTOTL, &
                                    N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
! test
                    IF (LCORREL()) THEN
                        RHO_ONE_CENTRE(:) = RHO_ONE_CENTRE_LAST(:) + &
                        &   0.2*(RHO_ONE_CENTRE(:) - RHO_ONE_CENTRE_LAST(:))
                        RHO_ONE_CENTRE_LAST(:) = RHO_ONE_CENTRE(:)
                    END IF
! test
                END IF
                CALL STOP_TIMING("G", IO%IU6, "MIXING")
                DWRITE0 "mixing is ok"

!---- ENDIF (MIX%IMIX/=0)     end of mixing
            END IF

            if (DEBUG_SC_Q) then
                do ISP = 1, WDES%NCDIJ
                    call FFT3D(CHTOT(1, ISP), GRIDC, 1)
                end do
                call M_INT(CHTOT, GRIDC, WDES)
                do ISP = 1, WDES%NCDIJ
                    call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                    call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
                end do
            end if

!-----ENDIF (.NOT.INFO%LCHCON)   end of charge update
        END IF

        IF (W%OVER_BAND) THEN
            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL STOP_TIMING("G", IO%IU6, "REDIS")
        END IF
!=======================================================================
! total time used for this step
!=======================================================================
        CALL SEPERATOR_TIMING(IO%IU6)
        CALL STOP_TIMING("LOOP", IO%IU6, XMLTAG='total')
!=======================================================================
!  important write statements
!=======================================================================

2440    FORMAT(/' eigenvalue-minimisations  :', I6, / &
                                    &       ' total energy-change (2. order) :', E14.7, '  (', E14.7, ')')
2441    FORMAT(/ &
                                    &       ' Broyden mixing:'/ &
                                    &       '  rms(total) =', E12.5, '    rms(broyden)=', E12.5, / &
                                    &       '  rms(prec ) =', E12.5/ &
                                    &       '  weight for this iteration ', F10.2)

2442    FORMAT(/' eigenvalues of (default mixing * dielectric matrix)'/ &
                '  average eigenvalue GAMMA= ', F8.4, /(10F8.4))

200     FORMAT(' number of electron ', F15.7, ' magnetization ', 3F15.7)
201     FORMAT(' augmentation part  ', F15.7, ' magnetization ', 3F15.7)

        DO I = 1, WDES%NCDIJ
            RHOTOT(I) = RHO0(GRIDC, CHTOT(1, I))
            RHOAUG(I) = RHOTOT(I) - RHO0(GRID_SOFT, CHDEN(1, I))
        END DO

        io_begin

! iteration counts
        WRITE (IO%IU6, 2440) ICOUEV, DESUM(N), DESUM1

! charge density
        WRITE (IO%IU6, 200) RHOTOT
        IF (INFO%LOVERL) THEN
            WRITE (IO%IU6, 201) RHOAUG
        END IF
! dipol moment
        IF (DIP%LCOR_DIP) CALL WRITE_DIP(IO%IU6)

! mixing
        IF (INFO%LMIX .AND. MIX%IMIX == 4) THEN
            IF (IERRBR /= 0) THEN
                IF (IO%IU0 >= 0) &
                    WRITE (IO%IU0, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
                    'mixing'' now and reset mixing at next step!'
                IF (IO%IU6 >= 0) &
                    WRITE (IO%IU6, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
                    'mixing'' now and reset mixing at next step!'
            END IF

            IF (IO%NWRITE >= 2 .OR. NSTEP == 1) THEN
                WRITE (IO%IU6, 2441) RMST, RMSC, RMSP, WEIGHT
                IF (ABS(RMST - RMSC)/RMST > 0.1_q) THEN
                    WRITE (IO%IU6, *) ' WARNING: grid for Broyden might be to small'
                END IF
            END IF
            IF (IO%IU0 >= 0) WRITE (IO%IU0, 308) RMST
            WRITE (17, 308) RMST
308         FORMAT('   ', E10.3)
            IF (MIX%NEIG > 0) THEN
                WRITE (IO%IU6, 2442) MIX%AMEAN, MIX%EIGENVAL(1:MIX%NEIG)
            END IF
        ELSE IF (INFO%LMIX) THEN
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 308) RMST
            WRITE (17, 308) RMST
        ELSE
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, *)
            WRITE (17, *)
        END IF

        if (DEBUG_SC_Q) then
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-mixing spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        end if

        io1: IF (IO%NWRITE >= 2 .OR. (NSTEP == 1)) THEN
! energy
            IF (LCORREL()) THEN
                WRITE (IO%IU6, 7241) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT + ECORE(), Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            ELSE
                WRITE (IO%IU6, 7240) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT, Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            END IF

            IF (LHFCALC) THEN
                WRITE (IO%IU6, '( "  exchange ACFDT corr.  = ",F18.8,"  see jH, gK, PRB 81, 115126")') E%EXHF_ACFDT
            END IF

7240        FORMAT(/ &
                                                               ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  atomic energy  EATOM  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                                       &        '  energy(sigma->0) =', F18.8)
7241        FORMAT(/ &
                                                               ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  core contrib.  ECORE  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                                       &        '  energy(sigma->0) =', F18.8)
72612       FORMAT(//&
                                                  &        '  METAGGA EXCHANGE AND CORRELATION (eV)'/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  LDA+GGA E(xc)  EXCG   = ', F18.6/ &
                                                  &        '  LDA+GGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  core xc             AE= ', 1F18.6/ &
                                                  &        '  metaGGA E(xc)  EXCM   = ', F18.6/ &
                                                  &        '  metaGGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  metaGGA core xc     AE= ', 1F18.6/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  METAGGA result:'/ &
                                                  &        '  free  energy   TOTEN  = ', F18.6, ' eV'// &
                                                  &        '  energy  without entropy=', F18.6, &
                                                  &        '  energy(sigma->0) =', F16.6)
            ELSE io1
            WRITE (IO%IU6, 7242) TOTEN, TOTEN - E%EENTROPY
7242        FORMAT(/'  free energy = ', E20.12, &
                                                                                    &        '  energy without entropy= ', E20.12)

        END IF io1
!     too slow on many servers nowadays
!     IF (IO%LOPEN) CALL WFORCE(IO%IU6)
!     IF (IO%LOPEN) CALL WFORCE(17)
        WRITE (IO%IU6, 130)
        io_end
!=======================================================================
!  perform some additional write statments if required
!=======================================================================
!-----Eigenvalues and weights
        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN

            ! calculate and print the core level shifts
            IF (INFO%LOVERL) THEN
                CALL CL_SHIFT_PW(GRIDC, LATT_CUR, IRDMAX, &
                                 T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
            ELSE
                IF (IO%IU0 >= 0) WRITE (IO%IU0, *) "WARNING: NC-PP core level shifts not calculated"
            END IF
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            CALL KPAR_SYNC_CELTOT(WDES, W)

            io_begin
            CALL RHOAT0(P, T_INFO, BETATO, LATT_CUR%OMEGA)

            WRITE (IO%IU6, 2202) EFERMI, REAL(E%CVZERO, KIND=q), E%PSCENC/INFO%NELECT + BETATO
2202        FORMAT(' E-fermi : ', F8.4, '     XC(G=0): ', F8.4, &
                                                                                    &         '     alpha+bet :', F8.4/)

            IF (INFO%IHARMONIC == 1) THEN
                CALL WRITE_EIGENVAL_RESIDUAL(WDES, W, IO%IU6)
            ELSE
                CALL WRITE_EIGENVAL(WDES, W, IO%IU6)
            END IF
            io_end
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            io_begin
!-----Charge-density along one line
            WRITE (IO%IU6, 130)
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'soft charge-density along one line, spin component', I
                WRITE (IO%IU6, '(10(6X,I4))') (II, II=0, 9)
                CALL WRT_RC_LINE(IO%IU6, GRID_SOFT, CHDEN(1, I))
                IF (INFO%LOVERL) THEN
                    WRITE (IO%IU6, *) 'total charge-density along one line'
                    CALL WRT_RC_LINE(IO%IU6, GRIDC, CHTOT(1, I))
                END IF
                WRITE (IO%IU6, *)
            END DO
!-----pseudopotential strength and augmentation charge
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'pseudopotential strength for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
            &             (CDIJ(L, LP, NI, I), L=1, MIN(8, P(1)%LMMAX))
!     &             (REAL(CDIJ(L,LP,NI,I),q),AIMAG(CDIJ(L,LP,1,I))*1000,L=1,MIN(16,P(1)%LMMAX))
                END DO
            END DO
            END DO

            IF (INFO%LOVERL) THEN
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'total augmentation occupancy for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
            &             (REAL(CRHODE(L, LP, NI, I), q), L=1, MIN(16, P(1)%LMMAX))
                END DO
!           DO LP=1,P(1)%LMMAX
!              WRITE(IO%IU6,'(16(F7.3,1X))') &
!     &             (AIMAG(CRHODE(L,LP,1,I))*1E6,L=1,MIN(16,P(1)%LMMAX))
!           ENDDO
            END DO
            END DO
            END IF
            io_end

        END IF
!=======================================================================
!  xml related output
!=======================================================================
        CALL XML_TAG("energy")
        IF (INFO%LABORT .OR. N == 1) THEN
            CALL XML_TAG_REAL("alphaZ", E%PSCENC)
            CALL XML_TAG_REAL("ewald", E%TEWEN)
            CALL XML_TAG_REAL("hartreedc", E%DENC)
            CALL XML_TAG_REAL("XCdc", E%XCENC)
            CALL XML_TAG_REAL("pawpsdc", E%PAWPS)
            CALL XML_TAG_REAL("pawaedc", E%PAWAE)
            CALL XML_TAG_REAL("eentropy", E%EENTROPY)
            CALL XML_TAG_REAL("bandstr", E%EBANDSTR)
            CALL XML_TAG_REAL("atom", INFO%EALLAT)
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        ELSE
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        END IF
        CALL XML_CLOSE_TAG
        CALL XML_CLOSE_TAG("scstep")
!=======================================================================
! relaxed core related output
!=======================================================================
        IF (INFO%LABORT .AND. LCORREL()) THEN
            CALL REPORT(.TRUE.)
! uncomment this line for specific postprocessing of relaxed core stuff
!        CALL RCPOSTPROC
!
        END IF
!======================== end of loop ENDLSC ===========================
! This is the end of the selfconsistent calculation loop
!=======================================================================
        IF (INFO%LABORT) THEN
            io_begin
            WRITE (IO%IU6, 131)
131         FORMAT(5X, //, &
                                                                        &  '------------------------ aborting loop because EDIFF', &
                                                                        &  ' is reached ----------------------------------------'//)
            io_end
            EXIT electron
        END IF
        INFO%LSOFT = .FALSE.
#ifndef noSTOPCAR
#ifndef F90_T3D
        CALL RDATAB(IO%LOPEN, 'STOPCAR', 99, 'LABORT', '=', '#', ';', 'L', &
        &            IDUM, RDUM, CDUM, INFO%LSOFT, CHARAC, NCOUNT, 1, IERR)
        ITMP = 0; IF (INFO%LSOFT) ITMP = 1; CALLMPI(M_sum_i(W%WDES%COMM, ITMP, 1))
        IF (ITMP > 0) INFO%LSOFT = .TRUE.
#endif
#endif
        IF (INFO%LSOFT) THEN
            io_begin
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, *) 'hard stop encountered!  aborting job ...'
            WRITE (IO%IU6, 13131)
13131       FORMAT(5X, //, &
                                                                        &  '------------------------ aborting loop because hard', &
                                                                       &  ' stop was set ---------------------------------------'//)
            io_end
            EXIT electron
        END IF
        TOTENL = TOTEN

    END DO electron

!
! calculate dipol corrections now
!
    IF (DIP%IDIPCO > 0) THEN
        IF (.NOT. DIP%LCOR_DIP) THEN
            CALL CDIPOL_CHTOT_REC(GRIDC, LATT_CUR, P, T_INFO, &
                                  CHTOT, CSTRF, CVTOT, WDES%NCDIJ, INFO%NELECT)

            CALL WRITE_DIP(IO%IU6)
            IF (IO%IU6 > 0) THEN
                WRITE (IO%IU6, *)
                WRITE (IO%IU6, *) &
                    " *************** adding dipol energy to TOTEN NOW **************** "
            END IF
            TOTEN = TOTEN + DIP%ECORR
        END IF
    END IF

! notify calling routine whether convergence has been reached
    INFO%LABORT = LABORT_WITHOUT_CONV

    DWRITE0 'electron left'

    RETURN
END SUBROUTINE ELMIN

!**********************************************************************
!
! write eigenvalues
!
!**********************************************************************

SUBROUTINE WRITE_EIGENVAL(WDES, W, IU6)
    USE wave
    USE prec
    TYPE(wavedes) WDES
    TYPE(wavespin) W          ! wavefunction
    INTEGER IU6
    ! local
    INTEGER ISP, NN, I
    INTEGER NB

    NB = WDES%NB_TOT

    IF (IU6 >= 0) THEN

        DO ISP = 1, WDES%ISPIN
            IF (WDES%ISPIN == 2) WRITE (IU6, '(/A,I1)') ' spin component ', ISP
            DO NN = 1, WDES%NKPTS
                WRITE (IU6, 2201) NN, WDES%VKPT(1, NN), WDES%VKPT(2, NN), WDES%VKPT(3, NN), &
             &      (I, REAL(W%CELTOT(I, NN, ISP), KIND=q), W%FERTOT(I, NN, ISP)*WDES%RSPIN, I=1, NB)
            END DO
        END DO

2201    FORMAT(/' k-point ', I5, ' :', 3X, 3F10.4/ &
                                                                            &         '  band No.  band energies     occupation '/ &
                                                                             &           (1X, I6, 3X, F10.4, 3X, F10.5))
    END IF

END SUBROUTINE WRITE_EIGENVAL

SUBROUTINE WRITE_EIGENVAL_RESIDUAL(WDES, W, IU6)
    USE wave
    USE prec
    TYPE(wavedes) WDES
    TYPE(wavespin) W          ! wavefunction
    INTEGER IU6
    ! local
    INTEGER ISP, NN, I
    INTEGER NB

    NB = WDES%NB_TOT

    IF (IU6 >= 0) THEN

        DO ISP = 1, WDES%ISPIN
            IF (WDES%ISPIN == 2) WRITE (IU6, '(/A,I1)') ' spin component ', ISP
            DO NN = 1, WDES%NKPTS
                WRITE (IU6, 2201) NN, WDES%VKPT(1, NN), WDES%VKPT(2, NN), WDES%VKPT(3, NN), &
                &      (I, REAL(W%CELTOT(I, NN, ISP), KIND=q), W%FERTOT(I, NN, ISP)*WDES%RSPIN, W%AUXTOT(I, NN, ISP), I=1, NB)
            END DO
        END DO

2201    FORMAT(/' k-point ', I5, ' :', 3X, 3F10.4/ &
                                                                      &         '  band No.  band energies  occupation  residual'/ &
                                                                              &           (1X, I6, 3X, F10.4, 5X, F10.5, 3X, E10.3))
    END IF

END SUBROUTINE WRITE_EIGENVAL_RESIDUAL

SUBROUTINE WRITE_EIGENVAL_NBANDS(WDES, W, IU6, NBANDS)
    USE wave
    USE prec
    TYPE(wavedes) WDES
    TYPE(wavespin) W          ! wavefunction
    INTEGER IU6
    INTEGER NBANDS
    ! local
    INTEGER ISP, NN, I
    INTEGER NB

    NB = NBANDS

    IF (IU6 >= 0) THEN

        DO ISP = 1, WDES%ISPIN
            IF (WDES%ISPIN == 2) WRITE (IU6, '(/A,I1)') ' spin component ', ISP
            DO NN = 1, WDES%NKPTS
                WRITE (IU6, 2201) NN, WDES%VKPT(1, NN), WDES%VKPT(2, NN), WDES%VKPT(3, NN), &
                     &      (I, REAL(W%CELTOT(I, NN, ISP), KIND=q), W%FERTOT(I, NN, ISP)*WDES%RSPIN, I=1, NB)
            END DO
        END DO

2201    FORMAT(/' k-point ', I5, ' :', 3X, 3F10.4/ &
                                                                            &         '  band No.  band energies     occupation '/ &
                                                                                     &           (1X, I6, 3X, F10.4, 3X, F10.5))
    END IF

END SUBROUTINE WRITE_EIGENVAL_NBANDS

SUBROUTINE ELMIN_SASC_IN( &
    HAMILTONIAN, KINEDEN, &
    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
    NSTEP, LMDIM, IRDMAX, NEDOS, &
    TOTEN, EFERMI, LDIMP, LMDIMP, CHTOTN)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj
! solvation__
    USE solvation
! solvation__
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A - B, D - H, O - Z)
!=======================================================================
!  structures
!=======================================================================
    TYPE(ham_handle) HAMILTONIAN
    TYPE(tau_handle) KINEDEN
    TYPE(type_info) T_INFO
    TYPE(potcar) P(T_INFO%NTYP)
    TYPE(wavedes) WDES
    TYPE(nonlr_struct) NONLR_S
    TYPE(nonl_struct) NONL_S
    TYPE(wavespin) W          ! wavefunction
    TYPE(wavespin) W_F        ! wavefunction for all bands simultaneous
    TYPE(wavespin) W_G        ! same as above
    TYPE(latt) LATT_CUR
    TYPE(dynamics) DYN
    TYPE(info_struct) INFO
    TYPE(in_struct) IO
    TYPE(mixing) MIX
    TYPE(kpoints_struct) KPOINTS
    TYPE(symmetry) SYMM
    TYPE(grid_3d) GRID       ! grid for wavefunctions
    TYPE(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    TYPE(grid_3d) GRIDC      ! grid for potentials/charge
    TYPE(grid_3d) GRIDUS     ! temporary grid in us.F
    TYPE(grid_3d) GRIDB      ! Broyden grid
    TYPE(transit) B_TO_C     ! index table between GRIDB and GRIDC
    TYPE(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE(energy) E
    TYPE(latt) LATT_INI

    INTEGER NSTEP, LMDIM, IRDMAX, NEDOS
    REAL(q) :: TOTEN, EFERMI

    COMPLEX(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q) CHTOTN(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    COMPLEX(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    REAL(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    REAL(q) :: XCSIF(3, 3)

! local
    REAL(q) :: TOTENL = 0
    REAL(q) :: DESUM1, DESUM(INFO%NELM)
    INTEGER :: IONODE, NODE_ME
!  needed temporary for aspherical GGA calculation
    OVERLAP, ALLOCATABLE ::  CDIJ_TMP(:, :, :, :)
! local l-projected wavefunction characters (not really used here)
    REAL(q) PAR(1, 1, 1, 1, WDES%NCDIJ), DOSPAR(1, 1, 1, WDES%NCDIJ)

    REAL(q), EXTERNAL :: RHO0
    INTEGER N, ISP, ICONJU, IROT, ICEL, I, II, IRDMAA, &
        IERR, IDUM, IFLAG, ICOUEV, ICOUEV2, NN, NORDER, IERRBR, L, LP, &
        NCOUNT
    REAL(q) BTRIAL, RDUM, RMS, ORT, TOTEN2, RMS2, RMST, &
        WEIGHT, BETATO, DESUM2, RMSC, RMSP
    REAL(q) RHOAUG(WDES%NCDIJ), RHOTOT(WDES%NCDIJ)
    COMPLEX(q) CDUM
    CHARACTER(LEN=1) CHARAC
    LOGICAL LDELAY, LABORT_WITHOUT_CONV
! parameters for FAST_SPHPRO
    INTEGER :: LDIMP, LMDIMP
    REAL(q) :: TIFOR(3, T_INFO%NIONS)

    IONODE = 0
    NODE_ME = 0
#ifdef MPI
    IONODE = WDES%COMM%IONODE
    NODE_ME = WDES%COMM%NODE_ME
#endif

    NELM = INFO%NELM
    NORDER = 0; IF (KPOINTS%ISMEAR >= 0) NORDER = KPOINTS%ISMEAR
    ! to make timing more sensefull syncronize now
    CALLMPI(MPI_barrier(WDES%COMM%MPI_COMM, ierror))
    CALL START_TIMING("LOOP")
    CALL START_TIMING("G")

    io_begin
    IF (IO%IU0 >= 0) WRITE (IO%IU0, 142)
    WRITE (17, 142)
142 FORMAT('       N       E                     dE             ' &
           , 'd eps       ncg     rms          rms(c)')
    io_end

    DESUM1 = 0
    INFO%LMIX = .FALSE.

130 FORMAT(5X, //, &
           &'----------------------------------------------------', &
           &'----------------------------------------------------'//)

140 FORMAT(5X, //, &
           &'--------------------------------------- Iteration ', &
           &I6, '(', I4, ')  ---------------------------------------'//)
    DWRITE0 'electron entered'

    CALL DIPOL_RESET()

    CALL SPAWN_PP(T_INFO, SYMM, WDES, P, IO)
    CALL INIT_CORE_REL(WDES, CRHODE, IO%IU0, IO%IU5, IO%IU6)

!-----------------------------------------------------------------------

!       IF (MIX%IMIX/=0 .AND. .NOT. INFO%LCHCON  .AND. MIX%MIXFIRST) THEN

!=======================================================================
    electron: DO N = 1, NELM

        CALL XML_TAG("scstep")

!======================================================================
        io_begin
        WRITE (IO%IU6, 140) NSTEP, N
        io_end
!=======================================================================
! if recalculation of total lokal potential is necessary (INFO%LPOTOK=.F.)
! call POTLOK: the subroutine calculates
! ) the hartree potential from the electronic  charge density
! ) the exchange correlation potential
! ) and the total lokal potential
!  in addition all double counting correction and forces are calculated
! &
! call SETDIJ
! calculates the Integral of the depletion charges * local potential
! and sets CDIJ
!=======================================================================

        CALL WVREAL(WDES, GRID, W) ! only for gamma some action

        IF (.NOT. INFO%LPOTOK) THEN

            ! core relaxation and repseudization
            IF (LCORREL()) THEN
!        CALL PW_TO_RADIAL(WDES,GRID_SOFT,CHDEN(:,1),LATT_CUR,T_INFO)
                CALL GET_AVERAGEPOT_PW(GRIDC, LATT_CUR, IRDMAX,  &
               &   T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
                CALL CORREL(RHO_ONE_CENTRE)
            END IF
#define no_update_potential
#ifdef no_update_potential
            CALL POTLOK_SASC_IN(GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, &
                                INFO, P, T_INFO, E, LATT_CUR, &
                                CHTOT, CSTRF, CVTOT, DENCOR, SV, SOFT_TO_C, XCSIF, CHTOTN)

            CALL POTLOK_METAGGA(KINEDEN, &
                                GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, INFO, P, T_INFO, E, LATT_CUR, &
                                CHDEN, CHTOT, DENCOR, CVTOT, SV, HAMILTONIAN%MUTOT, HAMILTONIAN%MU, SOFT_TO_C, XCSIF)

            CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C, WDES%COMM_INTER, &
                           LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)

            CALL STOP_TIMING("G", IO%IU6, "POTLOK")
            DWRITE0 'potlok is ok'

            CALL SETDIJ(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                        LMDIM, CDIJ, CQIJ, CVTOT, IRDMAA, IRDMAX)

            CALL SETDIJ_AVEC(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                             LMDIM, CDIJ, HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)

            CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ(1, 1, 1, 1), RHOLM, CRHODE(1, 1, 1, 1), &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.FALSE.)

            CALL UPDATE_CMBJ(GRIDC, T_INFO, LATT_CUR, IO%IU6)

            CALL STOP_TIMING("G", IO%IU6, "SETDIJ")
            DWRITE0 'setdij is ok'
#else
            CALL UPDATE_POTENTIAL( &
                KINEDEN, HAMILTONIAN, P, WDES, NONLR_S, NONL_S, LATT_CUR, &
                T_INFO, INFO, IO, &
                GRID, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, &
                CHTOT, DENCOR, CVTOT, CSTRF, &
                LMDIM, IRDMAX, CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, CHDEN, SV)
#endif

            IF (USELDApU()) CALL LDAPLUSU_PRINTOCC(WDES, T_INFO%NIONS, T_INFO%ITYP, IO%IU6)
!remove
            INFO%LPOTOK = .TRUE.
        END IF

        IF (LCORREL()) THEN
            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)
            DWRITE0 'orthch is ok'
        END IF

!======================== SUBROUTINE EDDSPX ============================
!
! these subroutines improve the electronic degrees of freedom
! using band by band schemes
! the Harris functional is used for the calculation
! of the total (free) energy so
! E  =  Tr[ H rho ] - d.c. (from input potential)
!
!=======================================================================
        DESUM1 = 0
        RMS = 0
        ICOUEV = 0

        LDELAY = .FALSE.
        ! if Davidson and RMM are selected, use Davidsons algorithm during
        ! delay phase
        IF (INFO%LRMM .AND. INFO%LDAVID .AND. (N <= ABS(INFO%NELMDL) .OR. N == 1)) LDELAY = .TRUE.
        ! if LDELAY is set, subspace rotation and orthogonalisations can be bypassed
        ! since they are done by the Davidson algorithm

!
! sub space rotation before eigenvalue optimization
!
        IF (INFO%LPDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3    ! exact diagonalization
            ELSE
                IFLAG = 4    ! using Loewdin perturbation theory
            END IF
            IF (INFO%IALGO == 3) THEN
                IFLAG = 0
            END IF
            IF (N < ABS(INFO%NELMDL)) IFLAG = 13

            IF (INFO%IALGO /= 2) THEN
                CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                            LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            END IF

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "eddiag is ok"
        ELSEIF (MOD(INFO%ICHARG, 10) == 5) THEN
            ! for ICHARG=5, the states have been rotated according to the supplied GAMMA
            ! diagonalize back to our current one-electron Hamiltonian
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!-----------------------------------------------------------------------
        select_algo: IF (INFO%LEXACT_DIAG) THEN
            CALL EDDIAG_EXACT(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                              LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, IO%IU6, E%EXHF, E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "EDDIAG_EXACT is ok"

        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
!
! RMM-DIIS algorithm
!
            CALL EDDRMM(HAMILTONIAN, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0, &
                        N < ABS(INFO%NELMDL) - ABS(INFO%NELMDL)/4)
            ! previous line selects  special algorithm during delay

            CALL STOP_TIMING("G", IO%IU6, "RMM-DIIS", XMLTAG="diis")
            DWRITE0 "eddrmm is ok"

        ELSE IF (INFO%LDAVID) THEN
!
! blocked Davidson algorithm,
!
            NSIM = WDES%NSIM*2
#ifdef MPI
            NSIM = ((WDES%NSIM*2 + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            CALL EDDAV(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                       LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                       LDELAY=.FALSE., LSUBROTI=INFO%LDIAG, LEMPTY=.FALSE., LHF=N >= ABS(INFO%NELMDL), &
                       EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDAV", XMLTAG="dav")
            DWRITE0 "edddav is ok"

        ELSE IF (INFO%IHARMONIC > 0) THEN
            NSIM = WDES%NSIM
#ifdef MPI
            NSIM = ((WDES%NSIM + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            IF (INFO%EREF == 0) THEN
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE.)
            ELSE
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE., EREF=INFO%EREF)
            END IF
            CALL STOP_TIMING("G", IO%IU6, "EINNER", XMLTAG="dav")
            ! since we iterate deep it is safer to recalculate projeciton operators
            ! and reorthogonalize
            CALL PROALL(GRID, LATT_CUR, NONLR_S, NONL_S, W)

            ELSE IF (INFO%IALGO == 5 .OR. INFO%IALGO == 6 .OR. &
           &         INFO%IALGO == 7 .OR. INFO%IALGO == 8 .OR. INFO%IALGO == 0) THEN select_algo

!
! CG (Teter, Alan, Payne) potential is fixed !!
!

            CALL EDSTEP(GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0)

            CALL STOP_TIMING("G", IO%IU6, "EDSTEP", XMLTAG="cg")
            DWRITE0 "edstep is ok"

        END IF select_algo
!-----------------------------------------------------------------------
! orthogonalise all bands (necessary only for residuum-minimizer
! or inner eigenvalue problems, since they iterate very deep)
!
        IF (.NOT. INFO%LORTHO .AND. .NOT. LDELAY) THEN

            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)

            CALL STOP_TIMING("G", IO%IU6, "ORTHCH", XMLTAG="orth")
            DWRITE0 "ortch is ok"
        END IF
!
! sub space rotation after eigen value optimization
!
        IF (INFO%LCDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3
            ELSE
                IFLAG = 4
            END IF

            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!=======================================================================
! recalculate the broadened density of states and fermi-weights
! recalculate depletion charge size
!=======================================================================
        CALL MRG_CEL(WDES, W)
        IF (INFO%IALGO /= 3) THEN
            E%EENTROPY = 0
            DOS = 0
            DOSI = 0
            CALL DENSTA(IO%IU0, IO%IU6, WDES, W, KPOINTS, INFO%NELECT, &
                        INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE., &
                        NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
        END IF
        DWRITE0 "densta is ok"
!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
        E%EBANDSTR = BANDSTRUCTURE_ENERGY(WDES, W)
        TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+ECORE()+ Ediel_sol
!-MM- Added to accomodate constrained moment calculations
        IF (M_CONSTRAINED()) TOTEN = TOTEN + E_CONSTRAINT()
!-MM- end of additions
!---- write total energy to OSZICAR file and stdout
        DESUM(N) = TOTEN - TOTENL
        ECONV = DESUM(N)
        io_begin
305     FORMAT('CG : ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                   &       I6, '  ', E10.3)
302     FORMAT('NONE ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                   &       I6, '  ', E10.3)
303     FORMAT('EIG: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                   &       I6, '  ', E10.3)
304     FORMAT('DIA: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                   &       I6, '  ', E10.3)
1303    FORMAT('RMM: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                             &       I6, '  ', E10.3)
10303   FORMAT('DAV: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                       &       I6, '  ', E10.3)
20303   FORMAT('JDH: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                       &       I6, '  ', E10.3)
30303   FORMAT('DAVI:', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                       &       I6, '  ', E10.3)

        IF (INFO%LEXACT_DIAG) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
            WRITE (17, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 1) THEN
            WRITE (17, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 2) THEN
            WRITE (17, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LDAVID) THEN
            WRITE (17, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 4) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 3) THEN
            WRITE (17, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 2) THEN
            WRITE (17, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE
            WRITE (17, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        END IF

        CALL STOP_TIMING("G", IO%IU6, "DOS")
        io_end
!=======================================================================
!  Test for Break condition
!=======================================================================

        INFO%LABORT = .FALSE.
        LABORT_WITHOUT_CONV = .FALSE.

!-----eigenvalues and energy must be converged
        !   IF(ABS(DESUM(N))<INFO%EDIFF.AND.ABS(DESUM1)<INFO%EDIFF) INFO%LABORT=.TRUE.
!-----charge-density not constant and in last cycle no change of charge
        !   IF (.NOT. INFO%LMIX .AND. .NOT. INFO%LCHCON .AND. MIX%IMIX/=0) INFO%LABORT=.FALSE.
!-----do not stop during the non-selfconsistent startup phase
        IF (N <= ABS(INFO%NELMDL)) INFO%LABORT = .FALSE.
!-----do not stop before minimum number of iterations is reached
        IF (N < ABS(INFO%NELMIN)) INFO%LABORT = .FALSE.
!-----but stop after INFO%NELM steps no matter where we are now
        IF (N >= INFO%NELM) THEN
            IF (.NOT. INFO%LABORT) LABORT_WITHOUT_CONV = .TRUE.
            INFO%LABORT = .TRUE.
        END IF

        IF ((IO%LORBIT >= 10) .AND. (MOD(N, 5) == 0) .AND. WDES%LNONCOLLINEAR) THEN
            CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
        END IF
! ======================================================================
! If the end of the electronic loop is reached
! calculate accurate initial state core level shifts
! if required
! ======================================================================
        IF (INFO%LABORT .AND. ACCURATE_CORE_LEVEL_SHIFTS()) THEN

            ALLOCATE (CDIJ_TMP(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ))
            CDIJ_TMP = CDIJ

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ_TMP(1, 1, 1, 1), RHOLM, CRHODE, &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.TRUE.)
            DEALLOCATE (CDIJ_TMP)
        END IF
!========================= subroutine CHSP  ============================
! if charge density is updated
!  ) first copy current charge to CHTOTL
!  ) set  INFO%LPOTOK to .F. this requires a recalculation of the local pot.
!  ) set INFO%LMIX to .T.
!  ) call subroutine SET_CHARGE to generate the new charge density
!  ) then performe mixing
! MIND:
! ) if delay is selected  do not update
! ) if convergence corrections to forces are calculated do not update charge
!   in last iteration
!=======================================================================
        IF (MOD(INFO%ICHARG, 10) == 5) THEN
            IF (IO%LORBIT == 14) CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
            CALL LPRJ_PROALL(W, WDES, GRID, P, CQIJ, LATT_CUR, T_INFO, INFO, IO%IU6, IO%IU0)
            CALL LPRJ_WRITE(IO%IU6, IO%IU0, W)
            CALL LPRJ_LDApU(IO, W)   ! write a LDA+U GAMMA file
            CALL LPRJ_DEALLOC_COVL
        END IF

        INFO%LMIX = .FALSE.
        MIX%NEIG = 0

        !   IF (.NOT. INFO%LCHCON .AND. .NOT. (INFO%LABORT .AND. INFO%LCORR ) &
        IF (.NOT. INFO%LCHCON .AND. N >= ABS(INFO%NELMDL)) THEN

            DO ISP = 1, WDES%NCDIJ
                CALL RC_ADD(CHTOT(1, ISP), 1.0_q, CHTOT(1, ISP), 0.0_q, CHTOTL(1, ISP), GRIDC)
            END DO

            IF (LDO_METAGGA() .AND. LMIX_TAU()) THEN
                DO ISP = 1, WDES%NCDIJ
                    CALL RC_ADD(KINEDEN%TAU(1, ISP), 1.0_q, KINEDEN%TAU(1, ISP), 0.0_q, KINEDEN%TAUL(1, ISP), GRIDC)
                END DO
            END IF

            RHOLM_LAST = RHOLM

            INFO%LPOTOK = .FALSE.

            IF (MOD(INFO%ICHARG, 10) == 5) THEN
                CALL REMOVE_VASP_LOCK(WDES%COMM) ! remove vasp.lock file
                CALL WAIT_VASP_LOCK(WDES%COMM)   ! wait that it is re-created by other process
                CALL ADD_GAMMA_FROM_FILE(WDES, W, KPOINTS, INFO%NELECT, INFO%NUP_DOWN, INFO%LABORT, IO)
            END IF

            CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
                            GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
                            LATT_CUR, P, SYMM, T_INFO, &
                            CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

            CALL SET_KINEDEN(GRID, GRID_SOFT, GRIDC, SOFT_TO_C, LATT_CUR, SYMM, &
                             T_INFO%NIONS, W, WDES, KINEDEN)

            CALL STOP_TIMING("G", IO%IU6, "CHARGE")

!-----------------------------------------------------------------------
            DO ISP = 1, WDES%NCDIJ
                CALL FFT3D(CHTOT(1, ISP), GRIDC, 1)
            END DO

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            call WRITE_CONSTRAINED_M(17, .FALSE.)
            io_end

            DO ISP = 1, WDES%NCDIJ
                CALL FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                CALL SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            END DO
!-----------------------------------------------------------------------

!       IF (MIX%IMIX/=0) THEN
!-----------------------------------------------------------------------
!-----ENDIF (.NOT.INFO%LCHCON)   end of charge update
        END IF

        IF (W%OVER_BAND) THEN
            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL STOP_TIMING("G", IO%IU6, "REDIS")
        END IF
!=======================================================================
! total time used for this step
!=======================================================================
        CALL SEPERATOR_TIMING(IO%IU6)
        CALL STOP_TIMING("LOOP", IO%IU6, XMLTAG='total')

!=======================================================================
!  important write statements
!=======================================================================

2440    FORMAT(/' eigenvalue-minimisations  :', I6, / &
                                                              &       ' total energy-change (2. order) :', E14.7, '  (', E14.7, ')')
2441    FORMAT(/ &
                                                                             &       ' Broyden mixing:'/ &
                                                                    &       '  rms(total) =', E12.5, '    rms(broyden)=', E12.5, / &
                                                                             &       '  rms(prec ) =', E12.5/ &
                                                                             &       '  weight for this iteration ', F10.2)

2442    FORMAT(/' eigenvalues of (default mixing * dielectric matrix)'/ &
                '  average eigenvalue GAMMA= ', F8.4, /(10F8.4))

200     FORMAT(' number of electron ', F15.7, ' magnetization ', 3F15.7)
201     FORMAT(' augmentation part  ', F15.7, ' magnetization ', 3F15.7)

        DO I = 1, WDES%NCDIJ
            RHOTOT(I) = RHO0(GRIDC, CHTOT(1, I))
            RHOAUG(I) = RHOTOT(I) - RHO0(GRID_SOFT, CHDEN(1, I))
        END DO

        io_begin

        ! iteration counts
        WRITE (IO%IU6, 2440) ICOUEV, DESUM(N), DESUM1

        ! charge density
        WRITE (IO%IU6, 200) RHOTOT
        IF (INFO%LOVERL) THEN
            WRITE (IO%IU6, 201) RHOAUG
        END IF
        ! dipol moment
        IF (DIP%LCOR_DIP) CALL WRITE_DIP(IO%IU6)

        ! mixing
!         IF (INFO%LMIX .AND. MIX%IMIX == 4) THEN
!             IF (IERRBR /= 0) THEN
!                 IF (IO%IU0 >= 0) &
!                     WRITE (IO%IU0, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
!                     'mixing'' now and reset mixing at next step!'
!                 IF (IO%IU6 >= 0) &
!                     WRITE (IO%IU6, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
!                     'mixing'' now and reset mixing at next step!'
!             END IF

!             IF (IO%NWRITE >= 2 .OR. NSTEP == 1) THEN
!                 WRITE (IO%IU6, 2441) RMST, RMSC, RMSP, WEIGHT
!                 IF (ABS(RMST - RMSC)/RMST > 0.1_q) THEN
!                     WRITE (IO%IU6, *) ' WARNING: grid for Broyden might be to small'
!                 END IF
!             END IF
!             IF (IO%IU0 >= 0) WRITE (IO%IU0, 308) RMST
!             WRITE (17, 308) RMST
! 308         FORMAT('   ', E10.3)
!             IF (MIX%NEIG > 0) THEN
!                 WRITE (IO%IU6, 2442) MIX%AMEAN, MIX%EIGENVAL(1:MIX%NEIG)
!             END IF
!         ELSE IF (INFO%LMIX) THEN
!             IF (IO%IU0 >= 0) &
!                 WRITE (IO%IU0, 308) RMST
!             WRITE (17, 308) RMST
!         ELSE
!             IF (IO%IU0 >= 0) &
!                 WRITE (IO%IU0, *)
!             WRITE (17, *)
!         END IF
        io1: IF (IO%NWRITE >= 2 .OR. (NSTEP == 1)) THEN
            ! energy
            IF (LCORREL()) THEN
                WRITE (IO%IU6, 7241) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT + ECORE(), Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            ELSE
                WRITE (IO%IU6, 7240) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT, Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            END IF

            IF (LHFCALC) THEN
                WRITE (IO%IU6, '( "  exchange ACFDT corr.  = ",F18.8,"  see jH, gK, PRB 81, 115126")') E%EXHF_ACFDT
            END IF

7240        FORMAT(/ &
                                                       &        ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  atomic energy  EATOM  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                                       &        '  energy(sigma->0) =', F18.8)
7241        FORMAT(/ &
                                                       &        ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  core contrib.  ECORE  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                   &        '  energy(sigma->0) =', F18.8)
72612       FORMAT(//&
                                                  &        '  METAGGA EXCHANGE AND CORRELATION (eV)'/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  LDA+GGA E(xc)  EXCG   = ', F18.6/ &
                                                  &        '  LDA+GGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  core xc             AE= ', 1F18.6/ &
                                                  &        '  metaGGA E(xc)  EXCM   = ', F18.6/ &
                                                  &        '  metaGGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  metaGGA core xc     AE= ', 1F18.6/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  METAGGA result:'/ &
                                                  &        '  free  energy   TOTEN  = ', F18.6, ' eV'// &
                                                  &        '  energy  without entropy=', F18.6, &
                              &        '  energy(sigma->0) =', F16.6)
            ELSE io1
            WRITE (IO%IU6, 7242) TOTEN, TOTEN - E%EENTROPY
7242        FORMAT(/'  free energy = ', E20.12, &
                                                                                   &        '  energy without entropy= ', E20.12)

        END IF io1
!     too slow on many servers nowadays
!     IF (IO%LOPEN) CALL WFORCE(IO%IU6)
!     IF (IO%LOPEN) CALL WFORCE(17)
        WRITE (IO%IU6, 130)
        io_end
!=======================================================================
!  perform some additional write statments if required
!=======================================================================
!-----Eigenvalues and weights
        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
       &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN

            ! calculate and print the core level shifts
            IF (INFO%LOVERL) THEN
                CALL CL_SHIFT_PW(GRIDC, LATT_CUR, IRDMAX, &
                                 T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
            ELSE
                IF (IO%IU0 >= 0) WRITE (IO%IU0, *) "WARNING: NC-PP core level shifts not calculated"
            END IF
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
       &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            CALL KPAR_SYNC_CELTOT(WDES, W)

            io_begin
            CALL RHOAT0(P, T_INFO, BETATO, LATT_CUR%OMEGA)

            WRITE (IO%IU6, 2202) EFERMI, REAL(E%CVZERO, KIND=q), E%PSCENC/INFO%NELECT + BETATO
2202        FORMAT(' E-fermi : ', F8.4, '     XC(G=0): ', F8.4, &
                                                                                                &         '     alpha+bet :', F8.4/)

            IF (INFO%IHARMONIC == 1) THEN
                CALL WRITE_EIGENVAL_RESIDUAL(WDES, W, IO%IU6)
            ELSE
                CALL WRITE_EIGENVAL(WDES, W, IO%IU6)
            END IF
            io_end
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
       &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            io_begin
!-----Charge-density along one line
            WRITE (IO%IU6, 130)
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'soft charge-density along one line, spin component', I
                WRITE (IO%IU6, '(10(6X,I4))') (II, II=0, 9)
                CALL WRT_RC_LINE(IO%IU6, GRID_SOFT, CHDEN(1, I))
                IF (INFO%LOVERL) THEN
                    WRITE (IO%IU6, *) 'total charge-density along one line'
                    CALL WRT_RC_LINE(IO%IU6, GRIDC, CHTOT(1, I))
                END IF
                WRITE (IO%IU6, *)
            END DO
!-----pseudopotential strength and augmentation charge
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'pseudopotential strength for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
           &             (CDIJ(L, LP, NI, I), L=1, MIN(8, P(1)%LMMAX))
!     &             (REAL(CDIJ(L,LP,NI,I),q),AIMAG(CDIJ(L,LP,1,I))*1000,L=1,MIN(16,P(1)%LMMAX))
                END DO
            END DO
            END DO

            IF (INFO%LOVERL) THEN
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'total augmentation occupancy for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
           &             (REAL(CRHODE(L, LP, NI, I), q), L=1, MIN(16, P(1)%LMMAX))
                END DO
!           DO LP=1,P(1)%LMMAX
!              WRITE(IO%IU6,'(16(F7.3,1X))') &
!     &             (AIMAG(CRHODE(L,LP,1,I))*1E6,L=1,MIN(16,P(1)%LMMAX))
!           ENDDO
            END DO
            END DO
            END IF
            io_end

        END IF
!=======================================================================
!  xml related output
!=======================================================================
        CALL XML_TAG("energy")
        IF (INFO%LABORT .OR. N == 1) THEN
            CALL XML_TAG_REAL("alphaZ", E%PSCENC)
            CALL XML_TAG_REAL("ewald", E%TEWEN)
            CALL XML_TAG_REAL("hartreedc", E%DENC)
            CALL XML_TAG_REAL("XCdc", E%XCENC)
            CALL XML_TAG_REAL("pawpsdc", E%PAWPS)
            CALL XML_TAG_REAL("pawaedc", E%PAWAE)
            CALL XML_TAG_REAL("eentropy", E%EENTROPY)
            CALL XML_TAG_REAL("bandstr", E%EBANDSTR)
            CALL XML_TAG_REAL("atom", INFO%EALLAT)
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        ELSE
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        END IF
        CALL XML_CLOSE_TAG
        CALL XML_CLOSE_TAG("scstep")
!=======================================================================
! relaxed core related output
!=======================================================================
        IF (INFO%LABORT .AND. LCORREL()) THEN
            CALL REPORT(.TRUE.)
! uncomment this line for specific postprocessing of relaxed core stuff
!        CALL RCPOSTPROC
!
        END IF
!======================== end of loop ENDLSC ===========================
! This is the end of the selfconsistent calculation loop
!=======================================================================
        IF (INFO%LABORT) THEN
            io_begin
            WRITE (IO%IU6, 131)
131         FORMAT(5X, //, &
                                                                        &  '------------------------ aborting loop because EDIFF', &
                                                                        &  ' is reached ----------------------------------------'//)
            io_end
            EXIT electron
        END IF
        INFO%LSOFT = .FALSE.
#ifndef noSTOPCAR
#ifndef F90_T3D
        CALL RDATAB(IO%LOPEN, 'STOPCAR', 99, 'LABORT', '=', '#', ';', 'L', &
       &            IDUM, RDUM, CDUM, INFO%LSOFT, CHARAC, NCOUNT, 1, IERR)
        ITMP = 0; IF (INFO%LSOFT) ITMP = 1; CALLMPI(M_sum_i(W%WDES%COMM, ITMP, 1))
        IF (ITMP > 0) INFO%LSOFT = .TRUE.
#endif
#endif
        IF (INFO%LSOFT) THEN
            io_begin
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, *) 'hard stop encountered!  aborting job ...'
            WRITE (IO%IU6, 13131)
13131       FORMAT(5X, //, &
                                                                         &  '------------------------ aborting loop because hard', &
                                                                       &  ' stop was set ---------------------------------------'//)
            io_end
            EXIT electron
        END IF
        TOTENL = TOTEN

    END DO electron

! calculate dipol corrections now
!
    IF (DIP%IDIPCO > 0) THEN
        IF (.NOT. DIP%LCOR_DIP) THEN
            CALL CDIPOL_CHTOT_REC(GRIDC, LATT_CUR, P, T_INFO, &
                                  CHTOT, CSTRF, CVTOT, WDES%NCDIJ, INFO%NELECT)

            CALL WRITE_DIP(IO%IU6)
            IF (IO%IU6 > 0) THEN
                WRITE (IO%IU6, *)
                WRITE (IO%IU6, *) &
                    " *************** adding dipol energy to TOTEN NOW **************** "
            END IF
            TOTEN = TOTEN + DIP%ECORR
        END IF
    END IF

    ! notify calling routine whether convergence has been reached
    INFO%LABORT = LABORT_WITHOUT_CONV

    DWRITE0 'electron left'

    RETURN
END SUBROUTINE ELMIN_SASC_IN

SUBROUTINE ELMIN_SASC( &
    HAMILTONIAN, KINEDEN, &
    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
    NSTEP, LMDIM, IRDMAX, NEDOS, &
    TOTEN, EFERMI, LDIMP, LMDIMP)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj
! solvation__
    USE solvation
! solvation__
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A - B, D - H, O - Z)
!=======================================================================
!  structures
!=======================================================================
    TYPE(ham_handle) HAMILTONIAN
    TYPE(tau_handle) KINEDEN
    TYPE(type_info) T_INFO
    TYPE(potcar) P(T_INFO%NTYP)
    TYPE(wavedes) WDES
    TYPE(nonlr_struct) NONLR_S
    TYPE(nonl_struct) NONL_S
    TYPE(wavespin) W          ! wavefunction
    TYPE(wavespin) W_F        ! wavefunction for all bands simultaneous
    TYPE(wavespin) W_G        ! same as above
    TYPE(latt) LATT_CUR
    TYPE(dynamics) DYN
    TYPE(info_struct) INFO
    TYPE(in_struct) IO
    TYPE(mixing) MIX
    TYPE(kpoints_struct) KPOINTS
    TYPE(symmetry) SYMM
    TYPE(grid_3d) GRID       ! grid for wavefunctions
    TYPE(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    TYPE(grid_3d) GRIDC      ! grid for potentials/charge
    TYPE(grid_3d) GRIDUS     ! temporary grid in us.F
    TYPE(grid_3d) GRIDB      ! Broyden grid
    TYPE(transit) B_TO_C     ! index table between GRIDB and GRIDC
    TYPE(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE(energy) E
    TYPE(latt) LATT_INI

    INTEGER NSTEP, LMDIM, IRDMAX, NEDOS
    REAL(q) :: TOTEN, EFERMI
    REAL(q) :: TOTEN_RESERVE

    COMPLEX(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    COMPLEX(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    REAL(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    REAL(q) :: XCSIF(3, 3)

! local
    REAL(q) :: TOTENL = 0
    REAL(q) :: DESUM1, DESUM(INFO%NELM)
    INTEGER :: IONODE, NODE_ME
!  needed temporary for aspherical GGA calculation
    OVERLAP, ALLOCATABLE ::  CDIJ_TMP(:, :, :, :)
! local l-projected wavefunction characters (not really used here)
    REAL(q) PAR(1, 1, 1, 1, WDES%NCDIJ), DOSPAR(1, 1, 1, WDES%NCDIJ)

    REAL(q), EXTERNAL :: RHO0
    INTEGER N, ISP, ICONJU, IROT, ICEL, I, II, IRDMAA, &
        IERR, IDUM, IFLAG, ICOUEV, ICOUEV2, NN, NORDER, IERRBR, L, LP, &
        NCOUNT
    REAL(q) BTRIAL, RDUM, RMS, ORT, TOTEN2, RMS2, RMST, &
        WEIGHT, BETATO, DESUM2, RMSC, RMSP
    REAL(q) RHOAUG(WDES%NCDIJ), RHOTOT(WDES%NCDIJ)
    COMPLEX(q) CDUM
    CHARACTER(LEN=1) CHARAC
    LOGICAL LDELAY, LABORT_WITHOUT_CONV
! parameters for FAST_SPHPRO
    INTEGER :: LDIMP, LMDIMP
    REAL(q) :: TIFOR(3, T_INFO%NIONS)

    IONODE = 0
    NODE_ME = 0
#ifdef MPI
    IONODE = WDES%COMM%IONODE
    NODE_ME = WDES%COMM%NODE_ME
#endif

    NELM = INFO%NELM
    NORDER = 0; IF (KPOINTS%ISMEAR >= 0) NORDER = KPOINTS%ISMEAR
! to make timing more sensefull syncronize now
    CALLMPI(MPI_barrier(WDES%COMM%MPI_COMM, ierror))
    CALL START_TIMING("LOOP")
    CALL START_TIMING("G")

    L_CONSTR = L_CONSTR_L
    SCTYPE_CURRENT = 1

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "-------------------------------------------------------------------------------"
    if (IO%IU0 >= 0) write (IO%IU0, *) "SASC(L) (Self-Adaptive Spin Constraint (Linear Lagragian))"
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, es9.3)') "Initial trial step size (alpha_trial) = ", INI_SC_ALPHA
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "Maximum number of steps in SC iteration = ", CONSTR_NUM_STEP
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "Number of INTERMEDIATE normal electronic steps = ", NELM_SC_INTER
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, es9.3)') "Convergence criterion of SC iteration (epsilon) = ", CONSTR_EPSILON
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, es9.3)') "Restriction of SC step size = ", CONSTR_RESTRICT
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "SASC electronic step begins from: ", NELM_SC_INITIAL
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0, a, i0)') "Constrained atoms: ", count(CONSTRL == 1)/3, "/", T_INFO%NIONS
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "Inner optimization algorithm: ", ALGO_SC
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "Inner diagonalization algorithm: ", ALGO_SC_DIAG
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, l1)') "Restriction of trial-step update: ", TRIAL_UPDATE_RESTRICT
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, l1)') "Debug mode: ", DEBUG_SC
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "--------------------------------------"
    if (IO%IU0 >= 0) write (IO%IU0, *) "The DEFINITION of atomic spins which are constrained (MW):"
    if (IO%IU0 >= 0) write (IO%IU0, *) "\vec{M}_{I}="
    if (IO%IU0 >= 0) write (IO%IU0, *) "\int_{\Omega_{I}} \vec{m}(\mathbf{r}) F_{I}(|\mathbf{r}|) d\mathbf{r}"
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "-------------------------------------------------------------------------------"
    io_end

    io_begin
    IF (IO%IU0 >= 0) WRITE (IO%IU0, 142)
    WRITE (17, 142)
142 FORMAT('       N       E                     dE             ' &
           , 'd eps       ncg     rms          rms(c)')
    io_end

    DESUM1 = 0
    INFO%LMIX = .FALSE.

130 FORMAT(5X, //, &
      &'----------------------------------------------------', &
      &'----------------------------------------------------'//)

140 FORMAT(5X, //, &
      &'--------------------------------------- Iteration ', &
      &I6, '(', I4, ')  ---------------------------------------'//)
    DWRITE0 'electron entered'

    CALL DIPOL_RESET()

    CALL SPAWN_PP(T_INFO, SYMM, WDES, P, IO)
    CALL INIT_CORE_REL(WDES, CRHODE, IO%IU0, IO%IU5, IO%IU6)

!-----------------------------------------------------------------------

    IF (MIX%IMIX /= 0 .AND. .NOT. INFO%LCHCON .AND. MIX%MIXFIRST) THEN
        INFO%LPOTOK = .FALSE.
        ! copy current density to last density
        DO ISP = 1, WDES%NCDIJ
            CALL RC_ADD(CHTOT(1, ISP), 1.0_q, CHTOT(1, ISP), 0.0_q, CHTOTL(1, ISP), GRIDC)
        END DO

        IF (LDO_METAGGA() .AND. LMIX_TAU()) THEN
            DO ISP = 1, WDES%NCDIJ
                CALL RC_ADD(KINEDEN%TAU(1, ISP), 1.0_q, KINEDEN%TAU(1, ISP), 0.0_q, KINEDEN%TAUL(1, ISP), GRIDC)
            END DO
        END IF

        RHOLM_LAST = RHOLM

        INFO%LPOTOK = .FALSE.

        ! add the available GAMMA file
        IF (MOD(INFO%ICHARG, 10) == 5) THEN
            CALL ADD_GAMMA_FROM_FILE(WDES, W, KPOINTS, INFO%NELECT, INFO%NUP_DOWN, .TRUE., IO)
        END IF

        ! recalculate charge density and  kinetic energy density
        CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
                        GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
                        LATT_CUR, P, SYMM, T_INFO, &
                        CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

        CALL SET_KINEDEN(GRID, GRID_SOFT, GRIDC, SOFT_TO_C, LATT_CUR, SYMM, &
                         T_INFO%NIONS, W, WDES, KINEDEN)

        CALL STOP_TIMING("G", IO%IU6, "CHARGE")

        INFO%LMIX = .TRUE.
! update one center charge density
        IF (LCORREL()) THEN
            CALL SET_RHO_ONE_CENTRE(CRHODE, RHO_ONE_CENTRE)
        END IF

        IF (MIX%IMIX == 4) THEN
!  broyden mixing ... :
            IF (LCORREL()) THEN
                CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                           (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                           LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                           RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR, &
                           NMIX_ONE_CENTRE, RHO_ONE_CENTRE, RHO_ONE_CENTRE_LAST)
                MIX%LRESET = .FALSE.
            ELSE
                CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                           (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                           LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                           RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR)
                MIX%LRESET = .FALSE.
            END IF
        ELSE
!  simple mixing ... :
            RMST = 0
            CALL MIX_SIMPLE(GRIDC, MIX, WDES%NCDIJ, CHTOT, CHTOTL, &
                            N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
! test
            IF (LCORREL()) THEN
                RHO_ONE_CENTRE(:) = RHO_ONE_CENTRE_LAST(:) + 0.2*(RHO_ONE_CENTRE(:) - RHO_ONE_CENTRE_LAST(:))
                RHO_ONE_CENTRE_LAST(:) = RHO_ONE_CENTRE(:)
            END IF
! test
        END IF
        CALL STOP_TIMING("G", IO%IU6, "MIXING")
    END IF
!=======================================================================
    electron: DO N = 1, NELM

        CALL XML_TAG("scstep")

        if (DEBUG_SC) then
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) beginning-of-SCF-step spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        end if
!======================================================================
        io_begin
        WRITE (IO%IU6, 140) NSTEP, N
        io_end
!=======================================================================
! if recalculation of total lokal potential is necessary (INFO%LPOTOK=.F.)
! call POTLOK: the subroutine calculates
! ) the hartree potential from the electronic  charge density
! ) the exchange correlation potential
! ) and the total lokal potential
!  in addition all double counting correction and forces are calculated
! &
! call SETDIJ
! calculates the Integral of the depletion charges * local potential
! and sets CDIJ
!=======================================================================

        CALL WVREAL(WDES, GRID, W) ! only for gamma some action

        IF (.NOT. INFO%LPOTOK) THEN

! core relaxation and repseudization
            IF (LCORREL()) THEN
!        CALL PW_TO_RADIAL(WDES,GRID_SOFT,CHDEN(:,1),LATT_CUR,T_INFO)
                CALL GET_AVERAGEPOT_PW(GRIDC, LATT_CUR, IRDMAX,  &
                &   T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
                CALL CORREL(RHO_ONE_CENTRE)
            END IF
#define no_update_potential
#ifdef no_update_potential

            where (CONSTRL == 0) L_CONSTR = 0.0 ! spin constraint masking
            CALL POTLOK(GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, &
                        INFO, P, T_INFO, E, LATT_CUR, &
                        CHTOT, CSTRF, CVTOT, DENCOR, SV, SOFT_TO_C, XCSIF)

            CALL POTLOK_METAGGA(KINEDEN, &
                                GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, WDES, INFO, P, T_INFO, E, LATT_CUR, &
                                CHDEN, CHTOT, DENCOR, CVTOT, SV, HAMILTONIAN%MUTOT, HAMILTONIAN%MU, SOFT_TO_C, XCSIF)

            CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C, WDES%COMM_INTER, &
                           LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)

            CALL STOP_TIMING("G", IO%IU6, "POTLOK")
            DWRITE0 'potlok is ok'

            CALL SETDIJ(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                        LMDIM, CDIJ, CQIJ, CVTOT, IRDMAA, IRDMAX)

            CALL SETDIJ_AVEC(WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, INFO%LOVERL, &
                             LMDIM, CDIJ, HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)

            CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ(1, 1, 1, 1), RHOLM, CRHODE(1, 1, 1, 1), &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.FALSE.)

            CALL UPDATE_CMBJ(GRIDC, T_INFO, LATT_CUR, IO%IU6)

            CALL STOP_TIMING("G", IO%IU6, "SETDIJ")
            DWRITE0 'setdij is ok'
#else
            CALL UPDATE_POTENTIAL( &
                KINEDEN, HAMILTONIAN, P, WDES, NONLR_S, NONL_S, LATT_CUR, &
                T_INFO, INFO, IO, &
                GRID, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, &
                CHTOT, DENCOR, CVTOT, CSTRF, &
                LMDIM, IRDMAX, CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, CHDEN, SV)
#endif

            IF (USELDApU()) CALL LDAPLUSU_PRINTOCC(WDES, T_INFO%NIONS, T_INFO%ITYP, IO%IU6)
!remove
            INFO%LPOTOK = .TRUE.
        END IF

        IF (LCORREL()) THEN
            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)
            DWRITE0 'orthch is ok'
        END IF

!======================== SUBROUTINE EDDSPX ============================
!
! these subroutines improve the electronic degrees of freedom
! using band by band schemes
! the Harris functional is used for the calculation
! of the total (free) energy so
! E  =  Tr[ H rho ] - d.c. (from input potential)
!
!=======================================================================
        DESUM1 = 0
        RMS = 0
        ICOUEV = 0

        LDELAY = .FALSE.
! if Davidson and RMM are selected, use Davidsons algorithm during
! delay phase
        IF (INFO%LRMM .AND. INFO%LDAVID .AND. (N <= ABS(INFO%NELMDL) .OR. N == 1)) LDELAY = .TRUE.
! if LDELAY is set, subspace rotation and orthogonalisations can be bypassed
! since they are done by the Davidson algorithm

!
! sub space rotation before eigenvalue optimization
!
        IF (INFO%LPDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3    ! exact diagonalization
            ELSE
                IFLAG = 4    ! using Loewdin perturbation theory
            END IF
            IF (INFO%IALGO == 3) THEN
                IFLAG = 0
            END IF
            IF (N < ABS(INFO%NELMDL)) IFLAG = 13

            IF (INFO%IALGO /= 2) THEN
                CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                            LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            END IF

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "eddiag is ok"
        ELSEIF (MOD(INFO%ICHARG, 10) == 5) THEN
            ! for ICHARG=5, the states have been rotated according to the supplied GAMMA
            ! diagonalize back to our current one-electron Hamiltonian
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!-----------------------------------------------------------------------
        select_algo: IF (INFO%LEXACT_DIAG) THEN
            CALL EDDIAG_EXACT(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                              LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, IO%IU6, E%EXHF, E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
            DWRITE0 "EDDIAG_EXACT is ok"

        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
!
! RMM-DIIS algorithm
!
            CALL EDDRMM(HAMILTONIAN, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0, &
                        N < ABS(INFO%NELMDL) - ABS(INFO%NELMDL)/4)
            ! previous line selects  special algorithm during delay

            CALL STOP_TIMING("G", IO%IU6, "RMM-DIIS", XMLTAG="diis")
            DWRITE0 "eddrmm is ok"

        ELSE IF (INFO%LDAVID) THEN
!
! blocked Davidson algorithm,
!
            NSIM = WDES%NSIM*2
#ifdef MPI
            NSIM = ((WDES%NSIM*2 + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            CALL EDDAV(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                       LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                       LDELAY=.FALSE., LSUBROTI=INFO%LDIAG, LEMPTY=.FALSE., LHF=N >= ABS(INFO%NELMDL), &
                       EXHF_ACFDT=E%EXHF_ACFDT)
            CALL STOP_TIMING("G", IO%IU6, "EDDAV", XMLTAG="dav")
            DWRITE0 "edddav is ok"

        ELSE IF (INFO%IHARMONIC > 0) THEN
            NSIM = WDES%NSIM
#ifdef MPI
            NSIM = ((WDES%NSIM + WDES%COMM_INTER%NCPU - 1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
            IF (INFO%EREF == 0) THEN
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE.)
            ELSE
                CALL EDDAV_INNER(HAMILTONIAN, P, GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, NSIM, &
                                 LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, E%EXHF, IO%IU6, IO%IU0, &
                                 LEMPTY=.FALSE., EREF=INFO%EREF)
            END IF
            CALL STOP_TIMING("G", IO%IU6, "EINNER", XMLTAG="dav")
            ! since we iterate deep it is safer to recalculate projeciton operators
            ! and reorthogonalize
            CALL PROALL(GRID, LATT_CUR, NONLR_S, NONL_S, W)

            ELSE IF (INFO%IALGO == 5 .OR. INFO%IALGO == 6 .OR. &
            &         INFO%IALGO == 7 .OR. INFO%IALGO == 8 .OR. INFO%IALGO == 0) THEN select_algo

!
! CG (Teter, Alan, Payne) potential is fixed !!
!
            CALL EDSTEP(GRID, INFO, LATT_CUR, NONLR_S, NONL_S, W, WDES, &
                        LMDIM, CDIJ, CQIJ, RMS, DESUM1, ICOUEV, SV, IO%IU6, IO%IU0)

            CALL STOP_TIMING("G", IO%IU6, "EDSTEP", XMLTAG="cg")
            DWRITE0 "edstep is ok"

        END IF select_algo
!-----------------------------------------------------------------------
! orthogonalise all bands (necessary only for residuum-minimizer
! or inner eigenvalue problems, since they iterate very deep)
!
        IF (.NOT. INFO%LORTHO .AND. .NOT. LDELAY) THEN

            CALL ORTHCH(WDES, W, INFO%LOVERL, LMDIM, CQIJ)

            CALL STOP_TIMING("G", IO%IU6, "ORTHCH", XMLTAG="orth")
            DWRITE0 "ortch is ok"
        END IF
!
! sub space rotation after eigen value optimization
!
        IF (INFO%LCDIAG .AND. .NOT. LDELAY) THEN

            IF (INFO%LDIAG) THEN
                IFLAG = 3
            ELSE
                IFLAG = 4
            END IF

            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL EDDIAG(HAMILTONIAN, GRID, LATT_CUR, NONLR_S, NONL_S, W, WDES, SYMM, &
                        LMDIM, CDIJ, CQIJ, IFLAG, SV, T_INFO, P, IO%IU0, E%EXHF, EXHF_ACFDT=E%EXHF_ACFDT)

            CALL STOP_TIMING("G", IO%IU6, "EDDIAG", XMLTAG="diag")
        END IF
!=======================================================================
! recalculate the broadened density of states and fermi-weights
! recalculate depletion charge size
!=======================================================================
        CALL MRG_CEL(WDES, W)
        IF (INFO%IALGO /= 3) THEN
            E%EENTROPY = 0
            DOS = 0
            DOSI = 0
            CALL DENSTA(IO%IU0, IO%IU6, WDES, W, KPOINTS, INFO%NELECT, &
                        INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE., &
                        NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
        END IF
        DWRITE0 "densta is ok"
!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
        E%EBANDSTR = BANDSTRUCTURE_ENERGY(WDES, W)
        TOTEN = E%EBANDSTR + E%DENC + E%XCENC + E%TEWEN + E%PSCENC + E%EENTROPY + E%PAWPS + E%PAWAE + INFO%EALLAT &
        & + E%EXHF + ECORE() + Ediel_sol
!-MM- Added to accomodate constrained moment calculations
        IF (M_CONSTRAINED()) TOTEN = TOTEN + E_CONSTRAINT()
!-MM- end of additions
!---- write total energy to OSZICAR file and stdout
        DESUM(N) = TOTEN - TOTENL
        ECONV = DESUM(N)
        io_begin
305     FORMAT('CG : ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                            &       I6, '  ', E10.3)
302     FORMAT('NONE ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                            &       I6, '  ', E10.3)
303     FORMAT('EIG: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                            &       I6, '  ', E10.3)
304     FORMAT('DIA: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                                            &       I6, '  ', E10.3)
1303    FORMAT('RMM: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                                                            &       I6, '  ', E10.3)
10303   FORMAT('DAV: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                               &       I6, '  ', E10.3)
20303   FORMAT('JDH: ', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                               &       I6, '  ', E10.3)
30303   FORMAT('DAVI:', I3, '   ', E20.12, '   ', E12.5, '   ', E12.5, &
                                                               &       I6, '  ', E10.3)

        IF (INFO%LEXACT_DIAG) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LRMM .AND. .NOT. LDELAY) THEN
            WRITE (17, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 1303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 1) THEN
            WRITE (17, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 20303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IHARMONIC == 2) THEN
            WRITE (17, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 30303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%LDAVID) THEN
            WRITE (17, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 10303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 4) THEN
            WRITE (17, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 304, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 3) THEN
            WRITE (17, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 303, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE IF (INFO%IALGO == 2) THEN
            WRITE (17, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 302, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        ELSE
            WRITE (17, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 305, ADVANCE='NO') N, TOTEN, DESUM(N), DESUM1, ICOUEV, RMS
        END IF

        CALL STOP_TIMING("G", IO%IU6, "DOS")
        io_end
!=======================================================================
!  Test for Break condition
!=======================================================================

        INFO%LABORT = .FALSE.
        LABORT_WITHOUT_CONV = .FALSE.

!-----eigenvalues and energy must be converged
        ! IF (ABS(DESUM(N)) < INFO%EDIFF .AND. ABS(DESUM1) < INFO%EDIFF) INFO%LABORT = .TRUE.
        IF (mod(N - NELM_SC_INITIAL, NELM_SC_INTER + 1) == 0 .AND. ABS(DESUM(N)) < INFO%EDIFF) INFO%LABORT = .TRUE.
!-----charge-density not constant and in last cycle no change of charge
        IF (.NOT. INFO%LMIX .AND. .NOT. INFO%LCHCON .AND. MIX%IMIX /= 0) INFO%LABORT = .FALSE.
!-----do not stop during the non-selfconsistent startup phase
        IF (N <= ABS(INFO%NELMDL)) INFO%LABORT = .FALSE.
!-----do not stop before minimum number of iterations is reached
        IF (N < ABS(INFO%NELMIN)) INFO%LABORT = .FALSE.
!-----but stop after INFO%NELM steps no matter where we are now
        IF (N >= INFO%NELM) THEN
            IF (.NOT. INFO%LABORT) LABORT_WITHOUT_CONV = .TRUE.
            INFO%LABORT = .TRUE.
        END IF

        IF ((IO%LORBIT >= 10) .AND. (MOD(N, 5) == 0) .AND. WDES%LNONCOLLINEAR) THEN
            CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
        END IF
! ======================================================================
! If the end of the electronic loop is reached
! calculate accurate initial state core level shifts
! if required
! ======================================================================
        IF (INFO%LABORT .AND. ACCURATE_CORE_LEVEL_SHIFTS()) THEN

            ALLOCATE (CDIJ_TMP(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ))
            CDIJ_TMP = CDIJ

            CALL SET_DD_PAW(WDES, P, T_INFO, INFO%LOVERL, &
                            WDES%NCDIJ, LMDIM, CDIJ_TMP(1, 1, 1, 1), RHOLM, CRHODE, &
                            E, LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL=.TRUE.)
            DEALLOCATE (CDIJ_TMP)
        END IF
!========================= subroutine CHSP  ============================
! if charge density is updated
!  ) first copy current charge to CHTOTL
!  ) set  INFO%LPOTOK to .F. this requires a recalculation of the local pot.
!  ) set INFO%LMIX to .T.
!  ) call subroutine SET_CHARGE to generate the new charge density
!  ) then performe mixing
! MIND:
! ) if delay is selected  do not update
! ) if convergence corrections to forces are calculated do not update charge
!   in last iteration
!=======================================================================
        IF (MOD(INFO%ICHARG, 10) == 5) THEN
            IF (IO%LORBIT == 14) CALL SPHPRO_FAST( &
                GRID, LATT_CUR, P, T_INFO, W, WDES, 71, IO%IU6, &
                INFO%LOVERL, LMDIM, CQIJ, LDIMP, LDIMP, LMDIMP, .FALSE., IO%LORBIT, PAR, &
                EFERMI, KPOINTS%EMIN, KPOINTS%EMAX)
            CALL LPRJ_PROALL(W, WDES, GRID, P, CQIJ, LATT_CUR, T_INFO, INFO, IO%IU6, IO%IU0)
            CALL LPRJ_WRITE(IO%IU6, IO%IU0, W)
            CALL LPRJ_LDApU(IO, W)   ! write a LDA+U GAMMA file
            CALL LPRJ_DEALLOC_COVL
        END IF

        INFO%LMIX = .FALSE.
        MIX%NEIG = 0

        IF (INFO%LABORT) THEN
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do
            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            write (17, *) "Added one:"
            CALL WRITE_CONSTRAINED_M(17, .TRUE.)
            io_end
            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        END IF

        ! IF (.NOT. INFO%LCHCON .AND. .NOT. (INFO%LABORT .AND. INFO%LCORR) .AND. N >= ABS(INFO%NELMDL)) THEN
        IF (.NOT. INFO%LCHCON .AND. .NOT. (INFO%LABORT .AND. INFO%LCORR)) THEN

            DO ISP = 1, WDES%NCDIJ
                CALL RC_ADD(CHTOT(1, ISP), 1.0_q, CHTOT(1, ISP), 0.0_q, CHTOTL(1, ISP), GRIDC)
            END DO

            IF (LDO_METAGGA() .AND. LMIX_TAU()) THEN
                DO ISP = 1, WDES%NCDIJ
                    CALL RC_ADD(KINEDEN%TAU(1, ISP), 1.0_q, KINEDEN%TAU(1, ISP), 0.0_q, KINEDEN%TAUL(1, ISP), GRIDC)
                END DO
            END IF

            RHOLM_LAST = RHOLM

            INFO%LPOTOK = .FALSE.

            IF (MOD(INFO%ICHARG, 10) == 5) THEN
                CALL REMOVE_VASP_LOCK(WDES%COMM) ! remove vasp.lock file
                CALL WAIT_VASP_LOCK(WDES%COMM)   ! wait that it is re-created by other process
                CALL ADD_GAMMA_FROM_FILE(WDES, W, KPOINTS, INFO%NELECT, INFO%NUP_DOWN, INFO%LABORT, IO)
            END IF

            CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
                            GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
                            LATT_CUR, P, SYMM, T_INFO, &
                            CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

            CALL SET_KINEDEN(GRID, GRID_SOFT, GRIDC, SOFT_TO_C, LATT_CUR, SYMM, &
                             T_INFO%NIONS, W, WDES, KINEDEN)

            CALL STOP_TIMING("G", IO%IU6, "CHARGE")

! ======================================================================
! where lambda optimization based on subspace rotation should be executed
! MIND:
!  ) must after SET_CHARGE to change the output density
! ======================================================================

            TOTEN_RESERVE = TOTEN

            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            call WRITE_CONSTRAINED_M(17, .TRUE.)
            if (DEBUG_SC) then
                if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-iterative-diagonalization spin:"
                if (IO%IU0 >= 0) write (IO%IU0, *) MW
                if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
                if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            end if
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do

            if (DEBUG_SC) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "(before) current total energy = ", TOTEN
                if (IO%IU0 >= 0) write (IO%IU0, *) "(before) reserved total energy = ", TOTEN_RESERVE
                if (IO%IU0 >= 0) write (IO%IU0, *) "(before) last total energy = ", TOTENL
                io_end
            end if

            if (N >= NELM_SC_INITIAL .and. mod(N - NELM_SC_INITIAL, NELM_SC_INTER + 1) == 0 .and. INFO%LABORT == .FALSE.) then

                if (SCDECOUPLE == 1) then
                    call lambda_inner_optimization_decouple( &
                        HAMILTONIAN, KINEDEN, &
                        P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                        T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                        GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                        CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                        CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                        CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                        NSTEP, LMDIM, IRDMAX, NEDOS, &
                        TOTEN, EFERMI, LDIMP, LMDIMP, N)

                else if (SCDECOUPLE == 2) then
                    call lambda_inner_optimization_decouple_more( &
                        HAMILTONIAN, KINEDEN, &
                        P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                        T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                        GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                        CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                        CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                        CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                        NSTEP, LMDIM, IRDMAX, NEDOS, &
                        TOTEN, EFERMI, LDIMP, LMDIMP, N)

                else
                    call lambda_inner_optimization( &
                        HAMILTONIAN, KINEDEN, &
                        P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                        T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                        GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                        CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                        CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                        CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                        NSTEP, LMDIM, IRDMAX, NEDOS, &
                        TOTEN, EFERMI, LDIMP, LMDIMP, N)
                end if

                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "final optimal lambda: "
                if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR
                io_end
            end if

            TOTEN = TOTEN_RESERVE

            if (DEBUG_SC) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "current total energy = ", TOTEN
                io_end
            end if

            if (DEBUG_SC) then
                do ISP = 1, WDES%NCDIJ
                    call FFT3D(CHTOT(1, ISP), GRIDC, 1)
                end do

                call M_INT(CHTOT, GRIDC, WDES)
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-optimization spin:"
                if (IO%IU0 >= 0) write (IO%IU0, *) MW
                if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
                if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
                io_end

                do ISP = 1, WDES%NCDIJ
                    call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                    call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
                end do
            end if
!-----------------------------------------------------------------------

            IF (MIX%IMIX /= 0) THEN
                INFO%LMIX = .TRUE.
! update one center charge density
                IF (LCORREL()) THEN
                    CALL SET_RHO_ONE_CENTRE(CRHODE, RHO_ONE_CENTRE)
                END IF

                IF (MIX%IMIX == 4) THEN
!  broyden mixing ... :
                    IF (LCORREL()) THEN
                        CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                        &   (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                        &   LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                        &   RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR, &
                        &   NMIX_ONE_CENTRE, RHO_ONE_CENTRE, RHO_ONE_CENTRE_LAST)
                        MIX%LRESET = .FALSE.
                    ELSE
                        CALL BRMIX(KINEDEN, GRIDB, GRIDC, IO, MIX, B_TO_C, &
                        &   (2*GRIDC%MPLWV), CHTOT, CHTOTL, WDES%NCDIJ, LATT_CUR%B, &
                        &   LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                        &   RMST, RMSC, RMSP, WEIGHT, .TRUE., IERRBR)
                        MIX%LRESET = .FALSE.
                    END IF
                ELSE
!  simple mixing ... :
                    RMST = 0
                    CALL MIX_SIMPLE(GRIDC, MIX, WDES%NCDIJ, CHTOT, CHTOTL, &
                                    N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
! test
                    IF (LCORREL()) THEN
                        RHO_ONE_CENTRE(:) = RHO_ONE_CENTRE_LAST(:) + &
                        &   0.2*(RHO_ONE_CENTRE(:) - RHO_ONE_CENTRE_LAST(:))
                        RHO_ONE_CENTRE_LAST(:) = RHO_ONE_CENTRE(:)
                    END IF
! test
                END IF
                CALL STOP_TIMING("G", IO%IU6, "MIXING")
                DWRITE0 "mixing is ok"

!---- ENDIF (MIX%IMIX/=0)     end of mixing
            END IF

            if (DEBUG_SC_Q) then
                do ISP = 1, WDES%NCDIJ
                    call FFT3D(CHTOT(1, ISP), GRIDC, 1)
                end do
                call M_INT(CHTOT, GRIDC, WDES)
                do ISP = 1, WDES%NCDIJ
                    call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                    call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
                end do
            end if
!-----ENDIF (.NOT.INFO%LCHCON)   end of charge update
        END IF

        IF (W%OVER_BAND) THEN
            CALL REDIS_PW_OVER_BANDS(WDES, W)
            CALL STOP_TIMING("G", IO%IU6, "REDIS")
        END IF
!=======================================================================
! total time used for this step
!=======================================================================
        CALL SEPERATOR_TIMING(IO%IU6)
        CALL STOP_TIMING("LOOP", IO%IU6, XMLTAG='total')

!=======================================================================
!  important write statements
!=======================================================================

2440    FORMAT(/' eigenvalue-minimisations  :', I6, / &
                                                              &       ' total energy-change (2. order) :', E14.7, '  (', E14.7, ')')
2441    FORMAT(/ &
                                                                                            &       ' Broyden mixing:'/ &
                                                                    &       '  rms(total) =', E12.5, '    rms(broyden)=', E12.5, / &
                                                                                            &       '  rms(prec ) =', E12.5/ &
                                                                                      &       '  weight for this iteration ', F10.2)

2442    FORMAT(/' eigenvalues of (default mixing * dielectric matrix)'/ &
                '  average eigenvalue GAMMA= ', F8.4, /(10F8.4))

200     FORMAT(' number of electron ', F15.7, ' magnetization ', 3F15.7)
201     FORMAT(' augmentation part  ', F15.7, ' magnetization ', 3F15.7)

        DO I = 1, WDES%NCDIJ
            RHOTOT(I) = RHO0(GRIDC, CHTOT(1, I))
            RHOAUG(I) = RHOTOT(I) - RHO0(GRID_SOFT, CHDEN(1, I))
        END DO

        io_begin

! iteration counts
        WRITE (IO%IU6, 2440) ICOUEV, DESUM(N), DESUM1

! charge density
        WRITE (IO%IU6, 200) RHOTOT
        IF (INFO%LOVERL) THEN
            WRITE (IO%IU6, 201) RHOAUG
        END IF
! dipol moment
        IF (DIP%LCOR_DIP) CALL WRITE_DIP(IO%IU6)

! mixing
        IF (INFO%LMIX .AND. MIX%IMIX == 4) THEN
            IF (IERRBR /= 0) THEN
                IF (IO%IU0 >= 0) &
                    WRITE (IO%IU0, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
                    'mixing'' now and reset mixing at next step!'
                IF (IO%IU6 >= 0) &
                    WRITE (IO%IU6, *) 'ERROR: Broyden mixing failed, tried ''simple '// &
                    'mixing'' now and reset mixing at next step!'
            END IF

            IF (IO%NWRITE >= 2 .OR. NSTEP == 1) THEN
                WRITE (IO%IU6, 2441) RMST, RMSC, RMSP, WEIGHT
                IF (ABS(RMST - RMSC)/RMST > 0.1_q) THEN
                    WRITE (IO%IU6, *) ' WARNING: grid for Broyden might be to small'
                END IF
            END IF
            IF (IO%IU0 >= 0) WRITE (IO%IU0, 308) RMST
            WRITE (17, 308) RMST
308         FORMAT('   ', E10.3)
            IF (MIX%NEIG > 0) THEN
                WRITE (IO%IU6, 2442) MIX%AMEAN, MIX%EIGENVAL(1:MIX%NEIG)
            END IF
        ELSE IF (INFO%LMIX) THEN
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, 308) RMST
            WRITE (17, 308) RMST
        ELSE
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, *)
            WRITE (17, *)
        END IF

        if (DEBUG_SC) then
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-mixing spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        end if

        io1: IF (IO%NWRITE >= 2 .OR. (NSTEP == 1)) THEN
! energy
            IF (LCORREL()) THEN
                WRITE (IO%IU6, 7241) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT + ECORE(), Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            ELSE
                WRITE (IO%IU6, 7240) E%PSCENC, E%TEWEN, E%DENC, E%EXHF, E%XCENC, E%PAWPS, E%PAWAE, &
                    E%EENTROPY, E%EBANDSTR, INFO%EALLAT, Ediel_sol, TOTEN, &
                    TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER)
            END IF

            IF (LHFCALC) THEN
                WRITE (IO%IU6, '( "  exchange ACFDT corr.  = ",F18.8,"  see jH, gK, PRB 81, 115126")') E%EXHF_ACFDT
            END IF

7240        FORMAT(/ &
                                                       ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  atomic energy  EATOM  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                                       &        '  energy(sigma->0) =', F18.8)
7241        FORMAT(/ &
                                                       ' Free energy of the ion-electron system (eV)'/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  alpha Z        PSCENC = ', F18.8/ &
                                                       &        '  Ewald energy   TEWEN  = ', F18.8/ &
                                                       &        '  -Hartree energ DENC   = ', F18.8/ &
                                                       &        '  -exchange      EXHF   = ', F18.8/ &
                                                       &        '  -V(xc)+E(xc)   XCENC  = ', F18.8/ &
                                                       &        '  PAW double counting   = ', 2F18.8/ &
                                                       &        '  entropy T*S    EENTRO = ', F18.8/ &
                                                       &        '  eigenvalues    EBANDS = ', F18.8/ &
                                                       &        '  core contrib.  ECORE  = ', F18.8/ &
                                                       &        '  Solvation  Ediel_sol  = ', F18.8/ &
                                                       &        '  ---------------------------------------------------'/ &
                                                       &        '  free energy    TOTEN  = ', F18.8, ' eV'// &
                                                       &        '  energy without entropy =', F18.8, &
                                   &        '  energy(sigma->0) =', F18.8)
72612       FORMAT(//&
                                                  &        '  METAGGA EXCHANGE AND CORRELATION (eV)'/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  LDA+GGA E(xc)  EXCG   = ', F18.6/ &
                                                  &        '  LDA+GGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  core xc             AE= ', 1F18.6/ &
                                                  &        '  metaGGA E(xc)  EXCM   = ', F18.6/ &
                                                  &        '  metaGGA PAW    PS : AE= ', 2F18.6/ &
                                                  &        '  metaGGA core xc     AE= ', 1F18.6/ &
                                                  &        '  ---------------------------------------------------'/ &
                                                  &        '  METAGGA result:'/ &
                                                  &        '  free  energy   TOTEN  = ', F18.6, ' eV'// &
                                                  &        '  energy  without entropy=', F18.6, &
                                                  &        '  energy(sigma->0) =', F16.6)
            ELSE io1
            WRITE (IO%IU6, 7242) TOTEN, TOTEN - E%EENTROPY
7242        FORMAT(/'  free energy = ', E20.12, &
                                                                                   &        '  energy without entropy= ', E20.12)

        END IF io1
!     too slow on many servers nowadays
!     IF (IO%LOPEN) CALL WFORCE(IO%IU6)
!     IF (IO%LOPEN) CALL WFORCE(17)
        WRITE (IO%IU6, 130)
        io_end
!=======================================================================
!  perform some additional write statments if required
!=======================================================================
!-----Eigenvalues and weights
        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN

            ! calculate and print the core level shifts
            IF (INFO%LOVERL) THEN
                CALL CL_SHIFT_PW(GRIDC, LATT_CUR, IRDMAX, &
                                 T_INFO, P, WDES%NCDIJ, CVTOT, MAX(INFO%ENAUG, INFO%ENMAX), IO%IU6)
            ELSE
                IF (IO%IU0 >= 0) WRITE (IO%IU0, *) "WARNING: NC-PP core level shifts not calculated"
            END IF
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            CALL KPAR_SYNC_CELTOT(WDES, W)

            io_begin
            CALL RHOAT0(P, T_INFO, BETATO, LATT_CUR%OMEGA)

            WRITE (IO%IU6, 2202) EFERMI, REAL(E%CVZERO, KIND=q), E%PSCENC/INFO%NELECT + BETATO
2202        FORMAT(' E-fermi : ', F8.4, '     XC(G=0): ', F8.4, &
                                                                                                &         '     alpha+bet :', F8.4/)

            IF (INFO%IHARMONIC == 1) THEN
                CALL WRITE_EIGENVAL_RESIDUAL(WDES, W, IO%IU6)
            ELSE
                CALL WRITE_EIGENVAL(WDES, W, IO%IU6)
            END IF
            io_end
        END IF

        IF (((NSTEP == 1 .OR. NSTEP == DYN%NSW) .AND. INFO%LABORT) .OR. &
        &     (IO%NWRITE >= 1 .AND. INFO%LABORT) .OR. IO%NWRITE >= 3) THEN
            io_begin
!-----Charge-density along one line
            WRITE (IO%IU6, 130)
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'soft charge-density along one line, spin component', I
                WRITE (IO%IU6, '(10(6X,I4))') (II, II=0, 9)
                CALL WRT_RC_LINE(IO%IU6, GRID_SOFT, CHDEN(1, I))
                IF (INFO%LOVERL) THEN
                    WRITE (IO%IU6, *) 'total charge-density along one line'
                    CALL WRT_RC_LINE(IO%IU6, GRIDC, CHTOT(1, I))
                END IF
                WRITE (IO%IU6, *)
            END DO
!-----pseudopotential strength and augmentation charge
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'pseudopotential strength for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
             &             (CDIJ(L, LP, NI, I), L=1, MIN(8, P(1)%LMMAX))
!     &             (REAL(CDIJ(L,LP,NI,I),q),AIMAG(CDIJ(L,LP,1,I))*1000,L=1,MIN(16,P(1)%LMMAX))
                END DO
            END DO
            END DO

            IF (INFO%LOVERL) THEN
            DO NI = 1, 1
            DO I = 1, WDES%NCDIJ
                WRITE (IO%IU6, *) 'total augmentation occupancy for first ion, spin component:', I
                DO LP = 1, P(1)%LMMAX
                    WRITE (IO%IU6, '(16(F7.3,1X))') &
             &             (REAL(CRHODE(L, LP, NI, I), q), L=1, MIN(16, P(1)%LMMAX))
                END DO
!           DO LP=1,P(1)%LMMAX
!              WRITE(IO%IU6,'(16(F7.3,1X))') &
!     &             (AIMAG(CRHODE(L,LP,1,I))*1E6,L=1,MIN(16,P(1)%LMMAX))
!           ENDDO
            END DO
            END DO
            END IF
            io_end

        END IF
!=======================================================================
!  xml related output
!=======================================================================
        CALL XML_TAG("energy")
        IF (INFO%LABORT .OR. N == 1) THEN
            CALL XML_TAG_REAL("alphaZ", E%PSCENC)
            CALL XML_TAG_REAL("ewald", E%TEWEN)
            CALL XML_TAG_REAL("hartreedc", E%DENC)
            CALL XML_TAG_REAL("XCdc", E%XCENC)
            CALL XML_TAG_REAL("pawpsdc", E%PAWPS)
            CALL XML_TAG_REAL("pawaedc", E%PAWAE)
            CALL XML_TAG_REAL("eentropy", E%EENTROPY)
            CALL XML_TAG_REAL("bandstr", E%EBANDSTR)
            CALL XML_TAG_REAL("atom", INFO%EALLAT)
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        ELSE
            CALL XML_ENERGY(TOTEN, TOTEN - E%EENTROPY, TOTEN - E%EENTROPY/(2 + NORDER))
        END IF
        CALL XML_CLOSE_TAG
        CALL XML_CLOSE_TAG("scstep")
!=======================================================================
! relaxed core related output
!=======================================================================
        IF (INFO%LABORT .AND. LCORREL()) THEN
            CALL REPORT(.TRUE.)
! uncomment this line for specific postprocessing of relaxed core stuff
!        CALL RCPOSTPROC
!
        END IF
!======================== end of loop ENDLSC ===========================
! This is the end of the selfconsistent calculation loop
!=======================================================================
        IF (INFO%LABORT) THEN
            io_begin
            WRITE (IO%IU6, 131)
131         FORMAT(5X, //, &
                                                                        &  '------------------------ aborting loop because EDIFF', &
                                                                        &  ' is reached ----------------------------------------'//)
            io_end
            EXIT electron
        END IF
        INFO%LSOFT = .FALSE.
#ifndef noSTOPCAR
#ifndef F90_T3D
        CALL RDATAB(IO%LOPEN, 'STOPCAR', 99, 'LABORT', '=', '#', ';', 'L', &
        &            IDUM, RDUM, CDUM, INFO%LSOFT, CHARAC, NCOUNT, 1, IERR)
        ITMP = 0; IF (INFO%LSOFT) ITMP = 1; CALLMPI(M_sum_i(W%WDES%COMM, ITMP, 1))
        IF (ITMP > 0) INFO%LSOFT = .TRUE.
#endif
#endif
        IF (INFO%LSOFT) THEN
            io_begin
            IF (IO%IU0 >= 0) &
                WRITE (IO%IU0, *) 'hard stop encountered!  aborting job ...'
            WRITE (IO%IU6, 13131)
13131       FORMAT(5X, //, &
                                                                         &  '------------------------ aborting loop because hard', &
                                                                       &  ' stop was set ---------------------------------------'//)
            io_end
            EXIT electron
        END IF
        TOTENL = TOTEN

        if (DEBUG_SC) then
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) end-of-SCF-step spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        end if
    END DO electron

! calculate dipol corrections now
!
    IF (DIP%IDIPCO > 0) THEN
        IF (.NOT. DIP%LCOR_DIP) THEN
            CALL CDIPOL_CHTOT_REC(GRIDC, LATT_CUR, P, T_INFO, &
                                  CHTOT, CSTRF, CVTOT, WDES%NCDIJ, INFO%NELECT)

            CALL WRITE_DIP(IO%IU6)
            IF (IO%IU6 > 0) THEN
                WRITE (IO%IU6, *)
                WRITE (IO%IU6, *) &
                    " *************** adding dipol energy to TOTEN NOW **************** "
            END IF
            TOTEN = TOTEN + DIP%ECORR
        END IF
    END IF

! notify calling routine whether convergence has been reached
    INFO%LABORT = LABORT_WITHOUT_CONV

    DWRITE0 'electron left'

    RETURN
END SUBROUTINE ELMIN_SASC

subroutine ELMIN_SASC_QUADRATIC(HAMILTONIAN, KINEDEN, &
                                P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                                T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                                GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                                CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                                CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                                CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                                NSTEP, LMDIM, IRDMAX, NEDOS, &
                                TOTEN, EFERMI, LDIMP, LMDIMP)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj

!=======================================================================
!  structures
!=======================================================================
    type(ham_handle) HAMILTONIAN
    type(tau_handle) KINEDEN
    type(type_info) T_INFO
    type(potcar) P(T_INFO%NTYP)
    type(wavedes) WDES
    type(nonlr_struct) NONLR_S
    type(nonl_struct) NONL_S
    type(wavespin) W
    ! type(wavespin) W_RESERVE, W_F_RESERVE, W_G_RESERVE             ! wavefunction
    type(wavespin) W_F     ! wavefunction for all bands simultaneous
    type(wavespin) W_G     ! same as above
    type(latt) LATT_CUR
    type(dynamics) DYN
    type(info_struct) INFO
    type(in_struct) IO
    type(mixing) MIX
    type(kpoints_struct) KPOINTS
    type(symmetry) SYMM
    type(grid_3d) GRID       ! grid for wavefunctions
    type(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    type(grid_3d) GRIDC      ! grid for potentials/charge
    type(grid_3d) GRIDUS     ! temporary grid in us.F
    type(grid_3d) GRIDB      ! Broyden grid
    type(transit) B_TO_C     ! index table between GRIDB and GRIDC
    type(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    type(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    type(energy) E
    type(latt) LATT_INI

    integer NSTEP, LMDIM, IRDMAX, NEDOS
    real(q) :: TOTEN, EFERMI
    real(q) :: TOTEN_RESERVE

!    real(q) :: ECONV

    complex(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    ! complex(q) CHTOT_RESERVE(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    ! complex(q) CHTOTL_RESERVE(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    complex(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    complex(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    integer N_MIX_PAW
    real(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
    ! real(q) RHOLM_RESERVE(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST_RESERVE(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    complex(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    real(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    real(q) :: XCSIF(3, 3)
! parameteL_CONSTRrs for FAST_SPHPRO
    integer :: LDIMP, LMDIMP

    real(q), allocatable :: nu(:, :), target_spin(:, :), spin(:, :), spin_old(:, :)
    real(q), allocatable :: spin_plus(:, :)
    real(q), allocatable :: delta_spin(:, :), delta_spin_old(:, :)
    real(q), allocatable :: search(:, :), search_old(:, :)

    real(q), allocatable :: alpha_trial(:, :), alpha_opt(:, :), alpha_new(:, :), beta(:, :)
    real(q), allocatable :: mean_error(:, :), mean_error_old(:, :)

    real(q), allocatable:: sum_k(:, :), sum_k2(:, :)
    real(q), allocatable:: ratio(:, :)

    real(q) :: ekt
    real(q) :: epsilon
    real(q) :: EDIFF_RESERVE
    integer :: num_atom
    integer :: num_step
    integer :: i_step

    allocate (nu(3, T_INFO%NIONS))
    allocate (sum_k, sum_k2, ratio, mold=nu)
    allocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old, mold=nu)
    allocate (search, search_old, mold=nu)
    allocate (alpha_trial, alpha_opt, alpha_new, mold=nu)
    allocate (beta, mean_error, mean_error_old, mold=nu)

    SCTYPE_CURRENT = 2
    L_CONSTR = L_CONSTR_Q
    where (CONSTRL_Q == 0) L_CONSTR = 0.0

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "-------------------------------------------------------------------------------"
    if (IO%IU0 >= 0) write (IO%IU0, *) "SASC(Q) (Self-Adaptive Spin Constraint (Quadratic Penalty))"
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, es9.3)') "Initial trial step size (ekt) = ", INI_SC_ALPHA_Q
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0)') "Maximum number of steps in SC iteration = ", CONSTR_NUM_STEP_Q
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, es9.3)') "Convergence criterion of SC iteration (epsilon) = ", CONSTR_EPSILON_Q
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, i0, a, i0)') "Constrained atoms: ", count(CONSTRL_Q == 1)/3, "/", T_INFO%NIONS
    if (IO%IU0 >= 0) write (IO%IU0, '(1x, a, l1)') "Debug mode: ", DEBUG_SC_Q
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "--------------------------------------"
    if (IO%IU0 >= 0) write (IO%IU0, *) "The DEFINITION of atomic spins which are constrained (MW):"
    if (IO%IU0 >= 0) write (IO%IU0, *) "\vec{M}_{I}="
    if (IO%IU0 >= 0) write (IO%IU0, *) "\int_{\Omega_{I}} \vec{m}(\mathbf{r}) F_{I}(|\mathbf{r}|) d\mathbf{r}"
    if (IO%IU0 >= 0) write (IO%IU0, *)  &
    & "-------------------------------------------------------------------------------"
    io_end

    if (EDIFF_Q > 0) then
        EDIFF_RESERVE = INFO%EDIFF
        INFO%EDIFF = EDIFF_Q
    end if

    call ELMIN( &
        HAMILTONIAN, KINEDEN, &
        P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
        T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
        GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
        CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
        CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
        CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
        NSTEP, LMDIM, IRDMAX, NEDOS, &
        TOTEN, EFERMI, LDIMP, LMDIMP)
!=======================================================================
! this part performs the SASC (self-adaptive magnetization constrain)
!=======================================================================

    ekt = INI_SC_ALPHA_Q
    target_spin = M_CONSTR
    num_step = CONSTR_NUM_STEP_Q
    epsilon = CONSTR_EPSILON_Q

    do i_step = 1, num_step

        if (i_step == 1) then
            nu = L_CONSTR
            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "Initial lambda:"
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR
            if (IO%IU0 >= 0) write (IO%IU0, *) "Initial spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            io_end
        else
            call ELMIN( &
                HAMILTONIAN, KINEDEN, &
                P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                NSTEP, LMDIM, IRDMAX, NEDOS, &
                TOTEN, EFERMI, LDIMP, LMDIMP)
            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "Optimal lambda: "
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR
            if (IO%IU0 >= 0) write (IO%IU0, *) "Current spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            io_end
        end if

        delta_spin = target_spin - spin  ! gradient
        where (CONSTRL_Q == 0) delta_spin = 0.0  ! mask delta_spin
        search = delta_spin

        mean_error = delta_spin

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, 15589) i_step, maxval(abs(delta_spin))
15589   format("Step = ", i0, "       Diff (+Inf-Norm) = ", es20.12)
        if (IO%IU0 >= 0) write (IO%IU0, *)  &
        & "==============================================================================="
        io_end

        if (maxval(abs(delta_spin)) < epsilon) then
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "Meet convergence criterion, exit."
            io_end
            exit
        end if

        if (i_step > 1) then
            ! Fletcher-Reeves
            beta = mean_error/mean_error_old
            ! where (CONSTRL == 0) beta = 0.0
            ! Polak-Ribiere
            ! beta = (mean_error - sum(delta_spin_old * delta_spin) / num_atom) / mean_error_old
            ! Hestenes-Stiefel
            ! beta = sum((delta_spin_old - delta_spin) * delta_spin) / sum((delta_spin_old - delta_spin) * search_old)
            ! Dai-Yuan (not working)
            ! beta = sum(delta_spin ** 2) / sum((delta_spin_old - delta_spin)* search_old))
            !search = search + beta * search_old
            search = search + beta*search_old
        end if

        if (i_step == 1) alpha_trial = 1.*ekt/maxval(abs(search))
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Trial alpha:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_trial
        io_end
        !===========================================================
        !  line search for an optimized alpha
        !===========================================================

        ! trial step
        nu = nu*(1.0 + alpha_trial*abs(search))
        L_CONSTR = nu
        where (CONSTRL_Q == 0) L_CONSTR = 0.0

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Trial lambda: "
        if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR
        io_end

        call ELMIN( &
            HAMILTONIAN, KINEDEN, &
            P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
            T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
            GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
            CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
            CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
            CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
            NSTEP, LMDIM, IRDMAX, NEDOS, &
            TOTEN, EFERMI, LDIMP, LMDIMP)
        spin_plus = MW

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Trial spin (spin_plus):"
        if (IO%IU0 >= 0) write (IO%IU0, *) spin_plus
        if (IO%IU0 >= 0) write (IO%IU0, *) "Target spin: "
        if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Target spin - Old spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) target_spin - spin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Trial spin - Old spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) spin_plus - spin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Target spin - Old spin)/(Trial spin - Old spin):"
        if (IO%IU0 >= 0) write (IO%IU0, *) (spin - target_spin)/(spin - spin_plus)
        io_end

        !if (sum(abs(spin-spin_plus))/num_atom < 5.0e-6) exit

        ! adjust alpha but not exact line search
        sum_k = ((target_spin - spin)*(spin_plus - spin))        !Benxu 20210510
        sum_k2 = (spin_plus - spin)**2
        ratio = (target_spin - spin)/(spin_plus - spin)
        if (maxval(abs(ratio)) > 2.) then
            ratio = 2.0*ratio/maxval(abs(ratio))
        end if
        alpha_opt = sign(1.6**(ratio - 1.), (ratio - 1.))*alpha

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Adjustment ratio:"
        if (IO%IU0 >= 0) write (IO%IU0, *) ratio
        if (IO%IU0 >= 0) write (IO%IU0, *) "Optimal alpha:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_opt
        if (IO%IU0 >= 0) write (IO%IU0, *) "|Target spin - Trial spin|:"
        if (IO%IU0 >= 0) write (IO%IU0, *) abs(spin_plus - target_spin)
        io_end

        alpha_new = alpha_opt

        ! restrict step size    not used at the moment ! Benxu 20210408
        ! if (maxval(abs(alpha_new * search)) > 3.* ekt  ) then
        !    alpha_new = sign(3.0, alpha_new)* ekt / maxval(abs(search))
        !    alpha_opt = sign(3.0, alpha_new)* ekt / maxval(abs(search))
        ! end if

        nu = nu*(1.0 + alpha_new*abs(spin_plus - target_spin)) !Benxu 20210416

        L_CONSTR = nu
        where (CONSTRL_Q == 0) L_CONSTR = 0.0

        search_old = search
        delta_spin_old = delta_spin
        mean_error_old = mean_error

    end do

    if (EDIFF_Q > 0) then
        INFO%EDIFF = EDIFF_RESERVE
    end if
    deallocate (nu)
    deallocate (sum_k, sum_k2, ratio)
    deallocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old)
    deallocate (search, search_old)
    deallocate (alpha_trial, alpha_opt, alpha_new)
    deallocate (beta, mean_error, mean_error_old)

end subroutine ELMIN_SASC_QUADRATIC

! MIX (no mix for subrot)
! ALGO
! W
! CHTOT
! NELM

subroutine lambda_inner_optimization(HAMILTONIAN, KINEDEN, &
                                     P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                                     T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                                     GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                                     CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                                     CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                                     CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                                     NSTEP, LMDIM, IRDMAX, NEDOS, &
                                     TOTEN, EFERMI, LDIMP, LMDIMP, &
                                     N)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj

!=======================================================================
!  structures
!=======================================================================
    type(ham_handle) HAMILTONIAN
    type(tau_handle) KINEDEN
    type(type_info) T_INFO
    type(potcar) P(T_INFO%NTYP)
    type(wavedes) WDES
    type(nonlr_struct) NONLR_S
    type(nonl_struct) NONL_S
    type(wavespin) W, W_RESERVE          ! wavefunction
    type(wavespin) W_F, W_F_RESERVE        ! wavefunction for all bands simultaneous
    type(wavespin) W_G, W_G_RESERVE        ! same as above
    type(latt) LATT_CUR
    type(dynamics) DYN
    type(info_struct) INFO
    type(in_struct) IO
    type(mixing) MIX
    type(kpoints_struct) KPOINTS
    type(symmetry) SYMM
    type(grid_3d) GRID       ! grid for wavefunctions
    type(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    type(grid_3d) GRIDC      ! grid for potentials/charge
    type(grid_3d) GRIDUS     ! temporary grid in us.F
    type(grid_3d) GRIDB      ! Broyden grid
    type(transit) B_TO_C     ! index table between GRIDB and GRIDC
    type(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    type(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    type(energy) E
    type(latt) LATT_INI

    integer NSTEP, LMDIM, IRDMAX, NEDOS
    real(q) :: TOTEN, EFERMI

!    real(q) :: ECONV

    complex(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOT_RESERVE(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    complex(q) CHTOTL_RESERVE(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    complex(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    complex(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    integer N_MIX_PAW
    real(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
    real(q) RHOLM_RESERVE(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST_RESERVE(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    complex(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    real(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    real(q) :: XCSIF(3, 3)
! parameters for FAST_SPHPRO
    integer :: LDIMP, LMDIMP

    real(q), allocatable :: nu(:, :), dnu(:, :), target_spin(:, :), spin(:, :)
    real(q), allocatable :: spin_plus(:, :)
    real(q), allocatable :: delta_spin(:, :), delta_spin_old(:, :)
    real(q), allocatable :: search(:, :), search_old(:, :)
    real(q), allocatable :: target_spin_mask(:, :), spin_mask(:, :), spin_plus_mask(:, :)

    real(q) :: epsilon
    real(q) :: alpha_trial, alpha_opt, alpha_plus, beta
    real(q) :: mean_error, mean_error_old, rms_error
    real(q) :: g
    real(q) :: restrict, restrict_current
    real(q) :: boundary

    integer :: num_atom
    integer :: num_step
    integer :: i_step
    integer :: cg_beta
    !character(len=1024) :: i_step_string

    real(q) :: sum_k, sum_k2

    integer :: N

    integer :: N_RESERVE, NELM_RESERVE, IALGO_RESERVE, IHARMONIC_RESERVE, NELMDL_RESERVE, NELMIN_RESERVE
    logical :: LCHCOS_RESERVE, LONESW_RESERVE, LONESW_AUTO_RESERVE, LDAVID_RESERVE, &
    & LRMM_RESERVE, LORTHO_RESERVE, LCDIAG_RESERVE, LPDIAG_RESERVE, LPRECONDH_RESERVE, &
    & LEXACT_DIAG_RESERVE

    allocate (nu(3, T_INFO%NIONS))
    allocate (dnu, mold=nu)
    allocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old, mold=nu)
    allocate (search, search_old, mold=nu)
    allocate (target_spin_mask, spin_mask, spin_plus_mask, mold=nu)

    target_spin = M_CONSTR
    dnu = 0
    ! num_atom = T_INFO%NIONS
    num_atom = count(CONSTRL == 1)/3

    if (DECAY_EPSILON > 0) then
        epsilon = CONSTR_EPSILON*(DECAY_EPSILON**(N - NELM_SC_INITIAL))
        if (epsilon < LBOUND_EPSILON) epsilon = LBOUND_EPSILON
    else
        epsilon = CONSTR_EPSILON
    end if

    alpha_trial = INI_SC_ALPHA
    num_step = CONSTR_NUM_STEP
    restrict = CONSTR_RESTRICT
    cg_beta = ALGO_SC

    CHTOT_RESERVE = CHTOT
    CHTOTL_RESERVE = CHTOTL
    RHOLM_RESERVE = RHOLM
    RHOLM_LAST_RESERVE = RHOLM_LAST
    W_RESERVE = W
    ! TOTEN_RESERVE = TOTEN
    ! io_begin
    ! if (IO%IU0 >= 0) write (IO%IU0, *) "(in) reserved total energy = ", TOTEN_RESERVE
    ! io_end

    NELM_RESERVE = INFO%NELM
    NELMDL_RESERVE = INFO%NELMDL
    NELMIN_RESERVE = INFO%NELMIN
    IALGO_RESERVE = INFO%IALGO
    LCHCOS_RESERVE = INFO%LCHCOS
    LONESW_RESERVE = INFO%LONESW
    LONESW_AUTO_RESERVE = INFO%LONESW_AUTO
    LDAVID_RESERVE = INFO%LDAVID
    LRMM_RESERVE = INFO%LRMM
    LORTHO_RESERVE = INFO%LORTHO
    LCDIAG_RESERVE = INFO%LCDIAG
    LPDIAG_RESERVE = INFO%LPDIAG
    LPRECONDH_RESERVE = INFO%LPRECONDH
    IHARMONIC_RESERVE = INFO%IHARMONIC
    LEXACT_DIAG_RESERVE = INFO%LEXACT_DIAG

    INFO%NELMDL = 0
    INFO%NELMIN = 0
    INFO%NELM = 1

    if (ALGO_SC_DIAG == 1 .or. ALGO_SC_DIAG == 3) then
        INFO%IALGO = 4
        INFO%LCHCOS = .TRUE.
        INFO%LONESW = .FALSE.
        INFO%LONESW_AUTO = .FALSE.
        INFO%LDAVID = .FALSE.
        INFO%LRMM = .FALSE.
        INFO%LORTHO = .TRUE.
        INFO%LCDIAG = .FALSE.
        INFO%LPDIAG = .TRUE.
        INFO%LPRECONDH = .FALSE.
        INFO%IHARMONIC = 0
        INFO%LEXACT_DIAG = .FALSE.
    else if (ALGO_SC_DIAG == 2) then
    else
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported diagonalization algorithm for SASC, abort."
        io_end
        stop
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda begins ..."
    if (IO%IU0 >= 0) write (IO%IU0, *) "Covergence criterion this loop:", epsilon
    io_end

    do i_step = 1, num_step

! restrict_current = restrict * 0.9 ** (i_step - 1) + 1.0e-3
        restrict_current = restrict

        if (i_step == 1) then
            nu = L_CONSTR
            L_CONSTR_L_DIAG = nu
            where (CONSTRL == 0) L_CONSTR_L_DIAG = 0.0
            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial lambda:"
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_DIAG
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "optimal delta lambda: "
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
            io_end

! Hamiltonian except spin constriant remain the same
            CHTOT = CHTOTL_RESERVE
            RHOLM = RHOLM_LAST_RESERVE
            W = W_RESERVE
            call ELMIN_SASC_IN( &
                HAMILTONIAN, KINEDEN, &
                P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                NSTEP, LMDIM, IRDMAX, NEDOS, &
                TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "current spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        end if

        delta_spin = spin - target_spin  ! gradient
        where (CONSTRL == 0) delta_spin = 0.0  ! mask delta_spin
        search = delta_spin
        mean_error = sum(delta_spin**2)/num_atom
        rms_error = sqrt(mean_error/3)

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, 5589) N, i_step, rms_error
5589    format("Step (Outer -- Inner) =  ", i0, " -- ", i0, "       RMS =", es20.12)
        if (IO%IU0 >= 0) write (IO%IU0, *) &
        & "-------------------------------------------------------------------------------"
        io_end

        if (rms_error < epsilon .or. i_step == num_step) then
            if (rms_error < epsilon) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Meet convergence criterion ( < ", epsilon, " ), exit."
                io_end
            else if (i_step == num_step) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, i0, a)') "Reach maximum number of steps ( ", num_step, " ), exit."
                io_end
            end if

            ! whether or not a final Davidson is appended to inner iteration
            if (ALGO_SC_DIAG == 3) then
                INFO%IALGO = IALGO_RESERVE
                INFO%LCHCOS = LCHCOS_RESERVE
                INFO%LONESW = LONESW_RESERVE
                INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
                INFO%LDAVID = LDAVID_RESERVE
                INFO%LRMM = LRMM_RESERVE
                INFO%LORTHO = LORTHO_RESERVE
                INFO%LCDIAG = LCDIAG_RESERVE
                INFO%LPDIAG = LPDIAG_RESERVE
                INFO%LPRECONDH = LPRECONDH_RESERVE
                INFO%IHARMONIC = IHARMONIC_RESERVE
                INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

                CHTOT = CHTOTL_RESERVE
                RHOLM = RHOLM_LAST_RESERVE
                W = W_RESERVE
                call ELMIN_SASC_IN( &
                    HAMILTONIAN, KINEDEN, &
                    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                    NSTEP, LMDIM, IRDMAX, NEDOS, &
                    TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)
            end if
            L_CONSTR = L_CONSTR_L_DIAG + L_CONSTR_L_ADD
            exit
        end if

        if (i_step > 1) then
            if (cg_beta == 1) then
                ! Fletcher-Reeves
                beta = mean_error/mean_error_old
            else if (cg_beta == 2) then
                ! Polak-Ribiere
                beta = (mean_error - sum(delta_spin_old*delta_spin)/num_atom)/mean_error_old
            else if (cg_beta == 3) then
                ! Hestenes-Stiefel
                beta = sum((delta_spin_old - delta_spin)*delta_spin)/sum((delta_spin_old - delta_spin)*search_old)
            else if (cg_beta == 4) then
                ! Dai-Yuan (not working)
                beta = sum(delta_spin**2)/sum((delta_spin_old - delta_spin)*search_old)
            else
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported spin constraint algorithm, abort."
                io_end
                stop
            end if
            search = search + beta*search_old
        end if

        ! restrict trial step
        boundary = abs(alpha_trial*maxval(abs(search)))

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "restriction of this step = ", restrict_current
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial before restrict = ", alpha_trial
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = ", boundary
        ! if (boundary > restrict_current) then
        !     if (IO%IU0 >= 0) write(IO%IU0,*) "trial need restriction: true"
        !     alpha_trial = sign(1.0, alpha_trial) * restrict_current / maxval(abs(search))
        !     boundary = abs(alpha_trial*maxval(abs(search)))
        !     if (IO%IU0 >= 0) write(IO%IU0,*) "alpha_trial after restrict = ", alpha_trial
        !     if (IO%IU0 >= 0) write(IO%IU0,*) "boundary after = ", boundary
        ! else
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial need restriction: false"
        ! end if
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_trial*search
        io_end

        dnu = dnu + alpha_trial*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
        io_end

        CHTOT = CHTOTL_RESERVE
        RHOLM = RHOLM_LAST_RESERVE
        W = W_RESERVE

        if (DEBUG_SC) then
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-trial-step spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        end if

        call ELMIN_SASC_IN( &
            HAMILTONIAN, KINEDEN, &
            P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
            T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
            GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
            CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
            CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
            CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
            NSTEP, LMDIM, IRDMAX, NEDOS, &
            TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

        spin_plus = MW

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "current spin(trial):"
        if (IO%IU0 >= 0) write (IO%IU0, *) spin_plus
        io_end

! adjust alpha but not exact line search
        target_spin_mask = target_spin
        spin_mask = spin
        spin_plus_mask = spin_plus
        where (CONSTRL == 0) 
            target_spin_mask = 0.0
            spin_mask = 0.0
            spin_plus_mask = 0.0
        end where

        sum_k = sum((target_spin_mask - spin_mask)*(spin_plus_mask - spin_mask))
        sum_k2 = sum((spin_mask - spin_plus_mask)**2)
        alpha_opt = sum_k*alpha_trial/sum_k2
        cos_theta = sum_k/sqrt(sum((target_spin_mask - spin_mask)**2)*sum((spin_plus_mask - spin_mask)**2))
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "cosine between target and trial: ", cos_theta
        io_end
        ! alpha_opt = adjust_alpha_noncollinear(spin, spin_plus, target_spin, alpha_trial)
        ! call adjust_alpha_noncollinear(spin, spin_plus, target_spin, alpha_trial, alpha_opt, IO%IU0, NODE_ME, IONODE)

! restrict adapted step
        boundary = abs(alpha_opt*maxval(abs(search)))
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt before restrict = ", alpha_opt
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = ", boundary
        io_end

        if (CONV_BOUND > 0 .and. boundary <= CONV_BOUND) then
            io_begin
          if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Reach limitation of current step ( boundary < ", CONV_BOUND, " ), exit."
            io_end
            exit
        end if

        if (CONSTR_RESTRICT > 0 .and. boundary > restrict_current) then
            alpha_opt = sign(1.0, alpha_opt)*restrict_current/maxval(abs(search))
            boundary = abs(alpha_opt*maxval(abs(search)))
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: true"
            if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt after restrict = ", alpha_opt
            if (IO%IU0 >= 0) write (IO%IU0, *) "boundary after = ", boundary
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: false"
            io_end
        end if

        alpha_plus = alpha_opt - alpha_trial

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_plus*search
        io_end

! update
        dnu = dnu + alpha_plus*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        search_old = search
        delta_spin_old = delta_spin
        mean_error_old = mean_error

! adjust trial step
        g = 1.5*abs(alpha_opt)/alpha_trial
        if (TRIAL_UPDATE_RESTRICT == .TRUE.) then
            if (g > 2.) g = 2.
            if (g < 0.5) g = 0.5
        end if
        alpha_trial = alpha_trial*g**0.7
        ! io_begin
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "g:",g
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial:", alpha_trial
        ! io_end

    end do

    deallocate (nu, dnu)
    deallocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old)
    deallocate (search, search_old)

    CHTOTL = CHTOTL_RESERVE
    !CHTOT = CHTOT
    RHOLM_LAST = RHOLM_LAST_RESERVE
    !RHOLM = RHOLM
    ! TOTEN = TOTEN_RESERVE
    ! io_begin
    ! if (IO%IU0 >= 0) write (IO%IU0, *) "(in) current total energy = ", TOTEN
    ! io_end
    ! W = W_RESERVE

    INFO%NELMDL = NELMDL_RESERVE
    INFO%NELMIN = NELMIN_RESERVE
    INFO%NELM = NELM_RESERVE
    INFO%IALGO = IALGO_RESERVE
    INFO%LCHCOS = LCHCOS_RESERVE
    INFO%LONESW = LONESW_RESERVE
    INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
    INFO%LDAVID = LDAVID_RESERVE
    INFO%LRMM = LRMM_RESERVE
    INFO%LORTHO = LORTHO_RESERVE
    INFO%LCDIAG = LCDIAG_RESERVE
    INFO%LPDIAG = LPDIAG_RESERVE
    INFO%LPRECONDH = LPRECONDH_RESERVE
    INFO%IHARMONIC = IHARMONIC_RESERVE
    INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

    INFO%LABORT = .FALSE.

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOTL(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOTL, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-iterative-diagonalization spin: (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOTL(1, ISP), CHTOTL(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOTL(1, ISP), GRIDC)
        end do
    end if

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOT(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOT, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-optimization spin:  (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
        end do
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda ends."
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    io_end

end subroutine lambda_inner_optimization

subroutine lambda_inner_optimization_decouple(HAMILTONIAN, KINEDEN, &
                                              P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                                              T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                                              GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                                              CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                                              CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                                              CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                                              NSTEP, LMDIM, IRDMAX, NEDOS, &
                                              TOTEN, EFERMI, LDIMP, LMDIMP, &
                                              N)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj

    !=======================================================================
    !  structures
    !=======================================================================
    type(ham_handle) HAMILTONIAN
    type(tau_handle) KINEDEN
    type(type_info) T_INFO
    type(potcar) P(T_INFO%NTYP)
    type(wavedes) WDES
    type(nonlr_struct) NONLR_S
    type(nonl_struct) NONL_S
    type(wavespin) W, W_RESERVE          ! wavefunction
    type(wavespin) W_F, W_F_RESERVE        ! wavefunction for all bands simultaneous
    type(wavespin) W_G, W_G_RESERVE        ! same as above
    type(latt) LATT_CUR
    type(dynamics) DYN
    type(info_struct) INFO
    type(in_struct) IO
    type(mixing) MIX
    type(kpoints_struct) KPOINTS
    type(symmetry) SYMM
    type(grid_3d) GRID       ! grid for wavefunctions
    type(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    type(grid_3d) GRIDC      ! grid for potentials/charge
    type(grid_3d) GRIDUS     ! temporary grid in us.F
    type(grid_3d) GRIDB      ! Broyden grid
    type(transit) B_TO_C     ! index table between GRIDB and GRIDC
    type(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    type(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    type(energy) E
    type(latt) LATT_INI

    integer NSTEP, LMDIM, IRDMAX, NEDOS
    real(q) :: TOTEN, EFERMI

!    real(q) :: ECONV

    complex(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOT_RESERVE(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    complex(q) CHTOTL_RESERVE(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    complex(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    complex(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

!   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
!  paw sphere charge density
    integer N_MIX_PAW
    real(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
    real(q) RHOLM_RESERVE(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST_RESERVE(N_MIX_PAW, WDES%NCDIJ)
!  charge-density and potential on soft grid
    complex(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
!  density of states
    real(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
!  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    real(q) :: XCSIF(3, 3)
! parameters for FAST_SPHPRO
    integer :: LDIMP, LMDIMP

    real(q), allocatable :: nu(:, :), dnu(:, :), target_spin(:, :), spin(:, :)
    real(q), allocatable :: spin_plus(:, :)
    real(q), allocatable :: delta_spin(:, :), delta_spin_old(:, :)
    real(q), allocatable :: search(:, :), search_old(:, :)
    real(q), allocatable :: alpha_trial(:), alpha_opt(:), alpha_plus(:), beta(:)
    real(q), allocatable :: mean_error(:), mean_error_old(:)
    real(q), allocatable :: g(:)
    real(q), allocatable :: search_max(:), boundary(:)
    real(q), allocatable :: mask_atom(:)
    real(q), allocatable :: sum_k(:), sum_k2(:)
    real(q), allocatable :: cos_theta(:)

    real(q) :: epsilon
    real(q) :: restrict, restrict_current
    real(q) :: rms_error

    integer :: num_atom
    integer :: num_step
    integer :: i_step
    integer :: cg_beta

    integer :: N

    integer :: N_RESERVE, NELM_RESERVE, IALGO_RESERVE, IHARMONIC_RESERVE, NELMDL_RESERVE, NELMIN_RESERVE
    logical :: LCHCOS_RESERVE, LONESW_RESERVE, LONESW_AUTO_RESERVE, LDAVID_RESERVE, &
    & LRMM_RESERVE, LORTHO_RESERVE, LCDIAG_RESERVE, LPDIAG_RESERVE, LPRECONDH_RESERVE, &
    & LEXACT_DIAG_RESERVE

    allocate (nu(3, T_INFO%NIONS))
    allocate (dnu, mold=nu)
    allocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old, mold=nu)
    allocate (search, search_old, mold=nu)
    allocate (alpha_trial(T_INFO%NIONS))
    allocate (alpha_opt, alpha_plus, beta, mean_error, mean_error_old, search_max, mask_atom, mold=alpha_trial)
    allocate (boundary, cos_theta, sum_k, sum_k2, mold=alpha_trial)

    target_spin = M_CONSTR
    dnu = 0
! num_atom = T_INFO%NIONS
    num_atom = count(CONSTRL == 1)/3
    mask_atom = CONSTRL(1, :)

    if (DECAY_EPSILON > 0) then
        epsilon = CONSTR_EPSILON*(DECAY_EPSILON**(N - NELM_SC_INITIAL))
        if (epsilon < LBOUND_EPSILON) epsilon = LBOUND_EPSILON
    else
        epsilon = CONSTR_EPSILON
    end if

    alpha_trial = INI_SC_ALPHA
    where (mask_atom == 0) alpha_trial = 0.0
    num_step = CONSTR_NUM_STEP
    restrict = CONSTR_RESTRICT
    cg_beta = ALGO_SC

    CHTOT_RESERVE = CHTOT
    CHTOTL_RESERVE = CHTOTL
    RHOLM_RESERVE = RHOLM
    RHOLM_LAST_RESERVE = RHOLM_LAST
    W_RESERVE = W
! TOTEN_RESERVE = TOTEN
! io_begin
! if (IO%IU0 >= 0) write (IO%IU0, *) "(in) reserved total energy = ", TOTEN_RESERVE
! io_end

    NELM_RESERVE = INFO%NELM
    NELMDL_RESERVE = INFO%NELMDL
    NELMIN_RESERVE = INFO%NELMIN
    IALGO_RESERVE = INFO%IALGO
    LCHCOS_RESERVE = INFO%LCHCOS
    LONESW_RESERVE = INFO%LONESW
    LONESW_AUTO_RESERVE = INFO%LONESW_AUTO
    LDAVID_RESERVE = INFO%LDAVID
    LRMM_RESERVE = INFO%LRMM
    LORTHO_RESERVE = INFO%LORTHO
    LCDIAG_RESERVE = INFO%LCDIAG
    LPDIAG_RESERVE = INFO%LPDIAG
    LPRECONDH_RESERVE = INFO%LPRECONDH
    IHARMONIC_RESERVE = INFO%IHARMONIC
    LEXACT_DIAG_RESERVE = INFO%LEXACT_DIAG

    INFO%NELMDL = 0
    INFO%NELMIN = 0
    INFO%NELM = 1

    if (ALGO_SC_DIAG == 1 .or. ALGO_SC_DIAG == 3) then
        INFO%IALGO = 4
        INFO%LCHCOS = .TRUE.
        INFO%LONESW = .FALSE.
        INFO%LONESW_AUTO = .FALSE.
        INFO%LDAVID = .FALSE.
        INFO%LRMM = .FALSE.
        INFO%LORTHO = .TRUE.
        INFO%LCDIAG = .FALSE.
        INFO%LPDIAG = .TRUE.
        INFO%LPRECONDH = .FALSE.
        INFO%IHARMONIC = 0
        INFO%LEXACT_DIAG = .FALSE.
    else if (ALGO_SC_DIAG == 2) then
    else
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported diagonalization algorithm for SASC, abort."
        io_end
        stop
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda begins (decoupled version) ..."
    if (IO%IU0 >= 0) write (IO%IU0, *) "Covergence criterion this loop:", epsilon
    io_end

    do i_step = 1, num_step

! restrict_current = restrict * 0.9 ** (i_step - 1) + 1.0e-3
        restrict_current = restrict

        if (i_step == 1) then
            nu = L_CONSTR
            L_CONSTR_L_DIAG = nu
            where (CONSTRL == 0) L_CONSTR_L_DIAG = 0.0
            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial lambda:"
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_DIAG
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "optimal delta lambda: "
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
            io_end

! Hamiltonian except spin constriant remain the same
            CHTOT = CHTOTL_RESERVE
            RHOLM = RHOLM_LAST_RESERVE
            W = W_RESERVE
            call ELMIN_SASC_IN( &
                HAMILTONIAN, KINEDEN, &
                P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                NSTEP, LMDIM, IRDMAX, NEDOS, &
                TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "current spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        end if

        delta_spin = spin - target_spin  ! gradient
        where (CONSTRL == 0) delta_spin = 0.0  ! mask delta_spin
        search = delta_spin
        ! objective function f(x) -> f(x)(N)
        mean_error = sum(delta_spin**2, dim=1)
        rms_error = sqrt(sum(delta_spin**2)/num_atom/3)

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, 5589) N, i_step, rms_error
5589    format("Step (Outer -- Inner) =  ", i0, " -- ", i0, "       RMS =", es20.12)
        if (IO%IU0 >= 0) write (IO%IU0, *) &
        & "-------------------------------------------------------------------------------"
        io_end

        if (rms_error < epsilon .or. i_step == num_step) then
            if (rms_error < epsilon) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Meet convergence criterion ( < ", epsilon, " ), exit."
                io_end
            else if (i_step == num_step) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, i0, a)') "Reach maximum number of steps ( ", num_step, " ), exit."
                io_end
            end if

            ! whether or not a final Davidson is appended to inner iteration
            if (ALGO_SC_DIAG == 3) then
                INFO%IALGO = IALGO_RESERVE
                INFO%LCHCOS = LCHCOS_RESERVE
                INFO%LONESW = LONESW_RESERVE
                INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
                INFO%LDAVID = LDAVID_RESERVE
                INFO%LRMM = LRMM_RESERVE
                INFO%LORTHO = LORTHO_RESERVE
                INFO%LCDIAG = LCDIAG_RESERVE
                INFO%LPDIAG = LPDIAG_RESERVE
                INFO%LPRECONDH = LPRECONDH_RESERVE
                INFO%IHARMONIC = IHARMONIC_RESERVE
                INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

                CHTOT = CHTOTL_RESERVE
                RHOLM = RHOLM_LAST_RESERVE
                W = W_RESERVE
                call ELMIN_SASC_IN( &
                    HAMILTONIAN, KINEDEN, &
                    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                    NSTEP, LMDIM, IRDMAX, NEDOS, &
                    TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)
            end if
            L_CONSTR = L_CONSTR_L_DIAG + L_CONSTR_L_ADD
            exit
        end if

        if (i_step > 1) then
            if (cg_beta == 1) then
                ! Fletcher-Reeves
                beta = mean_error/mean_error_old
            else if (cg_beta == 2) then
                ! Polak-Ribiere
                beta = (mean_error - sum(delta_spin_old*delta_spin, dim=1))/mean_error_old
            else if (cg_beta == 3) then
                ! Hestenes-Stiefel
                beta = sum((delta_spin_old - delta_spin)*delta_spin, dim=1)/sum((delta_spin_old - delta_spin)*search_old, dim=1)
            else if (cg_beta == 4) then
                ! Dai-Yuan (not working)
                beta = mean_error/sum((delta_spin_old - delta_spin)*search_old, dim=1)
            else
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported spin constraint algorithm, abort."
                io_end
                stop
            end if
            where (mask_atom == 0) beta = 0.0
            search = search + spread(beta, 1, 3)*search_old
            where (CONSTRL == 0) search = 0.0
        end if

! restrict trial step
        boundary = abs(alpha_trial*maxval(abs(search), dim=1))

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "restriction of this step = ", restrict_current
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial before restrict = "
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_trial
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = "
        if (IO%IU0 >= 0) write (IO%IU0, *) boundary
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial need restriction: false"
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) spread(alpha_trial, 1, 3)*search
        io_end

        dnu = dnu + spread(alpha_trial, 1, 3)*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
        io_end

        CHTOT = CHTOTL_RESERVE
        RHOLM = RHOLM_LAST_RESERVE
        W = W_RESERVE

        if (DEBUG_SC) then
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-trial-step spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        end if

        call ELMIN_SASC_IN( &
            HAMILTONIAN, KINEDEN, &
            P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
            T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
            GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
            CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
            CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
            CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
            NSTEP, LMDIM, IRDMAX, NEDOS, &
            TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

        spin_plus = MW

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "current spin(trial):"
        if (IO%IU0 >= 0) write (IO%IU0, *) spin_plus
        io_end

        ! adjust alpha but not exact line search
        sum_k = sum((target_spin - spin)*(spin_plus - spin), dim=1)
        sum_k2 = sum((spin - spin_plus)**2, dim=1)
        alpha_opt = sum_k*alpha_trial/sum_k2
        where (mask_atom == 0) alpha_opt = 0.0
        cos_theta = sum_k/sqrt(sum((target_spin - spin)**2, dim=1)*sum((spin_plus - spin)**2, dim=1))
        where (mask_atom == 0) cos_theta = 0.0
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "cosine between target and trial: ", cos_theta
        io_end

        ! restrict adapted step
        search_max = maxval(abs(search), dim=1)
        boundary = abs(alpha_opt*search_max)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt before restrict = "
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_opt
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = "
        if (IO%IU0 >= 0) write (IO%IU0, *) boundary
        io_end

        if (CONV_BOUND > 0 .and. maxval(boundary) <= CONV_BOUND) then
            io_begin
          if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Reach limitation of current step ( boundary < ", CONV_BOUND, " ), exit."
            io_end
            exit
        end if

        if (CONSTR_RESTRICT > 0 .and. maxval(boundary) > restrict_current) then
            where (boundary > restrict_current)
                alpha_opt = sign(1.0, alpha_opt)*restrict_current/search_max
            end where
            boundary = abs(alpha_opt*search_max)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: true"
            if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt after restrict = "
            if (IO%IU0 >= 0) write (IO%IU0, *) alpha_opt
            if (IO%IU0 >= 0) write (IO%IU0, *) "boundary after = "
            if (IO%IU0 >= 0) write (IO%IU0, *) boundary
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: false"
            io_end
        end if

        alpha_plus = alpha_opt - alpha_trial

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) spread(alpha_plus, 1, 3)*search
        io_end

        ! update
        dnu = dnu + spread(alpha_plus, 1, 3)*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        search_old = search
        delta_spin_old = delta_spin
        mean_error_old = mean_error

        ! adjust trial step
        g = 1.5*abs(alpha_opt)/alpha_trial
        if (TRIAL_UPDATE_RESTRICT == .TRUE.) then
            where (g > 2.) g = 2.
            where (g < 0.5) g = 0.5
            where (mask_atom == 0) g = 0.0
        end if
        alpha_trial = alpha_trial*g**0.7
        ! io_begin
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "g:",g
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial:", alpha_trial
        ! io_end

    end do

    deallocate (nu, dnu)
    deallocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old)
    deallocate (search, search_old)
    deallocate (alpha_opt, alpha_plus, beta, mean_error, mean_error_old, search_max, mask_atom, alpha_trial)
    deallocate (boundary, cos_theta, sum_k, sum_k2)

    CHTOTL = CHTOTL_RESERVE
    !CHTOT = CHTOT
    RHOLM_LAST = RHOLM_LAST_RESERVE
    !RHOLM = RHOLM
    ! TOTEN = TOTEN_RESERVE

    INFO%NELMDL = NELMDL_RESERVE
    INFO%NELMIN = NELMIN_RESERVE
    INFO%NELM = NELM_RESERVE
    INFO%IALGO = IALGO_RESERVE
    INFO%LCHCOS = LCHCOS_RESERVE
    INFO%LONESW = LONESW_RESERVE
    INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
    INFO%LDAVID = LDAVID_RESERVE
    INFO%LRMM = LRMM_RESERVE
    INFO%LORTHO = LORTHO_RESERVE
    INFO%LCDIAG = LCDIAG_RESERVE
    INFO%LPDIAG = LPDIAG_RESERVE
    INFO%LPRECONDH = LPRECONDH_RESERVE
    INFO%IHARMONIC = IHARMONIC_RESERVE
    INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

    INFO%LABORT = .FALSE.

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOTL(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOTL, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-iterative-diagonalization spin: (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOTL(1, ISP), CHTOTL(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOTL(1, ISP), GRIDC)
        end do
    end if

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOT(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOT, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-optimization spin:  (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
        end do
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda ends."
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    io_end

end subroutine lambda_inner_optimization_decouple

subroutine lambda_inner_optimization_decouple_more(HAMILTONIAN, KINEDEN, &
                                                   P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                                                   T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                                                   GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                                                   CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                                                   CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                                                   CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                                                   NSTEP, LMDIM, IRDMAX, NEDOS, &
                                                   TOTEN, EFERMI, LDIMP, LMDIMP, &
                                                   N)

    USE prec
    USE hamil_high
    USE morbitalmag
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
    USE david_inner
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
    USE Constrained_M_modular
    USE ini
    USE LDAPLUSU_MODULE
    USE core_rel
    USE pp_data
    USE gw_model
    USE meta
    USE locproj

    !=======================================================================
    !  structures
    !=======================================================================
    type(ham_handle) HAMILTONIAN
    type(tau_handle) KINEDEN
    type(type_info) T_INFO
    type(potcar) P(T_INFO%NTYP)
    type(wavedes) WDES
    type(nonlr_struct) NONLR_S
    type(nonl_struct) NONL_S
    type(wavespin) W, W_RESERVE          ! wavefunction
    type(wavespin) W_F, W_F_RESERVE        ! wavefunction for all bands simultaneous
    type(wavespin) W_G, W_G_RESERVE        ! same as above
    type(latt) LATT_CUR
    type(dynamics) DYN
    type(info_struct) INFO
    type(in_struct) IO
    type(mixing) MIX
    type(kpoints_struct) KPOINTS
    type(symmetry) SYMM
    type(grid_3d) GRID       ! grid for wavefunctions
    type(grid_3d) GRID_SOFT  ! grid for soft chargedensity
    type(grid_3d) GRIDC      ! grid for potentials/charge
    type(grid_3d) GRIDUS     ! temporary grid in us.F
    type(grid_3d) GRIDB      ! Broyden grid
    type(transit) B_TO_C     ! index table between GRIDB and GRIDC
    type(transit) C_TO_US    ! index table between GRIDC and GRIDUS
    type(transit) SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    type(energy) E
    type(latt) LATT_INI

    integer NSTEP, LMDIM, IRDMAX, NEDOS
    real(q) :: TOTEN, EFERMI

    !    real(q) :: ECONV

    complex(q) CHTOT(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOT_RESERVE(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
    complex(q) CHTOTL(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    complex(q) CHTOTL_RESERVE(GRIDC%MPLWV, WDES%NCDIJ)! old charge-density
    RGRID DENCOR(GRIDC%RL%NP)           ! partial core
    complex(q) CVTOT(GRIDC%MPLWV, WDES%NCDIJ) ! local potential
    complex(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP)! structure factor

    !   augmentation related quantities
    OVERLAP CDIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CQIJ(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ), &
        CRHODE(LMDIM, LMDIM, WDES%NIONS, WDES%NCDIJ)
    !  paw sphere charge density
    integer N_MIX_PAW
    real(q) RHOLM(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST(N_MIX_PAW, WDES%NCDIJ)
    real(q) RHOLM_RESERVE(N_MIX_PAW, WDES%NCDIJ), RHOLM_LAST_RESERVE(N_MIX_PAW, WDES%NCDIJ)
    !  charge-density and potential on soft grid
    complex(q) CHDEN(GRID_SOFT%MPLWV, WDES%NCDIJ)
    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
    !  density of states
    real(q) DOS(NEDOS, WDES%ISPIN), DOSI(NEDOS, WDES%ISPIN)
    !  Hamiltonian
    GDEF CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), &
        CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
    real(q) :: XCSIF(3, 3)
    ! parameters for FAST_SPHPRO
    integer :: LDIMP, LMDIMP

    real(q), allocatable :: nu(:, :), dnu(:, :), target_spin(:, :), spin(:, :)
    real(q), allocatable :: spin_plus(:, :)
    real(q), allocatable :: delta_spin(:, :), delta_spin_old(:, :)
    real(q), allocatable :: search(:, :), search_old(:, :)
    real(q), allocatable :: alpha_trial(:, :), alpha_opt(:, :), alpha_plus(:, :), beta(:, :)
    real(q), allocatable :: mean_error(:, :), mean_error_old(:, :)
    real(q), allocatable :: g(:, :)
    real(q), allocatable :: search_max(:, :), boundary(:, :)
    ! real(q), allocatable :: mask_atom(:)
    real(q), allocatable :: sum_k(:, :), sum_k2(:, :)
    ! real(q), allocatable :: cos_theta(:)

    real(q) :: epsilon
    real(q) :: restrict, restrict_current
    real(q) :: rms_error

    integer :: num_atom
    integer :: num_step
    integer :: i_step
    integer :: cg_beta

    integer :: N

    integer :: N_RESERVE, NELM_RESERVE, IALGO_RESERVE, IHARMONIC_RESERVE, NELMDL_RESERVE, NELMIN_RESERVE
    logical :: LCHCOS_RESERVE, LONESW_RESERVE, LONESW_AUTO_RESERVE, LDAVID_RESERVE, &
    & LRMM_RESERVE, LORTHO_RESERVE, LCDIAG_RESERVE, LPDIAG_RESERVE, LPRECONDH_RESERVE, &
    & LEXACT_DIAG_RESERVE

    allocate (nu(3, T_INFO%NIONS))
    allocate (dnu, mold=nu)
    allocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old, mold=nu)
    allocate (search, search_old, mold=nu)
    allocate (alpha_trial(3, T_INFO%NIONS))
    allocate (alpha_opt, alpha_plus, beta, mean_error, mean_error_old, search_max, mold=alpha_trial)
    allocate (boundary, sum_k, sum_k2, mold=alpha_trial)

    target_spin = M_CONSTR
    dnu = 0
    ! num_atom = T_INFO%NIONS
    num_atom = count(CONSTRL == 1)/3
    ! mask_atom = CONSTRL(1, :)

    if (DECAY_EPSILON > 0) then
        epsilon = CONSTR_EPSILON*(DECAY_EPSILON**(N - NELM_SC_INITIAL))
        if (epsilon < LBOUND_EPSILON) epsilon = LBOUND_EPSILON
    else
        epsilon = CONSTR_EPSILON
    end if

    alpha_trial = INI_SC_ALPHA
    where (CONSTRL == 0) alpha_trial = 0.0
    num_step = CONSTR_NUM_STEP
    restrict = CONSTR_RESTRICT
    cg_beta = ALGO_SC

    CHTOT_RESERVE = CHTOT
    CHTOTL_RESERVE = CHTOTL
    RHOLM_RESERVE = RHOLM
    RHOLM_LAST_RESERVE = RHOLM_LAST
    W_RESERVE = W
    ! TOTEN_RESERVE = TOTEN

    NELM_RESERVE = INFO%NELM
    NELMDL_RESERVE = INFO%NELMDL
    NELMIN_RESERVE = INFO%NELMIN
    IALGO_RESERVE = INFO%IALGO
    LCHCOS_RESERVE = INFO%LCHCOS
    LONESW_RESERVE = INFO%LONESW
    LONESW_AUTO_RESERVE = INFO%LONESW_AUTO
    LDAVID_RESERVE = INFO%LDAVID
    LRMM_RESERVE = INFO%LRMM
    LORTHO_RESERVE = INFO%LORTHO
    LCDIAG_RESERVE = INFO%LCDIAG
    LPDIAG_RESERVE = INFO%LPDIAG
    LPRECONDH_RESERVE = INFO%LPRECONDH
    IHARMONIC_RESERVE = INFO%IHARMONIC
    LEXACT_DIAG_RESERVE = INFO%LEXACT_DIAG

    INFO%NELMDL = 0
    INFO%NELMIN = 0
    INFO%NELM = 1

    if (ALGO_SC_DIAG == 1 .or. ALGO_SC_DIAG == 3) then
        INFO%IALGO = 4
        INFO%LCHCOS = .TRUE.
        INFO%LONESW = .FALSE.
        INFO%LONESW_AUTO = .FALSE.
        INFO%LDAVID = .FALSE.
        INFO%LRMM = .FALSE.
        INFO%LORTHO = .TRUE.
        INFO%LCDIAG = .FALSE.
        INFO%LPDIAG = .TRUE.
        INFO%LPRECONDH = .FALSE.
        INFO%IHARMONIC = 0
        INFO%LEXACT_DIAG = .FALSE.
    else if (ALGO_SC_DIAG == 2) then
    else
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported diagonalization algorithm for SASC, abort."
        io_end
        stop
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda begins (fully-decoupled version) ..."
    if (IO%IU0 >= 0) write (IO%IU0, *) "Covergence criterion this loop:", epsilon
    io_end

    do i_step = 1, num_step

        ! restrict_current = restrict * 0.9 ** (i_step - 1) + 1.0e-3
        restrict_current = restrict

        if (i_step == 1) then
            nu = L_CONSTR
            L_CONSTR_L_DIAG = nu
            where (CONSTRL == 0) L_CONSTR_L_DIAG = 0.0
            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial lambda:"
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_DIAG
            if (IO%IU0 >= 0) write (IO%IU0, *) "initial spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "optimal delta lambda: "
            if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
            io_end

            ! Hamiltonian except spin constriant remain the same
            CHTOT = CHTOTL_RESERVE
            RHOLM = RHOLM_LAST_RESERVE
            W = W_RESERVE
            call ELMIN_SASC_IN( &
                HAMILTONIAN, KINEDEN, &
                P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                NSTEP, LMDIM, IRDMAX, NEDOS, &
                TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

            spin = MW
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "current spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin
            if (IO%IU0 >= 0) write (IO%IU0, *) "target spin: "
            if (IO%IU0 >= 0) write (IO%IU0, *) target_spin
            io_end
        end if

        delta_spin = spin - target_spin  ! gradient
        where (CONSTRL == 0) delta_spin = 0.0  ! mask delta_spin
        search = delta_spin
        ! objective function f(x) -> f(x)(3, N)
        mean_error = delta_spin**2
        rms_error = sqrt(sum(delta_spin**2)/num_atom/3)

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, 5589) N, i_step, rms_error
5589    format("Step (Outer -- Inner) =  ", i0, " -- ", i0, "       RMS =", es20.12)
        if (IO%IU0 >= 0) write (IO%IU0, *) &
        & "-------------------------------------------------------------------------------"
        io_end

        if (rms_error < epsilon .or. i_step == num_step) then
            if (rms_error < epsilon) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Meet convergence criterion ( < ", epsilon, " ), exit."
                io_end
            else if (i_step == num_step) then
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, '(a, i0, a)') "Reach maximum number of steps ( ", num_step, " ), exit."
                io_end
            end if

            ! whether or not a final Davidson is appended to inner iteration
            if (ALGO_SC_DIAG == 3) then
                INFO%IALGO = IALGO_RESERVE
                INFO%LCHCOS = LCHCOS_RESERVE
                INFO%LONESW = LONESW_RESERVE
                INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
                INFO%LDAVID = LDAVID_RESERVE
                INFO%LRMM = LRMM_RESERVE
                INFO%LORTHO = LORTHO_RESERVE
                INFO%LCDIAG = LCDIAG_RESERVE
                INFO%LPDIAG = LPDIAG_RESERVE
                INFO%LPRECONDH = LPRECONDH_RESERVE
                INFO%IHARMONIC = IHARMONIC_RESERVE
                INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

                CHTOT = CHTOTL_RESERVE
                RHOLM = RHOLM_LAST_RESERVE
                W = W_RESERVE
                call ELMIN_SASC_IN( &
                    HAMILTONIAN, KINEDEN, &
                    P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
                    T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
                    GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
                    CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
                    CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                    CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
                    NSTEP, LMDIM, IRDMAX, NEDOS, &
                    TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)
            end if
            L_CONSTR = L_CONSTR_L_DIAG + L_CONSTR_L_ADD
            exit
        end if

        if (i_step > 1) then
            if (cg_beta == 1) then
                ! Fletcher-Reeves
                beta = mean_error/mean_error_old
            else if (cg_beta == 2) then
                ! Polak-Ribiere
                beta = (mean_error - delta_spin_old*delta_spin)/mean_error_old
            else if (cg_beta == 3) then
                ! Hestenes-Stiefel
                beta = (delta_spin_old - delta_spin)*delta_spin/((delta_spin_old - delta_spin)*search_old)
            else if (cg_beta == 4) then
                ! Dai-Yuan (not working)
                beta = mean_error/((delta_spin_old - delta_spin)*search_old)
            else
                io_begin
                if (IO%IU0 >= 0) write (IO%IU0, *) "Unsupported spin constraint algorithm, abort."
                io_end
                stop
            end if
            where (CONSTRL == 0) beta = 0.0
            search = search + beta*search_old
            where (CONSTRL == 0) search = 0.0
        end if

        ! restrict trial step
        boundary = abs(alpha_trial*search)

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "restriction of this step = ", restrict_current
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial before restrict = "
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_trial
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = "
        if (IO%IU0 >= 0) write (IO%IU0, *) boundary
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial need restriction: false"
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_trial*search
        io_end

        dnu = dnu + alpha_trial*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "trial delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) L_CONSTR_L_ADD
        io_end

        CHTOT = CHTOTL_RESERVE
        RHOLM = RHOLM_LAST_RESERVE
        W = W_RESERVE

        if (DEBUG_SC) then
            do ISP = 1, WDES%NCDIJ
                call FFT3D(CHTOT(1, ISP), GRIDC, 1)
            end do

            call M_INT(CHTOT, GRIDC, WDES)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-trial-step spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) MW
            if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
            if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
            io_end

            do ISP = 1, WDES%NCDIJ
                call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
                call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
            end do
        end if

        call ELMIN_SASC_IN( &
            HAMILTONIAN, KINEDEN, &
            P, WDES, NONLR_S, NONL_S, W, W_F, W_G, LATT_CUR, LATT_INI, &
            T_INFO, DYN, INFO, IO, MIX, KPOINTS, SYMM, GRID, GRID_SOFT, &
            GRIDC, GRIDB, GRIDUS, C_TO_US, B_TO_C, SOFT_TO_C, E, &
            CHTOT, CHTOTL, DENCOR, CVTOT, CSTRF, &
            CDIJ, CQIJ, CRHODE, N_MIX_PAW, RHOLM, RHOLM_LAST, &
            CHDEN, SV, DOS, DOSI, CHF, CHAM, ECONV, XCSIF, &
            NSTEP, LMDIM, IRDMAX, NEDOS, &
            TOTEN, EFERMI, LDIMP, LMDIMP, CHTOT_RESERVE)

        spin_plus = MW

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "current spin(trial):"
        if (IO%IU0 >= 0) write (IO%IU0, *) spin_plus
        io_end

        ! adjust alpha but not exact line search
        sum_k = (target_spin - spin)*(spin_plus - spin)
        sum_k2 = (spin - spin_plus)**2
        alpha_opt = sum_k*alpha_trial/sum_k2
        where (CONSTRL == 0) alpha_opt = 0.0

        ! restrict adapted step
        search_max = abs(search)
        boundary = abs(alpha_opt*search_max)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt before restrict = "
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_opt
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = "
        if (IO%IU0 >= 0) write (IO%IU0, *) boundary
        io_end

        if (CONV_BOUND > 0 .and. maxval(boundary) <= CONV_BOUND) then
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Reach limitation of current step ( boundary < ", CONV_BOUND, " ), exit."
            io_end
            exit
        end if

        if (CONSTR_RESTRICT > 0 .and. maxval(boundary) > restrict_current) then
            where (boundary > restrict_current)
                alpha_opt = sign(1.0, alpha_opt)*restrict_current/search_max
            end where
            boundary = abs(alpha_opt*search_max)
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: true"
            if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt after restrict = "
            if (IO%IU0 >= 0) write (IO%IU0, *) alpha_opt
            if (IO%IU0 >= 0) write (IO%IU0, *) "boundary after = "
            if (IO%IU0 >= 0) write (IO%IU0, *) boundary
            io_end
        else
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "restriction needed: false"
            io_end
        end if

        alpha_plus = alpha_opt - alpha_trial

        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "delta delta lambda:"
        if (IO%IU0 >= 0) write (IO%IU0, *) alpha_plus*search
        io_end

        ! update
        dnu = dnu + alpha_plus*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) L_CONSTR_L_ADD = 0.0

        search_old = search
        delta_spin_old = delta_spin
        mean_error_old = mean_error

        ! adjust trial step
        g = 1.5*abs(alpha_opt)/alpha_trial
        if (TRIAL_UPDATE_RESTRICT == .TRUE.) then
            where (g > 2.) g = 2.
            where (g < 0.5) g = 0.5
            where (CONSTRL == 0) g = 0.0
        end if
        alpha_trial = alpha_trial*g**0.7
        ! io_begin
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "g:",g
        ! if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_trial:", alpha_trial
        ! io_end

    end do

    deallocate (nu, dnu)
    deallocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old)
    deallocate (search, search_old)
    deallocate (alpha_opt, alpha_plus, beta, mean_error, mean_error_old, search_max, alpha_trial)
    deallocate (boundary, sum_k, sum_k2)

    CHTOTL = CHTOTL_RESERVE
    !CHTOT = CHTOT
    RHOLM_LAST = RHOLM_LAST_RESERVE
    !RHOLM = RHOLM
    ! TOTEN = TOTEN_RESERVE

    INFO%NELMDL = NELMDL_RESERVE
    INFO%NELMIN = NELMIN_RESERVE
    INFO%NELM = NELM_RESERVE
    INFO%IALGO = IALGO_RESERVE
    INFO%LCHCOS = LCHCOS_RESERVE
    INFO%LONESW = LONESW_RESERVE
    INFO%LONESW_AUTO = LONESW_AUTO_RESERVE
    INFO%LDAVID = LDAVID_RESERVE
    INFO%LRMM = LRMM_RESERVE
    INFO%LORTHO = LORTHO_RESERVE
    INFO%LCDIAG = LCDIAG_RESERVE
    INFO%LPDIAG = LPDIAG_RESERVE
    INFO%LPRECONDH = LPRECONDH_RESERVE
    INFO%IHARMONIC = IHARMONIC_RESERVE
    INFO%LEXACT_DIAG = LEXACT_DIAG_RESERVE

    INFO%LABORT = .FALSE.

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOTL(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOTL, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) before-iterative-diagonalization spin: (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOTL(1, ISP), CHTOTL(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOTL(1, ISP), GRIDC)
        end do
    end if

    if (DEBUG_SC) then
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOT(1, ISP), GRIDC, 1)
        end do

        call M_INT(CHTOT, GRIDC, WDES)
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) after-optimization spin:  (print in the inner loop)"
        if (IO%IU0 >= 0) write (IO%IU0, *) MW
        if (IO%IU0 >= 0) write (IO%IU0, *) "(Debug) target spin:"
        if (IO%IU0 >= 0) write (IO%IU0, *) M_CONSTR
        io_end

        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
        end do
    end if

    io_begin
    if (IO%IU0 >= 0) write (IO%IU0, *) "Inner optimization for lambda ends."
    if (IO%IU0 >= 0) write (IO%IU0, *) &
    & "==============================================================================="
    io_end

end subroutine lambda_inner_optimization_decouple_more

pure function adjust_alpha_noncollinear(spin_in, spin_plus_in, target_spin, alpha) result(alpha_opt)
    use prec
    implicit none
    real(q), intent(in) :: spin_in(:, :)
    real(q), intent(in) :: spin_plus_in(:, :)
    real(q), intent(in) :: target_spin(:, :)
    real(q), intent(in) :: alpha
    real(q) :: alpha_opt
    real(q) :: sum_k, sum_k2
    sum_k = sum((target_spin - spin_in)*(spin_plus_in - spin_in))
    sum_k2 = sum((spin_in - spin_plus_in)**2)
    alpha_opt = sum_k*alpha/sum_k2
end function adjust_alpha_noncollinear

