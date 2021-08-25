SUBROUTINE POTLOK_SASC_IN(GRID, GRIDC, GRID_SOFT, COMM_INTER, WDES, &
                          INFO, P, T_INFO, E, LATT_CUR, &
                          CHTOT, CSTRF, CVTOT, DENCOR, SV, SOFT_TO_C, XCSIF, CHTOTN)
    USE prec
    USE mpimy
    USE mgrid
    USE pseudo
    USE lattice
    USE poscar
    USE setexm
    USE base
    USE xcgrad
    USE wave
    USE mdipol
    USE meta
    USE Constrained_M_modular
    USE main_mpi, ONLY: COMM
! solvation__
    USE solvation
! solvation__
! bexternal__
    USE bexternal
! bexternal__

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A - B, D - H, O - Z)

    TYPE(grid_3d) GRID, GRIDC, GRID_SOFT
    TYPE(wavedes) WDES
    TYPE(transit) SOFT_TO_C
    TYPE(info_struct) INFO
    TYPE(type_info) T_INFO
    TYPE(potcar) P(T_INFO%NTYP)
    TYPE(energy) E
    TYPE(latt) LATT_CUR
    TYPE(communic) COMM_INTER

    RGRID SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
    COMPLEX(q) CSTRF(GRIDC%MPLWV, T_INFO%NTYP), &
        CHTOT(GRIDC%MPLWV, WDES%NCDIJ), CVTOT(GRIDC%MPLWV, WDES%NCDIJ)
    COMPLEX(q) CHTOTN(GRIDC%MPLWV, WDES%NCDIJ)
    RGRID DENCOR(GRIDC%RL%NP)
    REAL(q) XCSIF(3, 3), TMPSIF(3, 3)
! work arrays (allocated after call to FEXCG)
    COMPLEX(q), ALLOCATABLE::  CWORK1(:), CWORK(:, :)
    REAL(q) ELECTROSTATIC
    LOGICAL, EXTERNAL :: L_NO_LSDA_GLOBAL

#ifdef libbeef
    LOGICAL LBEEFCALCBASIS, LBEEFBAS
    COMMON/BEEFENS/LBEEFCALCBASIS, LBEEFBAS

    real(q), allocatable, save :: beefxc(:), beenergies(:)
    LOGICAL SAVELUSE_VDW
    INTEGER BEEFCOUNTER
#endif

    MWORK1 = MAX(GRIDC%MPLWV, GRID_SOFT%MPLWV)
    ALLOCATE (CWORK1(MWORK1), CWORK(GRIDC%MPLWV, WDES%NCDIJ))

!-----------------------------------------------------------------------
!
!  calculate the exchange correlation potential and the dc. correction
!
!-----------------------------------------------------------------------
    EXC = 0
    E%XCENC = 0
    E%EXCG = 0
    E%CVZERO = 0
    XCSIF = 0

    CVTOT = 0

#ifdef libbeef
    IF (LBEEFCALCBASIS) THEN
        BEEFCOUNTER = 1
        SAVELUSE_VDW = LUSE_VDW
!        for calculation of nscf xc energies for error estimates,
!        vdW interactions don't need to be calculated
        LUSE_VDW = .FALSE.
        IF (.NOT. ALLOCATED(beefxc)) ALLOCATE (beefxc(32))
        IF (.NOT. ALLOCATED(beenergies)) ALLOCATE (beenergies(2000))
    END IF
#endif

! test
! xc: IF (ISLDAXC()) THEN
    xc: IF (ISLDAXC() .AND. (.NOT. LDO_METAGGA())) THEN
! test
! transform the charge density to real space
        EXCG = 0
        XCENCG = 0
        CVZERG = 0
        TMPSIF = 0

        DO ISP = 1, WDES%NCDIJ
            CALL FFT3D(CHTOT(1, ISP), GRIDC, 1)
        END DO

#ifdef libbeef
4096    IF (LBEEFCALCBASIS) THEN
!          for calculation of nscf xc energies for error estimates,
!          cycle through beef legendre polynomial basis
            IF (BEEFCOUNTER .LT. 31) THEN
                CALL BEEFSETMODE(BEEFCOUNTER - 1)
            ELSE
                CALL BEEFSETMODE(BEEFCOUNTER - 34)
            END IF
        END IF
#endif

        IF (WDES%ISPIN == 2) THEN

! get the charge and the total magnetization
            CALL MAG_DENSITY(CHTOT, CWORK, GRIDC, WDES%NCDIJ)
! do LDA+U instead of LSDA+U
            IF (L_NO_LSDA_GLOBAL()) CWORK(:, 2) = 0
!
            IF (ISGGA()) THEN
                ! gradient corrections to LDA
                ! unfortunately FEXCGS requires (up,down) density
                ! instead of (rho,mag)
                CALL RL_FLIP(CWORK, GRIDC, 2, .TRUE.)
                ! GGA potential
                CALL FEXCGS(2, GRIDC, LATT_CUR, XCENCG, EXCG, CVZERG, TMPSIF, &
                            CWORK, CVTOT, DENCOR)
                CALL RL_FLIP(CWORK, GRIDC, 2, .FALSE.)
            END IF

! add LDA part of potential
            CALL FEXCF(GRIDC, LATT_CUR%OMEGA, &
                       CWORK(1, 1), CWORK(1, 2), DENCOR, CVTOT(1, 1), CVTOT(1, 2), &
                       E%CVZERO, EXC, E%XCENC, XCSIF, .TRUE.)
!gk COH
! add Coulomb hole
            CALL COHSM1_RGRID(2, CWORK(1, 1), CVTOT(1, 1), DENCOR, GRIDC, LATT_CUR%OMEGA, .TRUE.)
!gK COHend
! we have now the potential for up and down stored in CVTOT(:,1) and CVTOT(:,2)

! get the proper direction vx = v0 + hat m delta v
            CALL MAG_DIRECTION(CHTOT(1, 1), CVTOT(1, 1), GRIDC, WDES%NCDIJ)
        ELSEIF (WDES%LNONCOLLINEAR) THEN
            IF (ISGGA()) THEN
                ! GGA potential
                CALL FEXCGS(4, GRIDC, LATT_CUR, XCENCG, EXCG, CVZERG, TMPSIF, &
                            CHTOT, CVTOT, DENCOR)
            END IF

! FEXCF requires (up,down) density instead of (rho,mag)
            CALL MAG_DENSITY(CHTOT, CWORK, GRIDC, WDES%NCDIJ)
! quick hack to do LDA+U instead of LSDA+U
            IF (L_NO_LSDA_GLOBAL()) CWORK(:, 2) = 0
! end of hack
! add LDA part of potential
            CALL FEXCF(GRIDC, LATT_CUR%OMEGA, &
                       CWORK(1, 1), CWORK(1, 2), DENCOR, CVTOT(1, 1), CVTOT(1, 2), &
                       E%CVZERO, EXC, E%XCENC, XCSIF, .TRUE.)
!gk COH
! add Coulomb hole
            CALL COHSM1_RGRID(2, CWORK(1, 1), CVTOT(1, 1), DENCOR, GRIDC, LATT_CUR%OMEGA, .TRUE.)
!gK COHend
! we have now the potential for up and down stored in CVTOT(:,1) and CVTOT(:,2)
! get the proper direction vx = v0 + hat m delta v

            CALL MAG_DIRECTION(CHTOT(1, 1), CVTOT(1, 1), GRIDC, WDES%NCDIJ)
        ELSE
            IF (ISGGA()) THEN
                ! gradient corrections to LDA
                CALL FEXCG(GRIDC, LATT_CUR, XCENCG, EXCG, CVZERG, TMPSIF, &
                           CHTOT, CVTOT, DENCOR)
            END IF

! LDA part of potential
            CALL FEXCP(GRIDC, LATT_CUR%OMEGA, &
                       CHTOT, DENCOR, CVTOT, CWORK, E%CVZERO, EXC, E%XCENC, XCSIF, .TRUE.)
!gk COH
! add Coulomb hole
            CALL COHSM1_RGRID(1, CHTOT(1, 1), CVTOT(1, 1), DENCOR, GRIDC, LATT_CUR%OMEGA, .TRUE.)
!gK COHend
        END IF

#ifdef libbeef
        IF (LBEEFCALCBASIS) THEN
        IF (BEEFCOUNTER .GT. 0) THEN
            IF (BEEFCOUNTER .LE. 31) THEN
                beefxc(BEEFCOUNTER) = EXCG
                BEEFCOUNTER = BEEFCOUNTER + 1
            ELSE
                beefxc(32) = EXCG + beefxc(31)
#if defined(MPI) || defined(MPI_CHAIN)
                IF (COMM%NODE_ME == COMM%IONODE) THEN
#endif
                    IF (.NOT. LBEEFBAS) THEN
                        CALL BEEFRANDINITDEF
                        CALL BEEFENSEMBLE(beefxc, beenergies)
                        WRITE (8, *) "BEEFens 2000 ensemble energies"
                        DO BEEFCOUNTER = 1, 2000
                            WRITE (8, "(E35.15)") beenergies(BEEFCOUNTER)
                        END DO
                    END IF

                    WRITE (8, *) "BEEF xc energy contributions"
                    DO BEEFCOUNTER = 1, 32
                        WRITE (8, *), BEEFCOUNTER, ": ", beefxc(BEEFCOUNTER)
                    END DO
#if defined(MPI) || defined(MPI_CHAIN)
                END IF
#endif
                LBEEFCALCBASIS = .FALSE.
                BEEFCOUNTER = 0
                LUSE_VDW = SAVELUSE_VDW
                CALL BEEFSETMODE(-1)
            END IF
            EXC = 0
            E%XCENC = 0
            E%EXCG = 0
            E%CVZERO = 0
            XCSIF = 0
            CVTOT = 0
            EXCG = 0
            XCENCG = 0
            CVZERG = 0
            TMPSIF = 0

            GOTO 4096
        END IF
        END IF
#endif

        XCSIF = XCSIF + TMPSIF
        E%EXCG = EXC + EXCG
        E%XCENC = E%XCENC + XCENCG
        E%CVZERO = E%CVZERO + CVZERG

        ELSE xc
        DO ISP = 1, WDES%NCDIJ
            CALL FFT3D(CHTOT(1, ISP), GRIDC, 1)
        END DO
    END IF xc
!-MM- changes to accomodate constrained moments
!-----------------------------------------------------------------------
! add constraining potential
!-----------------------------------------------------------------------
#ifndef NGXhalf
#ifndef NGZhalf
    IF (M_CONSTRAINED()) THEN

        ! recreate old Hamiltonian
        L_CONSTR = L_CONSTR_L_DIAG
        CALL M_INT(CHTOT, GRIDC, WDES)
        CALL ADPT_CONSTRAINING_POT(CVTOT, GRIDC, WDES)

        ! update subspace Hamiltonian
        L_CONSTR = L_CONSTR_L_ADD
        do ISP = 1, WDES%NCDIJ
            call FFT3D(CHTOTN(1, ISP), GRIDC, 1)
        end do
        call M_INT(CHTOTN, GRIDC, WDES)
        do ISP = 1, WDES%NCDIJ
            call FFT_RC_SCALE(CHTOTN(1, ISP), CHTOTN(1, ISP), GRIDC)
            call SETUNB_COMPAT(CHTOTN(1, ISP), GRIDC)
        end do
        CALL ADPT_CONSTRAINING_POT(CVTOT, GRIDC, WDES)

    END IF
#endif
#endif
!-MM- end of addition

!-----------------------------------------------------------------------
! calculate the total potential
!-----------------------------------------------------------------------
! add external electrostatic potential
    DIP%ECORR = 0
    DIP%E_ION_EXTERN = 0

    IF (DIP%LCOR_DIP) THEN
! get the total charge and store it in CWORK
        IF (WDES%NCDIJ > 1) THEN
            CALL MAG_DENSITY(CHTOT, CWORK, GRIDC, WDES%NCDIJ)
        ELSE
            CALL RL_ADD(CHTOT, 1.0_q, CHTOT, 0.0_q, CWORK, GRIDC)
        END IF

        CALL CDIPOL(GRIDC, LATT_CUR, P, T_INFO, &
                    CWORK, CSTRF, CVTOT(1, 1), WDES%NCDIJ, INFO%NELECT)

        CALL EXTERNAL_POT(GRIDC, LATT_CUR, CVTOT(1, 1))
    ELSE
        CALL EXTERNAL_POT(GRIDC, LATT_CUR, CVTOT(1, 1))
    END IF

    DO ISP = 1, WDES%NCDIJ
        CALL FFT_RC_SCALE(CHTOT(1, ISP), CHTOT(1, ISP), GRIDC)
        CALL SETUNB_COMPAT(CHTOT(1, ISP), GRIDC)
    END DO
!-----------------------------------------------------------------------
! FFT of the exchange-correlation potential to reciprocal space
!-----------------------------------------------------------------------
    RINPL = 1._q/GRIDC%NPLWV
    DO ISP = 1, WDES%NCDIJ
        CALL RL_ADD(CVTOT(1, ISP), RINPL, CVTOT(1, ISP), 0.0_q, CVTOT(1, ISP), GRIDC)
        CALL FFT3D(CVTOT(1, ISP), GRIDC, -1)
    END DO
!-----------------------------------------------------------------------
! add the hartree potential and the double counting corrections
!-----------------------------------------------------------------------
    CALL POTHAR(GRIDC, LATT_CUR, CHTOT, CWORK, E%DENC)
    DO I = 1, GRIDC%RC%NP
        CVTOT(I, 1) = CVTOT(I, 1) + CWORK(I, 1)
    END DO
! solvation__
!-----------------------------------------------------------------------
! add the dielectric corrections to CVTOT and the energy
!-----------------------------------------------------------------------
    CALL SOL_Vcorrection(INFO, T_INFO, LATT_CUR, P, WDES, GRIDC, CHTOT, CVTOT)
! solvation__
!-----------------------------------------------------------------------
!  add local pseudopotential potential
!-----------------------------------------------------------------------
    IF (INFO%TURBO == 0) THEN
        CALL POTION(GRIDC, P, LATT_CUR, T_INFO, CWORK, CWORK1, CSTRF, E%PSCENC)
    ELSE
        CALL POTION_PARTICLE_MESH(GRIDC, P, LATT_CUR, T_INFO, CWORK, E%PSCENC, E%TEWEN)
    END IF

    ELECTROSTATIC = 0
    NG = 1
    col: DO NC = 1, GRIDC%RC%NCOL
        N2 = GRIDC%RC%I2(NC)
        N3 = GRIDC%RC%I3(NC)
        row: DO N1 = 1, GRIDC%RC%NROW
            SETFACT1
            SETFACT

            ELECTROSTATIC = ELECTROSTATIC + MULFACT CWORK(NG, 1)*CONJG(CHTOT(NG, 1))
            NG = NG + 1
        END DO row
    END DO col
    ELECTROSTATIC = ELECTROSTATIC + E%PSCENC - E%DENC + E%TEWEN

    E%PSCENC = E%PSCENC + DIP%ECORR + DIP%E_ION_EXTERN

    DO I = 1, GRIDC%RC%NP
        CVTOT(I, 1) = CVTOT(I, 1) + CWORK(I, 1)
    END DO
! bexternal__
    IF (LBEXTERNAL()) CALL BEXT_ADDV(CVTOT, GRIDC, SIZE(CVTOT, 2))
! bexternal__
    CALL POT_FLIP(CVTOT, GRIDC, WDES%NCDIJ)
!=======================================================================
! if overlap is used :
! copy CVTOT to SV and set contribution of unbalanced lattice-vectors
! to zero,  then  FFT of SV and CVTOT to real space
!=======================================================================

    DO ISP = 1, WDES%NCDIJ
        CALL SETUNB_COMPAT(CVTOT(1, ISP), GRIDC)
        CALL CP_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, CVTOT(1, ISP), CWORK1)
        CALL SETUNB(CWORK1, GRID_SOFT)
        CALL FFT3D(CWORK1, GRID_SOFT, 1)
        CALL RL_ADD(CWORK1, 1.0_q, CWORK1, 0.0_q, SV(1, ISP), GRID_SOFT)

!  final result is only correct for first in-band-group
! (i.e. proc with nodeid 1 in COMM_INTER)
!  copy to other in-band-groups using COMM_INTER
! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)
#ifdef realmode
        CALLMPI(M_bcast_d(COMM_INTER, SV(1, ISP), GRID%RL%NP))
#else
        CALLMPI(M_bcast_z(COMM_INTER, SV(1, ISP), GRID%RL%NP))
#endif
        CALL FFT3D(CVTOT(1, ISP), GRIDC, 1)
    END DO

    DEALLOCATE (CWORK1, CWORK)
    RETURN
END SUBROUTINE POTLOK_SASC_IN
