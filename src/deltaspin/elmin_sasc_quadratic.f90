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
