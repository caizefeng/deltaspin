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
    complex(q) CHTOT_last_step(GRIDC%MPLWV, WDES%NCDIJ) ! charge-density in real / reciprocal space
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
    real(q), allocatable :: dnu_last_step(:, :)
    real(q), allocatable :: spin_plus(:, :)
    real(q), allocatable :: delta_spin(:, :), delta_spin_old(:, :)
    real(q), allocatable :: search(:, :), search_old(:, :)
    real(q), allocatable :: target_spin_mask(:, :), spin_mask(:, :), spin_plus_mask(:, :)
    real(q), allocatable :: spin_nu_gradient(:, :, :, :), spin_nu_gradient_diag(:, :)
    real(q), allocatable :: spin_change(:, :), nu_change(:, :)
    real(q), allocatable :: max_gradient(:)
    integer, allocatable :: max_gradient_index(:, :)

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
    integer :: i, j, k, l
    integer :: N_RESERVE, NELM_RESERVE, IALGO_RESERVE, IHARMONIC_RESERVE, NELMDL_RESERVE, NELMIN_RESERVE
    logical :: LCHCOS_RESERVE, LONESW_RESERVE, LONESW_AUTO_RESERVE, LDAVID_RESERVE, &
    & LRMM_RESERVE, LORTHO_RESERVE, LCDIAG_RESERVE, LPDIAG_RESERVE, LPRECONDH_RESERVE, &
    & LEXACT_DIAG_RESERVE

    allocate (nu(3, T_INFO%NIONS))
    allocate (dnu, dnu_last_step, mold=nu)
    allocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old, mold=nu)
    allocate (search, search_old, mold=nu)
    allocate (target_spin_mask, spin_mask, spin_plus_mask, mold=nu)
    allocate (spin_nu_gradient(3, T_INFO%NIONS, 3, T_INFO%NIONS))
    allocate (spin_nu_gradient_diag(3, T_INFO%NIONS))
    allocate (spin_change, nu_change, mold=nu)
    allocate (max_gradient_index(2, T_INFO%NTYP), max_gradient(T_INFO%NTYP))

    target_spin = M_CONSTR
    dnu = 0
    dnu_last_step = 0
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

    lambda: do i_step = 1, num_step

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

            spin_change = MW - spin
            nu_change = L_CONSTR_L_ADD - dnu_last_step
            where (CONSTRL == 0) 
                spin_change = 0.0
                nu_change = 1.0
            end where
            do i = 1, 3
                do j = 1, T_INFO%NIONS
                    do k = 1, 3
                        do l = 1, T_INFO%NIONS
                            spin_nu_gradient(i, j, k, l) = spin_change(i, j) / nu_change(k, l)
                        end do
                    end do
                end do
            end do
            do i = 1, 3
                do j = 1, T_INFO%NIONS
                    spin_nu_gradient_diag(i, j) = spin_nu_gradient(i, j, i, j)
                end do
            end do

            i = 1
            do j = 1, T_INFO%NTYP
                max_gradient_index(:, j) = maxloc(abs(spin_nu_gradient_diag(:, i:(i+T_INFO%NITYP(j)-1))))
                max_gradient(j) = maxval(abs(spin_nu_gradient_diag(:, i:(i+T_INFO%NITYP(j)-1))))
                i = i + T_INFO%NITYP(j)
            end do

            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, *) "diagonal gradient:"
            if (IO%IU0 >= 0) write (IO%IU0, *) spin_nu_gradient_diag
            if (IO%IU0 >= 0) write (IO%IU0, *) "maximum gradient appears at: "
            do j = 1, T_INFO%NTYP
                if (IO%IU0 >= 0) write (IO%IU0, 15589) max_gradient_index(1, j), max_gradient_index(2, j)
            end do
            if (IO%IU0 >= 0) write (IO%IU0, *)
15589       format(" ( ", i0, " , ", i0, " )"\)
            if (IO%IU0 >= 0) write (IO%IU0, *) "maximum gradient: " 
            do j = 1, T_INFO%NTYP
                if (IO%IU0 >= 0) write (IO%IU0, '(1x, e14.8, 1x\)') max_gradient(j)
            end do
            if (IO%IU0 >= 0) write (IO%IU0, *)
            io_end

            do j = 1, T_INFO%NTYP
                if (i_step > CONSTR_NUM_STEP_MIN .and. CONV_BOUND_GRAD(j) > 0 .and. max_gradient(j) < CONV_BOUND_GRAD(j)) then
                    io_begin
                    if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a, i0, a)') "Reach limitation of current step ( maximum gradient < ", CONV_BOUND_GRAD(j), " in atom type ", j, " ), exit."
                    io_end
                    CHTOT = CHTOT_last_step
                    L_CONSTR = L_CONSTR_L_DIAG + dnu_last_step
                    exit lambda
                end if
            end do

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
            exit lambda
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

        CHTOT_last_step = CHTOT
        dnu_last_step = dnu
        dnu = dnu + alpha_trial*search
        L_CONSTR_L_ADD = dnu
        where (CONSTRL == 0) dnu_last_step = 0.0
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

! restrict adapted step
        boundary = abs(alpha_opt*maxval(abs(search)))
        io_begin
        if (IO%IU0 >= 0) write (IO%IU0, *) "alpha_opt before restrict = ", alpha_opt
        if (IO%IU0 >= 0) write (IO%IU0, *) "boundary before = ", boundary
        io_end

        if (CONV_BOUND > 0 .and. boundary < CONV_BOUND) then
            io_begin
            if (IO%IU0 >= 0) write (IO%IU0, '(a, es9.3, a)') "Reach limitation of current step ( boundary < ", CONV_BOUND, " ), exit."
            io_end
            CHTOT = CHTOT_last_step
            L_CONSTR = L_CONSTR_L_DIAG + dnu_last_step
            exit lambda
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

    end do lambda

    deallocate (nu, dnu)
    deallocate (target_spin, spin, spin_plus, delta_spin, delta_spin_old)
    deallocate (search, search_old)
    deallocate (spin_nu_gradient, spin_nu_gradient_diag)
    deallocate (spin_change, nu_change)
    deallocate (max_gradient_index, max_gradient)

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
