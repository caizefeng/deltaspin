#include "symbol.inc"
      MODULE openmp
      ! overall threading strategy

      INTEGER, SAVE :: omp_nthreads_hi=1

      INTEGER, SAVE :: omp_nthreads_lo=1
 
      INTEGER, SAVE :: omp_nthreads   =1

      ! specific routines

      INTEGER, SAVE :: omp_nthreads_hamiltmu=1

      INTEGER, SAVE :: omp_nthreads_alltoall=1

      INTEGER, SAVE :: omp_nthreads_nonlr_rspace=1

      LOGICAL, SAVE :: omp_nonlr_planewise=.TRUE.

      LOGICAL, SAVE :: omp_dfftw_init_threads=.TRUE.

      CONTAINS

!
      SUBROUTINE INIT_OMP(IO)
      USE prec
      USE base
      TYPE (in_struct) IO
      ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LDUM
      CHARACTER(1) CHARAC
      RETURN
      END SUBROUTINE INIT_OMP

      END MODULE openmp
!! @tableofcontents
!! @section general All changes
!! All datatypes, subroutines, and functions that have been changed
!! to enable parallelization under OpenMP.
!!
!! \li ::apply_gfac
!! \li ::apply_gfac_2_encuts
!! \li ::apply_gfac_der
!! \li ::apply_gfac_encuts
!! \li ::apply_gfac_exchange
!! \li ::apply_gfac_exchange_2
!! \li augfast::calc_dllmm_trans
!! \li augfast::depsum_two_bands_rholm_trace
!! \li augfast::depsum_vector
!! \li augfast::setup_aug_des
!! \li broyden::brmix
!! \li chi_base::add_responsefunction_cache
!! \li chi_base::add_responsefunction_int
!! \li chi_base::add_xi
!! \li chi_base::add_xi_spectral
!! \li chi_base::allocate_responsefun_cache
!! \li chi_base::clean_responsefunction_cache
!! \li chi_base::clean_responsefunction_int
!! \li chi_base::deallocate_responsefun
!! \li chi_base::responsefunction
!! \li chi_base::screened_two_electron_cached
!! \li choleski::orthch
!! \li ::crrexp_mul_work_gadd
!! \li ::crexp_mul_wave
!! \li ::complex_add
!! \li ::cmplx_cmplx_cmplx_mul
!! \li ::cmplx_cmplx_real_mul
!! \li ::cmplx_real_real_mul
!! \li ::dllmm_kernel
!! \li ::eccp_nl
!! \li ::eccp_nl_fock
!! \li ::exchange_gfac
!! \li ::fexcg_
!! \li ::fftbas
!! \li ::fftbas_plan_mpi
!! \li ::fftbrc
!! \li ::fftext_mpi
!! \li ::ffthamil
!! \li ::fftmakeplan
!! \li ::fftmakeplan_mpi
!! \li ::fftwav
!! \li ::fftwav_mpi
!! \li fock::allocate_fock_handle
!! \li fock::deallocate_fock_handle
!! \li fock::fock_acc
!! \li fock::fock_charge_mu
!! \li fock::fock_force
!! \li fock::fock_handle
!! \li fock::xc_fock_reader
!! \li fock_dbl::fock_all_dblbuf
!! \li force::forhar
!! \li force::forloc
!! \li ::ggaall_grid
!! \li greens_real_space::transpose_g_r_response
!! \li greens_real_space::transpose_g_proj_response
!! \li greens_real_space::transpose_r_g
!! \li greens_real_space::transpose_r_proj
!! \li greens_real_space_k::calculate_response_super
!! \li greens_real_space_k::fft_g_super
!! \li hamil::eccp
!! \li hamil::eccp_tau
!! \li hamil::eccp_vec
!! \li hamil::hamiltmu
!! \li hamil::hamiltmu_c
!! \li hamil::hamiltmu_tau
!! \li hamil::hamiltmu_vec
!! \li hamil::setup_precond
!! \li hamil::simple_precond
!! \li hamil_lrf::hamiltmu_commutator
!! \li hamil_lrf::hamiltmu_lrf
!! \li hamil_lrf::lrf_commutator
!! \li hamil_lrf::lrf_hamil
!! \li ::kinhamil
!! \li ::kinhamil_c
!! \li ::kinhamil_tau
!! \li ::kinhamil_vec
!! \li main_mpi::init_mpi
!! \li main_mpi::wrt_distr
!! \li ::map_backward
!! \li ::map_forward
!! \li ::map_gather
!! \li ::map_scatter
!! \li mpimy::m_init
!! \li nonl::phase
!! \li nonl::proj1
!! \li nonl::projxyz
!! \li nonl::projxyz_wa
!! \li nonl::spher
!! \li nonl::strenl
!! \li nonlr::fornlr
!! \li nonlr::nonlr_alloc
!! \li nonlr::nonlr_alloc_crrexp
!! \li nonlr::nonlr_dealloc
!! \li nonlr::nonlr_set_single_ion
!! \li nonlr::phaser
!! \li nonlr::phaserr
!! \li nonlr::phaser_hf
!! \li nonlr::real_optlay
!! \li nonlr::real_optlay_grid
!! \li nonlr::rnlpr
!! \li nonlr::rpro1
!! \li nonlr::rpro1_hf
!! \li nonlr::rpromu
!! \li nonlr::rpromu_hf
!! \li nonlr::rspher_all
!! \li nonlr_struct_def::nonlr_struct
!! \li nonl_high::w1_projall
!! \li openmp::init_omp
!! \li ::overl
!! \li ::overl1
!! \li ::overl1_c
!! \li ::overl_fock
!! \li pawfock::coloumb_4term
!! \li pawfock::coloumb_4term_ps
!! \li pawfock_inter::ntyp_slater
!! \li pawfock_inter::s
!! \li pawm::set_dd_paw
!! \li ::pw_charge
!! \li ::pw_charge_trace
!! \li ::pw_norm_with_metric
!! \li ::racc0
!! \li ::racc0mu
!! \li ::racc0mu_hf
!! \li ::racc0_hf
!! \li radial::rad_lda_xc
!! \li ::real_add
!! \li ::real_cmplx_cmplx_mul
!! \li ::real_real_cmplx_mul
!! \li ::real_real_real_mul
!! \li ::rholm_kernel
!! \li ::rholm_kernel_aux
!! \li ::rholm_kernel_dgemm
!! \li ::rholm_one_center_kernel
!! \li rmm_diis::eddrmm
!! \li rmm_diis_lr::linear_response_diis
!! \li rot::edwav
!! \li rs_greensfunc_kernel::set_rsgf_all
!! \li rs_greensfunc_kernel::rs_coulomb_green_func
!! \li rs_greensfunc_kernel::rs_greensfunc_type
!! \li screened_2e::determine_slot
!! \li screened_2e::determine_slot_inter
!! \li screened_2e::determine_slot_inter_weight
!! \li screened_2e::integrate_w_2e_simple
!! \li screened_2e::integrate_w_2e_spectral
!! \li screened_2e::integrate_w_2e_spectral_imag
!! \li screened_2e::qp_shift
!! \li screened_2e::qp_shift_pade
!! \li ::setdij_
!! \li ::start_profiling
!! \li ::stop_profiling
!! \li subrot::eddiag
!! \li ::truncate_high_frequency_one
!! \li twoelectron4o::apply_phase
!! \li us::depsum
!! \li us::fordep
!! \li ::vhamil
!! \li ::vhamil_trace
!! \li wave::delwav_omp
!! \li wave::newwav_omp
!! \li wave_cacher::allocate_cacher
!! \li wave_cacher::deallocate_cacher
!! \li wave_cacher::find_new_cacher
!! \li wave_cacher::remove_cacher
!! \li wave_cacher::store_cacher
!! \li wave_cacher::store_gw_acc
!! \li wave_high::cnorma
!! \li wave_high::w1_copy
!! \li wave_high::w1_daxpy
!! \li wave_high::w1_dot
!! \li wave_high::w1_dscal
!! \li wave_high::w1_gaxpy
!! \li ::work_mul_crexp
!! \li xi::calculate_xi
!!
!! @section simple Simple loop parallelism
!! In the following cases only basic loop parallelism under OpenMP was added.
!! Sometimes in conjunction with a reduction operation, and sometimes
!! conditional upon a certain loop size.
!! These loops often run over all points in a real or reciprocal space grid,
!! or over all basis vectors in an orbital.
!!
!! \li ::apply_gfac
!! \li ::apply_gfac_2_encuts
!! \li ::apply_gfac_der
!! \li ::apply_gfac_encuts
!! \li ::apply_gfac_exchange
!! \li ::apply_gfac_exchange_2
!! \li augfast::depsum_vector
!! \li broyden::brmix
!! \li ::crrexp_mul_work_gadd
!! \li ::crexp_mul_wave
!! \li ::complex_add
!! \li ::dllmm_kernel
!! \li ::cmplx_cmplx_cmplx_mul
!! \li ::cmplx_cmplx_real_mul
!! \li ::cmplx_real_real_mul
!! \li ::eccp_nl
!! \li ::eccp_nl_fock
!! \li ::exchange_gfac
!! \li ::fexcg_
!! \li ::ffthamil
!! \li force::forhar
!! \li force::forloc
!! \li ::ggaall_grid
!! \li ::kinhamil
!! \li nonl::phase
!! \li nonl::strenl
!! \li pawfock::coloumb_4term
!! \li pawfock::coloumb_4term_ps
!! \li ::pw_charge
!! \li ::pw_charge_trace
!! \li ::pw_norm_with_metric
!! \li radial::rad_lda_xc
!! \li ::real_add
!! \li ::real_cmplx_cmplx_mul
!! \li ::real_real_cmplx_mul
!! \li ::real_real_real_mul
!! \li ::rholm_kernel
!! \li ::rholm_kernel_aux
!! \li ::rholm_kernel_dgemm
!! \li ::rholm_one_center_kernel
!! \li rs_greensfunc_kernel::rs_coulomb_green_func
!! \li ::truncate_high_frequency_one
!! \li twoelectron4o::apply_phase
!! \li ::vhamil
!! \li ::vhamil_trace
!! \li ::work_mul_crexp
