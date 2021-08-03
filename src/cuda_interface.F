module cuda_interface
    use iso_c_binding
    implicit none

!TODO(ca): this list lists open to do's 
! * remove CUDA/.cu functions that are bound in the interface below and which
! do nothing more than forward to the corresponding cuda function
! * check what fortran.h is doing and if needed
! * define intents for all parameters?
interface 

! ========================================
! CUDA/cuda_main.cu
! ========================================
    !void cuda_init(int *nstreams, int *nsim)
    subroutine cuda_init(nstreams, nsim) bind(c,name="cuda_init_C")
        import
        integer(c_int)      :: nstreams
        integer(c_int)      :: nsim
    end subroutine cuda_init

    !void cuda_destroy(void)
    subroutine cuda_destroy() bind(c,name="cuda_destroy_C")
    end subroutine cuda_destroy

    !void cuda_mpi_init_C(int *CudaDevice)
    subroutine cuda_mpi_init(cudadevice) bind(c,name="cuda_mpi_init_C")
        import
        integer(c_int)     :: cudadevice
    end subroutine cuda_mpi_init

    !void cuda_device_reset_C(void)
    subroutine cuda_device_reset() bind(c,name="cuda_device_reset_C")
    end subroutine cuda_device_reset

    !void cuda_devicesynchronize(char *msg)
    subroutine cuda_devicesynchronize(msg) bind(c,name="cuda_devicesynchronize_C")
        import
        character(c_char)   :: msg(*)
    end subroutine cuda_devicesynchronize

    !void threadsynchronize(void)
    subroutine threadsynchronize() bind(c,name="threadsynchronize_C")
    end subroutine threadsynchronize

    !void cuda_streamsynchronize(int *sid)
    subroutine cuda_streamsynchronize(sid) bind(c,name="cuda_streamsynchronize_C")
        import
        integer(c_int)      :: sid
    end subroutine cuda_streamsynchronize

    !void cuda_all_stream_synchronize_C(void)
    subroutine cuda_all_stream_synchronize() &
        bind(c,name="cuda_all_stream_synchronize_C")
    end subroutine cuda_all_stream_synchronize

! ========================================
! CUDA/cuda_blas.cu
! ========================================

    !void cublas_init(void)
    subroutine cublas_init() bind(c,name="cublas_init_C")
    end subroutine cublas_init

    !void cublas_destroy(void)
    subroutine cublas_destroy() bind(c,name="cublas_destroy_C")
    end subroutine cublas_destroy

    !void cublas_set_vector(int *n, int *elemSize, void *x, int *incx, void **y,int *incy)
    subroutine cublas_set_vector(n,elemSize,x,incx,y,incy) &
        bind(c,name="cublas_set_vector_C")
        import
        integer(c_int)     :: n
        integer(c_int)     :: elemSize
        type(c_ptr), value :: x
        integer(c_int)     :: incx
        integer(c_intptr_t)  :: y
        integer(c_int)     :: incy
    end subroutine cublas_set_vector

    !void cublas_get_vector(int *n, int *elemSize, void **x, int *incx, void *y, int *incy)
    subroutine cublas_get_vector(n,elemSize,x,incx,y,incy) &
        bind(c,name="cublas_get_vector_C")
        import
        integer(c_int)      :: n
        integer(c_int)      :: elemSize
        integer(c_intptr_t) :: x
        integer(c_int)      :: incx
        type(c_ptr), value  :: y
        integer(c_int)      :: incy
    end subroutine cublas_get_vector

    !void cublas_set_matrix(int *rows, int *cols, int *elemSize, 
    !         void *A, int *lda,void **B, int *ldb)
    subroutine cublas_set_matrix(rows,cols,elemSize,A,lda,B,ldb) &
        bind(c,name="cublas_set_matrix_C")
        import
        integer(c_int)     :: rows
        integer(c_int)     :: cols
        integer(c_int)     :: elemSize
        type(c_ptr), value :: A
        integer(c_int)     :: lda
        integer(c_intptr_t)  :: B
        integer(c_int)     :: ldb
    end subroutine cublas_set_matrix

    !void cublas_set_matrix_async(int *sid, int *rows, int *cols, int *elemSize,
    !void *A, int *lda, void **B, int *ldb)
    subroutine cublas_set_matrix_async(sid, rows,cols,elemSize,A,lda,B,ldb) &
        bind(c,name="cublas_set_matrix_async_C")
        import
        integer(c_int)     :: sid
        integer(c_int)     :: rows
        integer(c_int)     :: cols
        integer(c_int)     :: elemSize
        type(c_ptr), value :: A
        integer(c_int)     :: lda
        integer(c_intptr_t)  :: B
        integer(c_int)     :: ldb
    end subroutine cublas_set_matrix_async

    !void cublas_get_matrix(int *rows, int *cols, int *elemSize, void **A, 
                ! int *lda,void *B, int *ldb)
    subroutine cublas_get_matrix(rows,cols,elemSize,A,lda,B,ldb) &
        bind(c,name="cublas_get_matrix_C")
        import
        integer(c_int)     :: rows
        integer(c_int)     :: cols
        integer(c_int)     :: elemSize
        integer(c_intptr_t)  :: A
        integer(c_int)     :: lda
        type(c_ptr), value :: B
        integer(c_int)     :: ldb
    end subroutine cublas_get_matrix

    subroutine cublas_get_matrix_async(sid,rows,cols,elemSize,A,lda,B,ldb) &
        bind(c,name="cublas_get_matrix_async_C")
        import
        integer(c_int)     :: sid
        integer(c_int)     :: rows
        integer(c_int)     :: cols
        integer(c_int)     :: elemSize
        integer(c_intptr_t)  :: A
        integer(c_int)     :: lda
        type(c_ptr), value :: B
        integer(c_int)     :: ldb
    end subroutine cublas_get_matrix_async

    !void cuda_zsetmatrix(int *sid, int *rows, int *cols, 
            !cuDoubleComplex **A, int *lda,
            !cuDoubleComplex **B, int *ldb)
    subroutine cuda_zsetmatrix(sid, rows, cols, A, lda, B, ldb) &
        bind(c,name="cuda_zsetmatrix_C") 
        import
        integer(c_int)      :: sid
        integer(c_int)      :: rows
        integer(c_int)      :: cols
        integer(c_intptr_t) :: A
        integer(c_int)      :: lda
        integer(c_intptr_t) :: B
        integer(c_int)      :: ldb
    end subroutine cuda_zsetmatrix

    !void cublas_ddot(int *n, double **x, int *incx, double **y, 
    ! int *incy, double *result)
    subroutine cublas_ddot(n, x, incx, y, incy, res) bind(c,name="cublas_ddot_C")
        import
        integer(c_int)    :: n
        integer(c_intptr_t) :: x
        integer(c_int)    :: incx
        integer(c_intptr_t) :: y
        integer(c_int)    :: incy
        real(c_double)      :: res
    end subroutine cublas_ddot

    !void cublas_zdotc(int *n, cuDoubleComplex **x, int *incx, 
    !  cuDoubleComplex **y, int *incy, cuDoubleComplex *result)
    subroutine cublas_zdotc(n, x, incx, y, incy, res) bind(c,name="cublas_zdotc_C")
        import
        integer(c_int)    :: n
        integer(c_intptr_t) :: x
        integer(c_int)    :: incx
        integer(c_intptr_t) :: y
        integer(c_int)    :: incy
        complex(C_DOUBLE_COMPLEX) :: res
    end subroutine cublas_zdotc

    !void cuda_zsub(int *n, cuDoubleComplex **c, cuDoubleComplex **a, 
        !cuDoubleComplex **b)
    subroutine cuda_zsub(n, c, a, b) bind(c,name="cuda_zsub_C")
        import
        integer(c_int)    :: n
        integer(c_intptr_t) :: c
        integer(c_intptr_t) :: a
        integer(c_intptr_t) :: b
    end subroutine cuda_zsub

    !void cuda_ddot(int *n, double **x, double **y, double **result)
    subroutine cuda_ddot(n, x, y, res) bind(c,name="cuda_ddot_C")
        import
        integer(c_int)    :: n
        integer(c_intptr_t) :: x
        integer(c_intptr_t) :: y
        integer(c_intptr_t) :: res
    end subroutine cuda_ddot

    !void cuda_zdotcblock_C(int *nblock, int *n, devptr_t *xptrs, 
    ! devptr_t *yptrs, cuDoubleComplex *result)
    subroutine cuda_zdotcblock(nblock, n, xptrs, yptrs, res) &
        bind(c,name="cuda_zdotcblock_C")
        import
        integer(c_int)    :: nblock
        integer(c_int)    :: n
        integer(c_intptr_t) :: xptrs(*)
        integer(c_intptr_t) :: yptrs(*)
        complex(C_DOUBLE_COMPLEX) :: res(*)
    end subroutine cuda_zdotcblock

    !void cuda_zdscalblock_C(int *nblock, int *n, double *alpha, devptr_t *xptrs)
    subroutine cuda_zdscalblock(nblock, n, alpha, xptrs) &
        bind(c,name="cuda_zdscalblock_C")
        import
        integer(c_int)    :: nblock
        integer(c_int)    :: n
        real(c_double) :: alpha(*)
        integer(c_intptr_t) :: xptrs(*)
    end subroutine cuda_zdscalblock

    !void cublas_zgemv_C(char *trans, int *m, int *n, cuDoubleComplex *alpha, 
    !cuDoubleComplex **A, int *lda,
    ! cuDoubleComplex **x, int *incx, cuDoubleComplex *beta, 
    !cuDoubleComplex **y, int *incy)
    subroutine cublas_zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) &
        bind(c,name="cublas_zgemv_C")
        import
        character(c_char) :: trans
        integer(c_int)    :: m
        integer(c_int)    :: n
        complex(C_DOUBLE_COMPLEX) :: alpha
        integer(c_intptr_t)       :: A
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: x
        integer(c_int)    :: incx
        complex(C_DOUBLE_COMPLEX) :: beta
        integer(c_intptr_t) y
        integer(c_int)    :: incy
    end subroutine cublas_zgemv

    !void cuda_gemmsmallmn_C(const int *StreamIdx, const char *transa, 
    ! const char *transb,
    ! const int *m_p, const int *n_p, const int *k_p, 
    ! const double *alpha, const devptr_t* A_p,
    ! const int *shiftA, const int *lda, const devptr_t* B_p, 
    ! const int *shiftB, const int *ldb,
    ! const double *beta, const devptr_t *C_p, const int *shiftC, const int *ldc)
    subroutine cuda_gemmsmallmn(sid, transa, transb, m, n, k, &
        alpha, devPtrA, shiftA, lda, &
        devPtrB, shiftB, ldb, beta, devPtrC, shiftC, ldc) &
        bind(c,name="cuda_gemmsmallmn_C")
        import
        integer(c_int)    :: sid
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        real(c_double)    :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: shiftA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: shiftB
        integer(c_int)    :: ldb
        real(c_double)    :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: shiftC
        integer(c_int)    :: ldc
    end subroutine cuda_gemmsmallmn

! ========================================
! CUDA/cuda_fft.cu
! ========================================

    !void cufft_init(int *nplans, int *nx, int *ny, int *nz)
    subroutine cufft_init(nplans, nx, ny, nz) bind(c,name="cufft_init_C")
        import
        integer(c_int)     :: nplans
        integer(c_int)     :: nx
        integer(c_int)     :: ny
        integer(c_int)     :: nz
    end subroutine cufft_init

    !void cufft_destroy(void)
    subroutine cufft_destroy() bind(c,name="cufft_destroy_C")
    end subroutine cufft_destroy

    !void getbatchedfftplan(cufftHandle **plan, int *rank, int *n, int *howmany,
    ! int *inembed, int *istride, int *idist, int *onembed, int *ostride,
    ! int *odist, cufftType *type)
    subroutine getbatchedfftplan(plan, rank, n, howmany, inembed, istride, idist, onembed, ostride, odist, thetype) &
        bind(c,name="getbatchedfftplan_C")
        import
        integer(c_intptr_t) :: plan
        integer(c_int)    :: rank 
        integer(c_int)    :: n 
        integer(c_int)    :: howmany 
        integer(c_int)    :: inembed 
        integer(c_int)    :: istride 
        integer(c_int)    :: idist 
        integer(c_int)    :: onembed 
        integer(c_int)    :: ostride 
        integer(c_int)    :: odist 
        integer(c_int)    :: thetype
    end subroutine getbatchedfftplan

    !void cufft_exec_plan_c2c(cufftHandle **plan, cuDoubleComplex **in, cuDoubleComplex **out,
     !int *direction)
    subroutine cufft_exec_plan_c2c(plan, input, output, direction) bind(c,name="cufft_exec_plan_c2c_C")
        import
        integer(c_intptr_t) :: plan
        integer(c_intptr_t) :: input
        integer(c_intptr_t) :: output
        integer(c_int)      :: direction
    end subroutine cufft_exec_plan_c2c

    !void cufft_exec_plan_r2c(cufftHandle **plan, double **in, cuDoubleComplex **out,
    subroutine cufft_exec_plan_r2c(plan, input, output) bind(c,name="cufft_exec_plan_r2c_C")
        import
        integer(c_intptr_t) :: plan
        integer(c_intptr_t) :: input
        integer(c_intptr_t) :: output
    end subroutine cufft_exec_plan_r2c

    !void cufft_exec_plan_c2r(cufftHandle **plan, cuDoubleComplex **in, double **out,
    subroutine cufft_exec_plan_c2r(plan, input, output) bind(c,name="cufft_exec_plan_c2r_C")
        import
        integer(c_intptr_t) :: plan
        integer(c_intptr_t) :: input
        integer(c_intptr_t) :: output
    end subroutine cufft_exec_plan_c2r

    !void cufft_execz2z(const int *sid, const int *n1, const int *n2, const int *n3,
    ! const devptr_t *idevPtr, const int *ishift, devptr_t *odevPtr, const int *oshift,
    ! const int *direction)
    subroutine cufft_execz2z(sid, n1, n2, n3, idevPtr, ishift, odevPtr, oshift, direction) &
        bind(c,name="cufft_execz2z_C")
        import
        integer(c_int)    :: sid
        integer(c_int)    :: n1
        integer(c_int)    :: n2
        integer(c_int)    :: n3
        integer(c_intptr_t) :: idevPtr
        integer(c_int)    :: ishift
        integer(c_intptr_t) :: odevPtr
        integer(c_int)    :: oshift
        integer(c_int)    :: direction
    end subroutine cufft_execz2z

    ! void cuda_fftwav(int *sid, int *n, cuDoubleComplex **cr, 
            !cuDoubleComplex **c, int **nindpw,
            !double **fftsca, int *real2cplx)
    subroutine cuda_fftwav(sid, n, cr, c, ninpdw, fftsca, real2cplx) bind(c,name="cuda_fftwav_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: n
        integer(c_intptr_t) :: cr
        integer(c_intptr_t) :: c
        integer(c_intptr_t) :: ninpdw
        integer(c_intptr_t) :: fftsca
        integer(c_int)      :: real2cplx
    end subroutine cuda_fftwav

    !void cuda_fftext(int *sid, int *n, cuDoubleComplex **c, 
        !cuDoubleComplex **cr, int **nindpw,
        !double **fftsca, int *ladd, int *real2cplx)
    subroutine cuda_fftext(sid, n, c, cr, nindpw, fftsca, ladd, real2cplx) bind(c,name="cuda_fftext_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: n
        integer(c_intptr_t) :: c
        integer(c_intptr_t) :: cr
        integer(c_intptr_t) :: nindpw
        integer(c_intptr_t) :: fftsca
        integer(c_int)      :: ladd
        integer(c_int)      :: real2cplx
    end subroutine cuda_fftext

! ========================================
! CUDA/cufft_int.cu
! ========================================

    !extern "C" void fftwav_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal)
    subroutine fftwav_cu(npl, ind, cr, c, grid, lreal) bind(c,name="fftwav_cu_C")
        import
        integer(c_int)     :: npl
        integer(c_int)     :: ind(*)
        complex(C_DOUBLE_COMPLEX) :: cr(*)
        complex(C_DOUBLE_COMPLEX) :: c(*)
        integer(c_int)     :: grid(*)
        integer(c_int)     :: lreal
    end subroutine fftwav_cu 

    !void fftext_cu_C(int *npl, int *ind, cuDoubleComplex *CR, cuDoubleComplex *C, int *grid, int* lreal, int* ladd)
    subroutine fftext_cu(npl, ind, cr, c, grid, lreal, ladd) &
        bind(c,name="fftext_cu_C")
        import
        integer(c_int)     :: npl
        integer(c_int)     :: ind(*)
        complex(C_DOUBLE_COMPLEX) :: cr(*)
        complex(C_DOUBLE_COMPLEX) :: c(*)
        integer(c_int)     :: grid(*)
        integer(c_int)     :: lreal
        integer(c_int)     :: ladd
    end subroutine fftext_cu 

    !extern "C" void fft_3d_c2c_C(int *nx, int *ny, int *nz, 
    !                        cuDoubleComplex *a_h, cuDoubleComplex *b_h,
    !                        int *DIR)
    subroutine fft_3d_c2c(nx, ny, nz, a_h, b_h, dir) bind(c,name="fft_3d_c2c_C")
        import
        integer(c_int)    :: nx
        integer(c_int)    :: ny
        integer(c_int)    :: nz
        complex(C_DOUBLE_COMPLEX) :: a_h(*)
        complex(C_DOUBLE_COMPLEX) :: b_h(*)
        integer(c_int)    :: dir
    end subroutine fft_3d_c2c

! ========================================
! CUDA/fock.cu
! ========================================
    !void setup_context_cu_C(
    !              global_desc** glob,     //!< global parameters to populate
    !              double* trans,          //!< rotation matrix 
    !              double* fermi_weights,  //!< array of weights by k-point and band
    !              double* kpoint_weights, //!< weights of each k-point
    !              int* ntype,             //!< number of species
    !              int* nion,              //!< number of ions of each species
    !              int* nband,             //!< number of bands
    !              int* nkpoint_full,      //!< number of k-points
    !              int* nkpoint_red,       //!< number of symmetry-unique k-points
    !              int* equiv_in,          //!< map of symmetry relation betweek k-points
    !              int* lmdim,             //!< another size
    !              int* lmmax_aug,         //!< part of dimension of trans
    !              double* rspin,          //!< something?
    !              int* comm
    subroutine setup_context_cu(glob, trans, fermi_weights, kpoint_weights, &
        ntype, nion, nband, nkpoint_full, nkpoint_red, equiv_in, &
        lmdim, lmmax_aug, rspin, comm) bind(c,name="setup_context_cu_C")
        import
        type(c_ptr) :: glob
        real(c_double)      :: trans
        real(c_double)      :: fermi_weights(*)
        real(c_double)      :: kpoint_weights(*)
        integer(c_int)      :: ntype
        integer(c_int)      :: nion(*)
        integer(c_int)      :: nband
        integer(c_int)      :: nkpoint_full
        integer(c_int)      :: nkpoint_red
        integer(c_int)      :: equiv_in(*)
        integer(c_int)      :: lmdim
        integer(c_int)      :: lmmax_aug
        real(c_double)      :: rspin
        integer(c_int)      :: comm
    end subroutine setup_context_cu        

    !void gather_waves_cu(
    !              wavefunction_desc** waves, //!< structure to be populated
    !              cuDoubleComplex* kwav,     //!< k-space wavefunctions
    !              cuDoubleComplex* proj,     //!< projections
    !              cuDoubleComplex* phase,    //!< phase shift for lshift
    !              int* index,                //!< map from sphere <-> sticks
    !              double* fermi_weights,     //!< fermi weights of local bands
    !              int* grid,                 //!< size of inflated grid (sticks)
    !              int* nkwav,                //!< size in k-space
    !              int* nproj,                //!< number of projectors (size of projection)
    !              int* nindex,               //!< size of k-space sphere
    !              int* ntype,                //!< number of types of ions
    !              int* nion,                 //!< number of ions of each type
    !              int* lmmax,                //!< max lm-number
    !              int* linv,                 //!< inversion flag
    !		  int* lshift,               //!< phase shift flag
    !              int* nband,                //!< number of bands in this structure
    !              int* npos,                 //!< position of first band (assumed sequential)
    !              int* do_fft,               //!< do the fft?
    !              int* do_redis              //!< redisribution among PEs?
    !                            )
    subroutine gather_waves_cu(waves, kwav, proj, phase, indx, fermi_weights, & 
        grid, nkwav, nproj, nindex, ntype, nion, lmmax, linv, lshift, &
        nband, npos, do_fft, do_redis) bind(c,name="gather_waves_cu_C")
        import
        type(c_ptr) :: waves
        complex(C_DOUBLE_COMPLEX) :: kwav
        complex(C_DOUBLE_COMPLEX) :: proj
        complex(C_DOUBLE_COMPLEX) :: phase
        integer(c_int)      :: indx
        real(c_double)      :: fermi_weights(*)
        integer(c_int)      :: grid(*)
        integer(c_int)      :: nkwav
        integer(c_int)      :: nproj
        integer(c_int)      :: nindex
        integer(c_int)      :: ntype
        integer(c_int)      :: nion(*)
        integer(c_int)      :: lmmax(*)
        integer(c_int)      :: linv
        integer(c_int)      :: lshift
        integer(c_int)      :: nband
        integer(c_int)      :: npos
        integer(c_int)      :: do_fft
        integer(c_int)      :: do_redis
    end subroutine gather_waves_cu

    !void gather_projectors_cu(
    !              projector_desc** projs, //!< structure to be populated
    !              double* rproj,          //!< real-space projectors
    !              cuDoubleComplex* phase,          //!< phase vector
    !              int* nli,               //!< maps from r-space into projection sphere
    !              int* nlimax_in,         //!< number of entries in nli per ion 
    !              int *nproj_in,          //!< number of projectors
    !              int *ntype,             //!< number of types of ions
    !              int *nion_in,           //!< number of ions
    !              int *lmmax_in,          //!< max lm-number
    !              int *irmax_in,          //!< max(nlimax_in)
    !              int *gather             //!< flag: sum values from each PE
    subroutine gather_projectors_cu(projs, rproj, phase, nli, nlimax_in, &
        nproj_in, ntype, nion_in, lmmax_in, &
        irmax_in, gather) bind(c,name="gather_projectors_cu_C")
        import
        type(c_ptr) :: projs
        real(c_double)      :: rproj 
        complex(C_DOUBLE_COMPLEX)      :: phase 
        integer(c_int)      :: nli
        integer(c_int)      :: nlimax_in
        integer(c_int)      :: nproj_in
        integer(c_int)      :: ntype
        integer(c_int)      :: nion_in(*)
        integer(c_int)      :: lmmax_in(*)
        integer(c_int)      :: irmax_in
        integer(c_int)      :: gather
    end subroutine gather_projectors_cu

    !void gather_dproj_cu_C(
    !                           cuDoubleComplex* dproj, //!< important...
    !                           int *nproj, //!< it's see cproj
    !                           int *nband, //!< number of local bands
    !                           int *ndir
    subroutine gather_dproj_cu(dproj, nproj, nband, ndir) &
        bind(c,name="gather_dproj_cu_C")
        import
        complex(C_DOUBLE_COMPLEX) :: dproj
        integer(c_int)      :: nproj
        integer(c_int)      :: nband
        integer(c_int)      :: ndir
    end subroutine gather_dproj_cu

    !void update_phase_cu_C(
    !                             projector_desc** projs, //!< struct to be updated 
    !                             cuDoubleComplex* phase//!< new phase vector
    subroutine update_phase_cu(projs, phase) bind(c,name="update_phase_cu_C")
        import
        type(c_ptr) :: projs
        complex(C_DOUBLE_COMPLEX)      :: phase 
    end subroutine update_phase_cu
        
    !void fft_waves_cu_C(
    !                          wavefunction_desc** waves, //!< wavefunctions to fft
    !                          int* start, //!< starting band
    !                          int* num //!< number of bands
    subroutine fft_waves_cu(waves, start, num) bind(c,name="fft_waves_cu_C")
        import
        type(c_ptr) :: waves
        integer(c_int)      :: start
        integer(c_int)      :: num
    end subroutine fft_waves_cu

    !void free_structs_cu(global_desc** glob, wavefunction_desc** waves, 
    ! projector_desc** projs)
    subroutine free_structs_cu(glob, waves, projs) bind(c,name="free_structs_cu_C")
        import
        type(c_ptr) :: glob
        type(c_ptr) :: waves
        type(c_ptr) :: projs
    end subroutine free_structs_cu 

    !void fock_acc_cu_C(
    !              global_desc** glob,         //!< Global parameters
    !              wavefunction_desc** waves1, //!< wavefunctions of k-point
    !              wavefunction_desc** waves2, //!< wavefunctinos of q-point
    !              projector_desc** proj,      //!< projectors
    !              double* potfak,             //!< local potential
    !              cuDoubleComplex *cxi,       //!< output wavefunctions
    !              cuDoubleComplex* ckappa,    //!< output projectors
    !              double* exchange,           //!< ACFDT correction
    !              int* loverl,                //!< overlap? 
    !              int* compute_exchange,      //!< compute exchange?
    !              int* nk,                    //!< k-point number
    !              int* nq                     //!< q-point number
    subroutine fock_acc_cu(glob, waves1, waves2, proj, &
        potfak, cxi, ckappa, exchange, loverl, compute_exchange, &
        nk, nq) bind(c,name="fock_acc_cu_C")
        import
        type(c_ptr) :: glob
        type(c_ptr) :: waves1
        type(c_ptr) :: waves2
        type(c_ptr) :: proj
        real(c_double)      :: potfak(*) 
        complex(C_DOUBLE_COMPLEX) :: cxi(*)
        complex(C_DOUBLE_COMPLEX) :: ckappa(*)
        real(c_double)      :: exchange 
        integer(c_int)      :: loverl
        integer(c_int)      :: compute_exchange
        integer(c_int)      :: nk
        integer(c_int)      :: nq
    end subroutine fock_acc_cu
        
    !void fock_force_cu_C(
    !              global_desc** glob,         //!< global parameters
    !              wavefunction_desc** waves1, //!< wavefunctions of k-point
    !              wavefunction_desc** waves2, //!< wavefunctions of q-point
    !              projector_desc** projs,      //!< projectors
    !              double* potfak,             //!< array of potentials (7 of them)
    !              double* sif,                //!< one output
    !              double* sif2,               //!< another output
    !              double* forhf,              //!< third output
    !              int* loverl,                //!< flag: overlap? 
    !              int* nk,                    //!< k-point number
    !              int* nq                     //!< q-point number
    subroutine fock_force_cu(glob, waves1, waves2, projs, &
        potfak, sif, sif2, forhf, loverl, &
        nk, nq) bind(c,name="fock_force_cu_C")
        import
        type(c_ptr) :: glob
        type(c_ptr) :: waves1
        type(c_ptr) :: waves2
        type(c_ptr) :: projs(*)
        real(c_double)      :: potfak 
        real(c_double)      :: sif(*) 
        real(c_double)      :: sif2(*) 
        real(c_double)      :: forhf(*) 
        integer(c_int)      :: loverl
        integer(c_int)      :: nk
        integer(c_int)      :: nq
    end subroutine fock_force_cu

! ========================================
! CUDA/rmm-diis.cu
! ========================================

    !void get_local_contribution_result_(int *size, cuDoubleComplex *RESULT)
    subroutine get_local_contribution_result(thesize, res) &
        bind(c,name="get_local_contribution_result_C")
        import
        integer(c_int)      :: thesize
        complex(C_DOUBLE_COMPLEX) :: res
    end subroutine get_local_contribution_result
        
    !__host__ void local_contributionv2_setup_C_
    !     (int *size, const devptr_t *W1_CR_t, const devptr_t *W2_CR_t,
    !      const devptr_t *SV_t, int *shiftSV, int *CSTE,
    !      int *ISPINOR, int *ISPINOR_, int * SV_is_real,
    !      int *IDX_p)
    subroutine local_contributionv2_setup(thesize, W1_CR_t, W2_CR_t, &
        SV_t, shiftSV, CSTE, ISPINOR, ISPINOR_2, SV_is_real, &
        IDX_p) bind(c,name="local_contributionv2_setup_C")
        import
        integer(c_int)      :: thesize
        integer(c_intptr_t)   :: W1_CR_t
        integer(c_intptr_t)   :: W2_CR_t
        integer(c_intptr_t)   :: SV_t
        integer(c_int)      :: shiftSV
        integer(c_int)      :: CSTE
        integer(c_int)      :: ISPINOR
        integer(c_int)      :: ISPINOR_2
        integer(c_int)      :: SV_is_real
        integer(c_int)      :: IDX_p
    end subroutine local_contributionv2_setup
        
    !void CUBLASAPI local_contributionv2_arm_C(int *NSIM_p, cuDoubleComplex **GPU_BUFFER_p)
    subroutine local_contributionv2_arm(NSIM_p, GPU_BUFF) &
        bind(c,name="local_contributionv2_arm_C")
        import
        integer(c_int)      :: NSIM_p
        integer(c_intptr_t)   :: GPU_BUFF
    end subroutine local_contributionv2_arm
        
    !void CUBLASAPI local_contributionv2_launch_C(int *NSIM_p, cuDoubleComplex ** GPU_BUFFER_p)
    subroutine local_contributionv2_launch(NSIM_p, GPU_BUFF) &
        bind(c,name="local_contributionv2_launch_C")
        import
        integer(c_int)      :: NSIM_p
        integer(c_intptr_t)   :: GPU_BUFF
    end subroutine local_contributionv2_launch

    !__host__ void local_contribution_C(int *size, const devptr_t *W1_CR_t,
    !         const devptr_t *W2_CR_t, const devptr_t *SV_t, int *shiftSV, 
    ! int *CSTE, int *ISPINOR,
    !     int *ISPINOR_, int * SV_is_real, 
    !cuDoubleComplex * CLOCAL, int * assign, int * normalize)
    subroutine local_contribution(thesize, W1_CR_t, W2_CR_t, &
        SV_t, shiftSV, CSTE, ISPINOR, ISPINOR_2, SV_is_real, &
        CLOCAL, assign, normalize) bind(c,name="local_contribution_C")
        import
        integer(c_int)      :: thesize
        integer(c_intptr_t)   :: W1_CR_t
        integer(c_intptr_t)   :: W2_CR_t
        integer(c_intptr_t)   :: SV_t
        integer(c_int)      :: shiftSV
        integer(c_int)      :: CSTE
        integer(c_int)      :: ISPINOR
        integer(c_int)      :: ISPINOR_2
        integer(c_int)      :: SV_is_real
        complex(C_DOUBLE_COMPLEX) :: CLOCAL
        integer(c_int)      :: assign
        integer(c_int)      :: normalize
    end subroutine local_contribution

    !void gpu_zdot_nocopy_2d_C(int *nArguments_p, void ** argumentsCPU_p, void ** argumentsGPU_p,
     !cuDoubleComplex ** ret, const int * StreamIdx)
    subroutine gpu_zdot_nocopy_2d(nArgs, argsCPU, argsGPU, ret, streamIdx) &
        bind(c,name="gpu_zdot_nocopy_2d_C")
        import
        integer(c_int)      :: nArgs
        integer(c_intptr_t)   :: argsCPU(*)
        integer(c_intptr_t)   :: argsGPU
        integer(c_intptr_t)   :: ret
        integer(c_int)      :: streamIdx
    end subroutine gpu_zdot_nocopy_2d
        
    !void gpu_daxpy2_2d_C(const int *nArguments_p, void ** argumentsCPU_p, void ** argumentsGPU_p,  
     !void **a_p)
    subroutine gpu_daxpy2_2d(nArgs, argsCPU, argsGPU, alpha) &
        bind(c,name="gpu_daxpy2_2d_C")
        import
        integer(c_int)      :: nArgs
        integer(c_intptr_t)   :: argsCPU(*)
        integer(c_intptr_t)   :: argsGPU
        integer(c_intptr_t)   :: alpha
    end subroutine gpu_daxpy2_2d
        
    !void gpu_dscal_2d_C(const int *nArguments_p, void ** argumentsCPU_p,
    !               void ** argumentsGPU_p,  void **a_p)
    subroutine gpu_dscal_2d(nArgs, argsCPU, argsGPU, alpha) &
        bind(c,name="gpu_dscal_2d_C")
        import
        integer(c_int)      :: nArgs
        integer(c_intptr_t)   :: argsCPU(*)
        integer(c_intptr_t)   :: argsGPU
        integer(c_intptr_t)   :: alpha
    end subroutine gpu_dscal_2d
     
! ========================================
! CUDA/fortran.c
! ========================================
    !void CUBLAS_DAXPY (const int *n, const double *alpha, const devptr_t *devPtrx, 
    !               const int *incx, const devptr_t *devPtry, const int *incy)
    subroutine cublas_daxpy(n, alpha, devPtrx, incx, devPtry, incy) &
        bind(c,name="CUBLAS_DAXPY_C")
        import
        integer(c_int)            :: n
        real(c_double)            :: alpha 
        integer(c_intptr_t)       :: devPtrx
        integer(c_int)            :: incx
        integer(c_intptr_t)       :: devPtry
        integer(c_int)            :: incy
    end subroutine cublas_daxpy

    !void CUBLAS_ZAXPY_C (const int *n, const cuDoubleComplex *alpha, 
    !               const devptr_t *devPtrx, const int *incx, 
    !               const devptr_t *devPtry, const int *incy)
    subroutine cublas_zaxpy(n, alpha, devPtrx, incx, devPtry, incy) &
        bind(c,name="CUBLAS_ZAXPY_C")
        import
        integer(c_int)            :: n
        complex(C_DOUBLE_COMPLEX) :: alpha
        integer(c_intptr_t)       :: devPtrx
        integer(c_int)            :: incx
        integer(c_intptr_t)       :: devPtry
        integer(c_int)            :: incy
    end subroutine cublas_zaxpy

    !void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
    !               const int *n, const int *k, const cuDoubleComplex *alpha,
    !               const devptr_t *devPtrA, const int *lda,
    !               const devptr_t *devPtrB, const int *ldb, 
    !               const cuDoubleComplex *beta, const devptr_t *devPtrC,
    !               const int *ldc)
    subroutine cublas_zgemm(transa, transb, m, n, k, alpha, devPtrA, lda, &
        devPtrB, ldb, beta, devPtrC, ldc) bind(c,name="CUBLAS_ZGEMM_C")
        import
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        complex(C_DOUBLE_COMPLEX) :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: ldb
        complex(C_DOUBLE_COMPLEX) :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: ldc
    end subroutine cublas_zgemm

    !void CUBLAS_ZGEMMSH (const char *transa, const char *transb, const int *m,
    !                 const int *n, const int *k, const cuDoubleComplex *alpha,
    !                 const devptr_t *devPtrA,const int *shiftA, const int *lda,
    !                 const devptr_t *devPtrB,const int *shiftB, const int *ldb,
    !                 const cuDoubleComplex *beta, 
    !                 const devptr_t *devPtrC,const int *shiftC,const int *ldc )
    subroutine cublas_zgemmsh(transa, transb, m, n, k, alpha, &
        devPtrA, shiftA, lda, devPtrB, shiftB, ldb, beta, &
        devPtrC, shiftC, ldc) bind(c,name="CUBLAS_ZGEMMSH_C")
        import
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        complex(C_DOUBLE_COMPLEX) :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: shiftA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: shiftB
        integer(c_int)    :: ldb
        complex(C_DOUBLE_COMPLEX) :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: shiftC
        integer(c_int)    :: ldc
    end subroutine cublas_zgemmsh

    !void CUBLAS_ZGEMMSH_ST (const int * StreamIdx, const char *transa, 
    ! const char *transb, const int *m,
    !                    const int *n, const int *k, const cuDoubleComplex *alpha,
    !                    const devptr_t *devPtrA,const int *shiftA, const int *lda,
    !                    const devptr_t *devPtrB,const int *shiftB, const int *ldb,
    !                    const cuDoubleComplex *beta, 
    !                    const devptr_t *devPtrC,const int *shiftC,const int *ldc )
    subroutine cublas_zgemmsh_st(sid, transa, transb, m, n, k, &
        alpha, devPtrA, shiftA, lda, &
        devPtrB, shiftB, ldb, beta, devPtrC, shiftC, ldc) &
        bind(c,name="CUBLAS_ZGEMMSH_ST_C")
        import
        integer(c_int)    :: sid
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        complex(C_DOUBLE_COMPLEX) :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: shiftA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: shiftB
        integer(c_int)    :: ldb
        complex(C_DOUBLE_COMPLEX) :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: shiftC
        integer(c_int)    :: ldc
    end subroutine cublas_zgemmsh_st

    !void CUBLAS_DGEMMSH (const char *transa, const char *transb, const int *m,
    !                 const int *n, const int *k, const double *alpha, 
    !                 const devptr_t *devPtrA,const int *shiftA, const int *lda,
    !                 const devptr_t *devPtrB,const int *shiftB, const int *ldb,
    !                 const double *beta,
    !                 const devptr_t *devPtrC,const int *shiftC,const int *ldc )
    subroutine cublas_dgemmsh(transa, transb, m, n, k, alpha, &
        devPtrA, shiftA, lda, devPtrB, shiftB, ldb, beta, &
        devPtrC, shiftC, ldc) bind(c,name="CUBLAS_DGEMMSH_C")
        import
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        real(c_double)    :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: shiftA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: shiftB
        integer(c_int)    :: ldb
        real(c_double)    :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: shiftC
        integer(c_int)    :: ldc
    end subroutine cublas_dgemmsh
    
    !void CUBLAS_DGEMMSH_ST_C (const int *StreamIdx,const char *transa, 
    ! const char *transb, const int *m,
    !                 const int *n, const int *k, const double *alpha, 
    !                 const devptr_t *devPtrA,const int *shiftA, const int *lda,
    !                 const devptr_t *devPtrB,const int *shiftB, const int *ldb,
    !                 const double *beta,
    !                 const devptr_t *devPtrC,const int *shiftC,const int *ldc )
    subroutine cublas_dgemmsh_st(sid, transa, transb, m, n, k, &
        alpha, devPtrA, shiftA, lda, &
        devPtrB, shiftB, ldb, beta, devPtrC, shiftC, ldc) &
        bind(c,name="CUBLAS_DGEMMSH_ST_C")
        import
        integer(c_int)    :: sid
        character(c_char) :: transa
        character(c_char) :: transb
        integer(c_int)    :: m
        integer(c_int)    :: n
        integer(c_int)    :: k
        real(c_double)    :: alpha
        integer(c_intptr_t)       :: devPtrA
        integer(c_int)    :: shiftA
        integer(c_int)    :: lda
        integer(c_intptr_t)       :: devPtrB
        integer(c_int)    :: shiftB
        integer(c_int)    :: ldb
        real(c_double)    :: beta
        integer(c_intptr_t) devPtrC
        integer(c_int)    :: shiftC
        integer(c_int)    :: ldc
    end subroutine cublas_dgemmsh_st

    !void CUBLAS_RESET_ST_C(void)
    subroutine cublas_reset_st() bind(c,name="CUBLAS_RESET_ST_C")
    end subroutine cublas_reset_st

! ========================================
! CUDA/hamil.cu
! ========================================

    !void cuda_vhamil_C(const int *sid, const int *nrspinors, 
    !const int *np, const int *mplwv, 
    ! const devptr_t *devptrCVR, const devptr_t *devptrSV, const int *shift,
    ! const devptr_t *devptrCR, const double *rinplw, const int *is_real)
    subroutine cuda_vhamil(sid, nrspinors, np, mplwv, devptrCVR, &
        devptrSV, shift, devptrCR, rinplw, is_real) &
        bind(c,name="cuda_vhamil_C")
        import
        integer(c_int)    :: sid
        integer(c_int)    :: nrspinors
        integer(c_int)    :: np
        integer(c_int)    :: mplwv
        integer(c_intptr_t) :: devptrCVR
        integer(c_intptr_t) :: devptrSV
        integer(c_int)    :: shift
        integer(c_intptr_t) :: devptrCR
        real(c_double)    :: rinplw
        integer(c_int)    :: is_real
    end subroutine cuda_vhamil

    !void cuda_kinhamil_C(int *sid, int *N, cuDoubleComplex **ch,
        !cuDoubleComplex **cw, double **datake, double *evalue)
    subroutine cuda_kinhamil(sid, N, ch, cw, datake, evalue) &
        bind(c,name="cuda_kinhamil_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: N
        integer(c_intptr_t) :: ch
        integer(c_intptr_t) :: cw
        integer(c_intptr_t) :: datake
        real(c_double)      :: evalue
    end subroutine cuda_kinhamil

    !void cuda_ffthamil_C(int *sid, int *N, 
    ! cuDoubleComplex **ch, cuDoubleComplex **cw,
    ! double *evalue)
    subroutine cuda_ffthamil(sid, N, ch, cw, evalue) &
        bind(c,name="cuda_ffthamil_C")
        import
        integer(c_int)    :: sid
        integer(c_int)    :: N
        integer(c_intptr_t) :: ch
        integer(c_intptr_t) :: cw
        real(c_double)    :: evalue
    end subroutine cuda_ffthamil
        
    !void cuda_initprecond_C(const int *nsim, const int *N, 
    ! const devptr_t *devptr_cw,
    ! const devptr_t *devptr_datake, double *ekin, const int *ngdim,
    ! const int *ngvector, const int *ialgo)
    subroutine cuda_initprecond(nsim, N, devptr_cw, devptr_datake, ekin, &
        ngdim, ngvector, ialgo) bind(c,name="cuda_initprecond_C")
        import
        integer(c_int)    :: nsim
        integer(c_int)    :: N
        integer(c_intptr_t) :: devptr_cw(*)
        integer(c_intptr_t) :: devptr_datake
        real(c_double)    :: ekin(*)
        integer(c_int)    :: ngdim
        integer(c_int)    :: ngvector
        integer(c_int)    :: ialgo
    end subroutine cuda_initprecond 

    !void cuda_calcprecond_C(const int *nsim, const int *N, devptr_t *devptr_cw,
    ! const devptr_t *devptr_datake, const devptr_t *devptr_precon, double *ekin, 
    ! double *evalue, const double *slocal, const double *de_att, const int *ngdim,
    ! const int *ngvector, const int *nrplwv, const int *ialgo)
    subroutine cuda_calcprecond(nsim, N, devptr_cw, devptr_datake, &
        devptr_precon, ekin, evalue, slocal, de_att, &
        ngdim, ngvector, nrplwv, ialgo) bind(c,name="cuda_calcprecond_C")
        import
        integer(c_int)    :: nsim
        integer(c_int)    :: N
        integer(c_intptr_t) :: devptr_cw(*)
        integer(c_intptr_t) :: devptr_datake
        integer(c_intptr_t) :: devptr_precon
        real(c_double)    :: ekin(*)
        real(c_double)    :: evalue(*)
        real(c_double)    :: slocal
        real(c_double)    :: de_att
        integer(c_int)    :: ngdim
        integer(c_int)    :: ngvector
        integer(c_int)    :: nrplwv
        integer(c_int)    :: ialgo
    end subroutine cuda_calcprecond 

    !void cuda_applyprecond_C(const int *sid, const int *N, devptr_t *devptr_cw1,
    ! devptr_t *devptr_cw2, devptr_t *devptr_precon, 
    ! const double *mul, const int *shift)
    subroutine cuda_applyprecond(sid, N, devptr_cw1, devptr_cw2, &
        devptr_precon, mul, shift) bind(c,name="cuda_applyprecond_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: N
        integer(c_intptr_t) :: devptr_cw1
        integer(c_intptr_t) :: devptr_cw2
        integer(c_intptr_t) :: devptr_precon
        real(c_double)      :: mul
        integer(c_int)      :: shift
    end subroutine cuda_applyprecond

    !void cuda_truncatehighfrequency_C(const int *N, const int *ldelaylspiral,
    ! devptr_t *devptr_cw, devptr_t *devptr_datake, const double *enini)
    subroutine cuda_truncatehighfrequency(N, ldelaylspiral, devptr_cw, &
        devptr_datake, enini) bind(c,name="cuda_truncatehighfrequency_C")
        import
        integer(c_int)      :: N
        integer(c_int)      :: ldelaylspiral
        integer(c_intptr_t) :: devptr_cw
        integer(c_intptr_t) :: devptr_datake
        real(c_double)      :: enini
    end subroutine cuda_truncatehighfrequency

    !void cuda_normwithmetric_C(int *sid, int *N, devptr_t *devptr_c, 
    ! devptr_t *devptr_metric,
    ! double *fnorm, double *fmetric, int *fmetric_present, int *shift)
    subroutine cuda_normwithmetric(sid, N, devptr_c, &
        devptr_metric, fnorm, fmetric, fmetric_present, shift) &
        bind(c,name="cuda_normwithmetric_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: N
        integer(c_intptr_t) :: devptr_c
        integer(c_intptr_t) :: devptr_metric
        real(c_double)      :: fnorm
        real(c_double)      :: fmetric
        integer(c_int)      :: fmetric_present
        integer(c_int)      :: shift
    end subroutine cuda_normwithmetric
        
    !void cuda_normwithmetricblock_C(int *nsim, int *N, devptr_t *devptr_c, 
    ! devptr_t *devptr_metric,
    ! double *fnorm, double *fmetric, int *fmetric_present)
    subroutine cuda_normwithmetricblock(nsim, N, devptr_c, &
        devptr_metric, fnorm, fmetric, fmetric_present) &
        bind(c,name="cuda_normwithmetricblock_C")
        import
        integer(c_int)      :: nsim
        integer(c_int)      :: N
        integer(c_intptr_t) :: devptr_c(*)
        integer(c_intptr_t) :: devptr_metric
        real(c_double)      :: fnorm(*)
        real(c_double)      :: fmetric(*)
        integer(c_int)      :: fmetric_present
    end subroutine cuda_normwithmetricblock

    !void cuda_pwcharge_C(const int *sid, const int *nrspinors, 
    ! const int *np, const int *mplwv,
    ! const devptr_t *devptr_charge, const int *ndim, const devptr_t *devptr_cr1,
    ! const devptr_t *devptr_cr2, const double *weight, const int *is_real)
    subroutine cuda_pwcharge(sid, nrspinors, np, mplwv, devptr_charge, &
        ndim, devptr_cr1, devptr_cr2, weight, is_real) &
        bind(c,name="cuda_pwcharge_C")
        import
        integer(c_int)      :: sid
        integer(c_int)      :: nrspinors
        integer(c_int)      :: np
        integer(c_int)      :: mplwv
        integer(c_intptr_t) :: devptr_charge
        integer(c_int)      :: ndim
        integer(c_intptr_t) :: devptr_cr1
        integer(c_intptr_t) :: devptr_cr2
        real(c_double)      :: weight
        integer(c_int)      :: is_real
    end subroutine cuda_pwcharge    
        
! ========================================
! CUDA/nonlr.cu
! ========================================
    !void gpu_crrexp_mul_wave_st_multi_C(const int *StreamIdx, const int *MArray,
    ! const int *MArray_size, const int *copy_MArray, const devptr_t *devPtrCR,
    ! const int *copy_cr_array, const int *shiftCR, const devptr_t *devPtrWORK,
    ! const int *shiftW1, const int *irmax, const int *ndata, const int *nsim,
    ! const devptr_t *devPtrNLI, const int *N1, const devptr_t *devPtrCRREXP,
    ! const int *ispiral, const int *c1, const int *c2, const int *nib,
    ! const int *end)
    subroutine gpu_crrexp_mul_wave_st_multi(StreamIdx, MArray, MArray_size, &
        copy_MArray, devPtrCR, copy_cr_array, shiftCR, devPtrWORK, shiftW1, &
        irmax, ndata, nsim, devPtrNLI, N1, devPtrCRREXP, ispiral, c1, &
        c2, nib, theend) bind(c,name="gpu_crrexp_mul_wave_st_multi_C")
        import
        integer(c_int)     :: StreamIdx
        integer(c_int)     :: MArray(*)
        integer(c_int)     :: MArray_size
        integer(c_int)     :: copy_MArray
        integer(c_intptr_t) :: devPtrCR(*)
        integer(c_int)     :: copy_cr_array
        integer(c_int)     :: shiftCR
        integer(c_intptr_t) :: devPtrWORK
        integer(c_int)     :: shiftW1
        integer(c_int)     :: irmax
        integer(c_int)     :: ndata
        integer(c_int)     :: nsim
        integer(c_intptr_t) :: devPtrNLI
        integer(c_int)     :: N1
        integer(c_intptr_t) :: devPtrCRREXP
        integer(c_int)     :: ispiral
        integer(c_int)     :: c1
        integer(c_int)     :: c2
        integer(c_int)     :: nib
        integer(c_int)     :: theend
    end subroutine gpu_crrexp_mul_wave_st_multi
        
    !void cuda_calccproj_C(const int *M, const int *block_size,
    ! const devptr_t *devptr_cproj, const int *lmbase, const int *npro_tot,
    ! const devptr_t *devptr_tmp, const int *shiftT1,
    ! const double *rinpl, const int *nsim, const int *nlm,
    ! const devptr_t *devptr_ldo, const int *ldo_ind_size, const int *ndata)
    subroutine cuda_calccproj(M, block_size, devptr_cproj, lmbase, &
        npro_tot, devptr_tmp, shiftT1, rinpl, nsim, nlm, devptr_ldo, &
        ldo_ind_size, ndata) bind(c,name="cuda_calccproj_C")
        import
        integer(c_int)     :: M
        integer(c_int)     :: block_size
        integer(c_intptr_t) :: devptr_cproj
        integer(c_int)     :: lmbase
        integer(c_int)     :: npro_tot
        integer(c_intptr_t) :: devptr_tmp
        integer(c_int)     :: shiftT1
        real(c_double)      :: rinpl
        integer(c_int)     :: nsim
        integer(c_int)     :: nlm
        integer(c_intptr_t) :: devptr_ldo
        integer(c_int)     :: ldo_ind_size
        integer(c_int)     :: ndata
    end subroutine cuda_calccproj
        
    !void cuda_calccracc_crrexp_C(int *sid, int *nsim, int *irmax, int *nlimax,
    ! devptr_t *devptr_cracc, int *shiftcracc, int *ldcracc,
    ! devptr_t *devptr_work, int *shiftwork, int *ldwork,
    ! devptr_t *devptr_crrexp, int *shiftcrrexp,
    ! int *ndata, devptr_t *devptr_nli, int *shiftnli,
    ! devptr_t *devptr_ldo, int *partial)
    subroutine cuda_calccracc_crrexp(sid, nsim, irmax, nlimax, &
        devptr_cracc, shiftcracc, ldcracc, &
        devptr_work, shiftwork, ldwork, &
        devptr_crrexp, shiftcrrexp, &
        ndata, devptr_nli, shiftnli, &
        devptr_ldo, partial) bind(c,name="cuda_calccracc_crrexp_C")
        import
        integer(c_int)     :: sid
        integer(c_int)     :: nsim
        integer(c_int)     :: irmax
        integer(c_int)     :: nlimax
        integer(c_intptr_t) :: devptr_cracc
        integer(c_int)     :: shiftcracc
        integer(c_int)     :: ldcracc
        integer(c_intptr_t) :: devptr_work
        integer(c_int)     :: shiftwork
        integer(c_int)     :: ldwork
        integer(c_intptr_t) :: devptr_crrexp
        integer(c_int)     :: shiftcrrexp
        integer(c_int)     :: ndata
        integer(c_intptr_t) :: devptr_nli
        integer(c_int)     :: shiftnli
        integer(c_intptr_t) :: devptr_ldo
        integer(c_int)     :: partial
    end subroutine cuda_calccracc_crrexp

    !void cuda_calccracc_C(int *sid, int *nsim, int *irmax, int *nlimax,
    ! devptr_t *devptr_cracc, int *shiftcracc, int *ldcracc,
    ! devptr_t *devptr_work, int *shiftwork, int *ldwork,
    ! int *ndata, devptr_t *devptr_nli, int *shiftnli,
    ! devptr_t *devptr_ldo, int *partial)
    subroutine cuda_calccracc(sid, nsim, irmax, nlimax, &
        devptr_cracc, shiftcracc, ldcracc, &
        devptr_work, shiftwork, ldwork, &
        ndata, devptr_nli, shiftnli, &
        devptr_ldo, partial) bind(c,name="cuda_calccracc_C")
        import
        integer(c_int)     :: sid
        integer(c_int)     :: nsim
        integer(c_int)     :: irmax
        integer(c_int)     :: nlimax
        integer(c_intptr_t) :: devptr_cracc
        integer(c_int)     :: shiftcracc
        integer(c_int)     :: ldcracc
        integer(c_intptr_t) :: devptr_work
        integer(c_int)     :: shiftwork
        integer(c_int)     :: ldwork
        integer(c_int)     :: ndata
        integer(c_intptr_t) :: devptr_nli
        integer(c_int)     :: shiftnli
        integer(c_intptr_t) :: devptr_ldo
        integer(c_int)     :: partial
    end subroutine cuda_calccracc

    !void cuda_splitcproj_C(int *nbatch, int *nsim, int *lmmaxc,
    ! devptr_t *devptr_tmp, int *ldtmp, int *sizetmp,
    ! devptr_t *devptr_cproj, int *shiftcproj, int *ldcproj,
    ! double *rinpl, int *ndata, devptr_t *devptr_ldo, int *cmplx, int *partial)
    subroutine cuda_splitcproj(nbatch, nsim, lmmaxc, devptr_tmp, &
        ldtmp, sizetmp, devptr_cproj, shiftcproj, ldcproj, &
        rinpl, ndata, devptr_ldo, iscmplx, partial) &
        bind(c,name="cuda_splitcproj_C")
        import
        integer(c_int)     :: nbatch
        integer(c_int)     :: nsim
        integer(c_int)     :: lmmaxc
        integer(c_intptr_t) :: devptr_tmp
        integer(c_int)     :: ldtmp
        integer(c_int)     :: sizetmp
        integer(c_intptr_t) :: devptr_cproj
        integer(c_int)     :: shiftcproj
        integer(c_int)     :: ldcproj
        real(c_double)      :: rinpl
        integer(c_int)     :: ndata
        integer(c_intptr_t) :: devptr_ldo
        integer(c_int)     :: iscmplx
        integer(c_int)     :: partial
    end subroutine cuda_splitcproj
        
! ========================================
! CUDA/cuda_mem.cu
! ========================================

    !int nvpinnedmalloc(void **ptr, size_t *size_)
    subroutine nvpinnedmalloc(ptr, thesize) bind(c,name="nvpinnedmalloc_C")
       import
       type(c_ptr)   :: ptr
       integer(c_size_t)     :: thesize
    end subroutine nvpinnedmalloc

    !int nvpinnedfree_C(void **ptr)
    subroutine nvpinnedfree(ptr) bind(c,name="nvpinnedfree_C")
        import
        type(c_ptr) :: ptr
    end subroutine nvpinnedfree

    !int allocate_gpu_memory_batched_init_C(const size_t *nBytes, size_t *offset,
        !size_t *remaining, devptr_t *devPtr)
    subroutine allocate_gpu_memory_batched_init(nBytes, offset, &
        remaining, devPtr) bind(c,name="allocate_gpu_memory_batched_init_C")
        import
        integer(c_size_t) :: nBytes
        integer(c_size_t) :: offset
        integer(c_size_t) :: remaining
        integer(c_intptr_t) :: devPtr
    end subroutine allocate_gpu_memory_batched_init

    !int allocate_gpu_memory_batched_request_C(const size_t *nBytes, 
        !size_t *curOffset_,
        !size_t *remaining_, devptr_t *bufferPtr, devptr_t *devicePtr)
    subroutine allocate_gpu_memory_batched_request(nBytes, curOffset, &
        remaining, bufferPtr, devicePtr) &
        bind(c,name="allocate_gpu_memory_batched_request_C")
        import
        integer(c_size_t)       :: nBytes
        integer(c_size_t)       :: curOffset
        integer(c_size_t)       :: remaining
        integer(c_intptr_t)     :: bufferPtr
        integer(c_intptr_t)     :: devicePtr
    end subroutine allocate_gpu_memory_batched_request

    !void cublas_alloc_safety_C(int *n, size_t *elemSize, void **devPtr)
    subroutine cublas_alloc_safety(n, elemSize, devPtr) &
        bind(c,name="cublas_alloc_safety_C")
        import
        integer(c_int)      :: n
        integer(c_size_t)   :: elemSize
        integer(c_intptr_t)   :: devPtr
    end subroutine cublas_alloc_safety

    !int cublas_free_C(const devptr_t *devicePtr)
    subroutine cublas_free(devPtr) & 
        bind(c,name="cublas_free_C")
        import
        integer(c_intptr_t) :: devPtr
    end subroutine cublas_free

    !void cuda_memcpydtod_C(int *sid, void **dst, void **src, int *n, size_t *size)
    subroutine cuda_memcpydtod(sid, dst, src, n, thesize) &
        bind(c,name="cuda_memcpydtod_C")
        import
        integer(c_int)      :: sid
        integer(c_intptr_t) :: dst
        integer(c_intptr_t) :: src
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memcpydtod

    !void cuda_memcpyhtod_C(int *sid, void **dst, void *src, int *n, size_t *size)
    subroutine cuda_memcpyhtod(sid, dst, src, n, thesize) &
        bind(c,name="cuda_memcpyhtod_C")
        import
        integer(c_int)      :: sid
        integer(c_intptr_t) :: dst
        type(c_ptr), value  :: src
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memcpyhtod

    !void cuda_memcpydtoh_C(int *sid, void *dst, void **src, int *n, size_t *size)
    subroutine cuda_memcpydtoh(sid, dst, src, n, thesize) &
        bind(c,name="cuda_memcpydtoh_C")
        import
        integer(c_int)      :: sid
        type(c_ptr), value  :: dst
        integer(c_intptr_t) :: src
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memcpydtoh

    !void cuda_memcpydtohshift_C(int *sid, intptr_t dst, 
            !int *shiftdst, intptr_t *src, int *shiftsrc,
            !int *n, size_t *size)
    subroutine cuda_memcpydtohshift(sid, dst, shiftdst, src, &
        shiftsrc, n, thesize) &
        bind(c,name="cuda_memcpydtohshift_C")
        import
        integer(c_int)      :: sid
        type(c_ptr), value  :: dst
        integer(c_int)      :: shiftdst
        integer(c_intptr_t) :: src
        integer(c_int)      :: shiftsrc
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memcpydtohshift
    
    !void cuda_memset_C(void **devPtr, int *value, int *n, size_t *size)
    subroutine cuda_memset(devPtr, thevalue, n, thesize) &
        bind(c,name="cuda_memset_C")
        import
        integer(c_intptr_t) :: devPtr
        integer(c_int)      :: thevalue
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memset

    !void cuda_memsetasync_C(int *sid, void **devPtr, int *value, int *n, size_t *size)
    subroutine cuda_memsetasync(sid, devPtr, thevalue, n, thesize) &
        bind(c,name="cuda_memsetasync_C")
        import
        integer(c_int)      :: sid
        integer(c_intptr_t) :: devPtr
        integer(c_int)      :: thevalue
        integer(c_int)      :: n
        integer(c_size_t)   :: thesize
    end subroutine cuda_memsetasync

! ========================================
! CUDA/mpi.cu
! ========================================

#ifdef GPUDIRECT
    !void cuda_alltoall_C(double **src_, double **dst_, const int *size_,
     !const int *procId_, const int *nProcs_, const int *MAX_)
    subroutine cuda_alltoall(src, dst, thesize, procid, nprocs, max_) &
        bind(c,name="cuda_alltoall_C")
        import
        integer(c_intptr_t) :: src
        integer(c_intptr_t) :: dst
        integer(c_int)      :: thesize
        integer(c_int)      :: procid
        integer(c_int)      :: nprocs
        integer(c_int)      :: max_
    end subroutine cuda_alltoall
    subroutine cuda_alltoall_host_dev(src, dst, thesize, procid, nprocs, max_) &
        bind(c,name="cuda_alltoall_host_dev_C")
        import
        type(c_ptr)         :: src
        integer(c_intptr_t) :: dst
        integer(c_int)      :: thesize
        integer(c_int)      :: procid
        integer(c_int)      :: nprocs
        integer(c_int)      :: max_
    end subroutine cuda_alltoall_host_dev
    subroutine cuda_alltoall_dev_host(src, dst, thesize, procid, nprocs, max_) &
        bind(c,name="cuda_alltoall_dev_host_C")
        import
        integer(c_intptr_t) :: src
        type(c_ptr)         :: dest
        integer(c_int)      :: thesize
        integer(c_int)      :: procid
        integer(c_int)      :: nprocs
        integer(c_int)      :: max_
    end subroutine cuda_alltoall_dev_host
#endif

! ========================================
! CUDA/davidson.cu
! ========================================

    !void cuda_correctcham_C(size_t *cham_all_gpu_ptr, 
    ! const int *nsim_ptr, const int *offset_ptr,
    ! const int *N_ptr, const int *set_diagonal_flag_ptr, 
    ! const double *diagonal_cpu )
    subroutine cuda_correctcham(cham_all_gpu_ptr, nsim_ptr, &
        offset_ptr, N_ptr, set_diagonal_flag_ptr, diagonal_cpu) &
        bind(c,name="cuda_correctcham_C")
        import
        integer(c_intptr_t) :: cham_all_gpu_ptr
        integer(c_int)      :: nsim_ptr
        integer(c_int)      :: offset_ptr
        integer(c_int)      :: N_ptr
        integer(c_int)      :: set_diagonal_flag_ptr
        type(c_ptr), value  :: diagonal_cpu
    end subroutine cuda_correctcham

! ========================================
! CUDA/potlok.cu
! ========================================

    !void cuda_ggaallgrid_C(const devptr_t * dcharg_p, 
    ! devptr_t * dworkg_p, devptr_t * dwork_p, devptr_t * excl_p, 
    ! const int * np_p, const double * autoa_p, const double * autoa3_p, 
    ! const double * autoa4_p, const double * rytoev_p, 
    ! const int * lexch_p, double * exc_ret, const double * ldascreen_p, 
    ! const int * luse_thomas_fermi_p, const int * force_pbe_p, 
    ! const double * aggac_p, const double * aggax_p, 
    ! const int * use_longrange_hf_p,
    ! const devptr_t * fs_rs_p, const devptr_t * ssplines_rho_s_p, 
    ! const devptr_t * ssplines_s_rho_p, const int * LOGRHO0_, 
    !const int *DWORK_IS_REAL)
    subroutine cuda_ggaallgrid( &
        dcharg_p,  &
        dworkg_p, dwork_p, excl_p,  &
        np_p, autoa_p, autoa3_p,  &
        autoa4_p, rytoev_p,  &
        lexch_p, exc_ret, ldascreen_p,  &
        luse_thomas_fermi_p, force_pbe_p,  &
        aggac_p, aggax_p,  &
        use_longrange_hf_p, &
        fs_rs_p, ssplines_rho_s_p,  &
        ssplines_s_rho_p, LOGRHO0_,  &
        DWORK_IS_REAL) &
        bind(c,name="cuda_ggaallgrid_C")
        import
        integer(c_intptr_t) :: dcharg_p 
        integer(c_intptr_t) :: dworkg_p 
        integer(c_intptr_t) :: dwork_p 
        integer(c_intptr_t) :: excl_p 
        integer(c_int)      :: np_p 
        real(c_double)      :: autoa_p 
        real(c_double)      :: autoa3_p 
        real(c_double)      :: autoa4_p 
        real(c_double)      :: rytoev_p 
        integer(c_int)      :: lexch_p 
        real(c_double)      :: exc_ret 
        real(c_double)      :: ldascreen_p 
        integer(c_int)      :: luse_thomas_fermi_p 
        integer(c_int)      :: force_pbe_p 
        real(c_double)      :: aggac_p 
        real(c_double)      :: aggax_p 
        integer(c_int)      :: use_longrange_hf_p
        integer(c_intptr_t) :: fs_rs_p 
        integer(c_intptr_t) :: ssplines_rho_s_p 
        integer(c_intptr_t) :: ssplines_s_rho_p 
        integer(c_int)      :: LOGRHO0_ 
        integer(c_int)      :: DWORK_IS_REAL
    end subroutine cuda_ggaallgrid

! ========================================
! CUDA/cuda_profiling.cu
! ========================================

    !void cuda_profilerstart_C(void)
    subroutine cuda_profilerstart() bind(c,name="cuda_profilerstart_C")
        import
    end subroutine cuda_profilerstart

    !void cuda_profilerstop_C(void)
    subroutine cuda_profilerstop() bind(c,name="cuda_profilerstop_C")
        import
    end subroutine cuda_profilerstop

    !void nvp_malloc_C(int *n)
    subroutine nvp_malloc(n) bind(c,name="nvp_malloc_C")
        import
        integer(c_int)      :: n
    end subroutine nvp_malloc

    !void nvp_free_C()
    subroutine nvp_free() bind(c,name="nvp_free_C")
        import
    end subroutine nvp_free

    !void nvp_start_C(int *n)
    subroutine nvp_start(tid) bind(c,name="nvp_start_C")
        import
        integer(c_int)      :: tid
    end subroutine nvp_start

    !void nvp_stop_C(int *n)
    subroutine nvp_stop(tid) bind(c,name="nvp_stop_C")
        import
        integer(c_int)      :: tid
    end subroutine nvp_stop

    !void nvp_print_C(int *n)
    subroutine nvp_print(tid) bind(c,name="nvp_print_C")
        import
        integer(c_int)      :: tid
    end subroutine nvp_print
! ========================================
! ========================================
! ========================================

    end interface
end module cuda_interface
