!-------- to be costumized by user (usually done in the makefile)-------
!#define vector              compile for vector machine
!#define essl                use ESSL instead of LAPACK
!#define single_BLAS         use single prec. BLAS

!#define wNGXhalf            gamma only wavefunctions (X-red)
!#define wNGZhalf            gamma only wavefunctions (Z-red)

!#define NGXhalf             charge stored in REAL array (X-red)
!#define NGZhalf             charge stored in REAL array (Z-red)
!#define NOZTRMM             replace ZTRMM by ZGEMM
!#define REAL_to_DBLE        convert REAL() to DBLE()
!#define MPI                 compile for parallel machine with MPI
!------------- end of user part --------------------------------
#ifdef single_BLAS
!
!   use single precission BLAS and LAPACK calls (usually on Cray)
!
#define ZGEMM  CGEMM
#define ZGEMV  CGEMV
#define ZDSCAL CSSCAL
#define ZDOTC  CDOTC
#define ZAXPY  CAXPY
#define ZHEEV  CHEEV
#define ZHEGV  CHEGV
#define ZTRMM  CTRMM
#define ZTRTRI CTRTRI
#define ZPOTRF CPOTRF
#define PZPOTRF PCPOTRF
#define PZTRTRI PCTRTRI
#define PZHEEVX PCHEEVX
#define ZCOPY   CCOPY
#define ZGETRF  CGETRF
#define ZGETRS  CGETRS
#define ZGEFA   CGEFA
#define ZGECO   CGECO
#define ZGEDI   CGEDI
#define ZHEEVX  CHEEVX

#define DGEMM  SGEMM
#define DGEMV  SGEMV
#define DGEGV  SGEGV
#define DDOT   SDOT
#define DSCAL  SSCAL
#define DAXPY  SAXPY
#define DSYEV  SSYEV
#define DSYGV  SSYGV
#define DTRMM  STRMM
#define DTRTRI STRTRI
#define DPOTRF SPOTRF
#define DGETRF SGETRF
#define DGETRS SGETRS
#define PDPOTRF PSPOTRF
#define PDTRTRI PSTRTRI
#define PDSYEVX PSSYEVX
#define DCOPY   SCOPY
#define DSYEVX  SSYEVX

#endif
#ifdef NGZhalf
!
!   charge density: half grid mode Z direction
!
#define realmode
#define  zm(n)             (n/2+1)
#define  xm(n)              n
#define  xmin(n)          (-(n/2-1))
#define  SETN1fromN1P     N1=MOD(N1P+xmin(GRIDC%NGX)+GRIDC%NGX,GRIDC%NGX)
#define  SETFACT1         FACTM=1
#define  SETFACT          IF (N3 /= 1) FACTM=2
#define  SETSQRT          IF (N3 /= 1) FACTM=SQRT(2._q)
#define  SETISQRT         IF (N3 /= 1) FACTM=1./SQRT(2._q)
#define  MULFACT          FACTM*
#elif defined(NGXhalf)
!
!   charge density: half grid mode X direction
!
#define realmode
#define  xm(n)             (n/2+1)
#define  xmin(n)            0
#define  zm(n)              n
#define  SETN1fromN1P     N1=N1P
#define  SETFACT1         FACTM=1
#define  SETFACT          IF (N1 /= 1) FACTM=2
#define  SETSQRT          IF (N1 /= 1) FACTM=SQRT(2._q)
#define  SETISQRT         IF (N1 /= 1) FACTM=1./SQRT(2._q)
#define  MULFACT          FACTM*
#else
!
!   charge density: full grid mode
!
#define  zm(n)              n
#define  xm(n)              n
#define  xmin(n)          (-(n/2-1))
#define  SETN1fromN1P      N1=MOD(N1P+xmin(GRIDC%NGX)+GRIDC%NGX,GRIDC%NGX)
#define  SETFACT1          FACTM=1
#define  SETFACT
#define  SETSQRT
#define  SETISQRT
#define  MULFACT
#endif
#ifdef realmode
!
!   charge density real
!
#define RGRID             REAL(q)
#define OVERLAP           REAL(q)
#define DIMREAL(n)        n*2
#define CHGCONJG(n)       (n)
#else
!
!   charge density complex
!
#define RGRID            COMPLEX(q)
#define OVERLAP          COMPLEX(q)
#define DIMREAL(n)       n
#define CHGCONJG(n)      CONJG(n)
#endif

#ifdef wNGZhalf
!
!   wavefunctions: half grid mode for Z direction
!
#define gammareal
#elif defined(wNGXhalf)
!
!   wavefunctions: half grid mode for X direction
!
#define gammareal
#else
!
!   wavefunctions: full grid mode
!
#endif
#ifdef  gammareal
!
!   wavefunctions real (gamma only)
!
#define GREAL(n)  REAL(n,KIND=q)
#define GCMPLX(n,m,q)  (n)
#define GDEF      REAL(q)
#define GDEFS     REAL(qs)
#define GDEFN     REAL(qn)
#define M_sum_g   M_sum_d
#define M_sumf_g M_sumf_d
#define M_bcast_g M_bcast_d
#define M_bcast_g_from M_bcast_d_from
#define CPROJ     GPROJ
#define GCONJG(n) (n)
#define GGEMM     DGEMM
#define GGEMV     DGEMV
#define GGETRF    DGETRF
#define GGETRS    DGETRS
#define GDOTC     DDOT
#define GDSCAL    DSCAL
#define GAXPY     DAXPY
#define GDOT      DDOT
#define PGGEMM    PDGEMM
#define PGGEADD   PDGEADD
#define zero      0._q
#define one       1._q
#define m_        2*
#define ndata     1
#define trans     'T'
#define gammaarg   NPLWKC,NINDPC,WVSCAL,WVSCAI
#define gammaargNK NPLWKC(NK),NINDPC(1,NK),WVSCAL(1,NK),WVSCAI(1,NK)
#else
!
!   wavefunctions complex
!
#define GREAL(n)  (n)
#define GCMPLX(n,m,q)  CMPLX(n,m,KIND=q)
#define GDEF      COMPLEX(q)
#define GDEFS     COMPLEX(qs)
#define GDEFN     COMPLEX(qn)
#define M_sumf_g  M_sumf_z
#define M_sum_g   M_sum_z
#define M_bcast_g M_bcast_z
#define M_bcast_g_from M_bcast_z_from
#define GCONJG(n) CONJG(n)
#define GGEMM     ZGEMM
#define GGEMV     ZGEMV
#define GGETRF    ZGETRF
#define GGETRS    ZGETRS
#define GDOTC     ZDOTC
#define GDSCAL    ZDSCAL
#define GAXPY     ZAXPY
#define GDOT      ZDOTC
#define PGGEMM    PZGEMM
#define PGGEADD   PZGEADD
#define zero      (0._q,0._q)
#define one       (1._q,0._q)
#define trans     'C'
#define m_
#define ndata     2
#define gammaarg
#define gammaargNK
#endif
!
!   common definitions
!
#define ztrans     'C'
#define dtrans     'T'
#define zzero      (0._q,0._q)
#define zone       (1._q,0._q)
#define dzero      0._q
#define done       1._q

#ifdef  REAL_to_DBLE
#define REAL(x) DBLE(x)
#endif
#if defined(MPI)
!
!   mpi parallel macros
!
#define CALLMPI(x) CALL x
#define CALLMPI_C(x) CALL x
#define STOP       CALL M_exit(); stop

#define io_begin IF (NODE_ME==IONODE) THEN
#define do_io    IF (NODE_ME==IONODE)
#define io_end   ENDIF
#define FFTMAKEPLAN FFTMAKEPLAN_MPI
#define FFT3D       FFT3D_MPI
#define FFTCHK      FFTCHK_MPI
#define FFT3RC      FFT3RC_MPI
#define FFTEXT      FFTEXT_MPI
#define FFTWAV      FFTWAV_MPI
#define FFTINI      FFTINI_MPI
#elif defined(MPI_CHAIN)
!
!   mpi nudged chain version
!
#define CALLMPI(x)
#define CALLMPI_C(x) CALL x
#define STOP       CALL M_exit(); STOP

#define io_begin IF (NODE_ME==IONODE) THEN
#define do_io    IF (NODE_ME==IONODE)
#define io_end   ENDIF
#else
#define CALLMPI(x)
#define CALLMPI_C(x)
#define nodes
#define do_io
#define io_begin
#define io_end
#undef use_shmem
#undef shmem_rproj
#undef shmem_bcast_buffer
#endif
#ifdef F90_T3D
!
!   T3D macros
!
!#define IVDEP
#endif
#if defined(T3D_SMA)
!
!   T3D SHMEM communication
!
#endif
#ifdef CRAY_MPP
!
!   CRAY_MPP macros
!
#define MPI_double_precision MPI_real
#define MPI_double_complex   MPI_complex
#endif
#ifdef USE_ERF
!
!  use instrinsic ERF
!
#define ERRF(x) ERF(x)
#define ERRFC(x) ERFC(x)
#endif

#define dwrite DWRITE
#ifdef debug
!
!  debugging primitives
!
#define DWRITE WRITE(0,*)
#define DWRITE0 do_io WRITE(0,*)
#else
#define DWRITE  !
#define DWRITE0 !
#endif

#define CW CPTWFP

#ifdef nonlr_single
!
! single precession non local projection operators  
!
#define RPROJ   RPROJS
#define qn      qs
#define DGEMVn  SGEMV
#define DGEMMn  SGEMM
#else
#define  qn    q
#define DGEMVn  DGEMV
#define DGEMMn  DGEMM
#endif


#if  ! (defined(NGXhalf)  || defined(NGYhalf) ||  defined(NGZhalf))
#define RHOLM_complex
#endif

#ifdef dotiming
!
!  some timing primitives
!
#define usetmg REAL(q) :: TV,TC,TV0,TC0
#define statmg CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror )) ;  CALL VTIME(TV0,TC0)
#define stotmg CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror )) ;  CALL VTIME(TV,TC) ; IF (NODE_ME==IONODE) WRITE(*,'(A,2F10.4)') 'timing ',TV-TV0,TC-TC0 ; CALL VTIME(TV0,TC0)
#define stotmgt(X) CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror )) ;  CALL VTIME(TV,TC) ; IF (NODE_ME==IONODE) WRITE(*,'(A,2F10.4)') X,TV-TV0,TC-TC0 ; CALL VTIME(TV0,TC0)
#define addtmg(T1,T2) CALLMPI(MPI_barrier(WDES%COMM%MPI_COMM,ierror)); CALL VTIME(TV,TC); T1=T1+TV-TV0;T2=T2+TC-TC0; TV0=TV;TC0=TC
#define addtmga(T1,T2) CALL VTIME(TV,TC); T1=T1+TV-TV0;T2=T2+TC-TC0; TV0=TV;TC0=TC
#define wrttmg(X,T1,T2) IF (NODE_ME==IONODE) WRITE(*,'(A,2F10.4)') X,T1,T2
#else
#define usetmg 
#define statmg 
#define stotmg 
#define stotmgt(X)
#define addtmg(T1,T2)
#define addtmga(T1,T2)
#define wrttmg(X,T1,T2)
#endif
!
! OpenMP macros
!
#ifdef _OPENMP45
#define _OPENMP40
#define _OPENMP
#define __simd_static          simd:static
#else
#define __simd_static          static
#endif
#ifdef _OPENMP40
#define _OPENMP
#endif
#ifdef _OPENMP
#define __omp_id               omp_id
#define __omp_get_max_threads  ,OMP_GET_MAX_THREADS()
#define __omp_get_num_threads  ,OMP_GET_NUM_THREADS()
#define __omp_get_thread_num   ,OMP_GET_THREAD_NUM()+1
#define __omp_arg(x)           ,x
#define __omp_arg_id           __omp_arg(__omp_id)
#define __omp_add(x)           x
#else
#define __omp_id               1
#define __omp_get_max_threads
#define __omp_get_num_threads
#define __omp_get_thread_num
#define __omp_arg(x)
#define __omp_arg_id
#define __omp_add(x)
#endif
!
! memalign macros
!
#ifdef memalign64
#define memalign
#define __align_NLM            104
#define __align_BTS            64
#define __align_DBL            8
#define __align_DIM(x)         ((x+__align_DBL-1)/__align_DBL)*__align_DBL
#else
#define __align_NLM            101
#define __align_DIM(x)         x
#endif
!
! profiling macros
!
#ifdef PROFILING
#define PROFILING_START(x)     CALL START_PROFILING(x)
#define PROFILING_STOP(x)      CALL STOP_PROFILING(x)
#else
#define PROFILING_START(x)
#define PROFILING_STOP(x)
#endif
!
! shmem macros
!
#ifdef shmem_rproj
#define use_shmem
#endif
#ifdef shmem_bcast_buffer
#define use_shmem
#endif
!
! quadruple precision 
!
#ifdef qd_emulate
#define QDPREAL                TYPE(qd_real)
#define QDPD(x)                qdreal(REAL(x,KIND=q))
#ifdef __GNUC__
#define QDPP(x)                qdreal('x')
#else
#define QDPP(x)                qdreal(#x)
#endif
#else
#define QDPREAL                REAL(qd)
#define QDPD(x)                REAL(x,KIND=qd)
#define QDPP(x)                REAL(x,KIND=qd)
#endif
!
! for the VASP2WANNIER90 interface
!
#ifdef VASP2WANNIER90v2
#define VASP2WANNIER90
#endif
!
! CUDA includes
!
#ifdef CUDA_GPU
#include "cuda_fft.inc"
#include "cuda_macros.inc"
#include "cuda_profiling.inc"
#endif
