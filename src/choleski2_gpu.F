!#define dotiming
!#define debug
#include "symbol.inc"

MODULE choleski
  USE prec
  USE dfast
  USE dfast_gpu

CONTAINS
!************************ SUBROUTINE ORTHCH ****************************
! RCS:  $Id: choleski2.F,v 1.5 2002/04/16 07:28:38 kresse Exp $
!
! this subroutine orthonormalizes a set of complex (wave-)functions
! using a Choleski-decomposition of the overlap matrix (O = L L^H)
! in conjunction with inversion of the result of this decomposition
! (U --> U^-1). If we have arbitrary functions {|cfin_i>} on input,
! we have first to build up the overlap matrix OVL_ij=<cfin_i|cfin_j>
! then we have to decompose it (OVL_ij=L_ik U_kj), have to invert
! U_ij and then to form the output set |cfout_i>=U^-1_ji|cfin_j>. As
! one can show easily it holds: <cfout_i|cfout_j>=delta_ij !! Voila!
!
!
! GPU part : HACENE Mohamed
!
!***********************************************************************

      SUBROUTINE ORTHCH(WDES,W, LOVERL,LMDIM,CQIJ,NKSTART,NKSTOP)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE scala
      USE wave_high
      USE wave
      USE wave_mpi
      USE gpu_data     
      USE c2f_interface, ONLY : VTIME
          
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavespin) W
      TYPE (wavedes)  WDES
      TYPE (wavedes1) WDES1
      TYPE (gpu_type) GPU
      
      INTEGER, PARAMETER :: STREAM_BLOCK_SIZE = 4
      INTEGER STREAM

      INTEGER, OPTIONAL :: NKSTART,NKSTOP

      INTEGER :: MY_NKSTART,MY_NKSTOP
      LOGICAL   LOVERL
      OVERLAP   CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      GDEF,ALLOCATABLE,TARGET :: CPROW(:,:),COVL(:,:)
! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:)
      GDEF   , POINTER :: CPROW_RED(:,:),CPROJ_RED(:,:)
      LOGICAL DO_REDIS
      TYPE (REDIS_PW_CTR),POINTER :: H_PW
      INTEGER(c_intptr_t) :: GPU_CPROW(STREAM_BLOCK_SIZE)
      INTEGER(c_intptr_t) :: GPU_CFW(STREAM_BLOCK_SIZE)
      
      COMPLEX(q)    fakec
      
      INTEGER :: II

      LOGICAL IS_ALREADY_REDIS   ! are the plane wave coefficients already redistributed
      LOGICAL LASYNC_            ! ansyncronous exchange of pw coefficients
REAL(q) TV,TV0,TC,TC0,TV1,TC1,TimeTrans,TimeLincom,TimeORTH,TimeLapack,TimeInit

nv_profile_start(NVP_ORTHCH)

    TimeTrans  = 0
    TimeORTH   = 0
    TimeLincom = 0
    TimeLapack = 0
    TimeInit   = 0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU=WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NCPU=1
      NODE_ME=0
      IONODE=0
#endif
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
      IF (NCPU /= 1) THEN

        DO_REDIS=.TRUE.
        NRPLWV_RED=WDES%NRPLWV/NCPU
        NPROD_RED =WDES%NPROD /NCPU
        LASYNC_=LASYNC
        ! it is possible that the bands are distributed of plane wave coefficients upon entry
        IS_ALREADY_REDIS=W%OVER_BAND
        ! of course no need to do it in overlap with calculations
        IF (IS_ALREADY_REDIS) LASYNC_=.FALSE.
      ELSE

        DO_REDIS=.FALSE.
        NRPLWV_RED=WDES%NRPLWV
        NPROD_RED =WDES%NPROD

      ENDIF
      NB_TOT=WDES%NB_TOT
      NBANDS=WDES%NBANDS
      NSTRIP=NSTRIP_STANDARD

! allocate work space
      ALLOCATE(CPROW(WDES%NPROD,NBANDS))
nv_profile_start(NVP_ORTHCH_MALLOC)
      CALL cublas_alloc_safety(NB_TOT*NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t),GPU%COVL)
nv_profile_stop(NVP_ORTHCH_MALLOC)

nv_profile_start(NVP_ORTHCH_INIT)
      IF (.NOT. LscaAWARE) THEN
         ALLOCATE(COVL(NB_TOT,NB_TOT))
      ELSE
         CALL INIT_scala(WDES%COMM_KIN, NB_TOT)
         ALLOCATE(COVL(SCALA_NP(), SCALA_NQ()))
      ENDIF

 CALL VTIME(TV0,TC0)
 TOT=0

      MY_NKSTART=1
      MY_NKSTOP=WDES%NKPTS
      IF(PRESENT(NKSTART))MY_NKSTART=NKSTART
      IF(PRESENT(NKSTOP))MY_NKSTOP=NKSTOP
!=======================================================================
      spin:    DO ISP=1,WDES%ISPIN
      kpoints: DO NK=MY_NKSTART,MY_NKSTOP
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      IF (LscaAWARE) CALL INIT_scala(WDES%COMM_KIN,WDES%NB_TOTK(NK,ISP))
      CALL SETWDES(WDES,WDES1,NK)
!   get pointers for redistributed wavefunctions
!   I can not guarantee that this works with all f90 compilers
!   please see comments in wave_mpi.F
      IF (NCPU /= 1) THEN
        CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROW_RED, NPROD_RED, NB_TOT, CPROW(1,1))
      ELSE
        CW_RED    => W%CW(:,:,NK,ISP)
        CPROJ_RED => W%CPROJ(:,:,NK,ISP)
        CPROW_RED => CPROW(:,:)
      ENDIF

!   set number of wavefunctions after redistribution
      NPL = WDES1%NPL
      NPRO= WDES1%NPRO

      CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

      NPRO_O=NPRO
      IF (.NOT. LOVERL) NPRO_O=0
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,4))
#endif
!=======================================================================
!  calculate overlap matrix (only upper triangle is needed):
!=======================================================================
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,5))
#endif
      IF (DO_REDIS .AND. LASYNC_) THEN
         CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PW)
         DO NPOS=1,NSTRIP
           CALL REDIS_PW_START(WDES, W%CW(1,NPOS,NK,ISP), NPOS, H_PW)
        ENDDO
      ENDIF

      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,ISP), W%CPROJ(1,1,NK,ISP),CPROW(1,1))
nv_profile_stop(NVP_ORTHCH_INIT)

     ! redistribute everything
nv_profile_start(NVP_ORTHCH_MALLOC)
     call cublas_alloc_safety (NRPLWV_RED*NB_TOT,int(c_sizeof(CW_RED(1,1)),c_size_t),GPU%CPTWFP)  
nv_profile_stop(NVP_ORTHCH_MALLOC)

nv_profile_start(NVP_ORTHCH_MEMCPY)
      statmg
      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
        stotmgt("RD_PROJ")
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        stotmgt("RD_PROJ")

#ifndef GPUDIRECT
        IF (.NOT. LASYNC_ .AND. .NOT. IS_ALREADY_REDIS)  CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
          call cublas_Set_Matrix(NRPLWV_RED,NB_TOT,int(c_sizeof(CW_RED(1,1)),c_int),c_loc(CW_RED),NRPLWV_RED,GPU%CPTWFP,NRPLWV_RED)
#else
          DO II=1,NBANDS
nv_profile_start(NVP_GPUDIRECT)
            call cuda_alltoall_host_dev(c_loc(W%CW(1,II,NK,ISP)),GPU%CPTWFP+(II-1)*int(c_sizeof(fakec),c_size_t)*WDES%NRPLWV,2*WDES1%NRPLWV, WDES%COMM%NODE_ME-1, WDES%COMM%NCPU, 131072) 
nv_profile_stop(NVP_GPUDIRECT)
          ENDDO          
#endif
        stotmgt("RD_PW  ")
      ELSE
         call cublas_Set_Matrix(NRPLWV_RED,NB_TOT,int(c_sizeof(CW_RED(1,1)),c_int),c_loc(CW_RED),NRPLWV_RED,GPU%CPTWFP,NRPLWV_RED)
      ENDIF
nv_profile_stop(NVP_ORTHCH_MEMCPY)

      IF (.NOT. LscaAWARE) THEN
nv_profile_start(NVP_ORTHCH_COMPUTE)
nv_profile_start(NVP_CUSTOM)
          call cuda_memset(GPU%COVL,0,NB_TOT*NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t))
          !CALL gpu_initz(-1,NB_TOT*NB_TOT,NB_TOT*NB_TOT,GPU%COVL,0,zero)
          CALL THREADSYNCHRONIZE ()
nv_profile_stop(NVP_CUSTOM)
nv_profile_stop(NVP_ORTHCH_COMPUTE)
       ENDIF
      CALL VTIME(TV,TC)   
      TimeInit = TimeInit + TC - TC1
      CALL VTIME(TV1,TC1)

nv_profile_start(NVP_ORTHCH_MALLOC)
      call cublas_alloc_safety (NPROD_RED*NB_TOT,int(c_sizeof(CPROJ_RED(1,1)),c_size_t),GPU%CPROJ)
nv_profile_stop(NVP_ORTHCH_MALLOC)
nv_profile_start(NVP_ORTHCH_MEMCPY)
      call cublas_Set_Matrix(NPROD_RED,NB_TOT,int(c_sizeof(CPROJ_RED(1,1)),c_int),c_loc(CPROJ_RED),NPROD_RED,GPU%CPROJ,NPROD_RED)
nv_profile_stop(NVP_ORTHCH_MEMCPY)
      
      CALL VTIME(TV,TC)   
      TimeTrans = TimeTrans + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,6))
#endif
!
! there is a strange bug in the PII optimized blas DGEMM, which seems
! to access in certain instances data beyond the last array element
! if a matrix is multiplied with a vector (second call to ORTH1)
! to work around this I avoid calling ORTH1 with NB_TOT-NPOS+1=1
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,7))
#endif

nv_profile_start(NVP_ORTHCH_MALLOC)
    do II=1,STREAM_BLOCK_SIZE
        call cublas_alloc_safety (NRPLWV_RED*NSTRIP*NCPU,int(c_sizeof(CW_RED(1,1)),c_size_t),GPU_CFW(II))   
        call cublas_alloc_safety (NPROD_RED*NSTRIP*NCPU,int(c_sizeof(CPROW_RED(1,1)),c_size_t),GPU_CPROW(II))
    end do
nv_profile_stop(NVP_ORTHCH_MALLOC)

nv_profile_start(NVP_ORTHCH_COMPUTE)
nv_profile_start(NVP_GEMM)
      STREAM = 0
      DO NPOS=1,NBANDS-NSTRIP,NSTRIP
        IF (DO_REDIS .AND. LASYNC_) THEN
        DO NPOS_=NPOS,NPOS+NSTRIP-1
          CALL REDIS_PW_STOP (WDES, W%CW(1,NPOS_,NK,ISP), NPOS_, H_PW)
          IF (NPOS_+NSTRIP<=NBANDS) &
          CALL REDIS_PW_START(WDES, W%CW(1,NPOS_+NSTRIP,NK,ISP), NPOS_+NSTRIP, H_PW)
        ENDDO
        ENDIF
        STREAM = MOD(STREAM, STREAM_BLOCK_SIZE)+1

        NPOS_RED  =(NPOS-1)*NCPU+1
        NSTRIP_RED=NSTRIP*NCPU

        CALL VTIME(TV1,TC1)
#ifdef GPUDIRECT
           call cuda_memcpydtod(STREAM,GPU_CFW(STREAM),GPU%CPTWFP+(NPOS_RED-1)*int(c_sizeof(fakec),c_size_t)*NRPLWV_RED,NRPLWV_RED*NSTRIP_RED,c_sizeof(CW_RED(1,1)))
#else
          call cublas_Set_Matrix_Async(STREAM, NRPLWV_RED,NSTRIP_RED,int(c_sizeof(CW_RED(1,1)),c_int),c_loc(CW_RED(1,NPOS_RED)),NRPLWV_RED,GPU_CFW(STREAM),NRPLWV_RED)
#endif
        
        call cublas_Set_Matrix_Async(STREAM, NPROD_RED,NSTRIP_RED,int(c_sizeof(CPROW_RED(1,1)),c_int),c_loc(CPROW_RED(1,NPOS_RED)),NPROD_RED,GPU_CPROW(STREAM),NPROD_RED)  
       
        CALL VTIME(TV,TC)  
        TimeTrans = TimeTrans + TC - TC1
        IF (.NOT. LscaAWARE) THEN
           !CALL ORTH1('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
           !     CPROW_RED(1,NPOS_RED),NB_TOT, &
           !     NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1))
           !TODO: ORTH1_GPU (SHIFT)
           CALL VTIME(TV1,TC1)
           CALL ORTH1_GPU_STREAM(STREAM,'U',GPU%CPTWFP,GPU_CFW(STREAM),GPU%CPROJ, &
                GPU_CPROW(STREAM),NB_TOT, &
                NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,GPU%COVL)
           !CALL THREADSYNCHRONIZE ()
           CALL VTIME(TV,TC)   
           TimeOrth = TimeOrth + TC - TC1           
        ELSE
           write(*,*) "ORTH1_DISTRI_GPU not YET implemented !!!"
           call exit()

           !CALL ORTH1_DISTRI('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
           !     CPROW_RED(1,NPOS_RED),NB_TOT, &
           !     NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1), & 
           !     WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))
        ENDIF
      ENDDO
        CALL THREADSYNCHRONIZE ()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_ORTHCH_COMPUTE)

        CALL CUBLAS_RESET_ST()

nv_profile_start(NVP_ORTHCH_MALLOC)
    do II=1,STREAM_BLOCK_SIZE
        call cublas_free(GPU_CFW(II))
        call cublas_free(GPU_CPROW(II))
    end do
nv_profile_stop(NVP_ORTHCH_MALLOC)
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,7))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,8))
#endif

      IF (DO_REDIS .AND. LASYNC_) THEN
      DO NPOS_=NPOS,NBANDS
          CALL REDIS_PW_STOP (WDES, W%CW(1,NPOS_,NK,ISP), NPOS_, H_PW)
      ENDDO
      ENDIF
      
      NPOS_RED  =(NPOS-1)*NCPU+1
      NSTRIP_RED=(NBANDS-NPOS+1)*NCPU
      CALL VTIME(TV1,TC1)
nv_profile_start(NVP_ORTHCH_MALLOC)
      call cublas_alloc_safety (NRPLWV_RED*NSTRIP_RED,int(c_sizeof(CW_RED(1,1)),c_size_t),GPU%CFW)   
      call cublas_alloc_safety (NPROD_RED*NSTRIP_RED,int(c_sizeof(CPROW_RED(1,1)),c_size_t),GPU%CPROW)
nv_profile_stop(NVP_ORTHCH_MALLOC)
      
nv_profile_start(NVP_ORTHCH_MEMCPY)
      call cublas_Set_Matrix(NPROD_RED,NSTRIP_RED,int(c_sizeof(CPROW_RED(1,1)),c_int),c_loc(CPROW_RED(1,NPOS_RED)),NPROD_RED,GPU%CPROW,NPROD_RED)  
      
#ifdef GPUDIRECT
           call cuda_memcpydtod(NULL_STREAM,GPU%CFW,GPU%CPTWFP+(NPOS_RED-1)*int(c_sizeof(fakec),c_size_t)*NRPLWV_RED, NRPLWV_RED*NSTRIP_RED,c_sizeof(CW_RED(1,1)))
#else
          call cublas_Set_Matrix(NRPLWV_RED,NSTRIP_RED,int(c_sizeof(CW_RED(1,1)),c_int),c_loc(CW_RED(1,NPOS_RED)),NRPLWV_RED,GPU%CFW,NRPLWV_RED)
#endif  
nv_profile_stop(NVP_ORTHCH_MEMCPY)
      
      CALL VTIME(TV,TC)   
      TimeTrans = TimeTrans + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,8))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,9))
    call start_orthchgpu_range()
#endif

      CALL VTIME(TV1,TC1)

      IF (.NOT. LscaAWARE) THEN
nv_profile_start(NVP_ORTHCH_COMPUTE)
nv_profile_start(NVP_GEMM)
         !CALL ORTH1('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
         !     CPROW_RED(1,NPOS_RED),NB_TOT, &
         !     NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1))
          CALL ORTH1_GPU('U',GPU%CPTWFP,GPU%CFW,GPU%CPROJ, &
               GPU%CPROW,NB_TOT, &
               NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,GPU%COVL)
          CALL THREADSYNCHRONIZE ()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_ORTHCH_COMPUTE)
      ELSE
          write(*,*) "ORTH1_DISTRI_GPU not YET implemented !!!"
          call exit()
         !CALL ORTH1_DISTRI('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
         !     CPROW_RED(1,NPOS_RED),NB_TOT, &
         !     NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1), &  
         !     WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))
      ENDIF

      CALL VTIME(TV,TC)   
      TimeOrth = TimeOrth + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,9))
    call end_range()
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,10))
#endif
    
      CALL VTIME(TV1,TC1)
nv_profile_start(NVP_ORTHCH_MALLOC)
      call cublas_free(GPU%CFW)
      call cublas_free(GPU%CPROW)
nv_profile_stop(NVP_ORTHCH_MALLOC)
      
nv_profile_start(NVP_ORTHCH_MEMCPY)
      call cublas_Get_Matrix(NB_TOT,NB_TOT,int(c_sizeof(COVL(1,1)),c_int),GPU%COVL,NB_TOT,c_loc(COVL),NB_TOT)
nv_profile_stop(NVP_ORTHCH_MEMCPY)
       
!      call print_covl_array(COVL(1,1), NB_TOT)
      CALL VTIME(TV,TC)   
      TimeTrans = TimeTrans + TC - TC1   
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,10))
#endif
      
      IF (DO_REDIS .AND. LASYNC_) CALL REDIS_PW_DEALLOC(H_PW)

      stotmgt("IN_PROD")
      IF (.NOT. LscaAWARE) THEN
         CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))
      ENDIF
      stotmgt("GLB_SUM")

#ifdef debug
      io_begin
      CALL DUMP_HAM( "overlap",WDES, COVL)
      io_end
#endif
!=======================================================================
! Choleski-decomposition of the overlap matrix + inversion of the result
! calling LAPACK-routines ZPOTRF (decomposition) and ZTRTRI (inversion):
!=======================================================================
nv_profile_start(NVP_ORTHCH_CHOLESKI)
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,11))
#endif

      CALL VTIME(TV1,TC1)
      IF (.NOT. LscaAWARE) THEN
         IF (LscaLAPACK .AND. LscaLU ) THEN
            CALL pPOTRF_TRTRI(WDES%COMM_KIN, COVL(1,1),NB_TOT, WDES%NB_TOTK(NK,ISP))
            CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1), NB_TOT*NB_TOT))
         ELSE
            INFO=0
#ifdef gammareal
            CALL DPOTRF &
#else
            CALL ZPOTRF &
#endif
                 & ('U',WDES%NB_TOTK(NK,ISP),COVL(1,1),NB_TOT,INFO)
            IF (INFO/=0) THEN
               WRITE(*,*) 'LAPACK: Routine ZPOTRF failed!',INFO,NK,ISP
               CALLMPI ( M_stop('LAPACK: Routine ZPOTRF failed!') )
               STOP
            ENDIF
#ifdef gammareal
            CALL DTRTRI &
#else
            CALL ZTRTRI &
#endif
                 & ('U','N',WDES%NB_TOTK(NK,ISP),COVL(1,1),NB_TOT,INFO)
            IF (INFO/=0) THEN
               WRITE(*,*) 'LAPACK: Routine ZTRTRI failed!',INFO,NK,ISP
               CALLMPI ( M_stop('LAPACK: Routine ZTRTRI failed!') )
               STOP
            ENDIF
         ENDIF
      ELSE
         CALL BG_pPOTRF_TRTRI(COVL(1,1), WDES%NB_TOTK(NK,ISP), INFO)
      ENDIF
      CALL VTIME(TV,TC)   
      TimeLapack = TimeLapack + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,11))
#endif

      stotmgt("ZPOTRF")
nv_profile_stop(NVP_ORTHCH_CHOLESKI)
!=======================================================================
!  construct the orthogonal set:
!=======================================================================
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,12))
#endif

      CALL VTIME(TV1,TC1)
nv_profile_start(NVP_ORTHCH_MEMCPY)
      CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(COVL(1,1)),c_int),c_loc(COVL),NB_TOT,GPU%COVL,NB_TOT)
nv_profile_stop(NVP_ORTHCH_MEMCPY)
      CALL VTIME(TV,TC)   
      TimeTrans = TimeTrans + TC - TC1

      CALL VTIME(TV1,TC1)

      IF (.NOT. LscaAWARE) THEN
nv_profile_start(NVP_ORTHCH_LINCOM)
         !CALL LINCOM('U',CW_RED,CPROJ_RED,COVL(1,1), &
         !     WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP),NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
         !     CW_RED,CPROJ_RED)
          CALL LINCOM_GPU('U',GPU%CPTWFP,GPU%CPROJ,GPU%COVL, &
               WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP),NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
               GPU%CPTWFP,GPU%CPROJ)
          CALL THREADSYNCHRONIZE ()
nv_profile_stop(NVP_ORTHCH_LINCOM)
      ELSE
         !CALL LINCOM_DISTRI('U',CW_RED(1,1),CPROJ_RED(1,1),COVL(1,1), &
         !     WDES%NB_TOTK(NK,ISP),NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, & 
         !     WDES%COMM_KIN, NBLK)
          write(*,*) "LINCOM_DISTRI_GPU not YET implemented !!!"
          call exit()
      ENDIF
      CALL VTIME(TV,TC)   
      TimeLincom = TimeLincom + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,12))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,13))
#endif

      CALL VTIME(TV1,TC1)
nv_profile_start(NVP_ORTHCH_MEMCPY)
      IF (NPRO/=0) THEN
         CALL cublas_Get_Matrix(NPROD_RED,NB_TOT,int(c_sizeof(CPROJ_RED(1,1)),c_int),GPU%CPROJ,NPROD_RED,c_loc(CPROJ_RED),NPROD_RED)
      ENDIF
      CALL cublas_Get_Matrix(NB_TOT,NB_TOT,int(c_sizeof(COVL(1,1)),c_int),GPU%COVL,NB_TOT,c_loc(COVL),NB_TOT)
nv_profile_stop(NVP_ORTHCH_MEMCPY)

#ifndef GPUDIRECT
nv_profile_start(NVP_ORTHCH_MEMCPY)
  CALL cublas_Get_Matrix(NRPLWV_RED,NB_TOT,int(c_sizeof(CW_RED(1,1)),c_int),GPU%CPTWFP,NRPLWV_RED,c_loc(CW_RED),NRPLWV_RED)
nv_profile_stop(NVP_ORTHCH_MEMCPY)

nv_profile_start(NVP_ORTHCH_MALLOC)
  call cublas_free(GPU%CPTWFP)
nv_profile_stop(NVP_ORTHCH_MALLOC)
#endif  

nv_profile_start(NVP_ORTHCH_MALLOC)
      call cublas_free(GPU%CPROJ)
nv_profile_stop(NVP_ORTHCH_MALLOC)

      CALL VTIME(TV,TC)   
      TimeTrans = TimeTrans + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,13))
#endif

      stotmgt("LINCOM")
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,14))
#endif

     !  back redistribution
      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        IF (LASYNC_ .OR. IS_ALREADY_REDIS) THEN
           W%OVER_BAND=.TRUE.
        ELSE
           ! if the routine was entered with already redistributed wave-functions
           ! then no need to redistribute them back
           !CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
#ifndef GPUDIRECT !NO GPU DIRECT
            CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))          
#else  
            DO II=1,NBANDS
nv_profile_start(NVP_GPUDIRECT)
              call cuda_alltoall_dev_host(GPU%CPTWFP+(II-1)*int(c_sizeof(fakec),c_size_t)*WDES%NRPLWV,c_loc(W%CW(1,II,NK,ISP)),2*WDES1%NRPLWV, WDES%COMM%NODE_ME-1, WDES%COMM%NCPU, 131072)
nv_profile_stop(NVP_GPUDIRECT)
            ENDDO
nv_profile_start(NVP_ORTHCH_MALLOC)
             call cublas_free(GPU%CPTWFP)
nv_profile_stop(NVP_ORTHCH_MALLOC)
#endif
        ENDIF
      ELSE
#ifdef GPUDIRECT
          CALL cublas_Get_Matrix(NRPLWV_RED,NB_TOT,sizeof(CW_RED(1,1)),GPU%CPTWFP,NRPLWV_RED,CW_RED,NRPLWV_RED)
          call cublas_free(GPU%CPTWFP)
#endif  
      ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,14))
#endif
      stotmgt("RD_PW  ")
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,3))
#endif
      DEALLOCATE(CPROW,COVL)
nv_profile_start(NVP_ORTHCH_MALLOC)
      CALL cublas_free(GPU%COVL)
nv_profile_stop(NVP_ORTHCH_MALLOC)

 CALL VTIME(TV,TC)
#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
       write(*,*) "Time ORTHCH.....................................=",TC-TC0
       write(*,*) "       Funtion Name         ","      Time (s)        ","      percentage"
       write(*,*) "Time Initialize............=",TimeInit,nint(TimeInit/(TC-TC0)*100)
       write(*,*) "Time Lincom................=",TimeLincom,nint(TimeLincom/(TC-TC0)*100)
       write(*,*) "Time Orth..................=",TimeOrth,nint(TimeOrth/(TC-TC0)*100)
       write(*,*) "Time Lapack................=",TimeLapack,nint(TimeLapack/(TC-TC0)*100)
       write(*,*) "Time Transfer..............=",TimeTrans,nint(TimeTrans/(TC-TC0)*100)
       write(*,*) "Total percent..............=",nint((TimeInit+TimeLapack+TimeLincom+TimeOrth+TimeTrans)/(TC-TC0)*100)
       write(*,*) "######################################################################"
#ifdef MPI
 endif
#endif
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,1))
    call end_range()
#endif

nv_profile_stop(NVP_ORTHCH)
      RETURN
    END SUBROUTINE ORTHCH


!************************ SUBROUTINE ORTHCH ****************************
! RCS:  $Id: choleski2.F,v 1.5 2002/04/16 07:28:38 kresse Exp $
!
! this subroutine determines a dual set of orbitals with the
! property 
!  < phi_i | phi_j> = delta_ij
! the dual set W_DUAL is passed down as a pointer to a wavespin variable
! if NC potential are used W_DUAL is simply linked to W
! otherwise if W_DUAL is not associated, the required arrays are
! allocated and determined (make sure to NULLIFY W_DUAL before 
!   calling the routined)
! if it is already associated, the routine recalculates W_DUAL from W but
! assumes that the proper allocation was done before
!
!***********************************************************************

    SUBROUTINE ORTHCH_DUAL(WDES,W,W_DUAL, LOVERL,LMDIM,CQIJ)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE scala
      USE wave_high
      USE wave
      USE wave_mpi
      IMPLICIT NONE

      TYPE (wavespin), TARGET  ::  W
      TYPE (wavespin), POINTER ::  W_DUAL
      TYPE (wavedes)  WDES
      TYPE (wavedes1) WDES1

      LOGICAL   LOVERL
      INTEGER   LMDIM
      OVERLAP   CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      GDEF,ALLOCATABLE,TARGET :: COVL(:,:)
! redistributed plane wave coefficients
      TYPE (wavefuna)    WA             ! array pointers
      TYPE (wavefuna)    WA_DUAL        ! array pointers
      INTEGER   NODE_ME, IONODE, NCPU
      INTEGER   NB_TOT, NBANDS, NSTRIP, NSTRIP_ACT, ISP, NK, N, I, INFO, NPOS, NPOS_RED, NSTRIP_RED
! scaAWARE is often a little bit tricky to implement
! routine was tested and currenty LscaAWARE_LOCAL is set to LscaAWARE
! see below
      LOGICAL :: LscaAWARE_LOCAL=.FALSE.
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU=WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NCPU=1
      NODE_ME=0
      IONODE=0
#endif
      IF (.NOT. LOVERL) THEN
         W_DUAL => W
         RETURN
      ENDIF

! if there are any subspected problems with duality comment out
! the following line (or set LscaAWARE_LOCAL =.FALSE.)
      LscaAWARE_LOCAL=LscaAWARE

      IF (.NOT. ASSOCIATED(W_DUAL)) THEN
         ALLOCATE(W_DUAL)
         CALL ALLOCW( WDES, W_DUAL)
!#define test_w_dual
#ifdef test_w_dual
         W_DUAL%CPROJ=W%CPROJ
         W_DUAL%CW   =W%CW
         W_DUAL%CELTOT=W%CELTOT
         W_DUAL%FERTOT=W%FERTOT
#endif
      ENDIF

#ifndef test_w_dual
      W_DUAL%CPROJ=W%CPROJ
      W_DUAL%CW   =W%CW
      W_DUAL%CELTOT=W%CELTOT
      W_DUAL%FERTOT=W%FERTOT
#endif
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
      NB_TOT=WDES%NB_TOT
      NBANDS=WDES%NBANDS
      NSTRIP=NSTRIP_STANDARD

      IF (.NOT. LscaAWARE_LOCAL) THEN
         ALLOCATE(COVL(NB_TOT,NB_TOT))
      ELSE
         CALL INIT_scala(WDES%COMM_KIN, NB_TOT)
         ALLOCATE(COVL(SCALA_NP(), SCALA_NQ()))
      ENDIF
!=======================================================================
      spin:    DO ISP=1,WDES%ISPIN
      kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      IF (LscaAWARE_LOCAL) CALL INIT_scala(WDES%COMM_KIN,WDES%NB_TOTK(NK,ISP))

      CALL SETWDES(WDES,WDES1,NK)
      WA     =ELEMENTS(W, WDES1, ISP)
      WA_DUAL=ELEMENTS(W_DUAL, WDES1, ISP)
!=======================================================================
!  calculate overlap matrix (only upper triangle is needed):
!=======================================================================

      statmg
      IF (WDES%DO_REDIS) THEN
         CALL REDISTRIBUTE_PROJ(WA)
         CALL REDISTRIBUTE_PW(WA)
         CALL REDISTRIBUTE_PROJ(WA_DUAL)
         CALL REDISTRIBUTE_PW(WA_DUAL)
      ENDIF

      IF (.NOT. LscaAWARE_LOCAL) THEN
         DO N=1,NB_TOT
            DO I=1,NB_TOT
               COVL(I,N)=(0._q,0._q)
            ENDDO
         ENDDO
      ENDIF
!
      DO NPOS=1,NBANDS,NSTRIP
        NSTRIP_ACT=MIN(WDES%NBANDS+1-NPOS,NSTRIP)
        NPOS_RED  =(NPOS-1)*NCPU+1
        NSTRIP_RED=NSTRIP_ACT*NCPU

        IF (.NOT. LscaAWARE_LOCAL) THEN
           ! W^+ x W_DUAL (W_DUAL=W usually)
           CALL ORTH1('U',WA%CW_RED(1,1),WA_DUAL%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
                WA_DUAL%CPROJ_RED(1,1),NB_TOT, &
                NPOS_RED,NSTRIP_RED, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
        ELSE
           CALL ORTH1_DISTRI('U',WA%CW_RED(1,1),WA_DUAL%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
                WA_DUAL%CPROJ_RED(1,1),NB_TOT, &
                NPOS_RED,NSTRIP_RED, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1), & 
                WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))
        ENDIF
      ENDDO
      
      stotmgt("IN_PROD")
      IF (.NOT. LscaAWARE_LOCAL) THEN
         CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))
      ENDIF
      stotmgt("GLB_SUM")

#if defined(debug) || defined test_w_dual
      io_begin
      CALL DUMP_HAM( "overlap",WDES, COVL)
      io_end
#endif
!=======================================================================
! Choleski-decomposition of the overlap matrix + inversion of the result
! calling LAPACK-routines ZPOTRF (decomposition) and ZPOTRI (inversion):
!=======================================================================
      IF (.NOT. LscaAWARE_LOCAL) THEN
         IF (LscaLAPACK .AND. LscaLU ) THEN
            CALL pPOTRF_POTRI(WDES%COMM_KIN, COVL(1,1),NB_TOT, WDES%NB_TOTK(NK,ISP))
            CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1), NB_TOT*NB_TOT))
         ELSE
            INFO=0
#ifdef gammareal
            CALL DPOTRF &
#else
            CALL ZPOTRF &
#endif
                 & ('U',WDES%NB_TOTK(NK,ISP),COVL(1,1),NB_TOT,INFO)
            IF (INFO/=0) THEN
               WRITE(*,*) 'LAPACK: Routine ZPOTRF failed!',INFO,NK,ISP
               CALLMPI ( M_stop('LAPACK: Routine ZPOTRF failed!') )
               STOP
            ENDIF
#ifdef gammareal
            CALL DPOTRI &
#else
            CALL ZPOTRI &
#endif
                 & ('U',WDES%NB_TOTK(NK,ISP),COVL(1,1),NB_TOT,INFO)
            IF (INFO/=0) THEN
               WRITE(*,*) 'LAPACK: Routine ZTRTRI failed!',INFO,NK,ISP
               CALLMPI ( M_stop('LAPACK: Routine ZTRTRI failed!') )
               STOP
            ENDIF
         ENDIF
         ! add lower triangle not calculated by POTRI
         DO N=1,NB_TOT
            DO I=N+1,NB_TOT
               COVL(I,N)=GCONJG(COVL(N,I))
            ENDDO
         ENDDO
      ELSE
         CALL BG_pPOTRF_POTRI(COVL(1,1), WDES%NB_TOTK(NK,ISP), INFO)
      ENDIF
#ifdef debug
      io_begin
      CALL DUMP_HAM( "transformation",WDES, COVL)
      io_end
#endif

      stotmgt("ZPOTRF")
!=======================================================================
!  construct the dual set by inplace transformation of W_DUAL
!=======================================================================
      IF (.NOT. LscaAWARE_LOCAL) THEN
         CALL LINCOM('F',WA_DUAL%CW_RED,WA_DUAL%CPROJ_RED,COVL(1,1), &
              WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
              WDES1%NPL_RED, WDES1%NPRO_RED, WDES1%NRPLWV_RED, WDES1%NPROD_RED, NB_TOT, &
              WA%CW_RED, WA%CPROJ_RED)
      ELSE
         ! POTRF set only the upper triangle of the transformation matrix
         ! add the lower triangle using DISTRI_HERM
         CALL LINCOM_DISTRI_HERM('F',WA_DUAL%CW_RED(1,1),WA_DUAL%CPROJ_RED(1,1),COVL(1,1), &
              WDES%NB_TOTK(NK,ISP), & 
              WDES1%NPL_RED, WDES1%NPRO_RED, WDES1%NRPLWV_RED, WDES1%NPROD_RED, NB_TOT, & 
              WDES%COMM_KIN, NBLK)
      ENDIF

      stotmgt("LINCOM")

     !  back redistribution
      IF (WDES%DO_REDIS) THEN
        CALL REDISTRIBUTE_PROJ(WA)
        CALL REDISTRIBUTE_PW  (WA)
        CALL REDISTRIBUTE_PROJ(WA_DUAL)
        CALL REDISTRIBUTE_PW  (WA_DUAL)
      ENDIF
      stotmgt("RD_PW  ")
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================
      DEALLOCATE(COVL)

      RETURN
      END SUBROUTINE ORTHCH_DUAL


!************************ SUBROUTINE DUMP_S ****************************
!
! dump the overlap matrix between the occupied states
! this is essentially identical to the previous routine
!
!***********************************************************************


      SUBROUTINE DUMP_S(WDES,W, LOVERL,LMDIM,CQIJ)
      USE prec
      USE scala
      USE wave_high
      USE wave
      USE wave_mpi
      USE dfast
      USE dfast_gpu
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavespin) W
      TYPE (wavedes)  WDES
      TYPE (wavedes1) WDES1

      LOGICAL   LOVERL
      OVERLAP   CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      GDEF,ALLOCATABLE,TARGET :: CPROW(:,:),COVL(:,:)

      ! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:)
      GDEF   , POINTER :: CPROW_RED(:,:),CPROJ_RED(:,:)
      LOGICAL DO_REDIS
      REAL(q) :: NELECT
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU=WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NCPU=1
      NODE_ME=0
      IONODE=0
#endif
      IF (NCPU /= 1) THEN

         DO_REDIS=.TRUE.
         NRPLWV_RED=WDES%NRPLWV/NCPU
         NPROD_RED =WDES%NPROD /NCPU
      ELSE
         DO_REDIS=.FALSE.
         NRPLWV_RED=WDES%NRPLWV
         NPROD_RED =WDES%NPROD
      ENDIF

      NB_TOT=WDES%NB_TOT
      NBANDS=WDES%NBANDS

      NSTRIP=NSTRIP_STANDARD

! allocate work space
      ALLOCATE(CPROW(WDES%NPROD,NBANDS),COVL(NB_TOT,NB_TOT))

      NELECT=0

      spin:    DO ISP=1,WDES%ISPIN
      kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES1,NK)
! get pointers for redistributed wavefunctions
      IF (NCPU /= 1) THEN
         CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
         CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
         CALL SET_GPOINTER(CPROW_RED, NPROD_RED, NB_TOT, CPROW(1,1))
      ELSE
         CW_RED    => W%CW(:,:,NK,ISP)
         CPROJ_RED => W%CPROJ(:,:,NK,ISP)
         CPROW_RED => CPROW(:,:)
      ENDIF

! set number of wavefunctions after redistribution
      NPL = WDES1%NPL
      NPRO= WDES1%NPRO

      CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

      NPRO_O=NPRO
      IF (.NOT. LOVERL) NPRO_O=0

      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,ISP), W%CPROJ(1,1,NK,ISP),CPROW(1,1))

! redistribute everything
      IF (DO_REDIS) THEN
         CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
         CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
         CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
      ENDIF

! calculate overlap
      DO N=1,NB_TOT
         DO I=1,NB_TOT
            COVL(I,N)=(0._q,0._q)
         ENDDO
      ENDDO

      DO NPOS=1,NBANDS-NSTRIP,NSTRIP
         NPOS_RED  =(NPOS-1)*NCPU+1
         NSTRIP_RED=NSTRIP*NCPU

         CALL ORTH1('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
              CPROW_RED(1,NPOS_RED),NB_TOT, &
              NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1))
      ENDDO

      
      NPOS_RED  =(NPOS-1)*NCPU+1
      NSTRIP_RED=(NBANDS-NPOS+1)*NCPU

      CALL ORTH1('U',CW_RED(1,1),CW_RED(1,NPOS_RED),CPROJ_RED(1,1), &
           CPROW_RED(1,NPOS_RED),NB_TOT, &
           NPOS_RED,NSTRIP_RED,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,COVL(1,1))

      DO N=1,NB_TOT
         NELECT=NELECT+ COVL(N, N)*W%FERWE(N,NK,ISP)*WDES%RSPIN*WDES%WTKPT(NK)
      ENDDO

      CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))

! back redistribution
      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
      ENDIF

      WRITE(*,*) 'NK=',NK,'ISP=',ISP
      NPL2=MIN(10,NB_TOT)
      DO  N1=1,NPL2
         WRITE(*,1)N1,(REAL( COVL(N1,N2) ,KIND=q) ,N2=1,NPL2)
      ENDDO
      WRITE(*,*)
#ifndef gammareal
      DO N1=1,NPL2
         WRITE(*,2)N1,(AIMAG(COVL(N1,N2)),N2=1,NPL2)
      ENDDO
#endif
      WRITE(*,*)
      WRITE(*,'(20E10.3)') 0,(REAL( COVL(N1,N1)-1 ,KIND=q) ,N1=1,NPL2)
      WRITE(*,*)
1     FORMAT(1I2,3X,20F9.5)
2     FORMAT(1I2,3X,20E9.1)

      ENDDO kpoints
      ENDDO spin

      WRITE(*,'(A,F14.8)') 'total number of electrons NELECT=',NELECT
      DEALLOCATE(CPROW,COVL)

      RETURN
      END SUBROUTINE
      END MODULE
