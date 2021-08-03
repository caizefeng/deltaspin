#include "symbol.inc"
!#define USE_ZHEEVX

MODULE david
CONTAINS
!************************ SUBROUTINE EDDDAV *****************************
! RCS:  $Id: davidson.F,v 1.5 2003/06/27 13:22:15 kresse Exp kresse $
!
! this subroutine performes a Davidson like optimsation of the
! wavefunctions i.e. it the expectation value
!     < phi | H |  phi >
! for NSIM bands in parallel
! 
! different preconditioners can be chosen using INFO%IALGO
!  INFO%IALGO   determine type of preconditioning and the algorithm
!    6    rms-minimization          +  TAP preconditioning
!    7    rms-minimization          +  no preconditioning
!    8    precond rms-minimization  +  TAP preconditioning
!    9    precond rms-minimization  +  Jacobi like preconditioning
!    (TAP Teter Alan Payne)
!  WEIMIN  treshhold for total energy minimisation
!    is the fermiweight of a band < WEIMIN,
!    minimisation will break after a maximum of two iterations
!  EBREAK  absolut break condition
!    intra-band minimisation is stopped if DE is < EBREAK
!
!
! GPU part : HACENE Mohamed
!***********************************************************************

  SUBROUTINE EDDAV(HAMILTONIAN, P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
       LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV, EXHF, IU6, IU0, LDELAY, LSUBROTI, LEMPTY, LHF, NKSTART, EXHF_ACFDT)
    USE iso_c_binding
    USE cuda_interface
    USE prec
    USE wave_high
    USE base
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE nonl_high_gpu
    USE hamil_high
    USE hamil_gpu
    USE constant
    USE scala
    USE fock
    USE pseudo
    USE dfast
    USE dfast_gpu
    USE gpu_data
    USE pead
    IMPLICIT NONE

    TYPE (ham_handle)  HAMILTONIAN
    TYPE (potcar)      P(:)
    TYPE (grid_3d)     GRID          ! descriptor for FFT grids
    TYPE (info_struct) INFO          ! INFO structure of VASP
    TYPE (latt)        LATT_CUR      !  
    TYPE (nonlr_struct) NONLR_S      ! descriptor for non local part of PP (real space)
    TYPE (nonl_struct) NONL_S        ! descriptor for non local part of PP (reciprocal space)
    TYPE (wavespin)    W             ! array for wavefunction
    TYPE (wavedes)     WDES          ! descriptor for wavefunction
    INTEGER NSIM                     ! simultaneously optimised bands
    INTEGER LMDIM                    ! dimension of arrays CQIJ and CDIJ
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    REAL (q)           RMS           ! on return: norm of residual vector summed over all bands
    REAL (q)           DESUM         ! on return: change of eigenvalues
    INTEGER            ICOUEV        ! number of intraband eigen value minimisations
    RGRID, TARGET ::   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    INTEGER            IU6           ! stdout
    INTEGER            IU0           ! sterr
    LOGICAL LDELAY                   ! delay phase (not used) 
    LOGICAL LSUBROTI                 ! perform subspace rotation
    LOGICAL LEMPTY                   ! optimize emtpy bands only
    LOGICAL, OPTIONAL :: LHF         ! calculate HF contributions
    INTEGER, OPTIONAL :: NKSTART     ! start k-point
    REAL(q),OPTIONAL :: EXHF_ACFDT
 ! local work arrays
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    TYPE (wavefuna)    WA            ! wavefunction array, points to W section for one k and spin
    TYPE (wavefuna)    WOPT          ! subset of wavefunctions currently optimized
    TYPE (wavefuna)    WHAM          ! action of H on these wavefunctions
    TYPE (wavefuna)    WS            ! stores CQIJ * wavefunctions
    TYPE (wavefuna)    WHF           ! HF part of Hamiltonian
    TYPE (wavespin)    W_ORIG        ! stores the original not redistributed wavefunctions
                                     ! required for HF
    GDEF, ALLOCATABLE, TARGET      ::   CORTHO(:,:) ! stores <psi_i| S |psi_k>
    INTEGER(c_intptr_t) GPU_CORTHO
    TYPE (wavefun1)    W1(NSIM)      ! wavefunction currently added to subspace
    ! redistributed plane wave coefficients

    GDEF,ALLOCATABLE, TARGET:: CHAM(:,:),COVL(:,:),CEIG(:,:),COVL_(:,:)
    REAL(q) :: EVALUE_INI(NSIM)      ! eigenvalue of that band at the beginning
    REAL(q) :: EVALUE_GLBL(NSIM)     ! same as eigenvalue but global
    REAL(q) :: EVALUE_INI_GLBL(NSIM) ! same as eigenvalue but global
    REAL(q) :: DEIT                  ! relative break criterion for that band
    INTEGER :: IT(NSIM)              ! current iteration for this band
    REAL(q) :: TRIAL(NSIM)           ! trial step for each band
    LOGICAL :: LSTOP                 ! optimisation finished
    LOGICAL :: LSUBROT               ! usually LSUBROTI
    LOGICAL :: DO_REDIS              ! redistribution of wavefunctions required
    ! nbands times nbands hamilton matrix and overlap matrix
    GDEF,ALLOCATABLE,TARGET::  CHAM_ALL(:,:),COVL_ALL(:,:)

!***********************************************************************
! TODO: remove use of gpu_type
! device pointers (i.e. integers store addresses to device memory)
!***********************************************************************
    INTEGER(c_intptr_t) GPU_CHAM_ALL
    INTEGER(c_intptr_t) GPU_W1_CW(NSIM),GPU_W1_CR(NSIM)
    INTEGER(c_intptr_t) GPU_W1_CW_ALL,GPU_W1_CR_ALL
    INTEGER(c_intptr_t) GPU_CWORK(NSIM),GPU_CWORK_ALL
    INTEGER(c_intptr_t) GPU_PTRS(NSIM)  ! array of device pointers
    INTEGER(c_intptr_t) GPU_WHAM_CW(NSIM*MAX(INFO%NDAV+1,2))  ! array of device pointers
    INTEGER(c_intptr_t) GPU_WHAM_CW_RED(NSIM*MAX(INFO%NDAV+1,2))
    INTEGER(c_intptr_t) GPU_WHAM_CW_ALL
    INTEGER(c_intptr_t) GPU_WOPT_CW_ALL,GPU_WOPT_CPROJ_ALL
    INTEGER(c_intptr_t) GPU_WOPT_CW(NSIM*MAX(INFO%NDAV+1,2))  ! array of device pointers
    INTEGER(c_intptr_t) GPU_WOPT_CW_RED(NSIM*MAX(INFO%NDAV+1,2))
    INTEGER(c_intptr_t) GPU_WOPT_CPROJ(NSIM*MAX(INFO%NDAV+1,2))  ! array of device pointers
    INTEGER(c_intptr_t) GPU_PRECON
    TYPE(gpu_type) GPU(NSIM)
    TYPE(gpu_type) GPU_WOPT, GPU_WA, GPU_WS

    ! redistributed plane wave coefficients
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER  :: LWORK=32
    GDEF,ALLOCATABLE    :: CWRK(:)
    REAL(q),ALLOCATABLE,TARGET ::  R(:)
    REAL(q),ALLOCATABLE ::  RWORK(:)
    INTEGER :: NB_MAX, MY_NKSTART
    LOGICAL :: LHFCALC_DAV
#ifndef USE_ZHEEVX
    INTEGER,PARAMETER :: IRWORK=3
#else
    INTEGER,PARAMETER :: IRWORK=7
    INTEGER,ALLOCATABLE :: IWORK(:), MINFO(:)
#endif
    REAL (q)   :: ABSTOL=1E-10_q, VL, VU, RCOND
    INTEGER    :: IL, IU, NB_CALC
    ! more local variables
    INTEGER :: pos_in_gpu_cfw, shift_cfw ! these variable remember row of
                                         ! GPU(1)%CFW which will be updated in the current davidson iteration
    INTEGER :: NODE_ME, NODE_MEI, IONODE, NCPU, NSIM_LOCAL, NSIM_, NSIM_LOCAL_, &
         NSUBD, NITER, I, NBANDS, NB_TOT, ISP, NK, NB_START, &
         NB_DONE, NP, N, ITER, NPP, M, MM,  IDUMP, &
         ISPINOR, N1, N2, NPOS_RED, IFAIL, II, NITER_NOW, IPREC, J, &
         size_of_cham_all ,iter_bands, max_bands
    REAL(q) :: SLOCAL, DE_ATT, EKIN, FAKT, X, X2, FNORM(NSIM), FPRE_(NSIM), DECEL, DEMAX, WSCAL(NSIM)
    COMPLEX(q) :: WSCALZ(NSIM)
    COMPLEX(q) :: CPT, CDCHF
    REAL(q) :: EXHF
    REAL(q) faketime,faker
    COMPLEX(q) zero_gpu,fakec
    CHARACTER(LEN = 1) :: NODE_MEC
    INTEGER :: SID,NB

nv_profile_start(NVP_EDDAV)

!***********************************************************************
! INIT section
! ------------
! initialization
!***********************************************************************
nv_profile_start(NVP_EDDAV_INIT)
    zero_gpu=0
    pos_in_gpu_cfw = 0
    faketime=0
    ! initialize the required variables for mpi
#ifdef MPI
    NODE_ME=WDES%COMM_KIN%NODE_ME
    IONODE =WDES%COMM_KIN%IONODE
    NODE_MEI=WDES%COMM_INTER%NODE_ME ! number of groups (each group holds one band)
    NCPU   =WDES%COMM_INTER%NCPU     ! number of groups (each group holds one band)
#else
    NODE_ME=1
    NODE_MEI=1
    IONODE =1
    NCPU=1
#endif
!DDEBUG
!write(*,*) "NODE_ME=",NODE_ME
!write(*,*) NODE_ME,"IONODE=",IONODE
!write(*,*) NODE_ME,"NODE_MEI=",NODE_MEI
!write(*,*) NODE_ME,"NCPU=",NCPU
!write(NODE_MEC,'(I1)') NODE_ME

!=======================================================================
! number of bands treated simultaneously this must be a multiple of NCPU
!=======================================================================
    NSIM_LOCAL=NSIM/NCPU  ! number of bands optimised on this node
    IF (NSIM_LOCAL*NCPU /= NSIM) THEN
       WRITE(*,*) 'internal ERROR in EDDAV NSIM is not correct',NSIM
       STOP
    ENDIF

    LSUBROT=LSUBROTI
    IF (NSIM>=WDES%NB_TOT) THEN
       LSUBROT=.FALSE.
    ENDIF

    NITER =MAX(INFO%NDAV+1,2) ! maximum number of iterations

    ! at least one optimisation step
    NSUBD =NITER*NSIM         ! maximum size of the subspace
    NB_MAX=NSUBD

    IF (LSUBROT) NB_MAX=MAX(NB_MAX,WDES%NB_TOT)

    CALL SETWDES(WDES,WDES1,0)

    CALL NEWWAVA(WOPT, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA(WHAM, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA_PROJ(WS, WDES1, NSIM_LOCAL)

    ALLOCATE(CHAM(NSUBD,NSUBD),COVL(NSUBD,NSUBD),CEIG(NSUBD,NSUBD),COVL_(NSUBD,NSUBD), &
    CORTHO(WDES%NB_TOT,NSIM),R(NB_MAX),RWORK(NB_MAX*IRWORK),CWRK(LWORK*WDES%NB_TOT))

    CALL cublas_Alloc_safety(NSUBD*NSUBD,int(c_sizeof(CHAM(1,1)),c_size_t),GPU(1)%CHAM)
    CALL cublas_Alloc_safety(NSUBD*NSUBD,int(c_sizeof(COVL(1,1)),c_size_t),GPU(1)%COVL)
    CALL cublas_Alloc_safety(WDES%NB_TOT*NSIM,int(c_sizeof(CORTHO(1,1)),c_size_t),GPU_CORTHO)
    LHFCALC_DAV=.FALSE.
    IF (PRESENT(LHF)) THEN
       IF (LHF .AND.  LHFCALC) THEN
          LHFCALC_DAV=.TRUE.
          CALL NEWWAVA(WHF, WDES1, NSIM_LOCAL)
          CALL ALLOCW(WDES,W_ORIG)
          W_ORIG%CW       =W%CW
          W_ORIG%CPROJ    =W%CPROJ
          W_ORIG%CELTOT   =W%CELTOT
          W_ORIG%FERTOT   =W%FERTOT
          W_ORIG%OVER_BAND=W%OVER_BAND
       ENDIF
    ENDIF

    DESUM =0
    RMS   =0
    ICOUEV=0

    ! average local potential
    SLOCAL=0
    DO I=1,GRID%RL%NP
       SLOCAL=SLOCAL+SV(I,1)
    ENDDO

    CALLMPI( M_sum_d(WDES%COMM_INB, SLOCAL, 1))
    SLOCAL=SLOCAL/GRID%NPLWV

    DO I=1,NSIM_LOCAL
       CALL NEWWAV(W1(I) , WDES1,.TRUE.)
    ENDDO

    COVL=0
    CHAM=0
    EXHF=0

    IF (PRESENT(EXHF_ACFDT)) EXHF_ACFDT=0
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS

    ! allocate array for the subspace diagonalisation
    IF (LSUBROT) THEN
       IF (.NOT. LscaAWARE) THEN
          ALLOCATE(CHAM_ALL(NB_TOT,NB_TOT))
          call cublas_Alloc_safety (NB_TOT*NB_TOT,int(c_sizeof(fakec),c_size_t),GPU_CHAM_ALL)
          size_of_cham_all = NB_TOT*NB_TOT
       ELSE
          CALL INIT_scala(WDES%COMM_KIN, NB_TOT)
          ALLOCATE(CHAM_ALL(SCALA_NP(),SCALA_NQ()))
       ENDIF
    ENDIF
#ifdef  USE_ZHEEVX
    IF (LSUBROT) ALLOCATE(IWORK(5*WDES%NB_TOT),MINFO(WDES%NB_TOT))
#endif
    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ELSE
       MY_NKSTART=1
    ENDIF

!***********************************************************************
! allocate SV on device
! copy SV, from host to device
!***********************************************************************
    call cublas_Alloc_safety (DIMREAL(GRID%MPLWV)*WDES%NCDIJ,int(c_sizeof(SV(1,1)),c_size_t),GPU(1)%SV)
    call cublas_Set_Vector (DIMREAL(GRID%MPLWV)*WDES%NCDIJ,int(c_sizeof(SV(1,1)),c_int),c_loc(SV),1,GPU(1)%SV,1)   
nv_profile_stop(NVP_EDDAV_INIT)

!***********************************************************************
! loops over spins and k-points
!***********************************************************************
    spin:    DO ISP=1,WDES%ISPIN
    kpoints: DO NK=MY_NKSTART,WDES%NKPTS

nv_profile_start(NVP_EDDAV_SKINIT)
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
       CALL SETWDES(WDES,WDES1,NK)
       IF (LscaAWARE) CALL INIT_scala(WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))

       WA=ELEMENTS(W, WDES1, ISP)

       DE_ATT=ABS(W%CELTOT(WDES%NB_TOTK(NK,ISP),NK,ISP)-W%CELTOT(1,NK,ISP))/4

       IF (INFO%LREAL) THEN
          CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
          call INIT_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
       ELSE
          CALL PHASE(WDES,NONL_S,NK)
       ENDIF

       ! redistribute over plane wave coefficients
       IF (WDES%DO_REDIS) THEN
nv_profile_start(NVP_REDIS_BANDS)
          CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
          CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
nv_profile_stop(NVP_REDIS_BANDS)
       ENDIF

       !IF (LSUBROT .AND. .NOT. LscaAWARE) CHAM_ALL=0
       IF (LSUBROT .AND. .NOT. LscaAWARE) THEN
           call cuda_memset(GPU_CHAM_ALL,0,size_of_cham_all,int(c_sizeof(fakec),c_size_t))
           !CALL gpu_initz(-1,size_of_cham_all,size_of_cham_all,GPU_CHAM_ALL,0,zero)
       ENDIF

       W1%NB=0       ! empty the list of bands, which are optimized currently
       NB_DONE=0     ! index of bands already optimised
       IF (LEMPTY) THEN
          IF (WDES%WTKPT(NK)/=0) THEN
             DO NB_DONE=NBANDS,1,-1
                IF (ABS(W%FERWE(NB_DONE,NK,ISP)) > INFO%WEIMIN) EXIT
             ENDDO
             WRITE(*,*) 'perform empty band optimisation starting at',NB_DONE
          ENDIF
       ENDIF

!***********************************************************************
! allocate device memory and set device pointers
!***********************************************************************
    ! allocate WHAM, WOPT, W1, CWORK, and PRECON on device
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM_LOCAL*NITER,int(c_sizeof(fakec),c_size_t),GPU_WHAM_CW_ALL)
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM_LOCAL*NITER,int(c_sizeof(fakec),c_size_t),GPU_WOPT_CW_ALL)
    call cublas_Alloc_safety(WDES%NPROD_RED*NSUBD,int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_size_t),GPU_WOPT_CPROJ_ALL)
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM_LOCAL,int(c_sizeof(fakec),c_size_t),GPU_W1_CW_ALL)
    call cublas_Alloc_safety(WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM_LOCAL,int(c_sizeof(fakec),c_size_t),GPU_W1_CR_ALL)
    call cublas_Alloc_safety(WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM_LOCAL,int(c_sizeof(fakec),c_size_t),GPU_CWORK_ALL)
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM,int(c_sizeof(faker),c_size_t),GPU_PRECON)

    ! for debugging the arrays only, remove later!
    !call cuda_memset(GPU_WHAM_CW_ALL,0,WDES1%NRPLWV*NSIM_LOCAL*NITER,int(c_sizeof(fakec),c_size_t))
    !call cuda_memset(GPU_WOPT_CW_ALL,0,WDES1%NRPLWV*NSIM_LOCAL*NITER,int(c_sizeof(fakec),c_size_t))
    WOPT%CW = 0
    WHAM%CW = 0

    ! device pointers
    DO NP=1,NSIM_LOCAL
        GPU_W1_CW(NP) = GPU_W1_CW_ALL + (NP-1)*WDES1%NRPLWV*int(c_sizeof(fakec),c_size_t)
        GPU_W1_CR(NP) = GPU_W1_CR_ALL + (NP-1)*WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t)
        GPU_CWORK(NP) = GPU_CWORK_ALL + (NP-1)*WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t)
    ENDDO
    ! device pointers
    DO NP=1,NSIM_LOCAL*NITER
        GPU_WHAM_CW(NP) = GPU_WHAM_CW_ALL + (NP-1)*WDES1%NRPLWV*int(c_sizeof(fakec),c_size_t)
        GPU_WOPT_CW(NP) = GPU_WOPT_CW_ALL + (NP-1)*WDES1%NRPLWV*int(c_sizeof(fakec),c_size_t)
        GPU_WOPT_CPROJ(NP) = GPU_WOPT_CPROJ_ALL + (NP-1)*WDES1%NPROD*int(c_sizeof(fakec),c_size_t)
        !IF(NODE_ME==1) write(*,*) NP,(NP-1)*WDES1%NRPLWV,GPU_WHAM_CW(NP)
    ENDDO
    ! device pointers redistributed arrays
    DO NP=1,NSIM_LOCAL*NITER*WDES1%NB_PAR
        GPU_WHAM_CW_RED(NP) = GPU_WHAM_CW_ALL + (NP-1)*WDES1%NRPLWV_RED*int(c_sizeof(fakec),c_size_t)
        GPU_WOPT_CW_RED(NP) = GPU_WOPT_CW_ALL + (NP-1)*WDES1%NRPLWV_RED*int(c_sizeof(fakec),c_size_t)
    ENDDO

    ! allocate DATAKE, NINDPW, WS, and WA on device
    call cublas_Alloc_safety(WDES1%NGDIM*WDES1%NRSPINORS,int(c_sizeof(WDES1%DATAKE(1,1)),c_size_t),GPU(1)%DATAKE)
    call cublas_Alloc_safety(WDES1%NGVECTOR,int(c_sizeof(GPU(1)%NINDPW),c_size_t),GPU(1)%NINDPW)  
    call cublas_Alloc_safety(WDES%NPROD_RED*NSUBD,int(c_sizeof(WS%CPROJ_RED(1,1)),c_size_t),GPU_WS%CPROW)
    call cublas_Alloc_safety(WDES%NRPLWV_RED*WDES%NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_size_t),GPU_WA%CPTWFP) 
    call cublas_Alloc_safety(WDES%NPROD_RED*WDES%NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_size_t),GPU_WA%CPROJ)
    call cublas_Set_Vector(WDES%NRPLWV_RED*WDES%NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),c_loc(WA%CW_RED),1,GPU_WA%CPTWFP,1)  
    call cublas_Set_Vector(WDES%NPROD_RED*WDES%NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_int),c_loc(WA%CPROJ_RED),1,GPU_WA%CPROJ,1)
    
    ! copy NINDPW and DATAKE, from host to device
    call cublas_Set_Vector(WDES1%NGVECTOR,int(c_sizeof(WDES1%NINDPW(1)),c_int),c_loc(WDES1%NINDPW),1,GPU(1)%NINDPW,1)
    call cublas_Set_Vector(WDES1%NGDIM*WDES1%NRSPINORS,int(c_sizeof(WDES1%DATAKE(1,1)),c_int),c_loc(WDES1%DATAKE),1,GPU(1)%DATAKE,1)
nv_profile_stop(NVP_EDDAV_SKINIT)

!***********************************************************************
! loop over bands
!***********************************************************************
    max_bands = iter_bands
    iter_bands = 0
    pos_in_gpu_cfw = 0
       bands: DO
       iter_bands = iter_bands + 1
!write(*,*) "***********************************************"
!write(*,*) "bands...",iter_bands
!***********************************************************************
! NEWBAND section
! ---------------
! check the NB list, whether there is any empty slot
! fill in a not yet optimized wavefunction into the slot
! please mind, that optimization is done in chunks of NCPU bands
! presently, we have more functionality in the scheduler  then actually used
!***********************************************************************
nv_profile_start(NVP_EDDAV_NEWBAND)
          GPU_PTRS=0
!***********************************************************************
! loop over batches of NV_NUM_STREAMS, issue memcpys to/from device
! first to hide behind mpi communication and FFTs that follow
!***********************************************************************
          DO NB=1,NSIM_LOCAL,NV_NUM_STREAMS
!***********************************************************************
! batched memcpys to hide behind mpi and fft that follow
!***********************************************************************
nv_profile_start(NVP_NEWBAND_CPU)
          SID=0
          newband: DO NP=NB,MIN(NSIM_LOCAL,NB+NV_NUM_STREAMS-1)
              IF (W1(NP)%NB==0 .AND.  NB_DONE < NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  W1(NP)%NB=NB_DONE
                  GPU_PTRS(NP) = GPU_W1_CW(NP)  ! set array of device pointers
                  ITER=1; NPP=(ITER-1)*NSIM_LOCAL+NP

                  ! copy WA to W1, from host to host
                  CALL W1_COPY( ELEMENT( WA, N), W1(NP))

                  ! replaced with lines below
                  !CALL W1_COPY( W1(NP), ELEMENT( WOPT, NPP) )

                  ! copy W1%CW to WOPT%VW, from host to device
                  call cuda_memcpyhtod(SID,GPU_WOPT_CW(NPP),c_loc(W1(NP)%CW),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
                  ! TODO: port W1%CPROJ to GPU
                  ! copy W1%CPROJ to WOPT%CPROJ, from host to host
                  IF (WDES1%LGAMMA) THEN
                      CALL DCOPY( WDES1%NPROD, W1(NP)%CPROJ(1), 1, WOPT%CPROJ(1,NPP), 1)
                  ELSE
                      CALL ZCOPY( WDES1%NPROD, W1(NP)%CPROJ(1), 1, WOPT%CPROJ(1,NPP), 1)
                  ENDIF

                  IDUMP=0
                  IT(NP)=0
                  SID=SID+1  ! stream id
                  IF(SID>=NV_NUM_STREAMS) SID=0
              ENDIF
          ENDDO newband
nv_profile_stop(NVP_NEWBAND_CPU)
!***********************************************************************
! redistribute W1 over bands or plane wave coefficients
! perform fft on W1
!***********************************************************************
nv_profile_start(NVP_NEWBAND_FFT)
nv_profile_start(NVP_FFT)
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL,NB+NV_NUM_STREAMS-1)
              IF (GPU_PTRS(NP) == 0) CYCLE
              ITER=1; NPP=(ITER-1)*NSIM_LOCAL+NP

              IF (.NOT. WDES%DO_REDIS) THEN
              ! copy WOPT%CW to W1%CW, from device to device
              call cuda_memcpydtod(SID,GPU_W1_CW(NP),GPU_WOPT_CW(NPP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              ELSE
              ! redisitribute W1%CPROJ over bands
              CALL REDIS_PROJ(WDES1, 1, W1(NP)%CPROJ(1))
nv_profile_start(NVP_REDIS_NEWBAND)
              ! synchronize stream with bacthed memcpys above
              call cuda_streamsynchronize(SID)
              ! redistribute W1%CW over bands
              CALL REDIS_PW(WDES1, 1, W1(NP)%CW(1))
              ! copy W1%CW, from host to device
              call cuda_memcpyhtod(SID,GPU_W1_CW(NP),c_loc(W1(NP)%CW),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
nv_profile_stop(NVP_REDIS_NEWBAND)
              ENDIF

              ! fft to real space
              CALL FFTWAV_W1_GPU_STREAM(SID,W1(NP),GPU_W1_CW(NP),GPU_W1_CR(NP),GPU(1)%NINDPW)
              SID=SID+1
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
nv_profile_stop(NVP_FFT)
nv_profile_stop(NVP_NEWBAND_FFT)
          call threadsynchronize()
          ENDDO
!***********************************************************************
! calculate initial eigenvalues and setup preconditioner
!***********************************************************************
nv_profile_start(NVP_NEWBAND_ECCP)
          newband_eccp: DO NP=1,NSIM_LOCAL
              IF (GPU_PTRS(NP) == 0) CYCLE
              N=W1(NP)%NB
              IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
                  print *,"ECCP_VEC not yet on GPU"
                  call cuda_device_reset()
                  stop
              ELSE IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                  print *,"ECCP_TAU not yet on GPU"
                  call cuda_device_reset()
                  stop
              ELSE
                  CALL ECCP_GPU(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),W%CELEN(N,NK,ISP),GPU_W1_CR(NP),GPU_W1_CR(NP),GPU(1)%SV,IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU_W1_CW(NP),GPU_W1_CW(NP),GPU(1)%DATAKE)
              ENDIF
              ! propagate calculated eigenvalues to all nodes
              EVALUE_INI(NP)=W%CELEN(N,NK,ISP)
              call threadsynchronize()

              call cuda_truncatehighfrequency( &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                merge(1,0,LDELAY.OR.WDES1%LSPIRAL), &
                GPU_W1_CW(NP), &
                GPU(1)%DATAKE, &
                INFO%ENINI)
          ENDDO newband_eccp
nv_profile_stop(NVP_NEWBAND_ECCP)

          IPREC=INFO%IALGO
          IF (LDELAY) IPREC=8

nv_profile_start(NVP_NEWBAND_PRECOND)
          ! setup the preconditioner
          CALL SETUP_PRECOND_GPU(WDES1,NSIM_LOCAL,IPREC,EVALUE_INI,SLOCAL,DE_ATT,GPU_PTRS,GPU(1)%DATAKE,GPU_PRECON)
nv_profile_stop(NVP_NEWBAND_PRECOND)
nv_profile_stop(NVP_EDDAV_NEWBAND)
!***********************************************************************
! increase iteration counter and check whether list is empty
!***********************************************************************
          LSTOP=.TRUE.
          W1%LDO  =.FALSE.
          NSIM_LOCAL_=0
          DO NP=1,NSIM_LOCAL
             N=W1(NP)%NB
             IF ( N /= 0 ) THEN
                LSTOP  =.FALSE.
                W1(NP)%LDO=.TRUE.     ! band not finished yet
                IT(NP) =IT(NP)+1      ! increase iteration count
                NSIM_LOCAL_=NSIM_LOCAL_+1
             ENDIF
          ENDDO
          IF (LSTOP) THEN
            EXIT bands
          ENDIF
          ! right now all bands are consecutively ordered in W1 or WOPT%CW
          ! but it can happen that we treat less band in the last round
          NSIM_=NSIM_LOCAL_ * NCPU

!***********************************************************************
! HF section
! ----------
! calculate the HF part and update CELEN accordingly
!***********************************************************************
nv_profile_start(NVP_EDDAV_HF)
          IF (LHFCALC_DAV) THEN
             IF (IT(1)==1) THEN
             ! first iteration
             ! update eigenvalues and d.c.
                CDCHF=0
                IF (PRESENT(EXHF_ACFDT)) THEN
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                        NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                        WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1, EXHF_ACFDT=EXHF_ACFDT)
                ELSE
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                        NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                        WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1)
                ENDIF
                ! add Fock contribution to eigenvalues
                DO NP=1,NSIM_LOCAL
                   N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                   W%CELEN(N,NK,ISP)=W%CELEN(N,NK,ISP)+ &
                        W1_DOT( ELEMENT( W_ORIG, WDES1, N, ISP), ELEMENT (WHF, NP))
                ENDDO
                ! d.c. corrections
                EXHF=EXHF+CDCHF
             ELSE
                CDCHF=0
                CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                     NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                     WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1)
             ENDIF
          ENDIF
!***********************************************************************
! distribute the eigenvalues onto all nodes
!***********************************************************************
          IF (IT(1)==1) THEN
             EVALUE_GLBL=0
             DO NP=1,NSIM_LOCAL
                N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                EVALUE_INI(NP)=W%CELEN(N,NK,ISP)
                EVALUE_GLBL((NP-1)*NCPU+NODE_MEI)=W%CELEN(N,NK,ISP)
             ENDDO
             CALLMPI( M_sum_d(WDES%COMM_INTER,EVALUE_GLBL,NSIM_))
             EVALUE_INI_GLBL= EVALUE_GLBL
          ENDIF
nv_profile_stop(NVP_EDDAV_HF)
!***********************************************************************
! BANDOPT section
! ---------------
! intra-band optimisation
!***********************************************************************
nv_profile_start(NVP_EDDAV_BANDOPT)
          !gK here I can shift the eigenvalues to zero
          !this must not change the Hamilton matrix 
          !EVALUE_INI=0
          !EVALUE_INI_GLBL=0

          ITER=IT(1)
          NPP=(ITER-1)*NSIM_LOCAL_+1   ! storage position in WOPT%CW, WHAM%CW
!***********************************************************************
! calculate residual vectors (H - epsilon) psi
! consists of FFTs and small GEMMs
!***********************************************************************
nv_profile_start(NVP_BANDOPT_HAMILTMU)
          !  store H | psi > in WHAM%CW
          IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
             print *, "HAMILTMU_VEC_GPU does not exist yet !"
             call cuda_device_reset()
             stop
          ELSEIF (ASSOCIATED(HAMILTONIAN%MU)) THEN
             print *, "HAMILTMU_TAU_GPU does not exist yet !"
             call cuda_device_reset()
             stop
          ELSE
             ! store result in GPU_WHAM_CW
             CALL HAMILTMU_GPU(WDES1,W1,NONLR_S,NONL_S,EVALUE_INI,CDIJ,CQIJ,SV,ISP,ELEMENTS(WHAM,NPP,NPP+NSIM_LOCAL_-1),faketime,GPU,GPU_W1_CW,GPU_W1_CR,GPU_CWORK,GPU_WHAM_CW(NPP:))
             call threadsynchronize()
nv_profile_stop(NVP_BANDOPT_HAMILTMU)
          ENDIF
!***********************************************************************
! add Fock contribution to (H - epsilon) psi
!***********************************************************************
nv_profile_start(NVP_BANDOPT_ADDFOCK)
          IF (LHFCALC_DAV) THEN
             ! add Fock contribution to (H - epsilon ) psi
             DO NP=1,NSIM_LOCAL_
                IF (.NOT. W1(NP)%LDO) CYCLE
                NPP=(ITER-1)*NSIM_LOCAL_+NP

                ! replaced with lines below
                !CALL W1_DAXPY( ELEMENT(WHF,NP), 1.0_q, ELEMENT(WHAM, NPP))

                ! update WHAM%CPROJ on host
                IF (WDES1%LGAMMA) THEN
                    CALL DAXPY( WDES1%NPRO, 1.0_q, WHF%CPROJ(1,NP), 1, WHAM%CPROJ(1,NPP), 1 )
                ELSE
                    CALL DAXPY( WDES1%NPRO*2, 1.0_q, WHF%CPROJ(1,NP), 1, WHAM%CPROJ(1,NPP), 1 )
                ENDIF
                ! copy WHF%CW, from host to device
                call cublas_Set_Vector(WDES1%NRPLWV,int(c_sizeof(fakec),c_int),c_loc(WHF%CW(1,NP)),1,GPU_W1_CW(NP),1)
                ! update WHAM%CW on device
                call cublas_daxpy(WDES1%NPL*2, 1.0_q,GPU_W1_CW(NP),1,GPU_WHAM_CW(NPP),1)
             ENDDO
             call threadsynchronize()
          ENDIF
nv_profile_stop(NVP_BANDOPT_ADDFOCK)

nv_profile_start(NVP_BANDOPT_OVERL)
          GPU_PTRS=0
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP
              GPU_PTRS(NP) = GPU_WHAM_CW(NPP)  ! set array of device pointers

              IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                  WDES1%NBANDS=1    ! WDES1%NBANDS is used only here to fake OVERL
                  CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
              ENDIF

              ! truncate high frequency values in WHAM%CW
              call cuda_truncatehighfrequency( &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                merge(1,0,LDELAY.OR.WDES1%LSPIRAL), &
                GPU_WHAM_CW(NPP), &
                GPU(1)%DATAKE, &
                INFO%ENINI)
          ENDDO
          call threadsynchronize()
nv_profile_stop(NVP_BANDOPT_OVERL)
!***********************************************************************
! calculate norm and metric
!***********************************************************************
nv_profile_start(NVP_BANDOPT_NORM)
          ! calculate norm and metric for NSIM bands (should calculate this only for ITER=1?)
          call cuda_normwithmetricblock( &
                NSIM_LOCAL_, &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                GPU_PTRS, &
                GPU_PRECON, &
                FNORM, &
                FPRE_, &
                1)
          DO NP=1,NSIM_LOCAL_
              CALLMPI(M_sum_s(WDES1%COMM_INB,2,FNORM(NP),FPRE_(NP),0._q, 0._q))
          ENDDO
          call threadsynchronize()
nv_profile_stop(NVP_BANDOPT_NORM)
!***********************************************************************
! copy WHAM to W1, host to host and device to device
!***********************************************************************
nv_profile_start(NVP_BANDOPT_COPY)
          SID=0
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP

              ! replaced with lines below
              !CALL W1_COPY( ELEMENT( WHAM, NPP), W1(NP))

              ! copy WHAM%CW to W1%CW, from device to device
              call cuda_memcpydtod(SID,GPU_W1_CW(NP),GPU_WHAM_CW(NPP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              ! copy WHAM%CPROJ to W1%CPROJ, from host to host
              IF (WDES1%LGAMMA) THEN
                  CALL DCOPY( WDES1%NPROD, WHAM%CPROJ(1,NPP), 1,  W1(NP)%CPROJ(1), 1)
              ELSE
                  CALL ZCOPY( WDES1%NPROD, WHAM%CPROJ(1,NPP), 1, W1(NP)%CPROJ(1), 1)
              ENDIF

              IF (ITER==1) THEN
                  RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*SQRT(ABS(FNORM(NP)))/NB_TOT*WDES%NRSPINORS
              ENDIF
              SID=SID+1  ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
          call threadsynchronize()
nv_profile_stop(NVP_BANDOPT_COPY)
!***********************************************************************
! mpi redistribution
!***********************************************************************
nv_profile_start(NVP_BANDOPT_END)
          IF (WDES%DO_REDIS) THEN
!***********************************************************************
! redistribute WHAM and WS over bands or plane wave coefficients
!***********************************************************************
#ifdef GPUDIRECT
          IF (INFO%LOVERL .AND. WDES%NPROD>0) THEN
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP
              ! redistribute WS%CPROJ over ?
              CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
          ENDDO
          ENDIF
          NPP=(ITER-1)*NSIM_LOCAL_+1   ! storage position in WOPT%CW, WHAM%CW
          ! redistribute WHAM%CW over ?
          CALL GPU_REDIS(WDES1,NSIM_LOCAL_,WDES1%NRPLWV,GPU_W1_CW(1),GPU_WHAM_CW(NPP))
#else
!***********************************************************************
! loop over batches of NV_NUM_STREAMS, as explained earlier
!***********************************************************************
          DO NB=1,NSIM_LOCAL_,NV_NUM_STREAMS
!***********************************************************************
! batched memcpys to hide behind mpi
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP
              ! copy WHAM%CW, from device to host
              call cuda_memcpydtoh(SID,c_loc(WHAM%CW(1,NPP)),GPU_WHAM_CW(NPP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              SID=SID+1  ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
!***********************************************************************
! redistribute WS over bands or plane wave coefficients
!***********************************************************************
          IF (INFO%LOVERL .AND. WDES%NPROD>0) THEN
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP
              ! redistribute WS%CPROJ over ?
              CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
          ENDDO
          ENDIF
!***********************************************************************
! redistribute WHAM over bands or plane wave coefficients
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=(ITER-1)*NSIM_LOCAL_+NP

              ! synchronize dtoh copy above
              call cuda_streamsynchronize(SID)
              ! redistribute WHAM%CW over ?
              CALL REDIS_PW(WDES1, 1, WHAM%CW(1,NPP))
              ! copy WHAM%CW, from host to device
              call cuda_memcpyhtod(SID,GPU_WHAM_CW(NPP),c_loc(WHAM%CW(1,NPP)),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              SID=SID+1  ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
          ENDDO
#endif
          call threadsynchronize()
          ENDIF
nv_profile_stop(NVP_BANDOPT_END)
nv_profile_stop(NVP_EDDAV_BANDOPT)

!DDEBUG
!IF(iter_bands==1 .AND. NNELM==5 .AND. NODE_ME==1) THEN
 !call cuda_print("wham_bopt"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,GPU_WHAM_CW(1))
 !call cuda_print("wopt_bopt"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,GPU_WOPT_CW(1))
 !call fortran_print("whamc_bopt"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,WHAM%CW(1,1))
 !call fortran_print("woptc_bopt"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,WOPT%CW(1,1))
 !call cuda_device_reset()
 !stop
!ENDIF

!***********************************************************************
! UPDATEHO section
! ----------------
! update the elements of the Hamilton matrix and of the overlap matrix
! in the space spanned by the present wave functions
!***********************************************************************
nv_profile_start(NVP_EDDAV_UPDATEHO)
          LSTOP=.FALSE.

          ! calulcate CQIJ * W1%CPROJ (required for overlap)
          ! get the index into the redistributed array
          ITER=IT(1)                  ! iter is right now the same for each band

          NPOS_RED=(ITER-1)*NSIM_+1   ! storage position in redistributed WOPT%CW, WHAM%CW

          CHAM(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
          COVL(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
!***********************************************************************
! copy WS and WOPT, from host to device
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_MEMCPY)
          call cublas_Set_Matrix(WDES%NPROD_RED,NSIM_,int(c_sizeof(WS%CPROJ_RED(1,1)),c_int),c_loc(WS%CPROJ_RED(1,1)),WDES%NPROD_RED,GPU_WS%CPROW,WDES%NPROD_RED)
          call cublas_Set_Matrix(WDES%NPROD_RED,NSIM_,int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_int),c_loc(WOPT%CPROJ_RED),WDES%NPROD_RED,GPU_WOPT_CPROJ(1),WDES%NPROD_RED)
!***********************************************************************
! copy COVL and CHAM, from host to device
!***********************************************************************
          call cublas_Set_Matrix(NSUBD,NSUBD,int(c_sizeof(COVL(1,1)),c_int),c_loc(COVL),NSUBD,GPU(1)%COVL,NSUBD)
          !TODO replace with memset 0 for first band iteration
          call cublas_Set_Matrix(NSUBD,NSUBD,int(c_sizeof(CHAM(1,1)),c_int),c_loc(CHAM),NSUBD,GPU(1)%CHAM,NSUBD)
nv_profile_stop(NVP_UPDATEHO_MEMCPY)

          if( ITER == 1) pos_in_gpu_cfw = 0
          shift_cfw = (WDES%NRPLWV_RED*pos_in_gpu_cfw)
          pos_in_gpu_cfw = pos_in_gpu_cfw + NSIM_
!***********************************************************************
! calculate GEMM, CHAM = WOPT * WHAM
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_CHAM)
nv_profile_start(NVP_GEMM)
          CALL ORTH1_GPU_shift('U',GPU_WOPT_CW_RED(1),0,GPU_WHAM_CW_RED(NPOS_RED),0,GPU_WOPT_CPROJ(1),0,GPU_WS%CPROW,0,NSUBD,NPOS_RED,NSIM_,WDES1%NPL_RED,0,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU(1)%CHAM,0)
          call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_UPDATEHO_CHAM)
!***********************************************************************
! copy CHAM, from device to host
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_MEMCPY)
          call cublas_Get_Matrix(NSUBD,NSUBD,int(c_sizeof(CHAM(1,1)),c_int),GPU(1)%CHAM,NSUBD,c_loc(CHAM),NSUBD)
nv_profile_stop(NVP_UPDATEHO_MEMCPY)
!***********************************************************************
! calculate GEMM: COVL = WOPT * WOPT
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_COVL)
nv_profile_start(NVP_GEMM)
          CALL ORTH1_GPU_SHIFT('U',GPU_WOPT_CW_RED(1),0,GPU_WOPT_CW_RED(NPOS_RED),0,GPU_WOPT_CPROJ(1),0,GPU_WS%CPROW,0,NSUBD,NPOS_RED,NSIM_,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU(1)%COVL,0)
          call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_UPDATEHO_COVL)
!DDBEBUG
!call cuda_print("cham_ho"//NODE_MEC//".dat",2,NSUBD*NSUBD,GPU(1)%CHAM)
!call cuda_print("covl_ho"//NODE_MEC//".dat",2,NSUBD*NSUBD,GPU(1)%COVL)
!***********************************************************************
! copy COVL, from device to host
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_MEMCPY)
          call cublas_Get_Matrix(NSUBD,NSUBD,int(c_sizeof(COVL(1,1)),c_int),GPU(1)%COVL,NSUBD,c_loc(COVL),NSUBD)
nv_profile_stop(NVP_UPDATEHO_MEMCPY)
!***********************************************************************
! start block of CPU work
! sum COVL and CHAM between mpi ranks
!***********************************************************************
nv_profile_start(NVP_UPDATEHO_ADD)
          CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,NPOS_RED),NSUBD*NSIM_))
          CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM(1,NPOS_RED),NSUBD*NSIM_))

          ! add remaining elements to COVL
          DO M=1,NSIM_*(ITER-1)
             DO I=1,NSIM_
                COVL(NPOS_RED-1+I,M)=GCONJG(COVL(M,NPOS_RED-1+I))
             ENDDO
          ENDDO
          ! correct CHAM by subtraction of epsilon COVL
          DO M=1,NSIM_*ITER
             DO I=1,NSIM_
                CHAM(M,NPOS_RED-1+I)=CHAM(M,NPOS_RED-1+I)+COVL(M,NPOS_RED-1+I)*EVALUE_INI_GLBL(I)
             ENDDO
          ENDDO

#ifndef gammareal
          DO N1=1,NSIM_*ITER
             IF (ABS(AIMAG(CHAM(N1,N1)))/MAX(1._q,ABS(REAL(CHAM(N1,N1), q)))>1E-5_q) THEN
                WRITE(*,*)'WARNING: Sub-Space-Matrix is not hermitian in DAV ',N1, &
                     AIMAG(CHAM(N1,N1))
             ENDIF
             CHAM(N1,N1)= REAL( CHAM(N1,N1) ,KIND=q)
          ENDDO
#endif
nv_profile_stop(NVP_UPDATEHO_ADD)
!***********************************************************************
! solve eigenvalue problem and calcualte lowest eigenvector, this
! eigenvector corresponds to a minimal residuum:
! CHAM(n1,n2) U(n2,1) = E(1) S(n1,n2)  U(n2,1)
!***********************************************************************
          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "Hamiltonian", WDES, CHAM, SIZE(CHAM,1), ITER*NSIM_)
             CALL DUMP_HAM_SELECTED( "overlap", WDES, COVL, SIZE(COVL,1), ITER*NSIM_)
             io_end
          ENDIF

          IF (NODE_ME==IONODE) THEN
             ! estimate L1 norm of the inverse of the overlap matrix
             ! which is essentially the condition number (since L1 norm of COVL is 1)
             COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
nv_profile_start(NVP_UPDATEHO_POTRF)
#ifdef gammareal
             CALL DPOTRF('U',ITER*NSIM_,COVL_,NSUBD,IFAIL)
#else
             CALL ZPOTRF('U',ITER*NSIM_,COVL_,NSUBD,IFAIL)
#endif
nv_profile_stop(NVP_UPDATEHO_POTRF)

             IF (IFAIL==0) THEN
nv_profile_start(NVP_UPDATEHO_POCON)
#ifdef gammareal
                CALL DPOCON('U',ITER*NSIM_,COVL_,NSUBD,1.0_q,RCOND,CWRK,RWORK,IFAIL)
#else 
                CALL ZPOCON('U',ITER*NSIM_,COVL_,NSUBD,1.0_q,RCOND,CWRK,RWORK,IFAIL)
#endif
nv_profile_stop(NVP_UPDATEHO_POCON)
             ENDIF

!***********************************************************************
! use MAGMA for diagon. of ITER*(NSUBD x NSUBD) subspace matrix COVL
!***********************************************************************
             ! ok if that is less than 1E-13 to 1E-14 stop
             ! to have some headroom I set it to 1E-13 (1E-14 worked in my tests however)
             IF (ABS(RCOND)<1E-13 .AND. IFAIL==0 ) IFAIL=1
             IF (IFAIL==0) THEN
                CEIG (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
                COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
nv_profile_start(NVP_UPDATEHO_ESOLVE)
#ifdef gammareal
#ifdef USE_MAGMA
                write(*,*) "WARNING: have not tested magma_dsygvd yet!"
                call magma_dsygvd(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,IFAIL)
#else
                CALL DSYGV(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,CWRK(1),LWORK*NB_TOT,IFAIL)
#endif
#else
#ifdef USE_MAGMA
    ! USE_ZHEEVX only with MAGMA because CPU code has not implemented this!
#ifndef USE_ZHEEVX
                call magma_zhegvd(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,IFAIL)
#else
                VL=0; VU=0; IL=0; IU=0
                call magma_zhegvdx(1,'V','A','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,VL,VU,IL,IU,NB_CALC,R,IFAIL)
#endif
#else
                CALL ZHEGV(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
#endif
nv_profile_stop(NVP_UPDATEHO_ESOLVE)
             ENDIF
          ENDIF

          ! communicate IFAIL to all nodes
          CALLMPI( M_bcast_i(WDES%COMM_KIN, IFAIL, 1))

          IF (IFAIL/=0) THEN
             IF (ITER>=3) THEN
             ITER=ITER-1
             CEIG (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
             COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
nv_profile_start(NVP_UPDATEHO_ESOLVE)
#ifdef gammareal
#ifdef USE_MAGMA
             write(*,*) "WARNING: have not tested magma_dsygvd yet!"
             call magma_dsygvd(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,IFAIL)
#else
             CALL DSYGV(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,CWRK(1),LWORK*NB_TOT,IFAIL)
#endif
#else
#ifdef USE_MAGMA
    ! USE_ZHEEVX only with MAGMA because CPU code has not implemented this!
#ifndef USE_ZHEEVX
             call magma_zhegvd(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,IFAIL)
#else
             VL=0; VU=0; IL=0; IU=0
             call magma_zhegvdx(1,'V','A','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,VL,VU,IL,IU,NB_CALC,R,IFAIL)
#endif
#else
             CALL ZHEGV(1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R,CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
#endif
nv_profile_stop(NVP_UPDATEHO_ESOLVE)
             ENDIF
             IF (IFAIL/=0) THEN
                IF (IU6>=0) &
                WRITE(IU6,219) IFAIL,ITER,ITER*NSIM_
                IF (IU0>=0) &
                WRITE(IU0,219) IFAIL,ITER,ITER*NSIM_
                STOP
             ENDIF
             ! force stop now (can get only worse)
             LSTOP=.TRUE.
             IT(:)=ITER
          ENDIF
219       FORMAT('Error EDDDAV: Call to ZHEGV failed. Returncode =',I4,I2,I4)

          ! now broadcase to make sure all nodes use identical CEIG
          CALLMPI( M_bcast_g(WDES%COMM_KIN, CEIG, NSUBD*ITER*NSIM_))
          CALLMPI( M_bcast_d(WDES%COMM_KIN, R, ITER*NSIM_))

          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "reconstructed Hamiltonian", WDES, CEIG, SIZE(CEIG,1), ITER*NSIM_)
             io_end
          ENDIF
!***********************************************************************
! end block of CPU work
!***********************************************************************
nv_profile_stop(NVP_EDDAV_UPDATEHO)

!***********************************************************************
! TOTE section
! ------------
! update energies and calculate total energy change
!***********************************************************************
nv_profile_start(NVP_EDDAV_TOTE)
          II=0
          DEMAX=0
          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
             DO NPOS_RED=(N-1)*NCPU+1,N*NCPU
                II=II+1
                W%CELTOT(NPOS_RED,NK,ISP)=R(II) ! update CELTOT array
                DECEL =R(II)-EVALUE_GLBL(II)    ! change in eigenenergy

                ! if the change in the eigenenergy is very small 
                DEMAX=MAX(DEMAX, ABS(DECEL))

                IF (IDUMP==2)  WRITE(*,'(E10.2,2H |)',ADVANCE='NO') DECEL
                DESUM=DESUM+WDES%RSPIN*WDES%WTKPT(NK)*W%FERTOT(NPOS_RED,NK,ISP)*DECEL
                EVALUE_GLBL(II)=R(II)  ! update 

             ENDDO
          ENDDO
nv_profile_stop(NVP_EDDAV_TOTE)

!***********************************************************************
! BREAKOPT section
! ----------------
! possuble break the optimization and new eigenenergy
! if stopping is selected (i.e. LSTOP=.TRUE.), store the optimized
! wavefuction back
!***********************************************************************
          ITER=IT(1)
          ! break if absolute change in eigenenergy is small
          IF (ITER>1 .AND. DEMAX < INFO%EBREAK) LSTOP=.TRUE.
          ! relative break criterion
          IF (ITER==2) DEIT=DEMAX*INFO%DEPER
          IF (ITER>2 .AND. DEMAX < DEIT) LSTOP=.TRUE.

          NITER_NOW=NITER
          ! sufficient iterations done
          IF (ITER >= NITER_NOW) LSTOP=.TRUE.
          IF (ITER >= NITER)     LSTOP=.TRUE.  ! certainly stop if storage requires this
          IF (LSTOP) THEN
nv_profile_start(NVP_EDDAV_BREAKOPT)
             NPOS_RED=(W1(1)%NB-1)*NCPU
!***********************************************************************
! allocate COVL on device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MALLOC)
             call cublas_Alloc_safety (NSUBD*NSUBD,int(c_sizeof(CEIG(1,1)),c_size_t),GPU(2)%COVL)
nv_profile_stop(NVP_BREAKOPT_MALLOC)
!***********************************************************************
! copy COVL, from host to device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MEMCPY)
             call cublas_Set_Matrix(NSUBD,NSUBD,int(c_sizeof(COVL(1,1)),c_int),c_loc(CEIG),NSUBD,GPU(2)%COVL,NSUBD)
nv_profile_stop(NVP_BREAKOPT_MEMCPY)
!***********************************************************************
! calculate GEMM: WA = WOPT * COVL
!***********************************************************************           
nv_profile_start(NVP_BREAKOPT_WACW)
nv_profile_start(NVP_GEMM)
             IF (WDES1%NPL_RED/=0) &
                 CALL CUBLAS_ZGEMMSH('N','N',m_ WDES1%NPL_RED,NSIM_,NSIM_*ITER,one,GPU_WOPT_CW_RED(1),0,m_ WDES%NRPLWV_RED,GPU(2)%COVL,0,NSUBD,zero,GPU_WA%CPTWFP,IDX(1,NPOS_RED+1,WDES1%NRPLWV_RED),m_ WDES%NRPLWV_RED)
             call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_BREAKOPT_WACW)
!***********************************************************************
! copy WA, from device to host async
!***********************************************************************
             !Async version, partially overlaps with the ZGEMMSH below
             call cublas_get_matrix_async( &
                 1, WDES1%NRPLWV_RED, &
                 NSIM_, &
                 int(c_sizeof(WA%CW_RED(1,1)),c_int), &
                 GPU_WA%CPTWFP+IDX(1,NPOS_RED+1,WDES1%NRPLWV_RED)*int(c_sizeof(WA%CW_RED(1,1)),c_size_t), &
                 WDES1%NRPLWV_RED, &
                 c_loc(WA%CW_RED(1,NPOS_RED+1)), &
                 WDES1%NRPLWV_RED)
             !call cublas_Get_Matrix_shz_async(1,WDES1%NRPLWV_RED,NSIM_,int(c_sizeof(WA%CW_RED(1,1)),c_size_t),GPU_WA%CPTWFP,IDX(1,NPOS_RED+1,WDES1%NRPLWV_RED),WDES1%NRPLWV_RED,WA%CW_RED(1,NPOS_RED+1),WDES1%NRPLWV_RED)
!***********************************************************************
! calculate GEMM: WA = WOPT * COVL
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_WACPROJ)
nv_profile_start(NVP_GEMM)
             IF (WDES1%NPRO_RED/=0) &
                 CALL CUBLAS_ZGEMMSH('N','N',WDES1%NPRO_RED,NSIM_,NSIM_*ITER,one,GPU_WOPT_CPROJ(1),0,WDES%NPROD_RED,GPU(2)%COVL,0,NSUBD,zero,GPU_WA%CPROJ,IDX(1,NPOS_RED+1,WDES1%NPROD_RED),WDES%NPROD_RED)
             call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_BREAKOPT_WACPROJ)
!***********************************************************************
! copy WA, from device to host
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MEMCPY)
             call cuda_memcpydtohshift(NULL_STREAM,c_loc(WA%CPROJ_RED(1,NPOS_RED+1)),0,GPU_WA%CPROJ,IDX(1,NPOS_RED+1,WDES1%NPROD_RED),WDES1%NPROD_RED*NSIM_,int(c_sizeof(WA%CPROJ_RED(1,1)),c_size_t))
             !call cublas_Get_Matrix_shz(WDES1%NPROD_RED,NSIM_,int(c_sizeof(WA%CPROJ_RED(1,1)),c_size_t),GPU_WA%CPROJ,IDX(1,NPOS_RED+1,WDES1%NPROD_RED),WDES1%NPROD_RED,WA%CPROJ_RED(1,NPOS_RED+1),WDES1%NPROD_RED)   
nv_profile_stop(NVP_BREAKOPT_MEMCPY)
!***********************************************************************
! free COVL on device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MALLOC)
             call cublas_free(GPU(2)%COVL)
nv_profile_stop(NVP_BREAKOPT_MALLOC)

             IF (LSUBROT) THEN
!***********************************************************************
! allocate COVL on device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MALLOC)
          call cublas_Alloc_safety (NSUBD*NSUBD,int(c_sizeof(CEIG(1,1)),c_size_t),GPU(2)%COVL)
nv_profile_stop(NVP_BREAKOPT_MALLOC)
!***********************************************************************
! copy COVL, from host to device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MEMCPY)
          call cublas_Set_Matrix(NSUBD,NSUBD,int(c_sizeof(COVL(1,1)),c_int),c_loc(CEIG),NSUBD,GPU(2)%COVL,NSUBD)
nv_profile_stop(NVP_BREAKOPT_MEMCPY)
!***********************************************************************
! compute GEMM: WOPT = WHAM * COVL
!***********************************************************************
          ! store in the corresponding (H - epsilon S)  in WOPT%CW_RED
nv_profile_start(NVP_BREAKOPT_WOPTCW)
nv_profile_start(NVP_GEMM)
          CALL CUBLAS_ZGEMMSH('N','N',WDES1%NPL_RED,NSIM_,NSIM_*ITER,one,GPU_WHAM_CW(1),0,WDES%NRPLWV_RED,GPU(2)%COVL,0,NSUBD,zero,GPU_WOPT_CW(1),0,WDES%NRPLWV_RED)
          call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_BREAKOPT_WOPTCW)

          IF ( LscaAWARE ) THEN
nv_profile_start(NVP_BREAKOPT_MEMCPY)
             !psnvidia: this memcpy might not be required at all
             !call cublas_Get_Matrix(WDES%NRPLWV_RED,NSIM_,int(c_sizeof(fakec),c_size_t),GPU_WOPT_CW(1),WDES%NRPLWV_RED,WOPT%CW_RED,WDES%NRPLWV_RED)
nv_profile_stop(NVP_BREAKOPT_MEMCPY)
          endif
!***********************************************************************
! free COVL on device
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_MALLOC)
          call cublas_free(GPU(2)%COVL)
nv_profile_stop(NVP_BREAKOPT_MALLOC)

                ! calculate epsilon COVL
                NPOS_RED =(W1(1)%NB-1)*NCPU+1
                IF (.NOT. LscaAWARE) THEN
!***********************************************************************
! calculate GEMM: CHAM = WA * WOPT
!***********************************************************************
nv_profile_start(NVP_BREAKOPT_WOPTCPROJ)
nv_profile_start(NVP_GEMM)
#ifdef MIXED_PRECISION_GPU
                   CALL ORTH1_GPU_SHIFT_SP('U',GPU_WA%CPTWFP,0,GPU_WOPT_CW(1),0,GPU_WA%CPROJ,0,GPU_WA%CPROJ,0,NB_TOT,NPOS_RED,NSIM_,WDES1%NPL_RED,0,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU_CHAM_ALL,0)
#else
                   CALL ORTH1_GPU_SHIFT('U',GPU_WA%CPTWFP,0,GPU_WOPT_CW(1),0,GPU_WA%CPROJ,0,GPU_WA%CPROJ,0,NB_TOT,NPOS_RED,NSIM_,WDES1%NPL_RED,0,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU_CHAM_ALL,0)
#endif
                   call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_BREAKOPT_WOPTCPROJ)

                ! correct the small NSIM_ times NSIM_ block
                ! which is incorrect since we have calculate H - S epsilon psi
                ! and not H psi and since  our psi are not orthogonal to each other
                ! this block is however anyway diagonal with the elements R(I)
                   IF (.NOT. LscaAWARE) THEN
nv_profile_start(NVP_BREAKOPT_CHAM)
                     IF (NODE_ME==IONODE) THEN
                       call cuda_correctcham(GPU_CHAM_ALL, NSIM_, NPOS_RED-1, NB_TOT, 1, c_loc(R))
                     else
                       call cuda_correctcham(GPU_CHAM_ALL, NSIM_, NPOS_RED-1, NB_TOT, 0, c_loc(R))
                     ENDIF
nv_profile_stop(NVP_BREAKOPT_CHAM)
                   else
                           print *, "WARNING: not yet implemented (1241). Size of cham_all differs if NOT LscaAware." !TODO
                           stop
                   endif
                ELSE
                   print *, "ORTH1_DISTRI_DAVIDSON_GPU DOES NOT YET EXIST !!!"
                   stop
                ENDIF
             ENDIF
             W1%NB=0
nv_profile_stop(NVP_EDDAV_BREAKOPT)
             CYCLE bands
          ENDIF
          ICOUEV=ICOUEV+NSIM_

!***********************************************************************
! CALCWAVEC section
! -----------------
! preconditioning of calculated residual vectors W1
! perform FFT to real space
!***********************************************************************
nv_profile_start(NVP_EDDAV_CALCWAVEC)
          IF ( .NOT. INFO%LREAL ) THEN
              SID=0
              DO NP=1,NSIM_LOCAL_
                  N=W1(NP)%NB; ITER=IT(NP)+1; IF (.NOT. W1(NP)%LDO) CYCLE
                  ! apply preconditioner to W1%CW
                  call cuda_applyprecond(SID,WDES1%NPL,GPU_W1_CW(NP),GPU_W1_CW(NP),GPU_PRECON,1.0_q,(NP-1)*WDES1%NRPLWV)
                  SID=SID+1 ! stream id
                  IF(SID>=NV_NUM_STREAMS) SID=0
              ENDDO
          ELSE
nv_profile_start(NVP_CALCWAVEC_FFT)
nv_profile_start(NVP_FFT)
              SID=0
              DO NP=1,NSIM_LOCAL_
                  N=W1(NP)%NB; ITER=IT(NP)+1; IF (.NOT. W1(NP)%LDO) CYCLE
                  ! apply preconditioner to W1%CW
                  call cuda_applyprecond(SID,WDES1%NPL,GPU_W1_CW(NP),GPU_W1_CW(NP),GPU_PRECON,1.0_q,(NP-1)*WDES1%NRPLWV)
                  ! fft to real space
                  CALL FFTWAV_W1_GPU_STREAM(SID,W1(NP),GPU_W1_CW(NP),GPU_W1_CR(NP),GPU(1)%NINDPW)
                  SID=SID+1 ! stream id
                  IF(SID>=NV_NUM_STREAMS) SID=0
              ENDDO
              call threadsynchronize()
nv_profile_stop(NVP_FFT)
nv_profile_stop(NVP_CALCWAVEC_FFT)
          ENDIF

          !updates W1%CPROJ. GPU data of CPROJ is copied to cpu and outdated afterwards
          !JBNV, if NSIM_LOCAL_ == 1, then W1_PROJALL_GPU takes a CPU only code
          !path, therefore copy back the GPU results to the host 
          if(NSIM_LOCAL_ == 1) then
              ! copy W1%CR, from device to host
              call cublas_Get_Vector(WDES1%GRID%MPLWV*WDES1%NRSPINORS,int(c_sizeof(fakec),c_int),GPU_W1_CR(1),1,c_loc(W1(1)%CR),1)
          endif
!***********************************************************************
! calculate the wavefunction character of precon. residual vectors W1
!***********************************************************************
nv_profile_start(NVP_CALCWAVEC_PROJALL)
          CALL W1_PROJALL_GPU(WDES1,W1,NONLR_S,NONL_S,NSIM_LOCAL_,GPU_W1_CR,faketime)
          call threadsynchronize()
nv_profile_stop(NVP_CALCWAVEC_PROJALL)
!***********************************************************************
! normalize precon. residual vectors W1
!***********************************************************************
nv_profile_start(NVP_CALCWAVEC_NORM)
          GPU_PTRS=0
          DO NP=1,NSIM_LOCAL_
              IF (W1(NP)%LDO) GPU_PTRS(NP)=GPU_W1_CW(NP)
          ENDDO

          ! replaced with lines below
          !CALL CNORMN(W1(NP),CQIJ, ISP, WSCAL)

          WSCAL=0
          WSCALZ=0
          call cuda_zdotcblock( &
                NSIM_LOCAL_, &
                WDES1%NPL, &
                GPU_PTRS, &
                GPU_PTRS, &
                WSCALZ)
          call threadsynchronize()
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              CALL CNORMA_GPU_CPUONLY(W1(NP), CQIJ, ISP, WSCAL(NP), GPU_W1_CW(NP))
              WSCAL(NP) = WSCAL(NP) + REAL(WSCALZ(NP),KIND=q)
              CALL CNORMA_GPU_CPU_MPI(W1(NP), WSCAL(NP), WSCAL(NP))
              CALL GDSCAL(WDES1%NPRO, WSCAL(NP), W1(NP)%CPROJ(1), 1)
          ENDDO
          call cuda_zdscalblock( &
                NSIM_LOCAL_, &
                WDES1%NPL, &
                WSCAL, &
                GPU_PTRS)
          call threadsynchronize()
        
          ! replaced with lines below
          !CALL W1_COPY(W1(NP), ELEMENT(WOPT, NPP))

          ! copy W1%CW to WOPT%CW, from device to device
          call cuda_memcpydtod(-1,GPU_WOPT_CW(IT(1)*NSIM_LOCAL_+1),GPU_W1_CW(1),NSIM_LOCAL_*WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
nv_profile_stop(NVP_CALCWAVEC_NORM)
!***********************************************************************
! do some CPU work
!***********************************************************************
nv_profile_start(NVP_CALCWAVEC_CPU)
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW

              ! copy W1%CPROJ to WOPT%CPROJ, from host to host
              IF (WDES1%LGAMMA) THEN
                  CALL DCOPY(WDES1%NPROD, W1(NP)%CPROJ(1), 1, WOPT%CPROJ(1,NPP), 1)
              ELSE
                  CALL ZCOPY(WDES1%NPROD, W1(NP)%CPROJ(1), 1, WOPT%CPROJ(1,NPP), 1)
              ENDIF
            
              IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                  WDES1%NBANDS=1    ! is used this only here not quite clean
                  CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
              ENDIF
          ENDDO
          call threadsynchronize()
nv_profile_stop(NVP_CALCWAVEC_CPU)
!***********************************************************************
! redistribute WOPT over bands or plane wave coefficients
!***********************************************************************
          IF (WDES%DO_REDIS) THEN
nv_profile_start(NVP_CALCWAVEC_REDIS)
nv_profile_start(NVP_REDIS_CALCWAVEC)
!***********************************************************************
! redistribute WOPT and WS over bands or plane wave coefficients
!***********************************************************************
#ifdef GPUDIRECT
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP
              ! redistribute WOPT%CPROJ over ?
              CALL REDIS_PROJ(WDES1, 1, WOPT%CPROJ(1,NPP))
              ! redistribute WS%CPROJ over ?
              IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                  CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
              ENDIF
          ENDDO
          NPP=ITER*NSIM_LOCAL_+1   ! storage position in WOPT%CW, WHAM%CW
          ! redistribute WOPT%CW over ?
          CALL GPU_REDIS(WDES1,NSIM_LOCAL_,WDES1%NRPLWV,GPU_W1_CW(1),GPU_WOPT_CW(NPP))
#else
!***********************************************************************
! loop over batches of NV_NUM_STREAMS, as explained earlier
!***********************************************************************
          DO NB=1,NSIM_LOCAL_,NV_NUM_STREAMS
!***********************************************************************
! loop over batches of NV_NUM_STREAMS, as explained earlier
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP
              ! copy W1%CW, from device to host
              call cuda_memcpydtoh(SID,c_loc(WOPT%CW(1,NPP)),GPU_W1_CW(NP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              SID=SID+1 ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
!***********************************************************************
! redistribute WOPT and WS over bands or plane wave coefficients
!***********************************************************************
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP
              ! redistribute WOPT%CPROJ over ?
              CALL REDIS_PROJ(WDES1, 1, WOPT%CPROJ(1,NPP))
              ! redistribute WS%CPROJ over ?
              IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                  CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
              ENDIF
          ENDDO
!***********************************************************************
! redistribute WOPT over bands or plane wave coefficients
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP
              ! synchronize dtoh copy above
              call cuda_streamsynchronize(SID)
              ! redistribute WOPT%CW over ?
              CALL REDIS_PW(WDES1, 1, WOPT%CW(1,NPP))
              ! copy WOPT%CW, from host to device
              call cuda_memcpyhtod(SID,GPU_WOPT_CW(NPP),c_loc(WOPT%CW(1,NPP)),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              SID=SID+1 ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
          ENDDO
#endif
          call threadsynchronize()
nv_profile_stop(NVP_REDIS_CALCWAVEC)
nv_profile_stop(NVP_CALCWAVEC_REDIS)
          ENDIF
nv_profile_stop(NVP_EDDAV_CALCWAVEC)

!***********************************************************************
! OO section
! ----------
! Overlap and Orthogonalization, this is where the big GEMMs happen
!***********************************************************************
nv_profile_start(NVP_EDDAV_OO)
          NPOS_RED=ITER*NSIM_+1   ! storage position in WOPT%CW_RED, WHAM%CW_RED
          CORTHO=0
!***********************************************************************
! copy WS, from host to device
!***********************************************************************
nv_profile_start(NVP_OO_MEMCPY)
          call cuda_memset(GPU_CORTHO,0,NB_TOT*NSIM,int(c_sizeof(CORTHO(1,1)),c_size_t))
          call cublas_Set_Matrix(WDES%NPROD_RED,NSIM_,int(c_sizeof(WS%CPROJ_RED(1,1)),c_int),c_loc(WS%CPROJ_RED(1,1)),WDES%NPROD_RED,GPU_WS%CPROW,WDES%NPROD_RED)
nv_profile_stop(NVP_OO_MEMCPY)
!***********************************************************************
! calculate GEMM: CORTHO = WA * WOPT
!***********************************************************************
nv_profile_start(NVP_OO_CORTHO)
nv_profile_start(NVP_GEMM)
#ifdef MIXED_PRECISION_GPU
          CALL ORTH1_GPU_SHIFT_SP('L',GPU_WA%CPTWFP,0,GPU_WOPT_CW_RED(NPOS_RED),0,GPU_WA%CPROJ,0,GPU_WS%CPROW,0,NB_TOT,1,NSIM_,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU_CORTHO,0)
#else
          CALL ORTH1_GPU_SHIFT('L',GPU_WA%CPTWFP,0,GPU_WOPT_CW_RED(NPOS_RED),0,GPU_WA%CPROJ,0,GPU_WS%CPROW,0,NB_TOT,1,NSIM_,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,GPU_CORTHO,0)
#endif
          call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_OO_CORTHO)
!***********************************************************************
! copy CORTHO, from device to host
!***********************************************************************
nv_profile_start(NVP_OO_MEMCPY)
          call cublas_Get_Matrix(NB_TOT,NSIM_,int(c_sizeof(CORTHO(1,1)),c_int),GPU_CORTHO,NB_TOT,c_loc(CORTHO),NB_TOT)  
nv_profile_stop(NVP_OO_MEMCPY)

          CALLMPI( M_sum_g(WDES%COMM_KIN, CORTHO(1,1), NB_TOT*NSIM_))
          
nv_profile_start(NVP_OO_MEMCPY)
               call cublas_Set_Matrix(NB_TOT,NSIM_,int(c_sizeof(CORTHO(1,1)),c_int),c_loc(CORTHO),NB_TOT,GPU_CORTHO,NB_TOT)   
nv_profile_stop(NVP_OO_MEMCPY)

          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "overlap", WDES, CORTHO, SIZE(CORTHO,1), NB_TOT)
             io_end
          ENDIF
!***********************************************************************
! copy CORTHO, from host to device
!***********************************************************************
          IF (WDES1%NPL_RED /=0 ) THEN
!***********************************************************************
! calculate GEMM: WOPT = WA * CORTHO
!***********************************************************************
nv_profile_start(NVP_OO_WOPTCW)
nv_profile_start(NVP_GEMM)
#ifdef MIXED_PRECISION_GPU
               CALL CUBLAS_ZGEMMSH_SP('N','N',m_ WDES1%NPL_RED,NSIM_,NB_TOT,-one,GPU_WA%CPTWFP,0,m_ WDES%NRPLWV_RED,GPU_CORTHO,0,NB_TOT,one,GPU_WOPT_CW_RED(NPOS_RED),0,m_ WDES%NRPLWV_RED)
#else
               CALL CUBLAS_ZGEMMSH('N','N',m_ WDES1%NPL_RED,NSIM_,NB_TOT,-one,GPU_WA%CPTWFP,0, m_ WDES%NRPLWV_RED,GPU_CORTHO,0,NB_TOT,one,GPU_WOPT_CW_RED(NPOS_RED),0,m_ WDES%NRPLWV_RED)
               call threadsynchronize()
#endif
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_OO_WOPTCW)
!DDEBUG
!call fortran_print("cortho_oo"//NODE_MEC//".dat",2,NB_TOT*NSIM_,CORTHO
          ENDIF
!***********************************************************************
! copy WOPT, from host to device
!***********************************************************************
          IF (WDES1%NPRO_RED /= 0) THEN
nv_profile_start(NVP_OO_MEMCPY)
               call cublas_set_matrix( &
                   WDES%NPROD_RED, &
                   NSIM_, &
                   int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_int), &
                   c_loc(WOPT%CPROJ_RED(1,NPOS_RED)), &
                   WDES%NPROD_RED, &
                   GPU_WOPT_CPROJ(1)+IDX(1,NPOS_RED,WDES%NPROD_RED)*int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_size_t), &
                   WDES%NPROD_RED)
nv_profile_stop(NVP_OO_MEMCPY)
!***********************************************************************
! calculate GEMM: WOPT = WA * CORTHO
!***********************************************************************
nv_profile_start(NVP_OO_WOPTCPROJ)
nv_profile_start(NVP_GEMM)
               CALL CUBLAS_ZGEMMSH('N','N',WDES1%NPRO_RED,NSIM_,NB_TOT,-one,GPU_WA%CPROJ,0,WDES%NPROD_RED,GPU_CORTHO,0,NB_TOT,one,GPU_WOPT_CPROJ(1),IDX(1,NPOS_RED,WDES%NPROD_RED),WDES%NPROD_RED)
               call threadsynchronize()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_OO_WOPTCPROJ)
!***********************************************************************
! copy WOPT, from device to host
!***********************************************************************
nv_profile_start(NVP_OO_MEMCPY)
               call cublas_get_matrix( &
                   WDES%NPROD_RED, &
                   NSIM_, &
                   int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_int), &
                   GPU_WOPT_CPROJ(1)+IDX(1,NPOS_RED,WDES%NPROD_RED)*int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_size_t), &
                   WDES%NPROD_RED, &
                   c_loc(WOPT%CPROJ_RED(1,NPOS_RED)), &
                   WDES%NPROD_RED)
               !call cublas_Get_Matrix_shz(WDES%NPROD_RED,NSIM_,int(c_sizeof(WOPT%CPROJ_RED(1,1)),c_size_t),GPU_WOPT_CPROJ(1),IDX(1,NPOS_RED,WDES%NPROD_RED),WDES%NPROD_RED,WOPT%CPROJ_RED(1,NPOS_RED),WDES%NPROD_RED)
               call threadsynchronize()
nv_profile_stop(NVP_OO_MEMCPY)
          ENDIF
nv_profile_stop(NVP_EDDAV_OO)

!***********************************************************************
! W1FFT section
! -------------
! store results back in W1 and peform an FFT to real space
!***********************************************************************
nv_profile_start(NVP_EDDAV_W1FFT)
          ! replaced with lines below
          !CALL W1_COPY( ELEMENT( WOPT, NPP), W1(NP))
          DO NP=1,NSIM_LOCAL_
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW

              ! copy WOPT%CPROJ to W1%CPROJ, from host to host
              IF (WDES1%LGAMMA) THEN
                 CALL DCOPY( WDES1%NPROD, WOPT%CPROJ(1,NPP), 1, W1(NP)%CPROJ(1), 1)
              ELSE
                 CALL ZCOPY( WDES1%NPROD, WOPT%CPROJ(1,NPP), 1, W1(NP)%CPROJ(1), 1)
              ENDIF
          ENDDO
!***********************************************************************
! loop over batches of NV_NUM_STREAMS, as explained earlier
!***********************************************************************
nv_profile_start(NVP_FFT)
nv_profile_start(NVP_REDIS_W1FFT)
          DO NB=1,NSIM_LOCAL_,NV_NUM_STREAMS
!***********************************************************************
! batched memcpys to hide behind mpi
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW
              ! copy WOPT%CW to W1%CW, from device to host, because used in FOCK_ACC!
              call cuda_memcpydtoh(SID,c_loc(W1(NP)%CW),GPU_WOPT_CW(NPP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              SID=SID+1  ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
!***********************************************************************
! redistribute W1 over bands or plane wave coefficients
!***********************************************************************
          SID=0
          DO NP=NB,MIN(NSIM_LOCAL_,NB+NV_NUM_STREAMS-1)
              N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
              NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW

              IF(.NOT. WDES%DO_REDIS) THEN
              ! copy WOPT%CW to W1%CW, from device to device
              call cuda_memcpydtod(SID,GPU_W1_CW(NP),GPU_WOPT_CW(NPP),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              ELSE
              ! redistribute W1%CPROJ over ?
              CALL REDIS_PROJ(WDES1, 1, W1(NP)%CPROJ(1))
              ! synchronize stream with bacthed memcpys above
              call cuda_streamsynchronize(SID)
              ! redistribute W1%CW over ?
              CALL REDIS_PW(WDES1, 1, W1(NP)%CW(1))
              ! copy W1%CW, from host to device
              call cuda_memcpyhtod(SID,GPU_W1_CW(NP),c_loc(W1(NP)%CW),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
              ENDIF

              ! fft to real space, overlap with mpi communication above
              CALL FFTWAV_W1_GPU_STREAM(SID,W1(NP),GPU_W1_CW(NP),GPU_W1_CR(NP),GPU(1)%NINDPW)

              SID=SID+1  ! stream id
              IF(SID>=NV_NUM_STREAMS) SID=0
          ENDDO
          ENDDO
          call threadsynchronize()
nv_profile_stop(NVP_REDIS_W1FFT)
nv_profile_stop(NVP_FFT)
nv_profile_stop(NVP_EDDAV_W1FFT)
!DDEBUG
!call cuda_print("wham_fft"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,GPU_WHAM_CW(1))
!call cuda_print("wopt_fft"//NODE_MEC//".dat",2,WDES1%NRPLWV*NSIM_LOCAL*NITER,GPU_WOPT_CW(1))

!write(*,*) NODE_MEC,iter_bands,BANDSTRUCTURE_ENERGY(WDES, W)
!IF(iter_bands==1 .AND. NNELM==5) THEN
 !call cuda_device_reset()
 !stop
!ENDIF
       ENDDO bands

       IF (INFO%LREAL) THEN
         call DESTROY_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
       ENDIF

!***********************************************************************
! SUBSPACEROT section
! -------------------
! perform the sub space rotation as last step
!***********************************************************************
       subr: IF(LSUBROT) THEN
nv_profile_start(NVP_EDDAV_SUBSPACEROT)
!***********************************************************************
! copy CHAM, from device to host
!***********************************************************************
nv_profile_start(NVP_SUBSPACEROT_MEMCPY)
           IF (.NOT. LscaAWARE) THEN
               call cublas_Get_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM_ALL(1,1)),c_int),GPU_CHAM_ALL,NB_TOT,c_loc(CHAM_ALL),NB_TOT)
           ELSE
               call cublas_Get_Matrix(SCALA_NP(),SCALA_NQ(),int(c_sizeof(CHAM_ALL(1,1)),c_int),GPU_CHAM_ALL,SCALA_NP(),c_loc(CHAM_ALL),SCALA_NP())
           ENDIF
nv_profile_stop(NVP_SUBSPACEROT_MEMCPY)
!***********************************************************************
! CPU work
!***********************************************************************
nv_profile_start(NVP_SUBSPACEROT_ADD)
          ! sum subspace matrix over all nodes
          IF (.NOT. LscaAWARE) THEN
              CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM_ALL(1,1),NB_TOT*NB_TOT))
              ! add lower triangle
              DO N=1,NB_TOT
                  DO NP=N+1,NB_TOT
                      CHAM_ALL(NP,N)=GCONJG(CHAM_ALL(N,NP))
                  ENDDO
              ENDDO
          ENDIF

#ifndef gammareal
          DO N1=1,NB_TOT
              IF (.NOT. LscaAWARE) THEN
                  IF (ABS(AIMAG(CHAM_ALL(N1,N1)))>1E-2_q .AND. IU0>=0) THEN
                      WRITE(IU0,*)'---->WARNING: Sub-Space-Matrix is not hermitian subr',AIMAG(CHAM_ALL(N1,N1)),N1
                  ENDIF
                  CHAM_ALL(N1,N1)= REAL( CHAM_ALL(N1,N1) ,KIND=q)
              ELSE
                  CALL BG_CHANGE_DIAGONALE(WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),IU0)
              ENDIF
          ENDDO
#endif
nv_profile_stop(NVP_SUBSPACEROT_ADD)

          ! here we support only scaLAPACK and LAPACK 
#if defined(MPI)
nv_profile_start(NVP_SUBSPACEROT_ZHEEVX)
          ! use scaLAPACK if available in parallel version
          IF ( LscaLAPACK .AND. .NOT. LscaAWARE) THEN
             CALL pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM_ALL(1,1), R,  NB_TOT, WDES%NB_TOTK(NK,ISP))
              CALLMPI( M_sum_g(WDES%COMM_KIN, CHAM_ALL(1,1),NB_TOT*NB_TOT))
              GOTO 1000
          ELSE IF ( LscaLAPACK ) THEN
              CALL BG_pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM_ALL(1,1), R,  WDES%NB_TOTK(NK,ISP))
              GOTO 1000
          ENDIF
nv_profile_stop(NVP_SUBSPACEROT_ZHEEVX)
#endif
!***********************************************************************
! use MAGMA if available
! warning: MAGMA uses different algorithm than CPU lapack routines!
!***********************************************************************
#ifdef USE_MAGMA
nv_profile_start(NVP_SUBSPACEROT_MEMCPY)
          call cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM_ALL(1,1)),c_int),c_loc(CHAM_ALL),NB_TOT,GPU_CHAM_ALL,NB_TOT)
nv_profile_stop(NVP_SUBSPACEROT_MEMCPY)
#endif

#ifdef gammareal
! gammareal
nv_profile_start(NVP_SUBSPACEROT_DSYEV)
#ifndef USE_ZHEEVX
#ifdef USE_MAGMA
          write(*,*) "WARNING: have not tested magma_dsyevd_gpu yet!"
          call magma_dsyevd_gpu('V','U',WDES%NB_TOTK(NK,ISP),GPU_CHAM_ALL,NB_TOT,R,IFAIL)
#else
          CALL DSYEV('V','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT,R,CWRK,LWORK*NB_TOT,IFAIL)
#endif
#else
          ABSTOL=1E-10_q
          VL=0 ; VU=0 ; IL=0 ; IU=0
#ifdef USE_MAGMA
          write(*,*) "WARNING: have not tested magma_dsyevdx_gpu yet!"
          call magma_dsyevdx_gpu('V','A','U',WDES%NB_TOTK(NK,ISP),GPU_CHAMM_ALL,NB_TOT,VL,VU,IL,IU,NB_CALC,R,IFAIL)
#else
          ALLOCATE(COVL_ALL(NB_TOT,NB_TOT))
          CALL DSYEVX('V','A','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT,VL,VU,IL,IU,ABSTOL, NB_CALC,R,COVL_ALL(1,1),NB_TOT,CWRK,LWORK*NB_TOT,RWORK,IWORK,MINFO,IFAIL)
          CHAM_ALL=COVL_ALL
          DEALLOCATE(COVL_ALL)
#endif
#endif
nv_profile_stop(NVP_SUBSPACEROT_DSYEV)
#else
! no gammareal
#ifndef USE_ZHEEVX
nv_profile_start(NVP_SUBSPACEROT_ZHEEV)
#ifdef USE_MAGMA
          call magma_zheevd_gpu('V','U',WDES%NB_TOTK(NK,ISP),GPU_CHAM_ALL,NB_TOT,R,IFAIL)
#else
          CALL ZHEEV('V','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT,R,CWRK,LWORK*NB_TOT,RWORK,IFAIL)
#endif
nv_profile_stop(NVP_SUBSPACEROT_ZHEEV)
#else
nv_profile_start(NVP_SUBSPACEROT_ZHEEVX)
          ABSTOL=1E-10_q
          VL=0 ; VU=0 ; IL=0 ; IU=0
#ifdef USE_MAGMA
          call magma_zheevdx_gpu('V','A','U',WDES%NB_TOTK(NK,ISP),GPU_CHAM_ALL,NB_TOT,VL,VU,IL,IU,NB_CALC,R,IFAIL)
#else
          ALLOCATE(COVL_ALL(NB_TOT,NB_TOT))
          CALL ZHEEVX( 'V','A','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT,VL,VU,IL,IU,ABSTOL ,NB_CALC,R,COVL_ALL(1,1),NB_TOT,CWRK,LWORK*NB_TOT,RWORK,IWORK,MINFO,IFAIL)
          CHAM_ALL=COVL_ALL
          DEALLOCATE(COVL_ALL)
#endif
nv_profile_stop(NVP_SUBSPACEROT_ZHEEVX)
#endif
#endif
          ! T3D uses a global sum which does not guarantee to give the same results on all nodes
          ! the following line is required to make the code waterproof (we had problems)
          ! since we now use a propritary sum (see mpi.F) we should not require
          ! this broadcast anymore
          !CALLMPI( M_bcast_g(WDES%COMM, CHAM_ALL(1,1), NB_TOT*NB_TOT))
1000      CONTINUE

          IF (IFAIL/=0) THEN
             WRITE(*,*) 'ERROR EDDIAG: Call to routine ZHEEV failed! '// &
                  &              'Error code was ',IFAIL
             WRITE(*,*) ' try to use ALGO = Exact if you use many bands (exact diagonalization)'
             STOP
          ENDIF

          DO N=1,WDES%NB_TOTK(NK,ISP)
              W%CELTOT(N,NK,ISP)=R(N)
          ENDDO
          !psnvidia: not req. once lsubspace rot is on GPU
!***********************************************************************
! copy CHAM and WA, from host to device
!***********************************************************************
#ifndef USE_MAGMA
nv_profile_start(NVP_SUBSPACEROT_MEMCPY)
          call cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM_ALL(1,1)),c_int),c_loc(CHAM_ALL),NB_TOT,GPU_CHAM_ALL,NB_TOT) 
nv_profile_stop(NVP_SUBSPACEROT_MEMCPY)
#endif
          IF (.NOT.LSTOP) THEN 
nv_profile_start(NVP_SUBSPACEROT_MEMCPY)
              call cublas_Set_Matrix(WDES1%NRPLWV_RED,NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),c_loc(WA%CW_RED),WDES1%NRPLWV_RED,GPU_WA%CPTWFP,WDES1%NRPLWV_RED)
              call cublas_Set_Matrix(WDES1%NPROD_RED,NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_int),c_loc(WA%CPROJ_RED),WDES1%NPROD_RED,GPU_WA%CPROJ,WDES1%NPROD_RED)
nv_profile_stop(NVP_SUBSPACEROT_MEMCPY)
          ENDIF
!***********************************************************************
! build linear combinations of wavefunctions
!***********************************************************************
          IF ( .NOT. LscaAWARE) THEN
nv_profile_start(NVP_SUBSPACEROT_LINCOM)
              CALL LINCOM_GPU('F',GPU_WA%CPTWFP,GPU_WA%CPROJ,GPU_CHAM_ALL, &
                   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
                   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,NB_TOT, &
                   GPU_WA%CPTWFP,GPU_WA%CPROJ)
              call threadsynchronize()
nv_profile_stop(NVP_SUBSPACEROT_LINCOM)
          ELSE
             print *,"LINCOM_DISTRI_GPU does not exist yet !!!"
             stop
          ENDIF
          DWRITE "lincom ok"
nv_profile_start(NVP_SUBSPACEROT_MEMCPY)
          call cublas_Get_Matrix(WDES1%NRPLWV_RED,NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),GPU_WA%CPTWFP,WDES1%NRPLWV_RED,c_loc(WA%CW_RED),WDES1%NRPLWV_RED)
          call cublas_Get_Matrix(WDES1%NPROD_RED,NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_int),GPU_WA%CPROJ,WDES1%NPROD_RED,c_loc(WA%CPROJ_RED),WDES1%NPROD_RED)
nv_profile_stop(NVP_SUBSPACEROT_MEMCPY)
nv_profile_stop(NVP_EDDAV_SUBSPACEROT)
       ENDIF subr
!***********************************************************************
! redistribute W over bands or plane wave coefficients
!***********************************************************************
nv_profile_start(NVP_EDDAV_SKEND)
       IF (WDES%DO_REDIS) THEN
nv_profile_start(NVP_REDIS_BANDS)
           CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
           CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
nv_profile_stop(NVP_REDIS_BANDS)
       ENDIF
!***********************************************************************
! free arrays on device
!***********************************************************************
       call cublas_free(GPU_W1_CW_ALL)
       call cublas_free(GPU_W1_CR_ALL)
       call cublas_free(GPU_CWORK_ALL)
       call cublas_free(GPU_PRECON)

       call cublas_free(GPU(1)%DATAKE)
       call cublas_free(GPU(1)%NINDPW)
  
       call cublas_free(GPU_WS%CPROW)

       call cublas_free(GPU_WHAM_CW_ALL)
       call cublas_free(GPU_WOPT_CW_ALL)
       call cublas_free(GPU_WOPT_CPROJ_ALL)
       call cublas_free(GPU_WA%CPTWFP)
       call cublas_free(GPU_WA%CPROJ)
nv_profile_stop(NVP_EDDAV_SKEND)
    END DO kpoints
    ENDDO spin


nv_profile_start(NVP_EDDAV_END)
    call cublas_free(GPU(1)%SV)

#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_CELTOT(WDES,W)
    ENDIF

    IF ((LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
    ENDIF
#endif

    CALLMPI( M_sum_d(WDES%COMM_KINTER, DESUM, 1))
    CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV, 1))

    ! RMS was only calculate for the band treated locally (sum over all nodes)
    CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))
    CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))

    DEALLOCATE(CHAM, COVL, CEIG, COVL_, CORTHO, R, RWORK, CWRK)
    IF (LSUBROT) DEALLOCATE(CHAM_ALL)
    IF (LSUBROT) THEN
       IF (.NOT. LscaAWARE) THEN
          call cublas_free(GPU_CHAM_ALL)
       ELSE
          call cublas_free(GPU_CHAM_ALL)
       ENDIF
    ENDIF
!***********************************************************************
! free arrays on host
!***********************************************************************
#ifdef  USE_ZHEEVX
    IF (LSUBROT) DEALLOCATE(IWORK,MINFO)
#endif
    DO I=1,NSIM_LOCAL
       CALL DELWAV(W1(I), .TRUE.)
    ENDDO

    CALL DELWAVA(WOPT)
    CALL DELWAVA(WHAM)
    CALL DELWAVA_PROJ(WS)

    IF (LHFCALC_DAV) THEN
       CALL DELWAVA(WHF)
       CALL DEALLOCW(W_ORIG)
       CALLMPI( M_sum_d(WDES1%COMM_KIN,EXHF,1))
       CALLMPI( M_sum_d(WDES1%COMM_KINTER,EXHF,1))
       IF (PRESENT(EXHF_ACFDT)) THEN
          CALLMPI( M_sum_d(WDES1%COMM_KIN,EXHF_ACFDT,1))
          CALLMPI( M_sum_d(WDES1%COMM_KINTER,EXHF_ACFDT,1))
       ENDIF

    ENDIF
!***********************************************************************
! free arrays on device
!***********************************************************************
    CALL cublas_free(GPU(1)%CHAM)
    CALL cublas_free(GPU(1)%COVL)
    CALL cublas_free(GPU_CORTHO)
    nv_profile_stop(NVP_EDDAV_END)
    nv_profile_stop(NVP_EDDAV)

    RETURN
  END SUBROUTINE EDDAV
END MODULE david
