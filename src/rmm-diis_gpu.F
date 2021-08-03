!#define debug
#include "symbol.inc"
MODULE rmm_diis
  USE prec
  LOGICAL :: USE_PROJALL_GPU = .TRUE. 
  LOGICAL :: USE_HAMIL_GPU   = .TRUE.
  REAL :: REF_PROJALL = 0
  REAL :: REF_HAMIL   = 0
CONTAINS
!************************ SUBROUTINE EDDRMM *****************************
! RCS:  $Id: rmm-diis.F,v 1.7 2002/08/14 13:59:42 kresse Exp $
!
! this subroutine performes an optimization of the trial wavefunctions
! minimizing the expectation value of the Hamiltonian  i.e.
!     < phi | H |  phi >
! or the norm of the residual vector  i.e.
!    r^2 = < phi | H -e S | H - e S | phi >
! or using an inverse iteration method. In the last case
!    || ( H - e_initial S)  | phi > - | phi_initial > ||
! is optimized.
! The full name of the residual vector  minimization method
! is residual vector minimiziation method-
! direct inversion of the iterative subspace (RMM-DIIS)
!    see: D. M. Wood and A. Zunger, J. Phys. A, 1343 (1985)
!    and  P. Pulay,  Chem. Phys. Lett. 73, 393 (1980).
!
!
!  INFO%IALGO   determine type of preconditioning and the algorithm
!    0    inverse iteration         +  TAP preconditioning
!    6    rms-minimization          +  TAP preconditioning
!    7    rms-minimization          +  no preconditioning
!    8    precond rms-minimization  +  TAP preconditioning
!    9    precond rms-minimization  +  Jacobi like preconditioning
!    (TAP Teter Alan Payne)
!  parameters:
!  LDELAY=.TRUE.
!          steepest descent eigenvalue minimization
!          maximum number of steps is 2
!  WEIMIN  treshhold for total energy minimisation
!    is the fermiweight of a band < WEIMIN,
!    minimisation will break after a maximum of two iterations
!  EBREAK  absolut break condition
!    intra-band minimisation is stopped if DE is < EBREAK
!  DEPER   intra-band break condition (see below)
!  ICOUEV  number of intraband evalue minimisations
!  DESUM   total change in eigenvalues
!  RMS     norm of residual vector
!
! GPU part : HACENE Mohamed
!***********************************************************************

  SUBROUTINE EDDRMM(HAMILTONIAN,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
       LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV,IU6,IU0, LDELAY)
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
    USE wave_mpi
    USE gpu_data
    USE c2f_interface, ONLY : VTIME
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    REAL(q), TARGET :: A2
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (grid_3d)     GRID
    TYPE (info_struct) INFO
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES

    LOGICAL LDELAY
    RGRID, TARGET ::  SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

    !----- local work arrays
    TYPE (wavedes1)    WDES1          ! descriptor for one k-point
    TYPE (wavefun1)    W1(WDES%NSIM)  ! current wavefunction
    TYPE (wavefun1)    W1_TMP         ! temporary variable
    TYPE (wavefun1)    WTMP(WDES%NSIM)! temporary trial wavefunction
    REAL(q) R(INFO%NDAV)
    PARAMETER  (LWORK=20)
    DIMENSION CWORK(LWORK*INFO%NDAV)
    DIMENSION RWORK(3*INFO%NDAV)
    ! work arrays
    TYPE(wavefuna) :: W_INI, WOPT
    REAL(q),ALLOCATABLE, TARGET:: PRECON(:,:)
    GDEF,ALLOCATABLE:: CHAM(:,:),CTMP(:,:),CWORK1(:)
    GDEF,ALLOCATABLE:: B(:),B_(:)
    INTEGER,ALLOCATABLE :: IPIV(:)
    INTEGER :: NB(WDES%NSIM)         ! contains a list of bands currently optimized
    REAL(q) :: EVALUE_INI(WDES%NSIM) ! eigenvalue of that band at the beginning
    REAL(q) :: EVALUE(WDES%NSIM)     ! eigenvalue during optimization
    REAL(q) :: DEIT(WDES%NSIM)       ! relative break criterion for that band
    REAL(q) :: IT(WDES%NSIM)         ! current iteration for this band
    REAL(q) :: FPRE(WDES%NSIM)       ! norm of residual vector for each band
    REAL(q) :: TRIAL(WDES%NSIM)      ! trial step for each band
    LOGICAL :: LSTOP
    LOGICAL :: LABORT(WDES%NSIM)     ! abort iteration on this band
    TYPE (REDIS_PW_CTR),POINTER :: H_PW
    !Use GPU 
    TYPE(gpu_type) GPU(WDES%NSIM)
    TYPE(gpu_type) HOSTS(WDES%NSIM)
    INTEGER stream(WDES%NSIM)
    INTEGER SIZE_OF_CW, ARBITRARY_VALUE
    !INTEGER SIZE_OF_VECTORS(4)
    REAL(q) TV,TV0,TC,TC0,TC1,TV1,TimePROJALL,TimePrecond,TimeHamil,TimeECCP
    REAL(q) TimeFFTWAV,TimeBlas,TimeTransHAMIL,TimeTrans,TimeTransPROJALL,TimeLapack,TimeInit,TTHamil
    REAL(q) faker
    COMPLEX(q) fakec
    INTEGER(c_intptr_t) :: GPU_CW_ARRAY(WDES%NSIM)
    INTEGER(c_intptr_t) :: WOPT_GPU(WDES%NSIM, 2*INFO%NDAV)
    INTEGER(c_intptr_t) :: WTMP_GPU(WDES%NSIM)
    INTEGER(c_intptr_t) :: A2_GPU
    INTEGER(c_intptr_t) :: GPU_W1_CR(WDES%NSIM)
    INTEGER(c_intptr_t) GPU_CWORK(WDES%NSIM),GPU_CWORK_ALL
    INTEGER(c_intptr_t) GPU_PRECON

!JBNV Variables required for the reordering of the i3 loop
    COMPLEX(q), TARGET :: A2_CPU_RES(WDES%NSIM*3)
    INTEGER(c_intptr_t) fakep 
    INTEGER(c_intptr_t) :: DotArguments_GPU
    INTEGER(c_intptr_t), TARGET :: DotArguments_CPU(WDES%NSIM*4*2) ! 4 Arg per NSIM, 2 Sets of vectors
    INTEGER DotArgumentsIndex        ! The number of arguments stored in  DotArguments 

    INTEGER(c_intptr_t), TARGET :: CombinedDAXPYArguments_CPU(WDES%NSIM*3*2) ! NSIM, 3 arg, for 2 daxpys
    COMPLEX(q), TARGET :: CombinedDAXPYArgumentsAlpha_CPU(WDES%NSIM*2) ! NSIM, for 2 daxpys
    INTEGER(c_intptr_t) :: CombinedDAXPYArguments_GPU !GPU Pointer        
    INTEGER(c_intptr_t) :: CombinedDAXPYArgumentsAlpha_GPU !GPU Pointer        
    INTEGER CDArgumentsIndex                 ! Number arguments stored for CombinedDAXPYArguments   

    INTEGER NSIMActive        ! The number of NSIM bands that are actually active in this iteration

    REAL(q) :: WSCAL_ARR(WDES%NSIM)        ! Partial results per NSIM

    INTEGER SIZE_SECOND_DAXPY_PART  ! Used in the combined daxpy call to specify
                                    ! the size of second set of vectors

    INTEGER(c_intptr_t) :: LOCAL_CONTRIB_RES_GPU
    COMPLEX(8), TARGET :: LOCAL_CONTRIB_RES_CPU(WDES%NSIM*2) ! Two types of result per nsim iteration 
    COMPLEX(8) :: NON_LOCAL_CONTRIB_RES_CPU(WDES%NSIM) ! One types of result per nsim iteration 


    INTEGER(c_size_t) :: BATCHED_MEMORY_REQUEST_SIZE
    INTEGER(c_size_t) :: BATCHED_MEMORY_OFFSET
    INTEGER(c_size_t) :: BATCHED_MEMORY_REMAINING


    REAL(q) :: EKIN(WDES%NSIM)
    LOGICAL :: PREV_ITER_ON_GPU(WDES%NSIM)
! DDNVIDIA local variables
    INTEGER,PARAMETER :: STREAM_SIZE=4
    INTEGER           :: SID
    INTEGER           :: NEWBAND_LAST_DONE
    INTEGER           :: NEWBAND_LAST_DONE_N

    TimePROJALL      = 0
    TimePrecond      = 0
    TimeHamil        = 0
    TimeECCP         = 0
    TimeFFTWAV       = 0
    TimeBlas         = 0
    TIMETRANSHAMIL   = 0 
    TTHamil          = 0
    TIMETRANS        = 0
    TimeTransPROJALL = 0
    TimeLapack       = 0
    TimeInit         = 0

    CALL VTIME(TV0,TC0)
   
    CALL VTIME(TV1,TC1)
    ARBITRARY_VALUE = 500000000
    


    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=WDES%COMM%NODE_ME
    IONODE =WDES%COMM%IONODE
#endif
!=======================================================================
!  INITIALISATION:
! maximum  number of iterations
! NRES position where H - E S| trial vector > is stored
!=======================================================================
    NSIM=WDES%NSIM
    NITER=INFO%NDAV
    IF (LDELAY) NITER=MIN(NITER,1)
    IF (LDELAY .AND. INFO%IALGO ==0) NITER=1
    NRES =INFO%NDAV

    DESUM =0
    RMS   =0
    ICOUEV=0

    SLOCAL=0
    DO I=1,GRID%RL%NP
       SLOCAL=SLOCAL+SV(I,1)
    ENDDO

    CALLMPI( M_sum_d(WDES%COMM_INB, SLOCAL, 1))
    SLOCAL=SLOCAL/GRID%NPLWV


    ALLOCATE(PRECON(WDES%NRPLWV,NSIM), &
         &        CHAM(NRES,NRES),CTMP(NRES,NRES),CWORK1(NRES), &
         &        B(NRES),B_(NRES),IPIV(NRES))

    CALL SETWDES(WDES,WDES1,0)
    CALL NEWWAVA(W_INI, WDES1, NSIM)
    CALL NEWWAVA(WOPT,  WDES1, NRES*2, NSIM)
    DO NP=1,NSIM
       CALL NEWWAV(WTMP(NP), WDES1, .TRUE.)
       CALL NEWWAV_R(W1(NP), WDES1)
    ENDDO
    CTMP=0
    CHAM=0

    call cublas_alloc_safety (NSIM*3,int(c_sizeof(fakec),c_size_t),A2_GPU)


    ! Allocate the variables for the i3 loop
    call cublas_alloc_safety (NSIM*4*2,int(c_sizeof(fakep),c_size_t),DotArguments_GPU) !8 is 64 byte = pointer
    call cublas_alloc_safety (NSIM*3*2,int(c_sizeof(fakep),c_size_t),CombinedDAXPYArguments_GPU) !8 is 64 byte = pointer
    call cublas_alloc_safety (NSIM*2,int(c_sizeof(fakec),c_size_t),CombinedDAXPYArgumentsAlpha_GPU) 
    call cublas_alloc_safety (NSIM*2,int(c_sizeof(fakec),c_size_t),LOCAL_CONTRIB_RES_GPU)



#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
write(*,*) "RMM-DIIS..."
write(*,*) "######################################################################"
#ifdef MPI
 endif
#endif
#endif
!=======================================================================
! do we have to distribute the wavefunctions back ?
!=======================================================================
    IF (W%OVER_BAND) THEN
#ifdef MPI
       NCPU=WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
       NCPU=1
#endif
       NSTRIP=MIN(NSIM,WDES%NBANDS)*2
       CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PW)
    ENDIF
    CALL VTIME(TV,TC)   
    TimeInit = TimeInit + TC - TC1
!=======================================================================
    spin:    DO ISP=1,WDES%ISPIN
    kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
    IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
    CALL VTIME(TV1,TC1)
    CALL SETWDES(WDES,WDES1,NK)
!=======================================================================
    !  first initiate communication between bands
    IF (W%OVER_BAND) THEN
       DO N=1,NSTRIP
          CALL REDIS_PW_START(WDES, W%CW(1,N,NK,ISP), N, H_PW)
       ENDDO
    ENDIF
!=======================================================================
    DE_ATT=ABS(W%CELTOT(WDES%NB_TOTK(NK,ISP),NK,ISP)-W%CELTOT(1,NK,ISP))/4

    IF (INFO%LREAL) THEN
       CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
       CALL VTIME(TV1,TC1)
       call INIT_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
       CALL VTIME(TV,TC)
       TimeTrans = TimeTrans + TC - TC1
    ELSE
       CALL PHASE(WDES,NONL_S,NK)
    ENDIF
    NB=0          ! empty the list of bands, which are optimized currently
    NB_DONE=0     ! index the bands allready optimised
!=======================================================================
    !GPU allocation GPU%CW1,GPU%CR1 <- WTMP
    !               GPU%CW2,GPU%CR2 <- W1

    CALL VTIME(TV,TC)   
    TimeInit = TimeInit + TC - TC1
    
    SIZE_OF_CW = WDES1%NRPLWV
    
    
    CALL VTIME(TV,TC)   
    
    
    !The below allocates a single piece of memory and then distributes this. This reduces the number of calls to malloc/free 
    !which improves the performance on a single GPU and even more on multi GPU systems.
    
    BATCHED_MEMORY_REQUEST_SIZE = 0
    
    !Compute required size
    BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM*int(c_sizeof(fakec),c_size_t)      
    DO NP=1,NSIM
      BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + SIZE_OF_CW*int(c_sizeof(fakec),c_size_t)
      BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + SIZE_OF_CW*int(c_sizeof(fakec),c_size_t)
      BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + SIZE_OF_CW*int(c_sizeof(fakec),c_size_t)
      BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t)
      DO I=1, 2*NRES
        BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + SIZE_OF_CW*int(c_sizeof(fakec),c_size_t)
      ENDDO
    ENDDO      
    BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + DIMREAL(GRID%MPLWV)*WDES%NCDIJ*int(c_sizeof(SV(1,1)),c_size_t)
    BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + WDES1%NGDIM*WDES1%NRSPINORS*int(c_sizeof(WDES1%DATAKE(1,1)),c_size_t)
    BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE + WDES1%NGVECTOR*int(c_sizeof(GPU(1)%NINDPW),c_size_t)
    BATCHED_MEMORY_REQUEST_SIZE = BATCHED_MEMORY_REQUEST_SIZE 

    ! allocate device memory
    call cublas_Alloc_safety(WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM,int(c_sizeof(fakec),c_size_t),GPU_CWORK_ALL)
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM,int(c_sizeof(faker),c_size_t),GPU_PRECON)
    ! set device pointers
    DO NP=1,NSIM
        GPU_CWORK(NP) = GPU_CWORK_ALL + (NP-1)*WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t)
    ENDDO
  
    !Allocate the single buffer
    call allocate_gpu_memory_batched_init(BATCHED_MEMORY_REQUEST_SIZE, BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING, GPU(1)%CR1_ALL)
  
    !Hand out the pieces
    call allocate_gpu_memory_batched_request (WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM*int(c_sizeof(fakec),c_size_t),BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(1)%CR1_ALL)
    DO NP=1,NSIM
      call allocate_gpu_memory_batched_request (SIZE_OF_CW*int(c_sizeof(fakec),c_size_t), BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, WTMP_GPU(NP))
      call allocate_gpu_memory_batched_request (SIZE_OF_CW*int(c_sizeof(fakec),c_size_t), BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(NP)%CW1)
      call allocate_gpu_memory_batched_request (SIZE_OF_CW*int(c_sizeof(fakec),c_size_t), BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(NP)%CW2)

      GPU(NP)%CR1 = GPU(1)%CR1_ALL + WDES1%GRID%MPLWV*WDES1%NRSPINORS*(NP-1)*int(c_sizeof(fakec),c_size_t)
      GPU_W1_CR(NP) = GPU(NP)%CR1
      call allocate_gpu_memory_batched_request (WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t), BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(NP)%CR2)
      DO I=1, 2*NRES
          call allocate_gpu_memory_batched_request(SIZE_OF_CW*int(c_sizeof(fakec),c_size_t), BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, WOPT_GPU(NP, I))
      ENDDO
    ENDDO
    
  
    call allocate_gpu_memory_batched_request (DIMREAL(GRID%MPLWV)*WDES%NCDIJ*int(c_sizeof(SV(1,1)),c_size_t),  BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(1)%SV)
    call allocate_gpu_memory_batched_request (WDES1%NGDIM*WDES1%NRSPINORS*int(c_sizeof(WDES1%DATAKE(1,1)),c_size_t),  BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(1)%DATAKE)
    call allocate_gpu_memory_batched_request (WDES1%NGVECTOR*int(c_sizeof(GPU(1)%NINDPW),c_size_t),  BATCHED_MEMORY_OFFSET, BATCHED_MEMORY_REMAINING,GPU(1)%CR1_ALL, GPU(1)%NINDPW)    
    
    CALL VTIME(TV1,TC1)  
    !write(*,*) 'New allocation took: ', TC1-TC

    ! allocate local arrays in RACCMU and RPROMU
    !call INIT_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
    
    ITER=IT(1)
    !print *,"ITER =",ITER,ITER*ITER
    call cublas_Set_Vector (WDES1%NGVECTOR,int(c_sizeof(WDES1%NINDPW(1)),c_int),c_loc(WDES1%NINDPW),1,GPU(1)%NINDPW,1)
    call cublas_Set_Vector (DIMREAL(GRID%MPLWV)*WDES%NCDIJ,int(c_sizeof(SV(1,1)),c_int),c_loc(SV),1,GPU(1)%SV,1)
    call cublas_Set_Vector (WDES1%NGDIM*WDES1%NRSPINORS,int(c_sizeof(WDES1%DATAKE(1,1)),c_int),c_loc(WDES1%DATAKE),1,GPU(1)%DATAKE,1)
    CALL VTIME(TV,TC)   
    TimeTrans = TimeTrans + TC - TC1

    PREV_ITER_ON_GPU = .FALSE.
    bands: DO
       !
       !  check the NB list, whether there is any empty slot
       !  fill in a not yet optimized wavefunction into the slot
       !
       NEWBAND_LAST_DONE = -1
       newband: DO NP=1,NSIM
          IF (NB(NP)==0 .AND.  NB_DONE < WDES%NBANDS ) THEN
            GPU_CW_ARRAY(NP) = WOPT_GPU(NP,1)
            PREV_ITER_ON_GPU(NP) = .FALSE.
          ELSE
            GPU_CW_ARRAY(NP) = 0;
          END IF
          IF (NB(NP)==0 .AND.  NB_DONE < WDES%NBANDS ) THEN
             CALL VTIME(TV1,TC1)
             NB_DONE=NB_DONE+1
             N     =NB_DONE
             NB(NP)=NB_DONE
             LABORT(NP)=.FALSE.

             IF (W%OVER_BAND) THEN
                CALL REDIS_PW_STOP (WDES, W%CW(1,N,NK,ISP), N, H_PW)
                IF (N+NSTRIP<=WDES%NBANDS) &
                     CALL REDIS_PW_START(WDES, W%CW(1,N+NSTRIP,NK,ISP), N+NSTRIP, H_PW)
             ENDIF
             CALL SETWAV(W,W1(NP),WDES1,N,ISP)  ! fill band N into W1(NP)
             CALL W1_COPY(W1(NP), ELEMENT(W_INI, NP))

             IDUMP=0
#ifdef debug
             IDUMP=2
#endif
#ifdef MPI
             IF (NODE_ME /= IONODE) IDUMP=0
#endif
             CALL VTIME(TV,TC)   
             TimeInit = TimeInit + TC - TC1
             IF (IDUMP==2) WRITE(*,'(I3,1X)',ADVANCE='NO') N

             CALL VTIME(TV1,TC1)
             IF (NEWBAND_LAST_DONE == -1) THEN
                call cublas_Set_Vector(SIZE_OF_CW,int(c_sizeof(fakec),c_int),c_loc(W1(NP)%CW),1,WOPT_GPU(NP,1),1)
             ELSE
                call cuda_memcpyhtod(NULL_STREAM,WOPT_GPU(NP,1), c_loc(W1(NP)%CW),SIZE_OF_CW,int(c_sizeof(fakec),c_size_t))
             ENDIF
             CALL VTIME(TV,TC)   
             TimeTrans = TimeTrans + TC - TC1
             
             IF (NEWBAND_LAST_DONE /= -1) THEN

             ! start with FFT and the exact evaluation of the eigenenergy
             CALL VTIME(TV1,TC1)
             CALL FFTWAV_W1_GPU_STREAM(NULL_STREAM,W1(NEWBAND_LAST_DONE),WOPT_GPU(NEWBAND_LAST_DONE,1), GPU(NEWBAND_LAST_DONE)%CR2, GPU(1)%NINDPW)
             CALL VTIME(TV,TC)
             TimeFFTWAV = TimeFFTWAV + TC - TC1   
             CALL VTIME(TV1,TC1)
             IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                write(*,*) 'Non gpu version for ECCP_TAU'
                call cuda_device_reset()
                stop
             ELSE
                 CALL ECCP_GPU(WDES1,W1(NEWBAND_LAST_DONE),W1(NEWBAND_LAST_DONE), &
                               LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),W%CELEN(NEWBAND_LAST_DONE_N,NK,ISP), &
                               GPU(NEWBAND_LAST_DONE)%CR2,GPU(NEWBAND_LAST_DONE)%CR2,GPU(1)%SV, &
                               IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),WOPT_GPU(NEWBAND_LAST_DONE,1), &
                               WOPT_GPU(NEWBAND_LAST_DONE,1),GPU(1)%DATAKE)
             CALL VTIME(TV,TC)
             TimeECCP = TimeECCP + TC - TC1 
             ENDIF
             !EVALUE_INI(NP)=W%CELEN(N,NK,ISP)
             EVALUE_INI(NEWBAND_LAST_DONE)=W%CELEN(NEWBAND_LAST_DONE_N,NK,ISP)

             IF (IDUMP==2) WRITE(*,'(F9.4,"E")',ADVANCE='NO') REAL( W%CELEN(N,NK,ISP) ,KIND=q)

             ! calculate the preconditioning matrix
             CALL VTIME(TV1,TC1)
             call cuda_truncatehighfrequency( &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                merge(1,0,LDELAY.OR.WDES1%LSPIRAL), &
                WOPT_GPU(NEWBAND_LAST_DONE,1), &
                GPU(1)%DATAKE, &
                INFO%ENINI)
             !CALL  TRUNCATE_HIGH_FREQUENCY_W1_GPU_NEW( W1(NEWBAND_LAST_DONE), WOPT_GPU(NEWBAND_LAST_DONE,1), LDELAY, INFO%ENINI, GPU)

             call cuda_memcpydtod(NULL_STREAM,GPU(NEWBAND_LAST_DONE)%CW2,WOPT_GPU(NEWBAND_LAST_DONE,1),WDES1%NGVECTOR*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t))
             IPREC=INFO%IALGO
             IF (LDELAY) IPREC=8
               !TODO: PORT TO GPU - NO TEST CASE!!!
               !CALL SETUP_PRECOND( W1(NP), IPREC, IDUMP, PRECON(1,NP), & 
               !   EVALUE_INI(NP)-SLOCAL, DE_ATT )
               !DEIT(NP)=0
               !IT(NP)  =0
               ! Ani verifier
             if (IPREC /= 0 .and. IPREC /= 8 .and. IPREC /= 6) then
             CALL SETUP_PRECOND( W1(NEWBAND_LAST_DONE), IPREC, IDUMP, PRECON(1,NEWBAND_LAST_DONE), & 
                  EVALUE_INI(NEWBAND_LAST_DONE)-SLOCAL, DE_ATT )
             end if  !TODO: PORT TO GPU - NO TEST CASE!!!
             DEIT(NEWBAND_LAST_DONE)=0
             IT(NEWBAND_LAST_DONE)  =0
               CALL VTIME(TV,TC)   
               TimeInit = TimeInit + TC - TC1
             ENDIF
               NEWBAND_LAST_DONE = NP
               NEWBAND_LAST_DONE_N = N
          ENDIF
       ENDDO newband
       IF (NEWBAND_LAST_DONE /= -1) THEN
             ! start with FFT and the exact evaluation of the eigenenergy
             CALL VTIME(TV1,TC1)
             CALL FFTWAV_W1_GPU_STREAM(NULL_STREAM,W1(NEWBAND_LAST_DONE),WOPT_GPU(NEWBAND_LAST_DONE,1), GPU(NEWBAND_LAST_DONE)%CR2, GPU(1)%NINDPW)
             CALL VTIME(TV,TC)
             TimeFFTWAV = TimeFFTWAV + TC - TC1

             CALL VTIME(TV1,TC1)
             CALL ECCP_GPU(WDES1,W1(NEWBAND_LAST_DONE),W1(NEWBAND_LAST_DONE),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),W%CELEN(NEWBAND_LAST_DONE_N,NK,ISP),&
                           GPU(NEWBAND_LAST_DONE)%CR2,GPU(NEWBAND_LAST_DONE)%CR2,GPU(1)%SV,IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),WOPT_GPU(NEWBAND_LAST_DONE,1),WOPT_GPU(NEWBAND_LAST_DONE,1),GPU(1)%DATAKE)
             CALL VTIME(TV,TC)
             TimeECCP = TimeECCP + TC - TC1

             EVALUE_INI(NEWBAND_LAST_DONE)=W%CELEN(NEWBAND_LAST_DONE_N,NK,ISP)
             IF (IDUMP==2) WRITE(*,'(F9.4,"E")',ADVANCE='NO') REAL( W%CELEN(NEWBAND_LAST_DONE_N,NK,ISP) ,KIND=q)

             ! calculate the preconditioning matrix
             CALL VTIME(TV1,TC1)
             call cuda_truncatehighfrequency( &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                merge(1,0, LDELAY.OR.WDES1%LSPIRAL), &
                WOPT_GPU(NEWBAND_LAST_DONE,1), &
                GPU(1)%DATAKE, &
                INFO%ENINI)
             !CALL  TRUNCATE_HIGH_FREQUENCY_W1_GPU_NEW( W1(NEWBAND_LAST_DONE), WOPT_GPU(NEWBAND_LAST_DONE,1), LDELAY, INFO%ENINI, GPU)

             call cuda_memcpydtod(NULL_STREAM,GPU(NEWBAND_LAST_DONE)%CW2,WOPT_GPU(NEWBAND_LAST_DONE,1),WDES1%NGVECTOR*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t))
             IPREC=INFO%IALGO
             IF (LDELAY) IPREC=8
             if (IPREC /= 0 .and. IPREC /= 8 .and. IPREC /= 6) then
             CALL SETUP_PRECOND( W1(NEWBAND_LAST_DONE), IPREC, IDUMP, PRECON(1,NEWBAND_LAST_DONE),EVALUE_INI(NEWBAND_LAST_DONE)-SLOCAL, DE_ATT )
             end if  !TODO: PORT TO GPU - NO TEST CASE!!!
             DEIT(NEWBAND_LAST_DONE)=0
             IT(NEWBAND_LAST_DONE)  =0
             CALL VTIME(TV,TC)   
             TimeInit = TimeInit + TC - TC1
       ENDIF
        if (INFO%IALGO == 0 .OR. INFO%IALGO == 8 .or. INFO%IALGO == 6) then
            CALL SETUP_PRECOND_GPU(WDES1,NSIM,INFO%IALGO,EVALUE_INI,SLOCAL,DE_ATT,GPU_CW_ARRAY,GPU(1)%DATAKE,GPU_PRECON)
        else
            call cuda_memcpyhtod(NULL_STREAM,GPU_PRECON,c_loc(PRECON(1,1)),NSIM*WDES1%NRPLWV,int(c_sizeof(faker),c_size_t))
        end if
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
       CALL VTIME(TV1,TC1)
       LSTOP=.TRUE.
       W1%LDO  =.FALSE.
       DO NP=1,NSIM
          IF ( NB(NP) /= 0 ) THEN
             LSTOP  =.FALSE.
             W1(NP)%LDO=.TRUE.     ! band not finished yet
             IT(NP) =IT(NP)+1   ! increase iteration count
          ENDIF
       ENDDO
       IF (LSTOP) EXIT bands
       CALL VTIME(TV,TC)
       TimeInit = TimeInit + TC - TC1
!=======================================================================
! intra-band minimisation
!=======================================================================
       CALL VTIME(TV1,TC1)
       i1: DO NP=1,NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i1
          ! fill current wavefunctions into work arrays WOPT at position (ITER,NP)
          !CALL W1_COPY(W1(NP), ELEMENT(WOPT, ITER, NP))
          CALL W1_COPY_WITHOUT_CW(W1(NP), ELEMENT(WOPT, ITER, NP))
          IF (GPU_CW_ARRAY(NP) == 0 .AND. PREV_ITER_ON_GPU(NP)) THEN
          call cuda_memcpydtod(NULL_STREAM,WOPT_GPU(NP,ITER),GPU(NP)%CW2,SIZE_OF_CW,int(c_sizeof(fakec),c_size_t))
          ENDIF
          PREV_ITER_ON_GPU(NP) = .TRUE.
          EVALUE(NP)=W%CELEN(N,NK,ISP)
       ENDDO i1
       CALL VTIME(TV,TC)   
       TimeInit = TimeInit + TC - TC1
       !  store H-epsilon Q | psi > temporarily in upmost storage positions (2*NRES)
       !  to have uniform stride for result array see ELEMENTS_SECOND
       CALL VTIME(TV1,TC1)

       IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
           write(*,*) 'No gpu version for HAMILTMU_TAU'  
           call cuda_device_reset()
           stop
       ELSE
          CALL HAMILTMU_GPU_NEW(WDES1,W1,NONLR_S,NONL_S,EVALUE_INI,CDIJ,CQIJ,SV,ISP,ELEMENTS_SECOND(WOPT, 2*NRES),TIMETRANSHAMIL,GPU,GPU_CWORK,WOPT_GPU(:,2*NRES),WOPT_GPU,IT,NRES)
       ENDIF
       CALL VTIME(TV,TC)
       TimeHamil = TimeHamil + (TC - TC1)

! DDNVIDIA stream index
       SID=0
       i2: DO NP=1,NSIM
          CALL VTIME(TV1,TC1)
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i2
          ! copy to proper storage position (NRES+ITER, NP)
          !CALL W1_COPY(ELEMENT(WOPT, 2*NRES, NP), ELEMENT(WOPT, NRES+ITER, NP) )
          CALL W1_COPY_WITHOUT_CW(ELEMENT(WOPT, 2*NRES, NP), ELEMENT(WOPT, NRES+ITER, NP) )

          !CALL TRUNCATE_HIGH_FREQUENCY_W1( ELEMENT( WOPT, NRES+ITER, NP), LDELAY, INFO%ENINI)
          !CALL PW_NORM_WITH_METRIC_W1(ELEMENT( WOPT, NRES+ITER, NP), FNORM, FPRE(NP), PRECON(1,NP))
          call cuda_memcpydtod(NULL_STREAM,WOPT_GPU(NP,NRES+ITER),WOPT_GPU(NP,2*NRES),WDES1%NGVECTOR*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t))
          !        WDES1%NGVECTOR*WDES1%NRSPINORS, int(c_sizeof(fakec),c_size_t)) !(PL) BEZ sensu po zmianie w HAMILTMU
          call cuda_truncatehighfrequency( &
                WDES1%NGVECTOR*WDES1%NRSPINORS, &
                merge(1,0,LDELAY.OR.WDES1%LSPIRAL), &
                WOPT_GPU(NP,NRES+ITER), &
                GPU(1)%DATAKE, &
                INFO%ENINI)
          !CALL TRUNCATE_HIGH_FREQUENCY_W1_GPU_NEW( ELEMENT( WOPT, NRES+ITER, NP), WOPT_GPU(NP, NRES+ITER), LDELAY, INFO%ENINI, GPU)
          call cuda_normwithmetric(NULL_STREAM,WDES1%NGVECTOR*WDES1%NRSPINORS,WOPT_GPU(NP,NRES+ITER),GPU_PRECON,FNORM,FPRE(NP),1,(NP-1)*WDES1%NRPLWV)
          call threadsynchronize()
          CALLMPI(M_sum_s(WDES1%COMM_INB, 2, FNORM, FMETRIC, 0._q,0._q))
          !CALL PW_NORM_WITH_METRIC_W1_GPU_X(ELEMENT( WOPT, NRES+ITER, NP), WOPT_GPU(NP, NRES+ITER), FNORM, FPRE(NP), (NP-1)*WDES%NRPLWV)
          !write(*,*) "GPU FNORM=",FNORM
          W1_TMP = ELEMENT(WOPT, NRES+ITER, NP)

          IF (INFO%IALGO==6)  FPRE(NP)=FNORM
          IF (IDUMP==2) WRITE(*,'(E9.2,"R")',ADVANCE='NO') SQRT(ABS(FNORM))
          IF (ITER==1) THEN
             RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)* &
                  &      SQRT(ABS(FNORM))/WDES%NB_TOT*WDES%NRSPINORS
          ENDIF
          ! norm of total error vector before start
          ! norm smaller than EBREAK stop |e -e(app)| < | Residuum |
          IF (ABS(FNORM)<INFO%EBREAK) THEN
             W1(NP)%LDO=.FALSE.
             CYCLE i2
          ENDIF
          !----------------------------------------------------------------------
          CALL VTIME(TV,TC)   
          TimeInit = TimeInit + TC - TC1

          optsubspace: IF (.NOT. LDELAY .AND. ITER > 1) THEN
             ! better conditioning for search
             ! w(iter-1)=w(iter)-w(iter-1)
             !CALL W1_DSCAL( ELEMENT( WOPT, ITER-1, NP), -1.0_q)
             !CALL W1_DAXPY( ELEMENT( WOPT, ITER, NP), 1.0_q, ELEMENT( WOPT, ITER-1, NP)) 
             CALL VTIME(TV1,TC1)

             call cuda_zsub(SIZE_OF_CW,WOPT_GPU(NP,ITER-1),WOPT_GPU(NP,ITER),WOPT_GPU(NP,ITER-1))
             !call gpu_subtract_cw(WOPT_GPU(NP, ITER-1), WOPT_GPU(NP,ITER),SIZE_OF_CW)
             !call cuda_print("woptgood.dat",'c',SIZE_OF_CW,WOPT_GPU(NP, ITER-1))

             CALL W1_DSCAL_WITHOUT_CW( ELEMENT( WOPT, ITER-1, NP), -1.0_q)
             CALL W1_DAXPY_WITHOUT_CW( ELEMENT( WOPT, ITER, NP), 1.0_q, ELEMENT( WOPT, ITER-1, NP)) 
             ! gradient(iter-1)=gradient(iter)-gradient(iter-1)
             !CALL W1_DSCAL( ELEMENT( WOPT, NRES+ITER-1, NP), -1.0_q)
             !CALL W1_DAXPY( ELEMENT( WOPT, NRES+ITER, NP), 1.0_q, ELEMENT( WOPT, NRES+ITER-1, NP))
             call cuda_zsub(SIZE_OF_CW,WOPT_GPU(NP,NRES+ITER-1),WOPT_GPU(NP,NRES+ITER),WOPT_GPU(NP,NRES+ITER-1))
             !call gpu_subtract_cw(WOPT_GPU(NP, NRES+ITER-1), WOPT_GPU(NP,NRES+ITER),SIZE_OF_CW)
             CALL W1_DSCAL_WITHOUT_CW( ELEMENT( WOPT, NRES+ITER-1, NP), -1.0_q)
             CALL W1_DAXPY_WITHOUT_CW( ELEMENT( WOPT, NRES+ITER, NP), 1.0_q, ELEMENT( WOPT, NRES+ITER-1, NP)) 

             CALL VTIME(TV,TC)
             TimeBlas = TimeBlas + TC - TC1

!***********************************************************************
! RMM-DIIS step
!
! minimize norm of residual vector in the subspace spanned by
! the set of wavefunctions stored in WOPT
! residual vectors are stored in WOPT starting at NRES
!
!***********************************************************************
             optsub: IF (INFO%IALGO /=0) THEN
!----------------------------------------------------------------------
! calculate  matrix INFO%IALGO=7 or 8
! CHAM(n2,n1)= <R(n2)| preconditioning |R(n1)>
! CTMP(n2,n1)= <phi(n2)| S |phi(n1)>
!----------------------------------------------------------------------
                CTMP=0
                CHAM=0
                buildh: DO N1=1,ITER
                   IF (INFO%IALGO==8 .OR. INFO%IALGO==9) THEN
!                   IF (INFO%IALGO==8) THEN
                      !CALL APPLY_PRECOND( ELEMENT(WOPT, NRES+N1,NP), WTMP(NP), PRECON(1,NP))
                      CALL VTIME(TV1,TC1)
                      call cuda_applyprecond(NULL_STREAM,WDES1%NPL,WOPT_GPU(NP,NRES+N1),WTMP_GPU(NP),GPU_PRECON,1.0_q,(NP-1)*WDES%NRPLWV)
                      !CALL APPLY_PRECOND_GPU_NEW( ELEMENT(WOPT, NRES+N1,NP), WOPT_GPU(NP, NRES+N1), WTMP_GPU(NP), (NP-1)*WDES%NRPLWV)
                      CALL VTIME(TV,TC)
                      TimePrecond = TimePrecond + TC - TC1
                   ELSE
                      !CALL W1_COPY( ELEMENT(WOPT, NRES+N1,NP), WTMP(NP))
                      PRINT *, "WARNING! NOT TESTED ON GPU - there may be errors!"
                      CALL W1_COPY_WITHOUT_CW( ELEMENT(WOPT, NRES+N1,NP), WTMP(NP))
                      call cuda_memcpydtod(NULL_STREAM,WTMP_GPU(NP),WOPT_GPU(NP,NRES+N1),SIZE_OF_CW,int(c_sizeof(fakec),c_size_t))
                   ENDIF
                   CALL VTIME(TV1,TC1)
                   ! elements WOPT(NRES+N1: NRES+ITER, NP)
                   !CALL W1_GEMV( one, ELEMENTS( WOPT, NRES+N1, NRES+ITER, NP),  WTMP(NP), zero, CWORK1, 1)
                   CALL W1_GEMV_GPU( one, ELEMENTS( WOPT, NRES+N1, NRES+ITER, NP),  WTMP_GPU(NP), zero, CWORK1, 1, WOPT_GPU(NP,:), N1, ITER, NRES)

                   DO N2=N1,ITER
                      CHAM(N2,N1)=      GREAL(CWORK1(N2-N1+1))
                      CHAM(N1,N2)=GREAL(GCONJG(CWORK1(N2-N1+1)))
                   ENDDO

                   ! elements WOPT(N1: ITER, NP)
                   !CALL W1_GEMV( one, ELEMENTS( WOPT, N1, ITER, NP),  ELEMENT( WOPT, N1, NP), zero, CWORK1, 1, CQIJ)
                   CALL W1_GEMV_GPU( one, ELEMENTS( WOPT, N1, ITER, NP),  WOPT_GPU(NP,N1), zero, CWORK1, 1, WOPT_GPU(NP,:),N1,ITER, 0, CQIJ)

                   DO N2=N1,ITER
                      CTMP(N2,N1)=      GREAL(CWORK1(N2-N1+1))
                      CTMP(N1,N2)=GREAL(GCONJG(CWORK1(N2-N1+1)))
                   ENDDO
                   CALL VTIME(TV,TC)
                   TimeBlas = TimeBlas + TC - TC1
                ENDDO buildh

#ifndef gammareal
                DO N1=1,ITER
                   IF (ABS(AIMAG(CHAM(N1,N1)))>1E-2_q) THEN
                      WRITE(*,*)'WARNING: Sub-Space-Matrix is not hermitian in rmm', &
                           AIMAG(CHAM(N1,N1))
                   ENDIF
                   CHAM(N1,N1)= REAL( CHAM(N1,N1) ,KIND=q)
                ENDDO
#endif

                ! solve eigenvalue-problem and calculate lowest eigenvector
                ! this eigenvector corresponds to a minimal residuum
                ! CHAM(n1,n2) U(n2,1) = E(1) S(n1,n2)  U(n2,1)

                IF (.FALSE.) THEN
                   io_begin
                   NPL2=MIN(10,ITER)
                   WRITE(6,*)
                   DO N1=1,NPL2
                      WRITE(6,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
                   ENDDO
                   WRITE(6,*)
#ifndef gammareal
                   DO N1=1,NPL2
                      WRITE(6,3)N1,(AIMAG(CHAM(N1,N2)),N2=1,NPL2)
                   ENDDO
                   WRITE(6,*)
#endif
                   DO N1=1,NPL2
                      WRITE(6,1)N1,(REAL( CTMP(N1,N2) ,KIND=q) ,N2=1,NPL2)
                   ENDDO
                   WRITE(6,*)
#ifndef gammareal
                   DO N1=1,NPL2
                      WRITE(6,3)N1,(AIMAG(CTMP(N1,N2)),N2=1,NPL2)
                   ENDDO
                   WRITE(6,*)
#endif

1                  FORMAT(1I2,3X,20F9.5)
3                  FORMAT(1I2,3X,20E9.1)
                   io_end
                ENDIF
                !
                ! onion award of the year for IBM,
                ! who use a completely different DSYGV calling sequence
                !
                CALL VTIME(TV1,TC1)
#ifdef gammareal
#ifdef essl
                CALL DSYGV &
                     &  (1,CHAM,NRES,CTMP,NRES,R,CHAM,NRES, &
                     &           ITER,CWORK(1),LWORK*INFO%NDAV)
#else
                CALL DSYGV &
                     &  (1,'V','U',ITER,CHAM,NRES,CTMP,NRES,R, &
                     &           CWORK(1),LWORK*INFO%NDAV,IFAIL)
#endif
#else
                CALL ZHEGV &
                     &  (1,'V','U',ITER,CHAM,NRES,CTMP,NRES,R, &
                     &           CWORK(1),LWORK*INFO%NDAV,RWORK(1),IFAIL)
#endif
                CALL VTIME(TV,TC)
                TimeLapack = TimeLapack + TC - TC1
    
                CALL VTIME(TV1,TC1)   
                ! just to be sure merge results from all nodes
                CALLMPI( M_bcast_g(WDES%COMM_INB, CHAM, NRES*NRES))

                IF (IFAIL/=0) THEN
                   IF (IU6>=0) &
                        WRITE(IU6,219) IFAIL,ITER,N
                   IF (IU0>=0) &
                        WRITE(IU0,219) IFAIL,ITER,N
                   !  try to save things somehow, goto next band
                   W1(NP)%LDO=.FALSE.
                   call cublas_get_vector(SIZE_OF_CW, int(c_sizeof(fakec),c_int),&
                           WOPT_GPU(NP,ITER),1,c_loc(W1(NP)%CW),1)
                   CYCLE i2
                ENDIF

219             FORMAT('WARNING in EDDRMM: call to ZHEGV failed, returncode =',I4,1X,I2,1X,I2)
                FPRE(NP)=R(1)
                IF (IDUMP==2)  WRITE(*,'(E9.2,"P")',ADVANCE='NO') SQRT(ABS(FPRE(NP)))

                !     write out 'optimal trial step' i.e step which would have minimized
                !     the residuum
                IF (ITER==2 .AND. IDUMP==2) THEN
                   OTRIAL= REAL( 1+CHAM(1,1)/CHAM(2,1) ,KIND=q) *TRIAL(NP)
                   WRITE(*,'(1X,F7.4,"o")',ADVANCE='NO') OTRIAL
                ENDIF

                !     some heuristic for numerical accuracy problems
                !     small residuum and negative step -> stop immediately
                IF (ITER==2) THEN
                   OTRIAL= REAL( 1+CHAM(1,1)/CHAM(2,1) ,KIND=q) *TRIAL(NP)
                   IF (OTRIAL <0 .AND.  ABS(FPRE(NP))< 1E-9_q) THEN
#ifndef NO_NUMPROB_WARN
                      IF (IU0>=0) WRITE(IU0,'(" num prob ")',ADVANCE='NO')
#endif
                      PRINT *, "WARNING: NOT TESTED ON THE GPU - may produce errors!"
                      call cublas_get_vector(SIZE_OF_CW,int(c_sizeof(fakec),c_int),&
                                        WOPT_GPU(NP,ITER),1,c_loc(W1(NP)%CW),1)
                      W1(NP)%LDO=.FALSE.
                      CYCLE i2
                   ENDIF
                ENDIF

                IF (.FALSE.) THEN
                   io_begin
                   NPL2=MIN(10,ITER)

                   WRITE(77,*)
                   DO N1=1,NPL2
                      WRITE(77,1)N1,R(N1),(REAL( CHAM(N2,N1) ,KIND=q) ,N2=1,NPL2)
                   ENDDO
                   WRITE(77,*)
#ifndef gammareal
                   DO N1=1,NPL2
                      WRITE(77,3)N1,R(N1),(AIMAG(CHAM(N2,N1)),N2=1,NPL2)
                   ENDDO
                   WRITE(77,*)
#endif
                   io_end
                ENDIF
                CALL VTIME(TV,TC)   
                TimeInit = TimeInit + TC - TC1
             ELSE optsub
               PRINT *, "ERROR: IALGO=0 NOT SUPPORTED ON THE GPU!"
!***********************************************************************
! inverse interation step
! minimize
!    | ( H - e S) | phi > - | phi_ini > |^ 2  -> min
! in the yet available subspace spanned by the wavefunction stored in WOPT%CW
! if one denotes these wavefunctions as phi_j, and R_j=  (H - e S) phi_j
! the following equation is obtained:
!  sum_ij  b_i* < R_i | R_j > b_j - sum_i b_i*<R_i |phi_ini> + c.c. -> min
! or equivalently
!  sum_j  < R_i | R_j > b_j  = <R_i |phi_ini>
! the new optimized wavefunction is given by solving this linear
! equation for b
!
!***********************************************************************
                CHAM=0
                B =0

                !    A(n2,n1)=    < phi_n2 |  ( H - e S) ( H - e S)  | phi_n1 >
                builda: DO N1=1,ITER
                   CALL W1_GEMV( one, ELEMENTS( WOPT, NRES+N1, NRES+ITER, NP),  ELEMENT( WOPT, NRES+N1, NP), &
                        zero, CWORK1, 1)
                   DO N2=N1,ITER
                      CHAM(N2,N1)=      GREAL(CWORK1(N2-N1+1))
                      CHAM(N1,N2)=GREAL(GCONJG(CWORK1(N2-N1+1)))
                   ENDDO
                ENDDO builda

                ! B(n1) = < phi_n1 |  ( H - e S) | phi_ini >
                CALL W1_GEMV( one, ELEMENTS( WOPT, NRES+1, NRES+ITER, NP),   &
                                   ELEMENT( W_INI,  NP), zero, B(1), 1)

                CTMP=CHAM
                B_=B
                ! calculate the solution of sum_j CHAM(i,j) * X(j) = B(i)
                ! overwrite B by X
                CALL GGETRF( ITER, ITER, CHAM, NRES, IPIV, IFAIL )
                IF (IFAIL ==0) &
                     CALL GGETRS('N', ITER, 1, CHAM, NRES, IPIV, B, NRES, IFAIL)

                IF (.FALSE.) THEN
                   ! dump the matrix and the solution vector
                   io_begin
                   N2=MIN(10,ITER)
                   WRITE(6,*)
                   DO N1=1,N2
                      WRITE(6,'("m",I3,8E14.7)')N1, CTMP(N1,1:N2)
                   ENDDO
                   WRITE(6,'(A4,8E14.7)') 'b', B_(1:N2)

                   WRITE(6,*)
                   WRITE(6,'(A4,8E14.7)') 'e', B (1:N2)
                   io_end
                ENDIF

                IF (ITER == 1) B(1) = 1

                IF (IFAIL/=0) THEN
                   IF (IU6>=0) &
                        WRITE(IU6,219) IFAIL,ITER,N
                   IF (IU0>=0) &
                        WRITE(IU0,219) IFAIL,ITER,N
                   !  try to save things somehow, goto next band
                   W1(NP)%LDO=.FALSE.
                   CYCLE i2
                ENDIF

                IF (ITER==2 .AND. IDUMP==2) THEN
                   ! write out 'optimal trial step' i.e step which would have minimized
                   ! the residuum
                   OTRIAL= REAL( 1+B(1)/B(2) ,KIND=q) *TRIAL(NP)
                   WRITE(*,'(1X,F7.4,"o")',ADVANCE='NO') OTRIAL
                ENDIF

                CHAM(1:ITER,1)=B(1:ITER)


                IF (IDUMP >= 3) THEN
                   ! set CWORK1(1) to < xi | xi >
                   C=W1_DOT( ELEMENT(W_INI,NP) , ELEMENT(W_INI,NP))
                   DO N1=1,ITER
                      DO N2=1,ITER
                         C=C+GCONJG(B(N2))*CTMP(N2,N1)*B(N1)
                      ENDDO
                      C=C-B_(N1)*GCONJG(B(N1))-GCONJG(B_(N1))*B(N1)
                   ENDDO
                   ! residual after the step
                   WRITE(*,'(1X,E9.2,"rs")',ADVANCE='NO') SQRT(ABS(C))
                ENDIF

             ENDIF optsub
!=======================================================================
! now performe the trial step (ITER > 1 use previous trial step)
! but restrict trial step to 1.0
!=======================================================================
             IF (ABS(W%FERWE(N,NK,ISP))<INFO%WEIMIN .AND. &
                 &    ABS(W%FERWE(N,NK,ISP)*DE)<INFO%WEIMIN .AND. ITER >= 2) LABORT(NP)=.TRUE.
             IF (ABS(FPRE(NP))<DEIT(NP)) LABORT(NP)=.TRUE.
             IF (ITER == NITER) LABORT(NP)=.TRUE.
             ! eventually set LABORT to .FALSE. to recover old behaviour
             LABORT(NP)=.FALSE.

             ! make trial step positive
             IF (TRIAL(NP)<0) TRIAL(NP)=ABS(TRIAL(NP))
             ! WTMP(NP)=0
             ! CALL W1_DSCAL( WTMP(NP), 0.0_q)
             CALL VTIME(TV1,TC1)
             CALL cuda_memset(WTMP_GPU(NP),0,WTMP(NP)%WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
             CALL W1_DSCAL_CPROJ(WTMP(NP), 0.0_q)

             ! WTMP=WTMP + CHAM(I,1) *WOPT(I,NP)
             DO I=1,ITER
                !CALL W1_GAXPY( ELEMENT(WOPT, I,NP), CHAM(I,1), WTMP(NP))
                CALL W1_GAXPY_GPU( ELEMENT(WOPT, I,NP), WOPT_GPU(NP,I), CHAM(I,1), WTMP(NP), WTMP_GPU(NP))
             ENDDO
             CALL VTIME(TV,TC)
             TimeBlas = TimeBlas + TC - TC1

             IF (.NOT.LABORT(NP)) THEN
               ! trial step on wavefunction
                DO I=1,ITER
                   !CALL APPLY_PRECOND( ELEMENT(WOPT, NRES+I,NP), W1(NP), PRECON(1,NP))
                   !CALL W1_GAXPY( W1(NP), -CHAM(I,1)*TRIAL(NP), WTMP(NP))
                CALL VTIME(TV1,TC1)
                call cuda_applyprecond(NULL_STREAM,WDES1%NPL,WOPT_GPU(NP,NRES+I),GPU(NP)%CW1,GPU_PRECON,1.0_q,(NP-1)*WDES%NRPLWV)
                !CALL APPLY_PRECOND_GPU_NEW( ELEMENT(WOPT, NRES+I,NP), WOPT_GPU(NP,NRES+I), GPU(NP)%CW1, (NP-1)*WDES%NRPLWV)
                CALL VTIME(TV,TC)
                TimePrecond = TimePrecond + TC - TC1

                CALL VTIME(TV1,TC1)
                CALL W1_GAXPY_GPU( W1(NP), GPU(NP)%CW1, -CHAM(I,1)*TRIAL(NP), WTMP(NP), WTMP_GPU(NP))
                CALL VTIME(TV,TC)
                TimeBlas = TimeBlas + TC - TC1
                ENDDO
             ENDIF

             ! transform the wave-function to real space
             ! CALL FFTWAV_W1(WTMP(NP))
             
             CALL VTIME(TV1,TC1)
             ! transform the wave-function to real space
             CALL FFTWAV_W1_GPU_STREAM(NULL_STREAM,WTMP(NP),WTMP_GPU(NP),GPU(NP)%CR1,GPU(1)%NINDPW)
             CALL VTIME(TV,TC)   
             TimeFFTWAV = TimeFFTWAV + TC - TC1
             
             CALL VTIME(TV1,TC1)
             IF (.NOT. USE_PROJALL_GPU) THEN 
                 ! copy from device to host memory
                 call cuda_memcpydtoh(NULL_STREAM,c_loc(WTMP(NP)%CR),GPU(NP)%CR1,WDES1%GRID%MPLWV*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t))
             ENDIF
             CALL VTIME(TV,TC)   
             TimeTrans = TimeTrans + TC - TC1
             TimeTransPROJALL = TimeTransPROJALL + TC - TC1
!***********************************************************************
!
! minimize energy starting from current wavefunctions stored in
! W1 along the current searchdirection stored in WOPT(NRES+ITER,NP)
!
!***********************************************************************
          ELSE optsubspace
             IF (ITER==1) THEN
               DEIT(NP)=ABS(FPRE(NP))*INFO%DEPER
             ENDIF
             IF (ITER == NITER) LABORT(NP)=.TRUE.

             ! trial vector in line minimization
             !CALL APPLY_PRECOND( ELEMENT(WOPT, NRES+ITER,NP), WTMP(NP), PRECON(1,NP))
             !CALL FFTWAV_W1(WTMP(NP))
             call cuda_applyprecond(NULL_STREAM,WDES1%NPL,WOPT_GPU(NP,NRES+ITER),WTMP_GPU(NP),GPU_PRECON,1.0_q,(NP-1)*WDES%NRPLWV)
             CALL VTIME(TV1,TC1)
             !CALL APPLY_PRECOND_GPU_NEW( ELEMENT(WOPT, NRES+ITER,NP), WOPT_GPU(NP,NRES+ITER), WTMP_GPU(NP), (NP-1)*WDES%NRPLWV)
             !call cuda_print("w1good.dat",'d',WDES1%NRPLWV,WTMP_GPU(NP))

             CALL VTIME(TV,TC)
             TimePrecond = TimePrecond + TC - TC1
             
             CALL VTIME(TV1,TC1)
             ! transform the wave-function to real space
             CALL FFTWAV_W1_GPU_STREAM(NULL_STREAM,WTMP(NP),WTMP_GPU(NP),GPU(NP)%CR1,GPU(1)%NINDPW)
             CALL VTIME(TV,TC)   
             TimeFFTWAV = TimeFFTWAV + TC - TC1             
             
             CALL VTIME(TV1,TC1)

             IF (.NOT. USE_PROJALL_GPU) THEN 
                ! copy from device to host memory
                call cuda_memcpydtoh(NULL_STREAM,c_loc(WTMP(NP)%CR),GPU(NP)%CR1,WDES1%GRID%MPLWV*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t))
             ENDIF
             
             CALL VTIME(TV,TC)   
             TimeTrans = TimeTrans + TC - TC1
             TimeTransPROJALL = TimeTransPROJALL + TC - TC1
          ENDIF optsubspace
        SID=SID+1
        IF(SID>=STREAM_SIZE) SID=0
       ENDDO i2
       CALL THREADSYNCHRONIZE()

       ! calculate results of projection operatores
       CALL VTIME(TV1,TC1)
       IF (USE_PROJALL_GPU) THEN 
          CALL W1_PROJALL_GPU(WDES1, WTMP, NONLR_S, NONL_S, NSIM, GPU_W1_CR, TimeTransPROJALL)
          CALL THREADSYNCHRONIZE ()
       ELSE
          CALL W1_PROJALL(WDES1, WTMP, NONLR_S, NONL_S, NSIM)
       ENDIF  
       CALL VTIME(TV,TC)
       TimePROJALL= TimePROJALL + TC - TC1

        ! NVIDIA
        ! This part has been completly reordered to allow more simultanious work
        ! of both the GPU and the CPU. In practise it means that many of the
        ! function calls are batched together and launched in one go. This
        ! eliminates large amounts of launch latency and enables CPU
        ! computations while the GPU is working on other parts of the data.
        ! This can give as much as over 3x performance improvement of the
        ! original loop. This method is applicable to other parts of the code 
        ! where the algorithm loops over NSIM

      !Setup the arguments for the GPU functions, these will overlap
      !with CPU code
      NSIMActive        = 0
      DotArgumentsIndex = 0
      CDArgumentsIndex  = 0
      prei3: DO NP=1,NSIM
                N    = NB(NP)
                ITER = IT(NP)

          mine2: IF (.NOT. LDELAY .AND. ITER > 1) THEN
                  !Do nothing
          ELSE mine2
             SIZE_SECOND_DAXPY_PART = 0
             IF (ASSOCIATED(WTMP(NP)%CR) .AND. ASSOCIATED(W1(NP)%CR)) THEN
                IF (SIZE(WTMP(NP)%CR) /=WTMP(NP)%WDES1%GRID%MPLWV*WTMP(NP)%WDES1%NRSPINORS) THEN
                    WRITE(0,*) 'internal error in W1_COPY: real space allocation is not correct'
                    STOP
                ENDIF
                SIZE_SECOND_DAXPY_PART = WTMP(NP)%WDES1%GRID%MPLWV*WTMP(NP)%WDES1%NRSPINORS*2
             END IF    

             !Store the properties for the required daxpy calls on the GPU, we
             !first gather all arguments such that we can launch the whole batch
             !in one go
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+1) = WTMP(NP)%WDES1%NPL*2
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+2) = WTMP_GPU(NP)
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+3) = GPU(NP)%CW2
                
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+4) = SIZE_SECOND_DAXPY_PART
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+5) = GPU(NP)%CR1
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+6) = GPU(NP)%CR2
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+1) = -1.0_q
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+2) = -1.0_q
             CDArgumentsIndex = CDArgumentsIndex + 6

             IF (WTMP(NP)%WDES1%LGAMMA) THEN
                !call cuda_zdot(WDES1%NPL,WOPT_GPU(NP,NRES+ITER),WTMP_GPU(NP),A2_GPU)
                call cuda_ddot( &
                        2*WDES1%NPL, &
                        WOPT_GPU(NP,NRES+ITER), &
                        WTMP_GPU(NP), &
                        A2_GPU)
                !call gpu_ddot_nocopy( 2* WTMP(NP)%WDES1%NPL,WOPT_GPU(NP,NRES+ITER), &
                !                         WTMP_GPU(NP), A2_GPU, 0, NSIM+NP-1)
             ELSE
                 !zzdot_nocopy_2d arguments
                 DotArguments_CPU(DotArgumentsIndex+1) = WTMP(NP)%WDES1%NPL      ! Size
                 DotArguments_CPU(DotArgumentsIndex+2) = WOPT_GPU(NP, NRES+ITER) ! First vector pointer
                 DotArguments_CPU(DotArgumentsIndex+3) = WTMP_GPU(NP)            ! Second vector pointer
                 DotArguments_CPU(DotArgumentsIndex+4) = 2*(NP-1)+0              ! Index to store
                 DotArgumentsIndex                     = DotArgumentsIndex + 4
             ENDIF
             
             !CNORMA dot product
             DotArguments_CPU(DotArgumentsIndex+1) = W1(NP)%WDES1%NPL      ! Size
             DotArguments_CPU(DotArgumentsIndex+2) = GPU(NP)%CW2 ! First vector pointer
             DotArguments_CPU(DotArgumentsIndex+3) = GPU(NP)%CW2 ! Second vector pointer
             DotArguments_CPU(DotArgumentsIndex+4) = 2*(NP-1)+1              ! Index to store
             DotArgumentsIndex                     = DotArgumentsIndex + 4
             
             CALL ECCP_GPU_ONLY_V2(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),&
                                   W%CELEN(N,NK,ISP), GPU(NP)%CR2,GPU(NP)%CR2,GPU(1)%SV,& 
                                   IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU(NP)%CW2, & 
                                   GPU(NP)%CW2,GPU(1)%DATAKE, NSIM, NP, LOCAL_CONTRIB_RES_GPU)
              NSIMActive = NSIMActive + 1
        ENDIF mine2
       ENDDO prei3

       
       !Copy all arguments to the GPU
       call cublas_Set_Vector(NSIM*4*2,int(c_sizeof(fakep),c_int),c_loc(DotArguments_CPU),1,DotArguments_GPU,1)
       call cublas_Set_Vector(NSIM*3*2,int(c_sizeof(fakep),c_int),c_loc(CombinedDAXPYArguments_CPU),1,CombinedDAXPYArguments_GPU,1)
       call cublas_Set_Vector(NSIM*2,int(c_sizeof(fakec),c_int),c_loc(CombinedDAXPYArgumentsAlpha_CPU),1,CombinedDAXPYArgumentsAlpha_GPU,1)
       call local_contributionv2_arm(NSIM, LOCAL_CONTRIB_RES_GPU)


       !Call the functions for the if-else-mine step
    
       !The next executed call combines all calls to compute the dot product,
       !the local contribution and compute the daxpy for the vertices. Note the
       !order is important since they do affect each other
       call gpu_daxpy2_2d(CDArgumentsIndex, CombinedDAXPYArguments_CPU, &
                          CombinedDAXPYArguments_GPU, CombinedDAXPYArgumentsAlpha_GPU)
       CDArgumentsIndex = 0 ! Reset

       call gpu_zdot_nocopy_2d(2*NSIMActive, DotArguments_CPU, DotArguments_GPU, A2_GPU, 0)

       call local_contributionv2_launch(NSIM, LOCAL_CONTRIB_RES_GPU)

       prei3cpu: DO NP=1, NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE prei3cpu
          IF (.NOT. LDELAY .AND. ITER > 1) THEN
                  !Do nothing
          ELSE
             CALL W1_DAXPY_GPU_CPUONLY(0,WTMP(NP),-1.0_q,W1(NP),WTMP_GPU(NP),GPU(NP)%CR1,GPU(NP)%CW2,GPU(NP)%CR2)
             CALL ECCP_GPU_CPU_ONLY(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W%CELEN(N,NK,ISP),&
                                    GPU(NP)%CR2,GPU(NP)%CR2,GPU(1)%SV,IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU(NP)%CW2,&
                                    GPU(NP)%CW2,GPU(1)%DATAKE, LOCAL_CONTRIB_RES_CPU((NP-1)*2+1),&
                                    LOCAL_CONTRIB_RES_CPU((NP-1)*2+2), NON_LOCAL_CONTRIB_RES_CPU(NP))
             CALL CNORMA_GPU_CPUONLY(W1(NP),CQIJ, ISP, WSCAL_ARR(NP),GPU(NP)%CW2)
           ENDIF
       ENDDO prei3cpu
    
       !Copy the results from the GPU work, this is done after the CPU work to
       !allow overlap
       call cublas_get_vector(NSIM*2, int(c_sizeof(fakec),c_int), A2_GPU,1,c_loc(A2_CPU_RES),1)
       call cublas_get_vector(NSIM*2, int(c_sizeof(fakec),c_int), LOCAL_CONTRIB_RES_GPU,1,c_loc(LOCAL_CONTRIB_RES_CPU),1)
       
       NSIMActive = 0
       i3: DO NP=1,NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3
!***********************************************************************
!
! finish trial step of RMM-DIIS/ inverse iteration by copying WTMP to W1
!
!***********************************************************************
          mine: IF (.NOT. LDELAY .AND. ITER > 1) THEN
             !CALL W1_COPY(WTMP(NP), W1(NP))
             CALL VTIME(TV1,TC1)
             CALL W1_COPY_GPU(0, WTMP(NP), W1(NP),WTMP_GPU(NP),GPU(NP)%CW2,GPU(NP)%CR1,GPU(NP)%CR2)
             CALL VTIME(TV,TC)
             TimeBlas = TimeBlas + TC - TC1
!***********************************************************************
!
! finish line minimization of <phi| H | phi> along trial direction
!
!***********************************************************************
          ELSE mine
             ! < g | phi > 1. order energy change
             ! A Anciaux à checker
             !A2=W1_DOT( ELEMENT( WOPT, NRES+ ITER, NP), WTMP(NP))
             !A2=-A2*2
             !CALL W1_DAXPY(WTMP(NP), -1.0_q, W1(NP))
             CALL VTIME(TV1,TC1)
             IF (WTMP(NP)%WDES1%LGAMMA) THEN
               !This case is not handled by the optimized i3, which is compelx only
               call cublas_ddot(2*WDES1%NPL,WOPT_GPU(NP,NRES+ITER),1,WTMP_GPU(NP),1,A2)
               !call gpu_ddot_nocopy( 2* WTMP(NP)%WDES1%NPL,WOPT_GPU(NP,NRES+ITER),WTMP_GPU(NP), A2_GPU, 0)
               !call cublas_get_vector(1,int(c_sizeof(fakec),c_size_t), A2_GPU,1,A2,1)
             ENDIF
             CALL VTIME(TV,TC)
             TimeBlas = TimeBlas + TC - TC1

             IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                write(*,*) 'No gpu version for ECCP_TAU'
                stop
                !CALL ECCP_TAU(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),W%CELEN(N,NK,ISP))
             ELSE
                !CALL ECCP(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W%CELEN(N,NK,ISP))
            CALL ECCP_GPU_CPU_MPI(WDES1, GRID, LOCAL_CONTRIB_RES_CPU((NP-1)*2+1),&
                                   LOCAL_CONTRIB_RES_CPU((NP-1)*2+2),&
                                  NON_LOCAL_CONTRIB_RES_CPU(NP), W%CELEN(N,NK,ISP))
             ENDIF
             CALL VTIME(TV1,TC1)
             !CALL CNORMA(W1(NP),CQIJ, ISP, WSCAL)
             CNORMA_DOT = A2_CPU_RES((NP-1)*2+2) 
             WSCAL = WSCAL_ARR(NP)+CNORMA_DOT
             CALL CNORMA_GPU_CPU_MPI(W1(NP), WSCAL, WSCAL)

             W%CELEN(N,NK,ISP) =W%CELEN(N,NK,ISP)*WSCAL**2
             A1= W%CELEN(N,NK,ISP)-EVALUE(NP)
             ! quadratic interpolation to find the minimum
             IF (WTMP(NP)%WDES1%LGAMMA) THEN
               call cublas_get_vector(1,int(c_sizeof(fakec),c_int), A2_GPU,1,c_loc(A2),1)
             ELSE
               A2 = A2_CPU_RES((NP-1)*2+1) !Get the result from our already computed buffer
             ENDIF

             A2=-A2*2

             TRIAL(NP)= -A2/(A1-A2)/2
             DE       = (A2+(A1-A2)*TRIAL(NP))*TRIAL(NP)
             ! avoid too large trial steps
             IF (IDUMP>=2) WRITE(*,'(1X,F7.4,"T")',ADVANCE='NO') TRIAL(NP)
             IF (.NOT. LDELAY) THEN
                IF (TRIAL(NP)>0 .AND.(TRIAL(NP) > 1)) TRIAL(NP)= 1
                IF (TRIAL(NP)>0 .AND.(TRIAL(NP)<0.1)) TRIAL(NP)=0.1
!TODO is there any use to allow for negative trial steps 
!I think right now it kind of sucks (in all but the first step ABS(TRIAL(NP)) is used
! so probably does not matter what is used here
                IF (TRIAL(NP)<0 .AND.(TRIAL(NP) <-1)) TRIAL(NP)=-1
                IF (TRIAL(NP)<0 .AND.(TRIAL(NP)>-.1)) TRIAL(NP)=-0.1
             ENDIF

             IF (IDUMP>=2) WRITE(*,'(1X,F7.4,"T")',ADVANCE='NO') TRIAL(NP)
             CALL VTIME(TV,TC)   
             TimeInit = TimeInit + TC - TC1
             ! set W1 finally
             !CALL W1_DAXPY(WTMP(NP), -(TRIAL(NP)-1), W1(NP))
             CALL VTIME(TV1,TC1)      
             
             SIZE_SECOND_DAXPY_PART = 0
             IF (ASSOCIATED(WTMP(NP)%CR) .AND. ASSOCIATED(W1(NP)%CR)) THEN
                IF (SIZE(WTMP(NP)%CR) /=WTMP(NP)%WDES1%GRID%MPLWV*WTMP(NP)%WDES1%NRSPINORS) THEN
                    WRITE(0,*) 'internal error in W1_COPY: real space allocation is not correct'
                    STOP
                ENDIF
                SIZE_SECOND_DAXPY_PART = WTMP(NP)%WDES1%GRID%MPLWV*WTMP(NP)%WDES1%NRSPINORS*2
             END IF    

             
             !Store the properties for the required daxpy calls on the GPU, we
             !first gather all arguments such that we can launch the whole batch
             !in one go
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+1) = WTMP(NP)%WDES1%NPL*2
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+2) = WTMP_GPU(NP)
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+3) = GPU(NP)%CW2
                
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+4) = SIZE_SECOND_DAXPY_PART
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+5) = GPU(NP)%CR1
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+6) = GPU(NP)%CR2
             
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+1) = -(TRIAL(NP)-1)
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+2) = -(TRIAL(NP)-1)

             CDArgumentsIndex = CDArgumentsIndex + 6
             NSIMActive       = NSIMActive + 1
             
             !Launch the CPU part, for small DAXPY on CPROJ
             CALL W1_DAXPY_GPU_CPUONLY(0,WTMP(NP),-(TRIAL(NP)-1),W1(NP),WTMP_GPU(NP),&
                                  GPU(NP)%CR1,GPU(NP)%CW2,GPU(NP)%CR2)
             CALL VTIME(TV,TC)

             TimeBlas = TimeBlas + TC - TC1      
          ENDIF mine
      ENDDO i3

      !Setup the arguments for batched dot-product call
      DotArgumentsIndex = 0
      i3part2: DO NP=1, NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part2
             DotArguments_CPU(DotArgumentsIndex+1) = W1(NP)%WDES1%NPL      ! Size
             DotArguments_CPU(DotArgumentsIndex+2) = GPU(NP)%CW2 ! First vector pointer
             DotArguments_CPU(DotArgumentsIndex+3) = GPU(NP)%CW2 ! Second vector pointer
             DotArguments_CPU(DotArgumentsIndex+4) = (NP-1)             ! Index to store
             DotArgumentsIndex                     = DotArgumentsIndex + 4
      ENDDO i3part2

      call cublas_Set_Vector(NSIM*4,int(c_sizeof(fakep),c_int),c_loc(DotArguments_CPU),1,DotArguments_GPU,1) !At most NSIM arguments

      !Launch combined/batched daxpy
      if (CDArgumentsIndex > 0) Then
               call cublas_Set_Vector(NSIM*3*2,int(c_sizeof(fakep),c_int),c_loc(CombinedDAXPYArguments_CPU),1,&
                                       CombinedDAXPYArguments_GPU,1)

               call cublas_Set_Vector(NSIM*2,int(c_sizeof(fakec),c_int),c_loc(CombinedDAXPYArgumentsAlpha_CPU),1,&
                                       CombinedDAXPYArgumentsAlpha_GPU,1)

               call gpu_daxpy2_2d(CDArgumentsIndex, CombinedDAXPYArguments_CPU, &
                              CombinedDAXPYArguments_GPU, CombinedDAXPYArgumentsAlpha_GPU)
                CDArgumentsIndex = 0 ! Reset
      ENDIF
      
      call gpu_zdot_nocopy_2d(DotArgumentsIndex / 4, DotArguments_CPU, DotArguments_GPU, A2_GPU, 0) ! N-items based on number of arguments actually set

      i3part2cpu: DO NP=1,NSIM
             N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part2cpu
             CALL CNORMA_GPU_CPUONLY(W1(NP),CQIJ, ISP, WSCAL_ARR(NP),GPU(NP)%CW2)
      ENDDO i3part2cpu

      !Copy the results from the GPU work, done after CPU work. At most NSIM results.
      call cublas_get_vector(NSIM, int(c_sizeof(fakec),c_int),A2_GPU,1,c_loc(A2_CPU_RES),1) 

      i3part2mpi: DO NP=1, NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part2mpi

!=======================================================================
! common code
!=======================================================================
          CALL VTIME(TV1,TC1)       
          ! inverse of norm
          ! CALL CNORMA(W1(NP),CQIJ, ISP, WSCAL)
          CNORMA_DOT = A2_CPU_RES(NP) ! Get the GPU result from the  buffer
          WSCAL = WSCAL_ARR(NP)+CNORMA_DOT

          CALL CNORMA_GPU_CPU_MPI(W1(NP), WSCAL, WSCAL)
          WSCAL_ARR(NP) = WSCAL
        ENDDO i3part2mpi
        !Scale the data, reuse the argument arrays made for the DAXPY kernels
        NSIMActive       = 0
        CDArgumentsIndex = 0
        i3part3scal: DO NP=1, NSIM
             N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part3scal
             WSCAL = WSCAL_ARR(NP)
        
             SIZE_SECOND_DAXPY_PART = 0
             IF (ASSOCIATED(W1(NP)%CR)) THEN
               IF (SIZE(W1(NP)%CR) /=W1(NP)%WDES1%GRID%MPLWV*W1(NP)%WDES1%NRSPINORS) THEN
                  WRITE(0,*) 'internal error in W1_COPY: real space allocation is not correct'
                  STOP
               ENDIF
               ! real space wavefunction complex, SCALE real
               IF (GPU(NP)%CR2 /= 0) THEN
                       SIZE_SECOND_DAXPY_PART = W1(NP)%WDES1%GRID%MPLWV*W1(NP)%WDES1%NRSPINORS*2
               ENDIF
             END IF


             !Store the properties for the required daxpy calls on the GPU, we
             !first gather all arguments such that we can launch the whole batch
             !in one go
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+1) = W1(NP)%WDES1%NRPLWV*2
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+2) = GPU(NP)%CW2
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+3) = SIZE_SECOND_DAXPY_PART
             CombinedDAXPYArguments_CPU(CDArgumentsIndex+4) = GPU(NP)%CR2
             CDArgumentsIndex = CDArgumentsIndex + 4
             
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+1) = WSCAL
             CombinedDAXPYArgumentsAlpha_CPU(NSIMActive*2+2) = WSCAL
             NSIMActive       = NSIMActive + 1
             
             !Setup arguments for local-contribution
             CALL ECCP_GPU_ONLY_V2(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),&
                                   W%CELEN(N,NK,ISP), GPU(NP)%CR2,GPU(NP)%CR2,GPU(1)%SV,& 
                                   IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU(NP)%CW2, & 
                                   GPU(NP)%CW2,GPU(1)%DATAKE, NSIM, NP, LOCAL_CONTRIB_RES_GPU)
        ENDDO i3part3scal
        !Launch combined/batched daxpy and local contribution
        if (CDArgumentsIndex > 0) Then
               call cublas_Set_Vector(NSIM*2*2,int(c_sizeof(fakep),c_int),c_loc(CombinedDAXPYArguments_CPU),1,&
                                       CombinedDAXPYArguments_GPU,1)
               call cublas_Set_Vector(NSIM*2,int(c_sizeof(fakec),c_int),c_loc(CombinedDAXPYArgumentsAlpha_CPU),1,&
                                       CombinedDAXPYArgumentsAlpha_GPU,1)
               call local_contributionv2_arm(NSIM, LOCAL_CONTRIB_RES_GPU)

               call gpu_dscal_2d(CDArgumentsIndex, CombinedDAXPYArguments_CPU, &
                                 CombinedDAXPYArguments_GPU, CombinedDAXPYArgumentsAlpha_GPU)
        
               CDArgumentsIndex = 0 ! Reset
                
               call local_contributionv2_launch(NSIM, LOCAL_CONTRIB_RES_GPU)
        ENDIF

      i3part3cpu: DO NP=1,NSIM             
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part3cpu
          WSCAL = WSCAL_ARR(NP)
          ! scale W1
          !CALL W1_DSCAL(W1(NP), WSCAL)
          CALL W1_DSCAL_GPU_CPUONLY( W1(NP), WSCAL,GPU(NP)%CW2,GPU(NP)%CR2)
          CALL VTIME(TV,TC)
          TimeBlas = TimeBlas + TC - TC1

          IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
             write(*,*) 'No gpu version for ECCP_TAU'
             call cuda_device_reset()
             stop
          ELSE
          CALL ECCP_GPU_CPU_ONLY(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),&
                                 W%CELEN(N,NK,ISP),GPU(NP)%CR2,GPU(NP)%CR2,GPU(1)%SV,& 
                                 IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU(NP)%CW2,GPU(NP)%CW2,&
                                 GPU(1)%DATAKE, LOCAL_CONTRIB_RES_CPU((NP-1)*2+1),&
                                 LOCAL_CONTRIB_RES_CPU((NP-1)*2+2),NON_LOCAL_CONTRIB_RES_CPU(NP))
          !Can only execute copy after ECCP_GPU_CPU_ONLY is complete
          call cuda_memcpydtoh(1,c_loc(W1(NP)%CW),GPU(NP)%CW2,SIZE_OF_CW,int(c_sizeof(fakec),c_size_t))
          ENDIF
        enddo i3part3cpu
        call cublas_get_vector(NSIM*2, int(c_sizeof(fakec),c_int), LOCAL_CONTRIB_RES_GPU,1,c_loc(LOCAL_CONTRIB_RES_CPU),1)

       i3part3: DO NP=1,NSIM
          N=NB(NP); ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE i3part3
          CALL VTIME(TV1,TC1)
         IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
             write(*,*) 'No gpu version for ECCP_TAU'
             call cuda_device_reset()
             stop
          ELSE
          call ECCP_GPU_CPU_MPI(WDES1, GRID, LOCAL_CONTRIB_RES_CPU((NP-1)*2+1),&
                                LOCAL_CONTRIB_RES_CPU((NP-1)*2+2),&
                                NON_LOCAL_CONTRIB_RES_CPU(NP), W%CELEN(N,NK,ISP))
          ENDIF
          CALL VTIME(TV,TC)
          TimeECCP = TimeECCP + TC - TC1
          DECEL =W%CELEN(N,NK,ISP)-EVALUE(NP)
          DE    =DECEL

          IF (IDUMP==2) WRITE(*,'(E10.2,2H |)',ADVANCE='NO') DECEL

          DESUM =DESUM +WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*DECEL
          ICOUEV=ICOUEV+1
!=======================================================================
! break of intra-band-minimisation
! at the moment we performe a break of the intra-band minimization if
! ) DE is less then INFO%DEPER % of the change in the first minimization
!     of this band (relative breakcondition)
! ) DE less then INFO%EBREAK (absolut breakcondition)
! ) if unoccupied band break after 2. iteration
!=======================================================================
          DE=ABS(DE)

! TODO this here is tricky, the code also stops if the absolute
! change of the one-electron energy is smaller than EBREAK
! possibly this is not sensible if the residuum is minimized
! since we might hit a point where the energy changes little
! for many iterations
          IF (DE<INFO%EBREAK) W1(NP)%LDO=.FALSE.
! or break if LABORT has been set to .TRUE.
          IF (LABORT(NP)) W1(NP)%LDO=.FALSE.
! old conditions
          IF (ABS(W%FERWE(N,NK,ISP))<INFO%WEIMIN .AND. &
               &    ABS(W%FERWE(N,NK,ISP)*DE)<INFO%WEIMIN .AND. ITER >= 2) W1(NP)%LDO=.FALSE.
          IF (ABS(FPRE(NP))<DEIT(NP)) W1(NP)%LDO=.FALSE.
          IF (ITER==1) THEN
             DEIT(NP)=ABS(FPRE(NP))*INFO%DEPER
          ENDIF
          IF (ITER == NITER) W1(NP)%LDO=.FALSE.
          CALL VTIME(TV,TC)   
          TimeInit = TimeInit + TC - TC1
          !call cublas_free (GPU%CPROJ1)
          !call cublas_free (GPU%CPROJ2)

       ENDDO i3part3
       !Sync to make sure that all async d2h copies are complete
       call THREADSYNCHRONIZE()

       ! one band just finished ?, set NB(NP) also to 0 and finish everything
       DO NP=1,NSIM
          N=NB(NP)
          IF (.NOT. W1(NP)%LDO .AND. N /=0 ) THEN
             NB(NP)=0
             IF (IDUMP==2)  WRITE(*,'(F9.4,2H q)')REAL( W%CELEN(N,NK,ISP) ,KIND=q)
             IF (IDUMP==10) WRITE(*,*)
          ENDIF
       ENDDO
    ENDDO bands

    IF (INFO%LREAL) THEN
      call DESTROY_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
    ENDIF

    call cublas_free(GPU(1)%CR1_ALL)
    call cublas_free(GPU_CWORK_ALL)
    call cublas_free(GPU_PRECON)
    ENDDO kpoints
    ENDDO spin
!=======================================================================
    CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))
    CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))

    CALLMPI( M_sum_d(WDES%COMM_INTER, DESUM, 1))
    CALLMPI( M_sum_d(WDES%COMM_KINTER, DESUM, 1))

    CALLMPI( M_sum_i(WDES%COMM_INTER, ICOUEV ,1))
    CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV ,1))

    IF (W%OVER_BAND) THEN
       W%OVER_BAND=.FALSE.
       CALL REDIS_PW_DEALLOC(H_PW)
    ENDIF

    DO NP=1,NSIM
       CALL DELWAV_R(W1(NP))
       CALL DELWAV(WTMP(NP) ,.TRUE.)
    ENDDO
    CALL DELWAVA(W_INI)
    CALL DELWAVA(WOPT)
    DEALLOCATE(PRECON,CHAM,CTMP,CWORK1,B,IPIV,B_)
    call cublas_free(A2_GPU)


    !Free i3 variables
    call cublas_free(DotArguments_GPU) !8 is 64 byte = pointer
    call cublas_free(CombinedDAXPYArguments_GPU) !8 is 64 byte = pointer
    call cublas_free(CombinedDAXPYArgumentsAlpha_GPU)
    call cublas_alloc_safety (NSIM*2,int(c_sizeof(fakec),c_size_t),LOCAL_CONTRIB_RES_GPU)

    !call print_id_stream()
     CALL VTIME(TV,TC)
#ifdef DEBUG_AND_WATCH
     if (WDES1%COMM%NODE_ME == 1) then
       write(*,*) "       Funtion Name         ","      Time (s)        ","      percentage"
       write(*,*) "Time RMM-DIIS..............=",TC-TC0
       write(*,*) "Time PROJALL...............=",TimePROJALL,nint(TimePROJALL/(TC-TC0)*100)
       IF (USE_PROJALL_GPU) THEN !Arbitrary value
         write(*,*) "Time Transfer in PROJALL...=",TimeTransPROJALL
       ENDIF
       !write(*,*) "Time Precond...............=",TimePrecond,TimePrecond/(TC-TC0)*100
       write(*,*) "Time Init..................=",TimeInit,nint(TimeInit/(TC-TC0)*100)
       write(*,*) "Time Hamil.................=",TimeHamil,nint(TimeHamil/(TC-TC0)*100)
       write(*,*) "Time Transfer in Hamil.....=",TIMETRANSHAMIL
       write(*,*) "Time ECCP..................=",TimeECCP,nint(TimeECCP/(TC-TC0)*100)
       write(*,*) "Time FFTWAV................=",TimeFFTWAV,nint(TimeFFTWAV/(TC-TC0)*100)
       write(*,*) "Time Blas..................=",TimeBlas,nint(TimeBlas/(TC-TC0)*100)
       write(*,*) "Time Lapack................=",TimeLapack,nint(TimeLapack/(TC-TC0)*100)
       write(*,*) "Time Transfer..............=",TimeTrans,nint(TimeTrans/(TC-TC0)*100)
       write(*,*) "Total percent..............=",nint((TimeBlas+TimePROJALL+TimeHamil+TimeECCP+TimeFFTWAV+TimeTrans+TimeLapack+TimeInit)/(TC-TC0)*100)
       write(*,*) "######################################################################"
     endif
#endif

    RETURN
  END SUBROUTINE EDDRMM
END MODULE rmm_diis
