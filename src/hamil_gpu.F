#include "symbol.inc"


MODULE hamil_gpu
  USE hamil
  USE prec
  USE wave_high
CONTAINS

SUBROUTINE ECCP_GPU(WDES1,W1,W2,LMDIM,CDIJ,GRID,SV, CE,CR1_GPU,CR2_GPU,SV_GPU,shiftSV,CW1_GPU,CW2_GPU,DATAKE_GPU)
  
    USE prec
    USE mpimy
    USE mgrid
    USE cuda_interface
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    INTEGER(c_intptr_t) CR1_GPU,CR2_GPU
    INTEGER(c_intptr_t) CW1_GPU,CW2_GPU
    INTEGER(c_intptr_t)      SV_GPU
    TYPE (wavefun1) :: W1,W2
    TYPE (wavedes1) :: WDES1
    TYPE (grid_3d)  :: GRID
    INTEGER(c_intptr_t) :: DATAKE_GPU

    INTEGER doAssign, doNormalize, sizeData
    INTEGER NGVECTOR, ISPINOR, SV_IS_REAL,shiftSV
    GDEF      CNL
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES1%NRSPINORS*WDES1%NRSPINORS) ! local potential
    OVERLAP CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)

nv_profile_start(NVP_ECCP)
    
!=======================================================================
! calculate the local contribution
!=======================================================================
    CLOCAL=zero
    NGVECTOR=WDES1%NGVECTOR

    doAssign = 1
    doNormalize = 0

#ifdef realmode
    SV_IS_REAL = 1
#else
    SV_IS_REAL = 0
#endif

nv_profile_start(NVP_ECCP_COMPUTE)
nv_profile_start(NVP_CUSTOM)
    DO ISPINOR =0,WDES1%NRSPINORS-1
       DO ISPINOR_=0,WDES1%NRSPINORS-1
          !GPU
          call local_contribution(GRID%RL%NP,CR1_GPU,CR2_GPU,SV_GPU,shiftSV+IDX(1,1+ISPINOR_+2*ISPINOR,DIMREAL(GRID%MPLWV)),GRID%MPLWV,ISPINOR,ISPINOR_,SV_IS_REAL,CLOCAL,doAssign, doNormalize)
          doAssign = 0
       ENDDO
    ENDDO
    !CLOCAL=CLOCAL/GRID%NPLWV
    CLOCAL=0
    doNormalize = GRID%NPLWV ! Next call the value in memory will be normalized

!=======================================================================
! kinetic energy contribution
!=======================================================================
    CKIN=0
    DO ISPINOR=0,WDES1%NRSPINORS-1
       !GPU
       call local_contribution (NGVECTOR,CW1_GPU,CW2_GPU,DATAKE_GPU,IDX(1,1+ISPINOR,WDES1%NGDIM),NGVECTOR,ISPINOR,ISPINOR,1,CKIN,doAssign, doNormalize)
       doNormalize = 0 ! Disable normalization again
    ENDDO
nv_profile_stop(NVP_CUSTOM)
nv_profile_stop(NVP_ECCP_COMPUTE)

!=======================================================================
! non local contribution
!=======================================================================
nv_profile_start(NVP_ECCP_CPU)
    CNL =0
    NPRO=0
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
       DO ISPINOR_=0,WDES1%NRSPINORS-1

          NPRO =ISPINOR *(WDES1%NPRO/2)
          NPRO_=ISPINOR_*(WDES1%NPRO/2)

          NIS =1
          DO NT=1,WDES1%NTYP
             LMMAXC=WDES1%LMMAX(NT)
             IF (LMMAXC==0) GOTO 310

             DO NI=NIS,WDES1%NITYP(NT)+NIS-1
                CALL ECCP_NL(LMDIM,LMMAXC,CDIJ(1,1,NI,1+ISPINOR_+2*ISPINOR),W1%CPROJ(NPRO_+1),W2%CPROJ(NPRO+1),CNL)
                NPRO = LMMAXC+NPRO
                NPRO_= LMMAXC+NPRO_
             ENDDO

310          NIS = NIS+WDES1%NITYP(NT)
          ENDDO
       ENDDO
    ENDDO spinor
nv_profile_stop(NVP_ECCP_CPU)

    ! Get the combined results of the above calls, note that we take the 
    ! maximum dimension of the two calls
    sizeData = GRID%RL%NP
    IF (NGVECTOR > sizeData) sizeData = NGVECTOR
    CKIN = 0
nv_profile_start(NVP_ECCP_COMPUTE)
nv_profile_start(NVP_CUSTOM)
    call get_local_contribution_result(sizeData, CKIN)
nv_profile_stop(NVP_CUSTOM)
nv_profile_stop(NVP_ECCP_COMPUTE)

    CE=GREAL(CLOCAL+CKIN+CNL)
    CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))

nv_profile_stop(NVP_ECCP)
  END SUBROUTINE ECCP_GPU

! This function only sets the arguments for the local_contribution GPU
! functions, it performs no actual kernel launch
SUBROUTINE ECCP_GPU_ONLY_V2(WDES1,W1,W2,LMDIM,CDIJ,GRID,SV, CE,CR1_GPU,CR2_GPU,SV_GPU,shiftSV,CW1_GPU,CW2_GPU,DATAKE_GPU,NSIM,IDX, GPU_RES_BUFFER)
    USE prec
    USE mpimy
    USE mgrid
    USE cuda_interface
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    INTEGER(c_intptr_t) CR1_GPU,CR2_GPU
    INTEGER(c_intptr_t) CW1_GPU,CW2_GPU
    INTEGER(c_intptr_t)      SV_GPU
    INTEGER(c_intptr_t)      DATAKE_GPU
    INTEGER(c_intptr_t) GPU_RES_BUFFER
    TYPE (wavefun1) :: W1,W2
    TYPE (wavedes1) :: WDES1
    TYPE (grid_3d)  :: GRID

    INTEGER NSIM, IDX

    INTEGER NGVECTOR, ISPINOR, SV_IS_REAL,shiftSV
    GDEF      CNL
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES1%NRSPINORS*WDES1%NRSPINORS) ! local potential
    OVERLAP CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
    
!=======================================================================
! calculate the local contribution
!=======================================================================
    CLOCAL=zero
    NGVECTOR=WDES1%NGVECTOR

#ifdef realmode
    SV_IS_REAL = 1
#else
    SV_IS_REAL = 0
#endif

    DO ISPINOR =0,WDES1%NRSPINORS-1
       DO ISPINOR_=0,WDES1%NRSPINORS-1
          !GPU
          call local_contributionv2_setup (GRID%RL%NP,CR1_GPU,CR2_GPU,SV_GPU,shiftSV+IDX(1,1+ISPINOR_+2*ISPINOR,DIMREAL(GRID%MPLWV)),GRID%MPLWV,ISPINOR,ISPINOR_,SV_IS_REAL, (IDX-1)*2+0)
       ENDDO
    ENDDO

!=======================================================================
! kinetic energy contribution
!=======================================================================
    DO ISPINOR=0,WDES1%NRSPINORS-1
       !GPU
       call local_contributionv2_setup (NGVECTOR,CW1_GPU,CW2_GPU,DATAKE_GPU,IDX(1,1+ISPINOR,WDES1%NGDIM),NGVECTOR,ISPINOR,ISPINOR,1,(IDX-1)*2+1)
    ENDDO
END SUBROUTINE ECCP_GPU_ONLY_V2


!Perform the global MPI sum on the per process computed data
SUBROUTINE ECCP_GPU_CPU_MPI(WDES1, GRID, CLOCAL, CKIN, CNL, CE)
    COMPLEX(q) CKIN
    COMPLEX(q) CLOCAL
    COMPLEX(q) CE
    GDEF      CNL
    TYPE (wavedes1) :: WDES1
    TYPE (grid_3d)  :: GRID
    ! Normalize CLOCAL and combine all sub values
    CLOCAL=CLOCAL/GRID%NPLWV

    CE=GREAL(CLOCAL+CKIN+CNL)

    CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))
END SUBROUTINE ECCP_GPU_CPU_MPI

! This one only performs the local CPU work. So performs no actions on data
! which resides on the GPU. That data is processed using the local_contribution
! calls
SUBROUTINE ECCP_GPU_CPU_ONLY(WDES1,W1,W2,LMDIM,CDIJ,GRID,SV, CE,CR1_GPU,CR2_GPU,SV_GPU,shiftSV,CW1_GPU,CW2_GPU,DATAKE_GPU, CLOCAL, CKIN, CNL_ARR)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    COMPLEX(q) CKIN
    COMPLEX(q) CNL_ARR
    COMPLEX(q) CLOCAL
    INTEGER(c_intptr_t) CR1_GPU,CR2_GPU
    INTEGER(c_intptr_t) CW1_GPU,CW2_GPU
    INTEGER(c_intptr_t)      SV_GPU
    INTEGER(c_intptr_t)      DATAKE_GPU
    TYPE (wavefun1) :: W1,W2
    TYPE (wavedes1) :: WDES1
    TYPE (grid_3d)  :: GRID

    INTEGER doAssign, doNormalize, sizeData
    INTEGER NGVECTOR, ISPINOR, SV_IS_REAL,shiftSV
    GDEF      CNL
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES1%NRSPINORS*WDES1%NRSPINORS) ! local potential
    OVERLAP CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
    
!=======================================================================
! calculate the local contribution
!=======================================================================
    NGVECTOR=WDES1%NGVECTOR


#ifdef realmode
    SV_IS_REAL = 1
#else
    SV_IS_REAL = 0
#endif

!=======================================================================
! non local contribution
!=======================================================================
#ifdef TAU_PROF
    call start_nonlocal_range()
#endif
    CNL =0
    NPRO=0

    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
       DO ISPINOR_=0,WDES1%NRSPINORS-1

          NPRO =ISPINOR *(WDES1%NPRO/2)
          NPRO_=ISPINOR_*(WDES1%NPRO/2)

          NIS =1
          DO NT=1,WDES1%NTYP
             LMMAXC=WDES1%LMMAX(NT)
             IF (LMMAXC==0) GOTO 310
#ifdef TAU_PROF
             call start_eccpnl_range()
#endif
             DO NI=NIS,WDES1%NITYP(NT)+NIS-1
                CALL ECCP_NL(LMDIM,LMMAXC,CDIJ(1,1,NI,1+ISPINOR_+2*ISPINOR),W1%CPROJ(NPRO_+1),W2%CPROJ(NPRO+1),CNL)
                NPRO = LMMAXC+NPRO
                NPRO_= LMMAXC+NPRO_
             ENDDO
#ifdef TAU_PROF
             call end_range()
#endif
310          NIS = NIS+WDES1%NITYP(NT)
          ENDDO
       ENDDO
    ENDDO spinor

    CNL_ARR = CNL

    !Normalization and MPI moved to the ECCP_GPU_CPU_MPI function
    ! Normalize CLOCAL and combine all sub values
!    CLOCAL=CLOCAL/GRID%NPLWV
!    CE=GREAL(CLOCAL+CKIN+CNL)
!    CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))
#ifdef TAU_PROF
    call end_range()
#endif
  END SUBROUTINE ECCP_GPU_CPU_ONLY


  SUBROUTINE SETUP_PRECOND_GPU(WDES1, NSIM, IALGO, EVALUE, SLOCAL, DE_ATT, GPU_PTRS, GPU_DATAKE, GPU_PRECON)
    USE cuda_interface
    IMPLICIT NONE
    ! input arguments
    TYPE (wavedes1) WDES1       ! wavefunction descriptor
    INTEGER NSIM                !
    INTEGER IALGO               ! chosen algorithm
    REAL(q) :: EVALUE(NSIM)     ! eigenvalue
    REAL(q) :: SLOCAL           ! average local potential
    REAL(q) :: DE_ATT           ! complex shift
    INTEGER(c_intptr_t) GPU_PTRS(NSIM)   ! array of device pointers
    INTEGER(c_intptr_t) :: GPU_DATAKE       ! device pointer
    INTEGER(c_intptr_t) GPU_PRECON       ! device pointer
    ! local
    REAL(q) :: EKIN(NSIM)
    INTEGER NP

    ! calculate EKIN if IALGO=0,8,6
    call cuda_initprecond( &
        NSIM, &
        WDES1%NGVECTOR*WDES1%NRSPINORS, &
        GPU_PTRS, &
        GPU_DATAKE, &
        EKIN, &
        WDES1%NGDIM, &
        WDES1%NGVECTOR, &
        IALGO)
    call threadsynchronize()

    ! mpi sum of EKIN
    IF (IALGO==0 .OR. IALGO==8 .OR. IALGO==6) THEN
        DO NP=1,NSIM
        CALLMPI( M_sum_d(WDES1%COMM_INB, EKIN(NP), 1))
        ENDDO
    ENDIF

    ! calculate preconditioner
    call cuda_calcprecond( &
        NSIM, &
        WDES1%NGVECTOR*WDES1%NRSPINORS, &
        GPU_PTRS, &
        GPU_DATAKE, &
        GPU_PRECON, &
        EKIN, &
        EVALUE, &
        SLOCAL, &
        DE_ATT, &
        WDES1%NGDIM, &
        WDES1%NGVECTOR, &
        WDES1%NRPLWV, &
        IALGO)
    call threadsynchronize()
  END SUBROUTINE SETUP_PRECOND_GPU


  ! This one is used by davidson
  SUBROUTINE HAMILTMU_GPU( &
       &    WDES1, W1, NONLR_S, NONL_S, EVALUE, &
       &    CDIJ, CQIJ, SV, ISP, WRESULT, TIMETRANS, GPU, &
       &    GPU_CW, GPU_CR, GPU_CWORK, GPU_CH)
    USE iso_c_binding
    USE cuda_interface
    USE mpimy
    USE mgrid
    USE gpu_data
    USE nonl_high
    IMPLICIT NONE

    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(:)
    TYPE (nonlr_struct)NONLR_S
    TYPE (nonl_struct) NONL_S    
    REAL(q)    EVALUE(:)               ! eigenvalues
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    RGRID      SV(:,:)
    INTEGER    ISP
    REAL(q) TIMETRANS,TV,TC,TV1,TC1
    TYPE(wavefuna)     WRESULT
    TYPE(gpu_type) GPU(SIZE(W1))
    INTEGER(c_intptr_t) :: GPU_CW(:)
    INTEGER(c_intptr_t) :: GPU_CR(:)
    INTEGER(c_intptr_t) :: GPU_CWORK(:)
    INTEGER(c_intptr_t) :: GPU_CH(:)
    ! local variables
    COMPLEX(q) SIZE_OF_COMPLEX,fakec
    INTEGER NP, NX, NY, NZ,i
    INTEGER(c_intptr_t) GPU_NINDPW
    RGRID      GPU_SV
    INTEGER,PARAMETER :: STREAM_SIZE=4
    INTEGER           :: SID

nv_profile_start(NVP_HAMILTMU)

    NX=WDES1%GRID%NGPTAR(1)
    NY=WDES1%GRID%NGPTAR(2)
    NZ=WDES1%GRID%NGPTAR(3)

! calculate the local contribution
nv_profile_start(NVP_HAMILTMU_VHAMIL)
    SID=0
    DO NP=1,SIZE(W1)
        IF ( W1(NP)%LDO ) THEN
            CALL VHAMIL_GPU(SID,WDES1,WDES1%GRID,GPU(1)%SV,ISP,GPU_CR(NP),GPU_CWORK(NP))
            SID=SID+1
            IF(SID>=STREAM_SIZE) SID=0
        ENDIF
    ENDDO
    call threadsynchronize()
nv_profile_stop(NVP_HAMILTMU_VHAMIL)

    IF (NONLR_S%LREAL) THEN
! calculate the non-local contribution in real-space
nv_profile_start(NVP_HAMILTMU_RACCMU)
       CALL RACCMU_GPU(NONLR_S,WDES1,W1,CDIJ,CQIJ,ISP,EVALUE,GPU_CWORK(1),TIMETRANS)
       call threadsynchronize()
nv_profile_stop(NVP_HAMILTMU_RACCMU)

nv_profile_start(NVP_HAMILTMU_KINHAMIL)
nv_profile_start(NVP_FFT)
       SID=0
       DO NP=1,SIZE(W1)
           IF ( W1(NP)%LDO ) THEN
               CALL KINHAMIL_GPU(SID,WDES1,WDES1%GRID,GPU_CWORK(NP),.FALSE.,GPU(1)%DATAKE,EVALUE(NP),GPU_CW(NP),GPU_CH(NP),GPU(1)%NINDPW)
               SID=SID+1
               IF(SID>=STREAM_SIZE) SID=0
           ENDIF
       ENDDO
       call threadsynchronize()
nv_profile_stop(NVP_FFT)
nv_profile_stop(NVP_HAMILTMU_KINHAMIL)
    ELSE
! calculate the non local contribution in reciprocal space
nv_profile_start(NVP_HAMILTMU_KINHAMIL)
        DO NP=1,SIZE(W1)
            IF ( W1(NP)%LDO ) THEN
                CALL VNLACC(NONL_S, W1(NP), CDIJ, CQIJ, ISP, EVALUE(NP), WRESULT%CW(:,NP))
                call cublas_Set_Vector(WDES1%NGVECTOR*WDES1%NRSPINORS,int(c_sizeof(SIZE_OF_COMPLEX),c_int),c_loc(WRESULT%CW(1,NP)),1,GPU_CH(NP),1)

nv_profile_start(NVP_FFT)
                CALL KINHAMIL_GPU(-1,WDES1,WDES1%GRID,GPU_CWORK(NP),.TRUE.,GPU(1)%DATAKE,EVALUE(NP),GPU_CW(NP),GPU_CH(NP),GPU(1)%NINDPW)
                call threadsynchronize()
nv_profile_stop(NVP_FFT)
            ENDIF
        ENDDO
nv_profile_stop(NVP_HAMILTMU_KINHAMIL)
    ENDIF

nv_profile_stop(NVP_HAMILTMU)

    RETURN
  END SUBROUTINE HAMILTMU_GPU


  ! This one is used by rmm-diis
  SUBROUTINE HAMILTMU_GPU_NEW( &
       &    WDES1, W1, NONLR_S, NONL_S, EVALUE, &
       &    CDIJ, CQIJ, SV, ISP, WRESULT,TIMETRANS,GPU, &
       &    GPU_CWORK1, GPU_CH, WOPT_GPU, IT, NRES)
    USE iso_c_binding
    USE cuda_interface
    USE mpimy
    USE mgrid
    USE gpu_data
    USE nonl_high
    IMPLICIT NONE

    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(:)
    TYPE (nonlr_struct)NONLR_S
    TYPE (nonl_struct) NONL_S    
    REAL(q)    EVALUE(:)               ! eigenvalues
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    RGRID      SV(:,:)
    INTEGER    ISP
    REAL(q) TIMETRANS,TV,TC,TV1,TC1
    TYPE(wavefuna)     WRESULT
    TYPE(gpu_type) GPU(SIZE(W1))
    
    INTEGER(c_intptr_t) :: GPU_CH(SIZE(W1))
    INTEGER(c_intptr_t) :: WOPT_GPU(:,:)
    REAL(q) :: IT(SIZE(W1))
    INTEGER :: NRES
    ! local variables
    INTEGER(c_intptr_t) :: GPU_CWORK1(SIZE(W1))
!    INTEGER(c_intptr_t) :: GPU_CWORK1_ALL
    COMPLEX(q) SIZE_OF_COMPLEX
    !Real(q) :: GPU_DATAKE
    INTEGER NP, NX, NY, NZ,i
    INTEGER(c_intptr_t) GPU_NINDPW
    RGRID      GPU_SV
! DDNVIDIA local variables
    INTEGER,PARAMETER :: STREAM_SIZE=4
    INTEGER           :: SID

    NX=WDES1%GRID%NGPTAR(1)
    NY=WDES1%GRID%NGPTAR(2)
    NZ=WDES1%GRID%NGPTAR(3)
    
! allocate device memory
!    call cublas_Alloc_safety (WDES1%GRID%MPLWV*WDES1%NRSPINORS*SIZE(W1),int(c_sizeof(SIZE_OF_COMPLEX),c_size_t),GPU_CWORK1_ALL)
!    DO NP=1,SIZE(W1)
!      GPU_CWORK1(NP) = GPU_CWORK1_ALL + (WDES1%GRID%MPLWV*WDES1%NRSPINORS) * (NP-1) * int(c_sizeof(SIZE_OF_COMPLEX),c_size_t)
!    ENDDO
    
! calculate the local contribution (result in GPU_CWORK1)
    SID=0
    DO NP=1,SIZE(W1)
        IF ( W1(NP)%LDO ) THEN
            !CALL VHAMIL_GPU_OLD(SID,WDES1,WDES1%GRID,GPU(1)%SV,IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU(NP)%CR2,GPU_CWORK1(NP))
            CALL VHAMIL_GPU(SID,WDES1,WDES1%GRID,GPU(1)%SV,ISP,GPU(NP)%CR2,GPU_CWORK1(NP))
            SID=SID+1
            IF(SID>=STREAM_SIZE) SID=0
        ENDIF
    ENDDO
    CALL THREADSYNCHRONIZE()

! calculate the non-local contribution in real-space (result in WRESULT%CW)
    IF (NONLR_S%LREAL) THEN
       CALL RACCMU_GPU(NONLR_S, WDES1, W1, CDIJ, CQIJ, ISP, EVALUE,GPU_CWORK1(1),TIMETRANS)
!       CALL THREADSYNCHRONIZE ()

! DDNVIDIA FFT CUDA streams implementation
       SID=0
       DO NP=1,SIZE(W1)
          IF ( W1(NP)%LDO ) THEN
             CALL KINHAMIL_GPU(SID,WDES1,WDES1%GRID,GPU_CWORK1(NP),.FALSE.,GPU(1)%DATAKE,EVALUE(NP),WOPT_GPU(NP,IT(NP)),GPU_CH(NP),GPU(1)%NINDPW)
             SID=SID+1
             IF(SID>=STREAM_SIZE) SID=0
          ENDIF
       ENDDO
       !CALL THREADSYNCHRONIZE()

! calculate the non local contribution in reciprocal space (result in WRESULT%CW)   
       ELSE
       DO NP=1,SIZE(W1)
          IF ( W1(NP)%LDO ) THEN
             CALL VNLACC(NONL_S, W1(NP), CDIJ, CQIJ, ISP, EVALUE(NP),  WRESULT%CW(:,NP))
             call cublas_Set_Vector(WDES1%NGVECTOR,int(c_sizeof(SIZE_OF_COMPLEX),c_int),c_loc(WRESULT%CW(1,NP)),1,GPU_CH(NP),1)
!JBNV, should this not be: WOPT_GPU(NP,IT(NP)) ?
             CALL KINHAMIL_GPU(-1,WDES1,WDES1%GRID, GPU_CWORK1(NP), .TRUE., GPU(1)%DATAKE, EVALUE(NP), WOPT_GPU(IT(NP),NP), GPU_CH(NP), GPU(1)%NINDPW)
             !CALL THREADSYNCHRONIZE ()
          ENDIF
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE HAMILTMU_GPU_NEW

  SUBROUTINE HAMILT_LOCAL_GPU(SID, W1, GPU_SV, ISP, LADD, LKIN,GPU_DATAKE,GPU_CW, GPU_CH, GPU_CR, GPU_CVR, GPU_NINDPW)
    USE mgrid
    USE cuda_interface
    IMPLICIT NONE

    TYPE (wavefun1)    W1
    INTEGER(c_intptr_t)   GPU_SV
    INTEGER :: ISP
    INTEGER(c_intptr_t) :: GPU_NINDPW
    INTEGER(c_intptr_t) :: GPU_CR,GPU_CW
    INTEGER(c_intptr_t) GPU_CH
    INTEGER(c_intptr_t)    GPU_DATAKE
    LOGICAL LADD
    LOGICAL, OPTIONAL :: LKIN
    ! local variables
    INTEGER(c_intptr_t) :: GPU_CVR
    COMPLEX(q) :: fake
    INTEGER :: SID, SID2

    ! memory for GPU_CVR is allocated in subrot.F
    !call cuda_alloc_cvr(SID,W1%WDES1%GRID%MPLWV*W1%WDES1%NRSPINORS,GPU_CVR)

    !CALL VHAMIL_GPU_OLD(SID,W1%WDES1,W1%WDES1%GRID,GPU_SV,IDX(1,ISP,DIMREAL(W1%WDES1%GRID%MPLWV)),GPU_CR,GPU_CVR)
    CALL VHAMIL_GPU(SID,W1%WDES1,W1%WDES1%GRID,GPU_SV,ISP,GPU_CR,GPU_CVR)
    call threadsynchronize()

    ! Use SID + 1000 to make sure that the FFTs are in the SAME stream 
    ! as memory copies and kernels launched in the calling function
    IF (.NOT. PRESENT(LKIN)) THEN
       CALL KINHAMIL_GPU(SID, W1%WDES1, W1%WDES1%GRID,GPU_CVR,LADD, GPU_DATAKE,0.0_q,GPU_CW, GPU_CH, GPU_NINDPW)
    ELSE
       IF (LKIN) THEN
          CALL KINHAMIL_GPU(SID, W1%WDES1, W1%WDES1%GRID,GPU_CVR,LADD, GPU_DATAKE,0.0_q,GPU_CW, GPU_CH, GPU_NINDPW)
       ELSE
          CALL FFTHAMIL_GPU(SID,W1%WDES1,W1%WDES1%GRID,GPU_CVR,LADD,0.0_q,GPU_CW,GPU_CH, GPU_NINDPW)
       ENDIF
    ENDIF
  END SUBROUTINE HAMILT_LOCAL_GPU

END MODULE hamil_gpu

   SUBROUTINE VHAMIL_GPU(SID,WDES1,GRID,GPU_SV,ISP,GPU_CR,GPU_CVR)
       USE prec
       USE mgrid
       USE wave
       USE cuda_interface
       IMPLICIT NONE

       ! input parameters
       TYPE (grid_3d)     GRID
       TYPE (wavedes1)    WDES1
       INTEGER(c_intptr_t)         GPU_SV
       INTEGER            ISP
       INTEGER(c_intptr_t)         GPU_CR, GPU_CVR
       ! local variables
       INTEGER SID, strideSV, SV_IS_REAL
       REAL(q) RINPLW


#ifdef realmode
       SV_IS_REAL = 1
#else
       SV_IS_REAL = 0
#endif

       RINPLW=1._q/GRID%NPLWV
       strideSV=IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV))

       ! wrapper to cuda kernel
       call cuda_vhamil( &
                SID, &
                WDES1%NRSPINORS, &
                WDES1%GRID%RL%NP, &
                WDES1%GRID%MPLWV, &
                GPU_CVR, &
                GPU_SV, &
                strideSV, &
                GPU_CR, &
                RINPLW, &
                SV_IS_REAL)


   END SUBROUTINE VHAMIL_GPU

   ! DDNVIDIA KINHAMIL CUDA streams implementation
   SUBROUTINE KINHAMIL_GPU(SID, WDES1, GRID, GPU_CVR,  LADD, GPU_DATAKE, EVALUE, GPU_CW, GPU_CH, GPU_NINDPW)
     USE iso_c_binding
     USE cuda_interface
     USE prec
     USE mgrid
     USE wave
     IMPLICIT NONE

     TYPE (wavedes1) WDES1
     TYPE (grid_3d)  GRID
     INTEGER(c_intptr_t) :: GPU_CVR
     LOGICAL    :: LADD          ! if .TRUE. add results to CH
     INTEGER(c_intptr_t) :: GPU_DATAKE
     REAL(q)    :: EVALUE        ! subtract EVALUE*wavefunction
     INTEGER(c_intptr_t) :: GPU_CW
     INTEGER(c_intptr_t) :: GPU_CH
     INTEGER(c_intptr_t) :: GPU_NINDPW
     INTEGER    :: SID           ! CUDA stream index
   ! local
     INTEGER ISPINOR, M, MM, SHIFT1, SHIFT2
     COMPLEX(q) fakec
     REAL(q)    faker

     DO ISPINOR=0,WDES1%NRSPINORS-1
        CALL FFTEXT_GPU(SID,WDES1%NGVECTOR,GPU_NINDPW,GPU_CVR,ISPINOR*WDES1%GRID%MPLWV,GPU_CH,ISPINOR*WDES1%NGVECTOR,GRID,LADD)
        SHIFT1 = ISPINOR*WDES1%NGVECTOR
        SHIFT2 = ISPINOR*WDES1%NGDIM
        call cuda_kinhamil(SID,WDES1%NGVECTOR,GPU_CH+SHIFT1*int(c_sizeof(fakec),c_size_t),GPU_CW+SHIFT1*int(c_sizeof(fakec),c_size_t),GPU_DATAKE+SHIFT2*int(c_sizeof(faker),c_size_t),EVALUE)
        !CALL GPU_CHCWED(SID,WDES1%NGVECTOR,GPU_CH,SHIFT,GPU_CW,SHIFT,GPU_DATAKE,IDX(1,ISPINOR+1,WDES1%NGDIM),EVALUE)
     ENDDO
   END SUBROUTINE KINHAMIL_GPU


   SUBROUTINE FFTHAMIL_GPU(SID, WDES1, GRID, GPU_CVR, LADD, EVALUE, GPU_CW, GPU_CH, GPU_NINDPW)
     USE iso_c_binding
     USE cuda_interface
     USE prec
     USE mgrid
     USE wave
     IMPLICIT NONE

     ! inputs
     INTEGER            SID
     TYPE (wavedes1)    WDES1
     TYPE (grid_3d)     GRID
     COMPLEX(q) ::      GPU_CVR                 ! usually potential times wavefunction
     LOGICAL    ::      LADD                    ! if .TRUE. add results to CH
     INTEGER(c_intptr_t) ::      GPU_DATAKE
     REAL(q)    ::      EVALUE                  ! subtract EVALUE*wavefunction
     INTEGER(c_intptr_t) ::      GPU_CW                  ! wavefunction
     INTEGER(c_intptr_t) ::      GPU_CH                  ! result
     INTEGER(c_intptr_t) ::      GPU_NINDPW
     ! locals
     INTEGER            ISPINOR,M,MM,SHIFT
     COMPLEX(q)         fakec

     DO ISPINOR=0,WDES1%NRSPINORS-1
        CALL FFTEXT_GPU(SID,WDES1%NGVECTOR,GPU_NINDPW,GPU_CVR,ISPINOR*WDES1%GRID%MPLWV,GPU_CH,ISPINOR*WDES1%NGVECTOR,GRID,LADD)
        SHIFT = ISPINOR*WDES1%NGVECTOR*int(c_sizeof(fakec),c_size_t)
        call cuda_ffthamil(NULL_STREAM,WDES1%NGVECTOR,GPU_CH+SHIFT,GPU_CW+SHIFT,EVALUE)
        !CALL GPU_CHCW(WDES1%NGVECTOR,GPU_CH,SHIFT,GPU_CW,SHIFT,EVALUE)
     ENDDO
   END SUBROUTINE FFTHAMIL_GPU
