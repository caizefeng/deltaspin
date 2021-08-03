#include "symbol.inc"


MODULE charge_gpu
  USE prec
  
CONTAINS

  SUBROUTINE SOFT_CHARGE_GPU(GRID,GRID_SOFT,W,WDES, CHDEN)
    USE iso_c_binding
    USE cuda_interface
    USE wave_high
    USE gridq
    USE hamil
    USE c2f_interface, ONLY : VTIME
    IMPLICIT COMPLEX(q) (C)

    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID,GRID_SOFT
    COMPLEX(q)   CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
  ! local
    TYPE (wavedes1)     WDES1
    TYPE (wavefun1)     W1
    INTEGER ISPINOR
    INTEGER, PARAMETER :: NSTRIPD=2
    TYPE (REDIS_PW_CTR),POINTER :: H_PW
    TYPE (GRIDQUANT) :: CHARGE_REAL_SPACE, CHARGE
    INTEGER I
    !GPU
    INTEGER(c_intptr_t)   GPU_CHARGE!(NDIM,WDES1%NRSPINORS*WDES1%NRSPINORS)
    INTEGER(c_intptr_t) :: GPU_CW, GPU_CR, GPU_NINDPW
    REAL(q) TV,TV0,TC,TC0
    COMPLEX(q)  fakec
    REAL(q)     faker
    INTEGER CHARGE_IS_REAL

#ifdef realmode
     CHARGE_IS_REAL=1
#else
     CHARGE_IS_REAL=0
#endif

    IF (W%OVER_BAND) THEN
#ifdef MPI
       NCPU=WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
       NCPU=1
#endif
       NSTRIP=MIN(NSTRIPD,WDES%NBANDS)
       CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PW)
    ENDIF

    CALL SETWDES(WDES, WDES1, 0)
    CALL NEWWAV_R(W1, WDES1)

    CALL GENERATE_GRID_QUANTITY(CHARGE, GRID_SOFT, CHDEN) 
    CALL ALLOCATE_GRID_QUANTITY_FORCE_RL(CHARGE_REAL_SPACE, GRID, GRID_SOFT, WDES%NCDIJ)

    CHARGE_REAL_SPACE=0.0_q
    CHARGE_REAL_SPACE%REALSPACE =.TRUE.
 CALL VTIME(TV0,TC0)
#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*)
 write(*,*) "CHARGE..."
 write(*,*) "######################################################################"
#ifdef MPI
 endif
#endif
#endif
    spin: DO ISP=1,WDES%ISPIN
       
       ! allocate device memory
       call cublas_alloc_safety (SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,int(c_sizeof(GPU_CHARGE),c_size_t),GPU_CHARGE)  

#ifdef realmode
       call cuda_memset(GPU_CHARGE,0,SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,int(c_sizeof(GPU_CHARGE),c_size_t))
       !CALL gpu_initd(SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,GPU_CHARGE,0,0)
#else
       call cuda_memset(GPU_CHARGE,0,SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,int(c_sizeof(GPU_CHARGE),c_size_t))
       !CALL gpu_initz(-1,SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,GPU_CHARGE,0,zero)
#endif
                                    
       kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL SETWDES(WDES, WDES1, NK)
          
          ! allocate device memory
          call cublas_alloc_safety (WDES1%NGVECTOR,int(c_sizeof(WDES1%NINDPW(1)),c_size_t),GPU_NINDPW)
          call cublas_alloc_safety (WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t),GPU_CW)
          call cublas_alloc_safety (WDES1%GRID%MPLWV*WDES1%NRSPINORS,int(c_sizeof(fakec),c_size_t),GPU_CR)
          ! copy from host to device
          call cublas_Set_Vector(WDES1%NGVECTOR,int(c_sizeof(WDES1%NINDPW(1)),c_int),c_loc(WDES1%NINDPW),1,GPU_NINDPW,1)
          IF (W%OVER_BAND) THEN
             DO N=1,NSTRIP
                CALL REDIS_PW_START(WDES, W%CW(1,N,NK,ISP), N, H_PW)
             ENDDO
          ENDIF
                                    
          band: DO N=1,WDES%NBANDS
             CALL SETWAV(W,W1,WDES1,N,ISP)

             IF (W%OVER_BAND) THEN
                CALL REDIS_PW_STOP (WDES, W%CW(1,N,NK,ISP), N, H_PW)
                IF (N+NSTRIP<=WDES%NBANDS) &
                     CALL REDIS_PW_START(WDES, W%CW(1,N+NSTRIP,NK,ISP), N+NSTRIP, H_PW)
             ENDIF

             WEIGHT=WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)
             IF (WEIGHT==0) CYCLE     
             
             call cublas_Set_Vector(WDES1%NRPLWV,int(c_sizeof(W1%CW(1)),c_int),c_loc(W1%CW),1,GPU_CW,1)
             CALL FFTWAV_W1_GPU(W1,GPU_CW, GPU_CR, GPU_NINDPW)
             !CALL PW_CHARGE_GPU(WDES1, GPU_CHARGE, SIZE(CHARGE_REAL_SPACE%RG,1), GPU_CR, WEIGHT)
             call cuda_pwcharge(NULL_STREAM,WDES1%NRSPINORS,WDES1%GRID%RL%NP,WDES1%GRID%MPLWV,GPU_CHARGE,SIZE(CHARGE_REAL_SPACE%RG,1),GPU_CR,GPU_CR,WEIGHT,CHARGE_IS_REAL)
             call threadsynchronize()

          ENDDO band 

          ! free device memory
          call cublas_free(GPU_NINDPW)
          call cublas_free(GPU_CW)
          call cublas_free(GPU_CR)
       ENDDO kpoints
       
       ! copy from device back to host
       call cublas_Get_Vector(SIZE(CHARGE_REAL_SPACE%RG,1)*WDES1%NRSPINORS*WDES1%NRSPINORS,int(c_sizeof(GPU_CHARGE),c_int),GPU_CHARGE,1,c_loc(CHARGE_REAL_SPACE%RG(1,ISP)),1)

       ! free device memory
       call cublas_free(GPU_CHARGE)
    ENDDO spin
    

    IF (W%OVER_BAND) THEN
       W%OVER_BAND=.FALSE.
       CALL REDIS_PW_DEALLOC(H_PW)
    ENDIF

    ! fourier-transformation of charge-density using GRID_SOFT
    ! only input data from first in-band-group is used, and all nodes
    ! are involved in the FFT, final result is distributed among nodes
    ! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)

    ! merge charge from all bands
#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
       DO I=1,CHARGE_REAL_SPACE%NCDIJ
#ifdef realmode
          CALLMPI( M_sum_d(WDES%COMM_KINTER, CHARGE_REAL_SPACE%RG(1,I),CHARGE_REAL_SPACE%GRID%RL%NP) )
#else
          CALLMPI( M_sum_z(WDES%COMM_KINTER, CHARGE_REAL_SPACE%RG(1,I),CHARGE_REAL_SPACE%GRID%RL%NP) )
#endif
       END DO
    END IF
#endif

    CALL SUMRL_GQ( CHARGE_REAL_SPACE, CHARGE, WDES%COMM_INTER)
    
    CALL FFT_GQ(CHARGE)
      
    ! set the charge-density of unbalanced lattice vectors to 0
    CALL SETUNB_GQ(CHARGE)

    CALL DELWAV_R(W1)
    CALL DEALLOCATE_GRID_QUANTITY(CHARGE_REAL_SPACE)

 CALL VTIME(TV,TC)
#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*) "Time CHARGE...........................................=",TC-TC0
 write(*,*) "######################################################################"
#ifdef MPI
 endif
#endif
#endif

  END SUBROUTINE SOFT_CHARGE_GPU


END MODULE charge_gpu
