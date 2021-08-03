#include "symbol.inc"
#ifndef CACHE_SIZE
#define CACHE_SIZE 8000
#endif
#ifndef MINLOOP
#define MINLOOP 1
#endif

#undef FFTMAKEPLAN
#undef FFT3D 
#undef FFT3DR 
#undef FFTCHK
#undef FFT3RC
#undef FFTEXT
#undef FFTWAV
#undef FFTINI


   SUBROUTINE FFTBAS_PLAN_MPI_GPU(A,GRID,ISIGN,IPLAN)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE smart_allocate
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRID
      REAL(q), TARGET :: A(*)
      INTEGER(c_intptr_t) GPU_A
      INTEGER ISIGN   !  direction of fft
      INTEGER IPLAN   !  make a plan (/=FFTW_ESTIMATE)
      COMPLEX(q),POINTER,SAVE ::  RCVBUF(:),SNDBUF(:)
      !INTEGER :: planx, plany, planz
      INTEGER(c_intptr_t) :: gpu_planx, gpu_plany, gpu_planz
      INTEGER A_SIZE
      COMPLEX(q) fakec

      include 'fftw3.f'
!=======================================================================
! initialization
!=======================================================================
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)

      CALL SMART_ALLOCATE_COMPLEX(RCVBUF,GRID%MPLWV)
      CALL SMART_ALLOCATE_COMPLEX(SNDBUF,GRID%MPLWV)

      IDX=NX
      IDY=NY
      IDZ=NZ

      call getbatchedfftplan(gpu_planx,1, NX, GRID%RC%NCOL, &
                              NX, 1, IDX, NX, 1, IDX, FFT_C2C)
      call getbatchedfftplan(gpu_plany,1, NY, GRID%IN%NCOL, &
                              NY, GRID%IN%NCOL, 1, NY, GRID%IN%NCOL, 1, FFT_C2C)
      IF (ISIGN==1) THEN
         !CALL dfftw_plan_many_dft(planx, 1, NX , GRID%RC%NCOL, &
         !                    A(1), NX, 1 , IDX, &
         !                    A(1), NX, 1 , IDX, &
         !                    FFTW_BACKWARD, IPLAN)
         !CALL dfftw_plan_many_dft(plany, 1, NY , GRID%IN%NCOL, &
         !                    A(1), NY, GRID%IN%NCOL, 1 , &
         !                    A(1), NY, GRID%IN%NCOL, 1 , &
         !                    FFTW_BACKWARD, IPLAN)
         IF (NZ/2+1==GRID%NGZ_rd) THEN
!           WRITE(*,*) 'detected real to complex'
           !CALL dfftw_plan_many_dft_c2r(planz, 1, NZ , GRID%RL_FFT%NCOL, &
           !                  A(1), NZ, 1, (IDZ+2)/2 , &
           !                  A(1), NZ, 1, IDZ+2 , &
           !                  IPLAN)
           call getbatchedfftplan(gpu_planz,1, NZ, GRID%RL_FFT%NCOL, &
                                NZ, 1, (IDZ+2)/2, NZ, 1, IDZ+2, FFT_C2R)
         ELSE
           WRITE(*,*) 'complex to complex!'          
           !CALL dfftw_plan_many_dft(planz, 1, NZ , GRID%RL_FFT%NCOL, &
           !                  A(1), NZ, 1, IDZ , &
           !                  A(1), NZ, 1, IDZ , &
           !                  FFTW_BACKWARD, IPLAN)
           call getbatchedfftplan(gpu_planz,1, NZ, GRID%RL_FFT%NCOL, &
                                NZ, 1, IDZ, NZ, 1, IDZ, FFT_C2C)
         ENDIF
      ELSE
         IF (NZ/2+1==GRID%NGZ_rd) THEN
!           WRITE(*,*) 'detected inverse real to complex'
           !CALL dfftw_plan_many_dft_r2c(planz, 1, NZ , GRID%RL_FFT%NCOL, &
           !                  A(1), NZ, 1, IDZ+2 , &
           !                  A(1), NZ, 1, (IDZ+2)/2 , &
           !                  IPLAN)
           call getbatchedfftplan(gpu_planz,1, NZ, GRID%RL_FFT%NCOL, &
                                NZ, 1, IDZ+2, NZ, 1, (IDZ+2)/2, FFT_R2C)

         ELSE
!           WRITE(*,*) 'detected inverse complex to complex'
           !CALL dfftw_plan_many_dft(planz, 1, NZ , GRID%RL_FFT%NCOL, &
           !                  A(1), NZ, 1, IDZ , &
           !                  A(1), NZ, 1, IDZ , &
           !                  FFTW_FORWARD, IPLAN)
           call getbatchedfftplan(gpu_planz,1, NZ, GRID%RL_FFT%NCOL, &
                                NZ, 1, IDZ, NZ, 1, IDZ, FFT_C2C)
         ENDIF
         !CALL dfftw_plan_many_dft(plany, 1, NY , GRID%IN%NCOL, &
         !                    A(1), NY, GRID%IN%NCOL, 1 , &
         !                    A(1), NY, GRID%IN%NCOL, 1 , &
         !                    FFTW_FORWARD, IPLAN)
         !CALL dfftw_plan_many_dft(planx, 1, NX , GRID%RC%NCOL, &
         !                    A(1), NX, 1 , IDX, &
         !                    A(1), NX, 1 , IDX, &
         !                    FFTW_FORWARD, IPLAN)
      ENDIF
      A_SIZE = GRID%MPLWV
      call cublas_alloc_safety(A_SIZE, int(c_sizeof(fakec),c_size_t), GPU_A)
      call cublas_Set_Vector(A_SIZE, int(c_sizeof(fakec),c_int), c_loc(A), 1, GPU_A, 1)

!=======================================================================
! do the transformation forward (q->r)
!=======================================================================
       IF (ISIGN ==1 .AND. IPLAN==FFTW_ESTIMATE) THEN
! transformation along first dimension:
         !CALL dfftw_execute(planx)
         call cufft_exec_plan_c2c(gpu_planx, GPU_A, GPU_A, ISIGN)
         call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         CALL MAP_FORWARD(A(1), GRID%IN%NALLOC, SNDBUF(1), RCVBUF(1), GRID%RC_IN, GRID%COMM)
! transformation along second dimension:
         !CALL dfftw_execute(plany)
         call cublas_Set_Vector(A_SIZE, int(c_sizeof(fakec),c_int),c_loc(A),1,GPU_A, 1) 
         call cufft_exec_plan_c2c(gpu_plany, GPU_A, GPU_A, ISIGN)
         call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         CALL MAP_FORWARD(A(1), GRID%RL_FFT%NALLOC, SNDBUF(1), RCVBUF(1), GRID%IN_RL, GRID%COMM)
! transformation along third dimension:
        ! CALL dfftw_execute(planz)
         call cublas_Set_Vector(A_SIZE, int(c_sizeof(fakec),c_int),c_loc(A),1,GPU_A, 1) 
         IF (NZ/2+1==GRID%NGZ_rd) THEN
           call cufft_exec_plan_c2r(gpu_planz, GPU_A, GPU_A)
           call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         ELSE
           call cufft_exec_plan_c2c(gpu_planz, GPU_A, GPU_A, ISIGN)
           call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         ENDIF
!=======================================================================
! do the transformation backward (r->q)
!=======================================================================
       ELSE IF(IPLAN==FFTW_ESTIMATE) THEN
! transformation along third dimension:
         !CALL  dfftw_execute(planz)
         IF (NZ/2+1==GRID%NGZ_rd) THEN
           call cufft_exec_plan_r2c(gpu_planz, GPU_A, GPU_A)
           call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         ELSE
           call cufft_exec_plan_c2c(gpu_planz, GPU_A, GPU_A, ISIGN)
           call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
         ENDIF
         CALL MAP_BACKWARD(A(1), GRID%IN%NALLOC, SNDBUF(1), RCVBUF(1), GRID%IN_RL, GRID%COMM)
         !CALL  dfftw_execute(plany)
         call cublas_Set_Vector(A_SIZE, int(c_sizeof(fakec),c_int),c_loc(A),1,GPU_A, 1) 
         call cufft_exec_plan_c2c(gpu_plany, GPU_A, GPU_A, ISIGN)
         call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1)        
! transformation along second dimension:
         CALL MAP_BACKWARD(A(1), GRID%RC%NALLOC, SNDBUF(1), RCVBUF(1), GRID%RC_IN, GRID%COMM)
! transformation along first dimension:
         !CALL dfftw_execute(planx)
         call cublas_Set_Vector(A_SIZE, int(c_sizeof(fakec),c_int),c_loc(A),1,GPU_A, 1) 
         call cufft_exec_plan_c2c(gpu_planx, GPU_A, GPU_A, ISIGN)
         call cublas_Get_Vector(A_SIZE, int(c_sizeof(fakec),c_int),GPU_A,1,c_loc(A), 1) 
      ENDIF
!Do we need to destroy any plans? Original was:
      !call dfftw_destroy_plan(planx)
      !call dfftw_destroy_plan(plany)
      !call dfftw_destroy_plan(planz)

      call cublas_free(GPU_A)
      RETURN
    END SUBROUTINE FFTBAS_PLAN_MPI_GPU
