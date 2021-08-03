#include "symbol.inc"

      MODULE main_gpu
      USE prec
      USE gpu, ONLY : use_gpu

      IMPLICIT NONE

      CONTAINS

!******************** SUBROUTINE GPU_BANNER ****************************
!
! Prints a GPU related banner to stdout.
!
!***********************************************************************
      SUBROUTINE GPU_BANNER(IO)
      USE base
      TYPE (in_struct) IO

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,*) ' '
         WRITE(IO%IU0,*) '*******************************************************************************'
         WRITE(IO%IU0,*) ' You are running the GPU port of VASP! When publishing results obtained with'
         WRITE(IO%IU0,*) ' this version, please cite:'
         WRITE(IO%IU0,*) '  - M. Hacene et al., http://dx.doi.org/10.1002/jcc.23096'
         WRITE(IO%IU0,*) '  - M. Hutchinson and M. Widom, http://dx.doi.org/10.1016/j.cpc.2012.02.017'
         WRITE(IO%IU0,*) ' '
         WRITE(IO%IU0,*) ' in addition to the usual required citations (see manual).'
         WRITE(IO%IU0,*) ' '
         WRITE(IO%IU0,*) ' GPU developers: A. Anciaux-Sedrakian, C. Angerer, and M. Hutchinson.'
         WRITE(IO%IU0,*) '*******************************************************************************'
         WRITE(IO%IU0,*) ' '
#ifdef GPUDIRECT
         WRITE(IO%IU0,*) 'Using CUDA Aware MPI / GPU-Direct ! '
#endif
      ENDIF
      RETURN
      END SUBROUTINE GPU_BANNER


!******************** SUBROUTINE GPU_TEST ******************************
!
!***********************************************************************
      SUBROUTINE GPU_TEST(INFO,NPAR,IO)
      USE base
      USE pead
      USE main_mpi
      USE cuda_interface
      USE scala, ONLY: LscaAWARE
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      INTEGER NPAR

      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (40) :: STRING

      IF (NPAR/=COMM_KIN%NCPU) THEN
         IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,*) ' '
            WRITE(IO%IU0,*) 'ERROR: It is currently not supported to run the GPU port of VASP with:'
            WRITE(IO%IU0,*) ' NPAR /= number of MPI-ranks divided by KPAR (i.e., NCORE/=1).'
            WRITE(IO%IU0,*) 'Currently: KPAR=',KPAR,' NPAR=',NPAR,' MPI-ranks=',COMM%NCPU
            WRITE(IO%IU0,*) ' '
         ENDIF
         STOP
      ENDIF

      ! Parse the ALGO-tag (again)
      LOPEN=.FALSE.
      OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
      STRING="--"
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'ALGO','=','#',';','S', &
     &            IDUM,RDUM,CDUM,LDUM,STRING,N,40,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''ALGO'' from file INCAR.'
      ENDIF
      CLOSE(IO%IU5)

      IF (.NOT.((INFO%IALGO==38).OR.(INFO%IALGO==48).OR.(INFO%IALGO==68)) &
     &     .OR.((STRING(1:1)/='n').AND.(STRING(1:1)/='f').AND.(STRING(1:1)/='v').AND.(STRING(1:2)/='--'))) THEN
         IF(IO%IU0>=0) THEN
             WRITE(IO%IU0,*) ' '
             WRITE(IO%IU0,*) 'WARNING: The GPU port of VASP has been extensively'
             WRITE(IO%IU0,*) 'tested for: ALGO=Normal, Fast, and VeryFast.'
             WRITE(IO%IU0,*) 'Other algorithms may produce incorrect results or'
             WRITE(IO%IU0,*) 'yield suboptimal performance. Handle with care!'
             WRITE(IO%IU0,*) ' ' 
         ENDIF
!        use_gpu=.FALSE.
      ENDIF

      IF (INFO%LREAL .eqv. .FALSE.) THEN
         IF(IO%IU0>=0) THEN
             WRITE(IO%IU0,*) ' '
             WRITE(IO%IU0,*) 'ERROR: It is currently not supported to run the GPU port of VASP with:'
             WRITE(IO%IU0,*) 'LREAL=.FALSE.'
             WRITE(IO%IU0,*) ' '
         ENDIF
         STOP
      ENDIF

      IF (LscaAWARE) THEN
         IF(IO%IU0>=0) THEN
             WRITE(IO%IU0,*) ''
             WRITE(IO%IU0,*) 'ERROR: It is currently not supported to run the GPU port of VASP with:'
             WRITE(IO%IU0,*) 'LscaAWARE=.TRUE'
             WRITE(IO%IU0,*) 'Set it to .FALSE. in the INCAR or recompile VASP without -DscaLAPACK'
             WRITE(IO%IU0,*) ''
         ENDIF
         STOP
      ENDIF

      IF (LPEAD_CALC_EPS().OR.LPEAD_NONZERO_EFIELD()) THEN
         IF(IO%IU0>=0) THEN
             WRITE(IO%IU0,*) ' '
             WRITE(IO%IU0,*) 'ERROR: It is currently not supported to run the GPU port of VASP with:'
             WRITE(IO%IU0,*) 'LCALCEPS=.TRUE. or EFIELD_PEAD/=0, ... exiting VASP.'
             WRITE(IO%IU0,*) ' '
         ENDIF
         STOP
      ENDIF

      RETURN
      END SUBROUTINE GPU_TEST


!******************** SUBROUTINE GPU_INIT ******************************
!
!***********************************************************************
      SUBROUTINE GPU_INIT(WDES)
      USE wave
      USE cuda_interface
      TYPE (wavedes) WDES
      ! local variables
      INTEGER NSIM

      ! initialize timers
      nv_profile_malloc(NUM_TIMERS)
      nv_profile_start(NVP_TOTAL)
      nv_profile_start(NVP_VASP) 

!     ! initialize CUDA
!     CALL cuda_init(NUM_CUDASIM)

      ! make sure we use the same NSIM as in electron.F
      NSIM=WDES%NSIM*2
#ifdef MPI
      NSIM=((WDES%NSIM*2+WDES%COMM_INTER%NCPU-1)/WDES%COMM_INTER%NCPU)*WDES%COMM_INTER%NCPU
#endif
      CALL cuda_init(NV_NUM_STREAMS, NSIM)
      ! initialize CUBLAS
      CALL cublas_init()
#ifdef USE_MAGMA
      ! initialize magma
      CALL magma_init()
#endif
      ! initialize CUFFT
      CALL cufft_init(NV_NUM_STREAMS,WDES%GRID%NGX,WDES%GRID%NGY,WDES%GRID%NGZ)

      RETURN
      END SUBROUTINE GPU_INIT


!******************** SUBROUTINE GPU_FINALIZE **************************
!
!***********************************************************************
      SUBROUTINE GPU_FINALIZE()
      USE cuda_interface

      ! destroy CUFFT
      CALL cufft_destroy()
#ifdef MAGMA
      ! cleanup magma
      CALL magma_finalize()
#endif
      ! destroy CUBLAS
      CALL cublas_destroy()
      ! destroy CUDA
      CALL cuda_destroy()

      ! finalize timing
      nv_profile_stop(NVP_VASP)
      nv_profile_stop(NVP_TOTAL)
      nv_profile_print(NODE_ME)
      nv_profile_free()

      RETURN
      END SUBROUTINE GPU_FINALIZE

      END MODULE main_gpu
