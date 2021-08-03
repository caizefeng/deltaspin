#include "symbol.inc"
!************************************************************************
! RCS:  $Id: shm.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!  this module contains the routines required to handle
!  shmem on the T3D
!  it allocates the required amount and returns a position to the
!  workspace as required
!
!***********************************************************************

!=======================================================================
!
! the first subroutine calculates the maximum amount of workspace
! required
!
!=======================================================================

    SUBROUTINE SHM_MAX(WDES, MPLMAX, MALLOC)
      USE prec
      USE wave
      USE main_mpi
      USE scala
      IMPLICIT NONE
      INTEGER MPLMAX            ! maxmimum number of fft coefficients
      INTEGER MALLOC            ! amount of workspace need
      INTEGER MSCALA
      TYPE (wavedes)    WDES
#ifdef gammareal
      INTEGER, PARAMETER :: MCOMP_=1
#else
      INTEGER, PARAMETER :: MCOMP_=2
#endif
      MALLOC=0

#ifdef F90_T3D
! we do need shm only if remote put is used (T3D_SMA)
#ifdef T3D_SMA
!
! ffts (fftmpi.F)
!
      MALLOC= MPLMAX*4
!
! subroutine REDIS_PW (wave_mpi.F)
!
      MALLOC= MAX(MALLOC, WDES%NRPLWV*2)
      MALLOC= MAX(MALLOC, WDES%NPROD*2)
!
! fast interchange global sum M_sumf_d (mpi.F)
! one can save workspace by undocumenting this line
!
      MALLOC= MAX(MALLOC, WDES%NB_TOT*WDES%NB_TOT*MCOMP_)
#endif
! AND for scaLAPACK
! scaLAPACK, T3D required shmem for scaLAPACK
! calculate required workspace now
!
      MSCALA=0
      IF ( LscaLAPACK ) THEN
         WRITE(*,*) 'internal warning in VASP: no shmem for scaLAPACK, but that should be fine'
         CALL INIT_scala(WDES%COMM_KIN, WDES%NB_TOT )
      ENDIF
      MALLOC=MAX(MALLOC,MSCALA)

!
! just in case global maximum of this value on all nodes
      CALLMPI( M_max_i(COMM_WORLD, MALLOC ,1))
#endif

    END SUBROUTINE SHM_MAX

!=======================================================================
!
! the second subroutine allocates the required amount of workspace
!
!=======================================================================

    SUBROUTINE SHM_ALLOC(MALLOC)
      USE prec
      USE scala
      IMPLICIT NONE
      INTEGER MALLOC            ! amount of workspace need
      INTEGER MALLOC_DONE       ! amount of workspace allocated
      INTEGER INFO
#ifdef F90_T3D
      COMMON /SHM/ MALLOC_DONE, PBUF
      POINTER ( PBUF, BUF )
      REAL(q) :: BUF(MALLOC)

      MALLOC_DONE=MALLOC
      IF (MALLOC /=0 ) CALL SHPALLOC( PBUF, MALLOC, INFO, 1)
#endif

      RETURN
    END SUBROUTINE SHM_ALLOC


!=======================================================================
!
! check whether sufficient workspace was allocated
!
!=======================================================================

    FUNCTION ISHM_CHECK(MALLOC)
      USE prec
      IMPLICIT NONE
      INTEGER ISHM_CHECK        ! returns 1 if workspace is sufficient
      INTEGER MALLOC            ! amount of workspace need
      INTEGER MALLOC_DONE       ! amount of workspace allocated

#ifdef F90_T3D
      COMMON /SHM/ MALLOC_DONE, PBUF
      POINTER ( PBUF, BUF )
      REAL(q) :: BUF(MALLOC)

      IF ( MALLOC_DONE >= MALLOC ) THEN
         ISHM_CHECK=1
      ELSE
         ISHM_CHECK=0
      ENDIF
#else
      ISHM_CHECK=0
#endif
      RETURN
    END FUNCTION ISHM_CHECK

