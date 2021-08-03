!
! There is one minor ``technical'' problem with MPI:
! MPI requires an include file mpi.h, which is usually not
! F90 free format conform-able (CRAY is one exception).
! Therefore the include file mpi.h must be copied to the
! directory vasp.4.4 and converted to f90 style and named mpif.h.
! This can be done using the following lines:
! \begin{verbatim}
!  > cp ...mpi.../include/mpi.h mpif.h
!  > ./convert mpif.h
! \end{verbatim}
! The convert utility converts a F77 fortran file to a F90 free
! format file and is supplied in the vasp.4 directory.
! On some platforms this is not required.
!      INCLUDE "mpif.h"


! There are no global local sum routines in MPI, thus some workspace
! is required to store the results of the global sum
!      INTEGER,PARAMETER ::  NZTMP=10000, NDTMP=NZTMP*2, NITMP=NDTMP
! workspace for integer, complex, and real
!      COMPLEX(q) :: ZTMP(NZTMP)
!      REAL(q)    :: DTMP(NDTMP)
!      INTEGER    :: ITMP(NITMP)

!      EQUIVALENCE (ZTMP,DTMP)
!      EQUIVALENCE (ZTMP,ITMP)

!      COMMON /SHM_MPI_STORAGE/ ZTMP
