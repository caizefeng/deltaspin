#include "symbol.inc"
#ifdef MPI
      MODULE mpi_shmem

      USE prec
      USE mpimy
      USE c2f_interface

      IMPLICIT NONE

      PUBLIC :: MPI_shmem_complex_1d,MPI_shmem_complex_2d,MPI_shmem_complex_3d, &
                MPI_shmem_real_1d,MPI_shmem_real_2d,MPI_shmem_real_3d, &
                MPI_shmem_alloc,MPI_shmem_dealloc

      PRIVATE

      TYPE MPI_shmem_desc
         INTEGER(c_int), ALLOCATABLE :: shmem_ids(:) 
         TYPE (c_ptr), ALLOCATABLE   :: shmem_ptr(:)
         TYPE (communic), POINTER    :: COMM
      END TYPE MPI_shmem_desc


      TYPE real_ptr_1d
         REAL(q), POINTER    :: fptr(:)
      END TYPE real_ptr_1d

      TYPE real_ptr_2d
         REAL(q), POINTER    :: fptr(:,:)
      END TYPE real_ptr_2d

      TYPE real_ptr_3d
         REAL(q), POINTER    :: fptr(:,:,:)
      END TYPE real_ptr_3d


      TYPE cmplx_ptr_1d
         COMPLEX(q), POINTER :: fptr(:)
      END TYPE cmplx_ptr_1d

      TYPE cmplx_ptr_2d
         COMPLEX(q), POINTER :: fptr(:,:)
      END TYPE cmplx_ptr_2d

      TYPE cmplx_ptr_3d
         COMPLEX(q), POINTER :: fptr(:,:,:)
      END TYPE cmplx_ptr_3d


      TYPE MPI_shmem_complex_1d
         TYPE (MPI_shmem_desc)     :: desc
         COMPLEX(q), POINTER       :: fptr(:)
         TYPE (cmplx_ptr_1d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_complex_1d

      TYPE MPI_shmem_complex_2d
         TYPE (MPI_shmem_desc)     :: desc
         COMPLEX(q), POINTER       :: fptr(:,:)
         TYPE (cmplx_ptr_2d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_complex_2d

      TYPE MPI_shmem_complex_3d
         TYPE (MPI_shmem_desc)     :: desc
         COMPLEX(q), POINTER       :: fptr(:,:,:)
         TYPE (cmplx_ptr_3d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_complex_3d

      TYPE MPI_shmem_real_1d
         TYPE (MPI_shmem_desc)     :: desc
         REAL(q), POINTER          :: fptr(:)
         TYPE (real_ptr_1d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_real_1d

      TYPE MPI_shmem_real_2d
         TYPE (MPI_shmem_desc)     :: desc
         REAL(q), POINTER          :: fptr(:,:)
         TYPE (real_ptr_2d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_real_2d

      TYPE MPI_shmem_real_3d
         TYPE (MPI_shmem_desc)     :: desc
         REAL(q), POINTER          :: fptr(:,:,:)
         TYPE (real_ptr_3d), ALLOCATABLE :: ptrs(:)
      END TYPE MPI_shmem_real_3d


      INTERFACE MPI_shmem_alloc
         MODULE PROCEDURE shmem_alloc_c_1d
         MODULE PROCEDURE shmem_alloc_c_2d
         MODULE PROCEDURE shmem_alloc_c_3d
         MODULE PROCEDURE shmem_alloc_r_1d
         MODULE PROCEDURE shmem_alloc_r_2d
         MODULE PROCEDURE shmem_alloc_r_3d
      END INTERFACE

      INTERFACE MPI_shmem_dealloc
         MODULE PROCEDURE shmem_dealloc_c_1d
         MODULE PROCEDURE shmem_dealloc_c_2d
         MODULE PROCEDURE shmem_dealloc_c_3d
         MODULE PROCEDURE shmem_dealloc_r_1d
         MODULE PROCEDURE shmem_dealloc_r_2d
         MODULE PROCEDURE shmem_dealloc_r_3d
      END INTERFACE

      CONTAINS

!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_c_1d(COMM,dim1,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_complex_1d) :: segment
      TYPE (communic), TARGET     :: COMM
      INTEGER                     :: dim1
      LOGICAL, OPTIONAL           :: LIMIT
      INTEGER, OPTIONAL           :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      COMPLEX(q)        :: cdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(cdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_c_1d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_c_2d(COMM,dim1,dim2,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_complex_2d) :: segment
      TYPE (communic), TARGET     :: COMM
      INTEGER                     :: dim1,dim2
      LOGICAL, OPTIONAL           :: LIMIT
      INTEGER, OPTIONAL           :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      COMPLEX(q)        :: cdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(cdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1*dim2,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1,dim2])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_c_2d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_c_3d(COMM,dim1,dim2,dim3,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_complex_3d) :: segment
      TYPE (communic), TARGET     :: COMM
      INTEGER                     :: dim1,dim2,dim3
      LOGICAL, OPTIONAL           :: LIMIT
      INTEGER, OPTIONAL           :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      COMPLEX(q)        :: cdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(cdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1*dim2*dim3,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1,dim2,dim3])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_c_3d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_r_1d(COMM,dim1,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_real_1d) :: segment
      TYPE (communic), TARGET  :: COMM
      INTEGER                  :: dim1
      LOGICAL, OPTIONAL        :: LIMIT
      INTEGER, OPTIONAL        :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      REAL(q)           :: rdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(rdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_r_1d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_r_2d(COMM,dim1,dim2,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_real_2d) :: segment
      TYPE (communic), TARGET  :: COMM
      INTEGER                  :: dim1,dim2
      LOGICAL, OPTIONAL        :: LIMIT
      INTEGER, OPTIONAL        :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      REAL(q)           :: rdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(rdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1*dim2,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1,dim2])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_r_2d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_alloc_r_3d(COMM,dim1,dim2,dim3,segment,LIMIT,ISTAT)
      TYPE (MPI_shmem_real_3d) :: segment
      TYPE (communic), TARGET  :: COMM
      INTEGER                  :: dim1,dim2,dim3
      LOGICAL, OPTIONAL        :: LIMIT
      INTEGER, OPTIONAL        :: ISTAT
      ! local variables
      INTEGER(c_size_t) :: segmentsize
      INTEGER           :: sizeoftype

      INTEGER           :: NCPU,myid,i

      REAL(q)           :: rdummy

      LOGICAL           :: LRANK0ONLY=.FALSE.

      CALL M_barrier(COMM)

      IF (PRESENT(LIMIT)) LRANK0ONLY=LIMIT
      IF (PRESENT(ISTAT)) ISTAT=0

      segment%desc%COMM => COMM

      sizeoftype=SIZEOF(rdummy)

      NCPU=COMM%NCPU

      segmentsize=sizeoftype*max(dim1*dim2*dim3,1)

      myid=COMM%NODE_ME

      IF (LRANK0ONLY) THEN
         IF (myid/=1) segmentsize=0
         NCPU=1
      ENDIF

      ALLOCATE(segment%desc%shmem_ids(NCPU))

      DO i=1,NCPU
         IF (i==myid) THEN
            IF (PRESENT(ISTAT)) THEN
               CALL getshmem_error(segmentsize,segment%desc%shmem_ids(i))
               IF (segment%desc%shmem_ids(i)==-1) ISTAT=ISTAT+1
            ELSE
               CALL getshmem(segmentsize,segment%desc%shmem_ids(i))
            ENDIF
         ELSE
            segment%desc%shmem_ids(i)=0
         ENDIF
      ENDDO

      IF (PRESENT(ISTAT)) THEN
         CALL M_sum_i(COMM,ISTAT,1)
         IF (ISTAT/=0) RETURN
      ENDIF

      CALL M_sum_i(COMM,segment%desc%shmem_ids(1),NCPU)

      ALLOCATE(segment%desc%shmem_ptr(NCPU),segment%ptrs(NCPU))

      DO i=1,NCPU
         CALL attachshmem(segment%desc%shmem_ids(i),segment%desc%shmem_ptr(i))
         CALL c_f_pointer(segment%desc%shmem_ptr(i),segment%ptrs(i)%fptr,[dim1,dim2,dim3])
         IF (i==myid) CALL destroyshmem(segment%desc%shmem_ids(i))
      ENDDO

      IF (LRANK0ONLY) THEN
         segment%fptr => segment%ptrs(1)%fptr
      ELSE
         segment%fptr => segment%ptrs(myid)%fptr
      ENDIF

      CALL M_barrier(COMM)

      RETURN
      END SUBROUTINE shmem_alloc_r_3d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_c_1d(segment)
      TYPE (MPI_shmem_complex_1d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_c_1d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_c_2d(segment)
      TYPE (MPI_shmem_complex_2d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_c_2d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_c_3d(segment)
      TYPE (MPI_shmem_complex_3d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_c_3d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_r_1d(segment)
      TYPE (MPI_shmem_real_1d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_r_1d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_r_2d(segment)
      TYPE (MPI_shmem_real_2d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_r_2d


!**********************************************************************
!
!**********************************************************************

      SUBROUTINE shmem_dealloc_r_3d(segment)
      TYPE (MPI_shmem_real_3d) :: segment
      ! local variables
      INTEGER :: i,n
      
      CALL M_barrier(segment%desc%COMM)

      n=SIZE(segment%desc%shmem_ids)

      DO i=1,n
         CALL detachshmem(segment%desc%shmem_ptr(i))
         NULLIFY(segment%ptrs(i)%fptr)
      ENDDO

      DEALLOCATE(segment%desc%shmem_ids,segment%desc%shmem_ptr,segment%ptrs)

      NULLIFY(segment%fptr)

      CALL M_barrier(segment%desc%COMM)

      NULLIFY(segment%desc%COMM)
 
      RETURN
      END SUBROUTINE shmem_dealloc_r_3d


      END MODULE mpi_shmem
#else
      MODULE mpi_shmem
      CONTAINS
      SUBROUTINE mpi_shmem_dummy
      WRITE(*,*)'Im a DEC compiler so I need this line'
      END SUBROUTINE
      END MODULE mpi_shmem
#endif
