#include "symbol.inc"
      MODULE main_mpi
      USE prec
      USE base
      USE mpimy
      USE openmp, ONLY : omp_nthreads
      IMPLICIT NONE

!***********************************************************************
! RCS:  $Id: main_mpi.F,v 1.3 2002/04/16 07:28:45 kresse Exp $
!
! This module initializes the communication univeres used in VASP
!
! we have five communicator (see below)
!
!***********************************************************************

      TYPE (communic),TARGET :: COMM_WORLD ! our world wide communicator
      TYPE (communic),TARGET :: COMM_CHAIN ! communication between images
      TYPE (communic),TARGET :: COMM       ! one image communicator
      TYPE (communic),TARGET :: COMM_KINTER ! between k-points communicator
      TYPE (communic),TARGET :: COMM_KIN   ! in k-point communicator
      TYPE (communic),TARGET :: COMM_INTER ! between-band communicator
      TYPE (communic),TARGET :: COMM_INB   ! in-band communicator
#ifdef use_shmem
      TYPE (communic),TARGET :: COMM_SHMEM
      TYPE (communic),TARGET :: COMM_intra_node,COMM_inter_node
#endif
      INTEGER :: IMAGES=0                  ! total number of images
      INTEGER :: KIMAGES=0                 ! distribution of k-points
      INTEGER :: NCORE_IN_IMAGE1=0         ! if IMAGES is 2 it is possible to specify the number of cores
                                           ! that are in the first image
      CHARACTER(LEN=10) ::  DIR_APP
      INTEGER           ::  DIR_LEN=0
      INTEGER           ::  KPAR,NCORE
#ifdef use_shmem
      INTEGER NCSHMEM
#else
      INTEGER, PARAMETER :: NCSHMEM=1
#endif

!***********************************************************************
!
! initialize MPI and read entry IMAGES and NPAR from INCAR
! and sub divide communicators
! because redistribution of nodes
! might be done one must do this as early as possible
!
!
!***********************************************************************

      CONTAINS
#if defined(MPI) || defined(MPI_CHAIN)
      SUBROUTINE INIT_MPI(NPAR,IO)

      TYPE (in_struct)   IO
      INTEGER NPAR
! local
      REAL(q)     RDUM  ; CHARACTER (1)   CHARAC
      COMPLEX(q)  CDUM  ; LOGICAL  LDUM
      INTEGER     N,IERR, IDUM
      INTEGER     node
      LOGICAL     LVCAIMAGES
      REAL(q)     VCAIMAGES
!!! test_
!!      CHARACTER*(MPI_MAX_PROCESSOR_NAME) myname
!!      INTEGER :: I,resultlen,ierror
!!! test_

      CALL M_init(COMM_WORLD)

      VCAIMAGES=-1
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'VCAIMAGES','=','#',';','F', &
           &        IDUM,VCAIMAGES,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) WRITE(IO%IU0,*)'Error reading item ''VCAIMAGES'' from file INCAR.'
         STOP
      ENDIF
      CALL XML_INCAR('VCAIMAGES','F',IDUM,VCAIMAGES,CDUM,LDUM,CHARAC,N)
      IF (VCAIMAGES==-1) THEN
         LVCAIMAGES=.FALSE.
      ELSE
         LVCAIMAGES=.TRUE.
      ENDIF

      NCORE_IN_IMAGE1=0
      IF (LVCAIMAGES) THEN
         IMAGES=2
         CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NCORE_IN_IMAGE1','=','#',';','I', &
     &        NCORE_IN_IMAGE1,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &        ((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''NCORE_IN_IMAGE1'' from file INCAR.'
            STOP
         ENDIF
      ELSE
         IMAGES=0
         CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'IMAGES','=','#',';','I', &
     &        IMAGES,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &        ((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''IMAGES'' from file INCAR.'
            STOP
         ENDIF
      ENDIF

      KIMAGES=0
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'KIMAGES','=','#',';','I', &
     &            KIMAGES,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''KIMAGES'' from file INCAR.'
         STOP
      ENDIF
      IF (KIMAGES>0) THEN
         IDUM=0
         CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'FOURORBIT','=','#',';','I', &
        &            IDUM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''FOURORBIT'' from file INCAR.'
            IDUM=0
         ENDIF
         IF (IDUM/=1.AND.IO%IU0>=0) THEN
            WRITE(IO%IU0,*)'Distribution of k-points over KIMAGES only works'
            WRITE(IO%IU0,*)'in combination with FOURORBIT=1, sorry, stopping ...'
            STOP 
         ENDIF          
      ENDIF
! KPAR division of kpoints,  default to unity in case only 1 k-point
      KPAR=1
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'KPAR','=','#',';','I', &
     &            KPAR,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''KPAR'' from file INCAR.'
         STOP
      ENDIF
      IF (KPAR>1.AND.KIMAGES>0) THEN
         WRITE (IO%IU0,*) 'Untested combination of FOURORBIT with KPAR'
         WRITE (IO%IU0,*) ' (k-point parallelization), sorry,  stopping...'
         STOP
      END IF
!----------------------------------------------------------------------
! M_divide: creates a 2 dimensional cartesian topology
!  for seperate images or work groups
!  each work group (image) will run VASP independently in one
!  sub directory (01-99) of the current directory
! this mode is required to support either independent calculations
! on parallel machines or the nudged elastic band method
!----------------------------------------------------------------------
      IF (IMAGES>0) THEN
         IF (IMAGES==2 .AND. NCORE_IN_IMAGE1/=0) THEN
            CALL M_initc( COMM_WORLD)
            ! M_divide2 calls M_initc for COMM and COMM_CHAIN
            CALL M_divide2( COMM_WORLD, NCORE_IN_IMAGE1, COMM_CHAIN, COMM)
         ELSE
            CALL M_divide( COMM_WORLD, IMAGES, COMM_CHAIN, COMM, .TRUE. )
            CALL M_initc( COMM_WORLD)
            CALL M_initc( COMM)
            CALL M_initc( COMM_CHAIN)
         ENDIF
      ELSEIF (KIMAGES>0) THEN
         CALL M_divide( COMM_WORLD, KIMAGES, COMM_CHAIN, COMM, .TRUE. )
         CALL M_initc( COMM_WORLD)
         CALL M_initc( COMM)
         CALL M_initc( COMM_CHAIN)      
      ELSE
         CALL M_initc( COMM_WORLD)
         COMM=COMM_WORLD
         COMM_CHAIN=COMM
      ENDIF
      IF ( COMM_WORLD%NODE_ME == 1 ) &
        WRITE(IO%IU0,'(" running on ",I4," total cores")') COMM_WORLD%NCPU
      IF ( COMM_WORLD%NODE_ME == 1 .AND. (IMAGES>0.OR.KIMAGES>0) ) &
        WRITE(IO%IU0,'(" each image running on ",I4," cores")') COMM%NCPU

      IF (KPAR>=1) THEN
!----------------------------------------------------------------------
! M_divide: creates a 2 dimensional cartesian topology within one
!  work group (image)
!  this is required for simultaneous distribution over k-points and bands
!----------------------------------------------------------------------
         CALL M_divide( COMM, KPAR, COMM_KINTER, COMM_KIN, .FALSE.)
         CALL M_initc( COMM)   ! probably not required but who knows
         CALL M_initc( COMM_KINTER)
         CALL M_initc( COMM_KIN)
         IF ( COMM_WORLD%NODE_ME == 1 ) &
         WRITE(IO%IU0,'(" distrk:  each k-point on ",I4," cores, ",I4," groups")') &
                        COMM_KIN%NCPU,COMM_KINTER%NCPU
      ELSE
         COMM_KINTER = COMM
         COMM_KIN    = COMM
      ENDIF

! NCORE species onto how many cores a band is distributed 
! often this values can be now set to the number of cores per node
! this is more handy than NPAR in most cases
      NCORE=1
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NCORE','=','#',';','I', &
     &            NCORE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''NCORE'' from file INCAR.'
         STOP
      ENDIF

      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NCORES_PER_BAND','=','#',';','I', &
     &            NCORE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''NCORES_PER_BAND'' from file INCAR.'
         STOP
      ENDIF
      NCORE=MAX(MIN(COMM_KIN%NCPU,NCORE),1)

! NPAR number of bands distributed over processors, defaults to
! COMM%NCPU/NCORE
      NPAR=MAX(COMM_KIN%NCPU/NCORE,1)

      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NPAR','=','#',';','I', &
     &            NPAR,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''NPAR'' from file INCAR.'
         STOP
      ENDIF


      IF (NPAR>=1) THEN
!----------------------------------------------------------------------
! M_divide: creates a 2 dimensional cartesian topology within one
!  work group (image)
!  this is required for simultaneous distribution over bands and plane
!  wave coefficients
!  communicators are created for inter-band in intra-band communication
!  NPAR is the number of bands over which is parallelized
! the resulting layout will be the following (NPAR=4, NCPU=8):
! (MPI uses allways Row-major layout)
! wave1           0 (0,0)        1 (0,1)
! wave2           2 (1,0)        3 (1,1)
! wave3           4 (2,0)        5 (2,1)
! wave4           6 (3,0)        7 (3,1)
! wave5           0 (0,0)        1 (0,1)
! etc.
!
! the sub-communicators COMM_INB are one dimensional communicators
! which allow communication within one row i.e. nodes are grouped to
!  0-1     2-3        4-5      6-7
! for shortness these groups will we called in-band-groups
! there communicators are called in-band-communicator
!
! the sub-communicators COMM_INTER are one dimensional communicators
! which allow communication within one column i.e nodes are grouped
!  0-2-4-6     and    1-3-5-7
! these groups will we called inter-band-groups
!
! the most complicated thing is the FFT of soft chargedensities
! the following algorithm is used:
! the soft chargedensity is calculated in real space,
! charge from all bands is merged to processor 0 and 1 using
! the COMM_INTER
! than a FFT involving all processors is done (on processors 2-7
! no components exist in real space)
! the final result in reciprocal space is defined on all processors
! (see SET_RL_GRID for more information)
!----------------------------------------------------------------------
         CALL M_divide( COMM_KIN, NPAR, COMM_INTER, COMM_INB, .FALSE.)
         CALL M_initc( COMM_KIN)   ! propably not required but who knows
         CALL M_initc( COMM_INTER)
         CALL M_initc( COMM_INB)
         IF ( COMM_WORLD%NODE_ME == 1 ) &
         WRITE(IO%IU0,'(" distr:  one band on ",I4," cores, ",I4," groups")') &
                        COMM_INB%NCPU,COMM_INTER%NCPU
      ELSE
         COMM_INTER = COMM
         COMM_INB   = COMM
      ENDIF


#ifdef use_shmem
      NCSHMEM=1
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NCSHMEM','=','#',';','I', &
     &            NCSHMEM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''NCSHMEM'' from file INCAR.'
         STOP
      ENDIF
      CALL M_divide_shmem(COMM_WORLD,COMM_INTER,NCSHMEM,COMM_SHMEM)
      CALL M_divide_intra_inter_node(COMM_WORLD,COMM_INTER,COMM_intra_node,COMM_inter_node)
#endif

      IF (COMM%NODE_ME /= COMM%IONODE) IO%IU6 = -1
      IF (COMM%NODE_ME /= COMM%IONODE) IO%IU0 = -1
      
      IF (KIMAGES>0.AND.COMM_WORLD%NODE_ME/=COMM_WORLD%IONODE) IO%IU6 = -1
      IF (KIMAGES>0.AND.COMM_WORLD%NODE_ME/=COMM_WORLD%IONODE) IO%IU0 = -1

      IF (KPAR>1.AND.COMM%NODE_ME/=COMM%IONODE) IO%IU6 = -1
      IF (KPAR>1.AND.COMM%NODE_ME/=COMM%IONODE) IO%IU0 = -1


      IF ( IO%IU0/=-1 .AND. IO%IU6 == -1) THEN
         WRITE(*,*)'internal ERROR: io-unit problem',IO%IU0,IO%IU6
         STOP
      ENDIF

      node=COMM_CHAIN%NODE_ME

      CALL MAKE_DIR_APP(node)

! if all nodes should write (giving complete mess) do not use the
! following line
      IF (COMM_WORLD%NODE_ME /= COMM_WORLD%IONODE .AND. IO%IU0>0) THEN
         OPEN(UNIT=IO%IU0,FILE=DIR_APP(1:DIR_LEN)//'stdout',STATUS='UNKNOWN')
      ENDIF
!----------------------------------------------------------------------
! try to go to subdir INCAR's for the rest
!----------------------------------------------------------------------
      IF (.NOT. IO%LOPEN) THEN
         CLOSE(IO%IU5)
      ENDIF
         
      OPEN(UNIT=IO%IU5,FILE=DIR_APP(1:DIR_LEN)//INCAR,STATUS='OLD',IOSTAT=IERR)
      IF (IERR==0) THEN
         INCAR=DIR_APP(1:DIR_LEN)//INCAR
         CLOSE(IO%IU5)
         IF (IO%IU0>=0) WRITE(IO%IU0,*) 'using from now: ',INCAR
      ENDIF
      IF (.NOT. IO%LOPEN) THEN
         OPEN(UNIT=IO%IU5,FILE=DIR_APP(1:DIR_LEN)//INCAR,STATUS='OLD',IOSTAT=IERR)
      ENDIF

!!! test_
!!      DO I=1,COMM_WORLD%NCPU
!!         IF (COMM_WORLD%NODE_ME==I) THEN
!!            CALL MPI_Get_processor_name(myname,resultlen,ierror)
!!            WRITE(*,'(5(A,I4,X),A,A)') 'world id:',COMM_WORLD%NODE_ME,'inb:',COMM_INB%NODE_ME,'int:',COMM_INTER%NODE_ME, &
!!                                       'intra:',COMM_intra_node%NODE_ME,'inter:',COMM_inter_node%NODE_ME,'cpu:',myname(1:resultlen)
!!         ENDIF
!!         CALL MPI_barrier(COMM_WORLD%MPI_COMM,ierror)
!!      ENDDO
!!! test_

      END SUBROUTINE
#endif
!***********************************************************************
!
! make the  directory entry which is used for
! fileio
!
!***********************************************************************

      SUBROUTINE MAKE_DIR_APP(node)
      INTEGER node

! in principle one can chose here any string one wants to use
! only DIR_LEN must be adjusted
      WRITE (DIR_APP  , "(I1,I1,'/')") MOD(node/10,10),MOD(node,10)
      IF (IMAGES==0.OR.KIMAGES>0) THEN
         DIR_LEN=0
      ELSE
         DIR_LEN=3
      ENDIF

      END SUBROUTINE MAKE_DIR_APP

!***********************************************************************
!
! once unit 6 is open write number of nodes and
! all other parameters
!
!
!***********************************************************************
      SUBROUTINE WRT_DISTR(IU6)
      INTEGER IU6
#ifdef MPI
      IF (IU6>=0) THEN
        WRITE(IU6,'(" running on ",I4," total cores")') COMM_WORLD%NCPU
        IF (IMAGES>0 ) &
        WRITE(IU6,'(" each image running on ",I4," cores")') COMM%NCPU
        WRITE(IU6,'(" distrk:  each k-point on ",I4," cores, ",I4," groups")') &
                        COMM_KIN%NCPU,COMM_KINTER%NCPU
        WRITE(IU6,'(" distr:  one band on NCORES_PER_BAND=",I4," cores, ",I4," groups")') &
                        COMM_INB%NCPU,COMM_INTER%NCPU

      ENDIF
#else
      WRITE(IU6,*)'serial version'
#endif
      END SUBROUTINE WRT_DISTR

      END MODULE
