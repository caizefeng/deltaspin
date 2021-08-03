#include "symbol.inc"

MODULE bse_te

!**********************************************************************
!
! This module solves the BSE equation using 
! the time-evolution method and iterative matrix diagonalization schemes
! developed in  Friedhelm Bechstedts group in Jena
! (Patrik Hahn, Frank Fuchs, Claudia Roedl)
! Some cleanup done by jF
! Gamma point version and further clean up by Tobias Sander
!
! TODO some references here
!
!**********************************************************************

  USE prec
  USE constant
  USE wave_high

  IMPLICIT NONE

  INTEGER,SAVE :: FIRST_ROW_INDEX=0
  INTEGER,SAVE :: NCV_LOCAL
!
! the flag single_prec_bse allows to select single precision storage of BSE 
! matrix and thus saves a factor 2 in storage
! this flag must be set identical to the bse.F routine otherwise the code will crash
!
#ifndef DOuble_prec_bse
#define single_prec_bse
#endif
!
! if check_index is defined a lot of internal consistency checks are
! performed. This might affect performance somewhat 
! (but the impact is not expected to be huge)
!
#define check_index

#ifdef single_prec_bse
  INTEGER, PARAMETER :: mq =SELECTED_REAL_KIND(5)
  GDEFS, ALLOCATABLE :: BSEMATRIX(:,:)
  GDEFS, ALLOCATABLE :: OPTMAT(:,:)
  REAL(mq), ALLOCATABLE :: LDAGW(:)

#define GDOTC CDOTC
#define GGEMV CGEMV
#ifndef gammareal
#define GGEMM CGEMM
#else 
#define GGEMM SGEMM
#endif
#define GGEEV CGEEV
#define GHEMV CHEMV
#define GHEMM CHEMM
#define GHEEV CHEEV
#define GHEEV CHEEV
#define GHEEVX CHEEVX
#define GNAUPD CNAUPD
#define GNEUPD CNEUPD
#define GAXPY CAXPY
#define DGNRM2 SCNRM2
#define MMPI_C MPI_COMPLEX
#define MMPI_R MPI_REAL


#else
  INTEGER, PARAMETER :: mq =SELECTED_REAL_KIND(10)
  GDEF, ALLOCATABLE :: BSEMATRIX(:,:)
  GDEF, ALLOCATABLE :: OPTMAT(:,:)
  REAL(mq), ALLOCATABLE :: LDAGW(:)

#define GDOTC ZDOTC
#define GGEMV ZGEMV
#define GGEMM ZGEMM
#define GGEEV ZGEEV
#define GHEMV ZHEMV
#define GHEMM ZHEMM
#define GHEEV ZHEEV
#define GHEEVX ZHEEVX
#define GNAUPD ZNAUPD
#define GNEUPD ZNEUPD
#define GAXPY ZAXPY
#define DGNRM2 DZNRM2
#define MMPI_C MPI_DOUBLE_COMPLEX
#define MMPI_R MPI_DOUBLE_PRECISION


#endif

  REAL, PRIVATE :: MAGIC=-123456


#ifdef MPI

#define mpi_include     include "mpif.h"
#define infwrite        IF (node_me==node_lead) write(*,*)
#define infwritef       IF (node_me==node_lead) write

#else

#define mpi_include
#define infwrite        write(*,*)
#define infwritef       write

#endif

!------------------------------------------------------------------------
! Variables for the BSE stuff - no counters please!!
      INTEGER, parameter :: nbfield=100
! Some parameters read from input file.
! Control flags.
      INTEGER,PRIVATE  :: EXCITON=1            ! (1) BSE, (0) IP
      INTEGER,PRIVATE  :: BEYONDTD=0           ! (1) going beyond, (0) sticking with Tamm-Dancoff approximation
      INTEGER,PRIVATE  :: ISPIN                ! Number of spin components (for spin flips this is one) ... just for Claudia :))
      INTEGER,PRIVATE  :: MAXK                 ! Number of k-points.
      REAL(mq),PRIVATE :: ENCUT_SP=50._mq      ! Energy cutoff to be used for the calculation of the spectra.
! Spectra.
      INTEGER,PRIVATE  :: XYZINFO=0            ! Directions for spectra: (1) x, (2) y, (3) z, (0) all
      REAL(mq),PRIVATE :: CELLVOL              ! Unit cell volume in Bohr^3.
      REAL(mq),PRIVATE :: GAMMA                ! Broadening for the spectra in eV.
!gK: output should go to vasprun.xml, spectra_file can be removed
      character(255),PRIVATE :: SPECTRA_FILE='spectra'   ! File name for calculated spectra.
      INTEGER,PRIVATE  :: NTDSTEPS=0           ! Maximum number of time steps for the TD.
      INTEGER,PRIVATE  :: LAST_TDSTEP          ! Number of the last actual time step in the TD.
      REAL(mq),PRIVATE :: EVMAX                ! Upper bound for eigenvalues in TD. Usually 1.5*encut_sp.
      INTEGER,PRIVATE  :: MATDIM               ! Matrix Dimension Used For The Calculation.
      INTEGER,PRIVATE, ALLOCATABLE  :: MATDIM_NODE(:)   ! Number Of Rows At Each Node.
      INTEGER,PRIVATE, ALLOCATABLE  :: MATSTART(:)      ! Matrix Element After Which Matrix At Each Node Begins.
      INTEGER,PRIVATE  :: MATDIM_BTD           ! Matrix Dimension Used For The Calculation In Beyond Tamm-Dancoff Mode.
! Some Arrays.
      REAL(mq),PRIVATE, ALLOCATABLE    :: KPTWEIGHT(:)    ! kptweight(Matdim) -> Respective K-Point Weight For Each Transition
      REAL(mq),PRIVATE, ALLOCATABLE    :: TIMESTEP_V(:,:) ! timestep_v(2,ntdsteps) -> Time Steps And Cumulated Time Steps For Td
!CR    COMPLEX(mq),private, allocatable :: scalar(:,:)   ! Scalar(ntdsteps) -> Scalar Product For Each Time Step (Main Quantity In Td) 
      COMPLEX(mq),PRIVATE, ALLOCATABLE  :: SCALAR(:,:) !TSa->CR
      REAL(mq),PRIVATE, PARAMETER     :: HARTREE  = 2._mq*REAL(Rytoev,mq) !27.2112_mq
      COMPLEX(mq),PRIVATE, PARAMETER  :: IMUN     = (0.0_mq,1.0_mq)
      COMPLEX(mq),PRIVATE, PARAMETER  :: INV_IMUN = (0.0_mq,-1.0_mq)
      COMPLEX(mq),PRIVATE, PARAMETER  :: CONE     = (1.0_mq,0.0_mq)
      COMPLEX(mq),PRIVATE, PARAMETER  :: CZERO    = (0.0_mq,0.0_mq)
#ifdef gammareal
      REAL(mq),PRIVATE,PARAMETER      :: RONE     =  1.0_mq
      REAL(mq),PRIVATE,PARAMETER      :: RZERO    =  0.0_mq
#endif

! Mpi 
!Aus Vasp Holen.
      INTEGER,PRIVATE :: NODE_LEAD=0 ! Number Of The Leading Process.
      INTEGER,PRIVATE :: NODE_ME     ! Number Of The My Process.
      INTEGER,PRIVATE :: NODES       ! Total Number Of Processes.
      INTEGER,PRIVATE :: MY_MPI_COMM
    CONTAINS


!****************** SUBROUTINE TW4O_STORE_STRIP_PREPARE    ************
!
! this subroutine stores the two electron 4 orbital integrals
! in a fashion that is suitable for interfacing the code
! to the time-evolution and iterative matrix diagonalization schemes
! developed in  Friedhelm Bechstedt group
! (Patrik Hahn, Frank Fuchs, Claudia Roedl)
!
! The data distribution is fairly simple. Stripes of the matrix
! are stored on each local node
!
! STRIP_SIMPLE
! the simple (first) version relies on this simple relationship
! and therefore does not require any communication overhead
! however, data distribution might not be optimal if the number
! of k-points is not significantly larger than the number of nodes
! 
!**********************************************************************

  SUBROUTINE TW4O_STORE_STRIP_PREPARE(WHF, IO, NCV,  LKPOINT_PARALLEL, NKPTS )
    USE base
    USE ini
    TYPE (wavespin)      WHF
    TYPE (in_struct)     IO
    INTEGER :: NCV              ! total number of pair states
    LOGICAL :: LKPOINT_PARALLEL ! parallelization over k-points
    INTEGER :: NKPTS

    INTEGER :: I

    ! number of local data is given by:  (total+nodes-node_id)/nodes
    NCV_LOCAL=(NCV+WHF%WDES%NB_PAR-WHF%WDES%NB_LOW)/WHF%WDES%NB_PAR

    I=NCV_LOCAL
    
    CALLMPI( M_sum_i(WHF%WDES%COMM, I, 1 ))
    IF (I /= NCV) THEN
       WRITE(0,*) 'internal error in TW4O_STORE_STRIP_SIMPLE_PREPARE: dimension wrong',I, NCV, NCV_LOCAL
       STOP
    ENDIF

!done allocate matdim_node here
    NODE_ME=WHF%WDES%NB_LOW-1
    NODES=WHF%WDES%NB_PAR
    ALLOCATE(MATDIM_NODE(0:NODES-1))
    ALLOCATE(MATSTART(0:NODES))

    MATDIM_NODE=0
    DO I=0,NODES-1
       IF (I==NODE_ME) MATDIM_NODE(I)=NCV_LOCAL
    ENDDO
    CALLMPI( M_sum_i(WHF%WDES%COMM, MATDIM_NODE, WHF%WDES%NB_PAR ))
    MATSTART(0)=0
    DO I=1,NODES
       MATSTART(I)=MATSTART(I-1)+MATDIM_NODE(I-1)
    ENDDO

    ! calculate the base (first index) on each node (one based as opposed to matstart)
    FIRST_ROW_INDEX=1
    DO I=1,WHF%WDES%NB_PAR
       IF (I==WHF%WDES%NB_LOW) EXIT
       ! add number of data on node I
       FIRST_ROW_INDEX=FIRST_ROW_INDEX+(NCV+WHF%WDES%NB_PAR-I)/WHF%WDES%NB_PAR
    ENDDO
    IF (FIRST_ROW_INDEX/=MATSTART(NODE_ME)+1) THEN
       WRITE(*,*) 'internal error in TW4O_STORE_STRIP_PREPARE: FIRST_ROW_INDEX not consistent with matstart'
       STOP
    ENDIF

#ifdef MPI
    MY_MPI_COMM=WHF%WDES%COMM%MPI_COMM
#endif

#ifdef single_prec_bse
    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE TE single prec attempting allocation of',1.0_q/1E9*NCV_LOCAL*NCV*4*ndata ,' Gbyte  rank=',NCV
    IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE TE single prec attempting allocation of',1.0_q/1E9*NCV_LOCAL*NCV*4*ndata ,' Gbyte  rank=',NCV
    CALL REGISTER_ALLOCATE(4._q*ndata* NCV_LOCAL*NCV , "bse")
#else
    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE attempting allocation of',1.0_q/1E9*NCV_LOCAL*NCV*8*ndata,' Gbyte  rank=',NCV
    IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE attempting allocation of',1.0_q/1E9*NCV_LOCAL*NCV*8*ndata,' Gbyte  rank=',NCV
    CALL REGISTER_ALLOCATE(8._q*ndata* NCV_LOCAL*NCV , "bse")
#endif

    ALLOCATE(BSEMATRIX(NCV_LOCAL, NCV))
    ALLOCATE(OPTMAT(NCV, 3),LDAGW(NCV))
    BSEMATRIX=MAGIC

  END SUBROUTINE TW4O_STORE_STRIP_PREPARE


!****************** SUBROUTINE TW4O_CHECK_STRIP ***********************
!
! small routine to check whether all matrix elements have
! been properly set
! this is done by comparing to the MAGIC number
! to which BSEMATRIX was initialised
!
!**********************************************************************

  SUBROUTINE TW4O_CHECK_STRIP(WHF)
    TYPE (wavespin) WHF
    INTEGER I,J
    
     DO I=1,SIZE(BSEMATRIX,1)
       DO J=1,SIZE(BSEMATRIX,2)
          IF (BSEMATRIX(I,J)==MAGIC) THEN
             WRITE(*,*) 'internal error in TW4O_CHECK_STRIP: BSEMATRIX was not properly set',I,J,WHF%WDES%NB_LOW,BSEMATRIX(I,J)
             STOP
          ENDIF
       ENDDO
    ENDDO
  END SUBROUTINE TW4O_CHECK_STRIP

!**********************************************************************
!
! the remaining routines have been added by Claudia Roedl
!
!**********************************************************************

      SUBROUTINE CALCULATE_BSE_TIME_EVOLUTION(SHIFT,OMEGA,ISPIN_IN,MAXK_IN,NCV,KPTWEIGHT_IN, ENCUT_SP_IN, LEXCITON, LBEYONDTD, NTDSTEPS_IN, IO)
      USE base
      USE constant
      IMPLICIT NONE
      INTEGER :: ISPIN_IN,MAXK_IN,NCV
      REAL(q) :: SHIFT
      REAL(q) :: OMEGA
      REAL(q) :: KPTWEIGHT_IN(NCV)
      REAL(q) :: ENCUT_SP_IN
      logical :: LEXCITON
      logical :: LBEYONDTD
      INTEGER :: NTDSTEPS_IN
      TYPE (in_struct)     IO

      IF (LEXCITON) THEN
         EXCITON=1
      ELSE
         EXCITON=0
      ENDIF
! gK test: exciton=0 is not yet tested 
! should make the code faster for independent particle calculations
      EXCITON=1

      IF (LBEYONDTD) then
         BEYONDTD=1
      ELSE
         BEYONDTD=0
      ENDIF
     
      ENCUT_SP=ENCUT_SP_IN

! default fot ntdsteps
      NTDSTEPS=20000
      IF (NTDSTEPS_IN>500) THEN
         NTDSTEPS=NTDSTEPS_IN
      ENDIF

      GAMMA=SHIFT
      CELLVOL=OMEGA/AUTOA**3
      ISPIN=ISPIN_IN
      MAXK=MAXK_IN

      MATDIM=NCV

      ALLOCATE(KPTWEIGHT(MATDIM))
!Anpassen fuer variable Gewichte.
      KPTWEIGHT(:)=KPTWEIGHT_IN(1)

      SPECTRA_FILE=TRIM(ADJUSTL(SPECTRA_FILE))

!      print*,ldagw(:)

      CALL CALC_AND_WRITE_SPECTRA(IO)

      END SUBROUTINE CALCULATE_BSE_TIME_EVOLUTION


!     ******************************************************************
!     These are values for the iteration scheme. The timesteps have to 
!     be smaller than the inverse max eigenvalue, otherwise it is not
!     converging. These values are a good choice, with good speed and
!     good convergence, so best is to leave them that way. However, if
!     you are encountering problems increase the values of evmax and tmax,
!     if the calculation is too time-consuming, try to decrease them.
!     (safer than to decrease tmax is to increase gamma!!)

!     Calculate the timesteps for the time-development scheme.

      SUBROUTINE CALC_TIMESTEPS(BROADENING,IO)
      USE base

      IMPLICIT NONE
      REAL(mq)  :: BROADENING
! local 
      TYPE (IN_STRUCT)     IO
      INTEGER   :: STEP
      REAL(mq)  :: TMAX,TIMESTEP,SUMTIME

      EVMAX=1.5*ENCUT_SP

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,'(1x,(a))') 'Timesteps:'
         WRITE(IO%IU0,*) EVMAX,BROADENING,NTDSTEPS
      ENDIF
      IF (IO%IU6>=0) THEN
         WRITE(IO%IU6,'(1x,A,F13.6)') 'EVMAX:',EVMAX
         WRITE(IO%IU6,'(1x,A,F8.4)') 'BROADENING:',BROADENING
         WRITE(IO%IU6,'(1x,A,I8)') 'TIME STEPS (TOTAL):',NTDSTEPS
         WRITE(IO%IU6,*)
      ENDIF

      TMAX=5._mq/BROADENING
!     TIMESTEP=1._mq/(EVMAX*60._mq)
      SUMTIME=0._mq

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,*) tmax,timestep,sumtime
      ENDIF

      DO STEP=1,NTDSTEPS
          TIMESTEP=(1._mq/(EVMAX*60._mq))*(60._mq-59._mq*exp(-BROADENING*SUMTIME/6._mq))
!         timestep=(1._mq/(EVMAX*60._mq))
         SUMTIME=SUMTIME+TIMESTEP
         TIMESTEP_V(1,STEP)=TIMESTEP
         TIMESTEP_V(2,STEP)=SUMTIME
    
         IF (MOD(STEP,100)==0) THEN
            IF (IO%IU0>=0) THEN
!               WRITE(IO%IU0,'(I6,2(1x,E13.6))') step,sumtime,timestep
            ENDIF
         ENDIF

         IF (SUMTIME.GE.TMAX) THEN
            LAST_TDSTEP=STEP
            EXIT
         ENDIF
      ENDDO !ENDDO step_loop        
 
      IF (SUMTIME.LT.TMAX) THEN
         IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,'(1x,(A))') 'Too many timesteps. - I bet there is something wrong.'
            WRITE(IO%IU0,'(1x,(A))') 'Calculating anyway ...'
         ENDIF
            LAST_TDSTEP=NTDSTEPS
      ENDIF

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,'(1x,(A),I5)') 'Number of time steps: ',LAST_TDSTEP
         WRITE(IO%IU0,'(1x,(A),E13.6,(A))') 'First time step: ',TIMESTEP_V(1,1),'1/eV'
         WRITE(IO%IU0,'(1x,(A),E13.6,(A))') 'Last  time step: ',TIMESTEP_V(1,LAST_TDSTEP),'1/eV'
         WRITE(IO%IU0,*)
      ENDIF
      IF (IO%IU6>=0) THEN
         WRITE(IO%IU6,'(1x,(A),I5)') 'Number of time steps: ',LAST_TDSTEP
         WRITE(IO%IU6,*)
         WRITE(IO%IU6,'(1x,(A),E13.6,(A))') 'First time step: ',TIMESTEP_V(1,1),'1/eV'
         WRITE(IO%IU6,'(1x,(A),E13.6,(A))') 'Last  time step: ',TIMESTEP_V(1,LAST_TDSTEP),'1/eV'
         WRITE(IO%IU6,*)
      ENDIF
 
      END SUBROUTINE CALC_TIMESTEPS

!===================================================================
!     Routine performing the time development algorithm for BSE:
!     We have to solve: i*hbar dv(t)/dt = H v(t).
!     Because of the leap-frog scheme used here we need two initial
!     vectors. So the second one is calculated using a double timestep.
!===================================================================

      SUBROUTINE TIME_DEVELOP_EXCI(DIR_MIN,DIR_MAX_PROJ,DIM,IO)
      USE base

      implicit none
      TYPE (in_struct)        :: IO 
      INTEGER                 :: DIR_MIN,DIR_MAX_PROJ,DIM
! local
      INTEGER                 :: IDIR,KDIR,LDIR,I,J,K 
      INTEGER                 :: DIRS, DIR_MAX
      REAL(mq)                :: Z1,Z2,ZZ
      COMPLEX(mq),ALLOCATABLE :: VECTOR_0(:,:)         ! initial vector (usually optical matrix elements)
      COMPLEX(mq),ALLOCATABLE :: VECTORELEMENTS(:,:)  ! holds the local block of v = bsematrix v
      COMPLEX(mq),ALLOCATABLE :: VECTOR_D(:,:)         ! on root node merged vector v= bsematrix v
      COMPLEX(mq),ALLOCATABLE :: VECTOR_1(:,:),VECTOR_2(:,:),VECTOR_3(:,:)
#ifdef gammareal
      REAL(mq),ALLOCATABLE    :: VECTOR1_REAL(:,:), VECTOR2_REAL(:,:)
      REAL(mq),ALLOCATABLE    :: VECTOR0_REAL(:,:)
      REAL(mq),ALLOCATABLE    :: VECTORELEMENTS_REAL(:,:)
#endif
! functions
      COMPLEX(mq)             :: GDOTC

      mpi_include
      INTEGER                 :: IERR,SENDER
#ifdef MPI
      INTEGER                 :: STATUS(MPI_STATUS_SIZE)
#endif

!=================================
      DIR_MAX=MIN(DIR_MAX_PROJ,3)
      DIRS=DIR_MAX-DIR_MIN+1  
!=================================
      ALLOCATE(VECTOR_0(DIM,DIR_MIN:DIR_MAX))
      ALLOCATE(VECTOR_D(DIM,DIR_MIN:DIR_MAX))
      ALLOCATE(VECTORELEMENTS(MATDIM_NODE(NODE_ME),DIR_MIN:DIR_MAX))
      ALLOCATE(VECTOR_1(DIM,DIR_MIN:DIR_MAX),VECTOR_2(DIM,DIR_MIN:DIR_MAX),VECTOR_3(DIM,DIR_MIN:DIR_MAX))
#ifdef gammareal
      ALLOCATE(VECTOR1_REAL(DIM,DIR_MIN:DIR_MAX+DIRS),VECTOR2_REAL(DIM,DIR_MIN:DIR_MAX+DIRS))
      ALLOCATE(VECTOR0_REAL(DIM,DIR_MIN:DIR_MAX))
      ALLOCATE(VECTORELEMENTS_REAL(MATDIM_NODE(NODE_ME),DIR_MIN:DIR_MAX+DIRS))
#endif
      SCALAR=CZERO
!===========================================
! starting point for leap frog scheme (t=0)
!===========================================
#ifdef gammareal
      VECTOR0_REAL(1:MATDIM,DIR_MIN:DIR_MAX)=OPTMAT(1:MATDIM,DIR_MIN:DIR_MAX) !optmat is REAL!
      VECTOR_0=CMPLX(OPTMAT(1:MATDIM,DIR_MIN:DIR_MAX))
#else
      VECTOR_0(1:MATDIM,DIR_MIN:DIR_MAX)=OPTMAT(1:MATDIM,DIR_MIN:DIR_MAX) !optmat is CMPLX!

#endif
!      IF (beyondtd==1) vector0(matdim+1:matdim_btd,dir_min:dir_max)=conjg(OPTMAT(1:matdim,dir_min:dir_max))

!#ifndef gammareal
!      IF (beyondtd==1) vector0(matdim+1:matdim_btd,dir_min:dir_max)=-conjg(OPTMAT(1:matdim,dir_min:dir_max))
!#endif

#ifdef gammareal
      VECTOR1_REAL(1:MATDIM,DIR_MIN:DIR_MAX)=VECTOR0_REAL(1:MATDIM,DIR_MIN:DIR_MAX)
      VECTOR1_REAL(1:MATDIM,DIR_MAX+1:DIR_MAX+DIRS)=RZERO
#else
      VECTOR_1=VECTOR_0
#endif

      IF (NODE_ME==NODE_LEAD) CALL CPU_TIME(Z1)

!====================================================================
! Determine second initial vector required for the leap-frog scheme:
! v(2) = v(0) + dt/(i*hbar) H v(0) + [dt/(i*hbar)]^2 H H v(0)
! (This should be something similar to successive approximation.)
!====================================================================

! vectorelements(stripes)=bsematrix(stripes) x vector1(full)
!#define bse_te_DGEMV

      IF (EXCITON==1) then
#ifdef bse_te_DGEMV
         DO IDIR=DIR_MIN,DIR_MAX
            CALL GGEMV('N',MATDIM_NODE(NODE_ME),DIM,CONE,BSEMATRIX, &
                 MATDIM_NODE(NODE_ME),VECTOR_1(1,IDIR),1,CZERO,VECTORELEMENTS(1,IDIR),1)
         ENDDO
#else
#ifdef gammareal
         ! IF gammareal GGEMM=DGEMM=>SGEMM  
         CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS*2,DIM,RONE, &
              BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR1_REAL,DIM, &
              RZERO,VECTORELEMENTS_REAL,MATDIM_NODE(NODE_ME))
#else
         CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS,DIM,CONE, & 
              BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR_1,DIM, & 
              CZERO,VECTORELEMENTS,MATDIM_NODE(NODE_ME))
#endif
#endif
      ELSEIF (EXCITON==0) THEN
         DO IDIR=DIR_MIN,DIR_MAX
            VECTORELEMENTS(:,IDIR)=LDAGW((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1))) &
                          *VECTOR_1((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1)),IDIR)
         ENDDO
      ENDIF !exciton==
!==================================================
!rebuild vectorelements & vector1 to complexe-type
!==================================================
#ifdef gammareal
      DO IDIR=DIR_MIN,DIR_MAX
         DO I=1,MATDIM_NODE(NODE_ME)
            VECTORELEMENTS(I,IDIR)=CMPLX(VECTORELEMENTS_REAL(I,IDIR),VECTORELEMENTS_REAL(I,IDIR+DIRS),mq)
         ENDDO
      ENDDO

      DO IDIR=DIR_MIN,DIR_MAX
         DO I=1,DIM
            VECTOR_1(I,IDIR)=CMPLX(VECTOR1_REAL(I,IDIR),VECTOR1_REAL(I,IDIR+DIRS),mq)
         ENDDO
      ENDDO
#endif
!===================================
! merge vectorelements into vectord
!===================================
      DO IDIR=DIR_MIN,DIR_MAX
         CALLMPI(MPI_GATHERV(VECTORELEMENTS(1,IDIR), MATDIM_NODE(NODE_ME), MMPI_C, VECTOR_D(1,IDIR), MATDIM_NODE, MATSTART, MMPI_C, NODE_LEAD, MY_MPI_COMM, IERR))
      ENDDO

      IF (NODE_ME==NODE_LEAD) THEN
         VECTOR_2=VECTOR_1+INV_IMUN*TIMESTEP_V(1,1)*VECTOR_D
         CALL CPU_TIME(ZZ)
         infwritef(*,'(1x,(a),f13.6,(a))') 'Time for vector 1: ',ZZ-Z1,' sec'
      ENDIF
!==================================================
! bcast vector1 to vectord (used in the next step)
!==================================================
      IF (NODE_ME==NODE_LEAD) VECTOR_1=VECTOR_D

      DO IDIR=DIR_MIN,DIR_MAX
         CALLMPI(MPI_BCAST(VECTOR_1(1,IDIR),DIM,MMPI_C,NODE_LEAD,MY_MPI_COMM,IERR))
      ENDDO

      IF (EXCITON==1) THEN
#ifdef bse_te_DGEMV
         DO IDIR=DIR_MIN,DIR_MAX
            CALL GGEMV('N',MATDIM_NODE(NODE_ME),DIM,CONE,BSEMATRIX, &
                    MATDIM_NODE(NODE_ME),VECTOR_1(1,IDIR),1,CZERO,VECTORELEMENTS(1,IDIR),1)
         ENDDO
#else
#ifdef gammareal
!======================================================================================================
! decompose CMPLX-type vector1 into REAL-type vector1_REAL for matrix-multiplic. "bsematrix x vector1" 
!======================================================================================================
         DO IDIR=DIR_MIN,DIR_MAX
               VECTOR1_REAL(:,IDIR)=REAL(VECTOR_1(:,IDIR),mq)
               VECTOR1_REAL(:,IDIR+DIRS)=AIMAG(VECTOR_1(:,IDIR))
         ENDDO
     
         CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS*2,DIM,RONE, &
              BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR1_REAL,DIM, &
              RZERO,VECTORELEMENTS_REAL,MATDIM_NODE(NODE_ME))
#else
         CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS,DIM,CONE, & 
              BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR_1,DIM, & 
              CZERO,VECTORELEMENTS,MATDIM_NODE(NODE_ME))
#endif
#endif
      ELSEIF (EXCITON==0) THEN
         DO IDIR=DIR_MIN,DIR_MAX
            VECTORELEMENTS(:,IDIR)=LDAGW((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1))) &
                          *VECTOR_1((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1)),IDIR)
         ENDDO
      ENDIF !exciton==1
!=====================================
! rebuild complex-type vectorelements 
!=====================================
#ifdef gammareal
      DO IDIR=DIR_MIN,DIR_MAX
         DO I=1,MATDIM_NODE(NODE_ME)
            VECTORELEMENTS(I,IDIR)=CMPLX(VECTORELEMENTS_REAL(I,IDIR),VECTORELEMENTS_REAL(I,IDIR+DIRS),mq)
         ENDDO
      ENDDO
#endif

      DO IDIR=DIR_MIN,DIR_MAX
         CALLMPI(MPI_GATHERV(VECTORELEMENTS(1,IDIR), MATDIM_NODE(NODE_ME), MMPI_C, VECTOR_D(1,IDIR), MATDIM_NODE, MATSTART, MMPI_C, NODE_LEAD, MY_MPI_COMM, IERR))
      ENDDO

      IF (NODE_ME==NODE_LEAD) THEN
         VECTOR_2=VECTOR_2+0.5_mq*(INV_IMUN*TIMESTEP_V(1,1))**2*VECTOR_D
         CALL CPU_TIME(Z2)
         infwritef(*,'(1x,(a),f13.6,(a))') 'Time for vector 2: ',z2-zz,' sec'
         infwritef(*,'(1x,(a),f13.2,(a))') 'The full time propagation will take approx. ',(z2-z1)*REAL(LAST_TDSTEP,mq)/2._mq,' sec' 
         infwrite
      ENDIF
!========================================================================
!     The calculation of all timesteps follows.
!     Solving the Schroedinger-like equation using the method of central
!     differences: i*hbar (v(i+2)-v(i))/(2*dt) = H v(i+1).
!     This yields: v(i+2) = v(i) + 2*dt/(i*hbar) H v(i+1).
!     We need scalar(i) = <v(0)|v(i)>.
!========================================================================

      IF (NODE_ME==NODE_LEAD) THEN
         CALL CPU_TIME(Z1)
#ifdef gammareal
         VECTOR_1=CMPLX(VECTOR0_REAL) ! why not vector1=vector0 since vector0=CMPLX(optmat)!
#else
         VECTOR_1=VECTOR_0
#endif
      ENDIF
     
      DO I=2,LAST_TDSTEP ! count timesteps
         ! bcast present wavefunction on which bsematrix is applied
         DO IDIR=DIR_MIN,DIR_MAX
            CALLMPI(MPI_BCAST(VECTOR_2(1,IDIR),DIM,MMPI_C,NODE_LEAD,MY_MPI_COMM,IERR))
         ENDDO
         IF (EXCITON==1) THEN
#ifdef bse_te_DGEMV
            DO IDIR=DIR_MIN,DIR_MAX
               CALL GGEMV('N',MATDIM_NODE(NODE_ME),DIM,CONE,BSEMATRIX, &
                       MATDIM_NODE(NODE_ME),VECTOR_2(1,IDIR),1,CZERO,VECTORELEMENTS(1,IDIR),1)
            ENDDO
#else
!======================================================================================================
! decompose CMPLX-type vector1 into REAL-type vector1_REAL for matrix-multiplic. "bsematrix x vector2" 
!======================================================================================================
#ifdef gammareal
            DO IDIR=DIR_MIN,DIR_MAX
                  VECTOR2_REAL(:,IDIR)=REAL(VECTOR_2(:,IDIR),mq)
                  VECTOR2_REAL(:,IDIR+DIRS)=AIMAG(VECTOR_2(:,IDIR))
            ENDDO

            CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS*2,DIM,RONE, &
                 BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR2_REAL,DIM, &
                 RZERO,VECTORELEMENTS_REAL,MATDIM_NODE(NODE_ME))
#else
            CALL GGEMM('N','N', MATDIM_NODE(NODE_ME),DIRS,DIM,CONE, & 
                 BSEMATRIX,MATDIM_NODE(NODE_ME),VECTOR_2,DIM, & 
                 CZERO,VECTORELEMENTS,MATDIM_NODE(NODE_ME))
#endif
#endif
         ELSEIF (EXCITON==0) THEN
            DO IDIR=DIR_MIN,DIR_MAX
               VECTORELEMENTS(:,IDIR)=LDAGW((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1))) &
                             *VECTOR_2((MATSTART(NODE_ME)+1):(MATSTART(NODE_ME+1)),IDIR)
            ENDDO
         ENDIF !exciton==
!====================================
!rebuild complex type vectorelements
!====================================
#ifdef gammareal
         DO IDIR=DIR_MIN,DIR_MAX
            VECTORELEMENTS(:,IDIR)=CMPLX(VECTORELEMENTS_REAL(:,IDIR),VECTORELEMENTS_REAL(:,IDIR+DIRS),mq)
         ENDDO
#endif
         DO IDIR=DIR_MIN,DIR_MAX
            CALLMPI(MPI_GATHERV(VECTORELEMENTS(1,IDIR), MATDIM_NODE(NODE_ME), MMPI_C, VECTOR_D(1,IDIR), MATDIM_NODE, MATSTART, MMPI_C, NODE_LEAD, MY_MPI_COMM, IERR))
         ENDDO

!CR------------------------------------------------------------------------------------------------------------------
!         IF (node_me==node_lead) THEN
!            IF (beyondtd==0) THEN
!               DO IDIR=dir_min,dir_max
!                     scalar(i-1,IDIR)=GDOTC(dim,vector0(1,IDIR),1,vector1(1,IDIR),1)
!               ENDDO
!            ELSEIF (beyondtd==1) THEN
!               DO IDIR=dir_min,dir_max
!                  scalar(i-1,IDIR)=GDOTC(matdim,vector0(1,IDIR),1,vector1(1,IDIR),1) &
!                       +GDOTC(matdim,-vector0(matdim+1,IDIR),1,vector1(matdim+1,IDIR),1)
!               ENDDO
!            ENDIF
!CR-------------------------------------------------------------------------------------------------------------------
         IF (NODE_ME==NODE_LEAD) THEN
            IF (BEYONDTD==0) THEN
               DO IDIR=DIR_MIN,DIR_MAX_PROJ
                  IF (IDIR<4) THEN
                     KDIR=IDIR
                     LDIR=IDIR
                  ELSEIF (IDIR==4) THEN
                     KDIR=1
                     LDIR=2
                  ELSEIF (IDIR==5) THEN 
                     KDIR=2
                     LDIR=3
                  ELSEIF (IDIR==6) THEN
                     KDIR=3
                     LDIR=1
                  ENDIF
                  SCALAR(I-1,IDIR)=GDOTC(DIM,VECTOR_0(1,KDIR),1,VECTOR_1(1,LDIR),1)
               ENDDO
            ELSEIF (BEYONDTD==1) THEN
               DO IDIR=DIR_MIN,DIR_MAX_PROJ
                  IF (IDIR<4) THEN
                     KDIR=IDIR
                     LDIR=IDIR
                  ELSEIF (IDIR==4) THEN
                     KDIR=1
                     LDIR=2
                  ELSEIF (IDIR==5) THEN
                     KDIR=2
                     LDIR=3
                  ELSEIF (IDIR==6) THEN
                     KDIR=1
                     LDIR=3
                  ENDIF
                  SCALAR(I-1,IDIR)=GDOTC(DIM,VECTOR_0(1,KDIR),1,VECTOR_1(1,LDIR),1) &
                        -CONJG(GDOTC(DIM,VECTOR_0(1,KDIR)*(-1._mq,0._mq),1,VECTOR_1(1,LDIR),1)) !NaN
               ENDDO
            ENDIF !ENDIF  beyondtd
WRITE(300,*) I-1,REAL(SCALAR(I-1,1)),AIMAG(SCALAR(I-1,1))
!============================
! propagate vector in time !
!============================
            VECTOR_3=VECTOR_1+2._mq*INV_IMUN*TIMESTEP_V(1,I)*VECTOR_D

!CR            if (abs(scalar(i-1),idir).gt.1000._mq) then
!CR               infwritef(*,'(1x,(a))') 'Scalar product is diverging.'
!CR               infwrite
!CR               goto 100
!CR            endif
            VECTOR_1=VECTOR_2
            VECTOR_2=VECTOR_3
!TEST
            write(78,'(3F14.7)') timestep_v(2,i),VECTOR_1(3,1) !anc
!TEST
            IF (MOD(I,100)==0) THEN
               CALL CPU_TIME(Z2)
!              infwritef(*,'(1x,(a),i5,(a),f13.2,(a))') 'Step ',i,' done: ',z2 - z1,' sec'
            ENDIF
         ENDIF !ENDIF node_me=node_lead
      ENDDO !ENDDO time_step(i)
!CR--------------------------------------------------------------------------------------------------------------------
!      IF (node_me==node_lead) THEN
!         i=last_tdstep+1
!         ! Account for changing sign in beyond Tamm-Dancoff mode.
!        IF (beyondtd==0) THEN
!            DO IDIR=dir_min,dir_max
!               scalar(i-1,IDIR)=GDOTC(dim,vector0(1,IDIR),1,vector1(1,IDIR),1) !TSa ??
!            ENDDO
!         ELSEIF (beyondtd==1) THEN
!            DO IDIR=dir_min,dir_max
!               scalar(i-1,IDIR)=GDOTC(matdim,vector0(1:matdim,IDIR),1,vector1(1:matdim,IDIR),1) &
!  !                  +GDOTC(matdim,-vector0(matdim+1:matdim_btd,IDIR),1,vector1(matdim+1:matdim_btd,IDIR),1)
! example
! gK             scalar(i-1,IDIR,jdir)=GDOTC(matdim,vector0(1:matdim,IDIR),1,vector1(1:matdim,jdir),1) &
!                    +GDOTC(matdim,-vector0(matdim+1:matdim_btd,IDIR),1,vector1(matdim+1:matdim_btd,JDIR),1)
!
!            ENDDO
!         ENDIF
!         call cpu_time(z2)
!         infwritef(*,'(1x,(a),f13.2,(a))') 'All time steps DOne. Time was ',(z2 - z1)/1._mq,' sec'
!         infwrite
!      ENDIF
!CR--------------------------------------------------------------------------------------------------------------------
     IF (NODE_ME==NODE_LEAD) THEN
         I=LAST_TDSTEP+1
         ! Account for changing sign in beyond Tamm-Dancoff mode.
         IF (BEYONDTD==0) THEN
            DO IDIR=DIR_MIN,DIR_MAX_PROJ
               IF (IDIR<4) THEN
                  KDIR=IDIR
                  LDIR=IDIR
               ELSEIF (IDIR==4) THEN
                  KDIR=1
                  LDIR=2
               ELSEIF (IDIR==5) THEN
                  KDIR=2
                  LDIR=3
               ELSEIF (IDIR==6) THEN
                  KDIR=3
                  LDIR=1
               ENDIF
               SCALAR(I-1,IDIR)=GDOTC(DIM,VECTOR_0(1,KDIR),1,VECTOR_1(1,LDIR),1) 
            ENDDO
         ELSEIF (BEYONDTD==1) THEN
            DO IDIR=DIR_MIN,DIR_MAX
               IF (IDIR<4) THEN
                  KDIR=IDIR
                  LDIR=IDIR
               ELSEIF (IDIR==4) THEN
                  KDIR=1
                  LDIR=2
               ELSEIF (IDIR==5) THEN
                  KDIR=2
                  LDIR=3
               ELSEIF (IDIR==6) THEN
                  KDIR=3
                  LDIR=1
               ENDIF
               SCALAR(I-1,IDIR)=GDOTC(DIM,VECTOR_0(1:MATDIM,KDIR),1,VECTOR_1(1:MATDIM,LDIR),1) &
                       -CONJG(GDOTC(DIM,VECTOR_0(1:MATDIM,KDIR),1,VECTOR_1(1:MATDIM,LDIR),1))
            ENDDO
         ENDIF !beyondtd==
         CALL CPU_TIME(Z2)
         infwritef(*,'(1x,(a),f13.2,(a))') 'All time steps done. Time was ',(Z2 - Z1)/1._mq,' sec'
         infwrite
         IF (IO%IU6>=0) WRITE(IO%IU6,'(1x,A,F13.2,A)') 'CPU time (time steps)',(Z2 - Z1)/1._mq,'sec'
         IF (IO%IU6>=0) WRITE(IO%IU6,*)
      ENDIF

 100  CONTINUE

      DEALLOCATE(VECTOR_1,VECTOR_2,VECTOR_3)
      DEALLOCATE(VECTOR_D)
      DEALLOCATE(VECTORELEMENTS)
      DEALLOCATE(VECTOR_0)
#ifdef gammareal
      DEALLOCATE(VECTOR1_REAL,VECTOR2_REAL)
      DEALLOCATE(VECTORELEMENTS_REAL)
      DEALLOCATE(VECTOR0_REAL)
#endif

      END SUBROUTINE TIME_DEVELOP_EXCI

!     *******************************************************************
!     Spectra.

      SUBROUTINE CALC_AND_WRITE_SPECTRA(IO)
      USE base

      IMPLICIT NONE
      TYPE (IN_STRUCT)     IO
! local
      INTEGER              :: DIR,IDIR,JDIR,DIR_MIN,DIR_MAX,I,J,LAMBDA !TSa
      INTEGER              :: MAXOMEGA
      REAL(mq)             :: NORM,W
      COMPLEX(mq)          :: PROJ,EXPFAC
      COMPLEX(mq), POINTER :: EPSILON(:,:)
!     complex(mq), pointer :: EPSILON(:,:,:) !TSa
      REAL(q),POINTER      :: EPSILON_RE(:,:,:), EPSILON_IM(:,:,:) !TSa
      CHARACTER(255)       :: STR_ENCUT_SP,STR_GAMMA,SPECTRA_FILE_SPEC
      LOGICAL              :: LSPECTRA_FILE

!local
      LOGICAL :: UNITOK,UNITOP

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,'(1x,(a))') 'Calculation of spectra started ...'
         WRITE(IO%IU0,*)
      ENDIF
      IF (IO%IU6>=0) THEN
         write(IO%IU6,'(1x,A)')'TIME EVOLUTION PARAMETER'
         write(IO%IU6,'(A)')'========================='
      ENDIF

      IF (XYZINFO==1) THEN
         DIR_MIN=1
         DIR_MAX=1
      ELSEIF (XYZINFO==2) THEN
         DIR_MIN=2
         DIR_MAX=2
      ELSEIF (XYZINFO==3) THEN
         DIR_MIN=3
         DIR_MAX=3
      ELSEIF (XYZINFO==0) THEN
         DIR_MIN=1
         DIR_MAX=3
      ELSE
         DIR_MIN=1
         DIR_MAX=6
      ENDIF


      ALLOCATE(TIMESTEP_V(2,NTDSTEPS))
      ALLOCATE(SCALAR(NTDSTEPS,DIR_MIN:DIR_MAX))
      CALL CALC_TIMESTEPS(GAMMA,IO)

      MAXOMEGA=int(ENCUT_SP*100+400)
      infwritef(*,'(1x,(A),F6.2)')'OMEGAMAX = ',MAXOMEGA*0.01_mq
      IF (IO%IU6>=0) THEN 
         WRITE(IO%IU6,'(1x,A,1x,I6)') 'NOMEGA  :',MAXOMEGA
         WRITE(IO%IU6,'(1x,A,1x,F8.5)') 'OMEGAMAX:',MAXOMEGA*0.01_mq 
         WRITE(IO%IU6,*)
      ENDIF 
 
      ALLOCATE(EPSILON(0:MAXOMEGA,7))
      ALLOCATE(EPSILON_RE(1:MAXOMEGA+1,3,3)) !TSa
      ALLOCATE(EPSILON_IM(1:MAXOMEGA+1,3,3)) !TSa

      NORM=4._mq*REAL(PI,mq)*(HARTREE**3)/(CELLVOL*REAL(MAXK,kind=mq)) 
      WRITE(*,*)'NORM:',NORM 
      IF (ISPIN==1) NORM=2._mq*NORM

      EPSILON=CZERO
      EPSILON_RE=0.0_mq !TSa
      EPSILON_IM=0.0_mq !TSa

      infwritef(*,'(1x,(a),i1)') 'Direction: ',DIR 
      IF (BEYONDTD==0) THEN
         CALL TIME_DEVELOP_EXCI(DIR_MIN,DIR_MAX,MATDIM,IO) 
      ELSEIF (BEYONDTD==1) THEN
         CALL TIME_DEVELOP_EXCI(DIR_MIN,DIR_MAX,MATDIM,IO)
      ENDIF
!=============
!calculate DF
!=============
      IF (NODE_ME==NODE_LEAD) THEN 
         DO DIR=DIR_MIN,DIR_MAX !dir_max ne dir_max in SR time_develop_exci :):)
            DO I=0,MAXOMEGA
               W=0.01_mq*REAL(I,mq)
               DO J=1,LAST_TDSTEP
                  EXPFAC=exp((IMUN*W-GAMMA)*TIMESTEP_V(2,J)) !expfac=exponential-factor
                  IF (BEYONDTD==0) THEN
                     EPSILON(I,DIR)=EPSILON(I,DIR)-2.0_mq*AIMAG(SCALAR(J,DIR))*EXPFAC*TIMESTEP_V(1,j)
                  ELSEIF (BEYONDTD==1) THEN
                     EPSILON(I,DIR)=EPSILON(I,DIR)-INV_IMUN*SCALAR(J,DIR)*EXPFAC*TIMESTEP_V(1,J)
                  ENDIF
               ENDDO 
               !epsilon(i,dir)=cone+norm*epsilon(i,dir)
               EPSILON(I,DIR)=NORM*EPSILON(I,DIR)
               IF (DIR<4) EPSILON(I,DIR)=CONE+EPSILON(I,DIR) !JF
            ENDDO !ENDDO omega_sum! 
         ENDDO !enddo dir
!==========================
!old ouput to spectra_file
!==========================
         LSPECTRA_FILE=.TRUE.
         IF (LSPECTRA_FILE) THEN
            EPSILON(:,7)=1._mq/3._mq*(EPSILON(:,1)+EPSILON(:,2)+EPSILON(:,3))
            IF (XYZINFO==0) EPSILON(:,4)=EPSILON(:,7)
            WRITE(STR_ENCUT_SP,'(F6.1)') ENCUT_SP
            WRITE(STR_GAMMA,'(F5.2)') GAMMA  
            WRITE(SPECTRA_FILE_SPEC,'(5(A))') TRIM(ADJUSTL(SPECTRA_FILE)), &
              '_encut',TRIM(ADJUSTL(STR_ENCUT_SP)),'_gamma',TRIM(ADJUSTL(STR_GAMMA))
            OPEN(30,FILE=SPECTRA_FILE_SPEC)
            WRITE(30,88)
            DO I=0,MAXOMEGA
               IF (XYZINFO>0) THEN
                  WRITE(30,'(3(1X,E13.6))') 0.01_mq*i,REAL(EPSILON(I,XYZINFO)),AIMAG(EPSILON(I,XYZINFO))
               ELSEIF (XYZINFO==0) THEN
                  WRITE(30,'(1X,F7.4,7(1X,F13.6))') 0.01_mq*i,REAL(EPSILON(I,1)),REAL(EPSILON(I,2)),REAL(EPSILON(I,3)), &
                                                           &  AIMAG(EPSILON(I,1)),AIMAG(EPSILON(I,2)),AIMAG(EPSILON(I,3))
!                  write(30,'(1x,F6.4,8(1x,F10.6))') 0.01_mq*i,(REAL(epsilon(i,dir)),AIMAG(epsilon(i,dir)),dir=1,4)
               ELSE 
                  WRITE(30,'(15(1X,E13.6))') 0.01_mq*i,(REAL(EPSILON(I,DIR)),AIMAG(EPSILON(I,DIR)),DIR=1,7)
               ENDIF
            ENDDO
            CLOSE(30)
         ENDIF
88 FORMAT(/2x,'freq',7x,'Re xx',9x,'Re yy',9x,'Re zz',10x,'Im xx',9x,'Im yy',9x,'Im zz'/&
           &'============================================================================================')

!TEST
          DO I=0,MAXOMEGA
             WRITE(98,'(1x,F7.4,6(1x,F13.6))') 0.01_mq*i, (REAL(EPSILON(I,DIR)),AIMAG(EPSILON(I,DIR)),DIR=1,1)
          ENDDO
!TEST
!================================
!prepare epsilon for vasprun.xml
!================================
         DO DIR=DIR_MIN,DIR_MAX
         IF (DIR<4) THEN ! xx,yy,zz direction
           DO I=0,MAXOMEGA
             EPSILON_RE(I+1,DIR,DIR)=REAL(EPSILON(I,DIR),mq) 
             EPSILON_IM(I+1,DIR,DIR)=AIMAG(EPSILON(I,DIR))
           ENDDO
         ELSEIF (DIR==4) THEN ! xy dir.
           DO I=0,MAXOMEGA
             EPSILON_RE(I+1,1,2)=REAL(EPSILON(I,DIR),mq)
             EPSILON_IM(I+1,1,2)=AIMAG(EPSILON(I,DIR))
             EPSILON_RE(I+1,2,1)=REAL(EPSILON(I,DIR),mq)
             EPSILON_IM(I+1,2,1)=AIMAG(EPSILON(I,DIR))
           ENDDO
         ELSEIF (DIR==5) THEN ! yz dir.
           DO I=0,MAXOMEGA
             EPSILON_RE(I+1,2,3)=REAL(EPSILON(I,DIR),mq)
             EPSILON_IM(I+1,2,3)=AIMAG(EPSILON(I,DIR))
             EPSILON_RE(I+1,3,2)=REAL(EPSILON(I,DIR),mq)
             EPSILON_IM(I+1,3,2)=AIMAG(EPSILON(I,DIR))
           ENDDO
         ELSEIF (DIR==6) THEN ! XZ DIR.
           DO I=1,MAXOMEGA
             EPSILON_RE(I+1,3,1)=REAL(EPSILON(I,DIR),mq)
             EPSILON_IM(I+1,3,1)=AIMAG(EPSILON(I,DIR))
             EPSILON_RE(I+1,1,3)=REAL(EPSILON(I,DIR),mq) 
             EPSILON_IM(I+1,1,3)=AIMAG(EPSILON(I,DIR)) 
           ENDDO
         ENDIF 
         ENDDO

      ENDIF ! node_me

      CALL XML_EPSILON_W(0.01_q, EPSILON_RE, EPSILON_IM, MAXOMEGA+1) 
      ! maxomega+1 due to allocation of epsilon_re/im -> compare XML_EPSILON_W in xml.F!

      IF (LSPECTRA_FILE) THEN
         infwritef(*,'(1x,(a),(a))') 'Spectra written to: ',TRIM(ADJUSTL(SPECTRA_FILE_SPEC))
         infwrite
      ENDIF

      DEALLOCATE(EPSILON)
      DEALLOCATE(EPSILON_RE) !TSa
      DEALLOCATE(EPSILON_IM) !TSa
      DEALLOCATE(TIMESTEP_V)
      DEALLOCATE(SCALAR)
      DEALLOCATE(MATDIM_NODE,MATSTART)

      END SUBROUTINE CALC_AND_WRITE_SPECTRA

END MODULE bse_te
!#endif
