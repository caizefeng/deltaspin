#include "symbol.inc"
!**********************************************************************
! RCS:  $Id: fftmpi_map.F,v 1.4 2002/08/14 13:59:38 kresse Exp $
!======================================================================
!
! these routines set up the communication patterns for the FFT
!
! they are inspired by the corresponding CASTEP routines
! but have been rewritten entirely using different algorithms
! whereever possible (this was not a simple matter in all cases)
!
!======================================================================

!**********************************************************************
! SUBROUTINE MAPSET initialises the communication parts of the
! grid structure
!
!   if LREAL is set mapping is done for a real <-> complex FFT
!   in z-direction
!**********************************************************************

  SUBROUTINE MAPSET(GRID)
    USE prec
    USE mpimy
    USE mgrid
    USE ini
    IMPLICIT NONE

    TYPE (grid_3d), TARGET :: GRID
    TYPE (grid_3d) RECV
    TYPE (communic),POINTER ::  C
    ! local variables
    INTEGER NA1,NA2,NA3,NC,I1,I10,I3,I2,I20,I4,I,ierror,ISND
    INTEGER X,Y,Z,XP,ZP
    INTEGER MAXCOL_X,MAXCOL_Y,MAXCOL_Z,ICOL_SND,ICOL_RCV
    INTEGER RMTSND(8), RMTRCV(8), JPATH,JNODE
    INTEGER NODE_ME,IONODE
    INTEGER, ALLOCATABLE :: RCV_TBL1(:),RCV_TBL2(:)

    ! only if z is the fast index in the FFT the parallel FFT is applicable
    IF (GRID%RL_FFT%NFAST==1) RETURN

    
    C=> GRID%COMM

    NODE_ME=C%NODE_ME
    IONODE =C%IONODE
    !
    ! find maximum number of columns in each representation on all nodes
    !
    MAXCOL_X=GRID%RC%NCOL
    MAXCOL_Y=GRID%IN%NCOL
    MAXCOL_Z=GRID%RL_FFT%NCOL

    CALLMPI( M_max_i(C,MAXCOL_X ,1))
    CALLMPI( M_max_i(C,MAXCOL_Y ,1))
    CALLMPI( M_max_i(C,MAXCOL_Z ,1))
    ALLOCATE(RECV%RC%I2(MAXCOL_X), RECV%RC%I3(MAXCOL_X), &
         RECV%IN%I2(MAXCOL_Y), RECV%IN%I3(MAXCOL_Y), &
         RECV%RL%I2(MAXCOL_Z), RECV%RL%I3(MAXCOL_Z))
    !
    ! allocate the arrays required for redistribution, all arrays with an I
    ! correspond to receiver mapping
    !
    NA1=GRID%RC%NALLOC     ! can send only number    of RC-data  on this node
    NC=C%NCPU+1
    ALLOCATE(GRID%RC_IN%TBL(NA1),  GRID%RC_IN%PTR(NC),  GRID%RC_IN%RMT(NC),  GRID%RC_IN%N(NC), &
         RCV_TBL1(NA1))
    CALL REGISTER_ALLOCATE(8._q*(SIZE(GRID%RC_IN%TBL)+SIZE(GRID%RC_IN%PTR)+SIZE(GRID%RC_IN%RMT)+SIZE(GRID%RC_IN%N)), "fftplans")

    NA2=GRID%IN%NALLOC     ! can receive only number of IN-data
    ALLOCATE(GRID%RC_IN%TBLI(NA2), GRID%RC_IN%PTRI(NC), GRID%RC_IN%RMTI(NC), GRID%RC_IN%NI(NC))

    CALL REGISTER_ALLOCATE(8._q*(SIZE(GRID%RC_IN%TBLI)+SIZE(GRID%RC_IN%PTRI)+SIZE(GRID%RC_IN%RMTI)+SIZE(GRID%RC_IN%NI)) , "fftplans")

    ! can send only number    of RC-data
    GRID%RC_IN%N=0
    GRID%RC_IN%NI=0

    ALLOCATE(GRID%IN_RL%TBL(NA2),  GRID%IN_RL%PTR(NC),  GRID%IN_RL%RMT(NC),  GRID%IN_RL%N(NC), &
         RCV_TBL2(NA2))

    CALL REGISTER_ALLOCATE(8._q*(SIZE(GRID%IN_RL%TBL)+SIZE(GRID%IN_RL%PTR)+SIZE(GRID%IN_RL%RMT)+SIZE(GRID%IN_RL%N)), "fftplans")

    NA3=GRID%RL_FFT%NALLOC     !  can receive onle number of RL-data
    ALLOCATE(GRID%IN_RL%TBLI(NA3), GRID%IN_RL%PTRI(NC), GRID%IN_RL%RMTI(NC), GRID%IN_RL%NI(NC))

    CALL REGISTER_ALLOCATE(8._q*(SIZE(GRID%IN_RL%TBLI)+SIZE(GRID%IN_RL%PTRI)+SIZE(GRID%IN_RL%RMTI)+SIZE(GRID%IN_RL%NI)) , "fftplans")

    GRID%IN_RL%N=0
    GRID%IN_RL%NI=0

    I1 = 0
    I3 = 0
    I2 = 0
    I4 = 0

    node: DO JNODE = 1, C%NCPU
       !
       ! copy the data distribution of node JNODE to all other nodes
       !
       IF (NODE_ME==JNODE) THEN
          !           copy own data to receive = send buffer
          RECV%RC%NCOL = GRID%RC%NCOL
          DO ICOL_SND = 1,GRID%RC%NCOL
             RECV%RC%I2(ICOL_SND) = GRID%RC%I2(ICOL_SND)
             RECV%RC%I3(ICOL_SND) = GRID%RC%I3(ICOL_SND)
          END DO
          RECV%IN%NCOL = GRID%IN%NCOL
          DO ICOL_SND = 1,GRID%IN%NCOL
             RECV%IN%I2(ICOL_SND) = GRID%IN%I2(ICOL_SND)
             RECV%IN%I3(ICOL_SND) = GRID%IN%I3(ICOL_SND)
          END DO
          RECV%RL%NCOL = GRID%RL_FFT%NCOL
          DO ICOL_SND = 1,GRID%RL_FFT%NCOL
             RECV%RL%I2(ICOL_SND) = GRID%RL_FFT%I2(ICOL_SND)
             RECV%RL%I3(ICOL_SND) = GRID%RL_FFT%I3(ICOL_SND)
          END DO
       END IF

       CALL MPI_barrier( C%MPI_COMM, ierror )

       ! and now send it to all nodes
       CALL M_bcast_i_from(C, RECV%RC%NCOL,1, JNODE)
       IF (RECV%RC%NCOL /= 0 ) THEN
         CALL M_bcast_i_from(C, RECV%RC%I2(1),RECV%RC%NCOL, JNODE)
         CALL M_bcast_i_from(C, RECV%RC%I3(1),RECV%RC%NCOL, JNODE)
       ENDIF

       CALL M_bcast_i_from(C, RECV%IN%NCOL,1, JNODE)
       IF (RECV%IN%NCOL /= 0 ) THEN
         CALL M_bcast_i_from(C, RECV%IN%I2(1),RECV%IN%NCOL, JNODE)
         CALL M_bcast_i_from(C, RECV%IN%I3(1),RECV%IN%NCOL, JNODE)
       ENDIF

       CALL M_bcast_i_from(C, RECV%RL%NCOL,1, JNODE)
       IF (RECV%RL%NCOL /= 0 ) THEN
         CALL M_bcast_i_from(C, RECV%RL%I2(1),RECV%RL%NCOL, JNODE)
         CALL M_bcast_i_from(C, RECV%RL%I3(1),RECV%RL%NCOL, JNODE)
       END IF

       ! now each node checks which data must be send from him to JNODE
       !
       ! Calculate RC-> IN (reciprocal -> intermediate) send table
       ! reciprocal:   x is fast index, columns correspond to (y=I2,z=I3)
       ! intermediate: y is fast index, columns correspond to (z=I2,x=I3)
       ! if z index matches send data

       GRID%RC_IN%PTR(JNODE) = I1
       I10 =I1

       DO ICOL_RCV = 1,RECV%IN%NCOL    ! loop over all receiver columns
          X=RECV%IN%I3(ICOL_RCV)
          Z=RECV%IN%I2(ICOL_RCV)
          DO ICOL_SND = 1,GRID%RC%NCOL ! loop over all sender columns
             ! does z index match
             Y =GRID%RC%I2(ICOL_SND)
             ZP=GRID%RC%I3(ICOL_SND)

             IF (Z==ZP) THEN
                I1 = I1 + 1
                ! position of data on local node and remote node
                GRID%RC_IN%TBL(I1) = X + (ICOL_SND-1)*GRID%NGX
                RCV_TBL1(I1)       = (Y-1)*RECV%IN%NCOL + ICOL_RCV
             END IF
          ENDDO
       ENDDO
       GRID%RC_IN%N(JNODE) = I1-I10

       ! sort send table ascendingly
       ! this improves cash coherency
       IF (I1-I10/=0) &
            CALL SORT_REDIS_ASC(I1-I10,GRID%RC_IN%TBL(I10+1),RCV_TBL1(I10+1))

       ! xalculate IN -> RL (intermediate -> real) send table
       ! intermediate: y is fast index, columns correspond to (z=I2,x=I3)
       ! real:         z is fast index, columns correspond to (x=I2,y=I3)
       ! if x index matches send data
       GRID%IN_RL%PTR(JNODE) = I2
       I20=I2


       DO ICOL_RCV = 1,RECV%RL%NCOL
          Y=RECV%RL%I3(ICOL_RCV)
          X=RECV%RL%I2(ICOL_RCV)
          DO ICOL_SND = 1,GRID%IN%NCOL
             XP=GRID%IN%I3(ICOL_SND)
             Z =GRID%IN%I2(ICOL_SND)
             ! does x index match
             IF (X==XP) THEN
                I2 = I2 + 1
                GRID%IN_RL%TBL (I2) = (Y-1)*GRID%IN%NCOL + ICOL_SND
                RCV_TBL2(I2)        = Z + (ICOL_RCV-1)*GRID%NGZ_rd
             END IF
          ENDDO
       ENDDO
       GRID%IN_RL%N(JNODE)= I2 - I20
       IF (I2-I20/=0) &
            CALL SORT_REDIS_ASC(I2-I20,GRID%IN_RL%TBL(I20+1),RCV_TBL2(I20+1))

       DWRITE 'node',NODE_ME,' to',JNODE,' data',I2-I20

    ENDDO node

    ! number of send data must be equal to data on local node
    IF (I1 /=  GRID%RC%NP) THEN
       WRITE(*,*)'internal error 1: MAPSET',I1, GRID%RC%NP
       STOP
    ENDIF
    IF (I2 /=  GRID%IN%NP) THEN
       WRITE(*,*)'internal error 2: MAPSET',I2, GRID%IN%NP
       STOP
    ENDIF

    ! now check whether data are only copied locally
    ! if yes set the LOCAL tag
    ! in some cases the RL and the IN layout are the same

    ISND=SUM(GRID%RC_IN%N(1:C%NCPU))-GRID%RC_IN%N(NODE_ME)
    CALL M_sum_i(C, ISND, 1)

    IF (ISND==0) THEN
       GRID%RC_IN%LOCAL=.TRUE.
       ! in this case the send table should be simply 1,...,N
       DO I=1,GRID%RC_IN%N(NODE_ME)
          IF (GRID%RC_IN%TBL(I)/=I) THEN
             WRITE(0,*)'internal error: MAPSET sendtable RC_IN incorrect',I,GRID%RC_IN%TBL(I)
             STOP
          ENDIF
       ENDDO
    ELSE
       GRID%RC_IN%LOCAL=.FALSE.
    ENDIF

    ISND=SUM(GRID%IN_RL%N(1:C%NCPU))-GRID%IN_RL%N(NODE_ME)
    CALL M_sum_i(C, ISND, 1)

    IF (ISND==0) THEN
       GRID%IN_RL%LOCAL=.TRUE.
       DO I=1,GRID%IN_RL%N(NODE_ME)
          IF (GRID%IN_RL%TBL(I)/=I) THEN
             WRITE(0,*)'internal error: MAPSET sendtable IN_RL incorrect',I,GRID%IN_RL%TBL(I)
             STOP
          ENDIF
       ENDDO
    ELSE
       GRID%IN_RL%LOCAL=.FALSE.
    ENDIF

    !
    ! we use a very simple exchange method in which the position of
    ! the send and received data are continously stored on all nodes 
    ! only the number of data, send from each node (N), is known yet
    ! get the number of data received from each node (NI) and calculate
    !  PTR (i) = sum_j=1,i N(i)
    !  PTRI(i) = sum_j=1,i NI(i)
    CALL M_alltoallv_simple(C, GRID%RC_IN%N, GRID%RC_IN%NI, GRID%RC_IN%PTR, GRID%RC_IN%PTRI)
    CALL M_alltoallv_simple(C, GRID%IN_RL%N, GRID%IN_RL%NI, GRID%IN_RL%PTR, GRID%IN_RL%PTRI)
    ! now set on sender node the remote address of receiver
    ! this is required for shmem_put
    CALL M_alltoallv_raddr(C, GRID%RC_IN%PTRI, GRID%RC_IN%RMT)
    CALL M_alltoallv_raddr(C, GRID%RC_IN%PTR , GRID%RC_IN%RMTI)
    CALL M_alltoallv_raddr(C, GRID%IN_RL%PTRI, GRID%IN_RL%RMT)
    CALL M_alltoallv_raddr(C, GRID%IN_RL%PTR , GRID%IN_RL%RMTI)

    ! send positions, where data has to be stored (after global exchange), to remote nodes
    CALL M_alltoall_i(C, RCV_TBL1,           GRID%RC_IN%PTR,  GRID%RC_IN%N, &
         GRID%RC_IN%TBLI(1), GRID%RC_IN%PTRI, GRID%RC_IN%NI )

    CALL M_alltoall_i(C, RCV_TBL2,           GRID%IN_RL%PTR,  GRID%IN_RL%N, &
         GRID%IN_RL%TBLI(1), GRID%IN_RL%PTRI, GRID%IN_RL%NI )
    DWRITE    'map done',NODE_ME,I1,I3,I2,I4,GRID%RC%NP,GRID%IN%NP,GRID%RL_FFT%NP


    ! in some cases the data distribution in the RL and IN layout
    ! is the same, in this case no data rearrangement (SCATTER, GATHER)is required
    ! (typically for LPLANE_WISE set to .TRUE. this is the case)
    GRID%RC_IN%LOCAL_COPY=.FALSE.
    GRID%IN_RL%LOCAL_COPY=.FALSE.

    IF (GRID%IN_RL%LOCAL) THEN

       DO I=1,GRID%IN_RL%NI(NODE_ME)
          IF (GRID%IN_RL%TBLI(I)/=I) EXIT
       ENDDO
       IF (I==GRID%IN_RL%NI(NODE_ME)+1) THEN
          GRID%IN_RL%LOCAL_COPY=.TRUE.
       ENDIF
    ENDIF

    DEALLOCATE(RCV_TBL1,RCV_TBL2)

    RETURN
  END SUBROUTINE MAPSET

!**********************************************************************
!
! this routine communicates the 3d fft data between nodes forward
! i.e. using send -> receive tables
!
!**********************************************************************

  SUBROUTINE MAP_FORWARD(XDAT, NZERO, SNDBUF, RCVBUF, MAP, COMM)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_map) MAP
    TYPE (communic) COMM
    COMPLEX(q) XDAT(*)
    COMPLEX(q) SNDBUF(*),RCVBUF(*)
    INTEGER I, NZERO

    PROFILING_START('map_forward')

    ! only local copy of data (layout of receiver and sender is the same)
    IF (MAP%LOCAL_COPY) THEN
       DO I=MAP%PTRI(COMM%NCPU+1)+1,NZERO
          XDAT(I)=0
       ENDDO
       PROFILING_STOP('map_forward')
       RETURN
    ENDIF

    IF (MAP%LOCAL) THEN
       ! local memory copy suffices if data are exchanged only locally
       DO I=1,MAP%PTR(COMM%NCPU+1)
          SNDBUF(I)=XDAT(I)
       ENDDO
    ELSE
       IF(MAP%PTR(COMM%NCPU+1)/=0) & 
            CALL MAP_GATHER(MAP%PTR(COMM%NCPU+1), MAP%TBL(1), XDAT, SNDBUF)
    ENDIF

    ! if the number of data received equals NZERO we do not need to 
    ! zero XDAT buffer (otherwise we have do)
    IF (NZERO >  MAP%PTRI(COMM%NCPU+1) ) THEN
       DO I=1,NZERO
          XDAT(I)=0
       ENDDO
    ENDIF

    ! be a little bit speed concerned, on one node no need to go through M_alltoallv_z
    IF (MAP%LOCAL) THEN
       CALL MAP_SCATTER(MAP%PTRI(COMM%NCPU+1), MAP%TBLI(1), SNDBUF, XDAT)
    ELSE
       CALL M_alltoallv_z(COMM, SNDBUF(1), MAP%PTR(1),  MAP%N(1), &
            RCVBUF(1), MAP%PTRI(1), MAP%NI(1), MAP%RMT(1))
       IF(MAP%PTRI(COMM%NCPU+1)/=0) & 
            CALL MAP_SCATTER(MAP%PTRI(COMM%NCPU+1), MAP%TBLI(1), RCVBUF, XDAT)
    ENDIF

    PROFILING_STOP('map_forward')

    RETURN
  END SUBROUTINE MAP_FORWARD


!**********************************************************************
!
! this routine communicates the 3d fft data between nodes backward
! i.e. using receive -> send tables
!
!**********************************************************************

  SUBROUTINE MAP_BACKWARD(XDAT, NZERO, SNDBUF, RCVBUF, MAP, COMM)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT NONE

    INTEGER I, NZERO,ISND
    TYPE (grid_map) MAP
    TYPE (communic) COMM
    COMPLEX(q) XDAT(*)
    COMPLEX(q) SNDBUF(*),RCVBUF(*)

    PROFILING_START('map_backward')

    ! only local copy of data (layout of receiver and sender is the same)
    IF (MAP%LOCAL_COPY) THEN
       DO I=MAP%PTR(COMM%NCPU+1)+1,NZERO
          XDAT(I)=0
       ENDDO
       PROFILING_STOP('map_backward')
       RETURN
    ENDIF

    IF (MAP%PTRI(COMM%NCPU+1)/=0) & 
         CALL MAP_GATHER(MAP%PTRI(COMM%NCPU+1), MAP%TBLI(1), XDAT, SNDBUF)

    ! if the number of data received equals NZERO we do not need to 
    ! zero XDAT buffer
    IF (NZERO >  MAP%PTR(COMM%NCPU+1) ) THEN
       DO I=1,NZERO
          XDAT(I)=0
       ENDDO
    ELSE
       !         WRITE(*,*) 'no zero',NZERO,MAP%PTR(COMM%NCPU+1)
    ENDIF

    IF (MAP%LOCAL) THEN
       ! local memory copy suffices if data are exchanged only locally
       DO I=1,MAP%PTR(COMM%NCPU+1)
          XDAT(I)=SNDBUF(I)
       ENDDO
    ELSE
       CALL M_alltoallv_z(COMM, SNDBUF(1), MAP%PTRI(1), MAP%NI(1), &
            RCVBUF(1), MAP%PTR(1),  MAP%N(1), MAP%RMTI(1))

       IF (MAP%PTR(COMM%NCPU+1)/=0) & 
            CALL MAP_SCATTER(MAP%PTR(COMM%NCPU+1), MAP%TBL(1), RCVBUF, XDAT)
    ENDIF

    PROFILING_STOP('map_backward')

    RETURN
  END SUBROUTINE MAP_BACKWARD



!**********************************************************************
!
! two small helper routine which perform scatter and gather operations
!
!**********************************************************************

  SUBROUTINE MAP_SCATTER(N, INDEX, SRC, DEST)
    USE prec
    INTEGER N,INDEX(N)
    COMPLEX(q) SRC(*),DEST(*)
    ! local variables
    INTEGER I

    PROFILING_START('map_scatter')

    DO I=1,N
       DEST(INDEX(I))=SRC(I)
    ENDDO

    PROFILING_STOP('map_scatter')

    RETURN
  END SUBROUTINE MAP_SCATTER

  SUBROUTINE MAP_GATHER(N, INDEX, SRC, DEST)
    USE prec
    INTEGER N,INDEX(N)
    COMPLEX(q) SRC(*),DEST(*)
    ! local variables
    INTEGER I

    PROFILING_START('map_gather')

    DO I=1,N
       DEST(I)=SRC(INDEX(I))
    ENDDO

    PROFILING_STOP('map_gather')

    RETURN
  END SUBROUTINE MAP_GATHER




!**********************************************************************
!
! two small helper routine which performe scatter and gather operations
! if the compiler is not very smart you might try the first two routines
! 
!**********************************************************************

  SUBROUTINE MAP_SCATTER_(N, INDEX, SRC, DEST)
    USE prec
    INTEGER N,INDEX(N)
    COMPLEX(q) SRC(*),DEST(*)
    ! local variables
    INTEGER I,IND1,IND2,IND3,IND4

    IND1=INDEX(1)
    IND2=INDEX(2)
    IND3=INDEX(3)
    IND4=INDEX(4)

    DO I=1,N-4
       DEST(IND1)=SRC(I)
       IND1=IND2
       IND2=IND3
       IND3=IND4
       IND4=INDEX(I+4)
    ENDDO
    DEST(IND1)=SRC(N-4)
    DEST(IND2)=SRC(N-3)
    DEST(IND3)=SRC(N-2)
    DEST(IND4)=SRC(N)

    RETURN
  END SUBROUTINE MAP_SCATTER_

  SUBROUTINE MAP_GATHER_(N, INDEX, SRC, DEST)
    USE prec
    INTEGER N,INDEX(N)
    COMPLEX(q) SRC(*),DEST(*)
    ! local variables
    INTEGER I,IND1,IND2,IND3,IND4

    IND1=INDEX(1)
    IND2=INDEX(2)
    IND3=INDEX(3)
    IND4=INDEX(4)

    DO I=1,N-4
       DEST(I)=SRC(IND1)
       IND1=IND2
       IND1=IND2
       IND1=IND2
       IND2=INDEX(I+2)
    ENDDO
    RETURN

    SRC(N-4)=DEST(IND1)
    SRC(N-3)=DEST(IND2)
    SRC(N-2)=DEST(IND3)
    SRC(N)  =DEST(IND4)

  END SUBROUTINE MAP_GATHER_
