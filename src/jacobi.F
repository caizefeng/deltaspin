#include "symbol.inc"
!=======================================================================
! RCS:  $Id: jacobi.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
! Module containing fast T3D/T3E specific implementation of
! Jacobis matrix diagonalization
!  written by I.J.Bush at Daresbury Laboratory in Feburary 1995.
! based on an algorithm  described by Littlefield (see below)
! put in MODULE and wrapper for VASP by gK
!=======================================================================

! should be compiled only if shmem-put is allowed (T3D_SMA)
! but the routine seems to work even on T3E with data streaming enabled
! thus one can always use it if F90_T3D is specified in the
! makefile

!#if defined(F90_T3D) && defined(gammareal)

#if defined(F90_T3D) && defined(gammareal)
 MODULE jacobi
!=======================================================================
!
! some module wide variables
! used for communication
!
!=======================================================================
      IMPLICIT none
      LOGICAL, PUBLIC :: LJACOBI=.TRUE.

      INTEGER, PARAMETER ::  YES = 1, NO = 0
      INTEGER, PARAMETER :: LOCKED = 1, UNLOCKED = 0
      INTEGER, PARAMETER :: max_size = 2024
      SAVE

      INTEGER    out_group_size
      INTEGER    out_info_lock, out_info_acknowledged

      INTEGER    in_group_size
      INTEGER    in_info_lock, in_info_acknowledged

      REAL       out_buffer( 1:max_size * 2 )
      INTEGER    out_buffer_lock, out_buffer_acknowledged

      REAL       in_buffer( 1:max_size * 2 )
      INTEGER    in_buffer_lock, in_buffer_acknowledged

      INTEGER    MY_COMM  ! mpi-communicator used for barrier

     ! we need symmetric arrays for the shmem communication
     ! COMMON seems to be the only way to guarantee
     ! symmetric objects in MODULES, SAVE did not work
     !
      COMMON / out_info_es / out_group_size, &
     &                       out_info_lock, out_info_acknowledged
      COMMON / in_info_es / in_group_size, &
     &                      in_info_lock, in_info_acknowledged
      COMMON / in_buffer_es / in_buffer, in_buffer_lock,  &
     &                        in_buffer_acknowledged
      COMMON / out_buffer_es / out_buffer, out_buffer_lock,  &
     &                         out_buffer_acknowledged

 CONTAINS

!=======================================================================
!
! jacDSSYEV
! diagonalization of an symmetric matrix using jacobis algorithm
! calling interface for a Jacobi solver written by Ian Bush (see below)
!
!=======================================================================

      SUBROUTINE jacDSSYEV(COMM,AMATIN,W,N)
      USE prec
      USE mpimy
      IMPLICIT NONE

      INCLUDE 'mpif.h'

      TYPE (communic) COMM
      INTEGER N             ! order of the matrix
      GDEF    AMATIN(N,N)   ! input/output matrix
      REAL(q) W(N)          ! eigenvalues
      INTEGER NCOL,NCOLS    ! largest number of columns held on any node +1
      INTEGER NPROCS        ! number of nodes used in jacobi
      INTEGER LDG,LDV       ! leading dimension of G and V
      INTEGER,ALLOCATABLE :: MAP(:,:)
      INTEGER,ALLOCATABLE :: RANK_ARRAY(:)
      REAL(q),ALLOCATABLE :: G(:,:)
      REAL(q),ALLOCATABLE :: V(:,:)
      REAL(q) :: TOLERANCE=1E-8
      LOGICAL :: LRANK=.TRUE. ! ranking allways required

! local
      INTEGER I,ICOL,ICOL_GLOBAL,NODE_ME,IONODE
      INTEGER NCOL_REMAIN,NPROCS_REMAIN
      INTEGER,INTRINSIC :: MY_PE
      INTEGER COLOR,KEY, IERROR, SIZE
      INTEGER, ALLOCATABLE :: hid_tmp(:),hid(:)
      INTEGER,INTRINSIC :: MY_PE

      NODE_ME=COMM%NODE_ME
      IONODE =COMM%IONODE

      LDG=N; LDV=N

   ! first get the hard ware ids of all nodes

      ALLOCATE( hid_tmp(0: COMM%NCPU-1), hid(0: COMM%NCPU-1) )
      hid_tmp =0
      hid_tmp( COMM%NODE_ME-1) = MY_PE()

      call MPI_allreduce( hid_tmp(0), hid(0), COMM%NCPU, &
              MPI_integer, MPI_sum, COMM%MPI_COMM, ierror )

   ! at least two rows per node, determine required number of nodes

      NPROCS=COMM%NCPU
      IF ( COMM%NCPU > N/2) THEN
        NPROCS=N/2
      ENDIF

   ! make routine bullet proof
   ! split up MPI-communicator

      COLOR=1
      IF (COMM%NODE_ME > NPROCS) COLOR = MPI_undefined
      KEY=COMM%NODE_ME

      CALL MPI_COMM_SPLIT(COMM, COLOR, KEY, MY_COMM, IERROR)
      IF ( IERROR /= MPI_success ) &
         CALL M_stop_ierr('jacDSSYEV: Error in MPI_comm_split',ierror)

      ! nodes not taking part can leave the routine

      IF (MY_COMM == MPI_comm_null ) THEN
        AMATIN=0
        W     =0
        GOTO 100
       ENDIF

   ! maximum number of columns + 1

      NCOLS = (N+NPROCS-1)/NPROCS+1

   ! allocate required work arrays

      ALLOCATE(MAP(NPROCS,3), G(LDG,NCOLS), RANK_ARRAY(NCOLS), V(LDV, NCOLS))

   ! transfere PE numbers

      MAP(1:NPROCS,1) = hid(0:NPROCS-1)

   ! setup remaining part of MAP

      ICOL_GLOBAL=1

      DO I=1,NPROCS
        NCOL_REMAIN  = N - ICOL_GLOBAL +1
        NPROCS_REMAIN= NPROCS -I +1
        NCOL  = (NCOL_REMAIN+NPROCS_REMAIN-1)/NPROCS_REMAIN
        MAP( I, 2) = ICOL_GLOBAL
        ICOL_GLOBAL= MIN(ICOL_GLOBAL+NCOL, N+1)
        MAP( I, 3) = ICOL_GLOBAL- MAP( I, 2)
        DWRITE0 I,MAP(I,:)
      ENDDO
      ! check the node pe (just in case ... MPI_COMM_SPLIT
      IF (my_pe() /= MAP(NODE_ME,1 )) THEN
        WRITE(*,*)'internal error (1) jacDSSYEV:',my_pe(),MAP(NODE_ME,1 )
        STOP
      ENDIF

    ! pick up required data (mind that only upper part of AMATIN
    !      is set up correctly)

      DO ICOL=1,MAP( NODE_ME, 3)
        ICOL_GLOBAL=ICOL-1+ MAP( NODE_ME, 2)
        G(1:ICOL_GLOBAL,ICOL)  = AMATIN(1:ICOL_GLOBAL,ICOL_GLOBAL)
        G(ICOL_GLOBAL+1:N,ICOL)= GCONJG(AMATIN(ICOL_GLOBAL,ICOL_GLOBAL+1:N))
      ENDDO

      CALL MPI_barrier( MY_COMM, ierror )

      CALL BFG( N, NCOLS, LDG, G(1,1), LDV, V(1,1),  &
     &                .TRUE. , TOLERANCE, &
     &                NPROCS, MAP, LRANK, RANK_ARRAY, &
     &                GDSUM_MAP )

    ! but the data back into AMATIN

      W=0
      AMATIN=0
      DO ICOL=1,NCOLS
        ICOL_GLOBAL=RANK_ARRAY(ICOL)
        W(ICOL_GLOBAL) = G(ICOL,ICOL)
        AMATIN(1:N, ICOL_GLOBAL) = V(1:N, ICOL)
      ENDDO

      DEALLOCATE(MAP, G, RANK_ARRAY, V)
      CALL MPI_comm_free(MY_COMM,IERROR)
      IF ( IERROR /= MPI_success ) &
         CALL M_stop_ierr('jacDSSYEV: Error in MPI_comm_free',ierror)

 100  CONTINUE
      DEALLOCATE(hid, hid_tmp)

      END SUBROUTINE

#undef zero
#undef one
!**********************************************************************
!
      SUBROUTINE bfg( n, ncols, ldg, G, ldv, V,  &
     &                initialize, tolerance, &
     &                nprocs, map, rank, rank_array, &
     &                global_sum )
!
!  Initial version of the BFG parallel Jacobi algorithm,
! written by I.J.Bush at Daresbury Laboratory in Feburary 1995.
!
!   This routine solves the eigenvalue problem
!
!         GV=VE
!
!  for real symmetric G. The algorithm used is very closely
! based on that described by Littlefield et al in Theor Chim Acta
! 84, 457 ( 1993 ). THIS IS A CRAY T3D SPECIFIC IMPLEMENTATION.
!
!  The matrices are distributed by blocks of columns. The only
! restriction on the 'width' of the blocks is that
! each processor MUST have at least two columns.
!
!
!  Argument list:
!
! N          : The order of G
! NCOLS      : INPUT:  The largest number of columns held on any
!                      processor + 1
!              OUTPUT: How many evals returned on this processor
! LDG        : Must be equal to N in this version
! G          : INPUT : The Portion of the matrix held locally
!              OUTPUT: The G( i, i ) elements,
!                      1 <= i <= NCOLS hold evals,
!                      the rest is destroyed
! LDV        : Must be equal to N in this version
! V          : INPUT : If INITIALIZE is FALSE holds a guess at
!                      the evecs.
!                      If INITIALIZE is TRUE not required to be set
!              OUTPUT: The ith column holds the evec coressponding
!                      to eval G( i, i )
! INITIALIZE: See V
! TOLERANCE : The tolerance on the off diagonal elements for rotations
!             to be skipped. If negative on input set to 1E-8
! NPROCS    : The number of processors to be used
! MAP       : Describes the mapping of the columns of G onto the
!             processors:
!             MAP( i, 1 ) : The PE number of processor i
!             MAP( i, 2 ) : The first column held by processor i
!             MAP( i, 3 ) : The number of columns held by processor i
!             1 <= i <= nprocs
! RANK      : If TRUE on output RANK_ARRAY( i ) will contain
!             the rank of eval G( i, i ) globally.
!             A value of 1 corresponds to the smallest eval.
! RANK_ARRAY: See RANK
! GLOBAL_SUM: External routine, calling syntax
!             CALL GLOBAL_SUM( array, size_array, map )
!             REAL    array( 1:size_array )
!             INTEGER size_array
!             Performs a sum of the elements of ARRAY across the
!             processors held in MAP.
!
      IMPLICIT none
!
!  Argument list declarations:
!
      INTEGER    n
      INTEGER    ncols
      INTEGER    ldg
      REAL       G( 1:ldg, 1:ncols )
      INTEGER    ldv
      REAL       V( 1:ldv, 1:ncols )
      LOGICAL    initialize
      REAL       tolerance
      INTEGER    nprocs
      INTEGER    map( 1:nprocs, 1:3 )
      LOGICAL    rank
      INTEGER    rank_array( 1:ncols )
!
!  External routines:
!
      INTRINSIC  my_pe
      INTEGER    my_pe
      REAL       sdot
      EXTERNAL   sdot
      EXTERNAL   scopy
      EXTERNAL   sgemv
!
!  Cardinals:
!
      REAL       zero, one, two
      PARAMETER( zero = 0.0, one = 1.0, two = 2.0 )
!
!  Local variables:
!
      REAL       a_pp, a_qq, a_pq
      REAL       theta, t, c, s
      REAL       csq, ssq
      REAL       temp( 1:ncols, 1:n ), temp_scal, temp_scal2
      REAL       diag( 1:ncols )
      REAL       evals( 1:n )
      INTEGER    my_proc
      INTEGER    me
      INTEGER    my_cols
      INTEGER    size_1, size_2
      INTEGER    start_1, start_2
      INTEGER    next_size
      INTEGER    where
      INTEGER    iterations, max_iterations
      INTEGER    start, end
      INTEGER    how_many( 1:nprocs ), array( 1:n ), array2( 1:n )
      INTEGER    index
      INTEGER    i, j, p, q, round
      LOGICAL    converged
      LOGICAL    which_way
      INTEGER    ierror
!
!  Set up the message buffers
!
      CALL esbuffers_init
!
!  Set up default tolerance
!
      IF( tolerance .LT. zero ) THEN
         tolerance = 1E-8
      END IF
!
!  Who am I ?
!
      my_proc = my_pe()
!
!  Now find out where I am in the map array
!
      me = 1
!
      DO WHILE( map( me, 1 ) .NE. my_proc )
         me = me + 1
      END DO
!
!  The method splits what I hold locally into two
! groups, called 1 and 2 here. Work out their sizes
! and where they sit in the G matrix.
!
      start_1 = 1
!
      my_cols = map( me, 3 )
!
      size_2  = my_cols / 2
      size_1  = my_cols - size_2
!
      start_2 = start_1 + size_1
!
!  Set up the evecs and G matrix as used internally. The method
! is one sided Jacobi, so have to form V(T)G at some points.
! ( (T) means transpose of the matrix preceding ).
!
      IF( initialize ) THEN
!
!  No guess at evecs provided, set them to the unit
! matrix. G is then as inputted.
!
         DO i = 1, my_cols
            DO j = 1, n
               V( j, i ) = zero
            END DO
            V( i + map( me, 2 ) - 1, i ) = one
         END DO
!
       ELSE
!
!  Guess for evecs provided, now have to do distributed
! matrix multiply. Very noddy method, simply scroll the
! columns of G around the processors evaluating
! all parts of G that will eventually be held locally.
!
         where = map( me, 2 )
!
         DO i = 1, n
!
            CALL esmat_send( n, nprocs, me, map, G( 1, 1 ) )
!
            CALL sgemv( 'T', n, my_cols,  &
     &                  one, V, ldv, &
     &                  G( 1, 1 ), 1,  &
     &                  zero, temp( 1, where ), 1 )
!
            where = where + 1
!
            IF( where .GT. n ) THEN
               where = 1
            END IF
!
            CALL scopy( n * ( my_cols - 1 ), G( 1, 2 ), 1, &
     &                                       G( 1, 1 ), 1 )
!
            CALL esmat_recv( n, nprocs, me, map, G( 1, my_cols ) )
!
         END DO
!
!  Note that as implemented here G really holds G(T) in
! the terminology of the Littlefield paper. Why ? Unit
! strides for good performance in dot products later.
!
         DO i = 1, my_cols
            CALL scopy( n, temp( i, 1 ), ncols, G( 1, i ), 1 )
         END DO
!
!  Probably excessive but just ensure buffer coherency
! after the above message passing.
!
         CALL esbuffers_init
!
      END IF
!
!  Set up the convergance flag
!
      converged = .FALSE.
!
!  Sensible upper bound for the number of sweeps of the
! matrix.
!
!gK: changed, we do not need high accuracy because we call this
! routine again and again
!     max_iterations = NINT( LOG( real( n ) ) / LOG( two ) ) + 2
      max_iterations = 2
!
      iterations = 0
!
! Start Main loop over sweeps
!
      DO WHILE( .NOT. converged .AND. iterations .LE. max_iterations )
      CALL MPI_barrier( MY_COMM, ierror )
!
! Be optimistic, things can only get worse !
!
         converged  = .TRUE.
         iterations = iterations + 1
!
!  What follows is basically a straight forward Jacobi
! with some non-trivial message passing patterns. This
! loop ensures that every part of G 'sees' every other part.
!
         DO round = 1, 2 * nprocs - 1
!
!  To save some time calculate the local diagonal parts
! of the original matrix.
!
            DO i = 1, my_cols
               diag( i ) = sdot( n, G( 1, i ), 1, V( 1, i ), 1 )
            END DO
!
!  What follows now is basically three very similar
! sections doing the Jacobi rotations. The only difference
! is the logic in the loop ordering, and related to this
! the logic of the message passing. The first of the sections
! will be commented heavily, the other two are very similar
! in compute terms so only the message passing will be covered.
!
            IF( round .NE. 1 ) THEN
!
!  Reach here is this is not the first loop of ROUND.
!
               which_way = MOD( round, 2 ) .EQ. 0 .AND. &
     &                     me .EQ. round / 2
!
!  Following the Littlefield paper ...
!
               IF( .NOT. which_way ) THEN
!
!  Reach here if group 1 is to be passed on to the next processor
!
!
!  Tell the next processor how big the local group one is.
!
                  CALL esinfo_send( me, nprocs, map, size_1 )
!
!  Loop over group 1
!
                  DO p = 1, size_1
!
                     a_pp = diag( p )
!
!  And over group 2
!
                     DO q = start_2, my_cols
!
!  Calculate the off-diag element of the original matrix.
!
                        a_pq = sdot( n, G( 1, p ), 1,  &
     &                                  V( 1, q ), 1 )
!
                        IF( ABS( a_pq ) .GT. tolerance ) THEN
!
!  It is too big, eliminate it. As we have had to get rid of
! something the calculation has not converged.
!
                           converged = .FALSE.
!
                           a_qq = diag( q )
!
                           theta = ( a_qq - a_pp ) / ( two * a_pq )
!
                           t = SIGN( one, theta ) / ( ABS( theta ) + &
     &                          SQRT( theta * theta + one ) )
                           c = one / SQRT( t * t + one )
                           s = t * c
!
!  Loops like this to avoid cache-thrashing between G and V
!
                           DO i = 1, n
                              temp_scal = G( i, p )
                              G( i, p ) = c * temp_scal -  &
     &                                    s * G( i, q )
                              G( i, q ) = s * temp_scal +  &
     &                                    c * G( i, q )
                           END DO
!
                           DO i = 1, n
                              temp_scal = V( i, p )
                              V( i, p ) = c * temp_scal -  &
     &                                    s * V( i, q )
                              V( i, q ) = s * temp_scal +  &
     &                                    c * V( i, q )
                           END DO
!
!  Update diagonal elements as held in A_PP and the DIAG array.
!
                           temp_scal2 = two * c * s * a_pq
!
                           csq = c * c
                           ssq = s * s
!
                           temp_scal = csq * a_pp - temp_scal2 + &
     &                                 ssq * a_qq
                           a_qq      = csq * a_qq + temp_scal2 + &
     &                                 ssq * a_pp
!
                           a_pp      = temp_scal
                           diag( q ) = a_qq
!
                        END IF
!
                     END DO
!
!  On the first round pick up how big the incoming group is.
! On latter rounds receive columns from that group, overwriting the
! part of G that we finished with one loop ago.
!
                     IF( p .EQ. 1 ) THEN
                        CALL esinfo_recv( me, nprocs, map, next_size )
                     ELSE IF( p .LE. next_size ) THEN
                        CALL esbuffer_recv( n, nprocs, me, map, &
     &                                      G( 1, p - 1 ),  &
     &                                      V( 1, p - 1 ) )
                     END IF
!
!  Send out the bit of G we have just finished with.
!
                     CALL esbuffer_send( n, nprocs, me, map, &
     &                                   G( 1, p ), V( 1, p ) )
!
                  END DO
!
!  Pick up any parts of G that have not yet been received.
!
                  p = MIN( size_1, next_size )
                  CALL esbuffer_recv( n, nprocs, me, map, &
     &                                G( 1, p ),  &
     &                                V( 1, p ) )
!
!  What follows ensures that the columns of group 2 always directly
! follow those of group 1 in G. Have to consider if the incoming
! group 1 is smmaller than, the same size as, or bigger
! then the present group 1. The same size has been covered,
! these two cover the other cases, shifting the parts of
! G about as required, and picking up any remaining message.
!
                  IF( next_size .LT. size_1 ) THEN
!
                     CALL scopy( n * size_2,  &
     &                           G( 1, start_2 ), 1, &
     &                           G( 1, next_size + 1 ), 1 )
                     CALL scopy( n * size_2,  &
     &                           V( 1, start_2 ), 1, &
     &                           V( 1, next_size + 1 ), 1 )
!
                  ELSE IF( next_size .GT. size_1 ) THEN
!
                     CALL scopy( n * size_2,  &
     &                           G( 1, start_2 ), - 1,  &
     &                           G( 1,  &
     &                           start_2 + next_size - size_1 ), - 1 )
                     CALL scopy( n * size_2,  &
     &                           V( 1, start_2 ), - 1,  &
     &                           V( 1,  &
     &                           start_2 + next_size - size_1 ), - 1 )
!
                     DO p = size_1 + 1, next_size
                        CALL esbuffer_recv( n, nprocs, me, map, &
     &                                      G( 1, p ), V( 1, p ) )
                     END DO
!
                  END IF
!
!  Update the sizes and positions of the group.
!
                  size_1  = next_size
                  my_cols = size_1 + size_2
                  start_2 = size_1 + 1
!
               ELSE
!
!  Very similar to above but now sending group 2 around.
!
                  CALL esinfo_send( me, nprocs, map, size_2 )
!
                  DO p = start_2, my_cols
                     a_pp = diag( p )
                     DO q = 1, size_1
!
                        a_pq = sdot( n, G( 1, p ), 1,  &
     &                                  V( 1, q ), 1 )
!
                        IF( ABS( a_pq ) .GT. tolerance ) THEN
!
                           converged = .FALSE.
!
                           a_qq = diag( q )
!
                           theta = ( a_qq - a_pp ) / ( two * a_pq )
!
                           t = SIGN( one, theta ) / ( ABS( theta ) + &
     &                          SQRT( theta * theta + one ) )
                           c = one / SQRT( t * t + one )
                           s = t * c
!
                           DO i = 1, n
                              temp_scal = G( i, p )
                              G( i, p ) = c * temp_scal -  &
     &                                    s * G( i, q )
                              G( i, q ) = s * temp_scal +  &
     &                                    c * G( i, q )
                           END DO
!
                           DO i = 1, n
                              temp_scal = V( i, p )
                              V( i, p ) = c * temp_scal -  &
     &                                    s * V( i, q )
                              V( i, q ) = s * temp_scal +  &
     &                                    c * V( i, q )
                           END DO
!
                           temp_scal2 = two * c * s * a_pq
!
                           csq = c * c
                           ssq = s * s
!
                           temp_scal = csq * a_pp - temp_scal2 + &
     &                                 ssq * a_qq
                           a_qq      = csq * a_qq + temp_scal2 + &
     &                                 ssq * a_pp
!
                           a_pp      = temp_scal
                           diag( q ) = a_qq
!
                        END IF
!
                     END DO
!
                     IF( p .EQ. start_2 ) THEN
                        CALL esinfo_recv( me, nprocs, map, next_size )
                     ELSE IF( p - start_2 + 1 .LE. next_size ) THEN
                        CALL esbuffer_recv( n, nprocs, me, map, &
     &                                      G( 1, p - 1 ),  &
     &                                      V( 1, p - 1 ) )
                     END IF
!
                     CALL esbuffer_send( n, nprocs, me, map, &
     &                                   G( 1, p ), V( 1, p ) )
!
                  END DO
!
!  Like above book-keeping for the various possible differences
! in group sizes. Note that this time because group 2
! always sits above group 1 we do not have to move things about
! in memory, just ensure that all the messages are received.
! THIS PART IS WHY NCOLS MUST BE LARGER THAN THE MAXIMUM
! NUMBER OF COLUMNS, for if both group 1 and group 2 are larger
! than average, we have assumed that there is spare space at the
! 'top end' of the matrix to store the excess parts of group 2.
!
                  IF( next_size .GE. size_2 ) THEN
                     DO p = my_cols, my_cols + ( next_size - size_2 )
                        CALL esbuffer_recv( n, nprocs, me, map, &
     &                                      G( 1, p ), V( 1, p ) )
                     END DO
                   ELSE
                      CALL esbuffer_recv( n, nprocs, me, map, &
     &                                    G( 1, next_size + size_1 ),  &
     &                                    V( 1, next_size + size_1 ) )
                  END IF

!
!  Update group sizes.
!
                  size_2  = next_size
                  my_cols = size_1 + size_2
!
               END IF
!
             ELSE
!
!  This bit is the first iteration of the round loop.
! As before but this time eliminate ALL off-diags
! that can be calculated from local data.
!
               CALL esinfo_send( me, nprocs, map, size_1 )
!
               DO p = 1, my_cols - 1
                  a_pp = diag( p )
                  DO q = p + 1, my_cols
!
                     a_pq = sdot( n, G( 1, p ), 1, V( 1, q ), 1 )
!
                     IF( ABS( a_pq ) .GT. tolerance ) THEN
!
                        converged = .FALSE.
!
                        a_qq = diag( q )
!
                        theta = ( a_qq - a_pp ) / ( two * a_pq )
!
                        t = SIGN( one, theta ) / ( ABS( theta ) + &
     &                       SQRT( theta * theta + one ) )
                        c = one / SQRT( t * t + one )
                        s = t * c
!
                        DO i = 1, n
                           temp_scal = G( i, p )
                           G( i, p ) = c * temp_scal - s * G( i, q )
                           G( i, q ) = s * temp_scal + c * G( i, q )
                        END DO
!
                        DO i = 1, n
                           temp_scal = V( i, p )
                           V( i, p ) = c * temp_scal - s * V( i, q )
                           V( i, q ) = s * temp_scal + c * V( i, q )
                        END DO
!
                        temp_scal2 = two * c * s * a_pq
!
                        csq = c * c
                        ssq = s * s
!
                        temp_scal = csq * a_pp - temp_scal2 + &
     &                              ssq * a_qq
                        a_qq      = csq * a_qq + temp_scal2 + &
     &                              ssq * a_pp
!
                        a_pp      = temp_scal
                        diag( q ) = a_qq
!
                     END IF
!
                  END DO
!
!  It turns out that the first time around always sends group
! 1. With some very minor extra complications this is like
! the first of these three sections.
!
                  IF( p .EQ. 1 ) THEN
                     CALL esinfo_recv( me, nprocs, map, next_size )
                  ELSE IF( p .LE. MIN( size_1, next_size ) ) THEN
                     CALL esbuffer_recv( n, nprocs, me, map, &
     &                                   G( 1, p - 1 ),  &
     &                                   V( 1, p - 1 ) )
                  END IF
!
                  IF( p .LE. size_1 ) THEN
                     CALL esbuffer_send( n, nprocs, me, map, &
     &                                   G( 1, p ), V( 1, p ) )
                  END IF
!
               END DO
!
               p = MIN( size_1, next_size )
               CALL esbuffer_recv( n, nprocs, me, map, &
     &                             G( 1, p ),  &
     &                             V( 1, p ) )
!
               IF( next_size .LT. size_1 ) THEN
!
                  CALL scopy( n * size_2, G( 1, start_2 ), 1, &
     &                                    G( 1, next_size + 1 ), 1 )
                  CALL scopy( n * size_2, V( 1, start_2 ), 1, &
     &                                    V( 1, next_size + 1 ), 1 )
!
               ELSE IF( next_size .GT. size_1 ) THEN
!
                  CALL scopy( n * size_2,  &
     &                        G( 1, start_2 ), - 1,  &
     &                        G( 1,  &
     &                           start_2 + next_size - size_1 ), - 1 )
                  CALL scopy( n * size_2,  &
     &                        V( 1, start_2 ), - 1,  &
     &                        V( 1,  &
     &                           start_2 + next_size - size_1 ), - 1 )
!
                  DO p = size_1 + 1, next_size
                     CALL esbuffer_recv( n, nprocs, me, map, &
     &                                   G( 1, p ), V( 1, p ) )
                  END DO
!
               END IF
!
!  Update group data.
!
               size_1  = next_size
               my_cols = size_1 + size_2
               start_2 = size_1 + 1
!
            END IF
!
         END DO
!
!  This is a bit perverse, but we want a global and on the
! convergance flag, but all we have is a global REAL sum.
! Hence the following pereversities.
!
         temp_scal = zero
         IF( converged ) THEN
            temp_scal = one
         END IF
!
         CALL global_sum( temp_scal, 1 , map)
!
         i = NINT( temp_scal )
         converged = i .EQ. NPROCS
!
      END DO
!
!  Calculation finished.
!
!  Calculate the local on diags of the matrix. These are the evals.
!
      DO i = 1, my_cols
         a_pp = sdot( n, G( 1, i ), 1, V( 1, i ), 1 )
         G( i, i ) = a_pp
      END DO
!
!  Return how many colums I presently hold, and therefore how
! many evals.
!
      ncols = my_cols
!
!  Rank the evals if required.
!
      IF( rank ) THEN
!
!  Again perversities due to only having a group sum. This time
! we want a gather, twice over.
!
         evals    = zero
         how_many = 0
!
         evals( me ) = ncols
!
         CALL global_sum( evals, nprocs, map )
!
         DO i = 1, nprocs
            how_many( i ) = NINT( evals( i ) )
         END DO
!
         evals = zero
!
         where = 1
         DO i = 1, me - 1
            where = where + how_many( i )
         END DO
!
         DO i = 1, ncols
            evals( i + where - 1 ) = G( i, i )
         END DO
!
         CALL global_sum( evals, n, map )
!
!  >>>>>>>>>>>>>>>>> SERIAL STARTS <<<<<<<<<<<<<<<<<<<
!
!
!  Index by Heapsort. This is simply a slight tidy up of
! the Numerical Recipies routine
!
!
!
!
         DO i = 1, n
            array( i ) = i
         END DO
!
         p = n / 2 + 1
         q = n
!
         DO WHILE( q .NE. 1 .OR. p .NE. 1 )
!
            IF( p .GT. 1 ) THEN
!
               p          = p - 1
               index      = array( p )
               temp_scal2 = evals( index )
!
             ELSE
!
               index      = array( q )
               temp_scal2 = evals( index )
!
               array( q ) = array( 1 )
!
               q = q - 1
!
            END IF
!
            IF( q .EQ. 1 .AND. p .EQ. 1 ) THEN
               array( 1 ) = index
!
             ELSE
!
               i = p
               j = p + p
!
               DO WHILE( j .LE. q )
                  IF( j .LT. q ) THEN
                     IF( evals( array( j     ) ) .LT. &
     &                   evals( array( j + 1 ) ) ) THEN
                        j = j + 1
                     END IF
                  END IF
                  IF( temp_scal2 .LT. evals( array( j ) ) ) THEN
                     array( i ) = array( j )
                     i = j
                     j = j + j
                   ELSE
                     j = q + 1
                  END IF
               END DO
!
               array( i ) = index
!
            END IF
!
         END DO
!
         array2 = 0
!
!  Turn the Indexing array into a ranking array
!
         DO i = 1, n
            array2( array( i ) ) = i
         END DO
!
         rank_array = 0
!
!   >>>>>>>>>>>>>>>>>>> SERIAL ENDS <<<<<<<<<<<<<<<<<<<<<<<
!
!
!  Copy those values which correspond to local evals
! from the ranking array into RANK_ARRAY
!
         DO i = 1, ncols
            rank_array( i ) = array2( i + where - 1 )
         END DO
!
      END IF
!
!  BYE !!!!!!
!
      END SUBROUTINE
!
!*********************************************************************

!*************************************************************************
!
!  I am not going to comment the following ! WHY ? Some very careful
! consideration of how the T3D cache works has gone into these
! and modifications may easily screw things up. Comments
! encourage modifications. Unles you are really sure you
! understand what is going on DO NOT TOUCH !
!
!
!
!*************************************************************************
!
      SUBROUTINE esbuffers_init
!
      SAVE
!
      out_info_acknowledged = YES
       in_info_acknowledged = YES
!
      out_buffer_acknowledged = YES
       in_buffer_acknowledged = YES
!
      out_info_lock = LOCKED
       in_info_lock = LOCKED
!
      out_buffer_lock = LOCKED
       in_buffer_lock = LOCKED
!
      END SUBROUTINE
!
!*************************************************************************
!
      SUBROUTINE esbuffer_send( n, nprocs, me, map, array1, array2 )
!
      SAVE
!
      INTEGER n
      INTEGER nprocs
      INTEGER me
      INTEGER map( 1:nprocs, 1:3 )
      REAL    array1( 1:n )
      REAL    array2( 1:n )
!
      INTEGER next
!
      IF( me .LT. nprocs ) THEN
         next = map( me + 1, 1 )
       ELSE
         next = map( 1, 1 )
      END IF
!
      DO WHILE( out_buffer_acknowledged .EQ. NO )
!DIR$ SUPPRESS out_buffer_acknowledged
      END DO
!
      out_buffer_acknowledged = NO
!
      CALL SCOPY( n, array1, 1, out_buffer, 1 )
      CALL SCOPY( n, array2, 1, out_buffer( n + 1 ), 1 )
!
      CALL shmem_udcflush()
!
      CALL shmem_put( in_buffer, out_buffer, 2 * n, next )
      CALL shmem_put( in_buffer_lock, UNLOCKED, 1, next )
!
      END SUBROUTINE
!
!****************************************************************************************************
!
      SUBROUTINE esmat_send( n, nprocs, me, map, array )
!
      SAVE
!
      INTEGER n
      INTEGER nprocs
      INTEGER me
      INTEGER map( 1:nprocs, 1:3 )
      REAL    array( 1:n )
!
      INTEGER next
!
      IF( me .GT. 1 ) THEN
         next = map( me - 1, 1 )
       ELSE
         next = map( nprocs, 1 )
      END IF
!
      DO WHILE( out_buffer_acknowledged .EQ. NO )
!DIR$ SUPPRESS out_buffer_acknowledged
      END DO
!
      out_buffer_acknowledged = NO
!
      CALL SCOPY( n, array, 1, out_buffer, 1 )
!
      CALL shmem_udcflush()
!
      CALL shmem_put( in_buffer, out_buffer, n, next )
      CALL shmem_put( in_buffer_lock, UNLOCKED, 1, next )
!
      END SUBROUTINE
!
!*************************************************************************
!
      SUBROUTINE esbuffer_recv( n, nprocs, me, map, array1, array2 )
!
      SAVE
!
      INTEGER n
      INTEGER nprocs
      INTEGER me
      INTEGER map( 1:nprocs, 1:3 )
      REAL    array1( 1:n )
      REAL    array2( 1:n )
!
      INTEGER previous
!
      IF( me .GT. 1 ) THEN
         previous = map( me - 1, 1 )
       ELSE
         previous = map( nprocs, 1 )
      END IF
!
      DO WHILE( in_buffer_lock .EQ. LOCKED )
!DIR$ SUPPRESS in_buffer_lock
      END DO
!
      CALL shmem_udcflush()
!
      in_buffer_lock = LOCKED
!
      CALL SCOPY( n, in_buffer,          1, array1, 1 )
      CALL SCOPY( n, in_buffer( n + 1 ), 1, array2, 1 )
!
      CALL shmem_put( out_buffer_acknowledged, YES, 1, previous )
!
      END SUBROUTINE
!
!******************************************************************************
!
      SUBROUTINE esmat_recv( n, nprocs, me, map, array )
!
      SAVE
!
      INTEGER n
      INTEGER nprocs
      INTEGER me
      INTEGER map( 1:nprocs, 1:3 )
      REAL    array( 1:n )
!
      INTEGER previous
!
      IF( me .LT. nprocs ) THEN
         previous = map( me + 1, 1 )
       ELSE
         previous = map( 1, 1 )
      END IF
!
      DO WHILE( in_buffer_lock .EQ. LOCKED )
!DIR$ SUPPRESS in_buffer_lock
      END DO
!
      CALL shmem_udcflush()
!
      in_buffer_lock = LOCKED
!
      CALL SCOPY( n, in_buffer, 1, array, 1 )
!
      CALL shmem_put( out_buffer_acknowledged, YES, 1, previous )
!
      END SUBROUTINE
!
!******************************************************************************
!
      SUBROUTINE esinfo_send( me, nprocs, map, group_size )
!
      SAVE
!
      INTEGER me
      INTEGER nprocs
      INTEGER map( 1:nprocs, 1:3 )
      INTEGER group_size
      INTEGER size_1
      INTEGER start_2
!
      INTEGER next
!
      IF( me .LT. nprocs ) THEN
         next = map( me + 1, 1 )
       ELSE
         next = map( 1, 1 )
      END IF
!
      DO WHILE( out_info_acknowledged .EQ. NO )
!DIR$ SUPPRESS out_info_acknowledged
      END DO
!
      out_info_acknowledged = NO
!
      out_group_size = group_size
      out_info_lock  = UNLOCKED
!
      CALL shmem_udcflush()
!
      CALL shmem_put( in_group_size, out_group_size, 2, next )
!
      END SUBROUTINE
!
!*************************************************************************
!
      SUBROUTINE esinfo_recv( me, nprocs, map, group_size )
!
      SAVE
!
      INTEGER me
      INTEGER nprocs
      INTEGER map( 1:nprocs, 1:3 )
      INTEGER group_size
      INTEGER size_1
      INTEGER start_2
!
      INTEGER previous
!
      IF( me .GT. 1 ) THEN
         previous = map( me - 1, 1 )
       ELSE
         previous = map( nprocs, 1 )
      END IF
!
!
      DO WHILE( in_info_lock .EQ. LOCKED )
!DIR$ SUPPRESS in_info_lock
      END DO
!
      CALL shmem_udcflush()
!
      in_info_lock = LOCKED
!
      group_size = in_group_size
!
      CALL shmem_put( out_info_acknowledged, YES, 1, previous )
!
      END SUBROUTINE
!
!******************************************************************************
!
      SUBROUTINE gdsum_map_old( array, n, map)
!
      SAVE
!
      INTEGER   n
      REAL      array( 1:n )
      INTEGER   map( * )
!
      INTEGER shmem_my_pe, shmem_n_pes
!
      INTEGER my_proc
      INTEGER dimension
      INTEGER direction, destination, plane
      INTEGER i,ierror
      LOGICAL first_call
!
      DATA first_call / .TRUE. /
!
      IF( first_call ) THEN
         first_call = .FALSE.
         my_proc = shmem_my_pe()
         dimension = NINT( LOG( REAL( shmem_n_pes() ) ) / LOG( 2.0 ) )
      END IF
!
      plane = 1
!
      DO i = 1, dimension
!
         direction = my_proc / plane -  &
     &                 2 * ( my_proc / ( 2 * plane ) )
         direction = 1 - 2 * direction
!
         destination = my_proc + plane * direction
!
         CALL MPI_barrier( MY_COMM, ierror )

         CALL shmem_put( in_buffer, array, n, destination )
         CALL MPI_barrier( MY_COMM, ierror )

         CALL shmem_udcflush()
!
         array( 1:n ) = array( 1:n ) + in_buffer( 1:n )
!
         plane = plane * 2
!
      END DO
!
      END SUBROUTINE
!
!**************************************************************

      SUBROUTINE gdsum_map( array, n ,map )
!
      INCLUDE 'mpif.h'

      INTEGER   n
      REAL      array( 1:n )
      INTEGER   map( * )
      INTEGER   ierror

      call MPI_allreduce( array,  in_buffer , n, &
     &                       MPI_double_precision, MPI_sum, &
     &                       MY_COMM, ierror )

      array(1:n)=in_buffer(1:n)

      END SUBROUTINE
!
!**************************************************************

 END MODULE
#else
 MODULE jacobi
   LOGICAL :: LJACOBI=.FALSE.
   CONTAINS

      SUBROUTINE jacDSSYEV(COMM,AMATIN,W,N)
      USE prec
      USE mpimy
      TYPE (communic) COMM
      INTEGER N             ! order of the matrix
      GDEF    AMATIN(N,N)   ! input/output matrix
      REAL(q) W(N)          ! eigenvalues
      WRITE(*,*) 'internal ERROR:  jacDSSYEV is not supported'
      STOP
      END SUBROUTINE

 END MODULE
#endif
