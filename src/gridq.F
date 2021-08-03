#include "symbol.inc"
     
MODULE gridq
!***********************************************************************
!
! this module implements high level routines
! to perform operations on a 3d grid
! to this end it defines a gridquant(ity) structure
! which holds all data required to store a possibly spin dependent
! local quantity such as charges or potentials
! it also implements routines to add, subtract, multiply or fft
! two gridquantities
! presently the multiply routines are not used, since equivalent
! low level routines are implemented in hamil.F
!
! one major complication is that quantities
! might be real or complex in real space, and they might be stored
! in a "half" grid mode in reciprocal space (i.e. since c(G)=c(-G)*
!  with only half of the c(G) stored)
! this is indicated by the flag REAL_IN_REALSPACE
! (should be identical to GRID%LREAL)
! 
! the second complication is that quantities might be stored in
! a spinor representation i.e. 
!  c(up,up), c(up,down), c(down,up), c(down,down)
! or in the magnetisation presentation
!  c(total), c(x), c(y), c(z)
! this is indicated by the flag SPINOR_REPRESENTATION
! 
! written by gK
!
!***********************************************************************
  USE prec
  USE mgrid

  IMPLICIT NONE

  TYPE gridquant
     COMPLEX(q), POINTER :: C(:,:)    ! pointer to the multidimensional gridquantity
     REAL(q), POINTER    :: R(:,:)    ! pointer to the same storage position (real)
     RGRID, POINTER      :: RG(:,:)   ! pointer to the same storage position (rgrid)
     TYPE (grid_3d), POINTER :: GRID
     INTEGER :: NCDIJ                 ! last dimension of arrays
     LOGICAL :: REAL_IN_REALSPACE     ! quantity is real in real space
     LOGICAL :: SPINOR_REPRESENTATION ! is the quantity in spinor representation
     LOGICAL :: REALSPACE             ! quantity is in real space
     LOGICAL :: IS_INITIALISED        ! quantity is initialised
     LOGICAL :: WAS_ALLOCATED         ! proper allocation
  END TYPE gridquant

!=======================================================================
!
! interface for plus minus and multiplication of gridquant structures
! these operators are synthetic sugar
! they allow to perform a limit number of operations such as
!
!  GQ = GQ1 * SCALE + GQ2* SCALE  
!
! applying the operators + or * returns a delayed_plus structure, 
! which stores and accumulates operands 
! until the assign operator finally executes the operation
!
! Alas, F90 is really crippled gK
! I found no way to implement a powerful grid algebra package,
! since constructors and destructors are missing in F90
! anyhow the present implementation fulfills its purpose
!
!=======================================================================

  TYPE delayed_plus
     TYPE (gridquant), POINTER :: OPERAND1, OPERAND2
     REAL(q) :: SCALE1, SCALE2
  END  TYPE delayed_plus

  TYPE delayed_mul
     TYPE (gridquant), POINTER :: OPERAND1, OPERAND2, OPERAND3
     LOGICAL :: LCONJG
     REAL(q) :: SCALE1, SCALE3
  END  TYPE delayed_mul

  TYPE delayed_conjg
     TYPE (gridquant), POINTER :: GQ
  END  TYPE delayed_conjg

  INTERFACE OPERATOR(+)
     MODULE PROCEDURE GQ_PLUS_GQ
     MODULE PROCEDURE GQ_PLUS_DP
     MODULE PROCEDURE DP_PLUS_GQ
     MODULE PROCEDURE DP_PLUS_DP

     MODULE PROCEDURE DM_PLUS_DP
     MODULE PROCEDURE DM_PLUS_GQ
  END INTERFACE


  INTERFACE OPERATOR(-)
     MODULE PROCEDURE GQ_MINUS_GQ
     MODULE PROCEDURE GQ_MINUS_DP
     MODULE PROCEDURE DP_MINUS_GQ
     MODULE PROCEDURE DP_MINUS_DP
  END INTERFACE

  INTERFACE OPERATOR(*)
     MODULE PROCEDURE GQ_MUL_SCALAR
     MODULE PROCEDURE SCALAR_MUL_GQ

     MODULE PROCEDURE GQ_MUL_GQ
     MODULE PROCEDURE DC_MUL_GQ
     MODULE PROCEDURE DM_MUL_SCALE
     MODULE PROCEDURE SCALE_MUL_DM
  END INTERFACE

  INTERFACE ASSIGNMENT(=)
     MODULE PROCEDURE GQ_ASSIGN_DP
     MODULE PROCEDURE GQ_ASSIGN_GQ
     MODULE PROCEDURE GQ_ASSIGN_SC
     MODULE PROCEDURE GQ_ASSIGN_CSC

     MODULE PROCEDURE GQ_ASSIGN_DM
  END INTERFACE

!=======================================================================
!
! interface to the routine returning a pointer to an SEQUENCED F77 
! like REAL array with a given storage convention
! this a substitute for the missing C-cast
! unfortunately no clean solution in F90 exist to alias a REAL and
! a complex pointer
!
!=======================================================================

  INTERFACE
     FUNCTION REAL_POINTER( N1, N2, CW)
       USE prec
       IMPLICIT NONE
       INTEGER N1,N2
       REAL(q), POINTER :: REAL_POINTER(:,:)
       COMPLEX(q), TARGET  :: CW
     END FUNCTION REAL_POINTER
  END INTERFACE

!=======================================================================
!
! interfaces to the routines performing operations
!     C = A * SCALE1 + B * SCALE2
! or
!     D = A * B * SCALE1 + C * SCALE3
! the interface is explicitly speficied here since a MODULE with
! these F77 routines would cause a problem if the first element
! of a pointer array is passed to the F77 routines
! 
!=======================================================================

  INTERFACE
     SUBROUTINE REAL_ADD(A,SCALE1,B,SCALE2,C,NP)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       REAL(q) :: C,A,B
       REAL(q) :: SCALE1, SCALE2
     END SUBROUTINE REAL_ADD
  END INTERFACE

  INTERFACE
     SUBROUTINE COMPLEX_ADD(A,SCALE1,B,SCALE2,C,NP)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       COMPLEX(q) :: C,A,B
       REAL(q) :: SCALE1, SCALE2
     END SUBROUTINE COMPLEX_ADD
  END INTERFACE
  
  INTERFACE
     SUBROUTINE REAL_REAL_REAL_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       REAL(q) :: A, B
       REAL(q) :: C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE REAL_REAL_REAL_MUL
  END INTERFACE
  INTERFACE
     SUBROUTINE CMPLX_CMPLX_REAL_MUL( NP, LCONJG, A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       LOGICAL LCONJG
       COMPLEX(q) :: A, B
       REAL(q) :: C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE CMPLX_CMPLX_REAL_MUL
  END INTERFACE

  INTERFACE
     SUBROUTINE REAL_REAL_CMPLX_MUL( NP, A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       REAL(q) :: A, B
       COMPLEX(q) :: C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE REAL_REAL_CMPLX_MUL
  END INTERFACE

  INTERFACE
     SUBROUTINE CMPLX_CMPLX_CMPLX_MUL( NP, LCONJG, A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       LOGICAL LCONJG
       COMPLEX(q) :: A, B
       COMPLEX(q) :: C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE CMPLX_CMPLX_CMPLX_MUL
  END INTERFACE


  INTERFACE
     SUBROUTINE CMPLX_REAL_REAL_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       COMPLEX(q) :: A
       REAL(q) :: B, C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE CMPLX_REAL_REAL_MUL
  END INTERFACE

  INTERFACE
     SUBROUTINE REAL_CMPLX_CMPLX_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
       USE prec
       IMPLICIT NONE
       INTEGER NP
       REAL(q) :: A
       COMPLEX(q) :: B, C, D
       REAL(q) :: SCALE1, SCALE3
     END SUBROUTINE REAL_CMPLX_CMPLX_MUL
  END INTERFACE


CONTAINS

!=======================================================================
!
! allocate and deallocate a grid quantity
!
!=======================================================================

  SUBROUTINE ALLOCATE_GRID_QUANTITY( GQ, GRID, NCDIJ)
    TYPE (gridquant) GQ
    TYPE (grid_3d), TARGET :: GRID
    INTEGER NCDIJ
    
    GQ%GRID=>GRID
    GQ%REAL_IN_REALSPACE=GRID%LREAL
    GQ%NCDIJ=NCDIJ
    ALLOCATE( GQ%C(GRID%MPLWV, NCDIJ))
    ! alias the real pointer
    GQ%R=> REAL_POINTER( GRID%MPLWV*2, NCDIJ, GQ%C(1,1))
#ifdef realmode
    GQ%RG=>GQ%R
#else
    GQ%RG=>GQ%C
#endif
    GQ%WAS_ALLOCATED=.TRUE.
    GQ%IS_INITIALISED=.FALSE.
    GQ%SPINOR_REPRESENTATION=.FALSE.

  END SUBROUTINE ALLOCATE_GRID_QUANTITY


  SUBROUTINE DEALLOCATE_GRID_QUANTITY( GQ)
    TYPE (gridquant) GQ

    IF (.NOT.GQ%WAS_ALLOCATED) THEN
       WRITE(0,*) 'internal error in DEALLOCATE_GRID_QUANTITY: gridquant not properly allocated'
       STOP
    ENDIF
    IF (ASSOCIATED(GQ%C)) THEN
       DEALLOCATE(GQ%C)
    ENDIF
  END SUBROUTINE DEALLOCATE_GRID_QUANTITY

!=======================================================================
!
! this subroutine allocates a grid quantity but enforces it
! to take the real space presentation of another GRID (GRID2)
! this usually implies that an FFT is not possible 
!
!=======================================================================


  SUBROUTINE ALLOCATE_GRID_QUANTITY_FORCE_RL( GQ, GRID, GRID2, NCDIJ)
    TYPE (gridquant) GQ
    TYPE (grid_3d), TARGET :: GRID
    TYPE (grid_3d) :: GRID2
    INTEGER NCDIJ
    
    GQ%GRID=>GRID
    GQ%REAL_IN_REALSPACE=GRID2%LREAL
    GQ%NCDIJ=NCDIJ
    ALLOCATE( GQ%C(GRID%MPLWV, NCDIJ))
    ! alias the real pointer
    GQ%R=> REAL_POINTER( GRID%MPLWV*2, NCDIJ, GQ%C(1,1))
#ifdef realmode
    GQ%RG=>GQ%R
#else
    GQ%RG=>GQ%C
#endif
    GQ%WAS_ALLOCATED=.TRUE.
    GQ%IS_INITIALISED=.FALSE.
    GQ%SPINOR_REPRESENTATION=.FALSE.

  END SUBROUTINE ALLOCATE_GRID_QUANTITY_FORCE_RL

!=======================================================================
!
! generate_grid_quantity might come handy if a variable has
! been allocated from the stack (which is faster) and a proper grid 
! quantity needs to be build from the allocate two dimensional array
!
!=======================================================================

  SUBROUTINE GENERATE_GRID_QUANTITY( GQ, GRID, C)
    TYPE (gridquant) GQ
    TYPE (grid_3d), TARGET :: GRID
    COMPLEX(q), TARGET :: C(:,:)
    
    GQ%GRID=>GRID
    GQ%REAL_IN_REALSPACE=GRID%LREAL
    GQ%C=> C
    GQ%NCDIJ = SIZE(C,2)
    IF ( GRID%MPLWV /= SIZE(C,1) ) THEN
       WRITE(0,*) 'internal error in GENERATE_GRID_QUANTITY: the supplied array C does not conform to GRID'
       WRITE(0,*) GRID%MPLWV /= SIZE(C,1)
    ENDIF
    ! alias the real pointer
    GQ%R=> REAL_POINTER( GRID%MPLWV*2, GQ%NCDIJ, GQ%C(1,1))
#ifdef realmode
    GQ%RG=>GQ%R
#else
    GQ%RG=>GQ%C
#endif
    GQ%WAS_ALLOCATED=.FALSE.
    GQ%IS_INITIALISED=.FALSE.
    GQ%SPINOR_REPRESENTATION=.FALSE.

  END SUBROUTINE GENERATE_GRID_QUANTITY

!=======================================================================
!
! generate_grid_quantity might come handy if a variable has
! been allocated from the stack (which is faster) and a proper grid 
! quantity needs to be build from the allocate two dimensional array
!
!=======================================================================

  SUBROUTINE SUBINDEX_GQ(GQR, GQ, I1, I2)
    TYPE (gridquant) GQ
    TYPE (gridquant) GQR
    INTEGER :: I1, I2
    
    CALL CHECK_ASSOCIATED_GQ( GQ)
    IF (I1<1) THEN
       WRITE(0,*) 'internal error in SUBINDEX_GQ: first index too small',I1
       STOP
    ENDIF
    IF (I2>GQ%NCDIJ) THEN
       WRITE(0,*) 'internal error in SUBINDEX_GQ: second index too large',I2
       STOP
    ENDIF
    GQR%GRID => GQ%GRID
    GQR%REAL_IN_REALSPACE=GQ%REAL_IN_REALSPACE
    GQR%C=> GQ%C(:,I1:I2)
    GQR%R=> GQ%R(:,I1:I2)
#ifdef realmode
    GQR%RG=>GQR%R
#else
    GQR%RG=>GQR%C
#endif
    GQR%NCDIJ = SIZE(GQR%C,2)
    ! alias the real pointer
    GQR%WAS_ALLOCATED =.FALSE.
    GQR%IS_INITIALISED=GQ%IS_INITIALISED
    GQR%REALSPACE     =GQ%REALSPACE

  END SUBROUTINE  SUBINDEX_GQ

!=======================================================================
!
! check whether a GQ is initialised
!
!=======================================================================

  SUBROUTINE CHECK_INITIALISED_GQ( GQ)
    TYPE (gridquant) GQ

    CALL CHECK_ASSOCIATED_GQ( GQ)

    IF (.NOT. GQ%IS_INITIALISED) THEN
       WRITE(0,*) 'internal error in CHECK_INITIALISED_GQ: a grid quantity is not initialised'
       STOP
    ENDIF

  END SUBROUTINE CHECK_INITIALISED_GQ

!=======================================================================
!
! check whether a GQ is properly set up (not necessarily initialised)
!
!=======================================================================

  SUBROUTINE CHECK_ASSOCIATED_GQ( GQ)
    TYPE (gridquant) GQ
    IF (.NOT. ASSOCIATED(GQ%GRID)) THEN
       WRITE(0,*) 'internal error in CHECK_INITIALISED_GQ: grid pointer is not associated'
       STOP
    ENDIF
    IF (.NOT. ASSOCIATED(GQ%C)) THEN
       WRITE(0,*) 'internal error in CHECK_INITIALISED_GQ: array is not associated'
       STOP
    ENDIF

  END SUBROUTINE CHECK_ASSOCIATED_GQ

!=======================================================================
!
! check whether two GQ are conform to each other
! this is the stringent version, which requires that
! the arrays are entirely conform 
!
!=======================================================================

  SUBROUTINE CHECK_CONFORM_GQ( GQ1, GQ2)
    TYPE (gridquant) GQ1, GQ2
    IF (GQ1%REALSPACE .NEQV. GQ2%REALSPACE) THEN
       WRITE(0,*)'internal error in CHECK_CONFORM: presentation not conform',GQ1%REALSPACE,GQ2%REALSPACE
       STOP
    ENDIF
    IF (GQ1%REALSPACE) THEN
       IF (GQ1%REAL_IN_REALSPACE .NEQV. GQ2%REAL_IN_REALSPACE) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: real presentation not conform',GQ1%REAL_IN_REALSPACE,GQ2%REAL_IN_REALSPACE
          STOP
       ENDIF
       IF (GQ1%GRID%RL%NP /= GQ2%GRID%RL%NP) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: real space not conform',GQ1%GRID%RL%NP,GQ2%GRID%RL%NP
          STOP
       ENDIF
    ELSE
       IF (GQ1%GRID%RC%NP /= GQ2%GRID%RC%NP) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: reciprocal space not conform',GQ1%GRID%RC%NP,GQ2%GRID%RC%NP
          STOP
       ENDIF
    ENDIF
    IF (GQ1%NCDIJ /= GQ2%NCDIJ) THEN
       WRITE(0,*)'internal error in CHECK_CONFORM: last dimensions are not conform',GQ1%NCDIJ, GQ2%NCDIJ
       STOP
    ENDIF
    IF (GQ1%NCDIJ/=1) THEN
       IF (GQ1%SPINOR_REPRESENTATION .NEQV. GQ2%SPINOR_REPRESENTATION) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: spinor presentation not conform',GQ1%SPINOR_REPRESENTATION, GQ2%SPINOR_REPRESENTATION
          STOP
       ENDIF
       
    ENDIF

  END SUBROUTINE CHECK_CONFORM_GQ

!=======================================================================
!
! check whether two GQ are conform to each other
! this is the sloppy version
! it does not check whether two arrays have the same real space
! presentation, and whether NCDIJ agrees
!
!=======================================================================

  SUBROUTINE CHECK_SLOPPY_CONFORM_GQ( GQ1, GQ2)
    TYPE (gridquant) GQ1, GQ2
    IF (GQ1%REALSPACE .NEQV. GQ2%REALSPACE) THEN
       WRITE(0,*)'internal error in CHECK_CONFORM: presentation not conform',GQ1%REALSPACE,GQ2%REALSPACE
       STOP
    ENDIF
    IF (GQ1%REALSPACE) THEN
       IF (GQ1%GRID%RL%NP /= GQ2%GRID%RL%NP) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: real space not conform',GQ1%GRID%RL%NP,GQ2%GRID%RL%NP
          STOP
       ENDIF
    ELSE
       IF (GQ1%GRID%RC%NP /= GQ2%GRID%RC%NP) THEN
          WRITE(0,*)'internal error in CHECK_CONFORM: reciprocal space not conform',GQ1%GRID%RL%NP,GQ2%GRID%RL%NP
          STOP
       ENDIF
    ENDIF

  END SUBROUTINE CHECK_SLOPPY_CONFORM_GQ

  
!=======================================================================
!
! change storage convention from
!  c(up,up), c(up,down), c(down,up), c(down,down)
! to the magnetisation presentation
!  c(total), c(x), c(y), c(z)
! and vice versa
!
!=======================================================================
  
  SUBROUTINE FLIP_GQ( GQ)
    TYPE (gridquant) GQ

    CALL CHECK_INITIALISED_GQ(GQ)

    IF (GQ%REALSPACE .AND. GQ%REAL_IN_REALSPACE) THEN
       ! use real array, number of data RL%NP
       CALL REAL_FLIP(GQ%R, GQ%NCDIJ, GQ%GRID%RL%NP, &
            .NOT. GQ%SPINOR_REPRESENTATION)
    ELSE IF (GQ%REALSPACE) THEN
       ! use compex array, number of data RL%NP
       CALL COMPLEX_FLIP(GQ%C, GQ%NCDIJ, GQ%GRID%RL%NP, &
            .NOT. GQ%SPINOR_REPRESENTATION)
    ELSE
       ! use compex array, number of data RC%NP
       CALL COMPLEX_FLIP(GQ%C, GQ%NCDIJ, GQ%GRID%RC%NP, &
            .NOT. GQ%SPINOR_REPRESENTATION)
    ENDIF
    GQ%SPINOR_REPRESENTATION=.NOT. GQ%SPINOR_REPRESENTATION
    
  END SUBROUTINE FLIP_GQ

!=======================================================================
!
! sum a grid quantity over a specific set of nodes
! using a communicator and finally store the result in a new array
! the final operation is only performed on the root nodes
! of the communicator since the other nodes in the communicator
! should not hold any data
!
! the broadcast operation performs the reverse operation
!
!=======================================================================

  SUBROUTINE SUMRL_GQ( GQ1, GQ2, COMM)
    TYPE (gridquant) GQ1, GQ2
    TYPE(communic) COMM
  ! local
    INTEGER I

    CALL CHECK_INITIALISED_GQ(GQ1)

    IF (.NOT. GQ1%REALSPACE) THEN
       WRITE(0,*) 'internal error in SUMRL_GQ: operates in real space only GQ1',GQ1%REALSPACE
       STOP
    ENDIF

#ifdef MPI
    IF ( GQ1%REAL_IN_REALSPACE) THEN
       DO I=1,GQ1%NCDIJ
          CALLMPI( M_sum_d(COMM, GQ1%R(1,I), GQ1%GRID%RL%NP))
       ENDDO
    ELSE
       DO I=1,GQ1%NCDIJ
          CALLMPI( M_sum_z(COMM, GQ1%C(1,I), GQ1%GRID%RL%NP))
       ENDDO
    END IF

    IF (COMM%NODE_ME==1) THEN
       GQ2=GQ1
    ELSE
       IF (GQ2%GRID%RL%NP /=0) THEN
          WRITE(0,*) 'internal error in SUMRL_GQ: the destination grid contains data on nodes where no operation is performed'
          STOP
       ENDIF
    ENDIF
    GQ2%IS_INITIALISED=GQ1%IS_INITIALISED
    GQ2%REALSPACE=GQ1%REALSPACE
#else
    GQ2=GQ1
#endif
    
  END SUBROUTINE SUMRL_GQ


  SUBROUTINE BROAD_CAST_GQ( GQ1, GQ2, COMM)
    TYPE (gridquant) GQ1, GQ2
    TYPE(communic) COMM
  ! local
    INTEGER I

    CALL CHECK_INITIALISED_GQ(GQ1)

    IF (.NOT. GQ1%REALSPACE) THEN
       WRITE(0,*) 'internal error in BROAD_CAST_GQ: operates in real space only GQ1',GQ1%REALSPACE
       STOP
    ENDIF
#ifdef MPI
    IF (COMM%NODE_ME==1) THEN
       GQ2=GQ1
    ELSE
       IF (GQ1%GRID%RL%NP /=0) THEN
          WRITE(0,*) 'internal error in BROAD_CAST_GQ: the source grid contains data on nodes where no operation is performed'
          STOP
       ENDIF
    ENDIF

    GQ2%IS_INITIALISED=GQ1%IS_INITIALISED
    GQ2%REALSPACE=GQ1%REALSPACE
    IF ( GQ1%REAL_IN_REALSPACE) THEN
       DO I=1,GQ1%NCDIJ
          CALLMPI( M_bcast_d(COMM, GQ2%R(1,I), GQ2%GRID%RL%NP))
       ENDDO
    ELSE
       DO I=1,GQ1%NCDIJ
          CALLMPI( M_bcast_z(COMM, GQ2%R(1,I), GQ2%GRID%RL%NP))
       ENDDO
    END IF
#else
    GQ2=GQ1
#endif

    
  END SUBROUTINE BROAD_CAST_GQ

!=======================================================================
!
! perform FFT from real to reciprocal space and vice versa
! the FFT routine handles the real to complex FFT's 
!
!=======================================================================
  
  SUBROUTINE FFT_GQ( GQ)
    TYPE (gridquant) GQ
  ! local
    INTEGER N
    REAL(q) RINPLW

    CALL CHECK_INITIALISED_GQ(GQ)
    RINPLW=1.0_q/GQ%GRID%NPLWV

    IF ( GQ%REAL_IN_REALSPACE .NEQV. GQ%GRID%LREAL) THEN
       WRITE(0,*) 'internal error in FFT_GQ: FFT using an enforced grid'
       STOP
    ENDIF


    DO N=1,GQ%NCDIJ
       IF (GQ%REALSPACE) THEN
          IF ( GQ%REAL_IN_REALSPACE) THEN
             CALL REAL_ADD(GQ%R(1,N),RINPLW,GQ%R(1,N),0.0_q,GQ%R(1,N),GQ%GRID%RL%NP)
          ELSE
             CALL COMPLEX_ADD(GQ%C(1,N),RINPLW,GQ%C(1,N),0.0_q,GQ%C(1,N),GQ%GRID%RL%NP)
          ENDIF
          CALL FFT3D(GQ%C(1,N), GQ%GRID, -1)
       ELSE
          CALL FFT3D(GQ%C(1,N), GQ%GRID,  1)
       ENDIF
    END DO

    GQ%REALSPACE=.NOT. GQ%REALSPACE

  END SUBROUTINE FFT_GQ

!=======================================================================
!
! set unbalanced lattice vectors to zero
!
!=======================================================================
  
  SUBROUTINE SETUNB_GQ( GQ)
    TYPE (gridquant) GQ
  ! local
    INTEGER N

    CALL CHECK_INITIALISED_GQ(GQ)
    IF (GQ%REALSPACE) THEN
       WRITE(0,*)'internal error in SETUNB_GQ: operates in reciprocal space only'
       STOP
    ENDIF

    DO N=1,GQ%NCDIJ
       CALL SETUNB(GQ%C(1,N), GQ%GRID)
    END DO
  END SUBROUTINE SETUNB_GQ


!=======================================================================
!
! X_PLUS_X store plus operations for delayed execution
! X_MINUS_X store minus operations for delayed execution
!
!=======================================================================

  FUNCTION GQ_PLUS_GQ( GQ1, GQ2)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1, GQ2
    TYPE (delayed_plus) GQ_PLUS_GQ

    GQ_PLUS_GQ%SCALE1=1
    GQ_PLUS_GQ%OPERAND1=> GQ1

    GQ_PLUS_GQ%SCALE2=1
    GQ_PLUS_GQ%OPERAND2=> GQ2
    
  END FUNCTION GQ_PLUS_GQ


  FUNCTION GQ_MINUS_GQ( GQ1, GQ2)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1, GQ2
    TYPE (delayed_plus) GQ_MINUS_GQ

    GQ_MINUS_GQ%SCALE1=1
    GQ_MINUS_GQ%OPERAND1=> GQ1

    GQ_MINUS_GQ%SCALE2=-1
    GQ_MINUS_GQ%OPERAND2=> GQ2
    
  END FUNCTION GQ_MINUS_GQ


  FUNCTION GQ_MUL_SCALAR( GQ1, SCALAR)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    REAL (q), INTENT(IN) :: SCALAR
    TYPE (delayed_plus) GQ_MUL_SCALAR

    GQ_MUL_SCALAR%SCALE1=SCALAR
    GQ_MUL_SCALAR%OPERAND1=> GQ1

    GQ_MUL_SCALAR%SCALE2=0
    GQ_MUL_SCALAR%OPERAND2=> GQ1
  END FUNCTION GQ_MUL_SCALAR


  FUNCTION SCALAR_MUL_GQ( SCALAR, GQ1)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    REAL (q), INTENT(IN) :: SCALAR
    TYPE (delayed_plus) SCALAR_MUL_GQ

    SCALAR_MUL_GQ%SCALE1=SCALAR
    SCALAR_MUL_GQ%OPERAND1=> GQ1
                 
    SCALAR_MUL_GQ%SCALE2=0
    SCALAR_MUL_GQ%OPERAND2=> GQ1
  END FUNCTION SCALAR_MUL_GQ


  FUNCTION GQ_PLUS_DP( GQ1, DP)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    TYPE (delayed_plus), INTENT(IN) :: DP
    TYPE (delayed_plus) GQ_PLUS_DP

    IF (DP%SCALE2/=0) THEN
       WRITE(0,*) 'internal error in GQ_PLUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    GQ_PLUS_DP%SCALE1  =DP%SCALE1
    GQ_PLUS_DP%OPERAND1=>DP%OPERAND1

    GQ_PLUS_DP%SCALE2  =1
    GQ_PLUS_DP%OPERAND2=>GQ1
  END FUNCTION GQ_PLUS_DP


  FUNCTION DP_PLUS_GQ(  DP, GQ1)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    TYPE (delayed_plus), INTENT(IN) :: DP
    TYPE (delayed_plus) DP_PLUS_GQ

    IF (DP%SCALE2/=0) THEN
       WRITE(0,*) 'internal error in DP_PLUS_GQ: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    DP_PLUS_GQ%SCALE1  =DP%SCALE1
    DP_PLUS_GQ%OPERAND1=>DP%OPERAND1

    DP_PLUS_GQ%SCALE2  =1
    DP_PLUS_GQ%OPERAND2=>GQ1
  END FUNCTION DP_PLUS_GQ


  FUNCTION GQ_MINUS_DP( GQ1, DP)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    TYPE (delayed_plus), INTENT(IN) :: DP
    TYPE (delayed_plus) GQ_MINUS_DP

    IF (DP%SCALE2/=0) THEN
       WRITE(0,*) 'internal error in GQ_MINUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    GQ_MINUS_DP%SCALE1  =1
    GQ_MINUS_DP%OPERAND1=>GQ1

    GQ_MINUS_DP%SCALE2  =-DP%SCALE1
    GQ_MINUS_DP%OPERAND2=>DP%OPERAND1
  END FUNCTION GQ_MINUS_DP


  FUNCTION DP_MINUS_GQ(  DP, GQ1)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    TYPE (delayed_plus), INTENT(IN) :: DP
    TYPE (delayed_plus) DP_MINUS_GQ

    IF (DP%SCALE2/=0) THEN
       WRITE(0,*) 'internal error in DP_MINUS_GQ: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    DP_MINUS_GQ%SCALE1  =DP%SCALE1
    DP_MINUS_GQ%OPERAND1=>DP%OPERAND1

    DP_MINUS_GQ%SCALE2  =-1
    DP_MINUS_GQ%OPERAND2=>GQ1
  END FUNCTION DP_MINUS_GQ


  FUNCTION DP_PLUS_DP( DP1, DP2)
    TYPE (delayed_plus), TARGET, INTENT(IN) :: DP1, DP2
    TYPE (delayed_plus) DP_PLUS_DP

    IF (DP1%SCALE2/=0 .OR. DP2%SCALE2/=0 ) THEN
        
       WRITE(0,*) 'internal error in DP_PLUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    DP_PLUS_DP%SCALE1   =DP1%SCALE1
    DP_PLUS_DP%OPERAND1=>DP1%OPERAND1

    DP_PLUS_DP%SCALE2   =DP2%SCALE1
    DP_PLUS_DP%OPERAND2=>DP2%OPERAND1
  END FUNCTION DP_PLUS_DP


  FUNCTION DP_MINUS_DP( DP1, DP2)
    TYPE (delayed_plus), TARGET, INTENT(IN) :: DP1, DP2
    TYPE (delayed_plus) DP_MINUS_DP

    IF (DP1%SCALE2/=0 .OR. & 
        DP2%SCALE2/=0 ) THEN
        
       WRITE(0,*) 'internal error in GQ_MINUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * SCALE + GQ2* SCALE  can be coded'
       STOP
    ENDIF

    DP_MINUS_DP%SCALE1   =DP1%SCALE1
    DP_MINUS_DP%OPERAND1=>DP1%OPERAND1

    DP_MINUS_DP%SCALE2   =-DP2%SCALE1
    DP_MINUS_DP%OPERAND2=>DP2%OPERAND1
  END FUNCTION DP_MINUS_DP


  FUNCTION GQ_MUL_GQ( GQ1, GQ2)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1, GQ2
    TYPE (delayed_mul) GQ_MUL_GQ

    GQ_MUL_GQ%SCALE1=1
    GQ_MUL_GQ%OPERAND1=> GQ1
    GQ_MUL_GQ%OPERAND2=> GQ2
    GQ_MUL_GQ%LCONJG=.FALSE.

    GQ_MUL_GQ%SCALE3=0
    GQ_MUL_GQ%OPERAND3=> GQ1
  END FUNCTION GQ_MUL_GQ


  FUNCTION CONJGGQ( GQ)
    TYPE (delayed_conjg) :: CONJGGQ
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ

    CONJGGQ%GQ => GQ
  END FUNCTION CONJGGQ


  FUNCTION DC_MUL_GQ( DC, GQ2)
    TYPE (delayed_conjg), INTENT(IN) :: DC
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ2
    TYPE (delayed_mul) DC_MUL_GQ

    DC_MUL_GQ%SCALE1=1
    DC_MUL_GQ%OPERAND1=> DC%GQ
    DC_MUL_GQ%OPERAND2=> GQ2
    DC_MUL_GQ%LCONJG=.TRUE.

    DC_MUL_GQ%SCALE3=0
    DC_MUL_GQ%OPERAND3=> GQ2
  END FUNCTION DC_MUL_GQ


  FUNCTION DM_MUL_SCALE( DM, SCALE)
    TYPE (delayed_mul), TARGET, INTENT(IN) :: DM
    REAL(q), INTENT(IN) :: SCALE
    TYPE (delayed_mul) DM_MUL_SCALE

    DM_MUL_SCALE%SCALE1=DM%SCALE1*SCALE
    DM_MUL_SCALE%SCALE3=DM%SCALE3*SCALE
    DM_MUL_SCALE%LCONJG=DM%LCONJG

    DM_MUL_SCALE%OPERAND1=>DM%OPERAND1
    DM_MUL_SCALE%OPERAND2=>DM%OPERAND2
    DM_MUL_SCALE%OPERAND3=>DM%OPERAND3
    
  END FUNCTION DM_MUL_SCALE


  FUNCTION SCALE_MUL_DM(  SCALE, DM)
    TYPE (delayed_mul), TARGET, INTENT(IN) :: DM
    REAL(q), INTENT(IN) :: SCALE
    TYPE (delayed_mul) SCALE_MUL_DM

    SCALE_MUL_DM%SCALE1=DM%SCALE1*SCALE
    SCALE_MUL_DM%SCALE3=DM%SCALE3*SCALE
    SCALE_MUL_DM%LCONJG=DM%LCONJG

    SCALE_MUL_DM%OPERAND1=>DM%OPERAND1
    SCALE_MUL_DM%OPERAND2=>DM%OPERAND2
    SCALE_MUL_DM%OPERAND3=>DM%OPERAND3
    
  END FUNCTION SCALE_MUL_DM


  FUNCTION DM_PLUS_GQ( DM, GQ1)
    TYPE (gridquant), TARGET, INTENT(IN) :: GQ1
    TYPE (delayed_mul), TARGET, INTENT(IN) :: DM
    TYPE (delayed_mul) DM_PLUS_GQ

    DM_PLUS_GQ%SCALE1=DM%SCALE1
    DM_PLUS_GQ%OPERAND1=>DM%OPERAND1
    DM_PLUS_GQ%OPERAND2=>DM%OPERAND2
    DM_PLUS_GQ%LCONJG=   DM%LCONJG

    DM_PLUS_GQ%SCALE3=1
    DM_PLUS_GQ%OPERAND3=> GQ1
    
  END FUNCTION DM_PLUS_GQ


  FUNCTION DM_PLUS_DP( DM, DP)
    TYPE (delayed_mul), TARGET, INTENT(IN) :: DM
    TYPE (delayed_plus), TARGET, INTENT(IN) :: DP
    TYPE (delayed_mul) DM_PLUS_DP

    IF ( DM%SCALE3 /=0) THEN
       WRITE(0,*) 'internal error in DM_PLUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * GQ2 *SCALE + GQ3 * SCALE  can be coded'
       STOP
    ENDIF

    IF ( DP%SCALE2 /=0) THEN
       WRITE(0,*) 'internal error in DM_PLUS_DP: only operations of the kind'
       WRITE(0,*) '  GQ = GQ1 * GQ2 *SCALE + GQ3 * SCALE  can be coded'
       STOP
    ENDIF

    DM_PLUS_DP%SCALE1=DM%SCALE1
    DM_PLUS_DP%OPERAND1=>DM%OPERAND1
    DM_PLUS_DP%OPERAND2=>DM%OPERAND2
    DM_PLUS_DP%LCONJG=   DM%LCONJG


    DM_PLUS_DP%SCALE3=DP%SCALE1
    DM_PLUS_DP%OPERAND3=>DP%OPERAND1
  END FUNCTION DM_PLUS_DP


!=======================================================================
!
! finally here the coding of the actual computations
!
!=======================================================================

  SUBROUTINE GQ_ASSIGN_DP( GQ1, DP)
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQ1
    TYPE (delayed_plus), INTENT(IN) :: DP

    CALL ADD_GRID_QUANTITY(  DP%OPERAND1, DP%SCALE1, DP%OPERAND2, DP%SCALE2, GQ1 )

  END SUBROUTINE GQ_ASSIGN_DP


  SUBROUTINE GQ_ASSIGN_GQ( GQ1, GQ2)
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQ1
    TYPE (gridquant), INTENT(IN) :: GQ2

    CALL ADD_GRID_QUANTITY( GQ2, 1.0_q, GQ2, 0.0_q , GQ1)

  END SUBROUTINE GQ_ASSIGN_GQ


  SUBROUTINE GQ_ASSIGN_SC( GQ1, SC)
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQ1
    REAL (q), INTENT(IN) :: SC

    IF (GQ1%REALSPACE .AND. GQ1%REAL_IN_REALSPACE) THEN
       GQ1%R=SC
    ELSE
       GQ1%C=SC
    END IF

    GQ1%IS_INITIALISED=.TRUE.
  END SUBROUTINE GQ_ASSIGN_SC


  SUBROUTINE GQ_ASSIGN_CSC( GQ1, SC)
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQ1
    COMPLEX(q), INTENT(IN) :: SC

    IF (GQ1%REALSPACE .AND. GQ1%REAL_IN_REALSPACE) THEN
       WRITE(0,*) 'warning GQ_ASSIGN_CSC: complex scalar assigned to a grid quantity'
       GQ1%R=SC
    ELSE
       GQ1%C=SC
    END IF

    GQ1%IS_INITIALISED=.TRUE.
  END SUBROUTINE GQ_ASSIGN_CSC


  SUBROUTINE GQ_ASSIGN_DM( GQ1, DM)
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQ1
    TYPE (delayed_mul), INTENT(IN) :: DM

    IF (DM%LCONJG) THEN
       CALL CONJGMUL_GRID_QUANTITY( DM%OPERAND1, DM%OPERAND2, DM%SCALE1, DM%OPERAND3, DM%SCALE3,  GQ1 )
    ELSE
       CALL MUL_GRID_QUANTITY( DM%OPERAND1, DM%OPERAND2, DM%SCALE1, DM%OPERAND3, DM%SCALE3,  GQ1 )
    ENDIF

  END SUBROUTINE GQ_ASSIGN_DM


!=======================================================================
!
! this subroutine adds two quantities on grids and store it in a third
!   C= A * SCALE1 + B * SCALE2
! presently all three arrays must conform 
!
!=======================================================================

  SUBROUTINE ADD_GRID_QUANTITY( GQ1, SCALE1, GQ2, SCALE2,  GQR )
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQR
    TYPE (gridquant), TARGET, INTENT(IN)  :: GQ1, GQ2
    REAL(q), INTENT(IN) :: SCALE1, SCALE2
    ! local
    INTEGER N

    IF (SCALE1 /=0) CALL CHECK_INITIALISED_GQ(GQ1)
    IF (SCALE2 /=0) CALL CHECK_INITIALISED_GQ(GQ2)
    CALL CHECK_ASSOCIATED_GQ(GQR)

    CALL CHECK_CONFORM_GQ( GQ2, GQ1)
    GQR%REALSPACE=GQ1%REALSPACE
    CALL CHECK_CONFORM_GQ( GQ1, GQR)
    GQR%IS_INITIALISED=.TRUE.

    DO N=1,GQR%NCDIJ

       IF (GQR%REALSPACE .AND. GQR%REAL_IN_REALSPACE) THEN
          ! use real array, number of data RL%NP
          CALL REAL_ADD( GQ1%R(1,N), SCALE1, GQ2%R(1,N), SCALE2,  GQR%R(1,N), GQR%GRID%RL%NP )
       ELSE IF (GQR%REALSPACE) THEN
          ! use compex array, number of data RL%NP
          CALL COMPLEX_ADD( GQ1%C(1,N), SCALE1, GQ2%C(1,N), SCALE2,  GQR%C(1,N), GQR%GRID%RL%NP )
       ELSE
          ! use compex array, number of data RL%NP
          CALL COMPLEX_ADD( GQ1%C(1,N), SCALE1, GQ2%C(1,N), SCALE2,  GQR%C(1,N), GQR%GRID%RC%NP )
       ENDIF
    ENDDO

  END SUBROUTINE ADD_GRID_QUANTITY

!=======================================================================
!
! this subroutine multiplies two quantities on grids and adds a third
! grid quantity and finally stores it in a forth grid quantity
!     C = CONJG(A) * B SCALE1 + D * SCALE3
! restrictions:
! applies only in real space
!
!=======================================================================

  SUBROUTINE CONJGMUL_GRID_QUANTITY( GQ1, GQ2, SCALE1, GQ3, SCALE3,  GQR )
    TYPE (gridquant), TARGET, INTENT(IN)  :: GQ1, GQ2
    TYPE (gridquant), TARGET :: GQ3
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQR
    REAL(q), INTENT(IN) :: SCALE1, SCALE3
    ! local
    INTEGER N1,N2
    REAL(q) :: SCALE

    ! check association and initialisation status
    CALL CHECK_INITIALISED_GQ(GQ1)
    CALL CHECK_INITIALISED_GQ(GQ2)
    IF (.NOT. GQ1%REALSPACE .OR. .NOT. GQ2%REALSPACE ) THEN
       WRITE(0,*) 'internal error in CONGMUL_GRID_QUANTITY: operates in real space only GQ1, GQ2',GQ1%REALSPACE,GQ2%REALSPACE
       STOP
    ENDIF
    CALL CHECK_ASSOCIATED_GQ(GQR)

    ! entire check of third quantity which might be aliased to GQR
    IF (SCALE3 /=0 ) THEN
       CALL CHECK_INITIALISED_GQ(GQ3)
       IF (.NOT. GQ3%REALSPACE) THEN
          WRITE(0,*) 'internal error in CONGMUL_GRID_QUANTITY: operates in real space only GQ3'
          STOP
       ENDIF
       GQR%REALSPACE=.TRUE.
       CALL CHECK_CONFORM_GQ( GQR, GQ3)
    ELSE
       GQ3=GQR
    ENDIF

    ! set result
    GQR%REALSPACE=GQ1%REALSPACE
    GQR%IS_INITIALISED=.TRUE.

    CALL CHECK_CONFORM_GQ( GQ2, GQ1)
    CALL CHECK_SLOPPY_CONFORM_GQ( GQR, GQ1)
    IF ( GQ1%NCDIJ*GQ2%NCDIJ /= GQR%NCDIJ ) THEN
       WRITE(0,*) 'internal error in CONJGMUL_GRID_QUANTITY: not conform', &
            GQ1%NCDIJ, GQ2%NCDIJ, GQR%NCDIJ
       STOP
    ENDIF

    ! spinor * spinor -> charge
    IF (GQ1%REAL_IN_REALSPACE .AND. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=0,GQ2%NCDIJ-1
          DO N2=0,GQ2%NCDIJ-1
             CALL REAL_REAL_REAL_MUL( GQR%GRID%RL%NP,  GQ1%R(1,N2+1), GQ2%R(1,N1+1), SCALE1, & 
                  GQ3%R(1, N1*GQ2%NCDIJ+ N2+1), SCALE3, GQR%R(1, N1*GQ2%NCDIJ+ N2+1)) 
          ENDDO
       ENDDO
    ELSE IF (GQ1%REAL_IN_REALSPACE .AND. .NOT. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=0,GQ2%NCDIJ-1
          DO N2=0,GQ2%NCDIJ-1
             CALL REAL_REAL_CMPLX_MUL( GQR%GRID%RL%NP,  GQ1%R(1,N2+1), GQ2%R(1,N1+1), SCALE1, & 
                  GQ3%C(1, N1*GQ2%NCDIJ+ N2+1), SCALE3, GQR%C(1, N1*GQ2%NCDIJ+ N2+1)) 
          ENDDO
       ENDDO
    ELSE IF (.NOT. GQ1%REAL_IN_REALSPACE .AND. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=0,GQ2%NCDIJ-1
          DO N2=0,GQ2%NCDIJ-1
             CALL CMPLX_CMPLX_REAL_MUL( GQR%GRID%RL%NP,  .TRUE., GQ1%C(1,N2+1), GQ2%C(1,N1+1), SCALE1, & 
                  GQ3%R(1, N1*GQ2%NCDIJ+ N2+1), SCALE3, GQR%R(1, N1*GQ2%NCDIJ+ N2+1)) 
          ENDDO
       ENDDO
    ELSE IF (.NOT. GQ1%REAL_IN_REALSPACE .AND. .NOT. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=0,GQ2%NCDIJ-1
          DO N2=0,GQ2%NCDIJ-1
             CALL CMPLX_CMPLX_CMPLX_MUL( GQR%GRID%RL%NP,  .TRUE., GQ1%C(1,N2+1), GQ2%C(1,N1+1), SCALE1, & 
                  GQ3%C(1, N1*GQ2%NCDIJ+ N2+1), SCALE3, GQR%C(1, N1*GQ2%NCDIJ+ N2+1)) 
          ENDDO
       ENDDO
    ENDIF

  END SUBROUTINE CONJGMUL_GRID_QUANTITY

!=======================================================================
!
! this subroutine multiplies two quantities on grids and add a third
! grid quantity and finally stores it in a forth grid quantity
!     C = A * B SCALE1 + D * SCALE3
! restrictions:
! applies only in real space
!
!=======================================================================

  SUBROUTINE MUL_GRID_QUANTITY( GQ1, GQ2, SCALE1, GQ3, SCALE3,  GQR )
    TYPE (gridquant), TARGET, INTENT(IN)  :: GQ1, GQ2
    TYPE (gridquant), TARGET :: GQ3
    TYPE (gridquant), TARGET, INTENT(OUT) :: GQR
    REAL(q), INTENT(IN) :: SCALE1, SCALE3
    ! local
    INTEGER N1,N2

    ! check association and initialisation status
    CALL CHECK_INITIALISED_GQ(GQ1)
    CALL CHECK_INITIALISED_GQ(GQ2)
    IF (.NOT. GQ1%REALSPACE .OR. .NOT. GQ2%REALSPACE ) THEN
       WRITE(0,*) 'internal error in MUL_GRID_QUANTITY: operates in real space only GQ1 or GQ2'
       STOP
    ENDIF
    CALL CHECK_ASSOCIATED_GQ(GQR)

    ! entire check of third quantity which might be aliased to GQR
    IF (SCALE3 /=0 ) THEN
       CALL CHECK_INITIALISED_GQ(GQ3)
       IF (.NOT. GQ3%REALSPACE) THEN
          WRITE(0,*) 'internal error in MUL_GRID_QUANTITY: operates in real space only GQ3'
          STOP
       ENDIF
       GQR%REALSPACE=.TRUE.
       CALL CHECK_CONFORM_GQ( GQR, GQ3)
    ELSE
       GQ3=GQR
    ENDIF

    ! set result
    GQR%REALSPACE=GQ1%REALSPACE
    GQR%IS_INITIALISED=.TRUE.

    CALL CHECK_CONFORM_GQ( GQ2, GQR)
    CALL CHECK_SLOPPY_CONFORM_GQ( GQ1, GQ2)
    IF ( GQR%NCDIJ*GQ2%NCDIJ == GQ1%NCDIJ ) THEN
       WRITE(0,*) 'internal error in MUL1_GRID_QUANTITY: not conform', &
            GQ1%NCDIJ, GQ2%NCDIJ, GQR%NCDIJ
    ENDIF

    ! potential * spinor -> spinor
    IF (GQ1%REAL_IN_REALSPACE .AND. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=1,GQ2%NCDIJ-1
          CALL REAL_REAL_REAL_MUL( GQR%GRID%RL%NP,  GQ1%R(1, N1*GQ2%NCDIJ+1), GQ2%R(1,1), SCALE1, & 
               GQ3%R(1, N1+1), SCALE3, GQR%R(1, N1+1)) 
          IF (GQ2%NCDIJ==2) THEN
             CALL REAL_REAL_REAL_MUL( GQR%GRID%RL%NP,  GQ1%R(1, N1*GQ2%NCDIJ+2), GQ2%R(1,2), SCALE1, & 
                  GQR%R(1, N1+1), 1.0_q, GQR%R(1, N1+1)) 
          ENDIF
       ENDDO
    ELSE IF (GQ1%REAL_IN_REALSPACE .AND. .NOT. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=1,GQ2%NCDIJ-1
          CALL REAL_CMPLX_CMPLX_MUL( GQR%GRID%RL%NP,  GQ1%R(1, N1*GQ2%NCDIJ+ 1), GQ2%C(1,1), SCALE1, & 
                  GQ3%C(1, N1+1), SCALE3, GQR%C(1, N1+1))
          IF (GQ2%NCDIJ==2) THEN
          CALL REAL_CMPLX_CMPLX_MUL( GQR%GRID%RL%NP,  GQ1%R(1, N1*GQ2%NCDIJ+ 2), GQ2%C(1,2), SCALE1, & 
                  GQR%C(1, N1+1), 1.0_q, GQR%C(1, N1+1))
          ENDIF
       ENDDO
    ELSE IF (.NOT. GQ1%REAL_IN_REALSPACE .AND. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=1,GQ2%NCDIJ-1
          CALL CMPLX_REAL_REAL_MUL( GQR%GRID%RL%NP,  GQ1%C(1, N1*GQ2%NCDIJ+ 1), GQ2%R(1,1), SCALE1, & 
                  GQ3%R(1, N1+1), SCALE3, GQR%R(1, N1+1))
          IF (GQ2%NCDIJ==2) THEN
          CALL CMPLX_REAL_REAL_MUL( GQR%GRID%RL%NP,  GQ1%C(1, N1*GQ2%NCDIJ+ 2), GQ2%R(1,2), SCALE1, & 
                  GQR%R(1, N1+1), 1.0_q, GQR%R(1, N1+1))
          ENDIF
       ENDDO
    ELSE IF (.NOT. GQ1%REAL_IN_REALSPACE .AND. .NOT. GQR%REAL_IN_REALSPACE ) THEN
       DO N1=1,GQ2%NCDIJ-1
          CALL CMPLX_CMPLX_CMPLX_MUL( GQR%GRID%RL%NP, .FALSE., GQ1%C(1, N1*GQ2%NCDIJ+ 1), GQ2%C(1,1), SCALE1, & 
                  GQ3%C(1, N1+1), SCALE3, GQR%C(1, N1+1))
          IF (GQ2%NCDIJ==2) THEN
          CALL CMPLX_CMPLX_CMPLX_MUL( GQR%GRID%RL%NP, .FALSE., GQ1%C(1, N1*GQ2%NCDIJ+ 2), GQ2%C(1,2), SCALE1, & 
                  GQR%C(1, N1+1), 1.0_q, GQR%C(1, N1+1))
          ENDIF
       ENDDO
    ENDIF


  END SUBROUTINE MUL_GRID_QUANTITY

!=======================================================================
!
! this subroutine dumps a grid quantity to stdout
!
!=======================================================================

  SUBROUTINE DUMP_GQ(GQ)
    TYPE (gridquant), TARGET, INTENT(IN)  :: GQ
    
    IF (GQ%REALSPACE .AND. GQ%REAL_IN_REALSPACE) THEN
       WRITE(6,*)'real space dump (real array)'
       CALL WRT_REAL_LINE(6, GQ%GRID, GQ%R)
    ELSE IF (GQ%REALSPACE) THEN
       WRITE(6,*)'real space dump (complex array)'
       CALL WRT_COMPLEX_LINE(6, GQ%GRID, GQ%R)
    ELSE
       WRITE(6,*)'reciprocal space dump'
       CALL WRT_COMPLEX_LINE(6, GQ%GRID, GQ%C)
    END IF

  END SUBROUTINE DUMP_GQ  

!=======================================================================
!
! rearranges a complex array from the spinor representation 
!  c(up,up), c(up,down), c(down,up), c(down,down)
! to  the magnetisation presentation
!  c(total), c(x), c(y), c(z)
! and back
!
!=======================================================================

  SUBROUTINE COMPLEX_FLIP(C, NCDIJ, NDATA, LBACK)
    USE prec
    IMPLICIT NONE
    INTEGER NCDIJ             ! second dimeions of array C and number of components (1,2 or 4)
    INTEGER NDATA             ! number of data in array
    COMPLEX(q) C(:, :)        ! complex array to be rearranged
    LOGICAL  :: LBACK         ! perform back transformation
    ! local
    REAL(q) FAC
    COMPLEX(q) :: C00, C01, C10, C11, CX, CY, CZ, CQU, CQD
    INTEGER K

    IF (NCDIJ==2) THEN
       FAC=1._q
       IF (LBACK) FAC=0.5_q
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          CQU=C(K,1)
          CQD=C(K,2)
          C(K,1)=FAC*(CQU+CQD)
          C(K,2)=FAC*(CQU-CQD)
       ENDDO
    ELSE IF ( NCDIJ==4 .AND. .NOT. LBACK) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          C00=C(K,1)
          C01=C(K,2)
          C10=C(K,3)
          C11=C(K,4)

          C(K,1)= C00+C11             
          C(K,2)= C01+C10             
          C(K,3)=(C01-C10)*(0._q,1._q)
          C(K,4)= C00-C11             
       ENDDO
    ELSE IF ( NCDIJ==4 .AND. LBACK) THEN
       FAC=0.5_q
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          C00=C(K,1)
          CX =C(K,2)
          CY =C(K,3)
          CZ =C(K,4)

          C(K,1)= (C00+CZ)*FAC           
          C(K,2)= (CX-CY*(0._q,1._q))*FAC
          C(K,3)= (CX+CY*(0._q,1._q))*FAC
          C(K,4)= (C00-CZ)*FAC           
       ENDDO
    ENDIF

  END SUBROUTINE COMPLEX_FLIP


  SUBROUTINE REAL_FLIP(C, NCDIJ, NDATA, LBACK)
    USE prec
    IMPLICIT NONE
    INTEGER NDIM              ! first dimension of array C
    INTEGER NCDIJ             ! second dimeions of array C and number of components (1,2 or 4)
    INTEGER NDATA             ! number of data in array
    REAL(q) C(:, :)           ! complex array to be rearranged
    LOGICAL  :: LBACK         ! perform back transformation
    ! local
    REAL(q) FAC
    REAL(q) :: C00, C01, C10, C11, CX, CY, CZ, CQU, CQD
    INTEGER K

    IF (NCDIJ==2) THEN
       FAC=1._q
       IF (LBACK) FAC=0.5_q
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          CQU=C(K,1)
          CQD=C(K,2)
          C(K,1)=FAC*(CQU+CQD)
          C(K,2)=FAC*(CQU-CQD)
       ENDDO
    ELSE IF ( NCDIJ==4 .AND. .NOT. LBACK) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          C00=C(K,1)
          C01=C(K,2)
          C10=C(K,3)
          C11=C(K,4)

          C(K,1)= C00+C11             
          C(K,2)= C01+C10             
          C(K,3)=(C01-C10)*(0._q,1._q)
          C(K,4)= C00-C11             
       ENDDO
    ELSE IF ( NCDIJ==4 .AND. LBACK) THEN
       FAC=0.5_q
!DIR$ IVDEP
!OCL NOVREC
       DO K=1,NDATA
          C00=C(K,1)
          CX =C(K,2)
          CY =C(K,3)
          CZ =C(K,4)

          C(K,1)= (C00+CZ)*FAC           
          C(K,2)= (CX-CY*(0._q,1._q))*FAC
          C(K,3)= (CX+CY*(0._q,1._q))*FAC
          C(K,4)= (C00-CZ)*FAC           
       ENDDO
    ENDIF

  END SUBROUTINE REAL_FLIP

END MODULE gridq

!=======================================================================
!
! this routine returns a pointer to an SEQUENCED F77 like array
! with a given storage convention
! first dimension is N1 second one N2
!
!=======================================================================

  FUNCTION REAL_POINTER( N1, N2, C)
    USE prec
    IMPLICIT NONE
    INTEGER N1,N2
    REAL(q), POINTER :: REAL_POINTER(:,:)
    REAL(q), TARGET  :: C(N1,N2)

    REAL_POINTER => C(:,:)
  END FUNCTION REAL_POINTER



!=======================================================================
!
!  dump the contents of a complex array on a grid
!  along three lines to an specified unit
!  parallel version currently no support
!
!=======================================================================

  SUBROUTINE WRT_COMPLEX_LINE(IU,GRID,CHDEN)
    USE prec
    USE mgrid
    IMPLICIT NONE
    INTEGER IU
    TYPE (grid_3d) GRID
    COMPLEX(q) CHDEN(GRID%NGX_rd,GRID%NGY,GRID%NGZ_rd)
    INTEGER NGX, NGY, NGZ, NX, NY, NZ
#ifndef MPI
    NGX=GRID%NGX
    NGY=GRID%NGY
    NGZ=GRID%NGZ
    
    WRITE(IU,'(2Hx ,10F10.4)') (CHDEN(NX,1,1) ,NX=1,(NGX/2)+1)
    WRITE(IU,'(2Hy ,10F10.4)') (CHDEN(1,NY,1) ,NY=1,(NGY/2)+1)
    WRITE(IU,'(2Hz ,10F10.4)') (CHDEN(1,1,NZ) ,NZ=1,(NGZ/2)+1)
#endif
    RETURN
  END SUBROUTINE WRT_COMPLEX_LINE


  SUBROUTINE WRT_REAL_LINE(IU,GRID,CHDEN)
    USE prec
    USE mgrid
    IMPLICIT NONE
    INTEGER IU
    TYPE (grid_3d) GRID
    COMPLEX(q) CHDEN(GRID%NGX_rd,GRID%NGY,GRID%NGZ_rd)
    INTEGER NGX, NGY, NGZ, NX, NY, NZ
#ifndef MPI
    NGX=GRID%NGX
    NGY=GRID%NGY
    NGZ=GRID%NGZ
    
    WRITE(IU,'(2Hx ,10F10.4)') (CHDEN(NX,1,1) ,NX=1,(NGX/2)+1)
    WRITE(IU,'(2Hy ,10F10.4)') (CHDEN(1,NY,1) ,NY=1,(NGY/2)+1)
    WRITE(IU,'(2Hz ,10F10.4)') (CHDEN(1,1,NZ) ,NZ=1,(NGZ/2)+1)
#endif
    RETURN
  END SUBROUTINE WRT_REAL_LINE



!=======================================================================
!
!  this modul contains helper routines to perform the operation
!     C = A * SCALE1 + B * SCALE2
!  C might be identical to A (but not to B)
!
!=======================================================================

  SUBROUTINE REAL_ADD(A,SCALE1,B,SCALE2,C,NP)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    REAL(q) :: C(NP),A(NP),B(NP)
    REAL(q) :: SCALE1, SCALE2
    ! local
    INTEGER N
    !
    !   C=A*SCALE1
    !
    !  )  case SCALE1=0
    IF (SCALE1==0)  THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=0
       ENDDO
    ELSE IF (SCALE1==1)  THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)
       ENDDO
       !  )  else
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*SCALE1
       ENDDO
    ENDIF
    !
    !   C=C+ B*SCALE2
    !
    IF (SCALE2==0) THEN
    ELSE IF (SCALE2==1) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=C(N)+B(N)
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=C(N)+B(N)*SCALE2
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE REAL_ADD


  SUBROUTINE COMPLEX_ADD(A,SCALE1,B,SCALE2,C,NP)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    COMPLEX(q) :: C(NP),A(NP),B(NP)
    REAL(q) :: SCALE1, SCALE2
    ! local
    INTEGER N
    !
    !   C=A*SCALE1
    !
    !  )  case SCALE1=0
    IF (SCALE1==0)  THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=0
       ENDDO
    ELSE IF (SCALE1==1)  THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)
       ENDDO
       !  )  else
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*SCALE1
       ENDDO
    ENDIF
    !
    !   C=C+ B*SCALE2
    !
    IF (SCALE2==0) THEN
    ELSE IF (SCALE2==1) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=C(N)+B(N)
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=C(N)+B(N)*SCALE2
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE COMPLEX_ADD

!=======================================================================
!
!  this modul contains helper routines to perform the operation
!     C = A * B SCALE1 + D * SCALE3
!  the result can be real or complex
!
!=======================================================================

  SUBROUTINE REAL_REAL_REAL_MUL( NP, A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    REAL(q) :: A(NP), B(NP)
    REAL(q) :: C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N

    IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1+ D(N)*SCALE3
       ENDDO
    ENDIF

  END SUBROUTINE REAL_REAL_REAL_MUL

  SUBROUTINE CMPLX_CMPLX_REAL_MUL( NP, LCONJG, A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    LOGICAL LCONJG
    COMPLEX(q) :: A(NP), B(NP)
    REAL(q) :: C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N

    IF (LCONJG) THEN
       IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=REAL(CONJG(A(N))*B(N),q)*SCALE1
          ENDDO
       ELSE
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=REAL(CONJG(A(N))*B(N),q)*SCALE1+ D(N)*SCALE3
          ENDDO
       ENDIF
    ELSE
       IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=REAL(A(N)*B(N),q)*SCALE1
          ENDDO
       ELSE
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=REAL(A(N)*B(N),q)*SCALE1+ D(N)*SCALE3
          ENDDO
       ENDIF
    ENDIF
    
  END SUBROUTINE CMPLX_CMPLX_REAL_MUL


  SUBROUTINE REAL_REAL_CMPLX_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    REAL(q) :: A(NP), B(NP)
    COMPLEX(q) :: C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N

    IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1+ D(N)*SCALE3
       ENDDO
    ENDIF
  END SUBROUTINE REAL_REAL_CMPLX_MUL


  SUBROUTINE CMPLX_CMPLX_CMPLX_MUL( NP, LCONJG, A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    LOGICAL LCONJG
    COMPLEX(q) :: A(NP), B(NP)
    COMPLEX(q) :: C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N
    IF (LCONJG) THEN
       IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=CONJG(A(N))*B(N)*SCALE1
          ENDDO
       ELSE
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=CONJG(A(N))*B(N)*SCALE1+ D(N)*SCALE3
          ENDDO
       ENDIF
    ELSE
       IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=A(N)*B(N)*SCALE1
          ENDDO
       ELSE
!DIR$ IVDEP
!OCL NOVREC
          DO N=1,NP
             C(N)=A(N)*B(N)*SCALE1+ D(N)*SCALE3
          ENDDO
       ENDIF
    ENDIF
  END SUBROUTINE CMPLX_CMPLX_CMPLX_MUL


  SUBROUTINE CMPLX_REAL_REAL_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    COMPLEX(q) :: A(NP)
    REAL(q) :: B(NP), C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N

    IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=REAL(A(N),q)*B(N)*SCALE1
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=REAL(A(N),q)*B(N)*SCALE1+ D(N)*SCALE3
       ENDDO
    ENDIF

  END SUBROUTINE CMPLX_REAL_REAL_MUL


  SUBROUTINE REAL_CMPLX_CMPLX_MUL( NP,  A, B, SCALE1, D, SCALE3, C)
    USE prec
    IMPLICIT NONE
    INTEGER NP
    REAL(q) :: A(NP)
    COMPLEX(q) :: B(NP), C(NP), D(NP)
    REAL(q) :: SCALE1, SCALE3
    INTEGER N

    IF (SCALE3==0) THEN
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1
       ENDDO
    ELSE
!DIR$ IVDEP
!OCL NOVREC
       DO N=1,NP
          C(N)=A(N)*B(N)*SCALE1+ D(N)*SCALE3
       ENDDO
    ENDIF
    
  END SUBROUTINE REAL_CMPLX_CMPLX_MUL
