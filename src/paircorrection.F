#include "symbol.inc"
MODULE classicfields
  USE prec
  USE base
  IMPLICIT NONE
  REAL(q), SAVE :: OFIELD_KAPPA=0
  REAL(q), SAVE :: OFIELD_K(3)=0
  REAL(q), SAVE :: OFIELD_A=0
  ! A bond of length R is fully counted if
  !   R**2 <= OFIELD_Q6_NEAR
  ! It is not counted at all if
  !   OFIELD_Q6_FAR <= R**2
  ! Inbetween the weighting is done continously and smoothly.
  REAL(q), SAVE :: OFIELD_Q6_NEAR = 0, OFIELD_Q6_FAR = 0
  REAL(q), SAVE :: OFIELD_LATTICE(3,3)

  REAL(q), SAVE :: OFIELD_ENERGY_CURRENT_STEP
  REAL(q), SAVE :: Q_CURRENT_STEP
  REAL(q), SAVE :: Q_ACCUMULATED
  REAL(q), SAVE :: Q2_ACCUMULATED
  INTEGER, SAVE :: Q_COUNTER=0

  INTEGER, SAVE :: DEBUGGING_READY = 0
  INTEGER, SAVE :: DEBUGGING_FPE_FLAGS = 0
  ! FIXME: for debugging only
  INTEGER, SAVE :: DEBUGGING_TIU0 = 0

!**********************************************************************
!                                           
! this subroutine reads the INCAR file to determine whether the user
! supplies any input to this module
!
!**********************************************************************
  CONTAINS
!   SUBROUTINE DEBUGGING_AWAIT_ATTACHEMENT()
!     USE IFPORT
!     IMPLICIT NONE
!     INTEGER :: STATUS
!     CHARACTER(256) :: HOSTNAME
!     STATUS = HOSTNM(HOSTNAME)
!     WRITE (*, '("Host ",A,", pid ",I8," waiting for debugger")') TRIM(HOSTNAME), GETPID()
!     DEBUGGING_READY = 0
!     DO WHILE (DEBUGGING_READY == 0)
!       CALL SLEEP(100)
!     END DO
!     WRITE (*, '(A)') 'Debugger attached!'
!   END SUBROUTINE

!   SUBROUTINE DEBUGGING_FPE_ENABLE()
!     USE IFCORE
!     IMPLICIT NONE
!     INTEGER(4) :: FPE_FLAGS
!     FPE_FLAGS = &
!       FPE_M_TRAP_UND + FPE_M_TRAP_OVF + FPE_M_TRAP_DIV0 + &
!       FPE_M_TRAP_INV + FPE_M_ABRUPT_UND + FPE_M_ABRUPT_DMZ
!     DEBUGGING_FPE_FLAGS = FOR_SET_FPE(FPE_FLAGS)
!   END SUBROUTINE

!   SUBROUTINE DEBUGGING_FPE_DISABLE()
!     USE IFCORE
!     IMPLICIT NONE
!     INTEGER(4) :: FPE_FLAGS
!     FPE_FLAGS = FOR_SET_FPE(DEBUGGING_FPE_FLAGS)
!   END SUBROUTINE

    FUNCTION OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE()
      LOGICAL :: OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE
      OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE = SUM(ABS(OFIELD_K)) > 0
    END FUNCTION

    FUNCTION OFIELD_Q6_IS_ACTIVE()
      LOGICAL :: OFIELD_Q6_IS_ACTIVE
      OFIELD_Q6_IS_ACTIVE = OFIELD_Q6_NEAR < OFIELD_Q6_FAR
    END FUNCTION

    SUBROUTINE CLASSICFIELDS_READER(IU5, IU6, IU0)
      USE vaspxml
      USE mpimy
      IMPLICIT NONE 
      INTEGER IU5   ! input device (usually INCAR)
      INTEGER IU0   ! stderr
      INTEGER IU6   ! stdout
   ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER(1) :: CHARAC
      INTEGER :: ERROR

      LOPEN=.FALSE.
#ifdef DEBUG
! wait for debugger attachement
      IF (IU0>=0) CALL DEBUGGING_AWAIT_ATTACHEMENT()
      CALL MPI_BARRIER(MPI_COMM_WORLD, ERROR)
#endif

! KAPPA spring for external order field
!
      OFIELD_KAPPA=0
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
      CALL RDATAB(LOPEN,INCAR,IU5,'OFIELD_KAPPA','=','#',';','F', &
     &            IDUM,OFIELD_KAPPA,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''OFIELD_KAPPA'' from file INCAR.'
         OFIELD_KAPPA=0
      ENDIF
      CALL XML_INCAR('OFIELD_KAPPA','F',IDUM,OFIELD_KAPPA,CDUM,LDUM,CHARAC,N)

!
! wave vector where order field is applied

      OFIELD_K=0
      CALL RDATAB(LOPEN,INCAR,IU5,'OFIELD_K','=','#',';','F', &
     &            IDUM,OFIELD_K,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''OFIELD_K'' from file INCAR.'
         OFIELD_K=0
      ENDIF
      CALL XML_INCAR_V('OFIELD_K1','F',IDUM,OFIELD_K,CDUM,LDUM,CHARAC,N)

!
! Steinhardt-Nelson Q6 parameters
      OFIELD_Q6_NEAR = 0
      CALL RDATAB(LOPEN,INCAR,IU5,'OFIELD_Q6_NEAR','=','#',';','F', &
     &            IDUM,OFIELD_Q6_NEAR,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''OFIELD_Q6_NEAR'' from file INCAR.'
         OFIELD_Q6_NEAR=0
      ENDIF
      CALL XML_INCAR('OFIELD_Q6_NEAR','F',IDUM,OFIELD_Q6_NEAR,CDUM,LDUM,CHARAC,N)
      OFIELD_Q6_NEAR = OFIELD_Q6_NEAR**2

      OFIELD_Q6_FAR = 0
      CALL RDATAB(LOPEN,INCAR,IU5,'OFIELD_Q6_FAR','=','#',';','F', &
     &            IDUM,OFIELD_Q6_FAR,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''OFIELD_Q6_FAR'' from file INCAR.'
         OFIELD_Q6_FAR=0
      ENDIF
      CALL XML_INCAR('OFIELD_Q6_FAR','F',IDUM,OFIELD_Q6_FAR,CDUM,LDUM,CHARAC,N)
      OFIELD_Q6_FAR = OFIELD_Q6_FAR**2

!
! soll value for order parameter
      OFIELD_A=0
 
      CALL RDATAB(LOPEN,INCAR,IU5,'OFIELD_A','=','#',';','F', &
     &            IDUM,OFIELD_A,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''OFIELD_A'' from file INCAR.'
         OFIELD_A=0
      ENDIF
      CALL XML_INCAR('OFIELD_A','F',IDUM,OFIELD_A,CDUM,LDUM,CHARAC,N)
      Q_COUNTER=0
      Q_ACCUMULATED=0
      Q2_ACCUMULATED=0
    
      CLOSE(IU5)

    END SUBROUTINE CLASSICFIELDS_READER


!***************************** CLASSICFIELDS_WRITE ********************
!
!   this subroutine writes the electric field parameters to the 
!   OUTCAR file
!
!**********************************************************************

    SUBROUTINE CLASSICFIELDS_WRITE(IU6)

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: IU6               ! output unit
      INTEGER :: I

      IF (IU6>=0) THEN
        IF (OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE()) THEN
          WRITE(IU6, 7225) &
            OFIELD_KAPPA, (OFIELD_K(I), I=1,3), OFIELD_A
        ELSE IF (OFIELD_Q6_IS_ACTIVE()) THEN
          WRITE(IU6, 7226) &
            OFIELD_KAPPA, SQRT(OFIELD_Q6_NEAR), SQRT(OFIELD_Q6_FAR), OFIELD_A
        END IF
      END IF

 7225 FORMAT( &
             ' External order field'/  &
             '   OFIELD_KAPPA = ',F18.4,  ' spring constant for field' / &
             '   OFIELD_K     = ',3F6.2,  ' reciprocal lattice vector for external field' / &
             '   OFIELD_A     = ',F18.4,  ' desired value for external field')

 7226 FORMAT( &
             ' External order field'/  &
             '   OFIELD_KAPPA   = ',F18.4,  ' spring constant for field' / &
             '   OFIELD_Q6_NEAR = ',F18.4,  ' within, bonds are fully weighted for Q6' / &
             '   OFIELD_Q6_FAR  = ',F18.4,  ' without, bonds are not weighted at all for Q6' / &
             '   OFIELD_A       = ',F18.4,  ' desired value for external field')
       
    END SUBROUTINE CLASSICFIELDS_WRITE

!************************** XML_WRITE_CLASSICFIELDS   *****************
!
!   this subroutine writes the electric field parameters to the XML file
!   if required
!
!**********************************************************************

    SUBROUTINE XML_WRITE_CLASSICFIELDS
      USE pseudo
      USE vaspxml
      IMPLICIT NONE

      LOGICAL :: LDUM
      INTEGER :: IDUM
      REAL(q) :: RDUM
      COMPLEX(q)  :: CDUM
      CHARACTER(1) :: CHARAC

      CALL XML_TAG("separator","External order field")
      CALL XML_INCAR('OFIELD_KAPPA','F',IDUM,OFIELD_KAPPA,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('OFIELD_K','F',IDUM,OFIELD_K,CDUM,LDUM,CHARAC,3)
      CALL XML_INCAR('OFIELD_Q6_NEAR','F',IDUM,SQRT(OFIELD_Q6_NEAR),CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('OFIELD_Q6_FAR','F',IDUM,SQRT(OFIELD_Q6_FAR),CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('OFIELD_A','F',IDUM,OFIELD_A,CDUM,LDUM,CHARAC,1)
      CALL XML_CLOSE_TAG

    END SUBROUTINE XML_WRITE_CLASSICFIELDS

  
  FUNCTION STEINHARDT_WEIGHT(R2)
    IMPLICIT NONE
    REAL(q) :: STEINHARDT_WEIGHT
    REAL(q), INTENT(IN) :: R2

    IF (OFIELD_Q6_FAR <= R2) THEN
      STEINHARDT_WEIGHT = 0.0_q
    ELSE IF (R2 <= OFIELD_Q6_NEAR) THEN
      STEINHARDT_WEIGHT = 1.0_q
    ELSE
      STEINHARDT_WEIGHT = &
        (OFIELD_Q6_FAR - R2)**2 * &
        (OFIELD_Q6_FAR - 3*OFIELD_Q6_NEAR + 2*R2) / &
        (OFIELD_Q6_FAR - OFIELD_Q6_NEAR)**3
    END IF
  END FUNCTION

  FUNCTION STEINHARDT_DWEIGHT(R2)
    IMPLICIT NONE
    REAL(q) :: STEINHARDT_DWEIGHT
    REAL(q), INTENT(IN) :: R2

    IF (OFIELD_Q6_FAR <= R2) THEN
      STEINHARDT_DWEIGHT = 0.0_q
    ELSE IF (R2 <= OFIELD_Q6_NEAR) THEN
      STEINHARDT_DWEIGHT = 0.0_q
    ELSE
      STEINHARDT_DWEIGHT = &
        6 * (OFIELD_Q6_FAR - R2) * (OFIELD_Q6_NEAR - R2) / &
        (OFIELD_Q6_FAR - OFIELD_Q6_NEAR)**3
    END IF
  END FUNCTION

  ! Computes the Steinhardt-Nelson Q6 order parameter and the
  ! derivatives DQ6 with respect to each ion coordinate.
  ! The derivates are evaluted symbolically and the algorithm scales
  ! like N_IONS^2.
  SUBROUTINE STEINHARDT_Q6_DQ6(POS, N_IONS, LATTICE, Q6, DQ6)
    IMPLICIT NONE
    ! ion/atom positions in direct coordinates
    REAL(q), INTENT(IN) :: POS(3,N_IONS)
    ! number of ions
    INTEGER, INTENT(IN) :: N_IONS
    ! column vectors of the direct lattice.
    ! The ion positions in cartesian coordinates = MATMUL(LATTICE, POS)
    REAL(q), INTENT(IN) :: LATTICE(3,3)
    ! Q6 order parameter
    REAL(q), INTENT(OUT) :: Q6
    ! derivatives of Q6 with respect to each ion position as given in POS
    REAL(q), INTENT(OUT) :: DQ6(3,N_IONS)

    ! ion indicies and spherical harmonic index
    INTEGER :: I, J, M
    ! vector from ion I to ion J, square of its length and its length
    REAL(q) :: R_IJ(3), R2, R
    ! weight of the pair IJ, sum of all weights
    REAL(q) :: W, TOTAL_W
    ! derivative of the sum of all weights with respect to each ion position
    ! in cartesian coordinates
    REAL(q) :: DTOTAL_W(3,N_IONS)
    ! cos(theta)  where theta is the angle between +z and R_IJ
    REAL(q) :: C
    ! exp(I*phi)*sin(theta)  where phi is the azimuthal angle of R_IJ
    COMPLEX(q) :: S
    ! Q6_m part of Q6 for each m. Note that |Q6_-m| = |Q6_+m|
    COMPLEX(q) :: Q6M(0:6)
    ! derivative of each Q6_m with respect to each ion position in cartesian
    ! coordinates
    COMPlEX(q) :: DQ6M(3,N_IONS,0:6)
    ! derivative of each Q6_m with respect to W, C and S for the current pair
    COMPLEX(q) :: PARTIAL_DQ6M(3,0:6)
    ! derivative of W, C and S with respect to each component of R_IJ
    COMPLEX(q) :: JACOBIAN(3,3)
    ! derivative of each Q6_m with respect to each component of R_IJ
    COMPLEX(q) :: DQ6M_IJ(3,0:6)
    ! constant factors of each |Q6_m|^2.
    ! The factor 2 comes from |Q6_-m| = |Q6_+m|
    REAL(q), PARAMETER :: FACTORS(0:6) = (/ &
      4., 2*168., 2*105., 2*420., 2*126., 2*2772., 2*231. &
    /) / 1024

    Q6M = 0
    DQ6M = 0
    TOTAL_W = 0
    DTOTAL_W = 0
    DO I = 1, N_IONS-1
      DO J = I+1, N_IONS
        R_IJ = MATMUL( &
          LATTICE, MODULO(POS(:,J) - POS(:,I) + 0.5_q, 1.0_q) - 0.5_q &
        )
        R2 = SUM(R_IJ*R_IJ)
        W = STEINHARDT_WEIGHT(R2)
        IF (W > 0.0_q) THEN
          R = SQRT(R2)
          ! cos(theta)
          C = R_IJ(3) / R
          ! exp(I*phi) * sin(theta)
          S = CMPLX(R_IJ(1), R_IJ(2)) / R

          ! TODO: use Horner scheme for evaluation and test speed
          ! Evaluate dQ6m/dw
          PARTIAL_DQ6M(1,0) = -5 + 105*c**2 - 315*c**4 + 231*c**6
          PARTIAL_DQ6M(1,1) = (5*c - 30*c**3 + 33*c**5) * s
          PARTIAL_DQ6M(1,2) = (1 - 18*c**2 + 33*c**4) * s**2
          PARTIAL_DQ6M(1,3) = (-3*c + 11*c**3) * s**3
          PARTIAL_DQ6M(1,4) = (-1 + 11*c**2) * s**4
          PARTIAL_DQ6M(1,5) = c * s**5
          PARTIAL_DQ6M(1,6) = s**6

          ! Evaluate all Q6m:
          Q6M = Q6M + W * PARTIAL_DQ6M(1,:)

          ! Evaluate dQ6m/dc
          PARTIAL_DQ6M(2,0) = W * (210*c - 1260*c**3 + 1386*c**5)
          PARTIAL_DQ6M(2,1) = W * (5 - 90*c**2 + 165*c**4) * s
          PARTIAL_DQ6M(2,2) = W * (-36*c + 132*c**3) * s**2
          PARTIAL_DQ6M(2,3) = W * (-3 + 33*c**2) * s**3
          PARTIAL_DQ6M(2,4) = W * 22*c * s**4
          PARTIAL_DQ6M(2,5) = W * s**5
          PARTIAL_DQ6M(2,6) = 0

          ! Evaluate dQ6m/ds
          ! TODO: compute common factors of dQ6m/dw and dQ6m/ds and use for both
          PARTIAL_DQ6M(3,0) = 0
          PARTIAL_DQ6M(3,1) = W * (5*c - 30*c**3 + 33*c**5)
          PARTIAL_DQ6M(3,2) = W * 2 *(1 - 18*c**2 + 33*c**4) * s
          PARTIAL_DQ6M(3,3) = W * 3 *(-3*c + 11*c**3) * s**2
          PARTIAL_DQ6M(3,4) = W * 4 *(-1 + 11*c**2) * s**3
          PARTIAL_DQ6M(3,5) = W * 5*c * s**4
          PARTIAL_DQ6M(3,6) = W * 6*s**5

          ! Evaluate the column vector (dw/dx, dw/dy, dw/dz)
          JACOBIAN(:,1) = STEINHARDT_DWEIGHT(R2) * 2 * R_IJ
          ! Evaluate the column vector (dc/dx, dc/dy, dc/dz)
          JACOBIAN(:,2) = (/ -C*R_IJ(1)/R2, -C*R_IJ(2)/R2, 1/R - C**2/R /)
          ! Evaluate the column vector (ds/dx, ds/dy, ds/dz)
          JACOBIAN(:,3) = (/ &
            1/R - S*R_IJ(1)/R2, (0,1)/R - S*R_IJ(2)/R2, -S*R_IJ(3)/R2 &
          /)

          DO M = 0, 6
            DQ6M_IJ(:,M) = MATMUL(JACOBIAN, PARTIAL_DQ6M(:,M))
          END DO
          DQ6M(:,J,:) = DQ6M(:,J,:) + DQ6M_IJ
          DQ6M(:,I,:) = DQ6M(:,I,:) - DQ6M_IJ

          TOTAL_W = TOTAL_W + W
          DTOTAL_W(:,J) = DTOTAL_W(:,J) + JACOBIAN(:,1)
          DTOTAL_W(:,I) = DTOTAL_W(:,J) - JACOBIAN(:,1)
        END IF
      END DO
    END DO

#if defined(DEBUG)
    IF (DEBUGGING_TIU0 >= 0) THEN
      WRITE (*, *) "Q6M:", Q6M
      DO M = 0, 6
        WRITE (*, '("DQ6M(",I1,"):")') M
        WRITE (*, *) DQ6M(:,:,M)
      END DO
      WRITE (*, *) "TOTAL_W", TOTAL_W
      WRITE (*, *) "DTOTAL_W", DTOTAL_W
    END IF
#endif

    ! Calculate the weighted average for each m
    Q6M = Q6M / TOTAL_W
    ! Calculate the derivative of the above expression with respect to each
    ! ion coordinate
    DO M = 0, 6
      DQ6M(:,:,M) = (DQ6M(:,:,M) - Q6M(M)*DTOTAL_W) / TOTAL_W
    END DO

    ! Compute sqrt( sum_m c_m |Q6_m|^2 )
    ! NOTE: Enable the compiler's f90 option to prevent conversion to KIND=1
    Q6 = SQRT(SUM(FACTORS * (REAL(Q6M)**2 + AIMAG(Q6M)**2)))

    ! Compute the derivative of the above expression with respect to each
    ! ion coordinate
    DQ6 = 0
    DO M = 0, 6
      DQ6 = DQ6 + FACTORS(M) / Q6 * ( &
        REAL(Q6M(M)) * REAL(DQ6M(:,:,M)) + &
        AIMAG(Q6M(M)) * AIMAG(DQ6M(:,:,M)) &
      )
    END DO

    ! convert the derivatives into direct coordinates.
    DQ6 = MATMUL(LATTICE, DQ6)

#if defined(DEBUG)
    IF (DEBUGGING_TIU0 >= 0) THEN
      WRITE (*, *) "DQ6:"
      WRITE (*, *) DQ6
    END IF
#endif
  END SUBROUTINE

  SUBROUTINE OFIELD_Q6_FIELD(POS, N_IONS, A, B, IU6, FOR, E)
    USE prec
    IMPLICIT NONE
    REAL(q), INTENT(IN) :: POS(3,N_IONS)
    INTEGER, INTENT(IN) :: N_IONS
    REAL(q), INTENT(IN) :: A(3,3), B(3,3)
    INTEGER, INTENT(IN) :: IU6
    REAL(q), INTENT(OUT) :: FOR(3,N_IONS), E

    REAL(q) :: DQ6(3,N_IONS)

    IF (Q_COUNTER == 0) THEN
      ! Only use the lattice shape of the initial configuration.
      ! Thus, the order parameter does not depend on the cell shape
      ! preventing stress.
      OFIELD_LATTICE = A
    END IF

    CALL STEINHARDT_Q6_DQ6(POS, N_IONS, OFIELD_LATTICE, Q_CURRENT_STEP, DQ6)
!    WRITE (IU6, *) 'Derivative of Q6 with respect to ion coordinates='
!    DO I = 1, N_IONS
!      WRITE (IU6, '(2X,3(F12.8))') DQ6(:,I)
!    END DO

    E = OFIELD_KAPPA*0.5_q * (OFIELD_A - Q_CURRENT_STEP)**2
    ! convert forces into cartesian coordinates
    FOR = MATMUL(B, -OFIELD_KAPPA * (Q_CURRENT_STEP - OFIELD_A) * DQ6)

    ! Do statistics
    OFIELD_ENERGY_CURRENT_STEP = E
    Q_ACCUMULATED  = Q_ACCUMULATED + Q_CURRENT_STEP
    Q2_ACCUMULATED = Q2_ACCUMULATED + Q_CURRENT_STEP**2
    Q_COUNTER      = Q_COUNTER + 1    
  END SUBROUTINE

  SUBROUTINE OFIELD_COLLECTIVE_DENSITY_FIELD(POS, N_IONS, A, B, FOR, E)
    USE prec
    USE constant
    IMPLICIT NONE
    REAL(q), INTENT(IN) :: POS(3, N_IONS)
    INTEGER, INTENT(IN) :: N_IONS
    REAL(q), INTENT(IN) :: A(3,3), B(3,3)
    REAL(q), INTENT(OUT) :: FOR(3, N_IONS), E

    INTEGER :: I, J
    REAL(q) :: G, INNER_DERIVATIVE, LOCAL_FOR(3, N_IONS)
    COMPLEX(q) :: RHOK

    DO I= 1, N_IONS
       G = DOT_PRODUCT(POS(:,I), OFIELD_K)
       RHOK = RHOK + EXP(-CITPI*G)
    ENDDO

    RHOK = RHOK / SQRT(REAL(N_IONS, KIND=q))
    Q_CURRENT_STEP = ABS(RHOK)
    Q_ACCUMULATED  = Q_ACCUMULATED + Q_CURRENT_STEP
    Q2_ACCUMULATED = Q2_ACCUMULATED + Q_CURRENT_STEP**2
    Q_COUNTER      = Q_COUNTER + 1

    E = OFIELD_KAPPA*0.5_q * (OFIELD_A - Q_CURRENT_STEP)**2

    DO I = 1, N_IONS
      G = DOT_PRODUCT(POS(:,I), OFIELD_K)
      INNER_DERIVATIVE = AIMAG(RHOK * EXP(CITPI*G))
      LOCAL_FOR(:,I) = OFIELD_K * &
        TPI * INNER_DERIVATIVE * OFIELD_KAPPA * (Q_CURRENT_STEP - OFIELD_A)
    END DO
    LOCAL_FOR = LOCAL_FOR / Q_CURRENT_STEP / SQRT(REAL(N_IONS, KIND=q))
    OFIELD_ENERGY_CURRENT_STEP = E

    FOR = MATMUL(B, LOCAL_FOR)
  END SUBROUTINE


!*********************************************************************
!   SUBROUTINE OFIELD
! this implements the external order field as suggested by
! Pedersen et al.
!  FIXME: update reference
! the module facilitates e.g. melting point calculations
!
!*********************************************************************

    SUBROUTINE OFIELD(COMM, IO, LATT_CUR, DYN, T_INFO, OFIELD_FOR, OFIELD_E)
      USE mpimy
      USE lattice
      USE poscar
      TYPE (communic), INTENT(IN)  :: COMM
      TYPE (in_struct), INTENT(IN) :: IO
      TYPE(latt), INTENT(IN)       :: LATT_CUR
      TYPE(dynamics), INTENT(IN)   :: DYN
      TYPE(type_info), INTENT(IN)  :: T_INFO
      REAL(q), INTENT(OUT)         :: OFIELD_FOR(3, T_INFO%NIONS)
      REAL(q), INTENT(OUT)         :: OFIELD_E

      DEBUGGING_TIU0 = IO%IU0

      IF (OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE()) THEN
        CALL OFIELD_COLLECTIVE_DENSITY_FIELD( &
          DYN%POSION, T_INFO%NIONS, LATT_CUR%A, LATT_CUR%B, &
          OFIELD_FOR, OFIELD_E &
        )
      ELSE IF (OFIELD_Q6_IS_ACTIVE()) THEN
!        CALL DEBUGGING_FPE_ENABLE()
        CALL OFIELD_Q6_FIELD( &
          DYN%POSION, T_INFO%NIONS, LATT_CUR%A, LATT_CUR%B, &
          IO%IU6, &
          OFIELD_FOR, OFIELD_E &
        )
!        CALL DEBUGGING_FPE_DISABLE()
      ELSE
        OFIELD_FOR = 0
        OFIELD_E = 0
      END IF
    END SUBROUTINE OFIELD

    SUBROUTINE OFIELD_WRITE(IU6)
      INTEGER, INTENT(IN) :: IU6

      IF (IU6>=0 .AND. (OFIELD_COLLECTIVE_DENSITY_IS_ACTIVE() .OR. OFIELD_Q6_IS_ACTIVE())) THEN
         WRITE(IU6,'( "  order field    energy = ",F16.6," eV"/)') OFIELD_ENERGY_CURRENT_STEP
         WRITE(IU6,'( "  Q=",F20.14," <Q>=",F14.7," <Q*Q-<Q><Q>>=",F14.7)') & 
              Q_CURRENT_STEP, Q_ACCUMULATED/Q_COUNTER,  & 
              Q2_ACCUMULATED/Q_COUNTER-(Q_ACCUMULATED/Q_COUNTER)**2
      ENDIF

    END SUBROUTINE OFIELD_WRITE


!*********************************************************************
!   SUBROUTINE PAIR_CORRECTION
!
! this subroutine can be used to calculate the effect of an additional
! pairwise additative potential
! the resulting energy is returned in ENERGY and the forces are
! added to the array FORCES
! the original routine was written by Gilles de Wijs
! it was cleaned up a little bit by gK
!
!*********************************************************************

    SUBROUTINE PAIR_CORRECTION(NIONS, NTYP, NI_TYP, &
         A,BNORM,XC,FORCES,ENERGY,IU5,IU6)

      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)

      INTEGER, PARAMETER :: NDIM=100 ! maximal number of points used for the presentation 
                                  ! of the pair potential
      REAL(q), PARAMETER ::  VOLF=1.005_q ! finite differences are used to evaluate the stress tensor
                                  ! set the magnitude of the volume change 
      INTEGER NIONS               ! number of ions
      INTEGER NTYP                ! number of species
      INTEGER NI_TYP(NTYP)        ! number of atoms per species
 
      REAL(q) XC(3,NIONS)         ! position of each ion

      REAL(q) A(3,3),BNORM(3)
      REAL(q) FORCES(3,NIONS)
      REAL(q),ALLOCATABLE :: CHECKF(:,:)
      REAL(q) V(3),TMP(NDIM)
      INTEGER IU5, IU6
      ! local saved variables
      REAL(q),SAVE :: PSP(NDIM,5),APACO,XMIN,XMAX,ARGSC
      INTEGER,SAVE :: NPOINTS
      ! status of routine (0=uninitialized, 1=used/initialized 2=unused)
      INTEGER :: STATUS=0
      CHARACTER (1) CHARAC
      LOGICAL :: LDUM
      INTEGER :: IDUM, IERR, I, I1MAX, I2MAX, I3MAX, I1, I2, I3, NI, NII, J, N
      REAL(q) :: RDUM
      COMPLEX(q)  :: CDUM

      ENERGY=0
!
! status== 2 immedate return
!
      IF ( STATUS == 2) RETURN
!
! status== 0 initialize (try reading PAIRPOT from INCAR file)
!
      IF ( STATUS == 0 ) THEN 
         CALL RDATAB(.TRUE.,INCAR,IU5,'PAIRPOT','=','#',';','F', &
     &     IDUM,TMP,CDUM,LDUM,CHARAC,N,NDIM,IERR)
         IF ((IERR/=0) .AND. (IERR/=3)) THEN
            WRITE(*,*)'Error reading item ''PAIRPOT'' from INCAR.'
            STOP
         ENDIF
         IF (IERR/=0 .OR. N==0) THEN
            STATUS=2
            RETURN
         ENDIF
         STATUS=1

         NPOINTS=N/2
         IF (IU6>0) THEN
            WRITE(IU6,"('found', I4,' pair potential data on INCAR')") NPOINTS
         ENDIF

         PSP(1:NPOINTS,1)=TMP(1:NPOINTS)
         PSP(1:NPOINTS,2)=TMP(NPOINTS+1:NPOINTS*2)
         
         APACO =PSP(NPOINTS,1)
         XMIN  =PSP(1,1)

         DO I=1,NPOINTS
            PSP(I,2)=PSP(I,2)-PSP(NPOINTS,2)
            PSP(I,1)=PSP(I,1)-XMIN
         ENDDO

         XMAX=PSP(NPOINTS,1)

         CALL SPLCOF(PSP(1,1),NPOINTS,NDIM,1E30_q)

         ARGSC=(NPOINTS-1)/XMAX

      ENDIF
!-----------------------------------------------------------------------
!  pressure is currently only correct for cubic cells
!-----------------------------------------------------------------------
      VOL= SQRT(A(1,1)**2+A(2,1)**2+A(3,1)**2)* &
     &     SQRT(A(1,2)**2+A(2,2)**2+A(3,2)**2)* &
     &     SQRT(A(1,3)**2+A(2,3)**2+A(3,3)**2)
      VOLP=VOLF
      VOLM=2.-VOLF
      VOLD=VOL*(VOLP-VOLM)
      VOLP=EXP(LOG(VOLP)/3.)
      VOLM=EXP(LOG(VOLM)/3.)
      AA1=ABS(A(1,1)*A(1,2)+A(2,1)*A(2,2)+A(3,1)*A(3,2))
      AA2=ABS(A(1,3)*A(1,2)+A(2,3)*A(2,2)+A(3,3)*A(3,2))
      AA3=ABS(A(1,1)*A(1,3)+A(2,1)*A(2,3)+A(3,1)*A(3,3))
      IF ( &
     &    (AA1.GT.0.001) .OR. &
     &    (AA2.GT.0.001) .OR. &
     &    (AA3.GT.0.001)) THEN
        IF (IU6.GT.0) WRITE(IU6,*) 'CELL IS NOT SC, PRESSURE CALCULATION INCORRECT'
      ENDIF

      ENERGY =0.
      ENERGYP=0.
      ENERGYM=0.
      
      ALLOCATE(CHECKF(3,NIONS))
      CHECKF=0
!-----------------------------------------------------------------------
!  IXMAX,Y,Z defines the maximum number of cells which
!  have to be used for the summation
!-----------------------------------------------------------------------
      I1MAX=AINT(APACO*BNORM(1)-.001)
      I2MAX=AINT(APACO*BNORM(2)-.001)
      I3MAX=AINT(APACO*BNORM(3)-.001)

      APACO2=APACO**2
!-----------------------------------------------------------------------
      DO I1=-I1MAX-1,I1MAX
      DO I2=-I2MAX-1,I2MAX
      DO I3=-I3MAX-1,I3MAX

      DO NI=1,NIONS
      DO NII=1,NIONS
         IF (NI==NII) CYCLE
         D1= I1+MOD(XC(1,NI)-XC(1,NII)+1,1._q)
         D2= I2+MOD(XC(2,NI)-XC(2,NII)+1,1._q)
         D3= I3+MOD(XC(3,NI)-XC(3,NII)+1,1._q)

         V(1)= D1*A(1,1)+D2*A(1,2)+D3*A(1,3)
         V(2)= D1*A(2,1)+D2*A(2,2)+D3*A(2,3)
         V(3)= D1*A(3,1)+D2*A(3,2)+D3*A(3,3)
         R2= V(1)**2 + V(2)**2 + V(3)**2

         IF (R2 > APACO2) CYCLE

         R2=SQRT(R2)
         R2P=R2*VOLP
         R2M=R2*VOLM

         DIST=R2
         DO J=1,3
            V(J)=V(J)/R2
         ENDDO

         CALL INTERP(R2 -XMIN,E ,DE ,PSP(1,1),NPOINTS,NDIM,ARGSC,IU6)
         CALL INTERP(R2P-XMIN,EP,DEP,PSP(1,1),NPOINTS,NDIM,ARGSC,IU6)
         CALL INTERP(R2M-XMIN,EM,DEM,PSP(1,1),NPOINTS,NDIM,ARGSC,IU6)
         ! double counting
         E =E /2.
         EP=EP/2.
         EM=EM/2.
!       WRITE (*,*) 'OO',R2,APACO2,ENERGY,I
!       WRITE (*,*) 'NI',NI,NII,' DE',DE,' DIST',DIST,' V1',V

         ENERGY =ENERGY +E
         ENERGYP=ENERGYP+EP
         ENERGYM=ENERGYM+EM
       
         DO J=1,3
            FORCES(J,NI) =FORCES(J,NI) -V(J)*DE
            CHECKF(J,NI) =CHECKF(J,NI) -V(J)*DE
         ENDDO

      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      PRESS=(ENERGYM-ENERGYP)/VOLD
!     EV/A**3 = 1.6022E-19 M^2 KG / S^2 / A**3 =
!               1.6022E-19 M^3 PA / A**3 =
!               1.6022E-19 M^3 BAR / A**3 / 100 000 =
!               1.6022E-19 BAR 10^30 / 100 000 = 1.6022E6 BAR

      IF (IU6>=0) THEN
         WRITE(IU6,*) 'PAIR: correction to pressure:',PRESS * 1.6022E3,'KB'
         WRITE(IU6,*) 'energy correction',ENERGY
         WRITE(IU6,*) 'force corrections'
         DO J=1,NIONS
            WRITE(IU6,*) CHECKF(1,J),CHECKF(2,J),CHECKF(3,J)
         ENDDO
      ENDIF

      DEALLOCATE (CHECKF)

      RETURN
    END SUBROUTINE PAIR_CORRECTION

!=======================================================================
!
!  interpolate the array PSP using evaluated spline coefficients
!
!=======================================================================

    SUBROUTINE INTERP(RIN,E,DE,PSP,NPOINTS,NDIM,ARGSC,IU6)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION PSP(NDIM,5)
      INTEGER :: NPOINTS, NDIM, IU6, I

      R2=RIN
      IF (R2 >= PSP(NPOINTS,1)) THEN
         E=0.
         DE=0.
         RETURN
      ENDIF
        
      IF (R2.LE.0) THEN
         IF (IU6.GT.0) WRITE(IU6,*) 'PAIR, WARNING: R2 TOO SMALL'
         R2=0.
      ENDIF
      I  =MAX(INT(R2*ARGSC)+1,1)
      REM=R2-PSP(I,1)
      E  =(PSP(I,2)+REM*(PSP(I,3)+ &
           &                REM*(PSP(I,4)  +REM*PSP(I,5))))
      DE = PSP(I,3)+REM*(PSP(I,4)*2+REM*PSP(I,5)*3)
      RETURN
    END SUBROUTINE INTERP
    

!*********************************************************************
!
! The following module implements an additional
! force field on the atoms ala Pederson et. al
!
!*********************************************************************


    

  END MODULE classicfields
