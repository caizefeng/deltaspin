#include "symbol.inc"
MODULE Constrained_M_modular

    USE prec

    INTEGER, PRIVATE, SAVE :: I_CONSTRAINED_M           ! Type of constraining
    INTEGER, PRIVATE, SAVE :: NTYP                      ! number of types
    INTEGER, PRIVATE, SAVE :: NIONS                     ! number of ions
    INTEGER, PRIVATE, SAVE :: IRMAX                     ! maximum number points in sphere
    INTEGER, PRIVATE, ALLOCATABLE, SAVE :: NITYP(:)      ! number of ions for each type
    INTEGER, PRIVATE, ALLOCATABLE, SAVE :: NLIMAX(:)     ! maximum index for each ion
    INTEGER, PRIVATE, ALLOCATABLE, SAVE :: NLI(:, :)      ! index for gridpoints

    REAL(q), PRIVATE, SAVE :: LAMBDA                    ! penalty factor
    REAL(q), PRIVATE, SAVE :: E_PENALTY                 ! constraint energy
    REAL(q), PRIVATE, SAVE :: E_EXPECT
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: RWIGS(:)      ! real space cutoff
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: POSION(:, :)   ! positions (required for setup)
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: WEIGHT(:, :)   ! weights for M(r)
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: X(:, :), Y(:, :), Z(:, :)
    REAL(q), PRIVATE, SAVE :: A(3, 3), B(3, 3)

    REAL(q), PRIVATE, PARAMETER :: TINY = 1E-5_q

    REAL(q), ALLOCATABLE, SAVE :: M_TOT(:, :)        ! Total moments
    REAL(q), ALLOCATABLE, SAVE :: MW(:, :)       ! Total weighed moments
    REAL(q), ALLOCATABLE, SAVE :: M_CONSTR(:, :) ! constraints on M per ion

    ! SASC declaration
    REAL(q), ALLOCATABLE :: L_CONSTR(:, :) ! penalty factor for dynamical optimization
    INTEGER, SAVE :: SCTYPE = 0 ! 0: noSASC 1:SASC(L) 2:SASC(Q) 3:SASC(Q+L)
    INTEGER, SAVE :: SCTYPE_CURRENT = 0

    ! SASC(L)
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: LAMBDA_LINE(:) ! penalty factor for dynamical optimization
    REAL(q), ALLOCATABLE :: L_CONSTR_L(:, :) ! penalty factor for dynamical optimization
    REAL(q), ALLOCATABLE :: L_CONSTR_L_DIAG(:, :) ! old nu to maintain the diag  
    REAL(q), ALLOCATABLE :: L_CONSTR_L_ADD(:, :) ! delta nu to add to H to create new diag question
    INTEGER, PRIVATE, ALLOCATABLE, SAVE :: CONSTRL_LINE(:)
    INTEGER, ALLOCATABLE, SAVE :: CONSTRL(:, :)
    INTEGER, SAVE :: ALGO_SC, ALGO_SC_DIAG
    INTEGER, SAVE :: NELM_SC_INITIAL, NELM_SC_INTER
    INTEGER, SAVE :: CONSTR_NUM_STEP
    INTEGER, SAVE :: SCDECOUPLE
    REAL(q), SAVE :: CONSTR_EPSILON
    REAL(q), SAVE :: INI_SC_ALPHA
    REAL(q), SAVE :: CONSTR_RESTRICT
    REAL(q), SAVE :: DECAY_EPSILON
    REAL(q), SAVE :: LBOUND_EPSILON
    REAL(q), SAVE :: CONV_BOUND
    LOGICAL, SAVE :: DEBUG_SC
    LOGICAL, SAVE :: TRIAL_UPDATE_RESTRICT

    ! SASC(Q)
    REAL(q), PRIVATE, ALLOCATABLE, SAVE :: LAMBDA_LINE_Q(:) ! penalty factor for dynamical optimization
    REAL(q), ALLOCATABLE :: L_CONSTR_Q(:, :) ! penalty factor for dynamical optimization
    INTEGER, PRIVATE, ALLOCATABLE, SAVE :: CONSTRL_LINE_Q(:)
    INTEGER, ALLOCATABLE, SAVE :: CONSTRL_Q(:, :)
    INTEGER, SAVE :: CONSTR_NUM_STEP_Q
    REAL(q), SAVE :: CONSTR_EPSILON_Q
    REAL(q), SAVE :: INI_SC_ALPHA_Q
    REAL(q), SAVE :: EDIFF_Q
    LOGICAL, SAVE :: DEBUG_SC_Q
CONTAINS

!**********************************************************************
!
! subroutine for performing self-adaptive spin constrain in VASP
!
!**********************************************************************

#include "deltaspin/constrained_m_reader.f90"
#include "deltaspin/e_constraint.f90"
#include "deltaspin/adpt_constraining_pot.f90"
#include "deltaspin/write_constrained_m.f90"

!=======================================================================
!
! write the parameters of the constrained moment calculations
!
!=======================================================================

    SUBROUTINE XML_WRITE_CONSTRAINED_M(NIONS_)

        USE vaspxml

        INTEGER :: NIONS_
        LOGICAL :: LDUM
        COMPLEX(q) :: CDUM
        CHARACTER(1) :: CHARAC

        CALL XML_INCAR('I_CONSTRAINED_M', 'I', I_CONSTRAINED_M, RDUM, CDUM, LDUM, CHARAC, 1)
! if I_CONSTRAINED_M<>0 we also need M_CONSTR and (possibly) LAMBDA
        IF (I_CONSTRAINED_M <= 0) RETURN

        CALL XML_INCAR_V('M_CONSTR', 'F', IDUM, AM_CONSTR, CDUM, LDUM, CHARAC, NIONS_*3)
        ! CALL XML_INCAR('LAMBDA','F',IDUM,LAMBDA,CDUM,LDUM,CHARAC,1)
        ! CALL XML_INCAR_V('LAMBDA', 'F', IDUM, LAMBDA_LINE, CDUM, LDUM, CHARAC, NIONS_*3)
    END SUBROUTINE

    SUBROUTINE CONSTRAINED_M_INIT(T_INFO, GRIDC, LATT_CUR)

        USE constant
        USE poscar
        USE mgrid
        USE lattice

        IMPLICIT COMPLEX(q) (C)
        IMPLICIT REAL(q) (A - B, D - H, O - Z)

        TYPE(type_info) T_INFO
        TYPE(grid_3d) GRIDC
        TYPE(latt) LATT_CUR

        IF (I_CONSTRAINED_M > 0) THEN
!=======================================================================
! If we want to do constrained moment calculations
! we will now have to initialize some stuff
!=======================================================================

            NTYP = T_INFO%NTYP
            ALLOCATE (NITYP(NTYP), POSION(3, NIONS), RWIGS(NTYP))

            NITYP = T_INFO%NITYP
            POSION = T_INFO%POSION
            RWIGS = T_INFO%RWIGS

            A = LATT_CUR%A
            B = LATT_CUR%B

!=======================================================================
            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 2 .OR. I_CONSTRAINED_M == 3) THEN
!=======================================================================
                NIS = 1

                type1: DO NT = 1, NTYP
                ions1: DO NI = NIS, NITYP(NT) + NIS - 1

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
                    F1 = 1._q/GRIDC%NGX
                    F2 = 1._q/GRIDC%NGY
                    F3 = 1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
                    D1 = RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
                    D2 = RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
                    D3 = RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

                    N3LOW = INT(POSION(3, NI)*GRIDC%NGZ - D3 + 10*GRIDC%NGZ + .99_q) - 10*GRIDC%NGZ
                    N2LOW = INT(POSION(2, NI)*GRIDC%NGY - D2 + 10*GRIDC%NGY + .99_q) - 10*GRIDC%NGY
                    N1LOW = INT(POSION(1, NI)*GRIDC%NGX - D1 + 10*GRIDC%NGX + .99_q) - 10*GRIDC%NGX

                    N3HI = INT(POSION(3, NI)*GRIDC%NGZ + D3 + 10*GRIDC%NGZ) - 10*GRIDC%NGZ
                    N2HI = INT(POSION(2, NI)*GRIDC%NGY + D2 + 10*GRIDC%NGY) - 10*GRIDC%NGY
                    N1HI = INT(POSION(1, NI)*GRIDC%NGX + D1 + 10*GRIDC%NGX) - 10*GRIDC%NGX

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
                    IND = 1

                    DO N2 = N2LOW, N2HI
                        X2 = (N2*F2 - POSION(2, NI))
                        N2P = MOD(N2 + 10*GRIDC%NGY, GRIDC%NGY)

                        DO N1 = N1LOW, N1HI
                            X1 = (N1*F1 - POSION(1, NI))
                            N1P = MOD(N1 + 10*GRIDC%NGX, GRIDC%NGX)

                            NCOL = GRIDC%RL%INDEX(N1P, N2P)
                            IF (NCOL == 0) CYCLE ! not on local node go on
                            IF (GRIDC%RL%I2(NCOL) /= N1P + 1 .OR. GRIDC%RL%I3(NCOL) /= N2P + 1) THEN
                                WRITE (*, *) 'STEPFNC_ALLOC: internal ERROR:', &
                                    GRIDC%RL%I2(NCOL), N1P + 1, GRIDC%RL%I3(NCOL), N2P + 1
                                STOP
                            END IF
!OCL SCALAR
                            DO N3 = N3LOW, N3HI
                                X3 = (N3*F3 - POSION(3, NI))

                                XC = X1*LATT_CUR%A(1, 1) + X2*LATT_CUR%A(1, 2) + X3*LATT_CUR%A(1, 3)
                                YC = X1*LATT_CUR%A(2, 1) + X2*LATT_CUR%A(2, 2) + X3*LATT_CUR%A(2, 3)
                                ZC = X1*LATT_CUR%A(3, 1) + X2*LATT_CUR%A(3, 2) + X3*LATT_CUR%A(3, 3)

                                D = SQRT(XC*XC + YC*YC + ZC*ZC)

                                IF (D <= RWIGS(NT)) IND = IND + 1

                            END DO
                        END DO
                    END DO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
                    IND = 1
                    DO N3 = N3LOW, N3HI
                        X3 = (N3*F3 - POSION(3, NI))
                        N3P = MOD(N3 + 10*GRIDC%NGZ, GRIDC%NGZ)

                        DO N2 = N2LOW, N2HI
                            X2 = (N2*F2 - POSION(2, NI))
                            N2P = MOD(N2 + 10*GRIDC%NGY, GRIDC%NGY)

                            NCOL = GRIDC%RL%INDEX(N2P, N3P)

                            DO N1 = N1LOW, N1HI
                                X1 = (N1*F1 - POSION(1, NI))

                                XC = X1*LATT_CUR%A(1, 1) + X2*LATT_CUR%A(1, 2) + X3*LATT_CUR%A(1, 3)
                                YC = X1*LATT_CUR%A(2, 1) + X2*LATT_CUR%A(2, 2) + X3*LATT_CUR%A(2, 3)
                                ZC = X1*LATT_CUR%A(3, 1) + X2*LATT_CUR%A(3, 2) + X3*LATT_CUR%A(3, 3)

                                D = SQRT(XC*XC + YC*YC + ZC*ZC)

                                IF (D <= RWIGS(NT)) IND = IND + 1

                            END DO
                        END DO
                    END DO
#endif
!-----------------------------------------------------------------------
!  determine IRMAX
!-----------------------------------------------------------------------

                    IRMAX = MAX(IRMAX, IND)

                END DO ions1
                NIS = NIS + NITYP(NT)
                END DO type1

!-----------------------------------------------------------------------
!  and allocate
!-----------------------------------------------------------------------

                ALLOCATE (NLIMAX(NIONS), NLI(IRMAX, NIONS), WEIGHT(IRMAX, NIONS), &
                          M_TOT(3, NIONS), MW(3, NIONS), X(IRMAX, NIONS), Y(IRMAX, NIONS), Z(IRMAX, NIONS))

!=======================================================================
! Now we will fill these nice arrays
!=======================================================================
                NIS = 1

                type2: DO NT = 1, NTYP
                ions2: DO NI = NIS, NITYP(NT) + NIS - 1

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
                    F1 = 1._q/GRIDC%NGX
                    F2 = 1._q/GRIDC%NGY
                    F3 = 1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
                    D1 = RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
                    D2 = RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
                    D3 = RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

                    N3LOW = INT(POSION(3, NI)*GRIDC%NGZ - D3 + 10*GRIDC%NGZ + .99_q) - 10*GRIDC%NGZ
                    N2LOW = INT(POSION(2, NI)*GRIDC%NGY - D2 + 10*GRIDC%NGY + .99_q) - 10*GRIDC%NGY
                    N1LOW = INT(POSION(1, NI)*GRIDC%NGX - D1 + 10*GRIDC%NGX + .99_q) - 10*GRIDC%NGX

                    N3HI = INT(POSION(3, NI)*GRIDC%NGZ + D3 + 10*GRIDC%NGZ) - 10*GRIDC%NGZ
                    N2HI = INT(POSION(2, NI)*GRIDC%NGY + D2 + 10*GRIDC%NGY) - 10*GRIDC%NGY
                    N1HI = INT(POSION(1, NI)*GRIDC%NGX + D1 + 10*GRIDC%NGX) - 10*GRIDC%NGX

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
                    IND = 1

                    DO N2 = N2LOW, N2HI
                        X2 = (N2*F2 - POSION(2, NI))
                        N2P = MOD(N2 + 10*GRIDC%NGY, GRIDC%NGY)

                        DO N1 = N1LOW, N1HI
                            X1 = (N1*F1 - POSION(1, NI))
                            N1P = MOD(N1 + 10*GRIDC%NGX, GRIDC%NGX)

                            NCOL = GRIDC%RL%INDEX(N1P, N2P)
                            IF (NCOL == 0) CYCLE ! not on local node go on
!OCL SCALAR
                            DO N3 = N3LOW, N3HI
                                X3 = (N3*F3 - POSION(3, NI))

                                XC = X1*LATT_CUR%A(1, 1) + X2*LATT_CUR%A(1, 2) + X3*LATT_CUR%A(1, 3)
                                YC = X1*LATT_CUR%A(2, 1) + X2*LATT_CUR%A(2, 2) + X3*LATT_CUR%A(2, 3)
                                ZC = X1*LATT_CUR%A(3, 1) + X2*LATT_CUR%A(3, 2) + X3*LATT_CUR%A(3, 3)

                                D = SQRT(XC*XC + YC*YC + ZC*ZC)

                                IF (D <= RWIGS(NT)) THEN

                                    N3P = MOD(N3 + 10*GRIDC%NGZ, GRIDC%NGZ)
                                    NLI(IND, NI) = 1 + N3P + GRIDC%NGZ*(NCOL - 1)

                                    QR = D*PI/RWIGS(NT)
                                    CALL SBESSEL(QR, BJ, 0)
                                    WEIGHT(IND, NI) = BJ

                                    X(IND, NI) = X1 + POSION(1, NI)
                                    Y(IND, NI) = X2 + POSION(2, NI)
                                    Z(IND, NI) = X3 + POSION(3, NI)

                                    IND = IND + 1
                                END IF
                            END DO
                        END DO
                    END DO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
                    IND = 1
                    DO N3 = N3LOW, N3HI
                        X3 = (N3*F3 - POSION(3, NI))
                        N3P = MOD(N3 + 10*GRIDC%NGZ, GRIDC%NGZ)

                        DO N2 = N2LOW, N2HI
                            X2 = (N2*F2 - POSION(2, NI))
                            N2P = MOD(N2 + 10*GRIDC%NGY, GRIDC%NGY)

                            NCOL = GRIDC%RL%INDEX(N2P, N3P)

                            DO N1 = N1LOW, N1HI
                                X1 = (N1*F1 - POSION(1, NI))

                                XC = X1*LATT_CUR%A(1, 1) + X2*LATT_CUR%A(1, 2) + X3*LATT_CUR%A(1, 3)
                                YC = X1*LATT_CUR%A(2, 1) + X2*LATT_CUR%A(2, 2) + X3*LATT_CUR%A(2, 3)
                                ZC = X1*LATT_CUR%A(3, 1) + X2*LATT_CUR%A(3, 2) + X3*LATT_CUR%A(3, 3)

                                D = SQRT(XC*XC + YC*YC + ZC*ZC)

                                IF (D <= RWIGS(NT)) THEN

                                    N1P = MOD(N1 + 10*GRIDC%NGX, GRIDC%NGX)
                                    NLI(IND, NI) = N1P + (NCOL - 1)*GRIDC%NGX + 1

                                    QR = D*PI/RWIGS(NT)
                                    CALL SBESSEL(QR, BJ, 0)
                                    WEIGHT(IND, NI) = BJ

                                    X(IND, NI) = X1 + POSION(1, NI)
                                    Y(IND, NI) = X2 + POSION(2, NI)
                                    Z(IND, NI) = X3 + POSION(3, NI)

                                    IND = IND + 1
                                END IF
                            END DO
                        END DO
                    END DO
#endif

                    INDMAX = IND - 1

                    NLIMAX(NI) = INDMAX

                END DO ions2
                NIS = NIS + NITYP(NT)
                END DO type2

!=======================================================================
            END IF
!=======================================================================

!=======================================================================
! We are done initializing
!=======================================================================

        END IF
        RETURN
    END SUBROUTINE CONSTRAINED_M_INIT

!***********************************************************************
!
! function to query whether we want to constrain the magnetic moments
!
!***********************************************************************

    FUNCTION M_CONSTRAINED()
        IMPLICIT NONE
        LOGICAL M_CONSTRAINED

        IF (I_CONSTRAINED_M > 0) THEN
            M_CONSTRAINED = .TRUE.
        ELSE
            M_CONSTRAINED = .FALSE.
        END IF

    END FUNCTION M_CONSTRAINED

!************************ SUBROUTINE M_INT  ****************************
!
! requires total charge as (charge,magnetization) in real space
!
!***********************************************************************

    SUBROUTINE M_INT(CHTOT, GRIDC, WDES)

        USE mpimy
        USE prec
        USE mgrid
        USE wave
        USE constant

        IMPLICIT COMPLEX(q) (C)
        IMPLICIT REAL(q) (A - B, D - H, O - Z)

        TYPE(wavedes) WDES
        TYPE(grid_3d) GRIDC

        RGRID CHTOT(DIMREAL(GRIDC%MPLWV), WDES%NCDIJ)

        spin: DO ISP = 2, WDES%NCDIJ

            NIS = 1

            ion_types: DO NT = 1, NTYP
            ions: DO NI = NIS, NITYP(NT) + NIS - 1

                SUM = 0
                SUM_WEIGHTED = 0
                RINPL = 1._q/GRIDC%NPLWV

                SELECT CASE (ISP)
! M_x
                CASE (2)
                    DO IND = 1, NLIMAX(NI)
                        QR = TPI*(WDES%QSPIRAL(1)*X(IND, NI) + &
                                  WDES%QSPIRAL(2)*Y(IND, NI) + &
                                  WDES%QSPIRAL(3)*Z(IND, NI))

                        SUM = SUM + RINPL*(COS(QR)*REAL(CHTOT(NLI(IND, NI), 2)) - &
                                           SIN(QR)*REAL(CHTOT(NLI(IND, NI), 3)))

                        SUM_WEIGHTED = SUM_WEIGHTED + RINPL*(COS(QR)*REAL(CHTOT(NLI(IND, NI), 2)) - &
                                                             SIN(QR)*REAL(CHTOT(NLI(IND, NI), 3)))*WEIGHT(IND, NI)
                    END DO
! M_y
                CASE (3)
                    DO IND = 1, NLIMAX(NI)
                        QR = TPI*(WDES%QSPIRAL(1)*X(IND, NI) + &
                                  WDES%QSPIRAL(2)*Y(IND, NI) + &
                                  WDES%QSPIRAL(3)*Z(IND, NI))

                        SUM = SUM + RINPL*(COS(QR)*REAL(CHTOT(NLI(IND, NI), 3)) + &
                                           SIN(QR)*REAL(CHTOT(NLI(IND, NI), 2)))

                        SUM_WEIGHTED = SUM_WEIGHTED + RINPL*(COS(QR)*REAL(CHTOT(NLI(IND, NI), 3)) + &
                                                             SIN(QR)*REAL(CHTOT(NLI(IND, NI), 2)))*WEIGHT(IND, NI)
                    END DO
! M_z
                CASE (4)
                    DO IND = 1, NLIMAX(NI)
                        SUM = SUM + RINPL*REAL(CHTOT(NLI(IND, NI), 4))
                        SUM_WEIGHTED = SUM_WEIGHTED + REAL(CHTOT(NLI(IND, NI), 4))* &
                                       WEIGHT(IND, NI)*RINPL
                    END DO

                END SELECT

                M_TOT(ISP - 1, NI) = SUM
                MW(ISP - 1, NI) = SUM_WEIGHTED
                !IF (I_CONSTRAINED_M==2) MW(ISP-1,NI)=M_TOT(ISP-1,NI)

            END DO ions
            NIS = NIS + NITYP(NT)
            END DO ion_types

        END DO spin

        CALLMPI(M_sum_d(GRIDC%COMM, M_TOT(1, 1), (WDES%NCDIJ - 1)*NIONS))
        CALLMPI(M_sum_d(GRIDC%COMM, MW(1, 1), (WDES%NCDIJ - 1)*NIONS))

        RETURN
    END SUBROUTINE

!************************ SUBROUTINE ADD_CONSTRAINING_POT **************
!
! expects CVTOT in (charge,magnetization) convention in real space
!
!***********************************************************************

    SUBROUTINE ADD_CONSTRAINING_POT(CVTOT, GRIDC, WDES)

        USE prec
        USE constant
        USE mgrid
        USE wave

        IMPLICIT COMPLEX(q) (C)
        IMPLICIT REAL(q) (A - B, D - H, O - Z)

        TYPE(wavedes) WDES
        TYPE(grid_3d) GRIDC

        REAL(q) MW_, MW_X, MW_Y, MW_IN_M_CONSTR

        RGRID CVTOT(DIMREAL(GRIDC%MPLWV), WDES%NCDIJ)

        spin: DO ISP = 2, WDES%NCDIJ

            NIS = 1

            ion_types: DO NT = 1, NTYP
            ions: DO NI = NIS, NITYP(NT) + NIS - 1

                IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
                    ABS(M_CONSTR(2, NI)) < TINY .AND. &
                    ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

                MW_IN_M_CONSTR = MW(1, NI)*M_CONSTR(1, NI) + &
                                 MW(2, NI)*M_CONSTR(2, NI) + &
                                 MW(3, NI)*M_CONSTR(3, NI)

                SELECT CASE (ISP)
! M_x
                CASE (2)

                    IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                        MW_X = MW(1, NI)
                        MW_X = MW_X - M_CONSTR(1, NI)*MW_IN_M_CONSTR

                        MW_Y = MW(2, NI)
                        MW_Y = MW_Y - M_CONSTR(2, NI)*MW_IN_M_CONSTR
                    END IF

                    IF (I_CONSTRAINED_M == 2) THEN
                        MW_X = MW(1, NI) - M_CONSTR(1, NI)
                        MW_Y = MW(2, NI) - M_CONSTR(2, NI)
                    END IF

                    DO IND = 1, NLIMAX(NI)
                        QR = -TPI*(WDES%QSPIRAL(1)*X(IND, NI) + &
                                   WDES%QSPIRAL(2)*Y(IND, NI) + &
                                   WDES%QSPIRAL(3)*Z(IND, NI))                             ! BenXu noted QSPIRAL = 0.0 initially

                        CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                   2*LAMBDA*WEIGHT(IND, NI)*(MW_X*COS(QR) - MW_Y*SIN(QR))
                    END DO
! M_y
                CASE (3)

                    IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                        MW_X = MW(1, NI)
                        MW_X = MW_X - M_CONSTR(1, NI)*MW_IN_M_CONSTR

                        MW_Y = MW(2, NI)
                        MW_Y = MW_Y - M_CONSTR(2, NI)*MW_IN_M_CONSTR
                    END IF

                    IF (I_CONSTRAINED_M == 2) THEN
                        MW_X = MW(1, NI) - M_CONSTR(1, NI)
                        MW_Y = MW(2, NI) - M_CONSTR(2, NI)
                    END IF

                    DO IND = 1, NLIMAX(NI)
                        QR = -TPI*(WDES%QSPIRAL(1)*X(IND, NI) + &
                                   WDES%QSPIRAL(2)*Y(IND, NI) + &
                                   WDES%QSPIRAL(3)*Z(IND, NI))

                        CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                   2*LAMBDA*WEIGHT(IND, NI)*(MW_Y*COS(QR) + MW_X*SIN(QR))
                    END DO
! M_z
                CASE (4)

                    IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                        MW_ = MW(ISP - 1, NI)
                        MW_ = MW_ - M_CONSTR(ISP - 1, NI)*MW_IN_M_CONSTR
                    END IF

                    IF (I_CONSTRAINED_M == 2) THEN
                        MW_ = MW(ISP - 1, NI) - M_CONSTR(ISP - 1, NI)
                    END IF

                    DO IND = 1, NLIMAX(NI)
                        CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                   2*LAMBDA*WEIGHT(IND, NI)*MW_
                    END DO

                END SELECT

            END DO ions
            NIS = NIS + NITYP(NT)
            END DO ion_types

        END DO spin

        RETURN
    END SUBROUTINE ADD_CONSTRAINING_POT


    SUBROUTINE CHECK_SPIRAL_ENCUT(WDES, INFO, LATT_CUR, IO)
        USE prec
        USE base
        USE wave
        USE lattice
        USE constant
        IMPLICIT NONE
        TYPE(wavedes) WDES
        TYPE(info_struct) INFO
        TYPE(latt) LATT_CUR
        TYPE(in_struct) IO

        REAL(q) GMAX
        REAL(q) QX, QY, QZ, QNORM
        REAL(q) ENMAX

        REAL(q) RTUT(3)
        INTEGER IDUM
        COMPLEX(q) CDUM
        LOGICAL LDUM

        GMAX = SQRT(INFO%ENINI/(4*PI*PI*HSQDTM))

        QX = ((WDES%QSPIRAL(1)/2)*LATT_CUR%B(1, 1) + &
       &      (WDES%QSPIRAL(2)/2)*LATT_CUR%B(1, 2) + &
       &       (WDES%QSPIRAL(3)/2)*LATT_CUR%B(1, 3))
        QY = ((WDES%QSPIRAL(1)/2)*LATT_CUR%B(2, 1) + &
       &      (WDES%QSPIRAL(2)/2)*LATT_CUR%B(2, 2) + &
       &       (WDES%QSPIRAL(3)/2)*LATT_CUR%B(2, 3))
        QZ = ((WDES%QSPIRAL(1)/2)*LATT_CUR%B(3, 1) + &
       &      (WDES%QSPIRAL(2)/2)*LATT_CUR%B(3, 2) + &
       &       (WDES%QSPIRAL(3)/2)*LATT_CUR%B(3, 3))

        QNORM = SQRT(QX*QX + QY*QY + QZ*QZ)

        ENMAX = 4*PI*PI*HSQDTM*(GMAX + QNORM)**2

        RTUT(1) = INFO%ENINI
        RTUT(2) = INFO%ENMAX
        RTUT(3) = ENMAX

        CALL VTUTOR('W', 'SPIN SPIRAL', RTUT, 3, IDUM, 1, CDUM, 1, LDUM, 1, IO%IU0, 2)

        RETURN
    END SUBROUTINE CHECK_SPIRAL_ENCUT

END MODULE Constrained_M_modular
