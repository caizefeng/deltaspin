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
    REAL(q), SAVE :: CONSTR_EPSILON
    REAL(q), SAVE :: INI_SC_ALPHA
    REAL(q), SAVE :: CONSTR_RESTRICT
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
    LOGICAL, SAVE :: DEBUG_SC_Q
CONTAINS
!=======================================================================
!
! initialise the constrained moment reader
!
!=======================================================================

    SUBROUTINE CONSTRAINED_M_READER(T_INFO, WDES, IU0, IU5)
        USE base
        USE wave
        USE poscar
        USE vaspxml
        USE constant

        TYPE(wavedes) WDES
        TYPE(type_info) T_INFO

        INTEGER IU0, IU5
        LOGICAL :: LOPEN, LDUM
        REAL(q) :: MNORM
        REAL(q) :: ALPHA, BETA
        REAL(q) :: QR, M_x, M_y
        REAL(q), ALLOCATABLE :: AM_CONSTR(:)
        COMPLEX(q) :: CDUM
        CHARACTER(1) :: CHARAC

        NIONS = T_INFO%NIONS

        LOPEN = .FALSE.
        OPEN (UNIT=IU5, FILE=INCAR, STATUS='OLD')

        I_CONSTRAINED_M = 0
        CALL RDATAB(LOPEN, INCAR, IU5, 'I_CONSTRAINED_M', '=', '#', ';', 'I', &
       &            I_CONSTRAINED_M, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
        IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
       &                    ((IERR == 0) .AND. (N < 1))) THEN
            IF (IU0 >= 0) &
                WRITE (IU0, *) 'Error reading item ''I_CONSTRAINED_M'' from file INCAR.'
            GOTO 150
        END IF
        CALL XML_INCAR('I_CONSTRAINED_M', 'I', I_CONSTRAINED_M, RDUM, CDUM, LDUM, CHARAC, N)
! if I_CONSTRAINED_M<>0 we also need M_CONSTR and (possibly) LAMBDA
        IF (I_CONSTRAINED_M > 0) THEN
            NMCONSTR = 3*NIONS
            ALLOCATE (AM_CONSTR(NMCONSTR), M_CONSTR(3, NIONS))
            ALLOCATE (L_CONSTR(3, NIONS))
            M_CONSTR = 0
! ... get constraints
            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 2) THEN
                ! constraints are read in as vectors
                CALL RDATAB(LOPEN, INCAR, IU5, 'M_CONSTR', '=', '#', ';', 'F', &
         &               IDUM, AM_CONSTR, CDUM, LDUM, CHARAC, N, NMCONSTR, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
         &                       ((IERR == 0) .AND. (N < NMCONSTR))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''M_CONSTR'' from file INCAR.'
                    GOTO 150
                END IF
                CALL XML_INCAR_V('M_CONSTR', 'F', IDUM, AM_CONSTR, CDUM, LDUM, CHARAC, N)

                DO NI = 1, NIONS
                    M_CONSTR(1, NI) = AM_CONSTR(3*(NI - 1) + 1)
                    M_CONSTR(2, NI) = AM_CONSTR(3*(NI - 1) + 2)
                    M_CONSTR(3, NI) = AM_CONSTR(3*(NI - 1) + 3)
                    IF (I_CONSTRAINED_M == 1) THEN
                        ! constraining vectors set to have unit length
                        MNORM = SQRT(M_CONSTR(1, NI)*M_CONSTR(1, NI) + &
                                     M_CONSTR(2, NI)*M_CONSTR(2, NI) + &
                                     M_CONSTR(3, NI)*M_CONSTR(3, NI))
                        MNORM = MAX(MNORM, TINY)
                        M_CONSTR(1:3, NI) = M_CONSTR(1:3, NI)/MNORM
                    END IF
                END DO
                DEALLOCATE (AM_CONSTR)
            ELSEIF (I_CONSTRAINED_M == 3) THEN
                ! constraints are read in as angles
                CALL RDATAB(LOPEN, INCAR, IU5, 'M_CONSTR', '=', '#', ';', 'F', &
         &               IDUM, AM_CONSTR, CDUM, LDUM, CHARAC, N, 2*NIONS, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
         &                       ((IERR == 0) .AND. (N < 2*NIONS))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''M_CONSTR'' from file INCAR.'
                    GOTO 150
                END IF
                CALL XML_INCAR_V('M_CONSTR', 'F', IDUM, AM_CONSTR, CDUM, LDUM, CHARAC, N)
                ! construct the constraining vectors
                DO NI = 1, NIONS
                    ALPHA = TPI*AM_CONSTR(2*(NI - 1) + 1)/360._q
                    BETA = TPI*AM_CONSTR(2*(NI - 1) + 2)/360._q
                    IF (ALPHA < 0 .OR. BETA < 0) CYCLE
                    M_CONSTR(1, NI) = COS(ALPHA)*SIN(BETA)
                    M_CONSTR(2, NI) = SIN(ALPHA)*SIN(BETA)
                    M_CONSTR(3, NI) = COS(BETA)
                    write (*, *) 'constr=', ni, m_constr(1:3, ni)
                    ! and apply a possible spiral
                    QR = TPI*(WDES%QSPIRAL(1)*T_INFO%POSION(1, NI) + &
                   &           WDES%QSPIRAL(2)*T_INFO%POSION(2, NI) + &
                   &              WDES%QSPIRAL(3)*T_INFO%POSION(3, NI))
                    M_x = M_CONSTR(1, NI)*COS(QR) - M_CONSTR(2, NI)*SIN(QR)
                    M_y = M_CONSTR(2, NI)*COS(QR) + M_CONSTR(1, NI)*SIN(QR)
                    M_CONSTR(1, NI) = M_x
                    M_CONSTR(2, NI) = M_y
                    write (*, *) 'constr+spir=', ni, m_constr(1:3, ni)
                END DO
                DEALLOCATE (AM_CONSTR)
            END IF

! =====================================
!  SASC parameters
! =====================================
            SCTYPE = 0 ! 0: noSASC 1:SASC(L) 2:SASC(Q) 3:SASC(Q+L)
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCTYPE', '=', '#', ';', 'I', &
           &   SCTYPE, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCTYPE'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            IF (SCTYPE == 0) THEN
                ALLOCATE (CONSTRL_LINE(NMCONSTR))
                ALLOCATE (CONSTRL(3, NIONS))

                LAMBDA=0
                CALL RDATAB(LOPEN,INCAR,IU5,'LAMBDA','=','#',';','F', &
            &               IDUM,LAMBDA,CDUM,LDUM,CHARAC,N,1,IERR)
                IF (((IERR/=0).AND.(IERR/=3)).OR. &
            &                       ((IERR==0).AND.(N<1))) THEN
                    IF (IU0>=0) &
                    WRITE(IU0,*)'Error reading item ''LAMBDA'' from file INCAR.'
                    GOTO 150
                ENDIF
                CALL XML_INCAR('LAMBDA','F',IDUM,LAMBDA,CDUM,LDUM,CHARAC,N)

                CONSTRL_LINE = 1
                CONSTRL = reshape(CONSTRL_LINE, (/3, NIONS/))
            ENDIF


            IF (SCTYPE == 1 .OR. SCTYPE == 3) THEN
                ALLOCATE (LAMBDA_LINE(NMCONSTR), CONSTRL_LINE(NMCONSTR))
                ALLOCATE (L_CONSTR_L(3, NIONS), CONSTRL(3, NIONS))
                ALLOCATE (L_CONSTR_L_DIAG(3, NIONS), L_CONSTR_L_ADD(3, NIONS))
! ... get penalty factor
                LAMBDA_LINE = 0
!         CALL RDATAB(LOPEN,INCAR,IU5,'LAMBDA','=','#',';','F', &
!     &               IDUM,LAMBDA,CDUM,LDUM,CHARAC,N,NMCONSTR,IERR)
                CALL RDATAB(LOPEN, INCAR, IU5, 'LAMBDA', '=', '#', ';', 'F', &
             &               IDUM, LAMBDA_LINE, CDUM, LDUM, CHARAC, N, NMCONSTR, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
             &                       ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''LAMBDA'' from file INCAR.'
                    GOTO 150
                END IF
                L_CONSTR_L = reshape(LAMBDA_LINE, (/3, NIONS/))

                CONSTRL_LINE = 1
                CALL RDATAB(LOPEN, INCAR, IU5, 'CONSTRL', '=', '#', ';', 'I', &
               &   CONSTRL_LINE, RDUM, CDUM, LDUM, CHARAC, N, NMCONSTR, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < NMCONSTR))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''CONSTRL'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF
                CONSTRL = reshape(CONSTRL_LINE, (/3, NIONS/))
                where (CONSTRL == 0) L_CONSTR_L = 0.0

                CONSTR_NUM_STEP = 500
                CALL RDATAB(LOPEN, INCAR, IU5, 'NSC', '=', '#', ';', 'I', &
               &   CONSTR_NUM_STEP, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''NSC'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                NELM_SC_INITIAL = 5
                CALL RDATAB(LOPEN, INCAR, IU5, 'NELMSCI', '=', '#', ';', 'I', &
               &   NELM_SC_INITIAL, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''NELMSCI'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                NELM_SC_INTER = 0
                CALL RDATAB(LOPEN, INCAR, IU5, 'NELMSCT', '=', '#', ';', 'I', &
               &   NELM_SC_INTER, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''NELMSC'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                ALGO_SC = 1
                CALL RDATAB(LOPEN, INCAR, IU5, 'IALGOSC', '=', '#', ';', 'I', &
               &   ALGO_SC, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''IALGOSC'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                ALGO_SC_DIAG = 1
                CALL RDATAB(LOPEN, INCAR, IU5, 'IALGOSC_DIAG', '=', '#', ';', 'I', &
               &   ALGO_SC_DIAG, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''IALGOSC_DIAG'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                CONSTR_EPSILON = 1e-8
                CALL RDATAB(LOPEN, INCAR, IU5, 'SCDIFF', '=', '#', ';', 'F', &
               &   IDUM, CONSTR_EPSILON, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''SCDIFF'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                INI_SC_ALPHA = 0.1
                CALL RDATAB(LOPEN, INCAR, IU5, 'INISC', '=', '#', ';', 'F', &
               &   IDUM, INI_SC_ALPHA, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''INISC'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                CONSTR_RESTRICT = 0.3
                CALL RDATAB(LOPEN, INCAR, IU5, 'SCCUT', '=', '#', ';', 'F', &
               &   IDUM, CONSTR_RESTRICT, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''SCCUT'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                TRIAL_UPDATE_RESTRICT = .TRUE.
                CALL RDATAB(LOPEN, INCAR, IU5, 'LCUTSC_TRIAL', '=', '#', ';', 'L', &
               &            IDUM, RDUM, CDUM, TRIAL_UPDATE_RESTRICT, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
               &                    ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''LCUTSC_TRIAL'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF

                DEBUG_SC = .FALSE.
                CALL RDATAB(LOPEN, INCAR, IU5, 'LDESC', '=', '#', ';', 'L', &
               &            IDUM, RDUM, CDUM, DEBUG_SC, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
               &                    ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''LDESC'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF

            END IF
! =====================================
!  SASC(Q) parameters
! =====================================
            IF (SCTYPE == 2 .OR. SCTYPE == 3) THEN
                !IF (IU0 >= 0) WRITE (IU0, *) 'Get to here 1'

                ALLOCATE (LAMBDA_LINE_Q(NMCONSTR), CONSTRL_LINE_Q(NMCONSTR))
                ALLOCATE (L_CONSTR_Q(3, NIONS), CONSTRL_Q(3, NIONS))
                !IF (IU0 >= 0) WRITE (IU0, *) 'Get to here 2'
! ... get penalty factor
                LAMBDA_LINE_Q = 0
!         CALL RDATAB(LOPEN,INCAR,IU5,'LAMBDA','=','#',';','F', &
!     &               IDUM,LAMBDA,CDUM,LDUM,CHARAC,N,NMCONSTR,IERR)
                CALL RDATAB(LOPEN, INCAR, IU5, 'LAMBDA_Q', '=', '#', ';', 'F', &
             &               IDUM, LAMBDA_LINE_Q, CDUM, LDUM, CHARAC, N, NMCONSTR, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
             &                       ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''LAMBDA_Q'' from file INCAR.'
                    GOTO 150
                END IF
                L_CONSTR_Q = reshape(LAMBDA_LINE_Q, (/3, NIONS/))

                CONSTRL_LINE_Q = 1
                CALL RDATAB(LOPEN, INCAR, IU5, 'CONSTRL_Q', '=', '#', ';', 'I', &
               &   CONSTRL_LINE_Q, RDUM, CDUM, LDUM, CHARAC, N, NMCONSTR, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < NMCONSTR))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''CONSTRL_Q'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF
                CONSTRL_Q = reshape(CONSTRL_LINE_Q, (/3, NIONS/))
                where (CONSTRL_Q == 0) L_CONSTR_Q = 0.0

                CONSTR_NUM_STEP_Q = 500
                CALL RDATAB(LOPEN, INCAR, IU5, 'NSC_Q', '=', '#', ';', 'I', &
               &   CONSTR_NUM_STEP_Q, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''NSC_Q'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                CONSTR_EPSILON_Q = 1e-8
                CALL RDATAB(LOPEN, INCAR, IU5, 'SCDIFF_Q', '=', '#', ';', 'F', &
               &   IDUM, CONSTR_EPSILON_Q, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''SCDIFF_Q'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                INI_SC_ALPHA_Q = 0.1
                CALL RDATAB(LOPEN, INCAR, IU5, 'INISC_Q', '=', '#', ';', 'F', &
               &   IDUM, INI_SC_ALPHA_Q, CDUM, LDUM, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) THEN
                        WRITE (IU0, *) 'Error reading item ''INISC_Q'' from file INCAR.'
                        WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                    END IF
                END IF

                DEBUG_SC_Q = .FALSE.
                CALL RDATAB(LOPEN, INCAR, IU5, 'LDESC_Q', '=', '#', ';', 'L', &
               &            IDUM, RDUM, CDUM, DEBUG_SC_Q, CHARAC, N, 1, IERR)
                IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
               &                    ((IERR == 0) .AND. (N < 1))) THEN
                    IF (IU0 >= 0) &
                        WRITE (IU0, *) 'Error reading item ''LDESC_Q'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF
        END IF
        ! CALL XML_INCAR('LAMBDA','F',IDUM,LAMBDA,CDUM,LDUM,CHARAC,N)
        ! CALL XML_INCAR_V('LAMBDA', 'F', IDUM, LAMBDA_LINE, CDUM, LDUM, CHARAC, NMCONSTR)
        CLOSE (IU5)

        RETURN

150     CONTINUE
        IF (IU0 >= 0) &
            WRITE (IU0, 151) IERR, N
151     FORMAT(' Error code was IERR=', I1, ' ... . Found N=', I5, ' data.')
        STOP

    END SUBROUTINE

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

!************************ SUBROUTINE ADPT_CONSTRAINING_POT **************
!
! expects CVTOT in (charge,magnetization) convention in real space
!
!***********************************************************************

    SUBROUTINE ADPT_CONSTRAINING_POT(CVTOT, GRIDC, WDES)

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
                        IF (SCTYPE_CURRENT == 1) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       -L_CONSTR(1, NI)*WEIGHT(IND, NI)
                        ELSE IF (SCTYPE_CURRENT == 2) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       2*L_CONSTR(1, NI)*WEIGHT(IND, NI)*MW_X
                        END IF
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
                        IF (SCTYPE_CURRENT == 1) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       -L_CONSTR(2, NI)*WEIGHT(IND, NI)
                        ELSE IF (SCTYPE_CURRENT == 2) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       2*L_CONSTR(2, NI)*WEIGHT(IND, NI)*MW_Y
                        END IF
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
                        IF (SCTYPE_CURRENT == 1) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       -L_CONSTR(3, NI)*WEIGHT(IND, NI)
                        ELSE IF (SCTYPE_CURRENT == 2) THEN
                            CVTOT(NLI(IND, NI), ISP) = CVTOT(NLI(IND, NI), ISP) + &
                                                       2*L_CONSTR(3, NI)*WEIGHT(IND, NI)*MW_
                        END IF
                    END DO

                END SELECT

            END DO ions
            NIS = NIS + NITYP(NT)
            END DO ion_types

        END DO spin

        RETURN
    END SUBROUTINE ADPT_CONSTRAINING_POT

!************************ FUNCTION E_CONSTRAINT ************************
!
!***********************************************************************

    FUNCTION E_CONSTRAINT()

        USE prec

        IMPLICIT COMPLEX(q) (C)
        IMPLICIT REAL(q) (A - B, D - H, O - Z)

        REAL(q) E_CONSTRAINT, MW_(3), MW_IN_M_CONSTR

        E_CONSTRAINT = 0; E_PENALTY = 0; E_EXPECT = 0

        ions: DO NI = 1, NIONS

            IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
                ABS(M_CONSTR(2, NI)) < TINY .AND. &
                ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

            MW_IN_M_CONSTR = MW(1, NI)*M_CONSTR(1, NI) + &
                             MW(2, NI)*M_CONSTR(2, NI) + &
                             MW(3, NI)*M_CONSTR(3, NI)

            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                DO I = 1, 3
                    MW_(I) = MW(I, NI) - M_CONSTR(I, NI)*MW_IN_M_CONSTR

                END DO
            END IF

            IF (I_CONSTRAINED_M == 2) THEN
                DO I = 1, 3
                    MW_(I) = MW(I, NI) - M_CONSTR(I, NI)
                END DO
            END IF

            ! Add penalty energy
            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                E_CONSTRAINT = E_CONSTRAINT - &
               &   LAMBDA*(MW_(1)*MW(1, NI) + MW_(2)*MW(2, NI) + MW_(3)*MW(3, NI))
                E_PENALTY = E_PENALTY + LAMBDA*(MW_(1)*MW_(1) + MW_(2)*MW_(2) + MW_(3)*MW_(3))
                E_EXPECT = E_EXPECT + 2*LAMBDA*(MW_(1)*MW(1, NI) + MW_(2)*MW(2, NI) + MW_(3)*MW(3, NI))
            END IF

            IF (I_CONSTRAINED_M == 2) THEN
 
                IF (SCTYPE_CURRENT == 0) THEN
                    E_CONSTRAINT=E_CONSTRAINT- &
                    &   LAMBDA*(MW(1,NI)*MW(1,NI)+MW(2,NI)*MW(2,NI)+MW(3,NI)*MW(3,NI))+ &
                    &   LAMBDA*(M_CONSTR(1,NI)*M_CONSTR(1,NI)+M_CONSTR(2,NI)*M_CONSTR(2,NI)+M_CONSTR(3,NI)*M_CONSTR(3,NI))
                    E_PENALTY=E_PENALTY+LAMBDA*(MW_(1)*MW_(1)+MW_(2)*MW_(2)+MW_(3)*MW_(3))
                    E_EXPECT=E_EXPECT+2*LAMBDA*(MW_(1)*MW(1,NI)+MW_(2)*MW(2,NI)+MW_(3)*MW(3,NI))

                ELSE IF (SCTYPE_CURRENT == 1) THEN
                    E_CONSTRAINT = E_CONSTRAINT + &
                     & (L_CONSTR(1, NI)*MW(1, NI) + L_CONSTR(2, NI)*MW(2, NI) + L_CONSTR(3, NI)*MW(3, NI))
                    E_PENALTY = E_PENALTY - (L_CONSTR(1, NI)*MW_(1) + L_CONSTR(2, NI)*MW_(2) + L_CONSTR(3, NI)*MW_(3))
                    E_EXPECT = E_EXPECT - (L_CONSTR(1, NI)*MW_(1) + L_CONSTR(2, NI)*MW_(2) + L_CONSTR(3, NI)*MW_(3))

                ELSE IF (SCTYPE_CURRENT == 2) THEN
                    E_CONSTRAINT = E_CONSTRAINT - &
                  & (L_CONSTR(1, NI)*MW(1, NI)*MW(1, NI) + L_CONSTR(2, NI)*MW(2, NI)*MW(2, NI) + L_CONSTR(3, NI)*MW(3, NI)*MW(3, NI)) + &
                  & (L_CONSTR(1,NI)*M_CONSTR(1,NI)*M_CONSTR(1,NI)+L_CONSTR(2,NI)*M_CONSTR(2,NI)*M_CONSTR(2,NI)+L_CONSTR(3,NI)*M_CONSTR(3,NI)*M_CONSTR(3,NI))
                    E_PENALTY = E_PENALTY + (L_CONSTR(1, NI)*MW_(1)*MW_(1) + L_CONSTR(2, NI)*MW_(2)*MW_(2) + L_CONSTR(3, NI)*MW_(3)*MW_(3))
                    E_EXPECT = E_EXPECT + 2*(L_CONSTR(1, NI)*MW_(1)*MW_(1) + L_CONSTR(2, NI)*MW_(2)*MW_(2) + L_CONSTR(3, NI)*MW_(3)*MW_(3))

                END IF
            END IF
        END DO ions

    END FUNCTION E_CONSTRAINT

!************************ SUBROUTINE WRITE_CONSTRAINED_M ***************
!
!***********************************************************************

    SUBROUTINE WRITE_CONSTRAINED_M(IU, LONG)

        USE prec

        IMPLICIT COMPLEX(q) (C)
        IMPLICIT REAL(q) (A - B, D - H, O - Z)

        !REAL(q) MW_(3),MW_IN_M_CONSTR
        REAL(q) MW_(3, NIONS), MW_IN_M_CONSTR
        REAL(q) MW_MASK(3, NIONS)
        REAL(q) FT(3, NIONS)
        LOGICAL LONG

        IF (I_CONSTRAINED_M == 0 .OR. IU < 0) RETURN
        IF (LONG) THEN
            ! WRITE(IU,'(/A7,E12.5,A11,E10.3)') ' E_p = ',E_PENALTY,'  lambda = ',LAMBDA
            ! WRITE(IU,*) 'ion             lambda*MW_perp'
            DO NI = 1, NIONS
                IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
                    ABS(M_CONSTR(2, NI)) < TINY .AND. &
                    ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

                MW_IN_M_CONSTR = MW(1, NI)*M_CONSTR(1, NI) + &
                                 MW(2, NI)*M_CONSTR(2, NI) + &
                                 MW(3, NI)*M_CONSTR(3, NI)

                IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                    DO I = 1, 3
                        MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)*MW_IN_M_CONSTR
                    END DO
                END IF

                IF (I_CONSTRAINED_M == 2) THEN
                    IF (SCTYPE_CURRENT == 0) THEN
                        DO I = 1, 3
                            MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)
                            ! FT(I, NI) = L_CONSTR(I, NI)
                        END DO
                    ELSE
                        DO I = 1, 3
                            MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)
                            FT(I, NI) = L_CONSTR(I, NI)
                        END DO
                    END IF
                END IF

                !WRITE(IU,'(I3,A2,E16.8,A2,E16.8,A2,E16.8)') &
                !       NI,'  ',LAMBDA*MW_(1,NI),'  ',LAMBDA*MW_(2,NI),'  ',LAMBDA*MW_(3,NI)
            END DO

            MW_MASK = MW_
            where (CONSTRL == 0) MW_MASK = 0.0
            WRITE (IU, '(/A7,E12.5,A11)') ' E_p = ', E_PENALTY
            WRITE (IU, '( A7,E12.5)') '<lVp>= ', E_EXPECT
            WRITE (IU, '( A7,E12.5)') ' DBL = ', E_CONSTRAINT()
            WRITE (IU, '( a,es20.12)') ' RMS = ', sqrt(sum(MW_MASK**2)/count(CONSTRL == 1))
            WRITE (IU, *) 'ion                     MW_current                            delta_MW'

            DO NI = 1, NIONS
                WRITE (IU, '(I3,3F18.9,3F18.9)') NI, MW(1:3, NI), MW_(1:3, NI)
            END DO

            IF (SCTYPE_CURRENT == 0) THEN 
               WRITE (IU, *) 'lambda                                     '
               DO NI = 1, NIONS
                   WRITE (IU, '(I3,E16.8)') NI, LAMBDA
               END DO
               WRITE (IU, *) " "
            ELSE IF (SCTYPE_CURRENT == 1) THEN 
               WRITE (IU, *) 'lambda                                                      Magnetic Force    '
               DO NI = 1, NIONS
                   WRITE (IU, '(I3,3E16.8,3E16.8)') NI, L_CONSTR(1:3, NI), FT(1:3, NI)
               END DO
               WRITE (IU, *) " "
            ELSE IF (SCTYPE_CURRENT == 2) THEN
               WRITE (IU, *) 'lambda                                     '
               DO NI = 1, NIONS
                   WRITE (IU, '(I3,3E16.8)') NI, L_CONSTR(1:3, NI)
               END DO
               WRITE (IU, *) " "
            END IF

        ELSE
            WRITE (IU, '(/A7,E12.5,A11,E10.3)') ' E_p = ', E_PENALTY, '  lambda = ', LAMBDA
            WRITE (IU, '(/A7,E12.5,A11)') ' E_p = ', E_PENALTY
            WRITE (IU, '( A7,E12.5)') ' DBL = ', E_CONSTRAINT()
            WRITE (IU, *) 'ion        MW_int                 M_int'
            DO NI = 1, NIONS
                WRITE (IU, '(I3,3F12.9,3F12.9)') NI, MW(1:3, NI), MW_(1:3, NI)
            END DO
        END IF

        RETURN
    END SUBROUTINE WRITE_CONSTRAINED_M

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
