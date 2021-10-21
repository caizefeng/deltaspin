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

            LAMBDA = 0
            CALL RDATAB(LOPEN, INCAR, IU5, 'LAMBDA', '=', '#', ';', 'F', &
        &               IDUM, LAMBDA, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. &
        &                       ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) &
                    WRITE (IU0, *) 'Error reading item ''LAMBDA'' from file INCAR.'
                GOTO 150
            END IF
            CALL XML_INCAR('LAMBDA', 'F', IDUM, LAMBDA, CDUM, LDUM, CHARAC, N)

            CONSTRL_LINE = 1
            CONSTRL = reshape(CONSTRL_LINE, (/3, NIONS/))
        END IF

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

            NELM_SC_INITIAL = 1
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

            DECAY_EPSILON = -1
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCDECAY', '=', '#', ';', 'F', &
           &   IDUM, DECAY_EPSILON, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCDECAY'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            LBOUND_EPSILON = 2E-7
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCDIFFB', '=', '#', ';', 'F', &
           &   IDUM, LBOUND_EPSILON, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCDIFFB'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            CONV_BOUND = -1
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCCONVB', '=', '#', ';', 'F', &
           &   IDUM, CONV_BOUND, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCCONVB'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            ALLOCATE (CONV_BOUND_GRAD(T_INFO%NTYP))
            CONV_BOUND_GRAD = -1
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCCONVB_GRAD', '=', '#', ';', 'F', &
           &   IDUM, CONV_BOUND_GRAD, CDUM, LDUM, CHARAC, N, T_INFO%NTYP, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCCONVB_GRAD'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            ALGO_GRAD_DECAY = 0
            CALL RDATAB(LOPEN, INCAR, IU5, 'IDECAY_GRAD', '=', '#', ';', 'I', &
           &   ALGO_GRAD_DECAY, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''IDECAY_GRAD'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            DECAY_GRADIENT = -1
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCDECAY_GRAD', '=', '#', ';', 'F', &
           &   IDUM, DECAY_GRADIENT, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCDECAY_GRAD'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            LBOUND_GRAD = 0.1
            CALL RDATAB(LOPEN, INCAR, IU5, 'SCGRADB', '=', '#', ';', 'F', &
           &   IDUM, LBOUND_GRAD, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''SCGRADB'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            GRAD_STEP_NUM = 0
            CALL RDATAB(LOPEN, INCAR, IU5, 'NGRAD', '=', '#', ';', 'I', &
           &   GRAD_STEP_NUM, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''NGRAD'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            ALLOCATE (GRAD_STEP_POINT(GRAD_STEP_NUM))
            GRAD_STEP_POINT = 0
            CALL RDATAB(LOPEN, INCAR, IU5, 'NGRAD_STEP', '=', '#', ';', 'I', &
           &   GRAD_STEP_POINT, RDUM, CDUM, LDUM, CHARAC, N, GRAD_STEP_NUM, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''NGRAD_STEP'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF

            ALLOCATE (BOUND_GRAD_SEQUENCE_LINE(T_INFO%NTYP*GRAD_STEP_NUM))
            ALLOCATE (BOUND_GRAD_SEQUENCE(T_INFO%NTYP, GRAD_STEP_NUM))
            BOUND_GRAD_SEQUENCE_LINE = -1
            CALL RDATAB(LOPEN, INCAR, IU5, 'NGRAD_VALUE', '=', '#', ';', 'F', &
           &   IDUM, BOUND_GRAD_SEQUENCE_LINE, CDUM, LDUM, CHARAC, N, T_INFO%NTYP*GRAD_STEP_NUM, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''NGRAD_VALUE'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
            END IF
            BOUND_GRAD_SEQUENCE = reshape(BOUND_GRAD_SEQUENCE_LINE, (/T_INFO%NTYP, GRAD_STEP_NUM/))

            CONSTR_NUM_STEP_MIN = 3
            CALL RDATAB(LOPEN, INCAR, IU5, 'NSCMIN', '=', '#', ';', 'I', &
           &   CONSTR_NUM_STEP_MIN, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''NSCMIN'' from file INCAR.'
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

            SCDECOUPLE = 0
            CALL RDATAB(LOPEN, INCAR, IU5, 'IDECOSC', '=', '#', ';', 'I', &
           &   SCDECOUPLE, RDUM, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''IDECOSC'' from file INCAR.'
                    WRITE (IU0, *) 'Error code was IERR=', IERR, ' Found N=', N, ' data items'
                END IF
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

            EDIFF_Q = -1.0
            CALL RDATAB(LOPEN, INCAR, IU5, 'EDIFF_Q', '=', '#', ';', 'F', &
           &   IDUM, EDIFF_Q, CDUM, LDUM, CHARAC, N, 1, IERR)
            IF (((IERR /= 0) .AND. (IERR /= 3)) .OR. ((IERR == 0) .AND. (N < 1))) THEN
                IF (IU0 >= 0) THEN
                    WRITE (IU0, *) 'Error reading item ''EDIFF_Q'' from file INCAR.'
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

150 CONTINUE
    IF (IU0 >= 0) &
        WRITE (IU0, 151) IERR, N
151 FORMAT(' Error code was IERR=', I1, ' ... . Found N=', I5, ' data.')
    STOP

END SUBROUTINE
