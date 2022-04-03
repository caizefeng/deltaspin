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
            ! IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(2, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

            MW_IN_M_CONSTR = MW(1, NI)*M_CONSTR(1, NI) + &
                             MW(2, NI)*M_CONSTR(2, NI) + &
                             MW(3, NI)*M_CONSTR(3, NI)

            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                DO I = 1, 3
                    MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)*MW_IN_M_CONSTR
                END DO
            END IF

            IF (I_CONSTRAINED_M == 2) THEN
                IF (SCTYPE_CURRENT == 0 .OR. SCTYPE_CURRENT == 2) THEN
                    DO I = 1, 3
                        MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)
                    END DO
                ELSE
                    DO I = 1, 3
                        MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)
                        FT(I, NI) = -L_CONSTR(I, NI)
                    END DO
                END IF
            END IF

        END DO

        MW_MASK = MW_
        IF (SCTYPE_CURRENT == 1) THEN
            where (CONSTRL == 0) MW_MASK = 0.0
        ELSE IF (SCTYPE_CURRENT == 2) THEN
            where (CONSTRL_Q == 0) MW_MASK = 0.0
        END IF

        WRITE (IU, '(/A7,E12.5,A11)') ' E_p = ', E_PENALTY
        WRITE (IU, '( A7,E12.5)') '<lVp>= ', E_EXPECT
        WRITE (IU, '( A7,E12.5)') ' DBL = ', E_CONSTRAINT()
        IF (SCTYPE_CURRENT == 0) THEN
            WRITE (IU, '( a,es20.12)') ' RMS = ', sqrt(sum(MW_MASK**2)/(3*NIONS))
        ELSE IF (SCTYPE_CURRENT == 1) THEN
            WRITE (IU, '( a,es20.12)') ' RMS = ', sqrt(sum(MW_MASK**2)/count(CONSTRL == 1))
        ELSE IF (SCTYPE_CURRENT == 2) THEN
            WRITE (IU, '( a,es20.12)') ' RMS = ', sqrt(sum(MW_MASK**2)/count(CONSTRL_Q == 1))
        END IF
        WRITE (IU, *) 'ion                     MW_current                            delta_MW'

        DO NI = 1, NIONS
            WRITE (IU, '(I3,3F18.9,3F18.9)') NI, MW(1:3, NI), MW_(1:3, NI)
        END DO

        WRITE (IU, *) 'ion                     M_current'

        DO NI = 1, NIONS
            WRITE (IU, '(I3,3F18.9)') NI, M_TOT(1:3, NI)
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
        DO NI = 1, NIONS
            ! IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(2, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

            MW_IN_M_CONSTR = MW(1, NI)*M_CONSTR(1, NI) + &
                             MW(2, NI)*M_CONSTR(2, NI) + &
                             MW(3, NI)*M_CONSTR(3, NI)

            IF (I_CONSTRAINED_M == 1 .OR. I_CONSTRAINED_M == 3) THEN
                DO I = 1, 3
                    MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)*MW_IN_M_CONSTR
                END DO
            END IF

            IF (I_CONSTRAINED_M == 2) THEN
                DO I = 1, 3
                    MW_(I, NI) = MW(I, NI) - M_CONSTR(I, NI)
                END DO
            END IF
        END DO
        ! WRITE (IU, '(/A7,E12.5,A11,E10.3)') ' E_p = ', E_PENALTY, '  lambda = ', LAMBDA
        ! WRITE (IU, '(/A7,E12.5,A11)') ' E_p = ', E_PENALTY
        ! WRITE (IU, '( A7,E12.5)') '<lVp>= ', E_EXPECT
        ! WRITE (IU, '( A7,E12.5)') ' DBL = ', E_CONSTRAINT()
        WRITE (IU, *) 'ion                     MW_current                            delta_MW'

        DO NI = 1, NIONS
            WRITE (IU, '(I3,3F18.9,3F18.9)') NI, MW(1:3, NI), MW_(1:3, NI)
        END DO
    END IF

    RETURN
END SUBROUTINE WRITE_CONSTRAINED_M
