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

        ! IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
        !     ABS(M_CONSTR(2, NI)) < TINY .AND. &
        !     ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

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
                E_CONSTRAINT = E_CONSTRAINT - &
                &   LAMBDA*(MW(1, NI)*MW(1, NI) + MW(2, NI)*MW(2, NI) + MW(3, NI)*MW(3, NI)) + &
                &   LAMBDA*(M_CONSTR(1, NI)*M_CONSTR(1, NI) + M_CONSTR(2, NI)*M_CONSTR(2, NI) + M_CONSTR(3, NI)*M_CONSTR(3, NI))
                E_PENALTY = E_PENALTY + LAMBDA*(MW_(1)*MW_(1) + MW_(2)*MW_(2) + MW_(3)*MW_(3))
                E_EXPECT = E_EXPECT + 2*LAMBDA*(MW_(1)*MW(1, NI) + MW_(2)*MW(2, NI) + MW_(3)*MW(3, NI))

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
