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

            ! IF (ABS(M_CONSTR(1, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(2, NI)) < TINY .AND. &
            !     ABS(M_CONSTR(3, NI)) < TINY) CYCLE ! we do not constrain this ion

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
