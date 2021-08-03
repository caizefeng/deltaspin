!***********************************************************************
!
!    Timer-Routine
!
!***********************************************************************

      SUBROUTINE VTIME(VPUTIM,CPUTIM)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
      CALL CLOCKV(VPUTIM,CPUTIM,0,2)
      RETURN
      END

      FUNCTION ERRF(X)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) X,ERRF
      ERRF=ERF(X)
      RETURN
      END

      FUNCTION ERRFC(X)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) X,ERRFC
      ERRFC=ERFC(X)
      RETURN
      END
