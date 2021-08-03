!***********************************************************************
!
!    Timer-Routine for CRAY
!
!***********************************************************************

      SUBROUTINE VTIME(VPUTIM,CPUTIM)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
      CPUTIM=SECONDR()
      VPUTIM=SECOND()
      RETURN
      END
