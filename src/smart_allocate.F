#include "symbol.inc"
!**********************************************************************
!
! this module can be used if time for allocate and
! deallocation turns out to be a problem
! 
!**********************************************************************


MODULE smart_allocate
  USE prec
  IMPLICIT NONE
  CONTAINS

  SUBROUTINE SMART_ALLOCATE_REAL(A,N)
    REAL(q),POINTER :: A(:)
    INTEGER N

    IF (ASSOCIATED(A)) THEN
       IF (SIZE(A) < N) THEN
          DEALLOCATE(A)
       ENDIF
    ENDIF
    IF (.NOT. ASSOCIATED(A)) THEN
       ALLOCATE(A(N))
    ENDIF

  END SUBROUTINE SMART_ALLOCATE_REAL

  SUBROUTINE SMART_ALLOCATE_WAVE(A,N)
    GDEF,POINTER :: A(:)
    INTEGER N

    IF (ASSOCIATED(A)) THEN
       IF (SIZE(A) < N) THEN
          DEALLOCATE(A)
       ENDIF
    ENDIF
    IF (.NOT. ASSOCIATED(A)) THEN
       ALLOCATE(A(N))
    ENDIF

  END SUBROUTINE SMART_ALLOCATE_WAVE

  SUBROUTINE SMART_ALLOCATE_COMPLEX(A,N)
    COMPLEX(q),POINTER :: A(:)
    INTEGER N

    IF (ASSOCIATED(A)) THEN
       IF (SIZE(A) < N) THEN
          DEALLOCATE(A)
       ENDIF
    ENDIF
    IF (.NOT. ASSOCIATED(A)) THEN
       ALLOCATE(A(N))
    ENDIF

  END SUBROUTINE SMART_ALLOCATE_COMPLEX

END MODULE smart_allocate
      
