#include "symbol.inc"

!************************ SUBROUTINE CHGLOC ****************************
! RCS:  $Id: chgloc.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!  calculate local charge density n(r)
!
!***********************************************************************

      SUBROUTINE CHGLOC(NBANDS,NKDIM,LDIMP,NIONS,ISPIN, &
     &     PAR,FERWE)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION PAR(NBANDS,NKDIM,LDIMP,NIONS,ISPIN)
      DIMENSION FERWE(NBANDS,NKDIM,ISPIN)

      RETURN
      END
