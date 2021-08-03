#include "symbol.inc"

!***********************************************************************
! RCS:  $Id: opergrid.F,v 1.3 2003/06/27 13:22:21 kresse Exp kresse $
!
!  sum all points on the grid and dump onto screen
!
!***********************************************************************

      SUBROUTINE SUMGRD(CSRC,GRID)
      USE prec
      USE mgrid

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRID
      DIMENSION CSRC(GRID%NGX_rd,GRID%NGY_rd,GRID%NGZ_rd)

      CSUM=0

      DO NZ=1,GRID%NGZ_rd
      DO NY=1,GRID%NGY_rd
      DO NX=1,GRID%NGX_rd
        CSUM=CSUM+CSRC(NX,NY,NZ)
      ENDDO
      ENDDO
      ENDDO
      WRITE(*,*) CSUM
      RETURN
      END

!***********************************************************************
!
!  sum all points on the real grid and dump onto screen
!
!***********************************************************************

      SUBROUTINE SUMRGR(CSRC, GRID)
      USE prec
      USE mgrid

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRID
      RGRID CSRC(GRID%NGX,GRID%NGY,GRID%NGZ)
      RGRID CSUM
      CSUM=0

      DO NZ=1,GRID%NGZ
      DO NY=1,GRID%NGY
      DO NX=1,GRID%NGX
        CSUM=CSUM+CSRC(NX,NY,NZ)
      ENDDO
      ENDDO
      ENDDO
      WRITE(*,*) CSUM
      RETURN
      END

!***********************SUBROUTINE DUMP1******************************
!
!  This debug-routine dumps a compressed array for one K-point
!
!**********************************************************************

      SUBROUTINE DUMP1(NK,NRPLWV,IGX,IGY,IGZ,NPLWKP,C1)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION IGX(NRPLWV,1),IGY(NRPLWV,1),IGZ(NRPLWV,1)
      DIMENSION NPLWKP(1)
      REAL(q)      C1(1)

      DO 50 K=1,NPLWKP(NK)
        WRITE(*,10)K,IGX(K,NK),IGY(K,NK),IGZ(K,NK), &
     & REAL( C1(K) ,KIND=q)
   50 CONTINUE
   10 FORMAT(I3,2X,3I2,6E14.7)
      WRITE(*,*)
      RETURN
      END SUBROUTINE

!***********************SUBROUTINE DUMPA ******************************
!
!  This debug-routine dumps a uncompressed array for one K-Point
!
!**********************************************************************

      SUBROUTINE DUMPA(NK,NGX,NGY,NGZ,CA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q)   CA(NGX,NGY,NGZ)

      NDUMP=MIN(NGX,NGY,NGZ,10)

      DO NZ=1,NDUMP
      WRITE(*,*)'NZ=',NZ
      DO NY=1,NDUMP
        WRITE(*,'(I2,24E10.3)') NY,(ABS(CA(NX,NY,NZ)),NX=1,NDUMP)
      ENDDO
      ENDDO

      RETURN
      END SUBROUTINE

!*********************************************************************
!
! routine to dump a (NxN) Matrix
!
!*********************************************************************

      SUBROUTINE DDUMP(C,NBANDS)
      USE prec
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      GDEF C(NBANDS,NBANDS)

      NBA=MIN(10,NBANDS)

      DO 830 N1=NBANDS-NBA+1,NBANDS
      WRITE(*,1)N1, &
     &         (REAL( C(N1,N2) ,KIND=q) ,N2=NBANDS-NBA+1,NBANDS)
 830  CONTINUE
#ifndef gammareal
      WRITE(*,*)
      DO 840 N1=NBANDS-NBA+1,NBANDS
      WRITE(*,2)N1, &
     &         (AIMAG(C(N1,N2)),N2=NBANDS-NBA+1,NBANDS)
 840  CONTINUE
#endif
      WRITE(*,*)

    1 FORMAT(1I2,3X,20F9.5)
    2 FORMAT(1I2,3X,20F9.5)
      RETURN
      END SUBROUTINE

