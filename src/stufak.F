#include "symbol.inc"
!************************ SUBROUTINE STUFAK ****************************
! RCS:  $Id: stufak.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
! this subroutine calculates the structure factor on the grid of
! reciprocal lattice vectors
! cstrf(g) = sum over ions (-exp(ig.r)) where r is the position of the
! ion
!***********************************************************************

      SUBROUTINE STUFAK(GRIDC,T_INFO,CSTRF)
      USE prec

      USE mpimy
      USE mgrid
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO

      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

! loop over all types of atoms
      NIS=1
      typ: DO NT=1,T_INFO%NTYP
      CALL STUFAK_ONE(GRIDC,T_INFO%NITYP(NT),T_INFO%POSION(1,NIS),T_INFO%VCA(NT),CSTRF(1,NT))
      NIS=NIS+T_INFO%NITYP(NT)

      ENDDO typ

      RETURN
      END


!************************ SUBROUTINE STUFAK_ONE ************************
!
! this subroutine calculates the structure factor on the grid of
! for one species (i.e. partial structure factor)
! cstrf(g) = sum over ions (-exp(ig.r)) where r is the position of the
! ion
!
!***********************************************************************

      SUBROUTINE STUFAK_ONE(GRIDC,NIONS,POSION,VCA,CSTRF)
      USE prec

      USE mpimy
      USE mgrid
      USE poscar
      USE constant

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      REAL(q) POSION(3,NIONS)
      REAL(q) VCA

      TYPE (grid_3d)     GRIDC

      COMPLEX(q) CSTRF(GRIDC%RC%NP)
      CSTRF=0
!=======================================================================
! calculate partial structur-factor
!=======================================================================
      ion: DO NI=1,NIONS
!=======================================================================
! loop over all grid points
!=======================================================================
#ifdef vector
!-----------------------------------------------------------------------
! simple version for vector machines
!-----------------------------------------------------------------------
      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        G3=POSION(3,NI)*GRIDC%LPCTZ(N3)
        G2=POSION(2,NI)*GRIDC%LPCTY(N2)
        G1=POSION(1,NI)*GRIDC%LPCTX(N1)

        CSTRF(N)=CSTRF(N)+EXP(-CITPI*(G3+G2+G1))*VCA
      ENDDO
#else
!-----------------------------------------------------------------------
! more envolved version which is faster on most (scalar) machines
! and includes support for parallel machines
!-----------------------------------------------------------------------
      CX =EXP(-CITPI*POSION(1,NI))
      G1 =POSION(1,NI)*xmin(GRIDC%NGX)

      col: DO NC=1,GRIDC%RC%NCOL
        N=(NC-1)*GRIDC%RC%NROW+1

        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)
        G2=POSION(2,NI)*GRIDC%LPCTY(N2)
        G3=POSION(3,NI)*GRIDC%LPCTZ(N3)
        CE=EXP(-CITPI*(G3+G2+G1))*VCA
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
        DO N1P=0,GRIDC%RC%NROW-1

          SETN1fromN1P
          CSTRF(N+N1)=CSTRF(N+N1)+CE
          CE=CE*CX
        ENDDO
      ENDDO col
#endif
!-----------------------------------------------------------------------
!  next ion + next type
!-----------------------------------------------------------------------
      ENDDO ion

      RETURN
      END


