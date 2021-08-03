
#include "symbol.inc"

! RCS:  $Id: symlib.F,v 1.5 2003/06/27 13:22:23 kresse Exp kresse $
!
! if you need someother preceission specify it in the makefile
!
MODULE sym_prec
  USE prec
  REAL(q) :: TINY=1.E-5_q
END MODULE
!********************* Symmetry package ********************************
!                                                                      *
!  Author: Juergen Furthmueller                                        *
!          Max-Planck-Institut fuer Metallforschung                    *
!          Heissenbergstr. 1                                           *
!          D 7000 Stuttgart 80                                         *
!                                                                      *
!          Tel.: Germany-711 6860 533                                  *
!                                                                      *
!***********************************************************************
!                                                                      *
!
!***********************************************************************
!
! periodic_image_convention uses the minimum image convention
! to test whether two atoms are equivalent
! this should be more reliable then the choice of 
! Juergen Furthmueller
! however, in most cases, results are identical, so for
! the time being stick to old "well-tested" approach
!
!***********************************************************************
#ifdef  periodic_image_convention
#define MIN_IMAG(x) (MOD(x+100.5_q,1._q)-0.5_q)
#else
#define MIN_IMAG(x) (x)
#endif


      SUBROUTINE SETGRP(S,SYMOP,GTRANS,NROT,NROTK,TAU,LATTYP,NOP, &
     &                         TAUROT,NSP,NSX,NA,NAX,INDEX,WRK,PRCHAN)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SETGRP is the primary user-interface of this symmetry      *
!   package. The function of SETGRP is to set up all possible          *
!   space group operators (INTEGER rotation matrices and nontrivial    *
!   translations given in crystal coordinates) of a lattice with       *
!   some arbitrary basis (atomic arrangement). This will be done by    *
!   first setting up the point group operators of the pure bravais     *
!   lattice without basis (empty supercell) and checking which of the  *
!   operations can reproduce the lattice with basis (to be done in     *
!   routine GETGRP - for further details see comments on GETGRP).      *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output! TAU will contain the same set of    *
!              atomic position as on input, but the order of storage   *
!              of the positions will be changed in some way described  *
!              in routine LATORD of the lattice vector package and     *
!              periodic boundary conditions are imposed on the atomic  *
!              coordinates to shift all atoms into a box defined by    *
!              the corner points (+/- 0.5, +/- 0.5, +/- 0.5).          *
!                                                                      *
!      LATTYP gives the bravais lattice type of the supercell as       *
!             coded in routine LATGEN (parameter IBRAV) of the         *
!             lattice vector package (LATTYP may run from 1 to 14).    *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      PRCHAN contains the unit number for standard output.            *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices for all        *
!                space group operations.                               *
!                                                                      *
!      SYMOP(3,3,48) contains the rotation matrices of the pure        *
!                bravais lattice.                                      *
!                                                                      *
!      GTRANS(3,48) contains the nonprimitive translations for         *
!                all space group operations.                           *
!                                                                      *
!      NROT contains the number of pure point group rotations (to      *
!           be stored in S(3,3,1)...S(3,3,NROT)) and                   *
!      NROTK contains the number of space group operations found       *
!           in routine GETGRP.                                         *
!      NOP contains the number of point group operations of the pure   *
!           bravais lattice without basis.                             *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S,SYMOP,SYMGEN(9,3),INV(9),R6Z(9),R3D(9),R2HEX(9)
      INTEGER R2TRI(9),R2YP(9),R4ZP(9),R4ZBC(9),R4ZFC(9),R2ZP(9)
      INTEGER R2YBC(9),R2ZBC(9),R2YBAS(9),R2YFC(9),R2ZFC(9),PRCHAN

      DIMENSION S(3,3,48),GTRANS(3,48),SYMOP(3,3,48),NA(NSP)
      DIMENSION TAU(NAX,3,NSX),TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX)

      SAVE INV,R6Z,R3D,R2HEX,R2TRI,R2YP,R4ZP,R4ZBC,R4ZFC,R2ZP,R2YBC
      SAVE R2ZBC,R2YBAS,R2YFC,R2ZFC

      DATA INV /-1,0,0,0,-1,0,0,0,-1/,R3D /0,0,1,1,0,0,0,1,0/
      DATA R6Z /1,-1,0,1,0,0,0,0,1/,R2HEX /1,-1,0,0,-1,0,0,0,-1/
      DATA R2TRI /-1,0,0,0,0,-1,0,-1,0/,R4ZP /0,-1,0,1,0,0,0,0,1/
      DATA R2YP /-1,0,0,0,1,0,0,0,-1/,R4ZBC /0,1,0,0,1,-1,-1,1,0/
      DATA R4ZFC /1,1,1,0,0,-1,-1,0,0/,R2ZP /-1,0,0,0,-1,0,0,0,1/
      DATA R2YBC /0,-1,1,0,-1,0,1,-1,0/,R2ZBC /0,1,-1,1,0,-1,0,0,-1/
      DATA R2YBAS /0,-1,0,-1,0,0,0,0,-1/,R2YFC /0,0,1,-1,-1,-1,1,0,0/
      DATA R2ZFC /0,1,0,1,0,0,-1,-1,-1/

! The pure translation lattice (bravais lattice) has some maximum
! symmetry. Set first up the point group operations for this symmetry.
! Remark: All these groups contain the inversion as a generator ... :
      CALL SGRCOP(INV,SYMGEN(1,1))
      IF (LATTYP==14) THEN
! Triclinic symmetry:
         CALL SGRGEN(SYMGEN,1,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,14)
   14 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' triclinic supercell.'/)
      ELSE IF (LATTYP==13) THEN
! Monoclinic symmetry (base centered / 'b-a' is rotation axis ...):
         CALL SGRCOP(R2YBAS,SYMGEN(1,2))
         CALL SGRGEN(SYMGEN,2,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,13)
   13 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' base centered monoclinic supercell.'/)
      ELSE IF (LATTYP==12) THEN
! Monoclinic symmetry (primitive cell / 'b-axis' is rotation axis ...):
         CALL SGRCOP(R2YP,SYMGEN(1,2))
         CALL SGRGEN(SYMGEN,2,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,12)
   12 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple monoclinic supercell.'/)
      ELSE IF (LATTYP==11) THEN
! Orthorhombic symmetry (base centered ...):
         CALL SGRCOP(R2ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YBAS,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,11)
   11 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' base centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==10) THEN
! Orthorhombic symmetry (face centered ...):
         CALL SGRCOP(R2ZFC,SYMGEN(1,2))
         CALL SGRCOP(R2YFC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,10)
   10 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' face centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==9) THEN
! Orthorhombic symmetry (body centered ...):
         CALL SGRCOP(R2ZBC,SYMGEN(1,2))
         CALL SGRCOP(R2YBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,9)
    9 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==8) THEN
! Orthorhombic symmetry (primitive cell ...):
         CALL SGRCOP(R2ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,8)
    8 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple orthorhombic supercell.'/)
      ELSE IF (LATTYP==7) THEN
! Trigonal (rhombohedral) symmetry:
         CALL SGRCOP(R2TRI,SYMGEN(1,2))
         CALL SGRCOP(R3D,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,7)
    7 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' trigonal (rhombohedral) supercell.'/)
      ELSE IF (LATTYP==6) THEN
! Tetragonal symmetry (body centered / 'c-axis' is the special axis):
         CALL SGRCOP(R4ZBC,SYMGEN(1,2))
         CALL SGRCOP(R2YBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,6)
    6 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered tetragonal supercell.'/)
      ELSE IF (LATTYP==5) THEN
! Tetragonal symmetry (primitive cell / 'c-axis' is the special axis):
         CALL SGRCOP(R4ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,5)
    5 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple tetragonal supercell.'/)
      ELSE IF (LATTYP==4) THEN
! Hexagonal symmetry ('c-axis' is the rotation axis):
         CALL SGRCOP(R6Z,SYMGEN(1,2))
         CALL SGRCOP(R2HEX,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,4)
    4 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' hexagonal supercell.'/)
      ELSE IF (LATTYP==3) THEN
! Cubic symmetry (face centered ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZFC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,3)
    3 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' face centered cubic supercell.'/)
      ELSE IF (LATTYP==2) THEN
! Cubic symmetry (body centered ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,2)
    2 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered cubic supercell.'/)
      ELSE IF (LATTYP==1) THEN
! Cubic symmetry (primitive cell ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,1)
    1 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple cubic supercell.'/)
      END IF
! Now select all symmetry operations which reproduce the lattice:
      CALL GETGRP(S,GTRANS,NROT,NROTK,SYMOP,NOP,TAU, &
     &                           TAUROT,NSP,NSX,NA,NAX,INDEX,WRK,PRCHAN)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE GETGRP(S,GTRANS,NROT,NROTK,SYMOP,NOP,TAU, &
     &                         TAUROT,NSP,NSX,NA,NAX,INDEX,WRK,PRCHAN)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine GETGRP is a secondary user-interface of this symmetry      *
!   package. The function of GETGRP is to return all possible space    *
!   group operators that reproduce a lattice with basis out of a       *
!   (maximum) pool of point group operations that is compatible with   *
!   the symmetry of the pure translation lattice without any basis.    *
!   The setup of the space group operations is done in the following   *
!   way: We pass through the pool of (possibly allowed) symmetry       *
!   operations and check each operation whether it can reproduce the   *
!   lattice with basis - possibly connected with some nontrivial       *
!   translation. This check will be done by routine CHKSYM (further    *
!   details will be described there). If the answer is positive the    *
!   operation will be stored (and counted) - otherwise it will be      *
!   thrown away.                                                       *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      SYMOP(3,3,48) contains the rotation matrices of the pure        *
!                bravais lattice.                                      *
!                                                                      *
!      NOP contains the number of rotation matrices in array SYMOP     *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (see comment on routine SETGRP)!     *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      PRCHAN contains the unit number for standard output.            *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices for all        *
!                space group operations.                               *
!                                                                      *
!      GTRANS(3,48) contains the nonprimitive translations for         *
!                all space group operations.                           *
!                                                                      *
!      NROT contains the number of pure point group rotations (to      *
!           be stored in S(3,3,1)...S(3,3,NROT)) and                   *
!      NROTK contains the number of space group operations found       *
!           in routine GETGRP.                                         *
!                                                                      *
!                                                                      *
!***********************************************************************
      LOGICAL TS
      INTEGER S,SYMOP,ZERO(9),HELP(3,3,48),PRCHAN

      DIMENSION S(3,3,48),GTRANS(3,48),SYMOP(3,3,48),TEMP(3,48),NA(NSP)
      DIMENSION TAU(NAX,3,NSX),TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX)

      SAVE ZERO
      DATA ZERO /0,0,0,0,0,0,0,0,0/

      NROT=0
      NROTK=0
! Check the operations of a 'pool' of all possible operations ... :
      DO 1 I=1,NOP
         CALL CHKSYM(SYMOP(1,1,I),GTRANS(1,I),TS,TAU,TAUROT, &
     &                                         NSP,NSX,NA,NAX,INDEX,WRK)
         IF (TS) THEN
! Hurray! We found some symmetry operation in subroutine CHKSYM ... :
            IF ((ABS(GTRANS(1,I))<TINY) &
     &               .AND.(ABS(GTRANS(2,I))<TINY) &
     &                          .AND.(ABS(GTRANS(3,I))<TINY)) THEN
! Pure point group operations (no nontrivial translation):
               NROT=NROT+1
               CALL SGRCOP(SYMOP(1,1,I),S(1,1,NROT))
               GTRANS(1,NROT)=0._q
               GTRANS(2,NROT)=0._q
               GTRANS(3,NROT)=0._q
            ELSE
! Space group operations:
               NROTK=NROTK+1
               CALL SGRCOP(SYMOP(1,1,I),HELP(1,1,NROTK))
               TEMP(1,NROTK)=GTRANS(1,I)
               TEMP(2,NROTK)=GTRANS(2,I)
               TEMP(3,NROTK)=GTRANS(3,I)
            END IF
         END IF
    1 CONTINUE
      IF (NROTK>0) THEN
! If there are operations with nontrivial translations then store them
! into the array S and GTRANS (placing them after the last pure point
! group operations S(I,J,NROT) ):
         DO 2 I=1,NROTK
            CALL SGRCOP(HELP(1,1,I),S(1,1,I+NROT))
            GTRANS(1,I+NROT)=TEMP(1,I)
            GTRANS(2,I+NROT)=TEMP(2,I)
            GTRANS(3,I+NROT)=TEMP(3,I)
    2    CONTINUE
      END IF
! Total number of space group operations:
      NROTK=NROTK+NROT
      IF (NROTK<48) THEN
! Fill the rest of arrays S and GTRANS with zeros ... :
         DO 3 I=NROTK+1,48
            CALL SGRCOP(ZERO,S(1,1,I))
            GTRANS(1,I)=0._q
            GTRANS(2,I)=0._q
            GTRANS(3,I)=0._q
    3    CONTINUE
      END IF
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) &
     &                                  WRITE(PRCHAN,100) NROTK,NROT,NOP
  100 FORMAT(/' Subroutine GETGRP returns: Found ',I2, &
     &        ' space group operations'/,' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' out of a pool of ',I2,' trial point group operations.'/)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE CHKSYM(S,GTRANS,TS,TAU,TAUROT,NSP,NSX,NA,NAX,INDEX,WRK)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine CHKSYM checks whether a point group symmetry element is    *
!   a valid symmetry operation on a supercell (possibly connected with *
!   some nontrivial translation) or not. The procedure is in principle *
!   very easy: The point group transformation will be imposed on all   *
!   atomic positions (for all species) and then the rotated supercell  *
!   will be compared with the original cell. There are only three      *
!   possibilities: All positions will be reproduced (pure point group  *
!   operation), all positions will be reproduced after an additional   *
!   unique translation of all atoms (general space group operation)    *
!   or it will not be possible to reproduce the supercell in any way   *
!   (no allowed operation). The translation can be guessed quite easy  *
!   by looking on the first atom: If we try all difference vectors     *
!   between the position of the first atom of the unrotated cell and   *
!   the positions of all atoms of the rotated cell then one of these   *
!   difference vectors must be the translation vector (if it exists!). *
!   This vector must apply to all atoms. So we translate all atoms in  *
!   the rotated cell by minus this vector - this should reproduce the  *
!   original positions of the unrotated cell. The central point is now *
!   only how to compare the rotated and the unrotated supercell. The   *
!   algorithm used bases on the following idea: If we sort all atomic  *
!   positions within a cell as it is described in routine LATORD of    *
!   the lattice vector package (sorting first the x-coordinates, then  *
!   sorting the y-coordinates of all atoms with equal x-coordinates    *
!   and finally sorting all z-cordinates for all atomic positions with *
!   equal x- and y-coordinates  -  or using other ordering sequences   *
!   instead of xyz-ordering ...) then we get some unique sequence of   *
!   ordered positions. Because the same set of input data will mean    *
!   identical output we can simply perform a one-by-one comparison of  *
!   the unrotated (ordered) and the rotated (ordered and translated)   *
!   coordinates. If we have found some group operation the comparison  *
!   will give the value true ('all positions are equal'). In all other *
!   cases the result will be false ('some positions are not equal'!    *
!   There is just one point one has to care about: Supercells which    *
!   are non-primitive. Then there exist also 'non-primitive primitive' *
!   translations (trivial translations of the generating cell ...)!    *
!   To avoid trouble with non-uniquely defined translations in this    *
!   case we just take the vectors with smallest coordinates (from all  *
!   the vectors which will be detected to be an allowed translation).  *
!                                                                      *
!   It should be trivial but will be remarked finally: The procedure   *
!   must be performed on every atomic species seperately and the       *
!   result (allowed operation or not / nontrivial translation) must be *
!   the same for all atomic species - otherwise the symmetry operation *
!   is not a valid operation ... .                                     *
!                                                                      *
!   Final very important notice: Comparison of atomic positions will   *
!   be done by looking at differences of coordinates (which must be    *
!   smaller than some tolerance). The allowed tolerances for the       *
!   coordinates are very small (1E-9 absolute)!!! Therefore all input  *
!   coordinates should be given with an accuray of at least +/-1E-10   *
!   because otherwise this routine could work incorrectly (could fail  *
!   to detect space group operations ...)!                             *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S(3,3) is the point group symmetry element to be checked for    *
!             validity (INTEGER rotation matrix).                      *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (see comment on routine SETGRP)!     *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      GTRANS(3) contains the nonprimitive translations which is       *
!                connected with the rotation matrix S(3,3).            *
!                                                                      *
!      TS is a logical value which returns whether S(3,3) is a valid   *
!         symmetry operation on the supercell or not.                  *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TS,TNTRAN,TCONT
      INTEGER S

      DIMENSION S(3,3),GTRANS(3),NA(NSP),TAU(NAX,3,NSX),TAUDIF(3)
      DIMENSION TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX),TAUSAV(3),TRA(3)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISMIN=1
      TRA(1)=2._q
      TRA(2)=2._q
      TRA(3)=2._q
      TS=.FALSE.
      ISTART=1
      IMINST=1
! For all atomic species ... :
      DO 3 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Search the species with smallest number of atoms ... :
         IF (NA(IS)<NA(ISMIN)) THEN
            ISMIN=IS
            IMINST=ISTART
         END IF
         DO 1 IA=ISTART,NA(IS)+ISTART-1
! First impose periodic boundary conditions on TAU (--> [-0.5,0.5) ):
            TAU(IA,1,ISP)=TAU(IA,1,ISP)-NINT(TAU(IA,1,ISP))
            TAU(IA,2,ISP)=TAU(IA,2,ISP)-NINT(TAU(IA,2,ISP))
            TAU(IA,3,ISP)=TAU(IA,3,ISP)-NINT(TAU(IA,3,ISP))
            TAU(IA,1,ISP)=MOD(TAU(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,2,ISP)=MOD(TAU(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,3,ISP)=MOD(TAU(IA,3,ISP)+100.5_q,1._q)-0.5_q
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
#ifdef  periodic_image_convention
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=TAU(IA,1,ISP)-1.0_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=TAU(IA,2,ISP)-1.0_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=TAU(IA,3,ISP)-1.0_q
#else
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
#endif
    1    CONTINUE
! Order the atomic coordiantes (for this species) ... :
         CALL LATORD(TAU(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
! Rotate all coordinates ... :
         DO 2 IA=ISTART,NA(IS)+ISTART-1
            TAUROT(IA,1,ISP)=S(1,1)*TAU(IA,1,ISP)+ &
     &                                S(2,1)*TAU(IA,2,ISP)+ &
     &                                              S(3,1)*TAU(IA,3,ISP)
            TAUROT(IA,2,ISP)=S(1,2)*TAU(IA,1,ISP)+ &
     &                                S(2,2)*TAU(IA,2,ISP)+ &
     &                                              S(3,2)*TAU(IA,3,ISP)
            TAUROT(IA,3,ISP)=S(1,3)*TAU(IA,1,ISP)+ &
     &                                S(2,3)*TAU(IA,2,ISP)+ &
     &                                              S(3,3)*TAU(IA,3,ISP)
! ... and impose periodic boundary conditions (TAUROT --> [-0.5,0.5) ):
            TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
#ifdef  periodic_image_convention
            IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-1.0_q
            IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-1.0_q
            IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-1.0_q
#else
            IF ((ABS(TAUROT(IA,1,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,2,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,3,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
#endif
    2    CONTINUE
! Order the rotated atomic coordiantes ... :
         CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
    3 CONTINUE
! Save the rotated coordinate of the first atom of species ISMIN:
      IF (TCONT) THEN
         TAUSAV(1)=TAUROT(IMINST,1,1)
         TAUSAV(2)=TAUROT(IMINST,2,1)
         TAUSAV(3)=TAUROT(IMINST,3,1)
         ISPMIN=1
      ELSE
         TAUSAV(1)=TAUROT(1,1,ISMIN)
         TAUSAV(2)=TAUROT(1,2,ISMIN)
         TAUSAV(3)=TAUROT(1,3,ISMIN)
         ISPMIN=ISMIN
      END IF
! All difference vectors between the first rotated atom and all possible
! original vectors of species ISMIN could be a translation vector being
! associated with the given rotation ... . So test all possibilities:
      DO 10 IATEST=IMINST,NA(ISMIN)+IMINST-1
! Set up the 'test vector' GTRANS ... :
         GTRANS(1)=TAU(IATEST,1,ISPMIN)-TAUSAV(1)
         GTRANS(2)=TAU(IATEST,2,ISPMIN)-TAUSAV(2)
         GTRANS(3)=TAU(IATEST,3,ISPMIN)-TAUSAV(3)
! GTRANS could possibly contain trivial translations:
         GTRANS(1)=MOD((GTRANS(1)+100._q),1._q)
         GTRANS(2)=MOD((GTRANS(2)+100._q),1._q)
         GTRANS(3)=MOD((GTRANS(3)+100._q),1._q)
! For reasons of safety:
#ifdef  periodic_image_convention
         IF ((ABS(GTRANS(1)-1._q))<(TINY*0.5_q)) GTRANS(1)=GTRANS(1)-1._q
         IF ((ABS(GTRANS(2)-1._q))<(TINY*0.5_q)) GTRANS(2)=GTRANS(2)-1._q
         IF ((ABS(GTRANS(3)-1._q))<(TINY*0.5_q)) GTRANS(3)=GTRANS(3)-1._q
#else
         IF ((ABS(GTRANS(1)-1._q))<(TINY*0.5_q)) GTRANS(1)=0._q
         IF ((ABS(GTRANS(2)-1._q))<(TINY*0.5_q)) GTRANS(2)=0._q
         IF ((ABS(GTRANS(3)-1._q))<(TINY*0.5_q)) GTRANS(3)=0._q
#endif
! If we had already detected some translation (non-primitve supercell?)
! we must only look at the vectors with coordinates smaller than those
! of the previously detected vector ... :
         IF ((GTRANS(1)>(TRA(1)+TINY)).OR. &
     &                 (GTRANS(2)>(TRA(2)+TINY)).OR. &
     &                             (GTRANS(3)>(TRA(3)+TINY))) GOTO 10
! Translate the rotated coordinates by GTRANS ...
         ISTART=1
         DO 5 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 4 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)+GTRANS(1)
               TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)+GTRANS(2)
               TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)+GTRANS(3)
! ... impose the periodic boundary condition on these coordinates ...
               TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! ... ensuring that all surface coordinates are put to -0.5 ...
#ifdef  periodic_image_convention
               IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-1.0_q
               IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-1.0_q
               IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-1.0_q
#else
               IF ((ABS(TAUROT(IA,1,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,2,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,3,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
#endif
    4       CONTINUE
! ... and order the coordinates:
            CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
    5    CONTINUE
! Now compare the two lattices 'one-by-one' whether they are identical
! (possibly differing by a translation vector GTRANS ...) or not ... .
         TNTRAN=.TRUE.
! For all atoms of the unit cell ... :
         ISTART=1
         DO 7 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 6 IA=ISTART,NA(IS)+ISTART-1
#ifdef periodic_image_convention
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
               TAUDIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               TAUDIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               TAUDIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
               TAUDIF(1)=MIN_IMAG(TAUDIF(1))
               TAUDIF(2)=MIN_IMAG(TAUDIF(2))
               TAUDIF(3)=MIN_IMAG(TAUDIF(3))
#else
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
               TAUDIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               TAUDIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               TAUDIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
               TAUDIF(1)=MOD((TAUDIF(1)+100._q),1._q)
               TAUDIF(2)=MOD((TAUDIF(2)+100._q),1._q)
               TAUDIF(3)=MOD((TAUDIF(3)+100._q),1._q)
! For reasons of safety:
               IF ((ABS(TAUDIF(1)-1._q))<(TINY*0.5_q)) TAUDIF(1)=0._q
               IF ((ABS(TAUDIF(2)-1._q))<(TINY*0.5_q)) TAUDIF(2)=0._q
               IF ((ABS(TAUDIF(3)-1._q))<(TINY*0.5_q)) TAUDIF(3)=0._q
#endif
! If TAUDIF is not equal (0.,0.,0.) TNTRAN=TNTRAN.AND.FALSE and so it
! remains FALSE forever ... (if only one position is not reproduced!).
! Only if all TAUDIFs are zero TNTRAN (starting with the value TRUE)
! will remain TRUE (that means we found an allowed symmetry operation).
               TNTRAN=TNTRAN.AND.((ABS(TAUDIF(1))<TINY) &
     &                            .AND.(ABS(TAUDIF(2))<TINY) &
     &                                    .AND.(ABS(TAUDIF(3))<TINY))
    6       CONTINUE
    7    CONTINUE
! Check was successful ... :
         IF (TNTRAN) THEN
            TS=.TRUE.
! Save the detected translation vector temporarily ... :
            TRA(1)=GTRANS(1)
            TRA(2)=GTRANS(2)
            TRA(3)=GTRANS(3)
         END IF
! Now we must take the next testvector ... . Therefore restore the
! original rotated coordinates by subtracting GTRANS ... :
         ISTART=1
         DO 9 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 8 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-GTRANS(1)
               TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-GTRANS(2)
               TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-GTRANS(3)
    8       CONTINUE
    9    CONTINUE
   10 CONTINUE
      IF (TS) THEN
         GTRANS(1)=TRA(1)
         GTRANS(2)=TRA(2)
         GTRANS(3)=TRA(3)
      END IF
! Uff ... . Bye bye!
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE DYNSYM(V,MAP,S,T,NROTK,NROT,PT,NP, &
     &                                   NSX,NSP,NAX,NA,NSAX,VR,W,IND,P)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine DYNSYM tests whether an array of vectors (e.g. velocities) *
!   located at different atomic positions is compatible with a given   *
!   symmetry operation. The test is done by symmetrizing the vectors   *
!   as in routine FSYM (see above) and comparing the result to the     *
!   non-symmetrized vectors. If the test will be negative the symmetry *
!   operation will be discarded from the set of given operations.      *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      V(NAX,3,NSX) contains the crystal coordinates of the vectors    *
!              located at the corresponding atomic position.           *
!                                                                      *
!      MAP(NAX,NSX,48) contains a table connecting the rotated and     *
!                   the unrotated positions (stored is the index of    *
!                   the rotated position = first index in array TAU).  *
!                   MAP can be generated by routine CLASS (see below). *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      T(3,48) contains the nonprimitive translations.                 *
!      NROTK contains the number of given space group operations.      *
!      NROT contains the number of pure point group operations.        *
!      PT contains all "primitive" translations associated with the    *
!                  generating unit cell of a nonprimitive supercell    *
!      NP is the number of "primitive" translations in PT              *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX, NAX and NSAX are array dimensioning parameters.            *
!                                                                      *
!      VR, W and IND are work arrays.                                  *
!                                                                      *
!      P contains the unit number for standard output.                 *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      Arrays S and T contain the compatible operations on output      *
!      (input is overwritten!), NROTK and NROT the numbers of elements *
!      in S and T being space group / point group operations.          *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S,GRP,P,OLDNR,OLDNRK
      LOGICAL TEST,TCONT

      DIMENSION V(NAX,3,NSX),MAP(NAX,NSX,NROTK,NP),S(3,3,48),NA(NSP)
      DIMENSION VR(NAX,3),GRP(3,3,48),T(3,48),TCOP(3,48),IM(48)
      DIMENSION PT(NSAX+2,3),IND(NSAX),W(NAX,3)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      OLDNR=NROT
      OLDNRK=NROTK
      DO IROT=1,48
         IM(IROT)=IROT
      ENDDO
      ISTART=1
! For all atomic species ... :
      DO 7 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! First copy the rotation matrices and translation vectors:
         GRP=S
         TCOP=T
         S=0
         T=0
! Copy the input vectors:
         DO 3 IA=ISTART,NA(IS)+ISTART-1
            W(IA,1)=V(IA,1,ISP)
            W(IA,2)=V(IA,2,ISP)
            W(IA,3)=V(IA,3,ISP)
    3    CONTINUE
         NR=0
! Apply all space group operations on the vectors:
         DO 6 IROT=1,NROTK
! Calculate the rotated vectors for all atoms:
            DO 4 IA=ISTART,NA(IS)+ISTART-1
               VR(IA,1)=GRP(1,1,IROT)*W(IA,1)+ &
     &                               GRP(2,1,IROT)*W(IA,2)+ &
     &                                             GRP(3,1,IROT)*W(IA,3)
               VR(IA,2)=GRP(1,2,IROT)*W(IA,1)+ &
     &                               GRP(2,2,IROT)*W(IA,2)+ &
     &                                             GRP(3,2,IROT)*W(IA,3)
               VR(IA,3)=GRP(1,3,IROT)*W(IA,1)+ &
     &                               GRP(2,3,IROT)*W(IA,2)+ &
     &                                             GRP(3,3,IROT)*W(IA,3)
    4       CONTINUE
! Compare them with the unrotated vectors:
            TEST=.TRUE.
            DO 5 IA=ISTART,NA(IS)+ISTART-1
! TEST will remain true if all rotated vectors are equal to the original
! vectors; otherwise TEST will become false!
               JA=MAP(IA,ISP,IM(IROT),1)
               TEST=TEST.AND. &
     &                (ABS(V(IA,1,ISP)-VR(JA,1))<TINY).AND. &
     &                     (ABS(V(IA,2,ISP)-VR(JA,2))<TINY).AND. &
     &                               (ABS(V(IA,3,ISP)-VR(JA,3))<TINY)
    5       CONTINUE
! Test was successful:
            IF (TEST) THEN
               NR=NR+1
               IF (IROT<=NROT) NRP=NR
! Store the symmetry operation:
               CALL SGRCOP(GRP(1,1,IROT),S(1,1,NR))
               T(1,NR)=TCOP(1,IROT)
               T(2,NR)=TCOP(2,IROT)
               T(3,NR)=TCOP(3,IROT)
               IM(NR)=IROT
            END IF
    6    CONTINUE
! New number of operations:
         NROT=NRP
         NROTK=NR
    7 CONTINUE
! And now a similar game for the 'primitive' translations:
      IF (NP==1) GOTO 13
      DO ITRANS=1,NP
         IND(ITRANS)=ITRANS
      ENDDO
      ISTART=1
      DO 12 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! First copy all translation vectors:
         DO 9 ITRANS=1,NP
            W(ITRANS,1)=PT(ITRANS,1)
            W(ITRANS,2)=PT(ITRANS,2)
            W(ITRANS,3)=PT(ITRANS,3)
            PT(ITRANS,1)=0._q
            PT(ITRANS,2)=0._q
            PT(ITRANS,3)=0._q
    9    CONTINUE
         NPRI=0
         TEST=.TRUE.
! Apply all primitive translations:
         DO 11 ITRANS=1,NP
! Compare vectors with vectors at translated positions:
            DO 10 IA=ISTART,NA(IS)+ISTART-1
! TEST will remain true if all translated vectors are equal to the
! original vectors; otherwise TEST will become false!
               JA=MAP(IA,ISP,1,IND(ITRANS))
               TEST=TEST.AND. &
     &               (ABS(V(IA,1,ISP)-V(JA,1,ISP))<TINY).AND. &
     &                   (ABS(V(IA,2,ISP)-V(JA,2,ISP))<TINY).AND. &
     &                            (ABS(V(IA,3,ISP)-V(JA,3,ISP))<TINY)
   10       CONTINUE
! Test was successful:
            IF (TEST) THEN
               NPRI=NPRI+1
! Store the symmetry operation:
               PT(NPRI,1)=W(ITRANS,1)
               PT(NPRI,2)=W(ITRANS,2)
               PT(NPRI,3)=W(ITRANS,3)
               IND(NPRI)=ITRANS
            ENDIF
   11    CONTINUE
! New number of operations:
         NP=NPRI
   12 CONTINUE
! Ready ... !
   13 IF ((P>=0).AND.(P<=99)) WRITE(P,50) NROTK,NROT,OLDNRK,OLDNR,NP
   50 FORMAT(/' Subroutine DYNSYM returns: Found ',I2, &
     &        ' space group operations'/,' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' out of a pool of ',I2,' trial space group operations'/, &
     &        ' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' and found also ',I5,' ''primitive'' translations'/)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE MAGSYM(ATOMOM,MAGROT,MAP,S,T,NROTK,NROT,PT,NP, &
     &                                 NSX,NSP,NAX,NA,NSAX,MAGR,W,IND,P)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine MAGSYM tests whether a given pattern of magnetic moments   *
!   located at different atomic positions is compatible with a given   *
!   symmetry operation. The test is done by checking atomic magnetic   *
!   moments at symmetry-related atoms -- they can be identical or may  *
!   differ by a factor (-1). If the test will be negative the symmetry *
!   operation will be discarded from the set of given operations. If   *
!   the test was positive the connection factor (1 or -1) is stored.   *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      ATOMOM(NAX,NSX) contains the atomic magnetic moments located    *
!                      at the corresponding atomic position.           *
!                                                                      *
!      MAP(NAX,NSX,48) contains a table connecting the rotated and     *
!                   the unrotated positions (stored is the index of    *
!                   the rotated position = first index in array TAU).  *
!                   MAP can be generated by routine CLASS (see below). *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      T(3,48) contains the nonprimitive translations.                 *
!      NROTK contains the number of given space group operations.      *
!      NROT contains the number of pure point group operations.        *
!      PT contains all "primitive" translations associated with the    *
!                  generating unit cell of a nonprimitive supercell    *
!      NP is the number of "primitive" translations in PT              *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX, NAX and NSAX are array dimensioning parameters.            *
!                                                                      *
!      MAGR, W and IND are work arrays.                                *
!                                                                      *
!      P contains the unit number for standard output.                 *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      Array MAGMOM(48,NP) contains the connection factors between a   *
!      magnetic moment pattern and the pattern at the rotated atomic   *
!      positions. In the ISING case we have two possibilities: 1 or -1 *
!                                                                      *
!      Arrays S and T contain the compatible operations on output      *
!      (input is overwritten!), NROTK and NROT the numbers of elements *
!      in S and T being space group / point group operations.          *
!                                                                      *
!                                                                      *
!***********************************************************************
 
      REAL(q) MAGROT,MAGR,MR
      INTEGER S,GRP,P,OLDNR,OLDNRK
      LOGICAL TEST,TCONT

      DIMENSION MAGROT(48,NSAX),MAP(NAX,NSX,NROTK,NP),S(3,3,48),NA(NSP)
      DIMENSION MAGR(NAX,3),GRP(3,3,48),T(3,48),TCOP(3,48),IM(48)
      DIMENSION PT(NSAX+2,3),IND(NSAX),W(NAX,3),ATOMOM(NAX,NSX)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      OLDNR=NROT
      OLDNRK=NROTK
      DO IROT=1,48
         IM(IROT)=IROT
      ENDDO
! Initialize array MAGMOM:
      MAGROT=0
! The first task will be to search the first atom with a non-vanishing moment
      TEST=.TRUE.
      IAMAG=0
      ISMAG=0
      ISTART=1
      DO 4 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
         DO 3 IA=ISTART,NA(IS)+ISTART-1
            TEST=TEST.AND.(ABS(ATOMOM(IA,ISP))<TINY)
            IF (.NOT.TEST) THEN
               IAMAG=IA
               ISMAG=ISP
               GOTO 5
            ENDIF
    3    CONTINUE
    4 CONTINUE
! all moments zero? -> no symmetry breaking!
    5 IF (TEST) THEN
         DO 7 ITRANS=1,NP
            DO 6 IROT=1,NROTK
               MAGROT(IROT,ITRANS)=1._q
    6       CONTINUE
    7    CONTINUE
         GOTO 19
      ENDIF
      ISTART=1
! For all atomic species ... :
      DO 12 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! First copy the rotation matrices and translation vectors:
         GRP=S
         TCOP=T
         S=0
         T=0
         NR=0
! Apply all space group operations to the atomic coordinates:
         DO 11 IROT=1,NROTK
! Rotate the first atom of the first species having a non-zero moment:
            JA=MAP(IAMAG,ISMAG,IM(IROT),1)
            TEST =(ABS(ABS(ATOMOM(IAMAG,ISMAG))-ABS(ATOMOM(JA,ISMAG)))<TINY)
! Expected transformation behaviour of the atomic moments for this rotation
! hint: if you want to extend the routine to non-collinear spin structures the
!       generalization would be to set an appropriate 3x3 transformation matrix
            MR=0._q
            IF (ABS(ATOMOM(IAMAG,ISMAG)-ATOMOM(JA,ISMAG))<TINY) MR= 1._q
! CHECK: IF YOU COMMENT OUT FOLLOWING LINE A COMPLETE SYMMETRY BREAKING OCCURS
            IF (ABS(ATOMOM(IAMAG,ISMAG)+ATOMOM(JA,ISMAG))<TINY) MR=-1._q
! Set the corresponding testmap of magnetic moments at all atomic positions
! hint: if you want to extend the routine to non-collinear spin structures the
!       generalization would be to apply the 3x3 transformation matrix from
!       above to a vector ATOMOM(1:3,IA,ISP) yielding a vector MAGR(JA,1:3)
            DO 9 IA=ISTART,NA(IS)+ISTART-1
               JA=MAP(IA,ISP,IM(IROT),1)
               MAGR(IA,3)=ATOMOM(JA,ISP)*MR
    9       CONTINUE
! If test did not fail already for the first atom ...
            IF (TEST) THEN
               DO 10 IA=ISTART,NA(IS)+ISTART-1
! TEST will remain true if all magnetic moments at the rotated positions are
! equal to the "expected moments" MAGR; otherwise TEST will become false!
                  TEST=TEST.AND.(ABS(ATOMOM(IA,ISP)-MAGR(IA,3))<TINY)
   10          CONTINUE
            ENDIF
! Test was successful:
            IF (TEST) THEN
               NR=NR+1
               IF (IROT<=NROT) NRP=NR
! Store the symmetry operation:
               CALL SGRCOP(GRP(1,1,IROT),S(1,1,NR))
               T(1,NR)=TCOP(1,IROT)
               T(2,NR)=TCOP(2,IROT)
               T(3,NR)=TCOP(3,IROT)
               IM(NR)=IROT
! ... and the transformation matrix for the magnetic moment
               MAGROT(NR,1)=MR
            END IF
   11    CONTINUE
! New number of operations:
         NROT=NRP
         NROTK=NR
   12 CONTINUE
! And now a similar game for the 'primitive' translations:
      IF (NP==1) GOTO 19
      DO ITRANS=1,NP
         IND(ITRANS)=ITRANS
      ENDDO
      ISTART=1
      DO 18 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! First copy all translation vectors:
         DO 14 ITRANS=1,NP
            W(ITRANS,1)=PT(ITRANS,1)
            W(ITRANS,2)=PT(ITRANS,2)
            W(ITRANS,3)=PT(ITRANS,3)
            PT(ITRANS,1)=0._q
            PT(ITRANS,2)=0._q
            PT(ITRANS,3)=0._q
   14    CONTINUE
         NPRI=0
         TEST=.TRUE.
! Apply all primitive translations:
         DO 17 ITRANS=1,NP
! Translate the first atom of the first species having a non-zero moment:
            JA=MAP(IAMAG,ISMAG,1,IND(ITRANS))
            TEST =(ABS(ABS(ATOMOM(IAMAG,ISMAG))-ABS(ATOMOM(JA,ISMAG)))<TINY)
! Expected transformation behaviour of the atomic moments for this rotation
! hint: if you want to extend the routine to non-collinear spin structures the
!       generalization would be to set an appropriate 3x3 transformation matrix
            MR=0._q
            IF (ABS(ATOMOM(IAMAG,ISMAG)-ATOMOM(JA,ISMAG))<TINY) MR= 1._q
! CHECK: IF YOU COMMENT OUT FOLLOWING LINE A COMPLETE SYMMETRY BREAKING OCCURS
            IF (ABS(ATOMOM(IAMAG,ISMAG)+ATOMOM(JA,ISMAG))<TINY) MR=-1._q
! Set the corresponding testmap of magnetic moments at all atomic positions
! hint: if you want to extend the routine to non-collinear spin structures the
!       generalization would be to apply the 3x3 transformation matrix from
!       above to a vector ATOMOM(1:3,IA,ISP) yielding a vector MAGR(JA,1:3)
            DO 15 IA=ISTART,NA(IS)+ISTART-1
               JA=MAP(IA,ISP,1,IND(ITRANS))
               MAGR(IA,3)=ATOMOM(JA,ISP)*MR
   15       CONTINUE
! If test did not fail already for the first atom ...
            IF (TEST) THEN
               DO 16 IA=ISTART,NA(IS)+ISTART-1
! TEST will remain true if all magnetic moments at the trasnlated positions
! are equal to the "expected moments" MAGR; otherwise TEST will become false!
                  TEST=TEST.AND.(ABS(ATOMOM(IA,ISP)-MAGR(IA,3))<TINY)
   16          CONTINUE
            ENDIF
! Test was successful:
            IF (TEST) THEN
               NPRI=NPRI+1
! Store the symmetry operation:
               PT(NPRI,1)=W(ITRANS,1)
               PT(NPRI,2)=W(ITRANS,2)
               PT(NPRI,3)=W(ITRANS,3)
               IND(NPRI)=ITRANS
! ... and the transformation matrix for the magnetic moment
               MAGROT(1,NPRI)=MR
            ENDIF
   17    CONTINUE
! New number of operations:
         NP=NPRI
   18 CONTINUE
! Ready ... !
   19 IF ((P>=0).AND.(P<=99)) WRITE(P,50) NROTK,NROT,OLDNRK,OLDNR,NP
! And now the most interesting moment: I claim that combined application of
! a rotation IROT and a translation ITRANS should result in a multiplication
! of the corresponding transformations for the magnetic moment pattern ... :
      DO 21 IROT=2,NROTK
         DO 20 ITRANS=2,NP
            MAGROT(IROT,ITRANS)=MAGROT(IROT,1)*MAGROT(1,ITRANS)  ! can this be true?
   20    CONTINUE
   21 CONTINUE
   50 FORMAT(/' Subroutine MAGSYM returns: Found ',I2, &
     &        ' space group operations'/,' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' out of a pool of ',I2,' trial space group operations'/, &
     &        ' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' and found also ',I5,' ''primitive'' translations'/)
      RETURN
      END

!***********************************************************************
!                                                                      *
      SUBROUTINE PSYM(TAU,MAP,S,T,PT,NR,NP,NAX,NSX,NSP,NA,TAUP)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine PSYM (re-)symmetrizes the atomic positions                 *
!                                                                      *
!***********************************************************************
      INTEGER S
      LOGICAL TCONT 

      DIMENSION TAU(NAX,3,NSX),MAP(NAX,NSX,NR,NP),NA(NSP)
      DIMENSION S(3,3,48),T(3,48),PT(3,NP),TAUP(NAX,3,NSX)
      DIMENSION TAUROT(3),D(3)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      TAUP=0._q; ISTART=1
! For all atomic species ... :
      species: DO IS=1,NSP
         ISP=IS; IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Loop over all atoms of species IS
         ions: DO IA=ISTART,NA(IS)+ISTART-1
! Loop over all spacegroup operations
            DO IROT=1,NR
! And over all primitive translations
               DO ITRANS=1,NP
! Symmetry operation (irot,itrans) brings atom JA to IA
                  JA=MAP(IA,ISP,IROT,ITRANS)
! Apply (irot,itrans) to JA
                  TAUROT(1)=S(1,1,IROT)*TAU(JA,1,ISP)+ &
     &                           S(2,1,IROT)*TAU(JA,2,ISP)+ &
     &                               S(3,1,IROT)*TAU(JA,3,ISP)+ &
     &                                            PT(ITRANS,1)+T(1,IROT)
                  TAUROT(2)=S(1,2,IROT)*TAU(JA,1,ISP)+ &
     &                           S(2,2,IROT)*TAU(JA,2,ISP)+ &
     &                               S(3,2,IROT)*TAU(JA,3,ISP)+ &
     &                                            PT(ITRANS,2)+T(2,IROT)
                  TAUROT(3)=S(1,3,IROT)*TAU(JA,1,ISP)+ &
     &                           S(2,3,IROT)*TAU(JA,2,ISP)+ &
     &                               S(3,3,IROT)*TAU(JA,3,ISP)+ &
     &                                            PT(ITRANS,3)+T(3,IROT)
! Bring TAUROT in a minimal image w.r.t. atom IA
                  D(:)=TAUROT(:)-TAU(IA,:,ISP)
                  DO IK=1,3
                     IF (D(IK)>0.5_q) D(IK)=D(IK)-NINT(D(IK))
                     IF (D(IK)<-.5_q) D(IK)=D(IK)+NINT(D(IK))
                  ENDDO
                  TAUROT(:)=TAU(IA,:,ISP)+D(:)
! Add to results 
                  TAUP(IA,:,ISP)=TAUP(IA,:,ISP)+TAUROT(:)
               ENDDO
            ENDDO
         ENDDO ions
      ENDDO species
! Average over the number of operations
      TAUP=TAUP/NR/NP

      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE PSYMMAKE(TAU,NPOSIN,S,T,PT,NR,NP,TAUSTAR,IA,NPOSROTMAX)
      USE prec
      USE sym_prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine PSYMMAKE                                                   *
!     input: TAU is an array of NPOSIN (virtual) atomic positions      *
!     output: TAUSTAR contains all IA symmetry equivalent positions    *
!             including TAU.                                           *
!                                                                      *
!***********************************************************************
      INTEGER S
      LOGICAL TCONT 

      DIMENSION TAU(NPOSIN,3)
      DIMENSION S(3,3,48),T(3,48),PT(NP,3)
      DIMENSION TAUSTAR(NR*NP*NPOSIN,3)   ! Doorgeven TAUSTAR moet anders

      DIMENSION TAUROT(3),D(3),TAUDIF(3)
      REAL(q),ALLOCATABLE :: TAUP(:,:)
      REAL(q) :: ATOL

      ATOL=1.E-10_q

      ALLOCATE (TAUP(NR*NP*NPOSIN,3))

      IF ((NR*NP*NPOSIN).NE.NPOSROTMAX) THEN
         CALL ERROR('PSYMMAKE ',' ARRAY DIMENSION WRONG',NPOSROTMAX)
         STOP
      ENDIF

! Now start ... :
      TAUP=0._q
! The site that generates the others is always at the start of the array
      IA=NPOSIN
      DO IJ=1,NPOSIN
        TAUP(IJ,:)=TAU(IJ,:)
      ENDDO
!     TAUP(1,:)=TAU(:)
! Loop over all NPOSIN atoms
      DO IJ=1,NPOSIN
! Loop over all spacegroup operations
         DO IROT=1,NR
! And over all primitive translations
            DO ITRANS=1,NP
! Apply (irot,itrans) to JA
               TAUROT(1)=S(1,1,IROT)*TAU(IJ,1)+ &
     &                   S(2,1,IROT)*TAU(IJ,2)+ &
     &                   S(3,1,IROT)*TAU(IJ,3)+ &
     &                                         PT(ITRANS,1)+T(1,IROT)
               TAUROT(2)=S(1,2,IROT)*TAU(IJ,1)+ &
     &                   S(2,2,IROT)*TAU(IJ,2)+ &
     &                   S(3,2,IROT)*TAU(IJ,3)+ &
     &                                         PT(ITRANS,2)+T(2,IROT)
               TAUROT(3)=S(1,3,IROT)*TAU(IJ,1)+ &
     &                   S(2,3,IROT)*TAU(IJ,2)+ &
     &                   S(3,3,IROT)*TAU(IJ,3)+ &
     &                                         PT(ITRANS,3)+T(3,IROT)
! Bring TAUROT in a minimal image w.r.t. 
               D(:)=TAUROT(:)-TAU(IJ,:)
               DO IL=1,3
               DO IK=1,3
                  IF (D(IK)>=0.5_q) D(IK)=D(IK)-NINT(D(IK))
                  IF (D(IK)<-.5_q) D(IK)=D(IK)-NINT(D(IK))
               ENDDO
               ENDDO
               TAUROT(:)=TAU(IJ,:)+D(:)
! Add to results 

               DO IK=1,IA
#ifdef periodic_image_convention
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
                  TAUDIF(1)=TAUP(IK,1)-TAUROT(1)
                  TAUDIF(2)=TAUP(IK,2)-TAUROT(2)
                  TAUDIF(3)=TAUP(IK,3)-TAUROT(3)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
                  TAUDIF(1)=MIN_IMAG(TAUDIF(1))
                  TAUDIF(2)=MIN_IMAG(TAUDIF(2))
                  TAUDIF(3)=MIN_IMAG(TAUDIF(3))
#else
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
                  TAUDIF(1)=TAUP(IK,1)-TAUROT(1)
                  TAUDIF(2)=TAUP(IK,2)-TAUROT(2)
                  TAUDIF(3)=TAUP(IK,3)-TAUROT(3)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
                  TAUDIF(1)=MOD((TAUDIF(1)+100._q),1._q)
                  TAUDIF(2)=MOD((TAUDIF(2)+100._q),1._q)
                  TAUDIF(3)=MOD((TAUDIF(3)+100._q),1._q)
! For reasons of safety:
                  IF ((ABS(TAUDIF(1)-1._q))<(TINY*0.5_q)) TAUDIF(1)=0._q
                  IF ((ABS(TAUDIF(2)-1._q))<(TINY*0.5_q)) TAUDIF(2)=0._q
                  IF ((ABS(TAUDIF(3)-1._q))<(TINY*0.5_q)) TAUDIF(3)=0._q
#endif
! If TAUDIF is not equal (0.,0.,0.) TNTRAN=TNTRAN.AND.FALSE and so it
! remains FALSE forever ... (if only one position is not reproduced!).
! Only if all TAUDIFs are zero TNTRAN (starting with the value TRUE)
! will remain TRUE (that means we found an allowed symmetry operation).
                  IF (      (ABS(TAUDIF(1))<TINY) &
     &                 .AND.(ABS(TAUDIF(2))<TINY) &
     &                 .AND.(ABS(TAUDIF(3))<TINY)) GOTO 10
               ENDDO
               IA=IA+1
               IF (IA.GT.(NR*NP*NPOSIN)) THEN
                  CALL ERROR('PSYMMAKE ',' Too much symmetry',IA)
                  STOP
               ENDIF
               TAUP(IA,:)=TAUROT(:)
   10          CONTINUE
            ENDDO
         ENDDO
      ENDDO

      DO IK=1,IA
        TAUSTAR(IK,:)=TAUP(IK,:)
      ENDDO

      DEALLOCATE(TAUP)

      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE FSYM(F,MAP,S,NR,NP,NSX,NSP,NAX,NA,FR,W, &
     &                                                A1,A2,A3,B1,B2,B3)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine FSYM symmetrizes an array of vectors (forces, velocities)  *
!   located at different atomic positions. The procedure is a quite    *
!   simple: Rotate the input vector and add it to the vector at the    *
!   rotated atomic position (for all space group operations ...).      *
!   Finally divide the result by the number of symmetry operations!    *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      F(3,NAX,NSX) contains the cartesian coordinates of the vector   *
!              located at the corresponding atomic position.           *
!                                                                      *
!      MAP(NAX,NSX,NROTK,NP) contains a table connecting the rotated   *
!                   and the unrotated positions (stored is the index of*
!                   the rotated position = first index in array TAU).  *
!                   MAP can be generated by routine CLASS (see below). *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NR contains the number of given space group operations.         *
!      NP contains the number of "primitive" translations in the cell. *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      FR and W are work arrays.                                       *
!                                                                      *
!      A1, A2 and A3 contain the direct (real space) lattice vectors.  *
!      B1, B2 and B3 contain the reciprocal lattice vectors.           *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      Array F contains the symmetrized vectors on output (input is    *
!      overwritten!).                                                  *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT
      INTEGER S

      DIMENSION F(3,NAX,NSX),MAP(NAX,NSX,NR,NP),S(3,3,48),NA(NSP)
      DIMENSION FR(NAX,3),W(NAX,3),A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISTART=1
! For all atomic species ... :
      DO 7 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! First transform the cartesian coordiates to crystal coordinates ... :
         DO 1 IA=ISTART,NA(IS)+ISTART-1
            W(IA,1)=B1(1)*F(1,IA,ISP)+B1(2)*F(2,IA,ISP)+ &
     &                                                 B1(3)*F(3,IA,ISP)
            W(IA,2)=B2(1)*F(1,IA,ISP)+B2(2)*F(2,IA,ISP)+ &
     &                                                 B2(3)*F(3,IA,ISP)
            W(IA,3)=B3(1)*F(1,IA,ISP)+B3(2)*F(2,IA,ISP)+ &
     &                                                 B3(3)*F(3,IA,ISP)
    1    CONTINUE
! Reset array F and divide the input vectors by NR*NP:
         DO 2 IA=ISTART,NA(IS)+ISTART-1
            W(IA,1)=W(IA,1)/FLOAT(NR*NP)
            W(IA,2)=W(IA,2)/FLOAT(NR*NP)
            W(IA,3)=W(IA,3)/FLOAT(NR*NP)
            F(1,IA,ISP)=0._q
            F(2,IA,ISP)=0._q
            F(3,IA,ISP)=0._q
    2    CONTINUE
! Apply all space group operations on the vectors:
         DO 6 IROT=1,NR
! Calculate the rotated vectors for all atoms:
            DO 3 IA=ISTART,NA(IS)+ISTART-1
               FR(IA,1)=S(1,1,IROT)*W(IA,1)+ &
     &                                S(2,1,IROT)*W(IA,2)+ &
     &                                               S(3,1,IROT)*W(IA,3)
               FR(IA,2)=S(1,2,IROT)*W(IA,1)+ &
     &                                S(2,2,IROT)*W(IA,2)+ &
     &                                               S(3,2,IROT)*W(IA,3)
               FR(IA,3)=S(1,3,IROT)*W(IA,1)+ &
     &                                S(2,3,IROT)*W(IA,2)+ &
     &                                               S(3,3,IROT)*W(IA,3)
    3       CONTINUE
! Apply all translations to all atoms:
            DO 5 ITRANS=1,NP
! Now add all rotated vectors to the corresponding atomic vector:
               DO 4 IA=ISTART,NA(IS)+ISTART-1
                  F(1,IA,ISP)=F(1,IA,ISP)+ &
     &                         FR(MAP(IA,ISP,IROT,ITRANS),1)*A1(1)+ &
     &                            FR(MAP(IA,ISP,IROT,ITRANS),2)*A2(1)+ &
     &                               FR(MAP(IA,ISP,IROT,ITRANS),3)*A3(1)
                  F(2,IA,ISP)=F(2,IA,ISP)+ &
     &                         FR(MAP(IA,ISP,IROT,ITRANS),1)*A1(2)+ &
     &                            FR(MAP(IA,ISP,IROT,ITRANS),2)*A2(2)+ &
     &                               FR(MAP(IA,ISP,IROT,ITRANS),3)*A3(2)
                  F(3,IA,ISP)=F(3,IA,ISP)+ &
     &                         FR(MAP(IA,ISP,IROT,ITRANS),1)*A1(3)+ &
     &                            FR(MAP(IA,ISP,IROT,ITRANS),2)*A2(3)+ &
     &                               FR(MAP(IA,ISP,IROT,ITRANS),3)*A3(3)
    4          CONTINUE
    5       CONTINUE
    6    CONTINUE
    7 CONTINUE
! Ready ... !
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE TSYM(T,S,NROT,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine TSYM symmetrizes a 3x3 tensor of second order given in     *
!   cartesian coordinates according to a given symmetry operations     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      T(3,3) contains on input the tensor to be symmetrized           *
!             (given in crystal coordinates)                           *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      A(3,3) contains the lattice vectors defining the unit cell      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      T(3,3) contains on output the symmetrized tensor (input will    *
!             be overwritten!)                                         *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S

      DIMENSION T(3,3),S(3,3,48),R(3,3,48),G(3,3,48),TIN(3,3)
      DIMENSION TROT(3,3),A(3,3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      TIN=T
      T=0._q
      R=S

      CALL SGRTRF(R,G,NROT,A(1,1),A(1,2),A(1,3),C1,C2,C3)
      DO 5 IROT=1,NROT
         DO 3 I=1,3
            DO 2 J=1,3
               TROT(I,J)=G(1,I,IROT)*G(1,J,IROT)*TIN(1,1)+ &
     &                     G(1,I,IROT)*G(2,J,IROT)*TIN(1,2)+ &
     &                       G(1,I,IROT)*G(3,J,IROT)*TIN(1,3)+ &
     &                         G(2,I,IROT)*G(1,J,IROT)*TIN(2,1)+ &
     &                           G(2,I,IROT)*G(2,J,IROT)*TIN(2,2)+ &
     &                             G(2,I,IROT)*G(3,J,IROT)*TIN(2,3)+ &
     &                               G(3,I,IROT)*G(1,J,IROT)*TIN(3,1)+ &
     &                                 G(3,I,IROT)*G(2,J,IROT)*TIN(3,2)+ &
     &                                  G(3,I,IROT)*G(3,J,IROT)*TIN(3,3)
    2       CONTINUE
    3    CONTINUE
! Sum up the rotated tensors (correctly normed with 1/NROT):
         T=T+TROT/FLOAT(NROT)
    5 CONTINUE
      RETURN
      END



!***********************************************************************
!                                                                      *
      SUBROUTINE TSYM_CMPLX(T,S,NROT,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine TSYM symmetrizes a 3x3 tensor of second order given in     *
!   cartesian coordinates according to a given symmetry operations     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      T(3,3) contains on input the tensor to be symmetrized           *
!             (given in crystal coordinates)                           *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      A(3,3) contains the lattice vectors defining the unit cell      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      T(3,3) contains on output the symmetrized tensor (input will    *
!             be overwritten!)                                         *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S
      COMPLEX(q) T(3,3), TROT(3,3), TIN(3,3)
      DIMENSION S(3,3,48),R(3,3,48),G(3,3,48)
      DIMENSION A(3,3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      TIN=T
      T=0._q
      R=S

      CALL SGRTRF(R,G,NROT,A(1,1),A(1,2),A(1,3),C1,C2,C3)
      DO 5 IROT=1,NROT
         DO 3 I=1,3
            DO 2 J=1,3
               TROT(I,J)=G(1,I,IROT)*G(1,J,IROT)*TIN(1,1)+ &
     &                     G(1,I,IROT)*G(2,J,IROT)*TIN(1,2)+ &
     &                       G(1,I,IROT)*G(3,J,IROT)*TIN(1,3)+ &
     &                         G(2,I,IROT)*G(1,J,IROT)*TIN(2,1)+ &
     &                           G(2,I,IROT)*G(2,J,IROT)*TIN(2,2)+ &
     &                             G(2,I,IROT)*G(3,J,IROT)*TIN(2,3)+ &
     &                               G(3,I,IROT)*G(1,J,IROT)*TIN(3,1)+ &
     &                                 G(3,I,IROT)*G(2,J,IROT)*TIN(3,2)+ &
     &                                  G(3,I,IROT)*G(3,J,IROT)*TIN(3,3)
    2       CONTINUE
    3    CONTINUE
! Sum up the rotated tensors (correctly normed with 1/NROT):
         T=T+TROT/FLOAT(NROT)
    5 CONTINUE
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE TSYM2(T,MAP,S,NR,NP,NSX,NSP,NAX,NA,TR,W, &
     &                                                A1,A2,A3,B1,B2,B3)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine TSYM2 symmetrizes an array of tensors (e.g. Born eff. ch.) *
!   located at different atomic positions. The procedure is a quite    *
!   simple: Rotate the input tensor and add it to the tensor at the    *
!   rotated atomic position (for all space group operations ...).      *
!   Finally divide the result by the number of symmetry operations!    *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      T(3,3,NAX,NSX) contains the tensors in cartesian coordinates    *
!              located at the atomic positions.                        *
!                                                                      *
!      MAP(NAX,NSX,NROTK,NP) contains a table connecting the rotated   *
!                   and the unrotated positions (stored is the index of*
!                   the rotated position = first index in array TAU).  *
!                   MAP can be generated by routine CLASS (see below). *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NR contains the number of given space group operations.         *
!      NP contains the number of "primitive" translations in the cell. *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      TR and W are work arrays.                                       *
!                                                                      *
!      A1, A2 and A3 contain the direct (real space) lattice vectors.  *
!      B1, B2 and B3 contain the reciprocal lattice vectors.           *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      Array T contains the symmetrized tensors on output (input is    *
!      overwritten!).                                                  *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT
      INTEGER S

      DIMENSION T(3,3,NAX,NSX),MAP(NAX,NSX,NR,NP),S(3,3,48),R(3,3,48),G(3,3,48),NA(NSP)
      DIMENSION TR(NAX,3,3),W(NAX,3,3),A1(3),A2(3),A3(3),B1(3),B2(3),B3(3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      R=S
      CALL SGRTRF(R,G,NR,A1,A2,A3,C1,C2,C3)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISTART=1
! For all atomic species ... :
      DO 7 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Reset array F and divide the input vectors by NR*NP:
         DO 8 IA=ISTART,NA(IS)+ISTART-1
            W(IA,:,:)=T(:,:,IA,ISP)/FLOAT(NR*NP)
            T(:,:,IA,ISP)=0._q
    8    CONTINUE
! Apply all space group operations on the vectors:
         DO 6 IROT=1,NR
! Calculate the rotated vectors for all atoms:
            DO 3 IA=ISTART,NA(IS)+ISTART-1
               DO 2 I=1,3
                  DO 1 J=1,3
                     TR(IA,I,J)=G(1,I,IROT)*G(1,J,IROT)*W(IA,1,1)+ &
     &                            G(1,I,IROT)*G(2,J,IROT)*W(IA,1,2)+ &
     &                              G(1,I,IROT)*G(3,J,IROT)*W(IA,1,3)+ &
     &                                G(2,I,IROT)*G(1,J,IROT)*W(IA,2,1)+ &
     &                                  G(2,I,IROT)*G(2,J,IROT)*W(IA,2,2)+ &
     &                                    G(2,I,IROT)*G(3,J,IROT)*W(IA,2,3)+ &
     &                                      G(3,I,IROT)*G(1,J,IROT)*W(IA,3,1)+ &
     &                                        G(3,I,IROT)*G(2,J,IROT)*W(IA,3,2)+ &
     &                                          G(3,I,IROT)*G(3,J,IROT)*W(IA,3,3)
    1             CONTINUE
    2          CONTINUE
    3       CONTINUE
! Apply all translations to all atoms:
            DO 5 ITRANS=1,NP
! Now add all rotated vectors to the corresponding atomic vector:
               DO 4 IA=ISTART,NA(IS)+ISTART-1
                  T(:,:,IA,ISP)=T(:,:,IA,ISP)+TR(MAP(IA,ISP,IROT,ITRANS),:,:)
    4          CONTINUE
    5       CONTINUE
    6    CONTINUE
    7 CONTINUE
! Ready ... !
      RETURN
      END


      SUBROUTINE TSYM3(T,S,NROT,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine TSYM4 symmetrizes a tensor of third order given in         *
!   cartesian coordinates according to a given symmetry operations     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      T(3,3,3)   contains on input the tensor to be symmetrized       *
!             (given in crystal coordinates)                           *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      A(3,3) contains the lattice vectors defining the unit cell      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      T(3,3,3)   contains on output the symmetrized tensor (input will*
!             be overwritten!)                                         *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S

      DIMENSION T(3,3,3),TIN(3,3,3),TROT(3,3,3),TROT2(3,3,3)
      DIMENSION S(3,3,48),R(3,3,48),G(3,3,48)
      DIMENSION A(3,3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      TIN=T
      T=0._q
      R=S

      CALL SGRTRF(R,G,NROT,A(1,1),A(1,2),A(1,3),C1,C2,C3)
      DO IROT=1,NROT
         ! first index
         DO I=1,3
            TROT(I,:,:)= & 
                 G(1,I,IROT)*TIN(1,:,:)+ & 
                 G(2,I,IROT)*TIN(2,:,:)+ & 
                 G(3,I,IROT)*TIN(3,:,:)
         ENDDO
         ! second index
         DO I=1,3
            TROT2(:,I,:)= & 
                 G(1,I,IROT)*TROT(:,1,:)+ & 
                 G(2,I,IROT)*TROT(:,2,:)+ & 
                 G(3,I,IROT)*TROT(:,3,:)
         ENDDO
         ! third index
         DO I=1,3
            TROT(:,:,I)= & 
                 G(1,I,IROT)*TROT2(:,:,1)+ & 
                 G(2,I,IROT)*TROT2(:,:,2)+ & 
                 G(3,I,IROT)*TROT2(:,:,3)
         ENDDO
! Sum up the rotated tensors (correctly normed with 1/NROT):
         T=T+TROT/FLOAT(NROT)
      ENDDO
      END

      SUBROUTINE TSYM4(T,S,NROT,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine TSYM4 symmetrizes a tensor of fourth order given in        *
!   cartesian coordinates according to a given symmetry operations     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      T(3,3,3,3) contains on input the tensor to be symmetrized       *
!             (given in crystal coordinates)                           *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      A(3,3) contains the lattice vectors defining the unit cell      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      T(3,3,3,3) contains on output the symmetrized tensor (input will*
!             be overwritten!)                                         *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S

      DIMENSION T(3,3,3,3),TIN(3,3,3,3),TROT(3,3,3,3),TROT2(3,3,3,3)
      DIMENSION S(3,3,48),R(3,3,48),G(3,3,48)
      DIMENSION A(3,3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      TIN=T
      T=0._q
      R=S

      CALL SGRTRF(R,G,NROT,A(1,1),A(1,2),A(1,3),C1,C2,C3)
      DO IROT=1,NROT
         ! first index
         DO I=1,3
            TROT(I,:,:,:)= & 
                 G(1,I,IROT)*TIN(1,:,:,:)+ & 
                 G(2,I,IROT)*TIN(2,:,:,:)+ & 
                 G(3,I,IROT)*TIN(3,:,:,:)
         ENDDO
         ! second index
         DO I=1,3
            TROT2(:,I,:,:)= & 
                 G(1,I,IROT)*TROT(:,1,:,:)+ & 
                 G(2,I,IROT)*TROT(:,2,:,:)+ & 
                 G(3,I,IROT)*TROT(:,3,:,:)
         ENDDO
         ! third index
         DO I=1,3
            TROT(:,:,I,:)= & 
                 G(1,I,IROT)*TROT2(:,:,1,:)+ & 
                 G(2,I,IROT)*TROT2(:,:,2,:)+ & 
                 G(3,I,IROT)*TROT2(:,:,3,:)
         ENDDO
         ! fourth index
         DO I=1,3
            TROT2(:,:,:,I)= & 
                 G(1,I,IROT)*TROT(:,:,:,1)+ & 
                 G(2,I,IROT)*TROT(:,:,:,2)+ & 
                 G(3,I,IROT)*TROT(:,:,:,3)
         ENDDO
! Sum up the rotated tensors (correctly normed with 1/NROT):
         T=T+TROT2/FLOAT(NROT)
      ENDDO
      END

      SUBROUTINE F1SYM(F,S,NROT,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine F1SYM symmetrizes a vector given in                        *
!   cartesian coordinates according to a given symmetry operations     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      F(3)   contains on input the vector to be symmetrized           *
!             (given in crystal coordinates)                           *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      A(3,3) contains the lattice vectors defining the unit cell      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      F(3)   contains on output the symmetrized vector (input will    *
!             be overwritten!)                                         *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S

      DIMENSION F(3),S(3,3,48),R(3,3,48),G(3,3,48),FIN(3)
      DIMENSION FROT(3),A(3,3),C1(3),C2(3),C3(3)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      FIN=F
      F=0._q
      R=S

      CALL SGRTRF(R,G,NROT,A(1,1),A(1,2),A(1,3),C1,C2,C3)
      DO 5 IROT=1,NROT
         DO 3 I=1,3
               FROT(I)=G(1,I,IROT)*FIN(1)+ &
     &                 G(2,I,IROT)*FIN(2)+ &
     &                 G(3,I,IROT)*FIN(3)
    3    CONTINUE
! Sum up the rotated tensors (correctly normed with 1/NROT):
         F=F+FROT/FLOAT(NROT)
    5 CONTINUE
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE POSMAP(TAU,S,T,NROT,PT,NP,NSAX,NSX,NSP,NAX,NA,MAP,TR)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine POSMAP gives a table that connects the rotated positions   *
!   with the original positions for each symmetry operation.           *
!   Note: The same action is also done by routine CLASS, but sometimes *
!         one might not need all the tables about atomic classes ... . *
!         In this case use routine POSMAP giving only a rotation map.  *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell.                         *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      T(3,48) contains the nonprimitive translations associated with  *
!              the rotations given in S.                               *
!      NROT contains the number of given space group operations.       *
!      PT(NSAX+2,3) contains all "primitive" translations according    *
!                   to the generating cell of a nonprimitive cell.     *
!      NP contains the number of "primitive" translations.             *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSAX, NSX and NAX are array dimensioning parameters.            *
!                                                                      *
!      TR is a work array.                                             *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      MAP(NAX,NSX,NROTK,NP) contains a table connecting the rotated and  *
!                   the unrotated positions (stored is the index of    *
!                   the rotated position = first index in array TAU).  *
!                   More clearly: MAP(IA,..,,IROT,ITRANS) tells us     *
!                   which atom JA will be rotated to atom IA by the    *
!                   transformation (IROT,ITRANS), or to which atom     *
!                   JA the current atom IA would be transformed by     *
!                   the inverse of transformation (IROT,ITRANS).       *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT
      INTEGER S

      DIMENSION TAU(NAX,3,NSX),S(3,3,48),T(3,48),NA(NSP)
      DIMENSION PT(NSAX+2,3),TR(NAX,3),MAP(NAX,NSX,NROT,NP)

! To be able to check whether all elements of MAP have been found,
! initialize them to -1 here.
      MAP=-1

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISTART=1
! For all atomic species ... :
      DO 7 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
         DO 1 IA=ISTART,NA(IS)+ISTART-1
! First impose periodic boundary conditions on TAU (--> [-0.5,0.5) ):
            TAU(IA,1,ISP)=TAU(IA,1,ISP)-INT(TAU(IA,1,ISP))
            TAU(IA,2,ISP)=TAU(IA,2,ISP)-INT(TAU(IA,2,ISP))
            TAU(IA,3,ISP)=TAU(IA,3,ISP)-INT(TAU(IA,3,ISP))
            TAU(IA,1,ISP)=MOD(TAU(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,2,ISP)=MOD(TAU(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,3,ISP)=MOD(TAU(IA,3,ISP)+100.5_q,1._q)-0.5_q
#ifndef periodic_image_convention
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
#endif
    1    CONTINUE
! Apply all space group operations on TAU ...
         DO 6 IROT=1,NROT
! ... and all "primitive" translations ... :
            DO 5 ITRANS=1,NP
! Calculate the rotated and translated coordinates for all atoms:
               DO 2 IA=ISTART,NA(IS)+ISTART-1
                  TR(IA,1)=S(1,1,IROT)*TAU(IA,1,ISP)+ &
     &                           S(2,1,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,1,IROT)*TAU(IA,3,ISP)+ &
     &                                            PT(ITRANS,1)+T(1,IROT)
                  TR(IA,2)=S(1,2,IROT)*TAU(IA,1,ISP)+ &
     &                           S(2,2,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,2,IROT)*TAU(IA,3,ISP)+ &
     &                                            PT(ITRANS,2)+T(2,IROT)
                  TR(IA,3)=S(1,3,IROT)*TAU(IA,1,ISP)+ &
     &                           S(2,3,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,3,IROT)*TAU(IA,3,ISP)+ &
     &                                            PT(ITRANS,3)+T(3,IROT)
    2          CONTINUE
               DO 3 IA=ISTART,NA(IS)+ISTART-1
! Impose periodic boundary conditions on the rotated coordinates:
                  TR(IA,1)=MOD(TR(IA,1)+100.5_q,1._q)-0.5_q
                  TR(IA,2)=MOD(TR(IA,2)+100.5_q,1._q)-0.5_q
                  TR(IA,3)=MOD(TR(IA,3)+100.5_q,1._q)-0.5_q
#ifndef periodic_image_convention
                  IF ((ABS(TR(IA,1)+0.5_q))<TINY) TR(IA,1)=-0.5_q
                  IF ((ABS(TR(IA,2)+0.5_q))<TINY) TR(IA,2)=-0.5_q
                  IF ((ABS(TR(IA,3)+0.5_q))<TINY) TR(IA,3)=-0.5_q
                  IF ((ABS(TR(IA,1)-0.5_q))<TINY) TR(IA,1)=-0.5_q
                  IF ((ABS(TR(IA,2)-0.5_q))<TINY) TR(IA,2)=-0.5_q
                  IF ((ABS(TR(IA,3)-0.5_q))<TINY) TR(IA,3)=-0.5_q
#endif
    3          CONTINUE
! Now compare all rotated and all original coordinates and generate a
! map for all rotations which connects rotated and original coordinates:
!ocl nopreex
               DO IA=ISTART,NA(IS)+ISTART-1
                  DO JA=ISTART,NA(IS)+ISTART-1
                    IF ((ABS(MIN_IMAG(TR(IA,1)-TAU(JA,1,ISP)))<TINY).AND. &
     &                    (ABS(MIN_IMAG(TR(IA,2)-TAU(JA,2,ISP)))<TINY).AND. &
     &                         (ABS(MIN_IMAG(TR(IA,3)-TAU(JA,3,ISP)))<TINY)) &
     &                                        MAP(JA,ISP,IROT,ITRANS)=IA
                  ENDDO
               ENDDO
    5       CONTINUE
    6    CONTINUE
    7 CONTINUE
! Now check MAP
      ISTART=1
! For all atomic species ... :
      DO IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
         DO IA=ISTART,NA(IS)+ISTART-1
            DO IROT=1,NROT
               DO ITRANS=1,NP
                  IF (MAP(IA,ISP,IROT,ITRANS)==-1) THEN
                      WRITE(*,'(/A/A)') ' POSMAP internal error: symmetry equivalent atom not found,', &
                     &                  '   you might try decreasing or increasing SYMPREC by an order of magnitude.'
                      STOP
                  ENDIF
               ENDDO
            ENDDO
         ENDDO
      ENDDO
! Ready ... !
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE CLASS(TAU,S,T,NROT,NSX,NSP,NAX,NA,NCL,ICL,MCL,MAP,TR,W)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine CLASS searches all classes of symmetry related atomic      *
!   positions and gives a table that connects the rotated positions    *
!   with the original positions for each symmetry operation.           *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell.                         *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      T(3,48) contains the nonprimitive translations associated with  *
!              the rotations given in S.                               *
!      NROT contains the number of given space group operations.       *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      TR and W are work arrays.                                       *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      NCL(NSX) contains the number of atomic classes (per species).   *
!                                                                      *
!      ICL(NAX,NSX) counts the number of atoms within one class        *
!                   (the first index running over all classes ...).    *
!                                                                      *
!      MCL(NAX,NAX,NSX) lists the members of every class (first index  *
!                   running over all members, second index running     *
!                   over all classes ...).                             *
!                                                                      *
!      MAP(NAX,NSX,48) contains a table connecting the rotated and     *
!                   the unrotated positions (stored is the index of    *
!                   the rotated position = first index in array TAU).  *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT
      INTEGER S

      DIMENSION TAU(NAX,3,NSX),S(3,3,48),T(3,48),NA(NSP),NCL(NSX)
      DIMENSION ICL(NAX,NSX),MCL(NAX,NAX,NSX),MAP(NAX,NSX,48)
      DIMENSION TR(NAX,3),W(NAX)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISTART=1
! For all atomic species ... :
      DO 8 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
         NCL(IS)=0
         DO 1 IA=ISTART,NA(IS)+ISTART-1
! First impose periodic boundary conditions on TAU (--> [-0.5,0.5) ):
            TAU(IA,1,ISP)=TAU(IA,1,ISP)-INT(TAU(IA,1,ISP))
            TAU(IA,2,ISP)=TAU(IA,2,ISP)-INT(TAU(IA,2,ISP))
            TAU(IA,3,ISP)=TAU(IA,3,ISP)-INT(TAU(IA,3,ISP))
            TAU(IA,1,ISP)=MOD(TAU(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,2,ISP)=MOD(TAU(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,3,ISP)=MOD(TAU(IA,3,ISP)+100.5_q,1._q)-0.5_q
#ifndef periodic_image_convention
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
#endif
            W(IA)=-1._q
    1    CONTINUE
! Apply all space group operations on TAU ... :
         DO 5 IROT=1,NROT
! Calculate the rotated coordinates for all atoms:
            DO 2 IA=ISTART,NA(IS)+ISTART-1
               TR(IA,1)=S(1,1,IROT)*TAU(IA,1,ISP)+ &
     &                          S(2,1,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,1,IROT)*TAU(IA,3,ISP)+T(1,IROT)
               TR(IA,2)=S(1,2,IROT)*TAU(IA,1,ISP)+ &
     &                          S(2,2,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,2,IROT)*TAU(IA,3,ISP)+T(2,IROT)
               TR(IA,3)=S(1,3,IROT)*TAU(IA,1,ISP)+ &
     &                          S(2,3,IROT)*TAU(IA,2,ISP)+ &
     &                               S(3,3,IROT)*TAU(IA,3,ISP)+T(3,IROT)
    2       CONTINUE
            DO 3 IA=ISTART,NA(IS)+ISTART-1
! Impose periodic boundary conditions on the rotated coordinates:
               TR(IA,1)=MOD(TR(IA,1)+100.5_q,1._q)-0.5_q
               TR(IA,2)=MOD(TR(IA,2)+100.5_q,1._q)-0.5_q
               TR(IA,3)=MOD(TR(IA,3)+100.5_q,1._q)-0.5_q
#ifndef periodic_image_convention
               IF ((ABS(TR(IA,1)+0.5_q))<TINY) TR(IA,1)=-0.5_q
               IF ((ABS(TR(IA,2)+0.5_q))<TINY) TR(IA,2)=-0.5_q
               IF ((ABS(TR(IA,3)+0.5_q))<TINY) TR(IA,3)=-0.5_q
               IF ((ABS(TR(IA,1)-0.5_q))<TINY) TR(IA,1)=-0.5_q
               IF ((ABS(TR(IA,2)-0.5_q))<TINY) TR(IA,2)=-0.5_q
               IF ((ABS(TR(IA,3)-0.5_q))<TINY) TR(IA,3)=-0.5_q
#endif
    3       CONTINUE
! Now compare all rotated and all original coordinates and generate a
! map for all rotations which connects rotated and original coordinates:
!ocl nopreex
            DO IA=ISTART,NA(IS)+ISTART-1
               DO JA=ISTART,NA(IS)+ISTART-1
                  IF ((ABS(MIN_IMAG(TR(IA,1)-TAU(JA,1,ISP)))<TINY).AND. &
     &                   (ABS(MIN_IMAG(TR(IA,2)-TAU(JA,2,ISP)))<TINY).AND. &
     &                          (ABS(MIN_IMAG(TR(IA,3)-TAU(JA,3,ISP)))<TINY)) &
     &                                               MAP(JA,ISP,IROT)=IA
               ENDDO
            ENDDO
    5    CONTINUE
! Search the classes ... :
         DO 7 IA=ISTART,NA(IS)+ISTART-1
! If atom was not yet used ... (W initially preset with -1):
            IF (W(IA)<0) THEN
! Next class:
               NCL(IS)=NCL(IS)+1
! Counter for number of members:
               ICL(NCL(IS)+ISTART-1,ISP)=0
               DO 6 IROT=1,NROT
! Rotated atom given by MAP ...
                  IF (W(MAP(IA,ISP,IROT))<0._q) THEN
! ... if not used: we found a new member for this class!
                     ICL(NCL(IS)+ISTART-1,ISP)= &
     &                                       ICL(NCL(IS)+ISTART-1,ISP)+1
! 'Index' of the member stored into MCL:
                     MCL(ICL(NCL(IS)+ISTART-1,ISP),NCL(IS)+ISTART-1,ISP) &
     &                                                 =MAP(IA,ISP,IROT)
                  END IF
! Indicate usage of the rotated coordinate:
                  W(MAP(IA,ISP,IROT))=1._q
    6          CONTINUE
            END IF
    7    CONTINUE
    8 CONTINUE
! Ready ... !
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE PGROUP(S,NROT,PGIND,SFNAME)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Given some set of point symmetry operations routine PGROUP returns *
!   the name of the point group defined by these symmetry operations.  *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      NROT contains the number of symmetry operations.                *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      PGIND gives the "name" of the group defined by following key:   *
!       1 --> C_1       9 --> C_3      17 --> D_4      25 --> C_6v     *
!       2 --> S_2      10 --> S_6      18 --> C_4v     26 --> D_3h     *
!       3 --> C_2      11 --> D_3      19 --> D_2d     27 --> D_6h     *
!       4 --> C_1h     12 --> C_3v     20 --> D_4h     28 --> T        *
!       5 --> C_2h     13 --> D_3d     21 --> C_6      29 --> T_h      *
!       6 --> D_2      14 --> C_4      22 --> C_3h     30 --> O        *
!       7 --> C_2v     15 --> S_4      23 --> C_6h     31 --> T_d      *
!       8 --> D_2h     16 --> C_4h     24 --> D_6      32 --> O_h      *
!      SFNAME is the explicit name in form of a string (Schoenflies).  *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S(3,3,48),PGIND,TRACE,DET
      CHARACTER (LEN=*) SFNAME
      CHARACTER (4) GNAME(32)

      SAVE GNAME
      DATA GNAME /'C_1 ','S_2 ','C_2 ','C_1h','C_2h','D_2 ','C_2v', &
     &              'D_2h','C_3 ','S_6 ','D_3 ','C_3v','D_3d','C_4 ', &
     &               'S_4 ','C_4h','D_4 ','C_4v','D_2d','D_4h','C_6 ', &
     &                'C_3h','C_6h','D_6 ','C_6v','D_3h','D_6h','T   ', &
     &                                      'T_h ','O   ','T_d ','O_h '/

! Need at least four characters to store the group name ... :
      IF (LEN(SFNAME)<4) CALL ERROR(' PGROUP', &
     &  ' Variable SFNAME declared too short in the calling program!', &
     &                                                      LEN(SFNAME))
! Trivial case: Only one symmetry operation (can only be E --> PGIND=1):
      IF (NROT==1) THEN
         PGIND=1
         SFNAME=GNAME(PGIND)
         RETURN
      END IF
! There is an other trivial case: group O_h (PGIND=32), because it is
! the only group having 48 elements ... !
      IF (NROT==48) THEN
         PGIND=32
         SFNAME=GNAME(PGIND)
         RETURN
      END IF
! An other trivial case is group D_4h (PGIND=20), because it is the only
! group having 16 elements ... !
      IF (NROT==16) THEN
         PGIND=20
         SFNAME=GNAME(PGIND)
         RETURN
      END IF
! And finally there is a fourth trivial case: it is group C_3 (PGIND=9),
! because it is the only group having 3 elements ... !
      IF (NROT==3) THEN
         PGIND=9
         SFNAME=GNAME(PGIND)
         RETURN
      END IF
! All other groups need further investigations and detailed analysis ...
! First determine the type of elements and count them. Possible elements
! are E, I, C_2,3,4,6 and S_2,3,4,6 (S_2 = m), E is trivial and always
! present. The type of a symmetry operation can be identified simply by
! calculating the trace and the determinant of the rotation matrix. The
! combination of these two quantities is specific for specific elements:
!
! Element:         E    I  C_2  C_3  C_4  C_6  S_2  S_6  S_4  S_3
! Trace:          +3   -3   -1    0   +1   +2   +1    0   -1   -2
! Determinant:    +1   -1   +1   +1   +1   +1   -1   -1   -1   -1

      INVERS=0
      NC2=0
      NC3=0
      NC4=0
      NC6=0
      NS2=0
      NS6=0
      NS4=0
      NS3=0
      DO 1 IR=1,NROT
         TRACE=S(1,1,IR)+S(2,2,IR)+S(3,3,IR)
! Found unity operator (trivial):
         IF (TRACE==3) GOTO 1
! Found inversion ...
         IF (TRACE==(-3)) THEN
            INVERS=1
            GOTO 1
         END IF
         DET=S(1,1,IR)*(S(2,2,IR)*S(3,3,IR)-S(2,3,IR)*S(3,2,IR))+ &
     &           S(1,2,IR)*(S(2,3,IR)*S(3,1,IR)-S(2,1,IR)*S(3,3,IR))+ &
     &               S(1,3,IR)*(S(2,1,IR)*S(3,2,IR)-S(2,2,IR)*S(3,1,IR))
! Found C_2:
         IF ((TRACE==(-1)).AND.(DET==1)) NC2=NC2+1
! Found S_2:
         IF ((TRACE==1).AND.(DET==(-1))) NS2=NS2+1
! Found C_3:
         IF ((TRACE==0).AND.(DET==1)) NC3=NC3+1
! Found S_6:
         IF ((TRACE==0).AND.(DET==(-1))) NS6=NS6+1
! Found C_4:
         IF ((TRACE==1).AND.(DET==1)) NC4=NC4+1
! Found S_4:
         IF ((TRACE==(-1)).AND.(DET==(-1))) NS4=NS4+1
! Found C_6:
         IF ((TRACE==2).AND.(DET==1)) NC6=NC6+1
! Found S_3:
         IF ((TRACE==(-2)).AND.(DET==(-1))) NS3=NS3+1
    1 CONTINUE
! Now we know which elements we have and so we know the group ... :
      IF (NROT==2) THEN
! Groups with 2 elements:
         IF (INVERS==1) THEN
! Contains inversion --> S_2 (PGIND=2):
            PGIND=2
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC2==1) THEN
! Contains twofold rotation --> C_2 (PGIND=3):
            PGIND=3
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==1) THEN
! Contains mirror plane --> C_1h (PGIND=4):
            PGIND=4
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
      IF (NROT==4) THEN
! Groups with 4 elements:
         IF (INVERS==1) THEN
! Contains inversion --> C_2h (PGIND=5):
            PGIND=5
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC2==3) THEN
! Contains three twofold rotations --> D_2 (PGIND=6):
            PGIND=6
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==2) THEN
! Contains two mirror planes --> C_2v (PGIND=7):
            PGIND=7
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC4==1) THEN
! Contains fourfold rotation --> C_4 (PGIND=14):
            PGIND=14
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS4==2) THEN
! Contains fourfold improper rotation --> S_4 (PGIND=15):
            PGIND=15
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
      IF (NROT==6) THEN
! Groups with 6 elements:
         IF (INVERS==1) THEN
! Contains inversion --> S_6 (PGIND=10):
            PGIND=10
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC2==3) THEN
! Contains three twofold rotations --> D_3 (PGIND=11):
            PGIND=11
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==3) THEN
! Contains three mirror planes --> C_3v (PGIND=12):
            PGIND=12
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC2==1) THEN
! Contains only one twofold rotations --> C_6 (PGIND=21):
            PGIND=21
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==1) THEN
! Contains only one mirror plane --> C_3h (PGIND=22):
            PGIND=22
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
      IF (NROT==8) THEN
! Groups with 8 elements:
         IF (NS2==3) THEN
! Contains three mirror planes --> D_2h (PGIND=8):
            PGIND=8
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==1) THEN
! Contains one mirror planes --> C_4h (PGIND=16):
            PGIND=16
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==0) THEN
! Contains no mirror planes --> D_4 (PGIND=17):
            PGIND=17
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==4) THEN
! Contains four mirror planes --> C_4v (PGIND=18):
            PGIND=18
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==2) THEN
! Contains two mirror planes --> D_2d (PGIND=19):
            PGIND=19
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
      IF (NROT==12) THEN
! Groups with 12 elements:
         IF (NS2==3) THEN
! Contains three mirror planes --> D_3d (PGIND=13):
            PGIND=13
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==1) THEN
! Contains one mirror planes --> C_6h (PGIND=23):
            PGIND=23
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC2==7) THEN
! Contains seven twofold rotations --> D_6 (PGIND=24):
            PGIND=24
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==6) THEN
! Contains six mirror planes --> C_6v (PGIND=25):
            PGIND=25
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS2==4) THEN
! Contains four mirror planes --> D_3h (PGIND=26):
            PGIND=26
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC3==8) THEN
! Contains eight threefold rotations --> T (PGIND=28):
            PGIND=28
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
      IF (NROT==24) THEN
! Groups with 24 elements:
         IF (NC6==2) THEN
! Contains two sixfold rotations --> D_6h (PGIND=27):
            PGIND=27
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (INVERS==1) THEN
! Contains inversion --> T_h (PGIND=29):
            PGIND=29
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NC4==6) THEN
! Contains six fourfold rotations --> O (PGIND=30):
            PGIND=30
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
         IF (NS4==6) THEN
! Contains six fourfold improper rotations --> T_d (PGIND=31):
            PGIND=31
            SFNAME=GNAME(PGIND)
            RETURN
         END IF
      END IF
! Ready!
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SGRGEN(SGEN,NGEN,S,NTOT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SGRGEN generates all point group symmetry operations from  *
!   the generation group. This routine is set up for INTEGER matrices. *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      SGEN(9,NGEN) contains the generator matrices.                   *
!      NGEN contains the number of generators.                         *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      SGEN(9,48) contains the generated symmetry elements.            *
!      NTOT contains the number of generated symmetry operations.      *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER SGEN(9,NGEN),S(9,48),H(9),HH(9),E(9),SIG(9)
      LOGICAL SGREQL

      SAVE E
      DATA E /1,0,0,0,1,0,0,0,1/

! There is always one trivial element: the idendity ...
      CALL SGRCOP(E,S(1,1))
      NTOT=1
! Take all generators ... :
      DO 80 IGEN=1,NGEN
         CALL SGRCOP(SGEN(1,IGEN),SIG)
! Extend the group by all products with SIG(i)=SGEN(i,IGEN):
         DO 9 IG=1,NTOT
            IF (SGREQL(S(1,IG),SIG)) GOTO 80
    9    CONTINUE
! Determine the order of the operation:
         CALL SGRCOP(SIG,H)
         DO 1 ITRY=1,100
            IORD=ITRY
            IF (SGREQL(H,E)) GOTO 2
            CALL SGRPRD(SIG,H,H)
    1    CONTINUE
! Products of type 'G1*(SIG**P)*G2':
    2    NNOW=NTOT
         DO 8 J=1,NTOT
            CALL SGRCOP(S(1,J),H)
            DO 10 IP=1,IORD-1
! H=SIG**IP
               CALL SGRPRD(SIG,H,H)
               DO 11 I=1,NTOT
! HH=SYMOPS_I*(SIG**IP)
                  CALL SGRPRD(S(1,I),H,HH)
                  DO 12 K=1,NNOW
                     IF (SGREQL(S(1,K),HH)) GOTO 11
   12             CONTINUE
                  NNOW=NNOW+1
                  IF (NNOW>48) GOTO 99
                  CALL SGRCOP(HH,S(1,NNOW))
   11          CONTINUE
   10       CONTINUE
            IF (J==1) N2=NNOW
    8    CONTINUE
! Products with more than one sandwiched SIGMA-factor:
         M1=NTOT+1
         M2=NNOW
         DO 20 I=2,50
            DO N=NTOT+1,N2
             DO M=M1,M2
               CALL SGRPRD(S(1,N),S(1,M),H)
               DO 22 K=1,NNOW
                  IF (SGREQL(S(1,K),H)) GOTO 21
   22          CONTINUE
               NNOW=NNOW+1
               IF (NNOW>48) GOTO 99
               CALL SGRCOP(H,S(1,NNOW))
   21        CONTINUE
             ENDDO
            ENDDO
            IF (M2==NNOW) GOTO 25
            M1=M2+1
            M2=NNOW
   20    CONTINUE
   25    CONTINUE
         NTOT=NNOW
   80 CONTINUE
      RETURN
! Error !
   99 CALL ERROR(' SGRGEN',' Too many elements',NNOW)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SGRPRD(S1,S2,S)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SGRPRD multiplies two (INTEGER-) symmetry operators.       *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S1(3,3) and S2(3,3) are the two matrices to be multiplied.      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S(3,3) contains the product matrix S1*S2.                       *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S1(3,3),S2(3,3),S(3,3),SUM,HELP(3,3)

      DO I=1,3
       DO  J=1,3
         SUM=0
         DO K=1,3
            SUM=SUM+S1(I,K)*S2(K,J)
         ENDDO
         HELP(I,J)=SUM
       ENDDO
      ENDDO
      CALL SGRCOP(HELP,S)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SGRCOP(S1,S2)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SGRPRD copies an (INTEGER-) symmetry operator.             *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S1(9) is the matrix to be copied.                               *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S2(9) is the result matrix (copy of S1).                        *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S1(9),S2(9)

      DO 10 I=1,9
         S2(I)=S1(I)
   10 CONTINUE
      RETURN
      END


!***********************************************************************
!                                                                      *
      LOGICAL FUNCTION SGREQL(S1,S2)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Function SGREQL checks whether two (INTEGER-) symmetry operators   *
!   are identical or not ... .                                         *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S1(9) and S2(9) are two matrices to be compared.                *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      SGREQL is the result (TRUE if S1=S2 and FALSE else).            *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S1(9),S2(9)

      SGREQL=.FALSE.
      DO 10 I=1,9
         IF (S1(I)/=S2(I)) RETURN
   10 CONTINUE
      SGREQL=.TRUE.
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE INVGRP(S,NROT,INVMAP)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine INVGRP searches the inverse operators of a given set of    *
!   'rotation' matrices and creates a map where to find the inverse    *
!   operator ('storage index' of the inverse operator).                *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S(3,3,48) contains the input matrices.                          *
!      NROT gives the number of matrices in S.                         *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      INVMAP(48) contains the index table for the inverse operators.  *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S(3,3,48),G(9),INVMAP(48),E(9)
      LOGICAL SGREQL

      SAVE E
      DATA E /1,0,0,0,1,0,0,0,1/

! For all symmetry operations ... :
      DO 200 IROT=1,NROT
! Test all symmetry operations to find the inverse ... :
         DO 100 JROT=1,NROT
! Multiply the 'rotation' matrix with the test 'rotation' matrix ...
            CALL SGRPRD(S(1,1,IROT),S(1,1,JROT),G(1))
! ... and if the result is equal to the unit operator (unit matrix)
! then we found the inverse operator ... .
            IF (SGREQL(G(1),E(1))) THEN
! Store the result ...
               INVMAP(IROT)=JROT
! ... and leave the 'test loop' ... .
               GOTO 150
            END IF
  100    CONTINUE

         CALL ERROR(' INVGRP', &
                    ' inverse of rotation matrix was not found (increase SYMPREC)',IROT)
         
  150    CONTINUE
  200 CONTINUE
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SGRCON(SA,SB,NROT,A1,A2,A3,B1,B2,B3)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SGRCON converts (INTEGER-) rotation matrices given in some *
!   arbitrary basis A to its representation in some other basis B.     *
!   WARNING: There is one restriction: the two basis sets A and B must *
!            be 'compatible' (same crystal class, inverse basis sets,  *
!            or B = integer linear combination of A) in order to get   *
!            again INTEGER output matrices!! Otherwise use SGRTRF!     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      SA(3,3,48) contains the input matrices (basis A1,A2,A3).        *
!      NROT gives the number of matrices in SA (and hence in SB).      *
!                                                                      *
!      A1(3),A2(3),A3(3) contain the first set of basis vectors.       *
!      B1(3),B2(3),B3(3) contain the second set of basis vectors.      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      SB(3,3,48) contains the converted matrices (basis B1,B2,B3).    *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER SA(3,3,48),SB(3,3,48)
      DIMENSION ADOTBI(3,3),BDOTAI(3,3),PROD(3,3)
      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3),R(3,3)
      DIMENSION AI1(3),AI2(3),AI3(3),BI1(3),BI2(3),BI3(3)

! Calculate reciprocal basis vectors for both sets of direct vectors:
      CALL RECIPS(1._q,A1(1),A2(1),A3(1),AI1(1),AI2(1),AI3(1))
      CALL RECIPS(1._q,B1(1),B2(1),B3(1),BI1(1),BI2(1),BI3(1))
! Set up the matrices of scalar products (a_i*bi_j) and (b_i*ai_j):
      BDOTAI(1,1)=AI1(1)*B1(1)+AI1(2)*B1(2)+AI1(3)*B1(3)
      BDOTAI(2,2)=AI2(1)*B2(1)+AI2(2)*B2(2)+AI2(3)*B2(3)
      BDOTAI(3,3)=AI3(1)*B3(1)+AI3(2)*B3(2)+AI3(3)*B3(3)
      BDOTAI(2,1)=AI1(1)*B2(1)+AI1(2)*B2(2)+AI1(3)*B2(3)
      BDOTAI(3,1)=AI1(1)*B3(1)+AI1(2)*B3(2)+AI1(3)*B3(3)
      BDOTAI(1,2)=AI2(1)*B1(1)+AI2(2)*B1(2)+AI2(3)*B1(3)
      BDOTAI(3,2)=AI2(1)*B3(1)+AI2(2)*B3(2)+AI2(3)*B3(3)
      BDOTAI(1,3)=AI3(1)*B1(1)+AI3(2)*B1(2)+AI3(3)*B1(3)
      BDOTAI(2,3)=AI3(1)*B2(1)+AI3(2)*B2(2)+AI3(3)*B2(3)
      ADOTBI(1,1)=BI1(1)*A1(1)+BI1(2)*A1(2)+BI1(3)*A1(3)
      ADOTBI(2,2)=BI2(1)*A2(1)+BI2(2)*A2(2)+BI2(3)*A2(3)
      ADOTBI(3,3)=BI3(1)*A3(1)+BI3(2)*A3(2)+BI3(3)*A3(3)
      ADOTBI(2,1)=BI1(1)*A2(1)+BI1(2)*A2(2)+BI1(3)*A2(3)
      ADOTBI(3,1)=BI1(1)*A3(1)+BI1(2)*A3(2)+BI1(3)*A3(3)
      ADOTBI(1,2)=BI2(1)*A1(1)+BI2(2)*A1(2)+BI2(3)*A1(3)
      ADOTBI(3,2)=BI2(1)*A3(1)+BI2(2)*A3(2)+BI2(3)*A3(3)
      ADOTBI(1,3)=BI3(1)*A1(1)+BI3(2)*A1(2)+BI3(3)*A1(3)
      ADOTBI(2,3)=BI3(1)*A2(1)+BI3(2)*A2(2)+BI3(3)*A2(3)
! For all (rotation-) matrices:
      DO 100 I=1,NROT
         R(1,1)=FLOAT(SA(1,1,I))
         R(2,1)=FLOAT(SA(2,1,I))
         R(3,1)=FLOAT(SA(3,1,I))
         R(1,2)=FLOAT(SA(1,2,I))
         R(2,2)=FLOAT(SA(2,2,I))
         R(3,2)=FLOAT(SA(3,2,I))
         R(1,3)=FLOAT(SA(1,3,I))
         R(2,3)=FLOAT(SA(2,3,I))
         R(3,3)=FLOAT(SA(3,3,I))
! Perform the transformation (matrix multiplications):
         DO II=1,3
          DO JJ=1,3
           SUM=0._q
           DO KK=1,3
              SUM=SUM+BDOTAI(II,KK)*R(KK,JJ)
           ENDDO
           PROD(II,JJ)=SUM
          ENDDO
         ENDDO
         DO II=1,3
          DO JJ=1,3
            SUM=0._q
            DO KK=1,3
               SUM=SUM+PROD(II,KK)*ADOTBI(KK,JJ)
            ENDDO
            R(II,JJ)=SUM
          ENDDO
         ENDDO
! Ready! Copy the result into SB ... :
         SB(1,1,I)=NINT(R(1,1))
         SB(2,1,I)=NINT(R(2,1))
         SB(3,1,I)=NINT(R(3,1))
         SB(1,2,I)=NINT(R(1,2))
         SB(2,2,I)=NINT(R(2,2))
         SB(3,2,I)=NINT(R(3,2))
         SB(1,3,I)=NINT(R(1,3))
         SB(2,3,I)=NINT(R(2,3))
         SB(3,3,I)=NINT(R(3,3))
! Error check: We must end up with INTEGER matrices ... !
         IF (ABS(R(1,1)-SB(1,1,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(2,1)-SB(2,1,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(3,1)-SB(3,1,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(1,2)-SB(1,2,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(2,2)-SB(2,2,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(3,2)-SB(3,2,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(1,3)-SB(1,3,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(2,3)-SB(2,3,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
         IF (ABS(R(3,3)-SB(3,3,I))>TINY) CALL ERROR(' SGRCON', &
     &           ' Found some non-integer element in rotation matrix',I)
  100 CONTINUE
! Good bye ... :
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SGRTRF(SA,SB,NROT,A1,A2,A3,B1,B2,B3)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SGRTRF converts (REAL-) rotation matrices given in some    *
!   arbitrary basis A to its representation in some other basis B.     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      SA(3,3,48) contains the input matrices (basis A1,A2,A3).        *
!      NROT gives the number of matrices in SA (and hence in SB).      *
!                                                                      *
!      A1(3),A2(3),A3(3) contain the first set of basis vectors.       *
!      B1(3),B2(3),B3(3) contain the second set of basis vectors.      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      SB(3,3,48) contains the converted matrices (basis B1,B2,B3).    *
!                                                                      *
!                                                                      *
!***********************************************************************

      DIMENSION SA(3,3,48),SB(3,3,48)
      DIMENSION ADOTBI(3,3),BDOTAI(3,3),PROD(3,3)
      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3),R(3,3)
      DIMENSION AI1(3),AI2(3),AI3(3),BI1(3),BI2(3),BI3(3)

! Calculate reciprocal basis vectors for both sets of direct vectors:
      CALL RECIPS(1._q,A1(1),A2(1),A3(1),AI1(1),AI2(1),AI3(1))
      CALL RECIPS(1._q,B1(1),B2(1),B3(1),BI1(1),BI2(1),BI3(1))
! Set up the matrices of scalar products (a_i*bi_j) and (b_i*ai_j):
      BDOTAI(1,1)=AI1(1)*B1(1)+AI1(2)*B1(2)+AI1(3)*B1(3)
      BDOTAI(2,2)=AI2(1)*B2(1)+AI2(2)*B2(2)+AI2(3)*B2(3)
      BDOTAI(3,3)=AI3(1)*B3(1)+AI3(2)*B3(2)+AI3(3)*B3(3)
      BDOTAI(2,1)=AI1(1)*B2(1)+AI1(2)*B2(2)+AI1(3)*B2(3)
      BDOTAI(3,1)=AI1(1)*B3(1)+AI1(2)*B3(2)+AI1(3)*B3(3)
      BDOTAI(1,2)=AI2(1)*B1(1)+AI2(2)*B1(2)+AI2(3)*B1(3)
      BDOTAI(3,2)=AI2(1)*B3(1)+AI2(2)*B3(2)+AI2(3)*B3(3)
      BDOTAI(1,3)=AI3(1)*B1(1)+AI3(2)*B1(2)+AI3(3)*B1(3)
      BDOTAI(2,3)=AI3(1)*B2(1)+AI3(2)*B2(2)+AI3(3)*B2(3)
      ADOTBI(1,1)=BI1(1)*A1(1)+BI1(2)*A1(2)+BI1(3)*A1(3)
      ADOTBI(2,2)=BI2(1)*A2(1)+BI2(2)*A2(2)+BI2(3)*A2(3)
      ADOTBI(3,3)=BI3(1)*A3(1)+BI3(2)*A3(2)+BI3(3)*A3(3)
      ADOTBI(2,1)=BI1(1)*A2(1)+BI1(2)*A2(2)+BI1(3)*A2(3)
      ADOTBI(3,1)=BI1(1)*A3(1)+BI1(2)*A3(2)+BI1(3)*A3(3)
      ADOTBI(1,2)=BI2(1)*A1(1)+BI2(2)*A1(2)+BI2(3)*A1(3)
      ADOTBI(3,2)=BI2(1)*A3(1)+BI2(2)*A3(2)+BI2(3)*A3(3)
      ADOTBI(1,3)=BI3(1)*A1(1)+BI3(2)*A1(2)+BI3(3)*A1(3)
      ADOTBI(2,3)=BI3(1)*A2(1)+BI3(2)*A2(2)+BI3(3)*A2(3)
! For all (rotation-) matrices:
      DO 100 I=1,NROT
         R(1,1)=SA(1,1,I)
         R(2,1)=SA(2,1,I)
         R(3,1)=SA(3,1,I)
         R(1,2)=SA(1,2,I)
         R(2,2)=SA(2,2,I)
         R(3,2)=SA(3,2,I)
         R(1,3)=SA(1,3,I)
         R(2,3)=SA(2,3,I)
         R(3,3)=SA(3,3,I)
! Perform the transformation (matrix multiplications):
         DO II=1,3
          DO  JJ=1,3
           SUM=0._q
           DO KK=1,3
              SUM=SUM+BDOTAI(II,KK)*R(KK,JJ)
           ENDDO
           PROD(II,JJ)=SUM
          ENDDO
         ENDDO
         DO II=1,3
          DO JJ=1,3
            SUM=0._q
            DO KK=1,3
               SUM=SUM+PROD(II,KK)*ADOTBI(KK,JJ)
            ENDDO
            R(II,JJ)=SUM
          ENDDO
         ENDDO
! Ready! Copy the result into SB ... :
         SB(1,1,I)=R(1,1)
         SB(2,1,I)=R(2,1)
         SB(3,1,I)=R(3,1)
         SB(1,2,I)=R(1,2)
         SB(2,2,I)=R(2,2)
         SB(3,2,I)=R(3,2)
         SB(1,3,I)=R(1,3)
         SB(2,3,I)=R(2,3)
         SB(3,3,I)=R(3,3)
  100 CONTINUE
! Good bye ... :
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE VECCON(VA,VB,NVEC,A1,A2,A3,B1,B2,B3)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine VECCON converts a set of vectors represented in some       *
!   arbitrary basis A to its representation in some other basis B.     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      VA(3,NVEC) contains the input vectors (basis A1,A2,A3).         *
!      NVEC gives the number of vectors in VA (and hence in VB).       *
!                                                                      *
!      A1(3),A2(3),A3(3) contain the first set of basis vectors.       *
!      B1(3),B2(3),B3(3) contain the second set of basis vectors.      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      VB(3,NVEC) contains the converted vectors (basis B1,B2,B3).     *
!                                                                      *
!                                                                      *
!***********************************************************************

      DIMENSION VA(3,NVEC),VB(3,NVEC)
      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)
      DIMENSION BI1(3),BI2(3),BI3(3),V(3),R(3)

! Calculate reciprocal basis vectors for second set of direct vectors:
      CALL RECIPS(1._q,B1(1),B2(1),B3(1),BI1(1),BI2(1),BI3(1))
! For all vectors ... :
      DO 100 I=1,NVEC
! Copy the vector:
         V(1)=VA(1,I)
         V(2)=VA(2,I)
         V(3)=VA(3,I)
! Perform the transformation:
         R(1)=A1(1)*V(1)+A2(1)*V(2)+A3(1)*V(3)
         R(2)=A1(2)*V(1)+A2(2)*V(2)+A3(2)*V(3)
         R(3)=A1(3)*V(1)+A2(3)*V(2)+A3(3)*V(3)
         V(1)=BI1(1)*R(1)+BI1(2)*R(2)+BI1(3)*R(3)
         V(2)=BI2(1)*R(1)+BI2(2)*R(2)+BI2(3)*R(3)
         V(3)=BI3(1)*R(1)+BI3(2)*R(2)+BI3(3)*R(3)
! Store the result:
         VB(1,I)=V(1)
         VB(2,I)=V(2)
         VB(3,I)=V(3)
  100 CONTINUE
! Good bye ... :
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE SETGRPV(S,SYMOP,GTRANS,NROT,NROTK,TAU,VEC,SPIN,LATTYP,NOP, &
     &                         TAUROT,NSP,NSX,NA,NAX,NC,INDEX,WRK,PRCHAN)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine SETGRP is the primary user-interface of this symmetry      *
!   package. The function of SETGRP is to set up all possible          *
!   space group operators (INTEGER rotation matrices and nontrivial    *
!   translations given in crystal coordinates) of a lattice with       *
!   some arbitrary basis (atomic arrangement). This will be done by    *
!   first setting up the point group operators of the pure bravais     *
!   lattice without basis (empty supercell) and checking which of the  *
!   operations can reproduce the lattice with basis (to be done in     *
!   routine GETGRP - for further details see comments on GETGRP).      *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output! TAU will contain the same set of    *
!              atomic position as on input, but the order of storage   *
!              of the positions will be changed in some way described  *
!              in routine LATORD of the lattice vector package and     *
!              periodic boundary conditions are imposed on the atomic  *
!              coordinates to shift all atoms into a box defined by    *
!              the corner points (+/- 0.5, +/- 0.5, +/- 0.5).          *
!                                                                      *
!      LATTYP gives the bravais lattice type of the supercell as       *
!             coded in routine LATGEN (parameter IBRAV) of the         *
!             lattice vector package (LATTYP may run from 1 to 14).    *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      PRCHAN contains the unit number for standard output.            *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices for all        *
!                space group operations.                               *
!                                                                      *
!      SYMOP(3,3,48) contains the rotation matrices of the pure        *
!                bravais lattice.                                      *
!                                                                      *
!      GTRANS(3,48) contains the nonprimitive translations for         *
!                all space group operations.                           *
!                                                                      *
!      NROT contains the number of pure point group rotations (to      *
!           be stored in S(3,3,1)...S(3,3,NROT)) and                   *
!      NROTK contains the number of space group operations found       *
!           in routine GETGRP.                                         *
!      NOP contains the number of point group operations of the pure   *
!           bravais lattice without basis.                             *
!                                                                      *
!                                                                      *
!***********************************************************************

      INTEGER S,SYMOP,SYMGEN(9,3),INV(9),R6Z(9),R3D(9),R2HEX(9)
      INTEGER R2TRI(9),R2YP(9),R4ZP(9),R4ZBC(9),R4ZFC(9),R2ZP(9)
      INTEGER R2YBC(9),R2ZBC(9),R2YBAS(9),R2YFC(9),R2ZFC(9),PRCHAN

      DIMENSION S(3,3,48),GTRANS(3,48),SYMOP(3,3,48),NA(NSP)
      DIMENSION TAU(NAX,3,NSX),TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX)
      DIMENSION VEC(NAX,3,NSX,NC),SPIN(NAX,NSX)

      SAVE INV,R6Z,R3D,R2HEX,R2TRI,R2YP,R4ZP,R4ZBC,R4ZFC,R2ZP,R2YBC
      SAVE R2ZBC,R2YBAS,R2YFC,R2ZFC

      DATA INV /-1,0,0,0,-1,0,0,0,-1/,R3D /0,0,1,1,0,0,0,1,0/
      DATA R6Z /1,-1,0,1,0,0,0,0,1/,R2HEX /1,-1,0,0,-1,0,0,0,-1/
      DATA R2TRI /-1,0,0,0,0,-1,0,-1,0/,R4ZP /0,-1,0,1,0,0,0,0,1/
      DATA R2YP /-1,0,0,0,1,0,0,0,-1/,R4ZBC /0,1,0,0,1,-1,-1,1,0/
      DATA R4ZFC /1,1,1,0,0,-1,-1,0,0/,R2ZP /-1,0,0,0,-1,0,0,0,1/
      DATA R2YBC /0,-1,1,0,-1,0,1,-1,0/,R2ZBC /0,1,-1,1,0,-1,0,0,-1/
      DATA R2YBAS /0,-1,0,-1,0,0,0,0,-1/,R2YFC /0,0,1,-1,-1,-1,1,0,0/
      DATA R2ZFC /0,1,0,1,0,0,-1,-1,-1/

! The pure translation lattice (bravais lattice) has some maximum
! symmetry. Set first up the point group operations for this symmetry.
! Remark: All these groups contain the inversion as a generator ... :
      CALL SGRCOP(INV,SYMGEN(1,1))
      IF (LATTYP==14) THEN
! Triclinic symmetry:
         CALL SGRGEN(SYMGEN,1,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,14)
   14 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' triclinic supercell.'/)
      ELSE IF (LATTYP==13) THEN
! Monoclinic symmetry (base centered / 'b-a' is rotation axis ...):
         CALL SGRCOP(R2YBAS,SYMGEN(1,2))
         CALL SGRGEN(SYMGEN,2,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,13)
   13 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' base centered monoclinic supercell.'/)
      ELSE IF (LATTYP==12) THEN
! Monoclinic symmetry (primitive cell / 'b-axis' is rotation axis ...):
         CALL SGRCOP(R2YP,SYMGEN(1,2))
         CALL SGRGEN(SYMGEN,2,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,12)
   12 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple monoclinic supercell.'/)
      ELSE IF (LATTYP==11) THEN
! Orthorhombic symmetry (base centered ...):
         CALL SGRCOP(R2ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YBAS,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,11)
   11 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' base centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==10) THEN
! Orthorhombic symmetry (face centered ...):
         CALL SGRCOP(R2ZFC,SYMGEN(1,2))
         CALL SGRCOP(R2YFC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,10)
   10 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' face centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==9) THEN
! Orthorhombic symmetry (body centered ...):
         CALL SGRCOP(R2ZBC,SYMGEN(1,2))
         CALL SGRCOP(R2YBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,9)
    9 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered orthorhombic supercell.'/)
      ELSE IF (LATTYP==8) THEN
! Orthorhombic symmetry (primitive cell ...):
         CALL SGRCOP(R2ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,8)
    8 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple orthorhombic supercell.'/)
      ELSE IF (LATTYP==7) THEN
! Trigonal (rhombohedral) symmetry:
         CALL SGRCOP(R2TRI,SYMGEN(1,2))
         CALL SGRCOP(R3D,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,7)
    7 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' trigonal (rhombohedral) supercell.'/)
      ELSE IF (LATTYP==6) THEN
! Tetragonal symmetry (body centered / 'c-axis' is the special axis):
         CALL SGRCOP(R4ZBC,SYMGEN(1,2))
         CALL SGRCOP(R2YBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,6)
    6 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered tetragonal supercell.'/)
      ELSE IF (LATTYP==5) THEN
! Tetragonal symmetry (primitive cell / 'c-axis' is the special axis):
         CALL SGRCOP(R4ZP,SYMGEN(1,2))
         CALL SGRCOP(R2YP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,5)
    5 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple tetragonal supercell.'/)
      ELSE IF (LATTYP==4) THEN
! Hexagonal symmetry ('c-axis' is the rotation axis):
         CALL SGRCOP(R6Z,SYMGEN(1,2))
         CALL SGRCOP(R2HEX,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,4)
    4 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' hexagonal supercell.'/)
      ELSE IF (LATTYP==3) THEN
! Cubic symmetry (face centered ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZFC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,3)
    3 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' face centered cubic supercell.'/)
      ELSE IF (LATTYP==2) THEN
! Cubic symmetry (body centered ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZBC,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,2)
    2 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' body centered cubic supercell.'/)
      ELSE IF (LATTYP==1) THEN
! Cubic symmetry (primitive cell ...):
         CALL SGRCOP(R3D,SYMGEN(1,2))
         CALL SGRCOP(R4ZP,SYMGEN(1,3))
         CALL SGRGEN(SYMGEN,3,SYMOP,NOP)
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) WRITE(PRCHAN,1)
    1 FORMAT(/' Routine SETGRP: Setting up the symmetry group for a '/, &
     &        ' simple cubic supercell.'/)
      END IF
! Now select all symmetry operations which reproduce the lattice:
      CALL GETGRPV(S,GTRANS,NROT,NROTK,SYMOP,NOP,TAU,VEC,SPIN, &
     &                           TAUROT,NSP,NSX,NA,NAX,NC,INDEX,WRK,PRCHAN)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE GETGRPV(S,GTRANS,NROT,NROTK,SYMOP,NOP,TAU,VEC,SPIN, &
     &                         TAUROT,NSP,NSX,NA,NAX,NC,INDEX,WRK,PRCHAN)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine GETGRP is a secondary user-interface of this symmetry      *
!   package. The function of GETGRP is to return all possible space    *
!   group operators that reproduce a lattice with basis out of a       *
!   (maximum) pool of point group operations that is compatible with   *
!   the symmetry of the pure translation lattice without any basis.    *
!   The setup of the space group operations is done in the following   *
!   way: We pass through the pool of (possibly allowed) symmetry       *
!   operations and check each operation whether it can reproduce the   *
!   lattice with basis - possibly connected with some nontrivial       *
!   translation. This check will be done by routine CHKSYM (further    *
!   details will be described there). If the answer is positive the    *
!   operation will be stored (and counted) - otherwise it will be      *
!   thrown away.                                                       *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      SYMOP(3,3,48) contains the rotation matrices of the pure        *
!                bravais lattice.                                      *
!                                                                      *
!      NOP contains the number of rotation matrices in array SYMOP     *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (see comment on routine SETGRP)!     *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      PRCHAN contains the unit number for standard output.            *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      S(3,3,48) contains the INTEGER rotation matrices for all        *
!                space group operations.                               *
!                                                                      *
!      GTRANS(3,48) contains the nonprimitive translations for         *
!                all space group operations.                           *
!                                                                      *
!      NROT contains the number of pure point group rotations (to      *
!           be stored in S(3,3,1)...S(3,3,NROT)) and                   *
!      NROTK contains the number of space group operations found       *
!           in routine GETGRP.                                         *
!                                                                      *
!                                                                      *
!***********************************************************************
      LOGICAL TS
      INTEGER S,SYMOP,ZERO(9),HELP(3,3,48),PRCHAN

      DIMENSION S(3,3,48),GTRANS(3,48),SYMOP(3,3,48),TEMP(3,48),NA(NSP)
      DIMENSION TAU(NAX,3,NSX),TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX)
      DIMENSION VEC(NAX,3,NSX,NC),SPIN(NAX,NSX)

      SAVE ZERO
      DATA ZERO /0,0,0,0,0,0,0,0,0/

      NROT=0
      NROTK=0
! Check the operations of a 'pool' of all possible operations ... :
      DO 1 I=1,NOP
         CALL CHKSYMV(SYMOP(1,1,I),GTRANS(1,I),TS,TAU,VEC,SPIN,TAUROT, &
     &                                         NSP,NSX,NA,NAX,NC,INDEX,WRK)
         IF (TS) THEN
! Hurray! We found some symmetry operation in subroutine CHKSYM ... :
            IF ((ABS(GTRANS(1,I))<TINY) &
     &               .AND.(ABS(GTRANS(2,I))<TINY) &
     &                          .AND.(ABS(GTRANS(3,I))<TINY)) THEN
! Pure point group operations (no nontrivial translation):
               NROT=NROT+1
               CALL SGRCOP(SYMOP(1,1,I),S(1,1,NROT))
               GTRANS(1,NROT)=0._q
               GTRANS(2,NROT)=0._q
               GTRANS(3,NROT)=0._q
            ELSE
! Space group operations:
               NROTK=NROTK+1
               CALL SGRCOP(SYMOP(1,1,I),HELP(1,1,NROTK))
               TEMP(1,NROTK)=GTRANS(1,I)
               TEMP(2,NROTK)=GTRANS(2,I)
               TEMP(3,NROTK)=GTRANS(3,I)
            END IF
         END IF
    1 CONTINUE
      IF (NROTK>0) THEN
! If there are operations with nontrivial translations then store them
! into the array S and GTRANS (placing them after the last pure point
! group operations S(I,J,NROT) ):
         DO 2 I=1,NROTK
            CALL SGRCOP(HELP(1,1,I),S(1,1,I+NROT))
            GTRANS(1,I+NROT)=TEMP(1,I)
            GTRANS(2,I+NROT)=TEMP(2,I)
            GTRANS(3,I+NROT)=TEMP(3,I)
    2    CONTINUE
      END IF
! Total number of space group operations:
      NROTK=NROTK+NROT
      IF (NROTK<48) THEN
! Fill the rest of arrays S and GTRANS with zeros ... :
         DO 3 I=NROTK+1,48
            CALL SGRCOP(ZERO,S(1,1,I))
            GTRANS(1,I)=0._q
            GTRANS(2,I)=0._q
            GTRANS(3,I)=0._q
    3    CONTINUE
      END IF
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) &
     &                                  WRITE(PRCHAN,100) NROTK,NROT,NOP
  100 FORMAT(/' Subroutine GETGRP returns: Found ',I2, &
     &        ' space group operations'/,' (whereof ',I2, &
     &        ' operations were pure point group operations)'/, &
     &        ' out of a pool of ',I2,' trial point group operations.'/)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE CHKSYMV(S,GTRANS,TS,TAU,VEC,SPIN,TAUROT,NSP,NSX,NA,NAX,NC,INDEX,WRK)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine CHKSYM checks whether a point group symmetry element is    *
!   a valid symmetry operation on a supercell (possibly connected with *
!   some nontrivial translation) or not. The procedure is in principle *
!   very easy: The point group transformation will be imposed on all   *
!   atomic positions (for all species) and then the rotated supercell  *
!   will be compared with the original cell. There are only three      *
!   possibilities: All positions will be reproduced (pure point group  *
!   operation), all positions will be reproduced after an additional   *
!   unique translation of all atoms (general space group operation)    *
!   or it will not be possible to reproduce the supercell in any way   *
!   (no allowed operation). The translation can be guessed quite easy  *
!   by looking on the first atom: If we try all difference vectors     *
!   between the position of the first atom of the unrotated cell and   *
!   the positions of all atoms of the rotated cell then one of these   *
!   difference vectors must be the translation vector (if it exists!). *
!   This vector must apply to all atoms. So we translate all atoms in  *
!   the rotated cell by minus this vector - this should reproduce the  *
!   original positions of the unrotated cell. The central point is now *
!   only how to compare the rotated and the unrotated supercell. The   *
!   algorithm used bases on the following idea: If we sort all atomic  *
!   positions within a cell as it is described in routine LATORD of    *
!   the lattice vector package (sorting first the x-coordinates, then  *
!   sorting the y-coordinates of all atoms with equal x-coordinates    *
!   and finally sorting all z-cordinates for all atomic positions with *
!   equal x- and y-coordinates  -  or using other ordering sequences   *
!   instead of xyz-ordering ...) then we get some unique sequence of   *
!   ordered positions. Because the same set of input data will mean    *
!   identical output we can simply perform a one-by-one comparison of  *
!   the unrotated (ordered) and the rotated (ordered and translated)   *
!   coordinates. If we have found some group operation the comparison  *
!   will give the value true ('all positions are equal'). In all other *
!   cases the result will be false ('some positions are not equal'!    *
!   There is just one point one has to care about: Supercells which    *
!   are non-primitive. Then there exist also 'non-primitive primitive' *
!   translations (trivial translations of the generating cell ...)!    *
!   To avoid trouble with non-uniquely defined translations in this    *
!   case we just take the vectors with smallest coordinates (from all  *
!   the vectors which will be detected to be an allowed translation).  *
!                                                                      *
!   It should be trivial but will be remarked finally: The procedure   *
!   must be performed on every atomic species seperately and the       *
!   result (allowed operation or not / nontrivial translation) must be *
!   the same for all atomic species - otherwise the symmetry operation *
!   is not a valid operation ... .                                     *
!                                                                      *
!   Final very important notice: Comparison of atomic positions will   *
!   be done by looking at differences of coordinates (which must be    *
!   smaller than some tolerance). The allowed tolerances for the       *
!   coordinates are very small (1E-9 absolute)!!! Therefore all input  *
!   coordinates should be given with an accuray of at least +/-1E-10   *
!   because otherwise this routine could work incorrectly (could fail  *
!   to detect space group operations ...)!                             *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      S(3,3) is the point group symmetry element to be checked for    *
!             validity (INTEGER rotation matrix).                      *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (see comment on routine SETGRP)!     *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      GTRANS(3) contains the nonprimitive translations which is       *
!                connected with the rotation matrix S(3,3).            *
!                                                                      *
!      TS is a logical value which returns whether S(3,3) is a valid   *
!         symmetry operation on the supercell or not.                  *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TS,TNTRAN,TCONT
      LOGICAL TSPIN,TMAYFLIP,TFLIPPED
      INTEGER S

      DIMENSION S(3,3),GTRANS(3),NA(NSP),TAU(NAX,3,NSX),TAUDIF(3)
      DIMENSION TAUROT(NAX,3,NSX),WRK(NAX,3),INDEX(NAX),TAUSAV(3),TRA(3)
      DIMENSION VEC(NAX,3,NSX,NC),VECROT(NAX,3,NSX,NC),VECDIF(3,NC)
      DIMENSION SPIN(NAX,NSX),SPINROT(NAX,NSX)

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
! Now start ... :
      ISMIN=1
      TRA(1)=2._q
      TRA(2)=2._q
      TRA(3)=2._q
      TS=.FALSE.
      ISTART=1
      IMINST=1
! For all atomic species ... :
      DO 3 IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Search the species with smallest number of atoms ... :
         IF (NA(IS)<NA(ISMIN)) THEN
            ISMIN=IS
            IMINST=ISTART
         END IF
         DO 1 IA=ISTART,NA(IS)+ISTART-1
! First impose periodic boundary conditions on TAU (--> [-0.5,0.5) ):
            TAU(IA,1,ISP)=TAU(IA,1,ISP)-NINT(TAU(IA,1,ISP))
            TAU(IA,2,ISP)=TAU(IA,2,ISP)-NINT(TAU(IA,2,ISP))
            TAU(IA,3,ISP)=TAU(IA,3,ISP)-NINT(TAU(IA,3,ISP))
            TAU(IA,1,ISP)=MOD(TAU(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,2,ISP)=MOD(TAU(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAU(IA,3,ISP)=MOD(TAU(IA,3,ISP)+100.5_q,1._q)-0.5_q
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
#ifdef  periodic_image_convention
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=TAU(IA,1,ISP)-1.0_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=TAU(IA,2,ISP)-1.0_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=TAU(IA,3,ISP)-1.0_q
#else
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
#endif
    1    CONTINUE
! Order the atomic coordiantes (for this species) ... :
         CALL LATORD(TAU(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
! and reorder VEC in the same way 
         DO IC=1,NC
            DO IA=1,NA(IS)
               WRK(IA,:)=VEC(ISTART+INDEX(IA)-1,:,ISP,IC) 
            ENDDO
            VEC(ISTART:ISTART+NA(IS)-1,:,ISP,IC)=WRK(1:NA(IS),:)
         ENDDO
! and SPIN as well
         DO IA=1,NA(IS)
            WRK(IA,1)=SPIN(ISTART+INDEX(IA)-1,ISP)
         ENDDO
         SPIN(ISTART:ISTART+NA(IS)-1,ISP)=WRK(1:NA(IS),1)
! Rotate all coordinates ... :
         DO 2 IA=ISTART,NA(IS)+ISTART-1
            TAUROT(IA,1,ISP)=S(1,1)*TAU(IA,1,ISP)+ &
     &                                S(2,1)*TAU(IA,2,ISP)+ &
     &                                              S(3,1)*TAU(IA,3,ISP)
            TAUROT(IA,2,ISP)=S(1,2)*TAU(IA,1,ISP)+ &
     &                                S(2,2)*TAU(IA,2,ISP)+ &
     &                                              S(3,2)*TAU(IA,3,ISP)
            TAUROT(IA,3,ISP)=S(1,3)*TAU(IA,1,ISP)+ &
     &                                S(2,3)*TAU(IA,2,ISP)+ &
     &                                              S(3,3)*TAU(IA,3,ISP)
! ... and impose periodic boundary conditions (TAUROT --> [-0.5,0.5) ):
            TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
            TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
            TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! It could happen that (due to roundoff errors!) the surface coordinates
! are not uniquely -0.5 but probably +0.49999999999... . Ensure that all
! these coordinates are transformed to -0.5!
#ifdef  periodic_image_convention
            IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-1.0_q
            IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-1.0_q
            IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-1.0_q
#else
            IF ((ABS(TAUROT(IA,1,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,2,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,3,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
            IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
#endif
! ... and rotate VEC as well
            DO IC=1,NC
               VECROT(IA,1,ISP,IC)=S(1,1)*VEC(IA,1,ISP,IC)+ &
     &                                   S(2,1)*VEC(IA,2,ISP,IC)+ &
     &                                                 S(3,1)*VEC(IA,3,ISP,IC)
               VECROT(IA,2,ISP,IC)=S(1,2)*VEC(IA,1,ISP,IC)+ &
     &                                   S(2,2)*VEC(IA,2,ISP,IC)+ &
     &                                                 S(3,2)*VEC(IA,3,ISP,IC)
               VECROT(IA,3,ISP,IC)=S(1,3)*VEC(IA,1,ISP,IC)+ &
     &                                   S(2,3)*VEC(IA,2,ISP,IC)+ &
     &                                                 S(3,3)*VEC(IA,3,ISP,IC)
            ENDDO
#ifndef oldsym
! ... for noncollinear spins we might have a flip
            DET=S(1,1)*S(2,2)*S(3,3)-S(1,1)*S(2,3)*S(3,2) + &
     &          S(1,2)*S(2,3)*S(3,1)-S(1,2)*S(2,1)*S(3,3) + &
     &          S(1,3)*S(2,1)*S(3,2)-S(1,3)*S(2,2)*S(3,1)
            VECROT(IA,:,ISP,3)=DET*VECROT(IA,:,ISP,3)
#endif
    2    CONTINUE
! Order the rotated atomic coordiantes ... :
         CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
! and reorder VECROT in the same way 
         DO IC=1,NC
            DO IA=1,NA(IS)
               WRK(IA,:)=VECROT(ISTART+INDEX(IA)-1,:,ISP,IC) 
            ENDDO
            VECROT(ISTART:ISTART+NA(IS)-1,:,ISP,IC)=WRK(1:NA(IS),:)
         ENDDO
! and SPINROT as well
         DO IA=1,NA(IS)
            WRK(IA,1)=SPIN(ISTART+INDEX(IA)-1,ISP)
         ENDDO
         SPINROT(ISTART:ISTART+NA(IS)-1,ISP)=WRK(1:NA(IS),1)
    3 CONTINUE
! Save the rotated coordinate of the first atom of species ISMIN:
      IF (TCONT) THEN
         TAUSAV(1)=TAUROT(IMINST,1,1)
         TAUSAV(2)=TAUROT(IMINST,2,1)
         TAUSAV(3)=TAUROT(IMINST,3,1)
         ISPMIN=1
      ELSE
         TAUSAV(1)=TAUROT(1,1,ISMIN)
         TAUSAV(2)=TAUROT(1,2,ISMIN)
         TAUSAV(3)=TAUROT(1,3,ISMIN)
         ISPMIN=ISMIN
      END IF
! All difference vectors between the first rotated atom and all possible
! original vectors of species ISMIN could be a translation vector being
! associated with the given rotation ... . So test all possibilities:
      DO 10 IATEST=IMINST,NA(ISMIN)+IMINST-1
! Set up the 'test vector' GTRANS ... :
         GTRANS(1)=TAU(IATEST,1,ISPMIN)-TAUSAV(1)
         GTRANS(2)=TAU(IATEST,2,ISPMIN)-TAUSAV(2)
         GTRANS(3)=TAU(IATEST,3,ISPMIN)-TAUSAV(3)
! GTRANS could possibly contain trivial translations:
         GTRANS(1)=MOD((GTRANS(1)+100._q),1._q)
         GTRANS(2)=MOD((GTRANS(2)+100._q),1._q)
         GTRANS(3)=MOD((GTRANS(3)+100._q),1._q)
! For reasons of safety:
#ifdef  periodic_image_convention
         IF ((ABS(GTRANS(1)-1._q))<(TINY*0.5_q)) GTRANS(1)=GTRANS(1)-1._q
         IF ((ABS(GTRANS(2)-1._q))<(TINY*0.5_q)) GTRANS(2)=GTRANS(2)-1._q
         IF ((ABS(GTRANS(3)-1._q))<(TINY*0.5_q)) GTRANS(3)=GTRANS(3)-1._q
#else
         IF ((ABS(GTRANS(1)-1._q))<(TINY*0.5_q)) GTRANS(1)=0._q
         IF ((ABS(GTRANS(2)-1._q))<(TINY*0.5_q)) GTRANS(2)=0._q
         IF ((ABS(GTRANS(3)-1._q))<(TINY*0.5_q)) GTRANS(3)=0._q
#endif
! If we had already detected some translation (non-primitve supercell?)
! we must only look at the vectors with coordinates smaller than those
! of the previously detected vector ... :
         IF ((GTRANS(1)>(TRA(1)+TINY)).OR. &
     &                 (GTRANS(2)>(TRA(2)+TINY)).OR. &
     &                             (GTRANS(3)>(TRA(3)+TINY))) GOTO 10
! Translate the rotated coordinates by GTRANS ...
         ISTART=1
         DO 5 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 4 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)+GTRANS(1)
               TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)+GTRANS(2)
               TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)+GTRANS(3)
! ... impose the periodic boundary condition on these coordinates ...
               TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! ... ensuring that all surface coordinates are put to -0.5 ...
#ifdef  periodic_image_convention
               IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-1.0_q
               IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-1.0_q
               IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-1.0_q
#else
               IF ((ABS(TAUROT(IA,1,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,2,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,3,ISP)+0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,1,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,1,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,2,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,2,ISP)=-0.5_q
               IF ((ABS(TAUROT(IA,3,ISP)-0.5_q))<TINY) &
     &                                             TAUROT(IA,3,ISP)=-0.5_q
#endif
    4       CONTINUE
! ... and order the coordinates:
            CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
! and reorder VECROT in the same way 
            DO IC=1,NC
               DO IA=1,NA(IS)
                  WRK(IA,:)=VECROT(ISTART+INDEX(IA)-1,:,ISP,IC) 
               ENDDO
               VECROT(ISTART:ISTART+NA(IS)-1,:,ISP,IC)=WRK(1:NA(IS),:)
            ENDDO
! and SPINROT as well
            DO IA=1,NA(IS)
               WRK(IA,1)=SPINROT(ISTART+INDEX(IA)-1,ISP)
            ENDDO
            SPINROT(ISTART:ISTART+NA(IS)-1,ISP)=WRK(1:NA(IS),1)
    5    CONTINUE
! Now compare the two lattices 'one-by-one' whether they are identical
! (possibly differing by a translation vector GTRANS ...) or not ... .
         TNTRAN=.TRUE.
! logicals to keep track of spinflips
         TMAYFLIP=.TRUE.; TFLIPPED=.FALSE.
! For all atoms of the unit cell ... :
         ISTART=1
         DO 7 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 6 IA=ISTART,NA(IS)+ISTART-1
#ifdef periodic_image_convention
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
               TAUDIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               TAUDIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               TAUDIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
               TAUDIF(1)=MIN_IMAG(TAUDIF(1))
               TAUDIF(2)=MIN_IMAG(TAUDIF(2))
               TAUDIF(3)=MIN_IMAG(TAUDIF(3))
#else
! Take the difference of the rotated and the unrotated coordinates after
! we translated them - the result should be (0.,0.,0.)!
               TAUDIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               TAUDIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               TAUDIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! TAUDIF may only differ from (0.,0.,0.) by some trivial translation:
               TAUDIF(1)=MOD((TAUDIF(1)+100._q),1._q)
               TAUDIF(2)=MOD((TAUDIF(2)+100._q),1._q)
               TAUDIF(3)=MOD((TAUDIF(3)+100._q),1._q)
! For reasons of safety:
               IF ((ABS(TAUDIF(1)-1._q))<(TINY*0.5_q)) TAUDIF(1)=0._q
               IF ((ABS(TAUDIF(2)-1._q))<(TINY*0.5_q)) TAUDIF(2)=0._q
               IF ((ABS(TAUDIF(3)-1._q))<(TINY*0.5_q)) TAUDIF(3)=0._q
#endif
! VECDIF is the difference between the vector quantities at the original
! and translated coordinates, the result should be (0.,0.,0.) as well.
               DO IC=1,NC
                  VECDIF(1,IC)=VEC(IA,1,ISP,IC)-VECROT(IA,1,ISP,IC)
                  VECDIF(2,IC)=VEC(IA,2,ISP,IC)-VECROT(IA,2,ISP,IC)
                  VECDIF(3,IC)=VEC(IA,3,ISP,IC)-VECROT(IA,3,ISP,IC)
               ENDDO
! SPINDIFM is the difference between the spins at the original and the
! translated coordinates.
               SPINDIFM=SPIN(IA,ISP)-SPINROT(IA,ISP)
! SPINDIFP is the sum of the spins at the original and the translated
! coordinates. This quantity is needed to include the possibilty of
! spinflips.
               SPINDIFP=SPIN(IA,ISP)+SPINROT(IA,ISP)
! If TAUDIF and VECDIF are not equal (0.,0.,0.) TNTRAN=TNTRAN.AND.FALSE
! and so it remains FALSE forever ... (if only one position is not reproduced!).
! Only if all TAUDIFs are zero TNTRAN (starting with the value TRUE)
! will remain TRUE (that means we found an allowed symmetry operation).
               TNTRAN=TNTRAN.AND.((ABS(TAUDIF(1))<TINY) &
     &                            .AND.(ABS(TAUDIF(2))<TINY) &
     &                                    .AND.(ABS(TAUDIF(3))<TINY))
! and the same for VECDIF
               DO IC=1,NC
                  TNTRAN=TNTRAN.AND.((ABS(VECDIF(1,IC))<TINY) &
     &                               .AND.(ABS(VECDIF(2,IC))<TINY) &
     &                                       .AND.(ABS(VECDIF(3,IC))<TINY))
               ENDDO
! treatment of the spin lattice.
               TSPIN=.FALSE.
! when the spins are zero at the original and the translated coordinates:
               IF ((ABS(SPINDIFM)<TINY).AND.(ABS(SPINDIFP)<TINY)) TSPIN=.TRUE.
! when the spins at the original and translated coordinates are nonzero and equal,
! and a spinflip was not already needed to connect two other sites of the lattices:
               IF ((.NOT.TSPIN).AND.(ABS(SPINDIFM)<TINY).AND.(.NOT.TFLIPPED)) THEN
                  TSPIN=.TRUE.; TMAYFLIP=.FALSE.
               ENDIF
! when the spins at the original and translated coordinates are nonzero and a
! spinflip is part of the translation and the spin may still be flipped or was
! already flipped before:
               IF ((.NOT.TSPIN).AND.(ABS(SPINDIFP)<TINY).AND.(TMAYFLIP.OR.TFLIPPED)) THEN
                  TSPIN=.TRUE.; TMAYFLIP=.FALSE.; TFLIPPED=.TRUE.
               ENDIF
! so finally considering spin yields:
               TNTRAN=TNTRAN.AND.TSPIN 
    6       CONTINUE
    7    CONTINUE
! Check was successful ... :
         IF (TNTRAN) THEN
            TS=.TRUE.
! Save the detected translation vector temporarily ... :
            TRA(1)=GTRANS(1)
            TRA(2)=GTRANS(2)
            TRA(3)=GTRANS(3)
         END IF
! Now we must take the next testvector ... . Therefore restore the
! original rotated coordinates by subtracting GTRANS ... :
         ISTART=1
         DO 9 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 8 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAUROT(IA,1,ISP)-GTRANS(1)
               TAUROT(IA,2,ISP)=TAUROT(IA,2,ISP)-GTRANS(2)
               TAUROT(IA,3,ISP)=TAUROT(IA,3,ISP)-GTRANS(3)
    8       CONTINUE
    9    CONTINUE
   10 CONTINUE
      IF (TS) THEN
         GTRANS(1)=TRA(1)
         GTRANS(2)=TRA(2)
         GTRANS(3)=TRA(3)
      END IF
! Uff ... . Bye bye!
      RETURN
      END


