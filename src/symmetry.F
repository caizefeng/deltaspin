#include "symbol.inc"
MODULE msymmetry
  USE prec
  USE lattice
! TODO place COMMON block /SYMM/ in this module
! and include msymmetry everywhere where /SYMM/ is used
!

   ! Crude helper type as substitute for linked-list.
   ! This means for j > LEN, VEC(:, j) is garbage!!
   TYPE MIN_DIST_T
      REAL(q) :: VEC(3, 8)
      INTEGER :: LEN = 0
   ENDTYPE MIN_DIST_T

!************************** TYPE PRIM_CELL_T ***************************
!
! Stores information in regard to the primitive cell.
!
! P_NUM           : # of atoms in the primitive cell
! S_NUM           : # of atoms in the super cell
! CELL_NUM        : # of primitive cells in the super cell
! _test_mani
! ATOM_IDX        : An array to store the indices of the periodic images
!                   of primitive cell atoms in the super cell.
!                   ATOM_IDX(n, j) labels the n-th atom in cell j.
!                   Hence, ATOM_IDX(1, :) labels the entire primitive cell.
! MIN_DIST(i, j)  : Stores the minimum-image distance vectors from the
!                   j-th primitve cell atom to the super cell atom with
!                   index i.
!                   Since more than one vector can exist due to
!                   Symmetry, MIN_DIST_T is used to store equivalent
!                   vectors.
! LATTICE         : Stores the primitive (reciprocal) lattice vectors.
!
!***********************************************************************
   TYPE PRIM_CELL_T
      INTEGER                       :: P_NUM
      INTEGER                       :: S_NUM
      INTEGER                       :: CELL_NUM
      INTEGER, ALLOCATABLE          :: ATOM_IDX(:, :)
      TYPE(MIN_DIST_T), ALLOCATABLE :: MIN_DIST(:, :)
      TYPE(LATT)                    :: LATTICE
      LOGICAL                       :: LINIT
   ENDTYPE PRIM_CELL_T

   ! Global read-only structure to access information about the primitive cell
   TYPE(PRIM_CELL_T), PROTECTED, SAVE :: GLOB_PRIM_CELL

  CONTAINS


!******************** SUBROUTINE INISYM ********************************
!
! subroutine inisym initialises and sets up all symmetry stuff needed by
! the charge-symmetrisation and the force-symmetrisation routines ...
! it sets up: the rotation matrices in direct space (ISYMOP)
!             the rotation matrices in reciprocal space (IGRPOP)
!             and a table indexing their inverse elements (INVMAP)
!             the non-trivial translations for each rotation (GTRANS)
!             a table (ROTMAP) which shows the connections between atoms
!             the number of primitive cells within a supercell (NPCELL)
!             and their coordinates (PTRANS) within the supercell
!             magnetization direction operations for each symmetry (MAGMOM)
! additionally some further output informations will be given on IU6
!
! this routine needs as input information: the lattice vectors (A)
!             the initial atomic positions (POSION)
!             the initial velocities (VEL)
!             the number of atomic species (NTYP)
!             the number of atoms of each species (NITYP)
!             some workarrays (TAU,TAUROT,WRKROT,INDROT)
!             the dimensioning parameter NIOND (max. no. of atoms)
!             the initial atomic magnetic moments (ATOMOM)
!             the number of spin components (ISPIN)
!             and the unit number where to write informations (IU6)
!             ... and good luck (and the hope that there are no bugs)!
!             MOST IMPORTANT: it needs coordinates etc. which have a
!                  precision better than the parameter TINY set in the
!                  SYMLIB or LATTLIB routines (I recommend at least
!                  10 valid digits - if possible take more ...)!!!!!!!
!                  Otherwise the programs might not accept equivalences
!                  between atomic coordinates, velocities etc. and will
!                  not find the correct full symmetry! If you have and
!                  if you want to avoid serious problems, please edit
!                  SYMLIB and LATTLIB and increase parameter TINY ... !
!
!***********************************************************************
#ifdef oldsym
      SUBROUTINE INISYM(A,POSION,VEL,LSFOR,LSDYN,NTYP,NITYP,NIOND, &
     &           PTRANS,NRTK,NPCLL,ROTMAP,TAU,TAUROT,WRKROT,INDROT,ATOMOM,SAXIS,MAGROT,ISPIN,IU6)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      CHARACTER (4) GRPNAM
      LOGICAL LSDYN,LSFOR(3,NIOND)
      REAL(q) MAGROT(48,NIOND),SAXIS(3),ALPHA,BETA,ATOMOM_(3)
      INTEGER, POINTER ::  ROTMAP(:,:,:)

      DIMENSION A(3,3),POSION(3,NIOND),VEL(3,NIOND),NITYP(NTYP)
      DIMENSION TAU(NIOND,3),TAUROT(NIOND,3),WRKROT(3*(NIOND+2))
      DIMENSION PTRANS(NIOND+2,3),INDROT(NIOND+2),ATOMOM(3*NIOND)

      DIMENSION A1(3),A2(3),A3(3),IOPS(3,3,48),CELDIM(6),PDIM(6)
      DIMENSION COO1(3),COO2(3),B1(3),B2(3),B3(3),P1(3),P2(3),P3(3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      GRPNAM='    '

!=======================================================================
! First me must detect the lattice type and find some set of lattice
! vectors which fulfills some standard crystallographic relations:
!=======================================================================
      DO 1 I=1,3
         A1(I)=A(I,1)
         A2(I)=A(I,2)
         A3(I)=A(I,3)
    1 ENDDO
      CALL LATTYP(A1,A2,A3,IBRAV,CELDIM,IU6)

!=======================================================================
! Find number of atoms and copy atomic positions to work array TAU
!=======================================================================
      NATOMS=0
      DO 2 I=1,NTYP
         NATOMS=NATOMS+NITYP(I)
    2 ENDDO
      DO 3 IA=1,NATOMS
         TAU(IA,1)=POSION(1,IA)
         TAU(IA,2)=POSION(2,IA)
         TAU(IA,3)=POSION(3,IA)
    3 ENDDO

!=======================================================================
! Here the first highlight: Do we have a primitive cell or not? This
! becomes important for symmetrisations on the Fourier grids (i.e. for
! the symmetrisation of the charge density) if we have no pure point
! symmetry because then we need the correct phase shifts when going from
! one grid point to a symmetry related grid point - but this requires
! the knowledge of  ALL  possible 'non-primitive' translations ... !
! And it is of course also needed for all other symmetrizations ...
!=======================================================================
      CALL PRICEL(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,P1,P2,P3,PTRANS, &
     &    NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,TAUROT,INDROT,WRKROT,IU6)
      AP(1,1)=P1(1)
      AP(2,1)=P1(2)
      AP(3,1)=P1(3)
      AP(1,2)=P2(1)
      AP(2,2)=P2(2)
      AP(3,2)=P2(3)
      AP(1,3)=P3(1)
      AP(2,3)=P3(2)
      AP(3,3)=P3(3)

!=======================================================================
! The next step is to copy all coordinates into a temporary array and
! to transform them to the representation in the basis (A1,A2,A3):
!=======================================================================
      DO 4 IA=1,NATOMS
         COO1(1)=POSION(1,IA)
         COO1(2)=POSION(2,IA)
         COO1(3)=POSION(3,IA)
         CALL VECCON(COO1,COO2,1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
         TAU(IA,1)=COO2(1)
         TAU(IA,2)=COO2(2)
         TAU(IA,3)=COO2(3)
    4 ENDDO

!=======================================================================
! Now try to find out which symmetry operations we have ... :
!=======================================================================
      IF (IU6>0) &
      WRITE(IU6,'(/2A)') 'Analysis of symmetry for initial ', &
     &                                         'positions (statically):'
      CALL SETGRP(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,IBRAV,NOP, &
     &                      TAUROT,NTYP,1,NITYP,NIOND,INDROT,WRKROT,IU6)
      CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
      IF (IU6>0) &
      WRITE(IU6,'(/4A)') 'The static configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
      IF (NROTK>NROT) THEN
         CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
      END IF

!=======================================================================
! Now transform all back to the representation in our original basis (A)
! and set up a table showing the symmetry connections between the atoms
!=======================================================================
      CALL SGRCON(ISYMOP,ISYMOP,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
      CALL VECCON(GTRANS,GTRANS,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
      DO 5 IA=1,NATOMS
         TAU(IA,1)=POSION(1,IA)
         TAU(IA,2)=POSION(2,IA)
         TAU(IA,3)=POSION(3,IA)
    5 ENDDO
      IF (ASSOCIATED(ROTMAP)) THEN
         DEALLOCATE(ROTMAP)
      ENDIF
      ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
      CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)

!=======================================================================
! Now test the dynamical symmetry! Which of all the symmetry operations
! are compatible with the given initial velocities? (Final answer ...)!
! Regenerate the (possibly now invalid) map ROTMAP ...
!=======================================================================
      IF (IU6>0) &
      WRITE(IU6,'(/2A)') 'Analysis of symmetry for dynamics ', &
     &                             '(positions and initial velocities):'
      DO 6 IA=1,NATOMS
         TAU(IA,1)=VEL(1,IA)
         TAU(IA,2)=VEL(2,IA)
         TAU(IA,3)=VEL(3,IA)
    6 ENDDO
      CALL DYNSYM(TAU,ROTMAP(1,1,1),ISYMOP,GTRANS,NROTK,NROT,PTRANS, &
     &         NPCELL,1,NTYP,NIOND,NITYP,NIOND,TAUROT,WRKROT,INDROT,IU6)
      CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
      IF (IU6>0) &
      WRITE(IU6,'(/4A)') 'The dynamic configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
      IF (NROTK>NROT) THEN
         CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
      END IF
      DO 7 IA=1,NATOMS
         TAU(IA,1)=POSION(1,IA)
         TAU(IA,2)=POSION(2,IA)
         TAU(IA,3)=POSION(3,IA)
    7 ENDDO
      IF (ASSOCIATED(ROTMAP)) THEN
         DEALLOCATE(ROTMAP)
      ENDIF
      ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
      CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)

!=======================================================================
! Now test the symmetry constraints due to selective dynamics! Which of
! all the symmetry operations are compatible with the given constraints?
! (Mostly final answer)! Regenerate the (possibly now invalid) map ROTMAP
!=======================================================================
      IF (LSDYN) THEN
         IF (IU6>0) &
         WRITE(IU6,'(/2A)') 'Analysis of constrained symmetry ', &
     &                                'for selective dynamics:'
! Generate a random 'velocity field' ...
         DO IA=1,NATOMS
          DO I=1,3
            WRKROT(3*IA-3+I)=RANE()
          ENDDO
         ENDDO
! ... and symmetrize it according to the unconstrained symmetry:
         CALL VECSYM(WRKROT,ROTMAP(1,1,1),NTYP,NITYP,NIOND,TAUROT,TAU)
! Now set the fixed coordinates to zero ('break symmetry') ...
         DO IA=1,NATOMS
          DO I=1,3
            TAU(IA,I)=WRKROT(3*IA-3+I)
            IF (.NOT.LSFOR(I,IA)) TAU(IA,I)=0._q
          ENDDO
         ENDDO
! ... and check again with routine DYNSYM:
         CALL DYNSYM(TAU,ROTMAP(1,1,1),ISYMOP,GTRANS,NROTK,NROT,PTRANS, &
     &         NPCELL,1,NTYP,NIOND,NITYP,NIOND,TAUROT,WRKROT,INDROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The constrained configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
         END IF
         DO 10 IA=1,NATOMS
            TAU(IA,1)=POSION(1,IA)
            TAU(IA,2)=POSION(2,IA)
            TAU(IA,3)=POSION(3,IA)
   10    ENDDO
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
      ENDIF

!=======================================================================
! Finally test the symmetry constraints due to local magnetic moments
! all the symmetry operations are compatible with the given constraints?
! (Very final answer)! Regenerate the (possibly now invalid) map ROTMAP
!=======================================================================
      IF (ISPIN==2) THEN
         IF (IU6>0) &
         WRITE(IU6,'(/2A)') 'Analysis of magnetic symmetry:'
         CALL MAGSYM(ATOMOM,MAGROT,ROTMAP,ISYMOP,GTRANS,NROTK,NROT,PTRANS, &
     &         NPCELL,1,NTYP,NIOND,NITYP,NIOND,TAUROT,WRKROT,INDROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The magnetic configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
         END IF
         DO 11 IA=1,NATOMS
            TAU(IA,1)=POSION(1,IA)
            TAU(IA,2)=POSION(2,IA)
            TAU(IA,3)=POSION(3,IA)
   11    ENDDO
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
      ELSE IF (ISPIN==4) THEN
!
! this is a quick hack (Juergen what are we supposed to do here ???)
!
         IF (IU6>0) &
         WRITE(IU6,'(/2A)') 'Analysis of non collinear symmetry:'
         CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
       CALL EULER(SAXIS,ALPHA,BETA)
         DO 16 IA=1,NATOMS
! Rotate ATOMOM from saxis to the cartesian axes in which
! the integer rotation matrices are defined
            ATOMOM_(1)=COS(BETA)*COS(ALPHA)*ATOMOM(1+(IA-1)*3)- &
                 SIN(ALPHA)*ATOMOM(2+(IA-1)*3)+ &
                 SIN(BETA)*COS(ALPHA)*ATOMOM(3+(IA-1)*3)
            ATOMOM_(2)=COS(BETA)*SIN(ALPHA)*ATOMOM(1+(IA-1)*3)+ &
                 COS(ALPHA)*ATOMOM(2+(IA-1)*3)+ &
                 SIN(BETA)*SIN(ALPHA)*ATOMOM(3+(IA-1)*3)
            ATOMOM_(3)=-SIN(BETA)*ATOMOM(1+(IA-1)*3)+ &
                 COS(BETA)*ATOMOM(3+(IA-1)*3)
! Bring to direct coordinates
            TAU(IA,1)=ATOMOM_(1)*B1(1)+ATOMOM_(2)*B1(2)+ATOMOM_(3)*B1(3)
            TAU(IA,2)=ATOMOM_(1)*B2(1)+ATOMOM_(2)*B2(2)+ATOMOM_(3)*B2(3)
            TAU(IA,3)=ATOMOM_(1)*B3(1)+ATOMOM_(2)*B3(2)+ATOMOM_(3)*B3(3)
 16      ENDDO
         CALL DYNSYM(TAU,ROTMAP(1,1,1),ISYMOP,GTRANS,NROTK,NROT,PTRANS, &
     &        NPCELL,1,NTYP,NIOND,NITYP,NIOND,TAUROT,WRKROT,INDROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The non collinear magnetic moments have a ', &
     &        'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
     &           'its full space group is ',GRPNAM,'.'
         END IF
         DO 17 IA=1,NATOMS
            TAU(IA,1)=POSION(1,IA)
            TAU(IA,2)=POSION(2,IA)
            TAU(IA,3)=POSION(3,IA)
 17      ENDDO
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)

      ENDIF

!=======================================================================
! Finally set up the transformation matrices in reciprocal space and a
! map of the inverse elements of each (reciprocal space) group element
!======================================================================
      CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL SGRCON(ISYMOP,IGRPOP,NROTK,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL INVGRP(IGRPOP,NROTK,INVMAP)

!=======================================================================
! 'Just for fun': translate all primitive translations GTRANS into the
! generating primitive cell of the supercell (for reasons of beauty):
!=======================================================================
      IF (NPCELL>1) THEN
         DO 12 I=1,NROTK
! Cartesian coordinates of vector ...
            COO1(1)=GTRANS(1,I)*A(1,1)+GTRANS(2,I)*A(1,2)+ &
     &                                                GTRANS(3,I)*A(1,3)
            COO1(2)=GTRANS(1,I)*A(2,1)+GTRANS(2,I)*A(2,2)+ &
     &                                                GTRANS(3,I)*A(2,3)
            COO1(3)=GTRANS(1,I)*A(3,1)+GTRANS(2,I)*A(3,2)+ &
     &                                                GTRANS(3,I)*A(3,3)
! Coordinates in basis (P1,P2,P3) = generating cell ...
            CALL RECIPS(1._q,P1,P2,P3,B1,B2,B3)
            COO2(1)=COO1(1)*B1(1)+COO1(2)*B1(2)+COO1(3)*B1(3)
            COO2(2)=COO1(1)*B2(1)+COO1(2)*B2(2)+COO1(3)*B2(3)
            COO2(3)=COO1(1)*B3(1)+COO1(2)*B3(2)+COO1(3)*B3(3)
! Periodic boundary conditions (translation into the cell) ... : We use
! a shift of 5.E-10 to guarantee that 0. remains zero, also if we have
! (due to numerical errors) a small negative value (lets say -1.E-12 ?)
! which would give a coordinate of 0.999999999999. On the other hand
! values like 0.999999999999 will be treated like 1. and will be shifted
! to 0. (or -1.E-12 ...). It is somehow strange but useful ...
            COO2(1)=MOD(COO2(1)+6._q+5.E-10_q,1._q)-5.E-10_q
            COO2(2)=MOD(COO2(2)+6._q+5.E-10_q,1._q)-5.E-10_q
            COO2(3)=MOD(COO2(3)+6._q+5.E-10_q,1._q)-5.E-10_q
! Cartesian coordinates of translated vector ...
            COO1(1)=COO2(1)*P1(1)+COO2(2)*P2(1)+COO2(3)*P3(1)
            COO1(2)=COO2(1)*P1(2)+COO2(2)*P2(2)+COO2(3)*P3(2)
            COO1(3)=COO2(1)*P1(3)+COO2(2)*P2(3)+COO2(3)*P3(3)
! Coordinates in supercell basis (A) ...
            CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
            GTRANS(1,I)=COO1(1)*B1(1)+COO1(2)*B1(2)+COO1(3)*B1(3)
            GTRANS(2,I)=COO1(1)*B2(1)+COO1(2)*B2(2)+COO1(3)*B2(3)
            GTRANS(3,I)=COO1(1)*B3(1)+COO1(2)*B3(2)+COO1(3)*B3(3)
   12    ENDDO
!=======================================================================
! If we have a nonprimitive supercell update once again (and finally)
! the symmetry connection tables for the atomic position including all
! the "primitive" translations as additional symmetry operations ...
!=======================================================================
         DO 13 IA=1,NATOMS
            TAU(IA,1)=POSION(1,IA)
            TAU(IA,2)=POSION(2,IA)
            TAU(IA,3)=POSION(3,IA)
   13    ENDDO
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
      END IF

      NRTK=NROTK
      NPCLL=NPCELL

! YEAH! It was a hard job, but now we got all we need ...
      RETURN
    END SUBROUTINE INISYM
#else
      SUBROUTINE INISYM(A,POSION,VEL,LSFOR,LSDYN,NTYP,NITYP,NIOND, &
     &           PTRANS,NRTK,NPCLL,ROTMAP,TAU,TAUROT,WRKROT,INDROT,ATOMOM,SAXIS,MAGROT,ISPIN,IU6)
      USE prec
! bexternal__
      USE bexternal
! bexternal__

      IMPLICIT REAL(q) (A-H,O-Z)
      CHARACTER (4) GRPNAM
      LOGICAL LSDYN,LSFOR(3,NIOND)
      REAL(q) MAGROT(48,NIOND),SAXIS(3),ALPHA,BETA,ATOMOM_(3)
      INTEGER, POINTER ::  ROTMAP(:,:,:)

      DIMENSION A(3,3),POSION(3,NIOND),VEL(3,NIOND),NITYP(NTYP)
      DIMENSION TAU(NIOND,3),TAUROT(NIOND,3),WRKROT(3*(NIOND+2))
      DIMENSION PTRANS(NIOND+2,3),INDROT(NIOND+2),ATOMOM(3*NIOND)
      DIMENSION TMP(NIOND,3,5),VEC(NIOND,3,3),SPIN(NIOND)

      DIMENSION A1(3),A2(3),A3(3),IOPS(3,3,48),CELDIM(6),PDIM(6)
      DIMENSION COO1(3),COO2(3),B1(3),B2(3),B3(3),P1(3),P2(3),P3(3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      GRPNAM='    '

!=======================================================================
! First me must detect the lattice type and find some set of lattice
! vectors which fulfills some standard crystallographic relations:
!=======================================================================
      DO 1 I=1,3
         A1(I)=A(I,1)
         A2(I)=A(I,2)
         A3(I)=A(I,3)
    1 ENDDO
      CALL LATTYP(A1,A2,A3,IBRAV,CELDIM,IU6)
!=======================================================================
! Find number of atoms and copy atomic positions to work array TMP(:,:,1)
!=======================================================================
      NATOMS=0
      DO 2 I=1,NTYP
         NATOMS=NATOMS+NITYP(I)
    2 ENDDO

      TMP=0._q
      DO 3 IA=1,NATOMS
         TMP(IA,1,1)=POSION(1,IA)
         TMP(IA,2,1)=POSION(2,IA)
         TMP(IA,3,1)=POSION(3,IA)
    3 ENDDO
!=======================================================================
! copy all coordinates into a temporary array and transform them to the
! representation in the basis (A1,A2,A3): store in TMP(:,:,2)
!=======================================================================
      DO 4 IA=1,NATOMS
         COO1(1)=POSION(1,IA)
         COO1(2)=POSION(2,IA)
         COO1(3)=POSION(3,IA)
         CALL VECCON(COO1,COO2,1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
         TMP(IA,1,2)=COO2(1)
         TMP(IA,2,2)=COO2(2)
         TMP(IA,3,2)=COO2(3)
    4 ENDDO

!=======================================================================
! Here the first highlight: Do we have a primitive cell or not? This
! becomes important for symmetrisations on the Fourier grids (i.e. for
! the symmetrisation of the charge density) if we have no pure point
! symmetry because then we need the correct phase shifts when going from
! one grid point to a symmetry related grid point - but this requires
! the knowledge of  ALL  possible 'non-primitive' translations ... !
! And it is of course also needed for all other symmetrizations ...
!=======================================================================
      IF (IU6>0) &
      WRITE(IU6,'(/2(/A))') &
     &   'Analysis of symmetry for initial positions (statically):', &
     &   '====================================================================='

      TAU=TMP(:,:,1); VEC=0._q; SPIN=0._q

      CALL PRICELV(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,VEC,SPIN,P1,P2,P3,PTRANS, &
     &    NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,3,TAUROT,INDROT,WRKROT,IU6)

      AP(1,1)=P1(1)
      AP(2,1)=P1(2)
      AP(3,1)=P1(3)
      AP(1,2)=P2(1)
      AP(2,2)=P2(2)
      AP(3,2)=P2(3)
      AP(1,3)=P3(1)
      AP(2,3)=P3(2)
      AP(3,3)=P3(3)

!=======================================================================
! Now try to find out which symmetry operations we have ... :
!=======================================================================
      TAU=TMP(:,:,2); VEC=0._q; SPIN=0._q

      CALL SETGRPV(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,VEC,SPIN,IBRAV,NOP, &
     &                      TAUROT,NTYP,1,NITYP,NIOND,3,INDROT,WRKROT,IU6)

      CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
      IF (IU6>0) &
      WRITE(IU6,'(/4A)') 'The static configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
      IF (NROTK>NROT) THEN
         CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
      END IF

!=======================================================================
! Now test the dynamical symmetry! Which of all the symmetry operations
! are compatible with the given initial velocities? (Final answer ...)!
! Regenerate the (possibly now invalid) map ROTMAP ...
!=======================================================================
      IF (IU6>0) &
      WRITE(IU6,'(/2(/A))') &
     &   'Analysis of symmetry for dynamics (positions and initial velocities):', &
     &   '====================================================================='

! store the velocities in TMP(:,:,3)
      DO 6 IA=1,NATOMS
         TMP(IA,1,3)=VEL(1,IA)
         TMP(IA,2,3)=VEL(2,IA)
         TMP(IA,3,3)=VEL(3,IA)
    6 ENDDO

      TAU=TMP(:,:,1)
      VEC(:,:,1)=TMP(:,:,3)

      CALL PRICELV(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,VEC,SPIN,P1,P2,P3,PTRANS, &
     &    NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,3,TAUROT,INDROT,WRKROT,IU6)

      AP(1,1)=P1(1)
      AP(2,1)=P1(2)
      AP(3,1)=P1(3)
      AP(1,2)=P2(1)
      AP(2,2)=P2(2)
      AP(3,2)=P2(3)
      AP(1,3)=P3(1)
      AP(2,3)=P3(2)
      AP(3,3)=P3(3)

      TAU=TMP(:,:,2)
      VEC(:,:,1)=TMP(:,:,3)
      DO IA=1,NATOMS
         CALL VECCON(VEC(IA,1:3,1),VEC(IA,1:3,1),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
      ENDDO

      CALL SETGRPV(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,VEC,SPIN,IBRAV,NOP, &
     &                      TAUROT,NTYP,1,NITYP,NIOND,3,INDROT,WRKROT,IU6)
      CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
      IF (IU6>0) &
      WRITE(IU6,'(/4A)') 'The dynamic configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
      IF (NROTK>NROT) THEN
         CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
      END IF

      CALL SGRCON(ISYMOP,ISYMOP,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
      CALL VECCON(GTRANS,GTRANS,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))

      TAU=TMP(:,:,1)
      IF (ASSOCIATED(ROTMAP)) THEN
         DEALLOCATE(ROTMAP)
      ENDIF
      ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
      CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)

!=======================================================================
! Now test the symmetry constraints due to selective dynamics! Which of
! all the symmetry operations are compatible with the given constraints?
! (Mostly final answer)! Regenerate the (possibly now invalid) map ROTMAP
!=======================================================================
      IF (LSDYN) THEN
         IF (IU6>0) &
         WRITE(IU6,'(/2(/A))') &
     &      'Analysis of constrained symmetry for selective dynamics:', &
     &      '====================================================================='

! Generate a random 'velocity field' ...
         DO IA=1,NATOMS
          DO I=1,3
            WRKROT(3*IA-3+I)=RANE()
          ENDDO
         ENDDO
! ... and symmetrize it according to the unconstrained symmetry:
         CALL VECSYM(WRKROT,ROTMAP(1,1,1),NTYP,NITYP,NIOND,TAUROT,TAU)
! Now set the fixed coordinates to zero ('break symmetry') ...
! and store the constrained velocity field in TMP(:,:,4)
         DO IA=1,NATOMS
          DO I=1,3
            TMP(IA,I,4)=WRKROT(3*IA-3+I)
            IF (.NOT.LSFOR(I,IA)) TMP(IA,I,4)=0._q
          ENDDO
         ENDDO

         TAU=TMP(:,:,1)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4)

         CALL PRICELV(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,VEC,SPIN,P1,P2,P3,PTRANS, &
     &       NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,3,TAUROT,INDROT,WRKROT,IU6)

         AP(1,1)=P1(1)
         AP(2,1)=P1(2)
         AP(3,1)=P1(3)
         AP(1,2)=P2(1)
         AP(2,2)=P2(2)
         AP(3,2)=P2(3)
         AP(1,3)=P3(1)
         AP(2,3)=P3(2)
         AP(3,3)=P3(3)

         TAU=TMP(:,:,2)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4)
         DO IA=1,NATOMS
            CALL VECCON(VEC(IA,1:3,1),VEC(IA,1:3,1),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
            CALL VECCON(VEC(IA,1:3,2),VEC(IA,1:3,2),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
         ENDDO

         CALL SETGRPV(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,VEC,SPIN,IBRAV,NOP, &
     &                         TAUROT,NTYP,1,NITYP,NIOND,3,INDROT,WRKROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The constrained configuration has the ', &
     &                                         'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
     &                                'its full space group is ',GRPNAM,'.'
         END IF

         CALL SGRCON(ISYMOP,ISYMOP,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
         CALL VECCON(GTRANS,GTRANS,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))

         TAU=TMP(:,:,1)
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                      NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
      ENDIF

!=======================================================================
! Finally test the symmetry constraints due to local magnetic moments
! all the symmetry operations are compatible with the given constraints?
! (Very final answer)! Regenerate the (possibly now invalid) map ROTMAP
!=======================================================================
      IF (ISPIN==2) THEN
         IF (IU6>0) &
         WRITE(IU6,'(/2(/A))') &
     &       'Analysis of structural, dynamic, and magnetic symmetry:', &
     &       '====================================================================='

         CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
! bexternal__
         IF (LBEXTERNAL()) THEN
            DO IA=1,NATOMS
               ATOMOM(IA)=ATOMOM(IA)+BEXT(1)*100
            ENDDO
         ENDIF
! bexternel__
! Moments along cartesian z-axis, handling like non-collinear moments (see also
! below); bring them to lattice coordinates and store in TMP(:,:,5)
         DO IA=1,NATOMS
            TMP(IA,3,5)=ATOMOM(IA)
         ENDDO

         TAU=TMP(:,:,1)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4)
         SPIN=TMP(:,3,5)

         CALL PRICELV(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,VEC,SPIN,P1,P2,P3,PTRANS, &
        &    NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,3,TAUROT,INDROT,WRKROT,IU6)

         AP(1,1)=P1(1)
         AP(2,1)=P1(2)
         AP(3,1)=P1(3)
         AP(1,2)=P2(1)
         AP(2,2)=P2(2)
         AP(3,2)=P2(3)
         AP(1,3)=P3(1)
         AP(2,3)=P3(2)
         AP(3,3)=P3(3)

         TAU=TMP(:,:,2)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4)
         DO IA=1,NATOMS
            CALL VECCON(VEC(IA,1:3,1),VEC(IA,1:3,1),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
            CALL VECCON(VEC(IA,1:3,2),VEC(IA,1:3,2),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
         ENDDO
         SPIN=TMP(:,3,5)

         CALL SETGRPV(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,VEC,SPIN,IBRAV,NOP, &
     &                         TAUROT,NTYP,1,NITYP,NIOND,3,INDROT,WRKROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The magnetic configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
         END IF

         CALL SGRCON(ISYMOP,ISYMOP,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
         CALL VECCON(GTRANS,GTRANS,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))

         TAU=TMP(:,:,1)
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)

         NROTKP=NROTK; NROTP=NROT; NPCELLP=NPCELL

         CALL MAGSYM(ATOMOM,MAGROT,ROTMAP,ISYMOP,GTRANS,NROTKP,NROTP,PTRANS, &
     &         NPCELL,1,NTYP,NIOND,NITYP,NIOND,TAUROT,WRKROT,INDROT,-1)

         IF ((NROTKP/=NROTK).OR.(NROTP/=NROT).AND.(NPCELLP/=NPCELL)) THEN
            WRITE(*,*) 'INISYM: ERROR: Unable to resolve symmetry ', &
     &                             'of colinear magnetic degrees of freedom.'
            STOP
         ENDIF
! bexternal__
         IF (LBEXTERNAL()) THEN
            DO IA=1,NATOMS
               ATOMOM(IA)=ATOMOM(IA)-BEXT(1)*100
            ENDDO
         ENDIF
! bexternal__
      ELSE IF (ISPIN==4) THEN
         IF (IU6>0) &
         WRITE(IU6,'(/2(/A))') &
     &      'Analysis of structural, dynamic, and magnetic symmetry:', &
     &      '====================================================================='

         CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
         CALL EULER(SAXIS,ALPHA,BETA)
! bexternal__
         IF (LBEXTERNAL()) THEN
            DO IA=1,NATOMS
               ATOMOM(1+(IA-1)*3)=ATOMOM(1+(IA-1)*3)+BEXT(1)*100
               ATOMOM(2+(IA-1)*3)=ATOMOM(2+(IA-1)*3)+BEXT(2)*100
               ATOMOM(3+(IA-1)*3)=ATOMOM(3+(IA-1)*3)+BEXT(3)*100
            ENDDO
         ENDIF
! bexternal__
         DO IA=1,NATOMS
! Rotate ATOMOM from saxis to the cartesian axes in which
! the integer rotation matrices are defined
            ATOMOM_(1)=COS(BETA)*COS(ALPHA)*ATOMOM(1+(IA-1)*3)- &
                 SIN(ALPHA)*ATOMOM(2+(IA-1)*3)+ &
                 SIN(BETA)*COS(ALPHA)*ATOMOM(3+(IA-1)*3)
            ATOMOM_(2)=COS(BETA)*SIN(ALPHA)*ATOMOM(1+(IA-1)*3)+ &
                 COS(ALPHA)*ATOMOM(2+(IA-1)*3)+ &
                 SIN(BETA)*SIN(ALPHA)*ATOMOM(3+(IA-1)*3)
            ATOMOM_(3)=-SIN(BETA)*ATOMOM(1+(IA-1)*3)+ &
                 COS(BETA)*ATOMOM(3+(IA-1)*3)
! Bring to direct coordinates and store in TMP(:,:,5)
            TMP(IA,1,5)=ATOMOM_(1)*B1(1)+ATOMOM_(2)*B1(2)+ATOMOM_(3)*B1(3)
            TMP(IA,2,5)=ATOMOM_(1)*B2(1)+ATOMOM_(2)*B2(2)+ATOMOM_(3)*B2(3)
            TMP(IA,3,5)=ATOMOM_(1)*B3(1)+ATOMOM_(2)*B3(2)+ATOMOM_(3)*B3(3)
         ENDDO

         TAU=TMP(:,:,1)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4); VEC(:,:,3)=TMP(:,:,5)

         CALL PRICELV(IBRAV,CELDIM,A(1,1),A(1,2),A(1,3),TAU,VEC,SPIN,P1,P2,P3,PTRANS, &
        &    NPCELL,IPTYP,PDIM,NTYP,1,NITYP,NIOND,3,TAUROT,INDROT,WRKROT,IU6)

         AP(1,1)=P1(1)
         AP(2,1)=P1(2)
         AP(3,1)=P1(3)
         AP(1,2)=P2(1)
         AP(2,2)=P2(2)
         AP(3,2)=P2(3)
         AP(1,3)=P3(1)
         AP(2,3)=P3(2)
         AP(3,3)=P3(3)

         TAU=TMP(:,:,2)
         VEC(:,:,1)=TMP(:,:,3); VEC(:,:,2)=TMP(:,:,4); VEC(:,:,3)=TMP(:,:,5)
         DO IA=1,NATOMS
            CALL VECCON(VEC(IA,1:3,1),VEC(IA,1:3,1),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
            CALL VECCON(VEC(IA,1:3,2),VEC(IA,1:3,2),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
            CALL VECCON(VEC(IA,1:3,3),VEC(IA,1:3,3),1,A(1,1),A(1,2),A(1,3),A1,A2,A3)
         ENDDO

         CALL SETGRPV(ISYMOP,IOPS,GTRANS,NROT,NROTK,TAU,VEC,SPIN,IBRAV,NOP, &
     &                         TAUROT,NTYP,1,NITYP,NIOND,3,INDROT,WRKROT,IU6)
         CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') 'The (compounded) configuration has the ', &
        &                                      'point symmetry ',GRPNAM,'.'
         IF (NROTK>NROT) THEN
            CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
            IF (IU6>0) &
            WRITE(IU6,*) 'The point group associated with ', &
        &                             'its full space group is ',GRPNAM,'.'
         END IF

         CALL SGRCON(ISYMOP,ISYMOP,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))
         CALL VECCON(GTRANS,GTRANS,NROTK,A1,A2,A3,A(1,1),A(1,2),A(1,3))

         TAU=TMP(:,:,1)
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
        &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
! bexternal__
         IF (LBEXTERNAL()) THEN
            DO IA=1,NATOMS
               ATOMOM(1+(IA-1)*3)=ATOMOM(1+(IA-1)*3)-BEXT(1)*100
               ATOMOM(2+(IA-1)*3)=ATOMOM(2+(IA-1)*3)-BEXT(2)*100
               ATOMOM(3+(IA-1)*3)=ATOMOM(3+(IA-1)*3)-BEXT(3)*100
            ENDDO
         ENDIF
! bexternal__
      ENDIF

!=======================================================================
! Finally set up the transformation matrices in reciprocal space and a
! map of the inverse elements of each (reciprocal space) group element
!======================================================================
      CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL SGRCON(ISYMOP,IGRPOP,NROTK,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL INVGRP(IGRPOP,NROTK,INVMAP)

!=======================================================================
! 'Just for fun': translate all primitive translations GTRANS into the
! generating primitive cell of the supercell (for reasons of beauty):
!=======================================================================
      IF (NPCELL>1) THEN
         DO 12 I=1,NROTK
! Cartesian coordinates of vector ...
            COO1(1)=GTRANS(1,I)*A(1,1)+GTRANS(2,I)*A(1,2)+ &
     &                                                GTRANS(3,I)*A(1,3)
            COO1(2)=GTRANS(1,I)*A(2,1)+GTRANS(2,I)*A(2,2)+ &
     &                                                GTRANS(3,I)*A(2,3)
            COO1(3)=GTRANS(1,I)*A(3,1)+GTRANS(2,I)*A(3,2)+ &
     &                                                GTRANS(3,I)*A(3,3)
! Coordinates in basis (P1,P2,P3) = generating cell ...
            CALL RECIPS(1._q,P1,P2,P3,B1,B2,B3)
            COO2(1)=COO1(1)*B1(1)+COO1(2)*B1(2)+COO1(3)*B1(3)
            COO2(2)=COO1(1)*B2(1)+COO1(2)*B2(2)+COO1(3)*B2(3)
            COO2(3)=COO1(1)*B3(1)+COO1(2)*B3(2)+COO1(3)*B3(3)
! Periodic boundary conditions (translation into the cell) ... : We use
! a shift of 5.E-10 to guarantee that 0. remains zero, also if we have
! (due to numerical errors) a small negative value (lets say -1.E-12 ?)
! which would give a coordinate of 0.999999999999. On the other hand
! values like 0.999999999999 will be treated like 1. and will be shifted
! to 0. (or -1.E-12 ...). It is somehow strange but useful ...
            COO2(1)=MOD(COO2(1)+6._q+5.E-10_q,1._q)-5.E-10_q
            COO2(2)=MOD(COO2(2)+6._q+5.E-10_q,1._q)-5.E-10_q
            COO2(3)=MOD(COO2(3)+6._q+5.E-10_q,1._q)-5.E-10_q
! Cartesian coordinates of translated vector ...
            COO1(1)=COO2(1)*P1(1)+COO2(2)*P2(1)+COO2(3)*P3(1)
            COO1(2)=COO2(1)*P1(2)+COO2(2)*P2(2)+COO2(3)*P3(2)
            COO1(3)=COO2(1)*P1(3)+COO2(2)*P2(3)+COO2(3)*P3(3)
! Coordinates in supercell basis (A) ...
            CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
            GTRANS(1,I)=COO1(1)*B1(1)+COO1(2)*B1(2)+COO1(3)*B1(3)
            GTRANS(2,I)=COO1(1)*B2(1)+COO1(2)*B2(2)+COO1(3)*B2(3)
            GTRANS(3,I)=COO1(1)*B3(1)+COO1(2)*B3(2)+COO1(3)*B3(3)
   12    ENDDO
!=======================================================================
! If we have a nonprimitive supercell update once again (and finally)
! the symmetry connection tables for the atomic position including all
! the "primitive" translations as additional symmetry operations ...
!=======================================================================
         TAU=TMP(:,:,1)
         IF (ASSOCIATED(ROTMAP)) THEN
            DEALLOCATE(ROTMAP)
         ENDIF
         ALLOCATE(ROTMAP(NIOND,NROTK,NPCELL))
         CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,PTRANS,NPCELL,NIOND,1, &
     &                                   NTYP,NIOND,NITYP,ROTMAP(1,1,1),TAUROT)
      END IF

      NRTK=NROTK
      NPCLL=NPCELL

! YEAH! It was a hard job, but now we got all we need ...
      IF (IU6>0) WRITE(IU6,50) NROTK,NROT,NPCELL
   50 FORMAT(//' Subroutine INISYM returns: Found ',I2, &
     &         ' space group operations'/,' (whereof ',I2, &
     &         ' operations are pure point group operations),'/, &
     &         ' and found ',I5,' ''primitive'' translations'/)

      RETURN
    END SUBROUTINE INISYM
#endif

!************************* SUBROUTINE WRTSYM ***************************
!
!***********************************************************************

      SUBROUTINE WRTSYM(NATOMS,NIOND,PTRANS,ROTMAP,MAGROT,ISPIN,IU6)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      INTEGER, POINTER :: ROTMAP(:,:,:)
      REAL(q), POINTER :: MAGROT(:,:)
      DIMENSION PTRANS(NIOND+2,3),INDROT(NIOND+2)
      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      IF (IU6>0) THEN
         DO I=1,NROTK
            WRITE(IU6,'(//X,A,I4)') 'irot  :',I
            WRITE(IU6,'(X,A)') '--------------------------------------------------------------------'
            WRITE(IU6,'(X,A,3I4/,8X,3I4/,8X,3I4)') 'isymop:',ISYMOP(:,:,I)
            WRITE(IU6,'(/X,A,3F14.7)') 'gtrans:',GTRANS(:,I)
            DO J=1,NPCELL
!              WRITE(IU6,'(/A,I4)') 'NP=',J
               WRITE(IU6,'(/X,A,3F14.7)') 'ptrans:',PTRANS(J,:)
               IF (ISPIN==2) WRITE(IU6,'(/X,A,F14.7)') 'magrot:',MAGROT(I,J)
               WRITE(IU6,*)
               DO IA=1,NATOMS
                  INDROT(ROTMAP(IA,I,J))=IA
               ENDDO
               WRITE(IU6,'(X,A)') 'rotmap:'
               DO IA=1,NATOMS,5
!                 WRITE(IU6,'(A)',ADVANCE='No') "("
                  DO K=IA,MIN(IA+4,NATOMS)
!                    WRITE(IU6,'("(",I4,"<-",I4,")",2X)',ADVANCE='No')  K,ROTMAP(K,I,J)
                     WRITE(IU6,'(X,"(",I4,"->",I4,")",X)',ADVANCE='No')  K,INDROT(K)
                  ENDDO
                  WRITE(IU6,*)
               ENDDO
            ENDDO
         ENDDO
      ENDIF
      RETURN
      END SUBROUTINE WRTSYM

!************************* SUBROUTINE NOSYMM ***************************
!
! subroutine nosymm should be called when no usage of any symmetry
! will be desired (initialise all stuff for point group C_1 ...)
! it sets up: the rotation matrices in direct space (ISYMOP)
!             the rotation matrices in reciprocal space (IGRPOP)
!             and a table indexing their inverse elements (INVMAP)
!             the non-trivial translations for each rotation (GTRANS)
!             a table (ROTMAP) which shows the connections between atoms
!             the number of primitive cells within a supercell (NPCELL)
!             and their coordinates (PTRANS) within the supercell
!     all will be initialised as if we had no symmetry (1 symmetry
!     operation = unity operator in both spaces, no non-trivial
!     translations, cell is a primitive cell ...)
! the routine needs the number of atomic species (NTYP)
!             the number of atoms of each species (NITYP)
!             the dimensioning parameter NIOND (max. no. of atoms)
!             and the unit number where to write informations (IU6)
!
!***********************************************************************

      SUBROUTINE NOSYMM(A,NTYP,NITYP,NIOND,PTRANS,NRTK,NPCLL,ROTMAP,MAGROT,ISPIN,IU6)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) MAGROT(48,NIOND)
      INTEGER, POINTER :: ROTMAP(:,:,:)
      DIMENSION PTRANS(NIOND+2,3),NITYP(NTYP),A(3,3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      IF (IU6>0) THEN
      WRITE(IU6,'(/,2A,/2A,/)') 'IMPORTANT INFORMATION: All ', &
     &                  'symmetrisations will be switched off!', &
     &                  'NOSYMM: (Re-)initialisation of all symmetry ', &
     &                                      'stuff for point group C_1.'
      ENDIF
      NROT=1
      NROTK=1
      ISYMOP(1,1,1)=1
      ISYMOP(2,2,1)=1
      ISYMOP(3,3,1)=1
      ISYMOP(1,2,1)=0
      ISYMOP(2,1,1)=0
      ISYMOP(1,3,1)=0
      ISYMOP(3,1,1)=0
      ISYMOP(2,3,1)=0
      ISYMOP(3,2,1)=0
      IGRPOP(1,1,1)=1
      IGRPOP(2,2,1)=1
      IGRPOP(3,3,1)=1
      IGRPOP(1,2,1)=0
      IGRPOP(2,1,1)=0
      IGRPOP(1,3,1)=0
      IGRPOP(3,1,1)=0
      IGRPOP(2,3,1)=0
      IGRPOP(3,2,1)=0
      GTRANS(1,1)=0._q
      GTRANS(2,1)=0._q
      GTRANS(3,1)=0._q
      INVMAP(1)=1
      NATOMS=0
      DO I=1,NTYP
         NATOMS=NATOMS+NITYP(I)
      ENDDO
      DO I=1,NATOMS
         ROTMAP(I,1,1)=I
      ENDDO
      IF (ISPIN==2) THEN
         DO I=1,NATOMS
            MAGROT(1,I)=1._q
         ENDDO
      ENDIF
      NPCELL=1
      PTRANS(1,1)=0._q
      PTRANS(1,2)=0._q
      PTRANS(1,3)=0._q
      AP(1,1)=A(1,1)
      AP(2,1)=A(2,1)
      AP(3,1)=A(3,1)
      AP(1,2)=A(1,2)
      AP(2,2)=A(2,2)
      AP(3,2)=A(3,2)
      AP(1,3)=A(1,3)
      AP(2,3)=A(2,3)
      AP(3,3)=A(3,3)
      NRTK=NROTK
      NPCLL=NPCELL
      RETURN
    END SUBROUTINE NOSYMM

!****************** SUBROUTINE SETUP_PRIMITIVE_CELL ********************
!
! This subroutine sets up the GLOB_PRIM_CELL structure.
! Additionally, information regarding the primitive cell is written
! out to the XML file.
!
! A      : Super-cell real-space lattice vectors.
! POSION : Ionic positions in direct coordinates of A.
!
!***********************************************************************

   SUBROUTINE SETUP_PRIMITIVE_CELL(A, POSION, IU0)
      USE sym_prec
      USE vaspxml
      USE lattice
      IMPLICIT NONE

      REAL(q), INTENT(IN)  :: A(3,3)
      REAL(q), INTENT(IN)  :: POSION(:, :)
      INTEGER              :: IU0

      ! local variables
      INTEGER              :: S_NUM, P_NUM
      INTEGER              :: S_IDX, P_IDX_1, P_IDX_2, CELL_IDX
      REAL(q), ALLOCATABLE :: P_POS(:, :)
      REAL(q)              :: R(3), DR(3)
      REAL(q)              :: ORIG_ATOM(3)
      REAL(q)              :: MIN_SQR, CUR_SQR
      INTEGER              :: MIN_IDX
      INTEGER              :: X, Y, Z

      ! XML stuff
      INTEGER, ALLOCATABLE :: XML_MAP(:, :)
      REAL(q), ALLOCATABLE :: XML_POS(:, :)

      ! Symmetry stuff
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP,OMEGA
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
      GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      ! on succesful exit we'll set this to .TRUE.
      GLOB_PRIM_CELL%LINIT=.FALSE.

      S_NUM = SIZE(POSION, 2)
      P_NUM = S_NUM / NPCELL

      ! sanity check
      IF (P_NUM * NPCELL /= S_NUM) THEN
         IF (IU0 >= 0) THEN
            WRITE(IU0, *) 'internal error in VASP: SETUP_PRIMITIVE_CELL, S_NUM not divisible by NPCELL', S_NUM, NPCELL
            WRITE(IU0, *) ' try to change SYMPREC in the INCAR file, or better, improve precision of the POSCAR file'
         ENDIF
         STOP
      ENDIF

      ! Set some variables for prim cell.
      GLOB_PRIM_CELL%S_NUM = SIZE(POSION, 2)
      GLOB_PRIM_CELL%P_NUM = S_NUM / NPCELL
      GLOB_PRIM_CELL%CELL_NUM = NPCELL

      IF (ALLOCATED(GLOB_PRIM_CELL%ATOM_IDX)) DEALLOCATE(GLOB_PRIM_CELL%ATOM_IDX)

      ALLOCATE(GLOB_PRIM_CELL%ATOM_IDX(NPCELL, P_NUM), P_POS(3, S_NUM))
      GLOB_PRIM_CELL%ATOM_IDX = 0

      ! Setup the primitive cell lattice.
      GLOB_PRIM_CELL%LATTICE%A = AP
      CALL LATTIC(GLOB_PRIM_CELL%LATTICE)

      ! Get direct coordinates w.r.t. the primitive cell.
      P_POS = MOD(POSION + 1000.5_q, 1._q) - 0.5_q
      P_POS = MATMUL(A,                                   P_POS)
      P_POS = MATMUL(TRANSPOSE(GLOB_PRIM_CELL%LATTICE%B), P_POS)

      !  ________________________________________________
      ! |                                                |
      ! |      Find 1st primitive cell atom and all      |
      ! |      it's copies in the super cell.            |
      ! |________________________________________________|

      CELL_IDX = 1
      GLOB_PRIM_CELL%ATOM_IDX(1, 1) = 1

      DO S_IDX = 2, S_NUM
         DR = MOD(P_POS(:, S_IDX) - P_POS(:, 1) + 1000.5_q, 1._q) - 0.5_q

         IF (ALL(ABS(DR) < TINY)) THEN
            CELL_IDX = CELL_IDX + 1
            GLOB_PRIM_CELL%ATOM_IDX(CELL_IDX, 1) = S_IDX
         ENDIF
      ENDDO

      ! Sanity check.
      IF (CELL_IDX /= NPCELL) THEN
         IF (IU0 >= 0) THEN
            WRITE(IU0, *) 'internal error in VASP: SETUP_PRIMITIVE_CELL, ' // &
            'Could not associate all atoms to their respective cells.'
            WRITE(IU0, *) ' try to change SYMPREC in the INCAR file, or better improve precision of the POSCAR file'
         ENDIF
         RETURN
      ENDIF

      !  ___________________________________________________
      ! |                                                   |
      ! |      Find more primitive cell atoms and link      |
      ! |      them with their corresponding cells.         |
      ! |___________________________________________________|

      ORIG_ATOM = POSION(:, GLOB_PRIM_CELL%ATOM_IDX(1, 1))
      
      ! Loop over any remaining primitive cell atoms.
      DO P_IDX_1 = 2, P_NUM
         
         MIN_SQR = HUGE(1._q)
         MIN_IDX = 0

         ! Find the next primitive atom with the shortest distance to atom 1.
         ions_1: DO S_IDX = 2, S_NUM
            ! Check if this atom has alread been added.
            DO P_IDX_2 = 1, P_NUM
               DO CELL_IDX = 1, NPCELL
                  IF (S_IDX == GLOB_PRIM_CELL%ATOM_IDX(CELL_IDX, P_IDX_2)) CYCLE ions_1
               ENDDO
            ENDDO

            ! Find the minimum-image distance to primitive atom 1 in the super cell.
            DO X = -1, 1
            DO Y = -1, 1
            DO Z = -1, 1
               DR = MATMUL(A, POSION(:, S_IDX) - ORIG_ATOM + [X, Y, Z])
               CUR_SQR = DOT_PRODUCT(DR, DR)
               IF (CUR_SQR < MIN_SQR) THEN
                  MIN_SQR = CUR_SQR
                  MIN_IDX = S_IDX
                  R = POSION(:, S_IDX) - ORIG_ATOM + [X, Y, Z] ! Save vector for later.
               ENDIF
            ENDDO
            ENDDO
            ENDDO
         ENDDO ions_1

         ! Sanity check.
         IF (MIN_IDX == 0) THEN
            IF (IU0 >= 0) THEN
               WRITE(IU0, '(A, I4, A)') 'internal error in VASP: SETUP_PRIMITIVE_CELL, ' // &
               'Primitive cell atom ', P_IDX_1, ' could not be found.'
               WRITE(IU0, *) ' try to change SYMPREC in the INCAR file, or better improve precision of the POSCAR file'
            ENDIF
            RETURN
         ENDIF

         ! Add the atom with the shortest distance to cell 1.
         GLOB_PRIM_CELL%ATOM_IDX(1, P_IDX_1) = MIN_IDX

         ! Find the corresponding copy of that atom for each cell.
         ions_2: DO S_IDX = 2, S_NUM
            ! Check if this atom has alread been added.
            DO P_IDX_2 = 1, P_NUM
               DO CELL_IDX = 1, NPCELL
                  IF (S_IDX == GLOB_PRIM_CELL%ATOM_IDX(CELL_IDX, P_IDX_2)) CYCLE ions_2
               ENDDO
            ENDDO

            ! Compare the minimum-image distance vector to the reference cell 1.
            DO CELL_IDX = 1, NPCELL
               DO X = -1, 1
               DO Y = -1, 1
               DO Z = -1, 1
                  DR = POSION(:, S_IDX) - POSION(:, GLOB_PRIM_CELL%ATOM_IDX(CELL_IDX, 1)) + [X, Y, Z]
                  IF (ALL(ABS(DR - R) < TINY)) THEN
                     ! Correct cell found! Now add the atom.
                     GLOB_PRIM_CELL%ATOM_IDX(CELL_IDX, P_IDX_1) = S_IDX
                     CYCLE ions_2
                  ENDIF
               ENDDO
               ENDDO
               ENDDO
            ENDDO
         ENDDO ions_2
      ENDDO

      ! sanity check
      IF (ANY(GLOB_PRIM_CELL%ATOM_IDX == 0)) THEN
         IF (IU0 >= 0) THEN
            WRITE(IU0, *) 'internal error in VASP: SETUP_PRIMITIVE_CELL, Not all atoms could be associated with cells.'
            WRITE(IU0, *) ' try to change SYMPREC in the INCAR file, or better improve precision of the POSCAR file'
         ENDIF
         RETURN
      ENDIF

      ! Write primitive positions to XML.
      ALLOCATE(XML_POS(3, P_NUM))
      XML_POS(:, :) = MOD(P_POS(:, GLOB_PRIM_CELL%ATOM_IDX(1, :)) + 1000.5_q, 1._q) - 0.5_q
      CALL XML_TAG("structure", "primitive_cell")
      CALL XML_CRYSTAL(GLOB_PRIM_CELL%LATTICE%A, GLOB_PRIM_CELL%LATTICE%B, GLOB_PRIM_CELL%LATTICE%OMEGA)
      CALL XML_POSITIONS(P_NUM, XML_POS)
      CALL XML_CLOSE_TAG("structure")
      DEALLOCATE(XML_POS)

      ! Write primitive indices to XML.
      ALLOCATE(XML_MAP(1, P_NUM))
      CALL XML_VECARRAY("primitive_index")
      XML_MAP(1, :) = GLOB_PRIM_CELL%ATOM_IDX(1, :)
      CALL XML_ARRAY_INT(XML_MAP)
      CALL XML_CLOSE_TAG
      DEALLOCATE(XML_MAP)

      CALL SETUP_MIN_DIST(A, POSION)

      ! successfully set up GLOB_PRIM_CELL
      GLOB_PRIM_CELL%LINIT=.TRUE.

      RETURN
   ENDSUBROUTINE SETUP_PRIMITIVE_CELL


!********************* SUBROUTINE SETUP_MIN_DIST ***********************
!
! This subroutine sets up the minimum-image-distance array stored in
! GLOB_PRIM_CELL. It is internally called by SETUP_PRIMITIVE_CELL.
!
! A      : Super-cell real-space lattice vectors.
! POSION : Ionic positions in direct coordinates of A.
!
!***********************************************************************

   SUBROUTINE SETUP_MIN_DIST(A, POSION)
      USE sym_prec
      IMPLICIT NONE

      REAL(q), INTENT(IN) :: A(3,3)
      REAL(q), INTENT(IN) :: POSION(:, :)

      ! locals
      INTEGER     :: I, J, K
      INTEGER     :: P_IDX, S_IDX
      REAL(q)     :: DIFF_LEN, DIFF_LEN_OLD, DIFF_VEC(3)

      ! Make sure old MIN_DIST is deallocated correctly.
      IF (ALLOCATED(GLOB_PRIM_CELL%MIN_DIST)) DEALLOCATE(GLOB_PRIM_CELL%MIN_DIST)

      ! Allocate MIN_DIST to hold all shortest distance vectors.
      ! Dublicates are stored in a nested list.
      ALLOCATE(GLOB_PRIM_CELL%MIN_DIST(GLOB_PRIM_CELL%S_NUM, GLOB_PRIM_CELL%P_NUM))

      ! Generate distance vectors from primitve atoms to super-cell atoms.
      ! This also takes care of possible ambiguities.
      DO P_IDX = 1, GLOB_PRIM_CELL%P_NUM
         DO S_IDX = 1, GLOB_PRIM_CELL%S_NUM
            ! Check if same atom
            IF (GLOB_PRIM_CELL%ATOM_IDX(1, P_IDX) == S_IDX) THEN
               GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN = 1
               GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%VEC(:, 1) = [0, 0, 0]
               CYCLE
            ENDIF

            DIFF_LEN_OLD = HUGE(0._q)

            ! Loop over all possible shifts along super-cell lattice vectors.
            DO K = -1, 1
            DO J = -1, 1
            DO I = -1, 1
               ! Calculate the difference vector between images
               DIFF_VEC = POSION(:, S_IDX) &
            &           - POSION(:, GLOB_PRIM_CELL%ATOM_IDX(1, P_IDX)) &
            &           + [I, J, K]
               DIFF_VEC = MATMUL(A         , DIFF_VEC)
               DIFF_LEN = DOT_PRODUCT(DIFF_VEC, DIFF_VEC) ! square is sufficient

               IF (ABS(DIFF_LEN - DIFF_LEN_OLD) < TINY) THEN ! We already got this one!
                  GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN = GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN + 1
                  GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%VEC(:, GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN) = DIFF_VEC
               ELSE IF (DIFF_LEN < DIFF_LEN_OLD) THEN ! We found a new one!
                  GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN = 1
                  GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%VEC(:, GLOB_PRIM_CELL%MIN_DIST(S_IDX, P_IDX)%LEN) = DIFF_VEC
                  DIFF_LEN_OLD = DIFF_LEN
               ENDIF
            ENDDO
            ENDDO
            ENDDO

         ENDDO
      ENDDO

   ENDSUBROUTINE SETUP_MIN_DIST


END MODULE msymmetry

!******************* SUBROUTINE RHOSYM *********************************
! RCS:  $Id: symmetry.F,v 1.8 2003/06/27 13:22:23 kresse Exp kresse $
!
! subroutine rhosym is an interface to the charge density symmetrization
! routines RHOSYR and RHOSYG which updates CHTOT (the reciprocal space
! charge density)
!
!***********************************************************************

      SUBROUTINE RHOSYM(CHTOT,GRIDC,PTRANS,NIOND,MAGROT,ISP)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRIDC

      COMPLEX(q)   CHTOT(GRIDC%RC%NP)
      REAL(q)   MAGROT(48,NIOND)
      DIMENSION PTRANS(NIOND+2,3)
! work arrays
      COMPLEX(q),ALLOCATABLE ::  CWORK(:),CWORK2(:)
      REAL(q),ALLOCATABLE    ::  WORK(:)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL


! Trivial case: no symmetry ... :
      IF (NROTK==1) RETURN

      N =GRIDC%NGX_rd*GRIDC%NGY*GRIDC%NGZ_rd
      NP=GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ
!-----------------------------------------------------------------------
! Pure point symmetry! we can use RHOSYR (real space symmetrization):
!-----------------------------------------------------------------------
      IF (NROT==NROTK .AND. NPCELL==1 .AND. .FALSE.)  THEN
         ALLOCATE(CWORK(N),WORK(NP))

         CALL FFT3D(CHTOT,GRIDC,1)
         ! merge charge to CWORK (result is real)
         CALL MRG_GRID_RL(GRIDC, CWORK, CHTOT)
         CALL RHOSYR(CWORK,GRIDC%NGX,GRIDC%NGX,GRIDC%NGY,GRIDC%NGY,GRIDC%NGZ,GRIDC%NGZ,MAGROT(1:48,1:1),ISP,WORK)
         ! and back the data go to CHTOT
         CALL DIS_GRID_RL(GRIDC, CWORK, CHTOT, .FALSE.)
! FFT of the symmetrized real space density (CHTOT):
         CALL FFT_RC_SCALE(CHTOT,CHTOT,GRIDC)
         CALL SETUNB_COMPAT(CHTOT,GRIDC)

         DEALLOCATE(CWORK,WORK)
      ELSE

!-----------------------------------------------------------------------
! General case including nontrivial translations! We have to work in
! reciprocal space, so we must use routine RHOSYG ... :
!-----------------------------------------------------------------------
         ALLOCATE(CWORK(N),CWORK2(N),WORK(NP))

         ! merge charge to CWORK
         CALL MRG_GRID_RC(GRIDC, CWORK, CHTOT)
         CALL RHOSYG(CWORK2,CWORK,GRIDC%NGX,GRIDC%NGX,GRIDC%NGY,GRIDC%NGY,&
               GRIDC%NGZ,GRIDC%NGZ,PTRANS,NIOND,MAGROT,ISP,WORK)
         ! and back the data go to CHTOT
         CALL DIS_GRID_RC(GRIDC, CWORK2, CHTOT, .FALSE.)

         DEALLOCATE(CWORK,CWORK2,WORK)
      END IF

      RETURN
      END SUBROUTINE


!******************* SUBROUTINE SYMFIELD *******************************
!
! Subroutine SYMFIELD is an interface to the magnetization
! symmetrization routine MAGSYG
!
!***********************************************************************

      SUBROUTINE SYMFIELD(MAGFLD,GRIDC,PTRANS,NIOND,MAGROT,SAXIS,LATT_CUR)

      USE prec
      USE mpimy
      USE mgrid
      USE lattice

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRIDC
      TYPE (latt) LATT_CUR

      COMPLEX(q) MAGFLD(GRIDC%MPLWV,3)

      REAL(q)   MAGROT(48,NIOND),SAXIS(3)
      DIMENSION PTRANS(NIOND+2,3)
! work arrays
      COMPLEX(q),ALLOCATABLE ::  CWORK(:),CWORK2(:)
      REAL(q),ALLOCATABLE    ::  WORK(:)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
#if defined(NGXhalf) || defined(NGZhalf)
      WRITE(0,*) 'SYMFIELD only supports full grid mode'
      STOP
#endif
! Trivial case: no symmetry ... :
      IF (NROTK==1) RETURN

      N =3*GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ
      NP=GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ

      ALLOCATE(CWORK(N),CWORK2(N),WORK(NP))
      ! merge magnetization density to CWORK
      DO I=1,3
         CALL MRG_GRID_RC(GRIDC, CWORK(NP*(I-1)+1), MAGFLD(1,I))
      ENDDO
      CALL MAGSYG(CWORK2,CWORK,GRIDC%NGX,GRIDC%NGX,GRIDC%NGY,GRIDC%NGY,&
               GRIDC%NGZ,GRIDC%NGZ,PTRANS,NIOND,MAGROT,SAXIS,LATT_CUR,WORK)
      ! and back the data go to MAGFLD
      DO I=1,3
         CALL DIS_GRID_RC(GRIDC, CWORK2(NP*(I-1)+1), MAGFLD(1,I), .FALSE.)
      ENDDO
      DEALLOCATE(CWORK,CWORK2,WORK)

      RETURN
      END


!******************* SUBROUTINE RHOSYG *********************************
!
! subroutine rhosyg symmetrizes the charge density in reciprocal space.
!
! this routine needs the charge density on the rec. space FFT-grid (RHO)
!                    the dimension parameters of array RHO (ID1,ID2,ID3)
!                    the number of FFT-grid points (NR1,NR2,NR3)
!                    all primitive cells lying in the supercell (PTRANS)
!                    the dimension parameter for array PTRANS (NIOND)
!                    and a workarray (DONE)
!
!***********************************************************************

      SUBROUTINE RHOSYG(RHOG,RHOGIN, &
     &              ID1,NR1,ID2,NR2,ID3,NR3,PTRANS,NIOND,MAGROT,ISP,DONE)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      COMPLEX(q) RHOG(ID1*ID2*ID3),RHOGIN(ID1*ID2*ID3)
      COMPLEX(q) SUM,PHASG,PHASE(48),RHOGRD
      REAL(q) MAGROT(48,NIOND)
      INTEGER G0(3),SG(3),GRPOP,TINDEX(4,48)
      DIMENSION PTRANS(NIOND+2,3),DONE(ID1*ID2*ID3)
      LOGICAL LCONJG

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,GRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      TWOPI=8._q*ATAN(1._q)
!=======================================================================
! Mark all points as 'not yet visited' ...
!=======================================================================
      DO IR=1,ID1*ID2*ID3
        DONE(IR)=-1._q
      ENDDO
!=======================================================================
! Pass over all FFT-grid points ...
!=======================================================================
      DO IR3=0,NR3-1
       G0(3)=MOD(IR3+NR3/2-1,NR3)-NR3/2+1
       DO IR2=0,NR2-1
         G0(2)=MOD(IR2+NR2/2-1,NR2)-NR2/2+1
         DO 80 IR1=0,NR1-1
! If the vector has already been visited take the next vector ... :
            IF (DONE(ID1*(ID2*IR3+IR2)+IR1+1)>0._q) GOTO 80
            G0(1)=MOD(IR1+NR1/2-1,NR1)-NR1/2+1
! New star ...
            SUM=(0._q,0._q)
! NG counts the number of G-vector within the symmetry star for G0 ...
            NG=0
            NSTAR=0
            DO 50 IROT=1,NROTK
!-----------------------------------------------------------------------
! Sum up the charges of all grid-points which are symmetry related to G0
!-----------------------------------------------------------------------
               IROTI=INVMAP(IROT)
               SG(1)=GRPOP(1,1,IROTI)*G0(1)+GRPOP(2,1,IROTI)*G0(2)+ &
     &                                            GRPOP(3,1,IROTI)*G0(3)
               SG(2)=GRPOP(1,2,IROTI)*G0(1)+GRPOP(2,2,IROTI)*G0(2)+ &
     &                                            GRPOP(3,2,IROTI)*G0(3)
               SG(3)=GRPOP(1,3,IROTI)*G0(1)+GRPOP(2,3,IROTI)*G0(2)+ &
     &                                            GRPOP(3,3,IROTI)*G0(3)

               IF ((SG(1)<-NR1/2+1).OR.(SG(1)>NR1/2).OR. &
     &             (SG(2)<-NR2/2+1).OR.(SG(2)>NR2/2).OR. &
     &             (SG(3)<-NR3/2+1).OR.(SG(3)>NR3/2)) CYCLE

               SG(1)=MOD(SG(1)+6*NR1,NR1)
               SG(2)=MOD(SG(2)+6*NR2,NR2)
               SG(3)=MOD(SG(3)+6*NR3,NR3)
               IGRID =ID1*(ID2*SG(3)+SG(2))+SG(1)+1
#ifdef NGXhalf
               IF (SG(1)<= (NR1/2)) THEN
                 IG1=SG(1)
                 IG2=SG(2)
                 IG3=SG(3)
                 LCONJG=.FALSE.
               ELSE
                 IG1=MOD(NR1-SG(1),NR1)
                 IG2=MOD(NR2-SG(2),NR2)
                 IG3=MOD(NR3-SG(3),NR3)
                 LCONJG=.TRUE.
               ENDIF
               IGRID2=(ID1/2+1)*(ID2*IG3+IG2)+IG1+1
#elif defined(NGZhalf)
               IF (SG(3)<= (NR3/2)) THEN
                 IG1=SG(1)
                 IG2=SG(2)
                 IG3=SG(3)
                 LCONJG=.FALSE.
               ELSE
                 IG1=MOD(NR1-SG(1),NR1)
                 IG2=MOD(NR2-SG(2),NR2)
                 IG3=MOD(NR3-SG(3),NR3)
                 LCONJG=.TRUE.
               ENDIF
               IGRID2=ID1*(ID2*IG3+IG2)+IG1+1
#elif defined(NGYhalf)
               WRITE(*,*)'internal ERROR: RHOSYG: NGYhalf mode not supported'
               STOP
#else
               IGRID2=IGRID
               LCONJG=.FALSE.
#endif
!-----------------------------------------------------------------------
! Phase factors for the summation due to nonprimitive translation ... :
!-----------------------------------------------------------------------
               GPHASE=TWOPI*(FLOAT(G0(1))*GTRANS(1,IROT)+ &
     &          FLOAT(G0(2))*GTRANS(2,IROT)+FLOAT(G0(3))*GTRANS(3,IROT))
               SCALE=1._q
               IF (ISP==2) SCALE=MAGROT(IROT,1)
               PHASGR=COS(GPHASE)*SCALE
               PHASGI=SIN(GPHASE)*SCALE
! WARNING: If we use nonprimitive supercells we must include all
! 'primitive' translations of the generating unitcell which are
! nonprimitive translations of the supercell ... :
               DO 20 ITRANS=2,NPCELL
                  GPHASE= &
     &                  ((GTRANS(1,IROT)+PTRANS(ITRANS,1))*FLOAT(G0(1))+ &
     &                   (GTRANS(2,IROT)+PTRANS(ITRANS,2))*FLOAT(G0(2))+ &
     &                   (GTRANS(3,IROT)+PTRANS(ITRANS,3))*FLOAT(G0(3)))
                  SCALE=1._q
                  IF (ISP==2) SCALE=MAGROT(IROT,ITRANS)
                  PHASGR=PHASGR+(COS(TWOPI*GPHASE)*SCALE)
                  PHASGI=PHASGI+(SIN(TWOPI*GPHASE)*SCALE)
   20          CONTINUE
               PHASGR=PHASGR/FLOAT(NPCELL)
               PHASGI=PHASGI/FLOAT(NPCELL)
               PHASG= CMPLX( PHASGR , PHASGI ,KIND=q)
! Symmetrized charge density for vector G0 summed up here ... :
               RHOGRD=RHOGIN(IGRID2)
               IF (LCONJG) RHOGRD=CONJG(RHOGRD)
               SUM=SUM+RHOGRD*CONJG(PHASG)

               NSTAR=NSTAR+1
!=======================================================================
! We need also the phase factors for the re-filling of the star ...
!=======================================================================
! Store the equivalent points and count the number of equivalent points:
               IF (DONE(IGRID)<0) THEN
! Here we found a new member of the star, so increase the counter NG ...
                  NG=NG+1
! ... remember coordinates for this member (needed for re-filling) ...
                  TINDEX(1,NG)=SG(1)
                  TINDEX(2,NG)=SG(2)
                  TINDEX(3,NG)=SG(3)
! ... and set the counter for the number of transformations which give
! us this member of the symmetry star (here first transformation) ...
                  TINDEX(4,NG)=1
! Intialise phase-factor table for this member (fill in the phase factor
! due to the nontrivial translation of the symmetry operation which has
! generated this new member of the symmetry star related to G0:
                  PHASE(NG)=PHASG
               ELSE
! Here the rotated vector has already been generated by some other
! symmetry operation, so we have now to sum up some information ...
                  DO 30 IG=1,NG
! Find out which member we have hit ... :
                     IF ((SG(1)==TINDEX(1,IG)).AND. &
     &                           (SG(2)==TINDEX(2,IG)).AND. &
     &                                  (SG(3)==TINDEX(3,IG))) GOTO 40
   30             CONTINUE
                  WRITE(*,*) 'RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. 1E-4'
                  STOP
! Here IG should point to the member which represents vector SG ...
   40             CONTINUE
! Count the number of symmetry operations which generate this vector:
                  TINDEX(4,IG)=TINDEX(4,IG)+1
! Add up the phase factor due to the nontrivial translation vector for
! this symmetry operation ... :
                  PHASE(IG)=PHASE(IG)+PHASG
               END IF
! Remember visited points ... :
               DONE(IGRID)=1._q
   50       CONTINUE
!=======================================================================
! Supply correct scaling to phase factors for re-filling ...
!=======================================================================
!OCL VDOPT(VL(8))
            DO 60 IG=1,NG
               PHASE(IG)=PHASE(IG)/FLOAT(TINDEX(4,IG))
   60       CONTINUE
! ... and correct scaling for SUM!
!           SUM=SUM/FLOAT(NROTK)
            SUM=SUM/FLOAT(NSTAR)
!=======================================================================
!  SUM contains now the symmetrized charge density at point G0.
!  Now fill the star of G  with this sum ... :
!=======================================================================
!OCL VDOPT(VL(8))
            DO 70 IG=1,NG
#ifdef NGXhalf
               IF (TINDEX(1,IG)> (NR1/2)) GOTO 70
#elif defined(NGZhalf)
               IF (TINDEX(3,IG)> (NR3/2)) GOTO 70
#endif
               IGRID= &
     &            xm(ID1)*(ID2* &
     &            (TINDEX(3,IG))+(TINDEX(2,IG)))+TINDEX(1,IG)+1
               RHOG(IGRID)=SUM*PHASE(IG)
   70       CONTINUE
   80    CONTINUE
       ENDDO
      ENDDO
      RETURN
      END


!******************* SUBROUTINE MAGSYG *********************************
!
! subroutine magsyg symmetrizes the magnetization density in reciprocal space.
!
! this routine needs the magnetization density on the rec. space FFT-grid (MAGG)
!                    the dimension parameters of array MAGG(ID1,ID2,ID3,3)
!                    the number of FFT-grid points (NR1,NR2,NR3)
!                    all primitive cells lying in the supercell (PTRANS)
!                    the dimension parameter for array PTRANS (NIOND)
!                    and a workarray (DONE)
!
!***********************************************************************

      SUBROUTINE MAGSYG(MAGG,MAGGIN, &
     &              ID1,NR1,ID2,NR2,ID3,NR3,PTRANS,NIOND, &
     &               MAGROT,SAXIS,LATT_CUR,DONE)

      USE prec
      USE lattice

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (latt) LATT_CUR

      COMPLEX(q)  MAGG(ID1*ID2*ID3,3),MAGGIN(ID1*ID2*ID3,3)
      COMPLEX(q)  PHASG,PHASE(48),MAGX,MAGY,MAGZ,MAGX_,MAGY_,MAGZ_
      COMPLEX(q)  MAGTX,MAGTY,MAGTZ
      COMPLEX(q)  MAGG_TEMP(3),MAGG_(3)
      REAL(q)     MAGROT(48,NIOND),SAXIS(3),ALPHA,BETA
      INTEGER     G0(3),SG(3),GRPOP,TINDEX(5,48)
      DIMENSION   PTRANS(NIOND+2,3),DONE(ID1*ID2*ID3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,GRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      LOGICAL    LDUMP
#ifndef oldsym
      INTEGER    DET
#endif

      TWOPI=8._q*ATAN(1._q)
      CALL EULER(SAXIS,ALPHA,BETA)
!=======================================================================
! Mark all points as 'not yet visited' ...
!=======================================================================
      DO IR=1,ID1*ID2*ID3
        DONE(IR)=-1._q
     ENDDO
!=======================================================================
! Pass over all FFT-grid points ...
!=======================================================================
      DO IR3=0,NR3-1
       G0(3)=MOD(IR3+NR3/2-1,NR3)-NR3/2+1
       DO IR2=0,NR2-1
         G0(2)=MOD(IR2+NR2/2-1,NR2)-NR2/2+1
         DO 80 IR1=0,NR1-1
! If the vector has already been visited take the next vector ... :
            IF (DONE(ID1*(ID2*IR3+IR2)+IR1+1)>0._q) GOTO 80
            G0(1)=MOD(IR1+NR1/2-1,NR1)-NR1/2+1
            LDUMP = .FALSE.
#ifdef debug
            IF (ABS(G0(1)) <= 1 .AND. ABS(G0(2)) <= 1 .AND. ABS(G0(3)) <= 1) LDUMP=.TRUE.
#ifndef oldsym
            LDUMP=.TRUE.
#endif
            IF (LDUMP) WRITE(*,*) '-------------------------------------------------------'
#endif
            MAGX=0 ; MAGY=0; MAGZ=0
! New star ...
! NG counts the number of G-vector within the symmetry star for G0 ...
            NG=0
            DO 50 IROT=1,NROTK
!-----------------------------------------------------------------------
! Seek out all grid-points which are symmetry related to G0
!-----------------------------------------------------------------------
               IROTI=INVMAP(IROT)
               SG(1)=GRPOP(1,1,IROTI)*G0(1)+GRPOP(2,1,IROTI)*G0(2)+ &
     &                                            GRPOP(3,1,IROTI)*G0(3)
               SG(2)=GRPOP(1,2,IROTI)*G0(1)+GRPOP(2,2,IROTI)*G0(2)+ &
     &                                            GRPOP(3,2,IROTI)*G0(3)
               SG(3)=GRPOP(1,3,IROTI)*G0(1)+GRPOP(2,3,IROTI)*G0(2)+ &
     &                                            GRPOP(3,3,IROTI)*G0(3)
               SG(1)=MOD(SG(1)+6*NR1,NR1)
               SG(2)=MOD(SG(2)+6*NR2,NR2)
               SG(3)=MOD(SG(3)+6*NR3,NR3)
               IGRID =ID1*(ID2*SG(3)+SG(2))+SG(1)+1

               IF (LDUMP) THEN
                  WRITE(*,'(3I3)') MOD(SG(1)+NR1/2,NR1)-NR1/2, &
                  MOD(SG(2)+NR2/2,NR2)-NR2/2,MOD(SG(3)+NR3/2,NR3)-NR3/2
                  WRITE(*,'("retrive  ",I6)') IGRID
#ifndef oldsym
                  WRITE(*,'("irot     ",I6)') IROT
#endif
               ENDIF
!-----------------------------------------------------------------------
! Phase factors for the summation due to nonprimitive translation ... :
!-----------------------------------------------------------------------
               GPHASE=TWOPI*(FLOAT(G0(1))*GTRANS(1,IROT)+ &
     &          FLOAT(G0(2))*GTRANS(2,IROT)+FLOAT(G0(3))*GTRANS(3,IROT))
               SCALE=1._q
               PHASGR=COS(GPHASE)*SCALE
               PHASGI=SIN(GPHASE)*SCALE
! WARNING: If we use nonprimitive supercells we must include all
! 'primitive' translations of the generating unitcell which are
! nonprimitive translations of the supercell ... :
               DO 20 ITRANS=2,NPCELL
                  GPHASE= &
     &                  ((GTRANS(1,IROT)+PTRANS(ITRANS,1))*FLOAT(G0(1))+ &
     &                   (GTRANS(2,IROT)+PTRANS(ITRANS,2))*FLOAT(G0(2))+ &
     &                   (GTRANS(3,IROT)+PTRANS(ITRANS,3))*FLOAT(G0(3)))
                  SCALE=1._q
                  PHASGR=PHASGR+(COS(TWOPI*GPHASE)*SCALE)
                  PHASGI=PHASGI+(SIN(TWOPI*GPHASE)*SCALE)
   20          CONTINUE
               PHASGR=PHASGR/FLOAT(NPCELL)
               PHASGI=PHASGI/FLOAT(NPCELL)
               PHASG= CMPLX( PHASGR , PHASGI ,KIND=q)
! Transform from "SAXIS basis" to the system of cartesian axes
! in which the integer rotation matrices are defined
               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGGIN(IGRID,:)
               MAGG_TEMP(1)=COS(BETA)*COS(ALPHA)*MAGGIN(IGRID,1)- &
            &              SIN(ALPHA)*MAGGIN(IGRID,2)+ &
            &               SIN(BETA)*COS(ALPHA)*MAGGIN(IGRID,3)
               MAGG_TEMP(2)=COS(BETA)*SIN(ALPHA)*MAGGIN(IGRID,1)+ &
            &              COS(ALPHA)*MAGGIN(IGRID,2)+ &
            &               SIN(BETA)*SIN(ALPHA)*MAGGIN(IGRID,3)
               MAGG_TEMP(3)=-SIN(BETA)*MAGGIN(IGRID,1)+ &
            &               COS(BETA)*MAGGIN(IGRID,3)
               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG_TEMP
! Go to reciprocal coordinates
               CALL CKARDIR(1,MAGG_TEMP,LATT_CUR%A)
! Rotate to the orientation of G0
#ifdef oldsym
               MAGTX=GRPOP(1,1,IROTI)*MAGG_TEMP(1)+ &
              &       GRPOP(2,1,IROTI)*MAGG_TEMP(2)+ &
              &        GRPOP(3,1,IROTI)* MAGG_TEMP(3)
               MAGTY=GRPOP(1,2,IROTI)*MAGG_TEMP(1)+ &
              &       GRPOP(2,2,IROTI)*MAGG_TEMP(2)+ &
              &        GRPOP(3,2,IROTI)*MAGG_TEMP(3)
               MAGTZ=GRPOP(1,3,IROTI)*MAGG_TEMP(1)+ &
              &       GRPOP(2,3,IROTI)*MAGG_TEMP(2)+ &
              &        GRPOP(3,3,IROTI)*MAGG_TEMP(3)
#else
               MAGTX=GRPOP(1,1,IROT)*MAGG_TEMP(1)+ &
                    &       GRPOP(2,1,IROT)*MAGG_TEMP(2)+ &
                    &        GRPOP(3,1,IROT)* MAGG_TEMP(3)
               MAGTY=GRPOP(1,2,IROT)*MAGG_TEMP(1)+ &
                    &       GRPOP(2,2,IROT)*MAGG_TEMP(2)+ &
                    &        GRPOP(3,2,IROT)*MAGG_TEMP(3)
               MAGTZ=GRPOP(1,3,IROT)*MAGG_TEMP(1)+ &
                    &       GRPOP(2,3,IROT)*MAGG_TEMP(2)+ &
                    &        GRPOP(3,3,IROT)*MAGG_TEMP(3)

               DET=GRPOP(1,1,IROT)*GRPOP(2,2,IROT)*GRPOP(3,3,IROT)- &
                   GRPOP(1,1,IROT)*GRPOP(2,3,IROT)*GRPOP(3,2,IROT)+ &
                   GRPOP(1,2,IROT)*GRPOP(2,3,IROT)*GRPOP(3,1,IROT)- &
                   GRPOP(1,2,IROT)*GRPOP(2,1,IROT)*GRPOP(3,3,IROT)+ &
                   GRPOP(1,3,IROT)*GRPOP(2,1,IROT)*GRPOP(3,2,IROT)- &
                   GRPOP(1,3,IROT)*GRPOP(2,2,IROT)*GRPOP(3,1,IROT)
               MAGTX=DET*MAGTX
               MAGTY=DET*MAGTY
               MAGTZ=DET*MAGTZ
#endif
! And back to cartesian coordinates
               MAGG_TEMP(1)=MAGTX
               MAGG_TEMP(2)=MAGTY
               MAGG_TEMP(3)=MAGTZ
               CALL CDIRKAR(1,MAGG_TEMP,LATT_CUR%B)

               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG_TEMP
! And back to SAXIS representation
               MAGTX=COS(BETA)*COS(ALPHA)*MAGG_TEMP(1)+ &
            &       COS(BETA)*SIN(ALPHA)*MAGG_TEMP(2)- &
            &        SIN(BETA)*MAGG_TEMP(3)
               MAGTY=-SIN(ALPHA)*MAGG_TEMP(1)+ &
            &        COS(ALPHA)*MAGG_TEMP(2)
               MAGTZ=SIN(BETA)*COS(ALPHA)*MAGG_TEMP(1)+ &
            &       SIN(BETA)*SIN(ALPHA)*MAGG_TEMP(2)+ &
            &        COS(BETA)*MAGG_TEMP(3)

               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGTX, MAGTY, MAGTZ
               IF (LDUMP) WRITE(*,'("  phase  ",6F14.7)') PHASG

! Symmetrized magnetization density for vector G0 summed up here ... :
               MAGX=MAGX+MAGTX*CONJG(PHASG)
               MAGY=MAGY+MAGTY*CONJG(PHASG)
               MAGZ=MAGZ+MAGTZ*CONJG(PHASG)
!=======================================================================
! We need also the phase factors for the re-filling of the star ...
!=======================================================================
! Store the equivalent points and count the number of equivalent points:
               IF (DONE(IGRID)<0) THEN
! Here we found a new member of the star, so increase the counter NG ...
                  NG=NG+1
! ... remember coordinates for this member (needed for re-filling) ...
                  TINDEX(1,NG)=SG(1)
                  TINDEX(2,NG)=SG(2)
                  TINDEX(3,NG)=SG(3)
! ... and set the counter for the number of transformations which give
! us this member of the symmetry star (here first transformation) ...
                  TINDEX(4,NG)=1
! ... and store the rotation that brings the rotated vector back to its
! original orientation as is needed when refilling the star
#ifdef oldsym
                  TINDEX(5,NG)=IROT
#else
                  TINDEX(5,NG)=IROTI
#endif
! Intialise phase-factor table for this member (fill in the phase factor
! due to the nontrivial translation of the symmetry operation which has
! generated this new member of the symmetry star related to G0:
                  PHASE(NG)=PHASG
               ELSE
! Here the rotated vector has already been generated by some other
! symmetry operation, so we have now to sum up some information ...
                  DO 30 IG=1,NG
! Find out which member we have hit ... :
                     IF ((SG(1)==TINDEX(1,IG)).AND. &
     &                           (SG(2)==TINDEX(2,IG)).AND. &
     &                                  (SG(3)==TINDEX(3,IG))) GOTO 40
   30             CONTINUE
! Here IG should point to the member which represents vector SG ...
   40             CONTINUE
! Count the number of symmetry operations which generate this vector:
                  TINDEX(4,IG)=TINDEX(4,IG)+1
! Add up the phase factor due to the nontrivial translation vector for
! this symmetry operation ... :
#ifdef oldsym
                  PHASE(IG)=PHASE(IG)+PHASG
#endif
               END IF
! Remember visited points ... :
               DONE(IGRID)=1._q
   50       CONTINUE
!=======================================================================
! Supply correct scaling to phase factors for re-filling ...
!=======================================================================
#ifdef oldsym
            DO 60 IG=1,NG
               PHASE(IG)=PHASE(IG)/FLOAT(TINDEX(4,IG))
   60       CONTINUE
#endif
! ... and correct scaling for SUM!
            MAGX=MAGX/FLOAT(NROTK)
            MAGY=MAGY/FLOAT(NROTK)
            MAGZ=MAGZ/FLOAT(NROTK)
            IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGX, MAGY, MAGZ
!=======================================================================
!  SUM contains now the symmetrized magnetization density at point G0.
!  Now fill the star of G  with this sum ... :
!=======================================================================
            DO 70 IG=1,NG
               IGRID =ID1*(ID2*TINDEX(3,IG)+TINDEX(2,IG))+TINDEX(1,IG)+1
               IF (LDUMP) WRITE(*,'("   store ",I6)') IGRID
#ifndef oldsym
               IF (LDUMP) WRITE(*,'("   phase ",2F14.7)') phase(ig)
               IF (LDUMP) WRITE(*,'("   irot ",I6)') tindex(5,ig)
#endif
               MAGX_=MAGX*PHASE(IG)
               MAGY_=MAGY*PHASE(IG)
               MAGZ_=MAGZ*PHASE(IG)
! Transform from "SAXIS basis" to the system of cartesian axes
! in which the integer rotation matrices are defined
               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG_,MAGY_,MAGZ_
               MAGG_TEMP(1)=COS(BETA)*COS(ALPHA)*MAGX_- &
            &              SIN(ALPHA)*MAGY_+ &
            &               SIN(BETA)*COS(ALPHA)*MAGZ_
               MAGG_TEMP(2)=COS(BETA)*SIN(ALPHA)*MAGX_+ &
            &              COS(ALPHA)*MAGY_+ &
            &               SIN(BETA)*SIN(ALPHA)*MAGZ_
               MAGG_TEMP(3)=-SIN(BETA)*MAGX_+ &
            &               COS(BETA)*MAGZ_
! Go to reciprocal coordinates
               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG_TEMP
               CALL CKARDIR(1,MAGG_TEMP,LATT_CUR%A)
! Rotate back to the original orientation
               MAGG_(1)=GRPOP(1,1,TINDEX(5,IG))*MAGG_TEMP(1)+ &
              &               GRPOP(2,1,TINDEX(5,IG))*MAGG_TEMP(2)+ &
              &                GRPOP(3,1,TINDEX(5,IG))*MAGG_TEMP(3)
               MAGG_(2)=GRPOP(1,2,TINDEX(5,IG))*MAGG_TEMP(1)+ &
              &               GRPOP(2,2,TINDEX(5,IG))*MAGG_TEMP(2)+ &
              &                GRPOP(3,2,TINDEX(5,IG))*MAGG_TEMP(3)
               MAGG_(3)=GRPOP(1,3,TINDEX(5,IG))*MAGG_TEMP(1)+ &
              &               GRPOP(2,3,TINDEX(5,IG))*MAGG_TEMP(2)+ &
              &                GRPOP(3,3,TINDEX(5,IG))*MAGG_TEMP(3)
#ifndef oldsym
               DET=GRPOP(1,1,TINDEX(5,IG))*&
                   GRPOP(2,2,TINDEX(5,IG))*&
                   GRPOP(3,3,TINDEX(5,IG)) - &
                   GRPOP(1,1,TINDEX(5,IG))*&
                   GRPOP(2,3,TINDEX(5,IG))*&
                   GRPOP(3,2,TINDEX(5,IG)) + &
                   GRPOP(1,2,TINDEX(5,IG))*&
                   GRPOP(2,3,TINDEX(5,IG))*&
                   GRPOP(3,1,TINDEX(5,IG)) - &
                   GRPOP(1,2,TINDEX(5,IG))*&
                   GRPOP(2,1,TINDEX(5,IG))*&
                   GRPOP(3,3,TINDEX(5,IG)) + &
                   GRPOP(1,3,TINDEX(5,IG))*&
                   GRPOP(2,1,TINDEX(5,IG))*&
                   GRPOP(3,2,TINDEX(5,IG)) - &
                   GRPOP(1,3,TINDEX(5,IG))*&
                   GRPOP(2,2,TINDEX(5,IG))*&
                   GRPOP(3,1,TINDEX(5,IG))
               MAGG_=MAGG_*DET
#endif
! And back to cartesian coordinates
               CALL CDIRKAR(1,MAGG_(1:3),LATT_CUR%B)
               IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG_
! And back to SAXIS representation
               MAGG(IGRID,1)=COS(BETA)*COS(ALPHA)*MAGG_(1)+ &
            &               COS(BETA)*SIN(ALPHA)*MAGG_(2)- &
            &                SIN(BETA)*MAGG_(3)
               MAGG(IGRID,2)=-SIN(ALPHA)*MAGG_(1)+ &
            &                COS(ALPHA)*MAGG_(2)
               MAGG(IGRID,3)=SIN(BETA)*COS(ALPHA)*MAGG_(1)+ &
            &               SIN(BETA)*SIN(ALPHA)*MAGG_(2)+ &
            &                COS(BETA)*MAGG_(3)
            IF (LDUMP) WRITE(*,'(9X,6F14.7)') MAGG(IGRID,:)
   70       CONTINUE
   80    CONTINUE
      ENDDO
      ENDDO
      RETURN
      END

!**************** SUBROUTINE EULER  ************************************
!
!     This routine calculates the Euler angles (alpha,beta)_i
!     for the site-dependent spin directions (sx,sy,sz)_i.
!     beta  is the angle between the z axis and s
!     alpha is the angle between x and (sx,sy,0)
!
!***********************************************************************

      SUBROUTINE EULER(S, ALPHA, BETA)

      USE prec
      USE constant

      IMPLICIT NONE
      REAL(q) T,T2,S(3), &
     &        ALPHA,BETA
! local

!       Normalisation of the spin directions
        T=SQRT(S(1)**2+S(2)**2+S(3)**2)

        IF (T<1.0E-10_q) THEN
          ALPHA=0
          BETA=0
        ELSE

          S(1)=S(1)/T
          S(2)=S(2)/T
          S(3)=S(3)/T

          T2=SQRT(S(1)**2+S(2)**2)
!       calculates the Euler angles (alpha,beta) for
!       each ions.
          IF(ABS(S(1))>=1.0E-10_q)THEN
            ALPHA=ATAN2(S(2),S(1))
          ELSE
!       in case sx(i)=0, avoids division by 0
            IF(ABS(S(2))>1.0E-10_q)THEN
              ALPHA=pi/2
            ENDIF
            IF(ABS(S(2))<1.0E-10_q)THEN
              ALPHA=0
            ENDIF
          ENDIF
          IF(ABS(S(3))>1.0E-10_q)THEN
            BETA=ATAN2(T2,S(3))
          ELSE
!       in case sz(i)=0, cos(beta(i))=0
            BETA=pi/2
          ENDIF
        ENDIF

      END SUBROUTINE EULER


!******************* SUBROUTINE RHOSYR *********************************
!
! subroutine rhosyr symmetrizes the charge density in direct space. This
! is easy and fast (it saves some FFTs) but it can only be used in the
! case of pure point symmetry - if we have nontrivial translations for
! certain symmetry elements we must use routine RHOSYG!
!
! this routine needs the charge density on the real space FFT-grid (RHO)
!                    the dimension parameters of array RHO (ID1,ID2,ID3)
!                    the number of FFT-grid points (NR1,NR2,NR3)
!                    and a workarray (DONE)
!
!***********************************************************************

      SUBROUTINE RHOSYR(RHO,ID1,NR1,ID2,NR2,ID3,NR3,MAGROT,ISP,DONE)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q)   RHO(ID1*ID2*ID3),MAGROT(48)
      DIMENSION DONE(ID1,ID2,ID3)
      INTEGER R(3),SR(3),S

      COMMON /SYMM/ S(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

!=======================================================================
! Mark all points as 'not yet visited' ...
!=======================================================================
      DO IR=1,ID1*ID2*ID3
         DONE(IR,1,1)=-1._q
      ENDDO
!=======================================================================
! Pass over all FFT-grid points ...
!=======================================================================
      DO IR3=1,NR3
       R(3)=IR3-1
       DO IR2=1,NR2
         R(2)=IR2-1
         DO 40 IR1=1,NR1
! If the vector has already been visited take the next vector ... :
            IF (DONE(IR1,IR2,IR3)>0._q) GOTO 40
            R(1)=IR1-1
! Note: in direct space the charge density is always real! So we use a
            SUM=0._q
!-----------------------------------------------------------------------
! Sum up the charges of all grid-points which are symmetry related to R
!-----------------------------------------------------------------------
            DO 20 IROT=1,NROT
               SCALE=1._q
               IF (ISP==2) SCALE=MAGROT(IROT)
               SR(1)=S(1,1,IROT)*R(1)+S(2,1,IROT)*R(2)+S(3,1,IROT)*R(3)
               SR(2)=S(1,2,IROT)*R(1)+S(2,2,IROT)*R(2)+S(3,2,IROT)*R(3)
               SR(3)=S(1,3,IROT)*R(1)+S(2,3,IROT)*R(2)+S(3,3,IROT)*R(3)
               SR(1)=MOD(SR(1)+6*NR1,NR1)
               SR(2)=MOD(SR(2)+6*NR2,NR2)
               SR(3)=MOD(SR(3)+6*NR3,NR3)
               IGRID=ID1*(ID2*SR(3)+SR(2))+SR(1)+1
               SUM=SUM+(RHO(IGRID)*SCALE)
   20       CONTINUE
! Correct scaling for SUM!
            SUM=SUM/FLOAT(NROT)
!-----------------------------------------------------------------------
!  SUM contains now the symmetrized charge density at point R.
!  Now fill the star of all symmetry related Rs  with this sum.
!-----------------------------------------------------------------------
            DO 30 IROT=1,NROT
               SCALE=1._q
               IF (ISP==2) SCALE=MAGROT(IROT)
               SR(1)=S(1,1,IROT)*R(1)+S(2,1,IROT)*R(2)+S(3,1,IROT)*R(3)
               SR(2)=S(1,2,IROT)*R(1)+S(2,2,IROT)*R(2)+S(3,2,IROT)*R(3)
               SR(3)=S(1,3,IROT)*R(1)+S(2,3,IROT)*R(2)+S(3,3,IROT)*R(3)
               SR(1)=MOD(SR(1)+6*NR1,NR1)
               SR(2)=MOD(SR(2)+6*NR2,NR2)
               SR(3)=MOD(SR(3)+6*NR3,NR3)
               IGRID=ID1*(ID2*SR(3)+SR(2))+SR(1)+1
! Remember the grid points which we have already visited ... :
               DONE(SR(1)+1,SR(2)+1,SR(3)+1)=1._q
               RHO(IGRID)=SUM*SCALE
   30       CONTINUE
   40    CONTINUE
      ENDDO
      ENDDO
      RETURN
      END


!******************* SUBROUTINE POSSYM *********************************
!
! subroutine POSSYM is an interface to the lattlib-routine PSYM
! this routine is used to (re-)symmetrize the atomic positions
!
!***********************************************************************

      SUBROUTINE POSSYM(POSION,ROTMAP,PTRANS,NIOND,NTYP,NITYP,FROT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER ROTMAP(NIOND,NROTK,NPCELL)

      DIMENSION POSION(3,NIOND),PTRANS(NIOND+2,3),NITYP(NTYP)
      DIMENSION TAU(NIOND,3),TAUWRK(NIOND,3),FROT(3,NIOND)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      ! copy the positions to TAU
      DO I=1,NIOND
         TAU(I,:)=POSION(:,I)
      ENDDO

      CALL PSYM(TAU,ROTMAP,ISYMOP,GTRANS,PTRANS,NROTK,NPCELL,NIOND,1,NTYP,NITYP,TAUWRK)

      ! copy the symmetrized positions to FROT
      DO I=1,NIOND
         FROT(:,I)=TAUWRK(I,:)
      ENDDO

      RETURN
      END

!******************* SUBROUTINE FORSYM *********************************
!
! subroutine forsym is a simplified interface to lattlib-routine FSYM
! this routine serves generally for symmetrisation of a vector field
! F(IATOM) given in   c a r t e s i a n   coordinates.
!
! this routine requires the vector-field to be symmetrized (FION)
!             the connection map (ROTMAP) set up in routine INISYM
!             the number of atomic species (NTYP)
!             the number of atoms of each species (NITYP)
!             some workarrays (FROT,WRKROT)
!             the dimensioning parameter NIOND (max. no. of atoms)
!             and the lattice vectors (A)
!
! the input data will be replaced by the symmetric data on output
!
!***********************************************************************

      SUBROUTINE FORSYM(FION,ROTMAP,NTYP,NITYP,NIOND,FROT,WRKROT,A)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER ROTMAP(NIOND,NROTK,NPCELL)

      DIMENSION FION(3,NIOND),NITYP(NTYP),FROT(NIOND,3),WRKROT(NIOND,3)
      DIMENSION A(3,3),B1(3),B2(3),B3(3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL FSYM(FION,ROTMAP,ISYMOP,NROTK,NPCELL,1,NTYP,NIOND,NITYP, &
     &                        FROT,WRKROT,A(1,1),A(1,2),A(1,3),B1,B2,B3)

      RETURN
      END


!******************* SUBROUTINE VECSYM *********************************
!
! subroutine vecsym is a simplified interface to lattlib-routine FSYM
! this routine serves generally for symmetrisation of a vector field
! V(IATOM) given in   d i r e c t   lattice coordinates.
!
! this routine requires the vector-field to be symmetrized (FION)
!             the connection map (ROTMAP) set up in routine INISYM
!             the number of atomic species (NTYP)
!             the number of atoms of each species (NITYP)
!             some workarrays (FROT,WRKROT)
!             and the dimensioning parameter NIOND (max. no. of atoms)
!
! the input data will be replaced by the symmetric data on output
!
!***********************************************************************

      SUBROUTINE VECSYM(VEC,ROTMAP,NTYP,NITYP,NIOND,FROT,WRKROT)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER ROTMAP(NIOND,NROTK,NPCELL)

      DIMENSION VEC(3,NIOND),NITYP(NTYP),FROT(NIOND,3),WRKROT(NIOND,3)
      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      SAVE A1,A2,A3,B1,B2,B3
      DATA A1 /1._q,0._q,0._q/, A2 /0._q,1._q,0._q/, A3 /0._q,0._q,1._q/
      DATA B1 /1._q,0._q,0._q/, B2 /0._q,1._q,0._q/, B3 /0._q,0._q,1._q/

      CALL FSYM(VEC,ROTMAP,ISYMOP,NROTK,NPCELL,1,NTYP,NIOND,NITYP, &
     &                                    FROT,WRKROT,A1,A2,A3,B1,B2,B3)

      RETURN
      END


!******************* SUBROUTINE TENSYM *********************************
!
! subroutine tensym is a simplified interface to symlib-routine TSYM2
! this routine serves to symmetrize of a site specific tensor
! TION(IATOM) given in   c a r t e s i a n   coordinates.
!
! this routine requires the tensors to be symmetrized (TION)
!             the connection map (ROTMAP) set up in routine INISYM
!             the number of atomic species (NTYP)
!             the number of atoms of each species (NITYP)
!             some workarrays (TROT,WRKROT)
!             the dimensioning parameter NIOND (max. no. of atoms)
!             and the lattice vectors (A)
!
! the input data will be replaced by the symmetric data on output
!
!***********************************************************************

      SUBROUTINE TENSYM(TION,ROTMAP,NTYP,NITYP,NIOND,A)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER ROTMAP(NIOND,NROTK,NPCELL)

      DIMENSION TION(3,3,NIOND),NITYP(NTYP),TROT(NIOND,3,3),WRKROT(NIOND,3,3)
      DIMENSION A(3,3),B1(3),B2(3),B3(3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      CALL RECIPS(1._q,A(1,1),A(1,2),A(1,3),B1,B2,B3)
      CALL TSYM2(TION,ROTMAP,ISYMOP,NROTK,NPCELL,1,NTYP,NIOND,NITYP, &
     &                        TROT,WRKROT,A(1,1),A(1,2),A(1,3),B1,B2,B3)

      RETURN
      END


!******************** SUBROUTINE IBZKPT ********************************
!
! subroutine ibzkpt reduces a given k-lattice in the full 1st Brillouin
! zone to its irreducible part (setting also the correct weights) ...
!
! this routine needs the reciprocal lattice vectors (B) [cartesian]
!                    the basis vectors of the k-lattice (BK) [cartesian]
!                    the shift of the k-lattice (SHIFT) [BK-coordinates]
!                    the dimension parameter NKDIM for arrays VKPT,WTKPT
!                    the dimension parameter NTETD for array IDTET
!                    the dimension parameter IKPTD for array IKPT
!                    a switch for usage of the tetrahedron method (LTET)
!                    the 'scaling factor' for the Bs (SCALE)
!                    and the unit where to write informations (IU6)
! it returns:        the number of irreducible k-points (NKPT)
!                    the k-point coordinates (VKPT) [B-coordinates]
!                    and the weighting factors for each k-point (WTKPT)
! and if LTET=T      the k-point 'connection table' (IKPT)
!                    the number of irreducible tetrahedra (NTET)
!                    the corners of the tetrahedra [IDTET(1-4,...)]
!                    the volume weight factor of a tetrahedron (VOLWGT)
!                    the degeneracy of each tetrahedron [IDTET(0,...)]
!                    and the corners of the tetrahedra [IDTET(1-4,...)]
!
!***********************************************************************

      SUBROUTINE IBZKPT(B,BK,SHIFT,NKPT,VKPT,WTKPT,NKDIM, &
     &                LTET,NTET,IDTET,NTETD,VOLWGT,IKPT,IKPTD,SCALE, &
                      LINVERSION,LNOSYM,LSHIFT_KPOINTS,IU6)
      USE prec
      USE main_mpi
#ifndef oldsym
      USE spinsym
#endif

      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL SGREQL,TKEEP,TZERO,THALF,LTET,OCCUP

      DIMENSION S(3,3,48),G(3,3,48),KGRPOP(3,3,48),INVERS(9),CDIMK(6)
      DIMENSION A(3,3),AK(3,3),V(3),VR(3),P1(3),P2(3),P3(3),CDIMR(6)
      DIMENSION B(3,3),BK(3,3),SHIFT(3),VKPT(3,NKDIM),WTKPT(NKDIM),T(3)
      DIMENSION PINV(3,3),SHTEST(3),IFULLS(3,3,48),ISWRK(3,3,48),NA(1)
      DIMENSION GWRK(3,48),TAUFUL(1,3,1),TRFULL(1,3,1),SWRK(1),INDEX(1)
      DIMENSION IFULLG(3,3,48),VTEST(3,48),IDTET(0:4,NTETD),BKT(3,3)
      DIMENSION IKPT(IKPTD,IKPTD,IKPTD),TESTV(48,3),PB1(3),PB2(3),PB3(3)
      DIMENSION TWORK(48,3),PWORK(50,3),WORK(50,3),IWORK(50)
      LOGICAL LINVERSION ! generate also k-points related to each other through inversion symmetry
                         ! even if inversion is a symmetry element
      LOGICAL LNOSYM     ! switch of all symmetry
                         ! in particular the symmetrization with respect to the symmetry operations
                         ! of the basis is bypassed
      LOGICAL LSHIFT_KPOINTS  ! add all k-points that are difference vectors between two
                         ! other k-points
                         ! for the GW case this is required
                         ! presently this works only in combination with LNOSYM
                         ! LSHIFT_KPOINTS is simply neglected
                         ! in this case the GW routine will stop issuing an error

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      SAVE TINY,INVERS
      DATA TINY /1.E-6_q/, INVERS /-1,0,0,0,-1,0,0,0,-1/

! First check: The generating cell must have positive volume:
      CALL CELVOL(BK(1,1),BK(1,2),BK(1,3),OMEGBK)
      IF (OMEGBK<(TINY*TINY*TINY)) CALL ERROR(' IBZKPT', &
     &               ' Volume of generating cell for k-mesh is zero!',3)
      IF (OMEGBK<0._q) THEN
         IF (IU6>=0) THEN
         WRITE(IU6,*) 'Warning IBZKPT: ', &
     &             'Generating cell for k-mesh has negative volume ...'
         ENDIF
         BK(1,1)=-BK(1,1)
         BK(2,1)=-BK(2,1)
         BK(3,1)=-BK(3,1)
         BK(1,2)=-BK(1,2)
         BK(2,2)=-BK(2,2)
         BK(3,2)=-BK(3,2)
         BK(1,3)=-BK(1,3)
         BK(2,3)=-BK(2,3)
         BK(3,3)=-BK(3,3)
      ENDIF
! We must only consider shifts in the interval (-0.5,0.5] ... :
      SHIFT(1)=MOD(SHIFT(1)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      SHIFT(2)=MOD(SHIFT(2)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      SHIFT(3)=MOD(SHIFT(3)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      TZERO=((ABS(SHIFT(1))<TINY).AND. &
     &              (ABS(SHIFT(2))<TINY).AND.(ABS(SHIFT(3))<TINY))
!=======================================================================
! Consistency check: we allow usage of other k-mesh lattice types than
! the lattice type of the reciprocal lattice (for example bcc-reciprocal
! lattice but simple cubic k-mesh ...). But two restrictions apply here:
! both types of lattices should at least belong to the same class of
! bravais lattices (cubic, tetragonal, orthorhombic, monoclinic, ...)
! or to some higher class which is a special case of the given class -
! that means we may mix sc,fcc,bcc but not bcc and monoclinic or so ...
! - and they should be commensurable (that means the reciprocal lattice
! vectors represented in the basis of the k-lattice should have INTEGER
! components!). Otherwise PRINT ERROR MESSAGE and STOP EXECUTION !!!!!
!=======================================================================
      CALL RECIPS(1._q,B(1,1),B(1,2),B(1,3),A(1,1),A(1,2),A(1,3))
      CALL RECIPS(1._q,BK(1,1),BK(1,2),BK(1,3),AK(1,1),AK(1,2),AK(1,3))
      V(1)=B(1,1)*AK(1,1)+B(2,1)*AK(2,1)+B(3,1)*AK(3,1)
      V(2)=B(1,1)*AK(1,2)+B(2,1)*AK(2,2)+B(3,1)*AK(3,2)
      V(3)=B(1,1)*AK(1,3)+B(2,1)*AK(2,3)+B(3,1)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 1st reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',1)
      V(1)=B(1,2)*AK(1,1)+B(2,2)*AK(2,1)+B(3,2)*AK(3,1)
      V(2)=B(1,2)*AK(1,2)+B(2,2)*AK(2,2)+B(3,2)*AK(3,2)
      V(3)=B(1,2)*AK(1,3)+B(2,2)*AK(2,3)+B(3,2)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 2nd reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',2)
      V(1)=B(1,3)*AK(1,1)+B(2,3)*AK(2,1)+B(3,3)*AK(3,1)
      V(2)=B(1,3)*AK(1,2)+B(2,3)*AK(2,2)+B(3,3)*AK(3,2)
      V(3)=B(1,3)*AK(1,3)+B(2,3)*AK(2,3)+B(3,3)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 3rd reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',3)
      PB1(1)=B(1,1)
      PB1(2)=B(2,1)
      PB1(3)=B(3,1)
      PB2(1)=B(1,2)
      PB2(2)=B(2,2)
      PB2(3)=B(3,2)
      PB3(1)=B(1,3)
      PB3(2)=B(2,3)
      PB3(3)=B(3,3)
      CALL LATTYP(PB1,PB2,PB3,IBRAVR,CDIMR,100)
      IBRAV=IBRAVR
! All cubic types (matched onto simple cubic):
      IF (IBRAVR<=3) IBRAVR=1
! All tetragonal types (matched onto simple tetragonal):
      IF ((IBRAVR<=6).AND.(IBRAVR>=5)) IBRAVR=5
! Trigonal (rhomboedric) belong to the hexagonal system ... :
      IF (IBRAVR==7) IBRAVR=4
! All orthorhombic types (matched onto simple orthorhombic):
      IF ((IBRAVR<=11).AND.(IBRAVR>=8)) IBRAVR=8
! All monoclinic types (matched onto simple monoclinic):
      IF ((IBRAVR<=13).AND.(IBRAVR>=12)) IBRAVR=12
      P1(1)=BK(1,1)
      P1(2)=BK(2,1)
      P1(3)=BK(3,1)
      P2(1)=BK(1,2)
      P2(2)=BK(2,2)
      P2(3)=BK(3,2)
      P3(1)=BK(1,3)
      P3(2)=BK(2,3)
      P3(3)=BK(3,3)
      CALL LATTYP(P1,P2,P3,IBRAVK,CDIMK,100)
! All cubic types (matched onto simple cubic):
      IF (IBRAVK<=3) IBRAVK=1
! All tetragonal types (matched onto simple tetragonal):
      IF ((IBRAVK<=6).AND.(IBRAVK>=5)) IBRAVK=5
! Trigonal (rhomboedric) belong to the hexagonal system ... :
      IF (IBRAVK==7) IBRAVK=4
! All orthorhombic types (matched onto simple orthorhombic):
      IF ((IBRAVK<=11).AND.(IBRAVK>=8)) IBRAVK=8
! All monoclinic types (matched onto simple monoclinic):
      IF ((IBRAVK<=13).AND.(IBRAVK>=12)) IBRAVK=12
! Now it is not yet the full truth ... : some lattice types can be very
! special cases of other lattice types (cubic is also tetragonal or
! tetragonal is also orthorhombic, 'all is triclinic' ...).
      IF ((IBRAVR<IBRAVK).OR.(((IBRAVR==4).OR.(IBRAVR>=12)).AND. &
     &  (IBRAVR/=IBRAVK))) CALL WARN(' IBZKPT',' Reciprocal lattice ' &
     &         //'and k-lattice belong to different class of lattices.' &
     &         //' Often results are still useful...', &
     &                                                    IBRAVR*IBRAVK)

!=======================================================================
! Test for symmetry conserving shifts!
!=======================================================================
      TKEEP=.TRUE.
      CALL RECIPS(1._q,P1,P2,P3,PINV(1,1),PINV(1,2),PINV(1,3))
      T(1)=SHIFT(1)*BK(1,1)+SHIFT(2)*BK(1,2)+SHIFT(3)*BK(1,3)
      T(2)=SHIFT(1)*BK(2,1)+SHIFT(2)*BK(2,2)+SHIFT(3)*BK(2,3)
      T(3)=SHIFT(1)*BK(3,1)+SHIFT(2)*BK(3,2)+SHIFT(3)*BK(3,3)
      SHTEST(1)=T(1)*PINV(1,1)+T(2)*PINV(2,1)+T(3)*PINV(3,1)
      SHTEST(2)=T(1)*PINV(1,2)+T(2)*PINV(2,2)+T(3)*PINV(3,2)
      SHTEST(3)=T(1)*PINV(1,3)+T(2)*PINV(2,3)+T(3)*PINV(3,3)
! Periodic boundary conditions: INTEGERS -> 0, +/- HALFS -> 0.5
      SHTEST(1)=MOD(SHTEST(1)+6.25_q,1._q)-0.25_q
      SHTEST(2)=MOD(SHTEST(2)+6.25_q,1._q)-0.25_q
      SHTEST(3)=MOD(SHTEST(3)+6.25_q,1._q)-0.25_q
! General test: We may only have the values 0.0 or 0.5 ... :
      IF (((ABS(SHTEST(1))>TINY).AND.(ABS(SHTEST(1)-0.5_q)>TINY)) &
     & .OR.((ABS(SHTEST(2))>TINY).AND.(ABS(SHTEST(2)-0.5_q)>TINY)) &
     & .OR.((ABS(SHTEST(3))>TINY).AND.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      TZERO=TZERO.AND.TKEEP
      THALF=TKEEP.AND.(.NOT.TZERO)
! There are no restrictions for IBRAV=8,12,14 - but for other IBRAV:
      IF (((IBRAV==1).OR.(IBRAV==3).OR.(IBRAV==6).OR. &
     &                     (IBRAV==7).OR.(IBRAV==10)).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==2).AND.((ABS(SHTEST(1))>TINY).OR. &
     &  (ABS(SHTEST(2))>TINY).OR.(ABS(SHTEST(3))>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==4).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  ((ABS(SHTEST(3))>TINY).AND.(ABS(SHTEST(3)-0.5_q)>TINY)))) &
     &                                                     TKEEP=.FALSE.
      IF (((IBRAV==5).OR.(IBRAV==11).OR.(IBRAV==13)).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &  ((ABS(SHTEST(3)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(1))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==9).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &  ((ABS(SHTEST(1)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3)-0.5_q)>TINY)).AND.((ABS(SHTEST(1))>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.

!=======================================================================
! Following some tests for setting up the k-mesh if we want to use the
! tetrahedron integration method: To get a simple scheme for finding
! all symmetry irreducible tetrahedra we must restrict ourself to
! special shifts of the k-mesh because for arbitrary shifts it is
! very hard to get an efficient scheme (or any scheme at all ...).
! The reason is: we could nomore deal with parallelepipeds but had to
! deal with more complicated polyhedra which had to be devided into
! tetrahedra - but such things are not yet implemented and are probably
! quite difficult to implement. So, sorry people - this must be enough!
! (There are a lot of programs which are not even able to deal with any
! shifts and it might be that this is even the only program which can
! at least deal with some shifts at all-folks, you ca not expect more!)
!=======================================================================
      IF ((IBRAV==1).OR.(IBRAV==3).OR.(IBRAV==7)) &
     &     THALF=THALF.AND.(ABS(SHTEST(1)-0.5_q)<TINY).AND. &
     &     (ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3)-0.5_q)<TINY)
      IF (IBRAV==2) THALF=THALF.AND.(((ABS(SHTEST(1)-0.5_q)<TINY) &
     & .AND.(ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3)-0.5_q)< &
     & TINY)).OR.((ABS(SHTEST(1)-0.5_q)<TINY).AND.(ABS(SHTEST(2))< &
     & TINY).AND.(ABS(SHTEST(3))<TINY)).OR.((ABS(SHTEST(1))<TINY) &
     &  .AND.(ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3))<TINY)) &
     &  .OR.((ABS(SHTEST(1))<TINY).AND.(ABS(SHTEST(2))<TINY).AND. &
     &                                    (ABS(SHTEST(3)-0.5_q)<TINY)))
      IF (IBRAV==4) THALF=THALF.AND.(ABS(SHTEST(1))<TINY).AND. &
     &         (ABS(SHTEST(2))<TINY).AND.(ABS(SHTEST(3)-0.5_q)<TINY)
      IF (LTET.AND.(.NOT.TZERO).AND.(.NOT.THALF)) CALL ERROR(' IBZKPT', &
     & ' Routine TETIRR needs special values for the k-mesh shifts!',-3)

      ! LNOSYM avoids that shifted meshes are blown up
      IF (.NOT.TKEEP .AND. .NOT. LNOSYM) THEN
! Get 'full' symmetry of bare (unshifted) k-lattice/k-lattice unit-cell:
         TAUFUL(1,1,1)=0._q
         TAUFUL(1,2,1)=0._q
         TAUFUL(1,3,1)=0._q
         NA(1)=1
         CALL SETGRP(ISWRK,IFULLS,GWRK,NDUM,NDUMK,TAUFUL,IBRAV,NRFULL, &
     &                                   TRFULL,1,1,NA,1,INDEX,SWRK,100)
         CALL SGRCON(IFULLS,IFULLG,NRFULL, &
     &                              PB1,PB2,PB3,BK(1,1),BK(1,2),BK(1,3))
      ELSE
! Symmetry conserving shift needs no further 'mesh-expansion':
         NRFULL=1
         IFULLG(1,1,1)=1
         IFULLG(2,1,1)=0
         IFULLG(3,1,1)=0
         IFULLG(1,2,1)=0
         IFULLG(2,2,1)=1
         IFULLG(3,2,1)=0
         IFULLG(1,3,1)=0
         IFULLG(2,3,1)=0
         IFULLG(3,3,1)=1
      END IF
#ifndef oldsym
      CALL SET_SPINROT(A, B, ISYMOP, NROTK, GTRANS, IU6)
#endif
!=======================================================================
! We may always use inversion (also if it is no real symmetry operation
! of the atomic lattice) - this is due to time-reversal symmetry ... :
!=======================================================================
      NROTKM=NROTK
      INVYES=0
      DO 1 I=1,NROTK
! Test existence of inversion operator:
         IF (SGREQL(IGRPOP(1,1,I),INVERS)) INVYES=1
! ... and copy the symmetry operators to some temporary array ... :
         KGRPOP(1,1,I)=IGRPOP(1,1,I)
         KGRPOP(2,1,I)=IGRPOP(2,1,I)
         KGRPOP(3,1,I)=IGRPOP(3,1,I)
         KGRPOP(1,2,I)=IGRPOP(1,2,I)
         KGRPOP(2,2,I)=IGRPOP(2,2,I)
         KGRPOP(3,2,I)=IGRPOP(3,2,I)
         KGRPOP(1,3,I)=IGRPOP(1,3,I)
         KGRPOP(2,3,I)=IGRPOP(2,3,I)
         KGRPOP(3,3,I)=IGRPOP(3,3,I)
    1 CONTINUE

      IF (LINVERSION) THEN
! If inversion is not yet included add inversion!
         IF (INVYES==0) THEN
            DO 2 I=1,NROTK
! ... this means in full consequence multiply all operators with I ... :
               CALL SGRPRD(INVERS,KGRPOP(1,1,I),KGRPOP(1,1,I+NROTK))
 2          CONTINUE
! ... so we have now a group with twice as much symmetry operations:
            NROTKM=2*NROTK
         END IF

      END IF
!=======================================================================
! First transform the symmetry operators to the representation for the
! basis generating the k-mesh (basis BK):
!=======================================================================
      DO 3 I=1,NROTKM
         S(1,1,I)=FLOAT(KGRPOP(1,1,I))
         S(2,1,I)=FLOAT(KGRPOP(2,1,I))
         S(3,1,I)=FLOAT(KGRPOP(3,1,I))
         S(1,2,I)=FLOAT(KGRPOP(1,2,I))
         S(2,2,I)=FLOAT(KGRPOP(2,2,I))
         S(3,2,I)=FLOAT(KGRPOP(3,2,I))
         S(1,3,I)=FLOAT(KGRPOP(1,3,I))
         S(2,3,I)=FLOAT(KGRPOP(2,3,I))
         S(3,3,I)=FLOAT(KGRPOP(3,3,I))
    3 CONTINUE
      CALL SGRTRF(S,G,NROTKM,B(1,1),B(1,2),B(1,3), &
     &                                          BK(1,1),BK(1,2),BK(1,3))

!=======================================================================
! Rough estimate: how many mesh-points do we need in each direction?
!=======================================================================
      BK1A1=BK(1,1)*A(1,1)+BK(2,1)*A(2,1)+BK(3,1)*A(3,1)
      BK2A1=BK(1,1)*A(1,2)+BK(2,1)*A(2,2)+BK(3,1)*A(3,2)
      BK3A1=BK(1,1)*A(1,3)+BK(2,1)*A(2,3)+BK(3,1)*A(3,3)
      BK1A2=BK(1,2)*A(1,1)+BK(2,2)*A(2,1)+BK(3,2)*A(3,1)
      BK2A2=BK(1,2)*A(1,2)+BK(2,2)*A(2,2)+BK(3,2)*A(3,2)
      BK3A2=BK(1,2)*A(1,3)+BK(2,2)*A(2,3)+BK(3,2)*A(3,3)
      BK1A3=BK(1,3)*A(1,1)+BK(2,3)*A(2,1)+BK(3,3)*A(3,1)
      BK2A3=BK(1,3)*A(1,2)+BK(2,3)*A(2,2)+BK(3,3)*A(3,2)
      BK3A3=BK(1,3)*A(1,3)+BK(2,3)*A(2,3)+BK(3,3)*A(3,3)
      IF (ABS(BK1A1)<TINY) BK1A1=1.E30_q
      IF (ABS(BK1A2)<TINY) BK1A2=1.E30_q
      IF (ABS(BK1A3)<TINY) BK1A3=1.E30_q
      IF (ABS(BK2A1)<TINY) BK2A1=1.E30_q
      IF (ABS(BK2A2)<TINY) BK2A2=1.E30_q
      IF (ABS(BK2A3)<TINY) BK2A3=1.E30_q
      IF (ABS(BK3A1)<TINY) BK3A1=1.E30_q
      IF (ABS(BK3A2)<TINY) BK3A2=1.E30_q
      IF (ABS(BK3A3)<TINY) BK3A3=1.E30_q
      NKX=NINT(MAX(MAX(1._q/ABS(BK1A1),1._q/ABS(BK2A1)),1._q/ABS(BK3A1)))
      NKY=NINT(MAX(MAX(1._q/ABS(BK1A2),1._q/ABS(BK2A2)),1._q/ABS(BK3A2)))
      NKZ=NINT(MAX(MAX(1._q/ABS(BK1A3),1._q/ABS(BK2A3)),1._q/ABS(BK3A3)))
! Testing dimensions of array IKPT ... :
      IF (LTET) THEN
         IF (NKX>IKPTD) CALL ERROR(' IBZKPT',' NKX>IKPTD',NKX)
         IF (NKY>IKPTD) CALL ERROR(' IBZKPT',' NKY>IKPTD',NKY)
         IF (NKZ>IKPTD) CALL ERROR(' IBZKPT',' NKZ>IKPTD',NKZ)
      END IF

!=======================================================================
! Now run over all mesh points and find out the irreducible points ...
!=======================================================================
      NKPT=0
      NTESTM=0
      DO I3=0,NKZ-1
       DO I2=0,NKY-1
         DO 7 I1=0,NKX-1
! k-point coordinates in the basis BK:
            V(1)=FLOAT(I1)+SHIFT(1)
            V(2)=FLOAT(I2)+SHIFT(2)
            V(3)=FLOAT(I3)+SHIFT(3)
! Take only k-points lying within the Brillouin zone (if we do not use
! the same lattice type for the k-lattice as for the reciprocal lattice
! it might happen that we have to cut some points ...):
            VR(1)=V(1)*BK(1,1)+V(2)*BK(1,2)+V(3)*BK(1,3)
            VR(2)=V(1)*BK(2,1)+V(2)*BK(2,2)+V(3)*BK(2,3)
            VR(3)=V(1)*BK(3,1)+V(2)*BK(3,2)+V(3)*BK(3,3)
! k-point coordinates in basis B:
            V(1)=VR(1)*A(1,1)+VR(2)*A(2,1)+VR(3)*A(3,1)
            V(2)=VR(1)*A(1,2)+VR(2)*A(2,2)+VR(3)*A(3,2)
            V(3)=VR(1)*A(1,3)+VR(2)*A(2,3)+VR(3)*A(3,3)
! shift into 1st Brillouin zone (unit cell of basis B):
            V(1)=MOD(V(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            V(2)=MOD(V(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            V(3)=MOD(V(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(1)=V(1)*B(1,1)+V(2)*B(1,2)+V(3)*B(1,3)
            VR(2)=V(1)*B(2,1)+V(2)*B(2,2)+V(3)*B(2,3)
            VR(3)=V(1)*B(3,1)+V(2)*B(3,2)+V(3)*B(3,3)
! k-point coordinates in basis BK:
            V(1)=VR(1)*AK(1,1)+VR(2)*AK(2,1)+VR(3)*AK(3,1)
            V(2)=VR(1)*AK(1,2)+VR(2)*AK(2,2)+VR(3)*AK(3,2)
            V(3)=VR(1)*AK(1,3)+VR(2)*AK(2,3)+VR(3)*AK(3,3)
! Now 'blow up' the list of points to be tested by symmetrizing ... :
            NTEST=1
            VTEST(1,1)=V(1)
            VTEST(2,1)=V(2)
            VTEST(3,1)=V(3)
            DO 100 IRFULL=1,NRFULL
! Rotate k-point and apply periodic boundary conditions to this k-point:
               VR(1)=V(1)*FLOAT(IFULLG(1,1,IRFULL))+ &
     &                         V(2)*FLOAT(IFULLG(2,1,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,1,IRFULL))
               VR(2)=V(1)*FLOAT(IFULLG(1,2,IRFULL))+ &
     &                         V(2)*FLOAT(IFULLG(2,2,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,2,IRFULL))
               VR(3)=V(1)*FLOAT(IFULLG(1,3,IRFULL))+ &
     &                         V(2)*FLOAT(IFULLG(2,3,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,3,IRFULL))
               T(1)=VR(1)*BK(1,1)+VR(2)*BK(1,2)+VR(3)*BK(1,3)
               T(2)=VR(1)*BK(2,1)+VR(2)*BK(2,2)+VR(3)*BK(2,3)
               T(3)=VR(1)*BK(3,1)+VR(2)*BK(3,2)+VR(3)*BK(3,3)
               VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
               VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
               VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
               VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
               T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
               T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
               VR(1)=T(1)*AK(1,1)+T(2)*AK(2,1)+T(3)*AK(3,1)
               VR(2)=T(1)*AK(1,2)+T(2)*AK(2,2)+T(3)*AK(3,2)
               VR(3)=T(1)*AK(1,3)+T(2)*AK(2,3)+T(3)*AK(3,3)
! Test whether point has already been generated ... :
               IEXIST=0
               DO 90 ITEST=1,NTEST
                  IF ((ABS(VR(1)-VTEST(1,ITEST))<TINY).AND. &
     &                   (ABS(VR(2)-VTEST(2,ITEST))<TINY).AND. &
     &                     (ABS(VR(3)-VTEST(3,ITEST))<TINY)) IEXIST=1
   90          CONTINUE
               IF (IEXIST==0) THEN
                  NTEST=NTEST+1
                  VTEST(1,NTEST)=VR(1)
                  VTEST(2,NTEST)=VR(2)
                  VTEST(3,NTEST)=VR(3)
               END IF
  100       CONTINUE
            NTESTM=MAX(NTESTM,NTEST)
! For all test vectors:
            DO 1000 ITEST=1,NTEST
               V(1)=VTEST(1,ITEST)
               V(2)=VTEST(2,ITEST)
               V(3)=VTEST(3,ITEST)
! Set test flag ...
               IEXIST=0
! For all symmetry operations:
               DO 5 I=1,NROTKM
! Rotate k-point and apply periodic boundary conditions to this k-point:
                  VR(1)=V(1)*G(1,1,I)+V(2)*G(2,1,I)+V(3)*G(3,1,I)
                  VR(2)=V(1)*G(1,2,I)+V(2)*G(2,2,I)+V(3)*G(3,2,I)
                  VR(3)=V(1)*G(1,3,I)+V(2)*G(2,3,I)+V(3)*G(3,3,I)
                  T(1)=VR(1)*BK(1,1)+VR(2)*BK(1,2)+VR(3)*BK(1,3)
                  T(2)=VR(1)*BK(2,1)+VR(2)*BK(2,2)+VR(3)*BK(2,3)
                  T(3)=VR(1)*BK(3,1)+VR(2)*BK(3,2)+VR(3)*BK(3,3)
                  VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
                  VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
                  VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
                  VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
                  T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
                  T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
                  VR(1)=T(1)*AK(1,1)+T(2)*AK(2,1)+T(3)*AK(3,1)
                  VR(2)=T(1)*AK(1,2)+T(2)*AK(2,2)+T(3)*AK(3,2)
                  VR(3)=T(1)*AK(1,3)+T(2)*AK(2,3)+T(3)*AK(3,3)
! Test the rotated k-point against all previously found k-points ... :
                  DO 4 IK=1,NKPT
                     IF ((ABS(VR(1)-VKPT(1,IK))<TINY).AND. &
     &                       (ABS(VR(2)-VKPT(2,IK))<TINY).AND. &
     &                             (ABS(VR(3)-VKPT(3,IK))<TINY)) THEN
! Here we found that the rotated k-point is equivalent to some k-point
! which has already been stored in our list ... ! This means that our
! original k-point is also equivalent to this k-point! So set test flag
! to remark existence and add up the weight for this k-point ... :
                        IEXIST=1
                        WTKPT(IK)=WTKPT(IK)+FLOAT(NRFULL/NTEST)
! It can only be equivalent to one of the previously found k-points:
                        GOTO 6
                     END IF
    4             CONTINUE
    5          CONTINUE
    6          CONTINUE
               IF (IEXIST==0) THEN
! No symmetry operation was able to reproduce any of the k-points found
! previously - so we must have found a new k-point! Count it, store it!
                  NKPT=NKPT+1
                  VKPT(1,NKPT)=V(1)
                  VKPT(2,NKPT)=V(2)
                  VKPT(3,NKPT)=V(3)
! ... and initialize the 'weight counter':
                  WTKPT(NKPT)=FLOAT(NRFULL/NTEST)
               END IF
 1000       CONTINUE
    7    CONTINUE
      ENDDO
      ENDDO
      IF (NKPT>NKDIM) CALL ERROR(' IBZKPT',' NKPT>NKDIM',NKPT)
!=======================================================================
! if LSHIFT_KPOINTS is set we need to include all k-points
! that are differences of two other k-points in the full
! k-point set
!=======================================================================
      NK_SHIFT=0
      IF (LSHIFT_KPOINTS .AND. LNOSYM) THEN
         NKTMP=NKPT
         DO NK=1,NKPT
            DO NK2=1,NKPT
               ! make new k-point and shift (for testing) it to 1st BZ
               VR(1)=VKPT(1,NK)-VKPT(1,NK2)
               VR(2)=VKPT(2,NK)-VKPT(2,NK2)
               VR(3)=VKPT(3,NK)-VKPT(3,NK2)
               ! apply periodic boundary conditions to this k-point:
               T(1)=VR(1)*BK(1,1)+VR(2)*BK(1,2)+VR(3)*BK(1,3)
               T(2)=VR(1)*BK(2,1)+VR(2)*BK(2,2)+VR(3)*BK(2,3)
               T(3)=VR(1)*BK(3,1)+VR(2)*BK(3,2)+VR(3)*BK(3,3)
               VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
               VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
               VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
               VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
               T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
               T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
               T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
               VR(1)=T(1)*AK(1,1)+T(2)*AK(2,1)+T(3)*AK(3,1)
               VR(2)=T(1)*AK(1,2)+T(2)*AK(2,2)+T(3)*AK(3,2)
               VR(3)=T(1)*AK(1,3)+T(2)*AK(2,3)+T(3)*AK(3,3)

               ! test against previous k-points
               IEXIST=0
               DO IK=1,NKTMP
                  IF ((ABS(VR(1)-VKPT(1,IK))<TINY).AND. &
     &                 (ABS(VR(2)-VKPT(2,IK))<TINY).AND. &
     &                 (ABS(VR(3)-VKPT(3,IK))<TINY)) THEN
                     IEXIST=1
                     EXIT
                  ENDIF
               ENDDO
               ! add if it does not exist yet
               IF (IEXIST==0) THEN
                  NK_SHIFT=NK_SHIFT+1
                  NKTMP=NKTMP+1
                  IF (NKTMP>SIZE(VKPT,2)) THEN
                     WRITE(*,*) 'internal error in mkpoints_full.F: K_TMP is not sufficiently large'
                     STOP
                  ENDIF
                  VKPT(1,NKTMP)=VR(1)
                  VKPT(2,NKTMP)=VR(2)
                  VKPT(3,NKTMP)=VR(3)
                  WTKPT(NKTMP) =0
               ENDIF
            ENDDO
         ENDDO
         NKPT=NKTMP
      END IF
!=======================================================================
! Following some code for setting up the "connection table" for the
! k-points on the full mesh with respect to the irreducible points
! if we want to use the tetrahedron integration method ... :
!=======================================================================
      IF (LTET.AND.THALF.AND.(.NOT.TKEEP)) THEN
! Here the 'complicated shifted' meshes leading to some other type
! of bravais lattice for the k-mesh ... (IBRAV=2,5,6,9,10,11,13 only):
         P1(1)=BK(1,1)
         P1(2)=BK(2,1)
         P1(3)=BK(3,1)
         P2(1)=BK(1,2)
         P2(2)=BK(2,2)
         P2(3)=BK(3,2)
         P3(1)=BK(1,3)
         P3(2)=BK(2,3)
         P3(3)=BK(3,3)
! Get the lattice type of the mesh generated by this shift ... :
         V(1)=MOD(SHIFT(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         V(2)=MOD(SHIFT(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         V(3)=MOD(SHIFT(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         NTEST=1
         TESTV(1,1)=V(1)
         TESTV(1,2)=V(2)
         TESTV(1,3)=V(3)
         DO 200 IRFULL=1,NRFULL
! Rotate shift and apply periodic boundary conditions to rotated shift:
            VR(1)=V(1)*FLOAT(IFULLG(1,1,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,1,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,1,IRFULL))
            VR(2)=V(1)*FLOAT(IFULLG(1,2,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,2,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,2,IRFULL))
            VR(3)=V(1)*FLOAT(IFULLG(1,3,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,3,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,3,IRFULL))
            VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
! Test whether point has already been generated ... :
            IEXIST=0
            DO 190 ITEST=1,NTEST
               IF ((ABS(VR(1)-TESTV(ITEST,1))<TINY).AND. &
     &                 (ABS(VR(2)-TESTV(ITEST,2))<TINY).AND. &
     &                     (ABS(VR(3)-TESTV(ITEST,3))<TINY)) IEXIST=1
  190       CONTINUE
            IF (IEXIST==0) THEN
               NTEST=NTEST+1
               TESTV(NTEST,1)=VR(1)
               TESTV(NTEST,2)=VR(2)
               TESTV(NTEST,3)=VR(3)
            END IF
  200    CONTINUE
         NA(1)=NTEST
         CALL PRICEL(IBRAV,CDIMK,P1,P2,P3,TESTV,BKT(1,1),BKT(1,2), &
     &   BKT(1,3),PWORK,NCELL,IMTYP,CDIMR,1,1,NA,48,TWORK,IWORK,WORK,-1)
! Cross check (debugging ...):
         IF (NTEST/=NCELL) CALL ERROR(' IBZKPT', &
     &   ' Fatal error in detecting k-mesh type!',NTEST*100+NCELL-NTEST)
         IF (IMTYP/=IBRAVK) CALL ERROR(' IBZKPT', &
     &   ' Fatal error detecting k-mesh type!',IMTYP*10000+IMTYP-IBRAVK)
! Rough estimate: how many mesh-points do we need in each direction?
         BK1A1=BKT(1,1)*A(1,1)+BKT(2,1)*A(2,1)+BKT(3,1)*A(3,1)
         BK2A1=BKT(1,1)*A(1,2)+BKT(2,1)*A(2,2)+BKT(3,1)*A(3,2)
         BK3A1=BKT(1,1)*A(1,3)+BKT(2,1)*A(2,3)+BKT(3,1)*A(3,3)
         BK1A2=BKT(1,2)*A(1,1)+BKT(2,2)*A(2,1)+BKT(3,2)*A(3,1)
         BK2A2=BKT(1,2)*A(1,2)+BKT(2,2)*A(2,2)+BKT(3,2)*A(3,2)
         BK3A2=BKT(1,2)*A(1,3)+BKT(2,2)*A(2,3)+BKT(3,2)*A(3,3)
         BK1A3=BKT(1,3)*A(1,1)+BKT(2,3)*A(2,1)+BKT(3,3)*A(3,1)
         BK2A3=BKT(1,3)*A(1,2)+BKT(2,3)*A(2,2)+BKT(3,3)*A(3,2)
         BK3A3=BKT(1,3)*A(1,3)+BKT(2,3)*A(2,3)+BKT(3,3)*A(3,3)
         IF (ABS(BK1A1)<TINY) BK1A1=1.E30_q
         IF (ABS(BK1A2)<TINY) BK1A2=1.E30_q
         IF (ABS(BK1A3)<TINY) BK1A3=1.E30_q
         IF (ABS(BK2A1)<TINY) BK2A1=1.E30_q
         IF (ABS(BK2A2)<TINY) BK2A2=1.E30_q
         IF (ABS(BK2A3)<TINY) BK2A3=1.E30_q
         IF (ABS(BK3A1)<TINY) BK3A1=1.E30_q
         IF (ABS(BK3A2)<TINY) BK3A2=1.E30_q
         IF (ABS(BK3A3)<TINY) BK3A3=1.E30_q
         NKX=NINT(MAX(MAX(1._q/ABS(BK1A1),1._q/ABS(BK2A1)),1._q/ABS(BK3A1)))
         NKY=NINT(MAX(MAX(1._q/ABS(BK1A2),1._q/ABS(BK2A2)),1._q/ABS(BK3A2)))
         NKZ=NINT(MAX(MAX(1._q/ABS(BK1A3),1._q/ABS(BK2A3)),1._q/ABS(BK3A3)))
! Take (one of) the correct symmetry conserving shift(s):
         NKXM=MOD(NKX,2)
         NKYM=MOD(NKY,2)
         NKZM=MOD(NKZ,2)
         SHIFT(1)=0.5_q*NKXM
         SHIFT(2)=0.5_q*NKYM
         SHIFT(3)=0.5_q*NKZM
! Check (debugging):
         NCODE=100*NKXM+10*NKYM+NKZM
         IF ((IBRAVK==1).AND. &
     &   ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &       CALL ERROR(' IBZKPT',' Could not get correct shifts',NCODE)
         IF ((IBRAVK==5).AND. &
     &   ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &   ((ABS(SHTEST(3)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(1))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &       CALL ERROR(' IBZKPT',' Could not get correct shifts',NCODE)
! Get the symmetry operations represented in the basis of the BKTs:
         CALL SGRTRF(G,S,NROTKM,BK(1,1),BK(1,2),BK(1,3), &
     &                                       BKT(1,1),BKT(1,2),BKT(1,3))
      ELSE IF (LTET) THEN
! Meshes with symmetry conserving shifts - take what we have already:
         BKT(1,1)=BK(1,1)
         BKT(2,1)=BK(2,1)
         BKT(3,1)=BK(3,1)
         BKT(1,2)=BK(1,2)
         BKT(2,2)=BK(2,2)
         BKT(3,2)=BK(3,2)
         BKT(1,3)=BK(1,3)
         BKT(2,3)=BK(2,3)
         BKT(3,3)=BK(3,3)
         DO 300 IROT=1,NROTKM
            S(1,1,IROT)=G(1,1,IROT)
            S(2,1,IROT)=G(2,1,IROT)
            S(3,1,IROT)=G(3,1,IROT)
            S(1,2,IROT)=G(1,2,IROT)
            S(2,2,IROT)=G(2,2,IROT)
            S(3,2,IROT)=G(3,2,IROT)
            S(1,3,IROT)=G(1,3,IROT)
            S(2,3,IROT)=G(2,3,IROT)
            S(3,3,IROT)=G(3,3,IROT)
  300    CONTINUE
      END IF
      IF (LTET.AND.(.NOT.TZERO)) THEN
! We may only have shifts +/-0.5 and zero: -0.5 and +0.5 are identical,
! for getting routine TETIRR working properly take always shifts +0.5!
         IF (ABS(SHTEST(1)-0.5_q)<TINY) SHIFT(1)=0.5_q
         IF (ABS(SHTEST(2)-0.5_q)<TINY) SHIFT(2)=0.5_q
         IF (ABS(SHTEST(3)-0.5_q)<TINY) SHIFT(3)=0.5_q
      END IF
      IF (LTET) THEN
! Testing dimensions of array IKPT ... :
         IF (NKX>IKPTD) CALL ERROR(' IBZKPT',' NKX>IKPTD',NKX)
         IF (NKY>IKPTD) CALL ERROR(' IBZKPT',' NKY>IKPTD',NKY)
         IF (NKZ>IKPTD) CALL ERROR(' IBZKPT',' NKZ>IKPTD',NKZ)
! Now pass through the mesh and get the 'connection table' ... :
         DO I3=0,NKZ-1
          DO I2=0,NKY-1
            DO 700 I1=0,NKX-1
! k-point coordinates in the basis BKT:
               V(1)=FLOAT(I1)+SHIFT(1)
               V(2)=FLOAT(I2)+SHIFT(2)
               V(3)=FLOAT(I3)+SHIFT(3)
! Set test flag ...
               IEXIST=0
! For all symmetry operations:
               DO 500 I=1,NROTKM
! Rotate k-point and apply periodic boundary conditions to this k-point:
                  VR(1)=V(1)*S(1,1,I)+V(2)*S(2,1,I)+V(3)*S(3,1,I)
                  VR(2)=V(1)*S(1,2,I)+V(2)*S(2,2,I)+V(3)*S(3,2,I)
                  VR(3)=V(1)*S(1,3,I)+V(2)*S(2,3,I)+V(3)*S(3,3,I)
                  T(1)=VR(1)*BKT(1,1)+VR(2)*BKT(1,2)+VR(3)*BKT(1,3)
                  T(2)=VR(1)*BKT(2,1)+VR(2)*BKT(2,2)+VR(3)*BKT(2,3)
                  T(3)=VR(1)*BKT(3,1)+VR(2)*BKT(3,2)+VR(3)*BKT(3,3)
                  VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
                  VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
                  VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
                  VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
                  T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
                  T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
! Now, here we take the coordinates in the BK-basis ... !
                  VR(1)=T(1)*AK(1,1)+T(2)*AK(2,1)+T(3)*AK(3,1)
                  VR(2)=T(1)*AK(1,2)+T(2)*AK(2,2)+T(3)*AK(3,2)
                  VR(3)=T(1)*AK(1,3)+T(2)*AK(2,3)+T(3)*AK(3,3)
! Test the rotated k-point against all previously found k-points ... :
                  DO 400 IK=1,NKPT
                     IF ((ABS(VR(1)-VKPT(1,IK))<TINY).AND. &
     &                       (ABS(VR(2)-VKPT(2,IK))<TINY).AND. &
     &                             (ABS(VR(3)-VKPT(3,IK))<TINY)) THEN
! Here we found that the rotated k-point is equivalent to some k-point:
                        IEXIST=1
                        IKPT(I1+1,I2+1,I3+1)=IK
! It can only be equivalent to one of the previously found k-points:
                        GOTO 600
                     END IF
  400             CONTINUE
  500          CONTINUE
  600          CONTINUE
               IF (IEXIST==0) THEN
! No symmetry operation was able to reproduce any of the k-points found
! previously - oh folks, this means very bad luck. Fatal error ... :
                  NCODE=I3*10000+I2*100+I1
                  CALL ERROR(' IBZKPT', &
     &                   ' Fatal error: unable to match k-point!',NCODE)
               END IF
  700       CONTINUE
          ENDDO
         ENDDO
      END IF

!=======================================================================
! last task to do: transform the coordinates from basis BK to basis B
! and supply the correct scaling to the weighting factors ...
!=======================================================================
      WEISUM=0._q
      DO 9 IK=1,NKPT
         V(1)=VKPT(1,IK)*BK(1,1)+VKPT(2,IK)*BK(1,2)+VKPT(3,IK)*BK(1,3)
         V(2)=VKPT(1,IK)*BK(2,1)+VKPT(2,IK)*BK(2,2)+VKPT(3,IK)*BK(2,3)
         V(3)=VKPT(1,IK)*BK(3,1)+VKPT(2,IK)*BK(3,2)+VKPT(3,IK)*BK(3,3)
! k-point coordinates in basis B:
         VKPT(1,IK)=V(1)*A(1,1)+V(2)*A(2,1)+V(3)*A(3,1)
         VKPT(2,IK)=V(1)*A(1,2)+V(2)*A(2,2)+V(3)*A(3,2)
         VKPT(3,IK)=V(1)*A(1,3)+V(2)*A(2,3)+V(3)*A(3,3)
! Sum up weights:
         WEISUM=WEISUM+WTKPT(IK)
    9 CONTINUE

! ... and very last task is to print the result (if desired):
      IF ((IU6>=0).AND.(IU6<=99)) THEN
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Subroutine IBZKPT returns following result:'
         WRITE(IU6,'(A)') ' ==========================================='
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A,I6,A)') ' Found ',NKPT,' irreducible k-points:'
         IF (NK_SHIFT>0) THEN
            WRITE(IU6,'(A,I6,A)') ' Added ',NK_SHIFT,' k-points corresponding to difference vectors'
         ENDIF
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Following reciprocal coordinates:'
         WRITE(IU6,'(A)') '            Coordinates               Weight'
         DO 11 IK=1,NKPT
            WRITE(IU6,'(3F10.6,4X,F10.6)') &
     &                        VKPT(1,IK),VKPT(2,IK),VKPT(3,IK),WTKPT(IK)
   11    CONTINUE
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Following cartesian coordinates:'
         WRITE(IU6,'(A)') '            Coordinates               Weight'
         DO 12 IK=1,NKPT
            V(1)=VKPT(1,IK)*B(1,1)+VKPT(2,IK)*B(1,2)+VKPT(3,IK)*B(1,3)
            V(2)=VKPT(1,IK)*B(2,1)+VKPT(2,IK)*B(2,2)+VKPT(3,IK)*B(2,3)
            V(3)=VKPT(1,IK)*B(3,1)+VKPT(2,IK)*B(3,2)+VKPT(3,IK)*B(3,3)
            V(1)=V(1)*SCALE
            V(2)=V(2)*SCALE
            V(3)=V(3)*SCALE
            WRITE(IU6,'(3F10.6,4X,F10.6)') V(1),V(2),V(3),WTKPT(IK)
   12    CONTINUE
         WRITE(IU6,'(A)') ' '
      END IF

! Result on file IBZKPT (can be copied to KPOINTS for following runs):
      IU=-1
! The IBZKPT is not written for the MPI version, because would
! be a little bit difficult to guarantee in all cases that only
! one proc writes the file
      IF (IU6>=0) THEN

      DO 1001 I=0,99
         IF ((I==0).OR.(I==5).OR.(I==6)) GOTO 1001
         INQUIRE(UNIT=I,OPENED=OCCUP)
         IF (.NOT.OCCUP) THEN
            IU=I
            GOTO 1002
         END IF
 1001 CONTINUE
 1002 CONTINUE
      IF (IU<0) THEN
         WRITE(*,*) &
     &      'Sorry: No IO-unit available! Cannot save data from IBZKPT!'
         GOTO 1009
      END IF
      OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'IBZKPT')
      WRITE(IU,'(A)') 'Automatically generated mesh'
      WRITE(IU,'(I8)') NKPT
      WRITE(IU,'(A)') 'Reciprocal lattice'
      DO 1005 IK=1,NKPT
         WRITE(IU,'(3F20.14,4X,I10)') VKPT(1,IK),VKPT(2,IK),VKPT(3,IK), &
     &                                NINT(WTKPT(IK))
 1005 CONTINUE
 1009 CONTINUE

      ENDIF
! Correct scaling for weights ... :
      DO  IK=1,NKPT
         WTKPT(IK)=WTKPT(IK)/WEISUM
      ENDDO

! Find inequivalent tetrahedra for tetrahedron integration method ... :
      IF (LTET) THEN
         IF (NKPT<4) CALL ERROR(' IBZKPT', &
     &           ' Tetrahedron method fails for NKPT<4. NKPT =',NKPT)
         CALL TETIRR(NTET,IDTET,NTETD,BK,NKX,NKY,NKZ,IKPT,IKPTD,IU6)
! Get the volume weight of each tetrahedron (according to the full mesh)
! where the factor 6 accounts for the fact that around each k-point we
! have examined a microcell giving 6 tetrahedra and the factor 2. will
! account for spin-degeneracy (and must not be included in the routines
! which perform the BZ-integrations!!! - just provide VOLWGT, do not add
! any changes, take it as is ...).
         VOLWGT=1._q/FLOAT(6*NKX*NKY*NKZ)

! Result on file IBZKPT:
! The IBZKPT is not written for the MPI version, because would
! be a little bit difficult to guarantee in all cases that only
! one proc writes the file
         IF (IU>=0) THEN
            WRITE(IU,'(A)') 'Tetrahedra'
            WRITE(IU,'(I10,F20.14)') NTET,VOLWGT
            DO 1007 ITET=1,NTET
               WRITE(IU,'(5I10)') (IDTET(KTET,ITET),KTET=0,4)
 1007       CONTINUE
         END IF
      END IF
      IF (IU>=0) CLOSE(IU)

      RETURN
      END


!******************** SUBROUTINE IBZKPT_FAST ***************************
!
! subroutine ibzkpt_fast reduces given k-meshs in the full 1st Brillouin
! zone to its irreducible part (setting also the correct weights) ...; in
! contrast to subroutine ibzkpt it accepts only special shifts (0,+/-0.5)!
! A significant speedup should be achieved for very large meshes due to
! the use of a connection table (only used for the tetrahedron generation
! in subroutine ibzkpt above) which avoids a lot of "search operations"
!
! presently not supported!
!
!
! this routine needs the reciprocal lattice vectors (B) [cartesian]
!                    the basis vectors of the k-lattice (BK) [cartesian]
!                    the shift of the k-lattice (SHIFT) [BK-coordinates]
!                    the dimension parameter NKDIM for arrays VKPT,WTKPT
!                    the dimension parameter NTETD for array IDTET
!                    the dimension parameter IKPTD for array IKPT
!                    a switch for usage of the tetrahedron method (LTET)
!                    the 'scaling factor' for the Bs (SCALE)
!                    and the unit where to write informations (IU6)
! it returns:        the number of irreducible k-points (NKPT)
!                    the k-point coordinates (VKPT) [B-coordinates]
!                    and the weighting factors for each k-point (WTKPT)
!                    the k-point 'connection table' (IKPT)
! and if LTET=T      the number of irreducible tetrahedra (NTET)
!                    the corners of the tetrahedra [IDTET(1-4,...)]
!                    the volume weight factor of a tetrahedron (VOLWGT)
!                    the degeneracy of each tetrahedron [IDTET(0,...)]
!                    and the corners of the tetrahedra [IDTET(1-4,...)]
!
!***********************************************************************

      SUBROUTINE IBZKPT_FAST(B,BK,SHIFT,NKPT,VKPT,WTKPT,NKDIM, &
     &                LTET,NTET,IDTET,NTETD,VOLWGT,IKPT,IKPTD,SCALE,IU6)
      USE prec
      USE main_mpi
#ifndef oldsym
      USE spinsym
#endif

      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL SGREQL,TKEEP,TZERO,THALF,LTET,OCCUP

      DIMENSION S(3,3,48),G(3,3,48),KGRPOP(3,3,48),INVERS(9),CDIMK(6)
      DIMENSION A(3,3),AK(3,3),V(3),VR(3),P1(3),P2(3),P3(3),CDIMR(6)
      DIMENSION B(3,3),BK(3,3),SHIFT(3),VKPT(3,NKDIM),WTKPT(NKDIM),T(3)
      DIMENSION PINV(3,3),SHTEST(3),IFULLS(3,3,48),ISWRK(3,3,48),NA(1)
      DIMENSION GWRK(3,48),TAUFUL(1,3,1),TRFULL(1,3,1),SWRK(1),INDEX(1)
      DIMENSION IFULLG(3,3,48),VTEST(3,48),IDTET(0:4,NTETD),BKT(3,3)
      DIMENSION IKPT(IKPTD,IKPTD,IKPTD),TESTV(48,3),PB1(3),PB2(3),PB3(3)
      DIMENSION TWORK(48,3),PWORK(50,3),WORK(50,3),IWORK(50),AKT(3,3)

      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      SAVE TINY,INVERS
      DATA TINY /1.E-6_q/, INVERS /-1,0,0,0,-1,0,0,0,-1/

! First check: The generating cell must have positive volume:
      CALL CELVOL(BK(1,1),BK(1,2),BK(1,3),OMEGBK)
      IF (OMEGBK<(TINY*TINY*TINY)) CALL ERROR(' IBZKPT', &
     &               ' Volume of generating cell for k-mesh is zero!',3)
      IF (OMEGBK<0._q) THEN
         IF (IU6>=0) THEN
         WRITE(IU6,*) 'Warning IBZKPT: ', &
     &             'Generating cell for k-mesh has negative volume ...'
         ENDIF
         BK(1,1)=-BK(1,1)
         BK(2,1)=-BK(2,1)
         BK(3,1)=-BK(3,1)
         BK(1,2)=-BK(1,2)
         BK(2,2)=-BK(2,2)
         BK(3,2)=-BK(3,2)
         BK(1,3)=-BK(1,3)
         BK(2,3)=-BK(2,3)
         BK(3,3)=-BK(3,3)
      ENDIF
! We must only consider shifts in the interval (-0.5,0.5] ... :
      SHIFT(1)=MOD(SHIFT(1)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      SHIFT(2)=MOD(SHIFT(2)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      SHIFT(3)=MOD(SHIFT(3)+60.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
      TZERO=((ABS(SHIFT(1))<TINY).AND. &
     &              (ABS(SHIFT(2))<TINY).AND.(ABS(SHIFT(3))<TINY))
!=======================================================================
! Consistency check: we allow usage of other k-mesh lattice types than
! the lattice type of the reciprocal lattice (for example bcc-reciprocal
! lattice but simple cubic k-mesh ...). But two restrictions apply here:
! both types of lattices should at least belong to the same class of
! bravais lattices (cubic, tetragonal, orthorhombic, monoclinic, ...)
! or to some higher class which is a special case of the given class -
! that means we may mix sc,fcc,bcc but not bcc and monoclinic or so ...
! - and they should be commensurable (that means the reciprocal lattice
! vectors represented in the basis of the k-lattice should have INTEGER
! components!). Otherwise PRINT ERROR MESSAGE and STOP EXECUTION !!!!!
!=======================================================================
      CALL RECIPS(1._q,B(1,1),B(1,2),B(1,3),A(1,1),A(1,2),A(1,3))
      CALL RECIPS(1._q,BK(1,1),BK(1,2),BK(1,3),AK(1,1),AK(1,2),AK(1,3))
      V(1)=B(1,1)*AK(1,1)+B(2,1)*AK(2,1)+B(3,1)*AK(3,1)
      V(2)=B(1,1)*AK(1,2)+B(2,1)*AK(2,2)+B(3,1)*AK(3,2)
      V(3)=B(1,1)*AK(1,3)+B(2,1)*AK(2,3)+B(3,1)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 1st reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',1)
      V(1)=B(1,2)*AK(1,1)+B(2,2)*AK(2,1)+B(3,2)*AK(3,1)
      V(2)=B(1,2)*AK(1,2)+B(2,2)*AK(2,2)+B(3,2)*AK(3,2)
      V(3)=B(1,2)*AK(1,3)+B(2,2)*AK(2,3)+B(3,2)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 2nd reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',2)
      V(1)=B(1,3)*AK(1,1)+B(2,3)*AK(2,1)+B(3,3)*AK(3,1)
      V(2)=B(1,3)*AK(1,2)+B(2,3)*AK(2,2)+B(3,3)*AK(3,2)
      V(3)=B(1,3)*AK(1,3)+B(2,3)*AK(2,3)+B(3,3)*AK(3,3)
      IF ((ABS(V(1)-NINT(V(1)))>TINY).OR. &
     &        (ABS(V(2)-NINT(V(2)))>TINY).OR. &
     &              (ABS(V(3)-NINT(V(3)))>TINY)) &
     &                 CALL ERROR(' IBZKPT',' 3rd reciprocal lattice '// &
     &                     'vector is incommensurable with k-lattice',3)
      PB1(1)=B(1,1)
      PB1(2)=B(2,1)
      PB1(3)=B(3,1)
      PB2(1)=B(1,2)
      PB2(2)=B(2,2)
      PB2(3)=B(3,2)
      PB3(1)=B(1,3)
      PB3(2)=B(2,3)
      PB3(3)=B(3,3)
      CALL LATTYP(PB1,PB2,PB3,IBRAVR,CDIMR,100)
      IBRAV=IBRAVR
! All cubic types (matched onto simple cubic):
      IF (IBRAVR<=3) IBRAVR=1
! All tetragonal types (matched onto simple tetragonal):
      IF ((IBRAVR<=6).AND.(IBRAVR>=5)) IBRAVR=5
! Trigonal (rhomboedric) belong to the hexagonal system ... :
      IF (IBRAVR==7) IBRAVR=4
! All orthorhombic types (matched onto simple orthorhombic):
      IF ((IBRAVR<=11).AND.(IBRAVR>=8)) IBRAVR=8
! All monoclinic types (matched onto simple monoclinic):
      IF ((IBRAVR<=13).AND.(IBRAVR>=12)) IBRAVR=12
      P1(1)=BK(1,1)
      P1(2)=BK(2,1)
      P1(3)=BK(3,1)
      P2(1)=BK(1,2)
      P2(2)=BK(2,2)
      P2(3)=BK(3,2)
      P3(1)=BK(1,3)
      P3(2)=BK(2,3)
      P3(3)=BK(3,3)
      CALL LATTYP(P1,P2,P3,IBRAVK,CDIMK,100)
! All cubic types (matched onto simple cubic):
      IF (IBRAVK<=3) IBRAVK=1
! All tetragonal types (matched onto simple tetragonal):
      IF ((IBRAVK<=6).AND.(IBRAVK>=5)) IBRAVK=5
! Trigonal (rhomboedric) belong to the hexagonal system ... :
      IF (IBRAVK==7) IBRAVK=4
! All orthorhombic types (matched onto simple orthorhombic):
      IF ((IBRAVK<=11).AND.(IBRAVK>=8)) IBRAVK=8
! All monoclinic types (matched onto simple monoclinic):
      IF ((IBRAVK<=13).AND.(IBRAVK>=12)) IBRAVK=12
! Now it is not yet the full truth ... : some lattice types can be very
! special cases of other lattice types (cubic is also tetragonal or
! tetragonal is also orthorhombic, 'all is triclinic' ...).
      IF ((IBRAVR<IBRAVK).OR.(((IBRAVR==4).OR.(IBRAVR>=12)).AND. &
     &  (IBRAVR/=IBRAVK))) CALL WARN(' IBZKPT',' Reciprocal lattice ' &
     &         //'and k-lattice belong to different class of lattices.', &
     &                                                    IBRAVR*IBRAVK)

!=======================================================================
! Test for symmetry conserving shifts!
!=======================================================================
      TKEEP=.TRUE.
      CALL RECIPS(1._q,P1,P2,P3,PINV(1,1),PINV(1,2),PINV(1,3))
      T(1)=SHIFT(1)*BK(1,1)+SHIFT(2)*BK(1,2)+SHIFT(3)*BK(1,3)
      T(2)=SHIFT(1)*BK(2,1)+SHIFT(2)*BK(2,2)+SHIFT(3)*BK(2,3)
      T(3)=SHIFT(1)*BK(3,1)+SHIFT(2)*BK(3,2)+SHIFT(3)*BK(3,3)
      SHTEST(1)=T(1)*PINV(1,1)+T(2)*PINV(2,1)+T(3)*PINV(3,1)
      SHTEST(2)=T(1)*PINV(1,2)+T(2)*PINV(2,2)+T(3)*PINV(3,2)
      SHTEST(3)=T(1)*PINV(1,3)+T(2)*PINV(2,3)+T(3)*PINV(3,3)
! Periodic boundary conditions: INTEGERS -> 0, +/- HALFS -> 0.5
      SHTEST(1)=MOD(SHTEST(1)+6.25_q,1._q)-0.25_q
      SHTEST(2)=MOD(SHTEST(2)+6.25_q,1._q)-0.25_q
      SHTEST(3)=MOD(SHTEST(3)+6.25_q,1._q)-0.25_q
! General test: We may only have the values 0.0 or 0.5 ... :
      IF (((ABS(SHTEST(1))>TINY).AND.(ABS(SHTEST(1)-0.5_q)>TINY)) &
     & .OR.((ABS(SHTEST(2))>TINY).AND.(ABS(SHTEST(2)-0.5_q)>TINY)) &
     & .OR.((ABS(SHTEST(3))>TINY).AND.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      TZERO=TZERO.AND.TKEEP
      THALF=TKEEP.AND.(.NOT.TZERO)
! There are no restrictions for IBRAV=8,12,14 - but for other IBRAV:
      IF (((IBRAV==1).OR.(IBRAV==3).OR.(IBRAV==6).OR. &
     &                     (IBRAV==7).OR.(IBRAV==10)).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==2).AND.((ABS(SHTEST(1))>TINY).OR. &
     &  (ABS(SHTEST(2))>TINY).OR.(ABS(SHTEST(3))>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==4).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  ((ABS(SHTEST(3))>TINY).AND.(ABS(SHTEST(3)-0.5_q)>TINY)))) &
     &                                                     TKEEP=.FALSE.
      IF (((IBRAV==5).OR.(IBRAV==11).OR.(IBRAV==13)).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &  ((ABS(SHTEST(3)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(1))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.
      IF ((IBRAV==9).AND. &
     &  ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &  ((ABS(SHTEST(1)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &  (ABS(SHTEST(3)-0.5_q)>TINY)).AND.((ABS(SHTEST(1))>TINY).OR. &
     &  (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &                                                     TKEEP=.FALSE.

!=======================================================================
! Following some tests for setting up the k-mesh: to get a simple scheme
! we must restrict ourself to special shifts of the k-mesh because for
! arbitrary shifts it is very hard to implement an efficient scheme ... .
!=======================================================================
      IF ((IBRAV==1).OR.(IBRAV==3).OR.(IBRAV==7)) &
     &     THALF=THALF.AND.(ABS(SHTEST(1)-0.5_q)<TINY).AND. &
     &     (ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3)-0.5_q)<TINY)
      IF (IBRAV==2) THALF=THALF.AND.(((ABS(SHTEST(1)-0.5_q)<TINY) &
     & .AND.(ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3)-0.5_q)< &
     & TINY)).OR.((ABS(SHTEST(1)-0.5_q)<TINY).AND.(ABS(SHTEST(2))< &
     & TINY).AND.(ABS(SHTEST(3))<TINY)).OR.((ABS(SHTEST(1))<TINY) &
     &  .AND.(ABS(SHTEST(2)-0.5_q)<TINY).AND.(ABS(SHTEST(3))<TINY)) &
     &  .OR.((ABS(SHTEST(1))<TINY).AND.(ABS(SHTEST(2))<TINY).AND. &
     &                                    (ABS(SHTEST(3)-0.5_q)<TINY)))
      IF (IBRAV==4) THALF=THALF.AND.(ABS(SHTEST(1))<TINY).AND. &
     &         (ABS(SHTEST(2))<TINY).AND.(ABS(SHTEST(3)-0.5_q)<TINY)
      IF ((.NOT.TZERO).AND.(.NOT.THALF)) THEN
! In order to avoid a hard error here, go back to the old slow implementation
        IKPT=-1
        WRITE(*,*) 'IBZKP_FAST presently not supported'
        STOP
!        CALL IBZKPT(B,BK,SHIFT,NKPT,VKPT,WTKPT,NKDIM, &
!     &              LTET,NTET,IDTET,NTETD,VOLWGT,IKPT,IKPTD,SCALE,IU6)
        RETURN
    ENDIF

      IF (.NOT.TKEEP) THEN
! Get 'full' symmetry of bare (unshifted) k-lattice/k-lattice unit-cell:
         TAUFUL(1,1,1)=0._q
         TAUFUL(1,2,1)=0._q
         TAUFUL(1,3,1)=0._q
         NA(1)=1
         CALL SETGRP(ISWRK,IFULLS,GWRK,NDUM,NDUMK,TAUFUL,IBRAV,NRFULL, &
     &                                   TRFULL,1,1,NA,1,INDEX,SWRK,100)
         CALL SGRCON(IFULLS,IFULLG,NRFULL, &
     &                              PB1,PB2,PB3,BK(1,1),BK(1,2),BK(1,3))
      ELSE
! Symmetry conserving shift needs no further 'mesh-expansion':
         NRFULL=1
         IFULLG(1,1,1)=1
         IFULLG(2,1,1)=0
         IFULLG(3,1,1)=0
         IFULLG(1,2,1)=0
         IFULLG(2,2,1)=1
         IFULLG(3,2,1)=0
         IFULLG(1,3,1)=0
         IFULLG(2,3,1)=0
         IFULLG(3,3,1)=1
      END IF
#ifndef oldsym
      CALL SET_SPINROT(A, B, ISYMOP, NROTK, GTRANS, IU6)
#endif
!=======================================================================
! We may always use inversion (also if it is no real symmetry operation
! of the atomic lattice) - this is due to time-reversal symmetry ... :
!=======================================================================
      NROTKM=NROTK
      INVYES=0
      DO 1 I=1,NROTK
! Test existence of inversion operator:
         IF (SGREQL(IGRPOP(1,1,I),INVERS)) INVYES=1
! ... and copy the symmetry operators to some temporary array ... :
         KGRPOP(1,1,I)=IGRPOP(1,1,I)
         KGRPOP(2,1,I)=IGRPOP(2,1,I)
         KGRPOP(3,1,I)=IGRPOP(3,1,I)
         KGRPOP(1,2,I)=IGRPOP(1,2,I)
         KGRPOP(2,2,I)=IGRPOP(2,2,I)
         KGRPOP(3,2,I)=IGRPOP(3,2,I)
         KGRPOP(1,3,I)=IGRPOP(1,3,I)
         KGRPOP(2,3,I)=IGRPOP(2,3,I)
         KGRPOP(3,3,I)=IGRPOP(3,3,I)
    1 CONTINUE
! If inversion is not yet included add inversion!
      IF (INVYES==0) THEN
         DO 2 I=1,NROTK
! ... this means in full consequence multiply all operators with I ... :
            CALL SGRPRD(INVERS,KGRPOP(1,1,I),KGRPOP(1,1,I+NROTK))
    2    CONTINUE
! ... so we have now a group with twice as much symmetry operations:
         NROTKM=2*NROTK
      END IF

!=======================================================================
! First transform the symmetry operators to the representation for the
! basis generating the k-mesh (basis BK):
!=======================================================================
      DO 3 I=1,NROTKM
         S(1,1,I)=FLOAT(KGRPOP(1,1,I))
         S(2,1,I)=FLOAT(KGRPOP(2,1,I))
         S(3,1,I)=FLOAT(KGRPOP(3,1,I))
         S(1,2,I)=FLOAT(KGRPOP(1,2,I))
         S(2,2,I)=FLOAT(KGRPOP(2,2,I))
         S(3,2,I)=FLOAT(KGRPOP(3,2,I))
         S(1,3,I)=FLOAT(KGRPOP(1,3,I))
         S(2,3,I)=FLOAT(KGRPOP(2,3,I))
         S(3,3,I)=FLOAT(KGRPOP(3,3,I))
    3 CONTINUE
      CALL SGRTRF(S,G,NROTKM,B(1,1),B(1,2),B(1,3), &
     &                                          BK(1,1),BK(1,2),BK(1,3))

!=======================================================================
! Rough estimate: how many mesh-points do we need in each direction?
!=======================================================================
      BK1A1=BK(1,1)*A(1,1)+BK(2,1)*A(2,1)+BK(3,1)*A(3,1)
      BK2A1=BK(1,1)*A(1,2)+BK(2,1)*A(2,2)+BK(3,1)*A(3,2)
      BK3A1=BK(1,1)*A(1,3)+BK(2,1)*A(2,3)+BK(3,1)*A(3,3)
      BK1A2=BK(1,2)*A(1,1)+BK(2,2)*A(2,1)+BK(3,2)*A(3,1)
      BK2A2=BK(1,2)*A(1,2)+BK(2,2)*A(2,2)+BK(3,2)*A(3,2)
      BK3A2=BK(1,2)*A(1,3)+BK(2,2)*A(2,3)+BK(3,2)*A(3,3)
      BK1A3=BK(1,3)*A(1,1)+BK(2,3)*A(2,1)+BK(3,3)*A(3,1)
      BK2A3=BK(1,3)*A(1,2)+BK(2,3)*A(2,2)+BK(3,3)*A(3,2)
      BK3A3=BK(1,3)*A(1,3)+BK(2,3)*A(2,3)+BK(3,3)*A(3,3)
      IF (ABS(BK1A1)<TINY) BK1A1=1.E30_q
      IF (ABS(BK1A2)<TINY) BK1A2=1.E30_q
      IF (ABS(BK1A3)<TINY) BK1A3=1.E30_q
      IF (ABS(BK2A1)<TINY) BK2A1=1.E30_q
      IF (ABS(BK2A2)<TINY) BK2A2=1.E30_q
      IF (ABS(BK2A3)<TINY) BK2A3=1.E30_q
      IF (ABS(BK3A1)<TINY) BK3A1=1.E30_q
      IF (ABS(BK3A2)<TINY) BK3A2=1.E30_q
      IF (ABS(BK3A3)<TINY) BK3A3=1.E30_q
      NKX=NINT(MAX(MAX(1._q/ABS(BK1A1),1._q/ABS(BK2A1)),1._q/ABS(BK3A1)))
      NKY=NINT(MAX(MAX(1._q/ABS(BK1A2),1._q/ABS(BK2A2)),1._q/ABS(BK3A2)))
      NKZ=NINT(MAX(MAX(1._q/ABS(BK1A3),1._q/ABS(BK2A3)),1._q/ABS(BK3A3)))

!=======================================================================
! Now run over all mesh points and find out the irreducible points ...
!=======================================================================
      IF (THALF.AND.(.NOT.TKEEP)) THEN
! Here the 'complicated shifted' meshes leading to some other type
! of bravais lattice for the k-mesh ... (IBRAV=2,5,6,9,10,11,13 only):
         P1(1)=BK(1,1)
         P1(2)=BK(2,1)
         P1(3)=BK(3,1)
         P2(1)=BK(1,2)
         P2(2)=BK(2,2)
         P2(3)=BK(3,2)
         P3(1)=BK(1,3)
         P3(2)=BK(2,3)
         P3(3)=BK(3,3)
! Get the lattice type of the mesh generated by this shift ... :
         V(1)=MOD(SHIFT(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         V(2)=MOD(SHIFT(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         V(3)=MOD(SHIFT(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
         NTEST=1
         TESTV(1,1)=V(1)
         TESTV(1,2)=V(2)
         TESTV(1,3)=V(3)
         DO 200 IRFULL=1,NRFULL
! Rotate shift and apply periodic boundary conditions to rotated shift:
            VR(1)=V(1)*FLOAT(IFULLG(1,1,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,1,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,1,IRFULL))
            VR(2)=V(1)*FLOAT(IFULLG(1,2,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,2,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,2,IRFULL))
            VR(3)=V(1)*FLOAT(IFULLG(1,3,IRFULL))+ &
     &                       V(2)*FLOAT(IFULLG(2,3,IRFULL))+ &
     &                                    V(3)*FLOAT(IFULLG(3,3,IRFULL))
            VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
! Test whether point has already been generated ... :
            IEXIST=0
            DO 190 ITEST=1,NTEST
               IF ((ABS(VR(1)-TESTV(ITEST,1))<TINY).AND. &
     &                 (ABS(VR(2)-TESTV(ITEST,2))<TINY).AND. &
     &                     (ABS(VR(3)-TESTV(ITEST,3))<TINY)) IEXIST=1
  190       CONTINUE
            IF (IEXIST==0) THEN
               NTEST=NTEST+1
               TESTV(NTEST,1)=VR(1)
               TESTV(NTEST,2)=VR(2)
               TESTV(NTEST,3)=VR(3)
            END IF
  200    CONTINUE
         NA(1)=NTEST
         CALL PRICEL(IBRAV,CDIMK,P1,P2,P3,TESTV,BKT(1,1),BKT(1,2), &
     &   BKT(1,3),PWORK,NCELL,IMTYP,CDIMR,1,1,NA,48,TWORK,IWORK,WORK,-1)
! Cross check (debugging ...):
         IF (NTEST/=NCELL) CALL ERROR(' IBZKPT', &
     &   ' Fatal error in detecting k-mesh type!',NTEST*100+NCELL-NTEST)
         IF (IMTYP/=IBRAVK) CALL ERROR(' IBZKPT', &
     &   ' Fatal error detecting k-mesh type!',IMTYP*10000+IMTYP-IBRAVK)
! Rough estimate: how many mesh-points do we need in each direction?
         BK1A1=BKT(1,1)*A(1,1)+BKT(2,1)*A(2,1)+BKT(3,1)*A(3,1)
         BK2A1=BKT(1,1)*A(1,2)+BKT(2,1)*A(2,2)+BKT(3,1)*A(3,2)
         BK3A1=BKT(1,1)*A(1,3)+BKT(2,1)*A(2,3)+BKT(3,1)*A(3,3)
         BK1A2=BKT(1,2)*A(1,1)+BKT(2,2)*A(2,1)+BKT(3,2)*A(3,1)
         BK2A2=BKT(1,2)*A(1,2)+BKT(2,2)*A(2,2)+BKT(3,2)*A(3,2)
         BK3A2=BKT(1,2)*A(1,3)+BKT(2,2)*A(2,3)+BKT(3,2)*A(3,3)
         BK1A3=BKT(1,3)*A(1,1)+BKT(2,3)*A(2,1)+BKT(3,3)*A(3,1)
         BK2A3=BKT(1,3)*A(1,2)+BKT(2,3)*A(2,2)+BKT(3,3)*A(3,2)
         BK3A3=BKT(1,3)*A(1,3)+BKT(2,3)*A(2,3)+BKT(3,3)*A(3,3)
         IF (ABS(BK1A1)<TINY) BK1A1=1.E30_q
         IF (ABS(BK1A2)<TINY) BK1A2=1.E30_q
         IF (ABS(BK1A3)<TINY) BK1A3=1.E30_q
         IF (ABS(BK2A1)<TINY) BK2A1=1.E30_q
         IF (ABS(BK2A2)<TINY) BK2A2=1.E30_q
         IF (ABS(BK2A3)<TINY) BK2A3=1.E30_q
         IF (ABS(BK3A1)<TINY) BK3A1=1.E30_q
         IF (ABS(BK3A2)<TINY) BK3A2=1.E30_q
         IF (ABS(BK3A3)<TINY) BK3A3=1.E30_q
         NKX=NINT(MAX(MAX(1._q/ABS(BK1A1),1._q/ABS(BK2A1)),1._q/ABS(BK3A1)))
         NKY=NINT(MAX(MAX(1._q/ABS(BK1A2),1._q/ABS(BK2A2)),1._q/ABS(BK3A2)))
         NKZ=NINT(MAX(MAX(1._q/ABS(BK1A3),1._q/ABS(BK2A3)),1._q/ABS(BK3A3)))
! Take (one of) the correct symmetry conserving shift(s):
         NKXM=MOD(NKX,2)
         NKYM=MOD(NKY,2)
         NKZM=MOD(NKZ,2)
         SHIFT(1)=0.5_q*NKXM
         SHIFT(2)=0.5_q*NKYM
         SHIFT(3)=0.5_q*NKZM
! Check (debugging):
         NCODE=100*NKXM+10*NKYM+NKZM
         IF ((IBRAVK==1).AND. &
     &   ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &       CALL ERROR(' IBZKPT',' Could not get correct shifts',NCODE)
         IF ((IBRAVK==5).AND. &
     &   ((ABS(SHTEST(1))>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(3))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3))>TINY)).AND. &
     &   ((ABS(SHTEST(3)-0.5_q)>TINY).OR.(ABS(SHTEST(2))>TINY).OR. &
     &   (ABS(SHTEST(1))>TINY)).AND.((ABS(SHTEST(1)-0.5_q)>TINY).OR. &
     &   (ABS(SHTEST(2)-0.5_q)>TINY).OR.(ABS(SHTEST(3)-0.5_q)>TINY))) &
     &       CALL ERROR(' IBZKPT',' Could not get correct shifts',NCODE)
! Get the symmetry operations represented in the basis of the BKTs:
         CALL SGRTRF(G,S,NROTKM,BK(1,1),BK(1,2),BK(1,3), &
     &                                       BKT(1,1),BKT(1,2),BKT(1,3))
      ELSE
! Meshes with symmetry conserving shifts - take what we have already got:
         BKT(1,1)=BK(1,1)
         BKT(2,1)=BK(2,1)
         BKT(3,1)=BK(3,1)
         BKT(1,2)=BK(1,2)
         BKT(2,2)=BK(2,2)
         BKT(3,2)=BK(3,2)
         BKT(1,3)=BK(1,3)
         BKT(2,3)=BK(2,3)
         BKT(3,3)=BK(3,3)
         DO 300 IROT=1,NROTKM
            S(1,1,IROT)=G(1,1,IROT)
            S(2,1,IROT)=G(2,1,IROT)
            S(3,1,IROT)=G(3,1,IROT)
            S(1,2,IROT)=G(1,2,IROT)
            S(2,2,IROT)=G(2,2,IROT)
            S(3,2,IROT)=G(3,2,IROT)
            S(1,3,IROT)=G(1,3,IROT)
            S(2,3,IROT)=G(2,3,IROT)
            S(3,3,IROT)=G(3,3,IROT)
  300    CONTINUE
      END IF
      IF (.NOT.TZERO) THEN
! We may only have shifts +/-0.5 and zero: -0.5 and +0.5 are identical,
! for getting routine TETIRR working properly take always shifts +0.5!
         IF (ABS(SHTEST(1)-0.5_q)<TINY) SHIFT(1)=0.5_q
         IF (ABS(SHTEST(2)-0.5_q)<TINY) SHIFT(2)=0.5_q
         IF (ABS(SHTEST(3)-0.5_q)<TINY) SHIFT(3)=0.5_q
      END IF
! Testing dimensions of array IKPT ... :
      IF (NKX>IKPTD) CALL ERROR(' IBZKPT',' NKX>IKPTD',NKX)
      IF (NKY>IKPTD) CALL ERROR(' IBZKPT',' NKY>IKPTD',NKY)
      IF (NKZ>IKPTD) CALL ERROR(' IBZKPT',' NKZ>IKPTD',NKZ)
      CALL RECIPS(1._q,BKT(1,1),BKT(1,2),BKT(1,3),AKT(1,1),AKT(1,2),AKT(1,3))
      IKPT=-1
      NKPT=0
! Now pass through the mesh and get the 'connection table' ... :
      DO I3=0,NKZ-1
       DO I2=0,NKY-1
         DO 700 I1=0,NKX-1
! k-point already visited:
            IF (IKPT(I1+1,I2+1,I3+1)>0) GOTO 700
! k-point coordinates in the basis BKT:
            V(1)=I1+SHIFT(1)
            V(2)=I2+SHIFT(2)
            V(3)=I3+SHIFT(3)
! Periodic boundary conditions (for compatibility with old version):
            T(1)=V(1)*BKT(1,1)+V(2)*BKT(1,2)+V(3)*BKT(1,3)
            T(2)=V(1)*BKT(2,1)+V(2)*BKT(2,2)+V(3)*BKT(2,3)
            T(3)=V(1)*BKT(3,1)+V(2)*BKT(3,2)+V(3)*BKT(3,3)
            VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
            VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
            VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
            VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
            T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
            T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
            T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
            IEXIST=0
            IF (THALF.AND.(.NOT.TKEEP)) THEN
! Here we are at a dangerous point when dealing with symmetry-breaking shifts:
! We may only store the k-point if it lies withing a unit cell spanned by B,
! however, some points on the BKT-mesh may also be outside of this unit cell!
               VR(1)=T(1)*AKT(1,1)+T(2)*AKT(2,1)+T(3)*AKT(3,1)
               VR(2)=T(1)*AKT(1,2)+T(2)*AKT(2,2)+T(3)*AKT(3,2)
               VR(3)=T(1)*AKT(1,3)+T(2)*AKT(2,3)+T(3)*AKT(3,3)
               J1=NINT(VR(1)-SHIFT(1))
               J2=NINT(VR(2)-SHIFT(2))
               J3=NINT(VR(3)-SHIFT(3))
               VR(1)=J1+SHIFT(1)
               VR(2)=J2+SHIFT(2)
               VR(3)=J3+SHIFT(3)
               IF ((ABS(VR(1)-V(1))>TINY).OR.(ABS(VR(2)-V(2))>TINY).OR. &
                   (ABS(VR(3)-V(3))>TINY)) IEXIST=-1
            ENDIF
! Now, here we take the coordinates in the BK-basis ... !
            VR(1)=T(1)*AK(1,1)+T(2)*AK(2,1)+T(3)*AK(3,1)
            VR(2)=T(1)*AK(1,2)+T(2)*AK(2,2)+T(3)*AK(3,2)
            VR(3)=T(1)*AK(1,3)+T(2)*AK(2,3)+T(3)*AK(3,3)
! The point outside could correspond to an existing point ...
            IF (THALF.AND.(.NOT.TKEEP)) THEN
               DO 90 IK=1,NKPT
                  IF ((ABS(VR(1)-VKPT(1,IK))<TINY).AND. &
                          (ABS(VR(2)-VKPT(2,IK))<TINY).AND. &
                              (ABS(VR(3)-VKPT(3,IK))<TINY)) THEN
                     IEXIST=IK
                     GOTO 91
                  ENDIF
   90          CONTINUE
   91          CONTINUE
            ENDIF
            IF (IEXIST==0) THEN
! Store the new k-point ...
               NKPT=NKPT+1
               VKPT(1,NKPT)=VR(1)
               VKPT(2,NKPT)=VR(2)
               VKPT(3,NKPT)=VR(3)
! ... and initialize the symmetry degeneracy counter:
               WTKPT(NKPT)=1._q
               IF (NKPT>NKDIM) CALL ERROR(' IBZKPT',' NKPT>NKDIM',NKPT)
! Set connection table for this k-point:
               IKPT(I1+1,I2+1,I3+1)=NKPT
            ELSE
! Otherwise dont store anything and only mark the point as 'visited':
               IKPT(I1+1,I2+1,I3+1)=MAX(IEXIST,0)
            ENDIF
! For all symmetry operations:
            DO 500 I=1,NROTKM
! Rotate k-point:
               VR(1)=V(1)*S(1,1,I)+V(2)*S(2,1,I)+V(3)*S(3,1,I)
               VR(2)=V(1)*S(1,2,I)+V(2)*S(2,2,I)+V(3)*S(3,2,I)
               VR(3)=V(1)*S(1,3,I)+V(2)*S(2,3,I)+V(3)*S(3,3,I)
! Get an index from this ... :
               J1=NINT(VR(1)-SHIFT(1))
               J2=NINT(VR(2)-SHIFT(2))
               J3=NINT(VR(3)-SHIFT(3))
! Consistency check for debugging:
               IF (ABS(J1-VR(1)+SHIFT(1))>TINY) CALL ERROR(' IBZKPT_FAST', &
                  ' Internal error: rotated coordinate J1 not an integer',J1)
               IF (ABS(J2-VR(2)+SHIFT(2))>TINY) CALL ERROR(' IBZKPT_FAST', &
                  ' Internal error: rotated coordinate J2 not an integer',J2)
               IF (ABS(J3-VR(3)+SHIFT(3))>TINY) CALL ERROR(' IBZKPT_FAST', &
                  ' Internal error: rotated coordinate J3 not an integer',J3)
! Shift back to interval [0,NKi] (periodic boundary conditions):
               J1=MOD(J1+60*NKX,NKX)
               J2=MOD(J2+60*NKY,NKY)
               J3=MOD(J3+60*NKZ,NKZ)
! Again we have to care about the problematic symmetry-breaking shifts:
               IF (THALF.AND.(.NOT.TKEEP)) THEN
! Periodic boundary conditions with respect to the unit cell spanned by B
                  T(1)=VR(1)*BKT(1,1)+VR(2)*BKT(1,2)+VR(3)*BKT(1,3)
                  T(2)=VR(1)*BKT(2,1)+VR(2)*BKT(2,2)+VR(3)*BKT(2,3)
                  T(3)=VR(1)*BKT(3,1)+VR(2)*BKT(3,2)+VR(3)*BKT(3,3)
                  VR(1)=T(1)*A(1,1)+T(2)*A(2,1)+T(3)*A(3,1)
                  VR(2)=T(1)*A(1,2)+T(2)*A(2,2)+T(3)*A(3,2)
                  VR(3)=T(1)*A(1,3)+T(2)*A(2,3)+T(3)*A(3,3)
                  VR(1)=MOD(VR(1)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(2)=MOD(VR(2)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  VR(3)=MOD(VR(3)+6.5_q-0.5_q*TINY,1._q)-0.5_q+0.5_q*TINY
                  T(1)=VR(1)*B(1,1)+VR(2)*B(1,2)+VR(3)*B(1,3)
                  T(2)=VR(1)*B(2,1)+VR(2)*B(2,2)+VR(3)*B(2,3)
                  T(3)=VR(1)*B(3,1)+VR(2)*B(3,2)+VR(3)*B(3,3)
                  VR(1)=T(1)*AKT(1,1)+T(2)*AKT(2,1)+T(3)*AKT(3,1)
                  VR(2)=T(1)*AKT(1,2)+T(2)*AKT(2,2)+T(3)*AKT(3,2)
                  VR(3)=T(1)*AKT(1,3)+T(2)*AKT(2,3)+T(3)*AKT(3,3)
! Get another index ... :
                  JJ1=NINT(VR(1)-SHIFT(1))
                  JJ2=NINT(VR(2)-SHIFT(2))
                  JJ3=NINT(VR(3)-SHIFT(3))
! Consistency check for debugging:
                  IF (ABS(JJ1-VR(1)+SHIFT(1))>TINY) CALL ERROR(' IBZKPT_FAST', &
                   ' Internal error: rotated coordinate JJ1 not an integer',JJ1)
                  IF (ABS(JJ2-VR(2)+SHIFT(2))>TINY) CALL ERROR(' IBZKPT_FAST', &
                   ' Internal error: rotated coordinate JJ2 not an integer',JJ2)
                  IF (ABS(JJ3-VR(3)+SHIFT(3))>TINY) CALL ERROR(' IBZKPT_FAST', &
                   ' Internal error: rotated coordinate JJ3 not an integer',JJ3)
! Shift back to interval [0,NKi]:
                  JJ1=MOD(JJ1+60*NKX,NKX)
                  JJ2=MOD(JJ2+60*NKY,NKY)
                  JJ3=MOD(JJ3+60*NKZ,NKZ)
! If these indices are equal to the original ones they were inside the unit
! cell spanned by B, otherwise they were outside the unit cell spanned by B,
! so mark them as 'visited' but dont store or count this k-point ...
                  IF (((JJ1/=J1).OR.(JJ2/=J2).OR. &
                       (JJ3/=J3)).AND.(IEXIST==0).AND. &
                       (IKPT(J1+1,J2+1,J3+1)<0)) IKPT(J1+1,J2+1,J3+1)=NKPT
                  IF ((IEXIST/=0).AND. &
                    (IKPT(J1+1,J2+1,J3+1)<0)) IKPT(J1+1,J2+1,J3+1)=MAX(IEXIST,0)
                  IF ((IKPT(JJ1+1,JJ2+1,JJ3+1)<0).AND.(IEXIST/=0)) THEN
! Set connection table for the rotated k-point:
                     IKPT(JJ1+1,JJ2+1,JJ3+1)=MAX(IEXIST,0)
                  END IF
                  IF ((IKPT(JJ1+1,JJ2+1,JJ3+1)<0).AND.(IEXIST==0)) THEN
! Set connection table for the rotated k-point:
                     IKPT(JJ1+1,JJ2+1,JJ3+1)=NKPT
! Increment the 'symmetry degeneracy counter':
                     WTKPT(NKPT)=WTKPT(NKPT)+1._q
                  END IF
               ENDIF
               IF (IKPT(J1+1,J2+1,J3+1)<0) THEN
! Set connection table for the rotated k-point:
                  IKPT(J1+1,J2+1,J3+1)=NKPT
! Increment the 'symmetry degeneracy counter':
                  WTKPT(NKPT)=WTKPT(NKPT)+1._q
               END IF
  500       CONTINUE
  700    CONTINUE
       ENDDO
      ENDDO

!=======================================================================
! last task to do: transform the coordinates from basis BK to basis B
! and supply the correct scaling to the weighting factors ...
!=======================================================================
      WEISUM=0._q
      DO 9 IK=1,NKPT
         V(1)=VKPT(1,IK)*BK(1,1)+VKPT(2,IK)*BK(1,2)+VKPT(3,IK)*BK(1,3)
         V(2)=VKPT(1,IK)*BK(2,1)+VKPT(2,IK)*BK(2,2)+VKPT(3,IK)*BK(2,3)
         V(3)=VKPT(1,IK)*BK(3,1)+VKPT(2,IK)*BK(3,2)+VKPT(3,IK)*BK(3,3)
! k-point coordinates in basis B:
         VKPT(1,IK)=V(1)*A(1,1)+V(2)*A(2,1)+V(3)*A(3,1)
         VKPT(2,IK)=V(1)*A(1,2)+V(2)*A(2,2)+V(3)*A(3,2)
         VKPT(3,IK)=V(1)*A(1,3)+V(2)*A(2,3)+V(3)*A(3,3)
! Sum up weights:
         WEISUM=WEISUM+WTKPT(IK)
    9 CONTINUE

! ... and very last task is to print the result (if desired):
      IF ((IU6>=0).AND.(IU6<=99)) THEN
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Subroutine IBZKPT returns following result:'
         WRITE(IU6,'(A)') ' ==========================================='
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A,I6,A)') ' Found ',NKPT,' irreducible k-points:'
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Following reciprocal coordinates:'
         WRITE(IU6,'(A)') '            Coordinates               Weight'
         DO 11 IK=1,NKPT
            WRITE(IU6,'(3F10.6,4X,F10.6)') &
     &                        VKPT(1,IK),VKPT(2,IK),VKPT(3,IK),WTKPT(IK)
   11    CONTINUE
         WRITE(IU6,'(A)') ' '
         WRITE(IU6,'(A)') ' Following cartesian coordinates:'
         WRITE(IU6,'(A)') '            Coordinates               Weight'
         DO 12 IK=1,NKPT
            V(1)=VKPT(1,IK)*B(1,1)+VKPT(2,IK)*B(1,2)+VKPT(3,IK)*B(1,3)
            V(2)=VKPT(1,IK)*B(2,1)+VKPT(2,IK)*B(2,2)+VKPT(3,IK)*B(2,3)
            V(3)=VKPT(1,IK)*B(3,1)+VKPT(2,IK)*B(3,2)+VKPT(3,IK)*B(3,3)
            V(1)=V(1)*SCALE
            V(2)=V(2)*SCALE
            V(3)=V(3)*SCALE
            WRITE(IU6,'(3F10.6,4X,F10.6)') V(1),V(2),V(3),WTKPT(IK)
   12    CONTINUE
         WRITE(IU6,'(A)') ' '
      END IF

! Result on file IBZKPT (can be copied to KPOINTS for following runs):
      IU=-1
! The IBZKPT is not written for the MPI version, because would
! be a little bit difficult to guarantee in all cases that only
! one proc writes the file
      IF (IU6>=0) THEN

      DO 1001 I=0,99
         IF ((I==0).OR.(I==5).OR.(I==6)) GOTO 1001
         INQUIRE(UNIT=I,OPENED=OCCUP)
         IF (.NOT.OCCUP) THEN
            IU=I
            GOTO 1002
         END IF
 1001 CONTINUE
 1002 CONTINUE
      IF (IU<0) THEN
         WRITE(*,*) &
     &      'Sorry: No IO-unit available! Cannot save data from IBZKPT!'
         GOTO 1009
      END IF
      OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'IBZKPT')
      WRITE(IU,'(A)') 'Automatically generated mesh'
      WRITE(IU,'(I8)') NKPT
      WRITE(IU,'(A)') 'Reciprocal lattice'
      DO 1005 IK=1,NKPT
         WRITE(IU,'(3F20.14,4X,I10)') VKPT(1,IK),VKPT(2,IK),VKPT(3,IK), &
     &                                NINT(WTKPT(IK))
 1005 CONTINUE
 1009 CONTINUE

      ENDIF
! Correct scaling for weights ... :
      DO IK=1,NKPT
         WTKPT(IK)=WTKPT(IK)/WEISUM
      ENDDO

! Find inequivalent tetrahedra for tetrahedron integration method ... :
      IF (LTET) THEN
         IF (NKPT<4) CALL ERROR(' IBZKPT', &
     &           ' Tetrahedron method fails for NKPT<4. NKPT =',NKPT)
         CALL TETIRR(NTET,IDTET,NTETD,BK,NKX,NKY,NKZ,IKPT,IKPTD,IU6)
! Get the volume weight of each tetrahedron (according to the full mesh)
! where the factor 6 accounts for the fact that around each k-point we
! have examined a microcell giving 6 tetrahedra and the factor 2. will
! account for spin-degeneracy (and must not be included in the routines
! which perform the BZ-integrations!!! - just provide VOLWGT, do not add
! any changes, take it as is ...).
         VOLWGT=1._q/FLOAT(6*NKX*NKY*NKZ)

! Result on file IBZKPT:
! The IBZKPT is not written for the MPI version, because would
! be a little bit difficult to guarantee in all cases that only
! one proc writes the file
         IF (IU>=0) THEN
            WRITE(IU,'(A)') 'Tetrahedra'
            WRITE(IU,'(I10,F20.14)') NTET,VOLWGT
            DO 1007 ITET=1,NTET
               WRITE(IU,'(5I10)') (IDTET(KTET,ITET),KTET=0,4)
 1007       CONTINUE
         END IF
      END IF
      IF (IU>=0) CLOSE(IU)

      RETURN
      END

#ifndef oldsym
      SUBROUTINE SET_SPINROT_WRAPPER(B,IU6)
      USE prec
      USE spinsym
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION A(3,3),B(3,3)
      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      CALL RECIPS(1._q,B(1,1),B(1,2),B(1,3),A(1,1),A(1,2),A(1,3))
      CALL SET_SPINROT(A,B,ISYMOP,NROTK,GTRANS,IU6)

      RETURN
      END SUBROUTINE SET_SPINROT_WRAPPER
#endif

!******************** SUBROUTINE TETIRR ********************************
!
! subroutine tetirr finds inequivalent tetrahedra in an equally spaced
! k-mesh (setting also the correct weights) ...
!
! this routine needs the basis vectors of the k-lattice (BK) [cartesian]
!                    the size of the k-mesh (NKX,NKY,NKZ)
!                    the k-point 'connection list' IKPT
!                    the dimension parameter IKPTD for array IKPT
!                    the dimension parameter NTETD for array IDTET
!                    and the unit where to write informations (IU6)
! it returns:        the number of irreducible tetrahedra (NTET)
!                    the k-index of the four corners [IDTET(1-4,...)]
!                    and the weighting factors [IDTET(0,...)]
!
!***********************************************************************

      SUBROUTINE TETIRR(NTET,IDTET,NTETD,BK,NKX,NKY,NKZ,IKPT,IKPTD,IU6)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION IDTET(0:4,NTETD),BK(3,3),IKPT(IKPTD,IKPTD,IKPTD)
      DIMENSION KCUT0(3,4,6),KCUT(3,4,6),P(3,4),IQ(4),IMC(0:1,0:1,0:1)

      SAVE KCUT0
      DATA KCUT0/ &
     &         0,0,0, 0,1,0, 1,1,0, 1,1,1,  0,0,0, 1,0,0, 1,1,0, 1,1,1, &
     &         0,0,0, 1,0,0, 1,0,1, 1,1,1,  0,0,0, 0,1,0, 0,1,1, 1,1,1, &
     &         0,0,0, 0,0,1, 0,1,1, 1,1,1,  0,0,0, 0,0,1, 1,0,1, 1,1,1 /

! Setting up the tetrahedra will be done cutting a microcell with eight
! corners into six tetrahedra. The edges of the tetrahedra are given by
! three edges, two face diagonals and one space diagonal of the cell.
! Giving the space diagonal, the way how to choose the rest is uniquely
! determined ... . But now there are four different possibilities how to
! choose the space diagonal! Prefer the choice which gives the shortest
! edges for all tetrahedra ('the most compact tetrahedra') - just to
! avoid very long 'interpolation distances' ... !
      LZ=0
      LXX=0
      LYY=0
      EDGMAX=1.E30_q
      EDGMIN=0._q
! For the four choices ...
      DO LX=0,1
       DO LY=0,1
! ... we set up the 'trial division' of a given cell into 6 tetrahedra:
         DO ITET=1,6
          DO IC=1,4
            KCUT(1,IC,ITET)=KCUT0(1,IC,ITET)
            KCUT(2,IC,ITET)=KCUT0(2,IC,ITET)
            KCUT(3,IC,ITET)=KCUT0(3,IC,ITET)
            IF (LX==1) KCUT(1,IC,ITET)=1-KCUT0(1,IC,ITET)
            IF (LY==1) KCUT(2,IC,ITET)=1-KCUT0(2,IC,ITET)
          ENDDO
         ENDDO
         EDMIN=1.E30_q
         EDMAX=0._q
! For this trial setting, loop over all tetrahedra ...,
         DO 4 ITET=1,6
! ... set up the cartesian coordinates of the four corner points ...,
            DO 2 IC=1,4
               P(1,IC)=KCUT(1,IC,ITET)*BK(1,1)+ &
     &                 KCUT(2,IC,ITET)*BK(1,2)+ &
     &                 KCUT(3,IC,ITET)*BK(1,3)
               P(2,IC)=KCUT(1,IC,ITET)*BK(2,1)+ &
     &                 KCUT(2,IC,ITET)*BK(2,2)+ &
     &                 KCUT(3,IC,ITET)*BK(2,3)
               P(3,IC)=KCUT(1,IC,ITET)*BK(3,1)+ &
     &                 KCUT(2,IC,ITET)*BK(3,2)+ &
     &                 KCUT(3,IC,ITET)*BK(3,3)
    2       CONTINUE
! ... and get the shortest and longest distance between two points in
! each tetrahedron (minimum/maximum taken over all tetrahedra ...):
            DO I=1,3
             DO J=I+1,4
               XX=ANRM2(P(1,I)-P(1,J),P(2,I)-P(2,J),P(3,I)-P(3,J))
               EDMAX=MAX(EDMAX,XX)
               EDMIN=MIN(EDMIN,XX)
             ENDDO
            ENDDO
    4    CONTINUE
! Now look at the global maximum: Have we found a cut with smaller
! maximum distance between two points within one tetrahedron than
! before? If yes: store it  (until we find something better ...)!
         IF (EDMAX<EDGMAX) THEN
            LXX=LX
            LYY=LY
            EDGMAX=EDMAX
            EDGMIN=EDMIN
         END IF
       ENDDO
      ENDDO
! Now set up the 'correct' cutup giving the most compact tetrahdra ... :
      DO ITET=1,6
       DO IC=1,4
         KCUT(1,IC,ITET)=KCUT0(1,IC,ITET)
         KCUT(2,IC,ITET)=KCUT0(2,IC,ITET)
         KCUT(3,IC,ITET)=KCUT0(3,IC,ITET)
         IF (LXX==1) KCUT(1,IC,ITET)=1-KCUT0(1,IC,ITET)
         IF (LYY==1) KCUT(2,IC,ITET)=1-KCUT0(2,IC,ITET)
       ENDDO
      ENDDO
! Now start searching the tetrahedra ... :
      NTET=0
! For all k-points ...
      DO I3=1,NKZ
       DO I2=1,NKY
        DO I1=1,NKX
! Set up microcell of 8 k-points (= 8 corners of unit cell of k-mesh):
         DO K1=0,1
          J1=MOD(I1+K1-1,NKX)+1
          DO K2=0,1
           J2=MOD(I2+K2-1,NKY)+1
           DO K3=0,1
            J3=MOD(I3+K3-1,NKZ)+1
! Get the identifiers (the irreducible k-point connected to I1,I2,I3):
            IMC(K1,K2,K3)=IKPT(J1,J2,J3)
           ENDDO
          ENDDO
         ENDDO
! From this microcell we can cut out six tetrahedra:
         DO 13 ITET=1,6
! Set the 4 corners (identifiers) of the actual tetrahedron:
            DO 8 IC=1,4
               K1=KCUT(1,IC,ITET)
               K2=KCUT(2,IC,ITET)
               K3=KCUT(3,IC,ITET)
               IQ(IC)=IMC(K1,K2,K3)
    8       CONTINUE
! Order the identifiers of the corners ...
            DO J=1,3
             DO I=1,4-J
               IF (IQ(I)>IQ(I+1)) THEN
                  II=IQ(I)
                  IQ(I)=IQ(I+1)
                  IQ(I+1)=II
               END IF
             ENDDO
            ENDDO
! First tetrahedron ...
            IF (NTET==0) GOTO 11
! Now test all tetrahedra found previously:
            DO 10 N=1,NTET
               IF ((IDTET(1,N)==IQ(1)) &
     &            .AND.(IDTET(2,N)==IQ(2)) &
     &               .AND.(IDTET(3,N)==IQ(3)) &
     &                  .AND.(IDTET(4,N)==IQ(4))) THEN
! We have found the same combination previously, so increment the
! counter for this type of tetrahedron ...
                  IDTET(0,N)=IDTET(0,N)+1
! ... and go to the next tetrahedron:
                  GOTO 13
               END IF
   10       CONTINUE
! New tetrahedron found if arriving here:
   11       CONTINUE
! Count it, ...
            NTET=NTET+1
! ... store the corner coordiantes (identifier) ...
            DO I=1,4
               IDTET(I,NTET)=IQ(I)
            ENDDO
! ... and initialize the counter for this new type of tetrahedron:
            IDTET(0,NTET)=1
   13    CONTINUE
        ENDDO
       ENDDO
      ENDDO
      IF (NTET>NTETD) CALL ERROR(' TETIRR',' NTET > NTETD',NTET)
! Now tell us the result ... :
      IF (IU6>=0) WRITE(IU6,15) NTET,6*NKX*NKY*NKZ
   15 FORMAT(1X,'TETIRR: Found ',I6,' inequivalent tetrahedra from ',I8)

      RETURN
      CONTAINS
      FUNCTION ANRM2(X,Y,Z)
      ANRM2=X*X*1.00001E0_q+Y*Y*1.00002E0_q+Z*Z*1.00003E0_q &
     &                           -X*0.000004E0_q-Y*0.000003E0_q-Z*0.000002E0_q
      END FUNCTION

      END


!************************* SUBROUTINE COUNTER_SPIRAL *******************
!
! This subroutine is called in case one does spin spiral calculations.
! It rotates the initial magnetic moments clockwise in the xy
! plane, according to
!
!    M_rot_x = M_x cos(q.R) + M_y sin(q.R)
!    M_rot_y = M_y cos(q.R) - M_x sin(q.R)
!    M_rot_z = M_z
!
! where q is the spin spiral propagation vector, and R is the position
! to which the initial local moment refers (i.e. an atomic position).
!
! This counters the following counter clockwise rotation of the moments
! in the xy plane along the propagation direction of the spiral.
!
!***********************************************************************

      SUBROUTINE COUNTER_SPIRAL(QSPIRAL,NIONS,POSION,ATOMOM)

      USE prec
      USE constant

      IMPLICIT NONE

      INTEGER NIONS,I
      REAL(q) QSPIRAL(3)
      REAL(q) POSION(3,NIONS),ATOMOM(3*NIONS)
      REAL(q) QR,M_x,M_y

      DO I=1,NIONS
         QR=TPI*(QSPIRAL(1)*POSION(1,I)+QSPIRAL(2)*POSION(2,I)+QSPIRAL(3)*POSION(3,I))

         M_x=ATOMOM(1+(I-1)*3)*COS(QR)+ATOMOM(2+(I-1)*3)*SIN(QR)
       M_y=ATOMOM(2+(I-1)*3)*COS(QR)-ATOMOM(1+(I-1)*3)*SIN(QR)

         ATOMOM(1+(I-1)*3)=M_x
         ATOMOM(2+(I-1)*3)=M_y
      ENDDO
      END SUBROUTINE COUNTER_SPIRAL


      SUBROUTINE ERROR(A,B,N)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
! Prints some message and stops execution ... . This routine is needed
! for compatibility with packages SYMLIB and LATTLIB (J. Furthmueller)
      CHARACTER (LEN=*) A,B
      WRITE(*,1) A,B,N
    1 FORMAT(//' VERY BAD NEWS! internal error in subroutine',A,':'/,A,I8)
      STOP
      END

      SUBROUTINE WARN(A,B,N)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      CHARACTER (LEN=*) A,B
      WRITE(*,1) A,B,N
    1 FORMAT(//' VERY BAD NEWS! internal error in subroutine',A,':'/,A,I8)
      END

!***********************************************************************
!
!   Routine POLSYM symmetrizes a vectorfield given in
!   cartesian coordinates according to the current symmetry operations
!   stored in the COMMON block /SYMM/
!
!   Input parameters:
!   -----------------
!
!      F(3)   contains on input the vector to be symmetrized
!             (given in crystal coordinates)
!
!
!      A(3,3) contains the lattice vectors defining the unit cell
!
!***********************************************************************


      SUBROUTINE POLSYM(F,A)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)

      REAL(q) F(3), A(3,3)


! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

    CALL F1SYM(F,ISYMOP,NROTK,A)

    END SUBROUTINE
