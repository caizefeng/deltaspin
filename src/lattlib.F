#include "symbol.inc"
! RCS:  $Id: lattlib.F,v 1.5 2003/06/27 13:22:18 kresse Exp kresse $
!
! if you need someother precesion specify it in the makefile
!
!****************** Lattice vector package *****************************
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
      SUBROUTINE LATGEN(IBRAV,CELLDM,A1,A2,A3,OMEGA)
      USE prec


      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine LATGEN sets up the crystallographic vectors A1, A2 and A3. *
!   The corresponding unit cells are all primitive cells.              *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      IBRAV specifies the various bravais lattices as follows:        *
!      (The lattice types are ordered by crystal classes and by        *
!       descending symmetry ...).                                      *
!                                                                      *
!       IBRAV |              lattice type                              *
!      -------|-------------------------------------------------       *
!         1   |      simple cubic cell                                 *
!         2   |      body-centered cubic cell                          *
!         3   |      face-centered cubic cell                          *
!         4   |      hexagonal cell                                    *
!         5   |      simple tetragonal cell                            *
!         6   |      body-centered tetragonal cell                     *
!         7   |      rhombohedral (trigonal) cell                      *
!         8   |      simple orthorhombic cell                          *
!         9   |      body-centered orthorhombic cell                   *
!        10   |      face-centered orthorhombic cell                   *
!        11   |      base centered orthorhombic cell                   *
!        12   |      simple monoclinic cell                            *
!        13   |      base-centered monoclinic cell                     *
!        14   |      triclinic cell                                    *
!      -------|-------------------------------------------------       *
!        15   |      face-centered cubic, baseplane 111-oriented       *
!                                                                      *
!  A very special meaning has IBRAV=0: You can give arbitrary vectors  *
!  A1, A2 and A3 (generated 'by hand' or by other routines) and store  *
!  them by setting CELLDM(1).le.0. and you can recall them by setting  *
!  CELLDM(1).gt. 0. (all stored vectors being scaled with CELLDM(1)).   *
!                                                                      *
!      CELLDM(I) controls the cell dimensions in the following way:    *
!                                                                      *
!        - CELLDM(1) is the length of the 'a-axis' measured in units   *
!                    of bohrs. (Absolute lattice constant).            *
!        - CELLDM(2) is the ratio 'b/a' in all lattices where b/=a     *
!                    (for example orthorombic or triclinic cells).     *
!        - CELLDM(3) is the ratio 'c/a' in all lattices where c/=a     *
!                    (for example hexagonal or tetragonal cells).      *
!        - CELLDM(4), CELLDM(5) and CELLDM(6) give the cosines of the  *
!                    angles between the lattice vectors (alpha, beta   *
!                    and gamma).                                       *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      A1, A2 and A3 are the three crystallographic vectors of the     *
!        primitive unit cell (given in cartesian coordinates x,y,z).   *
!                                                                      *
!      OMEGA is the volume of the unit cell.                           *
!                                                                      *
!                                                                      *
!   Remark:                                                            *
!   -------                                                            *
!                                                                      *
!      The primitive lattice vectors were taken from                   *
!      'The electron band theory of solids (Appendix E)'               *
!                                                                      *
!***********************************************************************

      DIMENSION CELLDM(6),A1(3),A2(3),A3(3),S1(3),S2(3),S3(3)

      SAVE S1,S2,S3,ISET
      DATA S1 /3*0._q/,S2 /3*0._q/,S3 /3*0._q/,ISET /0/

! Error !

      IF ((IBRAV<0).OR.(IBRAV>15)) CALL ERROR(' LATGEN', &
     &                          ' Value for IBRAV out of range',IBRAV)

! Special case IBRAV=0 :
! ----------------------

      IF (IBRAV==0) THEN
         ALAT=CELLDM(1)
         IF (ALAT<0._q) THEN
! Store the vectors ...:
            S1(1)=A1(1)
            S1(2)=A1(2)
            S1(3)=A1(3)
            S2(1)=A2(1)
            S2(2)=A2(2)
            S2(3)=A2(3)
            S3(1)=A3(1)
            S3(2)=A3(2)
            S3(3)=A3(3)
! Set flag:
            ISET=1
         ELSE
! Error:
            IF (ISET==0) CALL ERROR(' LATGEN', &
     &          ' No lattice vectors have been stored and IBRAV=0 .',-3)
! Recall the vectors and scale them with CELLDM(1):
            A1(1)=S1(1)*ALAT
            A1(2)=S1(2)*ALAT
            A1(3)=S1(3)*ALAT
            A2(1)=S2(1)*ALAT
            A2(2)=S2(2)*ALAT
            A2(3)=S2(3)*ALAT
            A3(1)=S3(1)*ALAT
            A3(2)=S3(2)*ALAT
            A3(3)=S3(3)*ALAT
         END IF
         GOTO 100
      END IF

! case IBRAV>0 ; setting up the predefined lattice vectors:
! ---------------------------------------------------------

! Define some local constants ...

      SR2=SQRT(2._q)
      SR3=SQRT(3._q)

! Preset the crystallographic vectors with (0,0,0) - then we need later
! only specify the non-zero components ...

      A1(1)=0._q
      A1(2)=0._q
      A1(3)=0._q
      A2(1)=0._q
      A2(2)=0._q
      A2(3)=0._q
      A3(1)=0._q
      A3(2)=0._q
      A3(3)=0._q

! Select the different Bravais-lattices:

      GOTO (2,4,6,8,10,12,14,16,18,20,22,24,26,28,30) IBRAV

! Simple cubic cell, the vectors are in scales of 'a':
!
!    (1,0,0)              (0,1,0)             (0,0,1)

    2 A1(1)=CELLDM(1)
      A2(2)=CELLDM(1)
      A3(3)=CELLDM(1)
      GOTO 100

! Body centered cubic cell, the vectors are in scales of 'a':
!
!    (-1/2,1/2,1/2)        (1/2,-1/2,1/2)      (1/2,1/2,-1/2)

    4 ABY2=CELLDM(1)*0.5_q
      A1(1)=-ABY2
      A1(2)=ABY2
      A1(3)=ABY2
      A2(1)=ABY2
      A2(2)=-ABY2
      A2(3)=ABY2
      A3(1)=ABY2
      A3(2)=ABY2
      A3(3)=-ABY2
      GOTO 100

! Face centered cubic cell, the vectors are in scales of 'a':
!
!    (0,1/2,1/2)          (1/2,0,1/2)          (1/2,1/2,0)

    6 ABY2=CELLDM(1)*0.5_q
      A1(2)=ABY2
      A1(3)=ABY2
      A2(1)=ABY2
      A2(3)=ABY2
      A3(1)=ABY2
      A3(2)=ABY2
      GOTO 100

! Hexagonal cell, the vectors are in scales of 'a':
!
!            o        o
!    (sin 120 ,cos 120 ,0)      (0,1,0)       (0,0,'c/a')
!
! Here CELLDM(3) controls the ratio 'c/a'.

    8 A1(1)=CELLDM(1)*SR3*0.5_q
      A1(2)=-CELLDM(1)*0.5_q
      A2(2)=CELLDM(1)
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! Simple tetragonal cell, the vectors are in scales of 'a':
!
!    (1,0,0)             (0,1,0)             (0,0,'c/a')
!
! Here CELLDM(3) controls the ratio 'c/a'.

   10 A1(1)=CELLDM(1)
      A2(2)=CELLDM(1)
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! Body centered tetragonal cell, the vectors are in scales of 'a':
!
!    (-1/2,1/2,'c/2a')     (1/2,-1/2,'c/2a')     (1/2,1/2,-'c/2a')
!
! Here CELLDM(3) controls the ratio 'c/a'.

   12 ABY2=CELLDM(1)*0.5_q
      CBY2=CELLDM(3)*ABY2
      A1(1)=-ABY2
      A1(2)=ABY2
      A1(3)=CBY2
      A2(1)=ABY2
      A2(2)=-ABY2
      A2(3)=CBY2
      A3(1)=ABY2
      A3(2)=ABY2
      A3(3)=-CBY2
      GOTO 100

! Rhombohedral (trigonal) cell, the vectors are in scales of 'a':
!
!    (0,'eta','zeta')
!               (-'eta'/2*sqrt(3),-'eta'/2,'zeta')
!                           ('eta'/2*sqrt(3),-'eta'/2,'zeta')
!
!  where: zeta=sqrt((2cos(alpha)+1)/3) and eta=sqrt(2/3*(1-cos(alpha))
!
!
! Here CELLDM(4) controls the cosine of angle 'ALPHA' (='BETA'='GAMMA')

   14 TERM1=SQRT(1._q+2._q*CELLDM(4))
      TERM2=SQRT(1._q-CELLDM(4))
      A1(2)=CELLDM(1)*TERM2*SR2/SR3
      A1(3)=CELLDM(1)*TERM1/SR3
      A2(1)=-CELLDM(1)*TERM2/SR2
      A2(2)=A2(1)/SR3
      A2(3)=A1(3)
      A3(1)=-A2(1)
      A3(2)=A2(2)
      A3(3)=A1(3)
      GOTO 100

! Simple orthorhombic cell, the vectors are in scales of 'a':
!
!    (1,0,0)             (0,'b/a',0)             (0,0,'c/a')
!
! Here CELLDM(2) controls the ratio 'b/a' and CELLDM(3) the ratio 'c/a'

   16 A1(1)=CELLDM(1)
      A2(2)=CELLDM(1)*CELLDM(2)
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! Body-centered orthorhombic cell, the vectors are in scales of 'a':
!
!    (-1/2,'b/2a','c/2a')   (1/2,-'b/2a','c/2a')   (1/2,'b/2a',-'c/2a')
!
! Here CELLDM(2) controls the ratio 'b/a' and CELLDM(3) the ratio 'c/a'

   18 ABY2=CELLDM(1)*0.5_q
      BBY2=CELLDM(2)*ABY2
      CBY2=CELLDM(3)*ABY2
      A1(1)=-ABY2
      A1(2)=BBY2
      A1(3)=CBY2
      A2(1)=ABY2
      A2(2)=-BBY2
      A2(3)=CBY2
      A3(1)=ABY2
      A3(2)=BBY2
      A3(3)=-CBY2
      GOTO 100

! Face-centered orthorhombic cell, the vectors are in scales of 'a':
!
!    (0,'b/2a','c/2a')      (1/2,0,'c/2a')       (1/2,'b/2a',0)
!
! Here CELLDM(2) controls the ratio 'b/a' and CELLDM(3) the ratio 'c/a'

   20 ABY2=CELLDM(1)*0.5_q
      BBY2=CELLDM(2)*ABY2
      CBY2=CELLDM(3)*ABY2
      A1(2)=BBY2
      A1(3)=CBY2
      A2(1)=ABY2
      A2(3)=CBY2
      A3(1)=ABY2
      A3(2)=BBY2
      GOTO 100

! Base-centered orthorhombic cell, the vectors are in scales of 'a':
!
!    (1/2,-'b/2a',0)      (1/2,'b/2a',0)       (0,0,'c/a')
!
! Here CELLDM(2) controls the ratio 'b/a' and CELLDM(3) the ratio 'c/a'

   22 ABY2=CELLDM(1)*0.5_q
      BBY2=CELLDM(2)*ABY2
      A1(1)=ABY2
      A1(2)=-BBY2
      A2(1)=ABY2
      A2(2)=BBY2
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! Simple monoclinic cell, the vectors are in scales of 'a':
!
!    (sin(beta),0,cos(beta))     (0,'b/a',0)       (0,0,'c/a')
!
! Here CELLDM(2) controls the ratio 'b/a', CELLDM(3) the ratio 'c/a' and
! CELLDM(5) the cosine of angle 'BETA'.

   24 SINBET=SQRT(1._q-CELLDM(5)**2)
      A1(1)=CELLDM(1)*SINBET
      A3(1)=CELLDM(1)*CELLDM(5)
      A2(2)=CELLDM(1)*CELLDM(2)
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! Base-centered monoclinic cell, the vectors are in scales of 'a':
!
!   (1/2,-'b/2a',0)  (1/2,'b/2a',0)  ('c/a'*cos(beta),0,'c/a'*sin(beta))
!
! Here CELLDM(2) controls the ratio 'b/a', CELLDM(3) the ratio 'c/a' and
! CELLDM(5) the cosine of angle 'BETA'.

   26 ABY2=CELLDM(1)*0.5_q
      BBY2=CELLDM(2)*ABY2
      SINBET=SQRT(1._q-CELLDM(5)*CELLDM(5))
      A1(1)=ABY2
      A1(2)=-BBY2
      A2(1)=ABY2
      A2(2)=BBY2
      A3(1)=CELLDM(1)*CELLDM(3)*CELLDM(5)
      A3(3)=CELLDM(1)*CELLDM(3)*SINBET
      GOTO 100

! Triclinic cell, the vectors are in scales of 'a':
!
!    (sin(beta),0,cos(beta))  (a2x,a2y,a2z)      (0,0,'c/a')
!
!  where:  a2x='b/a'*(cos(gamma)-cos(alpha)cos(beta))/sin(beta) ,
!          a2y=sqrt('b/a'**2*sin**2(alpha)-a2x**2)    and
!          a2z='b/a'*cos(alpha)) .
!
! Here CELLDM(2) controls the ratio 'b/a', CELLDM(3) the ratio 'c/a' and
! CELLDM(4) the cosine of angle 'ALPHA', CELLDM(5) the cosine of angle
! 'BETA' and CELLDM(6) the cosine of angle 'GAMMA'

   28 SINBET=SQRT(1._q-CELLDM(5)**2)
      TERM=SQRT((1._q+2._q*CELLDM(4)*CELLDM(5)*CELLDM(6)-CELLDM(4)**2 &
     &     -CELLDM(4)**2*CELLDM(5)**2-CELLDM(6)**2)/(1._q-CELLDM(5)**2))
      A1(1)=CELLDM(1)*SINBET
      A1(3)=CELLDM(1)*CELLDM(5)
      A2(1)=CELLDM(1)*CELLDM(2)*(CELLDM(6)-CELLDM(4)*CELLDM(5))/SINBET
      A2(2)=CELLDM(1)*CELLDM(2)*TERM
      A2(3)=CELLDM(1)*CELLDM(2)*CELLDM(4)
      A3(3)=CELLDM(1)*CELLDM(3)
      GOTO 100

! After the 14 bravais lattices in the 'conventional' standard settings
! we can define some further special settings of some bravais lattices:
! =====================================================================

! Face centered cubic cell, the basal plane (x,y) is choosen to be
! the <111> plane. The vectors are in scales of 'a':
!
!    (sqrt(1/8),sqrt(3/8),0)   (sqrt(1/8),-sqrt(3/8),0)   (0,0,sqrt(3))

   30 A1(1)=CELLDM(1)*0.5_q/SR2
      A1(2)=A1(1)*SR3
      A2(1)=A1(1)
      A2(2)=-A1(2)
      A3(3)=SR3*CELLDM(1)
      GOTO 100

! Calculate the cell volume:

  100 CALL CELVOL(A1,A2,A3,OMEGA)
      OMEGA=ABS(OMEGA)
      IF (OMEGA<(1.E-10_q*(CELLDM(1)**3))) CALL ERROR(' LATGEN', &
     &                                       ' Cellvolume is zero !',-3)

      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE LATTYP(A1,A2,A3,ITYP,CELDIM,PRCHAN)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine LATTYP identifies the type of bravais lattice spanned by   *
!   the primitive lattice vectors A1, A2 and A3. The lattice will be   *
!   transformed to a 'standard crystallographic setting' and the cell  *
!   dimensions will be estimated using the conventions of routine      *
!   LATGEN. The relation between 'original' and 'transformed' lattice  *
!   vectors will be given in matrix form.                              *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      A1, A2 and A3 are the three crystallographic vectors of the     *
!        primitive unit cell (given in cartesian coordinates x,y,z)    *
!        of the real space lattice given in units of bohrs ('a').      *
!      PRCHAN defines the unit number for printing output information. *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      A1, A2 and A3 contain on output the three crystallographic      *
!      vectors being transformed to some 'standard setting'.           *
!      ITYP gives the lattice type (1-14) as keyed in routine LATGEN.  *
!      CELDIM gives the cell dimensions as defined in routine LATGEN.  *
!                                                                      *
!***********************************************************************

      INTEGER PRCHAN
      DIMENSION A1(3),A2(3),A3(3),RB(3,3),XB(3,3),CELDIM(6),YB(3)
      DIMENSION CELLDM(6),SA1(3),SA2(3),SA3(3),CDMINP(6)

! 'Initialisation':
      CALL CELVOL(A1,A2,A3,OMEGA)
      IF (ABS(OMEGA)<(TINY*TINY*TINY)) CALL ERROR(' LATTYP', &
     &                                       ' Cellvolume is zero !',-3)
      CELDIM(1)=0._q
      CELDIM(2)=0._q
      CELDIM(3)=0._q
      CELDIM(4)=0._q
      CELDIM(5)=0._q
      CELDIM(6)=0._q
! Adjustement of the basis to right-hand-sense (by inversion of all
! three lattice vectors if necessary ...):
      CALL CELVOL(A1(1),A2(1),A3(1),OMEGA)
      IF (OMEGA<0) THEN
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) &
     &      WRITE(PRCHAN,*) ' Warning from LATTYP! Input basis '// &
     &                            'has left-hand-sense. Basis inverted!'
         A1(1)=-1._q*A1(1)
         A1(2)=-1._q*A1(2)
         A1(3)=-1._q*A1(3)
         A2(1)=-1._q*A2(1)
         A2(2)=-1._q*A2(2)
         A2(3)=-1._q*A2(3)
         A3(1)=-1._q*A3(1)
         A3(2)=-1._q*A3(2)
         A3(3)=-1._q*A3(3)
      END IF
! Save and copy the original lattice vectors:
      SA1(1)=A1(1)
      SA1(2)=A1(2)
      SA1(3)=A1(3)
      SA2(1)=A2(1)
      SA2(2)=A2(2)
      SA2(3)=A2(3)
      SA3(1)=A3(1)
      SA3(2)=A3(2)
      SA3(3)=A3(3)
      RB(1,1)=A1(1)
      RB(2,1)=A1(2)
      RB(3,1)=A1(3)
      RB(1,2)=A2(1)
      RB(2,2)=A2(2)
      RB(3,2)=A2(3)
      RB(1,3)=A3(1)
      RB(2,3)=A3(2)
      RB(3,3)=A3(3)
      XB(1,1)=A1(1)
      XB(2,1)=A1(2)
      XB(3,1)=A1(3)
      XB(1,2)=A2(1)
      XB(2,2)=A2(2)
      XB(3,2)=A2(3)
      XB(1,3)=A3(1)
      XB(2,3)=A3(2)
      XB(3,3)=A3(3)
! First test the original vectors (what do we apparently find?):
      ISTRGE=0
! Lengths of the axes and cosines between the axes:
      ABS1=XB(1,1)*XB(1,1)+XB(2,1)*XB(2,1)+XB(3,1)*XB(3,1)
      ABS2=XB(1,2)*XB(1,2)+XB(2,2)*XB(2,2)+XB(3,2)*XB(3,2)
      ABS3=XB(1,3)*XB(1,3)+XB(2,3)*XB(2,3)+XB(3,3)*XB(3,3)
      A2XA1=XB(1,1)*XB(1,2)+XB(2,1)*XB(2,2)+XB(3,1)*XB(3,2)
      A3XA1=XB(1,1)*XB(1,3)+XB(2,1)*XB(2,3)+XB(3,1)*XB(3,3)
      A3XA2=XB(1,2)*XB(1,3)+XB(2,2)*XB(2,3)+XB(3,2)*XB(3,3)
      ABS2M1=SQRT(ABS1+ABS2-2._q*A2XA1)
      ABS3M1=SQRT(ABS1+ABS3-2._q*A3XA1)
      ABS3M2=SQRT(ABS2+ABS3-2._q*A3XA2)
      ABS2P1=SQRT(ABS1+ABS2+2._q*A2XA1)
      ABS3P1=SQRT(ABS1+ABS3+2._q*A3XA1)
      ABS3P2=SQRT(ABS2+ABS3+2._q*A3XA2)
      A1P2M3=SQRT(ABS1+ABS2+ABS3+2._q*A2XA1-2._q*A3XA1-2._q*A3XA2)
      A2P3M1=SQRT(ABS1+ABS2+ABS3+2._q*A3XA2-2._q*A2XA1-2._q*A3XA1)
      A3P1M2=SQRT(ABS1+ABS2+ABS3+2._q*A3XA1-2._q*A3XA2-2._q*A2XA1)
      ABS123=A2XA1+A3XA1+A3XA2
      ABS1=SQRT(ABS1)
      ABS2=SQRT(ABS2)
      ABS3=SQRT(ABS3)
      A2XA1=A2XA1/ABS1/ABS2
      A3XA1=A3XA1/ABS1/ABS3
      A3XA2=A3XA2/ABS2/ABS3
      A1XA2=ABS(A2XA1)
      A1XA3=ABS(A3XA1)
      A2XA3=ABS(A3XA2)
! Warning extremly strange inputs (very large/small ratios 'c/a', 'b/a'
! or cosines very close to +/-1) could sometimes lead to some problems
! due to the finite precision of the machine ...). Here we test some
! parameters and if we find 'strange parameters', we (DO NOT) stop!
      IF (A1XA2>0.995_q) ISTRGE=ISTRGE+1
      IF (A2XA3>0.995_q) ISTRGE=ISTRGE+1
      IF (A1XA3>0.995_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS1)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS1)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS2)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS3)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS2)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS3)<0.001_q) ISTRGE=ISTRGE+1

! Following the identification of the crystal type, transformation of
! the basis to 'crystallographic standard settings' and estimate of the
! cell dimensions (using the conventions of routine LATGEN):
      CELLDM(1)=0._q
      CELLDM(2)=0._q
      CELLDM(3)=0._q
      CELLDM(4)=0._q
      CELLDM(5)=0._q
      CELLDM(6)=0._q
      IBRAV=15
! Crystal classes with A1*A2=A1*A3=A2*A3:
      IF ((ABS(A2XA1-A3XA1)<TINY).AND. &
     &                                  (ABS(A2XA1-A3XA2)<TINY)) THEN
! Crystal classes with A1*A2=A1*A3=A2*A3 and |A1|=|A2|=|A3|:
         IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                                    (ABS(ABS2-ABS3)<TINY)) THEN
! Orthogonal axes means simple cubic: (IBRAV=1)
            IF (A1XA2<TINY) THEN
               IBRAV=1
               CELLDM(1)=ABS1
! Cosine(alpha) equal to -1/3 means body-centered cubic: (IBRAV=2)
            ELSE IF (ABS(A2XA1+(1._q/3._q))<TINY) THEN
               IBRAV=2
               CELLDM(1)=ABS1*2._q/SQRT(3._q)
! Cosine(alpha) equal to +1/2 means face-centered cubic: (IBRAV=3)
            ELSE IF (ABS(A2XA1-0.5_q)<TINY) THEN
               IBRAV=3
               CELLDM(1)=ABS1*SQRT(2._q)
! All other angles means trigonal (rhomboedric): (IBRAV=7)
            ELSE
               IBRAV=7
               CELLDM(1)=ABS1
               CELLDM(4)=A2XA1
            END IF
! Crystal classes with inequal length of lattice vectors but also with
! A1*A2=A1*A3=A2*A3:
! Orthogonal axes:
         ELSE IF (A1XA2<TINY) THEN
! Two axes with equal lengths means simple tetragonal: (IBRAV=5)
! Adjustment: 'c-axis' shall be the special axis.
            IF (ABS(ABS1-ABS2)<TINY) THEN
               IBRAV=5
               CELLDM(1)=ABS1
               CELLDM(3)=ABS3/ABS1
! No axes with equal lengths means simple orthorhombic (IBRAV=8):
! Adjustment: Sort the axis by increasing lengths:
            ELSE IF (((ABS3-ABS2)>TINY).AND. &
     &                                   ((ABS2-ABS1)>TINY)) THEN
               IBRAV=8
               CELLDM(1)=ABS1
               CELLDM(2)=ABS2/ABS1
               CELLDM(3)=ABS3/ABS1
            END IF
         END IF
! Crystal classes with A1*A3=A2*A3=/A1*A2:
      ELSE IF (ABS(A3XA1-A3XA2)<TINY) THEN
! One axis orthogonal with respect to the other two axes:
         IF (A1XA3<TINY) THEN
! Equal length of the two nonorthogonal axes:
            IF (ABS(ABS1-ABS2)<TINY) THEN
! Cosine(alpha) equal to -1/2 means hexagonal: (IBRAV=4)
! Adjustment: 'c-axis' shall be the special axis.
               IF (ABS(A2XA1+0.5_q)<TINY) THEN
                  IBRAV=4
                  CELLDM(1)=ABS1
                  CELLDM(3)=ABS3/ABS1
! Other angles mean base-centered orthorhombic: (IBRAV=11)
! Adjustment: Cosine between A1 and A2 shall be lower than zero, the
!             'c-axis' shall be the special axis.
               ELSE IF (A2XA1<(-1._q*TINY)) THEN
                  IBRAV=11
                  CELLDM(1)=ABS2P1
                  CELLDM(2)=ABS2M1/ABS2P1
                  CELLDM(3)=ABS3/ABS2P1
               END IF
! Different length of the two axes means simple monoclinic (IBRAV=12):
! Adjustment: Cosine(gamma) should be lower than zero, special axis
!             shall be the 'b-axis'(!!) and |A1|<|A3|:
            ELSE IF ((A2XA1<(-1._q*TINY)).AND. &
     &                                       ((ABS1-ABS2)>TINY)) THEN
               IBRAV=12
               IF ((PRCHAN>=0).AND.(PRCHAN<=99)) &
     &            WRITE(PRCHAN,*) ' Warning from LATTYP: '// &
     &                'Monoclinic adjustement (A1->A3, A2->A1, A3->A2)!'
               CELLDM(1)=ABS2
               CELLDM(2)=ABS3/ABS2
               CELLDM(3)=ABS1/ABS2
               CELLDM(5)=A2XA1
               SA3(1)=XB(1,1)
               SA3(2)=XB(2,1)
               SA3(3)=XB(3,1)
               SA1(1)=XB(1,2)
               SA1(2)=XB(2,2)
               SA1(3)=XB(3,2)
               SA2(1)=XB(1,3)
               SA2(2)=XB(2,3)
               SA2(3)=XB(3,3)
            END IF
! Arbitrary angles between the axes:
         ELSE
! |A1|=|A2|=|A3| means body-centered tetragonal (IBRAV=6):
! Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
! orthogonal to one another and (adjustment!): |A1+A3|=|A2+A3|<|A1+A2|
            IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &               (ABS(ABS2-ABS3)<TINY).AND. &
     &                    (ABS(ABS3P1-ABS3P2)<TINY).AND. &
     &                        (ABS(ABS2P1-ABS3P1)>TINY).AND. &
     &                             (ABS(ABS3*ABS3+ABS123)<TINY)) THEN
               IBRAV=6
               CELLDM(1)=ABS3P1
               CELLDM(3)=ABS2P1/ABS3P1
! |A1|=|A2|=/|A3| means base-centered monoclinic (IBRAV=13):
! Adjustement: The cosine between A1 and A3 as well as the cosine
!              between A2 and A3 should be lower than zero.
            ELSE IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                          (A3XA1<(-1._q*TINY)).AND. &
     &                                       (A3XA2<(-1._q*TINY))) THEN
               IBRAV=13
               CELLDM(1)=ABS2P1
               CELLDM(2)=ABS2M1/ABS2P1
               CELLDM(3)=ABS3/ABS2P1
! Gilles Dewijs suggested the following change to lattlib.F
! since Gilles certainly did some testing I changed the statment
! according to his suggestions
!old
!              CELLDM(5)=A3XA2
!old
               CELLDM(5)=(A3XA2*ABS2*ABS3+A3XA1*ABS1*ABS3)/ABS2P1/ABS3      

            END IF
         END IF
! Crystal classes with A1*A2=/A1*A3=/A2*A3
      ELSE
! |A1|=|A2|=|A3| means body-centered orthorhombic (IBRAV=9):
! Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
! orthogonal to one another and (adjustment!): |A1+A2|>|A1+A3|>|A2+A3|
         IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &            (ABS(ABS2-ABS3)<TINY).AND. &
     &                  ((ABS3P1-ABS3P2)>TINY).AND. &
     &                         ((ABS2P1-ABS3P1)>TINY).AND. &
     &                             (ABS(ABS3*ABS3+ABS123)<TINY)) THEN
            IBRAV=9
            CELLDM(1)=ABS3P2
            CELLDM(2)=ABS3P1/ABS3P2
            CELLDM(3)=ABS2P1/ABS3P2
! |A1|=|A2-A3| and |A2|=|A1-A3| and |A3|=|A1-A2| means face-centered
! orthorhombic (IBRAV=10):
! Adjustment: |A1+A2-A3|>|A1+A3-A2|>|A2+A3-A1|
         ELSE IF ((ABS(ABS2M1-ABS3)<TINY).AND. &
     &                 (ABS(ABS3M1-ABS2)<TINY).AND. &
     &                       (ABS(ABS3M2-ABS1)<TINY).AND. &
     &                              ((A1P2M3-A3P1M2)>TINY).AND. &
     &                                   ((A3P1M2-A2P3M1)>TINY)) THEN
            IBRAV=10
            CELLDM(1)=A2P3M1
            CELLDM(2)=A3P1M2/A2P3M1
            CELLDM(3)=A1P2M3/A2P3M1
! Now there exists only one further possibility - triclinic (IBRAV=14):
! Adjustment: All three cosines shall be greater than zero and ordered:
         ELSE IF ((A2XA1>A3XA1).AND.(A3XA1>A3XA2).AND. &
     &                                             (A3XA2>TINY)) THEN
            IBRAV=14
            CELLDM(1)=ABS1
            CELLDM(2)=ABS2/ABS1
            CELLDM(3)=ABS3/ABS1
            CELLDM(4)=A3XA2
            CELLDM(5)=A3XA1
            CELLDM(6)=A2XA1
         END IF
      END IF
! Remember the result ... :
      IBRINP=IBRAV
      CDMINP(1)=CELLDM(1)
      CDMINP(2)=CELLDM(2)
      CDMINP(3)=CELLDM(3)
      CDMINP(4)=CELLDM(4)
      CDMINP(5)=CELLDM(5)
      CDMINP(6)=CELLDM(6)

! Warning extremly strange inputs (very large/small ratios 'c/a', 'b/a'
! or cosines very close to +/-1) could sometimes lead to some problems
! due to the finite precision of the machine ...). Here we test some
! parameters and if we find 'strange parameters' we (DO NOT) stop!
      ABS1=RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1)
      ABS2=RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2)
      ABS3=RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3)
      A2XA1=RB(1,1)*RB(1,2)+RB(2,1)*RB(2,2)+RB(3,1)*RB(3,2)
      A3XA1=RB(1,1)*RB(1,3)+RB(2,1)*RB(2,3)+RB(3,1)*RB(3,3)
      A3XA2=RB(1,2)*RB(1,3)+RB(2,2)*RB(2,3)+RB(3,2)*RB(3,3)
      ABS1=SQRT(ABS1)
      ABS2=SQRT(ABS2)
      ABS3=SQRT(ABS3)
      A1XA2=ABS(A2XA1/ABS1/ABS2)
      A1XA3=ABS(A3XA1/ABS1/ABS3)
      A2XA3=ABS(A3XA2/ABS2/ABS3)
      IF (A1XA2>0.99999_q) ISTRGE=ISTRGE+1
      IF (A2XA3>0.99999_q) ISTRGE=ISTRGE+1
      IF (A1XA3>0.99999_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS1)<0.00001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS1)<0.00001_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS2)<0.00001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS3)<0.00001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS2)<0.00001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS3)<0.00001_q) ISTRGE=ISTRGE+1

! Search a primitive basis with shortest lattice vectors:

      ICOUNT=0
      GOTO 199

! The algorithm is the following: Take first one vector and subtract or
! add the two other vectors until you have found the shortest vector.
! Do the same with the other two vectors ... . The iterative method
! used here garantuees that the cell volume will be kept constant and
! all cell vectors occuring during the iterations span primitive cells.
! Two parameters control the flow: ILOOP checks every loop whether a
! vector could be found with smaller or with equal length compared to
! the starting vector by subtraction of an other vector. If ILOOP is not
! set you cannot lower the vector by subtraction of an other vector so
! you have to try it with addition of the vectors. Therefore every loop
! has the following structure: First try it with Ai-ICOUNT*Aj. If the
! shortest vector is found for ICOUNT=0 then try to take Ai+ICOUNT*Aj.
! The loops break if the vector length starts to increase and the vector
! Ai will be set equal to the shortest vector (Ai <-- Ai +/- ICOUNT*Aj).
! IFLAG will be set equal to one whenever a shorter vector was found
! during the total iteration step (6 loops!). If IFLAG is not set we
! found a basis with shortest lattice vectors. If not we should start
! a new iteration ... .

! First try A1-ICOUNT*A2 (loop 1a):
  100 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,1)=RB(1,1)-RB(1,2)
      RB(2,1)=RB(2,1)-RB(2,2)
      RB(3,1)=RB(3,1)-RB(3,2)
      ABSR=SQRT(RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1))
! Breakpoint of loop 1a:
      IF (ABSR>(ABS1+TINY)) GOTO 102
! Found some shorter vector ? Then set IFLAG (and store the length):
      IF (ABSR<(ABS1-TINY)) THEN
         IFLAG=1
         ABS1=ABSR
      END IF
! Vector was not longer than the starting vector. Set ILOOP ... :
      ILOOP=1
      GOTO 100
! After leaving loop 1a we have subtracted A2 one times too often ... :
  102 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,1)=RB(1,1)+RB(1,2)
      RB(2,1)=RB(2,1)+RB(2,2)
      RB(3,1)=RB(3,1)+RB(3,2)
! Nothing found ? Try A1+ICOUNT*A2 (loop 1b):
      IF (ILOOP==0) THEN
  103    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,1)=RB(1,1)+RB(1,2)
         RB(2,1)=RB(2,1)+RB(2,2)
         RB(3,1)=RB(3,1)+RB(3,2)
         ABSR=SQRT(RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1))
! Breakpoint of loop 1b:
         IF (ABSR>(ABS1+TINY)) GOTO 105
! Found some shorter vector ? Then set IFLAG (and store the length):
         IF (ABSR<(ABS1-TINY)) THEN
            IFLAG=1
            ABS1=ABSR
         END IF
         GOTO 103
! After leaving loop 1b we have added A2 one times too often ... :
  105    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,1)=RB(1,1)-RB(1,2)
         RB(2,1)=RB(2,1)-RB(2,2)
         RB(3,1)=RB(3,1)-RB(3,2)
      END IF
! Reset ILOOP before starting the next loop ... :
      ILOOP=0
! Try A1-ICOUNT*A3 (loop 2a):
  110 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,1)=RB(1,1)-RB(1,3)
      RB(2,1)=RB(2,1)-RB(2,3)
      RB(3,1)=RB(3,1)-RB(3,3)
      ABSR=SQRT(RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1))
      IF (ABSR>(ABS1+TINY)) GOTO 112
      IF (ABSR<(ABS1-TINY)) THEN
         IFLAG=1
         ABS1=ABSR
      END IF
      ILOOP=1
      GOTO 110
  112 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,1)=RB(1,1)+RB(1,3)
      RB(2,1)=RB(2,1)+RB(2,3)
      RB(3,1)=RB(3,1)+RB(3,3)
! Nothing found ? Try A1+ICOUNT*A3 (loop 2b):
      IF (ILOOP==0) THEN
  113    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,1)=RB(1,1)+RB(1,3)
         RB(2,1)=RB(2,1)+RB(2,3)
         RB(3,1)=RB(3,1)+RB(3,3)
         ABSR=SQRT(RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1))
         IF (ABSR>(ABS1+TINY)) GOTO 115
         IF (ABSR<(ABS1-TINY)) THEN
            IFLAG=1
            ABS1=ABSR
         END IF
         GOTO 113
  115    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,1)=RB(1,1)-RB(1,3)
         RB(2,1)=RB(2,1)-RB(2,3)
         RB(3,1)=RB(3,1)-RB(3,3)
      END IF
      ILOOP=0
! Try A2-ICOUNT*A1 (loop 3a):
  120 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,2)=RB(1,2)-RB(1,1)
      RB(2,2)=RB(2,2)-RB(2,1)
      RB(3,2)=RB(3,2)-RB(3,1)
      ABSR=SQRT(RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2))
      IF (ABSR>(ABS2+TINY)) GOTO 122
      IF (ABSR<(ABS2-TINY)) THEN
         IFLAG=1
         ABS2=ABSR
      END IF
      ILOOP=1
      GOTO 120
  122 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,2)=RB(1,2)+RB(1,1)
      RB(2,2)=RB(2,2)+RB(2,1)
      RB(3,2)=RB(3,2)+RB(3,1)
! Nothing found ? Try A2+ICOUNT*A1 (loop 3b):
      IF (ILOOP==0) THEN
  123    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,2)=RB(1,2)+RB(1,1)
         RB(2,2)=RB(2,2)+RB(2,1)
         RB(3,2)=RB(3,2)+RB(3,1)
         ABSR=SQRT(RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2))
         IF (ABSR>(ABS2+TINY)) GOTO 125
         IF (ABSR<(ABS2-TINY)) THEN
            IFLAG=1
            ABS2=ABSR
         END IF
         GOTO 123
  125    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,2)=RB(1,2)-RB(1,1)
         RB(2,2)=RB(2,2)-RB(2,1)
         RB(3,2)=RB(3,2)-RB(3,1)
      END IF
      ILOOP=0
! Try A2-ICOUNT*A3 (loop 4a):
  130 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,2)=RB(1,2)-RB(1,3)
      RB(2,2)=RB(2,2)-RB(2,3)
      RB(3,2)=RB(3,2)-RB(3,3)
      ABSR=SQRT(RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2))
      IF (ABSR>(ABS2+TINY)) GOTO 132
      IF (ABSR<(ABS2-TINY)) THEN
         IFLAG=1
         ABS2=ABSR
      END IF
      ILOOP=1
      GOTO 130
  132 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,2)=RB(1,2)+RB(1,3)
      RB(2,2)=RB(2,2)+RB(2,3)
      RB(3,2)=RB(3,2)+RB(3,3)
! Nothing found ? Try A2+ICOUNT*A3 (loop 4b):
      IF (ILOOP==0) THEN
  133    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,2)=RB(1,2)+RB(1,3)
         RB(2,2)=RB(2,2)+RB(2,3)
         RB(3,2)=RB(3,2)+RB(3,3)
         ABSR=SQRT(RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2))
         IF (ABSR>(ABS2+TINY)) GOTO 135
         IF (ABSR<(ABS2-TINY)) THEN
            IFLAG=1
            ABS2=ABSR
         END IF
         GOTO 133
  135    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,2)=RB(1,2)-RB(1,3)
         RB(2,2)=RB(2,2)-RB(2,3)
         RB(3,2)=RB(3,2)-RB(3,3)
      END IF
      ILOOP=0
! Try A3-ICOUNT*A1 (loop 5a):
  140 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,3)=RB(1,3)-RB(1,1)
      RB(2,3)=RB(2,3)-RB(2,1)
      RB(3,3)=RB(3,3)-RB(3,1)
      ABSR=SQRT(RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3))
      IF (ABSR>(ABS3+TINY)) GOTO 142
      IF (ABSR<(ABS3-TINY)) THEN
         IFLAG=1
         ABS3=ABSR
      END IF
      ILOOP=1
      GOTO 140
  142 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,3)=RB(1,3)+RB(1,1)
      RB(2,3)=RB(2,3)+RB(2,1)
      RB(3,3)=RB(3,3)+RB(3,1)
! Nothing found ? Try A3+ICOUNT*A1 (loop 5b):
      IF (ILOOP==0) THEN
  143    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,3)=RB(1,3)+RB(1,1)
         RB(2,3)=RB(2,3)+RB(2,1)
         RB(3,3)=RB(3,3)+RB(3,1)
         ABSR=SQRT(RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3))
         IF (ABSR>(ABS3+TINY)) GOTO 145
         IF (ABSR<(ABS3-TINY)) THEN
            IFLAG=1
            ABS3=ABSR
         END IF
         GOTO 143
  145    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,3)=RB(1,3)-RB(1,1)
         RB(2,3)=RB(2,3)-RB(2,1)
         RB(3,3)=RB(3,3)-RB(3,1)
      END IF
      ILOOP=0
! Try A3-ICOUNT*A2 (loop 6a):
  150 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,3)=RB(1,3)-RB(1,2)
      RB(2,3)=RB(2,3)-RB(2,2)
      RB(3,3)=RB(3,3)-RB(3,2)
      ABSR=SQRT(RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3))
      IF (ABSR>(ABS3+TINY)) GOTO 152
      IF (ABSR<(ABS3-TINY)) THEN
         IFLAG=1
         ABS3=ABSR
      END IF
      ILOOP=1
      GOTO 150
  152 CONTINUE
      ICOUNT=ICOUNT+1
      RB(1,3)=RB(1,3)+RB(1,2)
      RB(2,3)=RB(2,3)+RB(2,2)
      RB(3,3)=RB(3,3)+RB(3,2)
! Nothing found ? Try A3+ICOUNT*A2 (loop 6b):
      IF (ILOOP==0) THEN
  153    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,3)=RB(1,3)+RB(1,2)
         RB(2,3)=RB(2,3)+RB(2,2)
         RB(3,3)=RB(3,3)+RB(3,2)
         ABSR=SQRT(RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3))
         IF (ABSR>(ABS3+TINY)) GOTO 155
         IF (ABSR<(ABS3-TINY)) THEN
            IFLAG=1
            ABS3=ABSR
         END IF
         GOTO 153
  155    CONTINUE
         ICOUNT=ICOUNT+1
         RB(1,3)=RB(1,3)-RB(1,2)
         RB(2,3)=RB(2,3)-RB(2,2)
         RB(3,3)=RB(3,3)-RB(3,2)
      END IF
! Nothing found during loops 1a-6b ? Hurray, we have got it !!
      IF (IFLAG==0) GOTO 200
! Next iteration ... :
  199 ABS1=SQRT(RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1))
      ABS2=SQRT(RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2))
      ABS3=SQRT(RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3))
      IFLAG=0
      ILOOP=0
      GOTO 100
  200 CONTINUE

! Adjustement of the basis to right-hand-sense (by inversion of all
! three lattice vectors if necessary ...):
      CALL CELVOL(RB(1,1),RB(1,2),RB(1,3),OMEGA)
      IF (OMEGA<0) THEN
         RB(1,1)=-1._q*RB(1,1)
         RB(2,1)=-1._q*RB(2,1)
         RB(3,1)=-1._q*RB(3,1)
         RB(1,2)=-1._q*RB(1,2)
         RB(2,2)=-1._q*RB(2,2)
         RB(3,2)=-1._q*RB(3,2)
         RB(1,3)=-1._q*RB(1,3)
         RB(2,3)=-1._q*RB(2,3)
         RB(3,3)=-1._q*RB(3,3)
      END IF

! Warning extremly strange inputs (very large/small ratios 'c/a', 'b/a'
! or cosines very close to +/-1) could sometimes lead to some problems
! due to the finite precision of the machine ...). Here we test some
! parameters and if we find 'strange parameters' we (DO NOT) stop!
      ABS1=RB(1,1)*RB(1,1)+RB(2,1)*RB(2,1)+RB(3,1)*RB(3,1)
      ABS2=RB(1,2)*RB(1,2)+RB(2,2)*RB(2,2)+RB(3,2)*RB(3,2)
      ABS3=RB(1,3)*RB(1,3)+RB(2,3)*RB(2,3)+RB(3,3)*RB(3,3)
      A2XA1=RB(1,1)*RB(1,2)+RB(2,1)*RB(2,2)+RB(3,1)*RB(3,2)
      A3XA1=RB(1,1)*RB(1,3)+RB(2,1)*RB(2,3)+RB(3,1)*RB(3,3)
      A3XA2=RB(1,2)*RB(1,3)+RB(2,2)*RB(2,3)+RB(3,2)*RB(3,3)
      ABS1=SQRT(ABS1)
      ABS2=SQRT(ABS2)
      ABS3=SQRT(ABS3)
      A1XA2=ABS(A2XA1/ABS1/ABS2)
      A1XA3=ABS(A3XA1/ABS1/ABS3)
      A2XA3=ABS(A3XA2/ABS2/ABS3)
      IF (A1XA2>0.995_q) ISTRGE=ISTRGE+1
      IF (A2XA3>0.995_q) ISTRGE=ISTRGE+1
      IF (A1XA3>0.995_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS1)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS1)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS3/ABS2)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS3)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS1/ABS2)<0.001_q) ISTRGE=ISTRGE+1
      IF ((ABS2/ABS3)<0.001_q) ISTRGE=ISTRGE+1

! Now construct several possible combinations of lattice vectors
! and test them all for specific criterions ... :
      ITYP=15
      COS1=1._q
      COS2=1._q
      COS3=1._q
      DO N9=-2,2
      DO N8=-2,2
      DO N7=-2,2
       DO N6=-2,2
       DO N5=-2,2
       ID1=N5*N9-N6*N8
       DO N4=-2,2
        ID2=N6*N7-N4*N9
        ID3=N4*N8-N5*N7
        DO N3=-2,2
        ID4=N3*ID3
        DO N2=-2,2
        ID5=N2*ID2+ID4
        DO N1=-2,2
         IF ((N1*ID1+ID5)==1) THEN
            XB(1,1)=N1*RB(1,1)+N2*RB(1,2)+N3*RB(1,3)
            XB(2,1)=N1*RB(2,1)+N2*RB(2,2)+N3*RB(2,3)
            XB(3,1)=N1*RB(3,1)+N2*RB(3,2)+N3*RB(3,3)
            XB(1,2)=N4*RB(1,1)+N5*RB(1,2)+N6*RB(1,3)
            XB(2,2)=N4*RB(2,1)+N5*RB(2,2)+N6*RB(2,3)
            XB(3,2)=N4*RB(3,1)+N5*RB(3,2)+N6*RB(3,3)
            XB(1,3)=N7*RB(1,1)+N8*RB(1,2)+N9*RB(1,3)
            XB(2,3)=N7*RB(2,1)+N8*RB(2,2)+N9*RB(2,3)
            XB(3,3)=N7*RB(3,1)+N8*RB(3,2)+N9*RB(3,3)
! Lengths of the axes and cosines between the axes:
            ABS1=XB(1,1)*XB(1,1)+XB(2,1)*XB(2,1)+XB(3,1)*XB(3,1)
            ABS2=XB(1,2)*XB(1,2)+XB(2,2)*XB(2,2)+XB(3,2)*XB(3,2)
            ABS3=XB(1,3)*XB(1,3)+XB(2,3)*XB(2,3)+XB(3,3)*XB(3,3)
            A2XA1=XB(1,1)*XB(1,2)+XB(2,1)*XB(2,2)+XB(3,1)*XB(3,2)
            A3XA1=XB(1,1)*XB(1,3)+XB(2,1)*XB(2,3)+XB(3,1)*XB(3,3)
            A3XA2=XB(1,2)*XB(1,3)+XB(2,2)*XB(2,3)+XB(3,2)*XB(3,3)
            ABS2M1=SQRT(ABS1+ABS2-2._q*A2XA1)
            ABS3M1=SQRT(ABS1+ABS3-2._q*A3XA1)
            ABS3M2=SQRT(ABS2+ABS3-2._q*A3XA2)
            ABS2P1=SQRT(ABS1+ABS2+2._q*A2XA1)
            ABS3P1=SQRT(ABS1+ABS3+2._q*A3XA1)
            ABS3P2=SQRT(ABS2+ABS3+2._q*A3XA2)
            A1P2M3=SQRT(ABS1+ABS2+ABS3+2._q*A2XA1-2._q*A3XA1-2._q*A3XA2)
            A2P3M1=SQRT(ABS1+ABS2+ABS3+2._q*A3XA2-2._q*A2XA1-2._q*A3XA1)
            A3P1M2=SQRT(ABS1+ABS2+ABS3+2._q*A3XA1-2._q*A3XA2-2._q*A2XA1)
            ABS123=A2XA1+A3XA1+A3XA2
            ABS1=SQRT(ABS1)
            ABS2=SQRT(ABS2)
            ABS3=SQRT(ABS3)
            A2XA1=A2XA1/ABS1/ABS2
            A3XA1=A3XA1/ABS1/ABS3
            A3XA2=A3XA2/ABS2/ABS3
            A1XA2=ABS(A2XA1)
            A1XA3=ABS(A3XA1)
            A2XA3=ABS(A3XA2)
! Warning extremly strange inputs (very large/small ratios 'c/a', 'b/a'
! or cosines very close to +/-1) could sometimes lead to some problems
! due to the finite precision of the machine ...). Here we test some
! parameters and if we find 'strange parameters', we (DO NOT) stop!
            IF (A1XA2>0.995_q) ISTRGE=ISTRGE+1
            IF (A2XA3>0.995_q) ISTRGE=ISTRGE+1
            IF (A1XA3>0.995_q) ISTRGE=ISTRGE+1
            IF ((ABS3/ABS1)<0.001_q) ISTRGE=ISTRGE+1
            IF ((ABS2/ABS1)<0.001_q) ISTRGE=ISTRGE+1
            IF ((ABS3/ABS2)<0.001_q) ISTRGE=ISTRGE+1
            IF ((ABS1/ABS3)<0.001_q) ISTRGE=ISTRGE+1
            IF ((ABS1/ABS2)<0.001_q) ISTRGE=ISTRGE+1
            IF ((ABS2/ABS3)<0.001_q) ISTRGE=ISTRGE+1

! Following the identification of the crystal type, transformation of
! the basis to 'crystallographic standard settings' and estimate of the
! cell dimensions (using the conventions of routine LATGEN):
            CELLDM(1)=0._q
            CELLDM(2)=0._q
            CELLDM(3)=0._q
            CELLDM(4)=0._q
            CELLDM(5)=0._q
            CELLDM(6)=0._q
            IBRAV=15
! Crystal classes with A1*A2=A1*A3=A2*A3:
            IF ((ABS(A2XA1-A3XA1)<TINY).AND. &
     &                                  (ABS(A2XA1-A3XA2)<TINY)) THEN
! Crystal classes with A1*A2=A1*A3=A2*A3 and |A1|=|A2|=|A3|:
               IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                                    (ABS(ABS2-ABS3)<TINY)) THEN
! Orthogonal axes means simple cubic: (IBRAV=1)
                  IF (A1XA2<TINY) THEN
                     IBRAV=1
                     CELLDM(1)=ABS1
! Cosine(alpha) equal to -1/3 means body-centered cubic: (IBRAV=2)
                  ELSE IF (ABS(A2XA1+(1._q/3._q))<TINY) THEN
                     IBRAV=2
                     CELLDM(1)=ABS1*2._q/SQRT(3._q)
! Cosine(alpha) equal to +1/2 means face-centered cubic: (IBRAV=3)
                  ELSE IF (ABS(A2XA1-0.5_q)<TINY) THEN
                     IBRAV=3
                     CELLDM(1)=ABS1*SQRT(2._q)
! All other angles means trigonal (rhomboedric): (IBRAV=7)
                  ELSE
                     IBRAV=7
                     CELLDM(1)=ABS1
                     CELLDM(4)=A2XA1
                  END IF
! Crystal classes with inequal length of lattice vectors but also with
! A1*A2=A1*A3=A2*A3:
! Orthogonal axes:
               ELSE IF (A1XA2<TINY) THEN
! Two axes with equal lengths means simple tetragonal: (IBRAV=5)
! Adjustment: 'c-axis' shall be the special axis.
                  IF (ABS(ABS1-ABS2)<TINY) THEN
                     IBRAV=5
                     CELLDM(1)=ABS1
                     CELLDM(3)=ABS3/ABS1
! No axes with equal lengths means simple orthorhombic (IBRAV=8):
! Adjustment: Sort the axis by increasing lengths:
                  ELSE IF (((ABS3-ABS2)>TINY).AND. &
     &                                   ((ABS2-ABS1)>TINY)) THEN
                     IBRAV=8
                     CELLDM(1)=ABS1
                     CELLDM(2)=ABS2/ABS1
                     CELLDM(3)=ABS3/ABS1
                  END IF
               END IF
! Crystal classes with A1*A3=A2*A3=/A1*A2:
            ELSE IF (ABS(A3XA1-A3XA2)<TINY) THEN
! One axis orthogonal with respect to the other two axes:
               IF (A1XA3<TINY) THEN
! Equal length of the two nonorthogonal axes:
                  IF (ABS(ABS1-ABS2)<TINY) THEN
! Cosine(alpha) equal to -1/2 means hexagonal: (IBRAV=4)
! Adjustment: 'c-axis' shall be the special axis.
                     IF (ABS(A2XA1+0.5_q)<TINY) THEN
                        IBRAV=4
                        CELLDM(1)=ABS1
                        CELLDM(3)=ABS3/ABS1
! Other angles mean base-centered orthorhombic: (IBRAV=11)
! Adjustment: Cosine between A1 and A2 shall be lower than zero, the
!             'c-axis' shall be the special axis.
                     ELSE IF (A2XA1<(-1._q*TINY)) THEN
                        IBRAV=11
                        CELLDM(1)=ABS2P1
                        CELLDM(2)=ABS2M1/ABS2P1
                        CELLDM(3)=ABS3/ABS2P1
                     END IF
! Different length of the two axes means simple monoclinic (IBRAV=12):
! Adjustment: Cosine(gamma) should be lower than zero, special axis
!             shall be the 'b-axis'(!!) and |A1|<|A3|:
                  ELSE IF ((A2XA1<(-1._q*TINY)).AND. &
     &                                       ((ABS1-ABS2)>TINY)) THEN
                     IBRAV=12
                     CELLDM(1)=ABS2
                     CELLDM(2)=ABS3/ABS2
                     CELLDM(3)=ABS1/ABS2
                     CELLDM(5)=A2XA1
                     YB(1)=XB(1,3)
                     YB(2)=XB(2,3)
                     YB(3)=XB(3,3)
                     XB(1,3)=XB(1,1)
                     XB(2,3)=XB(2,1)
                     XB(3,3)=XB(3,1)
                     XB(1,1)=XB(1,2)
                     XB(2,1)=XB(2,2)
                     XB(3,1)=XB(3,2)
                     XB(1,2)=YB(1)
                     XB(2,2)=YB(2)
                     XB(3,2)=YB(3)
                  END IF
! Arbitrary angles between the axes:
               ELSE
! |A1|=|A2|=|A3| means body-centered tetragonal (IBRAV=6):
! Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
! orthogonal to one another and (adjustment!): |A1+A3|=|A2+A3|/=|A1+A2|
                  IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                   (ABS(ABS2-ABS3)<TINY).AND. &
     &                      (ABS(ABS3P1-ABS3P2)<TINY).AND. &
     &                          (ABS(ABS2P1-ABS3P1)>TINY).AND. &
     &                             (ABS(ABS3*ABS3+ABS123)<TINY)) THEN
                     IBRAV=6
                     CELLDM(1)=ABS3P1
                     CELLDM(3)=ABS2P1/ABS3P1
! |A1|=|A2|=/|A3| means base-centered monoclinic (IBRAV=13):
! Adjustement: The cosine between A1 and A3 as well as the cosine
!              between A2 and A3 should be lower than zero.
                  ELSE IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                                (A3XA1<(-1._q*TINY)).AND. &
     &                                       (A3XA2<(-1._q*TINY))) THEN
                     IBRAV=13
                     CELLDM(1)=ABS2P1
                     CELLDM(2)=ABS2M1/ABS2P1
                     CELLDM(3)=ABS3/ABS2P1
                     CELLDM(5)=A3XA2
                  END IF
               END IF
! Crystal classes with A1*A2=/A1*A3=/A2*A3
            ELSE
! |A1|=|A2|=|A3| means body-centered orthorhombic (IBRAV=9):
! Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
! orthogonal to one another and (adjustment!): |A1+A2|>|A1+A3|>|A2+A3|
               IF ((ABS(ABS1-ABS2)<TINY).AND. &
     &                (ABS(ABS2-ABS3)<TINY).AND. &
     &                    ((ABS3P1-ABS3P2)>TINY).AND. &
     &                         ((ABS2P1-ABS3P1)>TINY).AND. &
     &                             (ABS(ABS3*ABS3+ABS123)<TINY)) THEN
                  IBRAV=9
                  CELLDM(1)=ABS3P2
                  CELLDM(2)=ABS3P1/ABS3P2
                  CELLDM(3)=ABS2P1/ABS3P2
! |A1|=|A2-A3| and |A2|=|A1-A3| and |A3|=|A1-A2| means face-centered
! orthorhombic (IBRAV=10):
! Adjustment: |A1+A2-A3|>|A1+A3-A2|>|A2+A3-A1|
               ELSE IF ((ABS(ABS2M1-ABS3)<TINY).AND. &
     &                      (ABS(ABS3M1-ABS2)<TINY).AND. &
     &                          (ABS(ABS3M2-ABS1)<TINY).AND. &
     &                               ((A1P2M3-A3P1M2)>TINY).AND. &
     &                                   ((A3P1M2-A2P3M1)>TINY)) THEN
                  IBRAV=10
                  CELLDM(1)=A2P3M1
                  CELLDM(2)=A3P1M2/A2P3M1
                  CELLDM(3)=A1P2M3/A2P3M1
! Now there exists only one further possibility - triclinic (IBRAV=14):
! Adjustment: All three cosines shall be greater than zero and ordered:
               ELSE IF ((A2XA1>A3XA1).AND.(A3XA1>A3XA2).AND. &
     &                                             (A3XA2>TINY)) THEN
                  IBRAV=14
                  CELLDM(1)=ABS1
                  CELLDM(2)=ABS2/ABS1
                  CELLDM(3)=ABS3/ABS1
                  CELLDM(4)=A3XA2
                  CELLDM(5)=A3XA1
                  CELLDM(6)=A2XA1
               END IF
            END IF

! Have we found some lattice with lower index IBRAV as all previous?
            IF ((IBRAV<ITYP).OR.((IBRAV==ITYP).AND. &
     &                 ((ABS(CELLDM(4))<(COS1-1.E-9_q)).AND. &
     &                      (ABS(CELLDM(5))<(COS2-1.E-9_q)).AND. &
     &                          (ABS(CELLDM(6))<(COS3-1.E-9_q))))) THEN
               ITYP=IBRAV
               COS1=ABS(CELLDM(4))
               COS2=ABS(CELLDM(5))
               COS3=ABS(CELLDM(6))
               CELDIM(1)=CELLDM(1)
               CELDIM(2)=CELLDM(2)
               CELDIM(3)=CELLDM(3)
               CELDIM(4)=CELLDM(4)
               CELDIM(5)=CELLDM(5)
               CELDIM(6)=CELLDM(6)
               A1(1)=XB(1,1)
               A1(2)=XB(2,1)
               A1(3)=XB(3,1)
               A2(1)=XB(1,2)
               A2(2)=XB(2,2)
               A2(3)=XB(3,2)
               A3(1)=XB(1,3)
               A3(2)=XB(2,3)
               A3(3)=XB(3,3)
            END IF
         END IF
        ENDDO
        ENDDO
        ENDDO
       ENDDO
       ENDDO
       ENDDO
      ENDDO
      ENDDO
      ENDDO

! Now we got the 'final answer' which lattice type we have. Now we have
! of course changed the original basis vector totally (interchanged or
! inverted and probably linearly combined ...?). If the original input
! vectors would have given the same answer (same IBRAV, same CELDIM) it
! would probably be nice to take them instead of a total different basis
! and so check against the result from our first trial at the beginning.
      IF (IBRINP==ITYP) THEN
         IFLAG=0
! Consistency check ... :
         IF (ABS(CELDIM(1)-CDMINP(1))>TINY) IFLAG=1
         IF (ABS(CELDIM(2)-CDMINP(2))>TINY) IFLAG=1
         IF (ABS(CELDIM(3)-CDMINP(3))>TINY) IFLAG=1
         IF (ABS(CELDIM(4)-CDMINP(4))>TINY) IFLAG=1
         IF (ABS(CELDIM(5)-CDMINP(5))>TINY) IFLAG=1
         IF (ABS(CELDIM(6)-CDMINP(6))>TINY) IFLAG=1
         IF (IFLAG==0) THEN
! Now take really the original input vectors ... :
            A1(1)=SA1(1)
            A1(2)=SA1(2)
            A1(3)=SA1(3)
            A2(1)=SA2(1)
            A2(2)=SA2(2)
            A2(3)=SA2(3)
            A3(1)=SA3(1)
            A3(2)=SA3(2)
            A3(3)=SA3(3)
         ELSE IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
            WRITE(PRCHAN,*) ' Warning from LATTYP: Got some problem '// &
     &                                           'with cell dimensions!'
            WRITE(PRCHAN,*) ' Tried to take original basis but could '// &
     &                                          'not verify dimensions!'
         END IF
      END IF

! Print result (or not ...)
      IF ((PRCHAN<0).OR.(PRCHAN>99)) RETURN
! Problems with strange parameters?
!     IF (ISTRGE>0) THEN
!              WRITE(PRCHAN,*)
!    C      ' Warning routine LATTYP! Problems with strange parameters.'
!              WRITE(PRCHAN,*)
!    C     ' The detection could become potentially unsafe ... !',ISTRGE
!     END IF
      IF (ITYP==1) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a simple cubic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
      ELSE IF (ITYP==2) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a body centered cubic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
      ELSE IF (ITYP==3) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a face centered cubic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
      ELSE IF (ITYP==4) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a hexagonal cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==5) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a simple tetragonal cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==6) THEN
         WRITE(PRCHAN,*) &
     &                 ' LATTYP: Found a body centered tetragonal cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==7) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a trigonal (rhomboedric) cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(alpha) =',CELDIM(4)
      ELSE IF (ITYP==8) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a simple orthorhombic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==9) THEN
         WRITE(PRCHAN,*) &
     &               ' LATTYP: Found a body centered orthorhombic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==10) THEN
         WRITE(PRCHAN,*) &
     &               ' LATTYP: Found a face centered orthorhombic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==11) THEN
         WRITE(PRCHAN,*) &
     &               ' LATTYP: Found a base centered orthorhombic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
      ELSE IF (ITYP==12) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a simple monoclinic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(beta)  =',CELDIM(5)
      ELSE IF (ITYP==13) THEN
         WRITE(PRCHAN,*) &
     &                 ' LATTYP: Found a base centered monoclinic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(beta)  =',CELDIM(5)
      ELSE IF (ITYP==14) THEN
         WRITE(PRCHAN,*) ' LATTYP: Found a triclinic cell.'
         WRITE(PRCHAN,'(A,F17.10)') ' ALAT       =',CELDIM(1)
         WRITE(PRCHAN,'(A,F17.10)') ' B/A-ratio  =',CELDIM(2)
         WRITE(PRCHAN,'(A,F17.10)') ' C/A-ratio  =',CELDIM(3)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(alpha) =',CELDIM(4)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(beta)  =',CELDIM(5)
         WRITE(PRCHAN,'(A,F17.10)') ' COS(gamma) =',CELDIM(6)
      END IF
      WRITE(PRCHAN,*) ' '
      WRITE(PRCHAN,*) ' Lattice vectors:'
      WRITE(PRCHAN,*) ' '
      WRITE(PRCHAN,400) ' A1 = (',A1(1),',',A1(2),',',A1(3),')'
      WRITE(PRCHAN,400) ' A2 = (',A2(1),',',A2(2),',',A2(3),')'
      WRITE(PRCHAN,400) ' A3 = (',A3(1),',',A3(2),',',A3(3),')'
  400 FORMAT(A,F15.10,A,F15.10,A,F15.10,A)

! Uff ... ! Good bye.
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE RECIPS(A,A1,A2,A3,B1,B2,B3)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine RECIPS sets up the reciprocal lattice vectors B1 ,B2 and . *
!   B3 for a given set of crystallographic vectors A1, A2, A3.         *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      A1, A2 and A3 are the three crystallographic vectors of the     *
!        primitive unit cell (given in cartesian coordinates x,y,z)    *
!        of the real space lattice given in units of bohrs ('a').      *
!        A contains the 'lattice constant'.                            *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      B1, B2 and B3 are the three crystallographic vectors of the     *
!        primitive unit cell (given in cartesian coordinates x,y,z) of *
!        the k-space (reciprocal space) lattice in units of '2*pi/a'.  *
!                                                                      *
!***********************************************************************

      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)

      DEN=0._q
      I=1
      J=2
      K=3
      S=1._q
    1 DO 2 IPERM=1,3
         DEN=DEN+S*A1(I)*A2(J)*A3(K)
         L=I
         I=J
         J=K
         K=L
    2 CONTINUE
      I=2
      J=1
      K=3
      S=-S
      IF (S<0._q) GOTO 1
      I=1
      J=2
      K=3
      DEN=A/ABS(DEN)
      DO 5 IR=1,3
         B1(IR)=DEN*(A2(J)*A3(K)-A2(K)*A3(J))
         B2(IR)=DEN*(A3(J)*A1(K)-A3(K)*A1(J))
         B3(IR)=DEN*(A1(J)*A2(K)-A1(K)*A2(J))
         L=I
         I=J
         J=K
         K=L
    5 CONTINUE
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE PBC(X1,Y1,Z1,X2,Y2,Z2,APBC,IBRAV,CELLDM)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine PBC calculates the periodic boundary conditions for all    *
!   types of lattices.                                                 *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      X1, Y1 and Z1 are (cartesian) input coordinates, APBC the       *
!      'lattice constant' of the microcell for which the periodic      *
!      boundary conditions have to be calculated, IBRAV specifies      *
!      the type of bravais lattice and CELLDM gives the dimensions     *
!      of the primitive unit cell as given in subroutine 'LATGEN'.     *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      X2, Y2 and Z3 are the 'correct' (cartesian) coordinates lying   *
!      inside the microcell (located at (0,0,0)). In all directions    *
!      the coordinates will be translated into the interval [-0.5,0.5) *
!      so that (0,0,0) will ly in the center of the cell.              *
!                                                                      *
!***********************************************************************

      DIMENSION A1(3),A2(3),A3(3),B1(3),B2(3),B3(3),CELLDM(6)

! Set up the primitive lattice vectors and reciprocal lattice vectors
      ALAT=CELLDM(1)
      CELLDM(1)=APBC
      CALL LATGEN(IBRAV,CELLDM,A1,A2,A3,OMEGA)
      CELLDM(1)=ALAT
      CALL RECIPS(1._q,A1,A2,A3,B1,B2,B3)

! Calculate the 'expansion coefficients' (using (Ai)o(Bj)=delta(i,j))
! of the expansion (X1,Y1,Z1) != R1*A1+R2*A2+R3*A3 :
      R1=X1*B1(1)+Y1*B1(2)+Z1*B1(3)
      R2=X1*B2(1)+Y1*B2(2)+Z1*B2(3)
      R3=X1*B3(1)+Y1*B3(2)+Z1*B3(3)

! Take the 'expansion coefficients' modulo [-1/2,1/2) :
      R1=R1-INT(R1)
      R1=MOD(R1+100.5_q,1._q)-0.5_q
      R2=R2-INT(R2)
      R2=MOD(R2+100.5_q,1._q)-0.5_q
      R3=R3-INT(R3)
      R3=MOD(R3+100.5_q,1._q)-0.5_q

! Construct the new vector R1*A1+R2*A2+R3*A3 = (X2,Y2,Z2) :
      X2=R1*A1(1)+R2*A2(1)+R3*A3(1)
      Y2=R1*A1(2)+R2*A2(2)+R3*A3(2)
      Z2=R1*A1(3)+R2*A2(3)+R3*A3(3)
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE CELVOL(A1,A2,A3,OMEGA)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine CELVOL calculates the volume of the cell spanned by the    *
!   vectors A1, A2 and A3.                                             *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      A1, A2 and A3 are the three crystallographic vectors of the     *
!        primitive unit cell (given in cartesian coordinates x,y,z)    *
!        of the real space lattice given in units of bohrs ('a').      *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      OMEGA is the cell volume. (Warning: Can also be negative !!)    *
!                                                                      *
!***********************************************************************

      DIMENSION A1(3),A2(3),A3(3)

      OMEGA=0._q
      S=1._q
      I=1
      J=2
      K=3
    1 DO 2 IPERM=1,3
         OMEGA=OMEGA+S*A1(I)*A2(J)*A3(K)
         L=I
         I=J
         J=K
         K=L
    2 CONTINUE
      I=2
      J=1
      K=3
      S=-S
      IF (S<0._q) GOTO 1
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE LATORD(TAU,NA,NAX,INDEX,WRK,IPERM)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine LATORD orders the atomic positions inside a supercell by   *
!   a unique ordering scheme described below.                          *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      TAU contains the atomic positions (for one species) to be       *
!        ordered.                                                      *
!      NA contains the number of atomic positions                      *
!      NAX is a dimensioning parameter for the arrays TAU, INDEX and   *
!        WRK where WRK is a pure work array and INDEX is a work array  *
!        and an output array.                                          *
!      IPERM is a permutation index that allows to permutate the three *
!        coordinates of each atoms cyclically before ordering (doing   *
!        the 'inverse' permutation after the ordering). Allowed values *
!        for IPERM are -3,-2,-1,0,1,2 and 3.                           *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      TAU will be changed on output and contains the ordered atomic   *
!        positions (for the given species) ... .                       *
!      INDEX contains the permutation table for the positions.         *
!                                                                      *
!                                                                      *
!   Ordering scheme: (short description)                               *
!   ------------------------------------                               *
!                                                                      *
!     First order the 'x-coordinates', then order the 'y-coordinates'  *
!     for all positions with equal 'x-coordinates' and finally order   *
!     the 'z-coordinates' for all atomic positions which have equal    *
!     'x-coordinates' and equal 'y-coordinates'. This gives a unique   *
!     order for a given set of atomic positions which allows a direct  *
!     comparison of two different supercells by a simple one-to-one    *
!     correspondence of two data arrays into which the positions have  *
!     been stored. Notice that you can change the ordering sequence    *
!     by using the parameter IPERM ... .                               *
!                                                                      *
!***********************************************************************

      DIMENSION TAU(NAX,3),INDEX(NAX),WRK(NAX,3)

! One atom is a trivial case ... :
      INDEX(1)=1
      IF (NA==1) RETURN

! Permutation of the x/y/z-coordinates (of the ordering sequence ...):
      IF (IPERM==-3) THEN
         DO 1010 IA=1,NA
            WRK(IA,1)=TAU(IA,2)
            WRK(IA,2)=TAU(IA,1)
            WRK(IA,3)=TAU(IA,3)
 1010    CONTINUE
      ELSE IF (IPERM==-2) THEN
         DO 1020 IA=1,NA
            WRK(IA,1)=TAU(IA,3)
            WRK(IA,2)=TAU(IA,2)
            WRK(IA,3)=TAU(IA,1)
 1020    CONTINUE
      ELSE IF (IPERM==-1) THEN
         DO 1030 IA=1,NA
            WRK(IA,1)=TAU(IA,1)
            WRK(IA,2)=TAU(IA,3)
            WRK(IA,3)=TAU(IA,2)
 1030    CONTINUE
      ELSE IF ((IPERM==0).OR.(IPERM==3)) THEN
         DO 1040 IA=1,NA
            WRK(IA,1)=TAU(IA,1)
            WRK(IA,2)=TAU(IA,2)
            WRK(IA,3)=TAU(IA,3)
 1040    CONTINUE
      ELSE IF (IPERM==1) THEN
         DO 1050 IA=1,NA
            WRK(IA,1)=TAU(IA,2)
            WRK(IA,2)=TAU(IA,3)
            WRK(IA,3)=TAU(IA,1)
 1050    CONTINUE
      ELSE IF (IPERM==2) THEN
         DO 1060 IA=1,NA
            WRK(IA,1)=TAU(IA,3)
            WRK(IA,2)=TAU(IA,1)
            WRK(IA,3)=TAU(IA,2)
 1060    CONTINUE
      ELSE
         CALL ERROR(' LATORD',' Invalid permutation index.',IPERM)
      END IF
      DO 1070 IA=1,NA
         TAU(IA,1)=WRK(IA,1)
         TAU(IA,2)=WRK(IA,2)
         TAU(IA,3)=WRK(IA,3)
 1070 CONTINUE

! Order the 'x-coordinates':
      CALL ORDER(NA,TAU(1,1),INDEX)
      DO I=1,3
       DO IA=1,NA
          WRK(IA,I)=TAU(INDEX(IA),I)
       ENDDO
      ENDDO
      DO I=1,3
       DO IA=1,NA
          TAU(IA,I)=WRK(IA,I)
       ENDDO
      ENDDO

! Now search all position with equal 'x-coordinates':
      OLD=TAU(1,1)
      WRK(1,1)=1._q
      WRK(1,2)=1._q
      WRK(1,3)=FLOAT(INDEX(1))
      ICOUNT=1
      DO 3 IA=2,NA
         WRK(IA,3)=INDEX(IA)
         IF (ABS(OLD-TAU(IA,1))>TINY) THEN
            OLD=TAU(IA,1)
            ICOUNT=ICOUNT+1
            WRK(ICOUNT,1)=FLOAT(IA)
            WRK(ICOUNT,2)=1._q
         ELSE
            WRK(ICOUNT,2)=WRK(ICOUNT,2)+1._q
         END IF
    3 CONTINUE

! Now take every group of positions with equal 'x-coordinates' and order
! it with respect to the 'y-coordinates':
      DO 5 IORD=1,ICOUNT
         NBEG=NINT(WRK(IORD,1))
         NORD=NINT(WRK(IORD,2))
         CALL ORDER(NORD,TAU(NBEG,2),INDEX(NBEG))
! The index table will be modified in such a way that we get a unique
! table for all NA positions ... :
         DO I=0,NORD-1
            INDEX(NBEG+I)=INDEX(NBEG+I)+NBEG-1
         ENDDO
    5 CONTINUE
! Permutate the positions and the index table of the first ordering step
! which has been saved into WRK(IA,3) ... :
      DO 8 IA=1,NA-1
         ICURR=IA
    6    IF (INDEX(ICURR)/=IA) THEN
            DO K=1,3
               TEMP=TAU(ICURR,K)
               TAU(ICURR,K)=TAU(INDEX(ICURR),K)
               TAU(INDEX(ICURR),K)=TEMP
            ENDDO
            TEMP=WRK(ICURR,3)
            WRK(ICURR,3)=WRK(INDEX(ICURR),3)
            WRK(INDEX(ICURR),3)=TEMP
            IT=ICURR
            ICURR=INDEX(ICURR)
            INDEX(IT)=IT
            IF (INDEX(ICURR)==IA) THEN
               INDEX(ICURR)=ICURR
               GOTO 8
            END IF
            GOTO 6
         END IF
    8 CONTINUE

! Now search all position with equal 'x-coordinates' and with equal
! 'y-coordinates' and order them with respect to the 'z-coordinate'.
! The table for the positions with equal 'x-coordinates' is still
! present in WRK(I,1) and WRK(I,2) so we need now only search goups with
! equal 'y-coordinates' within each group with equal 'x-coordiante':
      DO 11 IORD=1,ICOUNT
         NBEG=NINT(WRK(IORD,1))
         NORD=NINT(WRK(IORD,2))
         OLD=TAU(NBEG,2)
         IBEG=NBEG
         DO 10 IA=NBEG+1,NBEG+NORD
           IF ((ABS(OLD-TAU(MIN(IA,NA),2))>TINY).OR.(IA==(NBEG+NORD))) THEN
! Here a new 'y-coordinate' occurs. Order all 'z-coordinates' of the
! previous group of positions ...:
               CALL ORDER(IA-IBEG,TAU(IBEG,3),INDEX(IBEG))
! The index table will be modified in such a way that we get a unique
! table for all NA positions ... :
               DO I=0,IA-IBEG-1
                  INDEX(IBEG+I)=INDEX(IBEG+I)+IBEG-1
               ENDDO
               OLD=TAU(MIN(IA,NA),2)
               IBEG=IA
            END IF
   10    CONTINUE
   11 CONTINUE
! Final permutation of the positions according to the INDEX table for
! the third ordering step performed above and permutation of the index
! table that connects the actual positions after the first two ordering
! steps with the initial positions (stored into WRK(IA,3) ... ):
      DO 14 IA=1,NA-1
         ICURR=IA
   12    IF (INDEX(ICURR)/=IA) THEN
            DO K=1,3
               TEMP=TAU(ICURR,K)
               TAU(ICURR,K)=TAU(INDEX(ICURR),K)
               TAU(INDEX(ICURR),K)=TEMP
            ENDDO
            TEMP=WRK(ICURR,3)
            WRK(ICURR,3)=WRK(INDEX(ICURR),3)
            WRK(INDEX(ICURR),3)=TEMP
            IT=ICURR
            ICURR=INDEX(ICURR)
            INDEX(IT)=IT
            IF (INDEX(ICURR)==IA) THEN
               INDEX(ICURR)=ICURR
               GOTO 14
            END IF
            GOTO 12
         END IF
   14 CONTINUE

! Copy the permutation table that connects the actual sequence of atomic
! position with the initial sequence of positions into array INDEX:
      DO IA=1,NA
         INDEX(IA)=NINT(WRK(IA,3))
      ENDDO

! Back-permutation of the x/y/z-coordinates:
      IF (IPERM==-3) THEN
         DO 2010 IA=1,NA
            WRK(IA,1)=TAU(IA,2)
            WRK(IA,2)=TAU(IA,1)
            WRK(IA,3)=TAU(IA,3)
 2010    CONTINUE
      ELSE IF (IPERM==-2) THEN
         DO 2020 IA=1,NA
            WRK(IA,1)=TAU(IA,3)
            WRK(IA,2)=TAU(IA,2)
            WRK(IA,3)=TAU(IA,1)
 2020    CONTINUE
      ELSE IF (IPERM==-1) THEN
         DO 2030 IA=1,NA
            WRK(IA,1)=TAU(IA,1)
            WRK(IA,2)=TAU(IA,3)
            WRK(IA,3)=TAU(IA,2)
 2030    CONTINUE
      ELSE IF ((IPERM==0).OR.(IPERM==3)) THEN
         DO 2040 IA=1,NA
            WRK(IA,1)=TAU(IA,1)
            WRK(IA,2)=TAU(IA,2)
            WRK(IA,3)=TAU(IA,3)
 2040    CONTINUE
      ELSE IF (IPERM==1) THEN
         DO 2050 IA=1,NA
            WRK(IA,1)=TAU(IA,3)
            WRK(IA,2)=TAU(IA,1)
            WRK(IA,3)=TAU(IA,2)
 2050    CONTINUE
      ELSE IF (IPERM==2) THEN
         DO 2060 IA=1,NA
            WRK(IA,1)=TAU(IA,2)
            WRK(IA,2)=TAU(IA,3)
            WRK(IA,3)=TAU(IA,1)
 2060    CONTINUE
      END IF
      DO 2070 IA=1,NA
         TAU(IA,1)=WRK(IA,1)
         TAU(IA,2)=WRK(IA,2)
         TAU(IA,3)=WRK(IA,3)
 2070 CONTINUE

! Uff ... ! Bye bye ...
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE ORDER(N,ARRIN,INDEX)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Routine ORDER indexes a real array by ascending order. The scheme  *
!   bases on a heapsort algorithm which is described in the book of    *
!   W.H.PRESS et al.: NUMERICAL RECIPIES, CAMBRIDGE UNIV. PR., 1987.   *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      N contains the dimension of array ARRIN (number of data).       *
!      ARRIN contains the input numbers to be sorted (to be indexed).  *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      INDEX contains the permutation table for the ordered numbers.   *
!                                                                      *
!                                                                      *
!***********************************************************************

      DIMENSION ARRIN(N),INDEX(N)

      DO 1 J=1,N
         INDEX(J)=J
    1 CONTINUE
      IF (N==1) RETURN
      L=N/2+1
      IR=N
10    CONTINUE
      IF (L>1) THEN
         L=L-1
         INDEXT=INDEX(L)
         QQ=ARRIN(INDEXT)
      ELSE
         INDEXT=INDEX(IR)
         QQ=ARRIN(INDEXT)
         INDEX(IR)=INDEX(1)
         IR=IR-1
         IF (IR==1)THEN
            INDEX(1)=INDEXT
            RETURN
         END IF
      END IF
      I=L
      J=L+L
20    IF (J<=IR) THEN
         IF (J<IR) THEN
            IF(ARRIN(INDEX(J))<ARRIN(INDEX(J+1))) J=J+1
         END IF
         IF (QQ<ARRIN(INDEX(J))) THEN
            INDEX(I)=INDEX(J)
            I=J
            J=J+J
         ELSE
            J=IR+1
         END IF
         GOTO 20
      END IF
      INDEX(I)=INDEXT
      GOTO 10
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE PRICEL(IBRAV,CELDIM,A1,A2,A3,TAU,P1,P2,P3,PTRANS,NCELL, &
     &                IPTYP,PDIM,NSP,NSX,NA,NAX,TAUROT,INDEX,WRK,PRCHAN)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Subroutine PRICEL detects the generating cell of a supercell.      *
!   First all possible 'nontrivial' translations will be detected      *
!   by testing all difference vectors between atom '1' and all the     *
!   other atoms of the cell whether they can reproduce the lattice     *
!   when translating all atoms by this vector. The strategy will be    *
!   similar to that used in subroutine CHKSYM of program package       *
!   SYMLIB to detect nontrivial translations being associated with     *
!   point symmetry operations.                                         *
!   When all possible translations have been detected we will try      *
!   to detect the type of the generating cell by using subroutine      *
!   LATTYP of this package. Therefore we only have to find one         *
!   (arbitrary) 'primitive cell' (cell of smallest volume) that        *
!   can be spanned by three of all possible translations (including    *
!   the three trivial translations ...).                               *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      IBRAV and CELDIM(6) are the lattice type and the dimensioning   *
!              parameters of the supercell.                            *
!      A1, A2 and A3 are the lattice vectors defining the supercell.   *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (similar as in routine SETGRP)!      *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!      PRCHAN defines the unit number for printing output information. *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      P1, P2 and P3 are the lattice vectors of the primitive cell     *
!            generating the given supercell.                           *
!      PTRANS(NAX,3,NSX) contains the crystal coordinates of all       *
!            'primitive' translations within the supercell.            *
!                                                                      *
!      IPTYP returns the bravais lattice type of the generating        *
!            primitive cell according to the key given in subroutines  *
!            LATTYP and LATGEN.                                        *
!      PDIM(6) contains the cell dimensioning parameters of the        *
!            generating primitive cell according to the conventions    *
!            used in subroutines LATGEN and LATTYP.                    *
!                                                                      *
!      NCELL is an integer value which returns the detected number of  *
!            primitive cells within one supercell.                     *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TNTRAN,EXIST,TCONT
      INTEGER PRCHAN

      DIMENSION A1(3),A2(3),A3(3),NA(NSP),TAU(NAX,3,NSX),TRA(3),PDIM(6)
      DIMENSION P1(3),P2(3),P3(3),B1(3),B2(3),B3(3),DIF(3),WRK(NAX+2,3)
      DIMENSION TAUROT(NAX,3,NSX),INDEX(NAX+2),TAUSAV(3),PTRANS(NAX+2,3)
      DIMENSION CELDIM(6)

      EQUIVALENCE (TNTRAN,EXIST)

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
      ISTART=1
      IMINST=1
! For all atomic species ... :
      DO 2 IS=1,NSP
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
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
    1    CONTINUE
! Order the atomic coordiantes (for this species) ... :
         CALL LATORD(TAU(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
    2 CONTINUE
! Trivial case: There exists only one atom of species ISMIN. So we
! have already given the primitive cell ... :
      IF (NA(ISMIN)==1) THEN
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
            WRITE(PRCHAN,'(A)') ' Subroutine PRICEL returns:'
            WRITE(PRCHAN,'(A)') &
     &                    ' Original cell was already a primitive cell.'
            WRITE(PRCHAN,'(A)') ' '
         END IF
         NCELL=1
         PTRANS(1,1)=0._q
         PTRANS(1,2)=0._q
         PTRANS(1,3)=0._q
         P1(1)=A1(1)
         P1(2)=A1(2)
         P1(3)=A1(3)
         P2(1)=A2(1)
         P2(2)=A2(2)
         P2(3)=A2(3)
         P3(1)=A3(1)
         P3(2)=A3(2)
         P3(3)=A3(3)
         IPTYP=IBRAV
         PDIM(1)=CELDIM(1)
         PDIM(2)=CELDIM(2)
         PDIM(3)=CELDIM(3)
         PDIM(4)=CELDIM(4)
         PDIM(5)=CELDIM(5)
         PDIM(6)=CELDIM(6)
         RETURN
      END IF
! Save the rotated coordinate of the first atom of species ISMIN:
      IF (TCONT) THEN
         TAUSAV(1)=TAU(IMINST,1,1)
         TAUSAV(2)=TAU(IMINST,2,1)
         TAUSAV(3)=TAU(IMINST,3,1)
         ISPMIN=1
      ELSE
         TAUSAV(1)=TAU(1,1,ISMIN)
         TAUSAV(2)=TAU(1,2,ISMIN)
         TAUSAV(3)=TAU(1,3,ISMIN)
         ISPMIN=ISMIN
      END IF
! All difference vectors between the first atom and all other atoms
! of species ISMIN could be a translation vector that reproduces the
! lattice ... . So test all possibilities:
      ITRANS=3
      PTRANS(1,1)=1._q
      PTRANS(1,2)=0._q
      PTRANS(1,3)=0._q
      PTRANS(2,1)=0._q
      PTRANS(2,2)=1._q
      PTRANS(2,3)=0._q
      PTRANS(3,1)=0._q
      PTRANS(3,2)=0._q
      PTRANS(3,3)=1._q
      DO 7 IATEST=IMINST+1,NA(ISMIN)+IMINST-1
! Set up the 'test vector' TRA ... :
         TRA(1)=TAU(IATEST,1,ISPMIN)-TAUSAV(1)
         TRA(2)=TAU(IATEST,2,ISPMIN)-TAUSAV(2)
         TRA(3)=TAU(IATEST,3,ISPMIN)-TAUSAV(3)
! TRA could possibly contain trivial translations:
         TRA(1)=MOD((TRA(1)+100._q),1._q)
         TRA(2)=MOD((TRA(2)+100._q),1._q)
         TRA(3)=MOD((TRA(3)+100._q),1._q)
! for reasons of safety:
         IF ((ABS(TRA(1)-1._q))<(TINY*0.5_q)) TRA(1)=0._q
         IF ((ABS(TRA(2)-1._q))<(TINY*0.5_q)) TRA(2)=0._q
         IF ((ABS(TRA(3)-1._q))<(TINY*0.5_q)) TRA(3)=0._q
! Translate the atomic coordinates by TRA ...
         ISTART=1
         DO 4 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 3 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAU(IA,1,ISP)+TRA(1)
               TAUROT(IA,2,ISP)=TAU(IA,2,ISP)+TRA(2)
               TAUROT(IA,3,ISP)=TAU(IA,3,ISP)+TRA(3)
! ... impose the periodic boundary condition on these coordinates ...
               TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! ... ensuring that all surface coordinates are put to -0.5 ...
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
    3       CONTINUE
! ... and order the coordinates:
            CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
    4    CONTINUE
! Now compare the two lattices 'one-by-one' whether they are identical:
         TNTRAN=.TRUE.
! For all atoms of the unit cell ... :
         ISTART=1
         DO 6 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 5 IA=ISTART,NA(IS)+ISTART-1
! Take the difference of the translated and the original coordinates -
! the result should be (0.,0.,0.)!
               DIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               DIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               DIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! DIF may only differ from (0.,0.,0.) by some trivial translation:
               DIF(1)=MOD((DIF(1)+100._q),1._q)
               DIF(2)=MOD((DIF(2)+100._q),1._q)
               DIF(3)=MOD((DIF(3)+100._q),1._q)
! for reasons of safety:
               IF ((ABS(DIF(1)-1._q))<(TINY*0.5_q)) DIF(1)=0._q
               IF ((ABS(DIF(2)-1._q))<(TINY*0.5_q)) DIF(2)=0._q
               IF ((ABS(DIF(3)-1._q))<(TINY*0.5_q)) DIF(3)=0._q
! If DIF is not equal (0.,0.,0.) TNTRAN=TNTRAN.AND.FALSE and so it
! remains FALSE forever ... (if only one position is not reproduced!).
! Only if all DIFs are zero TNTRAN (starting with the value TRUE) will
! remain TRUE (that means we found an allowed translation vector).
               TNTRAN=TNTRAN.AND.((ABS(DIF(1))<TINY) &
     &                           .AND.(ABS(DIF(2))<TINY) &
     &                                .AND.(ABS(DIF(3))<TINY))
    5       CONTINUE
    6    CONTINUE
! Check was successful ... :
         IF (TNTRAN) THEN
            ITRANS=ITRANS+1
! Save the detected translation vector ... :
            PTRANS(ITRANS,1)=TRA(1)
            PTRANS(ITRANS,2)=TRA(2)
            PTRANS(ITRANS,3)=TRA(3)
         END IF
    7 CONTINUE
! Nothing found ... :
      IF (ITRANS==3) THEN
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
            WRITE(PRCHAN,'(A)') ' Subroutine PRICEL returns:'
            WRITE(PRCHAN,'(A)') &
     &                    ' Original cell was already a primitive cell.'
            WRITE(PRCHAN,'(A)') ' '
         END IF
         NCELL=1
         PTRANS(1,1)=0._q
         PTRANS(1,2)=0._q
         PTRANS(1,3)=0._q
         P1(1)=A1(1)
         P1(2)=A1(2)
         P1(3)=A1(3)
         P2(1)=A2(1)
         P2(2)=A2(2)
         P2(3)=A2(3)
         P3(1)=A3(1)
         P3(2)=A3(2)
         P3(3)=A3(3)
         IPTYP=IBRAV
         PDIM(1)=CELDIM(1)
         PDIM(2)=CELDIM(2)
         PDIM(3)=CELDIM(3)
         PDIM(4)=CELDIM(4)
         PDIM(5)=CELDIM(5)
         PDIM(6)=CELDIM(6)
         RETURN
      END IF
! Now we have to perform the next step: finding some cell with smallest
! volume (primitive cell) which can be spanned by three of the ITRANS
! vectors beeing stored in array PTRANS. Therefore sort the vectors:
      CALL LATORD(PTRANS(1,1),ITRANS,NAX+2,INDEX,WRK,0)
! Find first 'yz'-plane:
      XFIRST=PTRANS(1,1)
      DO 8 IX=2,ITRANS
         IF (ABS(XFIRST-PTRANS(IX,1))>TINY) THEN
            IPLANE=IX-1
            JPLANE=IX
            GOTO 9
         END IF
    8 CONTINUE
    9 B1(1)=PTRANS(JPLANE,1)
      B1(2)=PTRANS(JPLANE,2)
      B1(3)=PTRANS(JPLANE,3)
! Find first 'z'-line:
      YFIRST=PTRANS(1,2)
      DO 10 IY=2,IPLANE
         IF (ABS(YFIRST-PTRANS(IY,2))>TINY) THEN
            KPLANE=IY
            GOTO 11
         END IF
   10 CONTINUE
   11 B2(1)=PTRANS(KPLANE,1)
      B2(2)=PTRANS(KPLANE,2)
      B2(3)=PTRANS(KPLANE,3)
      B3(1)=PTRANS(1,1)
      B3(2)=PTRANS(1,2)
      B3(3)=PTRANS(1,3)
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
         WRITE(PRCHAN,'(A)') &
     &                    ' Subroutine PRICEL returns following result:'
         WRITE(PRCHAN,'(A)') ' '
      END IF
      P1(1)=B1(1)*A1(1)+B1(2)*A2(1)+B1(3)*A3(1)
      P1(2)=B1(1)*A1(2)+B1(2)*A2(2)+B1(3)*A3(2)
      P1(3)=B1(1)*A1(3)+B1(2)*A2(3)+B1(3)*A3(3)
      P2(1)=B2(1)*A1(1)+B2(2)*A2(1)+B2(3)*A3(1)
      P2(2)=B2(1)*A1(2)+B2(2)*A2(2)+B2(3)*A3(2)
      P2(3)=B2(1)*A1(3)+B2(2)*A2(3)+B2(3)*A3(3)
      P3(1)=B3(1)*A1(1)+B3(2)*A2(1)+B3(3)*A3(1)
      P3(2)=B3(1)*A1(2)+B3(2)*A2(2)+B3(3)*A3(2)
      P3(3)=B3(1)*A1(3)+B3(2)*A2(3)+B3(3)*A3(3)
      CALL LATTYP(P1,P2,P3,IPTYP,PDIM,PRCHAN)
! Now analyse the data ... :
      CALL CELVOL(A1,A2,A3,OMEGAI)
      OMEGAI=ABS(OMEGAI)
      CALL CELVOL(P1,P2,P3,OMEGAP)
      OMEGAP=ABS(OMEGAP)
      NCELL=NINT(OMEGAI/OMEGAP)
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
         WRITE(PRCHAN,'(A)') ' '
         WRITE(PRCHAN,'(I4,A)') NCELL, &
     &                       ' primitive cells build up your supercell.'
         WRITE(PRCHAN,'(A)') ' '
      END IF
      CALL RECIPS(1._q,A1,A2,A3,B1,B2,B3)
! Rough estimate: How much cells do we have in each direction?
      A1P1=P1(1)*B1(1)+P1(2)*B1(2)+P1(3)*B1(3)
      A2P1=P1(1)*B2(1)+P1(2)*B2(2)+P1(3)*B2(3)
      A3P1=P1(1)*B3(1)+P1(2)*B3(2)+P1(3)*B3(3)
      A1P2=P2(1)*B1(1)+P2(2)*B1(2)+P2(3)*B1(3)
      A2P2=P2(1)*B2(1)+P2(2)*B2(2)+P2(3)*B2(3)
      A3P2=P2(1)*B3(1)+P2(2)*B3(2)+P2(3)*B3(3)
      A1P3=P3(1)*B1(1)+P3(2)*B1(2)+P3(3)*B1(3)
      A2P3=P3(1)*B2(1)+P3(2)*B2(2)+P3(3)*B2(3)
      A3P3=P3(1)*B3(1)+P3(2)*B3(2)+P3(3)*B3(3)
      IF (ABS(A1P1)<TINY) A1P1=1.E30_q
      IF (ABS(A1P2)<TINY) A1P2=1.E30_q
      IF (ABS(A1P3)<TINY) A1P3=1.E30_q
      IF (ABS(A2P1)<TINY) A2P1=1.E30_q
      IF (ABS(A2P2)<TINY) A2P2=1.E30_q
      IF (ABS(A2P3)<TINY) A2P3=1.E30_q
      IF (ABS(A3P1)<TINY) A3P1=1.E30_q
      IF (ABS(A3P2)<TINY) A3P2=1.E30_q
      IF (ABS(A3P3)<TINY) A3P3=1.E30_q
      N1MAX=NINT(MAX(MAX(1._q/ABS(A1P1),1._q/ABS(A2P1)),1._q/ABS(A3P1)))-1
      N2MAX=NINT(MAX(MAX(1._q/ABS(A1P2),1._q/ABS(A2P2)),1._q/ABS(A3P2)))-1
      N3MAX=NINT(MAX(MAX(1._q/ABS(A1P3),1._q/ABS(A2P3)),1._q/ABS(A3P3)))-1
! Which cells lie really within the supercell?
      CALL RECIPS(1._q,A1,A2,A3,B1,B2,B3)
      ICOUNT=0
      DO I3=0,N3MAX
       DO I2=0,N2MAX
        DO I1=0,N1MAX
         X=I1*P1(1)+I2*P2(1)+I3*P3(1)
         Y=I1*P1(2)+I2*P2(2)+I3*P3(2)
         Z=I1*P1(3)+I2*P2(3)+I3*P3(3)
         XC=X*B1(1)+Y*B1(2)+Z*B1(3)
         YC=X*B2(1)+Y*B2(2)+Z*B2(3)
         ZC=X*B3(1)+Y*B3(2)+Z*B3(3)
         XC=MOD(XC+2._q*N1MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         YC=MOD(YC+2._q*N2MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         ZC=MOD(ZC+2._q*N3MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         EXIST=.FALSE.
         DO 12 I=1,ICOUNT
            EXIST=EXIST.OR.((ABS(XC-PTRANS(I,1))<TINY).AND. &
     &                          (ABS(YC-PTRANS(I,2))<TINY).AND. &
     &                                    (ABS(ZC-PTRANS(I,3))<TINY))
   12    CONTINUE
         IF (.NOT.EXIST) THEN
            ICOUNT=ICOUNT+1
            PTRANS(ICOUNT,1)=XC
            PTRANS(ICOUNT,2)=YC
            PTRANS(ICOUNT,3)=ZC
         END IF
        ENDDO
       ENDDO
      ENDDO
      IF (ICOUNT/=NCELL) &
     &   CALL ERROR(' PRICEL (probably precision problem, try to change SYMPREC in INCAR ?)', &
     &   ' Sorry, number of cells and number of vectors did not agree.', &
     &                                                            NCELL)
! Uff ... . Bye bye!
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE PRICELV(IBRAV,CELDIM,A1,A2,A3,TAU,VEC,SPIN,P1,P2,P3,PTRANS,NCELL, &
     &                 IPTYP,PDIM,NSP,NSX,NA,NAX,NC,TAUROT,INDEX,WRK,PRCHAN)
      USE prec
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
!   Subroutine PRICEL detects the generating cell of a supercell.      *
!   First all possible 'nontrivial' translations will be detected      *
!   by testing all difference vectors between atom '1' and all the     *
!   other atoms of the cell whether they can reproduce the lattice     *
!   when translating all atoms by this vector. The strategy will be    *
!   similar to that used in subroutine CHKSYM of program package       *
!   SYMLIB to detect nontrivial translations being associated with     *
!   point symmetry operations.                                         *
!   When all possible translations have been detected we will try      *
!   to detect the type of the generating cell by using subroutine      *
!   LATTYP of this package. Therefore we only have to find one         *
!   (arbitrary) 'primitive cell' (cell of smallest volume) that        *
!   can be spanned by three of all possible translations (including    *
!   the three trivial translations ...).                               *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      IBRAV and CELDIM(6) are the lattice type and the dimensioning   *
!              parameters of the supercell.                            *
!      A1, A2 and A3 are the lattice vectors defining the supercell.   *
!                                                                      *
!      TAU(NAX,3,NSX) contains the crystal coordinates of all atomic   *
!              positions within the supercell. Warning: TAU will be    *
!              modified on output (similar as in routine SETGRP)!      *
!                                                                      *
!      TAUROT,INDEX and WRK are work arrays.                           *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!      PRCHAN defines the unit number for printing output information. *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      P1, P2 and P3 are the lattice vectors of the primitive cell     *
!            generating the given supercell.                           *
!      PTRANS(NAX,3,NSX) contains the crystal coordinates of all       *
!            'primitive' translations within the supercell.            *
!                                                                      *
!      IPTYP returns the bravais lattice type of the generating        *
!            primitive cell according to the key given in subroutines  *
!            LATTYP and LATGEN.                                        *
!      PDIM(6) contains the cell dimensioning parameters of the        *
!            generating primitive cell according to the conventions    *
!            used in subroutines LATGEN and LATTYP.                    *
!                                                                      *
!      NCELL is an integer value which returns the detected number of  *
!            primitive cells within one supercell.                     *
!                                                                      *
!                                                                      *
!***********************************************************************

      LOGICAL TNTRAN,EXIST,TCONT
      LOGICAL TSPIN,TMAYFLIP,TFLIPPED
      INTEGER PRCHAN

      DIMENSION A1(3),A2(3),A3(3),NA(NSP),TAU(NAX,3,NSX),TRA(3),PDIM(6)
      DIMENSION P1(3),P2(3),P3(3),B1(3),B2(3),B3(3),DIF(3),WRK(NAX+2,3)
      DIMENSION TAUROT(NAX,3,NSX),INDEX(NAX+2),TAUSAV(3),PTRANS(NAX+2,3)
      DIMENSION VEC(NAX,3,NSX,NC),VECROT(NAX,3,NSX,NC),DIFV(3,NC)
      DIMENSION SPIN(NAX,NSX),SPINROT(NAX,NSX)
      DIMENSION CELDIM(6)

      EQUIVALENCE (TNTRAN,EXIST)

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
      ISTART=1
      IMINST=1
! For all atomic species ... :
      DO 2 IS=1,NSP
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
            IF ((ABS(TAU(IA,1,ISP)+0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)+0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)+0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
            IF ((ABS(TAU(IA,1,ISP)-0.5_q))<TINY) TAU(IA,1,ISP)=-0.5_q
            IF ((ABS(TAU(IA,2,ISP)-0.5_q))<TINY) TAU(IA,2,ISP)=-0.5_q
            IF ((ABS(TAU(IA,3,ISP)-0.5_q))<TINY) TAU(IA,3,ISP)=-0.5_q
    1    CONTINUE
! Order the atomic coordinates (for this species) ... :
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
    2 CONTINUE
! Trivial case: There exists only one atom of species ISMIN. So we
! have already given the primitive cell ... :
      IF (NA(ISMIN)==1) THEN
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
            WRITE(PRCHAN,'(A)') ' Subroutine PRICEL returns:'
            WRITE(PRCHAN,'(A)') &
     &                    ' Original cell was already a primitive cell.'
            WRITE(PRCHAN,'(A)') ' '
         END IF
         NCELL=1
         PTRANS(1,1)=0._q
         PTRANS(1,2)=0._q
         PTRANS(1,3)=0._q
         P1(1)=A1(1)
         P1(2)=A1(2)
         P1(3)=A1(3)
         P2(1)=A2(1)
         P2(2)=A2(2)
         P2(3)=A2(3)
         P3(1)=A3(1)
         P3(2)=A3(2)
         P3(3)=A3(3)
         IPTYP=IBRAV
         PDIM(1)=CELDIM(1)
         PDIM(2)=CELDIM(2)
         PDIM(3)=CELDIM(3)
         PDIM(4)=CELDIM(4)
         PDIM(5)=CELDIM(5)
         PDIM(6)=CELDIM(6)
         RETURN
      END IF
! Save the rotated coordinate of the first atom of species ISMIN:
      IF (TCONT) THEN
         TAUSAV(1)=TAU(IMINST,1,1)
         TAUSAV(2)=TAU(IMINST,2,1)
         TAUSAV(3)=TAU(IMINST,3,1)
         ISPMIN=1
      ELSE
         TAUSAV(1)=TAU(1,1,ISMIN)
         TAUSAV(2)=TAU(1,2,ISMIN)
         TAUSAV(3)=TAU(1,3,ISMIN)
         ISPMIN=ISMIN
      END IF
! All difference vectors between the first atom and all other atoms
! of species ISMIN could be a translation vector that reproduces the
! lattice ... . So test all possibilities:
      ITRANS=3
      PTRANS(1,1)=1._q
      PTRANS(1,2)=0._q
      PTRANS(1,3)=0._q
      PTRANS(2,1)=0._q
      PTRANS(2,2)=1._q
      PTRANS(2,3)=0._q
      PTRANS(3,1)=0._q
      PTRANS(3,2)=0._q
      PTRANS(3,3)=1._q
      DO 7 IATEST=IMINST+1,NA(ISMIN)+IMINST-1
! Set up the 'test vector' TRA ... :
         TRA(1)=TAU(IATEST,1,ISPMIN)-TAUSAV(1)
         TRA(2)=TAU(IATEST,2,ISPMIN)-TAUSAV(2)
         TRA(3)=TAU(IATEST,3,ISPMIN)-TAUSAV(3)
! TRA could possibly contain trivial translations:
         TRA(1)=MOD((TRA(1)+100._q),1._q)
         TRA(2)=MOD((TRA(2)+100._q),1._q)
         TRA(3)=MOD((TRA(3)+100._q),1._q)
! for reasons of safety:
         IF ((ABS(TRA(1)-1._q))<(TINY*0.5_q)) TRA(1)=0._q
         IF ((ABS(TRA(2)-1._q))<(TINY*0.5_q)) TRA(2)=0._q
         IF ((ABS(TRA(3)-1._q))<(TINY*0.5_q)) TRA(3)=0._q
! Translate the atomic coordinates by TRA ...
         ISTART=1
         DO 4 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 3 IA=ISTART,NA(IS)+ISTART-1
               TAUROT(IA,1,ISP)=TAU(IA,1,ISP)+TRA(1)
               TAUROT(IA,2,ISP)=TAU(IA,2,ISP)+TRA(2)
               TAUROT(IA,3,ISP)=TAU(IA,3,ISP)+TRA(3)
! ... impose the periodic boundary condition on these coordinates ...
               TAUROT(IA,1,ISP)=MOD(TAUROT(IA,1,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,2,ISP)=MOD(TAUROT(IA,2,ISP)+100.5_q,1._q)-0.5_q
               TAUROT(IA,3,ISP)=MOD(TAUROT(IA,3,ISP)+100.5_q,1._q)-0.5_q
! ... ensuring that all surface coordinates are put to -0.5 ...
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
    3       CONTINUE
! ... and order the coordinates:
            CALL LATORD(TAUROT(ISTART,1,ISP),NA(IS),NAX,INDEX,WRK,0)
! and setup VECROT with the same order
            DO IA=1,NA(IS)
               VECROT(ISTART+IA-1,:,ISP,:)=VEC(ISTART+INDEX(IA)-1,:,ISP,:) 
            ENDDO
! and SPINROT as well
            DO IA=1,NA(IS)
               SPINROT(ISTART+IA-1,ISP)=SPIN(ISTART+INDEX(IA)-1,ISP)
            ENDDO
    4    CONTINUE
! Now compare the two lattices 'one-by-one' whether they are identical:
         TNTRAN=.TRUE.
! logicals to keep track of spinflips
         TMAYFLIP=.TRUE.; TFLIPPED=.FALSE.
! For all atoms of the unit cell ... :
         ISTART=1
         DO 6 IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO 5 IA=ISTART,NA(IS)+ISTART-1
! Take the difference of the translated and the original coordinates -
! the result should be (0.,0.,0.)!
               DIF(1)=TAU(IA,1,ISP)-TAUROT(IA,1,ISP)
               DIF(2)=TAU(IA,2,ISP)-TAUROT(IA,2,ISP)
               DIF(3)=TAU(IA,3,ISP)-TAUROT(IA,3,ISP)
! DIF may only differ from (0.,0.,0.) by some trivial translation:
               DIF(1)=MOD((DIF(1)+100._q),1._q)
               DIF(2)=MOD((DIF(2)+100._q),1._q)
               DIF(3)=MOD((DIF(3)+100._q),1._q)
! for reasons of safety:
               IF ((ABS(DIF(1)-1._q))<(TINY*0.5_q)) DIF(1)=0._q
               IF ((ABS(DIF(2)-1._q))<(TINY*0.5_q)) DIF(2)=0._q
               IF ((ABS(DIF(3)-1._q))<(TINY*0.5_q)) DIF(3)=0._q
! DIFV is the difference between the vector quantities at the original
! and translated coordinates, the result should be (0.,0.,0.) as well.
               DO IC=1,NC
                  DIFV(1,IC)=VEC(IA,1,ISP,IC)-VECROT(IA,1,ISP,IC)
                  DIFV(2,IC)=VEC(IA,2,ISP,IC)-VECROT(IA,2,ISP,IC)
                  DIFV(3,IC)=VEC(IA,3,ISP,IC)-VECROT(IA,3,ISP,IC)
               ENDDO
! DIFM is the difference between the spins at the original and the
! translated coordinates.
               DIFM=SPIN(IA,ISP)-SPINROT(IA,ISP)
! DIFP is the sum of the spins at the original and the translated
! coordinates. This quantity is needed to include the possibilty of
! spinflips.
               DIFP=SPIN(IA,ISP)+SPINROT(IA,ISP)
! If DIF and DIFV are not equal (0.,0.,0.) TNTRAN=TNTRAN.AND.FALSE and so it
! remains FALSE forever ... (if only one position is not reproduced!).
! Only if all DIFs are zero TNTRAN (starting with the value TRUE) will
! remain TRUE (that means we found an allowed translation vector).
               TNTRAN=TNTRAN.AND.((ABS(DIF(1))<TINY) &
     &                           .AND.(ABS(DIF(2))<TINY) &
     &                                .AND.(ABS(DIF(3))<TINY))
! and the same for DIFV
               DO IC=1,NC
                  TNTRAN=TNTRAN.AND.((ABS(DIFV(1,IC))<TINY) &
     &                              .AND.(ABS(DIFV(2,IC))<TINY) &
     &                                   .AND.(ABS(DIFV(3,IC))<TINY))
               ENDDO
! treatment of the spin lattice.
               TSPIN=.FALSE.
! when the spins are zero at the original and the translated coordinates:
               IF ((ABS(DIFM)<TINY).AND.(ABS(DIFP)<TINY)) TSPIN=.TRUE.
! when the spins at the original and translated coordinates are nonzero and equal,
! and a spinflip was not already needed to connect two other sites of the lattices:
               IF ((.NOT.TSPIN).AND.(ABS(DIFM)<TINY).AND.(.NOT.TFLIPPED)) THEN
                  TSPIN=.TRUE.; TMAYFLIP=.FALSE.
               ENDIF
! when the spins at the original and translated coordinates are nonzero and a
! spinflip is part of the translation and the spin may still be flipped or was
! already flipped before:
               IF ((.NOT.TSPIN).AND.(ABS(DIFP)<TINY).AND.(TMAYFLIP.OR.TFLIPPED)) THEN
                  TSPIN=.TRUE.; TMAYFLIP=.FALSE.; TFLIPPED=.TRUE.
               ENDIF
! so finally considering spin yields:
               TNTRAN=TNTRAN.AND.TSPIN              
    5       CONTINUE
    6    CONTINUE
! Check was successful ... :
         IF (TNTRAN) THEN
            ITRANS=ITRANS+1
! Save the detected translation vector ... :
            PTRANS(ITRANS,1)=TRA(1)
            PTRANS(ITRANS,2)=TRA(2)
            PTRANS(ITRANS,3)=TRA(3)
         END IF
    7 CONTINUE
! Nothing found ... :
      IF (ITRANS==3) THEN
         IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
            WRITE(PRCHAN,'(A)') ' Subroutine PRICEL returns:'
            WRITE(PRCHAN,'(A)') &
     &                    ' Original cell was already a primitive cell.'
            WRITE(PRCHAN,'(A)') ' '
         END IF
         NCELL=1
         PTRANS(1,1)=0._q
         PTRANS(1,2)=0._q
         PTRANS(1,3)=0._q
         P1(1)=A1(1)
         P1(2)=A1(2)
         P1(3)=A1(3)
         P2(1)=A2(1)
         P2(2)=A2(2)
         P2(3)=A2(3)
         P3(1)=A3(1)
         P3(2)=A3(2)
         P3(3)=A3(3)
         IPTYP=IBRAV
         PDIM(1)=CELDIM(1)
         PDIM(2)=CELDIM(2)
         PDIM(3)=CELDIM(3)
         PDIM(4)=CELDIM(4)
         PDIM(5)=CELDIM(5)
         PDIM(6)=CELDIM(6)
         RETURN
      END IF
! Now we have to perform the next step: finding some cell with smallest
! volume (primitive cell) which can be spanned by three of the ITRANS
! vectors beeing stored in array PTRANS. Therefore sort the vectors:
      CALL LATORD(PTRANS(1,1),ITRANS,NAX+2,INDEX,WRK,0)
! Find first 'yz'-plane:
      XFIRST=PTRANS(1,1)
      DO 8 IX=2,ITRANS
         IF (ABS(XFIRST-PTRANS(IX,1))>TINY) THEN
            IPLANE=IX-1
            JPLANE=IX
            GOTO 9
         END IF
    8 CONTINUE
    9 B1(1)=PTRANS(JPLANE,1)
      B1(2)=PTRANS(JPLANE,2)
      B1(3)=PTRANS(JPLANE,3)
! Find first 'z'-line:
      YFIRST=PTRANS(1,2)
      DO 10 IY=2,IPLANE
         IF (ABS(YFIRST-PTRANS(IY,2))>TINY) THEN
            KPLANE=IY
            GOTO 11
         END IF
   10 CONTINUE
   11 B2(1)=PTRANS(KPLANE,1)
      B2(2)=PTRANS(KPLANE,2)
      B2(3)=PTRANS(KPLANE,3)
      B3(1)=PTRANS(1,1)
      B3(2)=PTRANS(1,2)
      B3(3)=PTRANS(1,3)
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
         WRITE(PRCHAN,'(A)') &
     &                    ' Subroutine PRICEL returns following result:'
         WRITE(PRCHAN,'(A)') ' '
      END IF
      P1(1)=B1(1)*A1(1)+B1(2)*A2(1)+B1(3)*A3(1)
      P1(2)=B1(1)*A1(2)+B1(2)*A2(2)+B1(3)*A3(2)
      P1(3)=B1(1)*A1(3)+B1(2)*A2(3)+B1(3)*A3(3)
      P2(1)=B2(1)*A1(1)+B2(2)*A2(1)+B2(3)*A3(1)
      P2(2)=B2(1)*A1(2)+B2(2)*A2(2)+B2(3)*A3(2)
      P2(3)=B2(1)*A1(3)+B2(2)*A2(3)+B2(3)*A3(3)
      P3(1)=B3(1)*A1(1)+B3(2)*A2(1)+B3(3)*A3(1)
      P3(2)=B3(1)*A1(2)+B3(2)*A2(2)+B3(3)*A3(2)
      P3(3)=B3(1)*A1(3)+B3(2)*A2(3)+B3(3)*A3(3)
      CALL LATTYP(P1,P2,P3,IPTYP,PDIM,PRCHAN)
! Now analyse the data ... :
      CALL CELVOL(A1,A2,A3,OMEGAI)
      OMEGAI=ABS(OMEGAI)
      CALL CELVOL(P1,P2,P3,OMEGAP)
      OMEGAP=ABS(OMEGAP)
      NCELL=NINT(OMEGAI/OMEGAP)
      IF ((PRCHAN>=0).AND.(PRCHAN<=99)) THEN
         WRITE(PRCHAN,'(A)') ' '
         WRITE(PRCHAN,'(I4,A)') NCELL, &
     &                       ' primitive cells build up your supercell.'
         WRITE(PRCHAN,'(A)') ' '
      END IF
      CALL RECIPS(1._q,A1,A2,A3,B1,B2,B3)
! Rough estimate: How much cells do we have in each direction?
      A1P1=P1(1)*B1(1)+P1(2)*B1(2)+P1(3)*B1(3)
      A2P1=P1(1)*B2(1)+P1(2)*B2(2)+P1(3)*B2(3)
      A3P1=P1(1)*B3(1)+P1(2)*B3(2)+P1(3)*B3(3)
      A1P2=P2(1)*B1(1)+P2(2)*B1(2)+P2(3)*B1(3)
      A2P2=P2(1)*B2(1)+P2(2)*B2(2)+P2(3)*B2(3)
      A3P2=P2(1)*B3(1)+P2(2)*B3(2)+P2(3)*B3(3)
      A1P3=P3(1)*B1(1)+P3(2)*B1(2)+P3(3)*B1(3)
      A2P3=P3(1)*B2(1)+P3(2)*B2(2)+P3(3)*B2(3)
      A3P3=P3(1)*B3(1)+P3(2)*B3(2)+P3(3)*B3(3)
      IF (ABS(A1P1)<TINY) A1P1=1.E30_q
      IF (ABS(A1P2)<TINY) A1P2=1.E30_q
      IF (ABS(A1P3)<TINY) A1P3=1.E30_q
      IF (ABS(A2P1)<TINY) A2P1=1.E30_q
      IF (ABS(A2P2)<TINY) A2P2=1.E30_q
      IF (ABS(A2P3)<TINY) A2P3=1.E30_q
      IF (ABS(A3P1)<TINY) A3P1=1.E30_q
      IF (ABS(A3P2)<TINY) A3P2=1.E30_q
      IF (ABS(A3P3)<TINY) A3P3=1.E30_q
      N1MAX=NINT(MAX(MAX(1._q/ABS(A1P1),1._q/ABS(A2P1)),1._q/ABS(A3P1)))-1
      N2MAX=NINT(MAX(MAX(1._q/ABS(A1P2),1._q/ABS(A2P2)),1._q/ABS(A3P2)))-1
      N3MAX=NINT(MAX(MAX(1._q/ABS(A1P3),1._q/ABS(A2P3)),1._q/ABS(A3P3)))-1
! Which cells lie really within the supercell?
      CALL RECIPS(1._q,A1,A2,A3,B1,B2,B3)
      ICOUNT=0
      DO I3=0,N3MAX
       DO I2=0,N2MAX
        DO I1=0,N1MAX
         X=I1*P1(1)+I2*P2(1)+I3*P3(1)
         Y=I1*P1(2)+I2*P2(2)+I3*P3(2)
         Z=I1*P1(3)+I2*P2(3)+I3*P3(3)
         XC=X*B1(1)+Y*B1(2)+Z*B1(3)
         YC=X*B2(1)+Y*B2(2)+Z*B2(3)
         ZC=X*B3(1)+Y*B3(2)+Z*B3(3)
         XC=MOD(XC+2._q*N1MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         YC=MOD(YC+2._q*N2MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         ZC=MOD(ZC+2._q*N3MAX+100._q+0.5_q*TINY,1._q)-0.5_q*TINY
         EXIST=.FALSE.
         DO 12 I=1,ICOUNT
            EXIST=EXIST.OR.((ABS(XC-PTRANS(I,1))<TINY).AND. &
     &                          (ABS(YC-PTRANS(I,2))<TINY).AND. &
     &                                    (ABS(ZC-PTRANS(I,3))<TINY))
   12    CONTINUE
         IF (.NOT.EXIST) THEN
            ICOUNT=ICOUNT+1
            PTRANS(ICOUNT,1)=XC
            PTRANS(ICOUNT,2)=YC
            PTRANS(ICOUNT,3)=ZC
         END IF
        ENDDO
       ENDDO
      ENDDO
      IF (ICOUNT/=NCELL) &
     &   CALL ERROR(' PRICEL (probably precision problem, try to change SYMPREC in INCAR ?)', &
     &   ' Sorry, number of cells and number of vectors did not agree.', &
     &                                                            NCELL)
! Uff ... . Bye bye!
      RETURN
      END
