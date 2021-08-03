#include "symbol.inc"
!***********************************************************************
!                                                                      *
      SUBROUTINE SYDMAT(DMAT,MAP,S,NR,NP,NSX,NSP,NAX,NA,WD,A1,A2,A3,LDO)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
! symmetrize interatomic force constant matrix DMAT                    *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT,LDO
      INTEGER S

      DIMENSION MAP(NAX,NSX,NR,NP),S(3,3,48),NA(NSP),T(3,3),TROT(3,3)
      DIMENSION A1(3),A2(3),A3(3),C1(3),C2(3),C3(3),R(3,3),G(3,3)
      DIMENSION DMAT(3,3,NAX,NSX,NAX,NSX),WD(3,3,NAX,NAX),LDO(NAX,NSX)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/


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
      TCONT=((NSX==1_q).AND.(NSP>1_q))
      IBEG2=1
! For all atomic species ... :
      DO IS2=1,NSP
         ISP2=IS2
         IF (TCONT) ISP2=1
         IF (TCONT.AND.(IS2>1_q)) IBEG2=IBEG2+NA(IS2-1)
         IBEG1=1
         DO IS1=1,NSP
            ISP1=IS1
            IF (TCONT) ISP1=1
            IF (TCONT.AND.(IS1>1_q)) IBEG1=IBEG1+NA(IS1-1)
! Save original DMAT:
            DO IA2=IBEG2,NA(IS2)+IBEG2-1
               DO IA1=IBEG1,NA(IS1)+IBEG1-1
                  DO I2=1,3
                     DO I1=1,3
                        VALUE=DMAT(I1,I2,IA1,ISP1,IA2,ISP2)
                        DMAT(I1,I2,IA1,ISP1,IA2,ISP2)=0._q
                        WD(I1,I2,IA1,IA2)=VALUE/FLOAT(NR*NP)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO
! Loop over all atoms:
            DO IA2=IBEG2,NA(IS2)+IBEG2-1
               DO IA1=IBEG1,NA(IS1)+IBEG1-1
! Sometimes we do not need all elements ...:
                  IF (.NOT.(LDO(IA1,ISP1).OR.LDO(IA2,ISP2))) CYCLE
                  DO IROT=1,NR
! Get rotation matrix transformed to cartesian coordinates:
                     R(1,1)=S(1,1,IROT)
                     R(2,1)=S(2,1,IROT)
                     R(3,1)=S(3,1,IROT)
                     R(1,2)=S(1,2,IROT)
                     R(2,2)=S(2,2,IROT)
                     R(3,2)=S(3,2,IROT)
                     R(1,3)=S(1,3,IROT)
                     R(2,3)=S(2,3,IROT)
                     R(3,3)=S(3,3,IROT)
                     CALL SGRTRF(R,G,1,A1,A2,A3,C1,C2,C3)
! Loop over all translations:
                     DO ITRANS=1,NP
! Indices of rotated atoms:
                        JA1=MAP(IA1,ISP1,IROT,ITRANS)
                        JA2=MAP(IA2,ISP2,IROT,ITRANS)
! Rotate the 3x3matrix associated with atom pair (IA1,IA2):
                        T(1,1)=WD(1,1,JA1,JA2)
                        T(2,1)=WD(2,1,JA1,JA2)
                        T(3,1)=WD(3,1,JA1,JA2)
                        T(1,2)=WD(1,2,JA1,JA2)
                        T(2,2)=WD(2,2,JA1,JA2)
                        T(3,2)=WD(3,2,JA1,JA2)
                        T(1,3)=WD(1,3,JA1,JA2)
                        T(2,3)=WD(2,3,JA1,JA2)
                        T(3,3)=WD(3,3,JA1,JA2)
                        DO I=1,3
                           DO J=1,3
                              TROT(I,J)=G(1,I)*G(1,J)*T(1,1)+ &
     &                                   G(1,I)*G(2,J)*T(1,2)+ &
     &                                    G(1,I)*G(3,J)*T(1,3)+ &
     &                                       G(2,I)*G(1,J)*T(2,1)+ &
     &                                        G(2,I)*G(2,J)*T(2,2)+ &
     &                                         G(2,I)*G(3,J)*T(2,3)+ &
     &                                            G(3,I)*G(1,J)*T(3,1)+ &
     &                                             G(3,I)*G(2,J)*T(3,2)+ &
     &                                              G(3,I)*G(3,J)*T(3,3)
                           ENDDO
                        ENDDO
! sum up contributions for pair (IA1,IA2):
                        DMAT(1,1,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(1,1,IA1,ISP1,IA2,ISP2)+TROT(1,1)
                        DMAT(2,1,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(2,1,IA1,ISP1,IA2,ISP2)+TROT(2,1)
                        DMAT(3,1,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(3,1,IA1,ISP1,IA2,ISP2)+TROT(3,1)
                        DMAT(1,2,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(1,2,IA1,ISP1,IA2,ISP2)+TROT(1,2)
                        DMAT(2,2,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(2,2,IA1,ISP1,IA2,ISP2)+TROT(2,2)
                        DMAT(3,2,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(3,2,IA1,ISP1,IA2,ISP2)+TROT(3,2)
                        DMAT(1,3,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(1,3,IA1,ISP1,IA2,ISP2)+TROT(1,3)
                        DMAT(2,3,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(2,3,IA1,ISP1,IA2,ISP2)+TROT(2,3)
                        DMAT(3,3,IA1,ISP1,IA2,ISP2)= &
     &                             DMAT(3,3,IA1,ISP1,IA2,ISP2)+TROT(3,3)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE STMAT(SMAT,MAP,S,NR,NP,NSX,NSP,NAX,NA,WS,A1,A2,A3,LDO)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!                                                                      *
! symmetrize strain-internal coordinate coupling matrix                *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT,LDO
      INTEGER S

      DIMENSION MAP(NAX,NSX,NR,NP),S(3,3,48),NA(NSP),T(3,3,3),TROT(3,3,3)
      DIMENSION A1(3),A2(3),A3(3),C1(3),C2(3),C3(3),R(3,3),G(3,3)
      DIMENSION SMAT(3,3,3,NAX,NSX),WS(3,3,3,NAX),LDO(NAX,NSX)

      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

      TCONT=((NSX==1_q).AND.(NSP>1_q))
      IBEG2=1
! For all atomic species ... :
      DO IS2=1,NSP
         ISP2=IS2
         IF (TCONT) ISP2=1
         IF (TCONT.AND.(IS2>1_q)) IBEG2=IBEG2+NA(IS2-1)
! Save original SMAT:
         DO IA2=IBEG2,NA(IS2)+IBEG2-1
            DO I2=1,3
               WS(:,:,I2,IA2)=SMAT(:,:,I2,IA2,ISP2)/FLOAT(NR*NP)
               SMAT(:,:,I2,IA2,ISP2)=0._q
            ENDDO
         ENDDO
! Loop over all atoms:
         DO IA2=IBEG2,NA(IS2)+IBEG2-1
! Sometimes we do not need all elements ...:
            IF (.NOT.(LDO(IA2,ISP2))) CYCLE
            DO IROT=1,NR
! Get rotation matrix transformed to cartesian coordinates:
               R(:,:)=S(:,:,IROT)
               CALL SGRTRF(R,G,1,A1,A2,A3,C1,C2,C3)
! Loop over all translations:
               DO ITRANS=1,NP
! Indices of rotated atoms:
                  JA2=MAP(IA2,ISP2,IROT,ITRANS)
! Rotate the 3x3x3matrix associated with atom pair (IA1,IA2):
                  T(:,:,:)=WS(:,:,:,JA2)
                  ! first index
                  DO I=1,3
                     TROT(I,:,:)= & 
                          G(1,I)*T(1,:,:)+ & 
                          G(2,I)*T(2,:,:)+ & 
                          G(3,I)*T(3,:,:)
                  ENDDO
                  ! second index
                  DO I=1,3
                     T(:,I,:)= & 
                          G(1,I)*TROT(:,1,:)+ & 
                          G(2,I)*TROT(:,2,:)+ & 
                          G(3,I)*TROT(:,3,:)
                  ENDDO
                  ! third index
                  DO I=1,3
                     TROT(:,:,I)= & 
                          G(1,I)*T(:,:,1)+ & 
                          G(2,I)*T(:,:,2)+ & 
                          G(3,I)*T(:,:,3)
                  ENDDO
                  ! sum up contributions for pair (IA1,IA2):
                  SMAT(:,:,:,IA2,ISP2)=SMAT(:,:,:,IA2,ISP2)+TROT(:,:,:)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE FREDOM(MAP,S,INVMAP,NR,NP,D,ND,NSX,NSP,NAX,NA, &
     &                      WD,IW,A1,A2,A3,B1,B2,B3,IDIRD,LALL)
      USE sym_prec
      USE msymmetry
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
!   Routine FREDOM tries to find out the number independent degrees of *
!   a given lattice. In principle one has 3*N degrees of freedom for a *
!   system with N atoms (displacement in x,y,z at each atom) but due   *
!   to symmetry the number of degrees of freedom is usually reduced to *
!   a much smaller number. The algorithm is quite simple: at each atom *
!   three test displacements (in x-,y-,z-direction) are performed and  *
!   all symmetry operations are applied to this displacements. This    *
!   generates other displacements at other atoms. All (maximum three!) *
!   linearly independent displacements are stored the rest thrown away *
!   and from each test displacement only those will be kept which had  *
!   not yet been generated previously (or can't be composed out of a   *
!   linear combination of previously generated displacements). There   *
!   is currently only one drawback of this routine: even if one has no *
!   symmetry in the system the number of degrees of freedom would at   *
!   least be reduced by three due to three possible rigid translations *
!   for any system. This is unfortunately not yet taken into account.  *
!   The symmetry reduced number of degrees of freedom might still be   *
!   reduced due to these translational invariances but not necessarily *
!   at all or not necessarily by three degrees of freedom. So in some  *
!   cases unnecessary extra-degrees of freedom might be returned by    *
!   this routine which are no real independant degrees of freedom!     *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      MAP(NAX,NSX,NR,NP) contains a table connecting the rotated or   *
!                translated and the original positions (stored is the  *
!                'atom number' of the rotated or translated position   *
!                for each given rotation and translation).             *
!      S(3,3,48) contains the INTEGER rotation matrices.               *
!      INVMAP(48) contains the index of the reciprocal operation       *
!                 of a given symmetry operation.                       *
!      NR contains the number of given space group operations.         *
!      NP contains the number of 'primitive' translations.             *
!                                                                      *
!      NSP is the number of atomic species.                            *
!      NA(NSP) contains the number of atoms per species.               *
!                                                                      *
!      NSX and NAX are array dimensioning parameters.                  *
!                                                                      *
!      A1(3),A2(3),A3(3) contain the lattice vectors of the supercell. *
!      B1(3),B2(3),B3(3) contain the corresponding reciprocal basis.   *
!                                                                      *
!      WD and IW are work arrays.                                      *
!      LALL  do all displacements in the primitive cell                *
!                                                                      *
!                                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!                                                                      *
!      D(3,3,NAX,NSX) returns all (symmetry-) independent directions   *
!                     found for every atom (maximum three vectors).    *
!                     The rest is filled with zeros ... .              *
!      ND(NAX,NSX) returns the number of independent displacements     *
!                  for each individual atom.                           *
!  optional:                                                           *
!      IDIRD(3,NAX,NSX) returns the cartesian index of the displacement*
!                                                                      *
!***********************************************************************

      LOGICAL TCONT,LNEW
      INTEGER S

      DIMENSION S(3,3,48),NA(NSP),VTEST(3,3),MAP(NAX,NSX,NR,NP)
      DIMENSION D(3,3,NAX,NSX),ND(NAX,NSX),WD(3,3,NAX),IW(NAX)
      DIMENSION INVMAP(48),A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)
      INTEGER :: IDIRD(3,NAX,NSX)
      LOGICAL :: LDONE(NAX)
      LOGICAL :: LALL
      
      SAVE VTEST
      DATA VTEST /1._q,0._q,0._q,0._q,1._q,0._q,0._q,0._q,1._q/

! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished be the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1._q..NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1_q).AND.(NSP>1_q))
      ISTART=1
      LDONE=.FALSE.
! For all atomic species ... :
      DO IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1_q)) ISTART=ISTART+NA(IS-1)
! First preset some arrays with zeros ...
         DO IA=ISTART,NA(IS)+ISTART-1
            ND(IA,ISP)=0
            IW(IA)=0
            D(1,1,IA,ISP)=0._q
            D(2,1,IA,ISP)=0._q
            D(3,1,IA,ISP)=0._q
            D(1,2,IA,ISP)=0._q
            D(2,2,IA,ISP)=0._q
            D(3,2,IA,ISP)=0._q
            D(1,3,IA,ISP)=0._q
            D(2,3,IA,ISP)=0._q
            D(3,3,IA,ISP)=0._q
            IDIRD(1,IA,ISP)=0
            IDIRD(2,IA,ISP)=0
            IDIRD(3,IA,ISP)=0
            WD(1,1,IA)=0._q
            WD(2,1,IA)=0._q
            WD(3,1,IA)=0._q
            WD(1,2,IA)=0._q
            WD(2,2,IA)=0._q
            WD(3,2,IA)=0._q
            WD(1,3,IA)=0._q
            WD(2,3,IA)=0._q
            WD(3,3,IA)=0._q
         ENDDO
! Loop over all atoms and (lattice) directions ... :
         DO IA=ISTART,NA(IS)+ISTART-1
            DO IDIR=1,3
! Get "test direction" on atom in direct coordinates ...
               V1=VTEST(1,IDIR)*B1(1)+ &
     &                           VTEST(2,IDIR)*B1(2)+VTEST(3,IDIR)*B1(3)
               V2=VTEST(1,IDIR)*B2(1)+ &
     &                           VTEST(2,IDIR)*B2(2)+VTEST(3,IDIR)*B2(3)
               V3=VTEST(1,IDIR)*B3(1)+ &
     &                           VTEST(2,IDIR)*B3(2)+VTEST(3,IDIR)*B3(3)
               VR1=VTEST(1,IDIR)
               VR2=VTEST(2,IDIR)
               VR3=VTEST(3,IDIR)
               IF (LALL) THEN
                  LNEW=.FALSE.
                  DO IREP=1, GLOB_PRIM_CELL%P_NUM
                     IF (IA==GLOB_PRIM_CELL%ATOM_IDX(1,IREP)) LNEW=.TRUE.
                  ENDDO
               ELSE
! Test whether direction was already generated previously:
! gK: I believe this should always yield cartesian displacements
                  CALL NEWDIR(VR1,VR2,VR3,WD(1,1,IA),IW(IA),LNEW)
               ENDIF
! store the displacement if it is a new one
               IF (LNEW) THEN
                  ND(IA,ISP)=ND(IA,ISP)+1
                  D(1,ND(IA,ISP),IA,ISP)=VR1
                  D(2,ND(IA,ISP),IA,ISP)=VR2
                  D(3,ND(IA,ISP),IA,ISP)=VR3
                  IDIRD(ND(IA,ISP),IA,ISP)=IDIR
               ENDIF
! Apply all symmetry operations ...
               DO IROT=1,NR
                  IROTI=INVMAP(IROT)
! Rotate the test vector ... :
                  VR1=S(1,1,IROT)*V1+S(2,1,IROT)*V2+S(3,1,IROT)*V3
                  VR2=S(1,2,IROT)*V1+S(2,2,IROT)*V2+S(3,2,IROT)*V3
                  VR3=S(1,3,IROT)*V1+S(2,3,IROT)*V2+S(3,3,IROT)*V3
! ... and transform the result back to cartesian coordinates:
                  X1=VR1*A1(1)+VR2*A2(1)+VR3*A3(1)
                  X2=VR1*A1(2)+VR2*A2(2)+VR3*A3(2)
                  X3=VR1*A1(3)+VR2*A2(3)+VR3*A3(3)
! Apply also all translations ...:
                  DO ITRANS=1,NP
! Get 'index' of rotated / translated atom:
                     JA=MAP(IA,ISP,IROTI,ITRANS)
                     VR1=X1
                     VR2=X2
                     VR3=X3
! found a symmetry equivalent atoms, mark it as "done"
                     IF (IA.NE.JA) LDONE(JA)=.TRUE.
! Test whether direction was already generated previously:
                     CALL NEWDIR(VR1,VR2,VR3,WD(1,1,JA),IW(JA),LNEW)
                     IF (LNEW) THEN
! If not then keep it in mind for later tests ...:
                        IW(JA)=IW(JA)+1
                        WD(1,IW(JA),JA)=VR1
                        WD(2,IW(JA),JA)=VR2
                        WD(3,IW(JA),JA)=VR3
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO
! That's all ... . Bye!
      RETURN
      END

      SUBROUTINE NEWDIR(X,Y,Z,VOLD,NOLD,LNEW)
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL LNEW
      DIMENSION VOLD(3,NOLD)


! Orthogonalize vector (X,Y,Z) on all directions
      DO I=1,NOLD
         P=X*VOLD(1,I)+Y*VOLD(2,I)+Z*VOLD(3,I)
         X=X-P*VOLD(1,I)
         Y=Y-P*VOLD(2,I)
         Z=Z-P*VOLD(3,I)
      ENDDO
! Get norm of the new vector ...
      VNORM=SQRT(X*X+Y*Y+Z*Z)
      LNEW=VNORM>TINY
      IF (LNEW) THEN
! Renorm projected vector to 1
         X=X/VNORM
         Y=Y/VNORM
         Z=Z/VNORM
      ENDIF
      RETURN
      END


!***********************************************************************
!                                                                      *
      SUBROUTINE MKDMAT(MAP,S,INVMAP,NR,NP,D,FD,ST,ND,NSX,NSP,NAX,NA, &
     &                             AD,AFD,AST,IW,FR,FW,FT,A1,A2,A3,B1,B2,B3)
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!   Routine MKDMAT calculates the full interatomic force constants     *
!   from a set of symmetry reduced force constants                     *
!   The input set of force constants is stored in FD for a set of      *
!   displacements supplied in D, the number of displacements is        *
!   give in ND                                                         *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      ND(NAX,NSX) number of considered displacements for each ion     *
!      FD(3,3,NAX,NSX,NAX,NSX) the first and third and fourth index    * 
!           specify the forces on the corresponding ion                *
!           if the ion specified by the fifth and sixth index is       *
!           displaced in the direction specified by the the 2nd index  *
!           the corresponding direction is stored in D(:,IDIR,IA,ISP)  *
!      D(3,3,NAX,NSX) considered displacement vectors for each ion     *
!           the third and fourth index specify the ion and the second  *
!           index corresponds to the displacement index                *
!           D(:,IDIR,IA,ISP) is the considered displacement            *
!      ST(3,3,3,NAX,NSX)  internal strain tensor                       *
!           stress upon moving the ions specified in the fourth and    *
!           and fifth index into the direction specified in the 3rd    *
!           index                                                      *
!   Output parameters:                                                 *
!   ------------------                                                 *
!      AFD(3,NAX,NSX,3,NAX) interatomic force constants                *
!      AST(3,3,3,NAX)  internal strain tensor                          *
!                                                                      *
!***********************************************************************

      LOGICAL TCONT,LNEW
      INTEGER S(3,3,48)

      DIMENSION NA(NSP),INVMAP(48),FD(3,3,NAX,NSX,NAX,NSX),ND(NAX,NSX)
      DIMENSION D(3,3,NAX,NSX),IW(NAX),AD(3,3,NAX),AFD(3,NAX,NSX,3,NAX)
      DIMENSION MAP(NAX,NSX,NR,NP),A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)
      DIMENSION FR(3,NAX,NSX),FW(3,NAX,NSX),FT(3,NAX,NSX)
      DIMENSION ST(3,3,3,NAX,NSX)
      DIMENSION AST(3,3,3,NAX)
      DIMENSION STROT(3,3),STNEW(3,3)
      DIMENSION G(3,3,48),R(3,3,48)
      DIMENSION C1(3),C2(3),C3(3)
      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

! construct the rotation matrices G in cartesian coordinates
! from the integer rotation matrices (S)
      R=S ! note R is a real matrix
      CALL SGRTRF(R,G,NR,A1(1),A2(1),A3(1),C1,C2,C3)
! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished by the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
      ISTART=1
! For all atomic species ... :
      DO IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Loop over all atoms and (lattice) directions ... :
         DO IA=ISTART,NA(IS)+ISTART-1
            IW(IA)=0
         ENDDO
         DO IA=ISTART,NA(IS)+ISTART-1
            DO IDIR=1,ND(IA,ISP)
! Get "test direction" and forces in direct coordinates ...
               V1=D(1,IDIR,IA,ISP)*B1(1)+ &
     &                     D(2,IDIR,IA,ISP)*B1(2)+D(3,IDIR,IA,ISP)*B1(3)
               V2=D(1,IDIR,IA,ISP)*B2(1)+ &
     &                     D(2,IDIR,IA,ISP)*B2(2)+D(3,IDIR,IA,ISP)*B2(3)
               V3=D(1,IDIR,IA,ISP)*B3(1)+ &
     &                     D(2,IDIR,IA,ISP)*B3(2)+D(3,IDIR,IA,ISP)*B3(3)
               ISTARF=1
               ! force in direct coordinates
               DO ISF=1,NSP
                  ISPF=ISF
                  IF (TCONT) ISPF=1
                  IF (TCONT.AND.(ISF>1)) ISTARF=ISTARF+NA(ISF-1)
                  DO IAF=ISTARF,NA(ISF)+ISTARF-1
                     FW(1,IAF,ISPF)=FD(1,IDIR,IAF,ISPF,IA,ISP)*B1(1)+ &
     &                                FD(2,IDIR,IAF,ISPF,IA,ISP)*B1(2)+ &
     &                                  FD(3,IDIR,IAF,ISPF,IA,ISP)*B1(3)
                     FW(2,IAF,ISPF)=FD(1,IDIR,IAF,ISPF,IA,ISP)*B2(1)+ &
     &                                FD(2,IDIR,IAF,ISPF,IA,ISP)*B2(2)+ &
     &                                  FD(3,IDIR,IAF,ISPF,IA,ISP)*B2(3)
                     FW(3,IAF,ISPF)=FD(1,IDIR,IAF,ISPF,IA,ISP)*B3(1)+ &
     &                                FD(2,IDIR,IAF,ISPF,IA,ISP)*B3(2)+ &
     &                                  FD(3,IDIR,IAF,ISPF,IA,ISP)*B3(3)
                  ENDDO
               ENDDO

! Apply all symmetry operations ...
               DO IROT=1,NR
                  IROTI=INVMAP(IROT)
! Rotate test vector ... :
                  VR1=S(1,1,IROT)*V1+S(2,1,IROT)*V2+S(3,1,IROT)*V3
                  VR2=S(1,2,IROT)*V1+S(2,2,IROT)*V2+S(3,2,IROT)*V3
                  VR3=S(1,3,IROT)*V1+S(2,3,IROT)*V2+S(3,3,IROT)*V3
! ... and transform back to cartesian coordinates:
                  X1=VR1*A1(1)+VR2*A2(1)+VR3*A3(1)
                  X2=VR1*A1(2)+VR2*A2(2)+VR3*A3(2)
                  X3=VR1*A1(3)+VR2*A2(3)+VR3*A3(3)
! Then rotate the forces:
                  ISTARF=1
                  DO ISF=1,NSP
                     ISPF=ISF
                     IF (TCONT) ISPF=1
                     IF (TCONT.AND.(ISF>1)) ISTARF=ISTARF+NA(ISF-1)
                     DO IAF=ISTARF,NA(ISF)+ISTARF-1
                        F1=FW(1,IAF,ISPF)
                        F2=FW(2,IAF,ISPF)
                        F3=FW(3,IAF,ISPF)
                        FR1=S(1,1,IROT)*F1+S(2,1,IROT)*F2+S(3,1,IROT)*F3
                        FR2=S(1,2,IROT)*F1+S(2,2,IROT)*F2+S(3,2,IROT)*F3
                        FR3=S(1,3,IROT)*F1+S(2,3,IROT)*F2+S(3,3,IROT)*F3
! ... and transform back to cartesian coordinates:
                        F1=FR1*A1(1)+FR2*A2(1)+FR3*A3(1)
                        F2=FR1*A1(2)+FR2*A2(2)+FR3*A3(2)
                        F3=FR1*A1(3)+FR2*A2(3)+FR3*A3(3)
                        FR(1,IAF,ISPF)=F1
                        FR(2,IAF,ISPF)=F2
                        FR(3,IAF,ISPF)=F3
                     ENDDO
                  ENDDO
                  ! rotate internal strain tensor
                  DO I=1,3
                     DO J=1,3
                        STROT(I,J)=G(1,I,IROT)*G(1,J,IROT)*ST(1,1,IDIR,IA,ISP)+ &
     &                       G(1,I,IROT)*G(2,J,IROT)*ST(1,2,IDIR,IA,ISP)+ &
     &                       G(1,I,IROT)*G(3,J,IROT)*ST(1,3,IDIR,IA,ISP)+ &
     &                       G(2,I,IROT)*G(1,J,IROT)*ST(2,1,IDIR,IA,ISP)+ &
     &                       G(2,I,IROT)*G(2,J,IROT)*ST(2,2,IDIR,IA,ISP)+ &
     &                       G(2,I,IROT)*G(3,J,IROT)*ST(2,3,IDIR,IA,ISP)+ &
     &                       G(3,I,IROT)*G(1,J,IROT)*ST(3,1,IDIR,IA,ISP)+ &
     &                       G(3,I,IROT)*G(2,J,IROT)*ST(3,2,IDIR,IA,ISP)+ &
     &                       G(3,I,IROT)*G(3,J,IROT)*ST(3,3,IDIR,IA,ISP)
                     ENDDO
                  ENDDO
! Apply all translations ...:
                  DO ITRANS=1,NP
! Index of rotated / translated atom:
                     JA=MAP(IA,ISP,IROTI,ITRANS)
                     VR1=X1
                     VR2=X2
                     VR3=X3
                     ISTARF=1
                     DO ISF=1,NSP
                        ISPF=ISF
                        IF (TCONT) ISPF=1
                        IF (TCONT.AND.(ISF>1)) &
     &                                    ISTARF=ISTARF+NA(ISF-1)
                        DO IAF=ISTARF,NA(ISF)+ISTARF-1
! Get the forces at the rotated positions:
                           JAF=MAP(IAF,ISPF,IROTI,ITRANS)
                           FT(1,JAF,ISPF)=FR(1,IAF,ISPF)
                           FT(2,JAF,ISPF)=FR(2,IAF,ISPF)
                           FT(3,JAF,ISPF)=FR(3,IAF,ISPF)
                        ENDDO
                     ENDDO
                     STNEW=STROT
! Test whether direction was already generated previously
! orthogonalize it to all previouly generated directions on this ion
! and accordingly update FT and STNEW
                     CALL CHKDIR(VR1,VR2,VR3,FT,STNEW,AD(1,1,JA), &
     &                       AFD(1,1,1,1,JA),AST(1,1,1,JA),IW(JA),NSX,NSP,NAX,NA,LNEW)
! now "check" in this new direction
                     IF (LNEW) THEN
                        ! increase displacement counter on ion JA
                        IW(JA)=IW(JA)+1
                        IF (IW(JA)>3) THEN
                           WRITE(*,*) 'internal error in MKDMAT: displacement counter exceeds 3'
                           STOP
                        ENDIF
                        ! mark displacement on ion JA
                        AD(1,IW(JA),JA)=VR1
                        AD(2,IW(JA),JA)=VR2
                        AD(3,IW(JA),JA)=VR3
                        ! copy force constants into AFD
                        ISTARF=1
                        DO ISF=1,NSP
                           ISPF=ISF
                           IF (TCONT) ISPF=1
                           IF (TCONT.AND.(ISF>1)) &
     &                                    ISTARF=ISTARF+NA(ISF-1)
                           DO IAF=ISTARF,NA(ISF)+ISTARF-1
                              AFD(1,IAF,ISPF,IW(JA),JA)=FT(1,IAF,ISPF)
                              AFD(2,IAF,ISPF,IW(JA),JA)=FT(2,IAF,ISPF)
                              AFD(3,IAF,ISPF,IW(JA),JA)=FT(3,IAF,ISPF)
                           ENDDO
                        ENDDO
                        ! copy strain into AST
                        AST(:,:,IW(JA),JA)=STNEW
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
! Transfer result back to D, FD, and ST and take cartesian representation ... :
         DO IA=ISTART,NA(IS)+ISTART-1
            IF (IW(IA)/=3) CALL ERROR('MKDMAT', &
     &               'Internal error: Couldn''t get all directions!',IA)
            CALL PRODIR(FD(1,1,1,1,IA,ISP),ST(1,1,1,IA,ISP),NSX,NSP,NAX,NA, &
     &                                       AD(1,1,IA),AFD(1,1,1,1,IA),AST(1,1,1,IA))
         ENDDO
      ENDDO

      RETURN
      END


      SUBROUTINE CHKDIR(X,Y,Z,F,ST,VOLD,FOLD,STOLD,NOLD,NSX,NSP,NAX,NA,LNEW)
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Check whether one found some new independent direction and
! transform also force accordingly ...
      LOGICAL TCONT,LNEW
      DIMENSION VOLD(3,3),F(3,NAX,NSX),FOLD(3,NAX,NSX,3),NA(NSP)
      DIMENSION ST(3,3),STOLD(3,3,3)

      TCONT=((NSX==1).AND.(NSP>1))
! orthogonalize vector (X,Y,Z) onto all directions
! that have already been found for this ion
      DO I=1,NOLD
         ! inproduct with previous direction
         P=X*VOLD(1,I)+Y*VOLD(2,I)+Z*VOLD(3,I)
         ! subtract out from displacementvector previous directions
         X=X-P*VOLD(1,I)
         Y=Y-P*VOLD(2,I)
         Z=Z-P*VOLD(3,I)
         ! same for force constants
         ISTART=1
         DO IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO IA=ISTART,NA(IS)+ISTART-1
               F(1,IA,ISP)=F(1,IA,ISP)-P*FOLD(1,IA,ISP,I)
               F(2,IA,ISP)=F(2,IA,ISP)-P*FOLD(2,IA,ISP,I)
               F(3,IA,ISP)=F(3,IA,ISP)-P*FOLD(3,IA,ISP,I)
            ENDDO
         ENDDO
         ! and strain
         ST=ST-P*STOLD(:,:,I)
      ENDDO
! norm of the new vector ...
      VNORM=SQRT(X*X+Y*Y+Z*Z)
      LNEW=VNORM>TINY
      IF (LNEW) THEN
! renorm projected vector to 1
         X=X/VNORM
         Y=Y/VNORM
         Z=Z/VNORM
! renorm force constants accordingly
         ISTART=1
         DO IS=1,NSP
            ISP=IS
            IF (TCONT) ISP=1
            IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
            DO IA=ISTART,NA(IS)+ISTART-1
               F(1,IA,ISP)=F(1,IA,ISP)/VNORM
               F(2,IA,ISP)=F(2,IA,ISP)/VNORM
               F(3,IA,ISP)=F(3,IA,ISP)/VNORM
            ENDDO
         ENDDO
! renorm stress tensor accordingly
         ST=ST/VNORM
      ENDIF
      RETURN
      END


      SUBROUTINE PRODIR(F,ST,NSX,NSP,NAX,NA,DIRS,FORCES,STRAIN)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Get expansion of cartesian directions in other directions ("basis")
      LOGICAL TCONT
      DIMENSION DIRS(3,3),F(3,3,NAX,NSX),FORCES(3,NAX,NSX,3),NA(NSP)
      DIMENSION ST(3,3,3),STRAIN(3,3,3)
      TCONT=((NSX==1).AND.(NSP>1))
! Here we assume always an orthonormal basis (= cartesian coordinates)
      ISTART=1
      DO IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
         DO IA=ISTART,NA(IS)+ISTART-1
            F(1,1,IA,ISP)=DIRS(1,1)*FORCES(1,IA,ISP,1)+ &
     &                    DIRS(1,2)*FORCES(1,IA,ISP,2)+ &
     &                    DIRS(1,3)*FORCES(1,IA,ISP,3)
            F(2,1,IA,ISP)=DIRS(1,1)*FORCES(2,IA,ISP,1)+ &
     &                    DIRS(1,2)*FORCES(2,IA,ISP,2)+ &
     &                    DIRS(1,3)*FORCES(2,IA,ISP,3)
            F(3,1,IA,ISP)=DIRS(1,1)*FORCES(3,IA,ISP,1)+ &
     &                    DIRS(1,2)*FORCES(3,IA,ISP,2)+ &
     &                    DIRS(1,3)*FORCES(3,IA,ISP,3)
            F(1,2,IA,ISP)=DIRS(2,1)*FORCES(1,IA,ISP,1)+ &
     &                    DIRS(2,2)*FORCES(1,IA,ISP,2)+ &
     &                    DIRS(2,3)*FORCES(1,IA,ISP,3)
            F(2,2,IA,ISP)=DIRS(2,1)*FORCES(2,IA,ISP,1)+ &
     &                    DIRS(2,2)*FORCES(2,IA,ISP,2)+ &
     &                    DIRS(2,3)*FORCES(2,IA,ISP,3)
            F(3,2,IA,ISP)=DIRS(2,1)*FORCES(3,IA,ISP,1)+ &
     &                    DIRS(2,2)*FORCES(3,IA,ISP,2)+ &
     &                    DIRS(2,3)*FORCES(3,IA,ISP,3)
            F(1,3,IA,ISP)=DIRS(3,1)*FORCES(1,IA,ISP,1)+ &
     &                    DIRS(3,2)*FORCES(1,IA,ISP,2)+ &
     &                    DIRS(3,3)*FORCES(1,IA,ISP,3)
            F(2,3,IA,ISP)=DIRS(3,1)*FORCES(2,IA,ISP,1)+ &
     &                    DIRS(3,2)*FORCES(2,IA,ISP,2)+ &
     &                    DIRS(3,3)*FORCES(2,IA,ISP,3)
            F(3,3,IA,ISP)=DIRS(3,1)*FORCES(3,IA,ISP,1)+ &
     &                    DIRS(3,2)*FORCES(3,IA,ISP,2)+ &
     &                    DIRS(3,3)*FORCES(3,IA,ISP,3)
         ENDDO
      ENDDO

      ST(:,:,1)=DIRS(1,1)*STRAIN(:,:,1)+ &
                DIRS(1,2)*STRAIN(:,:,2)+ &
                DIRS(1,3)*STRAIN(:,:,3)
      ST(:,:,2)=DIRS(2,1)*STRAIN(:,:,1)+ &
                DIRS(2,2)*STRAIN(:,:,2)+ &
                DIRS(2,3)*STRAIN(:,:,3)
      ST(:,:,3)=DIRS(3,1)*STRAIN(:,:,1)+ &
                DIRS(3,2)*STRAIN(:,:,2)+ &
                DIRS(3,3)*STRAIN(:,:,3)

      RETURN
      END

!***********************************************************************
!                                                                      *
      SUBROUTINE MKFOR(MAP,S,INVMAP,NR,NP,D,FD,ND, &
         & NSX,NSP,NAX,NA, &
         & A1,A2,A3,B1,B2,B3, LDUMP)
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
!                                                                      *
!***********************************************************************
!   Routine MKFOR  calculates the full interatomic forces              *
!   from a set of symmetry reduced forces obtained by finite           
!   difference or linear response calculations
!   The input set of forces is stored in FD for a set of               *
!   displacements supplied in D, the number of displacements is        *
!   give in ND                                                         *
!                                                                      *
!                                                                      *
!   Input parameters:                                                  *
!   -----------------                                                  *
!                                                                      *
!      ND(NAX,NSX) number of considered displacements for each ion     *
!      FD(3,NAX,NSX)  FD(:,IA,ISP)                                     *
!           specifies the energy derivative for the corresponding ion  *
!           if the ion is displaced in the direction specified by the  *
!           2nd index IDIR                                             *
!           the corresponding direction is stored in D(:,IDIR,IA,ISP)  *
!      D(3,3,NAX,NSX) considered displacement vectors for each ion     *
!           the third and fourth index specify the ion and the second  *
!           index corresponds to the displacement index                *
!           D(:,IDIR,IA,ISP) is the considered displacement            *
!      LDUMP dump intermediate results (for debugging)                 *
!   Output                                                             *
!   -----------------                                                  *
!      FD(3,NAX,NSX) forces in cartesian coordinates on each ion       *
!   Auxilary:                                                          *
!   ------------------                                                 *
!      AFD(3,NAX)                                                      *
!                                                                      *
!***********************************************************************
      LOGICAL LDUMP
      LOGICAL TCONT,LNEW
      INTEGER S(3,3,48)

      DIMENSION NA(NSP),INVMAP(48),FD(3,NAX,NSX),ND(NAX,NSX)
      DIMENSION D(3,3,NAX,NSX),IW(NAX),AD(3,3,NAX),AFD(3,NAX)
      DIMENSION MAP(NAX,NSX,NR,NP),A1(3),A2(3),A3(3),B1(3),B2(3),B3(3)
      REAL(q)   FW,FT
      DIMENSION G(3,3,48),R(3,3,48)
      DIMENSION C1(3),C2(3),C3(3)
      SAVE C1,C2,C3
      DATA C1 /1._q,0._q,0._q/, C2 /0._q,1._q,0._q/, C3 /0._q,0._q,1._q/

! construct the rotation matrices G in cartesian coordinates
! from the integer rotation matrices (S)
      R=S ! note R is a real matrix
      CALL SGRTRF(R,G,NR,A1(1),A2(1),A3(1),C1,C2,C3)
! Unfortunately one has always two possibilities for the book-keeping
! of atomic coordinates, forces etc.: One can distinguish between the
! species and the atoms within each species (two dimensional arrays
! "NAX,NSX") or one can only count the atoms (one dimensional arrays
! "NSAX") where the species are distinguished by the map NA(NSP) in
! the sense (and this should be the convention used - otherwise this
! symmetry package will not work correctly!!) that the atoms of species
! "1" are stored in elements 1...NA(1), atoms of species "2" in elements
! NA(1)+1...NA(1)+NA(2) and so on and atoms of species "NSP" are stored
! in elements NA(1)+NA(2)+...+NA(NSP-1)+1 ... NA(1)+NA(2)+...+NA(NSP).
! The package works for both cases - in the second case one must set
! the dimensioning parameter NSX=1 (only important for NSP>1) ... .
      TCONT=((NSX==1).AND.(NSP>1))
      ISTART=1
! For all atomic species ... :
      DO IS=1,NSP
         ISP=IS
         IF (TCONT) ISP=1
         IF (TCONT.AND.(IS>1)) ISTART=ISTART+NA(IS-1)
! Loop over all atoms and (lattice) directions ... :
         DO IA=ISTART,NA(IS)+ISTART-1
            IW(IA)=0
         ENDDO
         DO IA=ISTART,NA(IS)+ISTART-1
            DO IDIR=1,ND(IA,ISP)
! Get "test direction" and forces in direct coordinates ...
               V1=D(1,IDIR,IA,ISP)*B1(1)+ &
     &                     D(2,IDIR,IA,ISP)*B1(2)+D(3,IDIR,IA,ISP)*B1(3)
               V2=D(1,IDIR,IA,ISP)*B2(1)+ &
     &                     D(2,IDIR,IA,ISP)*B2(2)+D(3,IDIR,IA,ISP)*B2(3)
               V3=D(1,IDIR,IA,ISP)*B3(1)+ &
     &                     D(2,IDIR,IA,ISP)*B3(2)+D(3,IDIR,IA,ISP)*B3(3)
               FW=FD(IDIR,IA,ISP)

! Apply all symmetry operations ...
               DO IROT=1,NR
                  IROTI=INVMAP(IROT)
! Rotate test vector ... :
                  VR1=S(1,1,IROT)*V1+S(2,1,IROT)*V2+S(3,1,IROT)*V3
                  VR2=S(1,2,IROT)*V1+S(2,2,IROT)*V2+S(3,2,IROT)*V3
                  VR3=S(1,3,IROT)*V1+S(2,3,IROT)*V2+S(3,3,IROT)*V3
! ... and transform back to cartesian coordinates:
                  X1=VR1*A1(1)+VR2*A2(1)+VR3*A3(1)
                  X2=VR1*A1(2)+VR2*A2(2)+VR3*A3(2)
                  X3=VR1*A1(3)+VR2*A2(3)+VR3*A3(3)
! Apply all translations ...:
                  DO ITRANS=1,NP
! Index of rotated / translated atom:
                     JA=MAP(IA,ISP,IROTI,ITRANS)
                     VR1=X1
                     VR2=X2
                     VR3=X3
                     FT=FW
! Test whether direction was already generated previously
! orthogonalize it to all previouly generated directions on this ion
! and accordingly update FT
                     IF (LDUMP) WRITE(*,'("tst",2I4,6F8.4)') IA,JA, VR1, VR2, VR3, FT
                     CALL CHKDIR_FOR(VR1,VR2,VR3,FT,AD(1,1,JA), &
     &                       AFD(1,JA),IW(JA),LNEW)
                     IF (LNEW .AND. LDUMP ) THEN
                        WRITE(*,'("new",2I4,6F8.4)') IA, JA, VR1, VR2, VR3, FT
                     ENDIF
! now "check" in this new direction
                     IF (LNEW) THEN
                        ! increase displacement counter on ion JA
                        IW(JA)=IW(JA)+1
                        IF (IW(JA)>3) THEN
                           WRITE(*,*) 'internal error in MKDMAT: displacement counter exceeds 3'
                           STOP
                        ENDIF
                        ! mark displacement on ion JA
                        AD(1,IW(JA),JA)=VR1
                        AD(2,IW(JA),JA)=VR2
                        AD(3,IW(JA),JA)=VR3
                        AFD(IW(JA),JA)=FT
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
! Transfer result back to D, FD, and ST and take cartesian representation ... :
         DO IA=ISTART,NA(IS)+ISTART-1
            IF (IW(IA)/=3) CALL ERROR('MKDMAT', &
     &               'Internal error: Couldn''t get all directions!',IA)
            IF (LDUMP) WRITE(*,'("fin",3F8.4)') AFD(:,IA)
            IF (LDUMP) WRITE(*,'("fin",9F8.4)') AD(:,:,IA)
            CALL PRODIR_FOR(FD(1,IA,ISP),AD(1,1,IA),AFD(1,IA))
            IF (LDUMP) WRITE(*,'("for",9F8.4)') FD(:,IA,ISP)
         ENDDO
      ENDDO

      RETURN
      END

      SUBROUTINE CHKDIR_FOR(X,Y,Z,F,VOLD,FOLD,NOLD,LNEW)
      USE sym_prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Check whether one found some new independent direction and
! transform also force accordingly ...
      LOGICAL TCONT,LNEW
      REAL(q) :: F,VOLD(3,3),FOLD(3)

! orthogonalize vector (X,Y,Z) onto all directions
! that have already been found for this ion
      DO I=1,NOLD
         ! inproduct with previous direction
         P=X*VOLD(1,I)+Y*VOLD(2,I)+Z*VOLD(3,I)
         ! subtract out from displacementvector previous directions
         X=X-P*VOLD(1,I)
         Y=Y-P*VOLD(2,I)
         Z=Z-P*VOLD(3,I)
         ! same for force
         F=F-P*FOLD(I)
      ENDDO
! norm of the new vector ...
      VNORM=SQRT(X*X+Y*Y+Z*Z)
      LNEW=VNORM>TINY
      IF (LNEW) THEN
! renorm projected vector to 1
         X=X/VNORM
         Y=Y/VNORM
         Z=Z/VNORM
! renorm force constants accordingly
         F=F/VNORM
      ENDIF
      RETURN
      END

      SUBROUTINE PRODIR_FOR(F,DIRS,FORCES)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Get expansion of cartesian directions in other directions ("basis")
      LOGICAL TCONT
      DIMENSION DIRS(3,3),F(3),FORCES(3)
      F(1)=DIRS(1,1)*FORCES(1)+ &
     &     DIRS(1,2)*FORCES(2)+ &
     &     DIRS(1,3)*FORCES(3)
      F(2)=DIRS(2,1)*FORCES(1)+ &
     &     DIRS(2,2)*FORCES(2)+ &
     &     DIRS(2,3)*FORCES(3)
      F(3)=DIRS(3,1)*FORCES(1)+ &
     &     DIRS(3,2)*FORCES(2)+ &
     &     DIRS(3,3)*FORCES(3)

      RETURN
      END
