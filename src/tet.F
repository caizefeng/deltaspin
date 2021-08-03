

#include "symbol.inc"
! ======= Set of subroutines to be used for tetrahedron method =========
! RCS:  $Id: tet.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
! partly written by Juergen Furthmueller
! partly taken from an LMTO program of Jepsen and Anderson
!

!****************** SUBROUTINE BZINTS **********************************
!
      SUBROUTINE BZINTS(JOB,FERWE,CELEN,WTKPT,NBAND,NBANDD,NKPTS,IDTET, &
     &         NTET,ISPIN,RSPIN,VOLWGT,EMIN,EMAX,DOS,DOSI,NEDOS,EFERMI,SUMWEI, &
     &               SUME,IU6,PAR,DOSPAR,NKDIM,LDIMP,NIONS,NIOND,JOBPAR)
      USE prec
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      IMPLICIT COMPLEX(q) (C)
!
!***********************************************************************
!
! This routine performs BZ-integrations by the tetrahedron method. It
! has two basic job modes (controlled by flag 'JOB'):
!    JOB=0:         Calculate the DOS and the integrated DOS
!    JOB=-2,-1,1,2: Calculate the Fermi-weights etc., here you have
!                   for submodes: JOB<0 = without Bloechl-correction,
!                   JOB>0 = with Bloechl-correction, for JOB=+/-2 some
!                   additional output is provided (band energy, number
!                   of electrons, Fermi-energy, Bloechl-correction ...)
!
! Input-parameters are CELEN: the band structure data (epsilon_i,ik)
!                      WTKPT: the weights of the k-points
!                      NBAND: the number of bands
!                      NKPTS: the number of irreducible k-points
!                      IDTET: weights/'coordinates' of all tetrahedra
!                      NTET:  number of tetrahedra
!                      EMAX,EMIN: energy window for DOS/DOSI (JOB=0)
!                      NEDOS: number of energy points for DOS/DOSI
!                      EFERMI: approximate/exact Fermi energy (JOB/=0)
!                      IU6: I/O-unit where to write data ...
! Output quantities FERWE:  the Fermi weights for each state (JOB/=0)
!                   DOS:    the density of states (JOB=0)
!                   DOSI:   the integrated density of states (JOB=0)
!                   SUMWEI: the sum of all Fermi weights (JOB/=0)
!                   SUME:   eigenvalue sum ['band energy'] (JOB/=0)
!
!***********************************************************************


      DIMENSION FERWE(NBANDD,NKDIM,ISPIN),CELEN(NBANDD,NKDIM,ISPIN)
      DIMENSION IDTET(0:4,NTET),EC(4),WC(4,2),WTKPT(NKPTS),IQ(4)
      DIMENSION DOS(NEDOS,ISPIN),DOSI(NEDOS,ISPIN)
      DIMENSION PAR(NBANDD,NKDIM,LDIMP,NIOND,ISPIN)
      DIMENSION DOSPAR(NEDOS,LDIMP,NIOND,ISPIN)

! Fatal ERROR!
      IF (NKPTS<4) CALL ERROR(' BZINTS', &
     &        ' Tetrahedron method fails (number of k-points < 4)',NKPT)
      IF ((JOB<(-2)).OR.(JOB>2)) CALL ERROR(' BZINTS', &
     &  ' JOB must be +/-1 or +/-2 (make weights) or 0 (make DOS)!',JOB)
! Initialize arrays for DOS/integrated DOS (if JOB=0) ...
      IF (JOB==0) THEN
         DO ISP=1,ISPIN
         DO I=1,NEDOS
            DOS(I,ISP)=0._q
            DOSI(I,ISP)=0._q
         ENDDO
         ENDDO
      ELSE
! ... Fermi weights ...
         DO ISP=1,ISPIN
          DO IK=1,NKPTS
           DO I=1,NBAND
              FERWE(I,IK,ISP)=0._q
           ENDDO
          ENDDO
         ENDDO
      END IF
! ... and eigenvalue sums:
      SEV1=0._q
      SEV2=0._q
! Start looping over tetrahedra:
      DO ITET=1,NTET
! Get the four corner points:
       IQ(1)=IDTET(1,ITET)
       IQ(2)=IDTET(2,ITET)
       IQ(3)=IDTET(3,ITET)
       IQ(4)=IDTET(4,ITET)
       DO ISP=1,ISPIN
       DO I=1,NBAND
! Get the eigenvalues at each corner:
         EC(1)=CELEN(I,IQ(1),ISP)
         EC(2)=CELEN(I,IQ(2),ISP)
         EC(3)=CELEN(I,IQ(3),ISP)
         EC(4)=CELEN(I,IQ(4),ISP)
         IF (JOB==0) THEN
! Make the DOS:
            CALL SLINZ(VOLWGT*IDTET(0,ITET)*RSPIN,EC,EMIN,EMAX, &
     &            DOS(1,ISP),DOSI(1,ISP),NEDOS,IQ,PAR(1,1,1,1,ISP), &
     &              DOSPAR(1,1,1,ISP),NKDIM,LDIMP,NIONS,NBANDD,I,JOBPAR)
         ELSE
! Make the weights:
            CALL FSWGTS(VOLWGT*IDTET(0,ITET),EC,EFERMI,WC,(JOB>0))
! Band occupations, band energy and number of electrons ... :
            DO 10 IC=1,4
               SEV1=SEV1+WC(IC,1)*EC(IC)*RSPIN
               SEV2=SEV2+WC(IC,2)*EC(IC)*RSPIN
               SUMWEI=(WC(IC,1)+WC(IC,2))/WTKPT(IQ(IC))
               FERWE(I,IQ(IC),ISP)=FERWE(I,IQ(IC),ISP)+SUMWEI
   10       CONTINUE
         END IF
       ENDDO
       ENDDO
      ENDDO
! If JOB=+/-2 make additional checks and give some output (if desired):
      IF (ABS(JOB)==2) THEN
         SUME=SEV1+SEV2
! Check the sum of occupation numbers:
         SUMWEI=0._q
         DO ISP=1,ISPIN
          DO IK=1,NKPTS
           DO I=1,NBAND
              SUMWEI=SUMWEI+FERWE(I,IK,ISP)*WTKPT(IK)
           ENDDO
          ENDDO
         ENDDO
         IF ((IU6>=0).AND.(IU6<=99)) &
     &                       WRITE(IU6,40) EFERMI,RSPIN*SUMWEI,SUME,SEV2
      END IF
   40 FORMAT(1X,'BZINTS: Fermi energy:',F10.6,';',F10.6,' electrons'/ &
     &       9X,'Band energy:',F11.6,';  BLOECHL correction:',F10.6)
      RETURN
      END


!****************** SUBROUTINE SLINZ ***********************************
!
      SUBROUTINE SLINZ(VOLWGT,EC,EMIN,EMAX,DOS,DOSI,NEDOS,IQ, &
     &                PAR,DOSPAR,NKDIM,LDIMP,NIONS,NBANDD,ISTATE,JOBPAR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!
!***********************************************************************
!
! This subroutine adds up the contributions to the density of
! states and the number of states for one single tetrahedron
!
!  Input-parameters are VOLWGT: weight on tetrahedron
!                       EC: energies at corners of tetrahedron
!                       EMIN,EMAX: energy window
!                       NEDOS: number of energy points for DOS/DOSI
!  Output quantities DOS(K): DOS at E(K)=EMIN+(K-1)(EMAX-EMIN)/(NEDOS-1)
!                    DOSI(K): Integrated DOS at E(K)
!
!***********************************************************************



      DIMENSION EC(4),DOS(NEDOS),DOSI(NEDOS),ES(4),EC1(4),IQ(4)
      DIMENSION PAR(NBANDD,NKDIM,LDIMP,NIONS),DOSPAR(NEDOS,LDIMP,NIONS)

      DE=(EMAX-EMIN)/FLOAT(NEDOS-1)
! Sort the energies at the four corners (array EC) into array ES
      DO I=1,4
         EC1(I)=EC(I)
      ENDDO
      DO 3 I=1,4
         I00=1
         DO J=2,4
           IF (EC1(J)<EC1(I00)) I00=J
        ENDDO
         ES(I)=EC1(I00)
         EC1(I00)=1.E30_q
    3 CONTINUE
! Lowest energy still above EMAX ---> no contributions to DOS/DOSI ... :
      IF (ES(1)>=(EMAX+0.00000001_q*DE)) RETURN
! Highest energy still below EMIN ---> no contribution to DOS and
! contribution of complete tetrahedron to DOSI (1*VOLWGT) ... :
      IF (ES(4)<=(EMIN-0.00000001_q*DE)) THEN
         DO 4 I=1,NEDOS
            DOSI(I)=DOSI(I)+VOLWGT
    4    CONTINUE
         RETURN
      END IF
! Now the rest ...
      E1=ES(1)
      E2=ES(2)
      E3=ES(3)
      E4=ES(4)
! Now get the minimum and maximum index for the range we have to update
! DOS(I) and DOSI(I) [so that EMIN>E(ISTART) and EMAX<E(ISTOP)] ... :
      ISTART=MAX((INT((E1-EMIN)/DE-0.00000001_q)),1)
      ISTART=MIN(ISTART,NEDOS)
      ISTOP=MIN((INT((E4-EMIN)/DE+0.00000001_q))+2,NEDOS)
      ISTOP=MAX(ISTOP,1)
! Some constants occuring in the integration formulas ... :
      IF ((E3-E2)>0._q) THEN
         C3=VOLWGT*(E1+E2-E3-E4)/((E3-E1)*(E4-E1)*(E3-E2)*(E4-E2))
         C2=VOLWGT*3._q/((E3-E1)*(E4-E1))
      ELSE
         C3=0._q
         C2=0._q
      ENDIF
      C1=C2*(E2-E1)
      C0=C1*(E2-E1)/3._q
      IF ((E2-E1)>0._q) THEN
         CC12=VOLWGT/((E2-E1)*(E3-E1)*(E4-E1))
      ELSE
         CC12=0._q
      ENDIF
      IF ((E4-E3)>0._q) THEN
         CC34=VOLWGT/((E3-E4)*(E2-E4)*(E1-E4))
      ELSE
         CC34=0._q
      ENDIF
      DO 7 I=ISTART,ISTOP
         EACT=EMIN+(I-1)*DE
         ADDDOS=0._q
! Case EACT between E2,E3:
         IF ((E2<EACT).AND.(EACT<=E3)) THEN
            X=EACT-E2
            ADDDOS=C1+X*(2._q*C2+3._q*X*C3)
            DOSI(I)=DOSI(I)+C0+X*(C1+X*(C2+X*C3))
! Case EACT between E1,E2:
         ELSE IF ((E1<EACT).AND.(EACT<=E2)) THEN
            X=EACT-E1
            ADDDOS=3._q*CC12*X*X
            DOSI(I)=DOSI(I)+CC12*X*X*X
! Case EACT between E3,E4:
         ELSE IF ((E3<EACT).AND.(EACT<=E4)) THEN
            X=EACT-E4
            ADDDOS=-3._q*CC34*X*X
            DOSI(I)=DOSI(I)+VOLWGT-CC34*X*X*X
! Case EACT greater than E4 (might probably happen for I=ISTOP):
         ELSE IF (E4<=EACT) THEN
            DOSI(I)=DOSI(I)+VOLWGT
         END IF
         DOS(I)=DOS(I)+ADDDOS
         IF (JOBPAR/=0) THEN
            DO NI=1,NIONS
             DO LP=1,LDIMP
               PARWGT=0._q
               DO 5 IC=1,4
                  PARWGT=PARWGT+PAR(ISTATE,IQ(IC),LP,NI)
    5          CONTINUE
               DOSPAR(I,LP,NI)=DOSPAR(I,LP,NI)+PARWGT*0.25_q*ADDDOS
             ENDDO
            ENDDO
         ENDIF
    7 CONTINUE
! All energies higer than E(ISTOP) give same contribution to DOSI as
! in the case EACT greater than E4 above ... :
      IF (ISTOP<NEDOS) THEN
         DO 10 I=ISTOP+1,NEDOS
            DOSI(I)=DOSI(I)+VOLWGT
   10    CONTINUE
      END IF
      RETURN
      END


!****************** SUBROUTINE FSWGTS **********************************
!
      SUBROUTINE FSWGTS(VOLWGT,EC,EFERMI,W,BLOECH)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!
!***********************************************************************
!
! This routine makes the weights for integration up to EFERMI for
! one single tetrahedron ('contributions to occupation numbers').
!
!  Input-parameters are VOLWGT: weight on tetrahedron
!                       EC: energies at corners of tetrahedron
!                       EFERMI: Fermi energy
!  Output quantities W(I,1): 'normal' weights
!                    W(I,2): Bloechl-corrections to the weights
!
!***********************************************************************



      LOGICAL BLOECH
      DIMENSION EC(4),EC1(4),ES(4),W(4,2),W1(4),W2(4),ISORT(4)

! Sort the energies at the four corners (array EC) into array ES
      DO I=1,4
         EC1(I)=EC(I)
      ENDDO
      DO 3 I=1,4
         I00=1
         DO J=2,4
            IF (EC1(J)<EC1(I00)) I00=J
         ENDDO
         ES(I)=EC1(I00)
         ISORT(I)=I00
         EC1(I00)=1.E30_q
    3 CONTINUE
! Initialise weights and some other things:
      DF=0._q
      DO I=1,4
         W1(I)=0._q
         W2(I)=0._q
         W(I,1)=0._q
         W(I,2)=0._q
      ENDDO
! Each k-point belongs to four tetrahedra (and will be touched 4 times
! when looping over all tetrahedra -> 'redefine weight factor' -> VW4):
      VW4=VOLWGT/4._q
! Lowest energy still >=EFERMI ---> no contributions to weights ... :
      IF (ES(1)>=EFERMI) RETURN
! Highest energy still <=EFERMI ---> just add up full weight (VW4):
      IF (ES(4)<=EFERMI) THEN
         DO I=1,4
            W(I,1)=VW4
         ENDDO
         RETURN
      END IF
! Now the rest ... :
      E1=ES(1)
      E2=ES(2)
      E3=ES(3)
      E4=ES(4)
! Case EFERMI between E2,E3:
      IF ((E2<EFERMI).AND.(EFERMI<=E3)) THEN
         A31=(EFERMI-E1)/(E3-E1)
         A41=(EFERMI-E1)/(E4-E1)
         A32=(EFERMI-E2)/(E3-E2)
         A42=(EFERMI-E2)/(E4-E2)
         V1=A31*A41
         V2=A31*A42*(1._q-A41)
         V3=A42*A32*(1._q-A31)
         W1(1)=(V1*(3._q-A31-A41)+V2*(2._q-A31-A41)+V3*(1._q-A31))*VW4
         W1(2)=(V1+V2*(2._q-A42)+V3*(3._q-A32-A42))*VW4
         W1(3)=(V1*A31+V2*A31+V3*(A31+A32))*VW4
         W1(4)=(V1*A41+V2*(A41+A42)+V3*A42)*VW4
         DF=((E1+E2-E3-E4)*A32*A42+2*EFERMI-E1-E2)/((E3-E1)*(E4-E1))
         DF=3._q*VOLWGT*DF
! Case EFERMI between E1,E2:
      ELSE IF ((E1<EFERMI).AND.(EFERMI<=E2)) THEN
         A21=(EFERMI-E1)/(E2-E1)
         A31=(EFERMI-E1)/(E3-E1)
         A41=(EFERMI-E1)/(E4-E1)
         XXX=A21*A31*A41*VW4
         W1(1)=XXX*(4._q-A21-A31-A41)
         W1(2)=XXX*A21
         W1(3)=XXX*A31
         W1(4)=XXX*A41
         DF=3._q*VOLWGT*A31*A41/(E2-E1)
! Case EFERMI between E3,E4:
      ELSE IF ((E3<EFERMI).AND.(EFERMI<=E4)) THEN
         A14=(EFERMI-E4)/(E1-E4)
         A24=(EFERMI-E4)/(E2-E4)
         A34=(EFERMI-E4)/(E3-E4)
         XXX=A14*A24*A34*VW4
         W1(1)=VW4-XXX*A14
         W1(2)=VW4-XXX*A24
         W1(3)=VW4-XXX*A34
         W1(4)=VW4-XXX*(4._q-A14-A24-A34)
         DF=3._q*VOLWGT*A14*A24/(E4-E3)
      END IF
! Here the BLOECHL corrections to the weights (if desired) ... :
      IF (BLOECH) THEN
         DO 15 I=1,4
            W2(I)=0._q
            DO J=1,4
               W2(I)=W2(I)+(ES(J)-ES(I))*DF*0.025_q
            ENDDO
   15    CONTINUE
      END IF
! Now store the weights into W with the correct ordering ... :
      DO 20 I=1,4
         J=ISORT(I)
         W(J,1)=W1(I)
         W(J,2)=W2(I)
   20 CONTINUE
      RETURN
      END
