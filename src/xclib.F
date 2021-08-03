#include "symbol.inc"

MODULE xclib
!*******************************************************************
!
! the module xclib implements a number of common
! LDA exchange correlation functionals
! it has been moved into this file to allow inlining without 
! relying on inter file inlining (which is usually done only
! at rather high optimisation levels)
!
!*******************************************************************

  CONTAINS

    FUNCTION ECCA(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Ceperley-Alder correlation energy as parametrised by Perdew/Zunger
! (see Phys.Rev. B23,5048 [1981], Appendix).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      REAL(q),SAVE ::  A(2)=(/0.0622_q,0.0311_q/)
      REAL(q),SAVE ::  B(2)=(/-0.0960_q,-0.0538_q/)
!     REAL(q),SAVE ::  C(2)=(/0.0040_q,0.0014_q/)
      REAL(q),SAVE ::  C(2)=(/0.004038664055501747_q,0.001395274602717559_q/)
!     REAL(q),SAVE ::  D(2)=(/-0.0232_q,-0.0096_q/)
      REAL(q),SAVE ::  D(2)=(/-0.023264632546756681_q,-0.009602765503781227_q/)
      REAL(q),SAVE ::  G(2)=(/-0.2846_q,-0.1686_q/)
      REAL(q),SAVE ::  B1(2)=(/1.0529_q,1.3981_q/)
!     REAL(q),SAVE ::  B2(2)=(/0.3334_q,0.2611_q/)
      REAL(q),SAVE ::  B2(2)=(/0.333390000000000000_q, 0.261090000000000000_q/)
      IF (RS<=1.0_q) THEN
         RSL=LOG(RS)
         ECCA=A(IFLG)*RSL+B(IFLG)+C(IFLG)*RS*RSL+D(IFLG)*RS
      ELSE
         RSQ=SQRT(RS)
         ECCA=G(IFLG)/(1.0_q+B1(IFLG)*RSQ+B2(IFLG)*RS)
      END IF
      RETURN
    END FUNCTION ECCA

    FUNCTION VCCA(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Ceperley-Alder correlation potential as parametrised by Perdew/Zunger
! (see Phys.Rev. B23,5048 [1981], Appendix).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION A(2),B(2),C(2),D(2),BT1(2),BT2(2)
      PARAMETER(X76=7.0_q/6.0_q, X43=4.0_q/3.0_q, &
     &  AP=0.03110_q*2.0_q, BP=-0.0480_q*2.0_q, CP=0.0020_q*2.0_q, DP=-0.0116_q*2.0_q, &
     &  AF=0.01555_q*2.0_q, BF=-0.0269_q*2.0_q, CF=0.0007_q*2.0_q, DF=-0.0048_q*2.0_q, &
     &  BP1=BP-AP/3.0_q, CP1=2.0_q*CP/3.0_q, DP1=(2.0_q*DP-CP)/3.0_q, &
     &  BF1=BF-AF/3.0_q, CF1=2.0_q*CF/3.0_q, DF1=(2.0_q*DF-CF)/3.0_q)
      SAVE A,B,C,D,BT1,BT2
      DATA A/AP,AF/,B/BP1,BF1/,C/CP1,CF1/,D/DP1,DF1/, &
     &     BT1/1.0529_q,1.3981_q/,BT2/0.3334_q,0.2611_q/
!KRESSE/FURTH---get a continous energy functional
      c(1)  = 0.004038664055501747_q * 2._q/3._q
      d(1)  =-0.023264632546756681_q * 2._q/3._q  -  0.004038664055501747_q / 3._q
      bt2(1)= 0.333390000000000000_q
      c(2)  = 0.001395274602717559_q * 2._q/3._q
      d(2)  =-0.009602765503781227_q * 2._q/3._q  -  0.001395274602717559_q / 3._q
      bt2(2)= 0.261090000000000000_q
!KRESSE/FURTH
      IF (RS<=1.0_q) THEN
         RSL=LOG(RS)
         VCCA=A(IFLG)*RSL+B(IFLG)+C(IFLG)*RS*RSL+D(IFLG)*RS
      ELSE
         RSQ=SQRT(RS)
         VCCA=ECCA(RS,IFLG)*(1.0_q+X76*BT1(IFLG)*RSQ+X43*BT2(IFLG)*RS) &
     &                     /(1.0_q+    BT1(IFLG)*RSQ+    BT2(IFLG)*RS)
      END IF
      RETURN
    END FUNCTION VCCA


    FUNCTION ECVO(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The Ceperley-Alder correlation energy as given by the Pade approximation
! technique of Vosko et al. (Can.J.Phys. 58,1200 [1980], eq.{4.4} with
! the parameters given in table 5, page 1207).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION A(2),X02(2),B2(2),C2(2),Q2(2),XX02(2)
      PARAMETER(AP=0.0621814_q,X0P=-0.104980_q,BP=3.72744_q,CP=12.9352_q)
      PARAMETER(AF=0.0310907_q,X0F=-0.325000_q,BF=7.06042_q,CF=18.0578_q)
      PARAMETER(QP=6.1519908_q,QF=4.7309269_q)
      PARAMETER(XX0P=X0P*X0P+BP*X0P+CP,XX0F=X0F*X0F+BF*X0F+CF)
      SAVE A,X02,B2,C2,Q2,XX02
      DATA A/AP,AF/,X02/X0P,X0F/,B2/BP,BF/,C2/CP,CF/,Q2/QP,QF/, &
     &     XX02/XX0P,XX0F/
      X=SQRT(RS)
      XX=RS+B2(IFLG)*X+C2(IFLG)
      X0=X02(IFLG)
      B=B2(IFLG)
      C=C2(IFLG)
      QQ=Q2(IFLG)
      XX0=XX02(IFLG)
      ECVO=LOG((X-X0)*(X-X0)/XX)+2._q*(B+2._q*X0)/QQ*ATAN(QQ/(2._q*X+B))
      ECVO=-1._q*ECVO*B*X0/XX0+LOG(RS/XX)+2._q*B/QQ*ATAN(QQ/(2._q*X+B))
      ECVO=ECVO*A(IFLG)
      RETURN
    END FUNCTION ECVO

    FUNCTION VCVO(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The function ECVO(RS,IFLG)-RS/3.*d(ECVO(RS,IFLG))/d(RS) with function
! ECVO(RS,IFLG) given above (Ceperley-Alder potential derived from the
! approximation for ECVO of Vosko et al. discussed above).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION A(2),X02(2),B2(2),C2(2),Q2(2),XX02(2)
      PARAMETER(AP=0.0621814_q,X0P=-0.104980_q,BP=3.72744_q,CP=12.9352_q)
      PARAMETER(AF=0.0310907_q,X0F=-0.325000_q,BF=7.06042_q,CF=18.0578_q)
      PARAMETER(QP=6.1519908_q,QF=4.7309269_q)
      PARAMETER(XX0P=X0P*X0P+BP*X0P+CP,XX0F=X0F*X0F+BF*X0F+CF)
      SAVE A,X02,B2,C2,Q2,XX02
      DATA A/AP,AF/,X02/X0P,X0F/,B2/BP,BF/,C2/CP,CF/,Q2/QP,QF/, &
     &     XX02/XX0P,XX0F/
      X=SQRT(RS)
      XX=RS+B2(IFLG)*X+C2(IFLG)
      X0=X02(IFLG)
      B=B2(IFLG)
      C=C2(IFLG)
      QQ=Q2(IFLG)
      XX0=XX02(IFLG)
      VCVO=-4._q*B*(1._q-X0*(B+2._q*X0)/XX0)/(QQ*QQ+(2._q*X+B)*(2._q*X+B))
      VCVO=VCVO-(2._q*X+B)/XX*(1._q-B*X0/XX0)-2._q*B*X0/XX0/(X-X0)+2._q/X
      VCVO=ECVO(RS,IFLG)-VCVO*A(IFLG)*X/6._q
      RETURN
    END FUNCTION VCVO
!jP---------------------------------------------------------------------------------
    FUNCTION ECVOIII(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The RPA correlation energy as given by the Pade approximation
! technique of Vosko et al. (Can.J.Phys. 58,1200 [1980], eq.{4.4} with
! the parameters given in table 5, page 1207).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION A(2),X02(2),B2(2),C2(2),Q2(2),XX02(2)
      PARAMETER(AP=0.0621814_q,X0P=-0.409286_q,BP=13.0720_q,CP=42.7198_q)
      PARAMETER(AF=0.0310907_q,X0F=-0.743294_q,BF=20.1231_q,CF=101.578_q)
      PARAMETER(QP=0.0448999_q,QF=1.1716853_q)

      PARAMETER(XX0P=X0P*X0P+BP*X0P+CP,XX0F=X0F*X0F+BF*X0F+CF)
      SAVE A,X02,B2,C2,Q2,XX02
      DATA A/AP,AF/,X02/X0P,X0F/,B2/BP,BF/,C2/CP,CF/,Q2/QP,QF/, &
     &     XX02/XX0P,XX0F/
      X=SQRT(RS)
      XX=RS+B2(IFLG)*X+C2(IFLG)
      X0=X02(IFLG)
      B=B2(IFLG)
      C=C2(IFLG)
      QQ=Q2(IFLG)
      XX0=XX02(IFLG)
      ECVOIII=LOG((X-X0)*(X-X0)/XX)+2._q*(B+2._q*X0)/QQ*ATAN(QQ/(2._q*X+B))
      ECVOIII=-1._q*ECVOIII*B*X0/XX0+LOG(RS/XX)+2._q*B/QQ*ATAN(QQ/(2._q*X+B))
      ECVOIII=ECVOIII*A(IFLG)
      RETURN
    END FUNCTION ECVOIII

    FUNCTION VCVOIII(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The function ECVOIII(RS,IFLG)-RS/3.*d(ECVOIII(RS,IFLG))/d(RS) with function
! ECVOIII(RS,IFLG) given above (RPA potential derived from the
! approximation for ECVOIII of Vosko et al. discussed above).
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION A(2),X02(2),B2(2),C2(2),Q2(2),XX02(2)
      PARAMETER(AP=0.0621814_q,X0P=-0.409286_q,BP=13.0720_q,CP=42.7198_q)
      PARAMETER(AF=0.0310907_q,X0F=-0.743294_q,BF=20.1231_q,CF=101.578_q)
      PARAMETER(QP=0.0448999_q,QF=1.1716853_q)

      PARAMETER(XX0P=X0P*X0P+BP*X0P+CP,XX0F=X0F*X0F+BF*X0F+CF)
      SAVE A,X02,B2,C2,Q2,XX02
      DATA A/AP,AF/,X02/X0P,X0F/,B2/BP,BF/,C2/CP,CF/,Q2/QP,QF/, &
     &     XX02/XX0P,XX0F/
      X=SQRT(RS)
      XX=RS+B2(IFLG)*X+C2(IFLG)
      X0=X02(IFLG)
      B=B2(IFLG)
      C=C2(IFLG)
      QQ=Q2(IFLG)
      XX0=XX02(IFLG)
      VCVOIII=-4._q*B*(1._q-X0*(B+2._q*X0)/XX0)/(QQ*QQ+(2._q*X+B)*(2._q*X+B))
      VCVOIII=VCVOIII-(2._q*X+B)/XX*(1._q-B*X0/XX0)-2._q*B*X0/XX0/(X-X0)+2._q/X
      VCVOIII=ECVOIII(RS,IFLG)-VCVOIII*A(IFLG)*X/6._q
      RETURN
    END FUNCTION VCVOIII
!jP---------------------------------------------------------------------------------

    FUNCTION ECGL(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Gunnarson-Lundqvist correlation energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      PARAMETER(THIRD=1.0_q/3.0_q)
      SAVE C,R
      DATA C/0.0666_q,0.0406_q/,R/11.4_q,15.9_q/
      X=RS/R(IFLG)
      ECGL=-C(IFLG)*((1._q+X**3)*LOG(1._q+1._q/X)-THIRD+X*(0.5_q-X))
      RETURN
    END FUNCTION ECGL

    FUNCTION VCGL(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Gunnarson-Lundqvist correlation potential:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      SAVE C,R
      DATA C/0.0666_q,0.0406_q/,R/11.4_q,15.9_q/
      VCGL=-C(IFLG)*LOG(1._q+R(IFLG)/RS)
      RETURN
    END FUNCTION VCGL

    FUNCTION ECHL(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Hedin-Lundqvist correlation energy (J.Phys. C4,2064 [1971]):
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      PARAMETER(THIRD=1.0_q/3.0_q)
      SAVE C,R
      DATA C/0.045_q,0.0225_q/,R/21.0_q,52.917_q/
      X=RS/R(IFLG)
      ECHL=-C(IFLG)*((1._q+X**3)*LOG(1._q+1._q/X)-THIRD+X*(0.5_q-X))
      RETURN
    END FUNCTION ECHL

    FUNCTION VCHL(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Hedin-Lundqvist correlation potential (J.Phys. C4,2064 [1971]):
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      SAVE C,R
      DATA C/0.045_q,0.0225_q/,R/21.0_q,52.917_q/
      VCHL=-C(IFLG)*LOG(1._q+R(IFLG)/RS)
      RETURN
    END FUNCTION VCHL

    FUNCTION ECBH(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Barth-Hedin correlation energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      PARAMETER(THIRD=1.0_q/3.0_q)
      SAVE C,R
      DATA C/0.0504_q,0.0254_q/,R/30._q,75._q/
      X=RS/R(IFLG)
      ECBH=-C(IFLG)*((1._q+X**3)*LOG(1._q+1._q/X)-THIRD+X*(0.5_q-X))
      RETURN
    END FUNCTION ECBH

    FUNCTION VCBH(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Barth-Hedin correlation potential:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION C(2),R(2)
      SAVE C,R
      DATA C/0.0504_q,0.0254_q/,R/30._q,75._q/
      VCBH=-C(IFLG)*LOG(1._q+R(IFLG)/RS)
      RETURN
    END FUNCTION VCBH

      FUNCTION ECWI(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Wigner correlation energy (hopefully correct?):
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results --> warning: equals paramagnetic result
      DIMENSION CX(2),C(2),R(2)
      SAVE CX,C,R
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      DATA C/7.8_q,7.8_q/,R/0.88_q,0.88_q/
      X=CX(IFLG)/R(IFLG)*(1._q+C(IFLG)/RS)
      ECWI=-CX(IFLG)/X/RS
      RETURN
    END FUNCTION ECWI

    FUNCTION VCWI(RS,IFLG)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Wigner correlation potential (hopefully correct?):
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results --> warning: equals paramagnetic result
      DIMENSION CX(2),C(2),R(2)
      SAVE CX,C,R
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      DATA C/7.8_q,7.8_q/,R/0.88_q,0.88_q/
      X1=C(IFLG)/RS
      X2=1._q+X1
      X3=CX(IFLG)/R(IFLG)*X2
      B=1._q+1._q/X3
      F=1._q-X1/X2/(1._q+X3)
      F=1._q+F/3._q
      E=-CX(IFLG)*B/RS
      VCWI=(4._q/3._q-F*B)*CX(IFLG)/RS
      RETURN
    END FUNCTION VCWI

    FUNCTION EX(RS,IFLG,TREL)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Exchange energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
!     TREL  :  Relativistic correction or not (for details see
!              J.Phys. C12,2977(1979) )
      LOGICAL TREL
      DIMENSION CX(2)
      SAVE CX,CBETA
      DATA CX/0.9163305865663_q,1.1545041946774_q/,CBETA/0.0140_q/
      EX=-CX(IFLG)/RS
      IF (TREL) THEN
         B=CBETA/RS
         F=LOG(B+(SQRT(1+B*B)))/(B*B)
         F=(SQRT(1+B*B)/B)-F
!jF: the expression given above becomes numerically extremely instable for
!    very small values of B (small difference of two large numbers divided
!    by small number = noise) therefore use following for reasons of safety:
         IF (B.LT.1.E-5_q) F=B*(2._q/3._q-0.7_q*B*B)
         EX=(1._q-1.5_q*F*F)*EX
      END IF
      RETURN
    END FUNCTION EX

    FUNCTION VX(RS,IFLG,TREL)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Exchange potential:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
!     TREL  :  Relativistic correction or not (for details see
!              J.Phys. C12,2977(1979) )
      LOGICAL TREL
      DIMENSION CX(2)
      SAVE CX,CBETA
      DATA CX/1.2217741154217_q,1.5393389262365_q/,CBETA/0.0140_q/
      VX=-CX(IFLG)/RS
      IF (TREL) THEN
! Warning error in the paper of Bachelet et al. !!
         B=CBETA/RS
         F=LOG(B+(SQRT(1+B*B)))/B/SQRT(1+(B*B))
!        F=LOG(B+(SQRT(1+B*B)))/B/(1+(B*B))
!jF: potentially the expression given above becomes numerically instable for
!    very small values of B, therefore use following for reasons of safety:
         IF (B.LT.1.E-5_q) F=1._q-B*B*(2._q/3._q-B*B*31._q/30._q)
         VX=(-0.5_q+1.5_q*F)*VX
      END IF
      RETURN
    END FUNCTION VX
      
    FUNCTION FZ0(ZETA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! Interpolation function between paramagnetic and ferromagnetic results
! for exchange energy and exchange potential. The parameter ZETA is:
! ZETA=(RHO[upspin] - RHO[downspin]) / (RHO[upspin] + RHO[downspin]).
      PARAMETER(C43=4._q/3._q,FAC=1.92366105093153632_q)
      Z=ABS(ZETA)
      IF (Z==0._q) THEN
         FZ0=0._q
      ELSE IF (Z>=1._q) THEN
         FZ0=1._q
      ELSE
         FZ0=(((1._q+Z)**C43)+((1._q-Z)**C43)-2._q)*FAC
      END IF
      RETURN
    END FUNCTION FZ0

    FUNCTION FZ1(ZETA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The derivative dFZ0(ZETA)/d(ZETA), FZ0(ZETA) given above.
      PARAMETER(C13=1._q/3._q,FAC=2.56488140124204843_q)
      Z=ABS(ZETA)
      IF (Z==0._q) THEN
         FZ1=0._q
      ELSE IF (Z>=1._q) THEN
         FZ1=(2._q**C13)*FAC
      ELSE
         FZ1=(((1._q+Z)**C13)-((1._q-Z))**C13)*FAC
      END IF
      RETURN
    END FUNCTION FZ1

    FUNCTION ALPHA0(RS)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The spin stiffness [d(EXC(RS,ZETA))**2/d**2(ZETA) |ZETA=0] as given by
! Vosko et al. (Can.J.Phys. 58,1200 [1980], eq.{4.4} with the parameters
! given on page 1209 [fitting to low-density values using eq.{4.7.}]).
! Warning: the values are multiplied by 1./(d(FZ0(ZETA))**2/d**2(ZETA))
!          at ZETA=0, FZ0(ZETA) given above.
      PARAMETER(X0=-0.0047584_q,B=1.13107_q,C=13.0045_q,QQ=7.12311_q)
      PARAMETER(XX0=X0*X0+B*X0+C,A=-0.019751631321681_q)
      X=SQRT(RS)
      XX=RS+B*X+C
      ALPHA0=LOG((X-X0)*(X-X0)/XX)+2._q*(B+2._q*X0)/QQ*ATAN(QQ/(2._q*X+B))
      ALPHA0=-1._q*ALPHA0*B*X0/XX0+LOG(RS/XX)+2._q*B/QQ*ATAN(QQ/(2._q*X+B))
      ALPHA0=ALPHA0*A
      RETURN
    END FUNCTION ALPHA0

    FUNCTION ALPHA1(RS)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The function ALPHA0(RS)-RS/3.*dALPHA0(RS)/dRS, ALPHA0(RS) given above.
! Warning: the values are multiplied by 1./(d(FZ0(ZETA))**2/d**2(ZETA))
!          at ZETA=0, FZ0(ZETA) given above.
      PARAMETER(X0=-.0047584_q,B=1.13107_q,C=13.0045_q,QQ=7.12311_q)
      PARAMETER(XX0=X0*X0+B*X0+C,A=-0.0032919385536135_q)
      X=SQRT(RS)
      XX=RS+B*X+C
      ALPHA1=-4._q*B/QQ*(1._q-X0*(B+2._q*X0)/XX0)/(QQ*QQ+(2._q*X+B)*(2._q*X+B))
      ALPHA1=ALPHA1-(2._q*X+B)/XX*(1._q-B*X0/XX0)-2._q*B*X0/XX0/(X-X0)+2._q/X
      ALPHA1=ALPHA0(RS)-ALPHA1*A*X
      RETURN
    END FUNCTION ALPHA1
!----------------------------------------------------------------------
    FUNCTION ALPHA0_III(RS)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The spin stiffness [d(EXC(RS,ZETA))**2/d**2(ZETA) |ZETA=0] as given by
! Vosko et al. (Can.J.Phys. 58,1200 [1980], eq.{4.4} with the parameters
! given on page 1209 [fitting to low-density values using eq.{4.7.}]).
! Warning: the values are multiplied by 1./(d(FZ0(ZETA))**2/d**2(ZETA))
!          at ZETA=0, FZ0(ZETA) given above.
      PARAMETER(X0=-0.228344_q,B=1.06835_q,C=11.4813_q,QQ=6.69207_q)
      PARAMETER(XX0=X0*X0+B*X0+C,A=-0.019751631321681_q)
      X=SQRT(RS)
      XX=RS+B*X+C
      ALPHA0_III=LOG((X-X0)*(X-X0)/XX)+2._q*(B+2._q*X0)/QQ*ATAN(QQ/(2._q*X+B))
      ALPHA0_III=-1._q*ALPHA0_III*B*X0/XX0+LOG(RS/XX)+2._q*B/QQ*ATAN(QQ/(2._q*X+B))
      ALPHA0_III=ALPHA0_III*A
      RETURN
    END FUNCTION ALPHA0_III

    FUNCTION ALPHA1_III(RS)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! The function ALPHA0(RS)-RS/3.*dALPHA0(RS)/dRS, ALPHA0(RS) given above.
! Warning: the values are multiplied by 1./(d(FZ0(ZETA))**2/d**2(ZETA))
!          at ZETA=0, FZ0(ZETA) given above.
      PARAMETER(X0=-0.228344_q,B=1.06835_q,C=11.4813_q,QQ=6.69207_q)
      PARAMETER(XX0=X0*X0+B*X0+C,A=-0.0032919385536135_q)
      X=SQRT(RS)
      XX=RS+B*X+C
      ALPHA1_III=-4._q*B/QQ*(1._q-X0*(B+2._q*X0)/XX0)/(QQ*QQ+(2._q*X+B)*(2._q*X+B))
      ALPHA1_III=ALPHA1_III-(2._q*X+B)/XX*(1._q-B*X0/XX0)-2._q*B*X0/XX0/(X-X0)+2._q/X
      ALPHA1_III=ALPHA0_III(RS)-ALPHA1_III*A*X
      RETURN
    END FUNCTION ALPHA1_III
!----------------------------------------------------------------------

!----------------------------------------------------------------------
    SUBROUTINE CORPBE_LDA(RS,ZET,EC,VCUP,VCDN)
!----------------------------------------------------------------------
!  LDA part of the official PBE correlation code. K. Burke, May 14, 1996.
!  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3)
!       : ZET=RELATIVE SPIN POLARIZATION = (rhoup-rhodn)/rho
!  output: ec=lsd correlation energy from [a]
!        : vcup=lsd up correlation potential
!        : vcdn=lsd dn correlation potential
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! References:
! [a] J.P.~Perdew, K.~Burke, and M.~Ernzerhof, 
!     {\sl Generalized gradient approximation made simple}, sub.
!     to Phys. Rev.Lett. May 1996.
! [b] J. P. Perdew, K. Burke, and Y. Wang, {\sl Real-space cutoff
!     construction of a generalized gradient approximation:  The PW91
!     density functional}, submitted to Phys. Rev. B, Feb. 1996.
! [c] J. P. Perdew and Y. Wang, Phys. Rev. B {\bf 45}, 13244 (1992).
!----------------------------------------------------------------------
!----------------------------------------------------------------------
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      CALL gcor_xc(0.01554535_q,0.20548_q,14.1189_q,6.1977_q,3.3662_q, &
     &    0.62517_q,rtRS,EP,EPRS)
      CALL gcor_xc(0.0168869_q,0.11125_q,10.357_q,3.6231_q,0.88026_q, &
     &    0.49671_q,rtRS,ALFM,ALFRSM)
      ALFC = -ALFM
      Z4 = ZET**4
      F=((1._q+ZET)**THRD4+(1._q-ZET)**THRD4-2._q)/GAM
      EC = EU*(1._q-F*Z4)+EP*F*Z4-ALFM*F*(1._q-Z4)/FZZ
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! LSD potential from [c](A1)
! ECRS = dEc/drs [c](A2)
! ECZET=dEc/dzeta [c](A3)
! FZ = dF/dzeta [c](A4)
      ECRS = EURS*(1._q-F*Z4)+EPRS*F*Z4-ALFRSM*F*(1._q-Z4)/FZZ
      FZ = THRD4*((1._q+ZET)**THRD-(1._q-ZET)**THRD)/GAM
      ECZET = 4._q*(ZET**3)*F*(EP-EU+ALFM/FZZ)+FZ*(Z4*EP-Z4*EU &
     &        -(1._q-Z4)*ALFM/FZZ)
      COMM = EC -RS*ECRS/3._q-ZET*ECZET
      VCUP = COMM + ECZET
      VCDN = COMM - ECZET

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA
    
    
!----------------------------------------------------------------------
! Judith Harl
! functionals for range-separated ACFDT (LDA - short range RPA):
! a bit akward at the moment since the range separation parameter
! is hard coded for now. Hopefully this will change ...
!----------------------------------------------------------------------

    SUBROUTINE CORPBE_LDA_RPA(RS,ZET,EC,VCUP,VCDN)
! RPA correlation energy for jellium 
! Perdew, Wang, prB 45 13244 (1992)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EU,EURS)
      CALL gcor_xc_rpa(0.015545_q,0.035374_q,6.4869_q,1.3083_q,0.15180_q, &
     &    0.082349_q,rtRS,EP,EPRS)
      CALL gcor_xc(0.016887_q,0.028829_q,10.357_q,3.6231_q,0.47990_q, &
     &    0.12279_q,rtRS,ALFM,ALFRSM)
      ALFC = -ALFM
      Z4 = ZET**4
      F=((1._q+ZET)**THRD4+(1._q-ZET)**THRD4-2._q)/GAM
      EC = EU*(1._q-F*Z4)+EP*F*Z4-ALFM*F*(1._q-Z4)/FZZ
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! LSD potential from [c](A1)
! ECRS = dEc/drs [c](A2)
! ECZET=dEc/dzeta [c](A3)
! FZ = dF/dzeta [c](A4)
      ECRS = EURS*(1._q-F*Z4)+EPRS*F*Z4-ALFRSM*F*(1._q-Z4)/FZZ
      FZ = THRD4*((1._q+ZET)**THRD-(1._q-ZET)**THRD)/GAM
      ECZET = 4._q*(ZET**3)*F*(EP-EU+ALFM/FZZ)+FZ*(Z4*EP-Z4*EU &
     &        -(1._q-Z4)*ALFM/FZZ)
      COMM = EC -RS*ECRS/3._q-ZET*ECZET
      VCUP = COMM + ECZET
      VCDN = COMM - ECZET

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_RPA


!  jH-new This routine is used to evaluate the RPA+ correction term, 
!  jH-new which is E^{LDA,QMC}_{c}-E^{LDA,RPA}_{c}

    SUBROUTINE CORPBE_LDA_RPA_PLUS(RS,ZET,EC,VCUP,VCDN)
! RPA correlation energy for jellium 
! Perdew, Wang, prB 45 13244 (1992)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EURPA,EURSRPA)
      CALL gcor_xc_rpa(0.015545_q,0.035374_q,6.4869_q,1.3083_q,0.15180_q, &
     &    0.082349_q,rtRS,EPRPA,EPRSRPA)
      CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EUQM,EURSQM)
      CALL gcor_xc(0.01554535_q,0.20548_q,14.1189_q,6.1977_q,3.3662_q, &
     &    0.62517_q,rtRS,EPQM,EPRSQM)
      EU=EUQM-EURPA
      EP=EPQM-EPRPA
      EURS=EURSQM-EURSRPA
      EPRS=EPRSQM-EPRSRPA
! jH rpa spin stiffness
      CALL gcor_xc(0.016887_q,0.028829_q,10.357_q,3.6231_q,0.47990_q, &
     &    0.12279_q,rtRS,ALFMRPA,ALFRSMRPA)
! jH QMC spin stiffness
      CALL gcor_xc(0.0168869_q,0.11125_q,10.357_q,3.6231_q,0.88026_q, &
     &    0.49671_q,rtRS,ALFMQM,ALFRSMQM)     
      ALFCRPA = -ALFMRPA
      ALFCQM  = -ALFMQM
      Z4 = ZET**4
      F=((1._q+ZET)**THRD4+(1._q-ZET)**THRD4-2._q)/GAM
      ECRPA = EURPA*(1._q-F*Z4)+EPRPA*F*Z4-ALFMRPA*F*(1._q-Z4)/FZZ
      ECQM  =  EUQM*(1._q-F*Z4)+EPQM*F*Z4-ALFMQM*F*(1._q-Z4)/FZZ
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! LSD potential from [c](A1)
! ECRS = dEc/drs [c](A2)
! ECZET=dEc/dzeta [c](A3)
! FZ = dF/dzeta [c](A4)
! jH Aufpassen: Potential 
      ECRSRPA = EURSRPA*(1._q-F*Z4)+EPRSRPA*F*Z4-ALFRSMRPA*F*(1._q-Z4)/FZZ
      ECRSQM  =  EURSQM*(1._q-F*Z4)+EPRSQM*F*Z4-ALFRSMQM*F*(1._q-Z4)/FZZ
      FZ = THRD4*((1._q+ZET)**THRD-(1._q-ZET)**THRD)/GAM
      ECZETRPA = 4._q*(ZET**3)*F*(EPRPA-EURPA+ALFMRPA/FZZ)+FZ*(Z4*EPRPA-Z4*EURPA &
     &        -(1._q-Z4)*ALFMRPA/FZZ)
      ECZETQM = 4._q*(ZET**3)*F*(EPQM-EUQM+ALFMQM/FZZ)+FZ*(Z4*EPQM-Z4*EUQM &
     &        -(1._q-Z4)*ALFMQM/FZZ) 
      COMMRPA = ECRPA -RS*ECRSRPA/3._q-ZET*ECZETRPA
      COMMQM  = ECQM  -RS*ECRSQM/3._q-ZET*ECZETQM
      VCUP = COMMQM + ECZETQM - (COMMRPA + ECZETRPA)
      VCDN = COMMQM - ECZETQM - (COMMRPA - ECZETRPA)

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
!      EC=EC*2
       EC=(ECQM-ECRPA)*2
       VCUP=VCUP*2
       VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_RPA_PLUS

    SUBROUTINE CORPBE_LDA_SR_0_15au(RS,ZET,EC,VCUP,VCDN)
!This routines gives a correlation energy functional defined by
!    Ec,SR[n0]:=   Ec,MC[n0]    -   Ec,RPA,LR[n0] 
!and   Ec,RPA,LR[n0] = Ec,RPA[n0]-Ec,RPA,sr[n0]
!where Ec,MC is the Perdew-Wang parametrization of the
!Monte Carlo correlation energies [1], Ec,RPA the P-W 
!parametrization of the RPA energy [1], and Ec,RPA,sr the 
!v = (1/r)Erfc(-mu r) short-range ACFDT-RPA energies for
!jellium calculated by Judith 
! [1] Perdew,Wang 1992 prB 45, 13244
! 
!   Just for the paramagnetic case
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EU,EURS)
      ERPATOT = EU
      ERPATOTRS = EURS
       CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      EMC     = EU
      EMCRS   = EURS 
       CALL gcor_xc_2(0.031091_q,-0.0089687_q,5.1486_q,1.3952503_q, & 
     & 0.2284487_q,0.0965737_q,1.6078169_q,rtrs,EU,EURS)
      ERPASR = EU
      ERPASRRS = EURS
! SR correlation energy so definiert das HEG limit
      EU = EMC - (ERPATOT - ERPASR)
      EURS = EMCRS -  (ERPATOTRS - ERPASRRS)

! jH thats enough

      EC=EU
      ECRS=EURS
      COMM=EC-S*ECRS/3._q 
      VCUP = COMM    
      VCDN = COMM

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_SR_0_15au

    SUBROUTINE CORPBE_LDA_SR_0_5A(RS,ZET,EC,VCUP,VCDN)
!This routines gives a correlation energy functional defined by
!    Ec,SR[n0]:=   Ec,MC[n0]    -   Ec,RPA,LR[n0] 
!and   Ec,RPA,LR[n0] = Ec,RPA[n0]-Ec,RPA,sr[n0]
!where Ec,MC is the Perdew-Wang parametrization of the
!Monte Carlo correlation energies [1], Ec,RPA the P-W 
!parametrization of the RPA energy [1], and Ec,RPA,sr the 
!v = (1/r)Erfc(-mu r) short-range ACFDT-RPA energies for
!jellium calculated by Judith 
! [1] Perdew,Wang 1992 prB 45, 13244
! 
!   Just for the paramagnetic case
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EU,EURS)
      ERPATOT = EU
      ERPATOTRS = EURS
       CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      EMC     = EU
      EMCRS     = EURS
       CALL gcor_xc_2(0.031091_q,-0.0106268_q,5.1486_q,1.4094821_q, & 
     & 1.2291950_q,0.2579335_q,1.8066972_q,rtrs,EU,EURS)
      ERPASR = EU
      ERPASRRS = EURS
      EU = EMC - (ERPATOT - ERPASR)
      EURS = EMCRS -  (ERPATOTRS - ERPASRRS)

! jH thats enough

      EC=EU
      ECRS=EURS
      COMM=EC-S*ECRS/3._q
      VCUP = COMM
      VCDN = COMM

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_SR_0_5A

    SUBROUTINE CORPBE_LDA_SR_1_0A(RS,ZET,EC,VCUP,VCDN)
!This routines gives a correlation energy functional defined by
!    Ec,SR[n0]:=   Ec,MC[n0]    -   Ec,RPA,LR[n0] 
!and   Ec,RPA,LR[n0] = Ec,RPA[n0]-Ec,RPA,sr[n0]
!where Ec,MC is the Perdew-Wang parametrization of the
!Monte Carlo correlation energies [1], Ec,RPA the P-W 
!parametrization of the RPA energy [1], and Ec,RPA,sr the 
!v = (1/r)Erfc(-mu r) short-range ACFDT-RPA energies for
!jellium calculated by Judith 
! [1] Perdew,Wang 1992 prB 45, 13244
! 
!   Just for the paramagnetic case
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EU,EURS)
      ERPATOT = EU
      ERPATOTRS = EURS
       CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      EMC     = EU
      EMCRS     = EURS
       CALL gcor_xc_2(0.031091_q,-0.0119450_q,5.1486_q,1.5999830_q, & 
     & 4.7029103_q,1.3622380_q,1.9748249_q,rtrs,EU,EURS)
      ERPASR = EU
      ERPASRRS = EURS
      EU = EMC -(ERPATOT - ERPASR)
      EURS = EMCRS -(ERPATOTRS - ERPASRRS)

! jH thats enough

      EC=EU
      ECRS=EURS
      COMM=EC-S*ECRS/3._q
      VCUP = COMM
      VCDN = COMM

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_SR_1_0A

    SUBROUTINE CORPBE_LDA_SR_2_0A(RS,ZET,EC,VCUP,VCDN)
!This routines gives a correlation energy functional defined by
!    Ec,SR[n0]:=   Ec,MC[n0]    -   Ec,RPA,LR[n0] 
!and   Ec,RPA,LR[n0] = Ec,RPA[n0]-Ec,RPA,sr[n0]
!where Ec,MC is the Perdew-Wang parametrization of the
!Monte Carlo correlation energies [1], Ec,RPA the P-W 
!parametrization of the RPA energy [1], and Ec,RPA,sr the 
!v = (1/r)Erfc(-mu r) short-range ACFDT-RPA energies for
!jellium calculated by Judith 
! [1] Perdew,Wang 1992 prB 45, 13244
! 
!   Just for the paramagnetic case
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.
      logical lgga
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet=0.06672455060314922_q,delt=bet/gamma)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor_xc_rpa(0.031091_q,0.082477_q,5.1486_q,1.6483_q,0.23647_q, &
     &    0.20614_q,rtrs,EU,EURS)
      ERPATOT = EU
      ERPATOTRS = EURS
       CALL gcor_xc(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      EMC     = EU
      EMCRS     = EURS
       CALL gcor_xc_2(0.031091_q,0.0127716_q,5.1486_q,2.5109626_q, & 
     & 15.3666137_q,10.2843432_q,2.1207238_q,rtrs,EU,EURS)
      ERPASR = EU
      ERPASRRS = EURS
      EU = EMC - (ERPATOT - ERPASR)
      EURS = EMCRS - (ERPATOTRS - ERPASRRS) 

! jH thats enough

      EC=EU
      ECRS=EURS
      COMM=EC-S*ECRS/3._q
      VCUP = COMM
      VCDN = COMM

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units
      EC=EC*2
      VCUP=VCUP*2
      VCDN=VCDN*2
      RETURN
    END SUBROUTINE CORPBE_LDA_SR_2_0A


!----------------------------------------------------------------------
    FUNCTION PBE_ALPHA(RS)
!----------------------------------------------------------------------
!  Alpha (spin stiffness)
!  from the official PBE correlation code. K. Burke, May 14, 1996.
!  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3)
!  spin stiffness 
!----------------------------------------------------------------------
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))

      rtrs=dsqrt(rs)
      CALL gcor_xc(0.0168869_q,0.11125_q,10.357_q,3.6231_q,0.88026_q, &
     &    0.49671_q,rtRS,ALFM,ALFRSM)

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units

      PBE_ALPHA = -ALFM/FZZ*2
      RETURN
    END FUNCTION PBE_ALPHA

! jH RPA_ALPHA new routine for RPA spin stiffness 
!----------------------------------------------------------------------
    FUNCTION RPA_ALPHA(RS)
!----------------------------------------------------------------------
!  Alpha (spin stiffness)
!  from the official PBE correlation code. K. Burke, May 14, 1996.
!  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3)
!  spin stiffness
!  jH this parametrisation has been already impl in CORPBE_LDA_RPA
!  jH parametrisation from Perdew, Wang, prB 45, 13244 (1992)
!----------------------------------------------------------------------
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))

      rtrs=dsqrt(rs)
      CALL gcor_xc(0.016887_q,0.028829_q,10.357_q,3.6231_q,0.47990_q, &
     &    0.12279_q,rtRS,ALFM,ALFRSM)

      ! the convention of the subroutines in this package
      ! is to return Rydberg energy units

      RPA_ALPHA = -ALFM/FZZ*2
      RETURN
    END FUNCTION RPA_ALPHA


!----------------------------------------------------------------------
!######################################################################
!----------------------------------------------------------------------
    SUBROUTINE GCOR_XC(A,A1,B1,B2,B3,B4,rtrs,GG,GGRS)
! slimmed down version of GCOR used in PW91 routines, to interpolate
! LSD correlation energy, as given by (10) of
! J. P. Perdew and Y. Wang, Phys. Rev. B {\bf 45}, 13244 (1992).
! K. Burke, May 11, 1996.
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      Q0 = -2._q*A*(1._q+A1*rtrs*rtrs)
      Q1 = 2._q*A*rtrs*(B1+rtrs*(B2+rtrs*(B3+B4*rtrs)))
      Q2 = DLOG(1._q+1._q/Q1)
      GG = Q0*Q2
      Q3 = A*(B1/rtrs+2._q*B2+rtrs*(3._q*B3+4._q*B4*rtrs))
      GGRS = -2._q*A*A1*Q2-Q0*Q3/(Q1*(1._q+Q1))
      RETURN
    END SUBROUTINE GCOR_XC

!----------------------------------------------------------------------
!#########################JUDITH###################################
!----------------------------------------------------------------------

    SUBROUTINE GCOR_XC_RPA(A,A1,B1,B2,B3,B4,rtrs,GG,GGRS)
! This is the rs interpolation formula of Perdew/Wang Phys.Rev.B 45 (1992) 13244
! for the RPA case (p = 0.75). See FORMULA (10) in article 
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      Q0 = -2._q*A*(1._q+A1*rtrs*rtrs)
      Q1 = 2._q*A*rtrs*(B1+rtrs*(B2+rtrs*(B3+B4*sqrt(rtrs))))
      Q2 = DLOG(1._q+1._q/Q1)
      GG = Q0*Q2
      Q3 = A*(B1/rtrs+2._q*B2+rtrs*(3._q*B3+7._q/2._q*B4*sqrt(rtrs)))
      GGRS = -2._q*A*A1*Q2-Q0*Q3/(Q1*(1._q+Q1))
      RETURN
      END SUBROUTINE GCOR_XC_RPA

!----------------------------------------------------------------------
!###########################JUDITH################################
!----------------------------------------------------------------------

    SUBROUTINE GCOR_XC_2(A,A1,B1,B2,B3,B4,p,rtrs,GG,GGRS)
! This is the rs interpolation formula of Perdew/Wang Phys.Rev.B 45 (1992) 13244
! for p = 2. See FORMULA (10) in article. This interpolation is used to
! parametrsize the short-range for v = (1/r)*Erfc(-mu r) RPA jellium energie --> Judith
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      Q0 = -2._q*A*(1._q+A1*rtrs*rtrs)
      Q1 = 2._q*A*rtrs*(B1+rtrs*(B2+rtrs*(B3+B4*rtrs**(2._q*p-1._q))))
      Q2 = DLOG(1._q+1._q/Q1)
      GG = Q0*Q2
      Q3 = A*(B1/rtrs+2._q*B2+rtrs*(3._q*B3+(2._q*p+2._q)*B4*rtrs**(2._q*p-1._q)))
      GGRS = -2._q*A*A1*Q2-Q0*Q3/(Q1*(1._q+Q1))
      RETURN
      END SUBROUTINE GCOR_XC_2

!----------------------------------------------------------------------
!######################################################################
!----------------------------------------------------------------------
! Iann Gerber 18/11/04

!=====================================================================
! short range screened exchange functional
! EX_SR computes exchange-part with modified kernel
! interpolation formula of A. Savin and J. Toulouse 
!=====================================================================

    
    FUNCTION EX_SR(RS,RMU,IFLG)
      USE prec
      USE constant
      USE c2f_interface, ONLY : ERRF
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) :: RS        ! Wigner Seitz radius in a.u.
      REAL(q) :: RMU       ! inverse real space cutoff in a.u.
      INTEGER :: IFLG      ! paramagnetic (IFLG=1) or ferromagnetic (IFLG=2)
! Exchange energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION CX(2)
      SAVE CX
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      EX_=-CX(IFLG)/RS

! another IG bug: IFLAG replaced by IFLG
      IF (IFLG==2) THEN
         QFAC=(6._q*PI*PI)**(1._q/3._q)
      ELSE
         QFAC=(3._q*PI*PI)**(1._q/3._q)
      ENDIF

      RHO = 3/(4._q*PI)/RS**3
      QF=QFAC*(RHO)**(1._q/3._q)
      A=RMU/2._q/QF
!      FRAC=(1 - (8._q/3._q)*A*( SQRT(PI)*ERRF(1/(2*A)) + (2*A-4*A*A*A)*EXP(-1/(4*A*A)) &
!     &        -3*A + 4*A*A*A ))
!      EX_SR=EX_*FRAC
! IG Simple formula replaced by 

! Test on the value of A
      IF (A < 1E-9_q) THEN
! Limit for small A
         EX_SR=EX_
      ELSEIF (A <= 100._q) THEN
! Intermediate Values of A
         FRAC=(1.0_q - (8._q/3._q)*A*( SQRT(PI)*ERRF(1._q/(2._q*A)) + &
     &        (2._q*A-4._q*A*A*A)*EXP(-1._q/(4._q*A*A))-3._q*A + 4._q*A*A*A ))
         EX_SR=EX_*FRAC
      ELSEIF (A <= 1.E+9_q) THEN
! Development for large A
         FRAC=1._q/36._q/A/A
         EX_SR=EX_*FRAC
      ELSE
         EX_SR=0._q
      ENDIF
!      B=EXP(-1._q/(4._q*A*A))-1._q
!      C=2._q*A*A*B+1._q/2._q
!      FRAC=1.0_q - 8._q/3._q*A*(SQRT(PI)*ERRF(1._q/(2._q*A))+2._q*A*(B-C))
!      EX_SR=EX_*FRAC
      RETURN
    END FUNCTION EX_SR
      

    FUNCTION VX_SR(RS,RMU,IFLG)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) :: RS        ! Wigner Seitz radius in a.u.
      REAL(q) :: RMU       ! inverse real space cutoff in a.u.
      INTEGER :: IFLG      ! paramagnetic (IFLG=1) or ferromagnetic (IFLG=2)
! Exchange energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION CX(2)
      SAVE CX
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      VX_=-CX(IFLG)/RS

! another IG bug: IFLAG replaced by IFLG
      IF (IFLG==2) THEN
         QFAC=(6._q*PI*PI)**(1._q/3._q)
      ELSE
         QFAC=(3._q*PI*PI)**(1._q/3._q)
      ENDIF

      RHO = 3/(4._q*PI)/RS**3
      QF=QFAC*(RHO)**(1._q/3._q)
      A=RMU/2._q/QF

!      FRAC= 1._q - 32*A*A*A*A*EXP(-1/(4*A*A)) -8._q*A*A + (64._q/3._q)*A*A*A*A
!      VX_SR=EX_SR(RS,RMU,IFLG)+(1._q/3._q)*VX_*FRAC
! IG Simple formula replaced by 

! Test on the value of A
      IF (A < 1E-9_q) THEN
! Limit for small A
         VX_SR=(4._q/3._q)*VX_
      ELSEIF (A <= 100._q) THEN
! Intermediate Values of A
         FRAC= (1._q - 32._q*A*A*A*A*(EXP(-1._q/(4._q*A*A))-1._q) -8._q*A*A)
         VX_SR=EX_SR(RS,RMU,IFLG)+(1._q/3._q)*VX_*FRAC
      ELSEIF (A <= 1.E+9_q) THEN
! Development for large A
         VX_SR=-RMU/48._q/PI/A/A/A
      ELSE
         VX_SR=0._q
      ENDIF
     RETURN
   END FUNCTION VX_SR

   FUNCTION EC_SR(RS,RMU,EC) ! No spin polarization
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) :: RS        ! Wigner Seitz radius in a.u.
      REAL(q) :: RMU       ! inverse real space cutoff in a.u.

      SAVE U1,U2,V1,A,BET,GAM
      DATA U1/1.0270741452992294_q/
      DATA U2/-0.230160617208092_q/
      DATA V1/0.6196884832404359_q/
      DATA A/3.2581_q/
      DATA BET/163.44_q/
      DATA GAM/4.7125_q/

      D=32._q/3._q/PI
! changed to a.u. instead of A, iG please check
      RHO = 3/(4._q*PI)/RS**3

!     G(0) from Burke, Perdew & Ernzerhof
      GRS32=(GAM+RS)**3._q/2._q
      GRS12=SQRT(GAM+RS)
      G0=D*(GRS32+BET)*EXP(-A*GRS12)

      C1N=U1*RS + U2*RS*RS
      C1D=1._q+V1*RS
      C1=C1N/C1D
      C2D=0.5_q*PI*RHO*RHO*(G0-0.5_q)
      C2=EC/C2D

      SCALE=1._q + C1*RMU + C2*RMU*RMU

      EC_SR=EC/SCALE
      RETURN
    END FUNCTION EC_SR

    FUNCTION VC_SR(RS,RMU,VC,EC) ! No spin polarization
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) :: RS        ! Wigner Seitz radius in a.u.
      REAL(q) :: RMU       ! inverse real space cutoff in a.u.

      SAVE U1,U2,V1,A,BET,GAM
      DATA U1/1.0270741452992294_q/
      DATA U2/-0.230160617208092_q/
      DATA V1/0.6196884832404359_q/
      DATA A/3.2581_q/
      DATA BET/163.44_q/
      DATA GAM/4.7125_q/

      D=32._q/3._q/PI
! changed to a.u. instead of A, iG please check
      RHO = 3/(4._q*PI)/RS**3

!     G(0) from Burke, Perdew & Ernzerhof
      GRS32=(GAM+RS)**3._q/2._q
      GRS12=SQRT(GAM+RS)
      G0=D*(GRS32+BET)*EXP(-A*GRS12)

      C1N=U1*RS + U2*RS*RS
      C1D=1._q+V1*RS
      C1=C1N/C1D
      C2D=0.5_q*PI*RHO*RHO*(G0-0.5_q)
      C2=EC/C2D

      SCALE1=1._q + C1*RMU + C2*RMU*RMU

      DRS=-RS/3._q/RHO
      DG0=-0.5_q*D*EXP(-A*GRS12)*(A*BET-3._q*GAM-3._q*RS+A*GRS32)*DRS/GRS12

      DC1=(U1+2._q*U2*RS + U2*V1*RS*RS)*DRS/C1D/C1D
      DC2D=0.5_q*PI*RHO*(2._q*G0+RHO*DG0-1._q)

      DC2=(VC*C2D-EC*DC2D)/C2D/C2D

      SCALE2=DC1*RMU + DC2*RMU*RMU

      VC_SR=(VC*SCALE1-EC*SCALE2)/(SCALE1*SCALE1)
      RETURN
    END FUNCTION VC_SR

!----------------------------------------------------------------------
!######################################################################
!----------------------------------------------------------------------
            
!=====================================================================
! 
! Thomas-Fermi screened exchange functional
!  
!=====================================================================

    FUNCTION EX_SX(RS,RMU,IFLG)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
! Exchange energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION CX(2)
      SAVE CX
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      EX_=-CX(IFLG)/RS

!     RMU is the Thomas Fermi vector supplied in A, RS in a.u.
      RHO = 3/(4._q*PI)/(RS*AUTOA)**3
!      QF_GLOBAL=RMU**2*AUTOA*PI/4
!      A=RMU/QF_GLOBAL

      QF_LOCAL=(3._q*PI*PI*RHO)**(1._q/3._q)
      A=RMU/QF_LOCAL

      FRAC = 1 - (4._q/3._q) * A * ATAN(2._q/A) &
           & - (A*A/6._q) * ( 1._q - ( A*A/4._q  + 3._q) * LOG(1 + (4._q/(A*A) )))
      EX_SX=EX_*FRAC
      RETURN
    END FUNCTION EX_SX

    FUNCTION VX_SX(RS,RMU,IFLG)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
! Exchange energy:
!     IFLG=1:  Paramagnetic results
!     IFLG=2:  Ferromagnetic results
      DIMENSION CX(2)
      SAVE CX
      DATA CX/0.9163305865663_q,1.1545041946774_q/
      VX_=-CX(IFLG)/RS

      RHO = 3/(4._q*PI)/(RS*AUTOA)**3
!      QF_GLOBAL=RMU**2*AUTOA*PI/4
!      A=RMU/QF_GLOBAL
!      FRAC = 1 - (4._q/3._q) * A * ATAN(2._q/A) &
!           & - (A*A/6._q) * ( 1._q - ( A*A/4._q  + 3._q) * LOG(1 + (4._q/(A*A) )))
!      VX_SX=VX_*FRAC

      QF_LOCAL=(3._q*PI*PI*RHO)**(1._q/3._q)
      A=RMU/QF_LOCAL

      FRAC = 1 + (1._q/2._q) * A*A - (1._q/8._q) * LOG( 1 + (4._q/(A*A) )) * A*A*A*A &
           &  - (1._q/2._q) * LOG( 1 + (4._q/(A*A))) *A*A
      VX_SX=EX_SX(RS,RMU,IFLG)+(1._q/4._q)*VX_*FRAC


      RETURN
    END FUNCTION VX_SX

! jP
!--------------------------------------------------------------------
! The routines for the evaluation of the spin-polarized 
! range-separated exchange-correlation energy in the LDA,
! LSDSR, ECORRLR, VCORRLR, EXCHANGELR, VEXCHANGELR, ECPW, and GPW,
! as well as the functions
! G0, G0D, DPOL, DPOLD, QRPA, QRPAD
! are downloaded from Paola Gori-Giorgi's webpage
! http://www.lct.jussieu.fr/pagesperso/gori/elegas.html
! [Phys. Rev. B 73, 155111 (2006)].
! They have been adapted to f90 by jP (04/25/08).
!--------------------------------------------------------------------
      SUBROUTINE lsdsr(rs,z,mu,excsr,vxcsrup,vxcsrdown)
!   Hartree atomic units used
!   for given density parameter 'rs', relative spin polarization 'z'= (nu -nd)/n
!   and cutoff parameter 'mu' 
!   gives the complementary  short-range exchange-correlation
!   energy  (i.e., xc energy of jellium minus xc energy of long-range
!   interacting electron gas) => 'excsr'
!   and the corresponding exchange-correlation potentials for
!   spin-up and spin-down electrons => 'vxcsrup','vxcsrdown'
!   from Paziani, Moroni, Gori-Giorgi, and Bachelet, cond-mat/0601343
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) :: rs,z,mu,excsr,vxcsrup,vxcsrdown
        REAL(q) :: eclr,exlr,ec,ecd,ecz,ex
        REAL(q) :: vclrup,vclrdown,vxlrup,vxlrdown
        REAL(q) :: vxup,vxdown,vcup,vcdown
        REAL(q) :: alpha,cf

        alpha=(4.0_q/9.0_q/pi)**(1.0_q/3.0_q)
        cf=1.0_q/alpha
        
        ex=-3.0_q*cf/rs/8.0_q/pi*((1.0_q+z)**(4.0_q/3.0_q)+&
        &     (1.0_q-z)**(4.0_q/3.0_q))
        
        vxup=-(1.0_q+z)**(1.0_q/3.0_q)*(3.0_q/2.0_q/PI)**(2.0_q/3.0_q)/rs
        vxdown=-(1.0_q-z)**(1.0_q/3.0_q)*(3.0_q/2.0_q/PI)**(2.0_q/3.0_q)/rs
        
        call ecPW(rs,z,ec,ecd,ecz)
        vcup=ec-rs/3.0_q*ecd-(z-1.0_q)*ecz
        vcdown=ec-rs/3.0_q*ecd-(z+1.0_q)*ecz
        
        call exchangelr(rs,z,mu,exlr)
        call vexchangelr(rs,z,mu,vxlrup,vxlrdown)
        
        call ecorrlr(rs,z,mu,eclr)
        call vcorrlr(rs,z,mu,vclrup,vclrdown)
        
        excsr=ex+ec-(exlr+eclr)
        vxcsrup=vxup+vcup-(vxlrup+vclrup)
        vxcsrdown=vxdown+vcdown-(vxlrdown+vclrdown)

        RETURN
        END SUBROUTINE lsdsr

!----------------------------------------------------------------------
! Routine to calculate the spin-polarized
! short-range correlation energy 
! in the local density approximation -> see routine
! LSDSR; 
! Adaptation done by jP (04/25/08)
!----------------------------------------------------------------------
      SUBROUTINE ecorrsr(rs,z,mu,ecsr)
!   Hartree atomic units used
!   for given density parameter 'rs', relative spin polarization 'z'= (nu -nd)/n
!   and cutoff parameter 'mu' 
!   gives the complementary  short-range exchange-correlation
!   energy  (i.e., xc energy of jellium minus xc energy of long-range
!   interacting electron gas) => 'excsr'
!   and the corresponding exchange-correlation potentials for
!   spin-up and spin-down electrons => 'vxcsrup','vxcsrdown'
!   from Paziani, Moroni, Gori-Giorgi, and Bachelet, cond-mat/0601343
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) :: rs,z,mu,ecsr
        REAL(q) :: eclr,ec,ecd,ecz

        
        call ecPW(rs,z,ec,ecd,ecz)
        call ecorrlr(rs,z,mu,eclr)
        
        ecsr=ec-eclr
        RETURN
      END SUBROUTINE ecorrsr

!----------------------------------------------------------------------
! Routine to calculate the spin-polarized
! short-range correlation potentials
! in the local density approximation -> see routine
! LSDSR; 
! Adaptations w.r.t. VASP done by jP (04/25/08)
!----------------------------------------------------------------------
      SUBROUTINE vcorrsr(rs,z,mu,vcsrup,vcsrdown)
!   Hartree atomic units used
!   for given density parameter 'rs', relative spin polarization 'z'= (nu -nd)/n
!   and cutoff parameter 'mu' 
!   gives the complementary  short-range exchange-correlation
!   energy  (i.e., xc energy of jellium minus xc energy of long-range
!   interacting electron gas) => 'excsr'
!   and the corresponding exchange-correlation potentials for
!   spin-up and spin-down electrons => 'vxcsrup','vxcsrdown'
!   from Paziani, Moroni, Gori-Giorgi, and Bachelet, cond-mat/0601343
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) :: rs,z,mu,vcsrup,vcsrdown
        REAL(q) :: ec,ecd,ecz
        REAL(q) :: vclrup,vclrdown
        REAL(q) :: vcup,vcdown
        
        call ecPW(rs,z,ec,ecd,ecz)
        vcup=ec-rs/3.0_q*ecd-(z-1.0_q)*ecz
        vcdown=ec-rs/3.0_q*ecd-(z+1.0_q)*ecz
        
        call vcorrlr(rs,z,mu,vclrup,vclrdown)
        
        vcsrup=vcup-vclrup
        vcsrdown=vcdown-vclrdown
        RETURN
      END SUBROUTINE vcorrsr 
     
      SUBROUTINE ecorrlr(rs,z,mu,eclr)
!   Hartree atomic units used
!   for given density parameter rs, relative spin polarization z=(nu-nd)/n
!   and cutoff parameter mu 
!   gives the correlation energy of the LR gas
!    => eclr
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) :: rs,z,mu,eclr,ec,ecd,ecz
        REAL(q) :: alpha,cf,phi
        REAL(q) :: d2anti,d3anti
        REAL(q) :: coe2,coe3,coe4,coe5
        REAL(q) :: a1,a2,a3,a4,b0
        REAL(q) :: q1a,q2a,q3a,t1a,t2a,t3a,adib

        alpha=(4.0_q/9.0_q/pi)**(1.0_q/3.0_q)
        cf=1.0_q/alpha
        
        phi=((1.0_q+z)**(2.0_q/3.0_q)+(1.0_q-z)**(2.0_q/3.0_q))/2.0_q
!  parameters from the fit
        adib   = 0.784949_q   
        q1a    = -0.388_q   
        q2a    = 0.676_q   
        q3a    = 0.547_q   
        t1a    = -4.95_q   
        t2a    = 1._q    
        t3a    = 0.31_q   
        
        b0=adib*rs
        
        d2anti=(q1a*rs+q2a*rs**2)*exp(-abs(q3a)*rs)/rs**2
        d3anti=(t1a*rs+t2a*rs**2)*exp(-abs(t3a)*rs)/rs**3
        
        coe2=-3._q/8._q/rs**3*(1._q-z**2)*(gg0(rs)-0.5_q)
        
        coe3=-(1._q-z**2)*gg0(rs)/(sqrt(2._q*pi)*rs**3)
        
        if(abs(z).eq.1._q) then
           
           coe4=-9._q/64._q/rs**3*(dpol(rs) &
                &        -cf**2*2**(5._q/3._q)/5._q/rs**2) 
           coe5=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*dpol(rs)
           
        else
           
           coe4=-9._q/64._q/rs**3*(((1._q+z)/2._q)**2* &
                &        dpol(rs*(2/(1._q+z))**(1._q/3._q))+((1._q-z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q-z))**(1._q/3._q))+ &
                &        (1.-z**2)*d2anti-cf**2/10._q*((1._q+z)**(8._q/3._q) &
                &        +(1.-z)**(8._q/3._q))/rs**2)
           
           coe5=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*(((1._q+z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q+z))**(1._q/3._q))+((1._q-z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q-z))**(1._q/3._q))+(1._q-z**2)* &
                &        d3anti)
        endif
        
        call ecPW(rs,z,ec,ecd,ecz)
        
        a1=4._q*b0**6*coe3+b0**8*coe5
        a2=4._q*b0**6*coe2+b0**8*coe4+6._q*b0**4*ec
        a3=b0**8*coe3
        a4=b0**6*(b0**2*coe2+4._q*ec)
        
        eclr=(phi**3*Qrpa(mu*sqrt(rs)/phi)+a1*mu**3+a2*mu**4+a3*mu**5+ &
             &     a4*mu**6+b0**8*mu**8*ec)/((1._q+b0**2*mu**2)**4)
        
        RETURN
      END SUBROUTINE ecorrlr

      SUBROUTINE vcorrlr(rs,z,mu,vclrup,vclrdown)
!   Hartree atomic units used
!   for given density parameter rs, relative spin polarization z=(nu-nd)/n
!   and cutoff mu it gives the correlation LSD potential for LR interaction
!    => vclrup (spin-up electrons), vclrdown (spin-down electrons)
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) :: rs,z,mu,eclr,eclrrs,eclrz,vclrup,vclrdown
        REAL(q) :: ec,ecd,ecz
        REAL(q) :: alpha,cf,phi
        REAL(q) :: d2anti,d3anti
        REAL(q) :: d2antid,d3antid,x
        REAL(q) :: coe2,coe3,coe4,coe5
        REAL(q) :: coe2rs,coe3rs,coe4rs,coe5rs
        REAL(q) :: coe2z,coe3z,coe4z,coe5z
        REAL(q) :: a1,a2,a3,a4,a5,b0,a1rs,a2rs,a3rs,a4rs,a5rs
        REAL(q) :: b0rs,a1z,a2z,a3z,a4z,a5z,b0z
        REAL(q) :: q1a,q2a,q3a,t1a,t2a,t3a,adib
        
        alpha=(4._q/9._q/pi)**(1._q/3._q)
        cf=1._q/alpha
        
        phi=((1._q+z)**(2._q/3._q)+(1._q-z)**(2._q/3._q))/2._q
!  parameters from the fit
        adib   = 0.784949_q   
        q1a    = -0.388_q   
        q2a    = 0.676_q   
        q3a    = 0.547_q   
        t1a    = -4.95_q   
        t2a    = 1._q    
        t3a    = 0.31_q   
        
        b0=adib*rs
        
        d2anti=(q1a+q2a*rs)*exp(-q3a*rs)/rs
        d3anti=(t1a+t2a*rs)*exp(-t3a*rs)/rs**2
        
        d2antid=-((q1a + q1a*q3a*rs + q2a*q3a*rs**2)/ &
             &    rs**2)*exp(-q3a*rs)
        d3antid=-((rs*t2a*(1 + rs*t3a) + t1a*(2 + rs*t3a))/ &
             &    rs**3)*exp(-rs*t3a)
        
        coe2=-3._q/8._q/rs**3*(1._q-z**2)*(gg0(rs)-0.5_q)
        coe2rs=-3._q/8._q/rs**3*(1._q-z**2)*gg0d(rs)+ &
             &     9._q/8._q/rs**4*(1._q-z**2)*(gg0(rs)-0.5_q)
        coe2z=-3._q/8._q/rs**3*(-2._q*z)*(gg0(rs)-0.5_q)

        coe3=-(1._q-z**2)*gg0(rs)/(sqrt(2._q*pi)*rs**3)
        coe3rs=-(1._q-z**2)*gg0d(rs)/(sqrt(2._q*pi)*rs**3)+ &
             &    3._q*(1._q-z**2)*gg0(rs)/(sqrt(2._q*pi)*rs**4) 
        coe3z=2._q*z*gg0(rs)/(sqrt(2._q*pi)*rs**3)
        
        if(abs(z).eq.1._q) then
           
           coe4=-9._q/64._q/rs**3*(dpol(rs) &
                &        -cf**2*2**(5._q/3._q)/5._q/rs**2)
           coe4rs=-3._q/rs*coe4-9._q/64._q/rs**3*(dpold(rs) &
                &        +2._q*cf**2*2**(5._q/3._q)/5._q/rs**3)
           coe4z=-9._q/64._q/rs**3*(dpol(rs)-rs/6._q*dpold(rs)-2._q*d2anti &
                &       -4._q/15._q/rs**2*cf**2*2._q**(5._q/3._q))*z
           coe5=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*dpol(rs)
           coe5rs=-3._q/rs*coe5-9._q/40._q/(sqrt(2._q*pi)*rs**3)*dpold(rs)
           coe5z=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*(dpol(rs)-rs/6._q* &
                &       dpold(rs)-2._q*d3anti)*z
           
        else
           
           coe4=-9._q/64._q/rs**3*(((1._q+z)/2._q)**2* &
                &        dpol(rs*(2/(1._q+z))**(1._q/3._q))+((1._q-z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q-z))**(1._q/3._q))+ &
                &        (1.-z**2)*d2anti-cf**2/10._q*((1._q+z)**(8._q/3._q) &
                &        +(1.-z)**(8._q/3._q))/rs**2)
           coe4rs=-3._q/rs*coe4-9._q/64._q/rs**3*( &
                &        ((1._q+z)/2._q)**(5._q/3._q)*dpold(rs*(2/(1._q+z))** &
                &        (1._q/3._q))+((1._q-z)/2._q)**(5._q/3._q)* &
                &        dpold(rs*(2/(1._q-z))**(1._q/3._q))+(1._q-z**2)* &
                &        d2antid+cf**2/5._q*((1._q+z)**(8._q/3._q) &
                &        +(1._q-z)**(8._q/3._q))/rs**3)
           coe4z=-9._q/64._q/rs**3*(1._q/2._q*(1._q+z)* &
                &        dpol(rs*(2/(1._q+z))**(1._q/3._q))-1._q/2._q*(1._q-z)* &
                &        dpol(rs*(2/(1._q-z))**(1._q/3._q))-rs/6._q* &
                &        ((1._q+z)/2._q)**(2._q/3._q)*dpold(rs*(2/(1._q+z)) &
                &        **(1._q/3._q))+rs/6._q*((1._q-z)/2._q)**(2._q/3._q) &
                &        *dpold(rs*(2/(1._q-z))**(1._q/3._q))-2._q*z*d2anti- &
                &        4._q/15._q/rs**2*cf**2*((1._q+z)**(5._q/3._q)- &
                &        (1._q-z)**(5._q/3._q)))
           
           coe5=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*(((1._q+z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q+z))**(1._q/3._q))+((1._q-z)/2._q)**2 &
                &        *dpol(rs*(2._q/(1._q-z))**(1._q/3._q))+(1._q-z**2)* &
                &        d3anti)
           coe5rs=-3._q/rs*coe5-9._q/(40._q*sqrt(2._q*pi)*rs**3)*( &
                &        ((1._q+z)/2._q)**(5._q/3._q)*dpold(rs*(2/(1._q+z))** &
                &        (1._q/3._q))+((1._q-z)/2._q)**(5._q/3._q)* &
                &        dpold(rs*(2/(1._q-z))**(1._q/3._q))+(1._q-z**2)* &
                &        d3antid)
           coe5z=-9._q/40._q/(sqrt(2._q*pi)*rs**3)*(1._q/2._q*(1._q+z)* &
                &        dpol(rs*(2/(1._q+z))**(1._q/3._q))-1._q/2._q*(1._q-z)* &
                &        dpol(rs*(2/(1._q-z))**(1._q/3._q))-rs/6._q* &
                &        ((1._q+z)/2._q)**(2._q/3._q)*dpold(rs*(2/(1._q+z)) &
                &        **(1._q/3._q))+rs/6._q*((1._q-z)/2._q)**(2._q/3._q) &
                &        *dpold(rs*(2/(1._q-z))**(1._q/3._q))-2._q*z*d3anti)
           
        endif
        
        call ecPW(rs,z,ec,ecd,ecz)
        
        a1=4._q*b0**6*coe3+b0**8*coe5
        a1rs=24._q*adib*b0**5*coe3+4._q*b0**6*coe3rs+8._q*adib*b0**7* &
             &     coe5+b0**8*coe5rs
        a1z=4._q*b0**6*coe3z+b0**8*coe5z
        
        a2=4._q*b0**6*coe2+b0**8*coe4+6._q*b0**4*ec
        a2rs=24._q*adib*b0**5*coe2+4._q*b0**6*coe2rs+8._q*adib*b0**7* &
             &     coe4+b0**8*coe4rs+24._q*adib*b0**3*ec+6._q*b0**4*ecd
        a2z=4._q*b0**6*coe2z+b0**8*coe4z+6._q*b0**4*ecz
        
        a3=b0**8*coe3
        a3rs=8._q*adib*b0**7*coe3+b0**8*coe3rs
        a3z=b0**8*coe3z
        
        a4=b0**6*(b0**2*coe2+4._q*ec)
        a4rs=8._q*adib*b0**7*coe2+b0**8*coe2rs+24._q*adib*b0**5*ec+ &
             &     4._q*b0**6*ecd
        a4z=b0**6*(b0**2*coe2z+4._q*ecz)
        
        a5=b0**8*ec
        a5rs=8._q*adib*b0**7*ec+b0**8*ecd
        a5z=b0**8*ecz
        
        x=mu*sqrt(rs)/phi
        
        eclr=(phi**3*Qrpa(x)+a1*mu**3+a2*mu**4+a3*mu**5+ &
             &     a4*mu**6+a5*mu**8)/((1._q+b0**2*mu**2)**4)
        
        eclrrs=-4._q/(1._q+b0**2*mu**2)*2._q*adib*b0*mu**2*eclr+ &
             &     1._q/((1._q+b0**2*mu**2)**4)*(phi**2*mu/(2._q*sqrt(rs)) &
             &     *Qrpad(x)+ &
             &     a1rs*mu**3+a2rs*mu**4+a3rs*mu**5+a4rs*mu**6+a5rs*mu**8)
        
        if(z.eq.1._q) then
           vclrup=eclr-rs/3._q*eclrrs
           vclrdown=0._q
        elseif(z.eq.-1._q) then
           vclrup=0._q
           vclrdown=eclr-rs/3._q*eclrrs
        else
           
           eclrz=(phi**2*((1._q+z)**(-1._q/3._q)-(1._q-z)**(-1._q/3._q)) &
                &        *Qrpa(x)-phi*Qrpad(x)*mu*sqrt(rs)*((1._q+z)**(-1._q/3._q) &
                &        -(1._q-z)**(-1._q/3._q))/3._q+ &
                &        a1z*mu**3+a2z*mu**4+a3z*mu**5+ &
                &        a4z*mu**6+a5z*mu**8)/((1._q+b0**2*mu**2)**4)
           
           vclrup=eclr-rs/3._q*eclrrs-(z-1._q)*eclrz
           vclrdown=eclr-rs/3._q*eclrrs-(z+1._q)*eclrz
        endif
        RETURN
      END SUBROUTINE vcorrlr

      FUNCTION gg0(x)
        USE prec
        USE constant
!   on-top pair-distribution function
!   Gori-Giorgi and Perdew, PRB 64, 155102 (2001)
!   x -> rs
        IMPLICIT NONE
        REAL(q) :: gg0
        REAL(q) ::  C0f,D0f,E0f,F0f,x

        gg0              = 0.0_q
        C0f             = 0.0819306_q  
        D0f             = 0.752411_q    
        E0f             = -0.0127713_q   
        F0f             = 0.00185898_q   
        gg0=(1._q-(0.7317_q-D0f)*x+C0f*x**2+E0f*x**3+ &
             &     F0f*x**4)*exp(-abs(D0f)*x)/2._q
      RETURN
    END FUNCTION gg0

    FUNCTION gg0d(rs)
        USE prec
!   derivative of on-top pair-distribution function
!   Gori-Giorgi and Perdew, PRB 64, 155102 (2001)
        IMPLICIT NONE
        REAL(q) :: gg0d
        REAL(q) :: Bgg0,Cgg0,Dgg0,Egg0,Fgg0,rs

        gg0d             = 0.0_q
        Cgg0             = 0.0819306_q    
        Fgg0             = 0.752411_q     
        Dgg0             = -0.0127713_q   
        Egg0             = 0.00185898_q
        Bgg0             =0.7317_q-Fgg0
        gg0d=(-Bgg0+2*Cgg0*rs+3*Dgg0*rs**2+4*Egg0*rs**3)/2._q*exp(-Fgg0*rs) &
             &   - (Fgg0*(1 - Bgg0*rs + Cgg0*rs**2 + Dgg0*rs**3 + Egg0*rs**4))/ &
             &   2._q*exp(-Fgg0*rs)
        RETURN
      END FUNCTION gg0d

      FUNCTION dpol(rs)
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) dpol
        REAL(q) :: cf,rs,p2p,p3p

        cf=(9._q*pi/4._q)**(1._q/3._q)
        p2p    = 0.04_q 
        p3p    = 0.4319_q   
        dpol=2._q**(5._q/3._q)/5._q*cf**2/rs**2*(1._q+(p3p-0.454555_q)*rs) &
             &     /(1._q+p3p*rs+p2p*rs**2)
      RETURN
    END FUNCTION dpol
    
    FUNCTION dpold(rs)
        USE prec
        USE constant
        IMPLICIT NONE
        REAL(q) dpold
        REAL(q) ::  cf,rs,p2p,p3p

        cf=(9._q*pi/4._q)**(1._q/3._q)
        p2p    = 0.04_q 
        p3p    = 0.4319_q   
        dpold=2._q**(5._q/3._q)/5._q*cf**2* &
             & (-2. + (0.454555 - 4.*p3p)*rs + &
             &    (-4.*p2p + &
             &       (0.90911 - 2.*p3p)*p3p)*rs**2 &
             &      + p2p*(1.363665 - 3.*p3p)* &
             &     rs**3)/ &
             &  (rs**3*(1. + p3p*rs + p2p*rs**2)**2)
      RETURN
    END FUNCTION dpold

    FUNCTION Qrpa(x)
      USE prec
      USE constant
      IMPLICIT NONE
        REAL(q) Qrpa
      REAL(q) :: a2,b2,c2,d2,x,Acoul

      Acoul=2._q*(log(2._q)-1._q)/pi**2
      a2              = 5.84605_q 
      c2              = 3.91744_q 
      d2              = 3.44851_q
      b2=d2-3._q/(2._q*pi*Acoul)*(4._q/(9._q*pi))**(1._q/3._q)
      Qrpa=Acoul*log((1._q+a2*x+b2*x**2+c2*x**3)/(1._q+a2*x+d2*x**2))
      RETURN
    END FUNCTION Qrpa

    FUNCTION Qrpad(x)
      USE prec
      USE constant
      IMPLICIT NONE
        REAL(q) Qrpad
      REAL(q) :: a2,b2,c2,d2,x,Acoul
      
      Acoul=2._q*(log(2._q)-1._q)/pi**2
      a2              = 5.84605_q 
      c2              = 3.91744_q 
      d2              = 3.44851_q
      b2=d2-3._q/(2._q*pi*Acoul)*(4._q/(9._q*pi))**(1._q/3._q)
      Qrpad=Acoul*((x*(b2*(2._q + a2*x) + &
           &      c2*x*(3._q + 2._q*a2*x) + &
           &      d2*(-2._q - a2*x + c2*x**3)))/ &
           &  ((1._q + a2*x + d2*x**2)* &
           &    (1._q + a2*x + b2*x**2 + c2*x**3)))
      RETURN
    END FUNCTION Qrpad
    
    SUBROUTINE exchangelr(rs,z,mu,exlr)
      USE prec
      USE constant
      USE c2f_interface, ONLY : ERRF
!   Hartree atomic units used
!   for given density parameter rs, relative spin polarization z=(nu-nd)/n
!   and cutoff mu it gives the exchange energy of the LR gas
!    => exlr
      IMPLICIT NONE
      REAL(q) :: rs,z,mu,exlr
      REAL(q) :: alpha,fx,y

      alpha=(4._q/9._q/pi)**(1._q/3._q)
      if(abs(z).eq.1._q) then
         y=mu*alpha*rs/2._q/2._q**(1._q/3._q)
         fx=-((-3*y + 4*y**3 +(2*y - 4*y**3)*exp(-1./(4.*y**2)) + &
              &        sqrt(pi)*ERRF(1/(2.*y)))/pi)
         exlr=mu*fx
      else
         y=mu*alpha*rs/2._q/(1.+z)**(1._q/3._q)
         fx=-((-3*y + 4*y**3 +(2*y - 4*y**3)*exp(-1./(4.*y**2)) + &
              &        sqrt(pi)*ERRF(1/(2.*y)))/pi)
         exlr=(1._q+z)*mu*fx/2._q
         y=mu*alpha*rs/2._q/(1.-z)**(1._q/3._q)
         fx=-((-3*y + 4*y**3 +(2*y - 4*y**3)*exp(-1./(4.*y**2)) + &
              &        sqrt(pi)*ERRF(1/(2.*y)))/pi)
         exlr=exlr+(1._q-z)*mu*fx/2._q
      endif
      RETURN
    END SUBROUTINE exchangelr
    
    SUBROUTINE vexchangelr(rs,z,mu,vxlrup,vxlrdown)
      USE prec
      USE constant
      USE c2f_interface, ONLY : ERRF
!   Hartree atomic units used
!   for given density parameter rs, relative spin polarization z=(nu-nd)/n
!   and cutoff mu it gives the exchange LSD potential for LR interaction
!    => vxlrup (spin-up electrons), vxlrdown (spin-down electrons)
      IMPLICIT NONE
      REAL(q) :: rs,z,mu,vxlrup,vxlrdown
      REAL(q) :: alpha,fx,fx1,y,exlr,derrs,derz

      alpha=(4._q/9._q/pi)**(1._q/3._q)
      if(z.eq.1._q) then
         vxlrup=(rs*alpha*mu**2)/ &
              &   (2**(1._q/3._q)*pi) - (rs*alpha*mu**2)/(2**(1._q/3._q)*pi)* &
              &     exp(-2**(2._q/3._q)/(rs**2*alpha**2*mu**2)) - &
              &  (mu*ERRF(2**(1._q/3._q)/(rs*alpha*mu)))/sqrt(Pi)
         vxlrdown=0._q
      elseif(z.eq.-1._q) then
         vxlrdown=(rs*alpha*mu**2)/ &
              &   (2**(1._q/3._q)*pi) - (rs*alpha*mu**2)/(2**(1._q/3._q)*pi)* &
              &     exp(-2**(2._q/3._q)/(rs**2*alpha**2*mu**2)) - &
              &  (mu*ERRF(2**(1._q/3._q)/(rs*alpha*mu)))/sqrt(Pi)
         vxlrup=0._q
      else       
         y=mu*alpha*rs/2._q/(1.+z)**(1._q/3._q)
         fx=-((-3*y + 4*y**3 +(2*y - 4*y**3)*exp(-1./(4.*y**2)) + &
              &        sqrt(pi)*ERRF(1/(2.*y)))/pi)
         fx1=(3._q*(1 + (-4._q + 4._q*exp(-1._q/(4._q*y**2)))*y**2))/pi
         derrs=1._q/4._q*(1._q+z)**(2._q/3._q)*mu**2*alpha*fx1
         derz=1._q/2._q*mu*fx-1._q/6._q*fx1*mu*y
         vxlrup=rs/3._q*derrs+(z-1._q)*derz
         vxlrdown=rs/3._q*derrs+(z+1._q)*derz
         
         y=mu*alpha*rs/2._q/(1.-z)**(1._q/3._q)
         fx=-((-3*y + 4*y**3 +(2*y - 4*y**3)*exp(-1./(4.*y**2)) + &
              &        sqrt(pi)*ERRF(1/(2.*y)))/pi)
         fx1=(3._q*(1 + (-4._q + 4._q*exp(-1._q/(4._q*y**2)))*y**2))/pi
         derrs=1._q/4._q*(1._q-z)**(2._q/3._q)*mu**2*alpha*fx1
         derz=-1._q/2._q*mu*fx+1._q/6._q*fx1*mu*y
         vxlrup=vxlrup+rs/3._q*derrs+(z-1._q)*derz
         vxlrdown=vxlrdown+rs/3._q*derrs+(z+1._q)*derz
         
         call exchangelr(rs,z,mu,exlr)
         vxlrup=exlr-vxlrup
         vxlrdown=exlr-vxlrdown
      endif
      RETURN
    END SUBROUTINE vexchangelr

!-----------------------------------------------------------------------
! correlation energy and its derivative w.r.t. rs and z at mu=infinity
! Perdew & Wang PRB 45, 13244 (1992)
!-----------------------------------------------------------------------
    SUBROUTINE ecPW(x,y,ec,ecd,ecz)
      USE prec
      USE constant
! in Hartree; ec=ec(rs,zeta)
! x -> rs; y -> zeta
!   ecd is d/drs ec
!   ecz is d/dz ec
      IMPLICIT NONE
      REAL(q) :: f02,ff,x,y,ec,ecd,ec0,ec0d,ec1,ec1d
      REAL(q) :: aaa,G,Gd,alfac,alfacd,ecz
      
      f02=4._q/(9._q*(2._q**(1._q/3._q)-1._q))
      
      ff=((1._q+y)**(4._q/3._q)+(1._q-y)**(4._q/3._q)- &
           &     2._q)/(2._q**(4._q/3._q)-2._q)
      
      aaa=(1._q-log(2._q))/pi**2
      call  GPW(x,aaa,0.21370_q,7.5957_q,3.5876_q,&
           &     1.6382_q,0.49294_q,G,Gd)
      ec0=G
      ec0d=Gd
      
      aaa=aaa/2._q
      call GPW(x,aaa,0.20548_q,14.1189_q,6.1977_q,&
           &     3.3662_q,0.62517_q,G,Gd)
      ec1=G
      ec1d=Gd
      call GPW(x,0.016887_q,0.11125_q,10.357_q,3.6231_q,&
           &     0.88026_q,0.49671_q,G,Gd)
      alfac=-G
      alfacd=-Gd
      
      ec=ec0+alfac*ff/f02*(1._q-y**4)+(ec1-ec0)*ff*y**4
      ecd=ec0d+alfacd*ff/f02*(1._q-y**4)+(ec1d-ec0d)* &
           &     ff*y**4
      ecz=alfac*(-4._q*y**3)*ff/f02+alfac*(1._q-y**4)/f02* &
           &     4._q/3._q*((1._q+y)**(1._q/3._q)-(1._q-y)**(1._q/3._q))/ &
           &     (2._q**(4._q/3._q)-2._q)+(ec1-ec0)*(4._q*y**3*ff+ &
           &     4._q/3._q*((1._q+y)**(1._q/3._q)-(1._q-y)**(1._q/3._q))/ &
           &     (2._q**(4._q/3._q)-2._q)*y**4)
      
      RETURN
    END SUBROUTINE ecPW
    
    SUBROUTINE GPW(x,Ac,alfa1,beta1,beta2,beta3,beta4,G,Gd)
      USE prec
!   Gd is d/drs G
      IMPLICIT NONE
      REAL(q) ::  G,Gd,Ac,alfa1,beta1,beta2,beta3,beta4,x
      G=-2._q*Ac*(1._q+alfa1*x)*dlog(1._q+1._q/(2._q* &
           &     Ac*(beta1*x**0.5_q+ &
           &     beta2*x+beta3*x**1.5_q+beta4*x**2)))
      Gd=(1._q+alfa1*x)*(beta2+beta1/(2._q*sqrt(x))+3._q*beta3* &
           &     sqrt(x)/2._q+2._q*beta4*x)/((beta1*sqrt(x)+beta2*x+ &
           &     beta3*x**(3._q/2._q)+beta4*x**2)**2*(1._q+1._q/ &
           &     (2._q*Ac*(beta1*sqrt(x)+beta2*x+beta3*x**(3._q/2._q)+ &
           &     beta4*x**2))))-2._q*Ac*alfa1*dlog(1._q+1._q/(2._q*Ac* &
           &     (beta1*sqrt(x)+beta2*x+beta3*x**(3._q/2._q)+ &
           &     beta4*x**2)))
      RETURN
    END SUBROUTINE GPW
!----------------------------------------------------------------------
!----------------------------------------------------------------------

END MODULE xclib

!***********************************************************************
!
! VASP calles the routines in the xclib module  only via the
! subroutines in the module SETEXM
! (SETEXM is the actual interface layer between exchange correlation
!  functionals and VASP)
! it stores for instances which exchange correlation type
! VASP uses, or which interpolation is used (LFCI)
!
!***********************************************************************


  MODULE SETEXM
      USE prec
      IMPLICIT NONE
      INCLUDE "setexm.inc"

      !
      ! string read from INCAR for GGA=XX entry
      !
      CHARACTER (LEN=2),SAVE :: SZGGA
      !
      ! LEXCH specifies the exchange correlation type VASP applies
      ! throughout the calculations
      ! it either corresponds to the GGA entry
      ! or is defaulted from the POTCAR files
      !
      INTEGER,SAVE :: LEXCH=-1
      !
      ! LEXCH_TABLE specifies the exchange correlation type stored
      ! in the table EXCTAB
      ! the size of the table is specified by NSMA
      !
      INTEGER, PARAMETER, PRIVATE :: NSMA  =2000
      INTEGER,SAVE,PRIVATE :: LEXCH_TABLE=-1
      TYPE (exctable),SAVE :: EXCTAB

      ! interpolation of correlation from paramagnetic to 
      ! ferromagnetic case according to 
      ! Vosko, Wilk and Nusair, CAN. J. PHYS. 58, 1200 (1980)
      !
      INTEGER,SAVE :: LFCI=1
      !
      ! amount of LDA exchange
      ! this can be reduced e.g. if exact exchange is used
      ! (e.g. hybrid functionals)
      !
      REAL(q),SAVE :: LDAX=1
      REAL(q),SAVE :: ALDAC=1
      REAL(q),SAVE :: AGGAX=1
      REAL(q),SAVE :: AGGAC=1
      !
      ! screened LDA exchange parameter
      REAL(q),SAVE :: LDASCREEN=0
      ! screened LDA exchange parameter
      REAL(q),SAVE :: LDASCREENC=0
      ! LRANGE_SEPARATED_CORR=.TRUE.  range separated LDA correlation
      ! LRANGE_SEPARATED_CORR=.FALSE. complete LDA correlation (default)
      LOGICAL,SAVE :: LRANGE_SEPARATED_CORR=.FALSE.
      ! LUSE_LONGRANGE_HF  short range LDA exchange interaction only
      ! long range contribution is done in HF
      ! the default is that HF treats short range, and LDA long range
      ! but using this flag the behavior can inverted
      ! the flag should be identical to LRHFCALC in fock.F
      LOGICAL,SAVE :: LUSE_LONGRANGE_HF=.FALSE.
      ! LUSE_THOMAS_FERMI Thomas Fermi screening in local exchange
      ! should be identical to L_THOMAS_FERMI in fock.F
      LOGICAL,SAVE :: LUSE_THOMAS_FERMI=.FALSE.
      ! LUSE_MODEL_HF no local exchange in the short range
      ! a fraction LDAX of the local exchange in the long range limit
      ! should be identical to L_MODEL_HF
      LOGICAL,SAVE :: LUSE_MODEL_HF=.FALSE.
      !
      ! the exchange stack can be used to save 
      ! the present exchange parameters temporarily
      !
      INTEGER, SAVE, PRIVATE ::  ISTACK=0
      REAL(q),SAVE,PRIVATE :: EX_STACK(7,5)
      LOGICAL,SAVE,PRIVATE :: EX_LSTACK(4,5)

    CONTAINS

!******************* SUBROUTINE EXTYP *********************************
!
!  this subroutine interprets the string CEXCH
!  which determines the type of exchange correlation
!  and sets the integer LEXCH accordingly
!
!**********************************************************************

    SUBROUTINE EXTYP(CEXCH,LEXCH)
      USE prec
      USE main_mpi
      IMPLICIT NONE
      CHARACTER (2) CEXCH
      INTEGER LEXCH
#ifdef libbeef
      logical, external :: beef_set_type
#endif


      LEXCH=-1
        IF (CEXCH=='  ') THEN
          LEXCH=0
        ELSE IF (CEXCH=='HL') THEN
          LEXCH=1
        ELSE IF (CEXCH=='PZ') THEN
          LEXCH=2
        ELSE IF (CEXCH=='CA') THEN
          LEXCH=2
        ELSE IF (CEXCH=='WI') THEN
          LEXCH=3
        ELSE IF (CEXCH=='PB') THEN
          LEXCH=4
        ELSE IF (CEXCH=='PW') THEN
          LEXCH=5
        ELSE IF (CEXCH=='LM') THEN
          LEXCH=6
        ELSE IF (CEXCH=='91') THEN
          LEXCH=7
        ELSE IF (CEXCH=='PE') THEN
          LEXCH=8
        ELSE IF (CEXCH=='RP') THEN
          LEXCH=9
        ELSE IF (CEXCH=='VW') THEN
          LEXCH=10
        ELSE IF (CEXCH=='B3') THEN
          LEXCH=11
        ELSE IF (CEXCH=='B5') THEN
          LEXCH=12
!aem the AM05 functional added
        ELSE IF (CEXCH=='AM') THEN
          LEXCH=13
!aem the AM05 functional added
! jP: adding PBEsol
        ELSE IF (CEXCH=='PS') THEN
          LEXCH=14
! jP: adding PBEsol
        ELSE IF (CEXCH=='CO') THEN
          LEXCH=100
! BEEF XC
	ELSE IF (CEXCH=='BF') THEN
#ifdef libbeef
	  LEXCH=17
#if defined(MPI) || defined(MPI_CHAIN)
          IF (.not.beef_set_type(0, (COMM%NODE_ME==COMM%IONODE))) THEN
             IF (COMM%NODE_ME==COMM%IONODE) THEN
#else
          IF (.not.beef_set_type(0, .true.)) THEN
#endif
             WRITE(*,*) 'error setting beef type 0'
#if defined(MPI) || defined(MPI_CHAIN)
             ENDIF
#endif
             STOP
          ENDIF
#else
#if defined(MPI) || defined(MPI_CHAIN)
          IF (COMM%NODE_ME==COMM%IONODE) THEN
#endif
             WRITE(*,*) "VASP needs to be linked against libbeef for"
             WRITE(*,*) "Bayesian error estimation functional support."
             WRITE(*,*) "libbeef sources and binaries can be downloaded from suncat.stanford.edu"
#if defined(MPI) || defined(MPI_CHAIN)
          ENDIF
#endif
          STOP
#endif
! functionals for range-separated ACFDT (LDA - short range RPA):
! a bit akward at the moment since the range separation parameter
! is hard coded for now. Hopefully this will change ...
        ELSE IF (CEXCH=='RA') THEN
! jH-new RPA Perdew-Wang
          LEXCH=20      
        ELSE IF (CEXCH=='03') THEN
          ! \mu = 0.3 A^-1
          LEXCH=21
        ELSE IF (CEXCH=='05') THEN
          ! \mu = 0.5 A^-1
          LEXCH=22
        ELSE IF (CEXCH=='10') THEN
          ! \mu = 1.0 A^-1
          LEXCH=23
        ELSE IF (CEXCH=='20') THEN
          ! \mu = 2.0 A^-1
          LEXCH=24
        ELSE IF (CEXCH=='PL') THEN
! jH-new RPAplus Perdew-Wang         
          LEXCH=30
!vdw jk
        ELSE IF (CEXCH=='RE') THEN
          LEXCH=40
        ELSE IF (CEXCH=='OR') THEN
          LEXCH=41
        ELSE IF (CEXCH=='BO') THEN
          LEXCH=42
        ELSE IF (CEXCH=='MK') THEN
          LEXCH=43
        ELSE IF (CEXCH=='ML') THEN
          LEXCH=44
!vdw jk
        ENDIF
      RETURN
    END SUBROUTINE EXTYP


!***********************************************************************
!
! EX_MOD calculate the exchange energy, possibly reduced by
! short range exchange hole or reduce by the amount that is accounted
! for by the exact exchange
! VX_MOD corresponding potential
!
!***********************************************************************


    FUNCTION EX_MOD(RS,IFLG,TREL)
      USE xclib
      USE constant
      IMPLICIT NONE
      REAL(q) EX_MOD, RS
      INTEGER IFLG
      LOGICAL TREL

      EX_MOD=EX(RS,IFLG,TREL)
      
      IF (LDASCREEN/=0 .OR. LDAX/=1 ) THEN
         IF (LDASCREEN==0) THEN
            EX_MOD=EX_MOD*LDAX
         ELSE IF (LUSE_THOMAS_FERMI) THEN
            EX_MOD=EX_MOD-EX_SX(RS,LDASCREEN,IFLG)*(1-LDAX)
         ELSE IF (LUSE_LONGRANGE_HF) THEN
         ! Iann Gerber: use EX_SR (for LDAX=0, i.e. AEXX=1)
!            EX_MOD=(1-LDAX)*EX(RS,IFLG,.FALSE.)-(EX_MOD-EX_SR(RS,LDASCREEN*AUTOA,IFLG))*(1-LDAX)
            EX_MOD=EX(RS,IFLG,.FALSE.)-(EX_MOD-EX_SR(RS,LDASCREEN*AUTOA,IFLG))*(1-LDAX)
         ELSE IF (LUSE_MODEL_HF) THEN
         ! model sX 
            EX_MOD=(EX(RS,IFLG,.FALSE.)-EX_SR(RS,LDASCREEN*AUTOA,IFLG))*LDAX
         ELSE 
         ! WPBE: LDAX*EX_SR + EX_LR = EX - (1-LDAX)*EX_SR
            EX_MOD=EX(RS,IFLG,.FALSE.)-EX_SR(RS,LDASCREEN*AUTOA,IFLG)*(1-LDAX)
         ENDIF
      ENDIF
    END FUNCTION EX_MOD

    FUNCTION VX_MOD(RS,IFLG,TREL)
      USE xclib
      USE constant
      IMPLICIT NONE
      REAL(q) VX_MOD, RS
      INTEGER IFLG
      LOGICAL TREL

      VX_MOD=VX(RS,IFLG,TREL)
      
      IF (LDASCREEN/=0 .OR. LDAX/=1 ) THEN
         IF (LDASCREEN==0) THEN
            VX_MOD=VX_MOD*LDAX
         ELSE IF (LUSE_THOMAS_FERMI) THEN
            VX_MOD=VX_MOD-VX_SX(RS,LDASCREEN*AUTOA,IFLG)*(1-LDAX)
         ELSE IF (LUSE_LONGRANGE_HF) THEN
         ! Iann Gerber: use VX_SR (for LDAX=0, i.e. AEXX=1)
            VX_MOD=VX(RS,IFLG,.FALSE.)-(VX_MOD-VX_SR(RS,LDASCREEN*AUTOA,IFLG))*(1-LDAX)
         ELSE IF (LUSE_MODEL_HF) THEN
         ! model sX 
            VX_MOD=(VX(RS,IFLG,.FALSE.)-VX_SR(RS,LDASCREEN*AUTOA,IFLG))*LDAX
         ELSE
         ! WPBE: LDAX*VX_SR + VX_LR = VX - (1-LDAX)*VX_SR
            VX_MOD=VX(RS,IFLG,.FALSE.)-VX_SR(RS,LDASCREEN*AUTOA,IFLG)*(1-LDAX)
         ENDIF
      ENDIF
    END FUNCTION VX_MOD

!----------------------------------------------------------------------
!######################################################################
!----------------------------------------------------------------------
! Iann Gerber 18/11/04

! Splitting the correlation part: meaning only take into account of the 
! short-range part if necessary
      FUNCTION EC_MOD(RS,ZETA,IFLG,TREL)
      USE xclib
      USE constant
      IMPLICIT NONE
      REAL(q) EC_MOD, RS
      REAL(q) ZETA              ! relative spin-polarization zeta := (nu-nd)/(nu+nd)
      REAL(q) EC_MOD_TMP
      INTEGER IFLG
      LOGICAL TREL

      EC_MOD=ECVO(RS,IFLG)
      IF (LRANGE_SEPARATED_CORR) THEN
         IF (LDASCREENC==0) THEN
            EC_MOD=EC_MOD
! jP here, we ask for the short-range LSDA correlation energy
         ELSE
            CALL ECORRSR(RS,ZETA,LDASCREENC*AUTOA,EC_MOD)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
            EC_MOD=2.0_q*EC_MOD
            IF (LUSE_LONGRANGE_HF) THEN
! keep short range part of LDA correlation energy
!               EC_MOD=ECVO(RS,1)
!               EC_MOD=EC_SR(RS,LDASCREENC*AUTOA,EC_MOD)
               CALL ECORRSR(RS,ZETA,LDASCREENC*AUTOA,EC_MOD)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
               EC_MOD=2.0_q*EC_MOD
! keep long range part of the LDA correlation energy
!                  EC_MOD=ECVO(RS,1)
!                  EC_MOD=EC_MOD-EC_SR(RS,LDASCREENC*AUTOA,EC_MOD)  
            ELSE
               CALL ECORRLR(RS,ZETA,LDASCREENC*AUTOA,EC_MOD)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
               EC_MOD=2.0_q*EC_MOD            
            ENDIF
         ENDIF
      ENDIF
      END FUNCTION EC_MOD

      FUNCTION VC_MOD(RS,ZETA,IFLG,TREL)
      USE xclib
      USE constant
      IMPLICIT NONE
      REAL(q) VC_MOD, RS
      REAL(q)  ZETA, VCSRUP, VCSRDOWN, VCLRUP, VCLRDOWN
      INTEGER IFLG
      LOGICAL TREL
      
      VC_MOD=VCVO(RS,IFLG)
      IF (LRANGE_SEPARATED_CORR) THEN
         IF (LDASCREENC==0) THEN
            VC_MOD=VC_MOD
         ELSE
            CALL VCORRSR(RS,ZETA,LDASCREENC*AUTOA,VCSRUP,VCSRDOWN)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
            VCSRUP=2.0_q*VCSRUP
            VCSRDOWN=2.0_q*VCSRDOWN
            IF (LUSE_LONGRANGE_HF) THEN
! keep short range part of LDA correlation potential
!               VC_MOD=VC_SR(RS,LDASCREENC*AUTOA,VCVO(RS,IFLG),ECVO(RS,IFLG))
               CALL VCORRSR(RS,ZETA,LDASCREENC*AUTOA,VCSRUP,VCSRDOWN)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
               VCSRUP=2.0_q*VCSRUP
               VCSRDOWN=2.0_q*VCSRDOWN
            ELSE
! keep long range part of the LDA correlation potential
!              VC_MOD=VC_MOD-VC_SR(RS,LDASCREENC*AUTOA,VCVO(RS,IFLG),ECVO(RS,IFLG))
               CALL VCORRLR(RS,ZETA,LDASCREENC*AUTOA,VCLRUP,VCLRDOWN)
! jP: Routines of Paola Gori-Giorig written in a.u.
! jP: conversion to Rydberg units necessary for VASP
               VCSRUP=2.0_q*VCSRUP
               VCSRDOWN=2.0_q*VCSRDOWN
            ENDIF
         ENDIF
      ENDIF
      END FUNCTION VC_MOD
!----------------------------------------------------------------------
!######################################################################
!----------------------------------------------------------------------

!***********************************************************************
!
!  VASP interpolates the XC-energy density from a table (at least
!    the plane wave part)
!  the required table is generated here
!
!***********************************************************************

    SUBROUTINE SETUP_LDA_XC(ISPIN,IU6,IU0,IDIOT)
      USE ini
      IMPLICIT NONE

      INTEGER  ISPIN            ! spin

! arrays for tutor call
      INTEGER IU6,IU0,IDIOT
! temporary
      INTEGER N,NDUMMY
      REAL(q) AMARG
      CHARACTER (1) CSEL
      CHARACTER (2) CEXCH
      IF (LEXCH<0) THEN
         WRITE(*,*) 'internal ERROR in SETUP_LDA_XC: LEXCH has not been set up'
         STOP
      ENDIF
!
! set the exchange correlation type for the internal table
! before the table is used XCTABLE_CHECK should be called
! to ascertain that LEXCH_TABLE is correct and equivalent to LEXCH
!

      CALL SET_EX_TABLE(LEXCH,NEXCH,EXCTAB%EXCTAB,EXCTAB%NEXCHF,EXCTAB%RHOEXC,IU0)

      AMARG=1E30_q  ! natural boundary conditions required
      CALL SPLCOF(EXCTAB%EXCTAB(1,1,1),EXCTAB%NEXCHF(2),NEXCH,AMARG)

      IF (ISPIN==2) THEN
         CALL SPLCOF(EXCTAB%EXCTAB(1,1,2),EXCTAB%NEXCHF(2),NEXCH,AMARG)
         CALL SPLCOF(EXCTAB%EXCTAB(1,1,3),EXCTAB%NEXCHF(2),NEXCH,AMARG)
         CALL SPLCOF(EXCTAB%EXCTAB(1,1,4),EXCTAB%NEXCHF(2),NEXCH,AMARG)
         CALL SPLCOF(EXCTAB%EXCTAB(1,1,5),EXCTAB%NEXCHF(2),NEXCH,AMARG)
         CALL SPLCOF(EXCTAB%EXCTAB(1,1,6),EXCTAB%NEXCHF(2),NEXCH,AMARG)
      ENDIF
      LEXCH_TABLE=LEXCH

      IF (IU6>=0) WRITE(IU6,7004)LEXCH,EXCTAB%RHOEXC(1),EXCTAB%NEXCHF(1), &
     &                     EXCTAB%RHOEXC(2),EXCTAB%NEXCHF(2)
 7004 FORMAT(' exchange correlation table for  LEXCH = ',I8/ &
     &       '   RHO(1)= ',F8.3,5X,'  N(1)  = ',I8/ &
     &       '   RHO(2)= ',F8.3,5X,'  N(2)  = ',I8)

      RETURN
    END SUBROUTINE 


!************************ PROGRAM SET_EX_TABLE *************************
!
! set up the default xc-table
! i.e.
!  Ceperly Alder
!  with standard interpolation to spin
!  with relativistic correction
!
!***********************************************************************

    SUBROUTINE SET_EX_TABLE(LEXCH,N,EXCTAB,NEXCHF,RHOEXC,IU0)

      USE xclib

      INTEGER LEXCH, N, IU0
      REAL(q) :: EXCTAB(N,5,6),RHOEXC(2)
      INTEGER :: NEXCHF(2)
    ! local
      LOGICAL TREL
      CHARACTER (13) CEXCH
      INTEGER LEXCH_LDA
      INTEGER J,I
      REAL(q) :: SLATER, RHOSMA, RHOMAX, RH, EXCP, DEXF, DECF, ALPHA, DRHO, ZETA, FZA, FZB

      TREL=.TRUE.
! convert LEXCH to local format
      IF (LEXCH==0) THEN
        LEXCH_LDA=0
      ELSE IF (LEXCH==1) THEN
    ! Hedin Lundquist
        LEXCH_LDA=4
      ELSE IF (LEXCH==2) THEN
    ! Ceperly-Alder
        LEXCH_LDA=1
      ELSE IF (LEXCH==3) THEN
    ! Wigner
        LEXCH_LDA=6
!vdw jk
      ELSE IF (LEXCH==8 .OR. LEXCH==9 .OR. (LEXCH.ge.40 .and. LEXCH.le.50)) THEN
!vdw jk
    ! Pade approximation of Perdew
        LEXCH_LDA=7
      ELSE IF (LEXCH==10) THEN
    ! Iann Gerber: LDA exchange + VWN correlation
        LEXCH_LDA=2
      ELSE IF (LEXCH==11) THEN
    ! Joachim Paier: B3LYP LDA part is VWNIII-correlation
        LEXCH_LDA=11
      ELSE IF (LEXCH==12) THEN
    ! Joachim Paier: B3LYP LDA part is VWN5-correlation
        LEXCH_LDA=12
!aem AM05 uses the PW LDA correlation (same as PBE)
      ELSE IF (LEXCH==13) THEN
        LEXCH_LDA=7
!aem end addition
! jP: PBEsol - LDA part identical to PBE
      ELSE IF (LEXCH==14) THEN
        LEXCH_LDA=7
! jP: PBEsol - LDA part identical to PBE
!BEEF xc
#ifdef libbeef
      ELSE IF (LEXCH==17) THEN
        LEXCH_LDA=7
#endif
      ELSE IF (LEXCH==100) THEN
        LEXCH_LDA=-1
! jH-new RPA Perdew Wang 
      ELSE IF (LEXCH==20) THEN
       LEXCH_LDA=20
! functionals for range-separated ACFDT (LDA - short range RPA)
      ELSE IF (LEXCH==21) THEN
        ! \mu = 0.3 A^-1
        LEXCH_LDA=21
      ELSE IF (LEXCH==22) THEN
        ! \mu = 0.3 A^-1
        LEXCH_LDA=22
      ELSE IF (LEXCH==23) THEN
        ! \mu = 0.3 A^-1
        LEXCH_LDA=23
      ELSE IF (LEXCH==24) THEN
        ! \mu = 0.3 A^-1
        LEXCH_LDA=24
! jH-new RPAplus Perdew Wang
      ELSE IF (LEXCH==30) THEN
        LEXCH_LDA=30
      ELSE
        LEXCH_LDA=1
      ENDIF

      IF (LEXCH_LDA==0) THEN
         CEXCH='  '
      ELSE IF (LEXCH_LDA==1) THEN
         CEXCH='Ceperly-Alder'
      ELSE IF (LEXCH_LDA==2) THEN
         CEXCH='VW'
      ELSE IF (LEXCH_LDA==3) THEN
         CEXCH='GL'
      ELSE IF (LEXCH_LDA==4) THEN
         CEXCH='HL'
      ELSE IF (LEXCH_LDA==5) THEN
         CEXCH='BH'
      ELSE IF (LEXCH_LDA==6) THEN
         CEXCH='WI'
      ELSE IF (LEXCH_LDA==7) THEN
         CEXCH='PB'
      ELSE IF (LEXCH_LDA==-1) THEN
         CEXCH='no ex-corr'
      ELSE IF (LEXCH_LDA==11) THEN
! Joachim Paier
         CEXCH='VWN3'
      ELSE IF (LEXCH_LDA==12) THEN
! Joachim Paier
         CEXCH='VWN5'
! jH-new RPA Perdew Wang 
      ELSE IF (LEXCH_LDA==20) THEN 
       CEXCH='RPA Perdew-Wang'
! functionals for range-separated ACFDT (LDA - short range RPA)
      ELSE IF (LEXCH_LDA==21) THEN
        ! \mu = 0.3 A^-1
        CEXCH='0.3'
      ELSE IF (LEXCH_LDA==22) THEN
        ! \mu = 0.3 A^-1
        CEXCH='0.5'
      ELSE IF (LEXCH_LDA==23) THEN
        ! \mu = 0.3 A^-1
        CEXCH='1.0'
      ELSE IF (LEXCH_LDA==24) THEN
        ! \mu = 0.3 A^-1
        CEXCH='2.0'
! jH-new RPAplus Perdew Wang 
      ELSE IF (LEXCH_LDA==30) THEN
        CEXCH='RPA+'
      ELSE
         IF (IU0>=0) &
         WRITE(IU0,*) 'internal error in SET_EX_TABLE: Wrong exchange correlation type!', LEXCH_LDA
         STOP
      ENDIF
      

      IF (IU0>=0) THEN
         IF (LEXCH_LDA==7) THEN
            WRITE(IU0,*)'LDA part: xc-table for Pade appr. of Perdew'
         ELSE IF (LEXCH_LDA>20 .AND. LEXCH_LDA<25) THEN
            WRITE(IU0,'(A,A3,A)')'LDA part: xc-table for (LDA - short range RPA): hard-coded for mu= ',CEXCH,' A^-1'
         ELSE
            IF (LFCI==1) THEN 
               WRITE(IU0,*)'LDA part: xc-table for ',CEXCH, ', Vosko type interpolation para-ferro'
            ELSE
               WRITE(IU0,*)'LDA part: xc-table for ',CEXCH, ', standard interpolation'
            ENDIF
         ENDIF
      ENDIF

! Slater parameter
      SLATER=1._q
! standard interpolation   from para- to ferromagnetic corr
      RHOSMA=INT(0.5_q*1000)/1000._q
!#define hugeXCtable
#ifdef hugeXCtable
      RHOMAX=INT(2000.5_q*1000)/1000._q
      WRITE(0,*) '******* RHOMAX set to 2000 instead of 100 *****'
#else
      RHOMAX=INT(100.5_q*1000)/1000._q
#endif


      RHOEXC(1)=RHOSMA
      RHOEXC(2)=RHOMAX
      NEXCHF(1)=NSMA
      NEXCHF(2)=N

      IF (NSMA/=0) THEN
         RH=RHOSMA/NSMA/2
      ELSE
         RH=RHOMAX/N/100
      ENDIF
      J=1

      CALL EXCHG(LEXCH_LDA,RH,EXCP, &
     &           DEXF,DECF,ALPHA,SLATER,TREL)
        EXCTAB(J,1,1)=RH
        EXCTAB(J,1,2)=RH
        EXCTAB(J,1,3)=RH
        EXCTAB(J,1,4)=RH

        EXCTAB(J,2,1)=EXCP
        EXCTAB(J,2,2)=DEXF
        EXCTAB(J,2,3)=DECF
        EXCTAB(J,2,4)=ALPHA


      IF (NSMA/=0) THEN
         DRHO=RHOSMA/NSMA
         DO 100 I=1,NSMA-1
            J=I+1
            RH=DRHO*I
            CALL EXCHG(LEXCH_LDA,RH,EXCP, &
     &                 DEXF,DECF,ALPHA,SLATER,TREL)
          EXCTAB(J,1,1)=RH
          EXCTAB(J,1,2)=RH
          EXCTAB(J,1,3)=RH
          EXCTAB(J,1,4)=RH

          EXCTAB(J,2,1)=EXCP
          EXCTAB(J,2,2)=DEXF
          EXCTAB(J,2,3)=DECF
          EXCTAB(J,2,4)=ALPHA

  100    CONTINUE

         J=NSMA+1
         RH=RHOSMA
         CALL EXCHG(LEXCH_LDA,RH,EXCP, &
     &              DEXF,DECF,ALPHA,SLATER,TREL)
         EXCTAB(J,1,1)=RH
         EXCTAB(J,1,2)=RH
         EXCTAB(J,1,3)=RH
         EXCTAB(J,1,4)=RH

         EXCTAB(J,2,1)=EXCP
         EXCTAB(J,2,2)=DEXF
         EXCTAB(J,2,3)=DECF
         EXCTAB(J,2,4)=ALPHA
      ENDIF

      DRHO=(RHOMAX-RHOSMA)/(N-NSMA)
      DO 200 I=1,N-NSMA-1
         J=I+NSMA+1
         RH=DRHO*I+RHOSMA
         CALL EXCHG(LEXCH_LDA,RH,EXCP, &
     &              DEXF,DECF,ALPHA,SLATER,TREL)
         EXCTAB(J,1,1)=RH
         EXCTAB(J,1,2)=RH
         EXCTAB(J,1,3)=RH
         EXCTAB(J,1,4)=RH

         EXCTAB(J,2,1)=EXCP
         EXCTAB(J,2,2)=DEXF
         EXCTAB(J,2,3)=DECF
         EXCTAB(J,2,4)=ALPHA
  200 CONTINUE

      J=1
      ZETA=0
      FZA =.854960467080682810_q
      FZB =.854960467080682810_q
      EXCTAB(J,1,5)=ZETA
      EXCTAB(J,1,6)=ZETA
      EXCTAB(J,2,5)=FZA
      EXCTAB(J,2,6)=FZB

      DO 800 I=1,N-1
         ZETA=FLOAT(I)/FLOAT(N-1)
         FZA=FZ0(ZETA)/ZETA/ZETA
         FZB=FZ0(ZETA)/ZETA/ZETA
         J=I+1
         EXCTAB(J,1,5)=ZETA
         EXCTAB(J,1,6)=ZETA
         EXCTAB(J,2,5)=FZA
         EXCTAB(J,2,6)=FZB
  800 CONTINUE

      IF (.FALSE.) THEN
      DO 300 I=1,N
         WRITE(97,20) EXCTAB(I,1,1),EXCTAB(I,2,1)
  300 CONTINUE
      DO 400 I=1,N
         WRITE(97,20)  EXCTAB(I,1,2),EXCTAB(I,2,2)
  400 CONTINUE
      DO 500 I=1,N
         WRITE(97,20)  EXCTAB(I,1,3),EXCTAB(I,2,3)
  500 CONTINUE
      DO 600 I=1,N
         WRITE(97,20)  EXCTAB(I,1,4),EXCTAB(I,2,4)
  600 CONTINUE
      DO 700 I=1,N
         WRITE(97,20)  EXCTAB(I,1,5),EXCTAB(I,2,5)
  700 CONTINUE
      DO 710 I=1,N
         WRITE(97,20)  EXCTAB(I,1,6),EXCTAB(I,2,6)
  710 CONTINUE
   20 FORMAT((3(E24.16,2X)))
      ENDIF

      RETURN
   END SUBROUTINE SET_EX_TABLE


!**************** SUBROUTINE EXCHG *************************************
!
! EXCHG calculate the LDA part of xc-energy
! for the fully spin polarized case 
! and for the non magnetic case
! uses subroutines defined in xclib
! this function should be called with care, since the flag LEXCH_LDA
! is not compatible to LEXCH
!
!***********************************************************************

   SUBROUTINE EXCHG(LEXCH_LDA,RHO,EXCP,DEXF,DECF,ALPH,SLATER,TREL)
      USE constant
      USE xclib

      INTEGER LEXCH_LDA

      LOGICAL TREL
      REAL(q) :: RHO,EXCP,DEXF,DECF,ALPH,SLATER
   ! local
      REAL(q) :: RS, RH, ZETA, FZA, FZB, &
           RHOTHD, SE, ECLDA, ECD1LDA, ECD2LDA, ECLDA_MAG, A0

      IF (RHO==0) THEN
         EXCP=0._q
         DEXF=0._q
         DECF=0._q
         ALPH=0._q
         RETURN
      ENDIF

      RHOTHD = RHO**(1/3._q)
      RS = (3._q/(4._q*PI)/RHO)**(1/3._q) /AUTOA
      IF (LEXCH_LDA==0) THEN
         EXCP=SLATER*EX_MOD(RS,1,TREL)
         DEXF=SLATER*EX_MOD(RS,2,TREL)-EXCP
         DECF=0._q
         ALPH=0._q
      ELSE IF (LEXCH_LDA==1) THEN
         EXCP=EX_MOD(RS,1,TREL)+ECCA(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECCA(RS,2)*ALDAC-ECCA(RS,1)*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==2) THEN
! Iann Gerber 18/11/04
!        EXCP=EX_MOD(RS,1,TREL)+ECVO(RS,1)
         ZETA=0.0_q
         EXCP=EX_MOD(RS,1,TREL)+EC_MOD(RS,ZETA,1,TREL)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
!        DECF=ECVO(RS,2)*ALDAC-ECVO(RS,1)*ALDAC
! hack: ZETA set to 1.0_q or 0.0_q, respectively
         DECF=EC_MOD(RS,1._q,2,TREL)*ALDAC-EC_MOD(RS,0._q,1,TREL)*ALDAC
         ALPH=0._q
! Iann Gerber 18/11/04; updated to scrLSDA by Joachim Paier (28/04/08)
      ELSE IF (LEXCH_LDA==3) THEN
         EXCP=EX_MOD(RS,1,TREL)+ECGL(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECGL(RS,2)*ALDAC-ECGL(RS,1)*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==4) THEN
         EXCP=EX_MOD(RS,1,TREL)+ECHL(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECHL(RS,2)*ALDAC-ECHL(RS,1)*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==5) THEN
         EXCP=EX_MOD(RS,1,TREL)+ECBH(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECBH(RS,2)*ALDAC-ECBH(RS,1)*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==6) THEN
         EXCP=EX_MOD(RS,1,TREL)+ECWI(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECWI(RS,2)*ALDAC-ECWI(RS,1)*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==7) THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=1  ! ferromagnetic result
         CALL CORPBE_LDA(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)

         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
!        WRITE(77,'(5F14.7)') RS, ECCA(RS,1),ECLDA,ECCA(RS,2),ECLDA_MAG
         ALPH=0._q
      ELSE IF (LEXCH_LDA==11) THEN
! Joachim Paier
         EXCP=EX_MOD(RS,1,TREL)+ECVOIII(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECVOIII(RS,2)*ALDAC-ECVOIII(RS,1)*ALDAC
         ALPH=0._q
! check is VO similar to CA
!        WRITE(77,'(5F14.7)') RS, ECCA(RS,2),ECVO(RS,2)
!        WRITE(78,'(5F14.7)') RS, ECCA(RS,1),ECVO(RS,1)
      ELSE IF (LEXCH_LDA==12) THEN
! Joachim Paier
         EXCP=EX_MOD(RS,1,TREL)+ECVO(RS,1)*ALDAC
         DEXF=EX_MOD(RS,2,TREL)-EX_MOD(RS,1,TREL)
         DECF=ECVO(RS,2)*ALDAC-ECVO(RS,1)*ALDAC
         ALPH=0._q
! check is VO similar to CA
!        WRITE(77,'(5F14.7)') RS, ECCA(RS,2),ECVO(RS,2)
!        WRITE(78,'(5F14.7)') RS, ECCA(RS,1),ECVO(RS,1)
! jH-new RPA with Perdew Wang para
      ELSE IF (LEXCH_LDA==20) THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_RPA(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=1  ! ferromagnetic result
         CALL CORPBE_LDA_RPA(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q
! Judith Harl
! functionals for range-separated ACFDT (LDA - short range RPA):
! a bit akward at the moment since the range separation parameter
! is hard coded for now. Hopefully this will change ...
      ELSE IF (LEXCH_LDA==21) THEN
! jh --- fuer den spin-polarisierten Fall noch nicht moeglich
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_0_15au(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=0  ! auch paramagnetisch (nicht fuer spin-polarisiert)
         CALL CORPBE_LDA_SR_0_15au(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==22) THEN
! jh --- fuer den spin-polarisierten Fall noch nicht moeglich
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_0_5A(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=0  ! auch paramagnetisch (nicht fuer spin-polarisiert)
         CALL CORPBE_LDA_SR_0_5A(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==23) THEN
! jh --- fuer den spin-polarisierten Fall noch nicht moeglich
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_1_0A(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=0  ! auch paramagnetisch (nicht fuer spin-polarisiert)
         CALL CORPBE_LDA_SR_1_0A(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q
      ELSE IF (LEXCH_LDA==24) THEN
! jh --- fuer den spin-polarisierten Fall noch nicht moeglich
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_2_0A(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=0  ! auch paramagnetisch (nicht fuer spin-polarisiert)
         CALL CORPBE_LDA_SR_2_0A(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q         
! jH-new RPA-plus = ELDAQMC-ELDARPA, exchange term is zero 
      ELSE IF (LEXCH_LDA==30) THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_RPA_PLUS(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA)
         ZETA=1  ! ferromagnetic result
         CALL CORPBE_LDA_RPA_PLUS(RS,ZETA,ECLDA_MAG,ECD1LDA,ECD2LDA)
         EXCP=EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC
         DEXF=EX_MOD(RS,2,.FALSE.)-EX_MOD(RS,1,.FALSE.)
         DECF=ECLDA_MAG*ALDAC-ECLDA*ALDAC
         ALPH=0._q
      ELSE
         EXCP=0._q
         DEXF=0._q
         DECF=0._q
         ALPH=0._q
      ENDIF
      !
      ! for Perdew, Burke Ernzerhof we always use the 
      ! recommended  interpolation from nm to magnetic
      ! 
      IF (LEXCH_LDA==11) THEN
         A0=ALPHA0_III(RS)*ALDAC
         ALPH=DECF-A0
         DECF=A0
      ELSE IF (LEXCH_LDA==7) THEN
         A0=PBE_ALPHA(RS)*ALDAC
         ALPH=DECF-A0
         DECF=A0
! jH-new new switching for RPA 
      ELSE IF (LEXCH_LDA==20) THEN
         A0=RPA_ALPHA(RS)*ALDAC
         ALPH=DECF-A0
         DECF=A0
! jH-new new switching for RPAplus
      ELSE IF (LEXCH_LDA==30) THEN          
         A0=(PBE_ALPHA(RS)-RPA_ALPHA(RS))*ALDAC
         ALPH=DECF-A0
         DECF=A0
      ELSE IF (LFCI==1 .OR. LEXCH_LDA==12 ) THEN
         A0=ALPHA0(RS)*ALDAC
         ALPH=DECF-A0
         DECF=A0
      ENDIF

      EXCP=EXCP*RYTOEV/RHOTHD
      DEXF=DEXF*RYTOEV/RHOTHD
      DECF=DECF*RYTOEV/RHOTHD
      ALPH=ALPH*RYTOEV/RHOTHD

      RETURN
    END SUBROUTINE EXCHG

!************************ SUBROUTINE SET_LEXCH *************************
!
! in principle this is the only function that should be used
! to set the exchange correlation type
! however, presently the fock.F and pseudo.F modules access LEXCH
! directly
!
!***********************************************************************

    SUBROUTINE SET_LEXCH(LEXCH_SET)
      IMPLICIT NONE
      INTEGER LEXCH_SET

      LEXCH=LEXCH_SET

    END SUBROUTINE SET_LEXCH


    SUBROUTINE PUSH_LEXCH(LEXCH_TEMP)
      IMPLICIT NONE
      INTEGER LEXCH_TEMP

      ISTACK=ISTACK+1
      IF (ISTACK>SIZE(EX_STACK,2)) THEN
         WRITE(*,*) 'internal ERROR in PUSH_LEXCH: push already exhausted',ISTACK
         STOP
      ENDIF
      EX_STACK(1,ISTACK)=LEXCH
      LEXCH=LEXCH_TEMP
    END SUBROUTINE PUSH_LEXCH

    SUBROUTINE POP_LEXCH
      IF (ISTACK==0) THEN
         WRITE(*,*) 'internal ERROR in POP_LEXCH: push was not used'
         STOP
      ENDIF
      LEXCH=EX_STACK(1,ISTACK)
      ISTACK=ISTACK-1
      
    END SUBROUTINE POP_LEXCH

!************************ SUBROUTINE SET_LEXCH *************************
!
! subroutine to temporarily use an alternative local 
! exchange correlation functional
!
!***********************************************************************

    SUBROUTINE PUSH_XC_TYPE(LEXCH_TEMP, LDAX_TEMP, LDAC_TEMP, GGAX_TEMP, GGAC_TEMP, LDASCREEN_TEMP)
      IMPLICIT NONE
      INTEGER LEXCH_TEMP
      REAL(q) LDAX_TEMP, LDAC_TEMP, LDASCREEN_TEMP, GGAX_TEMP, GGAC_TEMP

      ISTACK=ISTACK+1
      IF (ISTACK>SIZE(EX_STACK,2)) THEN
         WRITE(*,*) 'internal ERROR in PUSH_XC_TYPE: push already exhausted',ISTACK
         STOP
      ENDIF
      EX_STACK (1,ISTACK)=LEXCH
      EX_STACK (2,ISTACK)=LDAX
      EX_STACK (3,ISTACK)=LDASCREEN
      EX_STACK (4,ISTACK)=LFCI
      EX_STACK (5,ISTACK)=AGGAX
      EX_STACK (6,ISTACK)=AGGAC
      EX_STACK (7,ISTACK)=ALDAC
      EX_LSTACK(1,ISTACK)=LRANGE_SEPARATED_CORR
      EX_LSTACK(2,ISTACK)=LUSE_LONGRANGE_HF
      EX_LSTACK(3,ISTACK)=LUSE_THOMAS_FERMI
      EX_LSTACK(4,ISTACK)=LUSE_MODEL_HF

      LEXCH    =LEXCH_TEMP
      LDAX     =LDAX_TEMP
      ALDAC    =LDAC_TEMP
      LDASCREEN=LDASCREEN_TEMP
      AGGAX    =GGAX_TEMP
      AGGAC    =GGAC_TEMP

    END SUBROUTINE PUSH_XC_TYPE


    SUBROUTINE POP_XC_TYPE
      IF (ISTACK==0) THEN
         WRITE(*,*) 'internal ERROR in POP_XC_TYPE: push was not used'
         STOP
      ENDIF
      LEXCH                =EX_STACK (1,ISTACK)
      LDAX                 =EX_STACK (2,ISTACK)
      LDASCREEN            =EX_STACK (3,ISTACK)
      LFCI                 =EX_STACK (4,ISTACK)
      AGGAX                =EX_STACK (5,ISTACK)
      AGGAC                =EX_STACK (6,ISTACK)
      ALDAC                =EX_STACK (7,ISTACK)
      LRANGE_SEPARATED_CORR=EX_LSTACK(1,ISTACK)
      LUSE_LONGRANGE_HF    =EX_LSTACK(2,ISTACK)
      LUSE_THOMAS_FERMI    =EX_LSTACK(3,ISTACK)
      LUSE_MODEL_HF        =EX_LSTACK(4,ISTACK)

      ISTACK=ISTACK-1
      
    END SUBROUTINE POP_XC_TYPE


!*********************** FUNCTION ISGGA  *******************************
!
! function that returns .TRUE. if a GGA functional has been selected
!
!***********************************************************************

    FUNCTION ISGGA()
      IMPLICIT NONE
      LOGICAL ISGGA
! Iann Gerber special case for splitting correlation functional LDA-only
      IF (LEXCH==10) THEN
         ISGGA=.FALSE.
      ELSE IF (LEXCH>3.AND.LEXCH<100) THEN
         ISGGA=.TRUE.
      ELSE
         ISGGA=.FALSE.
      ENDIF
    END FUNCTION ISGGA
      
!************************ FUNCTION ISLDAXC  ****************************
!
! function that returns .TRUE. if a local or semilocal xc functional is
! used
!
!***********************************************************************

    FUNCTION ISLDAXC()
      IMPLICIT NONE
      LOGICAL ISLDAXC
      IF (LEXCH>=0) THEN
         ISLDAXC=.TRUE.
      ELSE
         ISLDAXC=.FALSE.
      ENDIF
    END FUNCTION ISLDAXC

!************************ SUBROUTINE XCTABLE_CHECK *********************
!
! procedure that checks whether the exchange correlation table
! corresponds to the currently used exchange correlation type
! LEXCH
!
!***********************************************************************

    SUBROUTINE XCTABLE_CHECK
      IMPLICIT NONE
      
      IF (LEXCH /= LEXCH_TABLE) THEN
         WRITE(*,*) 'internal ERROR in XCTABLE_CHECK: the exchange table is not properly set up',LEXCH,LEXCH_TABLE
         STOP
      ENDIF
      
    END SUBROUTINE XCTABLE_CHECK


!*******************************************************************
!
!  calculate the exchange correlation potential
!  on a radial grid and the first 3 derivatives
!
!*******************************************************************

    SUBROUTINE EXCOR_DER_PARA(RHO, NDER, EXCA, TREL)
      USE constant
      IMPLICIT NONE

      LOGICAL TREL
      INTEGER NDER               ! number of derivatives
      REAL(q) RHO
      REAL(q) EXCA(4),EXCA_(4)
      REAL(q) RHOT,EXC0,EXC2,EXCD0,EXCD2,EPS
      ! this is the best compromise for densities between 1000 and 0.1
      REAL(q),PARAMETER :: DELTA=1E-3_q

      CALL EXCOR_PARA(RHO,EXCA(1),EXCA(2),TREL)

      IF (NDER>0) THEN
         EPS=DELTA*RHO

         RHOT=RHO-EPS
         CALL EXCOR_PARA(RHOT,EXC0,EXCD0,TREL)

         RHOT=RHO+EPS
         CALL EXCOR_PARA(RHOT,EXC2,EXCD2,TREL)
         ! 1st and 2nd derivative of energy
         EXCA_(2)=(EXC2-EXC0)/ (2*EPS)
         EXCA_(3)=(EXC2+EXC0-2*EXCA(1))/ (EPS*EPS)

         ! 2nd and 3nd derivative of potential=
         ! 1st and 2nd derivative of energy
         EXCA(3)=(EXCD2-EXCD0)/ (2*EPS)
         EXCA(4)=(EXCD2+EXCD0-2*EXCA(2))/ (EPS*EPS)
         ! WRITE(*,'(5E14.7)') EXCA(2),EXCA_(2),EXCA(3),EXCA_(3),EXCA(4)
      ENDIF

    END SUBROUTINE

    SUBROUTINE EXCOR_PARA(RHO,EXC,DEXC,TREL)
      USE constant
      USE xclib
      IMPLICIT NONE
      REAL(q) RHO,RS,EXC,DEXC
      REAL(q) ECLDA,ECDLDA,ECD2LDA,SK,T,EC,ECD,ECDD,ZETA
      LOGICAL TREL

      RS = ( (3._q/(4*PI)) / RHO)**(1/3._q) /AUTOA

      IF (LEXCH==-1) THEN
         EXC = 0
         DEXC= 0
      !
      ! LDA part: Pade approximation to Ceperly Alder results
      !
!aem AM05 functional (LEXCH==13) added, it uses the PW correlation in the LDA part.
! jP: adding PBEsol
!vdw jk
      ELSE IF (LEXCH==8.OR.LEXCH==9.OR.LEXCH==13 .OR. LEXCH==14 .or. LEXCH==17 .or. (LEXCH.ge.40 .and. LEXCH.le.50)) THEN
!vdw jk
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==20 CORPBE_LDA_RPA
      ELSE IF (LEXCH==20)  THEN
         ZETA=0
         CALL CORPBE_LDA_RPA(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==21 CORPBE_LDA_SR_0_15au
      ELSE IF (LEXCH==21)  THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_0_15au(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV 
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==22 CORPBE_LDA_SR_0_5A
      ELSE IF (LEXCH==22)  THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_0_5A(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==23 CORPBE_LDA_SR_1_0A
      ELSE IF (LEXCH==23)  THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_1_0A(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==24 CORPBE_LDA_SR_2_0A
      ELSE IF (LEXCH==24)  THEN
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_SR_2_0A(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)

         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
! jH-new LEXCH==30 CORPBE_LDA_RPA_PLUS
      ELSE IF (LEXCH==30)  THEN 
         ZETA=0  ! paramagnetic result
         CALL CORPBE_LDA_RPA_PLUS(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)            
         EXC = (EX_MOD(RS,1,.FALSE.)+ECLDA*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+ECDLDA*ALDAC)*RYTOEV
      !  
      ! LDA: range separated exchange + range separated correlation
      !   
      ELSEIF (LEXCH==10) THEN
!         EXC = (EX_MOD(RS,1,TREL)+EC_MOD(RS,1,TREL)*ALDAC)*RHO*RYTOEV
!         DEXC= (VX_MOD(RS,1,TREL)+VC_MOD(RS,1,TREL)*ALDAC)*RYTOEV      
! Relativistic correction have been removed  
! jP be careful ... 
! jP: what does EC_MOD do in that case - LEXCH==10  ...
         EXC = (EX_MOD(RS,1,.FALSE.)+EC_MOD(RS,0._q,1,.FALSE.)*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,.FALSE.)+VC_MOD(RS,0._q,1,.FALSE.)*ALDAC)*RYTOEV
      ELSEIF (LEXCH==11) THEN
! Joachim Paier
         EXC = (EX_MOD(RS,1,TREL)+ECVOIII(RS,1)*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,TREL)+VCVOIII(RS,1)*ALDAC)*RYTOEV
         ! ECLDA=ECVO(RS,1)
         ! ECDLDA=VCVO(RS,1)
         ! jP: compare with CALL CORPBE_LDA(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)
      ELSEIF (LEXCH==12) THEN
! Joachim Paier
         EXC = (EX_MOD(RS,1,TREL)+ECVO(RS,1)*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,TREL)+VCVO(RS,1)*ALDAC)*RYTOEV
         ! ECLDA=ECVO(RS,1)
         ! ECDLDA=VCVO(RS,1)
         ! jP: compare with CALL CORPBE_LDA(RS,ZETA,ECLDA,ECDLDA,ECD2LDA)
      ELSE IF (LEXCH==100) THEN
         EXC=0
         CALL COHSM1(RHO, DEXC)
      !
      ! LDA part: Perdew Zungers interpolation
      !
      ELSE
         EXC = (EX_MOD(RS,1,TREL)+ECCA(RS,1)*ALDAC)*RHO*RYTOEV
         DEXC= (VX_MOD(RS,1,TREL)+VCCA(RS,1)*ALDAC)*RYTOEV
      ENDIF

    END SUBROUTINE

!*******************************************************************
!
!  calculate the exchange correlation energy density
!  on a radial grid and the first 3 derivatives
!  for the spinpolarized case
!  as input the density (up and down) is required
!  derivatives with respect to up and down components are calculated
!
!*******************************************************************
      
    SUBROUTINE EXCOR_DER(RHOUP,RHODOWN,NDER,EXC,EXCD,EXCDD,EXCDDD,TREL)
      USE constant
      IMPLICIT NONE

      LOGICAL TREL
      INTEGER NDER               ! number of derivative
      REAL(q) RHOUP,RHODOWN      ! up and down component of density
      REAL(q) RHO(2),RHOIN(2)
      REAL(q) EXC,EXCD(2),EXCD_(2),EXCDD(2,2),EXCDD_(2,2),EXCDDD(2,2,2)
      REAL(q) TMP(-1:1,-1:1),T2(2,-1:1,-1:1)
      REAL(q) EPS(2)
      ! this is the best compromise for densities between 1000 and 0.1
      REAL(q),PARAMETER :: DELTA=1E-3_q
      INTEGER I,J

      EXCDD=0
      EXCDDD=0
      RHO(1)=RHOUP
      RHO(2)=RHODOWN
    ! function + derivative
      CALL EXCOR(RHO,EXC,EXCD,TREL)
      IF (NDER>0) THEN
        TMP(0,0) =EXC
        T2(:,0,0)=EXCD
        RHOIN=RHO
        ! calculate steps
        ! exc is approx rho^(4/3)
        ! the derivative thus rho^(1/3) and thus the derivative is
        ! of the order 1/rho
        EPS(1)=DELTA*MAX(RHO(1),RHO(2))
        EPS(2)=EPS(1)
        ! calculate all values on the 3x3 rectangle
        DO I=-1,1
          DO J=-1,1
             RHO(1)=RHOIN(1)+EPS(1)*I
             RHO(2)=RHOIN(2)+EPS(2)*J
             CALL EXCOR(RHO,TMP(I,J),T2(1,I,J),TREL)
          ENDDO
        ENDDO

    ! 1st and 2nd derivative of exchange correlation energy
    ! 1st derivative (EXCD_) is of course equal to EXCD
        EXCD_(1)   =(TMP(1,0)-TMP(-1,0)) / (2*EPS(1))
        EXCD_(2)   =(TMP(0,1)-TMP(0,-1)) / (2*EPS(2))
        EXCDD_(1,1)=(TMP(1,0)+TMP(-1,0)-2*TMP(0,0)) / (EPS(1)*EPS(1))
        EXCDD_(2,2)=(TMP(0,1)+TMP(0,-1)-2*TMP(0,0)) / (EPS(2)*EPS(2))
        EXCDD_(1,2)=((TMP(1,1)+TMP(-1,-1)-2*TMP(0,0))- &
                     (TMP(1,0)+TMP(-1,0)-2*TMP(0,0)) - &
                     (TMP(0,1)+TMP(0,-1)-2*TMP(0,0)))/(2*EPS(1)*EPS(2))
        EXCDD_(2,1)=EXCDD_(1,2)
    ! 1st derivative of potential = 2nd derivative of energy
        EXCDD (1,1)=(T2(1,1,0)-T2(1,-1,0)) / (2*EPS(1))
        EXCDD (2,2)=(T2(2,0,1)-T2(2,0,-1)) / (2*EPS(1))
        EXCDD (1,2)=(T2(1,0,1)-T2(1,0,-1)) / (2*EPS(1))
        EXCDD (2,1)=(T2(2,1,0)-T2(2,-1,0)) / (2*EPS(1))
    ! 2nd derivative of potential = 3rd derivative of energy
        EXCDDD(1,1,1)=(T2(1,1,0)+T2(1,-1,0)-2*T2(1,0,0)) / (EPS(1)*EPS(1))
        EXCDDD(1,2,2)=(T2(1,0,1)+T2(1,0,-1)-2*T2(1,0,0)) / (EPS(2)*EPS(2))
        EXCDDD(1,1,2)=((T2(1,1,1)+T2(1,-1,-1)-2*T2(1,0,0))- &
                       (T2(1,1,0)+T2(1,-1,0)-2*T2(1,0,0))- &
                       (T2(1,0,1)+T2(1,0,-1)-2*T2(1,0,0)))/(2*EPS(1)*EPS(2))
        EXCDDD(1,2,1)=EXCDDD(1,1,2)
        EXCDDD(2,1,1)=(T2(2,1,0)+T2(2,-1,0)-2*T2(2,0,0)) / (EPS(1)*EPS(1))
        EXCDDD(2,2,2)=(T2(2,0,1)+T2(2,0,-1)-2*T2(2,0,0)) / (EPS(2)*EPS(2))
        EXCDDD(2,1,2)=((T2(2,1,1)+T2(2,-1,-1)-2*T2(2,0,0)) - &
                       (T2(2,1,0)+T2(2,-1,0)-2*T2(2,0,0)) - &
                       (T2(2,0,1)+T2(2,0,-1)-2*T2(2,0,0)))/(2*EPS(1)*EPS(2))
        EXCDDD(2,2,1)=EXCDDD(2,1,2)
    !    WRITE(*,*) EXCDDD(2,1,2),EXCDDD(1,2,2),EXCDDD(2,2,2)
    !    WRITE(*,*) EXCDDD(1,1,2),EXCDDD(2,1,1),EXCDDD(1,1,1)
    ! plenty of cross checks are possible here
    !    WRITE(*,'(4E14.7)') EXCD,EXCD_
    !    WRITE(*,'(4E14.7)') EXCDD,EXCDD_
    !    WRITE(*,'(4E14.7)') EXCDDD
      ENDIF

    END SUBROUTINE EXCOR_DER


    SUBROUTINE EXCOR(RHO,EXC,EXCD,TREL)
      USE prec
      USE constant
      USE xclib

      IMPLICIT NONE
      LOGICAL TREL
      REAL(q) RHO(2),EXC,EXCD(2)
    ! local variables
      REAL(q) RS,ZETA,FZ,DFZ,EXP,EC,ECP,VXP,VCP,EXT,VX0,DVX,VX1,VX2,RH, &
              ECF,VXF,VCF,EXF,ALP,VALP,ZETA3,ZETA4

      RH=RHO(1)+RHO(2)
      RS = ( (3._q/(4*PI)) / RH)**(1/3._q) /AUTOA
      ZETA=(RHO(1)-RHO(2))/(RHO(1)+RHO(2))
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
      ZETA3=(ZETA*ZETA)*ZETA
      ZETA4=(ZETA*ZETA)*(ZETA*ZETA)

      IF (LEXCH==-1) THEN
         EXC=0
         EXCD=0
      !
      ! LDA part: Pade approximation to Ceperley Alder results
      !
!aem AM05 functional (LEXCH==13) added, it uses the PW correlation in the LDA part.
! jP: adding PBEsol
!vdw jk
      ELSE IF (LEXCH==8 .OR. LEXCH==9 .OR. LEXCH==13 .OR. LEXCH==14 .or. LEXCH==17 .or. (LEXCH.ge.40 .and. LEXCH.le.50)) THEN
!vdw jk
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,.FALSE.)*RH*RYTOEV ; EXF=EX_MOD(RS,2,.FALSE.)*RH*RYTOEV
         VXP=VX_MOD(RS,1,.FALSE.)*RYTOEV    ; VXF=VX_MOD(RS,2,.FALSE.)*RYTOEV

         CALL CORPBE_LDA(RS,ZETA,EC,VCP,VCF)
         EC=EC*ALDAC*RYTOEV*RH ; VCP=VCP*ALDAC*RYTOEV ; VCF=VCF*ALDAC*RYTOEV
         
         VX0=VXP +(VXF-VXP)*FZ
         DVX=(EXF-EXP)*DFZ/RH
         EXC= EXP+(EXF-EXP)*FZ+EC

         EXCD(1)= VX0-DVX*(ZETA-1)+VCP
         EXCD(2)= VX0-DVX*(ZETA+1)+VCF
      !
      ! LDA: short range exchange + short range correlation
      !   
! jH-new noch fuer RPA und RPA-plus 
      ELSEIF (LEXCH==20) THEN
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,.FALSE.)*RH*RYTOEV ; EXF=EX_MOD(RS,2,.FALSE.)*RH*RYTOEV
         VXP=VX_MOD(RS,1,.FALSE.)*RYTOEV    ; VXF=VX_MOD(RS,2,.FALSE.)*RYTOEV

         CALL CORPBE_LDA_RPA(RS,ZETA,EC,VCP,VCF)
         EC=EC*ALDAC*RYTOEV*RH ; VCP=VCP*ALDAC*RYTOEV ; VCF=VCF*ALDAC*RYTOEV
         
         VX0=VXP +(VXF-VXP)*FZ
         DVX=(EXF-EXP)*DFZ/RH
         EXC= EXP+(EXF-EXP)*FZ+EC

         EXCD(1)= VX0-DVX*(ZETA-1)+VCP
         EXCD(2)= VX0-DVX*(ZETA+1)+VCF
      ELSEIF (LEXCH==30) THEN
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,.FALSE.)*RH*RYTOEV ; EXF=EX_MOD(RS,2,.FALSE.)*RH*RYTOEV
         VXP=VX_MOD(RS,1,.FALSE.)*RYTOEV    ; VXF=VX_MOD(RS,2,.FALSE.)*RYTOEV

         CALL CORPBE_LDA_RPA_PLUS(RS,ZETA,EC,VCP,VCF)
         EC=EC*ALDAC*RYTOEV*RH ; VCP=VCP*ALDAC*RYTOEV ; VCF=VCF*ALDAC*RYTOEV

         VX0=VXP +(VXF-VXP)*FZ
         DVX=(EXF-EXP)*DFZ/RH
         EXC= EXP+(EXF-EXP)*FZ+EC

         EXCD(1)= VX0-DVX*(ZETA-1)+VCP
         EXCD(2)= VX0-DVX*(ZETA+1)+VCF


      ELSEIF (LEXCH==10) THEN
! jP used interpolation: PBE (LEXCH==8)
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,.FALSE.)*RH*RYTOEV ; EXF=EX_MOD(RS,2,.FALSE.)*RH*RYTOEV
         VXP=VX_MOD(RS,1,.FALSE.)*RYTOEV    ; VXF=VX_MOD(RS,2,.FALSE.)*RYTOEV

         EC= EC_MOD(RS,ZETA,2,TREL)*ALDAC*RYTOEV*RH 
         VCP=VC_MOD(RS,0._q,1,TREL)*ALDAC*RYTOEV 
         VCF=VC_MOD(RS,1._q,2,TREL)*ALDAC*RYTOEV
         
         VX0=VXP +(VXF-VXP)*FZ
         DVX=(EXF-EXP)*DFZ/RH
         EXC= EXP+(EXF-EXP)*FZ+EC

         EXCD(1)= VX0-DVX*(ZETA-1)+VCP
         EXCD(2)= VX0-DVX*(ZETA+1)+VCF

      ELSE IF (LEXCH==100) THEN
         EXC=0
         CALL COHSM1(RH, EXCD(1))
         EXCD(2)=EXCD(1)
      !
      ! LDA part: Perdew Zungers interpolation
      !
      ELSEIF (LEXCH==11) THEN
! Joachim Paier
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,TREL)*RH*RYTOEV ; EXF=EX_MOD(RS,2,TREL)*RH*RYTOEV
         ECP=ECVOIII(RS,1)*RH*RYTOEV     ; ECF=ECVOIII(RS,2)*RH*RYTOEV
!         ECP=ECVO(RS,1)*RH*RYTOEV     ; ECF=ECVO(RS,2)*RH*RYTOEV
         VXP=VX_MOD(RS,1,TREL)*RYTOEV    ; VXF=VX_MOD(RS,2,TREL)*RYTOEV
         VCP=VCVOIII(RS,1)*RYTOEV        ; VCF=VCVOIII(RS,2)*RYTOEV
!         VCP=VCVO(RS,1)*RYTOEV        ; VCF=VCVO(RS,2)*RYTOEV

         ALP =ALPHA0_III(RS)*RH*RYTOEV
!         ALP =ALPHA0(RS)*RH*RYTOEV
         VALP=ALPHA1_III(RS)*RYTOEV
!         VALP=ALPHA1(RS)*RYTOEV

         VX0=VXP+(VXF-VXP)*FZ+  (VCP+(VALP+(VCF-VCP-VALP)*ZETA4)*FZ)*ALDAC
         DVX=((EXF-EXP)*DFZ  +(ALP+(ECF-ECP-ALP)*ZETA4)*DFZ*ALDAC+ &
              4*(ECF-ECP-ALP)*ZETA3 *FZ *ALDAC )/RH
        ! the more usual expression for this is
        ! ECP*( 1 - FZ Z4) +EP*FZ*Z4-ALP*F*(1._q-Z4)/(8/(9 gamma))
         EXC= EXP+(EXF-EXP)*FZ +ECP*ALDAC+(ALP+(ECF-ECP-ALP)*ZETA4)*FZ*ALDAC

         EXCD(1)= VX0-DVX*(ZETA-1)
         EXCD(2)= VX0-DVX*(ZETA+1)
      ELSEIF (LEXCH==12) THEN
! Joachim Paier
         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,TREL)*RH*RYTOEV ; EXF=EX_MOD(RS,2,TREL)*RH*RYTOEV
         ECP=ECVO(RS,1)*RH*RYTOEV        ; ECF=ECVO(RS,2)*RH*RYTOEV
         VXP=VX_MOD(RS,1,TREL)*RYTOEV    ; VXF=VX_MOD(RS,2,TREL)*RYTOEV
         VCP=VCVO(RS,1)*RYTOEV           ; VCF=VCVO(RS,2)*RYTOEV

         ALP =ALPHA0(RS)*RH*RYTOEV
         VALP=ALPHA1(RS)*RYTOEV

         VX0=VXP+(VXF-VXP)*FZ+  (VCP+(VALP+(VCF-VCP-VALP)*ZETA4)*FZ)*ALDAC
         DVX=((EXF-EXP)*DFZ  +(ALP+(ECF-ECP-ALP)*ZETA4)*DFZ*ALDAC+ &
              4*(ECF-ECP-ALP)*ZETA3 *FZ *ALDAC )/RH
        ! the more usual expression for this is
        ! ECP*( 1 - FZ Z4) +EP*FZ*Z4-ALP*F*(1._q-Z4)/(8/(9 gamma))
         EXC= EXP+(EXF-EXP)*FZ +ECP*ALDAC+(ALP+(ECF-ECP-ALP)*ZETA4)*FZ*ALDAC

         EXCD(1)= VX0-DVX*(ZETA-1)
         EXCD(2)= VX0-DVX*(ZETA+1)
      ELSE

         FZ =FZ0(ZETA)          ! interpolation function for exchange from pm to fm
         DFZ=FZ1(ZETA)*SIGN(1._q,ZETA)
         EXP=EX_MOD(RS,1,TREL)*RH*RYTOEV ; EXF=EX_MOD(RS,2,TREL)*RH*RYTOEV
         ECP=ECCA(RS,1)*RH*RYTOEV        ; ECF=ECCA(RS,2)*RH*RYTOEV
         VXP=VX_MOD(RS,1,TREL)*RYTOEV    ; VXF=VX_MOD(RS,2,TREL)*RYTOEV
         VCP=VCCA(RS,1)*RYTOEV           ; VCF=VCCA(RS,2)*RYTOEV

         IF (LFCI==1) THEN
            ALP =ALPHA0(RS)*RH*RYTOEV
            VALP=ALPHA1(RS)*RYTOEV
            
            VX0=VXP+(VXF-VXP)*FZ+  (VCP+(VALP+(VCF-VCP-VALP)*ZETA4)*FZ)*ALDAC
            DVX=((EXF-EXP)*DFZ  +(ALP+(ECF-ECP-ALP)*ZETA4)*DFZ*ALDAC+ &
                              4*(ECF-ECP-ALP)*ZETA3 *FZ *ALDAC )/RH
        ! the more usual expression for this is
        ! ECP*( 1 - FZ Z4) +EP*FZ*Z4-ALP*F*(1._q-Z4)/(8/(9 gamma))
            EXC= EXP+(EXF-EXP)*FZ +ECP*ALDAC+(ALP+(ECF-ECP-ALP)*ZETA4)*FZ*ALDAC
         ELSE
            VX0= VXP+(VXF-VXP)*FZ +VCP*ALDAC+(VCF-VCP)*FZ*ALDAC
            DVX=((EXF-EXP)*DFZ    +(ECF-ECP)*DFZ)/RH*ALDAC
            EXC= EXP+(EXF-EXP)*FZ +ECP*ALDAC+(ECF-ECP)*FZ*ALDAC
         ENDIF

         EXCD(1)= VX0-DVX*(ZETA-1)
         EXCD(2)= VX0-DVX*(ZETA+1)
      ENDIF

    END SUBROUTINE EXCOR

  END MODULE


