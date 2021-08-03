!***********************************************************************
! RCS:  $Id: vdwforcefield.F,v 2.0 2013/09/10 tomas bucko$
!
!
!
!*********************************************************************

#include "symbol.inc"


MODULE vdwforcefield
  USE prec
  USE mymath
  USE base
  USE poscar
  USE lattice
  USE constant
  USE vaspxml
  USE main_mpi
  USE mpimy
  USE mgrid
  USE vdwD3
  USE mkpoints
  !USE symmetry
  USE radial
  USE c2f_interface, ONLY : ERRF

  IMPLICIT NONE 

  REAL(q), PARAMETER, PRIVATE :: rholimit=1e-10
  REAL(q), PARAMETER, PRIVATE :: rholow=2.0e-3/AUTOA3
  !REAL(q), PARAMETER, PRIVATE :: Rmax=7._q

    ! Lennard-Jones specific properties
  REAL(q), PARAMETER :: &
    LJ_RADIUS_DEFAULT = 15.0_q, &
    LJ_EPSILON_DEFAULT = 0.0_q, &
    LJ_SIGMA_DEFAULT = 3.0_q
  LOGICAL, PARAMETER :: &
    LJ_ONLY_DEFAULT = .FALSE.

  REAL(q), SAVE :: &
    LJ_RADIUS = LJ_RADIUS_DEFAULT, &
    LJ_EPSILON = LJ_EPSILON_DEFAULT, &
    LJ_SIGMA = LJ_SIGMA_DEFAULT
  LOGICAL, SAVE :: &
    LJ_ONLY = LJ_ONLY_DEFAULT

  CONTAINS

       FUNCTION LJ_IS_ACTIVE()
         LOGICAL :: LJ_IS_ACTIVE
         LJ_IS_ACTIVE = (LJ_EPSILON > 0.0_q)
       END FUNCTION

       subroutine TTDAMP(x,n,f2n)
       implicit none
       real(q) :: x,xk,sum2n,fac,f2n
       integer :: n,k
!!initialize for k=0
       fac=1.0_q
       sum2n=1.0_q
       xk=1.0_q
       do k=1,n
       xk=xk*x
       fac=fac*real(k,8)
       sum2n=sum2n+xk/fac
       end do
       f2n = 1._q - dexp(-x)*sum2n;
       return
       end subroutine TTDAMP

!
!  compute a piece of the Tang and Toennies universal damping function derivative
!            = { exp(-b*R) * SUM_k=0,n-1((b^(k+1)*R^(k-1)/k!) }
      subroutine TTDAMPD1a(b,R,n,d1f2n)
      implicit none
      real(q) :: b,R,bk,Rk,sum2n,fac,d1f2n
      integer :: n,k
!initialize for k=0
      fac=1._q
      sum2n=b/R
      bk=b
      Rk=1._q/R
      do k=1,n-1
      bk=bk*b
      Rk=Rk*R
      fac=fac*real(k,q)
      sum2n=sum2n+bk*Rk/fac
      end do
      d1f2n = dexp(-b*R)*sum2n;
      return
      end subroutine TTDAMPD1a

       
        FUNCTION damping_fermi(d,sR,r0,r) RESULT(damp)
        !c fermi-type damping function
          REAL(q),INTENT(in) :: d,sR,r0,r
          REAL(q) :: damp
          damp=1./(1.+EXP(-d*(r/(sR*r0)-1.)))
        END FUNCTION damping_fermi

        FUNCTION damping_fermi_deriv(d,sR,r0,r) RESULT(damp)
        !c derivative of fermi-type damping function
          REAL(q),INTENT(in) :: d,sR,r0,r
          REAL(q) :: damp
          !damp=d/(sR*r0)*EXP(-d*(r/(sR*r0)-1.))/(1.+EXP(-d*(r/(sR*r0)-1.)))**2
  
          damp=EXP(-d*(r/(sR*r0)-1.))
          damp=d/(sR*r0)*damp/((1+damp)*(1+damp))
        END FUNCTION damping_fermi_deriv

        FUNCTION vdw_e(c,d,r0,r) RESULT(ener)
        !c simple pair potential
          REAL(q) :: ener, c,d,r0,r
          ener=-(c/r**6)/(1.+EXP(-d*(r/r0-1.)))
        END FUNCTION vdw_e

        FUNCTION vdw_e_damptt(c6,r,b) RESULT(ener)
        !c Tang and Toennies damping function
          REAL(q) :: ener,c6,r,x,Rpow,f2n,b
          x=b*R
          Rpow=R**(-6)
!          write(*,*) 'c6,b,R',c6,b,R
          call TTDAMP(x,6,f2n)
          ener=-f2n*C6*Rpow
!          write(*,*)'ener',ener
        END FUNCTION vdw_e_damptt

        FUNCTION vdw_e_damp2(c,r,mu) RESULT(ener)
        !c simple pair potential, damping function 
        !c from ACFDT for two-body interactions
          REAL(q) :: ener, c, r,mu,rmu,f
          REAL(q) :: damp1, damp2, damp3

          rmu=r*mu
          damp1= 4._q*EXP(-2.*rmu**2)*rmu**2*(3.+4.*rmu**2+2.*rmu**4)/3./PI
          damp2=-4.*EXP(-rmu**2)*rmu*(3.+2.*rmu**2.)*ERRF(rmu)/3./(PI)**0.5
          damp3= ERRF(rmu)**2
          f=damp1+damp2+damp3
          ener=-(c/r**6)*f
        END FUNCTION vdw_e_damp2

        FUNCTION vdw_q(c,d,r0,r) RESULT(grad)
        !c vdW forces, Fermi damping 
          REAL(q) :: grad,f,c,d,r0,r
          f=1./(1.+exp(-d*(r/r0-1.)))
          grad=6*c/r**7*f-c/r**6*(f**2)*d/r0*exp(-d*(r/r0-1.))
        END FUNCTION vdw_q

        FUNCTION vdw_q_damp2(c,r,mu) RESULT(grad)
        !c vdw forces, damping function 
        !c from ACFDT for two-body interactions
          REAL(q) :: grad,f,c,r,mu,rmu
          REAL(q) :: df

          rmu=r*mu
          f= 4._q*EXP(-2.*rmu**2)*rmu**2*(3.+4.*rmu**2+2.*rmu**4)/3./PI
          f=f-4.*EXP(-rmu**2)*rmu*(3.+2.*rmu**2.)*ERRF(rmu)/3./(PI)**0.5
          f=f+ERRF(rmu)**2
         
          df=16./3./SQRT(PI)*EXP(-rmu**2)*mu*rmu**4*ERRF(rmu)
          df=df-16./3./PI*EXP(-2*rmu**2)*mu*(rmu**5+2*rmu**7)
          
          grad=6*c/r**7*f-c/r**6*df
        END FUNCTION vdw_q_damp2

        FUNCTION vdw_e_ulg(c,d,r0,r) RESULT(ener)
        !c addition by Sebastien Lebegue: energy for ULG
          REAL(q) :: ener, c,d,r0,r
          ener=-(c/(r**6+d*r0**6))
        END FUNCTION vdw_e_ulg

        FUNCTION vdw_q_ulg(c,d,r0,r) RESULT(grad)
        !c addition by Sebastien Lebegue: forces for ULG
          REAL(q) :: grad,f,c,d,r0,r
          grad= c*6.0*r**5/((r**6+d*r0**6)**2)
        END FUNCTION vdw_q_ulg

!        FUNCTION vdw_q_damptt(c6,r,b) RESULT(grad)
!        !c vdW forces, Tang and Toennies damping 
!! Gradient of the Tang and Toennies dispersion function for a fixed b, in direction of R
!        REAL(q) :: grad,c6,r,b,x,Rpow,f2n,d1f2n,tmp
!          Rpow=R**(-6)
!          call TTDAMP(x,6,f2n)
!          call TTDAMPD1a(b,R,6,d1f2n)
!          tmp=(1._q-f2n)*b/R-d1f2n
!          !grad=-(C6*Rpow)*(tmp-6.0_q*f2n/R/R)
!          grad=(C6*Rpow)*(tmp-6.0_q*f2n/R/R)
!        END FUNCTION vdw_q_damptt

        FUNCTION vdw_q_damptt(c6,r,b) RESULT(grad)
        !c vdW forces, Tang and Toennies damping 
! Gradient of the Tang and Toennies dispersion function for a fixed b, in direction of R
        REAL(q) :: grad,c6,r,b,x,Rpow,f2n,d1f2n,tmp
          x=b*R
          Rpow=R**(-6)
          call TTDAMP(x,6,f2n)
          call TTDAMPD1a(b,R,6,d1f2n)
          !tmp=(1._q-f2n)*b/R-d1f2n
          !grad=-(C6*Rpow)*(tmp-6.0_q*f2n/R/R)
          tmp=(1._q-f2n)*b-d1f2n*R
          grad=-(C6*Rpow)*(tmp-6.0_q*f2n/R)
!          write(*,*) 'grad',(1._q-f2n)*b,-d1f2n/R
          !grad=-(C6*Rpow)*(tmp-6.0_q*f2n/R/R)
        END FUNCTION vdw_q_damptt



        FUNCTION cwhereisit(a,x,n)
          INTEGER :: cwhereisit,n,i
          CHARACTER (LEN=2) :: a(n)
          CHARACTER (LEN=2) :: x
          cwhereisit=0
          DO i=1,n
            IF (x==a(i)) THEN 
              cwhereisit=i
              EXIT
            END IF 
          END DO
        END FUNCTION cwhereisit

        FUNCTION combination_ruleTS(cA,cB,alphaA,alphaB) RESULT(cAB)
        !c combination rule for C6 coeficients
          REAL(q) :: cA,cB,cAB
          REAL(q) :: alphaA,alphaB
          
          IF (cA>0._q .AND. cB>0._q) THEN 
            cAB=2*(cA*cB)/(alphaB/alphaA*cA+alphaA/alphaB*cB)
          ELSE 
            cAB=0._q
          ENDIF
        END FUNCTION combination_ruleTS

        SUBROUTINE gradient_C(nions,cA,cB,dcAB,alphaA,alphaB,dcA,dcB,dalphaA,dalphaB)
        !c derivative of C6 wrt. atomic displacement
          INTEGER :: nions
          REAL(q) :: cA,cB,cAB
          REAL(q) :: alphaA,alphaB
          REAL(q) :: dcAB(3,nions+3),dcA(3,nions+3),dcB(3,nions+3)
          REAL(q) :: dalphaA(3,nions+3),dalphaB(3,nions+3)
    
          dcAB=0._q
!           cAB=2*(cA*cB)/(alphaB/alphaA*cA+alphaA/alphaB*cB)
!           dcAB=cAB*(dcA/cA+dcB/cB)-2*cA*cB/(alphaB/alphaA*cA+alphaA/alphaB*cB)**2 &
!                          & *(dalphaB*cA/alphaA + dalphaA*cB/alphaB + &
!                          & alphaB/alphaA*dcA + alphaA/alphaB*dcB - &
!                          & alphaB/alphaA**2*cA*dalphaA-alphaA/alphaB**2*cB*dalphaB)
          dcAB=2._q*(cB*cB*alphaA/alphaB*dcA +cA*cA*alphaB/alphaA*dcB)
          dcAB=dcAB-2._q*cA*cB*((cB/alphaB-cA*alphaB/alphaA/alphaA)*dalphaA + &
                               &(cA/alphaA-cB*alphaA/alphaB/alphaB)*dalphaB     )
          dcAB=dcAB/(alphaB/alphaA*cA + alphaA/alphaB*cB)**2

        END SUBROUTINE gradient_C

        SUBROUTINE gradient_RA(nions,drrrA,rAts,alphaAts,alphaA,dalphaA,LSCALER0) 
        !c derivative of R0 for atom A wrt. atomic displacement
          INTEGER :: nions
          REAL(q) :: rAts,alphaAts,alphaA
          REAL(q) :: drrr(3,nions+3),drrrA(3,nions+3)
          REAL(q) :: dalphaA(3,nions+3)
          LOGICAL :: LSCALER0
    
          drrrA=0._q
          IF (LSCALER0) THEN
            drrrA=rAts/(alphaAts*alphaA**2)**(1._q/3._q)*dalphaA/3._q
          ENDIF
        END SUBROUTINE gradient_RA

        SUBROUTINE gradient_Escs(NIONS,dcAB,r,r0,d,grad,cAB,drrr)
          INTEGER :: NIONS
          REAL(q) :: dcAB(3,NIONS+3),grad(3,NIONS+3),drrr(3,NIONS+3)
          REAL(q) :: f,d,r0,r,cAB,sr
          
          grad=0._q
          f=1./(1.+exp(-d*(r/r0-1.)))
          !grad=6*dcAB/r**7*f
          grad=f*dcAB/r**6
          !grad=grad+cAB/r**6*f**2*exp(-d*(r/r0-1.))*d*r/r0**2*drrr
          !grad=grad-cAB/r**6*f**2*exp(-d*(r/r0-1.))*d*r*sr/r0**2*drrr
          grad=grad-cAB/r**6*f**2*exp(-d*(r/r0-1.))*d*r/r0**2*drrr
          !!!grad=0.5*f*dcAB/r**6
        END SUBROUTINE gradient_Escs

        SUBROUTINE vdw_read(IO,LVDW,IVDW,LRELVOL,NTYP,REFSTATE)
          !USE fileio
          TYPE (in_struct) :: IO
          LOGICAL :: LVDW,LRELVOL
          INTEGER :: IVDW
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          INTEGER :: NTYP
          INTEGER :: REFSTATE(NTYP)

          REFSTATE=0

          LOPEN=.FALSE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

            CALL RDATAB(LOPEN,INCAR,IO%IU5,'IVDW','=','#',';','I', &
                  IVDW,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) IVDW=0 
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
               IVDW=0 
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''IVDW'' from file INCAR'
            ENDIF


            !IF (IVDW==1 .OR. IVDW==2 .OR. IVDW==22 .OR. IVDW==3 .OR. IVDW==4 .OR. IVDW==5 .OR. IVDW==33 .OR. (IVDW>=11 .AND. IVDW<=14)) THEN 
            IF (IVDW==1 .OR. IVDW==2 .OR. IVDW==3 .OR. IVDW==4 .OR. (IVDW>=10 .AND. IVDW<=12) .OR.  & 
            &   IVDW==101 .OR. (IVDW>=20 .AND. IVDW<=22) .OR. IVDW==200 .OR. IVDW==210 .OR. IVDW==202 &
            &   .OR. IVDW==212 .OR. IVDW==612  ) THEN
              LVDW=.TRUE.
            ELSE
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW,CHARAC,N,1,IERR)
              IF (IERR==3) LVDW=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 LVDW=.FALSE.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDW'' from file INCAR'
              ENDIF
              IF (LVDW) IVDW=1
            ENDIF

            IF (LVDW) THEN
               !c compute relvol every step?
               CALL RDATAB(LOPEN,INCAR,IO%IU5,'LRELVOL','=','#',';','L', &
                  IDUM,RDUM,CDUM,LRELVOL,CHARAC,N,1,IERR)
              IF (IERR==3) LRELVOL=.TRUE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 LRELVOL=.TRUE.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LRELVOL'' from file INCAR'
              ENDIF
      
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_REFSTATE','=','#',';','I', &
              &   REFSTATE,RDUM,CDUM,LDUM,CHARAC,N,NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_REFSTATE'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ENDIF
            ENDIF 


            !CALL XML_TAG("parameters_embedded")
            !CALL XML_TAG("separator","dft+d2")
            !CALL XML_INCAR('LVDW','L',IDUM,RDUM,CDUM,LVDW,CHARAC,1)
  
            CLOSE(IO%IU5)

        END SUBROUTINE vdw_read

        SUBROUTINE vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM, &
        &          RELVOL,IVDW,REFSTATE,Overlap,M2,RELCHG,HCD,KPOINTS)
          !USE fileio
          TYPE (grid_3d) GRIDC
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: TOTEN
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER,SAVE :: historycounter=0
          LOGICAL,SAVE :: LVDW=.FALSE.
          INTEGER :: IVDW
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q) :: RELVOL(T_INFO%NIONS)
          INTEGER :: REFSTATE(T_INFO%NTYP)
          TYPE (kpoints_struct), OPTIONAL :: KPOINTS
          INTEGER :: i
          REAL(q) :: RELCHG(:),Overlap(:,:),M2(:),HCD(:)

          historycounter=historycounter+1
 
          SELECT CASE(IVDW)
            ! D2
            CASE(1,10)
              CALL vdw_forces_G(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,IVDW)

            ! D2 by Jonas Moellman - no guarantee for compatibility with IVDW=1, for test purposes only!!!
            CASE(101)
              CALL vdw_forces_D3(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,2,IVDW)

            !c D3(zero-damping) by Jonas Moellman
            CASE(11)
              CALL vdw_forces_D3(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,3,IVDW)

            !c D3(BJ-damping) by Jonas Moellman
            CASE(12)            
              CALL vdw_forces_D3(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,4,IVDW)

            ! TS methods: 2,20 - standard TS, 21 - TS/HI, 22 - TS/HI-FO
            ! CASE(2,3,4,5)
            CASE(2,20,21,22,23,24,25,26)
              CALL vdw_forces_TS(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,RELVOL,REFSTATE,IVDW)
            !CASE(3)
            !  CALL vdw_forces_TS(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,RELVOL,REFSTATE)
            !CASE(4)
            !  CALL vdw_forces_TS(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,RELVOL,REFSTATE)
            !CASE(5)
            !  CALL vdw_forces_TS(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,RELVOL,REFSTATE)
            CASE(202,212)
            !  IF (PRESENT(KPOINTS)) THEN
!                CALL vdw_forces_MBD(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter, &
!                &  RELVOL,REFSTATE,IVDW,SYMM,KPOINTS)
               CALL vdw_forces_MBD(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter, &
               &  RELVOL,REFSTATE,IVDW,KPOINTS)
             ! ELSE
             !   CALL vdw_forces_MBD(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter, &
             !   &  RELVOL,REFSTATE,IVDW)
             ! ENDIF
           ! ULG method by Sebastien Lebegue
            CASE(3)
              CALL vdw_forces_ulg(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,IVDW)
           
            ! dDsC method by Stephan Steinmann
            CASE(4)
              CALL vdw_forces_dDsC(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,RELVOL,IVDW,&
              & Overlap,M2,RELCHG,HCD)

            ! not really DFT+vdW 
            CASE(612)
              CALL vdw_forces_LJ(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,historycounter)
        
            
            CASE DEFAULT
              RETURN
          END SELECT
        END SUBROUTINE vdw_forces_main

       SUBROUTINE vdw_pairContrib(i,j,atrad,crit,r12_,ccc,rrr,dfactor,T_INFO,&
        & LATT_CUR,counter,energy,vdw_f2,virialstress)
        !c computes contributions for all relevant translations of atomic pair i,j 
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,ii, jj, kk

          DO ii=-crit(1),crit(1)
            DO jj=-crit(2),crit(2)
              DO kk=-crit(3),crit(3)
                 r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
                 r12norm=SUM(r12**2)**0.5
                 IF (r12norm>0.5 .AND. r12norm<atrad) THEN
                   counter=counter+1
                   if (i==j) then
                     energy=energy+0.5*vdw_e(ccc,dfactor,rrr,r12norm)
                     gradients=0.5*vdw_q(ccc,dfactor,rrr,r12norm)
                   else
                     energy=energy+vdw_e(ccc,dfactor,rrr,r12norm)
                     gradients=vdw_q(ccc,dfactor,rrr,r12norm)
                   endif
                   virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                   IF (i /= j) THEN
                     vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                     vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
                   ENDIF
                 END IF
               END DO
             END DO
           END DO
        END SUBROUTINE vdw_pairContrib

        SUBROUTINE vdw_pairContrib_dampTT(i,j,atrad,crit,r12_,ccc,bij,bij_asym,T_INFO,&
        & LATT_CUR,counter,energy,vdw_f2,virialstress)
        !c computes contributions for all relevant translations of atomic pair i,j 
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad,btt,bij,bij_asym
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,ii, jj, kk

          DO ii=-crit(1),crit(1)
            DO jj=-crit(2),crit(2)
              DO kk=-crit(3),crit(3)
                 !write(*,*) ii,jj,kk
                 if(ii.eq.0 .and. jj.eq.0 .and. kk.eq.0) then
                   btt=bij
                   !write(*,*) 'Here is THE pair'
                 else
                   btt=bij_asym
                 endif
                 r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
                 r12norm=SUM(r12**2)**0.5
                 IF (r12norm>0.5 .AND. r12norm<atrad) THEN
                   counter=counter+1
                   if (i==j) then
                     energy=energy+0.5*vdw_e_damptt(ccc,r12norm,btt)
                     gradients=0.5*vdw_q_damptt(ccc,r12norm,btt)
                   else
                     energy=energy+vdw_e_damptt(ccc,r12norm,btt)
                     !write(*,*) 'energy',energy
                     gradients=vdw_q_damptt(ccc,r12norm,btt)
                   endif
                   virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                   IF (i /= j) THEN
                     vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                     vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
                   ENDIF
                 END IF
               END DO
             END DO
           END DO
        END SUBROUTINE vdw_pairContrib_dampTT

        SUBROUTINE vdw_pairContrib_damp2(i,j,atrad,crit,r12_,ccc,mu,T_INFO,&
        & LATT_CUR,counter,energy,vdw_f2,virialstress)
        !c computes contributions for all relevant translations of atomic pair i,j 
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad,mu
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,ii, jj, kk

          DO ii=-crit(1),crit(1)
            DO jj=-crit(2),crit(2)
              DO kk=-crit(3),crit(3)
                 r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
                 r12norm=SUM(r12**2)**0.5
                 IF (r12norm>0.5 .AND. r12norm<atrad) THEN
                   counter=counter+1
                   if (i==j) then
                     energy=energy+0.5*vdw_e_damp2(ccc,r12norm,mu)
                     gradients=0.5*vdw_q_damp2(ccc,r12norm,mu)
                   else
                     energy=energy+vdw_e_damp2(ccc,r12norm,mu)
                     gradients=vdw_q_damp2(ccc,r12norm,mu)
                   endif
                   virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                   IF (i /= j) THEN
                     vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                     vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
                   ENDIF
                 END IF
               END DO
             END DO
           END DO
        END SUBROUTINE vdw_pairContrib_damp2

        SUBROUTINE vdw_pairContrib_ulg(i,j,atrad,crit,r12_,ccc,rrr,dfactor,T_INFO,&
        & LATT_CUR,counter,energy,vdw_f2,virialstress)
        !c addition by Sebastien Lebegue:
        !c computes contributions for all relevant translations of atomic pair i,j
        !c for ULG 
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,ii, jj, kk

          DO ii=-crit(1),crit(1)
            DO jj=-crit(2),crit(2)
              DO kk=-crit(3),crit(3)
                 r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
                 r12norm=SUM(r12**2)**0.5
                 IF (r12norm>0.5 .AND. r12norm<atrad) THEN
                   counter=counter+1
                   if (i==j) then
                     energy=energy+0.5*vdw_e_ulg(ccc,dfactor,rrr,r12norm)
                     gradients=0.5*vdw_q_ulg(ccc,dfactor,rrr,r12norm)
                   else
                     energy=energy+vdw_e_ulg(ccc,dfactor,rrr,r12norm)
                     gradients=vdw_q_ulg(ccc,dfactor,rrr,r12norm)
                   endif
                   virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                   IF (i /= j) THEN
                     vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                     vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
                   ENDIF
                 END IF
               END DO
             END DO
           END DO
        END SUBROUTINE vdw_pairContrib_ulg

        SUBROUTINE vdw_pairContribSCS(i,j,atrad,crit,r12_,ccc,dcAB,rrr,drrr,dfactor,T_INFO,&
        & LATT_CUR,counter,energy,vdw_f2,virialstress,extragrad,sr)
        !c computes contributions for all relevant translations of atomic pair i,j 
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),grad(3,T_INFO%NIONS+3)
          REAL(q) :: atrad
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm,sr
          integer :: crit(3)
          INTEGER :: i,j,counter,ii, jj, kk
          REAL(q) :: dcAB(3,T_INFO%NIONS+3),extragrad(3,T_INFO%NIONS+3),drrr(3,T_INFO%NIONS+3)

          DO ii=-crit(1),crit(1)
            DO jj=-crit(2),crit(2)
              DO kk=-crit(3),crit(3)
                 r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
                 r12norm=SUM(r12**2)**0.5
                 IF (r12norm>0.5 .AND. r12norm<atrad) THEN
                 !IF (r12norm>0.1) THEN
                   counter=counter+1
                   if (i==j) then
                     energy=energy+0.5*vdw_e(ccc,dfactor,rrr,r12norm)
                     gradients=0.5*vdw_q(ccc,dfactor,rrr,r12norm)
                   else
                     energy=energy+vdw_e(ccc,dfactor,rrr,r12norm)
                     gradients=vdw_q(ccc,dfactor,rrr,r12norm)
                   endif
                   virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                   IF (i /= j) THEN
                     vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                     vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
                   ENDIF 
                   CALL gradient_Escs(T_INFO%NIONS,dcAB,r12norm,rrr,dfactor,grad,ccc,drrr)
                   IF (i==j) THEN
                     !extragrad=extragrad+0.5*grad
                     extragrad=extragrad-0.5*grad
                     !extragrad=extragrad+0*grad
                   ELSE
                     !extragrad=extragrad+grad
                     !!extragrad=extragrad-0.5*grad
                     extragrad=extragrad-grad
                   ENDIF
                   !vdw_f2=vdw_f2+grad
                 END IF
               END DO
             END DO
           END DO
        END SUBROUTINE vdw_pairContribSCS

        SUBROUTINE ew_direct_sum_full(LATT_CUR, T_INFO, DYN,crit, c6,theta,energy,vdw_f2,virialstress)
        !c Ewald's summation - the real space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          REAL(q) :: c6ij,aa,theta,c6i,c6j
          real(q) :: r12(3),r12_(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,indx1,indx2,t1,t2,t3
          REAL(q) :: c6(T_INFO%NTYP) ! cc_*s6*conversion
          REAL(q) :: x(3,T_INFO%NIONS)

          x=MATMUL(LATT_CUR%A,DYN%POSION) 

          DO i=1,T_INFO%NIONS
            indx1=T_INFO%ITYP(i)
            c6i=c6(indx1)
            DO j=i,T_INFO%NIONS
              indx2=T_INFO%ITYP(j)
              c6j=c6(indx2)
              c6ij=-SQRT(c6i*c6j)
              r12_=x(:,i)-x(:,j)
              !r12_=DYN%POSION(:,i)- DYN%POSION(:,j)   

              DO t1=-crit(1),crit(1)
                !r12=0._q
                !r12(1)=r12_(1)+t1
                DO t2=-crit(2),crit(2)
                  !r12(2)=r12_(2)+t2
                  DO t3=-crit(3),crit(3)
                    !r12(3)=r12_(3)+t3                  
                    !r12=MATMUL(LATT_CUR%A,r12)
                    r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                    r12norm=SUM(r12**2)**0.5
                    aa=r12norm/theta
                    !aa=SUM(r12**2)**0.5/theta
                    IF (i==j) THEN
                      IF (aa>0.1_q) THEN
                         energy=energy+0.5*c6ij*aa**(-2.)*EXP(-aa*aa)*(aa**(-4)+aa**(-2)+0.5)
                         gradients=-0.5*c6ij*r12norm*(6./aa**8 + 6./aa**6 + 3./aa**4 + 1./aa**2)*EXP(-aa*aa)/theta**8
                      ENDIF
                    ELSE
                      energy=energy+c6ij*aa**(-2.)*EXP(-aa*aa)*(aa**(-4)+aa**(-2)+0.5)
                      gradients=-c6ij*r12norm*(6./aa**8 + 6./aa**6 + 3./aa**4 + 1./aa**2)*EXP(-aa*aa)/theta**8
                      vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                      vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm 
                    END IF
                    IF (r12norm>0._q) THEN
                      virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                    ENDIF

                  END DO
                END DO
              END DO
            ENDDO
          ENDDO
          energy=energy/theta**6
        END SUBROUTINE ew_direct_sum_full

        SUBROUTINE ew_direct_sum_part(i,j,r12_,c6ij,LATT_CUR, T_INFO, DYN,crit, theta,energy,vdw_f2,virialstress)
        !c Ewald's summation - the real space part for a pair i,j
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(in) :: c6ij,r12_(3),theta
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: energy,virialstress(3,3)
          REAL(q) :: aa,r12(3)
          REAL(q) :: gradients,r12norm
          INTEGER :: counter,t1,t2,t3

          energy=0._q
          vdw_f2=0._q
          virialstress=0._q

          DO t1=-crit(1),crit(1)
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                r12norm=SUM(r12**2)**0.5
                aa=r12norm/theta
                IF (i==j) THEN
                  IF (aa>0.1_q) THEN
                     energy=energy+0.5*c6ij*aa**(-2.)*EXP(-aa*aa)*(aa**(-4)+aa**(-2)+0.5)
                     gradients=-0.5*c6ij*r12norm*(6./aa**8 + 6./aa**6 + 3./aa**4 + 1./aa**2)*EXP(-aa*aa)/theta**8
                  ENDIF
                ELSE
                  energy=energy+c6ij*aa**(-2.)*EXP(-aa*aa)*(aa**(-4)+aa**(-2)+0.5)
                  gradients=-c6ij*r12norm*(6./aa**8 + 6./aa**6 + 3./aa**4 + 1./aa**2)*EXP(-aa*aa)/theta**8
                  vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
                  vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm 
                END IF
                IF (r12norm>0._q) THEN
                  virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
                ENDIF
              END DO
            END DO
          END DO
          
          energy=energy/theta**6
        END SUBROUTINE ew_direct_sum_part

        SUBROUTINE ew_reciprocal_sum_full(LATT_CUR, T_INFO, DYN,crit, c6,theta,energy,vdw_f2,virialstress)
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          REAL(q) :: c6ij,bb,theta,c6i,c6j,h
          real(q) :: r12(3)
          REAL(q) :: energy,virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,indx1,indx2,t1,t2,t3
          REAL(q) :: c6(T_INFO%NTYP) ! cc_*s6*conversion
          REAL(q) :: reciprocalLat(3,3)
          REAL(q) :: recvect_1(3),recvect_2(3),recvect_3(3)
          REAL(q) :: part1,part2,part3
          REAL(q) :: x(3,T_INFO%NIONS)
          REAL(q) :: energy1,energy2

          energy=0._q
          energy1=0._q
          energy2=0._Q
          reciprocalLat=LATT_CUR%B*TPI
          x=MATMUL(LATT_CUR%A,DYN%POSION) 

          DO i=1,T_INFO%NIONS
            indx1=T_INFO%ITYP(i)
            c6i=c6(indx1)
            energy2=energy2-c6i
            DO j=i,T_INFO%NIONS
              indx2=T_INFO%ITYP(j)
              c6j=c6(indx2)
              c6ij=-SQRT(c6i*c6j)
              IF (i==j) THEN
                energy1=energy1+c6ij
              ELSE
                energy1=energy1+2*c6ij
              ENDIF

              !r12=DYN%POSION(:,i)- DYN%POSION(:,j)   
              !r12=MATMUL(LATT_CUR%A,r12)
              r12=x(:,i)-x(:,j)
              DO t1=-crit(1),crit(1)
                !recvect_1=reciprocalLat(1,:)*t1     
                DO t2=-crit(2),crit(2)
                  !recvect_2=reciprocalLat(2,:)*t2
                  DO t3=-crit(3),crit(3)
                    IF ( .NOT. (t1==0 .AND. t2==0 .AND. t3==0)) THEN
                      !recvect_3=reciprocalLat(3,:)*t3
                      !recvect_3=recvect_1+recvect_2+recvect_3
                      recvect_3=MATMUL(1._q*(/t1,t2,t3/),(reciprocalLat))
                      h=SUM(recvect_3**2)**0.5
                      bb=0.5*h*theta
                      part1=c6ij*COS(SUM(recvect_3*r12))*h**3
                      part2=PI**0.5*(1._q-ERRF(bb))+(1./(2*bb*bb*bb)-1./bb)*EXP(-bb*bb)
                      IF (i==j) THEN
                        energy=energy+part1*part2
                        !vdw_f2(:,i)=vdw_f2(:,i)-0.5*recvect_3*c6ij*SIN(SUM(recvect_3*r12))*h**3*part2
                      ELSE
                        energy=energy+2*part1*part2
                        vdw_f2(:,i)=vdw_f2(:,i)-recvect_3*c6ij*SIN(SUM(recvect_3*r12))*h**3*part2
                      ENDIF
                      part3=3*h*(PI**0.5*(1._q-ERRF(bb))-EXP(-bb*bb)/bb)
                      virialstress=virialstress+part1/h**3*part3*OUTERPROD(3,recvect_3,recvect_3)
                    ENDIF
                  END DO
                END DO
              END DO
            ENDDO
          ENDDO

          virialstress=virialstress*PI**1.5/24./LATT_CUR%OMEGA

          energy=energy*PI**1.5/24./LATT_CUR%OMEGA
          virialstress(1,1)=virialstress(1,1)+energy
          virialstress(2,2)=virialstress(2,2)+energy
          virialstress(3,3)=virialstress(3,3)+energy

          energy1=energy1*PI**1.5/6./LATT_CUR%OMEGA/theta**3
          energy2=-energy2/12./theta**6
          energy=energy+energy1+energy2
          
          vdw_f2= vdw_f2*PI**1.5/12./LATT_CUR%OMEGA
          virialstress(1,1)=virialstress(1,1)+energy1
          virialstress(2,2)=virialstress(2,2)+energy1
          virialstress(3,3)=virialstress(3,3)+energy1
          
        END SUBROUTINE ew_reciprocal_sum_full

        SUBROUTINE ew_reciprocal_sum_part(i,j,r12,c6ij,LATT_CUR, T_INFO, DYN,crit, theta,energy,vdw_f2,virialstress)
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          REAL(q),INTENT(in) :: c6ij,r12(3),theta
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: energy,virialstress(3,3)
          REAL(q) :: bb,h,r12norm
          INTEGER :: counter,t1,t2,t3
          REAL(q) :: reciprocalLat(3,3)
          REAL(q) :: recvect_1(3),recvect_2(3),recvect_3(3)
          REAL(q) :: part1,part2,part3
          REAL(q) :: energy1,energy2

          energy=0._q; energy1=0._q; energy2=0._q
          virialstress=0._q
          vdw_f2=0._q

          !reciprocalLat=LATT_CUR%B*TPI
          reciprocalLat=TRANSPOSE(LATT_CUR%B)*TPI
          
          IF (i==j) THEN
            energy2=energy2+c6ij
            energy1=energy1+c6ij
          ELSE
            energy1=energy1+2*c6ij
          ENDIF

          DO t1=-crit(1),crit(1)    
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                IF ( .NOT. (t1==0 .AND. t2==0 .AND. t3==0)) THEN
                  recvect_3=MATMUL(1._q*(/t1,t2,t3/),(reciprocalLat))
                  h=SUM(recvect_3**2)**0.5
                  bb=0.5*h*theta
                  part1=c6ij*COS(SUM(recvect_3*r12))*h**3
                  part2=PI**0.5*(1._q-ERRF(bb))+(1./(2*bb*bb*bb)-1./bb)*EXP(-bb*bb)
                  IF (i==j) THEN
                    energy=energy+part1*part2
                    !vdw_f2(:,i)=vdw_f2(:,i)-0.5*recvect_3*c6ij*SIN(SUM(recvect_3*r12))*h**3*part2
                  ELSE
                    energy=energy+2*part1*part2
                    vdw_f2(:,i)=vdw_f2(:,i)-recvect_3*c6ij*SIN(SUM(recvect_3*r12))*h**3*part2
                  ENDIF
                  part3=3*h*(PI**0.5*(1._q-ERRF(bb))-EXP(-bb*bb)/bb)
                  IF (i==j) THEN
                    virialstress=virialstress+part1/h**3*part3*OUTERPROD(3,recvect_3,recvect_3)
                  ELSE
                    virialstress=virialstress+2*part1/h**3*part3*OUTERPROD(3,recvect_3,recvect_3)
                  ENDIF
                ENDIF
              END DO
            END DO
          END DO
  
          virialstress=virialstress*PI**1.5/24./LATT_CUR%OMEGA

          energy=energy*PI**1.5/24./LATT_CUR%OMEGA
          virialstress(1,1)=virialstress(1,1)+energy
          virialstress(2,2)=virialstress(2,2)+energy
          virialstress(3,3)=virialstress(3,3)+energy

          energy1=energy1*PI**1.5/6./LATT_CUR%OMEGA/theta**3
          energy2=-energy2/12./theta**6
          energy=energy+energy1+energy2
          
          vdw_f2= vdw_f2*PI**1.5/12./LATT_CUR%OMEGA
          virialstress(1,1)=virialstress(1,1)+energy1
          virialstress(2,2)=virialstress(2,2)+energy1
          virialstress(3,3)=virialstress(3,3)+energy1
          
        END SUBROUTINE ew_reciprocal_sum_part

        SUBROUTINE ew_correction_sum(LATT_CUR, T_INFO, DYN, c6,theta,energy)
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          REAL(q) :: c6ij,theta,c6i,c6j,h
          real(q) :: r12(3)
          REAL(q) :: virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: energy1,energy2,energy
          INTEGER :: i,j,counter,indx1,indx2
          REAL(q) :: c6(T_INFO%NTYP) ! cc_*s6*conversion
          
          energy1=0._q
          energy2=0._Q

          DO i=1,T_INFO%NIONS
            indx1=T_INFO%ITYP(i)
            c6i=c6(indx1)
            energy2=energy2-c6i
            DO j=i,T_INFO%NIONS
              indx2=T_INFO%ITYP(j)
              c6j=c6(indx2)
              c6ij=-SQRT(c6i*c6j)
              IF (i==j) THEN
                energy1=energy1+c6ij
              ELSE
                energy1=energy1+2*c6ij
              ENDIF
            ENDDO
          ENDDO
  
          energy1=energy1*PI**1.5/6./LATT_CUR%OMEGA/theta**3
          energy2=-energy2/12./theta**6
          energy=energy1+energy2
        END SUBROUTINE ew_correction_sum

        SUBROUTINE ew_determine_theta(LATT_CUR,theta)
        !c determine optimal value for damping parameter 
        !c for Ewald's summation
          TYPE(latt) :: LATT_CUR
          REAL(q) :: theta
          REAL(q) :: aa(3,3),bb(3,3),amin,bmin

          aa=MATMUL(TRANSPOSE(LATT_CUR%A),LATT_CUR%A)
          !bb=MATMUL(LATT_CUR%B,TRANSPOSE(LATT_CUR%B))
          bb=MATMUL(TRANSPOSE(LATT_CUR%B),LATT_CUR%B)
          bb=TPI**2*bb
  
          amin=aa(1,1)
          bmin=bb(1,1)
          IF (aa(2,2)<amin) amin=aa(2,2)
          IF (aa(3,3)<amin) amin=aa(3,3)
          IF (bb(2,2)<bmin) bmin=bb(2,2)
          IF (bb(3,3)<bmin) bmin=bb(3,3)

          theta=(4*amin/bmin)**0.25

        END SUBROUTINE ew_determine_theta

        SUBROUTINE ew_trial_1(r,theta,pref,x)
        !c auxiliary subroutine needed for iterative search of optimal 
        !c parameters for Ewald's summation
          REAL(q) :: r,theta,pref,x

!           write(*,*) r,theta,pref,ERRF(r/theta),'xxx'
          x=(1./r**4+1./r**2/theta**2+1./2/theta**4)*(1.-ERRF(r/theta))
          x=x*pref
        END SUBROUTINE ew_trial_1

        SUBROUTINE ew_trial_2(r,theta,pref,x)
        !c auxiliary subroutine needed for iterative search of optimal 
        !c parameters for Ewald's summation
          REAL(q) :: a,r,theta,pref,x
         
          a=r*theta/2.
          !write(*,*) r,theta,pref,ERRF(a),'xxx'
          x=2*a*exp(-a*a)+PI**0.5*(1.-ERRF(a))
          x=x*pref
        END SUBROUTINE ew_trial_2

        SUBROUTINE ew_find_cutoffs(T_INFO,LATT_CUR,c6,theta,amax,bmax)
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),PARAMETER :: epsil=1e-6,incrit=1e-6
          REAL(q) :: bav,pref,dr,x,c6i,c6j,c6ij,theta
          REAL(q) :: c6(T_INFO%NTYP)
          REAL(q) :: amax,bmax
          INTEGER :: i,j,indx1,indx2
          INTEGER, PARAMETER :: nmax=100

          bav=0._q
          DO i=1,T_INFO%NIONS
            indx1=T_INFO%ITYP(i)
            c6i=c6(indx1)
            bav=bav+c6i
            DO j=i+1,T_INFO%NIONS
              indx2=T_INFO%ITYP(j)
              c6j=c6(indx2)
              c6ij=SQRT(c6i*c6j)
              bav=bav+2*c6ij
            ENDDO
           ! write(*,*) 'bav',bav
          ENDDO
  
          pref=PI**1.5*bav*theta/LATT_CUR%OMEGA
          amax=10.
          dr=5.
          DO i=1,nmax
            IF (dr<=incrit) EXIT                       
            CALL ew_trial_1(amax,theta,pref,x)
            IF (ABS(x)<ABS(epsil)) THEN
              dr=dr/2
              amax=amax-dr
            ELSE
              amax=amax+dr
            ENDIF 
!             write(*,*), i,amax,x,epsil
          ENDDO
          IF (i>nmax) THEN
            WRITE(*,'(A,I4,A)') 'ew_find_cutoffs: ERROR1, unable to reach convergence in ',nmax,' cycles.'
            STOP
          ENDIF  
    
          pref=bav/6./PI**0.5/theta**6
          bmax=2.
          dr=1.
          DO i=1,nmax
            IF (dr<=incrit) EXIT           
            CALL ew_trial_2(bmax,theta,pref,x)
            IF (ABS(x)<ABS(epsil)) THEN
              dr=dr/2
              bmax=bmax-dr
            ELSE
              bmax=bmax+dr
            ENDIF 
          ENDDO
          IF (i>nmax) THEN
            WRITE(*,'(A,I3,A)') 'ew_find_cutoffs: ERROR2, unable to reach convergence in ',nmax,' cycles.'
            STOP
          ENDIF 

        END SUBROUTINE ew_find_cutoffs

        SUBROUTINE ew_damping_correction_full(LATT_CUR, T_INFO, DYN,crit,c6,r0,dfactor,energy,vdw_f2,virialstress)                                      
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          REAL(q) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q) :: atrad
          REAL(q) :: c6ij,c6i,c6j,h
          real(q) :: r12(3),r12_(3)
          REAL(q) :: virialstress(3,3)
          REAL(q) :: gradients,rrr,ccc,dfactor
          REAL(q) :: r12norm
          integer :: crit(3)
          INTEGER :: i,j,counter,indx1,indx2,t1,t2,t3
          REAL(q) :: c6(T_INFO%NTYP) ! cc_*s6*conversion
          REAL(q) :: r0(T_INFO%NTYP) ! r0_*sR
          REAL(q) :: energyRef,energyDamp,energy
          REAL(q) :: x(3,T_INFO%NIONS)
          REAL(q) :: gradientsRef, gradientsDamp
          REAL(q) :: vdw_f2Ref(3,T_INFO%NIONS), vdw_f2Damp(3,T_INFO%NIONS)
          REAL(q) :: virialstressRef(3,3),virialstressDamp(3,3)
     
          energyRef=0._q; energyDamp=0._q
          vdw_f2Ref=0._q; vdw_f2Damp=0._q
          virialstressRef=0._q; virialstressDamp=0._q

          x=MATMUL(LATT_CUR%A,DYN%POSION) 

          DO i=1,T_INFO%NIONS
            indx1=T_INFO%ITYP(i)
            c6i=c6(indx1)
            DO j=i,T_INFO%NIONS
              indx2=T_INFO%ITYP(j)
              c6j=c6(indx2)
              c6ij=SQRT(c6i*c6j)
              rrr=r0(indx1)+r0(indx2)
              !r12_=DYN%POSION(:,i)- DYN%POSION(:,j)   
              r12_=x(:,i)-x(:,j)

              DO t1=-crit(1),crit(1)
                !r12=0._q
                !r12(1)=r12_(1)+t1*1._q
                DO t2=-crit(2),crit(2)
                  !r12(2)=r12_(2)+t2*1._q
                  DO t3=-crit(3),crit(3)
                    !r12(3)=r12_(3)+t3*1_q
                    !r12=MATMUL(LATT_CUR%A,r12)
                    !r12=MATMUL(r12,TRANSPOSE(LATT_CUR%A))
                    r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                    r12norm=SUM(r12**2)**0.5
                    IF (i==j) THEN
                      IF (r12norm>0.1_q) THEN
                         energyRef=energyRef-0.5*c6ij/r12norm**6
                         gradientsRef=3*c6ij/r12norm**7
                         energyDamp=energyDamp+0.5*vdw_e(c6ij,dfactor,rrr,r12norm)
                         gradientsDamp=0.5*vdw_q(c6ij,dfactor,rrr,r12norm)
                      ENDIF
                    ELSE
                      energyRef=energyRef-c6ij/r12norm**6
                      gradientsRef=6*c6ij/r12norm**7
                      vdw_f2Ref(:,i)=vdw_f2Ref(:,i)+gradientsRef*r12/r12norm
                      vdw_f2Ref(:,j)=vdw_f2Ref(:,j)-gradientsRef*r12/r12norm 

                      energyDamp=energyDamp+vdw_e(c6ij,dfactor,rrr,r12norm)
                      gradientsDamp=vdw_q(c6ij,dfactor,rrr,r12norm)
                      vdw_f2Damp(:,i)=vdw_f2Damp(:,i)+gradientsDamp*r12/r12norm
                      vdw_f2Damp(:,j)=vdw_f2Damp(:,j)-gradientsDamp*r12/r12norm                  
                    END IF
                    IF (r12norm>0.1_q) THEN
                      virialstressRef=virialstressRef-gradientsRef*OUTERPROD(3,r12,r12)/r12norm
                      virialstressDamp=virialstressDamp-gradientsDamp*OUTERPROD(3,r12,r12)/r12norm
                    ENDIF
                  END DO
                END DO
              END DO
            ENDDO
          ENDDO
          energy=energy+(energyDamp-energyRef)
          vdw_f2=vdw_f2+(vdw_f2Damp-vdw_f2Ref)
          virialstress=virialstress+(virialstressDamp-virialstressRef)
          !vdw_f2=vdw_f2+(vdw_f2Damp)
          !virialstress=virialstress+(virialstressDamp)
          
        END SUBROUTINE ew_damping_correction_full

        SUBROUTINE ew_damping_correction_part(i,j,r12_,c6ij,rrr,LATT_CUR, T_INFO, DYN,crit,dfactor,energy,vdw_f2,virialstress)                                      
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(in) :: c6ij,rrr,r12_(3),dfactor
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: virialstress(3,3)
          REAL(q),INTENT(out) :: energy
          REAL(q) :: atrad,h,gradients,r12norm 
          REAL(q) :: r12(3)
          INTEGER :: counter,t1,t2,t3
          REAL(q) :: energyRef,energyDamp
          REAL(q) :: gradientsRef, gradientsDamp
          REAL(q) :: vdw_f2Ref(3,T_INFO%NIONS), vdw_f2Damp(3,T_INFO%NIONS)
          REAL(q) :: virialstressRef(3,3),virialstressDamp(3,3)
     
          energy=0._q; energyRef=0._q; energyDamp=0._q
          vdw_f2=0._q;vdw_f2Ref=0._q; vdw_f2Damp=0._q
          virialstress=0._q;virialstressRef=0._q; virialstressDamp=0._q
          

          DO t1=-crit(1),crit(1)
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                r12norm=SUM(r12**2)**0.5
                IF (i==j) THEN
                  IF (r12norm>0.1_q) THEN
                    energyRef=energyRef-0.5*c6ij/r12norm**6
                    gradientsRef=3*c6ij/r12norm**7
                    energyDamp=energyDamp+0.5*vdw_e(c6ij,dfactor,rrr,r12norm)
                    gradientsDamp=0.5*vdw_q(c6ij,dfactor,rrr,r12norm)
                  ENDIF
                ELSE
                  energyRef=energyRef-c6ij/r12norm**6
                  gradientsRef=6*c6ij/r12norm**7
                  vdw_f2Ref(:,i)=vdw_f2Ref(:,i)+gradientsRef*r12/r12norm
                  vdw_f2Ref(:,j)=vdw_f2Ref(:,j)-gradientsRef*r12/r12norm 

                  energyDamp=energyDamp+vdw_e(c6ij,dfactor,rrr,r12norm)
                  gradientsDamp=vdw_q(c6ij,dfactor,rrr,r12norm)
                  vdw_f2Damp(:,i)=vdw_f2Damp(:,i)+gradientsDamp*r12/r12norm
                  vdw_f2Damp(:,j)=vdw_f2Damp(:,j)-gradientsDamp*r12/r12norm                  
                END IF
                IF (r12norm>0.1_q) THEN
                  virialstressRef=virialstressRef-gradientsRef*OUTERPROD(3,r12,r12)/r12norm
                  virialstressDamp=virialstressDamp-gradientsDamp*OUTERPROD(3,r12,r12)/r12norm
                ENDIF
              END DO
            END DO
          END DO
          energy=(energyDamp-energyRef)
          vdw_f2=(vdw_f2Damp-vdw_f2Ref)
          virialstress=(virialstressDamp-virialstressRef)
          
        END SUBROUTINE ew_damping_correction_part

        SUBROUTINE ew_damping_correction_partDamp2(i,j,r12_,c6ij,rrr,LATT_CUR, T_INFO, DYN,crit,mu,energy,vdw_f2,virialstress)                                      
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(in) :: c6ij,rrr,r12_(3),mu
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: virialstress(3,3)
          REAL(q),INTENT(out) :: energy
          REAL(q) :: atrad,h,gradients,r12norm 
          REAL(q) :: r12(3)
          INTEGER :: counter,t1,t2,t3
          REAL(q) :: energyRef,energyDamp
          REAL(q) :: gradientsRef, gradientsDamp
          REAL(q) :: vdw_f2Ref(3,T_INFO%NIONS), vdw_f2Damp(3,T_INFO%NIONS)
          REAL(q) :: virialstressRef(3,3),virialstressDamp(3,3)
     
          energy=0._q; energyRef=0._q; energyDamp=0._q
          vdw_f2=0._q;vdw_f2Ref=0._q; vdw_f2Damp=0._q
          virialstress=0._q;virialstressRef=0._q; virialstressDamp=0._q
          

          DO t1=-crit(1),crit(1)
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                r12norm=SUM(r12**2)**0.5
                IF (i==j) THEN
                  IF (r12norm>0.1_q) THEN
                    energyRef=energyRef-0.5*c6ij/r12norm**6
                    gradientsRef=3*c6ij/r12norm**7
                    energyDamp=energyDamp+0.5*vdw_e_damp2(c6ij,r12norm,mu)           
                    gradientsDamp=0.5*vdw_q_damp2(c6ij,r12norm,mu)   
                  ENDIF
                ELSE
                  energyRef=energyRef-c6ij/r12norm**6
                  gradientsRef=6*c6ij/r12norm**7
                  vdw_f2Ref(:,i)=vdw_f2Ref(:,i)+gradientsRef*r12/r12norm
                  vdw_f2Ref(:,j)=vdw_f2Ref(:,j)-gradientsRef*r12/r12norm 

                  energyDamp=energyDamp+vdw_e_damp2(c6ij,r12norm,mu)          
                  gradientsDamp=vdw_q_damp2(c6ij,r12norm,mu)
                  vdw_f2Damp(:,i)=vdw_f2Damp(:,i)+gradientsDamp*r12/r12norm
                  vdw_f2Damp(:,j)=vdw_f2Damp(:,j)-gradientsDamp*r12/r12norm                  
                END IF
                IF (r12norm>0.1_q) THEN
                  virialstressRef=virialstressRef-gradientsRef*OUTERPROD(3,r12,r12)/r12norm
                  virialstressDamp=virialstressDamp-gradientsDamp*OUTERPROD(3,r12,r12)/r12norm
                ENDIF
              END DO
            END DO
          END DO
          energy=(energyDamp-energyRef)
          vdw_f2=(vdw_f2Damp-vdw_f2Ref)
          virialstress=(virialstressDamp-virialstressRef)
          
        END SUBROUTINE ew_damping_correction_partDamp2


        SUBROUTINE ew_damping_correction_partULG(i,j,r12_,c6ij,rrr,LATT_CUR, T_INFO, DYN,crit,dfactor,energy,vdw_f2,virialstress)                                      
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(in) :: c6ij,rrr,r12_(3),dfactor
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: virialstress(3,3)
          REAL(q),INTENT(out) :: energy
          REAL(q) :: atrad,h,gradients,r12norm 
          REAL(q) :: r12(3)
          INTEGER :: counter,t1,t2,t3
          REAL(q) :: energyRef,energyDamp
          REAL(q) :: gradientsRef, gradientsDamp
          REAL(q) :: vdw_f2Ref(3,T_INFO%NIONS), vdw_f2Damp(3,T_INFO%NIONS)
          REAL(q) :: virialstressRef(3,3),virialstressDamp(3,3)
     
          energy=0._q; energyRef=0._q; energyDamp=0._q
          vdw_f2=0._q;vdw_f2Ref=0._q; vdw_f2Damp=0._q
          virialstress=0._q;virialstressRef=0._q; virialstressDamp=0._q
          
          DO t1=-crit(1),crit(1)
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                r12norm=SUM(r12**2)**0.5
                IF (i==j) THEN
                  IF (r12norm>0.1_q) THEN
                    energyRef=energyRef-0.5*c6ij/r12norm**6
                    gradientsRef=3*c6ij/r12norm**7
                    energyDamp=energyDamp+0.5*vdw_e_ulg(c6ij,dfactor,rrr,r12norm) 
                    gradientsDamp=0.5*vdw_q_ulg(c6ij,dfactor,rrr,r12norm)
                  ENDIF
                ELSE
                  energyRef=energyRef-c6ij/r12norm**6
                  gradientsRef=6*c6ij/r12norm**7
                  vdw_f2Ref(:,i)=vdw_f2Ref(:,i)+gradientsRef*r12/r12norm
                  vdw_f2Ref(:,j)=vdw_f2Ref(:,j)-gradientsRef*r12/r12norm 

                  energyDamp=energyDamp+vdw_e_ulg(c6ij,dfactor,rrr,r12norm)
                  gradientsDamp=vdw_q_ulg(c6ij,dfactor,rrr,r12norm)
                  vdw_f2Damp(:,i)=vdw_f2Damp(:,i)+gradientsDamp*r12/r12norm
                  vdw_f2Damp(:,j)=vdw_f2Damp(:,j)-gradientsDamp*r12/r12norm                  
                END IF
                IF (r12norm>0.1_q) THEN
                  virialstressRef=virialstressRef-gradientsRef*OUTERPROD(3,r12,r12)/r12norm
                  virialstressDamp=virialstressDamp-gradientsDamp*OUTERPROD(3,r12,r12)/r12norm
                ENDIF
              END DO
            END DO
          END DO
          energy=(energyDamp-energyRef)
          vdw_f2=(vdw_f2Damp-vdw_f2Ref)
          virialstress=(virialstressDamp-virialstressRef)
          
        END SUBROUTINE ew_damping_correction_partULG

        SUBROUTINE ew_damping_correction_partSCS(i,j,r12_,c6ij,rrr,LATT_CUR, T_INFO, DYN,crit,dfactor,dcAB,drrr,energy,vdw_f2,virialstress,extragrad)                                      
        !c Ewald's summation - the reciprocal space part
          TYPE(latt) :: LATT_CUR
          TYPE(type_info)  :: T_INFO
          TYPE(dynamics) :: DYN
          INTEGER,INTENT(in) :: i,j,crit(3)
          REAL(q),INTENT(in) :: c6ij,rrr,r12_(3),dfactor
          REAL(q),INTENT(in) :: dcAB(3,T_INFO%NIONS+3),drrr(3,T_INFO%NIONS+3)
          REAL(q),INTENT(out) :: vdw_f2(3,T_INFO%NIONS)
          REAL(q),INTENT(out) :: virialstress(3,3)
          REAL(q),INTENT(out) :: energy,extragrad(3,T_INFO%NIONS+3)
          REAL(q) :: atrad,h,gradients,r12norm 
          REAL(q) :: r12(3)
          INTEGER :: counter,t1,t2,t3
          REAL(q) :: energyRef,energyDamp
          REAL(q) :: gradientsRef, gradientsDamp
          REAL(q) :: vdw_f2Ref(3,T_INFO%NIONS), vdw_f2Damp(3,T_INFO%NIONS)
          REAL(q) :: virialstressRef(3,3),virialstressDamp(3,3)
          REAL(q) :: grad(3,T_INFO%NIONS+3)
     
          energy=0._q; energyRef=0._q; energyDamp=0._q
          vdw_f2=0._q;vdw_f2Ref=0._q; vdw_f2Damp=0._q
          virialstress=0._q;virialstressRef=0._q; virialstressDamp=0._q
          extragrad=0._q
          

          DO t1=-crit(1),crit(1)
            DO t2=-crit(2),crit(2)
              DO t3=-crit(3),crit(3)
                r12=r12_+MATMUL(1._q*(/t1,t2,t3/),TRANSPOSE(LATT_CUR%A))
                r12norm=SUM(r12**2)**0.5
                IF (i==j) THEN
                  IF (r12norm>0.1_q) THEN
                    energyRef=energyRef-0.5*c6ij/r12norm**6
                    gradientsRef=3*c6ij/r12norm**7
                    energyDamp=energyDamp+0.5*vdw_e(c6ij,dfactor,rrr,r12norm)
                    gradientsDamp=0.5*vdw_q(c6ij,dfactor,rrr,r12norm)
                  ENDIF
                ELSE
                  energyRef=energyRef-c6ij/r12norm**6
                  gradientsRef=6*c6ij/r12norm**7
                  vdw_f2Ref(:,i)=vdw_f2Ref(:,i)+gradientsRef*r12/r12norm
                  vdw_f2Ref(:,j)=vdw_f2Ref(:,j)-gradientsRef*r12/r12norm 

                  energyDamp=energyDamp+vdw_e(c6ij,dfactor,rrr,r12norm)
                  gradientsDamp=vdw_q(c6ij,dfactor,rrr,r12norm)
                  vdw_f2Damp(:,i)=vdw_f2Damp(:,i)+gradientsDamp*r12/r12norm
                  vdw_f2Damp(:,j)=vdw_f2Damp(:,j)-gradientsDamp*r12/r12norm                  
                END IF
                IF (r12norm>0.1_q) THEN
                  virialstressRef=virialstressRef-gradientsRef*OUTERPROD(3,r12,r12)/r12norm
                  virialstressDamp=virialstressDamp-gradientsDamp*OUTERPROD(3,r12,r12)/r12norm
                ENDIF
                IF (r12norm>0.1_q) THEN
                  CALL gradient_Escs(T_INFO%NIONS,dcAB,r12norm,rrr,dfactor,grad,c6ij,drrr)
                  IF (i==j) THEN
                    extragrad=extragrad-0.5*grad
                  ELSE
                    extragrad=extragrad-grad
                  ENDIF
                ENDIF
              END DO
            END DO
          END DO
          energy=(energyDamp-energyRef)
          vdw_f2=(vdw_f2Damp-vdw_f2Ref)
          virialstress=(virialstressDamp-virialstressRef)
          
        END SUBROUTINE ew_damping_correction_partSCS

        SUBROUTINE vdw_forces_G(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,IVDW)
        !c D2 method of Grimme 
          USE setexm
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),SAVE :: atrad=50._q
          REAL(q) :: gradients
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),TOTEN
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: energy,ccc,rrr,virialstress(3,3)
          REAL(q) :: cc(87),r0(87)
          REAL(q), PARAMETER :: conversion=10.364425449557316
          REAL(q),SAVE :: sfactor=0.75_q,dfactor=20._q
          REAL(q),SAVE :: sR=1.00_q,s6=0.75_q
          REAL(q) :: userC(T_INFO%NTYP)
          CHARACTER (LEN=2) :: tags(87)
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER :: i,j,ii,jj,kk,counter,indx1,indx2
          INTEGER :: historycounter
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          integer :: crit1(3), crit2(3), crit3(3)
          real(q) :: criteria(3)
          real(q) :: x(3,T_INFO%NIONS),x1(3),x2(3),r12(3),r12_(3)
          real(q) :: r12norm
          real(q),ALLOCATABLE,SAVE :: cc_(:),r0_(:)
          integer :: dfnd1(T_INFO%NTYP),dfnd2(T_INFO%NTYP)
!          logical,save :: LEWALD=.TRUE.
           logical,save :: LEWALD=.FALSE.
          real(q) :: energy1,energy2,energy3,energyDum
          REAL(q) :: vdw_f21(3,T_INFO%NIONS),vdw_f22(3,T_INFO%NIONS),vdw_f23(3,T_INFO%NIONS),vdw_f2Dum(3,T_INFO%NIONS)
          REAL(q) :: virialstress1(3,3),virialstress2(3,3),virialstress3(3,3),virialstressDum(3,3)
          real(q) :: theta, amax,bmax
          INTEGER :: IVDW
          LOGICAL :: LUSERPARAM

          vdw_f21=0._q; vdw_f22=0._q; vdw_f23=0._q; vdw_f2Dum=0._q
          virialstress1=0._q; virialstress2=0._q; virialstress3=0._q; virialstressDum=0._q;


  1234 FORMAT(/,"  DFT-D2 method for vdW energy calculation",&
              /,"  -------------------------------------------------------------------")
  1235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"         C6(Jnm^6/mol)     R0(A)",&
              /,"   -----------------------------")
  1236 FORMAT("   ",A2,5X,F7.3,8X,F7.3)
  2250 FORMAT(/,"  IVDW = ",I3)
  1237 FORMAT("  VDW_RADIUS = ",F9.3," A")
  1233 FORMAT("  VDW_S6 = ",F9.3)
  1241 FORMAT("  VDW_SR = ",F9.3)
  1240 FORMAT("  VDW_D = ",F9.3)
  2246 FORMAT("  LVDW_EWALD = ",L1)
        

          if (historycounter==1) then
            ALLOCATE(cc_(T_INFO%NTYP),r0_(T_INFO%NTYP))
            tags=(/'H ','He','Li','Be','B ','C ','N ','O ','F ','Ne',&
                   'Na','Mg','Al','Si','P ','S ','Cl','Ar','K ','Ca',&
                   'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',&
                   'Ga','Ge','As','Se','Br','Kr','Rb','Sr',&
                   'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',&
                   'In','Sn','Sb','Te','I ','Xe','X ', &
                   'Cs','Ba','La','Ce','Pr','Nd','Pm',&
                   'Sm','Eu','Gd',&
                   'Tb','Dy','Ho','Er','Tm','Yb','Lu',&
                   'Hf','Ta','W ','Re','Os','Ir','Pt','Au','Hg',&
                   'Tl','Pb','Bi','Po','At','Rn'/)

            cc=(/0.14,0.08,1.61,1.61,3.13,1.75,1.23,0.70,0.75,0.63,&
                 5.71,5.71,10.79,9.23,7.84,5.57,5.07,4.61,10.8,10.8,&
                 10.8,10.8,10.8,10.8,10.8,10.8,10.8,10.8,10.8,10.8,&
                 16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,&
                 24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,&
                 37.32,38.71,38.44,31.74,31.50,29.99,29.99, &
                 315.275,226.994,176.252,140.68,140.68,140.68,140.68,&
                 140.68,140.68,140.68, &
                 140.68,140.68,140.68,140.68,140.68,140.68,140.68, &
                 105.112,81.24,81.24,81.24,81.24,81.24,81.24,81.24,57.364,&
                 57.254,63.162,63.540,55.283,57.171,56.64/)
            r0=(/1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,&
                 1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,&
                 1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,&
                 1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,&
                 1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,&
                 1.672,1.804,1.881,1.892,1.892,1.881,1.881,&              
                 1.802,1.762,1.720,1.753,1.753,1.753,1.753,&
                 1.753,1.753,1.753,&
                 1.753,1.753,1.753,1.753,1.753,1.753,1.753,&
                 1.788,1.772,1.772,1.772,1.772,1.772,1.772,1.772,1.758,&
                 1.989,1.944,1.898,2.005,1.991,1.924/)

            cc_=0._q;r0_=0._q
            dfnd1=0;dfnd2=0
            DO i=1,T_INFO%NTYP
              indx1=cwhereisit(tags,ELEM(i),SIZE(tags))
              IF (indx1>0) THEN
                dfnd1(i)=1
                dfnd2(i)=1
                cc_(i)=cc(indx1)
                r0_(i)=r0(indx1)
              END IF
            ENDDO

           
            LOPEN=.FALSE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_RADIUS','=','#',';','F', &
                    IDUM,atrad,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) atrad=50._q 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 atrad=50._q  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_RADIUS'' from file INCAR'
              ENDIF
              !c sanity test
              IF (atrad <1._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_RADIUS:',atrad
                      WRITE(IO%IU0,*) 'VDW_RADIUS must be greater than 1.0 Angstrom!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR('VDW_RADIUS','F',IDUM,atrad,CDUM,.TRUE.,CHARAC,1)

  
              !c read the parameter s6, use defaults if VDW_S6 or VDW_SCALING 
              !c is not defined by user
              LUSERPARAM=.FALSE.
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_S6','=','#',';','F', &
                    IDUM,s6,CDUM,LDUM,CHARAC,N,1,IERR)
              IF ((IERR==0) .AND. (N .GE. 1)) LUSERPARAM=.TRUE.
              IF (IERR==3)  LUSERPARAM=.FALSE.
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 !s6=0.75  
                 LUSERPARAM=.FALSE.
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_S6'' from file INCAR'
              ENDIF

              IF (.NOT. LUSERPARAM) THEN
               !c obsolete, to be replaced by VDW_S6
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_SCALING','=','#',';','F', &
                    IDUM,s6,CDUM,LDUM,CHARAC,N,1,IERR)
                IF ((IERR==0) .AND. (N .GE. 1)) LUSERPARAM=.TRUE.
                IF (IERR==3) LUSERPARAM=.FALSE.
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                  LUSERPARAM=.FALSE.
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''VDW_SCALING'' from file INCAR'
                ENDIF
              ENDIF

              IF (LUSERPARAM) THEN
                !c sanity test
                IF (s6 <0._q .OR. s6>10._q) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'Error: invalid value for VDW_SCALING:',s6
                    WRITE(IO%IU0,*) 'VDW_SCALING must be a value from interval (0,10)!'
                  ENDIF
                  STOP
                ENDIF
              ELSE
                SELECT CASE(LEXCH)
                  CASE(4) !PB 
                    s6=1.05
                  CASE(8) !PBE
                    s6=0.75
                  CASE(40) !revPBE
                    s6=1.25

                  !c we have default s6 only for PBE, revPBE, and PB functionals
                  !c We have to terminate if other functional is used and the 
                  !c user didn't provide the s6 value,
                  CASE DEFAULT
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'vdw_forces_G: ERROR unsupported xc-functional, LEXCH=',LEXCH
                      WRITE(IO%IU0,*) 'please define parameter VDW_S6 for this functional'
                    ENDIF
                    STOP  
                END SELECT
              ENDIF
            
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_SR','=','#',';','F', &
                    IDUM,sR,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) sR=1.00
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 sR=1.00 
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_SR'' from file INCAR'
              ENDIF
              !c sanity test
              IF (sR <0._q .OR. sR>10._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_SR:',sR
                      WRITE(IO%IU0,*) 'VDW_SR must be a value from interval (0,10)!'
                ENDIF
                STOP
              ENDIF

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_EWALD','=','#',';','L', &
                  IDUM,RDUM,CDUM,LEWALD,CHARAC,N,1,IERR)
              IF (IERR==3) LEWALD=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LEWALD=.FALSE.  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LEWALD'' from file INCAR'
              ENDIF


              !CALL XML_INCAR('VDW_SCALING','F',IDUM,sfactor,CDUM,.TRUE.,CHARAC,1)

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_D','=','#',';','F', &
                    IDUM,dfactor,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) dfactor=20. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 dfactor=20.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_D'' from file INCAR'
              ENDIF
              !c sanity test
              IF (dfactor <2._q .OR. dfactor>100._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_D:',dfactor
                      WRITE(IO%IU0,*) 'VDW_D must be a value from interval (2,100)!'
                ENDIF
                STOP
              ENDIF
              !CALL XML_INCAR('VDW_D','F',IDUM,dfactor,CDUM,.TRUE.,CHARAC,1)

              userC=0._q
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_C6'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                dfnd1=1
                cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)
              ENDIF

              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters C6 are NOT defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_C6 to define C6 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_C6','F',IDUM,cc_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_R0'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 .AND. userC(j)<10._q) THEN
                    dfnd2(j)=1
                    r0_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_R0 must be a value from interval (0.1,10)!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF


              IF (SUM(dfnd2) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters R0 are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd2(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_R0 to define R0 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_R0','F',IDUM,r0_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

            !CALL XML_CLOSE_TAG
            !CALL XML_CLOSE_TAG

            CLOSE(IO%IU5)
          end if 

          if (historycounter==1) then
            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)
              WRITE(IO%IU6,1235) 
              DO i=1,T_INFO%NTYP
                WRITE(IO%IU6,1236) ELEM(i),cc_(i),r0_(i)
              ENDDO
              WRITE(IO%IU6,2250) IVDW
              WRITE(IO%IU6,1237) atrad
              !WRITE(IO%IU6,1233) sfactor
              WRITE(IO%IU6,1233) s6
              WRITE(IO%IU6,1241) sR
              WRITE(IO%IU6,1240) dfactor
              WRITE(IO%IU6,2246) LEWALD
            end if
            !cc_=sfactor*conversion*cc_
          end if

          energy=0.
          gradients=0.
          vdw_f2=0.
          virialstress=0.
          counter=0
          x=MATMUL(LATT_CUR%A,DYN%POSION) 
          IF (.NOT. LEWALD) THEN
            !c determine the the number of images of simulation cell 
            !c in each direction needed for a given 
            !c cutoff radius
            criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit1=int(criteria)+1    

            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              DO j=i,T_INFO%NIONS
                r12_=x(:,i)-x(:,j)
                indx2=T_INFO%ITYP(j)
                ccc=s6*SQRT(cc_(indx1)*cc_(indx2))*conversion
                rrr=sR*(r0_(indx1)+r0_(indx2))
                IF (rrr>0._q) THEN
                  CALL vdw_pairContrib(i,j,atrad,crit1,r12_,ccc,rrr,dfactor,&
                     & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
                END IF
              END DO
            END DO
          ELSE
            
            !c determine damping parameter and cutoff radii for Ewald's summation
            CALL ew_determine_theta(LATT_CUR,theta)
            CALL ew_find_cutoffs(T_INFO,LATT_CUR,cc_*s6*conversion,theta,amax,bmax)

            !c number of translations of direct lattice vectors to be 
            !c considered in real-space part of Ewald's summation 
            criteria(1)=amax*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=amax*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=amax*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit1=int(criteria)+1

            !c number of translations of reciprocal latticevectors to be 
            !c considered in reciprocal part of Ewald's summation 
            criteria(1)=bmax*SUM(LATT_CUR%A(1,:)**2)**0.5/TPI
            criteria(2)=bmax*SUM(LATT_CUR%A(2,:)**2)**0.5/TPI
            criteria(3)=bmax*SUM(LATT_CUR%A(3,:)**2)**0.5/TPI
            crit2=int(criteria)+1

            !c number of translations of direct lattice vectors
            !c for short-range damping calculations
            !atrad=40.
            criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit3=int(criteria)+1

            energy1=0._q;energy2=0._q;energy3=0._q
            vdw_f21=0._q;vdw_f22=0._q;vdw_f23=0._q
            virialstress1=0._q; virialstress2=0._q;virialstress3=0._q
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              DO j=i,T_INFO%NIONS
                r12_=x(:,i)-x(:,j)
                indx2=T_INFO%ITYP(j)
                ccc=s6*SQRT(cc_(indx1)*cc_(indx2))*conversion
                rrr=sR*(r0_(indx1)+r0_(indx2))
                !c add real space contribution for a pair i,j
                CALL ew_direct_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit1, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy1=energy1+energyDum
                vdw_f21=vdw_f21+vdw_f2Dum
                virialstress1=virialstress1+virialstressDum
                !c add reciprocal space contribution for a pair i,j
                CALL ew_reciprocal_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit2, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy2=energy2+energyDum
                vdw_f22=vdw_f22+vdw_f2Dum
                virialstress2=virialstress2+virialstressDum
                !c add short range damping
                CALL ew_damping_correction_part(i,j,r12_,ccc,rrr,LATT_CUR, T_INFO, DYN,crit3,dfactor,energyDum,vdw_f2Dum,virialstressDum)   
                energy3=energy3+energyDum
                vdw_f23=vdw_f23+vdw_f2Dum
                virialstress3=virialstress3+virialstressDum
              END DO
            END DO


!             !c real-space part of Ewald's summation
!             CALL ew_direct_sum_full(LATT_CUR, T_INFO, DYN,crit1, cc_*s6*conversion,theta,energy1,vdw_f21,virialstress1) 
!             
!             !c reciprocal part of Ewald's summation 
!             CALL ew_reciprocal_sum_full(LATT_CUR, T_INFO, DYN,crit2,cc_*s6*conversion ,theta,energy2,vdw_f22,virialstress2)
! 
!             CALL ew_damping_correction_full(LATT_CUR, T_INFO, DYN,crit3,cc_*s6*conversion,sR*r0_,dfactor,energy3,vdw_f23,virialstress3)
!             IF (IO%IU6>=0) THEN
!               WRITE(*,*) 'theta',theta
!               WRITE(*,*) 'amax',amax
!               WRITE(*,*) 'bmax', bmax
!               WRITE(*,*) 'atrad',atrad
!               WRITE(*,*) 'energy1',energy1
!               WRITE(*,*) 'energy2',energy2
!               WRITE(*,*) 'energy3',energy3
!             END IF

            energy=energy1+energy2+energy3
            vdw_f2=vdw_f21+vdw_f22+vdw_f23
            virialstress=virialstress1+virialstress2+virialstress3
            !vdw_f2=vdw_f24
            !virialstress=virialstress4
            !vdw_f2=vdw_f21
            !virialstress=virialstress1
          ENDIF

          !c update forces
          TIFOR=TIFOR-vdw_f2

          !c update stress tensor   
          TSIF=TSIF+virialstress

          !c update energy
          TOTEN=TOTEN+energy  

  1238 FORMAT(/,"  Number of pair interactions contributing to vdW energy:",I8)
  !1239 FORMAT(  "  Estimated vdW energy (eV):",F10.5)
  1239 FORMAT(  "  Edisp (eV):",F11.5)

          IF (IO%IU6>=0) THEN
            IF (.NOT. LEWALD) WRITE(IO%IU6,1238) counter
            WRITE(IO%IU6,1239) energy
          end if

        END SUBROUTINE vdw_forces_G

        SUBROUTINE vdw_forces_ulg(IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,IVDW)
        !c addition by Sebastien Lebegue
        !c ULG method of Goddard et al., Phys. Chem. Lett. 3, 360 (2012).
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),SAVE :: atrad=50._q
          REAL(q) :: gradients
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),TOTEN
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: energy,ccc,rrr,virialstress(3,3)
          REAL(q) :: cc(104),r0(104)
          REAL(q), PARAMETER :: conversion=10.364425449557316
          REAL(q),SAVE :: s6=0.7012_q,dfactor=0.6966_q
          REAL(q) :: userC(T_INFO%NTYP)
          CHARACTER (LEN=2) :: tags(104)
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER :: i,j,ii,jj,kk,counter,indx1,indx2
          INTEGER :: historycounter
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          real(q) :: criteria(3)
          real(q) :: x(3,T_INFO%NIONS),x1(3),x2(3),r12(3),r12_(3)
          real(q) :: r12norm
          real(q),ALLOCATABLE,SAVE :: cc_(:),r0_(:)
          integer :: dfnd1(T_INFO%NTYP),dfnd2(T_INFO%NTYP)
          LOGICAL, SAVE :: LEWALD=.FALSE.
          !LOGICAL, SAVE :: LEWALD=.TRUE.
          real(q) :: energy1,energy2,energy3,energyDum
          REAL(q) :: vdw_f21(3,T_INFO%NIONS),vdw_f22(3,T_INFO%NIONS),vdw_f23(3,T_INFO%NIONS),vdw_f2Dum(3,T_INFO%NIONS)
          REAL(q) :: virialstress1(3,3),virialstress2(3,3),virialstress3(3,3),virialstressDum(3,3)
          real(q) :: theta, amax,bmax
          integer :: crit1(3), crit2(3), crit3(3)
          INTEGER :: IVDW

  1234 FORMAT(/,"  ULGs method for vdW energy calculation",&
              /,"  -------------------------------------------------------------------")
  1235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"         C6(Jnm^6/mol)     R0(A)",&
              /,"   -----------------------------")
  1236 FORMAT("   ",A2,5X,F7.3,8X,F7.3)
  2250 FORMAT(/,"  IVDW = ",I3)
  1237 FORMAT("  VDW_RADIUS = ",F9.3," A")
  1233 FORMAT("  VDW_S6 = ",F9.4," A")
  1240 FORMAT("  VDW_D = ",F9.4," A")
  2246 FORMAT("  LVDW_EWALD = ",L1)
  
        
          if (historycounter==1) then
            ALLOCATE(cc_(T_INFO%NTYP),r0_(T_INFO%NTYP))
            tags=(/'H ','He','Li','Be','B ','C ','N ','O ','F ','Ne',&
                   'Na','Mg','Al','Si','P ','S ','Cl','Ar','K ','Ca',&
                   'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',&
                   'Ga','Ge','As','Se','Br','Kr','Rb','Sr',&
                   'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',&
                   'In','Sn','Sb','Te','I ','Xe','X ',&
                   'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm',&
                   'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',&
                   'Lu','Hf','Ta','W ','Re','Os','Ir','Pt',&
                   'Au','Hg','Tl','Pb','Bi','Po','At','Rn',&
                   'Fr','Ra','Ac','Th','Pa','U ','Np','Pu',&
                   'Am','Cm','Bk','Cf','Es','Fm','Md','No',&
                   'Lr'/)
            cc=(/50.85,19.45,10.84,72.73,1667.93,684.95,331.72,220.59,144.92,97.71,&
                 42.27,168.76,8375.63,5047.02,3102.66,2365.06,1716.56,1239.14,214.79,734.03,&
                 48.63,34.83,30.91,22.90,17.52,15.85,15.71,15.54,18.23,110.34,&
                 5884.46,4659.42,3540.22,3217.50,2712.48,2218.63,387.86,1095.02,&
                 201.72,128.28,118.61,90.52,69.70,75.79,66.93,56.98,70.07,243.34,&
                 9467.13,8139.30,6695.92,6349.77,5629.95,4844.53,4844.53,&
                 764.44,1876.95,64.90,52.57,43.97,41.75,35.85,30.44,&
                 29.06,26.27,23.65,22.72,21.97,21.29,17.70,650.31,&
                 190.73,138.28,164.39,111.97,87.71,68.26,76.61,69.81,&
                 99.46,301.64,9176.49,8347.10,7215.20,7087.38,6523.94,5805.76,&
                 1384.13,1996.98,116.82,79.76,70.90,67.37,61.23,51.56,&
                 41.82,35.20,36.03,34.38,30.94,30.21,27.10,25.83,&
                 25.26/)
            r0=(/2.886,2.362,2.451,2.745,4.083,3.851,3.660,3.500,3.364,3.243,&
                 2.983,3.021,4.499,4.295,4.147,4.035,3.947,3.868,3.812,3.399,&
                 3.295,3.175,3.144,3.023,2.961,2.912,2.872,2.834,3.495,2.763,&
                 4.383,4.280,4.230,4.205,4.189,4.141,4.114,3.641,&
                 3.345,3.124,3.165,3.052,2.998,2.963,2.929,2.899,3.148,2.848,&
                 4.463,4.392,4.420,4.470,4.500,4.404,4.404,&
                 4.517,3.703,3.522,3.556,3.606,3.575,3.547,3.520,&
                 3.493,3.368,3.451,3.428,3.409,3.391,3.374,3.355,&
                 3.640,3.141,3.170,3.069,2.954,3.120,2.840,2.754,&
                 3.293,2.705,4.347,4.297,4.370,4.709,4.750,4.765,&
                 4.900,3.677,3.478,3.396,3.424,3.395,3.424,3.424,&
                 3.381,3.326,3.339,3.313,3.299,3.286,3.274,3.248,&
                 3.236/)

!           from kcal.A^6 to J.nm^6
            cc=cc*0.00418058785

            cc_=0._q;r0_=0._q
            dfnd1=0;dfnd2=0
            DO i=1,T_INFO%NTYP
              indx1=cwhereisit(tags,ELEM(i),SIZE(tags))
              IF (indx1>0) THEN
                dfnd1(i)=1
                dfnd2(i)=1
                cc_(i)=cc(indx1)
                r0_(i)=r0(indx1)
              END IF
            ENDDO

           
            LOPEN=.FALSE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_RADIUS','=','#',';','F', &
                    IDUM,atrad,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) atrad=50._q 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 atrad=50._q  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_RADIUS'' from file INCAR'
              ENDIF
              !c sanity test
              IF (atrad <1._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_RADIUS:',atrad
                      WRITE(IO%IU0,*) 'VDW_RADIUS must be greater than 1.0 Angstrom!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR('VDW_RADIUS','F',IDUM,atrad,CDUM,.TRUE.,CHARAC,1)

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_S6','=','#',';','F', &
                    IDUM,s6,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) s6=0.7012
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 s6=0.7012
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_S6'' from file INCAR'
              ENDIF
              !c sanity test
              IF (s6 <0._q .OR. s6>10._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_S6:',s6
                      WRITE(IO%IU0,*) 'VDW_SCALING must be a value from interval (0,10)!'
                ENDIF
                STOP
              ENDIF
               
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_EWALD','=','#',';','L', &
                  IDUM,RDUM,CDUM,LEWALD,CHARAC,N,1,IERR)
              IF (IERR==3) LEWALD=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LEWALD=.FALSE.  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LEWALD'' from file INCAR'
              ENDIF


              !CALL XML_INCAR('VDW_SCALING','F',IDUM,sfactor,CDUM,.TRUE.,CHARAC,1)

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_D','=','#',';','F', &
                    IDUM,dfactor,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) dfactor=0.6966
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 dfactor=0.6966
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_D'' from file INCAR'
              ENDIF
              !c sanity test
!              IF (dfactor <2._q .OR. dfactor>100._q) THEN
!                IF (IO%IU0>=0) THEN
!                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_D:',dfactor
!                      WRITE(IO%IU0,*) 'VDW_D must be a value from interval (2,100)!'
!                ENDIF
!                STOP
!              ENDIF
              !CALL XML_INCAR('VDW_D','F',IDUM,dfactor,CDUM,.TRUE.,CHARAC,1)

              userC=0._q
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_C6'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                dfnd1=1
                cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)
              ENDIF

              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters C6 are NOT defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_C6 to define C6 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_C6','F',IDUM,cc_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_R0'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 .AND. userC(j)<10._q) THEN
                    dfnd2(j)=1
                    r0_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_R0 must be a value from interval (0.1,10)!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF


              IF (SUM(dfnd2) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters R0 are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd2(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_R0 to define R0 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_R0','F',IDUM,r0_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

            !CALL XML_CLOSE_TAG
            !CALL XML_CLOSE_TAG

            CLOSE(IO%IU5)
          end if 

          

          if (historycounter==1) then
            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)
              WRITE(IO%IU6,1235) 
              DO i=1,T_INFO%NTYP
                WRITE(IO%IU6,1236) ELEM(i),cc_(i),r0_(i)
              ENDDO
              WRITE(IO%IU6,2250) IVDW
              WRITE(IO%IU6,1237) atrad
              WRITE(IO%IU6,1233) s6
              WRITE(IO%IU6,1240) dfactor
              WRITE(IO%IU6,2246) LEWALD
            end if
            !cc_=sfactor*conversion*cc_
          end if

         
          x=MATMUL(LATT_CUR%A,DYN%POSION)

          energy=0.
          gradients=0.
          vdw_f2=0.
          virialstress=0.
          counter=0

          IF (.NOT. LEWALD) THEN
            !c determine the the number of images of simulation cell 
            !c in each direction needed for a given 
            !c cutoff radius
            criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit1=int(criteria)+1    
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              DO j=i,T_INFO%NIONS               
                r12_=x(:,i)-x(:,j)
                indx2=T_INFO%ITYP(j)
                ccc=s6*SQRT(cc_(indx1)*cc_(indx2))*conversion
                rrr=SQRT(r0_(indx1)*r0_(indx2))
                IF (rrr>0) THEN
                  CALL vdw_pairContrib_ulg(i,j,atrad,crit1,r12_,ccc,rrr,dfactor,&
                     & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
                END IF                
              END DO
            END DO
          
          ELSE
            
            !c determine damping parameter (theta) and cutoff radii for Ewald's summation
            CALL ew_determine_theta(LATT_CUR,theta)
            CALL ew_find_cutoffs(T_INFO,LATT_CUR,cc_*s6*conversion,theta,amax,bmax)

            !c number of translations of direct lattice vectors to be 
            !c considered in real-space part of Ewald's summation 
            criteria(1)=amax*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=amax*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=amax*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit1=int(criteria)+1

            !c number of translations of reciprocal latticevectors to be 
            !c considered in reciprocal part of Ewald's summation 
            criteria(1)=bmax*SUM(LATT_CUR%A(1,:)**2)**0.5/TPI
            criteria(2)=bmax*SUM(LATT_CUR%A(2,:)**2)**0.5/TPI
            criteria(3)=bmax*SUM(LATT_CUR%A(3,:)**2)**0.5/TPI
            crit2=int(criteria)+1

            !c number of translations of direct lattice vectors
            !c for short-range damping calculations
            !atrad=40. !c remove this!!!
            criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit3=int(criteria)+1

            energy1=0._q;energy2=0._q;energy3=0._q
            vdw_f21=0._q;vdw_f22=0._q;vdw_f23=0._q
            virialstress1=0._q; virialstress2=0._q;virialstress3=0._q
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              DO j=i,T_INFO%NIONS
                r12_=x(:,i)-x(:,j)
                indx2=T_INFO%ITYP(j)
                ccc=s6*SQRT(cc_(indx1)*cc_(indx2))*conversion
                rrr=SQRT(r0_(indx1)*r0_(indx2))
                !c add real space contribution for a pair i,j
                CALL ew_direct_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit1, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy1=energy1+energyDum
                vdw_f21=vdw_f21+vdw_f2Dum
                virialstress1=virialstress1+virialstressDum
                !c add reciprocal space contribution for a pair i,j
                CALL ew_reciprocal_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit2, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy2=energy2+energyDum
                vdw_f22=vdw_f22+vdw_f2Dum
                virialstress2=virialstress2+virialstressDum
                !c add short range damping
                CALL ew_damping_correction_partULG(i,j,r12_,ccc,rrr,LATT_CUR, T_INFO, DYN,crit3,dfactor,energyDum,vdw_f2Dum,virialstressDum)   
                energy3=energy3+energyDum
                vdw_f23=vdw_f23+vdw_f2Dum
                virialstress3=virialstress3+virialstressDum
              END DO
            END DO


!             !c real-space part of Ewald's summation
!             CALL ew_direct_sum_full(LATT_CUR, T_INFO, DYN,crit1, cc_*s6*conversion,theta,energy1,vdw_f21,virialstress1) 
!             
!             !c reciprocal part of Ewald's summation 
!             CALL ew_reciprocal_sum_full(LATT_CUR, T_INFO, DYN,crit2,cc_*s6*conversion ,theta,energy2,vdw_f22,virialstress2)
! 
!             CALL ew_damping_correction_full(LATT_CUR, T_INFO, DYN,crit3,cc_*s6*conversion,sR*r0_,dfactor,energy3,vdw_f23,virialstress3)
!             IF (IO%IU6>=0) THEN
!               WRITE(*,*) 'theta',theta
!               WRITE(*,*) 'amax',amax
!               WRITE(*,*) 'bmax', bmax
!               WRITE(*,*) 'atrad',atrad
!               WRITE(*,*) 'energy1',energy1
!               WRITE(*,*) 'energy2',energy2
!               WRITE(*,*) 'energy3',energy3
!             END IF

            energy=energy1+energy2+energy3
            vdw_f2=vdw_f21+vdw_f22+vdw_f23
            virialstress=virialstress1+virialstress2+virialstress3
            !vdw_f2=vdw_f24
            !virialstress=virialstress4
            !vdw_f2=vdw_f21
            !virialstress=virialstress1
          ENDIF


          !c update forces
          TIFOR=TIFOR-vdw_f2

          !c update stress tensor   
          TSIF=TSIF+virialstress

          !c update energy
          TOTEN=TOTEN+energy  

  1238 FORMAT(/,"  Number of pair interactions contributing to vdW energy:",I8)
  !1239 FORMAT(  "  Estimated vdW energy (eV):",F10.5)
  1239 FORMAT(  "  Edisp (eV):",F11.5)
          IF (IO%IU6>=0) THEN
            IF (.NOT. LEWALD) WRITE(IO%IU6,1238) counter
            WRITE(IO%IU6,1239) energy
          end if

        END SUBROUTINE vdw_forces_ulg

  !! to the following scheme:
  !!
  !!  I\J |  1   2   3   4   .  <br/>
  !!  ----+-------------------- <br/>
  !!  1   |  0                  <br/>
  !!  2   |  1   2              <br/>
  !!  3   |  3   4   5          <br/>
  !!  4   |  6   7   8   9      <br/>
  !!  .   | 10   .   .   .   .  <br/>
  !!
  !! The subroutine finds I,J such that
  !!   K == I*(I-1)/2 + (J-1) and
  !!   1 <= J <= I.
  !! Note that K is zero based while I and J are 1 based.
  SUBROUTINE VDW_PAIR_TO_ARRAY_INDEX(K,I,J)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: K
    INTEGER, INTENT(OUT) :: I, J
    I = FLOOR(0.5 * (SQRT(8.0*K+1.0) + 1.0))
    J = K - I*(I-1)/2 + 1
    ! correct possible precision errors:
    IF (J < 1) THEN
      ! I is too high to satisfy the above equations
      J = J + I - 1
      I = I - 1
    ELSE IF (I < J) THEN
      ! I is too low to satisfy the above equations
      J = J - I
      I = I + 1
    END IF
  END SUBROUTINE

  !! row I and the given iteration plan I0,J0,I1,J1. See PLAN_ITERATION
  !! for details about I0,J0,I0,J0.
  SUBROUTINE VDW_PLAN_ROW(I0, J0, I1, J1, I, J_START, J_END)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: I0, J0, I1, J1, I
    INTEGER, INTENT(OUT) :: J_START, J_END
    IF (I == I0) THEN
      ! Use the start column of the iteration plan if we are in first row
      ! to iterate
      J_START = J0
    ELSE
      ! start from colum 1 otherwise
      J_START = 1
    END IF
    IF (I == I1) THEN
      ! Use the end column of the iteration plan if we are in last row
      ! to iterate
      J_END = J1
    ELSE
      ! end at I otherwise
      J_END = I
    END IF
  END SUBROUTINE

  !! NPROCS denotes the number of total processes.
  !! N is the number of ions to create a pair iteration plan for.
  !! (I0,J0) and (I1,J1) denote the first and last pair respectively to
  !! iterate over.
  !! E.g. (I0,J0),(I1,J1) = (1,1),(3,2) means, that the process is
  !! responsible for the following pairs:
  !! (1,1),(2,1),(2,2),(3,1) & (3,2)
  !! Note that RANK is 0 based.
  !! See \see PLAN_ROW() to determine start and end column for each iterated row.
  !! See \see TEST_PLAN() about how to iterate over each pair.
  SUBROUTINE VDW_PLAN_ITERATION(RANK, NPROCS, N, I0, J0, I1, J1)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: RANK, NPROCS, N
    INTEGER, INTENT(OUT) :: I0,J0, I1,J1
    INTEGER :: NPAIRS, K0, K1
    INTEGER :: MIN_PAIRS_PER_PROC, REMAINING_PAIRS

    ! Determine the total number of interaction pairs
    NPAIRS = N*(N+1) / 2
    ! Each process is at least responsible for NPAIRS / NPROCS pairs
    MIN_PAIRS_PER_PROC = NPAIRS / NPROCS
    ! Assuming each process is responsible for exactly MIN_PAIRS_PER_PROC
    ! at most (NPROCS-1) pairs are remaining
    REMAINING_PAIRS = NPAIRS - NPROCS * MIN_PAIRS_PER_PROC
  !  PRINT "(A,I4,A,I4)", "MIN_PAIRS_PER_PROC =", MIN_PAIRS_PER_PROC, ", REMAINGING_PAIRS =", REMAINING_PAIRS

    ! Next, we calculate the first and the last pair number, K0 and K1,
    ! the given process is responsible for.
    ! See PAIR_TO_ARRAY_INDEX for the enumeration scheme of K0 and K1.
    IF (RANK < REMAINING_PAIRS) THEN
      ! This and all processes with a lower rank must take one
      ! of the remaining pairs in addition to MIN_PAIRS_PER_PROC.
      K0 = (MIN_PAIRS_PER_PROC + 1) * RANK
      K1 = K0 + MIN_PAIRS_PER_PROC
    ELSE
      ! Processes with ranks from 0 to REMAINING_PAIRS-1 take
      ! one of the remaining pairs in addition to MIN_PAIRS_PER_PROC.
      ! Process with ranks from REMAINING_PAIRS to RANK only
      ! take MIN_PAIRS_PER_PROC
      K0 = (MIN_PAIRS_PER_PROC + 1) * REMAINING_PAIRS + &
            MIN_PAIRS_PER_PROC * (RANK - REMAINING_PAIRS)
      K1 = K0 + MIN_PAIRS_PER_PROC - 1
    END IF

    ! Finally convert the pair numbers into array indices.
    CALL VDW_PAIR_TO_ARRAY_INDEX(K0, I0, J0)
    CALL VDW_PAIR_TO_ARRAY_INDEX(K1, I1, J1)
  END SUBROUTINE

  FUNCTION vdw_e_LJ(eps,sigma,r) RESULT(e)
    IMPLICIT NONE
    REAL(q) :: e, f, eps,sigma,r
    f = (sigma/r)**6
    e = 4._q * eps * (f**2 - f)
  END FUNCTION vdw_e_LJ

  FUNCTION vdw_g_LJ(eps,sigma,r) RESULT(g)
    IMPLICIT NONE
    REAL(q) :: g, f, eps,sigma,r
    f = (sigma/r)**6
    g = 24._q * eps * (f - 2._q * f**2) / r
  END FUNCTION vdw_g_LJ

  !! atomic pair i,j.
  SUBROUTINE vdw_pairContrib_LJ(i,j,radius,crit,r12_,eps,sigma,&
  & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: i, j
    REAL(q), INTENT(IN) :: radius
    INTEGER, INTENT(IN) :: crit(3)
    REAL(q), INTENT(IN) :: r12_(3)
    REAL(q), INTENT(IN) :: eps, sigma
    TYPE(type_info), INTENT(IN) :: T_INFO
    TYPE(latt), INTENT(IN) :: LATT_CUR
    INTEGER, INTENT(INOUT) :: counter
    REAL(q), INTENT(INOUT) :: energy,vdw_f2(3, T_INFO%NIONS),virialstress(3,3)

    REAL(q) :: r12(3), gradients, r12norm, ENERGY_AT_CUTOFF
    INTEGER :: ii, jj, kk

    ENERGY_AT_CUTOFF = vdw_e_LJ(eps,sigma,radius)

    DO ii=-crit(1),crit(1)
      DO jj=-crit(2),crit(2)
        DO kk=-crit(3),crit(3)
          r12=r12_+MATMUL((/ii,jj,kk/),TRANSPOSE(LATT_CUR%A))
          r12norm=SUM(r12**2)**0.5
          IF (r12norm>0.5 .AND. r12norm<radius) THEN
            counter=counter+1
            IF (i==j) THEN
              energy=energy+0.5*(vdw_e_LJ(eps,sigma,r12norm) - ENERGY_AT_CUTOFF)
              gradients=0.5*vdw_g_LJ(eps,sigma,r12norm)
            ELSE
              energy=energy+vdw_e_LJ(eps,sigma,r12norm) - ENERGY_AT_CUTOFF
              gradients=vdw_g_LJ(eps,sigma,r12norm)
            END IF
            virialstress=virialstress-gradients*OUTERPROD(3,r12,r12)/r12norm
            IF (i /= j) THEN
              vdw_f2(:,i)=vdw_f2(:,i)+gradients*r12/r12norm
              vdw_f2(:,j)=vdw_f2(:,j)-gradients*r12/r12norm
            END IF
          END IF
        END DO
      END DO
    END DO
  END SUBROUTINE vdw_pairContrib_LJ

  !! The scale of t* depends on the ion mass, the lighter the ions are the shorter is t*
  FUNCTION VDW_SHORTEST_REDUCED_TIMEUNIT(T_INFO)
    IMPLICIT NONE
    TYPE(type_info), INTENT(IN) :: T_INFO
    REAL(q) :: VDW_SHORTEST_REDUCED_TIMEUNIT
    REAL(q) :: MINIMAL_ION_MASS

    MINIMAL_ION_MASS = MINVAL(T_INFO%POMASS) * AMTOKG
    VDW_SHORTEST_REDUCED_TIMEUNIT = SQRT(MINIMAL_ION_MASS / (LJ_EPSILON * EVTOJ)) * LJ_SIGMA * 1E+5
  END FUNCTION

  SUBROUTINE LJ_READER(IO)
    IMPLICIT NONE
    TYPE(in_struct), INTENT(IN) :: IO

    INTEGER :: IDUM, N, IERR
    LOGICAL :: LOPEN, LDUM
    REAL(q) :: RDUM
    COMPLEX(q) :: CDUM
    CHARACTER(1) :: CHARAC

    LOPEN = .FALSE.
    OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LJ_ONLY','=','#',';','F', &
      IDUM,RDUM,CDUM,LJ_ONLY,CHARAC,N,1,IERR)
    IF (IERR==3) LJ_ONLY = LJ_ONLY_DEFAULT
    IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
      LJ_ONLY = LJ_ONLY_DEFAULT
      IF (IO%IU0>=0) &
        WRITE(IO%IU0,*)'Error reading item ''LJ_ONLY'' from file INCAR'
    ENDIF

    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LJ_RADIUS','=','#',';','F', &
      IDUM,LJ_RADIUS,CDUM,LDUM,CHARAC,N,1,IERR)
    IF (IERR==3) LJ_RADIUS = LJ_RADIUS_DEFAULT
    IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
      LJ_RADIUS = LJ_RADIUS_DEFAULT
      IF (IO%IU0>=0) &
        WRITE(IO%IU0,*)'Error reading item ''LJ_RADIUS'' from file INCAR'
    ENDIF
    !c sanity test
    IF (LJ_RADIUS < 1._q) THEN
      IF (IO%IU0>=0) THEN
        WRITE(IO%IU0,*) 'Error: invalid value for LJ_RADIUS:', LJ_RADIUS
        WRITE(IO%IU0,*) 'Lennard-Jones cutoff radius must be at least 1.0 Angstrom!'
      ENDIF
      STOP
    ENDIF

    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LJ_EPSILON','=','#',';','F', &
      IDUM,LJ_EPSILON,CDUM,LDUM,CHARAC,N,1,IERR)
    IF (IERR==3) LJ_EPSILON = LJ_EPSILON_DEFAULT
    IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
      LJ_EPSILON = LJ_EPSILON_DEFAULT
      IF (IO%IU0>=0) &
        WRITE(IO%IU0,*)'Error reading item ''LJ_EPSILON'' from file INCAR'
    ENDIF
    !c sanity test
    IF (LJ_EPSILON < 0._q) THEN
      IF (IO%IU0>=0) THEN
        WRITE(IO%IU0,*) 'Error: invalid value for LJ_EPSILON:', LJ_EPSILON
        WRITE(IO%IU0,*) 'Lennard-Jones binding energy must be a positive!'
      ENDIF
      STOP
    ENDIF

    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LJ_SIGMA','=','#',';','F', &
      IDUM,LJ_SIGMA,CDUM,LDUM,CHARAC,N,1,IERR)
    IF (IERR==3) LJ_SIGMA = LJ_SIGMA
    IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
      LJ_SIGMA = LJ_SIGMA_DEFAULT
      IF (IO%IU0>=0) &
        WRITE(IO%IU0,*)'Error reading item ''LJ_SIGMA'' from file INCAR'
    ENDIF
    !c sanity test
    IF (LJ_SIGMA < 0.1_q .OR. LJ_SIGMA > LJ_RADIUS) THEN
      IF (IO%IU0>=0) THEN
        WRITE(IO%IU0,*) 'Error: invalid value for LJ_SIGMA:', LJ_SIGMA
        WRITE(IO%IU0,*) 'Lennard-Jones zero-potential distance must be in the interval (0.1,LJ_RADIUS)!'
      ENDIF
      STOP
    ENDIF

    CLOSE(IO%IU5)

    ! currently LJ can only be used exclusively:
    IF (LJ_IS_ACTIVE()) LJ_ONLY = .TRUE.
  END SUBROUTINE


  !! ionic forces TIFOR. The stress tensor TSIF and the total energy TOTEL
  !! is updated accordingly.
  !! All other arguments are input arguments.
  SUBROUTINE vdw_forces_LJ(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,historycounter)
    IMPLICIT NONE
    TYPE (grid_3d) GRIDC
    TYPE(in_struct), INTENT(IN) :: IO
    TYPE(latt), INTENT(IN) :: LATT_CUR
    TYPE(dynamics), INTENT(IN) :: DYN
    TYPE(type_info), INTENT(IN) :: T_INFO
    REAL(q), INTENT(INOUT) :: TSIF(3,3), TIFOR(3,T_INFO%NIONS), TOTEN
    INTEGER, INTENT(IN) :: historycounter

    REAL(q) :: vdw_f2(3,T_INFO%NIONS)
    REAL(q) :: energy,virialstress(3,3)
#ifdef DO_TEST
    REAL(q) :: vdw_f2_fd(3,T_INFO%NIONS)
    REAL(q) :: energy1_fd,energy2_fd,virialstress_fd(3,3), fd_stress(3,3)
    REAL(q) :: DIS
    INTEGER :: IDIR,JDIR
    TYPE(latt) :: LATT_FIN
    REAL(q) :: STRESS_ERROR
#endif
    INTEGER :: counter
    INTEGER, SAVE :: I0, J0, I1, J1
    INTEGER :: I, J, J_START, J_END

    INTEGER :: crit(3)
    REAL(q) :: criteria(3)
    REAL(q) :: x(3,T_INFO%NIONS),r12_(3)

    IF (historycounter==1) THEN
#ifdef MPI
      ! create evaluation plan for parallel execution
      CALL VDW_PLAN_ITERATION(GRIDC%COMM%NODE_ME-1, GRIDC%COMM%NCPU, T_INFO%NIONS, &
        & I0, J0, I1, J1)
#else
      ! create evaluation plan for serial execution
      CALL VDW_PLAN_ITERATION(0, 1, T_INFO%NIONS, I0, J0, I1, J1)
#endif

  2000 FORMAT(/,"  Lennard-Jones potential added for vdW interactions",&
              /,"  --------------------------------------------------")
  2001 FORMAT(/,"  LJ_ONLY     = ",L9," , ", A)
  2002 FORMAT("  LJ_RADIUS   = ",F9.3,"  A, cutoff radius")
  2003 FORMAT("  LJ_SIGMA    = ",F9.3,"  A, zero-potential radius")
  2004 FORMAT("  LJ_EPSILON  = ",F9.3," eV, energy minimum")
  2005 FORMAT("  shortest t* = ",F9.3," fs, reduced time t* for the lightest ion")
  2006 FORMAT("  reduced dt* = ",F9.3," t*, timestep in reduced time t* units for the lightest ion")
#ifdef MPI
  2007 FORMAT("  LJ evaluation plan created for ",I3," process(es)")
#endif
      IF (IO%IU6>=0) THEN
        WRITE(IO%IU6,2000)
        IF (LJ_ONLY) THEN
          WRITE(IO%IU6, 2001) LJ_ONLY, "Only Lennard-Jones used. No electronic calculations are done."
        ELSE
          WRITE(IO%IU6, 2001) LJ_ONLY, "Lennard-Jones potentials added to electronic potentials."
        END IF
        WRITE(IO%IU6,2002) LJ_RADIUS
        WRITE(IO%IU6,2003) LJ_SIGMA
        WRITE(IO%IU6,2004) LJ_EPSILON
        WRITE(IO%IU6,2005) VDW_SHORTEST_REDUCED_TIMEUNIT(T_INFO)
        WRITE(IO%IU6,2006) DYN%POTIM / VDW_SHORTEST_REDUCED_TIMEUNIT(T_INFO)
#ifdef MPI
        WRITE(IO%IU6,2007) GRIDC%COMM%NCPU
#endif
      END IF
    END IF

    !c determine the the number of images of simulation cell 
    !c in each direction needed for a given 
    !c cutoff radius
    criteria(1)=LJ_RADIUS*SUM(LATT_CUR%B(:,1)**2)**0.5
    criteria(2)=LJ_RADIUS*SUM(LATT_CUR%B(:,2)**2)**0.5
    criteria(3)=LJ_RADIUS*SUM(LATT_CUR%B(:,3)**2)**0.5
    crit=int(criteria)+1

    !c transform ion positions into cartesian coordinates
    x=MATMUL(LATT_CUR%A,DYN%POSION)

    energy=0.
    vdw_f2=0.
    virialstress=0.
    counter=0

    ! iterate over all (i,j) this process is responsible for
    DO I = I0,I1
      CALL VDW_PLAN_ROW(I0, J0, I1, J1, I, J_START, J_END)
      DO J = J_START,J_END
        r12_=x(:,I)-x(:,J)
        ! invoke the pair contribution subroutine
        CALL vdw_pairContrib_LJ(I,J,LJ_RADIUS,crit,r12_,LJ_EPSILON,LJ_SIGMA,&
          & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
      END DO
    END DO

#ifdef MPI
    CALL M_sum_d(GRIDC%COMM, energy, 1)
    CALL M_sum_d(GRIDC%COMM, vdw_f2, 3*T_INFO%NIONS)
    CALL M_sum_d(GRIDC%COMM, virialstress, 3*3)
    CALL M_sum_i(GRIDC%COMM, counter, 1)
#endif

    !c update forces
    TIFOR=TIFOR-vdw_f2
    !c update stress tensor

#ifdef DO_TEST
!-----------------------------------------------------------------------------
    DIS=fd_displacement
    DO IDIR=1,3
      DO JDIR=1,3
        LATT_FIN=LATT_CUR
        !  all directions
        DO I=1,3
          LATT_FIN%A(IDIR,I)=LATT_CUR%A(IDIR,I)+DIS*LATT_CUR%A(JDIR,I)
        ENDDO
        CALL LATTIC(LATT_FIN)
        x=MATMUL(LATT_FIN%A,DYN%POSION)

        energy1_fd=0.
        vdw_f2_fd=0.
        virialstress_fd=0.
        DO I = I0,I1
          CALL VDW_PLAN_ROW(I0, J0, I1, J1, I, J_START, J_END)
          DO J = J_START,J_END
            r12_=x(:,I)-x(:,J)
                                    ! invoke the pair contribution subroutine
            CALL vdw_pairContrib_LJ(I,J,LJ_RADIUS,crit,r12_,LJ_EPSILON,LJ_SIGMA,&
             & T_INFO,LATT_FIN,counter,energy1_fd,vdw_f2_fd,virialstress_fd)
          END DO
        END DO
#ifdef MPI
              CALL M_sum_d(GRIDC%COMM, energy1_fd, 1)
#endif

        LATT_FIN=LATT_CUR
        !  all directions
        DO I=1,3
          LATT_FIN%A(IDIR,I)=LATT_CUR%A(IDIR,I)-DIS*LATT_CUR%A(JDIR,I)
        ENDDO
        CALL LATTIC(LATT_FIN)
        x=MATMUL(LATT_FIN%A,DYN%POSION)

        energy2_fd=0.
        vdw_f2_fd=0.
        virialstress_fd=0.
        DO I = I0,I1
          CALL VDW_PLAN_ROW(I0, J0, I1, J1, I, J_START, J_END)
          DO J = J_START,J_END
            r12_=x(:,I)-x(:,J)
            ! invoke the pair contribution subroutine
            CALL vdw_pairContrib_LJ(I,J,LJ_RADIUS,crit,r12_,LJ_EPSILON,LJ_SIGMA,&
             & T_INFO,LATT_FIN,counter,energy2_fd,vdw_f2_fd,virialstress_fd)
          END DO
        END DO
#ifdef MPI
        CALL M_sum_d(GRIDC%COMM, energy2_fd, 1)
#endif

        fd_stress(IDIR,JDIR)=(energy2_fd-energy1_fd)/2/DIS
      ENDDO
    ENDDO
    STRESS_ERROR = SUM((virialstress-fd_stress)**2)**0.5
    WRITE(*,'(A,F14.7)') 'Distance(virialstress, finite difference stress)=', STRESS_ERROR
    IF (STRESS_ERROR > 1E-3) THEN
      WRITE(*,'(A)') "virial stress:"
      WRITE(*,'(3F14.7)') virialstress
      WRITE(*,'(A)') "finite difference stress:"
      WRITE(*,'(3F14.7)') fd_stress
    END IF
!    STOP
!-----------------------------------------------------------------------------
#endif
    TSIF=TSIF+virialstress
    !c update energy
    TOTEN=TOTEN+energy  

  2010 FORMAT(/,"  Number of pair interactions contributing to vdW energy:",I8)
  2011 FORMAT(  "  Estimated vdW energy (eV):",F10.5)
    IF (IO%IU6>=0) THEN
      WRITE(IO%IU6,2010) counter
      WRITE(IO%IU6,2011) energy
    END IF
  END SUBROUTINE vdw_forces_LJ


        SUBROUTINE vdw_forces_TS(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,relvol_,REFSTATE,IVDW)
        !c Tkatchenko-Scheffler method and its variants
          USE setexm
          TYPE (grid_3d) GRIDC
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),SAVE :: atrad=50._q,SCSRAD=120._q
          REAL(q) :: gradients
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),TOTEN
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: energy,ccc,rrr,virialstress(3,3)
          REAL(q) :: cc(71),r0(71),alpha(71)
          CHARACTER (LEN=2),SAVE :: tags(71)
          REAL(q), PARAMETER :: conversion=10.364425449557316
          REAL(q), PARAMETER :: conversion2=0.057652660443590298
          REAL(q),SAVE :: sR=0.94_q,dfactor=20._q,s6=1.00_q
          REAL(q) :: userC(T_INFO%NTYP),userC2(T_INFO%NIONS)     
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER :: i,j,ii,jj,kk,counter,indx1,indx2
          INTEGER :: historycounter
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          integer :: crit(3),crit1(3),crit2(3),crit3(3)
          real(q) :: criteria(3)
          real(q) :: x(3,T_INFO%NIONS),x1(3),x2(3),r12(3),r12_(3)
          real(q) :: r12norm
          real(q) :: cc_2(T_INFO%NIONS),r0_2(T_INFO%NIONS),alpha_2(T_INFO%NIONS)
          real(q) :: r0TS(T_INFO%NIONS),alphaTS(T_INFO%NIONS)
          real(q) :: alphaTSscreened(T_INFO%NIONS),c6TSscreened(T_INFO%NIONS),r0TSscreened(T_INFO%NIONS)
          real(q),ALLOCATABLE,SAVE :: cc_(:),r0_(:),alpha_(:)
          real(q) :: relvol_(T_INFO%NIONS)
          integer :: dfnd1(T_INFO%NTYP),dfnd2(T_INFO%NTYP),dfnd3(T_INFO%NIONS)
          LOGICAL,SAVE :: LVDWSCS=.FALSE.,LSCALER0=.TRUE.,LSCSGRAD=.TRUE.
          LOGICAL,ALLOCATABLE,SAVE :: LVDW_SAMETYPE(:)
          LOGICAL,DIMENSION(3),SAVE :: LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)
          LOGICAL,SAVE :: LCFDM=.FALSE.
          INTEGER,SAVE :: MBD_SIZE(3)
          REAL(q) :: dcA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q) :: dalphaA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q),PARAMETER :: dstep=1e-2
          REAL(q) :: dcAB(3,T_INFO%NIONS+3),extragrad(3,T_INFO%NIONS+3),extragradDum(3,T_INFO%NIONS+3)
          REAL(q) :: drrr(3,T_INFO%NIONS+3), drrrA(3,T_INFO%NIONS+3), drrrB(3,T_INFO%NIONS+3)
          INTEGER :: REFSTATE(T_INFO%NTYP)
          REAL(q),SAVE :: MBD_BETA=2.56 !range separating parameter for CFMD
          INTEGER,SAVE :: IDAMPF=0
          REAL(q) :: mu, Rp,Rq
          REAL(q) :: energy1,energy2,energy3,energyDum
          REAL(q) :: vdw_f21(3,T_INFO%NIONS),vdw_f22(3,T_INFO%NIONS),vdw_f23(3,T_INFO%NIONS),vdw_f2Dum(3,T_INFO%NIONS)
          REAL(q) :: virialstress1(3,3),virialstress2(3,3),virialstress3(3,3),virialstressDum(3,3)
          REAL(q) :: theta,amax,bmax
          LOGICAL, SAVE :: LEWALD=.FALSE.
          !LOGICAL, SAVE :: LEWALD=.TRUE.
          INTEGER :: IVDW
          LOGICAL :: LUSERPARAM
          
         

  1234 FORMAT(/,"  Tkatchenko/Scheffler method for vdW energy calculation",&
              /,"  ---------------------------------------------------------------------------------------------")
  1230 FORMAT(/,"  Atomic reference data used in the T-S method for vdW correction:",&
              /,"            C6at              R0at           ALPHAat      REFSTATE",&
              /,"            (au)              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")
  1235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6ts              R0ts           ALPHAts        RELVOL",&
              /,"            (au)              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")
  
  2235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6ts              R0ts           ALPHAts           C6scs          R0scs        ALPHAscs        RELVOL ",&
              /,"            (au)              (au)            (au)              (au)           (au)          (au)                 ",&
              /,"   -----------------------------------------------------------------------------------------------------------------")
  1236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  12360 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,I2)
  2236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  2250 FORMAT(/,"  IVDW = ",I3)
  1237 FORMAT("  VDW_RADIUS = ",F9.3," A")
  2237 FORMAT("  SCSRAD = ",F9.3," A")
  1233 FORMAT("  VDW_SR = ",F9.3)
  1241 FORMAT("  VDW_S6 = ",F9.3)
  1240 FORMAT("  VDW_D = ",F9.3)
  2241 FORMAT("  LSCALER0 = ",L1)
  2242 FORMAT("  LSCSGRAD = ",L1)     
  2243 FORMAT("  LVDW_ONECELL = ",L1,1X,L1,1X,L1)
  2244 FORMAT("  MBD_BETA = ",F9.3) 
  2245 FORMAT("  VDW_IDAMPF = ",I2) 
  2246 FORMAT("  LVDW_EWALD = ",L1)
  2247 FORMAT("  LVDW_SAMETYPE = ")
  2248 FORMAT("  VDW_MBD_SIZE = ",I2,X,I2,X,I2)
  2249 FORMAT("  LCFDM = ",L1)

          if (historycounter==1) then


            tags=(/'H ','Be','Mg','Ca','Sr','Ba',&
                   'He','Ne','Ar','Kr','Xe','Rn',&
                   'Li','Na','K ','Rb','Cs',&
                   'B ','Al','Ga','In','Tl',&
                   'C ','Si','Ge','Sn','Pb',&
                   'N ','P ','As','Sb','Bi',&
                   'O ','S ','Se','Te','Po',&
                   'F ','Cl','Br','I ','At',&
                   'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',&
                   'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',&            
                   'Hf','Ta','W ','Re','Os',  'Ir','Pt','Au','Hg'/)
            cc=(/6.50,214.,627.,2221.,3170.,5727.0,&
                 1.46,6.38,64.3,129.6,285.90,390.63,&
                 1387.,1556.,3897.,4691.,6582.08,&
                 99.5,528.,498.,707.05,717.44,&
                 46.6,305.,354.,587.42,697.0,&
                 24.2,185.,246.,459.322,571.0,&
                 15.6,134.,210.,396.,530.92,&
                 9.52,94.6,162.,385.,457.53,&
                 1383.,1044.,832.,602.,552.,482.,408.,373.,253.,284.,&
                 1968.58, 1677.91,1263.61,1028.73,1390.87,609.75,469.0,157.5,339.0,452.0, &
                 1274.8,1019.92,847.93,710.2,596.67,359.1,347.1,298.0,392.0/)
            cc=cc*conversion2
            alpha=(/4.50,38.,71.,160.,199.,275.0,&
                   1.38,2.67,11.1,16.8,27.30,33.54,&
                   164.2,162.7,292.9,319.2,427.12,&
                   21.0,60.,60.,70.22,69.92,&
                   12.0,37.,41.,55.95,61.8,&
                   7.4,25.,29.,43.67,49.02,&
                   5.4,19.6,25.,37.65,45.013,&
                   3.8,15.,20.,35.,38.93,&
                   120.,98.,84.,78.,63.,56.,50.,48.,42.,40.,&
                   126.737,119.97,101.603,88.42,80.08,65.90,56.1,23.68,50.6,39.7,&
                   99.52,82.53,71.041,63.04,55.055,42.51,39.68,36.5,33.9/)
            r0=(/1.64,2.21,2.26,2.46,2.40,2.52,&
                 1.40,1.54,1.88,2.02,2.16,2.24,&
                 2.20,1.97,1.96,1.97,2.00,&
                 2.06,2.29,2.22,2.24,2.07,&
                 1.90,2.22,2.22,2.28,2.28,&
                 1.77,2.12,2.17,2.26,2.29,&
                 1.69,2.04,2.14,2.23,2.17,&
                 1.61,1.96,2.08,2.21,2.15,&
                 2.43,2.39,2.35,2.11,2.10,2.24,2.21,2.02,1.99,2.13,&
                 2.55,2.40,2.24,2.17,2.16,2.11,2.09,1.94,2.02,2.11,&
                 2.23,2.20,2.16,2.13,2.03,2.12,2.07,2.04,2.11/)

            ALLOCATE(cc_(T_INFO%NTYP),r0_(T_INFO%NTYP),alpha_(T_INFO%NTYP))
            ALLOCATE(LVDW_SAMETYPE(T_INFO%NTYP))
            cc_2=0._q;r0_2=0._q;alpha_2=0._q
            cc_=0._q;r0_=0._q;alpha_=0._q
            LVDW_SAMETYPE=.TRUE.
            dfnd1=0;dfnd2=0
            DO i=1,T_INFO%NTYP
              indx1=cwhereisit(tags,ELEM(i),SIZE(tags))
              IF (indx1>0) THEN
                dfnd1(i)=1
                dfnd2(i)=1
                cc_(i)=cc(indx1)
                r0_(i)=r0(indx1)
                alpha_(i)=alpha(indx1)
              END IF
            ENDDO

           !c read in the input parameters
            LOPEN=.FALSE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

              !c is self-consistent screening to be taken into account?
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDWSCS','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDWSCS,CHARAC,N,1,IERR)
              IF (IERR==3) LVDWSCS=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 LVDWSCS=.FALSE.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDWSCS'' from file INCAR'
              ENDIF           

              !c is the coupled fluctuating dipoles model to be used?
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LCFDM','=','#',';','L', &
                  IDUM,RDUM,CDUM,LCFDM,CHARAC,N,1,IERR)
              IF (IERR==3) LCFDM=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 LCFDM=.FALSE.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LCFDM'' from file INCAR'
              ENDIF

              !c type of damping function 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_IDAMPF','=','#',';','I', &
                    IDAMPF,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) IDAMPF=0 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                IDAMPF=0  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_IDAMPF'' from file INCAR'
              ENDIF
              !c sanity test
              IF (IDAMPF <0 .OR. IDAMPF>1) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_SR:',sR
                      WRITE(IO%IU0,*) 'VDW_IDAMPF must be set to O or 1 !'
                ENDIF
                STOP
              ENDIF

              !c cutoff radius for vdW energy
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_RADIUS','=','#',';','F', &
                    IDUM,atrad,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) atrad=50._q 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 atrad=50._q  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_RADIUS'' from file INCAR'
              ENDIF
              !c sanity test
              IF (atrad <1._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_RADIUS:',atrad
                      WRITE(IO%IU0,*) 'VDW_RADIUS must be greater than 1.0 Angstrom!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR('VDW_RADIUS','F',IDUM,atrad,CDUM,.TRUE.,CHARAC,1)

              !c use Ewald's summation?
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_EWALD','=','#',';','L', &
                  IDUM,RDUM,CDUM,LEWALD,CHARAC,N,1,IERR)
              IF (IERR==3) LEWALD=.FALSE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LEWALD=.FALSE.  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LEWALD'' from file INCAR'
              ENDIF

              !c s6 parameter for damping function - usualy not needed in TS method
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_S6','=','#',';','F', &
                    IDUM,s6,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) s6=1.00 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN  
                 s6=1.00
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_S6'' from file INCAR'
              ENDIF
              !c sanity test
              IF (s6 <0._q .OR. s6>10._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_S6:',s6
                      WRITE(IO%IU0,*) 'VDW_S6 must be a value from interval (0,10)!'
                ENDIF
                STOP
              ENDIF

              LUSERPARAM=.FALSE.
              !c sR parameter for damping function
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_SR','=','#',';','F', &
                    IDUM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
              IF ((IERR==0) .AND. (N .GE. 1)) THEN
                LUSERPARAM=.TRUE.
                sR=RDUM
              ENDIF
              IF (IERR==3) LUSERPARAM=.FALSE.
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LUSERPARAM=.FALSE.
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_SR'' from file INCAR'
              ENDIF
          
              !c sanity test for the user-defined value of sR
              IF (LUSERPARAM) THEN 
                IF (sR <0._q .OR. sR>10._q) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'Error: invalid value for VDW_SR:',sR
                    WRITE(IO%IU0,*) 'VDW_SR must be a value from interval (0,10)!'
                  ENDIF
                  STOP
                ENDIF
              !c ...or use default values
              ELSE
                !c we have default s6 only for PBE
                !c Calculation has to be terminated if other functional is used and the 
                !c user didn't provide the sR value,  
                IF (.NOT.(LEXCH==8) ) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'vdw_forces_TS: ERROR unsupported xc-functional, LEXCH=',LEXCH
                    WRITE(IO%IU0,*) 'please define parameter VDW_SR for this functional'
                  ENDIF
                  STOP
                ELSE
                  !c defaults for different versions of TS method
                  SELECT CASE(IVDW) 
                    !c standard TS, SCS has different default!!!
                    CASE(2,20)                   
                      IF (LVDWSCS) THEN
                        sR=0.97
                      ELSE
                        sR=0.94
                      ENDIF
                    !c iterative Hirshfeld partitioning
                    CASE(21)
                      sR=0.95
                    CASE(22)
                      sR=0.95
                  END SELECT
                ENDIF
              ENDIF

              !CALL XML_INCAR('VDW_SR','F',IDUM,sR,CDUM,.TRUE.,CHARAC,1)

              !c damping parameter d 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_D','=','#',';','F', &
                    IDUM,dfactor,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) dfactor=20. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 dfactor=20.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_D'' from file INCAR'
              ENDIF
              !c sanity test
              IF (dfactor <1._q .OR. dfactor>100._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_D:',dfactor
                      WRITE(IO%IU0,*) 'VDW_D must be a value from interval (1,100)!'
                ENDIF
                STOP
              ENDIF
              !CALL XML_INCAR('VDW_D','F',IDUM,dfactor,CDUM,.TRUE.,CHARAC,1)

              !c user-provided C6 parameters in J*nm^6*mol^-1
              LUSERPARAM=.FALSE.
              userC=0._q
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_C6'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                LUSERPARAM=.TRUE.
                dfnd1=1
                cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)
              ENDIF

              !c if C6 in J*nm^6*mol^-1 is not found, try to read C6 in au 
              IF (.NOT. LUSERPARAM) THEN
                userC=0._q
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6au','=','#',';','F', &
                &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*)'Error reading item ''VDW_C6au'' from file INCAR.'
                    WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                  ENDIF
                  STOP
                ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                  dfnd1=1
                  cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)*conversion2
                ENDIF
              ENDIF
 
              !c now its time to make basic sanity test of C6 values
              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: missing parameters C6 for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_C6 or VDW_C6au to define C6 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_C6','F',IDUM,cc_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

             !c user-provided parameter R0 (in A) for the damping function 
              LUSERPARAM=.FALSE.
              userC=0._q
              dfnd1=dfnd2
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_R0'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                LUSERPARAM=.TRUE.
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 .AND. userC(j)<10._q) THEN
                    dfnd1(j)=1
                    r0_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_R0 must be a value from interval (0.1,10)!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF

              !c if R0 in A is not found, try to read  R0 (in au) 
              IF (.NOT. LUSERPARAM) THEN
                userC=0._q
                dfnd1=dfnd2
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0au','=','#',';','F', &
                &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*)'Error reading item ''VDW_R0au'' from file INCAR.'
                    WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                  ENDIF
                  STOP
                ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                  DO j=1,T_INFO%NTYP
                    IF (userC(j)>0.2 .AND. userC(j)<20._q) THEN
                      dfnd1(j)=1
                      r0_(j)=userC(j)*AUTOA
                    ELSE
                      IF (IO%IU0>=0) THEN
                        WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0au:',userC(j)
                        WRITE(IO%IU0,*) 'All values for VDW_R0au must be a value from interval (0.2,20)!'
                      ENDIF
                      STOP
                    END IF
                  END DO
                ENDIF
              ENDIF

              !c check if we have R0 for each atomic type defined in POSCAR
              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters R0 are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_R0 to define R0 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_R0','F',IDUM,r0_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              !c user-provided polarizabilities
              dfnd1=dfnd2
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_ALPHA','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_ALPHA'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                dfnd1=0
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 ) THEN
                    dfnd1(j)=1
                    alpha_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_ALPHA:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_ALPHA must be greater than 0.1 au!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF

              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters ALPHA are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_ALPHA to define ALPHA for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_ALPHA','F',IDUM,alpha_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              !c should the images of the original cell made by translating 
              !c along directions of lattice vectors be neglected? 
              !c (this is useful if one performs calculations of isolated molecule (LVDW_ONECELL= T T T)
              !c or a surface (LVDW_ONECELL= F F T)) 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_ONECELL','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW_ONECELL,CHARAC,N,3,IERR)
              IF (IERR==3) LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./) 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN
                 LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDW_ONECELL'' from file INCAR'
              ENDIF

              !c this parameter can be used to switch off the interactions 
              !c between the same type of atoms - this can be useful in some 
              !c special cases...
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_SAMETYPE','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW_SAMETYPE,CHARAC,N,T_INFO%NTYP,IERR)
              IF (IERR==3) LVDW_SAMETYPE=.TRUE. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                 LVDW_SAMETYPE=.TRUE.
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDW_SAMETYPE'' from file INCAR'
              ENDIF

              !c this setting is specific for coupled-fluctuating-dipoles model
              IF (LCFDM) THEN

                !c damping parameter
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'MBD_BETA','=','#',';','F', &
                  IDUM,MBD_BETA,CDUM,LDUM,CHARAC,N,1,IERR)
                IF (IERR==3) MBD_BETA=2.56 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                  MBD_BETA=2.56
                  IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''MBD_BETA'' from file INCAR'
                ENDIF

                !c this parameter controls the size of Hammiltonian for 
                !c system under periodic boundary conditions
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_MBD_SIZE','=','#',';','I', &
                    MBD_SIZE,RDUM,CDUM,LDUM,CHARAC,N,3,IERR)
                IF (IERR==3) MBD_SIZE=1 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN
                  MBD_SIZE=1
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''VDW_MBD_SIZE'' from file INCAR'
                ENDIF
                !c sanity test
                DO i=1,3
                  IF (MBD_SIZE(i)<1) THEN 
                    WRITE(IO%IU0,*)'Invalid value for ''VDW_MBD_SIZE'' read from file INCAR'
                    WRITE(IO%IU0,*)' ''VDW_MBD_SIZE'' is set to 1 1 1'
                    MBD_SIZE=1
                  ENDIF
                ENDDO
                
              ENDIF

              !c alternative damping function not available for TS-SCS
              IF (IDAMPF==1) LVDWSCS=.FALSE.

              !c this setting is specific for TS+SCS method
              IF (LVDWSCS) THEN

                !c should the R0 parameters (damping function) be rescaled 
                !c within the SCS calculation?
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'LSCALER0','=','#',';','L', &
                  IDUM,RDUM,CDUM,LSCALER0,CHARAC,N,1,IERR)
                IF (IERR==3) LSCALER0=.TRUE. 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   LSCALER0=.TRUE.  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''LSCALER0'' from file INCAR'
                ENDIF

                !c should the gradients be computed in the TS+SCS calculation?
                !c ...might be pretty slow for the large systems
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'LSCSGRAD','=','#',';','L', &
                  IDUM,RDUM,CDUM,LSCSGRAD,CHARAC,N,1,IERR)
                IF (IERR==3) LSCSGRAD=.TRUE. 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   LSCSGRAD=.TRUE.  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''LSCSGRAD'' from file INCAR'
                ENDIF

                !c cutoff radius for calculation of dipole-dipole interaction
                !c tensor - converges slowly so large value is needed
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'SCSRAD','=','#',';','F', &
                      IDUM,SCSRAD,CDUM,LDUM,CHARAC,N,1,IERR)
                IF (IERR==3) SCSRAD=120._q 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   SCSRAD=120._q  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''SCSRAD'' from file INCAR'
                ENDIF
                !c sanity test
                IF (SCSRAD <0.) THEN
                  IF (IO%IU0>=0) THEN
                        WRITE(IO%IU0,*) 'Error: invalid value for SCSRAD:',SCSRAD
                        WRITE(IO%IU0,*) 'SCSRAD must be a positive number!'
                  ENDIF
                  STOP
                ENDIF
                IF (SCSRAD==0._q) LVDWSCS=.FALSE.

!                 CALL RDATAB(LOPEN,INCAR,IO%IU5,'LCFDM','=','#',';','L', &
!                   IDUM,RDUM,CDUM,LCFDM,CHARAC,N,1,IERR)
!                 IF (IERR==3) LCFDM=.FALSE. 
!                 IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
!                    LCFDM=.FALSE.  
!                   IF (IO%IU0>=0) &
!                     WRITE(IO%IU0,*)'Error reading item ''LCFDM'' from file INCAR'
!                 ENDIF

!                 IF (LCFDM) THEN
!                   CALL RDATAB(LOPEN,INCAR,IO%IU5,'MBD_BETA','=','#',';','F', &
!                     IDUM,MBD_BETA,CDUM,LDUM,CHARAC,N,1,IERR)
!                   IF (IERR==3) MBD_BETA=2.56 
!                   IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
!                     MBD_BETA=2.56
!                     IF (IO%IU0>=0) &
!                     WRITE(IO%IU0,*)'Error reading item ''MBD_BETA'' from file INCAR'
!                   ENDIF
!                 ENDIF
              ENDIF

              !IF (IDAMPF==1) LEWALD=.FALSE.
            CLOSE(IO%IU5)
            
            !c write down the atomic reference data
            IF (IO%IU6>=0) THEN
                WRITE(IO%IU6,1230)
                IDUM=0
                DO i=1,T_INFO%NTYP
                  IDUM=IDUM+T_INFO%NITYP(i)
                  WRITE(IO%IU6,12360) ELEM(T_INFO%ITYP(idum)),cc_(T_INFO%ITYP(idum))/conversion2,&
                  & r0_(T_INFO%ITYP(idum))/AUTOA,alpha_(T_INFO%ITYP(idum)),REFSTATE(T_INFO%ITYP(idum))
                ENDDO
             ENDIF
           end if 

            !c basic sanity test + rescale the parameters 
            !c using the Hirshfeld volumes
            dfnd3=0
            !write(*,*) 'dfnd3a',dfnd3(1),dfnd3(2),dfnd3(3)
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              !write(*,*) 'sanity',i,indx1,cc_(indx1),alpha_(indx1),r0_(indx1),relvol_(i)
              !IF (cc_(indx1)>0._q .AND. alpha_(indx1)>0._q .AND. r0_(indx1)>0._q .AND. relvol_(i)>1.e-5) THEN
              IF (alpha_(indx1)>0._q .AND. r0_(indx1)>0._q .AND. relvol_(i)>1.e-5) THEN
                dfnd3(i)=1
                cc_2(i)=cc_(indx1)*relvol_(i)**2
                r0_2(i)=r0_(indx1)*relvol_(i)**0.3333333333
                alpha_2(i)=alpha_(indx1)*relvol_(i)
              ENDIF
            ENDDO

            !write(*,*) 'dfnd3b',dfnd3(1),dfnd3(2),dfnd3(3)

            !write(*,*) 'fdf',SUM(dfnd3), T_INFO%NIONS
            IF (SUM(dfnd3) .LT. T_INFO%NIONS) THEN
              IF (IO%IU0>=0) THEN
                WRITE(IO%IU0,*) 'Error: some force-field parameter for the following atom is not defined:'
                DO j=1,T_INFO%NIONS
                  IF (dfnd3(j)==0) THEN
                    WRITE(IO%IU0,*) ELEM(T_INFO%ITYP(j)),j
                  ENDIF
                ENDDO
              ENDIF
              STOP
            ENDIF

          IF (LVDWSCS) THEN
            !c convert C6 to eV A**6
            cc_2=cc_2*conversion
 
            !c compute screened C6, alpha, and r0
            CALL vdw_tsscs(GRIDC,alpha_2,cc_2,r0_2,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
            & dalphaA,c6TSscreened,dcA,r0TSscreened,SCSRAD,LSCALER0,LSCSGRAD,LCFDM,LVDW_ONECELL,&
            & TOTEN,MBD_BETA,MBD_SIZE,IO%IU0)


            !!!CALL vdw_tsscs_range_separated(GRIDC,alpha_2,cc_2,r0_2,dfactor,sR,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,dalphaA,&
            !!!& c6TSscreened, dcA,r0TSscreened,SCSRAD,LSCALER0,LSCSGRAD,LCFDM,LVDW_ONECELL,TOTEN,MBD_BETA,MBD_SIZE,IO%IU0)


! #ifdef debug
!             IF (IO%IU0>=0) THEN
!               DO i=1,T_INFO%NIONS
!                 write(*,*) '------------------------------'
!                 write(*,*) 'dC/dx, atom',i
!                 DO j=1,T_INFO%NIONS+3
!                   write(*,*) dca(i,1,j),dca(i,2,j),dca(i,3,j)
!                 ENDDO
!               ENDDO
!             ENDIF
! #endif

            !c convert C6 to Jnm^6/mol
            !cc_2=cc_2/conversion
            !C6TSscreened=C6TSscreened/conversion
            !c convert C6 to au
            !c write some output here
            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)
              WRITE(IO%IU6,2235) 
              DO i=1,T_INFO%NIONS
                WRITE(IO%IU6,2236) ELEM(T_INFO%ITYP(i)),cc_2(i)/(2*RYTOEV*AUTOA**6),r0_2(i)/AUTOA,alpha_2(i),&
                & C6TSscreened(i)/(2*RYTOEV*AUTOA**6),r0TSscreened(i)/AUTOA,alphaTSscreened(i),relvol_(i)
              ENDDO
              if (historycounter==1) then
                WRITE(IO%IU6,2250) IVDW
                WRITE(IO%IU6,1237) atrad
                WRITE(IO%IU6,2237) SCSRAD
                WRITE(IO%IU6,1241) s6
                WRITE(IO%IU6,1233) sR                
                WRITE(IO%IU6,1240) dfactor
                WRITE(IO%IU6,2241) LSCALER0
                WRITE(IO%IU6,2242) LSCSGRAD
                IF (.NOT. LEWALD) THEN 
                  WRITE(IO%IU6,2243) LVDW_ONECELL(1),LVDW_ONECELL(2),LVDW_ONECELL(3)
                  !WRITE(IO%IU6,ADVANCE='NO',FMT='(3X,A16)') "LVDW_SAMETYPE = "
                  WRITE(IO%IU6,ADVANCE='NO',FMT='(A16)') "  LVDW_SAMETYPE = "
                  DO i=1,T_INFO%NTYP
                    IF (i<T_INFO%NTYP) THEN
                      WRITE(IO%IU6,ADVANCE='NO',FMT='(L1,1X)') LVDW_SAMETYPE(i)
                    ELSE
                      WRITE(IO%IU6,ADVANCE='YES',FMT='(L1,1X)') LVDW_SAMETYPE(i)
                    ENDIF
                  ENDDO
                ENDIF
                WRITE(IO%IU6,2246) LEWALD
                IF (LCFDM) THEN
                  WRITE(IO%IU6,2249) LCFDM
                  WRITE(IO%IU6,2244) MBD_BETA
                  WRITE(IO%IU6,2248) MBD_SIZE(1),MBD_SIZE(2),MBD_SIZE(3)
                ENDIF
              endif
            ENDIF

            !c final conversion
            r0TS=sR*r0_2
            alphaTS=alpha_2
            r0_2=sR*r0TSscreened
            alpha_2=alphaTSscreened
            !cc_2=C6TSscreened*conversion
            cc_2=s6*C6TSscreened
          ELSE

            cc_2=cc_2*conversion   
            
            IF (LCFDM) THEN  
              call vdw_mbd(GRIDC,alpha_2,cc_2,r0_2,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
              & dalphaA,c6TSscreened,dcA,r0TSscreened,SCSRAD,LSCALER0,LSCSGRAD,LCFDM,LVDW_ONECELL,&
              & TOTEN,MBD_BETA,IO%IU0)
               !CALL vdw_cfdm2(GRIDC,T_INFO%NIONS,DYN%POSION,TRANSPOSE(LATT_CUR%A)/AUTOA,alpha_2, &
               !&   c6TS/(2*RYTOEV*AUTOA**6),rmax/AUTOA,translations,ECFDM,BETA,IU0)
            ENDIF
   
            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)              
              WRITE(IO%IU6,1235) 
              DO i=1,T_INFO%NIONS
                WRITE(IO%IU6,1236) ELEM(T_INFO%ITYP(i)),cc_2(i)/(2*RYTOEV*AUTOA**6),&
                & r0_2(i)/AUTOA,alpha_2(i),relvol_(i)
              ENDDO
              if (historycounter==1) then
                WRITE(IO%IU6,2250) IVDW
                WRITE(IO%IU6,1237) atrad
                WRITE(IO%IU6,2245) IDAMPF
                WRITE(IO%IU6,1241) s6
                WRITE(IO%IU6,1233) sR
                WRITE(IO%IU6,1240) dfactor
                IF (.NOT. LEWALD) THEN 
                  WRITE(IO%IU6,2243) LVDW_ONECELL(1),LVDW_ONECELL(2),LVDW_ONECELL(3)
                  WRITE(IO%IU6,ADVANCE='NO',FMT='(A16)') "  LVDW_SAMETYPE = "
                  DO i=1,T_INFO%NTYP
                    IF (i<T_INFO%NTYP) THEN
                      WRITE(IO%IU6,ADVANCE='NO',FMT='(L1,1X)') LVDW_SAMETYPE(i)
                    ELSE
                      WRITE(IO%IU6,ADVANCE='YES',FMT='(L1,1X)') LVDW_SAMETYPE(i)
                    ENDIF
                  ENDDO
                ENDIF
                IF (LCFDM) WRITE(IO%IU6,2244) MBD_BETA
                WRITE(IO%IU6,2246) LEWALD
              end if 
            end if
            r0_2=sR*r0_2
            cc_2=cc_2*s6
            !cc_2=cc_2*conversion
          ENDIF
          !end if

          !c determine the number of cell images 
          !c needed for each direction 
          criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
          criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
          criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
          crit=int(criteria)+1
          
          !c if desired, switch off the interaction with cell 
          !c images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              criteria(i)=0
              crit(i)=0
            ENDIF
          ENDDO


          x=MATMUL(LATT_CUR%A,DYN%POSION)

          energy=0.
          gradients=0.
          vdw_f2=0._q;extragrad=0._q
          virialstress=0.
          counter=0

          IF (.NOT. LEWALD) THEN
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              IF (IDAMPF==1) THEN
                Rp=(SQRT(2._q/PI)*alpha_2(i)/3.)**(1./3.)
              ENDIF

              IF (LVDWSCS .AND. LSCSGRAD ) THEN
                CALL gradient_RA(T_INFO%NIONS,drrrA,r0TS(i),alphaTS(i),alpha_2(i),dalphaA(i,:,:),LSCALER0)
              ENDIF

              DO j=i,T_INFO%NIONS              
                indx2=T_INFO%ITYP(j)
       
                !c if desired by user, interaction between the same atomic types
                !c is avoided
                LDUM=.TRUE.
                !IF (indx1 .EQ. indx2) LDUM=.FALSE.
                !IF (LVDW_SAMETYPE(indx1)) LDUM=.TRUE.
                IF ((indx1 .EQ. indx2) .AND. (.NOT. LVDW_SAMETYPE(indx1))) LDUM=.FALSE.

                r12_=DYN%POSION(:,i)-DYN%POSION(:,j)
                CALL min_imageV(3,r12_)
                r12_=MATMUL(LATT_CUR%A,r12_)
                !indx2=T_INFO%ITYP(j)
                ccc=combination_ruleTS(cc_2(i),cc_2(j),alpha_2(i),alpha_2(j)) 
                !ccc=ccc*conversion
                rrr=r0_2(i)+r0_2(j)
               
                RDUM=energy
                IF (rrr>0._q .AND. LDUM) THEN
                  IF (LVDWSCS .AND. LSCSGRAD) THEN
                    CALL gradient_C(T_INFO%NIONS,cc_2(i),cc_2(j),dcAB,alpha_2(i),alpha_2(j),&
                    & dcA(i,:,:),dcA(j,:,:),dalphaA(i,:,:),dalphaA(j,:,:))  
                    CALL gradient_RA(T_INFO%NIONS,drrrB,r0TS(j),alphaTS(j),alpha_2(j),dalphaA(j,:,:),LSCALER0)
                    drrr=drrrA+drrrB    
                    CALL vdw_pairContribSCS(i,j,atrad,crit,r12_,ccc,dcAB,rrr,drrr,dfactor,&
                       & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress,extragrad,sR)
                  ELSE
                    IF (IDAMPF==1) THEN
                      !Rp=(SQRT(2._q/PI)*alpha_2(i)/3.)**(1./3.)
                      Rq=(SQRT(2._q/PI)*alpha_2(j)/3.)**(1./3.)
                      mu=AUTOA*dfactor*(Rp**2+Rq**2)**0.5
                      mu=1./mu
                      CALL vdw_pairContrib_damp2(i,j,atrad,crit,r12_,ccc,mu,&
                       & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
                    ELSE
                      CALL vdw_pairContrib(i,j,atrad,crit,r12_,ccc,rrr,dfactor,&
                       & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
                    ENDIF
                  ENDIF
                END IF
                !IF (IO%IU6>=0) WRITE (*,*) i,j,energy-RDUM 
              END DO
            END DO
          ELSE
                        
            !c determine damping parameter and cutoff radii for Ewald's summation
            CALL ew_determine_theta(LATT_CUR,theta)
            CALL ew_find_cutoffs(T_INFO,LATT_CUR,cc_2,theta,amax,bmax)

            !c number of translations of direct lattice vectors to be 
            !c considered in real-space part of Ewald's summation 
            criteria(1)=amax*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=amax*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=amax*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit1=int(criteria)+1

            !c number of translations of reciprocal latticevectors to be 
            !c considered in reciprocal part of Ewald's summation 
            criteria(1)=bmax*SUM(LATT_CUR%A(1,:)**2)**0.5/TPI
            criteria(2)=bmax*SUM(LATT_CUR%A(2,:)**2)**0.5/TPI
            criteria(3)=bmax*SUM(LATT_CUR%A(3,:)**2)**0.5/TPI
            crit2=int(criteria)+1

            !c number of translations of direct lattice vectors
            !c for short-range damping calculations
            !atrad=40.
            criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
            criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
            criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
            crit3=int(criteria)+1

            energy1=0._q;energy2=0._q;energy3=0._q
            vdw_f21=0._q;vdw_f22=0._q;vdw_f23=0._q
            virialstress1=0._q; virialstress2=0._q;virialstress3=0._q
            extragrad=0._q
            DO i=1,T_INFO%NIONS
              IF (IDAMPF==1) THEN
                Rp=(SQRT(2._q/PI)*alpha_2(i)/3.)**(1./3.)
              ENDIF

              IF (LVDWSCS .AND. LSCSGRAD) THEN
                CALL gradient_RA(T_INFO%NIONS,drrrA,r0TS(i),alphaTS(i),alpha_2(i),dalphaA(i,:,:),LSCALER0)
              ENDIF
              DO j=i,T_INFO%NIONS
                r12_=x(:,i)-x(:,j)
                ccc=combination_ruleTS(cc_2(i),cc_2(j),alpha_2(i),alpha_2(j)) 
                rrr=r0_2(i)+r0_2(j)

                !c add real space contribution for a pair i,j
                CALL ew_direct_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit1, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy1=energy1+energyDum
                vdw_f21=vdw_f21+vdw_f2Dum
                virialstress1=virialstress1+virialstressDum
                !c add reciprocal space contribution for a pair i,j
                CALL ew_reciprocal_sum_part(i,j,r12_,-ccc,LATT_CUR, T_INFO, DYN,crit2, theta,energyDum,vdw_f2Dum,virialstressDum)
                energy2=energy2+energyDum
                vdw_f22=vdw_f22+vdw_f2Dum
                virialstress2=virialstress2+virialstressDum
                !c add short range damping
                IF (LVDWSCS .AND. LSCSGRAD) THEN
                    CALL gradient_C(T_INFO%NIONS,cc_2(i),cc_2(j),dcAB,alpha_2(i),alpha_2(j),&
                    & dcA(i,:,:),dcA(j,:,:),dalphaA(i,:,:),dalphaA(j,:,:))  
                    CALL gradient_RA(T_INFO%NIONS,drrrB,r0TS(j),alphaTS(j),alpha_2(j),dalphaA(j,:,:),LSCALER0)
                    drrr=drrrA+drrrB    
                    CALL ew_damping_correction_partSCS(i,j,r12_,ccc,rrr,LATT_CUR, T_INFO, DYN,crit3,dfactor,dcAB,drrr,energyDum,vdw_f2Dum,virialstressDum,extragradDum)
                    extragrad=extragrad+extragradDum
                    !CALL vdw_pairContribSCS(i,j,atrad,crit,r12_,ccc,dcAB,rrr,drrr,dfactor,&
                    !   & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress,extragrad,sR)
                ELSE
                  IF (IDAMPF==1) THEN
                    Rq=(SQRT(2._q/PI)*alpha_2(j)/3.)**(1./3.)
                    mu=AUTOA*dfactor*(Rp**2+Rq**2)**0.5
                    mu=1./mu
                    CALL ew_damping_correction_partDamp2(i,j,r12_,ccc,rrr,LATT_CUR, T_INFO, DYN,crit3,mu,energyDum,vdw_f2Dum,virialstressDum)
                  ELSE
                    CALL ew_damping_correction_part(i,j,r12_,ccc,rrr,LATT_CUR, T_INFO, DYN,crit3,dfactor,energyDum,vdw_f2Dum,virialstressDum)  
                  ENDIF
                ENDIF 
                energy3=energy3+energyDum
                vdw_f23=vdw_f23+vdw_f2Dum
                virialstress3=virialstress3+virialstressDum
              END DO
            END DO

!             IF (IO%IU6>=0) THEN
!               WRITE(*,*) 'theta',theta
!               WRITE(*,*) 'amax',amax
!               WRITE(*,*) 'bmax', bmax
!               WRITE(*,*) 'atrad',atrad
!               WRITE(*,*) 'energy1',energy1
!               WRITE(*,*) 'energy2',energy2
!               WRITE(*,*) 'energy3',energy3
!             END IF

            energy=energy1+energy2+energy3
            vdw_f2=vdw_f21+vdw_f22+vdw_f23
            virialstress=virialstress1+virialstress2+virialstress3
          ENDIF

         
          IF (LVDWSCS) THEN
            vdw_f2=vdw_f2+extragrad(:,1:T_INFO%NIONS)
            !c update forces
            TIFOR=TIFOR-vdw_f2

            !c update stress tensor   
            !TSIF=TSIF+virialstress+MATMUL(extragrad(:,T_INFO%NIONS+1:),TRANSPOSE(LATT_CUR%A))
            TSIF=TSIF+virialstress-MATMUL(extragrad(:,T_INFO%NIONS+1:),TRANSPOSE(LATT_CUR%A))

          ELSE 
            TIFOR=TIFOR-vdw_f2
            TSIF=TSIF+virialstress
          ENDIF

          !c update energy (unless it was already updated)
          IF (.NOT. LCFDM) TOTEN=TOTEN+energy  

  1238 FORMAT(/,"  Number of pair interactions contributing to vdW energy:",I8)
  !1239 FORMAT(  "  Estimated vdW energy (eV):",F10.5)
  1239 FORMAT(  "  Edisp (eV):",F11.5)
          IF (IO%IU6>=0) THEN
            IF (.NOT. LEWALD) WRITE(IO%IU6,1238) counter
            WRITE(IO%IU6,1239) energy
          ENDIF

        END SUBROUTINE vdw_forces_TS


        SUBROUTINE vdw_forces_dDsC(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,relvol_,IVDW,&
                 & Overlap,M2,RELCHG,HCD)
        !c dDsC method !!! and its variants
          USE setexm
          TYPE (grid_3d) GRIDC
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),SAVE :: atrad=50._q,SCSRAD=120._q
          REAL(q) :: gradients
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),TOTEN
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: energy,ccc,rrr,virialstress(3,3)
          REAL(q) :: cc(71),r0(71),alpha(71)
          CHARACTER (LEN=2),SAVE :: tags(71)
          REAL(q), PARAMETER :: conversion=10.364425449557316
          REAL(q), PARAMETER :: conversion2=0.057652660443590298
          REAL(q),SAVE :: sR=0.94_q,dfactor=20._q,s6=1.00_q
          REAL(q) :: userC(T_INFO%NTYP),userC2(T_INFO%NIONS)
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER :: i,j,ii,jj,kk,counter,indx1,indx2
          INTEGER :: historycounter
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          integer :: crit(3),crit1(3),crit2(3),crit3(3)
          real(q) :: criteria(3)
          real(q) :: x(3,T_INFO%NIONS),x1(3),x2(3),r12(3),r12_(3)
          real(q) :: r12norm
          real(q) :: cc_2(T_INFO%NIONS),r0_2(T_INFO%NIONS),alpha_2(T_INFO%NIONS)
          real(q) :: r0TS(T_INFO%NIONS),alphaTS(T_INFO%NIONS)
          real(q) :: alphaTSscreened(T_INFO%NIONS),c6TSscreened(T_INFO%NIONS),r0TSscreened(T_INFO%NIONS)
          real(q),ALLOCATABLE,SAVE :: cc_(:),r0_(:),alpha_(:)
          real(q) :: relvol_(T_INFO%NIONS),Overlap(T_INFO%NIONS,T_INFO%NIONS),M2(T_INFO%NIONS),RELCHG(T_INFO%NIONS)
          real(q) :: HCD(T_INFO%NIONS)
          integer :: dfnd1(T_INFO%NTYP),dfnd2(T_INFO%NTYP),dfnd3(T_INFO%NIONS)
          LOGICAL,SAVE :: LVDWSCS=.FALSE.,LSCALER0=.TRUE.,LSCSGRAD=.TRUE.
          LOGICAL,ALLOCATABLE,SAVE :: LVDW_SAMETYPE(:)
          LOGICAL,DIMENSION(3),SAVE :: LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)
          LOGICAL,SAVE :: LCFDM=.FALSE.
          INTEGER,SAVE :: MBD_SIZE(3)
          REAL(q) :: dcA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q) :: dalphaA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q),PARAMETER :: dstep=1e-2
          REAL(q) :: dcAB(3,T_INFO%NIONS+3),extragrad(3,T_INFO%NIONS+3),extragradDum(3,T_INFO%NIONS+3)
          REAL(q) :: drrr(3,T_INFO%NIONS+3), drrrA(3,T_INFO%NIONS+3), drrrB(3,T_INFO%NIONS+3)
          REAL(q),SAVE :: MBD_BETA=2.56 !range separating parameter for CFMD
          INTEGER,SAVE :: IDAMPF=0
          REAL(q) :: mu, Rp,Rq
          REAL(q) :: energy1,energy2,energy3,energyDum
          REAL(q) :: vdw_f21(3,T_INFO%NIONS),vdw_f22(3,T_INFO%NIONS),vdw_f23(3,T_INFO%NIONS),vdw_f2Dum(3,T_INFO%NIONS)
          REAL(q) :: virialstress1(3,3),virialstress2(3,3),virialstress3(3,3),virialstressDum(3,3)
          REAL(q) :: theta,amax,bmax,BTT,BTTasym,C6ij,rij
          REAL(q),SAVE :: att0,btt0
          INTEGER :: s0, ntot, s, nloc
          LOGICAL, SAVE :: LEWALD=.FALSE.
          INTEGER :: IVDW
          LOGICAL :: LUSERPARAM



  1234 FORMAT(/,"  dDsC method for vdW energy calculation",&
              /,"  ---------------------------------------------------------------------------------------------")
  1230 FORMAT(/,"  Atomic reference data used in the dDsC method for vdW correction:",&
              /,"                              R0at           ALPHAat      ",&
              /,"                              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")
  1235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6dDsC            XDM            ALPHAts        RELVOL",&
              /,"            (au)              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")

  2235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6ts              R0ts           ALPHAts           C6scs          R0scs        ALPHAscs        RELVOL ",&
              /,"            (au)              (au)            (au)              (au)           (au)          (au)                 ",&
              /,"   -----------------------------------------------------------------------------------------------------------------")
  1236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  12360 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3)
  2236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  2250 FORMAT(/,"  IVDW = ",I3)
  1237 FORMAT("  VDW_RADIUS = ",F9.3," A")
  2237 FORMAT("  SCSRAD = ",F9.3," A")
  1233 FORMAT("  VDW_SR aka BTT0 = ",F9.3)
  1241 FORMAT("  VDW_S6 aka ATT0 = ",F9.3)
  1240 FORMAT("  VDW_D = ",F9.3)
  2241 FORMAT("  LSCALER0 = ",L1)
  2242 FORMAT("  LSCSGRAD = ",L1)
  2243 FORMAT("  LVDW_ONECELL = ",L1,1X,L1,1X,L1)
  2244 FORMAT("  MBD_BETA = ",F9.3)
  2245 FORMAT("  VDW_IDAMPF = ",I2)
  2246 FORMAT("  LVDW_EWALD = ",L1)
  2247 FORMAT("  LVDW_SAMETYPE = ")
  2248 FORMAT("  VDW_MBD_SIZE = ",I2,X,I2,X,I2)
  2249 FORMAT("  LCFDM = ",L1)

          if (historycounter==1) then
            tags=(/'H ','Be','Mg','Ca','Sr','Ba',&
                   'He','Ne','Ar','Kr','Xe','Rn',&
                   'Li','Na','K ','Rb','Cs',&
                   'B ','Al','Ga','In','Tl',&
                   'C ','Si','Ge','Sn','Pb',&
                   'N ','P ','As','Sb','Bi',&
                   'O ','S ','Se','Te','Po',&
                   'F ','Cl','Br','I ','At',&
                   'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',&
                   'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',&
                   'Hf','Ta','W ','Re','Os',  'Ir','Pt','Au','Hg'/)
            cc=(/6.50,214.,627.,2221.,3170.,5727.0,&
                 1.46,6.38,64.3,129.6,285.90,390.63,&
                 1387.,1556.,3897.,4691.,6582.08,&
                 99.5,528.,498.,707.05,717.44,&
                 46.6,305.,354.,587.42,697.0,&
                 24.2,185.,246.,459.322,571.0,&
                 15.6,134.,210.,396.,530.92,&
                 9.52,94.6,162.,385.,457.53,&
                 1383.,1044.,832.,602.,552.,482.,408.,373.,253.,284.,&
                 1968.58, 1677.91,1263.61,1028.73,1390.87,609.75,469.0,157.5,339.0,452.0, &
                 1274.8,1019.92,847.93,710.2,596.67,359.1,347.1,298.0,392.0/)
            cc=cc*conversion2
            alpha=(/4.50,38.,71.,160.,199.,275.0,&
                   1.38,2.67,11.1,16.8,27.30,33.54,&
                   164.2,162.7,292.9,319.2,427.12,&
                   21.0,60.,60.,70.22,69.92,&
                   12.0,37.,41.,55.95,61.8,&
                   7.4,25.,29.,43.67,49.02,&
                   5.4,19.6,25.,37.65,45.013,&
                   3.8,15.,20.,35.,38.93,&
                   120.,98.,84.,78.,63.,56.,50.,48.,42.,40.,&
                   126.737,119.97,101.603,88.42,80.08,65.90,56.1,23.68,50.6,39.7,&
                   99.52,82.53,71.041,63.04,55.055,42.51,39.68,36.5,33.9/)
            r0=(/1.64,2.21,2.26,2.46,2.40,2.52,&
                 1.40,1.54,1.88,2.02,2.16,2.24,&
                 2.20,1.97,1.96,1.97,2.00,&
                 2.06,2.29,2.22,2.24,2.07,&
                 1.90,2.22,2.22,2.28,2.28,&
                 1.77,2.12,2.17,2.26,2.29,&
                 1.69,2.04,2.14,2.23,2.17,&
                 1.61,1.96,2.08,2.21,2.15,&
                 2.43,2.39,2.35,2.11,2.10,2.24,2.21,2.02,1.99,2.13,&
                 2.55,2.40,2.24,2.17,2.16,2.11,2.09,1.94,2.02,2.11,&
                 2.23,2.20,2.16,2.13,2.03,2.12,2.07,2.04,2.11/)

            ALLOCATE(cc_(T_INFO%NTYP),r0_(T_INFO%NTYP),alpha_(T_INFO%NTYP))
            ALLOCATE(LVDW_SAMETYPE(T_INFO%NTYP))
            cc_2=0._q;r0_2=0._q;alpha_2=0._q
            cc_=0._q;r0_=0._q;alpha_=0._q
            LVDW_SAMETYPE=.TRUE.
            dfnd1=0;dfnd2=0
            DO i=1,T_INFO%NTYP
              indx1=cwhereisit(tags,ELEM(i),SIZE(tags))
              IF (indx1>0) THEN
                dfnd1(i)=1
                dfnd2(i)=1
                cc_(i)=cc(indx1)
                r0_(i)=r0(indx1)
                alpha_(i)=alpha(indx1)
              END IF
            ENDDO

           !c read in the input parameters
            LOPEN=.FALSE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

              !c cutoff radius for vdW energy
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_RADIUS','=','#',';','F', &
                    IDUM,atrad,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) atrad=50._q
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 atrad=50._q
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_RADIUS'' from file INCAR'
              ENDIF
              !c sanity test
              IF (atrad <1._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_RADIUS:',atrad
                      WRITE(IO%IU0,*) 'VDW_RADIUS must be greater than 1.0 Angstrom!'
                ENDIF
                STOP
              ENDIF

              !c s6 parameter for damping function - usualy not needed in TS method
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_S6','=','#',';','F', &
                    IDUM,ATT0,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) ATT0=13.96
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 ATT0=13.96
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_S6'' from file INCAR for ATT0'
              ENDIF
              !c sanity test
              IF (ATT0 <0._q .OR. ATT0>20._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_S6:',ATT0
                      WRITE(IO%IU0,*) 'VDW_S6 must be a value from interval (0,20) for ATT0!'
                ENDIF
                STOP
              ENDIF

              LUSERPARAM=.FALSE.
              !c sR parameter for damping function
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_SR','=','#',';','F', &
                    IDUM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
              IF ((IERR==0) .AND. (N .GE. 1)) THEN
                LUSERPARAM=.TRUE.
                BTT0=RDUM
              ENDIF
              IF (IERR==3) LUSERPARAM=.FALSE.
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LUSERPARAM=.FALSE.
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_SR'' from file INCAR for BTT0'
              ENDIF

              !c sanity test for the user-defined value of sR
              IF (LUSERPARAM) THEN
                IF (BTT0 <0._q .OR. BTT0>2._q) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'Error: invalid value for VDW_SR:',BTT0
                    WRITE(IO%IU0,*) 'VDW_SR must be a value from interval (0,2) for BTT0!'
                  ENDIF
                  STOP
                ENDIF
              !c ...or use default values
              ELSE
                !c we have default s6 only for PBE
                !c Calculation has to be terminated if other functional is used and the 
                !c user didn't provide the sR value,  
                IF (.NOT.(LEXCH==8.OR.LEXCH==40) ) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'vdw_forces_dDsC: ERROR unsupported xc-functional, LEXCH=',LEXCH
                    WRITE(IO%IU0,*) 'please define parameter VDW_S6 (ATT0) and VDW_SR (BTT0) for this functional'
                  ENDIF
                  STOP
                ELSE
                  !c defaults for different versions of TS method
                  SELECT CASE(LEXCH)
                    !c PBE
                    CASE(8)
                    ATT0=13.96
                    BTT0=1.32
                    !c BP86
                    CASE(4)
                    ATT0=8.42
                    BTT0=1.39
                    !c revPBE
                    CASE(40)
                    ATT0=2.91
                    BTT0=1.70
                  END SELECT
                ENDIF
              ENDIF

              !c should the images of the original cell made by translating 
              !c along directions of lattice vectors be neglected? 
              !c (this is useful if one performs calculations of isolated molecule (LVDW_ONECELL= T T T)
              !c or a surface (LVDW_ONECELL= F F T)) 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_ONECELL','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW_ONECELL,CHARAC,N,3,IERR)
              IF (IERR==3) LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN
                 LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDW_ONECELL'' from file INCAR'
              ENDIF

            CLOSE(IO%IU5)

            !c basic sanity test + rescale the parameters 
            !c using the Hirshfeld volumes
          ENDIF
            dfnd3=0

!Parameters for PBE
!           ATT0=13.96
!           BTT0=1.32

            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              IF (alpha_(indx1)>0._q .AND. r0_(indx1)>0._q .AND. relvol_(i)>1.e-5) THEN
                dfnd3(i)=1
                alpha_2(i)=alpha_(indx1)*relvol_(i)
                cc_2(i)=alpha_2(i)*M2(i)*0.5_q
              ENDIF
            ENDDO

            IF (SUM(dfnd3) .LT. T_INFO%NIONS) THEN
              IF (IO%IU0>=0) THEN
                WRITE(IO%IU0,*) 'Error: some force-field parameter for the following atom is not defined:'
                DO j=1,T_INFO%NIONS
                  IF (dfnd3(j)==0) THEN
                    WRITE(IO%IU0,*) ELEM(T_INFO%ITYP(j)),j
                  ENDIF
                ENDDO
              ENDIF
              STOP
            ENDIF

            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)
              WRITE(IO%IU6,1235)
              DO i=1,T_INFO%NIONS
                WRITE(IO%IU6,1236) ELEM(T_INFO%ITYP(i)),cc_2(i)/(2*RYTOEV*AUTOA**6),&
                & M2(i),alpha_2(i),relvol_(i)
              ENDDO
              if (historycounter==1) then
                WRITE(IO%IU6,2250) IVDW
                WRITE(IO%IU6,1237) atrad
                WRITE(IO%IU6,1241) ATT0
                WRITE(IO%IU6,1233) BTT0
                  WRITE(IO%IU6,2243) LVDW_ONECELL(1),LVDW_ONECELL(2),LVDW_ONECELL(3)
              end if
            end if

          !c determine the number of cell images 
          !c needed for each direction 
          criteria(1)=atrad*SUM(LATT_CUR%B(:,1)**2)**0.5
          criteria(2)=atrad*SUM(LATT_CUR%B(:,2)**2)**0.5
          criteria(3)=atrad*SUM(LATT_CUR%B(:,3)**2)**0.5
          crit=int(criteria)+1

          !c if desired, switch off the interaction with cell 
          !c images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              criteria(i)=0
              crit(i)=0
            ENDIF
          ENDDO

          x=MATMUL(LATT_CUR%A,DYN%POSION)

          energy=0.
          gradients=0.
          vdw_f2=0._q;extragrad=0._q
          virialstress=0.
          counter=0

! SNS parallelization over the first pair of atoms
            s0=1
            ntot=T_INFO%NIONS
            call dist1d(gridc,s0, ntot, s, nloc)
            DO i=s,s+nloc-1
              indx1=T_INFO%ITYP(i)

              DO j=i,T_INFO%NIONS
                indx2=T_INFO%ITYP(j)

                r12_=DYN%POSION(:,i)-DYN%POSION(:,j)
                CALL min_imageV(3,r12_)
                r12_=MATMUL(LATT_CUR%A,r12_)
                rij= (SUM(r12_**2)**0.5)/AUTOA
                CALL GET_dDsC_PARAMETERS(T_INFO%NIONS,M2,alpha_2,RELCHG,HCD,Overlap(i,j),ATT0,BTT0,I,J,rij,BTT,BTTasym,C6ij)
                C6ij=C6ij*(2*RYTOEV*AUTOA**6)
                BTT=BTT/AUTOA
                BTTasym=BTTasym/AUTOA
                

                RDUM=energy
                      CALL vdw_pairContrib_dampTT(i,j,atrad,crit,r12_,C6ij,BTT,BTTasym,&
                       & T_INFO,LATT_CUR,counter,energy,vdw_f2,virialstress)
              END DO
            END DO
#ifdef MPI
            CALL M_sum_i(GRIDC%COMM, counter, 1)
            CALL M_sum_d(GRIDC%COMM, energy, 1)
            CALL M_sum_d(GRIDC%COMM, vdw_f2, 3*T_INFO%NIONS)
            CALL M_sum_d(GRIDC%COMM, virialstress, 9)
#endif
            TIFOR=TIFOR-vdw_f2
            TSIF=TSIF+virialstress

          !c update energy (unless it was already updated)
          TOTEN=TOTEN+energy

  1238 FORMAT(/,"  Number of pair interactions contributing to vdW energy:",I8)
  1239 FORMAT(  "  Edisp (eV):",F11.5)
          IF (IO%IU6>=0) THEN
            IF (.NOT. LEWALD) WRITE(IO%IU6,1238) counter
            WRITE(IO%IU6,1239) energy
         ENDIF
         RETURN
        END SUBROUTINE vdw_forces_dDsC

        SUBROUTINE GET_dDsC_PARAMETERS(NIONS,M2,alpha,RELCHG,HCD,Overlap,ATT0,BTT0,I,J,rij,BTT,BIJ_ASYM,C6ij)
        IMPLICIT NONE
        INTEGER :: NIONS,I,J
        REAL(q) :: M2(NIONS),alpha(NIONS),RELCHG(NIONS),HCD(NIONS),Overlap,ATT0,BTT0,rij,BTT,C6ij
        REAL(q),PARAMETER :: PI=3.1415926535897932384626433832795,THIRD=1.0D0/3.0D0
        REAL(q) :: BII_ASYM,BJJ_ASYM,X_TT,Fx,DNORM,BIJ_ASYM
! For periodic interactions (above 1 unit cell), we will use bij_asym, there is no overlap
        BII_ASYM=alpha(i)**(-THIRD)
        BJJ_ASYM=alpha(j)**(-THIRD)
! BTT0 is for polarizabilities in A^(3), but our polarizabilities are in a.u.^(3)
        BIJ_ASYM=2._q*BTT0/AUTOA*BII_ASYM*BJJ_ASYM/(BII_ASYM+BJJ_ASYM)

        X_TT=2._q*Overlap + DABS(RELCHG(I)*RELCHG(J))/rij
        X_TT=X_TT*(HCD(I)+HCD(J))/(HCD(I)*HCD(J))
        Fx= 2._q/(1._q + dexp(X_TT*ATT0))
        BTT=Fx*BIJ_ASYM
! Now the dispersion coefficients
        DNORM = alpha(i)*M2(j)+alpha(j)*M2(i)
        if(DNORM.lt.1.0D-3) then
           write(*,*) 'Unexptected stop: the following dDsC data is suspicious'
           write(*,*) 'Atom pair',I,J
           write(*,*) 'XDM',M2(i),M2(j)
           write(*,*) 'polarizability',alpha(i),alpha(j)
           STOP
        ENDIF
        C6ij=alpha(i)*alpha(j)*(M2(i)*M2(j))/DNORM;
        RETURN
        END SUBROUTINE GET_dDsC_PARAMETERS


        
!         SUBROUTINE vdw_forces_MBD(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,& 
!         &      relvol_,REFSTATE,IVDW,SYMM,KPOINTS)
        SUBROUTINE vdw_forces_MBD(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,ELEM,historycounter,& 
        &      relvol_,REFSTATE,IVDW,KPOINTS)
        !c Tkatchenko-Scheffler MBD method and its variants
          USE setexm
          TYPE (grid_3d) GRIDC
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(type_info)  :: T_INFO
          TYPE(latt) :: LATT_CUR
          REAL(q),SAVE :: atrad=50._q,SCSRAD=120._q
          REAL(q) :: gradients
          REAL(q) :: vdw_f2(3,T_INFO%NIONS),TOTEN
          REAL(q) :: TSIF(3,3),  TIFOR(3,T_INFO%NIONS)
          REAL(q) :: energy,ccc,rrr,virialstress(3,3)
          REAL(q) :: cc(71),r0(71),alpha(71)
          CHARACTER (LEN=2),SAVE :: tags(71)
          REAL(q), PARAMETER :: conversion=10.364425449557316
          REAL(q), PARAMETER :: conversion2=0.057652660443590298
          REAL(q),SAVE :: sR=0.94_q,dfactor=20._q,s6=1.00_q
          REAL(q) :: userC(T_INFO%NTYP),userC2(T_INFO%NIONS)     
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP)
          INTEGER :: i,j,ii,jj,kk,counter,indx1,indx2
          INTEGER :: historycounter
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          integer :: crit(3),crit1(3),crit2(3),crit3(3)
          real(q) :: criteria(3)
          real(q) :: x(3,T_INFO%NIONS),x1(3),x2(3),r12(3),r12_(3)
          real(q) :: r12norm
          real(q) :: cc_2(T_INFO%NIONS),r0_2(T_INFO%NIONS),alpha_2(T_INFO%NIONS)
          real(q) :: r0TS(T_INFO%NIONS),alphaTS(T_INFO%NIONS)
          real(q) :: alphaTSscreened(T_INFO%NIONS),r0TSscreened(T_INFO%NIONS)
          real(q),ALLOCATABLE,SAVE :: cc_(:),r0_(:),alpha_(:)
          real(q) :: relvol_(T_INFO%NIONS)
          integer :: dfnd1(T_INFO%NTYP),dfnd2(T_INFO%NTYP),dfnd3(T_INFO%NIONS)
          LOGICAL,SAVE :: LVDWSCS=.TRUE.,LSCALER0=.TRUE.,LSCSGRAD=.TRUE.
          LOGICAL,DIMENSION(3),SAVE :: LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)
          !LOGICAL,SAVE :: LCFDM=.FALSE.
          INTEGER,SAVE :: MBD_SIZE(3)
          REAL(q) :: dcA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q) :: dalphaA(T_INFO%NIONS,3,T_INFO%NIONS+3)
          REAL(q),PARAMETER :: dstep=1e-2
          REAL(q) :: dcAB(3,T_INFO%NIONS+3),extragrad(3,T_INFO%NIONS+3),extragradDum(3,T_INFO%NIONS+3)
          REAL(q) :: drrr(3,T_INFO%NIONS+3), drrrA(3,T_INFO%NIONS+3), drrrB(3,T_INFO%NIONS+3)
          INTEGER :: REFSTATE(T_INFO%NTYP)
          !REAL(q),SAVE :: MBD_BETA=2.56 !range separating parameter for CFMD
          INTEGER,SAVE :: IDAMPF=0
          REAL(q) :: mu, Rp,Rq
          REAL(q) :: energy1,energy2,energy3,energyDum
          REAL(q) :: vdw_f21(3,T_INFO%NIONS),vdw_f22(3,T_INFO%NIONS),vdw_f23(3,T_INFO%NIONS),vdw_f2Dum(3,T_INFO%NIONS)
          REAL(q) :: virialstress1(3,3),virialstress2(3,3),virialstress3(3,3),virialstressDum(3,3)
          REAL(q) :: theta,amax,bmax
          LOGICAL, SAVE :: LEWALD=.FALSE.
          !LOGICAL, SAVE :: LEWALD=.TRUE.
          INTEGER :: IVDW
          LOGICAL :: LUSERPARAM
          REAL(q) :: ECFDM
          REAL(q) :: dECFDM_A(3,T_INFO%NIONS),dECFDM_L(3,3)
          TYPE (kpoints_struct), OPTIONAL :: KPOINTS
          LOGICAL :: LMBDKGRID=.FALSE.
          LOGICAL :: LEXPANSION=.FALSE.
          !TYPE (symmetry),OPTIONAL ::   SYMM

          
         

  1234 FORMAT(/,"  Tkatchenko/Scheffler method for vdW energy calculation",&
              /,"  ---------------------------------------------------------------------------------------------")
  1230 FORMAT(/,"  Atomic reference data used in the T-S method for vdW correction:",&
              /,"            C6at              R0at           ALPHAat      REFSTATE",&
              /,"            (au)              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")
  1235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6ts              R0ts           ALPHAts        RELVOL",&
              /,"            (au)              (au)            (au)                ",&
              /,"   ---------------------------------------------------------------")
  
  2235 FORMAT(/,"  Parameters of vdW forcefield:",&
              /,"            C6ts              R0ts           ALPHAts           C6scs          R0scs        ALPHAscs        RELVOL ",&
              /,"            (au)              (au)            (au)              (au)           (au)          (au)                 ",&
              /,"   -----------------------------------------------------------------------------------------------------------------")
  1236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  12360 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,I2)
  2236 FORMAT("   ",A2,5X,F10.3,8X,F7.3,8X,F7.3,8X,F10.3,8X,F7.3,8X,F7.3,8X,F7.3)
  2250 FORMAT(/,"  IVDW = ",I3)
  1237 FORMAT("  VDW_RADIUS = ",F9.3," A")
  2237 FORMAT("  SCSRAD = ",F9.3," A")
  1233 FORMAT("  VDW_SR = ",F9.3)
  1241 FORMAT("  VDW_S6 = ",F9.3)
  1240 FORMAT("  VDW_D = ",F9.3)
  2241 FORMAT("  LSCALER0 = ",L1)
  2242 FORMAT("  LSCSGRAD = ",L1)     
  2243 FORMAT("  LVDW_ONECELL = ",L1,1X,L1,1X,L1)
  2244 FORMAT("  MBD_BETA = ",F9.3) 
  2245 FORMAT("  VDW_IDAMPF = ",I2) 
  2246 FORMAT("  LVDW_EWALD = ",L1)
  2248 FORMAT("  VDW_MBD_SIZE = ",I2,X,I2,X,I2)
  2249 FORMAT("  LCFDM = ",L1)

          if (historycounter==1) then


            tags=(/'H ','Be','Mg','Ca','Sr','Ba',&
                   'He','Ne','Ar','Kr','Xe','Rn',&
                   'Li','Na','K ','Rb','Cs',&
                   'B ','Al','Ga','In','Tl',&
                   'C ','Si','Ge','Sn','Pb',&
                   'N ','P ','As','Sb','Bi',&
                   'O ','S ','Se','Te','Po',&
                   'F ','Cl','Br','I ','At',&
                   'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',&
                   'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',&            
                   'Hf','Ta','W ','Re','Os',  'Ir','Pt','Au','Hg'/)
            cc=(/6.50,214.,627.,2221.,3170.,5727.0,&
                 1.46,6.38,64.3,129.6,285.90,390.63,&
                 1387.,1556.,3897.,4691.,6582.08,&
                 99.5,528.,498.,707.05,717.44,&
                 46.6,305.,354.,587.42,697.0,&
                 24.2,185.,246.,459.322,571.0,&
                 15.6,134.,210.,396.,530.92,&
                 9.52,94.6,162.,385.,457.53,&
                 1383.,1044.,832.,602.,552.,482.,408.,373.,253.,284.,&
                 1968.58, 1677.91,1263.61,1028.73,1390.87,609.75,469.0,157.5,339.0,452.0, &
                 1274.8,1019.92,847.93,710.2,596.67,359.1,347.1,298.0,392.0/)
            cc=cc*conversion2
            alpha=(/4.50,38.,71.,160.,199.,275.0,&
                   1.38,2.67,11.1,16.8,27.30,33.54,&
                   164.2,162.7,292.9,319.2,427.12,&
                   21.0,60.,60.,70.22,69.92,&
                   12.0,37.,41.,55.95,61.8,&
                   7.4,25.,29.,43.67,49.02,&
                   5.4,19.6,25.,37.65,45.013,&
                   3.8,15.,20.,35.,38.93,&
                   120.,98.,84.,78.,63.,56.,50.,48.,42.,40.,&
                   126.737,119.97,101.603,88.42,80.08,65.90,56.1,23.68,50.6,39.7,&
                   99.52,82.53,71.041,63.04,55.055,42.51,39.68,36.5,33.9/)
            r0=(/1.64,2.21,2.26,2.46,2.40,2.52,&
                 1.40,1.54,1.88,2.02,2.16,2.24,&
                 2.20,1.97,1.96,1.97,2.00,&
                 2.06,2.29,2.22,2.24,2.07,&
                 1.90,2.22,2.22,2.28,2.28,&
                 1.77,2.12,2.17,2.26,2.29,&
                 1.69,2.04,2.14,2.23,2.17,&
                 1.61,1.96,2.08,2.21,2.15,&
                 2.43,2.39,2.35,2.11,2.10,2.24,2.21,2.02,1.99,2.13,&
                 2.55,2.40,2.24,2.17,2.16,2.11,2.09,1.94,2.02,2.11,&
                 2.23,2.20,2.16,2.13,2.03,2.12,2.07,2.04,2.11/)

            ALLOCATE(cc_(T_INFO%NTYP),r0_(T_INFO%NTYP),alpha_(T_INFO%NTYP))
            
            cc_2=0._q;r0_2=0._q;alpha_2=0._q
            cc_=0._q;r0_=0._q;alpha_=0._q
           
            dfnd1=0;dfnd2=0
            DO i=1,T_INFO%NTYP
              indx1=cwhereisit(tags,ELEM(i),SIZE(tags))
              IF (indx1>0) THEN
                dfnd1(i)=1
                dfnd2(i)=1
                cc_(i)=cc(indx1)
                r0_(i)=r0(indx1)
                alpha_(i)=alpha(indx1)
              END IF
            ENDDO

           !c read in the input parameters
            LOPEN=.TRUE.
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
            !c is self-consistent screening to be taken into account?
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDWSCS','=','#',';','L', &
                IDUM,RDUM,CDUM,LVDWSCS,CHARAC,N,1,IERR)
            IF (IERR==3) LVDWSCS=.TRUE. 
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
               LVDWSCS=.TRUE.  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LVDWSCS'' from file INCAR'
            ENDIF 

              !c cutoff radius for vdW energy
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_RADIUS','=','#',';','F', &
                    IDUM,atrad,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) atrad=50._q 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 atrad=50._q  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_RADIUS'' from file INCAR'
              ENDIF
              !c sanity test
              IF (atrad <1._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_RADIUS:',atrad
                      WRITE(IO%IU0,*) 'VDW_RADIUS must be greater than 1.0 Angstrom!'
                ENDIF
                STOP
              ENDIF

              LUSERPARAM=.FALSE.
              !c sR parameter for damping function
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_SR','=','#',';','F', &
                    IDUM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
              IF ((IERR==0) .AND. (N .GE. 1)) THEN
                LUSERPARAM=.TRUE.
                sR=RDUM
              ENDIF
              IF (IERR==3) LUSERPARAM=.FALSE.
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                LUSERPARAM=.FALSE.
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_SR'' from file INCAR'
              ENDIF
          
              !c sanity test for the user-defined value of sR
              IF (LUSERPARAM) THEN 
                IF (sR <0._q .OR. sR>10._q) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'Error: invalid value for VDW_SR:',sR
                    WRITE(IO%IU0,*) 'VDW_SR must be a value from interval (0,10)!'
                  ENDIF
                  STOP
                ENDIF
              !c ...or use default values
              ELSE
                !c we have default s6 only for PBE
                !c Calculation has to be terminated if other functional is used and the 
                !c user didn't provide the sR value,  
                IF (.NOT.(LEXCH==8) ) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*) 'vdw_forces_TS: ERROR unsupported xc-functional, LEXCH=',LEXCH
                    WRITE(IO%IU0,*) 'please define parameter VDW_SR for this functional'
                  ENDIF
                  STOP
                ELSE
                  
                  !c defaults for different partitioning schemes
                  IF (IVDW==212) THEN
                    !c iterative Hirshfeld partitioning
                    sR=0.82
                  ELSE
                    !c Hirshfeld partitioning 
                    sR=0.83
                  ENDIF              
                ENDIF
              ENDIF

              !CALL XML_INCAR('VDW_SR','F',IDUM,sR,CDUM,.TRUE.,CHARAC,1)

              !c damping parameter d 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_D','=','#',';','F', &
                    IDUM,dfactor,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) dfactor=6. 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                 dfactor=20.  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''VDW_D'' from file INCAR'
              ENDIF
              !c sanity test
              IF (dfactor <1._q .OR. dfactor>100._q) THEN
                IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_D:',dfactor
                      WRITE(IO%IU0,*) 'VDW_D must be a value from interval (1,100)!'
                ENDIF
                STOP
              ENDIF
              !CALL XML_INCAR('VDW_D','F',IDUM,dfactor,CDUM,.TRUE.,CHARAC,1)

              !c user-provided C6 parameters in J*nm^6*mol^-1
              LUSERPARAM=.FALSE.
              userC=0._q
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_C6'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                LUSERPARAM=.TRUE.
                dfnd1=1
                cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)
              ENDIF

              !c if C6 in J*nm^6*mol^-1 is not found, try to read C6 in au 
              IF (.NOT. LUSERPARAM) THEN
                userC=0._q
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_C6au','=','#',';','F', &
                &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*)'Error reading item ''VDW_C6au'' from file INCAR.'
                    WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                  ENDIF
                  STOP
                ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                  dfnd1=1
                  cc_(1:T_INFO%NTYP)=userC(1:T_INFO%NTYP)*conversion2
                ENDIF
              ENDIF
 
              !c now its time to make basic sanity test of C6 values
              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: missing parameters C6 for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_C6 or VDW_C6au to define C6 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_C6','F',IDUM,cc_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

             !c user-provided parameter R0 (in A) for the damping function 
              LUSERPARAM=.FALSE.
              userC=0._q
              dfnd1=dfnd2
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_R0'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                LUSERPARAM=.TRUE.
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 .AND. userC(j)<10._q) THEN
                    dfnd1(j)=1
                    r0_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_R0 must be a value from interval (0.1,10)!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF

              !c if R0 in A is not found, try to read  R0 (in au) 
              IF (.NOT. LUSERPARAM) THEN
                userC=0._q
                dfnd1=dfnd2
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_R0au','=','#',';','F', &
                &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                  IF (IO%IU0>=0) THEN
                    WRITE(IO%IU0,*)'Error reading item ''VDW_R0au'' from file INCAR.'
                    WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                  ENDIF
                  STOP
                ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                  DO j=1,T_INFO%NTYP
                    IF (userC(j)>0.2 .AND. userC(j)<20._q) THEN
                      dfnd1(j)=1
                      r0_(j)=userC(j)*AUTOA
                    ELSE
                      IF (IO%IU0>=0) THEN
                        WRITE(IO%IU0,*) 'Error: invalid value for VDW_R0au:',userC(j)
                        WRITE(IO%IU0,*) 'All values for VDW_R0au must be a value from interval (0.2,20)!'
                      ENDIF
                      STOP
                    END IF
                  END DO
                ENDIF
              ENDIF

              !c check if we have R0 for each atomic type defined in POSCAR
              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters R0 are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_R0 to define R0 for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_R0','F',IDUM,r0_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              !c user-provided polarizabilities
              dfnd1=dfnd2
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_ALPHA','=','#',';','F', &
              &   IDUM,userC,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<T_INFO%NTYP))) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*)'Error reading item ''VDW_ALPHA'' from file INCAR.'
                  WRITE(IO%IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
                ENDIF
                STOP
              ELSE IF (IERR==0 .AND. N>=T_INFO%NTYP) THEN
                dfnd1=0
                DO j=1,T_INFO%NTYP
                  IF (userC(j)>0.1 ) THEN
                    dfnd1(j)=1
                    alpha_(j)=userC(j)
                  ELSE
                    IF (IO%IU0>=0) THEN
                      WRITE(IO%IU0,*) 'Error: invalid value for VDW_ALPHA:',userC(j)
                      WRITE(IO%IU0,*) 'All values for VDW_ALPHA must be greater than 0.1 au!'
                    ENDIF
                    STOP
                  END IF
                END DO
              ENDIF

              IF (SUM(dfnd1) .LT. T_INFO%NTYP) THEN
                IF (IO%IU0>=0) THEN
                  WRITE(IO%IU0,*) 'Error: force-field parameters ALPHA are not defined for the following elements:'
                  DO j=1,T_INFO%NTYP
                    IF (dfnd1(j)==0) THEN
                      WRITE(IO%IU0,*) ELEM(j)
                    ENDIF
                  ENDDO
                  WRITE(IO%IU0,*) 'Use flag VDW_ALPHA to define ALPHA for each atomic type in POSCAR!'
                ENDIF
                STOP
              ENDIF

              !CALL XML_INCAR_V('VDW_ALPHA','F',IDUM,alpha_,CDUM,.TRUE.,CHARAC,T_INFO%NTYP)

              !c should the images of the original cell made by translating 
              !c along directions of lattice vectors be neglected? 
              !c (this is useful if one performs calculations of isolated molecule (LVDW_ONECELL= T T T)
              !c or a surface (LVDW_ONECELL= F F T)) 
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_ONECELL','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW_ONECELL,CHARAC,N,3,IERR)
              IF (IERR==3) LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./) 
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN
                 LVDW_ONECELL=(/.FALSE.,.FALSE.,.FALSE./)  
                IF (IO%IU0>=0) &
                  WRITE(IO%IU0,*)'Error reading item ''LVDW_ONECELL'' from file INCAR'
              ENDIF

              

              !c this setting is specific for coupled-fluctuating-dipoles model
              

                !c damping parameter
!                 CALL RDATAB(LOPEN,INCAR,IO%IU5,'MBD_BETA','=','#',';','F', &
!                   IDUM,MBD_BETA,CDUM,LDUM,CHARAC,N,1,IERR)
!                 IF (IERR==3) MBD_BETA=2.56 
!                 IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
!                   MBD_BETA=2.56
!                   IF (IO%IU0>=0) &
!                   WRITE(IO%IU0,*)'Error reading item ''MBD_BETA'' from file INCAR'
!                 ENDIF

                !c this parameter controls the size of Hammiltonian for 
                !c system under periodic boundary conditions
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'VDW_MBD_SIZE','=','#',';','I', &
                    MBD_SIZE,RDUM,CDUM,LDUM,CHARAC,N,3,IERR)
                IF ((IERR==0).AND.(N==3)) LMBDKGRID=.TRUE.
                IF (IERR==3) MBD_SIZE=1 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN
                  MBD_SIZE=1
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''VDW_MBD_SIZE'' from file INCAR'
                ENDIF
                !c sanity test
                DO i=1,3
                  IF (MBD_SIZE(i)<1) THEN 
                    WRITE(IO%IU0,*)'Invalid value for ''VDW_MBD_SIZE'' read from file INCAR'
                    WRITE(IO%IU0,*)' ''VDW_MBD_SIZE'' is set to 1 1 1'
                    MBD_SIZE=1
                  ENDIF
                ENDDO
 
                !c should the gradients be computed in the TS+SCS calculation?
                !c ...might be pretty slow for the large systems
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'LSCSGRAD','=','#',';','L', &
                  IDUM,RDUM,CDUM,LSCSGRAD,CHARAC,N,1,IERR)
                IF (IERR==3) LSCSGRAD=.TRUE. 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   LSCSGRAD=.TRUE.  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''LSCSGRAD'' from file INCAR'
                ENDIF

                !c cutoff radius for calculation of dipole-dipole interaction
                !c tensor - converges slowly so large value is needed
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'SCSRAD','=','#',';','F', &
                      IDUM,SCSRAD,CDUM,LDUM,CHARAC,N,1,IERR)
                IF (IERR==3) SCSRAD=120._q 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   SCSRAD=120._q  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''SCSRAD'' from file INCAR'
                ENDIF
                !c sanity test
                IF (SCSRAD <0.) THEN
                  IF (IO%IU0>=0) THEN
                        WRITE(IO%IU0,*) 'Error: invalid value for SCSRAD:',SCSRAD
                        WRITE(IO%IU0,*) 'SCSRAD must be a positive number!'
                  ENDIF
                  STOP
                ENDIF
             
              IF (LVDWSCS) THEN             
                !c should the R0 parameters (damping function) be rescaled 
                !c within the SCS calculation?
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'LSCALER0','=','#',';','L', &
                  IDUM,RDUM,CDUM,LSCALER0,CHARAC,N,1,IERR)
                IF (IERR==3) LSCALER0=.TRUE. 
                IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
                   LSCALER0=.TRUE.  
                  IF (IO%IU0>=0) &
                    WRITE(IO%IU0,*)'Error reading item ''LSCALER0'' from file INCAR'
                ENDIF
              ELSE
                LSCALER0=.FALSE.
              ENDIF

            CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDWEXPANSION','=','#',';','L', &
                IDUM,RDUM,CDUM,LEXPANSION,CHARAC,N,1,IERR)
            IF (IERR==3) LEXPANSION=.FALSE. 
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
               LEXPANSION=.FALSE.  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LVDWEXPANSION'' from file INCAR'
            ENDIF 
              
            CLOSE(IO%IU5)
            
            !c write down the atomic reference data
            IF (IO%IU6>=0) THEN
                WRITE(IO%IU6,1230)
                IDUM=0
                DO i=1,T_INFO%NTYP
                  IDUM=IDUM+T_INFO%NITYP(i)
                  WRITE(IO%IU6,12360) ELEM(T_INFO%ITYP(idum)),cc_(T_INFO%ITYP(idum))/conversion2,&
                  & r0_(T_INFO%ITYP(idum))/AUTOA,alpha_(T_INFO%ITYP(idum)),REFSTATE(T_INFO%ITYP(idum))
                ENDDO
             ENDIF
           end if 

            !c basic sanity test + rescale the parameters 
            !c using the Hirshfeld volumes
            dfnd3=0
            !write(*,*) 'dfnd3a',dfnd3(1),dfnd3(2),dfnd3(3)
            DO i=1,T_INFO%NIONS
              indx1=T_INFO%ITYP(i)
              !write(*,*) 'sanity',i,indx1,cc_(indx1),alpha_(indx1),r0_(indx1),relvol_(i)
              !IF (cc_(indx1)>0._q .AND. alpha_(indx1)>0._q .AND. r0_(indx1)>0._q .AND. relvol_(i)>1.e-5) THEN
              IF (alpha_(indx1)>0._q .AND. r0_(indx1)>0._q .AND. relvol_(i)>1.e-5) THEN
                dfnd3(i)=1
                cc_2(i)=cc_(indx1)*relvol_(i)**2
                r0_2(i)=r0_(indx1)*relvol_(i)**0.3333333333
                alpha_2(i)=alpha_(indx1)*relvol_(i)
              ENDIF
            ENDDO

            !write(*,*) 'dfnd3b',dfnd3(1),dfnd3(2),dfnd3(3)

            !write(*,*) 'fdf',SUM(dfnd3), T_INFO%NIONS
            IF (SUM(dfnd3) .LT. T_INFO%NIONS) THEN
              IF (IO%IU0>=0) THEN
                WRITE(IO%IU0,*) 'Error: some force-field parameter for the following atom is not defined:'
                DO j=1,T_INFO%NIONS
                  IF (dfnd3(j)==0) THEN
                    WRITE(IO%IU0,*) ELEM(T_INFO%ITYP(j)),j
                  ENDIF
                ENDDO
              ENDIF
              STOP
            ENDIF

         
            !c convert C6 to eV A**6
            cc_2=cc_2*conversion
 
            !LSCSGRAD=.FALSE.
            ECFDM=0._q;dECFDM_A=0._q;dECFDM_L=0._q
            !CALL vdw_tsscs_range_separated(GRIDC,alpha_2,cc_2,r0_2/AUTOA,dfactor,sR,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
            !& SCSRAD,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IO%IU0)
            !IF (PRESENT(KPOINTS)) THEN
              !CALL vdw_tsscs_range_separated_k(GRIDC,alpha_2,cc_2,r0_2/AUTOA,dfactor,sR,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
              !& SCSRAD,LVDWSCS,LSCALER0,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IO%IU0,SYMM,KPOINTS)
              CALL vdw_tsscs_range_separated_k(GRIDC,alpha_2,cc_2,r0_2/AUTOA,dfactor,sR,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
              & SCSRAD,LVDWSCS,LSCALER0,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IO%IU0,IO%IU6,LMBDKGRID,LEXPANSION,KPOINTS)
            !ELSE
            !  CALL vdw_tsscs_range_separated_k(GRIDC,alpha_2,cc_2,r0_2/AUTOA,dfactor,sR,T_INFO%NIONS,DYN%POSION,LATT_CUR,alphaTSscreened,&
            !  & SCSRAD,LVDWSCS,LSCALER0,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IO%IU0,IO%IU6,LMBDKGRID,LEXPANSION)              
            !ENDIF
            
            IF (IO%IU0>=0) THEN
              write(*,*) '------------------------------'
              write(*,*) 'dECFDM_A'
              DO i=1,T_INFO%NIONS
                write(*,*) DECFDM_A(1,i), DECFDM_A(2,i),DECFDM_A(3,i)
              ENDDO

              write(*,*) '------------------------------'
              write(*,*) 'dECFDM_L'
              DO i=1,3
                write(*,*) DECFDM_L(1,i), DECFDM_L(2,i),DECFDM_L(3,i)
              ENDDO
            ENDIF

! #ifdef debug
!             IF (IO%IU0>=0) THEN
!               DO i=1,T_INFO%NIONS
!                 write(*,*) '------------------------------'
!                 write(*,*) 'dC/dx, atom',i
!                 DO j=1,T_INFO%NIONS+3
!                   write(*,*) dca(i,1,j),dca(i,2,j),dca(i,3,j)
!                 ENDDO
!               ENDDO
!             ENDIF
! #endif

            !c convert C6 to Jnm^6/mol
            !cc_2=cc_2/conversion
            !C6TSscreened=C6TSscreened/conversion
            !c convert C6 to au
            !c write some output here
     
            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1234)              
              WRITE(IO%IU6,1235) 
              DO i=1,T_INFO%NIONS
                WRITE(IO%IU6,1236) ELEM(T_INFO%ITYP(i)),cc_2(i)/(2*RYTOEV*AUTOA**6),&
                & r0_2(i)/AUTOA,alpha_2(i),relvol_(i)
              ENDDO
              if (historycounter==1) then
                WRITE(IO%IU6,2250) IVDW
                WRITE(IO%IU6,1237) atrad
                !WRITE(IO%IU6,2245) IDAMPF
                !WRITE(IO%IU6,1241) s6
                WRITE(IO%IU6,1233) sR
                WRITE(IO%IU6,1240) dfactor
                WRITE(IO%IU6,2237) SCSRAD
!                 IF (.NOT. LEWALD) THEN 
!                   WRITE(IO%IU6,2243) LVDW_ONECELL(1),LVDW_ONECELL(2),LVDW_ONECELL(3)                
!                 ENDIF
                !IF (LCFDM) WRITE(IO%IU6,2244) MBD_BETA
                IF (LMBDKGRID) WRITE(IO%IU6,2248) MBD_SIZE(1),MBD_SIZE(2),MBD_SIZE(3)
                !WRITE(IO%IU6,2246) LEWALD
                WRITE(IO%IU6,2242) LSCSGRAD
                WRITE(IO%IU6,2241) LSCALER0
              end if 
            end if
           

            TOTEN=TOTEN+ECFDM

            IF (LSCSGRAD) THEN
              !c update forces
              TIFOR=TIFOR+dECFDM_A
              !TIFOR=TIFOR-dECFDM_A

              !c update stress tensor   
              TSIF=TSIF+dECFDM_L
              !TSIF=TSIF-dECFDM_L
            ENDIF
      
  
  1239 FORMAT(  "  Edisp (eV):",F11.5)
          IF (IO%IU6>=0) THEN
            WRITE(IO%IU6,1239) ECFDM
          ENDIF

        END SUBROUTINE vdw_forces_MBD


        SUBROUTINE vdw_tsscs(GRIDC,alphaTS,c6TS_ev,r0,nions,POSION,LATT_CUR,alphaTSscreened,dalpha,&
        & c6TSscreened, dcA,r0TSscreened,rmax,LSCALER0,LSCSGRAD,LCFDM,LVDW_ONECELL,TOTEN,BETA,MBD_SIZE,IU0)
        !c determine SCS corrected C6, alpha, and R0
          TYPE (grid_3d) GRIDC
          TYPE (latt)        LATT_CUR
          INTEGER :: nions,i,j,ii,jj,mm,pp,qq,ppqq,ninfo,ppp,IU0
          INTEGER,PARAMETER :: odim=21
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),tau(3,3),dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: alphaTS(nions),c6TS(nions),r0(nions),omegaP(nions)
          REAL(q) :: c6TS_ev(nions)
          REAL(q) :: alphaTSscreened(nions),c6TSscreened(nions),FalphaTSscreened(odim,nions)
          REAL(q) :: r0TSscreened(nions)
          REAL(q) :: Amat(3*nions,3*nions)
          REAL(q) :: omega,alphaP,alphaQ
          REAL(q) :: ogrid(odim),ogridW(odim)
          REAL(q) :: Rp,Rq,Rpq
          INTEGER :: translations(3)
          REAL(q) :: alphaSCS(3,3)
          REAL(q) :: rmax !c cutoff radius in Angstroems
          REAL(q) :: dAmat(3*nions+9,3*nions),dT(3*nions,3*nions)
          REAL(q),ALLOCATABLE :: dTmatA(:,:,:),dTmatL(:,:,:)
          LOGICAL,SAVE :: LFIRST=.TRUE.
          LOGICAL :: LSCALER0,LSCSGRAD,LCFDM
          LOGICAL :: LVDW_ONECELL(3)
          REAL(q) :: dalpha(nions,3,nions+3),dca(nions,3,nions+3)
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: ECFDM,TOTEN
          REAL(q) :: BETA !range separating parameter for CFMD
          INTEGER :: MBD_SIZE(3)
          INTEGER, ALLOCATABLE,SAVE :: indexP(:),indexQ(:)
          REAL(q) :: alpha_omega(nions)
          REAL(q) :: R_omega(nions)
                  
          NODE_ME=1
          NCPU=1
#ifdef MPI
           NODE_ME=GRIDC%COMM%NODE_ME
           NCPU=GRIDC%COMM%NCPU
#endif

          IF (LSCSGRAD) THEN
            ALLOCATE(dTmatA(3,3*nions,3*nions))
            ALLOCATE(dTmatL(3*nions,3*nions,9))
            dTmatA=0._q
            dTmatL=0._q
          ENDIF


          !c conversion to atomic units
          lattmat_au=TRANSPOSE(LATT_CUR%A)/AUTOA
          c6TS=c6TS_ev/(2*RYTOEV*AUTOA**6)

          !c Gauss-Legendre grid and weigths for integration over frequencies
          ogrid=(/0.,0.0392901,0.118358,0.198912,0.282029,0.368919,&
                   &0.461006,0.560027,0.668179,0.788336,0.92439,&
                   &1.08179,1.26849,1.49661,1.78563,2.16917,&
                   &2.71062,3.54573,5.02734,8.44896,25.4517/)
          ogridW=(/0.,0.0786611,0.07964,0.0816475,0.0847872,0.0892294,&
                   &0.0952317,0.103172,0.113605,0.12735,0.145652,&
                   &0.170453,0.204917,0.254456,0.328962,0.448092,&
                   &0.655606,1.06596,2.06357,5.6851,50.9558/)

!           call gauss_chebyshev(odim, ogrid, ogridW)

          !c determine the number of images of simulation cell 
          !c in each direction needed for a given 
          !c cutoff radius         
          translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
          translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
          translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)
         
          !c if desired, switch off the interactions with cell images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              translations(i)=0
            ENDIF
          ENDDO

          !c determine characteristic excitation frequencies
          DO i=1,NIONS
            omegaP(i)=4./3.*c6TS(i)/alphaTS(i)**2
          ENDDO 
          
          dalpha=0._q
          dcA=0._q
          c6TSscreened=0._q
          
          !c prepare index tables
          ppqq=nions*(nions+1)/2
          IF (LFIRST) THEN
            ALLOCATE(indexP(ppqq))
            ALLOCATE(indexQ(ppqq))
            CALL index_table2d(nions,ppqq,indexP,indexQ)
          ENDIF

          !c loop over frequencies
          DO i=1,odim
            omega=ogrid(i)
            Amat=0._q
            dTmatA=0._q
            dTmatL=0._q
            DO pp=1,nions
              alpha_omega(pp)=alpha_freq(alphaTS(pp),omegaP(pp),omega)
              R_omega(pp)=((2./PI)**0.5*alpha_omega(pp)/3.)**(1./3.)
            ENDDO
!             DO mm=1,ppqq
            DO mm=NODE_ME,ppqq,NCPU
              pp=indexP(mm)
              qq=indexQ(mm)
           
              Rpq=(R_omega(pp)**2+R_omega(qq)**2)**0.5
              CALL scsTau_pq(POSION(:,pp),POSION(:,qq),Rpq,lattmat_au,translations, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
                  
              Amat(3*pp-2,3*qq-2:3*qq)=tau(1,:)
              Amat(3*pp-1,3*qq-2:3*qq)=tau(2,:)
              Amat(3*pp  ,3*qq-2:3*qq)=tau(3,:)
              Amat(3*qq-2,3*pp-2:3*pp)=tau(1,:)
              Amat(3*qq-1,3*pp-2:3*pp)=tau(2,:)
              Amat(3*qq  ,3*pp-2:3*pp)=tau(3,:)  

              IF (LSCSGRAD) THEN
                dTmatA(1:3,3*qq-2:3*qq,3*pp-2)=dtaux(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp-1)=dtauy(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp  )=dtauz(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-2)=-dtaux(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-1)=-dtauy(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq  )=-dtauz(:,:)

!                 dTmatA(1,3*qq-2:3*qq,3*pp-2)=dtaux(1,:)
!                 dTmatA(2,3*qq-2:3*qq,3*pp-2)=dtaux(2,:)
!                 dTmatA(3,3*qq-2:3*qq,3*pp-2)=dtaux(3,:)
!                 dTmatA(1,3*qq-2:3*qq,3*pp-1)=dtauy(1,:)
!                 dTmatA(2,3*qq-2:3*qq,3*pp-1)=dtauy(2,:)
!                 dTmatA(3,3*qq-2:3*qq,3*pp-1)=dtauy(3,:)
!                 dTmatA(1,3*qq-2:3*qq,3*pp  )=dtauz(1,:)
!                 dTmatA(2,3*qq-2:3*qq,3*pp  )=dtauz(2,:)
!                 dTmatA(3,3*qq-2:3*qq,3*pp  )=dtauz(3,:)
! 
!                 dTmatA(1,3*pp-2:3*pp,3*qq-2)=-dtaux(1,:)
!                 dTmatA(2,3*pp-2:3*pp,3*qq-2)=-dtaux(2,:)
!                 dTmatA(3,3*pp-2:3*pp,3*qq-2)=-dtaux(3,:)
!                 dTmatA(1,3*pp-2:3*pp,3*qq-1)=-dtauy(1,:)
!                 dTmatA(2,3*pp-2:3*pp,3*qq-1)=-dtauy(2,:)
!                 dTmatA(3,3*pp-2:3*pp,3*qq-1)=-dtauy(3,:)
!                 dTmatA(1,3*pp-2:3*pp,3*qq  )=-dtauz(1,:)
!                 dTmatA(2,3*pp-2:3*pp,3*qq  )=-dtauz(2,:)
!                 dTmatA(3,3*pp-2:3*pp,3*qq  )=-dtauz(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,1)=dtau_h0x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,1)=dtau_h0x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,1)=dtau_h0x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,2)=dtau_h0y(1,:) 
                dTmatL(3*pp-1,3*qq-2:3*qq,2)=dtau_h0y(2,:) 
                dTmatL(3*pp  ,3*qq-2:3*qq,2)=dtau_h0y(3,:) 

                dTmatL(3*pp-2,3*qq-2:3*qq,3)=dtau_h0z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,3)=dtau_h0z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,3)=dtau_h0z(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,4)=dtau_h1x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,4)=dtau_h1x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,4)=dtau_h1x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,5)=dtau_h1y(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,5)=dtau_h1y(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,5)=dtau_h1y(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,6)=dtau_h1z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,6)=dtau_h1z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,6)=dtau_h1z(3,:)
     
                dTmatL(3*pp-2,3*qq-2:3*qq,7)=dtau_h2x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,7)=dtau_h2x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,7)=dtau_h2x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,8)=dtau_h2y(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,8)=dtau_h2y(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,8)=dtau_h2y(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,9)=dtau_h2z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,9)=dtau_h2z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,9)=dtau_h2z(3,:) 
!!!!!!!!!!!!!!!!!!!!!!!!!!
                dTmatL(3*qq-2,3*pp-2:3*pp,1)=dtau_h0x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,1)=dtau_h0x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,1)=dtau_h0x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,2)=dtau_h0y(1,:) 
                dTmatL(3*qq-1,3*pp-2:3*pp,2)=dtau_h0y(2,:) 
                dTmatL(3*qq  ,3*pp-2:3*pp,2)=dtau_h0y(3,:) 

                dTmatL(3*qq-2,3*pp-2:3*pp,3)=dtau_h0z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,3)=dtau_h0z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,3)=dtau_h0z(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,4)=dtau_h1x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,4)=dtau_h1x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,4)=dtau_h1x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,5)=dtau_h1y(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,5)=dtau_h1y(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,5)=dtau_h1y(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,6)=dtau_h1z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,6)=dtau_h1z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,6)=dtau_h1z(3,:)
     
                dTmatL(3*qq-2,3*pp-2:3*pp,7)=dtau_h2x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,7)=dtau_h2x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,7)=dtau_h2x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,8)=dtau_h2y(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,8)=dtau_h2y(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,8)=dtau_h2y(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,9)=dtau_h2z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,9)=dtau_h2z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,9)=dtau_h2z(3,:) 

              ENDIF
            ENDDO

#ifdef MPI
           !c derivatives are synchronized later!!!
           CALL M_sum_d(GRIDC%COMM, Amat,9*nions*nions )
          
!            IF (LSCSGRAD) THEN
!              CALL M_sum_d(GRIDC%COMM, dTmatA,27*nions*nions )
!              CALL M_sum_d(GRIDC%COMM, dTmatL,81*nions*nions )
!            ENDIF 
#endif


            DO pp=1,nions
!               alphaP=alpha_freq(alphaTS(pp),omegaP(pp),omega)
!               Amat(3*pp-2,3*pp-2)=Amat(3*pp-2,3*pp-2)+1._q/alphaP
!               Amat(3*pp-1,3*pp-1)=Amat(3*pp-1,3*pp-1)+1._q/alphaP
!               Amat(3*pp,    3*pp)=Amat(3*pp,    3*pp)+1._q/alphaP
              Amat(3*pp-2,3*pp-2)=Amat(3*pp-2,3*pp-2)+1._q/alpha_omega(pp)
              Amat(3*pp-1,3*pp-1)=Amat(3*pp-1,3*pp-1)+1._q/alpha_omega(pp)
              Amat(3*pp,    3*pp)=Amat(3*pp,    3*pp)+1._q/alpha_omega(pp)
            ENDDO

            !c matrix inversion
            CALL SVDINVERSE(Amat,3*nions,ninfo)

            ! partial contraction of Amat 
            DO pp=1,nions
              alphaSCS=0._q
              DO qq=1,nions
                DO ii=1,3
                  DO jj=1,3
                    alphaSCS(ii,jj)=alphaSCS(ii,jj)+Amat(3*pp-3+ii,3*qq-3+jj)
                  ENDDO
                ENDDO
              ENDDO
              FalphaTSscreened(i,pp)=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.
                            
              !c screened C6
              c6TSscreened(pp)=c6TSscreened(pp)+FalphaTSscreened(i,pp)**2*ogridW(i) 
            ENDDO

!             2235 FORMAT("a",I2,2X,I2,2X,F10.6)
!               IF (IU0>=0) THEN     
!                 IF (i==1) THEN       
!                   DO ii=1,nions
!                     write(*,2235) i,ii,FalphaTSscreened(i,ii)
!                   ENDDO
!                 ENDIF
!               ENDIF

            IF (LSCSGRAD) THEN 
              DO ppp=1,3*nions+9
                dT=0._q
                IF (ppp .GT. 3*nions) THEN
                  dT=dTmatL(:,:,ppp-3*nions)
                ELSE                   
                  pp=(ppp-1)/3+1
                  DO qq=1,nions
                    dT(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                    dT(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                  ENDDO                  
                ENDIF
#ifdef MPI
                CALL M_sum_d(GRIDC%COMM, dT,9*nions*nions )  
#endif

                dAmat=-MATMUL(Amat,dT)
                dAmat=MATMUL(dAmat,Amat)
                DO pp=1,nions
                  alphaSCS=0._q
                  DO qq=1,nions
                    DO ii=1,3
                      DO jj=1,3
                        alphaSCS(ii,jj)=alphaSCS(ii,jj)+dAmat(3*pp-3+ii,3*qq-3+jj)
                      ENDDO
                    ENDDO
                  ENDDO
                  alphaP=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.

                  ii=MOD(ppp,3)
                  IF (ii==0) ii=3
                  jj=(ppp-ii)/3+1
                  IF (i==1) THEN
                    dalpha(pp,ii,jj)=alphaP !(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.
                  ENDIF
                  
                  dcA(pp,ii,jj)=dcA(pp,ii,jj)+alphaP*FalphaTSscreened(i,pp)*ogridW(i)
                ENDDO
              ENDDO
              
!              2236 FORMAT("da",I2,2X,I2,2X,F10.6,2X,F10.6,2X,F10.6)
!               
!              IF (IU0>=0) THEN  
!                IF (i==1) THEN
!                  DO ii=1,nions        
!                    DO pp=1,nions+3
!                      write(*,2236) i,ii,dalpha(ii,1,pp),dalpha(ii,2,pp),dalpha(ii,3,pp)
!                    ENDDO
!                  ENDDO
!                ENDIF
!              ENDIF

            ENDIF
          ENDDO

!           IF (LSCSGRAD .AND. IU0>=0) THEN 
!             DO pp=1,nions+3
!               DO jj=1,3
!                 write(*,*) 'atom:',pp,jj
!                 write(*,*) '---------------'
!                 write(*,*) dalpha(1,jj,pp),dalpha(2,jj,pp),dalpha(3,jj,pp),dalpha(4,jj,pp)
!               ENDDO
!             ENDDO
!           ENDIF


          c6TSscreened=c6TSscreened*3._q/PI
          IF (LSCSGRAD) dcA=dcA*6._q/PI

          !c screened static polarizabilities and c6
          alphaTSscreened(:)=FalphaTSscreened(1,:)

          !c rescale R0?
          IF (LSCALER0) THEN
            r0TSscreened=r0*(alphaTSscreened/alphaTS)**(1._q/3._q)
          ELSE
            r0TSscreened=r0
          ENDIF
 
          !c use cfdm model to compute many-body vdw energy
          !c not well tested yet - supposed to work only for 
          !c isolated molecules !!!
          IF (LCFDM) THEN
             CALL vdw_cfdm(GRIDC,nions,POSION,lattmat_au,LATT_CUR%B,alphaTSscreened, &
             &   c6TSscreened,r0TSscreened/AUTOA,rmax/AUTOA,ECFDM,BETA,MBD_SIZE,IU0)
            !CALL vdw_cfdm2(GRIDC,nions,POSION,lattmat_au,alphaTS, &
            !&   c6TS,rmax/AUTOA,translations,ECFDM,BETA,IU0)
            IF (IU0>=0) THEN
              write(*,*) 'ECFDM:',ECFDM,"(eV)"
              TOTEN=TOTEN+ECFDM
              write(*,*) 'TOTEN+ECFDM:',TOTEN,"(eV)"
            ENDIF
           
          ENDIF

          !c convert C6 to eVA**6
          c6TSscreened=c6TSscreened*(2*RYTOEV*AUTOA**6)
          dcA=dcA*(2*RYTOEV*AUTOA**5)
!           IF (LSCSGRAD .AND. IU0>=0) THEN 
!             DO pp=1,nions     
!               write(*,*) 'ion:',pp
!               write(*,*) 'dcA',dcA(pp,1,nions+1),dcA(pp,2,nions+1),dcA(pp,3,nions+1)
!               write(*,*) 'dcA',dcA(pp,1,nions+2),dcA(pp,2,nions+2),dcA(pp,3,nions+2)
!               write(*,*) 'dcA',dcA(pp,1,nions+3),dcA(pp,2,nions+3),dcA(pp,3,nions+3)
!             ENDDO
!           ENDIF
          dalpha=dalpha/AUTOA
          IF (LSCSGRAD) THEN
            !DEALLOCATE(dTmat)
            DEALLOCATE(dTmatA,dTmatL)
          ENDIF

          IF (LFIRST) LFIRST=.FALSE.
        END SUBROUTINE vdw_tsscs

        SUBROUTINE vdw_tsscs_range_separated(GRIDC,alphaTS,c6TS_ev,r0,dampD,sR,nions,POSION,LATT_CUR,alphaTSscreened,&
        & rmax,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IU0)
        !c determine SCS corrected C6, alpha, and R0
          TYPE (grid_3d) GRIDC
          TYPE (latt)        LATT_CUR
          INTEGER :: nions,i,j,ii,jj,mm,nn,pp,qq,ppqq,ninfo,ppp,IU0,pp_,qq_
          INTEGER,PARAMETER :: odim=12
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),tau(3,3),dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: alphaTS(nions),c6TS(nions),r0(nions),omegaP(nions)
          REAL(q) :: c6TS_ev(nions)
          REAL(q) :: alphaTSscreened(nions),FalphaTSscreened(odim,nions)
          REAL(q) :: Amat(3*nions,3*nions)
          REAL(q) :: omega,alphaP,alphaQ
          REAL(q) :: ogrid(odim),ogridW(odim)
          REAL(q) :: Rp,Rq,Rpq
          INTEGER :: translationsSR(3),translationsLR(3)
          REAL(q) :: alphaSCS(3,3)
          REAL(q) :: rmax !c cutoff radius in Angstroems        
          LOGICAL,SAVE :: LFIRST=.TRUE.
          LOGICAL :: LSCSGRAD
          LOGICAL :: LVDW_ONECELL(3)
          REAL(q) :: dalpha(nions,3,nions+3)
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: ECFDM
          INTEGER,INTENT(in) :: MBD_SIZE(3)
          INTEGER :: MBD_SIZE_TOTAL
          INTEGER, ALLOCATABLE,SAVE :: indexP(:),indexQ(:)
          REAL(q) :: alpha_omega(nions)
          REAL(q) :: R_omega(nions)
          REAL(q) :: dampD, sR ,R0pq     
          REAL(q) :: dAmat(3*nions+9,3*nions),dT(3*nions,3*nions)
          REAL(q),ALLOCATABLE :: dTmatA(:,:,:),dTmatL(:,:,:)
          REAL(q),ALLOCATABLE :: Tmat_lr(:,:),  Amat_lr(:,:),dAmat_lr(:,:),d_lr(:)
          REAL(q),ALLOCATABLE :: dTmatA_lr(:,:,:),dTmatL_lr(:,:,:)
          INTEGER :: ppqqLR
          INTEGER, ALLOCATABLE,SAVE :: indexPLR(:),indexQLR(:)
          REAL(q) :: alpha_
          REAL(q) :: lattmat_au_multi(3,3)
          REAL(q), ALLOCATABLE :: POSION_multi(:,:)
          REAL(q),ALLOCATABLE :: Fmat_lr(:,:),AT_lr(:,:),dT_lr(:,:)
          REAL(q),INTENT(out) :: dECFDM_A(3,nions),dECFDM_L(3,3)
          REAL(q) :: qdumm

          qdumm=0._q

          NODE_ME=1
          NCPU=1
#ifdef MPI
           NODE_ME=GRIDC%COMM%NODE_ME
           NCPU=GRIDC%COMM%NCPU
#endif

!           IF (LSCSGRAD) THEN
!             ALLOCATE(dTmatA(3,3*nions,3*nions))
!             ALLOCATE(dTmatL(3*nions,3*nions,9))
!             
!             dTmatA=0._q
!             dTmatL=0._q
!           ENDIF


          !c conversion to atomic units
          lattmat_au=TRANSPOSE(LATT_CUR%A)/AUTOA
          c6TS=c6TS_ev/(2*RYTOEV*AUTOA**6)

          !c Gauss-Legendre grid and weigths for integration over frequencies
!           ogrid=(/0.,0.0392901,0.118358,0.198912,0.282029,0.368919,&
!                    &0.461006,0.560027,0.668179,0.788336,0.92439,&
!                    &1.08179,1.26849,1.49661,1.78563,2.16917,&
!                    &2.71062,3.54573,5.02734,8.44896,25.4517/)
!           ogridW=(/0.,0.0786611,0.07964,0.0816475,0.0847872,0.0892294,&
!                    &0.0952317,0.103172,0.113605,0.12735,0.145652,&
!                    &0.170453,0.204917,0.254456,0.328962,0.448092,&
!                    &0.655606,1.06596,2.06357,5.6851,50.9558/)

          call gauss_chebyshev(odim, ogrid, ogridW)

!!!!!!!!!!!!!!!!!!!!!!!!Long range part
         !c prepare index tables
          MBD_SIZE_TOTAL=MBD_SIZE(1)*MBD_SIZE(2)*MBD_SIZE(3)
          ppqqLR=(MBD_SIZE_TOTAL*nions)*(MBD_SIZE_TOTAL*nions+1)/2
          !ppqq=(totmulti*nions)*(totmulti*nions-1)/2

          IF (LFIRST) THEN
            ALLOCATE(indexPLR(ppqqLR))
            ALLOCATE(indexQLR(ppqqLR))
            CALL index_table2d(nions*MBD_SIZE_TOTAL,ppqqLR,indexPLR,indexQLR)
          ENDIF
          
          IF (LSCSGRAD) THEN
            ALLOCATE(dAmat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
            ALLOCATE(dTmatA_lr(3,3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
            ALLOCATE(dTmatL_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL,9)) 
            ALLOCATE(dT_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
            !ALLOCATE(Fmat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))

            ALLOCATE(dTmatA(3,3*nions,3*nions))
            ALLOCATE(dTmatL(3*nions,3*nions,9))     
            dAmat_lr=0._q; dTmatA_lr=0._q ;dTmatL_lr=0._q ; dT_lr=0._q 
            dTmatA=0._q;dTmatL=0._q
          ENDIF

          !c determine frequency independent T-tensor (long range part)
          ALLOCATE(Tmat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
          ALLOCATE(Amat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
          ALLOCATE(AT_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
          ALLOCATE(Fmat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))
          ALLOCATE(d_lr(3*nions*MBD_SIZE_TOTAL))
          Tmat_lr=0._q
          dECFDM_A=0._q;dECFDM_L=0._q

          translationsLR(1)=nint(rmax/MBD_SIZE(1)*SUM(LATT_CUR%B(:,1)**2)**0.5)
          translationsLR(2)=nint(rmax/MBD_SIZE(2)*SUM(LATT_CUR%B(:,2)**2)**0.5)
          translationsLR(3)=nint(rmax/MBD_SIZE(3)*SUM(LATT_CUR%B(:,3)**2)**0.5)

          IF (IU0>=0) write(*,*) 'tra',translationsLR(1),translationsLR(2),translationsLR(3)

          ALLOCATE(POSION_multi(3,nions*MBD_SIZE_TOTAL))

          POSION_multi=0._q
        
          CALL multiply_cell(nions,POSION,lattmat_au,MBD_SIZE,POSION_multi,lattmat_au_multi)

!!!!!!!!!!!!!!!!!!!!!!!Long range, frequency independent interaction tensor and its derivatives
          DO mm=NODE_ME,ppqqLR,NCPU
            pp=indexPLR(mm)
            qq=indexQLR(mm)

            pp_=MOD(pp,nions)
            IF (pp_==0) pp_=nions

            qq_=MOD(qq,nions)
            IF (qq_==0) qq_=nions
 
            R0pq=(r0(pp_)+r0(qq_))

            !write(*,*) 'R0pq',R0pq
            !write(*,*) 'xy',pp,qq,pp_,qq_

            CALL Tau_pq_lr(POSION_multi(:,pp),POSION_multi(:,qq),R0pq,dampD,sR,lattmat_au_multi,translationsLR, &
            &             rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
            &              dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
            
                            
            !IF (pp_==1 .AND. qq_==2) THEN
            !  write(*,*) 'tau',tau(1,:)
            !  write(*,*) 'tau',tau(2,:)
            !  write(*,*) 'tau',tau(3,:)
            !ENDIF                            
          
            Tmat_lr(3*pp-2:3*pp,3*qq-2)=tau(:,1)
            Tmat_lr(3*pp-2:3*pp,3*qq-1)=tau(:,2)
            Tmat_lr(3*pp-2:3*pp,3*qq  )=tau(:,3)
            Tmat_lr(3*qq-2:3*qq,3*pp-2)=tau(:,1)
            Tmat_lr(3*qq-2:3*qq,3*pp-1)=tau(:,2)
            Tmat_lr(3*qq-2:3*qq,3*pp  )=tau(:,3)

            IF (LSCSGRAD) THEN
!                 write(*,*) 'dtaux',dtaux
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp-2)=dtaux(:,:)
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp-1)=dtauy(:,:)
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp  )=dtauz(:,:)
                dTmatA_lr(1:3,3*pp-2:3*pp,3*qq-2)=-dtaux(:,:)
                dTmatA_lr(1:3,3*pp-2:3*pp,3*qq-1)=-dtauy(:,:)
                dTmatA_lr(1:3,3*pp-2:3*pp,3*qq  )=-dtauz(:,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,1)=dtau_h0x(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,1)=dtau_h0x(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,1)=dtau_h0x(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,2)=dtau_h0y(1,:) 
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,2)=dtau_h0y(2,:) 
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,2)=dtau_h0y(3,:) 

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,3)=dtau_h0z(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,3)=dtau_h0z(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,3)=dtau_h0z(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,4)=dtau_h1x(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,4)=dtau_h1x(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,4)=dtau_h1x(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,5)=dtau_h1y(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,5)=dtau_h1y(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,5)=dtau_h1y(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,6)=dtau_h1z(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,6)=dtau_h1z(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,6)=dtau_h1z(3,:)
     
                dTmatL_lr(3*pp-2,3*qq-2:3*qq,7)=dtau_h2x(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,7)=dtau_h2x(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,7)=dtau_h2x(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,8)=dtau_h2y(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,8)=dtau_h2y(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,8)=dtau_h2y(3,:)

                dTmatL_lr(3*pp-2,3*qq-2:3*qq,9)=dtau_h2z(1,:)
                dTmatL_lr(3*pp-1,3*qq-2:3*qq,9)=dtau_h2z(2,:)
                dTmatL_lr(3*pp  ,3*qq-2:3*qq,9)=dtau_h2z(3,:) 
!!!!!!!!!!!!!!!!!!!!!!!!!!
                dTmatL_lr(3*qq-2,3*pp-2:3*pp,1)=dtau_h0x(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,1)=dtau_h0x(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,1)=dtau_h0x(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,2)=dtau_h0y(1,:) 
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,2)=dtau_h0y(2,:) 
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,2)=dtau_h0y(3,:) 

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,3)=dtau_h0z(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,3)=dtau_h0z(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,3)=dtau_h0z(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,4)=dtau_h1x(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,4)=dtau_h1x(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,4)=dtau_h1x(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,5)=dtau_h1y(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,5)=dtau_h1y(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,5)=dtau_h1y(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,6)=dtau_h1z(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,6)=dtau_h1z(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,6)=dtau_h1z(3,:)
     
                dTmatL_lr(3*qq-2,3*pp-2:3*pp,7)=dtau_h2x(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,7)=dtau_h2x(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,7)=dtau_h2x(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,8)=dtau_h2y(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,8)=dtau_h2y(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,8)=dtau_h2y(3,:)

                dTmatL_lr(3*qq-2,3*pp-2:3*pp,9)=dtau_h2z(1,:)
                dTmatL_lr(3*qq-1,3*pp-2:3*pp,9)=dtau_h2z(2,:)
                dTmatL_lr(3*qq  ,3*pp-2:3*pp,9)=dtau_h2z(3,:) 

              ENDIF

          ENDDO

#ifdef MPI
          CALL M_sum_d(GRIDC%COMM,Tmat_lr ,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )
!           IF (LSCSGRAD) THEN
!             CALL M_sum_d(GRIDC%COMM,dTmatA_lr ,27*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )
!             CALL M_sum_d(GRIDC%COMM,dTmatL_lr ,81*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )
!           ENDIF
#endif
          

!!!!!!!!!!!!!!!!!!!!!!11Short range part
          !c determine the number of images of simulation cell 
          !c in each direction needed for a given 
          !c cutoff radius         
          translationsSR(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
          translationsSR(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
          translationsSR(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)
         
          !c if desired, switch off the interactions with cell images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              translationsSR(i)=0
            ENDIF
          ENDDO

          !c determine characteristic excitation frequencies
          DO i=1,NIONS
            omegaP(i)=4./3.*c6TS(i)/alphaTS(i)**2
          ENDDO 
          
          !c prepare index tables
          ppqq=nions*(nions+1)/2
          IF (LFIRST) THEN
            ALLOCATE(indexP(ppqq))
            ALLOCATE(indexQ(ppqq))
            CALL index_table2d(nions,ppqq,indexP,indexQ)
          ENDIF

          !c loop over frequencies
          ECFDM=0._q
          DO i=1,odim
            omega=ogrid(i)
            Amat=0._q
            dTmatA=0._q
            dTmatL=0._q
            DO pp=1,nions
              alpha_omega(pp)=alpha_freq(alphaTS(pp),omegaP(pp),omega)
              R_omega(pp)=((2./PI)**0.5*alpha_omega(pp)/3.)**(1./3.)
            ENDDO

            DO mm=NODE_ME,ppqq,NCPU
              pp=indexP(mm)
              qq=indexQ(mm)
           
              Rpq=(R_omega(pp)**2+R_omega(qq)**2)**0.5

              R0pq=(r0(pp)+r0(qq))

              CALL Tau_pq_sr(POSION(:,pp),POSION(:,qq),Rpq,R0pq,dampD,sR,TRANSPOSE(lattmat_au),translationsSR, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
                  
              Amat(3*pp-2:3*pp,3*qq-2)=tau(:,1)
              Amat(3*pp-2:3*pp,3*qq-1)=tau(:,2)
              Amat(3*pp-2:3*pp,3*qq  )=tau(:,3)
              Amat(3*qq-2:3*qq,3*pp-2)=tau(:,1)
              Amat(3*qq-2:3*qq,3*pp-1)=tau(:,2)
              Amat(3*qq-2:3*qq,3*pp  )=tau(:,3)
          
              IF (LSCSGRAD) THEN
                dTmatA(1:3,3*qq-2:3*qq,3*pp-2)=dtaux(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp-1)=dtauy(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp  )=dtauz(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-2)=-dtaux(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-1)=-dtauy(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq  )=-dtauz(:,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,1)=dtau_h0x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,1)=dtau_h0x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,1)=dtau_h0x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,2)=dtau_h0y(1,:) 
                dTmatL(3*pp-1,3*qq-2:3*qq,2)=dtau_h0y(2,:) 
                dTmatL(3*pp  ,3*qq-2:3*qq,2)=dtau_h0y(3,:) 

                dTmatL(3*pp-2,3*qq-2:3*qq,3)=dtau_h0z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,3)=dtau_h0z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,3)=dtau_h0z(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,4)=dtau_h1x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,4)=dtau_h1x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,4)=dtau_h1x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,5)=dtau_h1y(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,5)=dtau_h1y(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,5)=dtau_h1y(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,6)=dtau_h1z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,6)=dtau_h1z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,6)=dtau_h1z(3,:)
     
                dTmatL(3*pp-2,3*qq-2:3*qq,7)=dtau_h2x(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,7)=dtau_h2x(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,7)=dtau_h2x(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,8)=dtau_h2y(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,8)=dtau_h2y(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,8)=dtau_h2y(3,:)

                dTmatL(3*pp-2,3*qq-2:3*qq,9)=dtau_h2z(1,:)
                dTmatL(3*pp-1,3*qq-2:3*qq,9)=dtau_h2z(2,:)
                dTmatL(3*pp  ,3*qq-2:3*qq,9)=dtau_h2z(3,:) 
!!!!!!!!!!!!!!!!!!!!!!!!!!
                dTmatL(3*qq-2,3*pp-2:3*pp,1)=dtau_h0x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,1)=dtau_h0x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,1)=dtau_h0x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,2)=dtau_h0y(1,:) 
                dTmatL(3*qq-1,3*pp-2:3*pp,2)=dtau_h0y(2,:) 
                dTmatL(3*qq  ,3*pp-2:3*pp,2)=dtau_h0y(3,:) 

                dTmatL(3*qq-2,3*pp-2:3*pp,3)=dtau_h0z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,3)=dtau_h0z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,3)=dtau_h0z(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,4)=dtau_h1x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,4)=dtau_h1x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,4)=dtau_h1x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,5)=dtau_h1y(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,5)=dtau_h1y(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,5)=dtau_h1y(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,6)=dtau_h1z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,6)=dtau_h1z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,6)=dtau_h1z(3,:)
     
                dTmatL(3*qq-2,3*pp-2:3*pp,7)=dtau_h2x(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,7)=dtau_h2x(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,7)=dtau_h2x(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,8)=dtau_h2y(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,8)=dtau_h2y(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,8)=dtau_h2y(3,:)

                dTmatL(3*qq-2,3*pp-2:3*pp,9)=dtau_h2z(1,:)
                dTmatL(3*qq-1,3*pp-2:3*pp,9)=dtau_h2z(2,:)
                dTmatL(3*qq  ,3*pp-2:3*pp,9)=dtau_h2z(3,:) 

              ENDIF
            ENDDO

#ifdef MPI
           !c derivatives are synchronized later...
           CALL M_sum_d(GRIDC%COMM, Amat,9*nions*nions )
!            IF (LSCSGRAD) THEN
!             CALL M_sum_d(GRIDC%COMM,dTmatA ,27*nions*nions )
!             CALL M_sum_d(GRIDC%COMM,dTmatL ,81*nions*nions )
!           ENDIF
#endif

            DO pp=1,nions
              Amat(3*pp-2,3*pp-2)=Amat(3*pp-2,3*pp-2)+1._q/alpha_omega(pp)
              Amat(3*pp-1,3*pp-1)=Amat(3*pp-1,3*pp-1)+1._q/alpha_omega(pp)
              Amat(3*pp,    3*pp)=Amat(3*pp,    3*pp)+1._q/alpha_omega(pp)
            ENDDO

            !c matrix inversion
            CALL SVDINVERSE(Amat,3*nions,ninfo)

            ! partial contraction of Amat 
            DO pp=1,nions
              alphaSCS=0._q
              DO qq=1,nions
                DO ii=1,3
                  DO jj=1,3
                    alphaSCS(ii,jj)=alphaSCS(ii,jj)+Amat(3*pp-3+ii,3*qq-3+jj)
                  ENDDO
                ENDDO
              ENDDO
              FalphaTSscreened(i,pp)=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.
              
              !c screened C6
              !!!c6TSscreened(pp)=c6TSscreened(pp)+FalphaTSscreened(i,pp)**2*ogridW(i) 
            ENDDO

!            2235 FORMAT("a",I2,2X,I2,2X,F10.6)
!            IF (IU0>=0) THEN            
!              DO ii=1,nions
!                write(*,2235) i,ii,FalphaTSscreened(i,ii)
!              ENDDO
!            ENDIF

            IF (LSCSGRAD) THEN 
              dalpha=0._q
              DO ppp=1,3*nions+9
                dT=0._q
                IF (ppp .GT. 3*nions) THEN
                  dT=dTmatL(:,:,ppp-3*nions)
                ELSE                   
                  pp=(ppp-1)/3+1
                  DO qq=1,nions
                    dT(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                    dT(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                  ENDDO                  
                ENDIF
#ifdef MPI
                CALL M_sum_d(GRIDC%COMM, dT,9*nions*nions )  
#endif
                dAmat=-MATMUL(Amat,dT)
                dAmat=MATMUL(dAmat,Amat)
                DO pp=1,nions
                  alphaSCS=0._q
                  DO qq=1,nions
                    DO ii=1,3
                      DO jj=1,3
                        alphaSCS(ii,jj)=alphaSCS(ii,jj)+dAmat(3*pp-3+ii,3*qq-3+jj)
                      ENDDO
                    ENDDO
                  ENDDO
                  alphaP=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.

                  ii=MOD(ppp,3)
                  IF (ii==0) ii=3
                  jj=(ppp-ii)/3+1
 
                  dalpha(pp,ii,jj)=alphaP 

                ENDDO
              ENDDO

!             2236 FORMAT("da",I2,2X,I2,2X,F10.6,2X,F10.6,2X,F10.6)
!               
!              IF (IU0>=0) THEN   
!                DO ii=1,nions        
!                  DO pp=1,nions+3
!                    write(*,2236) i,ii,dalpha(ii,1,pp),dalpha(ii,2,pp),dalpha(ii,3,pp)
!                  ENDDO
!                ENDDO
!              ENDIF

            ENDIF

            Amat_lr=0._q
            DO pp=1,nions*MBD_SIZE_TOTAL
              mm=MOD(pp,nions)
              IF (mm==0) mm=nions
              alpha_= FalphaTSscreened(i,mm)
              Amat_lr(3*pp-2,3*pp-2)=-alpha_
              Amat_lr(3*pp-1,3*pp-1)=-alpha_
              Amat_lr(3*pp  ,3*pp  )=-alpha_
            ENDDO

            AT_lr=MATMUL(Amat_lr,Tmat_lr)
            Fmat_lr=0._q
            DO ii=1,3*nions*MBD_SIZE_TOTAL
              Fmat_lr(ii,ii)=1._q
            ENDDO
            Fmat_lr=Fmat_lr-AT_lr
            !AT_lr=Fmat_lr
            
            
            
            !c compute energy
!             CALL EIGVAL_GENERAL(AT_lr,3*nions*MBD_SIZE_TOTAL,d_lr)
            CALL EIGVAL_GENERAL(Fmat_lr,3*nions*MBD_SIZE_TOTAL,d_lr)
            DO pp=1,3*nions*MBD_SIZE_TOTAL
             ! IF (IU0>=0) write(*,*) 'd_lr(pp)',d_lr(pp)
              !ECFDM=ECFDM + ogridW(i)*( log(1.+d_lr(pp))-1.+1./(1.+d_lr(pp) ))
              !ECFDM=ECFDM - ogridW(i) * ( LOG(1._q-d_lr(pp)) )
              IF (d_lr(pp)>0.) THEN
                ECFDM=ECFDM - ogridW(i) * LOG(d_lr(pp)) 
              ELSE
                IF (IU0>=0) WRITE(*,*) "vdW-MBD Warning: d_lr(pp)<0"
              ENDIF
            ENDDO
           
            !c compute gradients
            IF (LSCSGRAD)  THEN
              
              !c common prefactor (1-AT)^-1
              CALL SVDINVERSE(Fmat_lr,3*nions*MBD_SIZE_TOTAL,ninfo)
              

!               DO ppp=1,3*nions*MBD_SIZE_TOTAL+9
!                 dT_lr=0._q
!                 IF (ppp .GT. 3*nions*MBD_SIZE_TOTAL) THEN
!                   dT_lr=dTmatL_lr(:,:,ppp-3*nions*MBD_SIZE_TOTAL)
!                   !c derrivative wrt. lattice vector qq
!                   qq=(ppp-3*nions*MBD_SIZE_TOTAL-1)/3+1
!                 ELSE                   
!                   pp=(ppp-1)/3+1
!                   DO qq=1,nions*MBD_SIZE_TOTAL
!                     dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
!                     dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
!                   ENDDO 
!                   !c derrivative wrt. atom qq
!                   qq=(ppp-1)/(3*MBD_SIZE_TOTAL)+1                 
!                 ENDIF
!                 !c x-,y-,or z- component
!                 pp=MOD(ppp,3)
!                 IF (pp==0) pp=3
! #ifdef MPI
!                 CALL M_sum_d(GRIDC%COMM, dT_lr,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )  
! #endif
!                 dAmat_lr=0._q
!                 DO ii=1,nions*MBD_SIZE_TOTAL
!                   nn=(ii-1)/MBD_SIZE_TOTAL+1                  
!                   dAmat_lr(3*ii-2,3*ii-2) = -dalpha(nn,pp,qq)                 
!                   dAmat_lr(3*ii-1,3*ii-1) = -dalpha(nn,pp,qq)
!                   dAmat_lr(3*ii  ,3*ii  ) = -dalpha(nn,pp,qq)
!                 ENDDO
!                 dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
!  
!                 dT_lr=MATMUL(Amat_lr,dT_lr)
!                 dT_lr=dT_lr+dAmat_lr
!                 dT_lr=MATMUL(Fmat_lr,dT_lr)
!                 
!                 IF (ppp .GT. 3*nions*MBD_SIZE_TOTAL) THEN
!                   DO ii=1,3*nions*MBD_SIZE_TOTAL
!                     dECFDM_L(pp,qq) = dECFDM_L(pp,qq)- ogridW(i)*dT_lr(ii,ii)
!                   ENDDO
!                 ELSE
!                   DO ii=1,3*nions*MBD_SIZE_TOTAL
!                     dECFDM_A(pp,qq) = dECFDM_A(pp,qq)- ogridW(i)*dT_lr(ii,ii)
!                   ENDDO
!                 ENDIF
!               ENDDO
! ! 
! 
              !c x-components of forces
              dT_lr=0._q
              dAmat_lr=0._q
              DO pp=1,nions*MBD_SIZE_TOTAL
                ppp=3*pp-2                                              
                mm=MOD(pp,MBD_SIZE_TOTAL)
                
                DO qq=1,nions*MBD_SIZE_TOTAL
                  dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)+dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                  dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)             
                ENDDO               

                IF (mm==0) THEN
#ifdef MPI
                  CALL M_sum_d(GRIDC%COMM,dT_lr ,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )          
#endif 
                  qq=(pp-1)/MBD_SIZE_TOTAL+1
                  dAmat_lr=0._q
                  DO ii=1,nions*MBD_SIZE_TOTAL
                    nn=(ii-1)/MBD_SIZE_TOTAL+1                  
                    dAmat_lr(3*ii-2,3*ii-2) = -dalpha(nn,1,qq)                 
                    dAmat_lr(3*ii-1,3*ii-1) = -dalpha(nn,1,qq)
                    dAmat_lr(3*ii  ,3*ii  ) = -dalpha(nn,1,qq)
                  ENDDO
                  dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
 
!                   IF (IU0>=0) write(*,*) "dT_lr",dT_lr

                  dT_lr=MATMUL(Amat_lr,dT_lr)
!                    dT_lr=0._q 
                  dT_lr=dT_lr+dAmat_lr
                  dT_lr=MATMUL(Fmat_lr,dT_lr)

                  DO ii=1,3*nions*MBD_SIZE_TOTAL
                    dECFDM_A(1,qq) = dECFDM_A(1,qq)- ogridW(i)*dT_lr(ii,ii)
                  ENDDO
          
                  dT_lr=0._q
                ENDIF 
              ENDDO
              
!               IF (IU0>=0) write(*,*) 'fx',i,dECFDM(1,1),dECFDM(1,2)


              !c y-components of forces
              dT_lr=0._q
              DO pp=1,nions*MBD_SIZE_TOTAL
                ppp=3*pp-1    
                mm=MOD(pp,MBD_SIZE_TOTAL)
                            
                DO qq=1,nions*MBD_SIZE_TOTAL
                  dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)+dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                  dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq) 
                ENDDO   

                IF (mm==0) THEN
#ifdef MPI
                  CALL M_sum_d(GRIDC%COMM,dT_lr ,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )          
#endif 
                  qq=(pp-1)/MBD_SIZE_TOTAL+1
                  dAmat_lr=0._q
                  DO ii=1,nions*MBD_SIZE_TOTAL    
                    nn=(ii-1)/MBD_SIZE_TOTAL+1                  
                    dAmat_lr(3*ii-2,3*ii-2) = -dalpha(nn,2,qq)                 
                    dAmat_lr(3*ii-1,3*ii-1) = -dalpha(nn,2,qq)
                    dAmat_lr(3*ii  ,3*ii  ) = -dalpha(nn,2,qq)
                  ENDDO
                  dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
 
                  dT_lr=MATMUL(Amat_lr,dT_lr)
!                    dT_lr=0._q 
                  dT_lr=dT_lr+dAmat_lr
                  dT_lr=MATMUL(Fmat_lr,dT_lr)
           
                  DO ii=1,3*nions*MBD_SIZE_TOTAL
                    dECFDM_A(2,qq) = dECFDM_A(2,qq)- ogridW(i)*dT_lr(ii,ii)
                  ENDDO
                  dT_lr=0._q
                ENDIF                
              ENDDO   

              !c z-components of forces
              dT_lr=0._q            
              DO pp=1,nions*MBD_SIZE_TOTAL
                ppp=3*pp   
                mm=MOD(pp,MBD_SIZE_TOTAL)  

                DO qq=1,nions*MBD_SIZE_TOTAL 
                  dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)+dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                  dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)                
                ENDDO  

                IF (mm==0) THEN
#ifdef MPI
                  CALL M_sum_d(GRIDC%COMM,dT_lr ,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )          
#endif 
                  qq=(pp-1)/MBD_SIZE_TOTAL+1
                  dAmat_lr=0._q
                  DO ii=1,nions*MBD_SIZE_TOTAL
                    nn=(ii-1)/MBD_SIZE_TOTAL+1                  
                    dAmat_lr(3*ii-2,3*ii-2) = -dalpha(nn,3,qq)                 
                    dAmat_lr(3*ii-1,3*ii-1) = -dalpha(nn,3,qq)
                    dAmat_lr(3*ii  ,3*ii  ) = -dalpha(nn,3,qq)
                  ENDDO
                  dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
 
                  dT_lr=MATMUL(Amat_lr,dT_lr)
!                    dT_lr=0._q 
                  dT_lr=dT_lr+dAmat_lr
                  dT_lr=MATMUL(Fmat_lr,dT_lr)
                  
                  DO ii=1,3*nions*MBD_SIZE_TOTAL
                    dECFDM_A(3,qq) = dECFDM_A(3,qq)- ogridW(i)*dT_lr(ii,ii)
                  ENDDO
                  dT_lr=0._q
                ENDIF                 
              ENDDO  
 
              !c lattice components
              DO ppp=1,9    
                qq=(ppp-1)/3+1
                pp=MOD(ppp,3)
                IF (pp==0) pp=3
                dAmat_lr=0._q
                DO ii=1,nions*MBD_SIZE_TOTAL
                  nn=(ii-1)/MBD_SIZE_TOTAL+1 
!                   IF (IU0>=0) WRITE(*,*) "dalpha",dalpha(nn,pp,nions+qq)
                  dAmat_lr(3*ii-2,3*ii-2) = -dalpha(nn,pp,nions+qq)                 
                  dAmat_lr(3*ii-1,3*ii-1) = -dalpha(nn,pp,nions+qq)
                  dAmat_lr(3*ii  ,3*ii  ) = -dalpha(nn,pp,nions+qq)

                ENDDO

!                 DO ii=1,3*nions
!                   IF (IU0>=0) WRITE(*,*) "dAmat_lr(ii,ii)",i,dAmat_lr(ii,ii)
!                 ENDDO
                
                !IF (i==1) THEN
!                     DO ii=1,3*nions
!                       IF (IU0>=0) WRITE(*,*) "AT(ii,ii)",i,Fmat_lr(ii,ii)
!                     ENDDO
                !ENDIF

                dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
                dAmat_lr=dAmat_lr/MBD_SIZE(qq)
                              
                dT_lr=0._q              
                dT_lr=dTmatL_lr(:,:,ppp)
#ifdef MPI
               CALL M_sum_d(GRIDC%COMM,dT_lr ,9*nions*nions*MBD_SIZE_TOTAL*MBD_SIZE_TOTAL )          
#endif 
               
                dT_lr=MATMUL(Amat_lr,dT_lr)
!                 dT_lr=0._q
                dT_lr=dT_lr+dAmat_lr
                dT_lr=MATMUL(Fmat_lr,dT_lr)


!                 qdumm=0._q
! 
! !                  IF (i==odim) THEN
!                     DO ii=1,3*nions
!                       qdumm=qdumm+dT_lr(ii,ii)                    
!                     ENDDO
!                     IF (IU0>=0) WRITE(*,*) "qdumm",qdumm
! !                   ENDIF              

                DO ii=1,3*nions*MBD_SIZE_TOTAL
                  dECFDM_L(pp,qq) = dECFDM_L(pp,qq)-ogridW(i)*dT_lr(ii,ii)
                ENDDO
              ENDDO  
            ENDIF

          ENDDO
                             
          !c energy per cell in eV
          ECFDM=-ECFDM/(2.*pi*MBD_SIZE_TOTAL)*(2*RYTOEV)
          IF (LSCSGRAD) THEN
            dECFDM_A=-dECFDM_A/(2.*pi*MBD_SIZE_TOTAL)*(2*RYTOEV/AUTOA)

            dECFDM_L=MATMUL(dECFDM_L,lattmat_au_multi)
            dECFDM_L=-dECFDM_L/(2.*pi*MBD_SIZE_TOTAL)*(2*RYTOEV)
          ENDIF
         
          
          DEALLOCATE(POSION_multi)
          DEALLOCATE(Fmat_lr,AT_lr,d_lr,Amat_lr,Tmat_lr)

          IF (LSCSGRAD) THEN
            DEALLOCATE(dT_lr,dTmatA,dTmatL,dTmatL_lr, dTmatA_lr,dAmat_lr)
          ENDIF
 
          IF (LFIRST) LFIRST=.FALSE.
        END SUBROUTINE vdw_tsscs_range_separated

        !SUBROUTINE vdw_tsscs_range_separated_k(GRIDC,alphaTS,c6TS_ev,r0,dampD,sR,nions,POSION,LATT_CUR,alphaTSscreened,&
        !& rmax,LVDWSCS,LSCALER0,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IU0,SYMM,KPOINTS)
        SUBROUTINE vdw_tsscs_range_separated_k(GRIDC,alphaTS,c6TS_ev,r0,dampD,sR,nions,POSION,LATT_CUR,alphaTSscreened,&
        & rmax,LVDWSCS,LSCALER0,LSCSGRAD,LVDW_ONECELL,ECFDM,dECFDM_A,dECFDM_L,MBD_SIZE,IU0,IU6,LMBDKGRID,LEXPANSION,KPOINTS)
        
        !USE full_kpoints

        !c determine SCS corrected C6, alpha, and R0
          TYPE (grid_3d) GRIDC
          TYPE (latt)        LATT_CUR
          INTEGER :: nions,i,j,k,ii,jj,kk,mm,nn,pp,qq,ppqq,ninfo,ppp,IU0,IU6
          INTEGER,PARAMETER :: odim=13
          !INTEGER,PARAMETER :: odim=20
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),tau(3,3),dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          COMPLEX(q) :: tau_k(3,3),dtaux_k(3,3),dtauy_k(3,3),dtauz_k(3,3)
          COMPLEX(q) :: dtau_h0x_k(3,3),dtau_h0y_k(3,3),dtau_h0z_k(3,3)
          COMPLEX(q) :: dtau_h1x_k(3,3),dtau_h1y_k(3,3),dtau_h1z_k(3,3)
          COMPLEX(q) :: dtau_h2x_k(3,3),dtau_h2y_k(3,3),dtau_h2z_k(3,3)
          COMPLEX(q) :: dtau_R0_k(3,3)
          REAL(q) :: alphaTS(nions),c6TS(nions),r0(nions),omegaP(nions)
          REAL(q) :: r0screened(nions)
          REAL(q) :: c6TS_ev(nions)
          REAL(q) :: alphaTSscreened(nions),FalphaTSscreened(odim,nions)
          REAL(q) :: Amat(3*nions,3*nions)
          REAL(q) :: omega,alphaP,alphaQ
          REAL(q) :: ogrid(odim),ogridW(odim)
          REAL(q) :: Rp,Rq,Rpq
          INTEGER :: translationsSR(3),translationsLR(3)
          REAL(q) :: alphaSCS(3,3)
          REAL(q) :: rmax !c cutoff radius in Angstroems        
          LOGICAL,SAVE :: LFIRST=.TRUE.
          LOGICAL :: LVDWSCS,LSCSGRAD,LSCALER0
          LOGICAL :: LVDW_ONECELL(3)
          REAL(q) :: dalpha(odim,nions,3*nions+9)
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: ECFDM
          INTEGER,INTENT(in) :: MBD_SIZE(3)
          INTEGER :: MBD_SIZE_TOTAL
          INTEGER, ALLOCATABLE,SAVE :: indexP(:),indexQ(:)
          REAL(q) :: alpha_omega(nions)
          REAL(q) :: R_omega(nions)
          REAL(q) :: dampD, sR ,R0pq     
          REAL(q) :: dAmat(3*nions+9,3*nions),dT(3*nions,3*nions)
          REAL(q),ALLOCATABLE :: dTmatA(:,:,:),dTmatL(:,:,:)
          !REAL(q),ALLOCATABLE ::  dAmat_lr(:,:)
          COMPLEX(q),ALLOCATABLE ::  dAmat_lr(:,:)
          COMPLEX(q),ALLOCATABLE :: d_lr(:)
          COMPLEX(q),ALLOCATABLE :: Tmat_lr(:,:), dTmatA_lr(:,:,:),dTmatL_lr(:,:,:)
          COMPLEX(q),ALLOCATABLE :: AT_lr(:,:),Fmat_lr(:,:),Amat_lr(:,:)
          REAL(q) :: alpha_
          COMPLEX(q),ALLOCATABLE :: dT_lr(:,:),dT_R0_lr(:,:)        
          REAL(q),INTENT(out) :: dECFDM_A(3,nions),dECFDM_L(3,3) 
          REAL(q) :: ECFDM_tmp,dECFDM_A_tmp(3,nions),dECFDM_L_tmp(3,3)
          INTEGER :: NKPT,jX 
          REAL(q),ALLOCATABLE :: kvect(:,:),kweight(:)
          INTEGER, ALLOCATABLE :: knequiv(:)
          REAL(q) :: qdumm, qdumm1,qdumm2
          COMPLEX(q) :: zdumm0=CMPLX(0._q),zdumm1=CMPLX(1._q),zdummM1=CMPLX(-1._q)
          TYPE (kpoints_struct), OPTIONAL :: KPOINTS
          COMPLEX(q) :: OneMat(3*nions,3*nions),TwoMat(3*nions,3*nions),ThreeMat(3*nions,3*nions)
          COMPLEX(q) :: FourMat(3*nions,3*nions),FiveMat(3*nions,3*nions),SixMat(3*nions,3*nions)
          REAL(q) :: Edisp1,Edisp2,Edisp3,Edisp4,Edisp5,Edisp6
          LOGICAL :: LEXPANSION
          LOGICAL :: LMBDKGRID
          !TYPE (skpoints_full),SAVE,POINTER ::  KPOINTS_INTER_FULL => NULL() ! full interpolated grid
          !TYPE (symmetry), OPTIONAL ::  SYMM
          !INTEGER :: rotmat(3,3)

!           rotmat=0
! 
!           IF (LFIRST) THEN
!             IF (PRESENT(SYMM) .AND. PRESENT(KPOINTS)) THEN
!               IF(.NOT.ASSOCIATED(KPOINTS_INTER_FULL))ALLOCATE(KPOINTS_INTER_FULL)
!               CALL IBZKPT_HF(LATT_CUR,KPOINTS,KPOINTS_INTER_FULL,NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IU0,IU0)
!               write(*,*) "KPOINTS_INTER_FULL%NKPTS",KPOINTS_INTER_FULL%NKPTS
!               DO i=1,KPOINTS_INTER_FULL%NKPTS
!                 IF (IU0>=0) write(*,*) 'kk:', KPOINTS_INTER_FULL%VKPT(1,i),KPOINTS_INTER_FULL%VKPT(2,i),& 
!                 &  KPOINTS_INTER_FULL%VKPT(3,i),KPOINTS_INTER_FULL%WTKPT(i)
!                 IF (IU0>=0) write(*,*) 'neqiv:', KPOINTS_INTER_FULL%NEQUIV(i)
!                 IF (IU0>=0) write(*,*) 'isymop1:', KPOINTS_INTER_FULL%ISYMOP(1,1,i),&
!                 & KPOINTS_INTER_FULL%ISYMOP(2,1,i),KPOINTS_INTER_FULL%ISYMOP(3,1,i)
!                 IF (IU0>=0) write(*,*) 'isymop2:', KPOINTS_INTER_FULL%ISYMOP(1,2,i),&
!                 & KPOINTS_INTER_FULL%ISYMOP(2,2,i),KPOINTS_INTER_FULL%ISYMOP(3,2,i)
!                 IF (IU0>=0) write(*,*) 'isymop3:', KPOINTS_INTER_FULL%ISYMOP(1,3,i),&
!                 & KPOINTS_INTER_FULL%ISYMOP(2,3,i),KPOINTS_INTER_FULL%ISYMOP(3,3,i)
!                 IF (IU0>=0) write(*,*) 'irotop1:', KPOINTS_INTER_FULL%IROTOP(1,1,i),&
!                 & KPOINTS_INTER_FULL%IROTOP(2,1,i),KPOINTS_INTER_FULL%IROTOP(3,1,i)
!                 IF (IU0>=0) write(*,*) 'irotop2:', KPOINTS_INTER_FULL%IROTOP(1,2,i),&
!                 & KPOINTS_INTER_FULL%IROTOP(2,2,i),KPOINTS_INTER_FULL%IROTOP(3,2,i)
!                 IF (IU0>=0) write(*,*) 'irotop3:', KPOINTS_INTER_FULL%IROTOP(1,3,i),&
!                 & KPOINTS_INTER_FULL%IROTOP(2,3,i),KPOINTS_INTER_FULL%IROTOP(3,3,i)
!                 
!               ENDDO
!             ENDIF 
!           ENDIF
  
          qdumm=0._q

          NODE_ME=1
          NCPU=1
#ifdef MPI
           NODE_ME=GRIDC%COMM%NODE_ME
           NCPU=GRIDC%COMM%NCPU
#endif

          !c conversion to atomic units
          !lattmat_au=TRANSPOSE(LATT_CUR%A)/AUTOA
          lattmat_au=(LATT_CUR%A)/AUTOA
          c6TS=c6TS_ev/(2*RYTOEV*AUTOA**6)

          call gauss_chebyshev0(odim, ogrid, ogridW)
!           DO i=1,odim
!             write(*,*) i,ogrid(i),ogridW(i)
!           ENDDO
!!!!!!!!!!!!!!!!!!!!!!!!
          
          
          IF (LSCSGRAD) THEN
            ALLOCATE(dAmat_lr(3*nions,3*nions))
            ALLOCATE(dTmatA_lr(3,3*nions,3*nions))
            ALLOCATE(dTmatL_lr(3*nions,3*nions,9)) 
            ALLOCATE(dT_lr(3*nions,3*nions))
            IF (LSCALER0) THEN
              ALLOCATE(dT_R0_lr(3*nions,3*nions))
              dT_R0_lr=0._q
            ENDIF
            !ALLOCATE(Fmat_lr(3*nions*MBD_SIZE_TOTAL,3*nions*MBD_SIZE_TOTAL))

            ALLOCATE(dTmatA(3,3*nions,3*nions))
            ALLOCATE(dTmatL(3*nions,3*nions,9))     
            dAmat_lr=0._q; dTmatA_lr=0._q ;dTmatL_lr=0._q ; dT_lr=0._q 
            dTmatA=0._q;dTmatL=0._q
          ENDIF

          
          ALLOCATE(Tmat_lr(3*nions,3*nions))
          ALLOCATE(Amat_lr(3*nions,3*nions))
          ALLOCATE(AT_lr(3*nions,3*nions))
          ALLOCATE(Fmat_lr(3*nions,3*nions))
          ALLOCATE(d_lr(3*nions))
          
!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!
         !IF (PRESENT(SYMM) .AND. PRESENT(KPOINTS)) THEN
         !IF (PRESENT(KPOINTS) .AND. (.NOT. LMBDKGRID)) THEN
         IF ((.NOT. LMBDKGRID)) THEN
           NKPT=KPOINTS%NKPTS
           ALLOCATE(kvect(3,NKPT));ALLOCATE(kweight(NKPT));ALLOCATE(knequiv(NKPT))
           DO jX=1,NKPT
             kvect(:,jX)=KPOINTS%VKPT(:,jx)
             kweight(jX)=KPOINTS%WTKPT(jX)
           ENDDO        
!            NKPT=KPOINTS_INTER_FULL%NKPTS
!            ALLOCATE(kvect(3,NKPT));ALLOCATE(kweight(NKPT));ALLOCATE(knequiv(NKPT))
           
!            DO jX=1,NKPT             
!              ii=KPOINTS_INTER_FULL%NEQUIV(jX)
!              kvect(:,jX)=KPOINTS_INTER_FULL%VKPT(:,jX)
!              knequiv(jX)=ii
!              IF (jX .NE. ii) THEN
!                kweight(jX)=0._q
!                kweight(ii)=kweight(ii)+KPOINTS_INTER_FULL%WTKPT(jX)
!              ELSE
!                kweight(jX)=KPOINTS_INTER_FULL%WTKPT(jX)
!              ENDIF
!            ENDDO
         ELSE
           NKPT=MBD_SIZE(1)*MBD_SIZE(2)*MBD_SIZE(3)
           ALLOCATE(kvect(3,NKPT));ALLOCATE(kweight(NKPT));ALLOCATE(knequiv(NKPT))
           kvect=0.;kweight=1./NKPT
           jX=1
           DO i=1,MBD_SIZE(1)
             DO j=1,MBD_SIZE(2)
               DO k=1,MBD_SIZE(3)
!                  kvect(1,jX)=1./MBD_SIZE(1)*(i-1)-0.25
!                  kvect(2,jX)=1./MBD_SIZE(2)*(j-1)-0.25
!                  kvect(3,jX)=1./MBD_SIZE(3)*(k-1)-0.25
                 kvect(1,jX)=1./MBD_SIZE(1)*(i-1)
                 kvect(2,jX)=1./MBD_SIZE(2)*(j-1)
                 kvect(3,jX)=1./MBD_SIZE(3)*(k-1)
                 jX=jX+1
               ENDDO
             ENDDO
           ENDDO
         ENDIF
       
!!!!!!!!!!!!!!!!!!!!!!Short range part
          !c determine the number of images of simulation cell 
          !c in each direction needed for a given 
          !c cutoff radius         
          translationsSR(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)   
          translationsSR(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)   
          translationsSR(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)   

          translationsLR=translationsSR
!           translationsLR(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
!           translationsLR(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
!           translationsLR(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)
         
          !c if desired, switch off the interactions with cell images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              translationsSR(i)=0  
              translationsLR(i)=0
            ENDIF
          ENDDO

          !c determine characteristic excitation frequencies
          DO i=1,NIONS
            omegaP(i)=4./3.*c6TS(i)/alphaTS(i)**2
          ENDDO 
          
          !c prepare index tables
          ppqq=nions*(nions+1)/2
          IF (LFIRST) THEN
            ALLOCATE(indexP(ppqq))
            ALLOCATE(indexQ(ppqq))
            CALL index_table2d(nions,ppqq,indexP,indexQ)
          ENDIF

          !c loop over frequencies
          ECFDM=0._q
          loop_freq_scs: DO i=1,odim
            omega=ogrid(i)
            Amat=0._q
            dTmatA=0._q
            dTmatL=0._q
            DO pp=1,nions
              alpha_omega(pp)=alpha_freq(alphaTS(pp),omegaP(pp),omega)
              R_omega(pp)=((2./PI)**0.5*alpha_omega(pp)/3.)**(1./3.)
            ENDDO

            DO mm=NODE_ME,ppqq,NCPU
              pp=indexP(mm)
              qq=indexQ(mm)
           
              Rpq=(R_omega(pp)*R_omega(pp)+R_omega(qq)*R_omega(qq))**0.5

              R0pq=(r0(pp)+r0(qq))

              CALL Tau_pq_sr(POSION(:,pp),POSION(:,qq),Rpq,R0pq,dampD,sR,lattmat_au,translationsSR, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
                  
              Amat(3*pp-2:3*pp,3*qq-2)=tau(:,1)
              Amat(3*pp-2:3*pp,3*qq-1)=tau(:,2)
              Amat(3*pp-2:3*pp,3*qq  )=tau(:,3)
              Amat(3*qq-2:3*qq,3*pp-2)=tau(:,1)
              Amat(3*qq-2:3*qq,3*pp-1)=tau(:,2)
              Amat(3*qq-2:3*qq,3*pp  )=tau(:,3)
          
              IF (LSCSGRAD) THEN
                dTmatA(1:3,3*qq-2:3*qq,3*pp-2)=dtaux(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp-1)=dtauy(:,:)
                dTmatA(1:3,3*qq-2:3*qq,3*pp  )=dtauz(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-2)=-dtaux(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq-1)=-dtauy(:,:)
                dTmatA(1:3,3*pp-2:3*pp,3*qq  )=-dtauz(:,:)

                dTmatL(3*pp-2:3*pp,3*qq-2,1)=dtau_h0x(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,1)=dtau_h0x(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,1)=dtau_h0x(:,3)

                dTmatL(3*pp-2:3*pp,3*qq-2,2)=dtau_h0y(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,2)=dtau_h0y(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,2)=dtau_h0y(:,3)
               
                dTmatL(3*pp-2:3*pp,3*qq-2,3)=dtau_h0z(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,3)=dtau_h0z(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,3)=dtau_h0z(:,3)

                dTmatL(3*pp-2:3*pp,3*qq-2,4)=dtau_h1x(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,4)=dtau_h1x(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,4)=dtau_h1x(:,3)

                dTmatL(3*pp-2:3*pp,3*qq-2,5)=dtau_h1y(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,5)=dtau_h1y(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,5)=dtau_h1y(:,3)
               
                dTmatL(3*pp-2:3*pp,3*qq-2,6)=dtau_h1z(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,6)=dtau_h1z(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,6)=dtau_h1z(:,3)

                dTmatL(3*pp-2:3*pp,3*qq-2,7)=dtau_h2x(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,7)=dtau_h2x(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,7)=dtau_h2x(:,3)

                dTmatL(3*pp-2:3*pp,3*qq-2,8)=dtau_h2y(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,8)=dtau_h2y(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,8)=dtau_h2y(:,3)
               
                dTmatL(3*pp-2:3*pp,3*qq-2,9)=dtau_h2z(:,1)
                dTmatL(3*pp-2:3*pp,3*qq-1,9)=dtau_h2z(:,2)
                dTmatL(3*pp-2:3*pp,3*qq  ,9)=dtau_h2z(:,3)
!!!!!!!!!!!!!!!!!!!!!!!!!!               
                dTmatL(3*qq-2:3*qq,3*pp-2,1)=dtau_h0x(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,1)=dtau_h0x(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,1)=dtau_h0x(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,2)=dtau_h0y(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,2)=dtau_h0y(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,2)=dtau_h0y(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,3)=dtau_h0z(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,3)=dtau_h0z(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,3)=dtau_h0z(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,4)=dtau_h1x(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,4)=dtau_h1x(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,4)=dtau_h1x(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,5)=dtau_h1y(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,5)=dtau_h1y(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,5)=dtau_h1y(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,6)=dtau_h1z(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,6)=dtau_h1z(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,6)=dtau_h1z(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,7)=dtau_h2x(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,7)=dtau_h2x(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,7)=dtau_h2x(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,8)=dtau_h2y(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,8)=dtau_h2y(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,8)=dtau_h2y(:,3)

                dTmatL(3*qq-2:3*qq,3*pp-2,9)=dtau_h2z(:,1)
                dTmatL(3*qq-2:3*qq,3*pp-1,9)=dtau_h2z(:,2)
                dTmatL(3*qq-2:3*qq,3*pp  ,9)=dtau_h2z(:,3)
              ENDIF
            ENDDO

#ifdef MPI
           !c derivatives are synchronized later...
           CALL M_sum_d(GRIDC%COMM, Amat,9*nions*nions )
#endif

            DO pp=1,nions
              Amat(3*pp-2,3*pp-2)=Amat(3*pp-2,3*pp-2)+1._q/alpha_omega(pp)
              Amat(3*pp-1,3*pp-1)=Amat(3*pp-1,3*pp-1)+1._q/alpha_omega(pp)
              Amat(3*pp,    3*pp)=Amat(3*pp,    3*pp)+1._q/alpha_omega(pp)
            ENDDO

            !c matrix inversion
            !CALL SVDINVERSE(Amat,3*nions,ninfo)
            CALL INVERSE_SYM_D(Amat,3*nions)

            ! partial contraction of Amat 
            DO pp=1,nions
              alphaSCS=0._q
              DO qq=1,nions
                DO ii=1,3
                  DO jj=1,3
                    alphaSCS(ii,jj)=alphaSCS(ii,jj)+Amat(3*pp-3+ii,3*qq-3+jj)
                  ENDDO
                ENDDO
              ENDDO
              FalphaTSscreened(i,pp)=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3. 
            ENDDO

            IF (LSCSGRAD) THEN 
              dalpha(i,:,:)=0._q
              DO ppp=1,3*nions+9
                dT=0._q
                IF (ppp .GT. 3*nions) THEN
                  dT=dTmatL(:,:,ppp-3*nions)
                ELSE                   
                  pp=(ppp-1)/3+1
                  DO qq=1,nions
                    dT(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                    dT(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA(1:3,3*qq-2:3*qq,ppp)
                  ENDDO                  
                ENDIF
#ifdef MPI
                CALL M_sum_d(GRIDC%COMM, dT,9*nions*nions )  
#endif
                dAmat=-MATMUL(Amat,dT)
                dAmat=MATMUL(dAmat,Amat)
!               
!                 qdumm1=1._q;qdumm2=0._q
!                 dAmat=0._q
!                 CALL DGEMM('N','N',3*nions,3*nions,3*nions,qdumm1,Amat,3*nions,dT,3*nions, &
!                   &    qdumm2,dAmat,3*nions)
!                 dT=-dAmat
!                 dAmat=0._q
!                 CALL DGEMM('N','N',3*nions,3*nions,3*nions,qdumm1,dT,3*nions,Amat,3*nions, &
!                   &    qdumm2,dAmat,3*nions)
                
                DO pp=1,nions
                  alphaSCS=0._q
                  DO qq=1,nions
                    DO ii=1,3
                      DO jj=1,3
                        alphaSCS(ii,jj)=alphaSCS(ii,jj)+dAmat(3*pp-3+ii,3*qq-3+jj)
                      ENDDO
                    ENDDO
                  ENDDO
                  alphaP=(alphaSCS(1,1)+alphaSCS(2,2)+alphaSCS(3,3))/3.
                  dalpha(i,pp,ppp)=alphaP
                ENDDO
              ENDDO
            ENDIF
          ENDDO loop_freq_scs
          
          !c rescale R0?
          IF (LSCALER0) THEN
            r0screened(:)=r0*(FalphaTSscreened(odim,:)/alphaTS(:))**(1._q/3._q)
          ELSE
            r0screened=r0
          ENDIF
          
          !c switch off SCS on short range - this is mainly for test 
          !c puroses
          IF (.NOT. LVDWSCS) THEN
            r0screened=r0
            dalpha(:,:,:)=0._q
            DO ii=1,odim
              DO pp=1,nions
                FalphaTSscreened(ii,pp)=alpha_freq(alphaTS(pp),omegaP(pp),omega)
              ENDDO
            ENDDO
          ENDIF
           !c the short-range part is complete, now compute the k-dependent long-range part

!!!!!!!!!!!!!!!!!!!!!!!Long range, frequency independent interaction tensor and its derivatives
          dECFDM_A=0._q;dECFDM_L=0._q

          Fmat_lr=CMPLX(0._q)
          DO ii=1,3*nions
            Fmat_lr(ii,ii)=CMPLX(1._q)
          ENDDO

          !c loop over k-points
          loop_kpt: DO jX=1,NKPT
            IF (kweight(jX)==0.) CYCLE
            !c frequency independent tensor
            Tmat_lr=0._q  
            IF (LSCSGRAD) THEN
              dTmatA_lr=0._q ;dTmatL_lr=0._q  
              IF (LSCALER0) dT_R0_lr=0._q
            ENDIF
            ECFDM_tmp=0._q; dECFDM_L_tmp=0._q; dECFDM_A_tmp=0._q
            DO mm=NODE_ME,ppqq,NCPU
              pp=indexP(mm)
              qq=indexQ(mm)

              R0pq=(r0screened(pp)+r0screened(qq))
              !write(*,*) 'testL1',jX
              CALL Tau_pq_lr_k(kvect(:,jX),POSION(:,pp),POSION(:,qq),R0pq,dampD,sR,lattmat_au,translationsLR, &
              &             rmax/AUTOA,tau_k,dtaux_k,dtauy_k,dtauz_k,dtau_h0x_k,dtau_h0y_k,dtau_h0z_k,dtau_h1x_k,&
              &              dtau_h1y_k,dtau_h1z_k,dtau_h2x_k,dtau_h2y_k,dtau_h2z_k,dtau_R0_k,LSCSGRAD,LSCALER0)
              !write(*,*) 'testL2',jX
                                  
              Tmat_lr(3*pp-2:3*pp,3*qq-2)=tau_k(:,1)
              Tmat_lr(3*pp-2:3*pp,3*qq-1)=tau_k(:,2)
              Tmat_lr(3*pp-2:3*pp,3*qq  )=tau_k(:,3)            
              IF (pp .NE. qq) THEN
                tau_k=CONJG(tau_k)
                Tmat_lr(3*qq-2:3*qq,3*pp-2)=tau_k(:,1)
                Tmat_lr(3*qq-2:3*qq,3*pp-1)=tau_k(:,2)
                Tmat_lr(3*qq-2:3*qq,3*pp  )=tau_k(:,3)               
              ENDIF

              IF (LSCSGRAD) THEN
!                 write(*,*) 'dtaux',dtaux
                dtaux_k=CONJG(dtaux_k)
                dtauy_k=CONJG(dtauy_k)
                dtauz_k=CONJG(dtauz_k)
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp-2)=dtaux_k(:,:)
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp-1)=dtauy_k(:,:)
                dTmatA_lr(1:3,3*qq-2:3*qq,3*pp  )=dtauz_k(:,:)
                
                IF (pp .NE. qq) THEN
!                   dtaux_k=CONJG(dtaux_k)
!                   dtauy_k=CONJG(dtauy_k)
!                   dtauz_k=CONJG(dtauz_k)
                  dTmatA_lr(1:3,3*pp-2:3*pp,3*qq-2)=-dtaux_k(:,:)
                  dTmatA_lr(1:3,3*pp-2:3*pp,3*qq-1)=-dtauy_k(:,:)
                  dTmatA_lr(1:3,3*pp-2:3*pp,3*qq  )=-dtauz_k(:,:)
                ENDIF
                
                IF (LSCALER0) THEN
                  !dtau_R0_k=CONJG(dtau_R0_k)
                  dT_R0_lr(3*pp-2:3*pp,3*qq-2)=dtau_R0_k(:,1)
                  dT_R0_lr(3*pp-2:3*pp,3*qq-1)=dtau_R0_k(:,2)
                  dT_R0_lr(3*pp-2:3*pp,3*qq  )=dtau_R0_k(:,3)
                  IF (pp .NE. qq) THEN
                    dtau_R0_k=CONJG(dtau_R0_k)
                    dT_R0_lr(3*qq-2:3*qq,3*pp-2)=dtau_R0_k(:,1)
                    dT_R0_lr(3*qq-2:3*qq,3*pp-1)=dtau_R0_k(:,2)
                    dT_R0_lr(3*qq-2:3*qq,3*pp  )=dtau_R0_k(:,3)
                  ENDIF
                ENDIF
                
                dTmatL_lr(3*pp-2:3*pp,3*qq-2,1)=dtau_h0x_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,1)=dtau_h0x_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,1)=dtau_h0x_k(:,3)

                dTmatL_lr(3*pp-2:3*pp,3*qq-2,2)=dtau_h0y_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,2)=dtau_h0y_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,2)=dtau_h0y_k(:,3)
               
                dTmatL_lr(3*pp-2:3*pp,3*qq-2,3)=dtau_h0z_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,3)=dtau_h0z_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,3)=dtau_h0z_k(:,3)

                dTmatL_lr(3*pp-2:3*pp,3*qq-2,4)=dtau_h1x_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,4)=dtau_h1x_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,4)=dtau_h1x_k(:,3)

                dTmatL_lr(3*pp-2:3*pp,3*qq-2,5)=dtau_h1y_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,5)=dtau_h1y_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,5)=dtau_h1y_k(:,3)
               
                dTmatL_lr(3*pp-2:3*pp,3*qq-2,6)=dtau_h1z_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,6)=dtau_h1z_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,6)=dtau_h1z_k(:,3)

                dTmatL_lr(3*pp-2:3*pp,3*qq-2,7)=dtau_h2x_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,7)=dtau_h2x_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,7)=dtau_h2x_k(:,3)

                dTmatL_lr(3*pp-2:3*pp,3*qq-2,8)=dtau_h2y_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,8)=dtau_h2y_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,8)=dtau_h2y_k(:,3)
               
                dTmatL_lr(3*pp-2:3*pp,3*qq-2,9)=dtau_h2z_k(:,1)
                dTmatL_lr(3*pp-2:3*pp,3*qq-1,9)=dtau_h2z_k(:,2)
                dTmatL_lr(3*pp-2:3*pp,3*qq  ,9)=dtau_h2z_k(:,3)

!!!!!!!!!!!!!!!!!!!!!!!!!!
                IF (pp .NE. qq) THEN
                  dtau_h0x_k=CONJG(dtau_h0x_k)
                  dtau_h0y_k=CONJG(dtau_h0y_k)
                  dtau_h0z_k=CONJG(dtau_h0z_k)
                  dtau_h1x_k=CONJG(dtau_h1x_k)
                  dtau_h1y_k=CONJG(dtau_h1y_k)
                  dtau_h1z_k=CONJG(dtau_h1z_k)
                  dtau_h2x_k=CONJG(dtau_h2x_k)
                  dtau_h2y_k=CONJG(dtau_h2y_k)
                  dtau_h2z_k=CONJG(dtau_h2z_k)
                ENDIF
                
                dTmatL_lr(3*qq-2:3*qq,3*pp-2,1)=dtau_h0x_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,1)=dtau_h0x_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,1)=dtau_h0x_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,2)=dtau_h0y_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,2)=dtau_h0y_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,2)=dtau_h0y_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,3)=dtau_h0z_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,3)=dtau_h0z_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,3)=dtau_h0z_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,4)=dtau_h1x_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,4)=dtau_h1x_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,4)=dtau_h1x_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,5)=dtau_h1y_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,5)=dtau_h1y_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,5)=dtau_h1y_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,6)=dtau_h1z_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,6)=dtau_h1z_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,6)=dtau_h1z_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,7)=dtau_h2x_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,7)=dtau_h2x_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,7)=dtau_h2x_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,8)=dtau_h2y_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,8)=dtau_h2y_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,8)=dtau_h2y_k(:,3)

                dTmatL_lr(3*qq-2:3*qq,3*pp-2,9)=dtau_h2z_k(:,1)
                dTmatL_lr(3*qq-2:3*qq,3*pp-1,9)=dtau_h2z_k(:,2)
                dTmatL_lr(3*qq-2:3*qq,3*pp  ,9)=dtau_h2z_k(:,3)

              ENDIF
            ENDDO

#ifdef MPI
            CALL M_sum_z(GRIDC%COMM,Tmat_lr ,9*nions*nions )
            IF (LSCSGRAD) THEN
              IF (LSCALER0) CALL M_sum_z(GRIDC%COMM,dT_R0_lr ,9*nions*nions )
              CALL M_sum_z(GRIDC%COMM,dTmatA_lr ,27*nions*nions )
              CALL M_sum_z(GRIDC%COMM,dTmatL_lr ,81*nions*nions )
            ENDIF
#endif
           
            !c loop over frequencies, zero frequency is not needed
            loop_freq: DO i=1,(odim-1)
              Amat_lr=CMPLX(0._q)
              DO pp=1,nions
                alpha_= FalphaTSscreened(i,pp)
                Amat_lr(3*pp-2,3*pp-2)=CMPLX(-alpha_)
                Amat_lr(3*pp-1,3*pp-1)=CMPLX(-alpha_)
                Amat_lr(3*pp  ,3*pp  )=CMPLX(-alpha_)
              ENDDO
              
              
              IF (LEXPANSION) THEN
                OneMat=0.;TwoMat=0.;ThreeMat=0.;FourMat=0.
                FiveMat=0.;SixMat=0.
                !zdumm1=CMPLX(1._q);zdumm2=CMPLX(0._q) 
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,Amat_lr,3*nions,Tmat_lr,3*nions, &
                &    zdumm0,OneMat,3*nions) 
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,OneMat,3*nions,OneMat,3*nions, &
                &    zdumm0,TwoMat,3*nions) 
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,TwoMat,3*nions,OneMat,3*nions, &
                &    zdumm0,ThreeMat,3*nions)
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,ThreeMat,3*nions,OneMat,3*nions, &
                &    zdumm0,FourMat,3*nions)
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,FourMat,3*nions,OneMat,3*nions, &
                &    zdumm0,FiveMat,3*nions)
                CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,FiveMat,3*nions,OneMat,3*nions, &
                &    zdumm0,SixMat,3*nions)
              ENDIF
              
              
              !c 1-AT
              !zdumm1=CMPLX(-1._q);zdumm2=CMPLX(1._q) 
              AT_lr=Fmat_lr
              CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdummM1,Amat_lr,3*nions,Tmat_lr,3*nions, &
              &    zdumm1,AT_lr,3*nions)
              
                                     
              !c compute energy
              CALL ZEIGVAL_GENERAL(AT_lr,3*nions,d_lr)
              !CALL ZEIGVAL_GENERAL(Fmat_lr,3*nions,d_lr)

              DO pp=1,3*nions
                !IF (IU0>=0) WRITE(*,*) 'd_lr(pp)',d_lr(pp)
                IF (REAL(d_lr(pp))>0.) THEN
                  ECFDM_tmp=ECFDM_tmp-ogridW(i) * LOG(REAL(d_lr(pp)))
                  !!ECFDM=ECFDM - ogridW(i) * LOG(REAL(d_lr(pp)))*kweight(jX) 
                ELSE
                  IF (IU0>=0) WRITE(*,*) "Error(vdw_tsscs_range_separated_k): d_lr(pp)<=0"
                  STOP
                ENDIF
              ENDDO
              
              IF (LEXPANSION) THEN
                DO pp=1,3*nions
                  Edisp1=Edisp1-ogridW(i) * REAL(OneMat(pp,pp))*kweight(jX)
                  Edisp2=Edisp2-ogridW(i) * REAL(TwoMat(pp,pp))*kweight(jX)
                  Edisp3=Edisp3-ogridW(i) * REAL(ThreeMat(pp,pp))*kweight(jX)
                  Edisp4=Edisp4-ogridW(i) * REAL(FourMat(pp,pp))*kweight(jX)
                  Edisp5=Edisp5-ogridW(i) * REAL(FiveMat(pp,pp))*kweight(jX)
                  Edisp6=Edisp6-ogridW(i) * REAL(SixMat(pp,pp))*kweight(jX)
                ENDDO
              ENDIF
           
              !c compute gradients
              IF (LSCSGRAD)  THEN
              
                !c common prefactor (1-AT)^-1
                CALL INVERSE_Z(AT_lr,3*nions)
                
                !c x-components of forces
                DO pp=1,nions                  
                  ppp=3*pp-2                                                             
                
                  dAmat_lr=CMPLX(0._q)
                  DO ii=1,nions
                    dAmat_lr(3*ii-2,3*ii-2) = CMPLX(-dalpha(i,ii,ppp))                 
                    dAmat_lr(3*ii-1,3*ii-1) = CMPLX(-dalpha(i,ii,ppp))
                    dAmat_lr(3*ii  ,3*ii  ) = CMPLX(-dalpha(i,ii,ppp))
                  ENDDO

                  !zdumm1=CMPLX(1._q);zdumm2=CMPLX(0._q) 
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,dAmat_lr,3*nions,Tmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                  dAmat_lr=dT_lr
                                
                  dT_lr=CMPLX(0._q)              
                  DO qq=1,nions
                    IF (pp .GE. qq) THEN
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                    ELSE
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                    ENDIF                    
                  ENDDO               

! #ifdef MPI
!                   CALL M_sum_z(GRIDC%COMM,dT_lr ,9*nions*nions )          
! #endif 
                  IF (LSCALER0) THEN
                    DO ii=1,nions
                      qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp) 
                      qdumm=2*qdumm*sR/3._q
                      dT_lr(3*ii-2:3*ii,3*ii-2)=dT_lr(3*ii-2:3*ii,3*ii-2)+dT_R0_lr(3*ii-2:3*ii,3*ii-2)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii-1)=dT_lr(3*ii-2:3*ii,3*ii-1)+dT_R0_lr(3*ii-2:3*ii,3*ii-1)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii  )=dT_lr(3*ii-2:3*ii,3*ii  )+dT_R0_lr(3*ii-2:3*ii,3*ii  )*qdumm
                      DO jj=ii+1,nions
                        qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp) + &
                              r0screened(jj)/FalphaTSscreened(odim,jj)*dalpha(odim,jj,ppp)     
                        qdumm=qdumm*sR/3._q
                        dT_lr(3*ii-2:3*ii,3*jj-2)=dT_lr(3*ii-2:3*ii,3*jj-2)+dT_R0_lr(3*ii-2:3*ii,3*jj-2)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj-1)=dT_lr(3*ii-2:3*ii,3*jj-1)+dT_R0_lr(3*ii-2:3*ii,3*jj-1)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj  )=dT_lr(3*ii-2:3*ii,3*jj  )+dT_R0_lr(3*ii-2:3*ii,3*jj  )*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-2)=dT_lr(3*jj-2:3*jj,3*ii-2)+dT_R0_lr(3*jj-2:3*jj,3*ii-2)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-1)=dT_lr(3*jj-2:3*jj,3*ii-1)+dT_R0_lr(3*jj-2:3*jj,3*ii-1)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii  )=dT_lr(3*jj-2:3*jj,3*ii  )+dT_R0_lr(3*jj-2:3*jj,3*ii  )*qdumm 
                      ENDDO
                    ENDDO  
                  ENDIF 
              
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,Amat_lr,3*nions,dT_lr,3*nions, &
                  &    zdumm1,dAmat_lr,3*nions)
                 ! dT_lr=dAmat_lr
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,AT_lr,3*nions,dAmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)

                  DO ii=1,3*nions
                    !!dECFDM_A(1,pp) = dECFDM_A(1,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*kweight(jX)
                    dECFDM_A_tmp(1,pp) = dECFDM_A_tmp(1,pp)- ogridW(i)*REAL(dT_lr(ii,ii))
!                     dECFDM_A(1,pp) = dECFDM_A(1,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*KPOINTS_INTER_FULL%WTKPT(jX)
                  ENDDO   
                ENDDO
              
                !c y-components of forces
                DO pp=1,nions  
                  ppp=3*pp-1    

                  dAmat_lr=CMPLX(0._q)
                  DO ii=1,nions    
                    dAmat_lr(3*ii-2,3*ii-2) = CMPLX(-dalpha(i,ii,ppp))                 
                    dAmat_lr(3*ii-1,3*ii-1) = CMPLX(-dalpha(i,ii,ppp))
                    dAmat_lr(3*ii  ,3*ii  ) = CMPLX(-dalpha(i,ii,ppp))
                  ENDDO
                  !dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
                  !zdumm1=CMPLX(1._q);zdumm2=CMPLX(0._q) 
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,dAmat_lr,3*nions,Tmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                  dAmat_lr=dT_lr
                  
                  dT_lr=CMPLX(0._q)  
                  DO qq=1,nions
                    IF (pp .GE. qq) THEN
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                    ELSE
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                    ENDIF
                  ENDDO           

! #ifdef MPI
!                   CALL M_sum_z(GRIDC%COMM,dT_lr ,9*nions*nions )          
! #endif 
                  IF (LSCALER0) THEN
                    DO ii=1,nions
                      qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp)
                      qdumm=2*qdumm*sR/3._q
                      dT_lr(3*ii-2:3*ii,3*ii-2)=dT_lr(3*ii-2:3*ii,3*ii-2)+dT_R0_lr(3*ii-2:3*ii,3*ii-2)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii-1)=dT_lr(3*ii-2:3*ii,3*ii-1)+dT_R0_lr(3*ii-2:3*ii,3*ii-1)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii  )=dT_lr(3*ii-2:3*ii,3*ii  )+dT_R0_lr(3*ii-2:3*ii,3*ii  )*qdumm
                      DO jj=ii+1,nions 
                        qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp) + &
                              r0screened(jj)/FalphaTSscreened(odim,jj)*dalpha(odim,jj,ppp)
                        qdumm=qdumm*sR/3._q
                        dT_lr(3*ii-2:3*ii,3*jj-2)=dT_lr(3*ii-2:3*ii,3*jj-2)+dT_R0_lr(3*ii-2:3*ii,3*jj-2)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj-1)=dT_lr(3*ii-2:3*ii,3*jj-1)+dT_R0_lr(3*ii-2:3*ii,3*jj-1)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj  )=dT_lr(3*ii-2:3*ii,3*jj  )+dT_R0_lr(3*ii-2:3*ii,3*jj  )*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-2)=dT_lr(3*jj-2:3*jj,3*ii-2)+dT_R0_lr(3*jj-2:3*jj,3*ii-2)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-1)=dT_lr(3*jj-2:3*jj,3*ii-1)+dT_R0_lr(3*jj-2:3*jj,3*ii-1)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii  )=dT_lr(3*jj-2:3*jj,3*ii  )+dT_R0_lr(3*jj-2:3*jj,3*ii  )*qdumm 
                      ENDDO
                    ENDDO  
                  ENDIF 

                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,Amat_lr,3*nions,dT_lr,3*nions, &
                  &    zdumm1,dAmat_lr,3*nions)
                 ! dT_lr=dAmat_lr
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,AT_lr,3*nions,dAmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)

                  DO ii=1,3*nions
                    !!dECFDM_A(2,pp) = dECFDM_A(2,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*kweight(jX) 
                    dECFDM_A_tmp(2,pp) = dECFDM_A_tmp(2,pp)- ogridW(i)*REAL(dT_lr(ii,ii))
!                     dECFDM_A(2,pp) = dECFDM_A(2,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*KPOINTS_INTER_FULL%WTKPT(jX)
                  ENDDO           
                ENDDO   

                !c z-components of forces            
                DO pp=1,nions
                  ppp=3*pp    

                  dAmat_lr=CMPLX(0._q)
                  DO ii=1,nions
                    dAmat_lr(3*ii-2,3*ii-2) = CMPLX(-dalpha(i,ii,ppp))                 
                    dAmat_lr(3*ii-1,3*ii-1) = CMPLX(-dalpha(i,ii,ppp))
                    dAmat_lr(3*ii  ,3*ii  ) = CMPLX(-dalpha(i,ii,ppp))
                  ENDDO
                  !dAmat_lr=MATMUL(dAmat_lr,Tmat_lr)
                  !zdumm1=CMPLX(1._q);zdumm2=CMPLX(0._q) 
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,dAmat_lr,3*nions,Tmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                  dAmat_lr=dT_lr
                  
                  dT_lr=CMPLX(0._q)  
                  DO qq=1,nions
                    IF (pp .GE. qq) THEN
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                    ELSE
                      dT_lr(3*pp-2:3*pp,3*qq-2:3*qq)=CONJG(dTmatA_lr(1:3,3*qq-2:3*qq,ppp))
                      dT_lr(3*qq-2:3*qq,3*pp-2:3*pp)=dTmatA_lr(1:3,3*qq-2:3*qq,ppp)
                    ENDIF    
                  ENDDO     

! #ifdef MPI
!                   CALL M_sum_z(GRIDC%COMM,dT_lr ,9*nions*nions )          
! #endif                                     
                  IF (LSCALER0) THEN
                    DO ii=1,nions
                      qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp)
                      qdumm=2*qdumm*sR/3._q
                      dT_lr(3*ii-2:3*ii,3*ii-2)=dT_lr(3*ii-2:3*ii,3*ii-2)+dT_R0_lr(3*ii-2:3*ii,3*ii-2)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii-1)=dT_lr(3*ii-2:3*ii,3*ii-1)+dT_R0_lr(3*ii-2:3*ii,3*ii-1)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii  )=dT_lr(3*ii-2:3*ii,3*ii  )+dT_R0_lr(3*ii-2:3*ii,3*ii  )*qdumm
                      DO jj=ii+1,nions
                        qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,ppp) + &
                              r0screened(jj)/FalphaTSscreened(odim,jj)*dalpha(odim,jj,ppp)
                        qdumm=qdumm*sR/3._q
                        dT_lr(3*ii-2:3*ii,3*jj-2)=dT_lr(3*ii-2:3*ii,3*jj-2)+dT_R0_lr(3*ii-2:3*ii,3*jj-2)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj-1)=dT_lr(3*ii-2:3*ii,3*jj-1)+dT_R0_lr(3*ii-2:3*ii,3*jj-1)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj  )=dT_lr(3*ii-2:3*ii,3*jj  )+dT_R0_lr(3*ii-2:3*ii,3*jj  )*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-2)=dT_lr(3*jj-2:3*jj,3*ii-2)+dT_R0_lr(3*jj-2:3*jj,3*ii-2)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-1)=dT_lr(3*jj-2:3*jj,3*ii-1)+dT_R0_lr(3*jj-2:3*jj,3*ii-1)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii  )=dT_lr(3*jj-2:3*jj,3*ii  )+dT_R0_lr(3*jj-2:3*jj,3*ii  )*qdumm 
                      ENDDO
                    ENDDO  
                  ENDIF 

                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,Amat_lr,3*nions,dT_lr,3*nions, &
                  &    zdumm1,dAmat_lr,3*nions)
                 ! dT_lr=dAmat_lr
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,AT_lr,3*nions,dAmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                  
                  DO ii=1,3*nions
                    !!dECFDM_A(3,pp) = dECFDM_A(3,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*kweight(jX)     
                    dECFDM_A_tmp(3,pp) = dECFDM_A_tmp(3,pp)- ogridW(i)*REAL(dT_lr(ii,ii))
!                     dECFDM_A(3,pp) = dECFDM_A(3,pp)- ogridW(i)*REAL(dT_lr(ii,ii))*KPOINTS_INTER_FULL%WTKPT(jX)
                  ENDDO                 
                ENDDO  
 
                !c lattice components
                DO ppp=1,9    
                  nn=3*nions+ppp                                                
                  qq=(ppp-1)/3+1
                  pp=MOD(ppp,3)
                  IF (pp==0) pp=3
                  dAmat_lr=0._q
                  
                  DO ii=1,nions
                    dAmat_lr(3*ii-2,3*ii-2) = CMPLX(-dalpha(i,ii,nn))                 
                    dAmat_lr(3*ii-1,3*ii-1) = CMPLX(-dalpha(i,ii,nn))
                    dAmat_lr(3*ii  ,3*ii  ) = CMPLX(-dalpha(i,ii,nn))
                  ENDDO

                  !zdumm1=CMPLX(1._q);zdumm2=CMPLX(0._q) 
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,dAmat_lr,3*nions,Tmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                  dAmat_lr=dT_lr
                              
                  dT_lr=CMPLX(0._q)              
                  dT_lr=dTmatL_lr(:,:,ppp)
                  
! #ifdef MPI
!                   CALL M_sum_z(GRIDC%COMM,dT_lr ,9*nions*nions )          
! #endif                
                  IF (LSCALER0) THEN
                    DO ii=1,nions
                      qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,nn)  
                      qdumm=2*qdumm*sR/3._q
                      dT_lr(3*ii-2:3*ii,3*ii-2)=dT_lr(3*ii-2:3*ii,3*ii-2)+dT_R0_lr(3*ii-2:3*ii,3*ii-2)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii-1)=dT_lr(3*ii-2:3*ii,3*ii-1)+dT_R0_lr(3*ii-2:3*ii,3*ii-1)*qdumm
                      dT_lr(3*ii-2:3*ii,3*ii  )=dT_lr(3*ii-2:3*ii,3*ii  )+dT_R0_lr(3*ii-2:3*ii,3*ii  )*qdumm
                      DO jj=ii+1,nions
                        qdumm=r0screened(ii)/FalphaTSscreened(odim,ii)*dalpha(odim,ii,nn) + &
                              r0screened(jj)/FalphaTSscreened(odim,jj)*dalpha(odim,jj,nn)
                        qdumm=qdumm*sR/3._q
                        dT_lr(3*ii-2:3*ii,3*jj-2)=dT_lr(3*ii-2:3*ii,3*jj-2)+dT_R0_lr(3*ii-2:3*ii,3*jj-2)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj-1)=dT_lr(3*ii-2:3*ii,3*jj-1)+dT_R0_lr(3*ii-2:3*ii,3*jj-1)*qdumm
                        dT_lr(3*ii-2:3*ii,3*jj  )=dT_lr(3*ii-2:3*ii,3*jj  )+dT_R0_lr(3*ii-2:3*ii,3*jj  )*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-2)=dT_lr(3*jj-2:3*jj,3*ii-2)+dT_R0_lr(3*jj-2:3*jj,3*ii-2)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii-1)=dT_lr(3*jj-2:3*jj,3*ii-1)+dT_R0_lr(3*jj-2:3*jj,3*ii-1)*qdumm
                        dT_lr(3*jj-2:3*jj,3*ii  )=dT_lr(3*jj-2:3*jj,3*ii  )+dT_R0_lr(3*jj-2:3*jj,3*ii  )*qdumm 
                      ENDDO
                    ENDDO  
                  ENDIF      
                
                                                     
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,Amat_lr,3*nions,dT_lr,3*nions, &
                  &    zdumm1,dAmat_lr,3*nions)
                  !dT_lr=dAmat_lr
                  CALL ZGEMM('N','N',3*nions,3*nions,3*nions,zdumm1,AT_lr,3*nions,dAmat_lr,3*nions, &
                  &    zdumm0,dT_lr,3*nions)
                                   
                  DO ii=1,3*nions
                    !!dECFDM_L(pp,qq) = dECFDM_L(pp,qq)-ogridW(i)*REAL(dT_lr(ii,ii))*kweight(jX) 
                    dECFDM_L_tmp(pp,qq) = dECFDM_L_tmp(pp,qq)-ogridW(i)*REAL(dT_lr(ii,ii))
                  ENDDO
                ENDDO 
                !ENDDO sym_loop
              ENDIF

            ENDDO loop_freq ! frequencies          
            ECFDM=ECFDM+ECFDM_tmp*kweight(jX)
            IF (LSCSGRAD) THEN
              dECFDM_A=dECFDM_A+dECFDM_A_tmp*kweight(jX)
              dECFDM_L=dECFDM_L+dECFDM_L_tmp*kweight(jX)
            END IF

            IF (IU0>=0) THEN
              WRITE(*,*) 'kE:',kvect(1,jX),kvect(2,jX),kvect(3,jX)
              WRITE(*,*) 'E(k)',-ECFDM_tmp/(TPI*SUM(kweight))
              IF (LSCSGRAD) THEN
                DO pp=1,nions
                  WRITE(*,*) 'dE/dx',dECFDM_A_tmp(1,pp),dECFDM_A_tmp(2,pp),dECFDM_A_tmp(3,pp)
                ENDDO
                DO pp=1,3
                  WRITE(*,*) 'sigma',dECFDM_L_tmp(1,pp),dECFDM_L_tmp(2,pp),dECFDM_L_tmp(3,pp)
                ENDDO
              ENDIF
            ENDIF
          ENDDO loop_kpt !k-points
     
         !c energy per cell in eV
          ECFDM=-ECFDM/(TPI)*(2*RYTOEV)
          IF (LSCSGRAD) THEN
            !dECFDM_A=-dECFDM_A/(2.*pi*SUM(kweight))*(2*RYTOEV/AUTOA)
            dECFDM_A=-dECFDM_A/(TPI)*(2*RYTOEV/AUTOA)

            dECFDM_L=MATMUL(dECFDM_L,TRANSPOSE(lattmat_au))
            !dECFDM_L=-dECFDM_L/(2.*pi*SUM(kweight))*(2*RYTOEV)
            dECFDM_L=-dECFDM_L/(TPI)*(2*RYTOEV)
          ENDIF
         
          IF (LEXPANSION) THEN
            Edisp1=Edisp1/(TPI)*(2*RYTOEV)
            Edisp2=Edisp2/(TPI)*(2*RYTOEV)/2.
            Edisp3=Edisp3/(TPI)*(2*RYTOEV)/3.
            Edisp4=Edisp4/(TPI)*(2*RYTOEV)/4.
            Edisp5=Edisp5/(TPI)*(2*RYTOEV)/5.
            Edisp6=Edisp6/(TPI)*(2*RYTOEV)/6.
            IF (IU0>=0) THEN
              WRITE(IU6,*)
              WRITE(IU6,*) 'Nth order contribution to Edisp (eV)'
              WRITE(IU6,*) 'Edisp(N=2):',Edisp2
              WRITE(IU6,*) 'Edisp(N=3):',Edisp3
              WRITE(IU6,*) 'Edisp(N=4):',Edisp4
              WRITE(IU6,*) 'Edisp(N=5):',Edisp5
              WRITE(IU6,*) 'Edisp(N=6):',Edisp6
              WRITE(IU6,*) 'Edisp(N=inf):',ECFDM
            ENDIF
          ENDIF
                          
          DEALLOCATE(knequiv,kweight,kvect)

          DEALLOCATE(Fmat_lr,AT_lr,d_lr,Amat_lr,Tmat_lr)

          IF (LSCSGRAD) THEN
            IF (LSCALER0) THEN
              DEALLOCATE(dT_R0_lr)
            ENDIF
            DEALLOCATE(dT_lr,dTmatA,dTmatL,dTmatL_lr, dTmatA_lr,dAmat_lr)
          ENDIF
 
          IF (LFIRST) LFIRST=.FALSE.
        END SUBROUTINE vdw_tsscs_range_separated_k

        SUBROUTINE vdw_mbd(GRIDC,alphaTS,c6TS_ev,r0,nions,POSION,LATT_CUR,alphaTSscreened,dalpha,&
        & c6TSscreened, dcA,r0TSscreened,rmax,LSCALER0,LSCSGRAD,LCFDM,LVDW_ONECELL,TOTEN,BETA,IU0)
          TYPE (grid_3d) GRIDC
          TYPE (latt)        LATT_CUR
          INTEGER :: nions,i,j,ii,jj,pp,qq,ninfo,ppp,IU0
          INTEGER,PARAMETER :: odim=20
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),tau(3,3),dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: alphaTS(nions),c6TS(nions),r0(nions),omegaP(nions)
          REAL(q) :: c6TS_ev(nions)
          REAL(q) :: alphaTSscreened(nions),c6TSscreened(nions),FalphaTSscreened(odim,nions)
          REAL(q) :: r0TSscreened(nions)
          REAL(q) :: Amat(3*nions,3*nions),Tmat(3*nions,3*nions)
          REAL(q) :: omega,alphaP,alphaQ
          REAL(q) :: ogrid(odim),ogridW(odim)
          REAL(q) :: Rp,Rq,Rpq
          INTEGER :: translations(3)
          REAL(q) :: alphaSCS(3,3)
          REAL(q) :: rmax !c cutoff radius in Angstroems
          REAL(q) :: dTmat(3*nions+9,3*nions,3*nions),dAmat(3*nions+9,3*nions)
          LOGICAL :: LSCALER0,LSCSGRAD,LCFDM
          LOGICAL :: LVDW_ONECELL(3)
          REAL(q) :: dalpha(nions,3,nions+3),dca(nions,3,nions+3)
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: Edisp1,Edisp2,TOTEN
          REAL(q) :: BETA !range separating parameter 
          REAL(q) :: d(3*nions)         !c eigenvalues of the matrix
    
          NODE_ME=1
          NCPU=1
#ifdef MPI
          NODE_ME=GRIDC%COMM%NODE_ME
          NCPU=GRIDC%COMM%NCPU
#endif

          !c conversion to atomic units
          lattmat_au=TRANSPOSE(LATT_CUR%A)/AUTOA
          c6TS=c6TS_ev/(2*RYTOEV*AUTOA**6)

          !c Gauss-Legendre grid and weigths for integration over frequencies
!           ogrid=(/0.0392901,0.118358,0.198912,0.282029,0.368919,&
!                    &0.461006,0.560027,0.668179,0.788336,0.92439,&
!                    &1.08179,1.26849,1.49661,1.78563,2.16917,&
!                    &2.71062,3.54573,5.02734,8.44896,25.4517/)
!           ogridW=(/0.0786611,0.07964,0.0816475,0.0847872,0.0892294,&
!                    &0.0952317,0.103172,0.113605,0.12735,0.145652,&
!                    &0.170453,0.204917,0.254456,0.328962,0.448092,&
!                    &0.655606,1.06596,2.06357,5.6851,50.9558/)

          call gauss_chebyshev(odim, ogrid, ogridW)

          !c determine the number of images of simulation cell 
          !c in each direction needed for a given 
          !c cutoff radius         
          translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
          translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
          translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)
         
          !c if desired, switch off the interactions with cell images
          DO i=1,3
            IF (LVDW_ONECELL(i)) THEN
              translations(i)=0
            ENDIF
          ENDDO

          !c determine characteristic excitation frequencies
          DO i=1,NIONS
            omegaP(i)=4./3.*c6TS(i)/alphaTS(i)**2
          ENDDO 
          
          Edisp1=0._q
          Edisp2=0._q
          !c loop over frequencies
          DO i=1,odim
            omega=ogrid(i)
            Amat=0._q
            Tmat=0._q
            
            !c construct AT matrix for each frequency
            DO pp=1,nions
              alphaP=alpha_freq(alphaTS(pp),omegaP(pp),omega)
              !Rp=((2./PI)**0.5*alphaP/3.)**(1./3.)
              Rp=((2./PI)**0.5*alphaTS(pp)/3.)**(1./3.)
              Rpq=beta*(Rp**2+Rp**2)**0.5
            
              CALL scsTau_pq(POSION(:,pp),POSION(:,pp),Rpq,lattmat_au,translations, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,.FALSE.)

! #ifdef MPI
!                 CALL M_sum_d(GRIDC%COMM, tau, 9 )          
! #endif

              Tmat(3*pp-2,3*pp-2:3*pp)=tau(1,:)
              Tmat(3*pp-1,3*pp-2:3*pp)=tau(2,:)
              Tmat(3*pp  ,3*pp-2:3*pp)=tau(3,:)
     
              Amat(3*pp-2,3*pp-2)=-alphaP
              Amat(3*pp-1,3*pp-1)=-alphaP
              Amat(3*pp  ,3*pp  )=-alphaP

              DO qq=pp+1,nions
 
                !c alphas should be computed in a separate loop!!!
                alphaQ=alpha_freq(alphaTS(qq),omegaP(qq),omega) 
                !Rq=((2./PI)**0.5*alphaQ/3.)**(1./3.)
                Rq=((2./PI)**0.5*alphaTS(qq)/3.)**(1./3.)
                Rpq=beta*(Rp**2+Rq**2)**0.5
                  
                CALL scsTau_pq(POSION(:,pp),POSION(:,qq),Rpq,lattmat_au,translations, &
                     & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                     & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,.FALSE.)

! #ifdef MPI
!                 CALL M_sum_d(GRIDC%COMM, tau,9 )
! #endif
                  
                Tmat(3*pp-2,3*qq-2:3*qq)=tau(1,:)
                Tmat(3*pp-1,3*qq-2:3*qq)=tau(2,:)
                Tmat(3*pp  ,3*qq-2:3*qq)=tau(3,:)
                Tmat(3*qq-2,3*pp-2:3*pp)=tau(1,:)
                Tmat(3*qq-1,3*pp-2:3*pp)=tau(2,:)
                Tmat(3*qq  ,3*pp-2:3*pp)=tau(3,:)

              ENDDO
            ENDDO

            Amat=MATMUL(Amat,Tmat)

            CALL EIGVAL_GENERAL(Amat,3*nions,d)

            DO pp=1,3*nions
              Edisp1=Edisp1 + ogridW(i) * ( LOG(1._q+d(pp))-1._q+1._q/(1._q+d(pp)) )
              Edisp2=Edisp2 + ogridW(i) * ( LOG(1._q-d(pp)) )
!               Edisp2=Edisp2 + ogridW(i) * ( LOG(1._q-d(pp)) + d(pp) )
            ENDDO

          ENDDO

          Edisp1=-Edisp1/TPI
          Edisp2=Edisp2/TPI
          
          IF (IU0>=0) THEN
            write(*,*) 'Edisp1:',Edisp1,"(au)"           
            write(*,*) 'Edisp2:',Edisp2,"(au)"           
          ENDIF

          Edisp1=Edisp1*2*RYTOEV
          Edisp2=Edisp2*2*RYTOEV

          IF (IU0>=0) THEN
            write(*,*) 'Edisp1:',Edisp1,"(eV)"
            !TOTEN=TOTEN+ECFDM
            write(*,*) 'TOTEN+Edisp1:',TOTEN+Edisp1,"(eV)"
            write(*,*) 'Edisp2:',Edisp2,"(eV)"
            !TOTEN=TOTEN+ECFDM
            write(*,*) 'TOTEN+Edisp2:',TOTEN+Edisp2,"(eV)"
            TOTEN=TOTEN+Edisp1
!             TOTEN=TOTEN+Edisp2
          ENDIF
        END SUBROUTINE vdw_mbd

!=====================================================================================
  subroutine gauss_chebyshev(nb, omega, weight)
!-------------------------------------------------------------------------------------
! Description: Gauss-Chebyshev grid and weights for frequency integratiion
!
! Created   : G. Jansen, J.G. Angyan, 13 Mar 2009
!-------------------------------------------------------------------------------------
  implicit none
  !include "common/tapes"

! input
  integer, intent(in)    :: nb

! input/output
  real(q), intent(inout) :: omega(:), weight(:)

! local
  integer                :: i, j
  real(q)                :: t1, t2, t3
!   real(8)                :: pi
!   pi = acos(-1.d0)

!   write(iout,'(1x,a,i4)') 'Number of Gauss-Chebyshev frequencies:',nb
  t1=pi/(4._q*(nb))
  t2=2._q*t1
  j=0
  do i=nb,1,-1
    t3=t1*(2._q*(i)-1._q)
    omega(i) = 1._q/tan(t3)
    weight(i) = t2/(sin(t3))**2
    j=j+1
  enddo

 end subroutine gauss_chebyshev

 !=====================================================================================
  subroutine gauss_chebyshev0(nb, omega, weight)
!-------------------------------------------------------------------------------------
! Description: Gauss-Chebyshev grid and weights for frequency integratiion
! as gauss_chebyshev but a point omega=0 with zero weight is included
!
! Created   : G. Jansen, J.G. Angyan, 13 Mar 2009
!-------------------------------------------------------------------------------------
  implicit none
  !include "common/tapes"

! input
  integer, intent(in)    :: nb
  INTEGER :: mb

! input/output
  real(q), intent(inout) :: omega(:), weight(:)

! local
  integer                :: i, j
  real(q)                :: t1, t2, t3
!   real(8)                :: pi
!   pi = acos(-1.d0)

!   write(iout,'(1x,a,i4)') 'Number of Gauss-Chebyshev frequencies:',nb
  omega=0._q
  weight=0._q
  mb=nb-1
  
  t1=pi/(4._q*(mb))
  t2=2._q*t1
  j=0
  do i=mb,1,-1
    t3=t1*(2._q*(i)-1._q)
    omega(i) = 1._q/tan(t3)
    weight(i) = t2/(sin(t3))**2
    j=j+1
  enddo
 end subroutine gauss_chebyshev0
 
        FUNCTION alpha_freq(alpha,omegaP,omega)
        !c compute frequency-dependent polarizability
          REAL(q)::alpha,omegaP,omega,alpha_freq        
          alpha_freq=alpha/(1.+(omega/omegaP)**2)
        END FUNCTION alpha_freq

        SUBROUTINE vdw_cfdm(GRIDC,nions,POSION,lattmat_au,lattinv,alpha, &
            &                   c6,r0,rmax,ECFDM,BETA,multi,IU0)
        !c compute manybody vdw energy using the coupled fluctuating 
        !c dipoles model
          TYPE (grid_3d) GRIDC  
          INTEGER :: nions,i,j,ii,jj,pp,qq,mm,ninfo,ppp,IU0
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),lattinv(3,3),tau(3,3)       
          REAL(q) :: alpha(nions),c6(nions),r0(nions),omega(nions)
          REAL(q),ALLOCATABLE :: Hcfdm(:,:)
          REAL(q) :: alphaP,alphaQ
          REAL(q) :: Rp,Rq,Rpq
          REAL(q) :: EUqho,ECqho,Ecfdm
          INTEGER :: translations(3)
          REAL(q) :: rmax !c cutoff radius in Angstroems
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: beta               !range separating parameter
          REAL(q),ALLOCATABLE :: d(:)         !c eigenvalues of the matrix
          REAL(q),ALLOCATABLE :: u(:,:)     !c eigenvectors of the matrix
          REAL(q),ALLOCATABLE :: vt(:,:)
          INTEGER :: ppqq
          INTEGER,ALLOCATABLE,SAVE :: indexP(:),indexQ(:)
          REAL(q) :: lattmat_au_multi(3,3),lattinv_multi(3,3)
          REAL(q),ALLOCATABLE :: POSION_multi(:,:)
          INTEGER :: multi(3),totmulti
          INTEGER :: pp_, qq_
          LOGICAL :: LFIRST=.TRUE.

          NODE_ME=1
          NCPU=1
#ifdef MPI
          NODE_ME=GRIDC%COMM%NODE_ME
          NCPU=GRIDC%COMM%NCPU
#endif
      
          totmulti=multi(1)*multi(2)*multi(3)
          !ppqq=(totmulti*nions)*(totmulti*nions-1)/2
          ppqq=(totmulti*nions)*(totmulti*nions+1)/2

          ALLOCATE(Hcfdm(3*nions*totmulti,3*nions*totmulti))
          ALLOCATE(d(3*nions*totmulti))       
          ALLOCATE(u(3*nions*totmulti,3*nions*totmulti))    
          ALLOCATE(vt(3*nions*totmulti,3*nions*totmulti))
          ALLOCATE(POSION_multi(3,nions*totmulti))

          lattmat_au_multi=0._q; POSION_multi=0._q
        
          IF (LFIRST) THEN
            ALLOCATE(indexP(ppqq))
            ALLOCATE(indexQ(ppqq))
            CALL index_table2d(nions*totmulti,ppqq,indexP,indexQ)
          ENDIF

          CALL multiply_cell(nions,POSION,lattmat_au,multi,POSION_multi,lattmat_au_multi)

!           IF (IU0>=0) THEN     
!             write(*,*) AUTOA
!             write(*,*)    lattmat_au_multi(1,1), lattmat_au_multi(1,2), lattmat_au_multi(1,3) 
!             write(*,*)    lattmat_au_multi(2,1), lattmat_au_multi(2,2), lattmat_au_multi(2,3)   
!             write(*,*)    lattmat_au_multi(3,1), lattmat_au_multi(3,2), lattmat_au_multi(3,3)  
!             write(*,*) nions*totmulti
!             write(*,*) 'directs'
!             DO i=1,    nions*totmulti
!               write(*,*) POSION_multi(1,i), POSION_multi(2,i),POSION_multi(3,i)
!             ENDDO      
!           ENDIF
       
          lattinv_multi(:,1)=lattinv(:,1)/multi(1)
          lattinv_multi(:,2)=lattinv(:,2)/multi(2)
          lattinv_multi(:,3)=lattinv(:,3)/multi(3)

          translations(1)=nint(AUTOA*rmax*SUM(lattinv_multi(:,1)**2)**0.5)
          translations(2)=nint(AUTOA*rmax*SUM(lattinv_multi(:,2)**2)**0.5)
          translations(3)=nint(AUTOA*rmax*SUM(lattinv_multi(:,3)**2)**0.5)

!           IF (IU0>=0) THEN             
!               write(*,*) 'trans_',translations(1),translations(2),translations(3)          
!           ENDIF

          Hcfdm=0._q     

          omega=4./3.*c6/alpha**2

          EUqho=1.5_q*SUM(omega)     

          DO mm=NODE_ME,ppqq,NCPU
            pp=indexP(mm)
            qq=indexQ(mm)
            pp_=MOD(pp,nions)
            IF (pp_==0) pp_=nions
            qq_=MOD(qq,nions)
            IF (qq_==0) qq_=nions
            Rpq=r0(pp_)+r0(qq_)
            
            CALL cfdmTau_pq(POSION_multi(:,pp),POSION_multi(:,qq),Rpq,beta,lattmat_au_multi,translations, &
                   & rmax,tau)
            tau=tau*omega(pp_)*omega(qq_)*(alpha(pp_)*alpha(qq_))**0.5 


            Hcfdm(3*pp-2,3*qq-2:3*qq)=tau(1,:)
            Hcfdm(3*pp-1,3*qq-2:3*qq)=tau(2,:)
            Hcfdm(3*pp  ,3*qq-2:3*qq)=tau(3,:)
            Hcfdm(3*qq-2,3*pp-2:3*pp)=tau(1,:)
            Hcfdm(3*qq-1,3*pp-2:3*pp)=tau(2,:)
            Hcfdm(3*qq  ,3*pp-2:3*pp)=tau(3,:)  
          ENDDO
#ifdef MPI
          CALL M_sum_d(GRIDC%COMM,Hcfdm ,9*nions*nions*totmulti*totmulti )
#endif
          DO pp=1,nions*totmulti
            pp_=MOD(pp,nions)
            IF (pp_==0) pp_=nions
            Hcfdm(3*pp-2,3*pp-2)=Hcfdm(3*pp-2,3*pp-2)+omega(pp_)**2
            Hcfdm(3*pp-1,3*pp-1)=Hcfdm(3*pp-1,3*pp-1)+omega(pp_)**2
            Hcfdm(3*pp,    3*pp)=Hcfdm(3*pp,    3*pp)+omega(pp_)**2
          ENDDO
 
          CALL SVDVALVEC(Hcfdm,3*nions*totmulti,d,u,vt) !c calculates evectors and evalues

!           IF (IU0>=0) THEN  
!             DO i=1,3*nions*totmulti
!               write(*,*) 'dd',d(i)
!             ENDDO
!           ENDIF

          ECqho=0.5_q*SUM(d**0.5)/totmulti
          Ecfdm=(ECqho-EUqho)*2*RYTOEV

          DEALLOCATE(POSION_multi,vt,u,d,Hcfdm)

          IF (LFIRST) LFIRST=.FALSE.
        END SUBROUTINE vdw_cfdm

!         SUBROUTINE vdw_cfdm(GRIDC,nions,POSION,lattmat_au,alpha, &
!             &                   c6,r0,rmax,translations,ECFDM,BETA,ppqq,indexP,indexQ,IU0)
!         !c compute manybody vdw energy using the coupled fluctuating 
!         !c dipoles model
!           TYPE (grid_3d) GRIDC  
!           INTEGER :: nions,i,j,ii,jj,pp,qq,mm,ninfo,ppp,IU0
!           REAL(q) :: POSION(3,nions)
!           REAL(q) :: lattmat_au(3,3),tau(3,3)       
!           REAL(q) :: alpha(nions),c6(nions),r0(nions),omega(nions)
!           REAL(q),ALLOCATABLE :: Hcfdm(:,:)
!           REAL(q) :: alphaP,alphaQ
!           REAL(q) :: Rp,Rq,Rpq
!           REAL(q) :: EUqho,ECqho,Ecfdm
!           INTEGER :: translations(3)
!           REAL(q) :: rmax !c cutoff radius in Angstroems
!           INTEGER :: NODE_ME,NCPU
!           REAL(q) :: beta               !range separating parameter
!           REAL(q) :: d(3*nions)         !c eigenvalues of the matrix
!           REAL(q) :: u(3*nions,3*nions)     !c eigenvectors of the matrix
!           REAL(q) :: vt(3*nions,3*nions)
!           INTEGER :: ppqq
!           INTEGER :: indexP(ppqq),indexQ(ppqq)
!        
!           NODE_ME=1
!           NCPU=1
! #ifdef MPI
!           NODE_ME=GRIDC%COMM%NODE_ME
!           NCPU=GRIDC%COMM%NCPU
! #endif
!       
!           ALLOCATE(Hcfdm(3*nions,3*nions))
! 
!           Hcfdm=0._q     
! 
!           omega=4./3.*c6/alpha**2
!           EUqho=1.5_q*SUM(omega)          
! 
!           DO mm=NODE_ME,ppqq,NCPU
!             pp=indexP(mm)
!             qq=indexQ(mm)
! 
!             Rpq=r0(pp)+r0(qq)
!             CALL cfdmTau_pq(POSION(:,pp),POSION(:,qq),Rpq,beta,lattmat_au,translations, &
!                    & rmax,tau)
!  
!             tau=tau*omega(pp)*omega(qq)*(alpha(pp)*alpha(qq))**0.5   
!             Hcfdm(3*pp-2,3*qq-2:3*qq)=tau(1,:)
!             Hcfdm(3*pp-1,3*qq-2:3*qq)=tau(2,:)
!             Hcfdm(3*pp  ,3*qq-2:3*qq)=tau(3,:)
!             Hcfdm(3*qq-2,3*pp-2:3*pp)=tau(1,:)
!             Hcfdm(3*qq-1,3*pp-2:3*pp)=tau(2,:)
!             Hcfdm(3*qq  ,3*pp-2:3*pp)=tau(3,:)  
!           ENDDO
!    
!           CALL M_sum_d(GRIDC%COMM,Hcfdm ,9*nions*nions )
! 
!           DO pp=1,nions
!             Hcfdm(3*pp-2,3*pp-2)=Hcfdm(3*pp-2,3*pp-2)+omega(pp)**2
!             Hcfdm(3*pp-1,3*pp-1)=Hcfdm(3*pp-1,3*pp-1)+omega(pp)**2
!             Hcfdm(3*pp,    3*pp)=Hcfdm(3*pp,    3*pp)+omega(pp)**2
!           ENDDO
! 
!           CALL SVDVALVEC(Hcfdm,3*nions,d,u,vt) !c calculates evectors and evalues
!           IF (IU0>=0) THEN
!             DO i=1,3*nions
!               write(*,*) d(i)**0.5,omega((i-1)/3+1)
!             ENDDO
!           ENDIF
!           ECqho=0.5_q*SUM(d**0.5)
!           Ecfdm=(ECqho-EUqho)*2*RYTOEV
! 
!         
!           DEALLOCATE(Hcfdm)
!         END SUBROUTINE vdw_cfdm


        SUBROUTINE multiply_cell(nions,x,lattmat,multi,x_m,lattmat_m)
          INTEGER :: nions
          REAL(q) :: x(3,nions)
          INTEGER :: multi(3)
          REAL(q) :: lattmat(3,3),lattmat_m(3,3)
          REAL(q) :: x_m(3,nions*multi(1)*multi(2)*multi(3))
          INTEGER :: m1block,m2block,m3block,m4block
          REAL(q), ALLOCATABLE :: x1(:,:),x2(:,:)
          INTEGER :: m1,m2,m3,nindx1,nindx2
          REAL(q) :: mm

          m1block=nions
          m2block=m1block*multi(1)
          m3block=m2block*multi(2)
          m4block=m3block*multi(3)

          ALLOCATE(x1(3,m2block),x2(3,m3block))
          x1=0._q; x2=0._q; x_m=0._q
       

          lattmat_m(1,:)=lattmat(1,:)*multi(1)
          lattmat_m(2,:)=lattmat(2,:)*multi(2)
          lattmat_m(3,:)=lattmat(3,:)*multi(3)

          DO m1=1,multi(1)  
            nindx2= m1block*m1
            nindx1=  nindx2-m1block+1
            mm=1._q*(m1-1)
            x1(1,nindx1:nindx2)=x(1,1:m1block)+mm
            x1(2,nindx1:nindx2)=x(2,1:m1block)
            x1(3,nindx1:nindx2)=x(3,1:m1block)
          ENDDO

          x1(1,:)=x1(1,:)/multi(1)
 
          DO m2=1,multi(2)  
            nindx2= m2block*m2
            nindx1=  nindx2-m2block+1
            mm=1._q*(m2-1)
            x2(1,nindx1:nindx2)=x1(1,1:m2block)
            x2(2,nindx1:nindx2)=x1(2,1:m2block)+mm
            x2(3,nindx1:nindx2)=x1(3,1:m2block)
          ENDDO

          x2(2,:)=x2(2,:)/multi(2)

          DO m3=1,multi(3)  
            nindx2= m3block*m3
            nindx1=  nindx2-m3block+1
            mm=1._q*(m3-1)
            x_m(1,nindx1:nindx2)=x2(1,1:m3block)
            x_m(2,nindx1:nindx2)=x2(2,1:m3block)
            x_m(3,nindx1:nindx2)=x2(3,1:m3block)+mm
          ENDDO

          x_m(3,:)=x_m(3,:)/multi(3)

          DEALLOCATE(x1,x2)

       END SUBROUTINE multiply_cell

       SUBROUTINE vdw_cfdm2(GRIDC,nions,POSION,lattmat_au,alpha, &
            &                   c6,rmax,translations,ECFDM,BETA,IU0)
        !c compute manybody vdw energy using the coupled fluctuating 
        !c dipoles model
          TYPE (grid_3d) GRIDC  
          INTEGER :: nions,i,j,ii,jj,pp,qq,ninfo,ppp,IU0
          REAL(q) :: POSION(3,nions)
          REAL(q) :: lattmat_au(3,3),tau(3,3)       
          REAL(q) :: alpha(nions),c6(nions),omega(nions)
          REAL(q),ALLOCATABLE :: Hcfdm(:,:)
          REAL(q) :: alphaP,alphaQ
          REAL(q) :: Rp,Rq,Rpq
          REAL(q) :: EUqho,ECqho,Ecfdm
          INTEGER :: translations(3)
          REAL(q) :: rmax !c cutoff radius in Angstroems
          INTEGER :: NODE_ME,NCPU
          REAL(q) :: beta               !range separating parameter
          REAL(q) :: d(3*nions)         !c eigenvalues of the matrix
          REAL(q) :: u(3*nions,3*nions)     !c eigenvectors of the matrix
          REAL(q) :: vt(3*nions,3*nions)

          REAL(q) :: dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)

        
          NODE_ME=1
          NCPU=1
#ifdef MPI
          NODE_ME=GRIDC%COMM%NODE_ME
          NCPU=GRIDC%COMM%NCPU
#endif
      
          ALLOCATE(Hcfdm(3*nions,3*nions))
          Hcfdm=0._q     
!           DO i=1,nions
!             omega(i)=4./3.*c6(i)/alpha(i)**2
!           ENDDO 
          omega=4./3.*c6/alpha**2
          EUqho=1.5_q*SUM(omega)          

          DO pp=1,nions
            !Rpq=2*r0(pp) 
            !CALL cfdmTau_pq(POSION(:,pp),POSION(:,pp),Rpq,beta,lattmat_au,translations, &
            !         & rmax,tau,NODE_ME,NCPU)
            Rp=((2./PI)**0.5*alpha(pp)/3.)**(1./3.)
            Rpq=beta*(Rp**2+Rp**2)**0.5
            IF (IU0>=0) THEN
              write(*,*) 'Rpq',pp,pp,Rp,Rpq
            ENDIF
            CALL scsTau_pq(POSION(:,pp),POSION(:,pp),Rpq,lattmat_au,translations, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,.FALSE.)

!             IF (IU0>=0) THEN
!               write(*,*) 'tauPP1',tau(1,1),tau(1,2),tau(1,3)
!               write(*,*) 'tauPP2',tau(2,1),tau(2,2),tau(2,3)
!               write(*,*) 'tauPP3',tau(3,1),tau(3,2),tau(3,3)
!             ENDIF

! #ifdef MPI
!             CALL M_sum_d(GRIDC%COMM, tau, 9 )          
! #endif
            tau=tau*omega(pp)**2*alpha(pp)

            Hcfdm(3*pp-2,3*pp-2:3*pp)=tau(1,:)
            Hcfdm(3*pp-1,3*pp-2:3*pp)=tau(2,:)
            Hcfdm(3*pp  ,3*pp-2:3*pp)=tau(3,:)
            Hcfdm(3*pp-2,3*pp-2)=Hcfdm(3*pp-2,3*pp-2)+omega(pp)**2
            Hcfdm(3*pp-1,3*pp-1)=Hcfdm(3*pp-1,3*pp-1)+omega(pp)**2
            Hcfdm(3*pp,    3*pp)=Hcfdm(3*pp,    3*pp)+omega(pp)**2

            DO qq=pp+1,nions
!               Rpq=r0(pp)+r0(qq)
!               CALL cfdmTau_pq(POSION(:,pp),POSION(:,qq),Rpq,beta,lattmat_au,translations, &
!                    & rmax,tau,NODE_ME,NCPU)


              Rq=((2./PI)**0.5*alpha(qq)/3.)**(1./3.)
              Rpq=beta*(Rp**2+Rq**2)**0.5
              IF (IU0>=0) THEN
                write(*,*) 'Rpq',pp,qq,Rp,Rpq
              ENDIF
              CALL scsTau_pq(POSION(:,pp),POSION(:,qq),Rpq,lattmat_au,translations, &
                       & rmax/AUTOA,tau,dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
                       & dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,.FALSE.)

! #ifdef MPI
!               CALL M_sum_d(GRIDC%COMM,tau,9 )
! #endif
              IF (IU0>=0) THEN
                write(*,*) 'tauPQ1',tau(1,1),tau(1,2),tau(1,3)
                write(*,*) 'tauPQ2',tau(2,1),tau(2,2),tau(2,3)
                write(*,*) 'tauPQ3',tau(3,1),tau(3,2),tau(3,3)
              ENDIF
              tau=tau*omega(pp)*omega(qq)*(alpha(pp)*alpha(qq))**0.5   
              Hcfdm(3*pp-2,3*qq-2:3*qq)=tau(1,:)
              Hcfdm(3*pp-1,3*qq-2:3*qq)=tau(2,:)
              Hcfdm(3*pp  ,3*qq-2:3*qq)=tau(3,:)
              Hcfdm(3*qq-2,3*pp-2:3*pp)=tau(1,:)
              Hcfdm(3*qq-1,3*pp-2:3*pp)=tau(2,:)
              Hcfdm(3*qq  ,3*pp-2:3*pp)=tau(3,:)  
            ENDDO
          ENDDO
   
          CALL SVDVALVEC(Hcfdm,3*nions,d,u,vt) !c calculates evectors and evalues
          IF (IU0>=0) THEN
            DO i=1,3*nions
              write(*,*) d(i)**0.5,omega((i-1)/3+1)
            ENDDO
          ENDIF
          ECqho=0.5_q*SUM(d**0.5)
          Ecfdm=(ECqho-EUqho)*2*RYTOEV

          DEALLOCATE(Hcfdm)
        END SUBROUTINE vdw_cfdm2

        SUBROUTINE cfdmTau_pq(x1,x2,Rpq,beta,lattmat,translations,rmax,tau)
        !c dipole interaction tensor for coupled fluctuating dipoles model 
          INTEGER :: translations(3)
          INTEGER :: t1,t2,t3,i,j
          REAL(q) :: Rpq,rmax,r_norm,rrpq,A,part1,part2
          REAL(q) :: tau(3,3),lattmat(3,3)
          REAL(q) :: beta
          REAL(q) :: sigma
          REAL(q) :: x1(3),x2(3),dr(3),dr_(3),r(3)
          REAL :: r_test(3),r_testN
         
          tau=0._q
          dr_=x1-x2
 
          !c use minimum image convention
          CALL min_imageV(3,dr_)

          DO t1=-translations(1) ,translations(1)
          !DO t1=NODE_ME-1-translations(1),translations(1),NCPU
            dr=0._q
            dr(1)=dr_(1)+t1*1._q
            DO t2=-translations(2) ,translations(2)
              dr(2)=dr_(2)+t2*1._q
              DO t3=-translations(3) ,translations(3)
                !r_test=t1*lattmat(1,:)+t2*lattmat(2,:)+t3*lattmat(3,:)
                !r_testN=SUM(r_test**2)**0.5
                !if (r_testN .LE. rmax) THEN
                  dr(3)=dr_(3)+t3*1._q
                  r=MATMUL(dr,lattmat)
                  r_norm=SUM(r**2)**0.5
                  IF ((r_norm .GT. 1.e-1) .AND. (r_norm .LE. rmax)) THEN
                    sigma=(r_norm/Rpq)**beta
                    part1=3.-(3.+4.*beta*sigma-beta**2.*sigma+beta**2.*sigma**2.)*EXP(-sigma)
                    part1=part1/r_norm**5 
                    part2=(1.+beta*sigma)*EXP(-sigma)-1.
                    part2=part2/r_norm**3

                    DO i=1,3
                      tau(i,i)=tau(i,i)+part2
                      DO j=1,3
                        tau(i,j)=tau(i,j)+part1*r(i)*r(j)
                      ENDDO
                    ENDDO
                  ENDIF
                !endif
              ENDDO
            ENDDO
          ENDDO
          tau=-tau
        END SUBROUTINE cfdmTau_pq

        SUBROUTINE scsTau_pq(x1,x2,Rpq,lattmat,translations,rmax,tau,&
        &   dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
        &   dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
        !c dipole interaction tensor for TS-SCS
          INTEGER :: translations(3)
          INTEGER :: t1,t2,t3,i,j
          REAL(q) :: Rpq,rmax,r_norm,rrpq,A,part1,part2
          REAL(q) :: tau(3,3),lattmat(3,3)
          REAL(q) :: x1(3),x2(3),dr(3),dr_(3),r(3)
          REAL(q) :: dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: dt_x(3,3),dt_y(3,3),dt_z(3,3)
          LOGICAL :: LSCSGRAD
          !INTEGER :: NODE_ME,NCPU
          REAL :: r_test(3),r_testN
         
          tau=0._q
          dtaux=0._q;dtauy=0._q;dtauz=0._q
          dtau_h0x=0._q;dtau_h0y=0._q;dtau_h0z=0._q
          dtau_h1x=0._q;dtau_h1y=0._q;dtau_h1z=0._q
          dtau_h2x=0._q;dtau_h2y=0._q;dtau_h2z=0._q
          dr_=x1-x2

!           translations(1)=1;translations(2)=1;translations(3)=1

!           write(*,*) 'lattmat:'
!           write(*,*) lattmat(1,1),lattmat(1,2),lattmat(1,3)
!           write(*,*) lattmat(2,1),lattmat(2,2),lattmat(2,3)
!           write(*,*) lattmat(3,1),lattmat(3,2),lattmat(3,3)
          !c use minimum image convention
          CALL min_imageV(3,dr_)
          DO t1=-translations(1) ,translations(1)
          !DO t1=NODE_ME-1-translations(1),translations(1),NCPU
            dr=0._q
            dr(1)=dr_(1)+t1*1._q
            DO t2=-translations(2) ,translations(2)
              dr(2)=dr_(2)+t2*1._q
              DO t3=-translations(3) ,translations(3)
                r_test=t1*lattmat(1,:)+t2*lattmat(2,:)+t3*lattmat(3,:)
                r_testN=SUM(r_test**2)**0.5
                if (r_testN .LE. rmax) THEN

                dr(3)=dr_(3)+t3*1._q
                r=MATMUL(dr,lattmat)
                r_norm=SUM(r**2)**0.5
               ! IF ((r_norm .GT. 1.e-1) .AND. (r_norm .LE. rmax)) THEN
                IF ((r_norm .GT. 1.e-1) ) THEN
                  rrpq=r_norm/Rpq
                  A=(errf(rrpq)-(2./SQRT(PI))*(rrpq)*EXP(-rrpq**2))/r_norm**5
                  !A=(errf(rrpq)-(2./PI**0.5)*(rrpq)*EXP(-rrpq**2))/r_norm**5
                  DO i=1,3
                    DO j=1,3
                      IF (i==j) THEN
                        part1=-(3*r(i)**2-r_norm**2)*A
                      ELSE
                        part1=-(3*r(i)*r(j))*A
                      ENDIF
                      part2=(4./SQRT(PI))*(r(i)*r(j))/(r_norm**2*Rpq**3)*EXP(-rrpq**2)
                      tau(i,j)=tau(i,j)+(part1+part2)
                    ENDDO
                  ENDDO
                  IF (LSCSGRAD) THEN
!                     write(*,*) 'x00',r_norm,rrpq,Rpq
                    dt_x(1,1)=dTau_ii_di(r(1),r_norm,rrpq,Rpq)
                    dt_x(1,2)=dTau_ij_di(r(1),r(2),r_norm,rrpq,Rpq)
                    dt_x(1,3)=dTau_ij_di(r(1),r(3),r_norm,rrpq,Rpq)
                    dt_x(2,2)=dTau_jj_di(r(1),r(2),r_norm,rrpq,Rpq)
                    dt_x(3,3)=dTau_jj_di(r(1),r(3),r_norm,rrpq,Rpq)
                    dt_x(2,3)=dTau_jk_di(r(1),r(2),r(3),r_norm,rrpq,Rpq)

                    dt_y(2,2)=dTau_ii_di(r(2),r_norm,rrpq,Rpq)
                    dt_y(2,1)=dTau_ij_di(r(2),r(1),r_norm,rrpq,Rpq)
                    dt_y(2,3)=dTau_ij_di(r(2),r(3),r_norm,rrpq,Rpq)
                    dt_y(1,1)=dTau_jj_di(r(2),r(1),r_norm,rrpq,Rpq)
                    dt_y(3,3)=dTau_jj_di(r(2),r(3),r_norm,rrpq,Rpq)
                    dt_y(1,3)=dTau_jk_di(r(2),r(3),r(1),r_norm,rrpq,Rpq)
          
                    dt_z(3,3)=dTau_ii_di(r(3),r_norm,rrpq,Rpq)
                    dt_z(3,1)=dTau_ij_di(r(3),r(1),r_norm,rrpq,Rpq)
                    dt_z(3,2)=dTau_ij_di(r(3),r(2),r_norm,rrpq,Rpq)
                    dt_z(2,2)=dTau_jj_di(r(3),r(2),r_norm,rrpq,Rpq)
                    dt_z(1,1)=dTau_jj_di(r(3),r(1),r_norm,rrpq,Rpq)
                    dt_z(1,2)=dTau_jk_di(r(3),r(1),r(2),r_norm,rrpq,Rpq) 

                    dtaux=dtaux+dt_x                  
                    dtauy=dtauy+dt_y
                    dtauz=dtauz+dt_z

                    dtau_h0x=dtau_h0x+dt_x*dr(1)
!                     write(*,*) "dtau_h0x(1,1)",dtau_h0x(1,1),dt_x(1,1)
                    dtau_h1x=dtau_h1x+dt_x*dr(2)
                    dtau_h2x=dtau_h2x+dt_x*dr(3)
                    dtau_h0y=dtau_h0y+dt_y*dr(1)
                    dtau_h1y=dtau_h1y+dt_y*dr(2)
                    dtau_h2y=dtau_h2y+dt_y*dr(3)
                    dtau_h0z=dtau_h0z+dt_z*dr(1)
                    dtau_h1z=dtau_h1z+dt_z*dr(2)
                    dtau_h2z=dtau_h2z+dt_z*dr(3)
                  ENDIF
                ENDIF
              endif
              ENDDO
            ENDDO
          ENDDO
!           STOP
          IF (LSCSGRAD) THEN
            dtaux(2,1)=dtaux(1,2)
            dtaux(3,1)=dtaux(1,3)
            dtaux(3,2)=dtaux(2,3)
            dtauy(1,2)=dtauy(2,1)
            dtauy(3,2)=dtauy(2,3)
            dtauy(3,1)=dtauy(1,3)
            dtauz(1,3)=dtauz(3,1)
            dtauz(2,3)=dtauz(3,2)
            dtauz(2,1)=dtauz(1,2)

            dtau_h0x(2,1)=dtau_h0x(1,2)
            dtau_h0x(3,1)=dtau_h0x(1,3)
            dtau_h0x(3,2)=dtau_h0x(2,3)
            dtau_h1x(2,1)=dtau_h1x(1,2)
            dtau_h1x(3,1)=dtau_h1x(1,3)
            dtau_h1x(3,2)=dtau_h1x(2,3)
            dtau_h2x(2,1)=dtau_h2x(1,2)
            dtau_h2x(3,1)=dtau_h2x(1,3)
            dtau_h2x(3,2)=dtau_h2x(2,3)

            dtau_h0y(1,2)=dtau_h0y(2,1)
            dtau_h0y(3,2)=dtau_h0y(2,3)
            dtau_h0y(3,1)=dtau_h0y(1,3)
            dtau_h1y(1,2)=dtau_h1y(2,1)
            dtau_h1y(3,2)=dtau_h1y(2,3)
            dtau_h1y(3,1)=dtau_h1y(1,3)
            dtau_h2y(1,2)=dtau_h2y(2,1)
            dtau_h2y(3,2)=dtau_h2y(2,3)
            dtau_h2y(3,1)=dtau_h2y(1,3)

            dtau_h0z(1,3)=dtau_h0z(3,1)
            dtau_h0z(2,3)=dtau_h0z(3,2)
            dtau_h0z(2,1)=dtau_h0z(1,2)
            dtau_h1z(1,3)=dtau_h1z(3,1)
            dtau_h1z(2,3)=dtau_h1z(3,2)
            dtau_h1z(2,1)=dtau_h1z(1,2)
            dtau_h2z(1,3)=dtau_h2z(3,1)
            dtau_h2z(2,3)=dtau_h2z(3,2)
            dtau_h2z(2,1)=dtau_h2z(1,2)
          ENDIF
        END SUBROUTINE scsTau_pq
       
        SUBROUTINE Tau_pq_sr(x1,x2,Rpq,r0,dampD,sR,lattmat,translations,rmax,tau,&
        &   dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
        &   dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
        !c dipole interaction tensor for TS-SCS
          INTEGER :: translations(3)
          INTEGER :: t1,t2,t3,i,j
          REAL(q) :: Rpq,Rpq2,rmax,r_norm,r_norm2,rrpq,A,part1,part2
          REAL(q) :: tau(3,3),lattmat(3,3),tau_(3,3)
          REAL(q) :: x1(3),x2(3),dr(3),dr_(3),r(3)
          REAL(q) :: dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: dt_x(3,3),dt_y(3,3),dt_z(3,3)
          LOGICAL :: LSCSGRAD
          !REAL :: r_test(3),r_testN
          REAL :: r_0(3),r_trans(3),r_transN,r_transA(3),r_transB(3)
          REAL(q), INTENT(in) :: dampD, sR,r0
          REAL(q) :: fdamp,dfdamp,expfact,errffact
          REAL(q),PARAMETER :: two_over_sqrtPI=2._q/SQRT(PI),four_over_sqrtPI=2*two_over_sqrtPI
       
          tau=0._q;tau_=0._q
          dtaux=0._q;dtauy=0._q;dtauz=0._q
          dtau_h0x=0._q;dtau_h0y=0._q;dtau_h0z=0._q
          dtau_h1x=0._q;dtau_h1y=0._q;dtau_h1z=0._q
          dtau_h2x=0._q;dtau_h2y=0._q;dtau_h2z=0._q
          dr_=x1-x2
       
         !c use minimum image convention
         CALL min_imageV(3,dr_)
         r_0=dr_(1)*lattmat(:,1)+dr_(2)*lattmat(:,2)+dr_(3)*lattmat(:,3)
          DO t1=-translations(1) ,translations(1)
            dr=0._q
            dr(1)=dr_(1)+t1*1._q
            r_transA=t1*lattmat(:,1)
            !r1=dr(1)*lattmat(1,:)
            DO t2=-translations(2) ,translations(2)
              dr(2)=dr_(2)+t2*1._q
              r_transB=r_transA+t2*lattmat(:,2)
              !r2=dr(2)*lattmat(2,:)
              DO t3=-translations(3) ,translations(3)
                !r_trans=t1*lattmat(:,1)+t2*lattmat(:,2)+t3*lattmat(:,3)
                r_trans=r_transB+t3*lattmat(:,3)
                r_transN=SUM(r_trans* r_trans)**0.5
                IF (r_transN .LE. rmax) THEN
                  dr(3)=dr_(3)+t3*1._q
                  r=r_0+r_trans
                  r_norm2=r(1)*r(1)+r(2)*r(2)+r(3)*r(3)
                  r_norm=r_norm2**0.5
                  ! IF ((r_norm .GT. 1.e-1) .AND. (r_norm .LE. rmax)) THEN
                  
                  IF ((r_norm .GT. 1.e-1) ) THEN
                    fdamp=damping_fermi(dampD,sR,r0,r_norm)
                    rrpq=r_norm/Rpq
                    expfact=EXP(-rrpq*rrpq)
                    errffact=errf(rrpq)
                    !A=(errf(rrpq)-(2./SQRT(PI))*(rrpq)*EXP(-rrpq**2))/r_norm**5
                    !A=(errf(rrpq)-two_over_sqrtPI*(rrpq)*EXP(-rrpq**2))/(r_norm2*r_norm2*r_norm) !  r_norm**5
                    A=(errffact-two_over_sqrtPI*(rrpq)*expfact)/(r_norm2*r_norm2*r_norm)
                    tau_=0._q
                    DO i=1,3
                      part2=four_over_sqrtPI*(r(i)*r(i))/(r_norm2*Rpq*Rpq*Rpq)*expfact !EXP(-rrpq**2)
                      part1=-(3*r(i)*r(i)-r_norm2)*A
                      !tau_(i,i)=(part1+part2)*(1._q-fdamp)
                      tau_(i,i)=(part1+part2)
                      DO j=i+1,3
                        part2=four_over_sqrtPI*(r(i)*r(j))/(r_norm2*Rpq*Rpq*Rpq)*expfact !EXP(-rrpq**2)                        
                        part1=-(3*r(i)*r(j))*A
                        !tau_(i,j)=(part1+part2)*(1._q-fdamp)
                        tau_(i,j)=(part1+part2)
                        tau_(j,i)=tau_(i,j)                        
                      ENDDO
                    ENDDO
  
                    IF (LSCSGRAD) THEN
                      dfdamp=damping_fermi_deriv(dampD,sR,r0,r_norm)
                      Rpq2=Rpq*Rpq
                      
                      dt_x(1,1)=dTau_ii_di2(r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_y(2,2)=dTau_ii_di2(r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_z(3,3)=dTau_ii_di2(r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_x(2,2)=dTau_jj_di2(r(1),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_x(3,3)=dTau_jj_di2(r(1),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_y(1,1)=dTau_jj_di2(r(2),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_y(3,3)=dTau_jj_di2(r(2),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_z(2,2)=dTau_jj_di2(r(3),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_z(1,1)=dTau_jj_di2(r(3),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_x(2,3)=dTau_jk_di2(r(1),r(2),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
                      dt_x(1,2)=dt_y(1,1)
                      dt_x(1,3)=dt_z(1,1)
                      dt_y(2,1)=dt_x(2,2)
                      dt_y(2,3)=dt_z(2,2)
                      dt_z(3,1)=dt_x(3,3)
                      dt_z(3,2)=dt_y(3,3)
                      dt_y(1,3)=dt_x(2,3)
                      dt_z(1,2)=dt_x(2,3)
                      
!                       dt_x(1,1)=dTau_ii_di2(r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_x(1,2)=dTau_ij_di2(r(1),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_x(1,3)=dTau_ij_di2(r(1),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_x(2,2)=dTau_jj_di2(r(1),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_x(3,3)=dTau_jj_di2(r(1),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_x(2,3)=dTau_jk_di2(r(1),r(2),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
! 
!                       dt_y(2,2)=dTau_ii_di2(r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_y(2,1)=dTau_ij_di2(r(2),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_y(2,3)=dTau_ij_di2(r(2),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_y(1,1)=dTau_jj_di2(r(2),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_y(3,3)=dTau_jj_di2(r(2),r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_y(1,3)=dTau_jk_di2(r(2),r(3),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!           
!                       dt_z(3,3)=dTau_ii_di2(r(3),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_z(3,1)=dTau_ij_di2(r(3),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_z(3,2)=dTau_ij_di2(r(3),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_z(2,2)=dTau_jj_di2(r(3),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_z(1,1)=dTau_jj_di2(r(3),r(1),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact)
!                       dt_z(1,2)=dTau_jk_di2(r(3),r(1),r(2),r_norm,r_norm2,Rpq,Rpq2,expfact,errffact) 

                      dt_x=dt_x*(1._q-fdamp)-tau_*dfdamp*r(1)/r_norm 
                      dt_y=dt_y*(1._q-fdamp)-tau_*dfdamp*r(2)/r_norm
                      dt_z=dt_z*(1._q-fdamp)-tau_*dfdamp*r(3)/r_norm

!                       dt_x=dt_x*(1._q-fdamp)+tau_*dfdamp*r(1)/r_norm 
!                       dt_y=dt_y*(1._q-fdamp)+tau_*dfdamp*r(2)/r_norm
!                       dt_z=dt_z*(1._q-fdamp)+tau_*dfdamp*r(3)/r_norm
                                  
                      dtaux=dtaux+dt_x                 
                      dtauy=dtauy+dt_y
                      dtauz=dtauz+dt_z

                      dtau_h0x=dtau_h0x+dt_x*dr(1)
                      dtau_h1x=dtau_h1x+dt_x*dr(2)
                      dtau_h2x=dtau_h2x+dt_x*dr(3)
                      dtau_h0y=dtau_h0y+dt_y*dr(1)
                      dtau_h1y=dtau_h1y+dt_y*dr(2)
                      dtau_h2y=dtau_h2y+dt_y*dr(3)
                      dtau_h0z=dtau_h0z+dt_z*dr(1)
                      dtau_h1z=dtau_h1z+dt_z*dr(2)
                      dtau_h2z=dtau_h2z+dt_z*dr(3)
                    ENDIF

                    tau=tau+tau_*(1._q-fdamp)

                  ENDIF
                ENDIF
              ENDDO
            ENDDO
          ENDDO
!           STOP
          IF (LSCSGRAD) THEN
            dtaux(2,1)=dtaux(1,2)
            dtaux(3,1)=dtaux(1,3)
            dtaux(3,2)=dtaux(2,3)
            dtauy(1,2)=dtauy(2,1)
            dtauy(3,2)=dtauy(2,3)
            dtauy(3,1)=dtauy(1,3)
            dtauz(1,3)=dtauz(3,1)
            dtauz(2,3)=dtauz(3,2)
            dtauz(2,1)=dtauz(1,2)

            dtau_h0x(2,1)=dtau_h0x(1,2)
            dtau_h0x(3,1)=dtau_h0x(1,3)
            dtau_h0x(3,2)=dtau_h0x(2,3)
            dtau_h1x(2,1)=dtau_h1x(1,2)
            dtau_h1x(3,1)=dtau_h1x(1,3)
            dtau_h1x(3,2)=dtau_h1x(2,3)
            dtau_h2x(2,1)=dtau_h2x(1,2)
            dtau_h2x(3,1)=dtau_h2x(1,3)
            dtau_h2x(3,2)=dtau_h2x(2,3)

            dtau_h0y(1,2)=dtau_h0y(2,1)
            dtau_h0y(3,2)=dtau_h0y(2,3)
            dtau_h0y(3,1)=dtau_h0y(1,3)
            dtau_h1y(1,2)=dtau_h1y(2,1)
            dtau_h1y(3,2)=dtau_h1y(2,3)
            dtau_h1y(3,1)=dtau_h1y(1,3)
            dtau_h2y(1,2)=dtau_h2y(2,1)
            dtau_h2y(3,2)=dtau_h2y(2,3)
            dtau_h2y(3,1)=dtau_h2y(1,3)

            dtau_h0z(1,3)=dtau_h0z(3,1)
            dtau_h0z(2,3)=dtau_h0z(3,2)
            dtau_h0z(2,1)=dtau_h0z(1,2)
            dtau_h1z(1,3)=dtau_h1z(3,1)
            dtau_h1z(2,3)=dtau_h1z(3,2)
            dtau_h1z(2,1)=dtau_h1z(1,2)
            dtau_h2z(1,3)=dtau_h2z(3,1)
            dtau_h2z(2,3)=dtau_h2z(3,2)
            dtau_h2z(2,1)=dtau_h2z(1,2)
          ENDIF
        END SUBROUTINE Tau_pq_sr
      
        SUBROUTINE Tau_pq_lr(x1,x2,r0,dampD,sR,lattmat,translations,rmax,tau,&
        &   dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
        &   dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,LSCSGRAD)
        !c dipole interaction tensor for TS-SCS
          INTEGER :: translations(3)
          INTEGER :: t1,t2,t3,i,j
          REAL(q) :: rmax,r_norm,part1
          REAL(q) :: tau(3,3),lattmat(3,3),tau_(3,3)
          REAL(q) :: x1(3),x2(3),dr(3),dr_(3),r(3)
          REAL(q) :: dtaux(3,3),dtauy(3,3),dtauz(3,3)
          REAL(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          REAL(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          REAL(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          REAL(q) :: dt_x(3,3),dt_y(3,3),dt_z(3,3)
          LOGICAL :: LSCSGRAD
          REAL :: r_test(3),r_testN
          REAL(q), INTENT(in) :: dampD, sR,r0
          REAL(q) :: fdamp,dfdamp
          
         
          tau=0._q;tau_=0._q
          dtaux=0._q;dtauy=0._q;dtauz=0._q
          dtau_h0x=0._q;dtau_h0y=0._q;dtau_h0z=0._q
          dtau_h1x=0._q;dtau_h1y=0._q;dtau_h1z=0._q
          dtau_h2x=0._q;dtau_h2y=0._q;dtau_h2z=0._q
          dr_=x1-x2

          !c use minimum image convention
          !!CALL min_imageV(3,dr_)
          DO t1=-translations(1) ,translations(1)
            dr=0._q
            dr(1)=dr_(1)+t1*1._q
            DO t2=-translations(2) ,translations(2)
              dr(2)=dr_(2)+t2*1._q
              DO t3=-translations(3) ,translations(3)
                r_test=t1*lattmat(1,:)+t2*lattmat(2,:)+t3*lattmat(3,:)
                r_testN=SUM(r_test* r_test)**0.5
                IF (r_testN .LE. rmax) THEN

                  dr(3)=dr_(3)+t3*1._q
                  r=MATMUL(dr,lattmat)
                  r_norm=SUM(r*r)**0.5
                  ! IF ((r_norm .GT. 1.e-1) .AND. (r_norm .LE. rmax)) THEN
                  IF ((r_norm .GT. 1.e-1) ) THEN
                    fdamp=damping_fermi(dampD,sR,r0,r_norm)
                               
                    tau_=0._q    
                    DO i=1,3
                      part1=-(3*r(i)*r(i)-r_norm*r_norm)
                      !tau_(i,i)=(part1)*fdamp/r_norm**5
                      tau_(i,i)=(part1)/r_norm**5
                      DO j=i+1,3
                        part1=-(3*r(i)*r(j))
                        !tau_(i,j)=(part1)*fdamp/r_norm**5
                        tau_(i,j)=(part1)/r_norm**5
                        tau_(j,i)=tau_(i,j)
                      ENDDO
                    ENDDO
                    !tau_(2,1)=tau_(1,2)
                    !tau_(3,1)=tau_(1,3)
                    !tau_(3,2)=tau_(2,3)
!                     tau=tau+tau_

                    IF (LSCSGRAD) THEN
                      dfdamp=damping_fermi_deriv(dampD,sR,r0,r_norm)
                      
                      dt_x(1,1)=dTau_ii_0(r(1),r_norm)
                      dt_x(1,2)=dTau_ij_0(r(1),r(2),r_norm)
                      dt_x(1,3)=dTau_ij_0(r(1),r(3),r_norm)
                      dt_x(2,2)=dTau_jj_0(r(1),r(2),r_norm)
                      dt_x(3,3)=dTau_jj_0(r(1),r(3),r_norm)
                      dt_x(2,3)=dTau_jk_0(r(1),r(2),r(3),r_norm)

                      dt_y(2,2)=dTau_ii_0(r(2),r_norm)
                      dt_y(2,1)=dTau_ij_0(r(2),r(1),r_norm)
                      dt_y(2,3)=dTau_ij_0(r(2),r(3),r_norm)
                      dt_y(1,1)=dTau_jj_0(r(2),r(1),r_norm)
                      dt_y(3,3)=dTau_jj_0(r(2),r(3),r_norm)
                      dt_y(1,3)=dTau_jk_0(r(2),r(3),r(1),r_norm)
          
                      dt_z(3,3)=dTau_ii_0(r(3),r_norm)
                      dt_z(3,1)=dTau_ij_0(r(3),r(1),r_norm)
                      dt_z(3,2)=dTau_ij_0(r(3),r(2),r_norm)
                      dt_z(2,2)=dTau_jj_0(r(3),r(2),r_norm)
                      dt_z(1,1)=dTau_jj_0(r(3),r(1),r_norm)
                      dt_z(1,2)=dTau_jk_0(r(3),r(1),r(2),r_norm) 

                      !dt_x=0.;dt_y=0.;dt_z=0.

                      dt_x=dt_x*(fdamp)+tau_*dfdamp*r(1)/r_norm 
                      dt_y=dt_y*(fdamp)+tau_*dfdamp*r(2)/r_norm
                      dt_z=dt_z*(fdamp)+tau_*dfdamp*r(3)/r_norm

!                       dt_x=dt_x*(fdamp) -tau_*dfdamp*r(1)/r_norm 
!                       dt_y=dt_y*(fdamp) -tau_*dfdamp*r(2)/r_norm
!                       dt_z=dt_z*(fdamp) -tau_*dfdamp*r(3)/r_norm
!                                   
                      dtaux=dtaux+dt_x                 
                      dtauy=dtauy+dt_y
                      dtauz=dtauz+dt_z

                      dtau_h0x=dtau_h0x+dt_x*dr(1)
                      dtau_h1x=dtau_h1x+dt_x*dr(2)
                      dtau_h2x=dtau_h2x+dt_x*dr(3)
                      dtau_h0y=dtau_h0y+dt_y*dr(1)
                      dtau_h1y=dtau_h1y+dt_y*dr(2)
                      dtau_h2y=dtau_h2y+dt_y*dr(3)
                      dtau_h0z=dtau_h0z+dt_z*dr(1)
                      dtau_h1z=dtau_h1z+dt_z*dr(2)
                      dtau_h2z=dtau_h2z+dt_z*dr(3)
                    ENDIF
        
                    tau=tau+tau_*fdamp

                  ENDIF
                ENDIF
              ENDDO
            ENDDO
          ENDDO
          IF (LSCSGRAD) THEN
            dtaux(2,1)=dtaux(1,2)
            dtaux(3,1)=dtaux(1,3)
            dtaux(3,2)=dtaux(2,3)
            dtauy(1,2)=dtauy(2,1)
            dtauy(3,2)=dtauy(2,3)
            dtauy(3,1)=dtauy(1,3)
            dtauz(1,3)=dtauz(3,1)
            dtauz(2,3)=dtauz(3,2)
            dtauz(2,1)=dtauz(1,2)

            dtau_h0x(2,1)=dtau_h0x(1,2)
            dtau_h0x(3,1)=dtau_h0x(1,3)
            dtau_h0x(3,2)=dtau_h0x(2,3)
            dtau_h1x(2,1)=dtau_h1x(1,2)
            dtau_h1x(3,1)=dtau_h1x(1,3)
            dtau_h1x(3,2)=dtau_h1x(2,3)
            dtau_h2x(2,1)=dtau_h2x(1,2)
            dtau_h2x(3,1)=dtau_h2x(1,3)
            dtau_h2x(3,2)=dtau_h2x(2,3)

            dtau_h0y(1,2)=dtau_h0y(2,1)
            dtau_h0y(3,2)=dtau_h0y(2,3)
            dtau_h0y(3,1)=dtau_h0y(1,3)
            dtau_h1y(1,2)=dtau_h1y(2,1)
            dtau_h1y(3,2)=dtau_h1y(2,3)
            dtau_h1y(3,1)=dtau_h1y(1,3)
            dtau_h2y(1,2)=dtau_h2y(2,1)
            dtau_h2y(3,2)=dtau_h2y(2,3)
            dtau_h2y(3,1)=dtau_h2y(1,3)

            dtau_h0z(1,3)=dtau_h0z(3,1)
            dtau_h0z(2,3)=dtau_h0z(3,2)
            dtau_h0z(2,1)=dtau_h0z(1,2)
            dtau_h1z(1,3)=dtau_h1z(3,1)
            dtau_h1z(2,3)=dtau_h1z(3,2)
            dtau_h1z(2,1)=dtau_h1z(1,2)
            dtau_h2z(1,3)=dtau_h2z(3,1)
            dtau_h2z(2,3)=dtau_h2z(3,2)
            dtau_h2z(2,1)=dtau_h2z(1,2)
          ENDIF
        END SUBROUTINE Tau_pq_lr 

        SUBROUTINE Tau_pq_lr_k(kvect,x1,x2,r0,dampD,sR,lattmat,translations,rmax,tau,&
        &   dtaux,dtauy,dtauz,dtau_h0x,dtau_h0y,dtau_h0z,dtau_h1x,&
        &   dtau_h1y,dtau_h1z,dtau_h2x,dtau_h2y,dtau_h2z,dtau_R0,LSCSGRAD,LSCALER0)
        !c dipole interaction tensor for TS-SCS
          INTEGER :: translations(3)
          INTEGER :: t1,t2,t3,i,j
          REAL(q) :: rmax,r_norm,r_norm2,part1,one_over_rnorm7
          REAL(q) :: kvect(3)
          COMPLEX(q) :: tau(3,3),tau_(3,3)
          REAL(q) :: lattmat(3,3)  !c transposed lattice vectors
          REAL(q) :: x1(3),x2(3),dr(3),dr_(3),r(3)
          COMPLEX(q) :: dtaux(3,3),dtauy(3,3),dtauz(3,3)
          COMPLEX(q) :: dtau_h0x(3,3),dtau_h0y(3,3),dtau_h0z(3,3)
          COMPLEX(q) :: dtau_h1x(3,3),dtau_h1y(3,3),dtau_h1z(3,3)
          COMPLEX(q) :: dtau_h2x(3,3),dtau_h2y(3,3),dtau_h2z(3,3)
          COMPLEX(q) :: dt_x(3,3),dt_y(3,3),dt_z(3,3)
          COMPLEX(q) :: dtau_R0(3,3)
          LOGICAL :: LSCSGRAD,LSCALER0
          REAL :: r_0(3),r_trans(3),r_transN,r_transA(3),r_transB(3)
          REAL(q), INTENT(in) :: dampD, sR,r0
          REAL(q) :: fdamp,dfdamp
          REAL(q) :: kx, ky, kz,Are,Aim,tpikt
          INTEGER :: shift(3)
          !REAL(q) :: r1(3),r2(3),r3(3)
          
         
          tau=CMPLX(0._q);tau_=CMPLX(0._q)
          dtaux=CMPLX(0._q);dtauy=CMPLX(0._q);dtauz=CMPLX(0._q)
          dtau_h0x=0._q;dtau_h0y=0._q;dtau_h0z=0._q
          dtau_h1x=0._q;dtau_h1y=0._q;dtau_h1z=0._q
          dtau_h2x=0._q;dtau_h2y=0._q;dtau_h2z=0._q
          dtau_R0=0._q
          dr_=x1-x2

!           r_0=dr_(1)*lattmat(:,1)+dr_(2)*lattmat(:,2)+dr_(3)*lattmat(:,3)

          kx=TPI*kvect(1)
          ky=TPI*kvect(2)
          kz=TPI*kvect(3)

          !c use minimum image convention
         shift=0
         CALL min_imageV_shift(3,dr_,shift)
         r_0=dr_(1)*lattmat(:,1)+dr_(2)*lattmat(:,2)+dr_(3)*lattmat(:,3)
          DO t1=-translations(1) ,translations(1)
            dr=0._q
            dr(1)=dr_(1)+t1*1._q
            r_transA=t1*lattmat(:,1)
            !r1=dr(1)*lattmat(1,:)
            DO t2=-translations(2) ,translations(2)
              dr(2)=dr_(2)+t2*1._q
              r_transB=r_transA+t2*lattmat(:,2)
              !r2=dr(2)*lattmat(2,:)
              DO t3=-translations(3) ,translations(3)
                !r_trans=t1*lattmat(:,1)+t2*lattmat(:,2)+t3*lattmat(:,3)
                r_trans=r_transB+t3*lattmat(:,3)
                r_transN=SUM(r_trans* r_trans)**0.5
                IF (r_transN .LE. rmax) THEN
                  dr(3)=dr_(3)+t3*1._q
                  !r3=dr(3)*lattmat(3,:)
                  !r=MATMUL(dr,lattmat)
                  !r=r1+r2+r3
                  r=r_0+r_trans

                  r_norm2=r(1)*r(1)+r(2)*r(2)+r(3)*r(3)
                  r_norm=r_norm2**0.5
                  one_over_rnorm7=1./(r_norm2*r_norm2*r_norm2*r_norm)
                  
!                   IF ((r_norm .GT. 1.e-1) .AND. (r_norm .LE. rmax)) THEN
                  IF ((r_norm .GT. 1.e-1) ) THEN
                    fdamp=damping_fermi(dampD,sR,r0,r_norm)
                    tpikt=kx*(t1+shift(1))+ky*(t2+shift(2))+kz*(t3+shift(3))
                    Are=COS(tpikt)
                    Aim=SIN(tpikt) 
                    
                    tau_=CMPLX(0._q)    
                    DO i=1,3
                      part1=-(3*r(i)*r(i)-r_norm*r_norm)
                      !tau_(i,i)=(part1)/r_norm**5*CMPLX(Are,Aim)
                      tau_(i,i)=(part1) !*CMPLX(Are,Aim)
                      DO j=i+1,3
                        part1=-(3*r(i)*r(j))
                        !tau_(i,j)=(part1)/r_norm**5*CMPLX(Are,Aim)
                        tau_(i,j)=(part1) !*CMPLX(Are,Aim)
                        tau_(j,i)=tau_(i,j)
                      ENDDO
                    ENDDO
                    !tau_=tau_/r_norm**5 *CMPLX(Are,Aim)
!                     tau_=tau_/(r_norm2*r_norm2*r_norm)
                    tau_=tau_*r_norm2*one_over_rnorm7
                         
                    IF (LSCSGRAD) THEN
                      dfdamp=damping_fermi_deriv(dampD,sR,r0,r_norm)
                      If (LSCALER0) THEN
                        dtau_R0=dtau_R0-dfdamp*r_norm/(sR*r0)*tau_*CMPLX(Are,Aim)
                      ENDIF
                      
                      !c important symmetry that we use here:
                      !dTau_jj(x,y,...)=dTau_ij(y,x,...)
                      dt_x(1,1)=dTau_ii_02(r(1),r_norm2,one_over_rnorm7)
                      dt_y(2,2)=dTau_ii_02(r(2),r_norm2,one_over_rnorm7)
                      dt_z(3,3)=dTau_ii_02(r(3),r_norm2,one_over_rnorm7)
                      dt_x(2,2)=dTau_jj_02(r(1),r(2),r_norm2,one_over_rnorm7)
                      dt_x(3,3)=dTau_jj_02(r(1),r(3),r_norm2,one_over_rnorm7)
                      dt_y(1,1)=dTau_jj_02(r(2),r(1),r_norm2,one_over_rnorm7)
                      dt_y(3,3)=dTau_jj_02(r(2),r(3),r_norm2,one_over_rnorm7)
                      dt_z(2,2)=dTau_jj_02(r(3),r(2),r_norm2,one_over_rnorm7)
                      dt_z(1,1)=dTau_jj_02(r(3),r(1),r_norm2,one_over_rnorm7)
                      dt_x(2,3)=dTau_jk_02(r(1),r(2),r(3),one_over_rnorm7)
                      dt_x(1,2)=dt_y(1,1)
                      dt_x(1,3)=dt_z(1,1)
                      dt_y(2,1)=dt_x(2,2)
                      dt_y(2,3)=dt_z(2,2)
                      dt_z(3,1)=dt_x(3,3)
                      dt_z(3,2)=dt_y(3,3)
                      dt_y(1,3)=dt_x(2,3)
                      dt_z(1,2)=dt_x(2,3)
                      
!                       dt_x(1,1)=dTau_ii_02(r(1),r_norm2,one_over_rnorm7)
!                       dt_x(1,2)=dTau_ij_02(r(1),r(2),r_norm2,one_over_rnorm7)
!                       dt_x(1,3)=dTau_ij_02(r(1),r(3),r_norm2,one_over_rnorm7)
!                       dt_x(2,2)=dTau_jj_02(r(1),r(2),r_norm2,one_over_rnorm7)
!                       dt_x(3,3)=dTau_jj_02(r(1),r(3),r_norm2,one_over_rnorm7)
!                       dt_x(2,3)=dTau_jk_02(r(1),r(2),r(3),one_over_rnorm7)
! 
!                       dt_y(2,2)=dTau_ii_02(r(2),r_norm2,one_over_rnorm7)
!                       dt_y(2,1)=dTau_ij_02(r(2),r(1),r_norm2,one_over_rnorm7)
!                       dt_y(2,3)=dTau_ij_02(r(2),r(3),r_norm2,one_over_rnorm7)
!                       dt_y(1,1)=dTau_jj_02(r(2),r(1),r_norm2,one_over_rnorm7)
!                       dt_y(3,3)=dTau_jj_02(r(2),r(3),r_norm2,one_over_rnorm7)
!                       dt_y(1,3)=dTau_jk_02(r(2),r(3),r(1),one_over_rnorm7)
!                       
!           
!                       dt_z(3,3)=dTau_ii_02(r(3),r_norm2,one_over_rnorm7)
!                       dt_z(3,1)=dTau_ij_02(r(3),r(1),r_norm2,one_over_rnorm7)
!                       dt_z(3,2)=dTau_ij_02(r(3),r(2),r_norm2,one_over_rnorm7)
!                       dt_z(2,2)=dTau_jj_02(r(3),r(2),r_norm2,one_over_rnorm7)
!                       dt_z(1,1)=dTau_jj_02(r(3),r(1),r_norm2,one_over_rnorm7)
!                       dt_z(1,2)=dTau_jk_02(r(3),r(1),r(2),one_over_rnorm7) 
                      
                      
                      !dt_x=0.;dt_y=0.;dt_z=0.

!                       dt_x=dt_x*(fdamp)*CMPLX(Are,Aim)+tau_*dfdamp*r(1)/r_norm 
!                       dt_y=dt_y*(fdamp)*CMPLX(Are,Aim)+tau_*dfdamp*r(2)/r_norm
!                       dt_z=dt_z*(fdamp)*CMPLX(Are,Aim)+tau_*dfdamp*r(3)/r_norm

!                       dt_x=(dt_x*(fdamp) -tau_*dfdamp*r(1)/r_norm)*CMPLX(Are,Aim) 
!                       dt_y=(dt_y*(fdamp) -tau_*dfdamp*r(2)/r_norm)*CMPLX(Are,Aim)
!                       dt_z=(dt_z*(fdamp) -tau_*dfdamp*r(3)/r_norm)*CMPLX(Are,Aim)
!                                   
                      dt_x=(dt_x*(fdamp)+tau_*dfdamp*r(1)/r_norm)*CMPLX(Are,Aim)
                      dt_y=(dt_y*(fdamp)+tau_*dfdamp*r(2)/r_norm)*CMPLX(Are,Aim)
                      dt_z=(dt_z*(fdamp)+tau_*dfdamp*r(3)/r_norm)*CMPLX(Are,Aim)

                      dtaux=dtaux+dt_x                 
                      dtauy=dtauy+dt_y
                      dtauz=dtauz+dt_z

                      dtau_h0x=dtau_h0x+dt_x*dr(1)
                      dtau_h1x=dtau_h1x+dt_x*dr(2)
                      dtau_h2x=dtau_h2x+dt_x*dr(3)
                      dtau_h0y=dtau_h0y+dt_y*dr(1)
                      dtau_h1y=dtau_h1y+dt_y*dr(2)
                      dtau_h2y=dtau_h2y+dt_y*dr(3)
                      dtau_h0z=dtau_h0z+dt_z*dr(1)
                      dtau_h1z=dtau_h1z+dt_z*dr(2)
                      dtau_h2z=dtau_h2z+dt_z*dr(3)
                    ENDIF
        
                    tau=tau+tau_*fdamp*CMPLX(Are,Aim)

                  ENDIF
                ENDIF
              ENDDO
            ENDDO
          ENDDO
          IF (LSCSGRAD) THEN
            dtaux(2,1)=dtaux(1,2)
            dtaux(3,1)=dtaux(1,3)
            dtaux(3,2)=dtaux(2,3)
            dtauy(1,2)=dtauy(2,1)
            dtauy(3,2)=dtauy(2,3)
            dtauy(3,1)=dtauy(1,3)
            dtauz(1,3)=dtauz(3,1)
            dtauz(2,3)=dtauz(3,2)
            dtauz(2,1)=dtauz(1,2)

            dtau_h0x(2,1)=dtau_h0x(1,2)
            dtau_h0x(3,1)=dtau_h0x(1,3)
            dtau_h0x(3,2)=dtau_h0x(2,3)
            dtau_h1x(2,1)=dtau_h1x(1,2)
            dtau_h1x(3,1)=dtau_h1x(1,3)
            dtau_h1x(3,2)=dtau_h1x(2,3)
            dtau_h2x(2,1)=dtau_h2x(1,2)
            dtau_h2x(3,1)=dtau_h2x(1,3)
            dtau_h2x(3,2)=dtau_h2x(2,3)

            dtau_h0y(1,2)=dtau_h0y(2,1)
            dtau_h0y(3,2)=dtau_h0y(2,3)
            dtau_h0y(3,1)=dtau_h0y(1,3)
            dtau_h1y(1,2)=dtau_h1y(2,1)
            dtau_h1y(3,2)=dtau_h1y(2,3)
            dtau_h1y(3,1)=dtau_h1y(1,3)
            dtau_h2y(1,2)=dtau_h2y(2,1)
            dtau_h2y(3,2)=dtau_h2y(2,3)
            dtau_h2y(3,1)=dtau_h2y(1,3)

            dtau_h0z(1,3)=dtau_h0z(3,1)
            dtau_h0z(2,3)=dtau_h0z(3,2)
            dtau_h0z(2,1)=dtau_h0z(1,2)
            dtau_h1z(1,3)=dtau_h1z(3,1)
            dtau_h1z(2,3)=dtau_h1z(3,2)
            dtau_h1z(2,1)=dtau_h1z(1,2)
            dtau_h2z(1,3)=dtau_h2z(3,1)
            dtau_h2z(2,3)=dtau_h2z(3,2)
            dtau_h2z(2,1)=dtau_h2z(1,2)
          ENDIF
        END SUBROUTINE Tau_pq_lr_k



        FUNCTION dTau_ii_di(x,rnorm,rrpq,rpq)  RESULT(iidi)
        !c derivative of Tau_ii wrt. coordinate i
          REAL(q) :: x, rnorm, rrpq,rpq
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: iidi

          A=-errf(rrpq)
          B=(18*rpq**2+12*rnorm**2)/rnorm**4/rpq**3/SQRT(PI)
          B=B*EXP(-rrpq**2)
          C=-(30*rpq**4+20*rpq**2*rnorm**2+8*rnorm**4)
          C=C*EXP(-rrpq**2)/rnorm**6/rpq**5/SQRT(PI)   
          part1=(9*x/rnorm**5-15*x**3/rnorm**7)*A 
          part2=x*B
          part3=x**3*C
          iidi=part1+part2+part3
        END FUNCTION dTau_ii_di

        FUNCTION dTau_ij_di(x,y,rnorm,rrpq,rpq)  RESULT(ijdi)
        !c derivative of Tau_ij wrt. coordinate i
          REAL(q) :: x,y,rnorm,rrpq,rpq
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: ijdi

          A=-errf(rrpq)
          B=(6*rpq**2+4*rnorm**2)
          B=B*EXP(-rrpq**2)/rnorm**4/rpq**3/SQRT(PI)
          C=-(30*rpq**4+20*rpq**2*rnorm**2+8*rnorm**4)
          C=C*EXP(-rrpq**2)/rnorm**6/rpq**5/SQRT(PI)   
          part1=(3*y/rnorm**5-15*x**2*y/rnorm**7)*A 
          part2=y*B
          part3=x**2*y*C
          ijdi=part1+part2+part3
        END FUNCTION dTau_ij_di

        FUNCTION dTau_jj_di(x,y,rnorm,rrpq,rpq)  RESULT(jjdi)
        !c derivative of Tau_jj wrt. coordinate i
          REAL(q) :: x,y,rnorm,rrpq,rpq
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: jjdi

          A=-errf(rrpq)
          B=(6*rpq**2+4*rnorm**2)
          B=B*EXP(-rrpq**2)/rnorm**4/rpq**3/SQRT(PI)
          C=-(30*rpq**4+20*rpq**2*rnorm**2+8*rnorm**4)
          C=C*EXP(-rrpq**2)/rnorm**6/rpq**5/SQRT(PI)   
          part1=(3*x/rnorm**5-15*x*y**2/rnorm**7)*A 
          part2=x*B
          part3=x*y**2*C
          jjdi=part1+part2+part3
        END FUNCTION dTau_jj_di

        FUNCTION dTau_jk_di(x,y,z,rnorm,rrpq,rpq)  RESULT(jkdi)
        !c derivative of Tau_jk wrt. coordinate i
          REAL(q) :: x,y,z,rnorm,rrpq,rpq
          REAL(q) :: A,C
          REAL(q) :: part1, part3
          REAL(q) :: jkdi

          A=15*errf(rrpq)/rnorm**7
          C=-(30*rpq**4+20*rpq**2*rnorm**2+8*rnorm**4)
          C=C*EXP(-rrpq**2)/rnorm**6/rpq**5/SQRT(PI)   
          part1=x*y*z*A 
          part3=x*y*z*C
          jkdi=part1+part3
        END FUNCTION dTau_jk_di
        
        FUNCTION dTau_ii_di2(x,rnorm,rnorm2,rpq,rpq2,expfact,errffact)  RESULT(iidi)
        !c derivative of Tau_ii wrt. coordinate i
          REAL(q) :: x, rnorm,rpq,rpq2
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: iidi
          REAL(q) :: expfact,errffact,rnorm2

          A=-errffact
          B=12.*(1.5*rpq2+rnorm2) !/(rnorm2*rnorm2*rpq2*rpq*SQRT(PI))
          !B=B*expfact
          C=-8.*(rpq2*(3.75*rpq2+2.5*rnorm2)+rnorm2*rnorm2)
          !!!C=C*expfact/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))   
          !part1=(9./(rnorm2*rnorm2*rnorm)-15.*x*x/(rnorm2*rnorm2*rnorm2*rnorm))*A*x 
          part1=15.*A*x*(0.6*rnorm2-x*x)/(rnorm2*rnorm2*rnorm2*rnorm)
          part2=x*B
          part3=x*x*x*C
          iidi=part1+part2+part3
          
          iidi=part1+expfact*(part2*rnorm2*rpq2+part3)/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))
          
        END FUNCTION dTau_ii_di2

        FUNCTION dTau_ij_di2(x,y,rnorm,rnorm2,rpq,rpq2,expfact,errffact)  RESULT(ijdi)
        !c derivative of Tau_ij wrt. coordinate i
          REAL(q) :: x,y,rnorm,rpq,rpq2
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: ijdi
          REAL(q) :: expfact,errffact,rnorm2

          A=-errffact
          B=4.*(1.5*rpq2+rnorm2)
          !B=B*expfact/(rnorm2*rnorm2*rpq2*rpq*SQRT(PI))
          C=-8.*(rpq2*(3.75*rpq2+2.5*rnorm2)+rnorm2*rnorm2)
          !C=C*expfact/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))   
          !part1=(3./(rnorm2*rnorm2*rnorm)-15.*x*x/(rnorm2*rnorm2*rnorm2*rnorm))*A*y 
          part1=15.*A*y*(0.2*rnorm2-x*x)/(rnorm2*rnorm2*rnorm2*rnorm)
          part2=y*B
          part3=x*x*y*C
          ijdi=part1+expfact*(part2*rnorm2*rpq2+part3)/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))
        END FUNCTION dTau_ij_di2

        FUNCTION dTau_jj_di2(x,y,rnorm,rnorm2,rpq,rpq2,expfact,errffact)  RESULT(jjdi)
        !c derivative of Tau_jj wrt. coordinate i
          REAL(q) :: x,y,rnorm,rpq,rpq2
          REAL(q) :: A,B,C
          REAL(q) :: part1, part2, part3
          REAL(q) :: jjdi
          REAL(q) :: expfact,errffact,rnorm2

          A=-errffact 
          B=4.*(1.5*rpq2+rnorm2)
          !B=B*expfact/(rnorm2*rnorm2*rpq2*rpq*SQRT(PI))
          C=-8.*(rpq2*(3.75*rpq2+2.5*rnorm2)+rnorm2*rnorm2)
          !C=C*expfact/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))   
          !part1=(3./(rnorm2*rnorm2*rnorm)-15.*y*y/(rnorm2*rnorm2*rnorm2*rnorm))*A*x 
          part1=15.*A*x*(0.2*rnorm2-y*y)/(rnorm2*rnorm2*rnorm2*rnorm)
          part2=x*B
          part3=x*y*y*C
          jjdi=part1+expfact*(part2*rnorm2*rpq2+part3)/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))  
        END FUNCTION dTau_jj_di2

        FUNCTION dTau_jk_di2(x,y,z,rnorm,rnorm2,rpq,rpq2,expfact,errffact)  RESULT(jkdi)
        !c derivative of Tau_jk wrt. coordinate i
          REAL(q) :: x,y,z,rnorm,rpq,rpq2
          REAL(q) :: A,C
          REAL(q) :: part1, part3
          REAL(q) :: jkdi
          REAL(q) :: expfact,errffact,rnorm2

          A=15.*errffact/(rnorm2*rnorm2*rnorm2*rnorm)
          C=-8.*(rpq2*(3.75*rpq2+2.5*rnorm2)+rnorm2*rnorm2)
          C=C*expfact/(rnorm2*rnorm2*rnorm2*rpq2*rpq2*rpq*SQRT(PI))   
          part1=x*y*z*A 
          part3=x*y*z*C
          jkdi=part1+part3
        END FUNCTION dTau_jk_di2

        FUNCTION dTau_ii_0(x,rnorm)  RESULT(iidi)
        !c derivative of Tau_ii wrt. coordinate i
          REAL(q),INTENT(in) :: x, rnorm
          REAL(q) :: iidi
  
          iidi=-15.*x*(0.6*rnorm*rnorm-x*x)/rnorm**7
        END FUNCTION dTau_ii_0

        FUNCTION dTau_ij_0(x,y,rnorm)  RESULT(ijdi)
        !c derivative of Tau_ij wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,rnorm
          REAL(q) :: ijdi      
  
          ijdi=-15*y*(0.2*rnorm*rnorm-x*x)/rnorm**7
        END FUNCTION dTau_ij_0

        FUNCTION dTau_jj_0(x,y,rnorm)  RESULT(jjdi)
        !c derivative of Tau_jj wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,rnorm
          REAL(q) :: jjdi
       
          jjdi=-15*x*(0.2*rnorm*rnorm-y*y) /rnorm**7
        END FUNCTION dTau_jj_0

        FUNCTION dTau_jk_0(x,y,z,rnorm)  RESULT(jkdi)
        !c derivative of Tau_jk wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,z,rnorm        
          REAL(q) :: jkdi
         
          jkdi=15.*x*y*z/rnorm**7        
        END FUNCTION dTau_jk_0

        FUNCTION dTau_ii_02(x,r_norm2,one_over_rnorm7)  RESULT(iidi)
        !c derivative of Tau_ii wrt. coordinate i
          REAL(q),INTENT(in) :: x,r_norm2,one_over_rnorm7
          REAL(q) :: iidi
  
          iidi=-15.*x*one_over_rnorm7*(0.6*r_norm2-x*x)
        END FUNCTION dTau_ii_02

        FUNCTION dTau_ij_02(x,y,r_norm2,one_over_rnorm7)  RESULT(ijdi)
        !c derivative of Tau_ij wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,r_norm2,one_over_rnorm7
          REAL(q) :: ijdi      
  
          ijdi=-15*y*one_over_rnorm7*(0.2*r_norm2-x*x)
        END FUNCTION dTau_ij_02

        FUNCTION dTau_jj_02(x,y,r_norm2,one_over_rnorm7)  RESULT(jjdi)
        !c derivative of Tau_jj wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,r_norm2,one_over_rnorm7
          REAL(q) :: jjdi
       
          jjdi=-15*x*one_over_rnorm7*(0.2*r_norm2-y*y) 
        END FUNCTION dTau_jj_02

        FUNCTION dTau_jk_02(x,y,z,one_over_rnorm7)  RESULT(jkdi)
        !c derivative of Tau_jk wrt. coordinate i
          REAL(q),INTENT(in) :: x,y,z,one_over_rnorm7        
          REAL(q) :: jkdi
         
          jkdi=15.*x*y*z*one_over_rnorm7        
        END FUNCTION dTau_jk_02
        
    SUBROUTINE give_val0(x0,x,D,y,y0,n)
    !c simple linear interpolation for a function
    !c defined on logarithmic grid
      USE prec
      INTEGER :: n,indx1,indx2
      REAL(q) :: x0,y0,D,dx1,dx2
      REAL(q) :: x(n),y(n)

      IF (x0<x(1)) THEN
        y0=0._q
        RETURN
      ENDIF

      !indx1=INT((log(x0)-log(x(1)))/log(D))+1
      indx1=INT(log((x0/x(1)))/log(D)+1)
      indx2=indx1+1      
      dx1=x(indx2)-x(indx1)
      dx2=x0-x(indx1)
      y0=y(indx1)+(y(indx2)-y(indx1))/dx1*dx2
     
    END SUBROUTINE give_val0

    SUBROUTINE give_val0_new(x0,x,H,y,y0,n)
    !c simple linear interpolation for a function
    !c defined on logarithmic grid
      USE prec
      INTEGER :: n,indx1,indx2
      REAL(q) :: x0,y0,H,dx1,dx2
      REAL(q) :: x(n),y(n)

      IF (x0<x(1)) THEN
        y0=0._q
        RETURN
      ENDIF

      !indx1=INT((log(x0)-log(x(1)))/log(D))+1
      indx1=INT(log((x0/x(1)))/H+1)
      indx2=indx1+1      
      dx1=x(indx2)-x(indx1)
      dx2=x0-x(indx1)
      y0=y(indx1)+(y(indx2)-y(indx1))/dx1*dx2
     
    END SUBROUTINE give_val0_new

    SUBROUTINE give_val0_two(x0,x,H,yA,yB,yA0,yB0,n)
    !c simple linear interpolation for a function
    !c defined on logarithmic grid
      USE prec
      INTEGER :: n,indx1,indx2
      REAL(q) :: x0,yA0,yB0,H,dx1,dx2,x0_sq
      REAL(q) :: x(n),yA(n),yB(n)
      REAL(q) :: rdum

      IF (x0<x(1)) THEN
        yA0=0._q
        yB0=0._q
        RETURN
      ENDIF

      rdum=x0/x(1)
      !indx1=INT((log(x0)-log(x(1)))/log(D))+1
      !indx1=INT(log((x0/x(1)))/H+1)
      indx1=INT(log(rdum)/H+1)
      indx2=indx1+1      
      dx1=x(indx2)-x(indx1)
      dx2=x0-x(indx1)
      yA0=yA(indx1)+(yA(indx2)-yA(indx1))/dx1*dx2
      yB0=yB(indx1)+(yB(indx2)-yB(indx1))/dx1*dx2
     
    END SUBROUTINE give_val0_two


    SUBROUTINE min_imageX(dxn)
    !c minimum image convention in 1D
     USE prec
     REAL(q) :: dxn
  
      !write(*,*) 'dx1',dxn 
      DO 
        IF (dxn .GT. 0.5_q) THEN
          dxn=dxn-1._q
        ELSE IF (dxn .LE. -0.5_q) THEN
          dxn=dxn+1._q
        ELSE 
          EXIT
        END IF
      END DO
      !write(*,*) 'dx2',dxn 
    END SUBROUTINE min_imageX

    SUBROUTINE rel_vol_plane2(T_INFO,LATT_CUR,part,dimCHG2D,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions,ngx,ngy,ngz,nz,translations,Rmax,RELVOL,RELCHG,NODE_ME,NCPU)
      USE prec
      USE lattice
      USE poscar
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: dimCHG2D,ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part(dimCHG2D)
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      REAL(q) :: RELVOL,RELCHG
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      INTEGER :: NODE_ME,NCPU  
      REAL(q),PARAMETER :: mytiny=0._q

      dZ=positions(3)-(nz-1._q)/ngz
      CALL min_imageX(dZ)
      DO ny=NODE_ME,ngy,NCPU
        indxY=(ny-1)*ngx
        dY=positions(2)-(ny-1._q)/ngy
        CALL min_imageX(dY)
        DO nx=1,ngx
          indx=indxY+nx
          dX=positions(1)-(nx-1._q)/ngx
          CALL min_imageX(dX) 
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            DO t2=-translations(2),translations(2)
              dY_=dY+t2*1._q
              DO t1=-translations(1),translations(1)
                dX_=dX+t1*1._q
                diff=(/dX_,dY_,dZ_/)
                diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))    
                r=SUM(diff**2)**0.5
                IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
                !IF ((r>mytiny) .AND. (r .LT. Rmax)) THEN
                  CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)
                  !!! Let's remeber this:
                  !IF (ATval .GT. 1e-3) THEN
                    RELVOL=RELVOL+r*ATval*part(indx)
                    RELCHG=RELCHG+ATval*(1.- part(indx))/r**2
                  !ENDIF
                ENDIF
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      END DO
    END SUBROUTINE rel_vol_plane2

    SUBROUTINE rel_vol_planeDOM(T_INFO,LATT_CUR,part,WDF,P, &
    &                   positions,ngx,ngy,ngz,nx,ny,nz,translations,Rmax,RELVOL,RELCHG,M2,Overlap,XDM)
    !c compute contribution of a point nx,ny,nz to the Hirshfeld dominant charge
    !c and to <r^3> 
      use prec
      use rhfatm
      use poscar
      USE pseudo
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      TYPE (atomic) :: WDF(:)
      TYPE (potcar),TARGET ::   P(T_INFO%NTYP)
      TYPE (potcar), POINTER:: PP
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part,XDM!,overT
      INTEGER :: translations(3)
      REAL(q) :: positions(:,:)
      REAL(q),ALLOCATABLE :: dZ(:),dY(:),dX(:),dX_(:),dY_(:),dZ_(:)
      REAL(q) :: diff(3),r,rr,Rmax,diff1(3)
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3,IDOM!,IDOMA(1)
      REAL(q) :: RELVOL(T_INFO%NIONS),RELCHG(T_INFO%NIONS),RHOAT,RHOMAX
      REAL(q) :: Overlap(T_INFO%NIONS,T_INFO%NIONS),M2(T_INFO%NIONS)
      REAL(q),PARAMETER :: mytiny=0._q
      REAL(q),ALLOCATABLE :: diff2(:,:),diff3(:,:),Avalc(:),Avalv(:)
      INTEGER :: I,INDX,J
      ALLOCATE(dZ(T_INFO%NIONS),dY(T_INFO%NIONS),dX(T_INFO%NIONS))
      ALLOCATE(dZ_(T_INFO%NIONS),dY_(T_INFO%NIONS),dX_(T_INFO%NIONS))
      ALLOCATE(diff2(3,T_INFO%NIONS),diff3(3,T_INFO%NIONS),Avalc(T_INFO%NIONS),Avalv(T_INFO%NIONS))
      DO i=1,T_INFO%NIONS

      dZ(i)=positions(3,i)+(1._q-nz)/ngz
      dY(i)=positions(2,i)+(1._q-ny)/ngy
      dX(i)=positions(1,i)+(1._q-nx)/ngx

      CALL min_imageX(dZ(i))
      CALL min_imageX(dY(i))
      CALL min_imageX(dX(i))
      ENDDO
      DO t3=-translations(3) ,translations(3) 
      DO i=1,T_INFO%NIONS
        dZ_(i)=dZ(i)+t3*1._q
        diff3(:,i)=dZ_(i)*LATT_CUR%A(:,3)
      END DO
        DO t2=-translations(2),translations(2)
      DO i=1,T_INFO%NIONS
          dY_(i)=dY(i)+t2*1._q
          diff2(:,i)=dY_(i)*LATT_CUR%A(:,2)
      END DO
          DO t1=-translations(1),translations(1)
      RHOMAX=0._q
      Avalv=0._q
      Avalc=0._q
      RHOAT=0._q
      IDOM=1
      DO i=1,T_INFO%NIONS
           indx=T_INFO%ITYP(i)    
!           PP=>P(indx)            
            dX_(i)=dX(i)+t1*1._q
            diff1=dX_(i)*LATT_CUR%A(:,1)
            diff=diff1+diff2(:,i)+diff3(:,i)   
            rr=  SUM(diff*diff)  
            r=rr**0.5_q
            IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
              CALL give_val0(r,WDF(indx)%R%R,WDF(indx)%R%D,WDF(indx)%RHO,ATval,WDF(indx)%R%NMAX)
              Avalv(i)=ATval*r
              Avalc(i)=ATval/rr
              RHOAT=RHOAT+Avalc(i)
              Overlap(I,I)=Overlap(I,I)+Avalc(i)*(1._q- part)
!SNS: replace by a maxloc function
#if 0
                   if(Avalc(i).gt.RHOMAX) then
                       RHOMAX=Avalc(i)
                       IDOM=I
                   ENDIF
#endif

            ENDIF
      END DO
       IDOM=MAXLOC(Avalc,1)
              M2(IDOM)=M2(IDOM)+RHOAT*part*XDM
            IF(Avalc(IDOM).gt.1.0D-9.AND.RHOAT/(4._q*PI).gt.rholow) then
              RELVOL(IDOM)=RELVOL(IDOM)+Avalv(IDOM)*RHOAT/Avalc(IDOM)*part
            END IF
              RELCHG(IDOM)=RELCHG(IDOM)+RHOAT*(1._q- part)
      DO I=1,T_INFO%NIONS
      DO J=1,T_INFO%NIONS
         if(i.eq.j) cycle
                   if(RHOAT.gt.1e-4) then
                   Overlap(I,J)=Overlap(I,J)+(Avalc(I)*Avalc(J)/RHOAT)*part
                   endif
      END DO
      END DO

          ENDDO
        ENDDO
      ENDDO
      DEALLOCATE(dZ,dY,dX)
      DEALLOCATE(dZ_,dY_,dX_)
      DEALLOCATE(diff2,diff3,Avalc,Avalv)
    END SUBROUTINE rel_vol_planeDOM

    SUBROUTINE rel_vol_plane3(T_INFO,LATT_CUR,part,dimatCHG1D,&
    & atCHG1D, atGrid,H,positions,ngx,ngy,ngz,nx,ny,nz,translations,Rmax,RELVOL,RELCHG)
    !c compute contribution of a point nx,ny,nz to the Hirshfeld charge
    !c and to <r^3> for atom with position positions
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,rr,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3
      REAL(q) :: RELVOL,RELCHG
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: H
      REAL(q),PARAMETER :: mytiny=0._q
      REAL(q) :: diff1(3),diff2(3),diff3(3)

      dZ=positions(3)+(1._q-nz)/ngz
      dY=positions(2)+(1._q-ny)/ngy
      dX=positions(1)+(1._q-nx)/ngx

      CALL min_imageX(dZ)
      CALL min_imageX(dY)
      CALL min_imageX(dX)

! #ifdef MPI
!       DO t2=-translations(2),translations(2)
!         dY_=dY+t2*1._q
!         diff2=dY_*LATT_CUR%A(:,2)
!         DO t1=-translations(1),translations(1)
!          dX_=dX+t1*1._q
!          diff1=dX_*LATT_CUR%A(:,1)
!           DO t3=-translations(3) ,translations(3) 
!             dZ_=dZ+t3*1._q
!             diff3=dZ_*LATT_CUR%A(:,3)
! #else
      DO t3=-translations(3) ,translations(3) 
        dZ_=dZ+t3*1._q
        diff3=dZ_*LATT_CUR%A(:,3)
        DO t2=-translations(2),translations(2)
          dY_=dY+t2*1._q
          diff2=dY_*LATT_CUR%A(:,2)
          DO t1=-translations(1),translations(1)
            dX_=dX+t1*1._q
            diff1=dX_*LATT_CUR%A(:,1)
! #endif
            diff=diff1+diff2+diff3   
            rr=  SUM(diff*diff)  
            r=rr**0.5
            IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
              CALL give_val0_new(r,atGrid,H,atCHG1D,ATval,dimatCHG1D)
              RELVOL=RELVOL+r*ATval*part
              RELCHG=RELCHG+ATval*(1.- part)/rr
            ENDIF
          ENDDO
        ENDDO
      ENDDO

    END SUBROUTINE rel_vol_plane3

    SUBROUTINE rel_vol_plane4(part,RRmax,rdim,Rarray,RhoAtaarray,RELVOL,RELCHG)
    !c compute contribution of a point nx,ny,nz to the Hirshfeld charge
    !c and to <r^3> for atom with position positions_c 
      REAL(q) :: part,RRmax,rr,ATval
      REAL(q) :: RELVOL,RELCHG
      INTEGER :: dimatCHG1D
      INTEGER :: i,rdim
      REAL(q) :: Rarray(rdim),RhoAtaarray(rdim)

      DO i=1,rdim
        rr=Rarray(i)
        IF ((rr .GT. 0._q) .AND. (rr .LE. RRmax)) THEN
        !IF (r>0._q) THEN
          ATval=RhoAtaarray(i)
          RELVOL=RELVOL+ATval*part*rr**0.5
          RELCHG=RELCHG+ATval*(1.- part)/rr    
        !ELSE 
        !  EXIT
        ENDIF
      ENDDO

    END SUBROUTINE rel_vol_plane4

    SUBROUTINE hirshfeld_charge_point(T_INFO,LATT_CUR,part,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions_c,ngx,ngy,ngz,nx,ny,nz,translations,Rmax,RELCHG)
    !c compute contribution of a point nx,ny,nz to the Hirshfeld charge
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part
      INTEGER :: translations(3)
      REAL(q) :: positions_c(3)      !c position vector in Cart. coords
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3
      REAL(q) :: RELCHG
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      REAL(q),PARAMETER :: mytiny=0._q
      REAL(q) :: diff1(3),diff2(3),diff3(3)

      dZ=(1._q-nz)/ngz
      dY=(1._q-ny)/ngy
      dX=(1._q-nx)/ngx

#ifdef MPI
      DO t2=-translations(2),translations(2)
        dY_=dY+t2*1._q
        diff2=dY_*LATT_CUR%A(:,2)
        DO t1=-translations(1),translations(1)
          dX_=dX+t1*1._q
          diff1=dX_*LATT_CUR%A(:,1)
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            diff3=dZ_*LATT_CUR%A(:,3)
#else
      DO t3=-translations(3) ,translations(3) 
        dZ_=dZ+t3*1._q
        diff3=dZ_*LATT_CUR%A(:,3)
        DO t2=-translations(2),translations(2)
          dY_=dY+t2*1._q
          diff2=dY_*LATT_CUR%A(:,2)
          DO t1=-translations(1),translations(1)
            dX_=dX+t1*1._q
            diff1=dX_*LATT_CUR%A(:,1)
#endif
            diff=positions_c+diff1+diff2+diff3   
            !diff=(/dX_,dY_,dZ_/)
            !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))         
            r=SUM(diff**2)**0.5
            IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
              CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)
              RELCHG=RELCHG+ATval*(1.- part)/r**2
            ENDIF
          ENDDO
        ENDDO
      ENDDO

    END SUBROUTINE hirshfeld_charge_point

    SUBROUTINE AIM_volume_point(T_INFO,LATT_CUR,part,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions_c,ngx,ngy,ngz,nx,ny,nz,translations,Rmax,RELVOL)
    !c contribution of a point nx,ny,nz to value of <r^3> computed using Hirshfeld AIM scheme
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part
      INTEGER :: translations(3)
      REAL(q) :: positions_c(3)      !c position vector in Cart. coords
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3
      REAL(q) :: RELVOL
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      REAL(q),PARAMETER :: mytiny=0._q
      REAL(q) :: diff1(3),diff2(3),diff3(3)

      dZ=(1._q-nz)/ngz
      dY=(1._q-ny)/ngy
      dX=(1._q-nx)/ngx

!       dZ=positions_c(3)-(nz-1._q)/ngz
!       dY=positions_c(2)-(ny-1._q)/ngy
!       dX=positions_c(1)-(nx-1._q)/ngx

#ifdef MPI
      DO t2=-translations(2),translations(2)
        dY_=dY+t2*1._q
        diff2=dY_*LATT_CUR%A(:,2)
        DO t1=-translations(1),translations(1)
          dX_=dX+t1*1._q
          diff1=dX_*LATT_CUR%A(:,1)
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            diff3=dZ_*LATT_CUR%A(:,3)
#else
      DO t3=-translations(3) ,translations(3) 
        dZ_=dZ+t3*1._q
        diff3=dZ_*LATT_CUR%A(:,3)
        DO t2=-translations(2),translations(2)
          dY_=dY+t2*1._q
          diff2=dY_*LATT_CUR%A(:,2)
          DO t1=-translations(1),translations(1)
            dX_=dX+t1*1._q
            diff1=dX_*LATT_CUR%A(:,1)
#endif
            diff=positions_c+diff1+diff2+diff3   
            !diff=(/dX_,dY_,dZ_/)
            !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))         
            r=SUM(diff**2)**0.5
            IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
              CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)
              RELVOL=RELVOL+r*ATval*part
            ENDIF
          ENDDO
        ENDDO
      ENDDO

    END SUBROUTINE AIM_volume_point


    SUBROUTINE hirschfeld_charge_plane(T_INFO,LATT_CUR,part,dimCHG2D,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions,ngx,ngy,ngz,nz,translations,Rmax,RELCHG,RELVOL,NODE_ME,NCPU,DOWNSAMPLE)
    !c charge for Hirscfeld partitioning
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: dimCHG2D,ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part(dimCHG2D)
      INTEGER :: translations(3)
      REAL(q) :: positions(3),positions_c(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,rr,Rmax
      REAL(q) :: diff1(3),diff2(3),diff3(3)
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      REAL(q) :: BTval
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      REAL(q) :: RELVOL,RELCHG
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      INTEGER :: NODE_ME,NCPU,DOWNSAMPLE 
      !REAL(q),PARAMETER :: rholimit=1e-4 !c inspired by Bader's volumes

      !dZ=positions(3)-(nz-1._q)/ngz
      !CALL min_imageX(dZ)
      positions_c=MATMUL(positions,TRANSPOSE(LATT_CUR%A))  

      dZ=(1._q-nz)/ngz
      DO indx=NODE_ME,dimCHG2D,NCPU
        ny=indx/ngx+1
        nx=MOD(indx,ngx)
        IF (nx==0) THEN 
          nx=ngx
          ny=ny-1
        END IF
        !dY=positions(2)-(ny-1._q)/ngy
        !CALL min_imageX(dY)
        !dX=positions(1)-(nx-1._q)/ngx
        !CALL min_imageX(dX) 
        dY=(1._q-ny)/ngy
        !CALL min_imageX(dY)
        dX=(1._q-nx)/ngx


!       DO ny=DOWNSAMPLE*NODE_ME-(DOWNSAMPLE-1),ngy,DOWNSAMPLE*NCPU
!         indxY=(ny-1)*ngx
!         dY=positions(2)-(ny-1._q)/ngy
!         CALL min_imageX(dY)
!         DO nx=1,ngx,DOWNSAMPLE
!           indx=indxY+nx
!           dX=positions(1)-(nx-1._q)/ngx
!           CALL min_imageX(dX) 
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            diff3=dZ_*LATT_CUR%A(:,3)
            DO t2=-translations(2),translations(2)
              dY_=dY+t2*1._q
              diff2=dY_*LATT_CUR%A(:,2)
              DO t1=-translations(1),translations(1)
                dX_=dX+t1*1._q
                diff1=dX_*LATT_CUR%A(:,1)
                diff=positions_c+diff1+diff2+diff3
                !diff=(/dX_,dY_,dZ_/)
                !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))    
                rr=SUM(diff**2)
                r=rr**0.5
                IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
               
                  CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)
                  RELCHG=RELCHG+ATval*(1.- part(indx))/rr
                  RELVOL=RELVOL+r*ATval*part(indx)

                ENDIF
              ENDDO
            ENDDO
          ENDDO
!         ENDDO
      END DO
    END SUBROUTINE hirschfeld_charge_plane

    SUBROUTINE hirschfeld_dominant_charge_plane(T_INFO,LATT_CUR,part,part_promol,dimCHG2D,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions,ngx,ngy,ngz,nz,translations,Rmax,RELCHG,RELVOL,NODE_ME,NCPU,DOWNSAMPLE)
    !c charge for Hirschfeld partitioning
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: dimCHG2D,ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part(dimCHG2D),part_promol(dimCHG2D)
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      REAL(q) :: BTval
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      REAL(q) :: RELVOL,RELCHG
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      INTEGER :: NODE_ME,NCPU,DOWNSAMPLE 
      !REAL(q),PARAMETER :: rholimit=1e-4 !c inspired by Bader's volumes

      dZ=positions(3)-(nz-1._q)/ngz
      CALL min_imageX(dZ)
      DO ny=DOWNSAMPLE*NODE_ME-(DOWNSAMPLE-1),ngy,DOWNSAMPLE*NCPU
        indxY=(ny-1)*ngx
        dY=positions(2)-(ny-1._q)/ngy
        CALL min_imageX(dY)
        DO nx=1,ngx,DOWNSAMPLE
          indx=indxY+nx
          dX=positions(1)-(nx-1._q)/ngx
          CALL min_imageX(dX) 
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            DO t2=-translations(2),translations(2)
              dY_=dY+t2*1._q
              DO t1=-translations(1),translations(1)
                dX_=dX+t1*1._q
                diff=(/dX_,dY_,dZ_/)
                diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))    
                r=SUM(diff**2)**0.5
               !!! IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
                IF (r>0._q) THEN
                  CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)

                  !c contribut only if charge is greater than rholimit
                  BTval=ATval/(4*PI*r**2)
                  IF (BTval .GE. rholimit) THEN
                    RELCHG=RELCHG+ATval*(1.- part(indx))/r**2
                  ENDIF

                  IF (part_promol(indx)>0._q) THEN
                    BTval=ATval/part_promol(indx)/r**2
                    IF (BTval .GE. 0.5_q) THEN
                       !RELVOL=RELVOL+1._q
                       !RELVOL=RELVOL+r*ATval*part(indx)/BTval

                       RELVOL=RELVOL+part(indx)*part_promol(indx)*r**3
                       !write(*,*) 'kuku', BTval,RELVOL
                    ENDIF
                  ENDIF
                
                ENDIF
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      END DO
    END SUBROUTINE hirschfeld_dominant_charge_plane

    SUBROUTINE noninteracting_density(T_INFO,LATT_CUR,part,dimCHG2D,dimatCHG1D,&
    & atCHG1D, atGrid,D,positions,ngx,ngy,ngz,nz,translations,Rmax,NODE_ME,NCPU)
      USE prec
      USE lattice
      USE poscar
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: dimCHG2D,ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part(dimCHG2D)
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,Rmax
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D(dimatCHG1D) !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      INTEGER :: NODE_ME,NCPU  

      dZ=positions(3)-(nz-1._q)/ngz
      !write(*,*) 'dZ',dZ
      CALL min_imageX(dZ)
      DO ny=NODE_ME,ngy,NCPU
        indxY=(ny-1)*ngx
        dY=positions(2)-(ny-1._q)/ngy
        CALL min_imageX(dY)
        DO nx=1,ngx
          indx=indxY+nx
          dX=positions(1)-(nx-1._q)/ngx
          CALL min_imageX(dX) 
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            DO t2=-translations(2),translations(2)
              dY_=dY+t2*1._q
              DO t1=-translations(1),translations(1)
                dX_=dX+t1*1._q
                diff=(/dX_,dY_,dZ_/)
                diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))    
                r=SUM(diff**2)**0.5
                IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
                  CALL give_val0(r,atGrid,D,atCHG1D,ATval,dimatCHG1D)
                  !write(*,*) 'ghg',r,ATval/r**2/4/PI
                  part(indx)=part(indx)+ATval/r**2/4/PI  
                ENDIF
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      END DO
    END SUBROUTINE noninteracting_density

    SUBROUTINE noninteracting_density2(T_INFO,LATT_CUR,part1,part2,dimCHG2D,dimatCHG1D,&
    & atCHG1D_1,atCHG1D_2, atGrid,D,positions,ngx,ngy,ngz,nz,translations,Rmax,NODE_ME,NCPU,DOWNSAMPLE)
      USE prec
      USE lattice
      USE poscar
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: dimCHG2D,ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part1(dimCHG2D),part2(dimCHG2D)
      INTEGER :: translations(3)
      REAL(q) :: positions(3),positions_c(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,rr,Rmax
      REAL(q) :: diff1(3),diff2(3),diff3(3)
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      REAL(q) :: BTval
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D_1(dimatCHG1D), atCHG1D_2(dimatCHG1D)   !c 4*Pi*r^2*n(r)
      REAL(q) :: D
      INTEGER :: NODE_ME,NCPU,DOWNSAMPLE  
      !REAL(q),PARAMETER ::  rholimit=1e-4

      positions_c=MATMUL(positions,TRANSPOSE(LATT_CUR%A))

      !dZ=positions(3)-(nz-1._q)/ngz
      !write(*,*) 'dZ',dZ
     ! CALL min_imageX(dZ)
      dZ=(1._q-nz)/ngz

      DO indx=NODE_ME,dimCHG2D,NCPU
        ny=indx/ngx+1
        nx=MOD(indx,ngx)
        IF (nx==0) THEN 
          nx=ngx
          ny=ny-1
        END IF
        !dY=positions(2)-(ny-1._q)/ngy
        !CALL min_imageX(dY)
        !dX=positions(1)-(nx-1._q)/ngx
        !CALL min_imageX(dX) 
        dY=(1._q-ny)/ngy
        dX=(1._q-nx)/ngx

!       DO ny=DOWNSAMPLE*NODE_ME-(DOWNSAMPLE-1),ngy,DOWNSAMPLE*NCPU
!         indxY=(ny-1)*ngx
!         dY=positions(2)-(ny-1._q)/ngy
!         CALL min_imageX(dY)
!         DO nx=1,ngx,DOWNSAMPLE
!           indx=indxY+nx
!           dX=positions(1)-(nx-1._q)/ngx
!           CALL min_imageX(dX) 
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            diff3=dZ_*LATT_CUR%A(:,3)
            DO t2=-translations(2),translations(2)
              dY_=dY+t2*1._q
              diff2=dY_*LATT_CUR%A(:,2)
              DO t1=-translations(1),translations(1)
                dX_=dX+t1*1._q
                diff1=dX_*LATT_CUR%A(:,1)
                diff=positions_c+diff1+diff2+diff3                
                !diff=(/dX_,dY_,dZ_/)
                !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))    
                rr=SUM(diff**2)
                r=rr**0.5

                IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
                  CALL give_val0(r,atGrid,D,atCHG1D_1,ATval,dimatCHG1D)
                  part1(indx)=part1(indx)+ATval/rr

                  CALL give_val0(r,atGrid,D,atCHG1D_2,ATval,dimatCHG1D)
                  part2(indx)=part2(indx)+ATval/rr
                
                ENDIF
              ENDDO
            ENDDO
          ENDDO
!         ENDDO
      END DO
    END SUBROUTINE noninteracting_density2

    SUBROUTINE noninteracting_density3(T_INFO,LATT_CUR,part1,part2,dimatCHG1D,&
    & atCHG1D_1,atCHG1D_2, atGrid,D,positions,ngx,ngy,ngz,nx,ny,nz,translations,Rmax)
      USE prec
      USE lattice
      USE poscar
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part1,part2
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,rr,Rmax
      REAL(q) :: diff1(3),diff2(3),diff3(3)
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      REAL(q) :: BTval
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D_1(dimatCHG1D), atCHG1D_2(dimatCHG1D)   !c 4*Pi*r^2*n(r)
      REAL(q) :: D
   
      dZ=positions(3)+(1._q-nz)/ngz
      dY=positions(2)+(1._q-ny)/ngy
      dX=positions(1)+(1._q-nx)/ngx

      CALL min_imageX(dZ)
      CALL min_imageX(dY)
      CALL min_imageX(dX)


#ifdef MPI
      DO t2=-translations(2),translations(2)
        dY_=dY+t2*1._q
        diff2=dY_*LATT_CUR%A(:,2)
        DO t1=-translations(1),translations(1)
          dX_=dX+t1*1._q
          diff1=dX_*LATT_CUR%A(:,1)
          DO t3=-translations(3) ,translations(3) 
            dZ_=dZ+t3*1._q
            diff3=dZ_*LATT_CUR%A(:,3)
#else
      DO t3=-translations(3) ,translations(3) 
        dZ_=dZ+t3*1._q
        diff3=dZ_*LATT_CUR%A(:,3)
        DO t2=-translations(2),translations(2)
          dY_=dY+t2*1._q
          diff2=dY_*LATT_CUR%A(:,2)
          DO t1=-translations(1),translations(1)
            dX_=dX+t1*1._q
            diff1=dX_*LATT_CUR%A(:,1)
#endif
            diff=diff1+diff2+diff3   
            !diff=(/dX_,dY_,dZ_/)
            !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))         
   
            rr=SUM(diff**2)
            r=rr**0.5

            IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
              CALL give_val0(r,atGrid,D,atCHG1D_1,ATval,dimatCHG1D)
              part1=part1+ATval/rr

              CALL give_val0(r,atGrid,D,atCHG1D_2,ATval,dimatCHG1D)
              part2=part2+ATval/rr
                
            ENDIF
          ENDDO
        ENDDO
      ENDDO

    END SUBROUTINE noninteracting_density3

    SUBROUTINE noninteracting_density4(T_INFO,LATT_CUR,part1,part2,dimatCHG1D,&
    & atCHG1D_1,atCHG1D_2, atGrid,H,positions,ngx,ngy,ngz,nx,ny,nz,translations,RRmax,rdim,Rarray,Rhoatarray)
      USE prec
      USE lattice
      USE poscar
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER :: ngx,ngy,ngz,nx,ny,nz
      REAL(q) :: part1,part2
      INTEGER :: translations(3)
      REAL(q) :: positions(3)
      REAL(q) :: dZ,dY,dX,dX_,dY_,dZ_
      REAL(q) :: diff(3),r,rr,RRmax
      REAL(q) :: diff1(3),diff2(3),diff3(3)
      REAL(q) :: ATval  !c 4*Pi*r^2*n(r)
      REAL(q) :: BTval
      INTEGER :: t1,t2,t3,indxZ,indxY,indx
      INTEGER :: dimatCHG1D
      REAL(q) :: atGrid(dimatCHG1D)  !c logaritmic grid r(n+1)=D*r(n)
      REAL(q) :: atCHG1D_1(dimatCHG1D), atCHG1D_2(dimatCHG1D)   !c 4*Pi*r^2*n(r)
      REAL(q) :: H
      INTEGER :: rdim,NC
      REAL(q) :: Rarray(rdim),Rhoatarray(rdim)
   
      Rarray=0._q;Rhoatarray=0._q

      dZ=positions(3)+(1._q-nz)/ngz
      dY=positions(2)+(1._q-ny)/ngy
      dX=positions(1)+(1._q-nx)/ngx

      CALL min_imageX(dZ)
      CALL min_imageX(dY)
      CALL min_imageX(dX)

      NC=0
   
      DO t3=-translations(3) ,translations(3) 
        
        dZ_=dZ+t3*1._q
        diff3=dZ_*LATT_CUR%A(:,3)
        DO t2=-translations(2),translations(2)
          dY_=dY+t2*1._q
          diff2=dY_*LATT_CUR%A(:,2)
          DO t1=-translations(1),translations(1)
            NC=NC+1
            dX_=dX+t1*1._q
            diff1=dX_*LATT_CUR%A(:,1)

            diff=diff1+diff2+diff3   
            !diff=(/dX_,dY_,dZ_/)
            !diff=MATMUL(diff,TRANSPOSE(LATT_CUR%A))         
   
            rr=SUM(diff*diff)
            !r=rr**0.5
            Rarray(NC)=rr

            !IF ((r>0._q) .AND. (r .LT. Rmax)) THEN
            IF ((rr>0._q) .AND. (rr .LT. RRmax)) THEN
              r=rr**0.5
              !NC=NC+1
              !Rarray(NC)=r

!               CALL give_val0_new(r,atGrid,H,atCHG1D_1,ATval,dimatCHG1D)
!               Rhoatarray(NC)=ATval
!               part1=part1+ATval/rr
! 
!               CALL give_val0_new(r,atGrid,H,atCHG1D_2,ATval,dimatCHG1D)
!               part2=part2+ATval/rr
!               !Rhoatarray(NC)=ATval

              CALL give_val0_two(r,atGrid,H,atCHG1D_1,atCHG1D_2,ATval,BTval,dimatCHG1D)
              Rhoatarray(NC)=ATval
              part1=part1+ATval/rr
              part2=part2+BTval/rr
            ENDIF
          ENDDO
        ENDDO
      ENDDO

    END SUBROUTINE noninteracting_density4

   SUBROUTINE avol_rad(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: atch_(atchDIM) !4*PI*n(r)*r^5
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d

      dr=(d-1._q)/d
      atch_=atch*r**3
      avol=atch_(1)/2*r(1)*dr
      DO i=2,atchDIM
        avol=avol+(atch_(i)+atch_(i-1))/2*r(i)*dr
      ENDDO
      
    END SUBROUTINE avol_rad

   SUBROUTINE avol_rad_limit(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: atch_(atchDIM) !4*PI*n(r)*r^5
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d
      !REAL(q),PARAMETER :: rholimit=1e-4 
      REAL(q) :: dummy

      dr=(d-1._q)/d
      atch_=atch*r*r*r
      avol=atch_(1)/2*r(1)*dr
      DO i=2,atchDIM
        dummy=atch(i)/(4*PI*r(i)*r(i))
        IF (r(i)<0.5) dummy=1.
        IF (dummy>rholimit) THEN
          avol=avol+(atch_(i)+atch_(i-1))/2*r(i)*dr
        ELSE
          EXIT
        ENDIF
      ENDDO    
    END SUBROUTINE avol_rad_limit

    SUBROUTINE avol_rad_limit_stephan(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: atch_(atchDIM) !4*PI*n(r)*r^5
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d
      !REAL(q),PARAMETER :: rholimit=1e-4 
      REAL(q) :: dummy

      dr=(d-1._q)/d
      atch_=atch*r**3
      avol=atch_(1)/2*r(1)*dr
      DO i=2,atchDIM
        dummy=atch(i)/4/PI/r(i)**2
        IF (r(i)<0.5) dummy=1.
        !IF (dummy>rholimit) THEN
        IF (dummy>rholow) THEN
          avol=avol+(atch_(i)+atch_(i-1))/2*r(i)*dr
        ELSE
          EXIT
        ENDIF
      ENDDO
    END SUBROUTINE avol_rad_limit_stephan

   SUBROUTINE avol_rad_dominant(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: atch_(atchDIM) !4*PI*n(r)*r^5
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d
      !REAL(q),PARAMETER :: rholimit=1e-4 
      REAL(q) :: dummy
      REAL(q) :: ratom

      dr=(d-1._q)/d
      atch_=atch*r**3
      avol=atch_(1)/2*r(1)*dr
      ratom=r(1)
      DO i=2,atchDIM
        dummy=atch(i)/4/PI/r(i)**2
        IF (r(i)<0.5) dummy=1.
        IF (dummy>rholimit) THEN
          ratom=r(i)
        ELSE
          EXIT
        ENDIF
      ENDDO    

      avol=4./3.*PI*ratom**3
    END SUBROUTINE avol_rad_dominant

   SUBROUTINE rho_set_zero(atchDIM,atch,r)
   !c set rho(i:)=0 if rho(i)<rholimit
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2    
      REAL(q) :: r(atchDIM)
      REAL(q) :: dummy
 
      DO i=2,atchDIM
        dummy=atch(i)/(4*PI*r(i)*r(i))
        IF (dummy<rholimit) THEN
          atch(i:)=0._q
          EXIT
        ENDIF
      ENDDO    
    END SUBROUTINE rho_set_zero

   SUBROUTINE achg_rad(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d

      dr=(d-1._q)/d
      avol=atch(1)/2*r(1)*dr
      DO i=2,atchDIM
        avol=avol+(atch(i)+atch(i-1))/2*r(i)*dr
      ENDDO
      
    END SUBROUTINE achg_rad

    SUBROUTINE achg_rad_limit(atchDIM,atch,r,d,avol)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER :: atchDIM,i
      REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
      REAL(q) :: r(atchDIM)
      REAL(q) :: avol,dr,d
      !REAL(q),PARAMETER :: rholimit=1e-4 
      REAL(q) :: dummy

      dr=(d-1._q)/d
      avol=atch(1)/2*r(1)*dr
      DO i=2,atchDIM
        dummy=atch(i)/(4*PI*r(i)*r(i))
        IF (r(i)<0.5) dummy=1.
        IF (dummy>rholimit) THEN
          avol=avol+(atch(i)+atch(i-1))/2*r(i)*dr
        ENDIF
      ENDDO
      
    END SUBROUTINE achg_rad_limit


!tb test beg
    SUBROUTINE weight_charge(GRIDC, IO,DYN,T_INFO, INFO,LATT_CUR,P,  RELVOL,NALLOC,CWORK1,CWORK2,REFSTATE)
      USE prec
      USE mpimy
      USE mgrid
      USE AEDENS 
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us    
      USE rhfatm 
      USE constant
      IMPLICIT NONE

      TYPE (grid_3d) GRIDC
      TYPE (type_info)   T_INFO
      TYPE (info_struct) INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar),TARGET ::   P(T_INFO%NTYP) 
      TYPE(dynamics) :: DYN
      TYPE (in_struct)   IO
      !INTEGER :: CWORK_DIM
!       COMPLEX(q)  ::  CWORK1(CWORK_DIM),CWORK2(CWORK_DIM)
#ifdef MPI
      RGRID :: CWORK1(GRIDC%RL%NROW,GRIDC%RL%NCOL), CWORK2(GRIDC%RL%NROW,GRIDC%RL%NCOL)
      REAL(q),ALLOCATABLE :: CWORK3(:,:),  CWORK4(:,:)
#else
      RGRID :: CWORK1(DIMREAL(GRIDC%MPLWV)), CWORK2(DIMREAL(GRIDC%MPLWV)) !(GRIDC%NGX*GRIDC%NGY,NZ),CWORK2(GRIDC%NGX*GRIDC%NGY,NZ)
      REAL(q),ALLOCATABLE :: CWORK3(:),  CWORK4(:)
#endif
!       RGRID,ALLOCATABLE :: CWORK3(:),  CWORK4(:)
     !REAL(q),ALLOCATABLE :: CWORK3(:,:),  CWORK4(:,:)
      REAL(q),ALLOCATABLE::  WORK1(:),WORK2(:), WORK3(:),WORK4(:)
      INTEGER NALLOC,NZ
      INTEGER ISTAT
      REAL(q),PARAMETER :: rmax=7. !c cutoff to the radial grid
      !REAL(q),PARAMETER :: rmax=15. !c cutoff to the radial grid
      INTEGER :: translations(3)
      REAL(q) :: RELVOL(T_INFO%NIONS),RCHG(T_INFO%NIONS)
      REAL(q),ALLOCATABLE,SAVE :: ATVOL(:),ATCHG(:)
      INTEGER :: i,j,indx
      INTEGER NODE_ME,IONODE,NCPU
      LOGICAL,SAVE :: LFIRST=.TRUE.
      REAL(q) :: dV
      TYPE(atomic),ALLOCATABLE,SAVE :: WDF(:),WDF_0(:)
      TYPE (potcar), POINTER:: PP
      REAL(q) :: chrg
      REAL(q) :: rdummy,y0
      INTEGER :: REFSTATE(T_INFO%NTYP)
      LOGICAL,SAVE :: LIONIC=.FALSE.
      LOGICAL :: LIONREF=.TRUE.  !denominator in V^eff/V^free
      LOGICAL :: LFROZENORBITAL=.FALSE.
      REAL(q) :: chg1,chg2
      LOGICAL,SAVE :: LVDW_RELVOLONE=.FALSE.
      LOGICAL :: LOPEN
      CHARACTER*1 :: CHARAC
      INTEGER :: IERR,N,IDUM
      COMPLEX(q) :: CDUM
      REAL(q) :: RDUM
      REAL(q) :: positions_c(3)
      INTEGER NC,NX,NY
      REAL(q),SAVE :: BCK_CHARGE

      
     !c user may chose to define his own vdW parameters and avoid 
     !c the rescaling by RELVOL - this is achieved by setting 
     !c LVDW_RELVOLONE=.TRUE.
     IF (LFIRST) THEN
       LOPEN=.FALSE.
       OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

       CALL RDATAB(LOPEN,INCAR,IO%IU5,'LVDW_RELVOLONE','=','#',';','L', &
                  IDUM,RDUM,CDUM,LVDW_RELVOLONE,CHARAC,N,1,IERR)
       IF (IERR==3) LVDW_RELVOLONE=.FALSE. 
       IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         LVDW_RELVOLONE=.FALSE.  
         IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''LVDW_RELVOLONE'' from file INCAR'
       ENDIF
       CLOSE(IO%IU5)
     ENDIF

     IF (LVDW_RELVOLONE) THEN
       RELVOL=1.
       RETURN
     ENDIF

      
2230 FORMAT(/,"  Hirshfeld charges:",&
            /,"            ion         q(e)     ",&
            /,"   ------------------------------")
2236 FORMAT("   ",I4,5X,A2,5X,F10.3)


      NODE_ME=1
      NCPU=1
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      NCPU=GRIDC%COMM%NCPU
#endif

      RELVOL=0._q
      RCHG=0._q

      !c increment for the volume integration
      dV=LATT_CUR%OMEGA/GRIDC%NGX/GRIDC%NGY/GRIDC%NGZ/(4*PI) 

      !c compute atomic volumes for non-interacting system
      IF (LFIRST) THEN
        !c determine background charge used to stabilize ions
        BCK_CHARGE=INFO%NELECT
        DO i=1,T_INFO%NTYP
          BCK_CHARGE =BCK_CHARGE-T_INFO%NITYP(i)*P(i)%ZVALF
        ENDDO


        !c detect if reference state other than that of neutral atom 
        !c is used
        IF (SUM(REFSTATE**2)>0) THEN
          LIONIC=.TRUE.
         ! write(*,*) 'ionic state detected!!!',REFSTATE
        ENDIF

        LFIRST=.FALSE.
        IF (LIONIC) ALLOCATE(WDF_0(T_INFO%NTYP))
        ALLOCATE(WDF(T_INFO%NTYP))
        ALLOCATE(ATVOL(T_INFO%NTYP),ATCHG(T_INFO%NTYP))

        DO i=1,T_INFO%NTYP
          PP=>P(i)
          CALL DFATOM2(PP,WDF(i),IO,REFSTATE(i),LFROZENORBITAL)
          CALL RHO_SET_ZERO(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R)
!           DO j=1,WDF(i)%NS
!             IF (IO%IU6>0)  write(*,*) 'occ:',WDF(i)%OCC(j),WDF(i)%E(j)
!           ENDDO
!           IF (IO%IU6>0) THEN
!             DO j=1,WDF(i)%R%NMAX
!               write(*,*) "rion", WDF(i)%R%R(j),WDF(i)%RHO(j)
!             ENDDO
!           END IF
          !!!CALL avol_rad(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATVOL(i))
          CALL avol_rad_limit(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATVOL(i))
          !!!CALL achg_rad(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATCHG(i))
          CALL achg_rad_limit(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATCHG(i))
          !c if needed, compute radial charge distribution for neutral atom
          IF (LIONIC) THEN
            CALL DFATOM2(PP,WDF_0(i),IO,0,.FALSE.)
            IF (.NOT. LIONREF) THEN
              CALL avol_rad(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATVOL(i))
            ENDIF
          ENDIF
        ENDDO
      ENDIF

      !c determine the number of images of simulation cell
      !c in each direction needed for a given 
      !c cutoff radius
       translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
       translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
       translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)

       IF (IO%IU6>0) write(*,*) 'translations:',translations(1),translations(2),translations(3)

       RELVOL=0._q;RCHG=0._q

#ifdef MPI 
      !c fix the promolecular density if needed   
      IF (LIONIC) THEN
!         ALLOCATE(CWORK3(DIMREAL(GRIDC%MPLWV)),CWORK4(DIMREAL(GRIDC%MPLWV)))
        ALLOCATE(CWORK3(GRIDC%RL%NROW,GRIDC%RL%NCOL),CWORK4(GRIDC%RL%NROW,GRIDC%RL%NCOL))
        CWORK3=0._q; CWORK4=0._q
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          PP=>P(indx) 
          !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
          DO NC=1,GRIDC%RL%NCOL
            NX= GRIDC%RL%I2(NC)
            NY= GRIDC%RL%I3(NC)
            DO NZ=1,GRIDC%NGZ 
              CALL noninteracting_density3(T_INFO,LATT_CUR,CWORK3(NZ,NC),CWORK4(NZ,NC),WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF_0(indx)%RHO,WDF(indx)%R%R,&
              & WDF_0(indx)%R%D,DYN%POSION(:,i) ,&
              & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax)
            ENDDO
          ENDDO
        ENDDO
        DO NC=1,GRIDC%RL%NCOL
          NX= GRIDC%RL%I2(NC)
          NY= GRIDC%RL%I3(NC)
          DO NZ=1,GRIDC%NGZ 
            !IF (REAL(CWORK1(NZ,NC))>1e-8 .AND. REAL(CWORK4(NZ,NC))>1e-8) THEN
            IF (ABS(CWORK4(NZ,NC))>1e-8) THEN
              CWORK1(NZ,NC)=CWORK1(NZ,NC)*CWORK3(NZ,NC)/CWORK4(NZ,NC)            
            ENDIF
          ENDDO
        ENDDO

        DEALLOCATE(CWORK3,CWORK4)
      ENDIF

      DO i=1,T_INFO%NIONS
        indx=T_INFO%ITYP(i)
        PP=>P(indx) 
        !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
        DO NC=1,GRIDC%RL%NCOL
          NX= GRIDC%RL%I2(NC)
          NY= GRIDC%RL%I3(NC)
          DO NZ=1,GRIDC%NGZ 
            IF (REAL(CWORK1(NZ,NC))>1e-8 .AND. REAL(CWORK2(NZ,NC))>1e-8) THEN
            !IF (ABS(CWORK1(NZ,NC))>1e-8) THEN
              RDUM=REAL(CWORK2(NZ,NC)/CWORK1(NZ,NC)) 
            !  RDUM=ABS(CWORK2(NZ,NC)/CWORK1(NZ,NC))   
            !IF ((CWORK1(NZ,NC))>1e-8) THEN 
            !  RDUM=(CWORK2(NZ,NC)/CWORK1(NZ,NC))      
              CALL rel_vol_plane3(T_INFO,LATT_CUR,RDUM,WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF(indx)%R%R,&
              & PP%R%H, DYN%POSION(:,i),&
              & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL(i),RCHG(i))
            ENDIF
          ENDDO
        ENDDO
      ENDDO

      CALL M_sum_d(GRIDC%COMM, RELVOL,T_INFO%NIONS )
      CALL M_sum_d(GRIDC%COMM, RCHG,T_INFO%NIONS )
     
#else
      !c fix the promolecular density if needed   
      IF (LIONIC) THEN
        ALLOCATE(CWORK3(DIMREAL(GRIDC%MPLWV)),CWORK4(DIMREAL(GRIDC%MPLWV)))
        CWORK3=0._q; CWORK4=0._q
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          PP=>P(indx) 
          !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A))
 
          NC=0
          DO nz=1,GRIDC%NGZ
            DO ny=1,GRIDC%NGY
              DO nx=1,GRIDC%NGX
                 NC=NC+1
                 CALL noninteracting_density3(T_INFO,LATT_CUR,CWORK3(NC),CWORK4(NC),WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF_0(indx)%RHO,WDF(indx)%R%R,&
                 & WDF_0(indx)%R%D,DYN%POSION(:,i),&
                 & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax)    
              ENDDO
            ENDDO
          ENDDO
        ENDDO
        NC=0
        DO nz=1,GRIDC%NGZ
          DO ny=1,GRIDC%NGY
            DO nx=1,GRIDC%NGX
              NC=NC+1
              IF (REAL(CWORK1(NC))>1e-8 .AND. REAL(CWORK4(NC))>1e-8) THEN
                CWORK1(NC)=CWORK1(NC)*CWORK3(NC)/CWORK4(NC)            
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        DEALLOCATE(CWORK3,CWORK4)
      ENDIF

       DO i=1,T_INFO%NIONS
        indx=T_INFO%ITYP(i)
        PP=>P(indx) 
        !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
  
        NC=0
        DO nz=1,GRIDC%NGZ
          DO ny=1,GRIDC%NGY
            DO nx=1,GRIDC%NGX
               NC=NC+1
               IF (REAL(CWORK1(NC))>1e-8 .AND. REAL(CWORK2(NC))>1e-8) THEN
                 RDUM=REAL(CWORK2(NC)/CWORK1(NC))            
                 CALL rel_vol_plane3(T_INFO,LATT_CUR,RDUM,WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF(indx)%R%R,&
                 & PP%R%H, DYN%POSION(:,i),&
                 & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL(i),RCHG(i))
               ENDIF
            ENDDO
          ENDDO
        ENDDO
      ENDDO
#endif

      RELVOL=RELVOL*dV
     

      DO i=1,T_INFO%NIONS
        RELVOL(i)=RELVOL(i)/ATVOL(T_INFO%ITYP(i))
        !IF (IO%IU6>0) write(*,*) "xdf",i,RELVOL(i),ATVOL(T_INFO%ITYP(i))
      ENDDO

      RCHG=RCHG*dV
      CALL neutralize_charge(T_INFO%NIONS,RCHG,BCK_CHARGE)

      !c add charge of reference state
      IF (LIONIC) THEN
        DO i=1,T_INFO%NIONS 
          indx=T_INFO%ITYP(i)            
          RCHG(i)= RCHG(i)+REFSTATE(indx)
        ENDDO
      ENDIF
  
      !IF (IO%IU6>0 .AND. IO%NWRITE==3) THEN
      IF (IO%IU6>0 ) THEN
        WRITE(IO%IU6,2230)
        DO i=1,T_INFO%NIONS 
          indx=T_INFO%ITYP(i)               
          WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,RCHG(i)
        ENDDO
      ENDIF  

      RETURN

   END SUBROUTINE weight_charge

    SUBROUTINE hirshfeld_iterative(GRIDC, IO,DYN,T_INFO,INFO, LATT_CUR,P,CWORK1,CWORK2,RELVOL,IVDW)
      USE ini
      USE mpimy
      USE mgrid
      USE AEDENS 
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us    
      USE rhfatm 
      USE constant
      IMPLICIT NONE

      TYPE (grid_3d) GRIDC
      TYPE (type_info)   T_INFO
      TYPE (info_struct) INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar),TARGET ::   P(T_INFO%NTYP) 
      TYPE(dynamics) :: DYN
      TYPE (in_struct)   IO
#ifdef MPI
      RGRID :: CWORK1(GRIDC%RL%NROW,GRIDC%RL%NCOL), CWORK2(GRIDC%RL%NROW,GRIDC%RL%NCOL)
     ! REAL(q),ALLOCATABLE :: CWORK3(:,:),  CWORK4(:,:)
#else
      RGRID :: CWORK1(DIMREAL(GRIDC%MPLWV)), CWORK2(DIMREAL(GRIDC%MPLWV)) !(GRIDC%NGX*GRIDC%NGY,NZ),CWORK2(GRIDC%NGX*GRIDC%NGY,NZ)
      !REAL(q),ALLOCATABLE :: CWORK3(:),  CWORK4(:)
#endif
      INTEGER :: NALLOC,NX,NY,NZ,NC
      REAL(q),PARAMETER :: rmax=7. !c cutoff for the radial grid
      !REAL(q),PARAMETER :: rmax=15.
      REAL(q), SAVE :: rrmax
      INTEGER :: translations(3)
      REAL(q),ALLOCATABLE,SAVE :: RELCHG(:)
      REAL(q) :: RELCHG_old(T_INFO%NIONS),dRELCHG(T_INFO%NIONS),dRELCHG_old(T_INFO%NIONS)
      REAL(q) :: RELVOL(T_INFO%NIONS)
      INTEGER :: i,j,k,indx,indx2,iter
      INTEGER NODE_ME,IONODE,NCPU
      REAL(q) :: dV
      TYPE(atomic),ALLOCATABLE,SAVE :: WDF_0(:)
      TYPE(atomic),ALLOCATABLE :: WDF(:),WDF_new(:),WDF_work(:)
      TYPE (potcar), POINTER:: PP
      REAL(q) :: chrg
      REAL(q) :: rdummy
      INTEGER :: defstates(2,T_INFO%NIONS),  defstates_new(2,T_INFO%NIONS) 
      LOGICAL :: oxstates(T_INFO%NTYP,17),oxstates_old(T_INFO%NTYP,17)
      INTEGER :: noxstates(T_INFO%NTYP),noxstates_total
      INTEGER :: map_ox_states(T_INFO%NIONS)
      REAL(q) :: test_ATCHG
      INTEGER,PARAMETER :: maxiter=150 !c max. number of iterations
      REAL :: step
      LOGICAL :: LSTOP,LRESET
      LOGICAL :: LFIRST=.TRUE.
      !REAL(q),parameter :: iterToleration=5.e-5
      REAL(q),SAVE :: iterToleration=5.e-5
      INTEGER,ALLOCATABLE,SAVE :: anionic_capacity(:)
      REAL(q),ALLOCATABLE,SAVE :: ATVOL(:),ATCHG(:)
      REAL(q) :: chg1,chg2,chg3
      REAL(q),PARAMETER :: MIX=1.0 
      INTEGER :: DOWNSAMPLE=1
      REAL(q) :: ATVOL_dum
      LOGICAL :: LTESTHIRSH=.FALSE. !c for test purposes only - mimics standard Hirshfeld partitioning
      LOGICAL :: LFROZENORBITAL=.FALSE.
      REAL(q) :: positions_c(3,T_INFO%NIONS)
      REAL(q) :: RDUM,RDUM1,RDUM2, RDUM3,  RDUM4
      REAL(q),ALLOCATABLE :: Rarray(:,:),RhoAtaarray(:,:)
      INTEGER :: totTrans
      REAL(q),SAVE :: BCK_CHARGE
      REAL(q) :: SQerror,SQerror_old
      LOGICAL :: LOPEN
      CHARACTER*1 :: CHARAC
      INTEGER :: IERR,N,IDUM
      COMPLEX(q) :: CDUM
      LOGICAL :: LDUM
      INTEGER :: IVDW

      
      NODE_ME=1
      NCPU=1
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      NCPU=GRIDC%COMM%NCPU
#endif

      LSTOP=.FALSE.
      LRESET=.FALSE.

      IF (IVDW==22) LFROZENORBITAL=.TRUE.
     
2230 FORMAT(/,"  Hirshfeld charges:",&
            /,"            ion         q(e)     ",&
            /,"   ------------------------------")
2231 FORMAT(/,"  Hirshfeld-I charges:",&
            /,"            ion         q(e)        dq(e)    ",&
            /,"   ------------------------------------------")
2232 FORMAT(/,"   Hirshfeld-I converged in",I4,' cycles. Convergence criterion |dq|<',E10.3E2,' e')
2236 FORMAT("   ",I4,5X,A2,5X,F10.3)
2237 FORMAT("   ",I4,5X,A2,5X,F10.3,5X,E10.3E2)
    
      !c increment for the volume integration
      dV=LATT_CUR%OMEGA/GRIDC%NGX/GRIDC%NGY/GRIDC%NGZ/(4*PI) 

      positions_c=MATMUL(LATT_CUR%A,DYN%POSION) 


      IF (LFIRST) THEN
        LOPEN=.FALSE.
        OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'HITOLER','=','#',';','F', &
                  IDUM,iterToleration,CDUM,LDUM,CHARAC,N,1,IERR)
        IF (IERR==3)  iterToleration=5.e-5
        IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
          iterToleration=5.e-5
          IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''HITOLER'' from file INCAR'
        ENDIF
        IF (iterToleration<1e-8 .OR. iterToleration>1e-1) THEN
          IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,*)'hirshfeld_iterative Error: invalid value for HITOLER',iterToleration
            WRITE(IO%IU0,*)'the permitted values are 1e-8<HITOLER<1e-1'
          ENDIF
          STOP
        ENDIF
        CLOSE(UNIT=IO%IU5)

        rrmax=rmax*rmax

        ALLOCATE(WDF_0(T_INFO%NTYP))
        ALLOCATE(anionic_capacity(T_INFO%NTYP))
        ALLOCATE(ATVOL(T_INFO%NTYP),ATCHG(T_INFO%NTYP))
        ALLOCATE(RELCHG(T_INFO%NIONS))
        RELCHG=0._q

        !c determine background charge used to stabilize ions
        BCK_CHARGE=INFO%NELECT
        DO i=1,T_INFO%NTYP
          BCK_CHARGE =BCK_CHARGE-T_INFO%NITYP(i)*P(i)%ZVALF
        ENDDO

        !c compute radial charge densities and volumes for non-interacting atoms
        DO i=1,T_INFO%NTYP
          PP=>P(i)
          CALL DFATOM2(PP,WDF_0(i),IO,0,.FALSE.)
          CALL RHO_SET_ZERO(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R)
          CALL max_anion(WDF_0(i)%NS,WDF_0(i)%OCC,WDF_0(i)%L,anionic_capacity(i))
          CALL avol_rad_limit(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATVOL(i))
         ! CALL SIMPI(WDF_0(i)%R,WDF_0(i)%RHO(:)*WDF_0(i)%R%R(:)**3,ATVOL(i))
          CALL achg_rad_limit(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATCHG(i))
!           CALL SIMPI(WDF_0(i)%R,WDF_0(i)%RHO(:),ATCHG(i))
          !IF (IO%IU6>0) write(*,*) "nn",i,0,ATVOL(i)
        ENDDO
      ENDIF 

      !IF (IO%IU6>0)  write(*,*) "anionic_capacity",anionic_capacity

     !c determine the number of images of simulation cell
     !c in each direction needed for a given 
     !c cutoff radius
      translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
      translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
      translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)

      IF (IO%IU6>0) write(*,*) 'Hirshfeld-I algorithm:'
      IF (IO%IU6>0) write(*,*) 'translations:',translations(1),translations(2),translations(3)

      totTrans=(2*translations(1)+1)*(2*translations(2)+1)*(2*translations(3)+1)
      ALLOCATE(Rarray(totTrans,T_INFO%NIONS),RhoAtaarray(totTrans,T_INFO%NIONS))

      !c first, compute charges using classical Hirschfeld scheme  
      IF (LFIRST) THEN  
        dRELCHG=0._q
        RELVOL=0._q 

#ifdef MPI
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          !PP=>P(indx) 
          !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
          DO NC=1,GRIDC%RL%NCOL
            NX= GRIDC%RL%I2(NC)
            NY= GRIDC%RL%I3(NC)
            DO NZ=1,GRIDC%NGZ 
              IF (REAL(CWORK1(NZ,NC))>1e-8 .AND. REAL(CWORK2(NZ,NC))>1e-8) THEN
                RDUM=REAL(CWORK2(NZ,NC)/CWORK1(NZ,NC))            
                CALL rel_vol_plane3(T_INFO,LATT_CUR,RDUM,WDF_0(indx)%R%NMAX,WDF_0(indx)%RHO,WDF_0(indx)%R%R,&
                & WDF_0(indx)%R%H, DYN%POSION(:,i),&
                & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL(i),dRELCHG(i))
              ENDIF
            ENDDO
          ENDDO
        ENDDO

        CALL M_sum_d(GRIDC%COMM, RELVOL,T_INFO%NIONS )
        CALL M_sum_d(GRIDC%COMM, dRELCHG,T_INFO%NIONS )
     
#else  
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          !PP=>P(indx) 
          !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
  
          NC=0
          DO nz=1,GRIDC%NGZ
            DO ny=1,GRIDC%NGY
              DO nx=1,GRIDC%NGX
                 NC=NC+1
                 IF (REAL(CWORK1(NC))>1e-8 .AND. REAL(CWORK2(NC))>1e-8) THEN
                   RDUM=REAL(CWORK2(NC)/CWORK1(NC))            
                   CALL rel_vol_plane3(T_INFO,LATT_CUR,RDUM,WDF_0(indx)%R%NMAX,WDF_0(indx)%RHO,WDF_0(indx)%R%R,&
                   & WDF_0(indx)%R%H, DYN%POSION(:,i),&
                   & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL(i),dRELCHG(i))
                 ENDIF
              ENDDO
            ENDDO
          ENDDO
        ENDDO
#endif

        dRELCHG=dRELCHG*dV
        !dRELCHG_old=dRELCHG

        CALL neutralize_charge(T_INFO%NIONS,dRELCHG,BCK_CHARGE)

        !RELCHG=RELCHG_old

        IF (IO%IU6>0) THEN
          WRITE(IO%IU6,2230)
          DO i=1,T_INFO%NIONS 
            indx=T_INFO%ITYP(i)         
            WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,dRELCHG(i)
          ENDDO
        ENDIF  

        !c nasty trick made in attempt to make the first step in H-I faster
        !c using the fact that 
        !c Hirshfeld charges are typically 3-5 times smaller than the H-I charges:
        step=3.
        !SQerror_old=SQRT(sum(dRELCHG**2)/T_INFO%NIONS)
        !IF (IO%IU6>0)  write(*,*) 'sqerror:', step,SQerror_old
  
        IF (ABS(BCK_CHARGE) .LE. 1.e-5) THEN 
          RELCHG=step*dRELCHG
        ELSE
          RELCHG=1.*dRELCHG
        ENDIF
        !step=1.

      ENDIF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     

      !c find lower and upper integers for partial charges
      CALL find_int_limits(T_INFO,RELCHG,defstates,anionic_capacity)

      !c identify needed oxidation states  
      CALL finx_ox_states(T_INFO,defstates,oxstates,noxstates,noxstates_total)
      
      ALLOCATE(WDF(noxstates_total))
      
      map_ox_states=0
      indx=0
      DO i=1,T_INFO%NTYP
        PP=>P(i)
        DO j=-8,8
          IF (oxstates(i,j+9)) THEN
            indx=indx+1
            IF (j==0) THEN
              WDF(indx)=WDF_0(i)
            ELSE
              IF (IVDW==25) THEN
                LFROZENORBITAL=.FALSE.
                IF (j<0) LFROZENORBITAL=.TRUE.
              ENDIF

              IF (IVDW==26) THEN
                LFROZENORBITAL=.TRUE.
                IF ((j>=0) .OR. ((j==-1) .AND. (anionic_capacity(i)==1))) LFROZENORBITAL=.FALSE.
              ENDIF

              !IF (IO%IU6>0) write(*,*) "ox",i,j
              CALL DFATOM2(PP,WDF(indx),IO,j,LFROZENORBITAL)
              !CALL RHO_SET_ZERO(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R)
              CALL RHO_SET_ZERO(WDF_0(i)%R%NMAX,WDF(indx)%RHO,WDF_0(i)%R%R)
              !CALL SIMPI(WDF(indx)%R,WDF(indx)%RHO*WDF(indx)%R%R(:)**3,RDUM3)
              !IF (IO%IU6>0) write(*,*) "nn",i,j,RDUM3
            ENDIF
            DO k=1,T_INFO%NIONS
              IF (T_INFO%ITYP(k)==i .AND. defstates(1,k)==j) map_ox_states(k)=indx
            ENDDO
          ENDIF
        ENDDO
      ENDDO

      ALLOCATE(WDF_work(T_INFO%NIONS))
      DO i=1,T_INFO%NIONS    
        indx2=map_ox_states(i)
        ALLOCATE(WDF_work(i)%RHO(WDF(indx2)%R%NMAX))     
        IF (LTESTHIRSH)   RELCHG(i)=0._q 
        WDF_work(i)%RHO=WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i))      
     ENDDO

     !CALL STOP_TIMING("GG",IO%IU6,'HI-init')

     !CALL START_TIMING("GG")
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     RELCHG_old=RELCHG
     step=1.
     loop_interation: DO iter=1,maxiter

       chg1=0._q;chg2=0._q;chg3=0._q
       dRELCHG=0._q
       RELVOL=0._q
 
#ifdef MPI 
       !IF (iter==1) CALL START_TIMING("H")
            
       DO NC=1,GRIDC%RL%NCOL
         NX= GRIDC%RL%I2(NC)
         NY= GRIDC%RL%I3(NC)      
         DO NZ=1,GRIDC%NGZ 
           RDUM1=REAL(CWORK1(NZ,NC))
           IF (RDUM1>1e-8) THEN 
             RDUM2=REAL(CWORK2(NZ,NC))
             RDUM3=0._q;RDUM4=0._q
             Rarray=0._q
             RhoAtaarray=0._q

             DO i=1,T_INFO%NIONS
               indx=T_INFO%ITYP(i)
               indx2=map_ox_states(i)   
               CALL noninteracting_density4(T_INFO,LATT_CUR,RDUM3,RDUM4,WDF(indx2)%R%NMAX,WDF_work(i)%RHO,WDF_0(indx)%RHO,WDF(indx2)%R%R,&
               & WDF_0(indx)%R%H,DYN%POSION(:,i) ,&
               & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rrmax,totTrans,Rarray(:,i),RhoAtaarray(:,i))          
             ENDDO

             !IF (RDUM1>1e-8 .AND. RDUM4 >1e-8) THEN
             !RDUM3=RDUM4/(RDUM1*RDUM3)
             !IF (RDUM4 >1e-8 .AND. RDUM3>1e-8) THEN
             !  RDUM=RDUM1*RDUM3
!              IF (RDUM4 >1e-8) THEN
!                RDUM3=RDUM1*RDUM3/RDUM4  
!                IF (RDUM3>1e-8) THEN
!                  DO i=1,T_INFO%NIONS                  
!                    RDUM=RDUM2/RDUM3    
             RDUM3=RDUM1*RDUM3
             IF (RDUM4 >1e-8 .AND. RDUM3>1e-8) THEN
               !RDUM3=RDUM4/(RDUM1*RDUM3) 
               RDUM3=RDUM4/RDUM3
               RDUM=RDUM2*RDUM3           
               DO i=1,T_INFO%NIONS                  
                 !RDUM=RDUM2*RDUM3  
                 CALL rel_vol_plane4(RDUM,rrmax,totTrans,Rarray(:,i),RhoAtaarray(:,i),RELVOL(i),dRELCHG(i))   
               ENDDO
             ENDIF
           ENDIF 
         ENDDO
       ENDDO

       !IF (iter==1) CALL STOP_TIMING("H",IO%IU6,'HI-integration')
       !IF (iter==1) CALL START_TIMING("H")
       CALL M_sum_d(GRIDC%COMM, dRELCHG,T_INFO%NIONS )
       !IF (iter==1) CALL STOP_TIMING("H",IO%IU6,'HI-MPI')
    
#else
      !c fix the promolecular density if needed      
     
       NC=0
       DO nz=1,GRIDC%NGZ
         DO ny=1,GRIDC%NGY
           DO nx=1,GRIDC%NGX
             NC=NC+1    
             RDUM3=0._q;RDUM4=0._q
             Rarray=0._q
             RhoAtaarray=0._q
             DO i=1,T_INFO%NIONS
               indx=T_INFO%ITYP(i)
               indx2=map_ox_states(i)   
               CALL noninteracting_density4(T_INFO,LATT_CUR,RDUM3,RDUM4,WDF(indx2)%R%NMAX,WDF_work(i)%RHO,WDF_0(indx)%RHO,WDF(indx2)%R%R,&
               & WDF_0(indx)%R%H,DYN%POSION(:,i),&
               & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,totTrans,Rarray(:,i),RhoAtaarray(:,i))
             ENDDO

             IF (REAL(CWORK1(NC))>1e-8 .AND. RDUM4>1e-8) THEN
               RDUM3=CWORK1(NC)*RDUM3/RDUM4 
               !RDUM3=(CWORK1(NC)-BCK_CHARGE)*RDUM3/RDUM4 
               IF (RDUM3>1e-8) THEN
                 DO i=1,T_INFO%NIONS   
                   RDUM=REAL(CWORK2(NC))/RDUM3                   
                   CALL rel_vol_plane4(RDUM,rmax,totTrans,Rarray(:,i),RhoAtaarray(:,i),RELVOL(i),dRELCHG(i))     
                 ENDDO
               ENDIF           
             ENDIF
           ENDDO
         ENDDO
       ENDDO
       
#endif
         
       
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
       dRELCHG=dRELCHG*dV
       IF (iter .eq. 1) dRELCHG_old=dRELCHG

       CALL neutralize_charge(T_INFO%NIONS,dRELCHG,0._q)
       

       IF (LTESTHIRSH)   dRELCHG=0._q

       IF (DOWNSAMPLE .EQ. 1) THEN
         IF (MAXVAL(ABS(dRELCHG)) .LE. iterToleration) LSTOP=.TRUE.
       ELSE
         IF (MAXVAL(ABS(dRELCHG)) .LE. 10*iterToleration) DOWNSAMPLE=1
       ENDIF

       RELCHG_old=RELCHG
       dRELCHG_old=dRELCHG
       RELCHG=step*dRELCHG_old+RELCHG_old

            
       !IF (LTESTHIRSH)   RELCHG=0._q
       IF (LTESTHIRSH)   dRELCHG=0._q

       IF (IO%IU6>0) THEN
         write(*,*) 'H-I iteration:',iter
         DO i=1,T_INFO%NIONS 
           write(*,*) "atom:",i,RELCHG(i),(RELCHG(i)-RELCHG_old(i))/step
           !write(*,*) "atom:",i,RELCHG(i),dRELCHG(i)
         ENDDO
       ENDIF

       !c check if we need new limits for ox. states
       CALL find_int_limits(T_INFO,RELCHG,defstates_new,anionic_capacity)
       CALL check_ox_consistency(T_INFO%NIONS,defstates,defstates_new,LRESET)

       IF (LRESET) THEN
         !IF (IO%IU6>0) write(*,*) 'WARNING: new ox. states have to be generated!!!'
         defstates=defstates_new
         oxstates_old=oxstates
         CALL finx_ox_states(T_INFO,defstates,oxstates,noxstates,noxstates_total)
 
         ALLOCATE(WDF_new(noxstates_total))

         map_ox_states=0
         indx=0
         indx2=0
         DO i=1,T_INFO%NTYP
           PP=>P(i)
           DO j=-8,8
             IF (oxstates_old(i,j+9)) indx2=indx2+1

             IF (oxstates(i,j+9)) THEN
               indx=indx+1
               IF (oxstates_old(i,j+9)) THEN
                 WDF_new(indx)=WDF(indx2)
               ELSE
                 !IF (IO%IU6>0) write(*,*) "ox",i,j          
                 CALL DFATOM2(PP,WDF_new(indx),IO,j,LFROZENORBITAL)
                 CALL RHO_SET_ZERO(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R)
                 !IF (j>0) THEN
                 !  CALL DFATOM2(PP,WDF_new(indx),IO,j,.TRUE.)
                 !ELSE
                 !  CALL DFATOM2(PP,WDF_new(indx),IO,j,LFROZENORBITAL)
                 !END IF

                 !CALL SIMPI(WDF_new(indx)%R,WDF_new(indx)%RHO*WDF_new(indx)%R%R(:)**3,RDUM3)
                 !IF (IO%IU6>0) write(*,*) "nn",i,j,RDUM3
               ENDIF
               DO k=1,T_INFO%NIONS
                 IF (T_INFO%ITYP(k)==i .AND. defstates(1,k)==j) map_ox_states(k)=indx
               ENDDO
             ENDIF
           ENDDO
         ENDDO
         DEALLOCATE(WDF)
         ALLOCATE(WDF(noxstates_total))
         WDF=WDF_new
         DEALLOCATE(WDF_new)

         !!DO i=1,T_INFO%NIONS    
         !!  indx2=map_ox_states(i)   
         !!  WDF_work(i)%RHO=(1._q-MIX)*WDF_work(i)%RHO+MIX*(WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i)))           
         !!ENDDO

         LRESET=.FALSE.
       ENDIF 

      
       DO i=1,T_INFO%NIONS     
         indx2=map_ox_states(i)   
         WDF_work(i)%RHO=(1._q-MIX)*WDF_work(i)%RHO+MIX*(WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i)))   
       ENDDO 
 
 
       IF (LSTOP) THEN
         EXIT
       ELSE IF (iter .GE. maxiter) THEN
         IF (IO%IU6>0) write(*,*) "Error(hirshfeld_iterative): Hirshfeld-I did not converge in",maxiter,"steps"
         STOP                                                                                                              
       ENDIF

       !CALL STOP_TIMING("GG",IO%IU6,'HI-alliter')maxiter
     ENDDO loop_interation

     IF (IO%IU6>0) THEN
       WRITE(IO%IU6,2231)
       DO i=1,T_INFO%NIONS 
         indx=T_INFO%ITYP(i)         
         WRITE(IO%IU6,2237) i,P(indx)%ELEMENT,RELCHG(i),RELCHG(i)-RELCHG_old(i)
       ENDDO
        WRITE(IO%IU6,2232) iter,iterToleration
     ENDIF


#ifdef MPI 
     CALL M_sum_d(GRIDC%COMM, RELVOL,T_INFO%NIONS )
#endif

     RELVOL=RELVOL*dV
      
     DO i=1,T_INFO%NIONS
       IF (IVDW==23 .OR. IVDW==24 .OR. IVDW==25 .OR. IVDW==26) THEN
         indx=T_INFO%ITYP(i)
         PP=>P(indx)
         CALL avol_rad_limit(WDF_0(indx)%R%NMAX,WDF_work(i)%RHO,WDF_0(indx)%R%R,PP%R%D,rdummy)
         !write(*,*) 'rdummy',i,rdummy
       ELSE 
         rdummy=ATVOL(T_INFO%ITYP(i))
       ENDIF
       !RELVOL(i)=RELVOL(i)/ATVOL(T_INFO%ITYP(i))
       RELVOL(i)=RELVOL(i)/rdummy
     ENDDO

    
     DEALLOCATE(Rarray,RhoAtaarray)
     DEALLOCATE(WDF_work)     
     DEALLOCATE(WDF)

     LFIRST=.FALSE.
       
     !CALL STOP_TIMING("T",IO%IU6,'HI-total')

     RETURN
   END SUBROUTINE hirshfeld_iterative

   SUBROUTINE hirshfeld_dominant(GRIDC, IO,DYN,T_INFO, INFO,LATT_CUR,P,  RELVOL,NALLOC,CWORK1,CWORK2,REFSTATE,ISPIN,XDM, &
     &RCHG,Overlap,M2,HCD )
      USE prec
      USE mpimy
      USE mgrid
      USE AEDENS 
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us    
      USE rhfatm 
      USE constant
      IMPLICIT NONE

      TYPE (grid_3d) GRIDC
      TYPE (type_info)   T_INFO
      TYPE (info_struct) INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar),TARGET ::   P(T_INFO%NTYP) 
      TYPE(dynamics) :: DYN
      TYPE (in_struct)   IO
#ifdef MPI
      RGRID :: CWORK1(GRIDC%RL%NROW,GRIDC%RL%NCOL), CWORK2(GRIDC%RL%NROW,GRIDC%RL%NCOL)
      REAL(q),ALLOCATABLE :: CWORK3(:,:),  CWORK4(:,:)
#else
      RGRID :: CWORK1(DIMREAL(GRIDC%MPLWV)), CWORK2(DIMREAL(GRIDC%MPLWV)) !(GRIDC%NGX*GRIDC%NGY,NZ),CWORK2(GRIDC%NGX*GRIDC%NGY,NZ)
      REAL(q),ALLOCATABLE :: CWORK3(:),  CWORK4(:)
#endif
      INTEGER ISTAT,ISPIN
      REAL(q) :: XDM(GRIDC%RL%NP)
      REAL(q),ALLOCATABLE::  WORK1(:),WORK2(:), WORK3(:),WORK4(:)
      INTEGER NALLOC,NZ
      REAL(q),PARAMETER :: rmax=7. !c cutoff to the radial grid
      INTEGER :: translations(3)
      REAL(q) :: RELVOL(T_INFO%NIONS),RCHG(T_INFO%NIONS),HCD(T_INFO%NIONS)
      REAL(q),ALLOCATABLE,SAVE :: ATVOL(:),ATCHG(:)
      INTEGER :: i,j,indx
      INTEGER NODE_ME,IONODE,NCPU
      LOGICAL,SAVE :: LFIRST=.TRUE.
      REAL(q) :: dV!,dV2,dV3,dV4
      TYPE(atomic),ALLOCATABLE,SAVE :: WDF(:),WDF_0(:)
      TYPE (potcar), POINTER:: PP
      REAL(q) :: chrg
      REAL(q) :: rdummy,y0
      INTEGER :: REFSTATE(T_INFO%NTYP)
      LOGICAL,SAVE :: LIONIC=.FALSE.
      LOGICAL :: LIONREF=.TRUE.  !denominator in V^eff/V^free
      LOGICAL :: LFROZENORBITAL=.FALSE.
      REAL(q) :: chg1,chg2
      LOGICAL,SAVE :: LVDW_RELVOLONE=.FALSE.
      LOGICAL :: LOPEN
      CHARACTER*1 :: CHARAC
      INTEGER :: IERR,N,IDUM
      COMPLEX(q) :: CDUM
      REAL(q) :: RDUM!,tXDM,stXDM
      REAL(q) :: positions_c(3)
      INTEGER NC,NX,NY,NP
      REAL(q),SAVE :: BCK_CHARGE
      REAL(q) :: Overlap(T_INFO%NIONS,T_INFO%NIONS),M2(T_INFO%NIONS)
      
2230 FORMAT(/,"  Hirshfeld-DOM and Hirshfeld charges and relative Volumes:   ",&
            /,"            ion         q(e)           q(e)           rel_vol ",&
            /,"   -----------------------------------------------------------")
2236 FORMAT("   ",I4,5X,A2,5X,F10.3,5X,F10.3,5X,F10.3)


      NODE_ME=1
      NCPU=1
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      NCPU=GRIDC%COMM%NCPU
#endif
      DO i=1,T_INFO%NIONS
        RELVOL(i)=0._q
        RCHG(i)=0._q
        M2(i)=0._q
        DO j=1,T_INFO%NIONS
          Overlap(i,j)=0._q
        END DO
      ENDDO
      !c increment for the volume integration
      dV=LATT_CUR%OMEGA/GRIDC%NGX/GRIDC%NGY/GRIDC%NGZ/(4*PI) 

      !c compute atomic volumes for non-interacting system
      IF (LFIRST) THEN
        !c determine background charge used to stabilize ions
        BCK_CHARGE=INFO%NELECT
        DO i=1,T_INFO%NTYP
          BCK_CHARGE =BCK_CHARGE-T_INFO%NITYP(i)*P(i)%ZVALF
        ENDDO

        !c detect if reference state other than that of neutral atom 
        !c is used
        IF (SUM(REFSTATE**2)>0) THEN
          LIONIC=.TRUE.
         ! write(*,*) 'ionic state detected!!!',REFSTATE
        ENDIF

        LFIRST=.FALSE.
        IF (LIONIC) ALLOCATE(WDF_0(T_INFO%NTYP))
        ALLOCATE(WDF(T_INFO%NTYP))
        ALLOCATE(ATVOL(T_INFO%NTYP),ATCHG(T_INFO%NTYP))

        DO i=1,T_INFO%NTYP
          PP=>P(i)
          CALL DFATOM2(PP,WDF(i),IO,REFSTATE(i),LFROZENORBITAL)
          CALL avol_rad_limit_stephan(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATVOL(i))
          CALL achg_rad(WDF(i)%R%NMAX,WDF(i)%RHO,WDF(i)%R%R,PP%R%D,ATCHG(i))
          !c if needed, compute radial charge distribution for neutral atom
          IF (LIONIC) THEN
            CALL DFATOM2(PP,WDF_0(i),IO,0,.FALSE.)
            IF (.NOT. LIONREF) THEN
              CALL avol_rad(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATVOL(i))
            ENDIF
          ENDIF
        ENDDO
      ENDIF

      !c determine the number of images of simulation cell
      !c in each direction needed for a given 
      !c cutoff radius
       translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
       translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
       translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)

       IF (IO%IU6>0) write(*,*) 'translations:',translations(1),translations(2),translations(3)

#ifdef MPI 
      !c fix the promolecular density if needed   
      IF (LIONIC) THEN
        ALLOCATE(CWORK3(GRIDC%RL%NROW,GRIDC%RL%NCOL),CWORK4(GRIDC%RL%NROW,GRIDC%RL%NCOL))
        CWORK3=0._q; CWORK4=0._q
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          PP=>P(indx) 
          !positions_c=MATMUL(DYN%POSION(:,i),TRANSPOSE(LATT_CUR%A)) 
          DO NC=1,GRIDC%RL%NCOL
            NX= GRIDC%RL%I2(NC)
            NY= GRIDC%RL%I3(NC)
            DO NZ=1,GRIDC%NGZ 
              CALL noninteracting_density3(T_INFO,LATT_CUR,CWORK3(NZ,NC),CWORK4(NZ,NC),WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF_0(indx)%RHO,WDF(indx)%R%R,&
              & WDF_0(indx)%R%D,DYN%POSION(:,i) ,&
              & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax)
            ENDDO
          ENDDO
        ENDDO
        DO NC=1,GRIDC%RL%NCOL
          NX= GRIDC%RL%I2(NC)
          NY= GRIDC%RL%I3(NC)
          DO NZ=1,GRIDC%NGZ 
            IF (ABS(CWORK4(NZ,NC))>1e-8) THEN
              CWORK1(NZ,NC)=CWORK1(NZ,NC)*CWORK3(NZ,NC)/CWORK4(NZ,NC)            
            ENDIF
          ENDDO
        ENDDO

        DEALLOCATE(CWORK3,CWORK4)
      ENDIF

        NP=0
        DO NC=1,GRIDC%RL%NCOL
          NX= GRIDC%RL%I2(NC)
          NY= GRIDC%RL%I3(NC)
          DO NZ=1,GRIDC%NGZ 
          NP=NP+1
            IF (REAL(CWORK1(NZ,NC))>1e-8 .AND. REAL(CWORK2(NZ,NC))>1e-8) THEN
              RDUM=REAL(CWORK2(NZ,NC)/CWORK1(NZ,NC)) 
                 CALL rel_vol_planeDOM(T_INFO,LATT_CUR,RDUM,WDF,P,&
                 & DYN%POSION(:,:),&
                 & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL,RCHG,M2,Overlap,XDM(NP))
            ENDIF
          ENDDO
        ENDDO
      CALL M_sum_d(GRIDC%COMM, RELVOL,T_INFO%NIONS )
      CALL M_sum_d(GRIDC%COMM, M2,T_INFO%NIONS )
      CALL M_sum_d(GRIDC%COMM, RCHG,T_INFO%NIONS )
      CALL M_sum_d(GRIDC%COMM, Overlap,T_INFO%NIONS*T_INFO%NIONS )
     
#else
      !c fix the promolecular density if needed   
      IF (LIONIC) THEN
        ALLOCATE(CWORK3(DIMREAL(GRIDC%MPLWV)),CWORK4(DIMREAL(GRIDC%MPLWV)))
        CWORK3=0._q; CWORK4=0._q
        DO i=1,T_INFO%NIONS
          indx=T_INFO%ITYP(i)
          PP=>P(indx) 
          NC=0
          DO nz=1,GRIDC%NGZ
            DO ny=1,GRIDC%NGY
              DO nx=1,GRIDC%NGX
                 NC=NC+1
                 CALL noninteracting_density3(T_INFO,LATT_CUR,CWORK3(NC),CWORK4(NC),WDF(indx)%R%NMAX,WDF(indx)%RHO,WDF_0(indx)%RHO,WDF(indx)%R%R,&
                 & WDF_0(indx)%R%D,DYN%POSION(:,i),&
                 & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax)    
              ENDDO
            ENDDO
          ENDDO
        ENDDO
        NC=0
        DO nz=1,GRIDC%NGZ
          DO ny=1,GRIDC%NGY
            DO nx=1,GRIDC%NGX
              NC=NC+1
              IF (REAL(CWORK1(NC))>1e-8 .AND. REAL(CWORK4(NC))>1e-8) THEN
                CWORK1(NC)=CWORK1(NC)*CWORK3(NC)/CWORK4(NC)            
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        DEALLOCATE(CWORK3,CWORK4)
      ENDIF
     
      DO i=1,T_INFO%NIONS
        indx=T_INFO%ITYP(i)
        PP=>P(indx) 
        NC=0
        DO nz=1,GRIDC%NGZ
          DO ny=1,GRIDC%NGY
            DO nx=1,GRIDC%NGX
               NC=NC+1
               IF (REAL(CWORK1(NC))>1e-8 .AND. REAL(CWORK2(NC))>1e-8) THEN
                 RDUM=REAL(CWORK2(NC)/CWORK1(NC))            
                 CALL rel_vol_planeDOM(T_INFO,LATT_CUR,RDUM,WDF,P,&
                 & DYN%POSION(:,:),&
                 & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NX,NY,NZ,translations,rmax,RELVOL,RCHG,M2,Overlap,XDM(NC))
               ENDIF
            ENDDO
          ENDDO
        ENDDO
      ENDDO
#endif

      RELVOL=RELVOL*dV
      Overlap=Overlap*dV 
      M2=M2*dV

      DO i=1,T_INFO%NIONS
        RELVOL(i)=RELVOL(i)/ATVOL(T_INFO%ITYP(i))
      ENDDO

      RCHG=RCHG*dV
#if 0
      IF (IO%IU6>0 ) THEN
        WRITE(IO%IU6,2230)
        DO i=1,T_INFO%NIONS 
          indx=T_INFO%ITYP(i)         
          WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,RCHG(i),Overlap(i,i),RELVOL(I)
        ENDDO
      ENDIF  
#endif
      CALL neutralize_charge2(T_INFO%NIONS,RCHG,Overlap,BCK_CHARGE)
  
      DO i=1,T_INFO%NIONS 
      indx=T_INFO%ITYP(i)
      HCD(I)=P(indx)%ZCORE+P(indx)%ZVALF-RCHG(i) !!Check sign
!      write(*,*) 'Population',HCD(I)!,P(indx)%ZCORE,P(indx)%ZVALF
      END DO
      !IF (IO%IU6>0 .AND. IO%NWRITE==3) THEN
      IF (IO%IU6>0 ) THEN
        WRITE(IO%IU6,2230)
        DO i=1,T_INFO%NIONS 
          indx=T_INFO%ITYP(i)         
          WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,RCHG(i),Overlap(i,i),RELVOL(I)
        ENDDO
      ENDIF  

#if 0
       write(IO%IU6,*) "Hirshfeld Overlaps"
       DO i=1,T_INFO%NIONS
          write(IO%IU6,*) "M2",I,M2(I),RELVOL(I)
          DO j=i+1,T_INFO%NIONS
             write(IO%IU6,*) I,J,Overlap(I,J)
          ENDDO
       ENDDO
#endif
      RETURN

   END SUBROUTINE hirshfeld_dominant


    SUBROUTINE hirschfeld_dominant_iterative(GRIDC, IO,DYN,T_INFO, LATT_CUR,P,NALLOC,CWORK_DIM,CWORK1,CWORK2,RELVOL,LFROZENORBITAL)
      USE prec
      USE mpimy
      USE mgrid
      USE AEDENS 
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us    
      USE rhfatm 
      USE constant
      IMPLICIT NONE

      TYPE (grid_3d) GRIDC
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar),TARGET ::   P(T_INFO%NTYP) 
      TYPE(dynamics) :: DYN
      TYPE (in_struct)   IO
      INTEGER :: CWORK_DIM
      COMPLEX(q)  ::  CWORK1(CWORK_DIM),CWORK2(CWORK_DIM)
      REAL(q),ALLOCATABLE::  WORK1(:),WORK2(:), WORK3(:),WORK4(:)
      INTEGER :: NALLOC,NZ
      REAL(q),PARAMETER :: rmax=7. !c cutoff for the radial grid
      INTEGER :: translations(3)
      REAL(q),ALLOCATABLE,SAVE :: RELCHG(:)
      REAL(q) :: RELCHG_old(T_INFO%NIONS)
      REAL(q) :: RELVOL(T_INFO%NIONS)
      REAL(q),ALLOCATABLE,SAVE :: RELVOL_at(:)
      INTEGER :: i,j,k,indx,indx2,iter
      INTEGER NODE_ME,IONODE,NCPU
      REAL(q) :: dV
      TYPE(atomic),ALLOCATABLE,SAVE :: WDF_0(:)
      TYPE(atomic),ALLOCATABLE :: WDF(:),WDF_new(:),WDF_work(:)
      TYPE (potcar), POINTER:: PP
      REAL(q) :: chrg
      REAL(q) :: rdummy,y0
      INTEGER :: defstates(2,T_INFO%NIONS),  defstates_new(2,T_INFO%NIONS) 
      LOGICAL :: oxstates(T_INFO%NTYP,17),oxstates_old(T_INFO%NTYP,17)
      INTEGER :: noxstates(T_INFO%NTYP),noxstates_total
      INTEGER :: map_ox_states(T_INFO%NIONS)
      REAL(q) :: test_ATCHG
      INTEGER,PARAMETER :: maxiter=150 !c max. number of iterations
      LOGICAL :: LSTOP,LRESET
      LOGICAL :: LFIRST=.TRUE.
      REAL(q),parameter :: iterToleration=1.e-5
      INTEGER,ALLOCATABLE,SAVE :: anionic_capacity(:)
      REAL(q),ALLOCATABLE,SAVE :: ATVOL(:),ATCHG(:)
      REAL(q) :: chg1,chg2,chg3
      REAL(q),PARAMETER :: MIX=1.0 
      INTEGER :: DOWNSAMPLE=1
      REAL(q) :: ATVOL_dum
      LOGICAL :: LTESTHIRSH=.FALSE. !c for test purposes - mimics standard Hirshfeld partitioning
      LOGICAL :: LFROZENORBITAL   !=.FALSE.
      REAL(q),ALLOCATABLE::  WORK5(:)
      !REAL(q),PARAMETER :: rholimit=1e-4
      REAL(q) :: dummy_q
 
      NODE_ME=1
      NCPU=1
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      NCPU=GRIDC%COMM%NCPU
#endif

      ALLOCATE(WORK1(NALLOC),WORK2(NALLOC))
      ALLOCATE(WORK3(NALLOC),WORK4(NALLOC))
      ALLOCATE(WORK5(NALLOC))

      LSTOP=.FALSE.
      LRESET=.FALSE.
      
2230 FORMAT(/,"  Hirshfeld charges:",&
            /,"            ion         q(e)     ",&
            /,"   ------------------------------")
2231 FORMAT(/,"  Hirshfeld-I charges:",&
            /,"            ion         q(e)     ",&
            /,"   ------------------------------")
2236 FORMAT("   ",I4,5X,A2,5X,F10.3)
    
      !c increment for the volume integration
      dV=LATT_CUR%OMEGA/GRIDC%NGX/GRIDC%NGY/GRIDC%NGZ/(4*PI) 

      IF (LFIRST) THEN
        ALLOCATE(WDF_0(T_INFO%NTYP))
        ALLOCATE(anionic_capacity(T_INFO%NTYP))
        ALLOCATE(ATVOL(T_INFO%NTYP),ATCHG(T_INFO%NTYP))
        ALLOCATE(RELCHG(T_INFO%NIONS))
        RELCHG=0._q
        !beg test
        ALLOCATE(RELVOL_at(T_INFO%NIONS))
        RELVOL_at=0._q
        !end test

        !c compute radial charge densities and volumes for non-interacting atoms
        DO i=1,T_INFO%NTYP
          PP=>P(i)
          CALL DFATOM2(PP,WDF_0(i),IO,0,.FALSE.)
          CALL max_anion(WDF_0(i)%NS,WDF_0(i)%OCC,WDF_0(i)%L,anionic_capacity(i))
          !CALL avol_rad_dominant(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATVOL(i))
          CALL avol_rad_limit(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATVOL(i))
          CALL achg_rad_limit(WDF_0(i)%R%NMAX,WDF_0(i)%RHO,WDF_0(i)%R%R,PP%R%D,ATCHG(i))
        ENDDO
      ENDIF 

      !IF (IO%IU6>0)  write(*,*) "anionic_capacity",anionic_capacity

     !c determine the number of images of simulation cell
     !c in each direction needed for a given 
     !c cutoff radius
      translations(1)=nint(rmax*SUM(LATT_CUR%B(:,1)**2)**0.5)
      translations(2)=nint(rmax*SUM(LATT_CUR%B(:,2)**2)**0.5)
      translations(3)=nint(rmax*SUM(LATT_CUR%B(:,3)**2)**0.5)

      IF (IO%IU6>0) write(*,*) 'translations:',translations(1),translations(2),translations(3)

      !c first, compute charges using classical Hirschfeld scheme  
      IF (LFIRST) THEN           
        RELCHG_old=0._q
        RELVOL=0._q
        !beg test
        RELVOL_at=0._q
        !end test

        dummy_q=0._q
        DO NZ=1,GRIDC%NGZ
          CALL MRG_GRID_RL_PLANE(GRIDC, WORK1, CWORK1, NZ)
          CALL MRG_GRID_RL_PLANE(GRIDC, WORK2, CWORK2, NZ)
         
          !beg test 
          WORK5=0._q
          DO i=1,NALLOC
            IF (WORK1(i) .GE. rholimit*LATT_CUR%OMEGA) THEN
              WORK5(i)=WORK1(i)
            ENDIF
          ENDDO  
          WORK5=WORK5/LATT_CUR%OMEGA*4*PI
          WORK3=1._q                    
          !end test 

          DO i=1,NALLOC
            IF (WORK1(i)>1e-8) THEN
              WORK2(i)=WORK2(i)/WORK1(i)
            ELSE
              WORK2(i)=0._q
            ENDIF
          ENDDO
          
          DO i=1,T_INFO%NIONS
            indx=T_INFO%ITYP(i)
            CALL hirschfeld_dominant_charge_plane(T_INFO,LATT_CUR,WORK2,WORK1,NALLOC,WDF_0(indx)%R%NMAX,WDF_0(indx)%RHO,WDF_0(indx)%R%R,&
            & WDF_0(indx)%R%D, DYN%POSION(:,i),&
            & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NZ,translations,rmax,RELCHG_old(i),RELVOL(i),NODE_ME,NCPU,1)

            !beg test
            CALL hirschfeld_dominant_charge_plane(T_INFO,LATT_CUR,WORK3,WORK5,NALLOC,WDF_0(indx)%R%NMAX,WDF_0(indx)%RHO,WDF_0(indx)%R%R,&
            & WDF_0(indx)%R%D, DYN%POSION(:,i),&
            & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NZ,translations,rmax,dummy_q,RELVOL_at(i),NODE_ME,NCPU,1)
            !end test
          ENDDO
          WORK3=0._q
          WORK5=0._q
        ENDDO

#ifdef MPI
        CALL M_sum_d(GRIDC%COMM, RELCHG_old,T_INFO%NIONS )
        CALL M_sum_d(GRIDC%COMM, RELVOL_at,T_INFO%NIONS )
#endif
        RELCHG_old=RELCHG_old*dV
        !beg test
        !RELVOL_at=4*PI*RELVOL_at*dV*DOWNSAMPLE**3
        RELVOL_at=RELVOL_at*dV*DOWNSAMPLE**3
        !end test

        CALL neutralize_charge(T_INFO%NIONS,RELCHG_old,0._q)

        RELCHG=RELCHG_old

        IF (IO%IU6>0) THEN
          WRITE(IO%IU6,2230)
          DO i=1,T_INFO%NIONS 
            indx=T_INFO%ITYP(i)         
            WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,RELCHG_old(i)
          ENDDO
        ENDIF  
      ENDIF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
 
      !c find lower and upper integers for partial charges
      CALL find_int_limits(T_INFO,RELCHG,defstates,anionic_capacity)

      !c identify needed oxidation states  
      CALL finx_ox_states(T_INFO,defstates,oxstates,noxstates,noxstates_total)
      
      ALLOCATE(WDF(noxstates_total))
      
      map_ox_states=0
      indx=0
      DO i=1,T_INFO%NTYP
        PP=>P(i)
        DO j=-8,8
          IF (oxstates(i,j+9)) THEN
            indx=indx+1
            IF (j==0) THEN
              WDF(indx)=WDF_0(i)
            ELSE
              IF (IO%IU6>0) write(*,*) "ox",i,j
              CALL DFATOM2(PP,WDF(indx),IO,j,LFROZENORBITAL)
            ENDIF
            DO k=1,T_INFO%NIONS
              IF (T_INFO%ITYP(k)==i .AND. defstates(1,k)==j) map_ox_states(k)=indx
            ENDDO
          ENDIF
        ENDDO
      ENDDO

      ALLOCATE(WDF_work(T_INFO%NIONS))
      DO i=1,T_INFO%NIONS    
        indx2=map_ox_states(i)
        ALLOCATE(WDF_work(i)%RHO(WDF(indx2)%R%NMAX))     
        IF (LTESTHIRSH)   RELCHG(i)=0._q 
        WDF_work(i)%RHO=WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i))      
     ENDDO

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     IF (IO%IU6>0) write(*,*) 'RELCHG:',RELCHG
!!cc
     DO iter=1,maxiter

     !  chg1=0._q;chg2=0._q;chg3=0._q
       IF (LTESTHIRSH)   RELCHG=0._q
       RELCHG_old=RELCHG
       RELVOL=0._q
       DO NZ=1,GRIDC%NGZ,DOWNSAMPLE
         WORK3=0._q
         WORK4=0._q
         DO i=1,T_INFO%NIONS 
           indx2=map_ox_states(i)   
           indx=T_INFO%ITYP(i)
           CALL noninteracting_density2(T_INFO,LATT_CUR,WORK3,WORK4,NALLOC,WDF(indx2)%R%NMAX,WDF_work(i)%RHO,WDF_0(indx)%RHO,WDF(indx2)%R%R,&
           & WDF_0(indx)%R%D, DYN%POSION(:,i),&
           & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NZ,translations,rmax,NODE_ME,NCPU,DOWNSAMPLE)

         ENDDO
#ifdef MPI
         CALL M_sum_d(GRIDC%COMM,WORK3 , NALLOC)
         CALL M_sum_d(GRIDC%COMM,WORK4 , NALLOC)        
#endif
         CALL MRG_GRID_RL_PLANE(GRIDC, WORK1, CWORK1, NZ)
         CALL MRG_GRID_RL_PLANE(GRIDC, WORK2, CWORK2, NZ)

        !c promolecular density for iterative Hirshfeld-dominant scheme
         WORK5=0._q
         DO i=1,NALLOC
           IF (WORK2(i) .GE. rholimit*LATT_CUR%OMEGA) THEN
             WORK5(i)=WORK3(i)
             !IF (WORK4(i)>1e-8) THEN
             !  WORK5(i)=(WORK1(i)*WORK3(i))/WORK4(i)/LATT_CUR%OMEGA*4*PI
             !ENDIF
           ENDIF
         ENDDO
  
         WORK2=WORK2*WORK4
         WORK1=WORK1*WORK3

         DO i=1,NALLOC
           IF (WORK1(i)>1e-8) THEN
             WORK2(i)=WORK2(i)/WORK1(i)
           ELSE
             WORK2(i)=0._q
           ENDIF
         ENDDO

!          WORK3=WORK3*LATT_CUR%OMEGA/4/PI
!          DO i=1,NALLOC
!            IF (WORK3(i)>1e-8) THEN
!             WORK2(i)=WORK2(i)/WORK3(i)
!            ELSE
!              WORK2(i)=0._q
!            ENDIF
!          ENDDO
         
         DO i=1,T_INFO%NIONS
           indx2=map_ox_states(i)
           indx=T_INFO%ITYP(i)
           CALL hirschfeld_dominant_charge_plane(T_INFO,LATT_CUR,WORK2,WORK5,NALLOC,WDF(indx2)%R%NMAX,WDF_work(i)%RHO,WDF(indx2)%R%R,&
           & WDF(indx2)%R%D, DYN%POSION(:,i),&
           & GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ,NZ,translations,rmax,RELCHG(i),RELVOL(i),NODE_ME,NCPU,DOWNSAMPLE)
           
         ENDDO
       ENDDO
      
#ifdef MPI
       CALL M_sum_d(GRIDC%COMM, RELCHG,T_INFO%NIONS )
       CALL M_sum_d(GRIDC%COMM, RELVOL,T_INFO%NIONS )
#endif

!        DO i=1,T_INFO%NIONS
!          IF (IO%IU6>0) THEN
!              write(*,*) 'pupu',RELVOL(i),4*PI*RELVOL(i)*dV*DOWNSAMPLE**3
!          ENDIF
!        ENDDO


       RELCHG=RELCHG*dV*DOWNSAMPLE**3
       CALL neutralize_charge(T_INFO%NIONS,RELCHG,0._q)

       IF (LTESTHIRSH)   RELCHG=0._q

       IF (DOWNSAMPLE .EQ. 1) THEN
         IF (MAXVAL(ABS(RELCHG)) .LE. iterToleration) LSTOP=.TRUE.
       ELSE
         IF (MAXVAL(ABS(RELCHG)) .LE. 10*iterToleration) DOWNSAMPLE=1
       ENDIF
       RELCHG=RELCHG+RELCHG_old

       IF (LTESTHIRSH)   RELCHG=0._q

       IF (IO%IU6>0) THEN
         write(*,*) 'iteration:',iter
         DO i=1,T_INFO%NIONS 
           write(*,*) "atom:",i,RELCHG(i),RELCHG(i)-RELCHG_old(i)
         ENDDO
       ENDIF

       !RELVOL=4*PI*RELVOL*dV*DOWNSAMPLE**3
       RELVOL=RELVOL*dV*DOWNSAMPLE**3 
      
       DO i=1,T_INFO%NIONS
         write(*,*) 'rcheck:',RELVOL(i),RELVOL_at(i),ATVOL(T_INFO%ITYP(i))
         !RELVOL(i)=RELVOL(i)/ATVOL(T_INFO%ITYP(i))
         RELVOL(i)=RELVOL(i)/RELVOL_at(i)
         write(*,*) 'relvol:', i,relvol(i)
       ENDDO

       !c check if we need new limits for ox. states
       CALL find_int_limits(T_INFO,RELCHG,defstates_new,anionic_capacity)
       CALL check_ox_consistency(T_INFO%NIONS,defstates,defstates_new,LRESET)

       IF (LRESET) THEN
         IF (IO%IU6>0) write(*,*) 'WARNING: new ox. states have to be generated!!!'
         defstates=defstates_new
         oxstates_old=oxstates
         CALL finx_ox_states(T_INFO,defstates,oxstates,noxstates,noxstates_total)
 
         ALLOCATE(WDF_new(noxstates_total))

         map_ox_states=0
         indx=0
         indx2=0
         DO i=1,T_INFO%NTYP
           PP=>P(i)
           DO j=-8,8
             IF (oxstates_old(i,j+9)) indx2=indx2+1

             IF (oxstates(i,j+9)) THEN
               indx=indx+1
               IF (oxstates_old(i,j+9)) THEN
                 WDF_new(indx)=WDF(indx2)
               ELSE
                 IF (IO%IU6>0) write(*,*) "ox",i,j          
                CALL DFATOM2(PP,WDF_new(indx),IO,j,LFROZENORBITAL)
               ENDIF
               DO k=1,T_INFO%NIONS
                 IF (T_INFO%ITYP(k)==i .AND. defstates(1,k)==j) map_ox_states(k)=indx
               ENDDO
             ENDIF
           ENDDO
         ENDDO
         DEALLOCATE(WDF)
         ALLOCATE(WDF(noxstates_total))
         WDF=WDF_new
         DEALLOCATE(WDF_new)

         DO i=1,T_INFO%NIONS    
           indx2=map_ox_states(i)   
           WDF_work(i)%RHO=(1._q-MIX)*WDF_work(i)%RHO+MIX*(WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i)))            
         ENDDO

         IF (IO%IU6>0) THEN
           write(*,*) "atomic charges from standard Hirschfeld_r:"
           DO i=1,T_INFO%NIONS 
             indx=T_INFO%ITYP(i)         
             write(*,*) "atom0_r:",i,RELCHG(i),defstates(1,i),defstates(2,i)
           ENDDO
         ENDIF

         LRESET=.FALSE.
        !STOP
       ENDIF 

      
       DO i=1,T_INFO%NIONS     
         indx2=map_ox_states(i)          
         WDF_work(i)%RHO=(1._q-MIX)*WDF_work(i)%RHO+MIX*(WDF(indx2+1)%RHO*(RELCHG(i)-1._q*defstates(1,i))+WDF(indx2)%RHO*(1._q*defstates(2,i)-RELCHG(i)))   
       ENDDO 
 
 
       IF (LSTOP) THEN
         EXIT
       ELSE IF (iter .GE. maxiter) THEN
         IF (IO%IU6>0) write(*,*) "Error(hirshfeld_iterative): Hirshfeld-I did not converge in",maxiter,"steps"
         STOP                                                                                                              
       ENDIF
     ENDDO

     IF (IO%IU6>0) THEN
       WRITE(IO%IU6,2231)
       DO i=1,T_INFO%NIONS 
         indx=T_INFO%ITYP(i)         
         WRITE(IO%IU6,2236) i,P(indx)%ELEMENT,RELCHG(i)
       ENDDO
     ENDIF


     DEALLOCATE(WDF_work)     
     DEALLOCATE(WDF)
     DEALLOCATE(WORK5)
     DEALLOCATE(WORK4,WORK3)
     DEALLOCATE(WORK2,WORK1)

     LFIRST=.FALSE.
     RETURN
   END SUBROUTINE hirschfeld_dominant_iterative

   SUBROUTINE neutralize_charge(nions,RELCHG,BCK_CHARGE)
     INTEGER :: nions,i
     REAL(q) :: RELCHG(nions)
     REAL(q) :: BCK_CHARGE
     REAL(q) :: totchg

     totchg=(SUM(RELCHG)+BCK_CHARGE)/nions
     DO i=1,nions
       RELCHG(i)=RELCHG(i)-totchg
     ENDDO

   END SUBROUTINE neutralize_charge

   SUBROUTINE neutralize_charge2(nions,RELCHG,Overlap,BCK_CHARGE)
     INTEGER :: nions,i,j
     REAL(q) :: RELCHG(nions)
     REAL(q) :: Overlap(nions,nions)
     REAL(q) :: BCK_CHARGE
     REAL(q) :: totchg,totchg2

     totchg=(SUM(RELCHG)+BCK_CHARGE)/nions
     totchg2=totchg/nions
     DO i=1,nions
       RELCHG(i)=RELCHG(i)-totchg
     Overlap(i,i)=Overlap(i,i)-totchg
     DO j=1,nions
       if(i==j)cycle
     Overlap(i,j)=Overlap(i,j)-totchg2
     ENDDO
     END DO
     RETURN
   END SUBROUTINE neutralize_charge2


   SUBROUTINE check_ox_consistency(nions,defstates,defstates_new,LRESET)
     INTEGER :: nions,i
     INTEGER :: defstates(2,nions)
     INTEGER :: defstates_new(2,nions)
     LOGICAL :: LRESET

     LRESET=.FALSE.
     DO i=1,nions
       IF (defstates(1,i) .NE. defstates_new(1,i)) THEN
         LRESET=.TRUE.
         EXIT
       ENDIF
     ENDDO     
   END SUBROUTINE

   SUBROUTINE find_int_limits(T_INFO,relchg,defstates,acapacity)
     TYPE (type_info)   T_INFO
     INTEGER :: i,nt
     REAL(q) :: relchg(T_INFO%NIONS)
     INTEGER :: defstates(2,T_INFO%NIONS)
     INTEGER :: acapacity(T_INFO%NTYP)

     DO i=1,T_INFO%NIONS
        nt= T_INFO%ITYP(i)
        if (RELCHG(i) .GE. 0._q) THEN
          defstates(1,i)=INT((RELCHG(i)))
          defstates(2,i)=INT((RELCHG(i)))+1
        else        
          !c we have to be carefull not to add more electrons
          !c than available unocupied shells...
          defstates(1,i)=MAX(INT((RELCHG(i)))-1,-acapacity(nt))
          defstates(2,i)=defstates(1,i)+1 
          !defstates(1,i)=INT((RELCHG(i)))-1
          !defstates(2,i)=INT((RELCHG(i)))
        end if 
      ENDDO
   END SUBROUTINE find_int_limits

   SUBROUTINE finx_ox_states(T_INFO,defstates,oxstates,noxstates,noxstates_total)
   !c identify all oxidation states that we need
     TYPE (type_info)   T_INFO
     INTEGER :: i,j,nt,indx
     INTEGER :: defstates(2,T_INFO%NIONS)
     LOGICAL :: oxstates(T_INFO%NTYP,17)
     INTEGER :: noxstates(T_INFO%NTYP)
     INTEGER :: noxstates_total

     oxstates=.FALSE.
     DO i=1,T_INFO%NIONS
       nt= T_INFO%ITYP(i)    
       DO j=-8,8
         IF (defstates(1,i)==j .OR. defstates(2,i)==j) then
            indx=j+9
            !write(*,*) 'tyt',defstates(1,i),defstates(2,i)
            oxstates(nt,indx)=.TRUE.
            !write(*,*) 'lol',oxstates(nt,indx)
          end if
       ENDDO
     ENDDO

     !c count how many ox. states we need for each ionic type
     noxstates=0
     noxstates_total=0
     DO i=1,T_INFO%NTYP
       DO j=1,17
         IF (oxstates(i,j)) THEN
           noxstates(i)=noxstates(i)+1  
           noxstates_total=noxstates_total+1
         ENDIF
       ENDDO
     ENDDO

   END SUBROUTINE finx_ox_states

   SUBROUTINE max_anion(NDIM,OCC,L,anCapacity)
   !c check how many electrons can be added
   !c to create anion
       INTEGER :: NDIM,i,j
       REAL(q) :: OCC(NDIM)
       REAL(q) :: E(NDIM)
       INTEGER :: L(NDIM)
       REAL(q) :: emin 
       INTEGER :: nmin,maxocc
       INTEGER :: anCapacity

       anCapacity=0
      
       DO i=1,NDIM
         maxocc=4*L(i)+2
         anCapacity=anCapacity+(maxocc-OCC(i))
       ENDDO
       
      !anCapacity=1
   END SUBROUTINE max_anion

!tb test end

SUBROUTINE dist1d(gridc,s0, ntot, s, nloc)
  IMPLICIT NONE
  TYPE (grid_3d) GRIDC
  INTEGER, INTENT(in) :: s0, ntot
  INTEGER, INTENT(out) :: s, nloc
  INTEGER :: me, npes, ierr, naver, rem
          ME=1
          npes=1
#ifdef MPI
           ME=GRIDC%COMM%NODE_ME-1
           npes=GRIDC%COMM%NCPU
#endif
!  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, npes, ierr)
!  CALL MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)
  naver = ntot/npes
  rem = MODULO(ntot,npes)
  s = s0 + MIN(rem,me) + me*naver
  nloc = naver
  IF( me.LT.rem ) nloc = nloc+1
!  write(*,*) "I", ME, "am working on",s, " to", nloc!
END SUBROUTINE dist1d


END MODULE





