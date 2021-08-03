#include "symbol.inc"
!************************ SUBROUTINE FEXCF *****************************
! RCS:  $Id: potex1.F,v 1.2 2000/11/15 08:23:50 kresse Exp $
!
! this subroutine calculates the LDA contribution to the 
! exchange correlation potential from the charge density and
! magnetisation density supplied in CHDENR and CHMAGR (real space)
! The energy corrections due to overcounting the exchange correlation 
! energy on summing the electronic eigenvalues and the forces on the unit 
! cell due to the exchange correlation energy are also calculated.
!
! To make the calculations efficient the required functions are
! interpolated from tables (see setex.F)
! this makes the routine extremely efficient without sacrificing
! precision
!***********************************************************************

      SUBROUTINE FEXCF(GRID,OMEGA, &
          CHDENR,CHMAGR,DENCOR,CVXC1,CVXC2,CVZERO,EXC,XCENC,XCSIF,LADD)
      USE prec

      USE mpimy
      USE mgrid
      USE setexm
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)  GRID

      RGRID  CHDENR(GRID%RL%NP),CHMAGR(GRID%RL%NP)
      RGRID  DENCOR(GRID%RL%NP)
      RGRID  CVXC1(GRID%RL%NP),CVXC2(GRID%RL%NP)

      LOGICAL      LADD
      REAL(q)      XCSIF(3,3)
!=======================================================================
!  calculate exchange-correlation on the grid using the tables
!=======================================================================

      CALL XCTABLE_CHECK

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRID%COMM%NODE_ME
      IONODE =GRID%COMM%IONODE
#endif
      IMAX=0

      CVZERO=0._q
      EXC=0._q
      XCF=0._q
      XCFO=0._q

      EXCSC2=(EXCTAB%NEXCHF(2)-EXCTAB%NEXCHF(1))/(EXCTAB%RHOEXC(2)-EXCTAB%RHOEXC(1))
      EXCSC1= EXCTAB%NEXCHF(1)/MAX(EXCTAB%RHOEXC(1),1E-10_q)

      IMIN=EXCTAB%NEXCHF(2)
!DIR$ IVDEP
!cdir nodep
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO N=1,GRID%RL%NP

        DVAL= REAL( CHDENR(N) ,KIND=q)
        DENS=DVAL+DENCOR(N)
        DENS=MAX(DENS,1E-8_q)

        DMAG= REAL( CHMAGR(N) ,KIND=q)

        ZETA=DMAG/DENS
        IF (ZETA>= 1._q) ZETA= 0.9999999999999_q
        IF (ZETA<=-1._q) ZETA=-0.9999999999999_q
        ZETA3=ZETA*ZETA*ZETA
        ZETA4=ZETA3*ZETA

        RHO  =DENS/OMEGA
        RHO13=EXP(LOG(RHO)/3._q)
        RHO43=RHO*RHO13
        RHOD =4._q*RHO13/3._q
        RHOP =0.5_q*(DVAL+DMAG)
        RHOM =0.5_q*(DVAL-DMAG)
        ARG  =(RHO*EXCSC1)+1
        IF (ARG>=EXCTAB%NEXCHF(1)+1) ARG=(RHO-EXCTAB%RHOEXC(1))*EXCSC2+EXCTAB%NEXCHF(1)+1
        ZARG =ABS(ZETA)*(EXCTAB%NEXCHF(2)-1)+1

        I   =INT(ARG)
        IMAX=MAX(I,IMAX)
        I   =MIN(I,EXCTAB%NEXCHF(2))
        IZ  =INT(ZARG)
        REM =RHO-EXCTAB%EXCTAB(I,1,1)
        ZREM=ABS(ZETA)-EXCTAB%EXCTAB(IZ,1,5)

        EXCP0=EXCTAB%EXCTAB(I,2,1)+REM*(EXCTAB%EXCTAB(I,3,1)+ &
     &                      REM*(EXCTAB%EXCTAB(I,4,1)+REM*EXCTAB%EXCTAB(I,5,1)))
        VXCP0=EXCTAB%EXCTAB(I,3,1)+REM*(EXCTAB%EXCTAB(I,4,1)*2+REM*EXCTAB%EXCTAB(I,5,1)*3)
        EXCP =EXCP0*RHO43
        VXCP =VXCP0*RHO43+EXCP0*RHOD

        DEXF0=EXCTAB%EXCTAB(I,2,2)+REM*(EXCTAB%EXCTAB(I,3,2)+ &
     &                      REM*(EXCTAB%EXCTAB(I,4,2)+REM*EXCTAB%EXCTAB(I,5,2)))
        DVXF0=EXCTAB%EXCTAB(I,3,2)+REM*(EXCTAB%EXCTAB(I,4,2)*2+REM*EXCTAB%EXCTAB(I,5,2)*3)
        DEXF1=DEXF0*RHO13
        DEXF =DEXF0*RHO43
        DVXF =DVXF0*RHO43+DEXF0*RHOD

        DECF0=EXCTAB%EXCTAB(I,2,3)+REM*(EXCTAB%EXCTAB(I,3,3)+ &
     &                      REM*(EXCTAB%EXCTAB(I,4,3)+REM*EXCTAB%EXCTAB(I,5,3)))
        DVCF0=EXCTAB%EXCTAB(I,3,3)+REM*(EXCTAB%EXCTAB(I,4,3)*2+REM*EXCTAB%EXCTAB(I,5,3)*3)
        DECF1=DECF0*RHO13
        DECF =DECF0*RHO43
        DVCF =DVCF0*RHO43+DECF0*RHOD

        ALPH0=EXCTAB%EXCTAB(I,2,4)+REM*(EXCTAB%EXCTAB(I,3,4)+ &
     &                      REM*(EXCTAB%EXCTAB(I,4,4)+REM*EXCTAB%EXCTAB(I,5,4)))
        DALPH=EXCTAB%EXCTAB(I,3,4)+REM*(EXCTAB%EXCTAB(I,4,4)*2+REM*EXCTAB%EXCTAB(I,5,4)*3)
        DECA1=ALPH0*RHO13
        DECA =ALPH0*RHO43
        DVCA =DALPH*RHO43+ALPH0*RHOD

        FZETX=EXCTAB%EXCTAB(IZ,2,5)+ZREM*(EXCTAB%EXCTAB(IZ,3,5)+ &
     &                      ZREM*(EXCTAB%EXCTAB(IZ,4,5)+ZREM*EXCTAB%EXCTAB(IZ,5,5)))
        DZETX=EXCTAB%EXCTAB(IZ,3,5)+ZREM*(EXCTAB%EXCTAB(IZ,4,5)*2+ &
     &                     ZREM*EXCTAB%EXCTAB(IZ,5,5)*3)
        DZETX=SIGN(1._q,ZETA)*ZETA*ZETA*DZETX+2*ZETA*FZETX
        FZETX=ZETA*ZETA*FZETX

        FZETC=EXCTAB%EXCTAB(IZ,2,6)+ZREM*(EXCTAB%EXCTAB(IZ,3,6)+ &
     &                      ZREM*(EXCTAB%EXCTAB(IZ,4,6)+ZREM*EXCTAB%EXCTAB(IZ,5,6)))
        DZETC=EXCTAB%EXCTAB(IZ,3,6)+ZREM*(EXCTAB%EXCTAB(IZ,4,6)*2+ &
     &                     ZREM*EXCTAB%EXCTAB(IZ,5,6)*3)
        DZETC=SIGN(1._q,ZETA)*ZETA*ZETA*DZETC+2*ZETA*FZETC
        FZETC=ZETA*ZETA*FZETC

        EXCT =EXCP+DEXF*FZETX+(DECF+DECA*ZETA4)*FZETC
        VXC0 =VXCP+DVXF*FZETX+(DVCF+DVCA*ZETA4)*FZETC
        DVXC =DEXF1*DZETX+(DECF1+DECA1*ZETA4)*DZETC+4._q*DECA1*ZETA3*FZETC
        VXC1 =VXC0-DVXC*(ZETA-1)
        VXC2 =VXC0-DVXC*(ZETA+1)
        VXCS =0.5_q*(VXC1+VXC2)

        CVZERO=CVZERO+VXCS
        EXC   =EXC   +EXCT
        XCF   =XCF   -VXC1*RHOP-VXC2*RHOM
        XCFO  =XCFO  -VXCS*DENCOR(N)

        IF (LADD) THEN
           CVXC1(N)= CVXC1(N)+VXC1
           CVXC2(N)= CVXC2(N)+VXC2
        ELSE
           CVXC1(N)= VXC1
           CVXC2(N)= VXC2
        ENDIF

      ENDDO
      CALLMPI( M_max_i(GRID%COMM, IMAX, 1))
!=======================================================================
!  check if the size of the table is sufficent
!=======================================================================
      IF (IMAX>EXCTAB%NEXCHF(2)*.96_q) THEN
        do_io WRITE(*,*)'ERROR FEXCF: supplied exchange-correlation table'
        do_io WRITE(*,*)'   is too small, maximal index :',IMAX
        STOP
      ENDIF
!=======================================================================
! calculate the g=0 component of the exchange-correlation potential
! CVZERO which is used in the electron dynamics subroutine
!=======================================================================
      CVZERO=CVZERO/GRID%NPLWV

      EXC=EXC*OMEGA
      XCF =EXC+XCF
      EXC =EXC /GRID%NPLWV
      XCF =XCF /GRID%NPLWV
      XCFO=XCFO/GRID%NPLWV

      CALLMPI( M_sum_s(GRID%COMM,3, EXC,XCF,XCFO, 0._q ))
      CALLMPI( M_sum_z(GRID%COMM,CVZERO,1))
!=======================================================================
! calculate the force on the unit cell due to the change in the exchange
! correlation energy on changing the size of the cell
! the uniform part of the partial core correction (from
! the 1/OMEGA dependency of the core charge is treated here)
!=======================================================================
#ifdef debug
        do_io  WRITE(*,'(A,F24.14)') '<rho*exc> =',EXC
        do_io  WRITE(*,'(A,F24.14)') '<rho*vxc> =',EXC-XCF
        do_io  WRITE(*,'(A,F24.14)') '    xcenc =',XCF
#endif
      XCSIF=0._q

      XCSIF(1,1)=-XCF-XCFO
      XCSIF(2,2)=-XCF-XCFO
      XCSIF(3,3)=-XCF-XCFO
!=======================================================================
! double counting correction to the total energy
!=======================================================================
      XCENC=XCF

      RETURN
      END
