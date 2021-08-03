#include "symbol.inc"
!************************ SUBROUTINE FEXCP *****************************
! RCS:  $Id: potex2.F,v 1.3 2001/01/31 11:51:53 kresse Exp $
!
! this subroutine calculates the LDA contribution to the 
! exchange correlation potential from the charge density supplied
! int CHDENR (real space)
! The energy corrections due to overcounting the exchange correlation 
! energy on summing the electronic eigenvalues and the forces on the unit 
! cell due to the exchange correlation energy are also calculated.
!
! To make the calculations efficient the required functions are
! interpolated from tables (see setex.F)
! this makes the routine extremely efficient without sacrificing
! precision
!***********************************************************************

      SUBROUTINE FEXCP(GRID,OMEGA, &
          CHDENR,DENCOR,CVXC,DVXC,CVZERO,EXC,XCENC,XCSIF,LADD)
      USE prec

      USE mpimy
      USE mgrid
      USE setexm
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)  GRID

      RGRID CHDENR(GRID%RL%NP)
      RGRID DENCOR(GRID%RL%NP)
      RGRID DVXC(GRID%RL%NP)
      RGRID CVXC(GRID%RL%NP)

      REAL(q)  XCSIF(3,3)
      LOGICAL :: LADD

      CALL XCTABLE_CHECK
!=======================================================================
!  calculate exchange-correlation on the grid using the tables
!=======================================================================
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
      EXCSC1=EXCTAB%NEXCHF(1)/MAX(EXCTAB%RHOEXC(1),1E-10_q)

      IMIN=EXCTAB%NEXCHF(2)
!DIR$ IVDEP
!cdir nodep
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO N=1,GRID%RL%NP
        DENS= REAL( CHDENR(N) ,KIND=q) +DENCOR(N)
        DENS=MAX(DENS,1E-8_q)

        RHO =ABS(DENS)/OMEGA
        ARG=(RHO*EXCSC1)+1
        IF (ARG>=EXCTAB%NEXCHF(1)+1) ARG=(RHO-EXCTAB%RHOEXC(1))*EXCSC2+EXCTAB%NEXCHF(1)+1

        I=INT(ARG)
        IMAX=MAX(I,IMAX)

        I=MIN(I,EXCTAB%NEXCHF(2))
        
        REM=RHO-EXCTAB%EXCTAB(I,1,1)
        RHO13=EXP(LOG(RHO)/3._q)

        RHO0   = RHO13*RHO
        RHOD   = 4._q/3._q*RHO13
        RHODD  = 4._q/9._q*RHO13/MAX(RHO,1E-3_q)

        EXC0   = EXCTAB%EXCTAB(I,2,1)+REM*(EXCTAB%EXCTAB(I,3,1)+ &
     &                       REM*(EXCTAB%EXCTAB(I,4,1)+REM*EXCTAB%EXCTAB(I,5,1)))
        EXCD   = EXCTAB%EXCTAB(I,3,1)+REM*(EXCTAB%EXCTAB(I,4,1)*2+REM*EXCTAB%EXCTAB(I,5,1)*3)
        EXCDD  = EXCTAB%EXCTAB(I,4,1)*2+REM*EXCTAB%EXCTAB(I,5,1)*6

        EXE    = EXC0*RHO0
        VXC    = EXC0*RHOD+EXCD*RHO0

        CVZERO=CVZERO+VXC
        EXC   =EXC   +EXE
        XCF   =XCF   -VXC*REAL( CHDENR(N) ,KIND=q)
        XCFO  =XCFO  -VXC*DENCOR(N)

        IF (LADD) THEN
           CVXC(N)= CVXC(N)+VXC
        ELSE
           CVXC(N)= VXC
        ENDIF
        DVXC(N)= EXC0*RHODD+2*EXCD*RHOD+EXCDD*RHO0
      ENDDO
!=======================================================================
!  check if the size of the table is sufficent
!=======================================================================
      CALLMPI( M_max_i(GRID%COMM, IMAX, 1))
      IF (IMAX>EXCTAB%NEXCHF(2)*.96_q) THEN
        do_io WRITE(*,*)'ERROR FEXCP: supplied Exchange-correletion table'
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
!#define debug
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
! correction to the total energy
!=======================================================================
      XCENC=XCF

      RETURN
      END
