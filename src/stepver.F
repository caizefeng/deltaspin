#include "symbol.inc"
!*********************************************************************
! RCS:  $Id: stepver.F,v 1.2 2002/04/16 07:28:52 kresse Exp $
!
!  subroutine STEP performs a verlet step on the ions 
!
!  This part of the program uses slightly different reduced units:
!
!  particel coordinates are in direct lattice   range [0,1]
!  unit time     UT  =  POTIM * 1E-15
!    in reduced units one timestep for the ion movents takes 1 UT
!  unit energy   UE  = 1 eV
!
!  these parameters must be supplied by the main porgram
!  NIONS    number of IONS
!  A        direct lattice
!  TS       soll-temperature
!  POTIM    length of timestep
!  X        on entry: current positions  (time step n)
!           on exit:  updated positions  (time step n-1)
!  V        velocities of ions in r.u.
!           these are the velocities at the time step n-1/2
!           (on first call initial velocities)
!  D2C      D2C must contain half the accelations of the ions in r.u.
!           evaluated at the current positions (time step n)
!  SMASS    mass Parameter for nose dynamic, it has the dimension
!           of a Energy*time**2 = mass*length**2 and is supplied in
!           atomic mass * ULX**2 (this makes it  easy to define
!           the parameter)
!         <0     microcanonical ensemble
!                if ISCALE is 1 scale velocities to TS
!         -2     dont change velocities at all
!
!  execpt for initialisation only D2C must be recalculated by
!  the main program
!
!  Following varibles are used by step, and should not be changed
!  S        nose-parameter
!  D2       used as temporary work array to calculate consisten velocit.
!  D3C unused
!
!  following Quantities are return
!  XC       'exact positions' after the corrector step (C-step)
!           position are equal to those at the previous timestep
!  X        updated position of ions (see above)
!  EKIN     kinetic energy of system evaluated after C-step
!  EPS      kinetic energy of fictivious Nose-parameter after C-step
!  ES       potential energy of fictivious Nose-parameter after C-step
!  DISMAX   maximum distance the ions have moved in r.u.
!
! the standard verlet is given by : x(n+1)=x(n)+(x(n)-x(n-1))+a(n)
! this is equivalent to 1. order equation (velocity Verlet)
!           x(n+1)=x(n)+v(n)+a(n)
!           v(n+1)=v(n)+a(n)       with v(n)=x(n)-x(n-1)
!*********************************************************************

      SUBROUTINE STEP(INIT,ISCALE,NIONS,A,ANORM,D2C,SMASS,X,XC, &
     &      POTIM,POMASS,NTYP,ITYP,TS,V,D2,D3,S, &
     &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IU6)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION  X(3,NIONS),XC(3,NIONS), &
     &           V(3,NIONS),D2(3,NIONS),D3(3,NIONS)
      DIMENSION  ITYP(NIONS),POMASS(NTYP)
      DIMENSION  S(4)
      DIMENSION  A(3,3),ANORM(3)
!=======================================================================
!   Some important Parameters, to convert to UL,UT,UE
!   BOLK is the Bolzmannk.,BOLKEV is the Bolzmannk. in EV
!=======================================================================
      PARAMETER (EVTOJ=1.60217733E-19_q,AMTOKG=1.6605402E-27_q, &
     &           BOLK=1.380658E-23_q,BOLKEV=8.6173857E-5_q)

!=======================================================================
!   D2C ..are the calculated vectors for the accelerations
!=======================================================================
      DIMENSION  D2C(3,NIONS)

!=======================================================================
!   set unit length,unit time ...
!=======================================================================
      UL =1E-10_q*ANORM(1)
      UT =POTIM*1E-15_q
!=======================================================================
!   convert SMASS to more usefull units
!   SMASS is in         (atomic mass* UL**2)
!   what we want now is (eV*UT**2)
!
!=======================================================================
      FACT=(AMTOKG/EVTOJ)*(UL/UT)**2

      IF (SMASS>0) THEN
        SQQ=SMASS *FACT
      ELSE
        SQQ=1.0_q
        IF (SMASS==-2) THEN
! SMASS=-2 set D2C and D2 and D3 to 0 (simulate no accelerations at all)
        DO N=1,3
        DO  I=1,NIONS
          D2C(N,I)=0
          D2(N,I) =0
          D3(N,NIONS)=0
        ENDDO
        ENDDO
      ENDIF
      ENDIF

!=======================================================================
!   INIT=0 initialize variables and perform first step
!   copy D2C to     D2
!        X   to     XC
!   set  D3  to     0
!   also initialize S
!=======================================================================
      IF (INIT==0) THEN
        INIT=1
!-------use kinetic Energy to calculate initial values for S
        CALL EKINC(EKIN,NIONS,NTYP,ITYP,POMASS,POTIM,A,V)
        S(1)=1._q
        S(2)=0._q
        S(3)=0._q
        IF (SMASS>0) S(3)=(EKIN-NDEGREES_OF_FREEDOM*BOLKEV*TS/2)*S(1)/SQQ
        S(4)=0._q
      ENDIF
!=======================================================================
!   calculate velocitites kinetic energy and temperature at
!   current timestep
!   D2= v(current timestep) = v(n)+a(v)/2
!   this requires an iterative refinement, because a(v) depends on the
!   v(current timestep) in the Nose formulation
!=======================================================================
      DO N=1,3
      DO I=1,NIONS
         D2(N,I)=V(N,I)
      ENDDO
      ENDDO
      SVEL=S(2)

      DO 210 ITER=0,10
      IF (SMASS>0) THEN
        SC=SVEL/S(1)
      ELSE
        SC=0
      ENDIF
      CALL EKINC(EKIN,NIONS,NTYP,ITYP,POMASS,POTIM,A,D2)

      ERROR=0
      DO N=1,3
      DO I=1,NIONS
         ERROR=ERROR+(V(N,I)+D2C(N,I)-SC*0.5_q*D2(N,I)-D2(N,I))**2
         D2(N,I)=V(N,I)+D2C(N,I)-SC*0.5_q*D2(N,I)
      ENDDO
      ENDDO

      IF (SMASS>0) THEN
      ERROR=ERROR+ &
     &  (S(2)+S(1)*((EKIN-NDEGREES_OF_FREEDOM*BOLKEV*TS/2)/SQQ+0.5_q*SC**2)-SVEL)**2
      SVEL=S(2)+S(1)*((EKIN-NDEGREES_OF_FREEDOM*BOLKEV*TS/2)/SQQ+0.5_q*SC**2)
      ENDIF
      IF (SQRT(ERROR)<1E-10_q) GOTO 220
  210 CONTINUE
  220 CONTINUE
!=======================================================================
!   scale velocities if requested
!=======================================================================

      IF (ISCALE==1) THEN
      SCALE=SQRT(TS/(2*EKIN/(BOLKEV*NDEGREES_OF_FREEDOM)))
      IF (IU6>=0) &
      WRITE(IU6,11) SCALE
   11 FORMAT('scaling velocities: factor=',F10.4)

      DO I =1,3
      DO NI=1,NIONS
         V(I,NI) =V(I,NI)*SCALE
         D2(I,NI)=D2(I,NI)*SCALE
      ENDDO
      ENDDO

      CALL EKINC(EKIN,NIONS,NTYP,ITYP,POMASS,POTIM,A,D2)

      ENDIF
!*********************************************************************
!   final verlet step
!*********************************************************************
      IF (SMASS>0) THEN
        SC=SVEL/S(1)
        EPS=0.5_q*SQQ*(SVEL/S(1))**2
        ES =NDEGREES_OF_FREEDOM*BOLKEV*TS*LOG(S(1))
      ELSE
        SC=0
        EPS=0
        ES=0
      ENDIF

      DO N=1,3
      DO I=1,NIONS
         XC(N,I)=X(N,I)
         V(N,I) =V(N,I)+2*D2C(N,I)-SC*D2(N,I)
         X(N,I) =X(N,I)+V(N,I)
         IF(X(N,I)>=1._q) X(N,I)=X(N,I)-1._q
         IF(X(N,I)<0._q) X(N,I)=X(N,I)+1._q
      ENDDO
      ENDDO
      IF (SMASS>0) THEN
      S(2)=S(2)+S(1)*((2*EKIN-BOLKEV*TS*NDEGREES_OF_FREEDOM)/SQQ+SC**2)
      S(1)=S(1)+S(2)
      ENDIF
!=======================================================================
!   calculate the maximum distance moved by the ions
!=======================================================================

      DISMAX=0
      DO 120 I=1,NIONS
         DISMAX=MAX(DISMAX, &
     &        V(1,I)**2+ &
     &        V(2,I)**2+ &
     &        V(3,I)**2)
  120 CONTINUE

      DISMAX=SQRT(DISMAX)

!=======================================================================
!   now the routine drops back to MAIN
!   MAIN evaluates the accelerations D2C at the predictor
!   coordinates X
!   (these accelerations will serve as corrector-derivatives of second
!   order).
!=======================================================================

      RETURN
      END
