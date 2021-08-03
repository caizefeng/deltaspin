#include "symbol.inc"
!*********************************************************************
! RCS:  $Id: dyna.F,v 1.3 2001/05/10 13:52:31 kresse Exp $
!
!   EKINC calculates kinetic Energy
!   the conversion to the correct units is a little bit tricky,
!   but the result is simply in eV
!
!*********************************************************************

      SUBROUTINE EKINC(EKIN,NIONS,NTYP,ITYP,POMASS,POTIM,A,V)
      USE prec
      USE lattice
      USE ini
      USE constant

      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION V(3,NIONS),VTMP(3)
      DIMENSION A(3,3)
      DIMENSION  ITYP(NIONS),POMASS(NTYP)

      IF (POTIM==0) THEN
        EKIN=0.0_q
        RETURN
      ENDIF

!-----------------------------------------------------------------------
!   set unit length,unit time ...
!-----------------------------------------------------------------------
      UL=1E-10_q
      UT=POTIM*1E-15_q

!-----this factor converts  (atomic mass* UL**2/UT**2) to UE (eV)
      FACT=(AMTOKG/EVTOJ)*(UL/UT)**2
!-----------------------------------------------------------------------
!   convert to kartesian coordinetes and calculated EKIN
!-----------------------------------------------------------------------
      EKIN=0
      DO NI=1,NIONS
        VTMP(1)=   V(1,NI)
        VTMP(2)=   V(2,NI)
        VTMP(3)=   V(3,NI)
        CALL  DIRKAR(1,VTMP,A)
        NT=ITYP(NI)
        EKIN=EKIN+ (VTMP(1)**2+VTMP(2)**2+VTMP(3)**2)*POMASS(NT)
!       WRITE(*,'("ekin:", I4,F10.5)') NI,(VTMP(1)**2+VTMP(2)**2+VTMP(3)**2)*POMASS(NT)*FACT/2._q
      ENDDO
      EKIN= EKIN*FACT/2._q
      RETURN
      END



!*********************************************************************
!  subroutine INITIO initializes the particles velocities,
!    and if required particles positions
!  NIONS  number of ions
!  TEMP   soll temperature
!  X      positions
!  V      velocities
!  INIT   if 1 calculate position of articles too
!
!*********************************************************************

      SUBROUTINE INITIO(NIONS,LSDYN,NDEGREES_OF_FREEDOM,NTYP,ITYP,TEMP, &
                        POMASS,POTIM,X,V,LSFOR,A,B,INIT,COMM,IU)
      USE prec
      USE lattice
      USE ini
      USE constant

      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION X(3,NIONS),V(3,NIONS)
      DIMENSION A(3,3),B(3,3)
      DIMENSION ITYP(NIONS),POMASS(NTYP)
      LOGICAL   LSDYN,LSFOR(3,NIONS)

      REAL(q) RNULL,BMP

      RNULL=0._q

!=======================================================================
!   set unit length,unit time ...
!=======================================================================
      UL =1E-10_q
      UT =POTIM*1E-15_q
!=======================================================================
!  calculate the mean deviations of the maxwell-bolzmann-disribution
!  once again there is the magic scaling factor of above
!=======================================================================

!-----These Factors convert  (atomic mass* UL**2/UT**2) to UE (eV)
      FACT= (AMTOKG/EVTOJ)*(UL/UT)**2

!=======================================================================
!  if you want to place particles too INIT must be 1
!  find out if IONS is a magic number
!  possible starting configurations are in ascending order:
!    fcc    magic number is 4*n**3   NPEC=4
!    bcc    magic number is 2*n**3   NPEC=2
!     sc    magic number is   n**3   NPEC=1
!  NPEC is the number of particles per unit cell
!  M    number of unit cells in each direction
!=======================================================================
      NPEC=1
      M=NINT((REAL( NIONS ,KIND=q) /NPEC)**(1._q/3._q)) +1

      IF (INIT==1) THEN

      NPEC=4
      M=NINT((REAL( NIONS ,KIND=q) /NPEC)**(1._q/3._q))
      IF (NPEC*M**3 == NIONS) GOTO 100
      NPEC=2
      M=NINT((REAL( NIONS ,KIND=q) /NPEC)**(1._q/3._q))
      IF (NPEC*M**3 == NIONS) GOTO 100
      NPEC=1
      M=NINT((REAL( NIONS ,KIND=q) /NPEC)**(1._q/3._q))
      IF (NPEC*M**3 == NIONS) GOTO 100

      NPEC=0

  100 CONTINUE
      IF (IU>=0) THEN
         IF (NPEC==4) WRITE(IU,1)
         IF (NPEC==2) WRITE(IU,2)
         IF (NPEC==1) WRITE(IU,3)
         IF (NPEC==0) WRITE(IU,4)
    1 FORMAT(' Initial configuration is fcc')
    2 FORMAT(' Initial configuration is bcc')
    3 FORMAT(' Initial configuration is sc')
    4 FORMAT(' NIONS doesn`t corespond to fcc,bcc or sc'/ &
     &       'initial configuration will be based on sc-lattice')
      ENDIF
      IF (NPEC==0) THEN
        NPEC=1
        M=NINT((REAL( NIONS ,KIND=q) /NPEC)**(1._q/3._q)) +1
      ENDIF
      ENDIF

!=======================================================================
!  KP =  actual total number of particles that have already been
!            placed
!  PP = length of an elementary cell
!=======================================================================
      PP=1._q/ REAL( M ,KIND=q)
      PPH=PP/2._q

!=======================================================================
!     XX0,YY0,ZZ0 = coordinates of the first particle of each lattice:
!     NPEC=4     (0,0,0) (0,pph,pph), (pph,pph,0) or (pph,0,pph)
!     NPEC=2     (0,0,0) (pph,pph,pph)
!     NPEC=1     (0,0,0)
!=======================================================================
      KP=0
      DO LIA=1,NPEC
         IF (NPEC==4) THEN
           XX0=AINT(LIA/2.5_q)*PPH
           YY0=(INT(LIA/2._q)-2*INT(LIA/4._q))*PPH
           ZZ0=(REAL( (-1)**LIA+1 ,KIND=q) /2._q)*PPH
         ELSE IF (NPEC==2) THEN
           XX0=(LIA-1)*PPH
           YY0=(LIA-1)*PPH
           ZZ0=(LIA-1)*PPH
         ELSE IF (NPEC==0) THEN
           XX0=0
           YY0=0
           ZZ0=0
         ENDIF

         DO IX=0,M-1
         DO IY=0,M-1
         DO IZ=0,M-1

           KP=KP+1
           IF(KP>NIONS) GOTO 300
           IF (INIT==1) THEN
             X(1,KP)=XX0+IX*PP
             X(2,KP)=YY0+IY*PP
             X(3,KP)=ZZ0+IZ*PP
           ENDIF

           NT=ITYP(KP)
           BMP=SQRT(TEMP*BOLKEV/(POMASS(NT)*FACT))

           ! note that this will always create the same sequence of
           ! random numbers
           V(1,KP)=RANG(RNULL,BMP)
           V(2,KP)=RANG(RNULL,BMP)
           V(3,KP)=RANG(RNULL,BMP)
#ifdef tbdyn
           ! for tbdyn we use RANE_ION which is properly seeded
           V(1,KP)=RANG_RANDOM_NUMBER(RNULL,BMP)
           V(2,KP)=RANG_RANDOM_NUMBER(RNULL,BMP)
           V(3,KP)=RANG_RANDOM_NUMBER(RNULL,BMP)
#endif
         ENDDO
         ENDDO
         ENDDO
      ENDDO
!=======================================================================
!  if you dont have a magic number, you will end up here
!=======================================================================
  300 CONTINUE
      ! send velocities to all cores 
      CALLMPI( M_bcast_d( COMM, V, SIZE(V)) )
!=======================================================================
!  convert from cartesian koordinates to direct lattice
!  remove any spurious drift and
!  make one scaling step
!=======================================================================
      CALL  KARDIR(NIONS,V,B)

      ! no selective dynamic: remove the overall drift from the forces
      IF (.NOT. LSDYN) THEN
         CALL SYMVEL(NIONS,NTYP,ITYP,POMASS,X,V,A,B)
         CALL SYMVEL(NIONS,NTYP,ITYP,POMASS,X,V,A,B)
      ELSE
         DO I =1,3
         DO NI=1,NIONS
            IF (.NOT.LSFOR(I,NI)) V(I,NI)=0
         ENDDO
         ENDDO
         
      ENDIF

      CALL EKINC(EKIN,NIONS,NTYP,ITYP,POMASS,POTIM,A,V)

      IF (EKIN==0) THEN
         SCALE=0
      ELSE
         SCALE=SQRT(TEMP*NDEGREES_OF_FREEDOM/(2*EKIN/BOLKEV))
      ENDIF
!
! avoid scaling 
!
!      IF (IU>=0) WRITE(IU,*) 'initial velocities are not scaled'
!      SCALE=1

      DO I =1,3
      DO NI=1,NIONS
         V(I,NI)=V(I,NI)*SCALE
      ENDDO
      ENDDO

      RETURN
      END

!
! This subroutine produces a normal distribution (Gaussian distribution)
! using the built in fortran random number generator
! which is initialized with a random seed for tbdyn
!

      FUNCTION RANG_RANDOM_NUMBER(RNULL,WIDTH)
      USE prec
      USE random_seeded

      IMPLICIT REAL(q) (A-H,O-Z)

      REAL(q) RNULL, WIDTH, TWOPI, RANG
      PARAMETER ( TWOPI = 6.283185307179586_q )
      REAL(q), PARAMETER :: myTiny=1.E-8_q
      REAL(q) :: num1, num2

      CALL RANE_ION(num1)
      CALL RANE_ION(num2)

      !c RANDOM_NUMBER generates x such that 0<=x<1
      !c but zero in LOG is a problem hence we must check RANE(num2)
      !c (RANE(num1) is guaranteed to be OK) 
      DO
        IF (num2 .GT. myTiny) EXIT
        CALL RANE_ION(num2)
      ENDDO
      
      RANG_RANDOM_NUMBER = COS( TWOPI*num1 ) * SQRT( -2._q*LOG(num2) )
      RANG_RANDOM_NUMBER = WIDTH * RANG_RANDOM_NUMBER  +  RNULL

      RETURN
      END


!*************************** SYMVEC ***********************************
!  this subroutine removes any drift from a supplied vector
!  if only one ion no shift is removed
!*********************************************************************

      SUBROUTINE SYMVEC(NIONS,V)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION V(3,NIONS)

      DX=0
      DY=0
      DZ=0
      IF (NIONS==1) RETURN

      DO 100 N=1,NIONS
        DX=DX+V(1,N)
        DY=DY+V(2,N)
        DZ=DZ+V(3,N)
  100 CONTINUE

      DX=-DX/NIONS
      DY=-DY/NIONS
      DZ=-DZ/NIONS

      DO 200 N=1,NIONS
        V(1,N)=V(1,N)+DX
        V(2,N)=V(2,N)+DY
        V(3,N)=V(3,N)+DZ
  200 CONTINUE
      RETURN
      END


!*************************** SYMVEL **********************************
!  this subroutine removes any drift from the velocities
!
!*********************************************************************

      SUBROUTINE SYMVEL(NIONS,NTYP,ITYP,POMASS,X,V,A,B)
      USE prec
      USE lattice
      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION V(3,NIONS),X(3,NIONS)
      DIMENSION A(3,3),B(3,3)
      DIMENSION  ITYP(NIONS),POMASS(NTYP)
      DIMENSION  W(3),POS(3),CENTER(3),TMP(3),TMP2(3),TMP3(3),D(3,3)
      DIMENSION  AMYCEN(3),IPIV(3)

      DX=0
      DY=0
      DZ=0
      AVERAGE=0
      IF (NIONS.EQ.1) RETURN
!
! set here center of BOX
      AMYCEN(1)=0
      AMYCEN(2)=0
      AMYCEN(3)=0

      DO J=1,3
        CENTER(J)=0
        W(J)=0
        TMP(J)=0
        DO I=1,3
          D(I,J)=0
      ENDDO
      ENDDO

      DO N=1,NIONS
         NT=ITYP(N)
         AVERAGE=AVERAGE+POMASS(NT)
         DO  J=1,3
            POS(J)=MOD(X(J,N)-AMYCEN(J)+60.5,1._q)-0.5
            TMP(J)=TMP(J)+V(J,N)*POMASS(NT)
            CENTER(J)=CENTER(J)+POS(J)*POMASS(NT)
         ENDDO
      ENDDO

      DO J=1,3
         TMP(J)=-TMP(J)/AVERAGE
         CENTER(J)=CENTER(J)/AVERAGE
      ENDDO

      DO N=1,NIONS
         DO J=1,3
            V(J,N)=V(J,N)+TMP(J)
         ENDDO
      ENDDO
! uncomment return for zero rotation
      RETURN

!
! remove rotational degrees of freedom with respect to AMYCEN
! !!! MIND: works only for a cubic box (I was lazy)  !!!
!     L= r x v m (store -L in W)
      DO N=1,NIONS
         DO J=1,3
            POS(J)=MOD(X(J,N)-AMYCEN(J)+60.5,1._q)-0.5-CENTER(J)
            TMP3(J)=V(J,N)
         ENDDO
         CALL DIRKAR(1,POS,A)
         CALL DIRKAR(1,TMP3,A)

         CALL EXPRO(TMP,POS,TMP3)
         NT=ITYP(N)
         DO J=1,3
            W(J)=W(J)-TMP(J)*POMASS(NT)
         ENDDO
!       D = r x e x r
         DO J=1,3
            DO  I=1,3
               TMP(I)=0
            ENDDO
            TMP(J)=1
            CALL EXPRO(TMP2,POS,TMP)
            CALL EXPRO(TMP,TMP2,POS)
            DO I=1,3
               D(I,J)=D(I,J)+TMP(I)*POMASS(NT)
            ENDDO
         ENDDO
      ENDDO
!
! if you want to set to zero only for one component
! uncomment the corresponding line
      W(1)=0
      W(2)=0
!      W(3)=0
!   solve D w = -L
      INFO=0
      CALL DGETRF( 3, 3, D, 3, IPIV, INFO )
      CALL DGETRS('N', 3, 1, D, 3, IPIV, W, 3, INFO)

      DO N=1,NIONS
         DO J=1,3
            POS(J)=MOD(X(J,N)-AMYCEN(J)+60.5,1._q)-0.5-CENTER(J)
         ENDDO
         CALL DIRKAR(1,POS,A)
         CALL EXPRO(TMP,W,POS)
         CALL KARDIR(1,TMP,B)
         DO J=1,3
            V(J,N)=V(J,N)+TMP(J)
         ENDDO
      ENDDO
      RETURN
      END


!*************************** CHECK ***********************************
!   this subroutine checks the consistency of forces and total energy
!   it needs the position of the ions POSION, the forces according
!   to these positions and the total Energy
!
!*********************************************************************

      SUBROUTINE CHECK(NIONS,POSION,TIFOR,EWIFOR,TOTEN,TEWEN,A,IU)
      USE prec
      USE lattice
      USE ini

      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION POSION(3,NIONS)
      DIMENSION TIFOR(3,NIONS),EWIFOR(3,NIONS)

      INTEGER, SAVE :: NIOND=-1
      REAL(q), ALLOCATABLE, SAVE :: POSIOL(:,:),TIFORL(:,:),EWIFOL(:,:)
      DIMENSION TMP(3)
      DIMENSION A(3,3)
      REAL(q), SAVE :: TOTENL=0,TEWENL=0

      IF (NIOND==-1) THEN
         NIOND=NIONS
         ALLOCATE(POSIOL(3,NIOND),TIFORL(3,NIOND),EWIFOL(3,NIOND))
      ENDIF

      IF (NIOND<NIONS) THEN
        IF (IU>=0) WRITE(IU,*)'WARNING: CHECK: NIOND is too small'
        RETURN
      ENDIF

      IF (TOTENL==0) GOTO 200

      ED1=0
      ED2=0
      EDEW1=0
      EDEW2=0

      DO 50 NI=1,NIONS
        TMP(1)=MOD(POSION(1,NI)-POSIOL(1,NI)+1.5_q,1._q)-.5_q
        TMP(2)=MOD(POSION(2,NI)-POSIOL(2,NI)+1.5_q,1._q)-.5_q
        TMP(3)=MOD(POSION(3,NI)-POSIOL(3,NI)+1.5_q,1._q)-.5_q
        CALL DIRKAR(1,TMP,A)
        ED1=ED1 &
     &   +(TMP(1)*TIFOR(1,NI)+TMP(2)*TIFOR(2,NI)+TMP(3)*TIFOR(3,NI))
        ED2=ED2 &
     &   +(TMP(1)*TIFORL(1,NI)+TMP(2)*TIFORL(2,NI)+TMP(3)*TIFORL(3,NI))

        EDEW1=EDEW1 &
     &   +(TMP(1)*EWIFOR(1,NI)+TMP(2)*EWIFOR(2,NI)+TMP(3)*EWIFOR(3,NI))
        EDEW2=EDEW2 &
     &   +(TMP(1)*EWIFOL(1,NI)+TMP(2)*EWIFOL(2,NI)+TMP(3)*EWIFOL(3,NI))
  50  CONTINUE

      IF (IU>=0) THEN
      WRITE(IU,1)(ED1+ED2)/2,ED1,ED2,TOTENL-TOTEN, &
     &           (ED1+ED2)/2-(TOTENL-TOTEN)
  1   FORMAT(' d Force =' ,E14.7,'[',E10.3,',',E10.3, &
     &       ']  d Energy =',E14.7,E10.3)
      WRITE(IU,2)(EDEW1+EDEW2)/2,EDEW1,EDEW2,TEWENL-TEWEN, &
     &           (EDEW1+EDEW2)/2-(TEWENL-TEWEN)
  2   FORMAT(' d Force =' ,E14.7,'[',E10.3,',',E10.3, &
     &       ']  d Ewald  =',E14.7,E10.3)
      ENDIF

!=======================================================================
!  copy new Positions and forces to store
!=======================================================================

  200 CONTINUE

      TOTENL=TOTEN
      TEWENL=TEWEN

      DO I=1,3
      DO NI=1,NIONS
         POSIOL(I,NI)=POSION(I,NI)
         TIFORL(I,NI)=TIFOR (I,NI)
         EWIFOL(I,NI)=EWIFOR(I,NI)
      ENDDO
      ENDDO

      RETURN
      END


!*********************************************************************
! subroutine IONCGR
! subroutine performes a steepest descent/ conjugate gradient step
! on the ions and the cell
! it contains considerable heuristic to make the minimization
! efficient and
! uses a varient of Brents algorithm from numerical recipies
! for the line minimization
! especially the question how long a line minimization should be
! continued is of requires a lot of fiddling
!
! IFLAG
!   on call
!           0  initial trial step (steepest descent)
!           1  initial trial step (conjugate gradient)
!           2  move ions to minimum
!   on exit
!           0  currently line optimization is done
!           1  new trial step has been performed, main can break
!           2  energy is possibly converged
! F         contains scaled forces on ions  divided by mass of ions
! FACT      scaling factor of forces
! FSIF      stress (in cartesian units) scaled by FACTSI
! FACTSI    scaling factor for stress
! POSION    coordinates of ions
! A,B       direct and reciprocal lattice
!
! FL        used to store old forces  (set by IONCGR)
! S         step  into which search is performed    (set by IONCGR)
! POSIONC   old coordinates (set by IONCGR)
! POSIONOLD coordinates that were not recalculated to be in the unit cell
! some things which are used in our heuristics:
! EBREAK    absolut accuracy of energies (should be set by caller)
!           determines whether cubic interpolation is used
! EDIFFG    required accuracy for energy (>0)
! LSTOP2    external routine can signal that a break condition was met
!
! routine modified by Robin Hirschl in Oct. 2000 to avoid
! too large step when search was started near a saddle point
!*********************************************************************

      SUBROUTINE IONCGR(IFLAG,NIONS,TOTEN,A,B,NFREE,POSION,POSIOC, &
     &      FACT,F,FACTSI,FSIF,FL,S,DISMAX,IU6,IU0, &
     &      EBREAK,EDIFFG,E1TEST,LSTOP2)
      USE prec
      USE lattice
      USE ini
      USE chain

      IMPLICIT REAL(q) (A-H,O-Z)

      DIMENSION F(3,NIONS),FL(3,NIONS),S(3,NIONS)
      DIMENSION POSION(3,NIONS),POSIOC(3,NIONS)
      DIMENSION A(3,3),B(3,3)
      DIMENSION TMP(3)
      LOGICAL   LRESET,LTRIAL,LBRENT
      LOGICAL   LSTOP2

      SAVE  AC,FSIFL,SSIF
      DIMENSION AC(3,3),FSIFL(3,3),SSIF(3,3),FSIF(3,3)

      SAVE TOTEN1,DMOVED,E1ORD1,ORTH,GAMMA,GNORM,GNORMF,GNORML
      SAVE GNORM1,GNORM2,STEP,CURVET,SNORM,SNORMOLD
      SAVE DMOVEL,LTRIAL,LBRENT
      DATA ICOUNT/0/, ICOUNT2/0/, STEP /1._q/,SNORM/1E-10_q/
      DATA LTRIAL/.FALSE./

!=======================================================================
!  if IFLAG =0 initialize everything
!=======================================================================
      IF (IFLAG==0) THEN
        DO NI=1,NIONS
        DO I=1,3
          S(I,NI) =0
          FL(I,NI)=0
        ENDDO
        ENDDO

        DO I=1,3
        DO J=1,3
          SSIF (I,J)=0
          FSIFL(I,J)=0
        ENDDO
        ENDDO
        LTRIAL=.FALSE.
        ICOUNT=0
        SNORM =1E-10_q
        SNORMOLD =1E10_q
        CURVET=0
      ENDIF

! if previous step was a trial step then continue with line minimization
      IF (LTRIAL) THEN
        GOTO 400
      ENDIF
!=======================================================================
!  calculate quantities necessary to conjugate directions
!=======================================================================
      GNORML=0
      GNORM =0
      GNORMF=0
      ORTH  =0
      IF (FACT/=0) THEN
      DO NI=1,NIONS
         GNORML = GNORML+1._q/FACT* &
     &    (FL (1,NI)*FL (1,NI)+FL(2,NI) *FL(2,NI) +FL(3,NI) *FL(3,NI))
         GNORM  = GNORM+ 1._q/FACT*( &
     &    (F(1,NI)-FL(1,NI))*F(1,NI) &
     &   +(F(2,NI)-FL(2,NI))*F(2,NI) &
     &   +(F(3,NI)-FL(3,NI))*F(3,NI))
         GNORMF = GNORMF+1._q/FACT* &
     &    (F(1,NI)*F(1,NI)+F(2,NI)*F(2,NI)+F(3,NI)*F(3,NI))
         ORTH   = ORTH+  1._q/FACT* &
     &    (F(1,NI)*S(1,NI)+F(2,NI)*S(2,NI)+F(3,NI)*S(3,NI))
      ENDDO
      ENDIF

      GNORM1=GNORMF

      IF (FACTSI/=0) THEN
      DO I=1,3
      DO J=1,3
         GNORML=GNORML+ FSIFL(I,J)*FSIFL(I,J)/FACTSI
         GNORM =GNORM +(FSIF(I,J)-FSIFL(I,J))*FSIF(I,J)/FACTSI
         GNORMF=GNORMF+ FSIF(I,J)* FSIF(I,J)/FACTSI
         ORTH  =ORTH  + FSIF(I,J)* SSIF(I,J)/FACTSI
      ENDDO
      ENDDO
      ENDIF
      GNORM2=GNORMF-GNORM1

      CALLMPI_C( sum_chain( GNORM ))
      CALLMPI_C( sum_chain( GNORML ))
      CALLMPI_C( sum_chain( GNORMF))
      CALLMPI_C( sum_chain( GNORM1))
      CALLMPI_C( sum_chain( GNORM2))
      CALLMPI_C( sum_chain( ORTH ))

!=======================================================================
!  calculate Gamma
!  improve line optimization if necessary
!=======================================================================
      IF (IFLAG==0) THEN
        ICOUNT=0
        GAMMA=0
      ELSE
!       this statement for Polak-Ribiere
        GAMMA=GNORM /GNORML
!       this statement for Fletcher-Reeves
!        GAMMA=GNORMF/GNORML
      ENDIF

      GAMMIN=1._q
      IFLAG=1
      IF (IU0>=0) &
      WRITE(IU0,30)CURVET,CURVET*GNORMF,CURVET*(ORTH/SQRT(SNORM))**2

   30 FORMAT(' curvature: ',F6.2,' expect dE=',E10.3,' dE for cont linesearch ',E10.3)
! required accuracy not reached in line minimization
! improve line minimization
! several conditions must be met:

! orthonormality not sufficient
!      WRITE(0,*) ORTH, MAX(GAMMA,GAMMIN),GNORMF/5, ABS(CURVET*(ORTH/SQRT(SNORM))**2), LSTOP2
      IF (ABS(ORTH)*MAX(GAMMA,GAMMIN)>ABS(GNORMF)/5 &
! expected energy change along line search must be larger then required accuracy
     &    .AND. &
     &   ( (EDIFFG>0 .AND. &
     &       ABS(CURVET*(ORTH/SQRT(SNORM))**2)>EDIFFG) &
! or force must be large enough that break condition is not met
     &    .OR. &
     &     (EDIFFG<0 .AND..NOT.LSTOP2) &
     &   ) &
! last call must have been a line minimization
     &    .AND. LBRENT &
     &  ) GOTO 400

!---- improve the trial step by adding some amount of the optimum step
      IF (ICOUNT/=0) STEP=STEP+0.2_q*STEP*(DMOVEL-1)
!---- set GAMMA to zero if line minimization was not sufficient
      IF (5*ABS(ORTH)*GAMMA>ABS(GNORMF)) THEN
         GAMMA=0
         ICOUNT=0
      ENDIF
!---- if GNORM is very small signal calling routine to stop
      IF (CURVET/=0 .AND.ABS((GNORMF)*CURVET*2)<EDIFFG) THEN
        IFLAG=2
      ENDIF

      ICOUNT=ICOUNT+1
      ICOUNT2=ICOUNT2+1
!-----------------------------------------------------------------------
! performe trial step
!-----------------------------------------------------------------------
      E1ORD1=0
      DMOVED=0
      SNORM =1E-10_q

!----- set GAMMA to zero for initial steepest descent steps (Robin Hirschl)
!      have to discuss this
!      IF (ICOUNT2<=NFREE) GAMMA=0
      
      DO NI=1,NIONS
!----- store last gradient
        FL(1,NI)=F(1,NI)
        FL(2,NI)=F(2,NI)
        FL(3,NI)=F(3,NI)
!----- conjugate the direction to the last direction
        S(1,NI) = F(1,NI)+ GAMMA   * S(1,NI)
        S(2,NI) = F(2,NI)+ GAMMA   * S(2,NI)
        S(3,NI) = F(3,NI)+ GAMMA   * S(3,NI)

        IF (FACT/=0) THEN
        SNORM = SNORM  +  1/FACT * &
     &  (S(1,NI)*S(1,NI)+ S(2,NI)*S(2,NI) + S(3,NI)*S(3,NI))
        ENDIF
      ENDDO

      DO I=1,3
         DO J=1,3
            FSIFL(I,J)=FSIF(I,J)
            AC(I,J)   = A(I,J)
        SSIF(I,J) = FSIF(I,J)+ GAMMA* SSIF(I,J)
      ENDDO
      ENDDO

      IF (FACTSI/=0) THEN
         DO I=1,3
            DO J=1,3
               SNORM = SNORM  + 1/FACTSI *      SSIF(I,J)* SSIF(I,J)
            ENDDO
         ENDDO
      ENDIF

      CALLMPI_C( sum_chain( SNORM))

!----- if SNORM increased, rescale STEP (to avoid too large trial steps) 
!      (Robin Hirschl)
      IF (SNORM>SNORMOLD) THEN
         STEP=STEP*(SNORMOLD/SNORM)
      ENDIF
      SNORMOLD=SNORM
      
      DO NI=1,NIONS
!----- search vector from cartesian to direct lattice
         TMP(1) = S(1,NI)
         TMP(2) = S(2,NI)
         TMP(3) = S(3,NI)
         CALL KARDIR(1,TMP,B)

!----- trial step in direct grid
         POSIOC(1,NI)= POSION(1,NI)
         POSIOC(2,NI)= POSION(2,NI)
         POSIOC(3,NI)= POSION(3,NI)

         POSION(1,NI)= TMP(1)*STEP+POSIOC(1,NI)
         POSION(2,NI)= TMP(2)*STEP+POSIOC(2,NI)
         POSION(3,NI)= TMP(3)*STEP+POSIOC(3,NI)
         DMOVED= MAX( DMOVED,S(1,NI)*STEP,S(2,NI)*STEP,S(3,NI)*STEP)

!----- keep ions in unit cell (Robin Hirschl)
         POSION(1,NI)= MOD(POSION(1,NI),1._q)
         POSION(2,NI)= MOD(POSION(2,NI),1._q)
         POSION(3,NI)= MOD(POSION(3,NI),1._q)

!----- force * trial step = 1. order energy change
         IF (FACT/=0) THEN
            E1ORD1= E1ORD1 - 1._q * STEP / FACT * &
     &           (S(1,NI)*F(1,NI)+ S(2,NI)*F(2,NI) + S(3,NI)*F(3,NI))
         ENDIF
      ENDDO

      IF (FACTSI/=0) THEN
      DO I=1,3
      DO J=1,3
         E1ORD1= E1ORD1 - STEP / FACTSI * SSIF(I,J)* FSIF(I,J)
      ENDDO
      ENDDO
      ENDIF

      DO J=1,3
         DO I=1,3
            A(I,J)=AC(I,J)
            DO K=1,3
               A(I,J)=A(I,J) + SSIF(I,K)*AC(K,J)*STEP
            ENDDO
         ENDDO
      ENDDO

      CALLMPI_C( sum_chain( E1ORD1 ))

      LRESET = .TRUE.
      X=0
      Y=TOTEN
      FP=E1ORD1
      IFAIL=0

      CALL ZBRENT(IU0,LRESET,EBREAK,X,Y,FP,XNEW,XNEWH,YNEW,YD,IFAIL)
      DMOVEL=1

      IF (IU0>=0) THEN
         WRITE(IU0,10) GAMMA,GNORM1,GNORM2,ORTH,STEP
 10      FORMAT(' trial: gam=',F8.5,' g(F)= ',E10.3, &
     &        ' g(S)= ',E10.3,' ort =',E10.3,' (trialstep =',E10.3,')')
         WRITE(IU0,11) SNORM
 11      FORMAT(' search vector abs. value= ',E10.3)
      ENDIF
      TOTEN1= TOTEN
      E1TEST=E1ORD1
      LTRIAL=.TRUE.
      RETURN
!=======================================================================
! calculate optimal step-length and go to the minimum
!=======================================================================
!-----------------------------------------------------------------------
!  1. order energy change due to displacement at the new position
!-----------------------------------------------------------------------
  400 CONTINUE
      E1ORD2=0
      IF (FACT/=0) THEN
      DO NI=1,NIONS
        E1ORD2= E1ORD2 - 1._q * STEP / FACT * &
     &  (S(1,NI)*F(1,NI)+ S(2,NI)*F(2,NI) + S(3,NI)*F(3,NI))
      ENDDO
      ENDIF

      IF (FACTSI/=0) THEN
      DO I=1,3
      DO J=1,3
        E1ORD2= E1ORD2 - STEP / FACTSI * SSIF(I,J)* FSIF(I,J)
      ENDDO
      ENDDO
      ENDIF

      CALLMPI_C( sum_chain( E1ORD2 ))
!-----------------------------------------------------------------------
!  calculate position of minimum
!-----------------------------------------------------------------------
      CONTINUE
      LRESET = .FALSE.
      X=DMOVEL
      Y=TOTEN
      FP=E1ORD2
      IFAIL=0

      CALL ZBRENT(IU0,LRESET,EBREAK,X,Y,FP,XNEW,XNEWH,YNEW,YD,IFAIL)
!     estimate curvature
      CURVET=YD/(E1ORD2/STEP/SQRT(SNORM))**2

      DMOVE =XNEW
      DMOVEH=XNEWH

!    previous step was trial step than give long output
      IF (LTRIAL) THEN
      LBRENT=.TRUE.
      E2ORD  = TOTEN-TOTEN1
      E2ORD2 = (E1ORD1+E1ORD2)/2
      IF (IU0>=0) &
      WRITE(IU0,45) E2ORD,E2ORD2,E1ORD1,E1ORD2
   45 FORMAT(' trial-energy change:',F12.6,'  1 .order',3F12.6)

      E1TEST=TOTEN1-YNEW
      DISMAX=DMOVE*DMOVED
      IF (IU0>=0) &
      WRITE(IU0,20) DMOVE*STEP,DMOVEH*STEP,DISMAX,YNEW,YNEW-TOTEN1
   20 FORMAT(' step: ',F8.4,'(harm=',F8.4,')', &
     &       '  dis=',F8.5,'  next Energy=',F13.6, &
     &       ' (dE=',E10.3,')')

      IF (GAMMA==0) THEN
       IF (IU6>=0) &
        WRITE(IU6,*)'Steepest descent step on ions:'
      ELSE
      IF (IU6>=0) &
        WRITE(IU6,*)'Conjugate gradient step on ions:'
      ENDIF

      IF (IU6>=0) &
      WRITE(IU6,40) E2ORD,(E1ORD1+E1ORD2)/2,E1ORD1,E1ORD2, &
     &         GNORM,GNORMF,GNORML, &
     &         GNORM1,GNORM2,ORTH,GAMMA,STEP, &
     &         DMOVE*STEP,DMOVEH*STEP,DISMAX,YNEW,YNEW-TOTEN1

   40 FORMAT(' trial-energy change:',F12.6,'  1 .order',3F12.6/ &
     &       '  (g-gl).g =',E10.3,'      g.g   =',E10.3, &
     &       '  gl.gl    =',E10.3,/ &
     &       ' g(Force)  =',E10.3,'   g(Stress)=',E10.3, &
     &       ' ortho     =',E10.3,/ &
     &       ' gamma     =',F10.5,/ &
     &       ' trial     =',F10.5,/ &
     &       ' opt step  =',F10.5,'  (harmonic =',F10.5,')', &
     &       ' maximal distance =',F10.8/ &
     &       ' next E    =',F13.6,'   (d E  =',F10.5,')')
      ELSE
      IF (IU0>=0) &
      WRITE(IU0,25) DMOVE*STEP,YNEW,YNEW-TOTEN1
   25 FORMAT(' opt : ',F8.4,'  next Energy=',F13.6, &
     &       ' (dE=',E10.3,')')
!    do not make another call to ZBRENT if reuqired accuracy was reached
      IF (ABS(YD)<EDIFFG) LBRENT=.FALSE.
      ENDIF
!-----------------------------------------------------------------------
!    move ions to the minimum
!-----------------------------------------------------------------------
      DO NI=1,NIONS
!----- search vector from cartesian to direct lattice
        TMP(1) = S(1,NI)
        TMP(2) = S(2,NI)
        TMP(3) = S(3,NI)
        CALL KARDIR(1,TMP,B)

        POSIOC(1,NI)= POSION(1,NI)
        POSIOC(2,NI)= POSION(2,NI)
        POSIOC(3,NI)= POSION(3,NI)

        POSION(1,NI)= TMP(1)*(DMOVE-DMOVEL)*STEP+POSIOC(1,NI)
        POSION(2,NI)= TMP(2)*(DMOVE-DMOVEL)*STEP+POSIOC(2,NI)
        POSION(3,NI)= TMP(3)*(DMOVE-DMOVEL)*STEP+POSIOC(3,NI)

!----- keep ions in unit cell (Robin Hirschl)
        POSION(1,NI)=MOD(POSION(1,NI),1._q)
        POSION(2,NI)=MOD(POSION(2,NI),1._q)
        POSION(3,NI)=MOD(POSION(3,NI),1._q)
        
      ENDDO

      DO J=1,3
      DO I=1,3
      A(I,J)=AC(I,J)
      DO K=1,3
        A(I,J)=A(I,J) + SSIF(I,K)*AC(K,J)*DMOVE*STEP
      ENDDO
      ENDDO
      ENDDO
      DMOVEL=DMOVE
      IFLAG=0
      LTRIAL=.FALSE.

      RETURN

      END

!*********************************************************************
!
! subroutine ION_VEL_QUENCH
! subroutine uses a damped second order equation of motion
! velocities are zeroed out if the force is antiparallel to the
! velocity
! changed to QUICKMIN algorithm as suggested by Hannes Jonsson
!
!*********************************************************************

      SUBROUTINE ION_VEL_QUENCH(NIONS,A,B,IU6,IU0,LSDYN, &
            POSION,POSIOC,FACT,F,FACTSI,FSIF,VEL,E1TEST)
      USE prec
      USE lattice
      USE ini
      USE chain

      IMPLICIT NONE
      INTEGER NIONS,IU6,IU0
      REAL(q) F(3,NIONS),VEL(3,NIONS)
      REAL(q) POSION(3,NIONS),POSIOC(3,NIONS)
      REAL(q) A(3,3),B(3,3),FSIF(3,3)
      REAL(q) FACT,FACTSI,E1TEST
      LOGICAL LSDYN
! local static
      REAL(q), SAVE :: SIFVEL(3,3)
      INTEGER, SAVE :: IFLAG=0
! local temporary
      REAL(q) TMP(3),AC(3,3)
      INTEGER :: NI,I,J,K
      REAL(q) PROJ,NORM,GNORM1,GNORM2

!=======================================================================
!  if IFLAG =0 initialize everything
!=======================================================================
      IF (IFLAG==0) THEN

         VEL=0
         SIFVEL=0
         IFLAG=1

      ENDIF
!=======================================================================
!  this is really simple
!=======================================================================
      GNORM1=0; GNORM2=0

      IF (FACT/=0) THEN
         TMP=SUM(F,DIM=2)

         VEL=VEL+F
!gK modifications to adopt QUICKMIN
!         DO NI=1,NIONS
!            PROJ=SUM(VEL(:,NI)*F(:,NI))
!            NORM=SUM(F(:,NI)*F(:,NI))
!
!            GNORM1=GNORM1+NORM/FACT
!            ! zero out those forces that are antiparallel
!            IF (PROJ<0) VEL(:,NI)=0
!         ENDDO

         PROJ=SUM(VEL(:,1:NIONS)*F(:,1:NIONS))
         NORM=SUM(F(:,1:NIONS)*F(:,1:NIONS))

         GNORM1=NORM/FACT

         ! zero out all velocities and make vel parallel to force 
         IF (PROJ<0) THEN
            VEL(:,1:NIONS)=0
         ELSE 
            VEL(:,1:NIONS)=PROJ*F(:,1:NIONS)/NORM
         ENDIF

         VEL=VEL+F
         ! one peculiarity of the original algorithm was that the center of mass
         ! was not conserved (QUICKMIN has removed this problem)
         TMP=SUM(VEL,DIM=2)/NIONS
         ! if no ions are fixed, then center of mass should be preserved
         IF (.NOT. LSDYN) THEN
            DO NI=1,NIONS
               VEL(:,NI)=VEL(:,NI)-TMP
            ENDDO
            TMP=SUM(VEL,DIM=2)
            IF (IU0>0) WRITE(IU0,*)'summed vel is now',TMP
         ENDIF

      ENDIF
      IF (FACTSI/=0) THEN
         SIFVEL=SIFVEL+FSIF

         NORM=SUM(FSIF*FSIF)
         PROJ=SUM(FSIF*SIFVEL)

         GNORM2=NORM/FACTSI
         IF (PROJ<0) SIFVEL=0

         SIFVEL=SIFVEL+FSIF

      ENDIF
!-----------------------------------------------------------------------
!  1. order energy change along step
!-----------------------------------------------------------------------
      E1TEST=0
      IF (FACT/=0) THEN
         E1TEST= E1TEST -  SUM(VEL*F)/ FACT
      ENDIF

      IF (FACTSI/=0) THEN
         E1TEST= E1TEST - SUM(SIFVEL* FSIF)/ FACTSI
      ENDIF

      CALLMPI_C( sum_chain( GNORM1 ))
      CALLMPI_C( sum_chain( GNORM2 ))
      CALLMPI_C( sum_chain( E1TEST ))

      IF (IU0>=0) WRITE(IU0,10) GNORM1,GNORM2,E1TEST
   10 FORMAT('quench:  g(F)= ',E10.3,' g(S)= ',E10.3,' dE (1.order)=',E10.3)
!-----------------------------------------------------------------------
!    move ions
!-----------------------------------------------------------------------
      DO NI=1,NIONS
!----- velocitities from cartesian to direct lattice
         TMP = VEL(:,NI)
         CALL KARDIR(1,TMP,B)

         POSIOC(:,NI)= POSION(:,NI)
         POSION(:,NI)= TMP+POSIOC(:,NI)
      ENDDO

      AC=A
      DO J=1,3
      DO I=1,3
         A(I,J)=AC(I,J)
         DO K=1,3
            A(I,J)=A(I,J) + SIFVEL(I,K)*AC(K,J)
         ENDDO
      ENDDO
      ENDDO

      RETURN

      END

!*********************************************************************
!
! subroutine IONDAMPED
! subroutine uses a damped second order equation of motion
! a friction term is applied to damp the ionic motion
!
!*********************************************************************

      SUBROUTINE IONDAMPED(NIONS,A,B,IU6,IU0,LSDYN, &
            POSION,POSIOC,FACT,F,FACTSI,FSIF,VEL,E1TEST,FRICTION)
      USE prec
      USE lattice
      USE ini
      USE chain

      IMPLICIT NONE
      INTEGER NIONS,IU6,IU0
      REAL(q) F(3,NIONS),VEL(3,NIONS)
      REAL(q) POSION(3,NIONS),POSIOC(3,NIONS)
      REAL(q) A(3,3),B(3,3),FSIF(3,3)
      REAL(q) FACT,FACTSI,E1TEST
      REAL(q) FRICTION
      LOGICAL LSDYN
! local static
      REAL(q), SAVE :: SIFVEL(3,3)
      INTEGER, SAVE :: IFLAG=0
! local temporary
      REAL(q) TMP(3),AC(3,3)
      INTEGER :: NI,I,J,K
      REAL(q) PROJ,NORM,GNORM1,GNORM2

!=======================================================================
!  if IFLAG =0 initialize everything
!=======================================================================
      IF (IFLAG==0) THEN

         VEL=0
         SIFVEL=0
         IFLAG=1

      ENDIF
!=======================================================================
!  this is really simple
!=======================================================================
      GNORM1=0; GNORM2=0

      IF (FACT/=0) THEN
         TMP=SUM(F,DIM=2)
         ! v_next = v + a*2 - friction ( v + v_next ) /2

         VEL=((1-FRICTION/2)*VEL+2*F)/(1+FRICTION/2)

         NORM=SUM(F* F)
         GNORM1=NORM/FACT

      ENDIF
      IF (FACTSI/=0) THEN
         SIFVEL=((1-FRICTION/2)*SIFVEL+2*FSIF)/(1+FRICTION/2)

         NORM =SUM(FSIF*FSIF)
         GNORM2=NORM/FACTSI

      ENDIF
!-----------------------------------------------------------------------
!  1. order energy change along step
!-----------------------------------------------------------------------
      E1TEST=0
      IF (FACT/=0) THEN
         E1TEST= E1TEST -  SUM(VEL*F)/ FACT
      ENDIF

      IF (FACTSI/=0) THEN
         E1TEST= E1TEST - SUM(SIFVEL* FSIF)/ FACTSI
      ENDIF

      CALLMPI_C( sum_chain( GNORM1 ))
      CALLMPI_C( sum_chain( GNORM2 ))
      CALLMPI_C( sum_chain( E1TEST ))

      IF (IU0>=0) WRITE(IU0,10) GNORM1,GNORM2,E1TEST
   10 FORMAT('damped:  g(F)= ',E10.3,' g(S)= ',E10.3,' dE (1.order)=',E10.3)
!-----------------------------------------------------------------------
!    move ions
!-----------------------------------------------------------------------
      DO NI=1,NIONS
!----- velocitities from cartesian to direct lattice
         TMP = VEL(:,NI)
         CALL KARDIR(1,TMP,B)

         POSIOC(:,NI)= POSION(:,NI)
         POSION(:,NI)= TMP+POSIOC(:,NI)
      ENDDO

      AC=A
      DO J=1,3
      DO I=1,3
         A(I,J)=AC(I,J)
         DO K=1,3
            A(I,J)=A(I,J) + SIFVEL(I,K)*AC(K,J)
         ENDDO
      ENDDO
      ENDDO

      RETURN

      END


!*********************************************************************
!
! subroutine to set selected force components to zero
!
!*********************************************************************

      SUBROUTINE SET_SELECTED_FORCES_ZERO(T_INFO,VEL,TIFOR,LATT_CUR)
      USE lattice
      USE poscar
      IMPLICIT NONE

      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      REAL(q)  TIFOR(3,T_INFO%NIONS),VEL(3,T_INFO%NIONS)
! local
      INTEGER NI,M
      REAL(q) VTMP(3)


      IF (T_INFO%LSDYN) THEN
!-----Selective dynamics here ... :
         DO NI=1,T_INFO%NIONS
!-----reset the velocities of the selected coordinates ...
            DO M=1,3
               IF (.NOT.T_INFO%LSFOR(M,NI)) VEL(M,NI)=0._q
            ENDDO
!-----and reset the selected force coordinates (warning: forces are
!     given in cartesian, selection is made in direct coordinates!):
            DO M=1,3
              VTMP(M)=TIFOR(M,NI)
            ENDDO
            CALL KARDIR(1,VTMP,LATT_CUR%B)
            DO M=1,3
              IF (.NOT.T_INFO%LSFOR(M,NI)) VTMP(M)=0._q
            ENDDO
            CALL DIRKAR(1,VTMP,LATT_CUR%A)
            DO M=1,3
              TIFOR(M,NI)=VTMP(M)
            ENDDO
         ENDDO
      ENDIF
      END SUBROUTINE


      SUBROUTINE SET_SELECTED_VEL_ZERO(T_INFO,VEL,LATT_CUR)
      USE lattice
      USE poscar
      IMPLICIT NONE

      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      REAL(q)  VEL(3,T_INFO%NIONS)
! local
      INTEGER NI,M
      REAL(q) VTMP(3)


      IF (T_INFO%LSDYN) THEN
!-----Selective dynamics here ... :
         DO NI=1,T_INFO%NIONS
!-----reset the velocities of the selected coordinates ...
            DO M=1,3
               IF (.NOT.T_INFO%LSFOR(M,NI)) VEL(M,NI)=0._q
            ENDDO
         ENDDO
      ENDIF
      END SUBROUTINE

!*********************************************************************
!   subroutine PACO
!   calculation of  pair-correlation function
!   as input the 'exact positions' after the corrector step XC
!   should be used
!   the routine uses a simple discrete sampling sceme
!   IGRID   is the array for the discrete sampling
!   NGRID   number of sampling points
!   AGRID   maximum distance in Angstroem
!   This routine is optimized for a vector-computer
!   on a scalar CPU different algorithms might be faster
!*********************************************************************

      SUBROUTINE SPACO(NIONS,WEIGHT,XC,A,BNORM,SIPACO,NPACO,APACO)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION XC(3,NIONS)
      DIMENSION SIPACO(0:NPACO)
      DIMENSION A(3,3),BNORM(3)

!=======================================================================
!  IXMAX,Y,Z defines the maximum number of cells over which the
!  pair-correlation-function is evaluted
!=======================================================================
      I1MAX=AINT(APACO*BNORM(1)-.001_q)
      I2MAX=AINT(APACO*BNORM(2)-.001_q)
      I3MAX=AINT(APACO*BNORM(3)-.001_q)

      APACO2=APACO**2
      SCALE =NPACO/APACO
!=======================================================================
!  summation is only performed over (i.gt.np), as the sum is
!  symmetric in these arguments.
!=======================================================================
      IADD=1
      DO I1=-I1MAX-1,I1MAX
      DO I2=-I2MAX-1,I2MAX
      DO I3=-I3MAX-1,I3MAX

      DO NI=1,NIONS
      DO NII=NI+1,NIONS
         D1= I1+MOD(XC(1,NI)-XC(1,NII)+1,1._q)
         D2= I2+MOD(XC(2,NI)-XC(2,NII)+1,1._q)
         D3= I3+MOD(XC(3,NI)-XC(3,NII)+1,1._q)
         R2= (D1*A(1,1)+D2*A(1,2)+D3*A(1,3)) **2 &
     &        + (D1*A(2,1)+D2*A(2,2)+D3*A(2,3)) **2 &
     &        + (D1*A(3,1)+D2*A(3,2)+D3*A(3,3)) **2

         IF (R2<APACO2) THEN
            INDEX=INT(SQRT(R2)*SCALE)
            SIPACO(INDEX)=SIPACO(INDEX)+IADD*WEIGHT
         ENDIF
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      RETURN
      END
