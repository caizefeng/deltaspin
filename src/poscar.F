#include "symbol.inc"
      MODULE POSCAR
      USE prec

!
! poscar description input file
! only included if MODULES are not supported
!
      TYPE type_info
!only T_INFO
        CHARACTER*40 SZNAM2           ! name of poscar file
        INTEGER NTYPD                 ! dimension for types
        INTEGER NTYP                  ! number of types
        INTEGER NTYPPD                ! dimension for types inc. empty spheres
        INTEGER NTYPP                 ! number of types empty spheres
        INTEGER NIOND                 ! dimension for ions
        INTEGER NIONPD                ! dimension for ions inc. empty spheres
        INTEGER NIONS                 ! actual number of ions
        INTEGER NIONP                 ! actual number of ions inc. empty spheres
        LOGICAL LSDYN                 ! selective dynamics (yes/ no)
        LOGICAL LDIRCO                ! positions in direct/recproc. lattice
        REAL(q), POINTER :: POSION(:,:)  ! positions usually same as DYN%POSION
        LOGICAL,POINTER ::  LSFOR(:,:) ! selective dynamics
        INTEGER, POINTER :: ITYP(:)   ! type for each ion
        INTEGER, POINTER :: NITYP(:)  ! number of ions for each type
        REAL(q), POINTER :: POMASS(:) ! mass for each ion type
        REAL(q), POINTER :: RWIGS(:)  ! wigner seitz radius for each ion type
        REAL(q), POINTER :: ROPT(:)   ! optimization radius for each type
        REAL(q), POINTER :: ATOMOM(:) ! initial local spin density for each ion
        REAL(q), POINTER :: DARWIN_R(:) ! parameter for darwin like mass term at each ion
        REAL(q), POINTER :: DARWIN_V(:) ! parameter for darwin like mass term at each ion 
        REAL(q), POINTER :: VCA(:)    ! weight of each species for virtual crystal approximation 
        REAL(q), POINTER :: ZCT(:)    ! "charge transfer" charges for non-scf calculations
        REAL(q), POINTER :: RGAUS(:)  ! widths for Gaussian CT charge distributions
        CHARACTER (LEN=2), POINTER :: TYPE(:)  ! type information for each ion
      END TYPE


      TYPE dynamics
!only DYN
        REAL(q), POINTER :: POSION(:,:) ! positions
        REAL(q), POINTER :: POSIOC(:,:) ! old positions
        REAL(q), POINTER :: VEL(:,:)  ! velocities
        REAL(q), POINTER :: D2(:,:)   ! predictor corrector/coordinates
        REAL(q), POINTER :: D2C(:,:)  ! predictor corrector/coordinates
        REAL(q), POINTER :: D3(:,:)   ! predictor corrector/coordinates
        REAL(q) A(3,3)                ! current lattice (presently unused)
        REAL(q) AC(3,3)               ! old lattice (presently unused)
        REAL(q) SNOSE(4)              ! nose thermostat
        INTEGER IBRION                ! mode for relaxation
        INTEGER ISIF                  ! mode for stress/ ionic relaxation
        REAL(q) POTIM                 ! time step
        REAL(q) EDIFFG                ! accuracy for ionic relaxation
        REAL(q), POINTER :: POMASS(:) ! mass of each ion for dynamics
        REAL(q) SMASS                 ! mass of nose thermostat
        REAL(q) PSTRESS               ! external pressure
        REAL(q) TEBEG, TEEND          ! temperature during run
        REAL(q) TEMP                  ! current temperature
        INTEGER NSW                   ! number of ionic steps
        INTEGER NBLOCK,KBLOCK         ! blocks
        INTEGER INIT                  ! predictore corrector initialized
        INTEGER NFREE                 ! estimated ionic degrees of freedom
      END TYPE

      CONTAINS
!=======================================================================
! RCS:  $Id: poscar.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!  Read UNIT=15: POSCAR file scan for total number of ions
!  and number of types
!  only T_INFO%NITYP is allocated at this point
!  all other arrays are allocated in RD_POSCAR
!=======================================================================
      SUBROUTINE RD_POSCAR_HEAD(LATT_CUR, T_INFO, &
     &           NIOND,NIONPD, NTYPD,NTYPPD, IU0, IU6)
      USE prec
      USE lattice
      USE main_mpi

      IMPLICIT NONE

      INTEGER NIOND,NIONPD,NTYPPD,NTYPD
      INTEGER IU0,IU6

      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      INTEGER NITYP(10000)          ! hard limit 10000 ions :->
      CHARACTER (LEN=2) TYPE(10000) ! type information
! temporary varibales
      CHARACTER (1)    CHARAC
      CHARACTER (255)  INPLIN,INPWRK
      INTEGER        NI,I,NT,NSCALE
      REAL(q)        SCALEX,SCALEY,SCALEZ
      INTEGER, EXTERNAL :: NITEMS

! Now extract from file POSCAR how many ion types we have ...
      OPEN(UNIT=15,FILE=DIR_APP(1:DIR_LEN)//'POSCAR',STATUS='OLD',ERR=1000)

      READ(15,'(A1)',ERR=147,END=147) CHARAC

! one scaling parameter or one for x, y and z
      READ(15,'(A)',ERR=147,END=147) INPLIN
! how many words/data items? --> number of ion types on file POSCAR!
      NSCALE=NITEMS(INPLIN,INPWRK,.TRUE.,'F')
      IF (NSCALE==1) THEN
         READ(INPLIN,*) LATT_CUR%SCALE
         SCALEX=1
         SCALEY=1
         SCALEZ=1
      ELSE IF (NSCALE==3) THEN
         LATT_CUR%SCALE=1
         READ(INPLIN,*) SCALEX,SCALEY,SCALEZ
      ELSE
         IF (IU0>=0) WRITE(IU0,*) 'ERROR: there must be 1 or 3 items on line 2 of POSCAR'
         STOP
      ENDIF

      DO I=1,3
        READ(15,*,ERR=147,END=147) LATT_CUR%A(1,I),LATT_CUR%A(2,I),LATT_CUR%A(3,I)
      ENDDO

      IF (LATT_CUR%SCALE<0._q) THEN
!----alternatively give a volume (=abs(scale)) and adjust the lengths of
!----the three lattice vectors to get the correct desired volume ... :
         CALL LATTIC(LATT_CUR)
         LATT_CUR%SCALE=(ABS(LATT_CUR%SCALE) &
     &                 / ABS(LATT_CUR%OMEGA))**(1._q/3._q)
      ENDIF
      
      LATT_CUR%A(1,:) =LATT_CUR%A(1,:)*SCALEX*LATT_CUR%SCALE
      LATT_CUR%A(2,:) =LATT_CUR%A(2,:)*SCALEY*LATT_CUR%SCALE
      LATT_CUR%A(3,:) =LATT_CUR%A(3,:)*SCALEZ*LATT_CUR%SCALE
         
      CALL LATTIC(LATT_CUR)

      IF (LATT_CUR%OMEGA<0) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'ERROR: the triple product of the basis vectors ', &
     &     'is negative exchange two basis vectors'
        STOP
      ENDIF

! 6th line, contains either the number of ions or their type
      READ(15,'(A)',ERR=147,END=147) INPLIN
! how many words/data items? --> number of ion types on file POSCAR!
      READ(INPLIN,*,ERR=147,END=147) CHARAC
      IF (.NOT.(CHARAC>='0' .AND. CHARAC<='9')) THEN
         T_INFO%NTYP=NITEMS(INPLIN,INPWRK,.TRUE.,'A')
         READ(INPLIN,*,ERR=147,END=147) (TYPE(NI),NI=1,T_INFO%NTYP)
         IF (IU0>=0) THEN
            WRITE(IU0,'(A,20A3)') ' POSCAR found type information on POSCAR ',TYPE(1:T_INFO%NTYP)
         ENDIF

         READ(15,'(A)',ERR=147,END=147) INPLIN
         IF (T_INFO%NTYP/=NITEMS(INPLIN,INPWRK,.TRUE.,'I')) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'ERROR: the type information is not consistent with the number of types'
            STOP 
         ENDIF
      ELSE
         T_INFO%NTYP=NITEMS(INPLIN,INPWRK,.TRUE.,'I')
         TYPE(1:T_INFO%NTYP+1)='  '
      ENDIF

      T_INFO%NTYPP=T_INFO%NTYP
! let me know how many ions
      READ(INPLIN,*,ERR=147,END=147) (NITYP(NI),NI=1,T_INFO%NTYP)
! how many ions do we have on file POSCAR ... ?
      T_INFO%NIONS=0
      DO NI=1,T_INFO%NTYP
         T_INFO%NIONS=T_INFO%NIONS+NITYP(NI)
      END DO

! there might be empty spheres scan for them

      T_INFO%NIONP=T_INFO%NIONS
      T_INFO%NTYPP=T_INFO%NTYP

      READ(15,'(A1)',ERR=147,END=147) CHARAC
      IF ((CHARAC=='S').OR.(CHARAC=='s')) &
     &   READ(15,'(A1)',ERR=147,END=147) CHARAC
      DO NI=1,T_INFO%NIONS
         READ(15,'(A1)',ERR=147,END=147) CHARAC
      END DO

      READ(15,'(A1)',ERR=147,END=147) CHARAC
      IF ((CHARAC=='E').OR.(CHARAC=='e')) THEN
! this is also important for us ...
         READ(15,'(A)',ERR=147,END=147) INPLIN
! how many words/data items? --> number of empty sphere types!
         T_INFO%NTYPP=T_INFO%NTYPP+NITEMS(INPLIN,INPWRK,.TRUE.,'I')
         READ(INPLIN,*) (NITYP(NT),NT=T_INFO%NTYP+1,T_INFO%NTYPP)
         DO NT=T_INFO%NTYP+1,T_INFO%NTYPP
           T_INFO%NIONP=T_INFO%NIONP+NITYP(NT)
         ENDDO
         TYPE(T_INFO%NTYP+1:T_INFO%NTYPP)='  '
      ENDIF
! ... precise details later in the program ...
  147 REWIND 15
! set the require allocation parameters

      NIOND =T_INFO%NIONS
      NTYPD =T_INFO%NTYP
      NIONPD=T_INFO%NIONP
      NTYPPD=T_INFO%NTYPP

      ALLOCATE(T_INFO%NITYP(NTYPPD),T_INFO%TYPE(NTYPPD))

      T_INFO%NITYP(1:NTYPPD)=NITYP(1:NTYPPD)
      T_INFO%TYPE(1:NTYPPD) =TYPE (1:NTYPPD)

      IF (IU0>=0) &
      WRITE(IU0,1) DIR_APP(1:DIR_LEN),NTYPPD,NIONPD

    1 FORMAT(' ',A,'POSCAR found : ',I2,' types and ',I7,' ions' )

      CLOSE(UNIT=15)
      RETURN
 1000 CONTINUE
!
! all nodes report to unit 6 which have IU6 defined
! (guarantees that a sensible error message is allways written out)
!
      IF (IU6>=0) THEN
         WRITE(*,"(A,A)")'ERROR: the following files does not exist ', &
             DIR_APP(1:DIR_LEN)//'POSCAR'
      ENDIF
      STOP
      END SUBROUTINE

!=======================================================================
!
!  Read UNIT=15: POSCAR Startjob and Continuation-job
!
!=======================================================================
      SUBROUTINE RD_POSCAR(LATT_CUR, T_INFO, DYN, &
     &           NIOND,NIONPD, NTYPD,NTYPPD, &
     &           IU0,IU6)
      USE prec
      USE lattice
      USE main_mpi

      IMPLICIT NONE

      INTEGER NIOND,NIONPD,NTYPPD,NTYPD
      CHARACTER (255)  INPLIN,INPWRK
      INTEGER, EXTERNAL :: NITEMS
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
      INTEGER IU0,IU6        ! io unit
! temporary
      CHARACTER (1)  CSEL
      INTEGER I,NT,NI,NSCALE
      REAL(q) SCALEX,SCALEY,SCALEZ
      REAL(q) POTIMR

      OPEN(UNIT=15,FILE=DIR_APP(1:DIR_LEN)//'POSCAR',STATUS='OLD')

      IF (IU6>=0) WRITE(IU6,*)
!-----Basis vectors and scaling parameter ('lattice constant')
      READ(15,'(A40)') T_INFO%SZNAM2
      IF (IU6>=0) WRITE(IU6,*)'POSCAR: ',T_INFO%SZNAM2

! one scaling parameter or one for x, y and z
      READ(15,'(A)') INPLIN
! how many words/data items? --> number of ion types on file POSCAR!
      NSCALE=NITEMS(INPLIN,INPWRK,.TRUE.,'F')
      IF (NSCALE==1) THEN
         READ(INPLIN,*) LATT_CUR%SCALE
         SCALEX=1
         SCALEY=1
         SCALEZ=1
      ELSE IF (NSCALE==3) THEN
         LATT_CUR%SCALE=1
         READ(INPLIN,*) SCALEX,SCALEY,SCALEZ
      ELSE
         IF (IU0>=0) WRITE(IU0,*)'ERROR: there must be 1 or 3 items on line 2 of POSCAR'
         STOP   
      ENDIF

      DO I=1,3
        READ(15,*) LATT_CUR%A(1,I),LATT_CUR%A(2,I),LATT_CUR%A(3,I)
      ENDDO

      IF (LATT_CUR%SCALE<0._q) THEN
!----alternatively give a volume (=abs(scale)) and adjust the lengths of
!----the three lattice vectors to get the correct desired volume ... :
         CALL LATTIC(LATT_CUR)
         LATT_CUR%SCALE=(ABS(LATT_CUR%SCALE)  &
     &                 / ABS(LATT_CUR%OMEGA))**(1._q/3._q)
      ENDIF

      LATT_CUR%A(1,:) =LATT_CUR%A(1,:)*SCALEX*LATT_CUR%SCALE
      LATT_CUR%A(2,:) =LATT_CUR%A(2,:)*SCALEY*LATT_CUR%SCALE
      LATT_CUR%A(3,:) =LATT_CUR%A(3,:)*SCALEZ*LATT_CUR%SCALE

      CALL LATTIC(LATT_CUR)

      IF (LATT_CUR%OMEGA<0) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'ERROR: the triple product of the basis vectors ', &
     &     'is negative exchange two basis vectors'
        STOP
      ENDIF

      T_INFO%NIOND =NIOND
      T_INFO%NIONPD=NIONPD
      T_INFO%NTYPD =NTYPD
      T_INFO%NTYPPD=NTYPPD
      ALLOCATE(T_INFO%LSFOR(3,NIOND),T_INFO%ITYP(NIOND))

      T_INFO%LSFOR=.TRUE.

!-----number of atoms per type
      READ(15,'(A)') INPLIN

      READ(INPLIN,*) CSEL
      IF (.NOT.(CSEL>='0' .AND. CSEL<='9')) THEN
         READ(INPLIN,*) (T_INFO%TYPE(NI),NI=1,T_INFO%NTYP)
         READ(15,'(A)') INPLIN
         IF (T_INFO%NTYP/=NITEMS(INPLIN,INPWRK,.TRUE.,'I')) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'ERROR: the type information is not consistent with the number of types'
            STOP 
         ENDIF
      ENDIF

      READ(INPLIN,*) (T_INFO%NITYP(NT),NT=1,T_INFO%NTYP)
!---- Set up the table from which we get type of each ion
      NI=1
      DO NT=1,T_INFO%NTYP
      DO NI=NI,T_INFO%NITYP(NT)+NI-1
        T_INFO%ITYP(NI)=NT
      ENDDO
      ENDDO
!
!   positions
!
      T_INFO%NIONS=0
      DO NT=1,T_INFO%NTYP
      T_INFO%NIONS= T_INFO%NIONS+ T_INFO%NITYP(NT)
      ENDDO

      T_INFO%NIONP=T_INFO%NIONS

      IF (T_INFO%NIONS>NIOND) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'ERROR: MAIN: increase NIOND',T_INFO%NIONS
        STOP
      ENDIF

      READ(15,'(A1)') CSEL
      T_INFO%LSDYN=((CSEL=='S').OR.(CSEL=='s'))
      IF (T_INFO%LSDYN) READ(15,'(A1)') CSEL
      IF (CSEL=='K'.OR.CSEL=='k'.OR. &
     &    CSEL=='C'.OR.CSEL=='c') THEN
        CSEL='K'
        IF (IU6>=0) &
        WRITE(IU6,*)' positions in cartesian coordinates'

        T_INFO%LDIRCO=.FALSE.
      ELSE
        IF (IU6>=0) &
        WRITE(IU6,*)' positions in direct lattice'
        T_INFO%LDIRCO=.TRUE.
      ENDIF

      ALLOCATE(DYN%POSION(3,NIONPD),DYN%POSIOC(3,NIONPD), &
     &         DYN%D2C(3,NIOND), &
     &         DYN%VEL(3,NIOND),DYN%D2(3,NIOND),DYN%D3(3,NIOND))

! alias T_INFO%POSION
      T_INFO%POSION => DYN%POSION
      DYN%POSION=0
      DYN%VEL   =0
      DYN%D2    =0
      DYN%D2C   =0
      DYN%D3    =0
!tb start
      LATT_CUR%AVEL=0
!tb end


      DO NI=1,T_INFO%NIONS
      IF (T_INFO%LSDYN) THEN
      READ(15,*,ERR=400,END=400) DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI), &
     &      T_INFO%LSFOR(1,NI),T_INFO%LSFOR(2,NI),T_INFO%LSFOR(3,NI)
      ELSE
      READ(15,*,ERR=400,END=400) DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI)
      ENDIF
      ENDDO

      IF (CSEL=='K') THEN
        DYN%POSION(1,:)=LATT_CUR%SCALE*DYN%POSION(1,:)*SCALEX
        DYN%POSION(2,:)=LATT_CUR%SCALE*DYN%POSION(2,:)*SCALEY
        DYN%POSION(3,:)=LATT_CUR%SCALE*DYN%POSION(3,:)*SCALEZ
        
        CALL KARDIR(T_INFO%NIONS,DYN%POSION,LATT_CUR%B)
      ENDIF
      CALL TOPRIM(T_INFO%NIONS,DYN%POSION)
      DYN%POSIOC=DYN%POSION

      DYN%INIT=0
      DYN%SNOSE(1)=1
      LATT_CUR%INITlatv=-1

!
!   empty spheres
!
      READ(15,'(A1)',ERR=424,END=410) CSEL
 ! read in new components and velocities of lattice vector components
      IF ((CSEL=='L').OR.(CSEL=='l')) THEN
        READ(15,*,ERR=4001,END=4001)  LATT_CUR%INITlatv
        IF (  LATT_CUR%INITlatv>=0) THEN
          DO NI=1,3
            READ(15,'(3E16.8)',ERR=4001,END=4001) LATT_CUR%AVEL(1,NI),LATT_CUR%AVEL(2,NI),LATT_CUR%AVEL(3,NI)
          ENDDO
          LATT_CUR%AVEL=LATT_CUR%AVEL*DYN%POTIM
        ENDIF
        
        IF (  LATT_CUR%INITlatv>0) THEN
          DO NI=1,3
             READ(15,'(3E16.8)',ERR=4001,END=4001) LATT_CUR%A(1,NI),LATT_CUR%A(2,NI),LATT_CUR%A(3,NI)
          ENDDO
        ENDIF
        CALL LATTIC(LATT_CUR)

        READ(15,'(A1)',ERR=424,END=410) CSEL
      ENDIF

  424 IF ((CSEL=='E').OR.(CSEL=='e')) THEN
        IF (T_INFO%NTYPP>NTYPPD) THEN
        IF (IU0>=0) &
         WRITE(IU0,*)'ERROR: MAIN: increase NEMPTY',T_INFO%NTYPP-T_INFO%NTYP
         STOP
        ENDIF
        READ(15,*,ERR=410,END=410) (T_INFO%NITYP(NT),NT=T_INFO%NTYP+1,T_INFO%NTYPP)
        DO NT=T_INFO%NTYP+1,T_INFO%NTYPP
          T_INFO%NIONP=T_INFO%NIONP+T_INFO%NITYP(NT)
        ENDDO
        IF (T_INFO%NIONP>NIONPD) THEN
        IF (IU0>=0) &
         WRITE(IU0,*)'ERROR: MAIN: increase NEMPTY',T_INFO%NIONP-T_INFO%NIONS
         STOP
        ENDIF
        T_INFO%NIONP=T_INFO%NIONP

        DO NI=T_INFO%NIONS+1,T_INFO%NIONP
         READ(15,*,ERR=410,END=410) &
     &      DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI)
        ENDDO
        IF (.NOT.T_INFO%LDIRCO) THEN
          DO NI=T_INFO%NIONS+1,T_INFO%NIONP
            DYN%POSION(1,NI)=LATT_CUR%SCALE*DYN%POSION(1,NI)*SCALEX
            DYN%POSION(2,NI)=LATT_CUR%SCALE*DYN%POSION(2,NI)*SCALEY
            DYN%POSION(3,NI)=LATT_CUR%SCALE*DYN%POSION(3,NI)*SCALEZ
            CALL KARDIR(1,DYN%POSION(1:3,NI),LATT_CUR%B)
          ENDDO
        ENDIF
        DYN%POSIOC=DYN%POSION
        READ(15,'(A1)',ERR=425,END=410) CSEL
      ENDIF
      DYN%POSIOC=DYN%POSION

  425 IF (CSEL=='K'.OR.CSEL=='k'.OR.CSEL==' ' &
     &    .OR.CSEL=='C'.OR.CSEL=='c') THEN
        CSEL='K'
        IF (IU6>=0) &
        WRITE(IU6,*)' velocities in cartesian coordinates'
      ELSE
        IF (IU6>=0) &
        WRITE(IU6,*)' velocities in direct lattice'
      ENDIF

!
!-----if we have velocities, read them in and transform from
!     cartesian coordinates to direct lattice
      DO NI=1,T_INFO%NIONS
        READ(15,*,ERR=410,END=410)  &
     &            DYN%VEL(1,NI),DYN%VEL(2,NI),DYN%VEL(3,NI)
        !tb start
        IF (CSEL=='K' .AND. DYN%IBRION/=44 .AND. DYN%IBRION/=40) THEN
        !IF (CSEL=='K') THEN
        !tb end
        CALL  KARDIR(1,DYN%VEL(1:3,NI),LATT_CUR%B)
        DYN%VEL(1:3,NI)=DYN%VEL(1:3,NI)*DYN%POTIM
        ENDIF
      ENDDO
      
!
!-----try to read in predictor Coordinates
!
      READ(15,*,ERR=430,END=430)
      READ(15,*,ERR=430,END=430) DYN%INIT
 
      !c initialization of ionic and latice DOFs must be consistent
      !c kill the job otherwise
      !c the folloving combinations of INITlatv and INIT make sense:
      !c     -1 -1 - just structure is read in
      !c     -1  0 - structure and ionic velocities
      !c      0 -1 - structure and lattice velocities
      !c      0  0 - structure and all velocities
      !c      1  1 - continuation of previous MD
      IF (DYN%IBRION==0 .AND. (DYN%ISIF>=3)) THEN
        IF ((DYN%INIT/=1 .AND.  LATT_CUR%INITlatv==1) .OR. (DYN%INIT==1 .AND.  LATT_CUR%INITlatv/=1)) GOTO 4002
      ENDIF

!-----if INIT is there and it is 1 we have predictor-coordinates on the
!-----file so we can start with them
      IF (DYN%INIT==0) GOTO 430
      READ(15,*) POTIMR
      IF (POTIMR/=DYN%POTIM) THEN
        IF (IU6>=0) THEN
           WRITE(IU6,*)
           WRITE(IU6,*)' There are predictor-coordinates on the file.'
           WRITE(IU6,*)' we can''t use them due to change of POTIM!'
        ENDIF
        GOTO 430
      ENDIF

!-----Read in Nose-Parameter
      READ(15,*) DYN%SNOSE
!-----Read in predictor-coordinates (always in direct lattice)
      READ(15,*) (DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI),NI=1,T_INFO%NIONS)
      READ(15,*) (DYN%D2(1,NI),DYN%D2(2,NI),DYN%D2(3,NI),NI=1,T_INFO%NIONS)
      READ(15,*) (DYN%D3(1,NI),DYN%D3(2,NI),DYN%D3(3,NI),NI=1,T_INFO%NIONS)
      IF (IU6>=0) THEN
         WRITE(IU6,*)
         WRITE(IU6,*)' Using predictor-coordinates on the file'
      ENDIF

      CLOSE(UNIT=15)
      RETURN
!-----------------------------------------------------------------------
!  Reading Inputfile 15 finished
!  if you end up at 430  INIT is set to 0,
!    INIT is used in the call to STEP  (predictors are not initialised)
!    in that way we tell STEP that it must initialize everything for us
!----------------------------------------------------------------------
  400 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,*)' No initial positions read in'
      STOP

  4001 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,*)' No initial lattice velocities or vectors read in'
      STOP

  4002 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,*)' Inconsistent initialization of ion. and lat. DOFs'
      STOP

  410 CONTINUE
      DYN%INIT=-1
      IF (IU6>=0) &
      WRITE(IU6,*)' No initial velocities read in'

      CLOSE(UNIT=15)
      !c initialized lattice DOFs and unitialized atomic DOFs makes no sense
      IF (DYN%IBRION==0 .AND. (DYN%ISIF==3 .OR.  DYN%ISIF==8 .OR. DYN%ISIF==9 .OR. DYN%ISIF==10)) THEN
        IF ( LATT_CUR%INITlatv==1 ) THEN
          IF (IU0>=0) WRITE(IU0,*)' Inconsistent initialization of ion. and lat. DOFs'
          STOP
        ENDIF
      ENDIF

      RETURN

  430 DYN%INIT=0
      CLOSE(UNIT=15)
      !c using next lattice vectors and previous atomic positions makes no sense
      IF (DYN%IBRION==0 .AND. (DYN%ISIF==3 .OR.  DYN%ISIF==8 .OR. DYN%ISIF==9 .OR. DYN%ISIF==10)) THEN
        IF ( LATT_CUR%INITlatv==1 ) THEN
          IF (IU0>=0) WRITE(IU0,*)' Inconsistent initialization of ion. and lat. DOFs'
          STOP
        ENDIF
      ENDIF
      RETURN

      END SUBROUTINE

!*********************************************************************
!  
! this subroutine counts the dregress of freedom
!
!*********************************************************************

      SUBROUTINE COUNT_DEGREES_OF_FREEDOM( T_INFO, NDEGREES_OF_FREEDOM, &
              IU6, IU0, IBRION)
      USE prec
      TYPE (type_info) :: T_INFO
      COMPLEX (q) :: CDUM; REAL(q) :: RDUM ; LOGICAL :: LDUM

      IF ( T_INFO%LSDYN ) THEN
         NDEGREES_OF_FREEDOM=0
         DO N=1,T_INFO%NIONS
            DO J=1,3
               IF ( T_INFO%LSFOR(J,N)) &
                    NDEGREES_OF_FREEDOM=NDEGREES_OF_FREEDOM+1
            ENDDO
         ENDDO
      ELSE
         NDEGREES_OF_FREEDOM=3*T_INFO%NIONS-3
      ENDIF
! zero degrees of freedom, do not make me happy
! so in that case I simply set NDEGREES_OF_FREEDOM to 3*NIONS
! this avoids floating point exceptions in lots of places
      IF (NDEGREES_OF_FREEDOM==0) NDEGREES_OF_FREEDOM=3*T_INFO%NIONS
      
      END SUBROUTINE COUNT_DEGREES_OF_FREEDOM

!*************************** SYMVEL **********************************
!  
!  this subroutine removes any drift from the velocities
!  and warns the user if that the drift has been removed
!
!*********************************************************************

      SUBROUTINE SYMVEL_WARNING(NIONS, NTYP, ITYP,POMASS,V,IU6,IU0)
      USE prec
      USE lattice
      IMPLICIT REAL(q) (A-H,O-Z)

      REAL(q) V(3,NIONS)
      INTEGER ITYP(NIONS)
      REAL(q) POMASS(NTYP)
      REAL(q) TMP(3),AVERAGE
      LOGICAL LWARN
      COMPLEX(q)  CDUM  ; LOGICAL  LDUM; REAL(q) RDUM; INTEGER IDUM


      AVERAGE=0
      TMP=0
!
      DO N=1,NIONS
         NT=ITYP(N)
         AVERAGE=AVERAGE+POMASS(NT)
         DO  J=1,3
            TMP(J)=TMP(J)+V(J,N)*POMASS(NT)
         ENDDO
      ENDDO

      LWARN=.FALSE.

      DO J=1,3
         IF ( ABS(TMP(J))> 1E-6) THEN
            LWARN=.TRUE.
         ENDIF
         TMP(J)=-TMP(J)/AVERAGE
      ENDDO

      DO N=1,NIONS
         DO J=1,3
            V(J,N)=V(J,N)+TMP(J)
         ENDDO
      ENDDO
      IF (LWARN) THEN
      CALL VTUTOR('W','CENTER OF MASS DRIFT',RDUM,1, &
                        ITUT,1,CDUM,1,LDUM,1,IU6,3)
      CALL VTUTOR('W','CENTER OF MASS DRIFT',RDUM,1, &
                        ITUT,1,CDUM,1,LDUM,1,IU0,3)
      ENDIF

      RETURN
      END SUBROUTINE


!*************************SUBROUTINE OUTPOS ****************************
!
!   write lattice parameters and positions to specified unit
!   use POSCAR compatibel format
!   LLONG specifies wether a long or short format is created
!   should be called only on IONODE !!
!
!***********************************************************************

      SUBROUTINE OUTPOS(IU, LLONG,SZNAM, T_INFO, SCALE, A, LSDYN, POSION)
      USE prec
      IMPLICIT NONE

      LOGICAL LLONG,LSDYN
      TYPE (type_info) :: T_INFO
      REAL(q) SCALE
      REAL(q) A(3,3)
      REAL(q) POSION(3,T_INFO%NIONS)
      CHARACTER (40) FORM
      CHARACTER (40) SZNAM
      INTEGER IU
! LOCAL
      INTEGER NT, NI, I
!-----direct lattice
      WRITE(IU,'(A40)') SZNAM

      WRITE(IU,*)  SCALE
      IF (LLONG) THEN
        FORM='(1X,3F22.16)'
      ELSE
        FORM='(1X,3F12.6)'
      ENDIF
      WRITE(IU,FORM) (A(1,I)/SCALE,A(2,I)/SCALE,A(3,I)/SCALE,I=1,3)

      IF (T_INFO%TYPE(1)/='  ') THEN
         WRITE(IU,'(20A5)') (T_INFO%TYPE(NT),NT=1,T_INFO%NTYP)
      ENDIF

      WRITE(IU,'(20I6)') (T_INFO%NITYP(NT),NT=1,T_INFO%NTYP)
      IF (LSDYN) WRITE(13,'(A18)') 'Selective dynamics'
      WRITE(IU,'(A6)')'Direct'

      IF (LSDYN) THEN
      IF (LLONG) THEN
        FORM='(3F20.16,3L4)'
      ELSE
        FORM='(3F10.6,3L2)'
      ENDIF
      ELSE
      IF (LLONG) THEN
        FORM='(3F20.16)'
      ELSE
        FORM='(3F10.6)'
      ENDIF
      ENDIF

      IF (LSDYN) THEN
          WRITE(IU,FORM) &
     &      (POSION(1,NI),POSION(2,NI),POSION(3,NI), &
     &       T_INFO%LSFOR(1,NI),T_INFO%LSFOR(2,NI),T_INFO%LSFOR(3,NI),NI=1,T_INFO%NIONS)
         ELSE
          WRITE(IU,FORM) &
     &      (POSION(1,NI),POSION(2,NI),POSION(3,NI),NI=1,T_INFO%NIONS)
      ENDIF
      IF (.NOT.LLONG) WRITE(IU,*)
      RETURN
      END SUBROUTINE

!=======================================================================
!
! read positions from stdin
!
!=======================================================================
      SUBROUTINE INPOS(LATT_CUR, T_INFO, DYN, IU6, IU0, LSTOP, MYCOMM)
      USE prec
      USE lattice
      USE main_mpi
      IMPLICIT NONE

      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
      INTEGER :: IU6, IU0
      LOGICAL :: LSTOP
      TYPE (communic) :: MYCOMM
    ! local
      INTEGER NI, IERR

      IF (IU6>=0) THEN
         IF (IU0>=0) WRITE(IU0,'(A)') 'POSITIONS: reading from stdin'
         DO NI=1,T_INFO%NIONS
            READ(*,*,  IOSTAT=IERR) DYN%POSION(1,NI), DYN%POSION(2,NI), DYN%POSION(3,NI)
            IF (IERR/=0) EXIT
         ENDDO

         IF (IERR/=0) THEN
            LSTOP=.TRUE.
         ELSE
            LSTOP=.FALSE.
         ENDIF
         WRITE(*,'(3F14.7)') DYN%POSION
         WRITE(*,'(3F14.7)') DYN%POSIOC
         CALLMPI( M_sum_i(MYCOMM, IERR, 1))
         IF (IERR==0) THEN
            CALLMPI( M_sum_d(MYCOMM, DYN%POSION(1,1), T_INFO%NIONS*3))
         ENDIF
         IF (IU0>=0) WRITE(IU0,'(A)') 'POSITIONS: read from stdin'
      ELSE
         IERR=0
         CALLMPI( M_sum_i(MYCOMM, IERR, 1))
         IF (IERR==0) THEN
            DYN%POSION(:,1:T_INFO%NIONS)=0
            CALLMPI( M_sum_d(MYCOMM, DYN%POSION(1,1), T_INFO%NIONS*3))
         ENDIF
      ENDIF
      END SUBROUTINE INPOS



!*************************SUBROUTINE OUTPOS_TRAIL  *********************
! write trailer for CONTCAR file
!
!    should be called only on IONODE !!
!***********************************************************************

      SUBROUTINE OUTPOS_TRAIL(IU,LOPEN, LATT_CUR, T_INFO, DYN)
      USE prec
      USE lattice
      IMPLICIT NONE

      INTEGER IU
      LOGICAL LOPEN
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
! local variables
      INTEGER NT,NI
      REAL(q) :: VTMP(3)

      IF (DYN%IBRION==0 .AND. (DYN%ISIF==3 .OR. DYN%ISIF==8 .OR. DYN%ISIF==9 .OR. DYN%ISIF==10)) THEN
        !tb start new
        ! WRITE(IU,'(A)') 'Lattice velocities' 
        ! WRITE(IU,480) &
        !&      (LATT_CUR%AVEL(1,NI)/DYN%POTIM,LATT_CUR%AVEL(2,NI)/DYN%POTIM,LATT_CUR%AVEL(3,NI)/DYN%POTIM,NI=1,3)
        IF ( LATT_CUR%INITlatv==1) THEN
          WRITE(IU,'(A)') 'Lattice velocities and vectors'
          WRITE(IU,*)  LATT_CUR%INITlatv
          WRITE(IU,480) &
          &      (LATT_CUR%AVEL(1,NI)/DYN%POTIM,LATT_CUR%AVEL(2,NI)/DYN%POTIM,LATT_CUR%AVEL(3,NI)/DYN%POTIM,NI=1,3)
          WRITE(IU,480) &
          &      (LATT_CUR%A(1,NI),LATT_CUR%A(2,NI),LATT_CUR%A(3,NI),NI=1,3)
        ENDIF
        !tb end new
      ENDIF


      IF (T_INFO%NIONP>T_INFO%NIONS) THEN
         WRITE(IU,'(A)') 'Empty spheres'
         WRITE(IU,'(20I5)') (T_INFO%NITYP(NT),NT=T_INFO%NTYP+1,T_INFO%NTYPP)
         WRITE(IU,'(3F20.16)') &
     &      (DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI),NI=T_INFO%NIONS+1,T_INFO%NIONP)
      ENDIF
      WRITE(IU,*)
!-----write out velocities
      DO NI=1,T_INFO%NIONS
      !tb start
        IF (DYN%IBRION/=44 .AND. DYN%IBRION/=40) THEN
          VTMP(1)=   DYN%VEL(1,NI)/DYN%POTIM
          VTMP(2)=   DYN%VEL(2,NI)/DYN%POTIM
          VTMP(3)=   DYN%VEL(3,NI)/DYN%POTIM
          CALL  DIRKAR(1,VTMP,LATT_CUR%A)
        ELSE
          VTMP(1)=   DYN%VEL(1,NI)
          VTMP(2)=   DYN%VEL(2,NI)
          VTMP(3)=   DYN%VEL(3,NI)
        END IF
        !VTMP(1)=   DYN%VEL(1,NI)/DYN%POTIM
        !VTMP(2)=   DYN%VEL(2,NI)/DYN%POTIM
        !VTMP(3)=   DYN%VEL(3,NI)/DYN%POTIM
        !CALL  DIRKAR(1,VTMP,LATT_CUR%A)
     !tb end
        WRITE(IU,480) VTMP(1),VTMP(2),VTMP(3)
      ENDDO
  480 FORMAT(3E16.8)
!-----if there was a call to STEP write out predictor-coordinates
      IF (DYN%INIT==1) THEN
      WRITE(IU,*)
      WRITE(IU,*) DYN%INIT
      WRITE(IU,*) DYN%POTIM
!-----write Nose-Parameter
      WRITE(IU,'(4E16.8)') DYN%SNOSE
      WRITE(IU,480) (DYN%POSION(1,NI),DYN%POSION(2,NI),DYN%POSION(3,NI),NI=1,T_INFO%NIONS)
      WRITE(IU,480) (DYN%D2(1,NI),DYN%D2(2,NI),DYN%D2(3,NI),NI=1,T_INFO%NIONS)
      WRITE(IU,480) (DYN%D3(1,NI),DYN%D3(2,NI),DYN%D3(3,NI),NI=1,T_INFO%NIONS)
      ENDIF

      IF (LOPEN) THEN
         CALL REOPEN(IU)
      ELSE
         REWIND IU
      ENDIF
      RETURN
      END SUBROUTINE

!***********************************************************************
!  write out initial header for XDATCAR
!***********************************************************************

      SUBROUTINE XDAT_HEAD(IU, T_INFO, LATT_CUR, DYN, SZNAM1)
      USE prec
      USE lattice
      IMPLICIT NONE

      INTEGER IU
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
      CHARACTER (40) SZNAM1
! local variables
      REAL(q) AOMEGA
      INTEGER NT
      INTEGER I
      CHARACTER (40) FORM
      INTEGER, PARAMETER :: SCALE=1
      LOGICAL, PARAMETER :: LLONG=.FALSE.

!
!     WRITE(IU,'(A40)') SZNAM1
!     IF (T_INFO%TYPE(1)/='  ') THEN
!        WRITE(IU,'(20A5)') (T_INFO%TYPE(NT),NT=1,T_INFO%NTYP)
!     ENDIF
      
      WRITE(IU,'(A40)') SZNAM1
      WRITE(IU,*)  SCALE
      IF (LLONG) THEN
        FORM='(1X,3F22.16)'
      ELSE
        FORM='(1X,3F12.6)'
      ENDIF
      WRITE(IU,FORM) (LATT_CUR%A(1,I)/SCALE,LATT_CUR%A(2,I)/SCALE,LATT_CUR%A(3,I)/SCALE,I=1,3)

      IF (T_INFO%TYPE(1)/='  ') THEN
         WRITE(IU,'(20A5)') (T_INFO%TYPE(NT),NT=1,T_INFO%NTYP)
      ENDIF

      WRITE(IU,'(20I4)') (T_INFO%NITYP(NT),NT=1,T_INFO%NTYP)

      RETURN
      END SUBROUTINE

!***********************************************************************
!  write out initial header for XDATCAR, use current lattice 
!***********************************************************************

      SUBROUTINE XDAT_HEAD_AC(IU, T_INFO, LATT_CUR, DYN, SZNAM1)
      USE prec
      USE lattice
      IMPLICIT NONE

      INTEGER IU
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
      CHARACTER (40) SZNAM1
      REAL(q) AOMEGA
      INTEGER NT
      INTEGER I
      CHARACTER (40) FORM
      INTEGER, PARAMETER :: SCALE=1
      LOGICAL, PARAMETER :: LLONG=.FALSE.

      WRITE(IU,'(A40)') SZNAM1
      WRITE(IU,*)  SCALE
      IF (LLONG) THEN
        FORM='(1X,3F22.16)'
      ELSE
        FORM='(1X,3F12.6)'
      ENDIF
      WRITE(IU,FORM) (DYN%AC(1,I)/SCALE,DYN%AC(2,I)/SCALE,DYN%AC(3,I)/SCALE,I=1,3)
      IF (T_INFO%TYPE(1)/='  ') THEN
         WRITE(IU,'(20A5)') (T_INFO%TYPE(NT),NT=1,T_INFO%NTYP)
      ENDIF

      WRITE(IU,'(20I4)') (T_INFO%NITYP(NT),NT=1,T_INFO%NTYP)

      RETURN
      END SUBROUTINE

!***********************************************************************
!
!  nearest neighboar table
!  (special wish of Roland Stumpf, and I think a good idea indeed)
!
!***********************************************************************

      SUBROUTINE NEAREST_NEIGHBOUR(IU6, IU0, T_INFO,L, RWIGS)
      USE prec
      USE lattice
      IMPLICIT NONE

      INTEGER IU6, IU0
      TYPE (latt)      :: L       ! lattice
      TYPE (type_info),TARGET :: T_INFO
      REAL(q) :: RWIGS(T_INFO%NTYP)
! local variables
      INTEGER, PARAMETER :: MAXNEIG=100
      REAL(q),POINTER :: POSION(:,:)
      INTEGER I1,I2,I3,NII,NI,NIONS,NT,NTT,NSWP,NOUT,I,II,IND,IDUM,IDIOT
      REAL(q) D,DX,DY,DZ,RWIGS1,RWIGS2,DIS,SWP,RWIGS_MIN,RDUM
      COMPLEX(q) CDUM

      INTEGER NEIGT(T_INFO%NIONS,MAXNEIG),NEIGN(T_INFO%NIONS)
      REAL(q) DNEIG(T_INFO%NIONS,MAXNEIG)
      LOGICAL LWARN,LDUM
      INTEGER IREP

      POSION => T_INFO%POSION
!--------------------------------------------------------------------
! build up nearest neighbor table
!--------------------------------------------------------------------
      IF (IU6 < 0) RETURN

      NIONS=T_INFO%NIONS
      IF (NIONS<1000) THEN
         IREP=1
      ELSE
      ! large systems, does not make sense to go over replicated cells
         IREP=0
      ENDIF

      RWIGS_MIN=1000

      DO NI=1,NIONS
         NT=T_INFO%ITYP(NI)
         RWIGS1=RWIGS(NT)*1.2
         IF (RWIGS1 <= 0.1) RWIGS1=1.0
         RWIGS_MIN=MIN(RWIGS1,RWIGS_MIN)

         IND=1
         I1=0
         I2=0
         I3=0

         DO I1=-IREP,IREP
         DO I2=-IREP,IREP
         DO I3=-IREP,IREP

            DO NII=1,NIONS
               NTT=T_INFO%ITYP(NII)
               RWIGS2=RWIGS(NTT)*1.2
               IF (RWIGS2 <= 0.1) RWIGS2=1.0

               DIS=RWIGS1+RWIGS2

               DX = MOD(POSION(1,NI)-POSION(1,NII)+10.5_q,1._q) -.5+I1
               DY = MOD(POSION(2,NI)-POSION(2,NII)+10.5_q,1._q) -.5+I2
               DZ = MOD(POSION(3,NI)-POSION(3,NII)+10.5_q,1._q) -.5+I3

               D =SQRT((DX*L%A(1,1)+DY*L%A(1,2)+DZ*L%A(1,3))**2 &
                  + (DX*L%A(2,1)+DY*L%A(2,2)+DZ*L%A(2,3))**2 &
                  + (DX*L%A(3,1)+DY*L%A(3,2)+DZ*L%A(3,3))**2)
               IF (NII/=NI .AND. D < DIS .AND. IND<=MAXNEIG) THEN
                  NEIGT(NI,IND)=NII
                  DNEIG(NI,IND)=D
                  IND=IND+1
               ENDIF
            ENDDO
         ENDDO
         ENDDO
         ENDDO
         NEIGN(NI)=IND-1
      ENDDO
!--------------------------------------------------------------------
! sort by length
!--------------------------------------------------------------------
      WRITE(IU6,*) 'ion  position               nearest neighbor table'
      DO NI=1,NIONS
         DO I =1,NEIGN(NI)
         DO II=1,I-1
            IF (DNEIG(NI,I) < DNEIG(NI,II)) THEN
               NSWP       =NEIGT(NI,I)
               NEIGT(NI,I)=NEIGT(NI,II)
               NEIGT(NI,II)=NSWP
               SWP        =DNEIG(NI,I)
               DNEIG(NI,I)=DNEIG(NI,II)
               DNEIG(NI,II)=SWP
            ENDIF
         ENDDO
         ENDDO

         NOUT     =MIN(NEIGN(NI),16)
         IF (NIONS<1000) THEN
            ! short format
            WRITE(IU6,11) NI,POSION(:,NI),(NEIGT(NI,IND),DNEIG(NI,IND),IND=1,NOUT)
         ELSE
            ! long format
            WRITE(IU6,111) NI,POSION(:,NI),(NEIGT(NI,IND),DNEIG(NI,IND),IND=1,NOUT)
         ENDIF
 11      FORMAT(I4,3F7.3,'-',8(I4,F5.2),(/,26X,8(I4,F5.2)))
 111     FORMAT(I6,3F7.3,'-',8(I6,F5.2),(/,26X,8(I6,F5.2)))
      ENDDO
      WRITE(IU6,*)

      LWARN=.FALSE.

      DO NI=1,NIONS
         IF (NEIGN(NI) > 0) THEN
            IF (DNEIG(NI,1) < RWIGS_MIN) LWARN=.TRUE.
         ENDIF
      ENDDO

      IDIOT=3
      IF (LWARN) THEN
         CALL VTUTOR('W','POSITION',RDUM,1, &
     &        IDUM,1,CDUM,1,LDUM,1,IU6,IDIOT)
         CALL VTUTOR('W','POSITION',RDUM,1, &
     &        IDUM,1,CDUM,1,LDUM,1,IU0,IDIOT)
      ENDIF

      END SUBROUTINE

      END MODULE
