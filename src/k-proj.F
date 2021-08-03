
#include "symbol.inc"

!***********************************************************************
!
! this module implements a projection onto k-points
!
!***********************************************************************

MODULE mkproj
  USE prec
  USE poscar
  USE nonl_high

  IMPLICIT none

  LOGICAL, SAVE :: LKPROJ
  REAL(q), SAVE :: KPROJ_THRESHOLD

CONTAINS
!**********************************************************************
!
! read all variables related to exchange correlation treatment
! from the INCAR file
! this set both the variables in the setex module and
! in the local fock module
!
!**********************************************************************

  SUBROUTINE KPROJ_READER(IU5, IU0 )
   
      IMPLICIT NONE
      INTEGER IU5, IU0
  ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE='INCAR',STATUS='OLD')

      LKPROJ=.FALSE.

      CALL RDATAB(LOPEN,'INCAR',IU5,'LKPROJ','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LKPROJ,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LKPROJ'' from file INCAR.'
         LKPROJ=.FALSE.
      ENDIF
      CALL XML_INCAR('LKPROJ','L',IDUM,RDUM,CDUM,LKPROJ,CHARAC,N)
      
      KPROJ_THRESHOLD=1.E-2_q
      CALL RDATAB(LOPEN,'INCAR',IU5,'KPROJ_THRESHOLD','=','#',';','F', &
     &            IDUM,KPROJ_THRESHOLD,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''KPROJ_THRESHOLD'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('KPROJ_THRESHOLD','F',IDUM,KPROJ_THRESHOLD,CDUM,LDUM,CHARAC,N)

      CLOSE(IU5)

    END SUBROUTINE KPROJ_READER

!=======================================================================
!  Read UNIT=15: POSCAR.prim file scan for total number of ions
!  and number of types
!  only T_INFO%NITYP is allocated at this point
!  all other arrays are allocated in RD_POSCAR
!=======================================================================

      SUBROUTINE RD_POSCAR_PRIM_HEAD(LATT_PRIM, T_INFO_PRIM, &
     &           NIOND,NIONPD, NTYPD,NTYPPD, IU0, IU6)
      USE prec
      USE lattice
      USE main_mpi

      IMPLICIT NONE

      INTEGER NIOND,NIONPD,NTYPPD,NTYPD
      INTEGER IU0,IU6

      TYPE (latt)::       LATT_PRIM
      TYPE (type_info) :: T_INFO_PRIM
      INTEGER NITYP(10000)          ! hard limit 10000 ions :->
      CHARACTER (LEN=2) TYPE(10000) ! type information
! temporary varibales
      CHARACTER (1)    CHARAC
      CHARACTER (255)  INPLIN,INPWRK
      INTEGER        NI,I,NT,NSCALE
      REAL(q)        SCALEX,SCALEY,SCALEZ
      INTEGER, EXTERNAL :: NITEMS

! Now extract from file POSCAR how many ion types we have ...
      OPEN(UNIT=15,FILE=DIR_APP(1:DIR_LEN)//'POSCAR.prim',STATUS='OLD',ERR=1000)

      READ(15,'(A1)',ERR=147,END=147) CHARAC

! one scaling parameter or one for x, y and z
      READ(15,'(A)',ERR=147,END=147) INPLIN
! how many words/data items? --> number of ion types on file POSCAR!
      NSCALE=NITEMS(INPLIN,INPWRK,.TRUE.,'F')
      IF (NSCALE==1) THEN
         READ(INPLIN,*) LATT_PRIM%SCALE
         SCALEX=1
         SCALEY=1
         SCALEZ=1
      ELSE IF (NSCALE==3) THEN
         LATT_PRIM%SCALE=1
         READ(INPLIN,*) SCALEX,SCALEY,SCALEZ
      ELSE
         IF (IU0>=0) WRITE(IU0,*) 'ERROR: there must be 1 or 3 items on line 2 of POSCAR.prim'
         STOP
      ENDIF

      DO I=1,3
        READ(15,*,ERR=147,END=147) LATT_PRIM%A(1,I),LATT_PRIM%A(2,I),LATT_PRIM%A(3,I)
      ENDDO

      IF (LATT_PRIM%SCALE<0._q) THEN
!----alternatively give a volume (=abs(scale)) and adjust the lengths of
!----the three lattice vectors to get the correct desired volume ... :
         CALL LATTIC(LATT_PRIM)
         LATT_PRIM%SCALE=(ABS(LATT_PRIM%SCALE) &
     &                 / ABS(LATT_PRIM%OMEGA))**(1._q/3._q)
      ENDIF
      
      LATT_PRIM%A(1,:) =LATT_PRIM%A(1,:)*SCALEX*LATT_PRIM%SCALE
      LATT_PRIM%A(2,:) =LATT_PRIM%A(2,:)*SCALEY*LATT_PRIM%SCALE
      LATT_PRIM%A(3,:) =LATT_PRIM%A(3,:)*SCALEZ*LATT_PRIM%SCALE
         
      CALL LATTIC(LATT_PRIM)

      IF (LATT_PRIM%OMEGA<0) THEN
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
         T_INFO_PRIM%NTYP=NITEMS(INPLIN,INPWRK,.TRUE.,'A')
         READ(INPLIN,*,ERR=147,END=147) (TYPE(NI),NI=1,T_INFO_PRIM%NTYP)
         IF (IU0>=0) THEN
            WRITE(IU0,'(A,20A3)') ' POSCAR.prim found type information on POSCAR ',TYPE(1:T_INFO_PRIM%NTYP)
         ENDIF

         READ(15,'(A)',ERR=147,END=147) INPLIN
         IF (T_INFO_PRIM%NTYP/=NITEMS(INPLIN,INPWRK,.TRUE.,'I')) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'ERROR: the type information is not consistent with the number of types'
            STOP 
         ENDIF
      ELSE
         T_INFO_PRIM%NTYP=NITEMS(INPLIN,INPWRK,.TRUE.,'I')
         TYPE(1:T_INFO_PRIM%NTYP+1)='  '
      ENDIF

      T_INFO_PRIM%NTYPP=T_INFO_PRIM%NTYP
! let me know how many ions
      READ(INPLIN,*,ERR=147,END=147) (NITYP(NI),NI=1,T_INFO_PRIM%NTYP)
! how many ions do we have on file POSCAR ... ?
      T_INFO_PRIM%NIONS=0
      DO NI=1,T_INFO_PRIM%NTYP
         T_INFO_PRIM%NIONS=T_INFO_PRIM%NIONS+NITYP(NI)
      END DO

! there might be empty spheres scan for them

      T_INFO_PRIM%NIONP=T_INFO_PRIM%NIONS
      T_INFO_PRIM%NTYPP=T_INFO_PRIM%NTYP

      READ(15,'(A1)',ERR=147,END=147) CHARAC
      IF ((CHARAC=='S').OR.(CHARAC=='s')) &
     &   READ(15,'(A1)',ERR=147,END=147) CHARAC
      DO NI=1,T_INFO_PRIM%NIONS
         READ(15,'(A1)',ERR=147,END=147) CHARAC
      END DO

      READ(15,'(A1)',ERR=147,END=147) CHARAC
      IF ((CHARAC=='E').OR.(CHARAC=='e')) THEN
! this is also important for us ...
         READ(15,'(A)',ERR=147,END=147) INPLIN
! how many words/data items? --> number of empty sphere types!
         T_INFO_PRIM%NTYPP=T_INFO_PRIM%NTYPP+NITEMS(INPLIN,INPWRK,.TRUE.,'I')
         READ(INPLIN,*) (NITYP(NT),NT=T_INFO_PRIM%NTYP+1,T_INFO_PRIM%NTYPP)
         DO NT=T_INFO_PRIM%NTYP+1,T_INFO_PRIM%NTYPP
           T_INFO_PRIM%NIONP=T_INFO_PRIM%NIONP+NITYP(NT)
         ENDDO
         TYPE(T_INFO_PRIM%NTYP+1:T_INFO_PRIM%NTYPP)='  '
      ENDIF
! ... precise details later in the program ...
  147 REWIND 15
! set the require allocation parameters

      NIOND =T_INFO_PRIM%NIONS
      NTYPD =T_INFO_PRIM%NTYP
      NIONPD=T_INFO_PRIM%NIONP
      NTYPPD=T_INFO_PRIM%NTYPP

      ALLOCATE(T_INFO_PRIM%NITYP(NTYPPD),T_INFO_PRIM%TYPE(NTYPPD))

      T_INFO_PRIM%NITYP(1:NTYPPD)=NITYP(1:NTYPPD)
      T_INFO_PRIM%TYPE(1:NTYPPD) =TYPE (1:NTYPPD)

      IF (IU0>=0) &
      WRITE(IU0,1) DIR_APP(1:DIR_LEN),NTYPPD,NIONPD

    1 FORMAT(' ',A,'POSCAR.prim found : ',I2,' types and ',I4,' ions' )

      CLOSE(UNIT=15)
      RETURN
 1000 CONTINUE
!
! all nodes report to unit 6 which have IU6 defined
! (guarantees that a sensible error message is allways written out)
!
      IF (IU6>=0) THEN
         WRITE(*,"(A,A)")'ERROR: the following files does not exist ', &
             DIR_APP(1:DIR_LEN)//'POSCAR.prim'
      ENDIF
      STOP
      END SUBROUTINE

!=======================================================================
!
!  Read UNIT=15: POSCAR Startjob and Continuation-job
!
!=======================================================================

      SUBROUTINE RD_POSCAR_PRIM(LATT_CUR, T_INFO, DYN, &
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

      OPEN(UNIT=15,FILE=DIR_APP(1:DIR_LEN)//'POSCAR.prim',STATUS='OLD')

      IF (IU6>=0) WRITE(IU6,*)
!-----Basis vectors and scaling parameter ('lattice constant')
      READ(15,'(A40)') T_INFO%SZNAM2
      IF (IU6>=0) WRITE(IU6,*)'POSCAR.prim: ',T_INFO%SZNAM2

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
         IF (IU0>=0) WRITE(IU0,*)'ERROR: there must be 1 or 3 items on line 2 of POSCAR.prim'
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
!
!   empty spheres
!
      READ(15,'(A1)',ERR=424,END=410) CSEL
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
        IF (CSEL=='K' .AND. DYN%IBRION/=44 ) THEN
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

  410 CONTINUE
      DYN%INIT=-1
      IF (IU6>=0) &
      WRITE(IU6,*)' No initial velocities read in'

      CLOSE(UNIT=15)
      RETURN

  430 DYN%INIT=0
      CLOSE(UNIT=15)
      RETURN

      END SUBROUTINE


!***********************************************************************
! search a particular k-point and return
! the equivalent k-point in the array, if not found return -1
! this is equal to the routine KPOINT_IN_FULL_GRID, except 
! that this routine returns -1 and no error if point is not found
! in the KPOINTS_F array
!
!***********************************************************************

  FUNCTION FIND_KPOINT_IN_PC(N1,N2,N3,VKPT,PNK)
    !USE sym_prec
    USE lattice
    INTEGER :: FIND_KPOINT_IN_PC
    REAL(q) N1,N2,N3
    REAL(q),PARAMETER :: TINY=1E-8_q
    REAL(q) VKPT(:,:)
    INTEGER PNK
    ! local
    INTEGER IND
    DO IND=1,PNK
       IF ( &
         (ABS(MOD(N1-VKPT(1,IND)+10.5_q,1._q)-0.5_q)<TINY) .AND. &
         (ABS(MOD(N2-VKPT(2,IND)+10.5_q,1._q)-0.5_q)<TINY) .AND. &
         (ABS(MOD(N3-VKPT(3,IND)+10.5_q,1._q)-0.5_q)<TINY)) EXIT
    ENDDO

    IF (IND>PNK) THEN
       ! no kpoint found, set nk=-1
       IND=-1
    ENDIF

    FIND_KPOINT_IN_PC=IND
  END FUNCTION FIND_KPOINT_IN_PC

!**********************************************************************
!
!**********************************************************************
    SUBROUTINE KPROJ(IU5, IU0, IU6, GRID, LATT_CUR, W, SYM, CQIJ)
      USE lattice
      USE constant
      USE msymmetry
      USE base
      USE main_mpi
      USE vaspxml
      USE wave_high

      IMPLICIT NONE
      INTEGER IU5, IU0, IU6  ! units for IO
      TYPE (wavespin)    :: W          ! wavefunction
      TYPE (latt)        :: LATT_CUR 
      TYPE (grid_3d)     :: GRID
      TYPE (symmetry)    :: SYM
      OVERLAP            :: CQIJ(:,:,:,:) ! overlap operator
      ! local
      TYPE (latt)        :: LATT_PRIM
      TYPE (type_info)   :: T_INFO_PRIM
      TYPE (dynamics)    :: DYN_PRIM
      INTEGER :: NIOND,NIONPD,NTYPD,NTYPPD
      INTEGER :: N1, N2, N3, IND
      INTEGER :: NK, KPT, NGVECTOR, NGVECTORM, NKPTS_PRIM, NKPTS_IRZ
      INTEGER :: ISP, N, I
      REAL(q) :: TEMP
      REAL(q) :: G1, G2, G3, GIX, GIY, GIZ, GX, GY, GZ
      REAL(q) :: R1, R2, R3, K1, K2, K3, RS1, RS2, RS3
      REAL(q), ALLOCATABLE :: VKPT(:,:)      ! k-points in the BZ of primtive cell
      REAL(q), ALLOCATABLE :: VKPT_IRZ(:,:)   
      REAL(q), ALLOCATABLE :: WTKPT_IRZ(:)
      REAL(q), ALLOCATABLE :: KAPPA(:,:,:,:)
      INTEGER, ALLOCATABLE :: NKPTS_PRIM_INDEX(:,:), INDEX_IN_IRZ(:)
      TYPE (wavedes1)    WDES1
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL,NB_GLOBAL,ISPINOR
      REAL(q)  GTRANS, AP, SUM_KAPPA
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

      CALL KPROJ_READER(IU5, IU0)
      ! quick return if possible
      IF (.NOT. LKPROJ) RETURN

#ifdef MPI
      IF (W%WDES%COMM_INB%NCPU /=1) THEN
         CALL VTUTOR('E','KPROJ',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU6,3)
         CALL VTUTOR('E','KPROJ',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
         RETURN
      ENDIF
#endif

      IF (IU0>=0) THEN
         WRITE(IU0,*) 'start k-points projection onto the first Brillouin zone of primitive cell.'
         WRITE(IU0,*) 'reading POSCAR.prim'
      ENDIF
!=======================================================================
! read header of POSCAR.prim file to get NTYPD, NTYPDD, NIOND and NIONPD
!=======================================================================
      CALL RD_POSCAR_PRIM_HEAD(LATT_PRIM, T_INFO_PRIM, &
     &           NIOND, NIONPD, NTYPD,NTYPPD, IU0, IU6)

      ALLOCATE(T_INFO_PRIM%ATOMOM(3*NIOND))
      T_INFO_PRIM%ATOMOM=0

      CALL RD_POSCAR_PRIM(LATT_PRIM, T_INFO_PRIM, DYN_PRIM, &
     &           NIOND, NIONPD, NTYPD,NTYPPD, IU0, IU6)

      IF (SYM%ISYM>0) THEN
         CALL INISYM(LATT_PRIM%A, DYN_PRIM%POSION, DYN_PRIM%VEL, T_INFO_PRIM%LSFOR, &
              T_INFO_PRIM%LSDYN,T_INFO_PRIM%NTYP,T_INFO_PRIM%NITYP,NIOND, &
              SYM%PTRANS,SYM%NROT,SYM%NPTRANS,SYM%ROTMAP, &
              SYM%TAU,SYM%TAUROT,SYM%WRKROT, &
              SYM%INDROT,T_INFO_PRIM%ATOMOM,W%WDES%SAXIS,SYM%MAGROT,W%WDES%NCDIJ,IU6)
      ENDIF

      DEALLOCATE(T_INFO_PRIM%ATOMOM)

      NGVECTORM=1
      DO NK=1,W%WDES%NKPTS
         NGVECTOR=W%WDES%NGVECTOR(NK)
         NGVECTORM=MAX(NGVECTOR,NGVECTORM)
         !IF (IU0>=0) write(IU0,*) 'NK,NGVECTORM=', NK, NGVECTORM
      ENDDO

      ALLOCATE(VKPT(3,NGVECTORM*W%WDES%NKPTS),NKPTS_PRIM_INDEX(NGVECTORM,W%WDES%NKPTS))

      VKPT=0
      NKPTS_PRIM=0
!=======================================================================
! main loop over all special points
!=======================================================================
      kpoint: DO NK=1,W%WDES%NKPTS
         NGVECTOR=W%WDES%NGVECTOR(NK)
         CALL SETWDES(W%WDES,WDES1,NK)
!=======================================================================
! loop over all G-vectors in the basis at this k-point
!=======================================================================
         DO IND=1,NGVECTOR

            N1=MOD(W%WDES%IGX(IND,NK)+GRID%NGX,GRID%NGX)+1
            N2=MOD(W%WDES%IGY(IND,NK)+GRID%NGY,GRID%NGY)+1 
            N3=MOD(W%WDES%IGZ(IND,NK)+GRID%NGZ,GRID%NGZ)+1

! get G-vector of respective k-point and coefficient
            G1=(GRID%LPCTX(N1)+W%WDES%VKPT(1,NK))
            G2=(GRID%LPCTY(N2)+W%WDES%VKPT(2,NK))
            G3=(GRID%LPCTZ(N3)+W%WDES%VKPT(3,NK))
            GIX=(G1*LATT_CUR%B(1,1)+G2*LATT_CUR%B(1,2)+G3*LATT_CUR%B(1,3))
            GIY=(G1*LATT_CUR%B(2,1)+G2*LATT_CUR%B(2,2)+G3*LATT_CUR%B(2,3))
            GIZ=(G1*LATT_CUR%B(3,1)+G2*LATT_CUR%B(3,2)+G3*LATT_CUR%B(3,3))

!=======================================================================
! bring the G vectors into the first BZ of primitive cell
!=======================================================================
            R1=(GIX*LATT_PRIM%A(1,1)+GIY*LATT_PRIM%A(2,1)+GIZ*LATT_PRIM%A(3,1))
            R2=(GIX*LATT_PRIM%A(1,2)+GIY*LATT_PRIM%A(2,2)+GIZ*LATT_PRIM%A(3,2))
            R3=(GIX*LATT_PRIM%A(1,3)+GIY*LATT_PRIM%A(2,3)+GIZ*LATT_PRIM%A(3,3))

            K1=MOD(R1+1000.5_q,1._q)-0.5_q
            K2=MOD(R2+1000.5_q,1._q)-0.5_q
            K3=MOD(R3+1000.5_q,1._q)-0.5_q

!=======================================================================
! find the corresponding K and make a list of k-vectors in prim cell
!=======================================================================
            KPT=FIND_KPOINT_IN_PC(K1,K2,K3,VKPT,NKPTS_PRIM)

            ! k-point not found in the existing list
            ! add the k-point to the list
            IF (KPT<0) THEN
               NKPTS_PRIM=NKPTS_PRIM+1
               VKPT(1,NKPTS_PRIM)=K1
               VKPT(2,NKPTS_PRIM)=K2
               VKPT(3,NKPTS_PRIM)=K3
               KPT=NKPTS_PRIM
            ENDIF
            NKPTS_PRIM_INDEX(IND,NK)=KPT

         ENDDO
      ENDDO kpoint
      !IF (IU0>=0) WRITE(IU0,*) NKPTS_PRIM_INDEX
      IF (IU0>=0) WRITE(IU0,*) 'K-point list is generated..'
      IF (IU6>=0) WRITE(IU6,'(A,I6)') 'NKPTS_PRIM=', NKPTS_PRIM
      
      ALLOCATE(VKPT_IRZ(3,NKPTS_PRIM), INDEX_IN_IRZ(NKPTS_PRIM),WTKPT_IRZ(NKPTS_PRIM))

      CALL IBZKPT_LIST(LATT_PRIM, VKPT, NKPTS_PRIM, VKPT_IRZ, NKPTS_IRZ, &
               INDEX_IN_IRZ, SYM%ROTMAP, SYM%MAGROT, SYM%ISYM, IU6, IU0)
      
      WTKPT_IRZ=0
      DO NK=1,NKPTS_PRIM
         WTKPT_IRZ(INDEX_IN_IRZ(NK))=WTKPT_IRZ(INDEX_IN_IRZ(NK))+1
      ENDDO

      IF (IU0>=0) WRITE(IU0,*) 'Symmetry operation is done...'
      !IF (IU0>=0) WRITE(IU0,*) INDEX_IN_IRZ

      ALLOCATE(KAPPA(W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN,NKPTS_IRZ))

      KAPPA=0
!=======================================================================
! loop over k-points and bands
!=======================================================================
      spin:    DO ISP=1,W%WDES%ISPIN
      kpoints: DO NK=1,W%WDES%NKPTS
         CALL SETWDES(W%WDES,WDES1,NK)
         NGVECTOR=W%WDES%NGVECTOR(NK)
      band: DO NB_GLOBAL=1,W%WDES%NB_TOT
         N=NB_LOCAL(NB_GLOBAL,WDES1)
         IF(N==0) CYCLE band
!=======================================================================
! take one wavefunction loop over all G vector and add Kappa 
! to the array Kappa that has the size of the number of k-points
!=======================================================================
         DO ISPINOR=0,WDES1%NRSPINORS-1
         DO IND=1,NGVECTOR
            KPT=INDEX_IN_IRZ(NKPTS_PRIM_INDEX(IND,NK))
            IF (KPT==0) THEN
               WRITE(0,*) 'internal error in KPROJ: G vector was not assigned to k-vector', & 
               W%WDES%IGX(IND,NK),W%WDES%IGY(IND,NK),W%WDES%IGZ(IND,NK)
               STOP
            ENDIF
            TEMP=REAL(W%CPTWFP(IND+ISPINOR*NGVECTOR,N,NK,ISP)*CONJG(W%CPTWFP(IND+ISPINOR*NGVECTOR,N,NK,ISP)))
            KAPPA(NB_GLOBAL,NK,ISP,KPT)=KAPPA(NB_GLOBAL,NK,ISP,KPT)+TEMP
         ENDDO
         ENDDO
      ENDDO band
      ENDDO kpoints
      ENDDO spin
      CALLMPI( M_sum_d( W%WDES%COMM, KAPPA(1,1,1,1), SIZE(KAPPA)) )

!     IF (IU6>=0) THEN
!        WRITE(IU6,*) 'KPROJ_THRESHOLD=', KPROJ_THRESHOLD
!        WRITE(IU6,*) 
!        WRITE(IU6,*) '|K1|, eigenvalues, KAPPA'
!        WRITE(IU6,*) 
!     ENDIF
!     DO ISP=1,W%WDES%ISPIN
!     DO NK=1,W%WDES%NKPTS
!     DO KPT=1,NKPTS_IRZ
!     DO N=1,W%WDES%NB_TOT
!        SUM_KAPPA=SUM(KAPPA(N,NK,ISP,1:NKPTS_IRZ))
!        KAPPA(N,NK,ISP,1:NKPTS_IRZ)=KAPPA(N,NK,ISP,1:NKPTS_IRZ)/SUM_KAPPA
!
!     ! for Gamma-L direction 
!        IF (ABS(VKPT_IRZ(3,KPT))==ABS(VKPT_IRZ(2,KPT)) .AND. &
!            ABS(VKPT_IRZ(1,KPT))==ABS(VKPT_IRZ(2,KPT))) THEN
!           IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!              IF (IU6>=0) THEN
!                 WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(2,KPT)), &
!                                       REAL(W%CELTOT(N,NK,ISP)), &
!                                       KAPPA(N,NK,ISP,KPT)
!              ENDIF
!           ENDIF
!        ELSEIF (ABS(VKPT_IRZ(3,KPT))>1.E-3_q .AND. &
!            ABS(VKPT_IRZ(1,KPT))<1.E-4_q .AND. &
!            ABS(VKPT_IRZ(2,KPT))<1.E-4_q) THEN
!           IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!              IF (IU6>=0) THEN
!                 WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(3,KPT)), &
!                                       REAL(W%CELTOT(N,NK,ISP)), &
!                                       KAPPA(N,NK,ISP,KPT)
!              ENDIF
!           ENDIF
!        ELSEIF (ABS(VKPT_IRZ(2,KPT))>1.E-3_q .AND. &
!            ABS(VKPT_IRZ(1,KPT))<1.E-4_q .AND. &
!            ABS(VKPT_IRZ(3,KPT))<1.E-4_q) THEN
!           IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!              IF (IU6>=0) THEN
!                 WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(2,KPT)), &
!                                       REAL(W%CELTOT(N,NK,ISP)), &
!                                       KAPPA(N,NK,ISP,KPT)
!              ENDIF
!           ENDIF
!        ELSEIF (ABS(VKPT_IRZ(1,KPT))>1.E-3_q .AND. &
!            ABS(VKPT_IRZ(2,KPT))<1.E-4_q .AND. &
!            ABS(VKPT_IRZ(3,KPT))<1.E-4_q) THEN
!           IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!              IF (IU6>=0) THEN
!                 WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(1,KPT)), &
!                                       REAL(W%CELTOT(N,NK,ISP)), &
!                                       KAPPA(N,NK,ISP,KPT)
!              ENDIF
!           ENDIF
!        ENDIF
!     !! for Gamma-L direction 
!
!     ! for Gamma-X direction 
!     !   IF (ABS(VKPT_IRZ(3,KPT))<1.E-3_q .AND. &
!     !       ABS(VKPT_IRZ(1,KPT)-VKPT_IRZ(2,KPT))<1.E-3_q) THEN
!     !      IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!     !         IF (IU6>=0) THEN
!     !            WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(2,KPT)), &
!     !                                  REAL(W%CELTOT(N,NK,ISP)), &
!     !                                  KAPPA(N,NK,ISP,KPT)
!     !         ENDIF
!     !      ENDIF
!     !   ELSEIF (ABS(VKPT_IRZ(2,KPT))<1.E-3_q .AND. &
!     !       ABS(VKPT_IRZ(1,KPT)-VKPT_IRZ(3,KPT))<1.E-3_q) THEN
!     !      IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!     !         IF (IU6>=0) THEN
!     !            WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(3,KPT)), &
!     !                                  REAL(W%CELTOT(N,NK,ISP)), &
!     !                                  KAPPA(N,NK,ISP,KPT)
!     !         ENDIF
!     !      ENDIF
!     !   ELSEIF (ABS(VKPT_IRZ(1,KPT))<1.E-3_q .AND. &
!     !       ABS(VKPT_IRZ(2,KPT)-VKPT_IRZ(3,KPT))<1.E-3_q) THEN
!     !      IF (KAPPA(N,NK,ISP,KPT)>KPROJ_THRESHOLD) THEN
!     !         IF (IU6>=0) THEN
!     !            WRITE(IU6,'(3F12.6)') ABS(VKPT_IRZ(3,KPT)), &
!     !                                  REAL(W%CELTOT(N,NK,ISP)), &
!     !                                  KAPPA(N,NK,ISP,KPT)
!     !         ENDIF
!     !      ENDIF
!     !   ENDIF
!     ! for Gamma-X direction 
!     ENDDO
!     ENDDO
!     ENDDO
!     ENDDO
      
      IF (IU6>=0) THEN
         OPEN(UNIT=110,FILE=DIR_APP(1:DIR_LEN)//'PRJCAR',STATUS='UNKNOWN')
         WRITE(110,'(A)') 'Basis vectors reciprocal space of POSCAR.prim (units of 2pi):'
         WRITE(110,'(3F14.7)') LATT_PRIM%B
         WRITE(110,'(/A,I6)') 'number of k-points in IBZ of POSCAR.prim:',NKPTS_IRZ
         WRITE(110,'(/A)') "             b1            b2            b3      weight"
         DO KPT=1,NKPTS_IRZ
            WRITE(110,'(I4,3F14.7,3X,I4)') KPT,VKPT_IRZ(:,KPT),INT(WTKPT_IRZ(KPT))
         ENDDO

         DO ISP=1,W%WDES%ISPIN
            WRITE(110,'(/A,I4)') 'spin component:',ISP
            DO NK=1,W%WDES%NKPTS
               WRITE(110,'("k-point (associated with POSCAR):",I6,2X,"vkpt:",3F14.7,2X,"weight:",F14.7)') &
              &   NK,W%WDES%VKPT(:,NK),W%WDES%WTKPT(NK)
               DO N=1,W%WDES%NB_TOT
                  WRITE(110,'("band:",I6,2X,"energy:",F14.7)') N,REAL(W%CELTOT(N,NK,ISP),q)
                  I=0
                  DO KPT=1,NKPTS_IRZ
                     WRITE(110,'(E15.7)',ADVANCE='No') KAPPA(N,NK,ISP,KPT)
                     I=I+1; IF (MOD(I,10)==0) WRITE(110,*)
                  ENDDO 
                  IF (MOD(I,10)/=0) WRITE(110,*)
               ENDDO
            ENDDO
         ENDDO 
         CLOSE(110)
      ENDIF

      CALL XML_TAG("kpoints", comment="kpoints in IRZ of POSCAR.prim")
      CALL XML_KPOINTS_LIST(VKPT_IRZ(:,1:NKPTS_IRZ), WTKPT_IRZ(1:NKPTS_IRZ))
      CALL XML_CLOSE_TAG("kpoints")
      
      CALL XML_KPROJ(KAPPA, W%CELTOT, W%FERTOT, W%WDES%NB_TOT, W%WDES%NKPTS, W%WDES%ISPIN, NKPTS_IRZ)

      DEALLOCATE(VKPT, NKPTS_PRIM_INDEX, VKPT_IRZ, WTKPT_IRZ, INDEX_IN_IRZ)
      DEALLOCATE(KAPPA)

    END SUBROUTINE KPROJ


!******************** SUBROUTINE IBZKPT_LIST ***************************
!
! subroutine IBZKPT_LIST is deduced from ibzkpt
! it determines a list of k-points inside the IRZ by applying all
! symmetry operations to a supplied list of k-points
!
! it returns:        an array linking the k-points in IRZ 
!    to k-points in the full Brillouin zone
!
!***********************************************************************

      SUBROUTINE IBZKPT_LIST(LATT_CUR, VKPT, NKPTS, VKPT_IRZ, NKPTS_IRZ, INDEX_IN_IRZ, & 
           ROTMAP, MAGROT, ISYM, IU6, IU0)
      USE lattice
      USE mkpoints
      USE main_mpi

      IMPLICIT NONE
! passed structures and variables
      TYPE (latt)           LATT_CUR 
      INTEGER ISYM
      INTEGER               IU6,IU0
      INTEGER ROTMAP(:,:,:)
      REAL(q) :: VKPT(:,:)       ! k-points in the Brillouin zone
      REAL(q) :: VKPT_IRZ(:,:)   ! k-points in the IRZ
      INTEGER :: NKPTS           ! total number of k-points
      INTEGER :: NKPTS_IRZ       ! number of k-points in the IRZ
      INTEGER :: INDEX_IN_IRZ(:) ! index from each k-point in the Brillouin zone into IRZ
      REAL(q) MAGROT(:,:)
! common symmetry variables
      INTEGER ISYMOP, NROT, IGRPOP, NROTK, INVMAP, NPCELL
      REAL(q) GTRANS, AP
      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
! local variables and structures
      REAL(q) TINY, V(3), VR(3), VT(3), ROP(3,3)
      REAL(q) KPT_WEIGHT(NKPTS)
      INTEGER INVERS(9),IOP(3,3)
      INTEGER NK,NKF,NOP
      LOGICAL LINV
! external routine
      LOGICAL,EXTERNAL :: SGREQL
      LOGICAL,ALLOCATABLE :: KPOINT_ALREADY_FOUND(:)
! set data
      DATA TINY /1.E-6_q/, INVERS /-1,0,0,0,-1,0,0,0,-1/

      ALLOCATE(KPOINT_ALREADY_FOUND(NKPTS))

      KPOINT_ALREADY_FOUND=.FALSE.
      NKPTS_IRZ=0
      INDEX_IN_IRZ=0
      LINV=.FALSE.
!=======================================================================
! now do all point group operations with each k-point and check wether we
! generate a new k-point (in 1st BZ). If so, store it as well as the
! generating operation. By the way, check whether inversion is already one 
! of the sym ops.
!=======================================================================
      DO NK=1,NKPTS
       IF (KPOINT_ALREADY_FOUND(NK)) THEN
          CYCLE
       ELSE
! store the k-point
         NKPTS_IRZ=NKPTS_IRZ+1
         IF (NKPTS_IRZ>SIZE(VKPT_IRZ,2)) THEN
            WRITE(*,*) 'IBZKPT_LIST: internal error.. VKPT_IRZ is not sufficiently large'
            STOP
         ENDIF
         INDEX_IN_IRZ(NK)=NKPTS_IRZ
         VKPT_IRZ(1,NKPTS_IRZ)=VKPT(1,NK)
         VKPT_IRZ(2,NKPTS_IRZ)=VKPT(2,NK)
         VKPT_IRZ(3,NKPTS_IRZ)=VKPT(3,NK)             

         DO NOP=1,NROTK
! test existence of inversion op
            IF (SGREQL(IGRPOP(1,1,NOP),INVERS)) LINV=.TRUE.
! copy symmetry op to real array
            ROP=IGRPOP(:,:,NOP)
! make new k-point and shift (for testing) it to 1st BZ
            VR(1)=VKPT(1,NK)
            VR(2)=VKPT(2,NK)
            VR(3)=VKPT(3,NK) 
            V(1)=VR(1)*ROP(1,1)+VR(2)*ROP(2,1)+VR(3)*ROP(3,1)
            V(2)=VR(1)*ROP(1,2)+VR(2)*ROP(2,2)+VR(3)*ROP(3,2)
            V(3)=VR(1)*ROP(1,3)+VR(2)*ROP(2,3)+VR(3)*ROP(3,3)
! bring the point to the primitive cell
            VT(1)=MOD(V(1)+6.5_q,1._q)-0.5_q
            VT(2)=MOD(V(2)+6.5_q,1._q)-0.5_q
            VT(3)=MOD(V(3)+6.5_q,1._q)-0.5_q
! test against all other k-points
            test1: DO NKF=1,NKPTS
               IF(( ABS(MOD(VT(1)-VKPT(1,NKF)+6.5,1._q)-0.5_q)<TINY) .AND. &
                  ( ABS(MOD(VT(2)-VKPT(2,NKF)+6.5,1._q)-0.5_q)<TINY) .AND. &
                  ( ABS(MOD(VT(3)-VKPT(3,NKF)+6.5,1._q)-0.5_q)<TINY)) THEN
                  INDEX_IN_IRZ(NKF)=NKPTS_IRZ
                  KPOINT_ALREADY_FOUND(NKF)=.TRUE.
                  EXIT test1
               ENDIF
            ENDDO test1
         ENDDO

!=======================================================================
! did not find LINV -> now we have to do it all over again with 
! all operators multiplied with INVERS
!=======================================================================
         IF (.NOT. LINV .AND. ISYM>=0) THEN
            DO NOP=1,NROTK
               ! apply inversion symmetry to form to get IOP
               CALL SGRPRD(INVERS,IGRPOP(1,1,NOP),IOP(1,1))
               ROP=IOP  ! copy symmetry op to real array
! make new k-point and shift it (for testing) to 1st BZ
               VR(1)=VKPT(1,NK)
               VR(2)=VKPT(2,NK)
               VR(3)=VKPT(3,NK) 
               V(1)=VR(1)*ROP(1,1)+VR(2)*ROP(2,1)+VR(3)*ROP(3,1)
               V(2)=VR(1)*ROP(1,2)+VR(2)*ROP(2,2)+VR(3)*ROP(3,2)
               V(3)=VR(1)*ROP(1,3)+VR(2)*ROP(2,3)+VR(3)*ROP(3,3)
! bring the point to the primitive cell
               VT(1)=MOD(V(1)+6.5_q,1._q)-0.5_q
               VT(2)=MOD(V(2)+6.5_q,1._q)-0.5_q
               VT(3)=MOD(V(3)+6.5_q,1._q)-0.5_q
! test against all other k-points
               test2: DO NKF=1,NKPTS
                  IF(( ABS(MOD(VT(1)-VKPT(1,NKF)+6.5,1._q)-0.5)<TINY) .AND. &
                     ( ABS(MOD(VT(2)-VKPT(2,NKF)+6.5,1._q)-0.5)<TINY) .AND. &
                     ( ABS(MOD(VT(3)-VKPT(3,NKF)+6.5,1._q)-0.5)<TINY)) THEN
                     INDEX_IN_IRZ(NKF)=NKPTS_IRZ
                     KPOINT_ALREADY_FOUND(NKF)=.TRUE.
                     EXIT test2
                  ENDIF
               ENDDO test2
            ENDDO
         ENDIF
        ENDIF
      ENDDO


      KPT_WEIGHT=0
      DO NK=1,NKPTS
         KPT_WEIGHT(INDEX_IN_IRZ(NK))=KPT_WEIGHT(INDEX_IN_IRZ(NK))+1
      ENDDO

      DEALLOCATE(KPOINT_ALREADY_FOUND)

      IF (IU6>=0) THEN
         WRITE(IU6,*)
         WRITE(IU6,*) 'reciprocal lattice vectors of the primitive cell:'
         WRITE(IU6,'(3X,3F13.9)') LATT_CUR%B
         WRITE(IU6,*)
         WRITE(IU6,*) 'number of k-points in IRZ ',NKPTS_IRZ
         WRITE(IU6,*)' k-points in reciprocal lattice and weights: '
         DO NK=1,NKPTS_IRZ
            WRITE(IU6,'(1X,3F12.8,F12.3)') VKPT_IRZ(:,NK), KPT_WEIGHT(NK)
         ENDDO
      ENDIF
    END SUBROUTINE IBZKPT_LIST


END MODULE mkproj
