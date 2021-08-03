       MODULE dynconstr
        USE random_seeded
        USE prec
        USE constant
        USE poscar
        USE base
        USE lattice
        USE ini
        USE chain
        USE vaspxml
        USE mymath
        USE internals
        USE wave
        USE main_mpi
        USE npt_dynamics
        IMPLICIT NONE 
        !INCLUDE "hills.inc"

        TYPE hills_io
          INTEGER :: STRUCTINPUT
          INTEGER :: PENALTY
        END TYPE hills_io

        TYPE penalty_data
          INTEGER         :: number
          TYPE(gauss_peak),POINTER :: gauss(:)
          REAL(q),POINTER :: force(:)
          REAL(q),POINTER :: wall(:,:)
        END TYPE penalty_data

        TYPE gauss_peak
          REAL(q),POINTER :: position(:)
          REAL(q)         :: high
          REAL(q)         :: width
        END TYPE gauss_peak

        TYPE hills_data
          TYPE(gauss_peak),POINTER :: gauss(:)
          REAL(q),POINTER :: velocity(:)
          REAL(q),POINTER :: position(:)
          REAL(q),POINTER :: mass(:)
          REAL(q),POINTER :: force_constant(:)
          REAL(q),POINTER :: force(:)
          REAL(q)         :: SNOSE(4)
          REAL(q)         :: SQQ
          REAL(q)         :: potential
          REAL(q)         :: stride
          REAL(q)         :: andersen_prob
          REAL(q)         :: temperature        !temperature of fict. particles
          INTEGER         :: number
          INTEGER         :: bin
          INTEGER         :: maxstride
          LOGICAL         :: variable_width
        END TYPE
        CONTAINS

        SUBROUTINE STEP_tb(DYN,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EKIN_LAT,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,g_io,WDES,SIF,ISCALE,TEIN,TIFOR)
        !c main driver for different MD algorithms
          TYPE(dynamics) :: DYN
          TYPE(dynamics),SAVE :: DYN_dummy
          TYPE(type_info) :: T_INFO
          TYPE(type_info),SAVE :: T_INFO_dummy
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io          
          TYPE (wavedes)  ::   WDES   
          TYPE(coordstructure),SAVE :: ICOORDINATES
          TYPE(hills_data),SAVE :: hills
          TYPE(penalty_data),SAVE :: penalty     
          REAL(q) :: EKIN,EKIN_Lat,EPS,ES,DISMAX,EPOT,TEMPER,TEIN
          INTEGER :: i,j,K
          INTEGER :: NDEGREES_OF_FREEDOM,cDOF
          INTEGER,SAVE :: iconst0,iconst2,iconst5,iconst6,iconst7,iconst8,iconst9
          REAL(q),SAVE :: ANDERSEN_PROB
          REAL(q),SAVE,ALLOCATABLE :: REF_C(:,:)
          INTEGER,SAVE ::MDALGO,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: SCALING=0
          INTEGER,SAVE :: counter=0
          INTEGER,SAVE :: NSUBSYS(3)
          REAL(q),SAVE :: TSUBSYS(3),PSUBSYS(3)
          REAL(q),SAVE,ALLOCATABLE :: GAMMA(:) ! friction coeficints in ps^(-1) for atomic coordinates
          REAL(q),SAVE :: GAMMA_L  ! friction coeficints in ps^(-1) for lattice components
          REAL(q) :: SIF(3,3)
          INTEGER :: ISCALE
          INTEGER IDUM
          LOGICAL ::LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q) :: TIFOR(3,T_INFO%NIONS)
          INTEGER :: Ltxyz(3)
          LOGICAL,SAVE :: LSCALE=.TRUE. !c rescale initial velocities so as to have exact Tinst
          LOGICAL, SAVE :: LCMASS=.TRUE. !c remove the center-of-mass motion?
          REAL(q) :: CMASS(3,1)
          INTEGER,SAVE :: NDUMMYATOMS
          REAL(q) :: DUMMY_POSITIONS(3)
          LOGICAL, SAVE :: LDUMMYATOMS=.FALSE.
          REAL(q), SAVE, ALLOCATABLE :: DUMMY_K(:),DUMMY_R0(:),hspring_R0(:),hspring_V0(:),hspring_K(:),hspring_force(:)
          REAL(q) :: DUMMY_MASS,ESPRING
          REAL(q),SAVE,ALLOCATABLE :: spring_accel(:,:)
          LOGICAL :: Lhspring_V0=.FALSE.
         

          counter=counter+1
          !CALL XML_VEL(T_INFO%NIONS,DYN%VEL)

          !DYN%POSIOC=DYN%POSION

          TEIN=0._q;ESPRING=0._q

          IF (counter==1) THEN
            NSUBSYS=T_INFO%NIONS
            TSUBSYS=0._q
            PSUBSYS=0._q
            ALLOCATE(GAMMA(T_INFO%NTYP))
            GAMMA=0._q
            NDUMMYATOMS=0;DUMMY_MASS=1._q
            DUMMY_POSITIONS=0._q
            CALL DYN_READER(IO,MDALGO,ANDERSEN_PROB,SCALING,NSUBSYS,TSUBSYS,PSUBSYS,GAMMA, &
            &    GAMMA_L,NDUMMYATOMS,DUMMY_POSITIONS,DUMMY_MASS,T_INFO%NIONS,T_INFO%NTYP)
            
            IF (MDALGO==3 .OR. MDALGO==31 ) LCMASS=.FALSE.

            !c  georg's fix 
            IF (MDALGO>0) THEN
            ! copy current positions considered in VASP to POSIOC
               DYN%POSIOC=DYN%POSION
            ENDIF
            
            !c take care of dummy atoms
            IF (NDUMMYATOMS>0) THEN
              LDUMMYATOMS=.TRUE.
              ALLOCATE (DYN_dummy%POSION(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (DYN_dummy%POSIOC(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (DYN_dummy%VEL(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (DYN_dummy%D2(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (DYN_dummy%D2C(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (DYN_dummy%D3(3,T_INFO%NIONS+NDUMMYATOMS))
              !ALLOCATE (DYN_dummy%POMASS(T_INFO%NTYP+1))
              DYN_dummy%POSION(:,1:T_INFO%NIONS)=DYN%POSION(:,1:T_INFO%NIONS)
              DYN_dummy%POSION(:,T_INFO%NIONS+1)=DUMMY_POSITIONS
              DYN_dummy%POSIOC=DYN_dummy%POSION
              !DYN_dummy%POSIOC(:,1:T_INFO%NIONS)=DYN%POSIOC(:,1:T_INFO%NIONS)
              DYN_dummy%VEL=0._q;DYN_dummy%D2=0._q;DYN_dummy%D2C=0._q;DYN_dummy%D3=0._q
              DYN_dummy%VEL(:,1:T_INFO%NIONS)=DYN%VEL(:,1:T_INFO%NIONS)
              DYN_dummy%D2(:,1:T_INFO%NIONS)=DYN%D2(:,1:T_INFO%NIONS)
              DYN_dummy%D2C(:,1:T_INFO%NIONS)=DYN%D2C(:,1:T_INFO%NIONS)
              DYN_dummy%D3(:,1:T_INFO%NIONS)=DYN%D3  (:,1:T_INFO%NIONS)            
              !DYN_dummy%POMASS(1:T_INFO%NTYP)=DYN%POMASS(1:T_INFO%NTYP)
              DYN_dummy%A=DYN%A
              DYN_dummy%AC=DYN%AC
              DYN_dummy%SNOSE=DYN%SNOSE
              DYN_dummy%IBRION=DYN%IBRION
              DYN_dummy%ISIF=DYN%ISIF
              DYN_dummy%POTIM=DYN%POTIM
              DYN_dummy%EDIFFG=DYN%EDIFFG
              DYN_dummy%SMASS=DYN%SMASS
              DYN_dummy%PSTRESS=DYN%PSTRESS
              DYN_dummy%TEBEG=DYN%TEBEG             
              DYN_dummy%TEEND=DYN%TEEND
              DYN_dummy%TEMP=DYN%TEMP
              DYN_dummy%NSW=DYN%NSW
              DYN_dummy%NBLOCK=DYN%NBLOCK
              DYN_dummy%KBLOCK=DYN%KBLOCK
              DYN_dummy%INIT=DYN%INIT
              DYN_dummy%NFREE=DYN%NFREE
              
              ALLOCATE (T_INFO_dummy%POSION(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (T_INFO_dummy%LSFOR(3,T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (T_INFO_dummy%ITYP(T_INFO%NIONS+NDUMMYATOMS))
              ALLOCATE (T_INFO_dummy%NITYP(T_INFO%NTYPPD+1))
              ALLOCATE (T_INFO_dummy%POMASS(T_INFO%NTYP+1))
              
              T_INFO_dummy%POSION=0._q
              T_INFO_dummy%POSION(:,1:T_INFO%NIONS)=T_INFO%POSION(:,1:T_INFO%NIONS)
              T_INFO_dummy%POSION(:,T_INFO%NIONS+1)=DUMMY_POSITIONS
              
              T_INFO_dummy%LSFOR(:,1:T_INFO%NIONS)=T_INFO%LSFOR(:,1:T_INFO%NIONS)
              T_INFO_dummy%LSFOR(:,T_INFO%NIONS:)=.TRUE.
              
              T_INFO_dummy%ITYP(1:T_INFO%NIONS)=T_INFO%ITYP(1:T_INFO%NIONS)
              T_INFO_dummy%ITYP(T_INFO%NIONS+1:)=T_INFO_dummy%ITYP(T_INFO%NIONS)+1
              
              T_INFO_dummy%NITYP(1:T_INFO%NTYPPD)=T_INFO%NITYP(1:T_INFO%NTYPPD)
              T_INFO_dummy%NITYP(T_INFO%NTYPPD+1)=NDUMMYATOMS
              
              T_INFO_dummy%POMASS(1:T_INFO%NTYP)=T_INFO%POMASS(1:T_INFO%NTYP)
              T_INFO_dummy%POMASS(T_INFO%NTYP+1:)=DUMMY_MASS
              
              T_INFO_dummy%NTYP=T_INFO%NTYP+1
              T_INFO_dummy%NIOND=T_INFO%NIONS+NDUMMYATOMS
              T_INFO_dummy%NIONPD=T_INFO%NIONPD+NDUMMYATOMS
              T_INFO_dummy%NIONS=T_INFO%NIONS+NDUMMYATOMS
              T_INFO_dummy%NIONP=T_INFO%NIONP+NDUMMYATOMS
              T_INFO_dummy%LSDYN=T_INFO%LSDYN
              T_INFO_dummy%LDIRCO=T_INFO_dummy%LDIRCO
            ENDIF
                                   
            IF (LDUMMYATOMS) THEN
              NDEGREES_OF_FREEDOM_= NDEGREES_OF_FREEDOM+3*NDUMMYATOMS
            ELSE
              NDEGREES_OF_FREEDOM_=NDEGREES_OF_FREEDOM
            ENDIF

            !c Langevin dynamics does not fix total momentum
            !c hence NDEGREES_OF_FREEDOM=3*T_INFO%NIONS, not 3*T_INFO%NIONS-3
            IF (MDALGO==3 .OR. MDALGO==31) THEN
               Ltxyz=1
               DO i=1,T_INFO%NIONS
                 DO j=1,3
                   IF (.NOT. T_INFO%LSFOR(j,i)) THEN
                     Ltxyz(j)=0
                   ENDIF
                 ENDDO
               ENDDO
               NDEGREES_OF_FREEDOM_=NDEGREES_OF_FREEDOM+SUM(Ltxyz)
             ENDIF


            !c write out some outpout
            IF (MDALGO>0) THEN
              IF (IO%IU6>0) THEN
                !c write the input parameter summary into OUTCAR
                write(IO%IU6,FMT='(/,2X,A22)') 'MD-specific parameters'
                write(IO%IU6,FMT='(3X,A22,X,I2)')    '             MDALGO = ', MDALGO
                IF (MDALGO==1 .OR. MDALGO==10 .OR. MDALGO==11) THEN
                  write(IO%IU6,FMT='(3X,A22,X,F10.8)')         '      ANDERSEN_PROB = ',ANDERSEN_PROB
                ENDIF
                IF (MDALGO==3 .OR. MDALGO==31) THEN
                  write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)')  '     LANGEVIN_GAMMA = '
                  DO i=1,T_INFO%NTYP
                    write(IO%IU6,ADVANCE='NO',FMT='(X,F8.3)') GAMMA(i)
                  ENDDO
                  write(IO%IU6,ADVANCE='NO',FMT='(/)')
                  IF (DYN%ISIF==3) THEN
                    write(IO%IU6,FMT='(3X,A22,X,F8.3)')  '   LANGEVIN_GAMMA_L = ',GAMMA_L
                  ENDIF
                ENDIF

                !write(g_io%REPORT,FMT='(3X,A22,X,I6)')    '            SCALING = ' , SCALING
                IF (MDALGO==13) THEN              
                  write(IO%IU6,FMT='(3X,A22,X,I5,X,I5,X,I5)')    '           NSUBSYS = ', NSUBSYS(1), NSUBSYS(2),NSUBSYS(3)
                  write(IO%IU6,FMT='(3X,A22,X,F8.3,X,F8.3,X,F8.3)')    '           TSUBSYS = ', TSUBSYS(1), TSUBSYS(2),TSUBSYS(3)
                  write(IO%IU6,FMT='(3X,A22,X,F8.3,X,F8.3,X,F8.3)')    '           PSUBSYS = ', PSUBSYS(1), PSUBSYS(2),PSUBSYS(3)
                ENDIF

                !c write the input parameter summary into REPORT
                write(g_io%REPORT,FMT='(3X,A22,X,I2)')    '             MDALGO = ', MDALGO
                IF (MDALGO==1 .OR. MDALGO==10 .OR. MDALGO==11 .OR. MDALGO==4) THEN
                  write(g_io%REPORT,FMT='(3X,A22,X,F10.8)') '      ANDERSEN_PROB = ',ANDERSEN_PROB
                ENDIF

                IF (MDALGO==3 .OR. MDALGO==31) THEN
                  write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)')  '     LANGEVIN_GAMMA = '
                  DO i=1,T_INFO%NTYP
                    write(g_io%REPORT,ADVANCE='NO',FMT='(X,F8.3)') GAMMA(i)
                  ENDDO
                  write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
                  IF (DYN%ISIF==3) THEN
                    write(g_io%REPORT,FMT='(3X,A22,X,F8.3)')  '   LANGEVIN_GAMMA_L = ',GAMMA_L
                  ENDIF
                ENDIF

                write(g_io%REPORT,FMT='(3X,A22,X,I6)')    '            SCALING = ' , SCALING
                IF (MDALGO==13) THEN              
                  write(g_io%REPORT,FMT='(3X,A22,X,I5,X,I5,X,I5)')    '           NSUBSYS = ', NSUBSYS(1), NSUBSYS(2),NSUBSYS(3)
                  write(g_io%REPORT,FMT='(3X,A22,X,F8.3,X,F8.3,X,F8.3)')    '           TSUBSYS = ', TSUBSYS(1), TSUBSYS(2),TSUBSYS(3)
                  write(g_io%REPORT,FMT='(3X,A22,X,F8.3,X,F8.3,X,F8.3)')    '           PSUBSYS = ', PSUBSYS(1), PSUBSYS(2),PSUBSYS(3)
                ENDIF
              ENDIF
          
              nullify(ICOORDINATES%COORDSTRUCT)
              ALLOCATE(ICOORDINATES%COORDSTRUCT(10))
              ICOORDINATES%NUMINTERNALS=0
            
              cDOF=0
              IF (LDUMMYATOMS) THEN
                CALL constraints_reader(T_INFO_dummy,g_io,DYN_dummy%POSION,TRANSPOSE(LATT_CUR%A),ICOORDINATES,IO)
              ELSE
                CALL constraints_reader(T_INFO,g_io,DYN%POSION,TRANSPOSE(LATT_CUR%A),ICOORDINATES,IO)
              ENDIF
             
              hills%number=0
              iconst0=0
              iconst2=0
              iconst5=0
              iconst6=0
              iconst7=0
              iconst8=0
              iconst9=0
              !c identify type of coordinates defined in ICONST
              DO i=1,ICOORDINATES%NUMINTERNALS
                SELECT CASE(ICOORDINATES%COORDSTRUCT(i)%STATUS)
                  CASE(0) !c number of hard constraints
                    iconst0=iconst0+1
                  CASE(2)
                    iconst2=iconst2+1
                  CASE(5) !c number of collective variables for mMD
                    iconst5=iconst5+1
                  CASE(6)
                    iconst6=iconst6+1
                  CASE(7) !c number of monitored coordinates
                    iconst7=iconst7+1
                  CASE(8) !c harmonic springs will act on these internal coordinates
                    iconst8=iconst8+1
                  CASE(9) !c number of dummy atoms conections
                    iconst9=iconst9+1  
                END SELECT    
              ENDDO
              
              !c reference positions, their velocities, and force-constants for spring connections, internal coordinates
              IF (iconst8>0) THEN
                ALLOCATE(hspring_R0(iconst8),hspring_V0(iconst8),hspring_K(iconst8),hspring_force(iconst8))
                hspring_R0=0._q;hspring_K=0._q;hspring_force=0._q
                CALL HARMONIC_SPRING_READER(IO,iconst8,g_io,hspring_R0,hspring_V0,hspring_K,Lhspring_V0)
              ENDIF

              !c reference positions and force-constants for spring connections, dummy atoms
              IF (iconst9>0) THEN
                IF (LDUMMYATOMS) THEN 
                  ALLOCATE(spring_accel(3,T_INFO_dummy%NIONS))
                ELSE
                  ALLOCATE(spring_accel(3,T_INFO%NIONS))
                ENDIF           
                ALLOCATE(DUMMY_R0(iconst9),DUMMY_K(iconst9))
                spring_accel=0._q
                DUMMY_R0=0.;DUMMY_K=0._q
                CALL DUMMY_ATOM_READER(IO,iconst9,g_io,ICOORDINATES,MDALGO,DUMMY_R0,DUMMY_K,LDUMMYATOMS,DUMMY_POSITIONS,DUMMY_MASS) 
                !!!c convert the forceconstants from au to eV/A**2
                !!DUMMY_K=DUMMY_K*2*RYTOEV/AUTOA/AUTOA
              ENDIF
                                                   
              !c mMD stuff
              hills%number=iconst5+iconst6
              IF (MDALGO==10 .OR. MDALGO==11 .OR. MDALGO==20 .OR. MDALGO==21 .OR. MDALGO==31 ) THEN
                hills%number=0
                DO i=1,ICOORDINATES%NUMINTERNALS
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                    hills%number=hills%number+1
                  ENDIF
                ENDDO

                IF (hills%number<1 .AND. (MDALGO==10 .OR. MDALGO==20)) THEN
                  IF (IO%IU0>0) THEN
                    WRITE(IO%IU0,*) 'Error meta-MD: at least one coord. with STATUS=5 must be defined if MDALGO=10|20'
                  ENDIF
                  STOP
                ENDIF
                ALLOCATE(hills%mass(hills%number),hills%force_constant(hills%number))
                ALLOCATE(hills%velocity(hills%number),hills%position(hills%number))
                ALLOCATE(hills%force(hills%number))
                hills%mass=0._q; hills%force_constant=0._q
                hills%velocity=0._q; hills%position=0._q
                hills%potential=0._q; hills%force=0._q

                CALL HILLS_READER(IO,DYN,hills,g_io,ICOORDINATES,MDALGO)
                CALL PENALTY_READER(hills,penalty,IO,g_io)
                !OPEN(UNIT=g_io%STRUCTINPUT,FILE=DIR_APP(1:DIR_LEN)//'HILLSPOT',STATUS='REPLACE')
                !OPEN(UNIT=g_io%STRUCTINPUT,FILE='HILLSPOT',STATUS='REPLACE')
                OPEN(UNIT=g_io%STRUCTINPUT,FILE='HILLSPOT',STATUS='UNKNOWN')
                IF (g_io%STRUCTINPUT>=0) THEN
                  DO i=1,penalty%number
                    DO j=1,hills%number
                      write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') penalty%gauss(i)%position(j)
                    ENDDO
                    write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') penalty%gauss(i)%high
                    write(g_io%STRUCTINPUT,ADVANCE='YES',FMT='(X,F9.5)') penalty%gauss(i)%width
                  ENDDO
                ENDIF
                CLOSE(g_io%STRUCTINPUT)
              ENDIF

              IF (DYN%ISIF==3) THEN
                LDUMMYATOMS=.FALSE.
                IF (LDUMMYATOMS) THEN
                  CALL CONSTRAINED_DOF(DYN_dummy,T_INFO_dummy,3*T_INFO_dummy%NIONS+9,LATT_CUR,ICOORDINATES,cDOF)
                ELSE
                  CALL CONSTRAINED_DOF(DYN,T_INFO,3*T_INFO%NIONS+9,LATT_CUR,ICOORDINATES,cDOF)
                ENDIF                
              ELSE
                IF (LDUMMYATOMS) THEN
                  CALL CONSTRAINED_DOF(DYN_dummy,T_INFO_dummy,3*T_INFO_dummy%NIONS,LATT_CUR,ICOORDINATES,cDOF)
                ELSE
                  CALL CONSTRAINED_DOF(DYN,T_INFO,3*T_INFO%NIONS,LATT_CUR,ICOORDINATES,cDOF)
                ENDIF         
              ENDIF
            
              NDEGREES_OF_FREEDOM=NDEGREES_OF_FREEDOM_-cDOF
          
              IF (IO%IU6>0) THEN
                write(g_io%REPORT,FMT='(/,3X,A35,X,I5)')    'original number of atomic DOF:     ', NDEGREES_OF_FREEDOM_
                write(g_io%REPORT,FMT='(3X,A35,X,I5)')    'number of independent constraints: ', cDOF
                write(g_io%REPORT,FMT='(3X,A35,X,I5)')    'number of active DOF:              ', NDEGREES_OF_FREEDOM
              ENDIF
            END IF
            !c write summaray of flags in vasprun.xml
            !CALL DYN_XML_WRITE(MDALGO,ANDERSEN_PROB,SCALING,NSUBSYS,TSUBSYS,PSUBSYS)
          ELSE 
            !c Georg's fix
            IF (MDALGO>0) THEN
             ! copy current positions considered in VASP to POSIOC
                DYN%POSIOC=DYN%POSION
            ENDIF

          !CALL XML_TAG("stepmd")
          !CALL XML_INCAR('stepnum','I',counter,RDUM,CDUM,LDUM,CHARAC,1)

          ENDIF


          IF (MDALGO>0) THEN
            IF (IO%IU6>0) THEN
              write(g_io%REPORT,FMT='(/,A40)') '========================================'
              write(g_io%REPORT,FMT='(A20,X,I7)') 'MD step No.',counter
              write(g_io%REPORT,FMT='(A40)') '========================================'
            ENDIF

            IF (DYN%INIT==-1) THEN
              LSCALE=.TRUE.
              !IF (MDALGO==3 .OR. MDALGO==31 ) LCMASS=.FALSE.
              
              IF (LDUMMYATOMS) THEN
                CALL init_velocities(DYN_dummy%VEL,T_INFO_dummy,DYN_dummy,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
              ELSE
                CALL init_velocities(DYN%VEL,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)  
              ENDIF
              
               IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A40)') "Atomic velocities initialized by STEP_tb"
              DYN%INIT=0
            ENDIF
            
            !c define the reference coordinates needed
            !c to avoid problems with definition of internal parameters
            IF (counter==1) THEN
              IF (LDUMMYATOMS) THEN
                ALLOCATE(REF_C(3,T_INFO_dummy%NIONS))
              ELSE
                ALLOCATE(REF_C(3,T_INFO%NIONS))
              ENDIF
              REF_C=0._q
            ELSE
              IF (LDUMMYATOMS) THEN
                CALL minimize_difference(DYN_dummy%POSIOC,REF_C,T_INFO_dummy%NIONS)
              ELSE
                CALL minimize_difference(DYN%POSIOC,REF_C,T_INFO%NIONS)
              ENDIF
            ENDIF

            !c update position of minimum for harmonic spring
            IF (iconst8>0 .AND. Lhspring_V0) THEN
              hspring_R0(:)=hspring_R0(:)+DYN%POTIM*hspring_V0(:)
              IF (IO%IU6>0) THEN
                write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '          SPRING_R0 = '
                DO i=1,iconst8
                  write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') HSPRING_R0(i)
                ENDDO
              ENDIF
            ENDIF

            IF (LDUMMYATOMS) THEN
              !DYN_dummy%D2=0._q
              !DYN_dummy%D2(:,1:T_INFO%NIONS)=DYN%D2(:,1:T_INFO%NIONS)
              DYN_dummy%D2C=0._q
              DYN_dummy%D2C(:,1:T_INFO%NIONS)=DYN%D2C(:,1:T_INFO%NIONS)
              REF_C=DYN_dummy%POSIOC        
              CALL minimize_difference(DYN_dummy%POSION,DYN_dummy%POSIOC,T_INFO_dummy%NIONS)

              !c update internal coordinates
              CALL DEAL_XYZ(T_INFO_dummy,DYN_dummy%POSIOC,REF_C,LATT_CUR%A,ICOORDINATES)

              !c monitor coordinates with STATUS==7, stop if we get out of 
              !c limits
              IF (iconst7>0) CALL tis_stop(DYN_dummy,T_INFO_dummy,INFO,LATT_CUR,IO,g_io,ICOORDINATES,iconst7)
              
              !c add spring forces due to dummy atoms
              IF (iconst9>0) THEN
                IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Spring_coord'
                DO i=1,ICOORDINATES%NUMINTERNALS
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==9) THEN
                    IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,X,F15.5)') &
                    &   'dc>',ICOORDINATES%COORDSTRUCT(i)%TAG,ICOORDINATES%COORDSTRUCT(i)%VALUE
                  ENDIF
                ENDDO
                !c write down all atomic coordinates
                IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Positions'
                DO i=1,T_INFO_dummy%NIONS
                  IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A11,X,F20.16,X,F20.16,X,F20.16)') &
                  &   'frac_coord>',DYN_dummy%POSIOC(1,i),DYN_dummy%POSIOC(2,i),DYN_dummy%POSIOC(3,i)
                ENDDO
                
!                 !c write down all atomic coordinates
!                 IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Veocities'
!                 DO i=1,T_INFO_dummy%NIONS
!                   IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A9,X,F20.16,X,F20.16,X,F20.16)') &
!                   &   'frac_vel>',DYN_dummy%VEL(1,i),DYN_dummy%VEL(2,i),DYN_dummy%VEL(3,i)
!                 ENDDO
                
                CALL dummy_spring_force(iconst9,T_INFO_dummy,LATT_CUR,DYN_dummy,DUMMY_R0,DUMMY_K,&
                & ICOORDINATES,REF_C,spring_accel,ESPRING)             
                DYN_dummy%D2C=DYN_dummy%D2C-spring_accel             
              ENDIF           
            ELSE
              REF_C=DYN%POSIOC        
              CALL minimize_difference(DYN%POSION,DYN%POSIOC,T_INFO%NIONS)

              !c update internal coordinates
              CALL DEAL_XYZ(T_INFO,DYN%POSIOC,REF_C,LATT_CUR%A,ICOORDINATES)

              !c monitor coordinates with STATUS==7, stop if we get out of 
              !c limits
              IF (iconst7>0) CALL tis_stop(DYN,T_INFO,INFO,LATT_CUR,IO,g_io,ICOORDINATES,iconst7)

              !c monitor values of coordinates affected by harmonic bias potential 
              IF ((iconst8>0) .AND. (IO%IU6>0)) THEN
                write(g_io%REPORT,FMT='(/,2X,A12)') '>Spring_coord'
                DO i=1,ICOORDINATES%NUMINTERNALS
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==8) THEN
                    write(g_io%REPORT,FMT='(3X,A3,X,A2,X,F15.5)') &
                    &  'sc>',ICOORDINATES%COORDSTRUCT(i)%TAG,ICOORDINATES%COORDSTRUCT(i)%VALUE
                  ENDIF
                ENDDO
              ENDIF
            
              !tb check beg
              !c add spring forces due to dummy atoms
              IF (iconst9>0) THEN
                IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Monit_coord'
                DO i=1,ICOORDINATES%NUMINTERNALS
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==9) THEN
                    IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,X,F15.5)') &
                    &   'dc>',ICOORDINATES%COORDSTRUCT(i)%TAG,ICOORDINATES%COORDSTRUCT(i)%VALUE
                  ENDIF
                ENDDO              
                CALL dummy_spring_force(iconst9,T_INFO,LATT_CUR,DYN,DUMMY_R0,DUMMY_K,&
                & ICOORDINATES,REF_C,spring_accel,ESPRING)
                DYN%D2C=DYN%D2C-spring_accel
              ENDIF               
              !tb check end
            ENDIF 
          ENDIF

          !c rescale ionic velocities
          IF (DYN%ISIF==2 .AND. (MDALGO>0 .AND. MDALGO/=13) .AND. SCALING >0 ) THEN
            IF (MOD(counter,SCALING)==0) THEN
              CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
              TEMPER=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM
              DYN%VEL=DYN%VEL*(DYN%TEMP/TEMPER)**0.5

              IF (IO%IU6>0) THEN
                  write(g_io%REPORT,FMT='(/,2X,A29)') 'Velocities have been rescaled'
              ENDIF
            ENDIF
          ENDIF


          !c take care of the cetenter-of-mass motion
          IF (.NOT. T_INFO%LSDYN) THEN
            IF (LDUMMYATOMS) THEN
              !c remove drift from velocities
              IF (LCMASS) CALL SYMVEL_WARNING(T_INFO_dummy%NIONS,T_INFO_dummy%NTYP,T_INFO_dummy%ITYP,T_INFO_dummy%POMASS, &
              &            DYN_dummy%VEL,IO%IU6,IO%IU0)
               
              CALL SYMVEL(T_INFO_dummy%NIONS,T_INFO_dummy%NTYP,T_INFO_dummy%ITYP,T_INFO_dummy%POMASS, &
              &            DYN_dummy%POSION,DYN_dummy%D2C,LATT_CUR%A,LATT_CUR%B)

            ELSE
              !c remove drift from velocities
              IF (LCMASS) CALL SYMVEL_WARNING(T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS, &
              &            DYN%VEL,IO%IU6,IO%IU0)
              !c remove drift from forces
              CALL SYMVEL(T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS, &
              &            DYN%POSION,DYN%D2C,LATT_CUR%A,LATT_CUR%B)
            ENDIF
          ELSE
            IF (LDUMMYATOMS) THEN
              !c remove velocities for frozen directions
              CALL SET_SELECTED_VEL_ZERO(T_INFO_dummy, DYN_dummy%VEL,LATT_CUR)
            ELSE
              !c remove velocities for frozen directions
              CALL SET_SELECTED_VEL_ZERO(T_INFO, DYN%VEL,LATT_CUR)
            ENDIF
          ENDIF


          !c MD with Nose-Hoover thermostat
          IF (MDALGO==2) THEN
            IF (LDUMMYATOMS) THEN
              CALL STEP_HOOVER(DYN_dummy,ICOORDINATES,T_INFO_dummy,INFO,LATT_CUR,EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
              &              IO,EPOT,g_io,iconst0,iconst2,TEIN)
              IF (IO%IU6>0) CALL NOSE_OUT(g_io,DYN_dummy)
            ELSE
              CALL STEP_HOOVER(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR,EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
              &              IO,EPOT,g_io,iconst0,iconst2,TEIN)
              IF (IO%IU6>0) CALL NOSE_OUT(g_io,DYN)
            ENDIF
          !c Langevin dynamics, velocity verlet (to be replaced by leap-frog)
          ELSE IF (MDALGO==3) THEN
            IF (DYN%ISIF==3) THEN
              CALL STEP_LANGEVIN_ISIF3(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EKIN_Lat,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
              &      EPOT,GAMMA,GAMMA_L,g_io,SIF,iconst0,iconst2,0, &
              &      iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
            ELSE 
              CALL STEP_LANGEVIN(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR,EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                &              NDEGREES_OF_FREEDOM_,IO,EPOT,GAMMA,g_io,iconst0,iconst2,0,0,&
                &      iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
            ENDIF
         
          !c MD with Lowe thermostat
          ELSE IF (MDALGO==4) THEN           
            CALL STEP_LOWE(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR,EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                &              NDEGREES_OF_FREEDOM_,IO,EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)


          !c metaDynamics with fictituous variables, Andersen thermostat 
          ELSE IF (MDALGO==10) THEN
            CALL HILLS_METHOD(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,&
              &      g_io,WDES,iconst0,iconst2,TEIN)

          !c metaDynamics with fictituous variables, Nose-Hoover thermostat 
          ELSE IF (MDALGO==20) THEN
            CALL HILLS_METHOD_nose(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,& 
              &      g_io,WDES,iconst0,iconst2,TEIN)
            IF (IO%IU6>0) CALL NOSE_OUT(g_io,DYN)

          !c direct metaDynamics, Andersen thermostat 
          ELSE IF (MDALGO==11) THEN
            CALL HILLS_METHOD_DIRECT(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,&
              &      g_io,WDES,iconst0,iconst2,iconst5,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)

          !c direct metaDynamics, Nose-Hoover thermostat
          ELSE IF (MDALGO==21) THEN
            CALL HILLS_METHOD_DIRECT_nose(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,& 
              &      g_io,WDES,iconst0,iconst2,iconst5,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
            IF (IO%IU6>0) CALL NOSE_OUT(g_io,DYN)

          ELSE IF (MDALGO==31) THEN
            IF (DYN%ISIF==3) THEN
              CALL STEP_LANGEVIN_ISIF3(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
              &      EKIN,EKIN_Lat,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
              &      EPOT,GAMMA,GAMMA_L,g_io,SIF,iconst0,iconst2,iconst5, &
              &      iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
            ELSE 
              CALL STEP_LANGEVIN(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR,EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                &              NDEGREES_OF_FREEDOM_,IO,EPOT,GAMMA,g_io,iconst0,iconst2,iconst5,0,&
                &              iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
            ENDIF


          !c dynamics with multiple Andersen thermostats
          ELSE IF (MDALGO==13) THEN
            CALL STEP_NANDERSEN(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR,EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
              &              NDEGREES_OF_FREEDOM_,IO,EPOT,g_io,NSUBSYS,TSUBSYS,PSUBSYS,iconst0,iconst2,TEIN)

          !c MD with Andersen thermostat
          ELSE IF (MDALGO==1) THEN
            !c experimental: Rahman-Parrinello ensemble
            !!!IF (DYN%ISIF==3) THEN
            !!!  CALL STEP_ANDERSEN_ISIF3(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
            !!!     &      EKIN,EKIN_Lat,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,EPOT,&
            !!!     &      ANDERSEN_PROB,g_io,SIF,iconst0,iconst2,TEIN)
            !!!!c NVT ensemble
            !!!ELSE
              IF (LDUMMYATOMS) THEN
                CALL STEP_ANDERSEN(DYN_dummy,ICOORDINATES,T_INFO_dummy,INFO,LATT_CUR,EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                &              NDEGREES_OF_FREEDOM_,IO,EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
              ELSE
                !CALL STEP_ANDERSEN(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR,EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                !&              NDEGREES_OF_FREEDOM_,IO,EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
                CALL STEP_ANDERSEN_STOCH(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR,EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,&
                &              NDEGREES_OF_FREEDOM_,IO,EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
              ENDIF
            !!!ENDIF
          ELSE
             !IF ( DYN%ISIF==3 ) then
             IF ( DYN%ISIF==3 .OR. DYN%ISIF==8 .OR. DYN%ISIF==9 .OR. DYN%ISIF==10 ) then 
               CALL step_NPT( DYN%INIT, DYN%ISIF, NDEGREES_OF_FREEDOM+3, T_INFO%NIONS, T_INFO%NITYP,  &
               &       T_INFO%NTYP, LATT_CUR,DYN%PSTRESS, &
               &       EPOT, &
               &       T_INFO%POMASS, DYN%POSION, &
               &       DYN%POSIOC, DYN%POTIM, DYN%SMASS, DYN%SNOSE, DYN%TEMP, TIFOR,SIF, DYN%VEL, &
               &       DYN%D2C, DYN%D2, IO%IU6,IO%IU5,IO%IU0)
               CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                        & DYN%POTIM,LATT_CUR%A,DYN%D2)             
               TEIN = 2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM+3)
             ELSE
              !c standard MD module
              CALL STEP(DYN%INIT,ISCALE,T_INFO%NIONS,LATT_CUR%A,LATT_CUR%ANORM,DYN%D2C,DYN%SMASS,DYN%POSION,DYN%POSIOC, &
               DYN%POTIM,T_INFO%POMASS,T_INFO%NTYP,T_INFO%ITYP,DYN%TEMP,DYN%VEL,DYN%D2,DYN%D3,DYN%SNOSE, &
               EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM, IO%IU6)

              TEIN = 2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM
             ENDIF
          ENDIF   
          
#ifdef MPI
          IF (LDUMMYATOMS) THEN
            CALL M_bcast_d(WDES%COMM, DYN_dummy%POSION , T_INFO_dummy%NIONS*3)
            CALL M_bcast_d(WDES%COMM, DYN_dummy%VEL , T_INFO_dummy%NIONS*3)
          ELSE
            CALL M_bcast_d(WDES%COMM, DYN%POSION , T_INFO%NIONS*3)
            CALL M_bcast_d(WDES%COMM, DYN%VEL , T_INFO%NIONS*3)
          ENDIF
          CALL M_bcast_d(WDES%COMM, LATT_CUR%A, 9)
          CALL M_bcast_d(WDES%COMM, LATT_CUR%Avel, 9) 
          CALL LATTIC(LATT_CUR)
#endif

          IF (MDALGO>0) THEN
            DYN%INIT=1
            IF (DYN%ISIF>=3) THEN
              IF (LATT_CUR%INITlatv==0) LATT_CUR%INITlatv=1
            ENDIF
            IF (LDUMMYATOMS) THEN
              DYN_dummy%INIT=1
              CALL put_in_box(T_INFO_dummy%NIONS,DYN_dummy%POSION)
              DYN_dummy%POSIOC=DYN_dummy%POSION
              DYN%POSION(:,:)=DYN_dummy%POSION(:,1:T_INFO%NIONS)
              !DYN%POSIOC=DYN%POSION
              DYN%VEL(:,:)=DYN_dummy%VEL(:,1:T_INFO%NIONS)
              DYN%SNOSE=DYN_dummy%SNOSE
            ELSE
              CALL put_in_box(T_INFO%NIONS,DYN%POSION)
              !c on output, POSIOC contains the previous step
              !c and POSION the current one
              !DYN%POSIOC=DYN%POSION
            ENDIF
            IF (IO%IU6>=0) THEN
              CALL WFORCE(g_io%REPORT)
            ENDIF
          ENDIF

          !c stepmd
          !CALL XML_CLOSE_TAG
        END SUBROUTINE STEP_tb

        SUBROUTINE DYN_XML_WRITE(MDALGO,ANDERSEN_PROB,SCALING,NSUBSYS,TSUBSYS,PSUBSYS)
          INTEGER IDUM,SCALING,NIONS,MDALGO
          LOGICAL ::LDUM
          REAL(q) :: RDUM,ANDERSEN_PROB
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          INTEGER :: NSUBSYS(3)
          REAL(q) :: TSUBSYS(3),PSUBSYS(3)

          CALL XML_TAG("parameters_embedded")
          CALL XML_TAG("separator","mdynamics")
          CALL XML_INCAR('MDALGO','I',MDALGO,RDUM,CDUM,LDUM,CHARAC,1)
          IF (MDALGO==1 .OR. MDALGO==10 .OR. MDALGO==11 .OR. MDALGO==4) THEN
            CALL XML_INCAR('ANDERSEN_PROB','F',IDUM,ANDERSEN_PROB,CDUM,LDUM,CHARAC,1)
         
                !write(g_io%REPORT,FMT='(3X,A22,X,I6)')    '            SCALING = ' , SCALING
          ELSE IF (MDALGO==13) THEN   
            CALL XML_INCAR_V('NSUBSYS','I',NSUBSYS,RDUM,CDUM,LDUM,CHARAC,3)
            CALL XML_INCAR_V('TSUBSYS','F',IDUM,TSUBSYS,CDUM,LDUM,CHARAC,3)
            CALL XML_INCAR_V('PSUBSYS','F',IDUM,PSUBSYS,CDUM,LDUM,CHARAC,3)
          ENDIF  
          CALL XML_CLOSE_TAG
          CALL XML_CLOSE_TAG
       
        END SUBROUTINE DYN_XML_WRITE

        SUBROUTINE DYN_READER(IO,MDALGO,ANDERSEN_PROB,SCALING,NSUBSYS,TSUBSYS,PSUBSYS,GAMMA,&
        &  GAMMA_L,NDUMMYATOMS,DUMMY_POSITIONS,DUMMY_MASS,NIONS,NTYP)
        !c read general input parameters for MD
          TYPE (in_struct) :: IO
          INTEGER IDUM,THERMOSTAT,SCALING,NIONS,NTYP,MDALGO
          INTEGER :: K,N,IERR,NSUB,i
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM,ANDERSEN_PROB
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          INTEGER :: NSUBSYS(3)
          REAL(q) :: TSUBSYS(3),PSUBSYS(3)
          REAL(q) :: GAMMA(NTYP) !c friction coeficient in ps^(-1)
          REAL(q) :: GAMMA_L !c friction coeficient in ps^(-1) for lattice parameters
          INTEGER :: NDUMMYATOMS
          REAL(q) :: DUMMY_POSITIONS(3),DUMMY_MASS

          LOPEN=.FALSE.
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
         
          !c THERMOSTAT - obsolete, replaced by MDALGO
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'THERMOSTAT','=','#',';','I', &
                THERMOSTAT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) THERMOSTAT=0
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            THERMOSTAT=1
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''THERMOSTAT'' from file INCAR'
          ENDIF
          IF (THERMOSTAT<0) THERMOSTAT=0

          !c MDALGO
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'MDALGO','=','#',';','I', &
                MDALGO,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) MDALGO=THERMOSTAT
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            MDALGO=THERMOSTAT
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''MDALGO'' from file INCAR'
          ENDIF
          IF (MDALGO<0) MDALGO=0


          !c multiple thermostat stuff
          IF (MDALGO==13) THEN
            !c NSUBSYS - subsystems with different temperature
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'NSUBSYS','=','#',';','I', &
                  NSUBSYS,RDUM,CDUM,LDUM,CHARAC,N,3,IERR)
            DO i=1,NIONS
              IF (NSUBSYS(i) .GT. NIONS) THEN
                IF (IO%IU0>0) THEN
                  WRITE(IO%IU0,*) 'NSUBSYS(i) must not be greater than the number of atoms defined in POSCAR'
                ENDIF
                STOP
              END IF
            END DO
            IF (N==2 .AND. NSUBSYS(2)==NIONS) N=1
            IF (N==3 .AND. NSUBSYS(3)==NIONS) N=2
            IF ((N==1 .OR. N==2) .AND. (NSUBSYS(1)<NSUBSYS(2)) .AND. (NSUBSYS(2) .LE. NSUBSYS(3))) THEN
              NSUB=N+1
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'PSUBSYS','=','#',';','F', &
                  IDUM,PSUBSYS,CDUM,LDUM,CHARAC,N,3,IERR)
              IF (N == NSUB) THEN
                CALL RDATAB(LOPEN,INCAR,IO%IU5,'TSUBSYS','=','#',';','F', &
                  IDUM,TSUBSYS,CDUM,LDUM,CHARAC,N,3,IERR)
                IF (N /= NSUB) THEN
                  IF (IO%IU0>0) THEN
                    WRITE(IO%IU0,*) 'Error reading item ''TSUBSYS'' from file INCAR'
                  ENDIF
                  STOP
                ENDIF
                DO i=1,NSUB
                  IF (TSUBSYS(i) .LE. 0) THEN
                    IF (IO%IU0>0) THEN
                      WRITE(IO%IU0,*) 'zero in item ''TSUBSYS'' from file INCAR, all TSUBSYS(i) must be greater than 0'
                    ENDIF
                  STOP
                  END IF
                ENDDO
              ELSE
                IF (IO%IU0>0) THEN
                  WRITE(IO%IU0,*) 'Error reading item ''PSUBSYS'' from file INCAR'
                ENDIF
                STOP
              END IF
            ELSE
              IF (IO%IU0>0) THEN
                WRITE(IO%IU0,*) 'Error reading item ''NSUBSYS'' from file INCAR, two or three sub-systems must be defined'
              ENDIF
              STOP
            END IF           
          END IF

          !c ANDERSEN_PROB - probability of collision
          IF (MDALGO==1 .OR. MDALGO==10 .OR. MDALGO==11 .OR. MDALGO==13 .OR. MDALGO==4) THEN
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'ANDERSEN_PROB','=','#',';','F', &
                IDUM,ANDERSEN_PROB,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) ANDERSEN_PROB=0._q 
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
               ANDERSEN_PROB=0._q  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''ANDERSEN_PROB'' from file INCAR'
            ENDIF
            IF (ANDERSEN_PROB <0._q) ANDERSEN_PROB=0._q
          ENDIF

          IF (MDALGO==3 .OR. MDALGO==31) THEN
            !c friction coeficients for atomic positions
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'LANGEVIN_GAMMA','=','#',';','F', &
                IDUM,GAMMA,CDUM,LDUM,CHARAC,N,NTYP,IERR)
            IF (IERR==3) GAMMA=0._q 
            !c terminate if user didn't provide GAMMA for each atomic type
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<NTYP))) THEN
               GAMMA=0._q  
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LANGEVIN_GAMMA'' from file INCAR'
              STOP
            ENDIF
            !c terminate if negative value of GAMMA is defined 
            DO i=1,NTYP
              IF (GAMMA(i) <0._q) THEN
                IF (IO%IU0>=0) &
                write(IO%IU0,*) 'Error reading item ''LANGEVIN_GAMMA'' from file INCAR: all LANGEVIN_GAMMA must be positive numbers'
                STOP
              ENDIF 
            ENDDO
           
            !c friction coeficient for lattice parameters
            GAMMA_L=0._q
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'LANGEVIN_GAMMA_L','=','#',';','F', &
                IDUM,GAMMA_L,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) GAMMA_L=0._q 
            IF (GAMMA_L <0._q) THEN
                IF (IO%IU0>=0) &
                write(IO%IU0,*) 'Error reading item ''LANGEVIN_GAMMA_L'' from file INCAR: LANGEVIN_GAMMAL must be a positive number'
                STOP
            ENDIF 
 
          ENDIF


          !c SCALING - in how many steps should the velocities be rescaled?
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SCALING','=','#',';','I', &
                SCALING,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) SCALING=0
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            SCALING=0
          !  IF (IO%IU0>=0) &
          !     WRITE(IO%IU0,*)'Error reading item ''SCALING'' from file INCAR'
          ENDIF
          IF (SCALING<0) SCALING=0
          
          
          NDUMMYATOMS=0
          !c dummy atoms presently available only for Andersen and Nose-Hoover thermostat
          IF (MDALGO==1 .OR. MDALGO==2) THEN
            NDUMMYATOMS=1
            !c DUMMY_POSITIONS - positions of dummy atoms
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'DUMMY_POSITIONS','=','#',';','F', &
                IDUM,DUMMY_POSITIONS,CDUM,LDUM,CHARAC,N,3,IERR)
            IF (IERR==3) THEN
              DUMMY_POSITIONS=0._q 
              NDUMMYATOMS=0
            ENDIF
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<3))) THEN 
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''DUMMY_POSITIONS'' from file INCAR'
                STOP
            ENDIF
                   
            DUMMY_MASS=1._q
            IF (NDUMMYATOMS>0) THEN     
              !c DUMMY_MASS - mass of dummy atoms
              CALL RDATAB(LOPEN,INCAR,IO%IU5,'DUMMY_MASS','=','#',';','F', &
                IDUM,DUMMY_MASS,CDUM,LDUM,CHARAC,N,1,IERR)
              IF (IERR==3) THEN
                DUMMY_MASS=1._q 
              ENDIF
              IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN 
              IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''DUMMY_MASS'' from file INCAR'
                STOP
              ENDIF
              IF (DUMMY_MASS .LE. 0._q) THEN
                IF (IO%IU0>=0) &
                write(IO%IU0,*) 'Error reading item ''DUMMY_MASS'' from file INCAR: DUMMY_MASS must be a positive number'
                STOP
              ENDIF 
            END IF 
          ENDIF
          
          

          CLOSE(IO%IU5)
        END SUBROUTINE DYN_READER

        SUBROUTINE READER_ICONST0(IO,g_io,iconst0,BMTOL,BMTOLSOFT,BMSCA,BMITER,LBLUEOUT,EQUI_REGIME,INCREM)
        !c read input parameters specific for constrained dynamics
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          INTEGER IDUM, K,N,IERR,i,ii,iconst0
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q) :: BMTOL,BMTOLSOFT,BMSCA,INCREM(iconst0)
          INTEGER :: BMITER,EQUI_REGIME
          LOGICAL :: LBLUEOUT

          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
            
          !c tolerance for the shake algorithm
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SHAKETOL','=','#',';','F', &
                IDUM,BMTOL,CDUM,LDUM,CHARAC,N,ii,IERR)
          IF (IERR==3) BMTOL=1e-5
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            BMTOL=1e-5
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''SHAKETOL'' from file INCAR'
          ENDIF
          IF (BMTOL<0.) BMTOL=1e-5

          !c tolerance for the shake algorithm
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SHAKETOLSOFT','=','#',';','F', &
                IDUM,BMTOLSOFT,CDUM,LDUM,CHARAC,N,ii,IERR)
          IF (IERR==3) BMTOLSOFT=BMTOL
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            BMTOLSOFT=BMTOL
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''SHAKETOLSOFT'' from file INCAR'
          ENDIF
          IF (BMTOLSOFT<0.) BMTOLSOFT=BMTOL


          !c scaling of step taken in one iteration in the SHAKE algorithm
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SHAKESCA','=','#',';','F', &
                IDUM,BMSCA,CDUM,LDUM,CHARAC,N,ii,IERR)
          IF (IERR==3) BMSCA=2._q
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            BMSCA=2._q
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''SHAKESCA'' from file INCAR'
          ENDIF
          IF (BMSCA<0.) BMSCA=2._q

          !c maximal number of iterations in the SHAKE algorithm
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SHAKEMAXITER','=','#',';','I', &
                BMITER,RDUM,CDUM,LDUM,CHARAC,N,ii,IERR)
          IF (IERR==3) BMITER=1000
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            BMITER=1000
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''SHAKEMAXITER'' from file INCAR'
          ENDIF
          IF (BMITER<10) BMITER=1000

          !c write down output for blue-moon?
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'LBLUEOUT','=','#',';','L', &
              IDUM,RDUM,CDUM,LBLUEOUT,CHARAC,N,1,IERR)
          IF (IERR==3) LBLUEOUT=.FALSE.
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            LBLUEOUT=.FALSE.
            IF (IO%IU0>=0) &
             WRITE(IO%IU0,*)'Error reading item ''LBLUEOUT'' from file INCAR'
          ENDIF

          !c EQUI_REGIME - length of the equilibration preriod (dont constrain
          !c anything
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'EQUI_REGIME','=','#',';','I', &
              EQUI_REGIME,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) EQUI_REGIME=0
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            EQUI_REGIME=0
            !IF (IO%IU0>=0) &
            !   WRITE(IO%IU0,*)'Error reading item ''EQUI_REGIME'' from file INCAR'
          ENDIF
          IF (EQUI_REGIME<0) EQUI_REGIME=0

          !c INCREM
          IF (iconst0>0) THEN 
            INCREM=0._q
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'INCREM','=','#',';','F', &
                IDUM,INCREM,CDUM,LDUM,CHARAC,N,iconst0,IERR)
            IF (IERR==3) INCREM=0._q
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst0))) THEN
              INCREM=0._q
              !IF (IO%IU0>=0) &
              !  & WRITE(IO%IU0,*)'Error reading item ''INCREM'' from file INCAR'
            ENDIF
          END IF

          IF (IO%IU6>0) THEN
            write(g_io%REPORT,FMT='(/,2X,A31)')    '>parameters for SHAKE algorithm'
            !write(g_io%REPORT,FMT='(3X,A17,X,I5)')    '   EQUI_REGIME = ',EQUI_REGIME
            write(g_io%REPORT,FMT='(3X,A15,X,I8)')         'SHAKEMAXITER = ',BMITER
            write(g_io%REPORT,FMT='(3X,A15,X,E13.6E2)')    '    SHAKETOL = ',BMTOL
            write(g_io%REPORT,FMT='(3X,A15,X,F12.8)')      '    SHAKESCA = ',BMSCA
          END IF 
          !write(g_io%REPORT,FMT='(3X,A17,X,F12.8)') '        INCREM = ',INCREM
          CLOSE(IO%IU5)
        END SUBROUTINE READER_ICONST0

        SUBROUTINE READER_ICONST7(IO,g_io,ICOORDINATES,iconst7,LMIN,LMAX,valueA,valueB)
        !c read input parameters specific coordinates with STATUS=7
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER IDUM, K,N,IERR,i,ii,iconst7
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          LOGICAL :: LMIN,LMAX
          REAL(q) :: valueA(iconst7),valueB(iconst7)

          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'VALUE_MIN','=','#',';','F', &
            IDUM,valueA,CDUM,LDUM,CHARAC,N,iconst7,IERR)
          !write(*,*) 'xx1',valueA(:),IERR,N,iconst7
          IF (IERR/=0) THEN
            LMIN=.FALSE.
            !IF (IO%IU0>=0) &
            !  WRITE(IO%IU0,*)'Error reading item ''VALUE_MIN'' from file INCAR'
          ELSE IF ((IERR==0).AND.(N .NE. iconst7)) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'Error reading item ''VALUE_MIN'' from file INCAR'
            STOP
          ENDIF

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'VALUE_MAX','=','#',';','F', &
            IDUM,valueB,CDUM,LDUM,CHARAC,N,iconst7,IERR)
          !write(*,*) 'xx2',valueB(:),IERR,N,iconst7
          IF (IERR/=0) THEN
            LMAX=.FALSE.
            !IF (IO%IU0>=0) &
            !  WRITE(IO%IU0,*)'Error reading item ''VALUE_MAX'' from file INCAR'
          ELSE IF ((IERR==0).AND.(N .NE. iconst7)) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'Error reading item ''VALUE_MAX'' from file INCAR'
            STOP
          ENDIF
              
          CLOSE(IO%IU5)

          DO i=1,iconst7
            IF (valueB(i)<valueA(i)) THEN
              LMIN=.FALSE.
              LMAX=.FALSE.
            END IF
          ENDDO
           
          IF (LMIN) THEN
            IF (IO%IU6>0) THEN
              write(g_io%REPORT,FMT='(/,2X,A42)')    '>limiting values for monitored coordinates'
              write(g_io%REPORT,FMT='(3X,A12)',ADVANCE='NO')  'VALUE_MIN = '
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
                  write(g_io%REPORT,FMT='(X,F9.5)',ADVANCE='NO') valueA(i)
                ENDIF
              ENDDO
              write(g_io%REPORT,FMT='(X)',ADVANCE='YES')
            ENDIF
          ENDIF

          IF (LMAX) THEN
            IF (IO%IU6>0) THEN
              write(g_io%REPORT,FMT='(3X,A12)',ADVANCE='NO')  'VALUE_MAX = '
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
                  write(g_io%REPORT,FMT='(X,F9.5)',ADVANCE='NO') valueB(i)
                ENDIF
              ENDDO
              write(g_io%REPORT,FMT='(X)',ADVANCE='YES')
            ENDIF 
          ENDIF 
        END SUBROUTINE READER_ICONST7

        SUBROUTINE ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,ESPRING,EPS,ES,counter)
        !c write some output from MD
          TYPE(gadget_io) :: g_io
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          INTEGER :: counter
          REAL(q) :: ETOTAL,EKIN,EPOT,EPS,ES,ECONST,ESPRING

          ETOTAL=EKIN+EPOT+ECONST+EPS+ES+ESPRING
            

            write(g_io%REPORT,FMT='(/,2X,A9)') '>Energies'
            write(g_io%REPORT,FMT='(7X,7(X,A17))')      'E_tot','E_pot','E_kin','E_spring','E_const','EPS','ES'
            write(g_io%REPORT,FMT='(3X,A4,6(X,E17.8E2))') 'e_b>', ETOTAL,EPOT,EKIN,ESPRING,ECONST,EPS,ES

            !write(g_io%REPORT,FMT='(/,2X,A11)') '>Positions:'
            !DO i=1,T_INFO%NIONS
            !  write(g_io%REPORT,FMT='(3X,A4, I5,3(X,F9.5))') 'x_b>', i,DYN%POSION(:,i)
            !ENDDO

            !write(g_io%REPORT,FMT='(/,2X,A12)') '>Velocities:'
            !DO i=1,T_INFO%NIONS
            !  write(g_io%REPORT,FMT='(3X,A4, I5,3(X,F9.5))') 'v_b>', i,DYN%VEL(:,i)
            !ENDDO

            !write(g_io%REPORT,FMT='(/,2X,A14)') '>Acceleration:'
            !DO i=1,T_INFO%NIONS
            !  write(g_io%REPORT,FMT='(3X,A4, I5,3(X,F9.5))') 'a_b>', i,DYN%D2C(:,i)
            !ENDDO

            !write(g_io%REPORT,FMT='(/,2X,A20)') '>Acceleration_const:'
            !DO i=1,T_INFO%NIONS
            !  write(g_io%REPORT,FMT='(3X,A4, I5,3(X,F9.5))') 'a_c>', i,C_FORCE(:,i)
            !ENDDO

        END SUBROUTINE ENERGY_OUT

        SUBROUTINE NOSE_OUT(g_io,DYN)
        !c write some output from MD
          TYPE(gadget_io) :: g_io
          TYPE(dynamics) :: DYN

            write(g_io%REPORT,FMT='(/,2X,A12)') '>Thermostat:'
            write(g_io%REPORT,FMT='(7X,5(X,A9))') &
                     "s(n+1)","sdot(n+1)","s(n)","sdot(n)","Qs"
            write(g_io%REPORT,FMT='(3X,A4,5(X,F9.5))') 's_b>',&
                     DYN%SNOSE(1),DYN%SNOSE(2),DYN%SNOSE(4),DYN%SNOSE(3),DYN%SMASS
        END SUBROUTINE NOSE_OUT

        SUBROUTINE TEMPERATURE_OUT(g_io,DYN,T_new)
        !c write some output from MD
          TYPE(gadget_io) :: g_io
          TYPE(dynamics) :: DYN
           REAL(q) :: T_new

          write(g_io%REPORT,FMT='(/,2X,A12)') '>Temperature'
          write(g_io%REPORT,FMT='(9X,2(X,A10))') &
                & "T_sim","T_inst"    
          write(g_io%REPORT,FMT='(3X,A6,2(X,F10.3))') &
                & 'tmprt>', DYN%TEMP,T_new    
        END SUBROUTINE TEMPERATURE_OUT


        SUBROUTINE CONSTRAINED_DOF(DYN,T_INFO,bdim,LATT_CUR,ICOORDINATES,cDOF)
         !c identify number of constrained (non-redundant) degrees of freedom
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: i,bdim,cDOF
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,bdim)
          REAL(q) :: norm

          cDOF=0
                  
          IF (ICOORDINATES%NUMINTERNALS>=1) THEN          
            BMAT=0._q
            CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.TRUE.)
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS/=0) THEN
                BMAT(i,:)=0._q
              ENDIF
            ENDDO
            IF (ICOORDINATES%NUMINTERNALS>1) CALL ORTHO_NORMALIZE(BMAT)
            DO i=1,ICOORDINATES%NUMINTERNALS
              norm=VECTORSIZE(bdim,BMAT(i,:))
              IF (norm>1e-5) THEN
                cDOF=cDOF+1 
              END IF
            ENDDO
          ENDIF 
        END SUBROUTINE CONSTRAINED_DOF

        SUBROUTINE STEP_ANDERSEN(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
          &      EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
        !c Andersen thermostat
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ECONST,TEIN,ESPRING
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          REAL(q) :: ANDERSEN_PROB
          INTEGER :: random_counter
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          LOGICAL :: LSCALE,LCMASS

          LSCALE=.FALSE.
          LCMASS=.FALSE.
          !c add this step to total count
          counter=counter+1

          AMASS=0._q

          VEL_tmp=0._q
          CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
          CMASS=0._q

          DYN%VEL=DYN%VEL+DYN%D2C
          DYN%VEL=DYN%VEL+DYN%D2C
          random_counter=0
          IF (DYN%INIT==1) THEN
            DO i=1,T_INFO%NIONS
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                DYN%VEL(1:3,i)=VEL_tmp(1:3,i)
              ENDIF
            ENDDO
          ENDIF

          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)
          DO i=1,T_INFO%NIONS
            DO j=1,3
              IF (T_INFO%LSFOR(j,i)) DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
          ENDDO
          !CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          C_FORCE=0._q
          DYN%POSION=DYN%POSIOC+DYN%VEL

          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          IF (IO%IU6>0) THEN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,ESPRING,0._q,0._q,counter)
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN)
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter            
          ENDIF


          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_ANDERSEN

        SUBROUTINE STEP_ANDERSEN_STOCH(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
          &      EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
        ! this version removes the center of mass drift from the final velocities
        ! in each step
        ! before randomizing the individual velocities
        ! a random center of mass drift is added 
        ! see below
        ! this routine can be used as an alternative to STEP_ANDERSEN
        ! the main advantage is that the routine is "stateless"

          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          REAL(q) :: ANDERSEN_PROB
          INTEGER :: random_counter
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          REAL(q) :: TOTAL_MASS
          INTEGER,SAVE :: Ltxyz(3)

          !c add this step to total count
          counter=counter+1

          !c determine if coordinates for some direction
          !c are frozen
          IF (counter==1) THEN
            Ltxyz(:)=1
            DO j=1,3
              DO i=1,T_INFO%NIONS
                IF (.NOT. T_INFO%LSFOR(j,i)) Ltxyz(j)=0
              ENDDO
            ENDDO

            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="X ") Ltxyz(1)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Y ") Ltxyz(2)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Z ") Ltxyz(3)=0
              ENDIF
            ENDDO
          ENDIF

          AMASS=0._q

          VEL_tmp=0._q

          ! Add a random velicity that is exactly
          ! distributed like the centre of mass velocity
          TOTAL_MASS = 0.0_q
          DO I = 1, T_INFO%NIONS
            TOTAL_MASS = TOTAL_MASS + T_INFO%POMASS(T_INFO%ITYP(I))
          END DO
          CMASS(:,1) = BOLTZMANN_VELOCITY(DYN, LATT_CUR, TOTAL_MASS)

          !c don't add translation for frozen direction
          DO i=1,3
            IF (Ltxyz(i)==0) CMASS(i,1)=0._q
          ENDDO

          DO i=1,T_INFO%NIONS
            DO j=1,3
              DYN%VEL(j,i)=DYN%VEL(j,i)+CMASS(j,1)
            ENDDO
          ENDDO

          CALL boltzmann_velocities(T_INFO,DYN,LATT_CUR,VEL_tmp)

          DYN%VEL=DYN%VEL+DYN%D2C
          DYN%VEL=DYN%VEL+DYN%D2C
          random_counter=0
          IF (DYN%INIT==1) THEN
            DO i=1,T_INFO%NIONS
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                DYN%VEL(1:3,i)=VEL_tmp(1:3,i)
              ENDIF
            ENDDO
          ENDIF

          CMASS=0._q
          !c GIVE_CMASS takes care of frozen directions (corresponding
          !c component set to zero)
          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          ! store the centre of mass drift
          ! so that we can add it again before swaping numbers
          DO i=1,T_INFO%NIONS
            DO j=1,3
              DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
         ENDDO
          !CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

         !c velocity for the step t-1/2dt
          VEL_tmp=DYN%VEL

          !c velocity for the step t+1/2dt
          !DYN%VEL=DYN%VEL+DYN%D2C
          !DYN%VEL=DYN%VEL+DYN%D2C
          
          DYN%POSION=DYN%POSIOC+DYN%VEL

          ECONST=0._q
          IF (iconst0>0 .OR. iconst2>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          !!c velocity for the step t
          !!VEL_tmp=(VEL_tmp+DYN%VEL)/2
          !!CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
          !!             & DYN%POTIM,LATT_CUR%A,VEL_tmp)

          !c due to compatibility reasons we still compute
          !c kinetic energy and T using velocities from the step t+1/2
          !c but in the future we should use VEL_tmp (velocity for step t) instead
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          IF (IO%IU6>0) THEN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN)
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter
          ENDIF


          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_ANDERSEN_STOCH


        SUBROUTINE STEP_LOWE(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
          &      EPOT,ANDERSEN_PROB,g_io,iconst0,iconst2,TEIN)
        !c Andersen thermostat
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          REAL(q) :: ANDERSEN_PROB
          INTEGER :: random_counter
          REAL(q) :: AMASS,C_FORCE_L(3,3)

          !c add this step to total count
          counter=counter+1

          AMASS=0._q

          !VEL_tmp=0._q
          !CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,.FALSE.,.TRUE.)
          !CMASS=0._q

          DYN%VEL=DYN%VEL+DYN%D2C
          DYN%VEL=DYN%VEL+DYN%D2C
          random_counter=0
          IF (DYN%INIT==1) THEN
            CALL LOWE_VELOCITY(DYN,T_INFO,LATT_CUR,IO,NDEGREES_OF_FREEDOM_+3,ANDERSEN_PROB,random_counter)
          ENDIF

!           CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)
!           IF (IO%IU6>0) write(*,*) "CMASS:",CMASS(1,1),CMASS(2,1),CMASS(3,1)
!           DO i=1,T_INFO%NIONS
!             DO j=1,3
!               IF (T_INFO%LSFOR(j,i)) DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
!             ENDDO
!           ENDDO
          !CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          C_FORCE=0._q
          DYN%POSION=DYN%POSIOC+DYN%VEL

          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          IF (IO%IU6>0) THEN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN)
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter            
          ENDIF


          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_LOWE

       SUBROUTINE LOWE_VELOCITY(DYN,T_INFO,LATT_CUR,IO,NDEGREES_OF_FREEDOM,ANDERSEN_PROB,random_counter)
          USE prec
          USE poscar
          USE base
          USE lattice
          USE mymath
          USE chain
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
!           REAL(q) :: V_tmp(3,T_INFO%NIONS)
          REAL(q) :: V_tmp(3),V_tmp2(3)
          REAL(q) :: dX(3),dXc(3),dV(3)
          REAL(q) :: M1,M2,Mtot
          REAL(q) :: r,rc,dummy,ANDERSEN_PROB
          INTEGER :: NDEGREES_OF_FREEDOM
          INTEGER :: i,j,random_counter,count
           REAL(q) :: RNULL,UL,UT,FACT,BMP
!           INTEGER :: NT


          !c this parameter should be user-controlled
          rc=7.0

          random_counter=0
          count=0

          !c generate velocities according to gauss distribution
          RNULL=0._q
          UL =1E-10_q
          UT =DYN%POTIM*1E-15_q
          FACT= (AMTOKG/EVTOJ)*(UL/UT)**2
!       
!           V_tmp=0._q
!           DO i=1,T_INFO%NIONS
!             NT=T_INFO%ITYP(i)
!             BMP=SQRT(2*DYN%TEMP*BOLKEV/(T_INFO%POMASS(NT)*FACT))
!             V_tmp(1,i)=boltzmann_distribution(RNULL,BMP) 
!             V_tmp(2,i)=boltzmann_distribution(RNULL,BMP)
!             V_tmp(3,i)=boltzmann_distribution(RNULL,BMP)
!           ENDDO
! 
!          CALL KARDIR(T_INFO%NIONS,V_tmp,LATT_CUR%B)


        !!  CALL init_velocities(V_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM,.FALSE.)
      
          DO i=1,T_INFO%NIONS
            M1=T_INFO%POMASS(T_INFO%ITYP(i))
            DO j=i+1,T_INFO%NIONS   
              M2=T_INFO%POMASS(T_INFO%ITYP(j))
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB/(T_INFO%NIONS-1)) THEN
                dX=DYN%POSION(:,j)-DYN%POSION(:,i)
                WHERE(dX>0.5_q) dX=dX-1._q
                WHERE(dX<-0.5_q) dX=dX+1._q
                dXc=MATMUL(dX,TRANSPOSE(LATT_CUR%A))
                r=(SUM(dXc*dXc))**0.5_q
                IF (r<=rc) THEN
                  count=count+1
                  random_counter=random_counter+1
                  dXc=dXc/r
                  
                  !Mtot=(T_INFO%POMASS(T_INFO%ITYP(i))+T_INFO%POMASS(T_INFO%ITYP(j)))/2
                  Mtot=M1*M2/(M1+M2)
                  BMP=SQRT(DYN%TEMP*BOLKEV/(Mtot*FACT))
                  V_tmp(:)=boltzmann_distribution(RNULL,BMP)*dXc
                  V_tmp2(:)=(DYN%VEL(:,j)-DYN%VEL(:,i))
                  V_tmp2=MATMUL(V_tmp2,TRANSPOSE(LATT_CUR%A))
                  V_tmp2=SUM(V_tmp2*dXc)*dXc
                  V_tmp=V_tmp-V_tmp2
                  V_tmp=MATMUL(V_tmp,(LATT_CUR%B))
                  !CALL KARDIR(1,V_tmp,LATT_CUR%B)
!                   M1=T_INFO%POMASS(T_INFO%ITYP(j))/(T_INFO%POMASS(T_INFO%ITYP(i))+T_INFO%POMASS(T_INFO%ITYP(j)))
!                   M2=T_INFO%POMASS(T_INFO%ITYP(i))/(T_INFO%POMASS(T_INFO%ITYP(i))+T_INFO%POMASS(T_INFO%ITYP(j)))
                  DYN%VEL(:,i)=DYN%VEL(:,i)-V_tmp*Mtot/M1
                  DYN%VEL(:,j)=DYN%VEL(:,j)+V_tmp*Mtot/M2                  

! 
! 
!                   dX=dX/(SUM(dX*dX))**0.5_q
!                   V_tmp-SUM((DYN%VEL(:,i)-DYN%VEL(:,j))*)
! 
! 
!                   M1=T_INFO%POMASS(T_INFO%ITYP(j))/(T_INFO%POMASS(T_INFO%ITYP(i))+T_INFO%POMASS(T_INFO%ITYP(j)))
!                   M2=T_INFO%POMASS(T_INFO%ITYP(i))/(T_INFO%POMASS(T_INFO%ITYP(i))+T_INFO%POMASS(T_INFO%ITYP(j)))
!                   dX=dX/(SUM(dX*dX))**0.5_q
!                   dV=((V_tmp(:,i)-V_tmp(:,j))-(DYN%VEL(:,i)-DYN%VEL(:,j)))/2.
!                   dV=SUM(dV*dX)*dX
!                   DYN%VEL(:,i)=DYN%VEL(:,i)+dV*M1
!                   DYN%VEL(:,j)=DYN%VEL(:,j)-dV*M2             
                ENDIF
              ENDIF            
            ENDDO
          ENDDO

          !c tset velocities for fixed atoms to zero 
          DO i=1,T_INFO%NIONS
            IF (.NOT. T_INFO%LSFOR(1,i)) DYN%VEL(1,i)=0._q 
            IF (.NOT. T_INFO%LSFOR(2,i)) DYN%VEL(2,i)=0._q
            IF (.NOT. T_INFO%LSFOR(3,i)) DYN%VEL(3,i)=0._q
          ENDDO

        END SUBROUTINE LOWE_VELOCITY



        SUBROUTINE STEP_LANGEVIN(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
          &      EPOT,GAMMA,g_io,iconst0,iconst2,iconst5,iconst6,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
        !c Langevin thermostat, velocity verlet algorithm!!!
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(penalty_data) :: penalty
          TYPE(hills_data) :: hills
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ECONST
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2,iconst5,iconst6,iconst8
          REAL(q) :: CMASS1(3,1),CMASS2(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: V_FORCE(3,T_INFO%NIONS),C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          INTEGER :: random_counter
          REAL(q) :: AMASS,TEIN
          REAL(q) :: C_FORCE_L(3,3),V_FORCE_L(3,3)
          REAL(q) :: GAMMA(T_INFO%NTYP) !c friction coeficient (in ps^(-1)) for each atomic type
          REAL(q) :: D2C_stoch(3,T_INFO%NIONS),D2C_momentum(3,T_INFO%NIONS)
          REAL(q) :: ekin1,ekin2,tein1,tein2
          REAL(q),PARAMETER :: tol=1e-6
          INTEGER,PARAMETER :: MAXITER=1000
          REAL(q) :: err2
          REAL(q) :: vel_(3,T_INFO%NIONS), vel_old(3,T_INFO%NIONS)
          REAL(q) :: hills_accel(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS),hspring_accel(3,T_INFO%NIONS)
          REAL(q) :: hills_accel_L(3,3),penalty_accel_L(3,3),hspring_accel_L(3,3)
          INTEGER :: NI, NT
          REAL(q) :: FACT,VTMP(3)
          REAL(q) :: maxForce
          LOGICAL :: LSTOP2
          REAL(q),SAVE :: EPOT_old
          REAL(q) :: hspring_R0(iconst8),hspring_K(iconst8),hspring_force(iconst8)
          
         
 72     FORMAT( ' POSITION    ',35X,'REST-FORCE (eV/Angst)'/ &
     &          ' ----------------------------------------------', &
     &          '-------------------------------------')
 76     FORMAT((3F13.5,3X,3F14.6))


          !c add this step to total count
          counter=counter+1

          !c make sure that energy convergence criterion is not fullfiled 
          !c in the first relaxation step (irelevant for MD)
          IF (counter==1) EPOT_old=EPOT+10*ABS(DYN%EDIFFG)

          V_FORCE_L=0._q
          C_FORCE_L=0._q
          V_FORCE=0._q
          C_FORCE=0._q
          hills_accel_L=0._q;penalty_accel_L=0._q;hspring_accel_L=0._q
          hills_accel=0._q;penalty_accel=0._q;hspring_accel=0._q

      !c metadynamics stuff 
          !c update history of fictioous particles and of collective variables
          IF (iconst5 >0) THEN
            j=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                j=j+1
                hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE 
              ENDIF
            ENDDO
          
            !c add new hill
            IF (counter .GE. hills%bin) THEN
              CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
            ENDIF 

            !c calculate the penalty potential
            CALL penalty_potential(penalty,hills,ICOORDINATES)
 
            !c compute contributions from bias potentials
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,penalty%force,ICOORDINATES,DYN%POSIOC,penalty_accel,penalty_accel_L,3*T_INFO%NIONS,0._q)
            !hills%force=hills%force+penalty%force
            !CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,hspring_force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF
          
          !c calculate forces due to harmonic springs
          !IF (iconst8>0) call harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
          IF (iconst8>0) THEN
            CALL harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
            CALL harmonic_spring_accel(T_INFO,LATT_CUR,DYN,hspring_force,ICOORDINATES,DYN%POSIOC,hspring_accel,hspring_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !c compute accel. due to friction term
          CALL friction_Forces_stoch(D2C_stoch,T_INFO,DYN,LATT_CUR,GAMMA)

          !c v(t)dt=v´(t)dt+f(t)/(2m)*dt^2
          DYN%VEL=DYN%VEL+DYN%D2C+D2C_stoch

          !IF (iconst5 .GT. 0) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel)
          IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0)) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)
          !c factor of 1/2 was missing - CHECK!!!
          !IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0)) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)/2

   
          vel_=DYN%VEL

          !c velocities for the time slice (t) depend on themselves
          !c iterative solution is needed
          i=0
          DO
            i=i+1
            IF (i .GT. MAXITER) THEN
              IF (IO%IU6>0) THEN
                write(*,*) "Error: convergence problem!"
              ENDIF
              STOP
            ENDIF 
            vel_old=DYN%VEL
  
            !c compute accel. due to friction term 
            CALL friction_Forces_momentum(D2C_momentum,T_INFO,DYN,GAMMA)
   
            DYN%VEL=vel_+D2C_momentum 

            IF (ICOORDINATES%NUMINTERNALS>0) THEN                            
               CALL Rattle(counter,T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,&
               &   AMASS,LATT_CUR%A,ECONST,iconst0,iconst2,1)
              !CALL Rattle_vel(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
              DYN%VEL=DYN%VEL+V_FORCE
            ENDIF
                            
            vel_old=ABS(vel_old-DYN%VEL)
            err2=MAXVAL(vel_old)
            IF (err2<tol) EXIT
          ENDDO

          !DYN%VEL=DYN%VEL+DYN%D2C+D2C_momentum+D2C_stoch
          CALL langevin_temperature(EKIN1,EKIN2,TEIN1,TEIN2,T_INFO,DYN,LATT_CUR%A,GAMMA)
          
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

!           IF (IO%IU6>0) THEN
!             CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,counter)
!             CALL TEMPERATURE_OUT(g_io,DYN,TEIN) 
!             IF (IO%NWRITE==3) THEN
!               write(g_io%REPORT,FMT='(9X,2(X,A10))') &
!                  & "T(Lang.)","T(nLang.)"    
!               write(g_io%REPORT,FMT='(3X,A6,2(X,F10.3))') &
!                  & 'tmpLg>',  TEIN1,TEIN2   
!             ENDIF      
!           ENDIF

          !c v'(t+dt)*dt=v(t)*dt+f(t)/(2m)*dt^2
          DYN%VEL=DYN%VEL+DYN%D2C+D2C_momentum+D2C_stoch

          !c metydynamics or biased MD
          !IF (iconst5 .GT. 0 ) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel)
          IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0) ) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)
          !c factor of 1/2 was missing - CHECK!!!
          !IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0)) DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)/2


!           IF (ICOORDINATES%NUMINTERNALS>0) THEN           
!            DYN%VEL=DYN%VEL+V_FORCE/2._q
!           ENDIF

          !c compute position of c. of mass for the slice t:
          CALL GIVE_CMASS(T_INFO,DYN%POSIOC,CMASS1)

          !c r(t+dt)=r(t)+v'(t+dt)*dt
          DYN%POSION=DYN%POSIOC+DYN%VEL

          
          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            !c and now RATTLE
            CALL Rattle(counter,T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,&
            &   AMASS,LATT_CUR%A,ECONST,iconst0,iconst2,0)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          !c if this MD subroutine is used for relaxation
          IF (DYN%TEBEG<0.1) THEN
            !c compute the total restoring force
            IF (iconst0>0) THEN
              C_FORCE=C_FORCE+V_FORCE
              V_FORCE=0._q

              V_FORCE=DYN%D2C+C_FORCE

              FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
              NI=1
              DO NT=1,T_INFO%NTYP
                DO NI=NI,T_INFO%NITYP(NT)+NI-1
                  V_FORCE(1,NI)=V_FORCE(1,NI)/FACT*2*T_INFO%POMASS(NT)
                  V_FORCE(2,NI)=V_FORCE(2,NI)/FACT*2*T_INFO%POMASS(NT)
                  V_FORCE(3,NI)=V_FORCE(3,NI)/FACT*2*T_INFO%POMASS(NT)
                ENDDO
              ENDDO

              CALL DIRKAR(T_INFO%NIONS,V_FORCE,LATT_CUR%A)

              !c write ionic forces after removing the restoring forces
              IF (IO%IU0>0) THEN
                WRITE(IO%IU6,72)
                DO J=1,T_INFO%NIONS
                  VTMP=T_INFO%POSION(1:3,J)
                  CALL  DIRKAR(1,VTMP,LATT_CUR%A)
                  WRITE(IO%IU6,76) VTMP,(V_FORCE(I,J),I=1,3)
                ENDDO
              ENDIF 
            ENDIF
            
            !c test the convergence criterion
            maxForce=0._q
            DO i=1,T_INFO%NIONS
              dummy=V_FORCE(1,i)*V_FORCE(1,i)+V_FORCE(2,i)*V_FORCE(2,i)+V_FORCE(3,i)*V_FORCE(3,i)
              IF (dummy>maxForce) &
                & maxForce=dummy
            END DO
            maxForce=maxForce**0.5
            IF (IO%IU0>0) THEN
              write(*,*) 'maxForce',maxForce,DYN%EDIFFG
            ENDIF 

            IF (DYN%EDIFFG<0.) THEN
              LSTOP2=.TRUE.

              IF (maxForce>ABS(DYN%EDIFFG)) LSTOP2=.FALSE.

              IF (LSTOP2) THEN
                INFO%LSTOP=.TRUE.
                !c take care of this POSION/POSIOC issue
                DYN%POSIOC=DYN%POSION
                RETURN
              END IF
            ELSE IF (counter>1) THEN
              IF (IO%IU0>0) THEN
              write(*,*) 'dEPOT',ABS(EPOT-EPOT_old),DYN%EDIFFG
            ENDIF
              INFO%LSTOP=(ABS(EPOT-EPOT_old)<DYN%EDIFFG)
            ENDIF
          ENDIF

          !!c compute position of c. of mass for the slice t+dt:
          !CALL GIVE_CMASS(T_INFO,DYN%POSION,CMASS2)

          !remove shift of c. of mass:
!           DO i=1,T_INFO%NIONS
!             DO j=1,3
!               IF (T_INFO%LSFOR(j,i)) DYN%POSION(j,i)=DYN%POSION(j,i)-(CMASS2(j)-CMASS1(j))
!             ENDDO
!           ENDDO
 
          IF (IO%IU6>0) THEN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN) 
            IF (IO%NWRITE==3) THEN
              write(g_io%REPORT,FMT='(9X,2(X,A10))') &
                 & "T(Lang.)","T(nLang.)"    
              write(g_io%REPORT,FMT='(3X,A6,2(X,F10.3))') &
                 & 'tmpLg>',  TEIN1,TEIN2   
            ENDIF   

            !c output for metadynamics
            IF (iconst5>0) THEN
              write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                  write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5)') &
                  & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE
                ENDIF
              ENDDO   
            ENDIF    
          ENDIF

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q

          EPOT_old=EPOT
        END SUBROUTINE STEP_LANGEVIN

        SUBROUTINE STEP_LANGEVIN_ISIF3(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EKIN_Lat,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,&
          &      EPOT,GAMMA,GAMMA_L,g_io,SIF,iconst0,iconst2,iconst5,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
        !c Langevin thermostat, velocity verlet algorithm!!!
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(penalty_data) :: penalty
          TYPE(hills_data) :: hills
          REAL(q) :: EKIN,EKIN_Lat,EPS,ES,DISMAX,TEMPER,EPOT,ECONST
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2,iconst5,iconst8
          REAL(q) :: CMASS(3,1)
          REAL(q) :: V_FORCE(3,T_INFO%NIONS),C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: GAMMA(T_INFO%NTYP) !c friction coeficient (in ps^(-1)) for each atomic type
          REAL(q) :: GAMMA_L !c friction coeficient (in ps^(-1)) for lattice components
          REAL(q) :: D2C_stoch(3,T_INFO%NIONS),D2C_momentum(3,T_INFO%NIONS)
          REAL(q) :: AC(3,3),Aaccel(3,3),Avel_tmp(3,3),Ginv(3,3),Gdot(3,3)
          REAL(q),SAVE :: Amass !,Avel(3,3)
          REAL(q) :: dumM(9),FACT
          REAL(q) :: SIF(3,3) 
          REAL(q) :: TEIN,TEIN_at,TEIN_lat,EKIN_at
          REAL(q) :: V_FORCE_L(3,3),C_FORCE_L(3,3)
          REAL(q),SAVE :: ROTMAT1(3,3)
          REAL(q) :: ROTMAT2(3,3)
          REAL(q) :: kinetic_stress(3,3),SIF_momentum(3,3),SIF_stoch(3,3)
          INTEGER :: IDUM,IERR,N
          LOGICAL :: LDUM,LOPEN
          COMPLEX(q) :: CDUM
          REAL(q) :: RDUM
          CHARACTER*1 :: CHARAC
          REAL(q),PARAMETER :: tol=1e-6
          INTEGER,PARAMETER :: MAXITER=100
          REAL(q) :: err1,err2
          REAL(q) :: Avel_(3,3),Avel_old(3,3)
          REAL(q) :: vel_(3,T_INFO%NIONS), vel_old(3,T_INFO%NIONS)
          LOGICAL, SAVE :: LATTICE_CONSTRAINTS(3) ! specifies which of the lattice parameters (a_1,a_2,a_3) are free (true) or fixed (false)
          INTEGER, SAVE :: LATTICE_DOF ! the total number of the lattice degrees of freedom according to the applied lattice constraints
          REAL(q) :: hills_accel(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS),hspring_accel(3,T_INFO%NIONS)
          REAL(q) :: hills_accel_L(3,3),penalty_accel_L(3,3),hspring_accel_L(3,3)
          INTEGER :: NI, NT
          REAL(q) :: VTMP(3)
          REAL(q) :: dummy, maxForce
          LOGICAL :: LSTOP2
          REAL(q),SAVE :: EPOT_old
          REAL(q) :: hspring_R0(iconst8),hspring_K(iconst8),hspring_force(iconst8)

 72     FORMAT( ' POSITION    ',35X,'REST-FORCE (eV/Angst)'/ &
     &          ' ----------------------------------------------', &
     &          '-------------------------------------')
 76     FORMAT((3F13.5,3X,3F14.6))

          !c add this step to total count
          counter=counter+1

          V_FORCE_L=0._q
          C_FORCE_L=0._q
          V_FORCE=0._q
          C_FORCE=0._q
          hills_accel_L=0._q;penalty_accel_L=0._q;hspring_accel_L=0._q
          hills_accel=0._q;penalty_accel=0._q;hspring_accel=0._q

          !c first rotate cell with cell vectors to triangular matrix
          CALL ROTATE_CELL(IO%IU6,LATT_CUR,ROTMAT1)

          !c transform the triangular matrix with lattice vectors back to the original form 
          CALL ROTATE_BACK(LATT_CUR,ROTMAT1)

          IF (counter==1) THEN
            !c make sure that energy convergence criterion is not fullfiled 
            !c in the first relaxation step (irelevant for MD)
            IF (counter==1) EPOT_old=EPOT+10*ABS(DYN%EDIFFG)

            ! read-in mass for lattice parameters
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
           
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'PMASS','=','#',';','F', &
                IDUM,Amass,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) Amass=1000._q
            IF (((IERR/=0).AND.(IERR/=3)) .OR. ((IERR==0).AND.(N<1))) THEN
              Amass=1000._q
              IF (IO%IU0>=0) &
                & WRITE(IO%IU0,*)'Error reading item ''PMASS'' from file INCAR'
            ENDIF

            LATTICE_CONSTRAINTS = .TRUE.
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'LATTICE_CONSTRAINTS','=','#',';','L', &
                IDUM,RDUM,CDUM,LATTICE_CONSTRAINTS,CHARAC,N,3,IERR)
            IF (IERR==3) LATTICE_CONSTRAINTS = .TRUE.
            IF (((IERR/=0).AND.(IERR/=3)) .OR. ((IERR==0).AND.(N/=3))) THEN
              LATTICE_CONSTRAINTS = .TRUE.
              IF (IO%IU0>=0) &
                & WRITE(IO%IU0,*)'Error reading lattice constraints ''LATTICE_CONSTRAINTS'' from file INCAR'
            ENDIF

            CLOSE(IO%IU5)
   
            ! count the lattice degrees of freedom
            LATTICE_DOF = 0
            DO I = 1, 3
              IF (LATTICE_CONSTRAINTS(I)) LATTICE_DOF = LATTICE_DOF + 1
            END DO
            LATTICE_DOF = LATTICE_DOF * LATTICE_DOF

            IF (IO%IU6>0) THEN
              write(g_io%REPORT,FMT='(/,2X,A35)') '>Damping parameter for lattice DOF:'
              write(g_io%REPORT,FMT='(3X,A22,X,F10.5)')    '              PMASS = ', AMASS  
              write(g_io%REPORT,FMT='(3X,A22,X,3L,A,I1,A)') 'LATTICE_CONSTRAINTS = ', (LATTICE_CONSTRAINTS(I), I=1,3), &
              & ' (X,Y,Z) = ', LATTICE_DOF, ' degree(s) of freedom'
              write(g_io%REPORT,FMT='(/)')
            ENDIF 

            !c initialize the lattice velocities if needed 
            IF (LATT_CUR%INITlatv<0) THEN
              IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A41)')  "Lattice velocities initialized by STEP_tb"
              CALL init_Avelocities(LATT_CUR%Avel,DYN,Amass)
              CALL APPLY_STRESS_CONSTRAINT(LATT_CUR%AVEL, Aaccel, LATTICE_CONSTRAINTS)
              LATT_CUR%INITlatv=1
            ENDIF
          ENDIF


       !c metadynamics stuff 
           !c update history of fictioous particles and of collective variables
           IF (iconst5 >0) THEN
             j=0
             DO i=1,ICOORDINATES%NUMINTERNALS
               IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                 j=j+1
                 hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE 
               ENDIF
             ENDDO
          
             !c add new hill
             IF (counter .GE. hills%bin) THEN
               CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
             ENDIF 

             !c calculate the penalty potential
             CALL penalty_potential(penalty,hills,ICOORDINATES)
 
             !c compute contributions from bias potentials
             CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS+9,AMASS)
             CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,penalty%force,ICOORDINATES,DYN%POSIOC,penalty_accel,penalty_accel_L,3*T_INFO%NIONS+9,AMASS)
             !hills%force=hills%force+penalty%force
             !CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,hspring_force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS+9,AMASS)
           ENDIF

          !c calculate forces due to harmonic springs
          !IF (iconst8>0) call harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
          IF (iconst8>0) THEN
            CALL harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
            CALL harmonic_spring_accel(T_INFO,LATT_CUR,DYN,hspring_force,ICOORDINATES,DYN%POSIOC,hspring_accel,hspring_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !!!DYN%POSIOC=DYN%POSION

          !c compute internal stress for the current time slice
          Aaccel=SIF !!+kinetic_stress

          !c take care of external pressure
          DO i=1,3
            Aaccel(i,i)=Aaccel(i,i)-DYN%PSTRESS/(EVTOJ*1E22_q)*LATT_CUR%OMEGA
          ENDDO
          
          !c compute forces
          Aaccel=MATMUL(Aaccel,(LATT_CUR%B))
         
          !c compute (one half of)acceleration
          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          Aaccel=Aaccel*FACT/2/Amass 
       
          !c compute stochastic forces
          CALL friction_Forces_Lstoch(SIF_stoch,DYN,GAMMA_L,AMASS)
          CALL friction_Forces_stoch(D2C_stoch,T_INFO,DYN,LATT_CUR,GAMMA)
          Aaccel=Aaccel+SIF_stoch


          ! update partly the velocities
          LATT_CUR%Avel=LATT_CUR%Avel+Aaccel
          ! set Avel here
          CALL APPLY_STRESS_CONSTRAINT(LATT_CUR%AVEL, Aaccel, LATTICE_CONSTRAINTS)
          DYN%VEL=DYN%VEL+DYN%D2C+D2C_stoch

!           IF (iconst5 .GT. 0) THEN
!             LATT_CUR%Avel=LATT_CUR%Avel-(hills_accel_L+penalty_accel_L)
!             DYN%VEL=DYN%VEL-(hills_accel+penalty_accel)
!           ENDIF
          IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0)) THEN
            LATT_CUR%Avel=LATT_CUR%Avel-(hills_accel_L+penalty_accel_L+hspring_accel_L)
            DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)
          ENDIF

          Avel_=LATT_CUR%Avel
          vel_=DYN%VEL
          
          !c the line below was incorrect!!
          !!Ginv=MATMUL(LATT_CUR%B,TRANSPOSE(LATT_CUR%B))
          Ginv=MATMUL(TRANSPOSE(LATT_CUR%B),LATT_CUR%B)

          !c velocities for time slice (t) depend on v(t)
          !c - iterative solution is needed
          i=0
          DO
            i=i+1
            IF (i .GT. MAXITER) THEN
              IF (IO%IU6>0) THEN
                write(*,*) "Error: convergence problem in MD!"
              ENDIF
              STOP
            ENDIF 
            Avel_old=LATT_CUR%Avel
            vel_old=DYN%VEL
            !c the line below was incorrect!!
            !!Gdot=MATMUL(LATT_CUR%Avel,TRANSPOSE(LATT_CUR%A))
            Gdot=MATMUL(TRANSPOSE(LATT_CUR%Avel),(LATT_CUR%A))
            Gdot=0.5*(Gdot+TRANSPOSE(Gdot))
            Gdot=MATMUL(Ginv,Gdot)

            !c compute kinetic stress
            CALL COMPUTE_KINETIC_STRESS(T_INFO,LATT_CUR,DYN,kinetic_stress,SIF,-1)
            kinetic_stress=MATMUL(kinetic_stress,(LATT_CUR%B))
            FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
            kinetic_stress=kinetic_stress*FACT/2/Amass 

            !c compute accel. due to friction term 
            CALL friction_Forces_Lmomentum(SIF_momentum,LATT_CUR%Avel,DYN,GAMMA_L)
            CALL friction_Forces_momentum(D2C_momentum,T_INFO,DYN,GAMMA)

            LATT_CUR%Avel=Avel_+SIF_momentum+kinetic_stress

            ! set Avel here
            CALL APPLY_STRESS_CONSTRAINT(LATT_CUR%AVEL, Aaccel, LATTICE_CONSTRAINTS)

            DYN%VEL=vel_+D2C_momentum-MATMUL(Gdot,DYN%VEL)

            !c geometric constraints
            IF (ICOORDINATES%NUMINTERNALS>0) THEN                            
               CALL Rattle(counter,T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS+9,&
               &   AMASS,LATT_CUR%A,ECONST,iconst0,iconst2,1)
              !CALL Rattle_vel(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
              DYN%VEL=DYN%VEL+V_FORCE
              LATT_CUR%Avel=LATT_CUR%Avel+V_FORCE_L
            ENDIF

            Avel_old=ABS(Avel_old-LATT_CUR%Avel)
            vel_old=ABS(vel_old-DYN%VEL)
            err1=MAXVAL(Avel_old)
            err2=MAXVAL(vel_old)
            IF ((err1<tol) .AND. (err2<tol)) EXIT
          ENDDO

          !update acceleration for lattice components for the time slice (t)
          Aaccel=Aaccel+SIF_momentum+kinetic_stress

          CALL APPLY_STRESS_CONSTRAINT(LATT_CUR%AVEL, Aaccel, LATTICE_CONSTRAINTS)

          !c compute temperature for the slice t
          CALL KINETIC_E(EKIN_at,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)

          TEIN_at = 2*EKIN_at/BOLKEV/NDEGREES_OF_FREEDOM_

          CALL KINETIC_E_Lat(EKIN_Lat,AMASS,DYN%POTIM,LATT_CUR%AVEL)
   
          !!c upper triangle only
          ECONST=0._q
          !!TEIN_lat = 2*EKIN_Lat/BOLKEV/6
          !!TEIN=2*(EKIN+EKIN_Lat) /BOLKEV/(NDEGREES_OF_FREEDOM_+6) 
          TEIN_lat = 2*EKIN_Lat/BOLKEV/LATTICE_DOF
          EKIN=EKIN_at+EKIN_lat
          TEIN=2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM_+LATTICE_DOF) 
          

          !c v(t+dt/2)=v(t)dt+f(t)/(2m)*dt^2
          LATT_CUR%Avel=LATT_CUR%Avel+Aaccel 

          ! set Avel here
          CALL APPLY_STRESS_CONSTRAINT(LATT_CUR%AVEL, Aaccel, LATTICE_CONSTRAINTS)

          DYN%VEL=DYN%VEL+DYN%D2C+D2C_momentum+D2C_stoch-MATMUL(Gdot,DYN%VEL)

          !c add contribution from bias potential
!           IF (iconst5 .GT. 0) THEN
!             LATT_CUR%Avel=LATT_CUR%Avel-(hills_accel_L+penalty_accel_L)
!             DYN%VEL=DYN%VEL-(hills_accel+penalty_accel)
!           ENDIF

          IF ((iconst5 .GT. 0) .OR. (iconst8 .GT. 0)) THEN
            LATT_CUR%Avel=LATT_CUR%Avel-(hills_accel_L+penalty_accel_L+hspring_accel_L)
            DYN%VEL=DYN%VEL-(hills_accel+penalty_accel+hspring_accel)
          ENDIF

          !c r(t+dt)=r(t)+v'(t+dt)*dt
          AC=LATT_CUR%A+LATT_CUR%Avel
          DYN%POSION=DYN%POSIOC+DYN%VEL

          !c geometric constraints (RATTLE)
          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            CALL Rattle(counter,T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS+9,&
            &   AMASS,AC,ECONST,iconst0,iconst2,0)
            DYN%POSION=DYN%POSION+C_FORCE
            AC=AC+C_FORCE_L
            DYN%VEL=DYN%VEL+C_FORCE
            LATT_CUR%Avel=LATT_CUR%Avel+C_FORCE_L
          ENDIF

          IF (IO%IU6>0) THEN     
            write(g_io%REPORT,FMT='(/,3X,A4,4(X,F10.3))') 't_b>', DYN%TEMP,TEIN_at, TEIN_lat,TEIN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)

            !c output for metadynamics
            IF (iconst5>0) THEN
              write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                  write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5)') &
                  & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE
                ENDIF
              ENDDO   
            ENDIF 
          ENDIF

          !c if this MD subroutine is used for relaxation
          IF (DYN%TEBEG<0.1) THEN

            !c take care of constraints
            IF (iconst0>0) THEN
              !c compute the total restoring force
              C_FORCE=C_FORCE+V_FORCE
              V_FORCE=0._q

              V_FORCE=DYN%D2C+C_FORCE

              FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
              NI=1
              DO NT=1,T_INFO%NTYP
                DO NI=NI,T_INFO%NITYP(NT)+NI-1
                  V_FORCE(1,NI)=V_FORCE(1,NI)/FACT*2*T_INFO%POMASS(NT)
                  V_FORCE(2,NI)=V_FORCE(2,NI)/FACT*2*T_INFO%POMASS(NT)
                  V_FORCE(3,NI)=V_FORCE(3,NI)/FACT*2*T_INFO%POMASS(NT)
                ENDDO
              ENDDO

              CALL DIRKAR(T_INFO%NIONS,V_FORCE,LATT_CUR%A)

             !c write ionic forces after removing the restoring forces
              IF (IO%IU0>0) THEN
                WRITE(IO%IU6,72)
                DO J=1,T_INFO%NIONS
                  VTMP=T_INFO%POSION(1:3,J)
                  CALL  DIRKAR(1,VTMP,LATT_CUR%A)
                  WRITE(IO%IU6,76) VTMP,(V_FORCE(I,J),I=1,3)
                ENDDO
              ENDIF 
            ENDIF

            !c test the convergence criteria
            maxForce=0._q
            DO i=1,T_INFO%NIONS
              dummy=V_FORCE(1,i)*V_FORCE(1,i)+V_FORCE(2,i)*V_FORCE(2,i)+V_FORCE(3,i)*V_FORCE(3,i)
              IF (dummy>maxForce) &
                & maxForce=dummy
            END DO
            maxForce=maxForce**0.5
            IF (IO%IU0>0) THEN
              write(*,*) 'maxForce',maxForce,DYN%EDIFFG
            ENDIF 

            IF (DYN%EDIFFG<0.) THEN
              LSTOP2=.TRUE.

              IF (maxForce>ABS(DYN%EDIFFG)) LSTOP2=.FALSE.

              IF (LSTOP2) THEN
                INFO%LSTOP=.TRUE.
                !DYN%POSIOC=DYN%POSION
                DYN%POSION=DYN%POSIOC
                RETURN
              END IF
            
            ELSE IF (counter>1) THEN
            !IF (counter>1) THEN
              IF (IO%IU0>0) THEN
                write(*,*) 'dEPOT',ABS(EPOT-EPOT_old),DYN%EDIFFG
              ENDIF
              INFO%LSTOP=(ABS(EPOT-EPOT_old)<DYN%EDIFFG)
            ENDIF
          ENDIF


          !c update direct and reciprocal lattice:
          LATT_CUR%A=AC
          CALL LATTIC(LATT_CUR)

         !c first rotate cell with cell vectors to triangular matrix
         CALL ROTATE_CELL(IO%IU6,LATT_CUR,ROTMAT2)

         !c transform the triangular matrix with lattice vectors back to the original form 
         CALL ROTATE_BACK(LATT_CUR,ROTMAT1) 

         ROTMAT1=ROTMAT2
!           !c transform the triangular matrix with lattice vectors back to the original form 
!           CALL ROTATE_BACK(LATT_CUR,ROTMAT)
 
          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_LANGEVIN_ISIF3

        SUBROUTINE isotrop( m)
          REAL(q) :: m(3,3)
          REAL(q) :: mean

          mean=(m(1,1)+m(2,2)+m(3,3))/3
          m=0
          m(1,1)=mean
          m(2,2)=mean
          m(3,3)=mean
        END SUBROUTINE isotrop

        SUBROUTINE APPLY_STRESS_CONSTRAINT(AVEL, STRESS, FREE_AXES)
          REAL(q), INTENT(INOUT) :: AVEL(3,3), STRESS(3,3)
          LOGICAL, INTENT(IN) :: FREE_AXES(3)
          INTEGER :: I, J
!          RETURN
          DO I = 1,3
            IF (.NOT. FREE_AXES(I)) THEN
              AVEL(I,:) = 0.0_q
              AVEL(:,I) = 0.0_q
!              STRESS(I,:) = 0.0_q
!              STRESS(:,I) = 0.0_q
            END IF
          END DO
        END SUBROUTINE APPLY_STRESS_CONSTRAINT

        SUBROUTINE ROTATE_CELL(IU,LATT_CUR,ROTMAT)
          TYPE(latt) :: LATT_CUR
          REAL(q) :: newA(3,3)
          REAL(q) :: oldA(3,3)
          REAL(q) :: atmp(3,3) 
          REAL(q) :: vtmp(3),vtmpNorm
          REAL(q) :: ROTMAT(3,3),ROTMAT_inv(3,3)
          REAL(q) :: r1,r2,r3
          REAL(q) :: cos23,cosX,cosY,sin23,sinX,sinY
          INTEGER :: ninfo,IU


          ROTMAT=0._q

          oldA=LATT_CUR%A
          atmp=MATMUL(transpose(LATT_CUR%A),LATT_CUR%A) 

          newA=0._q
          newA(3,3) = sqrt( atmp(3,3) )
          newA(3,2) = atmp(3,2) / sqrt( atmp(3,3) )
          newA(2,2) = sqrt( atmp(2,2) -   &
          &   atmp(3,2) * atmp(3,2) / atmp(3,3) )
          newA(3,1) = atmp(3,1) / sqrt( atmp(3,3) )
          newA(2,1) = ( atmp(3,3) * atmp(2,1) - atmp(3,2) * atmp(3,1) ) / &
          &      sqrt( atmp(3,3) * atmp(3,3) * atmp(2,2) - &
          &      atmp(3,3) * atmp(3,2) * atmp(3,2) )
          newA(1,1) = sqrt( atmp(1,1) -  &
          &    newA(3,1) * newA(3,1) - newA(2,1) * newA(2,1) )


!           r1=SQRT(atmp(1,1))
!           r2=SQRT(atmp(2,2))
!           r3=SQRT(atmp(3,3))
!           cos23=atmp(2,3)/r2/r3
!           sin23=SQRT(1._q-cos23**2)
! 
!           newA=0._q
!           newA(3,3)=r3
!           newA(3,2)=r2*cos23
!           newA(2,2)=r2*sin23
! 
!           !c vector orthogonal to the (b,c) plane
!           vtmp=CROSSPROD(3,oldA(1:3,2),oldA(1:3,3))
!           vtmpNorm=SQRT(SUM(vtmp**2))
!           !vtmp=vtmp/vtmpNorm
!           cosX=SUM(oldA(1:3,1)*vtmp)/r1/vtmpNorm
!           !sinX=SQRT(1._q-cosX**2)
!           newA(1,1)=r1*ABS(cosX)
! 
!           !newA(1,1)=r1*sinX
! 
!           !c the part of vector a that is orthogonal to the (b,c) plane
!           vtmp=oldA(1:3,1)-r1*cosX*vtmp/vtmpNorm
!           !vtmp=oldA(1:3,1)-SUM(oldA(1:3,1)*vtmp)*vtmp
!           vtmpNorm=SQRT(SUM(vtmp**2))
!           cosY=SUM(vtmp*oldA(1:3,3))/r3/vtmpNorm
!           sinY=SQRT(1-cosY**2)
!           newA(3,1)=vtmpNorm*cosY
!           newA(2,1)=vtmpNorm*sinY

         

!           IF (IU>=0) THEN
!             write(*,*) "oldA"
!             write(*,*) oldA(1,1),oldA(1,2),oldA(1,3)
!             write(*,*) oldA(2,1),oldA(2,2),oldA(2,3)
!             write(*,*) oldA(3,1),oldA(3,2),oldA(3,3)
! 
!             write(*,*) "newA"
!             write(*,*) newA(1,1),newA(1,2),newA(1,3)
!             write(*,*) newA(2,1),newA(2,2),newA(2,3)
!             write(*,*) newA(3,1),newA(3,2),newA(3,3)
!           ENDIF

          !c compute the matrix that transforms the triangular matrix
          !c to the original matrix
          ROTMAT=newA
          CALL SVDINVERSE(ROTMAT,3,ninfo)
          ROTMAT=MATMUL(oldA,ROTMAT)
          ROTMAT_inv=ROTMAT
          CALL SVDINVERSE(ROTMAT_inv,3,ninfo)

          atmp=MATMUL(ROTMAT,newA)   
!           IF (IU>=0) THEN
!             write(*,*) "atmp"
!             write(*,*) atmp(1,1),atmp(1,2),atmp(1,3)
!             write(*,*) atmp(2,1),atmp(2,2),atmp(2,3)
!             write(*,*) atmp(3,1),atmp(3,2),atmp(3,3)
!           ENDIF

          !c update LATT_CUR
          LATT_CUR%A=newA
          CALL LATTIC(LATT_CUR)
          LATT_CUR%AVEL=MATMUL(ROTMAT_inv,LATT_CUR%AVEL)

        END SUBROUTINE ROTATE_CELL

        SUBROUTINE ROTATE_BACK(LATT_CUR,ROTMAT)
        !c rotate the triangular to the original matrix
          TYPE(latt) :: LATT_CUR
          REAL(q) :: newA(3,3)
          REAL(q) :: oldA(3,3)
          REAL(q) :: ROTMAT(3,3)

          oldA=LATT_CUR%A
          newA=MATMUL(ROTMAT,oldA)          
          LATT_CUR%A=newA
          CALL LATTIC(LATT_CUR)
          LATT_CUR%AVEL=MATMUL(ROTMAT,LATT_CUR%AVEL)
        END SUBROUTINE ROTATE_BACK

        SUBROUTINE COMPUTE_KINETIC_STRESS(T_INFO,LATT_CUR,DYN,kinetic_stress,TSIF,IU)
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(dynamics) :: DYN 
          REAL(q) :: Vtmp(3,T_INFO%NIONS)
          REAL(q) :: tmp(3,3),kinetic_stress(3,3),TSIF(3,3)
          INTEGER ::NI,NT,IU
          REAL(q) :: fac,fakt

          kinetic_stress=0._q

          Vtmp=DYN%VEL/DYN%POTIM
          Vtmp=MATMUL(LATT_CUR%A,Vtmp)
          
          NI=1
          DO NT=1, T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
              tmp=OUTERPROD(3,Vtmp(:,NI),Vtmp(:,NI))
              kinetic_stress=kinetic_stress+tmp*T_INFO%POMASS(nt)
            ENDDO
          ENDDO

          kinetic_stress=(kinetic_stress+transpose(kinetic_stress))/2
          kinetic_stress=kinetic_stress/LATT_CUR%OMEGA

          fakt = evtoj*1e22_q/latt_cur%omega
          fac  = amtokg *  &
          &     1e5_q  *  &
          &     1e5_q  *  &
          &     1e30_q *  &
          &     1e-8_q

          IF (IU>=0) THEN
          write(iu,'(''  ideal gas correction = '',f9.2,'' kB'')') &
     &     fac*(kinetic_stress(1,1)+kinetic_stress(2,2)+kinetic_stress(3,3))/3
      write(iu,'(''  Total+kin._'',6f12.3)') &
     &     tsif(1,1)*fakt + fac*kinetic_stress(1,1),  &
     &     tsif(2,2)*fakt + fac*kinetic_stress(2,2),  &
     &     tsif(3,3)*fakt + fac*kinetic_stress(3,3),  &
     &     tsif(1,2)*fakt + fac*kinetic_stress(1,2),  &
     &     tsif(2,3)*fakt + fac*kinetic_stress(2,3),  &
     &     tsif(3,1)*fakt + fac*kinetic_stress(3,1)
          ENDIF
       
          kinetic_stress=kinetic_stress*fac/fakt

        END SUBROUTINE COMPUTE_KINETIC_STRESS

        SUBROUTINE STEP_ANDERSEN_ISIF3(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EKIN_Lat,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO, &
          &      EPOT,ANDERSEN_PROB,g_io,SIF,iconst0,iconst2,TEIN)
        !c Andersen thermostat
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,EKIN_at,EKIN_Lat,ECONST
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,N
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          REAL(q) :: ANDERSEN_PROB
          INTEGER :: random_counter,IDUM,IERR
          LOGICAL :: LDUM,LOPEN
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q) :: AC(3,3),Aaccel(3,3),Avel_tmp(3,3),Ginv(3,3),Gdot(3,3)
          REAL(q),SAVE :: Amass !,Avel(3,3)
          REAL(q) :: dumM(9),FACT
          REAL(q) :: SIF(3,3) 
          REAL(q) :: TEIN,TEIN_at,TEIN_lat
          REAL(q) :: C_FORCE_L(3,3)
          REAL(q) :: ROTMAT(3,3)
          REAL(q) :: kinetic_stress(3,3)
          LOGICAL :: LSCALE,LCMASS

          LSCALE=.FALSE.
          LCMASS=.FALSE.
     
          !c first rotate cell with cell vectors to triangular matrix
          CALL ROTATE_CELL(IO%IU6,LATT_CUR,ROTMAT)
        

          IF (IO%IU6>0) THEN
              write(*,*) 'AVEL:'
              write(*,*) LATT_CUR%Avel(1,1),LATT_CUR%Avel(2,1),LATT_CUR%Avel(3,1)
              write(*,*) LATT_CUR%Avel(1,2),LATT_CUR%Avel(2,2),LATT_CUR%Avel(3,2)
              write(*,*) LATT_CUR%Avel(1,3),LATT_CUR%Avel(2,3),LATT_CUR%Avel(3,3)
          ENDIF 

          !c add this step to total count
          counter=counter+1
         
          !IF (DYN%INIT==0) THEN
          IF (counter==1) THEN
            ! read-in mass for lattice parameters
            OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
           
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'PMASS','=','#',';','F', &
                IDUM,Amass,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) Amass=1000._q
            IF (((IERR/=0).AND.(IERR/=3)) .OR. ((IERR==0).AND.(N<1))) THEN
              Amass=1000._q
              IF (IO%IU0>=0) &
                & WRITE(IO%IU0,*)'Error reading item ''PMASS'' from file INCAR'
            ENDIF

            CLOSE(IO%IU5)
   
            IF (IO%IU6>0) THEN
              write(g_io%REPORT,FMT='(/,2X,A35)') '>Damping parameter for lattice DOF:'
              write(g_io%REPORT,FMT='(3X,A22,X,F10.5)')    '              PMASS = ', AMASS  
              write(g_io%REPORT,FMT='(/)')
            ENDIF 

            !c initialize the lattice velocities if needed 
            IF (LATT_CUR%INITlatv<0) THEN
              IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A41)')  "Lattice velocities initialized by STEP_tb"
              CALL init_Avelocities(LATT_CUR%Avel,DYN,Amass)
              LATT_CUR%INITlatv=1
            ENDIF
          ENDIF

          !!!DYN%POSIOC=DYN%POSION

          !c compute kinetic stress, this should be done using velocities
          !c computed for the same time step as the stress tensor 
          CALL COMPUTE_KINETIC_STRESS(T_INFO,LATT_CUR,DYN,kinetic_stress,SIF,-1)

!           IF (IO%IU6>=0) THEN
!             DO i=1,3
!               write(*,*) 'kinetic_stress:', kinetic_stress(i,1),kinetic_stress(i,2),kinetic_stress(i,3)
!             ENDDO
!           ENDIF

          Aaccel=SIF+kinetic_stress
 
          DO i=1,3
            !write(*,*) 'SIF:', SIF(i,1),SIF(i,2),SIF(i,3)
            Aaccel(i,i)=Aaccel(i,i)-DYN%PSTRESS/(EVTOJ*1E22_q)*LATT_CUR%OMEGA
          ENDDO
          
          Aaccel=MATMUL(Aaccel,LATT_CUR%B)
          
          !c the line below was incorrect!!
          !!Ginv=MATMUL(LATT_CUR%B,TRANSPOSE(LATT_CUR%B))
          Ginv=MATMUL(TRANSPOSE(LATT_CUR%B),LATT_CUR%B)
          Gdot=MATMUL(TRANSPOSE(LATT_CUR%Avel),LATT_CUR%A)
          Gdot=0.5*(Gdot+TRANSPOSE(Gdot))
          
          Gdot=MATMUL(Ginv,Gdot)
          

           !upper triangle only!!!
           DO i=1,3
             DO j=1,3
               IF (j>i)  Aaccel(i,j)=0._q
            ENDDO 
          ENDDO
          
          !DO i=1,3
          !  write(*,*) 'Aa2',Aaccel(i,1),Aaccel(i,2),Aaccel(i,3)
          !ENDDO   

          !c compute acceleration
          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          Aaccel=Aaccel*FACT/2/Amass 

          !DO i=1,3
          !  write(*,*) 'Aa3',Aaccel(i,1),Aaccel(i,2),Aaccel(i,3)
          !ENDDO   

          !DO i=1,3
          !  write(*,*) 'Avel',Avel(i,1),Avel(i,2),Avel(i,3)
          !ENDDO   

          
          VEL_tmp=0._q
          CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
          CALL init_Avelocities(Avel_tmp,DYN,Amass)

          

          CMASS=0._q

          !DO i=1,3
          !  write(*,*) 'Aa3',Gdot(i,1),Gdot(i,2),Gdot(i,3)
          !ENDDO  

          DYN%VEL=DYN%VEL+DYN%D2C-MATMUL(Gdot,DYN%VEL)

          LATT_CUR%Avel=LATT_CUR%Avel+Aaccel

          random_counter=0
          IF (DYN%INIT==1) THEN
            DO i=1,T_INFO%NIONS
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                DYN%VEL(:,i)=VEL_tmp(:,i)
              ENDIF
            ENDDO

            DO i=1,3              
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                LATT_CUR%Avel(:,i)=Avel_tmp(:,i)
              ENDIF
            ENDDO            
          ENDIF

          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)
          DO i=1,T_INFO%NIONS
            DO j=1,3
              IF (T_INFO%LSFOR(j,i)) DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
          ENDDO
          !CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          DYN%VEL=DYN%VEL+DYN%D2C
          LATT_CUR%Avel=LATT_CUR%Avel+Aaccel

          C_FORCE=0._q
          C_FORCE_L=0._q
          DYN%POSION=DYN%POSIOC+DYN%VEL
          AC=LATT_CUR%A+LATT_CUR%Avel

          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,&
            &   3*T_INFO%NIONS+9,AMASS,AC,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            AC=AC+C_FORCE_L
          ENDIF
          DYN%VEL=DYN%VEL+C_FORCE
          LATT_CUR%AVEL=LATT_CUR%AVEL+C_FORCE_L

          !c update direct and reciprocal lattice:
          LATT_CUR%A=AC
          CALL LATTIC(LATT_CUR)

          CALL KINETIC_E(EKIN_at,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)

          TEIN_at = 2*EKIN_at/BOLKEV/NDEGREES_OF_FREEDOM_

          CALL KINETIC_E_Lat(EKIN_Lat,AMASS,DYN%POTIM,LATT_CUR%AVEL)
          !TEIN_lat = 2*EKIN_Lat/BOLKEV/9
          !TEIN=2*(EKIN+EKIN_Lat) /BOLKEV/(NDEGREES_OF_FREEDOM_+9)
          !c upper triangle only
          TEIN_lat = 2*EKIN_Lat/BOLKEV/6
          EKIN=EKIN_at+EKIN_Lat
          TEIN=2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM_+6) 
          IF (IO%IU6>0) THEN     
            write(g_io%REPORT,FMT='(/,3X,A4,4(X,F10.3))') 't_b>', DYN%TEMP,TEIN_at, TEIN_lat,TEIN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter
          ENDIF


          !c transform the triangular matrix with lattice vectors for the original form 
          CALL ROTATE_BACK(LATT_CUR,ROTMAT)

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_ANDERSEN_ISIF3


        SUBROUTINE STEP_NANDERSEN(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_,IO,EPOT,&
          &      g_io,NSUBSYS,TSUBSYS,PSUBSYS,iconst0,iconst2,TEIN)
        !c Andersen thermostat
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ECONST
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,NDEGREES_OF_FREEDOM_
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp1(3,T_INFO%NIONS),VEL_tmp2(3,T_INFO%NIONS),VEL_tmp3(3,T_INFO%NIONS)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: dummy
          INTEGER :: random_counter(3)
          INTEGER :: NSUBSYS(3)
          REAL(q) :: TSUBSYS(3),PSUBSYS(3)
          REAL(q) :: EKIN1,EKIN2,EKIN3,TEIN1,TEIN2,TEIN3,TEIN
          INTEGER,SAVE :: NDOF(3)
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          LOGICAL :: LSCALE,LCMASS

          LSCALE=.FALSE.
          LCMASS=.TRUE.

          !c add this step to total count
          counter=counter+1
          !!!DYN%POSIOC=DYN%POSION

          AMASS=0._q

          !c generate Anderesen particles with desired temperatures (3 subsystems assumed)
          VEL_tmp1=0._q;VEL_tmp2=0._q;VEL_tmp3=0._q
          DYN%TEMP=TSUBSYS(1)
          CALL init_velocities(VEL_tmp1,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
          IF (NSUBSYS(2)>NSUBSYS(1)) THEN
            DYN%TEMP=TSUBSYS(2)
            CALL init_velocities(VEL_tmp2,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
            IF (NSUBSYS(3)>NSUBSYS(2)) THEN
              DYN%TEMP=TSUBSYS(3)
              CALL init_velocities(VEL_tmp3,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_,LSCALE,LCMASS)
            ENDIF
          ENDIF

          VEL_tmp=VEL_tmp1
          
          IF (NSUBSYS(1)<NSUBSYS(2)) THEN
            DO i=NSUBSYS(1)+1,NSUBSYS(2)
              VEL_tmp(1:3,i)=VEL_tmp2(1:3,i)
            ENDDO
            IF (NSUBSYS(2)<NSUBSYS(3)) THEN
              DO i=NSUBSYS(2)+1,NSUBSYS(3)
                VEL_tmp(1:3,i)=VEL_tmp3(1:3,i)
              ENDDO
            ENDIF
          ENDIF

          !c remove translational motion
          !CMASS=0._q
          !CALL GIVE_CMASS(T_INFO,VEL_tmp,CMASS)
          !DO i=1,T_INFO%NIONS
          !  DO j=1,3
          !    IF (T_INFO%LSFOR(j,i)) VEL_tmp(j,i)=VEL_tmp(j,i)-CMASS(j,1)
          !  ENDDO
          !ENDDO
          !CMASS=0._q        

          IF (counter==1) THEN
            !c velocity initialization
            IF (DYN%INIT==0) DYN%VEL=VEL_tmp

            !c determine number of degrees of freedom for each subsystem
            NDOF=0
            DO i=1,NSUBSYS(1)
              DO j=1,3
                IF (T_INFO%LSFOR(j,i)) NDOF(1)=NDOF(1)+1
              ENDDO
            ENDDO
            DO i=NSUBSYS(1)+1,NSUBSYS(2)
              DO j=1,3
                IF (T_INFO%LSFOR(j,i)) NDOF(2)=NDOF(2)+1
              ENDDO
            ENDDO
            DO i=NSUBSYS(2)+1,NSUBSYS(3)
              DO j=1,3
                IF (T_INFO%LSFOR(j,i)) NDOF(3)=NDOF(3)+1
              ENDDO
            ENDDO
          END IF

          DYN%VEL=DYN%VEL+2*DYN%D2C
          random_counter=0

          IF (DYN%INIT==1) THEN
            DO i=1,NSUBSYS(1)
              CALL RANE_ION(dummy)
              IF (dummy<PSUBSYS(1)) THEN
                random_counter(1)=random_counter(1)+1
                DYN%VEL(1:3,i)=VEL_tmp(1:3,i)
              ENDIF
            ENDDO
            DO i=NSUBSYS(1)+1,NSUBSYS(2)
              CALL RANE_ION(dummy)
              IF (dummy<PSUBSYS(2)) THEN
                random_counter(2)=random_counter(2)+1
                DYN%VEL(1:3,i)=VEL_tmp(1:3,i)
              ENDIF
            ENDDO
            DO i=NSUBSYS(2)+1,NSUBSYS(3)
              CALL RANE_ION(dummy)
              IF (dummy<PSUBSYS(3)) THEN
                random_counter(3)=random_counter(3)+1
                DYN%VEL(1:3,i)=VEL_tmp(1:3,i)
              ENDIF
            ENDDO
          ENDIF

          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)
          DO i=1,T_INFO%NIONS
            DO j=1,3
              IF (T_INFO%LSFOR(j,i)) DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
          ENDDO
          !CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          C_FORCE=0._q
          DYN%POSION=DYN%POSIOC+DYN%VEL

          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES, &
            &   g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF
          
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)

          TEIN = 2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM_

          IF (IO%IU6>0) CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)

          !c compute temperatures for all subsystems
          VEL_tmp=0._q
          VEL_tmp(1:3,1:NSUBSYS(1))=DYN%VEL(1:3,1:NSUBSYS(1))

          CALL KINETIC_E(EKIN1,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,VEL_tmp)
          TEIN1 = 2*EKIN1/BOLKEV/NDOF(1)*SUM(NDOF)/NDEGREES_OF_FREEDOM_ 


          IF (NSUBSYS(2)>NSUBSYS(1)) THEN
            VEL_tmp=0._q
            VEL_tmp(1:3,NSUBSYS(1)+1:NSUBSYS(2))=DYN%VEL(1:3,NSUBSYS(1)+1:NSUBSYS(2))
            CALL KINETIC_E(EKIN2,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                        & DYN%POTIM,LATT_CUR%A,VEL_tmp)
            TEIN2 = 2*EKIN2/BOLKEV/NDOF(2)*SUM(NDOF)/NDEGREES_OF_FREEDOM_
            IF (NSUBSYS(3)>NSUBSYS(2)) THEN
              VEL_tmp=0._q
              VEL_tmp(1:3,NSUBSYS(2)+1:)=DYN%VEL(1:3,NSUBSYS(2)+1:)
              CALL KINETIC_E(EKIN3,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                         & DYN%POTIM,LATT_CUR%A,VEL_tmp)                
              TEIN3 = 2*EKIN3/BOLKEV/NDOF(3)*SUM(NDOF)/NDEGREES_OF_FREEDOM_              
            END IF
          END IF

           
          IF (IO%IU6>0) THEN
            IF (NSUBSYS(2)>NSUBSYS(1)) THEN
              IF (NSUBSYS(3)>NSUBSYS(2)) THEN            
                write(g_io%REPORT,FMT='(/,2X,A32,X,I5,X,I5,X,I5)') '>Thermostat, num. of collisions:',random_counter(1),random_counter(2),random_counter(3)
                write(g_io%REPORT,FMT='(/,2X,A5,X,F10.3,X,F10.3,X,F10.3)') 'Tsub:',TEIN1,TEIN2,TEIN3
              ELSE
                write(g_io%REPORT,FMT='(/,2X,A32,X,I5,X,I5)') '>Thermostat, num. of collisions:',random_counter(1),random_counter(2)
                write(g_io%REPORT,FMT='(/,2X,A5,X,F10.3,X,F10.3)') 'Tsub:',TEIN1,TEIN2 
              ENDIF
            END IF 
          ENDIF

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE STEP_NANDERSEN

        SUBROUTINE STEP_HOOVER(DYN,ICOORDINATES,T_INFO,INFO,LATT_CUR, &
          &      EKIN,ESPRING,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT, &
          &      g_io,iconst0,iconst2,TEIN)
        !c Nose-Hover thermostat
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          REAL(q) :: UL,UT,SQQ,EKIN,EKIN_,EPS,SC,ES,DISMAX,TEMPER,ECONST
          INTEGER :: i,j,NDEGREES_OF_FREEDOM
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2
          REAL(q) :: POSION_(3,T_INFO%NIONS)  !,POSIOC_(3,T_INFO%NIONS)
          INTEGER :: NUMCONST
          REAL(q) :: C_FORCE(3,T_INFO%NIONS),VEL_(3,T_INFO%NIONS)
          REAL(q) :: EPOT,TEIN,ESPRING
          REAL(q) :: AMASS,C_FORCE_L(3,3)

          !c add this step to total count
          counter=counter+1
          !DYN%POSIOC=DYN%POSION
          UL =1E-10_q*LATT_CUR%ANORM(1)
          UT =DYN%POTIM*1E-15_q
          SQQ=DYN%SMASS*(AMTOKG/EVTOJ)*(UL/UT)**2
          
          AMASS=0._q

          IF (DYN%INIT==0) THEN
            DYN%SNOSE=0._q
          ELSE
            DYN%VEL=(1._q-DYN%SNOSE(2)/2._q)*DYN%VEL+DYN%D2C
          ENDIF
          VEL_=DYN%VEL
          EPS=0.5_q*(DYN%SNOSE(2)**2)*SQQ
          ES =NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP*(DYN%SNOSE(1))

          DYN%VEL=(DYN%VEL+DYN%D2C)/(1._q+DYN%SNOSE(2)/2._q)
          DYN%POSION=DYN%POSIOC+DYN%VEL
          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF
           
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,VEL_)
   
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM
          IF (IO%IU6>0) THEN
            CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,ESPRING,EPS,ES,counter)
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN)
          ENDIF         

          CALL KINETIC_E(EKIN_,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
                           
          DYN%SNOSE(3)=DYN%SNOSE(2)
          DYN%SNOSE(2)=DYN%SNOSE(2)+(2*EKIN_-NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP)/SQQ
          DYN%SNOSE(4)=DYN%SNOSE(1)
          DYN%SNOSE(1)=DYN%SNOSE(1)+(DYN%SNOSE(2)+DYN%SNOSE(3))/2._q

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
        END SUBROUTINE STEP_HOOVER

        SUBROUTINE HILLS_READER(IO,DYN,hills,g_io,ICOORDINATES,MDALGO)
        !c metadynamics - read input parameters
          TYPE(coordstructure) :: ICOORDINATES
          TYPE (in_struct) :: IO
          TYPE(dynamics) :: DYN
          TYPE(hills_data) :: hills
          TYPE(gadget_io) :: g_io
          INTEGER IDUM,i,j,K,N,IERR
          INTEGER :: MDALGO
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q) :: width,high,FACT,BMP,ti

          LOPEN=.FALSE.
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

          !c size of the bin to update the bias potential
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_BIN','=','#',';','I', &
                hills%bin,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) hills%bin=DYN%NSW
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            hills%bin=DYN%NSW
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''HILLS_BIN'' from file INCAR'
          ENDIF
          IF (hills%bin<1) hills%bin=DYN%NSW
          IF (hills%bin>DYN%NSW) hills%bin=DYN%NSW

          ALLOCATE(hills%gauss(DYN%NSW/hills%bin))
          
           !c stride 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_MAXSTRIDE','=','#',';','I', &
                hills%maxstride,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) hills%maxstride=1
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            hills%maxstride=1
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''HILLS_MAXSTRIDE'' from file INCAR'
          ENDIF
          IF (hills%maxstride<1) hills%maxstride=1

          !c 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_STRIDE','=','#',';','F', &
                IDUM,hills%stride,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) hills%stride=1.5
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            hills%stride=1.5
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''HILLS_H'' from file INCAR'
          ENDIF
          IF ( hills%stride .LE. 0._q) hills%stride=1.5


          !c high of the gauss peak in the bias potential (eV)
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_H','=','#',';','F', &
                IDUM,high,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) high=0.001
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            high=0.001
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''HILLS_H'' from file INCAR'
          ENDIF
          IF ( high<0._q) high=0.001
          
          !c width of the gauss peak in the bias potential 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_W','=','#',';','F', &
                IDUM,width,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) width=0.001
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            width=0.001
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''HILLS_W'' from file INCAR'
          ENDIF
          IF ( width .LE. 0._q) width=0.001
          
          DO i=1,DYN%NSW/hills%bin
            ALLOCATE(hills%gauss(i)%position(hills%number))
            hills%gauss(i)%position(:)=0._q
            hills%gauss(i)%high=high
            hills%gauss(i)%width=width
          ENDDO

          !c is variable width desired?
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_VARIABLE_W','=','#',';','L', &
                IDUM,RDUM,CDUM,hills%variable_width,CHARAC,N,1,IERR)
          IF (IERR==3) hills%variable_width=.FALSE.
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            hills%variable_width=.FALSE.
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''HILLS_VARIABLE_WIDTH'' from file INCAR'
          ENDIF



          !c only for mMD with fictituous particles
          IF ((MDALGO==10 .OR. MDALGO==20) .AND. (hills%number>0)) THEN
            !c force constant to couple the fictious and real particles 
            K=SIZE(hills%force_constant)
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_K','=','#',';','F', &
                IDUM,hills%force_constant,CDUM,LDUM,CHARAC,N,K,IERR)
            IF (IERR==3) hills%force_constant=0.5
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
              hills%force_constant=0.5
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_K'' from file INCAR'
            ENDIF
            DO i=1,K
              IF ( hills%force_constant(i)<0._q) hills%force_constant(i)=0.5
            ENDDO 

            !c force constant to couple the fictious and real particles 
            K=SIZE(hills%mass)
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_M','=','#',';','F', &
                  IDUM,hills%mass,CDUM,LDUM,CHARAC,N,K,IERR)
            IF (IERR==3) hills%mass=10.0
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
              hills%mass=10.0
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_M'' from file INCAR'
            ENDIF
            DO i=1,K
              IF ( hills%mass(i) .LE. 0._q) hills%mass(i)=10.0
            ENDDO

            !c temperature of the fictitious particles
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_TEMPERATURE','=','#',';','F', &
                IDUM,hills%temperature,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) hills%temperature=DYN%TEBEG
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
              hills%temperature=DYN%TEBEG
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_TEMPERATURE'' from file INCAR'
            ENDIF
            IF ( hills%temperature<=0._q) hills%temperature=DYN%TEBEG


            !c metapositions
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_POSITION','=','#',';','F', &
                IDUM,hills%position,CDUM,LDUM,CHARAC,N,hills%number,IERR)
            IF ((IERR==3) .OR. (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1)))) THEN
              j=0
              DO i =1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                  j=j+1
                  hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE
                ENDIF
              ENDDO
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_POSITION'' from file INCAR'
            ENDIF
         
            !c metavelocities
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_VELOCITY','=','#',';','F', &
                  IDUM,hills%velocity,CDUM,LDUM,CHARAC,N,hills%number,IERR)
            IF ((IERR==3) .OR. (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1)))) THEN
              FACT= (AMTOKG/EVTOJ)*(1E-10_q/(DYN%POTIM*1E-15_q))**2
              DO i=1, hills%number
                BMP=SQRT(hills%temperature*BOLKEV/(hills%mass(i)*FACT))
                hills%velocity(i)=boltzmann_distribution(0._q,BMP)
              ENDDO
              ti=0._q
              DO i=1, hills%number
                ti=ti+(hills%velocity(i)/DYN%POTIM)**2*hills%mass(i)
              ENDDO
              ti=ti*AMTOKG/BOLKEV/EVTOJ*1E10_q/hills%number
              hills%velocity(:)= hills%velocity(:)*SQRT(hills%temperature/ti)
              !IF (IO%IU0>=0) &
              !WRITE(IO%IU0,*)'Error reading item ''HILLS_VELOCITY'' from file INCAR'
            ENDIF

            !c probability of the collision for the andersen thermostat
            !c for the fictitious particles
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_ANDERSEN_PROB','=','#',';','F', &
                  IDUM,hills%andersen_prob,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) hills%andersen_prob=0._q
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
              hills%andersen_prob=0._q
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_ANDERSEN_PROB'' from file INCAR'
            ENDIF
            IF ( hills%andersen_prob<0._q) hills%andersen_prob=0._q

            !c mass for nose-hover for fictitious particles
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_SQQ','=','#',';','F', &
                  IDUM,hills%sqq,CDUM,LDUM,CHARAC,N,1,IERR)
            IF (IERR==3) hills%sqq=100._q
            IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
              hills%sqq=100._q
              !IF (IO%IU0>=0) &
              !  WRITE(IO%IU0,*)'Error reading item ''HILLS_SQQ'' from file INCAR'
            ENDIF
            IF ( hills%sqq<0._q) hills%sqq=100._q
          ENDIF

          !write (*,*) 'hills%mass',hills%mass
          IF (IO%IU6>0) THEN
            !c write the input parameters summary into OUTCAR
            write(IO%IU6,FMT='(3X,A22,X,I10)')   '          HILLS_BIN = ',  hills%bin
            write(IO%IU6,FMT='(3X,A22,X,F10.8)') '            HILLS_H = ',  high
            write(IO%IU6,FMT='(3X,A22,X,F10.8)') '            HILLS_W = ' , width
            write(IO%IU6,FMT='(3X,A22,X,I5)')    '    HILLS_MAXSTRIDE = ',  hills%maxstride            
            write(IO%IU6,FMT='(3X,A22,X,F10.8)') '       HILLS_STRIDE = ',  hills%stride
            write(IO%IU6,FMT='(3X,A22,X,L10)')   '   HILLS_VARIABLE_W = ' , hills%variable_width

            !c write the input parameters summary into REPORT 
            write(g_io%REPORT,FMT='(3X,A22,X,I10)')   '          HILLS_BIN = ',  hills%bin
            write(g_io%REPORT,FMT='(3X,A22,X,F10.8)') '            HILLS_H = ',  high
            write(g_io%REPORT,FMT='(3X,A22,X,F10.8)') '            HILLS_W = ' , width
            write(g_io%REPORT,FMT='(3X,A22,X,I5)')    '    HILLS_MAXSTRIDE = ',  hills%maxstride            
            write(g_io%REPORT,FMT='(3X,A22,X,F10.8)') '       HILLS_STRIDE = ',  hills%stride
            write(g_io%REPORT,FMT='(3X,A22,X,L10)')   '   HILLS_VARIABLE_W = ' , hills%variable_width

            !c some parameters for the mMD with fictitious variables
            IF ((MDALGO==10 .OR. MDALGO==20) .AND. (hills%number>0) ) THEN
              !c write the input parameters summary into OUTCAR
              write(IO%IU6,FMT='(3X,A22,X,F10.3)') '  HILLS_TEMPERATURE = ',  hills%temperature
              write(IO%IU6,FMT='(3X,A22,X,F10.8)') 'HILLS_ANDERSEN_PROB = ',  hills%andersen_prob

              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '            HILLS_K = '
              DO i=1,k
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') hills%force_constant(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')

              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '            HILLS_M = '
              DO i=1,k
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.6)') hills%mass(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')

              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '     HILLS_POSITION = '
              DO i=1,hills%number
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.6)') hills%position(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')

              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '     HILLS_VELOCITY = '
              DO i=1,hills%number
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.6)') hills%velocity(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')

              write(IO%IU6,FMT='(3X,A22,X,F10.3)')   '          HILLS_SQQ = ',  hills%sqq

              !c write the input parameters summary into REPORT 
              write(g_io%REPORT,FMT='(3X,A22,X,F10.3)') '  HILLS_TEMPERATURE = ',  hills%temperature
              write(g_io%REPORT,FMT='(3X,A22,X,F10.8)') 'HILLS_ANDERSEN_PROB = ',  hills%andersen_prob

              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '            HILLS_K = '
              DO i=1,k
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') hills%force_constant(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')

              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '            HILLS_M = '
              DO i=1,k
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.6)') hills%mass(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')

              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '     HILLS_POSITION = '
              DO i=1,hills%number
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.6)') hills%position(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')

              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '     HILLS_VELOCITY = '
              DO i=1,hills%number
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.6)') hills%velocity(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')

              write(g_io%REPORT,FMT='(3X,A22,X,F10.3)')   '          HILLS_SQQ = ',  hills%sqq
            ENDIF
          ENDIF
          CLOSE(IO%IU5)

        END SUBROUTINE HILLS_READER
        
        SUBROUTINE HARMONIC_SPRING_READER(IO,iconst8,g_io,spring_R0,spring_V0,spring_K,Lspring_V0)
        !c harmonic springs acting on internal coordinates (status = 8) - read in the input parameters
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          INTEGER IDUM,i,j,K,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          INTEGER :: iconst8
          REAL(q) :: spring_R0(iconst8),spring_V0(iconst8),spring_K(iconst8)
          LOGICAL :: LPROBLEM
          LOGICAL :: Lspring_V0
          
          IF (iconst8 .LE. 0) RETURN
        
          LPROBLEM=.FALSE.
        
          SPRING_R0=0._q
          SPRING_K=0._q
        
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
         
          !c force constant (in a.u.) 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SPRING_K','=','#',';','F', &
                IDUM,SPRING_K,CDUM,LDUM,CHARAC,N,iconst8,IERR)
          
          IF (IERR==3) THEN
            WRITE(IO%IU0,*)'Error: spring coordinates defined but the parameter ''SPRING_K'' not provided in INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst8))) THEN             
            WRITE(IO%IU0,*)'Error reading item ''SPRING_K'' from file INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          
          !c reference value (in A, rad., etc...)  
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SPRING_R0','=','#',';','F', &
                IDUM,SPRING_R0,CDUM,LDUM,CHARAC,N,iconst8,IERR)
          IF (IERR==3) THEN
            WRITE(IO%IU0,*)'Error: spring coordinates defined but the parameter ''SPRING_R0'' not provided in INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst8))) THEN             
            WRITE(IO%IU0,*)'Error reading item ''SPRING_R0'' from file INCAR'
            LPROBLEM=.TRUE.
          ENDIF
        
          !c velocity for R0 (in A/fs, rad./fs, etc...)  
          !c optional
          SPRING_V0=0._q
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SPRING_V0','=','#',';','F', &
                IDUM,SPRING_V0,CDUM,LDUM,CHARAC,N,iconst8,IERR)
          IF (IERR==3) THEN
            WRITE(IO%IU0,*)'Error: spring coordinates defined but the parameter ''SPRING_V0'' not provided in INCAR'
            !LPROBLEM=.TRUE.
          ELSE
            Lspring_V0=.TRUE.
          ENDIF
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst8))) THEN             
            WRITE(IO%IU0,*)'Error reading item ''SPRING_R0'' from file INCAR'
            LPROBLEM=.TRUE.
            Lspring_V0=.FALSE.
          ENDIF

          !c terminate if some important parameter was not read-in
          IF (LPROBLEM) STOP
          
          !write (*,*) 'hills%mass',hills%mass
          
          IF (IO%IU6>0) THEN
                    
            write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '           SPRING_K = '
            DO i=1,iconst8
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') SPRING_K(i)
            ENDDO
            write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '          SPRING_R0 = '
            DO i=1,iconst8
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') SPRING_R0(i)
            ENDDO
            write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            
            IF (Lspring_V0) THEN
              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '          SPRING_V0 = '
              DO i=1,iconst8
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') SPRING_V0(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            ENDIF 

            !c write the input parameters summary into OUTCAR
            write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '           SPRING_K = '
            DO i=1,iconst8
              write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') SPRING_K(i)
            ENDDO
            write(IO%IU6,ADVANCE='NO',FMT='(/)')
            
            write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '          SPRING_R0 = '
            DO i=1,iconst8
              write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') SPRING_R0(i)
            ENDDO
            write(IO%IU6,ADVANCE='NO',FMT='(/)')

            IF (Lspring_V0) THEN
              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '          SPRING_V0 = '
              DO i=1,iconst8
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') SPRING_V0(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')
            ENDIF
          ENDIF

          CLOSE(IO%IU5)
          
        END SUBROUTINE HARMONIC_SPRING_READER

        SUBROUTINE DUMMY_ATOM_READER(IO,iconst9,g_io,ICOORDINATES,MDALGO,DUMMY_R0,DUMMY_K,LDUMMYATOMS,DUMMY_POSITIONS,DUMMY_MASS)
        !c metadynamics - read input parameters
          TYPE(coordstructure) :: ICOORDINATES
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          INTEGER IDUM,i,j,K,N,IERR
          INTEGER :: MDALGO
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          INTEGER :: iconst9
          REAL(q) :: DUMMY_R0(iconst9),DUMMY_K(iconst9)
          LOGICAL :: LPROBLEM
          LOGICAL :: LDUMMYATOMS
          REAL(q) :: DUMMY_POSITIONS(3),DUMMY_MASS
          
          IF (iconst9 .LE. 0) RETURN
        
          LPROBLEM=.FALSE.
        
          DUMMY_R0=0._q
          DUMMY_K=0._q
        
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
         
          !c force constant (in a.u.) to couple the dummy atoms with the real system 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DUMMY_K','=','#',';','F', &
                IDUM,DUMMY_K,CDUM,LDUM,CHARAC,N,iconst9,IERR)
          
          IF (IERR==3) THEN
            WRITE(IO%IU0,*)'Error: spring coordinates defined but the parameter ''DUMMY_K'' not provided in INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst9))) THEN             
            WRITE(IO%IU0,*)'Error reading item ''DUMMY_K'' from file INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          
          !c reference value (in A, rad., etc...) to couple the dummy atoms with the real system 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DUMMY_R0','=','#',';','F', &
                IDUM,DUMMY_R0,CDUM,LDUM,CHARAC,N,iconst9,IERR)
          IF (IERR==3) THEN
            WRITE(IO%IU0,*)'Error: spring coordinates defined but the parameter ''DUMMY_R0'' not provided in INCAR'
            LPROBLEM=.TRUE.
          ENDIF
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<iconst9))) THEN             
            WRITE(IO%IU0,*)'Error reading item ''DUMMY_R0'' from file INCAR'
            LPROBLEM=.TRUE.
          ENDIF
        
          !c terminate if some important parameter was not read-in
          IF (LPROBLEM) STOP
          
          !write (*,*) 'hills%mass',hills%mass
          
          IF (IO%IU6>0) THEN
            IF (LDUMMYATOMS) THEN
              write(g_io%REPORT,FMT='(3X,A22,X,F10.6)') '         DUMMY_MASS = ',  DUMMY_MASS
              write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '    DUMMY_POSITIONS = '
              DO i=1,3
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_POSITIONS(i)
              ENDDO
              write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            ENDIF
                    
            write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '            DUMMY_K = '
            DO i=1,iconst9
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_K(i)
            ENDDO
            write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '           DUMMY_R0 = '
            DO i=1,iconst9
                write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_R0(i)
            ENDDO
            write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
            
            !c write the input parameters summary into OUTCAR
            IF (LDUMMYATOMS) THEN
              write(IO%IU6,FMT='(3X,A22,X,F10.6)') '         DUMMY_MASS = ',  DUMMY_MASS
              write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '    DUMMY_POSITIONS = '
              DO i=1,3
                write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_POSITIONS(i)
              ENDDO
              write(IO%IU6,ADVANCE='NO',FMT='(/)')
            ENDIF
            write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '            DUMMY_K = '
            DO i=1,k
              write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_K(i)
            ENDDO
            write(IO%IU6,ADVANCE='NO',FMT='(/)')
            
            write(IO%IU6,ADVANCE='NO',FMT='(3X,A22)') '           DUMMY_R0 = '
            DO i=1,k
              write(IO%IU6,ADVANCE='NO',FMT='(X,F10.5)') DUMMY_R0(i)
            ENDDO
            write(IO%IU6,ADVANCE='NO',FMT='(/)')
          ENDIF

          CLOSE(IO%IU5)
          
        END SUBROUTINE DUMMY_ATOM_READER

        SUBROUTINE PENALTY_READER(hills,penalty,IO,g_io)
        !c metadynamics - read existing bias potential and rigid walls 
          TYPE(penalty_data) :: penalty
          TYPE(hills_data) :: hills
          TYPE(gadget_io) :: g_io
          TYPE(gauss_peak),POINTER :: reallocate_(:)
          TYPE (in_struct) :: IO
          INTEGER :: counter,ios,i,mold
          INTEGER IDUM,K,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          CHARACTER(LEN=2000) :: line
          
          ALLOCATE(penalty%wall(hills%number,2))
          penalty%wall(:,1)=-1000.0
          penalty%wall(:,2)=1000.0

          LOPEN=.FALSE.
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

          !c 
          K=hills%number
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_WALL_LOWER','=','#',';','F', &
                IDUM,penalty%wall(:,1),CDUM,LDUM,CHARAC,N,K,IERR)
          IF (IERR==3) penalty%wall(:,1)=-1000.0
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            penalty%wall(:,1)=-1000.0
            !IF (IO%IU0>=0) &
            !  WRITE(IO%IU0,*)'Error reading item ''HILLS_WALL_LOWER'' from file INCAR'
          ENDIF
         
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'HILLS_WALL_UPPER','=','#',';','F', &
                IDUM,penalty%wall(:,2),CDUM,LDUM,CHARAC,N,K,IERR)
          IF (IERR==3) penalty%wall(:,2)=1000.0
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            penalty%wall(:,2)=1000.0
            !IF (IO%IU0>=0) &
            !  WRITE(IO%IU0,*)'Error reading item ''HILLS_WALL_UPPER'' from file INCAR'
          ENDIF
          CLOSE(IO%IU5)

          
          !IF (IO%IU6>0) THEN
          !write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '   HILLS_WALL_LOWER = '
          !DO i=1,hills%number
          !  write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') penalty%wall(i,1)
          !ENDDO
          !write(g_io%REPORT,ADVANCE='NO',FMT='(/)')

          !write(g_io%REPORT,ADVANCE='NO',FMT='(3X,A22)') '   HILLS_WALL_UPPER = '
          !DO i=1,hills%number
          !  write(g_io%REPORT,ADVANCE='NO',FMT='(X,F10.5)') penalty%wall(i,2)
          !ENDDO
          !write(g_io%REPORT,ADVANCE='NO',FMT='(/)')
          !ENDIF

          ALLOCATE(penalty%force(hills%number))
          penalty%force=0._q

          !OPEN(UNIT=g_io%PENALTY,FILE=DIR_APP(1:DIR_LEN)//'PENALTYPOT',STATUS='UNKNOWN')
          OPEN(UNIT=g_io%PENALTY,FILE='PENALTYPOT',STATUS='UNKNOWN')
          
          penalty%number=10
          ALLOCATE(penalty%gauss(penalty%number))
          DO i=1,penalty%number
            ALLOCATE(penalty%gauss(i)%position(hills%number))
            penalty%gauss(i)%position(:)=0._q
          ENDDO

          counter=1
          DO
            IF (counter .GT. penalty%number) THEN
              mold=penalty%number
              penalty%number=penalty%number+10
              ALLOCATE(reallocate_(penalty%number))
              IF (.NOT. ASSOCIATED(penalty%gauss)) RETURN
              reallocate_(1:mold)=penalty%gauss(1:mold)
              DEALLOCATE(penalty%gauss)
              penalty%gauss=>reallocate_
              DO i=mold+1,penalty%number
                ALLOCATE(penalty%gauss(i)%position(hills%number))
                penalty%gauss(i)%position(:)=0._q
              ENDDO
              penalty%gauss(mold+1:)%high=0._q;penalty%gauss(mold+1:)%width=0._q 
            ENDIF

            !READ(g_io%PENALTY,FMT=*,IOSTAT=ios) penalty%gauss(counter)%position(:),&
            !& penalty%gauss(counter)%high,penalty%gauss(counter)%width
            !IF (ios/=0) THEN 
            !  counter=counter-1
            !  EXIT
            !ENDIF

            READ(g_io%PENALTY,FMT='(A2000)',IOSTAT=ios) line
            IF (ios/=0) THEN 
              counter=counter-1
              EXIT
            ENDIF
            IF (LEN_TRIM(line)>0) THEN
              line=TRIM(ADJUSTL(line))
              READ(line,FMT=*,IOSTAT=ios) penalty%gauss(counter)%position(:),&
              & penalty%gauss(counter)%high,penalty%gauss(counter)%width
              IF (ios/=0) THEN 
                IF (IO%IU0>=0) write(IO%IU0,*) 'Error reading PENALTYPOT (item',counter,')'
                STOP
              ELSE
                IF (penalty%gauss(counter)%width <= 0._q) THEN
                  IF (IO%IU0>=0) write(IO%IU0,*) 'Error reading PENALTYPOT (item',counter,'): W must be a positive number'
                  STOP
                ENDIF
                counter=counter+1
              ENDIF
            ENDIF
          ENDDO

          penalty%number=counter
          ALLOCATE(reallocate_(counter))
          IF (.NOT. ASSOCIATED(penalty%gauss)) RETURN
          reallocate_(1:counter)=penalty%gauss(1:counter)
          DEALLOCATE(penalty%gauss)
          penalty%gauss=>reallocate_
              
          CLOSE(g_io%PENALTY)
        END SUBROUTINE PENALTY_READER
 

        SUBROUTINE HILLS_METHOD(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,g_io,& 
          &      WDES,iconst0,iconst2,TEIN)
        !c NVT metadynamics (Andersen thermostat) with fictitious coordinates 
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(hills_data) :: hills
          TYPE (wavedes)  ::   WDES
          TYPE(penalty_data) :: penalty
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ti,ti_old,T_old,T_new,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,random_counter
          INTEGER,SAVE :: counter=0        
          INTEGER :: iconst0,iconst2
          REAL(q) :: dummy,FACT,BMP,ANDERSEN_PROB
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: spring_accel(3,T_INFO%NIONS)
          REAL(q),SAVE,ALLOCATABLE :: spring_forceS(:)
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          !LOGICAL :: LSCALE,LCMASS
          REAL(q) :: TOTAL_MASS
          INTEGER,SAVE :: Ltxyz(3)

          !LSCALE=.FALSE.
          !LCMASS=.TRUE.


          !c add this step to total count
          counter=counter+1

          !c determine if coordinates for some direction
          !c are frozen
          IF (counter==1) THEN
            Ltxyz(:)=1
            DO j=1,3
              DO i=1,T_INFO%NIONS
                IF (.NOT. T_INFO%LSFOR(j,i)) Ltxyz(j)=0
              ENDDO
            ENDDO

            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="X ") Ltxyz(1)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Y ") Ltxyz(2)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Z ") Ltxyz(3)=0
              ENDIF
            ENDDO
          ENDIF
         
          AMASS=0._q

          IF (counter==1) THEN
            ALLOCATE(spring_forceS(hills%number))            
            spring_forceS=0._q
            j=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              !IF (penalty%number==0) THEN
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                  j=j+1
                 !!! hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE
                  IF (hills%position(j)<=(penalty%wall(j,1)+hills%gauss(1)%width)) THEN
                    hills%position(j)=hills%position(j)+2*hills%gauss(1)%width
                  ENDIF
                  IF (hills%position(j)>=(penalty%wall(j,2)-hills%gauss(1)%width)) THEN
                    hills%position(j)=hills%position(j)-2*hills%gauss(1)%width
                  ENDIF
                ENDIF
            ENDDO
          ENDIF 

          !c calculate additional potentaials (spring, bias, and penalty)
          CALL hills_spring_force(T_INFO,LATT_CUR,DYN,hills,ICOORDINATES, DYN%POSIOC,spring_forceS,spring_accel)

          IF (counter .GE. hills%bin) THEN
            CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
          ENDIF 

          CALL penalty_potential(penalty,hills,ICOORDINATES)

          !c real variables +  andersen thermostat
          VEL_tmp=0._q
          !CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM,LSCALE,LCMASS)
            
          ! Add a random velicity that is exactly
          ! distributed like the centre of mass velocity
          TOTAL_MASS = 0.0_q
          DO I = 1, T_INFO%NIONS
            TOTAL_MASS = TOTAL_MASS + T_INFO%POMASS(T_INFO%ITYP(I))
          END DO
          CMASS(:,1) = BOLTZMANN_VELOCITY(DYN, LATT_CUR, TOTAL_MASS)

          !c don't add translation for frozen direction
          DO i=1,3
            IF (Ltxyz(i)==0) CMASS(i,1)=0._q
          ENDDO

          DO i=1,T_INFO%NIONS
            DO j=1,3
              DYN%VEL(j,i)=DYN%VEL(j,i)+CMASS(j,1)
            ENDDO
          ENDDO

!          CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM_)
          CALL boltzmann_velocities(T_INFO,DYN,LATT_CUR,VEL_tmp)

          DYN%VEL=DYN%VEL+2*(DYN%D2C-spring_accel)
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          T_old = 2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          CMASS=0._q
          random_counter=0
          IF (DYN%INIT==1) THEN
            DO i=1,T_INFO%NIONS
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                DYN%VEL(:,i)=VEL_tmp(:,i)
              ENDIF
            ENDDO
          ENDIF

          CMASS=0._q
          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)
          DO i=1,T_INFO%NIONS
            DO j=1,3
              DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
          ENDDO
         
          !c velocity for the step t-1/2dt
          VEL_tmp=DYN%VEL


          DYN%POSION=DYN%POSIOC+DYN%VEL
          ECONST=0._q
          IF (iconst0>0 .OR. iconst2>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES, &
            &  g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          !!c velocity for the step t
          !!VEL_tmp=(VEL_tmp+DYN%VEL)/2
          !!CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
          !!             & DYN%POTIM,LATT_CUR%A,VEL_tmp)

          !c due to compatibility reasons we still compute
          !c kinetic energy and T using velocities from the step t+1/2
          !c but in the future we should use VEL_tmp (velocity for step t) instead
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          IF (IO%IU6>0) CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)


          !c fictitious variables
          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          hills%velocity=hills%velocity+FACT*(spring_forceS-hills%force-penalty%force)/hills%mass

          !c current temperature of fict. particles
          ti=0._q
          DO i=1, hills%number
            ti=ti+(hills%velocity(i)/DYN%POTIM)**2*hills%mass(i)
          ENDDO
          ti=ti*AMTOKG/BOLKEV/EVTOJ*1E10_q/hills%number
          ti_old=ti

          !c apply Andersen thermostat for fictitious particles
          FACT= (AMTOKG/EVTOJ)*(1E-10_q/(DYN%POTIM*1E-15_q))**2
          DO i=1, hills%number
            CALL RANE_ION(dummy)
            IF (dummy<hills%andersen_prob) THEN
              BMP=SQRT(hills%temperature*BOLKEV/(hills%mass(i)*FACT))
              hills%velocity(i)=boltzmann_distribution(0._q,BMP)
            ENDIF
          ENDDO

          !c new temperature of fict. particles
          ti=0._q
          DO i=1, hills%number
            ti=ti+(hills%velocity(i)/DYN%POTIM)**2*hills%mass(i)
          ENDDO
          ti=ti*AMTOKG/BOLKEV/EVTOJ*1E10_q/hills%number

          hills%position=hills%position+hills%velocity

          !c reflecting walls
          DO i=1,hills%number
            IF (hills%position(i)<=penalty%wall(i,1)) THEN
              hills%position(i)=penalty%wall(i,1)+(penalty%wall(i,1)-hills%position(i))
              hills%velocity(i)=-hills%velocity(i)
            ENDIF
            IF (hills%position(i)>=penalty%wall(i,2)) THEN
              hills%position(i)=penalty%wall(i,2)+(penalty%wall(i,2)-hills%position(i))
              hills%velocity(i)=-hills%velocity(i)
            ENDIF
          ENDDO

#ifdef MPI
            CALL  M_bcast_d(WDES%COMM, hills%position, hills%number)
#endif

          IF (hills%number>0) THEN
            IF (IO%IU6>0) THEN
              j=0
              write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
              write(g_io%REPORT,ADVANCE='YES',FMT='(10X,A9,X,A9)') &
                  & "S(q)","s"
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                  j=j+1                
                  write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5,X,F9.5)') &
                  & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE,hills%position(j)
                ENDIF
              ENDDO     
            ENDIF
          ENDIF

          IF (IO%IU6>0) THEN
            write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A12)') '>Temperature'
            write(g_io%REPORT,ADVANCE='YES',FMT='(10X,A12,X,A12,X,A12,X,A12)') &
                & "T_sim","T_inst","t_sim","t_inst"
            write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F12.5,X,F12.5,X,F12.5,X,F12.5)') &
                & 'tmprt>', DYN%TEMP,T_old,hills%temperature,ti_old
            !write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F12.5,X,F12.5,X,F12.5,X,F12.5)') &
            !    & 'tmprt>',T_old,T_new,ti_old,ti
             
          ENDIF

          IF (IO%IU6>0) THEN
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter
          ENDIF

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE HILLS_METHOD

        SUBROUTINE HILLS_METHOD_nose(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,g_io, & 
          &      WDES,iconst0,iconst2,TEIN)
        !c NVT metadynamics (Nose-Hover thermostat) with fictitious coordinates 
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(hills_data) :: hills
          TYPE (wavedes)  ::   WDES
          TYPE(penalty_data) :: penalty
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,ti,ti_old,T_old,T_new,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,random_counter
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2        
          REAL(q) :: dummy,FACT,BMP,ANDERSEN_PROB
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: spring_accel(3,T_INFO%NIONS)
          REAL(q),SAVE,ALLOCATABLE :: spring_forceS(:)
          REAL    :: SQQ,hEKIN,UT,UL
          REAL(q) :: AMASS,C_FORCE_L(3,3)

          counter=counter+1
          UL =1E-10_q*LATT_CUR%ANORM(1)
          UT =DYN%POTIM*1E-15_q
          SQQ=DYN%SMASS*(AMTOKG/EVTOJ)*(UL/UT)**2

          AMASS=0._q

          IF (DYN%INIT==0) THEN
             DYN%SNOSE=0._q
          ENDIF

          IF (counter==1) THEN
            hills%SNOSE=0.0
            ALLOCATE(spring_forceS(hills%number))
            spring_forceS=0._q
 
            j=0
            DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                  j=j+1
                  !!!hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE
                  IF (hills%position(j)<=(penalty%wall(j,1)+hills%gauss(1)%width)) THEN
                    hills%position(j)=hills%position(j)+2*hills%gauss(1)%width
                  ENDIF
                  IF (hills%position(j)>=(penalty%wall(j,2)-hills%gauss(1)%width)) THEN
                    hills%position(j)=hills%position(j)-2*hills%gauss(1)%width
                  ENDIF
                ENDIF
            ENDDO
          ENDIF 
          
          !c calculate additional potentaials (spring, bias, and penalty)
          CALL hills_spring_force(T_INFO,LATT_CUR,DYN,hills,ICOORDINATES,DYN%POSIOC,spring_forceS,spring_accel)
          IF (counter .GE. hills%bin) THEN
            CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
          ENDIF 
          CALL penalty_potential(penalty,hills,ICOORDINATES)

          !c real variables +  nose-hoover thermostat
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          T_old = 2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM)
          
          IF (counter>1) DYN%VEL=(1._q-DYN%SNOSE(2)/2._q)*DYN%VEL+DYN%D2C-spring_accel
          EPS=0.5_q*(DYN%SNOSE(2)**2)*SQQ
          ES =NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP*(DYN%SNOSE(1))
          DYN%VEL=(DYN%VEL+DYN%D2C-spring_accel)/(1._q+DYN%SNOSE(2)/2._q)
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          T_new = 2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM)
          TEIN=T_new
          DYN%SNOSE(3)=DYN%SNOSE(2)
          DYN%SNOSE(2)=DYN%SNOSE(2)+(2*EKIN-NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP)/SQQ
          DYN%SNOSE(4)=DYN%SNOSE(1)
          DYN%SNOSE(1)=DYN%SNOSE(1)+(DYN%SNOSE(2)+DYN%SNOSE(3))/2._q

          !write(*,*) 'spring_accel',spring_accel
          DYN%POSION=DYN%POSIOC+DYN%VEL
          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,&
            &  g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          IF (IO%IU6>0) CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,EPS,ES,counter)
         
          !c fictitious variables
          !c current temperature of fict. particles
          ti=0._q
          DO i=1, hills%number
            ti=ti+(hills%velocity(i))**2*hills%mass(i)
          ENDDO
          ti=ti*AMTOKG/BOLKEV/EVTOJ*1E10_q/hills%number/(DYN%POTIM**2)
          ti_old=ti

          UL=1E-10_q
          UT=DYN%POTIM*1E-15_q
          IF (hills%andersen_prob > 1E-5_q) THEN
            !c apply Andersen thermostat for fictitious particles
            FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
            hills%velocity=hills%velocity+FACT*(spring_forceS-hills%force-penalty%force)/hills%mass
            FACT= (AMTOKG/EVTOJ)*(1E-10_q/(DYN%POTIM*1E-15_q))**2
            DO i=1, hills%number
              CALL RANE_ION(dummy)
              IF (dummy<hills%andersen_prob) THEN
                BMP=SQRT(hills%temperature*BOLKEV/(hills%mass(i)*FACT))
                hills%velocity(i)=boltzmann_distribution(0._q,BMP)
              ENDIF
            ENDDO
            hills%position=hills%position+hills%velocity
          ELSE
            !c apply Nose-Hover thermostat for fictitious particles
            FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
            IF (counter>1)  hills%velocity=(1._q-hills%SNOSE(2)/2._q)*hills%velocity+&
                                       FACT*(spring_forceS-hills%force-penalty%force)/hills%mass/2.0

            hills%velocity=(hills%velocity+FACT*(spring_forceS-hills%force-penalty%force)/hills%mass/2.0)/&
                           (1._q+hills%SNOSE(2)/2._q)
            hills%position=hills%position+hills%velocity

            hEKIN=0._q
            DO i=1, hills%number
              hEKIN=hEKIN+hills%velocity(i)**2*hills%mass(i)
            ENDDO
          
            !!!!!!!!!!!!!:check:!!!!!!!!1 hEKIN=hEKIN*AMTOKG/EVTOJ*(UL/UT)**2
            hEKIN=0.5*hEKIN*AMTOKG/EVTOJ*(UL/UT)**2

            FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
            !FACT=EVTOJ/AMTOKG *UT**2/UL
            hills%SNOSE(3)=hills%SNOSE(2)
            hills%SNOSE(2)=hills%SNOSE(2)+FACT*(2*hEKIN-hills%number*BOLKEV*hills%temperature)/(hills%SQQ)
            !!!hills%SNOSE(2)=hills%SNOSE(2)+(2*hEKIN-hills%number*BOLKEV*hills%temperature)/(hills%SQQ)
            hills%SNOSE(4)=hills%SNOSE(1)
            hills%SNOSE(1)=hills%SNOSE(1)+(hills%SNOSE(2)+hills%SNOSE(3))/2._q 
          ENDIF

          !c new temperature of fict. particles
          ti=0._q
          DO i=1, hills%number
            ti=ti+(hills%velocity(i))**2*hills%mass(i)
          ENDDO
          ti=ti*AMTOKG/BOLKEV/EVTOJ*1E10_q/hills%number/(DYN%POTIM**2)
          

          !c reflecting walls
          DO i=1,hills%number
            IF (hills%position(i)<=penalty%wall(i,1)) THEN
              hills%position(i)=penalty%wall(i,1)+(penalty%wall(i,1)-hills%position(i))
              hills%velocity(i)=-hills%velocity(i)
            ENDIF
            IF (hills%position(i)>=penalty%wall(i,2)) THEN
              hills%position(i)=penalty%wall(i,2)+(penalty%wall(i,2)-hills%position(i))
              hills%velocity(i)=-hills%velocity(i)
            ENDIF
          ENDDO


#ifdef MPI
            CALL  M_bcast_d(WDES%COMM, hills%position, hills%number)
#endif

         IF (hills%number>0) THEN
            IF (IO%IU6>0) THEN
              j=0
              write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
              write(g_io%REPORT,ADVANCE='YES',FMT='(10X,A9,X,A9)') &
                  & "S(q)","s"
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
                  j=j+1                
                  write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5,X,F9.5)') &
                  & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE,hills%position(j)
                ENDIF
              ENDDO     
            ENDIF
          ENDIF

          IF (IO%IU6>0) THEN
            write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A12)') '>Temperature'
            write(g_io%REPORT,ADVANCE='YES',FMT='(10X,A12,X,A12,X,A12,X,A12)') &
                & "T_sim","T_inst","t_sim","t_inst"
            write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F12.5,X,F12.5,X,F12.5,X,F12.5)') &
                & 'tmprt>', DYN%TEMP,T_old,hills%temperature,ti_old
            !write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F12.5,X,F12.5,X,F12.5,X,F12.5)') &
            !    & 'tmprt>',T_old,T_new,ti_old,ti
             
          ENDIF

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
        END SUBROUTINE HILLS_METHOD_nose

       SUBROUTINE dummy_spring_force(iconst9,T_INFO,LATT_CUR,DYN,DUMMY_R0,DUMMY_K,ICOORDINATES,&
       &  REF_C,spring_accel,spring_energy)
          !c calculates the spring force necessary to 
          !c keep values of chosen geometric parameters
          !c close to those for collective variables
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(dynamics) :: DYN 
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: iconst9
          INTEGER :: i,j,NI,NT
          REAL(q) :: DUMMY_R0(iconst9),DUMMY_K(iconst9)
          REAL(q) :: REF_C(3,T_INFO%NIONS)
          REAL(q) :: FACT,deltaS
          REAL(q) :: s_force(3*T_INFO%NIONS)
          REAL(q) :: spring_force(3,T_INFO%NIONS),spring_accel(3,T_INFO%NIONS)
          REAL(q) :: dummy(3,T_INFO%NIONS)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS)
          REAL(q) :: spring_energy

          
          BMAT=0._q
          !CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.FALSE.)
          CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.TRUE.)
          !CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)
          spring_force=0._q
          spring_accel=0._q
          spring_energy=0._q
          
          j=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==9) THEN
              j=j+1
              deltaS=ICOORDINATES%COORDSTRUCT(i)%VALUE-DUMMY_R0(j)
              !IF (ICOORDINATES%COORDSTRUCT(i)%TAG=='A ' .OR. &
              !  ICOORDINATES%COORDSTRUCT(i)%TAG=='T ') THEN
              IF (ICOORDINATES%COORDSTRUCT(i)%TAG=='T ') THEN  
                DO
                  IF (deltaS>pi) THEN
                    deltaS=2*pi-deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
                DO
                  IF (deltaS<=-pi) THEN
                    deltaS=2*pi+deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
              ENDIF
              !spring_forceS(j)=DUMMY_K(j)*deltaS
              !s_force=spring_forceS(j)*BMAT(i,:)
              spring_energy=spring_energy+0.5*DUMMY_K(j)*deltaS*deltaS
              s_force=DUMMY_K(j)*deltaS*BMAT(i,:)
              dummy=0._q
              CALL ONETOTHREE(T_INFO%NIONS,dummy,s_force)
              !dummy=MATMUL(LATT_CUR%B,dummy) !!!!!!!!!
              spring_force=spring_force+dummy
            ENDIF
          ENDDO
          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          NI=1
          DO NT=1,T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
!               write(*,*) 'NI',NI
!               write(*,*) 'NT',NT
!               write(*,*) "T_INFO%LSFOR(:,NI)",T_INFO%LSFOR(:,NI)
!               write(*,*) "T_INFO%POMASS(NT)",T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(1,NI)) spring_accel(1,NI)=spring_force(1,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(2,NI)) spring_accel(2,NI)=spring_force(2,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(3,NI)) spring_accel(3,NI)=spring_force(3,NI)*FACT/2/T_INFO%POMASS(NT)
            ENDDO 
          ENDDO

          !!spring_accel=MATMUL(LATT_CUR%B,spring_accel) !!!!!!!!!
          spring_accel=MATMUL(TRANSPOSE(LATT_CUR%B),spring_accel)
          !write(*,*) 'spring_accel',spring_accel
        END SUBROUTINE dummy_spring_force        
        
        SUBROUTINE hills_spring_force(T_INFO,LATT_CUR,DYN,hills,ICOORDINATES,REF_C,spring_forceS,spring_accel)
          !c metadynamics with fictitious coordinates - calculates the spring force necessary to 
          !c keep values of chosen geometric parameters
          !c close to those for collective variables
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(dynamics) :: DYN 
          TYPE(hills_data) :: hills
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: i,j,NI,NT
          REAL(q) :: REF_C(3,T_INFO%NIONS)
          REAL(q) :: FACT,deltaS
          REAL(q) :: s_force(3*T_INFO%NIONS)
          REAL(q) :: spring_force(3,T_INFO%NIONS),spring_accel(3,T_INFO%NIONS)
          REAL(q) :: spring_forceS(:)
          REAL(q) :: dummy(3,T_INFO%NIONS)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS)
          !REAL(q),ALLOCATABLE :: BMAT(:,:)


          !ALLOCATE(BMAT(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS))
          BMAT=0._q
          CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.FALSE.)
          !CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)
          spring_force=0._q
          spring_accel=0._q
          
          j=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==6) THEN
              j=j+1
              deltaS=ICOORDINATES%COORDSTRUCT(i)%VALUE-hills%position(j)
              IF (ICOORDINATES%COORDSTRUCT(i)%TAG=='A ' .OR. &
                ICOORDINATES%COORDSTRUCT(i)%TAG=='T ') THEN
                DO
                  IF (deltaS>pi) THEN
                    deltaS=2*pi-deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
                DO
                  IF (deltaS<=-pi) THEN
                    deltaS=2*pi+deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
              ENDIF
              spring_forceS(j)=hills%force_constant(j)*deltaS
              s_force=spring_forceS(j)*BMAT(i,:)
              dummy=0._q
              CALL ONETOTHREE(T_INFO%NIONS,dummy,s_force)
              dummy=MATMUL(LATT_CUR%B,dummy) !!!!!!!!!
              spring_force=spring_force+dummy
            ENDIF
          ENDDO
          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          NI=1
          DO NT=1,T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
              IF (T_INFO%LSFOR(1,NI)) spring_accel(1,NI)=spring_force(1,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(2,NI)) spring_accel(2,NI)=spring_force(2,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(3,NI)) spring_accel(3,NI)=spring_force(3,NI)*FACT/2/T_INFO%POMASS(NT)
            ENDDO 
          ENDDO

          !!spring_accel=MATMUL(LATT_CUR%B,spring_accel) !!!!!!!!!
          spring_accel=MATMUL(TRANSPOSE(LATT_CUR%B),spring_accel)
          !DEALLOCATE(BMAT)
          !write(*,*) 'spring_forceS',spring_forceS
          !write(*,*) 'spring_accel',spring_accel
        END SUBROUTINE hills_spring_force

        SUBROUTINE hills_bias_potential(hills,penalty,ICOORDINATES,DYN,step,g_io,IO)
          !c here we calculate the history-dependent bias potential and forces
          TYPE(hills_data) :: hills
          TYPE(penalty_data) :: penalty
          TYPE(gadget_io) :: g_io
          TYPE(coordstructure) :: ICOORDINATES
          TYPE(dynamics) :: DYN 
          TYPE (in_struct) :: IO
          REAL(q) :: snorm,w1,w2
          INTEGER,SAVE :: counter=0            !c meta-time
          INTEGER  ::step
          INTEGER :: i,j,k,dmn1,dmn2
          INTEGER,SAVE :: last_upgrade=0
          REAL(q) :: dummy,ti

          IF (step<hills%bin .OR. hills%number<1) RETURN
          !OPEN(UNIT=g_io%STRUCTINPUT,FILE=DIR_APP(1:DIR_LEN)//'HILLSPOT',STATUS='UNKNOWN',POSITION='APPEND')
          OPEN(UNIT=g_io%STRUCTINPUT,FILE='HILLSPOT',STATUS='UNKNOWN',POSITION='APPEND')
          
          IF (MOD(step,hills%bin)==0) THEN

            IF (counter==0) THEN
              !c add another hill
              last_upgrade=step
              counter=counter+1
              hills%gauss(counter)%position(:)=hills%position(:)
              DO j=1,hills%number
                IF (g_io%STRUCTINPUT>=0 .AND. (IO%IU0>0)) THEN
                  write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') hills%gauss(counter)%position(j)
                ENDIF
              ENDDO

              IF ((g_io%STRUCTINPUT>=0) .AND. (IO%IU0>0)) THEN
                write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') hills%gauss(counter)%high
                write(g_io%STRUCTINPUT,ADVANCE='YES',FMT='(X,F9.5)') hills%gauss(counter)%width
              ENDIF

            ELSE IF ((SQRT(SUM((hills%gauss(counter)%position(:)-hills%position(:))**2))>& 
            &    hills%stride*hills%gauss(counter)%width) & 
            &    .OR. (MOD((step-last_upgrade),hills%maxstride*hills%bin)==0)) THEN
              !c add another hill
              last_upgrade=step
              counter=counter+1
              hills%gauss(counter)%position(:)=hills%position(:)
              DO j=1,hills%number
                IF (g_io%STRUCTINPUT>=0 .AND. (IO%IU0>0)) THEN
                  write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') hills%gauss(counter)%position(j)
                ENDIF
              ENDDO
              
              !c if variable width is desired  - similarly as in the original metadynamics
              !c implementation
              IF (hills%variable_width) THEN
                w1=(hills%gauss(counter)%width)**2
                w2=SUM((hills%gauss(counter)%position(:)-hills%gauss(counter-1)%position(:))**2)
                hills%gauss(counter)%width=SQRT(w1*w2/(w1+w2))
              ENDIF

              IF ((g_io%STRUCTINPUT>=0) .AND. (IO%IU0>0)) THEN
                write(g_io%STRUCTINPUT,ADVANCE='NO',FMT='(X,F9.5)') hills%gauss(counter)%high
                write(g_io%STRUCTINPUT,ADVANCE='YES',FMT='(X,F9.5)') hills%gauss(counter)%width
              ENDIF
            
            ENDIF 
          ENDIF
          
          hills%potential=0._q
          hills%force=0._q
          DO i=1,counter   !c sum over all bins
            snorm=0._q
            !c sum up the potential for the current bin
            k=0
            DO j=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(j)%STATUS==6) THEN
                k=k+1
                IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5) snorm=snorm+(hills%position(k)-hills%gauss(i)%position(k))**2
              ENDIF
            ENDDO

            !c sum up the forces for the current bin 
            k=0
            DO j=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(j)%STATUS==6) THEN
                k=k+1
                IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5) THEN
                  hills%force(k)=hills%force(k) &
                  & -(hills%position(k)-hills%gauss(i)%position(k))*hills%gauss(i)%high/(hills%gauss(i)%width**2)* &
                  & exp(-snorm/2._q/hills%gauss(i)%width**2)
                ENDIF
              ENDIF
            ENDDO
          ENDDO
          CLOSE(g_io%STRUCTINPUT)
        END SUBROUTINE hills_bias_potential

        SUBROUTINE hills_bias_direct(T_INFO,LATT_CUR,DYN,internal_force,ICOORDINATES,REF_C,penalty_accel,penalty_accel_L,IBDIM,AMASS)
          !c bias potential for metadynamics without fictitious particles
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(dynamics) :: DYN 
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: i,j,k,NI,NT
          INTEGER :: IBDIM
          REAL(q) :: REF_C(3,T_INFO%NIONS)
          REAL(q) :: FACT
          REAL(q) :: internal_force(:)
          !REAL(q) :: hspring_force(:)
          REAL(q) :: p_force(3*T_INFO%NIONS)
          REAL(q) :: p_force_L(9)
          REAL(q) :: penalty_force(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS)
          REAL(q) :: penalty_force_L(3,3),penalty_accel_L(3,3)
          REAL(q) :: dummy(3,T_INFO%NIONS),dummy_L(3,3)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: AMASS !c mass for lattice DOFs

          BMAT=0._q
          CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.FALSE.)
          !CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)

          penalty_force=0._q
          penalty_accel=0._q

          penalty_force_L=0._q
          penalty_accel_L=0._q

          j=0;k=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
              j=j+1
              p_force=internal_force(j)*BMAT(i,1:3*T_INFO%NIONS)
              dummy=0._q
              CALL ONETOTHREE(T_INFO%NIONS,dummy,p_force)
              dummy=MATMUL(LATT_CUR%B,dummy) !!!!!!!!!
              penalty_force=penalty_force+dummy

              !c lattice dynamics
              IF (IBDIM==3*T_INFO%NIONS+9) THEN
                p_force_L=internal_force(j)*BMAT(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
                dummy_L=0._q
                CALL ONETOTHREE(3,dummy_L,p_force_L)
                dummy_L=TRANSPOSE(dummy_L)
                penalty_force_L=penalty_force_L+dummy_L
              ENDIF

            ENDIF
           
!             !c harmonic spring contribution 
!             IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==8) THEN
!               k=k+1
!               p_force=hspring_force(k)*BMAT(i,1:3*T_INFO%NIONS)
!               dummy=0._q
!               CALL ONETOTHREE(T_INFO%NIONS,dummy,p_force)
!               dummy=MATMUL(LATT_CUR%B,dummy) !!!!!!!!!
!               penalty_force=penalty_force+dummy
! 
!               !c lattice dynamics
!               IF (IBDIM==3*T_INFO%NIONS+9) THEN
!                 p_force_L=hspring_force(k)*BMAT(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
!                 dummy_L=0._q
!                 CALL ONETOTHREE(3,dummy_L,p_force_L)
!                 dummy_L=TRANSPOSE(dummy_L)
!                 penalty_force_L=penalty_force_L+dummy_L
!               ENDIF
!             ENDIF

          ENDDO


          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          NI=1
          DO NT=1,T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
              IF (T_INFO%LSFOR(1,NI)) penalty_accel(1,NI)=penalty_force(1,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(2,NI)) penalty_accel(2,NI)=penalty_force(2,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(3,NI)) penalty_accel(3,NI)=penalty_force(3,NI)*FACT/2/T_INFO%POMASS(NT)
            ENDDO 
          ENDDO
          
          penalty_accel=MATMUL(TRANSPOSE(LATT_CUR%B),penalty_accel) 
          !penalty_accel=MATMUL(LATT_CUR%B,penalty_accel) 

          IF (IBDIM==3*T_INFO%NIONS+9) THEN
            penalty_accel_L=penalty_force_L*FACT/2/AMASS    
          ENDIF

        END SUBROUTINE hills_bias_direct

        SUBROUTINE harmonic_spring_accel(T_INFO,LATT_CUR,DYN,hspring_force,ICOORDINATES,REF_C,penalty_accel,penalty_accel_L,IBDIM,AMASS)
          !c bias potential for metadynamics without fictitious particles
          TYPE(type_info) :: T_INFO
          TYPE(latt) :: LATT_CUR
          TYPE(dynamics) :: DYN 
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: i,j,k,NI,NT
          INTEGER :: IBDIM
          REAL(q) :: REF_C(3,T_INFO%NIONS)
          REAL(q) :: FACT
          REAL(q) :: hspring_force(:)
          REAL(q) :: p_force(3*T_INFO%NIONS)
          REAL(q) :: p_force_L(9)
          REAL(q) :: penalty_force(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS)
          REAL(q) :: penalty_force_L(3,3),penalty_accel_L(3,3)
          REAL(q) :: dummy(3,T_INFO%NIONS),dummy_L(3,3)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: AMASS !c mass for lattice DOFs

          BMAT=0._q
          CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.FALSE.)
          !CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)

          penalty_force=0._q
          penalty_accel=0._q

          penalty_force_L=0._q
          penalty_accel_L=0._q

          j=0;k=0
          DO i=1,ICOORDINATES%NUMINTERNALS           
            !c harmonic spring contribution 
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==8) THEN
              k=k+1
              p_force=hspring_force(k)*BMAT(i,1:3*T_INFO%NIONS)
              dummy=0._q
              CALL ONETOTHREE(T_INFO%NIONS,dummy,p_force)
              dummy=MATMUL(LATT_CUR%B,dummy) !!!!!!!!!
              penalty_force=penalty_force+dummy

              !c lattice dynamics
              IF (IBDIM==3*T_INFO%NIONS+9) THEN
                p_force_L=hspring_force(k)*BMAT(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
                dummy_L=0._q
                CALL ONETOTHREE(3,dummy_L,p_force_L)
                dummy_L=TRANSPOSE(dummy_L)
                penalty_force_L=penalty_force_L+dummy_L
              ENDIF
            ENDIF

          ENDDO


          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          NI=1
          DO NT=1,T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
              IF (T_INFO%LSFOR(1,NI)) penalty_accel(1,NI)=penalty_force(1,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(2,NI)) penalty_accel(2,NI)=penalty_force(2,NI)*FACT/2/T_INFO%POMASS(NT)
              IF (T_INFO%LSFOR(3,NI)) penalty_accel(3,NI)=penalty_force(3,NI)*FACT/2/T_INFO%POMASS(NT)
            ENDDO 
          ENDDO
          
          penalty_accel=MATMUL(TRANSPOSE(LATT_CUR%B),penalty_accel) 
          !penalty_accel=MATMUL(LATT_CUR%B,penalty_accel) 

          IF (IBDIM==3*T_INFO%NIONS+9) THEN
            penalty_accel_L=penalty_force_L*FACT/2/AMASS    
          ENDIF

        END SUBROUTINE harmonic_spring_accel

        SUBROUTINE penalty_potential(penalty,hills,ICOORDINATES)
          !c user provided bias potential
          TYPE(hills_data) :: hills
          TYPE(penalty_data) :: penalty
          TYPE(coordstructure) :: ICOORDINATES
          REAL(q) :: pnorm
          INTEGER :: i,j,k
 
          penalty%force=0._q
          DO i=1,penalty%number  
            pnorm=0._q !!!
            k=0
            DO j=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(j)%STATUS==6) THEN
                k=k+1
                 IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5) &
                 &  pnorm=pnorm+(hills%position(k)-penalty%gauss(i)%position(k))**2
              ENDIF
            ENDDO

            !c sum up the forces 
            k=0
            DO j=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5 .OR. ICOORDINATES%COORDSTRUCT(j)%STATUS==6) THEN
                k=k+1
                IF (ICOORDINATES%COORDSTRUCT(j)%STATUS==5) THEN
                  penalty%force(k)=penalty%force(k)&   !!!!
                  & -(hills%position(k)-penalty%gauss(i)%position(k))* &
                  & penalty%gauss(i)%high/(penalty%gauss(i)%width**2)* &
                  & exp(-pnorm/2._q/penalty%gauss(i)%width**2)   !!!!
                ENDIF
              ENDIF
            ENDDO
          ENDDO
        END SUBROUTINE penalty_potential

        SUBROUTINE harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
          !c forces due to harmonic springs attached to internal coordinates
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: i,j,k
          INTEGER :: iconst8
          REAL(q) :: hspring_R0(iconst8),hspring_K(iconst8),hspring_force(iconst8)
          REAL(q) :: deltaS
 
          hspring_force=0._q
          k=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==8) THEN
              k=k+1
              deltaS=(ICOORDINATES%COORDSTRUCT(i)%VALUE-hspring_R0(k))
              !c take care of periodicity in definition of angle and torsion angle
              IF (ICOORDINATES%COORDSTRUCT(i)%TAG=='A ' .OR. &
              &   ICOORDINATES%COORDSTRUCT(i)%TAG=='T ') THEN
                DO
                  IF (deltaS>pi) THEN
                    deltaS=2*pi-deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
                DO
                  IF (deltaS<=-pi) THEN
                    deltaS=2*pi+deltaS
                  ELSE
                    EXIT
                  ENDIF
                ENDDO
              ENDIF
              hspring_force(k)=hspring_force(k)+hspring_K(k)*deltaS
            ENDIF
          ENDDO
   
        END SUBROUTINE harmonic_spring_force

        SUBROUTINE HILLS_METHOD_DIRECT(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,&
          &      g_io,WDES,iconst0,iconst2,iconst5,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
        !c NVT metadynamics (Andersen thermostat) without fictitious particles 
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(hills_data) :: hills
          TYPE (wavedes)  ::   WDES
          TYPE(penalty_data) :: penalty
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,T_old,T_new,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,random_counter
          INTEGER,SAVE :: counter=0
          INTEGER :: iconst0,iconst2,iconst5,iconst8        
          REAL(q) :: dummy,BMP,ANDERSEN_PROB
          REAL(q) :: CMASS(3,1)
          REAL(q) :: VEL_tmp(3,T_INFO%NIONS)
          REAL(q) :: hills_accel(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS),hspring_accel(3,T_INFO%NIONS)
          REAL(q) :: hills_accel_L(3,3),penalty_accel_L(3,3),hspring_accel_L(3,3)
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          !LOGICAL :: LSCALE,LCMASS
          REAL(q) :: hspring_R0(iconst8),hspring_K(iconst8),hspring_force(iconst8)
          REAL(q) :: TOTAL_MASS
          INTEGER,SAVE :: Ltxyz(3)

          !LSCALE=.FALSE.
          !LCMASS=.TRUE.

          !c add this step to total count
          counter=counter+1

          !c determine if coordinates for some direction
          !c are frozen
          IF (counter==1) THEN
            Ltxyz(:)=1
            DO j=1,3
              DO i=1,T_INFO%NIONS
                IF (.NOT. T_INFO%LSFOR(j,i)) Ltxyz(j)=0
              ENDDO
            ENDDO

            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="X ") Ltxyz(1)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Y ") Ltxyz(2)=0
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG=="Z ") Ltxyz(3)=0
              ENDIF
            ENDDO
          ENDIF


          AMASS=0._q 
          hills_accel_L=0._q;penalty_accel_L=0._q;hspring_accel_L=0._q
          hills_accel=0._q;penalty_accel=0._q;hspring_accel=0._q
           
          IF (iconst5>0) THEN
            !c update history of fictioous particles and of collerctive variables
            j=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                j=j+1
                hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE 
              ENDIF
            ENDDO
          
            !c add new hill
            IF (counter .GE. hills%bin) THEN
              CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
            ENDIF 

            !c calculate the penalty potential
            CALL penalty_potential(penalty,hills,ICOORDINATES)

            !c compute contributions from bias potentials
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,penalty%force,ICOORDINATES,DYN%POSIOC,penalty_accel,penalty_accel_L,3*T_INFO%NIONS,0._q)
            !hills%force=hills%force+penalty%force
            !CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,hspring_force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !c calculate forces due to harmonic springs
          IF (iconst8>0) THEN
            CALL harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
            CALL harmonic_spring_accel(T_INFO,LATT_CUR,DYN,hspring_force,ICOORDINATES,DYN%POSIOC,hspring_accel,hspring_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !c real variables +  andersen thermostat
          VEL_tmp=0._q
          !!CALL init_velocities(VEL_tmp,T_INFO,DYN,LATT_CUR,NDEGREES_OF_FREEDOM,LSCALE,LCMASS)
        
          ! Add a random velicity that is exactly
          ! distributed like the centre of mass velocity
          TOTAL_MASS = 0.0_q
          DO I = 1, T_INFO%NIONS
            TOTAL_MASS = TOTAL_MASS + T_INFO%POMASS(T_INFO%ITYP(I))
          END DO
          CMASS(:,1) = BOLTZMANN_VELOCITY(DYN, LATT_CUR, TOTAL_MASS)

          !c don't add translation for frozen direction
          DO i=1,3
            IF (Ltxyz(i)==0) CMASS(i,1)=0._q
          ENDDO

          DO i=1,T_INFO%NIONS
             DO j=1,3
                DYN%VEL(j,i)=DYN%VEL(j,i)+CMASS(j,1)
             ENDDO
          ENDDO

          !c boltzmann_velocities takes care of frozen atoms (velocities
          !c set to zero)
          CALL boltzmann_velocities(T_INFO,DYN,LATT_CUR,VEL_tmp)


          !DYN%VEL=DYN%VEL+2*(DYN%D2C-hills_accel-penalty_accel)
          !DYN%VEL=DYN%VEL+2*(DYN%D2C-hills_accel)
          DYN%VEL=DYN%VEL+2*(DYN%D2C-hills_accel-penalty_accel-hspring_accel)
       
          random_counter=0
          IF (DYN%INIT==1) THEN
            DO i=1,T_INFO%NIONS
              CALL RANE_ION(dummy)
              IF (dummy<ANDERSEN_PROB) THEN
                random_counter=random_counter+1
                DYN%VEL(:,i)=VEL_tmp(:,i)
              ENDIF
            ENDDO
          ENDIF

          CMASS=0._q
          !c GIVE_CMASS takes care of frozen directions (corresponding
          !c component set to zero)
          CALL GIVE_CMASS(T_INFO,DYN%VEL,CMASS)

          !!c don't add translation for frozen direction
          !DO i=1,3
          !  IF (Ltxyz(i)==0) CMASS(i,1)=0._q
          !ENDDO

          DO i=1,T_INFO%NIONS
            DO j=1,3
              DYN%VEL(j,i)=DYN%VEL(j,i)-CMASS(j,1)
            ENDDO
          ENDDO

          !c velocity for the step t-1/2dt
          VEL_tmp=DYN%VEL

          !!c velocity for the step t+1/2dt
          !DYN%VEL=DYN%VEL+2*(DYN%D2C-hills_accel-penalty_accel-hspring_accel)


          !CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
          !             & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          !T_new = 2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM
          !TEIN=T_new

          !write(*,*) 'spring_accel',spring_accel
          DYN%POSION=DYN%POSIOC+DYN%VEL
          ECONST=0._q
          !IF (ICOORDINATES%NUMINTERNALS>0) THEN
          IF (iconst0>0 .OR. iconst2>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,& 
            &   g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          !!c velocity for the step t
          !!VEL_tmp=(VEL_tmp+DYN%VEL)/2
          !!CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
          !!             & DYN%POTIM,LATT_CUR%A,VEL_tmp)
 
          !c due to compatibility reasons we still compute
          !c kinetic energy and T using velocities from the step t+1/2
          !c but in the future we should use VEL_tmp (velocity for step t) instead
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          TEIN=2*EKIN/BOLKEV/NDEGREES_OF_FREEDOM

          IF (IO%IU6>0) CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,0._q,0._q,counter)

          !IF (g_io%REPORT>=0) THEN
          IF (IO%IU6>0) THEN
            j=0
            write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                j=j+1
                write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5)') &
                & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE
                !write(*,*) 'test_val',ICOORDINATES%COORDSTRUCT(i)%VALUE,hills%position(j)
              ENDIF
            ENDDO     
          ENDIF

          IF (IO%IU6>0) THEN
            CALL TEMPERATURE_OUT(g_io,DYN,TEIN)            
          ENDIF

          !IF (g_io%REPORT>=0) THEN
          IF (IO%IU6>0) THEN
            write(g_io%REPORT,FMT='(/,2X,A32,X,I16)') '>Thermostat, num. of collisions:',random_counter
          ENDIF

          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
          EPS=0.0_q
          ES =0.0_q
        END SUBROUTINE HILLS_METHOD_DIRECT

        !SUBROUTINE RIGID_WALLS_direct()
        !  TYPE(type_info) :: T_INFO
        !  TYPE(latt) :: LATT_CUR
        !  TYPE(dynamics) :: DYN 
        ! ! TYPE(hills_data) :: hills
        !  TYPE(coordstructure) :: ICOORDINATES
        !  INTEGER :: i,j,NI,NT
        !  REAL(q) :: REF_C(3,T_INFO%NIONS)
        !  REAL(q) :: FACT
        !  REAL(q) :: internal_force(:)
        !  REAL(q) :: p_force(3*T_INFO%NIONS)
        !  REAL(q) :: penalty_force(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS)
        !  REAL(q) :: dummy(3,T_INFO%NIONS)
        !  REAL(q),ALLOCATABLE :: BMAT(:,:)
        !
        !  ALLOCATE(BMAT(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS))
        !  BMAT=0._q
        !  CALL BMATRIX(T_INFO,REF_C,REF_C,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.FALSE.)
        !  CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)
        !
        !  CALL DEAL_XYZ(T_INFO,REF_C,REF_C,LATT_CUR%A,ICOORDINATES)
        !  j=0
        !  DO i=1,ICOORDINATES%NUMINTERNALS
        !    IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
        !      j=j+1
        !      IF (ICOORDINATES%COORDSTRUCT(i)%VALUE<=(penalty%wall(j,1)+hills%gauss(1)%width)) THEN
        !        DYN%POSION=DYN%POSION   SUM(BMAT(i,:)*DYN
        !        hills%velocity(i)=-hills%velocity(i)
        !        hills%position(j)=hills%position(j)+2*hills%gauss(1)%width
        !      ENDIF
        !      IF (hills%position(j)>=(penalty%wall(j,2)-hills%gauss(1)%width)) THEN
        !        hills%position(j)=hills%position(j)-2*hills%gauss(1)%width
        !      ENDIF
        !    ENDIF
        ! ENDDO
        !
        !END SUBROUTINE RIGID_WALLS_direc
       

        SUBROUTINE HILLS_METHOD_DIRECT_nose(DYN,ICOORDINATES,hills,penalty,T_INFO,INFO,LATT_CUR, &
          &      EKIN,EPS,ES,DISMAX,NDEGREES_OF_FREEDOM,IO,EPOT,ANDERSEN_PROB,& 
          & g_io,WDES,iconst0,iconst2,iconst5,iconst8,hspring_R0,hspring_K,hspring_force,TEIN)
        !c NVT metadynamics (Nose-Hover thermostat) without fictitious particles 
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(gadget_io) :: g_io
          TYPE(hills_data) :: hills
          TYPE (wavedes)  ::   WDES
          TYPE(penalty_data) :: penalty
          REAL(q) :: C_FORCE(3,T_INFO%NIONS)
          REAL(q) :: EKIN,EPS,ES,DISMAX,TEMPER,EPOT,T_old,T_new,ECONST,TEIN
          INTEGER :: i,j,ii,jj,NDEGREES_OF_FREEDOM,random_counter
          INTEGER,SAVE :: counter=0        
          INTEGER :: iconst0,iconst2,iconst5,iconst8
          REAL(q) :: dummy,BMP,ANDERSEN_PROB
          REAL    :: SQQ,hEKIN,UT,UL
          REAL(q) :: hills_accel(3,T_INFO%NIONS),penalty_accel(3,T_INFO%NIONS),hspring_accel(3,T_INFO%NIONS)
          REAL(q) :: hills_accel_L(3,3),penalty_accel_L(3,3),hspring_accel_L(3,3)
          REAL(q) :: AMASS,C_FORCE_L(3,3)
          REAL(q) :: hspring_R0(iconst8),hspring_K(iconst8),hspring_force(iconst8)

          counter=counter+1
          !DYN%POSIOC=DYN%POSION
          UL =1E-10_q*LATT_CUR%ANORM(1)
          UT =DYN%POTIM*1E-15_q
          SQQ=DYN%SMASS*(AMTOKG/EVTOJ)*(UL/UT)**2

          AMASS=0._q
          hills_accel_L=0._q;penalty_accel_L=0._q;hspring_accel_L=0._q
          hills_accel=0._q;penalty_accel=0._q;hspring_accel=0._q

          IF (DYN%INIT==0) THEN
            DYN%SNOSE=0._q
          ENDIF

          IF (counter==1) THEN
            hills%SNOSE=0.0         
          ENDIF 
          

          IF (iconst5>0) THEN 
            !c update history of fictioous particles and of collerctive variables
            j=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                j=j+1
                hills%position(j)=ICOORDINATES%COORDSTRUCT(i)%VALUE
              ENDIF
            ENDDO
          
            !c calculate additional potentaials (bias, and penalty)
            IF (counter .GE. hills%bin) THEN
              CALL hills_bias_potential(hills,penalty,ICOORDINATES,DYN,counter,g_io,IO)
            ENDIF 
            CALL penalty_potential(penalty,hills,ICOORDINATES)

            !c compute contributions from bias potentials
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
            CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,penalty%force,ICOORDINATES,DYN%POSIOC,penalty_accel,penalty_accel_L,3*T_INFO%NIONS,0._q)
            !hills%force=hills%force+penalty%force
            !CALL hills_bias_direct(T_INFO,LATT_CUR,DYN,hills%force,hspring_force,ICOORDINATES,DYN%POSIOC,hills_accel,hills_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !c calculate forces due to harmonic springs
          !IF (iconst8>0) call harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
          IF (iconst8>0) THEN
            CALL harmonic_spring_force(iconst8,hspring_R0,hspring_K,hspring_force,ICOORDINATES)
            CALL harmonic_spring_accel(T_INFO,LATT_CUR,DYN,hspring_force,ICOORDINATES,DYN%POSIOC,hspring_accel,hspring_accel_L,3*T_INFO%NIONS,0._q)
          ENDIF

          !c real variables +  nose-hoover thermostat
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          T_old = 2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM)
          IF (DYN%INIT==1) DYN%VEL=(1._q-DYN%SNOSE(2)/2._q)*DYN%VEL+DYN%D2C-hills_accel-penalty_accel

          EPS=0.5_q*(DYN%SNOSE(2)**2)*SQQ
          ES =NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP*(DYN%SNOSE(1))
          DYN%VEL=(DYN%VEL+DYN%D2C-hills_accel-penalty_accel-hspring_accel)/(1._q+DYN%SNOSE(2)/2._q)       
          !DYN%VEL=(DYN%VEL+DYN%D2C-hills_accel-penalty_accel)/(1._q+DYN%SNOSE(2)/2._q)
          !DYN%VEL=(DYN%VEL+DYN%D2C-hills_accel)/(1._q+DYN%SNOSE(2)/2._q)
          CALL KINETIC_E(EKIN,T_INFO%NIONS,T_INFO%NTYP,T_INFO%ITYP,T_INFO%POMASS,&
                       & DYN%POTIM,LATT_CUR%A,DYN%VEL)
          T_new = 2*EKIN/BOLKEV/(NDEGREES_OF_FREEDOM)
          TEIN=T_new
          DYN%SNOSE(3)=DYN%SNOSE(2)
          DYN%SNOSE(2)=DYN%SNOSE(2)+(2*EKIN-NDEGREES_OF_FREEDOM*BOLKEV*DYN%TEMP)/SQQ
          DYN%SNOSE(4)=DYN%SNOSE(1)
          DYN%SNOSE(1)=DYN%SNOSE(1)+(DYN%SNOSE(2)+DYN%SNOSE(3))/2._q

          DYN%POSION=DYN%POSIOC+DYN%VEL
          ECONST=0._q
          IF (ICOORDINATES%NUMINTERNALS>0) THEN
            C_FORCE=0._q
            CALL Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR,IO,C_FORCE,C_FORCE_L,ICOORDINATES,&
            &  g_io,3*T_INFO%NIONS,AMASS,LATT_CUR%A,ECONST,iconst0,iconst2)
            DYN%POSION=DYN%POSION+C_FORCE
            DYN%VEL=DYN%VEL+C_FORCE
          ENDIF

          IF (IO%IU6>0) CALL ENERGY_OUT(g_io,DYN,T_INFO,EKIN,EPOT,ECONST,0._q,EPS,ES,counter)

#ifdef MPI
            CALL  M_bcast_d(WDES%COMM, hills%position, hills%number)
#endif

          IF (IO%IU6>0) THEN
            j=0
            write(g_io%REPORT,ADVANCE='YES',FMT='(/,2X,A13)') '>Metadynamics'
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==5) THEN
                j=j+1
                write(g_io%REPORT,ADVANCE='YES',FMT='(3X,A6,X,F9.5)') &
                & 'fic_p>',ICOORDINATES%COORDSTRUCT(i)%VALUE
                !write(*,*) 'test_val',ICOORDINATES%COORDSTRUCT(i)%VALUE,hills%position(j)
              ENDIF
            ENDDO     
          ENDIF

          IF (IO%IU6>0) THEN
            CALL TEMPERATURE_OUT(g_io,DYN,T_new)
          ENDIF


          DISMAX=0
          DO I=1,T_INFO%NIONS
             DISMAX=MAX(DISMAX, &
             &(DYN%VEL(1,I))**2+ &
             &(DYN%VEL(2,I))**2+ &
             &(DYN%VEL(3,I))**2)
          ENDDO
          DISMAX=SQRT(DISMAX)
        END SUBROUTINE HILLS_METHOD_DIRECT_nose

        SUBROUTINE tis_stop(DYN,T_INFO,INFO,LATT_CUR,IO,g_io,ICOORDINATES,iconst7)
          !c monitor coordinates with STATUS=7
          !c terminate if the value doesn't fall into interval (VALUE_MIN,VALUE_MAX)
          !c this is useful mainly for the transition interface sampling calculations:
          TYPE(coordstructure) :: ICOORDINATES
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(dynamics) :: DYN      
          TYPE (in_struct) :: IO
          TYPE(latt) :: LATT_CUR   
          TYPE(gadget_io) :: g_io
          REAL(q),SAVE,ALLOCATABLE :: valueA(:),valueB(:)
          INTEGER :: iconst7
          INTEGER,SAVE :: counter=0
          INTEGER :: IDUM,IERR,N,i,ii
          LOGICAL, SAVE :: LMIN=.TRUE.
          LOGICAL, SAVE :: LMAX=.TRUE.
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC

          !IF (counter==1) THEN
          !  !c identify the number of parameters with STATUS==7
          !  iconst7=0
          !  DO i=1,ICOORDINATES%NUMINTERNALS
          !    IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
          !      iconst7=iconst7+1
          !    ENDIF
          !  ENDDO
          !ENDIF

          !c dont continue if no geometric parameters with STATUS==7
          IF (iconst7==0) THEN
            RETURN
          ENDIF
         
          counter=counter+1

          IF (counter==1) THEN
            ALLOCATE(valueA(iconst7),valueB(iconst7))
            valueA=-1000._q
            valueB= 1000._q

            !c read in the method-specific input parameters
            CALL READER_ICONST7(IO,g_io,ICOORDINATES,iconst7,LMIN,LMAX,valueA,valueB)
          ENDIF

          IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Monit_coord'
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
              IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,X,F15.5)') &
                'mc>',ICOORDINATES%COORDSTRUCT(i)%TAG,ICOORDINATES%COORDSTRUCT(i)%VALUE
            ENDIF
          ENDDO

          IF (LMIN) THEN
            INFO%LSTOP=.TRUE.
            ii=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
                ii=ii+1
                IF (ICOORDINATES%COORDSTRUCT(i)%VALUE .GE. valueA(ii)) THEN
                  INFO%LSTOP=.FALSE.
                ENDIF
              ENDIF
            ENDDO
            IF (INFO%LSTOP) THEN
              IF (IO%IU0>=0) write(*,*) 'REACHED LOWER LIMIT FOR COORD.'
              IF (INFO%LSTOP) RETURN
            ENDIF
          ENDIF
  
          IF (LMAX) THEN
            INFO%LSTOP=.TRUE.
            ii=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==7) THEN
                ii=ii+1
                IF (ICOORDINATES%COORDSTRUCT(i)%VALUE .LE. valueB(ii)) THEN
                  INFO%LSTOP=.FALSE.
                ENDIF
              ENDIF
            ENDDO
            IF (INFO%LSTOP) THEN
              IF (IO%IU0>=0) write(*,*) 'REACHED UPPER LIMIT FOR COORD.'
              IF (INFO%LSTOP) RETURN
            ENDIF
          ENDIF
        END SUBROUTINE tis_stop

        SUBROUTINE Lproj_out_step(T_INFO,INFO,DYN,LATT_CUR, &
                IO,C_FORCE,C_FORCE_L,ICOORDINATES,g_io,IBDIM,AMASS,NewA,ECONST,iconst0,iconst2)
          !c this is basically the SHAKE algorithm...
          TYPE(type_info) :: T_INFO
          TYPE(dynamics) :: DYN      
          TYPE (in_struct) :: IO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR     
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(gadget_io) :: g_io
          REAL(q) ::xpos(3*T_INFO%NIONS)  ! positions in row format
          INTEGER :: i,j,k  ,ii ,jj,idummy                 ! indeces
          INTEGER :: ninfo,IBDIM
          INTEGER,SAVE :: counter=0
          REAL(q) :: BMAT_0(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: GMAT(ICOORDINATES%NUMINTERNALS) !c B matrix for working and reference coords.
          REAL(q) :: PRIMS1(ICOORDINATES%NUMINTERNALS),PRIMS2(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: gamma(ICOORDINATES%NUMINTERNALS),gamma0(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS),ZET_DET
          REAL(q),ALLOCATABLE,SAVE :: MASSES(:)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS),C_FORCE_(3,T_INFO%NIONS)
          REAL(q) :: POSION_(3,T_INFO%NIONS),POSIOC_(3,T_INFO%NIONS)
          REAL(q) :: err,err1
          REAL(q) :: UL,UT,FACT,ECONST
          INTEGER IDUM,IERR,N
          INTEGER :: iconst0,iconst2
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q),SAVE,ALLOCATABLE :: INCREM(:)
          INTEGER,SAVE :: EQUI_REGIME=0
          LOGICAL,SAVE :: LBLUEOUT=.FALSE.
          REAL(q),SAVE :: BMTOL,BMTOLSOFT,BMSCA 
          INTEGER,SAVE :: BMITER
          REAL(q) :: AMASS,NewA(3,3),NewA_(3,3),NewB(3,3),C_FORCE_L(3,3),C_FORCE_L_(3,3),xposL(9)
          REAL(q) :: BMAT_square(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS),BMAT_inv(IBDIM,ICOORDINATES%NUMINTERNALS)

          !c don't continue if no geometric constraints are defined
          IF (iconst0==0 .AND. iconst2==0) THEN
            RETURN
          ENDIF

          counter=counter+1
          UL=1E-10_q
          UT=DYN%POTIM*1E-15_q
          C_FORCE=0._q
          C_FORCE_L=0._q

          POSIOC_=DYN%POSIOC
          POSION_=DYN%POSION
          NewA_=NewA
          NewB=TRANSPOSE(NewA)
          CALL SVDINVERSE(NewB,3,idummy)
          
          IF (counter==1) THEN
            ICOORDINATES%COORDSTRUCT(:)%DVALUE=ICOORDINATES%COORDSTRUCT(:)%VALUE  !xx

            !c masses in amu
            ALLOCATE(MASSES(IBDIM))
            MASSES=0._q
            DO i=1,T_INFO%NIONS
              MASSES(3*i-2:3*i)=(T_INFO%POMASS(T_INFO%ITYP(i)))
            ENDDO
            IF (IBDIM==3*T_INFO%NIONS+9) THEN
              DO i=1,3
                DO j=1,3
                  MASSES(3*T_INFO%NIONS+3*(i-1)+j)=AMASS
                ENDDO
              ENDDO
            ENDIF

            ALLOCATE(INCREM(iconst0))
            INCREM=0._q
            !c read in some method-specific input parameters
            CALL READER_ICONST0(IO,g_io,iconst0,BMTOL,BMTOLSOFT,BMSCA,BMITER,LBLUEOUT,EQUI_REGIME,INCREM)
          END IF

          !c dont fix coordinates in the equilibration period
          IF (EQUI_REGIME .GE. counter) RETURN

          !c everything for old coordinates
          !c read constraints
          j=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
              j=j+1
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE+counter*INCREM(j) 
            ELSE IF  (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              j=j+1
              !PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%VALUE
            ENDIF
          ENDDO

          BMAT_0=0._q
          CALL BMATRIX(T_INFO,DYN%POSIOC,DYN%POSIOC,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT_0,.TRUE.)
          !!CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT_0,.FALSE.)
          !c upper triangle inly
          IF (IBDIM==3*T_INFO%NIONS+9) THEN
            BMAT_0(:,3*T_INFO%NIONS+4)=0._q
            BMAT_0(:,3*T_INFO%NIONS+7)=0._q
            BMAT_0(:,3*T_INFO%NIONS+8)=0._q
          ENDIF

          !CALL ICONST_BMAT(BMAT_0,ICOORDINATES,T_INFO)
          BMAT=0._q

          err=0._q
          gamma=0._q
          gamma0=0._q
          ii=0

          DO i=1,ICOORDINATES%NUMINTERNALS      
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              err=err+1
            ENDIF
          ENDDO
        
          DO
            !c exit if no constraints are defined
            IF (ii==0 .AND. err<BMTOL) THEN
              EXIT
            ENDIF
            ii=ii+1

            !c terminate with error message if SHAKE doesn't converge in BMITER steps
            IF (ii>BMITER) THEN  
              !INFO%LSTOP=.TRUE.
              IF (IO%IU6>0) write(*,*) 'Error: SHAKE algorithm did not converge! err=',err
              IF (abs(err).LE. BMTOLSOFT) THEN
                IF (IO%IU6>0) write(*,*) "I'll try to recover this problem in the next step"
                EXIT
              ELSE
                IF (IO%IU6>0) write(*,*) "Error too large, I have to terminate this calculation!"
                STOP
              ENDIF
              !RETURN
            END IF
           
            !c values for constraint parameters in the current step 
            CALL DEAL_XYZ(T_INFO,DYN%POSION,DYN%POSION,NewA,ICOORDINATES)
            PRIMS2=0._q
            PRIMS2=ICOORDINATES%COORDSTRUCT(:)%VALUE
          
            CALL DECYCLE(PRIMS1,PRIMS2,ICOORDINATES)
            ICOORDINATES%COORDSTRUCT(:)%VALUE=PRIMS2(:)
            IF (err<BMTOL) EXIT
            BMAT=0._q
       
            !c Jacobi matrix
            CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(NewA),ICOORDINATES,BMAT,.TRUE.)
            !c upper triangle only
            IF (IBDIM==3*T_INFO%NIONS+9) THEN
              BMAT(:,3*T_INFO%NIONS+4)=0._q
              BMAT(:,3*T_INFO%NIONS+7)=0._q
              BMAT(:,3*T_INFO%NIONS+8)=0._q
            ENDIF

            jj=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                jj=jj+1
                IF (jj==1) err=ABS(PRIMS2(i)-PRIMS1(i))
                err1=ABS(PRIMS2(i)-PRIMS1(i))
                IF (err1 .GT. err) err=err1
                gamma(i)=(PRIMS2(i)-PRIMS1(i))/SUM(BMAT_0(i,:)/MASSES(:)*BMAT(i,:))/BMSCA     !!!
                gamma0(i)=gamma0(i)+(PRIMS2(i)-PRIMS1(i))/SUM(BMAT_0(i,:)/MASSES(:)*BMAT(i,:))/BMSCA   !!!

                !c lattice dynamics
                IF (IBDIM==3*T_INFO%NIONS+9) THEN
                  C_FORCE_L_=0._q
                  xposL=-gamma(i)*BMAT_0(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)/MASSES(3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
                  CALL ONETOTHREE(3,C_FORCE_L_,xposL) 
                  C_FORCE_L_=TRANSPOSE(C_FORCE_L_) !!
                  NewA(:,:)=NewA(:,:)+C_FORCE_L_
                  NewB=TRANSPOSE(NewA)
                  CALL SVDINVERSE(NewB,3,idummy)
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                    C_FORCE_L=C_FORCE_L+C_FORCE_L_
                  ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                    C_FORCE_L=C_FORCE_L+2*C_FORCE_L_
                  ENDIF
                ENDIF

                xpos=-gamma(i)*BMAT_0(i,1:3*T_INFO%NIONS)/MASSES(1:3*T_INFO%NIONS)
                !!!xpos=-gamma(i)*BMAT(i,1:3*T_INFO%NIONS)/MASSES(1:3*T_INFO%NIONS)
                C_FORCE_=0._q
                CALL ONETOTHREE(T_INFO%NIONS,C_FORCE_,xpos)
                !C_FORCE_=MATMUL(TRANSPOSE(LATT_CUR%B),C_FORCE_)
                C_FORCE_=MATMUL(TRANSPOSE(NewB),C_FORCE_)
                !beg 
                DO j=1,T_INFO%NIONS
                  IF (.NOT. T_INFO%LSFOR(1,j)) C_FORCE_(1,j)=0._q
                  IF (.NOT. T_INFO%LSFOR(2,j)) C_FORCE_(2,j)=0._q
                  IF (.NOT. T_INFO%LSFOR(3,j)) C_FORCE_(3,j)=0._q
                ENDDO
                !end
                DYN%POSION(:,:)=DYN%POSION(:,:)+C_FORCE_(:,:)
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                   C_FORCE=C_FORCE+C_FORCE_
                ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                   C_FORCE=C_FORCE+2*C_FORCE_
                ENDIF

              END IF
            ENDDO                     
            !!BMAT_0=BMAT
            !IF (IO%IU0>=0) write(IO%IU0,*) 'err',err
          ENDDO

          gamma=-gamma0

          !CALL XML_INCAR('shakeiter','I',ii,RDUM,CDUM,LDUM,CHARAC,1)
          IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A19,I4,A6)') 'SHAKE converged in ',ii,' steps'

          !CALL XML_TAG("separator","gconstraints")
          IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Const_coord'
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              
              err=PRIMS2(i)-PRIMS1(i)
              IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,2(X,F9.5),X,E13.6E2)') 'cc>',ICOORDINATES%COORDSTRUCT(i)%TAG,&
                  PRIMS1(i),PRIMS2(i),err
            ENDIF
          ENDDO
          !CALL XML_CLOSE_TAG

!!!!!!!!!!!!!!!!!!!!!!!!!
          !c write down the componenets needed to compute the free energy gradients
          !c this can be time-consuming if too many constraints are defined
          IF (LBLUEOUT) THEN                
            CALL make_Z(T_INFO,IBDIM,DYN%POSIOC,LATT_CUR%A,ICOORDINATES,MASSES,ZET) 
            ZET_DET=z_det(ZET,ICOORDINATES,IO) 
            

            CALL make_G(T_INFO,DYN,LATT_CUR,ICOORDINATES,MASSES,IBDIM,ZET,ZET_DET,GMAT,IO)
            GMAT=GMAT*BOLKEV*DYN%TEMP      
         
            FACT=(DYN%POTIM**2)*EVTOJ/(AMTOKG) *1E-10_q
            gamma=gamma/FACT

            LDUM=.TRUE.
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .AND. IO%IU6>0) THEN 
                IF (LDUM) THEN
                  write(g_io%REPORT,FMT='(/,2X,A10)') '>Blue_moon'
                  write(g_io%REPORT,FMT='(8X,A6,8X,A10,4X,A3,11X,A23)') 'lambda','|z|^(-1/2)','GkT','|z|^(-1/2)*(lambda+GkT)'
                  LDUM=.FALSE.
                ENDIF
                write(g_io%REPORT,FMT='(3X,A4,(X,E13.6E2),(X,E13.6E2),(X,E13.6E2),X,E13.6E2)') &
                    'b_m>', gamma(i),1._q/SQRT(ZET_DET),GMAT(i),1._q/SQRT(ZET_DET)*(gamma(i)+GMAT(i))
              ENDIF
            ENDDO
          ENDIF

          !calculate the energy contributions due to constraints
          ECONST=0._q
          CALL DEAL_XYZ(T_INFO,POSION_,DYN%POSION,LATT_CUR%A,ICOORDINATES)
          PRIMS2=0._q
          PRIMS2=ICOORDINATES%COORDSTRUCT(:)%VALUE
          CALL DECYCLE(PRIMS1,PRIMS2,ICOORDINATES)
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
              ECONST=ECONST+gamma(i)*(PRIMS2(i)-PRIMS1(i))
            ENDIF
          ENDDO

          DYN%POSIOC=POSIOC_
          DYN%POSION=POSION_
          NewA=NewA_
        END SUBROUTINE Lproj_out_step

        SUBROUTINE Rattle(counter,T_INFO,INFO,DYN,LATT_CUR, &
                IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,IBDIM,AMASS,NewA,ECONST,iconst0,iconst2,IMODE)
          TYPE(type_info) :: T_INFO
          TYPE(dynamics) :: DYN      
          TYPE (in_struct) :: IO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR     
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(gadget_io) :: g_io
          REAL(q) ::xpos(3*T_INFO%NIONS)  ! positions in row format
          INTEGER :: i,j,k  ,ii ,jj,idummy                 ! indeces
          INTEGER :: ninfo,IBDIM
          INTEGER :: IMODE
          INTEGER :: counter
          !INTEGER,SAVE :: counter=0
          REAL(q) :: BMAT_0(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: GMAT(ICOORDINATES%NUMINTERNALS) !c B matrix for working and reference coords.
          REAL(q) :: PRIMS1(ICOORDINATES%NUMINTERNALS),PRIMS2(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: gamma(ICOORDINATES%NUMINTERNALS),gamma0(ICOORDINATES%NUMINTERNALS),theta(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS),ZET_DET
          REAL(q),ALLOCATABLE,SAVE :: MASSES(:)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS),C_FORCE_(3,T_INFO%NIONS)
          REAL(q) :: V_FORCE(3,T_INFO%NIONS),V_FORCE_L(3,3)
          REAL(q) :: POSION_(3,T_INFO%NIONS),POSIOC_(3,T_INFO%NIONS), VEL_(3,T_INFO%NIONS)
          REAL(q) :: err,err1
          REAL(q) :: UL,UT,FACT,ECONST
          INTEGER IDUM,IERR,N
          INTEGER :: iconst0,iconst2
          LOGICAL :: LOPEN,LDUM
          LOGICAL,SAVE :: LFIRST=.TRUE.
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q),SAVE,ALLOCATABLE :: INCREM(:)
          INTEGER,SAVE :: EQUI_REGIME=0
          LOGICAL,SAVE :: LBLUEOUT=.FALSE.
          REAL(q),SAVE :: BMTOL,BMTOLSOFT,BMSCA 
          INTEGER,SAVE :: BMITER
          REAL(q) :: AMASS,NewA(3,3),NewA_(3,3),NewB(3,3),C_FORCE_L(3,3),C_FORCE_L_(3,3),xposL(9)
          REAL(q) :: AVEL_(3,3)
          REAL(q) :: BMAT_square(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS),BMAT_inv(IBDIM,ICOORDINATES%NUMINTERNALS)

          V_FORCE_L=0._q
          C_FORCE_L=0._q
          V_FORCE=0._q
          C_FORCE=0._q

          !c don't continue if no geometric constraints are defined
          IF (iconst0==0 .AND. iconst2==0) THEN
            RETURN
          ENDIF

          !counter=counter+1

          UL=1E-10_q
          UT=DYN%POTIM*1E-15_q
          C_FORCE=0._q
          C_FORCE_L=0._q

          POSIOC_=DYN%POSIOC
          POSION_=DYN%POSION
          NewA_=NewA
          NewB=TRANSPOSE(NewA)
          CALL SVDINVERSE(NewB,3,idummy)
          
          IF (LFIRST) THEN
            LFIRST=.FALSE.
            ICOORDINATES%COORDSTRUCT(:)%DVALUE=ICOORDINATES%COORDSTRUCT(:)%VALUE  !xx

            !c masses in amu
            ALLOCATE(MASSES(IBDIM))
            MASSES=0._q
            DO i=1,T_INFO%NIONS
              MASSES(3*i-2:3*i)=(T_INFO%POMASS(T_INFO%ITYP(i)))
            ENDDO
            IF (IBDIM==3*T_INFO%NIONS+9) THEN
              DO i=1,3
                DO j=1,3
                  MASSES(3*T_INFO%NIONS+3*(i-1)+j)=AMASS
                ENDDO
              ENDDO
            ENDIF

            ALLOCATE(INCREM(iconst0))
            INCREM=0._q
            !c read in some method-specific input parameters
            CALL READER_ICONST0(IO,g_io,iconst0,BMTOL,BMTOLSOFT,BMSCA,BMITER,LBLUEOUT,EQUI_REGIME,INCREM)
          END IF

          !c dont fix coordinates in the equilibration period
          IF (EQUI_REGIME .GE. counter) RETURN

          !c everything for old coordinates
          !c read constraints
          j=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
              j=j+1
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE+counter*INCREM(j) 
            ELSE IF  (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              j=j+1
              !PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%VALUE
            ENDIF
          ENDDO

          SELECT CASE(IMODE)
          CASE(0)
            BMAT_0=0._q
            CALL BMATRIX(T_INFO,DYN%POSIOC,DYN%POSIOC,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT_0,.TRUE.)
 
            BMAT=0._q

            err=0._q
            gamma=0._q
            gamma0=0._q
            ii=0

            DO i=1,ICOORDINATES%NUMINTERNALS      
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                err=err+1
              ENDIF
            ENDDO
        
            DO
              !c exit if no constraints are defined
              IF (ii==0 .AND. err<BMTOL) THEN
                EXIT
              ENDIF
              ii=ii+1

              !c terminate with error message if SHAKE doesn't converge in BMITER steps
              !!IF (ii>BMITER) THEN  
              !!  IF (IO%IU0>=0) write(*,*) 'Error: SHAKE algorithm did not converge!'
              !!  STOP
              !!END IF

              !c terminate with error message if SHAKE doesn't converge in BMITER steps
              IF (ii>BMITER) THEN  
                !INFO%LSTOP=.TRUE.
                IF (IO%IU6>0) write(*,*) 'Error: RATTLE algorithm did not converge! err=',err
                IF (abs(err).LE. BMTOLSOFT) THEN
                  IF (IO%IU6>0) write(*,*) "I'll try to recover this problem in the next step"
                  EXIT
                ELSE
                  IF (IO%IU6>0) write(*,*) "Error too large, I have to terminate this calculation!"
                  STOP
                ENDIF
                !RETURN
              END IF

           
              !c values for constraint parameters in the current step 
              CALL DEAL_XYZ(T_INFO,DYN%POSION,DYN%POSION,NewA,ICOORDINATES)
              PRIMS2=0._q
              PRIMS2=ICOORDINATES%COORDSTRUCT(:)%VALUE
          
              CALL DECYCLE(PRIMS1,PRIMS2,ICOORDINATES)
              ICOORDINATES%COORDSTRUCT(:)%VALUE=PRIMS2(:)
              IF (err<BMTOL) EXIT
              BMAT=0._q
       
              !c Jacobi matrix
              CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(NewA),ICOORDINATES,BMAT,.TRUE.)
 
              jj=0
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                  jj=jj+1
                  IF (jj==1) err=ABS(PRIMS2(i)-PRIMS1(i))
                  err1=ABS(PRIMS2(i)-PRIMS1(i))
                  IF (err1 .GT. err) err=err1
                  gamma(i)=(PRIMS2(i)-PRIMS1(i))/SUM(BMAT_0(i,:)/MASSES(:)*BMAT(i,:))/BMSCA     !!!
                  gamma0(i)=gamma0(i)+(PRIMS2(i)-PRIMS1(i))/SUM(BMAT_0(i,:)/MASSES(:)*BMAT(i,:))/BMSCA   !!!

                  !c lattice dynamics
                  IF (IBDIM==3*T_INFO%NIONS+9) THEN
                    C_FORCE_L_=0._q
                    xposL=-gamma(i)*BMAT_0(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)/MASSES(3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
                    CALL ONETOTHREE(3,C_FORCE_L_,xposL) 
                    C_FORCE_L_=TRANSPOSE(C_FORCE_L_) !!
                    NewA(:,:)=NewA(:,:)+C_FORCE_L_
                    NewB=TRANSPOSE(NewA)
                    CALL SVDINVERSE(NewB,3,idummy)
                    IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                      C_FORCE_L=C_FORCE_L+C_FORCE_L_
!                     ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
!                       C_FORCE_L=C_FORCE_L+2*C_FORCE_L_
                    ENDIF
                  ENDIF

                  xpos=-gamma(i)*BMAT_0(i,1:3*T_INFO%NIONS)/MASSES(1:3*T_INFO%NIONS)
                  C_FORCE_=0._q
                  CALL ONETOTHREE(T_INFO%NIONS,C_FORCE_,xpos)
                  !C_FORCE_=MATMUL(TRANSPOSE(LATT_CUR%B),C_FORCE_)
                  C_FORCE_=MATMUL(TRANSPOSE(NewB),C_FORCE_)

                  !c remove force for frozen fractional coords 
                  DO j=1,T_INFO%NIONS
                    IF (.NOT. T_INFO%LSFOR(1,j)) C_FORCE_(1,j)=0._q
                    IF (.NOT. T_INFO%LSFOR(2,j)) C_FORCE_(2,j)=0._q
                    IF (.NOT. T_INFO%LSFOR(3,j)) C_FORCE_(3,j)=0._q
                  ENDDO

                  DYN%POSION(:,:)=DYN%POSION(:,:)+C_FORCE_(:,:)
                  IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                     C_FORCE=C_FORCE+C_FORCE_
!                   ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
!                      C_FORCE=C_FORCE+2*C_FORCE_
                  ENDIF

                END IF
              ENDDO                     
            !!BMAT_0=BMAT
            !IF (IO%IU0>=0) write(IO%IU0,*) 'err',err
            ENDDO

            gamma=-gamma0

            !CALL XML_INCAR('shakeiter','I',ii,RDUM,CDUM,LDUM,CHARAC,1)
            IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A24,I4,A6)') 'RATTLE_pos converged in ',ii,' steps'

            !CALL XML_TAG("separator","gconstraints")
            IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A12)') '>Const_coord'
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN              
                err=PRIMS2(i)-PRIMS1(i)
                IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,2(X,F9.5),X,E13.6E2)') 'cc>',ICOORDINATES%COORDSTRUCT(i)%TAG,&
                &   PRIMS1(i),PRIMS2(i),err
              ENDIF
            ENDDO
            !CALL XML_CLOSE_TAG

            !calculate the energy contributions due to constraints
            ECONST=0._q
            CALL DEAL_XYZ(T_INFO,POSION_,DYN%POSION,LATT_CUR%A,ICOORDINATES)
            PRIMS2=0._q
            PRIMS2=ICOORDINATES%COORDSTRUCT(:)%VALUE
            CALL DECYCLE(PRIMS1,PRIMS2,ICOORDINATES)
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                ECONST=ECONST+gamma(i)*(PRIMS2(i)-PRIMS1(i))
              ENDIF
            ENDDO

            DYN%POSIOC=POSIOC_
            DYN%POSION=POSION_
            NewA=NewA_
          CASE(1)
            BMAT=0._q
       
            !c Jacobi matrix
            CALL BMATRIX(T_INFO,DYN%POSIOC,DYN%POSIOC,TRANSPOSE(NewA),ICOORDINATES,BMAT,.TRUE.)

            CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)

            VEL_=DYN%VEL
            V_FORCE=0._q

            AVEL_=LATT_CUR%Avel
            V_FORCE_L=0._q

            err=0._q

            DO i=1,ICOORDINATES%NUMINTERNALS      
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                err=err+1
              ENDIF
            ENDDO

            !c start iterations to fullfill constraints for velocities
            DO ii=1,BMITER+1

              !c exit if no constraints are defined
              IF (ii==1 .AND. err<BMTOL) THEN
                !V_FORCE=VEL_-DYN%VEL
                EXIT
              ENDIF
           
              !c terminate with error message if SHAKE doesn't converge in BMITER steps
              !!IF (ii==BMITER+1) THEN  
              !!  IF (IO%IU0>=0) write(*,*) 'Error: RATTLE_vel algorithm did not converge!'
              !!  STOP
              !!END IF

              !c terminate with error message if SHAKE doesn't converge in BMITER steps
              IF (ii>BMITER) THEN  
                !INFO%LSTOP=.TRUE.
                IF (IO%IU6>0) write(*,*) 'Error: RATTLE_vel algorithm did not converge! err=',err
                IF (abs(err).LE. BMTOLSOFT) THEN
                  IF (IO%IU6>0) write(*,*) "I'll try to recover this problem in the next step"
                  EXIT
                ELSE
                  IF (IO%IU6>0) write(*,*) "Error too large, I have to terminate this calculation!"
                  STOP
                ENDIF
                !RETURN
              END IF
           
              IF (err<BMTOL) THEN 
                EXIT
              ENDIF

              PRIMS2=0._q
              VEL_=MATMUL(LATT_CUR%A,VEL_)
              CALL vel_bmat_product(T_INFO%NIONS,ICOORDINATES%NUMINTERNALS,IBDIM,VEL_,AVEL_,BMAT,PRIMS2)
              VEL_=MATMUL(TRANSPOSE(LATT_CUR%B),VEL_)

             ! IF (IO%IU0>=0) write(*,*) "errVEL:",ii,MAXVAL(ABS(PRIMS2))

              jj=0
              DO i=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                  jj=jj+1
                  IF (jj==1) err=ABS(PRIMS2(i)-INCREM(jj))
                  err1=ABS(PRIMS2(i)-INCREM(jj))
                  IF (err1 .GT. err) err=err1
                  RDUM=SUM(BMAT(i,:)/MASSES(:)*BMAT(i,:))
                  theta(i)=0.
                  IF (ABS(RDUM) .GT. 1e-6) theta(i)=(PRIMS2(i)-INCREM(jj))/RDUM/BMSCA     !!!
                
                  !c lattice dynamics
                  IF (IBDIM==3*T_INFO%NIONS+9) THEN
                    C_FORCE_L_=0._q
                    xposL=-theta(i)*BMAT(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)/MASSES(3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
                    CALL ONETOTHREE(3,C_FORCE_L_,xposL) 
                    C_FORCE_L_=TRANSPOSE(C_FORCE_L_) !!
                    AVEL_=AVEL_+2*C_FORCE_L_
                  ENDIF
             
                  xpos=-theta(i)*BMAT(i,1:3*T_INFO%NIONS)/MASSES(1:3*T_INFO%NIONS)
                  C_FORCE_=0._q
                  CALL ONETOTHREE(T_INFO%NIONS,C_FORCE_,xpos)

                  !C_FORCE_=MATMUL(TRANSPOSE(LATT_CUR%B),C_FORCE_)
                  C_FORCE_=MATMUL(TRANSPOSE(NewB),C_FORCE_)

                  !c remove force for frozen fractional coords 
                  DO j=1,T_INFO%NIONS
                    IF (.NOT. T_INFO%LSFOR(1,j)) C_FORCE_(1,j)=0._q
                    IF (.NOT. T_INFO%LSFOR(2,j)) C_FORCE_(2,j)=0._q
                    IF (.NOT. T_INFO%LSFOR(3,j)) C_FORCE_(3,j)=0._q
                  ENDDO

                  VEL_=VEL_+2*C_FORCE_(:,:)
               END IF
              ENDDO                     
            ENDDO
            V_FORCE=VEL_-DYN%VEL
            IF (IBDIM==3*T_INFO%NIONS+9) V_FORCE_L=AVEL_-LATT_CUR%AVEL

            !CALL XML_INCAR('shakeiter','I',ii,RDUM,CDUM,LDUM,CHARAC,1)
            IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A24,I4,A6)') 'RATTLE_vel converged in ',ii,' steps'

            !CALL XML_TAG("separator","gconstraints")
            IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A10)') '>Const_vel'
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                err=PRIMS2(i)
                IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,X,E13.6E2)') 'cv>',ICOORDINATES%COORDSTRUCT(i)%TAG,&
                &    err
              ENDIF
            ENDDO
          END SELECT
        END SUBROUTINE Rattle

        SUBROUTINE Rattle_vel(T_INFO,INFO,DYN,LATT_CUR, &
                IO,C_FORCE,C_FORCE_L,V_FORCE,V_FORCE_L,ICOORDINATES,g_io,IBDIM,AMASS,NewA,ECONST,iconst0,iconst2)
          TYPE(type_info) :: T_INFO
          TYPE(dynamics) :: DYN      
          TYPE (in_struct) :: IO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR     
          TYPE(coordstructure) :: ICOORDINATES     !c working and reference coords.
          TYPE(gadget_io) :: g_io
          REAL(q) ::xpos(3*T_INFO%NIONS)  ! positions in row format
          INTEGER :: i,j,k  ,ii ,jj,idummy                 ! indeces
          INTEGER :: ninfo,IBDIM
          INTEGER,SAVE :: counter=0
          REAL(q) :: BMAT_0(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: GMAT(ICOORDINATES%NUMINTERNALS) !c B matrix for working and reference coords.
          REAL(q) :: PRIMS1(ICOORDINATES%NUMINTERNALS),PRIMS2(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: gamma(ICOORDINATES%NUMINTERNALS),gamma0(ICOORDINATES%NUMINTERNALS),theta(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS),ZET_DET
          REAL(q),ALLOCATABLE,SAVE :: MASSES(:)
          REAL(q) :: C_FORCE(3,T_INFO%NIONS),C_FORCE_(3,T_INFO%NIONS)
          REAL(q) :: V_FORCE(3,T_INFO%NIONS),V_FORCE_L(3,3)
          REAL(q) :: POSION_(3,T_INFO%NIONS),POSIOC_(3,T_INFO%NIONS), VEL_(3,T_INFO%NIONS)
          REAL(q) :: err,err1
          REAL(q) :: UL,UT,FACT,ECONST
          INTEGER IDUM,IERR,N
          INTEGER :: iconst0,iconst2
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC
          REAL(q),SAVE,ALLOCATABLE :: INCREM(:)
          INTEGER,SAVE :: EQUI_REGIME=0
          LOGICAL,SAVE :: LBLUEOUT=.FALSE.
          REAL(q),SAVE :: BMTOL,BMTOLSOFT,BMSCA 
          INTEGER,SAVE :: BMITER
          REAL(q) :: AMASS,NewA(3,3),NewA_(3,3),NewB(3,3),C_FORCE_L(3,3),C_FORCE_L_(3,3),xposL(9)
          REAL(q) :: Avel_(3,3)
          REAL(q) :: BMAT_square(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS),BMAT_inv(IBDIM,ICOORDINATES%NUMINTERNALS)

          !c don't continue if no geometric constraints are defined
          IF (iconst0==0 .AND. iconst2==0) THEN
            RETURN
          ENDIF

          counter=counter+1
          UL=1E-10_q
          UT=DYN%POTIM*1E-15_q
          C_FORCE=0._q
          C_FORCE_L=0._q

          POSIOC_=DYN%POSIOC
          POSION_=DYN%POSION
          NewA_=NewA
          NewB=TRANSPOSE(NewA)
          CALL SVDINVERSE(NewB,3,idummy)
          
          IF (counter==1) THEN
            ICOORDINATES%COORDSTRUCT(:)%DVALUE=ICOORDINATES%COORDSTRUCT(:)%VALUE  !xx

            !c masses in amu
            ALLOCATE(MASSES(IBDIM))
            MASSES=0._q
            DO i=1,T_INFO%NIONS
              MASSES(3*i-2:3*i)=(T_INFO%POMASS(T_INFO%ITYP(i)))
            ENDDO
            IF (IBDIM==3*T_INFO%NIONS+9) THEN
              DO i=1,3
                DO j=1,3
                  MASSES(3*T_INFO%NIONS+3*(i-1)+j)=AMASS
                ENDDO
              ENDDO
            ENDIF

            ALLOCATE(INCREM(iconst0))
            INCREM=0._q
            !c read in some method-specific input parameters
            CALL READER_ICONST0(IO,g_io,iconst0,BMTOL,BMTOLSOFT,BMSCA,BMITER,LBLUEOUT,EQUI_REGIME,INCREM)
          END IF

          !c dont fix coordinates in the equilibration period
          IF (EQUI_REGIME .GE. counter) RETURN

          !c everything for old coordinates
          !c read constraints
          j=0
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
              j=j+1
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE+counter*INCREM(j) 
            ELSE IF  (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              j=j+1
              !PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%DVALUE
              PRIMS1(i)=ICOORDINATES%COORDSTRUCT(i)%VALUE
            ENDIF
          ENDDO

          !BMAT_0=0._q
          !CALL BMATRIX(T_INFO,DYN%POSIOC,DYN%POSIOC,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT_0,.TRUE.)
          !!CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT_0,.FALSE.)
          !c upper triangle inly
!           IF (IBDIM==3*T_INFO%NIONS+9) THEN
!             BMAT_0(:,3*T_INFO%NIONS+4)=0._q
!             BMAT_0(:,3*T_INFO%NIONS+7)=0._q
!             BMAT_0(:,3*T_INFO%NIONS+8)=0._q
!           ENDIF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          BMAT=0._q
       
          !c Jacobi matrix
          CALL BMATRIX(T_INFO,DYN%POSION,DYN%POSION,TRANSPOSE(NewA),ICOORDINATES,BMAT,.TRUE.)

          VEL_=DYN%VEL
          V_FORCE=0._q

          AVEL_=LATT_CUR%Avel
          V_FORCE_L=0._q

          err=0._q
          ii=0

          DO i=1,ICOORDINATES%NUMINTERNALS      
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              err=err+1
            ENDIF
          ENDDO

          !c start iterations to fullfill constraints for velocities
          DO ii=1,BMITER+1

            !c exit if no constraints are defined
            IF (ii==1 .AND. err<BMTOL) THEN
              !V_FORCE=VEL_-DYN%VEL
              EXIT
            ENDIF
           
            !c terminate with error message if SHAKE doesn't converge in BMITER steps
            IF (ii==BMITER+1) THEN  
              IF (IO%IU0>=0) write(*,*) 'Error: RATTLE_vel algorithm did not converge!'
              STOP
            END IF
           
            IF (err<BMTOL) THEN 
              !V_FORCE=VEL_-DYN%VEL
              EXIT
            ENDIF

            PRIMS2=0._q
            VEL_=MATMUL(LATT_CUR%A,VEL_)
            CALL vel_bmat_product(T_INFO%NIONS,ICOORDINATES%NUMINTERNALS,IBDIM,VEL_,AVEL_,BMAT,PRIMS2)
            VEL_=MATMUL(TRANSPOSE(LATT_CUR%B),VEL_)

           IF (IO%IU0>=0) write(*,*) "errVEL:",ii,MAXVAL(ABS(PRIMS2))
            !c upper triangle only
!             IF (IBDIM==3*T_INFO%NIONS+9) THEN
!               BMAT(:,3*T_INFO%NIONS+4)=0._q
!               BMAT(:,3*T_INFO%NIONS+7)=0._q
!               BMAT(:,3*T_INFO%NIONS+8)=0._q
!             ENDIF

            jj=0
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
                jj=jj+1
                IF (jj==1) err=ABS(PRIMS2(i)-INCREM(jj))
                err1=ABS(PRIMS2(i)-INCREM(jj))
                IF (err1 .GT. err) err=err1
                theta(i)=(PRIMS2(i)-INCREM(jj))/SUM(BMAT(i,:)/MASSES(:)*BMAT(i,:))/BMSCA     !!!
                
!                 !c lattice dynamics
!                 IF (IBDIM==3*T_INFO%NIONS+9) THEN
!                   C_FORCE_L_=0._q
!                   xposL=-gamma(i)*BMAT_0(i,3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)/MASSES(3*T_INFO%NIONS+1:3*T_INFO%NIONS+9)
!                   CALL ONETOTHREE(3,C_FORCE_L_,xposL) 
!                   C_FORCE_L_=TRANSPOSE(C_FORCE_L_) !!
!                   NewA(:,:)=NewA(:,:)+C_FORCE_L_
!                   NewB=TRANSPOSE(NewA)
!                   CALL SVDINVERSE(NewB,3,idummy)
!                   IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
!                     C_FORCE_L=C_FORCE_L+C_FORCE_L_
!                   ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
!                     C_FORCE_L=C_FORCE_L+2*C_FORCE_L_
!                   ENDIF
!                 ENDIF

                
                xpos=-theta(i)*BMAT(i,1:3*T_INFO%NIONS)/MASSES(1:3*T_INFO%NIONS)
                C_FORCE_=0._q
                CALL ONETOTHREE(T_INFO%NIONS,C_FORCE_,xpos)

                !C_FORCE_=MATMUL(TRANSPOSE(LATT_CUR%B),C_FORCE_)
                C_FORCE_=MATMUL(TRANSPOSE(NewB),C_FORCE_)
                VEL_=VEL_+2*C_FORCE_(:,:)

!                 IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
!                    C_FORCE=C_FORCE+C_FORCE_
!                 ELSE IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
!                    C_FORCE=C_FORCE+2*C_FORCE_
!                 ENDIF

              END IF
            ENDDO                     
            !!BMAT_0=BMAT
            !IF (IO%IU0>=0) write(IO%IU0,*) 'err',err
          ENDDO
          V_FORCE=VEL_-DYN%VEL

          !CALL XML_INCAR('shakeiter','I',ii,RDUM,CDUM,LDUM,CHARAC,1)
          IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A24,I4,A6)') 'RATTLE_vel converged in ',ii,' steps'

          !CALL XML_TAG("separator","gconstraints")
          IF (IO%IU6>0) write(g_io%REPORT,FMT='(/,2X,A10)') '>Const_vel'
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .OR. ICOORDINATES%COORDSTRUCT(i)%STATUS==2) THEN
              
              !err=theta(i)
              err=PRIMS2(i)
              IF (IO%IU6>0) write(g_io%REPORT,FMT='(3X,A3,X,A2,X,E13.6E2)') 'cv>',ICOORDINATES%COORDSTRUCT(i)%TAG,&
                  & err
            ENDIF
          ENDDO
          !CALL XML_CLOSE_TAG
        END SUBROUTINE Rattle_vel

        SUBROUTINE vel_bmat_product(n,m,ibdim,vel,avel,bmat,prod)
         INTEGER :: n,m,ibdim
         INTEGER :: i,j,indx
         REAL(q) :: vel(3,N),avel(3,3)
         REAL(q) :: bmat(M,ibdim)  
         REAL(q) :: prod(M)

         prod=0._q
         DO i=1,n 
           DO j=1,m
             indx=3*i-2
             prod(j)=prod(j)+vel(1,i)*bmat(j,indx)+vel(2,i)*bmat(j,indx+1)+vel(3,i)*bmat(j,indx+2)
           ENDDO
         ENDDO

         IF (ibdim==(3*N+9)) THEN
           DO j=1,m
             prod(j)=prod(j)+avel(1,1)*bmat(j,3*N+1)+avel(2,1)*bmat(j,3*N+2)+avel(3,1)*bmat(j,3*N+3)
             prod(j)=prod(j)+avel(1,2)*bmat(j,3*N+4)+avel(2,2)*bmat(j,3*N+5)+avel(3,2)*bmat(j,3*N+6)
             prod(j)=prod(j)+avel(1,3)*bmat(j,3*N+7)+avel(2,3)*bmat(j,3*N+8)+avel(3,3)*bmat(j,3*N+9)
           ENDDO
         ENDIF

        END SUBROUTINE vel_bmat_product

        SUBROUTINE make_Z(T_INFO,IBDIM,x,A,ICOORDINATES,MASSES,ZET)        
        !c compute the mass metric tensor (Z) for Bluemoon calculations
        !c due to ortogonalization of BMAT, Z is a diagonal matrix
        !c elements for redundant coordinates are zero
          TYPE(type_info) :: T_INFO
          TYPE(coordstructure) :: ICOORDINATES 
          INTEGER :: IBDIM,i
          REAL(q) :: x(3,T_INFO%NIONS)
          REAL(q) :: A(3,3)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: MASSES(IBDIM)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS)

          ZET=0._q
          
          IF (ICOORDINATES%NUMINTERNALS>0) THEN   
            BMAT=0._q      
            CALL BMATRIX(T_INFO,x,x,TRANSPOSE(A),ICOORDINATES,BMAT,.TRUE.) 
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS/=0) THEN
                BMAT(i,:)=0._q
              ELSE 
                BMAT(i,:)=BMAT(i,:)/SQRT(MASSES(:))
              ENDIF
            ENDDO
            CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)
          
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
                ZET(i)=SUM(BMAT(i,:)*BMAT(i,:))
                !ZET(i)=SUM(BMAT(i,:)*BMAT(i,:)/MASSES(:))
              END IF
            ENDDO
          ENDIF 

        END SUBROUTINE make_Z

       SUBROUTINE make_Z2(T_INFO,IBDIM,x,A,ICOORDINATES,MASSES,ZET)        
        !c compute the mass metric tensor (Z) for Bluemoon calculations
        !c elements for redundant coordinates are zero
          TYPE(type_info) :: T_INFO
          TYPE(coordstructure) :: ICOORDINATES 
          INTEGER :: IBDIM,i,j
          REAL(q) :: x(3,T_INFO%NIONS)
          REAL(q) :: A(3,3)
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,IBDIM)
          REAL(q) :: MASSES(IBDIM)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS)

          ZET=0._q
          
          IF (ICOORDINATES%NUMINTERNALS>0) THEN   
            BMAT=0._q      
            CALL BMATRIX(T_INFO,x,x,TRANSPOSE(A),ICOORDINATES,BMAT,.TRUE.) 
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS/=0) THEN
                BMAT(i,:)=0._q
              ELSE 
                BMAT(i,:)=BMAT(i,:)/SQRT(MASSES(:))
              ENDIF
            ENDDO
            
          
            DO i=1,ICOORDINATES%NUMINTERNALS
              DO j=1,ICOORDINATES%NUMINTERNALS
                IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0 .AND. ICOORDINATES%COORDSTRUCT(j)%STATUS==0) THEN
                  ZET(i,j)=SUM(BMAT(i,:)*BMAT(j,:))
                  !ZET(i)=SUM(BMAT(i,:)*BMAT(i,:)/MASSES(:))
                END IF
              ENDDO
            ENDDO
          ENDIF 
        END SUBROUTINE make_Z2

        FUNCTION z_det(ZET,ICOORDINATES,IO) RESULT(zdet)
        !c compute det(Z) for a diagonal matrix Z
          INTEGER :: i
          TYPE(coordstructure) :: ICOORDINATES   
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: zdet
          TYPE (in_struct) :: IO

          zdet=1._q
          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%STATUS==0) THEN
              IF (ABS(ZET(i))<1e-7) THEN
                IF (IO%IU0>0) THEN
                  WRITE(IO%IU0,*) 'Error LBLUEOUT: lineary dependent constraints,|z|=0'
                ENDIF
                STOP
              ENDIF
              !IF (ABS(ZET(i))>1e-5) THEN
              zdet=zdet*ZET(i)
              !ENDIF
            END IF
          ENDDO
        END FUNCTION z_det


        SUBROUTINE make_G(T_INFO,DYN,LATT_CUR,ICOORDINATES,MASSES,IBDIM,ZET_0,ZET_DET_0,GMAT,IO)
          TYPE(type_info) :: T_INFO
          TYPE(dynamics) :: DYN      
          TYPE(latt) :: LATT_CUR 
          TYPE(coordstructure) :: ICOORDINATES  
          REAL(q) :: BMAT(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS),BMAT_m(ICOORDINATES%NUMINTERNALS,3*T_INFO%NIONS)
          REAL(q) :: ZET_0(ICOORDINATES%NUMINTERNALS),ZET_0_inv(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: ZET(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: ZET2(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS),ZET2_inv(ICOORDINATES%NUMINTERNALS,ICOORDINATES%NUMINTERNALS)
          REAL(q) :: GMAT(ICOORDINATES%NUMINTERNALS)
          REAL(q) :: MASSES(3*T_INFO%NIONS)
          REAL(q) :: POSIOC_b(3,T_INFO%NIONS),POSIOC_cart(3,T_INFO%NIONS)
          REAL(q) :: STEP
          REAL(q) :: Zpd,Zmd,ZET_DET_0
          INTEGER :: IBDIM
          INTEGER :: i,j,k,l,kk,ii
          REAL(q) :: dZx(3*T_INFO%NIONS)
          TYPE (in_struct) :: IO



          STEP=1.E-4
          CALL DEAL_XYZ(T_INFO,DYN%POSIOC,DYN%POSIOC,LATT_CUR%A,ICOORDINATES)
          CALL make_Z2(T_INFO,IBDIM,DYN%POSIOC,LATT_CUR%A,ICOORDINATES,MASSES,ZET2)

          BMAT=0._q
          IF (ICOORDINATES%NUMINTERNALS>=1) THEN          
            CALL BMATRIX(T_INFO,DYN%POSIOC,DYN%POSIOC,TRANSPOSE(LATT_CUR%A),ICOORDINATES,BMAT,.TRUE.)  
            DO i=1,ICOORDINATES%NUMINTERNALS
              IF (ICOORDINATES%COORDSTRUCT(i)%STATUS/=0) THEN
                BMAT(i,:)=0._q
              ENDIF
            ENDDO
            !CALL ICONST_BMAT(BMAT,ICOORDINATES,T_INFO)
          ENDIF 

          BMAT_m=0._q
          DO i=1,ICOORDINATES%NUMINTERNALS
            BMAT_m(i,:)=BMAT(i,:)/MASSES(:)
          ENDDO
          
          POSIOC_b=DYN%POSIOC
          POSIOC_cart=MATMUL(LATT_CUR%A,POSIOC_b)
          GMAT=0._q
          
         
          DO ii=1,3*T_INFO%NIONS
            j=MOD(ii,3)
            if (j==0) j=3
            i=(ii-j)/3+1
           
            !c take a forward step
            DYN%POSIOC=POSIOC_cart
            DYN%POSIOC(j,i)=POSIOC_cart(j,i)+STEP
            DYN%POSIOC=MATMUL(TRANSPOSE(LATT_CUR%B),DYN%POSIOC)
            CALL DEAL_XYZ(T_INFO,DYN%POSIOC,DYN%POSIOC,LATT_CUR%A,ICOORDINATES)
            CALL make_Z(T_INFO,IBDIM,DYN%POSIOC,LATT_CUR%A,ICOORDINATES,MASSES,ZET)   
            Zpd=z_det(ZET,ICOORDINATES,IO)    
            
            !c take a backward step
            DYN%POSIOC=POSIOC_cart
            DYN%POSIOC(j,i)=POSIOC_cart(j,i)-STEP
            DYN%POSIOC=MATMUL(TRANSPOSE(LATT_CUR%B),DYN%POSIOC)
            CALL DEAL_XYZ(T_INFO,DYN%POSIOC,DYN%POSIOC,LATT_CUR%A,ICOORDINATES)
            CALL make_Z(T_INFO,IBDIM,DYN%POSIOC,LATT_CUR%A,ICOORDINATES,MASSES,ZET)
            Zmd=z_det(ZET,ICOORDINATES,IO)    
            dZx(ii)=(Zpd-Zmd)/2/STEP
          ENDDO

          GMAT=MATMUL(BMAT_m,dZx)

          ZET2_inv=ZET2
          CALL SVDINVERSE(ZET2_inv,ICOORDINATES%NUMINTERNALS,i)
          GMAT=MATMUL(ZET2_inv,GMAT)

          !DO i=1,ICOORDINATES%NUMINTERNALS
          !  IF (ABS(ZET_0(i))>1e-5) THEN
          !    GMAT(i)=GMAT(i)/ZET_0(i)
          !  ELSE 
          !   !c add error handling!!!
          !    GMAT(i)=0._q
          !  ENDIF
          !ENDDO

          GMAT=GMAT/2/ZET_DET_0

          !c restore the original values 
          DYN%POSIOC=POSIOC_b
          CALL DEAL_XYZ(T_INFO,DYN%POSIOC,DYN%POSIOC,LATT_CUR%A,ICOORDINATES)
        END SUBROUTINE make_G
 
      END MODULE dynconstr
