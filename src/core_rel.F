#include "symbol.inc"
!#define testout
!#define bloechl
!***********************************************************************
! MODULE PP_DATA
!
! This module contains all the routines needed to setup the 
! pseudopotential datastructure for the relaxed core method    
!
!***********************************************************************
      MODULE pp_data
      
      USE prec
      USE radial
      USE pseudo_struct
      
      IMPLICIT NONE

      TYPE (potcar), TARGET, ALLOCATABLE, SAVE :: RCP(:)
      
      INTEGER, SAVE :: NIONS_GLOBAL
      INTEGER, SAVE :: NIONS_LOCAL,NIRRED_IONS
      INTEGER, ALLOCATABLE, SAVE :: MAP2GLOBAL(:)
      INTEGER, ALLOCATABLE, SAVE :: MAP2TYPE(:),MAP2TYPE_ORIG(:)
      INTEGER, ALLOCATABLE, SAVE :: TYPE2ION(:)

      LOGICAL, SAVE :: LCLSHIFT
      
      LOGICAL, SAVE :: LSPAWN=.FALSE.
      LOGICAL, PRIVATE, SAVE :: LSPAWNDONE=.FALSE.
      
      CONTAINS

      
!*********************** SUBROUTINE SPAWN_PP ****************************
!
! This subroutine determines the set of ions (on the local node) that
! are unrelated by symmetry. It takes only those types into account that
! are set to be treated with the relaxed core method, i.e., for which
! INCAR flag LRCTYPE is set to true. (default LRCTYPE=.TRUE. for all)
!
! For each ion in the aforemention set a entry in the RCP pseudopotential
! data structure is created.
!
! In addition several variables and maps in the pp_data module are set:
!
! NIONS_GLOBAL : total number of ions
! NIONS_LOCAL  : number of ions on local node
! NIRRED_IONS  : size of the set of "irreducible" ions
!
! MAP2GLOBAL   : maps a local ion id number to its global counterpart
! MAP2TYPE_ORIG: maps a local ion id number to the original type
!                designation (in accordance with the POSCAR information)
! MAP2TYPE     : maps the local ion id number to the type designation
!                in the RCP datastructure
! TYPE2ION     : maps a type designation in the RCP datastructure to
!                the local id number of the irreducible ion that will
!                "represent" this type 
!
!************************************************************************
      SUBROUTINE SPAWN_PP(T_INFO,SYMM,WDES,P,IO)
      USE base
      USE prec
      USE base
      USE wave
      USE poscar
      USE pseudo
      
      IMPLICIT NONE
      
      TYPE (type_info) T_INFO
      TYPE (symmetry) SYMM
      TYPE (wavedes) WDES
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (in_struct) IO
      
      INTEGER I,NI,NT,NIP,NTP
      INTEGER NI_LOC,NI_GLB,NIP_LOC,NIP_GLB,NI_START
      INTEGER IROT,ITRANS
      
      INTEGER, ALLOCATABLE :: IRRED_IONS(:)
      LOGICAL, ALLOCATABLE :: LSPWNTYPE(:),LSPWNION(:)
      
      INTEGER IU0,IU5
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM

      ! quick return if already done
      IF (LSPAWNDONE) RETURN
      IU0=IO%IU0; IU5=IO%IU5
!========================================================================
! read relevant stuff from INCAR
!========================================================================
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

! LRELCOR: do we want to perform a relaxed core calculation?
! (N.B.: if not, there is no need to spawn)
      LSPAWN=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IU5,'LRELCOR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LSPAWN,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRELCOR'' from file INCAR.'
         GOTO 150
      ENDIF

! LRCTYPE: which types will be treated with the relaxed core method
! and will therefor have to be spawned      
      ALLOCATE(LSPWNTYPE(T_INFO%NTYP))
      LSPWNTYPE=.TRUE.
      IF (LSPAWN) THEN
      ! read which types are submitted to core relaxation
         CALL RDATAB(LOPEN,INCAR,IU5,'LRCTYPE','=','#',';','L', &
        &            IDUM,RDUM,CDUM,LSPWNTYPE,CHARAC,N,T_INFO%NTYP,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<T_INFO%NTYP))) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LRCTYPE'' from file INCAR.'
            GOTO 150
         ENDIF   
      ENDIF
      CALL XML_INCAR('LRCTYPE','L',IDUM,RDUM,CDUM,LSPWNTYPE,CHARAC,N)

      ! finished reading from INCAR
      CLOSE(IU5)
!========================================================================
! Now the work starts
!========================================================================

! store the total (global) number of ions
      NIONS_GLOBAL=T_INFO%NIONS
! get me the number of ions on the local node
      NIONS_LOCAL=0
      DO NI=1,T_INFO%NIONS
         NT=T_INFO%ITYP(NI)
         NIP=NI_LOCAL(NI,WDES%COMM_INB)
         IF ( NIP/=0 ) NIONS_LOCAL=NIONS_LOCAL+1
      ENDDO
! set up a map (MAP2GLOBAL) between the local ion number and its global 
! counterpart, and a map (MAP2TYPE_ORIG) between the local ion number and
! its original type designation, and set whether this ion is a candidate for 
! spawning (LSPWNION).
      ALLOCATE(MAP2GLOBAL(NIONS_LOCAL),MAP2TYPE_ORIG(NIONS_LOCAL), &
     &            LSPWNION(NIONS_LOCAL))

      MAP2GLOBAL   =-1
      MAP2TYPE_ORIG=-1
      LSPWNION     =.FALSE.

      DO NI=1,T_INFO%NIONS
         NT=T_INFO%ITYP(NI)
         NIP=NI_LOCAL(NI,WDES%COMM_INB)
         IF ( NIP/=0 .AND. ASSOCIATED(P(NT)%QPAW) ) THEN
            MAP2GLOBAL(NIP)=NI
            MAP2TYPE_ORIG(NIP)=NT
            LSPWNION(NIP)=LSPWNTYPE(NT)
         ENDIF
      ENDDO
!========================================================================
      IF (LSPAWN) THEN
!========================================================================

! determine the local ions unrelated by symmetry (IRRED_IONS)
! and create a map (MAP2TYPE) that connect the local ion id number to
! a type number in the spawned pseudopotental datastructure (RCP)
      ALLOCATE(IRRED_IONS(NIONS_LOCAL),MAP2TYPE(NIONS_LOCAL))
      IRRED_IONS=0
      MAP2TYPE=0
      ! get the local index of the first ion to be spawned
      DO NI_START=1,NIONS_LOCAL
         IF (LSPWNION(NI_START)) EXIT
      ENDDO
      ! are there any?
      IF (NI_START>NIONS_LOCAL) THEN
         WRITE(*,*) 'SPAWN_PP: There seems to be no work for me'
         LSPAWNDONE=.TRUE.
         NIRRED_IONS=0
         RETURN
      ENDIF
      ! initialize first elements in the arrays
      NIRRED_IONS=1
      IRRED_IONS(1)=NI_START      
      MAP2TYPE(NI_START)=1
      ! now start the filling procedure
      ion : DO NI_LOC=NI_START,NIONS_LOCAL
         ! check whether this ion should be considered
         IF (.NOT.LSPWNION(NI_LOC)) THEN
            ! this ion is mapped to type zero, which does not
            ! exist. Alternatively one could consider mapping
            ! to the type in the original PP data structure
            MAP2TYPE(NI_LOC)=0
            CYCLE ion
         ENDIF
         NI_GLB=MAP2GLOBAL(NI_LOC)
         ! check whether NI_LOC is related by symmetry
         ! to an entry in the list IRRED_IONS
         DO IROT=1,SYMM%NROT
         DO ITRANS=1,SYMM%NPTRANS
            ! apply symmetry operation
            NIP_GLB=SYMM%ROTMAP(NI_GLB,IROT,ITRANS)
            ! get me the local index of this ion
            NIP_LOC=NI_LOCAL(NIP_GLB,WDES%COMM_INB)
            ! if not on local node cycle   
            IF (NIP_LOC==0) CYCLE
            ! test if this "type" is already in the list
            DO I=1,NIRRED_IONS
               IF (NIP_LOC==IRRED_IONS(I)) THEN
                  ! the type designation of this ion will be the
                  ! same as the type designation of its symmetry
                  ! equivalent companion
                  MAP2TYPE(NI_LOC)=MAP2TYPE(IRRED_IONS(I))
                  CYCLE ion
               ENDIF 
            ENDDO            
         ENDDO
         ENDDO
         ! if not in the list then add it
         NIRRED_IONS=NIRRED_IONS+1
         IRRED_IONS(NIRRED_IONS)=NI_LOC
         ! and link this new type to the local ion id number
         MAP2TYPE(NI_LOC)=NIRRED_IONS
      ENDDO ion

! create a map (TYPE2ION) that relates the "types" to the
! local id number of the ion that will represent it
      ALLOCATE(TYPE2ION(NIRRED_IONS))
      DO NT=1,NIRRED_IONS
         TYPE2ION(NT)=IRRED_IONS(NT)
      ENDDO
            
#ifdef testout
      WRITE(*,'(A,I4)') 'number of types',NIRRED_IONS
      WRITE(*,*)
      WRITE(*,*) 'type map to ion'
      DO NT=1,NIRRED_IONS
         WRITE(*,'(A,I4,A,I4)') 'type',NT,'  maps to ion',TYPE2ION(NT)
      ENDDO
      WRITE(*,*)
      WRITE(*,*) 'ion map to type'
      DO NI=1,NIONS_LOCAL
         WRITE(*,'(A,I4,A,I4)') ' ion',NI,' maps to type',MAP2TYPE(NI)
      ENDDO
#endif

!========================================================================
! now spawn the pseudopotentials!
! N.B. NIRRED_IONS now sort plays the role T_INFO%NTYP used to have,
! i.e. each symmetry inequivalent atom is now a "type" on its own in the
! RCP datastructure. 
! MAP2TYPE relates each local ion to its new "type" number
!========================================================================
      ALLOCATE(RCP(NIRRED_IONS))

      DO NTP=1,NIRRED_IONS
         NI_LOC=IRRED_IONS(NTP)
         NI_GLB=MAP2GLOBAL(NI_LOC)
         NT=T_INFO%ITYP(NI_GLB)         
         ! copy the information from P(NT) into RCP(NTP)
         RCP(NTP)=P(NT)
         ! nullify pointers to information that will be
         ! changed during the core relaxation
         NULLIFY(RCP(NTP)%DION,RCP(NTP)%QION,  &
        &        RCP(NTP)%QTOT,RCP(NTP)%QPAW,  &
        &        RCP(NTP)%QDEP,RCP(NTP)%E,      &
        &        RCP(NTP)%POTAE,RCP(NTP)%POTPS,RCP(NTP)%POTPSC, &
        &        RCP(NTP)%RHOAE,RCP(NTP)%RHOPS, &
        &        RCP(NTP)%WAE,RCP(NTP)%WPS,RCP(NTP)%AUG)
         ! allocate anew
         ALLOCATE(RCP(NTP)%DION(SIZE(P(NT)%DION,1),SIZE(P(NT)%DION,2)), &
        &         RCP(NTP)%QION(SIZE(P(NT)%QION,1),SIZE(P(NT)%QION,2)), &
        &         RCP(NTP)%QTOT(SIZE(P(NT)%QTOT,1),SIZE(P(NT)%QTOT,2)), &
        &         RCP(NTP)%QPAW(SIZE(P(NT)%QPAW,1),SIZE(P(NT)%QPAW,2),0:SIZE(P(NT)%QPAW,3)-1), &
        &         RCP(NTP)%QDEP(SIZE(P(NT)%QDEP,1),SIZE(P(NT)%QDEP,2),0:SIZE(P(NT)%QDEP,3)-1), &
        &         RCP(NTP)%E(P(NT)%LMAX), &
        &         RCP(NTP)%POTAE(SIZE(P(NT)%POTAE)), &
        &         RCP(NTP)%POTPS(SIZE(P(NT)%POTPS)), &
        &         RCP(NTP)%POTPSC(SIZE(P(NT)%POTPSC)), &
        &         RCP(NTP)%RHOAE(SIZE(P(NT)%RHOAE)), &
        &         RCP(NTP)%RHOPS(SIZE(P(NT)%RHOPS)), &
        &         RCP(NTP)%WAE(SIZE(P(NT)%WAE,1),SIZE(P(NT)%WAE,2)), &
        &         RCP(NTP)%WPS(SIZE(P(NT)%WPS,1),SIZE(P(NT)%WPS,2)), &
        &         RCP(NTP)%AUG(SIZE(P(NT)%AUG,1),0:SIZE(P(NT)%AUG,2)-1))
         ! and fill them
         RCP(NTP)%DION     =P(NT)%DION
         RCP(NTP)%QION     =P(NT)%QION
         RCP(NTP)%QTOT     =P(NT)%QTOT
         RCP(NTP)%QPAW     =P(NT)%QPAW
         RCP(NTP)%QDEP     =P(NT)%QDEP
         RCP(NTP)%E        =P(NT)%E
         RCP(NTP)%POTAE    =P(NT)%POTAE
         RCP(NTP)%POTPS    =P(NT)%POTPS
         RCP(NTP)%POTPSC   =P(NT)%POTPSC
         RCP(NTP)%RHOAE    =P(NT)%RHOAE
         RCP(NTP)%RHOPS    =P(NT)%RHOPS
         RCP(NTP)%WAE      =P(NT)%WAE
         RCP(NTP)%WPS      =P(NT)%WPS
         RCP(NTP)%AUG      =P(NT)%AUG
         ! N.B. other structures will be allocated 
         ! and filled during initialization of the
         ! relaxed core method. In essence the stuff
         ! that is allocated here is the information
         ! that has to be synchronized across all nodes
         ! lateron, and as such must be allocated on
         ! all nodes.
      ENDDO

      DEALLOCATE(IRRED_IONS,LSPWNTYPE,LSPWNION)
!========================================================================      
      ENDIF
!========================================================================
      LSPAWNDONE=.TRUE.
      RETURN
      
  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP
      
      END SUBROUTINE SPAWN_PP


!*********************** FUNCTION SET_PP_POINTER ************************
!
! This function is used only in the core relaxation module. It takes 
! the local id number of an ion and returns a pointer to the relevant 
! pseudopotential information, in addition it sets the logical LCLSHIFT 
! to true if this ion is a candidate for a final state core level shift 
! calculation.
!
!************************************************************************
      FUNCTION SET_PP_POINTER(NI)
      USE prec
      USE pseudo
      USE cl
      IMPLICIT NONE
      TYPE (potcar), POINTER :: SET_PP_POINTER
      INTEGER :: NI      ! ion index
      IF (LSPAWN.AND.LSPAWN_PP_DONE().AND.MAP2TYPE(NI)/=0) THEN
         SET_PP_POINTER=>RCP(MAP2TYPE(NI))
         LCLSHIFT=(MAP2TYPE_ORIG(NI)==NT_CL)
      ELSE
         WRITE(*,*) 'SET_PP_POINTER: Error, called with LSPAWN=',LSPAWN
         WRITE(*,*) '                           LSPAWN_PP_DONE=',LSPAWN_PP_DONE()
         WRITE(*,'(A,I4,A,I4)') &
        &           '                              ION => TYPE=',NI,' =>',MAP2TYPE(NI)             
         STOP
      ENDIF
      END FUNCTION SET_PP_POINTER


!*********************** FUNCTION LSPAWN_PP_DONE ************************
!
! The value of this function is true when the pseudopotential data
! structure has been spawned. 
!
!************************************************************************
      FUNCTION LSPAWN_PP_DONE()
      IMPLICIT NONE
      LOGICAL LSPAWN_PP_DONE
      LSPAWN_PP_DONE=LSPAWNDONE
      END FUNCTION LSPAWN_PP_DONE

      END MODULE pp_data

      
!*********************** FUNCTION PP_POINTER ****************************
!
! This function takes the local id number and type designation of an ion 
! and returns a pointer to the relevant pseudopotential information, an
! interface to it is defined in pseudo.F
!
!************************************************************************      
      FUNCTION PP_POINTER(P,NI,NT)
      USE pp_data
      IMPLICIT NONE 
      TYPE (potcar),TARGET :: P(:)
      TYPE (potcar), POINTER :: PP_POINTER
      INTEGER :: NI      ! ion index
      INTEGER :: NT      ! type index
      IF (LSPAWN.AND.LSPAWN_PP_DONE()) THEN
         IF (MAP2TYPE(NI)/=0) THEN
            PP_POINTER=>RCP(MAP2TYPE(NI))
         ELSE
            PP_POINTER=>P(NT)
         ENDIF
      ELSE
         PP_POINTER=>P(NT)
      ENDIF
      END FUNCTION PP_POINTER


!************************************************************************
! MODULE CORE_REL
!
! This module contains the subroutines that enable VASP to perform
! a relaxation of the core electronic charge density
!
!************************************************************************            
      MODULE core_rel
      
      USE prec
      USE pseudo
      USE radial
      USE mpimy
      USE pp_data
      USE vaspxml

      IMPLICIT NONE
      
      PRIVATE :: POT,TRANS_DLM_RC,SET_PAW_AUG_RC,CHARGE
      
! arrays for mixing the onsite valence charge densities
      INTEGER, SAVE :: NMIX_ONE_CENTRE
      REAL(q), ALLOCATABLE, SAVE :: RHO_ONE_CENTRE(:),RHO_ONE_CENTRE_LAST(:)
! communication in parallel mode      
      TYPE (communic), PRIVATE, SAVE :: COMM_INTER
      INTEGER, SAVE, PRIVATE :: NCPU,NODE_ME,IONODE
      INTEGER, SAVE, PRIVATE :: COMM_BUFF_SIZE
! number of types      
      INTEGER, SAVE, PRIVATE :: NTYP
! distribution over nodes
      LOGICAL, ALLOCATABLE, SAVE :: DO_TYPE_LOCAL(:)
! unit for writing to OUTCAR
      INTEGER, SAVE, PRIVATE :: IOUT
! maximum number of core states (30 should be enough)
      INTEGER, PARAMETER, PRIVATE :: MAX_NUM_CORE_STATES=30
! control variables      
      LOGICAL, SAVE, PRIVATE :: LRELCOR
      LOGICAL, SAVE, PRIVATE :: LMIMICFC
      LOGICAL, SAVE, PRIVATE :: LMATCHRW
      LOGICAL, SAVE, PRIVATE :: LADAPTIVE
      LOGICAL, SAVE, PRIVATE :: LONLYSEMICORE
      CHARACTER(3), SAVE, PRIVATE :: RADEQ
      INTEGER, SAVE, PRIVATE :: IDELAY,ISTRIDE
      INTEGER, SAVE, PRIVATE :: IPRINT,NCALLED
      REAL(q), SAVE, PRIVATE :: SCALE_METRIC
      REAL(q), SAVE, PRIVATE :: E_SEMICORE
! logarithmic derivative output
      INTEGER, PARAMETER, PRIVATE :: NPOINTS=1000
! initialization done already?     
      LOGICAL, PRIVATE, SAVE :: LINITDONE=.FALSE.
      
      CONTAINS
      
      
!*********************** SUBROUTINE INIT_CORE_REL ***********************      
!
! This subroutine performs the initialization of the core relaxation
! procedure. It is called in ELMIN, at the start of the electronic
! minimization. It performs the following tasks:
!
! 1) read relevant information from INCAR
! 2) allocate the additional pseudopotential data structure 
! 3) calculate the expansion coefficients that determine the
!    partial waves stored on the POTCAR file
! 4) calculate the core contribution to the total energy
!    for the atomic valence configuration
! 5) correctly set the AE core charge density PP%RHOAE and the
!    local pseudo-core-potential (V_H[\tilde{n}_Zc]) in case we
!    perform a core level shift calculation
! 
!************************************************************************            
      SUBROUTINE INIT_CORE_REL(WDES,CRHODE,IU0,IU5,IU6)

      USE prec
      USE base
      USE poscar
      USE pseudo
      USE wave
      USE us
      USE cl
      USE paw
      USE ini
      USE vaspxml

      IMPLICIT NONE

      TYPE(wavedes) WDES
      TYPE(potcar), POINTER :: PP

      INTEGER IU0,IU5,IU6
      OVERLAP CRHODE(:,:,:,:)
      
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      REAL(q), ALLOCATABLE :: RHOLM(:)
      
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM
      
      INTEGER NMIX_ELEMENTS
      INTEGER LYMAX,LMMAX
      INTEGER LMDIM,ISPIN
      INTEGER I,NI,NIP,NT,NI_,CHANNEL,IS
      INTEGER, EXTERNAL :: MAXL_AUG
      REAL(q) F,FDER, RWIGS
      
      ! quick return if possible
      IF (LINITDONE) RETURN

!========================================================================
! read relevant stuff from INCAR
!========================================================================
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

! LRELCOR: do we want to perform core relaxation?
      LRELCOR=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IU5,'LRELCOR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRELCOR,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRELCOR'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LRELCOR','L',IDUM,RDUM,CDUM,LRELCOR,CHARAC,N)
      ! quick return if we do not perform core relaxation
      IF (.NOT.(LRELCOR)) RETURN

! LMIMICFC: do we want to mimic a frozen core calculation?
! Only the core contributions to the total energy will be
! calculated
      LMIMICFC=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IU5,'LMIMICFC','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LMIMICFC,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LMIMICFC'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LMIMICFC','L',IDUM,RDUM,CDUM,LMIMICFC,CHARAC,N)

! LUSERWIGS: do we want to match the PS partial waves at PP%RWIGS?
! If not, they are matched at the PAW sphere boundary.
      LMATCHRW=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LMATCHRW','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LMATCHRW,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LMATCHRW'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LMATCHRW','L',IDUM,RDUM,CDUM,LMATCHRW,CHARAC,N)
      

! RADEQ: which radial equation do we use? 
! Scalar relativistic, Scalar relativistic + SOC, or Dirac. 
      CALL RDATAB(LOPEN,INCAR,IU5,'RADEQ','=','#',';','S', &
     &            IDUM,RDUM,CDUM,LDUM,INPLIN,N,40,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RADEQ'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RADEQ','S',IDUM,RDUM,CDUM,LDUM,INPLIN,N)
      CALL STRIP(INPLIN,N,'L')
      IF (INPLIN(1:2)=='DI'.OR.INPLIN(1:2)=='Di'.OR.INPLIN(1:2)=='dI'.OR.INPLIN(1:2)=='di') THEN
         RADEQ='DIR'
      ELSEIF (INPLIN(1:2)=='SO'.OR.INPLIN(1:2)=='So'.OR.INPLIN(1:2)=='sO'.OR.INPLIN(1:2)=='so') THEN
         RADEQ='SOC'
      ELSE
         RADEQ='SCA'
      ENDIF

! RCREP: write a detailed report after every RCREP-th step
! Concerns the LDERAE*, LDERPS*, and RELCOR* files
      IPRINT=0; NCALLED=0
      CALL RDATAB(LOPEN,INCAR,IU5,'RCREP','=','#',';','I', &
     &            IPRINT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RCREP'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RCREP','I',IPRINT,RDUM,CDUM,LDUM,CHARAC,N)

! RCNDL: wait RCNDL steps into the SCF procedure before starting
! updating the core charge density
      IDELAY=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'RCNDL','=','#',';','I', &
     &            IDELAY,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RCNDL'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RCNDL','I',IDELAY,RDUM,CDUM,LDUM,CHARAC,N)

! RCSTRD: perform a relaxtion of the core every RCSTRD-th step in
! the SCF procedure
      ISTRIDE=1
      CALL RDATAB(LOPEN,INCAR,IU5,'RCSTRD','=','#',';','I', &
     &            ISTRIDE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RCSTRD'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RCSTRD','I',ISTRIDE,RDUM,CDUM,LDUM,CHARAC,N)

! RCMIX: scale the onsite charge densities that are passed to the (Broyden) 
! mixer with RCMIX
      SCALE_METRIC=0.1_q
      CALL RDATAB(LOPEN,INCAR,IU5,'RCMIX','=','#',';','F', &
     &            IDUM,SCALE_METRIC,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RCMIX'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RCMIX','F',IDUM,SCALE_METRIC,CDUM,LDUM,CHARAC,N)

! ESEMICORE: defines the semicore states (Default: -2 Ry)
      E_SEMICORE=-27.211652
      CALL RDATAB(LOPEN,INCAR,IU5,'ESEMICORE','=','#',';','F', &
     &            IDUM,E_SEMICORE,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ESEMICORE'' from file INCAR.'
         GOTO 150
      ENDIF
!     CALL XML_INCAR('ESEMICORE','F',IDUM,E_SEMICORE,CDUM,LDUM,CHARAC,N)

! LADAPTIVE: do we want to adapt the linearization energies to iron
! out divergencies and extra nodes in the AE partial waves
      LADAPTIVE=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LADAPTELIN','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LADAPTIVE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LADAPTELIN'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LADAPTELIN','L',IDUM,RDUM,CDUM,LADAPTIVE,CHARAC,N)

! LONLYSEMICORE: do we want to correct only the linearization energies
! of the semicore states
      LONLYSEMICORE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LONLYSEMICORE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LONLYSEMICORE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LONLYSEMICORE'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LONLYSEMICORE','L',IDUM,RDUM,CDUM,LONLYSEMICORE,CHARAC,N)
      ! finished reading from INCAR
      CLOSE(IU5)

!========================================================================
! Some checks
!========================================================================
      ! check if the pseudopotential data has been spawned
      IF (.NOT.LSPAWN_PP_DONE()) THEN
         WRITE(*,*) 'INIT_CORE_REL: pseudopotential data was not spawned'
         WRITE(*,*) '           ... exiting, sorry!'
         STOP
      ENDIF
      ! and if there are any types for which the core has to be relaxed
      IF (NIRRED_IONS==0) THEN
         WRITE(*,*) 'INIT_CORE_REL: There seems to be no core relaxation demanded'
         LINITDONE=.TRUE.
         LRELCOR=.FALSE.
         RETURN
      ENDIF
      ! check if each node holds all ions
      IF (NIONS_LOCAL/=NIONS_GLOBAL) THEN
         WRITE(*,*) 'INIT_CORE_REL: distribution of ions over nodes not implemented,'
         WRITE(*,*) '               i.e., NPAR has to equal the number of processors'
         WRITE(*,*) '           ... exiting, sorry!'
         STOP
      ENDIF

!========================================================================
! Start initializing
!========================================================================

! Store some stuff in the module variables
      ! store unit for writing to OUTCAR
      IOUT=IU6
      ! store number of irreducible types
      NTYP=NIRRED_IONS
      ! store stuff for communication on parallel machines
      NCPU=1
      NODE_ME=0
#ifdef MPI
      NCPU=WDES%COMM%NCPU
      NODE_ME=WDES%COMM%NODE_ME
      IONODE=WDES%COMM%IONODE
      COMM_INTER=WDES%COMM_INTER
#endif

! write type information to OUTCAR
      io_begin
      WRITE(IOUT,*)
      WRITE(IOUT,*) 'Initialize core relaxation'
      WRITE(IOUT,*) '--------------------------'
      WRITE(IOUT,*)
      WRITE(IOUT,'(A,I4)') 'number of types:',NIRRED_IONS
      WRITE(IOUT,*)
      WRITE(IOUT,'(A9)') 'type=>ion'
      DO NT=1,NIRRED_IONS
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)         
         WRITE(IOUT,'(A4,I4,A2,A2,A14,I4)') &
        &   'type',NT,' (',PP%ELEMENT,')  maps to ion',TYPE2ION(NT)
      ENDDO
      WRITE(IOUT,*)
      WRITE(IOUT,'(A10)') ' ion=>type'
      DO NI=1,NIONS_LOCAL
         IF (MAP2TYPE(NI)==0) CYCLE
         PP=>SET_PP_POINTER(NI)
         WRITE(IOUT,'(A4,I4,A2,A2,A14,I4)') &
        &   ' ion',NI,' (',PP%ELEMENT,') maps to type',MAP2TYPE(NI)
      ENDDO
      io_end

! distribute the irreducible types over the nodes within 
! COMM%INTER when running on multiple processors
      ALLOCATE(DO_TYPE_LOCAL(NTYP))
      DO_TYPE_LOCAL=.TRUE.      
#ifdef MPI   
      DO NT=1,NTYP
         DO_TYPE_LOCAL(NT)= &
        &   (MOD(NT+COMM_INTER%NODE_ME,COMM_INTER%NCPU)==0)
      ENDDO
#endif

! allocate additional data structure 
      NMIX_ELEMENTS=0
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         IF (DO_TYPE_LOCAL(NT)) THEN
            ! for types that are treated locally
            NMIX_ELEMENTS=NMIX_ELEMENTS+2*PP%R%NMAX
            ALLOCATE(PP%C(PP%LMAX,3), &
           &         PP%BETA(PP%R%NMAX,PP%LMAX), &
           &         PP%RHOAE00(PP%R%NMAX), &
           &         PP%RHOPS00(PP%R%NMAX), &
           &         PP%RHOPSPW(PP%R%NMAX), &
           &         PP%POTAEC(PP%R%NMAX), &
           &         PP%V00_AE(PP%R%NMAX), &
           &         PP%WKINAE(PP%R%NMAX,PP%LMAX), &
           &         PP%V00_PS(PP%R%NMAX), &
           &         PP%WKINPS(PP%R%NMAX,PP%LMAX), &
           &         PP%DIJ(PP%LMAX,PP%LMAX), &
           &         PP%QIJ(PP%LMAX,PP%LMAX), &
           &         PP%CLEV(MAX_NUM_CORE_STATES), &
           &         PP%E_ORIG(PP%LMAX))
            ! initialize new stuff to zero            
            PP%C=0
            PP%BETA=0
            PP%RHOAE00=0; PP%RHOPS00=0; PP%RHOPSPW=0
            PP%V00_AE=0; PP%WKINAE=0
            PP%V00_PS=0; PP%WKINPS=0
            PP%DIJ=0; PP%QIJ=0
            PP%E_ORIG=0
            ! fill projectors
            chann: DO CHANNEL=1,PP%LMAX
               IF (PP%LPS(CHANNEL)>PP%LMAX_CALC) CYCLE chann
               ! transfer projectors onto radial grid
               ! n.b. we store r*P(r)
               grid: DO I=1,PP%R%NMAX
                  IF (PP%R%R(I)>PP%PSPRNL(NPSRNL,1,1)) EXIT grid
                  CALL SPLVAL(PP%R%R(I),F,FDER,PP%PSPRNL(1,1,CHANNEL),NPSRNL,NPSRNL)                  
                  PP%BETA(I,CHANNEL)=F*PP%R%R(I)
               ENDDO grid
            ENDDO chann
            ! if we want to match the AE- and PS-partial waves at PP%RWIGS
            ! set it to the firstgrid point where all projectors are zero
            IF (LUSERWIGS()) THEN
               RWIGS=0._q
               DO CHANNEL=1,PP%LMAX
                  DO I=PP%R%NMAX,1,-1
                     IF (PP%BETA(I,CHANNEL)/=0._q) EXIT
                  ENDDO
                  RWIGS=MAX(RWIGS,PP%R%R(I+1))
               ENDDO
               PP%RWIGS=RWIGS
#ifdef testout
               WRITE(*,'(A,I4,A,F14.7)') 'INIT_CORE_REL: PP%RWIGS of type ',NT,' set to:',RWIGS
#endif            
            ENDIF
         ELSE
            ! allocate for all types (eases io)
            ALLOCATE(PP%C(PP%LMAX,3),PP%CLEV(MAX_NUM_CORE_STATES))
            PP%C=0; PP%CLEV=0
         ENDIF
      ENDDO

! allocate the arrays used to mix the onsite valence charge densities
      ALLOCATE(RHO_ONE_CENTRE(NMIX_ELEMENTS), &
     &   RHO_ONE_CENTRE_LAST(NMIX_ELEMENTS))
! and initialize
      NMIX_ONE_CENTRE=NMIX_ELEMENTS
      RHO_ONE_CENTRE=0; RHO_ONE_CENTRE_LAST=0
      
! initialize communication/synchronization of PP data
      CALL INIT_SYNC_PP()
      
!========================================================================
! allocation finished, now the real work starts
!========================================================================

! get dimensions of CRHODE, and perform some consistency checks
      LMDIM=SIZE(CRHODE,1); ISPIN=SIZE(CRHODE,4)
      IF (ISPIN/=1.AND.ISPIN/=2.AND.ISPIN/=4) THEN
         WRITE(*,*) 'INIT_CORE_REL: CRHODE wrongly dimensioned (1):',ISPIN
         STOP
      ENDIF
      IF (ISPIN/=WDES%NCDIJ) THEN
         WRITE(*,*) 'INIT_CORE_REL: CRHODE wrongly dimensioned (2):',ISPIN,WDES%NCDIJ
         STOP
      ENDIF
      IF (SIZE(CRHODE,2)/=LMDIM) THEN
         WRITE(*,*) 'INIT_CORE_REL: CRHODE wrongly dimensioned (3):',LMDIM,SIZE(CRHODE,2)
         STOP
      ENDIF
      IF (SIZE(CRHODE,3)/=NIONS_LOCAL) THEN
         WRITE(*,*) 'INIT_CORE_REL: CRHODE wrongly dimensioned (4):',NIONS_LOCAL,SIZE(CRHODE,3)
         STOP
      ENDIF

! get the initial AE-core, AE-total and the PS-valence density
      IF (.FALSE.) THEN
      ! restart from a previous relaxed core calculation
         WRITE(*,*) 'INIT_CORE_REL: restart from previous relaxed core calculations'
         WRITE(*,*) '               is not possible... yet'
         STOP
      ELSE
      ! cold start or restart from a frozen core calculation
      ! calculate the AE and PS radial valence charge densities
      ! PP%RHOAE00 and PP%RHOPS00 from the initial partial waves
      ! and compensation charges, and the current onsite 
      ! occupancies CRHODE
         DO NT=1,NTYP
            IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
            NI=TYPE2ION(NT)
            PP=>SET_PP_POINTER(NI)
            CALL CHARGE(CRHODE(:,:,NI,1),PP)
         ENDDO
      ENDIF
      ! determine the expansion coefficients, PP%C, that define the partial 
      ! waves stored on the POTCAR file
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ! subtract PP%AVERAGEPOT(1) [= \int V^1(r)\hat{Q}^{00}(r) dr ]
         ! from the reference energies. PP%AVERAGEPOT(1) has been set
         ! by SET_AVERAGEPOT as it is called in SET_PAW_ATOM_POT (paw.F).
         PP%E_ORIG(:)=PP%E(:)
         PP%E(:)=PP%E(:)-PP%AVERAGEPOT(1)
#ifdef testout
         WRITE(*,*) 'INIT_CORE_REL: PP%AVERAGEPOT(AE)=',PP%AVERAGEPOT(1)
#endif
         CALL PS_PAR_WAVE_POTCAR(PP)
      ENDDO
#ifdef MPI
      ! synchronize pseudopotential information across nodes
      CALL SYNC_PP()
#endif
      ! and give output
      io_begin
      WRITE(IOUT,*)
      WRITE(IOUT,*) 'Information from POTCAR'
      CALL WRTPP(IOUT)
      io_end
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         PP=>SET_PP_POINTER(NI)
         ! get core contributions to the total energy for the
         ! atomic valence configuration
         CALL ECORE0(PP)
#ifdef testout         
!        CALL ECORE0_TEST(PP)
#endif
      ENDDO
#ifdef MPI
      ! synchronize pseudopotential information across nodes
      CALL SYNC_PP()
#endif
      io_begin
      WRITE(IOUT,*)
      WRITE(IOUT,*) 'Core states (SCF)'
      CALL WRTSC(IOUT)
      io_end
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         ! switch off the use of the old core level shift
         ! routines in case they were going to be applied to
         ! a type that has been earmarked for core relaxation
         ! i.e. set ICORELEVEL=0 (n.b.: on all nodes)
         IF (MAP2TYPE_ORIG(NI)==NT_CL) ICORELEVEL=0
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         PP=>SET_PP_POINTER(NI)
         ! update core charge densities at this early point:
         ! necessary for core level shift calculations since
         ! the core ionicity might have to be adjusted          
         IF (.NOT.LMIMFC()) CALL SCCORE(PP)
         ! change PS core potential in case of
         ! core level shift calculations
         CALL POTPSC_UPD(PP)
      ENDDO
#ifdef MPI
      ! synchronize pseudopotential information across nodes
      CALL SYNC_PP()
#endif
      ! fill the array RHO_ONE_CENTRE_LAST
      CALL GLUE_CHARGE(RHO_ONE_CENTRE_LAST)
      ! end of initialization
      LINITDONE=.TRUE.

      RETURN

  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP

      END SUBROUTINE INIT_CORE_REL


!*********************** SUBROUTINE REPORT ******************************
! 
! This subroutine writes 
! 
!  PP%RHOAE    : AE core charge density
!                 n_c
!  PP%RHOPS    : PS partial core charge density
!                 \tilde{n}_c
!  PP%RHOAE00  : AE valence charge density
!                 n_v
!  PP%RHOPS00  : PS valence charge density
!                 \tilde{n_v}
!  PP%RHOPSPW  : PS valence charge density deduced from PW grid (CHDEN)
!                 \tilde{n_v}
!  PP%V00_AE   : AE total local potential
!                 V_H[n_v+n_Zc]+V_xc[n_v+n_c]
!  PP%V00_PS   : PS total local potential
!                 V_H[\tilde{n}_v]+V_xc[\tilde{n}_v+\tilde{n}_c]+POTPSC
!  PP%POTAE    : AE local potential (valence contributions)
!                 V00_AE-V_H[n_Zc]
!  PP%POTPS    : PS local potential (valence contributions)
!                 V00_PS-POTPSC
!  PP%POTAEC   : AE local potential (core contributions)
!                 V_H[n_Zc]
!  PP%POTPSC   : PS local potential (core contributions)
!                 V_H[\tilde{n}_Zc]
!  PP%WAE      : AE partial waves
!                 |\phi_i>
!  PP%WPS      : PS partial waves
!                 |\tilde{\phi}_i>
!
! to a file called RELCOR.type#.step#
!
!************************************************************************
      SUBROUTINE REPORT(LAST)
      
      USE prec
      USE pseudo
      USE main_mpi
      
      IMPLICIT NONE

      TYPE(potcar), POINTER :: PP

      LOGICAL, OPTIONAL :: LAST

      INTEGER K,IUNIT
      INTEGER NI,NT
      INTEGER, PARAMETER :: IUBASE=100
      CHARACTER (LEN=16) :: FILEOUT
      
      IUNIT=IUBASE+NODE_ME
      
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ! open file
         IF (PRESENT(LAST)) THEN
            IF (LAST) THEN
               WRITE(FILEOUT,'(A7,I4.4,A5)') "RELCOR.",NT,".LAST"
            ELSE
               WRITE(FILEOUT,'(A7,I4.4,A1,I4.4)') "RELCOR.",NT,".",NCALLED
            ENDIF
         ELSE
            WRITE(FILEOUT,'(A7,I4.4,A1,I4.4)') "RELCOR.",NT,".",NCALLED
         ENDIF
         
         OPEN(UNIT=IUNIT,FILE=DIR_APP(1:DIR_LEN)//FILEOUT,STATUS='NEW',ERR=100)
         ! output all quantities on radial grid
         DO K=1,PP%R%NMAX
            WRITE(IUNIT,'(25F20.8)') PP%R%R(K), &
           &   PP%RHOAE(K),PP%RHOPS(K), &
           &   PP%RHOAE00(K),PP%RHOPS00(K),PP%RHOPSPW(K), &
           &   PP%V00_AE(K),PP%V00_PS(K), &
           &   PP%POTAE(K),PP%POTPS(K), &
           &   PP%POTAEC(K),PP%POTPSC(K), &
           &   PP%WAE(K,1:SIZE(PP%WAE,2)),PP%WPS(K,1:SIZE(PP%WPS,2))
         ENDDO
         ! close file
         CLOSE(IUNIT)
 100     CONTINUE        
      ENDDO
      
      RETURN
      END SUBROUTINE REPORT


!*********************** SUBROUTINE WRTPP *******************************
! 
! This subroutine writes 
!
!  PP%DION  : unscreened PAW strength parameters
!  PP%QION  : integrated augmentation charges
!  PP%C     : PS partial wave expansion coefficients
!
! to the OUTCAR file
!
!************************************************************************
      SUBROUTINE WRTPP(IUNIT)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      INTEGER IUNIT
      INTEGER I,J,K,NI,NT

      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         IF (LUSERWIGS()) THEN
            WRITE(IUNIT,'(/,A5,I4,A4,4X,A9,F10.4,/)') ' Ion:',MAP2GLOBAL(TYPE2ION(NT)),": "//PP%ELEMENT, &
           &                              'r(match)=',PP%RWIGS
         ELSE
            WRITE(IUNIT,'(/,A5,I4,A4,/)') ' Ion:',MAP2GLOBAL(TYPE2ION(NT)),": "//PP%ELEMENT
         ENDIF
         WRITE(IUNIT,'(A16)') 'DIJ (unscreened)'
         DO I=1,PP%LMAX
            WRITE(IUNIT,'(10F15.7)') (PP%DION(I,J), J=1,PP%LMAX)
         ENDDO               
         WRITE(IUNIT,'(A3)') 'QIJ'
         DO I=1,PP%LMAX
            WRITE(IUNIT,'(10F15.7)') (PP%QION(I,J), J=1,PP%LMAX)
         ENDDO      
         WRITE(IUNIT,'(A3)') 'AIJ'
         DO I=1,PP%LMAX
            WRITE(IUNIT,'(10F15.7)') (PP%C(I,J), J=1,3)
         ENDDO      
      ENDDO

      RETURN
      END SUBROUTINE WRTPP


!*********************** SUBROUTINE WRTSC *******************************
! 
! This subroutine writes 
!
!  PP%CLEV  : the core state eigenenergies
!
! to the OUTCAR file
!
! N.B.: the core state eigenenergies are shifted by an amount
!
!       dE = \int_\Omega \tilde{V}_eff(r) \hat{Q}^{00}(r) dr -
!              \int \tilde{V}^1(r)\hat{Q}^{00}(r) dr 
!
!          = (PP%AVEARGEPOT(3)-PP%AVERAGEPOT(2))
!
! (See Diplomarbeit of L. Koehler, 
!  or L. Koehler and G. Kresse, to be published)
!
!************************************************************************      
      SUBROUTINE WRTSC(IUNIT)
      USE prec
      USE pseudo
      USE cl
      
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      INTEGER IUNIT
      INTEGER I,J,K,NI,NT
      INTEGER MAXNL,IND,N,L
      REAL(q) OCC,ESHIFT
      CHARACTER(1) :: LLABEL(4)
      
      LLABEL(1)="S"; LLABEL(2)="P"; LLABEL(3)="D"; LLABEL(4)="F"

      types: DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ESHIFT=PP%AVERAGEPOT(3)-PP%AVERAGEPOT(2)
         
         WRITE(IUNIT,'(/,A5,I4,A4,/)') ' Ion:',MAP2GLOBAL(TYPE2ION(NT)),": "//PP%ELEMENT

         CALL CALCULATE_MAX_N_L(PP%ZCORE,MAXNL)
         CALL INIT_N_L(IND,N,L)
         I=1
         levels: DO
            IF (.NOT.NEXT_N_L(PP%ZCORE,MAXNL,IND,N,L,OCC)) EXIT levels  
            IF (RADEQ=="DIR") THEN
            ! radial Dirac-equation
               IF (L/=0) THEN
               ! write out eigenvalue (n,j=l-1/2) i.e. K=L
                  WRITE(IUNIT,'(2X,I1,A1,"-",3X,F16.8)') N,LLABEL(L+1),PP%CLEV(I)+ESHIFT
                  I=I+1
               ENDIF
               ! write out eigenvalue (n,j=l+1/2) i.e. K=-L-1
               WRITE(IUNIT,'(2X,I1,A1,"+",3X,F16.8)') N,LLABEL(L+1),PP%CLEV(I)+ESHIFT  
            ELSEIF (RADEQ=="SOC") THEN
            ! radial scalar relativistic equation including SOC term
               IF (L/=0) THEN
               ! write out eigenvalue (n,j=l-1/2) i.e. K=L
                  WRITE(IUNIT,'(2X,I1,A1,"-",3X,F16.8)') N,LLABEL(L+1),PP%CLEV(I)+ESHIFT
                  I=I+1
               ENDIF
               ! write out eigenvalue (n,j=l+1/2) i.e. K=-L-1
               WRITE(IUNIT,'(2X,I1,A1,"+",3X,F16.8)') N,LLABEL(L+1),PP%CLEV(I)+ESHIFT
            ELSE
            ! radial scalar relativistic equation
               ! write out eigenvalue (n,l)
               WRITE(IUNIT,'(2X,I1,A1,4X,F16.8)') N,LLABEL(L+1),PP%CLEV(I)+ESHIFT 
            ENDIF
            I=I+1
         ENDDO levels              
         ! write core energy
         WRITE(IUNIT,*)
         WRITE(IUNIT,'(A,F16.8)') 'Core contribution to total energy =',PP%ECORE(1)+PP%EATOM
         WRITE(IUNIT,*)
      ENDDO types

      RETURN
      END SUBROUTINE WRTSC

      
!*********************** QUERY FUNCTIONS ********************************      
!************************************************************************            

      FUNCTION LCORREL()
      IMPLICIT NONE
      LOGICAL LCORREL
      IF (LRELCOR) THEN
         LCORREL=.TRUE.
      ELSE
         LCORREL=.FALSE.
      ENDIF      
      END FUNCTION LCORREL

      FUNCTION LCORREL_INIT_DONE()
      IMPLICIT NONE
      LOGICAL LCORREL_INIT_DONE
      LCORREL_INIT_DONE=LINITDONE
      END FUNCTION LCORREL_INIT_DONE

      FUNCTION LREPORT()
      IMPLICIT NONE
      LOGICAL LREPORT
      LREPORT=.FALSE.
      ! Always give output on initialization
      IF (NCALLED==0) LREPORT=.TRUE.
      ! provide output at every iprint-th step
      IF (IPRINT/=0.AND.NCALLED/=0) THEN
         LREPORT=(MOD(NCALLED,IPRINT)==0)
      ENDIF
      END FUNCTION LREPORT

      FUNCTION LUPDATE()
      IMPLICIT NONE
      LOGICAL LUPDATE
      LUPDATE=(NCALLED.GE.IDELAY)
      END FUNCTION LUPDATE

      FUNCTION LMIMFC()
      IMPLICIT NONE
      LOGICAL LMIMFC
      LMIMFC=LMIMICFC
      END FUNCTION LMIMFC

      FUNCTION LSKIP()
      IMPLICIT NONE
      LOGICAL LSKIP
      LSKIP=(MOD(NCALLED,ISTRIDE).NE.0)      
      END FUNCTION LSKIP

      FUNCTION LUSERWIGS()
      IMPLICIT NONE
      LOGICAL LUSERWIGS
      LUSERWIGS=LMATCHRW
      END FUNCTION LUSERWIGS

      FUNCTION LADAPTELIN()
      IMPLICIT NONE
      LOGICAL LADAPTELIN
!     LADAPTELIN=LADAPTIVE
      LADAPTELIN=(LADAPTIVE.AND.(NCALLED.GE.IDELAY))
      END FUNCTION LADAPTELIN

      FUNCTION LSEMICOREONLY()
      IMPLICIT NONE
      LOGICAL LSEMICOREONLY
      LSEMICOREONLY=LONLYSEMICORE
      END FUNCTION LSEMICOREONLY

      FUNCTION IS_BOUND(PP,CHANNEL)
      USE pp_data
      IMPLICIT NONE
      TYPE(potcar) PP
      INTEGER CHANNEL
      LOGICAL IS_BOUND
      IF (CHANNEL<1.OR.CHANNEL>PP%LMAX) THEN
         IS_BOUND=.FALSE.
      ELSE
         IS_BOUND=(PP%QATO(CHANNEL,CHANNEL)/=0._q)
      ENDIF
      END FUNCTION IS_BOUND

      FUNCTION IS_1ST_OF_L(PP,CHANNEL)
      USE pp_data
      IMPLICIT NONE
      TYPE(potcar) PP
      INTEGER CHANNEL,I
      LOGICAL IS_1ST_OF_L
      IF (CHANNEL<1.OR.CHANNEL>PP%LMAX) THEN
         IS_1ST_OF_L=.FALSE.
      ELSE
         DO I=1,CHANNEL
            IF (PP%LPS(I)==PP%LPS(CHANNEL)) EXIT
         ENDDO
         IS_1ST_OF_L=(I==CHANNEL)
      ENDIF
      END FUNCTION IS_1ST_OF_L

      FUNCTION IS_2ND_OF_L(PP,CHANNEL)
      USE pp_data
      IMPLICIT NONE
      TYPE(potcar) PP
      INTEGER CHANNEL,I
      LOGICAL IS_2ND_OF_L
      IF (CHANNEL<1.OR.CHANNEL>PP%LMAX) THEN
         IS_2ND_OF_L=.FALSE.
      ELSE
         DO I=1,CHANNEL
            IF (PP%LPS(I)==PP%LPS(CHANNEL)) EXIT
         ENDDO
         IS_2ND_OF_L=(I==CHANNEL-1)
      ENDIF
      END FUNCTION IS_2ND_OF_L

      FUNCTION IS_ONLY_OF_L(PP,CHANNEL)
      USE pp_data
      IMPLICIT NONE
      TYPE(potcar) PP
      INTEGER CHANNEL,I
      LOGICAL IS_ONLY_OF_L
      IS_ONLY_OF_L=IS_1ST_OF_L(PP,CHANNEL).AND. &
     &               (.NOT.IS_2ND_OF_L(PP,CHANNEL+1))
      END FUNCTION IS_ONLY_OF_L

      FUNCTION IS_SEMICORE(PP,CHANNEL)
      USE pp_data
      IMPLICIT NONE
      TYPE(potcar) PP
      INTEGER CHANNEL
      LOGICAL IS_SEMICORE
      IF (CHANNEL<1.OR.CHANNEL>PP%LMAX) THEN
         IS_SEMICORE=.FALSE.
      ELSE
         IS_SEMICORE=((PP%E_ORIG(CHANNEL)<E_SEMICORE).AND.IS_BOUND(PP,CHANNEL))
      ENDIF
      END FUNCTION IS_SEMICORE


!*********************** SUBROUTINE CORREL ******************************
!
! This subroutine is called from ELMIN to perform the relaxation of
! the core electronic density, and to adjust the pseudopotential 
! accordingly. As input it takes the onsite valence charge densities 
! for the atoms that are treated with the relaxed core method.
!
!************************************************************************
      SUBROUTINE CORREL(CHARRAY)
      USE prec
      USE pseudo
      USE poscar
      USE radial
      USE us
      USE paw

      IMPLICIT NONE

      TYPE(potcar), POINTER :: PP

      REAL(q) CHARRAY(:)

      INTEGER NI,NT
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      REAL(q), ALLOCATABLE :: RHOLM(:)

      ! number of times the core has been relaxed
      NCALLED=NCALLED+1

      ! skip; relax the core every RCSTRD-th step in the
      ! electronic minimization
      IF (LSKIP()) RETURN

!     IF (.NOT.LUPDATE().AND.NCALLED/=1) RETURN

      ! for the first call the AE and PS valence charge density are taken
      ! as established by the initialization afterwards it is established 
      ! through mixing
      IF (NCALLED/=1) THEN
         CALL SPLIT_CHARGE(CHARRAY)
      ENDIF

      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         IF (.NOT.LMIMFC()) THEN
            ! SCF calculation of the core charge density
            CALL SCCORE(PP)
            ! determine the AE partial waves
            CALL AE_PAR_WAVE(PP,NT)
            ! determine the PS partial waves
            CALL PS_PAR_WAVE(PP,NT)
            ! recalculate the quantities that
            ! define the compensation charges
            CALL SET_PAW_AUG_RC(PP)
         ELSE
            ! for LMIMICFC=.TRUE. this will only set PP%AVERAGEPOT(2)
            CALL PS_PAR_WAVE(PP,NT)
         ENDIF
         ! calculate the core contributions
         ! to the total energy
         CALL GET_ECORE(PP)
      ENDDO
#ifdef MPI
      ! synchronize pseudopotential information across nodes
      CALL SYNC_PP()
#endif
      ! and give output at every RCREP-th step
      IF (LREPORT()) THEN
         io_begin      
         WRITE(IOUT,*) 'Core states (SCF)'
         CALL WRTSC(IOUT)
         WRITE(IOUT,*)
         WRITE(IOUT,*) 'Re-pseudization'
         CALL WRTPP(IOUT)
         WRITE(IOUT,*) 
         io_end
         CALL REPORT
      ENDIF
      RETURN
      END SUBROUTINE CORREL


!*********************** FUNCTION ECORE *********************************
!
! This function is called from ELMIN and equals the sum of the 
! core contributions to the total energy over all ions that are
! treated with the relaxed core method.
!
!************************************************************************
      FUNCTION ECORE()
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      INTEGER NI,NT,NIP
      REAL(q) ECORE
      IF (LCORREL()) THEN
         ECORE=0
         ion: DO NI=1,NIONS_GLOBAL
            NT=MAP2TYPE(NI)
            IF (NT==0) CYCLE ion
            NIP=TYPE2ION(NT)
            PP=>SET_PP_POINTER(NIP)
!           ECORE=ECORE+PP%ECORE(1)
!           ECORE=ECORE+PP%ECORE(2)
            ECORE=ECORE+PP%ECORE(1)-PP%ECORE(2)
         ENDDO ion
      ELSE
         ECORE=0
      ENDIF      
      END FUNCTION ECORE


!*********************** SUBROUTINE CHARGE ******************************
!
! This subroutine calculates the onsite AE and PS charge densities
! PP%RHOAE00 and PP%RHOPS00, respectively, from the onsite occupancies
! CRHODE, using the relevant pseudopotential information stored in the
! datastructure PP.
!
!************************************************************************
      SUBROUTINE CHARGE(CRHODE,PP)
      
      USE prec
      USE pseudo
      USE radial
      USE paw
      USE us
      
      IMPLICIT NONE
      
      TYPE (potcar) PP
      
      OVERLAP CRHODE(:,:)
      INTEGER LYMAX,LMMAX,I
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      REAL(q), DIMENSION(SIZE(CRHODE,1)*SIZE(CRHODE,1)) :: RHOLM
      INTEGER, EXTERNAL :: MAXL_AUG
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1))
      
      RHOLM=0
#ifdef testout
      WRITE(*,*) 'CHARGE: trace of onsite occupancy matrix:'
      WRITE(*,'(20F14.7)') (REAL(CRHODE(I,I)),I=1,SIZE(CRHODE,1))
#endif
      CALL TRANS_RHOLM(CRHODE(:,:),RHOLM(:),PP)
      ! valence AE charge density
      RHO=0
      CALL RAD_CHARGE(RHO(:,:,1),PP%R,RHOLM(:),PP%LMAX,PP%LPS,PP%WAE)
      ! spherical part of valence AE charge density
      PP%RHOAE00(:)=RHO(:,1,1)
      ! valence PS charge density
      RHO=0
      CALL RAD_CHARGE(RHO(:,:,1),PP%R,RHOLM(:),PP%LMAX,PP%LPS,PP%WPS)
!     write(101,'(2F14.7)') (PP%R%R(i),RHO(i,1,1),i=1,pp%r%nmax)
      ! add compensation charges to valence PS density
      CALL RAD_AUG_CHARGE(RHO(:,:,1),PP%R,RHOLM(:),PP%LMAX,PP%LPS, &
     &                     LYMAX,PP%AUG,PP%QPAW)
      ! spherical part of valence+compensation PS charge density
      PP%RHOPS00(:)=RHO(:,1,1)

      DEALLOCATE(RHO)      
      RETURN
      END SUBROUTINE CHARGE


      SUBROUTINE ADJUST_CHARGE(PP)
      USE prec
      USE pseudo
      USE radial
      
      IMPLICIT NONE
      
      TYPE (potcar) PP
      INTEGER I,J,NRWIGS
      REAL(q) DRHO(PP%R%NMAX)

      CALL GRAD(PP%R,PP%RHOAE00,DRHO)
      ! find me the matching radius
      DO I=1,PP%R%NMAX-1
         IF (PP%R%R(I)>=PP%RWIGS) EXIT
      ENDDO
      NRWIGS=I
      ! check whether the charge density is a decreasing
      ! quantity at PP%RWIGS; otherwise return

      IF (DRHO(NRWIGS)>0._q) RETURN
      ! search for a stationary point between 
      ! PP%RWIGS and PP%R%R(PP%R%NMAX) 
      DO I=NRWIGS,PP%R%NMAX-1
         IF (0.501*ABS(SIGN(1._q,DRHO(I))-SIGN(1._q,DRHO(I-1)))>1._q) EXIT
      ENDDO
      DO J=I,PP%R%NMAX
         PP%RHOAE00(J)=PP%RHOAE00(I)
         PP%RHOPS00(J)=PP%RHOPS00(I)
      ENDDO

      RETURN
      END SUBROUTINE ADJUST_CHARGE



!*********************** SUBROUTINE GLUE_CHARGE *************************
!
! This subroutine glues the arrays PP%RHOAE00 and PP%RHOPS00
! (the onsite AE and PS valence charge densities), of all types
! treated with the relaxed core method, together into the array
! CHARRAY.
! In addition it scales all elements with the metric: 
! sqrt(PP%R%SI(r))*SCALE_METRIC
!
! The latter is done because the array CHARRAY will be passed
! down into the broyden mixer.
!
!************************************************************************
      SUBROUTINE GLUE_CHARGE(CHARRAY)
      
      USE prec
      USE poscar
      USE pseudo
      
      IMPLICIT NONE
      
      TYPE (potcar), POINTER :: PP
      
      REAL(q) CHARRAY(:)
      
      INTEGER NELEMENTS
      INTEGER NI,NT,I
      INTEGER IBASE
      
      ! calculate total number of elements to be transfered
      NELEMENTS=0
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         NELEMENTS=NELEMENTS+2*PP%R%NMAX
      ENDDO
      ! is target array large enough to hold these elements?
      IF (NELEMENTS>SIZE(CHARRAY)) THEN
         WRITE(*,*) 'GLUE_ARRAY: internal error, target array too small:',NELEMENTS,SIZE(CHARRAY)
         STOP
      ENDIF
      ! fill array
      IBASE=1
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ! AE valence charge density
         DO I=1,PP%R%NMAX
            CHARRAY(IBASE+I-1)=PP%RHOAE00(I)*SQRT(PP%R%SI(I))*SCALE_METRIC
         ENDDO
         IBASE=IBASE+PP%R%NMAX
         ! PS valence charge density
         DO I=1,PP%R%NMAX
            CHARRAY(IBASE+I-1)=PP%RHOPS00(I)*SQRT(PP%R%SI(I))*SCALE_METRIC
         ENDDO
         IBASE=IBASE+PP%R%NMAX
      ENDDO

      RETURN      
      END SUBROUTINE GLUE_CHARGE


!*********************** SUBROUTINE SPLIT_CHARGE ************************
!
! This subroutine is the complement to the GLUE_CHARGE routine, and
! splits the array CHARRAY up in its respective PP%RHOAE00 and PP%RHOPS00
! components.
! It divides all elements by:
!   sqrt(PP%R%SI(r))*SCALE_METRIC
!
!************************************************************************
      SUBROUTINE SPLIT_CHARGE(CHARRAY)
      
      USE prec
      USE poscar
      USE pseudo
      
      IMPLICIT NONE
      
      TYPE (potcar), POINTER :: PP
      
      REAL(q) CHARRAY(:)
      
      INTEGER NELEMENTS
      INTEGER NI,NT,I
      INTEGER IBASE
      
      ! calculate total number of elements to be transfered
      NELEMENTS=0
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         NELEMENTS=NELEMENTS+2*PP%R%NMAX
      ENDDO
      ! is target array large enough to hold these elements?
      IF (NELEMENTS>SIZE(CHARRAY)) THEN
         WRITE(*,*) 'SPLIT_ARRAY: internal error, target array too small:',NELEMENTS,SIZE(CHARRAY)
         STOP
      ENDIF
      ! split array
      IBASE=1
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ! AE valence charge density
         DO I=1,PP%R%NMAX
            PP%RHOAE00(I)=CHARRAY(IBASE+I-1)/SQRT(PP%R%SI(I))/SCALE_METRIC
         ENDDO
         IBASE=IBASE+PP%R%NMAX
         ! PS valence charge density
         DO I=1,PP%R%NMAX
            PP%RHOPS00(I)=CHARRAY(IBASE+I-1)/SQRT(PP%R%SI(I))/SCALE_METRIC
         ENDDO
         IBASE=IBASE+PP%R%NMAX
      ENDDO

      RETURN
      END SUBROUTINE SPLIT_CHARGE



!*********************** SUBROUTINE SET_RHO_ONE_CENTRE ******************
!
! This subroutine is called from ELMIN. As input it takes the onsite
! occupancies CHRODE and calculates the onsite AE and PS charge
! densities for the ions that are treated with the relaxed core method.
! These onsite densities are glued together into array CHARRAY.
!
!************************************************************************
      SUBROUTINE SET_RHO_ONE_CENTRE(CRHODE,CHARRAY)

      USE prec
      USE poscar
      USE pseudo
      
      IMPLICIT NONE

      TYPE (potcar), POINTER :: PP
            
      OVERLAP CRHODE(:,:,:,:)
      REAL(q) CHARRAY(:)
      
      INTEGER NI,NT
      
      ! update onsite AE and PS charge density
      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         CALL CHARGE(CRHODE(:,:,NI,1),PP)
      ENDDO
      ! and glue them together
      CALL GLUE_CHARGE(CHARRAY)      
      
      RETURN
      END SUBROUTINE SET_RHO_ONE_CENTRE


!*********************** SUBROUTINE INIT_SYNC_PP ************************
!
! This subroutine calculates the size of the communication buffer 
! (COMM_BUFF_SIZE), needed to synchronize the pseudopotential information 
! across all nodes.
!
!************************************************************************
      SUBROUTINE INIT_SYNC_PP()
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      INTEGER NI,NT
      INTEGER BUFF_SIZE
      
      BUFF_SIZE=0
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ! PP parameters
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%DION,1)*SIZE(PP%DION,2)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%QION,1)*SIZE(PP%QION,2)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%QTOT,1)*SIZE(PP%QTOT,2)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%QPAW,1)*SIZE(PP%QPAW,2)*SIZE(PP%QPAW,3)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%QDEP,1)*SIZE(PP%QDEP,2)*SIZE(PP%QDEP,3)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%AUG ,1)*SIZE(PP%AUG ,2)
         ! Partial waves
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%WAE ,1)*SIZE(PP%WAE ,2)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%WPS ,1)*SIZE(PP%WPS ,2)
         ! densities
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%RHOAE)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%RHOPS)
         ! potentials
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%POTAE)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%POTPS)
         BUFF_SIZE=BUFF_SIZE+SIZE(PP%POTPSC)
         ! Expansion coefficients
         BUFF_SIZE=BUFF_SIZE+PP%LMAX*3
         ! PP%ZCORE
         BUFF_SIZE=BUFF_SIZE+1
         ! PP%AVERAGEPOT
         BUFF_SIZE=BUFF_SIZE+3
         ! Core state eigenenergies
         BUFF_SIZE=BUFF_SIZE+MAX_NUM_CORE_STATES
         ! PP%ECORE
         BUFF_SIZE=BUFF_SIZE+2
      ENDDO
      
      COMM_BUFF_SIZE=BUFF_SIZE

#ifdef testout
      WRITE(*,*) 'INIT_SYNC: BUFF_SIZE=',BUFF_SIZE
#endif
      RETURN
      END SUBROUTINE INIT_SYNC_PP

      
!*********************** SUBROUTINE SYNC_PP *****************************
!
! This subroutine synchronizes the pseudopotential information across
! the nodes.
! Updated are: PP%DION, PP%QION, PP%QTOT, PP%QPAW, PP%QDEP, PP%AUG,
!              PP%WAE, PP%WPS, PP%RHOAE, PP%RHOPS, PP%POTAE, PP%POTPS,
!              PP%POTPSC, PP%C, PP%ZCORE, PP%AVERAGEPOT, PP%CLEV,
!              PP%ECORE
!
!************************************************************************      
      SUBROUTINE SYNC_PP()
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      INTEGER NI,NT
      INTEGER I,J,K
      INTEGER COMM_BUFF_POS
      INTEGER ENTRY_SIZE
      REAL(q), ALLOCATABLE :: COMM_BUFF(:)
      
      ! allocate comm buffer
      ALLOCATE(COMM_BUFF(COMM_BUFF_SIZE))
      ! clear comm buffer
      COMM_BUFF=0 
      ! fill comm buffer
      COMM_BUFF_POS=1
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ENTRY_SIZE=SIZE(PP%DION,1)*SIZE(PP%DION,2)+ &
        &           SIZE(PP%QION,1)*SIZE(PP%QION,2)+ &
        &           SIZE(PP%QTOT,1)*SIZE(PP%QTOT,2)+ &
        &           SIZE(PP%QPAW,1)*SIZE(PP%QPAW,2)*SIZE(PP%QPAW,3)+ &
        &           SIZE(PP%QDEP,1)*SIZE(PP%QDEP,2)*SIZE(PP%QDEP,3)+ &
        &           SIZE(PP%AUG ,1)*SIZE(PP%AUG ,2)+ &
        &           SIZE(PP%WAE ,1)*SIZE(PP%WAE ,2)+ &
        &           SIZE(PP%WPS ,1)*SIZE(PP%WPS ,2)+ &
        &           SIZE(PP%RHOAE)+SIZE(PP%RHOPS) + &
        &           SIZE(PP%POTAE)+SIZE(PP%POTPS)+SIZE(PP%POTPSC) + &
        &           PP%LMAX*3+1+3+MAX_NUM_CORE_STATES+2
         IF (COMM_BUFF_POS+ENTRY_SIZE>COMM_BUFF_SIZE+1) THEN
            WRITE(*,*) 'SYNC_PP: PP communication buffer too small (1):', &
           &   COMM_BUFF_POS+ENTRY_SIZE-1,COMM_BUFF_SIZE
            STOP
         ENDIF
         IF (DO_TYPE_LOCAL(NT)) THEN
         ! PP parameters
            DO I=1,SIZE(PP%DION,2)
               J=SIZE(PP%DION,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%DION(1:J,I)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QION,2)
               J=SIZE(PP%QION,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%QION(1:J,I)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QTOT,2)
               J=SIZE(PP%QTOT,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%QTOT(1:J,I)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QPAW,3)
            DO J=1,SIZE(PP%QPAW,2)
               K=SIZE(PP%QPAW,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+K-1)=PP%QPAW(1:K,J,I-1)
               COMM_BUFF_POS=COMM_BUFF_POS+K
            ENDDO
            ENDDO
            DO I=1,SIZE(PP%QDEP,3)
            DO J=1,SIZE(PP%QDEP,2)
               K=SIZE(PP%QDEP,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+K-1)=PP%QDEP(1:K,J,I-1)
               COMM_BUFF_POS=COMM_BUFF_POS+K
            ENDDO
            ENDDO
            DO I=1,SIZE(PP%AUG,2)
               J=SIZE(PP%AUG,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%AUG(1:J,I-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
         ! Partial waves
            DO I=1,SIZE(PP%WAE,2)
               J=SIZE(PP%WAE,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%WAE(1:J,I)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%WPS,2)
               J=SIZE(PP%WPS,1)
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)=PP%WPS(1:J,I)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
         ! Densities
            I=SIZE(PP%RHOAE)
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)=PP%RHOAE(1:I)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%RHOPS)
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)=PP%RHOPS(1:I)
            COMM_BUFF_POS=COMM_BUFF_POS+I
         ! Potentials
            I=SIZE(PP%POTAE)
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)=PP%POTAE(1:I)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%POTPS)
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)=PP%POTPS(1:I)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%POTPSC)
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)=PP%POTPSC(1:I)
            COMM_BUFF_POS=COMM_BUFF_POS+I            
         ! Expansion coefficients
            DO I=1,3
               COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+PP%LMAX-1)=PP%C(1:PP%LMAX,I)
               COMM_BUFF_POS=COMM_BUFF_POS+PP%LMAX
            ENDDO
         ! PP%ZCORE
            COMM_BUFF(COMM_BUFF_POS)=PP%ZCORE
            COMM_BUFF_POS=COMM_BUFF_POS+1
         ! PP%AVERAGEPOT
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+2)=PP%AVERAGEPOT(1:3)
            COMM_BUFF_POS=COMM_BUFF_POS+3
         ! Core state eigenenergies
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+MAX_NUM_CORE_STATES-1)= &
           &   PP%CLEV(1:MAX_NUM_CORE_STATES)
            COMM_BUFF_POS=COMM_BUFF_POS+MAX_NUM_CORE_STATES
         ! PP%ECORE
            COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+1)=PP%ECORE(1:2)
            COMM_BUFF_POS=COMM_BUFF_POS+2
         ELSE
            COMM_BUFF_POS=COMM_BUFF_POS+ENTRY_SIZE
         ENDIF
      ENDDO
#ifdef debug
      IF (COMM_BUFF_POS/=COMM_BUFF_SIZE+1) THEN
         WRITE(*,*) 'SYNC_PP: buffer not completely filled:', &
        &   COMM_BUFF_POS-1,COMM_BUFF_SIZE
      ENDIF
#endif      
      ! communicate
      CALLMPI( M_sum_d(COMM_INTER, COMM_BUFF, COMM_BUFF_SIZE) )
      ! fill PP data structure
      COMM_BUFF_POS=1
      DO NT=1,NTYP
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         ENTRY_SIZE=SIZE(PP%DION,1)*SIZE(PP%DION,2)+ &
        &           SIZE(PP%QION,1)*SIZE(PP%QION,2)+ &
        &           SIZE(PP%QTOT,1)*SIZE(PP%QTOT,2)+ &
        &           SIZE(PP%QPAW,1)*SIZE(PP%QPAW,2)*SIZE(PP%QPAW,3)+ &
        &           SIZE(PP%QDEP,1)*SIZE(PP%QDEP,2)*SIZE(PP%QDEP,3)+ &
        &           SIZE(PP%AUG ,1)*SIZE(PP%AUG ,2)+ &
        &           SIZE(PP%WAE ,1)*SIZE(PP%WAE ,2)+ &
        &           SIZE(PP%WPS ,1)*SIZE(PP%WPS ,2)+ &
        &           SIZE(PP%RHOAE)+SIZE(PP%RHOPS) + &
        &           SIZE(PP%POTAE)+SIZE(PP%POTPS)+SIZE(PP%POTPSC) + &
        &           PP%LMAX*3+1+3+MAX_NUM_CORE_STATES+2
         IF (COMM_BUFF_POS+ENTRY_SIZE>COMM_BUFF_SIZE+1) THEN
            WRITE(*,*) 'SYNC_PP: PP communication buffer too small (2):', &
           &   COMM_BUFF_POS+ENTRY_SIZE-1,COMM_BUFF_SIZE
            STOP
         ENDIF
         IF (.NOT.DO_TYPE_LOCAL(NT)) THEN
         ! PP parameters
            DO I=1,SIZE(PP%DION,2)
               J=SIZE(PP%DION,1)
               PP%DION(1:J,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QION,2)
               J=SIZE(PP%QION,1)
               PP%QION(1:J,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QTOT,2)
               J=SIZE(PP%QTOT,1)
               PP%QTOT(1:J,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%QPAW,3)
            DO J=1,SIZE(PP%QPAW,2)
               K=SIZE(PP%QPAW,1)
               PP%QPAW(1:K,J,I-1)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+K-1)
               COMM_BUFF_POS=COMM_BUFF_POS+K
            ENDDO
            ENDDO
            DO I=1,SIZE(PP%QDEP,3)
            DO J=1,SIZE(PP%QDEP,2)
               K=SIZE(PP%QDEP,1)
               PP%QDEP(1:K,J,I-1)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+K-1)
               COMM_BUFF_POS=COMM_BUFF_POS+K
            ENDDO
            ENDDO
            DO I=1,SIZE(PP%AUG,2)
               J=SIZE(PP%AUG,1)
               PP%AUG(1:J,I-1)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
         ! Partial waves
            DO I=1,SIZE(PP%WAE,2)
               J=SIZE(PP%WAE,1)
               PP%WAE(1:J,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
            DO I=1,SIZE(PP%WPS,2)
               J=SIZE(PP%WPS,1)
               PP%WPS(1:J,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+J-1)
               COMM_BUFF_POS=COMM_BUFF_POS+J
            ENDDO
         ! Densities
            I=SIZE(PP%RHOAE)
            PP%RHOAE(1:I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%RHOPS)
            PP%RHOPS(1:I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)
            COMM_BUFF_POS=COMM_BUFF_POS+I
         ! Potentials
            I=SIZE(PP%POTAE)
            PP%POTAE(1:I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%POTPS)
            PP%POTPS(1:I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)
            COMM_BUFF_POS=COMM_BUFF_POS+I
            I=SIZE(PP%POTPSC)
            PP%POTPSC(1:I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+I-1)
            COMM_BUFF_POS=COMM_BUFF_POS+I            
         ! Expansion coefficients
            DO I=1,3
               PP%C(1:PP%LMAX,I)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+PP%LMAX-1)
               COMM_BUFF_POS=COMM_BUFF_POS+PP%LMAX
            ENDDO
         ! PP%ZCORE
            PP%ZCORE=COMM_BUFF(COMM_BUFF_POS)
            COMM_BUFF_POS=COMM_BUFF_POS+1
         ! PP%AVERAGEPOT
            PP%AVERAGEPOT(1:3)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+2)
            COMM_BUFF_POS=COMM_BUFF_POS+3
         ! Core state eigenenergies
            PP%CLEV(1:MAX_NUM_CORE_STATES)= &
           &   COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+MAX_NUM_CORE_STATES-1)
            COMM_BUFF_POS=COMM_BUFF_POS+MAX_NUM_CORE_STATES
         ! PP%ECORE
            PP%ECORE(1:2)=COMM_BUFF(COMM_BUFF_POS:COMM_BUFF_POS+1)
            COMM_BUFF_POS=COMM_BUFF_POS+2
         ELSE
            COMM_BUFF_POS=COMM_BUFF_POS+ENTRY_SIZE
         ENDIF
      ENDDO      
#ifdef debug
      IF (COMM_BUFF_POS/=COMM_BUFF_SIZE+1) THEN
         WRITE(*,*) 'SYNC_PP: buffer not completely emptied:', &
        &   COMM_BUFF_POS-1,COMM_BUFF_SIZE
      ENDIF
#endif
      ! deallocate comm buffer
      DEALLOCATE(COMM_BUFF)
      ! finished
      RETURN
      END SUBROUTINE SYNC_PP


!*********************** SUBROUTINE SCCORE ******************************
!
! Performs a SCF calculation of the core states for a fixed valence
! charge density PP%RHOAE00
!
! Input:  PP%RHOAE   : current core charge density
!         PP%RHOAE00 : current valence charge density
!
! Output: PP%RHOAE   : new self-consistent core charge density
!
! Additionally the calls to ATOM will set:
!         PP%V00_AE        : spherical component of total 
!                            onsite AE potential
!         PP%AVERAGEPOT(1) : \int V^1(r)\hat{Q}^{00}(r) dr
!         PP%CLEV          : core state eigenenergies
!
!
! N.B.: A possible adjustment of the core eigenstate occupations
!       used to calculate final state core level shift is taken
!       into account!
!
!************************************************************************
      SUBROUTINE SCCORE(PP)
      
      USE prec
      USE constant
      USE pseudo
      USE radial
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER I
      INTEGER ITER
      INTEGER NMIN
      
      REAL(q) EIG
      REAL(q) DIFF,MAXDIF,MINDIF
      REAL(q) QCORE
      LOGICAL LCONV

      INTEGER LYMAX,LMMAX
      
      REAL(q), DIMENSION(PP%R%NMAX) :: RHOC,WORK
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      
      INTEGER, PARAMETER :: MAXITER=100
      REAL(q), PARAMETER :: MIX=0.5_q
      REAL(q), PARAMETER :: TINY=1.0E-7

      INTEGER, EXTERNAL :: MAXL_AUG
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1))
      RHO=0
      RHO(:,1,1)=PP%RHOAE00(:)+PP%RHOAE(:)
      RHOC=PP%RHOAE
      LCONV=.FALSE.
      
! SCF calculation of the core charge density             
      scf: DO ITER=1,MAXITER
         ! calculate core charge density
         CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK)
         ! quick return if possible
         IF (.NOT.LUPDATE()) THEN
            ! on initialization we have to update the core charge
            ! density in case of core level shift calculations
#ifdef testout
            WRITE(*,*) 'SCCORE: LCLSHIFT=',LCLSHIFT
#endif
            IF ((LCLSHIFT).AND.(.NOT.LINITDONE)) PP%RHOAE(:)=WORK(:)
            DEALLOCATE(RHO)
            RETURN
         ENDIF
         ! mix
         MAXDIF=0._q
         DO I=1,PP%R%NMAX
            DIFF=WORK(I)-RHOC(I)
            RHOC(I)=RHOC(I)+MIX*DIFF
            MAXDIF=MAX(MAXDIF,ABS(DIFF))
         ENDDO
         
         RHO(:,1,1)=PP%RHOAE00(:)+RHOC(:)
         ! check convergence
         IF (MAXDIF<TINY) LCONV=.TRUE.
         
         IF (LCONV) THEN
            PP%RHOAE=RHOC
            CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK)
#ifdef testout
            CALL SIMPI(PP%R,RHOC,QCORE)
            WRITE(*,'(/," AE         core contains ",F12.5," electrons")') QCORE*2*SQRT(PI)
            CALL SIMPI(PP%R,PP%RHOPS,QCORE)
            WRITE(*,'(/," PS partial core contains ",F12.5," electrons")') QCORE*2*SQRT(PI)
#endif
            DEALLOCATE(RHO)
            RETURN
         ENDIF
      ENDDO scf

      WRITE(*,'(/,"SCCORE: solution not found within ",I4," attempts: MAXDIF=",F12.5)') &
     &   MAXITER,MAXDIF
      
      DEALLOCATE(RHO)
      STOP
      RETURN
      END SUBROUTINE SCCORE


!*********************** SUBROUTINE GET_ECORE ***************************
!
! Calculate the additional contributions of the core charge density
! to the total energy of the system:
!
! \sum^{Nc}_{i=1} \eps_i 
!      -\int n_c(r) (V_H[n_v+n_Zc]+V_xc[n_v+n_c]) dr
!      +1/2 \int n_c(r) V_H[n_c] dr + \int n_c(r) V_H[n_Z] dr 
! 
! These contributions are stored in PP%ECORE(1)
!
!************************************************************************      
      SUBROUTINE GET_ECORE(PP)
      USE prec
      USE constant
      USE pseudo
      USE radial
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER I
      
      REAL(q) QCORE,QVAL
      REAL(q) EIG,EKIN,DBLC,DHARTREE1,DHARTREE2,DEXC

      INTEGER LYMAX,LMMAX
      
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      
      REAL(q) RDUM,RHOREAD
      
      INTEGER, EXTERNAL :: MAXL_AUG
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1))
      
      RHO=0
      RHO(:,1,1)=PP%RHOAE00(:)+PP%RHOAE(:)

      CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK)
      ! \int n_c(r) (V_H[n_v+n_Zc]+V_xc[n_v+n_c]) dr
      DBLC=0
      DO I=1,PP%R%NMAX
         DBLC=DBLC+2*SQRT(PI)*PP%RHOAE(I)*PP%V00_AE(I)*PP%R%SI(I)
      ENDDO
      ! kinetic energy of the core electrons
      EKIN=EIG-DBLC
      ! \int n_c(r) V_H[n_c] dr
      CALL RAD_POT_HAR(0,PP%R,WORK,PP%RHOAE,DHARTREE1)
      ! \int n_c(r) V_H[n_Z] dr
      DHARTREE2=0
      DO I=1,PP%R%NMAX
         DHARTREE2=DHARTREE2- &
        &   PP%RHOAE(I)*PP%R%SI(I)* &
        &   2*SQRT(PI)*FELECT*(PP%ZVALF_ORIG+PP%ZCORE)/PP%R%R(I)
      ENDDO
      ! Exc[rhoc]
      CALL RAD_CORE_XC(PP%R,PP%RHOAE,DEXC)
#ifdef testout
      CALL SIMPI(PP%R,PP%RHOAE00,QVAL)
      CALL SIMPI(PP%R,PP%RHOAE,QCORE)
      WRITE(*,*) 'GET_ECORE:'
      WRITE(*,*) '---------------------------------------------'
      WRITE(*,*) 'Z_t                   :',(QVAL+QCORE)*2*SQRT(PI)
      WRITE(*,*) 'Z_v                   :',QVAL*2*SQRT(PI)
      WRITE(*,*) 'Z_c                   :',QCORE*2*SQRT(PI)
      WRITE(*,*) 
      WRITE(*,*) 'core eigenvalue sum   :',EIG
      WRITE(*,*) '-int V(r)*rhoc(r) dr  :',-DBLC
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'core kinetic energy   :',EKIN
      WRITE(*,*) '1/2(n_c)(n_c)         :',DHARTREE1/2
      WRITE(*,*) '(n_c)(n_z)            :',DHARTREE2
      WRITE(*,*) 'PP%DEXCCORE           :',PP%DEXCCORE
      WRITE(*,*) '-PP%EATOM             :',-PP%EATOM
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'Total                 :', &
     &   EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM
      WRITE(*,*) 'E_xc[n_c]             :',DEXC
      WRITE(*,*)
#endif         
      ! contribution to the total energy
      PP%ECORE(1)=EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM

      DEALLOCATE(RHO)
      RETURN
      END SUBROUTINE GET_ECORE


!*********************** SUBROUTINE ECORE0 ******************************
!
! This subroutine essentially does the same thing as GET_ECORE, with
! one difference; instead of using the actual onsite valence charge
! density PP%RHOAE00 it uses the valence charge density derived from
! the atomic occupancies. It therefore calculates an atomic reference
! energy.
!
! These contributions are stored in PP%ECORE(2)
! 
!************************************************************************      
      SUBROUTINE ECORE0(PP)
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE paw
      IMPLICIT NONE
      TYPE (potcar),POINTER :: PP
      INTEGER I
      INTEGER LYMAX,LMMAX      
      INTEGER, EXTERNAL :: MAXL_AUG
      REAL(q) EIG,DBLC,EKIN,DHARTREE1,DHARTREE2,DEXC,QVAL,QCORE,ESHIFT
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK
      REAL(q), ALLOCATABLE :: RHO(:,:,:),RHOLM(:)
      OVERLAP ,ALLOCATABLE :: CRHODE(:,:)
      REAL(q) RDUM,RHOREAD
      REAL(q), ALLOCATABLE :: V(:,:,:)
      LOGICAL LCLSHIFT_STORE
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2

      ALLOCATE(CRHODE(LMMAX,LMMAX),RHOLM(LMMAX*LMMAX), &
     &   RHO(PP%R%NMAX,LMMAX,1),V(PP%R%NMAX,LMMAX,1))

      ! set CRHODE to the atomic occupancies
#ifdef testout
      WRITE(*,*) 'ECORE0: Trace of atomic occupancy matrix:'
      WRITE(*,*) (PP%QATO(I,I),I=1,PP%LMAX)
#endif      
      CRHODE=0
      CALL SET_CRHODE_ATOM(CRHODE(:,:),PP)
      ! calculate the valence AE charge density
      RHOLM=0
      CALL TRANS_RHOLM(CRHODE(:,:),RHOLM(:),PP)
      RHO=0
      CALL RAD_CHARGE(RHO(:,:,1),PP%R,RHOLM(:),PP%LMAX,PP%LPS,PP%WAE)
      CALL SIMPI(PP%R,RHO(:,1,1),QVAL)
      ! add the AE core charge density
      RHO(:,1,1)=RHO(:,1,1)+PP%RHOAE(:)
      ! get the sum of the core eigenvalues
      ESHIFT=PP%AVERAGEPOT(1)
      LCLSHIFT_STORE=LCLSHIFT
      LCLSHIFT=.FALSE.
      CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK)
#ifdef testout
!     WRITE(98,'(2F20.8)') (PP%R%R(I),RHO(I,1,1),I=1,PP%R%NMAX)
!     WRITE(99,'(2F20.8)') (PP%R%R(I),PP%V00_AE(I),I=1,PP%R%NMAX)
!     STOP
#endif      
      LCLSHIFT=LCLSHIFT_STORE
      ! shift the core state eigenvalues (for output lateron)
      ESHIFT=ESHIFT-PP%AVERAGEPOT(1)
      PP%CLEV(:)=PP%CLEV(:)+ESHIFT
      ! \int n_c(r) (V_H[n_v+n_Zc]+V_xc[n_v+n_c]) dr
      DBLC=0
      DO I=1,PP%R%NMAX
         DBLC=DBLC+2*SQRT(PI)*PP%RHOAE(I)*PP%V00_AE(I)*PP%R%SI(I)
      ENDDO
      ! kinetic energy of the core electrons
      EKIN=EIG-DBLC
      ! \int n_c(r) V_H[n_c] dr
      CALL RAD_POT_HAR(0,PP%R,WORK,PP%RHOAE,DHARTREE1)
      ! \int n_c(r) V_H[n_Z] dr
      DHARTREE2=0
      DO I=1,PP%R%NMAX
         DHARTREE2=DHARTREE2- &
        &   PP%RHOAE(I)*PP%R%SI(I)* &
        &   2*SQRT(PI)*FELECT*(PP%ZVALF_ORIG+PP%ZCORE)/PP%R%R(I)
      ENDDO
      ! Exc[rhoc]
      CALL RAD_CORE_XC(PP%R,PP%RHOAE,DEXC)
#ifdef testout
      CALL SIMPI(PP%R,PP%RHOAE,QCORE)
      WRITE(*,*) 'ECORE0:'
      WRITE(*,*) '---------------------------------------------'
      WRITE(*,*) 'Z_t                   :',(QVAL+QCORE)*2*SQRT(PI)
      WRITE(*,*) 'Z_v                   :',QVAL*2*SQRT(PI)
      WRITE(*,*) 'Z_c                   :',QCORE*2*SQRT(PI)
      WRITE(*,*) 
      WRITE(*,*) 'core eigenvalue sum   :',EIG
      WRITE(*,*) '-int V(r)*rhoc(r) dr  :',-DBLC
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'core kinetic energy   :',EKIN
      WRITE(*,*) '1/2(n_c)(n_c)         :',DHARTREE1/2
      WRITE(*,*) '(n_c)(n_z)            :',DHARTREE2
      WRITE(*,*) 'PP%DEXCCORE           :',PP%DEXCCORE
      WRITE(*,*) '-PP%EATOM             :',-PP%EATOM
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'Total                 :', &
     &   EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM
      WRITE(*,*) 'E_xc[n_c]             :',DEXC
      WRITE(*,*)
#endif         
      ! contribution to the total energy
      PP%ECORE(2)=EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM

      DEALLOCATE(CRHODE,RHOLM,RHO)            
      RETURN
      END SUBROUTINE ECORE0


!*********************** SUBROUTINE ATOM ********************************      
!
! Solve the spherical atomic problem for its core states
! On input RHO is the total charge density (core+valence)
! in (charge, magnetization) format, on output
! RHOC holds the core charge density, and EIG the sum of
! the eigenvalues of the core states
!
! Additionaly updated variables:
!    PP%V00_AE, PP%AVERAGEPOT(1), PP%CLEV
!
!************************************************************************
      SUBROUTINE ATOM(RHO,Z,PP,EIG,RHOC)
      
      USE prec
      USE constant 
      USE pseudo
      USE radial
      USE cl
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER, INTENT(IN) :: Z
      REAL(q), DIMENSION(:,:,:), INTENT(IN) :: RHO
      REAL(q), DIMENSION(:), INTENT(OUT) :: RHOC
      REAL(q), INTENT(OUT) :: EIG
      
      REAL(q), DIMENSION(SIZE(RHO,1),SIZE(RHO,2),SIZE(RHO,3)) :: V
! local variables
      INTEGER :: I,ISPIN,ZC
      INTEGER :: MAXNL,IND,N,L,K   
      REAL(q) :: OCC,SCALE,E,AVERAGEPOT
      REAL(q), DIMENSION(SIZE(RHO,1)) :: V00
      REAL(q), DIMENSION(SIZE(RHO,1)) :: DV
      REAL(q), DIMENSION(SIZE(RHO,1)) :: RHO_
      
      REAL(q) RDUM,VREAD
      
      CHARACTER(1) :: LLABEL(4)
                  
      LLABEL(1)="S"; LLABEL(2)="P"; LLABEL(3)="D"; LLABEL(4)="F"
      
      ISPIN=SIZE(RHO,3)
! check grid consistency
      IF (SIZE(RHO,1)<PP%R%NMAX.OR.SIZE(RHO,1)/=SIZE(RHOC)) THEN
         WRITE(*,*) 'ATOM: Grid inconsistency:',SIZE(RHO,1),SIZE(RHOC),PP%R%NMAX
         STOP
      ENDIF      
! initialize
      V=0; DV=0 ;EIG=0; RHOC=0
! get effective potential      
      CALL POT(RHO,Z,PP%R,V)
! take the spherical component      
      SCALE=1/(2*SQRT(PI))
      V00(:)=V(:,1,1)*SCALE
! store spherical potential
      PP%V00_AE=V00
! recalculate PP%AVERAGEPOT
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+V00(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      PP%AVERAGEPOT(1)=AVERAGEPOT
      DO I=1,PP%R%NMAX
         V00(I)=V00(I)-(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))            
      ENDDO
#ifdef testout
      WRITE(*,*) 'ATOM: PP%AVERAGEPOT(AE)=',PP%AVERAGEPOT(1)
#endif         
! For the "radial scalar relativistic + SOC" we need dV/dr as well
      IF (RADEQ=="SOC") THEN
         CALL GRAD(PP%R,V00(1:PP%R%NMAX),DV(1:PP%R%NMAX))
      ENDIF
! solve the atomic problem for the core states
      ZC=Z-PP%ZVALF_ORIG
!     CALL CALCULATE_MAX_N_L(REAL(ZC,q),MAXNL)
      CALL CALCULATE_MAX_N_L(PP%ZCORE,MAXNL)

      CALL INIT_N_L(IND,N,L)
! clear structure that stores core state eigenenergies
      PP%CLEV=0; I=1
! start
      DO
         IF (.NOT.NEXT_N_L(PP%ZCORE,MAXNL,IND,N,L,OCC)) EXIT     
         IF (RADEQ=="DIR") THEN
! radial Dirac-equation
            IF (L/=0) THEN
      ! get density for (n,j=l-1/2) i.e. K=L
               K=L
               CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,Z,KAPPA=K)
               RHOC=RHOC+LMULT(N,L)*RHO_
               E=E+(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))
               EIG=EIG+LMULT(N,L)*E
               PP%CLEV(I)=E
               I=I+1
            ENDIF
      ! get density for (n,j=l+1/2) i.e. K=-L-1
            K=-L-1
            CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,Z,KAPPA=K)
            RHOC=RHOC+LMULT(N,L)*RHO_
            E=E+(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))
            EIG=EIG+LMULT(N,L)*E
            PP%CLEV(I)=E
         ELSEIF (RADEQ=="SOC") THEN
! radial scalar relativistic equation including SOC term
      ! get density for (n,j=l-1/2) i.e. K=L
            IF (L/=0) THEN
               K=L
               CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,Z,KAPPA=K,DPOT=DV)
               RHOC=RHOC+LMULT(N,L)*RHO_
               E=E+(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))
               EIG=EIG+LMULT(N,L)*E            
               PP%CLEV(I)=E
               I=I+1
            ENDIF
      ! get density for (n,j=l+1/2) i.e. K=-L-1
            K=-L-1
            CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,Z,KAPPA=K,DPOT=DV)
            RHOC=RHOC+LMULT(N,L)*RHO_
            E=E+(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))
            EIG=EIG+LMULT(N,L)*E
            PP%CLEV(I)=E
         ELSE
! radial scalar relativistic equation
      ! get density for (n,l)
            CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,Z)
            RHOC=RHOC+LMULT(N,L)*RHO_
            E=E+(PP%E(1)+AVERAGEPOT-PP%E_ORIG(1))
            EIG=EIG+LMULT(N,L)*E
            PP%CLEV(I)=E
         ENDIF
         I=I+1
      ENDDO

      RHOC=RHOC*SCALE

      RETURN
      END SUBROUTINE ATOM


!*********************** FUNCTION LMULT *********************************
!
! This function returns the occupation of the core state (N,L),
! possibly reduced by Z_CL in case of final state core level
! shift calculations. The latter feature is controlled by the
! LCLSHIFT logical, that is set by the call to SET_PP_POINTER
!
!************************************************************************
      FUNCTION LMULT(N,L)
      USE cl
      IMPLICIT NONE
      INTEGER N,L
      REAL(q) LMULT
      IF (RADEQ=="DIR".OR.RADEQ=="SOC") THEN
         ! Dirac or scalar relativistic + SOC term
         LMULT=2*(L+0.5)+1
         ! reduce to calculate core level shifts
         IF (LCLSHIFT.AND.N==N_CL.AND.L==L_CL) LMULT=LMULT-Z_CL/2
      ELSE
         ! scalar relaticistic case
         LMULT=4*L+2
         ! reduce to calculate core level shifts
         IF (LCLSHIFT.AND.N==N_CL.AND.L==L_CL) LMULT=LMULT-Z_CL
      ENDIF
      END FUNCTION LMULT


!*********************** SUBROUTINE AE_PAR_WAVE *************************      
!
! This subroutine recalculates the AE partial waves PP%WAE
!
! It updates:
!         PP%V00_AE        : spherical component of total 
!                            onsite AE potential
!         PP%AVERAGEPOT(1) : \int V^1(r)\hat{Q}^{00}(r) dr
!         PP%POTAE         : AE local potential (valence contributions)
!                            PP%V00_AE-V_H[n_Zc]
!         PP%POTAEC        : V_H[n_Zc]
!         PP%WKINAE        : the kinetic energy density of the
!                            AE partial waves
!
! In addition it can write the logarithmic derivatives to a file
! called: LDERAE.type#.step#
!
! N.B.: the partial waves are calculated at the energies
!       PP%E(:)+PP%AVERAGEPOT(1)
!
!************************************************************************      
      SUBROUTINE AE_PAR_WAVE(PP,ITYP)
      
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE main_mpi

      IMPLICIT NONE

      TYPE(potcar) PP

      INTEGER LYMAX,LMMAX
      INTEGER I,CHANNEL,LPREV
      INTEGER Z
      INTEGER ITYP
      REAL(q) E,BC,DBC
      REAL(q) EMIN,EMAX
      REAL(q), PARAMETER :: DE=15._q
      REAL(q) SCALE,WN
      REAL(q), DIMENSION(PP%R%NMAX) :: V00
      REAL(q), ALLOCATABLE :: V(:,:,:), RHO(:,:,:)

      REAL(q), DIMENSION(PP%R%NMAX) :: WTMP
      
      REAL(q) AVERAGEPOT
      
      INTEGER NRWIGS
      
      INTEGER IUNIT
      INTEGER, PARAMETER :: IUBASE=100
      CHARACTER (LEN=16) :: FILEOUT

      REAL(q), PARAMETER :: TINY=1.0E-8_q
      INTEGER, EXTERNAL :: MAXL_AUG
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1),V(PP%R%NMAX,LMMAX,1))
! setup the AE potential
      RHO=0
      RHO(:,1,1)=PP%RHOAE00(:)+PP%RHOAE(:)      
      V=0
      Z=INT(PP%ZVALF_ORIG+PP%ZCORE)
#ifdef testout
!     WRITE(*,*) 'AE_PAR_WAVE: Z_t=',Z
#endif
      CALL POT(RHO,Z,PP%R,V)
!     CALL POT(RHO,Z,PP%R,V,add_gga=.false.)
! take the spherical component      
      SCALE=1/(2*SQRT(PI))      
      V00(:)=V(:,1,1)*SCALE
! store spherical potential
      PP%V00_AE=V00
! recalculate PP%AVERAGEPOT
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+V00(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      PP%AVERAGEPOT(1)=AVERAGEPOT
#ifdef testout      
      WRITE(*,*) 'AE_PAR_WAVE: AVERAGEPOT(AE)=',PP%AVERAGEPOT(1)
#endif
! calculate partial waves and their respective kinetic energy densities
      DO CHANNEL=1,PP%LMAX
         E=PP%E(CHANNEL)+AVERAGEPOT
         ! get partial wave
         CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,PP%WAE(:,CHANNEL),WN=1._q)
         ! adaptively linearize ?
         IF (LADAPTELIN()) CALL ADAPT_ELIN(PP,V00,CHANNEL)
         ! calculate kinetic energy density
         DO I=1,PP%R%NMAX        
            PP%WKINAE(I,CHANNEL)=(E-V00(I))*PP%WAE(I,CHANNEL)
         ENDDO
      ENDDO
#ifdef testout     
!     IF (LADAPTELIN()) THEN
         WRITE(*,*) 'AE_PAR_WAVE: linearizing at:'
         WRITE(*,'(13X,A)') ' CH#        E'
         DO CHANNEL=1,PP%LMAX
            WRITE(*,'(13X,I3,F14.7)') CHANNEL,PP%E(CHANNEL)+AVERAGEPOT
         ENDDO
!     ENDIF
#endif
! update PP%POTAE (spherical part of valence only AE potential)
      RHO=0
      RHO(:,1,1)=PP%RHOAE00(:)
      V=0
      CALL POT(RHO,0,PP%R,V,PP%RHOAE)
!     CALL POT(RHO,0,PP%R,V,PP%RHOAE,add_gga=.false.)     
      PP%POTAE(:)=-V(:,1,1)*SCALE
! and AE core potential
      PP%POTAEC(:)=PP%V00_AE(:)+PP%POTAE(:)      
! match PS core potential to AE core potential beyond PP%RWIGS
      IF (LUSERWIGS()) THEN
         ! which grid point corresponds to PP%RWIGS?
         DO I=1,PP%R%NMAX-1
            IF (PP%R%R(I)>=PP%RWIGS) EXIT
            PP%POTPSC(I)=PP%POTPSC(I)+PP%POTAEC(PP%R%NMAX)-PP%POTPSC(PP%R%NMAX)
         ENDDO
         NRWIGS=I
!        DO I=1,NRWIGS-1
!           PP%POTPSC(I)=PP%POTPSC(I)+PP%POTAEC(NRWIGS)-PP%POTPSC(NRWIGS)
!        ENDDO
         DO I=NRWIGS,PP%R%NMAX
            PP%POTPSC(I)=PP%POTAEC(I)
         ENDDO
      ENDIF
          
! output for analysis ?
      IF (LREPORT()) THEN
! Write logarithmic derivatives to file
      IUNIT=IUBASE+NODE_ME
      WRITE(FILEOUT,'(A7,I4.4,A1,I4.4)') "LDERAE.",ITYP,".",NCALLED
      OPEN(UNIT=IUNIT,FILE=DIR_APP(1:DIR_LEN)//FILEOUT,STATUS='NEW',ERR=110)
      
      EMIN=MINVAL(PP%E(1:PP%LMAX))+AVERAGEPOT-DE
      EMAX=MAXVAL(PP%E(1:PP%LMAX))+AVERAGEPOT+DE

      DO I=1,1000
         E=REAL(EMIN+(I-1)*(EMAX-EMIN)/1000)
         WRITE(UNIT=IUNIT,FMT='(F20.8)',ADVANCE='NO') E
         LPREV=-1
         DO CHANNEL=1,PP%LMAX
            IF (PP%LPS(CHANNEL)==LPREV) CYCLE
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,WTMP,WN=1._q)
            CALL GET_BC(PP%R,WTMP,BC,DBC)
            WRITE(IUNIT,FMT='(F20.8)',ADVANCE='NO') DBC/BC
            LPREV=PP%LPS(CHANNEL)
         ENDDO
         WRITE(IUNIT,FMT='(A1)')
      ENDDO
      
      CLOSE(IUNIT)

 110  CONTINUE
      ENDIF
      
      DEALLOCATE(RHO,V)
      RETURN
      END SUBROUTINE AE_PAR_WAVE


!*********************** SUBROUTINE ADAPT_ELIN **************************      
!
!************************************************************************      

      SUBROUTINE ADAPT_ELIN(PP,V00,CHANNEL)
      USE prec
      USE constant
      USE pseudo
      USE cl
      IMPLICIT NONE

      TYPE(potcar) PP
      INTEGER CHANNEL 
      REAL(q), DIMENSION(PP%R%NMAX) :: V00

      INTEGER MAXNL,IND,N,NADD,L,LNEXT
      INTEGER MAXN_CORE(0:3)
      INTEGER NODES_TRG,NODES_ACT
      INTEGER I,J
      
      REAL(q) OCC     
      REAL(q) AVERAGEPOT,ESTART,E,E1,E2,dE 
      REAL(q), DIMENSION(PP%R%NMAX) :: W0,W1,RHO_
      REAL(q) WRMAX_TRG,WR0,DWR0,WR1,DWR1
      
      LOGICAL LCONT
      
      CHARACTER (LEN=1) :: LV(5) = (/ "s","p","d","f","g"  /)
      
      INTEGER, PARAMETER :: NTRIES=200
      REAL(q), PARAMETER :: WTINY=1E-3_q
      REAL(q), PARAMETER :: DTINY=1E-3_q
      REAL(q), PARAMETER :: ESTEP=1._q
                 
      LCONT=.FALSE.
! Does this channel derive from a bound state? If not, exit early.
! To answer this question we check whether the atomic occupancy of
! the channel, PP%QATO(channel,channel), differs from zero.
      LCONT=IS_BOUND(PP,CHANNEL)
! Adding this condition will also adapt the linearization
! energy of the first of a pair of channels which atomic
! occupancy is zero
      LCONT=(IS_1ST_OF_L(PP,CHANNEL).AND.IS_2ND_OF_L(PP,CHANNEL+1))
! Is this the second occupied channel of a pair with the same
! l-quantum number? If yes we have to up the main quantum
! number with 1
      NADD=0      
      IF (IS_2ND_OF_L(PP,CHANNEL).AND.IS_BOUND(PP,CHANNEL)) THEN
         NADD=1
      ENDIF
! Are we in SEMICOREONLY mode? Then treat only semicore states      
      IF (LSEMICOREONLY()) LCONT=IS_SEMICORE(PP,CHANNEL)

! Do we have work to do here?
      IF (.NOT.LCONT) RETURN
!      
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+V00(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
!      
      MAXN_CORE(0)=0; MAXN_CORE(1)=1; MAXN_CORE(2)=2; MAXN_CORE(3)=3
      CALL CALCULATE_MAX_N_L(REAL(PP%ZCORE,q),MAXNL)
      CALL INIT_N_L(IND,N,L)
      DO
         IF (.NOT.NEXT_N_L(PP%ZCORE,MAXNL,IND,N,L,OCC)) EXIT
         MAXN_CORE(L)=N
      ENDDO
#ifdef testout
      WRITE(*,'(1X,A,A2)',ADVANCE='NO') 'ADAPT_ELIN: found: ',PP%ELEMENT
      DO I=0,MAXVAL(PP%LPS(1:PP%LMAX))
         WRITE(*,'(1X,I1,A1)',ADVANCE='NO') MAXN_CORE(I)+1,LV(I+1)
      ENDDO
      WRITE(*,'(A)') ' (partial waves)'
#endif
! Desired number of nodes      
      L=PP%LPS(CHANNEL)
      N=MAXN_CORE(L)+1+NADD
      NODES_TRG=N-L-1
! Get L-partial wave at ESTART
      ESTART=PP%E(CHANNEL)+AVERAGEPOT      
      CALL PAR_WAVE_FKT(ESTART,L,V00,PP%R,W0(:),WN=1._q)
! Count its nodes
      NODES_ACT=0
      DO I=2,PP%R%NMAX
         NODES_ACT=NODES_ACT+0.501*ABS(SIGN(1._q,W0(I))-SIGN(1._q,W0(I-1)))         
      ENDDO

      IF (NODES_ACT>NODES_TRG.OR.IS_SEMICORE(PP,CHANNEL)) THEN
         DO I=1,PP%R%NMAX
            V00(I)=V00(I)-(ESTART-PP%E_ORIG(CHANNEL))            
         ENDDO
         CALL CORE_WAVE_FKT(RHO_,E,N,L,V00,PP%R,INT(PP%ZVALF_ORIG+PP%ZCORE),A_=W1)
         DO I=1,PP%R%NMAX
            V00(I)=V00(I)+(ESTART-PP%E_ORIG(CHANNEL))            
         ENDDO
         E=E+(ESTART-PP%E_ORIG(CHANNEL))
#ifdef testout
         IF (IS_SEMICORE(PP,CHANNEL)) THEN
            WRITE(*,'(/,"ADAPT_ELIN: treated as semicore state -> channel:",I2," (",I1,A1")")') CHANNEL,N,LV(L+1)
            WRITE(*,'(  "            linearizing at E=",F14.7," (E_OLD=",F14.7,")")') E,PP%E(CHANNEL)+AVERAGEPOT
         ELSE
            WRITE(*,'(/,"ADAPT_ELIN: corrected number of nodes in channel:",I2," (",I1,A1")")') CHANNEL,N,LV(L+1)
            WRITE(*,'(  "            linearizing at E=",F14.7," (E_OLD=",F14.7,")")') E,PP%E(CHANNEL)+AVERAGEPOT
         ENDIF
         WRITE(149+CHANNEL,'(2F14.7)') (PP%R%R(I),W1(I),I=1,PP%R%NMAX)
#endif         
! Store new linearization energy
! (Uncomment following block to shift the "second" channel by the same amount as the "first")
         IF (IS_1ST_OF_L(PP,CHANNEL).AND.IS_2ND_OF_L(PP,CHANNEL+1).AND. &
        &    (.NOT.IS_BOUND(PP,CHANNEL+1))) THEN
            PP%E(CHANNEL+1)=PP%E(CHANNEL+1)+(E-AVERAGEPOT-PP%E(CHANNEL))
         ENDIF
! End of block         
         PP%E(CHANNEL)=E-AVERAGEPOT
         PP%WAE(:,CHANNEL)=W1(:)         
         RETURN
      ENDIF
      
      IF (NODES_ACT<NODES_TRG) THEN
! This shouldn't happen; at least if this is the case I don't
! have a clue yet what to do about it
#ifdef testout
         WRITE(*,'(/,"ADAPT_ELIN: partial wave has wrong number of nodes:",2I4)') NODES_TRG,NODES_ACT
         WRITE(*,'("n=",I2," l=",I2," channel=",I2," e=",F14.7)') N,L,CHANNEL,E
#endif
         RETURN
      ENDIF
      
! Number of nodes is correct, check for divergency,
! defined here as sign(dW/dr)=sign(W)    
      CALL GET_BC(PP%R,W0,WR0,DWR0)
      IF (SIGN(1._q,WR0)==SIGN(1._q,DWR0)) THEN
! W0 is diverging at PP%R%NMAX, start searching for
! the nearest linearization energy at which dW/dr=0,      
         E1=ESTART
! Second bound of search interval
         dE=ESTEP
         E2=ESTART
         bound2: DO
            E2=E2+dE
            CALL PAR_WAVE_FKT(E2,L,V00,PP%R,W1(:),WN=1._q)
            CALL GET_BC(PP%R,W1,WR1,DWR1)
#ifdef testout
!           WRITE(*,*) 'CHANNEL=',CHANNEL
!           WRITE(*,*) 'N=',N,'L=',L
!           WRITE(*,*) 'ESTART=',ESTART,'DE=',DE,'E2=',E2
!           WRITE(*,*) 'WR0=',WR0,'DWR0=',DWR0
!           WRITE(*,*) 'WR1=',WR1,'DWR1=',DWR1
!           WRITE(*,*)
#endif
            IF (SIGN(1._q,W1(PP%R%NMAX))/=SIGN(1._q,W0(PP%R%NMAX))) THEN
               E2=E2-dE
               dE=dE/2
            ELSEIF (ABS(DWR1)>ABS(DWR0).AND. &
           &         SIGN(1._q,DWR1)==SIGN(1._q,DWR0)) THEN
               E2=E2-dE
               dE=-dE
            ELSEIF (SIGN(1._q,DWR1)/=SIGN(1._q,DWR0)) THEN
               EXIT bound2
            ENDIF
         ENDDO bound2
! Start bisectioning
         bisec2: DO J=1,NTRIES+1
            E=E2+(E1-E2)/2
            CALL PAR_WAVE_FKT(E,L,V00,PP%R,W1(:),WN=1._q)
            CALL GET_BC(PP%R,W1,WR1,DWR1)
            ! Break condition 
            IF (ABS(DWR1)<DTINY) THEN
               EXIT bisec2
            ENDIF
            ! Adjust interval
            IF (SIGN(1._q,DWR1)==SIGN(1._q,DWR0)) THEN
               E1=E
            ELSE
               E2=E
            ENDIF
         ENDDO bisec2
         IF (J==NTRIES+2) THEN
! Unable to find E within NTRIES steps
            WRITE(*,'(/,"ADAPT_ELIN: solution (2) not found within ",I4," attempts")') NTRIES
            WRITE(*,'("n=",I2," l=",I2," channel=",I2," e=",F14.7)') N,L,CHANNEL,PP%E(CHANNEL)
            RETURN
         ENDIF
! Count nodes (doublecheck)
         NODES_ACT=0
         DO I=2,PP%R%NMAX
            NODES_ACT=NODES_ACT+0.501*ABS(SIGN(1._q,W1(I))-SIGN(1._q,W1(I-1)))         
         ENDDO
         IF (NODES_ACT/=NODES_TRG) THEN
! Something went wrong, now we have the wrong number of nodes in our partial wave
            WRITE(*,'(/,"ADAPT_ELIN: partial wave has wrong number of nodes:",2I4)') NODES_TRG,NODES_ACT
            WRITE(*,'("n=",I2," l=",I2," channel=",I2," e=",F14.7)') N,L,CHANNEL,E
            RETURN
         ENDIF
#ifdef testout
         WRITE(*,'(/,"ADAPT_ELIN: corrected divergency in channel:",I2," (",I1,A1")")') CHANNEL,N,LV(L+1)
         WRITE(*,'(  "            linearizing at E=",F14.7," (E_OLD=",F14.7,")")') E,PP%E(CHANNEL)+AVERAGEPOT
!        WRITE(159+CHANNEL,'(2F14.7)') (PP%R%R(i),W1(i),i=1,PP%R%NMAX)
#endif         
! Store new linearization energy
! (Uncomment following block to shift the "second" channel by the same amount as the "first")
         IF (IS_1ST_OF_L(PP,CHANNEL).AND.IS_2ND_OF_L(PP,CHANNEL+1).AND. &
        &    (.NOT.IS_BOUND(PP,CHANNEL+1))) THEN
            PP%E(CHANNEL+1)=PP%E(CHANNEL+1)+(E-AVERAGEPOT-PP%E(CHANNEL))
         ENDIF
! End of block         
         PP%E(CHANNEL)=E-AVERAGEPOT
         PP%WAE(:,CHANNEL)=W1(:)
         RETURN
      ENDIF
               
      RETURN      
      END SUBROUTINE ADAPT_ELIN


!*********************** SUBROUTINE POTPSC_UPD **************************
!
! Update PP%POTPSC (V_H[\tilde{n}_Zc]) in accordance with increased 
! ionicity, Z_CL, of the pseudo core in the case of a final state core 
! level shift calculation.
!
!************************************************************************      

      SUBROUTINE POTPSC_UPD(PP)

      USE prec
      USE constant
      USE pseudo
      USE radial
      USE cl
      
      IMPLICIT NONE

      TYPE(potcar) PP

      REAL(q) DRHOC(PP%R%NMAX)
      REAL(q) DVC(PP%R%NMAX)
      REAL(q) DHARTREE
      INTEGER I
      
      IF (.NOT.LCLSHIFT) RETURN
      
      DRHOC(:)=PP%AUG(:,0)*Z_CL*(1._q/(2*SQRT(PI)))
      CALL RAD_POT_HAR(0,PP%R,DVC,DRHOC,DHARTREE)
      PP%POTPSC=PP%POTPSC-DVC*(1._q/(2*SQRT(PI)))

!     PP%RHOPS(:)=PP%RHOPS(:)*(PP%ZVALF/PP%ZVALF_ORIG) !????
      
      RETURN
      END SUBROUTINE POTPSC_UPD
      
!*********************** SUBROUTINE PS_PAR_WAVE *************************      
!
! This subroutine recalculates the pseudo partial waves PP%WPS, and
! the PAW parameters (in the manner described in ...):
!
!         PP%DIJ
!         PP%QIJ
!         PP%QION
!         PP%C
!
! The PP%DIJ are unscreened in SET_PAW_AUG_RC and then stored in PP%DION
!
! It updates:
!         PP%V00_PS        : spherical component of total 
!                            onsite PS potential
!         PP%AVERAGEPOT(2) : \int \tilde{V}^1(r)\hat{Q}^{00}(r) dr
!         PP%POTPS         : PS local potential (valence contributions)
!                            PP%V00_PS-PP%POTPSC
!         PP%WKINPS        : the kinetic energy density of the
!                            PS partial waves
! It also transforms:
!         PP%WAE
!         PP%WKINAE
! so that they are consistent with the old projectors
! (See Eq. (X) in M.Marsman, G.Kresse, to be published)
!
! In addition it can write the logarithmic derivatives to a file
! called: LDERPS.type#.step#
!
! N.B.: the partial waves are calculated at the energies
!       PP%E(:)+PP%AVERAGEPOT(1), so beware that the latter variable
!       has been set correctly (a preceding call ATOM or AE_PAR_WAVE
!       ensures this) 
!
!************************************************************************      
#ifdef bloechl      
      SUBROUTINE PS_PAR_WAVE(PP,ITYP)
     
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE main_mpi
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER LYMAX,LMMAX
      INTEGER I,J,K,L,M,N,LPREV,CHANNEL
      INTEGER ITYP
      INTEGER NRWIGS
      
      INTEGER IPIV(2)
      REAL(q) WRK(2),RCOND,DET(2)
      REAL(q) A(3,2),B1(3),B2(3),C(2,3),SC(2)
      REAL(q) OVL(2,2,3)
      REAL(q) QAE(2,2),QPS(2,2)
      REAL(q) BIJ(2,2),BIN(2,2)
      REAL(q) DIJ(2,2),QIJ(2,2)
      REAL(q) BC,DBC
      REAL(q) E
      REAL(q) AVERAGEPOT
      REAL(q) TMP(2,2)
      REAL(q) SCALE
      REAL(q) CDUM(3)
      REAL(q) EMIN,EMAX
      REAL(q), PARAMETER :: DE=15._q

      REAL(q), DIMENSION(PP%LMAX,3) :: CNEW
      REAL(q), DIMENSION(PP%R%NMAX) :: V00
      REAL(q), ALLOCATABLE :: V(:,:,:), RHO(:,:,:)
      REAL(q), DIMENSION(PP%R%NMAX,2,3) :: W
      REAL(q), DIMENSION(PP%R%NMAX,2) :: WTMP
      REAL(q), DIMENSION(PP%R%NMAX,PP%LMAX) :: WFAE_DUAL
      REAL(q), DIMENSION(PP%R%NMAX,PP%LMAX) :: WKINAE_DUAL
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK
      real(q) :: sum,sum2,gc
      REAL(q), PARAMETER :: TINY=1.0E-6
      
      INTEGER IUNIT
      INTEGER, PARAMETER :: IUBASE=100
      CHARACTER (LEN=16) :: FILEOUT
      
      INTEGER, EXTERNAL :: MAXL_AUG
#ifdef testout
      WRITE(*,*)
      WRITE(*,*) 'Re-pseudize the partial waves'
      WRITE(*,*) '-----------------------------------------'
#endif
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1),V(PP%R%NMAX,LMMAX,1))
! setup the potential
      RHO=0
      RHO(:,1,1)=PP%RHOPS00(:)
      V=0
      CALL POT(RHO,0,PP%R,V,PP%RHOPS,PP%POTPSC)
! take the spherical component
      SCALE=1/(2*SQRT(PI))      
      V00(:)=V(:,1,1)*SCALE
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+V00(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      PP%AVERAGEPOT(2)=AVERAGEPOT
#ifdef testout
      WRITE(*,*) 'PS_PAR_WAVE: AVERAGEPOT(PS)=',PP%AVERAGEPOT(2)
#endif
! early exit in case LMIMICFC=.TRUE. (only PP%AVERAGEPOT(2) is set)
      IF (LMIMFC()) RETURN
! store spherical potential
      PP%V00_PS=V00
! solve for partial waves
      IF (LUSERWIGS()) THEN
         ! which grid point corresponds to PP%RWIGS?
         DO I=1,PP%R%NMAX-1
            IF (PP%R%R(I)>=PP%RWIGS) EXIT
         ENDDO
         NRWIGS=I
      ENDIF
      CNEW=0
      LPREV=-1
      chann: DO CHANNEL=1,PP%LMAX
         IF (PP%LPS(CHANNEL)/=LPREV) THEN
         ! first linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,2),INH=PP%BETA(:,CHANNEL))
            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2))
            ENDIF
            B1(1)=(A(1,2)-DBC/BC*A(1,1))
            B1(2)=(A(2,2)-DBC/BC*A(2,1))
            C(1,1)=1._q
            C(1,2)=-B1(1)/B1(2)
            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL)=C(1,1)*W(I,1,1)+C(1,2)*W(I,1,2)
            ENDDO

            SC(1)=PP%WPS(PP%R%NMAX,CHANNEL)/PP%WAE(PP%R%NMAX,CHANNEL)

            IF (LUSERWIGS()) THEN
               IF (ABS(PP%BETA(NRWIGS,CHANNEL))>TINY) THEN
                  WRITE(*,'(A)') 'PS_PAR_WAVE: WARNING'
                  WRITE(*,'(A,I2,A)') &
                 & '  projector for channel',CHANNEL,' non-neglegible at matching radius:'
                  WRITE(*,'(A,F8.5,A,F10.6)') &
                 & '  beta(',PP%R%R(NRWIGS),') =',PP%BETA(NRWIGS,CHANNEL)
               ENDIF
               SC(1)=PP%WPS(NRWIGS,CHANNEL)/PP%WAE(NRWIGS,CHANNEL)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL)=SC(1)*PP%WAE(I,CHANNEL)
               ENDDO
            ENDIF

            CNEW(CHANNEL,1:2)=C(1,1:2)
            
            DO I=1,PP%R%NMAX
               PP%WKINPS(I,CHANNEL)=(E-V00(I))*PP%WPS(I,CHANNEL)
            ENDDO
            
            DO I=1,PP%R%NMAX
               WORK(I)=-C(1,2)*PP%BETA(I,CHANNEL)*PP%WPS(I,CHANNEL)
            ENDDO
            CALL SIMPI(PP%R,WORK,BIJ(1,1))
            
            DO I=1,PP%R%NMAX
               WORK(I)=PP%WAE(I,CHANNEL)*PP%WAE(I,CHANNEL)
            ENDDO
            CALL SIMPI(PP%R,WORK,QAE(1,1))
            
            DO I=1,PP%R%NMAX
               WORK(I)=PP%WPS(I,CHANNEL)*PP%WPS(I,CHANNEL)
            ENDDO
            CALL SIMPI(PP%R,WORK,QPS(1,1))
            
            QIJ(1,1)=QAE(1,1)*SC(1)*SC(1)-QPS(1,1)

            PP%E(:)=PP%E(:)+PP%AVERAGEPOT(1)
            
            DIJ(1,1)=BIJ(1,1)*(2*RYTOEV)+PP%E(CHANNEL)*QIJ(1,1)
            DIJ(1,1)=C(1,2)/BIJ(1,1)*DIJ(1,1)*C(1,2)/BIJ(1,1)
            QIJ(1,1)=C(1,2)/BIJ(1,1)*QIJ(1,1)*C(1,2)/BIJ(1,1)

            PP%DIJ(CHANNEL,CHANNEL)=DIJ(1,1)
            PP%QIJ(CHANNEL,CHANNEL)=QIJ(1,1)

            PP%E(:)=PP%E(:)-PP%AVERAGEPOT(1)
            
            DO I=1,PP%R%NMAX
               WTMP(I,1)=C(1,2)/BIJ(1,1)*PP%WPS(I,CHANNEL)
            ENDDO
            PP%WPS(:,CHANNEL)=WTMP(:,1)

            DO I=1,PP%R%NMAX
               PP%WKINPS(I,1)=C(1,2)/BIJ(1,1)*PP%WKINPS(I,CHANNEL)
            ENDDO
            
            DO I=1,PP%R%NMAX
               WTMP(I,1)=C(1,2)/BIJ(1,1)*PP%WAE(I,CHANNEL)*SC(1)
            ENDDO
            WFAE_DUAL(:,CHANNEL)=WTMP(:,1)

            DO I=1,PP%R%NMAX
               WTMP(I,1)=C(1,2)/BIJ(1,1)*PP%WKINAE(I,CHANNEL)*SC(1)
            ENDDO
            WKINAE_DUAL(:,CHANNEL)=WTMP(:,1)                                    
         ELSE
         ! first linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL-1)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL-1),V00,PP%R,W(:,1,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL-1),V00,PP%R,W(:,1,2),INH=PP%BETA(:,CHANNEL-1))
            ! inhomogenous solution using the second projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL-1),V00,PP%R,W(:,1,3),INH=PP%BETA(:,CHANNEL))
         ! second linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,2,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,2,2),INH=PP%BETA(:,CHANNEL-1))
            ! inhomogenous solution using the second projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,2,3),INH=PP%BETA(:,CHANNEL))
            ! calculate the overlap between the projectors and W(1:6) to be able
            OVL=0
            DO I=1,2
               ! projectors should be zero at matching radius
               IF (LUSERWIGS().AND.ABS(PP%BETA(NRWIGS,CHANNEL-2+I))>TINY) THEN
                  WRITE(*,'(A)') 'PS_PAR_WAVE: WARNING'
                  WRITE(*,'(A,I2,A)') &
                 & '  projector for channel',CHANNEL-2+I,' non-neglegible at matching radius:'
                  WRITE(*,'(A,F8.5,A,F10.6)') &
                 & '  beta(',PP%R%R(NRWIGS),') =',PP%BETA(NRWIGS,CHANNEL-2+I)
               ENDIF
            DO J=1,2
            DO K=1,3
               DO N=1,PP%R%NMAX
                  WORK(N)=PP%BETA(N,CHANNEL-2+I)*W(N,J,K)
               ENDDO
               CALL SIMPI(PP%R,WORK,OVL(I,J,K))
            ENDDO
            ENDDO
            ENDDO

            DO I=1,2
            DO J=1,2
               DO K=1,PP%R%NMAX
                  WORK(K)=PP%WAE(K,CHANNEL-2+I)*PP%WAE(K,CHANNEL-2+J)
               ENDDO
               CALL SIMPI(PP%R,WORK,QAE(I,J))
            ENDDO
            ENDDO

            ! |\psi_1> = c_11 |\psi_11> + c_12 |\psi_12> + c_13 |\psi_13>
            ! get C(1,2) for C(1,1)=1, and C(1,3)=0
            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL-1),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,3),A(3,1),A(3,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL-1),BC,DBC)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2))
               CALL GET_BC(PP%R,W(:,1,3),A(3,1),A(3,2))
            ENDIF
            B1(1)=(A(1,2)-DBC/BC*A(1,1))
            B1(2)=(A(2,2)-DBC/BC*A(2,1))
            B1(3)=(A(3,2)-DBC/BC*A(3,1))

! Procedure 1:
! set weight of 2st inhomogeneous solution, c_13, to zero 
            C(1,1)=1._q
            C(1,3)=0
            C(1,2)=-B1(1)/B1(2)-C(1,3)*B1(3)/B1(2)
!
! Procedure 2:
! fix weight of 1st inhomogeneous solution, c_12, at the 'atomic' value
!           C(1,1)=1._q
!           C(1,2)=PP%C(CHANNEL-1,2)
!           C(1,3)=-B1(1)/B1(3)-C(1,2)*B1(2)/B1(3)
!
! Procedure 3:
! fix weight of 2st inhomogeneous solution, c_13, at the 'atomic' value
!           C(1,1)=1._q
!           C(1,3)=PP%C(CHANNEL-1,3)
!           C(1,2)=-B1(1)/B1(2)-C(1,3)*B1(3)/B1(2)
!
! Procedure 4:
! take second boundary condition into account
!           C(1,1)=1._q
!           C(1,3)=(B1(1)/B1(2)-B2(1)/B2(2))/(B2(3)/B2(2)-B1(3)/B1(2))
!           C(1,2)=-B1(1)/B1(2)-C(1,3)*B1(3)/B1(2)
!
            CNEW(CHANNEL-1,:)=C(1,:)

            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL-1)=C(1,1)*W(I,1,1)+C(1,2)*W(I,1,2)+C(1,3)*W(I,1,3)
            ENDDO

            SC(1)=PP%WPS(PP%R%NMAX,CHANNEL-1)/PP%WAE(PP%R%NMAX,CHANNEL-1)

            IF (LUSERWIGS()) THEN
               SC(1)=PP%WPS(NRWIGS,CHANNEL-1)/PP%WAE(NRWIGS,CHANNEL-1)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL-1)=SC(1)*PP%WAE(I,CHANNEL-1)
               ENDDO
            ENDIF
            
            ! setup the necessary matrix to allow for a fast calculation
            ! of C(2,1) and C(2,3) for a given C(2,2)
            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,2),A(2,1),A(2,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,3),A(3,1),A(3,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC)
               CALL GET_BC(PP%R,W(:,2,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,2,2),A(2,1),A(2,2))
               CALL GET_BC(PP%R,W(:,2,3),A(3,1),A(3,2))            
            ENDIF
            B1(1)=(A(1,2)-DBC/BC*A(1,1))
            B1(2)=(A(2,2)-DBC/BC*A(2,1))
            B1(3)=(A(3,2)-DBC/BC*A(3,1))

! Procedure 1:
! fix weight of 1st inhomogeneous solution, c_22, at 'atomic' value
!           C(2,1)=1._q
!           C(2,2)=PP%C(CHANNEL,2)
!           C(2,3)=-B1(1)/B1(3)-C(2,2)*B1(2)/B1(3)
!
! Procedure 2:
! fix weight of 2nd inhomogeneous solution, c_23, at 'atomic' value
            C(2,1)=1._q            
            C(2,3)=PP%C(CHANNEL,3)
            C(2,2)=-B1(1)/B1(2)-C(2,3)*B1(3)/B1(2)
!            
! Procedure 3:
! take second boundary condition into account
!           C(2,1)=1._q
!           C(2,3)=(B1(1)/B1(2)-B2(1)/B2(2))/(B2(3)/B2(2)-B1(3)/B1(2))
!           C(2,2)=-B1(1)/B1(2)-C(2,3)*B1(3)/B1(2)
!
            CNEW(CHANNEL,:)=C(2,:)

            PP%E(:)=PP%E(:)+PP%AVERAGEPOT(1)

            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL)=C(2,1)*W(I,2,1)+C(2,2)*W(I,2,2)+C(2,3)*W(I,2,3)
            ENDDO

            SC(2)=PP%WPS(PP%R%NMAX,CHANNEL)/PP%WAE(PP%R%NMAX,CHANNEL)

            IF (LUSERWIGS()) THEN
               SC(2)=PP%WPS(NRWIGS,CHANNEL)/PP%WAE(NRWIGS,CHANNEL)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL)=SC(2)*PP%WAE(I,CHANNEL)
               ENDDO
            ENDIF
            
            DO I=1,PP%R%NMAX
               PP%WKINPS(I,CHANNEL-1)=(V00(PP%R%NMAX)+PP%E(CHANNEL-1)-V00(I))*PP%WPS(I,CHANNEL-1)
               PP%WKINPS(I,CHANNEL  )=(V00(PP%R%NMAX)+PP%E(CHANNEL  )-V00(I))*PP%WPS(I,CHANNEL  )
            ENDDO
            
            QPS=0
            DO I=1,2
            DO J=1,2
               DO K=1,PP%R%NMAX
                  WORK(K)=PP%WPS(K,CHANNEL-2+I)*PP%WPS(K,CHANNEL-2+J)
               ENDDO
               CALL SIMPI(PP%R,WORK,QPS(I,J))
            ENDDO
            ENDDO

            QIJ=0
            DO I=1,2
            DO J=1,2
               QIJ(I,J)=QAE(I,J)*SC(I)*SC(J)-QPS(I,J)
            ENDDO
            ENDDO

            ! setup the matrix BIJ = < \tilde{\psi}_i | \chi_j |>
            BIJ=0
            DO I=1,2
            DO J=1,2
               DO M=1,2
               DO N=1,3
                  BIJ(I,J)=BIJ(I,J)-C(I,N)*OVL(M,I,N)*C(J,M+1)
               ENDDO
               ENDDO
            ENDDO
            ENDDO
            
            ! invert
            BIN=BIJ
            CALL DGECO(BIN,2,2,IPIV,RCOND,WRK)
            CALL DGEDI(BIN,2,2,IPIV,DET,WRK,'01')
            
            DIJ=0
            DO I=1,2
            DO J=1,2
               DIJ(I,J)=BIJ(I,J)*(2*RYTOEV)+PP%E(CHANNEL-2+J)*QIJ(I,J)
            ENDDO
            ENDDO

            DIJ(1,2)=(DIJ(1,2)+DIJ(2,1))/2
            DIJ(2,1)=DIJ(1,2)

            TMP=0
            DO L=1,2
            DO N=1,2
               DO I=1,2
               DO J=1,2
               DO K=1,2
               DO M=1,2
                  TMP(L,N)=TMP(L,N)+C(K,L+1)*BIN(K,I)*DIJ(I,J)*BIN(M,J)*C(M,N+1)
               ENDDO
               ENDDO
               ENDDO
               ENDDO
            ENDDO
            ENDDO
            DIJ=TMP
            PP%DIJ(CHANNEL-1:CHANNEL,CHANNEL-1:CHANNEL)=DIJ(1:2,1:2)
            
            TMP=0
            DO L=1,2
            DO N=1,2
               DO I=1,2
               DO J=1,2
               DO K=1,2
               DO M=1,2
                  TMP(L,N)=TMP(L,N)+C(K,L+1)*BIN(K,I)*QIJ(I,J)*BIN(M,J)*C(M,N+1)
               ENDDO
               ENDDO
               ENDDO
               ENDDO
            ENDDO
            ENDDO
            QIJ=TMP
            PP%QIJ(CHANNEL-1:CHANNEL,CHANNEL-1:CHANNEL)=QIJ(1:2,1:2)
            
            PP%E(:)=PP%E(:)-PP%AVERAGEPOT(1)
            
            WTMP=0
            DO K=1,2
               DO I=1,2
               DO J=1,2
                  DO N=1,PP%R%NMAX
                     WTMP(N,K)=WTMP(N,K)+C(J,K+1)*BIN(J,I)*PP%WPS(N,CHANNEL-2+I)
                  ENDDO
               ENDDO
               ENDDO
            ENDDO
#ifdef testout
!           WRITE(299+CHANNEL-1,'(2F14.7)') (PP%R%R(I),PP%WPS(I,CHANNEL-1),I=1,PP%R%NMAX)
#endif
            PP%WPS(:,CHANNEL-1)=WTMP(:,1)
            PP%WPS(:,CHANNEL  )=WTMP(:,2)

            WTMP=0
            DO K=1,2
               DO I=1,2
               DO J=1,2
                  DO N=1,PP%R%NMAX
                     WTMP(N,K)=WTMP(N,K)+C(J,K+1)*BIN(J,I)*PP%WAE(N,CHANNEL-2+I)*SC(I)
                  ENDDO
               ENDDO
               ENDDO
            ENDDO
#ifdef testout
!           WRITE(199+CHANNEL-1,'(2F14.7)') (PP%R%R(I),PP%WAE(I,CHANNEL-1),I=1,PP%R%NMAX)
#endif
            WFAE_DUAL(:,CHANNEL-1)=WTMP(:,1)
            WFAE_DUAL(:,CHANNEL  )=WTMP(:,2)

            WTMP=0
            DO K=1,2
               DO I=1,2
               DO J=1,2
                  DO N=1,PP%R%NMAX
                     WTMP(N,K)=WTMP(N,K)+C(J,K+1)*BIN(J,I)*PP%WKINPS(N,CHANNEL-2+I)
                  ENDDO
               ENDDO
               ENDDO
            ENDDO
            PP%WKINPS(:,CHANNEL-1)=WTMP(:,1)
            PP%WKINPS(:,CHANNEL  )=WTMP(:,2)

            WTMP=0
            DO K=1,2
               DO I=1,2
               DO J=1,2
                  DO N=1,PP%R%NMAX
                     WTMP(N,K)=WTMP(N,K)+C(J,K+1)*BIN(J,I)*PP%WKINAE(N,CHANNEL-2+I)*SC(I)
                  ENDDO
               ENDDO
               ENDDO
            ENDDO
            WKINAE_DUAL(:,CHANNEL-1)=WTMP(:,1)
            WKINAE_DUAL(:,CHANNEL  )=WTMP(:,2)
         ENDIF         
         LPREV=PP%LPS(CHANNEL)
      ENDDO chann
      PP%WAE=WFAE_DUAL
      PP%WKINAE=WKINAE_DUAL

      PP%QION=PP%QIJ
      PP%C=CNEW
#ifdef testout
      WRITE(*,*) 'AIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F12.8)') (CNEW(I,J), J=1,3)
      ENDDO
      WRITE(*,*) 'DIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F12.8)') (PP%DIJ(I,J), J=1,PP%LMAX)
      ENDDO
      WRITE(*,*) 'QIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F12.8)') (PP%QIJ(I,J), J=1,PP%LMAX)
      ENDDO      
#endif

! update PP%POTPS (spherical part of valence only PS potential)
      PP%POTPS(:)=-(V00(:)-PP%POTPSC(:))
#ifdef testout
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+PP%POTPSC(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      WRITE(*,*) 'PS_PAR_WAVE: AVERAGEPOT(PSC)=',AVERAGEPOT

      WRITE(*,*)
      WRITE(*,*) 'PS_PAR_WAVE: duality < p_i | psi_j >'
      DO K=1,PP%LMAX
      DO J=1,PP%LMAX
         IF (PP%LPS(K)==PP%LPS(J)) THEN     
            DO I=1,PP%R%NMAX
               WORK(I)=PP%BETA(I,K)*PP%WPS(I,J)
            ENDDO
            CALL SIMPI(PP%R,WORK,RCOND)
         ELSE
            RCOND=0._q
         ENDIF
         WRITE(*,'(F14.7)',ADVANCE='NO') RCOND
      ENDDO
         WRITE(*,*)
      ENDDO
#endif
! output for analysis
      IF (LREPORT()) THEN
! Write logarithmic derivatives to file
      IUNIT=IUBASE+NODE_ME
      WRITE(FILEOUT,'(A7,I4.4,A1,I4.4)') "LDERPS.",ITYP,".",NCALLED
      OPEN(UNIT=IUNIT,FILE=DIR_APP(1:DIR_LEN)//FILEOUT,STATUS='NEW',ERR=120)      

      EMIN=MINVAL(PP%E(1:PP%LMAX))+PP%AVERAGEPOT(1)-DE
      EMAX=MAXVAL(PP%E(1:PP%LMAX))+PP%AVERAGEPOT(1)+DE
      
      DO I=1,1000
         E=REAL(EMIN+(I-1)*(EMAX-EMIN)/1000)
         WRITE(UNIT=IUNIT,FMT='(F20.8)',ADVANCE='NO') E
         LPREV=-1
         DO CHANNEL=1,PP%LMAX
            IF (PP%LPS(CHANNEL)/=LPREV) THEN
               CALL PSSOLVE(PP,V00(1:PP%R%NMAX),PP%DIJ,PP%QIJ,E,PP%BETA,PP%LPS(CHANNEL), &
              & CDUM,WORK)
               CALL GET_BC(PP%R,WORK,BC,DBC)
               WRITE(IUNIT,FMT='(F20.8)',ADVANCE='NO') DBC/BC
            ENDIF
            LPREV=PP%LPS(CHANNEL)
         ENDDO
         WRITE(IUNIT,FMT='(A1)')
      ENDDO
      
      CLOSE(IUNIT)

 120  CONTINUE
      ENDIF
            
      DEALLOCATE(RHO,V)

      RETURN
      END SUBROUTINE PS_PAR_WAVE
#else
      SUBROUTINE PS_PAR_WAVE(PP,ITYP)
     
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE main_mpi
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER LYMAX,LMMAX
      INTEGER I,J,K,L,M,N,LPREV,CHANNEL
      INTEGER ITYP
      INTEGER NRWIGS
      
      REAL(q) A(3,2),B1(3),B2(3),C(2,3),SC(2)
      REAL(q) OVL(2,2,3)
      REAL(q) QAE(2,2),QPS(2,2)
      REAL(q) BIJ(2,2),BIN(2,2)
      REAL(q) DIJ(2,2),QIJ(2,2)
      REAL(q) BC,DBC
      REAL(q) E
      REAL(q) AVERAGEPOT
      REAL(q) TMP(2,2)
      REAL(q) SCALE
      REAL(q) CDUM(3)
      REAL(q) EMIN,EMAX
      REAL(q), PARAMETER :: DE=15._q

      REAL(q), DIMENSION(PP%LMAX,3) :: CNEW
      REAL(q), DIMENSION(PP%R%NMAX) :: V00
      REAL(q), ALLOCATABLE :: V(:,:,:), RHO(:,:,:)
      REAL(q), DIMENSION(PP%R%NMAX,2,3) :: W
      REAL(q), DIMENSION(PP%R%NMAX,2) :: WTMP
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK
! variables for "new" procedure
      INTEGER IPV(6)
      REAL(q) AA(6,6),BB(6),RCOND,WRK(6)
!      
      REAL(q), PARAMETER :: TINY=1.0E-6
      
      INTEGER IUNIT
      INTEGER, PARAMETER :: IUBASE=100
      CHARACTER (LEN=16) :: FILEOUT
      
      INTEGER, EXTERNAL :: MAXL_AUG
#ifdef testout
      WRITE(*,*)
      WRITE(*,*) 'Re-pseudize the partial waves'
      WRITE(*,*) '-----------------------------------------'
#endif
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1),V(PP%R%NMAX,LMMAX,1))
! setup the potential
      RHO=0
      RHO(:,1,1)=PP%RHOPS00(:)
      V=0
      CALL POT(RHO,0,PP%R,V,PP%RHOPS,PP%POTPSC)
!     CALL POT(RHO,0,PP%R,V,PP%RHOPS,PP%POTPSC,add_gga=.false.)
! take the spherical component
      SCALE=1/(2*SQRT(PI))      
      V00(:)=V(:,1,1)*SCALE
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+V00(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      PP%AVERAGEPOT(2)=AVERAGEPOT
#ifdef testout
      WRITE(*,*) 'PS_PAR_WAVE: AVERAGEPOT(PS)=',PP%AVERAGEPOT(2)
#endif
! early exit in case LMIMICFC=.TRUE. (only PP%AVERAGEPOT(2) is set)
      IF (LMIMFC()) RETURN
! store spherical potential
      PP%V00_PS=V00
! solve for partial waves
      IF (LUSERWIGS()) THEN
         ! which grid point corresponds to PP%RWIGS?
         DO I=1,PP%R%NMAX-1
            IF (PP%R%R(I)>=PP%RWIGS) EXIT
         ENDDO
         NRWIGS=I
      ENDIF
      CNEW=0
      LPREV=-1
      chann: DO CHANNEL=1,PP%LMAX
         IF (IS_ONLY_OF_L(PP,CHANNEL)) THEN
         ! first linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,2),INH=PP%BETA(:,CHANNEL))
            OVL=0
            ! projectors should be zero at matching radius
            IF (LUSERWIGS().AND.ABS(PP%BETA(NRWIGS,CHANNEL))>TINY) THEN
               WRITE(*,'(A)') 'PS_PAR_WAVE: WARNING'
               WRITE(*,'(A,I2,A)') &
              & '  projector for channel',CHANNEL,' non-neglegible at matching radius:'
               WRITE(*,'(A,F8.5,A,F10.6)') &
              & '  beta(',PP%R%R(NRWIGS),') =',PP%BETA(NRWIGS,CHANNEL)
            ENDIF
            DO K=1,2
               DO N=1,PP%R%NMAX
                  WORK(N)=PP%BETA(N,CHANNEL)*W(N,1,K)
               ENDDO
               CALL SIMPI(PP%R,WORK,OVL(1,1,K))
            ENDDO

            AA=0._q; BB=0._q

            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2))
            ENDIF
            AA(2,1)=A(1,2)-A(1,1)*DBC/BC
            AA(2,2)=A(2,2)-A(2,1)*DBC/BC
            
            AA(1,1)=OVL(1,1,1); AA(1,2)=OVL(1,1,2)
            
            BB(1)=1._q

            CALL DGECO(AA(1:2,1:2),2,2,IPV(1:2),RCOND,WRK(1:2))
            CALL DGESL(AA(1:2,1:2),2,2,IPV(1:2),BB(1:2),0)


            C(1,1)=BB(1); C(1,2)=BB(2); C(1,3)=0._q
            CNEW(CHANNEL,:)=C(1,:)
            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL)=C(1,1)*W(I,1,1)+C(1,2)*W(I,1,2)
            ENDDO
            SC(1)=PP%WPS(PP%R%NMAX,CHANNEL)/PP%WAE(PP%R%NMAX,CHANNEL)
            IF (LUSERWIGS()) THEN
               SC(1)=PP%WPS(NRWIGS,CHANNEL)/PP%WAE(NRWIGS,CHANNEL)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL)=SC(1)*PP%WAE(I,CHANNEL)
               ENDDO
            ENDIF

            DO I=1,PP%R%NMAX
               PP%WKINPS(I,CHANNEL)=(E-V00(I))*PP%WPS(I,CHANNEL)
            ENDDO

            DO N=1,PP%R%NMAX
               PP%WAE(N,CHANNEL)=PP%WAE(N,CHANNEL)*SC(1)
               PP%WKINAE(N,CHANNEL)=PP%WKINAE(N,CHANNEL)*SC(1)
            ENDDO
                        
            DO I=1,PP%R%NMAX
               WORK(I)=PP%WAE(I,CHANNEL)*PP%WAE(I,CHANNEL)
            ENDDO
            CALL SIMPI(PP%R,WORK,QAE(1,1))
            
            DO I=1,PP%R%NMAX
               WORK(I)=PP%WPS(I,CHANNEL)*PP%WPS(I,CHANNEL)
            ENDDO
            CALL SIMPI(PP%R,WORK,QPS(1,1))
            
            QIJ(1,1)=QAE(1,1)-QPS(1,1)

            BIJ(1,1)=-C(1,2)

            PP%E(:)=PP%E(:)+PP%AVERAGEPOT(1)
            
            DIJ(1,1)=BIJ(1,1)*(2*RYTOEV)+PP%E(CHANNEL)*QIJ(1,1)

            PP%E(:)=PP%E(:)-PP%AVERAGEPOT(1)
            
            PP%DIJ(CHANNEL,CHANNEL)=DIJ(1,1)
            PP%QIJ(CHANNEL,CHANNEL)=QIJ(1,1)
         ELSE
            IF (IS_2ND_OF_L(PP,CHANNEL)) CYCLE
            ! first linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,2),INH=PP%BETA(:,CHANNEL))
            ! inhomogenous solution using the second projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL),V00,PP%R,W(:,1,3),INH=PP%BETA(:,CHANNEL+1))
            ! second linearization energy for this L
            ! homogeneous solution
            E=PP%E(CHANNEL+1)+PP%AVERAGEPOT(1)
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL+1),V00,PP%R,W(:,2,1),WN=1._q)
            ! inhomogenous solution using the first projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL+1),V00,PP%R,W(:,2,2),INH=PP%BETA(:,CHANNEL))
            ! inhomogenous solution using the second projector for this L
            CALL PAR_WAVE_FKT(E,PP%LPS(CHANNEL+1),V00,PP%R,W(:,2,3),INH=PP%BETA(:,CHANNEL+1))
            ! calculate the overlap between the projectors and W(1:6) to be able
            OVL=0
            DO I=1,2
               ! projectors should be zero at matching radius
               IF (LUSERWIGS().AND.ABS(PP%BETA(NRWIGS,CHANNEL-1+I))>TINY) THEN
                  WRITE(*,'(A)') 'PS_PAR_WAVE: WARNING'
                  WRITE(*,'(A,I2,A)') &
                 & '  projector for channel',CHANNEL-1+I,' non-neglegible at matching radius:'
                  WRITE(*,'(A,F8.5,A,F10.6)') &
                 & '  beta(',PP%R%R(NRWIGS),') =',PP%BETA(NRWIGS,CHANNEL-1+I)
               ENDIF
            DO J=1,2
            DO K=1,3
               DO N=1,PP%R%NMAX
                  WORK(N)=PP%BETA(N,CHANNEL-1+I)*W(N,J,K)
               ENDDO
               CALL SIMPI(PP%R,WORK,OVL(I,J,K))
            ENDDO
            ENDDO
            ENDDO

            AA=0._q; BB=0._q
            
            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,1,3),A(3,1),A(3,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL),BC,DBC)
               CALL GET_BC(PP%R,W(:,1,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,1,2),A(2,1),A(2,2))
               CALL GET_BC(PP%R,W(:,1,3),A(3,1),A(3,2))
            ENDIF
            AA(3,1)=A(1,2)-A(1,1)*DBC/BC
            AA(3,2)=A(2,2)-A(2,1)*DBC/BC
            AA(3,3)=A(3,2)-A(3,1)*DBC/BC

            IF (LUSERWIGS()) THEN
               ! match logarithmic derivative to AE partial wave at PP%RWIGS
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL+1),BC,DBC,PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,1),A(1,1),A(1,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,2),A(2,1),A(2,2),PP%RWIGS)
               CALL GET_BC(PP%R,W(:,2,3),A(3,1),A(3,2),PP%RWIGS)
            ELSE
               ! match logarithmic derivative to AE partial wave at R%NMAX-1
               CALL GET_BC(PP%R,PP%WAE(:,CHANNEL+1),BC,DBC)
               CALL GET_BC(PP%R,W(:,2,1),A(1,1),A(1,2))
               CALL GET_BC(PP%R,W(:,2,2),A(2,1),A(2,2))
               CALL GET_BC(PP%R,W(:,2,3),A(3,1),A(3,2))            
            ENDIF
            AA(6,4)=A(1,2)-A(1,1)*DBC/BC
            AA(6,5)=A(2,2)-A(2,1)*DBC/BC
            AA(6,6)=A(3,2)-A(3,1)*DBC/BC

            AA(1,1)=OVL(1,1,1); AA(1,2)=OVL(1,1,2); AA(1,3)=OVL(1,1,3)
            AA(2,1)=OVL(2,1,1); AA(2,2)=OVL(2,1,2); AA(2,3)=OVL(2,1,3)
            
            AA(4,4)=OVL(1,2,1); AA(4,5)=OVL(1,2,2); AA(4,6)=OVL(1,2,3)
            AA(5,4)=OVL(2,2,1); AA(5,5)=OVL(2,2,2); AA(5,6)=OVL(2,2,3)
            
            BB(1)=1._q; BB(5)=1._q

            CALL DGECO(AA,6,6,IPV,RCOND,WRK)
            CALL DGESL(AA,6,6,IPV,BB,0)

            C(1,1)=BB(1); C(1,2)=BB(2); C(1,3)=BB(3)
            CNEW(CHANNEL,:)=C(1,:)
            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL)=C(1,1)*W(I,1,1)+C(1,2)*W(I,1,2)+C(1,3)*W(I,1,3)
            ENDDO
            SC(1)=PP%WPS(PP%R%NMAX,CHANNEL)/PP%WAE(PP%R%NMAX,CHANNEL)
            IF (LUSERWIGS()) THEN
               SC(1)=PP%WPS(NRWIGS,CHANNEL)/PP%WAE(NRWIGS,CHANNEL)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL)=SC(1)*PP%WAE(I,CHANNEL)
               ENDDO
            ENDIF
            
            C(2,1)=BB(4); C(2,2)=BB(5); C(2,3)=BB(6)
            CNEW(CHANNEL+1,:)=C(2,:)
            DO I=1,PP%R%NMAX
               PP%WPS(I,CHANNEL+1)=C(2,1)*W(I,2,1)+C(2,2)*W(I,2,2)+C(2,3)*W(I,2,3)
            ENDDO
            SC(2)=PP%WPS(PP%R%NMAX,CHANNEL+1)/PP%WAE(PP%R%NMAX,CHANNEL+1)
            IF (LUSERWIGS()) THEN
               SC(2)=PP%WPS(NRWIGS,CHANNEL+1)/PP%WAE(NRWIGS,CHANNEL+1)
               DO I=NRWIGS,PP%R%NMAX
                  PP%WPS(I,CHANNEL+1)=SC(2)*PP%WAE(I,CHANNEL+1)
               ENDDO
            ENDIF
#ifdef testout 
!           WRITE(99+CHANNEL,'(2F20.8)') (PP%R%R(K),PP%WPS(K,CHANNEL), K=1,PP%R%NMAX)
!           WRITE(99+CHANNEL+1,'(2F20.8)') (PP%R%R(K),PP%WPS(K,CHANNEL+1), K=1,PP%R%NMAX)
#endif
            DO I=1,PP%R%NMAX
               PP%WKINPS(I,CHANNEL  )=(V00(PP%R%NMAX)+PP%E(CHANNEL  )-V00(I))*PP%WPS(I,CHANNEL  )
               PP%WKINPS(I,CHANNEL+1)=(V00(PP%R%NMAX)+PP%E(CHANNEL+1)-V00(I))*PP%WPS(I,CHANNEL+1)
            ENDDO

            DO I=1,2
               DO N=1,PP%R%NMAX
                  PP%WAE(N,CHANNEL-1+I)=PP%WAE(N,CHANNEL-1+I)*SC(I)
                  PP%WKINAE(N,CHANNEL-1+I)=PP%WKINAE(N,CHANNEL-1+I)*SC(I)
               ENDDO
            ENDDO

            DO I=1,2
            DO J=1,2
               DO K=1,PP%R%NMAX
                  WORK(K)=PP%WAE(K,CHANNEL-1+I)*PP%WAE(K,CHANNEL-1+J)
               ENDDO
               CALL SIMPI(PP%R,WORK,QAE(I,J))
            ENDDO
            ENDDO

            QPS=0
            DO I=1,2
            DO J=1,2
               DO K=1,PP%R%NMAX
                  WORK(K)=PP%WPS(K,CHANNEL-1+I)*PP%WPS(K,CHANNEL-1+J)
               ENDDO
               CALL SIMPI(PP%R,WORK,QPS(I,J))
            ENDDO
            ENDDO

            QIJ=0
            DO I=1,2
            DO J=1,2
               QIJ(I,J)=QAE(I,J)-QPS(I,J)
            ENDDO
            ENDDO

            ! setup the matrix BIJ = < \tilde{\psi}_i | \chi_j |>
            BIJ=0
            DO I=1,2
            DO J=1,2
               BIJ(I,J)=-C(J,I+1)
            ENDDO
            ENDDO
            
            PP%E(:)=PP%E(:)+PP%AVERAGEPOT(1)

            DIJ=0
            DO I=1,2
            DO J=1,2
               DIJ(I,J)=BIJ(I,J)*(2*RYTOEV)+PP%E(CHANNEL-1+J)*QIJ(I,J)
            ENDDO
            ENDDO

            PP%E(:)=PP%E(:)-PP%AVERAGEPOT(1)

            DIJ(1,2)=(DIJ(1,2)+DIJ(2,1))/2
            DIJ(2,1)=DIJ(1,2)

            DO I=1,2
            DO J=1,2
               PP%QIJ(CHANNEL-1+I,CHANNEL-1+J)=QIJ(I,J)
               PP%DIJ(CHANNEL-1+I,CHANNEL-1+J)=DIJ(I,J)
            ENDDO
            ENDDO
         ENDIF   
      ENDDO chann

      PP%QION=PP%QIJ
      PP%C=CNEW
#ifdef testout
      WRITE(*,*) 'AIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F15.7)') (CNEW(I,J), J=1,3)
      ENDDO
      WRITE(*,*) 'DIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F15.7)') (PP%DIJ(I,J), J=1,PP%LMAX)
      ENDDO
      WRITE(*,*) 'QIJ'
      DO I=1,PP%LMAX
         WRITE(*,'(10F15.7)') (PP%QIJ(I,J), J=1,PP%LMAX)
      ENDDO      
#endif

! update PP%POTPS (spherical part of valence only PS potential)
      PP%POTPS(:)=-(V00(:)-PP%POTPSC(:))
#ifdef testout
      AVERAGEPOT=0
      DO I=1,PP%R%NMAX
         AVERAGEPOT=AVERAGEPOT+PP%POTPSC(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO
      WRITE(*,*) 'PS_PAR_WAVE: AVERAGEPOT(PSC)=',AVERAGEPOT
      WRITE(*,'(A,F14.7)') 'PS_PAR_WAVE: charge inferred from potential at RMAX'
      WRITE(*,'(A,F14.7)') 'PS charge core   :',PP%POTPSC(PP%R%NMAX)*PP%R%R(PP%R%NMAX)/RYTOEV/AUTOA/2

      WRITE(*,*)
      WRITE(*,*) 'PS_PAR_WAVE: duality < p_i | psi_j >'
      DO K=1,PP%LMAX
      DO J=1,PP%LMAX
         IF (PP%LPS(K)==PP%LPS(J)) THEN     
            DO I=1,PP%R%NMAX
               WORK(I)=PP%BETA(I,K)*PP%WPS(I,J)
            ENDDO
            CALL SIMPI(PP%R,WORK,OVL(1,1,1))
         ELSE
            OVL(1,1,1)=0._q
         ENDIF
         WRITE(*,'(F14.7)',ADVANCE='NO') OVL(1,1,1)
      ENDDO
         WRITE(*,*)
      ENDDO
#endif
! output for analysis
      IF (LREPORT()) THEN
! Write logarithmic derivatives to file
      IUNIT=IUBASE+NODE_ME
      WRITE(FILEOUT,'(A7,I4.4,A1,I4.4)') "LDERPS.",ITYP,".",NCALLED
      OPEN(UNIT=IUNIT,FILE=DIR_APP(1:DIR_LEN)//FILEOUT,STATUS='NEW',ERR=120)      

      EMIN=MINVAL(PP%E(1:PP%LMAX))+PP%AVERAGEPOT(1)-DE
      EMAX=MAXVAL(PP%E(1:PP%LMAX))+PP%AVERAGEPOT(1)+DE
      
      DO I=1,1000
         E=REAL(EMIN+(I-1)*(EMAX-EMIN)/1000)
         WRITE(UNIT=IUNIT,FMT='(F20.8)',ADVANCE='NO') E
         LPREV=-1
         DO CHANNEL=1,PP%LMAX
            IF (PP%LPS(CHANNEL)/=LPREV) THEN
               CALL PSSOLVE(PP,V00(1:PP%R%NMAX),PP%DIJ,PP%QIJ,E,PP%BETA,PP%LPS(CHANNEL), &
              & CDUM,WORK)
               CALL GET_BC(PP%R,WORK,BC,DBC)
               WRITE(IUNIT,FMT='(F20.8)',ADVANCE='NO') DBC/BC
            ENDIF
            LPREV=PP%LPS(CHANNEL)
         ENDDO
         WRITE(IUNIT,FMT='(A1)')
      ENDDO
      
      CLOSE(IUNIT)

 120  CONTINUE
      ENDIF
            
      DEALLOCATE(RHO,V)

      RETURN
      END SUBROUTINE PS_PAR_WAVE
#endif

!*********************** SUBROUTINE PS_PAR_WAVE_POTCAR ******************      
!
! This subroutine determines the expansion coefficients PP%C
! consistent with the information stored on the POTCAR file.
! It has to be called during the initialization of the relaxed
! core method.
!
!************************************************************************      
      SUBROUTINE PS_PAR_WAVE_POTCAR(PP)
          
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE ini
      USE us
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
            
      INTEGER ISPIN,I,J,K,CHANNEL,LPREV
      INTEGER LYMAX,LMMAX,LM,LMP
      REAL(q) E,VCORR
      REAL(q) EKIN
      REAL(q) SCALE
      REAL(q) F,FDER
      REAL(q) BC,DBC
      REAL(q) C(3)
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: DIJ,QIJ
      REAL(q), DIMENSION(PP%R%NMAX) :: V00,WORK
      REAL(q), ALLOCATABLE :: V(:,:,:)
      REAL(q), DIMENSION(PP%R%NMAX,PP%LMAX) :: BETA,W,T,W_DUAL,T_DUAL
      REAL(q) DLM(PP%LMDIM*PP%LMDIM)
      OVERLAP DTMP(PP%LMDIM,PP%LMDIM,1),DHXC(PP%LMDIM,PP%LMDIM)
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: OVL,QQ
      INTEGER IPIV(PP%LMAX),M,N
      REAL(q) RCOND,DET(2),WRK(PP%LMAX)
      
      REAL(q) EMIN,EMAX
      REAL(q), PARAMETER :: DE=15._q
      INTEGER, EXTERNAL :: MAXL_AUG

      BETA=0
      DO CHANNEL=1,PP%LMAX
         IF (PP%LPS(CHANNEL)>PP%LMAX_CALC) CYCLE
         ! transfer projectors onto radial grid
         ! n.b. we store r*P(r)
         DO I=1,PP%R%NMAX
            IF (PP%R%R(I)>PP%PSPRNL(NPSRNL,1,1)) EXIT
            CALL SPLVAL(PP%R%R(I),F,FDER,PP%PSPRNL(:,:,CHANNEL),NPSRNL,NPSRNL)                  
            BETA(I,CHANNEL)=F*PP%R%R(I)
         ENDDO
#ifdef testout
!        WRITE(*,*) 'projector ',channel
!        WRITE(*,'(2F20.8)') (PP%R%R(I),BETA(I,CHANNEL), I=1,PP%R%NMAX)
#endif
      ENDDO
! setup the potential from the information stored on the POTCAR
      LYMAX=MAXL_AUG(1,PP); LMMAX=(LYMAX+1)**2
      ALLOCATE(V(PP%R%NMAX,LMMAX,1))
      SCALE=1/(2*SQRT(PI))      
      V=0; V00=0
      V00(1:PP%R%NMAX)=-PP%POTPS(1:PP%R%NMAX)+PP%POTPSC(1:PP%R%NMAX)

      DO I=1,PP%R%NMAX
         V00(I)=V00(I)+PP%VPSRMAX-V00(PP%R%NMAX)
      ENDDO
      
      V(:,1,1)=V00(:)/SCALE
#ifdef testout
! the spherical component of the PS potential as reconstructed from the
! information stored on the POTCAR file is written to unit 102
!     WRITE(102,'(2F20.8)') (PP%R%R(I),V00(I), I=1,PP%R%NMAX)
!     WRITE(103,'(2F20.8)') (PP%R%R(I),PP%POTPSC(I), I=1,PP%R%NMAX)
#endif
! contribution of the valence electrons to the strength parameters
      CALL RAD_POT_WEIGHT(PP%R,1,LYMAX,V)
      DLM=0; DTMP=0; DHXC=0
      CALL RAD_PROJ(V(:,:,1),PP%R,1._q,DLM,PP%LMAX,PP%LPS,PP%WPS)
      CALL TRANS_DLM_RC(DTMP(:,:,1),DLM,PP)

      DLM=0; DTMP=0; DHXC=0
      CALL RAD_AUG_PROJ(V(:,:,1),PP%R,DLM,PP%LMAX,PP%LPS,LYMAX,PP%AUG,PP%QPAW)
      CALL TRANS_DLM_RC(DTMP(:,:,1),DLM,PP)

      DHXC=-DTMP(:,:,1)
! setup D_{ij} and Q_{ij}
      LM=1
      DO I=1,PP%LMAX
      LMP=1
      DO J=1,PP%LMAX
         DIJ(I,J)=PP%DION(I,J)+DHXC(LM,LMP)
         QIJ(I,J)=PP%QION(I,J)
         LMP=LMP+2*PP%LPS(J)+1
      ENDDO
      LM=LM+2*PP%LPS(I)+1
      ENDDO      

! get the partial waves and the expansion coefficients that
! define them
      DO CHANNEL=1,PP%LMAX
         E=PP%E(CHANNEL)+V00(PP%R%NMAX)-PP%VPSRMAX+PP%AVERAGEPOT(1)
         CALL PSSOLVE(PP,V00(1:PP%R%NMAX),DIJ,QIJ,E,BETA,PP%LPS(CHANNEL),C, &
        &   W(1:PP%R%NMAX,CHANNEL),T(1:PP%R%NMAX,CHANNEL))
         PP%C(CHANNEL,:)=C(:)         
      ENDDO
#ifdef testout
! dualize the partial waves and write them to unit 800+L
      DO I=1,PP%LMAX
      DO J=1,PP%LMAX
         DO K=1,PP%R%NMAX
            WORK(K)=BETA(K,I)*W(K,J)
         ENDDO
         CALL SIMPI(PP%R,WORK,OVL(I,J))
         IF (PP%LPS(I)/=PP%LPS(J)) OVL(I,J)=0
      ENDDO
      ENDDO

      CALL DGECO(OVL,PP%LMAX,PP%LMAX,IPIV,RCOND,WRK)
      CALL DGEDI(OVL,PP%LMAX,PP%LMAX,IPIV,DET,WRK,'01')

      W_DUAL=0; T_DUAL=0
      DO I=1,PP%LMAX
         DO J=1,PP%LMAX
            DO K=1,PP%R%NMAX
               W_DUAL(K,I)=W_DUAL(K,I)+OVL(J,I)*W(K,J)
               T_DUAL(K,I)=T_DUAL(K,I)+OVL(J,I)*T(K,J)
            ENDDO
         ENDDO
#ifdef testout
!        WRITE(799+I,'(2F20.8)') (PP%R%R(K),W_DUAL(K,I), K=1,PP%R%NMAX)
!        WRITE(849+I,'(2F20.8)') (PP%R%R(K),PP%WPS(K,I), K=1,PP%R%NMAX)
!        WRITE(859+I,'(2F20.8)') (PP%R%R(K),PP%WAE(K,I), K=1,PP%R%NMAX)
#endif
      ENDDO
! calculate logarithmic derivatives and write them to unit 900+L
      EMIN=MINVAL(PP%E(1:PP%LMAX))+V00(PP%R%NMAX)-PP%VPSRMAX+PP%AVERAGEPOT(1)-DE
      EMAX=MAXVAL(PP%E(1:PP%LMAX))+V00(PP%R%NMAX)-PP%VPSRMAX+PP%AVERAGEPOT(1)+DE
      LPREV=-1
      DO CHANNEL=1,PP%LMAX
         IF (PP%LPS(CHANNEL)/=LPREV) THEN
         DO I=1,1000   
            E=REAL(EMIN+(I-1)*(EMAX-EMIN)/1000)
            CALL PSSOLVE(PP,V00(1:PP%R%NMAX),DIJ,QIJ,E,BETA,PP%LPS(CHANNEL),C, &
           &   WORK)
            CALL GET_BC(PP%R,WORK,BC,DBC)
            WRITE(899+PP%LPS(CHANNEL)+1,'(2F20.8)') E+PP%POTAE(PP%R%NMAX)-V00(PP%R%NMAX),DBC/BC
         ENDDO
         ENDIF
         LPREV=PP%LPS(CHANNEL)
      ENDDO
#endif      

      DEALLOCATE(V)

      RETURN
      END SUBROUTINE PS_PAR_WAVE_POTCAR


!*********************** SUBROUTINE PSSOLVE *****************************
!
! This subroutine solves 
!
!************************************************************************
      SUBROUTINE PSSOLVE(PP,V,DIJ,QIJ,E,BETA,LQN,C,W,T)
      
      USE prec
      USE pseudo
      USE radial
      USE constant
            
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER, INTENT(IN) :: LQN
      REAL(q), INTENT(IN) :: E
      REAL(q), DIMENSION(PP%R%NMAX), INTENT(IN) :: V
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX), INTENT(IN) :: DIJ,QIJ
      REAL(q), DIMENSION(PP%R%NMAX,PP%LMAX), INTENT(IN) :: BETA
      REAL(q), DIMENSION(3), INTENT(OUT) :: C
      
      REAL(q), DIMENSION(PP%R%NMAX), INTENT(OUT), OPTIONAL :: W
      REAL(q), DIMENSION(PP%R%NMAX), INTENT(OUT), OPTIONAL :: T
      
      INTEGER I,J,K,L
      INTEGER IPIV(PP%LMAX)
      REAL(q), DIMENSION(PP%R%NMAX,PP%LMAX) :: HOM,IHOM,THOM,TIHOM
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: HPROJ,IPROJ,M,A
      REAL(q) WORK(PP%R%NMAX),WRK(PP%LMAX),RCOND,DET(2)
      
! solve the homogenous schroedinger-equation  (T+V-E) |\phi_loc>  =   0
! and the inhomogenous differential equations (T+V-E) |\phi_i>    = |\beta_i>
      DO I=1,PP%LMAX
         ! homogeneous solution
         CALL PAR_WAVE_FKT(E,PP%LPS(I),V,PP%R,HOM(:,I),WN=1._q)
         DO J=1,PP%R%NMAX
            THOM(J,I)=(E-V(J))*HOM(J,I)
         ENDDO
         ! inhomogeneous solution
         CALL PAR_WAVE_FKT(E,PP%LPS(I),V,PP%R,IHOM(:,I),INH=BETA(:,I))
         DO J=1,PP%R%NMAX
            TIHOM(J,I)=(E-V(J))*IHOM(J,I)+(2*RYTOEV)*BETA(J,I)
         ENDDO
      ENDDO
      ! calculate projections 
      HPROJ=0; IPROJ=0
      DO I=1,PP%LMAX
      DO J=1,PP%LMAX
         DO K=1,PP%R%NMAX
            WORK(K)=BETA(K,I)*HOM(K,J)
         ENDDO
         CALL SIMPI(PP%R,WORK,HPROJ(I,J))
         IF (PP%LPS(I)/=PP%LPS(J)) HPROJ(I,J)=0
         DO K=1,PP%R%NMAX
            WORK(K)=BETA(K,I)*IHOM(K,J)
         ENDDO
         CALL SIMPI(PP%R,WORK,IPROJ(I,J))
         IF (PP%LPS(I)/=PP%LPS(J)) IPROJ(I,J)=0
      ENDDO
      ENDDO
! setup matrix M_{ik}=\delta_{ik}+\sum_j (D_{ij} - E Q_{ij}) < beta_j | \phi_k >
      M=0
      DO I=1,PP%LMAX
      DO K=1,PP%LMAX
         IF (I==K) M(I,K)=M(I,K)+1
         DO J=1,PP%LMAX
            M(I,K)=M(I,K)+(DIJ(I,J)-E*QIJ(I,J))/(2*RYTOEV)*IPROJ(J,K)
         ENDDO
      ENDDO
      ENDDO
! and invert M_{ik}
      CALL DGECO(M,PP%LMAX,PP%LMAX,IPIV,RCOND,WRK)
      CALL DGEDI(M,PP%LMAX,PP%LMAX,IPIV,DET,WRK,'01')
! calculate A_{lk}= -\sum_{ij} M^{-1}_{ki}(D_{ij} - E Q_{ij})< \beta_j | \phi_loc_l >
      A=0
      DO L=1,PP%LMAX
      DO K=1,PP%LMAX
         DO I=1,PP%LMAX
         DO J=1,PP%LMAX
            A(L,K)=A(L,K)-M(K,I)*(DIJ(I,J)-E*QIJ(I,J))/(2*RYTOEV)*HPROJ(J,L)
         ENDDO
         ENDDO
      ENDDO
      ENDDO
! store the coefficients that define the partial wave for L=LQN,
! and optionally the partial wave and its kinetic energy density     
      C=0 
      ! homogeneous solution
      C(1)=1._q      
      DO I=1,PP%LMAX
         IF (PP%LPS(I)/=LQN) CYCLE
         IF (PRESENT(W)) W(:)=HOM(:,I)
         IF (PRESENT(T)) T(:)=THOM(:,I)         
      ENDDO
      ! inhomogeneous contributions
      K=2
      DO I=1,PP%LMAX
         IF (PP%LPS(I)/=LQN) CYCLE
         IF (PRESENT(W)) W(:)=W(:)+A(I,I)*IHOM(:,I)
         IF (PRESENT(T)) T(:)=T(:)+A(I,I)*TIHOM(:,I)
         C(K)=A(I,I); K=K+1
      ENDDO

      RETURN
      END SUBROUTINE PSSOLVE

      
!*********************** SUBROUTINE PAR_WAVE_FKT ************************      
!
! Solve the radial scalar relativistic schroedinger (optionally +SOC)
! equation with a local potential V, for a partial wave with angular
! moment L at an energy E, by outward integration.
!
!************************************************************************      
      SUBROUTINE PAR_WAVE_FKT(E,L,V,R,W,WN,INH)
      
      USE prec
      USE cl
      USE radial
      
      IMPLICIT NONE
      
      TYPE(rgrid) R
      
      INTEGER, INTENT(IN) :: L
      REAL(q), INTENT(IN) :: E
      REAL(q), DIMENSION(:), INTENT(IN) :: V
      REAL(q), DIMENSION(:), INTENT(OUT) :: W
      REAL(q), OPTIONAL, INTENT(IN) :: WN
      REAL(q), OPTIONAL, DIMENSION(:), INTENT(IN) :: INH
            
      INTEGER I,K,KI,NODES,NERR
      REAL(q) RA,WN_,SCALE
      REAL(q), DIMENSION(R%NMAX) :: A,B
      REAL(q), PARAMETER :: C = 137.037    ! speed of light
      LOGICAL DBL
      
! perform some checks on the input
      IF (SIZE(V)<R%NMAX) THEN
         WRITE(*,*) 'PAR_WAVE_FKT: Grid inconsistency (1):',R%NMAX,SIZE(V)
         STOP
      ENDIF
      IF (PRESENT(INH)) THEN
         IF (SIZE(INH)<R%NMAX) THEN
            WRITE(*,*) 'PAR_WAVE_FKT: Grid inconsistency (2):',R%NMAX,SIZE(INH)
            STOP
         ENDIF
      ENDIF     
      
      IF (PRESENT(INH)) THEN
         CALL OUTINT(E,L,R,V(1:R%NMAX),KI,NODES,A,B,NERR,RC=R%REND,IH=INH)
      ELSE
         CALL OUTINT(E,L,R,V(1:R%NMAX),KI,NODES,A,B,NERR,RC=R%REND)
      ENDIF
! calculate norm of W and normalize to WN      
      RA=R%REND/(R%D**(R%NMAX-KI))
      WN_=RA*(A(KI)**2+(B(KI)/C)**2)/2
      RA=RA+RA
      DBL=.FALSE.
!     DO K=R%NMAX-1,2,-1
      DO K=KI-1,2,-1
         RA=RA/R%D
         DBL=.NOT.DBL
         WN_=WN_+RA*(A(K)*A(K)+(B(K)/C)**2)
         IF (DBL) WN_=WN_+RA*(A(K)*A(K)+(B(K)/C)**2)
      ENDDO
      WN_=R%H*(WN_+R%RSTART*(A(1)**2+(B(1)/C)**2)/2)/3._q
      W=0
      SCALE=1._q
      IF (PRESENT(WN)) SCALE=SQRT(WN/WN_)
      W(1:KI)=A(1:KI)*SCALE
      RETURN
      END SUBROUTINE PAR_WAVE_FKT


!*********************** SUBROUTINE PAR_WAVE_FKT2 ***********************      
!
! Solve the radial scalar relativistic schroedinger (optionally +SOC)
! equation with a local potential V, for a partial wave with angular
! moment L at an energy E, by outward integration.
!
!************************************************************************      
      SUBROUTINE PAR_WAVE_FKT2(E,L,V,R,W,WN,INH)
      
      USE prec
      USE cl
      USE radial
      
      IMPLICIT NONE
      
      TYPE(rgrid) R
      
      INTEGER, INTENT(IN) :: L
      REAL(q), INTENT(IN) :: E
      REAL(q), DIMENSION(:), INTENT(IN) :: V
      REAL(q), DIMENSION(:), INTENT(OUT) :: W
      REAL(q), OPTIONAL, INTENT(IN) :: WN
      REAL(q), OPTIONAL, DIMENSION(:), INTENT(IN) :: INH
            
      INTEGER I,K,KI,KJ,NODES,NERR
      REAL(q) RA,WN_,MA,MB
      REAL(q), DIMENSION(R%NMAX) :: A,B
      REAL(q), PARAMETER :: C = 137.03     ! speed of light
      LOGICAL DBL
      
! perform some checks on the input
      IF (SIZE(V)<R%NMAX) THEN
         WRITE(*,*) 'PAR_WAVE_FKT: Grid inconsistency (1):',R%NMAX,SIZE(V)
         STOP
      ENDIF
      IF (PRESENT(INH)) THEN
         IF (SIZE(INH)<R%NMAX) THEN
            WRITE(*,*) 'PAR_WAVE_FKT: Grid inconsistency (2):',R%NMAX,SIZE(INH)
            STOP
         ENDIF
      ENDIF     
      
      IF (PRESENT(INH)) THEN
         CALL OUTINT(E,L,R,V(1:R%NMAX),KI,NODES,A,B,NERR,IH=INH)
      ELSE
         CALL OUTINT(E,L,R,V(1:R%NMAX),KI,NODES,A,B,NERR)
      ENDIF
      
      MA=A(KI)
      MB=B(KI)

      CALL INWINT(E,L,R,V(1:R%NMAX),KI,KJ,NODES,A,B)

      MA=MA/A(KI)
      MB=MB/B(KI)

      DO K=KI,KJ
         A(K)=A(K)*MA
         B(K)=B(K)*MA
      ENDDO

! calculate norm of W and normalize to WN      
      RA=R%REND
      WN_=RA*(A(R%NMAX)**2+(B(R%NMAX)/C)**2)/2
      RA=RA+RA
      DBL=.FALSE.
      DO K=R%NMAX-1,2,-1
         RA=RA/R%D
         DBL=.NOT.DBL
         WN_=WN_+RA*(A(K)*A(K)+(B(K)/C)**2)
         IF (DBL) WN_=WN_+RA*(A(K)*A(K)+(B(K)/C)**2)
      ENDDO
      WN_=R%H*(WN_+R%RSTART*(A(1)**2+(B(1)/C)**2)/2)/3._q
      W=0
      W(1:R%NMAX)=A(1:R%NMAX)
      IF (PRESENT(WN)) W=W*SQRT(WN/WN_)
      RETURN
      END SUBROUTINE PAR_WAVE_FKT2


      SUBROUTINE GET_BC(R,W,WR,DWR,RMAX)
      USE prec
      USE radial      
      IMPLICIT NONE      
      TYPE(rgrid) R      
      REAL(q) W(:)
      REAL(q) WR,DWR
      REAL(q) H
      INTEGER I
      REAL(q), OPTIONAL :: RMAX
      
      IF(PRESENT(RMAX)) THEN
         IF (RMAX<=R%R(R%NMAX-1)) THEN
            DO I=1,R%NMAX
               IF (R%R(I)>=RMAX) EXIT
            ENDDO
         ELSE
            I=R%NMAX
         ENDIF
      ELSE
         I=R%NMAX
      ENDIF
      
      H=R%H
      I=MAX(I,2)
      DWR=(W(I)-W(I-2))/2/H
      WR=W(I)
      
      RETURN
      END SUBROUTINE GET_BC

      
!*********************** SUBROUTINE POT *********************************      
! Setup the onsite potential V (on a radial grid)
!
! Input
! RHO(:,:,:) charge density, in (charge,magnetization) format 
! Z          nuclear charge
! R          grid
! RHOC(:)    partial core density (optional)
! POTC(:)    frozen core potential (optional)
!
! Output
! V(:,:,:)   potential, in (charge,magnetization) format
!************************************************************************
      SUBROUTINE POT(RHO,Z,R,V,RHOC,VC,EXC,ADD_GGA)
      
      USE prec
      USE constant
      USE radial
      USE setexm
      
      IMPLICIT NONE
      
      TYPE(rgrid) R
      
      INTEGER, INTENT(IN) :: Z
      REAL(q), DIMENSION(:,:,:), INTENT(IN) :: RHO
      REAL(q), DIMENSION(:,:,:), INTENT(OUT) :: V
      REAL(q), DIMENSION(:), OPTIONAL, INTENT(IN) :: RHOC
      REAL(q), DIMENSION(:), OPTIONAL, INTENT(IN) :: VC
      REAL(q), OPTIONAL, INTENT(OUT):: EXC
      LOGICAL, OPTIONAL, INTENT(IN) :: ADD_GGA
! local variables
      INTEGER ISPIN,LMAX,NMAX
      INTEGER K,L,M,LM,ISP
      REAL(q) SCALE,SUM,QINT,QINF
      REAL(q), DIMENSION(:,:,:), ALLOCATABLE :: WORK1
      REAL(q), DIMENSION(:,:),ALLOCATABLE :: WORK2

      LOGICAL,PARAMETER :: TREL=.TRUE. ! use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TLDA=.TRUE. ! calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      ! in this case non spherical contributions are missing
      REAL(q) :: EXCG,DHARTREE,DEXC,DVXC,DEXC_GGA,DVXC_GGA,DOUBLEC
      LOGICAL :: ADDGGA

      ADDGGA=ISGGA()
      IF (PRESENT(ADD_GGA)) ADDGGA=ADD_GGA
      
! get dimensions and perform some checks
      LMAX=INT(SQRT(REAL(SIZE(RHO,2)))-1)      
      IF (((LMAX+1)*(LMAX+1))/=SIZE(RHO,2)) THEN
         WRITE(*,*) 'POT: LMAX and the 2nd dimension of RHO do not match:',((LMAX+1)*(LMAX+1)),SIZE(RHO,2)
         STOP
      ENDIF
      IF (SIZE(V,2)<SIZE(RHO,2)) THEN
         WRITE(*,*) 'POT: 2nd dimension of V too small',SIZE(V,2),SIZE(RHO,2)
      ENDIF
      ISPIN=SIZE(RHO,3)
      IF (ISPIN/=1.AND.ISPIN/=2.AND.ISPIN/=4) THEN
         WRITE(*,*) 'POT: ISPIN /= 1,2, or 4:',ISPIN
         STOP
      ENDIF
      NMAX=SIZE(RHO,1)
      IF (NMAX<R%NMAX) THEN
         WRITE(*,*) 'POT: Grid inconsistency (1):',R%NMAX,NMAX
         STOP
      ENDIF
      IF (PRESENT(RHOC)) THEN
         IF (SIZE(RHOC)<R%NMAX) THEN
            WRITE(*,*) 'POT: Grid inconsistency (2):',R%NMAX,NMAX,SIZE(RHOC)
            STOP
         ENDIF
      ENDIF
      IF (PRESENT(VC)) THEN
         IF (SIZE(VC)<R%NMAX) THEN
            WRITE(*,*) 'POT: Grid inconsistency (3):',R%NMAX,NMAX,SIZE(VC)
            STOP
         ENDIF
      ENDIF

      SCALE=2*SQRT(PI)

      V=0

      IF (ISPIN==1.OR.ISPIN==2) THEN
         ALLOCATE(WORK1(NMAX,(LMAX+1)*(LMAX+1),ISPIN),WORK2(NMAX,ISPIN))
         WORK1=RHO
      ELSEIF (ISPIN==4) THEN
         ALLOCATE(WORK1(NMAX,(LMAX+1)*(LMAX+1),2),WORK2(NMAX,2))
         CALL RAD_MAG_DENSITY(RHO,WORK1,LMAX,R)
      ENDIF

!========================================================================
! Hartree potential
!========================================================================
      DHARTREE=0

      DO L=0,LMAX
      DO M=0,2*L
         LM=L*L+M+1
         CALL RAD_POT_HAR(L,R,V(:,LM,1),WORK1(:,LM,1),SUM)
         DHARTREE=DHARTREE+SUM
      ENDDO
      ENDDO

#ifdef testout
      ! integrate charge
      CALL SIMPI(R,WORK1(:,1,1),QINT)
      ! infer integrated charge from Hartree potential
      QINF=V(R%NMAX,1,1)*R%R(R%NMAX)/FELECT/2/SQRT(PI)
      ! write
      WRITE(*,'(A,F14.7,A,F14.7)') 'POT: Q_int=',QINT*SCALE,' Q_inf=',QINF
#endif      

      IF (PRESENT(VC)) THEN
         DO K=1,R%NMAX
            V(K,1,1)=V(K,1,1)+VC(K)*SCALE
         ENDDO
      ENDIF
      IF (ISPIN==2.OR.ISPIN==4) V(:,:,2)=V(:,:,1)

!========================================================================
! add nuclear potential
!========================================================================

      DO K=1,R%NMAX
         V(K,1,1)=V(K,1,1)-FELECT*SCALE*Z/R%R(K)
      ENDDO
      IF (ISPIN==2.OR.ISPIN==4) V(:,1,2)=V(:,1,1)

!========================================================================
! LDA exchange correlation energy, potential
! and double counting corrections
!========================================================================
      DEXC=0
      DVXC=0
      
      DO ISP=1,MIN(ISPIN,2)
         DO K=1,R%NMAX
            WORK2(K,ISP)=WORK1(K,1,ISP)/(SCALE*R%R(K)*R%R(K))
         ENDDO
      ENDDO
      ! add partial core charge if present
      IF (PRESENT(RHOC)) THEN
         DO K=1,R%NMAX
            WORK2(K,1)=WORK2(K,1)+RHOC(K)/(SCALE*R%R(K)*R%R(K))
         ENDDO
      ENDIF
      
 lda: IF (TLDA) THEN
         IF (ISPIN==1) THEN
            CALL RAD_LDA_XC(R,TREL,LMAX,WORK2(:,1),WORK1(:,:,1),V(:,:,1),DEXC,DVXC,.TRUE.)
         ELSE
            CALL RAD_LDA_XC_SPIN(R,TREL,LMAX,WORK2,WORK1,V,DEXC,DVXC,.TRUE.)
         ENDIF
      ENDIF lda
!========================================================================
! GGA if required
!========================================================================
      DEXC_GGA=0
      DVXC_GGA=0
      
! gga: IF (ISGGA()) THEN
  gga: IF (ADDGGA) THEN
         IF (ISPIN==1) THEN
            CALL RAD_GGA_XC(R,TLDA,WORK2(:,1),WORK1(:,1,1),V(:,1,1),DEXC_GGA,DVXC_GGA)
         ELSE
            IF (PRESENT(RHOC)) THEN
               DO K=1,R%NMAX
                  WORK2(K,1)=(WORK1(K,1,1)+RHOC(K)+WORK1(K,1,2))/(2*SCALE*R%R(K)*R%R(K))
                  WORK2(K,1)=MAX(WORK2(K,1), 1E-7_q)
                  WORK2(K,2)=(WORK1(K,1,1)+RHOC(K)-WORK1(K,1,2))/(2*SCALE*R%R(K)*R%R(K))
                  WORK2(K,2)=MAX(WORK2(K,2), 1E-7_q)
               ENDDO
            ELSE
               DO K=1,R%NMAX
                  WORK2(K,1)=(WORK1(K,1,1)+WORK1(K,1,2))/(2*SCALE*R%R(K)*R%R(K))
                  WORK2(K,1)=MAX(WORK2(K,1), 1E-7_q)
                  WORK2(K,2)=(WORK1(K,1,1)-WORK1(K,1,2))/(2*SCALE*R%R(K)*R%R(K))
                  WORK2(K,2)=MAX(WORK2(K,2), 1E-7_q)
               ENDDO
            ENDIF
            ! add partial core charge if present
            CALL RAD_GGA_XC_SPIN(R,TLDA,WORK2,WORK1(:,1,:),V(:,1,:),DEXC_GGA,DVXC_GGA)
         ENDIF
      ENDIF gga
!========================================================================
! done
!========================================================================
#ifdef debug
      WRITE(*,1)  N,-DHARTREE/2, -DVXC, DEXC, -DVXC_GGA, DEXC_GGA
1     FORMAT(' -Hartree, -vxc, exc, charge:',I4,6F14.7)
#endif
      ! Classical double counting correction:
      ! E_dc = -1/2 \int rho(r) V_H[rho(r)] dr + E_xc[rho+rhoc]
      !          -  \int rho(r) V_xc[rho(r)+rhoc(r)] dr
      DOUBLEC= -DHARTREE/2+DEXC-DVXC+DEXC_GGA-DVXC_GGA

      EXCG= DEXC+DEXC_GGA
      IF (PRESENT(EXC)) EXC=EXCG
      
      IF (ISPIN==2) THEN
         WORK1(1:R%NMAX,:,1)=(V(1:R%NMAX,1:SIZE(RHO,2),1)+V(1:R%NMAX,1:SIZE(RHO,2),2))/2
         WORK1(1:R%NMAX,:,2)=(V(1:R%NMAX,1:SIZE(RHO,2),1)-V(1:R%NMAX,1:SIZE(RHO,2),2))/2
         V=WORK1
      ENDIF
      IF (ISPIN==4) CALL RAD_MAG_DIRECTION(RHO,WORK1,V,LMAX,R)
      
      DEALLOCATE(WORK1,WORK2)
      RETURN
      END SUBROUTINE POT


!*********************** SUBROUTINE TRANS_DLM_RC *******************
!
!  transform D(llp,L,M) to the representation D(lm,l'm')
!  using Clebsch Gordan coefficients and add to another array
!
!  D(lm,l'm') =  sum C(LM,ll',mm') D(llp,LM)
!  where C(LM,ll',mm') = \int Y_LM Y_lm Y_l'm' d Omega
!
!  the storage layout of D(llp,LM) is somewhat akward see above
!
!*******************************************************************
      SUBROUTINE TRANS_DLM_RC( DLLMM, DLM, P)
      USE pseudo
      USE asa
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
      REAL(q) DLM(:)       ! local charge for each L,M
      ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX,INMIN,INMAX

      ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

      ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)
      ! Lmin and Lmax
      LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)

      ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
      JBASE=IBASE-LMIN*LMIN
#ifdef debug
      DO M=LMIN,LMAX
         WRITE(0,'(3I3,10F14.7)') CH1,CH2,M, &
               (DLM(JBASE+M*M+MP),MP=1,2*M+1)
      ENDDO
#endif

      INMIN=1000
      INMAX =-1000

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)
         DO  IC=ISTART,IEND-1
            INMIN=MIN(JS(IC)+JBASE,INMIN)
            INMAX=MAX(JS(IC)+JBASE,INMAX)
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                            DLM(JS(IC)+JBASE)*YLM3(IC)
         ENDDO
      ENDDO
      ENDDO
      ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

      RETURN
      END SUBROUTINE TRANS_DLM_RC


!*********************** SUBROUTINE SET_PAW_AUG_RC *****************
!
!  Sets up the compensation charges on the radial grid, 
!  and spline coefficients which are used to interpolate 
!  compensation charges in us.F
!
!  In addition it unscreens the strength parameters PP%DIJ
!  and stores them in PP%DION
!
!  DION_ij = D_ij - \int \tilde{V}(r) \hat{Q}_{ij}(r) dr
!
!*******************************************************************
      SUBROUTINE SET_PAW_AUG_RC(PP)

      USE prec
      USE constant
      USE pseudo
      USE radial
      USE us

      IMPLICIT NONE

      TYPE(potcar) :: PP

      ! local variables
      INTEGER CH1,CH2
      INTEGER LL,LLP,LMIN,LMAX,LMAIN
      INTEGER L,LP,N,I,J
      INTEGER IRMAX,NMAX_STORE
      INTEGER, PARAMETER :: NQ=2
      REAL(q)  QQ(NQ)
      REAL(q)  A(NQ),B,ALPHA
      REAL(q)  RES,SUM,QR,BJ,STEP,X
      REAL(q),PARAMETER ::  TH=1E-6_q
      REAL(q), DIMENSION(PP%R%NMAX) :: RHO

      INTEGER LYMAX,LMMAX
      INTEGER LM,LMP
      REAL(q) SCALE
      REAL(q), ALLOCATABLE :: V(:,:,:)
      REAL(q) DLM(PP%LMDIM*PP%LMDIM)
      OVERLAP DTMP(PP%LMDIM,PP%LMDIM,1),DUNSCR(PP%LMDIM,PP%LMDIM)
      INTEGER, EXTERNAL :: MAXL_AUG
      
      IF (.NOT. ASSOCIATED(PP%QPAW)) RETURN

      CALL RAD_ALIGN(PP%R,PP%R%RMAX)

      DO IRMAX=1,PP%R%NMAX-1
         IF (PP%RDEP>0 .AND. PP%R%R(IRMAX)-PP%RDEP > -5E-3) EXIT
      ENDDO
!     WRITE(*,*) 'test SET_PAW_AUG_RC',IRMAX,PP%R%NMAX

      ! set the simpson weights in concordance with IRMAX
      NMAX_STORE=PP%R%NMAX; PP%R%NMAX=IRMAX
      CALL SET_SIMP(PP%R) 

      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
      ! quantum numbers l and lp of these two channels
         LL =PP%LPS(CH1)
         LLP=PP%LPS(CH2)
         IF (LL==LLP) THEN
            RHO=0
            DO I=1,IRMAX
               RHO(I)=PP%WAE(I,CH1)*PP%WAE(I,CH2)
            ENDDO
            CALL SIMPI(PP%R, RHO , RES)
            PP%QTOT(CH1,CH2)=RES
         ENDIF
      ENDDO
      ENDDO

      ! reset the simpson weights
      PP%R%NMAX=NMAX_STORE
      CALL SET_SIMP(PP%R)

      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
      ! quantum numbers l and lp of these two channels
         LL =PP%LPS(CH1)
         LLP=PP%LPS(CH2)
         ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         DO LMAIN=LMIN,LMAX,2
            DO I=1,PP%R%NMAX
               RHO(I)=(PP%WAE(I,CH1)*PP%WAE(I,CH2)- &
              &   PP%WPS(I,CH1)*PP%WPS(I,CH2))*PP%R%R(I)**LMAIN
            ENDDO
            CALL SIMPI(PP%R, RHO , RES)
            PP%QPAW(CH1,CH2,LMAIN)=RES
         ENDDO
      ENDDO
      ENDDO
            
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         IF (PP%LPS(CH1)==PP%LPS(CH2)) THEN
            PP%QION(CH1,CH2)=PP%QPAW(CH1,CH2,0)
         ENDIF
      ENDDO
      ENDDO

      ! maximum l in augmentation charges
      LMAX=MAXVAL(PP%LPS(1:PP%LMAX))
      LMAX=LMAX*2               

      DO L=0,LMAX
      ! find q values
         CALL AUG_SETQ(L,PP%R,PP%R%RMAX,QQ,A,.FALSE.)
         ! setup augmentation charge on radial grid  rho(r) r^2
         DO N=1,PP%R%NMAX
            SUM=0
            IF (PP%R%R(N) <= PP%R%RMAX) THEN
               DO I=1,NQ
                  QR=QQ(I)*PP%R%R(N)
                  CALL SBESSEL( QR, BJ, L)
                  SUM=SUM+BJ*A(I)*PP%R%R(N)*PP%R%R(N)
               ENDDO
            ENDIF
            PP%AUG(N,L)=SUM      
         ENDDO

!!!! Spline coefficients must not be recalculated per se

         ! setup spline for augmentation charge
         ! the spline ends at PSDMAX*(NPSRNL-1)/NPSRNL see SETDEP
!        STEP= PP%R%RMAX/(NPSRNL-1)
!        PP%PSDMAX=NPSRNL*STEP
!
!        DO N=1,NPSRNL
!           X=STEP*(N-1)
!           SUM=0
!           DO I=1,NQ
!              QR=QQ(I)*X
!              CALL SBESSEL( QR, BJ, L)
!              SUM=SUM+BJ*A(I)
!           ENDDO
!           PP%QDEP(N,1,L) = X
!           PP%QDEP(N,2,L) = SUM
!        ENDDO
!        ! derivative at startpoint
!        X=STEP/100
!        SUM=0
!        DO I=1,NQ
!           QR=QQ(I)*X
!           CALL SBESSEL( QR, BJ, L)
!           SUM=SUM+BJ*A(I)
!        ENDDO
!        SUM=(SUM-PP%QDEP(1,2,L))/X
!        CALL SPLCOF(PP%QDEP(1,1,L),NPSRNL,NPSRNL,SUM)
      ENDDO

! Calculate  \int \tilde{V}(r) \hat{Q}_{ij}(r) dr
! and set PP%DION
      LYMAX=MAXL_AUG(1,PP); LMMAX=(LYMAX+1)**2
      ALLOCATE(V(PP%R%NMAX,LMMAX,1))
      SCALE=1/(2*SQRT(PI))
      V=0
      V(:,1,1)=PP%V00_PS(:)/SCALE
      CALL RAD_POT_WEIGHT(PP%R,1,LYMAX,V)
      DLM=0; DTMP=0; DUNSCR=0
      CALL RAD_AUG_PROJ(V(:,:,1),PP%R,DLM,PP%LMAX,PP%LPS,LYMAX,PP%AUG,PP%QPAW)
      CALL TRANS_DLM_RC(DTMP(:,:,1),DLM,PP)
      
      DUNSCR=DTMP(:,:,1)
      LM=1
      DO I=1,PP%LMAX
      LMP=1
      DO J=1,PP%LMAX
         PP%DION(I,J)=PP%DIJ(I,J)+DUNSCR(LM,LMP)
         LMP=LMP+2*PP%LPS(J)+1
      ENDDO
      LM=LM+2*PP%LPS(I)+1
      ENDDO
#ifdef testout
      WRITE(*,*) 'DION'
      DO I=1,PP%LMAX
         WRITE(*,'(10F12.8)') (PP%DION(I,J), J=1,PP%LMAX)
      ENDDO
#endif
      DEALLOCATE(V)
      RETURN
      END SUBROUTINE SET_PAW_AUG_RC


      SUBROUTINE RCPOSTPROC
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar),POINTER :: PP
      INTEGER NT,NI

      DO NT=1,NTYP
         IF (.NOT.DO_TYPE_LOCAL(NT)) CYCLE
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)      
         CALL GET_ECORE_TEST(PP)
      ENDDO
        
      RETURN
      END SUBROUTINE RCPOSTPROC

      SUBROUTINE GET_ECORE_TEST(PP)
      USE prec
      USE constant
      USE pseudo
      USE radial
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER I
      
      REAL(q) QCORE
      REAL(q) EIG,EKIN,DBLC,DHARTREE1,DHARTREE2,DEXC

      INTEGER LYMAX,LMMAX
      
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK1,WORK2
      REAL(q), ALLOCATABLE :: RHO(:,:,:)
      REAL(q) ZVAL,ZCORE
      
      INTEGER, EXTERNAL :: MAXL_AUG
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2
      
      ALLOCATE(RHO(PP%R%NMAX,LMMAX,1))
      
      RHO=0
      RHO(:,1,1)=PP%RHOAE00(:)+PP%RHOAE(:)

      ZVAL=PP%ZVALF_ORIG
      ZCORE=PP%ZCORE
      
      PP%ZVALF_ORIG=PP%ZVALF_ORIG+6
      PP%ZCORE=PP%ZCORE-6
      
      CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK1)
      ! \int n_c(r) (V_H[n_v+n_Zc]+V_xc[n_v+n_c]) dr
      DBLC=0
      DO I=1,PP%R%NMAX
         DBLC=DBLC+2*SQRT(PI)*WORK1(I)*PP%V00_AE(I)*PP%R%SI(I)
      ENDDO
      ! kinetic energy of the core electrons
      EKIN=EIG-DBLC
      ! \int n_c(r) V_H[n_c] dr
      CALL RAD_POT_HAR(0,PP%R,WORK2,WORK1,DHARTREE1)
      ! \int n_c(r) V_H[n_Z] dr
      DHARTREE2=0
      DO I=1,PP%R%NMAX
         DHARTREE2=DHARTREE2- &
        &   WORK1(I)*PP%R%SI(I)* &
        &   2*SQRT(PI)*FELECT*(PP%ZVALF_ORIG+PP%ZCORE)/PP%R%R(I)
      ENDDO
      ! Exc[rhoc]
      CALL RAD_CORE_XC(PP%R,WORK1,DEXC)
#ifdef testout
      CALL SIMPI(PP%R,WORK1,QCORE)
      WRITE(*,*) 'GET_ECORE_TEST:'
      WRITE(*,*) '---------------------------------------------'
      WRITE(*,*) 'Z_t                   :',PP%ZVALF_ORIG+PP%ZCORE
      WRITE(*,*) 'Z_c                   :',QCORE*2*SQRT(PI)
      WRITE(*,*) 
      WRITE(*,*) 'core eigenvalue sum   :',EIG
      WRITE(*,*) '-int V(r)*rhoc(r) dr  :',-DBLC
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'core kinetic energy   :',EKIN
      WRITE(*,*) '1/2(n_c)(n_c)         :',DHARTREE1/2
      WRITE(*,*) '(n_c)(n_z)            :',DHARTREE2
      WRITE(*,*) 'DEXC                  :',DEXC
      WRITE(*,*) '-PP%EATOM             :',-PP%EATOM
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'Total                 :', &
     &   EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM
!     WRITE(*,*) 'E_xc[n_c]             :',DEXC
      WRITE(*,*)
#endif         
      ! contribution to the total energy
!     PP%ECORE(1)=EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM
!     PP%ECORE(1)=0._q

      DEALLOCATE(RHO)
      RETURN
      END SUBROUTINE GET_ECORE_TEST
      
      SUBROUTINE ECORE0_TEST(PP)
      USE prec
      USE constant
      USE pseudo
      USE radial
      USE paw
      IMPLICIT NONE
      TYPE (potcar),POINTER :: PP
      INTEGER I
      INTEGER LYMAX,LMMAX      
      INTEGER, EXTERNAL :: MAXL_AUG
      REAL(q) EIG,DBLC,EKIN,DHARTREE1,DHARTREE2,DEXC,QCORE,ESHIFT
      REAL(q), DIMENSION(PP%R%NMAX) :: WORK1,WORK2
      REAL(q), ALLOCATABLE :: RHO(:,:,:),RHOLM(:)
      OVERLAP ,ALLOCATABLE :: CRHODE(:,:)
      LOGICAL LCLSHIFT_STORE
      
      REAL(q) ZVAL,ZCORE
      
      LYMAX=MAXL_AUG(1,PP)
      LMMAX=(LYMAX+1)**2

      ALLOCATE(CRHODE(LMMAX,LMMAX),RHOLM(LMMAX*LMMAX), &
     &   RHO(PP%R%NMAX,LMMAX,1))

      ! set CRHODE to the atomic occupancies
#ifdef testout
      WRITE(*,*) 'ECORE0: Trace of atomic occupancy matrix:'
      WRITE(*,*) (PP%QATO(I,I),I=1,PP%LMAX)
#endif      
      CRHODE=0
      CALL SET_CRHODE_ATOM(CRHODE(:,:),PP)
      ! calculate the valence AE charge density
      RHOLM=0
      CALL TRANS_RHOLM(CRHODE(:,:),RHOLM(:),PP)
      RHO=0
      CALL RAD_CHARGE(RHO(:,:,1),PP%R,RHOLM(:),PP%LMAX,PP%LPS,PP%WAE)
      ! add the AE core charge density
      RHO(:,1,1)=RHO(:,1,1)+PP%RHOAE(:)
#ifdef testout
!     WRITE(200,'(2F14.7)') (PP%R%R(I),RHO(I,1,1),I=1,PP%R%NMAX)
!     WRITE(201,'(2F14.7)') (PP%R%R(I),PP%RHOAE(I),I=1,PP%R%NMAX)
#endif
      ! get the sum of the core eigenvalues
      ESHIFT=PP%AVERAGEPOT(1)
      LCLSHIFT_STORE=LCLSHIFT
      LCLSHIFT=.FALSE.

      ZVAL=PP%ZVALF_ORIG
      ZCORE=PP%ZCORE

      PP%ZVALF_ORIG=PP%ZVALF_ORIG+6
      PP%ZCORE=PP%ZCORE-6

      CALL ATOM(RHO,INT(PP%ZVALF_ORIG+PP%ZCORE),PP,EIG,WORK1)
      LCLSHIFT=LCLSHIFT_STORE
      ! shift the core state eigenvalues (for output lateron)
      ESHIFT=ESHIFT-PP%AVERAGEPOT(1)
      PP%CLEV(:)=PP%CLEV(:)+ESHIFT
      ! \int n_c(r) (V_H[n_v+n_Zc]+V_xc[n_v+n_c]) dr
      DBLC=0
      DO I=1,PP%R%NMAX
         DBLC=DBLC+2*SQRT(PI)*WORK1(I)*PP%V00_AE(I)*PP%R%SI(I)
      ENDDO
      ! kinetic energy of the core electrons
      EKIN=EIG-DBLC
      ! \int n_c(r) V_H[n_c] dr
      CALL RAD_POT_HAR(0,PP%R,WORK2,WORK1,DHARTREE1)
      ! \int n_c(r) V_H[n_Z] dr
      DHARTREE2=0
      DO I=1,PP%R%NMAX
         DHARTREE2=DHARTREE2- &
        &   WORK1(I)*PP%R%SI(I)* &
        &   2*SQRT(PI)*FELECT*(PP%ZVALF_ORIG+PP%ZCORE)/PP%R%R(I)
      ENDDO
      ! Exc[rhoc]
      CALL RAD_CORE_XC(PP%R,WORK1,DEXC)
#ifdef testout
      CALL SIMPI(PP%R,WORK1,QCORE)
      WRITE(*,*) 'ECORE0_TEST:'
      WRITE(*,*) '---------------------------------------------'
      WRITE(*,*) 'Z_t                   :',PP%ZVALF_ORIG+PP%ZCORE
      WRITE(*,*) 'Z_c                   :',QCORE*2*SQRT(PI)
      WRITE(*,*) 
      WRITE(*,*) 'core eigenvalue sum   :',EIG
      WRITE(*,*) '-int V(r)*rhoc(r) dr  :',-DBLC
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'core kinetic energy   :',EKIN
      WRITE(*,*) '1/2(n_c)(n_c)         :',DHARTREE1/2
      WRITE(*,*) '(n_c)(n_z)            :',DHARTREE2
      WRITE(*,*) 'PP%DEXCCORE           :',DEXC
      WRITE(*,*) '-PP%EATOM             :',-PP%EATOM
      WRITE(*,*) '                      --------------------- +'
      WRITE(*,*) 'Total                 :', &
     &   EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM
!     WRITE(*,*) 'E_xc[n_c]             :',DEXC
      WRITE(*,*)
#endif         
      ! contribution to the total energy
!     PP%ECORE(2)=EKIN+DHARTREE1/2+DHARTREE2+PP%DEXCCORE-PP%EATOM

      PP%ZVALF_ORIG=ZVAL
      PP%ZCORE=ZCORE

      DEALLOCATE(CRHODE,RHOLM,RHO)            
      RETURN
      END SUBROUTINE ECORE0_TEST
            
      END MODULE core_rel

      
!*********************** SUBROUTINE GET_AVERAGEPOT_PW **************
!
! this subroutine calculates the average electrostatic potential at 
! a particular site, using either the one normalized augmentation 
! charges (PAW), or a spherical test charge with an automatically 
! chosen radius (NC, US-PP).
!
!*******************************************************************
      SUBROUTINE GET_AVERAGEPOT_PW( GRIDC, LATT_CUR, IRDMAX, &
     &       T_INFO, P, NCDIJ, CVTOT, ENAUG, IU6)
      USE prec
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE asa
      USE radial
      USE constant
      USE pseudo
      USE pp_data
      USE core_rel
      
      IMPLICIT NONE

      TYPE (grid_3d)     GRIDC    ! grid descriptor
      TYPE (type_info)   T_INFO   ! type descriptor
      TYPE (latt)        LATT_CUR ! lattice descriptor
      TYPE (potcar), TARGET :: P(T_INFO%NTYPD)
      INTEGER NCDIJ               ! number of dimensions of local potential
      RGRID :: CVTOT(DIMREAL(GRIDC%MPLWV),NCDIJ)  ! local potential
      INTEGER  IRDMAX             ! allocation required for augmentation
      REAL(q)  ENAUG              ! cutoff for augmentation grid
      INTEGER  IU6                ! IO unit
!  work arrays
      TYPE (rgrid), TARGET :: R
      TYPE (rgrid), POINTER :: RP
      TYPE (potcar), POINTER :: PP
      
      REAL(q), ALLOCATABLE :: AVERAGEPOT_PW(:)
      REAL(q), ALLOCATABLE :: DIST(:),DEP(:),POT(:),YLM(:,:),XS(:),YS(:),ZS(:)
      INTEGER, ALLOCATABLE :: NLI(:)
      REAL(q) :: NORM,SUM1,SUM2, RINPL, RCORE
      INTEGER :: LMYDIM, LYMAX, NI, NT, NPS, I, IRDMAX_LOCAL, N, INDMAX, IND

      INTEGER, PARAMETER :: NQ=2
      REAL(q)  QQ(NQ)    ! parameters of Besselfunctions
      REAL(q)  A(NQ)     ! parameters of Besselfunctions

      ALLOCATE(AVERAGEPOT_PW(T_INFO%NIONS))
           
      LMYDIM=1        ! number of lm pairs
!=======================================================================
! first set default grid (used for non PAW potentials)
!=======================================================================
      RCORE = 6/SQRT(ENAUG /RYTOEV)

      IRDMAX_LOCAL =4*PI*RCORE**3/3/(LATT_CUR%OMEGA/ &
     &     (GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ))+200

      IRDMAX_LOCAL=MAX(IRDMAX,IRDMAX_LOCAL)

      R%NMAX  =400
      R%RSTART=1E-2
      R%REND  =RCORE
      R%D     =(R%REND/R%RSTART)**(1._q/(R%NMAX-1)) 
      R%H     =LOG(R%D)

      ALLOCATE(R%R(R%NMAX))

      DO I=1,R%NMAX
         R%R(I)=R%RSTART*EXP((I-1)*R%H)
      ENDDO

      R%RMAX=  R%R(R%NMAX)
      R%REND  =RCORE

      CALL SET_SIMP(R)

      ALLOCATE( DIST(IRDMAX_LOCAL),DEP(IRDMAX_LOCAL),POT(IRDMAX_LOCAL), &
     &        YLM(IRDMAX_LOCAL,LMYDIM),NLI(IRDMAX_LOCAL), &
     &        XS(IRDMAX_LOCAL),YS(IRDMAX_LOCAL),ZS(IRDMAX_LOCAL))

      AVERAGEPOT_PW = 0
      NORM  = 0

!=======================================================================
! loop over all ions
!=======================================================================
      ion: DO NI=1,T_INFO%NIONS
      NT=T_INFO%ITYP(NI)
      PP=>PP_POINTER(P,NI,NT)
      !
      ! PAW use the conventional grids (as supplied)
      !
         IF (.NOT. ASSOCIATED(PP%QPAW)) THEN
            RP=>R
         ELSE
            RP=>PP%R
         ENDIF

         CALL AUG_SETQ(0,RP,RP%RMAX,QQ,A,LCOMPAT=.FALSE.)

         A=A/PI/4
         RINPL=1._q/GRIDC%NPLWV

!-----------------------------------------------------------------------
! calculate the spherical harmonics YLM and the distance DIST  between
! grid-points and central atom (DEP and POT are work-arrays)
!-----------------------------------------------------------------------
         LYMAX=0

         NPS=1000000  ! (cutoff sphere is given by R%RMAX *(NPS-1)/NPS
                      ! use all points
         CALL SETYLM_AUG(GRIDC,LATT_CUR,T_INFO%POSION(1,NI),RP%RMAX,NPS, &
     &        LMYDIM,LYMAX,YLM(1,1),IRDMAX_LOCAL,INDMAX, &
     &        0.0_q,0.0_q,0.0_q,DIST(1),NLI(1), &
     &        XS(1),YS(1),ZS(1))
         !
         ! in the spin polarised case the up and down potential 
         ! is averaged
         ! 
         DO N=1,INDMAX
            POT(N)=CVTOT(NLI(N),1)
         ENDDO
         IF (NCDIJ==2) THEN
           DO N=1,INDMAX
             POT(N)=(POT(N)+CVTOT(NLI(N),2))/2
           ENDDO
         ENDIF
         IF (NCDIJ==4) THEN
           DO N=1,INDMAX
             POT(N)=(POT(N)+CVTOT(NLI(N),4))/2
           ENDDO
         ENDIF
!=======================================================================
! calculate the integral  int V Y(0,0) Q(0)
!=======================================================================
         CALL SETDEP_B(RP,QQ,A,LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))

         SUM1=0
         SUM2=0
!DIR$ IVDEP
!OCL NOVREC
         DO IND=1,INDMAX
            SUM1=SUM1+POT(IND)*DEP(IND)
            SUM2=SUM2+DEP(IND)
         ENDDO
         AVERAGEPOT_PW(NI)=SUM1*RINPL
         NORM=NORM+SUM2*RINPL
!=======================================================================
      ENDDO ion
!-----------------------------------------------------------------------
      DEALLOCATE(DIST,DEP,POT,YLM,XS,YS,ZS,NLI)
      NULLIFY(R%R)
! reduce AVERAGEPOT_PW and NORM
      CALLMPI( M_sum_d(GRIDC%COMM , AVERAGEPOT_PW ,T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM , NORM ,1))
#ifdef testout
      WRITE(*,*) 'GET_AVERAGEPOT_PW: Av(V) on PW grid in PAW sphere'
      WRITE(*,*)
      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I4,F14.7)') NI,AVERAGEPOT_PW(NI)
      ENDDO
      WRITE(*,*)
#endif
      IF (LCORREL().AND.LSPAWN_PP_DONE().AND.LCORREL_INIT_DONE()) THEN
         DO NT=1,NIRRED_IONS
            NI=TYPE2ION(NT)
            PP=>SET_PP_POINTER(NI)
            PP%AVERAGEPOT(3)=AVERAGEPOT_PW(NI)
         ENDDO
      ENDIF
      DEALLOCATE(AVERAGEPOT_PW)
      RETURN
      END SUBROUTINE GET_AVERAGEPOT_PW


      SUBROUTINE PW_TO_RADIAL(WDES,GRIDC,CHTOT,LATT_CUR,T_INFO)
      
      USE prec
      USE wave
      USE mpimy
      USE mgrid
      USE lattice
      USE poscar
      USE pp_data
      USE core_rel
      USE constant

      IMPLICIT NONE

      TYPE(wavedes) WDES
      TYPE(grid_3d) GRIDC
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      
      TYPE(potcar), POINTER :: PP

      COMPLEX(q) CHTOT(GRIDC%RC%NP)

      INTEGER NI,NT,NRWIGS,NR,NP,N
      INTEGER I,J,K
      INTEGER N1,N2,N3,N1P,N2P,N3P,IND
      INTEGER IPH,ITH,NTH
      INTEGER P1(4),P2(4),P3(4)

      REAL(q),ALLOCATABLE ::  CWORK(:)
      complex(q) chwork(10)
      REAL(q) F1,F2,F3
      REAL(q) POSION(3),VEC(3),POS(3)
      REAL(q) R,PH,TH
      REAL(q) CH(4,4,4),F,DF
      REAL(q) SUM

      REAL(q), PARAMETER :: D=0.1_q

      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

      N =GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ
      ALLOCATE(CWORK(N))
      chwork(1:10)=chtot(1:10)
      CALL FFT3D(CHTOT,GRIDC,1)
      ! merge charge to CWORK (result is real)
      CALL MRG_GRID_RL(GRIDC, CWORK, CHTOT)
      ! FFT of the symmetrized real space density (CHTOT):
      CALL FFT_RC_SCALE(CHTOT,CHTOT,GRIDC)
      CALL SETUNB_COMPAT(CHTOT,GRIDC)

      types: DO NT=1,NIRRED_IONS
         NI=TYPE2ION(NT)
         PP=>SET_PP_POINTER(NI)
         IF (DO_TYPE_LOCAL(NT)) THEN
            POSION=T_INFO%POSION(:,NI)

            DO I=1,PP%R%NMAX-1
               IF (PP%R%R(I)>=PP%RWIGS) EXIT
            ENDDO
            NRWIGS=I-80

            DO NR=NRWIGS,PP%R%NMAX
               NP=0
               SUM=0
               R=PP%R%R(NR)
               NTH=PI*R/D
               DO ITH=0,NTH-1
               DO IPH=0,2*NTH-1
                  PH=IPH*PI/NTH
                  TH=ITH*PI/NTH
!                 ph=pi/4
!                 th=0
                  
                  VEC(1)=R*COS(TH)*COS(PH)
                  VEC(2)=R*COS(TH)*SIN(PH)
                  VEC(3)=R*SIN(TH)
                  
                  CALL KARDIR(1,VEC,LATT_CUR%B)
                  
                  POS=POSION+VEC
                  ! find nearest grid point
                  N1=FLOOR(POS(1)/F1)
                  N2=FLOOR(POS(2)/F2)
                  N3=FLOOR(POS(3)/F3)
                  
                  DO I=1,4
                     P1(I)=N1-2+I
                     P2(I)=N2-2+I
                     P3(I)=N3-2+I
                  ENDDO

                  DO I=1,4
                  DO J=1,4
                  DO K=1,4
                     N1P=MOD(P1(I)+10*GRIDC%NGX,GRIDC%NGX)
                     N2P=MOD(P2(J)+10*GRIDC%NGY,GRIDC%NGY)
                     N3P=MOD(P3(K)+10*GRIDC%NGZ,GRIDC%NGZ)
                     IND=N1P+N2P*GRIDC%NGX+N3P*GRIDC%NGX*GRIDC%NGY+1
                     CH(I,J,K)=REAL(CWORK(IND)) !*LATT_CUR%OMEGA
                  ENDDO
                  ENDDO
                  ENDDO
                  ! interpolate
                  CALL POLIN3(P1,P2,P3,CH,4,4,4,POS(1)/F1,POS(2)/F2,POS(3)/F3,F,DF)
                  NP=NP+1
                  SUM=SUM+F
               ENDDO
               ENDDO
               SUM=SUM/NP*PP%R%R(NR)*PP%R%R(NR)/LATT_CUR%OMEGA*(2*SQRT(PI))
!              WRITE(*,'(F14.7,I8,F14.7)') PP%R%R(NR),NP,SUM
!              WRITE(*,*) SUM
!              WRITE(100,'(2F14.7)') PP%R%R(NR),SUM*PP%R%R(NR)*PP%R%R(NR)/LATT_CUR%OMEGA*(2*SQRT(PI))
               PP%RHOPSPW(NR)=SUM
            ENDDO
         ENDIF
      ENDDO types
      DEALLOCATE(CWORK)
      RETURN
      END SUBROUTINE PW_TO_RADIAL
      
      
      SUBROUTINE POLIN3(IXA,IYA,IZA,FA,NX,NY,NZ,X,Y,Z,F,DF)

      USE prec

      IMPLICIT NONE

      INTEGER                    I,J
      INTEGER                    NX,NY,NZ,NXMAX,NYMAX,NZMAX
      INTEGER                    IXA(NX),IYA(NY),IZA(NZ)

      REAL(q)                    X,Y,Z
      REAL(q)                    F,DF
      REAL(q)                    XA(NX),YA(NY),ZA(NZ),FA(NX,NY,NZ)

      PARAMETER(NXMAX=10,NYMAX=10,NZMAX=10)

      REAL(q)                    FXTMP(NXMAX),FYTMP(NYMAX),FZTMP(NZMAX)

      XA=REAL(IXA); YA=REAL(IYA); ZA=REAL(IZA)

      DO I=1,NZ
         DO J=1,NY
            FXTMP(1:4)=FA(1:4,J,I)
            CALL POLINT_(XA,FXTMP(1:NX),NX,X,FYTMP(J),DF)
         ENDDO
         CALL POLINT_(YA,FYTMP(1:NY),NY,Y,FZTMP(I),DF)
      ENDDO
      CALL POLINT_(ZA,FZTMP(1:NZ),NZ,Z,F,DF)

      RETURN
      END SUBROUTINE POLIN3


      SUBROUTINE POLINT_(XA,YA,N,X,Y,DY)

      USE prec

      IMPLICIT NONE

      INTEGER                    N,NMAX
      INTEGER                    I,M,NS

      REAL(q)                    X,Y,DY
      REAL(q)                    XA(N),YA(N)
      REAL(q)                    DEN,DIF,DIFT,HO,HP,W

      PARAMETER(NMAX=10)

      REAL(q)                    C(NMAX),D(NMAX)


      NS=1
      DIF=ABS(X-XA(1))
      DO I=1,N
         DIFT=ABS(X-XA(I))
         IF (DIFT.LT.DIF) THEN
            NS=I
            DIF=DIFT
         ENDIF
         C(I)=YA(I)
         D(I)=YA(I)
      ENDDO
! MM. When closest table entry is known to be XA(2)
!     NS=2
!     DIF=ABS(X-XA(2))
!     C=YA
!     D=YA
! MM.
      Y=YA(NS)
      NS=NS-1
      DO M=1,N-1
         DO I=1,N-M
            HO=XA(I)-X
            HP=XA(I+M)-X
            W=C(I+1)-D(I)
            DEN=HO-HP
! MM. When input XA's are all inequivalent (to within rounoff)
!           IF(DEN.EQ.0._q)PAUSE 'FAILURE IN POLINT'
! MM.
            DEN=W/DEN
            D(I)=HP*DEN
            C(I)=HO*DEN
         ENDDO
         IF (2*NS.LT.N-M)THEN
            DY=C(NS+1)
         ELSE
            DY=D(NS)
            NS=NS-1
         ENDIF
         Y=Y+DY
      ENDDO

      RETURN
      END SUBROUTINE POLINT_
         
         
               
!************************************************************************
!     Numerical stuff
!************************************************************************
      subroutine dgeco(a,lda,n,ipvt,rcond,z)
      use prec
      integer lda,n,ipvt(1)
      real(q) a(lda,1),z(1)
      real(q) rcond
!
!     dgeco factors a real(q) matrix by gaussian elimination
!     and estimates the condition of the matrix.
!
!     if  rcond  is not needed, dgefa is slightly faster.
!     to solve  a*x = b , follow dgeco by dgesl.
!     to compute  inverse(a)*c , follow dgeco by dgesl.
!     to compute  determinant(a) , follow dgeco by dgedi.
!     to compute  inverse(a) , follow dgeco by dgedi.
!
!     on entry
!
!        a       real(q)(lda, n)
!                the matrix to be factored.
!
!        lda     integer
!                the leading dimension of the array  a .
!
!        n       integer
!                the order of the matrix  a .
!
!     on return
!
!        a       an upper triangular matrix and the multipliers
!                which were used to obtain it.
!                the factorization can be written  a = l*u  where
!                l  is a product of permutation and unit lower
!                triangular matrices and  u  is upper triangular.
!
!        ipvt    integer(n)
!                an integer vector of pivot indices.
!
!        rcond   real(q)
!                an estimate of the reciprocal condition of  a .
!                for the system  a*x = b , relative perturbations
!                in  a  and  b  of size  epsilon  may cause
!                relative perturbations in  x  of size  epsilon/rcond .
!                if  rcond  is so small that the logical expression
!                           1.0 + rcond .eq. 1.0
!                is true, then  a  may be singular to working
!                precision.  in particular,  rcond  is zero  if
!                exact singularity is detected or the estimate
!                underflows.
!
!        z       real(q)(n)
!                a work vector whose contents are usually unimportant.
!                if  a  is close to a singular matrix, then  z  is
!                an approximate null vector in the sense that
!                norm(a*z) = rcond*norm(a)*norm(z) .
!
!     linpack. this version dated 08/14/78 .
!     cleve moler, university of new mexico, argonne national lab.
!
!     subroutines and functions
!
!     linpack dgefa
!     blas daxpy,ddot,dscal,dasum
!     fortran abs,max,sign
!
!     internal variables
!
      real(q) ddot,ek,t,wk,wkm
      real(q) anorm,s,dasum,sm,ynorm
      integer info,j,k,kb,kp1,l
!
!
!     compute 1-norm of a
!
      anorm = 0.0_q
      do 10 j = 1, n
         anorm = max(anorm,dasum(n,a(1,j),1))
   10 continue
!
!     factor
!
      call dgefa(a,lda,n,ipvt,info)
!
!     rcond = 1/(norm(a)*(estimate of norm(inverse(a)))) .
!     estimate = norm(z)/norm(y) where  a*z = y  and  trans(a)*y = e .
!     trans(a)  is the transpose of a .  the components of  e  are
!     chosen to cause maximum local growth in the elements of w  where
!     trans(u)*w = e .  the vectors are frequently rescaled to avoid
!     overflow.
!
!     solve trans(u)*w = e
!
      ek = 1.0_q
      do 20 j = 1, n
         z(j) = 0.0_q
   20 continue
      do 100 k = 1, n
         if (z(k) .ne. 0.0_q) ek = sign(ek,-z(k))
         if (abs(ek-z(k)) .le. abs(a(k,k))) go to 30
            s = abs(a(k,k))/abs(ek-z(k))
            call dscal(n,s,z,1)
            ek = s*ek
   30    continue
         wk = ek - z(k)
         wkm = -ek - z(k)
         s = abs(wk)
         sm = abs(wkm)
         if (a(k,k) .eq. 0.0_q) go to 40
            wk = wk/a(k,k)
            wkm = wkm/a(k,k)
         go to 50
   40    continue
            wk = 1.0_q
            wkm = 1.0_q
   50    continue
         kp1 = k + 1
         if (kp1 .gt. n) go to 90
            do 60 j = kp1, n
               sm = sm + abs(z(j)+wkm*a(k,j))
               z(j) = z(j) + wk*a(k,j)
               s = s + abs(z(j))
   60       continue
            if (s .ge. sm) go to 80
               t = wkm - wk
               wk = wkm
               do 70 j = kp1, n
                  z(j) = z(j) + t*a(k,j)
   70          continue
   80       continue
   90    continue
         z(k) = wk
  100 continue
      s = 1.0_q/dasum(n,z,1)
      call dscal(n,s,z,1)
!
!     solve trans(l)*y = w
!
      do 120 kb = 1, n
         k = n + 1 - kb
         if (k .lt. n) z(k) = z(k) + ddot(n-k,a(k+1,k),1,z(k+1),1)
         if (abs(z(k)) .le. 1.0_q) go to 110
            s = 1.0_q/abs(z(k))
            call dscal(n,s,z,1)
  110    continue
         l = ipvt(k)
         t = z(l)
         z(l) = z(k)
         z(k) = t
  120 continue
      s = 1.0_q/dasum(n,z,1)
      call dscal(n,s,z,1)
!
      ynorm = 1.0_q
!
!     solve l*v = y
!
      do 140 k = 1, n
         l = ipvt(k)
         t = z(l)
         z(l) = z(k)
         z(k) = t
         if (k .lt. n) call daxpy(n-k,t,a(k+1,k),1,z(k+1),1)
         if (abs(z(k)) .le. 1.0_q) go to 130
            s = 1.0_q/abs(z(k))
            call dscal(n,s,z,1)
            ynorm = s*ynorm
  130    continue
  140 continue
      s = 1.0_q/dasum(n,z,1)
      call dscal(n,s,z,1)
      ynorm = s*ynorm
!
!     solve  u*z = v
!
      do 160 kb = 1, n
         k = n + 1 - kb
         if (abs(z(k)) .le. abs(a(k,k))) go to 150
            s = abs(a(k,k))/abs(z(k))
            call dscal(n,s,z,1)
            ynorm = s*ynorm
  150    continue
         if (a(k,k) .ne. 0.0_q) z(k) = z(k)/a(k,k)
         if (a(k,k) .eq. 0.0_q) z(k) = 1.0_q
         t = -z(k)
         call daxpy(k-1,t,a(1,k),1,z(1),1)
  160 continue
!     make znorm = 1.0
      s = 1.0_q/dasum(n,z,1)
      call dscal(n,s,z,1)
      ynorm = s*ynorm
!
      if (anorm .ne. 0.0_q) rcond = ynorm/anorm
      if (anorm .eq. 0.0_q) rcond = 0.0_q
      return
      end
      subroutine dgedi(a,lda,n,ipvt,det,work,job)
      use prec
      integer lda,n,ipvt(1),job
      real(q) a(lda,1),det(2),work(1)
!
!     dgedi computes the determinant and inverse of a matrix
!     using the factors computed by dgeco or dgefa.
!
!     on entry
!
!        a       real(q)(lda, n)
!                the output from dgeco or dgefa.
!
!        lda     integer
!                the leading dimension of the array  a .
!
!        n       integer
!                the order of the matrix  a .
!
!        ipvt    integer(n)
!                the pivot vector from dgeco or dgefa.
!
!        work    real(q)(n)
!                work vector.  contents destroyed.
!
!        job     integer
!                = 11   both determinant and inverse.
!                = 01   inverse only.
!                = 10   determinant only.
!
!     on return
!
!        a       inverse of original matrix if requested.
!                otherwise unchanged.
!
!        det     real(q)(2)
!                determinant of original matrix if requested.
!                otherwise not referenced.
!                determinant = det(1) * 10.0**det(2)
!                with  1.0 .le. abs(det(1)) .lt. 10.0
!                or  det(1) .eq. 0.0 .
!
!     error condition
!
!        a division by zero will occur if the input factor contains
!        a zero on the diagonal and the inverse is requested.
!        it will not occur if the subroutines are called correctly
!        and if dgeco has set rcond .gt. 0.0 or dgefa has set
!        info .eq. 0 .
!
!     linpack. this version dated 08/14/78 .
!     cleve moler, university of new mexico, argonne national lab.
!
!     subroutines and functions
!
!     blas daxpy,dscal,dswap
!     fortran abs,mod
!
!     internal variables
!
      real(q) t
      real(q) ten
      integer i,j,k,kb,kp1,l,nm1
!
!     compute determinant
!
      if (job/10 .eq. 0) go to 70
         det(1) = 1.0_q
         det(2) = 0.0_q
         ten = 10.0_q
         do 50 i = 1, n
            if (ipvt(i) .ne. i) det(1) = -det(1)
            det(1) = a(i,i)*det(1)
!        ...exit
            if (det(1) .eq. 0.0_q) go to 60
   10       if (abs(det(1)) .ge. 1.0_q) go to 20
               det(1) = ten*det(1)
               det(2) = det(2) - 1.0_q
            go to 10
   20       continue
   30       if (abs(det(1)) .lt. ten) go to 40
               det(1) = det(1)/ten
               det(2) = det(2) + 1.0_q
            go to 30
   40       continue
   50    continue
   60    continue
   70 continue
!
!     compute inverse(u)
!
      if (mod(job,10) .eq. 0) go to 150
         do 100 k = 1, n
            a(k,k) = 1.0_q/a(k,k)
            t = -a(k,k)
            call dscal(k-1,t,a(1,k),1)
            kp1 = k + 1
            if (n .lt. kp1) go to 90
            do 80 j = kp1, n
               t = a(k,j)
               a(k,j) = 0.0_q
               call daxpy(k,t,a(1,k),1,a(1,j),1)
   80       continue
   90       continue
  100    continue
!
!        form inverse(u)*inverse(l)
!
         nm1 = n - 1
         if (nm1 .lt. 1) go to 140
         do 130 kb = 1, nm1
            k = n - kb
            kp1 = k + 1
            do 110 i = kp1, n
               work(i) = a(i,k)
               a(i,k) = 0.0_q
  110       continue
            do 120 j = kp1, n
               t = work(j)
               call daxpy(n,t,a(1,j),1,a(1,k),1)
  120       continue
            l = ipvt(k)
            if (l .ne. k) call dswap(n,a(1,k),1,a(1,l),1)
  130    continue
  140    continue
  150 continue
      return
      end
      subroutine dgefa(a,lda,n,ipvt,info)
      use prec
      integer lda,n,ipvt(1),info
      real(q) a(lda,1)
!
!     dgefa factors a real(q) matrix by gaussian elimination.
!
!     dgefa is usually called by dgeco, but it can be called
!     directly with a saving in time if  rcond  is not needed.
!     (time for dgeco) = (1 + 9/n)*(time for dgefa) .
!
!     on entry
!
!        a       real(q)(lda, n)
!                the matrix to be factored.
!
!        lda     integer
!                the leading dimension of the array  a .
!
!        n       integer
!                the order of the matrix  a .
!
!     on return
!
!        a       an upper triangular matrix and the multipliers
!                which were used to obtain it.
!                the factorization can be written  a = l*u  where
!                l  is a product of permutation and unit lower
!                triangular matrices and  u  is upper triangular.
!
!        ipvt    integer(n)
!                an integer vector of pivot indices.
!
!        info    integer
!                = 0  normal value.
!                = k  if  u(k,k) .eq. 0.0 .  this is not an error
!                     condition for this subroutine, but it does
!                     indicate that dgesl or dgedi will divide by zero
!                     if called.  use  rcond  in dgeco for a reliable
!                     indication of singularity.
!
!     linpack. this version dated 08/14/78 .
!     cleve moler, university of new mexico, argonne national lab.
!
!     subroutines and functions
!
!     blas daxpy,dscal,idamax
!
!     internal variables
!
      real(q) t
      integer idamax,j,k,kp1,l,nm1
!
!
!     gaussian elimination with partial pivoting
!
      info = 0
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 k = 1, nm1
         kp1 = k + 1
!
!        find l = pivot index
!
         l = idamax(n-k+1,a(k,k),1) + k - 1
         ipvt(k) = l
!
!        zero pivot implies this column already triangularized
!
         if (a(l,k) .eq. 0.0_q) go to 40
!
!           interchange if necessary
!
            if (l .eq. k) go to 10
               t = a(l,k)
               a(l,k) = a(k,k)
               a(k,k) = t
   10       continue
!
!           compute multipliers
!
            t = -1.0_q/a(k,k)
            call dscal(n-k,t,a(k+1,k),1)
!
!           row elimination with column indexing
!
            do 30 j = kp1, n
               t = a(l,j)
               if (l .eq. k) go to 20
                  a(l,j) = a(k,j)
                  a(k,j) = t
   20          continue
               call daxpy(n-k,t,a(k+1,k),1,a(k+1,j),1)
   30       continue
         go to 50
   40    continue
            info = k
   50    continue
   60 continue
   70 continue
      ipvt(n) = n
      if (a(n,n) .eq. 0.0_q) info = n
      return
      end
      subroutine dgesl(a,lda,n,ipvt,b,job)
      use prec
      integer lda,n,ipvt(1),job
      real(q) a(lda,1),b(1)
!
!     dgesl solves the double precision system
!     a * x = b  or  trans(a) * x = b
!     using the factors computed by dgeco or dgefa.
!
!     on entry
!
!        a       double precision(lda, n)
!                the output from dgeco or dgefa.
!
!        lda     integer
!                the leading dimension of the array  a .
!
!        n       integer
!                the order of the matrix  a .
!
!        ipvt    integer(n)
!                the pivot vector from dgeco or dgefa.
!
!        b       double precision(n)
!                the right hand side vector.
!
!        job     integer
!                = 0         to solve  a*x = b ,
!                = nonzero   to solve  trans(a)*x = b  where
!                            trans(a)  is the transpose.
!
!     on return
!
!        b       the solution vector  x .
!
!     error condition
!
!        a division by zero will occur if the input factor contains a
!        zero on the diagonal.  technically this indicates singularity
!        but it is often caused by improper arguments or improper
!        setting of lda .  it will not occur if the subroutines are
!        called correctly and if dgeco has set rcond .gt. 0.0
!        or dgefa has set info .eq. 0 .
!
!     to compute  inverse(a) * c  where  c  is a matrix
!     with  p  columns
!           call dgeco(a,lda,n,ipvt,rcond,z)
!           if (rcond is too small) go to ...
!           do 10 j = 1, p
!              call dgesl(a,lda,n,ipvt,c(1,j),0)
!        10 continue
!
!     linpack. this version dated 08/14/78 .
!     cleve moler, university of new mexico, argonne national lab.
!
!     subroutines and functions
!
!     blas daxpy,ddot
!
!     internal variables
!
      real(q) ddot,t
      integer k,kb,l,nm1
!
      nm1 = n - 1
      if (job .ne. 0) go to 50
!
!        job = 0 , solve  a * x = b
!        first solve  l*y = b
!
         if (nm1 .lt. 1) go to 30
         do 20 k = 1, nm1
            l = ipvt(k)
            t = b(l)
            if (l .eq. k) go to 10
               b(l) = b(k)
               b(k) = t
   10       continue
            call daxpy(n-k,t,a(k+1,k),1,b(k+1),1)
   20    continue
   30    continue
!
!        now solve  u*x = y
!
         do 40 kb = 1, n
            k = n + 1 - kb
            b(k) = b(k)/a(k,k)
            t = -b(k)
            call daxpy(k-1,t,a(1,k),1,b(1),1)
   40    continue
      go to 100
   50 continue
!
!        job = nonzero, solve  trans(a) * x = b
!        first solve  trans(u)*y = b
!
         do 60 k = 1, n
            t = ddot(k-1,a(1,k),1,b(1),1)
            b(k) = (b(k) - t)/a(k,k)
   60    continue
!
!        now solve trans(l)*x = y
!
         if (nm1 .lt. 1) go to 90
         do 80 kb = 1, nm1
            k = n - kb
            b(k) = b(k) + ddot(n-k,a(k+1,k),1,b(k+1),1)
            l = ipvt(k)
            if (l .eq. k) go to 70
               t = b(l)
               b(l) = b(k)
               b(k) = t
   70       continue
   80    continue
   90    continue
  100 continue
      return
      end
