#include "symbol.inc"


!*********************************************************************
!
!  this module contains all necessary routines for the constrained RPA
!  mK
!
! TODO: implement LCAO basis set
!*********************************************************************

MODULE crpa
   USE prec 
   USE mlwf
   USE wnpr 
   IMPLICIT NONE
   !Projection onto correlated sub subspace (for CRPA) 
   GDEF, ALLOCATABLE, SAVE      :: UCRPA(:,:,:,:)
   GDEF, ALLOCATABLE, SAVE      :: UCDER(:,:,:,:)
   !number of states
   INTEGER, SAVE                :: DMFT_num_wann = -1 
   !target CRPA bands
   INTEGER, SAVE                :: NCRPALOW = -1
   INTEGER, SAVE                :: NCRPAHIGH = -1
   !Wannier projections (basis set)
   !COMPLEX(q),ALLOCATABLE, SAVE :: UWAN(:,:,:,:)
   GDEF,ALLOCATABLE, SAVE :: UWAN(:,:,:,:)
 
   !disentanglement of Miyake
   LOGICAL, SAVE                :: LDISENTANGLE= .FALSE.  
   !target Bloch states for disentangled CRPA
   LOGICAL, ALLOCATABLE, SAVE   :: LCRPABAND(:,:,:)
   !weighted method of Friedrich and Shih
   LOGICAL, SAVE                :: LWEIGHTED= .FALSE.  
   !weights for weighted CRPA
   REAL(q), ALLOCATABLE, SAVE   :: CRPA_WEIGHTS(:,:,:)
   
   !use bloch basis  
   LOGICAL, SAVE                :: LUSEBLOCH = .FALSE.   
   !use MLWF basis (this is the default)
   LOGICAL, SAVE                :: LUSEMLWF = .FALSE.
   !use LCAO basis 
   LOGICAL, SAVE                :: LUSELCAO = .FALSE.
   !use orthogonal LCAOs
   LOGICAL, SAVE                :: LUSEORTH_LCAOS=.FALSE.
   
   !Constrained RPA and non-local RPA flags (used in chi.F)
   LOGICAL, SAVE :: LCRPA=.FALSE.
   LOGICAL, SAVE :: LNLRPA=.FALSE.
   !this defines the target states in localized basis
   INTEGER, ALLOCATABLE, SAVE   :: NTARGET_STATES(:)
   !optionally the Wannier projections are read from UWAN
   LOGICAL, SAVE                :: LUFROMFILE=.FALSE.
 
   !include Drude term in chi ?
   LOGICAL, SAVE                :: LDRUDE=.FALSE.
   !plot CRPA banstructure    
   LOGICAL, SAVE                :: LCRPAPLOT = .FALSE. 
   !compute full RPA 
   LOGICAL, SAVE                :: LFRPA = .FALSE.  
   !regularize
   LOGICAL, PRIVATE, SAVE       :: LREGULARIZE = .TRUE.  

   !some constants
   REAL(q), PRIVATE, PARAMETER  :: TINY=1E-11_q
   REAL(q), PRIVATE, PARAMETER  :: TINYLARGE=1E-6_q
   !number of interations in Jacobi diagonalizer
   INTEGER, SAVE                :: MAXLIE = -1

   LOGICAL, PRIVATE, SAVE       :: LPLOTDIS = .FALSE. 
 
   !correlated polariazbility needs to be computed?
   LOGICAL, SAVE :: LCHIC=.FALSE. 
   LOGICAL, SAVE :: LSUBTRACT_CHIC=.FALSE. 
   LOGICAL, SAVE :: LALLOCATED_WCORR=.FALSE. 

   !relevant bands for correlated Green's function
   INTEGER, SAVE :: NCRPAMIN = -1
   INTEGER, SAVE :: NCRPAMAX = 1E8
CONTAINS

!*********************************************************************
!
! reader of important flags for CRPA 
!
!*********************************************************************
   SUBROUTINE CRPA_READER( IU5, IU6, IU0, NBANDS, L2E4W, L2E4WALL, LDOWNSAMPLE)
      USE vaspxml
      USE full_kpoints
      USE mkpoints
      USE base
      USE mlwf, ONLY: LWANNIER90,LWANNIER90_RUN
      USE wnpr, ONLY: LWANPROJ
      USE pead
      USE main_mpi, ONLY: NCSHMEM
      IMPLICIT NONE
      INTEGER IU5, IU6, IU0
      INTEGER NBANDS
      LOGICAL L2E4W, L2E4WALL
      LOGICAL  LDOWNSAMPLE
      ! local
      INTEGER I, IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN, LDUM
      CHARACTER (1) :: CHARAC
      CHARACTER (40)   STRING
   
      !security break, SHMEM not supported for CRPA calculations
      IF ( (LCRPA .OR. LNLRPA .OR. L2E4W .OR. L2E4WALL) .AND. NCSHMEM > 1 )  THEN
         CALL VTUTOR('E','SHMEMCRPA', &
        &               0.0_q,1,1,2,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
         CALL VTUTOR('S','SHMEMCRPA', &
        &               0.0_q,1,1,2,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
      ENDIF 
  
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      !maximum iterations in diagonalization routines
      MAXLIE=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'MAXLIE','=','#',';','I', &
          &            MAXLIE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''MAXLIE'' from file INCAR.'
         MAXLIE=-1
      ENDIF
      IF ( MAXLIE < 0 ) THEN
         MAXLIE = 150
      ENDIF  
      CALL XML_INCAR('MAXLIE','I',MAXLIE,RDUM,CDUM,LDUM,CHARAC,N)
  
      !crpa bands
      NCRPALOW=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'NCRPALOW','=','#',';','I', &
          &            NCRPALOW,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NCRPALOW'' from file INCAR.'
         NCRPALOW=0
      ENDIF
      CALL XML_INCAR('NCRPALOW','I',NCRPALOW,RDUM,CDUM,LDUM,CHARAC,N)
  
      NCRPAHIGH=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'NCRPAHIGH','=','#',';','I', &
          &            NCRPAHIGH,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NCRPAHIGH'' from file INCAR.'
         NCRPAHIGH=0
      ENDIF
      CALL XML_INCAR('NCRPAHIGH','I',NCRPAHIGH,RDUM,CDUM,LDUM,CHARAC,N)
  
      ! plot bands used for correlated CHI in CRPA 
      LCRPAPLOT=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LCRPAPLOT','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LCRPAPLOT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LCRPAPLOT'' from file INCAR.'
         LCRPAPLOT=.FALSE.
      ENDIF
      CALL XML_INCAR('LCRPAPLOT','L',IDUM,RDUM,CDUM,LCRPAPLOT,CHARAC,N)

      ! disentangled bands are plot 
      LPLOTDIS=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LPLOTDIS','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LPLOTDIS,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LPLOTDIS'' from file INCAR.'
         LPLOTDIS=.FALSE.
      ENDIF
      CALL XML_INCAR('LPLOTDIS','L',IDUM,RDUM,CDUM,LPLOTDIS,CHARAC,N)

      !also force downsampling of WAVECAR if CRPA bands are plotted 
      IF ( LCRPAPLOT .AND. .NOT. LDOWNSAMPLE ) THEN
         IF ( IU0>=0 ) WRITE(IU0,*)'LDOWNSAMPLE forced by LCRPAPLOT '
         LDOWNSAMPLE = LCRPAPLOT 
      ENDIF 
   
      !disentanglement approach is done
      LDISENTANGLE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LDISENTANGLE','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LDISENTANGLE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LDISENTANGLE'' from file INCAR.'
         LDISENTANGLE=.FALSE.
      ENDIF
      CALL XML_INCAR('DISENTANGLE','L',IDUM,RDUM,CDUM,LDISENTANGLE,CHARAC,N)
  
      !weighted approach is done
      LWEIGHTED=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LWEIGHTED','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LWEIGHTED,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LWEIGHTED'' from file INCAR.'
         LWEIGHTED=.FALSE.
      ENDIF
      CALL XML_INCAR('WEIGHTING','L',IDUM,RDUM,CDUM,LWEIGHTED,CHARAC,N)
  
      !don't use orthogonal LCAOs by default
      LUSEORTH_LCAOS=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LUSEORTH_LCAOS','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LUSEORTH_LCAOS,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LUSEORTH_LCAOS'' from file INCAR.'
         LUSEORTH_LCAOS=.FALSE.
      ENDIF
      CALL XML_INCAR('LUSEORTH_LCAOS','L',IDUM,RDUM,CDUM,LUSEORTH_LCAOS,CHARAC,N)
   
      !find out which basis is desired 
      CALL RDATAB(LOPEN,INCAR,IU5,'LOCALIZED_BASIS','=','#',';','S', &
           &            IDUM,RDUM,CDUM,LDUM,STRING,N,40,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LOCALIZED_BASIS'' from file INCAR.'
         LUSEMLWF=.TRUE.
      ENDIF
    
      !set correct Flags for basis choice
      CALL BASIS_FROM_INCAR(STRING,IU0, IU6)
  
      ! target Wannier states in constrained RPA
      IF ( NBANDS <= 0 ) THEN
          ALLOCATE(NTARGET_STATES(1000))
      ELSE
         ! read target states 
         ALLOCATE(NTARGET_STATES(NBANDS))
      ENDIF 
  
      NTARGET_STATES=1
      CALL RDATAB(LOPEN,INCAR,IU5,'NTARGET_STATES','=','#',';','I', &
           &            NTARGET_STATES,RDUM,CDUM,LDUM,CHARAC,N,NBANDS,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1.OR.N>NBANDS))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''NTARGET_STATES'' from file INCAR.'
      ENDIF
      CALL XML_INCAR_V('NTARGET_STATES','I',NTARGET_STATES, RDUM,CDUM,LDUM,CHARAC,N)
      IF ((LCRPA.OR.LNLRPA).AND.N==0) THEN
         CALL VTUTOR('W','TARGET_NOT_SET', &
        &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
         CALL VTUTOR('W','TARGET_NOT_SET', &
        &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
      ENDIF
  
      ! correct the entries, if user is too stupid 
      DO I=1,NBANDS
         IF(NTARGET_STATES(I)>1)  NTARGET_STATES(I)=1
         IF(NTARGET_STATES(I)<0)  NTARGET_STATES(I)=0      
      ENDDO

      ! full rpa is used
      LFRPA=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LFRPA','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LFRPA,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LFRPA'' from file INCAR.'
         LFRPA=.FALSE.
      ENDIF
      CALL XML_INCAR('LFRPA','L',IDUM,RDUM,CDUM,LFRPA,CHARAC,N)

      ! regularization of correlated projectors
      LREGULARIZE=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LREGULARIZE','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LREGULARIZE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LREGULARIZE'' from file INCAR.'
         LREGULARIZE=.TRUE.
      ENDIF
      CALL XML_INCAR('LREGULARIZE','L',IDUM,RDUM,CDUM,LREGULARIZE,CHARAC,N)

      ! include Drude term (by default)
      LDRUDE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LDRUDE','=','#',';','L', &
           &            IDUM,RDUM,CDUM,LDRUDE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''LDRUDE'' from file INCAR.'
         LDRUDE=.FALSE.
      ENDIF
      CALL XML_INCAR('LDRUDE','L',IDUM,RDUM,CDUM,LDRUDE,CHARAC,N)
  
      IF ( LCRPA .OR. LNLRPA .OR. L2E4W .OR. L2E4WALL ) THEN
         CALL PEAD_REQUEST

         !force downsampling of WAVECAR 
         IF ( .NOT. LDOWNSAMPLE ) THEN
            IF ( IU0>=0) WRITE(IU0,*)'Downsampling of WAVECAR forced by CRPA_READER'
            IF ( IU6>=0) WRITE(IU6,*)'Downsampling of WAVECAR forced by CRPA_READER'
            LDOWNSAMPLE=.TRUE.
         ENDIF 
         IF( LUSEMLWF ) THEN
            !switch on wannier90 ( default)
            LWANNIER90=.TRUE.
            LWANNIER90_RUN=.TRUE.
         ELSEIF ( LUSELCAO ) THEN
            !switch to LCAOs 
            LWANPROJ=.TRUE.
         ENDIF 
       
         ! lowest CRPA bands being represented in the localized basis must be given
         ! for standard CRPA calculations
         IF ( .NOT.(LDISENTANGLE .OR. LWEIGHTED .OR. NCRPAHIGH >0 )) THEN
           ! IF ( NCRPALOW <1  ) THEN
           !    CALL VTUTOR('E','NCRPALOW', &
           !   &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
           !    CALL VTUTOR('S','NCRPALOW', &
           !   &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
           ! ENDIF
         ENDIF
          
      ENDIF

      CLOSE(IU5)
   END SUBROUTINE CRPA_READER

!**********************************************************************
!
! small helper routine, set the basis for Coulomb integrals 
! and DMFT interaction matrix elements   
! 
!**********************************************************************

   SUBROUTINE BASIS_FROM_INCAR( STRING, IU0, IU6 )
      USE fock
      USE wnpr, ONLY: LWANPROJ
      CHARACTER (LEN=*) :: STRING
      INTEGER :: N, IU0, IU6
      INTEGER, EXTERNAL :: LENGTH
 
      CALL STRIP(STRING,N,'L')
      CALL LOWER(STRING)
      N=LENGTH(STRING)
 
      IF ( STRING(1:N)=='mlwf' ) THEN
         LUSELCAO = .FALSE.
         LUSEBLOCH = .FALSE.
         LUSEORTH_LCAOS = .FALSE.
         LUSEMLWF = .TRUE.
         IF( IU0 >= 0 ) WRITE(IU0,*)'Coulomb integrals will be evaluated in MLWF basis'
      ELSEIF( STRING(1:N)=='lcao' )THEN
         LUSELCAO = .TRUE.
         LUSEBLOCH = .FALSE.
         LUSEMLWF = .FALSE.
         IF(IU0>=0) WRITE(IU0,*)'Sorry LCAO not implemented yet..'
         STOP
         IF(LUSEORTH_LCAOS ) THEN
            IF(IU0>=0) WRITE(IU0,*)'Coulomb integrals will be evaluated in orthogonal LCAO basis'
         ELSE
            IF(IU0>=0) WRITE(IU0,*)'Coulomb integrals will be evaluated in LCAO basis'
         ENDIF 
      ELSEIF( STRING(1:N)=='bloch')THEN
         LUSEBLOCH = .TRUE.
         LUSEMLWF = .FALSE.
         LUSELCAO = .FALSE.
         LUSEORTH_LCAOS = .FALSE.
         IF( IU0 >= 0 ) WRITE(IU0,*)'Coulomb integrals will be evaluated in BLOCH basis'
      ELSE
         !tutor warning is only necessary if LCRPA or LNLRPA is chosen !!
         IF ( LCRPA .OR. LNLRPA ) THEN
            IF( IU0 >= 0 ) WRITE(IU0,*)'Coulomb integrals will be evaluated in MLWF basis'
         ENDIF 
         LUSELCAO = .FALSE.
         LUSEBLOCH = .FALSE.
         LUSEORTH_LCAOS = .FALSE.
         LUSEMLWF = .TRUE.
      ENDIF 
 
   ENDSUBROUTINE BASIS_FROM_INCAR 

!**********************************************************************
!
! set flags for calculation of effective interaction parameter U,U',J
!
!**********************************************************************

   SUBROUTINE CRPA_SETUP( WDES, W, W_CORR, LMDIM, T_INFO, P, CQIJ,&
      LATT_CUR, INFO, KPOINTS, NQ_IBZ, IO )  
      USE base
      USE pseudo
      USE wave_high
      USE mkpoints
      USE lattice
      USE ini
      USE constant
      USE poscar
      USE mlwf 
      USE full_kpoints
      USE fileio
      !needed for interpolation
      USE hamil_high
      USE mgrid
      USE nonl_high
      TYPE (wavedes)     WDES
      TYPE (wavespin)    W
      TYPE (wavespin)    W_CORR
      INTEGER            LMDIM
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      TYPE (latt)        LATT_CUR
      TYPE (info_struct) INFO
      TYPE (kpoints_struct) KPOINTS
      INTEGER            NQ_IBZ
      TYPE (in_struct)   IO
      !local
      INTEGER, ALLOCATABLE :: WTARGET(:)
      INTEGER              :: I, COUNTER, NK,ISP
    
      IF ( LNLRPA .OR. .NOT. LCRPA ) THEN
         LCHIC=.FALSE.
         RETURN
      ENDIF 

      !compute correlated polarizability?
      LCHIC=.FALSE. 

      !---------------------------------------------------------------------------------------
      IF ( LUSEBLOCH ) THEN  !use Bloch basis 
      !---------------------------------------------------------------------------------------
         ! if the interaction parameter are computed in the Bloch basis 
         ! we don't need Wannier functions
         ! number of basis elements is then the total # of bands 
         DMFT_num_wann = WDES%NB_TOT
        
         !set targetbands NCRPALOW and NCRPAHIGH if not full RPA is done
         IF ( .NOT. LFRPA ) THEN
            !write a warning if the INCAR input is overwritten
            IF ( NCRPALOW<= NCRPAHIGH .AND. NCRPAHIGH > 0 ) THEN
               IF( IO%IU0>=0) WRITE( IO%IU0,*)' Warning: CRPA bands will'//&
                  ' be overwritten by NTARGET_STATES'
            ENDIF 
 
            !find target states
            NCRPALOW = WDES%NB_TOT
            NCRPAHIGH = 1
            COUNTER=0
            DO I = 1, DMFT_num_wann
               IF ( NTARGET_STATES (I) == 1 ) THEN
                  NCRPALOW = MIN( NCRPALOW, I ) 
                  NCRPAHIGH = MAX( NCRPAHIGH, I ) 
               ENDIF
            ENDDO
            !relevant bands for correlated Green's function
            NCRPAMIN=NCRPALOW
            NCRPAMAX=NCRPAHIGH 
            IF( (NCRPALOW <= NCRPAHIGH .AND. NCRPALOW > 0) ) THEN
               IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A,I5," - ",I5)')' Bands neglected in CRPA '//&
                  'polarizability:',NCRPALOW,NCRPAHIGH
               IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A,I5," - ",I5)')' Bands neglected in CRPA '//&
                  'polarizability:',NCRPALOW,NCRPAHIGH
            ENDIF 
            IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(" Correlated subspace contains the Bloch bands: "'//&
               ',I4,"-",I4)')NCRPALOW,NCRPAHIGH
            IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(" Correlated subspace contains the Bloch bands: "'//&
               ',I4,"-",I4)')NCRPALOW,NCRPAHIGH
         ELSE
            IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(" Target bands: ")',ADVANCE="NO")
            IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(" Target bands: ")',ADVANCE="NO")
            !relevant bands for correlated Green's function
            NCRPAMIN=WDES%NB_TOT
            NCRPAMAX= 1 
            DO I = 1, DMFT_num_wann
               IF ( NTARGET_STATES (I) == 1 ) THEN
                  IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(I4)',ADVANCE="NO")I
                  IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(I4)',ADVANCE="NO")I
                  NCRPAMIN=MIN(NCRPAMIN,I)
                  NCRPAMAX=MAX(NCRPAMAX,I)
               ENDIF
            ENDDO
            IF ( IO%IU0>=0 ) WRITE(IO%IU0,*)
            IF ( IO%IU6>=0 ) WRITE(IO%IU6,*)
         ENDIF
      !---------------------------------------------------------------------------------------
      ELSE      ! use Wannier or LCAO basis 
      !---------------------------------------------------------------------------------------
         !otherwise we need Wannier functions 
         IF( LUSEMLWF ) THEN
             !first, look for an existing WANN file
             CALL READ_WANPROJ(WDES,IO)
             IF ( .NOT. LUFROMFILE ) THEN
                CALL MLWF_WANNIER90( WDES, W, P, CQIJ, T_INFO, LATT_CUR, INFO, IO)
             ENDIF 
             !set DMFT_num_wann to DMFT_num_wann
             DMFT_num_wann = MLWF_num_wann
         ELSEIF( LUSELCAO) THEN
!            IF ( LUSEORTH_LCAOS) THEN
!                CALL WNPR_PROJECT_ORTH( W, WDES, T_INFO, INFO, P, CQIJ, LATT_CUR, IO)
!            ELSE
                CALL WNPR_PROJECT( W, WDES, T_INFO, INFO, P, CQIJ, LATT_CUR, IO)
!            ENDIF
            DMFT_num_wann=WNPR_num_wann
         ENDIF 

         !consistency check, no targets present -> full RPA is done 
         IF ( SUM(NTARGET_STATES(1:DMFT_num_wann)) == 0 ) THEN
            LFRPA=.TRUE.
            IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')' no targets chosen: fully screened RPA forced!'
         ENDIF

         !maybe some bands are removed completely (highest priority)
         IF( NCRPALOW <= NCRPAHIGH .AND. NCRPALOW > 0 ) THEN
            IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A,I5," - ",I5)')' Bands neglected in CRPA '//&
               'polarizability:',NCRPALOW,NCRPAHIGH
            IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A,I5," - ",I5)')' Bands neglected in CRPA '//&
               'polarizability:',NCRPALOW,NCRPAHIGH
            !relevant bands for correlated Green's function
            NCRPAMIN=NCRPALOW
            NCRPAMAX=NCRPAHIGH 
         ELSE
            IF (LWEIGHTED .AND. .NOT. LFRPA) THEN
               !this computes Friedrich and Shih's CRPA weights
               CALL DETERMINE_CRPA_WEIGHTS( WDES, W, NQ_IBZ, IO ) 
               IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')' CRPA polarizability determined '//&
                  'using weighted approach'
               IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A)')' CRPA polarizability determined '//&
                  'using weighted approach'
               !at this stage CRPAWEIGHTS are allocated only -> weighted CRPA method
            ELSE
               !otherwise disentanglement or separation is done.
               !additional wavefunction is necessary in this case
               CALL ALLOCW( W%WDES, W_CORR)
               W_CORR%CW    =W%CW
               W_CORR%CPROJ =W%CPROJ 
               W_CORR%CELTOT=W%CELTOT
               W_CORR%FERTOT=W%FERTOT
               W_CORR%AUXTOT=W%AUXTOT
               LALLOCATED_WCORR = .TRUE. 
               IF ( LDISENTANGLE ) THEN 
                  !disentangle the wavefunction using Miyake's disentanglement approach
                  CALL DISENTANGLE( WDES, W_CORR, LMDIM, CQIJ, LATT_CUR, & 
                     INFO, KPOINTS, NQ_IBZ, IO )  
                  IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')' CRPA polarizability determined '//&
                     'using disentanglement approach'
                  IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A)')' CRPA polarizability determined '//&
                     'using disentanglement approach'
               ELSEIF ( (.NOT.LFRPA) .OR. LNLRPA ) THEN
                  !CRPA as it should be done: remove correlated polarizability from full polarizability
#ifdef old_version
                  !replace Bloch states by projected Wannier states
                  CALL SEPARATE_STATES( WDES, W_ALL , W_CORR, LMDIM, CQIJ, LATT_CUR, & 
                     INFO, KPOINTS, NQ_IBZ, IO )  
#else
                  !use numerically more stable version. 
                  CALL SEPARATE_CORRELATED_STATES( WDES, W , W_CORR, LMDIM, CQIJ, LATT_CUR, &
                     INFO, KPOINTS, NQ_IBZ, IO )  
#endif

                  IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')' CRPA polarizability determined '//&
                     'using Kubo formalism'
                  IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A)')' CRPA polarizability determined '//&
                     'using Kubo formalism'
                  !compute correlated polarizability only for CRPA 
                  LCHIC = (LCRPA .AND. (.NOT. LNLRPA))
               ENDIF 
            ENDIF
         ENDIF
         !print information about correlated subspace
         CALL PRINT_CSPACE_INFO(IO%IU0,IO%IU6)
      !---------------------------------------------------------------------------------------
      ENDIF   ! basis choice 
      !---------------------------------------------------------------------------------------
 
      !maybe RPA is done only
      IF( LFRPA .OR. LNLRPA ) THEN 
         !maybe CRPA_WEIGHTS are allocated -> get rid of it 
         IF ( ALLOCATED(CRPA_WEIGHTS) ) DEALLOCATE(CRPA_WEIGHTS)
         !in case of disentangled RPA we need this
         IF ( ALLOCATED(LCRPABAND) ) LCRPABAND=.FALSE.
         IF ( ALLOCATED(UCRPA) .AND. .NOT. (LDISENTANGLE .OR. LNLRPA) ) DEALLOCATE(UCRPA) 
         IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')' RPA polarizability is used '
         IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A)')' RPA polarizability is used '
      ENDIF 
      
      !stop here if Wannier projection matrices are only written
      IF ( LCRPAPLOT ) THEN
         CALL STOP_XML
         IF ( IO%IU0>=0 ) WRITE(IO%IU0,'(A)')'U.WAN written, proceed with plotting bands'
         IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(A)')'U.WAN written, proceed with plotting bands'
         CALLMPI_C(M_exit())
      ENDIF 
   ENDSUBROUTINE CRPA_SETUP

!**********************************************************************
!
! disentanglement of W into correlated and non-correlated part 
! using Miyake's approch from PRB. 80, 155134 (2009) 
! 
!**********************************************************************

   SUBROUTINE DISENTANGLE( WDES, W , LMDIM, CQIJ, LATT_CUR, & 
      INFO, KPOINTS, NQ_IBZ, IO )  
      USE base
      USE wave_high
      USE full_kpoints
      USE lattice
      USE constant
      USE mlwf!, ONLY : U_matrix,U_matrix_opt,lexclude_band,lwindow,kpt_latt,num_kpts
      USE wnpr
      USE choleski
      USE m_mrgrnk 
      USE mkpoints 
      USE ini
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      INTEGER LMDIM
      OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      TYPE (kpoints_struct) KPOINTS
      INTEGER NQ_IBZ
      TYPE(in_struct) IO
      ! local variables
      TYPE(wavedes1) WDES1
      TYPE(wavefuna) WA
      INTEGER I,J,K,IP
      INTEGER ISP,NK
      COMPLEX(q), ALLOCATABLE :: NEWEIG(:,:,:)
      COMPLEX(q), ALLOCATABLE :: NEWEIGC(:,:,:)
      REAL(q), ALLOCATABLE ::  COREIG(:)  !correlated eigenvalues
      REAL(q), ALLOCATABLE :: NCOREIG(:)  !noncorrelated
      REAL(q), ALLOCATABLE :: ALLEIG(:)  !disentangled eigenvalues 
      GDEF, ALLOCATABLE :: U(:,:)
      GDEF, ALLOCATABLE :: UNCORR(:,:)
      GDEF, ALLOCATABLE :: UPLOT(:,:)
      INTEGER,ALLOCATABLE:: NSORT(:)
      INTEGER :: NROWMIN1, NROWMAX1
      INTEGER :: NCOLMIN1, NCOLMAX1
      INTEGER :: NMIN, NMAX, IDAT(2)
     
      CALL CHECK_FULL_KPOINTS
  
      !consistency check: Disentanglement needs a minimial basis 
      !that is: the Wannier basis should not be larger than the correlated sub space
      J = 0
      DO I = 1, DMFT_num_wann
         IF ( NTARGET_STATES(I) == 1 ) J = J+1
      ENDDO 
      IDAT(1)=DMFT_num_wann
      IDAT(2)=J
      IF ( J /= DMFT_num_wann ) THEN
         CALL VTUTOR('E','DISENTANGLE', &
        &               0.0_q,1,IDAT,2,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
         CALL VTUTOR('S','DISENTANGLE', &
        &               0.0_q,1,IDAT,2,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
      ENDIF 

      !allocate wannier projection matrices
      ALLOCATE(U(WDES%NB_TOT,WDES%NB_TOT))
      ALLOCATE(UNCORR(WDES%NB_TOT,WDES%NB_TOT))
      ALLOCATE(UPLOT(WDES%NB_TOT,WDES%NB_TOT))
  
      !nullify CRPA projector and reallocate it
      IF ( ALLOCATED ( UCRPA ) ) DEALLOCATE(UCRPA) 
      ALLOCATE(UCRPA(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS, WDES%ISPIN) )
      UCRPA = 0
  
      !new eigenvalues
      ALLOCATE(NEWEIG(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      NEWEIG=zero
      ALLOCATE(NEWEIGC(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      NEWEIGC=zero
  
      !entangled and disentangled eigenvalues 
      ALLOCATE(ALLEIG(WDES%NB_TOT))    
      ALLOCATE(COREIG(WDES%NB_TOT))    
      ALLOCATE(NCOREIG(WDES%NB_TOT))    
  
      !sorting array 
      ALLOCATE( NSORT( WDES%NB_TOT ) ) 
      IF (ALLOCATED(LCRPABAND)) DEALLOCATE(LCRPABAND)
      ALLOCATE( LCRPABAND( WDES%ISPIN, WDES%NKPTS, WDES%NB_TOT  ) )
      LCRPABAND=.FALSE.
  
      !find largest set of bloch bands considered 
      NROWMIN1 = WDES%NB_TOT
      NROWMAX1 = 1 
      NCOLMIN1 = WDES%NB_TOT
      NCOLMAX1 = 1 
      NMIN=WDES%NB_TOT
      NMAX=1
      ! find sub block of Wannier functions
      DO ISP=1,WDES%ISPIN
         ksearch: DO NK=1,WDES%NKPTS
            !determine local projector U
            CALL DETERMINE_WANNIER_PROJECTOR(WDES,NK,ISP,UCRPA(:,:,NK,ISP),IO%IU0)
            !----------------------------------------------------------------------  
            !construct overlap
            U=zero
            DO I=1,WDES%NB_TOT
               DO J=1,WDES%NB_TOT
                  DO K = 1, WDES%NB_TOT
                     U(I,J) = U(I,J) + UCRPA(I,K,NK,ISP)*GCONJG(UCRPA(J,K,NK,ISP))
                  ENDDO 
               ENDDO
            ENDDO
            !extract nonzero DMFT_num_wann x DMFT_num_wann block
            !find largest set of bands considered 
            DO I = 1, WDES%NB_TOT
               DO  J = 1, WDES%NB_TOT
                  IF ( ABS( U(I,J)) > TINY ) THEN
                      NROWMIN1 = MIN( I, NROWMIN1 )
                      NROWMAX1 = MAX( I, NROWMAX1 )
                  ENDIF
               ENDDO
            ENDDO 
            NMIN=MIN(NMIN,NROWMIN1)
            NMAX=MAX(NMAX,NROWMAX1)
         ENDDO ksearch
      ENDDO
      !block not found if NROWMAX1 - NROWMIN1 + 1 /= DMFT_num_wann
      IF (  NROWMAX1-NROWMIN1+1< DMFT_num_wann ) THEN
         IF ( IO%IU0>=0 ) WRITE(IO%IU0,*)'internal error in VASP from DISENTANGLE: sub block not found'
         STOP
      ENDIF 

      !relevant bands for correlated Green's function
      NCRPAMIN=NROWMIN1 
      NCRPAMAX=NROWMAX1 

      !----------------------------------------------------------------------------------------
      ! loop over spin and all k-points
      spin: DO ISP=1,WDES%ISPIN
      !----------------------------------------------------------------------------------------
         kpts: DO NK=1,WDES%NKPTS
         !-------------------------------------------------------------------------------------
            !determine local projector U
            CALL DETERMINE_WANNIER_PROJECTOR(WDES,NK,ISP,UCRPA(:,:,NK,ISP),IO%IU0)
            !to plot correlated states 
            IF ( LCRPAPLOT ) UPLOT(:,:) = UCRPA(:,:,NK,ISP)
         
            !orthogonalize correlated sub space
            CALL DIAGONALIZE_CORR_SYSTEM(W,UCRPA(:,:,NK,ISP),DMFT_num_wann,COREIG,NK,ISP,.TRUE.)
            ! diagonalizing the Hamiltonian for each k-point 
            ! destroys the maximally localized property of the Wannier functions
            ! also it changes the symmetry of the band structure. 
            U(:,:)=UCRPA(:,:,NK,ISP)
   
            !------------------------------------------------------------------------------
            !determine non-correlated eigenvalues 
            !UNCORR is the projector onto non-correlated system
            UNCORR=zero 
            DO I = 1, WDES%NB_TOT
               DO K = 1, WDES%NB_TOT
                  DO J = 1, DMFT_num_wann
                     UNCORR(K,I)=UNCORR(K,I)-GCONJG(U(I,J))*U(K,J)
                  ENDDO         
               ENDDO              
            ENDDO          
            DO I=1,WDES%NB_TOT
               UNCORR(I,I)= UNCORR(I,I) + 1._q
            ENDDO

            CALL DIAGONALIZE_NONCORR_SYSTEM(W,UNCORR,WDES%NB_TOT-DMFT_num_wann,NCOREIG,NK,NROWMIN1,NROWMAX1,ISP)
            !backup UNCORR for chi_base
            IF ( LPLOTDIS ) UPLOT(:,:) = UNCORR(:,:)

            !rearrange 
            !------|---------|----------------|
            ! nc   1   cor   2     nc         3
            ! nc  - non-correlated 
            ! cor - correlated 
            ALLEIG(1:NROWMIN1-1)=NCOREIG(1:NROWMIN1-1)
            ALLEIG(NROWMIN1:NROWMIN1+DMFT_num_wann-1)=COREIG(1:DMFT_num_wann)
            ALLEIG(NROWMIN1+DMFT_num_wann:WDES%NB_TOT)=NCOREIG(NROWMIN1:WDES%NB_TOT-DMFT_num_wann)
            !also projectors have to be set up accordingly
            !put the correlated block to 1-2
            U=UNCORR
            DO I = 1, WDES%NB_TOT
               U(I,NROWMIN1:NROWMIN1+DMFT_num_wann-1)=UCRPA(I,1:DMFT_num_wann,NK,ISP)
            ENDDO
            !shift the correlated block to 1-2
            UNCORR(:,:)=UCRPA(:,:,NK,ISP)
            UCRPA(:,:,NK,ISP)=zero
            DO I = 1, WDES%NB_TOT
               UCRPA(I,NROWMIN1:NROWMIN1+DMFT_num_wann-1,NK,ISP)=UNCORR(I,1:DMFT_num_wann)
            ENDDO
            UNCORR=U
             
            !UNCORR ->  corr + non-corr
            !UCRPA ->  corr only 
            !----------------------------------------------------------------------
            !sort eigenvalues 
            CALL MRGRNK(ALLEIG,NSORT) 
            !determine sorting matrix
            !U gives sorted eigenvalues 
            U=zero
            DO I=1,WDES%NB_TOT
               IP=NSORT(I)
               U(IP,I) = 1._q
               !identify correlated states for later purpose
               IF ( NROWMIN1 <= IP .AND. IP <= NROWMIN1+DMFT_num_wann-1 ) THEN
                  LCRPABAND(ISP,NK,I)=.TRUE.
               ENDIF
            ENDDO 
            !all new disentangled eigenvalues are sorted and stored in NEWEIG,
            !correlated ones are in NEWEIGC
            NEWEIG(:,NK,ISP)=zero
            NEWEIGC(:,NK,ISP)=zero
            DO I = 1,WDES%NB_TOT
               DO J=1,WDES%NB_TOT
                  NEWEIG(I,NK,ISP)=NEWEIG(I,NK,ISP)+U(J,I)*ALLEIG(J)
               IF ( J >= NROWMIN1 .AND. J <= NROWMIN1+DMFT_num_wann-1 ) THEN 
                     NEWEIGC(I,NK,ISP)=NEWEIGC(I,NK,ISP)+U(J,I)*ALLEIG(J)
                  ENDIF 
               ENDDO
            ENDDO
            !for MCRPA we need to rotate the HEAD appropriately
            UCRPA(:,:,NK,ISP)=zero
            DO I = 1,WDES%NB_TOT
               DO J=1,WDES%NB_TOT
                  DO K = 1, WDES%NB_TOT 
                     UCRPA(I,J, NK,ISP) = UCRPA(I,J,NK,ISP)+UNCORR(I,K)*U(K,J)
                  ENDDO
              ENDDO
            ENDDO
#ifdef debug
WRITE(500,*)'all NK=',NK
WRITE(500,'(16F12.6)')REAL(NEWEIG(:,NK,ISP),q)
DO I = 1, 16
WRITE(500,'(16F12.6)')REAL(UCRPA(I,1:16,NK,ISP),q)
ENDDO
#endif

            !first rotate correlated + noncorrelated function
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(W,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix UCRPA
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,UCRPA(1,1,NK,ISP), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
  
            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF

#ifdef debug
            !first rotate correlated + noncorrelated function
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(W,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix UCRPA
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,UNCORR(1,1), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
  
            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
            !sort
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(W,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,U(1,1), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
  
            !for MCRPA we need to rotate the HEAD appropriately
            UCRPA(:,:,NK,ISP)=zero
            DO I = 1,WDES%NB_TOT
               DO J=1,WDES%NB_TOT
                  DO K = 1, WDES%NB_TOT 
                     UCRPA(I,J, NK,ISP) = UCRPA(I,J,NK,ISP)+UNCORR(I,K)*U(K,J)
                  ENDDO
              ENDDO
            ENDDO
#endif
            !maybe we want to plot the non-correlated disentangled bands
            IF (LPLOTDIS ) UCRPA(:,:,NK,ISP)=UPLOT(:,:)
!!TEST  non-correlated states should determine head
!            !for MCRPA we need to rotate the HEAD appropriately
!            UCRPA(:,:,NK,ISP)=zero
!            DO I = 1,WDES%NB_TOT
!               DO J=1,WDES%NB_TOT
!                  DO K = 1, WDES%NB_TOT 
!                     UCRPA(I,J, NK,ISP) = UCRPA(I,J,NK,ISP)+UPLOT(I,K)*U(K,J)
!                  ENDDO
!              ENDDO
!            ENDDO
!!TEST
!WRITE(600,*)'all NK=',NK
#ifdef debug
WRITE(600,'(16F12.6)')REAL(NEWEIG(:,NK,ISP),q)
DO I = 1, 16
WRITE(600,'(16F12.6)')REAL(UCRPA(I,1:16,NK,ISP),q)
ENDDO
#endif
        !--------------------------------------------------------------------------------------
         ENDDO kpts
      !----------------------------------------------------------------------------------------
      ENDDO spin
      !----------------------------------------------------------------------------------------
      DEALLOCATE(U)
      DEALLOCATE(UNCORR,ALLEIG,COREIG,NCOREIG)

      !backup old eigenvalues  
      NEWEIGC(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN) = &
         W%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)
      
      !last but not least save eigenvalues 
      W%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)= &
         NEWEIG(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)
  
#ifdef debug
      !check S-orthogonality 
      CALL DUMP_S( WDES, W, INFO%LOVERL, LMDIM, CQIJ ) 
#endif 
      DO ISP = 1,WDES%ISPIN
         IF (WDES%ISPIN==2) WRITE(IO%IU6,'(/A,I1)') ' spin component ',ISP
         DO NK = 1,WDES%NKPTS
             IF ( IO%NWRITE < 3 .AND. NK > NQ_IBZ ) CYCLE
             IF( IO%IU6>=0 ) THEN
                 WRITE(IO%IU6,2202)NK,WDES%VKPT(:,NK), &
                 (I,REAL(NEWEIGC(I,NK,ISP),q),REAL( W%CELTOT(I,NK,ISP),KIND=q), &
                 W%FERTOT(I,NK,ISP)*WDES%RSPIN, &
                 LCRPABAND(ISP,NK,I), I=NMIN,NMAX )
             ENDIF
         ENDDO
      ENDDO
  
      DEALLOCATE(NEWEIG)
      DEALLOCATE(NEWEIGC)
  
      !dump wannier matrices for ploting later on
      IF ( LCRPAPLOT ) THEN
         IF(IO%IU0>=0)THEN
            ! write rotation matrices
            OPEN(UNIT=199,FILE='U.WAN',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
            WRITE(199) WDES%NB_TOT
            WRITE(199) UCRPA
            CLOSE(199)       
         ENDIF  
      ENDIF 
  
      ! maybe the disentangled full RPA is wanted, 
      ! in this case we kill LCRPABAND,
      ! that all states are included in screending 
      IF ( LFRPA ) THEN
         IF ( IO%IU0>=0) WRITE(IO%IU0,'(A)')' Disentangled fully screened interaction chosen'
         DEALLOCATE(LCRPABAND)
      ENDIF  

2202  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
         &    '  band No.  old energies   dis. energies     occupation   cor. state?'/ &
         &    (3X,I4,3X,F10.4,5X,F10.4,7X,F10.5,3X,L4,4X))
  
      RETURN
   ENDSUBROUTINE DISENTANGLE

!**********************************************************************
! 
! This Routine separates correlated states from noncorrelated states. 
! First, the original Bloch states (W) are replaced by the eigensystem of 
! the Wannier Hamiltonian. This is necessary, since the projection is 
! not perfect for surjective maps from Bloch to the Wannier space.
! Second, the correlated states (WC) are determined using the regularized
! correlated projector. 
! 
!**********************************************************************

   SUBROUTINE SEPARATE_STATES( WDES, W , WC, LMDIM, CQIJ, LATT_CUR, & 
      INFO, KPOINTS, NQ_IBZ, IO )  
      USE base
      USE wave_high
      USE full_kpoints
      USE lattice
      USE constant
      USE mlwf!, ONLY : U_matrix,U_matrix_opt,lexclude_band,lwindow,kpt_latt,num_kpts
      USE wnpr
      USE choleski
      USE m_mrgrnk 
      USE mkpoints 
      USE ini
!      USE lie 
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      TYPE(wavespin) WC
      INTEGER LMDIM
      OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      TYPE (kpoints_struct) KPOINTS
      INTEGER NQ_IBZ
      TYPE(in_struct) IO
      ! local variables
      TYPE(wavedes1) WDES1
      TYPE(wavefuna) WA
      INTEGER I,J,K,IP
      INTEGER ISP,NK
      COMPLEX(q), ALLOCATABLE :: NEWEIG(:,:,:)
      REAL(q), ALLOCATABLE ::  COREIG(:)  !correlated eigenvalues
      GDEF, ALLOCATABLE :: WINDOWFUN(:,:,:)
      GDEF, ALLOCATABLE :: U(:,:)
      GDEF, ALLOCATABLE :: UPLOT(:,:)
      INTEGER :: NMIN, NMAX
      INTEGER :: NMIN1, NMAX1
      GDEF, ALLOCATABLE :: D(:,:,:)
      INTEGER :: NK_IN_KPOINTS_FULL_ORIG
     
      CALL CHECK_FULL_KPOINTS
  
      !allocate wannier projection matrices
      ALLOCATE(U(WDES%NB_TOT,WDES%NB_TOT))
      ALLOCATE(UPLOT(WDES%NB_TOT,WDES%NB_TOT))
  
      !nullify CRPA projector and reallocate it
      IF ( ALLOCATED ( UCRPA ) ) DEALLOCATE(UCRPA) 
      ALLOCATE(UCRPA(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS, WDES%ISPIN) )
      UCRPA = zero
      IF ( ALLOCATED ( UCDER ) ) DEALLOCATE(UCDER) 
      ALLOCATE(UCDER(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS, WDES%ISPIN) )
      UCDER = zero
      !nullify weight and allocate the appropriate memory for the weighting factor
      IF (ALLOCATED(CRPA_WEIGHTS)) DEALLOCATE(CRPA_WEIGHTS)
      ALLOCATE( CRPA_WEIGHTS(WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN ) )
  
      !new eigenvalues
      ALLOCATE(NEWEIG(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      NEWEIG=zero
    
      !window function will be necessary
      ALLOCATE(WINDOWFUN(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      WINDOWFUN=zero

      !for jacobi diagonalization
      ALLOCATE(D(WDES%NB_TOT,WDES%NB_TOT,WDES%ISPIN))
  
      ! to be sure if NCRPALOW is set
      IF ( NCRPALOW <1  ) THEN
         CALL VTUTOR('E','NCRPALOW', &
        &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
         CALL VTUTOR('S','NCRPALOW', &
        &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
      ENDIF
      NMIN=NCRPALOW
      NMAX=NMIN+DMFT_num_wann-1

      NMIN1=WDES%NB_TOT
      NMAX1=1
      ALLOCATE(COREIG(WDES%NB_TOT))    
  
      !----------------------------------------------------------------------------------------
      ! loop over spin and all k-points
      spin: DO ISP=1,WDES%ISPIN
      !----------------------------------------------------------------------------------------
         kpts: DO NK=1,WDES%NKPTS
         !-------------------------------------------------------------------------------------
            !initialize the identity matrix
            DO I = 1, WDES%NB_TOT
                UCRPA(I,I,NK,ISP)=one
            ENDDO
            !determine local projector U
            CALL DETERMINE_WANNIER_PROJECTOR(WDES,NK,ISP,U(:,:),IO%IU0)
            !replace the bands represented by the localized basis set
            UCRPA(1:WDES%NB_TOT,NMIN:NMAX,NK,ISP)=U(1:WDES%NB_TOT,1:DMFT_num_wann)
            IF ( LCRPAPLOT ) THEN
               UPLOT = zero
               UPLOT(:,:) = UCRPA(:,:,NK,ISP)
            ENDIF

            !orthogonalize correlated sub space
            CALL ORTHOGONALIZE_SYSTEM(W,UCRPA(:,:,NK,ISP),WDES%NB_TOT,COREIG,NK,ISP)
            !backup rotation matrix 
            U(:,:)=UCRPA(:,:,NK,ISP)
            !new eigenvalues
            NEWEIG(1:WDES%NB_TOT,NK,ISP)=COREIG(1:WDES%NB_TOT)

            !backup for CDER rotation
            !the roation of the CDER needs to be done on the full mesh unfortunately  
            UCDER(:,:,NK,ISP)=UCRPA(:,:,NK,ISP)

            !first rotate all states using U
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(W,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix U
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,U(1,1), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
  
            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF

            !determine Wannier projector
            CALL DETERMINE_WANNIER_PROJECTOR( WDES, NK, ISP, U, IO%IU0)
            !corresponding correlated projector
            CALL DETERMINE_CORRELATED_PROJECTOR(WDES%NB_TOT,U,UCRPA(:,:,NK,ISP),NK,ISP,IO%IU0)
            !save the correlated probabilities
            DO I=1, WDES%NB_TOT
               CRPA_WEIGHTS(I,NK,ISP)=REAL(UCRPA(I,I,NK,ISP),q)
               IF ( ABS(CRPA_WEIGHTS(I,NK,ISP))>TINYLARGE) THEN
                  NMIN1=MIN(NMIN1,I)
                  NMAX1=MAX(NMAX1,I)
               ENDIF
            ENDDO
            !diagonalize the Hamiltonian using Givens rotations (Jacobi rotation) 
            CALL JOINT_DIAGONALIZER(D,UCRPA,INFO%EDIFF,WDES%NB_TOT,NK,WDES%ISPIN,&
               IO%IU0,IO%IU6,MAXLIE,.TRUE.)
            !the diagonalized projector will contain only 1 and 0 entries 
            !This defines the Window function
            DO I=1,WDES%NB_TOT
               WINDOWFUN(I,NK,ISP)=UCRPA(I,I,NK,ISP)
               CRPA_WEIGHTS(I,NK,ISP)=WINDOWFUN(I,NK,ISP)
            ENDDO
            CALL DETERMINE_CORRELATED_PROJECTOR(WDES%NB_TOT,U,UCRPA(:,:,NK,ISP),NK,ISP,IO%IU0,&
               WINDOWFUN)

            !rotate correlated function
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(WC,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, WC%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, WC%CW   (1,1,NK,ISP))
            ENDIF
            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix U
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,UCRPA(1,1,NK,ISP), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
  
            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, WC%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, WC%CW   (1,1,NK,ISP))
            ENDIF
  
        !--------------------------------------------------------------------------------------
         ENDDO kpts
      !----------------------------------------------------------------------------------------
      ENDDO spin
      !----------------------------------------------------------------------------------------
      DEALLOCATE(U)
 
      !correlated part has updated eigenvalues
      WC%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)= &
         NEWEIG(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)
      !backup old eigenvalues  
      NEWEIG(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN) = &
         W%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)
      !and update old eigenvalues
      W%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)= &
         WC%CELTOT(1:WDES%NB_TOT,1:WDES%NKPTS,1:WDES%ISPIN)
 
#ifdef debug
      !check S-orthogonality 
      CALL DUMP_S( WDES, W, INFO%LOVERL, LMDIM, CQIJ ) 
#endif 
      !update occupancies of correlated bands
      IF ( IO%IU6>=0 ) WRITE(IO%IU6,'(/A)')'Original and rotated energies:'
      DO ISP = 1,WDES%ISPIN
         IF (WDES%ISPIN==2) WRITE(IO%IU6,'(/A,I1)') ' spin component ',ISP
         DO NK = 1,WDES%NKPTS
             IF ( IO%NWRITE < 3 .AND. NK > NQ_IBZ ) CYCLE
             IF( IO%IU6>=0 ) THEN
                 WRITE(IO%IU6,2202)NK,WDES%VKPT(:,NK), &
                 (I,REAL(NEWEIG(I,NK,ISP),q),REAL( W%CELTOT(I,NK,ISP),KIND=q), &
                 ABS(W%CELTOT(I,NK,ISP)-NEWEIG(I,NK,ISP)),CRPA_WEIGHTS(I,NK,ISP),&
                 REAL(WINDOWFUN(I,NK,ISP),q),I=NMIN1,NMAX1)
             ENDIF
         ENDDO
      ENDDO
  
      DEALLOCATE(NEWEIG)
      DEALLOCATE(D,WINDOWFUN)
 
      IF (ALLOCATED(CRPA_WEIGHTS)) DEALLOCATE(CRPA_WEIGHTS)

      !only for NLRPA UCDER is kept
      IF (.NOT. LNLRPA ) DEALLOCATE(UCDER)

      ! maybe the disentangled full RPA is wanted, 
      ! in this case we kill LCRPABAND,
      ! that all states are included in screending 
      IF ( LFRPA ) THEN
         IF ( IO%IU0>=0) WRITE(IO%IU0,'(A)')' rotated fully screened interaction chosen'
      ENDIF  
  
2202  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
         &    '  band No.  original   rotated      error  weights  window'/ &
         &    (I10,F10.4,F10.4,F11.4,F9.3,F8.3))
      RETURN
   ENDSUBROUTINE SEPARATE_STATES 

!**********************************************************************
!
! separates correlated states from non-correlated states by using
! regularized correlated projectors at each k-point.
! the original wavefunction is untouched. 
! this algorithm is quite stable if the projection of the Bloch to 
! Wannier state is surjective.
! 
!**********************************************************************

   SUBROUTINE SEPARATE_CORRELATED_STATES( WDES, W , WC, LMDIM, CQIJ, LATT_CUR, & 
       INFO, KPOINTS, NQ_IBZ, IO )  
      USE base
      USE wave_high
      USE full_kpoints
      USE lattice
      USE constant
      USE mlwf!, ONLY : U_matrix,U_matrix_opt,lexclude_band,lwindow,kpt_latt,num_kpts
      USE wnpr
      USE choleski
      USE m_mrgrnk 
      USE mkpoints 
!      USE lie
      USE ini
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      TYPE(wavespin) WC
      INTEGER LMDIM
      OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      TYPE (kpoints_struct) KPOINTS
      INTEGER         NQ_IBZ
      TYPE(in_struct) IO
      ! local variables
      TYPE(wavedes1) WDES1
      TYPE(wavefuna) WA
      INTEGER I 
      INTEGER ISP,NK
      GDEF, ALLOCATABLE :: U(:,:)
      GDEF, ALLOCATABLE :: WINDOWFUN(:,:,:)
      GDEF, ALLOCATABLE :: D(:,:,:)
      INTEGER :: NMIN, NMAX
     
      CALL CHECK_FULL_KPOINTS
  
      !allocate wannier projection matrices
      ALLOCATE(U(WDES%NB_TOT,WDES%NB_TOT))
      ALLOCATE(WINDOWFUN(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      WINDOWFUN=zero
  
      !nullify CRPA projector and reallocate it
      IF ( ALLOCATED ( UCRPA ) ) DEALLOCATE(UCRPA) 
      ALLOCATE(UCRPA(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS, WDES%ISPIN) )
      UCRPA = 0
      !nullify weight and allocate the appropriate memory for the weighting factor
      IF (ALLOCATED(CRPA_WEIGHTS)) DEALLOCATE(CRPA_WEIGHTS)
      ALLOCATE( CRPA_WEIGHTS(WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN ) )
  
#ifdef debug
      CALL FIND_ON_LIE_MANIFOLD(UCRPA(:,:,1,1),4,'MIN',NTARGET_STATES,INFO%EDIFF,MAXLIE,1,IO,&
         .FALSE.,.TRUE.) 
      STOP
#endif 
  
      ALLOCATE(D(WDES%NB_TOT,WDES%NB_TOT,WDES%ISPIN))
  
      NMIN=WDES%NB_TOT
      NMAX=1
      DO ISP=1,WDES%ISPIN
         DO NK=1, WDES%NKPTS
            !determine Wannier projector
            CALL DETERMINE_WANNIER_PROJECTOR( WDES, NK, ISP, U, IO%IU0)
            !corresponding correlated projector
            CALL DETERMINE_CORRELATED_PROJECTOR(WDES%NB_TOT,U,UCRPA(:,:,NK,ISP),NK,ISP,IO%IU0)
            !save the correlated probabilities
            DO I=1, WDES%NB_TOT
               CRPA_WEIGHTS(I,NK,ISP)=REAL(UCRPA(I,I,NK,ISP),q)
            ENDDO
            !diagonalize the Hamiltonian using Givens rotations (Jacobi rotation) 
            CALL JOINT_DIAGONALIZER(D,UCRPA,INFO%EDIFF,WDES%NB_TOT,NK,WDES%ISPIN,&
               IO%IU0,IO%IU6,MAXLIE,.TRUE.)
            !the diagonalized projector will contain only 1 and 0 entries 
            !This defines the Window function
            DO I=1,WDES%NB_TOT
               WINDOWFUN(I,NK,ISP)=UCRPA(I,I,NK,ISP)
               CRPA_WEIGHTS(I,NK,ISP)=WINDOWFUN(I,NK,ISP)
               IF ( ABS(WINDOWFUN(I,NK,ISP)) > 0 )THEN
                  NMIN=MIN(NMIN,I)
                  NMAX=MAX(NMAX,I)
               ENDIF
            ENDDO
            !determine again the correlated projector, this time multiplied with the Window function
            CALL DETERMINE_CORRELATED_PROJECTOR(WDES%NB_TOT,U,UCRPA(:,:,NK,ISP),NK,ISP,IO%IU0,&
               WINDOWFUN)
  
            !rotate the Wavefunction 
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(WC,WDES1,ISP)
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, WC%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, WC%CW   (1,1,NK,ISP))
            ENDIF
            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix U
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,UCRPA(1,1,NK,ISP), &
             &   WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
             &   WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
             &   WA%CW_RED,WA%CPROJ_RED)
            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, WC%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, WC%CW   (1,1,NK,ISP))
            ENDIF
         ENDDO
      ENDDO
      !get rid of auxillary matrix 
      DEALLOCATE(U)

      !relevant bands for correlated Green's function
      NCRPAMIN=NMIN 
      NCRPAMAX=NMAX 

      !dump some information 
      DO ISP = 1,WDES%ISPIN
         IF (WDES%ISPIN==2 .AND. IO%IU6>= 0 ) WRITE(IO%IU6,'(/A,I1)') ' spin component ',ISP
         DO NK = 1,WDES%NKPTS
             !per default print only irreducible wedge of Brillouin zone
             IF ( IO%NWRITE < 3 .AND. NK > NQ_IBZ ) CYCLE
             IF( IO%IU6>=0 ) THEN
                 WRITE(IO%IU6,2202)NK,WDES%VKPT(:,NK), &
                 (I,REAL(W%CELTOT(I,NK,ISP),q),REAL( WC%CELTOT(I,NK,ISP),KIND=q), &
                 REAL(WINDOWFUN(I,NK,ISP),q), CRPA_WEIGHTS(I,NK,ISP), &
                  I=NMIN,NMAX )
             ENDIF
         ENDDO
      ENDDO
  
      DEALLOCATE(CRPA_WEIGHTS)
  
2202  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
         &    '  band No.  old energies   rot. energies     windowfun    weights'/ &
         &    (3X,I4,3X,F10.4,5X,F10.4,7X,F10.5,3X,F10.5,4X))
  
      RETURN
   ENDSUBROUTINE SEPARATE_CORRELATED_STATES

!**********************************************************************
!
! determines CRPA_WEIGHTS for weighted approach of cRPA 
!
!**********************************************************************

   SUBROUTINE DETERMINE_CRPA_WEIGHTS( WDES, W, NQ_IBZ, IO ) 
      USE base 
      IMPLICIT NONE
      TYPE(wavedes)   WDES
      TYPE(wavespin)  W
      INTEGER         NQ_IBZ
      TYPE(in_struct) IO
      ! local variables
      INTEGER IU0
      INTEGER IU6
      INTEGER I 
      INTEGER ISP,NK
      INTEGER NMIN,NMAX
      GDEF, ALLOCATABLE    :: U(:,:)
      GDEF, ALLOCATABLE    :: UP(:,:)
      REAL(q) :: RN 
 
      CALL CHECK_FULL_KPOINTS
      IU0=IO%IU0
      IU6=IO%IU6
 
      !allocate memory for effective rotation matrices
      ALLOCATE(U(WDES%NB_TOT,WDES%NB_TOT))
      ALLOCATE(UP(WDES%NB_TOT,WDES%NB_TOT))
    
      !FCRPA 
      !nullify weight and allocate the appropriate memory for the weighting factor
      IF (ALLOCATED(CRPA_WEIGHTS)) DEALLOCATE(CRPA_WEIGHTS)
      ALLOCATE( CRPA_WEIGHTS(WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN ) )
 
      NMIN=WDES%NB_TOT
      NMAX=1
      !----------------------------------------------------------------------------------------
      spin: DO ISP=1,WDES%ISPIN  !spin
      !----------------------------------------------------------------------------------------
         kpoints: DO NK=1,WDES%NKPTS  !k-points
         !-------------------------------------------------------------------------------------
            !get the CRPA weights 
            CALL DETERMINE_WANNIER_PROJECTOR( WDES, NK, ISP, U, IU0)
            CALL DETERMINE_CORRELATED_PROJECTOR(WDES%NB_TOT,U,UP,NK,ISP,IU0)
  
            !now U is the rotation matrix at kpoint NK with spin ISP
            !finally calculate the weight for CRPA
            DO I = 1, WDES%NB_TOT
               CRPA_WEIGHTS(I,NK,ISP) = REAL(UP(I,I),q)
               IF ( REAL(UP(I,I),q)>TINYLARGE ) THEN
                  NMIN = MIN(NMIN,I)
                  NMAX = MAX(NMAX,I)
               ENDIF 
            ENDDO              
         !-------------------------------------------------------------------------------------
         ENDDO kpoints
      !----------------------------------------------------------------------------------------
      ENDDO spin
      !----------------------------------------------------------------------------------------
      DEALLOCATE(U)
      DEALLOCATE(UP)

      !set relevant bands for correlated Green's function
      NCRPAMIN=NMIN
      NCRPAMAX=NMAX

      !dump weights
      DO ISP=1,WDES%ISPIN  !spin
         IF (WDES%ISPIN==2) THEN
            IF ( IU6>=0 ) WRITE(IU6,'(/A,I1)') ' spin component ',ISP
         ENDIF 
         DO NK=1,WDES%NKPTS  !k-points
            !default output is the irreducible wedge of the BZ
            IF ( IO%NWRITE < 3 .AND. NK > NQ_IBZ ) CYCLE
            IF( IU6>=0  ) THEN
               WRITE(IU6,1000)NK,WDES%VKPT(:,NK), &
               (I,REAL( W%CELTOT(I,NK,ISP),q), CRPA_WEIGHTS(I,NK,ISP),&
               W%FERTOT(I,NK,ISP)*WDES%RSPIN, I=NMIN,NMAX)
            ENDIF
            !also # of states at each k-point sound be number of states J
            RN = SUM(CRPA_WEIGHTS(1:WDES%NB_TOT,NK,ISP))
            IF ( ABS(RN) < TINYLARGE ) THEN
               IF (IU0>=0) WRITE(IU0,'(A,E13.6,A,I3,A,2I4)')' WARNING: # of target states is ',RN,&
                  ' and not ',SUM(NTARGET_STATES(1:DMFT_num_wann)),' for k-point,ISP',NK,ISP
            ENDIF
         ENDDO
      ENDDO 
 
      RETURN
1000  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
      &         '  band No.     energies   CRPA-weights   occupation'/ &
      &           (3X,I4,3X,F10.4,3X,F12.4,3X,F10.5))
   END SUBROUTINE DETERMINE_CRPA_WEIGHTS
 
!**********************************************************************
!
! Determine Wannier transformation matrix
!
!**********************************************************************

   SUBROUTINE  DETERMINE_WANNIER_PROJECTOR(WDES,NK,ISP,U,IU0)
      USE prec
      USE wave
      USE mlwf
      USE wnpr
      USE full_kpoints
      IMPLICIT NONE
      TYPE(wavedes) WDES
      INTEGER       NK
      INTEGER       ISP
      GDEF          U(:,:) 
      INTEGER       IU0
      !local
      INTEGER I,J,K,L,IP,IPP
      INTEGER       NKP 
      INTEGER NW,NB,NBEXCL,NBwin
      INTEGER ISPINOR
      GDEF CTMP

      !depending on the baisis choice the transformation matrix 
      !needs to be set up appropriately

      !----------------------------------------------------------------
      IF( LUSEMLWF ) THEN   ! maximally localized Wannier functions
      !----------------------------------------------------------------
         IF ( .NOT. LUFROMFILE ) THEN
            !fix the number of Wannier and considered Bloch bands
            NW=SIZE(U_matrix,1)       !# of Wannier bands
            NB=SIZE(U_matrix_opt,1)   !# of Bloch bands
            NBEXCL=0
            DO I=1,SIZE(lexclude_band)
               IF (lexclude_band(I)) NBEXCL=NBEXCL+1
            ENDDO
            IF ((NB+NBEXCL)/=WDES%NB_TOT) THEN
               WRITE(*,*) 'DETERMINE_WANNIER_PROJECTOR ERROR: '//&
                  'inconsistent number of bands for MLWF:', NB+NBEXCL,WDES%NB_TOT 
            ENDIF

            !second, check again consistency of numbers of considered and desired bands
            IF (SIZE(lexclude_band)/=WDES%NB_TOT) THEN
               IF (IU0>=0) WRITE(*,*) 'DETERMINE_WANNIER_PROJECTOR ERROR: '//&
                 'lexclude_band array is corrupt for MLWF:', &
               &   SIZE(lexclude_band),WDES%NB_TOT
               STOP
            ENDIF
            NBEXCL=0
            DO I=1,SIZE(lexclude_band)
               IF (lexclude_band(I)) NBEXCL=NBEXCL+1
            ENDDO
            IF ((NB+NBEXCL)/=WDES%NB_TOT) THEN
               IF (IU0>=0) WRITE(*,*) 'DETERMINE_WANNIER_PROJECTOR ERROR: '//&
               'inconsistent number of bands for MLWF :', &
               &   NB+NBEXCL,WDES%NB_TOT 
            ENDIF

            ! find the corresponding entry in KPOINTS_FULL, that contains
            ! the set of k-points used in the computation of the rotation matrices
            DO NKP=1,KPOINTS_FULL%NKPTS
               IF (LIDENTICAL_KPOINT(WDES%VKPT(:,NK),kpt_latt(:,NKP))) exit 
            ENDDO
            IF (NKP>KPOINTS_FULL%NKPTS) THEN
               IF (IU0>=0) WRITE(*,*)'DETERMINE_WANNIER_PROJECTOR ERROR: no matching k-point found'//&
                 ' in kpt_latt',NK
               STOP
            ENDIF

            ! setup the effective rotation matrix U at the present k-point
            U=zero
            IP=0; IPP=0
            bands: DO I=1,WDES%NB_TOT
               IF (lexclude_band(I)) CYCLE bands
               IP=IP+1
               IF (.NOT.lwindow(IP,NKP,ISP)) CYCLE bands
               IPP=IPP+1
               DO J=1,NW
                  CTMP=zero 
                  DO K=1,NW
                     CTMP=CTMP+U_matrix_opt(IPP,K,NKP,ISP)*U_matrix(K,J,NKP,ISP)
                  ENDDO
                  U(I,J)=CTMP
               ENDDO
            ENDDO bands
         ELSE
            !read projector from file 
            IF ( .NOT. ALLOCATED(UWAN) )THEN
               IF (IU0>=0) WRITE(*,*)'DETERMINE_WANNIER_PROJECTOR ERROR: UWAN not allocated'
               STOP
            ENDIF
            U(1:WDES%NB_TOT,1:WDES%NB_TOT)=UWAN(1:WDES%NB_TOT,1:WDES%NB_TOT,NK,ISP)
         ENDIF
      !----------------------------------------------------------------
      ELSEIF ( LUSELCAO ) THEN
      !----------------------------------------------------------------
         !the transformation matrix must be allocated for this basis choice
         IF( .NOT.ALLOCATED(WNPR_projections) ) THEN
            IF ( IU0>=0 ) WRITE(IU0,'(A,L4)')'DETERMINE_WANNIER_PROJECTOR ERROR: WNPR_projections '//&
               'not allocated!'
            STOP
         ENDIF 
   
         !currently we don't support spinors!
         IF ( WDES%NRSPINORS>1 ) THEN
            IF ( IU0>=0 ) WRITE(IU0,'(A,L4)')'DETERMINE_WANNIER_PROJECTOR ERROR: Currently no'//&
               'spinors are supported. I know I have to fix that:)'
            STOP
         ENDIF 

         ! find the corresponding entry in KPOINTS_FULL, that contains
         ! the set of k-points used in the computation of the rotation matrices
         DO NKP=1,KPOINTS_FULL%NKPTS
            IF (LIDENTICAL_KPOINT(WDES%VKPT(:,NK),KPOINTS_FULL%VKPT(:,NKP))) exit 
         ENDDO
         IF (NKP>KPOINTS_FULL%NKPTS) THEN
            IF (IU0>=0)WRITE(*,*)'DETERMINE_WANNIER_PROJECTOR ERROR: no matching k-point found'//&
            ' in KPOINTS_FULL',NK
            STOP
         ENDIF

         U=zero
         DO J=1,WNPR_num_wann
            DO I=1,WDES%NB_TOT
               IF (.NOT.WNPR_use_band(I,NKP,ISP)) CYCLE
               DO ISPINOR=0,WDES%NRSPINORS-1
                  U(I,J+WNPR_num_wann*ISPINOR)=WNPR_projections(I,NKP,MAX(ISP,ISPINOR+1),J)
               ENDDO
            ENDDO
         ENDDO

      !----------------------------------------------------------------
      ELSEIF ( LUSEBLOCH ) THEN
      !----------------------------------------------------------------
         !maybe to most simple tranformation :)
         U=zero
         DO I=1,WDES%NB_TOT
            U(I,I) = 1._q
         ENDDO
      !----------------------------------------------------------------
      ENDIF 
      !----------------------------------------------------------------
      
   ENDSUBROUTINE DETERMINE_WANNIER_PROJECTOR

!**********************************************************************
!
! Determine correlated projector P
!
!**********************************************************************

   SUBROUTINE  DETERMINE_CORRELATED_PROJECTOR(NB_TOT,U,P,NK,ISP,IU0,W)
      IMPLICIT NONE
      INTEGER       NB_TOT
      GDEF          U(:,:) 
      GDEF          P(:,:) 
      INTEGER       IU0,NK,ISP
      GDEF,OPTIONAL :: W(:,:,:) !window function
      !local
      INTEGER I,J,K
 
      ! test will be necessary 
      IF ( SIZE( U,1) /= SIZE(P,1) .AND. SIZE(P,1) < NB_TOT ) THEN
         IF (IU0>=0) WRITE(IU0,*)'internal error in VASP from DETERMINE_CORRELATED_PROJECTOR: Size 1 of U and P'//&
                ' inconsistent', SIZE( U,1) ,SIZE(P,1) , NB_TOT
         STOP
      ENDIF 
      IF ( SIZE( U,2) /= SIZE(P,2) .AND. SIZE(P,2) < NB_TOT ) THEN
         IF (IU0>=0) WRITE(IU0,*)'internal error in VASP from DETERMINE_CORRELATED_PROJECTOR: Size 2 of U and P'//&
                ' inconsistent', SIZE( U,2) ,SIZE(P,2) , NB_TOT
         STOP
      ENDIF 
 
      P(:,:)=zero 
      !determine correlated projector U 
      DO I = 1, NB_TOT
         DO K = 1, NB_TOT
            !only correlated states contribute to projector 
            DO J = 1, DMFT_num_wann
               IF ( NTARGET_STATES(J) == 1) THEN
                  P(K,I)=P(K,I)+GCONJG(U(I,J))*U(K,J)
               ENDIF
            ENDDO         
         ENDDO              
      ENDDO    
 
      IF ( ALLOCATED(CRPA_WEIGHTS))THEN
        DO I = 1, NB_TOT
           CRPA_WEIGHTS(I,NK,ISP)=REAL(P(I,I),q)
        ENDDO
      ENDIF
      !if Window function is present we multiply the projector with it
      IF (PRESENT(W) .AND. LREGULARIZE) THEN
         !check if sizes are correct
         IF ( SIZE(W,1) /= NB_TOT ) THEN
            IF(IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from DETERMINE_CORRELATED_PROJECTOR:'//&
               'Size 1 of W and U inconsistent',SIZE(W,1),NB_TOT
            STOP
         ENDIF 
         !check if sizes are correct
         IF ( SIZE(W,2) < NK ) THEN
            IF(IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from DETERMINE_CORRELATED_PROJECTOR:'//&
               'Size 2 of W and U inconsistent',SIZE(W,2),NK
            STOP
         ENDIF 
         !check if sizes are correct
         IF ( SIZE(W,3) < ISP) THEN
            IF(IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from DETERMINE_CORRELATED_PROJECTOR:'//&
               'Size 3 of W and U inconsistent',SIZE(W,3),ISP
            STOP
         ENDIF 
 
         DO I = 1, NB_TOT
            P(:,I)=P(:,I)*W(I,NK,ISP)
         ENDDO              
#ifdef verbose 
         IF ( IU0>=0 ) THEN
            WRITE(*,'(A,I4)')'******************* WINDOW *********** NK=',NK
            CALL DUMP_PROJ(P,NB_TOT,3, 17, 33,.TRUE.)
         ENDIF 
      ELSE 
         IF ( IU0>=0 ) THEN
            WRITE(*,'(A,I4)')'++++++++++++++++++ original ++++++++++ NK=',NK
            CALL DUMP_PROJ(P,NB_TOT,3, 17, 33, .TRUE.)
          ENDIF 
#endif
      ENDIF
 
 
   ENDSUBROUTINE DETERMINE_CORRELATED_PROJECTOR

!**********************************************************************
!
! orthogonalizes non-correlated sub system
! 
!**********************************************************************

  SUBROUTINE DIAGONALIZE_NONCORR_SYSTEM(W, U, NCORR, EIG , NK,NCOLSTART,NCOLSTOP, ISP)
      USE base
      USE wave_high
      USE constant
      IMPLICIT NONE
      TYPE(wavespin) :: W
      GDEF           :: U(:,:)
      INTEGER        :: NCORR
      REAL(q)        :: EIG(:)
      INTEGER        :: NK , ISP
      INTEGER        :: NCOLSTART,NCOLSTOP
      !local
      INTEGER                :: NB_TOT
      INTEGER                :: I, J, K
      GDEF, ALLOCATABLE      :: S(:,:)
      GDEF, ALLOCATABLE      :: H(:,:), HP(:,:)
      REAL(q), ALLOCATABLE   :: SIGMA(:)
      INTEGER                :: NC
      REAL(q)                :: RTMP
      LOGICAL, ALLOCATABLE   :: LTAG(:)
     
      !the total number of bands is NBANDS
      NB_TOT = W%WDES%NB_TOT
 
      !check the size of EIG needs to be OK
      IF ( SIZE(EIG)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: EIG not large enough',SIZE(EIG),NB_TOT
         STOP
      ENDIF 
      !also check if U is allocated and has proper size 
      IF ( SIZE(U,1)/=NB_TOT .OR. SIZE(U,2)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: U not large enough',SIZE(U,1),SIZE(U,2),NB_TOT
         STOP
      ENDIF 

      !allocate auxillary matrices 
      ALLOCATE(S(NB_TOT,NB_TOT))
      ALLOCATE(H(NB_TOT,NB_TOT))
      ALLOCATE(HP(NB_TOT,NB_TOT))
      ALLOCATE(SIGMA(NB_TOT))
      S(:,:)=U(:,:)
      H=zero
      SIGMA=0
 
      ! build correlated hamiltonian
      H=zero
      DO I = 1, NB_TOT
         DO J = 1, NB_TOT
             DO K = 1, NB_TOT
                H(I,J) = H(I,J) + U(K, J)*REAL(W%CELTOT(K,NK,ISP),q)*GCONJG(U(K,I))
             ENDDO
         ENDDO
      ENDDO

      !now diagonalize Hamiltonian 
      CALL DIAGH( H, NB_TOT, EIG)
      ! check result
      ! find number of non-zero eigenvalues (should be NCORR eigenvalues )
      NC = 0
      SIGMA=0
      SIGMA=EIG
      EIG=0  
      DO I = 1, NB_TOT
 
         IF ( ABS(SIGMA(I))> TINYLARGE) THEN
           NC = NC + 1 
           EIG(NC)=SIGMA(I)
         ENDIF
 
      ENDDO
      IF ( NC /= NCORR ) THEN
         WRITE(*,*)'internal error in VASP from DIAGONALIZE_NONCORR_SYSTEM # of corr. states inconsistent',&
         NC , NCORR, NK
         STOP
      ENDIF 
 
      !compute effective rotation matrix
      U=zero
#ifdef gammareal
      CALL DGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#else
      CALL ZGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#endif 

      !resulting matrix has a block of mostly zero entries between NCOLSTART and NCOLSTOP
      !these band indices are determined by the energy window in the Wannier projection
      !there are only NCOLSTOP-NCOLSTART+1-DMFT_num_states non-zero states in this block 
      !these columns have to be identified and shifted to the right in this block 
      H=zero
      H=U
      ALLOCATE(LTAG(NB_TOT))
      LTAG(:)=.FALSE. 
      !scan relevant block for non-zero elements 
!WRITE(*,*)NK,NB_TOT
      DO J = NCOLSTART, NCOLSTOP 
         RTMP=0
         DO I = 1, NB_TOT
            RTMP = RTMP + ABS(H(I,J))
            !RTMP = RTMP + REAL(H(I,J),q)
         ENDDO 
         !tag column for non-zero entries 
         IF ( RTMP > 1.E-12_q ) LTAG(J)=.TRUE. 
! WRITE(*,'(I4,L4,F12.6," for ",16F10.4)')J,LTAG(J),RTMP,REAL(H(:,J))
      ENDDO
     
      S=H
      S(:,NCOLSTART:NCOLSTOP)=0
      J=0
      DO I = NCOLSTART, NCOLSTOP
         IF ( LTAG(I)) THEN
            J=J+1
            S(:,NCOLSTOP-J+1)=H(:,I)
         ENDIF
      ENDDO

#ifdef debug
WRITE(100,*)'non-corr NK=',NK,NCOLSTART,NCOLSTOP
WRITE(100,'(16L12)')LTAG(1:16)
DO I = 1, NB_TOT
WRITE(100,'(16F12.6)')REAL(U(I,1:16),q)
ENDDO
WRITE(100,*)'IMAG'
DO I = 1, NB_TOT
WRITE(100,'(16F12.6)')IMAG(U(I,1:16))
ENDDO
      
WRITE(100,*)'rearranged for NK=',NK,NCOLSTART,NCOLSTOP
WRITE(100,'(16L12)')LTAG(1:16)
WRITE(100,'(16F12.6)')EIG(1:16)
DO I = 1, NB_TOT
WRITE(100,'(16F12.6)')REAL(S(I,1:16),q)
ENDDO
WRITE(100,*)'IMAG'
DO I = 1, NB_TOT
WRITE(100,'(16F12.6)')IMAG(S(I,1:16))
ENDDO
#endif

      !restore 
      U=S
      !get rid of auxilary arrays 
      DEALLOCATE( S ) 
      DEALLOCATE( H ) 
      DEALLOCATE( HP ) 
      DEALLOCATE( LTAG )

      RETURN
   ENDSUBROUTINE DIAGONALIZE_NONCORR_SYSTEM 

  SUBROUTINE ORTHOGONALIZE_SYSTEM(W, U, NCORR, EIG , NK, ISP)
      USE base
      USE wave_high
      USE constant
      IMPLICIT NONE
      TYPE(wavespin) :: W
      GDEF           :: U(:,:)
      INTEGER        :: NCORR
      REAL(q)        :: EIG(:)
      INTEGER        :: NK , ISP
      !local
      INTEGER                :: NB_TOT
      INTEGER                :: I, J, K
      GDEF, ALLOCATABLE      :: S(:,:)
      GDEF, ALLOCATABLE      :: H(:,:), HP(:,:)
      REAL(q), ALLOCATABLE   :: SIGMA(:)
      INTEGER                :: NC
     
      !the total number of bands is NBANDS
      NB_TOT = W%WDES%NB_TOT
 
      !check the size of EIG needs to be OK
      IF ( SIZE(EIG)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: EIG not large enough',SIZE(EIG),NB_TOT
         STOP
      ENDIF 
      !also check if U is allocated and has proper size 
      IF ( SIZE(U,1)/=NB_TOT .OR. SIZE(U,2)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: U not large enough',SIZE(U,1),SIZE(U,2),NB_TOT
         STOP
      ENDIF 
 
      !allocate auxillary matrices 
      ALLOCATE(S(NB_TOT,NB_TOT))
      ALLOCATE(H(NB_TOT,NB_TOT))
      ALLOCATE(HP(NB_TOT,NB_TOT))
      ALLOCATE(SIGMA(NB_TOT))
      S(:,:)=U(:,:)
      H=zero
      SIGMA=0
 
      ! build correlated hamiltonian
      H=zero
      DO I = 1, NB_TOT
         DO J = 1, NB_TOT
             DO K = 1, NB_TOT
                H(I,J) = H(I,J) + U(K, J)*W%CELTOT(K,NK,ISP)*GCONJG(U(K,I))
             ENDDO
         ENDDO
      ENDDO
 
      !now diagonalize Hamiltonian 
      CALL DIAGH( H, NB_TOT, EIG)
      ! check result
      ! find number of non-zero eigenvalues (should be NCORR eigenvalues )
      NC = 0
      SIGMA=0
      SIGMA=EIG
      EIG=0  
      DO I = 1, NB_TOT
         EIG(I)=SIGMA(I)
      ENDDO
 
      !compute effective rotation matrix
      U=zero
#ifdef gammareal
      CALL DGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#else
      CALL ZGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#endif 
 
      !now we can get rid of the auxilary matrices S and H
      DEALLOCATE( S ) 
      DEALLOCATE( H ) 
      DEALLOCATE( HP ) 
  
      RETURN
   ENDSUBROUTINE ORTHOGONALIZE_SYSTEM 

!**********************************************************************
!
! orthogonalizes correlated sub system
! 
!**********************************************************************

   SUBROUTINE DIAGONALIZE_CORR_SYSTEM(W, U, NCORR, EIG , NK, ISP, LORTHOGONALIZE)
      USE base
      USE wave_high
      USE constant
      IMPLICIT NONE
      TYPE(wavespin) :: W
      GDEF           :: U(:,:)
      INTEGER        :: NCORR
      REAL(q)        :: EIG(:)
      INTEGER        :: NK , ISP
      LOGICAL        :: LORTHOGONALIZE
      ! local
      INTEGER                :: NB_TOT, NB_SUB
      INTEGER                :: I, J, K
      GDEF, ALLOCATABLE      :: S(:,:)
      GDEF, ALLOCATABLE      :: H(:,:), HP(:,:)
      REAL(q), ALLOCATABLE   :: SIGMA(:)
      INTEGER                :: NC
     
      !the total number of bands is NBANDS
      NB_TOT = W%WDES%NB_TOT
      !number of states in the subspace
      NB_SUB = NCORR
 
      !check the size of EIG needs to be OK
      IF ( SIZE(EIG)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: EIG not large enough',SIZE(EIG),NB_TOT
         STOP
      ENDIF 
      !also check if U is allocated and has proper size 
      IF ( SIZE(U,1)/=NB_TOT .OR. SIZE(U,2)/=NB_TOT ) THEN
         WRITE(*,*)'ERROR: U not large enough',SIZE(U,1),SIZE(U,2),NB_TOT
         STOP
      ENDIF 
 
      !allocate auxillary matrices 
      ALLOCATE(S(NB_TOT,NB_TOT))
      ALLOCATE(H(NB_TOT,NB_TOT))
      ALLOCATE(HP(NB_TOT,NB_TOT))
      ALLOCATE(SIGMA(NB_TOT))
      S(:,:)=U(:,:)
      H=zero
      SIGMA=0
 
      ! build correlated hamiltonian
      H=zero
      DO I = 1, NB_TOT
         DO J = 1, NB_TOT
             DO K = 1, NB_TOT
                H(I,J) = H(I,J) + U(K, J)*REAL(W%CELTOT(K,NK,ISP),q)*GCONJG(U(K,I))
             ENDDO
         ENDDO
      ENDDO
       
      !now diagonalize Hamiltonian 
      CALL DIAGH( H, NB_SUB, EIG)
      ! check result
      ! find number of non-zero eigenvalues (should be NCORR eigenvalues )
      NC = 0
      SIGMA=0
      SIGMA=EIG
      EIG=0  
      DO I = 1, NB_SUB
         IF ( ABS(SIGMA(I))>TINYLARGE ) THEN
           NC = NC + 1 
           EIG(NC)=SIGMA(I)
         ENDIF
      ENDDO
      IF ( NC /= NCORR ) THEN
         WRITE(*,*)'internal error in VASP from DIAGONALIZE_CORR_SYSTEM # of corr. states inconsistent',&
         NC , NCORR, NK
         STOP
      ENDIF 
      !compute effective rotation matrix
      U=zero
#ifdef gammareal
      CALL DGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#else
      CALL ZGEMM( 'N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
         H(1,1), NB_TOT, 0._q, U(1,1), NB_TOT)
#endif 
 

      IF ( LORTHOGONALIZE ) THEN
         S=U
         H=U
         HP=zero
         !build overlap
#ifdef gammareal
         CALL DGEMM( 'T', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
            H(1,1), NB_TOT, 0._q, HP(1,1), NB_TOT)
#else
         CALL ZGEMM( 'C', 'N', NB_TOT, NB_TOT, NB_TOT, 1._q, S(1,1), NB_TOT,&
            H(1,1), NB_TOT, 0._q, HP(1,1), NB_TOT)
#endif 

#ifdef debug
WRITE(100,*)'overlap NK=',NK
DO I = 1, 16
WRITE(100,'(16F12.6)')REAL(HP(I,1:16),q)
ENDDO
WRITE(100,*)'IMAG'
DO I = 1, 16
WRITE(100,'(16F12.6)')IMAG(HP(I,1:16))
ENDDO
#endif

      ENDIF

      !now we can get rid of the auxilary matrices S and H
      DEALLOCATE( S ) 
      DEALLOCATE( H ) 
      DEALLOCATE( HP ) 

      RETURN
   ENDSUBROUTINE DIAGONALIZE_CORR_SYSTEM 

!**********************************************************************
!
! Diagonalizes a matrix CUNI and returns its eigenvalues in HFEIG
!
!**********************************************************************

   SUBROUTINE DIAGH(CUNI, NBANDS, HFEIG )
      GDEF    :: CUNI(:,:)
      INTEGER    :: NBANDS
      REAL(q) :: HFEIG(NBANDS)
      INTEGER :: NEIG
    ! local
      INTEGER, PARAMETER :: LWORK=32
      GDEF    ::CWRK(LWORK*NBANDS)
      REAL(q) :: W(3*NBANDS)
      REAL(q):: EV(NBANDS,NBANDS)
      INTEGER :: IFAIL, NDIM, N
      NDIM = SIZE(CUNI,1)
  

#ifdef gammareal
         CALL DSYEV &
           ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
           W,  IFAIL)
      IF (IFAIL/=0) THEN
         WRITE(*,*) 'internal error in VASP from DIAGH: Call to routine DSYEV failed! '// &
              &              'Error code was ',IFAIL
#else
         CALL ZHEEV &
           ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CWRK,LWORK*NBANDS, &
           W,  IFAIL)
      IF (IFAIL/=0) THEN
         WRITE(*,*) 'internal error in VASP from DIAGH: Call to routine ZHEEV failed! '// &
              &              'Error code was ',IFAIL
#endif 
         STOP
      ENDIF
   END SUBROUTINE DIAGH

!**********************************************************************
! checks if UWAN exists, if so correct entries are found for the grid
!**********************************************************************

   SUBROUTINE READ_WANPROJ(WDES,IO, LSETDMFT_WANN)
      USE base
      USE wave_high
      USE mkpoints
      USE constant
      USE mlwf 
      USE full_kpoints
      USE fileio
      IMPLICIT NONE 
      TYPE (wavedes)     WDES
      TYPE (in_struct)   IO
      LOGICAL, OPTIONAL :: LSETDMFT_WANN
      !local 
      INTEGER            NKPTS,NB_TOT,NW,ERROR
      INTEGER            ISPIN,IERR
      INTEGER            ISP,NKP,NK, I,J,I_,J_,K
      REAL(q)            VKPT(3)
      REAL(q)            RTMP1,RTMP2
      
      !if projection matrices are set up, return imediately
      IF ( ALLOCATED(UWAN)) RETURN
   
      INQUIRE(FILE=WANPROJFILE,EXIST=LUFROMFILE)
 
      !allocate UWAN matrices
      ALLOCATE(UWAN(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      UWAN=0
      NKP=0
      NW=0
      IERR=0
      !if file exist only root opens it and reads it 
      IF ( LUFROMFILE ) THEN
#ifdef MPI
         IF ( WDES%COMM%NODE_ME==1) THEN
#endif
            WRITE(*,'(" Found ",A," file, reading...")',ADVANCE='NO')WANPROJFILE
            OPEN(UNIT=99,FILE=WANPROJFILE,STATUS='OLD')
            
            !read header: number of k-points, bands and wannier states
            READ(99,*)ISPIN,NKPTS,NB_TOT,NW
            !maybe we need to kill the job here
            IF ( NKPTS < WDES%NKPTS ) THEN
               WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent NKPTS',NKPTS,WDES%NKPTS
               IERR=-1
            ELSEIF( ISPIN < WDES%ISPIN) THEN
               WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent ISPIN',ISPIN,WDES%ISPIN
               IERR=-2
            ELSEIF( NB_TOT /= WDES%NB_TOT) THEN
               WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent NBTOT',NB_TOT,WDES%NB_TOT
               IERR=-3
            ENDIF 
            !skip the k-points list
            DO NK=1,NKPTS
               READ(99,*)I,VKPT(:)
            ENDDO
 
            K=0
            scan: DO !scan the wann file 
               K=K+1
               !read spin polarization and current k-point from file:
               READ(99,*,IOSTAT=ERROR)ISP,NB_TOT,VKPT(1:3)
               !look in the k-point list, if there is an entry for this point
               k_list: DO NK = 1, WDES%NKPTS
                  !check if current k-point of list corresponds to k-point of file
                  IF (LIDENTICAL_KPOINT(WDES%VKPT(:,NK),VKPT(:))) THEN
                     ! if an entry is found, store the corresponding matrix to UWANN
                     NKP=NKP+1  !this is the total number of k-points found in list
                     DO I =1, NB_TOT !bands
                        DO J =1,NW   !wannier states
                           READ(99,*,IOSTAT=ERROR)I_,J_,RTMP1,RTMP2
                           IF ( ERROR == 0 ) THEN
                              !store the matrix 
                              UWAN(I_,J_,NK,ISP)=CMPLX(RTMP1,RTMP2,q)
                           ELSE
                              WRITE(*,*)' ERROR'
                              EXIT scan
                           ENDIF
                        ENDDO
                     ENDDO 
                     !if point in list is found, get out of list loop
                     EXIT k_list
                  ENDIF
               ENDDO k_list
              
               !if no point was found in k-points list, skip to next point
               IF ( NK > WDES%NKPTS )THEN
                  DO I =1, NB_TOT
                     DO J =1,NW 
                        READ(99,*,IOSTAT=ERROR)I_,J_,RTMP1,RTMP2
                        IF ( ERROR /= 0 ) THEN
                           WRITE(*,*)'ERROR (2)'
                           EXIT scan
                        ENDIF
                     ENDDO
                  ENDDO 
               ENDIF
               !maybe we have enough points
               IF ( NKP==WDES%ISPIN*WDES%NKPTS)THEN
                  WRITE(*,*)'All k-points found'
                  EXIT scan
               ENDIF
               !also get out if all points in file have been done
               IF ( K==NKPTS*ISPIN) EXIT scan
            ENDDO  scan
            CLOSE(99)
#ifdef MPI        
         ENDIF
#endif

         !Terminate job eventually
         CALLMPI( M_bcast_i(WDES%COMM,NKP,1) ) 
         IF (NKP/=WDES%NKPTS*WDES%ISPIN) THEN
            WRITE(*,*)'ERROR: not all k-points could be found in '//WANPROJFILE,NKP,WDES%NKPTS
            STOP
         ENDIF

         CALLMPI( M_bcast_i(WDES%COMM,IERR,1) ) 
         IF ( IERR==-1 ) THEN
            WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent NKPTS',NKPTS,WDES%NKPTS
            STOP
         ELSEIF ( IERR==-2 ) THEN
            WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent ISPIN',ISPIN,WDES%ISPIN
            STOP
         ELSEIF ( IERR ==-3 )THEN
            WRITE(*,*)'internal error in VASP: READ_WANPROJ reports inconsistent NBTOT',NB_TOT,WDES%NB_TOT
            STOP
         ENDIF

         !communicate the number of Wannier states
         CALLMPI(M_sum_i(WDES%COMM,NW,1))
         MLWF_num_wann=NW
         IF ( PRESENT( LSETDMFT_WANN) ) DMFT_num_wann=NW
         !when everything went well, communicate the matrix to other nodes, and we are done
         CALLMPI(M_sum_z(WDES%COMM,UWAN,WDES%NKPTS*WDES%NB_TOT*WDES%NB_TOT*WDES%ISPIN))
      ELSE
         DEALLOCATE(UWAN)
      ENDIF 
   ENDSUBROUTINE READ_WANPROJ

!**********************************************************************
! helper: prints information about correlated subspace
!**********************************************************************

   SUBROUTINE PRINT_CSPACE_INFO(IU0,IU6)
      IMPLICIT NONE
      INTEGER IU0, IU6
      !local 
      INTEGER, ALLOCATABLE :: WTARGET(:)
      INTEGER              :: COUNTER 
      INTEGER              :: I
 
      ALLOCATE(WTARGET(DMFT_num_wann))
      WTARGET=0
      COUNTER=0
      DO I = 1, DMFT_num_wann
         IF ( NTARGET_STATES (I) == 1 ) THEN
            COUNTER = COUNTER + 1
            WTARGET(COUNTER) = I
         ENDIF
      ENDDO
      IF ( COUNTER >0 )THEN
         IF ( IU0>=0 ) WRITE(IU0,'(" Correlated subspace contains the Wannier'//&
            ' states #: ",/10I4)')WTARGET(1:COUNTER)
         IF ( IU6>=0 ) WRITE(IU6,'(" Correlated subspace contains the Wannier'//&
            ' states #: ",/10I4)')WTARGET(1:COUNTER)
      ELSE
         IF ( IU0>=0 ) WRITE(IU0,'(" No states in correalted subspace present")')
         IF ( IU6>=0 ) WRITE(IU6,'(" No states in correalted subspace present")')
      ENDIF
      DEALLOCATE(WTARGET)
     
   ENDSUBROUTINE PRINT_CSPACE_INFO 

!*************************SUBROUTINE INWAV_DOWNFOLD*********************
!
!   read wavefunctions from file
!
!   the routine can read WAVECAR files generated for different
!   lattice vectors and cutoff (both in the serial and parallel case)
!
!   it is also possible to restart a spin polarised calculation
!   from a non spin polarised wavefunction file or
!
!***********************************************************************

      SUBROUTINE INWAV_DOWNFOLD(IO, WDES, W, SYMM, KPOINTS, GRID, LATT_CUR, LATT_INI, ISTART, EFERMI)
      USE prec
      USE base
      USE wave_high
      USE mgrid
      USE lattice
      USE mkpoints
      USE fileio 
      USE mlr_optic , ONLY:IKPT_SELECT
 
      TYPE (in_struct)   IO
      TYPE (wavedes)  WDES
      TYPE (wavedes1) WDES1
      TYPE (wavespin) W
      TYPE (symmetry)    SYMM
      TYPE (kpoints_struct) :: KPOINTS
      TYPE (grid_3d)  GRID
      TYPE (latt)     LATT_CUR,LATT_INI
      INTEGER ISTART 
      REAL (q) :: EFERMI

! local work arrays
      REAL(q) VKPT(3)
      REAL(q) VKPT2(3)
      INTEGER,ALLOCATABLE    :: IND(:),INDI(:)
      COMPLEX(q),ALLOCATABLE :: CW1(:),CW2(:)
      LOGICAL ::  SINGLE_PREC, SPAN_RECORDS
      COMPLEX(qs),  ALLOCATABLE :: CRD(:)
      LOGICAL, POINTER :: LCONJG(:)=>null()
      INTEGER, POINTER :: INDEX(:)=>null()
      REAL(q), ALLOCATABLE :: INBUF(:)
!wavecar read in
      COMPLEX(q),ALLOCATABLE:: CELTOT(:,:)   ! global array for eigenvalues
      REAL(q)   ,ALLOCATABLE:: FERTOT(:,:)   ! global array for fermi-weights
      INTEGER    :: NK
      INTEGER    :: NKPT_COUNT
 
      INTEGER  IU0, ISP, J, K, I , IS
      INTEGER  IFAIL , IRECL 
      INTEGER  IONODE,NALLO, ISPINREAD
      INTEGER  NODE_ME, NBANDF, MALLOC
      INTEGER   NCOMM, NALLOC
      REAL(q) RTAG, RDUM, RISPIN,  RKPTSF,RBANDF 
      REAL(q) ENMAXF, RNPL
      INTEGER IREC, NKPTSF, NPLREAD, NPL

      IU0=IO%IU0

      NALLOC = MAXVAL(WDES%NPLWKP_TOT)

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif
!
! parse the header
!
        RTAG=0
        READ(12,REC=1,ERR=200) RDUM,RISPIN,RTAG
        IF (RTAG==53300) THEN
           IF (IU0>=0) WRITE(IU0,*) "VASP.5.3 WAVECAR encountered"
           SINGLE_PREC=.TRUE.
           SPAN_RECORDS=.TRUE.
        ELSE IF (RTAG==53310) THEN
           IF (IU0>=0) WRITE(IU0,*) "VASP.5.3 double precision WAVECAR encountered"
           SINGLE_PREC=.FALSE.
           SPAN_RECORDS=.TRUE.
        ELSE IF (RTAG==45200) THEN
           SINGLE_PREC=.TRUE.
           SPAN_RECORDS=.FALSE.
        ELSE
           IF (IU0>=0) WRITE(IU0,*) "double precision WAVECAR encountered, converting it"
           SINGLE_PREC=.FALSE.
           SPAN_RECORDS=.FALSE.
        ENDIF

        ISPINREAD=NINT(RISPIN)
        READ(12,REC=2,ERR=200) RKPTSF,RBANDF,ENMAXF, &
     &                         ((LATT_INI%A(I,J),I=1,3),J=1,3),EFERMI

        IREC=2
        NKPTSF=NINT(RKPTSF)
        NBANDF=NINT(RBANDF)

        ALLOCATE(CELTOT(NBANDF,ISPINREAD))
        CELTOT=0
        ALLOCATE(FERTOT(NBANDF,ISPINREAD))
        FERTOT=0

        !is every k-point taken into account?
        ALLOCATE(IKPT_SELECT(NKPTSF))
        DO K = 1, NKPTSF
           IKPT_SELECT(K)=K
        ENDDO
      
        CALL LATTIC(LATT_INI)
!
! if number of k-points in WAVECAR > then current number of k-points, we need to select
! appropriate k-points 
!        
       IF ( NKPTSF > WDES%NKPTS ) THEN
         IF ( IU0>=0 )WRITE(IU0,'(A,I4,A)',ADVANCE='NO')&
            ' WAVECAR contains more than',WDES%NKPTS,&
            ' k-points, downsampling...'
         DO K = 1, NKPTSF
            IKPT_SELECT(K)=-1
         ENDDO
          CALL FIND_KPOINTS_IN_WAVECAR(WDES,LATT_CUR,SYMM,KPOINTS,NKPTSF,IO%IU0)
       ENDIF 
       
!=======================================================================
! read WAVECAR file, number of bands agree
!=======================================================================
      IF (.NOT.(WDES%NRSPINORS==2 .AND. NBANDF*WDES%NRSPINORS == WDES%NB_TOT)) THEN
        spin:    DO ISP=1, MIN(WDES%ISPIN, ISPINREAD)
        NKPT_COUNT=0
        kpts: DO K=1,NKPTSF
 
        IF ( IKPT_SELECT(K)> NKPTSF ) THEN
           IREC=IRECL
        ELSE
           IRECL=IREC
        ENDIF

        IF ( IKPT_SELECT(K) > 0 ) THEN
           NKPT_COUNT = NKPT_COUNT + 1
           CALL SETWDES(WDES,WDES1,IKPT_SELECT(K))
#ifdef gammareal
           CALL GEN_INDEX_GAMMA(WDES1, LATT_CUR%B, WDES%ENMAX, INDEX, LCONJG)
#endif
        io_begin
        IFAIL=0
        IF (.NOT. SPAN_RECORDS) THEN
           IREC=IREC+1
           !read energy and occupancy for kpoint of file
           READ(12,REC=IREC,ERR=230) RNPL,VKPT, &
                (W%CELTOT(J,IKPT_SELECT(K),ISP),W%FERTOT(J,IKPT_SELECT(K),ISP),J=1,MIN(WDES%NB_TOT,NBANDF))
        ELSE
           ALLOCATE(INBUF(4+NBANDF*3))
           CALL READ_TO_BUF(12, IREC, IO%IRECLW/IO%ICMPLX*2, INBUF, IFAIL)
           RNPL   =INBUF(1)
           VKPT(1)=INBUF(2)
           VKPT(2)=INBUF(3)
           VKPT(3)=INBUF(4)
           DO J=1,MIN(WDES%NB_TOT,NBANDF)
              W%CELTOT(J,IKPT_SELECT(K),ISP)=CMPLX(INBUF(5+(J-1)*3),INBUF(6+(J-1)*3),q)
              W%FERTOT(J,IKPT_SELECT(K),ISP)=INBUF(7+(J-1)*3)
           ENDDO
           DEALLOCATE(INBUF)

        ENDIF
        NPLREAD=NINT(RNPL)
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 230

        CALLMPI( M_bcast_i( WDES%COMM, IREC, 1))
        CALLMPI( M_bcast_i( WDES%COMM, NPLREAD, 1))
        NPL=WDES%NPLWKP_TOT(IKPT_SELECT(K))

        MALLOC=MAX(NPL, NPLREAD)
        ALLOCATE(CW1(MALLOC),CW2(MALLOC),CRD(MALLOC),IND(MALLOC),INDI(MALLOC))

        io_begin
        ! create an index to allow for change of cutoff or cell size
        CALL REPAD_INDEX_ARRAY(GRID, WDES%VKPT(:,IKPT_SELECT(K)), VKPT, LATT_CUR%B,  LATT_INI%B, &
                WDES%ENMAX, ENMAXF, NPL/WDES%NRSPINORS, NPLREAD/WDES%NRSPINORS, IND, INDI, MALLOC, IFAIL )
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 220

        band: DO J=1,NBANDF
          IREC=IREC+1
          IF (J>WDES%NB_TOT) CYCLE
          io_begin
            IF (SINGLE_PREC) THEN
               READ(12,REC=IREC,ERR=240) (CRD(I),I=1,NPLREAD)
               CW2(1:NPLREAD)=CRD(1:NPLREAD)
            ELSE
               READ(12,REC=IREC,ERR=240) (CW2(I),I=1,NPLREAD)
            ENDIF

            CW1=0
            DO IS=1,WDES%NRSPINORS
               ! store the wave function coefficients according to new cutoff
               CALL REPAD_WITH_INDEX_ARRAY( MALLOC, IND, INDI, &
                  CW1((IS-1)*NPL/WDES%NRSPINORS+1), CW2((IS-1)*NPLREAD/WDES%NRSPINORS+1))
            ENDDO

          io_end
#ifdef MPI
          IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
             CALLMPI( M_bcast_z(WDES%COMM_KINTER,CW1,SIZE(CW1)) )
          END IF
#endif
#ifdef gammareal
          ! gamma only go from seriel layout to parallel layout if required
          CALL DIS_PW_BAND_GAMMA(WDES1, J, CW1, W%CW(1,1,IKPT_SELECT(K),ISP), INDEX(1), LCONJG(1))
#else
          CALL DIS_PW_BAND(WDES1, J, CW1, W%CW(1,1,IKPT_SELECT(K),ISP))
#endif
        ENDDO band

#ifdef gammareal
        CALL FREE_INDEX_GAMMA(INDEX, LCONJG)
#endif
        DEALLOCATE(CW1,CW2,CRD,IND,INDI)
        ELSE    !in this case we don't store but proceed to read the WAVECAR
           io_begin
              IF(IO%NWRITE>2) WRITE(IU0,'(A,I5,A)')' kpoint,',K,' skipped.'
              IFAIL=0
              IF (.NOT. SPAN_RECORDS) THEN
                 IREC=IREC+1
                 !read energy and occupancy for kpoint of file
                 READ(12,REC=IREC,ERR=230) RNPL,VKPT, &
                      (CELTOT(J,ISP),FERTOT(J,ISP),J=1,MIN(WDES%NB_TOT,NBANDF))
              ELSE
                 ALLOCATE(INBUF(4+NBANDF*3))
                 CALL READ_TO_BUF(12, IREC, IO%IRECLW/IO%ICMPLX*2, INBUF, IFAIL)
                 RNPL   =INBUF(1)
                 VKPT(1)=INBUF(2)
                 VKPT(2)=INBUF(3)
                 VKPT(3)=INBUF(4)
                 DO J=1,MIN(WDES%NB_TOT,NBANDF)
                    CELTOT(J,ISP)=CMPLX(INBUF(5+(J-1)*3),INBUF(6+(J-1)*3),q)
                    FERTOT(J,ISP)=INBUF(7+(J-1)*3)
                 ENDDO
                 DEALLOCATE(INBUF)
   
              ENDIF
              NPLREAD=NINT(RNPL)
           io_end
   
           CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
           IF (IFAIL /=0) GOTO 230
   
           CALLMPI( M_bcast_i( WDES%COMM, IREC, 1))
           CALLMPI( M_bcast_i( WDES%COMM, NPLREAD, 1))
           NPL=1
   
           MALLOC=MAX(NPL, NPLREAD)
           ALLOCATE(CW1(MALLOC),CW2(MALLOC),CRD(MALLOC),IND(MALLOC),INDI(MALLOC))
   
           band2: DO J=1,NBANDF
             IREC=IREC+1
             IF (J>WDES%NB_TOT) CYCLE
             io_begin
               IF (SINGLE_PREC) THEN
                  READ(12,REC=IREC,ERR=240) (CRD(I),I=1,NPLREAD)
                  CW2(1:NPLREAD)=CRD(1:NPLREAD)
               ELSE
                  READ(12,REC=IREC,ERR=240) (CW2(I),I=1,NPLREAD)
               ENDIF
             io_end
           ENDDO band2
   
#ifdef gammareal
           CALL FREE_INDEX_GAMMA(INDEX, LCONJG)
#endif
           DEALLOCATE(CW1,CW2,CRD,IND,INDI)
   
           ENDIF   !decide if something is stored
        ENDDO kpts

        IF (NKPTSF > WDES%NKPTS .AND. ISP == MIN(WDES%ISPIN, ISPINREAD) ) THEN
           IF ( NKPT_COUNT == WDES%NKPTS ) THEN
              IF(IU0>=0)WRITE(IU0,*)'Success'
           ELSE
              IF(IU0>=0)WRITE(IU0,*)' FAILED!'
           ENDIF 
           IREC=IREC+(NKPTSF-WDES%NKPTS)*(WDES%NB_TOT+1)
        ENDIF

        ! copy eigenvalues and weights to all nodes
        NCOMM=WDES%NB_TOT*WDES%NKPTS
        CALLMPI( M_bcast_z(WDES%COMM, W%CELTOT(1,1,ISP),NCOMM ))
        CALLMPI( M_bcast_d(WDES%COMM, W%FERTOT(1,1,ISP),NCOMM ))

        ENDDO spin

        IF (ISPINREAD > WDES%ISPIN .AND. IU0>=0 ) THEN
           WRITE(IU0,*) 'down-spin wavefunctions not read'
        ENDIF

        IF (NBANDF<WDES%NB_TOT) THEN
           IF (IU0>=0) WRITE(IU0,*) 'random initialization beyond band ',NBANDF+1
           CALL WFINIT(WDES, W, 1E10_q, NBANDF+1) ! ENINI=1E10 not cutoff restriction
        ENDIF

        IF (IU0 >= 0) WRITE(IU0,*) 'the WAVECAR file was read successfully'

        IF (WDES%ISPIN<=ISPINREAD) THEN
           IF ( NKPTSF==WDES%NKPTS) DEALLOCATE(IKPT_SELECT)
           RETURN
        ENDIF
!
!  spin down is missing
!
        IF (IU0>=0) &
        WRITE(IU0,*) 'No down-spin wavefunctions found', &
     &             ' --> setting down-spin equal up-spin ...'

        DO K=1,WDES%NKPTS
          W%CELTOT(1:WDES%NB_TOT,K,2)=W%CELTOT(1:WDES%NB_TOT,K,1)
          W%FERTOT(1:WDES%NB_TOT,K,2)=W%FERTOT(1:WDES%NB_TOT,K,1)
          NPL=WDES%NPLWKP(K)
          W%CW(1:NPL,1:WDES%NBANDS,K,2)=W%CW(1:NPL,1:WDES%NBANDS,K,1)
        ENDDO

        IF ( NKPTSF==WDES%NKPTS) DEALLOCATE(IKPT_SELECT)
        RETURN
!=======================================================================
! read collinear WAVECAR file for a non collinear run
!=======================================================================
      ELSE
        IF(IU0>=0) &
           WRITE(IU0,*)' Sorry: downsampling for non-collinear WAVECARs currently not supported'
        STOP
      ENDIF

      IF (IU0>=0) THEN
         WRITE(IU0,*)'ERROR: while reading WAVECAR, file is incompatible'
         IF (WDES%ENMAX /= ENMAXF) &
            WRITE(*,*)'the energy cutoff has changed (new,old) ',WDES%ENMAX,ENMAXF
         IF  (NBANDF /= WDES%NB_TOT) &
            WRITE(*,*)'the number of bands has changed (new,old) ',WDES%NB_TOT,NBANDF
         IF (NKPTSF /= WDES%NKPTS) &
            WRITE(*,*)'the number of k-points has changed (new,old) ',WDES%NKPTS,NKPTSF
      ENDIF
      STOP
!=======================================================================
! can not do anything with WAVECAR 
! hard stop, pull all breaks
!=======================================================================
  200 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading WAVECAR, header is corrupt'
      STOP

  220 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading WAVECAR, plane wave coefficients changed', &
          NPL,NPLREAD
      STOP

  230 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading eigenvalues from WAVECAR',K,ISP
      STOP

  240 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading plane wave coef. from WAVECAR',K,ISP,J
      STOP
      
      END SUBROUTINE INWAV_DOWNFOLD


!**********************************************************************
! finds correct k-points in WAVECAR
!**********************************************************************

   SUBROUTINE FIND_KPOINTS_IN_WAVECAR(WDES,LATT_CUR,SYMM,KPOINTS,&
      IRZ_WAVECAR,IU0)
      USE base
      USE wave_high
      USE mkpoints
      USE constant
      USE full_kpoints
      USE lattice
      USE mlrf_main ! contains KINTER
      USE mlr_optic , ONLY: IKPT_SELECT

      IMPLICIT NONE           
      !kpoints of WAVECAR
      TYPE (kpoints_struct) :: KPOINTS
      TYPE (kpoints_struct) :: KPOINTS_WAVECAR
      TYPE (wavedes)        :: WDES
      TYPE (latt)           :: LATT_CUR
      TYPE (symmetry)       :: SYMM
      INTEGER               :: IRZ_WAVECAR
      INTEGER               :: IU0
      !local 
      INTEGER, PARAMETER    :: KMULT_MAX=36
      INTEGER               :: KMULT
      INTEGER               :: NK,NKP
      INTEGER, DIMENSION(3) :: MULTIPL
      
      !find correct k-point grid
      !create a denser k-point grid 
      KPOINTS_WAVECAR=KPOINTS
      nkpts_grow:DO KMULT = 1, KMULT_MAX
         MULTIPL(:) = KMULT
         CALL RD_KPOINTS(KPOINTS_WAVECAR, LATT_CUR, &
            SYMM%ISYM>=0, SYMM%ISYM<0, -1, -1, MULTIPL = MULTIPL)
         IF ( KPOINTS_WAVECAR%NKPTS == IRZ_WAVECAR) EXIT nkpts_grow
      ENDDO nkpts_grow

      !if k-point grid found  KMULT is below KMULT_MAX
      IF  ( KMULT > KMULT_MAX ) THEN
         IF ( IU0>=0) WRITE(IU0,*)'Error FIND_KPOINTS_IN_WAVECAR:'
         IF ( IU0>=0) WRITE(IU0,*)' cannot find k-point grid corresponding to WAVECAR'
         STOP
      ENDIF 

      !now we have the correct grid, 
      !find out which kpoints of original grid corresponds to current grid
      KMULT=0
      DO NK = 1, KPOINTS_WAVECAR%NKPTS
         !check if current k-point of list corresponds to k-point of file
         DO NKP=1, WDES%NKPTS
            IF (LIDENTICAL_KPOINT(WDES%VKPT(:,NKP),KPOINTS_WAVECAR%VKPT(:,NK))) THEN
               KMULT=KMULT+1
               IKPT_SELECT(NK)=NKP
            ENDIF
         ENDDO
#ifdef debug
      IF ( IU0>=0 )WRITE(*,*)NK,IKPT_SELECT(NK)
#endif 
      ENDDO

      IF ( KMULT/=KPOINTS%NKPTS) THEN
         IF  (IU0>=0) WRITE(IU0,*)'Error FIND_KPOINTS_IN_WAVECAR:'//& 
            'not correct points found',KMULT,KPOINTS%NKPTS
         STOP
      ENDIF     
 
   ENDSUBROUTINE FIND_KPOINTS_IN_WAVECAR


!**********************************************************************
!
! small dumping routine for P, P^2, P^3, ....
!
!**********************************************************************
  SUBROUTINE DUMP_PROJ( P, NB_TOT, NORDER, NDUMP, NSTOP, LPRINT)
    GDEF              :: P (NB_TOT,NB_TOT)
    INTEGER           :: NB_TOT, NORDER
    INTEGER,OPTIONAL  :: NDUMP
    INTEGER,OPTIONAL  :: NSTOP
    LOGICAL,OPTIONAL  :: LPRINT
    !local 
    INTEGER           :: N1, N2, NPL2
    INTEGER           :: I 
    GDEF              :: AUX(NB_TOT,NB_TOT)
    GDEF              :: BUX(NB_TOT,NB_TOT)
    GDEF              :: CUX(NB_TOT,NB_TOT)

    AUX=P
    CUX=P
    BUX=P
 
DO   I = 1, MIN(NORDER,4)  

    WRITE(*,'(A,I2)')' P^',I

    IF ( PRESENT( NDUMP ) .AND. PRESENT(NSTOP) ) THEN
       DO N1=NDUMP,NSTOP
          WRITE(*,1)N1,(REAL( AUX(N1,N2) ,KIND=q) ,N2=NDUMP,NSTOP)
       ENDDO
       WRITE(*,*)
#ifndef gammareal
       IF ( PRESENT (LPRINT ) ) THEN
       DO N1=NDUMP,NSTOP
          WRITE(6,2)N1,(AIMAG( AUX(N1,N2) ) ,N2=NDUMP,NSTOP)
       ENDDO
       WRITE(*,*)
       ENDIF
#endif
    ELSE IF ( PRESENT( NDUMP ) ) THEN
       NPL2=MIN(NDUMP,NB_TOT)
       DO N1=1,NPL2
          WRITE(*,1)N1,(REAL( AUX(N1,N2) ,KIND=q) ,N2=1,NPL2)
       ENDDO
       WRITE(*,*)

#ifndef gammareal
       IF ( PRESENT (LPRINT ) ) THEN
       DO N1=1,NPL2
          WRITE(6,2)N1,(AIMAG( AUX(N1,N2)),N2=1,NPL2)
       ENDDO
       WRITE(*,*)
       ENDIF 
#endif
    ELSE
       NPL2=MIN(12,NB_TOT)
       DO N1=1,NPL2
          WRITE(*,1)N1,(REAL( AUX(N1,N2) ,KIND=q) ,N2=1,NPL2)
       ENDDO
       WRITE(*,*)
#ifndef gammareal
       IF ( PRESENT (LPRINT ) ) THEN
       DO N1=1,NPL2
          WRITE(6,2)N1,(AIMAG( AUX(N1,N2)),N2=1,NPL2)
       ENDDO
       WRITE(*,*)
       ENDIF 
#endif
    ENDIF 

       CUX=TRANSPOSE(GCONJG(AUX))
       BUX=MATMUL(CUX,AUX)
       AUX=BUX
ENDDO 

1   FORMAT(1I3,3X,40F9.5)
2   FORMAT(1I3,3X,40F9.5)

  END SUBROUTINE DUMP_PROJ


!the following routines are for joint diagonalization 

!**********************************************************************
!
! diagonalizes a set of matrices as good as possible 
! if LSINGLE is true only matrix at k=NKTOT is diagonalized with 
! Jacobi algorithm. The matlab code is written by 
!  author       = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
!  journal      = "{SIAM} J. Mat. Anal. Appl.",
!  title        = "Jacobi angles for simultaneous diagonalization",
!  pages        = "161--164",
!  volume       = "17",
!  number       = "1",
!  month        = jan,
!  year         = {1996}
!
!**********************************************************************

   SUBROUTINE JOINT_DIAGONALIZER(D,MAT,THRESH,M,NKPTS,ISPIN,IU0,IU6,MAXITER,LSINGLE)
      USE prec
      USE constant
      IMPLICIT NONE
      GDEF :: D(:,:,:)     !matrix which diagonalizes all input matrices 
      GDEF :: MAT(:,:,:,:) !set of matrices
      REAL(q)    :: THRESH       !convergence criterium 
      INTEGER    :: M            !dimension of one matrix
      INTEGER    :: NKPTS        !number of matrices to be diagonalized
      INTEGER    :: ISPIN        !spin number
      INTEGER    :: IU0, IU6     !for writing
      INTEGER    :: MAXITER      !number of allowed iterations
      LOGICAL,OPTIONAL :: LSINGLE
      !local 
      INTEGER                :: I, J, K, ISP, ITER
      INTEGER                :: ALGOINFO 
      INTEGER                :: NM
      COMPLEX(q),ALLOCATABLE :: A(:,:,:)     !working array 
      COMPLEX(q),ALLOCATABLE :: AT1(:,:)     !working array 
      COMPLEX(q)             :: B(3,3)       !rotation matrix
      COMPLEX(q)             :: BT(3,3)      !the conjugate transpose of B
      INTEGER,ALLOCATABLE    :: IP(:)
      INTEGER,ALLOCATABLE    :: IQ(:)
      COMPLEX(q),ALLOCATABLE :: G(:,:)       !Givens rotation matrix
      COMPLEX(q),ALLOCATABLE :: GU(:,:)      !Givens rotation matrix
      COMPLEX(q)             :: VCP(3,3)
      COMPLEX(q)             :: DIA(3,3)
      INTEGER                :: KU(3)        !sort array 
      REAL(q)                :: KMAT(3,3)
      REAL(q)                :: ANGLES(3)
      INTEGER                :: PAIR(2)
      REAL(q)                :: C
      COMPLEX(q)             :: S
      COMPLEX(q),ALLOCATABLE :: V(:,:)       !diagonalizer matrix
      COMPLEX(q),ALLOCATABLE :: V1(:,:)      !rotated matrix
      COMPLEX(q),ALLOCATABLE :: V2(:,:)      !rotated matrix
      COMPLEX(q),ALLOCATABLE :: A1(:,:)      !rotated matrix
      COMPLEX(q),ALLOCATABLE :: A2(:,:)      !rotated matrix
      REAL(q)                :: LA(3)
      INTEGER                :: P,QI
      REAL(q)                :: RAUX(3,3)
      REAL(q)                :: SBAK
      LOGICAL                :: LCONTINUE 
      INTEGER                :: NKTOT
      
      IF ( PRESENT(LSINGLE) .AND. LSINGLE ) THEN
         NKTOT=1
      ELSE
         NKTOT=NKPTS
      ENDIF 

      
      !some standard test
      IF ( SIZE(MAT,1) /= SIZE(MAT,2) ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            'input matrices are not square!',SIZE(MAT,1),SIZE(MAT,2)
         STOP
      ENDIF 
      IF ( SIZE(D,1) /= SIZE(MAT,1) ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            ' sizes of matrices inconsistent!',SIZE(D,1),SIZE(MAT,1)
         STOP
      ENDIF 
      IF ( SIZE(D,1) /= SIZE(D,2) ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            ' diagonalizing matrix must be square!',SIZE(D,1),SIZE(D,2)
         STOP
      ENDIF 
      IF ( SIZE(D,3) < ISPIN ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            ' diagonalizing matrix must contain enough spin dimensions!',SIZE(D,3),ISPIN
         STOP
      ENDIF 
      IF ( SIZE(MAT,2) < M ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            'dimension of input matrices too small!',SIZE(MAT,2),M
         STOP
      ENDIF 
      IF ( SIZE(MAT,3) < NKTOT ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            'too few input matrices given!',SIZE(MAT,3),NKTOT
         STOP
      ENDIF 
      IF ( SIZE(MAT,4) < ISPIN ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from JOINT_DIAGONALIZER:'//&
            'too few input matrices given (ISPIN)!',SIZE(MAT,3),ISPIN
         STOP
      ENDIF 
    
      !A will be the working array 
      NM=NKTOT*M
      ALLOCATE(A(M,M,NKTOT))
      ALLOCATE(AT1(M,2*NKTOT))
      !initialize loop arrays
      ALLOCATE(IP(NKTOT),IQ(NKTOT))
      !Givens matrices
      ALLOCATE(G(3,NKTOT),GU(2,2))
      ALLOCATE(V(M,M)) 
      ALLOCATE(V1(M,2),V2(M,2)) 
      ALLOCATE(A1(2,NM),A2(2,NM)) 

      !----------------------------------------------------------------------
      spin : DO ISP=1, ISPIN 
      !----------------------------------------------------------------------
      
         A=(0._q,0._q)
         !resave input matrices 
         IF ( PRESENT(LSINGLE) .AND. LSINGLE ) THEN
            DO I=1, M
               DO J=1, M
#ifdef gammareal
                  A(I,J,NKTOT)=CMPLX(MAT(I,J,NKPTS,ISP),0._q,q)
#else
                  A(I,J,NKTOT)=MAT(I,J,NKPTS,ISP)
#endif 
               ENDDO
            ENDDO
         ELSE
            DO I=1, M
               DO J=1, M
                  DO K=1, NKTOT
#ifdef gammareal
                     A(I,J,K)=CMPLX(MAT(I,J,K,ISP),0._q,q)
#else
                     A(I,J,K)=MAT(I,J,K,ISP)
#endif
                  ENDDO
               ENDDO
            ENDDO
         ENDIF 
         !set B
         B(1,1)=(1._q,0._q)
         B(1,2)=(0._q,0._q)
         B(1,3)=(0._q,0._q)
         B(2,1)=(0._q,0._q)
         B(2,2)=(1._q,0._q)
         B(2,3)=(1._q,0._q)
         B(3,1)=(0._q,0._q)
         B(3,2)=(0._q,-1._q)
         B(3,3)=(0._q,1._q)
         BT=CONJG(TRANSPOSE(B))

         IP=0
         IQ=0
   
         G=(0._q,0._q)
         GU=(0._q,0._q)

         !initialize VCP and DIA
         VCP=(0._q,0._q)
         DIA=VCP
         LA=0
         KMAT=0._q
         ANGLES=0._q
         PAIR=0
         C=0._q
         S=(0._q,0._q)
          
         !initialize V to identity 
         V=(0._q,0._q)
         DO I=1,M
            V(I,I)=(1._q,0._q)
         ENDDO
      
         !----------------------------------------------------------------------  
         minloop: DO ITER = 1, MAXITER
         !----------------------------------------------------------------------  
            LCONTINUE=.FALSE.  !sould we continue 
            !p is the row index, q is the column index
            DO P=1,M-1
               DO QI=P+1,M

                  !we consider 3x3 submatrices 
                  DO I = 1, NKTOT
                     IP(I)=P+(I-1)*M
                     IQ(I)=QI+(I-1)*M
                  ENDDO
                  !compute the Givens angles
                  DO I=1,NKTOT
                     G(1,I)=A(P,IP(I),1)-A(QI,IQ(I),1) 
                     G(2,I)=A(P,IQ(I),1) 
                     G(3,I)=A(QI,IP(I),1) 
                  ENDDO
                  RAUX=REAL(MATMUL(B,MATMUL(MATMUL(G,CONJG(TRANSPOSE(G))),BT)),q)

                  !get the eigensystem
                  CALL RSYS_EIGENSYSTEM(RAUX,VCP,LA,IU0)
                  !maybe sort eigenvalues (LAPACK should do this anyhow)
                  KU(1)=1; KU(2)=2; KU(3)=3 
 
                  !define angles
                  ANGLES(1:3)=REAL(VCP(1:3,KU(3)),q)
                  
                  !maybe the first angle is negative, then multiply by (-1)
                  IF ( ANGLES(1) < 0 ) ANGLES = - ANGLES
                  
                  C= SQRT(0.5_q+ANGLES(1)/2._q)
                  S=0.5_q*( ANGLES(2)- (0._q,1._q)*ANGLES(3))/C              
                  !------------------------------------------------------------
                  !convergence test 
                  IF ( ABS(S) > THRESH) THEN
                     LCONTINUE = .TRUE.  !continue iterations in this case 
                     !if test not passed -> update A and V by a Givens rotation
                     PAIR(1) = P
                     PAIR(2) = QI
                     !Givens rotation matrix 
                     GU(1,1) = CMPLX(C,0._q,q)
                     GU(1,2) = -CONJG(S) 
                     GU(2,1) = S 
                     GU(2,2) = CMPLX(C,0._q,q)
 
                     !rotate relevant part of V 
                     DO I =1 ,2
                        V1(1:M,I)=V(1:M,PAIR(I))
                     ENDDO
                     V2 = MATMUL(V1,GU)
                     DO I =1 ,2
                        V(1:M,PAIR(I))=V2(1:M,I)
                     ENDDO

                     !rotate A as well
                     DO I =1 ,2
                        A1(I,1:NM)=A(PAIR(I),1:NM,1)
                     ENDDO
                     A2 = MATMUL(CONJG(TRANSPOSE(GU)),A1)
                     DO I =1 ,2
                        A(PAIR(I),1:NM,1)=A2(I,1:NM)
                     ENDDO
                    
                     !save relevant part of A temporarily to ATMP
                     AT1=(0._q,0._q)
                     DO I = 1, M
                        DO J = 1, 2*NKTOT
                           IF ( J <= NKTOT ) THEN
                              AT1(I,J)=C*A(I,IP(J),1)+S*A(I,IQ(J),1)
                           ELSE
                              AT1(I,J)=-CONJG(S)*A(I,IP(J-NKTOT),1)+C*A(I,IQ(J-NKTOT),1)
                           ENDIF 
                        ENDDO
                     ENDDO
                     !overwrite these matrix elements of A by AT1
                     DO I = 1, M
                        DO J = 1, 2*NKTOT
                           IF ( J <= NKTOT ) THEN
                              A(I,IP(J),1) = AT1(I,J)  
                           ELSE
                              A(I,IQ(J-NKTOT),1) = AT1(I,J)  
                           ENDIF 
                        ENDDO
                     ENDDO
                  ENDIF   !update necessary?
                  !----------------------------------------------------------
               ENDDO  !columns
            ENDDO  !rows

            !alternative breaking contition for single diagonalizations
            IF ( PRESENT(LSINGLE) .AND. LSINGLE ) THEN
               IF ( ITER == 1 ) THEN
                  SBAK=ABS(S) 
               ELSE
                  SBAK=ABS(ABS(S)-SBAK) 
               ENDIF 
               IF ( SBAK < THRESH ) THEN
                  LCONTINUE=.FALSE.              
               ELSE
                  SBAK=ABS(S)
               ENDIF
            ENDIF

            !breaking condition    
            IF ( ITER < MAXITER ) THEN
               IF ( PRESENT(LSINGLE) .AND. LSINGLE ) THEN
               ELSE
                  !dump convergence info 
                  IF ( IU0 >= 0 ) THEN
                     WRITE(IU0,'(A,I4,A,E13.6)')' iteration: ', ITER, ' error ',ABS(S)
                  ENDIF 
               ENDIF 
               IF ( .NOT. LCONTINUE ) THEN
                  ALGOINFO=1
                  EXIT minloop
               ENDIF 
            ELSE
               IF ( IU0>=0 ) WRITE(IU0,'(A,I5,A,E13.4)') 'WARNING from JOINT_DIAGONALIZER:'//&
                  'not converged after ',MAXITER,' iterations. ERROR=',ABS(S)
                  ALGOINFO=-1
            ENDIF 
            

         !-------------------------------------------------------------------
         ENDDO minloop  
         !-------------------------------------------------------------------

         !save the result
         !resave input matrices 
         IF ( PRESENT(LSINGLE) .AND. LSINGLE ) THEN
            DO I=1, M
               DO J=1, M
#ifdef gammareal
                  MAT(I,J,NKPTS,ISP)=REAL(A(I,J,NKTOT),q)
#else
                  MAT(I,J,NKPTS,ISP)=A(I,J,NKTOT)
#endif
               ENDDO
            ENDDO
            !dump convergence info 
            !IF ( IU0 >= 0 ) THEN
            !   WRITE(IU0,'(A,I4,A,E13.6)')'K-Point ',NKPTS,' done E=',ABS(S)
            !ENDIF 
         ELSE
            DO I=1, M
               DO J=1, M
                  DO K=1, NKTOT
#ifdef gammareal
                     MAT(I,J,K,ISP)=A(I,J,K)
#else
                     MAT(I,J,K,ISP)=REAL(A(I,J,K),q)
#endif
                  ENDDO
               ENDDO
            ENDDO
         ENDIF
         !V is the diagonalizer 
         DO I=1, M
            DO J=1, M
#ifdef gammareal
               D(I,J,ISP)=REAL(V(I,J),q)
#else
               D(I,J,ISP)= V(I,J)
#endif
            ENDDO
         ENDDO

#ifdef debug
WRITE(*,*)'Diagonalizer for spin ',ISP
DO I = 1, M
   WRITE(*,'(12F14.4)') REAL(D(I,1:M,ISP),q)
ENDDO
WRITE(*,*)' Imaginary part'
DO I = 1, M
   WRITE(*,'(12F14.4)') REAL((0._q,-1._q)*D(I,1:M,ISP),q)
ENDDO
K=3
WRITE(*,*)'NK=',K
DO I = 1, M
   WRITE(*,'(12F20.16)') REAL(MAT(I,1:M,K,ISP),q)
ENDDO
WRITE(*,*)' Imaginary part'
DO I = 1, M
   WRITE(*,'(12F20.16)') REAL((0._q,-1._q)*MAT(I,1:M,K,ISP),q)
ENDDO
#endif 

      !----------------------------------------------------------------------
      ENDDO spin 
      !----------------------------------------------------------------------
      RETURN
   ENDSUBROUTINE JOINT_DIAGONALIZER


!**********************************************************************
!
! gets the eigensystem of real symmetric matrix 
!
!**********************************************************************
    
   SUBROUTINE RSYS_EIGENSYSTEM(C,EV,E,IU0)
      REAL(q)                :: C(:,:)
      COMPLEX(q)             :: EV(:,:)
      REAL(q)                :: E(:)
      INTEGER                :: IU0
      !local
      INTEGER                :: M, N
      INTEGER                :: LDC
      INTEGER                :: I, J
      INTEGER,PARAMETER      :: LWMAX=10000
      INTEGER                :: INFO, LWORK
      REAL(q),ALLOCATABLE    :: RWORK(:) 
      REAL(q),ALLOCATABLE    :: CC(:,:)            
      REAL(q),ALLOCATABLE    :: W(:)            
      REAL(q),ALLOCATABLE    :: WORK(:)            

      !get the size of the matrix and the LDs
      M=SIZE(C,2)
      N=SIZE(C,1)
      LDC = M
    
      IF ( SIZE(EV,1) /= N ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from RSYS_EIGENSYSTEM:'//&
            'EV not proper size',SIZE(EV,1),N
         STOP
      ENDIF 
      IF ( SIZE(E) /= N ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from RSYS_EIGENSYSTEM:'//&
            'E not proper size',SIZE(E,1),N
         STOP
      ENDIF 
      IF ( M /= N ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from RSYS_EIGENSYSTEM:'//&
            'matrix not square',M,N
         STOP
      ENDIF 

      !we work in CC
      ALLOCATE(CC(N,M))
      CC=C

      ALLOCATE(RWORK(2*LDC))
      ALLOCATE(W(LDC), WORK(LWMAX))

      ! Query the optimal workspace.
      LWORK = -1
      CALL DSYEV( 'Vectors', 'Upper', N, CC, LDC, W, WORK, LWORK, INFO )
      LWORK = MIN( LWMAX, INT( WORK( 1 ) ) )

      ! solve the problem 
      CALL DSYEV( 'Vectors', 'Upper', N, CC, LDC, W, WORK, LWORK, INFO )

      ! Check for convergence.
      IF( INFO /= 0 ) THEN
         IF ( IU0>=0) WRITE(IU0,'(A,2I4)')'internal error in VASP from RSYS_EIGENSYSTEM:'//&
            'DSYEV returns',INFO
         STOP
      ENDIF
       
      !save eigenvectors
      DO I = 1, M
         DO J = 1, M
            EV(I,J)=CMPLX(CC(I,J),0._q,q)
         ENDDO
         E(I)=W(I)
      ENDDO
       
      !get rid of auxilary arrays
      DEALLOCATE(CC)
      DEALLOCATE(RWORK,W,WORK)
      RETURN
   ENDSUBROUTINE  RSYS_EIGENSYSTEM

END MODULE crpa
