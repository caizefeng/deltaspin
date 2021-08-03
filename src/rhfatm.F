!#define debug
#include "symbol.inc"
!***********************************************************************
!***********************************************************************
! This module (rhfatm) contains the routines that adapt the PAW datasets
! to be used in Hartree-Fock calculations, basically by replacing the
! DFT frozen core by its HF counterpart.
! The main public module subroutine is RHFATM_SET_PP, that is called by
! SET_PAW_ATOM_POT_RHF (the HF counterpart of SET_PAW_ATOM_POT).
! The latter is called from main.F if LRHFATM=.TRUE. in the INCAR file.
!***********************************************************************
!***********************************************************************

      MODULE rhfatm
      USE prec
      USE radial

      PRIVATE :: DFATOM,DFSOLVE,DF_PAR_WAVE, &
     &   HFATOM,HFSOLVE,HF_PAR_WAVE,HF_PAR_WAVE_v1,HF_PAR_WAVE_v2, &
     &   SET_DIJ,ORTHO_PAR_WAVE,SET_PP,HF_CORE_VALENCE_EXCHANGE, &
     &   RAD_POT_DFT,RAD_POT_DIRECT,RAD_POT_DIRECT_VALENCE_ONLY, &
     &   RAD_POT_EXCHANGE,RAD_POT_EXCHANGE_SINGLE, &
     &   INITWF,WMIX,WSHOOT,PWSHOOT,ALLOCW,DEALLOCW,OUTINT,INWINT,REPORT, &
     &   IS_BOUND,N_OF_CHANNEL,IR_MATCH,IR_END,IR_CORE,IR_SWITCH,IR_WMAX, &
     &   RCRCUT

      TYPE atomic
         ! Number of occupied nl-subshells
         INTEGER NS
         ! Number of core nl-subshells
         INTEGER NSCORE
         ! Number of valence nl-subshells
         INTEGER NSVAL
         ! Total number of electrons
         REAL(q) Z
         ! Number of core electrons
         REAL(q) ZCORE
         ! Number of valence electrons
         REAL(q) ZVAL
         ! Eigenenergy
         REAL(q), POINTER :: E(:)
         ! Main quantum number
         INTEGER, POINTER :: N(:)
         ! Angular momentum quantum number
         INTEGER, POINTER :: L(:)
         ! Occupation of subshell nl
         REAL(q), POINTER :: OCC(:)
         ! Wavefunctions, large and small components
         REAL(q), POINTER :: W(:,:),A(:,:),B(:,:)
         ! Action of the kinetic energy operator on the wavefunctions
         REAL(q), POINTER :: WKINAE(:,:)
         ! Partial waves and linearization energies
         REAL(q), POINTER :: WAE(:,:),WPS(:,:),EPS(:)
         ! Action of the kinetic energy operator on the partial waves
         REAL(q), POINTER :: PARWKINAE(:,:),PARWKINPS(:,:)
         ! Projector functions
         REAL(q), POINTER :: BETA(:,:)
         ! AE and PS (partial) core charge density
         REAL(q), POINTER :: RHOAE(:),RHOPS(:)
         ! local atomic reference potential (valence only)
         REAL(q), POINTER :: POTAE(:), POTPS(:)
         ! AE and PS core potential (V_H[n_c+N_Z] and V_H[\tilde{n}_Zc])
         REAL(q), POINTER :: POTAEC(:),POTPSC(:)
         ! Atomic reference energy
         REAL(q) EATOM
         ! Radial grid
         TYPE (rgrid) :: R
         REAL(q), POINTER :: RHO(:)
      END TYPE atomic

      ! read from INCAR, switches on the RHFATM routines
      LOGICAL, PRIVATE, SAVE :: LuseRHFATM=.FALSE.

      ! Light speed in atomic units
      REAL(q), PRIVATE, SAVE :: C=137.037_q
!     REAL(q), PRIVATE, SAVE :: C=1.E+8

      ! Minimal radial grid size (in Angstroem)
      REAL(q), PARAMETER, PRIVATE :: RMAX=250._q

      ! Sets an upper bound on the amplitude of the tails of the partial waves
      REAL(q), PRIVATE, SAVE :: maxParWaveAmplitude=10._q

      ! The partial waves are integrated outwards at least up to
      REAL(q), PRIVATE, SAVE :: minParWaveRadius=0._q

      ! Switch off the exchange interactions in the small components?
!     LOGICAL, PRIVATE, SAVE :: noSmallExchange = .True.
      LOGICAL, PRIVATE, SAVE :: noSmallExchange = .False.

      CONTAINS

!******************** SUBROUTINE ***************************************
!
! Template for public subroutines
!
!***********************************************************************
!     SUBROUTINE RHFATM_*()
!     USE prec
!     IMPLICIT NONE
!
!     RETURN
!     END SUBROUTINE RHFATM_*


!***********************************************************************
!
! RHFATM_TEST is a nifty little routine that
!
!***********************************************************************
      SUBROUTINE RHFATM_TEST(PP,W,WDES,LATT_CUR,IO)
      USE prec
      USE base
      USE pseudo
      USE lattice
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(wavedes) WDES 
      TYPE(potcar) PP
      TYPE(latt) LATT_CUR
      TYPE(in_struct) IO
      ! local variables
      TYPE(wavefun1) W1
      TYPE(wavedes1) WDES1
      INTEGER NB_GLB,NB_GLB_STRT,NB_GLB_STOP,NB,K
      INTEGER LOW,LHI,LL,L,LP,M,MMAX,LMIND,LMIND_,LMBASE 
      REAL(q) RHOPS(WDES%GRID%MPLWV),RHO1AE(PP%R%NMAX),RHO1PS(PP%R%NMAX)

      CALL SETWDES(WDES,WDES1,1)
      CALL NEWWAV(W1,WDES1,.TRUE.)

      NB_GLB_STRT=1
      NB_GLB_STOP=5

      RHOPS=0
      DO NB_GLB=NB_GLB_STRT,NB_GLB_STOP
         NB=NB_LOCAL(NB_GLB,WDES1)
         IF (NB==0) CYCLE         
         CALL W1_COPY(ELEMENT(W,WDES1,NB,1),W1)
         CALL FFTWAV_W1(W1)
         RHOPS(:)=RHOPS(:)+W1%CR(:)*W1%CR(:)
      ENDDO

      CALLMPI( M_sum_d(WDES1%COMM_INTER,RHOPS(1),WDES1%GRID%MPLWV) )

      RHOPS=RHOPS/(NB_GLB_STOP-NB_GLB_STRT+1) 

      IF (IO%IU6>0) THEN
         OPEN(1000)
         DO K=1,WDES1%GRID%NGX
            WRITE(1000,'(2F20.10)') (K-1)*LATT_CUR%ANORM(1)/WDES1%GRID%NGX, &
           &   RHOPS(K)/LATT_CUR%OMEGA
         ENDDO
         CLOSE(1000)
      ENDIF

      DO NB_GLB=1,WDES%NB_TOT
         NB=NB_LOCAL(NB_GLB,WDES1)
         IF (NB==0) CYCLE 
         LMBASE =0
         LOW=1
         RHO1AE=0; RHO1PS=0
         block: DO
            LL=PP%LPS(LOW)
            DO LHI=LOW,PP%LMAX
               IF (LL/=PP%LPS(LHI)) EXIT
            ENDDO
            LHI=LHI-1

            MMAX=2*LL+1

            DO L =LOW,LHI
            DO LP=LOW,LHI

            DO M=1,MMAX
               LMIND  =LMBASE +(L -LOW) *MMAX+M
               LMIND_ =LMBASE +(LP-LOW) *MMAX+M
               RHO1AE(:)=RHO1AE(:) &
              &   +W%CPROJ(LMIND,NB,1,1)*PP%WAE(:,L)*W%CPROJ(LMIND_,NB,1,1)*PP%WAE(:,LP)
               RHO1PS(:)=RHO1PS(:) &
              &   -W%CPROJ(LMIND,NB,1,1)*PP%WPS(:,L)*W%CPROJ(LMIND_,NB,1,1)*PP%WPS(:,LP)
            ENDDO

            ENDDO
            ENDDO

            LMBASE =LMBASE +(LHI-LOW+1)*MMAX
            LOW=LHI+1
            IF (LOW > PP%LMAX) EXIT block
         ENDDO block

         OPEN(1000+NB_GLB)
         WRITE(1000+NB_GLB,'(3F20.10)') (PP%R%R(K),RHO1AE(K),RHO1PS(K),K=1,PP%R%NMAX)
         CLOSE(1000+NB_GLB)
      ENDDO

      RETURN
      END SUBROUTINE RHFATM_TEST


!******************** SUBROUTINE RHFATM_READER *************************
!
! RHFATM_READER reads the keywords LRHFATM and MAXPWAMP from INCAR.
! If LRHFATM=.TRUE. the DFT core of the PAW dataset will be replaced
! by its Hartree-Fock counterpart.
! MAXPWAMP (=10 by default) sets the maximal absolute amplitude the
! divergent partial waves are allowed to reach (beyond that they are
! set to zero).
!
!***********************************************************************
      SUBROUTINE RHFATM_READER(IO)
      USE prec
      USE base
      USE vaspxml
      IMPLICIT NONE
      TYPE(in_struct) IO
      ! local variables
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC
      CHARACTER (40)   SZNAM

      LOPEN=.FALSE.
      OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LRHFATM','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LuseRHFATM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LRHFATM'' from file INCAR.'
         LuseRHFATM=.FALSE.
      ENDIF

      CALL XML_INCAR('LRHFATM','L',IDUM,RDUM,CDUM,LuseRHFATM,CHARAC,N)

      IF (LuseRHFATM) THEN
         CALL RDATAB(LOPEN,INCAR,IO%IU5,'MAXPWAMP','=','#',';','F', &
        &            IDUM,maxParWaveAmplitude,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''MAXPWAMP'' from file INCAR.'
            maxParWaveAmplitude=10._q
         ENDIF
         CALL XML_INCAR('MAXPWAMP','F',IDUM,maxParWaveAmplitude,CDUM,LDUM,CHARAC,N)
      ENDIF

      CLOSE(IO%IU5)

      RETURN
      END SUBROUTINE RHFATM_READER


!******************** SUBROUTINE RHFATM_WRITER *************************
!
! Writes the pertinent INCAR-keys to OUTCAR
!
!***********************************************************************
      SUBROUTINE RHFATM_WRITER(IO)
      USE prec
      USE base
      IMPLICIT NONE
      TYPE(in_struct) IO
      ! local variables
      IF (IO%IU6>0) THEN
         WRITE(IO%IU6,10) LuseRHFATM,maxParWaveAmplitude 
      ENDIF
  10  FORMAT(/' Adapt original PAW datasets for Hartree-Fock calculations',/&
     &        ' ---------------------------------------------------------',/&
     &        '   LRHFATM =',L6,/&
     &        '  MAXPWAMP =',F8.1) 
      RETURN
      END SUBROUTINE RHFATM_WRITER


!******************** SUBROUTINE RHFATM_SET_PP *************************
!
! RHFATM_SET_PP is the main public subroutine of the rhfatm module.
! This subroutine is called for each PAW dataset in the POTCAR file.
! It first solves the DFT and HF spherical nonspinpolarized atoms
! (DFATOM and HFATOM respectively). Subsequently the DFT and HF partial
! waves are (re)calculated (in DF_PAR_WAVE and HF_PAR_WAVE).
! N.B.: presently the HF partial waves are take to be equal to the
! original DFT partial waves.
! These are orthogonalized to the HF core states (ORTHO_PAR_WAVE).
! Subsequently the PAW parameters DIJ and QIJ are calculated (SET_PP),
! and the Hartree-Fock core-valence exchange is added to the PAW
! strength parameters DIJ (HF_CORE_VALENCE_EXCH).
! The HF atomic reference energy is stored in PP%EATOM.
!
!***********************************************************************
      SUBROUTINE RHFATM_SET_PP(PP,IO)
      USE prec
      USE base
      USE pseudo
!     USE radial
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(in_struct) IO
      ! local variables
      TYPE(atomic) WDF,WHF

      ! Solve the atomic problem using DFT
      CALL DFATOM(PP,WDF,IO)

      ! Solve the atomic problem at the HF level
!     CALL HFATOM(PP,WHF,IO)
      CALL HFATOM(PP,WHF,IO,W0=WDF)

!     IF (PP%R%R(PP%R%NMAX)<WHF%R%R(IR_CORE(WHF))) THEN
!        WRITE(*,'(A,X,A2,2(X,F14.7))') &
!       &   'RHFATM_SET_PP: ERROR, radial grid on POTCAR to small:', &
!       &   PP%ELEMENT,PP%R%R(PP%R%NMAX),WHF%R%R(IR_CORE(WHF))
!        STOP
!     ENDIF

      minParWaveRadius=WHF%R%R(IR_CORE(WHF))

      ! Recompute the DFT partial waves
      CALL DF_PAR_WAVE(PP,WDF,IO)

      ! Set the HF partial waves
      CALL HF_PAR_WAVE(PP,WDF,WHF,IO)
!     CALL HF_PAR_WAVE_v3(PP,WDF,WHF,IO)

      ! Orthogonalize the HF partial waves to the HF core states
      CALL ORTHO_PAR_WAVE(PP,WHF,WHF,IO)

      ! Compute the PAW parameters
!     WHF%RHOAE=WDF%RHOAE
      CALL SET_PP(PP,WDF,WHF,IO)

      ! Add the core-valence exchange to the PP strength paramters
      CALL HF_CORE_VALENCE_EXCH(PP,WHF,IO)

      ! Copy the energy of the atomic valence electronic configuration
      ! to the PP data structure (Beware of the sign: in VASP the total
      ! energy is specified w.r.t. the atomic energy)
      PP%EATOM=-WHF%EATOM

      ! Copy the core charge to the PP data structure
      PP%ZCORE=WDF%ZCORE

      CALL DEALLOCW(WDF)
      CALL DEALLOCW(WHF)

      RETURN
      END SUBROUTINE RHFATM_SET_PP


!******************** SUBROUTINE RHFATM_DEALLOCW ***********************
!
!***********************************************************************
      SUBROUTINE RHFATM_DEALLOCW(W)
      IMPLICIT NONE
      TYPE(atomic) W
      CALL DEALLOCW(W)
      RETURN
      END SUBROUTINE RHFATM_DEALLOCW


!******************** SUBROUTINE RHFATM_CROP_PSEUDO ********************
!
!***********************************************************************
      SUBROUTINE RHFATM_CROP_PSEUDO(P,T_INFO,LOVERL,IO)
      USE prec
      USE base
      USE poscar
      USE pseudo
      USE vaspxml
      IMPLICIT NONE
      TYPE(type_info) T_INFO
      TYPE(potcar), TARGET :: P(T_INFO%NTYP)
      TYPE(in_struct) IO
      LOGICAL LOVERL
      ! local variables
      TYPE(atomic) WDF
      TYPE (potcar), POINTER:: PP
      INTEGER NT,I,CHANNEL
      INTEGER NMAX,NMAXP
      REAL(q) F,FDER,RPROJ,RCUT
      REAL(q), ALLOCATABLE :: RTMP(:,:)

      INTEGER IU0,IU5
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM

      REAL(q) ESEMICORE
      REAL(q), ALLOCATABLE :: RCRHOCUT(:)
      LOGICAL LRELCOR,LADAPTELIN
      LOGICAL, ALLOCATABLE :: LRCTYPE(:)

      ! Quick return if possible
      IF (.NOT.LOVERL) RETURN

      ! Read relevant tags from the INCAR file
      LOPEN=.FALSE.
      IU0=IO%IU0; IU5=IO%IU5
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
! LRELCOR: do we want to perform a relaxed core calculation?
      LRELCOR=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LRELCOR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRELCOR,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRELCOR'' from file INCAR.'
         GOTO 150
      ENDIF

      IF (.NOT.LRELCOR) RETURN
      
! LRCTYPE: which types will be treated with the relaxed core method
      ALLOCATE(LRCTYPE(T_INFO%NTYP))
      LRCTYPE=.TRUE.
      ! read which types are submitted to core relaxation
      CALL RDATAB(LOPEN,INCAR,IU5,'LRCTYPE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRCTYPE,CHARAC,N,T_INFO%NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<T_INFO%NTYP))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRCTYPE'' from file INCAR.'
         GOTO 150
      ENDIF

! RCRHOCUT: defines the extent of the core
      ALLOCATE(RCRHOCUT(T_INFO%NTYP))
      RCRHOCUT=1E-5
      CALL RDATAB(LOPEN,INCAR,IU5,'RCRHOCUT','=','#',';','F', &
     &            IDUM,RCRHOCUT,CDUM,LDUM,CHARAC,N,T_INFO%NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<T_INFO%NTYP))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''RCRHOCUT'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('RCRHOCUT','F',IDUM,RCRHOCUT,CDUM,LDUM,CHARAC,N)

! LADAPTELIN: do we want to adapt the linearization energies to iron
! out divergencies and extra nodes in the AE partial waves
      LADAPTELIN=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LADAPTELIN','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LADAPTELIN,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LADAPTELIN'' from file INCAR.'
         GOTO 150
      ENDIF

! ESEMICORE: defines the semicore states (Default: -2 Ry)
      ESEMICORE=-27.211652
      CALL RDATAB(LOPEN,INCAR,IU5,'ESEMICORE','=','#',';','F', &
     &            IDUM,ESEMICORE,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ESEMICORE'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('ESEMICORE','F',IDUM,ESEMICORE,CDUM,LDUM,CHARAC,N)

      ! finished reading from INCAR
      CLOSE(IU5)      
      
      DO NT=1,T_INFO%NTYP
         IF (.NOT.LRCTYPE(NT)) CYCLE
         PP=> P(NT)
         ! define the extent of the core
         RCUT=RCRCUT(RCRHOCUT(NT),ESEMICORE,PP,IO)

         ! check at which radius the projectors are all zero
         ALLOCATE(RTMP(P(NT)%R%NMAX,P(NT)%LMAX))
         RTMP=0
         chann: DO CHANNEL=1,P(NT)%LMAX
            IF (P(NT)%LPS(CHANNEL)>P(NT)%LMAX_CALC) CYCLE chann
            ! transfer projectors onto radial grid
            ! n.b. we store r*P(r)
            grid: DO I=1,P(NT)%R%NMAX
               IF (P(NT)%R%R(I)>P(NT)%PSPRNL(NPSRNL,1,1)) EXIT grid
               CALL SPLVAL(P(NT)%R%R(I),F,FDER,P(NT)%PSPRNL(1,1,CHANNEL),NPSRNL,NPSRNL)                  
               RTMP(I,CHANNEL)=F*P(NT)%R%R(I)
            ENDDO grid
         ENDDO chann
         RPROJ=0._q
         DO CHANNEL=1,P(NT)%LMAX
            DO I=P(NT)%R%NMAX-8,1,-1
               IF (RTMP(I,CHANNEL)/=0._q) EXIT
            ENDDO
            RPROJ=MAX(RPROJ,P(NT)%R%R(I+8))
         ENDDO
         DEALLOCATE(RTMP)
         
         IF (RCUT<0) RCUT=P(NT)%R%REND
         IF (RCUT<P(NT)%R%RMAX) THEN
            RCUT=P(NT)%R%RMAX
         ENDIF
         IF (RCUT<RPROJ) THEN
            RCUT=RPROJ
         ENDIF
         ! and the corresponding grid point
         DO NMAXP=1,P(NT)%R%NMAX
            IF (P(NT)%R%R(NMAXP)>RCUT) EXIT
         ENDDO
         NMAXP=NMAXP !-1
         ! the grid must have an uneven number of points
         NMAXP=MIN(NMAXP+(MODULO(NMAXP,2)-1),PP%R%NMAX)
         ! crop the relevant arrays
         NMAX=P(NT)%R%NMAX
         ALLOCATE(RTMP(NMAX,6))
         RTMP=0
         RTMP(1:NMAX,1)=P(NT)%POTAE(1:NMAX)
         RTMP(1:NMAX,2)=P(NT)%POTPS(1:NMAX)
         RTMP(1:NMAX,3)=P(NT)%POTPSC(1:NMAX)
         RTMP(1:NMAX,4)=P(NT)%RHOAE(1:NMAX)
         RTMP(1:NMAX,5)=P(NT)%RHOPS(1:NMAX)
         RTMP(1:NMAX,6)=P(NT)%TAUAE(1:NMAX)

         NULLIFY(P(NT)%POTAE,P(NT)%POTPS,P(NT)%POTPSC,P(NT)%RHOAE,P(NT)%RHOPS,P(NT)%TAUAE)
         ALLOCATE(P(NT)%POTAE(NMAXP),P(NT)%POTPS(NMAXP),P(NT)%POTPSC(NMAXP), &
        &   P(NT)%RHOAE(NMAXP),P(NT)%RHOPS(NMAXP),P(NT)%TAUAE(NMAXP))
                 
         P(NT)%POTAE(1:NMAXP) =RTMP(1:NMAXP,1)
         P(NT)%POTPS(1:NMAXP) =RTMP(1:NMAXP,2)
         P(NT)%POTPSC(1:NMAXP)=RTMP(1:NMAXP,3)
         P(NT)%RHOAE(1:NMAXP) =RTMP(1:NMAXP,4)
         P(NT)%RHOPS(1:NMAXP) =RTMP(1:NMAXP,5)
         P(NT)%TAUAE(1:NMAXP) =RTMP(1:NMAXP,6)
         
         DEALLOCATE(RTMP)
         
         N=SIZE(P(NT)%WAE,2)
         ALLOCATE(RTMP(NMAX,2*N))
         RTMP=0
         DO I=1,N
            RTMP(1:NMAX,I)=P(NT)%WAE(1:NMAX,I)
            RTMP(1:NMAX,I+N)=P(NT)%WPS(1:NMAX,I)
         ENDDO
         NULLIFY(P(NT)%WAE,P(NT)%WPS)
         ALLOCATE(P(NT)%WAE(NMAXP,N),P(NT)%WPS(NMAXP,N))
         DO I=1,N
            P(NT)%WAE(1:NMAXP,I)=RTMP(1:NMAXP,I)
            P(NT)%WPS(1:NMAXP,I)=RTMP(1:NMAXP,I+N)
         ENDDO
         DEALLOCATE(RTMP)
         
         N=SIZE(P(NT)%AUG,2)-1
         ALLOCATE(RTMP(NMAX,0:N))
         RTMP=0
         RTMP(1:NMAX,0:N)=P(NT)%AUG(1:NMAX,0:N)
         NULLIFY(P(NT)%AUG)
         ALLOCATE(P(NT)%AUG(NMAXP,0:N))
         P(NT)%AUG(1:NMAXP,0:N)=RTMP(1:NMAXP,0:N)
         DEALLOCATE(RTMP)
         
         ! and crop RCP(NTP)%R accordingly
         P(NT)%R%NMAX=NMAXP
         NULLIFY(P(NT)%R%R,P(NT)%R%SI)
         ALLOCATE(P(NT)%R%R(NMAXP),P(NT)%R%SI(NMAXP))
         DO I=1,NMAXP
            P(NT)%R%R(I)=P(NT)%R%RSTART*EXP(P(NT)%R%H*(I-1))
         ENDDO
         P(NT)%R%REND=P(NT)%R%R(NMAXP)
         CALL SET_SIMP(P(NT)%R)         
      ENDDO

      ! write some stuff to the OUTCAR file
      IF (IO%IU6>0) THEN
         IF (LRELCOR) THEN
            WRITE(IO%IU6,'(/A)') ' Cropping the radial grids:'
            WRITE(IO%IU6,'(A)') '  Typ   RC    r(nmax)   RCRHOCUT'
            DO I=1,T_INFO%NTYP
               IF (.NOT.LRCTYPE(I)) THEN
                  WRITE(IO%IU6,'(X,I4,L6,F10.4)') I,LRCTYPE(I),P(I)%R%R(P(I)%R%NMAX)
               ELSE
                  WRITE(IO%IU6,'(X,I3,L6,F10.4,3X,E10.4)') I,LRCTYPE(I),P(I)%R%R(P(I)%R%NMAX),RCRHOCUT(I)
               ENDIF
            ENDDO
            WRITE(IO%IU6,'(/2X,A,F10.4/)') 'ESEMICORE =',ESEMICORE
         ENDIF
      ENDIF      

      RETURN
      
  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP
       
      END SUBROUTINE RHFATM_CROP_PSEUDO


!******************** SUBROUTINE RHFATM_EXTEND_PSEUDO ******************
!
!***********************************************************************
      SUBROUTINE RHFATM_RECALC_PSEUDO(P,T_INFO,LOVERL,IO)
      USE prec
      USE base
      USE poscar
      USE pseudo
      IMPLICIT NONE
      TYPE(type_info) T_INFO
      TYPE(potcar), TARGET :: P(T_INFO%NTYP)
      TYPE(in_struct) IO
      LOGICAL LOVERL
      ! local variables
      TYPE(atomic) WDF
      TYPE (potcar), POINTER:: PP
      INTEGER NT

      IF (.NOT.LOVERL) RETURN
      DO NT=1,T_INFO%NTYP
         PP=> P(NT)
         
         ! Solve the atomic problem using DFT
         CALL DFATOM(PP,WDF,IO)
         
         PP%POTPSC(:)=PP%POTPSC(:)-(PP%POTPSC(PP%R%NMAX)-WDF%POTAEC(PP%R%NMAX))
!        PP%POTPSC(IR_PSMAX(PP):PP%R%NMAX)=WDF%POTAEC(IR_PSMAX(PP):PP%R%NMAX)
         
         ! Recompute the DFT partial waves
         minParWaveRadius=WDF%R%R(IR_CORE(WDF))
         CALL DF_PAR_WAVE(PP,WDF,IO)

         ! Compute the PAW parameters
         CALL SET_PP(PP,WDF,WDF,IO)

         ! Cleanup         
         CALL DEALLOCW(WDF)
      ENDDO
       
      RETURN
      END SUBROUTINE RHFATM_RECALC_PSEUDO


!******************** SUBROUTINE RHFATM_DFATOM *************************
!
! just a wrapper for a completely nomverbose call to DFATOM
!
!***********************************************************************
      SUBROUTINE RHFATM_DFATOM(PP,W)
      USE pseudo
      USE base
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      ! local variables
      TYPE(in_struct) IODUM

      IODUM%IU6=-1; IODUM%IU0=-1
      CALL DFATOM(PP,W,IODUM)

      RETURN
      END SUBROUTINE RHFATM_DFATOM


!******************** FUNCTION LRHFATM *********************************
!
! LRHFATM returns .TRUE. if LRHFATM=.TRUE. was read from the INCAR
! by RHFATM_READER.
!
!***********************************************************************
      FUNCTION LRHFATM()
      IMPLICIT NONE
      LOGICAL LRHFATM
      LRHFATM=LuseRHFATM
      END FUNCTION LRHFATM


!c tb beg
     SUBROUTINE remove_electron(NDIM,OCC,E)
     !c decrease occupancy of the highest occupied shell
     !c so as to prepare cationic state
       INTEGER :: NDIM,i,j
       REAL(q) :: OCC(NDIM)
       REAL(q) :: E(NDIM)
       REAL(q) :: emax 
       INTEGER :: nmax

       !c check if there is still some electron to be removed
       IF (sum(OCC)<1._q) THEN
         write(*,*) 'REMOVE_ELECTRON: ERROR, no electron available' 
         STOP
       ENDIF
      
       nmax=1
       emax=E(nmax)
       DO i=1,NDIM
         IF (OCC(i)>=1._q .AND. E(i)>emax) THEN
           emax=E(i)
           nmax=i
         ENDIF
       ENDDO

       OCC(nmax)=OCC(nmax)-1._q
  
     END SUBROUTINE remove_electron

    SUBROUTINE add_electron(NDIM,OCC,E,L)
     !c increase occupancy of the lowest available shell
     !c so as to prepare anionic state
       INTEGER :: NDIM,i,j
       REAL(q) :: OCC(NDIM)
       REAL(q) :: E(NDIM)
       INTEGER :: L(NDIM)
       REAL(q) :: emin 
       INTEGER :: nmin,maxocc

       !c check if there is a free shell
       nmin=0
       DO i=1,NDIM
         maxocc=4*L(i)+2
         IF (OCC(i)<maxocc) THEN
           emin=E(i)
           nmin=i
           EXIT
         ENDIF
       ENDDO
       IF (nmin==0) THEN
         write(*,*) 'ADD_ELECTRON: ERROR, no shell available' 
         STOP
       ENDIF
      
       DO i=1,NDIM
         maxocc=4*L(i)+2
         IF (OCC(i)<maxocc .AND. E(i)<emin) THEN
           emin=E(i)
           nmin=i
         ENDIF
       ENDDO

       OCC(nmin)=OCC(nmin)+1._q
  
     END SUBROUTINE add_electron
!c tb end

!******************** FUNCTION DFATOM2 *********************************
!
! this routine has been written by Tomas Bucko 
! it is required for the TS vdW corrections
! and iterative Hirshfeld method
!
!***********************************************************************
     SUBROUTINE DFATOM2(PP,W,IO,qstate,LFROZENORBITAL)
      USE prec
      USE base
      USE pseudo
      USE setexm
      USE constant

      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J,K,ii
      REAL(q) SUM
      REAL(q) DRHO
      REAL(q), ALLOCATABLE :: RHO(:),RHOP(:),V(:),RHO_frozen(:),RHO_compenzation(:),RHO_cor(:)
      INTEGER, PARAMETER :: NSCF=1000
      REAL(q), PARAMETER :: MIX=0.2_q
      REAL(q), PARAMETER :: EDIFF=1.0E-7
      ! gK & tB: 14.03.2016 increased RHODIFF slightly, since As did not
      ! converge
      !REAL(q), PARAMETER :: RHODIFF= 5.0E-7
      REAL(q), PARAMETER :: RHODIFF=1.0E-6
      REAL(q), PARAMETER :: RHODIFF2=1.0E-4
      INTEGER,SAVE :: tmp=0
      INTEGER :: qstate
      LOGICAL :: LFROZENORBITAL
      REAL(q),PARAMETER :: RcompMAX=7. ! radius for positive compensation charge
      REAL(q) :: Rcomp,Rcomp_old,qdummy, qdummy_old
      !TYPE (Anderson_Context) , POINTER :: AErho
      !REAL(q), ALLOCATABLE :: RHO_cor(:)
      REAL(q) :: DOUBLEC,TOTEN
      LOGICAL :: LRELAXCORE=.TRUE.  !c relax core electrons in ions?
      !!LOGICAL :: LRELAXCORE=.FALSE. !c relax core electrons in ions?
      LOGICAL LFIRST,LNOGGA



      ! Initialize data structure
      CALL ALLOCW(PP,W)

      ! Initialize wave functions
      CALL INITWF(PP,W,IO)

      ALLOCATE(RHO(W%R%NMAX),RHOP(W%R%NMAX),V(W%R%NMAX),RHO_frozen(W%R%NMAX),RHO_compenzation(W%R%NMAX),RHO_cor(W%R%NMAX))

      RHO_compenzation=0._q


      ! Switch to the exchange-correlation functional
      ! that was used for the generation of the PAW data set
      CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q,0.0_q)

      ! Initial density (nothing fancy, treating the "small" and "large"
      ! components properly is done in DFSOLVE during the scf cycles below)
      RHO=0
      DO I=1,W%NS
         RHO(:)=RHO(:)+W%OCC(I)*W%W(:,I)*W%W(:,I)
      ENDDO

      ! first we try to converge without gradient corrections
      LNOGGA=.FALSE. ; IF (ISGGA()) LNOGGA=.TRUE.
    
      scf: DO I=1,NSCF
         RHOP=RHO
         ! Get the potential
         !CALL RAD_POT_DFT(RHO,INT(W%Z),W%R,V)
         CALL RAD_POT_DFT2(RHO,RHO_compenzation,NINT(W%Z),W%R,V,0,DOUBLEC,LGGA=.NOT.LNOGGA)
         !CALL RAD_POT_DFT(RHO,NINT(W%Z),W%R,V,LGGA=.NOT.LNOGGA)


         ! Solve for the wave functions
         CALL DFSOLVE(W,V,RHO)

         ! Mix the old and new densities
         DRHO=0._q
         DRHO=MAXVAL(ABS(RHO-RHOP))
#ifdef debug
         IF (IO%IU6>0) WRITE(*,'(I3,2F14.6)') I,DRHO
#endif

         RHO(:)=RHOP(:)+MIX*(RHO(:)-RHOP(:))

         IF (DRHO<RHODIFF) THEN
           IF (LNOGGA) THEN
           ! if we use a GGA functional we switch on the gradient
           ! correction after reaching convergence with LDA, and
           ! try to converge again
             LNOGGA=.FALSE.
           ELSE
           ! final convergence reached, exit scf
             EXIT scf
           ENDIF
         ENDIF
      ENDDO scf

      ! Switch back to HF exchange
      CALL POP_XC_TYPE
      IF (I>NSCF) THEN
         WRITE(*,'(A,I3,A)') 'DFATOM2: ERROR, unable to reach convergence in ',NSCF,' cycles.'   
         STOP
      ELSE
!          ! Recompute POTAE and POTAEC
!          CALL RAD_POT_DFT(RHO,INT(W%Z),W%R,V)
!          CALL RAD_POT_DFT(W%RHOAE,INT(W%Z),W%R,W%POTAEC,LXC=.FALSE.)
         TOTEN=0._q
         CALL RAD_POT_DFT2(RHO,RHO_compenzation,NINT(W%Z),W%R,V,0,TOTEN)
         CALL RAD_POT_DFT2(W%RHOAE,RHO_compenzation,NINT(W%Z),W%R,W%POTAEC,0,DOUBLEC,LXC=.FALSE.)
         W%POTAE=V-W%POTAEC
         DO ii=1,W%NS
           TOTEN=TOTEN+W%OCC(ii)*W%E(ii)
         ENDDO
        ! IF (IO%IU6>0)  write(*,*) 'TOTEN',TOTEN
!          ! Write atomic eigenvalues to OUTCAR
!          CALL REPORT(PP,W,IO,"(DFT)")
      ENDIF


!       CALL estimate_polarizability(W%R%NMAX,RHO,W%R%R,W%R%D,W%Z,Rcomp)
!       IF (IO%IU6>0)  write(*,*) 'alpha1',Rcomp/AUTOA**3
      CALL SIMPI(W%R,RHO*W%R%R(:)**3,Rcomp)
!       IF (IO%IU6>0)  write(*,*) 'alpha2',Rcomp/AUTOA**3
      Rcomp=Rcomp**0.333333333333333   
!       CALL find_compensation_radius(W%R%NMAX,RHO,W%R%R,W%R%D,Rcomp)  
     ! IF (IO%IU6>0)  write(*,*) 'rComp',Rcomp
  
      

       !c if we deal with ions, special tretment is required...
        IF (qstate .GT. 0) THEN
          DO i=1,qstate
            CALL remove_electron(W%NS,W%OCC,W%E)
          ENDDO
        ELSE IF (qstate .LT. 0) THEN 
          DO i=1,abs(qstate)
            CALL add_electron(W%NS,W%OCC,W%E,W%L)
          ENDDO
         ! CALL add_electron(W%NS,W%OCC,W%E,W%L)
        ENDIF 
        IF (qstate .NE. 0) THEN
          RHO=0
          DO i=1,W%NS
            RHO(:)=RHO(:)+W%OCC(i)*W%W(:,i)*W%W(:,i)
          ENDDO

          RHO_cor=0._q
          DO i=1,W%NSCORE
            RHO_cor(:)=RHO_cor(:)+W%OCC(i)*W%W(:,i)*W%W(:,i)
          ENDDO
         ! CALL SIMPI(W%R,RHO_cor,qdummy)
         !  write(*,*) 'test Ccharge:',qdummy,W%NSCORE
        ENDIF

      IF ((qstate .NE. 0) .AND.(.NOT. LFROZENORBITAL)) THEN
        RHO_frozen(:)=RHO(:)
        CALL compensation_watson(W,IO,RHO,RHO_cor,qstate,Rcomp,LRELAXCORE)
       ! CALL compensation_uniform(W,IO,RHO,RHO_cor,qstate,LRELAXCORE)
      ENDIF

!       CALL estimate_polarizability(W%R%NMAX,RHO,W%R%R,W%R%D,W%Z,Rcomp)
!       IF (IO%IU6>0)  write(*,*) 'alpha1',Rcomp/AUTOA**3
!       CALL SIMPI(W%R,RHO*W%R%R(:)**3,Rcomp)
!       IF (IO%IU6>0)  write(*,*) 'alpha2',Rcomp/AUTOA**3

      W%RHO=RHO
      DEALLOCATE(RHO,RHOP,V,RHO_frozen,RHO_compenzation,RHO_cor)

      RETURN
      END SUBROUTINE DFATOM2

      SUBROUTINE compensation_watson(W,IO,RHO,RHO_cor,qstate,Rcomp,LRELAXCORE)
        USE prec
        USE base
        USE pseudo
        USE setexm
        USE constant

        IMPLICIT NONE
        TYPE(atomic) W
        TYPE(in_struct) IO
        REAL(q) :: RHO(W%R%NMAX),RHO_cor(W%R%NMAX)
        REAL(q), ALLOCATABLE :: V(:),RHOP(:)
        INTEGER I,J,K,ii
        REAL(q) DRHO
        INTEGER, PARAMETER :: NSCF=1000
        REAL(q), PARAMETER :: MIX=0.2_q
        REAL(q), PARAMETER :: RHODIFF=5.0E-7
        INTEGER :: qstate
        REAL(q) :: Rcomp,Rcomp_old,qdummy, qdummy_old
        REAL(q) :: DOUBLEC,TOTEN
        LOGICAL :: LRELAXCORE

        ALLOCATE(V(W%R%NMAX),RHOP(W%R%NMAX))
        V(:)=0._q;RHOP(:)=0._q

        qdummy_old=0.
        Rcomp_old=0.
      
!       scfRcomp: DO K=1,100
        !c do not apply compensation charge on cations
        IF (qstate .GT. 0) THEN
          Rcomp=0._q
          qdummy=0._q
        ELSE
          !CALL SIMPI(W%R,RHO*W%R%R(:),Rcomp)
          !Rcomp=Rcomp**0.5
!           CALL SIMPI(W%R,RHO*W%R%R(:)**2,Rcomp)
!           Rcomp=Rcomp**0.5
!           CALL SIMPI(W%R,RHO*W%R%R(:)**3,Rcomp)
!           Rcomp=Rcomp**0.333333333333333        
          qdummy=1._q*qstate  
!           CALL find_compensation_radius(W%R%NMAX,RHO,W%R%R,W%R%D,Rcomp)
           ! Rcomp=Rcomp_old+0.1
        ENDIF

!           IF (IO%IU6>0) WRITE(*,*) 'Rcomp:',K, Rcomp
!           IF ((K>1) .AND. (ABS(Rcomp-Rcomp_old)) .LE. 1e-3) THEN
!             EXIT scfRcomp
!           ENDIF
          Rcomp_old=Rcomp

          scf2: DO I=1,NSCF          
            RHOP=RHO
   
            ! Get the potential
            CALL RAD_POT_DFT3(RHO,INT(W%Z),W%R,V,qdummy,Rcomp,DOUBLEC)

            IF (LRELAXCORE) THEN
              !c Solve for the wave functions
              CALL DFSOLVE(W,V,RHO)
            ELSE
              ! Solve for the wave functions but don't move with the core electrons
              CALL DFSOLVE2(W,V,RHO,RHO_cor)
              !CALL SIMPI(W%R,RHO,qdummy)
!               write(*,*) 'cCharge2',qdummy
            ENDIF

            ! Mix the old and new densities
            DRHO=0._q
            DRHO=MAXVAL(ABS(RHO-RHOP))

!             IF (IO%IU6>0) WRITE(*,*) 'drho',I,DRHO
#ifdef debug 
            IF (IO%IU6>0) WRITE(*,'(I3,2F14.6)') I,DRHO
#endif

            RHO(:)=RHOP(:)+MIX*(RHO(:)-RHOP(:))

            IF (DRHO<RHODIFF) THEN
              EXIT scf2
            ENDIF
           
          ENDDO scf2
          IF (I>NSCF) THEN
            WRITE(*,'(A,I4,A)') 'SUBROUTINE COMPENSATION_WATSON: ERROR1, unable to reach convergence in ',NSCF,' cycles.'  
            STOP
          ELSE
            TOTEN=0._q
        
            CALL RAD_POT_DFT3(RHO,INT(W%Z),W%R,V,qdummy,Rcomp,TOTEN)
            CALL RAD_POT_DFT3(W%RHOAE,INT(W%Z),W%R,W%POTAEC,qdummy,Rcomp,DOUBLEC,LXC=.FALSE.)
            DO ii=1,W%NS
              TOTEN=TOTEN+W%OCC(ii)*W%E(ii)
            ENDDO

            !IF (IO%IU6>0) write(*,*) 'TOTEN_',Rcomp,qdummy,TOTEN

           
          ENDIF
!         ENDDO scfRcomp
! 
!         IF (K>100) THEN
!           WRITE(*,'(A,I3,A)') 'COMPENSATION_WATSON: ERROR2, unable to reach convergence in ',100,' cycles.'  
!           STOP
!         ENDIF    

        DEALLOCATE(RHOP,V)

      END SUBROUTINE compensation_watson

      SUBROUTINE compensation_uniform(W,IO,RHO,RHO_cor,qstate,LRELAXCORE)
        USE prec
        USE base
        USE pseudo
        USE setexm
        USE constant

        IMPLICIT NONE
        TYPE(atomic) W
        TYPE(in_struct) IO
        REAL(q) :: RHO(W%R%NMAX),RHO_cor(W%R%NMAX)
        REAL(q), ALLOCATABLE :: V(:),RHOP(:),RHO_compenzation(:)
        INTEGER I,J,K,ii
        REAL(q) DRHO
        INTEGER, PARAMETER :: NSCF=1000
        REAL(q), PARAMETER :: MIX=0.2_q
        REAL(q), PARAMETER :: RHODIFF=5.0E-7
        INTEGER :: qstate
        REAL(q) :: Rcomp,Rcomp_old,qdummy
        REAL(q) :: DOUBLEC,TOTEN
        LOGICAL :: LRELAXCORE

        ALLOCATE(V(W%R%NMAX),RHOP(W%R%NMAX),RHO_compenzation(W%R%NMAX))
        V(:)=0._q;RHOP(:)=0._q;RHO_compenzation=0._q
        
        Rcomp_old=0.
      
        Rcomp=0.

        scfRcomp: DO K=1,100
          RHO_compenzation=0._q

          !c do not apply compensation charge on cations
          IF (qstate .GT. 0) THEN
            Rcomp=0._q
            qdummy=0._q
          ELSE
            !CALL SIMPI(W%R,RHO*W%R%R(:),Rcomp)
            !Rcomp=Rcomp**0.5
!             CALL SIMPI(W%R,RHO*W%R%R(:)**2,Rcomp)
!             Rcomp=Rcomp**0.5
            CALL SIMPI(W%R,RHO*W%R%R(:)**3,Rcomp)
            Rcomp=Rcomp**0.333333333333333        
!             qdummy=1._q*qstate  
            !CALL find_compensation_radius(W%R%NMAX,RHO,W%R%R,W%R%D,Rcomp)
           ! Rcomp=Rcomp_old+0.1
          ENDIF

 
          IF (IO%IU6>0) WRITE(*,*) 'Rcomp:',K, Rcomp
          IF ((K>1) .AND. (ABS(Rcomp-Rcomp_old) .LE. 1e-3)) THEN
            EXIT scfRcomp
          ENDIF
          Rcomp_old=Rcomp

          IF (Rcomp .GT. 0.1_q) THEN
            DO J=1,W%R%NMAX            
              IF (W%R%R(J) .LE. Rcomp) THEN
                RHO_compenzation(J)=(3._q*qstate/Rcomp**3*W%R%R(J)**2)      
              ELSE               
                RHO_compenzation(J)=(3._q*qstate/Rcomp**3*W%R%R(J)**2)*EXP(-100._q*(Rcomp-W%R%R(J))**2)
              ENDIF
         
             !IF (IO%IU6>0) WRITE(*,*) 'rrcomp: ',W%R%R(J),RHO_compenzation(J)

            ENDDO
          ENDIF

          CALL SIMPI(W%R,RHO_compenzation,qdummy)
          IF (qdummy .GT. 0._q) RHO_compenzation=RHO_compenzation/qdummy*qstate    
         
                      
          scf2: DO I=1,NSCF          
            RHOP=RHO
   
            ! Get the potential
            CALL RAD_POT_DFT2(RHO,RHO_compenzation,INT(W%Z),W%R,V,qstate,DOUBLEC)

            IF (LRELAXCORE) THEN
              !c Solve for the wave functions
              CALL DFSOLVE(W,V,RHO)
            ELSE
              ! Solve for the wave functions but don't move with the core electrons
              CALL DFSOLVE2(W,V,RHO,RHO_cor)
              !CALL SIMPI(W%R,RHO,qdummy)
!               write(*,*) 'cCharge2',qdummy
            ENDIF

            ! Mix the old and new densities
            DRHO=0._q
            DRHO=MAXVAL(ABS(RHO-RHOP))

!             IF (IO%IU6>0) WRITE(*,*) 'drho',I,DRHO
#ifdef debug 
            IF (IO%IU6>0) WRITE(*,'(I3,2F14.6)') I,DRHO
#endif

            RHO(:)=RHOP(:)+MIX*(RHO(:)-RHOP(:))

            IF (DRHO<RHODIFF) THEN
              EXIT scf2
            ENDIF

            ! Copy RHO to RHOP
           
          ENDDO scf2
          IF (I>NSCF) THEN
            WRITE(*,'(A,I4,A)') 'COMPENSATION_UNIFORM: ERROR1, unable to reach convergence in ',NSCF,' cycles.'  
            STOP
          ELSE
            TOTEN=0._q
            CALL RAD_POT_DFT2(RHO,RHO_compenzation,INT(W%Z),W%R,V,qstate,TOTEN)
            CALL RAD_POT_DFT2(W%RHOAE,RHO_compenzation,INT(W%Z),W%R,W%POTAEC,qstate,DOUBLEC,LXC=.FALSE.)
            W%POTAE=V-W%POTAEC
            DO ii=1,W%NS
              TOTEN=TOTEN+W%OCC(ii)*W%E(ii)
            ENDDO
            IF (Rcomp>0._q) THEN
              TOTEN=TOTEN-1.5*FELECT*INT(W%Z)*qstate/Rcomp
            ENDIF
          !  IF (IO%IU6>0) write(*,*) 'TOTEN_',Rcomp,TOTEN
!             IF (IO%IU6>0) write(*,*) 'REPUL',Rcomp,-1.5*FELECT*INT(W%Z)*qstate/Rcomp !0.6_q*FELECT*qstate**2/Rcomp   !-1.5*FELECT*INT(W%Z)*qstate/Rcomp
!             IF (IO%IU6>0) write(*,*) 'KUKU',Rcomp,TOTEN-1.5*FELECT*INT(W%Z)*qstate/Rcomp  !+0.6_q*FELECT*qstate**2/Rcomp  !-1.5*FELECT*INT(W%Z)*qstate/Rcomp
        
          ENDIF
        ENDDO scfRcomp

        IF (K>100) THEN
          WRITE(*,'(A,I3,A)') 'COMPENSATION_UNIFORM: ERROR2, unable to reach convergence in ',100,' cycles.'  
          STOP
        ENDIF    
       
        DEALLOCATE(RHO_compenzation,RHOP,V)

!         CALL SIMPI(W%R,RHO*W%R%R(:)**3,Rcomp)
!         Rcomp=Rcomp**0.333333333333333        
!         IF (IO%IU6>0) WRITE(*,*) 'Rcomp_final:',Rcomp

      END SUBROUTINE compensation_uniform


      SUBROUTINE find_compensation_radius(atchDIM,atch,r,d,rcomp)
        USE prec
        USE constant
        IMPLICIT NONE
        INTEGER :: atchDIM,i
        REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
        REAL(q) :: r(atchDIM)
        REAL(q) :: avol,dr,d,rcomp,charge
        !REAL(q),PARAMETER :: ascale=2.5
        REAL(q),PARAMETER :: ascale=1.0
        REAL(q),PARAMETER :: tolerance=1.e-2

        dr=(d-1._q)/d
        avol=atch(1)/2*r(1)*dr
        DO i=2,atchDIM
          avol=avol+(atch(i)+atch(i-1))/2*r(i)*dr
        ENDDO
        charge=avol

        !CALL SIMPI(r,atch,charge)
        
 
        !write(*,*) 'CHarge:',charge

        avol=atch(1)/2*r(1)*dr
        DO i=2,atchDIM
          avol=avol+(atch(i)+atch(i-1))/2*r(i)*dr
          IF (abs(charge-avol)/charge < 1.e-2) THEN
          !IF (abs(charge-avol)< tolerance) THEN
            rcomp=r(i)
            !rcomp=(tolerance-atch(i-1))/(atch(i)-atch(i-1))*(r(i)-r(i-1))+r(i-1)

            EXIT
          ENDIF
        ENDDO

        !!rcomp=ascale*rcomp
    END SUBROUTINE find_compensation_radius

    SUBROUTINE estimate_polarizability(atchDIM,atch,r,d,zet,alpha)
        USE prec
        USE constant
        IMPLICIT NONE
        INTEGER :: atchDIM,i
        REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
        REAL(q) :: Nout(atchDIM)
        REAL(q) :: r(atchDIM)
        REAL(q) :: alpha,dr,d,charge,zet
        !REAL(q),PARAMETER :: ascale=2.5
        REAL(q),PARAMETER :: ascale=1.0
        REAL(q),PARAMETER :: tolerance=1.e-2

        dr=(d-1._q)/d
        Nout(1)=atch(1)/2*r(1)*dr
        DO i=2,atchDIM
          Nout(i)=Nout(i-1)+(atch(i)+atch(i-1))/2*r(i)*dr
        ENDDO
        
        charge=Nout(atchDim)
        DO i=1,atchDIM
          Nout(i)=charge-Nout(i)
        ENDDO

        alpha=atch(1)/2*r(1)**4*dr/(Nout(1)+1._q-Nout(i)/zet)
        DO i=2,atchDIM
          alpha=alpha+(atch(i)+atch(i-1))/2*r(i)**4*dr/(Nout(i)+1._q-Nout(i)/zet)
        ENDDO

    END SUBROUTINE estimate_polarizability


!     SUBROUTINE compute_charge(atchDIM,atch,r,d,charge)
!       USE prec
!       USE constant
!       IMPLICIT NONE
!       INTEGER :: atchDIM,i
!       REAL(q) :: atch(atchDIM) !4*PI*n(r)*r^2
!       REAL(q) :: r(atchDIM)
!       REAL(q) :: dr,d,charge
!       
! 
!         dr=(d-1._q)/d
!         charge=atch(1)/2*r(1)*dr
!        ! write(*,*) 'cHargE:',1,charge
!         DO i=2,atchDIM
!           charge=charge+(atch(i)+atch(i-1))/2*r(i)*dr
!         !  write(*,*) 'cHargE:',i,charge
!         ENDDO
!        write(*,*) 'cHargE:',charge
!     END SUBROUTINE compute_charge

!tb end



!***********************************************************************
!***********************************************************************
!
! PRIVATE ! PRIVATE ! PRIVATE ! PRIVATE ! PRIVATE ! PRIVATE ! PRIVATE ! 
!
! Below this point all subroutine and functions are PRIVATE to this
! module
!
!***********************************************************************
!***********************************************************************


!******************** SUBROUTINE DFATOM ********************************
!
! DFATOM solves the radial scalar relativistic Schroedinger equation for
! the spherical nonspinpolarized atom. Exchange and correlation are
! described by the density functional (LDA or GGA) used in the original
! construction of the PAW dataset.
! On exit the following components of the datastructure W have been set:
!   NS,NSCORE,NSVAL,Z,ZCORE,ZVAL,N,L,OCC,E,A,B,W,WKINAE,RHOAE,R,BETA
! (see the definition of TYPE atomic in the header of the module).
!
!***********************************************************************
      SUBROUTINE DFATOM(PP,W,IO)
      USE prec
      USE base
      USE pseudo
      USE setexm
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J,K
      REAL(q) SUM
      REAL(q) DRHO
      REAL(q), ALLOCATABLE :: RHO(:),RHOP(:),V(:)
      INTEGER, PARAMETER :: NSCF=500
      REAL(q), PARAMETER :: MIX=0.2_q
      REAL(q), PARAMETER :: EDIFF=1.0E-7
      REAL(q), PARAMETER :: RHODIFF=1.0E-7
      LOGICAL LFIRST,LNOGGA

      ! Initialize data structure
      CALL ALLOCW(PP,W)

      ! Initialize wave functions
      CALL INITWF(PP,W,IO)

#ifdef debug
      IF (IO%IU6>0) THEN
      DO I=1,PP%LMAX
         OPEN(100+I)
         WRITE(100+I,'(3F20.10)') (PP%R%R(J),PP%WAE(J,I),PP%WPS(J,I), J=1,PP%R%NMAX)
         CLOSE(100+I)
         OPEN(150+I)
         WRITE(150+I,'(2F20.10)') (PP%R%R(J),W%BETA(J,I), J=1,PP%R%NMAX)
         CLOSE(150+I)
      ENDDO
      DO I=1,W%NS
         OPEN(200+I)
         WRITE(200+I,'(2F20.10)') (W%R%R(J),W%A(J,I), J=1,W%R%NMAX)
         CLOSE(200+I)
      ENDDO
      ENDIF
#endif

      ALLOCATE(RHO(W%R%NMAX),RHOP(W%R%NMAX),V(W%R%NMAX))

      LFIRST=.TRUE.

      ! Switch to the exchange-correlation functional
      ! that was used for the generation of the PAW data set
      CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q,0.0_q)

      ! Initial density (nothing fancy, treating the "small" and "large"
      ! components properly is done in DFSOLVE during the scf cycles below)
      RHO=0
      DO I=1,W%NS
         RHO(:)=RHO(:)+W%OCC(I)*W%W(:,I)*W%W(:,I)
      ENDDO
#ifdef debug
      CALL SIMPI(W%R,RHO,SUM)
      IF (IO%IU6>0) WRITE(*,'(A,I4,F14.7)') 'DFATOM: total initial charge:',NINT(W%Z),SUM
#endif
      ! first we try to converge without gradient corrections
      LNOGGA=.FALSE. ; IF (ISGGA()) LNOGGA=.TRUE.

      scf: DO I=1,NSCF
         RHOP=RHO
         ! Get the potential
         CALL RAD_POT_DFT(RHO,NINT(W%Z),W%R,V,LGGA=.NOT.LNOGGA)

         ! Solve for the wave functions
         CALL DFSOLVE(W,V,RHO)

         ! Mix the old and new densities
         DRHO=0._q
         DRHO=MAXVAL(ABS(RHO-RHOP))
         RHO(:)=RHOP(:)+MIX*(RHO(:)-RHOP(:))

#ifdef debug
         IF (IO%IU6>0) WRITE(*,'(I3,2F14.6)') I,DRHO
#endif

         IF (DRHO<RHODIFF) THEN
            IF (LNOGGA) THEN
               ! if we use a GGA functional we switch on the gradient
               ! correction after reaching convergence with LDA, and
               ! try to converge again
               LNOGGA=.FALSE.
            ELSE
               ! final convergence reached, exit scf
               EXIT scf
            ENDIF
         ENDIF
         ! Copy RHO to RHOP
         LFIRST=.FALSE.
      ENDDO scf

      ! Switch back to HF exchange
      CALL POP_XC_TYPE

      IF (I>NSCF) THEN
         WRITE(*,'(A,I3,A)') 'DFATOM: ERROR, unable to reach convergence in ',NSCF,' cycles.'   
         STOP
      ELSE
         ! Recompute POTAE and POTAEC
         CALL RAD_POT_DFT(RHO,NINT(W%Z),W%R,V)
         CALL RAD_POT_DFT(W%RHOAE,NINT(W%Z),W%R,W%POTAEC,LXC=.FALSE.)
         W%POTAE=V-W%POTAEC
         ! Write atomic eigenvalues to OUTCAR
         CALL REPORT(PP,W,IO,"(DFT)")
#ifdef debug
         IF (IO%IU6>0) THEN
         DO I=1,W%NS
            OPEN(500+I)
            WRITE(500+I,'(4F20.10)') (W%R%R(J),W%W(J,I),W%A(J,I),W%B(J,I),J=1,W%R%NMAX)
            CLOSE(500+I)
         ENDDO
         OPEN(550)
         WRITE(550,'(3F20.10)') (W%R%R(J),W%POTAEC(J),W%POTAE(J),J=1,W%R%NMAX)
         CLOSE(550)
         OPEN(560)
         WRITE(560,'(3F20.10)') (PP%R%R(J),PP%POTPSC(J),-PP%POTAE(J),J=1,PP%R%NMAX)
         CLOSE(560)
         ENDIF
#endif
      ENDIF

      DEALLOCATE(RHO,RHOP,V)

      RETURN
      END SUBROUTINE DFATOM


!******************** SUBROUTINE DFSOLVE *******************************
!
! DFSOLVE is a wrapper for the subroutine CORE_WAVE_FKT in module cl.
! The latter computes solutions to the scalar relativistic Schroedinger
! equation with local potential V(r) by inward and outward integration.
! On exit the following components of the datastructure W have been set:
!   E,A,B,W,RHOAE
! (see the definition of TYPE atomic in the header of the module).
!
!***********************************************************************
      SUBROUTINE DFSOLVE(W,V,RHO)
      USE prec
      USE cl
      USE radial
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) V(W%R%NMAX)
      REAL(q) RHO(W%R%NMAX)
      ! local variables
      INTEGER I,K
      REAL(q) WN,WNA
      REAL(q) DRHO(W%R%NMAX)
      REAL(q) TMP(W%R%NMAX)
      
      W%RHOAE=0
      
      RHO=0
      DO I=1,W%NS
         CALL CORE_WAVE_FKT(DRHO,W%E(I),W%N(I),W%L(I),V,W%R,NINT(W%Z), &
        &   A_=W%A(:,I),B_=W%B(:,I))
         W%B(:,I)=W%B(:,I)*C
         RHO=RHO+W%OCC(I)*DRHO
         ! Normalize wave function
         TMP(:)=W%A(:,I)*W%A(:,I)
         CALL SIMPI(W%R,TMP,WNA)
         TMP(:)=W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C
         CALL SIMPI(W%R,TMP,WN)
!        W%W(:,I)=W%A(:,I)/SQRT(WN/WNA)
         W%W(:,I)=W%A(:,I)/SQRT(WNA)
#ifdef debug
!        OPEN(800+I)
!        WRITE(800+I,'(3F20.10)') (W%R%R(K),W%A(K,I),W%B(K,I),K=1,W%R%NMAX)
!        CLOSE(800+I)
#endif
         ! store AE core charge
         IF (I<=W%NSCORE) W%RHOAE(:)=W%RHOAE(:)+W%OCC(I)*DRHO(:)
      ENDDO
      
      RETURN
      END SUBROUTINE DFSOLVE

!tb beg
      SUBROUTINE DFSOLVE2(W,V,RHO,RHO_cor)
      USE prec
      USE cl
      USE radial
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) V(W%R%NMAX)
      REAL(q) RHO(W%R%NMAX),RHO_cor(W%R%NMAX)
      ! local variables
      INTEGER I,K
      REAL(q) WN,WNA
      REAL(q) DRHO(W%R%NMAX)
      REAL(q) TMP(W%R%NMAX)

      W%RHOAE=RHO_cor

      RHO=RHO_cor

      DO I=W%NSCORE+1,W%NS
         CALL CORE_WAVE_FKT(DRHO,W%E(I),W%N(I),W%L(I),V,W%R,INT(W%Z), &
        &   A_=W%A(:,I),B_=W%B(:,I))
         W%B(:,I)=W%B(:,I)*C
         RHO=RHO+W%OCC(I)*DRHO
         ! Normalize wave function
         TMP(:)=W%A(:,I)*W%A(:,I)
         CALL SIMPI(W%R,TMP,WNA)
         TMP(:)=W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C
         CALL SIMPI(W%R,TMP,WN)
         W%W(:,I)=W%A(:,I)/SQRT(WNA)
      ENDDO
        

      RETURN
      END SUBROUTINE DFSOLVE2
!tb end



!******************** SUBROUTINE DF_PAR_WAVE ***************************
!
! DF_PAR_WAVE recomputes the all-electron partial waves.
! The AE partial waves that correspond to bound states of the atom, are
! simply copied from W%W to W%WAE. The divergent partial waves are found
! by outward integration (up to at least r=minParWaveRadius).
! For r>minParWaveRadius the partial waves are cutoff after their
! absolute amplitude exceeds maxParWaveAmplitude. The latter can be set
! by means of the MAXPWAMP INCAR-key.
! The pseudized partial waves are simply copied from the PP structure.
! N.B.: The present setup in rhfatm should yield AE partial waves that
! are identical to the ones stored in the original PP data structure.
! They are recalculated only because the divergent partial waves in
! PP data structures are cut of at quite small radii (too small for
! our purposes here).
! On exit the following components of the datastructure W have been set:
!   EPS,WAE,WPS,PARWKINAE,PARWKINPS
! (see the definition of TYPE atomic in the header of the module).
!
!***********************************************************************
      SUBROUTINE DF_PAR_WAVE(PP,W,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE setexm
      USE constant
      USE paw
      IMPLICIT NONE
      TYPE(atomic) W
      TYPE(potcar), POINTER :: PP
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J,K
      INTEGER IRM
      INTEGER KI,NERR
      INTEGER LM,LMP,LYMAX,LMMAX
      REAL(q) SCALE
      REAL(q) SUM,WN
      REAL(q) DWDR0,DWDR
      REAL(q) DLM(PP%LMDIM*PP%LMDIM)
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: DIJ,QIJ
      OVERLAP DLLMM(PP%LMDIM,PP%LMDIM,1),DHXC(PP%LMDIM,PP%LMDIM)
      REAL(q), ALLOCATABLE :: RHO(:),V(:),A(:),B(:),TMP(:),DTMP(:),VPS(:),VTMP(:,:,:)
      REAL(q), PARAMETER :: WDIFF=1E-6
      REAL(q), PARAMETER :: DWDIFF=1E-6
      LOGICAL LERR


      ALLOCATE(RHO(W%R%NMAX),V(W%R%NMAX),A(W%R%NMAX),B(W%R%NMAX), &
     &   TMP(W%R%NMAX),DTMP(W%R%NMAX))

      ! Switch to the exchange-correlation functional
      ! that was used for the generation of the PAW data set
      CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q,0.0_q)

      ! Compute the atomic charge density (beware the "small" and "large"
      ! components have to be properly set by DFATOM)
      RHO=0
      DO I=1,W%NS
         RHO(:)=RHO(:)+W%OCC(I)*(W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C)
      ENDDO
#ifdef debug
      CALL SIMPI(W%R,RHO,SUM)
      IF (IO%IU6>0) WRITE(*,'(A,I4,F14.7)') 'DF_PAR_WAVE: total charge:',INT(W%Z),SUM
#endif
      ! Get the potential
      CALL RAD_POT_DFT(RHO,NINT(W%Z),W%R,V)

      DO I=1,PP%LMAX
         IF (IS_BOUND(PP,I)) THEN
            ! For partial waves that correspond to bound states,
            ! try and find the matching atomic state in W
            LERR=.TRUE.
            DO J=1,W%NS
               IF (N_OF_CHANNEL(PP,I)==W%N(J).AND.PP%LPS(I)==W%L(J)) THEN
                  W%WAE(:,I)=W%W(:,J); W%EPS(I)=W%E(J)
                  LERR=.FALSE.
               ENDIF
            ENDDO
            ! Not found
            IF (LERR) THEN
               WRITE(*,'(A,2I4)') 'DF_PAR_WAVE: ERROR, no matching bound state:', &
               &   I,N_OF_CHANNEL(PP,I),PP%LPS(I)
               STOP
            ENDIF
         ELSE
!
! What to do here? Many possibilities ... let's keep it flexible for now ...
!
!           ! The partial waves that do not correspond to bound states are
!           ! simply copied
!           W%WAE(:,I)=0; W%WAE(1:PP%R%NMAX,I)=PP%WAE(1:PP%R%NMAX,I)
!           W%EPS(I)=PP%E(I)
            ! To obtain the partial waves that correspond to unbound states,
            ! integrate outwards at the linearization energies PP%E
            A(1)=PP%WAE(1,I)
            CALL OUTINT(PP%E(I),PP%LPS(I),INT(W%Z),W%R,V(1:W%R%NMAX), &
           &   KI,A,B,NERR,RC=W%R%REND)
            ! match at the matching radius
            W%WAE(:,I)=A(:)
            IRM=IR_MATCH(PP,I)
            W%WAE(:,I)=W%WAE(:,I)*(PP%WAE(IRM,I)/W%WAE(IRM,I))
            W%EPS(I)=PP%E(I)
            DO K=IRM+1,W%R%NMAX
               IF (W%R%R(K)>minParWaveRadius .AND. &
              &   ABS(W%WAE(K,I))>maxParWaveAmplitude) THEN
                  W%WAE(K:W%R%NMAX,I)=0
!                 WRITE(*,*) i,'k=',k,'  irm=',irm
                  EXIT
               ENDIF 
            ENDDO
         ENDIF
         W%PARWKINAE(:,I)=0
         W%PARWKINAE(:,I)=(W%EPS(I)-V(:))*W%WAE(:,I)         
      ENDDO
      ! Copy pseudo partial waves
      DO I=1,PP%LMAX
         IRM=IR_MATCH(PP,I)
         W%WPS(1:IRM-1,I)=PP%WPS(1:IRM-1,I)
         W%WPS(IRM:,I)=W%WAE(IRM:,I)*SIGN(1._q,W%WAE(IRM,I)*PP%WAE(IRM,I))
      ENDDO

      ! Ensure the signs of the recomputed AE partial waves
      ! are consistent with those stored on the POTCAR file
      DO  I=1,PP%LMAX
         IRM=IR_MATCH(PP,I)
         W%PARWKINAE(:,I)=W%PARWKINAE(:,I)*SIGN(1._q,W%WAE(IRM,I)*PP%WAE(IRM,I))
         W%WAE(:,I)=W%WAE(:,I)*SIGN(1._q,W%WAE(IRM,I)*PP%WAE(IRM,I))
      ENDDO

      ! Check consistency of recomputed AE partial waves
      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,'(/X,A,X,A2)') 'DF_PAR_WAVE:',PP%ELEMENT
      WRITE(IO%IU6,'(2X,A,6X,A,5X,4X,A,5X,A,1X,5X,A,4X,A,2X,A)') &
     &    'ch','r_m','r_max-r_end', &
     &   'phi_new','delta phi', &
     &   'dphi_new/dr','delta dphi_dr'
      DO I=1,PP%LMAX
         IRM=IR_MATCH(PP,I)
         CALL GRAD(PP%R,PP%WAE(:,I),TMP(1:PP%R%NMAX))
         DWDR0=TMP(IRM)
         CALL GRAD(W%R,W%WAE(:,I),TMP)
         DWDR=TMP(IRM)
         DO K=IRM,PP%R%NMAX
            IF (W%R%R(K)>minParWaveRadius .AND. &
           &   ABS(W%WAE(K,I))>maxParWaveAmplitude) EXIT
         ENDDO
         K=K-1
         WRITE(IO%IU6,'(I4,6F14.7)',ADVANCE="No") I, &
        &   W%R%R(IRM),PP%R%R(PP%R%NMAX)-PP%R%R(K), &
        &   W%WAE(IRM,I),PP%WAE(IRM,I)-W%WAE(IRM,I),DWDR,DWDR0-DWDR
         IF (ABS(PP%WAE(IRM,I)-W%WAE(IRM,I))>WDIFF .OR. ABS(DWDR0-DWDR)>DWDIFF) THEN
            WRITE(IO%IU6,'(X,A1)') '*'
         ELSE
            WRITE(IO%IU6,*)
         ENDIF
      ENDDO
      ENDIF

      ! Compute T | \tilde{\psi} >
      ALLOCATE(VPS(PP%R%NMAX))
      VPS(:)=-PP%POTPS(:)+PP%POTPSC(:)
!     VPS(:)=VPS(:)+PP%VPSRMAX-V00(PP%R%NMAX)
#ifdef debug
      IF (IO%IU6>0) THEN
!        WRITE(*,'(A,F14.7,A,F14.7)') 'PSDMAX=',PP%PSDMAX,' PSRMAX=',PP%PSRMAX
         OPEN(450)
         WRITE(450,'(3F20.10)') (W%R%R(I),V(I),VPS(I),I=1,PP%R%NMAX)
         CLOSE(450)
      ENDIF
#endif

      LYMAX=MAXVAL(PP%LPS(1:PP%LMAX))
      IF (ASSOCIATED(PP%QPAW)) LYMAX=LYMAX*2
      LMMAX=(LYMAX+1)**2
      ALLOCATE(VTMP(PP%R%NMAX,LMMAX,1))
      SCALE=1/(2*SQRT(PI))      
      VTMP=0; VTMP(:,1,1)=VPS(:)/SCALE
      ! Reconstruct the PAW strength parameters for the reference system
      CALL RAD_POT_WEIGHT(PP%R,1,LYMAX,VTMP)
      DLM=0; DLLMM=0; DHXC=0
      CALL RAD_AUG_PROJ(VTMP(:,:,1),PP%R,DLM,PP%LMAX,PP%LPS,LYMAX,PP%AUG,PP%QPAW)
      CALL TRANS_DLM(DLLMM(:,:,1),DLM,PP)

      DHXC=-DLLMM(:,:,1)
      ! Compute D_{ij} and Q_{ij}
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

      ! T |\tilde{psi}_k> = (e_k - \tilde{V}_loc) |\tilde{psi}_k>
      !                       + \sum_i |\tilde{p}_i> (e_k*Q_ik - D_ik)
      W%PARWKINPS=0
      DO I=1,PP%LMAX
!        W%PARWKINPS(1:PP%R%NMAX,I)=(PP%E(I)-VPS(1:PP%R%NMAX))*PP%WPS(1:PP%R%NMAX,I)
         W%PARWKINPS(1:PP%R%NMAX,I)=(PP%E(I)-VPS(1:PP%R%NMAX))*W%WPS(1:PP%R%NMAX,I)
         DO J=1,PP%LMAX
            IF (PP%LPS(I)/=PP%LPS(J)) CYCLE
            W%PARWKINPS(1:PP%R%NMAX,I)=W%PARWKINPS(1:PP%R%NMAX,I)+&
           &   (PP%E(I)*QIJ(J,I)-DIJ(J,I))*W%BETA(1:PP%R%NMAX,J)
         ENDDO
      ENDDO

      ! Switch back to HF exchange
      CALL POP_XC_TYPE

#ifdef debug
      IF (IO%IU6>0) THEN
      WRITE(*,*)
      WRITE(*,'(A)') 'DF_PAR_WAVE: <p_i|tilde{phi}_j> duality'
      DO I=1,PP%LMAX
         WRITE(*,'(I4)',ADVANCE="No") I
         DO J=1,PP%LMAX
            TMP=0; SUM=0
            IF (PP%LPS(I)==PP%LPS(J)) TMP(:)=W%BETA(:,I)*W%WPS(:,J)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(*,*)
      ENDDO
      WRITE(*,*)

      DO I=1,PP%LMAX
         CALL GRAD(W%R,W%WPS(:,I),TMP)
         CALL GRAD(W%R,TMP,DTMP)
         OPEN(450+I)
         WRITE(450+I,'(7F20.10)') (W%R%R(J),W%WAE(J,I),W%WPS(J,I),W%PARWKINAE(J,I),W%PARWKINPS(J,I), &
        &   -(DTMP(J)-PP%LPS(I)*(PP%LPS(I)+1)/W%R%R(J)/W%R%R(J)*W%WPS(J,I))*HSQDTM, &
        &   TMP(J),J=1,W%R%NMAX)
         CLOSE(450+I)
      ENDDO
      ENDIF
#endif

      DEALLOCATE(RHO,V,A,B,TMP,DTMP,VPS,VTMP)
      
      RETURN
      END SUBROUTINE DF_PAR_WAVE


!******************** SUBROUTINE HFATOM ********************************
!
! HFATOM solves the radial scalar relativistic Hartree-Fock equations
! for the spherical nonspinpolarized atom, and calculates the HF atomic
! reference energy.
! On exit the following components of the datastructure W have been set:
!   NS,NSCORE,NSVAL,Z,ZCORE,ZVAL,N,L,OCC,
!   E,A,B,W,WKINAE,RHOAE,R,BETA,EATOM
! (see the definition of TYPE atomic in the header of the module).
!
!***********************************************************************
      SUBROUTINE HFATOM(PP,W,IO,W0)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      TYPE(in_struct) IO
      TYPE(atomic), OPTIONAL, INTENT(IN) :: W0
      ! local variables
      TYPE(atomic) WP
      INTEGER I,J,K
      REAL(q) DW,DE
      REAL(q) WN,WNA
      REAL(q) EKIN,EHAR,EHARV,EXCH,EXCHC,EIGNSM
      REAL(q), ALLOCATABLE :: V(:),VV(:),XA(:,:),XB(:,:),VX(:,:),DXB(:),TMP(:)
      INTEGER, PARAMETER :: NSCF=100
      REAL(q), PARAMETER :: EDIFF=1E-7
      REAL(q), PARAMETER :: WDIFF=1E-7
      LOGICAL LFIRST

      ! Initialize data structure
      CALL ALLOCW(PP,W)
      CALL ALLOCW(PP,WP)

      ! Initialize wave functions
      CALL INITWF(PP,W,IO)
!     IF (PRESENT(W0)) W=W0
      IF (PRESENT(W0)) THEN
         W%A=W0%A ; W%B=W0%B ; W%W=W0%W ; W%E=W0%E
      ENDIF

      WP%A=W%A; WP%B=W%B; WP%W=W%W; WP%E=W%E
      
      DO I=1,W%NS
#ifdef debug
         IF (IO%IU6>0) THEN
            WRITE(*,'(A,I4,A,F20.15,A,F20.15)') 'i=',I,' WA(1)=',W%A(1,I),' WB(1)=',W%B(1,I)
         ENDIF
#endif
         IF (ABS(W%A(1,I))<1E-6_q) W%A(1,I)=1E-6_q
      ENDDO
      
#ifdef debug
      IF (IO%IU6>0) THEN      
      DO I=1,PP%LMAX
         OPEN(100+I)
         WRITE(100+I,'(2F20.10)') (PP%R%R(J),PP%WAE(J,I), J=1,PP%R%NMAX)
         CLOSE(100+I)
      ENDDO
      DO I=1,W%NS
         OPEN(200+I)
         WRITE(200+I,'(2F20.10)') (W%R%R(J),W%A(J,I), J=1,W%R%NMAX)
         CLOSE(200+I)
      ENDDO
      ENDIF
#endif
      
      ALLOCATE(V(W%R%NMAX),XA(W%R%NMAX,W%NS),XB(W%R%NMAX,W%NS),VX(W%R%NMAX,W%NS), &
     &   DXB(W%R%NMAX),TMP(W%R%NMAX),VV(W%R%NMAX))
      
      LFIRST=.TRUE.
      
      scf: DO I=1,NSCF
         
         ! Get direct terms of the Coulomb interaction
         CALL RAD_POT_DIRECT(W,V)

         ! Get exchange terms of the Coulomb interaction
         CALL RAD_POT_EXCHANGE(W,XA,XB,VX)

         ! Solve the HF Schroedinger equations
         CALL HFSOLVE(W,V,XA,XB,VX)

         ! Mix wave functions
         CALL WMIX(W,WP,DW,LFIRST)
         ! Break conditions
         DE=MAXVAL(ABS(W%E-WP%E))

#ifdef debug
         IF (IO%IU6>0) WRITE(*,'(I3,2F14.6)') I,DE,DW
#endif

         IF (DE<EDIFF.AND.DW<WDIFF) THEN
            EXIT scf
         ENDIF
         ! Copy W to WP
         WP%A=W%A; WP%B=W%B; WP%W=W%W; WP%E=W%E
         LFIRST=.FALSE.
      ENDDO scf

      IF (I>NSCF) THEN
         WRITE(*,'(A,I3,A)') 'HFATOM: ERROR, unable to reach convergence in ',NSCF,' cycles.'   
         STOP
      ELSE
         ! Compute the atomic reference energy
         CALL RAD_POT_DIRECT(W,V)
         CALL RAD_POT_EXCHANGE(W,XA,XB,VX)

         ! Store the action of the kinetic energy operator
         W%WKINAE=0
         DO I=1,W%NS
            CALL GRAD(W%R,XB(:,I),DXB)
!           TMP(:)=((W%E(I)-V(:))*W%A(:,I)+XA(:,I))
            TMP(:)=((W%E(I)-V(:))*W%A(:,I)+XA(:,I)- &
            &   (DXB(:)+XB(:,I)/W%R%R(:))/C/C/2*AUTOA)
            W%WKINAE(:,I)=TMP(:)
            ! rescale 
            TMP(:)=W%A(:,I)*W%A(:,I)
            CALL SIMPI(W%R,TMP,WNA)
            TMP(:)=W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C
            CALL SIMPI(W%R,TMP,WN)
            W%WKINAE(:,I)=W%WKINAE(:,I)*SQRT(WN/WNA)
         ENDDO

         ! Kinetic energy
         EKIN=0
         DO I=1,W%NS
            CALL GRAD(W%R,XB(:,I),DXB)
!           TMP(:)=W%A(:,I)*((W%E(I)-V(:))*W%A(:,I)+XA(:,I))
            TMP(:)=W%A(:,I)*((W%E(I)-V(:))*W%A(:,I)+XA(:,I)- &
            &   (DXB(:)+XB(:,I)/W%R%R(:))/C/C/2*AUTOA)
            CALL SIMPI(W%R,TMP,DE)
            EKIN=EKIN+DE*W%OCC(I)
         ENDDO

         ! Hartree energy
         EHAR=0
         DO I=1,W%NS
            TMP(:)=W%A(:,I)*(V(:)-FELECT*W%Z/W%R%R(:))*W%A(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EHAR=EHAR+DE*W%OCC(I)
         ENDDO
         EHAR=EHAR/2

         ! Exchange energy
         EXCH=0
         DO I=1,W%NS
            TMP(:)=W%A(:,I)*XA(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EXCH=EXCH+DE*W%OCC(I)
         ENDDO
         EXCH=-EXCH/2

         ! Exchange energy of the core electrons
         EXCHC=0
         DO I=1,W%NSCORE
            TMP(:)=W%A(:,I)*XA(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EXCHC=EXCHC+DE*W%OCC(I)
         ENDDO
         EXCHC=-EXCHC/2

         W%EATOM=EKIN+EHAR+EXCH

         ! Sum of the eigenvalues of the valence states
         EIGNSM=0
         DO I=W%NSCORE+1,W%NS
            EIGNSM=EIGNSM+W%E(I)*W%OCC(I)
         ENDDO

         ! 1/2 \int n_v(r) V_H[n_v](r) dr 
         CALL RAD_POT_DIRECT_VALENCE_ONLY(W,VV)
         EHARV=0
         DO I=W%NSCORE+1,W%NS
            TMP(:)=W%A(:,I)*VV(:)*W%A(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EHARV=EHARV+DE*W%OCC(I)
         ENDDO
         EHARV=EHARV/2

        ! Exchange energy of the valence electrons
         CALL RAD_POT_EXCHANGE(W,XA,XB,VX,JLO=W%NSCORE+1,JHI=W%NS)
         EXCH=0
         DO I=W%NSCORE+1,W%NS
            TMP(:)=W%A(:,I)*XA(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EXCH=EXCH+DE*W%OCC(I)
         ENDDO
         EXCH=-EXCH/2

         W%EATOM=EIGNSM-EHARV-EXCH
#ifdef debug
         IF (IO%IU6>0) THEN
            WRITE(*,'(4(A,F20.10))') 'EIGNSM=',EIGNSM,' -E_H=',-EHARV,' -E_X=',-EXCH,' E_tot=',W%EATOM
         ENDIF
#endif

         ! Kinetic energy of the valence electrons
         EKIN=0
         DO I=W%NSCORE+1,W%NS
            CALL GRAD(W%R,XB(:,I),DXB)
!           TMP(:)=W%A(:,I)*((W%E(I)-V(:))*W%A(:,I)+XA(:,I))
            TMP(:)=W%A(:,I)*((W%E(I)-V(:))*W%A(:,I)+XA(:,I)- &
            &   (DXB(:)+XB(:,I)/W%R%R(:))/C/C/2*AUTOA)
            CALL SIMPI(W%R,TMP,DE)
            EKIN=EKIN+DE*W%OCC(I)
         ENDDO

         ! \int n_v(r) V_H[n_Zc](r) dr
         EHAR=0
         DO I=W%NSCORE+1,W%NS
            TMP(:)=W%A(:,I)*(V(:)-VV(:))*W%A(:,I)
            CALL SIMPI(W%R,TMP,DE)
            EHAR=EHAR+DE*W%OCC(I)
         ENDDO

!        W%EATOM=EKIN+EHARV+EHAR+EXCH
#ifdef debug
!        IF (IO%IU6>0) THEN
!           WRITE(*,'(4(A,F20.10))') 'EKIN=',EKIN,' E_H=',EHARV+EHAR,' E_X=',EXCH,' E_tot=',W%EATOM
!        ENDIF
#endif


         CALL REPORT(PP,W,IO,"(HF)")
#ifdef debug
         IF (IO%IU6>0) THEN
         DO I=1,W%NS
            OPEN(400+I)
            WRITE(400+I,'(4F20.10)') (W%R%R(J),W%W(J,I),W%A(J,I),W%B(J,I),J=1,W%R%NMAX)
            CLOSE(400+I)
         ENDDO
         ENDIF
#endif
      ENDIF

      CALL DEALLOCW(WP)

      DEALLOCATE(V,VV,XA,XB,VX,DXB,TMP)

      RETURN
      END SUBROUTINE HFATOM


!******************** SUBROUTINE HFSOLVE *******************************
!
! HFSOLVE is a wrapper for the subroutine WSHOOT.
! The latter computes solutions to the scalar relativistic Hartree-Fock
! equations by inward and outward integration.
! On exit the following components of the datastructure W have been set:
!   E,A,B,W,RHOAE
! (see the definition of TYPE atomic in the header of the module).
!
!***********************************************************************
      SUBROUTINE HFSOLVE(W,V,XA,XB,VX)
      USE prec
      USE pseudo
      USE cl
      USE constant
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) V(W%R%NMAX)
      REAL(q) XA(W%R%NMAX,W%NS)
      REAL(q) XB(W%R%NMAX,W%NS)
      REAL(q) VX(W%R%NMAX,W%NS)
      ! local variables
      INTEGER I,K
      REAL(q) WN,WNA
      REAL(q), DIMENSION(W%R%NMAX) :: RHO,VI,XIA,XIB,TMP

      W%RHOAE=0

      DO I=1,W%NS
         VI(:)=V(:)-VX(:,I)
         XIA(:)=XA(:,I)-VX(:,I)*W%A(:,I)
         XIB(:)=XB(:,I)-VX(:,I)*W%B(:,I)

         IF (noSmallExchange) XIB=0

#ifdef debug
!        OPEN(800+I)
!        WRITE(800+I,'(5F20.10)') &
!       &   (W%R%R(K),V(K),V(K)-VX(K,I),XA(K,I),XA(K,I)-VX(K,I)*W%A(K,I),K=1,W%R%NMAX)
!        CLOSE(800+I)
#endif
         CALL WSHOOT(W%R,W%N(I),W%L(I),INT(W%Z),VI,W%E(I),W%A(:,I),XIA,XIB,W%A(:,I),W%B(:,I))   
         ! Normalize wave function
         TMP(:)=W%A(:,I)*W%A(:,I)
         CALL SIMPI(W%R,TMP,WNA)
         TMP(:)=W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C
         CALL SIMPI(W%R,TMP,WN)
         W%W(:,I)=W%A(:,I)*SQRT(WN/WNA)
         ! Store AE core charge
         IF (I<=W%NSCORE) W%RHOAE(:)=W%RHOAE(:)+W%OCC(I)*TMP(:)
      ENDDO

      RETURN
      END SUBROUTINE HFSOLVE


!******************** SUBROUTINE HF_PAR_WAVE ***************************
!
! HF_PAR_WAVE simply copies the DFT all-electron and pseudized partial
! waves and the action of the kinetic energy operator on the
! aforementioned from W0 to W.
! On exit the following components of the datastructure W have been set:
!   WAE,WPS,PARWKINAE,PARWKINPS
! (see the definition of TYPE atomic in the header of the module).

!***********************************************************************
      SUBROUTINE HF_PAR_WAVE(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W0,W
      TYPE(in_struct) IO
      ! local variables
      INTEGER I
      
      DO I=1,PP%LMAX
         W%WAE(:,I)=W0%WAE(:,I)
         W%WPS(:,I)=W0%WPS(:,I)
         W%PARWKINAE(:,I)=W0%PARWKINAE(:,I)
         W%PARWKINPS(:,I)=W0%PARWKINPS(:,I)
      ENDDO
            
      RETURN
      END SUBROUTINE HF_PAR_WAVE


!******************** SUBROUTINE HF_PAR_WAVE ***************************
!
!***********************************************************************
      SUBROUTINE HF_PAR_WAVE_v1(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W,W0
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J
      INTEGER IRM
      REAL(q) SUM
      REAL(q) DW,LGDR0,LGDR
      REAL(q), ALLOCATABLE :: V(:),XA(:),XB(:)
      REAL(q), ALLOCATABLE :: A(:),B(:),AP(:),BP(:),TMP(:),DTMP(:)
      REAL(q), ALLOCATABLE :: BI(:),BIN(:),DV(:),DA(:),DXB(:)
      REAL(q), ALLOCATABLE :: OVL(:,:)
      INTEGER, PARAMETER :: NSCF=1000
      REAL(q), PARAMETER :: WDIFF=1E-7

      ALLOCATE(V(W%R%NMAX),XA(W%R%NMAX),XB(W%R%NMAX), &
     &   A(W%R%NMAX),B(W%R%NMAX),AP(W%R%NMAX),BP(W%R%NMAX), &
     &   BI(W%R%NMAX),BIN(W%R%NMAX),DV(W%R%NMAX),DA(W%R%NMAX),DXB(W%R%NMAX), &
     &   TMP(W%R%NMAX),DTMP(W%R%NMAX))

      ! Get the direct potential
      CALL RAD_POT_DIRECT(W,V)

      DO I=1,PP%LMAX
         ! Initial partial wave
         A=W0%WAE(:,I); B=0
         ! Determine the matching point
         IRM=IR_MATCH(PP,I)
         scf: DO J=1,NSCF
            AP=A; BP=B
            ! Get the exchange potential
            CALL RAD_POT_EXCHANGE_SINGLE(PP%LPS(I),A,B,W,XA,XB)

            IF (noSmallExchange) XB=0

            ! Compute the new partial wave
            CALL PWSHOOT(W%R,PP%LPS(I),INT(W%Z),W%EPS(I),V,XA,XB,A,B,IRM)
            ! Break condition
            DW=MAXVAL(ABS(A-AP))
            IF (DW<WDIFF) EXIT scf

            ! simple mixing
            A=AP+0.1_q*(A-AP)
            B=BP+0.1_q*(B-BP)

         ENDDO scf
         IF (J>NSCF) THEN
            WRITE(*,'(A,I4,A,I4,A)') &
           &   'HF_PAR_WAVE: ERROR, partial wave',I, &
           &   ' not found within ',NSCF,' tries.'
            STOP
         ENDIF
         W%WAE(:,I)=A
         ! Compute the AE kinetic energy density
         W%PARWKINAE(:,I)=0
         BIN(:)=2._q+(W%EPS(I)-V(:))/2/RYTOEV/C/C; BI(:)=1/BIN(:)
         CALL GRAD(W%R,V,DV); CALL GRAD(W%R,W%WAE(:,I),DA); CALL GRAD(W%R,XB,DXB)
! This includes all scalar relativistic contributions to the kinetic energy 
!        W%PARWKINAE(1:IRM,I)=(W%EPS(I)-V(1:IRM))*W%WAE(1:IRM,I)+XA(1:IRM)
! This includes scalar relativistic contributions to the kinetic energy that
! involve only the large component 
         W%PARWKINAE(1:IRM,I)=(W%EPS(I)-V(1:IRM))*W%WAE(1:IRM,I)+XA(1:IRM)- &
        &   (DXB(1:IRM)+XB(1:IRM)/W%R%R(1:IRM))/C/C/2*AUTOA
! the following is strictly T |A>, i.e. -h^2/2m (d^2/dr^2 - l(l+1)/r^2) a(r)
!        W%PARWKINAE(1:IRM,I)=(W%EPS(I)-V(1:IRM))*W%WAE(1:IRM,I)/(2*RYTOEV) + &
!       &   (W%EPS(I)-V(1:IRM))*(W%EPS(I)-V(1:IRM))*W%WAE(1:IRM,I)/C/C/2/(2*RYTOEV)/(2*RYTOEV) + &
!       &   BI(1:IRM)*DV(1:IRM)*AUTOA/(2*RYTOEV)*(DA(1:IRM)-W%WAE(1:IRM,I)/W%R%R(1:IRM))/C/C/2*AUTOA + &
!       &   BIN(1:IRM)*XA(1:IRM)/2/(2*RYTOEV) - &
!       &   (DXB(1:IRM)+XB(1:IRM)/W%R%R(1:IRM))/C/C/2*AUTOA/(2*RYTOEV)
!        W%PARWKINAE(1:IRM,I)=W%PARWKINAE(1:IRM,I)*(2*RYTOEV)
      ENDDO

      ! Copy pseudo partial waves
      W%WPS=W0%WPS
      ! Copy T | \tilde{\psi} >
      W%PARWKINPS=W0%PARWKINPS

      ! At this point one might choose to make the PS partial
      ! waves exactly dual to the projectors. For the moment
      ! we suffice by ensuring the sign is right.
      ALLOCATE(OVL(PP%LMAX,PP%LMAX))
      OVL=0
      DO I=1,PP%LMAX
         DO J=1,PP%LMAX
            TMP=0
            IF (PP%LPS(I)==PP%LPS(J)) TMP(:)=W%BETA(:,I)*W%WPS(:,J)
            CALL SIMPI(W%R,TMP,OVL(I,J))
         ENDDO
      ENDDO
      DO I=1,PP%LMAX
         W%WPS(:,I)=W%WPS(:,I)*SIGN(1._q,OVL(I,I))
         W%WAE(:,I)=W%WAE(:,I)*SIGN(1._q,OVL(I,I))
         W%PARWKINAE(:,I)=W%PARWKINAE(:,I)*SIGN(1._q,OVL(I,I))
         W%PARWKINPS(:,I)=W%PARWKINPS(:,I)*SIGN(1._q,OVL(I,I))
         IRM=IR_MATCH(PP,I)
         W%PARWKINAE(IRM+1:,I)=W%PARWKINPS(IRM+1:,I)
      ENDDO

#ifdef debug
      IF (IO%IU6>0) THEN
      WRITE(*,*)
      WRITE(*,'(A)') 'HF_PAR_WAVE: new partial waves'
      DO I=1,PP%LMAX
         ! Boundary condition
         IRM=IR_MATCH(PP,I)
         CALL GRAD(W%R,W0%WAE(:,I),TMP)
         LGDR0=TMP(IRM)/W0%WAE(IRM,I)
         CALL GRAD(W%R,W%WAE(:,I),TMP)
         LGDR=TMP(IRM)/W%WAE(IRM,I)
         CALL GRAD(W%R,TMP,DTMP)
         WRITE(*,'(I4,6F14.7)') I,W%R%R(IRM),LGDR0,LGDR,W0%WAE(IRM,I),W%WAE(IRM,I),W%EPS(I)
         OPEN(350+I)
         WRITE(350+I,'(7F20.10)') (W%R%R(J),W%WAE(J,I),W%WPS(J,I),W%PARWKINAE(J,I),W%PARWKINPS(J,I), &
        &   -(DTMP(J)-PP%LPS(I)*(PP%LPS(I)+1)/W%R%R(J)/W%R%R(J)*W%WAE(J,I))*HSQDTM, &
        &   TMP(J),J=1,W%R%NMAX)
         CLOSE(350+I)
      ENDDO
      
      WRITE(*,*)
      WRITE(*,'(A)') 'HF_PAR_WAVE: Core-Valence(0) orthogonality'
      DO I=1,W%NSCORE
         WRITE(*,'(I4)',ADVANCE="No") I
         DO J=1,PP%LMAX
            TMP=0; SUM=0
            IF (W%L(I)==PP%LPS(J)) TMP(:)=W%W(:,I)*W0%WAE(:,J)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(*,*)
      ENDDO

      WRITE(*,*)
      WRITE(*,'(A)') 'HF_PAR_WAVE: Core-Valence orthogonality'
      DO I=1,W%NSCORE
         WRITE(*,'(I4)',ADVANCE="No") I
         DO J=1,PP%LMAX
            TMP=0; SUM=0
            IF (W%L(I)==PP%LPS(J)) TMP(:)=W%W(:,I)*W%WAE(:,J)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(*,*)
      ENDDO

!     WRITE(*,'(A)') 'HF_PAR_WAVE: Valence-Valence overlap'
!     DO I=W%NSCORE+1,W%NS
!        WRITE(*,'(I4)',ADVANCE="No") I
!        DO J=1,PP%LMAX
!           TMP=0; SUM=0
!           IF (W%L(I)==PP%LPS(J)) TMP(:)=W%W(:,I)*W%WAE(:,J)
!           CALL SIMPI(W%R,TMP,SUM)
!           WRITE(*,'(F14.7)',ADVANCE="No") SUM
!        ENDDO
!        WRITE(*,*)
!     ENDDO
!     WRITE(*,*)
      
      WRITE(*,*)
      WRITE(*,'(A)') 'HF_PAR_WAVE: <p_i|tilde{phi}_j> duality'
      DO I=1,PP%LMAX
         WRITE(*,'(I4)',ADVANCE="No") I
         DO J=1,PP%LMAX
            TMP=0; SUM=0
            IF (PP%LPS(I)==PP%LPS(J)) TMP(:)=W%BETA(:,I)*W%WPS(:,J)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(*,*)
      ENDDO
      WRITE(*,*)
      ENDIF
#endif
            
      DEALLOCATE(V,XA,XB,A,B,AP,BP,BI,BIN,DV,DA,DXB,TMP,DTMP,OVL)
      
      RETURN
      END SUBROUTINE HF_PAR_WAVE_v1


!******************** SUBROUTINE HF_PAR_WAVE ***************************
!      
!***********************************************************************
      SUBROUTINE HF_PAR_WAVE_v2(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      USE constant
      USE c2f_interface, ONLY : ERRF,ERRFC
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W0,W
      TYPE(in_struct) IO
      ! external function
      ! local variables
      INTEGER I,J,IRM,K
      REAL(q), DIMENSION(W%R%NMAX) :: WTMP,DW,DW0
      LOGICAL LERR 
      REAL(q) RSW,RWMAX
      REAL(q), PARAMETER :: A=4._q, RSWINT=1.5_q
      
      DO I=1,PP%LMAX
         ! AE partial waves
         IF (IS_BOUND(PP,I)) THEN
            ! For partial waves that correspond to bound states,
            ! try and find the matching atomic state in W
            LERR=.TRUE.
            DO J=1,W%NS
               IF (N_OF_CHANNEL(PP,I)==W%N(J).AND.PP%LPS(I)==W%L(J)) THEN
                  IRM=IR_MATCH(PP,I)
                  WTMP(:)=W%A(:,J)*SIGN(1._q,W%A(IRM,J)*W0%WAE(IRM,I))
                  RWMAX=W%R%R(IR_WMAX(W0%WAE(:,I)))
                  RSW=W%R%R(IR_SWITCH(W%A(:,J),W0%WAE(:,I),IR_WMAX(W0%WAE(:,I))))

!                 IF ((RSW>RWMAX).AND.((RSW+RSWINT/2)<W%R%REND).AND.(RSW-RSWINT/2)>W%R%RSTART) THEN
                  IF ((RSW>RWMAX).AND.((RSW+RSWINT/2)<PP%R%REND)) THEN
                     ! partial wave
                     DO K=1,W%R%NMAX
                        W%WAE(K,I)=W0%WAE(K,I)+ERRFC(A*(W%R%R(K)-RSW))*(WTMP(K)-W0%WAE(K,I))/2._q
                     ENDDO
                     ! kinetic energy
                     CALL GRAD(W%R,WTMP,DW); CALL GRAD(W0%R,W0%WAE(:,I),DW0)
                     DO K=1,W%R%NMAX
                        W%PARWKINAE(K,I)=ERRFC(A*(W%R%R(K)-RSW))*W%WKINAE(K,J)/2._q+ &
                       &   (1+ERRF(A*(W%R%R(K)-RSW)))*W0%PARWKINAE(K,I)/2._q- &
                       &   HSQDTM*(A/SQRT(PI))*EXP(-A*A*(W%R%R(K)-RSW)*(W%R%R(K)-RSW))* &
                       &   (DW0(K)-DW(K)-A*(W%R%R(K)-RSW)*(W0%WAE(K,I)-W%A(K,J)))
                     ENDDO
                  ELSE
                     W%WAE(:,I)=W0%WAE(:,I)
                     W%PARWKINAE(:,I)=W0%PARWKINAE(:,I)
                  ENDIF
                  LERR=.FALSE.
               ENDIF
            ENDDO
            ! Not found
            IF (LERR) THEN
               WRITE(*,'(A,2I4)') 'HF_PAR_WAVE: ERROR, no matching bound state:', &
               &   I,N_OF_CHANNEL(PP,I),PP%LPS(I)
               STOP
            ENDIF
         ELSE
            ! Simply copy for unbound states
            W%WAE(:,I)=W0%WAE(:,I)
            W%PARWKINAE(:,I)=W0%PARWKINAE(:,I)
         ENDIF
         ! PS partial waves
         W%WPS(:,I)=W0%WPS(:,I)
         W%PARWKINPS(:,I)=W0%PARWKINPS(:,I)
      ENDDO

#ifdef debug
      IF (IO%IU6>0) THEN
      DO I=1,PP%LMAX
         OPEN(600+I)
         WRITE(600+I,'(7F20.10)') (W%R%R(J),W%WAE(J,I),W0%WAE(J,I),W%WPS(J,I), &
        &   W%PARWKINAE(J,I),W0%PARWKINAE(J,I),W%PARWKINPS(J,I),J=1,W%R%NMAX)
         CLOSE(600+I)
      ENDDO
      ENDIF
#endif
      
      RETURN
      END SUBROUTINE HF_PAR_WAVE_v2


!******************** SUBROUTINE HF_PAR_WAVE ***************************
!
!***********************************************************************
      SUBROUTINE HF_PAR_WAVE_v3(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W,W0
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J,K
      INTEGER IRM
      INTEGER KI,NERR
      REAL(q) SUM
      REAL(q) DW,LGDR0,LGDR
      REAL(q), ALLOCATABLE :: V(:),XA(:),XB(:)
      REAL(q), ALLOCATABLE :: A(:),B(:),AP(:),BP(:),TMP(:),DTMP(:)
      REAL(q), ALLOCATABLE :: BI(:),BIN(:),DV(:),DA(:),DXB(:)
      REAL(q), ALLOCATABLE :: OVL(:,:)
      INTEGER, PARAMETER :: NSCF=10000
      REAL(q), PARAMETER :: WDIFF=1E-7

      ALLOCATE(V(W%R%NMAX),XA(W%R%NMAX),XB(W%R%NMAX), &
     &   A(W%R%NMAX),B(W%R%NMAX),AP(W%R%NMAX),BP(W%R%NMAX), &
     &   BI(W%R%NMAX),BIN(W%R%NMAX),DV(W%R%NMAX),DA(W%R%NMAX),DXB(W%R%NMAX), &
     &   TMP(W%R%NMAX),DTMP(W%R%NMAX))

      ! Get the direct potential
      CALL RAD_POT_DIRECT(W,V)
! test
      WRITE(*,'(F14.7)') W%E(1:W%NS)
!     STOP
! test
      DO I=1,PP%LMAX
! test
         ! Lets focus on the unbound partial waves 
         IF (IS_BOUND(PP,I)) CYCLE
! test
         ! Initial partial wave
         A=W0%WAE(:,I); B=0
         ! Determine the matching point
         IRM=IR_MATCH(PP,I)
         DO K=IRM+1,W%R%NMAX-10
!           IF (W%R%R(K)>minParWaveRadius .AND. &
!          &   ABS(W0%WAE(K,I))>maxParWaveAmplitude) THEN
!           IF (ABS(W0%WAE(K,I))>maxParWaveAmplitude) THEN
            IF (W%R%R(K)>3.50_q) THEN
               WRITE(*,*) i,'k=',k,'  irm=',irm,'  r=',w%r%r(irm)
               EXIT
            ENDIF 
         ENDDO
! test
         WRITE(*,*) i,'k=',k,'  irm=',irm,'  r=',w%r%r(k)
         irm=k
         A(IRM+1:)=0
! test
 
         scf: DO J=1,NSCF
            AP=A; BP=B
            ! Get the exchange potential
            CALL RAD_POT_EXCHANGE_SINGLE(PP%LPS(I),A,B,W,XA,XB)

            IF (noSmallExchange) XB=0

            ! Compute the new partial wave
!           CALL PWSHOOT(W%R,PP%LPS(I),INT(W%Z),W%EPS(I),V,XA,XB,A,B,IRM)
!           CALL OUTINT(W%EPS(I),PP%LPS(I),INT(W%Z),W%R,V,KI,A,B,NERR,RC=W%R%R(IRM+5),IHA=XA,IHB=XB)
            CALL OUTINT(-62.06_q,PP%LPS(I),INT(W%Z),W%R,V,KI,A,B,NERR,RC=W%R%R(IRM+5),IHA=XA,IHB=XB)
            ! Break condition
            DW=MAXVAL(ABS(A-AP))
! test
            WRITE(*,*) J,KI,'DW=',DW
! test
            IF (DW<WDIFF) EXIT scf
            ! simple mixing
            A=AP+0.01_q*(A-AP)
            B=BP+0.01_q*(B-BP)

         ENDDO scf
! test
         WRITE(350+I,'(3F20.10)') (W%R%R(K),W0%WAE(K,I),A(K),K=1,W%R%NMAX)
! test
         IF (J>NSCF) THEN
            WRITE(*,'(A,I4,A,I4,A)') &
           &   'HF_PAR_WAVE: ERROR, partial wave',I, &
           &   ' not found within ',NSCF,' tries.'
            STOP
         ENDIF
! test
         STOP
! test
      ENDDO

      DEALLOCATE(V,XA,XB,A,B,AP,BP,BI,BIN,DV,DA,DXB,TMP,DTMP)
! test
      STOP
! test
      
      RETURN
      END SUBROUTINE HF_PAR_WAVE_v3


!******************** SUBROUTINE ORTHO_PAR_WAVE ************************
!
! ORTHO_PAR_WAVE orthogonalizes the partial waves stored in W0 with
! respect to the atomic core wave functions stores in W.
! The action of the kinetic energy on the partial waves is transformed
! correspondingly.
!
!***********************************************************************
      SUBROUTINE ORTHO_PAR_WAVE(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W0,W
      TYPE(in_struct) IO
      ! local variables
      INTEGER CH1,CH2
      REAL(q) SUM
      REAL(q) TMP(W%R%NMAX)

      ! Evaluate core-valence orthogonality
      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'ORTHO_PAR_WAVE: initial Core-Valence overlap'
      DO CH1=1,W%NSCORE
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            TMP=0; SUM=0
            IF (W%L(CH1)==PP%LPS(CH2)) TMP(:)=W%W(:,CH1)*W0%WAE(:,CH2)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF

      ! Orthogonalize the partial waves of W0
      ! with respect to the core states of W
      DO CH1=1,PP%LMAX
         DO CH2=1,W%NSCORE
            IF (PP%LPS(CH1)/=W%L(CH2)) CYCLE
            TMP=0; SUM=0
            TMP(:)=W%W(:,CH2)*W0%WAE(:,CH1)
            CALL SIMPI(W%R,TMP,SUM)
            W0%WAE(:,CH1)=W0%WAE(:,CH1)-W%W(:,CH2)*SUM
            W0%PARWKINAE(:,CH1)=W0%PARWKINAE(:,CH1)-W%WKINAE(:,CH2)*SUM
         ENDDO
      ENDDO
            
      RETURN
      END SUBROUTINE ORTHO_PAR_WAVE


!******************** SUBROUTINE SET_DIJ *******************************
!      
!***********************************************************************
      SUBROUTINE SET_DIJ(PP,W,RHOC,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE constant
      USE paw
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      TYPE(in_struct) IO
      REAL(q) RHOC(:)
      ! local variables
      INTEGER CH1,CH2,I
      INTEGER IRND1,IRND2,IRM
      REAL(q) SUM,SCALE
      REAL(q) QINTP,QINTW,QINF
      REAL(q) DQINFPS,DQINFAE,DQINFAEP
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: DIJ,EDIJ
      REAL(q), DIMENSION(W%R%NMAX) :: RHO,VOLD,VNEW
      REAL(q), PARAMETER :: TINY=1E-6
      
      ! Calculate V_H[n_Zc] of the original core
      SCALE=1/(2*SQRT(PI))
      RHO=W%RHOAE*SCALE
      ! Calculate the Hartree potential of the AE core electrons (V_H[n_c])
      VOLD=0
      CALL RAD_POT_HAR(0, W%R, VOLD, RHO, SUM)
      ! add the Coulomb potential of the nucleus
      VOLD=VOLD*SCALE - FELECT/W%R%R*W%Z
#ifdef debug
      ! integrate charge
      CALL SIMPI(PP%R,PP%RHOAE,QINTP); CALL SIMPI(W%R,RHO,QINTW)
      ! infer integrated charge from Hartree potential
      QINF=VOLD(PP%R%NMAX)*PP%R%R(PP%R%NMAX)/FELECT
      ! write
      IF (IO%IU6>0) WRITE(*,'(A,F14.7,A,F14.7)') 'SET_DIJ: dQ_int=',(QINTP-QINTW)/SCALE,' Q_inf=',-QINF
      IF (IO%IU6>0) WRITE(99,'(3F20.10)') (PP%R%R(I),VOLD(I),PP%POTPSC(I),I=1,PP%R%NMAX)
#endif      

      ! Store the new AE core charge density (n_c) into PP%RHOAE
      PP%RHOAE(1:PP%R%NMAX)=RHOC(1:PP%R%NMAX)*SCALE
      RHO=RHOC*SCALE
      ! Calculate the Hartree potential of the AE core electrons (V_H[n_c])
      VNEW=0
      CALL RAD_POT_HAR(0, W%R, VNEW, RHO, SUM)
      ! add the Coulomb potential of the nucleus
      VNEW=VNEW*SCALE - FELECT/W%R%R*W%Z
      
      IF (IO%IU6>0) THEN
         WRITE(IO%IU6,'(/X,A,X,A2)') 'SET_DIJ:',PP%ELEMENT
         WRITE(IO%IU6,'(7X,A,6X,4X,A,4X,A,2X,1X,A,4X,A,3X,4X,A)') &
        &   'r','dQ_inf(PS)','dQ_inf(old)','dQ_inf(new)','dV(old)','dV(new)'
         IRM=IR_PSMAX(PP)
         DQINFPS =W%ZVAL+PP%POTPSC(IRM)*PP%R%R(IRM)/FELECT
         DQINFAE =W%ZVAL+VOLD(IRM)*PP%R%R(IRM)/FELECT
         DQINFAEP=W%ZVAL+VNEW(IRM)*PP%R%R(IRM)/FELECT
         WRITE(IO%IU6,'(6F14.7)') PP%R%R(IRM),DQINFPS,DQINFAE,DQINFAEP, &
        &   VOLD(IRM)-PP%POTPSC(IRM),VNEW(IRM)-PP%POTPSC(IRM)
         
         IRM=PP%R%NMAX
         DQINFPS =W%ZVAL+PP%POTPSC(IRM)*PP%R%R(IRM)/FELECT
         DQINFAE =W%ZVAL+VOLD(IRM)*PP%R%R(IRM)/FELECT
         DQINFAEP=W%ZVAL+VNEW(IRM)*PP%R%R(IRM)/FELECT
         WRITE(IO%IU6,'(6F14.7)') PP%R%R(IRM),DQINFPS,DQINFAE,DQINFAEP, &
        &   VOLD(IRM)-PP%POTPSC(IRM),VNEW(IRM)-PP%POTPSC(IRM)
         WRITE(IO%IU6,*)       
      ENDIF
      
      ! Calculate the corrections to DION
      DIJ=0
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         RHO=0; SUM=0
         IF (PP%LPS(CH1)==PP%LPS(CH2)) THEN
!           IRND1=IR_END(PP,CH1); IRND2=IR_END(PP,CH2)
!           IF ((ABS(VNEW(IRND1)-VOLD(IRND1))>TINY.OR. &
!          &     ABS(VNEW(IRND2)-VOLD(IRND2))>TINY).AND.IO%IU6>0) THEN
!               WRITE(*,'(A,I4,I5,F14.7,I4,I5,2F14.7)') 'SET_DIJ: WARNING:', &
!              &   CH1,IRND1,ABS(VNEW(IRND1)-VOLD(IRND1)),CH2,IRND2,ABS(VNEW(IRND2)-VOLD(IRND2)), &
!              &   ABS(VNEW(W%R%NMAX)-VOLD(W%R%NMAX))
!           ENDIF
            DO I=1,PP%R%NMAX
               RHO(I)=PP%WAE(I,CH1)*(VNEW(I)-VOLD(I))*PP%WAE(I,CH2)
            ENDDO
            CALL SIMPI(PP%R,RHO(1:PP%R%NMAX),DIJ(CH1,CH2))
         ENDIF
      ENDDO
      ENDDO

      EDIJ=0
      IRM=IR_PSMAX(PP)
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         RHO=0; SUM=0
         IF (PP%LPS(CH1)==PP%LPS(CH2)) THEN
            DO I=IRM,PP%R%NMAX
               RHO(I)=PP%WAE(I,CH1)*(VNEW(I)-VOLD(I))*PP%WAE(I,CH2)
            ENDDO
            CALL SIMPI(PP%R,RHO(1:PP%R%NMAX),EDIJ(CH1,CH2))
         ENDIF
      ENDDO
      ENDDO

      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'SET_DIJ: change to DION: < phi_i | V[n_c_hf] - V[n_c_dft] | phi_j >'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") DIJ(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      WRITE(IO%IU6,'(1X,A,F8.5,A)') 'SET_DIJ: change to DION: for r >',PP%R%R(IRM),' [PS(DR)MAX]'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") EDIJ(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF

      ! Evaluate DION
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         PP%DION(CH1,CH2)=PP%DION(CH1,CH2)+DIJ(CH1,CH2)
      ENDDO
      ENDDO      

      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'SET_DIJ: DION'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") PP%DION(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF

      ! Make DION hermitian
      DO CH1=1,PP%LMAX
      DO CH2=CH1+1,PP%LMAX
         PP%DION(CH1,CH2)=(PP%DION(CH1,CH2)+PP%DION(CH2,CH1))/2
         PP%DION(CH2,CH1)=PP%DION(CH1,CH2)
      ENDDO
      ENDDO            

      RETURN
      END SUBROUTINE SET_DIJ


!******************** SUBROUTINE SET_PP ********************************
!
! SET_PP calculates the PAW parameters:
!
! ) The augmentation charges QION:
!  Q_ij = <phi_i|phi_j> - <\tilde{phi}_i|\tilde{phi}_j>
!
! ) The unscreened strength parameters DION:
! D_ij = <phi_i|T+V|phi_j> - <\tilde{phi}_i|T+\tilde{V}|\tilde{phi}_j>
!        -\int \hat{Q}^00_ij(r)\tilde{V}(r) dr
!
! with V=V_H[n_c+Z]+V_ref and \tilde{V}=V_H[\tilde{n_Zc}]+\tilde{V}_ref
! where V_ref and \tilde{V}_ref are the AE and PS potentials of the
! atomic reference system (valence contributions only), stored as
! -PP%POTAE and -PP%POTPS (note the sign!).
! In the above n_c is the HF core charge density and V_H[\tilde{n_Zc}]
! is Hartree potential of the ionic core, stored in PP%POTPSC.
! The charge density \hat{Q}^00_ij(r) is the compensation charge
! density for L=0 and M=0 [see Eq. (27) in PRB 59, 1758 (1999)]. 
!
! On exit the following components of PP have been set:
!   WAE,WPS,QTOT,QPAW,QION,DION
! (see the definition of TYPE potcar in pseudo.F )
!***********************************************************************
      SUBROUTINE SET_PP(PP,W0,W,IO)
      USE prec
      USE base
      USE pseudo
      USE radial
      USE constant
      USE paw
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W0,W
      TYPE(in_struct) IO
      ! local variables
      INTEGER CH1,CH2,I
      INTEGER IRM
      INTEGER LL,LLP,LMIN,LMAX,LMAIN
      INTEGER IRMAX,NMAX_STORE
      INTEGER LM,LMP,LYMAX,LMMAX
      REAL(q) SUM,SCALE
      REAL(q) QINTP,QINTW,QINF
      REAL(q) DQINFPS,DQINFAE,DQINFAEP
      REAL(q) DLM(PP%LMDIM*PP%LMDIM)
      REAL(q), DIMENSION(PP%LMAX,PP%LMAX) :: QION,DIJ,DION,DD
      OVERLAP DLLMM(PP%LMDIM,PP%LMDIM,1),DHXC(PP%LMDIM,PP%LMDIM)
      REAL(q) TMP(PP%R%NMAX)
      REAL(q), DIMENSION(W%R%NMAX) :: RHO,VOLD,VNEW
      REAL(q), ALLOCATABLE :: VTMP(:,:,:)

      ! Copy the partial waves from W to PP
      PP%WAE(1:PP%R%NMAX,1:PP%LMAX)=W%WAE(1:PP%R%NMAX,1:PP%LMAX)
      PP%WPS(1:PP%R%NMAX,1:PP%LMAX)=W%WPS(1:PP%R%NMAX,1:PP%LMAX)

      DO IRMAX=1,PP%R%NMAX-1
         IF (PP%RDEP>0 .AND. PP%R%R(IRMAX)-PP%RDEP > -5E-3) EXIT
      ENDDO
!     WRITE(*,*) 'test SET_PP',IRMAX,PP%R%NMAX

      ! set the simpson weights in accordance with IRMAX
      NMAX_STORE=PP%R%NMAX; PP%R%NMAX=IRMAX
      CALL SET_SIMP(PP%R)

      ! Recalculate QTOT, QPAW, and QION
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         ! quantum numbers l and lp of these two channels
         LL =PP%LPS(CH1)
         LLP=PP%LPS(CH2)
         IF (LL==LLP) THEN
            TMP=0
            TMP(1:IRMAX)=PP%WAE(1:IRMAX,CH1)*PP%WAE(1:IRMAX,CH2)
            CALL SIMPI(PP%R,TMP,SUM)
            PP%QTOT(CH1,CH2)=SUM
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
            TMP(:)=(PP%WAE(:,CH1)*PP%WAE(:,CH2)- &
           &   PP%WPS(:,CH1)*PP%WPS(:,CH2))*PP%R%R(:)**LMAIN
            CALL SIMPI(PP%R,TMP,SUM)
            PP%QPAW(CH1,CH2,LMAIN)=SUM
         ENDDO
      ENDDO
      ENDDO

      QION(1:PP%LMAX,1:PP%LMAX)=PP%QION(1:PP%LMAX,1:PP%LMAX)
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         IF (PP%LPS(CH1)==PP%LPS(CH2)) THEN
            PP%QION(CH1,CH2)=PP%QPAW(CH1,CH2,0)
         ENDIF
      ENDDO
      ENDDO

      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'SET_PP: change to QION'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") PP%QION(CH1,CH2)-QION(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      WRITE(IO%IU6,*) 'SET_PP: QION'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") PP%QION(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF

      ! Calculate V_H[n_Zc] of the original core
      SCALE=1/(2*SQRT(PI))
      RHO=W0%RHOAE*SCALE
      ! Calculate the Hartree potential of the AE core electrons (V_H[n_c])
      VOLD=0
      CALL RAD_POT_HAR(0, W0%R, VOLD, RHO, SUM)
      ! add the Coulomb potential of the nucleus
      VOLD=VOLD*SCALE - FELECT/W0%R%R*W0%Z
#ifdef debug
      ! integrate charge
      CALL SIMPI(PP%R,PP%RHOAE,QINTP); CALL SIMPI(W%R,RHO,QINTW)
      ! infer integrated charge from Hartree potential
      QINF=VOLD(PP%R%NMAX)*PP%R%R(PP%R%NMAX)/FELECT
      ! write
      IF (IO%IU6>0) WRITE(*,'(A,F14.7,A,F14.7)') 'SET_PP: dQ_int=',(QINTP-QINTW)/SCALE,' Q_inf=',-QINF
#endif      

      ! Store the new AE core charge density (n_c) into PP%RHOAE
      PP%RHOAE(1:PP%R%NMAX)=W%RHOAE(1:PP%R%NMAX)*SCALE
      RHO=W%RHOAE*SCALE
      ! Calculate the Hartree potential of the AE core electrons (V_H[n_c])
      VNEW=0
      CALL RAD_POT_HAR(0, W%R, VNEW, RHO, SUM)
      ! add the Coulomb potential of the nucleus
      VNEW=VNEW*SCALE - FELECT/W%R%R*W%Z
#ifdef debug
      IF (IO%IU6>0) WRITE(99,'(4F20.10)') (PP%R%R(I),VOLD(I),VNEW(I),PP%POTPSC(I),I=1,PP%R%NMAX)
#endif
      
      IF (IO%IU6>0) THEN
         WRITE(IO%IU6,'(/X,A,X,A2)') 'SET_PP:',PP%ELEMENT
         WRITE(IO%IU6,'(7X,A,6X,4X,A,4X,A,2X,1X,A,4X,A,3X,4X,A)') &
        &   'r','dQ_inf(PS)','dQ_inf(old)','dQ_inf(new)','dV(old)','dV(new)'
         IRM=IR_PSMAX(PP)
         DQINFPS =W%ZVAL+PP%POTPSC(IRM)*PP%R%R(IRM)/FELECT
         DQINFAE =W%ZVAL+VOLD(IRM)*PP%R%R(IRM)/FELECT
         DQINFAEP=W%ZVAL+VNEW(IRM)*PP%R%R(IRM)/FELECT
         WRITE(IO%IU6,'(6F14.7)') PP%R%R(IRM),DQINFPS,DQINFAE,DQINFAEP, &
        &   VOLD(IRM)-PP%POTPSC(IRM),VNEW(IRM)-PP%POTPSC(IRM)
         
         IRM=PP%R%NMAX
         DQINFPS =W%ZVAL+PP%POTPSC(IRM)*PP%R%R(IRM)/FELECT
         DQINFAE =W%ZVAL+VOLD(IRM)*PP%R%R(IRM)/FELECT
         DQINFAEP=W%ZVAL+VNEW(IRM)*PP%R%R(IRM)/FELECT
         WRITE(IO%IU6,'(6F14.7)') PP%R%R(IRM),DQINFPS,DQINFAE,DQINFAEP, &
        &   VOLD(IRM)-PP%POTPSC(IRM),VNEW(IRM)-PP%POTPSC(IRM)
         WRITE(IO%IU6,*)       
      ENDIF

! test
!     PP%POTAE=0
!     PP%POTPS=0
! test

      ! Recalculate DIJ
      DIJ=0
      DO CH1=1,PP%LMAX
      DO CH2=1,PP%LMAX
         TMP=0
         IF (PP%LPS(CH1)==PP%LPS(CH2)) THEN
            DO I=1,PP%R%NMAX
               TMP(I)=PP%WAE(I,CH1)*W%PARWKINAE(I,CH2)-PP%WPS(I,CH1)*W%PARWKINPS(I,CH2)+ &
              &   PP%WAE(I,CH1)*VNEW(I)*PP%WAE(I,CH2)-PP%WPS(I,CH1)*PP%POTPSC(I)*PP%WPS(I,CH2) &
              &  -PP%WAE(I,CH1)*PP%POTAE(I)*PP%WAE(I,CH2)+PP%WPS(I,CH1)*PP%POTPS(I)*PP%WPS(I,CH2)
            ENDDO
            CALL SIMPI(PP%R,TMP,DIJ(CH1,CH2))
         ENDIF
      ENDDO
      ENDDO

      ! "unscreen"
      LYMAX=MAXVAL(PP%LPS(1:PP%LMAX))
      IF (ASSOCIATED(PP%QPAW)) LYMAX=LYMAX*2
      LMMAX=(LYMAX+1)**2
      ALLOCATE(VTMP(PP%R%NMAX,LMMAX,1))
      VTMP=0; VTMP(:,1,1)=(-PP%POTPS(:)+PP%POTPSC(:))/SCALE
      ! Reconstruct the PAW strength parameters for the reference system
      CALL RAD_POT_WEIGHT(PP%R,1,LYMAX,VTMP)
      DLM=0; DLLMM=0; DHXC=0
      CALL RAD_AUG_PROJ(VTMP(:,:,1),PP%R,DLM,PP%LMAX,PP%LPS,LYMAX,PP%AUG,PP%QPAW)
      CALL TRANS_DLM(DLLMM(:,:,1),DLM,PP)

      DHXC=DLLMM(:,:,1)

      ! Compute DION
      LM=1
      DO CH1=1,PP%LMAX
      LMP=1
      DO CH2=1,PP%LMAX
         DION(CH1,CH2)=DIJ(CH1,CH2)+DHXC(LM,LMP)
!        DION(CH1,CH2)=DIJ(CH1,CH2)
!        DION(CH1,CH2)=DIJ(CH1,CH2)+PP%E(CH2)*PP%QION(CH1,CH2)+DHXC(LM,LMP)
         LMP=LMP+2*PP%LPS(CH2)+1
      ENDDO
      LM=LM+2*PP%LPS(CH1)+1
      ENDDO      

      ! Make DION hermitian
      DO CH1=1,PP%LMAX
      DO CH2=CH1+1,PP%LMAX
         DION(CH1,CH2)=(DION(CH1,CH2)+DION(CH2,CH1))/2
         DION(CH2,CH1)=DION(CH1,CH2)
      ENDDO
      ENDDO            

      DD(1:PP%LMAX,1:PP%LMAX)=DION(1:PP%LMAX,1:PP%LMAX)-PP%DION(1:PP%LMAX,1:PP%LMAX)
      PP%DION(1:PP%LMAX,1:PP%LMAX)=DION(1:PP%LMAX,1:PP%LMAX)
      
      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'SET_PP: change to DION'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") DD(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      WRITE(IO%IU6,*) 'SET_PP: DION'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") PP%DION(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF

      DEALLOCATE(VTMP)

      RETURN
      END SUBROUTINE SET_PP


!******************** SUBROUTINE HF_CORE_VALENCE_EXCH ******************
!
! HF_CORE_VALENCE_EXCH adds the Hartree-Fock core-valence exchange
! interaction to the PAW strength parameters DION
!
! sum_c int dr' phi_c(r') phi_v(r') int dr phi_c(r) phi_v(r)/(r-r')
!
! where phi_c are the HF wave functions of the core states, and phi_v
! are the AE partial waves.
! 
!***********************************************************************
      SUBROUTINE HF_CORE_VALENCE_EXCH(PP,W,IO)
      USE prec
      USE base
      USE paw
      USE pseudo
      USE radial
      USE pawfock
      IMPLICIT NONE
      TYPE(atomic) W
      TYPE(in_struct) IO
      TYPE(potcar), POINTER :: PP
      ! local variables
      INTEGER CH1,CH2,CHANNELS
      INTEGER LYMAX,LMMAX,M
      INTEGER IRM,K
      INTEGER, ALLOCATABLE :: LC(:)      
      REAL(q), ALLOCATABLE :: A(:,:)
      REAL(q), ALLOCATABLE :: S(:,:,:,:,:)
      OVERLAP, ALLOCATABLE :: COCC(:,:),DFOCK(:,:),DHARTREE(:,:)
      REAL(q) TMP(PP%LMAX,PP%LMAX)
      REAL(q) SUM
      REAL(q) RTMP(PP%R%NMAX)
      
      CHANNELS=W%NSCORE+PP%LMAX

!     ALLOCATE(A(PP%R%NMAX,CHANNELS),LC(CHANNELS))
      ALLOCATE(A(W%R%NMAX,CHANNELS),LC(CHANNELS))
      
      A=0
      ! Copy core wave functions to A
      DO CH1=1,W%NSCORE
         LC(CH1)=W%L(CH1)
!        A(1:PP%R%NMAX,CH1)=W%W(1:PP%R%NMAX,CH1)
!        A(1:PP%R%NMAX,CH1)=W%A(1:PP%R%NMAX,CH1)
         A(1:W%R%NMAX,CH1)=W%A(1:W%R%NMAX,CH1)
      ENDDO
      ! Copy valence partial waves to A
      DO CH1=1,PP%LMAX
         LC(W%NSCORE+CH1)=PP%LPS(CH1)
!        A(:,W%NSCORE+CH1)=PP%WAE(:,CH1)
         A(:,W%NSCORE+CH1)=W%WAE(:,CH1)
!        IRM=IR_MATCH(PP,CH1)
!        DO K=IRM+1,PP%R%NMAX
!           IF (ABS(A(K,W%NSCORE+CH1))>maxParWaveAmplitude) THEN
!              A(K:PP%R%NMAX,W%NSCORE+CH1)=0
!              EXIT
!           ENDIF
!        ENDDO
      ENDDO

      LYMAX =3    ! no g orbitals at this point
      ALLOCATE(S(CHANNELS,CHANNELS,CHANNELS,CHANNELS,0:LYMAX))

      S=0
!     CALL COLOUMB_4TERM(A,LC,PP%R,CHANNELS,S,8)
      CALL COLOUMB_4TERM(A,LC,W%R,CHANNELS,S,8)

      LMMAX=0
      DO CH1=1,CHANNELS
         LMMAX=LMMAX+(LC(CH1)*2+1)
      ENDDO
      ALLOCATE(COCC(LMMAX,LMMAX),DHARTREE(LMMAX,LMMAX),DFOCK(LMMAX,LMMAX))
      
      ! Set diagonal components of the occupancy matrix (core only)
      COCC=0
      LMMAX=0
      DO CH1=1,W%NSCORE
         DO M=1,LC(CH1)*2+1
            COCC(LMMAX+M,LMMAX+M)=1
         ENDDO
         LMMAX=LMMAX+(LC(CH1)*2+1)
      ENDDO

      CALL CALC_DHARTREE(S,COCC,CHANNELS,LC,DHARTREE,DFOCK)

      TMP=PP%DION

      CALL ADD_CDIJ(DFOCK(LMMAX+1:,LMMAX+1:),PP)

      TMP=PP%DION-TMP
      IF (IO%IU6>0) THEN
      WRITE(IO%IU6,*) 'HF_CORE_VALENCE_EXCH: DION: contributions from core-valence exchange'
      DO CH1=1,PP%LMAX
         WRITE(IO%IU6,'(I4)',ADVANCE="No") CH1
         DO CH2=1,PP%LMAX
            WRITE(IO%IU6,'(F14.7)',ADVANCE="No") TMP(CH1,CH2)
         ENDDO
         WRITE(IO%IU6,*)
      ENDDO
      ENDIF      

      DEALLOCATE(A,S,COCC,DHARTREE,DFOCK)
            
      RETURN
      END SUBROUTINE HF_CORE_VALENCE_EXCH


!******************** SUBROUTINE REPORT ********************************
!
! REPORT writes the atomic eigenvalues to OUTCAR
!      
!***********************************************************************
      SUBROUTINE REPORT(PP,W,IO,TAG)
      USE prec
      USE base
      USE pseudo
      IMPLICIT NONE
      TYPE(atomic) W
      TYPE(potcar), POINTER :: PP
      TYPE(in_struct) IO
      CHARACTER(LEN=*), OPTIONAL :: TAG
      ! local variables
      INTEGER I
      CHARACTER(LEN=1) :: LLABEL(4)=(/ "S", "P", "D", "F" /)
      
      IF (IO%IU6>0) THEN
         IF (PRESENT(TAG)) THEN
            WRITE(IO%IU6,'(/,2X,A,A2,X,A)') 'Atomic solution for: ',PP%ELEMENT,TAG
         ELSE
            WRITE(IO%IU6,'(/,2X,A,A2)') 'Atomic solution for: ',PP%ELEMENT
         ENDIF
         WRITE(IO%IU6,'(1X,A)')    '-------------------------'
         DO I=1,W%NS
            WRITE(IO%IU6,'(2X,I1,A1,4X,F16.8)') W%N(I),LLABEL(W%L(I)+1),W%E(I)
         ENDDO
      ENDIF
      
      RETURN
      END SUBROUTINE REPORT


!******************** SUBROUTINE WSHOOT ********************************
!      
!***********************************************************************
      SUBROUTINE WSHOOT(R,N,L,Z,V,E,AP,XA,XB,A,B)
      USE prec
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(rgrid) R
      INTEGER N,L,Z
      REAL(q) E
      REAL(q), DIMENSION(R%NMAX) :: V,AP,XA,XB,A,B
      ! local variables
      INTEGER NTIMES
      INTEGER KI,KJ,K
      INTEGER NNODES,NODES,NERR
      REAL(q) EMIN,EMAX
      REAL(q) AH(R%NMAX),BH(R%NMAX)
      REAL(q) AA,BB,TMP(R%NMAX)
      REAL(q) MA,MB,DA,DB
      REAL(q) RA,WN,RJ
      REAL(q) QCOEF,WMIN,DE,DEP
      LOGICAL DBL,LCONV

      INTEGER, PARAMETER :: NTRIES=1000

      EMIN = -Z*Z*RYTOEV*2/(N)**2
      EMAX = -1.0E-5

      IF (E<EMIN.OR.E>EMAX) THEN
          E=(EMAX+EMIN)/2
      ENDIF

      NNODES = N-L-1
      NERR=0

      DEP=0
      LCONV=.FALSE.

      A=AP; AH=AP

 tries: DO NTIMES=1,NTRIES
         NERR=0

         ! outward integration of inhomogeneous equations 
         CALL OUTINT(E,L,Z,R,V,KI,A,B,NERR,IHA=XA,IHB=XB)

         MA=A(KI)
         MB=B(KI)

         ! inward integration of inhomogeneous equations
         CALL INWINT(E,L,R,V,KI,KJ,A,B,IHA=XA,IHB=XB)

         DA=A(KI)-MA

         ! inward integration of homogeneous equations
         CALL INWINT(E,L,R,V,KI,KJ,AH,BH)

         ! make A continuous at KI
         DO K=KI,KJ
            A(K)=A(K)-DA/AH(KI)*AH(K)
            B(K)=B(K)-DA/AH(KI)*BH(K)
         ENDDO

         DB=MB-B(KI)
         
         ! count nodes in the interval [1,KI]
         NODES=0
         DO K=2,KI
            NODES=NODES+.501*ABS(SIGN(1._q,A(K))-SIGN(1._q,A(K-1)))
         ENDDO
         
         ! too few nodes
         IF (NODES<NNODES) THEN
            IF (E>EMIN) EMIN=E
            E=0.5_q*(E+EMAX)
            IF ((EMAX-EMIN)<(E*1E-10)) THEN
               WRITE(*,'(A,I5,2I4,3F10.5)') 'WSHOOT: too few nodes: ',NODES,N,L,EMIN,E,EMAX
               STOP
            ENDIF
            CYCLE tries      
         ENDIF

         ! too many nodes
         IF (NODES>NNODES) THEN
            IF (E<EMAX) EMAX=E
            E=0.5*(E+EMIN)
            IF ((EMAX-EMIN)<(E*1E-10)) THEN 
               WRITE(*,'(A,I5,2I4,3F10.5)') 'WSHOOT: too many nodes: ',NODES,N,L,EMIN,E,EMAX
               STOP
            ENDIF
            CYCLE tries
         ENDIF         
         
         ! set wave function beyond KJ
         IF(MIN(ABS(A(KJ)),ABS(B(KJ)))<1.0E-25_q) THEN
            ! to zero
            A(KJ+1:R%NMAX)=0; B(KJ+1:R%NMAX)=0
         ELSE
            ! exponential decay
            WMIN=(1.E-35_q)/MIN(ABS(A(KJ)),ABS(B(KJ)))
            QCOEF=SQRT(-(E+E))
            RJ=R%RSTART*R%D**(KJ-1)
            RA=RJ
            DO K=KJ+1,R%NMAX
               RA=RA*R%D
               WN=EXP(QCOEF*(RJ-RA))
               IF (WN<WMIN) THEN
                  A(K:R%NMAX)=0; B(K:R%NMAX)=0
                  EXIT
               ENDIF     
               A(K)=WN*A(KJ)
               B(K)=WN*B(KJ)
            ENDDO
         ENDIF

         ! count nodes in the interval [KI,KJ]
         NODES=0
         DO K=KI,KJ
            NODES=NODES+.501*ABS(SIGN(1._q,A(K))-SIGN(1._q,A(K-1)))
         ENDDO

         ! compute the square of the norm
         TMP=A*A
         CALL SIMPI(R,TMP,AA)
         TMP=B*B
         CALL SIMPI(R,TMP,BB)
         WN=AA+BB/C/C
         
         !first order change in the eigenvalue
         DE=DB*A(KI)/WN
          
         IF (DE*DEP<0) DE=DE*0.5
         IF ((E>-0.001).AND.(DE>0.01)) LCONV=.TRUE.
         IF ((EMAX-EMIN)<(-E*1E-10)) LCONV=.TRUE.
         IF (DE>0) EMIN=E
         IF (DE<0) EMAX=E
         IF (.NOT. LCONV) E=E+DE
         IF (ABS(DE)<(-E*1E-10)) LCONV=.TRUE.
         IF ((E>=EMAX).OR.(E<=EMIN)) E=0.5*(EMAX+EMIN)
         IF (NODES/=0) LCONV=.FALSE.
         DEP=DE

         IF (LCONV) THEN
            WN=1./SQRT(WN)
            A=A*WN
            B=B*WN    
            IF (NERR/=0) &
           &   WRITE(*,'("WSHOOT:",I3," error ocurred in Milne procedure")') NERR 
!           WRITE(*,'("n=",I4," l=",I4," j=",F5.1," energy interval:",4F16.8)') N,L,J,EMIN,E,EMAX,DE
!           WRITE(*,'(A,I4)') 'ntimes=',NTIMES
            RETURN
         ENDIF
      ENDDO tries

      ! solution was not found after NTRIES attempts
      WRITE(*,'("WSHOOT: ERROR, solution not found within ",I4," attempts")') NTRIES
      WRITE(*,'("n=",I4," l=",I4," energy interval:",2F20.10)') N,L,EMIN,EMAX
      ! count nodes in the interval [KI,KJ]
      NODES=0
      DO K=KI,KJ
         NODES=NODES+.501*ABS(SIGN(1._q,A(K))-SIGN(1._q,A(K-1)))
      ENDDO
      IF (NODES/=0) WRITE(*,'(A,I4,2(X,I4))') 'WSHOOT: nodes in inward integration: ',NODES,KI,KJ
#ifdef debug
      WRITE(999,'(3F20.10)') (R%R(K),A(K),B(K),K=1,R%NMAX)
#endif
      STOP    
      
      RETURN
      END SUBROUTINE WSHOOT


!******************** SUBROUTINE PWSHOOT *******************************
!      
!***********************************************************************
      SUBROUTINE PWSHOOT(R,L,Z,E,V,XA,XB,A,B,IRM)
      USE prec
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(rgrid) R
      INTEGER L,Z
      INTEGER IRM
      REAL(q), DIMENSION(R%NMAX) :: V,XA,XB,A,B
      ! local variables
      INTEGER K
      INTEGER NTIMES
      INTEGER KI,NERR
      INTEGER NODES0,NODES
      REAL(q) LGDR,LGDR0
      REAL(q) E,EMIN,EMAX,EP
      REAL(q), ALLOCATABLE :: TMP(:),ATMP(:),BTMP(:)
      INTEGER, PARAMETER :: NTRIES=100
      REAL(q), PARAMETER :: EDIFF=1E-7
      REAL(q), PARAMETER :: LGDRDIFF=1E-7

      ! Define the search interval
      EMIN=E-10*RYTOEV
      EMAX=E+10*RYTOEV

      ALLOCATE(TMP(R%NMAX),ATMP(R%NMAX),BTMP(R%NMAX))

      ! Count initial number of nodes
      ! inside the matching radius
      NODES0=0
      DO K=2,IRM-1
         NODES0=NODES0+.501*ABS(SIGN(1._q,A(K))-SIGN(1._q,A(K-1)))
      ENDDO

      ! Boundary condition at the matching point
      CALL GRAD(R,A,TMP)
      LGDR0=TMP(IRM)/A(IRM)

      ATMP=A; BTMP=B
      
      tries: DO NTIMES=1,NTRIES
         EP=E
         
         ! outward integration of the imhomogeneous equations
         CALL OUTINT(E,L,Z,R,V,KI,ATMP,BTMP,NERR,RC=R%R(IRM+5),IHA=XA,IHB=XB)
         
         CALL GRAD(R,ATMP,TMP)
         LGDR=TMP(IRM)/ATMP(IRM)

         ! Count current number of nodes inside the matching radius
         NODES=0
         DO K=2,IRM
            NODES=NODES+.501*ABS(SIGN(1._q,ATMP(K))-SIGN(1._q,ATMP(K-1)))
         ENDDO
         
         IF (NODES<NODES0) THEN
            EMIN=E
            E=(E+EMAX)/2
            CYCLE tries
         ENDIF
         
         IF (NODES>NODES0) THEN
            EMAX=E
            E=(E+EMIN)/2
            CYCLE tries
         ENDIF

         IF (ABS(LGDR-LGDR0)<LGDRDIFF) EXIT tries
         
         IF (LGDR>LGDR0) THEN
            EMIN=E
            E=(E+EMAX)/2                        
         ENDIF         

         IF (LGDR<LGDR0) THEN
            EMAX=E
            E=(E+EMIN)/2            
         ENDIF         
         
         IF (ABS(E-EP)<EDIFF) EXIT tries
      ENDDO tries

      IF (NTIMES>NTRIES) THEN
         WRITE(*,'(A,I4,A)') &
        &   'PWSHOOT: ERROR, unable to match logarithmic derivatives within',NTRIES,' attempts.'
         WRITE(*,'(2I4,5F14.7)') NODES0,NODES,LGDR0,LGDR,E,EMIN,EMAX
#ifdef debug
         WRITE(999,'(2F20.10)') (R%R(K),ATMP(K),K=1,R%NMAX)
#endif
         STOP
      ENDIF

      A(1:IRM-1)=ATMP(1:IRM-1)*(A(IRM)/ATMP(IRM))
      
      DEALLOCATE(TMP)
      
      RETURN
      END SUBROUTINE PWSHOOT


!******************** SUBROUTINE WMIX **********************************
!      
!***********************************************************************
      SUBROUTINE WMIX(W,WP,DWMAX,LRESET)
      USE prec
      IMPLICIT NONE
      TYPE(atomic) W
      TYPE(atomic) WP
      REAL(q), OPTIONAL :: DWMAX
      LOGICAL, OPTIONAL :: LRESET
      ! local variables
      INTEGER I,K
      REAL(q) A,B
      REAL(q) DW(W%NS)
      REAL(q), ALLOCATABLE, SAVE :: DWPREV(:),BPREV(:)
 
      IF (PRESENT(LRESET)) THEN
         IF (LRESET) THEN
            IF (ALLOCATED(BPREV)) DEALLOCATE(BPREV)
            IF (ALLOCATED(DWPREV)) DEALLOCATE(DWPREV)
         ENDIF
      ENDIF
      
      IF (.NOT.ALLOCATED(DWPREV)) THEN
         ALLOCATE(DWPREV(W%NS),BPREV(W%NS))
         DWPREV=0; BPREV=0.2_q
      ENDIF
            
      DW=0
      DO I=1,W%NS
         DO K=1,SIZE(W%W,1)
            IF (ABS(W%A(K,I)-WP%A(K,I))>ABS(DW(I))) &
           &    DW(I)=WP%A(K,I)-W%A(K,I)
         ENDDO
         
         B=BPREV(I)
         IF ((DW(I)*DWPREV(I)>0_q).AND.(BPREV(I)<=0.8_q)) &
        &   B=B+0.1_q
         IF ((DW(I)*DWPREV(I)<0_q).AND.(BPREV(I)>=0.2_q)) &
        &   B=B-0.1_q
         A=1._q-B
         BPREV(I)=B

#ifdef debug
!        WRITE(*,'(A,I4,A,F4.1,A,F4.1)') 'i=',i,' a=',A,'  b=',B
#endif
         
         DO K=1,SIZE(W%W,1)
            W%A(K,I)=B*W%A(K,I)+A*WP%A(K,I)
            W%B(K,I)=B*W%B(K,I)+A*WP%B(K,I)
         ENDDO
      ENDDO
      
      DWPREV=DW
      
      IF (PRESENT(DWMAX)) DWMAX=MAXVAL(ABS(DW))
      
      RETURN
      END SUBROUTINE WMIX


!******************** SUBROUTINE RAD_POT_DIRECT ************************
!      
!***********************************************************************
      SUBROUTINE RAD_POT_DIRECT(W,V,ILO,IHI)
      USE prec
      USE pseudo      
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) V(W%R%NMAX)
      INTEGER, OPTIONAL :: ILO,IHI
      ! local variables
      INTEGER I
      INTEGER ISTART,IEND
      REAL(q) SUM
      REAL(q) SCALE
      REAL(q) RHO(W%R%NMAX)
      
      SCALE=1/(2*SQRT(PI))

      ! Compute the charge density
      RHO=0
      ISTART=1; IEND=W%NS
      IF (PRESENT(ILO)) ISTART=ILO
      IF (PRESENT(IHI)) IEND=IHI
      DO I=ISTART,IEND
         RHO(:)=RHO(:)+(W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C)*W%OCC(I)
      ENDDO
      RHO=RHO*SCALE
      ! Electronic contribution
      V=0
      CALL RAD_POT_HAR(0, W%R, V, RHO, SUM)
      ! Nuclear contribution
      V=V*SCALE - FELECT/W%R%R*W%Z
      
      RETURN
      END SUBROUTINE RAD_POT_DIRECT


!******************** SUBROUTINE RAD_POT_DIRECT_VALENCE_ONLY ***********
!
!***********************************************************************
      SUBROUTINE RAD_POT_DIRECT_VALENCE_ONLY(W,V)
      USE prec
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) V(W%R%NMAX)
      ! local variables
      INTEGER I
      REAL(q) SUM
      REAL(q) SCALE
      REAL(q) RHO(W%R%NMAX)

      SCALE=1/(2*SQRT(PI))

      ! Compute the charge density
      RHO=0
      DO I=W%NSCORE+1,W%NS
         RHO(:)=RHO(:)+(W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C)*W%OCC(I)
      ENDDO
      RHO=RHO*SCALE
      ! Electronic contribution
      V=0
      CALL RAD_POT_HAR(0, W%R, V, RHO, SUM)
      V=V*SCALE

      RETURN
      END SUBROUTINE RAD_POT_DIRECT_VALENCE_ONLY


!******************** SUBROUTINE RAD_POT_EXCHANGE **********************
!      
!***********************************************************************
      SUBROUTINE RAD_POT_EXCHANGE(W,XA,XB,VX,JLO,JHI)
      USE prec
      USE asa
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(atomic) W
      REAL(q) XA(W%R%NMAX,W%NS)
      REAL(q) XB(W%R%NMAX,W%NS)
      REAL(q) VX(W%R%NMAX,W%NS)
      INTEGER, OPTIONAL :: JLO,JHI
      ! local variables
      INTEGER I,J
      INTEGER LI,LJ,MI,MJ,M
      INTEGER LL,LMIN,LMAX
      INTEGER LMINDX,ISTART,IEND,IC
      INTEGER JSTART,JEND
      REAL(q) SUM
      REAL(q) SCALE
      REAL(q) V(W%R%NMAX)
      REAL(q) RHO(W%R%NMAX)
      REAL(q), ALLOCATABLE, SAVE :: LLL(:,:,:)
      
      ! Setup LLL(li,lj,L)=\sum_M \sum_mi \sum_mj |<li,mi,lj,mj|L,M>|^2
      IF (.NOT.ALLOCATED(LLL)) THEN
         ALLOCATE(LLL(0:3,0:3,0:6))
         LLL=0
         DO LI=0,3
         DO LJ=0,3
            CALL YLM3LOOKUP(LI,LJ,LMINDX)
            DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMINDX=LMINDX+1
               ISTART=INDCG(LMINDX)
               IEND  =INDCG(LMINDX+1)
               DO IC=ISTART,IEND-1
                  LLL(LI,LJ,JL(IC))=LLL(LI,LJ,JL(IC))+YLM3(IC)*YLM3(IC)
               ENDDO
            ENDDO
            ENDDO
         ENDDO
         ENDDO
      ENDIF

      JSTART=1; JEND=W%NS 
      IF (PRESENT(JLO)) JSTART=JLO
      IF (PRESENT(JHI)) JEND=JHI
      
      XA=0; XB=0; VX=0       
      DO I=1,W%NS
      DO J=JSTART,JEND
         LI=W%L(I); LJ=W%L(J)
         LMIN=ABS(LI-LJ); LMAX=LI+LJ
         
         SCALE=W%OCC(J)/((2*LI+1)*(2*LJ+1))
         
         RHO(:)=W%A(:,I)*W%A(:,J)+W%B(:,I)*W%B(:,J)/C/C
                  
         DO LL=LMIN,LMAX,2
            CALL RAD_POT_HAR(LL, W%R, V, RHO, SUM)
            XA(:,I)=XA(:,I)+SCALE*LLL(LI,LJ,LL)*V(:)*W%A(:,J)
            XB(:,I)=XB(:,I)+SCALE*LLL(LI,LJ,LL)*V(:)*W%B(:,J)
            IF (I==J) THEN
               VX(:,I)=VX(:,I)+SCALE*LLL(LI,LJ,LL)*V(:)
            ENDIF 
         ENDDO
      ENDDO
      ENDDO
      XA=XA/2; XB=XB/2; VX=VX/2

!     IF (noSmallExchange) XB=0

      RETURN
      END SUBROUTINE RAD_POT_EXCHANGE


!******************** SUBROUTINE RAD_POT_EXCHANGE_SINGLE ***************
!      
!***********************************************************************
      SUBROUTINE RAD_POT_EXCHANGE_SINGLE(L,WA,WB,W,XA,XB)
      USE prec
      USE asa
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(atomic) W
      INTEGER L
      REAL(q), DIMENSION(W%R%NMAX) :: WA,WB,XA,XB
      ! local variables
      INTEGER J
      INTEGER LI,LJ,MI,MJ,M
      INTEGER LL,LMIN,LMAX
      INTEGER LMINDX,ISTART,IEND,IC
      REAL(q) SUM
      REAL(q) SCALE
      REAL(q) V(W%R%NMAX)
      REAL(q) RHO(W%R%NMAX)
      REAL(q), ALLOCATABLE, SAVE :: LLL(:,:,:)
      
      ! Setup LLL(li,lj,L)=\sum_M \sum_mi \sum_mj |<li,mi,lj,mj|L,M>|^2
      IF (.NOT.ALLOCATED(LLL)) THEN
         ALLOCATE(LLL(0:3,0:3,0:6))
         LLL=0
         DO LI=0,3
         DO LJ=0,3
            CALL YLM3LOOKUP(LI,LJ,LMINDX)
            DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMINDX=LMINDX+1
               ISTART=INDCG(LMINDX)
               IEND  =INDCG(LMINDX+1)
               DO IC=ISTART,IEND-1
                  LLL(LI,LJ,JL(IC))=LLL(LI,LJ,JL(IC))+YLM3(IC)*YLM3(IC)
               ENDDO
            ENDDO
            ENDDO
         ENDDO
         ENDDO
      ENDIF
 
      XA=0; XB=0       
      DO J=1,W%NS
         LI=L; LJ=W%L(J)
         LMIN=ABS(LI-LJ); LMAX=LI+LJ
         
         SCALE=W%OCC(J)/((2*LI+1)*(2*LJ+1))
         
         RHO(:)=WA(:)*W%A(:,J)+WB(:)*W%B(:,J)/C/C
                  
         DO LL=LMIN,LMAX,2
            CALL RAD_POT_HAR(LL, W%R, V, RHO, SUM)
            XA(:)=XA(:)+SCALE*LLL(LI,LJ,LL)*V(:)*W%A(:,J)
            XB(:)=XB(:)+SCALE*LLL(LI,LJ,LL)*V(:)*W%B(:,J)
         ENDDO
      ENDDO
      XA=XA/2; XB=XB/2
      
!     IF (noSmallExchange) XB=0

      RETURN
      END SUBROUTINE RAD_POT_EXCHANGE_SINGLE


!*********************** SUBROUTINE RAD_POT_DFT ************************
!
!***********************************************************************
      SUBROUTINE RAD_POT_DFT(RHO,Z,R,V,EXC,LXC,LGGA)

      USE prec
      USE constant
      USE radial
      USE setexm

      IMPLICIT NONE

      TYPE(rgrid) R      
      INTEGER Z
      REAL(q) RHO(R%NMAX)
      REAL(q) V(R%NMAX)
      REAL(q), OPTIONAL :: EXC
      LOGICAL, OPTIONAL :: LXC
      LOGICAL, OPTIONAL :: LGGA
      ! local variables
      INTEGER K
      REAL(q) SCALE
      REAL(q) QINT,QINF
      REAL(q), DIMENSION(:,:,:), ALLOCATABLE :: RWRK1,VWRK
      REAL(q), DIMENSION(:,:),ALLOCATABLE :: RWRK2
      REAL(q) EXCG,DHARTREE,DEXC,DVXC,DEXC_GGA,DVXC_GGA,DOUBLEC
      LOGICAL LADDXC,LADDGC
      ! Use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TREL=.TRUE.
!     LOGICAL,PARAMETER :: TREL=.FALSE.
      ! Calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      ! in this case non spherical contributions are missing
      LOGICAL,PARAMETER :: TLDA=.TRUE.
      ! running averaging-out of spikes in GGA potential
      INTEGER, PARAMETER :: IVAV=10,IVAVT=5
      INTEGER IREFIN,IFAULT,I,J,IPOINT,N
      REAL(q) VAV,VAV2,VAR

      LADDXC=.TRUE.
      IF (PRESENT(LXC)) LADDXC=LXC

      LADDGC=.TRUE.
      IF (PRESENT(LGGA)) LADDGC=LGGA
      
      SCALE=2*SQRT(PI)

      ALLOCATE(RWRK1(R%NMAX,1,1),RWRK2(R%NMAX,1),VWRK(R%NMAX,1,1))
      VWRK=0
      RWRK1(:,1,1)=RHO(:)/SCALE

      ! Hartree potential
      DHARTREE=0
      CALL RAD_POT_HAR(0,R,VWRK(:,1,1),RWRK1(:,1,1),DHARTREE)

#ifdef debug
      ! integrate charge
      CALL SIMPI(R,RWRK1(:,1,1),QINT)
      ! infer integrated charge from Hartree potential
      QINF=VWRK(R%NMAX,1,1)*R%R(R%NMAX)/FELECT/2/SQRT(PI)
      ! write
!     WRITE(*,'(A,F14.7,A,F14.7)') 'RAD_POT_DFT: Q_int=',QINT*SCALE,' Q_inf=',QINF
#endif      

      ! Add nuclear potential
      DO K=1,R%NMAX
         VWRK(K,1,1)=VWRK(K,1,1)-FELECT*SCALE*Z/R%R(K)
      ENDDO

      ! LDA exchange correlation energy, potential,
      ! and double counting corrections
      DEXC=0
      DVXC=0
      
      DO K=1,R%NMAX
         RWRK2(K,1)=RWRK1(K,1,1)/(SCALE*R%R(K)*R%R(K))
      ENDDO
      
 lda: IF (TLDA.AND.LADDXC) THEN
         CALL RAD_LDA_XC(R,TREL,0,RWRK2(:,1),RWRK1(:,:,1),VWRK(:,:,1),DEXC,DVXC,.FALSE.)
      ENDIF lda

      ! GGA if required
      DEXC_GGA=0
      DVXC_GGA=0
      
 gga: IF (ISGGA().AND.LADDXC.AND.LADDGC) THEN
         CALL RAD_GGA_XC(R,TLDA,RWRK2(:,1),RWRK1(:,1,1),VWRK(:,1,1),DEXC_GGA,DVXC_GGA)

         ! smooth the potential near the practical infinity (if necessary)
         refinement: DO IREFIN=1,100
            IFAULT=0
            DO I=1,R%NMAX
               ! Examine only points at large distances ( > 10 a.u. should be enough ?)
               IF ((R%R(I)<10._q*AUTOA).OR.(I<=IVAV)) CYCLE 
               VAV=0._q; VAV2=0._q; IPOINT=0
               ! Averaged potential around point I (excluding I and a small surrounding
               ! because at I we could have some   l o c a l   problems ...)
               DO J=I-IVAV,MIN(I+IVAV,N)
                  IF (ABS(I-J)<IVAVT) CYCLE 
                  IPOINT=IPOINT+1
                  VAV=VAV+VWRK(J,1,1)
                  VAV2=VAV2+VWRK(J,1,1)*VWRK(J,1,1)
               ENDDO 
               VAV =VAV/IPOINT
               VAV2=VAV2/IPOINT
               VAR =SQRT(VAV2-VAV*VAV)
               VAV=VAV
               ! The 'smoothness criterion' below is quite empirical 
               IF (ABS(VWRK(I,1,1)-VAV)>VAR*4) THEN
#ifdef debug
                  WRITE(*,*) '<V>, <V V>, sigma', VAV*VAV,VAV2,VAR
                  WRITE(*,*) 'Refinement step ',IREFIN
                  WRITE(*,*) 'Serious problem with potential at R =',R%R(I)
                  WRITE(*,*) 'Potential VR(R)                     =',VWRK(I,1,1)
                  WRITE(*,*) 'Resetting potential to VR_new(R)    =',VAV
#endif
                  VWRK(I,1,1)=VAV
                  IFAULT=IFAULT+1
               ENDIF
            ENDDO
            IF (IFAULT==0) EXIT refinement
         ENDDO refinement
      ENDIF gga

      V(:)=VWRK(:,1,1)/SCALE

      ! Classical double counting correction:
      ! E_dc = -1/2 \int rho(r) V_H[rho(r)] dr + E_xc[rho+rhoc]
      !          -  \int rho(r) V_xc[rho(r)+rhoc(r)] dr
      DOUBLEC= -DHARTREE/2+DEXC-DVXC+DEXC_GGA-DVXC_GGA

      EXCG= DEXC+DEXC_GGA
      IF (PRESENT(EXC)) EXC=EXCG
            
      DEALLOCATE(RWRK1,RWRK2,VWRK)
      RETURN
      END SUBROUTINE RAD_POT_DFT


!tb beg
!*********************** SUBROUTINE RAD_POT_DFT2 ************************
!includes compensation background charge needed for anions
!***********************************************************************
      SUBROUTINE RAD_POT_DFT2(RHO,RHO_comp,Z,R,V,qstate,DOUBLEC,EXC,LXC,LGGA)

      USE prec
      USE constant
      USE radial
      USE setexm

      IMPLICIT NONE

      TYPE(rgrid) R      
      INTEGER Z
      REAL(q) RHO(R%NMAX),RHO_comp(R%NMAX)
      REAL(q) V(R%NMAX)
      REAL(q), OPTIONAL :: EXC
      LOGICAL, OPTIONAL :: LXC
      LOGICAL, OPTIONAL :: LGGA
      ! local variables
      INTEGER K
      REAL(q) SCALE
      REAL(q) QINT,QINF
      REAL(q), DIMENSION(:,:,:), ALLOCATABLE :: RWRK1,VWRK,VWRK2
      REAL(q), DIMENSION(:,:),ALLOCATABLE :: RWRK2
      REAL(q) EXCG,DHARTREE,DEXC,DVXC,DEXC_GGA,DVXC_GGA,DOUBLEC
      LOGICAL LADDXC,LADDGC
      ! Use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TREL=.TRUE.
!     LOGICAL,PARAMETER :: TREL=.FALSE.
      ! Calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      ! in this case non spherical contributions are missing
      LOGICAL,PARAMETER :: TLDA=.TRUE.
      INTEGER :: qstate
      ! running averaging-out of spikes in GGA potential
      INTEGER, PARAMETER :: IVAV=10,IVAVT=5
      INTEGER IREFIN,IFAULT,I,J,IPOINT,N
      REAL(q) VAV,VAV2,VAR

      LADDXC=.TRUE.
      IF (PRESENT(LXC)) LADDXC=LXC

      LADDGC=.TRUE.
      IF (PRESENT(LGGA)) LADDGC=LGGA
      
      SCALE=2*SQRT(PI)

      ALLOCATE(RWRK1(R%NMAX,1,1),RWRK2(R%NMAX,1),VWRK(R%NMAX,1,1),VWRK2(R%NMAX,1,1))

      VWRK=0
      !IF (qstate .LT. 0) THEN
      IF (qstate .NE. 0) THEN
        RWRK1(:,1,1)=(RHO_comp(:)+RHO(:))/SCALE
       
      ELSE 
        RWRK1(:,1,1)=RHO(:)/SCALE
      ENDIF

      ! Hartree potential
      DHARTREE=0
      CALL RAD_POT_HAR(0,R,VWRK(:,1,1),RWRK1(:,1,1),DHARTREE)


#ifdef debug
      ! integrate charge
      CALL SIMPI(R,RWRK1(:,1,1),QINT)
      ! infer integrated charge from Hartree potential
      QINF=VWRK(R%NMAX,1,1)*R%R(R%NMAX)/FELECT/2/SQRT(PI)
      ! write
!     WRITE(*,'(A,F14.7,A,F14.7)') 'RAD_POT_DFT: Q_int=',QINT*SCALE,' Q_inf=',QINF
#endif      

      !c don't use compensation charge in XC calculation
      !IF (qstate .LT. 0) THEN
      IF (qstate .NE. 0) THEN
        RWRK1(:,1,1)=RWRK1(:,1,1)-RHO_comp(:)/SCALE
      ENDIF

      ! Add nuclear potential
      DO K=1,R%NMAX
         VWRK(K,1,1)=VWRK(K,1,1)-FELECT*SCALE*Z/R%R(K)
      ENDDO

      ! LDA exchange correlation energy, potential,
      ! and double counting corrections
      DEXC=0
      DVXC=0
      
      DO K=1,R%NMAX
         RWRK2(K,1)=RWRK1(K,1,1)/(SCALE*R%R(K)*R%R(K))
      ENDDO
      
 lda: IF (TLDA.AND.LADDXC) THEN
         CALL RAD_LDA_XC(R,TREL,0,RWRK2(:,1),RWRK1(:,:,1),VWRK(:,:,1),DEXC,DVXC,.FALSE.)
      ENDIF lda

      ! GGA if required
      DEXC_GGA=0
      DVXC_GGA=0

gga: IF (ISGGA().AND.LADDXC.AND.LADDGC) THEN
         CALL RAD_GGA_XC(R,TLDA,RWRK2(:,1),RWRK1(:,1,1),VWRK(:,1,1),DEXC_GGA,DVXC_GGA)

         ! smooth the potential near the practical infinity (if necessary)
         refinement: DO IREFIN=1,100
            IFAULT=0
            DO I=1,R%NMAX
               ! Examine only points at large distances ( > 10 a.u. should be enough ?)
               IF ((R%R(I)<10._q*AUTOA).OR.(I<=IVAV)) CYCLE
               VAV=0._q; VAV2=0._q; IPOINT=0
               ! Averaged potential around point I (excluding I and a small surrounding
               ! because at I we could have some   l o c a l   problems ...)
               DO J=I-IVAV,MIN(I+IVAV,N)
                  IF (ABS(I-J)<IVAVT) CYCLE
                  IPOINT=IPOINT+1
                  VAV=VAV+VWRK(J,1,1)
                  VAV2=VAV2+VWRK(J,1,1)*VWRK(J,1,1)
               ENDDO
               VAV =VAV/IPOINT
               VAV2=VAV2/IPOINT
               VAR =SQRT(VAV2-VAV*VAV)
               VAV=VAV
               ! The 'smoothness criterion' below is quite empirical 
               IF (ABS(VWRK(I,1,1)-VAV)>VAR*4) THEN
#ifdef debug
                  WRITE(*,*) '<V>, <V V>, sigma', VAV*VAV,VAV2,VAR
                  WRITE(*,*) 'Refinement step ',IREFIN
                  WRITE(*,*) 'Serious problem with potential at R =',R%R(I)
                  WRITE(*,*) 'Potential VR(R)                     =',VWRK(I,1,1)
                  WRITE(*,*) 'Resetting potential to VR_new(R)    =',VAV
#endif
                  VWRK(I,1,1)=VAV
                  IFAULT=IFAULT+1
               ENDIF
            ENDDO
            IF (IFAULT==0) EXIT refinement
         ENDDO refinement
      ENDIF gga
      
!  gga: IF (ISGGA().AND.LADDXC) THEN
!          CALL RAD_GGA_XC(R,TLDA,RWRK2(:,1),RWRK1(:,1,1),VWRK(:,1,1),DEXC_GGA,DVXC_GGA)
!       ENDIF gga

      V(:)=VWRK(:,1,1)/SCALE

      ! Classical double counting correction:
      ! E_dc = -1/2 \int rho(r) V_H[rho(r)] dr + E_xc[rho+rhoc]
      !          -  \int rho(r) V_xc[rho(r)+rhoc(r)] dr
      DOUBLEC= -DHARTREE/2+DEXC-DVXC+DEXC_GGA-DVXC_GGA

!       WRITE(*,*) "DOUBLEC",DOUBLEC
!       WRITE(*,*) "DHARTREE",DHARTREE
!       WRITE(*,*) "DEXC",DEXC
!       WRITE(*,*) "DVXC",DVXC
!       WRITE(*,*) "DEXC_GGA",DEXC_GGA    
!       WRITE(*,*) "DVXC_GGA",DVXC_GGA

      EXCG= DEXC+DEXC_GGA
      IF (PRESENT(EXC)) EXC=EXCG
            
      DEALLOCATE(RWRK1,RWRK2,VWRK,VWRK2)
      RETURN
      END SUBROUTINE RAD_POT_DFT2

      SUBROUTINE RAD_POT_DFT3(RHO,Z,R,V,qdummy,Rcomp,DOUBLEC,EXC,LXC)
      !c compensation charge via Watson sphere
      USE prec
      USE constant
      USE radial
      USE setexm

      IMPLICIT NONE

      TYPE(rgrid) R      
      INTEGER Z
      REAL(q) RHO(R%NMAX)
      REAL(q) V(R%NMAX)
      REAL(q), OPTIONAL :: EXC
      LOGICAL, OPTIONAL :: LXC
      ! local variables
      INTEGER I,K
      REAL(q) SCALE
      REAL(q) QINT,QINF
      REAL(q), DIMENSION(:,:,:), ALLOCATABLE :: RWRK1,VWRK,VWRK2
      REAL(q), DIMENSION(:,:),ALLOCATABLE :: RWRK2
      REAL(q) EXCG,DHARTREE,DEXC,DVXC,DEXC_GGA,DVXC_GGA,DOUBLEC
      LOGICAL LADDXC
      ! Use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TREL=.TRUE.
!     LOGICAL,PARAMETER :: TREL=.FALSE.
      ! Calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      ! in this case non spherical contributions are missing
      LOGICAL,PARAMETER :: TLDA=.TRUE.
     ! INTEGER :: qstate !c charge of ion = minus compensation charge
      REAL(q) :: qdummy
      REAL(q) :: Rcomp !c radius of Watson sphere

      LADDXC=.TRUE.
      IF (PRESENT(LXC)) LADDXC=LXC
      
      SCALE=2*SQRT(PI)

      ALLOCATE(RWRK1(R%NMAX,1,1),RWRK2(R%NMAX,1),VWRK(R%NMAX,1,1),VWRK2(R%NMAX,1,1))

      VWRK=0
      !IF (qstate .LT. 0) THEN
      

      RWRK1(:,1,1)=RHO(:)/SCALE

      ! Hartree potential
      DHARTREE=0
      CALL RAD_POT_HAR(0,R,VWRK(:,1,1),RWRK1(:,1,1),DHARTREE)

      ! Add nuclear potential
      DO K=1,R%NMAX
         VWRK(K,1,1)=VWRK(K,1,1)-FELECT*SCALE*Z/R%R(K)
      ENDDO

      !c potential due to Watson sphere
      IF (qdummy .LT. 0._q) THEN
        DO K=1,R%NMAX
          IF (R%R(K) .LE. Rcomp) THEN
            VWRK(K,1,1)=VWRK(K,1,1)+FELECT*SCALE*qdummy/Rcomp
          ELSE
            VWRK(K,1,1)=VWRK(K,1,1)+FELECT*SCALE*qdummy/R%R(K)
          ENDIF
        ENDDO
      ENDIF

      ! LDA exchange correlation energy, potential,
      ! and double counting corrections
      DEXC=0
      DVXC=0
      
      DO K=1,R%NMAX
         RWRK2(K,1)=RWRK1(K,1,1)/(SCALE*R%R(K)*R%R(K))
      ENDDO
      
 lda: IF (TLDA.AND.LADDXC) THEN
         CALL RAD_LDA_XC(R,TREL,0,RWRK2(:,1),RWRK1(:,:,1),VWRK(:,:,1),DEXC,DVXC,.FALSE.)
      ENDIF lda

      ! GGA if required
      DEXC_GGA=0
      DVXC_GGA=0
      
 gga: IF (ISGGA().AND.LADDXC) THEN
         CALL RAD_GGA_XC(R,TLDA,RWRK2(:,1),RWRK1(:,1,1),VWRK(:,1,1),DEXC_GGA,DVXC_GGA)
      ENDIF gga

      V(:)=VWRK(:,1,1)/SCALE

      ! Classical double counting correction:
      ! E_dc = -1/2 \int rho(r) V_H[rho(r)] dr + E_xc[rho+rhoc]
      !          -  \int rho(r) V_xc[rho(r)+rhoc(r)] dr
      DOUBLEC= -DHARTREE/2+DEXC-DVXC+DEXC_GGA-DVXC_GGA

!       WRITE(*,*) "DOUBLEC",DOUBLEC
!       WRITE(*,*) "DHARTREE",DHARTREE
!       WRITE(*,*) "DEXC",DEXC
!       WRITE(*,*) "DVXC",DVXC
!       WRITE(*,*) "DEXC_GGA",DEXC_GGA    
!       WRITE(*,*) "DVXC_GGA",DVXC_GGA

      EXCG= DEXC+DEXC_GGA
      IF (PRESENT(EXC)) EXC=EXCG
            
      DEALLOCATE(RWRK1,RWRK2,VWRK,VWRK2)
      RETURN
      END SUBROUTINE RAD_POT_DFT3

!tb end


!******************** SUBROUTINE INITWF ********************************
!      
!***********************************************************************
      SUBROUTINE INITWF(PP,W,IO)
      USE prec
      USE cl
      USE ini
      USE base
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      TYPE(in_struct) IO
      ! local variables
      INTEGER I,J,CHANNEL
      REAL(q) SCALE,SUM,WN,WNA,ALPHA
      REAL(q) F,FDER
      REAL(q), DIMENSION(PP%R%NMAX) :: RHO,V
      REAL(q), DIMENSION(W%R%NMAX) :: TMP
      REAL(q), PARAMETER :: TINY = 1E-5_q

      SCALE=1/(2*SQRT(PI))

      V=0
      ! Hartree potential of the core electrons
      CALL RAD_POT_HAR(0, PP%R, V, PP%RHOAE, SUM)
      ! Coulomb potential of the nucleus
      V=V*SCALE - FELECT/PP%R%R*W%Z      
      ! Reference valence atomic potential
      V=V-PP%POTAE
#ifdef debug
      IF (IO%IU6>0) WRITE(300,'(2F20.10)') (PP%R%R(J),V(J), J=1,PP%R%NMAX)
#endif
      ! Compute the core wave functions      
      W%W=0; W%A=0; W%B=0
      DO I=1,W%NSCORE
         CALL CORE_WAVE_FKT(RHO, W%E(I), W%N(I), W%L(I), V, PP%R, NINT(W%Z), &
        &   A_=W%A(1:PP%R%NMAX,I), B_=W%B(1:PP%R%NMAX,I))
         W%B(:,I)=W%B(:,I)*C
         ! Normalize
         TMP(:)=W%A(:,I)*W%A(:,I)
         CALL SIMPI(W%R,TMP,WNA)
         TMP(:)=W%A(:,I)*W%A(:,I)+W%B(:,I)*W%B(:,I)/C/C
         CALL SIMPI(W%R,TMP,WN)         
         W%W(:,I)=W%A(:,I)*SQRT(WN/WNA)         
      ENDDO
      ! Copy the relevant valence partial waves
      I=W%NSCORE+1
      DO CHANNEL=1,PP%LMAX
         ! Cycle when this channel is unoccupied
         IF (PP%QATO(CHANNEL,CHANNEL)<TINY) CYCLE
         ! Consistency check
         IF (I>W%NS) THEN
            WRITE(*,'(A,I3)') 'INITWF: ERROR, W inconsistently dimensioned:',CHANNEL
            STOP
         ENDIF
         ! Consistency check
         IF (W%L(I)/=PP%LPS(CHANNEL)) THEN
            WRITE(*,'(A,2I3)') 'INITWF: ERROR, angular moments do not match:',PP%LPS(CHANNEL),W%L(I)
            STOP
         ENDIF
         W%A(1:PP%R%NMAX,I)=PP%WAE(1:PP%R%NMAX,CHANNEL)
         W%W(1:PP%R%NMAX,I)=PP%WAE(1:PP%R%NMAX,CHANNEL)
         I=I+1
      ENDDO
      ! Sanity check
      IF ((I-1)/=W%NS) THEN
         WRITE(*,'(A,2I3)') 'INITWF: ERROR, did not find all wave functions:',I-1,W%NS
         STOP      
      ENDIF
      ! Extend the valence wave function if necessary
      IF (PP%R%NMAX<W%R%NMAX) THEN
         DO I=W%NSCORE+1,W%NS
            ! determine coefficient
            TMP=W%A(:,I)*W%A(:,I)
            CALL SIMPI(W%R,TMP,WN)
            IF (WN<1._q) THEN
               ALPHA=W%A(PP%R%NMAX,I)**2/(1-WN)/2
               DO J=PP%R%NMAX+1,W%R%NMAX
                  W%A(J,I)=W%A(PP%R%NMAX,I)*EXP(-ALPHA*(W%R%R(J)-W%R%R(PP%R%NMAX)))
               ENDDO
            ELSE
               W%A(PP%R%NMAX+1:W%R%NMAX,I)=0._q
            ENDIF
            ! normalize
            TMP=W%A(:,I)*W%A(:,I)
            CALL SIMPI(W%R,TMP,WN)
            W%A(:,I)=W%A(:,I)/SQRT(WN)
            W%W(:,I)=W%A(:,I)
         ENDDO    
      ENDIF
#ifdef debug
      IF (IO%IU6>0) THEN
      WRITE(*,*)
      WRITE(*,'(A)') 'INITWF: Core-Valence orthogonality'
      DO I=1,W%NSCORE
         WRITE(*,'(I4)',ADVANCE="No") I
         DO CHANNEL=1,PP%LMAX
            TMP=0 ; SUM=0
            IF (W%L(I)==PP%LPS(CHANNEL)) THEN
               DO J=1,PP%R%NMAX
                  TMP(J)=W%W(J,I)*PP%WAE(J,CHANNEL)
               ENDDO
            ENDIF
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM 
         ENDDO
         WRITE(*,*)
      ENDDO
      WRITE(*,*)
      ENDIF
#endif

      ! Transfer the projectors onto the radial grid
      W%BETA=0
      DO CHANNEL=1,PP%LMAX
         IF (PP%LPS(CHANNEL)>PP%LMAX_CALC) CYCLE
         ! n.b. we store r*P(r)
         grid: DO I=1,PP%R%NMAX
            IF (PP%R%R(I)>PP%PSPRNL(NPSRNL,1,1)) EXIT grid
            CALL SPLVAL(PP%R%R(I),F,FDER,PP%PSPRNL(1,1,CHANNEL),NPSRNL,NPSRNL)                  
            W%BETA(I,CHANNEL)=F*PP%R%R(I)
         ENDDO grid
      ENDDO
#ifdef debug
      IF (IO%IU6>0) THEN
      WRITE(*,*)
      WRITE(*,'(A)') 'INITWF: <p_i|tilde{phi}_j> duality'
      DO I=1,PP%LMAX
         WRITE(*,'(I4)',ADVANCE="No") I
         DO J=1,PP%LMAX
            TMP=0; SUM=0
            IF (PP%LPS(I)==PP%LPS(J)) TMP(:)=W%BETA(:,I)*PP%WPS(:,J)
            CALL SIMPI(W%R,TMP,SUM)
            WRITE(*,'(F14.7)',ADVANCE="No") SUM
         ENDDO
         WRITE(*,*)
      ENDDO
      WRITE(*,*)
      ENDIF
#endif

      RETURN
      END SUBROUTINE INITWF


!******************** SUBROUTINE ALLOCW *********************************
!      
!***********************************************************************
      SUBROUTINE ALLOCW(PP,W)
      USE prec
      USE pseudo
      USE radial
      USE constant
      USE cl
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(atomic) W
      ! local variables
      INTEGER I,N
      REAL(q) Z
      REAL(q), PARAMETER :: TINY = 1E-5_q

      ! copy the grid
      W%R=PP%R
      ! extend the grid if necessary
      I=PP%R%NMAX
      IF (PP%R%REND<RMAX) THEN
         I=0
         DO
            I=I+1
            IF (PP%R%RSTART*EXP(PP%R%H*(I-1))>=RMAX) EXIT
         ENDDO
      ENDIF
      W%R%NMAX=I
      ! Allocate grid
      NULLIFY(W%R%R,W%R%SI)
      ALLOCATE(W%R%R(W%R%NMAX),W%R%SI(W%R%NMAX))
      DO I=1,W%R%NMAX
         W%R%R(I)=W%R%RSTART*EXP(PP%R%H*(I-1))
      ENDDO
      W%R%REND=W%R%R(W%R%NMAX)
      CALL SET_SIMP(W%R)

      W%Z=SUM(PP%ATOMIC_OCC)
!     W%ZCORE=PP%ZCORE
      W%ZVAL=PP%ZVALF_ORIG
      W%ZCORE=W%Z-W%ZVAL

      N=0
      DO I=1,SIZE(PP%ATOMIC_E)
         IF (PP%ATOMIC_OCC(I)<TINY) CYCLE
         N=N+1
      ENDDO
      W%NS=N

      NULLIFY(W%E,W%N,W%L,W%OCC,W%W,W%A,W%B,W%WKINAE)
      ALLOCATE(W%E(N),W%N(N),W%L(N),W%OCC(N))
      ALLOCATE(W%W(W%R%NMAX,N),W%A(W%R%NMAX,N),W%B(W%R%NMAX,N),W%WKINAE(W%R%NMAX,N))
      W%E=0; W%N=0; W%L=0; W%OCC=0; W%W=0; W%A=0; W%B=0; W%WKINAE=0

      Z=0
      N=0
      W%NSCORE=0
      DO I=1,SIZE(PP%ATOMIC_E)
         IF (PP%ATOMIC_OCC(I)<TINY) CYCLE
         N=N+1
         W%E(N)=PP%ATOMIC_E(I)
         W%N(N)=PP%ATOMIC_N(I)
         W%L(N)=PP%ATOMIC_L(I)
         W%OCC(N)=PP%ATOMIC_OCC(I)
         Z=Z+W%OCC(N)
         IF ((Z-W%ZCORE)<TINY) W%NSCORE=W%NSCORE+1
      ENDDO
      W%NSVAL=W%NS-W%NSCORE

      NULLIFY(W%WAE,W%WPS,W%EPS,W%BETA,W%PARWKINAE,W%PARWKINPS)
      ALLOCATE(W%EPS(PP%LMAX))
      ALLOCATE(W%WAE(W%R%NMAX,PP%LMAX),W%WPS(W%R%NMAX,PP%LMAX))
      ALLOCATE(W%BETA(W%R%NMAX,PP%LMAX))
      ALLOCATE(W%PARWKINAE(W%R%NMAX,PP%LMAX),W%PARWKINPS(W%R%NMAX,PP%LMAX))
      W%WAE=0; W%WPS=0; W%BETA=0; W%PARWKINAE=0; W%PARWKINPS=0

      W%EPS(1:PP%LMAX)=PP%E(1:PP%LMAX)

      NULLIFY(W%RHOAE,W%RHOPS)
      ALLOCATE(W%RHOAE(W%R%NMAX),W%RHOPS(W%R%NMAX))
      W%RHOAE=0; W%RHOPS=0

      NULLIFY(W%RHO)
      ALLOCATE(W%RHO(W%R%NMAX))
      W%RHO=0

      NULLIFY(W%POTAE,W%POTPS,W%POTAEC,W%POTPSC)
      ALLOCATE(W%POTAE(W%R%NMAX),W%POTPS(W%R%NMAX),W%POTAEC(W%R%NMAX),W%POTPSC(W%R%NMAX))
      W%POTAE=0; W%POTPS=0; W%POTAEC=0; W%POTPSC=0
      
      RETURN
      END SUBROUTINE ALLOCW


!******************** SUBROUTINE DEALLOCW ******************************
!      
!***********************************************************************
      SUBROUTINE DEALLOCW(W)
      USE prec
      IMPLICIT NONE
      TYPE(atomic) W

      DEALLOCATE(W%E,W%L,W%OCC,W%W,W%A,W%B,W%WKINAE,W%R%R,W%R%SI)
      DEALLOCATE(W%WAE,W%WPS,W%BETA,W%PARWKINAE,W%PARWKINPS,W%EPS)
      DEALLOCATE(W%RHOAE,W%RHOPS)
      DEALLOCATE(W%POTAE,W%POTPS,W%POTAEC,W%POTPSC)
      DEALLOCATE(W%RHO)

      NULLIFY(W%E,W%N,W%L,W%OCC,W%W,W%A,W%B,W%WKINAE,W%R%R,W%R%SI)
      NULLIFY(W%WAE,W%WPS,W%BETA,W%PARWKINAE,W%PARWKINPS,W%EPS)
      NULLIFY(W%RHOAE,W%RHOPS)
      NULLIFY(W%POTAE,W%POTPS,W%POTAEC,W%POTPSC)
      NULLIFY(W%RHO)

      RETURN
      END SUBROUTINE DEALLOCW


!******************** SUBROUTINE OUTINT ********************************
!
!      Performs outward integration. The Runge-Kutta procedure
!      is used to provide the starting values for the MILNE
!      integration.
!
!***********************************************************************
      SUBROUTINE OUTINT(E,L,Z,R,APOT,KI,A,B,NERR,KAPPA,DPOT,RC,IHA,IHB)

      USE constant
      USE radial
      
      IMPLICIT NONE
      
      TYPE(rgrid) :: R
      
      INTEGER :: L                                ! angular moment
      INTEGER :: Z                                ! nuclear charge
      INTEGER :: NODES                            ! number of nodes
      INTEGER :: NERR                             ! error in MILNE-procedure
      INTEGER, OPTIONAL :: KAPPA                  ! relativistic quantum number

      REAL(q) :: APOT(R%NMAX)                     ! V(r)
      REAL(q), OPTIONAL :: DPOT(R%NMAX)           ! dV(r)/dr
      REAL(q) :: A(R%NMAX),B(R%NMAX)              ! Wave functions
      REAL(q), OPTIONAL :: RC                     ! Cutoff range for outward integration
      REAL(q), OPTIONAL :: IHA(R%NMAX)            ! Inhomogeneous part A   
      REAL(q), OPTIONAL :: IHB(R%NMAX)            ! Inhomogeneous part B  

      INTEGER :: KI,IK,K,KP1
      INTEGER :: KM1,KM2,KM3,KM4,KM5,KIT

      REAL(q) :: DX2
      REAL(q) :: RA
      REAL(q) :: VR(R%NMAX)                       ! Potential*r, P. in atomic units
      REAL(q) :: IHRA(R%NMAX)                     ! Inhomogeneous part * r, in atomic units
      REAL(q) :: IHRB(R%NMAX)                     ! Inhomogeneous part * r, in atomic units
      REAL(q) :: DV
      REAL(q) :: HOC,XK,P,XC,BGC,TR,TC,VC,UC,WC,IHCA,IHCB
      REAL(q) :: X,UNP,WNP,UNP2,WNP2,XMFT,TEST
      REAL(q) :: DA(5),DB(5)                      ! to calculate the wave functions
                                                  ! in Runge-Kutta Routine
      REAL(q) :: AP(R%NMAX),BP(R%NMAX)            ! temporary storage in MILNE-Routine
      REAL(q) :: E,EAU                            ! Energie, E. in atom units
      REAL(q) :: RN,RNOT                          ! R%REND, R%RSTART in atomic units

      REAL(q) :: LAMBDA,D,AA

      ! convert to atomic units
      EAU=E/(2*RYTOEV)
      VR=APOT/(2*RYTOEV)*R%R/AUTOA
      
      IHRA=0
      IF (PRESENT(IHA)) IHRA=IHA/(2*RYTOEV)*R%R/AUTOA
      IHRB=0
      IF (PRESENT(IHB)) IHRB=IHB/(2*RYTOEV)*R%R/AUTOA/C/C

      RNOT=R%RSTART/AUTOA
      RN=R%REND/AUTOA

      DX2=R%H/2._q
      XK=L*(L+1)
      XMFT=4.4444444444E-2_q*R%H
      TEST=1.0E6_q

      ! find the classical turning point
      IF (PRESENT(RC)) THEN
         RA=RN
         DO KI=R%NMAX-1,11,-1
            RA=RA/R%D
            IF (RA<=RC/AUTOA) EXIT
         ENDDO
         KI=KI+1
      ELSE
         RA=RN
         DO KI=R%NMAX-1,11,-1
            RA=RA/R%D
            IF ((EAU*RA)>=VR(KI))  EXIT
         ENDDO
         KI=KI+1
         IF ((KI+10)>=R%NMAX) KI=R%NMAX-11
      ENDIF

      ! set up starting value, A(1) has to be supplied
!     B(1)=REAL(L,q)/2._q/RNOT*A(1)
      D=1/(1+(2*C*C+EAU)*RNOT/Z)
      LAMBDA=SQRT(L*(L+1)-Z**2/C/C+(1+D)**2/4)+(1-D)/2-1
      AA=-(2*(1+EAU/C/C)*Z)/((LAMBDA+2+D)*(LAMBDA+1)-L*(L+1)+Z**2/C/C)
!     B(1)=C*C*D/Z*A(1)*(LAMBDA+AA*RNOT/(1+AA*RNOT))
      B(1)=C*C*D/Z*A(1)*LAMBDA

! Use the 4th-order Runge-Kutta procedure to setup the starting
! values necessary for the following Milne predictor-corrector
! numerical integration.
!
! 4th-order Runge-Kutta
!
! k_1 = h/2 f( x_n,       y_n )
! k_2 = h/2 f( x_n + h/2, y_n + 1/2 k_1 )
! k_3 = h/2 f( x_n + h/2, y_n + 1/2 k_2 )
! k_4 = h/2 f( x_n + h,   y_n + k_3 )
! 
! y_n+1 = y_n + 1/3 k_1 + 2/3 k_2 + 2/3 k_3 + 1/3 k_4 
!
      X=LOG(RNOT)
      DO K=1,5
         KP1=K+1
         XC=X
         BGC=-VR(K)
         IHCA=IHRA(K)
         IHCB=IHRB(K)
         IF (PRESENT(DPOT)) DV=DPOT(K)/(2*RYTOEV)
         WC=B(K)
         UC=A(K)
         DO IK=1,4
            RA=EXP(XC)
            TR=2*RA
            TC=EAU*RA+BGC
            VC=(1/C/C)*TC
            DA(IK)=DX2*((VC+TR)*WC+UC+IHCB)
            DB(IK)=DX2*((-1.)*WC-(TC-XK/(VC+TR))*UC-IHCA)
            IF (PRESENT(KAPPA)) THEN
               DA(IK)=DX2*((VC+TR)*WC-KAPPA*UC)
               DB(IK)=DX2*(KAPPA*WC-TC*UC-IHCA)
            ENDIF
            IF (PRESENT(DPOT)) THEN
               DA(IK)=DX2*((VC+TR)*WC+UC)
               DB(IK)=DX2*((-1.)*WC-(TC-XK/(VC+TR))*UC-DV/2*RA*RA/C/C/(VC+TR)/(VC+TR)*(KAPPA+1)*UC-IHCA)
            ENDIF
            SELECT CASE(IK)
            CASE(1)
               XC=XC+DX2
               UC=UC+DA(1)
               WC=WC+DB(1)
               BGC=0.5_q*(BGC-VR(KP1))
               IHCA=0.5_q*(IHCA+IHRA(KP1))
               IHCB=0.5_q*(IHCB+IHRB(KP1))
               IF (PRESENT(DPOT)) DV =0.5_q*(DV+DPOT(KP1)/(2*RYTOEV))
            CASE(2)
               UC=UC+DA(2)-DA(1)
               WC=WC+DB(2)-DB(1)
            CASE(3)
               XC=XC+DX2
               UC=UC+2*DA(3)-DA(2)
               WC=WC+2*DB(3)-DB(2)
               BGC=-VR(KP1)
               IHCA=IHRA(KP1)
               IHCB=IHRB(KP1)
               IF (PRESENT(DPOT)) DV = DPOT(KP1)/(2*RYTOEV)
            END SELECT
         ENDDO
         IK=4
         B(KP1)=B(K)+(DB(1)+DB(4)+2*(DB(2)+DB(3)))*0.33333333333333_q
         A(KP1)=A(K)+(DA(1)+DA(4)+2*(DA(2)+DA(3)))*0.33333333333333_q
         AP(KP1)=B(KP1)*(VC+TR)+A(KP1)+IHCB
         BP(KP1)=-B(KP1)-(TC-XK/(VC+TR))*A(KP1)-IHCA
         IF (PRESENT(KAPPA)) THEN
            AP(KP1)=B(KP1)*(VC+TR)-KAPPA*A(KP1)
            BP(KP1)=KAPPA*B(KP1)-TC*A(KP1)-IHCA
         ENDIF
         IF (PRESENT(DPOT)) THEN
            AP(KP1)=B(KP1)*(VC+TR)+A(KP1)
            BP(KP1)=-B(KP1)-(TC-XK/(VC+TR))*A(KP1)-DV/2*RA*RA/C/C/(VC+TR)/(VC+TR)*(KAPPA+1)*A(KP1)-IHCA
         ENDIF
         X=X+R%H
      ENDDO

! 5th order Milne integration (predictor-corrector)
      RA=EXP(X)
      DO K=6,KI-1
         RA=RA*R%D
         KP1=K+1
         KM1=K-1
         KM2=K-2
         KM3=K-3
         KM4=K-4
         KM5=K-5
         TR=2*RA
         TC=EAU*RA-VR(KP1)
         IHCA=IHRA(KP1)
         IHCB=IHRB(KP1)
         IF (PRESENT(DPOT)) DV=DPOT(KP1)/(2*RYTOEV)
         VC=1/C/C*TC
! predictor
         UNP=A(KM5)+0.3_q*R%H*(11*(AP(K)+AP(KM4))+26*AP(KM2)-14*(AP(KM1)+AP(KM3)))
         WNP=B(KM5)+0.3_q*R%H*(11*(BP(K)+BP(KM4))+26*BP(KM2)-14*(BP(KM1)+BP(KM3)))
         KIT=0

  33     AP(KP1)=(VC+TR)*WNP+UNP+IHCB
         BP(KP1)=(-1)*WNP-(TC-XK/(VC+TR))*UNP-IHCA
         IF (PRESENT(KAPPA)) THEN
            AP(KP1)=(VC+TR)*WNP-KAPPA*UNP
            BP(KP1)=KAPPA*WNP-TC*UNP-IHCA
         ENDIF
         IF (PRESENT(DPOT)) THEN
            AP(KP1)=(VC+TR)*WNP+UNP
            BP(KP1)=(-1)*WNP-(TC-XK/(VC+TR))*UNP-DV/2*RA*RA/C/C/(VC+TR)/(VC+TR)*(KAPPA+1)*UNP-IHCA
         ENDIF
! corrector         
         UNP2=A(KM3)+(7*(AP(KP1)+AP(KM3))+32*(AP(KM2)+AP(K))+12*AP(KM1))*XMFT
         WNP2=B(KM3)+(7*(BP(KP1)+BP(KM3))+32*(BP(KM2)+BP(K))+12*BP(KM1))*XMFT

         IF((ABS(TEST*(UNP2-UNP)) > ABS(UNP2)) &
        &    .OR. (ABS(TEST*(WNP2-WNP)) > ABS(WNP2))) THEN
            IF (KIT < 10) THEN
               KIT=KIT+1
               WNP=WNP2
               UNP=UNP2
               GOTO 33
            ELSE IF (NERR >= 0) THEN
               NERR=NERR+1
            ELSE
               WRITE(*,83) E,K,UNP2,UNP,WNP2,WNP
  83           FORMAT(10X,"HARD TEST, E= ",E20.8,I5,4E20.8)
            ENDIF
         ENDIF

         B(KP1)=WNP2
         A(KP1)=UNP2

      ENDDO
      RETURN

      END SUBROUTINE OUTINT


!******************** SUBROUTINE INWINT ********************************
!
!      PERFORM THE  INWARD INTEGRATION.   THIS ROUTINE IS A
!      DERIVATIVE OF START2/DIFF2.   MUCH UNNECESSARY INDEXING
!      HAS BEEN REMOVED AMONG OTHER THINGS.    DALE KOELLING
!      MODIFICATION TO FOLLY13 EQUATIONS...DECEMBER 1, 1976...BNH.
!
!***********************************************************************
      SUBROUTINE INWINT(E,L,R,APOT,KI,KJ,A,B,KAPPA,DPOT,IHA,IHB)

      USE radial
      USE constant

      IMPLICIT NONE

      TYPE (rgrid) :: R                  ! Grid

      INTEGER :: L                       ! angular moment
      INTEGER, OPTIONAL :: KAPPA         ! relativistic quantum number
      INTEGER :: KJ,KI                   ! Start-, Endpunkt der Int.
      REAL(q) :: E                       ! energy
      REAL(q) :: APOT(R%NMAX)            ! V(r)
      REAL(q), OPTIONAL :: DPOT(R%NMAX)  ! dV(r)/dr
      REAL(q), OPTIONAL :: IHA(R%NMAX)
      REAL(q), OPTIONAL :: IHB(R%NMAX)
      REAL(q) :: A(R%NMAX),B(R%NMAX)     ! wave function

      INTEGER :: M,K,I      
      REAL(q) :: EAU                     ! E in atom units
      REAL(q) :: VR(R%NMAX)              ! V(r) in atom units
      REAL(q) :: IHRA(R%NMAX)
      REAL(q) :: IHRB(R%NMAX)
      REAL(q) :: DV
      REAL(q) :: DA(5),DB(5)     
      REAL(q) :: RN,RNOT                 ! R%REND, R%RSTART in atomunits
      REAL(q) :: RA
      REAL(q) :: DR                      ! Gridvariable
      REAL(q) :: H3                      ! Hilfsvariable
      REAL(q) :: RP12,RP21,ATK,BTK
! test
      REAL(q),PARAMETER :: EPS=75.       ! VALUE USED TO DETERMINE THE PRACTICAL INFINITY
!     REAL(q),PARAMETER :: EPS=2500.     ! VALUE USED TO DETERMINE THE PRACTICAL INFINITY
! test
      REAL(q),PARAMETER :: F1=1.045833333,F2=2.691666666,F3=1.1, &
             F4=0.4416666666, F5=0.07916666666 
      REAL(q),PARAMETER :: G1=0.9666666666,G2=4.133333333,G3=0.8, &
             G4=0.1333333333,G5=0.03333333333
      REAL(q),PARAMETER :: FG1=0.9333333333,FG2=4.266666666,FG3=1.6

      H3=-R%H/3.
      DR=1./R%D

      ! convert to atomic units
      EAU=E/(2*RYTOEV)
      VR=APOT/(2*RYTOEV)*R%R/AUTOA

      IHRA=0
      IF (PRESENT(IHA)) IHRA=IHA/(2*RYTOEV)*R%R/AUTOA
      IHRB=0
      IF (PRESENT(IHB)) IHRB=IHB/(2*RYTOEV)*R%R/AUTOA/C/C
      
      RNOT=R%RSTART/AUTOA
      RN=R%REND/AUTOA

      RA=RNOT*(R%D**(KI+10))       

      DO KJ=KI+11,R%NMAX-1
         IF(RA*(VR(KJ)-EAU*RA) > EPS) EXIT
         RA=RA*R%D
      ENDDO

      ! set up starting values, A(KJ) has to be supplied
      IF (ABS(A(KJ))<1.E-10_q) A(KJ)=1E-10_q 
!     B(KJ)=SQRT(-EAU/(2.0_q*C*C+EAU))*A(KJ)
!     WRITE(*,'(A,I4,A,I4,A,I4,3F20.15)') 'ki=',KI,' kj=',KJ,' nmax=',R%NMAX,RA*AUTOA,A(KJ),B(KJ)
      B(KJ)=-A(KJ)/(2+EAU/C/C)*SQRT(-2*EAU*(1+EAU/C/C/2))

      DO M=1,4
         K=KJ-M
         A(K)=A(KJ)
         B(K)=B(KJ)
      ENDDO

      DO I=1,6
         K=KJ+1
         RA=RN*DR**(R%NMAX-K)
         DO M=1,5
            K=K-1
            RA=RA*DR
            RP12=RA+RA+(EAU*RA-VR(K))/C/C
            RP21=VR(K)-EAU*RA+L*(L+1.)/RP12
            IF (PRESENT(DPOT)) DV=DPOT(K)/(2*RYTOEV)
            DA(M)=H3*(1.*A(K)+RP12*B(K)+IHRB(K))
            DB(M)=H3*((-1.)*B(K)+RP21*A(K)-IHRA(K))
            IF (PRESENT(KAPPA)) THEN
               DA(M)=H3*(-KAPPA*A(K)+RP12*B(K))
               DB(M)=H3*(KAPPA*B(K)-(EAU*RA-VR(K))*A(K))
            ENDIF
            IF (PRESENT(DPOT)) THEN
               DA(M)=H3*(1.*A(K)+RP12*B(K))
               DB(M)=H3*((-1.)*B(K)+RP21*A(K)-DV/2*RA*RA/C/C/RP12/RP12*(KAPPA+1)*A(K))
            ENDIF
         ENDDO
         M=KJ-1
         A(M)=A(KJ)+F1*DA(1)+F2*DA(2)+F4*DA(4)-(F3*DA(3)+F5*DA(5))
         B(M)=B(KJ)+F1*DB(1)+F2*DB(2)+F4*DB(4)-(F3*DB(3)+F5*DB(5))
         M=M-1
         A(M)=A(KJ)+G1*DA(1)+G2*DA(2)+G3*DA(3)+G4*DA(4)-G5*DA(5)
         B(M)=B(KJ)+G1*DB(1)+G2*DB(2)+G3*DB(3)+G4*DB(4)-G5*DB(5)
         M=M-1
         A(M)=A(KJ)+1.0125_q*DA(1)+3.825_q*DA(2)+2.7_q*DA(3)+1.575_q*DA(4) &
                 -0.1125_q*DA(5)
         B(M)=B(KJ)+1.0125_q*DB(1)+3.825_q*DB(2)+2.7_q*DB(3)+1.575_q*DB(4) &
                 -0.1125_q*DB(5)
         M=M-1
         A(M)=A(KJ)+FG1*DA(1)+FG2*DA(2)+FG3*DA(3)+FG2*DA(4)+FG1*DA(5)
         B(M)=B(KJ)+FG1*DB(1)+FG2*DB(2)+FG3*DB(3)+FG2*DB(4)+FG1*DB(5)
      ENDDO 

      DA(1)=DA(2)
      DA(2)=DA(3)
      DA(3)=DA(4)
      DB(1)=DB(2)
      DB(2)=DB(3)
      DB(3)=DB(4)
      K=KJ-3
      RA=RN*DR**(R%NMAX-K)

      DO K=KJ-4,KI,-1
         RA=RA*DR
         RP12=RA+RA+(EAU*RA-VR(K))/C/C
         RP21=VR(K)-EAU*RA+L*(L+1.)/RP12
         IF (PRESENT(DPOT)) DV=DPOT(K)/(2*RYTOEV)
         ATK=A(K+4)+8.0_q*(DA(3)+DA(1)-0.5_q*DA(2))
         BTK=B(K+4)+8.0_q*(DB(3)+DB(1)-0.5_q*DB(2))
         DA(4)=H3*(1.*ATK+RP12*BTK+IHRB(K))
         DB(4)=H3*((-1.)*BTK+RP21*ATK-IHRA(K))
         IF (PRESENT(KAPPA)) THEN
            DA(4)=H3*(-KAPPA*ATK+RP12*BTK)
            DB(4)=H3*(KAPPA*BTK-(EAU*RA-VR(K))*ATK)
         ENDIF
         IF (PRESENT(DPOT)) THEN
            DA(4)=H3*(1.*ATK+RP12*BTK)
            DB(4)=H3*((-1.)*BTK+RP21*ATK-DV/2*RA*RA/C/C/RP12/RP12*(KAPPA+1)*ATK)
         ENDIF
         A(K)=A(K+1)+1.125_q*DA(4)+2.375_q*DA(3)-0.625_q*DA(2)+0.125_q*DA(1)
         B(K)=B(K+1)+1.125_q*DB(4)+2.375_q*DB(3)-0.625_q*DB(2)+0.125_q*DB(1)
         DA(1)=DA(2)
         DA(2)=DA(3)
         DB(1)=DB(2)
         DB(2)=DB(3)
         DA(3) =H3*(1.*A(K)+RP12*B(K)+IHRB(K))
         DB(3) =H3*((-1.)*B(K)+RP21*A(K)-IHRA(K))      
         IF (PRESENT(KAPPA)) THEN
            DA(3)=H3*(-KAPPA*A(K)+RP12*B(K))
            DB(3)=H3*(KAPPA*B(K)-(EAU*RA-VR(K))*A(K))
         ENDIF
         IF (PRESENT(DPOT)) THEN
            DA(3)=H3*(1.*A(K)+RP12*B(K))
            DB(3)=H3*((-1.)*B(K)+RP21*A(K)-DV/2*RA*RA/C/C/RP12/RP12*(KAPPA+1)*A(K))
         ENDIF
      ENDDO

      RETURN
      END SUBROUTINE INWINT


!******************** FUNCTION IS_BOUND ********************************
!      
!***********************************************************************
      FUNCTION IS_BOUND(PP,CHANNEL)
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER CHANNEL
      LOGICAL IS_BOUND
      IF (CHANNEL<1.OR.CHANNEL>PP%LMAX) THEN
         IS_BOUND=.FALSE.
      ELSE
         IS_BOUND=(PP%QATO(CHANNEL,CHANNEL)/=0._q)
      ENDIF
      END FUNCTION IS_BOUND


!******************** FUNCTION N_OF_CHANNEL ****************************
!      
!***********************************************************************
      FUNCTION N_OF_CHANNEL(PP,CHANNEL)
      USE prec
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER CHANNEL
      INTEGER N_OF_CHANNEL
      ! local variables
      INTEGER NODES,K
      N_OF_CHANNEL=-1
      IF (CHANNEL>=1.AND.CHANNEL<=PP%LMAX) THEN
         NODES=0
         DO K=2,PP%R%NMAX
            NODES=NODES+.501*ABS(SIGN(1._q,PP%WAE(K,CHANNEL))-SIGN(1._q,PP%WAE(K-1,CHANNEL)))
         ENDDO
         N_OF_CHANNEL=NODES+PP%LPS(CHANNEL)+1
      ENDIF
      END FUNCTION N_OF_CHANNEL


!******************** FUNCTION IR_MATCH ********************************
!      
!***********************************************************************
      FUNCTION IR_MATCH(PP,CHANNEL)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER CHANNEL
      INTEGER IR_MATCH
      ! local variables
      INTEGER K
      IR_MATCH=PP%R%NMAX
      DO K=PP%R%NMAX,1,-1
         IF (PP%WAE(K,CHANNEL)/=PP%WPS(K,CHANNEL)) EXIT
         IR_MATCH=K
      ENDDO
      END FUNCTION IR_MATCH


!******************** FUNCTION IR_PSMAX ********************************
!      
!***********************************************************************
      FUNCTION IR_PSMAX(PP)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER IR_PSMAX
      ! local variables
      INTEGER K
      IR_PSMAX=PP%R%NMAX
      DO K=PP%R%NMAX,2,-1
         IF ((PP%R%R(K)<PP%PSDMAX).OR.(PP%R%R(K)<PP%PSRMAX)) EXIT
         IR_PSMAX=K
      ENDDO
      END FUNCTION IR_PSMAX


!******************** FUNCTION IR_END **********************************
!      
!***********************************************************************
      FUNCTION IR_END(PP,CHANNEL)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER CHANNEL
      INTEGER IR_END
      ! local variables
      INTEGER K
      REAL(q), PARAMETER :: TINY=1E-5
      IR_END=PP%R%NMAX
      DO K=PP%R%NMAX,1,-1
         IF (ABS(PP%WAE(K,CHANNEL))>TINY) EXIT
         IR_END=K
      ENDDO
      END FUNCTION IR_END


!******************** FUNCTION IR_CORE *********************************
!      
!***********************************************************************
      FUNCTION IR_CORE(W)
      USE prec
      IMPLICIT NONE
      TYPE(atomic) W
      INTEGER IR_CORE
      ! local variables
      INTEGER I,K
      REAL(q), PARAMETER :: TINY=1E-6
      IR_CORE=0
      DO I=1,W%NSCORE
         DO K=W%R%NMAX,2,-1
            IF (ABS(W%W(K,I))>TINY) EXIT 
         ENDDO
         IR_CORE=MAX(IR_CORE,K)
      ENDDO
      END FUNCTION IR_CORE


!******************** FUNCTION IR_SWITCH *******************************
!      
!***********************************************************************
      FUNCTION IR_SWITCH(W1,W2,ILO)
      USE prec
      IMPLICIT NONE
      REAL(q) W1(:),W2(:)
      INTEGER ILO,IR_SWITCH
      ! local variables
      INTEGER I
      IR_SWITCH=MIN(SIZE(W1),SIZE(W2))
      DO I=MIN(SIZE(W1),SIZE(W2)),ILO,-1
         IF (SIGN(1._q,W1(I-1)-W2(I-1))/=SIGN(1._q,W1(I)-W2(I))) IR_SWITCH=I
      ENDDO
      END FUNCTION IR_SWITCH


!******************** FUNCTION IR_WMAX *******************************
!      
!***********************************************************************
      FUNCTION IR_WMAX(W)
      USE prec
      IMPLICIT NONE
      REAL(q) W(:)
      INTEGER IR_WMAX
      ! local variables
      INTEGER I
      REAL(q) WMAX
      IR_WMAX=1; WMAX=0
      DO I=1,SIZE(W)
         IF (ABS(W(I))>WMAX) THEN
            WMAX=ABS(W(I)); IR_WMAX=I
         ENDIF
      ENDDO
      END FUNCTION IR_WMAX


!******************** SUBROUTINE RCRCUT ********************************
!
!***********************************************************************
      FUNCTION RCRCUT(RHOCUT,ECUT,PP,IO)
      USE base
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      TYPE(in_struct) IO
      REAL(q) RHOCUT,ECUT
      REAL(q) RCRCUT
      ! local variables
      TYPE(atomic) WDF
      INTEGER I
      
      RCRCUT=-1
      
      ! Solve the atomic problem using DFT
      CALL DFATOM(PP,WDF,IO)

      ! W%RHOAE holds the core charge, now add the charge density
      ! of possible semicore states (W%E<ECUT)
      DO I=WDF%NSCORE+1,WDF%NS
         IF (WDF%E(I)<ECUT) THEN
            WDF%RHOAE(:)=WDF%RHOAE(:)+WDF%OCC(I)*(WDF%A(:,I)*WDF%A(:,I)+WDF%B(:,I)*WDF%B(:,I)/C/C)
         ENDIF
      ENDDO
      
      ! Find the radius at which W%RHOAE exceeds the threshold RHOCUT
      DO I=WDF%R%NMAX,1,-1
         IF (WDF%RHOAE(I)>RHOCUT) EXIT
      ENDDO
      
      IF (I/=0.AND.I<=PP%R%NMAX) RCRCUT=PP%R%R(I)
      
      ! Cleanup
      CALL DEALLOCW(WDF)
      
      END FUNCTION RCRCUT


      END MODULE rhfatm
