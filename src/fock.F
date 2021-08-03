!#define dotiming
#include "symbol.inc"

!***********************************************************************
!
! this module implements the exact exchange operator (Hartree Fock)
!
! - until 01.10.03 written by Robin Hirschl
! - 08.10.03 cleaned up gK
! - 11.10.03 forces added by gK
! - 12.10.03 screened exchange added by gK
! - 21.10.03 one center onsite PAW corrections included gK
! - 30.10.03 core valence exchange added by gK
! - 32.02.04 screened exchange updated gK
! - 32.09.04 downsampling on HF grid added gK
! - 32.09.04 k-points downsampling added gK
! - 32.10.04 stress added gK
! - 18.04.05 modification of singularity correction to treat
!            k-points not included in the exchange kernel
!***********************************************************************


MODULE fock
  USE prec
  USE poscar
  USE nonl_high
  USE augfast
  USE wave_high
  USE fock_multipole
  IMPLICIT none

  ! use HF Hamiltonian
  LOGICAL,SAVE :: LHFCALC=.FALSE.

  ! precision for HF type calculations
  ! L  low    (coarse grid for HF, normal augmentation charge)
  ! F  fast   (coarse grid for HF, soft augmentation charge)
  ! N  normal (normal grid for HF, normal augmentation charge)
  ! A  accurate (normal grid for HF, soft augmentation charge)
  CHARACTER (6), SAVE ::    SZPRECFOCK='      '
  REAL(q) :: ENCUTFOCK  ! test version
                        ! apply spherical cutoff for HF part
                        ! recommended default ENCUT for PRECFOCK = Fast

  ! use EXX-optimized effective potential method
  ! usually this implies that the HF Hamiltonian
  ! is not apllied in the calculations of H phi
  INTEGER,SAVE :: EXXOEP=0

  ! this flag forces the use the the Fock Hamiltonian when calculating
  ! H phi
  ! throughout all routines (even if EXXOEP not zero)
  LOGICAL,SAVE :: LHFCALC_FORCE=.FALSE.

  ! LSYMGRAD determines whether symmetry is used to restore
  ! the proper symmetry of the gradient
  LOGICAL,SAVE :: LSYMGRAD=.FALSE.

  ! amount of exact exchange (the remainder is treated in the LDA framework)
  ! amount of exact exchange (the remainder is treated in the LDA framework)
  ! throughout all routines
  REAL(q),SAVE :: AEXX=0

  ! use Hartree Fock only in one center AE terms
  LOGICAL,SAVE :: LHFONE=.FALSE.

  ! screening length in screened exchange methods
  REAL(q),SAVE :: HFSCREEN=0

  ! screening length in screened exchange methods for correlation
  REAL(q),SAVE :: HFSCREENC=0

  ! cutoff for HF kernel applicable to 0 and 3 dimensional systems (molecules)
  REAL(q),SAVE :: HFRCUT=0

  ! smoothed cutoff function applied in correlated calculations
  REAL(q),SAVE :: HFRCUT_SMOOTH=1._q/200._q

  ! parameter for the determination of the convergence corrections
  REAL(q),SAVE :: HFALPHA

  ! this variable allows to reduce the exchange operator to a coarse
  ! grid q which is reduced in the first second and third direction
  ! by the values supplied by the user
  INTEGER, SAVE :: NKREDX=1, NKREDY=1, NKREDZ=1

  ! this variable allows to shift the grid on which the Fock
  ! operator is evaluated
  ! works only in combination with at least one NKRED set to 2
  LOGICAL :: HFKIDENT=.FALSE.

  ! this variable allows to shift the grid on which the Fock
  ! operator is evaluated
  ! works only in combination with at least one NKRED set to 2
  LOGICAL :: SHIFTRED=.FALSE.

  ! this variable allows to use only every second point in the HF
  ! grid, gamma point is included
  ! sort of similar to NKRED, but resulting in a net downsampling by 2
  LOGICAL :: EVENONLY =.FALSE.

  ! similar to the above flag, but it allows to use odd grid points only
  ! that is gamma is not included
  LOGICAL :: ODDONLY =.FALSE.

  ! maximum L quantum number to which the charge density
  ! is augmented on the plane wave grid
  ! this has quite some performance impact, but unfortunately
  ! LMAX_FOCK can hardly be ever decreased without noticeable
  ! loss in precision
  INTEGER, SAVE :: LMAX_FOCK

  ! maximum L quantum number to which the charge density
  ! is accurately augmented on the plane wave grid
  ! VASP usually restores all moments (monopole, dipole, quadrupole etc.)
  ! on the plane wave grid).
  ! In some routines like GW, RPA correlation, the one center terms
  ! are not implemented and one would like to restore the AE charge density 
  ! very accurately on the plane wave grid.
  ! This has very significant performance impact, and since the one 
  ! center terms correct for any difference between PS and AE
  ! wavefunction, use this flag only for GW type calculations
  ! NMAX_FOCKAE determines how many functions are used for each
  ! channel. Only up to two are supported
  INTEGER, SAVE :: LMAX_FOCKAE
  INTEGER, SAVE :: NMAX_FOCKAE
  ! this flag forces the use of LMAXFOCKAE_IN_DFT in every part of the code
  ! including DFT
  LOGICAL, SAVE :: LFOCKAEDFT=.FALSE.

  REAL(q),ALLOCATABLE, SAVE :: QMAX_FOCKAE(:)
    
  ! LRSCOR=.TRUE.  Range separated LDA correlation
  ! LRSCOR=.FALSE. Complete LDA correlation
  LOGICAL       :: LRSCOR      

  ! LRHFCALC long range HF interaction only
  ! the default is short range HF interaction only
  ! the variable should be identical to LUSE_LONGRANGE_HF in xclib
  LOGICAL       :: LRHFCALC
  !
  ! temporarily switch off the HF treatment
  !
  LOGICAL       :: LSTACK_FOCK , LHFCALC_STACK

  ! L_THOMAS_FERMI Thomas Fermi screening in HF exchange
  ! should be identical to LUSE_THOMAS_FERMI in fock.F
  LOGICAL        :: L_THOMAS_FERMI

  ! LUSE_MODEL_HF short range full HF
  ! long range a fraction of the full HF specified by AEXX
  LOGICAL       :: L_MODEL_HF

  ! calculate four orbital integrals
  INTEGER        :: FOURORBIT
  REAL(q), SAVE  :: ENCUT4O

  ! this matrix stores the transformation matrix that
  ! is required to go from the occupancies RHO(lm, l'm') to
  ! the onsite L dependent occupancies RHO(LM)
  ! the matrix differs for each type
#ifndef PGI16
  REAL(q), POINTER, SAVE, CONTIGUOUS :: TRANS_MATRIX_FOCK(:,:,:,:) 
#else
  REAL(q), POINTER, SAVE             :: TRANS_MATRIX_FOCK(:,:,:,:)
#endif
  ! this structures stores the data required to perform 
  ! a fast augmentation of the pseudo charge density within
  ! the HF related routines
  TYPE (nonlr_struct), SAVE, TARGET :: FAST_AUG_FOCK

  ! structure to store the data layout of the one-center charge densities
  ! in the HF routines
  ! this must match the FAST_AUG_FOCK data layout
  TYPE (wavedes1), SAVE :: AUG_DES

  ! this structure stores the wavefunctions descriptor
  ! for the HF related routines
  ! usually it is equivalent to WDES, but it is possible
  ! to use a coarser FFT grid for all HF related routines
  TYPE (wavedes),SAVE, POINTER :: WDES_FOCK

  ! WDES_FOCK differs from WDES
  LOGICAL :: LWDES_FOCK=.FALSE.

  ! this structure stores the related grid structure
  ! for the 3d-FFT of orbitals
  ! usually simply a pointer to GRID
  ! in the parallel cases, this grid might contain 
  ! only the minimal required grid points to reduce computational cost
  ! NOTE: for Gamma only in real space the arrays must be declared as COMPLEX
  ! as for all orbital FFT's in VASP
  TYPE (grid_3d), POINTER :: GRID_FOCK

  ! store the 3d structure to perform a 3d FFT of charges and potentials
  ! within HF related routines
  ! stores all grid points in real and reciprocal space
  ! NOTE: for Gamma only in real space the arrays must be declared as GDEF
  ! see also mgrid.F GEN_GRID_HF for further comments
  TYPE (grid_3d) :: GRIDHF

  ! leading dimensions of CDIJ, or CRHODE
  INTEGER :: LMDIM_FOCK

  ! use model GW
  INTEGER :: MODEL_GW=0

  !
  ! static dielectric constant for GW
  REAL(q) MODEL_GW_EPS0

  !
  ! parameter alpha for model GW
  ! although the Bechstedt et al. Solid State Comm. 84, 765 suggests to use 1.5_q
  ! it seems most Bechstedt people use 1.0
  REAL(q) :: MODEL_GW_ALPHA=1.0_q

  ! FSG_STORE stores the convergence correction
  ! array is set up by the SETUP_FOCK routines
  ! for each k-point one value is stored
  !
  REAL(q), POINTER :: FSG_STORE(:)
  ! 
  ! NABNDSGWLOW_FOCK allows to exclude low energy states from the HF
  ! Hamiltonian
  !
  INTEGER :: NBANDSGWLOW_FOCK
  !
  ! blocking factor in FOCK_ACC and FOCK_FORCE
  !
  INTEGER :: NBLOCK_FOCK=64
  ! 
  ! this handle in principle allows to store all quantities 
  ! required to calculate the accelerations on the wavefunction in HF case
  !
  TYPE fock_handle
     TYPE (wavedes) :: WDES
     INTEGER  :: LMDIM
     COMPLEX(q),POINTER :: CXI(:,:,:)   ! stores acceleration in real space for each band
     GDEF, POINTER :: CKAPPA(:,:,:)     ! stores NL accelerations
     GDEF, POINTER, CONTIGUOUS :: CDLM(:)
     GDEF, POINTER :: CDIJ(:,:,:,:)
     TYPE (wavedes1) :: WDESK           ! 
     TYPE (wavefun1) :: W1              ! really only a dummy W1%CPROJ=> CDLM
     TYPE (wavefun1) :: WQ              ! wavefunction 
  END TYPE fock_handle

CONTAINS
!**********************************************************************
!
! read all variables related to exchange correlation treatment
! from the INCAR file
! this set both the variables in the setex module and
! in the local fock module
!
!**********************************************************************

  SUBROUTINE XC_FOCK_READER(IU5, IU0, IU6, SZPREC, ISIF, ISYM, IALGO, & 
      OMEGA, NTYP, NIONS, IMIX, AMIX, AMIX_MAG, BMIX, BMIX_MAG, LVTOT)
   
      USE vaspxml
      USE base
      USE setexm
      USE full_kpoints
      USE vdw_df_helper, ONLY : SET_vdW_IO_UNITS
      IMPLICIT NONE

      INTEGER IU5, IU0, IU6
      CHARACTER(LEN=6) :: SZPREC
      INTEGER ISIF            ! calculate stress tensor or not (defaults to 0 for HF)
      INTEGER ISYM            ! symmetry
      INTEGER IALGO           ! applied algorithm
      REAL(q) :: OMEGA
      INTEGER :: NTYP, NIONS
      INTEGER :: IMIX
      REAL(q) :: AMIX, AMIX_MAG, BMIX, BMIX_MAG
      LOGICAL :: LVTOT
  ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      LOGICAL, EXTERNAL :: CALCULATE_RESPONSE_FUNCTIONS, USE_OEP_IN_GW, LDMATRIX, FAST_FOCK, CALCULATE_RPA_FORCES
      CHARACTER (1) :: CHARAC
      CHARACTER (40)   SZNAM

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      AEXX=0.0
      AGGAX=1.0
      AGGAC=1.0
      LDAX =1.0
      ALDAC=1.0
      HFSCREEN=0.0
      HFSCREENC=0.0
      HFRCUT=0.0
      PARAM1=0.1234000000_q
      PARAM2=1.00000000_q
      PARAM3=0._q
      LUSE_VDW=.FALSE.
      Zab_VDW=-0.8491_q
! rVV10
      BPARAM = 6.3_q 
      CPARAM = 0.0093_q
      
      LRSCOR=.FALSE.
      LHFCALC=CALCULATE_RESPONSE_FUNCTIONS().OR.LDMATRIX()
      LRHFCALC=.FALSE.
      L_MODEL_HF=.FALSE.
      L_THOMAS_FERMI=.FALSE.

! HF calculations 
      CALL RDATAB(LOPEN,INCAR,IU5,'LHFCALC','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LHFCALC,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LHFCALC'' from file INCAR.'
         LHFCALC=.FALSE.
      ELSE IF (LHFCALC .AND. IERR==0 .AND. N>=1) THEN
         ! if LHFCALC was READ (and only then) default AEXX to 0.25
         AEXX=0.25
      ENDIF
      CALL XML_INCAR('LHFCALC','L',IDUM,RDUM,CDUM,LHFCALC,CHARAC,N)

! EXXOEP default is from chi.F
      EXXOEP=0
      CALL RDATAB(LOPEN,INCAR,IU5,'EXXOEP','=','#',';','I', &
     &            EXXOEP,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''EXXOEP'' from file INCAR.'
         EXXOEP=0
      ENDIF
      EXXOEP=MAX(0,MIN(EXXOEP,4))
      CALL XML_INCAR('EXXOEP','I',EXXOEP,RDUM,CDUM,LDUM,CHARAC,N)

! LSYMGRAD apply symmetry to restore symmetry of the gradient
      CALL RDATAB(LOPEN,INCAR,IU5,'LSYMGRAD','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LSYMGRAD,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LSYMGRAD'' from file INCAR.'
         LSYMGRAD=.FALSE.
      ENDIF
      CALL XML_INCAR('LSYMGRAD','L',IDUM,RDUM,CDUM,LSYMGRAD,CHARAC,N)

! HF calculations one center terms
      CALL RDATAB(LOPEN,INCAR,IU5,'LHFONE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LHFONE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LHFONE'' from file INCAR.'
         LHFONE=.FALSE.
      ENDIF
      CALL XML_INCAR('LHFONE','L',IDUM,RDUM,CDUM,LHFONE,CHARAC,N)

! force LHFCALC in some cases
      IF (EXXOEP>0 .OR. CALCULATE_RESPONSE_FUNCTIONS()) THEN
         LHFCALC=.TRUE.
      ENDIF
! long range correlation
      CALL RDATAB(LOPEN,INCAR,IU5,'LRSCOR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRSCOR,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRSCOR'' from file INCAR.'
         LRSCOR=.FALSE.
      ENDIF
      CALL XML_INCAR('LRSCOR','L',IDUM,RDUM,CDUM,LRHFCALC,CHARAC,N)

! long range HF exclusively
      CALL RDATAB(LOPEN,INCAR,IU5,'LRHFCALC','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRHFCALC,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRHFCALC'' from file INCAR.'
         LRHFCALC=.FALSE.
      ENDIF
      CALL XML_INCAR('LRHFCALC','L',IDUM,RDUM,CDUM,LRHFCALC,CHARAC,N)
! model HF

      CALL RDATAB(LOPEN,INCAR,IU5,'LMODELHF','=','#',';','L', &
     &            IDUM,RDUM,CDUM,L_MODEL_HF,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LMODELHF'' from file INCAR.'
         L_MODEL_HF=.FALSE.
      ENDIF

      CALL XML_INCAR('LMODELHF','L',IDUM,RDUM,CDUM,L_MODEL_HF,CHARAC,N)
      IF (L_MODEL_HF) THEN
         LRHFCALC=.FALSE.
      END IF


      CALL RDATAB(LOPEN,INCAR,IU5,'LTHOMAS','=','#',';','L', &
     &            IDUM,RDUM,CDUM,L_THOMAS_FERMI,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LTHOMAS'' from file INCAR.'
         LHFCALC=.FALSE.
      ENDIF

      CALL XML_INCAR('LTHOMAS','L',IDUM,RDUM,CDUM,L_THOMAS_FERMI,CHARAC,N)

      FOURORBIT=0
      CALL RDATAB(LOPEN,INCAR,IU5,'FOURORBIT','=','#',';','I', &
     &            FOURORBIT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''FOURORBIT'' from file INCAR.'
         FOURORBIT=0
      ENDIF
      FOURORBIT=MAX(0,MIN(FOURORBIT,2))
      CALL XML_INCAR('FOURORBIT','I',FOURORBIT,RDUM,CDUM,LDUM,CHARAC,N)

      ENCUT4O=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'ENCUT4O','=','#',';','F', &
     &            IDUM,ENCUT4O,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ENCUT4O'' from file INCAR.'
         ENCUT4O=-1
      ENDIF
      CALL XML_INCAR('ENCUT4O','F',IDUM,ENCUT4O,CDUM,LDUM,CHARAC,N)

! If range separation, THOMAS fermi screening or four orbital terms are demanded 
! the calculation of the Fock exchange should be switched on as well
      IF (LRSCOR .OR. LRHFCALC .OR. L_MODEL_HF .OR. L_THOMAS_FERMI .OR. FOURORBIT>0) THEN
         LHFCALC=.TRUE.
      ENDIF

      IF (LHFCALC .AND. LHFONE) THEN
         CALL VTUTOR('E','LHFONE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU6,3)
         CALL VTUTOR('S','LHFONE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
         STOP
      ENDIF

!
! read in the GGA tag (enforces GGA, even if GGA was not used for PP creation)
!
      SZNAM='--'
      CALL RDATAB(LOPEN,INCAR,IU5,'GGA','=','#',';','S', &
     &            IDUM,RDUM,CDUM,LDUM,SZNAM,N,40,IERR)
      IF ((IERR/=0).AND.(IERR/=3)) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''GGA'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('GGA','S',IDUM,RDUM,CDUM,LDUM,SZNAM,N)

      CALL STRIP(SZNAM,N,'L')
      CALL UPPER(SZNAM)
      SZGGA=SZNAM(1:2)

      CALL EXTYP(SZGGA,LEXCH)
! If we want a range separation in the LDA correlation then
! we have to switch to the Vosko-Wilk-Nusair treatment of
! LDA correlation
      IF (LRSCOR) CALL EXTYP('VW',LEXCH)

! spin interpolation according to Vosko Wilk and Nusair
      LFCI=0
      CALL RDATAB(LOPEN,INCAR,IU5,'VOSKOWN','=','#',';','I', &
     &            LFCI,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''VOSKOWN'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('VOSKOWN','I',LFCI,RDUM,CDUM,LDUM,CHARAC,N)

!parameters for x-functionals
      CALL RDATAB(LOPEN,INCAR,IU5,'PARAM1','=','#',';','F', &
     &            IDUM,PARAM1,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''PARAM1'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('PARAM1','F',IDUM,PARAM1,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'PARAM2','=','#',';','F', &
     &            IDUM,PARAM2,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''PARAM2'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('PARAM2','F',IDUM,PARAM2,CDUM,LDUM,CHARAC,N)


      CALL RDATAB(LOPEN,INCAR,IU5,'PARAM3','=','#',';','F', &
     &            IDUM,PARAM3,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''PARAM3'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('PARAM3','F',IDUM,PARAM3,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'LUSE_VDW','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LUSE_VDW,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LUSE_VDW'' from file INCAR.'
         LUSE_VDW=.FALSE.
      ENDIF
      CALL XML_INCAR('LUSE_VDW','L',IDUM,RDUM,CDUM,LUSE_VDW,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'Zab_VDW','=','#',';','F', &
     &            IDUM,Zab_VDW,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''Zab_VDW'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('Zab_VDW','F',IDUM,Zab_VDW,CDUM,LDUM,CHARAC,N)

!
! read rVV10 parameters
!
      CALL RDATAB(LOPEN,INCAR,IU5,'BPARAM','=','#',';','F', &
     &            IDUM,BPARAM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''BPARAM'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('BPARAM','F',IDUM,BPARAM,CDUM,LDUM,CHARAC,N) 

      CALL RDATAB(LOPEN,INCAR,IU5,'CPARAM','=','#',';','F', &
     &            IDUM,CPARAM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''CPARAM'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('CPARAM','F',IDUM,CPARAM,CDUM,LDUM,CHARAC,N) 

!
! read in MODEL_GW flag
!
      CALL RDATAB(LOPEN,INCAR,IU5,'MODEL_GW','=','#',';','I', &
           &            MODEL_GW,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''MODEL_GW'' from file INCAR.'
         MODEL_GW=0
      ENDIF
      CALL XML_INCAR('MODEL_GW','I',MODEL_GW,RDUM,CDUM,LDUM,CHARAC,N)
      IF (MODEL_GW>0) THEN
         LHFCALC=.TRUE.
         ! default the screening length to typical values for semiconductors
         HFSCREEN=2.0
         ! switch local exchange and correlation to Coulomb hole
         CALL SET_LEXCH(100)
      ENDIF
!
! read model screening constants
!
      MODEL_GW_EPS0=0.6_q*OMEGA/NIONS
      CALL RDATAB(LOPEN,INCAR,IU5,'MODEL_EPS0','=','#',';','F', &
           &            IDUM,MODEL_GW_EPS0,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                  ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''MODEL_EPS0'' from file INCAR.'
         MODEL_GW_EPS0=0.6_q*OMEGA/NIONS
         IF (IU0>=0) &
              WRITE(IU0,*)'In order to save things I use the default',MODEL_GW_EPS0
      ENDIF
    
      CALL XML_INCAR('MODEL_EPS0','F',IDUM,MODEL_GW_EPS0,CDUM,LDUM,CHARAC,N)


      MODEL_GW_ALPHA=1._q
      CALL RDATAB(LOPEN,INCAR,IU5,'MODEL_ALPHA','=','#',';','F', &
           &            IDUM,MODEL_GW_ALPHA,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                  ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''MODEL_ALPHA'' from file INCAR.'
         WRITE(IU0,*)'In order to save things I use the default (1.)'
         MODEL_GW_ALPHA=1._q
      ENDIF

      CALL XML_INCAR('MODEL_ALPHA','F',IDUM,MODEL_GW_ALPHA,CDUM,LDUM,CHARAC,N)
      !
      ! NKRED reduce k-points in Fock exchange
      !
      NKREDX=1
      NKREDY=1
      NKREDZ=1
      CALL RDATAB(LOPEN,INCAR,IU5,'NKRED','=','#',';','I', &
           &            NKREDX,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NKRED'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('NKRED','I',NKREDX,RDUM,CDUM,LDUM,CHARAC,N)

      IF (N>=1) THEN
         NKREDY=NKREDX
         NKREDZ=NKREDX
      ENDIF

      CALL RDATAB(LOPEN,INCAR,IU5,'NKREDX','=','#',';','I', &
           &            NKREDX,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NKREDX'' from file INCAR.'
      ENDIF

      CALL XML_INCAR('NKREDX','I',NKREDX,RDUM,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'NKREDY','=','#',';','I', &
           &            NKREDY,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NKREDY'' from file INCAR.'
      ENDIF

      CALL XML_INCAR('NKREDY','I',NKREDY,RDUM,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'NKREDZ','=','#',';','I', &
           &            NKREDZ,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NKREDZ'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('NKREDZ','I',NKREDZ,RDUM,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'SHIFTRED','=','#',';','L', &
           &        IDUM,RDUM,CDUM,SHIFTRED,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &        ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''SHIFTRED'' from file INCAR.'
         SHIFTRED=.FALSE.
      ENDIF
      CALL XML_INCAR('SHIFTRED','L',IDUM,RDUM,CDUM,SHIFTRED,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'HFKIDENT','=','#',';','L', &
           &        IDUM,RDUM,CDUM,HFKIDENT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &        ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''HFKIDENT'' from file INCAR.'
         SHIFTRED=.FALSE.
      ENDIF
      CALL XML_INCAR('HFKIDENT','L',IDUM,RDUM,CDUM,HFKIDENT,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'ODDONLY','=','#',';','L', &
           &        IDUM,RDUM,CDUM,ODDONLY,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &        ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''ODDONLY'' from file INCAR.'
         ODDONLY=.FALSE.
      ENDIF
      CALL XML_INCAR('ODDONLY','L',IDUM,RDUM,CDUM,ODDONLY,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'EVENONLY','=','#',';','L', &
           &        IDUM,RDUM,CDUM,EVENONLY,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &        ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''EVENONLY'' from file INCAR.'
         EVENONLY=.FALSE.
      ENDIF
      CALL XML_INCAR('EVENONLY','L',IDUM,RDUM,CDUM,EVENONLY,CHARAC,N)

      IF (LHFCALC) THEN
!
! read lowest band to be included in HF term
!
         NBANDSGWLOW_FOCK=1
     
         CALL RDATAB(LOPEN,INCAR,IU5,'NBANDSGWLOW','=','#',';','I', &
              &            NBANDSGWLOW_FOCK,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''NBANDSGWLOW'' from file INCAR.'
            NBANDSGWLOW_FOCK=1
         ENDIF
         CALL XML_INCAR('NBANDSGWLOW','I',NBANDSGWLOW_FOCK,RDUM,CDUM,LDUM,CHARAC,N)
         NBANDSGWLOW_FOCK=MAX(NBANDSGWLOW_FOCK,1)
!
! reset ISIF to 0, and reread it from file
!
         ISIF=0
         CALL RDATAB(LOPEN,INCAR,IU5,'ISIF','=','#',';','I', &
              &            ISIF,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
! re-read symmetry tag (default is ISYM=3) for HF-type calculations
         IF (EXXOEP==0 .AND. .NOT. USE_OEP_IN_GW()) ISYM=3
         CALL RDATAB(LOPEN,INCAR,IU5,'ISYM','=','#',';','I', &
                         ISYM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (EXXOEP==0 .AND. .NOT. USE_OEP_IN_GW() .AND. ISYM == 2) THEN
            CALL VTUTOR('W','ISYM2',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
            CALL VTUTOR('W','ISYM2',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU6,3)
         ENDIF

! re-read BMIX tag (default is BMIX=0.001 for OEP methods)

         IF (EXXOEP/=0 .OR. USE_OEP_IN_GW()) THEN
            AMIX=0.4
            CALL RDATAB(LOPEN,INCAR,IU5,'AMIX','=','#',';','F', &
     &           IDUM,AMIX,CDUM,LDUM,CHARAC,N,1,IERR)
            AMIX_MAG=AMIX
            CALL RDATAB(LOPEN,INCAR,IU5,'AMIX_MAG','=','#',';','F', &
     &           IDUM,AMIX_MAG,CDUM,LDUM,CHARAC,N,1,IERR)
            BMIX=0.001_q
            CALL RDATAB(LOPEN,INCAR,IU5,'BMIX','=','#',';','F', &
     &           IDUM,BMIX,CDUM,LDUM,CHARAC,N,1,IERR)
            BMIX_MAG=BMIX
            CALL RDATAB(LOPEN,INCAR,IU5,'BMIX_MAG','=','#',';','F', &
     &            IDUM,BMIX_MAG,CDUM,LDUM,CHARAC,N,1,IERR)
            LVTOT=.TRUE.
            CALL RDATAB(LOPEN,INCAR,IU5,'LVTOT','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LVTOT,CHARAC,N,1,IERR)
         ELSE
! no Broyden mixer for HF, tends to blow up, simple Kerker mixer as default
! hence reread IMIX with different default
            IF ((IALGO>=60 .OR. IALGO<50) .AND. .NOT. CALCULATE_RPA_FORCES())  THEN
               IMIX=1
               CALL RDATAB(LOPEN,INCAR,IU5,'IMIX','=','#',';','I', &
     &              IMIX,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
            ENDIF
         ENDIF
!
! HFLMAX maximum L quantum number for HF (LMAXFOCK is read alternatively)
!
         LMAX_FOCK=4
         CALL RDATAB(LOPEN,INCAR,IU5,'HFLMAX','=','#',';','I', &
              &            LMAX_FOCK,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''HFLMAX'' from file INCAR.'
         ENDIF
         CALL RDATAB(LOPEN,INCAR,IU5,'LMAXFOCK','=','#',';','I', &
              &            LMAX_FOCK,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''LMAXFOCK'' from file INCAR.'
         ENDIF

         CALL XML_INCAR('LMAXFOCK','I',LMAX_FOCK,RDUM,CDUM,LDUM,CHARAC,N)

         ! default for LMAX_FOCKAE is set in chi.F (RESPONSE_READER)
         CALL RDATAB(LOPEN,INCAR,IU5,'LMAXFOCKAE','=','#',';','I', &
              &            LMAX_FOCKAE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''LMAXFOCKAE'' from file INCAR.'
         ENDIF

         ! values below -1 are not sensible (-1 no AE augmentation)
         LMAX_FOCKAE=MAX(-1,LMAX_FOCKAE)
         CALL XML_INCAR('LMAXFOCKAE','I',LMAX_FOCKAE,RDUM,CDUM,LDUM,CHARAC,N)

         ! default for QMAX_FOCKAE is set in chi.F (RESPONSE_READER)
         ALLOCATE(QMAX_FOCKAE(NTYP))
         QMAX_FOCKAE=0
         CALL RDATAB(LOPEN,INCAR,IU5,'QMAXFOCKAE','=','#',';','F', &
              &            IDUM,QMAX_FOCKAE,CDUM,LDUM,CHARAC,N,NTYP,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<NTYP))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''QMAXFOCKAE'' from file INCAR.'
         ENDIF
         CALL XML_INCAR_V('QMAXFOCKAE','F',IDUM,QMAX_FOCKAE,CDUM,LDUM,CHARAC,N)

         NMAX_FOCKAE=1
         CALL RDATAB(LOPEN,INCAR,IU5,'NMAXFOCKAE','=','#',';','I', &
              &            NMAX_FOCKAE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''NMAXFOCKAE'' from file INCAR.'
         ENDIF
         NMAX_FOCKAE=MIN(2,MAX(1,NMAX_FOCKAE))

         IF (NMAX_FOCKAE>=2) THEN
            CALL VTUTOR('W','NMAXFOCKAE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU6,3)
            CALL VTUTOR('W','NMAXFOCKAE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
         ENDIF

         CALL XML_INCAR('NMAXFOCKAE','I',NMAX_FOCKAE,RDUM,CDUM,LDUM,CHARAC,N)

         IF ((EXXOEP/=0 .OR. USE_OEP_IN_GW()).AND. LMAX_FOCKAE>=0) THEN
            LFOCKAEDFT=.TRUE.
         ELSE
            LFOCKAEDFT=.FALSE.
         ENDIF

         CALL RDATAB(LOPEN,INCAR,IU5,'LFOCKAEDFT','=','#',';','L', &
     &        IDUM,RDUM,CDUM,LFOCKAEDFT,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &        ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LFOCKAEDFT'' from file INCAR.'
            LFOCKAEDFT=.FALSE.
         ENDIF
         CALL XML_INCAR('LFOCKAEDFT','L',IDUM,RDUM,CDUM,LFOCKAEDFT,CHARAC,N)

!
! read in HFALPHA from INCAR file, default value -1 (automatic determination)
!
         HFALPHA=-1

         CALL RDATAB(LOPEN,INCAR,IU5,'HFALPHA','=','#',';','F', &
              IDUM,HFALPHA,CDUM,LDUM,CHARAC,IDUM,10,IERR)
         IF ((IERR/=0) .AND. (IERR/=3)) THEN
            WRITE(*,*)'Error reading item ''HFALPHA'' from INCAR.'
         ENDIF
         CALL XML_INCAR('HFALPHA','F',IDUM,HFALPHA,CDUM,LDUM,CHARAC,N)
!
! read in MCALPHA from INCAR file, default value -1 (automatic determination)
!
         MCALPHA=0
         CALL RDATAB(LOPEN,INCAR,IU5,'MCALPHA','=','#',';','F', &
              IDUM,MCALPHA,CDUM,LDUM,CHARAC,IDUM,10,IERR)
         IF ((IERR/=0) .AND. (IERR/=3)) THEN
            WRITE(*,*)'Error reading item ''MCALPHA'' from INCAR.'
         ENDIF
         CALL XML_INCAR('MCALPHA','F',IDUM,MCALPHA,CDUM,LDUM,CHARAC,N)
!
!
! read in HFSCREEN, screening length in Thomas-Fermi and Scuserias 
! screened exchange, as well as in the model GW
! Thomas-Fermi and model GW: q_TF
! Scuserias default: 0.6 A-1
!
         CALL RDATAB(LOPEN,INCAR,IU5,'HFSCREEN','=','#',';','F', &
              &            IDUM,HFSCREEN,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''HFSCREEN'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('HFSCREEN','F',IDUM,HFSCREEN,CDUM,LDUM,CHARAC,N)

!
! screening length for correlation
!
         HFSCREENC=HFSCREEN
         CALL RDATAB(LOPEN,INCAR,IU5,'HFSCREENC','=','#',';','F', &
              &            IDUM,HFSCREENC,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''HFSCREENC'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('HFSCREENC','F',IDUM,HFSCREENC,CDUM,LDUM,CHARAC,N)

!
! read in HFRCUT
! truncate the potential kernel at a finite radius R
!
         CALL RDATAB(LOPEN,INCAR,IU5,'HFRCUT','=','#',';','F', &
              &            IDUM,HFRCUT,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''HFRCUT'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('HFRCUT','F',IDUM,HFRCUT,CDUM,LDUM,CHARAC,N)
!
! read in AEXX amount of exact exchange and AGGAX (default 1-AEXX)
!
         ! PBE0 parameters
         ! C. Adamo, V. Barone, J. Chem. Phys. 110, 6158 (1999)
         ! M. Ernzerhof, G. E. Scuseria, J. Chem. Phys. 110, 5029 (1999)
         ! B3LYP parameters  AEXX=0.2 ; AGGAX=0.72 ; AGGAC=0.81

         IF (LRHFCALC .OR. L_THOMAS_FERMI .OR. MODEL_GW>0 .OR. EXXOEP/=0 .OR. USE_OEP_IN_GW()) THEN
            AEXX=1.0
         ENDIF
         CALL RDATAB(LOPEN,INCAR,IU5,'AEXX','=','#',';','F', &
              &            IDUM,AEXX,CDUM,LDUM,CHARAC,N,1,IERR)

         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''AEXX'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('AEXX','F',IDUM,AEXX,CDUM,LDUM,CHARAC,N)

         AGGAX=1-AEXX
         LDAX =1-AEXX

         ! for HF type calculations and full exchange, no correlation is usually included
         IF (AEXX==1.0) THEN
            ALDAC=0.0
            AGGAC=0.0
         ENDIF

         CALL RDATAB(LOPEN,INCAR,IU5,'AGGAX','=','#',';','F', &
              &            IDUM,AGGAX,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''AGGAX'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('AGGAX','F',IDUM,AGGAX,CDUM,LDUM,CHARAC,N)

         CALL RDATAB(LOPEN,INCAR,IU5,'ALDAX','=','#',';','F', &
              &            IDUM,LDAX,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''ALDAX'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('ALDAX','F',IDUM,LDAX,CDUM,LDUM,CHARAC,N)

!
! read in ENCUTFOCK cutoff for Fock exchange
! terminate if user selects it
!
         ENCUTFOCK=-1
         CALL RDATAB(LOPEN,INCAR,IU5,'ENCUTFOCK','=','#',';','F', &
              &            IDUM,ENCUTFOCK,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''ENCUTFOCK'' from file INCAR.'
         ENDIF
!         IF (N>0) THEN
!            CALL VTUTOR('E','ENCUTFOCK',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!            IU6,3)
!            CALL VTUTOR('S','ENCUTFOCK',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!            IU0,3)
!            STOP
!         ENDIF

! HF calculations one center terms
         SZPRECFOCK='      '
         IF (FAST_FOCK()) SZPRECFOCK='fast'
         CALL RDATAB(LOPEN,INCAR,IU5,'PRECFOCK','=','#',';','S', &
     &        IDUM,RDUM,CDUM,LDUM,SZPRECFOCK,N,40,IERR)
         IF ((IERR/=0).AND.(IERR/=3)) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''PRECFOCK'' from file INCAR.'
            SZPRECFOCK='normal'
         ENDIF
         CALL STRIP(SZPRECFOCK,N,'L')
         CALL LOWER(SZPRECFOCK)
         IF (SZPRECFOCK(1:1)=='l') THEN
            SZPRECFOCK='low   '
         ELSE IF (SZPRECFOCK(1:1)=='m') THEN
            SZPRECFOCK='medium'
         ELSE IF (SZPRECFOCK(1:1)=='f') THEN
            SZPRECFOCK='fast'
         ELSE IF (SZPRECFOCK(1:1)=='n') THEN
            SZPRECFOCK='normal'
         ELSE IF (SZPRECFOCK(1:1)=='a') THEN
            SZPRECFOCK='accur.'
         ELSE IF (SZPRECFOCK(1:1)=='s') THEN
            SZPRECFOCK='soft  '
         ELSE
            SZPRECFOCK='normal'
!            CALL VTUTOR('W','PRECFOCK',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!            IU6,3)
!            CALL VTUTOR('W','PRECFOCK',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!            IU0,3)
         ENDIF
         CALL XML_INCAR('PRECFOCK','S',IDUM,RDUM,CDUM,LDUM,SZPRECFOCK,N)
         
         
      ELSE IF (LHFONE) THEN
!
! read in AEXX amount of exact exchange and AGGAX (default 1-AEXX)
! one center PBE0 parameter
!
         AEXX=0.25

         CALL RDATAB(LOPEN,INCAR,IU5,'AEXX','=','#',';','F', &
              &            IDUM,AEXX,CDUM,LDUM,CHARAC,N,1,IERR)

         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''AEXX'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('AEXX','F',IDUM,AEXX,CDUM,LDUM,CHARAC,N)

      ENDIF
!
! amount of GGA correlation
!
      CALL RDATAB(LOPEN,INCAR,IU5,'AGGAC','=','#',';','F', &
     &            IDUM,AGGAC,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''AGGAC'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('AGGAC','F',IDUM,AGGAC,CDUM,LDUM,CHARAC,N)
!
! amount of LDA correlation
!
      CALL RDATAB(LOPEN,INCAR,IU5,'ALDAC','=','#',';','F', &
     &            IDUM,ALDAC,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ALDAC'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('ALDAC','F',IDUM,ALDAC,CDUM,LDUM,CHARAC,N)
!
! blocking factor in FOCK_ACC and FOCK_FORCE
!
      NBLOCK_FOCK=64
      CALL RDATAB(LOPEN,INCAR,IU5,'NBLOCK_FOCK','=','#',';','I', &
     &            NBLOCK_FOCK,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''NBLOCK_FOCK'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('NBLOCK_FOCK','I',NBLOCK_FOCK,RDUM,CDUM,LDUM,CHARAC,N)

      CLOSE(IU5)

      IF (LUSE_VDW) THEN
         CALL VTUTOR('I','vdW-Klimes',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IU6,3)
         CALL VTUTOR('I','vdW-Klimes',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IU0,3)
         CALL SET_vdW_IO_UNITS(IU0, IU6, ISIF)
      ENDIF
!
! set the corresponding variables in the LDA module (xclib.F)
! 
      LDASCREEN =HFSCREEN
      LDASCREENC=HFSCREENC
      LRANGE_SEPARATED_CORR=LRSCOR
      LUSE_LONGRANGE_HF=LRHFCALC
      LUSE_THOMAS_FERMI=L_THOMAS_FERMI
      LUSE_MODEL_HF=L_MODEL_HF

      IF (NKREDX==1 .AND. NKREDY==1 .AND. NKREDZ==1 .AND. .NOT. EVENONLY .AND. .NOT. ODDONLY ) & 
        HFKIDENT=.FALSE.

      IF (LHFCALC .AND. HFRCUT/=0 .AND. HFKIDENT) THEN
         CALL VTUTOR('S','HFPARAM',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
         STOP
      ENDIF

      IF (LHFCALC) THEN
         CALL USE_FULL_KPOINTS
      ENDIF

      ! OEP methods do not work for low, fast or soft (double grid techniques)
      IF (EXXOEP>0) THEN
         IF ( SZPRECFOCK(1:1) /= SZPREC(1:1) .AND. SZPRECFOCK(1:1) /= 'm') THEN
            CALL VTUTOR('W','OEPdouble',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
              IU0,3)
            SZPRECFOCK='normal'
         ENDIF
      ENDIF


    END SUBROUTINE XC_FOCK_READER


!**********************************************************************
!
! write the Fock and exchange correlation related 
! parameters to the OUTCAR file
!
!**********************************************************************

  SUBROUTINE WRITE_FOCK(IU6)
      USE setexm
      IMPLICIT NONE

      INTEGER IU6
      REAL(q), EXTERNAL :: ENCUTGW_IN_CHI

      REAL(q) :: ENCUTGW

      ENCUTGW=ENCUTGW_IN_CHI()

      IF (LHFCALC) THEN
         IF (LMAX_FOCKAE==0) LFOCKAEDFT=.FALSE.
      ELSE
         LFOCKAEDFT=.FALSE.
      ENDIF

      IF (IU6>=0 .AND. LHFCALC) THEN
         WRITE(IU6,10 ) SZGGA,LEXCH,LFCI,EXXOEP,LHFCALC,LSYMGRAD,SZPRECFOCK,LRHFCALC,LRSCOR,L_THOMAS_FERMI,L_MODEL_HF,ENCUT4O, &
           LMAX_FOCK,LMAX_FOCKAE,NMAX_FOCKAE,LFOCKAEDFT,NKREDX,NKREDY,NKREDZ,SHIFTRED,HFKIDENT,ODDONLY,EVENONLY,HFALPHA,MCALPHA,AEXX,HFSCREEN,HFSCREENC,HFRCUT, &
           LDAX,AGGAX,ALDAC,AGGAC,ENCUTFOCK,NBANDSGWLOW_FOCK,NBLOCK_FOCK
      ENDIF
      IF (IU6>=0 .AND. .NOT.LHFCALC) THEN
         WRITE(IU6,100) SZGGA,LEXCH,LFCI,LHFCALC,LHFONE,AEXX
      ENDIF
      IF (IU6>=0 .AND. MODEL_GW>0) THEN
         WRITE(IU6,20) MODEL_GW,MODEL_GW_EPS0, MODEL_GW_ALPHA
20    FORMAT(' Model GW treatment:'  / &
           '   MODEL_GW   =',I6,  '    type of model GW         '/ & 
           '   MODEL_EPSO =',F6.2,  '    static dielectric constant'/&
           '   MODEL_ALPHA=',F6.2,  '    alpha'/)
      ENDIF

      IF (IU6>=0 .AND. LUSE_VDW) THEN
         WRITE(IU6,40) LUSE_VDW,Zab_VDW,PARAM1,PARAM2,PARAM3,BPARAM,CPARAM
40    FORMAT(' vdW DFT:'  / &
           '   LUSE_VDW   =',L6,  '    switch on vdW DFT'/ & 
           '   Zab_VDW    =',F6.4,  '    correlation parameter'/&
           '   PARAM1     =',F6.4,  /&
           '   PARAM2     =',F6.4,  /&
           '   PARAM3     =',F6.4,  /&
           '   -- rVV10 --',           /&
           '   BPARAM     =',F7.4,  /&
           '   CPARAM     =',F7.4,  /)
      ENDIF


      IF (ENCUTGW>=0 .AND. IU6>=0 .AND. EXXOEP /= 0) THEN
         WRITE(IU6,30) ENCUTGW
30    FORMAT(' Cutoff for OEP:'  / &
           '   ENCUTGW    =',F6.1,  '    cutoff applied to OEP potential'/)
      ENDIF
      
  10  FORMAT(' Exchange correlation treatment:'  / &
             '   GGA     =',A6,  '    GGA type'/ & 
             '   LEXCH   =',I6,  '    internal setting for exchange type'/ & 
             '   VOSKOWN= ',I6,  '    Vosko Wilk Nusair interpolation'/&
             '   EXXOEP  =',I6,  '    0=HF, 1=EXX-LHF (local Hartree Fock) 2=EXX OEP'/ &
             '   LHFCALC =',L6,  '    Hartree Fock is set to'/ &
             '   LSYMGRAD=',L6,  '    symmetrize gradient (conserves proper symmetry)'/ &
             '   PRECFOCK=',A6,  '    Normal, Fast or Accurate (Low or Medium for compatibility)'/ &
             '   LRHFCALC=',L6,  '    long range Hartree Fock'/ &
             '   LRSCOR  =',L6,  '    long range correlation only (use DFT for short range part)'/ &
             '   LTHOMAS =',L6,  '    Thomas Fermi screening in HF'/ &
             '   LMODELHF=',L6,  '    short range full HF, long range fraction AEXX'/ &
             '   ENCUT4O =',F6.1,'   cutoff for four orbital integrals eV'/ &
             '   LMAXFOCK=',I6  ,'    L truncation for augmentation on plane wave grid'  / &
             '   LMAXFOCKAE=',I4,'    L truncation for all-electron charge restoration on plane wave grid'  / &
             '   NMAXFOCKAE=',I4,'    number of basis functions for all-electron charge restoration' / &
             '   LFOCKAEDFT=',L6,  '  apply the AE augmentation even for DFT' /&
             '   NKREDX  =',I6  ,'    reduce k-point grid by'    / &
             '   NKREDY  =',I6  ,'    reduce k-point grid by'    / &
             '   NKREDZ  =',I6  ,'    reduce k-point grid by'    / &
             '   SHIFTRED=',L6  ,'    shift reduced grid of Gamma'    / &
             '   HFKIDENT=',L6  ,'    idential grid for each k-point'    / &
             '   ODDONLY =',L6  ,'    use only odd q-grid points'    / &
             '   EVENONLY=',L6  ,'    use only even q-grid points'    / &
             '   HFALPHA =',F10.4,' decay constant for conv. correction'/ &
             '   MCALPHA =',F10.4,' extent of test-charge in conv. correction in multipole expansion'/ &
             '   AEXX    =',F10.4,' exact exchange contribution' / &
             '   HFSCREEN=',F10.4,' screening length (either q_TF or 0.3 A-1)'/ &
             '   HFSCREENC=',F9.4,' screening length for correlation (either q_TF or 0.3 A-1)'/ &
             '   HFRCUT  =',F10.4,' spherical cutoff for potential kernel'/ &
             '   ALDAX   =',F10.4,' LDA exchange part'           / &
             '   AGGAX   =',F10.4,' GGA exchange part'           / &
             '   ALDAC   =',F10.4,' LDA correlation'             / &
             '   AGGAC   =',F10.4,' GGA correlation'             / &
             '   ENCUTFOCK=',F6.1,' apply spherical cutoff to Coloumb kernel'/ &
             '   NBANDSGWLOW=',I6  ,'    first orbital included in HF term'/ &
             '   NBLOCK_FOCK=',I6  ,'    blocking factor in FOCK_ACC')


  100 FORMAT(' Exchange correlation treatment:'  / &
             '   GGA     =',A6,  '    GGA type'/ & 
             '   LEXCH   =',I6,  '    internal setting for exchange type'/ & 
             '   VOSKOWN= ',I6,  '    Vosko Wilk Nusair interpolation'/&
             '   LHFCALC =',L6,  '    Hartree Fock is set to'/&
             '   LHFONE  =',L6,  '    Hartree Fock one center treatment'/ &
             '   AEXX    =',F10.4,' exact exchange contribution' / &
             )
       
    END SUBROUTINE WRITE_FOCK


!**********************************************************************
!
!      XML_WRITE_XC_FOCK
!   this subroutine writes the exchange parameters and the Fock
!   parameters to a file
!
!**********************************************************************

    SUBROUTINE XML_WRITE_XC_FOCK
      USE setexm
      USE vaspxml
      IMPLICIT NONE
! local
      INTEGER IDUM, N
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LDUM
      CHARACTER (1) :: CHARAC
 
      CALL XML_TAG("separator","electronic exchange-correlation")

      N=1
      CALL XML_INCAR('GGA','S',IDUM,RDUM,CDUM,LDUM,SZGGA,N)
      CALL XML_INCAR('VOSKOWN','I',LFCI,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('LHFCALC','L',IDUM,RDUM,CDUM,LHFCALC,CHARAC,N)
      CALL XML_INCAR('PRECFOCK','S',IDUM,RDUM,CDUM,LDUM,SZPRECFOCK,N)
      CALL XML_INCAR('LSYMGRAD','L',IDUM,RDUM,CDUM,LSYMGRAD,CHARAC,N)
      CALL XML_INCAR('LHFONE','L',IDUM,RDUM,CDUM,LHFONE,CHARAC,N)
      CALL XML_INCAR('LRHFCALC','L',IDUM,RDUM,CDUM,LRHFCALC,CHARAC,N)
      CALL XML_INCAR('LTHOMAS','L',IDUM,RDUM,CDUM,L_THOMAS_FERMI,CHARAC,N)
      CALL XML_INCAR('LMODELHF','L',IDUM,RDUM,CDUM,L_MODEL_HF,CHARAC,N)
      CALL XML_INCAR('ENCUT4O','F',IDUM,ENCUT4O,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('EXXOEP','I',EXXOEP,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('FOURORBIT','I',FOURORBIT,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('AEXX','F',IDUM,AEXX,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('HFALPHA','F',IDUM,HFALPHA,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('MCALPHA','F',IDUM,MCALPHA,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('ALDAX','F',IDUM,LDAX,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('AGGAX','F',IDUM,AGGAX,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('ALDAC','F',IDUM,ALDAC,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('AGGAC','F',IDUM,AGGAC,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('NKREDX','I',NKREDX,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('NKREDY','I',NKREDY,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('NKREDZ','I',NKREDZ,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('SHIFTRED','L',IDUM,RDUM,CDUM,SHIFTRED,CHARAC,N)
      CALL XML_INCAR('ODDONLY','L',IDUM,RDUM,CDUM,ODDONLY,CHARAC,N)
      CALL XML_INCAR('EVENONLY','L',IDUM,RDUM,CDUM,EVENONLY,CHARAC,N)
      CALL XML_INCAR('LMAXFOCK','I',LMAX_FOCK,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('NMAXFOCKAE','I',NMAX_FOCKAE,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('LFOCKAEDFT','L',IDUM,RDUM,CDUM,LFOCKAEDFT,CHARAC,N)
      CALL XML_INCAR('HFSCREEN','F',IDUM,HFSCREEN,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('HFSCREENC','F',IDUM,HFSCREENC,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('NBANDSGWLOW','I',NBANDSGWLOW_FOCK,RDUM,CDUM,LDUM,CHARAC,N)

      CALL XML_CLOSE_TAG
      CALL XML_TAG("separator","vdW DFT")

      CALL XML_INCAR('LUSE_VDW','L',IDUM,RDUM,CDUM,LUSE_VDW,CHARAC,N)
      CALL XML_INCAR('Zab_VDW','F',IDUM,Zab_VDW,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('PARAM1','F',IDUM,PARAM1,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('PARAM2','F',IDUM,PARAM2,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('PARAM3','F',IDUM,PARAM3,CDUM,LDUM,CHARAC,N)

      CALL XML_CLOSE_TAG

      CALL XML_TAG("separator","model GW")
      CALL XML_INCAR('MODEL_GW','I',MODEL_GW,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('MODEL_EPS0','F',IDUM,MODEL_GW_EPS0,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR('MODEL_ALPHA','F',IDUM,MODEL_GW_ALPHA,CDUM,LDUM,CHARAC,N)
      CALL XML_CLOSE_TAG


    END SUBROUTINE XML_WRITE_XC_FOCK


!***********************************************************************
!
! initialise the augmentation descriptors for the
! fast augmentation
! here is what is done:
! ) setup grid for HF operator and some stuff for nonlocal terms ((c) Georg)
!   ) this is alligned in real space with GRID (and GRID_US)
!   ) in reciprocal space it uses only half of the G-vectors
!     if gammareal is selected
!   ) points are distributed among the NPAR processors in COMM_INB
!
!***********************************************************************

    SUBROUTINE SETUP_FOCK(T_INFO, P, WDES, GRID, LATT_CUR, LMDIM, SZPREC, IU6, IU0 )
      USE pseudo
      USE constant
      USE asa
      USE wave
      USE mgrid
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      TYPE (wavedes), TARGET :: WDES ! wave function descriptor
      TYPE (grid_3d), TARGET :: GRID ! 3d- grid descriptor for wavefunctions
      TYPE (latt)    :: LATT_CUR     ! lattice parameters
      INTEGER        :: LMDIM        ! leading dimension of arrays like CDIJ
      INTEGER :: IU6, IU0            ! io units
      CHARACTER(LEN=6) :: SZPREC
    ! local
      INTEGER :: NT, L, NK, IND
      INTEGER :: N1, N2, N3
      REAL(q) :: G1, G2, G3, GIX, GIY, GIZ, ENERGI
      INTEGER :: NGX_FOCK, NGY_FOCK, NGZ_FOCK
      REAL(q) :: WFACT
    ! work array to perform 
      COMPLEX(q), ALLOCATABLE :: CWORK1(:)
      REAL(q) :: ENMAX
      REAL(q) :: SCALE
      LOGICAL, EXTERNAL :: CALCULATE_RESPONSE_FUNCTIONS,LDMATRIX

      IF (.NOT. LHFCALC .AND. .NOT. CALCULATE_RESPONSE_FUNCTIONS() .AND. .NOT. LDMATRIX()) RETURN
#ifdef MPI
!      IF (WDES%GRID%COMM%NCPU /=1 .OR. LUSE_PARALLEL_FFT) THEN
!         CALL VTUTOR('E','HFNPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!         IU6,3)
!         CALL VTUTOR('S','HFNPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
!         IU0,3)
!      ENDIF
#endif
      IF (IU6>=0) CALL WFORCE(IU6)

      ! check if PPs are OK
      typ:  DO NT=1,T_INFO%NTYP        
         IF ((WDES%LOVERL) .AND. ( .NOT. ASSOCIATED( P(NT)%QPAW )) .AND. (P(NT)%PSDMAX/=0)) THEN
            CALL VTUTOR('S','HFUSPP',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                 IU0,3)
            CALL VTUTOR('S','HFUSPP',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                 IU6,3)
         ENDIF
      ENDDO typ
!==========================================================================
! generate the required GRID structures and the modified descriptors
!  GRID_FOCK and WDES_FOCK
!==========================================================================
      ! simple case, link the descriptors
      GRID_FOCK=> GRID
      WDES_FOCK=> WDES
      LWDES_FOCK=.FALSE.
      ! difficult case regenerate the grids
      IF ( SZPRECFOCK(1:1) /= SZPREC(1:1) .AND. SZPRECFOCK(1:1) /= 'm' ) THEN
          ! first allocate the required structures and initialise to default
          ALLOCATE(GRID_FOCK)
          ALLOCATE(WDES_FOCK)
          GRID_FOCK=GRID
          WDES_FOCK=WDES
          LWDES_FOCK=.TRUE.

          IF (SZPRECFOCK(1:1)=='a') THEN
             ! high precision do not allow for wrap around
             WFACT=4
          ELSE 
             ! medium-low precision allow for small wrap around
             WFACT=3
          ENDIF
          
          ! determine minimum required values for NGX, Y and NGZ
          IF (SZPRECFOCK(1:1)=='a' .OR. SZPRECFOCK(1:1)=='n') THEN
             NGX_FOCK =SQRT(WDES%ENMAX /RYTOEV)/(2*PI/(LATT_CUR%ANORM(1)/AUTOA))*WFACT+1.0_q
             NGY_FOCK =SQRT(WDES%ENMAX /RYTOEV)/(2*PI/(LATT_CUR%ANORM(2)/AUTOA))*WFACT+1.0_q
             NGZ_FOCK= SQRT(WDES%ENMAX /RYTOEV)/(2*PI/(LATT_CUR%ANORM(3)/AUTOA))*WFACT+1.0_q
          ELSE
             NGX_FOCK=0
             NGY_FOCK=0
             NGZ_FOCK=0
          ENDIF

          ! new version loops over all k-points in the entire BZ
          ! to determine the required NGX
          DO N3=1,GRID%NGZ_rd
             DO N2=1,GRID%NGY
                DO NK=1,WDES%NKPTS_FOR_GEN_LAYOUT
                   G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
                   G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
                   DO N1=1,GRID%NGX_rd
                      G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
                      GIX= (G1*LATT_CUR%B(1,1)+G2*LATT_CUR%B(1,2)+G3*LATT_CUR%B(1,3)) *TPI
                      GIY= (G1*LATT_CUR%B(2,1)+G2*LATT_CUR%B(2,2)+G3*LATT_CUR%B(2,3)) *TPI
                      GIZ= (G1*LATT_CUR%B(3,1)+G2*LATT_CUR%B(3,2)+G3*LATT_CUR%B(3,3)) *TPI
                      ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))

                      ! exclude some components for gamma-only version (C(G)=C*(-G))
                      IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                         IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                         IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE
                      ENDIF

                      IF(ENERGI<WDES%ENMAX) THEN
                         NGX_FOCK=MAX(NGX_FOCK,ABS(GRID%LPCTX(N1))*2+2)
                         NGY_FOCK=MAX(NGY_FOCK,ABS(GRID%LPCTY(N2))*2+2)
                         NGZ_FOCK=MAX(NGZ_FOCK,ABS(GRID%LPCTZ(N3))*2+2)
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ENDDO

          ! in principle that should do
          ! however the previous algorithm sometimes leads to meshes that violate
          ! symmetry
          ! search for the next larger values that conserves symmetry
          WFACT=MAX(NGX_FOCK/LATT_CUR%ANORM(1),NGY_FOCK/LATT_CUR%ANORM(2),NGZ_FOCK/LATT_CUR%ANORM(3))+1E-6
          NGX_FOCK=WFACT*LATT_CUR%ANORM(1)
          NGY_FOCK=WFACT*LATT_CUR%ANORM(2)
          NGZ_FOCK=WFACT*LATT_CUR%ANORM(3)

          GRID_FOCK%NGPTAR(1)=NGX_FOCK
          GRID_FOCK%NGPTAR(2)=NGY_FOCK
          GRID_FOCK%NGPTAR(3)=NGZ_FOCK

          CALL FFTCHK(GRID_FOCK%NGPTAR)
          ! reinitialize the loop counters
          CALL INILGRD(GRID_FOCK%NGPTAR(1),GRID_FOCK%NGPTAR(2),GRID_FOCK%NGPTAR(3),GRID_FOCK)

          IF (IU6>=0) THEN
             WRITE(IU6,20) GRID_FOCK%NGPTAR
20           FORMAT(/' FFT grid for exact exchange (Hartree Fock) ',/ &
                  '  NGX =',I3,'; NGY =',I3,'; NGZ =',I3,/)
          ENDIF

          ! set WDES_FOCK%GRID to GRID_FOCK and generate the data layout for FFT's
          CALL GEN_LAYOUT(GRID_FOCK,WDES_FOCK, LATT_CUR%B, LATT_CUR%B, IU6, .TRUE.)
          CALL GEN_INDEX (GRID_FOCK,WDES_FOCK, LATT_CUR%B, LATT_CUR%B,-1, IU6, .TRUE.)

          CALLMPI( MAPSET(GRID_FOCK))  ! generate the communication maps (patterns) for FFT
          ! now check whether it is equivalent to original layout in reciprocal space (G)
          ! this is required since the FFTEXT and FFTWAV would not work from the orbital arrays
          CALL CHECK_GEN_LAYOUT( WDES,  WDES_FOCK, WDES%NKPTS)
          !  init FFT (required if real to complex FFT is used)
          CALL FFTINI(WDES_FOCK%NINDPW(1,1),WDES_FOCK%NGVECTOR(1),WDES_FOCK%NKPTS,WDES_FOCK%NGDIM,GRID_FOCK)
          ! set plane wave index array INDPW between NKPTS and NKDIM
          CALL SET_INDPW_FOCK(GRID_FOCK,WDES_FOCK)
      ENDIF
!==========================================================================
! other initializations
!==========================================================================
      ! set the indexing arrays for k-points outside IRZ
      CALL SET_INDPW_FOCK(GRID,WDES)

      GRIDHF%COMM=>WDES_FOCK%COMM_INB
      CALL INILGRD(GRID_FOCK%NGX,GRID_FOCK%NGY,GRID_FOCK%NGZ,GRIDHF)
      CALL GEN_GRID_HF(GRIDHF, GRID_FOCK)

      !  generate the communication maps (patterns) for FFT
      CALLMPI( MAPSET(GRIDHF))

      ALLOCATE(CWORK1(GRID_FOCK%MPLWV))
      CALL INIDAT(GRID_FOCK%RC%NP,CWORK1(1))
      CALL FFTMAKEPLAN(CWORK1(1),GRID_FOCK)
      DEALLOCATE(CWORK1)

#ifdef gammareal
      IF (.NOT.GRIDHF%LREAL) THEN
         WRITE(0,*) 'internal error in SETUP_FOCK: GRIDHF is not real'
         WRITE(0,*) ' the gamma only version in combination with Hartree-Fock requires'
         WRITE(0,*) ' to compile the code using -DNGXhalf or -DNGZhalf (mpiversion)'
         STOP
      ENDIF
#endif

      ! set up the structure (FAST_AUG_FOCK) to perform the fast augmentation 
      ! of the charge density

      IF (WDES%LOVERL .AND. LMAX_FOCK>=0) THEN
         CALL SETUP_FASTAUG( T_INFO, P, WDES, GRIDHF, LATT_CUR, LMDIM, SZPRECFOCK, & 
              FAST_AUG_FOCK, TRANS_MATRIX_FOCK, & 
              LMAX_FOCK, LMAX_FOCKAE, NMAX_FOCKAE, QMAX_FOCKAE, EXXOEP, IU6, IU0 )

         CALL RSPHER(GRIDHF, FAST_AUG_FOCK, LATT_CUR)
         FAST_AUG_FOCK%RPROJ= FAST_AUG_FOCK%RPROJ*SQRT(LATT_CUR%OMEGA)
         IF (IU6>=0) WRITE(IU6,*) 'Maximum index for augmentation-charges in exchange ',MAXVAL(FAST_AUG_FOCK%NLIMAX)

         ! set up AUG_DES describing data layout of one center terms
         CALL SETUP_AUG_DES(GRIDHF, WDES_FOCK, AUG_DES, FAST_AUG_FOCK )
      ELSE
         ALLOCATE(AUG_DES%LMMAX(SIZE(WDES%LMMAX)))
         ! to make live easier set LMMAX (number of channels) to zero for each type
         ! also set NPRO, NPROD etc. to zero
         AUG_DES%LMMAX=0
         AUG_DES%NPRO =0
         AUG_DES%NPROD=0
         AUG_DES%NPRO_TOT=0
      ENDIF

!      CALL SET_FOCK_ALPHA(GRIDHF%NGPTAR(1),GRIDHF%NGPTAR(2),GRIDHF%NGPTAR(3),LATT_CUR)

      ! set the decay constant for the exponential used to
      ! calculate convergence corrections
      ! this new setup yields stable values even for PRECFOCK = F
      ! 
      IF (HFALPHA==-1) THEN
         HFALPHA=6/(WDES%ENMAX /RYTOEV)/(2*PI/AUTOA)
      ENDIF

      IF (MCALPHA<0) THEN
         !     ideally nothing should depend on the value here: we started with ***/200 and
         !     then converged on this - seems to work:
         !        MCALPHA=(WDES%ENMAX /RYTOEV)/12
         MCALPHA=(WDES%ENMAX / RYTOEV)/18.9
         !        used in MP2 routines so they know that mcalpha was set here      
         MCALPHA_DEFAULT=.TRUE.

         IF (IU6>=0) THEN
            WRITE(IU6,"( / &
     &       ' MCALPHA set automatically to default value (depends on ENMAX)',/ & 
     &       '   MCALPHA =',F10.4,' extent of test-charge in conv. correction in multipole expansion'/)") MCALPHA
         ENDIF

      ENDIF

      ! precalculate and store convergence corrections
      IF (HFRCUT<0) THEN
         HFRCUT=0
         SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
         IF (ODDONLY .OR. EVENONLY) SCALE=SCALE*2
         HFRCUT=(LATT_CUR%OMEGA/SCALE/4/PI*3)**(1.0_q/3.0_q)
         IF (IU0>=0) WRITE(IU0,*) 'HFRCUT set to (new)',HFRCUT
         IF (IU6>=0) WRITE(IU6,*) 'HFRCUT set to (new)',HFRCUT
      ENDIF

      ALLOCATE (FSG_STORE(WDES%NKPTS))
      DO NK=1,WDES%NKPTS
         FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
      ENDDO

      IF (MCALPHA/=0) THEN 
         CALL FOCK_MULTIPOLE_CORR_SETUP(LATT_CUR, GRIDHF)
         CALL FOCK_MULTIPOLE_PROJ_SETUP(LATT_CUR, GRIDHF)
      ENDIF

      IF (IU6>=0) THEN
         WRITE(IU6,*) ' SETUP_FOCK is finished'
         CALL WFORCE(IU6)
      ENDIF

    END SUBROUTINE SETUP_FOCK

!************************ SUBROUTINE RESETUP_FOCK_WDES *****************
!
! this routine resets the WDES_FOCK and must be called when 
! WDES has been updated (for instance if the number of k-points)
! has been changed
!
!***********************************************************************

    SUBROUTINE RESETUP_FOCK_WDES(WDES, LATT_CUR, LATT_INI, IU6)
      USE wave
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (wavedes),TARGET :: WDES       ! wave function descriptor
      TYPE (latt)    :: LATT_CUR          ! lattice parameters
      TYPE (latt)    :: LATT_INI          ! initial lattice parameters (determining the basis set)
      INTEGER IU6
    ! local
      LOGICAL, EXTERNAL :: CALCULATE_RESPONSE_FUNCTIONS,LDMATRIX
      INTEGER :: NK
    

      IF (.NOT. LHFCALC .AND. .NOT. CALCULATE_RESPONSE_FUNCTIONS() .AND. .NOT. LDMATRIX()) RETURN

      IF ( LWDES_FOCK ) THEN
         CALL DEALLOCWDES(WDES_FOCK,LEXTEND=.TRUE.)
         IF (WDES%LOVERL .AND. LMAX_FOCK>=0) THEN
            CALL DEALLOC_AUG_DES(AUG_DES)
         ENDIF

         WDES_FOCK=WDES
         CALL GEN_LAYOUT(GRID_FOCK,WDES_FOCK, LATT_CUR%B, LATT_INI%B, IU6, .TRUE.)
         CALL GEN_INDEX (GRID_FOCK,WDES_FOCK, LATT_CUR%B, LATT_INI%B, -1,IU6, .TRUE.)

         ! now check whether it is equivalent to original layout
         ! this is required since the FFTEXT and FFTWAV would not work otherwise
         CALL CHECK_GEN_LAYOUT( WDES,  WDES_FOCK, WDES%NKPTS)
         
         IF (WDES%LOVERL .AND. LMAX_FOCK>=0) THEN
            CALL SETUP_AUG_DES(GRIDHF, WDES_FOCK, AUG_DES, FAST_AUG_FOCK )
         ENDIF
         ! set plane wave index array INDPW between NKPTS and NKDIM
         CALL SET_INDPW_FOCK(GRID_FOCK,WDES_FOCK)
      ELSE
         GRID_FOCK=> WDES%GRID
         WDES_FOCK=> WDES
      ENDIF
      CALL SET_INDPW_FOCK(WDES%GRID, WDES)

      ! re-determined convergence corrections

      DEALLOCATE(FSG_STORE)

      ALLOCATE (FSG_STORE(WDES%NKPTS))
      DO NK=1,WDES%NKPTS
         FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
      ENDDO

    END SUBROUTINE RESETUP_FOCK_WDES

!************************ SUBROUTINE RESETUP_FOCK     ******************
!
! this routine resets the FAST_AUG_FOCK structure
! must be called when the lattice vectors have been change
!
!***********************************************************************

    SUBROUTINE RESETUP_FOCK(WDES, LATT_CUR)
      USE wave
      USE lattice
      IMPLICIT NONE

      TYPE (wavedes) :: WDES       ! wave function descriptor
      TYPE (latt)    :: LATT_CUR   ! lattice parameters
      LOGICAL, EXTERNAL :: CALCULATE_RESPONSE_FUNCTIONS,LDMATRIX
      INTEGER :: NK
      LOGICAL :: LREALLOCATE

      IF ((LHFCALC .OR. CALCULATE_RESPONSE_FUNCTIONS() .OR. LDMATRIX()) .AND. WDES%LOVERL .AND. LMAX_FOCK>=0 ) THEN
         CALL REAL_OPTLAY(GRIDHF, LATT_CUR, FAST_AUG_FOCK, .TRUE., LREALLOCATE, -1,-1)
         IF (LREALLOCATE) THEN
            ! reallocate real space projectors
            CALL NONLR_DEALLOC(FAST_AUG_FOCK)
            CALL NONLR_ALLOC(FAST_AUG_FOCK)
         END IF

         CALL RSPHER(GRIDHF, FAST_AUG_FOCK, LATT_CUR)
         FAST_AUG_FOCK%RPROJ= FAST_AUG_FOCK%RPROJ*SQRT(LATT_CUR%OMEGA)

         ! recalculate convergence correction
         DO NK=1,WDES%NKPTS
            FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
         ENDDO

      ENDIF

    END SUBROUTINE RESETUP_FOCK
 
!************************ SUBROUTINE SET_FOCK_KPOINTS ******************
!
! this routine resets the NKDIM entry in the WDES array to the 
! total number of k-points in the entire BZ
! in this way the WDES and NONL structure will be allocated to allow 
! a handling of all k-points in the entire BZ
! this is in principle no longer required since the FOCK routine
! now calculates the wave function character using symmetry
! whereas previously it had to do it using the non local projector functions
!
!***********************************************************************

  SUBROUTINE SET_FOCK_KPOINTS(NKDIM)
    USE prec
    USE full_kpoints
    IMPLICIT NONE
    INTEGER NKDIM
    REAL(q), POINTER :: VKPT(:,:)

    IF (.NOT. LHFCALC) RETURN

    CALL CHECK_FULL_KPOINTS
    NKDIM=KPOINTS_FULL%NKPTS

  END SUBROUTINE SET_FOCK_KPOINTS

!********************** SUBROUTINE SET_INDPW_FOCK  *********************
!
! SET_INDPW_FOCK sets the INDPW array for the wavefunctions belonging to
! k-points that have symmetry-equivalents k-points in the IBZ
!
!  the following arrays in the WDES array are initialised as well:
!   NGVECTOR
!   IGX, IGY, IGZ
!   NINDPW
!
!***********************************************************************

    SUBROUTINE SET_INDPW_FOCK(GRID, WDES )
      USE prec
      USE mgrid
      USE wave
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) :: GRID
      TYPE (wavedes) :: WDES

      IF (.NOT. LHFCALC) RETURN
      
      CALL CHECK_FULL_KPOINTS
      CALL SET_INDPW_FULL(GRID, WDES, KPOINTS_FULL)

    END SUBROUTINE SET_INDPW_FOCK



!********************** SUBROUTINE SET_GFAC_VKPT **********************
!
! setup the (possibly screened)  Coloumb kernel 
!
! 4 pi e^2 / (G+k-q)**2 * screening
!
! or if LCONJG is set
!
! 4 pi e^2 / (G+k+q)**2 * screening
!
! additionally set the zero grid point value to the convergence
! correction FSG if NK==NQ (see below SET_FSG)
!
! for maximal performance
! the routine includes a weighting factor 
! 1/NRPLWV
! and furthermore includes a q-point
! weighting factor
!    KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ
!
! whereas the routine  SET_GFAC_WITHOUT_WEIGHT does not include
! a k-point weight
!
! see below for more details
! 
!**********************************************************************

    SUBROUTINE SET_GFAC_VKPT(GRID, LATT_CUR, VKPT, NQ, FSG, POTFAK, ENCUT)
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      REAL(q) :: VKPT(3)           ! coordinate of k-point in reciprocal lattice units
      INTEGER :: NQ                ! k-point at which contribution to exchange kernel is evaluated
      REAL(q) :: FSG
      REAL(q), OPTIONAL :: ENCUT
      REAL(q) :: POTFAK(GRID%MPLWV)
   ! local
      INTEGER    NI,NC,N1,N2,N3
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,GSQUP,GQUAD,SCALE, RHOEFF, FOMEGAP2

      CALL CHECK_FULL_KPOINTS

      ! effective electron density in a.u.^-3
      RHOEFF=(HFSCREEN*HFSCREEN*AUTOA*AUTOA/(4._q*EXP(LOG(3._q/PI)/3._q)))**3
      ! plasma frequency
      FOMEGAP2=16._q*PI*RHOEFF/(AUTOA*AUTOA*AUTOA*AUTOA)

      ! all k-points in the full grid have equal weight equal to first one
      SCALE=KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ *EDEPS/LATT_CUR%OMEGA/TPI**2
      IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2

      ! set correct phase in FAST_AUG structure
      CALL PHASER_HF(GRID, LATT_CUR, FAST_AUG_FOCK,VKPT)

      DKX=VKPT(1)*LATT_CUR%B(1,1)+VKPT(2)*LATT_CUR%B(1,2)+VKPT(3)*LATT_CUR%B(1,3)
      DKY=VKPT(1)*LATT_CUR%B(2,1)+VKPT(2)*LATT_CUR%B(2,2)+VKPT(3)*LATT_CUR%B(2,3)
      DKZ=VKPT(1)*LATT_CUR%B(3,1)+VKPT(2)*LATT_CUR%B(3,2)+VKPT(3)*LATT_CUR%B(3,3)

      NI=0
      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         row: DO N1=1,GRID%RC%NROW
            NI=NI+1
            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))
            GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
            IF (GSQU<KPOINTS_FULL%VKPTMINDIST2/40) THEN
               POTFAK(NI)=FSG
            ELSE
               IF (MODEL_GW>0) THEN
                  GSQUP=GSQU*(TPI*TPI)
                  GQUAD=GSQUP*GSQUP
                  ! corrected by gK 04.04.2014 (seems MODEL_GW_ALPHA is usually set to 1.0_q)
                  ! although the Bechstedt et al. Solid State Comm. 84, 765 suggests to use 1.5_q
                  POTFAK(NI)=SCALE/(GSQU)/ &
                       (1+1/(1/(MODEL_GW_EPS0-1)+MODEL_GW_ALPHA*GSQUP/(HFSCREEN*HFSCREEN)+GQUAD/FOMEGAP2))
               ELSE IF (HFRCUT/=0) THEN
                  ! spherical cutoff on Coloumb kernel
                  ! see for instance C.A. Rozzi, PRB 73, 205119 (2006)
                  POTFAK(NI)=SCALE/(GSQU)*(1-COS(SQRT(GSQU)*TPI*HFRCUT))
               ELSE IF (HFSCREEN==0) THEN
                  ! Coloumb kernel
                  POTFAK(NI)=SCALE/(GSQU)
               ELSE IF (L_THOMAS_FERMI) THEN
                  ! screened Thomas Fermi kernel
                  POTFAK(NI)=SCALE/(GSQU+HFSCREEN*HFSCREEN/(TPI*TPI))
               ELSE IF (LRHFCALC) THEN
                  ! exponentially screened kernel corresponding to erfc( HFSCREEN r)/r 
                  POTFAK(NI)=SCALE/GSQU*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
               ELSE IF (L_MODEL_HF) THEN
                  ! AEXX * case(HFSCREEN==0) + (1-AEXX)* default
                  POTFAK(NI)=SCALE/GSQU*(1-(1-AEXX)*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
               ELSE
                  POTFAK(NI)=SCALE/GSQU*(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
               ENDIF
            ENDIF
            IF (PRESENT(ENCUT)) THEN
               IF (HSQDTM*TPI*TPI*GSQU>ENCUT) THEN
                  POTFAK(NI)=0
               ENDIF
            ENDIF
            IF (ENCUTFOCK>0) THEN
               IF (HSQDTM*TPI*TPI*GSQU>ENCUTFOCK) THEN
                  POTFAK(NI)=0
               ENDIF
            ENDIF
         ENDDO row
      ENDDO col

      CALL SETUNB_REAL(POTFAK,GRID)

      IF (L_MODEL_HF) THEN
         DO NC=1,GRID%RC%NP
            POTFAK(NC)=POTFAK(NC)*(1.0_q/GRID%NPLWV)
         ENDDO
      ELSE

         DO NC=1,GRID%RC%NP
            POTFAK(NC)=POTFAK(NC)*(AEXX/GRID%NPLWV)
         ENDDO
      ENDIF
    END SUBROUTINE SET_GFAC_VKPT

!
! standard calling interface to the SET_GFAC routine
! requires one to pass down two k-points for which the Coulomb kernel is required
! it is assumed that the exchange operator is acting on an orbitals at psi_k
! \sum_q w_q phi_q(r)   phi^*_q(r') psi_k(r') / | r-r'|
! 
    SUBROUTINE SET_GFAC(GRID, LATT_CUR, NK, NQ, FSG, POTFAK, LCONJG, ENCUT)
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK, NQ
      REAL(q) :: FSG
      LOGICAL, OPTIONAL :: LCONJG
      REAL(q), OPTIONAL :: ENCUT
      REAL(q) :: POTFAK(GRID%MPLWV)

      IF (PRESENT(ENCUT).AND. PRESENT(LCONJG)) THEN
         CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)+KPOINTS_FULL%VKPT(:,NQ), NQ, FSG, POTFAK, ENCUT)
      ELSE IF (PRESENT(ENCUT)) THEN
         CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ), NQ, FSG, POTFAK, ENCUT)
      ELSE IF (PRESENT(LCONJG)) THEN
         CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)+KPOINTS_FULL%VKPT(:,NQ), NQ, FSG, POTFAK)
      ELSE
         CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ), NQ, FSG, POTFAK)
      ENDIF

    END SUBROUTINE SET_GFAC

    SUBROUTINE SET_GFAC_WITHOUT_WEIGHT(GRID, LATT_CUR, NK, NQ, FSG, POTFAK, LCONJG)
      USE constant
      USE mgrid
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK, NQ
      REAL(q) :: FSG
      REAL(q) :: POTFAK(GRID%MPLWV)
      LOGICAL, OPTIONAL :: LCONJG
      ! local
      INTEGER NC
      REAL(q) SCALE

      IF (PRESENT(LCONJG)) THEN
         CALL SET_GFAC(GRID, LATT_CUR, NK, NQ, FSG, POTFAK, LCONJG)
      ELSE
         CALL SET_GFAC(GRID, LATT_CUR, NK, NQ, FSG, POTFAK)
      ENDIF

      SCALE=1/(KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ)
      IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE/2

      DO NC=1,GRID%RC%NP
         POTFAK(NC)=POTFAK(NC)*SCALE
      ENDDO

    END SUBROUTINE SET_GFAC_WITHOUT_WEIGHT

!********************** SUBROUTINE SET_GFAC_WAVEFUN *******************
!
! setup the (possibly screened)  Coloumb kernel 
!
! this version calculates the kernel within the spherical cutoff
! sphere defined by the plane wave cutoff
! it is required in the context of GW routines and BSE routines
! 
!**********************************************************************

    SUBROUTINE SET_GFAC_WAVEFUN(WDES1, LATT_CUR, FSG, POTFAK, ENCUT, ENCUTSOFT)
      USE constant
      USE wave
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (wavedes1) WDES1   ! descriptor for a specific k-point
      TYPE (latt) :: LATT_CUR 
      REAL(q)     :: FSG
      REAL(q) :: POTFAK(WDES1%NGVECTOR)
      REAL(q), OPTIONAL :: ENCUT, ENCUTSOFT
   ! local
      INTEGER    NI,NP
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,GSQUP,GQUAD,SCALE, SCALEFSG, RHOEFF, FOMEGAP2
      REAL(q) :: E

      ! effective electron density in a.u.^-3
      RHOEFF=(HFSCREEN*HFSCREEN*AUTOA*AUTOA/(4._q*EXP(LOG(3._q/PI)/3._q)))**3
      ! plasma frequency
      FOMEGAP2=16._q*PI*RHOEFF/(AUTOA*AUTOA*AUTOA*AUTOA)

      SCALE=EDEPS/LATT_CUR%OMEGA/TPI**2

      IF (KPOINTS_FULL%WTKPT(1)==0) THEN
         WRITE(*,*) 'internal error in  SET_GFAC_WAVEFUN: division by zero in SCALEFSG, and FSG anyway most likely wrong'
         WRITE(*,*) ' needs to be fixed in  CALCULATE_LOCAL_FIELD_PREPARE as well'
         STOP
      ENDIF
      
      SCALEFSG=1/(KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ)
      IF (ODDONLY .OR. EVENONLY ) SCALEFSG=SCALEFSG/2

      DKX=WDES1%VKPT(1)*LATT_CUR%B(1,1)+ &
          WDES1%VKPT(2)*LATT_CUR%B(1,2)+ &
          WDES1%VKPT(3)*LATT_CUR%B(1,3)
      DKY=WDES1%VKPT(1)*LATT_CUR%B(2,1)+ &
          WDES1%VKPT(2)*LATT_CUR%B(2,2)+ &
          WDES1%VKPT(3)*LATT_CUR%B(2,3)
      DKZ=WDES1%VKPT(1)*LATT_CUR%B(3,1)+ &
          WDES1%VKPT(2)*LATT_CUR%B(3,2)+ &
          WDES1%VKPT(3)*LATT_CUR%B(3,3)

      NP=WDES1%NGVECTOR

      DO NI=1,NP
       
         GX=(WDES1%IGX(NI)*LATT_CUR%B(1,1)+WDES1%IGY(NI)* &
              LATT_CUR%B(1,2)+WDES1%IGZ(NI)*LATT_CUR%B(1,3))
         GY=(WDES1%IGX(NI)*LATT_CUR%B(2,1)+WDES1%IGY(NI)* &
              LATT_CUR%B(2,2)+WDES1%IGZ(NI)*LATT_CUR%B(2,3))
         GZ=(WDES1%IGX(NI)*LATT_CUR%B(3,1)+WDES1%IGY(NI)* &
              LATT_CUR%B(3,2)+WDES1%IGZ(NI)*LATT_CUR%B(3,3))
         
         GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
         IF (GSQU<KPOINTS_FULL%VKPTMINDIST2/40) THEN
            POTFAK(NI)=FSG*SCALEFSG
         ELSE
            IF (MODEL_GW>0) THEN
               GSQUP=GSQU*(TPI*TPI)
               GQUAD=GSQUP*GSQUP
               POTFAK(NI)=SCALE/(GSQU)/ &
                    (1+1/(1/(MODEL_GW_EPS0-1)+MODEL_GW_ALPHA*GSQUP/(HFSCREEN*HFSCREEN)+GQUAD/FOMEGAP2))
            ELSE IF (HFRCUT/=0) THEN
               POTFAK(NI)=SCALE/(GSQU)*(1-COS(SQRT(GSQU)*TPI*HFRCUT))
            ELSE IF (HFSCREEN==0) THEN
               POTFAK(NI)=SCALE/(GSQU)
            ELSE IF (L_THOMAS_FERMI) THEN
               POTFAK(NI)=SCALE/(GSQU+HFSCREEN*HFSCREEN/(TPI*TPI))
            ELSE IF (LRHFCALC) THEN
               POTFAK(NI)=SCALE/GSQU*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
            ELSE IF (L_MODEL_HF) THEN
               ! AEXX * case(HFSCREEN==0) + (1-AEXX)* default
               POTFAK(NI)=SCALE/GSQU*(1-(1-AEXX)*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
            ELSE
               POTFAK(NI)=SCALE/GSQU*(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
            ENDIF
         ENDIF

         ! smooth cutoff function between  ENCUTSOFT and ENCUT
         E=HSQDTM*(GSQU*TPI**2)
         IF (PRESENT(ENCUT).AND. PRESENT(ENCUTSOFT)) THEN 
            IF (E>ENCUTSOFT) THEN
               POTFAK(NI)=POTFAK(NI)*(1+COS((E-ENCUTSOFT)/(ENCUT-ENCUTSOFT)*PI))/2
            ENDIF
         ENDIF
      ENDDO

      IF (L_MODEL_HF) THEN
         DO NI=1,NP
            POTFAK(NI)=POTFAK(NI)*(1.0_q/WDES1%GRID%NPLWV)
         ENDDO
      ELSE
         DO NI=1,NP
            POTFAK(NI)=POTFAK(NI)*(AEXX/WDES1%GRID%NPLWV)
         ENDDO
      ENDIF

    END SUBROUTINE SET_GFAC_WAVEFUN


!********************** SUBROUTINE SET_FSG_QDER ************************
!
! calculate derivative of kernel with respect to q
! can be done easily anyltically, but really no time to dvelve into
! this for all functionals
!
! open issues: the derivative of FSG (the singularity correction)
! w.r.t k is not included presently
! I think it should be zero anyhow
!
!
!**********************************************************************

    SUBROUTINE SET_GFAC_QDER(GRID, LATT_CUR, NK, NQ, FSG, POTFAK, IDIR)
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK, NQ
      REAL(q) :: FSG
      REAL(q) :: POTFAK(GRID%MPLWV)
      INTEGER :: IDIR                  ! direction
    ! local
      REAL(q) :: POTFAK_(GRID%MPLWV)
      REAL(q) :: DISPL(3)
      REAL(q), PARAMETER :: DIS=fd_displacement
      
      ! shift k-point slightly
      DISPL=0
      DISPL(IDIR)=DIS/TPI
      CALL KARDIR(1, DISPL(1), LATT_CUR%A)

      CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ)-DISPL, NQ, FSG, POTFAK_)
      CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ)+DISPL, NQ, FSG, POTFAK)

      POTFAK(1:GRID%RC%NP)=(POTFAK(1:GRID%RC%NP)-POTFAK_(1:GRID%RC%NP))/DIS/2

    END SUBROUTINE SET_GFAC_QDER


!********************** SUBROUTINE SET_FOCK_ALPHA ********************
!
! the Posternak, Baldereschi convergence correction is defined
! as the energy difference between a single exponential charge 
!    exp(-HFALPHA*(G+k)^2*2 pi 2)
! and periodically repeated charges which are located
! on a lattice reciprocal to the generating  k-point set 
! ideally we would like to have rather contracted charges
! but there are limits imposed by the FFT grid
!
!**********************************************************************

   SUBROUTINE SET_FOCK_ALPHA(NGX,NGY,NGZ,LATT_CUR)
      USE prec
      USE lattice
      IMPLICIT NONE
      
      INTEGER NGX,NGY,NGZ
      TYPE (latt) LATT_CUR
      REAL(q) GSQU

      IF (HFALPHA==-1) THEN
         ! the minimum value for ALPHA is determine by the FFT grid
         ! seek shortest wave vector
         ! NGn * b(n) * a(n)/||a(n)||
         ! updated 24.10.2009 NGX/2-> NGX and summation in calculation of FSG
         ! from -NGX to +NGX
         GSQU=MIN(NGX/2/LATT_CUR%ANORM(1), &
                  NGY/2/LATT_CUR%ANORM(2), &
                  NGZ/2/LATT_CUR%ANORM(3))**2

         ! the larger GSQU is the smaller alpha can be
         ! corresponding to a more contracted charge
         ! the default below leads to errors of 0.5 meV
         ! compared to including an infinite number of G vector
         ! HFALPHA=1/GSQU/5 gK updated on 09.10.2008
         ! larger alpha results in more screened potential at large
         ! distances, this speeds up convergence
         HFALPHA=1/GSQU/2
      ENDIF

    END SUBROUTINE SET_FOCK_ALPHA

!********************** SUBROUTINE SET_FSG  ***************************
!
! integratable singularity of exchange term:
! calculate the average electrostatic potential prefactor divided
! by the number of k-points
! essentially  
!  4 pi e^2 / G^2 / NKPTS
!
! for the orbitals with k=k' and n=n' (i.e. those that are affected
!  by divergence of the HF term in periodic boundary conditions)
! this is the convergence accelerator suggested by F. Giggy and Baldereschi
!
!   sum_G,k epsilon/omega/pi^2 exp(-HFALPHA*(G+k)^2*2 pi 2)/ (G+k)^2
!
! if the generation scheme of the k-point mesh is known use a simple 
! monopole correction
!
!**********************************************************************

    FUNCTION SET_FSG(GRID, LATT_CUR, NK_ORIG)
      USE prec
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE
      
      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER, OPTIONAL :: NK_ORIG
    ! local
      INTEGER NK,NQ
      REAL(q) :: SET_FSG
      REAL(q) :: HFALPHAP
      INTEGER    NC,N1,N2,N3, NN1, NN2, NN3, LPX, LPY, LPZ
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,SCALE,FSG,FACTM
      TYPE (latt) LATT_EWALD

      CALL CHECK_FULL_KPOINTS


!
! spherical cutoff on Coloumb kernel
! see for instance C.A. Rozzi, PRB 73, 205119 (2006)
!
      IF (HFRCUT/=0) THEN
         SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
         IF (ODDONLY .OR. EVENONLY) SCALE=SCALE*2
         FSG=HFRCUT*HFRCUT/2*EDEPS/LATT_CUR%OMEGA*SCALE
!
! simple version, that can be applied if
! a Monkhorst Pack or Gamma centered grid is used
! updated 13.03.2015, faster calculation of FSG for shifted grids
      ELSE IF (KPOINTS_FULL%NKPTS_NON_ZERO==KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ &
!      ELSE IF (KPOINTS_FULL%NKPTS==KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ &
           .AND. (HFSCREEN==0.OR. MODEL_GW>0 ).AND. .NOT. ODDONLY .AND. .NOT. EVENONLY .AND. .NOT. HFKIDENT) THEN
         LATT_EWALD=LATT_CUR
         LATT_EWALD%A(:,1)=LATT_EWALD%A(:,1)*KPOINTS_FULL%NKPX/NKREDX
         LATT_EWALD%A(:,2)=LATT_EWALD%A(:,2)*KPOINTS_FULL%NKPY/NKREDY
         LATT_EWALD%A(:,3)=LATT_EWALD%A(:,3)*KPOINTS_FULL%NKPZ/NKREDZ
         CALL LATTIC(LATT_EWALD)
         CALL EWALD_MONO(FSG,1.0_q,LATT_EWALD)
         FSG=FSG*2
         IF (MODEL_GW>0) FSG=FSG/MODEL_GW_EPS0
      ELSE
!
! Giggy, Posternak, Baldereschi version
!
         IF (PRESENT(NK_ORIG)) THEN
            NK=NK_ORIG
         ELSE
            NK=1
         ENDIF

         FSG=0
         DO NQ=1,KPOINTS_FULL%NKPTS
            IF (KPOINTS_FULL%WTKPT(NQ)==0) CYCLE
            SCALE=KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ *EDEPS/LATT_CUR%OMEGA/TPI**2
            IF (ODDONLY .OR. EVENONLY) SCALE=SCALE*2
            IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-KPOINTS_FULL%VKPT(:,NK))) CYCLE

#ifndef gammareal
            IF (.NOT. HFKIDENT) THEN
            DKX=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(1,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(1,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(1,3)
            DKY=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(2,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(2,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(2,3)
            DKZ=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(3,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(3,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(3,3)
            ELSE
            DKX=( -KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(1,1)+ &
                 (-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(1,2)+ &
                 (-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(1,3)
            DKY=( -KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(2,1)+ &
                 (-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(2,2)+ &
                 (-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(2,3)
            DKZ=( -KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(3,1)+ &
                 (-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(3,2)+ &
                 (-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(3,3)
            ENDIF
#else
            DKX=0._q; DKY=0._q; DKZ=0._q
#endif
            DO NN1=0,GRID%NGX,GRID%NGX
            DO NN2=0,GRID%NGY,GRID%NGY
            DO NN3=0,GRID%NGZ,GRID%NGZ
            col: DO NC=1,GRID%RC%NCOL
               N2=GRID%RC%I2(NC)
               N3=GRID%RC%I3(NC)

               row: DO N1=1,GRID%RC%NROW
                  LPX=MOD(GRID%LPCTX(N1)+NN1+GRID%NGX-1,2*GRID%NGX)-GRID%NGX+1
                  LPY=MOD(GRID%LPCTY(N2)+NN2+GRID%NGY-1,2*GRID%NGY)-GRID%NGY+1
                  LPZ=MOD(GRID%LPCTZ(N3)+NN3+GRID%NGZ-1,2*GRID%NGZ)-GRID%NGZ+1
                  GX=(LPX*LATT_CUR%B(1,1)+LPY*LATT_CUR%B(1,2)+LPZ*LATT_CUR%B(1,3))
                  GY=(LPX*LATT_CUR%B(2,1)+LPY*LATT_CUR%B(2,2)+LPZ*LATT_CUR%B(2,3))
                  GZ=(LPX*LATT_CUR%B(3,1)+LPY*LATT_CUR%B(3,2)+LPZ*LATT_CUR%B(3,3))
                  GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

                  FACTM=1._q
                  ! IF (GRID%LREAL .AND. (N1/=1 .AND. (N1/=GRID%RC%NROW))) FACTM=2._q
                  IF ( GRID%NGX /= GRID%NGX_rd .AND. (N1/=1 .AND. (N1/= GRID%NGX_rd))) FACTM=2._q
                  IF ( GRID%NGZ /= GRID%NGZ_rd .AND. (N3/=1 .AND. (N3/= GRID%NGZ_rd))) FACTM=2._q

                  IF (GSQU < KPOINTS_FULL%VKPTMINDIST2/40) THEN
                  ELSE
                     IF (MODEL_GW>0) THEN
                        FSG=FSG+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/GSQU/MODEL_GW_EPS0
                     ELSE IF (HFSCREEN==0) THEN
                        ! Coulomb kernel
                        FSG=FSG+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/GSQU
                     ELSE IF (L_THOMAS_FERMI) THEN
                        FSG=FSG+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/(GSQU+HFSCREEN*HFSCREEN*(1.0_q/(TPI*TPI)))
                     ELSE IF (LRHFCALC) THEN
                        ! long range Coulomb kernel
                        FSG=FSG+FACTM*SCALE*EXP(GSQU*((-HFALPHA-1/(4*HFSCREEN*HFSCREEN))*TPI*TPI))/GSQU
                     ELSE IF (L_MODEL_HF) THEN
                        FSG=FSG+FACTM*SCALE/GSQU*EXP(-GSQU*(HFALPHA*TPI*TPI)) &
                               *(1-(1-AEXX)*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
                     ELSE
                        ! screened Coulomb kernel (short range)
                        FSG=FSG+FACTM*SCALE/GSQU*EXP(-GSQU*(HFALPHA*TPI*TPI)) &
                           *(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
                     ENDIF
                  ENDIF
               ENDDO row
            ENDDO col
            ENDDO
            ENDDO
            ENDDO
         ENDDO

         CALLMPI (M_sum_d(GRID%COMM,FSG,1))

         IF (MODEL_GW>0) THEN
            ! add selfenergy of an isolated erfc(r)  charge distribution / epsilon
            FSG=-FSG+EDEPS/(TPI**3)*2._q*HFALPHA*(PI/HFALPHA)**(3._q/2._q)/MODEL_GW_EPS0
         ELSE IF (HFSCREEN==0) THEN
            ! add selfenergy of an isolated erfc(r)  charge distribution
            FSG=-FSG+EDEPS/(TPI**3)*2._q*HFALPHA*(PI/HFALPHA)**(3._q/2._q)
         ELSE IF (L_THOMAS_FERMI) THEN
            FSG=0
         ELSE IF (LRHFCALC) THEN
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            !
            ! we need the selfenergy of an erfc(r) charge distribution 
            ! for the long range Coulomb kernel
            !
            FSG=-FSG+EDEPS/(TPI**3)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ELSE IF (L_MODEL_HF) THEN
            !
            ! AEXX * case(HFSCREEN==0) + (1-AEXX)* default
            !
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            FSG=-FSG+EDEPS/(TPI**3)*2._q*HFALPHA *(PI/HFALPHA )**(3._q/2._q)- &
                  (1-AEXX)*EDEPS/(TPI**3)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ELSE 
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            !
            ! we need the selfenergy of an erfc(r) charge distribution in
            ! a screened short range Coloumb potential 
            !
            FSG=-FSG+EDEPS/(TPI**3)*2._q*HFALPHA *(PI/HFALPHA )**(3._q/2._q)- &
                 EDEPS/(TPI**3)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ENDIF
      ENDIF
      SET_FSG=FSG

    END FUNCTION SET_FSG


!********************** SUBROUTINE SET_GFAC_DER ***********************
!
! setup the derivative of the (possibly screened) 
! Coloumb kernel  e^2 /4 pi (G+k-q)**2 * (1-exp(  (G+k-q)**2/4/alpha^2))
! corresponding the erfc(alpha*r)/r
! additionally set the zero grid point value to FSG
! if NK==NQ (see below SET_FSG)
! 
!**********************************************************************

    SUBROUTINE SET_GFAC_DER(GRID, LATT_CUR, NK, NQ, FSG, POTFAK)
      USE prec
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE
      
      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK,NQ
      REAL(q) :: FSG(0:6)
      REAL(q) :: POTFAK(GRID%MPLWV,0:6)
   
      INTEGER    NI,NC,N1,N2,N3,I
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,GSQU2,SCALE,EXP_GSQU_DER,FACTM

      CALL CHECK_FULL_KPOINTS

      SCALE=KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ *EDEPS/LATT_CUR%OMEGA/TPI**2
      IF (ODDONLY .OR. EVENONLY) SCALE=SCALE*2
#ifndef gammareal
      DKX=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(1,1)+ &
           (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(1,2)+ &
           (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(1,3)
      DKY=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(2,1)+ &
           (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(2,2)+ &
           (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(2,3)
      DKZ=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(3,1)+ &
           (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(3,2)+ &
           (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(3,3)
! set correct phase in FAST_AUG structure
      CALL PHASER_HF(GRID, LATT_CUR, FAST_AUG_FOCK,KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ))
#else
      DKX=0._q; DKY=0._q; DKZ=0._q
#endif
      NI=0

      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         
         row: DO N1=1,GRID%RC%NROW
            NI=NI+1
            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))+DKX
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))+DKY
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))+DKZ
            GSQU=GX**2+GY**2+GZ**2
            GSQU2=GSQU*GSQU

            FACTM=1._q
            IF ( GRID%NGX /= GRID%NGX_rd .AND. (N1/=1 .AND. (N1/= GRID%NGX_rd))) FACTM=2._q
            IF ( GRID%NGZ /= GRID%NGZ_rd .AND. (N3/=1 .AND. (N3/= GRID%NGZ_rd))) FACTM=2._q

            IF ((GRID%LPCTX(N1)==0).AND.(GRID%LPCTY(N2)==0).AND.&
                 (GRID%LPCTZ(N3)==0) .AND. (NK==NQ)) THEN
               POTFAK(NI,0)=FSG(0)
               POTFAK(NI,1)=FSG(1)*FACTM
               POTFAK(NI,2)=FSG(2)*FACTM
               POTFAK(NI,3)=FSG(3)*FACTM
               POTFAK(NI,4)=FSG(4)*FACTM
               POTFAK(NI,5)=FSG(5)*FACTM
               POTFAK(NI,6)=FSG(6)*FACTM
            ELSE
               IF (HFSCREEN==0) THEN
                  ! derivative  -1/2 d (1/OMEGA/G^2) / d t_ij = G_i G_j/G^4 /OMEGA  - 1/OMEGA/G^2/2 delta_ij
                  POTFAK(NI,0)=SCALE/(GSQU)
                  EXP_GSQU_DER=SCALE/GSQU2
               ELSE IF (L_THOMAS_FERMI) THEN
                  POTFAK(NI,0)=SCALE/(GSQU+HFSCREEN*HFSCREEN/(TPI*TPI))
               ELSE IF (LRHFCALC) THEN
                  POTFAK(NI,0)=SCALE/GSQU*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))
                  EXP_GSQU_DER=SCALE*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))) &
                              *(1/GSQU2+TPI*TPI/(4*HFSCREEN*HFSCREEN)/GSQU)
               ELSE IF (L_MODEL_HF) THEN
                  ! AEXX * case(HFSCREEN==0) + (1-AEXX)* default
                  POTFAK(NI,0)=SCALE/GSQU*(1-(1-AEXX)*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
               ELSE
                  POTFAK(NI,0)=(SCALE/GSQU )*(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
                  ! negative derivative of previous equation
                  EXP_GSQU_DER=(SCALE/GSQU2)*(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))- &
                               (SCALE/GSQU )*(EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))* & 
                               (TPI*TPI/(4*HFSCREEN*HFSCREEN))
               ENDIF
               POTFAK(NI,1)=(GX*GX*EXP_GSQU_DER-POTFAK(NI,0)/2)*FACTM
               POTFAK(NI,2)=(GX*GY*EXP_GSQU_DER)*FACTM
               POTFAK(NI,3)=(GY*GY*EXP_GSQU_DER-POTFAK(NI,0)/2)*FACTM
               POTFAK(NI,4)=(GX*GZ*EXP_GSQU_DER)*FACTM
               POTFAK(NI,5)=(GY*GZ*EXP_GSQU_DER)*FACTM
               POTFAK(NI,6)=(GZ*GZ*EXP_GSQU_DER-POTFAK(NI,0)/2)*FACTM
            ENDIF
         ENDDO row
      ENDDO col

      IF (L_MODEL_HF) THEN
         DO I=0,6
            DO NC=1,GRID%RC%NP
               POTFAK(NC,I)=POTFAK(NC,I)*(1.0_q/GRID%NPLWV)
            ENDDO
         ENDDO
      ELSE
         DO I=0,6
            DO NC=1,GRID%RC%NP
               POTFAK(NC,I)=POTFAK(NC,I)*(AEXX/GRID%NPLWV)
            ENDDO
         ENDDO
      ENDIF
      
      DO I=0,6
         CALL SETUNB_REAL(POTFAK(1,I),GRID)
      ENDDO

    END SUBROUTINE SET_GFAC_DER


!********************** SUBROUTINE SET_FSG_DER  ************************
!
! integratable singularity of exchange term:
! calculate the derivative of the average electrostatic potential prefactor
! for the orbitals with k=k' and n=n' (i.e. those that are affected
!  by divergence of the HF term in periodic boundary conditions)
!
!  Thesis gK Appendix B might be helpfull
!  for the Coloum kernel this is:
!  d / d t_ij  C/ Omega sum_G e(-alpha G^2) / G^2
!       =  delta_ij C/ Omega sum_G e(-alpha G^2) / G^2 +
!        + C/ Omega sum_G e(-alpha G^2) ( alpha / G^2 + 1/ G^4) G_i G_j
!
!**********************************************************************

    SUBROUTINE SET_FSG_DER(GRID, LATT_CUR, FSG)
      USE prec
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      IMPLICIT NONE
      
      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK,NQ
      REAL(q) :: FSG(0:6)
      REAL(q) :: HFALPHAP
    ! local
      INTEGER    NC,N1,N2,N3, NN1, NN2, NN3, LPX, LPY, LPZ
      REAL(q) :: DKX,DKY,DKZ,GX,GY,GZ,GSQU,GSQU2,SCALE,EXP_GSQU_DER,FACTM
      INTEGER :: IDIR, JDIR, I
      REAL(q) :: FSG1, FSG2
      TYPE (latt) LATT_EWALD, LATT_FIN
      REAL(q) :: DIS=fd_displacement

      CALL CHECK_FULL_KPOINTS
!
! this version is for molecules
! stress calculations (which require the derivative of FSG with respect to the volume)
! are not applicable
!
      IF (HFRCUT/=0) THEN
         FSG=0
         FSG(0)=HFRCUT*HFRCUT*EDEPS/LATT_CUR%OMEGA/2
!
! simple version, that can be applied for
! a simple Monkhorst Pack or Gamma centered grid
! updated 13.03.2015, faster calculation of FSG for shifted grids
      ELSE IF (KPOINTS_FULL%NKPTS_NON_ZERO==KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ .AND. HFSCREEN==0 &
!      ELSE IF (KPOINTS_FULL%NKPTS==KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ .AND. HFSCREEN==0 &
         .AND. .NOT. ODDONLY .AND. .NOT. EVENONLY) THEN
         LATT_EWALD=LATT_CUR

         LATT_EWALD%A(:,1)=LATT_EWALD%A(:,1)*KPOINTS_FULL%NKPX/NKREDX
         LATT_EWALD%A(:,2)=LATT_EWALD%A(:,2)*KPOINTS_FULL%NKPY/NKREDY
         LATT_EWALD%A(:,3)=LATT_EWALD%A(:,3)*KPOINTS_FULL%NKPZ/NKREDZ

         CALL LATTIC(LATT_EWALD)
         CALL EWALD_MONO(FSG(0),1.0_q,LATT_EWALD)
         FSG(0)=FSG(0)*2

         I=0
         DO IDIR=1,3
            DO JDIR=1,IDIR
               I=I+1
               LATT_FIN=LATT_EWALD
               LATT_FIN%A(IDIR,:)=LATT_EWALD%A(IDIR,:)+DIS*LATT_EWALD%A(JDIR,:)
               CALL LATTIC(LATT_FIN)
               CALL EWALD_MONO(FSG1,1.0_q,LATT_FIN)
               LATT_FIN%A(IDIR,:)=LATT_EWALD%A(IDIR,:)-DIS*LATT_EWALD%A(JDIR,:)
               CALL LATTIC(LATT_FIN)
               CALL EWALD_MONO(FSG2,1.0_q,LATT_FIN)
               FSG(I)=(FSG1-FSG2)/2/DIS
            ENDDO
         ENDDO
      ELSE

         FSG=0
         NK=1
         DO NQ=1,KPOINTS_FULL%NKPTS
            SCALE=KPOINTS_FULL%WTKPT(NQ)*NKREDX*NKREDY*NKREDZ *EDEPS/LATT_CUR%OMEGA/TPI**2
            IF (ODDONLY .OR. EVENONLY) SCALE=SCALE*2
            IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-KPOINTS_FULL%VKPT(:,NK))) CYCLE

#ifndef gammareal
            DKX=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(1,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(1,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(1,3)
            DKY=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(2,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(2,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(2,3)
            DKZ=(KPOINTS_FULL%VKPT(1,NK)-KPOINTS_FULL%VKPT(1,NQ))*LATT_CUR%B(3,1)+ &
                 (KPOINTS_FULL%VKPT(2,NK)-KPOINTS_FULL%VKPT(2,NQ))*LATT_CUR%B(3,2)+ &
                 (KPOINTS_FULL%VKPT(3,NK)-KPOINTS_FULL%VKPT(3,NQ))*LATT_CUR%B(3,3)
#else
            DKX=0._q; DKY=0._q; DKZ=0._q
#endif
            DO NN1=0,GRID%NGX,GRID%NGX
            DO NN2=0,GRID%NGY,GRID%NGY
            DO NN3=0,GRID%NGZ,GRID%NGZ

            col: DO NC=1,GRID%RC%NCOL
               N2=GRID%RC%I2(NC)
               N3=GRID%RC%I3(NC)
               
               row: DO N1=1,GRID%RC%NROW
                  LPX=MOD(GRID%LPCTX(N1)+NN1+GRID%NGX-1,2*GRID%NGX)-GRID%NGX+1
                  LPY=MOD(GRID%LPCTY(N2)+NN2+GRID%NGY-1,2*GRID%NGY)-GRID%NGY+1
                  LPZ=MOD(GRID%LPCTZ(N3)+NN3+GRID%NGZ-1,2*GRID%NGZ)-GRID%NGZ+1

                  GX=(LPX*LATT_CUR%B(1,1)+LPY*LATT_CUR%B(1,2)+LPZ*LATT_CUR%B(1,3))+DKX
                  GY=(LPX*LATT_CUR%B(2,1)+LPY*LATT_CUR%B(2,2)+LPZ*LATT_CUR%B(2,3))+DKY
                  GZ=(LPX*LATT_CUR%B(3,1)+LPY*LATT_CUR%B(3,2)+LPZ*LATT_CUR%B(3,3))+DKZ
                  GSQU=GX**2+GY**2+GZ**2
                  GSQU2=GSQU*GSQU
                  
                  FACTM=1._q
                  ! IF (GRID%LREAL .AND. (N1/=1 .AND. (N1/=GRID%RC%NROW))) FACTM=2._q
                  IF ( GRID%NGX /= GRID%NGX_rd .AND. (N1/=1 .AND. (N1/= GRID%NGX_rd))) FACTM=2._q
                  IF ( GRID%NGZ /= GRID%NGZ_rd .AND. (N3/=1 .AND. (N3/= GRID%NGZ_rd))) FACTM=2._q
                  
                  IF ((GRID%LPCTX(N1)==0).AND.(GRID%LPCTY(N2)==0).AND.&
                       (GRID%LPCTZ(N3)==0) .AND. (NK==NQ)) THEN
                  ELSE
                     IF (HFSCREEN==0) THEN
                        ! Coloumb kernel
                        FSG(0)=FSG(0)+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/GSQU
                        ! negative derivative of Coloumb kernel with  respect to G^2 (GSQU)
                        EXP_GSQU_DER =FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))*(1/GSQU2+HFALPHA*TPI*TPI/GSQU)
                     ELSE IF (L_THOMAS_FERMI) THEN
                        FSG(0)=FSG(0)+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/(GSQU+HFSCREEN*HFSCREEN*(1.0_q/(TPI*TPI)))
                        WRITE(*,*) 'Stress not implemented'
                        STOP
                     ELSE IF (LRHFCALC) THEN
                        ! long range Coulomb kernel
                        FSG(0)=FSG(0)+FACTM*SCALE*EXP(GSQU*((-HFALPHA-1/(4*HFSCREEN*HFSCREEN))*TPI*TPI))/GSQU
!                       WRITE(*,*) 'Stress not implemented'
!                       STOP
                        ! negative derivative of kernel with  respect to G^2 (GSQU)
                        EXP_GSQU_DER =FACTM*SCALE*EXP(GSQU*((-HFALPHA-1/(4*HFSCREEN*HFSCREEN))*TPI*TPI)) &
                                    *(1/GSQU2+(HFALPHA+1/(4*HFSCREEN*HFSCREEN))*TPI*TPI/GSQU)

                     ELSE IF (L_MODEL_HF) THEN
                        FSG=FSG+FACTM*SCALE/GSQU*EXP(-GSQU*(HFALPHA*TPI*TPI)) &
                             *(1-(1-AEXX)*EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
                        WRITE(*,*) 'Stress not implemented'
                        STOP
                     ELSE
                        ! screened Coulomb kernel (short range)
                        FSG(0)=FSG(0)+FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/GSQU &
                             *(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))
                        !
                        ! negative derivative of kernel with  respect to G^2 (GSQU)
                        EXP_GSQU_DER=FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))*(1/GSQU2+(HFALPHA*TPI*TPI)/GSQU) &
                             *(1-EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN))))- &
                             FACTM*SCALE*EXP(-GSQU*(HFALPHA*TPI*TPI))/GSQU* &
                             EXP(-GSQU*(TPI*TPI/(4*HFSCREEN*HFSCREEN)))*(TPI*TPI/(4*HFSCREEN*HFSCREEN))
                     ENDIF
                     ! derivative of d G^2 / d t_ij = - 2 G_i G_j
                     FSG(1)=FSG(1)+GX*GX*EXP_GSQU_DER
                     FSG(2)=FSG(2)+GX*GY*EXP_GSQU_DER
                     FSG(3)=FSG(3)+GY*GY*EXP_GSQU_DER
                     FSG(4)=FSG(4)+GX*GZ*EXP_GSQU_DER
                     FSG(5)=FSG(5)+GY*GZ*EXP_GSQU_DER
                     FSG(6)=FSG(6)+GZ*GZ*EXP_GSQU_DER
                  ENDIF
               ENDDO row
            ENDDO col
            ENDDO
            ENDDO
            ENDDO
         ENDDO
         
         CALLMPI (M_sum_d(GRID%COMM,FSG,7))

         ! add contribution from one over volume dependency of FSG
         FSG(1)=FSG(1)-FSG(0)/2
         FSG(3)=FSG(3)-FSG(0)/2
         FSG(6)=FSG(6)-FSG(0)/2

      
         IF (HFSCREEN==0) THEN
            ! selfenergy of an isolated erfc(r) charge distribution
            FSG(0)=-FSG(0)+EDEPS/(TPI**3._q)*2._q*HFALPHA*(PI/HFALPHA)**(3._q/2._q)
            
         ELSE IF (L_THOMAS_FERMI) THEN
            ! Thomas Fermi, no analytical formula but small hence zero it
            FSG=0
         ELSE IF (LRHFCALC) THEN
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            !
            ! selfenergy of an erfc(r) charge distribution 
            ! for the long range Coulomb kernel
            !
            FSG(0)=-FSG(0)+EDEPS/(TPI**3._q)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ELSE IF (L_MODEL_HF) THEN
            !
            ! AEXX * case(HFSCREEN==0) + (1-AEXX)* default
            !
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            FSG(0)=-FSG(0)+EDEPS/(TPI**3)*2._q*HFALPHA *(PI/HFALPHA )**(3._q/2._q)- &
                 (1-AEXX)*EDEPS/(TPI**3)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ELSE 
            HFALPHAP=HFALPHA+1/HFSCREEN/HFSCREEN/4
            !
            ! selfenergy of an erfc(r) charge distribution in
            ! a screened short range Coloumb potential 
            !
            FSG(0)=-FSG(0)+EDEPS/(TPI**3._q)*2._q*HFALPHA *(PI/HFALPHA )**(3._q/2._q)- &
                 EDEPS/(TPI**3._q)*2._q*HFALPHAP*(PI/HFALPHAP)**(3._q/2._q)
         ENDIF
         
         FSG(1:6)=-FSG(1:6)
      ENDIF
    END SUBROUTINE SET_FSG_DER

!***********************************************************************
!
! Determine whether a specific q point (usually k1-k2) should
! be used in the calculation of Bloch integrals such 
! as exact exchange
! 
!***********************************************************************

    FUNCTION SKIP_THIS_KPOINT_IN_FOCK(VKPT)
      USE full_kpoints
      REAL(q) VKPT(3)
      LOGICAL SKIP_THIS_KPOINT_IN_FOCK

      SKIP_THIS_KPOINT_IN_FOCK=.FALSE.
      IF (ODDONLY .AND. ABS(MODULO(NINT(VKPT(1)*KPOINTS_FULL%NKPX+ & 
                                        VKPT(2)*KPOINTS_FULL%NKPY+ & 
                                        VKPT(3)*KPOINTS_FULL%NKPZ),2))<=1E-6)  SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
      IF (EVENONLY.AND. ABS(MODULO(NINT(VKPT(1)*KPOINTS_FULL%NKPX+ & 
                                        VKPT(2)*KPOINTS_FULL%NKPY+ & 
                                        VKPT(3)*KPOINTS_FULL%NKPZ+1),2))<=1E-6)  SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
      IF (SHIFTRED) THEN
         IF (NKREDX==2 .AND. MOD(FLOOR(VKPT(1)*KPOINTS_FULL%NKPX+100.5_q),NKREDX)==0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
         IF (NKREDY==2 .AND. MOD(FLOOR(VKPT(2)*KPOINTS_FULL%NKPY+100.5_q),NKREDY)==0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
         IF (NKREDY==2 .AND. MOD(FLOOR(VKPT(3)*KPOINTS_FULL%NKPZ+100.5_q),NKREDZ)==0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
      ELSE IF (NKREDX>=2 .OR. NKREDY>=2 .OR. NKREDZ>=2) THEN
         IF (MOD(FLOOR((VKPT(1)+32)*KPOINTS_FULL%NKPX+.5_q),NKREDX)/=0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
         IF (MOD(FLOOR((VKPT(2)+32)*KPOINTS_FULL%NKPY+.5_q),NKREDY)/=0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
         IF (MOD(FLOOR((VKPT(3)+32)*KPOINTS_FULL%NKPZ+.5_q),NKREDZ)/=0) SKIP_THIS_KPOINT_IN_FOCK=.TRUE.
      ENDIF
    END FUNCTION SKIP_THIS_KPOINT_IN_FOCK

!***********************************************************************
!
! allocate the handle for HF type calculations
! 
!***********************************************************************

    SUBROUTINE ALLOCATE_FOCK_HANDLE( FH , LMDIM , NBANDSGW)
      USE ini
      TYPE (fock_handle) :: FH
      INTEGER LMDIM, NBANDSGW

!     ALLOCATE(FH)

      FH%LMDIM=LMDIM
      ALLOCATE( &
           FH%CXI(GRID_FOCK%MPLWV*WDES_FOCK%NRSPINORS, NBANDSGW, 2), &
           FH%CKAPPA(WDES_FOCK%NPROD, NBANDSGW, 2),  &
           FH%CDIJ(LMDIM,LMDIM,WDES_FOCK%NIONS,WDES_FOCK%NRSPINORS), &
           FH%CDLM(AUG_DES%NPROD*WDES_FOCK%NRSPINORS))

      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CXI), "fockhandle")
#ifdef gammareal
      CALL REGISTER_ALLOCATE(8._q*SIZE( FH%CKAPPA), "fockhandle")
      CALL REGISTER_ALLOCATE(8._q*SIZE( FH%CDIJ), "fockhandle")
      CALL REGISTER_ALLOCATE(8._q*SIZE( FH%CDLM), "fockhandle")
#else
      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CKAPPA), "fockhandle")
      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CDIJ), "fockhandle")
      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CDLM), "fockhandle")
#endif
      
      CALL SETWDES(WDES_FOCK, FH%WDESK, 0)

      CALL NEWWAV(FH%WQ, FH%WDESK, .TRUE.)
      FH%W1%CPROJ=>FH%CDLM(:)

      CALL CLEAR_FOCK_HANDLE( FH)

    END SUBROUTINE ALLOCATE_FOCK_HANDLE


    SUBROUTINE ALLOCATE_FOCK_HANDLE_NOACC( FH , LMDIM)
      USE ini
      TYPE (fock_handle) :: FH
      INTEGER LMDIM, NBANDSGW

      FH%LMDIM=LMDIM
      ALLOCATE( &
           FH%CDIJ(LMDIM,LMDIM,WDES_FOCK%NIONS,WDES_FOCK%NRSPINORS), &
           FH%CDLM(AUG_DES%NPROD*WDES_FOCK%NRSPINORS))

#ifdef gammareal
      CALL REGISTER_ALLOCATE(8._q*SIZE( FH%CDIJ), "fockhandle")
      CALL REGISTER_ALLOCATE(8._q*SIZE( FH%CDLM), "fockhandle")
#else
      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CDIJ), "fockhandle")
      CALL REGISTER_ALLOCATE(16._q*SIZE( FH%CDLM), "fockhandle")
#endif
      
      CALL SETWDES(WDES_FOCK, FH%WDESK, 0)

      CALL NEWWAV(FH%WQ, FH%WDESK, .TRUE.)
      FH%W1%CPROJ=>FH%CDLM(:)

      FH%CDIJ=0; FH%CDLM=0

    END SUBROUTINE ALLOCATE_FOCK_HANDLE_NOACC


!***********************************************************************
! 
!
!***********************************************************************

    SUBROUTINE DEALLOCATE_FOCK_HANDLE( FH)
      USE ini
      TYPE (fock_handle) :: FH
      TYPE (wavespin) W
      INTEGER LMDIM

      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CXI), "fockhandle")
#ifdef gammareal
      CALL DEREGISTER_ALLOCATE(8._q*SIZE( FH%CKAPPA), "fockhandle")
      CALL DEREGISTER_ALLOCATE(8._q*SIZE( FH%CDIJ), "fockhandle")
      CALL DEREGISTER_ALLOCATE(8._q*SIZE( FH%CDLM), "fockhandle")
#else
      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CKAPPA), "fockhandle")
      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CDIJ), "fockhandle")
      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CDLM), "fockhandle")
#endif

      DEALLOCATE(FH%CXI, FH%CKAPPA, FH%CDIJ, FH%CDLM)

      NULLIFY(FH%W1%CPROJ)
      CALL DELWAV(FH%WQ, .TRUE.)


!     DEALLOCATE(FH)
      
    END SUBROUTINE DEALLOCATE_FOCK_HANDLE


    SUBROUTINE DEALLOCATE_FOCK_HANDLE_NOACC( FH)
      USE ini
      TYPE (fock_handle) :: FH
      TYPE (wavespin) W
      INTEGER LMDIM

#ifdef gammareal
      CALL DEREGISTER_ALLOCATE(8._q*SIZE( FH%CDIJ), "fockhandle")
      CALL DEREGISTER_ALLOCATE(8._q*SIZE( FH%CDLM), "fockhandle")
#else
      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CDIJ), "fockhandle")
      CALL DEREGISTER_ALLOCATE(16._q*SIZE( FH%CDLM), "fockhandle")
#endif

      DEALLOCATE(FH%CDIJ, FH%CDLM)

      NULLIFY(FH%W1%CPROJ)
      CALL DELWAV(FH%WQ, .TRUE.)

    END SUBROUTINE DEALLOCATE_FOCK_HANDLE_NOACC


    SUBROUTINE CLEAR_FOCK_HANDLE( FH )
      TYPE (fock_handle) :: FH

      FH%CXI   =0
      FH%CKAPPA=0
      FH%CDIJ  =0
      FH%CDLM  =0  
    END SUBROUTINE CLEAR_FOCK_HANDLE


!***********************************************************************
!
! Determine the action of the Hamiltonian on all bands and
! store the result in a set of wavefunctions WXI
! 
!***********************************************************************


  SUBROUTINE FOCK_ACC_ALL(GRID, LATT_CUR, NONLR_S, NONL_S, W, WXI, &
     &    LMDIM, LKINETIC, P, CQIJ, EXHF)
      USE prec
      USE wave_mpi
      USE wave
      USE lattice
      USE mpimy
      USE mgrid
      
      USE nonl_high
      USE pseudo

      IMPLICIT NONE

      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (potcar)      P(:)
      TYPE (wavespin), TARGET :: W
      TYPE (wavespin)    WXI
      INTEGER LMDIM
      LOGICAL LKINETIC
      INTEGER IU0
      OVERLAP            CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      REAL(q) EXHF
!    local
      TYPE (wavedes), POINTER :: WDES
      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      TYPE (wavefun1)    W1             ! current wavefunction
      COMPLEX(q) CDCHF
      INTEGER ISP, NK, N, M, MM, NSTRIP, NSTRIP_ACT, NPOS, ISPINOR

      WDES=>W%WDES
      NSTRIP=((W%WDES%NSIM*2+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)

      WXI%CPROJ=0
      WXI%CW=0

      CDCHF=0

      spin:  DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
         WXI%CW(:,:,NK,ISP)=0   ! Zero accelerations
      ELSE
#endif
         IF (NONLR_S%LREAL) THEN
            CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
         ELSE
            CALL PHASE(WDES,NONL_S,NK)
         ENDIF

         CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)
         DO NPOS=1,WDES%NBANDS,NSTRIP
            NSTRIP_ACT=MIN(WDES%NBANDS+1-NPOS,NSTRIP)
            IF (AEXX/=0) THEN
               CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W,  &
                 NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                 WXI%CW(:,NPOS:,NK,ISP), P, CQIJ, CDCHF )
            ELSE
               WXI%CW(:,NPOS:NPOS+NSTRIP_ACT-1,NK,ISP)=0
            ENDIF
         ENDDO
         IF (LKINETIC) THEN
         DO NPOS=1,WDES%NBANDS
            DO ISPINOR=0,WDES1%NRSPINORS-1
               DO M=1,WDES1%NGVECTOR
                  MM=M+ISPINOR*WDES1%NGVECTOR
                  WXI%CW(MM,NPOS,NK,ISP)=WXI%CW(MM,NPOS,NK,ISP)+W%CW(MM,NPOS,NK,ISP)*WDES1%DATAKE(M,ISPINOR+1)
               ENDDO
            ENDDO
         ENDDO
         ENDIF
#ifdef MPI
      ENDIF
#endif
      END DO kpoint
      END DO spin

      CALLMPI( M_sum_z(WDES1%COMM_KIN,CDCHF,1))
      CALLMPI( M_sum_z(WDES1%COMM_KINTER,CDCHF,1))
      EXHF=CDCHF
      RETURN
    END SUBROUTINE FOCK_ACC_ALL


    SUBROUTINE FOCK_ALL( LATT_CUR, NONLR_S, NONL_S, W, &
     &    LMDIM, P, CQIJ, EXHF)
      USE prec
      USE wave_mpi
      USE wave
      USE lattice
      USE mpimy
      USE mgrid
      
      USE nonl_high
      USE pseudo

      IMPLICIT NONE

      TYPE (latt)        LATT_CUR
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (potcar)      P(:)
      TYPE (wavespin), TARGET :: W
      INTEGER LMDIM
      LOGICAL LKINETIC
      INTEGER IU0
      OVERLAP            CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      REAL(q) EXHF
!    local
      TYPE (wavedes), POINTER :: WDES
      TYPE (wavedes1) WDES1          ! descriptor for one k-point
      COMPLEX(q) CDCHF
      INTEGER ISP, NK, NSTRIP, NSTRIP_ACT, NPOS, ISPINOR
      TYPE (wavefuna) WXI 
 
      WDES=>W%WDES
      NSTRIP=((W%WDES%NSIM*2+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)
  

      CALL SETWDES(WDES,WDES1,0)
      CALL NEWWAVA(WXI, WDES1, NSTRIP)
 
      CDCHF=0
      
      spin:  DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).EQ.WDES%COMM_KINTER%NODE_ME-1) THEN
#endif

         WXI%CW(:,:)=0      ! Zero accelerations
         IF (NONLR_S%LREAL) THEN
            CALL PHASER(W%WDES%GRID,LATT_CUR,NONLR_S,NK,WDES)
         ELSE
            CALL PHASE(WDES,NONL_S,NK)
         ENDIF

         CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,W%WDES%GRID)
         DO NPOS=1,WDES%NBANDS,NSTRIP
            NSTRIP_ACT=MIN(WDES%NBANDS+1-NPOS,NSTRIP)
            IF (AEXX/=0) THEN
               CALL FOCK_ACC(W%WDES%GRID, LMDIM, LATT_CUR, W,  &
                 NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                 WXI%CW, P, CQIJ, CDCHF )
            ENDIF
         ENDDO
#ifdef MPI
      ENDIF
#endif
      END DO kpoint
      END DO spin

      CALLMPI( M_sum_z(WDES1%COMM_KIN,CDCHF,1))
      CALLMPI( M_sum_z(WDES1%COMM_KINTER,CDCHF,1))
      EXHF=CDCHF
      RETURN
    END SUBROUTINE FOCK_ALL



!************************ SUBROUTINE FOCK_FORCE ************************
!
! caculate the Hellmann-Feynman forces acting on the ions
! this routine follows closely the previous FOCK_ACC routine
! written by gK
!
!***********************************************************************

  SUBROUTINE FOCK_FORCE(GRID_, LATT_CUR, W, LMDIM, CQIJ, &
       NONLR_S, NONL_S, P, FORHF, SIFHF, LSIF , CDIJ0, SV, IU0, IU6)
    USE sym_prec
    USE nonl_high
    USE wave
    USE wave_high
    USE mpimy
    USE mgrid
    USE lattice
    USE constant
    USE pseudo
    USE full_kpoints
    USE paw
!   USE openmp, ONLY : omp_nthreads_hi
    IMPLICIT NONE

! passed variables
    TYPE (grid_3d)  GRID_                          ! not used (scheduled for removal)
    TYPE (latt)     LATT_CUR
    TYPE (wavespin) W
    INTEGER LMDIM
    OVERLAP              CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
!test
    OVERLAP              CDIJ0 (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    RGRID   SV(DIMREAL(W%WDES%GRID%MPLWV),W%WDES%NRSPINORS*W%WDES%NRSPINORS) ! local potential
    INTEGER IU0, IU6
!endtest
    TYPE (nonlr_struct)  NONLR_S
    TYPE (nonl_struct)   NONL_S
    TYPE (potcar)        P(NONLR_S%NTYP)
    GDEF,ALLOCATABLE ::  GWORK(:) ! fock pot in real sp
    REAL(q) FORHF(3,NONLR_S%NIONS)
    REAL(q) SIFHF(3,3)
    LOGICAL LSIF
! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1) :: W1, WQ
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    REAL(q) :: WEIGHT
    REAL(q),ALLOCATABLE :: FSG(:)
    INTEGER NK, ISP, NPOS
    INTEGER ISPINOR
    INTEGER N, N_, MQ, MM, NP, NGLB, NGLBN, NSTRIP, IDIR, ISP_IRZ
    INTEGER NQ
    INTEGER IERR
    LOGICAL LSHIFT
    GDEF,      ALLOCATABLE :: CPROJKXYZ(:,:,:) ! stores ionic derivative of wave character of curr. strip
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:,:)  ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:,:)      ! 1/(G+dk)**2 (G)

    REAL(q)    :: DISPL1(3,NONLR_S%NIONS),DISPL2(3,NONLR_S%NIONS), DIS, EVALUE, ENL(NONLR_S%NIONS)
    TYPE (nonlr_struct), ALLOCATABLE :: FAST_AUG(:)
    GDEF, ALLOCATABLE :: CPROJXYZ(:,:,:)

    INTEGER :: NT, NIS, NI,  LMMAXC, LBASE, L, LP, NIP, ISTAT, NB_TOT
    INTEGER ierror
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF

    INTEGER NDIR, I, J
    TYPE (latt)     LATT_FIN1, LATT_FIN2
    REAL(q)         SIF(0:6),SIF2(0:6)
    REAL(q)    :: WEIGHT_Q

    PROFILING_START('fock_force')

    IF (MODEL_GW>0 .OR. AEXX==0) THEN
       FORHF=0
       SIFHF=0
       RETURN
    ENDIF

    NULLIFY(ROT_HANDLE)
!==========================================================================
! some MPI variable initialisation
! and other initialisations
!==========================================================================
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK
    CALL CHECK_FULL_KPOINTS
    IF (.NOT. W%WDES%LOVERL .AND. .NOT. LSIF) THEN
       PROFILING_STOP('fock_force')
       RETURN
    ENDIF

    IF (LSIF) THEN
       NDIR=9
    ELSE
       NDIR=3
    ENDIF

    ! MPI dividable by NB_PAR
    NSTRIP=((WHF%WDES%NSIM*2+WHF%WDES%NB_PAR-1)/WHF%WDES%NB_PAR)
    NGLB=NSTRIP*WHF%WDES%NB_PAR
    IF (NBLOCK_FOCK>0) NGLB=MIN(NGLB,NBLOCK_FOCK)

    ! only occupied bands need to be included in force evaluation
    NB_TOT=0
    DO ISP=1,WHF%WDES%ISPIN
    DO NK=1,WHF%WDES%NKPTS
    DO NPOS=1,WHF%WDES%NB_TOT
       IF (ABS(W%FERTOT(NPOS, NK, ISP))>1E-10) NB_TOT=MAX(NB_TOT,NPOS)
    ENDDO
    ENDDO
    ENDDO
    NB_TOT=((NB_TOT+WHF%WDES%NB_PAR-1)/WHF%WDES%NB_PAR)*WHF%WDES%NB_PAR

    IF (WHF%WDES%LOVERL) THEN
       ! allocate all required structures
       ALLOCATE(FAST_AUG(NDIR),CPROJXYZ(WHF%WDES%NPROD, WHF%WDES%NBANDS, NDIR), &
            CPROJKXYZ(WHF%WDES%NPROD,NGLB,NDIR), &
            CRHOLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS), &
            CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS,0:NDIR), &
            CDLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS),STAT=ISTAT)

       ! problem with allocate return gracefully
       ! so that WAVECAR can be written (hopefully)
       IF (ISTAT/=0) THEN
          CALL VTUTOR('W','FOCKFORCE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
          CALL VTUTOR('W','FOCKFORCE',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
          PROFILING_STOP('fock_force')
          RETURN
       ENDIF
!==========================================================================
! setup first derivative of augmentation charges with respect
! to ionic positions
!==========================================================================
       DIS=fd_displacement

       DO IDIR=1,3
          CALL COPY_FASTAUG(FAST_AUG_FOCK, FAST_AUG(IDIR))
          DISPL1=0
          DISPL1(IDIR,:)=-DIS
          
          DISPL2=0
          DISPL2(IDIR,:)=+DIS
          CALL RSPHER_ALL(GRIDHF, FAST_AUG(IDIR),LATT_CUR,LATT_CUR,LATT_CUR, DISPL1, DISPL2,1)
          ! phase factor identical to FAST_AUG_FOCK
          IF (ASSOCIATED(FAST_AUG(IDIR)%CRREXP)) DEALLOCATE(FAST_AUG(IDIR)%CRREXP)
          FAST_AUG(IDIR)%CRREXP=>FAST_AUG_FOCK%CRREXP
          FAST_AUG(IDIR)%RPROJ=FAST_AUG(IDIR)%RPROJ*SQRT(LATT_CUR%OMEGA)*(1._q/(2._q*DIS))
       ENDDO
    ENDIF
!==========================================================================
! setup first derivative of augmentation charges with respect
! to lattice vectors
!==========================================================================
    IF (LSIF .AND. W%WDES%LOVERL) THEN
       IDIR=3
       DO I=1,3
          DO J=1,I
             IDIR=IDIR+1
             LATT_FIN1%A=LATT_CUR%A
             LATT_FIN2%A=LATT_CUR%A
             LATT_FIN1%A(I,:)=LATT_CUR%A(I,:)+DIS*LATT_CUR%A(J,:)
             LATT_FIN2%A(I,:)=LATT_CUR%A(I,:)-DIS*LATT_CUR%A(J,:)

             CALL LATTIC(LATT_FIN1)
             CALL LATTIC(LATT_FIN2)
             
             CALL COPY_FASTAUG(FAST_AUG_FOCK, FAST_AUG(IDIR))
             DISPL1=0
             CALL RSPHER_ALL(GRIDHF, FAST_AUG(IDIR),LATT_FIN2,LATT_FIN1,LATT_CUR,DISPL1, DISPL1,1,LOMEGA=.TRUE.)
             ! phase factor identical to FAST_AUG_FOCK
             IF (ASSOCIATED(FAST_AUG(IDIR)%CRREXP)) DEALLOCATE(FAST_AUG(IDIR)%CRREXP)
             FAST_AUG(IDIR)%CRREXP=>FAST_AUG_FOCK%CRREXP
             FAST_AUG(IDIR)%RPROJ=FAST_AUG(IDIR)%RPROJ*(1._q/(2._q*DIS))
          ENDDO
       ENDDO
    ENDIF

! allocate memory
    ALLOCATE(POTFAK(GRIDHF%MPLWV,0:NDIR-3),FSG(0:NDIR-3), GWORK(m_ GRIDHF%MPLWV))

    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ , WDESQ,.TRUE.)
    CALL SETWDES(WHF%WDES,WDESK,0)
! we have to do V_HF on NGLB bands
    ALLOCATE(WIN(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(WIN(N) , WDESK,.TRUE.)
    ENDDO

! average electrostatic potential prefactor for k=k' and n=n'
    IF (LSIF) THEN
       CALL SET_FSG_DER(GRIDHF, LATT_CUR, FSG)
    ELSE
       FSG(0)=SET_FSG(GRIDHF, LATT_CUR)
    ENDIF
!==========================================================================
! main loop over spin, k-points, and bands (in blocks of NGLB)
!==========================================================================
    FORHF=0
    SIF=0
    SIF2=0

    spin:   DO ISP=1,WHF%WDES%ISPIN
    kpoint: DO NK=1,WHF%WDES%NKPTS
#ifdef MPI
    IF (MOD(NK-1,WHF%WDES%COMM_KINTER%NCPU).NE.WHF%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
    CALL SETWDES(WHF%WDES,WDESK,NK)

! first derivative of  wavefunction character with respect to all ionic positions
    IF (WHF%WDES%LOVERL) THEN
       IF (NONLR_S%LREAL) THEN
          CALL PHASER(W%WDES%GRID,LATT_CUR,NONLR_S,NK,W%WDES)
          CALL RPROXYZ(W%WDES%GRID, NONLR_S, P, LATT_CUR, W, W%WDES, ISP, NK, CPROJXYZ)
          IF (LSIF) & 
               CALL RPROLAT_DER(W%WDES%GRID, NONLR_S, P, LATT_CUR, W, W%WDES, ISP, NK, CPROJXYZ(1,1,4))
       ELSE
          CALL PHASE(W%WDES,NONL_S,NK)
          CALL PROJXYZ(NONL_S, W%WDES, W, LATT_CUR, ISP, NK, CPROJXYZ)
          IF (LSIF) &
               CALL PROJLAT_DER(P, NONL_S, W%WDES, W, LATT_CUR, ISP, NK, CPROJXYZ(1,1,4))
       ENDIF
    ENDIF

    band:   DO NPOS=1,NB_TOT,NGLB
    NGLBN  =MIN(NB_TOT-NPOS+1,NGLB)
!==========================================================================
! fourier transform the bands to be accelerated to real space (CWRN)
! then distribute the CWRN array to all nodes
!==========================================================================
    DO N=NPOS,NPOS+NGLBN-1
       IF ((MOD(N-1,WHF%WDES%NB_PAR)+1)/=WHF%WDES%NB_LOW) CYCLE
       CALL W1_COPY( ELEMENT( WHF, WDESK, (N-1)/WHF%WDES%NB_PAR+1, ISP),WIN(N-NPOS+1))
       CALL FFTWAV_W1(WIN(N-NPOS+1))
    ENDDO

! if LOVERL copy the projectors into CPROJK array
    IF (WHF%WDES%LOVERL) THEN
       DO IDIR=1,NDIR
          DO N=NPOS,NPOS+NGLBN-1
             IF ((MOD(N-1,WHF%WDES%NB_PAR)+1)/=WHF%WDES%NB_LOW) CYCLE
             CPROJKXYZ(1:WHF%WDES%NPROD,N-NPOS+1,IDIR)=CPROJXYZ(1:WHF%WDES%NPROD,(N-1)/WHF%WDES%NB_PAR+1,IDIR)
          ENDDO
       ENDDO
    ENDIF
! distribute WIN and CPROJKXYZ to all nodes
#ifdef MPI
    IF (WHF%WDES%DO_REDIS) THEN
       DO N=1,NGLBN
          N_=N+NPOS-1           ! global storage index of present band
          CALL M_bcast_z_from(WDESK%COMM_INTER,WIN(N)%CR(1), &
               GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,MOD(N_-1,WHF%WDES%NB_PAR)+1)
          IF (WHF%WDES%LOVERL) THEN
#ifndef gammareal
             CALL M_bcast_z_from(WDESK%COMM_INTER,WIN(N)%CPROJ(1), &
                  WHF%WDES%NPROD,MOD(N_-1,WHF%WDES%NB_PAR)+1)
             DO IDIR=1,NDIR
                CALL M_bcast_z_from(WDESK%COMM_INTER,CPROJKXYZ(:,N,IDIR), &
                  WHF%WDES%NPROD,MOD(N_-1,WHF%WDES%NB_PAR)+1)
             ENDDO
#else
             CALL M_bcast_d_from(WDESK%COMM_INTER,WIN(N)%CPROJ(1), &
                  WHF%WDES%NPROD,MOD(N_-1,WHF%WDES%NB_PAR)+1)
             DO IDIR=1,NDIR
                CALL M_bcast_d_from(WDESK%COMM_INTER,CPROJKXYZ(:,N,IDIR), &
                  WHF%WDES%NPROD,MOD(N_-1,WHF%WDES%NB_PAR)+1)
             ENDDO
#endif
          ENDIF
       ENDDO
    ENDIF
#endif
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!  start literal copy
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
       IF( KPOINTS_FULL%WTKPT(NQ)==0 .OR. &
            (HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(WHF%WDES%VKPT(:,NQ))) .OR. &
            (.NOT.HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK)))) CYCLE

       WEIGHT_Q=1
       IF (ALLOCATED(WEIGHT_K_POINT_PAIR_SMALL_GROUP) .AND. LSYMGRAD ) THEN
          IF (WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)==0) CYCLE
          WEIGHT_Q=WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)
       ENDIF

       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))
       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF

   ! set POTFAK for this q and k point
       IF (LSIF) THEN
          CALL SET_GFAC_DER(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)
       ELSE
          CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG(0),POTFAK(1,0))
       ENDIF

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
!
! cant do it in the outer band loop because then the redistribution gets messed up
!
       mband: DO MQ=1,WHF%WDES%NBANDS
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband
          IF ((MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW<NBANDSGWLOW_FOCK) CYCLE mband

          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.

             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
         ! calculate charge phi_k nk(r) phi*_q nq(r) 
          nband: DO N=1,NGLBN
             N_=N+NPOS-1 ! global storage index of present band

             CALL PW_CHARGE_TRACE(WDESK, GWORK(1), WIN(N)%CR(1), WQ%CR(1))

             ! add augmentation part to charge (if required)
             IF (WHF%WDES%LOVERL) THEN
                CALL DEPSUM_TWO_BANDS_RHOLM_TRACE(WIN(N)%CPROJ(:),WQ%CPROJ(:), WDESK, AUG_DES,  &
                      TRANS_MATRIX_FOCK, CRHOLM,1._q,WHF%WDES%LOVERL)
                
                AUG_DES%RINPL=1._q ! multiplicator used by RACC0
                CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES,CRHOLM(1), GWORK(1))
             ENDIF
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1),GRIDHF,-1)
      ! derivative of G with respect to cell shape

             WEIGHT=WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WHF%FERTOT(N_,NK,ISP)* &
                  WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)*WEIGHT_Q
             IF (LSIF) THEN
                CALL APPLY_GFAC_DER(GRIDHF, GWORK(1), POTFAK(1,0), SIF(0), WEIGHT )
             ENDIF
      over:  IF (WHF%WDES%LOVERL) THEN
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL APPLY_GFAC(GRIDHF, GWORK(1), POTFAK(1,0))
      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1),GRIDHF,1)
!==========================================================================
! end literal copy from FOCK_ACC
!==========================================================================
             WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)
             AUG_DES%RINPL=WEIGHT/GRIDHF%NPLWV ! multiplicator for RPRO1
#ifdef ForceTest
             GWORK(1:GRID%MPLWV)=SV(1:GRID%MPLWV,1)
             AUG_DES%RINPL=1._q/GRIDHF%NPLWV
#endif
   ! calculate D_LM
             W1%CPROJ => CDLM(:) ! build the descriptor for RPRO1
             CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, W1, GWORK(:))
             IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
   ! transform D_LM -> D_lml'm'
             CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ(:,:,:,:,0),CDLM)
   ! similar: first derivative of \int V_x(r) Q_ij(r) with respect to ionic positions
             DO IDIR=1,NDIR
                CALL RPRO1_HF(FAST_AUG(IDIR),AUG_DES, W1, GWORK(:))
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ(:,:,:,:,IDIR),CDLM)
             ENDDO
             WEIGHT=WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WHF%FERTOT(N_,NK,ISP)*WEIGHT_Q

#ifdef ForceTest
             EVALUE=W%CELTOT(N_,NK,ISP)
             IF ( NQ/=NK.OR.MQ/=N_) THEN
                WEIGHT=0
             ELSE
                WRITE(*,*) NK,N_,EVALUE,WEIGHT
             ENDIF

#endif
             dir: DO IDIR=1,NDIR
             ENL=0

             DO ISPINOR=0,WHF%WDES%NRSPINORS-1

                LBASE =ISPINOR *WHF%WDES%NPRO/2
          
                NIS=1
                typ: DO NT=1,WHF%WDES%NTYP
                   LMMAXC=WHF%WDES%LMMAX(NT)
                   IF (LMMAXC==0) GOTO 510
                   ! derivate with respect to projectors and with respect to augmentation
                   ! d/d lambda \sum_ij <phi |beta_j> <beta_i| phi> int V(r) Q_ij(r) dr
                   !          = \sum_ij 2 Re[<phi |beta_j> d/d lambda <beta_i| phi> int V(r) Q_ij(r) dr ]
                   !           +\sum_ij <phi |beta_j> <beta_i| phi> int V(r) d/d lambda Q_ij(r) dr
                   ! the factor 2 is included in the CPROJKXYZ (see PROXYZ RPORXYZ)
                   ion: DO NI=NIS,WHF%WDES%NITYP(NT)+NIS-1
#ifdef ForceTest
                      DO L=1 ,LMMAXC
                      DO LP=1,LMMAXC
                      !test for derivative of wave function character with positions
                         ENL(NI)=ENL(NI)+WEIGHT*WQ%CPROJ(LBASE+LP)*GCONJG(CPROJKXYZ(LBASE+L,N,IDIR))* &
                              (CDIJ0(LP,L,NI,1)-EVALUE*CQIJ(LP,L,NI,1))
                      !test for derivative with respect to augmentation charges
                         ENL(NI)=ENL(NI)+WEIGHT*WQ%CPROJ(LBASE+LP)*GCONJG(WQ%CPROJ(LBASE+L))* &
                              CDIJ(LP,L,NI,1+ISPINOR,IDIR)
                      ENDDO
                      ENDDO
#else
                      CALL ECCP_NL_FOCK(LMDIM,LMMAXC,CDIJ(1,1,NI,1+ISPINOR,0),CDIJ(1,1,NI,1+ISPINOR,IDIR), &
                            WQ%CPROJ(LBASE+1),CPROJKXYZ(LBASE+1,N,IDIR),WIN(N)%CPROJ(LBASE+1),ENL(NI),WEIGHT)
#endif

                   LBASE = LMMAXC+LBASE
                   ENDDO ion
510                NIS = NIS+WHF%WDES%NITYP(NT)
                ENDDO typ
             ENDDO

             IF (IDIR<=3) THEN
                DO NI=1,WHF%WDES%NIONS
                   NIP=NI_GLOBAL(NI, WHF%WDES%COMM_INB)
                   FORHF(IDIR,NIP)=FORHF(IDIR,NIP)-ENL(NI)
                ENDDO
             ELSE
                DO NI=1,WHF%WDES%NIONS
                   SIF2(IDIR-3)=SIF2(IDIR-3)+ENL(NI)
                ENDDO
             ENDIF
          ENDDO dir

          ENDIF over

          ENDDO nband
       ENDDO mband
    ENDDO qpoints

    ENDDO band

    ENDDO kpoint
    ENDDO spin
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER, FORHF(1,1),NONLR_S%NIONS*3))
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER, SIF, 7))
    CALLMPI( M_sum_d(WHF%WDES%COMM_KINTER, SIF2, 7))

    CALLMPI( M_sum_d(WDESK%COMM_KIN, FORHF(1,1),NONLR_S%NIONS*3))
    CALLMPI( M_sum_d(WDESK%COMM_KIN, SIF, 7))
    CALLMPI( M_sum_d(WDESK%COMM_KIN, SIF2, 7))
    
#ifdef ForceTest
    WRITE(*,*)'non local force in HF routine'
    WRITE(*,'(3F14.7)') FORHF
    WRITE(*,'(6F14.7)') SIF(1:6)
    WRITE(*,'(6F14.7)') SIF2(1:6)
#endif
    SIF=SIF2+SIF
    ! implemented is the negative exchange
    FORHF=-FORHF
    IDIR=0
    DO I=1,3
       DO J=1,I
          IDIR=IDIR+1
          SIFHF(I,J)=SIF(IDIR)
          SIFHF(J,I)=SIF(IDIR)
       ENDDO
    ENDDO

    IF (WHF%WDES%LOVERL) THEN
       DO IDIR=1,NDIR
          NULLIFY(FAST_AUG(IDIR)%CRREXP)
          CALL NONLR_DEALLOC(FAST_AUG(IDIR))
       ENDDO
       DEALLOCATE(FAST_AUG,CPROJXYZ,CPROJKXYZ,CRHOLM,CDIJ,CDLM)
    ENDIF
    DEALLOCATE(POTFAK,FSG)
    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN)

    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

    PROFILING_STOP('fock_force')


  END SUBROUTINE FOCK_FORCE


!************************ SUBROUTINE FOCK_CHARGE ***********************
! 
! calculate the total charge density (i.e. plane wave +
! fast augmentation) from two orbitals
!   w1 * w2^*
!
!**********************************************************************

  SUBROUTINE FOCK_CHARGE( W1, W2, GCHG, CRHOLM)
    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(:)           ! accumulated charged
    GDEF ::  CRHOLM(:)         ! temporary

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_TRACE(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL)
      
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF
  END SUBROUTINE FOCK_CHARGE

!
! blocked version working on a set of input orbitals W1
!

  SUBROUTINE FOCK_CHARGE_MU( W1, W2, GCHG, CRHOLM)
    USE openmp, ONLY : omp_nthreads_hi
    TYPE (wavefun1) :: W1(:), W2  ! two wavefunctions
    GDEF ::  GCHG(:,:)            ! accumulated charged
    GDEF ::  CRHOLM(:,:)          ! temporary
    INTEGER N,NSTRIP,NSTRIP_ACT


    PROFILING_START('fock_charge_mu')

    DO N=1,SIZE(W1)
    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE(W1(N)%WDES1, GCHG(1,N), W1(N)%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1(N)%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_TRACE(W1(N)%CPROJ(:),W2%CPROJ(:), W1(N)%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM(:,N), 1._q, W1(N)%WDES1%LOVERL)
    ENDIF
    ENDDO

    IF (W1(1)%WDES1%LOVERL) THEN
       AUG_DES%RINPL=1._q       ! multiplicator used by RACC0
       IF (SIZE(CRHOLM,1) /= AUG_DES%NPROD*W1(1)%WDES1%NRSPINORS) THEN
          WRITE(0,*) 'internal error in VASP: FOCK_CHARGE_MU size mismatch', SIZE(CRHOLM,1), AUG_DES%NPROD
          STOP
       ENDIF
       NSTRIP=SIZE(W1)
       DO N=1,SIZE(W1),NSTRIP
          NSTRIP_ACT=MIN(SIZE(W1)-N+1,NSTRIP)
          CALL RACC0MU_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1,N), SIZE(CRHOLM,1), GCHG(1,N), SIZE(GCHG,1), NSTRIP_ACT)
       ENDDO
    ENDIF

    PROFILING_STOP('fock_charge_mu')

  END SUBROUTINE FOCK_CHARGE_MU


  ! only augmentation part
  SUBROUTINE FOCK_CHARGE_AUG( W1, W2, GCHG, CRHOLM)
    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(:)           ! accumulated charged
    GDEF ::  CRHOLM(:)         ! temporary

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_TRACE(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL)
       
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF
  END SUBROUTINE FOCK_CHARGE_AUG

! conjugated version
! calculate the total charge density (i.e. plane wave +
! fast augmentation) from two wavefunctions without conjugation of the second wavefunction
!   w1 * w2

  SUBROUTINE FOCK_CHARGE_NO_CONJG( W1, W2, GCHG, CRHOLM)
    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(:)           ! accumulated charged
    GDEF ::  CRHOLM(:)         ! temporary

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE_NO_CONJG(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_NO_CONJG(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL)
       
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF
  END SUBROUTINE FOCK_CHARGE_NO_CONJG


!************************ SUBROUTINE FOCK_ACC  *************************
!
! This is the main subroutine for calculating the Fock exchange 
! operator accelerations  F | psi_n> for a stripe of bands from
! NPOS to NPOS+NSTRIP at the k-points NK and for spin ISP
! the wavefunction onto which the action should be evaluated is
! either taken from the original array W or from WIN_ORIG
! is supplied 
! the convergence corrections are only correct if WIN_ORIG
! 
!**********************************************************************

!
! currently two version are available
! the first one blocks over pairs of states 
! the second does not block over pairs of states
! for many augmentations channels the blocked version is usually fast

#ifndef non_blocked
!
! blocked version
!
  SUBROUTINE FOCK_ACC(GRID_, LMDIM, LATT_CUR, W, &
       NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIPN, &
       CH, P, CQIJ, CDCHF, WIN_ORIG, LSYMGRAD, EXHF_ACFDT)
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE pseudo
    USE openmp, ONLY : omp_nthreads_hi
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (grid_3d) GRID_                          ! not used (scheduled for removal)
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct)  NONL_S
    TYPE (potcar)       P(:)
    INTEGER NK,ISP,NPOS,NSTRIPN
    COMPLEX(q)       :: CH(:,:)                   ! accelerations in rec. space
    OVERLAP             CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    COMPLEX(q) CDCHF                              ! double counting correction
    TYPE (wavefun1), OPTIONAL :: WIN_ORIG(:)      ! original wavefunctions distributed over bands
    LOGICAL,OPTIONAL :: LSYMGRAD
    REAL(q),OPTIONAL :: EXHF_ACFDT                ! difference between HF exchange, and ACFDT exchange 
                                                  ! compare equation (12) in Harl et al. PRB 81 115126 (2010)
! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1), TARGET :: WQ
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    REAL(q) :: WEIGHT
    REAL(q) :: FSG                                ! singularity correction setting V(G=0) to a finite value
    REAL(q) :: FSG_AEXX                           ! alternative treatment, by adding original orbital
    INTEGER ISPINOR
    INTEGER N, N_, MQ, NP, NGLB, MM, ISP_IRZ
    INTEGER NQ, NQ_USED
    LOGICAL LSHIFT
    GDEF,      ALLOCATABLE :: GWORK(:,:)       ! fock pot in real sp
    COMPLEX(q),ALLOCATABLE :: CXI(:,:)         ! acc. in real space
    GDEF,      ALLOCATABLE :: CKAPPA(:,:)      ! stores NL accelerations
    GDEF,      ALLOCATABLE :: CRHOLM(:,:)      ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:)    ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:,:)        ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! 1/(G+dk)**2 (G)
    REAL(q)                :: EXX,FD
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF
    COMPLEX(q) :: CWORK(W%WDES%GRID%MPLWV*W%WDES%NRSPINORS)
    INTEGER, SAVE :: NWARN
    REAL(q) :: WEIGHT_Q
    REAL(q) :: EXHF
    INTEGER IBLK,NBLK,NBSTART,NBSTOP


    PROFILING_START('fock_acc')

    IF (AEXX==0) THEN
       CH=0
       PROFILING_STOP('fock_acc')
       RETURN
    ENDIF
#ifdef MPI
    IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1 .AND. NWARN<10) THEN
       NWARN=NWARN+1
       WRITE(*,*) 'fock contribution at unconventional k-point'
    END IF
#endif
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    CALL CHECK_FULL_KPOINTS
    NULLIFY(ROT_HANDLE)
! allocate memory, we have to do the action of the Hamiltonian acceleration on nstripn bands
    NBLK=NSTRIPN*W%WDES%NB_PAR
    IF (NBLOCK_FOCK>0) NBLK=MIN(NBLK,NBLOCK_FOCK)

    ALLOCATE( &
         GWORK(m_ GRIDHF%MPLWV,NBLK), &
         CXI(GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,NBLK), &
         CKAPPA(WHF%WDES%NPROD,NBLK), &
         CRHOLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS,NBLK), &
         W1(NBLK), &
         CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS), &
         CDLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS,NBLK), &
         POTFAK(GRIDHF%MPLWV))


    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ, WDESQ, .TRUE.)

    CALL SETWDES(WHF%WDES,WDESK,NK)
    ALLOCATE(WIN(NBLK))
    DO N=1,NBLK
       CALL NEWWAV(WIN(N) , WDESK, .TRUE.)
    ENDDO

! average electrostatic potential for k=k' and n=n'
    FSG=FSG_STORE(NK)
 
    IF (MCALPHA==0) THEN
       IF (L_MODEL_HF) THEN
          FSG_AEXX=FSG
       ELSE
          FSG_AEXX=FSG*AEXX
       ENDIF
! there are two in principle equivalent ways to perform
! the convergence correction
! either FSG_AEXX or FSG must be set
       IF (PRESENT(WIN_ORIG)) THEN
! numerically slightly less accurate version
! applies to any orbital
          FSG_AEXX=0
       ELSE
! this one is more accurate but works only for wavefunctions in the 
! set of wavefunctions spanned by W
          FSG=0
       ENDIF
    ELSE
! k=0 term in potfak must be free of other kinds of finite size errors
       FSG=0
       FSG_AEXX=0
    ENDIF
!==========================================================================
! initialise variables
!==========================================================================
    CH=0; EXHF=0
    block: DO IBLK=1,NSTRIPN*W%WDES%NB_PAR,NBLK

    NGLB=MIN(NSTRIPN*W%WDES%NB_PAR-IBLK+1,NBLK)
    ! global index of the first and last band in this block
    NBSTART=(NPOS-1)*W%WDES%NB_PAR+IBLK
    NBSTOP=NBSTART+NGLB-1

    CDIJ=0; CDLM=0; CXI=0; CKAPPA=0
!==========================================================================
! fourier transform the bands for which the HF exchange need to be calculated
! to real space, then gather to WIN
!==========================================================================
    IF (PRESENT(WIN_ORIG)) THEN
       CALL W1_GATHER_STRIP( WHF, IBLK, IBLK+NGLB-1, WIN_ORIG, WIN)
    ELSE
       CALL W1_GATHER_GLB( WHF, NBSTART, NBSTOP, ISP, WIN)
    ENDIF
    NQ_USED=0
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
      IF( KPOINTS_FULL%WTKPT(NQ)==0 .OR. &
         (HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(WHF%WDES%VKPT(:,NQ))) .OR. &
         (.NOT.HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK)))) CYCLE
       NQ_USED=NQ_USED+1
       WEIGHT_Q=1
       IF (ALLOCATED(WEIGHT_K_POINT_PAIR_SMALL_GROUP) .AND. PRESENT(LSYMGRAD) ) THEN
          IF (LSYMGRAD) THEN
             IF (WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)==0) CYCLE
             WEIGHT_Q=WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)
          ENDIF
       ENDIF
          
       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF

   ! set POTFAK for this q and k point
       CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,WHF%WDES%NBANDS
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband
          IF ((MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW<NBANDSGWLOW_FOCK) CYCLE mband


          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
      ! calculate charge phi_q nq(r) phi_k nk(r)
          CALL FOCK_CHARGE_MU( WIN(1:NGLB), WQ, GWORK, CRHOLM)
          WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)/GRIDHF%NPLWV*WEIGHT_Q
          nband: DO N=1,NGLB
             N_=NBSTART+N-1        ! global storage index of present band
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1,N),GRIDHF,-1)
      ! model-GW set GFAC state dependent
             IF (MODEL_GW==2) THEN
                CALL MODEL_GW_SET_GFAC(GRIDHF, LATT_CUR, NK, NQ, KPOINTS_FULL%NEQUIV(NQ), &
                   N_, (MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW, ISP, ISP_IRZ, FSG, POTFAK)
             ENDIF

             IF (MCALPHA/=0) THEN
                ! with finite size corrections:
                CALL APPLY_GFAC_MULTIPOLE(GRIDHF, GWORK(1,N), POTFAK(1))
             ELSE
                CALL APPLY_GFAC_EXCHANGE(GRIDHF, GWORK(1,N), POTFAK(1), EXX)
                IF (PRESENT(EXHF_ACFDT)) THEN
                   EXX=EXX*(0.5_q/GRIDHF%NPLWV)  ! divide by grid points
                   ! correct for self-interaction 
                   IF (N_==(MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW .AND. NQ==NK) THEN
#ifdef MPI
                      IF (W%WDES%COMM_INB%NODE_ME==1) THEN
                         EXX=EXX+FSG_AEXX*0.5_q  ! one node adds corrections
                      ENDIF
#else
                      EXX=EXX+FSG_AEXX*0.5_q
#endif
                   ENDIF
                   ! use smaller occupancy
                   ! FD=MIN(WHF%FERTOT(N_,NK,ISP),WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))
                   ! use one-electron occupancy of state at greater energy
                   IF (REAL(WHF%CELTOT(N_,NK,ISP),q) > REAL(WHF%CELEN(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ),q)) THEN
                      FD=WHF%FERTOT(N_,NK,ISP)
                   ELSE
                      FD=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)
                   ENDIF

                   EXHF=EXHF-EXX &
                         *WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WEIGHT_Q* &
                        (FD- & 
                         WHF%FERTOT(N_,NK,ISP)*WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))
                ENDIF
             ENDIF

      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1,N),GRIDHF,1)

      ! add to acceleration xi in real space
             CALL VHAMIL_TRACE(WDESK, GRID_FOCK, GWORK(1,N), WQ%CR(1), CXI(1,N), WEIGHT)
          ENDDO nband
          IF (WHF%WDES%LOVERL) THEN
             ! calculate D_LM
             ! build the descriptor for RPRO1
             DO N=1,NGLB
                W1(N)%CPROJ => CDLM(:,N)
             ENDDO
             AUG_DES%RINPL=WEIGHT ! multiplicator for RPRO1
             CALL RPROMU_HF(FAST_AUG_FOCK, AUG_DES, W1, NGLB, GWORK(1,1), SIZE(GWORK,1))
             DO N=1,NGLB
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2, N)=CDLM(1:AUG_DES%NPRO, N)
      ! transform D_LM -> D_lml'm'
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ, CDLM(:,N))
      ! add D_lml'm' to kappa_lm_N (sum over l'm')
                CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1), WQ%CPROJ(1), CKAPPA(1,N),.TRUE.)
             ENDDO
          ENDIF
       ENDDO mband
    ENDDO qpoints
    IF (((ODDONLY.OR. EVENONLY) .AND. NQ_USED*2 /=KPOINTS_FULL%NKPTS_NON_ZERO) .OR. &
        (.NOT. (ODDONLY.OR. EVENONLY) .AND. NQ_USED*NKREDX*NKREDY*NKREDZ /=KPOINTS_FULL%NKPTS_NON_ZERO)) THEN
       WRITE(0,*) 'internal error in FOCK_ACC: number of k-points incorrect',NQ_USED,KPOINTS_FULL%NKPTS_NON_ZERO, & 
            NKREDX, NKREDY, NKREDZ
       STOP
    ENDIF
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
! collect CXI and CKAPPA
    IF (WHF%WDES%DO_REDIS) THEN
       PROFILING_START('m_sum')
       CALLMPI( M_sum_z(WDESK%COMM_INTER,CXI(1,1),GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS*NGLB))
       CALLMPI( M_sum_g(WDESK%COMM_INTER,CKAPPA(1,1),WHF%WDES%NPROD*NGLB))
       PROFILING_STOP('m_sum')
    END IF

! generate the descriptor for the original WDES
! only difference to WDESK is the FFT mesh (GRID and not GRIDHF)
    CALL SETWDES(W%WDES,WDESQ,NK)

! fourier transform local accelerations xi (only own bands)
    fft_back:DO N=1,NGLB
       IF (MOD(IBLK+N-1-1,W%WDES%NB_PAR)+1/=W%WDES%NB_LOW) CYCLE fft_back
       N_=(IBLK+N-1-1)/W%WDES%NB_PAR+1


! add CKAPPA to CXI (full acceleration on band N now in CXI)
       IF (WHF%WDES%LOVERL) THEN
          IF (.NOT.PRESENT(WIN_ORIG)) THEN
             ! convergence correction non local part
             WQ%CPROJ(:)=FSG_AEXX*WHF%CPROJ(:,N_+NPOS-1,NK,ISP)*WHF%FERWE(N_+NPOS-1,NK,ISP)
             CALL OVERL1(WDESK, LMDIM, CQIJ(1,1,1,1), CQIJ(1,1,1,1), 0.0_q, WQ%CPROJ(1), WIN(1)%CPROJ(1))
             CKAPPA(:,N)=CKAPPA(:,N)+WIN(1)%CPROJ
          ENDIF

          IF (NONLR_S%LREAL) THEN
             CWORK=0
             CALL RACC0(NONLR_S, WDESQ, CKAPPA(1,N), CWORK(1))
             DO ISPINOR=0,WDESQ%NRSPINORS-1
                CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                     CWORK(1+ISPINOR*WDESQ%GRID%MPLWV), &
                     CH(1+ISPINOR*WDESQ%NGVECTOR,N_),WDESQ%GRID,.TRUE.)
             ENDDO
          ELSE
             CALL VNLAC0(NONL_S, WDESK, CKAPPA(1,N), CH(1,N_))
          ENDIF
       ENDIF
       IF (.NOT.PRESENT(WIN_ORIG)) THEN
       ! convergence correction non local part
          CH(:,N_)=CH(:,N_)+FSG_AEXX*WHF%CW(:,N_+NPOS-1,NK,ISP)*WHF%FERWE(N_+NPOS-1,NK,ISP)
       ENDIF

       ! double counting hence subtract half the self energy
       ! and change sign since we have use e^2 to calculate the potential
       WEIGHT=WHF%FERWE(N_+NPOS-1,NK,ISP)*WHF%WDES%WTKPT(NK)*0.5_q*WHF%WDES%RSPIN

       DO ISPINOR=0,WDESK%NRSPINORS-1
          CALL FFTEXT(WDESK%NGVECTOR,WDESK%NINDPW(1), &
               CXI(1+ISPINOR*GRID_FOCK%MPLWV,N), &
               CH(1+ISPINOR*WDESK%NGVECTOR,N_),GRID_FOCK,.TRUE.)

          DO NP=1,WDESK%NGVECTOR
             MM=NP+ISPINOR*WDESK%NGVECTOR
             CH(MM,N_)=-CH(MM,N_)
             CDCHF=CDCHF-CONJG(CH(MM,N_))*WHF%CW(MM,N_+NPOS-1,NK,ISP)*WEIGHT
            ENDDO
       ENDDO
    ENDDO fft_back

    ENDDO block

    IF (PRESENT(EXHF_ACFDT)) EXHF_ACFDT=EXHF_ACFDT+EXHF


    DEALLOCATE(GWORK,CXI,CKAPPA,CRHOLM,CDIJ,CDLM,W1,POTFAK)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NBLK
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN)

    PROFILING_STOP('fock_acc')

  END SUBROUTINE FOCK_ACC

#else
!
! non blocked version
!
  SUBROUTINE FOCK_ACC(GRID_, LMDIM, LATT_CUR, W, &
       NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIPN, &
       CH, P, CQIJ, CDCHF, WIN_ORIG, LSYMGRAD, EXHF_ACFDT)
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE pseudo
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (grid_3d) GRID_                          ! not used (scheduled for removal)
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct)  NONL_S
    TYPE (potcar)       P(:)
    INTEGER NK,ISP,NPOS,NSTRIPN
    COMPLEX(q)       :: CH(:,:)                   ! accelerations in rec. space
    OVERLAP             CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    COMPLEX(q) CDCHF                              ! double counting correction
    TYPE (wavefun1), OPTIONAL :: WIN_ORIG(:)      ! original wavefunctions distributed over bands
    LOGICAL,OPTIONAL :: LSYMGRAD
    REAL(q),OPTIONAL :: EXHF_ACFDT                ! difference between HF exchange, and ACFDT exchange 
                                                  ! compare equation (12) in Harl et al. PRB 81 115126 (2010)
! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1) :: W1, WQ
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    REAL(q) :: WEIGHT
    REAL(q) :: FSG                                ! singularity correction setting V(G=0) to a finite value
    REAL(q) :: FSG_AEXX                           ! alternative treatment, by adding original orbital
    INTEGER ISPINOR
    INTEGER N, N_, MQ, NP, NGLB, MM, ISP_IRZ
    INTEGER NQ, NQ_USED
    LOGICAL LSHIFT
    GDEF, ALLOCATABLE      :: GWORK(:)         ! fock pot in real sp
    COMPLEX(q),ALLOCATABLE :: CXI(:,:)         ! acc. in real space
    GDEF,      ALLOCATABLE :: CKAPPA(:,:)      ! stores NL accelerations
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:)    ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! 1/(G+dk)**2 (G)
    REAL(q)                :: EXX,FD
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF
    COMPLEX(q) :: CWORK(W%WDES%GRID%MPLWV*W%WDES%NRSPINORS)
    INTEGER, SAVE :: NWARN
    REAL(q)    :: WEIGHT_Q

    IF (AEXX==0) THEN
       CH=0
       RETURN
    ENDIF
#ifdef MPI
    IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1 .AND. NWARN<10) THEN
       NWARN=NWARN+1
       WRITE(*,*) 'fock contribution at unconventional k-point'
    END IF
#endif
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    CALL CHECK_FULL_KPOINTS
    NULLIFY(ROT_HANDLE)
! allocate memory, we have to do the action of the Hamiltonian acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIPN*W%WDES%NB_PAR

    ALLOCATE( &
         GWORK(m_ GRIDHF%MPLWV), &
         CXI(GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,NGLB), &
         CKAPPA(WHF%WDES%NPROD,NGLB),                   &
         CRHOLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS), &
         CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS), &
         CDLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS), &
         POTFAK(GRIDHF%MPLWV))

    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ , WDESQ,.TRUE.)

    CALL SETWDES(WHF%WDES,WDESK,NK)
    ALLOCATE(WIN(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(WIN(N) , WDESK, .TRUE.)
    ENDDO

! average electrostatic potential for k=k' and n=n'
    FSG=FSG_STORE(NK)
 
    IF (MCALPHA==0) THEN
       IF (L_MODEL_HF) THEN
          FSG_AEXX=FSG
       ELSE
          FSG_AEXX=FSG*AEXX
       ENDIF
! there are two in principle equivalent ways to perform
! the convergence correction
! either FSG_AEXX or FSG must be set
       IF (PRESENT(WIN_ORIG)) THEN
! numerically slightly less accurate version
! applies to any orbital
          FSG_AEXX=0
       ELSE
! this one is more accurate but works only for wavefunctions in the 
! set of wavefunctions spanned by W
          FSG=0
       ENDIF
       !     k=0 term in potfak must be free of other kinds of finite size errors
    ELSE
       FSG=0
       FSG_AEXX=0
    ENDIF
!==========================================================================
! initialise variables
!==========================================================================
    CDIJ=0; CDLM=0; CXI=0; CKAPPA=0; CH=0
!==========================================================================
! fourier transform the bands for which the HF exchange need to be calculated
! to real space, then gather to WIN
!==========================================================================
    IF (PRESENT(WIN_ORIG)) THEN
       CALL W1_GATHER_W1( WHF, NSTRIPN, WIN_ORIG, WIN)
    ELSE
       CALL W1_GATHER( WHF, NPOS, NPOS+NSTRIPN-1, ISP, WIN)
    ENDIF
    NQ_USED=0
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
      IF( KPOINTS_FULL%WTKPT(NQ)==0 .OR. &
         (HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(WHF%WDES%VKPT(:,NQ))) .OR. &
         (.NOT.HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK)))) CYCLE
       NQ_USED=NQ_USED+1
       WEIGHT_Q=1
       IF (ALLOCATED(WEIGHT_K_POINT_PAIR_SMALL_GROUP) .AND. PRESENT(LSYMGRAD) ) THEN
          IF (LSYMGRAD) THEN
             IF (WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)==0) CYCLE
             WEIGHT_Q=WEIGHT_K_POINT_PAIR_SMALL_GROUP(NK,NQ)
          ENDIF
       ENDIF
          
       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF

   ! set POTFAK for this q and k point
       CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,WHF%WDES%NBANDS
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband
          IF ((MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW<NBANDSGWLOW_FOCK) CYCLE mband


          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
      ! calculate charge phi_q nq(r) phi_k nk(r)
          nband: DO N=1,NGLB
             CALL FOCK_CHARGE( WIN(N), WQ, GWORK, CRHOLM)
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1),GRIDHF,-1)
      ! model-GW set GFAC state dependent
             IF (MODEL_GW==2) &
                  CALL MODEL_GW_SET_GFAC(GRIDHF, LATT_CUR, NK, NQ, KPOINTS_FULL%NEQUIV(NQ), &
                  (NPOS-1)*W%WDES%NB_PAR+N, (MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW, ISP, ISP_IRZ,FSG, POTFAK)
             IF (MCALPHA/=0) THEN
                ! with finite size corrections:
                CALL APPLY_GFAC_MULTIPOLE(GRIDHF, GWORK(1), POTFAK(1))
             ELSE
                CALL APPLY_GFAC_EXCHANGE(GRIDHF, GWORK(1), POTFAK(1), EXX)
                IF (PRESENT(EXHF_ACFDT)) THEN
                   EXX=EXX*(0.5_q/GRIDHF%NPLWV)  ! divide by grid points
                   N_=(NPOS-1)*WHF%WDES%NB_PAR+N ! global storage index of present band
                   ! correct for self-interaction 
                   IF (N_==(MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW .AND. NQ==NK) THEN
#ifdef MPI
                      IF (W%WDES%COMM_INB%NODE_ME==1) THEN
                         EXX=EXX+FSG_AEXX*0.5_q  ! one node adds corrections
                      ENDIF
#else
                      EXX=EXX+FSG_AEXX*0.5_q
#endif
                   ENDIF
                   ! use smaller occupancy
                   ! FD=MIN(WHF%FERTOT(N_,NK,ISP),WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))
                   ! use one-electron occupancy of state at greater energy
                   IF (REAL(WHF%CELTOT(N_,NK,ISP),q) > REAL(WHF%CELEN(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ),q)) THEN
                      FD=WHF%FERTOT(N_,NK,ISP)
                   ELSE
                      FD=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)
                   ENDIF

                   EXHF_ACFDT=EXHF_ACFDT-EXX &
                         *WHF%WDES%RSPIN*WHF%WDES%WTKPT(NK)*WEIGHT_Q* &
                        (FD- & 
                         WHF%FERTOT(N_,NK,ISP)*WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))
                ENDIF
             ENDIF

      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1),GRIDHF,1)

      ! add to acceleration xi in real space
             WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)/GRIDHF%NPLWV*WEIGHT_Q
             CALL VHAMIL_TRACE(WDESK, GRID_FOCK, GWORK(1), WQ%CR(1), CXI(1,N), WEIGHT)
                
             IF (WHF%WDES%LOVERL) THEN
! add to acceleration kappa
      ! calculate D_LM
                ! build the descriptor for RPRO1
                W1%CPROJ => CDLM(:)
                AUG_DES%RINPL=WEIGHT ! multiplicator for RPRO1
                CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, W1, GWORK)
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
      ! transform D_LM -> D_lml'm'
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ,CDLM)
      ! add D_lml'm' to kappa_lm_N (sum over l'm')
                CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1), WQ%CPROJ(1), CKAPPA(1,N),.TRUE.)
             ENDIF
          ENDDO nband
       ENDDO mband
    ENDDO qpoints
    IF (((ODDONLY.OR. EVENONLY) .AND. NQ_USED*2 /=KPOINTS_FULL%NKPTS_NON_ZERO) .OR. &
        (.NOT. (ODDONLY.OR. EVENONLY) .AND. NQ_USED*NKREDX*NKREDY*NKREDZ /=KPOINTS_FULL%NKPTS_NON_ZERO)) THEN
       WRITE(0,*) 'internal error in FOCK_ACC: number of k-points incorrect',NQ_USED,KPOINTS_FULL%NKPTS_NON_ZERO, & 
            NKREDX, NKREDY, NKREDZ
       STOP
    ENDIF
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
! collect CXI and CKAPPA
    IF (WHF%WDES%DO_REDIS) THEN
       CALLMPI( M_sum_z(WDESK%COMM_INTER,CXI(1,1),GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS*NGLB))
       CALLMPI( M_sum_g(WDESK%COMM_INTER,CKAPPA(1,1),WHF%WDES%NPROD*NGLB))
    END IF

! generate the descriptor for the original WDES
! only difference to WDESK is the FFT mesh (GRID and not GRIDHF)
    CALL SETWDES(W%WDES,WDESQ,NK)

! fourier transform local accelerations xi (only own bands)
    fft_back:DO N=W%WDES%NB_LOW,NGLB,W%WDES%NB_PAR
       N_=(N-1)/W%WDES%NB_PAR+1

! add CKAPPA to CXI (full acceleration on band N now in CXI)
       IF (WHF%WDES%LOVERL) THEN
          IF (.NOT.PRESENT(WIN_ORIG)) THEN
             ! convergence correction non local part
             WQ%CPROJ(:)=FSG_AEXX*WHF%CPROJ(:,N_+NPOS-1,NK,ISP)*WHF%FERWE(N_+NPOS-1,NK,ISP)
             CALL OVERL1(WDESK, LMDIM, CQIJ(1,1,1,1), CQIJ(1,1,1,1), 0.0_q, WQ%CPROJ(1), WIN(1)%CPROJ(1))
             CKAPPA(:,N)=CKAPPA(:,N)+WIN(1)%CPROJ
          ENDIF

          IF (NONLR_S%LREAL) THEN
             CWORK=0
             CALL RACC0(NONLR_S, WDESQ, CKAPPA(1,N), CWORK(1))
             DO ISPINOR=0,WDESQ%NRSPINORS-1
                CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                     CWORK(1+ISPINOR*WDESQ%GRID%MPLWV), &
                     CH(1+ISPINOR*WDESQ%NGVECTOR,N_),WDESQ%GRID,.TRUE.)
             ENDDO
          ELSE
             CALL VNLAC0(NONL_S, WDESK, CKAPPA(1,N), CH(1,N_))
          ENDIF
       ENDIF
       IF (.NOT.PRESENT(WIN_ORIG)) THEN
       ! convergence correction non local part
          CH(:,N_)=CH(:,N_)+FSG_AEXX*WHF%CW(:,N_+NPOS-1,NK,ISP)*WHF%FERWE(N_+NPOS-1,NK,ISP)
       ENDIF

       ! double counting hence subtract half the self energy
       ! and change sign since we have use e^2 to calculate the potential
       WEIGHT=WHF%FERWE(N_+NPOS-1,NK,ISP)*WHF%WDES%WTKPT(NK)*0.5_q*WHF%WDES%RSPIN

       DO ISPINOR=0,WDESK%NRSPINORS-1
          CALL FFTEXT(WDESK%NGVECTOR,WDESK%NINDPW(1), &
               CXI(1+ISPINOR*GRID_FOCK%MPLWV,N), &
               CH(1+ISPINOR*WDESK%NGVECTOR,N_),GRID_FOCK,.TRUE.)

          DO NP=1,WDESK%NGVECTOR
             MM=NP+ISPINOR*WDESK%NGVECTOR
             CH(MM,N_)=-CH(MM,N_)
             CDCHF=CDCHF-CONJG(CH(MM,N_))*WHF%CW(MM,N_+NPOS-1,NK,ISP)*WEIGHT
            ENDDO
       ENDDO
    ENDDO fft_back
    DEALLOCATE(GWORK,CXI,CKAPPA,CRHOLM,CDIJ,CDLM,POTFAK)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN)

  END SUBROUTINE FOCK_ACC
#endif

END MODULE fock

!************************ SUBROUTINE FOCK_CHARGE_NOINT *****************
! 
! calculate the total charge density (i.e. plane wave +
! fast augmentation) from two wavefunctions  w1 x w2^*
! function without explicit interface
! 
!**********************************************************************

  SUBROUTINE FOCK_CHARGE_NOINT( W1, W2, GCHG, CRHOLM, NCRHOLM)
    USE fock
    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(*)           ! accumulated charged
    INTEGER :: NCRHOLM         ! size of CRHOLM
    GDEF ::  CRHOLM(NCRHOLM)   ! work array

    PROFILING_START('fock_charge_noint')

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_TRACE(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL)
       
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF

    PROFILING_STOP('fock_charge_noint')

  END SUBROUTINE FOCK_CHARGE_NOINT

  SUBROUTINE FOCK_CHARGE_NOINT_NOAE( W1, W2, GCHG, CRHOLM, NCRHOLM)
    USE fock

    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(*)           ! accumulated charged
    INTEGER :: NCRHOLM         ! size of CRHOLM
    GDEF ::  CRHOLM(NCRHOLM)   ! work array

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_TRACE_NOAE(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL, FAST_AUG_FOCK%LMAX )
       
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF
  END SUBROUTINE FOCK_CHARGE_NOINT_NOAE

! calculate the total charge density (i.e. plane wave +
! fast augmentation) from two wavefunctions without conjugation of the second wavefunction
!   w1 * w2

  SUBROUTINE FOCK_CHARGE_NO_CONJG_NOINT( W1, W2, GCHG, CRHOLM, NCRHOLM)
    USE fock

    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(*)           ! accumulated charged
    INTEGER :: NCRHOLM         ! size of CRHOLM
    GDEF ::  CRHOLM(NCRHOLM)   ! temporary

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE_NO_CONJG(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_RHOLM_NO_CONJG(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, 1._q, W1%WDES1%LOVERL)
       
       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF
  END SUBROUTINE FOCK_CHARGE_NO_CONJG_NOINT

!************************ FOCK_CHARGE_ONE_CENTER_NOINT ****************
! 
! calculate the total charge density (i.e. plane wave +
! fast augmentation) from two wavefunctions  w1 x w2^*
! also set the one center charge density 
! function without explicit interface 
! 
!**********************************************************************

  SUBROUTINE FOCK_CHARGE_ONE_CENTER_NOINT( W1, W2, GCHG, H, CRHO_ONE_CENTER, CRHOLM, NCRHOLM)
    USE fock
    IMPLICIT NONE

    TYPE (wavefun1) :: W1, W2  ! two wavefunctions
    GDEF ::  GCHG(*)           ! accumulated charged
    TYPE (one_center_handle) :: H
    GDEF ::  CRHO_ONE_CENTER(H%TOTAL_ENTRIES) ! accumulated one center charge
    INTEGER :: NCRHOLM         ! size of CRHOLM
    GDEF ::  CRHOLM(NCRHOLM)   ! work array

    PROFILING_START('fock_charge_one_center_noint')

    ! get the plane wave contribution to the charge
    CALL PW_CHARGE_TRACE(W1%WDES1, GCHG(1), W1%CR(1), W2%CR(1))

    ! add augmentation part to charge (if required)
    IF (W1%WDES1%LOVERL) THEN
       CALL DEPSUM_TWO_BANDS_ONE_CTR_TR(W1%CPROJ(:),W2%CPROJ(:), W1%WDES1, AUG_DES, &
            TRANS_MATRIX_FOCK, CRHOLM, H, CRHO_ONE_CENTER, 1._q, W1%WDES1%LOVERL)

       AUG_DES%RINPL=1._q ! multiplicator used by RACC0
       CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, CRHOLM(1), GCHG(1))
    ENDIF

    PROFILING_STOP('fock_charge_one_center_noint')

  END SUBROUTINE FOCK_CHARGE_ONE_CENTER_NOINT



!************************ SUBROUTINE FOCK_QDER  ***********************
!
! this subroutine calculates the action of the derivative of
! the Fock part of the Hamiltonian acting onto a wavefunction
!
!  d H_Fock / d q psi
!
! it is required for the calculation of optical properties
! presently the derivative with respect to the augmentation charges
! is not implemented, but this contribution is expected to be rather
! small
! 
!**********************************************************************

  SUBROUTINE FOCK_QDER(GRID_, LMDIM, LATT_CUR, W, &
       NONLR_S, NONLR_D, NONL_S, NONL_D, IDIR, NK, ISP, NPOS, NSTRIPN, &
       CH, P, CQIJ)
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE pseudo
    USE fock
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (grid_3d) GRID_                       ! not used (scheduled for removal)
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (nonlr_struct) NONLR_S, NONLR_D       ! second one for derivatives
    TYPE (nonl_struct)  NONL_S, NONL_D         ! second one for derivatives
    TYPE (potcar)      P(:)
    INTEGER :: IDIR                            ! derivative with respect to cart. component IDIR
    INTEGER NK,ISP,NPOS,NSTRIPN
    COMPLEX(q)       :: CH(W%WDES%NRPLWV,NSTRIPN) ! accelerations in rec. space
    OVERLAP             CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1) :: W1, WQ
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    TYPE (wavefun1),ALLOCATABLE :: WTMP(:)
    LOGICAL, ALLOCATABLE :: LDO(:)
    REAL(q) :: WEIGHT
    REAL(q) :: FSG                             ! singularity correction
    INTEGER ISPINOR
    INTEGER N, N_, MQ, NP, NGLB, MM, ISP_IRZ
    INTEGER NQ, NQ_USED
    LOGICAL LSHIFT
    GDEF    :: GWORK (m_ GRIDHF%MPLWV,W%WDES%NRSPINORS) ! fock pot in real sp
    GDEF    :: GWORK2(m_ GRIDHF%MPLWV,W%WDES%NRSPINORS) ! fock pot in real sp
    COMPLEX(q),ALLOCATABLE :: CXI(:,:)         ! acc. in real space
    GDEF,      ALLOCATABLE :: CKAPPA1(:,:)     ! stores NL accelerations
    GDEF,      ALLOCATABLE :: CKAPPA2(:,:)     ! stores NL accelerations
    GDEF,TARGET,ALLOCATABLE:: CPROJD(:,:)      ! derivatives of proj operators with respect to q
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:)    ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! Kernel usually 1/(G+dk)**2
    REAL(q),   ALLOCATABLE :: POTFAKQ(:)       ! derivative of Kernel w.r.t. k
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF
    COMPLEX(q) :: CWORK(W%WDES%GRID%MPLWV*W%WDES%NRSPINORS)

    IF (AEXX==0) THEN
       CH=0
       RETURN
    ENDIF

#ifdef MPI
    IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) THEN
       CH=0
       RETURN
    END IF
#endif
! use temporarily another WDES

    WHF=W
    WHF%WDES => WDES_FOCK

    CALL CHECK_FULL_KPOINTS
    NULLIFY(ROT_HANDLE)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIPN*W%WDES%NB_PAR

    ALLOCATE( &
         CXI(GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,NGLB), &
         CKAPPA1(WHF%WDES%NPROD,NGLB), &
         CKAPPA2(WHF%WDES%NPROD,NGLB), &
         CPROJD (WHF%WDES%NPROD,NGLB), &
         CRHOLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS), &
         CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS), &
         CDLM(AUG_DES%NPROD*WHF%WDES%NRSPINORS), &
         POTFAK(GRIDHF%MPLWV),POTFAKQ(GRIDHF%MPLWV), &
         LDO(NGLB))

    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ , WDESQ,.TRUE.)

    CALL SETWDES(WHF%WDES,WDESK,NK)
    ALLOCATE(WIN(NGLB))
    ALLOCATE(WTMP(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(WIN(N) , WDESK,.TRUE.)
       ! WTMP is identical to WIN, except for CPROJ entry
       ! which will contain the derivative of the projectors w.r.t. k
       WTMP(N)=WIN(N)
       WTMP(N)%CPROJ => CPROJD(:,N)
    ENDDO

! average electrostatic potential for k=k' and n=n'
    FSG=FSG_STORE(NK)
!==========================================================================
! initialise variables
!==========================================================================
    CDIJ=0; CDLM=0; CXI=0; CKAPPA1=0; CKAPPA2=0
!==========================================================================
! fourier transform the bands for which the HF exchange need to be calculated
! to real space, then gather to WIN
!==========================================================================
    CALL W1_GATHER( WHF, NPOS, NPOS+NSTRIPN-1, ISP, WIN)

    ! calculate the derivative of the wave function character
    ! (could be done locally and merged, but this is simpler)
    ! generate the descriptor for the original WDES
    ! only difference to WDESK is the FFT mesh (GRID and not GRIDHF)
    CALL SETWDES(W%WDES,WDESQ,NK)
    
    IF (NONLR_S%LREAL) THEN
       LDO=.TRUE.
       CALL RPROMU(NONLR_D,WDESQ,WTMP, NGLB, LDO)
    ELSE
       DO N=1,NGLB
          CALL PROJ1(NONL_D,WDESQ, WTMP(N))
       ENDDO
    ENDIF

    NQ_USED=0
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
      IF( KPOINTS_FULL%WTKPT(NQ)==0 .OR. &
         (HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(WHF%WDES%VKPT(:,NQ))) .OR. &
         (.NOT.HFKIDENT.AND.SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK)))) CYCLE

       NQ_USED=NQ_USED+1

       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))
       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF

   ! set POTFAK for this q and k point
       CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)
       CALL SET_GFAC_QDER(GRIDHF,LATT_CUR,NK,NQ,0.0_q,POTFAKQ, IDIR)

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,WHF%WDES%NBANDS
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband
          IF ((MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW<NBANDSGWLOW_FOCK) CYCLE mband

          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
          nband: DO N=1,NGLB
! contribution sum_i int Q_ij(r) V(r) d3r <p_i | phi> -> CKAPPA1
! applies only for US PP and PAW
             IF (WHF%WDES%LOVERL) THEN
                CALL FOCK_CHARGE( WIN(N), WQ, GWORK(:,1), CRHOLM)
      ! fft to reciprocal space
                CALL FFT3D(GWORK(1,1),GRIDHF,-1)
                GWORK2=GWORK      ! save for later use
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
                CALL APPLY_GFAC(GRIDHF, GWORK(1,1), POTFAK(1))
      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
                CALL FFT3D(GWORK(1,1),GRIDHF,1)
                WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)/GRIDHF%NPLWV
      ! calculate D_LM
                ! build the descriptor for RPRO1
                W1%CPROJ => CDLM(:)
                AUG_DES%RINPL=WEIGHT ! multiplicator for RPRO1
                CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, W1, GWORK(:,1))
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
      ! transform D_LM -> D_lml'm'
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ,CDLM)
      ! add D_lml'm' to kappa_lm_N (sum over l'm')
                CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1), WQ%CPROJ(1), CKAPPA1(1,N),.TRUE.)
             ELSE
                GWORK2=0
             ENDIF

! contribution int d Q_ij(r)/ d q V(r) d3r <p_i | phi>
! presently only derivatives with respect to the change of the projection
! operators are implemented
! what is lacking is the derivative of Q_ij(r) w.r.t q
             GWORK=0
             CALL FOCK_CHARGE_AUG( WTMP(N), WQ, GWORK(:,1), CRHOLM)
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1,1),GRIDHF,-1)
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL APPLY_GFAC(GRIDHF, GWORK(1,1), POTFAK(1))
      ! add derivative of Kernel with respect to q
             CALL APPLY_GFAC_QDER(GRIDHF, GWORK2(1,1),  GWORK(1,1), POTFAKQ(1))
      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1,1),GRIDHF,1)
      ! add to acceleration xi in real space
             WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)/GRIDHF%NPLWV
             CALL VHAMIL_TRACE(WDESK, GRID_FOCK, GWORK(1,1), WQ%CR(1), CXI(1,N), WEIGHT)
                
             IF (WHF%WDES%LOVERL) THEN
! add to acceleration kappa
      ! calculate D_LM
                ! build the descriptor for RPRO1
                W1%CPROJ => CDLM(:)
                AUG_DES%RINPL=WEIGHT ! multiplicator for RPRO1
                CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, W1, GWORK(:,1))
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
      ! transform D_LM -> D_lml'm'
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ,CDLM)
      ! add D_lml'm' to kappa_lm_N (sum over l'm')
                CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1), WQ%CPROJ(1), CKAPPA2(1,N),.TRUE.)
             ENDIF
          ENDDO nband
       ENDDO mband
    ENDDO qpoints
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
! collect CXI and CKAPPA1 and CKAPPA2
    IF (WHF%WDES%DO_REDIS) THEN
       CALLMPI( M_sum_z(WDESK%COMM_INTER,CXI(1,1),GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS*NGLB))
       CALLMPI( M_sum_g(WDESK%COMM_INTER,CKAPPA1(1,1),WHF%WDES%NPROD*NGLB))
       CALLMPI( M_sum_g(WDESK%COMM_INTER,CKAPPA2(1,1),WHF%WDES%NPROD*NGLB))
    END IF

! generate the descriptor for the original WDES
! only difference to WDESK is the FFT mesh (GRID and not GRIDHF)
    CALL SETWDES(W%WDES,WDESQ,NK)

    CH=0
! fourier transform local accelerations xi (only own bands)
    fft_back:DO N=W%WDES%NB_LOW,NGLB,W%WDES%NB_PAR
       N_=(N-1)/W%WDES%NB_PAR+1

! add CKAPPA1 to CXI (full acceleration on band N now in CXI)
       IF (WHF%WDES%LOVERL) THEN
          IF (NONLR_S%LREAL) THEN
            ! contribution - r_idir | p_i > C1_i
             CWORK=0
             CALL RACC0(NONLR_D, WDESQ, CKAPPA1(1,N), CWORK(1))
             CWORK=-CWORK
             CALL RACC0(NONLR_S, WDESQ, CKAPPA2(1,N), CWORK(1))

             DO ISPINOR=0,WDESQ%NRSPINORS-1
                CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                     CWORK(1+ISPINOR*WDESQ%GRID%MPLWV), &
                     CH(1+ISPINOR*WDESQ%NGVECTOR,N_),WDESQ%GRID,.TRUE.)
             ENDDO
          ELSE
             ! contribution - | -i d p_i/ d k> C_i
             CALL VNLAC0(NONL_D, WDESK, CKAPPA1(1,N), CH(1,N_))
             CH(:,N_)=-CH(:,N_)
             CALL VNLAC0(NONL_S, WDESK, CKAPPA2(1,N), CH(1,N_))
          ENDIF
       ENDIF

       DO ISPINOR=0,WDESK%NRSPINORS-1
          CALL FFTEXT(WDESK%NGVECTOR,WDESK%NINDPW(1), &
               CXI(1+ISPINOR*GRID_FOCK%MPLWV,N), &
               CH(1+ISPINOR*WDESK%NGVECTOR,N_),GRID_FOCK,.TRUE.)

          DO NP=1,WDESK%NGVECTOR
             MM=NP+ISPINOR*WDESK%NGVECTOR
             CH(MM,N_)=-CH(MM,N_)
            ENDDO
       ENDDO
    ENDDO fft_back

    DEALLOCATE(CXI,CKAPPA1,CKAPPA2,CPROJD,CRHOLM,CDIJ,CDLM,POTFAK,POTFAKQ,LDO)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN,WTMP)
  END SUBROUTINE FOCK_QDER


!***********************************************************************
!
! interface function that returns true if Hartree-Fock is used
! for EXX-OEP certain subroutine must not include the Hartree-Fock term
! since the local potential contains effective contribution
! - subrot
! - linear_optics
! - one center terms
!
!***********************************************************************

    FUNCTION USEFOCK_CONTRIBUTION()
      USE fock
      LOGICAL USEFOCK_CONTRIBUTION
      LOGICAL, EXTERNAL :: USE_OEP_IN_GW

      USEFOCK_CONTRIBUTION=LHFCALC .AND. ((EXXOEP==0 .AND. .NOT. USE_OEP_IN_GW()) &
           .OR. LHFCALC_FORCE)

    END FUNCTION USEFOCK_CONTRIBUTION

!***********************************************************************
!
! interface function that returns true if Hartree-Fock is used
! for the core-valence interaction
!
!***********************************************************************

    FUNCTION USEFOCK_ONECENTER()
      USE fock
      LOGICAL USEFOCK_ONECENTER

      USEFOCK_ONECENTER = LHFCALC .OR. LHFONE

    END FUNCTION USEFOCK_ONECENTER

!***********************************************************************
!
! interface function that returns true if Hartree-Fock is used
! for the one center AE term (one center HF treatment only)
!
!***********************************************************************

    FUNCTION USEFOCK_AE_ONECENTER()
      USE fock
      LOGICAL USEFOCK_AE_ONECENTER

      USEFOCK_AE_ONECENTER=LHFONE

    END FUNCTION USEFOCK_AE_ONECENTER

!***********************************************************************
!
! function to apply one-center HF type Hamiltonians
!
!***********************************************************************

    SUBROUTINE APPLY_ONE_CENTER_AEXX
      USE setexm
      USE fock
      IF (LHFONE) CALL PUSH_XC_TYPE(LEXCH, 1-AEXX, ALDAC, 1-AEXX, AGGAC, 0.0_q)

    END SUBROUTINE APPLY_ONE_CENTER_AEXX

    SUBROUTINE RESTORE_ONE_CENTER_AEXX
      USE setexm
      USE fock
      IF (LHFONE) CALL POP_XC_TYPE

    END SUBROUTINE RESTORE_ONE_CENTER_AEXX

!***********************************************************************
!
! interface function that returns the maximum L quantum number
! for the one centre augmentation charges
!
!***********************************************************************

    FUNCTION FOCK_LMAXONECENTER()
      USE fock
      INTEGER  FOCK_LMAXONECENTER
      FOCK_LMAXONECENTER=LMAX_FOCK

    END FUNCTION FOCK_LMAXONECENTER

!***********************************************************************
!
! interface function that returns true if the one center
! contributions are calculated using 
!
!***********************************************************************

    FUNCTION ONE_CENTER_NMAX_FOCKAE()
      USE fock
      INTEGER ONE_CENTER_NMAX_FOCKAE

      IF (LFOCKAEDFT) THEN
         ONE_CENTER_NMAX_FOCKAE=NMAX_FOCKAE
      ELSE
         ONE_CENTER_NMAX_FOCKAE=0
      ENDIF

    END FUNCTION ONE_CENTER_NMAX_FOCKAE

!***********************************************************************
!
! switch off HF temporarily 
!
!***********************************************************************

    SUBROUTINE PUSH_FOCK()
      USE fock
      IF (LSTACK_FOCK) THEN
         WRITE(*,*) 'internal ERROR in PUSH_FOCK: push already used'
         STOP
      ENDIF
      LHFCALC_STACK=LHFCALC
      LHFCALC=.FALSE.
      LSTACK_FOCK=.TRUE.

    END SUBROUTINE PUSH_FOCK

    SUBROUTINE POP_FOCK()
      USE fock
      IF (.NOT. LSTACK_FOCK) THEN
         WRITE(*,*) 'internal ERROR in POP_FOCK: push was not used'
         STOP
      ENDIF
      LHFCALC = LHFCALC_STACK
      LSTACK_FOCK=.FALSE.

    END SUBROUTINE POP_FOCK

!********************** SUBROUTINE APPLY_GFAC   ************************
!
!  this subroutine multiplies the charge density by the potential kernel
!  (essentially e^2 epsilon_0 / 4 pi (G +k-q)^2)
!
!***********************************************************************

   SUBROUTINE APPLY_GFAC(GRID, CWORK, POTFAK)

     USE prec
     USE mgrid
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)

     PROFILING_START('apply_gfac')

     ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
     DO NP=1,GRID%RC%NP
        CWORK(NP)=POTFAK(NP)*CWORK(NP)
     ENDDO

     PROFILING_STOP('apply_gfac')

   END SUBROUTINE APPLY_GFAC

!
! this version also additionally returns the total energy contribution
! sum_G rho_G POTFAK(G) rho*_G
!

   SUBROUTINE APPLY_GFAC_EXCHANGE(GRID, CWORK, POTFAK, EXCHANGE)

     USE prec
     USE mgrid
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)
     REAL(q)     :: EXCHANGE

     PROFILING_START('apply_gfac_exchange')

     ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
     EXCHANGE=0
     IF (GRID%REAL2CPLX) THEN
        IF (.NOT. ASSOCIATED(GRID%FFTWEIGHT)) THEN
           WRITE(0,*) 'internal error in APPLY_GFAC_EXCHANGE: FFTWEIGHT is not associated'
           STOP
        ENDIF
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORK(NP))*GRID%FFTWEIGHT(NP)
           CWORK(NP)=POTFAK(NP)*CWORK(NP)
        ENDDO
     ELSE
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORK(NP))
           CWORK(NP)=POTFAK(NP)*CWORK(NP)
        ENDDO
     ENDIF

     PROFILING_STOP('apply_gfac_exchange')

   END SUBROUTINE APPLY_GFAC_EXCHANGE

!
! this version contracts over a second supplied density i.e.
! sum_G rho^1_G POTFAK(G) rho*^2_G
!

   SUBROUTINE APPLY_GFAC_EXCHANGE_2(GRID, CWORK, CWORKG, POTFAK, EXCHANGE)

     USE prec
     USE mgrid
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)
     COMPLEX(q)  :: CWORKG(GRID%MPLWV)
     COMPLEX(q)  :: EXCHANGE
     ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
     EXCHANGE=0
     IF (GRID%REAL2CPLX) THEN
        IF (.NOT. ASSOCIATED(GRID%FFTWEIGHT)) THEN
           WRITE(0,*) 'internal error in APPLY_GFAC_EXCHANGE: FFTWEIGHT is not associated'
           STOP
        ENDIF
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORKG(NP))*GRID%FFTWEIGHT(NP)
           CWORK(NP)=POTFAK(NP)*CWORK(NP)
        ENDDO
     ELSE
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORKG(NP))
           CWORK(NP)=POTFAK(NP)*CWORK(NP)
        ENDDO
     ENDIF
   END SUBROUTINE APPLY_GFAC_EXCHANGE_2

!********************** SUBROUTINE EXCHANGE_GFAC   *********************
!
!  this subroutine multiplies calculates
!  sum_G e^2 / 4 pi (G +k-q)^2)  rho_G  rho^*_G
!
!***********************************************************************

   SUBROUTINE EXCHANGE_GFAC(GRID, CWORK, POTFAK, EXCHANGE)

     USE prec
     USE mgrid
     IMPLICIT NONE

     TYPE (grid_3d) GRID
     INTEGER NP
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)
     REAL(q)     :: EXCHANGE

     PROFILING_START('exchange_gfac')

     ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
     EXCHANGE=0
     IF (GRID%REAL2CPLX) THEN
        IF (.NOT. ASSOCIATED(GRID%FFTWEIGHT)) THEN
           WRITE(0,*) 'internal error in APPLY_GFAC_EXCHANGE: FFTWEIGHT is not associated'
           STOP
        ENDIF
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORK(NP))*GRID%FFTWEIGHT(NP)
        ENDDO
     ELSE
        DO NP=1,GRID%RC%NP
           EXCHANGE=EXCHANGE+POTFAK(NP)*CWORK(NP)*CONJG(CWORK(NP))
        ENDDO
     ENDIF

     PROFILING_STOP('exchange_gfac')

   END SUBROUTINE EXCHANGE_GFAC

!********************** SUBROUTINE APPLY_GFAC_WAVEFUN ******************
!
!  this subroutine multiplies the charge density by the potential kernel
!  within a cutoff sphere defined by a plane wave cutoff
!
!***********************************************************************

   SUBROUTINE APPLY_GFAC_WAVEFUN(WDESQ, CWORK, POTFAK)

     USE prec
     USE wave
     IMPLICIT NONE

     TYPE(wavedes1) :: WDESQ
     INTEGER NQ
     REAL(q)     :: POTFAK(WDESQ%NGVECTOR)
     COMPLEX(q)  :: CWORK(WDESQ%NGVECTOR)
   ! local
     INTEGER :: NP, I

     PROFILING_START('apply_gfac_wavefun')

     DO NP=1,WDESQ%NGVECTOR
        CWORK(NP)=POTFAK(NP)*CWORK(NP)
     ENDDO

     PROFILING_STOP('apply_gfac_wavefun')

   END SUBROUTINE APPLY_GFAC_WAVEFUN

!********************** SUBROUTINE APPLY_GFAC_DER   ********************
!
!  this subroutine multiplies the charge density by the q derivative of 
!  the potential kernel and multiplies by -i
!
!***********************************************************************

   SUBROUTINE APPLY_GFAC_QDER(GRID, CWORK2, CWORK, POTFAK)

     USE prec
     USE mgrid
     IMPLICIT NONE
      
     TYPE (grid_3d) GRID
     INTEGER NP
     REAL(q)     :: POTFAK(GRID%MPLWV)
     COMPLEX(q)  :: CWORK2(GRID%MPLWV)    ! charge density
     COMPLEX(q)  :: CWORK(GRID%MPLWV)     ! result
     REAL(q)     :: CSUM

     PROFILING_START('apply_gfac_qder')

     DO NP=1,GRID%RC%NP
        CWORK(NP)=(POTFAK(NP)*CWORK2(NP))*(0.0_q,-1.0_q)+CWORK(NP)
     ENDDO

     PROFILING_STOP('apply_gfac_qder')

   END SUBROUTINE APPLY_GFAC_QDER


!********************** SUBROUTINE APPLY_GFAC_DER   ********************
!
!  this subroutine multiplies the charge density by the potential kernel
!  (essentially e^2 epsilon_0 / 4 pi (G +k-q)^2)
!
!***********************************************************************

   SUBROUTINE APPLY_GFAC_DER(GRID, CWORK, POTFAK, DER, WEIGHT)

     USE prec
     USE mgrid
     IMPLICIT NONE
      
     TYPE (grid_3d) GRID
     INTEGER NP,I
     REAL(q)     :: POTFAK(GRID%MPLWV,0:6)
     COMPLEX(q)  :: CWORK(GRID%MPLWV)
     REAL(q)     :: DER(0:6), WEIGHT
     REAL(q)     :: A

     PROFILING_START('apply_gfac_der')

     DO I=0,6
        A=0
        DO NP=1,GRID%RC%NP
           A=A+POTFAK(NP,I)*CWORK(NP)*CONJG(CWORK(NP))
        ENDDO
        DER(I)=DER(I)+A/GRID%NPLWV*WEIGHT
     ENDDO

     PROFILING_STOP('apply_gfac_der')

   END SUBROUTINE APPLY_GFAC_DER



!********************** SUBROUTINE HFPARAMETERS ************************
!
!  this subroutine returns the parameters for hybride functionals
!  it is called in xcspin, xcgrad, setex and radial
!
!***********************************************************************

    SUBROUTINE HFPARAMETERS(BEXX)
      USE prec
      USE fock
      IMPLICIT NONE
      REAL(q) BEXX

      BEXX=AEXX

      IF (L_MODEL_HF) THEN
         BEXX=1.0
      ENDIF

    END SUBROUTINE HFPARAMETERS

!********************** SUBROUTINE RSPARAMETERS ************************
!
!  this subroutine returns the parameters for range-separated
!  functionals it is called by RAD_POT_EX_HAR in pawfock.F
!
!***********************************************************************

    SUBROUTINE RSPARAMETERS(RMU, LRHF, BRSEXX, BEXX)
      USE prec
      USE fock
      IMPLICIT NONE
      REAL(q) RMU
      LOGICAL LRHF
      REAL(q) :: BRSEXX  ! amount of range seperated exchange
      REAL(q) :: BEXX    ! amount of exact exchange

      RMU=HFSCREEN
      LRHF=LRHFCALC

      IF (L_MODEL_HF) THEN
         BRSEXX=(1-AEXX)
         BEXX=AEXX
      ELSEIF ((HFSCREEN/=0.AND.(.NOT.L_THOMAS_FERMI))) THEN
         BRSEXX=AEXX
         BEXX=0.0
      ELSE
         BRSEXX=0.0
         BEXX=AEXX
      ENDIF

    END SUBROUTINE RSPARAMETERS

!********************** SUBROUTINE OVERL_FOCK *************************
!
! calcs D_lml'm'<p_l'm'|psi_m> and adds it to CRESUL (being kappa_n) 
! (m is not explicitly in the routine but take care when calling it)
!
! modified from OVERL in dfast.F
!
!**********************************************************************

      SUBROUTINE OVERL_FOCK(WDES, LMDIM, CDIJ, CPROJ , CRESUL, LADD)
      USE prec
      USE wave
      
      IMPLICIT NONE
! passed variables
      TYPE (wavedes) WDES
      GDEF CRESUL(WDES%NPROD),CPROJ(WDES%NPROD)
      LOGICAL LOVERL
      INTEGER LMDIM
      GDEF CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS)
      LOGICAL LADD
! local variables
      INTEGER   :: ISPINOR, NPRO
      INTEGER   :: NIS,NT,LMMAXC,NI,L,LP 

      PROFILING_START('overl_fock')

      IF (.NOT. LADD) THEN
        CRESUL=0
      ENDIF

      spinor: DO ISPINOR=0,WDES%NRSPINORS-1            
         NPRO =ISPINOR *WDES%NPRO/2
         NIS =1
         DO NT=1,WDES%NTYP
            LMMAXC=WDES%LMMAX(NT)
            IF (LMMAXC==0) GOTO 100
            DO NI=NIS,WDES%NITYP(NT)+NIS-1
               DO L =1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                  DO LP=1,LMMAXC
                     CRESUL(L+NPRO)=CRESUL(L+NPRO)+ &
                          CDIJ(LP,L,NI,1+ISPINOR)* &
                          CPROJ(LP+NPRO)
                  ENDDO
               ENDDO
               NPRO = LMMAXC+NPRO
            ENDDO
100         NIS = NIS+WDES%NITYP(NT)
         ENDDO
      ENDDO spinor
 
      PROFILING_STOP('overl_fock')

      RETURN
    END SUBROUTINE OVERL_FOCK

!**********************************************************************
!
! F77 kernel to calculate the contribution to the ion decomposed
! energy in the Fock routine
!
!**********************************************************************

    SUBROUTINE ECCP_NL_FOCK(LMDIM,LMMAXC,CDIJ0,CDIJ1, &
         CPROJ,CPROJ0,CPROJ1,ENL,WEIGHT)
      USE prec
      IMPLICIT NONE
      INTEGER LMMAXC, LMDIM
      REAL(q) ENL
      REAL(q) WEIGHT
      GDEF CDIJ0(LMDIM,LMDIM),CDIJ1(LMDIM,LMDIM)
      GDEF CPROJ(LMMAXC),CPROJ0(LMMAXC),CPROJ1(LMMAXC)
    ! local
      INTEGER L, LP
      REAL(q) ENL_ADD

      PROFILING_START('eccp_nl_fock')

      ENL_ADD=0
      DO L=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
         DO LP=1,LMMAXC
            ENL_ADD=ENL_ADD+REAL(CPROJ(LP)*(CDIJ0(LP,L)*GCONJG(CPROJ0(L))+ &
                                            CDIJ1(LP,L)*GCONJG(CPROJ1(L))))
         ENDDO
      ENDDO
      ENL=ENL+ENL_ADD*WEIGHT

      PROFILING_STOP('eccp_nl_fock')

    END SUBROUTINE ECCP_NL_FOCK

!==========================================================================
! 
! this subroutine adds a quadratic field to the potential
! to correct for the errors cause by the periodic FFT 
! this is a "second" order correction and applicable only in the case
! of isolated molecules 
!
! WORK IN PROGRESS, BUT PROPABLY NOT VERY IMPORTANT
!
!==========================================================================


    SUBROUTINE QUADRATIC_FIELD_CORRECTION(GRID, GWORK, LATT_CUR) 
      USE prec
      USE lattice
      USE mgrid
      USE constant
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      GDEF           GWORK(m_ GRID%MPLWV)    ! fock pot in real sp
      TYPE (latt)    LATT_CUR
    ! local
      INTEGER IDIP, NOUT, NOUTH, IDIR, INDMIN(3), NI, NC, N1, N2, N3, II
      REAL(q) QUADFAC, QUAD_FIELD, CUTOFF, POTCOR, XX
    !
    ! a smooth cutoff function is used for potentials and charge densities
    ! around the dividing plane
    ! WIDTH must be at least 1
    ! 4 grid points around the step function are usually a good choice
      REAL(q), PARAMETER :: WIDTH=4
    
  
      QUAD_FIELD=TPI/LATT_CUR%OMEGA*FELECT/3
   ! correct the potential
      dir: DO IDIP=1,3
         NOUT=GRID%NGPTAR(IDIP)
         NOUTH=NOUT/2
         QUADFAC=QUAD_FIELD*(LATT_CUR%ANORM(IDIP)/NOUT)**2
         IDIR=IDIP
         IF (GRID%RL%NFAST==3) THEN
            IDIR=MOD(IDIP,3)+1 ! mpi version: x-> N2, y-> N3, z-> N1
         ENDIF
         INDMIN(IDIR)=GRID%NGPTAR(IDIP)/2
         NI=0
         DO NC=1,GRID%RL%NCOL
            N2=GRID%RL%I2(NC)
            N3=GRID%RL%I3(NC)
            DO N1=1,GRID%RL%NROW
               NI=NI+1
               IF (IDIR==1) II=MOD(N1-INDMIN(1)+NOUT,NOUT)-NOUTH
               IF (IDIR==2) II=MOD(N2-INDMIN(2)+NOUT,NOUT)-NOUTH
               IF (IDIR==3) II=MOD(N3-INDMIN(3)+NOUT,NOUT)-NOUTH
               XX=ABS(ABS(II)-NOUTH)
               IF (XX > WIDTH) THEN
                  CUTOFF=1.0
               ELSE
                  CUTOFF=ABS(SIN(PI*XX/WIDTH/2))
               ENDIF
               POTCOR=-QUADFAC*II*II*CUTOFF*CUTOFF
               GWORK(NI)=GWORK(NI)+POTCOR
            ENDDO
         ENDDO
      ENDDO dir
    END SUBROUTINE QUADRATIC_FIELD_CORRECTION


!==========================================================================
!
! standard calling interface to the SET_GFAC routine
! requires one to pass down two k-points for which the Coulomb kernel is required
! this one is without explicitly compiled interface
!
!==========================================================================

    SUBROUTINE SET_GFAC_NOINTER(GRID, LATT_CUR, NK, NQ, FSG, POTFAK)
      USE fock
      USE lattice
      USE full_kpoints
      IMPLICIT NONE

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      INTEGER NK, NQ
      REAL(q) :: FSG
      REAL(q) :: POTFAK(GRID%MPLWV)

      CALL SET_GFAC_VKPT(GRID, LATT_CUR, KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ), NQ, FSG, POTFAK)
    END SUBROUTINE SET_GFAC_NOINTER
