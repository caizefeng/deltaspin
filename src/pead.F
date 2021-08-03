#define debug
#include "symbol.inc"
!***********************************************************************
!
! This module contains the routines that calculate the derivative
! d\psi_k / dk, in accordance with the finite difference approximation
! presented by Nunes and Gonze [Phys. Rev. B 63, 155107 (2001)] in their
! "perturbation expansion after discretization" (PEAD) formulation
! of the Berry-phase treatment of the homogeneous electric field
! perturbation in insulators.
!
! Additionally this module contains a routine that evaluates the
! macroscopic polarization within the Berry-phase formalism.
! This routine will in future replace the implementation of the
! Berry-phase formalism in elpol.F
!
!***********************************************************************

      MODULE pead
      USE prec
      USE wave
      USE kpoints_change

      ! Born effective charges
      LOGICAL, SAVE :: LBORN = .FALSE.
      REAL(q), SAVE, ALLOCATABLE ::  BORN_CHARGES_PEAD(:,:,:)
            
      PRIVATE :: DPSI_DK_BERRY,DPSI_DK_EV,DPSI_DK_ORTHO, &
     &   CALC_POLARIZATION,IMLNDET,DETS,SET_EV,CALC_OVERLAP,CALC_OVERLAP_, &
     &   SET_PHASE_SHIFT,OVERL_AND_APPLY_PHASE_SHIFT,APPLY_PHASE_SHIFT_CPROJ, &
     &   SET_PHASE_SHIFT_GRID,SET_PHASE_SHIFT_GRID_RSPACE,APPLY_PHASE_SHIFT_GRID, &
     &   SETYLM_AUG2,SETDIJ_R,DUMP_HAM_PEAD,DUMP_HAM_FILE, &
     &   LSKIP_EDOTP_DURING_ELMIN,GENERATE_KPOINTS_TRANS_FULL, &
     &   W1_ROTATE,WA_ROTATE,SETUP_STRINGS,CALC_EWFOR_AND_EWSIF,ONE_CENTRE_CHARGE, &
     &   SETQIJB,OVERL_AND_APPLY_PHASE_SHIFT_C

!
! determines whether Q_ij d <p_j| psi> dk  is returned or 
!   <p_j| psi> dk
! the later case is e.g. required for calculating magnetic moments
! using the Berry phase
!
      LOGICAL, SAVE :: LPEAD_RETURN_Q_CPROJ=.TRUE.
!
! Local variables read from INCAR
!
      ! read from INCAR, switches on the pead routines
      LOGICAL, PRIVATE, SAVE :: LPEAD=.FALSE.

      ! order of the numerical derivatives in the calculation
      ! of the macroscopic polarization and d\psi/dk
      INTEGER, PRIVATE, SAVE :: PEAD_ORDER=4

      ! for LCALCPOL=.TRUE. VASP calculates the macroscopic
      ! polarization after electronic convergence has
      ! been reached
      LOGICAL, PRIVATE, SAVE :: LCALCPOL=.FALSE.

      ! for LCALCEPS=.TRUE. VASP calculates the dielectric
      ! matrix after electronic convergence has been reached
      LOGICAL, PRIVATE, SAVE :: LCALCEPS=.FALSE.

      ! for LPEAD_SCF=.FALSE. calls to PEAD_ACC_CALC
      ! will be skipped after the first
      LOGICAL, PRIVATE, SAVE :: LPEAD_SCF=.TRUE.

      ! the external electric field
      REAL(q), PRIVATE, SAVE :: EFIELD_PEAD(3)

      ! skip the calculation of the macroscopic dipole
      ! moment during the electronic minimization (saves time)
      LOGICAL, PRIVATE, SAVE :: LSKIP_EDOTP

      ! skip the calculation of the non-selfconsistent
      ! response to the electric field
      LOGICAL, PRIVATE, SAVE :: LSKIP_NSCF
      
      ! skip the calculation of the selfconsistent
      ! response to the electric field
      LOGICAL, PRIVATE, SAVE :: LSKIP_SCF

      ! origin w.r.t. which the ionic dipole moment
      ! will be calculated (DIPOL-tag: same as read in dipol.F)
      REAL(q), PRIVATE, SAVE :: POSCEN(3)=-100
!
! Local variable set by REQUEST_PEAD
!
      ! A call to REQUEST_PEAD *before* the call to PEAD_READER
      ! will set EXTERNAL_REQUEST_FOR_PEAD=.TRUE., which in turn
      ! will ensure that LPEAD=.TRUE. regardless of what is specified
      ! in the INCAR. This way the routines in mlwf.F and wnpr.F can
      ! switch on the PEAD routines.
      LOGICAL, PRIVATE, SAVE :: EXTERNAL_REQUEST_FOR_PEAD=.FALSE.
!
! Local variables set by PEAD_SETUP and PEAD_RESETUP_WDES
!
      ! d_IJ = \int r*Q_ij(r) dr
      OVERLAP, PRIVATE, ALLOCATABLE, SAVE :: d_IJ(:,:,:,:,:)

      ! WDES_FULL_PEAD contains the descriptors for the wave functions
      ! at all k-points in the full Brillouin zone
!     TYPE(wavedes), PRIVATE, SAVE, POINTER :: WDES_FULL_PEAD
      TYPE(wavedes), SAVE, POINTER :: WDES_FULL_PEAD
      
      ! KPOINTS_TRANS_FULL_PEAD contains the information needed
      ! to obtain the wave functions at any point in the full first
      ! Brillouin zone from their symmetry related counterparts in the
      ! irreducible Brillouin zone
!     TYPE(skpoints_trans), PRIVATE, SAVE, TARGET :: KPOINTS_TRANS_FULL_PEAD
      TYPE(skpoints_trans), SAVE, TARGET :: KPOINTS_TRANS_FULL_PEAD

!
! Local variables set by PEAD_POLARIZATION_CALC
!
      ! stores the spin resolved dipole moment
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: RSTORE(:,:)

      ! stores the spin resolved change in dipole moment
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: DRSTORE(:,:)

      ! stores the spin resolved "expectation value" dipole moment
      ! needed to compute the double counting corrections
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: R_EV_STORE(:,:)

      ! R_EV_INIT, R_BP_INIT and SUM_PHASE_ON_STRING_INIT store
      ! the "expectation value" and "Berry phase" contributions
      ! when PEAD_POLARIZATION_CALC is called with LINITIALIZE=.TRUE.
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: R_EV_INIT(:,:)
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: R_BP_INIT(:,:)
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: SUM_PHASE_ON_STRING_INIT(:,:,:,:)

      ! EPSILON_SCF and EPSILON_NSCF store the dielectric tensor,
      ! with and without local field effects, respectively
      REAL(q), PRIVATE, SAVE :: EPSILON_NSCF(3,3)
      REAL(q), PRIVATE, SAVE :: EPSILON_SCF(3,3)

      ! IONIC_DIPOLE stores the total dipole moment of the ions
      ! w.r.t. to POSCEN (as specified by the DIPOL-tag)
      REAL(q), PRIVATE, SAVE :: IONIC_DIPOLE(3)

      ! NSTRINGS contains the number of k-point strings in
      ! each reciprocal space direction
      INTEGER, PRIVATE, SAVE :: NSTRINGS(3)

      ! MAP_TO_STRING maps each k-point in the full brillouin
      ! zone to a certain k-point string
      INTEGER, PRIVATE, ALLOCATABLE, SAVE :: MAP_TO_STRING(:,:)

!
! Local variables used by the PEAD_ACC_* routines
!
      ! LFIRST_CALL_ACC_CALC is set to FALSE after
      ! the first call to PEAD_ACC_CALC_ALL
      LOGICAL, PRIVATE, SAVE :: LFIRST_CALL_ACC_CALC=.TRUE.

      ! store the gradients dpsi/dk for one k-point
      COMPLEX(q), PRIVATE, ALLOCATABLE, SAVE :: RPHI_STORE_LOCAL(:,:,:)
      GDEF , PRIVATE, ALLOCATABLE, SAVE :: RPHI_STORE_LOCAL_CPROJ(:,:,:)

      ! store the gradients dpsi/dk for all k-points
      ! in the irreducible Brillouin zone
      COMPLEX(qs), PRIVATE, ALLOCATABLE, SAVE :: RPHI_STORE_LOCAL_ALL(:,:,:,:,:)
      GDEFS , PRIVATE, ALLOCATABLE, SAVE :: RPHI_STORE_LOCAL_ALL_CPROJ(:,:,:,:,:)

!
! Local variables set by CHECK_OCCUPATIONS
!
      ! after a call to CHECK_OCCUPATIONS, LINSULATING stores
      ! whether the system is insulating or not
      LOGICAL, PRIVATE, SAVE :: LINSULATING=.FALSE.      

      ! stores the number of the highest occupied band
      INTEGER, PRIVATE, ALLOCATABLE, SAVE :: NOCC(:)

!
! Local variables used by PEAD_(RE)STORE_WAVE_ORIG
!
      ! stores a copy of the wave functions
      TYPE(wavespin), SAVE :: W_STORE

!
! Local variables, miscellaneous
!
      ! LDOIO is set to true for the node that
      ! is responsible for writing output
      LOGICAL, PRIVATE, SAVE :: LDOIO

      ! UNIT6 is a local copy of IO%IU6
      ! convenient for calls to the timing routines
      INTEGER, PRIVATE, SAVE :: UNIT6

      ! LOCCUPIED_ONLY=.TRUE. limits the calculation
      ! of the overlap matrix <u_k1_n|S|u_k2_m> to
      ! occupied states only
      LOGICAL, PRIVATE, SAVE :: LOCCUPIED_ONLY=.TRUE.

      ! signals the adiabatic switching of the 
      ! macroscopic electric field
      LOGICAL, PRIVATE, SAVE :: LFIELD_SWITCHED_ON=.FALSE.

      ! LRESTART_EDWAV=.TRUE. sets ICOUNT=-1 in EDWAV and
      ! forces a reinitialisation of EDWAV, this is necessary
      ! in case IALGO/=3 before the electric field was switched on
      LOGICAL, PRIVATE, SAVE :: LRESTART_EDWAV=.FALSE.

      ! Weights for the numerical differentiation stencils
      REAL(q) FAC(4,4)
      DATA FAC / 1.0_q,         0.0_q,         0.0_q,         0.0_q, &
     &           1.333333333_q, -.166666666_q, 0.0_q,         0.0_q, &
     &           1.5_q,         -.3_q,         0.033333333_q, 0.0_q, &
     &           1.6_q,         -.4_q,         0.076190476_q, -.007142857_q /

#ifdef gammareal      
      LOGICAL, PRIVATE, SAVE :: LIMAGINARY_PART=.FALSE.
#endif

      CONTAINS

!***********************************************************************
!******************** PUBLIC PROCEDURES ********************************
!***********************************************************************


!******************** SUBROUTINE PEAD_READER ***************************
!
! Reads LPEAD, LCALCPOL, EFIELD_PEAD, IPEAD, and SKIP_EDOTP
! from the INCAR file
!
!***********************************************************************
      SUBROUTINE PEAD_READER(IU5, IU6, IU0)
      USE base
      USE vaspxml
      USE full_kpoints
      USE main_mpi
      IMPLICIT NONE
      INTEGER IU5,IU6,IU0
      ! local variables
      INTEGER I
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM,LRPA
      CHARACTER (1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      CALL RDATAB(LOPEN,INCAR,IU5,'LPEAD','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LPEAD,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LPEAD'' from file INCAR.'
         LPEAD=.FALSE.
      ENDIF

      CALL XML_INCAR('LPEAD','L',IDUM,RDUM,CDUM,LPEAD,CHARAC,N)

      LCALCPOL=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LCALCPOL','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LCALCPOL,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LCALCPOL'' from file INCAR.'
         LCALCPOL=.FALSE.
      ENDIF

      CALL XML_INCAR('LCALCPOL','L',IDUM,RDUM,CDUM,LCALCPOL,CHARAC,N)

      EFIELD_PEAD=0
      CALL RDATAB(LOPEN,INCAR,IU5,'EFIELD_PEAD','=','#',';','F', &
     &            IDUM,EFIELD_PEAD,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<3))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''EFIELD_PEAD'' from file INCAR.'
         EFIELD_PEAD=0
      ENDIF
      CALL XML_INCAR_V('EFIELD_PEAD','F',IDUM,EFIELD_PEAD,CDUM,LDUM,CHARAC,N)

      LCALCEPS=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LCALCEPS','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LCALCEPS,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LCALCEPS'' from file INCAR.'
         LCALCEPS=.FALSE.
      ENDIF

      CALL XML_INCAR('LCALCEPS','L',IDUM,RDUM,CDUM,LCALCEPS,CHARAC,N)

      IF (LPEAD_CALC_EPS()) THEN
         DO I=1,3
            IF (ABS(EFIELD_PEAD(I))<1E-5_q) EFIELD_PEAD(I)=0.01_q
         ENDDO
      ENDIF

      IF (LPEAD_NONZERO_EFIELD().OR.LPEAD_CALC_POL().OR.LPEAD_CALC_EPS()) LPEAD=.TRUE.

      IF (LUSEPEAD()) THEN
#ifdef MPI
         ! sanity check, the PEAD routines need NCORE=1
         IF (NCORE/=1) THEN
            CALL VTUTOR('E','PEAD NCORE not 1',RDUM,1,IDUM,1,CDUM,1,LDUM,1,IU0,3)
            STOP
         ENDIF
#endif

         PEAD_ORDER=4
         CALL RDATAB(LOPEN,INCAR,IU5,'IPEAD','=','#',';','I', &
        &            PEAD_ORDER,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'Error reading item ''IPEAD'' from file INCAR.'
         ENDIF
         CALL XML_INCAR('IPEAD','I',PEAD_ORDER,RDUM,CDUM,LDUM,CHARAC,N)

         LSKIP_EDOTP=.FALSE.
         CALL RDATAB(LOPEN,INCAR,IU5,'SKIP_EDOTP','=','#',';','L', &
        &            IDUM,RDUM,CDUM,LSKIP_EDOTP,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''SKIP_EDOTP'' from file INCAR.'
            LSKIP_EDOTP=.FALSE.
         ENDIF
         CALL XML_INCAR('SKIP_EDOTP','L',IDUM,RDUM,CDUM,LSKIP_EDOTP,CHARAC,N)

         LRPA=.FALSE.
         CALL RDATAB(LOPEN,INCAR,IU5,'LRPA','=','#',';','L', &
        &            IDUM,RDUM,CDUM,LRPA,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''LRPA'' from file INCAR.'
            LRPA=.FALSE.
         ENDIF
         LSKIP_NSCF=.NOT.LRPA
         CALL XML_INCAR('LRPA','L',IDUM,RDUM,CDUM,LRPA,CHARAC,N)

         LSKIP_SCF=.FALSE.
         CALL RDATAB(LOPEN,INCAR,IU5,'SKIP_SCF','=','#',';','L', &
        &            IDUM,RDUM,CDUM,LSKIP_SCF,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''SKIP_SCF'' from file INCAR.'
            LSKIP_SCF=.FALSE.
         ENDIF
         CALL XML_INCAR('SKIP_SCF','L',IDUM,RDUM,CDUM,LSKIP_SCF,CHARAC,N)

         POSCEN=-100
         CALL RDATAB(LOPEN,INCAR,IU5,'DIPOL','=','#',';','F', &
        &            IDUM,POSCEN,CDUM,LDUM,CHARAC,N,3,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N/=3))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''DIPOL'' from file INCAR.'
            POSCEN=0
         ENDIF

         CALL XML_INCAR_V('DIPOL','F',IDUM,POSCEN,CDUM,LDUM,CHARAC,N)

         !
         CALL USE_FULL_KPOINTS
      ENDIF
      
      CLOSE(IU5)
      
      RETURN
      END SUBROUTINE PEAD_READER


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_WRITER(IU6)
      USE prec
      IMPLICIT NONE
      INTEGER IU6
      ! local variables
      INTEGER I
      REAL(q) EFIELD_TMP(3)
      
      IF (IU6>0.AND.LPEAD) THEN
         WRITE(IU6,10) LPEAD,PEAD_ORDER,LCALCPOL,LCALCEPS,EFIELD_PEAD, &
        &   LSKIP_EDOTP,.NOT.LSKIP_NSCF,LSKIP_SCF
      ENDIF

   10 FORMAT(' PEAD related settings:'/&
             '   LPEAD      =',L6,'    switch on PEAD'/&
             '   IPEAD      =',I6,'    finite difference order for dpsi/dk'/&
             '   LCALCPOL   =',L6,'    calculate macroscopic polarization'/&
             '   LCALCEPS   =',L6,'    calculate dielectric tensor'/&
             '   EFIELD_PEAD=',3F10.4,/&
             '   SKIP_EDOTP =',L6,/&
             '   LRPA       =',L6,/&
             '   SKIP_SCF   =',L6)  
            
      RETURN
      END SUBROUTINE PEAD_WRITER


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_SETUP(WDES,GRID,GRIDC,GRIDUS,C_TO_US,KPOINTS, &
     &   LATT_CUR,LATT_INI,T_INFO,P,LMDIM,LOVERL,IRDMAX,IO)
      USE prec
      USE base
      USE fock
      USE wave
      USE lattice
      USE poscar
      USE pseudo
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavedes), TARGET :: WDES
      TYPE(grid_3d) GRID,GRIDC,GRIDUS
      TYPE(transit) C_TO_US
      TYPE(kpoints_struct) KPOINTS
      TYPE(latt) LATT_CUR
      TYPE(latt) LATT_INI
      TYPE(type_info) T_INFO
      TYPE(potcar) P(:)
      TYPE(in_struct) IO
      INTEGER LMDIM,IRDMAX
      LOGICAL LOVERL
      ! local variables
      INTEGER IDIR,ITUT
      REAL(q) RTUT
      COMPLEX(q) CDUM
      LOGICAL LDUM
      OVERLAP TMP(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      
      IF (.NOT.LUSEPEAD()) RETURN

      IF (ALLOCATED(d_IJ)) THEN
         DEALLOCATE(d_IJ)
      ENDIF
      
      CALL CHECK_FULL_KPOINTS

      ! sanity check
      IF (KPOINTS_FULL%NKPTS/=KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ) THEN
         CALL VTUTOR('E','KPOINTS PEAD',RTUT,1,ITUT,1,CDUM,1,LDUM,1,IO%IU0,2)
         STOP
      ENDIF
      
      CALL GENERATE_KPOINTS_TRANS_FULL(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6)
      
      ! set d_IJ (d_IJ = \int r*Q_ij(r) dr)
      ALLOCATE(d_IJ(3,LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ))
      DO IDIR=1,3
         CALL SETDIJ_R(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
        &   LOVERL,LMDIM,TMP,IRDMAX,IDIR)
         d_IJ(IDIR,:,:,:,:)=TMP(:,:,:,:)
      ENDDO

      ! for convenience
#ifdef MPI
      LDOIO=(WDES%COMM%NODE_ME==WDES%COMM%IONODE)
#else
      LDOIO=.TRUE.
#endif
      UNIT6=IO%IU6
      
      RETURN
      END SUBROUTINE PEAD_SETUP


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_RESETUP_WDES(WDES,GRID,KPOINTS,LATT_CUR,LATT_INI,IO)
      USE prec
      USE base
      USE mgrid
      USE lattice
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavedes), TARGET :: WDES
      TYPE(grid_3d) GRID
      TYPE(latt) LATT_CUR
      TYPE(latt) LATT_INI
      TYPE(kpoints_struct) KPOINTS
      TYPE(in_struct) IO

      IF (.NOT.LUSEPEAD()) RETURN
           
      CALL CHECK_FULL_KPOINTS

      CALL GENERATE_KPOINTS_TRANS_FULL(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6)

      RETURN
      END SUBROUTINE PEAD_RESETUP_WDES


!******************** SUBROUTINE ***************************************
!
!  subroutine to adapt the wavefunctions to new (lower) symmetry
!
!***********************************************************************

      SUBROUTINE PEAD_RESYMMETRIZE( &
     &   KPOINTS,WDES,NONLR_S,NONL_S,W,W_F,W_G,CHAM,CHF, &
     &   GRID,P,T_INFO,INFO,LATT_CUR,LATT_INI,DYN,SYMM,IO)
      USE prec
      USE base
      USE pseudo
      USE poscar
      USE mgrid
      USE lattice
      USE wave_high
      USE msymmetry
      USE mkpoints
      USE nonl_high
      USE kpoints_change
      USE fock
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      TYPE(wavespin) W_F
      TYPE(wavespin) W_G
      TYPE(type_info) T_INFO
      TYPE(info_struct) INFO
      TYPE(latt) LATT_CUR
      TYPE(latt) LATT_INI
      TYPE(dynamics) DYN
      TYPE(symmetry) SYMM
      TYPE(in_struct) IO
      TYPE(grid_3d) GRID
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE(kpoints_struct) KPOINTS
      GDEF , POINTER :: CHF(:,:,:,:),CHAM(:,:,:,:)
      ! local variables
      TYPE (skpoints_trans) KPOINTS_TRANS
      INTEGER I,NCDIJ
      REAL(q) EFIELD_DIR(3)
      REAL(q), ALLOCATABLE :: VEL(:,:)

      ! quick return if the electric field is essentially zero
!     IF (.NOT.LPEAD_NONZERO_EFIELD()) RETURN

      ALLOCATE(VEL(SIZE(DYN%VEL,1),SIZE(DYN%VEL,2)))

      IF (SYMM%ISYM>0) THEN
         VEL=DYN%VEL
         EFIELD_DIR=EFIELD_PEAD
         CALL KARDIR(1,EFIELD_DIR,LATT_CUR%B)
         DO I=1,T_INFO%NIOND         
            VEL(1:3,I)=VEL(1:3,I)+EFIELD_DIR(1:3)
         ENDDO
         NCDIJ=INFO%ISPIN
         IF (WDES%LNONCOLLINEAR) NCDIJ=4

         CALL INISYM(LATT_CUR%A,DYN%POSION,VEL,T_INFO%LSFOR, &
             T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND, &
             SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP, &
             SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
             SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,NCDIJ,IO%IU6)
! test
!        CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
!       &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,INFO%ISPIN,IO%IU6)
!        SYMM%ISYM=-1
! test
#ifdef oldsym
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
             SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
             T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#else
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
             SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
             T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#endif
         CALL KPAR_SYNC_ALL(WDES,W)
         CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
         CALL PEAD_RESETUP_WDES(WDES,GRID,KPOINTS,LATT_CUR,LATT_INI,IO)

         CALL REALLOCATE_WAVE( W  , GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS)
         
         CALL DEALLOCW(W_F)
         CALL ALLOCW(WDES,W_F)
!gK 05.08.2010 removed
!         CALL REALLOCATE_WAVE( W_F, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS)
         
         CALL DEALLOCW(W_G)
         CALL ALLOCW(WDES,W_G)
!gK 05.08.2010 removed
!         CALL REALLOCATE_WAVE( W_G, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS)

         IF (ASSOCIATED(CHF) )  DEALLOCATE(CHF)
         IF (ASSOCIATED(CHAM) ) DEALLOCATE(CHAM)
         ALLOCATE(CHAM(W%WDES%NB_TOT,W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN))
         ALLOCATE( CHF(W%WDES%NB_TOT,W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN))
      ENDIF     
      
      DEALLOCATE(VEL)

      CALL RESETUP_FOCK( WDES, LATT_CUR)
      CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)
      
      RETURN
      END SUBROUTINE PEAD_RESYMMETRIZE


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_STORE_WAVE_ORIG(W)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      ! local variables
      TYPE(wavedes1) WDESK
      INTEGER NK,ISP
      ! just a safety measure, deallocate first
      CALL DEALLOCW(W_STORE)
      ! then allocate anew
      CALL ALLOCW(W%WDES,W_STORE)
      ! W_STORE%WDES points at W%WDES
      W_STORE%WDES=>W%WDES
      ! and hardcopy W to W_STORE
      spin: DO ISP=1,W%WDES%ISPIN
      kpoints: DO NK=1,W%WDES%NKPTS
         CALL SETWDES(W%WDES,WDESK,NK)
         CALL WA_COPY( ELEMENTS(W,WDESK,ISP),ELEMENTS(W_STORE,WDESK,ISP) )
         W_STORE%CELTOT(:,NK,ISP)=W%CELTOT(:,NK,ISP)
         W_STORE%FERTOT(:,NK,ISP)=W%FERTOT(:,NK,ISP)
         W_STORE%AUXTOT(:,NK,ISP)=W%AUXTOT(:,NK,ISP)
      ENDDO kpoints
      ENDDO spin
      
      RETURN
      END SUBROUTINE PEAD_STORE_WAVE_ORIG


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_RESTORE_WAVE_ORIG(W)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      ! local variables
      TYPE(wavedes1) WDESK
      INTEGER NK,ISP      
      ! hardcopy W_STORE to W
      spin: DO ISP=1,W%WDES%ISPIN
      kpoints: DO NK=1,W%WDES%NKPTS
         CALL SETWDES(W%WDES,WDESK,NK)
         CALL WA_COPY( ELEMENTS(W_STORE,WDESK,ISP),ELEMENTS(W,WDESK,ISP) )
         W%CELTOT(:,NK,ISP)=W_STORE%CELTOT(:,NK,ISP)
         W%FERTOT(:,NK,ISP)=W_STORE%FERTOT(:,NK,ISP)
         W%AUXTOT(:,NK,ISP)=W_STORE%AUXTOT(:,NK,ISP)
      ENDDO kpoints
      ENDDO spin      
      
!     CALL DEALLOCW(W_STORE)
      
      RETURN
      END SUBROUTINE PEAD_RESTORE_WAVE_ORIG


!******************** SUBROUTINE PEAD_DPSI_DK_1K ***********************
!
!***********************************************************************
      SUBROUTINE PEAD_DPSI_DK_1K( &
     &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE mkpoints
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE(kpoints_struct) KPOINTS
      INTEGER ISP
      REAL(q) K(3)
      OVERLAP CQIJ(:,:,:,:)
      COMPLEX(q) RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,3)
      GDEF RPHI_CPROJ(W%WDES%NPROD, W%WDES%NBANDS,3)

      SELECT CASE(PEAD_ORDER)
      CASE(1)
      ! First order differentiation            
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ)
      CASE(2)
      ! Second order differentiation
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=1,PREFAC=1.333333333_q)
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=2,PREFAC=-.166666666_q)
      CASE(3)
      ! Third order differentiation
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=1,PREFAC=1.5_q)
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=2,PREFAC=-.3_q)      
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=3,PREFAC=0.033333333_q)
         CASE(4)
         ! Fourth order differentiation
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=1,PREFAC=1.6_q)
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=2,PREFAC=-.4_q)
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=3,PREFAC=0.076190476_q)
         CALL DPSI_DK_BERRY( &
        &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ,STRIDE=4,PREFAC=-.007142857_q)
      END SELECT

      IF (LPEAD_RETURN_Q_CPROJ) THEN
         CALL DPSI_DK_EV(W,K,ISP,RPHI_1K_CPROJ=RPHI_CPROJ)      
      ENDIF

!     CALL DPSI_DK_ORTHO( &
!    &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_1K=RPHI,RPHI_1K_CPROJ=RPHI_CPROJ)
            
      RETURN
      END SUBROUTINE PEAD_DPSI_DK_1K


!******************** SUBROUTINE PEAD_DPSI_DK_ALL **********************
!
! This routine is called from linear_response.F
!
!***********************************************************************
      SUBROUTINE PEAD_DPSI_DK_ALL( &
     &   W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)
      USE ini
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE mkpoints
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE(kpoints_struct) KPOINTS
      OVERLAP CQIJ(:,:,:,:)
      COMPLEX(qs) RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      GDEFS RPHI_CPROJ(W%WDES%NPROD, W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      ! local variables
      INTEGER ISP,IK
      REAL(q) K(3)
      REAL(q) WEIGHT

      CALL START_TIMING("DPSIDK")

      RPHI=0
      RPHI_CPROJ=0
     
      spin: DO ISP=1,W%WDES%ISPIN
      kpoint: DO IK=1,KPOINTS%NKPTS
#ifdef MPI
         IF (MOD(IK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE 
#endif
         WEIGHT=W%WDES%RSPIN*W%WDES%WTKPT(IK)
         K(:)=KPOINTS%VKPT(:,IK)

         SELECT CASE(PEAD_ORDER)
         CASE(1)
         ! First order differentiation            
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)
         CASE(2)
         ! Second order differentiation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=1,PREFAC=1.333333333_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=2,PREFAC=-.166666666_q)
         CASE(3)
         ! Third order differentiation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=1,PREFAC=1.5_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=2,PREFAC=-.3_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=3,PREFAC=0.033333333_q)
         CASE(4)
         ! Fourth order differentiation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=1,PREFAC=1.6_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=2,PREFAC=-.4_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=3,PREFAC=0.076190476_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,STRIDE=4,PREFAC=-.007142857_q)
         END SELECT
         
         IF (LPEAD_RETURN_Q_CPROJ) THEN
            CALL DPSI_DK_EV(W,K,ISP,RPHI_CPROJ)
         ENDIF

!        CALL DPSI_DK_ORTHO( &
!       &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)

!        RPHI(:,:,IK,ISP,1:3)=WEIGHT*RPHI(:,:,IK,ISP,1:3)
!        RPHI_CPROJ(:,:,IK,ISP,1:3)=WEIGHT*RPHI(:,:,IK,ISP,1:3)

      ENDDO kpoint
      ENDDO spin     

      CALLMPI( M_sum_single(W%WDES%COMM_KINTER,RPHI(1,1,1,1,1),2*SIZE(RPHI)) )
#ifdef gammareal
      CALLMPI( M_sum_single(W%WDES%COMM_KINTER,RPHI_CPROJ(1,1,1,1,1),SIZE(RPHI_CPROJ)) )
#else
      CALLMPI( M_sum_single(W%WDES%COMM_KINTER,RPHI_CPROJ(1,1,1,1,1),2*SIZE(RPHI_CPROJ)) )
#endif

      CALL STOP_TIMING("DPSIDK",UNIT6)

      RETURN
      END SUBROUTINE PEAD_DPSI_DK_ALL


!******************** SUBROUTINE PEAD_DPSI_DK_IDIR *********************
!
! This routine is called from linear_optics.F
!
!***********************************************************************     
      SUBROUTINE PEAD_DPSI_DK_IDIR( &
     &   W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,IDIR,DWDK)
      USE ini
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE mkpoints
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE(kpoints_struct) KPOINTS
      OVERLAP CQIJ(:,:,:,:)
      INTEGER IDIR
      TYPE(wavespin) DWDK
      ! local variables
      INTEGER ISP,IK
      REAL(q) K(3)
      REAL(q) WEIGHT
      
!     DWDK%CW=0
!     DWDK%CPROJ=0
    
      CALL START_TIMING("DPSIDK")
     
      spin: DO ISP=1,W%WDES%ISPIN
      kpoint: DO IK=1,KPOINTS%NKPTS
#ifdef MPI
         IF (MOD(IK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE 
#endif      
         WEIGHT=W%WDES%RSPIN*W%WDES%WTKPT(IK)
         K(:)=KPOINTS%VKPT(:,IK)

         SELECT CASE(PEAD_ORDER)
         CASE(1)
         ! First order differentiation            
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK)
         CASE(2)
         ! Second order differentation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=1,PREFAC=1.333333333_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=2,PREFAC=-.166666666_q)
         CASE(3)
         ! Third order differentation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=1,PREFAC=1.5_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=2,PREFAC=-.3_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=3,PREFAC=0.033333333_q)
         CASE(4)
         ! Fourth order differentiation
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=1,PREFAC=1.6_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=2,PREFAC=-.4_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=3,PREFAC=0.076190476_q)
            CALL DPSI_DK_BERRY( &
           &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK,STRIDE=4,PREFAC=-.007142857_q)
         END SELECT
    
         IF (LPEAD_RETURN_Q_CPROJ) THEN
            CALL DPSI_DK_EV(W,K,ISP,DIR=IDIR,DWDK_DIR=DWDK)
         ENDIF

!        CALL DPSI_DK_ORTHO( &
!       &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,DIR=IDIR,DWDK_DIR=DWDK)
      
!        DWDK%CW(:,:,IK,ISP)=WEIGHT*DWDK%CW(:,:,IK,ISP)
!        DWDK%CPROJ(:,:,IK,ISP)=WEIGHT*DWDK%CPROJ(:,:,IK,ISP)
               
      ENDDO kpoint
      ENDDO spin     

      CALL STOP_TIMING("DPSIDK",UNIT6)

      RETURN
      END SUBROUTINE PEAD_DPSI_DK_IDIR


!******************** SUBROUTINE PEAD_POLARIZATION_CALC ****************
!
!***********************************************************************
      SUBROUTINE PEAD_POLARIZATION_CALC( &
     &   W,P,CQIJ,LATT_CUR,T_INFO,LINITIALIZE,LSKIP_IF_POSSIBLE)
      USE ini
      USE prec
      USE base
      USE wave
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE msymmetry
      USE mgrid
      USE nonl_high
      USE mkpoints      
      USE kpoints_change
      USE full_kpoints
      use mdipol

      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(latt) LATT_CUR
      LOGICAL, OPTIONAL :: LINITIALIZE
      LOGICAL, OPTIONAL :: LSKIP_IF_POSSIBLE
      OVERLAP CQIJ(:,:,:,:)
      ! local variables
      INTEGER I,J
      REAL(q) DIPOLE_DIRECT(3),RDUM
      LOGICAL LINIT,LSKIP
      
      LSKIP=.FALSE.
      IF (PRESENT(LSKIP_IF_POSSIBLE)) LSKIP=LSKIP_IF_POSSIBLE
      
      ! early exit if possible
      IF (.NOT.LPEAD) RETURN
      IF (LPEAD_NONZERO_EFIELD().AND.LSKIP) RETURN      
      IF (.NOT.LCALCPOL.AND..NOT.LPEAD_NONZERO_EFIELD()) RETURN

      LINIT=.FALSE.
      IF (PRESENT(LINITIALIZE)) LINIT=LINITIALIZE

      CALL START_TIMING("CALCP")
      
      CALL CALC_POLARIZATION(W,P,CQIJ,LATT_CUR,T_INFO,LINIT)
 
      CALL STOP_TIMING("CALCP",UNIT6)
      
      CALL POINT_CHARGE_DIPOL(T_INFO,LATT_CUR,P%ZVALF,POSCEN,DIPOLE_DIRECT,RDUM)
      IONIC_DIPOLE=0
      DO I=1,3
         DO J=1,3
            IONIC_DIPOLE(J)=IONIC_DIPOLE(J)+DIPOLE_DIRECT(I)*LATT_CUR%A(J,I)
         ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE PEAD_POLARIZATION_CALC


!******************** SUBROUTINE PEAD_POLARIZATION_WRITE ***************
!
! Writes macroscopic dipole moment to the OUTCAR and vaspxml files
!
!***********************************************************************
      SUBROUTINE PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING)
      USE prec
      USE base
      USE lattice
      USE vaspxml
      USE constant
      USE full_kpoints
      IMPLICIT NONE
      TYPE(in_struct) IO
      TYPE(latt) LATT_CUR
      LOGICAL, OPTIONAL :: LWARNING
      ! local variables      
      INTEGER I
      REAL(q) DK(3)
      REAL(q) RSUM(3),DRSUM(3)
      LOGICAL LWARN

      ! early exit if possible      
      IF (.NOT.LPEAD.OR..NOT.LDOIO) RETURN
      IF (.NOT.ALLOCATED(RSTORE)) RETURN
      ! first we write the ionic dipole moment (comes in handy sometimes)
      WRITE(IO%IU6,*)
      WRITE(IO%IU6,90) IONIC_DIPOLE
      CALL XML_VEC_REAL(IONIC_DIPOLE,'PION')
  90  FORMAT('            Ionic dipole moment: p[ion]=(',3F12.5,' ) electrons Angst')
      ! write spin resolved macroscopic dipole moment
      RSUM=0
      WRITE(IO%IU6,*)
      IF (SIZE(RSTORE,2)==2) THEN
         WRITE(IO%IU6,100) RSTORE(:,1)
         WRITE(IO%IU6,110) RSTORE(:,2)
         WRITE(IO%IU6,*)
         RSUM(:)=RSTORE(:,1)+RSTORE(:,2)
         CALL XML_VEC_REAL(RSTORE(:,1),'PSP1')
         CALL XML_VEC_REAL(RSTORE(:,2),'PSP2')
      ELSE
         RSUM(:)=RSTORE(:,1)
      ENDIF
      ! write total macroscopic dipole moment
      WRITE(IO%IU6,120) RSUM
      WRITE(IO%IU6,*)

      CALL XML_VEC_REAL(RSUM,'PELC')
      
 100  FORMAT('    Spin resolved dipole moment: p[sp1]=(',3F12.5,' ) electrons Angst')
 110  FORMAT('                                 p[sp2]=(',3F12.5,' ) electrons Angst')
 120  FORMAT(' Total electronic dipole moment: p[elc]=(',3F12.5,' ) electrons Angst')

      LWARN=.FALSE.
      IF (PRESENT(LWARNING)) LWARN=LWARNING
      ! write warning concerning spurious contributions
      IF (LWARN) THEN
         DK(1)=1.0_q/REAL(KPOINTS_FULL%NKPX,q)
         DK(2)=1.0_q/REAL(KPOINTS_FULL%NKPY,q)
         DK(3)=1.0_q/REAL(KPOINTS_FULL%NKPZ,q)
         WRITE(IO%IU6,'(A)',ADVANCE='No') '   in minimal image convention w.r.t.  '
         WRITE(IO%IU6,'(A,3F12.5,A)') &
           &   ' (',LATT_CUR%A(:,1)*(2/SIZE(RSTORE,2))/(KPOINTS_FULL%NKPTS*DK(1)),' )'
         DO I=2,3
            WRITE(IO%IU6,'(39X,A,3F12.5,A)') &
           &   ' (',LATT_CUR%A(:,I)*(2/SIZE(RSTORE,2))/(KPOINTS_FULL%NKPTS*DK(I)),' )'
         ENDDO
      ENDIF
      WRITE(IO%IU6,*)
      
      IF (.NOT.ALLOCATED(DRSTORE)) RETURN
      ! write spin resolved change in macroscopic dipole moment
      DRSUM=0
      IF (SIZE(RSTORE,2)==2) THEN
         WRITE(IO%IU6,200) DRSTORE(:,1)
         WRITE(IO%IU6,210) DRSTORE(:,2)
         WRITE(IO%IU6,*)
         DRSUM(:)=DRSTORE(:,1)+DRSTORE(:,2)
         CALL XML_VEC_REAL(DRSTORE(:,1),'DPSP1')
         CALL XML_VEC_REAL(DRSTORE(:,2),'DPSP2')
      ELSE
         DRSUM(:)=DRSTORE(:,1)
      ENDIF
      ! write total macroscopic dipole moment
      WRITE(IO%IU6,220) DRSUM
      WRITE(IO%IU6,*)

      CALL XML_VEC_REAL(DRSUM,'DPELC')
      
 200  FORMAT('                                dp[sp1]=(',3F12.5,' ) electrons Angst')
 210  FORMAT('                                dp[sp2]=(',3F12.5,' ) electrons Angst')
 220  FORMAT(' Total change in dipole moment: dp[elc]=(',3F12.5,' ) electrons Angst')

      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      ! write diagonal elements of the dielectric matrix
      WRITE(IO%IU6,300,ADVANCE='No')
      DO I=1,3
         IF (ABS(EFIELD_PEAD(I))>1E-5_q) THEN
            WRITE(IO%IU6,'(F12.5)',ADVANCE='No')  &
           &   EDEPS/LATT_CUR%OMEGA*DRSUM(I)/EFIELD_PEAD(I)+1
         ELSE
            WRITE(IO%IU6,'(A)',ADVANCE='No') '       ---- '
         ENDIF
      ENDDO         
      WRITE(IO%IU6,*) ')'
      WRITE(IO%IU6,*)
      
 300  FORMAT('                            diag[e(oo)]=(')

      RETURN
      END SUBROUTINE PEAD_POLARIZATION_WRITE


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_ACC_CALC(W,K,ISP,P,CQIJ,LATT_CUR,T_INFO)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      INTEGER ISP
      REAL(q) K(3)
      OVERLAP CQIJ(:,:,:,:)
      
      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      
      IF (.NOT.ALLOCATED(RPHI_STORE_LOCAL)) &
     &   ALLOCATE(RPHI_STORE_LOCAL(W%WDES%NRPLWV,W%WDES%NBANDS,3))
      
      IF (.NOT.ALLOCATED(RPHI_STORE_LOCAL_CPROJ)) &
     &   ALLOCATE(RPHI_STORE_LOCAL_CPROJ(W%WDES%NPROD,W%WDES%NBANDS,3))
      
      RPHI_STORE_LOCAL=0
      RPHI_STORE_LOCAL_CPROJ=0
      
      CALL PEAD_DPSI_DK_1K( &
     &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI_STORE_LOCAL,RPHI_STORE_LOCAL_CPROJ)
      
      RETURN
      END SUBROUTINE PEAD_ACC_CALC


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_ACC_CALC_ALL(W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE mkpoints
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE(kpoints_struct) KPOINTS
      OVERLAP CQIJ(:,:,:,:)
      
      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      ! do not recalculate RPHI if LPEAD_NOSCF=.TRUE. 
      IF (.NOT.LPEAD_SCF.AND..NOT.LFIRST_CALL_ACC_CALC) RETURN
      
      IF (.NOT.ALLOCATED(RPHI_STORE_LOCAL_ALL)) &
     &   ALLOCATE(RPHI_STORE_LOCAL_ALL(W%WDES%NRPLWV,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3))
      
      IF (.NOT.ALLOCATED(RPHI_STORE_LOCAL_ALL_CPROJ)) &
     &   ALLOCATE(RPHI_STORE_LOCAL_ALL_CPROJ(W%WDES%NPROD,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3))
      
      RPHI_STORE_LOCAL_ALL=0
      RPHI_STORE_LOCAL_ALL_CPROJ=0
      
      CALL PEAD_DPSI_DK_ALL( &
     &   W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,RPHI_STORE_LOCAL_ALL,RPHI_STORE_LOCAL_ALL_CPROJ)
      
      LFIRST_CALL_ACC_CALC=.FALSE.
 
      RETURN
      END SUBROUTINE PEAD_ACC_CALC_ALL


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************           
!     SUBROUTINE PEAD_ACC_ADD(CF,CPROF,NB)
!     USE prec
!     IMPLICIT NONE
!     INTEGER NB
!     COMPLEX(q) CF(:)
!     GDEF CPROF(:)
!     ! local variables
!     INTEGER IDIR
!
!     ! quick return if field is switched off
!     IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
!     ! quick return if the electric field is essentially zero
!     IF (.NOT.LPEADSCF()) RETURN 
!
!     DO IDIR=1,3
!        CF(:)=CF(:)+EFIELD(IDIR)*RPHI_STORE_LOCAL(:,NB,IDIR)
!        CPROF(:)=CPROF(:)+EFIELD(IDIR)*RPHI_STORE_LOCAL_CPROJ(:,NB,IDIR)         
!     ENDDO
!
!     RETURN
!     END SUBROUTINE PEAD_ACC_ADD


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_ACC_ADD(CF,CPROF,NB,NK,ISP)
      USE prec
      IMPLICIT NONE
      INTEGER NB,NK,ISP
      COMPLEX(q) CF(:)
      GDEF CPROF(:)
      ! local variables
      INTEGER IDIR,NMAX1,NMAX2

      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN

      NMAX1=SIZE(CF)
      IF (NMAX1>SIZE(RPHI_STORE_LOCAL_ALL,1)) THEN
         WRITE(*,*) 'PEAD_ACC_ADD: ERROR: RPHI_STORE_LOCAL_ALL too small:',NK
      ENDIF

      NMAX2=SIZE(CPROF)
      IF (NMAX2>SIZE(RPHI_STORE_LOCAL_ALL_CPROJ,1)) THEN
         WRITE(*,*) 'PEAD_ACC_ADD: ERROR: RPHI_STORE_LOCAL_ALL_CPROJ too small:',NK
      ENDIF

      DO IDIR=1,3
         CF(1:NMAX1)=CF(1:NMAX1)+EFIELD_PEAD(IDIR)*RPHI_STORE_LOCAL_ALL(1:NMAX1,NB,NK,ISP,IDIR)
         CPROF(1:NMAX2)=CPROF(1:NMAX2)+EFIELD_PEAD(IDIR)*RPHI_STORE_LOCAL_ALL_CPROJ(1:NMAX2,NB,NK,ISP,IDIR)         
      ENDDO

      RETURN
      END SUBROUTINE PEAD_ACC_ADD


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_ACC_ADD_PW(CF,NB,NK,ISP)
      USE prec
      IMPLICIT NONE
      INTEGER NB,NK,ISP
      COMPLEX(q) CF(:)
      ! local variables
      INTEGER IDIR

      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN

      DO IDIR=1,3
         CF(:)=CF(:)+EFIELD_PEAD(IDIR)*RPHI_STORE_LOCAL_ALL(:,NB,NK,ISP,IDIR)
      ENDDO

      RETURN
      END SUBROUTINE PEAD_ACC_ADD_PW


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_ACC_ADD_CPROJ(CPROF,NB,NK,ISP)
      USE prec
      IMPLICIT NONE
      INTEGER NB,NK,ISP
      GDEF CPROF(:)
      ! local variables
      INTEGER IDIR

      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN

      DO IDIR=1,3
         CPROF(:)=CPROF(:)+EFIELD_PEAD(IDIR)*RPHI_STORE_LOCAL_ALL_CPROJ(:,NB,NK,ISP,IDIR)         
      ENDDO

      RETURN
      END SUBROUTINE PEAD_ACC_ADD_CPROJ


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************      
      SUBROUTINE PEAD_ACC_DEALLOCATE()
      USE prec
      IMPLICIT NONE
      IF (ALLOCATED(RPHI_STORE_LOCAL_ALL)) DEALLOCATE(RPHI_STORE_LOCAL_ALL)
      IF (ALLOCATED(RPHI_STORE_LOCAL_ALL_CPROJ)) DEALLOCATE(RPHI_STORE_LOCAL_ALL_CPROJ)
      RETURN
      END SUBROUTINE PEAD_ACC_DEALLOCATE


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************      
      SUBROUTINE PEAD_EIGENVALUES(W,NK,ISP)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      INTEGER NK,ISP
      ! local variables
      TYPE(wavedes1) WDESK
      TYPE(wavefuna) WK
      OVERLAP , ALLOCATABLE :: CQIJ(:,:,:,:)
      INTEGER NB,IDIR
      INTEGER LMDIM,NPRO,NPRO_,ISPINOR,ISPINOR_,LMMAXC,NT,NI,NIS
      GDEF CNL
      
      ! quick return if field is switched off
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      ! or in the abscence of non-local contributions
      IF (.NOT.W%WDES%LOVERL .OR. .NOT.(W%WDES%NPROD>0) ) RETURN
      
      CALL SETWDES(W%WDES,WDESK,NK)
      WK=ELEMENTS(W,WDESK,ISP)
      
      ALLOCATE(CQIJ(SIZE(d_IJ,2),SIZE(d_IJ,3),SIZE(d_IJ,4),SIZE(d_IJ,5)))
      LMDIM = SIZE(d_IJ,2)


      bands: DO NB=1,WDESK%NBANDS 

      IF ((W%WDES%NB_LOW+(NB-1)*W%WDES%NB_PAR)>W%WDES%NB_TOTK(NK,ISP)) THEN
         W%CELEN(NB,NK,ISP)=10000._q
      ELSE
      
         CNL =0

         dir: DO IDIR=1,3

         CQIJ(:,:,:,:)=EFIELD_PEAD(IDIR)*d_IJ(IDIR,:,:,:,:)

         spinor: DO ISPINOR=0,WDESK%NRSPINORS-1
         DO ISPINOR_=0,WDESK%NRSPINORS-1

            NPRO =ISPINOR *(WDESK%NPRO/2)
            NPRO_=ISPINOR_*(WDESK%NPRO/2)

            NIS =1
            DO NT=1,WDESK%NTYP
               LMMAXC=WDESK%LMMAX(NT)
               IF (LMMAXC/=0) THEN
                  DO NI=NIS,WDESK%NITYP(NT)+NIS-1
                     CALL ECCP_NL(LMDIM,LMMAXC,CQIJ(1,1,NI,1+ISPINOR_+2*ISPINOR), &
                    &   WK%CPROJ(NPRO_+1,NB),WK%CPROJ(NPRO+1,NB),CNL)
                     NPRO = LMMAXC+NPRO
                     NPRO_= LMMAXC+NPRO_
                  ENDDO
               ENDIF
               NIS = NIS+WDESK%NITYP(NT)
            ENDDO
         ENDDO
         ENDDO spinor

         ENDDO dir

         ! communicate
         CALLMPI( M_sum_g(WDESK%COMM_INB, CNL, 1))
         ! and store
         W%CELEN(NB,NK,ISP)=W%CELEN(NB,NK,ISP)+CNL
         
      ENDIF
      
      ENDDO bands
      
      DEALLOCATE(CQIJ)

      RETURN
      END SUBROUTINE PEAD_EIGENVALUES


!****************** SUBROUTINE PEAD_EDOTP ******************************
!      
!***********************************************************************
      SUBROUTINE PEAD_EDOTP(W,P,CQIJ,LATT_CUR,T_INFO,E,LFORCE)
      USE ini
      USE prec
      USE base
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(latt) LATT_CUR
      TYPE(energy) E
      OVERLAP CQIJ(:,:,:,:)
      LOGICAL, OPTIONAL :: LFORCE
      ! local variables
      INTEGER I
      REAL(q) RSUM(3)
      LOGICAL LFRC
      
      LFRC=.FALSE.
      IF (PRESENT(LFORCE)) LFRC=LFORCE

      E%EDOTP=0
      
      ! early exits if possible
      IF (.NOT.LPEAD.OR..NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      IF (LSKIP_EDOTP_DURING_ELMIN().AND..NOT.LFRC) RETURN
      IF (.NOT.LSKIP_EDOTP_DURING_ELMIN().AND.LFRC) RETURN
      
      CALL START_TIMING("EDOTP")

      ! calculate the macroscopic polarization
      CALL CALC_POLARIZATION(W,P,CQIJ,LATT_CUR,T_INFO)

      ! interaction energy
      RSUM(:)=RSTORE(:,1)+R_EV_STORE(:,1)
      IF (W%WDES%ISPIN==2) RSUM(:)=RSUM(:)+RSTORE(:,2)+R_EV_STORE(:,2)
      DO I=1,3
         E%EDOTP=E%EDOTP-RSUM(I)*EFIELD_PEAD(I)
      ENDDO

      CALL STOP_TIMING("EDOTP",UNIT6)

      END SUBROUTINE PEAD_EDOTP


!******************** SUBROUTINE ***************************************
!
! A wrapper for CALC_OVERLAP
!      
!***********************************************************************
      SUBROUTINE PEAD_CALC_OVERLAP( &
     & W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S,LQIJB)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      REAL(q) K1(3),K2(3)
      INTEGER ISP
      OVERLAP CQIJ(:,:,:,:)
      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      LOGICAL LQIJB

      CALL CALC_OVERLAP(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S,LUSEQIJB=LQIJB)

      RETURN
      END SUBROUTINE PEAD_CALC_OVERLAP


!******************** SUBROUTINE ***************************************
!      
! Just a wrapper for WA_ROTATE
!
!***********************************************************************
      SUBROUTINE PEAD_WA_ROTATE(W,P,LATT_CUR,ISP,WA)
      USE prec
      USE pseudo
      USE poscar
      USE lattice
      USE wave_high
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(wavefuna) WA
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      INTEGER ISP
      
      CALL WA_ROTATE(W,P,LATT_CUR,ISP,WA)
      
      RETURN
      END SUBROUTINE PEAD_WA_ROTATE
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_SWITCH_EFIELD_ON()
      USE prec
      IMPLICIT NONE
      LFIELD_SWITCHED_ON=LPEAD_NONZERO_EFIELD().AND.LPEAD      
      RETURN
      END SUBROUTINE PEAD_SWITCH_EFIELD_ON


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_SWITCH_EFIELD_OFF()
      USE prec
      IMPLICIT NONE
      LFIELD_SWITCHED_ON=.FALSE.      
      RETURN
      END SUBROUTINE PEAD_SWITCH_EFIELD_OFF
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_CHECK_FIELD_CONDITION(W,LATT_CUR,IO)
      USE prec
      USE base
      USE lattice
      USE wave_high
      USE full_kpoints
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(latt) LATT_CUR
      TYPE(in_struct) IO
      ! local variables
      INTEGER ISP,IK,N,I
      REAL(q) DK(3)
      REAL(q) EHOMO,ELUMO,EGAP,ECRIT
      INTEGER IDUM
      REAL(q) RDUM,RTUT(6)
      COMPLEX(q) CDUM
      LOGICAL LDUM,LWARN

      LWARN=.FALSE.

      CALL CHECK_FULL_KPOINTS
      CALL CHECK_OCCUPATIONS(W)

      DK(1)=1.0_q/REAL(KPOINTS_FULL%NKPX,q)
      DK(2)=1.0_q/REAL(KPOINTS_FULL%NKPY,q)
      DK(3)=1.0_q/REAL(KPOINTS_FULL%NKPZ,q)
      
      ! get the HOMO
      EHOMO=-10000._q      
      DO ISP=1,W%WDES%ISPIN
         DO IK=1,SIZE(W%CELTOT,2)
            EHOMO=MAX(EHOMO,REAL(W%CELTOT(NOCC(ISP),IK,ISP),q))
         ENDDO
      ENDDO
      ! get the LUMO
      ELUMO=10000._q
      DO ISP=1,W%WDES%ISPIN
         IF (NOCC(ISP)==W%WDES%NB_TOT) CYCLE
         DO IK=1,SIZE(W%CELTOT,2)
            ELUMO=MIN(ELUMO,REAL(W%CELTOT(NOCC(ISP)+1,IK,ISP),q))
         ENDDO
      ENDDO      
      ! Eg=ELUMO-EHOMO
      IF (ELUMO<10000._q) THEN
         EGAP=ELUMO-EHOMO
         ! check Zener condition for critical field strength
         DO I=1,3
            ECRIT=ABS(EFIELD_PEAD(1)*LATT_CUR%A(1,I)+ &
           &           EFIELD_PEAD(2)*LATT_CUR%A(2,I)+ &
           &            EFIELD_PEAD(3)*LATT_CUR%A(3,I))
            RTUT(I)=ECRIT
            RTUT(I+3)=EGAP*DK(I)/10
            IF (ECRIT>EGAP*DK(I)/10) LWARN=.TRUE.
         ENDDO
         IF (LWARN) THEN
         ! write a warning that VASP consider the EFIELD_PEAD
         ! to be too large for comfort
         CALL VTUTOR('U','PEAD large field',RTUT,6,IDUM,1,CDUM,1,LDUM,1,IO%IU0,1)
         ENDIF
      ELSE
         ! write a warning that VASP cannot estimate whether
         ! the critical field strength is being exceeded
         CALL VTUTOR('U','PEAD no virtuals',RDUM,1,IDUM,1,CDUM,1,LDUM,1,IO%IU0,1)
      ENDIF
      
      RETURN
      END SUBROUTINE PEAD_CHECK_FIELD_CONDITION


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************            
      SUBROUTINE PEAD_SET_NB_TOTK(W)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      ! local variables
      INTEGER IK,ISP
      
      IF (.NOT.LPEAD) RETURN
      IF (.NOT.LPEAD_EFIELD_SWITCHED_ON()) RETURN
      
      CALL CHECK_OCCUPATIONS(W)
      IF (.NOT.IS_INSULATING()) RETURN
      
      DO ISP=1,W%WDES%ISPIN
      DO IK=1,W%WDES%NKPTS
         W%WDES%NB_TOTK(IK,ISP)=NOCC(ISP)
      ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE PEAD_SET_NB_TOTK


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_UNSET_NB_TOTK(W)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      ! local variables
      INTEGER IK,ISP
            
      DO ISP=1,W%WDES%ISPIN
      DO IK=1,W%WDES%NKPTS
         W%WDES%NB_TOTK(IK,ISP)=W%WDES%NB_TOT
      ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE PEAD_UNSET_NB_TOTK


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_RESET_EDWAV()
      USE prec
      IMPLICIT NONE
      LRESTART_EDWAV=.TRUE.
      RETURN
      END SUBROUTINE PEAD_RESET_EDWAV


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_REINIT_EDWAV(ICOUNT,GNORML,GNORM,AVERAG,BSTEP)
      USE prec
      IMPLICIT NONE
      INTEGER ICOUNT
      REAL(q) GNORML,GNORM,AVERAG,BSTEP
      IF (LRESTART_EDWAV) THEN
         ICOUNT=-1
         GNORML=0; GNORM=0; AVERAG=0; BSTEP=0
         LRESTART_EDWAV=.FALSE.      
      ENDIF
      RETURN
      END SUBROUTINE PEAD_REINIT_EDWAV


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_SWITCH_SCF_ON()
      USE prec
      IMPLICIT NONE
      LPEAD_SCF=LPEAD      
      RETURN
      END SUBROUTINE PEAD_SWITCH_SCF_ON


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE PEAD_SWITCH_SCF_OFF()
      USE prec
      IMPLICIT NONE
      LPEAD_SCF=.FALSE.
      RETURN
      END SUBROUTINE PEAD_SWITCH_SCF_OFF


!******************** SUBROUTINE ***************************************
!
! This subroutine is called from main.F and takes care of one or all
! of the following:
!
!  ) calculate the macroscopic dipole moment in the cell
!  ) calculate the SCF and/or non-SCF response to an electric
!    field specified through EFIELD_PEAD in the INCAR file
!  ) calculate the dielectric tensor excluding and/or
!    including local field effects
!
! the huge number of variables passed to this routine is unfortunately
! unavoidable on account of the fact that it should be able to call
! ELMIN_ALL and FORCE_AND_STRESS :-(
!
!***********************************************************************
      SUBROUTINE PEAD_ELMIN( &
     &   HAMILTONIAN,KINEDEN, &
     &   P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
     &   T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
     &   GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
     &   CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
     &   CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
     &   CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV, &
     &   NSTEP,LMDIM,IRDMAX,NEDOS, &
     &   TOTEN,EFERMI,LDIMP,LMDIMP )
      USE us
      USE prec
      USE base
      USE mgrid
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE mkpoints
      USE nonl_high
      USE wave_high
      USE hamil_high
      USE msymmetry
      USE vaspxml
      USE pot
      USE pawm
      USE fock
      USE meta
      IMPLICIT NONE
      TYPE (ham_handle) HAMILTONIAN
      TYPE (tau_handle) KINEDEN
      TYPE (energy) E
      TYPE (mixing) MIX
      TYPE (wavedes) WDES
      TYPE (dynamics) DYN
      TYPE (in_struct) IO
      TYPE (symmetry) SYMM
      TYPE (info_struct) INFO
      TYPE (type_info) T_INFO
      TYPE (wavespin) W,W_F,W_G
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (latt) LATT_CUR,LATT_INI
      TYPE (kpoints_struct) KPOINTS
      TYPE (transit) B_TO_C,C_TO_US,SOFT_TO_C
      TYPE (grid_3d) GRID,GRID_SOFT,GRIDC,GRIDUS,GRIDB

      INTEGER N_MIX_PAW
      INTEGER LDIMP,LMDIMP
      INTEGER NSTEP,LMDIM,IRDMAX,NEDOS
      REAL(q) ECONV,TOTEN,EFERMI
      REAL(q) DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
      REAL(q) RHOLM(N_MIX_PAW,WDES%NCDIJ),RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)

      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CHTOTL(GRIDC%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CVTOT(GRIDC%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      COMPLEX(q) CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      RGRID DENCOR(GRIDC%RL%NP)
      RGRID SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)

      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
     &   CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
     &   CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      GDEF , POINTER :: CHF(:,:,:,:),CHAM(:,:,:,:)

      ! symmetry related quantities (common block)
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &   GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      
      ! local variables
      INTEGER I,J,N
      INTEGER IDIR
      REAL(q) XCSIF(3,3),PRESS
      REAL(q) TIFOR(3,T_INFO%NIOND),TSIF(3,3)
      REAL(q) EWFOR(3,T_INFO%NIOND),EWSIF(3,3)
      REAL(q) EWIFOR(3,T_INFO%NIOND),EWISIF(3,3)
      REAL(q) F_INIT(3,T_INFO%NIOND),SIF_INIT(3,3)
      REAL(q) BORN_NSCF(3,3,T_INFO%NIOND),BORN_SCF(3,3,T_INFO%NIOND)
      REAL(q) PIEZO_NSCF(3,3,3),PIEZO_SCF(3,3,3)
      REAL(q) DRSCF(3,3),DRNSCF(3,3)
      REAL(q) EFIELD_STORE(3),DE
      REAL(q) FACT
      CHARACTER(3) :: IDIR_TEXT(3)=(/"x","y","z"/)
      INTEGER :: IRDMAA
      INTEGER :: IALGO_SAVE
      LOGICAL :: LDIAG_SAVE,LSUBROT_SAVE
      REAL(q) :: EDIFF_SAVE
!=======================================================================            
! calculate macroscopic polarization
!=======================================================================
      IF (LPEAD_CALC_POL()) THEN
         IF (.NOT.IS_INSULATING()) THEN
            CALL CHECK_OCCUPATIONS(W)
            IF (.NOT.IS_INSULATING()) THEN
               CALL VTUTOR('W','PEADPOLNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
               CALL VTUTOR('W','PEADPOLNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
               RETURN
            ENDIF
         ENDIF
         CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO)
!        CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
         CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)
      ENDIF
      
!=======================================================================
! calculate the dielectric matrix
!=======================================================================
      IF (LPEAD_CALC_EPS()) THEN
         IF (.NOT.IS_INSULATING()) THEN
            CALL CHECK_OCCUPATIONS(W)
            IF (.NOT.IS_INSULATING()) THEN
               CALL VTUTOR('W','PEADEPSNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
               CALL VTUTOR('W','PEADEPSNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
               RETURN
            ENDIF
         ENDIF

         DRSCF=0 ; DRNSCF=0
         BORN_SCF=0 ; BORN_NSCF=0
         PIEZO_SCF=0 ; PIEZO_NSCF=0
         
         EFIELD_STORE=EFIELD_PEAD

         ! if INFO%LONESW=.FALSE. choose the damped all-bands-simultaneous algorithm
         ! and switch off the scf subspace preconditioning
         IALGO_SAVE=INFO%IALGO ; LDIAG_SAVE=INFO%LDIAG ; LSUBROT_SAVE=INFO%LSUBROT
         EDIFF_SAVE=INFO%EDIFF
         IF (.NOT.INFO%LONESW) INFO%IALGO=3
         INFO%LDIAG=.FALSE. ; INFO%LSUBROT=.FALSE.
         INFO%EDIFF=INFO%EDIFF*1E-2

         DO IDIR=1,3
!-----------------------------------------------------------------------
! Reduce symmetry
!-----------------------------------------------------------------------
            ! set cartesian component IDIR of EFIELD_PEAD
            EFIELD_PEAD=0
            IF (ABS(EFIELD_STORE(IDIR))>1.E-5_q) THEN
               EFIELD_PEAD(IDIR)=EFIELD_STORE(IDIR)
            ELSE
               EFIELD_PEAD(IDIR)=0.01
            ENDIF
            ! switch on the electric field
            CALL PEAD_SWITCH_EFIELD_ON()
            ! check the Zener field condition
            CALL PEAD_CHECK_FIELD_CONDITION(W,LATT_CUR,IO)
            ! reduce the symmetry in accordance with the
            ! electric field and reallocate the wave functions
            CALL PEAD_RESYMMETRIZE(KPOINTS,WDES,NONLR_S,NONL_S,W,W_F,W_G,CHAM,CHF, &
           &    GRID,P,T_INFO,INFO,LATT_CUR,LATT_INI,DYN,SYMM,IO)

            ! no delay before selfconsistency 
            IF (INFO%NELMDL<0) INFO%NELMDL=0
             
            ! reset EDWAV (rot.F)
            CALL PEAD_RESET_EDWAV()
            W_F%CELTOT=W%CELTOT

            ! temporarily switch the electric field off
            CALL PEAD_SWITCH_EFIELD_OFF()

            ! optimize wavefunctions at added kpoints
            CALL ELMIN_ALL( &
                HAMILTONIAN,KINEDEN, &
                P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
                T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
                CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
                NSTEP,LMDIM,IRDMAX,NEDOS, &
                TOTEN,EFERMI,LDIMP,LMDIMP)

            ! switch the electric field back on
            CALL PEAD_SWITCH_EFIELD_ON()
            ! store the wave functions       
            CALL PEAD_STORE_WAVE_ORIG(W)

            ! calculate the initial polarization
            CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LINITIALIZE=.TRUE.)
!           CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
            CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)

            ! calculate the initial forces and stress
            EWIFOR=0 ; EWISIF=0
            CALL FORCE_AND_STRESS( &
                KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
                T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM_LAST,RHOLM_LAST, &
                CHDEN,SV,LMDIM,IRDMAX, &
                .FALSE.,DYN%ISIF/=0,.FALSE.,.FALSE., &
                XCSIF,EWISIF,TSIF,EWIFOR,TIFOR,PRESS, TOTEN)
            ! store the initial forces and stress
            F_INIT=TIFOR
            SIF_INIT=TSIF
            
            ! limit the optimization of the wave functions
            ! to the occupied states only
            CALL PEAD_SET_NB_TOTK(W)
            ! reset ICOUNT in EDWAV (rot.F)
            CALL PEAD_RESET_EDWAV()
            ! force calculation of the accelerations in the first step
            LFIRST_CALL_ACC_CALC=.TRUE.

!-----------------------------------------------------------------------         
! calculate the non-selfconsistent response to the electric field
!-----------------------------------------------------------------------         
            IF (LNSCF_RESPONSE()) THEN
            INFO%LCHCON=.TRUE.
            CALL PEAD_SWITCH_SCF_OFF()

            EWFOR=0 ; EWSIF=0
            CALL CALC_EWFOR_AND_EWSIF( &
           &   W,KPOINTS,P,CQIJ,CRHODE,NONLR_S,NONL_S,LATT_CUR,T_INFO,DYN,SYMM,IDIR, &
           &   EWFOR,EWSIF)        

            ! For the non-selfconsistent response we always
            ! use the damped all-bands-simultaneous algorithm
            INFO%IALGO=3

            CALL ELMIN_ALL( &
                HAMILTONIAN,KINEDEN, &
                P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
                T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
                CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
                NSTEP,LMDIM,IRDMAX,NEDOS, &
                TOTEN,EFERMI,LDIMP,LMDIMP)

            ! switch back to the user specified algorithm
            IF (INFO%LONESW) INFO%IALGO=IALGO_SAVE

            ! calculate and write the polarization
            CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO)
!           CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LSKIP_IF_POSSIBLE=.TRUE.)
!           CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
            CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)

            ! store DRSTORE into DRNSCF(IDIR)
            DRNSCF(:,IDIR)=DRSTORE(:,1)
            IF (W%WDES%ISPIN==2) DRNSCF(:,IDIR)=DRNSCF(:,IDIR)+DRSTORE(:,2)

            ! calculate the forces and stress
            EWIFOR=0 ; EWISIF=0
            CALL FORCE_AND_STRESS( &
                KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
                T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
                CHDEN,SV,LMDIM,IRDMAX, &
                .FALSE.,DYN%ISIF/=0,.FALSE.,.FALSE., &
                XCSIF,EWISIF,TSIF,EWIFOR,TIFOR,PRESS, TOTEN)

            BORN_NSCF(IDIR,:,:)=(TIFOR(:,:)-F_INIT(:,:))/EFIELD_PEAD(IDIR)
            BORN_NSCF(IDIR,:,:)=-(BORN_NSCF(IDIR,:,:)+EWFOR(:,:))

            PIEZO_NSCF(IDIR,:,:)=(TSIF(:,:)-SIF_INIT(:,:))/EFIELD_PEAD(IDIR)
            PIEZO_NSCF(IDIR,:,:)=PIEZO_NSCF(IDIR,:,:)+EWSIF(:,:)

            ENDIF

!-----------------------------------------------------------------------         
! calculate the selfconsistent response to the electric field
!-----------------------------------------------------------------------         
            IF (LSCF_RESPONSE()) THEN
            INFO%LCHCON=.FALSE.
            CALL PEAD_SWITCH_SCF_ON()

            IF (LNSCF_RESPONSE()) THEN
               ! restore the original wave functions
               CALL PEAD_RESTORE_WAVE_ORIG(W)
               W_F%CELTOT=W%CELTOT
            ENDIF

            IF (INFO%LREAL) CALL RSPHER(GRID,NONLR_S,LATT_CUR)

            CALL ELMIN_ALL( &
                HAMILTONIAN,KINEDEN, &
                P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
                T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
                CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
                NSTEP,LMDIM,IRDMAX,NEDOS, &
                TOTEN,EFERMI,LDIMP,LMDIMP)

            ! calculate and write the polarization
            CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO)
!           CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LSKIP_IF_POSSIBLE=.TRUE.)
!           CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
            CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)

            ! store DRSTORE into DRSCF(IDIR)
            DRSCF(:,IDIR)=DRSTORE(:,1)
            IF (W%WDES%ISPIN==2) DRSCF(:,IDIR)=DRSCF(:,IDIR)+DRSTORE(:,2)

            ! calculate the forces and stress
            EWIFOR=0 ; EWISIF=0
            CALL FORCE_AND_STRESS( &
                KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
                T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
                GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
                CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
                CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
                CHDEN,SV,LMDIM,IRDMAX, &
                .FALSE.,DYN%ISIF/=0,.FALSE.,.FALSE., &
                XCSIF,EWISIF,TSIF,EWIFOR,TIFOR,PRESS, TOTEN)

            EWFOR=0 ; EWSIF=0
            CALL CALC_EWFOR_AND_EWSIF( &
           &   W,KPOINTS,P,CQIJ,CRHODE,NONLR_S,NONL_S,LATT_CUR,T_INFO,DYN,SYMM,IDIR, &
           &   EWFOR,EWSIF)

            BORN_SCF(IDIR,:,:)=(TIFOR(:,:)-F_INIT(:,:))/EFIELD_PEAD(IDIR)
            BORN_SCF(IDIR,:,:)=-(BORN_SCF(IDIR,:,:)+EWFOR(:,:))

            PIEZO_SCF(IDIR,:,:)=(TSIF(:,:)-SIF_INIT(:,:))/EFIELD_PEAD(IDIR)
            PIEZO_SCF(IDIR,:,:)=PIEZO_SCF(IDIR,:,:)+EWSIF(:,:)

            ENDIF

!-----------------------------------------------------------------------
! Restore unperturbed solution
!-----------------------------------------------------------------------
            CALL PEAD_ACC_DEALLOCATE
            ! restore the original wave functions
            CALL PEAD_RESTORE_WAVE_ORIG(W)
            CALL DEALLOCW(W_STORE)
            ! reset NB_TOTK to include all bands
            CALL PEAD_UNSET_NB_TOTK(W)

            ! restore the original symmetry
            EFIELD_PEAD=0._q
            CALL PEAD_RESYMMETRIZE(KPOINTS,WDES,NONLR_S,NONL_S,W,W_F,W_G,CHAM,CHF, &
           &    GRID,P,T_INFO,INFO,LATT_CUR,LATT_INI,DYN,SYMM,IO)

            ! recalculate charge density
            CALL DEPSUM(W,WDES,LMDIM,CRHODE,INFO%LOVERL)
            CALL US_FLIP(WDES,LMDIM,CRHODE,INFO%LOVERL,.FALSE.)
            CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
               GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
               LATT_CUR, P, SYMM, T_INFO, &
               CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

            CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
                  INFO,P,T_INFO,E,LATT_CUR, &
                  CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
            CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                 LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

            CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
                 WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
                 E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

         ENDDO

!-----------------------------------------------------------------------
! Write output
!-----------------------------------------------------------------------         
         ! symmetrize DRNSCF, DRSCF
         IF (SYMM%ISYM>0) THEN
            IF (LNSCF_RESPONSE()) THEN
               CALL TSYM(DRNSCF,ISYMOP,NROTK,LATT_CUR%A)
               CALL TENSYM(BORN_NSCF, &
              &   SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,LATT_CUR%A)
            ENDIF
            IF (LSCF_RESPONSE()) THEN
               CALL TSYM(DRSCF, ISYMOP,NROTK,LATT_CUR%A)
               CALL TENSYM(BORN_SCF, &
              &   SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,LATT_CUR%A)
            ENDIF
         ENDIF 
         
         ! calculate EPSILON_NSCF and EPSILON_SCF
         DO I=1,3
            IF (ABS(EFIELD_STORE(I))>1.E-5_q) THEN
               DE=EFIELD_STORE(I)
            ELSE
               DE=0.01
            ENDIF
            DO J=1,3
               EPSILON_NSCF(I,J)=EDEPS/LATT_CUR%OMEGA*DRNSCF(J,I)/DE
               EPSILON_SCF(I,J) =EDEPS/LATT_CUR%OMEGA*DRSCF(J,I)/DE
               IF (I==J) THEN
                  EPSILON_NSCF(I,J)=EPSILON_NSCF(I,J)+1
                  EPSILON_SCF(I,J) =EPSILON_SCF(I,J) +1
               ENDIF
            ENDDO
         ENDDO

         ! write EPSILON_NSCF and EPSILON_SCF to
         ! OUTCAR and vasprun.xml
         IF (LDOIO) THEN            
            IF (LNSCF_RESPONSE()) &
           &   WRITE(IO%IU6,100) '(excluding local field effects)',EPSILON_NSCF
            IF (LSCF_RESPONSE()) &
           &   WRITE(IO%IU6,110) '(including local field effects)',EPSILON_SCF

  100       FORMAT(// &
           &  " MICROSCOPIC STATIC DIELECTRIC TENSOR ",A/, &
           &  " -------------------------------------"/, &
           &       3(6X,3F10.3/), &
           &  " -------------------------------------"/)

  110       FORMAT(// &
           &  " MACROSCOPIC STATIC DIELECTRIC TENSOR ",A/, &
           &  " -------------------------------------"/, &
           &       3(6X,3F10.3/), &
           &  " -------------------------------------"/)
            
            IF (LNSCF_RESPONSE()) CALL XML_TENSOR("epsilon_nscf",EPSILON_NSCF)
            IF (LSCF_RESPONSE())  CALL XML_TENSOR("epsilon_scf" ,EPSILON_SCF )
         ENDIF
         
         ! write Born effective charge tensors to
         ! OUTCAR and vasprun.xml
         IF (LDOIO) THEN
            IF (LNSCF_RESPONSE()) THEN
               WRITE (IO%IU6,*)
               WRITE (IO%IU6,*) 'BORN EFFECTIVE CHARGES (excluding local field effects)'
               WRITE (IO%IU6,*) '------------------------------------------------------'

               DO N=1,T_INFO%NIONS
                  WRITE (IO%IU6,'(" ion ",I4)') N
                  DO IDIR=1,3
                     WRITE (IO%IU6,'(I5,3F12.5)') IDIR,BORN_NSCF(IDIR,:,N)
                  ENDDO
               ENDDO
            ENDIF
            IF (LSCF_RESPONSE()) THEN
               WRITE (IO%IU6,*)
               WRITE (IO%IU6,*) 'BORN EFFECTIVE CHARGES (including local field effects)'
               WRITE (IO%IU6,*) '------------------------------------------------------'

               DO N=1,T_INFO%NIONS
                  WRITE (IO%IU6,'(" ion ",I4)') N
                  DO IDIR=1,3
                     WRITE (IO%IU6,'(I5,3F12.5)') IDIR,BORN_SCF(IDIR,:,N)
                  ENDDO
               ENDDO
               CALL XML_BORN_CHARGES(BORN_SCF,T_INFO%NIONS)
               ! set Born effective charges in global array
               IF (.NOT. LBORN) THEN
                  LBORN=.TRUE.
                  ALLOCATE(BORN_CHARGES_PEAD(3,3,T_INFO%NIOND))
               ENDIF
               BORN_CHARGES_PEAD=BORN_SCF
            ENDIF
         ENDIF
         
         ! write PIEZO electric tensor to OUTCAR and vasprun.xml
         IF (LDOIO) THEN
            FACT=EVTOJ*1E20_q/LATT_CUR%OMEGA
            IF (LNSCF_RESPONSE()) THEN
               CALL TSYM3(PIEZO_NSCF,ISYMOP,NROTK,LATT_CUR%A)
               WRITE(IO%IU6,200) '(excluding local field effects) (e Angst)'
               DO I=1,3
                  WRITE(IO%IU6,210) IDIR_TEXT(I),(PIEZO_NSCF(I,J,J),J=1,3), & 
                 &   PIEZO_NSCF(I,1,2),PIEZO_NSCF(I,2,3),PIEZO_NSCF(I,3,1)
               ENDDO
               WRITE(IO%IU6,200) '(excluding local field effects) (C/m^2)'
               DO I=1,3
                  WRITE(IO%IU6,210) IDIR_TEXT(I),(PIEZO_NSCF(I,J,J)*FACT,J=1,3), & 
                 &   PIEZO_NSCF(I,1,2)*FACT,PIEZO_NSCF(I,2,3)*FACT,PIEZO_NSCF(I,3,1)*FACT
               ENDDO
            ENDIF
            IF (LSCF_RESPONSE()) THEN
               CALL TSYM3(PIEZO_SCF,ISYMOP,NROTK,LATT_CUR%A)
               WRITE(IO%IU6,200) '(including local field effects) (e Angst)'
               DO I=1,3
                  WRITE(IO%IU6,210) IDIR_TEXT(I),(PIEZO_SCF(I,J,J),J=1,3), & 
                 &   PIEZO_SCF(I,1,2),PIEZO_SCF(I,2,3),PIEZO_SCF(I,3,1)
               ENDDO
               WRITE(IO%IU6,200) '(including local field effects) (C/m^2)'
               DO I=1,3
                  WRITE(IO%IU6,210) IDIR_TEXT(I),(PIEZO_SCF(I,J,J)*FACT,J=1,3), & 
                 &   PIEZO_SCF(I,1,2)*FACT,PIEZO_SCF(I,2,3)*FACT,PIEZO_SCF(I,3,1)*FACT
               ENDDO
            ENDIF

  200       FORMAT(/ ' PIEZOELECTRIC TENSOR ',A,/ &
           &   10X,'XX', 10X,'YY', 10X,'ZZ',10X,'XY', 10X,'YZ', 10X,'ZX'/ &
           &   '  --------------------------------------------------------------------------------')
  210       FORMAT(2X,A1,6F12.5)
         ENDIF

         INFO%IALGO=IALGO_SAVE ; INFO%LDIAG=LDIAG_SAVE ; INFO%LSUBROT=LSUBROT_SAVE
         INFO%EDIFF=EDIFF_SAVE
         RETURN
      ENDIF

!=======================================================================
! response to electric field
!=======================================================================
      IF (LPEAD_NONZERO_EFIELD()) THEN         
         IF (.NOT.IS_INSULATING()) THEN
            CALL CHECK_OCCUPATIONS(W)
            IF (.NOT.IS_INSULATING()) THEN
               CALL VTUTOR('W','PEADEPSNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
               CALL VTUTOR('W','PEADEPSNOMETAL',0.0_q,1, &
               &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
               RETURN
            ENDIF
         ENDIF
         ! switch on the electric field
         CALL PEAD_SWITCH_EFIELD_ON()
         ! check the Zener field condition
         CALL PEAD_CHECK_FIELD_CONDITION(W,LATT_CUR,IO)
         ! reduce the symmetry in accordance with the
         ! electric field and reallocate the wave functions
         CALL PEAD_RESYMMETRIZE(KPOINTS,WDES,NONLR_S,NONL_S,W,W_F,W_G,CHAM,CHF, &
        &    GRID,P,T_INFO,INFO,LATT_CUR,LATT_INI,DYN,SYMM,IO)

         ! if INFO%LONESW=.FALSE. choose the damped all-bands-simultaneous algorithm
         ! and switch off the scf subspace preconditioning
         IALGO_SAVE=INFO%IALGO ; LDIAG_SAVE=INFO%LDIAG ; LSUBROT_SAVE=INFO%LSUBROT
         EDIFF_SAVE=INFO%EDIFF
         IF (.NOT.INFO%LONESW) INFO%IALGO=3
         INFO%LDIAG=.FALSE. ; INFO%LSUBROT=.FALSE.
         INFO%EDIFF=INFO%EDIFF*1E-2

         ! no delay before selfconsistency 
         IF (INFO%NELMDL<0) INFO%NELMDL=0

         ! reset EDWAV (rot.F)
         CALL PEAD_RESET_EDWAV()
         W_F%CELTOT=W%CELTOT

         ! temporarily switch the electric field off
         CALL PEAD_SWITCH_EFIELD_OFF()

         ! optimize wavefunctions at added kpoints
         CALL ELMIN_ALL( &
             HAMILTONIAN,KINEDEN, &
             P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
             T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
             GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
             CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
             CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
             CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
             NSTEP,LMDIM,IRDMAX,NEDOS, &
             TOTEN,EFERMI,LDIMP,LMDIMP)

         ! switch the electric field back on
         CALL PEAD_SWITCH_EFIELD_ON()
         ! store the wave functions       
         CALL PEAD_STORE_WAVE_ORIG(W)
         
         ! calculate the initial polarization
         CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LINITIALIZE=.TRUE.)
!        CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
         CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)

         ! limit the optimization of the wave functions
         ! to the occupied states only
         CALL PEAD_SET_NB_TOTK(W)
         ! reset ICOUNT in EDWAV (rot.F)
         CALL PEAD_RESET_EDWAV()

         LFIRST_CALL_ACC_CALC=.TRUE.

         IF (LNSCF_RESPONSE()) THEN
         ! calculate the non-selfconsistent response to the electric field
         INFO%LCHCON=.TRUE.
         CALL PEAD_SWITCH_SCF_OFF()

         ! For the non-selfconsistent response we always
         ! use the damped all-bands-simultaneous algorithm
         INFO%IALGO=3

         CALL ELMIN_ALL( &
             HAMILTONIAN,KINEDEN, &
             P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
             T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
             GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
             CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
             CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
             CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
             NSTEP,LMDIM,IRDMAX,NEDOS, &
             TOTEN,EFERMI,LDIMP,LMDIMP)

         ! switch back to the user specified algorithm
         IF (INFO%LONESW) INFO%IALGO=IALGO_SAVE

         ! calculate and write the polarization
         CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO)
!        CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LSKIP_IF_POSSIBLE=.TRUE.)
!        CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
         CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)
         ENDIF

         IF (LSCF_RESPONSE()) THEN
         ! calculate the selfconsistent response to the electric field
         INFO%LCHCON=.FALSE.
         CALL PEAD_SWITCH_SCF_ON()

         IF (LNSCF_RESPONSE()) THEN
            ! restore the original wave functions
            CALL PEAD_RESTORE_WAVE_ORIG(W)
            W_F%CELTOT=W%CELTOT
         ENDIF

         CALL ELMIN_ALL( &
             HAMILTONIAN,KINEDEN, &
             P,WDES,NONLR_S,NONL_S,W,W_F,W_G,LATT_CUR,LATT_INI, &
             T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
             GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
             CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
             CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
             CHDEN,SV,DOS,DOSI,CHF,CHAM,ECONV,XCSIF, &
             NSTEP,LMDIM,IRDMAX,NEDOS, &
             TOTEN,EFERMI,LDIMP,LMDIMP)

         ! calculate and write the polarization
         CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO)
!        CALL PEAD_POLARIZATION_CALC(W,P,CQIJ,LATT_CUR,T_INFO,LSKIP_IF_POSSIBLE=.TRUE.)
!        CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR,LWARNING=.TRUE.)
         CALL PEAD_POLARIZATION_WRITE(IO,LATT_CUR)
         ENDIF

         CALL PEAD_ACC_DEALLOCATE
         ! restore the original wave functions
         CALL PEAD_RESTORE_WAVE_ORIG(W)
         CALL DEALLOCW(W_STORE)

         ! restore the original symmetry
         EFIELD_PEAD=0._q
         CALL PEAD_RESYMMETRIZE(KPOINTS,WDES,NONLR_S,NONL_S,W,W_F,W_G,CHAM,CHF, &
        &    GRID,P,T_INFO,INFO,LATT_CUR,LATT_INI,DYN,SYMM,IO)

         ! recalculate charge density
         CALL DEPSUM(W,WDES,LMDIM,CRHODE,INFO%LOVERL)
         CALL US_FLIP(WDES,LMDIM,CRHODE,INFO%LOVERL,.FALSE.)
         CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
            GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
            LATT_CUR, P, SYMM, T_INFO, &
            CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

         CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
              INFO,P,T_INFO,E,LATT_CUR, &
              CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
         CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                  LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

         CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
              WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
              E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

         INFO%IALGO=IALGO_SAVE ; INFO%LDIAG=LDIAG_SAVE ; INFO%LSUBROT=LSUBROT_SAVE
         INFO%EDIFF=EDIFF_SAVE
      ENDIF
            
      RETURN
      END SUBROUTINE PEAD_ELMIN


!***********************************************************************
!******************** PUBLIC QUERY FUNCTIONS ***************************
!***********************************************************************


!***********************************************************************
!
! Query function, LUSEPEAD=.TRUE. if LPEAD=.TRUE. in the INCAR file
!
!***********************************************************************
      FUNCTION LUSEPEAD()
      IMPLICIT NONE
      LOGICAL LUSEPEAD
      LUSEPEAD=LPEAD.OR.EXTERNAL_REQUEST_FOR_PEAD
      END FUNCTION LUSEPEAD


!***********************************************************************
!
! LPEAD_NONZERO_EFIELD is true when EFIELD_PEAD is nonzero
!
!***********************************************************************
      FUNCTION LPEAD_NONZERO_EFIELD()
      IMPLICIT NONE
      LOGICAL LPEAD_NONZERO_EFIELD
      LPEAD_NONZERO_EFIELD= &
     &   (SQRT(EFIELD_PEAD(1)*EFIELD_PEAD(1)+ &
     &          EFIELD_PEAD(2)*EFIELD_PEAD(2)+ &
     &           EFIELD_PEAD(3)*EFIELD_PEAD(3))>=1.E-5_q)
      END FUNCTION LPEAD_NONZERO_EFIELD


!***********************************************************************
!
! LPEAD_EFIELD_SWITCHED_ON is true when the electric field is
! switched on 
!
!***********************************************************************
      FUNCTION LPEAD_EFIELD_SWITCHED_ON()
      IMPLICIT NONE
      LOGICAL LPEAD_EFIELD_SWITCHED_ON
      LPEAD_EFIELD_SWITCHED_ON=LFIELD_SWITCHED_ON
      END FUNCTION LPEAD_EFIELD_SWITCHED_ON


!***********************************************************************
!
! LPEAD_CALC_POL is true when VASP should calculate the
! macroscopic polarization after electronic convergence
! has been attained
!
!***********************************************************************
      FUNCTION LPEAD_CALC_POL()
      IMPLICIT NONE
      LOGICAL LPEAD_CALC_POL
      LPEAD_CALC_POL=LCALCPOL
      END FUNCTION LPEAD_CALC_POL


!***********************************************************************
!
! LPEAD_CALC_EPS is true when VASP should calculate the
! complete dielectric matrix after electronic convergence
! has been attained
!
!***********************************************************************
      FUNCTION LPEAD_CALC_EPS()
      IMPLICIT NONE
      LOGICAL LPEAD_CALC_EPS
      LPEAD_CALC_EPS=LCALCEPS
      END FUNCTION LPEAD_CALC_EPS


!***********************************************************************
!
! LPEAD_NO_SCF is true when VASP should perform a non-selfconsistent
! calculation of the response of the wave functions to the electric
! field
!
!***********************************************************************
      FUNCTION LPEAD_NO_SCF()
      IMPLICIT NONE
      LOGICAL LPEAD_NO_SCF
      LPEAD_NO_SCF=.NOT.LPEAD_SCF.AND.LPEAD
      END FUNCTION LPEAD_NO_SCF


!***********************************************************************
!
! LPEAD_ABORT
!
!***********************************************************************
      FUNCTION LPEAD_ABORT(INFO,RMS)
      USE prec
      USE base
      IMPLICIT NONE
      TYPE (info_struct) INFO
      REAL(q) RMS
      LOGICAL LPEAD_ABORT
      IF (LSKIP_EDOTP_DURING_ELMIN().AND.LPEAD_EFIELD_SWITCHED_ON()) THEN
         LPEAD_ABORT=(ABS(RMS)<INFO%EDIFF)
      ELSE
         LPEAD_ABORT=INFO%LABORT
      ENDIF
      END FUNCTION LPEAD_ABORT


!***********************************************************************
!
! PEAD_REQUEST will set EXTERNAL_REQUEST_FOR_PEAD=.TRUE., which in turn
! will ensure that LPEAD=.TRUE. regardless of what is specified in the 
! INCAR. This way the routines in mlwf.F and wnpr.F can switch on the
! PEAD routines.
!
!***********************************************************************
      SUBROUTINE PEAD_REQUEST
      EXTERNAL_REQUEST_FOR_PEAD=.TRUE.
      RETURN
      END SUBROUTINE PEAD_REQUEST


!***********************************************************************
!******************** PRIVATE PROCEDURES *******************************
!***********************************************************************


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE GENERATE_KPOINTS_TRANS_FULL(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,IU6) 
      USE prec
      USE mgrid
      USE lattice
      USE mkpoints
      USE wave_high
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(grid_3d), TARGET :: GRID
      TYPE(latt) LATT_CUR
      TYPE(latt) LATT_INI
      TYPE(kpoints_struct) KPOINTS
      INTEGER IU6
      ! local variables
!     TYPE(wavedes) WDES_TMP
      TYPE(grid_3d) GRID_TMP
      LOGICAL, SAVE :: LFIRST=.TRUE. 
      
      IF (.NOT.LFIRST) THEN
         CALL DEALLOCWDES(WDES_FULL_PEAD,LEXTEND=.TRUE.)
         CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS_FULL_PEAD)
      ENDIF

      ALLOCATE(WDES_FULL_PEAD)

      ! copy the loop counters and dimensions from GRID to GRID_TMP
      GRID_TMP=GRID
      ! copy all data from WDES to WDES_TMP      
      WDES_FULL_PEAD=WDES
      WDES_FULL_PEAD%NKDIM =KPOINTS_FULL%NKPTS
      WDES_FULL_PEAD%NKPTS =KPOINTS_FULL%NKPTS
      WDES_FULL_PEAD%NKPTS_FOR_GEN_LAYOUT=KPOINTS_FULL%NKPTS
      WDES_FULL_PEAD%VKPT=>KPOINTS_FULL%VKPT

      CALL GEN_LAYOUT(GRID_TMP,WDES_FULL_PEAD,LATT_CUR%B,LATT_INI%B, IU6,.TRUE.)
      CALL GEN_INDEX(GRID_TMP,WDES_FULL_PEAD,LATT_CUR%B,LATT_INI%B,-1, -1,.TRUE.)
      
      CALL GENERATE_KPOINTS_TRANS( &
     &   GRID_TMP,KPOINTS%NKPTS,WDES_FULL_PEAD,KPOINTS_FULL,KPOINTS_TRANS_FULL_PEAD)
      
      CALL DEALLOC_GRD(GRID_TMP)
      WDES_FULL_PEAD%GRID=>GRID
      
      LFIRST=.FALSE.

      RETURN
      END SUBROUTINE GENERATE_KPOINTS_TRANS_FULL


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE W1_ROTATE(W,P,LATT_CUR,NB,ISP,ROT_HANDLE,W1)
      USE prec
      USE pseudo
      USE poscar
      USE lattice
      USE wave_high
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(wavefun1) W1
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      INTEGER NB,ISP
      TYPE(rotation_handle), POINTER :: ROT_HANDLE
      ! local variables
      INTEGER N,ISPINOR
      INTEGER NK,NK_OLD,ISP_OLD
      INTEGER NT,NIS,NI,NIP,NPRO,NPROP

      NK=W1%WDES1%NK
      NK_OLD=KPOINTS_TRANS_FULL_PEAD%NK_OLD(NK)
      
      ISP_OLD=ISP
      IF (KPOINTS_TRANS_FULL_PEAD%SPINFLIP(NK)==1) ISP_OLD=3-ISP

      ! plane wave contributions
      DO ISPINOR=0,W%WDES%NRSPINORS-1
         CALL ROTATE_WAVE( &
        &   W%WDES%NGVECTOR(NK), &
        &   W1%CW(1+ISPINOR*W%WDES%NGVECTOR(NK)), & 
        &   W%CW(1+ISPINOR*W%WDES%NGVECTOR(NK_OLD),NB,NK_OLD,ISP_OLD), &
        &   KPOINTS_TRANS_FULL_PEAD%CPHASE(1,NK), KPOINTS_TRANS_FULL_PEAD%NINDPW(1,NK),  &
        &   KPOINTS_TRANS_FULL_PEAD%LINV(NK), KPOINTS_TRANS_FULL_PEAD%LSHIFT(NK))
      ENDDO
      
      ! one-center contributions
      CALL GENERATE_ROT_HANDLE(ROT_HANDLE,KPOINTS_TRANS_FULL_PEAD,P,LATT_CUR,NK,W1%WDES1)
      
      DO ISPINOR=0,W%WDES%NRSPINORS-1
         NIS=1
         NPRO=ISPINOR*(W%WDES%NPRO/2)+1
         DO NT=1,W%WDES%NTYP
            DO NI=NIS,W%WDES%NITYP(NT)+NIS-1
               NIP=KPOINTS_TRANS_FULL_PEAD%ROTMAP(NI,NK)
               NPROP=ROT_HANDLE%NPRO_NI(NIP)+ISPINOR*(W%WDES%NPRO/2)

               CALL ROTATE_VECTOR( &
              &   KPOINTS_TRANS_FULL_PEAD%LINV(NK), &
              &   W%CPROJ(NPROP,NB,NK_OLD,ISP_OLD),W1%CPROJ(NPRO), &
              &   ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,ROT_HANDLE%SL(1,1,0), &
              &   P(NT))

               NPRO=W%WDES%LMMAX(NT)+NPRO
            ENDDO
            NIS=NIS+W%WDES%NITYP(NT)
         ENDDO
      ENDDO
            
      RETURN
      END SUBROUTINE W1_ROTATE


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE WA_ROTATE(W,P,LATT_CUR,ISP,WA)
      USE prec
      USE pseudo
      USE poscar
      USE lattice
      USE wave_high
      USE kpoints_change
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(wavefuna) WA
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      INTEGER ISP
      ! local variables
      TYPE(rotation_handle), POINTER :: ROT_HANDLE => NULL()
      INTEGER N,ISPINOR
      INTEGER NK,NK_OLD,ISP_OLD
      INTEGER NT,NIS,NI,NIP,NPRO,NPROP
      
      NK=WA%WDES1%NK
      NK_OLD=KPOINTS_TRANS_FULL_PEAD%NK_OLD(NK)
      
      ISP_OLD=ISP
      IF (KPOINTS_TRANS_FULL_PEAD%SPINFLIP(NK)==1) ISP_OLD=3-ISP

      ! plane wave contributions
      DO N=1,W%WDES%NBANDS
         DO ISPINOR=0,W%WDES%NRSPINORS-1
            CALL ROTATE_WAVE( &
           &   W%WDES%NGVECTOR(NK), &
           &   WA%CW(1+ISPINOR*W%WDES%NGVECTOR(NK),N), & 
           &   W%CW(1+ISPINOR*W%WDES%NGVECTOR(NK_OLD),N,NK_OLD,ISP_OLD), &
           &   KPOINTS_TRANS_FULL_PEAD%CPHASE(1,NK), KPOINTS_TRANS_FULL_PEAD%NINDPW(1,NK),  &
           &   KPOINTS_TRANS_FULL_PEAD%LINV(NK), KPOINTS_TRANS_FULL_PEAD%LSHIFT(NK))
         ENDDO
      ENDDO
      
      ! one-center contributions
      CALL GENERATE_ROT_HANDLE(ROT_HANDLE,KPOINTS_TRANS_FULL_PEAD,P,LATT_CUR,NK,WA%WDES1)
      
      DO N=1,W%WDES%NBANDS
         DO ISPINOR=0,W%WDES%NRSPINORS-1
            NIS=1
            NPRO=ISPINOR*(W%WDES%NPRO/2)+1
            DO NT=1,W%WDES%NTYP
               DO NI=NIS,W%WDES%NITYP(NT)+NIS-1
                  NIP=KPOINTS_TRANS_FULL_PEAD%ROTMAP(NI,NK)
                  NPROP=ROT_HANDLE%NPRO_NI(NIP)+ISPINOR*(W%WDES%NPRO/2)
                  
                  CALL ROTATE_VECTOR( &
                 &   KPOINTS_TRANS_FULL_PEAD%LINV(NK), &
                 &   W%CPROJ(NPROP,N,NK_OLD,ISP_OLD),WA%CPROJ(NPRO,N), &
                 &   ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,ROT_HANDLE%SL(1,1,0), &
                 &   P(NT))
                 
                  NPRO=W%WDES%LMMAX(NT)+NPRO
               ENDDO
               NIS=NIS+W%WDES%NITYP(NT)
            ENDDO
         ENDDO
      ENDDO
      
      CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
      
      RETURN
      END SUBROUTINE WA_ROTATE


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE CALC_EWFOR_AND_EWSIF( &
     &   W,KPOINTS,P,CQIJ,CRHODE,NONLR_S,NONL_S,LATT_CUR,T_INFO,DYN,SYMM,IDIR, &
     &   EWFOR,EWSIF)
      USE prec
      USE base
      USE mpimy
      USE poscar
      USE pseudo
      USE lattice
      USE mkpoints
      USE wave_high
      USE nonl_high
      USE msymmetry
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(dynamics) DYN
      TYPE(symmetry) SYMM
      TYPE(type_info) T_INFO
      TYPE(nonl_struct) NONL_S
      TYPE(nonlr_struct) NONLR_S
      TYPE(kpoints_struct) KPOINTS
      OVERLAP CQIJ(:,:,:,:),CRHODE(:,:,:,:)
      INTEGER IDIR
      REAL(q) EWFOR(3,T_INFO%NIOND)
      REAL(q) EWSIF(3,3)
      ! symmetry related quantities (common block)
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &   GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      ! local variables
      TYPE(wavespin) DWDK
      INTEGER NK,ISP,ISPINOR
      INTEGER NDIR
      INTEGER NIS,NT,NI,NIP
      INTEGER LMMAXC,LBASE,L,N,I,J
      REAL(q) ENL(T_INFO%NIONS)
      REAL(q) EWISIF(6)
      GDEF, ALLOCATABLE :: CPROJXYZ(:,:,:)

      EWFOR=0
      EWSIF=0
      EWISIF=0

      IF (.NOT.W%WDES%LOVERL) RETURN
     
      IF (DYN%ISIF/=0) THEN
         NDIR=9
      ELSE
         NDIR=3
      ENDIF
      
      CALL ALLOCW(W%WDES,DWDK)
      
      CALL PEAD_DPSI_DK_IDIR( &
     &   W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,IDIR,DWDK)

      ALLOCATE(CPROJXYZ(W%WDES%NPROD,W%WDES%NBANDS,NDIR))

      DO ISP=1,W%WDES%ISPIN
      DO NK=1,W%WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE 
#endif      
         IF (NONLR_S%LREAL) THEN             
            CALL PHASER(W%WDES%GRID,LATT_CUR,NONLR_S,NK,W%WDES)
            CALL RPROXYZ(W%WDES%GRID,NONLR_S,P,LATT_CUR,W,W%WDES,ISP,NK,CPROJXYZ(1,1,1))
            IF (NDIR>3) &
           &   CALL RPROLAT_DER(W%WDES%GRID,NONLR_S,P,LATT_CUR,W,W%WDES,ISP,NK,CPROJXYZ(1,1,4))
         ELSE
            CALL PHASE(W%WDES,NONL_S,NK)
            CALL PROJXYZ(NONL_S,W%WDES,W,LATT_CUR,ISP,NK,CPROJXYZ,.TRUE.)
            IF (NDIR>3) &
           &   CALL PROJLAT_DER(P,NONL_S,W%WDES,W,LATT_CUR,ISP,NK,CPROJXYZ(1,1,4))
         ENDIF

         DO I=1,NDIR
            DO N=1,W%WDES%NBANDS
               ENL=0
               DO ISPINOR=0,W%WDES%NRSPINORS-1
                  LBASE=ISPINOR*W%WDES%NPRO/2
                   
                  NIS=1
                  typ: DO NT=1,W%WDES%NTYP
                     LMMAXC=W%WDES%LMMAX(NT)
                     IF (LMMAXC==0) GOTO 510
                     ion: DO NI=NIS,W%WDES%NITYP(NT)+NIS-1

                     DO L=1,LMMAXC
                        ENL(NI)=ENL(NI)+CPROJXYZ(LBASE+L,N,I)*GCONJG(DWDK%CPROJ(LBASE+L,N,NK,ISP))
                     ENDDO

                     LBASE=LMMAXC+LBASE
                     ENDDO ion
  510                NIS=NIS+W%WDES%NITYP(NT)
                  ENDDO typ
               ENDDO
               IF (I<=3) THEN
                  DO NI=1,W%WDES%NIONS
                     NIP=NI_GLOBAL(NI,W%WDES%COMM_INB)
                     EWFOR(I,NIP)=EWFOR(I,NIP)-ENL(NI)*W%WDES%RSPIN*W%WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)
                  ENDDO
               ELSE
                  DO NI=1,W%WDES%NIONS
                     EWISIF(I-3)=EWISIF(I-3)+ENL(NI)*W%WDES%RSPIN*W%WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
      ENDDO
      ENDDO
      CALLMPI( M_sum_d(W%WDES%COMM, EWFOR(1,1),T_INFO%NIONS*3))
      CALLMPI( M_sum_d(W%WDES%COMM, EWISIF(1),6))

      DEALLOCATE(CPROJXYZ)
      CALL DEALLOCW(DWDK)

      NI=0
      DO I=1,3
         DO J=1,I
            NI=NI+1
            EWSIF(I,J)=EWISIF(NI)
            EWSIF(J,I)=EWISIF(NI)
         ENDDO
      ENDDO

      IF (SYMM%ISYM>0) THEN
         CALL FORSYM(EWFOR,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,  & 
        &   SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)
         CALL TSYM(EWSIF,ISYMOP,NROTK,LATT_CUR%A)
      ENDIF

!   add rigid ion contribution
      DO NI=1,T_INFO%NIONS
         ENL(NI)=ONE_CENTRE_CHARGE(W%WDES,T_INFO,P,NI,SIZE(CQIJ,1),CQIJ,CRHODE)
         EWFOR(IDIR,NI)=EWFOR(IDIR,NI)-P(T_INFO%ITYP(NI))%ZVALF*T_INFO%VCA(T_INFO%ITYP(NI))-ENL(NI)
      ENDDO
      
      RETURN
      END SUBROUTINE CALC_EWFOR_AND_EWSIF


!******************** SUBROUTINE DPSI_DK_BERRY *************************
!
! This subroutine implements the following expression
!
! D_{k,\alpha} = - c * i/2 \sum_{\beta} A_{\alpha\beta}/dk_{\beta} *
!              [ |u_{m,k+p*dk_{\beta}}> S^{-1}_{mn}(k,k+p*dk_{\beta})-
!                 |u_{m,k-p*dk_{\beta}}> S^{-1}_{mn}(k,k-p*dk_{\beta}) ]
! 
! \alpha = 1,2,3 labels the cartesion directions x, y, and z
!
! dk_{\beta} is the distance between neighbouring k-points along
! direction \beta in reciprocal space (in direct coordinates),
! dk_1,2,3 = 2*pi/NKPX , 2*pi/NKPY , 2*pi/NKPZ
!
! c = PREFAC, p = STRIDE
!
! For c=1 and p=1 one recovers Eq. (96) in PRB 63, 155107 (2001), i.e.,
! a first order finite difference expression for the derivative
! d\psi_k/dk_{\alpha}
!
!***********************************************************************
      SUBROUTINE DPSI_DK_BERRY( W,K,ISP,P,CQIJ,LATT_CUR,T_INFO, &
     &   RPHI,RPHI_CPROJ,DIR,DWDK_DIR,RPHI_1K,RPHI_1K_CPROJ,STRIDE,PREFAC)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high
      USE wave_mpi
      USE full_kpoints
      USE sym_prec
      USE constant
      USE dfast
      
      IMPLICIT NONE
      
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      REAL(q) K(3)
      INTEGER ISP
      OVERLAP CQIJ(:,:,:,:)
      INTEGER, OPTIONAL :: STRIDE
      REAL(q), OPTIONAL :: PREFAC
      ! output for one direction
      INTEGER, OPTIONAL :: DIR
      TYPE(wavespin) , OPTIONAL :: DWDK_DIR
      ! output for one k-point
      COMPLEX(q), OPTIONAL :: RPHI_1K(W%WDES%NRPLWV,W%WDES%NBANDS,3)
      GDEF , OPTIONAL :: RPHI_1K_CPROJ(W%WDES%NPROD,W%WDES%NBANDS,3)
      ! output for all direction
      COMPLEX(qs), OPTIONAL :: RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      GDEFS , OPTIONAL :: RPHI_CPROJ(W%WDES%NPROD, W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      ! local variables
      TYPE(wavefuna) DWDK
      TYPE(wavespin) WP
      TYPE(wavefun1) WQ
      TYPE(wavefuna) WHAM
      TYPE(wavedes1), TARGET :: WDESK,WDESQ,WDESQ_IRZ
      TYPE(rotation_handle), POINTER :: ROT_HANDLE
      
      REAL(q) PHASE(3)
      COMPLEX(q), ALLOCATABLE :: SV(:)

      INTEGER ISP_IRZ,ISPINOR      
      INTEGER NK,NQ,NBANDS,NPOS,NSTRIP,NSTRIP_ACT,N,NP
      INTEGER NPOS_RED,NSTRIP_RED,NSTRIP_RED_ACT      
      LOGICAL LSHIFT

      INTEGER I,SGN,J
      REAL(q) K1(3),K2(3)
      REAL(q) DK(3)
      REAL(q) RCOND
      COMPLEX(q) CDET(2)
      INTEGER, ALLOCATABLE :: IPVT(:)
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
      COMPLEX(q) WEIGHT

      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      GDEF, ALLOCATABLE :: SP(:,:)

      INTEGER STRD
      REAL(q) PRFAC

      NULLIFY(ROT_HANDLE)

      IF (.NOT.IS_INSULATING()) THEN
         CALL CHECK_OCCUPATIONS(W)
         IF (.NOT.IS_INSULATING()) THEN
            RETURN
         ENDIF
      ENDIF

      STRD=1
      IF (PRESENT(STRIDE)) STRD=STRIDE 
      PRFAC=1._q
      IF (PRESENT(PREFAC)) PRFAC=PREFAC

      WP=W
      WP%WDES=>WDES_FULL_PEAD

      CALL CHECK_FULL_KPOINTS

      ISP_IRZ=ISP
      NSTRIP=NSTRIP_STANDARD

      ! search for kpoint k in BZ
      NK=KPOINT_IN_FULL_GRID(K,KPOINTS_FULL)
      ! NK must be a kpoint in the IBZ
      IF (NK/=KPOINTS_FULL%NEQUIV(NK)) THEN
         WRITE(*,'(A,3F7.4)') 'DPSI_DK_BERRY: ERROR, k-point not in IBZ:',K(1:3)
         STOP
      ENDIF
      ! allocate temporary wavefunction array at NK
      CALL SETWDES(WP%WDES,WDESK,NK)
      CALL NEWWAVA(DWDK,WDESK,WDESK%NBANDS)

      ! allocate workspace
      ALLOCATE(IPVT(NOCC(ISP)),CWORK(NOCC(ISP)))
      ALLOCATE(SP(NSTRIP*W%WDES%NB_PAR,W%WDES%NB_TOT))

      DK(1)=1.0_q/REAL(KPOINTS_FULL%NKPX,q)
      DK(2)=1.0_q/REAL(KPOINTS_FULL%NKPY,q)
      DK(3)=1.0_q/REAL(KPOINTS_FULL%NKPZ,q)

      ! calculate | d\tilde{\psi}_{nk_j}/dk > 
      directions: DO I=1,3
      ! set DWDK to zero
      DWDK%CPTWFP=0
      DWDK%CPROJ=0
#ifdef gammareal
      LIMAGINARY_PART=.FALSE.
#endif
      deltak: DO SGN=1,-1,-2
         ! add + < k | k+b_i >  or  - < k | k-b_i >
         K1=K
         K2=K; K2(I)=K2(I)+SGN*STRD*DK(I)
         CALL CALC_OVERLAP(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S,LONLYOCC=LOCCUPIED_ONLY)
!        IF (LDOIO) CALL DUMP_HAM_PEAD("     S_k1_k2",W%WDES,S)
         ! Find LU decomposition and reciprocal condition of S
         CALL ZGECO(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,RCOND,CWORK)
         IF (RCOND.GT.0._q) THEN
            ! Calculate inv(S)
            CALL ZGEDI(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,CDET,CWORK,1)
         ELSE
            ! S is singular to working precision
            IF (LDOIO) THEN
               WRITE(*,'(A,3F8.4,A,3F8.4)') &
              &   'DPSI_DK_BERRY: ERROR, S(k1,k2) is singular: k1=',K1,' k2=',K2
            ENDIF
            STOP
         ENDIF
         ! only S(1:NOCC,1:NOCC) is to be used, set the rest to zero
         IF (NOCC(ISP)<W%WDES%NB_TOT) THEN
            S(:,NOCC(ISP)+1:)=0
            S(NOCC(ISP)+1:,:)=0
         ENDIF

!        IF (LDOIO) CALL DUMP_HAM_PEAD(" inv S_k1_k2",W%WDES,S)
#ifdef gammareal
         IF (SGN/=1) LIMAGINARY_PART=.TRUE.
#endif
         ! search for kpoint k2 in BZ
         NQ=KPOINT_IN_FULL_GRID(K2,KPOINTS_FULL)
            
         CALL SETWDES(WP%WDES,WDESQ,NQ)
         CALL SETWDES(WP%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

         CALL NEWWAV(WQ, WDESQ,.TRUE.)

         CALL NEWWAVA(WHAM,DWDK%WDES1,NSTRIP)

         CALL SET_PHASE_SHIFT(K1,K2,PHASE)

         ALLOCATE(SV(WDESQ%GRID%MPLWV))
         CALL SET_PHASE_SHIFT_GRID_RSPACE(PHASE,WDESQ,SV)
         
         IF (LOCCUPIED_ONLY) THEN
            NBANDS=CEILING(REAL(NOCC(ISP))/REAL(WP%WDES%NB_PAR))
         ELSE
            NBANDS=WP%WDES%NBANDS
         ENDIF
         
         strips: DO NPOS=1,NBANDS,NSTRIP

            WHAM%CW=0
            WHAM%CPROJ=0

            NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

            instrip: DO N=NPOS,NPOS+NSTRIP_ACT-1
               NP=N-NPOS+1

               IF (NQ==KPOINTS_FULL%NEQUIV(NQ)) THEN
                  CALL W1_COPY(ELEMENT(WP, WDESQ, N, ISP), WQ)
!                 CALL FFTWAV_W1(WQ)
               ELSE
                  CALL W1_ROTATE(WP,P,LATT_CUR,N,ISP,ROT_HANDLE,WQ)
! #ifndef gammareal
!                 ! symmetry must be considered if the wavefunctions for
!                 ! k-point NQ are not stored in WP (i.e. not in the IBZ)
!                 LSHIFT=.FALSE.
!                 IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
!                     (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
!                     (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
!                 CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WP, WDESQ_IRZ, N, ISP_IRZ), &
!                    ROT_HANDLE, P, LATT_CUR, LSHIFT)
! #endif
               ENDIF
               CALL FFTWAV_W1(WQ)

               IF (LPEAD_RETURN_Q_CPROJ) THEN
               ! calculate sum_j Qij < beta_j |wq>   (Qij = \int Q_ij(r) dr)
                  CALL OVERL_AND_APPLY_PHASE_SHIFT( &
                       &   PHASE,T_INFO,CQIJ,WDESQ,WQ,WHAM%CPROJ(:,NP))

               ELSE
              ! apply phase shift to the projections
                  CALL APPLY_PHASE_SHIFT_CPROJ( &
                       &   PHASE,T_INFO,WDESQ,WQ,WHAM%CPROJ(:,NP))
               ENDIF

               ! Evaluate sv|wq> (in real space)
               CALL APPLY_PHASE_SHIFT_GRID(WQ,SV)
               ! fft sv|wq> -> reciprocal space (layout consistent with kpoint NK)
               DO ISPINOR=0,DWDK%WDES1%NRSPINORS-1
                  CALL FFTEXT(DWDK%WDES1%NGVECTOR,DWDK%WDES1%NINDPW(1), &
                 &            WQ%CR(1+ISPINOR*WDESQ%GRID%MPLWV), &
                 &            WHAM%CW(1+ISPINOR*DWDK%WDES1%NGVECTOR,NP), &
                 &            DWDK%WDES1%GRID,.FALSE.)
               ENDDO
            ENDDO instrip            

            ! redistribution of WHAM from "over bands" to
            ! "over pw components"
            CALL REDISTRIBUTE_PW(ELEMENTS(WHAM,1,NSTRIP_ACT))
            CALL REDISTRIBUTE_PROJ(ELEMENTS(WHAM,1,NSTRIP_ACT))

            ! add WHAM to DWDK

            NPOS_RED=(NPOS-1)*DWDK%WDES1%NB_PAR+1
            NSTRIP_RED= NSTRIP*DWDK%WDES1%NB_PAR
            NSTRIP_RED_ACT=NSTRIP_ACT*DWDK%WDES1%NB_PAR

            SP=0
#ifndef gammareal
            SP(1:NSTRIP_RED_ACT,:)=SGN*S(NPOS_RED:NPOS_RED+NSTRIP_RED_ACT-1,:)
#else
            IF (.NOT.LIMAGINARY_PART) THEN
               SP(1:NSTRIP_RED_ACT,:)=SGN*AIMAG(S(NPOS_RED:NPOS_RED+NSTRIP_RED_ACT-1,:))
            ELSE
               SP(1:NSTRIP_RED_ACT,:)=SGN* REAL(S(NPOS_RED:NPOS_RED+NSTRIP_RED_ACT-1,:))
            ENDIF
#endif
!           IF (LDOIO) CALL DUMP_HAM_PEAD(" SP",W%WDES,SP)

            IF (DWDK%WDES1%NPL_RED/=0) &
            CALL GGEMM('N', 'N', m_ DWDK%WDES1%NPL_RED , DWDK%WDES1%NB_TOT, NSTRIP_RED, &
           &               one, WHAM%CW_RED(1,1), m_ DWDK%WDES1%NRPLWV_RED, &
           &               SP(1,1), NSTRIP_RED, one, &
           &               DWDK%CW_RED(1,1), m_ DWDK%WDES1%NRPLWV_RED)

            IF (DWDK%WDES1%NPRO_O_RED/=0) &
            CALL GGEMM('N', 'N', DWDK%WDES1%NPRO_O_RED , DWDK%WDES1%NB_TOT, NSTRIP_RED, &
           &               one, WHAM%CPROJ_RED(1,1), DWDK%WDES1%NPROD_RED,  &
           &               SP(1,1), NSTRIP_RED, one, &
           &               DWDK%CPROJ_RED(1,1), DWDK%WDES1%NPROD_RED)

         ENDDO strips

         ! some deallocation to be done
         CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
         CALL DELWAV(WQ,.TRUE.)
         CALL DELWAVA(WHAM)
         DEALLOCATE(SV)
      ENDDO deltak
#ifdef gammareal
      LIMAGINARY_PART=.FALSE.
#endif
      ! redistribute DWDK from "over pw components" to
      ! "over bands"
      CALL REDISTRIBUTE_PW(DWDK)
      CALL REDISTRIBUTE_PROJ(DWDK)

#ifndef gammareal
      WEIGHT=-PRFAC*(0._q,0.5_q)/DK(I)/TPI
#else
      WEIGHT= PRFAC/DK(I)/TPI
#endif
!     WEIGHT=CMPLX(0._q,0.5_q, q)*WP%WDES%RSPIN*WP%WDES%WTKPT(NK)/DK(I)/TPI

      ! add DWDK to RPHI and RHPI_CPROJ
      IF (PRESENT(RPHI)) THEN
         IF (PRESENT(RPHI_CPROJ)) THEN
            DO J=1,3
               RPHI(:,:,NK,ISP,J)=RPHI(:,:,NK,ISP,J)+WEIGHT*LATT_CUR%A(J,I)*DWDK%CW(:,:)
               RPHI_CPROJ(:,:,NK,ISP,J)=RPHI_CPROJ(:,:,NK,ISP,J)+WEIGHT*LATT_CUR%A(J,I)*DWDK%CPROJ(:,:)
            ENDDO
         ENDIF
      ENDIF

      ! add DWDK to DWDK_DIR      
      IF (PRESENT(DIR)) THEN
         IF (PRESENT(DWDK_DIR)) THEN
            DWDK_DIR%CW(:,:,NK,ISP)=DWDK_DIR%CW(:,:,NK,ISP)+WEIGHT*LATT_CUR%A(DIR,I)*DWDK%CW(:,:)
            DWDK_DIR%CPROJ(:,:,NK,ISP)=DWDK_DIR%CPROJ(:,:,NK,ISP)+WEIGHT*LATT_CUR%A(DIR,I)*DWDK%CPROJ(:,:)
         ENDIF
      ENDIF

      ! add DWDK to RPHI_1K and RHPI_1K_CPROJ
      IF (PRESENT(RPHI_1K)) THEN
         IF (PRESENT(RPHI_1K_CPROJ)) THEN
            DO J=1,3
               RPHI_1K(:,:,J)=RPHI_1K(:,:,J)+WEIGHT*LATT_CUR%A(J,I)*DWDK%CW(:,:)
               RPHI_1K_CPROJ(:,:,J)=RPHI_1K_CPROJ(:,:,J)+WEIGHT*LATT_CUR%A(J,I)*DWDK%CPROJ(:,:)
            ENDDO
         ENDIF
      ENDIF

      ENDDO directions

      ! deallocate workspace
      CALL DELWAVA(DWDK)
      DEALLOCATE(IPVT,CWORK)
      DEALLOCATE(SP)
      
      RETURN
      END SUBROUTINE DPSI_DK_BERRY


!******************** SUBROUTINE DPSI_DK_EV ****************************
!
! This subroutine implements the following expression
!
! D_{kn,alpha,i} = \sum_n \sum_j (d_ij)_alpha < p_j | \tilde{\psi}_kn >
!
! d_ij = -\int Q_{ij}(r) r dr  (note the sign!)
!
! \alpha = 1,2,3 labels the cartesion directions x, y, and z
!
! This is analogous to the socalled "expectation value"
! term in the Berry-phase formalism (see SUBROUTINE SET_EV in elpol.F).
!
!***********************************************************************
      SUBROUTINE DPSI_DK_EV(W,K,ISP,RPHI_CPROJ,DIR,DWDK_DIR,RPHI_1K_CPROJ)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high
      USE full_kpoints
      IMPLICIT NONE
      TYPE(wavespin) W
      REAL(q) K(3)
      INTEGER ISP
      ! output for one direction
      INTEGER, OPTIONAL :: DIR
      TYPE(wavespin) , OPTIONAL :: DWDK_DIR
      ! output for one k-point
      GDEF , OPTIONAL :: RPHI_1K_CPROJ(W%WDES%NPROD,W%WDES%NBANDS,3)
      ! output for all directions
      GDEFS , OPTIONAL :: RPHI_CPROJ(W%WDES%NPROD, W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      ! local variables
      TYPE(wavefuna) DWDK
      TYPE(wavefuna) WK
      TYPE(wavedes1) WDESK
      INTEGER I,NK,N
      REAL(q) WEIGHT
      OVERLAP, ALLOCATABLE :: TMP(:,:,:,:)

      IF (.NOT.IS_INSULATING()) THEN
         CALL CHECK_OCCUPATIONS(W)
         IF (.NOT.IS_INSULATING()) THEN
            RETURN
         ENDIF
      ENDIF

      CALL CHECK_FULL_KPOINTS

      ! search for kpoint k in BZ
      NK=KPOINT_IN_FULL_GRID(K,KPOINTS_FULL)
      ! NK must be a kpoint in the IBZ
      IF (NK/=KPOINTS_FULL%NEQUIV(NK)) THEN
         WRITE(*,'(A,3F7.4)') 'DPSI_DK_EV: ERROR, k-point not in IBZ:',K(1:3)
         STOP
      ENDIF

      ! allocate temporary wavefunction array at NK
      CALL SETWDES(W%WDES,WDESK,NK)
      CALL NEWWAVA_PROJ(DWDK,WDESK,WDESK%NBANDS)
      WK=ELEMENTS(W,WDESK,ISP)

      ALLOCATE(TMP(SIZE(d_IJ,2),SIZE(d_IJ,3),SIZE(d_IJ,4),SIZE(d_IJ,5)))

      directions: DO I=1,3
         DWDK%CPROJ=0

         TMP(:,:,:,:)=d_IJ(I,:,:,:,:)

         CALL OVERL(WDESK,WDESK%LOVERL,SIZE(TMP,1),TMP(1,1,1,1), &
        &           WK%CPROJ(1,1),DWDK%CPROJ(1,1))

         ! unoccupied bands do not contribute
         DO N=1,WDESK%NBANDS
            IF ( (WDESK%NB_LOW+WDESK%NB_PAR*(N-1))>NOCC(ISP) ) DWDK%CPROJ(:,N)=0
         ENDDO

         WEIGHT=1._q
!        WEIGHT=W%WDES%RSPIN*W%WDES%WTKPT(NK)
         ! add DWDK%CPROJ to RPHI_CPROJ
         IF (PRESENT(RPHI_CPROJ)) THEN
            RPHI_CPROJ(:,:,NK,ISP,I)=RPHI_CPROJ(:,:,NK,ISP,I)+WEIGHT*DWDK%CPROJ(:,:)
         ENDIF

         ! add DWDK%CPROJ to DWDK_DIR         
         IF (PRESENT(DIR)) THEN
            IF (PRESENT(DWDK_DIR).AND.(I==DIR)) THEN
               DWDK_DIR%CPROJ(:,:,NK,ISP)=DWDK_DIR%CPROJ(:,:,NK,ISP)+WEIGHT*DWDK%CPROJ(:,:)
            ENDIF
         ENDIF

         ! add DWDK%CPROJ to RPHI_1K_CPROJ
         IF (PRESENT(RPHI_1K_CPROJ)) THEN
            RPHI_1K_CPROJ(:,:,I)=RPHI_1K_CPROJ(:,:,I)+WEIGHT*DWDK%CPROJ(:,:)
         ENDIF

      ENDDO directions

      CALL DELWAVA_PROJ(DWDK)
      DEALLOCATE(TMP)
                  
      RETURN
      END SUBROUTINE DPSI_DK_EV


!******************** SUBROUTINE DPSI_DK_ORTHO *************************
!
! This subroutine orthogonalize the gradient d\psi/dk with respect
! to the occupied states
!
!***********************************************************************
      SUBROUTINE DPSI_DK_ORTHO( &
     &   W,K,ISP,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ,DIR,DWDK_DIR)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high
      USE wave_mpi
      USE full_kpoints
      USE sym_prec
      USE constant
      USE dfast
      
      IMPLICIT NONE
      
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      REAL(q) K(3)
      INTEGER ISP
      OVERLAP CQIJ(:,:,:,:)
      ! output for all directions
      COMPLEX(qs), OPTIONAL ::RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      GDEFS , OPTIONAL :: RPHI_CPROJ(W%WDES%NPROD, W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3)
      ! output for one direction
      INTEGER, OPTIONAL :: DIR
      TYPE(wavespin) , OPTIONAL :: DWDK_DIR
      ! local variables
      TYPE(wavefuna) DWDK
      TYPE(wavespin) WP
      TYPE(wavefuna) WHAM
      TYPE(wavedes1), TARGET :: WDESK
      
      REAL(q) PHASE(3)

      INTEGER NK,NPOS,NSTRIP,NSTRIP_ACT,N,NP
      INTEGER NPOS_RED,NSTRIP_RED,NSTRIP_RED_ACT      

      INTEGER I
      COMPLEX(q) WEIGHT

      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      COMPLEX(q), ALLOCATABLE :: SP(:,:)
      OVERLAP, ALLOCATABLE :: TMP(:,:,:,:)

      IF (.NOT.IS_INSULATING()) THEN
         CALL CHECK_OCCUPATIONS(W)
         IF (.NOT.IS_INSULATING()) THEN
            RETURN
         ENDIF
      ENDIF

      WP=W
      WP%WDES=>WDES_FULL_PEAD

      CALL CHECK_FULL_KPOINTS

      NSTRIP=NSTRIP_STANDARD

      ! search for kpoint k in BZ
      NK=KPOINT_IN_FULL_GRID(K,KPOINTS_FULL)
      ! NK must be a kpoint in the IBZ
      IF (NK/=KPOINTS_FULL%NEQUIV(NK)) THEN
         WRITE(*,'(A,3F7.4)') 'DPSI_DK_ORTHO: ERROR, k-point not in IBZ:',K(1:3)
         STOP
      ENDIF
      ! allocate temporary wavefunction arrays at NK
      CALL SETWDES(WP%WDES,WDESK,NK)
      CALL NEWWAVA(DWDK,WDESK,WDESK%NBANDS)
      CALL NEWWAVA(WHAM,DWDK%WDES1,NSTRIP)

      ! allocate workspace
      ALLOCATE(SP(NSTRIP*W%WDES%NB_PAR,W%WDES%NB_TOT))
      ALLOCATE(TMP(SIZE(d_IJ,2),SIZE(d_IJ,3),SIZE(d_IJ,4),SIZE(d_IJ,5)))

      ! calculate | d\tilde{\psi}_{nk_j}/dk > 
      directions: DO I=1,3
         ! set DWDK to zero
         DWDK%CPTWFP=0
         DWDK%CPROJ=0

         ! calculate S = sum_ij <u_mk|i> d_ij <j|u_nk>
         TMP(:,:,:,:)=d_IJ(I,:,:,:,:)
         CALL CALC_OVERLAP(W,K,K,ISP,P,TMP,LATT_CUR,T_INFO,S,LPLANEWAVE=.FALSE.)
         ! only S(1:NOCC,1:NOCC) is to be used, set the rest to zero
         IF (NOCC(ISP)<W%WDES%NB_TOT) THEN
            S(:,NOCC(ISP)+1:)=0
            S(NOCC(ISP)+1:,:)=0
         ENDIF

!        IF (LDOIO) CALL DUMP_HAM_PEAD(" sum_ij <u_mk|i> d_ij <j|u_nk>",W%WDES,S)

         PHASE=0._q

         strips: DO NPOS=1,W%WDES%NBANDS,NSTRIP
            NSTRIP_ACT=MIN(WP%WDES%NBANDS+1-NPOS,NSTRIP)

            instrip: DO N=NPOS,NPOS+NSTRIP_ACT-1
               NP=N-NPOS+1
               CALL W1_COPY(ELEMENT(WP, WDESK, N, ISP), ELEMENT(WHAM, NP))
               ! calculate sum_j Qij < beta_j |wq>   (Qij = \int Q_ij(r) dr)
               CALL OVERL_AND_APPLY_PHASE_SHIFT( &
              &   PHASE,T_INFO,CQIJ,WDESK,ELEMENT(WHAM,NP))
            ENDDO instrip            

            ! redistribution of WHAM from "over bands" to
            ! "over pw components"
            CALL REDISTRIBUTE_PW(ELEMENTS(WHAM,1,NSTRIP_ACT))
            CALL REDISTRIBUTE_PROJ(ELEMENTS(WHAM,1,NSTRIP_ACT))

            ! add WHAM to DWDK
            NPOS_RED=(NPOS-1)*DWDK%WDES1%NB_PAR+1
            NSTRIP_RED= NSTRIP*DWDK%WDES1%NB_PAR
            NSTRIP_RED_ACT=NSTRIP_ACT*DWDK%WDES1%NB_PAR

            SP=0
            SP(1:NSTRIP_RED_ACT,:)=S(NPOS_RED:NPOS_RED+NSTRIP_RED_ACT-1,:)

            IF (DWDK%WDES1%NPL_RED/=0) &
            CALL GGEMM('N', 'N', m_ DWDK%WDES1%NPL_RED , DWDK%WDES1%NB_TOT, NSTRIP_RED, &
           &               -one, WHAM%CW_RED(1,1), m_ DWDK%WDES1%NRPLWV_RED, &
           &               SP(1,1), NSTRIP_RED, one, &
           &               DWDK%CW_RED(1,1), m_ DWDK%WDES1%NRPLWV_RED)

            IF (DWDK%WDES1%NPRO_O_RED/=0) &
            CALL GGEMM('N', 'N', DWDK%WDES1%NPRO_O_RED , DWDK%WDES1%NB_TOT, NSTRIP_RED, &
           &               -one, WHAM%CPROJ_RED(1,1), DWDK%WDES1%NPROD_RED,  &
           &               SP(1,1), NSTRIP_RED, one, &
           &               DWDK%CPROJ_RED(1,1), DWDK%WDES1%NPROD_RED)

         ENDDO strips

         ! redistribute DWDK from "over pw components" to
         ! "over bands"
         CALL REDISTRIBUTE_PW(DWDK)
         CALL REDISTRIBUTE_PROJ(DWDK)

         WEIGHT=1._q
!        WEIGHT=WP%WDES%RSPIN*WP%WDES%WTKPT(NK)

         ! add DWDK to RPHI and RHPI_CPROJ         
         IF (PRESENT(RPHI)) THEN
            IF (PRESENT(RPHI_CPROJ)) THEN
               RPHI(:,:,NK,ISP,I)=RPHI(:,:,NK,ISP,I)+WEIGHT*DWDK%CW(:,:)
               RPHI_CPROJ(:,:,NK,ISP,I)=RPHI_CPROJ(:,:,NK,ISP,I)+WEIGHT*DWDK%CPROJ(:,:)
            ENDIF
         ENDIF
         
         ! add DWDK to DWDK_DIR         
         IF (PRESENT(DIR)) THEN
            IF (PRESENT(DWDK_DIR).AND.(I==DIR)) THEN
               DWDK_DIR%CW(:,:,NK,ISP)=DWDK_DIR%CW(:,:,NK,ISP)+WEIGHT*DWDK%CW(:,:)
               DWDK_DIR%CPROJ(:,:,NK,ISP)=DWDK_DIR%CPROJ(:,:,NK,ISP)+WEIGHT*DWDK%CPROJ(:,:)
            ENDIF
         ENDIF

      ENDDO directions
      
      ! deallocate workspace
      CALL DELWAVA(WHAM)
      CALL DELWAVA(DWDK)
      DEALLOCATE(TMP)
      DEALLOCATE(SP)

      RETURN
      END SUBROUTINE DPSI_DK_ORTHO


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE CALC_POLARIZATION(W,P,CQIJ,LATT_CUR,T_INFO,LINITIALIZE)
      USE prec
      USE base
      USE wave
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE msymmetry
      USE mgrid
      USE nonl_high
      USE mkpoints      
      USE kpoints_change
      USE full_kpoints
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(latt) LATT_CUR
      LOGICAL, OPTIONAL :: LINITIALIZE
      OVERLAP CQIJ(:,:,:,:)
      ! local variables
      INTEGER ISP,IDIR,IK,ISTR,PO
      INTEGER I,J
      REAL(q) DK(3),K1(3),K2(3)
      REAL(q) R(3,W%WDES%ISPIN),R_EV(3,W%WDES%ISPIN),R_BP(3,W%WDES%ISPIN),RSUM(3)
      REAL(q) DR(3,W%WDES%ISPIN),DR_BP(3,W%WDES%ISPIN),DRSUM(3)
      REAL(q) IMLND,IMLNDS
      REAL(q) IMLNDS_ORDER(PEAD_ORDER)
      REAL(q) DELTA(3,3)
      LOGICAL LINIT
      LOGICAL LSTRINGS
      REAL(q), ALLOCATABLE :: SUM_PHASE_ON_STRING(:,:,:,:)
! test
      COMPLEX(q) DS,DET_MEAN
      COMPLEX(q), ALLOCATABLE :: PROD_DETS_ON_STRING(:,:,:,:)
! test
      
      IF (.NOT.IS_INSULATING()) THEN
         CALL CHECK_OCCUPATIONS(W)
         IF (.NOT.IS_INSULATING()) THEN
            WRITE(*,*) 'CALC_POLARIZATION: ERROR, system is not insulating, skipping...'
            RETURN
         ENDIF
      ENDIF

      CALL SETUP_STRINGS(LSTRINGS)
      IF (.NOT.LSTRINGS) THEN
         WRITE(*,*) 'CALC_POLARIZATION: ERROR, cannot set up strings, skipping...'
         RETURN
      ENDIF
      
      LINIT=.FALSE.
      IF (PRESENT(LINITIALIZE)) LINIT=LINITIALIZE
                
      DK(1)=1.0_q/REAL(KPOINTS_FULL%NKPX,q)
      DK(2)=1.0_q/REAL(KPOINTS_FULL%NKPY,q)
      DK(3)=1.0_q/REAL(KPOINTS_FULL%NKPZ,q)


      ALLOCATE(SUM_PHASE_ON_STRING(MAXVAL(NSTRINGS),PEAD_ORDER,3,W%WDES%ISPIN))
      SUM_PHASE_ON_STRING=0
      ALLOCATE(PROD_DETS_ON_STRING(MAXVAL(NSTRINGS),PEAD_ORDER,3,W%WDES%ISPIN))
      PROD_DETS_ON_STRING=1.0_q

      ! evaluate "expectation value" term
      R_EV=0
      DO ISP=1,W%WDES%ISPIN
#ifdef realmode
         CALL SET_EV(W%WDES,W,ISP,SIZE(d_IJ,2),d_IJ,W%WDES%LOVERL,R_EV(:,ISP))    
#else
         CALL SET_EV(W%WDES,W,ISP,SIZE(d_IJ,2),REAL(d_IJ,q),W%WDES%LOVERL,R_EV(:,ISP))
#endif
         CALL POLSYM(R_EV,LATT_CUR%A)
      ENDDO

      ! evaluate "Berry phase" terms
      R_BP=0
      DO ISP=1,W%WDES%ISPIN
         DO IDIR=1,3
            IMLNDS_ORDER=0
            fullbz: DO IK=1,KPOINTS_FULL%NKPTS
#ifdef MPI
               IF (MOD(IK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE 
#endif
               K1(:)=KPOINTS_FULL%VKPT(:,IK)
               DO PO=1,PEAD_ORDER
                  K2=K1
                  K2(IDIR)=K2(IDIR)+DK(IDIR)*PO
                  ! calculate and store the product of the determinants
                  ! on the strings of k-points
                  DS=DETS(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO)
                  DS=DS/SQRT(REAL(DS,q)**2+AIMAG(DS)**2)
                  PROD_DETS_ON_STRING(MAP_TO_STRING(IK,IDIR),PO,IDIR,ISP)= &
                 &   PROD_DETS_ON_STRING(MAP_TO_STRING(IK,IDIR),PO,IDIR,ISP)*DS
                  ! and the sum of Im{ln|S(k,k')|}
                  IMLND=AIMAG(LOG(DS))
                  SUM_PHASE_ON_STRING(MAP_TO_STRING(IK,IDIR),PO,IDIR,ISP)= &
                 &   SUM_PHASE_ON_STRING(MAP_TO_STRING(IK,IDIR),PO,IDIR,ISP)+IMLND                 
               ENDDO
            ENDDO fullbz
         ENDDO                  
      ENDDO

      CALLMPI( M_prodb_z(W%WDES%COMM_KINTER,PROD_DETS_ON_STRING(1,1,1,1),SIZE(PROD_DETS_ON_STRING)) )
      CALLMPI( M_sum_d(W%WDES%COMM_KINTER,SUM_PHASE_ON_STRING(1,1,1,1),SIZE(SUM_PHASE_ON_STRING)) )

      ! evaluate the polarization (first order terms only)
      DO ISP=1,W%WDES%ISPIN
         DO IDIR=1,3
            DET_MEAN=0
            DO ISTR=1,NSTRINGS(IDIR)
               DET_MEAN=DET_MEAN+PROD_DETS_ON_STRING(ISTR,1,IDIR,ISP)
            ENDDO
            DET_MEAN=DET_MEAN/NSTRINGS(IDIR)
            IMLNDS=0
            DO ISTR=1,NSTRINGS(IDIR)
               IMLND=AIMAG(LOG(PROD_DETS_ON_STRING(ISTR,1,IDIR,ISP)/DET_MEAN))
               IMLNDS=IMLNDS+IMLND
               IF (LDOIO.AND.(ABS(IMLND/TPI)>0.25_q)) THEN
                  WRITE(*,'(A,I3,A,I3,/A,I4,A,2F10.5,A,/A,F10.5,A)') & 
                 &   'PEAD_POLARIZATION_CALC: WARNING: reciprocal direction ',IDIR,'  spin channel ',ISP, & 
                 &   '   contribution from string ',ISTR,'  not well clustered around <|S|>_av = (',DET_MEAN,' )', &
                 &   '   Im(ln[|S|/<|S|>_av])/2pi =',IMLND/TPI,' > 1/4'               
               ENDIF
            ENDDO
            IMLNDS=IMLNDS/NSTRINGS(IDIR)

            R_BP(:,ISP)=R_BP(:,ISP)-W%WDES%RSPIN*(AIMAG(LOG(DET_MEAN))+IMLNDS)*LATT_CUR%A(:,IDIR)/TPI
!           R_BP(:,ISP)=R_BP(:,ISP)-IMLNDS*LATT_CUR%A(:,IDIR)* &
!          &   W%WDES%RSPIN/(KPOINTS_FULL%NKPTS*DK(IDIR))/TPI
         ENDDO
      ENDDO

      R=-R_EV+R_BP
      IF (.NOT.ALLOCATED(RSTORE)) ALLOCATE(RSTORE(3,W%WDES%ISPIN))
      IF (.NOT.ALLOCATED(R_EV_STORE)) ALLOCATE(R_EV_STORE(3,W%WDES%ISPIN))
      RSTORE=R
      R_EV_STORE=R_EV
 
      IF (LINIT) THEN
         IF (.NOT.ALLOCATED(R_EV_INIT)) ALLOCATE(R_EV_INIT(3,W%WDES%ISPIN))
         R_EV_INIT=R_EV
         IF (.NOT.ALLOCATED(R_BP_INIT)) ALLOCATE(R_BP_INIT(3,W%WDES%ISPIN))
         R_BP_INIT=R_BP
         IF (ALLOCATED(SUM_PHASE_ON_STRING_INIT)) DEALLOCATE(SUM_PHASE_ON_STRING_INIT)
         ALLOCATE(SUM_PHASE_ON_STRING_INIT(MAXVAL(NSTRINGS),PEAD_ORDER,3,W%WDES%ISPIN))
         SUM_PHASE_ON_STRING_INIT=SUM_PHASE_ON_STRING         
      ENDIF
      
      ! if initialized then calculate the difference in polarization
      IF (.NOT.LINIT.AND.ALLOCATED(SUM_PHASE_ON_STRING_INIT)) THEN
         DR_BP=0
         DO ISP=1,W%WDES%ISPIN
            DO IDIR=1,3
               IMLNDS_ORDER=0
               DO PO=1,PEAD_ORDER
                  DO ISTR=1,NSTRINGS(IDIR)
                     IMLND=MOD(SUM_PHASE_ON_STRING(ISTR,PO,IDIR,1)- &
                    &   SUM_PHASE_ON_STRING_INIT(ISTR,PO,IDIR,1),TPI)
                     IF (ABS(IMLND+TPI)<ABS(IMLND)) IMLND=IMLND+TPI
                     IF (ABS(IMLND-TPI)<ABS(IMLND)) IMLND=IMLND-TPI
                     IMLNDS_ORDER(PO)=IMLNDS_ORDER(PO)+IMLND
                  ENDDO
               ENDDO
               IMLNDS=0
               SELECT CASE(PEAD_ORDER)
                  CASE(1)
                     IMLNDS=IMLNDS+IMLNDS_ORDER(1)
                  CASE(2)
                     IMLNDS=IMLNDS+IMLNDS_ORDER(1)*1.333333333_q
                     IMLNDS=IMLNDS-IMLNDS_ORDER(2)*0.166666666_q                 
                  CASE(3)
                     IMLNDS=IMLNDS+IMLNDS_ORDER(1)*1.5_q
                     IMLNDS=IMLNDS-IMLNDS_ORDER(2)*0.3_q
                     IMLNDS=IMLNDS+IMLNDS_ORDER(3)*0.033333333_q
                  CASE(4)
                     IMLNDS=IMLNDS+IMLNDS_ORDER(1)*1.6_q
                     IMLNDS=IMLNDS-IMLNDS_ORDER(2)*0.4_q
                     IMLNDS=IMLNDS+IMLNDS_ORDER(3)*0.076190476_q
                     IMLNDS=IMLNDS-IMLNDS_ORDER(4)*0.007142857_q
               END SELECT
               DR_BP(:,ISP)=DR_BP(:,ISP)-IMLNDS*LATT_CUR%A(:,IDIR)* &
              &   W%WDES%RSPIN/(KPOINTS_FULL%NKPTS*DK(IDIR))/TPI
            ENDDO
         ENDDO

         DR=-(R_EV-R_EV_INIT)+DR_BP

         IF (.NOT.ALLOCATED(DRSTORE)) ALLOCATE(DRSTORE(3,W%WDES%ISPIN))
         DRSTORE=DR
         
         RSTORE=-R_EV_INIT+R_BP_INIT+DR
      ENDIF

      ! write to stdout
      IF (LDOIO) THEN
         RSUM(:)=RSTORE(:,1)
         IF (W%WDES%ISPIN==2) RSUM(:)=RSUM(:)+RSTORE(:,2)
         WRITE(*,'(A,3(E10.3,X),A)') ' p_tot=( ',RSUM(1:3),')'
      ENDIF

      IF (LDOIO.AND.LPEAD_EFIELD_SWITCHED_ON().AND. &
     &    (.NOT.LINIT).AND.ALLOCATED(R_EV_INIT)) THEN
         DRSUM(:)=DRSTORE(:,1)
         IF (W%WDES%ISPIN==2) DRSUM(:)=DRSUM(:)+DRSTORE(:,2)
         WRITE(*,'(A,3(E10.3,X),A)',ADVANCE='No') 'dp_tot=( ',DRSUM(1:3),')  diag[e(oo)]=('
         DO I=1,3
            IF (ABS(EFIELD_PEAD(I))>1E-5_q) THEN
               WRITE(*,'(F9.5)',ADVANCE='No')  &
              &   EDEPS/LATT_CUR%OMEGA*DRSUM(I)/EFIELD_PEAD(I)+1
            ELSE
               WRITE(*,'(A)',ADVANCE='No') '    ---  '
            ENDIF
         ENDDO         
         WRITE(*,'(A)') ' )'
      ENDIF

! test
!     IF (LDOIO.AND.LSTRINGS) THEN
!        DO IDIR=1,3
!           WRITE(*,*) 'direction:',IDIR
!           DO PO=1,PEAD_ORDER
!              WRITE(*,*) 'order:',PO
!              IF (ALLOCATED(SUM_PHASE_ON_STRING_INIT)) THEN
!                 WRITE(*,'(I4,F14.7)') &
!                &  (I,SUM_PHASE_ON_STRING(I,PO,IDIR,1)-SUM_PHASE_ON_STRING_INIT(I,PO,IDIR,1), I=1,NSTRINGS(IDIR))
!              ELSE
!                 WRITE(*,'(I4,F14.7)') &
!                &  (I,SUM_PHASE_ON_STRING(I,PO,IDIR,1), I=1,NSTRINGS(IDIR))
!              ENDIF
!           ENDDO
!        ENDDO
!     ENDIF
! test            
      
      DEALLOCATE(SUM_PHASE_ON_STRING)
      DEALLOCATE(PROD_DETS_ON_STRING)

      RETURN
      END SUBROUTINE CALC_POLARIZATION


!******************** FUNCTION IMLNDET *********************************
!      
!***********************************************************************
      FUNCTION IMLNDET(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO)
      USE prec
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(latt) LATT_CUR
      OVERLAP CQIJ(:,:,:,:)
      INTEGER ISP
      REAL(q) K1(3),K2(3)
      REAL(q) IMLNDET
      ! local variables
      REAL(q) RCOND
      COMPLEX(q) CDET(2)
      INTEGER, ALLOCATABLE :: IPVT(:)
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      
      ALLOCATE(IPVT(NOCC(ISP)),CWORK(NOCC(ISP)))
      
      CALL CALC_OVERLAP(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S)
      ! Find LU decomposition and reciprocal condition of S
      CALL ZGECO(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,RCOND,CWORK)
      IF (RCOND.GT.0._q) THEN
         ! Calculate determinant of S
         CALL ZGEDI(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,CDET,CWORK,10)
         IMLNDET=AIMAG(LOG(CDET(1)*10._q**CDET(2)))
      ELSE
         ! S is singular to working precision
         IF (LDOIO) THEN
            WRITE(*,'(A,3F8.4,A,3F8.4)') &
           &   'IMLNDET: ERROR, S(k1,k2) is singular: k1=',K1,' k2=',K2
         ENDIF
         STOP
      ENDIF
      
      DEALLOCATE(IPVT,CWORK)
      
      END FUNCTION IMLNDET


!******************** FUNCTION DETS ************************************
!      
!***********************************************************************
      FUNCTION DETS(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO)
      USE prec
      USE pseudo
      USE poscar
      USE lattice
      USE constant
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(potcar) P(T_INFO%NTYP)
      TYPE(latt) LATT_CUR
      OVERLAP CQIJ(:,:,:,:)
      INTEGER ISP
      REAL(q) K1(3),K2(3)
      COMPLEX(q) DETS
      ! local variables
      REAL(q) RCOND
      COMPLEX(q) CDET(2)
      INTEGER, ALLOCATABLE :: IPVT(:)
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      
      ALLOCATE(IPVT(NOCC(ISP)),CWORK(NOCC(ISP)))
      
      CALL CALC_OVERLAP(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S)
      ! Find LU decomposition and reciprocal condition of S
      CALL ZGECO(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,RCOND,CWORK)
      IF (RCOND.GT.0._q) THEN
         ! Calculate determinant of S
         CALL ZGEDI(S,W%WDES%NB_TOT,NOCC(ISP),IPVT,CDET,CWORK,10)
         DETS=CDET(1)*10._q**CDET(2)
      ELSE
         ! S is singular to working precision
         IF (LDOIO) THEN
            WRITE(*,'(A,3F8.4,A,3F8.4)') &
           &   'DETS: ERROR, S(k1,k2) is singular: k1=',K1,' k2=',K2
         ENDIF
         STOP
      ENDIF
      
      DEALLOCATE(IPVT,CWORK)
      
      END FUNCTION DETS


!************************ SUBROUTINE SET_EV ****************************
!
! This subroutine calculates the "expectation-value" (EV) term,
!
!   Sum_t{ Sum_ij{ d_t_ij <U_nk|B_k_ti> <B_k_tj|U_nk> } }
!
!***********************************************************************
      SUBROUTINE SET_EV(WDES,W,ISP,LMDIM,dIJ,LOVERL,RSUM)
      
      USE prec
      USE wave
      USE mpimy
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE(wavedes)  WDES
      TYPE(wavespin) W
      TYPE(wavedes1) WDES1
      TYPE(wavefun1) W1
      REAL(q) RSUM(3),RSUM_(3)
      LOGICAL LOVERL
      REAL(q) dIJ(3,LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      

      RSUM=0
   
      IF (LOVERL) THEN
      kpoint: DO IK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(IK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      bands: DO IB=1,WDES%NBANDS
          
         CALL SETWDES(WDES,WDES1,IK)
         CALL SETWAV(W,W1,WDES1,IB,ISP)
         
         RSUM_=0
          
         spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
         DO ISPINOR_=0,WDES1%NRSPINORS-1

         NPRO =ISPINOR *WDES1%NPRO/2
         NPRO_=ISPINOR_*WDES1%NPRO/2

         NIS=1

         DO NT=1,WDES1%NTYP
         LMMAXC=WDES1%LMMAX(NT)
         IF (LMMAXC==0) GOTO 230

         DO NI=NIS,WDES1%NITYP(NT)+NIS-1
               DO L=1,LMMAXC
               DO LP=1,LMMAXC
                  RSUM_(1:3)=RSUM_(1:3)+ &
                 &  dIJ(1:3,LP,L,NI,ISP+ISPINOR_+2*ISPINOR)*W1%CPROJ(LP+NPRO_)*GCONJG(W1%CPROJ(L+NPRO))
               ENDDO
               ENDDO

               NPRO = LMMAXC+NPRO
               NPRO_= LMMAXC+NPRO_
         ENDDO
 230     NIS = NIS+WDES1%NITYP(NT)
         ENDDO
         ENDDO
         ENDDO spinor

      WEIGHT=WDES1%RSPIN*W1%FERWE*WDES%WTKPT(IK)
      RSUM=RSUM+RSUM_*WEIGHT

      ENDDO bands
      ENDDO kpoint
      ENDIF

#ifdef MPI
      CALLMPI( M_sum_d( WDES1%COMM, RSUM, 3))
#endif      

      RETURN
      END SUBROUTINE SET_EV


!******************** SUBROUTINE CALC_OVERLAP **************************
!      
!***********************************************************************
      SUBROUTINE CALC_OVERLAP( &
     &   W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S, &
     &   LPLANEWAVE,LONECENTER,LONLYOCC,LUSEQIJB &
     &)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE wave_high

      IMPLICIT NONE
      
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      REAL(q) K1(3),K2(3)
      INTEGER ISP
      OVERLAP CQIJ(:,:,:,:)
      COMPLEX(q) S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      LOGICAL, OPTIONAL :: LPLANEWAVE,LONECENTER
      LOGICAL, OPTIONAL :: LONLYOCC
      LOGICAL, OPTIONAL :: LUSEQIJB
      ! local variables      
      GDEF S_(W%WDES%NB_TOT,W%WDES%NB_TOT)
      LOGICAL LPWV,LONE,LOCC,LQIJB
      
      LPWV=.TRUE.
      LONE=.TRUE.
      LOCC=.FALSE.
      LQIJB=.FALSE.
      IF (PRESENT(LPLANEWAVE)) LPWV=LPLANEWAVE
      IF (PRESENT(LONECENTER)) LONE=LONECENTER
      IF (PRESENT(LONLYOCC)) LOCC=LONLYOCC
      IF (PRESENT(LUSEQIJB)) LQIJB=LUSEQIJB

      S=0
      
      CALL CALC_OVERLAP_(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S_,LPWV,LONE,LOCC,LQIJB)
      S=S_
#ifdef gammareal
      LIMAGINARY_PART=.TRUE.
      CALL CALC_OVERLAP_(W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S_,LPWV,LONE,LOCC,LQIJB)
      S=S+CMPLX(0.0,S_,q)
      LIMAGINARY_PART=.FALSE.
#endif

      RETURN
      END SUBROUTINE CALC_OVERLAP
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************      
      SUBROUTINE CALC_OVERLAP_( &
     &   W,K1,K2,ISP,P,CQIJ,LATT_CUR,T_INFO,S, &
     &   LPLANEWAVE,LONECENTER,LONLYOCC,LUSEQIJB &
     &)
      USE prec
      USE poscar
      USE pseudo
      USE lattice
      USE full_kpoints
      USE wave_high
      USE dfast
      USE sym_prec
! test
      USE paw
! test
      
      IMPLICIT NONE
      
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      REAL(q) K1(3),K2(3)
      INTEGER ISP
      OVERLAP CQIJ(:,:,:,:)
      GDEF S(W%WDES%NB_TOT,W%WDES%NB_TOT)
      LOGICAL LPLANEWAVE,LONECENTER
      LOGICAL LONLYOCC
      LOGICAL LUSEQIJB

      ! local variables
      TYPE(wavefun1) WQ
      TYPE(wavespin) WP
      TYPE(wavefuna) WK,WHAM
      TYPE(wavedes1), TARGET :: WDESK,WDESQ,WDESQ_IRZ
      
      TYPE(rotation_handle), POINTER :: ROT_HANDLE

      REAL(q) PHASE(3)
      COMPLEX(q), ALLOCATABLE :: SV(:)
! test
      COMPLEX(q), ALLOCATABLE :: QIJB(:,:,:,:)
! test
      
      INTEGER ISP_IRZ,ISPINOR
      INTEGER NK,NQ,NBANDS,NPOS,NSTRIP,NSTRIP_ACT,N,NP
      INTEGER NPOS_RED,NSTRIP_RED
      LOGICAL LSHIFT
      
      S=0
      
      WP=W
      WP%WDES=>WDES_FULL_PEAD
            
      CALL CHECK_FULL_KPOINTS

      NULLIFY(ROT_HANDLE)      

      ISP_IRZ=ISP
      NSTRIP=NSTRIP_STANDARD
      
      ! search for kpoint k1 in BZ
      NK=KPOINT_IN_FULL_GRID(K1,KPOINTS_FULL)
      CALL SETWDES(WP%WDES,WDESK,NK)
      IF (NK==KPOINTS_FULL%NEQUIV(NK)) THEN
         ! k1 is a kpoint in the IBZ
         WK=ELEMENTS(WP,WDESK,ISP)
      ELSE
         ! k1 is not a kpoint in the IBZ
         CALL NEWWAVA(WK,WDESK,WDESK%NBANDS)
         CALL WA_ROTATE(WP,P,LATT_CUR,ISP,WK)
      ENDIF
      ! redistribute wavefunctions and projections at K1
      ! from "over bands" to "over pw components"
      CALL REDISTRIBUTE_PW(WK)
      CALL REDISTRIBUTE_PROJ(WK)

      ! search for kpoint k2 in BZ
      NQ=KPOINT_IN_FULL_GRID(K2,KPOINTS_FULL)
            
      CALL SETWDES(WP%WDES,WDESQ,NQ)
      CALL SETWDES(WP%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

      CALL NEWWAV(WQ, WDESQ,.TRUE.)

      CALL NEWWAVA(WHAM,WDESK,NSTRIP)

      CALL SET_PHASE_SHIFT(K1,K2,PHASE)
            
      ALLOCATE(SV(WDESQ%GRID%MPLWV))
      CALL SET_PHASE_SHIFT_GRID_RSPACE(PHASE,WDESQ,SV)

      IF (LONLYOCC) THEN
         NBANDS=CEILING(REAL(NOCC(ISP))/REAL(WP%WDES%NB_PAR))
      ELSE
         NBANDS=WP%WDES%NBANDS
      ENDIF

! test
      IF (LUSEQIJB) THEN
         ALLOCATE(QIJB(SIZE(CQIJ,1),SIZE(CQIJ,2),SIZE(CQIJ,3),SIZE(CQIJ,4)))
         CALL SETQIJB(K2-K1,WP%WDES,T_INFO,P,LATT_CUR,QIJB)
      ENDIF
! test

      strips: DO NPOS=1,NBANDS,NSTRIP
         NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

         ! when K1 and K2 refer to the same point in the IBZ
         ! then the bands [NPOS,NPOS+NSTRIP_ACT] must be
         ! temporarily redistributed from "over pw components"
         ! to "over bands"
         IF (KPOINTS_FULL%NEQUIV(NK)==KPOINTS_FULL%NEQUIV(NQ).AND. &
        &    NK==KPOINTS_FULL%NEQUIV(NK)) THEN
            CALL REDISTRIBUTE_PW(ELEMENTS(WK,NPOS,NPOS+NSTRIP_ACT-1))
            CALL REDISTRIBUTE_PROJ(ELEMENTS(WK,NPOS,NPOS+NSTRIP_ACT-1))
         ENDIF
         
         instrip: DO N=NPOS,NPOS+NSTRIP_ACT-1
            NP=N-NPOS+1
            
            IF (NQ==KPOINTS_FULL%NEQUIV(NQ)) THEN
               CALL W1_COPY(ELEMENT(WP, WDESQ, N, ISP), WQ)
!              CALL FFTWAV_W1(WQ)
            ELSE
               CALL W1_ROTATE(WP,P,LATT_CUR,N,ISP,ROT_HANDLE,WQ)
! #ifndef gammareal
!              ! symmetry must be considered if the wavefunctions for
!              ! k-point NQ are not stored in WP (i.e. not in the IBZ)
!              LSHIFT=.FALSE.
!              IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
!                  (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
!                  (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
!              CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WP, WDESQ_IRZ, N, ISP_IRZ), &
!                 ROT_HANDLE, P, LATT_CUR, LSHIFT)
! #endif
            ENDIF
            CALL FFTWAV_W1(WQ)
            
            IF (LONECENTER.AND.(.NOT.LUSEQIJB)) THEN
               !  calculate Qij < beta_j |wq>   (Qij = \int Q_ij(r) dr)
               CALL OVERL_AND_APPLY_PHASE_SHIFT(PHASE,T_INFO,CQIJ,WDESQ,WQ,WHAM%CPROJ(:,NP))
            ELSEIF (LONECENTER.AND.LUSEQIJB) THEN
               !  calculate Qij(k2-k1) < beta_j |wq>   (Qij(k2-k1) = \int exp[-i(k2-k1)r]Q_ij(r) dr)
               CALL OVERL_AND_APPLY_PHASE_SHIFT_C(PHASE,T_INFO,QIJB,WDESQ,WQ,WHAM%CPROJ(:,NP))
            ELSE
               WHAM%CPROJ(:,NP)=0
            ENDIF
            IF (LPLANEWAVE) THEN
               !  evaluate sv|wq> (in real space)
               CALL APPLY_PHASE_SHIFT_GRID(WQ,SV)
               ! fft sv|wq> -> reciprocal space (layout consistent with kpoint NK)
               DO ISPINOR=0,WDESK%NRSPINORS-1
                  CALL FFTEXT(WDESK%NGVECTOR,WDESK%NINDPW(1), &
                              WQ%CR(1+ISPINOR*WDESQ%GRID%MPLWV), &
                              WHAM%CW(1+ISPINOR*WDESK%NGVECTOR,NP),WDESK%GRID,.FALSE.)
               ENDDO
            ELSE
               WHAM%CW(:,NP)=0
            ENDIF
         ENDDO instrip
         ! when K1 and K2 refer to the same point in the IBZ
         ! then the bands [NPOS,NPOS+NSTRIP_ACT] must now be
         ! redistributed from "over bands" to "over pw components"
         IF (KPOINTS_FULL%NEQUIV(NK)==KPOINTS_FULL%NEQUIV(NQ).AND. &
        &    NK==KPOINTS_FULL%NEQUIV(NK)) THEN
            CALL REDISTRIBUTE_PW(ELEMENTS(WK,NPOS,NPOS+NSTRIP_ACT-1))
            CALL REDISTRIBUTE_PROJ(ELEMENTS(WK,NPOS,NPOS+NSTRIP_ACT-1))
         ENDIF
         ! redistribution of WHAM from "over bands" to
         ! "over pw components"
         CALL REDISTRIBUTE_PW(ELEMENTS(WHAM,1,NSTRIP_ACT))
         CALL REDISTRIBUTE_PROJ(ELEMENTS(WHAM,1,NSTRIP_ACT))

         ! < w_{m,k1} | sv | w_{n,k2} >
         NPOS_RED  =(NPOS-1)*WDESK%NB_PAR+1
         NSTRIP_RED=NSTRIP_ACT*WDESK%NB_PAR
         CALL ORTH2(&
        &           WK%CW_RED(1,1),WHAM%CW_RED(1,1), &
        &           WK%CPROJ_RED(1,1),WHAM%CPROJ_RED(1,1), &
        &           WK%WDES1%NB_TOT,NPOS_RED,NSTRIP_RED, &
        &           WK%WDES1%NPL_RED,WK%WDES1%NPRO_RED, &
        &           WK%WDES1%NRPLWV_RED,WK%WDES1%NPROD_RED, &
        &           S(1,1))
                  
      ENDDO strips
      ! communicate
      statmg
      CALLMPI( M_sum_g(WDESK%COMM_KIN,S(1,1),WDESK%NB_TOT*WDESK%NB_TOT))

      ! back distribute wavefunctions and projections at K1
      ! from "over pw components" to "over bands"
      CALL REDISTRIBUTE_PW(WK)
      CALL REDISTRIBUTE_PROJ(WK)

      ! some deallocation to be done
      CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
      IF (NK/=KPOINTS_FULL%NEQUIV(NK)) CALL DELWAVA(WK)
      CALL DELWAV(WQ,.TRUE.)
      CALL DELWAVA(WHAM)
      DEALLOCATE(SV)
! test
      IF (LUSEQIJB) DEALLOCATE(QIJB)
! test

      RETURN
      END SUBROUTINE CALC_OVERLAP_
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE SET_PHASE_SHIFT(K1,K2,PHASE)
      USE prec
      USE full_kpoints
      IMPLICIT NONE
      REAL(q) K1(3),K2(3),PHASE(3)
      ! local variables
      INTEGER NK1,NK2
      INTEGER I
      REAL(q) DK1(3),DK2(3)
      REAL(q), PARAMETER :: TINY=1E-4
      LOGICAL LERR
      NK1=KPOINT_IN_FULL_GRID(K1,KPOINTS_FULL)
      NK2=KPOINT_IN_FULL_GRID(K2,KPOINTS_FULL)
      ! take the difference between K1 and the
      ! corresponding kpoint in the 1st BZ
      DO I=1,3
         DK1(I)=K1(I)-KPOINTS_FULL%VKPT(I,NK1)
      ENDDO
      ! take the difference between K2 and the
      ! corresponding kpoint in the 1st BZ
      DO I=1,3
         DK2(I)=K2(I)-KPOINTS_FULL%VKPT(I,NK2)
      ENDDO
      ! only full reciprocal lattice vectors enter into PHASE
      LERR=.FALSE.
      DO I=1,3
!        IF ((ABS(DK1(I))-1._q)>TINY) LERR=.TRUE. 
!        IF (ABS(MOD(DK1(I)+10._q,1._q))>TINY) LERR=.TRUE.
         IF (ABS(DK1(I)-REAL(NINT(DK1(I)))) > TINY ) LERR=.TRUE.
      ENDDO
      IF (LERR) THEN
         WRITE(*,*) 'SET_PHASE_SHIFT: ERROR, DK1 is not a reciprocal lattice vector'
         WRITE(*,'(3F8.3,4X,3F8.3)') K1(:),KPOINTS_FULL%VKPT(:,NK1)
         STOP
      ENDIF
      LERR=.FALSE.
      DO I=1,3
!        IF ((ABS(DK2(I))-1._q)>TINY) LERR=.TRUE. 
!        IF (ABS(MOD(DK2(I)+10._q,1._q))>TINY) LERR=.TRUE. 
         IF (ABS(DK2(I)-REAL(NINT(DK2(I)))) > TINY ) LERR=.TRUE.
      ENDDO
      IF (LERR) THEN
         WRITE(*,*) 'SET_PHASE_SHIFT: ERROR, DK2 is not a reciprocal lattice vector'
         WRITE(*,'(3F8.3,4X,3F8.3)') K2(:),KPOINTS_FULL%VKPT(:,NK2)
         STOP
      ENDIF
      
      PHASE=-(DK2-DK1)

      RETURN
      END SUBROUTINE SET_PHASE_SHIFT
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************      
      SUBROUTINE OVERL_AND_APPLY_PHASE_SHIFT( &
     &   PHASE,T_INFO,CQIJ,WDES1,W1,CPROJ_RESUL)
      USE prec
      USE wave
      USE poscar
      USE constant
      IMPLICIT NONE
      TYPE(wavedes1) WDES1
      TYPE(wavefun1) W1
      TYPE(type_info) T_INFO
      REAL(q) PHASE(3)
      OVERLAP CQIJ(:,:,:,:)
      GDEF, OPTIONAL :: CPROJ_RESUL(:)
      ! local variables
      GDEF CRESUL(WDES1%NPRO)
      INTEGER ISPINOR,ISPINOR_
      INTEGER NPRO,NPRO_
      INTEGER NIS,NT,NI,NIP
      INTEGER LMMAXC,L,LP,N
      COMPLEX(q) CGDR,CVALUE
      ! consistency tests
      IF ((SIZE(CQIJ,1)/=SIZE(CQIJ,2)).OR.(SIZE(CQIJ,1)<MAXVAL(WDES1%LMMAX)).OR. &
     &    (SIZE(CQIJ,3)/=WDES1%NIONS)) THEN
         WRITE(*,*) 'OVERL_AND_APPLY_PHASE_SHIFT: ERROR, CQIJ inconsistently dimensioned'
         STOP
      ENDIF

      CVALUE=(1._q,0._q)
#ifdef gammareal
      IF (LIMAGINARY_PART) CVALUE=(0._q,-1._q)
#endif

      CRESUL=0
      spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
      DO ISPINOR_=0,WDES1%NRSPINORS-1

         NPRO =ISPINOR *(WDES1%NPRO/2)
         NPRO_=ISPINOR_*(WDES1%NPRO/2)

         NIS =1
         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC/=0) THEN
               DO NI=NIS,WDES1%NITYP(NT)+NIS-1
                  NIP=NI_GLOBAL(NI,WDES1%COMM_INB)

                  CGDR=CITPI*(T_INFO%POSION(1,NIP)*PHASE(1)+ &
                 &              T_INFO%POSION(2,NIP)*PHASE(2)+ &
                 &               T_INFO%POSION(3,NIP)*PHASE(3))

                  DO L =1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                     DO LP=1,LMMAXC
                        CRESUL(L+NPRO)=CRESUL(L+NPRO)+ &
                           CQIJ(LP,L,NI,ISPINOR_+2*ISPINOR+1)*W1%CPROJ(LP+NPRO_)*EXP(CGDR)*CVALUE
                     ENDDO
                                          
                  ENDDO
                  NPRO = LMMAXC+NPRO
                  NPRO_= LMMAXC+NPRO_
               ENDDO
            ENDIF
            NIS = NIS+WDES1%NITYP(NT)
         ENDDO
      ENDDO
      ENDDO spinor
      
      ! store into CPROJ_RESUL or back into W1%CPROJ
      IF (PRESENT(CPROJ_RESUL)) THEN
         IF (SIZE(CPROJ_RESUL)<SIZE(CRESUL)) THEN
            WRITE(*,*) 'OVERL_AND_APPLY_PHASE_SHIFT: ERROR, CPROJ_RESUL too small'
            STOP
         ENDIF
         DO N=1,WDES1%NPRO
            CPROJ_RESUL(N)=CRESUL(N)
         ENDDO
      ELSE
         DO N=1,WDES1%NPRO
            W1%CPROJ(N)=CRESUL(N)
         ENDDO
      ENDIF

      RETURN
      END SUBROUTINE OVERL_AND_APPLY_PHASE_SHIFT
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************      
      SUBROUTINE OVERL_AND_APPLY_PHASE_SHIFT_C( &
     &   PHASE,T_INFO,CQIJ,WDES1,W1,CPROJ_RESUL)
      USE prec
      USE wave
      USE poscar
      USE constant
      IMPLICIT NONE
      TYPE(wavedes1) WDES1
      TYPE(wavefun1) W1
      TYPE(type_info) T_INFO
      REAL(q) PHASE(3)
      COMPLEX(q) CQIJ(:,:,:,:)
      GDEF, OPTIONAL :: CPROJ_RESUL(:)
      ! local variables
      GDEF CRESUL(WDES1%NPRO)
      INTEGER ISPINOR,ISPINOR_
      INTEGER NPRO,NPRO_
      INTEGER NIS,NT,NI,NIP
      INTEGER LMMAXC,L,LP,N
      COMPLEX(q) CGDR,CVALUE
      ! consistency tests
      IF ((SIZE(CQIJ,1)/=SIZE(CQIJ,2)).OR.(SIZE(CQIJ,1)<MAXVAL(WDES1%LMMAX)).OR. &
     &    (SIZE(CQIJ,3)/=WDES1%NIONS)) THEN
         WRITE(*,*) 'OVERL_AND_APPLY_PHASE_SHIFT_C: ERROR, CQIJ inconsistently dimensioned'
         STOP
      ENDIF

      CVALUE=(1._q,0._q)
#ifdef gammareal
      IF (LIMAGINARY_PART) CVALUE=(0._q,-1._q)
#endif

      CRESUL=0
      spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
      DO ISPINOR_=0,WDES1%NRSPINORS-1

         NPRO =ISPINOR *(WDES1%NPRO/2)
         NPRO_=ISPINOR_*(WDES1%NPRO/2)

         NIS =1
         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC/=0) THEN
               DO NI=NIS,WDES1%NITYP(NT)+NIS-1
                  NIP=NI_GLOBAL(NI,WDES1%COMM_INB)

                  CGDR=CITPI*(T_INFO%POSION(1,NIP)*PHASE(1)+ &
                 &              T_INFO%POSION(2,NIP)*PHASE(2)+ &
                 &               T_INFO%POSION(3,NIP)*PHASE(3))

                  DO L =1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                     DO LP=1,LMMAXC
                        CRESUL(L+NPRO)=CRESUL(L+NPRO)+ &
                           CQIJ(LP,L,NI,ISPINOR_+2*ISPINOR+1)*W1%CPROJ(LP+NPRO_)*EXP(CGDR)*CVALUE
                     ENDDO
                                          
                  ENDDO
                  NPRO = LMMAXC+NPRO
                  NPRO_= LMMAXC+NPRO_
               ENDDO
            ENDIF
            NIS = NIS+WDES1%NITYP(NT)
         ENDDO
      ENDDO
      ENDDO spinor
      
      ! store into CPROJ_RESUL or back into W1%CPROJ
      IF (PRESENT(CPROJ_RESUL)) THEN
         IF (SIZE(CPROJ_RESUL)<SIZE(CRESUL)) THEN
            WRITE(*,*) 'OVERL_AND_APPLY_PHASE_SHIFT_C: ERROR, CPROJ_RESUL too small'
            STOP
         ENDIF
         DO N=1,WDES1%NPRO
            CPROJ_RESUL(N)=CRESUL(N)
         ENDDO
      ELSE
         DO N=1,WDES1%NPRO
            W1%CPROJ(N)=CRESUL(N)
         ENDDO
      ENDIF

      RETURN
      END SUBROUTINE OVERL_AND_APPLY_PHASE_SHIFT_C


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE APPLY_PHASE_SHIFT_CPROJ( &
     &   PHASE,T_INFO,WDES1,W1,CPROJ_RESUL)
      USE prec
      USE wave
      USE poscar
      USE constant
      IMPLICIT NONE
      TYPE(wavedes1) WDES1
      TYPE(wavefun1) W1
      TYPE(type_info) T_INFO
      REAL(q) PHASE(3)
      GDEF, OPTIONAL :: CPROJ_RESUL(:)
      ! local variables
      GDEF CRESUL(WDES1%NPRO)
      INTEGER ISPINOR
      INTEGER NPRO
      INTEGER NIS,NT,NI,NIP
      INTEGER LMMAXC,L,N
      COMPLEX(q) CGDR,CVALUE

      CVALUE=(1._q,0._q)
#ifdef gammareal
      IF (LIMAGINARY_PART) CVALUE=(0._q,-1._q)
#endif

      CRESUL=0
      spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
         NPRO =ISPINOR *(WDES1%NPRO/2)
         NIS =1
         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC/=0) THEN
               DO NI=NIS,WDES1%NITYP(NT)+NIS-1
                  NIP=NI_GLOBAL(NI,WDES1%COMM_INB)

                  CGDR=CITPI*(T_INFO%POSION(1,NIP)*PHASE(1)+ &
                 &              T_INFO%POSION(2,NIP)*PHASE(2)+ &
                 &               T_INFO%POSION(3,NIP)*PHASE(3))

                  DO L=1,LMMAXC
                     CRESUL(L+NPRO)=W1%CPROJ(L+NPRO)*EXP(CGDR)*CVALUE
                  ENDDO                                          
                  NPRO=LMMAXC+NPRO
               ENDDO
            ENDIF
            NIS = NIS+WDES1%NITYP(NT)
         ENDDO
      ENDDO spinor
      
      ! store into CPROJ_RESUL or back into W1%CPROJ
      IF (PRESENT(CPROJ_RESUL)) THEN
         IF (SIZE(CPROJ_RESUL)<SIZE(CRESUL)) THEN
            WRITE(*,*) 'APPLY_PHASE_SHIFT_CPROJ: ERROR, CPROJ_RESUL too small'
            STOP
         ENDIF
         DO N=1,WDES1%NPRO
            CPROJ_RESUL(N)=CRESUL(N)
         ENDDO
      ELSE
         DO N=1,WDES1%NPRO
            W1%CPROJ(N)=CRESUL(N)
         ENDDO
      ENDIF

      RETURN
      END SUBROUTINE APPLY_PHASE_SHIFT_CPROJ      
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE SET_PHASE_SHIFT_GRID(PHASE,WDES1,SV)
      USE prec
      USE wave
      IMPLICIT NONE
      TYPE(wavedes1) WDES1
      REAL(q) PHASE(3)
      COMPLEX(q) SV(WDES1%GRID%MPLWV)
      ! local variables
      INTEGER I,IPW,IX,IY,IZ
      INTEGER G(3)
      REAL(q) FFTSCA
      COMPLEX(q) CVALUE
      COMPLEX(q) SVWORK(WDES1%GRID%MPLWV)      
      LOGICAL LFOUND

      G=NINT(PHASE)

      CVALUE=(1._q,0._q)
#ifdef gammareal
      IF (LIMAGINARY_PART) CVALUE=(0._q,-1._q)
      ! this is a bit ad hoc but should work as
      ! long as only 1 component of G/=0
      DO I=1,3
         IF (SIGN(1,G(I))==-1) THEN
            G(I)=-G(I)
            IF (LIMAGINARY_PART) CVALUE=-CVALUE
         ENDIF
      ENDDO
#endif

      SVWORK=0

      FFTSCA=1._q
#ifdef gammareal
      IF (G(1)/=0.OR.G(2)/=0.OR.G(3)/=0) THEN
         FFTSCA=1/SQRT(2._q)
      ENDIF
#endif
      LFOUND=.FALSE.
      DO IPW=1,WDES1%NGVECTOR
         IX=WDES1%IGX(IPW)
         IY=WDES1%IGY(IPW)
         IZ=WDES1%IGZ(IPW)
         IF (IX==G(1).AND.IY==G(2).AND.IZ==G(3)) THEN
            SVWORK(IPW)=CVALUE*FFTSCA
            LFOUND=.TRUE.
         ENDIF
      ENDDO
      ! have we been able to set the correct component?
      IF (.NOT.LFOUND) THEN
         WRITE(*,'(A,3I3)') 'SET_PHASE_SHIFT_GRID: ERROR, unable to set pw-component:',G
         STOP
      ENDIF
      ! take the "shifting potential" to real space
      CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),SV,SVWORK,WDES1%GRID)

      RETURN
      END SUBROUTINE SET_PHASE_SHIFT_GRID
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE SET_PHASE_SHIFT_GRID_RSPACE(PHASE,WDES1,SV)
      USE prec
      USE wave
      USE constant
      IMPLICIT NONE
      TYPE(wavedes1) WDES1
      REAL(q) PHASE(3)
      COMPLEX(q) SV(WDES1%GRID%MPLWV)
      ! local variables
      INTEGER N1,N2,N3,IND
      REAL(q) G(3)
      REAL(q) A1,A2,A3
      REAL(q) F1,F2,F3
      REAL(q) PHI
      COMPLEX(q) CVALUE,SHIFT

      G=PHASE

      SHIFT=0._q
#ifdef gammareal
      IF (LIMAGINARY_PART) SHIFT=-0.25_q
#endif

      SV=0

      F1=1._q/WDES1%GRID%NGX
      F2=1._q/WDES1%GRID%NGY
      F3=1._q/WDES1%GRID%NGZ

      IF (WDES1%GRID%RL%NFAST==3) THEN
! z is fast index
         DO N2=0,WDES1%GRID%NGY-1
         A2=N2*F2       

         DO N1=0,WDES1%GRID%NGX-1
         A1=N1*F1

         DO N3=0,WDES1%GRID%NGZ-1
         A3=N3*F3

#ifdef gammareal
            PHI=TPI*(A1*G(1)+A2*G(2)+A3*G(3)+SHIFT)
            CVALUE=COS(PHI)
#else
            PHI=TPI*(A1*G(1)+A2*G(2)+A3*G(3))
            CVALUE=CMPLX(COS(PHI),SIN(PHI),q)
#endif            
       
            IND=N3+N1*WDES1%GRID%NGZ+N2*WDES1%GRID%NGZ*WDES1%GRID%NGX+1
            
            SV(IND)=CVALUE
                
         ENDDO
         ENDDO
         ENDDO
         
      ELSE
! x is fast index
         DO N3=0,WDES1%GRID%NGZ-1
         A3=N3*F3

         DO N2=0,WDES1%GRID%NGY-1
         A2=N2*F2       
       
         DO N1=0,WDES1%GRID%NGX-1
         A1=N1*F1

#ifdef gammareal
            PHI=TPI*(A1*G(1)+A2*G(2)+A3*G(3)+SHIFT)
            CVALUE=COS(PHI)
#else
            PHI=TPI*(A1*G(1)+A2*G(2)+A3*G(3))
            CVALUE=CMPLX(COS(PHI),SIN(PHI),q)
#endif            

            IND=N1+N2*WDES1%GRID%NGX+N3*WDES1%GRID%NGX*WDES1%GRID%NGY+1
            
            SV(IND)=CVALUE

         ENDDO
         ENDDO
         ENDDO

      ENDIF

      RETURN
      END SUBROUTINE SET_PHASE_SHIFT_GRID_RSPACE
      

!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE APPLY_PHASE_SHIFT_GRID(W1,SV)
      USE prec
      USE wave
      IMPLICIT NONE
      TYPE (wavefun1) W1
      COMPLEX(q) SV(:)
      ! local variables
      INTEGER M,MM
      REAL(q) WEIGHT
      ! consistency check
      IF (SIZE(SV)/=W1%WDES1%GRID%RL%NP) THEN
         WRITE(*,*) 'APPLY_PHASE_SHIFT_GRID: ERROR, grid inconsistency'
         STOP
      ENDIF
      WEIGHT=1._q/W1%WDES1%GRID%NPLWV
!DIR$ IVDEP
!OCL NOVREL
      DO M=1,W1%WDES1%GRID%RL%NP
         W1%CR(M)=SV(M)*W1%CR(M)*WEIGHT
      ENDDO
      IF (W1%WDES1%NRSPINORS==2) THEN
!DIR$ IVDEP
!OCL NOVREL
         DO M=1,W1%WDES1%GRID%RL%NP
            MM =M+W1%WDES1%GRID%MPLWV
            W1%CR(MM)=SV(M)*W1%CR(MM)*WEIGHT
         ENDDO
      ENDIF
      
      RETURN
      END SUBROUTINE APPLY_PHASE_SHIFT_GRID


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE CHECK_OCCUPATIONS(W)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE(wavespin) W
      ! local variables
      INTEGER ISP,IK,N
      REAL(q),PARAMETER :: TINY=1E-4_q

      IF (.NOT.ALLOCATED(NOCC)) THEN
         ALLOCATE(NOCC(W%WDES%ISPIN))
      ENDIF

      DO ISP=1,W%WDES%ISPIN
         NOCC(ISP)=0
         DO IK=1,SIZE(W%FERTOT,2)
            DO N=1,W%WDES%NB_TOT
               IF (W%FERTOT(N,IK,ISP).LE.TINY) EXIT
            ENDDO
            N=N-1
            IF (IK==1) THEN
               NOCC(ISP)=N
            ELSE
               IF (N/=NOCC(ISP)) THEN
!                 WRITE(*,*) 'CHECK_OCCUPATIONS: ERROR, system is not insulating.'
                  LINSULATING=.FALSE.
                  RETURN
               ENDIF
            ENDIF
         ENDDO
      ENDDO
      LINSULATING=.TRUE.

      RETURN
      END SUBROUTINE CHECK_OCCUPATIONS


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE SETUP_STRINGS(LSTRINGS)
      USE prec
      USE sym_prec
      USE full_kpoints
      IMPLICIT NONE
      LOGICAL LSTRINGS
      ! local variables
      INTEGER N
      INTEGER IDIR,IK,ISTR
      INTEGER NK(3)
      REAL(q) K(2)
      REAL(q), ALLOCATABLE :: ROOT_MEMBER(:,:,:)
      INTEGER, ALLOCATABLE :: NUM_MEMBERS(:,:)
      INTEGER, ALLOCATABLE :: MAP_TO_KPOINT(:,:,:)
      LOGICAL :: LERR=.FALSE.

      IF (ALLOCATED(MAP_TO_STRING)) DEALLOCATE(MAP_TO_STRING)

      N=MAX(KPOINTS_FULL%NKPX,KPOINTS_FULL%NKPY,KPOINTS_FULL%NKPZ)

      ALLOCATE(ROOT_MEMBER(2,N*N,3))      
      ALLOCATE(NUM_MEMBERS(N*N,3))
      ALLOCATE(MAP_TO_STRING(KPOINTS_FULL%NKPTS,3))
      ALLOCATE(MAP_TO_KPOINT(N,N*N,3))

      NSTRINGS=0
      NUM_MEMBERS=0
      MAP_TO_STRING=0
            
      DO IDIR=1,3
         DO IK=1,KPOINTS_FULL%NKPTS
            SELECT CASE(IDIR)
               CASE(1)
                  K(1)=KPOINTS_FULL%VKPT(2,IK)
                  K(2)=KPOINTS_FULL%VKPT(3,IK)
               CASE(2)
                  K(1)=KPOINTS_FULL%VKPT(1,IK)
                  K(2)=KPOINTS_FULL%VKPT(3,IK)
               CASE(3)
                  K(1)=KPOINTS_FULL%VKPT(1,IK)
                  K(2)=KPOINTS_FULL%VKPT(2,IK)
            END SELECT
            IF (NSTRINGS(IDIR)>0) THEN
               ! does K belong to an existing string?
               DO ISTR=1,NSTRINGS(IDIR)
                  IF ( & 
                    (ABS(MOD(K(1)-ROOT_MEMBER(1,ISTR,IDIR)+10.5_q,1._q)-0.5_q)<TINY) .AND. &
                    (ABS(MOD(K(2)-ROOT_MEMBER(2,ISTR,IDIR)+10.5_q,1._q)-0.5_q)<TINY)) EXIT
               ENDDO
               IF (ISTR>NSTRINGS(IDIR)) THEN
               ! add a new string
                  NSTRINGS(IDIR)=NSTRINGS(IDIR)+1
                  ROOT_MEMBER(:,NSTRINGS(IDIR),IDIR)=K(:)
                  NUM_MEMBERS(NSTRINGS(IDIR),IDIR)=1
                  MAP_TO_STRING(IK,IDIR)=NSTRINGS(IDIR)
                  MAP_TO_KPOINT(NUM_MEMBERS(NSTRINGS(IDIR),IDIR),NSTRINGS(IDIR),IDIR)=IK
               ELSE
               ! add a member to an existing string
                  NUM_MEMBERS(ISTR,IDIR)=NUM_MEMBERS(ISTR,IDIR)+1
                  MAP_TO_STRING(IK,IDIR)=ISTR
                  MAP_TO_KPOINT(NUM_MEMBERS(ISTR,IDIR),ISTR,IDIR)=IK
               ENDIF
            ELSE
            ! add the initial string
               ROOT_MEMBER(:,1,IDIR)=K(:)
               NSTRINGS(IDIR)=1
               NUM_MEMBERS(1,IDIR)=1
               MAP_TO_STRING(IK,IDIR)=1
               MAP_TO_KPOINT(NUM_MEMBERS(NSTRINGS(IDIR),IDIR),NSTRINGS(IDIR),IDIR)=IK
            ENDIF
         ENDDO
      ENDDO
      
      ! test the consistency of the string attributions
      DO IDIR=1,3
         SELECT CASE(IDIR)
            CASE(1)
               N=KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ
            CASE(2)
               N=KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPZ
            CASE(3)
               N=KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY
         END SELECT
         ! check the number of strings found         
         IF (NSTRINGS(IDIR)/=N) THEN
            WRITE(*,*) 'SETUP_STRINGS: ERROR, did not find all strings', &
           & IDIR,NSTRINGS(IDIR),N
            LERR=.TRUE.
         ENDIF
         ! check the number of members per string
         DO ISTR=1,NSTRINGS(IDIR)
            IF (NUM_MEMBERS(ISTR,IDIR)/=(KPOINTS_FULL%NKPTS/N)) THEN
               WRITE(*,*) 'SETUP_STRINGS: ERROR, did not find all members', &
              & IDIR,ISTR,NUM_MEMBERS(ISTR,IDIR),KPOINTS_FULL%NKPTS/N
               LERR=.TRUE.
            ENDIF
         ENDDO
      ENDDO

!     IF (LDOIO) THEN
      IF (.FALSE.) THEN
      DO IDIR=1,3
         WRITE(*,*) 'direction:',IDIR
         DO ISTR=1,NSTRINGS(IDIR)
            WRITE(*,*) 'string:',ISTR
            WRITE(*,'(3F14.7,I4)') &
           &   (KPOINTS_FULL%VKPT(:,MAP_TO_KPOINT(N,ISTR,IDIR)), &
           &    MAP_TO_STRING(MAP_TO_KPOINT(N,ISTR,IDIR),IDIR), N=1,NUM_MEMBERS(ISTR,IDIR))
         ENDDO
      ENDDO
      ENDIF
      
      DEALLOCATE(NUM_MEMBERS)
      DEALLOCATE(ROOT_MEMBER)
      DEALLOCATE(MAP_TO_KPOINT)

      LSTRINGS=.NOT.LERR

      RETURN
      END SUBROUTINE SETUP_STRINGS


!************************ SUBROUTINE SETYLM_AUG2 ***********************
!
! this subroutine performes the following tasks
! ) finds the points, which are within a certain cutoff around one ion
! ) calculates the distance of each of this points from the ion
! ) calculates the spherical harmonics Y_lm(Omega(r-R(ion))
! DISX,Y,Z are additional displacements of the ions
!
! N.B. the only difference between this subroutine and the SETYLM_AUG
! in us.F, is that SETYLM_AUG2 also writes the arrays XS,YS and ZS 
! containing the coordinates of the points at which the spherical
! harmonics are given.
! 
!***********************************************************************
      SUBROUTINE SETYLM_AUG2(GRID,LATT_CUR,POSION,PSDMAX,NPSRNL, &
     &  LMYDIM,LYDIM,YLM,IRMAX,INDMAX,DISX,DISY,DISZ,XS,YS,ZS,DIST)
       
      USE prec
      USE mpimy
      USE mgrid
      USE lattice
      USE asa
      USE constant

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR

      REAL(q) POSION(3)
      REAL(q) DIST(IRMAX)
      REAL(q) YLM(IRMAX,LMYDIM)

      REAL(q) XS(IRMAX),YS(IRMAX),ZS(IRMAX)

      IF ((LYDIM+1)**2 > LMYDIM) THEN
         WRITE(0,*)'internal error: LMYDIM is too small',LYDIM,LMYDIM
         STOP
      ENDIF

      XS=0;YS=0;ZS=0;DIST=0

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRID%NGX
      F2=1._q/GRID%NGY
      F3=1._q/GRID%NGZ

      ARGSC=NPSRNL/PSDMAX

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= PSDMAX*LATT_CUR%BNORM(1)*GRID%NGX
      D2= PSDMAX*LATT_CUR%BNORM(2)*GRID%NGY
      D3= PSDMAX*LATT_CUR%BNORM(3)*GRID%NGZ

      N3LOW= INT(POSION(3)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
      N2LOW= INT(POSION(2)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
      N1LOW= INT(POSION(1)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

      N3HI = INT(POSION(3)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
      N2HI = INT(POSION(2)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
      N1HI = INT(POSION(1)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX
!-----------------------------------------------------------------------
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      IND=1

      DO N2=N2LOW,N2HI
      X2=(N2*F2-POSION(2))
      N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-POSION(1))
      N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

      NCOL=GRID%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE
      IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'internal ERROR SETYLM:',N1P+1,N2P+1,NCOL, &
           GRID%RL%I2(NCOL),N1P , GRID%RL%I3(NCOL),N2P
        STOP
      ENDIF

!OCL SCALAR
      DO N3=N3LOW,N3HI
        X3=(N3*F3-POSION(3))

        X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
        Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
        Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

        D=SQRT(X*X+Y*Y+Z*Z)
        ARG=(D*ARGSC)+1
        NADDR=INT(ARG)

        IF (NADDR<NPSRNL) THEN
          ZZ=Z-DISZ
          YY=Y-DISY
          XX=X-DISX
        ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
        ! is done using the well known formula  | R+d | = | R | + d . R/|R|
        ! this improves the stability of finite differences considerable
          IF (D<1E-4_q) THEN
            DIST(IND)=1E-4_q
          ELSE
            DIST(IND)=MAX(D-(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
          ENDIF

          XS(IND)  =XX/DIST(IND)
          YS(IND)  =YY/DIST(IND)
          ZS(IND)  =ZZ/DIST(IND)

          IND=IND+1
        ENDIF
      ENDDO; ENDDO; ENDDO
#else
      IND=1

      DO N3=N3LOW,N3HI
      X3=(N3*F3-POSION(3))

      DO N2=N2LOW,N2HI
      X2=(N2*F2-POSION(2))

      DO N1=N1LOW,N1HI
      X1=(N1*F1-POSION(1))

      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(X*X+Y*Y+Z*Z)
      ARG=(D*ARGSC)+1
      NADDR=INT(ARG)

      IF (NADDR<NPSRNL) THEN
        ZZ=Z-DISZ
        YY=Y-DISY
        XX=X-DISX
        ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
        ! is done using the well known formula  | R+d | = | R | + d . R/|R|
        ! this improves the stability of finite differences considerable
        IF (D<1E-4_q) THEN
          DIST(IND)=1E-4_q
        ELSE
          DIST(IND)=MAX(D-(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
        ENDIF

        XS(IND)  =XX/DIST(IND)
        YS(IND)  =YY/DIST(IND)
        ZS(IND)  =ZZ/DIST(IND)

        IND=IND+1
      ENDIF
      ENDDO; ENDDO; ENDDO
#endif
!-----------------------------------------------------------------------
!  compare maximum index with INDMAX
!-----------------------------------------------------------------------
      INDMAX=IND-1
      IF (INDMAX>IRMAX) THEN
        WRITE(*,*) &
     &  'internal ERROR: DEPLE:  IRDMAX must be increased to',INT(INDMAX*1.1)
        STOP
      ENDIF

      CALL SETYLM(LYDIM,INDMAX,YLM,XS,YS,ZS)

      DO IND=1,INDMAX 
         XS(IND)=XS(IND)*DIST(IND)
         YS(IND)=YS(IND)*DIST(IND)
         ZS(IND)=ZS(IND)*DIST(IND)
      ENDDO

      RETURN
      END SUBROUTINE SETYLM_AUG2


!************************ SUBROUTINE SETDIJ_R **************************
!
! This subroutine calculates minus the dipole moment of the augmentation
! charges
!  tau_idir,i,j = - sum_r (r-R)_idir Q_i,j(r-R)
! 
!***********************************************************************
    SUBROUTINE SETDIJ_R(WDES,GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,dIJ,IRDMAX,IDIR)

      USE prec
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us

      IMPLICIT NONE

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER ::  GRIDC
      TYPE (transit)             C_TO_US  ! index table between GRIDC and GRIDUS
      TYPE (latt)                LATT_CUR
      TYPE (wavedes)             WDES

      REAL(q)                    SUM(3),d_LM(256)

      INTEGER                    IRDMAX   ! allocation required for augmentation
      LOGICAL                    LOVERL,LADDITIONAL
      INTEGER                    LMDIM 
      OVERLAP                    dIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q), ALLOCATABLE ::    XS(:),YS(:),ZS(:)
      REAL(q), ALLOCATABLE ::    DIST(:),DEP(:),YLM(:,:)
! local
      TYPE (potcar), POINTER :: PP

      INTEGER :: IDIR, LYDIM, LMYDIM, NI, NT, LYMAX, INDMAX, ISP_INC, ISP, LDEP_INDEX
      INTEGER :: LM, L, LMP, LP, LL, LLP, M, MPLOW, MP, INDYLM, INDPYL, IND
      REAL(q) :: RINPL, TSUM
      INTEGER, EXTERNAL :: MAXL_AUG, MAXL1
#ifdef MPI
      OVERLAP, ALLOCATABLE ::    RTMP(:,:,:,:)
      INTEGER NIP
      ALLOCATE(RTMP(LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
#else
#define RTMP dIJ
#endif

      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)
      IF (LADDITIONAL) THEN
         GRIDC => GRIDUS
      ELSE
         GRIDC => GRIDC_
      ENDIF

!=======================================================================
      RTMP  =0
!=======================================================================

      IF (.NOT.LOVERL) GOTO 2000

      LYDIM=MAXL_AUG(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE(XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX), &
     &  DIST(IRDMAX),DEP(IRDMAX),YLM(IRDMAX,LMYDIM))
!=======================================================================
! loop over all ions
!=======================================================================
      RINPL=1._q/GRIDC%NPLWV

      ion: DO NI=1,T_INFO%NIONS
      NT=T_INFO%ITYP(NI)
      PP=>PP_POINTER(P, NI, NT)
      
      ! for this ion (this type of ion) no depletion charge
      IF (PP%PSDMAX==0) CYCLE
!-----------------------------------------------------------------------
! calculate the spherical harmonics YLM and the distance DIST  between
! grid-points and central atom (DEP is a work-arrays)
!-----------------------------------------------------------------------
      LYMAX=MAXL1(PP)
      IF ( ASSOCIATED(PP%QPAW) ) THEN
         LYMAX=LYMAX*2
      ENDIF
      CALL SETYLM_AUG2(GRIDC,LATT_CUR,T_INFO%POSION(1:3,NI), &
     &  PP%PSDMAX,NPSRNL,LMYDIM,LYMAX,YLM(1,1),IRDMAX,INDMAX, &
     &  0._q,0._q,0._q,XS(1),YS(1),ZS(1),DIST(1))

      IF (WDES%LNONCOLLINEAR) THEN
         ISP_INC=3
      ELSE
         ISP_INC=1
      ENDIF
 spin:DO ISP=1,WDES%NCDIJ,ISP_INC  

 lpaw:IF ( .NOT. ASSOCIATED(PP%QPAW) ) THEN
!=======================================================================
! US-PP
!=======================================================================
      LDEP_INDEX=1

    ! loop over all channels (l,epsilon)
      LM =1
      l_loop:  DO L =1,PP%LMAX
      LMP=LM
      lp_loop: DO LP=L,PP%LMAX
      IF (PP%NDEP(L,LP)==0) GOTO 510

      CALL SETDEP(PP%QDEP(1,1,LDEP_INDEX),PP%PSDMAX,NPSRNL, &
     &  LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
      LDEP_INDEX=LDEP_INDEX+ABS(PP%NDEP(L,LP))

    ! quantum numbers l and lp of these two channels
      LL =PP%LPS(L )
      LLP=PP%LPS(LP)

    ! loop over all m mp
      m_loop:  DO M=1,2*LL+1
      MPLOW=1 ; IF (L==LP) MPLOW=M
      mp_loop: DO MP=MPLOW,2*LLP+1

      !   calculate the indices into the array containing the spherical
      !   harmonics
         INDYLM =LL**2   +M
         INDPYL =LLP**2  +MP

         SUM=0
         TSUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC
         DO IND=1,INDMAX
            SUM(1)=SUM(1)-XS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(2)=SUM(2)-YS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(3)=SUM(3)-ZS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
         ENDDO

         !   add to array d_IJ and make symmetric
         RTMP(LM+M-1,LMP+MP-1,NI,ISP)=SUM(IDIR)*RINPL
         RTMP(LMP+MP-1,LM+M-1,NI,ISP)=SUM(IDIR)*RINPL
      ENDDO mp_loop
      ENDDO m_loop

 510  LMP=LMP+2*LLP+1
      ENDDO lp_loop
      LM =LM +2*LL +1
      ENDDO l_loop
   ELSE lpaw
!=======================================================================
!
! PAW approach
! 
!=======================================================================
      IF (LYMAX==0) CYCLE ! Is L=1 present?

      CALL SETDEP(PP%QDEP(1,1,1),PP%PSDMAX,NPSRNL, &
     &        LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))

      d_LM=0
      ! Only L=1 matters
      DO M=1,3 
         INDYLM=M+1
         SUM=0
         TSUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC
         DO IND=1,INDMAX
            SUM(1)=SUM(1)-XS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(2)=SUM(2)-YS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(3)=SUM(3)-ZS(IND)*DEP(IND)*YLM(IND,INDYLM)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)
         ENDDO
         d_LM(INDYLM)=SUM(IDIR)*RINPL
      ENDDO

      !   and transform LM -> ll'mm' i.e. d_LM -> d_IJ
      CALL CALC_DLLMM( RTMP(:,:,NI,ISP),d_LM,PP)


      ENDIF lpaw

      RTMP(:,:,NI,ISP)=RTMP(:,:,NI,ISP)*T_INFO%VCA(NT)
      
      ENDDO spin
!=======================================================================
      ENDDO ion
!=======================================================================
      DEALLOCATE(XS,YS,ZS,DIST,DEP,YLM)
#ifdef realmode
      CALLMPI( M_sum_d(GRIDC%COMM, RTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_z(GRIDC%COMM, RTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#endif

 2000 CONTINUE

#ifdef MPI
      ion2: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion2
         DO ISP=1,WDES%NCDIJ
            dIJ(:,:,NIP,ISP)=RTMP(:,:,NI,ISP)
         ENDDO
      ENDDO ion2
      DEALLOCATE(RTMP)
#else
#undef RTMP
#endif
      RETURN
    END SUBROUTINE SETDIJ_R


!************************ FUNCTION ONE_CENTRE_CHARGE ********************
!
! calculate the total one centre charge density
!
!***********************************************************************

    FUNCTION ONE_CENTRE_CHARGE(WDES, T_INFO, P, ION, LMDIM, CQIJ, CRHODE)
      USE prec
      USE wave
      USE poscar
      USE mpimy
      USE pseudo
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (type_info)   T_INFO
      INTEGER ION
      INTEGER LMDIM
      OVERLAP    CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
      OVERLAP    CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
      TYPE (potcar) :: P(T_INFO%NTYP)
      REAL(q) ONE_CENTRE_CHARGE
! local
      INTEGER NIP, NT
      INTEGER LM, LL, LHI, LOW, MMAX, L, LP, M
      REAL(q) SUM

      NIP=NI_LOCAL(ION, WDES%COMM_INB)  ! local storage index in CRHODE1
      NT=T_INFO%ITYP(ION)

      SUM=0
 
      IF (NIP/=0) THEN

      LOW=1
      LM =1
      block: DO
         LL=P(NT)%LPS(LOW)
         ! search block with same L
         DO LHI=LOW,P(NT)%LMAX
            IF (LL/=P(NT)%LPS(LHI)) EXIT
         ENDDO
         LHI=LHI-1
         MMAX=2*LL+1

         DO L =LOW,LHI
         DO LP=LOW,LHI
            DO M =0,MMAX-1
               SUM=SUM+CQIJ(LM+(L-LOW)*MMAX+M,LM+(LP-LOW)*MMAX+M,NIP,1) &
                    *CRHODE(LM+(L-LOW)*MMAX+M,LM+(LP-LOW)*MMAX+M,NIP,1)
            ENDDO
         ENDDO
         ENDDO
      
         ! set new LOW value and LM value and go on
         LM=LM+(LHI-LOW+1)*MMAX
         LOW=LHI+1
         IF (LOW > P(NT)%LMAX) EXIT block
      ENDDO block

      ENDIF
      
      CALLMPI( M_sum_s(WDES%COMM_INB, 1, SUM, 0._q, 0._q, 0._q))
      ONE_CENTRE_CHARGE=-SUM
      RETURN
    END FUNCTION ONE_CENTRE_CHARGE


!******************** SUBROUTINE SETQIJB *******************************
!      
!***********************************************************************
      SUBROUTINE SETQIJB( &
     & B,WDES,T_INFO,P,LATT_CUR,QIJB &
     &)
      USE asa
      USE poscar
      USE pseudo
      USE lattice
      USE constant
      USE wave_high
      USE radial, ONLY: SPHERICAL_BESSEL_TRANSFORM_R2Q
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(type_info) T_INFO
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      REAL(q) B(3)
      COMPLEX(q) QIJB(:,:,:,:)
      ! local variables
      TYPE (potcar), POINTER :: PP
      INTEGER ISP,ISP_INC
      INTEGER NI,NIP,NT
      INTEGER Ch1,Ch2
      INTEGER LMAX,LMMAX,LMDIM
      INTEGER L,LL,LLP,LM,LMP,M,MP
      INTEGER LDEP_INDEX,LMINDX,ISTART,IEND,IC
      REAL(q) BX,BY,BZ,BB
      REAL(q) XYZPTS(1,3)
      
      REAL(q), ALLOCATABLE :: YLM(:,:)
      REAL(q), ALLOCATABLE :: FR(:),FG(:,:),G(:)

      COMPLEX(q) CSET,CFAK
      COMPLEX(q) QLLMM(SIZE(QIJB,1),SIZE(QIJB,2))

      QIJB=0

      IF (.NOT.WDES%LOVERL) RETURN

      BX=(B(1)*LATT_CUR%B(1,1)+B(2)*LATT_CUR%B(1,2)+B(3)*LATT_CUR%B(1,3))*TPI
      BY=(B(1)*LATT_CUR%B(2,1)+B(2)*LATT_CUR%B(2,2)+B(3)*LATT_CUR%B(2,3))*TPI
      BZ=(B(1)*LATT_CUR%B(3,1)+B(2)*LATT_CUR%B(3,2)+B(3)*LATT_CUR%B(3,3))*TPI

      BB=MAX(SQRT(BX*BX+BY*BY+BZ*BZ),1E-10_q)

      LMDIM=SIZE(QIJB,1)
      
      LMAX=0
      DO NT=1,SIZE(P)
         LMAX=MAX(LMAX,MAXVAL(P(NT)%LPS))
      ENDDO
      LMAX=2*LMAX
      LMMAX=(LMAX+1)**2

      ALLOCATE(YLM(1,LMMAX))
      XYZPTS(1,1)=BX/BB; XYZPTS(1,2)=BY/BB; XYZPTS(1,3)=BZ/BB
      CALL SETYLM(LMAX,1,YLM,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

      ISP_INC=1
      IF (WDES%LNONCOLLINEAR) ISP_INC=3

      CSET=CMPLX(0._q,-1._q,q)
      
      ion: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI,WDES%COMM_INB); IF (NIP==0) CYCLE ion

! Strictly speaking we should account for e^(-i b \cdot R_ion)
! (see Eq. B.7 of J. Phys.:Condens. Matter 19, 036215 (2007))
!     CFAK=-CITPI*(T_INFO%POSION(1,NI)*B(1)+T_INFO%POSION(2,NI)*B(2)+T_INFO%POSION(3,NI)*B(3))
!     CFAK=EXP(CFAK)
! but in VASP this factor should not be added
!(see remarks below Eq 10.35 of the thesis of gK)
      CFAK=CMPLX(1._q,0._q,q)
      
      NT=T_INFO%ITYP(NI); PP=>PP_POINTER(P,NI,NT)
      IF (PP%PSDMAX==0) CYCLE ion

      ALLOCATE(FR(PP%R%NMAX),G(1),FG(1,LMAX+1))
      
      G(1)=BB
            
      spin: DO ISP=1,WDES%NCDIJ,ISP_INC
      
      QLLMM=0

      LDEP_INDEX=1 
      ! loop over all channels
      LM=1
      chan1: DO Ch1=1,PP%LMAX
      LMP=LM
      chan2: DO Ch2=Ch1,PP%LMAX

      ! angular moments l and lp of these channels
      LL=PP%LPS(Ch1)
      LLP=PP%LPS(Ch2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

      FR(:)=(PP%WAE(:,Ch1)*PP%WAE(:,Ch2)-PP%WPS(:,Ch1)*PP%WPS(:,Ch2))/PP%R%R(:)/PP%R%R(:)

      DO L=0,LL+LLP
         CALL SPHERICAL_BESSEL_TRANSFORM_R2Q(L,PP%R,FR,G(:),FG(:,L+1))
      ENDDO

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO IC=ISTART,IEND-1
            QLLMM(LM+M-1,LMP+MP-1)=QLLMM(LM+M-1,LMP+MP-1)+ &
           & CFAK*(CSET**REAL(JL(IC),q))*YLM3(IC)*YLM(1,JS(IC))*FG(1,JL(IC)+1)
         ENDDO
      ENDDO
      ENDDO

      ! fill lower triangle
      IF (Ch1/=Ch2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            QLLMM(LMP+MP-1,LM+M-1)=QLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      LMP=LMP+2*LLP+1
      ENDDO chan2
      LM =LM +2*LL +1
      ENDDO chan1

      QIJB(:,:,NIP,ISP)=QLLMM(:,:)

      ENDDO spin
      
      DEALLOCATE(FR,G,FG)

      ENDDO ion
      
      DEALLOCATE(YLM)

      RETURN
      END SUBROUTINE SETQIJB



!******************** SUBROUTINE DUMP_HAM_PEAD *************************
!
! dump a "Hamilton matrix" between the calculated states
!
!***********************************************************************
      SUBROUTINE DUMP_HAM_PEAD( STRING, WDES, CHAM)
      USE wave
      CHARACTER (LEN=*) :: STRING
      TYPE (wavedes)     WDES
      COMPLEX(q)  CHAM(WDES%NB_TOT,WDES%NB_TOT)
      INTEGER N1, N2, NPL2
      INTEGER NB_TOT

      NB_TOT=WDES%NB_TOT

      WRITE(*,*) STRING
      NPL2=MIN(20,NB_TOT)
      DO N1=1,NPL2
         WRITE(*,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
      ENDDO
      WRITE(*,*)
      DO N1=1,NPL2
         WRITE(*,2)N1,(AIMAG( CHAM(N1,N2)),N2=1,NPL2)
      ENDDO
      WRITE(*,*)
  1   FORMAT(1I2,3X,40F9.5)
  2   FORMAT(1I2,3X,40F9.5)

      END SUBROUTINE DUMP_HAM_PEAD


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************
      SUBROUTINE DUMP_HAM_FILE( UNIT, STRING, WDES, CHAM)
      USE wave
      CHARACTER (LEN=*) :: STRING
      TYPE (wavedes)     WDES
      COMPLEX(q)  CHAM(WDES%NB_TOT,WDES%NB_TOT)
      INTEGER UNIT
      INTEGER N1, N2, NPL2
      INTEGER NB_TOT

      NB_TOT=WDES%NB_TOT

      WRITE(UNIT,*) STRING
      NPL2=MIN(10,NB_TOT)
      DO N1=1,NPL2
         WRITE(UNIT,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
      ENDDO
      WRITE(UNIT,*)
      DO N1=1,NPL2
         WRITE(UNIT,2)N1,(AIMAG( CHAM(N1,N2)),N2=1,NPL2)
      ENDDO
      WRITE(UNIT,*)
  1   FORMAT(1I2,3X,40F9.5)
  2   FORMAT(1I2,3X,40E9.1)

      END SUBROUTINE DUMP_HAM_FILE


!***********************************************************************
!******************** PRIVATE QUERY FUNCTIONS **************************
!***********************************************************************


!***********************************************************************
!      
!***********************************************************************
      FUNCTION IS_INSULATING()
      IMPLICIT NONE
      LOGICAL IS_INSULATING
      IS_INSULATING=LINSULATING
      END FUNCTION IS_INSULATING


!***********************************************************************
!
!***********************************************************************
      FUNCTION LSKIP_EDOTP_DURING_ELMIN()
      IMPLICIT NONE
      LOGICAL LSKIP_EDOTP_DURING_ELMIN
      LSKIP_EDOTP_DURING_ELMIN=LSKIP_EDOTP
      END FUNCTION LSKIP_EDOTP_DURING_ELMIN


!***********************************************************************
!
!***********************************************************************
      FUNCTION LNSCF_RESPONSE()
      IMPLICIT NONE
      LOGICAL LNSCF_RESPONSE
      LNSCF_RESPONSE=.NOT.LSKIP_NSCF
      END FUNCTION LNSCF_RESPONSE

      
!***********************************************************************
!
!***********************************************************************
      FUNCTION LSCF_RESPONSE()
      IMPLICIT NONE
      LOGICAL LSCF_RESPONSE
      LSCF_RESPONSE=.NOT.LSKIP_SCF
      END FUNCTION LSCF_RESPONSE

      END MODULE pead
