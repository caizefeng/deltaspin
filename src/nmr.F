#include "symbol.inc"
MODULE morbitalmag
  USE prec
  USE pseudo
  USE mgrid
  USE us
  IMPLICIT none

  ! use orbital magnetization  (periodical+molecular converse approach and direct molecular approach)

  LOGICAL, SAVE :: ORBITALMAG=.FALSE.

  ! use linear response to calculate shielding tensor

  LOGICAL, SAVE :: LCHIMAG=.FALSE.

  ! write current response to B field to file

  LOGICAL, SAVE :: LWRTCUR=.FALSE.

  ! use bloch summations to obtain orbital magnetization (converse approach)

  LOGICAL, SAVE :: LMAGBLOCH=.FALSE.

  ! use gauge transformation for zero moment terms

  LOGICAL, SAVE :: LGAUGE=.TRUE.

  ! use constant B field with sawtooth vector potential ( direct approach)

  LOGICAL, SAVE :: LBFCONST=.FALSE.

  ! use nuclear independent calculation
  ! if LBFCONST=.TRUE. (direct approach) two cases : 
  ! (1) LNICSALL=.TRUE. calculate the nics in all points i.e all the real-space mesh (fine-grid)
  !                      and the output is writed as the charge density file FILE='NICSCAR'   
  ! (2) LNICSALL=.FALSE. read coordinates positions for the nics calculations in FILE='POSNICS'	

  LOGICAL, SAVE :: NUCIND=.FALSE.
  LOGICAL, SAVE :: LNICSALL=.TRUE.
  
  ! coordinates where the magnetic moment is put in the converse approach (if NUCIND=.TRUE.)

  REAL(q), SAVE :: MAGPOS(3)
 
  ! atom which is magnetic (default is 0 no atom)
  ! for NICS (converse approach) this should be set to zero

  INTEGER,SAVE :: MAGATOM=0

  ! atom which the moment is evaluated

  INTEGER,SAVE :: JATOM=0

  ! magnitude of magnetic moment

  REAL(q), SAVE :: MAGDIPOL(3)

  ! magnitude of magnetic moment

  REAL(q), SAVE :: AVECCONST(3)

  ! magnitude of magnetic moment from current density

  REAL(q), SAVE :: MAGDIPOLOUT(3)

  ! magnitude of applied constant magnetic field

  REAL(q), SAVE :: BCONST(3)

  ! magnitude of reciprocal space displacement for finite difference

  REAL(q), SAVE :: DQ

  ! stencil for the computation of chi_bare

  INTEGER, SAVE :: ICHIBARE

  ! reduce symmetry in accordance with the cartesian finite
  ! difference vectors used in the linear response NMR

  LOGICAL, SAVE :: LNMR_SYM_RED

  ! use ZORA approximation in linear response NMR

  LOGICAL, SAVE :: LZORA

  ! reconstruct and use the B-component in the AE one-center terms
  ! of the linear response NMR

  LOGICAL, SAVE :: LBONE

  ! use augmentation correction correction in the plane wave charge density,
  ! i.e. two-centre terms in the induced field, in the  linear response NMR
  ! default: not used

  LOGICAL, SAVE :: LLRAUG=.FALSE.

  INTEGER, EXTERNAL :: MAXL1

  INTEGER, SAVE :: NPOS_NICS
  REAL(q), ALLOCATABLE, SAVE :: TAUNICS(:,:)

  REAL(q) :: ENCUT_AVEC=100000

!**********************************************************************
!
! we define the momentum operator as
!  p = hbar / 2 m_e (-i nabla + |e| A)
! The current operator is thus
!  j(r) = e <psi| (-i [r nabla + nabla r] + |e| A(r)) |psi> ->
!       = - |e| Imag <psi |r nabla| psi> - |e| A(r) <psi|r|psi>
!
! comment on units are in place here
! the magnetic dipole moment is read in in Bohr magnetons
! the current operator is defined as          <phi| nabla | phi>
! the units of the stored vector fields are   eV
! quantities such as
!  hbar < psi| nabla  | psi>   + <psi | e / c A | psi>
! become
!  < psi| nabla  | psi>   + <psi | e /(c hbar) A | psi> ->
!  < psi| nabla  | psi>   + <psi | AVEC | psi> * MOMTOMOM/ MAGMOMTOENERGY
!  < psi| nabla  | psi>   + <psi | M x r / r^3  | psi> * MOMTOMOM
!
!**********************************************************************

! current augmentation restores the multipoles of the 
! difference between the AE and PS current by adding an augmentation current
! at each site such that the multipole development of the PS current
! density is identical to the multipole development of the AE current
! density
! for direct (LGAUGE=.TRUE.) and converse approach (LGAUGE=.FALSE) 
! results are independent of this flag (tested for P4)
! #define current_augmentation

! this flag determines whether we use current augmentation in 
! the Hamiltonian on the plane wave grid
! i.e. evaluate D_ij= \int A(r) j_ij(r)
! #define dij_augmentation

#define norm_conserving


  CONTAINS


!**********************************************************************
!
! read all variables related to the orbital magnetization
! from the INCAR file
!
!**********************************************************************

  SUBROUTINE ORBITALMAG_READER(IU0, IU5, IU6, NIONS)
   
      USE vaspxml
      USE base
      IMPLICIT NONE
      INTEGER IU5, IU0, IU6
      INTEGER :: NIONS
  ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC
      CHARACTER (40)   SZNAM

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      ORBITALMAG=.FALSE.

! read in at which atom we place a magnetic moment
      MAGATOM=0
      CALL RDATAB(LOPEN,INCAR,IU5,'MAGATOM','=','#',';','I', &
     &            MAGATOM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''MAGATOM'' from file INCAR.'
         MAGATOM=0
      ENDIF
      MAGATOM=MAX(0,MIN(NIONS,MAGATOM))
      CALL XML_INCAR('MAGATOM','I',MAGATOM,RDUM,CDUM,LDUM,CHARAC,N)

! read in at which atom we place a magnetic moment
      JATOM=MAGATOM
      CALL RDATAB(LOPEN,INCAR,IU5,'JATOM','=','#',';','I', &
     &            JATOM,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''JATOM'' from file INCAR.'
         JATOM=0
      ENDIF
      JATOM=MAX(0,MIN(NIONS,JATOM))
      CALL XML_INCAR('JATOM','I',JATOM,RDUM,CDUM,LDUM,CHARAC,N)
 
! magnitude of magnetic moment
      MAGDIPOL=0
      CALL RDATAB(LOPEN,INCAR,IU5,'MAGDIPOL','=','#',';','F', &
     &            IDUM,MAGDIPOL,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''MAGDIPOL'' from file INCAR.'
         MAGDIPOL=0
         MAGATOM=0
      ENDIF
      CALL XML_INCAR_V('MAGDIPOL','F',IDUM,MAGDIPOL,CDUM,LDUM,CHARAC,N)

      IF (MAGATOM>0) ORBITALMAG=.TRUE.
! magnitude of magnetic moment
      AVECCONST=0
      CALL RDATAB(LOPEN,INCAR,IU5,'AVECCONST','=','#',';','F', &
     &            IDUM,AVECCONST,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''AVECCONST'' from file INCAR.'
         AVECCONST=0
      ENDIF
      CALL XML_INCAR_V('AVECCONST','F',IDUM,MAGDIPOL,CDUM,LDUM,CHARAC,N)

! read in whether ORBITALMAG is set or not
      CALL RDATAB(LOPEN,INCAR,IU5,'ORBITALMAG','=','#',';','L', &
     &            IDUM,RDUM,CDUM,ORBITALMAG,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ORBITALMAG'' from file INCAR.'
         ORBITALMAG=.FALSE.
      ENDIF

      CALL XML_INCAR('ORBITALMAG','L',IDUM,RDUM,CDUM,ORBITALMAG,CHARAC,N)

! read in whether LCHIMAG is set or not
      CALL RDATAB(LOPEN,INCAR,IU5,'LCHIMAG','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LCHIMAG,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LCHIMAG'' from file INCAR.'
         LCHIMAG=.FALSE.
      ENDIF

      CALL XML_INCAR('LCHIMAG','L',IDUM,RDUM,CDUM,LCHIMAG,CHARAC,N)

! magnitude of dq
      DQ=1.E-3_q
      CALL RDATAB(LOPEN,INCAR,IU5,'DQ','=','#',';','F', &
     &            IDUM,DQ,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''DQ'' from file INCAR.'
         DQ=1.E-3_q
      ENDIF

      CALL XML_INCAR('DQ','F',IDUM,DQ,CDUM,LDUM,CHARAC,N)

! stencil used to compute chi_bare
      ICHIBARE=1
      CALL RDATAB(LOPEN,INCAR,IU5,'ICHIBARE','=','#',';','I', &
     &            ICHIBARE,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ICHIBARE'' from file INCAR.'
         ICHIBARE=1
      ENDIF
      IF (ICHIBARE<1.OR.ICHIBARE>3) ICHIBARE=1

      CALL XML_INCAR('ICHIBARE','I',ICHIBARE,RDUM,CDUM,LDUM,CHARAC,N)

! read in whether LWRTCUR is set or not
      LWRTCUR=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LWRTCUR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LWRTCUR,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LWRTCUR'' from file INCAR.'
         LWRTCUR=.FALSE.
      ENDIF

      CALL XML_INCAR('LWRTCUR','L',IDUM,RDUM,CDUM,LWRTCUR,CHARAC,N)

! read in whether LNMR_SYM_RED is set or not
      LNMR_SYM_RED=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LNMR_SYM_RED','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LNMR_SYM_RED,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LNMR_SYM_RED'' from file INCAR.'
         LNMR_SYM_RED=.FALSE.
      ENDIF

      CALL XML_INCAR('LNMR_SYM_RED','L',IDUM,RDUM,CDUM,LNMR_SYM_RED,CHARAC,N)

! read in whether LZORA is set or not
      LZORA=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LZORA','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LZORA,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LZORA'' from file INCAR.'
         LZORA=.FALSE.
      ENDIF

      CALL XML_INCAR('LZORA','L',IDUM,RDUM,CDUM,LZORA,CHARAC,N)

! read in whether LBONE is set or not
      LBONE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LBONE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LBONE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LBONE'' from file INCAR.'
         LBONE=.FALSE.
      ENDIF

      CALL XML_INCAR('LBONE','L',IDUM,RDUM,CDUM,LBONE,CHARAC,N)
! let LBONE take precedence over LZORA
      IF (LBONE.AND.LZORA) THEN
         LZORA=.FALSE.
! and warn the user ...
         CALL VTUTOR('W','LBONEANDLZORA',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
         CALL VTUTOR('W','LBONEANDLZORA',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
      ENDIF

! read in whether LMAGBLOCH is set or not
      CALL RDATAB(LOPEN,INCAR,IU5,'LMAGBLOCH','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LMAGBLOCH,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LMAGBLOCH'' from file INCAR.'
         LMAGBLOCH=.FALSE.
      ENDIF

      CALL XML_INCAR('LBLOCHMAG','L',IDUM,RDUM,CDUM,LMAGBLOCH,CHARAC,N)

! read in whether LGAUGE is set or not
      LGAUGE=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LGAUGE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LGAUGE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LGAUGE'' from file INCAR.'
         LGAUGE=.FALSE.
      ENDIF

      CALL XML_INCAR('LGAUGE','L',IDUM,RDUM,CDUM,LGAUGE,CHARAC,N)

! read in whether LBFCONST is set or not
      LBFCONST=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LBFCONST','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LBFCONST,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LBFCONST'' from file INCAR.'
         LBFCONST=.FALSE.
      ENDIF

      CALL XML_INCAR('LBFCONST','L',IDUM,RDUM,CDUM,LBFCONST,CHARAC,N)

! magnitude of magnetic moment
      BCONST=0
      CALL RDATAB(LOPEN,INCAR,IU5,'BCONST','=','#',';','F', &
     &            IDUM,BCONST,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''BCONST'' from file INCAR.'
         BCONST=0
      ENDIF

      CALL XML_INCAR_V('BCONST','F',IDUM,BCONST,CDUM,LDUM,CHARAC,N)

! nuclear independent calculation
        NUCIND=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IU5,'NUCIND','=','#',';','L', &
     &            IDUM,RDUM,CDUM,NUCIND,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''NUCIND'' from file INCAR.'
         NUCIND=.FALSE.
      ENDIF

      CALL XML_INCAR('NUCIND','L',IDUM,RDUM,CDUM,NUCIND,CHARAC,N)

! read in whether LNICSALL is set or not
      LNICSALL=.TRUE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LNICSALL','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LNICSALL,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LNICSALL'' from file INCAR.'
         LNICSALL=.FALSE.
      ENDIF

      CALL XML_INCAR('LNICSALL','L',IDUM,RDUM,CDUM,LNICSALL,CHARAC,N)

!read coordinates of NICS calculation (converse approach)
 
      MAGPOS=0
      CALL RDATAB(LOPEN,INCAR,IU5,'MAGPOS','=','#',';','F', &
     &            IDUM,MAGPOS,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''MAGPOS'' from file INCAR.'
         BCONST=0
      ENDIF

      CALL XML_INCAR_V('MAGPOS','F',IDUM,BCONST,CDUM,LDUM,CHARAC,N)

! read in whether LLRAUG is set or not
      LLRAUG=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LLRAUG','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LLRAUG,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LLRAUG'' from file INCAR.'
         LLRAUG=.FALSE.
      ENDIF

      CALL XML_INCAR('LLRAUG','L',IDUM,RDUM,CDUM,LLRAUG,CHARAC,N)


      CLOSE(IU5)

      NPOS_NICS=0
      IF (NUCIND.AND.(.NOT.LNICSALL).AND.(.NOT.LMAGBLOCH)) THEN
        OPEN(UNIT=10,FILE='POSNICS',STATUS="UNKNOWN")
        READ(10,*) NPOS_NICS
        ALLOCATE(TAUNICS(NPOS_NICS,3))
        DO N=1,NPOS_NICS
           READ(10,*) TAUNICS(N,1),TAUNICS(N,2),TAUNICS(N,3)
        ENDDO
        CLOSE(10)
      ENDIF

    END SUBROUTINE ORBITALMAG_READER

!**********************************************************************
!
! write the Fock and exchange correlation related
! parameters to the OUTCAR file
!
!**********************************************************************

    SUBROUTINE WRITE_ORBITALMAG(IU6)
      IMPLICIT NONE
      INTEGER IU6

      IF (IU6>=0 .AND. (ORBITALMAG.OR. LCHIMAG)) THEN
         IF (IU6>=0 .AND. LBFCONST) THEN
            IF (IU6>=0 .AND. NUCIND) THEN 
               WRITE(IU6,20 ) ORBITALMAG, LCHIMAG, DQ, LGAUGE, NUCIND, LNICSALL, AVECCONST, LBFCONST,BCONST
            ELSE
              WRITE(IU6,30 ) ORBITALMAG, LCHIMAG, DQ, LGAUGE, AVECCONST, LBFCONST,BCONST
            ENDIF
         ELSE
           WRITE(IU6,10 ) ORBITALMAG, LMAGBLOCH, LCHIMAG, DQ, LGAUGE, NUCIND, MAGPOS,MAGATOM, JATOM, MAGDIPOL 
         ENDIF 
      ENDIF
      IF (IU6>=0 .AND. .NOT. ORBITALMAG) THEN
         WRITE(IU6,100) ORBITALMAG, LCHIMAG, DQ, LLRAUG
      ENDIF

     
  10  FORMAT(' Orbital magnetization related:'  / &
             '   ORBITALMAG=',L6,   '  switch on orbital magnetization'/ &
             '   LMAGBLOCH =',L6,   '  use Bloch summation to obtain orbital mag.'/ &
             '   LCHIMAG   =',L6,   '  perturbation theory with respect to B field'/ &
             '   DQ        =',F10.6,'  dq finite difference perturbation B field'/ &
             '   LGAUGE    =',L6,   '  gauge transformation for 0 moment terms'/ &
             '   NUCIND    =',L6,   '  nuclear independent calculation        '/ &
             '   MAGPOS    =',3F10.4'  position of magnetic moment in nuclear independent calculation'/ &
             '   MAGATOM   =',I6   ,'  atom on which magnetic dipole is placed' / &
             '   JATOM     =',I6   ,'  atom on which magnetic dipole is evaluated' / &
             '   MAGDIPOL  =',3F10.4,' magnetic dipole'/ )

  20  FORMAT(' Orbital magnetization related:'  / &
             '   ORBITALMAG=',L6,   '  switch on orbital magnetization'/ &
             '   LCHIMAG   =',L6,   '  perturbation theory with respect to B field'/ &
             '   DQ        =',F10.6,'  dq finite difference perturbation B field'/ &
             '   LGAUGE    =',L6,   '  gauge transformation for 0 moment terms'/ &
             '   NUCIND    =',L6,   '  nuclear independent calculation (direct) '/ &
             '   LNICSALL  =',L6,   '  T = all points, F= read positions in POSNICS file'/&
             '   AVECCONST =',3F10.4,' constant vector potential (G=0)'/ &
             '   LBFCONST  =',L6,   '  constant B field applied'/ &
             '   BCONST    =',3F10.4,' applied magnetic field'/)


  30  FORMAT(' Orbital magnetization related:'  / &
             '   ORBITALMAG=',L6,   '  switch on orbital magnetization'/ &
             '   LCHIMAG   =',L6,   '  perturbation theory with respect to B field'/ &
             '   DQ        =',F10.6,'  dq finite difference perturbation B field'/ &
             '   LGAUGE    =',L6,   '  gauge transformation for 0 moment terms'/&
             '   AVECCONST =',3F10.4,' constant vector potential (G=0)'/ &
             '   LBFCONST  =',L6,   '  constant B field applied'/ &
             '   BCONST    =',3F10.4,' applied magnetic field'/)



  100 FORMAT(' Orbital magnetization related:'  / &
             '   ORBITALMAG=',L6,   '  switch on orbital magnetization'/ &
             '   LCHIMAG   =',L6,   '  perturbation theory with respect to B field'/ &
             '   DQ        =',F10.6,'  dq finite difference perturbation B field'/ &
             '   LLRAUG    =',L6,   '  two centre corrections for induced B field'/ &
             )

       
    END SUBROUTINE WRITE_ORBITALMAG


    SUBROUTINE XML_WRITE_ORBITALMAG
      USE vaspxml
      IMPLICIT NONE
! local
      INTEGER IDUM, N
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LDUM
      CHARACTER (1) :: CHARAC

      CALL XML_TAG("separator","orbital magnetization")
      N=1
      CALL XML_INCAR('NUCIND','L',IDUM,RDUM,CDUM,NUCIND,CHARAC,N)
      CALL XML_INCAR_V('MAGPOS','F',IDUM,MAGPOS,CDUM,LDUM,CHARAC,3)
      CALL XML_INCAR('LNICSALL','L',IDUM,RDUM,CDUM,LNICSALL,CHARAC,N)	
      CALL XML_INCAR('ORBITALMAG','L',IDUM,RDUM,CDUM,ORBITALMAG,CHARAC,N)
      CALL XML_INCAR('LMAGBLOCH','L',IDUM,RDUM,CDUM,LMAGBLOCH,CHARAC,N)
      CALL XML_INCAR('LCHIMAG','L',IDUM,RDUM,CDUM,LCHIMAG,CHARAC,N)
      CALL XML_INCAR('LGAUGE','L',IDUM,RDUM,CDUM,LGAUGE,CHARAC,N)
      CALL XML_INCAR('MAGATOM','I',MAGATOM,RDUM,CDUM,LDUM,CHARAC,N)
      CALL XML_INCAR_V('MAGDIPOL','F',IDUM,MAGDIPOL,CDUM,LDUM,CHARAC,3)
      CALL XML_INCAR_V('AVECCONST','F',IDUM,AVECCONST,CDUM,LDUM,CHARAC,3)
      CALL XML_CLOSE_TAG

    END SUBROUTINE XML_WRITE_ORBITALMAG


    SUBROUTINE WRITE_ORBITALMAGOUT(IU6)
      IMPLICIT NONE
      INTEGER IU6

      IF (IU6>=0 .AND. ORBITALMAG) THEN
         WRITE(IU6,10 ) MAGDIPOLOUT
      ENDIF

     
  10  FORMAT(' Output orbital magnetization:'  / &
             '   MAGDIPOLOUT  =',3F20.16,' magnetic dipole'/)

    END SUBROUTINE WRITE_ORBITALMAGOUT

    SUBROUTINE XML_WRITE_ORBITALMAGOUT
      USE vaspxml
      IMPLICIT NONE
! local
      INTEGER IDUM, N
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LDUM
      CHARACTER (1) :: CHARAC

      CALL XML_TAG("separator","orbital magnetization")
      CALL XML_INCAR_V('MAGDIPOLOUT','F',IDUM,MAGDIPOLOUT,CDUM,LDUM,CHARAC,3)
      CALL XML_CLOSE_TAG

    END SUBROUTINE XML_WRITE_ORBITALMAGOUT

!***************************** SUBROUTINE ALLOCATE_AVEC ****************************
!
! allocate and deallocate orbital magnetization dependent quantities
!
!***********************************************************************************

    SUBROUTINE ALLOCATE_AVEC(AVEC, AVTOT, GRID, GRIDC )
      USE mgrid
      USE ini
      TYPE (grid_3d)     GRID, GRIDC
      COMPLEX(q),POINTER :: AVTOT(:,:)    ! local vector magnetization potential
      RGRID  ,POINTER    :: AVEC(:,:)     ! soft part of vector magnetization potential
     
      IF (ORBITALMAG) THEN
         ALLOCATE(AVEC(DIMREAL(GRID%MPLWV),3),AVTOT(GRIDC%MPLWV,3))
         CALL REGISTER_ALLOCATE(16._q*(SIZE(AVTOT)+SIZE(AVEC)), "grid")

      ELSE
         NULLIFY(AVTOT, AVEC)
      ENDIF

    END SUBROUTINE ALLOCATE_AVEC


    SUBROUTINE DEALLOCATE_AVEC(AVEC, AVTOT)
      USE mgrid
      USE ini
      COMPLEX(q),POINTER :: AVTOT(:,:)    ! local vector magnetization potential
      RGRID  ,POINTER    :: AVEC(:,:)     ! soft part of vector magnetization potential
     
      IF (ORBITALMAG) THEN
         CALL DEREGISTER_ALLOCATE(16._q*(SIZE(AVTOT)+SIZE(AVEC)), "grid")
         DEALLOCATE(AVEC,AVTOT)
      ENDIF
      NULLIFY(AVTOT, AVEC)

    END SUBROUTINE DEALLOCATE_AVEC


!***************************** SUBROUTINE CALC_JPAW ********************************
!
! Note: vasp inverts the indexing internally for reasons of speed hence
!
! J(i,j,LM,alpha) = int dr Y_LM(r) r^L
!      <phi^AE_j |nabla |r><r| + |r><r| nabla |phi^AE_i> -
!      <phi^PS_j |nabla |r><r| + |r><r| nabla |phi^PS_i>
!
!
! J(i,j,LM,alpha) =  int dr Y_LM(r) r^L
!       (  ( phi^AE_j(r) r nabla phi^AE_i(r) -  phi^AE_i(r) |r nabla |phi^AE_j(r))
!         -( phi^PS_j(r) r nabla phi^PS_i(r) -  phi^PS_i(r) |r nabla |phi^PS_j(r)))
!
! comment second terms stem from <phi^AE_j |nabla r Y_lm(r) r^L |phi^AE_i> =
!         int dr phi^AE_j(r) nabla r Y_lm(r) r^L phi^AE_i(r)
! partial integration
!
! to get the momentum operator J has to be multiplied by -i hbar
!
!  P= -i hbar nabla
!
!
!***********************************************************************************

    SUBROUTINE CALC_JPAW_HAMIL(T_INFO, P)
      USE poscar
      USE pseudo
      TYPE (type_info):: T_INFO        ! type information
      TYPE (potcar), TARGET  :: P(:)
      INTEGER NT

      CALL CALC_JPAW(T_INFO, P)
!     WRITE(*,*) ' forcing JPAW in Hamiltonian to 0, L=0'

      DO NT=1,SIZE(P)
         IF ( ASSOCIATED(P(NT)%JPAW)) THEN
!            P(NT)%JPAW=0
!            P(NT)%JPAW(:,:,1,:)=0
!            P(NT)%JPAW(:,:,2:,:)=0
!            P(NT)%JPAW(:,:,5:,:)=0
         ENDIF
      ENDDO
    END SUBROUTINE CALC_JPAW_HAMIL

    SUBROUTINE CALC_JPAW(T_INFO, P)
      USE poscar
      USE pseudo
      USE constant
      USE asa
      USE radial                                                                ! ?
      USE wave
      USE paw
     IMPLICIT NONE
     
      TYPE (type_info):: T_INFO        ! type information
      TYPE (potcar), TARGET  :: P(:)
! local
      TYPE (potcar), POINTER :: PP
      INTEGER CHANNELS          ! number of channels
     !local variables
      INTEGER NMAX,I,LMAX,NT,NT_MAGATOM
      INTEGER I0,I1
      INTEGER L0,L1,L2,L3
      INTEGER M0,M1,M2,M3
      INTEGER LM0,LM1,LM2,LM3,LM2MAX
      INTEGER ISTART1,IEND1,LMIND1,IC1
      INTEGER LMAXNABLA,K
      REAL(q) :: CGOR
      REAL(q) :: SUM
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:)
! additional stuff required for diamagnetic augmentation currents
      INTEGER :: LMRHO, LRHO, MRHO
      OVERLAP, ALLOCATABLE :: CTMP(:,:)
      REAL(q), ALLOCATABLE :: YLM_MxR_YLM(:,:,:)
      REAL(q) :: DLM(256)
! allocate
!     IF (.NOT. ORBITALMAG) RETURN
!test to get the NABIJ
!     CALL SET_NABIJ_AUG(P(1),SIZE(P))

      IF (MAGATOM>0) THEN
         NT_MAGATOM=T_INFO%ITYP(MAGATOM)
         PP=> P(NT_MAGATOM)
         LMAX     =MAXL1(PP)
         LMAXNABLA=LMAX+1       ! nabla Y_lm has components up to LMAX+1
      ELSE
         NT_MAGATOM=0
      ENDIF

!-----------------------------------------------------------------------
      DO NT=1,SIZE(P)
!-----------------------------------------------------------------------
      IF ( .NOT. ASSOCIATED(P(NT)%QPAW )) CYCLE
      IF ( ASSOCIATED(P(NT)%JPAW)) CYCLE

      PP=> P(NT)
      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX))

      CHANNELS = PP%LMAX
      LMAX     =MAXL1(PP)
      LMAXNABLA=LMAX+1   ! nabla Y_lm has components up to LMAX+1

      ALLOCATE(PP%JPAW(PP%LMMAX,PP%LMMAX,(LMAX+LMAXNABLA+1)*(LMAX+LMAXNABLA+1),3))


! ALLOCATE and setup nabla array
      ALLOCATE(YLM_NABLA_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))
      ALLOCATE(YLM_X_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))
      ALLOCATE(YLM_MxR_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q

      CALL SETYLM_NABLA_YLM(LMAXNABLA,YLM_NABLA_YLM,YLM_X_YLM)
      ! store < Y_lm | M x r | Y_l'm' >
      YLM_MxR_YLM(:,:,1)=MAGDIPOL(2)*YLM_X_YLM(:,:,3)-MAGDIPOL(3)*YLM_X_YLM(:,:,2)
      YLM_MxR_YLM(:,:,2)=MAGDIPOL(3)*YLM_X_YLM(:,:,1)-MAGDIPOL(1)*YLM_X_YLM(:,:,3)
      YLM_MxR_YLM(:,:,3)=MAGDIPOL(1)*YLM_X_YLM(:,:,2)-MAGDIPOL(2)*YLM_X_YLM(:,:,1)

      PP%JPAW=0._q   
!-----------------------------------------------------------------------
! calculate paramagnetic augmentation current
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------
      LM2MAX=0

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
!        WTMP(:)=1/PP%R%R         
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

!test
!        DWPS=0

         DO M1=1,2*L1+1

            LM3=1
            DO L3=0,LMAXNABLA

            CALL YLM3LOOKUP(L0,L3,LMIND1)

            DO M0=1,2*L0+1
            DO M3=1,2*L3+1
               LMIND1 = LMIND1+1
 
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               ! this loop goes over all possible l'', m'' being non zero in CG coefficien
               DO IC1=ISTART1,IEND1-1
                  ! JS(IC1)) is a compound index for l'' and m''
                  ! JL(IC1)) is the l quantum number L (zero based)
                  ! see asa.F line 286 how compound index is defined,
                  ! l''  m''    JS
                  ! 0    0      1
                  ! 1    -1     2
                  ! 1    0      3
                  ! 1    1      4
                  ! 2    -2     5
                  ! coefficient C(l'',m'',l,m,l'm') is stored in YLM3(IC1)
                  ! any statments at this place will run over all  (lm, l'm', l'',m'')
                  ! any statments at this place will run over all  (l0m0, l2m2, l3,m3)

                  CGOR=YLM3(IC1)
                  LM2 =JS(IC1)
                  L2  =JL(IC1)

                  DO I=1,3
                  IF (ABS(YLM_X_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                    TMP=0._q
                    SUM=0._q
! Y_lm3 nabla Y_lm1 = YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K)
!
                     DO K=1,PP%R%NMAX
                        TMP(K) = CGOR * PP%R%R(K)**L2 * (                                   &
      &     PP%WAE(K,I0) * ( DWAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
      &                          + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WAE(K,I1) )  &
      &   - PP%WPS(K,I0) * ( DWPS(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
      &                          + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WPS(K,I1) )  &
      &                  )
                     ENDDO
                     CALL SIMPI(PP%R,TMP,SUM)
                     IF (LM2 <=0 .OR. LM2 > SIZE(PP%JPAW,3)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 3 into JPAR exceeds bounds'
                        STOP
                     ENDIF
                     IF (LM0+M0-1 <=0 .OR. LM0+M0-1 > SIZE(PP%JPAW,1)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 1 into JPAR exceeds bounds'
                        STOP
                     ENDIF
                     IF (LM1+M1-1 <=0 .OR. LM1+M1-1 > SIZE(PP%JPAW,2)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 2 into JPAR exceeds bounds'
                        STOP
                     ENDIF

                     IF (LM3+M3-1 <=0 .OR. LM3+M3-1 > SIZE(YLM_X_YLM,1)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 1 into YLM_X_YLM exceeds bounds'
                        STOP
                     ENDIF
                     IF (L1*L1+M1 <=0 .OR. L1*L1+M1 > SIZE(YLM_X_YLM,2)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 2 into YLM_X_YLM exceeds bounds'
                        STOP
                     ENDIF
                     IF (LM3+M3-1 <=0 .OR. LM3+M3-1 > SIZE(YLM_NABLA_YLM,1)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 1 into YLM_NABLA_YLM exceeds bounds'
                        STOP
                     ENDIF
                     IF (L1*L1+M1 <=0 .OR. L1*L1+M1 > SIZE(YLM_NABLA_YLM,2)) THEN
                        WRITE(0,*) 'internal error in CALC_JPAW: index 2 into YLM_NABLA_YLM exceeds bounds'
                        STOP
                     ENDIF
 
              !  J(i,j,LM,alpha) =  phi^AE_j(r) r nabla phi^AE_i(r) -phi^PS_j(r) r nabla phi^PS_i(r)
                     PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) = PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) + SUM
                     LM2MAX=MAX(LM2, LM2MAX)
                  ENDIF
                  ENDDO
               ENDDO  ! IC1
            ENDDO ! M3
            ENDDO ! M0

            LM3=LM3+2*L3+1
            ENDDO ! L3

         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0
      ! make JPAW antisymmetric
      DO I0=1,PP%LMMAX
         DO I1=I0,PP%LMMAX
            PP%JPAW(I0,I1,:,:)= (PP%JPAW(I0,I1,:,:)-PP%JPAW(I1,I0,:,:))/2
            PP%JPAW(I1,I0,:,:)= -PP%JPAW(I0,I1,:,:)
         ENDDO
      ENDDO
      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM, YLM_MxR_YLM)
      DEALLOCATE(WTMP,DWAE,DWPS,TMP)
      ENDDO ! NT

    END SUBROUTINE CALC_JPAW


!***************************** SUBROUTINE VECTORPOT ********************************
!
! calculate the vector potential in reciprocal space on both the
! fine (GRIDC) as well as the plane wave grid (GRID)
!
! AVTOT and AVEC are set up in real space
!
!***********************************************************************************
   
    SUBROUTINE VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C, COMM_INTER,  LATT_CUR, POSION, AVEC, AVTOT )
      USE lattice
      USE constant
      USE poscar

      TYPE (grid_3d)     GRID,GRIDC,GRID_SOFT
      TYPE (transit)     SOFT_TO_C
      TYPE (communic)    COMM_INTER
      TYPE (latt)        LATT_CUR

      RGRID, POINTER     ::  AVEC(:,:)
      COMPLEX(q),POINTER ::  AVTOT(:,:)
      REAL(q)   ::  POSION(:,:)
    ! local
      INTEGER   ::  N, NC, N1, N2, N3, NI, I
      INTEGER   ::  NX, NY, NZ, NG
      REAL(q)   ::  G(3), TMP(3), G2, ENERGY, XX, CUTOFF
      REAL(q)   ::  TMP2(3),X,Y,Z
      COMPLEX(q)::  CPHASE
      COMPLEX(q), ALLOCATABLE::  CWORK1(:)
! 4 grid points around the step function are usually a good choice
      REAL(q), PARAMETER :: WIDTH=4

      IF (.NOT.ORBITALMAG) RETURN
      IF (.NOT. ASSOCIATED(AVEC) .OR. .NOT. ASSOCIATED(AVTOT)) THEN
         WRITE(0,*) 'internal error in VECTORPOT: AVEC or AVTOT not associated'
         STOP
      ENDIF

      bfconst: IF (LBFCONST) THEN
!=======================================================================
! A = B x r / 2
! Because of the jump, the vectorpotential is set up in real space
!=======================================================================

        AVTOT=0

        NG=0
        DO NC=1,GRIDC%RL%NCOL
           N2= GRIDC%RL%I2(NC)
           N3= GRIDC%RL%I3(NC)

           DO N1=1,GRIDC%RL%NROW
              NG=NG+1
              IF (GRIDC%RL%NFAST==3) THEN
                 NX=N2
                 NY=N3
                 NZ=N1
              ELSE
                 NX=N1
                 NY=N2
                 NZ=N3
              ENDIF
              TMP(1)=MOD((REAL(NX,q)-1)/GRIDC%NGX+10.5_q,1._q)-0.5_q
              TMP(2)=MOD((REAL(NY,q)-1)/GRIDC%NGY+10.5_q,1._q)-0.5_q
              TMP(3)=MOD((REAL(NZ,q)-1)/GRIDC%NGZ+10.5_q,1._q)-0.5_q

              CUTOFF=1.0

              XX=ABS(NX-1-GRIDC%NGX/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF
              XX=ABS(NY-1-GRIDC%NGY/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF
              XX=ABS(NZ-1-GRIDC%NGZ/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF

              CALL DIRKAR(1, TMP, LATT_CUR%A)
 
              CALL EXPRO(TMP2(1), BCONST(1), TMP(1))

              TMP2(:)=TMP2(:)/2._q*MAGMOMTOENERGY*CUTOFF
 
              AVTOT(NG,1)=TMP2(1)
              AVTOT(NG,2)=TMP2(2)
              AVTOT(NG,3)=TMP2(3)
           ENDDO
        ENDDO


! goto reciprocal space
        DO I=1,3
          CALL RL_ADD(AVTOT(1,I),1._q/GRIDC%NPLWV,AVTOT(1,I),0.0_q,AVTOT(1,I),GRIDC)
          CALL FFT3D(AVTOT(1,I),GRIDC,-1)
        ENDDO

      ELSE bfconst

!=======================================================================
! We only get here if a constant field is NOT applied, i.e. the converse approach
!=======================================================================

        AVTOT=0.0_q

        NI=MAGATOM

!fmv
        IF (NUCIND) THEN
        X=POSION(1,NI)
        Y=POSION(2,NI)
        Z=POSION(3,NI)
        POSION(1,NI)=MAGPOS(1)
        POSION(2,NI)=MAGPOS(2)
        POSION(3,NI)=MAGPOS(3)
        ENDIF
!fmv

        N=0
        col: DO NC=1,GRIDC%RC%NCOL
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)
        row: DO N1=1,GRIDC%RC%NROW
          N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
          G(1)= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
          G(2)= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
          G(3)= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

          G2=G(1)**2+G(2)**2+G(3)**2
 
          CPHASE=EXP(-CITPI*(POSION(3,NI)*GRIDC%LPCTZ(N3)+POSION(2,NI)*GRIDC%LPCTY(N2)+POSION(1,NI)*GRIDC%LPCTX(N1)))
!=======================================================================
! Equ. (5) of .............
!  4 pi i / Omega m_s x G / G^2 exp(-i G . r_s)
!=======================================================================
          ENERGY=HSQDTM*G2

          IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &           (GRIDC%LPCTZ(N3)/=0))) THEN
             ! copy EXPRO for performance reasons from lattice.F
             CALL EXPRO(TMP, MAGDIPOL, G)
             ! convert from magnetic moment (supplied in multiples of Bohr
             ! magneton) to energy
             ! factor two from A p = p A   in transversal gauge
             AVTOT(N,:)=-TMP/MAX(G2,1E-5_q)*CPHASE*2.0_q*CITPI/LATT_CUR%OMEGA*MAGMOMTOENERGY
             IF (ENERGY>ENCUT_AVEC) AVTOT(N,:)=0
          ELSE
             AVTOT(N,:)=AVECCONST
          ENDIF
        ENDDO row
        ENDDO col


        !fmv
        IF (NUCIND) THEN
        POSION(1,NI)=X
        POSION(2,NI)=Y
        POSION(3,NI)=Z
        ENDIF
        !fmv



      ENDIF bfconst
     
      ALLOCATE(CWORK1(MAX(GRIDC%MPLWV,GRID_SOFT%MPLWV)))
      DO I=1,3
         CALL SETUNB(AVTOT(1,I),GRIDC)
         CALL CP_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,AVTOT(1,I),CWORK1)
         CALL FFT3D(CWORK1,GRID_SOFT, 1)
         CALL RL_ADD(CWORK1,1.0_q,CWORK1,0.0_q,AVEC(1,I),GRID_SOFT)

    !  final result is only correct for first in-band-group
    ! (i.e. proc with nodeid 1 in COMM_INTER)
    !  copy to other in-band-groups using COMM_INTER
    ! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)
#ifdef realmode
         CALLMPI( M_bcast_d(COMM_INTER, AVEC(1,I), GRID%RL%NP))
#else
         CALLMPI( M_bcast_z(COMM_INTER, AVEC(1,I), GRID%RL%NP))
#endif
         CALL FFT3D(AVTOT(1,I),GRIDC,1)
      ENDDO
     

      DO N=1,GRID%RL%NP
         TMP=AVEC(N,:)
         ! convert from cartesian to coordinates into direct lattice units
         CALL KARDIR(1, TMP, LATT_CUR%B)
         AVEC(N,:)=TMP
      ENDDO

      DEALLOCATE(CWORK1)

    END SUBROUTINE VECTORPOT


!***************************** SUBROUTINE CURRENT   ********************************
!
! routine to evaluate current density
!
!***********************************************************************************

    SUBROUTINE CURRENT( W, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, P, LATT_CUR, &
          AVEC, AVTOT, CHTOT, NONLR_S, NONL_S, &
          T_INFO, LMDIM, CRHODE, CDIJ, CQIJ,  IRDMAX, IU6, IU0 , NWRITE)
      USE wave_high
      USE lattice
      USE mgrid
      USE constant
      USE poscar
      USE us
      USE pseudo
      USE paw
      USE nonl_high
      USE hamil

      TYPE (wavespin)    W             ! wavefunction
      TYPE (grid_3d)     GRID_SOFT     ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d)     GRIDC         ! full, fine grid
      TYPE (grid_3d)     GRIDUS        ! doubled grid
      TYPE (transit)     SOFT_TO_C     !
      TYPE (transit)     C_TO_US       !
      TYPE (latt)        LATT_CUR      ! lattice
      TYPE (potcar), TARGET :: P(:)          ! pseudopotential information
      TYPE (wavefun1)    W1            ! wavefunction array
      TYPE (wavefun1)    W0            ! wavefunction array
      RGRID, POINTER  :: AVEC(:,:)     ! vector potential on course grid
      COMPLEX(q),POINTER ::  AVTOT(:,:)! vector potential on fine grid
      COMPLEX(q) ::       CHTOT(:,:)   ! charge density on fine grid
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct)  NONL_S
      TYPE (type_info):: T_INFO        ! type information
      INTEGER         :: LMDIM         !
      OVERLAP         :: CRHODE(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP         :: CDIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP         :: CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      INTEGER         :: IRDMAX        ! required allocation
      INTEGER         :: IU6, IU0, NWRITE

    ! local
      INTEGER ISP, NK, N, I, NI, NIP, M, NT
      INTEGER NG, NC, N1, N2, N3, NX, NY, NZ, IRDMAA
      INTEGER ITMP
      INTEGER IDIR
      TYPE (wavedes1)    WDES1
      TYPE (potcar), POINTER :: PP
      OVERLAP CDUMMY(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)

    ! Gilles: allocate instead of automatic arrays
      COMPLEX(q), ALLOCATABLE :: CWORK1(:)
      COMPLEX(q), ALLOCATABLE :: CWORK2(:)
      RGRID, ALLOCATABLE      :: JVEC_TMP(:,:)
      COMPLEX(q), ALLOCATABLE :: JVEC(:)
      RGRID, ALLOCATABLE      :: CHDEN_TMP(:)
      COMPLEX(q), ALLOCATABLE :: JTOT(:,:), DIV_J(:)
      REAL(q)    :: WEIGHT
      REAL(q)    :: TMP(3),TMP2(3),JTMP(3),POS(3),TMP3(3)
      REAL(q)    :: TMP1(3)
      REAL(q)    :: MU(3)
      REAL(q)    :: MU_PARA(3), MU_DIA(3), MU_PARA_ONE_CTR(3), MU_DIA_ONE_CTR(3),MU_LQ_ONE_CTR(3),MU_NL(3)
      REAL(q)    :: DISPL(3,T_INFO%NIONS), DIV_ONE_CENTER(0:3,T_INFO%NIONS)
      REAL(q)    :: B_OUT_PARA_ONE_CENTR(3,T_INFO%NIONS), B_OUT_DIA_ONE_CENTR(3,T_INFO%NIONS)
      REAL(q)    :: B_OUT_PARA(3,T_INFO%NIONS), B_OUT_DIA(3,T_INFO%NIONS)
      REAL(q)    :: B_NICS_OUT_PARA(3,T_INFO%NIONS), B_NICS_OUT_DIA(3,T_INFO%NIONS)
      INTEGER :: L,LP
    ! variables required for calculation of augmentation current from non-local part
      TYPE (nonlr_struct),TARGET :: NONLR_D! projector times r (only CRREXP distinct)
      TYPE (nonl_struct)  NONL_D           ! k derivative of projector in real space
!gil  TYPE (wavefun1)    W1                ! wavefunction
      COMPLEX(q)    :: C_NONL(T_INFO%NIONS,3)   !Gilles: think this should be complex
      COMPLEX(q)    :: C_NONL_GLOB(T_INFO%NIONS,3)  !Gilles
      COMPLEX(q)    :: CTMP(T_INFO%NIONS)
      REAL(q)    :: CUTOFF,XX
! 4 grid points around the step function are usually a good choice
      REAL(q), PARAMETER :: WIDTH=4

      IF (.NOT.ORBITALMAG) RETURN
      IF (.NOT. ASSOCIATED(AVEC) .OR. .NOT. ASSOCIATED(AVTOT)) THEN
         WRITE(0,*) 'internal error in VECTORPOT: AVEC or AVTOT not associated'
         STOP
      ENDIF

#ifdef MPI
      IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('CURRENT: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif
!test reallocate the P%JPAW array
      DO NT=1,SIZE(P)
         IF ( ASSOCIATED(P(NT)%JPAW)) DEALLOCATE(P(NT)%JPAW)
      ENDDO
      CALL CALC_JPAW(T_INFO, P)

!     WRITE(*,*) ' forcing JPAW to 0 L=0'
      DO NT=1,SIZE(P)
         IF ( ASSOCIATED(P(NT)%JPAW)) THEN
!            P(NT)%JPAW=0
!            P(NT)%JPAW(:,:,1,:)=0
!            P(NT)%JPAW(:,:,2:,:)=0
!            P(NT)%JPAW(:,:,5:,:)=0
         ENDIF
      ENDDO

      IF (IU6>=0) WRITE(IU6,130)

      ALLOCATE(CWORK1(W%WDES%NRPLWV))
      ALLOCATE(CWORK2(W%WDES%GRID%MPLWV))
      ALLOCATE(JVEC_TMP(W%WDES%GRID%RL%NP,3))
      ALLOCATE(JVEC(GRID_SOFT%MPLWV))
      ALLOCATE(CHDEN_TMP(W%WDES%GRID%RL%NP))
      ALLOCATE(JTOT(GRIDC%MPLWV,3), DIV_J(GRIDC%MPLWV))

      B_OUT_PARA=0
      B_OUT_DIA=0
      MU_PARA=0
      MU_DIA=0

      CALL SETWDES(W%WDES,WDES1,0)
      CALL NEWWAV(W1 , WDES1,.TRUE.)

!-----------------------------------------------------------------------
! first step calculate current density and charge density
! in real space
!-----------------------------------------------------------------------

      JVEC_TMP=0
      CHDEN_TMP=0

      spin:  DO ISP=1,W%WDES%ISPIN
      kpoint: DO NK=1,W%WDES%NKPTS
         CALL SETWDES(W%WDES,WDES1,NK)

         DO N=1,W%WDES%NBANDS
            CALL W1_COPY( ELEMENT( W, WDES1, N, ISP), W1)
            CALL FFTWAV_W1(W1)
            WEIGHT=WDES1%RSPIN*W%FERWE(N,NK,ISP)*W%WDES%WTKPT(NK)
           
            CALL PSCURRENT( WDES1, WDES1%GRID, W1%CR(1), WDES1%IGX(1), WDES1%IGY(1), WDES1%IGZ(1), &
                 WDES1%VKPT, WEIGHT, AVEC(1,1), CWORK1, CWORK2, W1%CW(1), JVEC_TMP, CHDEN_TMP, IU0)
         ENDDO
      ENDDO kpoint
      ENDDO spin

!-----------------------------------------------------------------------
! change from reciprocal space presentation to cartesian coordinates
! sign change because electronic charge is negative
!-----------------------------------------------------------------------
      DO M=1,W%WDES%GRID%RL%NP
         DO I=1,3
            TMP(I)=JVEC_TMP(M,I)
         ENDDO
         CALL DIRKAR(1,TMP,LATT_CUR%B)
         DO I=1,3
            JVEC_TMP(M,I)=-TMP(I)
         ENDDO
      ENDDO

!     WRITE(*,*) 'JVEC'
!     CALL WRT_RL_LINE(6,GRID_SOFT,JVEC_TMP(1,1))
!     CALL WRT_RL_LINE(6,GRID_SOFT,JVEC_TMP(1,2))
!     CALL WRT_RL_LINE(6,GRID_SOFT,JVEC_TMP(1,3))

 130  FORMAT (5X, //, &
     &'----------------------------------------------------', &
     &'----------------------------------------------------'//)

!-----------------------------------------------------------------------
! bring the current density as well as soft pseudo density
! to the fine grid
!-----------------------------------------------------------------------
      JTOT=0
      DO I=1,3
! now merge the current from all nodes
#ifdef realmode
         CALLMPI( M_sum_d(W%WDES%COMM_INTER, JVEC_TMP(1,I), W%WDES%GRID%RL%NP))
#else
         CALLMPI( M_sum_z(W%WDES%COMM_INTER, JVEC_TMP(1,I), W%WDES%GRID%RL%NP))
#endif
         CALL FFT_RC_SCALE(JVEC_TMP(1,I),JVEC(1),GRID_SOFT)
         ! set the current density of unbalanced lattic-vectors to 0
         CALL SETUNB(JVEC(1),GRID_SOFT)

         ! bring to full grid
         IF (.NOT.W%WDES%LOVERL) THEN
            CALL RC_ADD(JVEC(1),1.0_q,JVEC(1),0.0_q,JTOT(1,I),GRID_SOFT)
         ELSE
           CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C,JVEC(1),JTOT(1,I))
           CALL SETUNB_COMPAT(JTOT(1,I),GRIDC)
        ENDIF
      ENDDO

#ifdef realmode
      CALLMPI( M_sum_d(W%WDES%COMM_INTER, CHDEN_TMP(1), W%WDES%GRID%RL%NP))
#else
      CALLMPI( M_sum_z(W%WDES%COMM_INTER, CHDEN_TMP(1), W%WDES%GRID%RL%NP))
#endif
      ! use JVEC as work array
      CALL FFT_RC_SCALE(CHDEN_TMP(1),JVEC(1),GRID_SOFT)
         ! set the current density of unbalanced lattic-vectors to 0
      CALL SETUNB(JVEC(1),GRID_SOFT)

      ! bring to full grid
      CHTOT=0
      WRITE(*,*) 'only plane wave contribution to CHTOT calculated'
      IF (.NOT.W%WDES%LOVERL) THEN
         CALL RC_ADD(JVEC(1),1.0_q,JVEC(1),0.0_q,CHTOT(1,1),GRID_SOFT)
      ELSE
         CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C,JVEC(1),CHTOT(1,1))
         CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
      ENDIF

      ! add augmentation currents
      DISPL=0

#ifdef current_augmentation
      IF (LBFCONST) THEN
         CALL CURRENT_AUGMENTATION( &
              W%WDES, GRIDC, GRIDUS, C_TO_US, &
              LATT_CUR, P, T_INFO, &
              LMDIM, CRHODE(1,1,1,1), JTOT(1,1), IRDMAX, DISPL, .TRUE.)
      ELSE
! in the converse approach, the monopole current augmentation term is
! not added, instead we have an addition term R x [ V_nl - e Q_nl , r]
! see below
         CALL CURRENT_AUGMENTATION( &
              W%WDES, GRIDC, GRIDUS, C_TO_US, &
              LATT_CUR, P, T_INFO, &
              LMDIM, CRHODE(1,1,1,1), JTOT(1,1), IRDMAX, DISPL, .FALSE.)
      ENDIF
#else
      WRITE(*,*) 'only plane wave contribution to JTOT calculated'
#endif
      ! bring total pseudo current and total charge to real space
      DO I=1,3
         CALL FFT3D(JTOT(1,I),GRIDC,1)
      ENDDO
      CALL FFT3D(CHTOT(1,1),GRIDC,1)
!-----------------------------------------------------------------------
! add contributions from non-local pseudopotential
! these term replace the monopole term in AUGMENTATION_CURRENT
!  i R x [V_nl - epsilon_i Q_nl, (r-R)_idir]
!-----------------------------------------------------------------------
      MU_NL=0
      IF (.NOT.LBFCONST) THEN
!
      C_NONL=0._q
      C_NONL_GLOB=0._q

      DO IDIR=1,3
      IF (NONLR_S%LREAL) THEN
!        NONLR_D=NONLR_S
         CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
         CALL NONLR_ALLOC_CRREXP(NONLR_D)
         CALL RSPHER(W%WDES%GRID,NONLR_S,LATT_CUR)  !W%WDES
      ELSE
         CALL SPHER(W%WDES%GRID,NONL_S,P,W%WDES,LATT_CUR, 1)   !W%WDES      P = POTCAR info
         CALL NONL_ALLOC_DER(NONL_S, NONL_D)                    ! NONL_D%QPROJ is allocated, dimension taken from NONL_S
         CALL SPHER_DER(W%WDES%GRID,NONL_D, P, W%WDES, LATT_CUR, IDIR)    !W%WDES
      ENDIF

      DO ISP=1,W%WDES%ISPIN
         DO NK=1,W%WDES%NKPTS
            CALL SETWDES(W%WDES,WDES1,NK)
            IF (NONLR_S%LREAL) THEN
               !   e^ik(r-R)
               CALL PHASER(W%WDES%GRID ,LATT_CUR,NONLR_S,NK,W%WDES)    !W%WDES
               ! r e^ik(r-R)
               CALL PHASERR(W%WDES%GRID,LATT_CUR,NONLR_D,NK,W%WDES,IDIR)    !W%WDES
            ELSE
               !   e^ik(r-R)
               CALL PHASE(W%WDES, NONL_S, NK)
               NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            ENDIF
              
            ! if we want to add the current from the non-local part on the plane wave grid
            ! we can essentially copy lines 395-440 from hamil_lrf.F 
            ! right now we only calculate R x [ V_NL , r-R ] here
            DO N=1,W%WDES%NBANDS                                      ! W1 --> W
               CALL W1_COPY( ELEMENT( W, WDES1, N, ISP), W1)
               CALL FFTWAV_W1(W1)
               WEIGHT=WDES1%RSPIN*W%FERWE(N,NK,ISP)*W%WDES%WTKPT(NK)
               ! caculate <beta_i| r-R| orbital>
               IF (NONLR_S%LREAL) THEN
                  CALL RPRO1(NONLR_D,WDES1, W1)         !W1%WDES1
               ELSE
                  CALL PROJ1(NONL_D, WDES1, W1)
               ENDIF
               ! calculate <orbital|beta_j> D_ij -e Q_ij <beta_i| r-R| orbital>
               CTMP=0
               CALL ECCP_NL_ALL_ION(WDES1,ELEMENT( W, WDES1, N, ISP),W1,CDIJ,CQIJ,REAL(W%CELEN(N,NK,ISP),q),CTMP(:))
               C_NONL(:,IDIR)=C_NONL(:,IDIR)+WEIGHT*CTMP(:)
               CTMP=0
               CALL ECCP_NL_ALL_ION(WDES1,W1,ELEMENT( W, WDES1, N, ISP),CDIJ,CQIJ,REAL(W%CELEN(N,NK,ISP),q),CTMP(:))
               C_NONL(:,IDIR)=C_NONL(:,IDIR)-WEIGHT*CTMP(:)
            ENDDO
         ENDDO
      ENDDO

      IF (NONLR_S%LREAL) THEN
         CALL  NONLR_DEALLOC_CRREXP(NONLR_D)
      ELSE
         CALL  NONL_DEALLOC_DER(NONL_D)
      ENDIF
      ENDDO    ! IDIR
!      C_NONL=AIMAG(C_NONL)*2   ! is still a complex
      C_NONL=C_NONL*(0.0_q,1.0_q)   ! is still a complex

      DO NI=1,W%WDES%NIONS
         NIP=NI_GLOBAL(NI, W%WDES%COMM_INB)
         C_NONL_GLOB(NIP,:)=C_NONL_GLOB(NIP,:)+C_NONL(NI,:)    ! summation should not be necessay
      ENDDO

      CALLMPI( M_sum_z(W%WDES%COMM, C_NONL_GLOB, SIZE(C_NONL_GLOB)))  ! another communicator, seems consistent with elsewhere

      IF (JATOM>=1) THEN
        POS=T_INFO%POSION(:,JATOM)
      ELSE
        POS=0
      ENDIF

      MU_NL=0._q
      DO NI=1,T_INFO%NIONS
         IF (IU0>=0) WRITE(*,'(6F14.7)') C_NONL_GLOB(NI,:)*1E6
!        TMP(:)=MOD(T_INFO%POSION(:,NI)-POS+10.5_q,1.0_q)-0.5_q
! bring grid point and moment position both to interval (-0.5,0.5).
            TMP(1) =MOD(T_INFO%POSION(1,NI)+10.5_q,1._q)-0.5_q
            TMP(2) =MOD(T_INFO%POSION(2,NI)+10.5_q,1._q)-0.5_q
            TMP(3) =MOD(T_INFO%POSION(3,NI)+10.5_q,1._q)-0.5_q
            TMP3(1)=MOD(POS(1)+10.5_q,1._q)-0.5_q
            TMP3(2)=MOD(POS(2)+10.5_q,1._q)-0.5_q
            TMP3(3)=MOD(POS(3)+10.5_q,1._q)-0.5_q
            TMP(1)=TMP(1)-TMP3(1)
            TMP(2)=TMP(2)-TMP3(2)
            TMP(3)=TMP(3)-TMP3(3)
         CALL DIRKAR(1, TMP, LATT_CUR%A)
         TMP1(1) = REAL(C_NONL_GLOB(NI,1),KIND=q)
         TMP1(2) = REAL(C_NONL_GLOB(NI,2),KIND=q)
         TMP1(3) = REAL(C_NONL_GLOB(NI,3),KIND=q)
         CALL EXPRO (TMP2,TMP,TMP1)
         MU_NL(:)=MU_NL(:)+TMP2(:)
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! construct total current density
! upon entry JTOT contains the negative paramagnetic current contribution
! to take into account that electrons are negative
!-----------------------------------------------------------------------
      tmpgil: DO ITMP=1,2
! first just the gradient wf contribution
! then separately the vectorpotential contribution
      IF (ITMP.EQ.2) THEN
         JTOT=0
         
         DO I=1,3
            DO M=1,GRIDC%RL%NP
! j = e hbar/m_e sum_n Im(psi nabla psi) - e^2/m_e c psi^2 A
               JTOT(M,I)= JTOT(M,I) - (MOMTOMOM/MAGMOMTOENERGY)*AVTOT(M,I)*CHTOT(M,1)
! J is in units such that \int r x JTOT dV will yield moment in Bohr magnetons
! cgs: mu = 1/(2c) \int r x j dV =
!      1/(2c) e \hbar/m_e \int r x Im(phi nabla phi) dV  -  1/(2c) e \hbar/m_eA \int r x e/(c hbar) A rho dV
!                    mu_B \int r x Im(phi nabla phi) dV  -                 mu_B \int r x e/(c hbar) A rho dV
! the second integrand:  \int r x e/(c hbar) A rho dV =
!                        \int r x     (e/c hbar) MAGMOM \mu_B (r-r_s)/|r-r_s|^3 rho dV =
!                        \int r x     (e/c hbar) MAGMOM (\hbar e/(2c m_e)) (r-r_s)/|r-r_s|^3 rho      dV
!      MOMTOMOM=              1/AUTOA       e^2/(2 m_e c^2)                AUTOA^2           AUTOA^-3 AUTOA^3  = 1/AUTOA/c/c/2
! Note that MOMTOMOM is dimensionless. In SI units this is difficult to see. In cgs units one has to use that
! the unit of charge is sqrt(g cm^3/s^2).
            ENDDO
         ENDDO
      ENDIF

      IF (LBFCONST) THEN
        IF (ITMP.EQ.1) CALL PS_BFIELD(JTOT(1,1),GRIDC,LATT_CUR,T_INFO,B_OUT_PARA,B_NICS_OUT_PARA,IU0,IU6)
        IF (ITMP.EQ.2) CALL PS_BFIELD(JTOT(1,1),GRIDC,LATT_CUR,T_INFO,B_OUT_DIA,B_NICS_OUT_DIA,IU0,IU6)
      ENDIF
!-----------------------------------------------------------------------
! calculate the total moment
!-----------------------------------------------------------------------

      MU=0

      NG=0
      DO NC=1,GRIDC%RL%NCOL
         N2= GRIDC%RL%I2(NC)
         N3= GRIDC%RL%I3(NC)

         DO N1=1,GRIDC%RL%NROW
            NG=NG+1
            IF (GRIDC%RL%NFAST==3) THEN
               NX=N2
               NY=N3
               NZ=N1
            ELSE
               NX=N1
               NY=N2
               NZ=N3
            ENDIF
!           TMP(1)=MOD((REAL(NX,q)-1)/GRIDC%NGX-POS(1)+10.5_q,1._q)-0.5_q
!           TMP(2)=MOD((REAL(NY,q)-1)/GRIDC%NGY-POS(2)+10.5_q,1._q)-0.5_q
!           TMP(3)=MOD((REAL(NZ,q)-1)/GRIDC%NGZ-POS(3)+10.5_q,1._q)-0.5_q
! bring grid point and moment position both to interval (-0.5,0.5).
            TMP(1)=MOD((REAL(NX,q)-1)/GRIDC%NGX+10.5_q,1._q)-0.5_q
            TMP(2)=MOD((REAL(NY,q)-1)/GRIDC%NGY+10.5_q,1._q)-0.5_q
            TMP(3)=MOD((REAL(NZ,q)-1)/GRIDC%NGZ+10.5_q,1._q)-0.5_q
            TMP3(1)=MOD(POS(1)+10.5_q,1._q)-0.5_q
            TMP3(2)=MOD(POS(2)+10.5_q,1._q)-0.5_q
            TMP3(3)=MOD(POS(3)+10.5_q,1._q)-0.5_q
            TMP(1)=TMP(1)-TMP3(1)
            TMP(2)=TMP(2)-TMP3(2)
            TMP(3)=TMP(3)-TMP3(3)
!test
!            TMP(1)=MOD(T_INFO%POSION(1,2)-POS(1)+10.5_q,1._q)-0.5_q
!            TMP(2)=MOD(T_INFO%POSION(2,2)-POS(2)+10.5_q,1._q)-0.5_q
!            TMP(3)=MOD(T_INFO%POSION(3,2)-POS(3)+10.5_q,1._q)-0.5_q
!testend
            CALL DIRKAR(1, TMP, LATT_CUR%A)
           
            JTMP(1)=REAL(JTOT(NG,1))
            JTMP(2)=REAL(JTOT(NG,2))
            JTMP(3)=REAL(JTOT(NG,3))
            CALL EXPRO(TMP2(1), TMP(1), JTMP(1))

              CUTOFF=1.0

              XX=ABS(NX-1-GRIDC%NGX/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF
              XX=ABS(NY-1-GRIDC%NGY/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF
              XX=ABS(NZ-1-GRIDC%NGZ/2)
              IF (XX <= WIDTH)  THEN
                 CUTOFF= ABS(SIN(PI*XX/WIDTH/2))*CUTOFF
              ENDIF

            MU(1)=MU(1)+TMP2(1)*CUTOFF
            MU(2)=MU(2)+TMP2(2)*CUTOFF
            MU(3)=MU(3)+TMP2(3)*CUTOFF
         ENDDO
      ENDDO

      MU(:)=MU(:)/GRIDC%NPLWV
      CALLMPI( M_sum_d(GRIDC%COMM, MU, 3))
      MAGDIPOLOUT(:) = MU(:)

      IF (ITMP.EQ.1) MU_PARA(:)=MU(:)
      IF (ITMP.EQ.2) MU_DIA(:)=MU(:)

!      IF (IU6>=0) WRITE (IU6,'("induced magnetic moment   *1E6",3E16.7)') MAGDIPOLOUT*1E6
!      IF (IU0>=0) WRITE (IU0,'("induced magnetic moment   *1E6",3E16.7)') MAGDIPOLOUT*1E6

      ENDDO tmpgil

      IF (.NOT. LBFCONST) THEN
! here one-centre terms for converse method
         MU_PARA_ONE_CTR=0
         MU_DIA_ONE_CTR=0
         MU_LQ_ONE_CTR=0

         ! type of magnetic ion
         NT=T_INFO%ITYP(MAGATOM)
         PP=>P(NT)
         NIP=NI_LOCAL(MAGATOM,W%WDES%COMM_INB)
         IF (NIP/=0) THEN
            IF (ALLOCATED(DO_LOCAL)) THEN
            IF (DO_LOCAL(MAGATOM)) THEN                  !bug NT was NI, 11 jun 2008, 24 jul 2008: NT-->MAGATOM
               CALL CALC_MU_MAGATOM(PP, CRHODE(:,:,NIP,1),MU_PARA_ONE_CTR,MU_DIA_ONE_CTR,.TRUE.)
               CALL CALC_MU_LQ(PP,CRHODE(:,:,NIP,1),MU_LQ_ONE_CTR)
            ENDIF
            ENDIF
         ENDIF
         CALLMPI( M_sum_d(W%WDES%COMM, MU_PARA_ONE_CTR(1), SIZE(MU_PARA_ONE_CTR)))
         CALLMPI( M_sum_d(W%WDES%COMM, MU_DIA_ONE_CTR(1), SIZE(MU_DIA_ONE_CTR)))
         CALLMPI( M_sum_d(W%WDES%COMM, MU_LQ_ONE_CTR(1), SIZE(MU_LQ_ONE_CTR)))
! add paramagnetic one centre terms for each atom 
! this replaces MU_PARA_ONE_CTR
#ifndef current_augmentation
         MU_LQ_ONE_CTR=0
         ! type of magnetic ion
         DO NI=1,T_INFO%NIONS        ! index of ion

            NIP=NI_LOCAL(NI, W%WDES%COMM_INB)   ! local index of ion
            IF (NIP==0) CYCLE                   ! not stored locally return
            NT=T_INFO%ITYP(NI)  ! type of ion
            PP=> P(NT)

            IF ( ASSOCIATED(PP%QPAW)) THEN
               IF (DO_LOCAL(NI)) THEN
                  CALL CALC_MU_LQ(PP,CRHODE(:,:,NIP,1),MU_LQ_ONE_CTR)
               ENDIF
            ENDIF

         ENDDO
         CALLMPI( M_sum_d(W%WDES%COMM, MU_LQ_ONE_CTR(1), SIZE(MU_LQ_ONE_CTR)))
         MU_PARA_ONE_CTR=MU_LQ_ONE_CTR
#endif
      ELSE
! here one-centre for direct method
         B_OUT_PARA_ONE_CENTR=0
         B_OUT_DIA_ONE_CENTR=0
         DO NI=1,T_INFO%NIONS        ! index of ion

            NIP=NI_LOCAL(NI, W%WDES%COMM_INB)   ! local index of ion
            IF (NIP==0) CYCLE                   ! not stored locally return
            NT=T_INFO%ITYP(NI)                  ! type of ion
            PP=> P(NT)

            IF ( ASSOCIATED(PP%QPAW)) THEN
              IF (DO_LOCAL(NI)) THEN
                 CALL CALC_B_MAGATOM(PP, CRHODE(:,:,NIP,1),NIP,NI, &
                      B_OUT_PARA_ONE_CENTR(:,NI),B_OUT_DIA_ONE_CENTR(:,NI))
              ENDIF
            ENDIF
         ENDDO
         CALLMPI( M_sum_d(W%WDES%COMM, B_OUT_PARA_ONE_CENTR(1,1), SIZE(B_OUT_PARA_ONE_CENTR)))
         CALLMPI( M_sum_d(W%WDES%COMM, B_OUT_DIA_ONE_CENTR(1,1), SIZE(B_OUT_DIA_ONE_CENTR)))
      ENDIF

! write outcar and stdout
      IF (LBFCONST) THEN
        IF(.NOT. LNICSALL) THEN

          !write output nics
          IF(IU6 >= 0) THEN
            WRITE (IU6,*)
            WRITE (IU6,*)'-------------------------------------------------------------'
            WRITE (IU6,*)'         NICS AT POISITION READ IN FILE POSNICS              '
            WRITE (IU6,*)'                       (dia + para)                          '
            WRITE (IU6,*)'-------------------------------------------------------------'
            WRITE (IU6,*)'                   MAGNETIC FIELD    (mu_B/Angstrom3/1e6)    '
            WRITE (IU6,*)' ATOM                 X                 Y                  Z '
            WRITE (IU6,*)'-------------------------------------------------------------'
            DO NI=1,NPOS_NICS
               WRITE (IU6,'(I6,3E18.6)') NI,B_NICS_OUT_PARA(1,NI)+B_NICS_OUT_DIA(1,NI),&
                                                 B_NICS_OUT_PARA(2,NI)+B_NICS_OUT_DIA(2,NI),&
                                                 B_NICS_OUT_PARA(3,NI)+B_NICS_OUT_DIA(3,NI)
            ENDDO
          ENDIF

         ENDIF !LNICSALL


         B_OUT_PARA(:,:)=B_OUT_PARA(:,:)*1E6
         B_OUT_DIA(:,:) =B_OUT_DIA(:,:) *1E6
         B_OUT_PARA_ONE_CENTR(:,:)=B_OUT_PARA_ONE_CENTR(:,:)*1E6
         B_OUT_DIA_ONE_CENTR(:,:) =B_OUT_DIA_ONE_CENTR(:,:) *1E6
         IF (IU6>=0) THEN
            WRITE (IU6,*)
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*) '                       MAGNETIC FIELD (mu_B/Angstrom3/1e6)'
            WRITE (IU6,*) ' ATOM                 X                 Y                 Z'
            WRITE (IU6,*) '-------------------------------------------------------------'
            DO NI=1,T_INFO%NIONS
               WRITE (IU6,'(I6,3E18.6)') NI, &
                  B_OUT_PARA(1,NI)+B_OUT_DIA(1,NI)+B_OUT_PARA_ONE_CENTR(1,NI)+B_OUT_DIA_ONE_CENTR(1,NI), &
                  B_OUT_PARA(2,NI)+B_OUT_DIA(2,NI)+B_OUT_PARA_ONE_CENTR(2,NI)+B_OUT_DIA_ONE_CENTR(2,NI), &
                  B_OUT_PARA(3,NI)+B_OUT_DIA(3,NI)+B_OUT_PARA_ONE_CENTR(3,NI)+B_OUT_DIA_ONE_CENTR(3,NI)
            ENDDO
            WRITE (IU6,*)
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*) '  plane wave paramagnetic contribution'
            WRITE (IU6,*) '-------------------------------------------------------------'
            DO NI=1,T_INFO%NIONS
               WRITE (IU6,'(I6,3E18.6)') NI,B_OUT_PARA(1,NI), &
                                            B_OUT_PARA(2,NI), &
                                            B_OUT_PARA(3,NI)
            ENDDO
            WRITE (IU6,*)
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*) '  plane wave diamagnetic contribution'
            WRITE (IU6,*) '-------------------------------------------------------------'
            DO NI=1,T_INFO%NIONS
               WRITE (IU6,'(I6,3E18.6)') NI,B_OUT_DIA(1,NI), &
                                            B_OUT_DIA(2,NI), &
                                            B_OUT_DIA(3,NI)
            ENDDO
            WRITE (IU6,*)
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*) '  one center paramagnetic contribution'
            WRITE (IU6,*) '-------------------------------------------------------------'
            DO NI=1,T_INFO%NIONS
               WRITE (IU6,'(I6,3E18.6)') NI,B_OUT_PARA_ONE_CENTR(1,NI), &
                                            B_OUT_PARA_ONE_CENTR(2,NI), &
                                            B_OUT_PARA_ONE_CENTR(3,NI)
            ENDDO
            WRITE (IU6,*)
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*) '  one center diamagnetic contribution'
            WRITE (IU6,*) '-------------------------------------------------------------'
            DO NI=1,T_INFO%NIONS
               WRITE (IU6,'(I6,3E18.6)') NI,B_OUT_DIA_ONE_CENTR(1,NI), &
                                            B_OUT_DIA_ONE_CENTR(2,NI), &
                                            B_OUT_DIA_ONE_CENTR(3,NI)
            ENDDO
            WRITE (IU6,*) '-------------------------------------------------------------'
            WRITE (IU6,*)

         ENDIF
      ELSE ! LBFCONST=.FALSE.
         IF(.NOT. NUCIND) THEN
         MU_PARA(:)=MU_PARA(:)*1E6
         MU_DIA(:)=MU_DIA(:)*1E6
         MU_PARA_ONE_CTR(:)=MU_PARA_ONE_CTR(:)*1E6
         MU_DIA_ONE_CTR(:)=MU_DIA_ONE_CTR(:)*1E6
         MU_NL(:)=MU_NL(:)*1E6
         IF (IU6>=0) THEN
            WRITE (IU6,*)
            WRITE (IU6,*)'-------------------------------------------------------------'
            WRITE (IU6,*)'               molecular converse approach                   '
            WRITE (IU6,*)'-------------------------------------------------------------'

            WRITE (IU6,*)
            WRITE (IU6,111)
            WRITE (IU6,*) '                                                 MAGNETIC MOMENT (mu_B/1e6)'
            WRITE (IU6,1111)
            WRITE (IU6,111)
            WRITE (IU6,'(I6,22X,3F18.6)') MAGATOM, &
                          MU_PARA(1)+MU_DIA(1)+MU_PARA_ONE_CTR(1)+MU_DIA_ONE_CTR(1), &
                          MU_PARA(2)+MU_DIA(2)+MU_PARA_ONE_CTR(2)+MU_DIA_ONE_CTR(2), &
                          MU_PARA(3)+MU_DIA(3)+MU_PARA_ONE_CTR(3)+MU_DIA_ONE_CTR(3)
            WRITE (IU6,'(I6,22X,3F18.6)') MAGATOM, &
                          MU_PARA(1)+MU_DIA(1)+MU_PARA_ONE_CTR(1)+MU_DIA_ONE_CTR(1)+MU_NL(1)/HSQDTM/2, &
                          MU_PARA(2)+MU_DIA(2)+MU_PARA_ONE_CTR(2)+MU_DIA_ONE_CTR(2)+MU_NL(2)/HSQDTM/2, &
                          MU_PARA(3)+MU_DIA(3)+MU_PARA_ONE_CTR(3)+MU_DIA_ONE_CTR(3)+MU_NL(3)/HSQDTM/2
            WRITE (IU6,*)
            WRITE (IU6,'(A11,3F18.6)') ' plane wave'
            WRITE (IU6,'(A28,3F18.6)') '   paramagnetic contribution',MU_PARA(:)
            WRITE (IU6,'(A28,3F18.6)') '   diamagnetic contribution ',MU_DIA(:)
            WRITE (IU6,'(A11,3F18.6)') ' one centre'
            WRITE (IU6,'(A28,3F18.6)') '   paramagnetic contribution',MU_PARA_ONE_CTR(:)
            WRITE (IU6,'(A28,3F18.6)') '   diamagnetic contribution ',MU_DIA_ONE_CTR(:)
            WRITE (IU6,'(A28,3F18.6)') '   non-local contribution   ',MU_NL(:)/HSQDTM/2
            WRITE (IU6,111)
            WRITE (IU6,*)
         ENDIF
         ENDIF
      ENDIF  !END LBFCONST

  111 FORMAT('-----------------------------------------------------------------------------------')
 1111 FORMAT('  ATOM                                       X                 Y                 Z')

      ! total charge back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)

      DEALLOCATE(CWORK1)
      DEALLOCATE(CWORK2)
      DEALLOCATE(JVEC_TMP)
      DEALLOCATE(JVEC)
      DEALLOCATE(CHDEN_TMP)
      DEALLOCATE(JTOT, DIV_J)

    END SUBROUTINE CURRENT

!**************** SUBROUTINE EXPRO_CRC *********************************
! EXPRO
! caclulates the x-product of two vectors
! all vectors are complex
!
!***********************************************************************

      SUBROUTINE EXPRO_CRC(H,U1,U2)
      USE prec
      IMPLICIT NONE
      COMPLEX(q) H,U2
      REAL(q) U1
      DIMENSION H(3),U1(3),U2(3)

      H(1)=U1(2)*U2(3)-U1(3)*U2(2)
      H(2)=U1(3)*U2(1)-U1(1)*U2(3)
      H(3)=U1(1)*U2(2)-U1(2)*U2(1)

      RETURN
      END SUBROUTINE

!***************************** SUBROUTINE CALC_DD_MAGATOM **************************
!
! one centre terms:
!
! for constant magnetic field (or slowly varying vector potential A), everything can 
! be done on the plane wave grid calculating
!  D_ij = \int(r) j_ij(r) A(r) d^3 r  [j_ij is the augmentation current]
! (see SETDIJ_AVEC_)
! if SETDIJ_AVEC is not used, SET_DD_BFCONST can be alternatively used for
! a constant magnetic field
!
! for the converse approach the one centre term for the atom where the magnetic moment
! is calculated as
!
!      <phi^AE_j | nabla . A + A . nabla |phi^AE_i> -
!      <phi^PS_j | nabla . A + A . nabla |phi^PS_i>
!
! where A = m_s x (r-r_s)/|r-r_s|^3
!         = [ms_x, ms_y, ms_z] x [Y_x, Y_y, Y_z] sqrt{4 pi/3} /|r-r_s|^2
!         = (\hat{x} (ms_y Y_z - ms_z Y_y)
!            \hat{y} (ms_z Y_x - ms_x Y_z)
!            \hat{z} (ms_x Y_y - ms_y Y_x)) sqrt{4 pi/3} /|r-r_s|^2
!         = (\hat{x} (ms_y Y(lm=3) - ms_z Y(lm=2))
!            \hat{y} (ms_z Y(lm=4) - ms_x Y(lm=3))
!            \hat{z} (ms_x Y(lm=2) - ms_y Y(lm=4))) sqrt{4 pi/3} /|r-r_s|^2
!
! units: - (-|e|)/(m_e c) AUTOA**3 mu_B hbar
!
! lacking for hamiltonian:
!    multipy by MAGMOMTOENERGY to get eV
!    multipy by -i to get proper action momentum operator
!
!    AGRAD still lives locally
!
! Note: vasp inverts the indexing internally for reasons of speed
!
! Based on CALC_JPAW: see over there
!
!  P= -i hbar nabla
!
!***********************************************************************************


    SUBROUTINE SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)
      USE pseudo
      USE poscar
      USE constant
      USE asa
      USE radial
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)         :: WDES          ! wave function descriptor
      TYPE (type_info)       :: T_INFO        ! type information
      TYPE (potcar), TARGET  :: P(:)          ! pseudopotential header
      INTEGER                :: LMDIM
      OVERLAP                :: CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
! local
      TYPE (potcar), POINTER :: PP
      INTEGER CHANNELS          ! number of channels
     !local variables
      INTEGER NI, NIP, ISP, LYMAX
      INTEGER NMAX,I,LMAX,NT
      INTEGER I0,I1
      INTEGER L0,L1,L2,L3
      INTEGER M0,M1,M3
      INTEGER LM0,LM1,LM2,LM3,LM2MAX
      INTEGER ISTART1,IEND1,LMIND1,IC1
      INTEGER LMAXNABLA,K
      REAL(q) :: CGOR
      REAL(q) :: SUM
      REAL(q) :: FACT
      REAL(q), ALLOCATABLE :: AGRAD(:,:)
      OVERLAP, ALLOCATABLE :: CTMP(:,:)
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:)

      IF (.NOT. ORBITALMAG) RETURN

!-----------------------------------------------------------------------
! if constant field is applied, everything already handled by SETDIJ_AVEC 
! alternatively we can use the routine SET_DD_BFCONST
! 20.04.2011  for constant field  SET_DD_BFCONST and SETDIJ_AVEC 
!             do exactly the same for P4
!-----------------------------------------------------------------------
      IF (LBFCONST) THEN
#ifndef dij_augmentation
         CALL SET_DD_BFCONST(WDES, T_INFO, P, LMDIM, CDIJ)
#endif
         RETURN
      ENDIF

      FACT = SQRT(4._q*PI/3._q)
      NI=MAGATOM                           ! index of ion
      NIP=NI_LOCAL(NI, WDES%COMM_INB)      ! local index of ion

      IF (NIP==0) RETURN                   ! not stored locally return
      NT=T_INFO%ITYP(NI)                   ! type of ion

      PP=> P(NT)
      IF ( .NOT. ASSOCIATED(PP%QPAW)) RETURN

      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX))

      CHANNELS = PP%LMAX

! ALLOCATE help arrays

      LMAXNABLA=MAXL1(PP)+1   ! nabla Y_lm has components up to LMAX+1
! isn't this double?

      ALLOCATE(AGRAD(PP%LMMAX,PP%LMMAX), CTMP(PP%LMMAX,PP%LMMAX))

! ALLOCATE and setup nabla array
      ALLOCATE(YLM_NABLA_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))
      ALLOCATE(YLM_X_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q

      CALL SETYLM_NABLA_YLM(LMAXNABLA,YLM_NABLA_YLM,YLM_X_YLM)
 
      AGRAD=0._q   
!-----------------------------------------------------------------------
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------
      LM2MAX=0

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
!        WTMP(:)=1/PP%R%R         
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

!test
!        DWPS=0

         DO M1=1,2*L1+1

            LM3=1
            DO L3=0,LMAXNABLA

            CALL YLM3LOOKUP(L0,L3,LMIND1)

            DO M0=1,2*L0+1
            DO M3=1,2*L3+1
               LMIND1 = LMIND1+1
 
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               ! this loop goes over all possible l'', m'' being non zero in CG coefficien
               DO IC1=ISTART1,IEND1-1
                  ! JS(IC1)) is a compound index for l'' and m''
                  ! JL(IC1)) is the l quantum number L (zero based)
                  ! see asa.F line 286 how compound index is defined,
                  ! l''  m''    JS
                  ! 0    0      1
                  ! 1    -1     2
                  ! 1    0      3
                  ! 1    1      4
                  ! 2    -2     5
                  ! coefficient C(l'',m'',l,m,l'm') is stored in YLM3(IC1)
                  ! any statments at this place will run over all  (lm, l'm', l'',m'')
                  ! any statments at this place will run over all  (l0m0, l2m2, l3,m3)

                  CGOR=YLM3(IC1)
                  LM2 =JS(IC1)
                  L2  =JL(IC1)

                  ! filter out LM that contribute to vectorpotential
                  IF (L2.EQ.1) THEN

                     DO I=1,3
                     IF (ABS(YLM_X_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                       TMP=0._q
                       SUM=0._q
! 1/R%R(K)**2 needed to convert WR and R%R(K)**2 cancel
! it is not so bad to have the radial integration here
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * (                                                         &
      &        PP%WAE(K,I0) * ( DWAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
      &                             + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WAE(K,I1) )  &
      &      - PP%WPS(K,I0) * ( DWPS(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
      &                             + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WPS(K,I1) )  &
      &                     ) * FACT / PP%R%R(K)**2
                        ENDDO
                        CALL SIMPI(PP%R,TMP,SUM)
                        IF (LM0+M0-1 <=0 .OR. LM0+M0-1 > SIZE(AGRAD,1)) THEN
                           WRITE(0,*) 'internal error in CALC_D_MAGATOM: index 1 into AGRAD exceeds bounds'
                           STOP
                        ENDIF
                        IF (LM1+M1-1 <=0 .OR. LM1+M1-1 > SIZE(AGRAD,2)) THEN
                           WRITE(0,*) 'internal error in CALC_D_MAGATOM: index 2 into AGRAD exceeds bounds'
                           STOP
                        ENDIF
 
                        IF (LM2.EQ.2) THEN
                           !p_y
                           IF (I.EQ.1) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) - SUM*MAGDIPOL(3)
                           ENDIF
                           IF (I.EQ.3) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) + SUM*MAGDIPOL(1)
                           ENDIF
                        ENDIF
                        IF (LM2.EQ.3) THEN
                           !p_z
                           IF (I.EQ.1) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) + SUM*MAGDIPOL(2)
                           ENDIF
                           IF (I.EQ.2) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) - SUM*MAGDIPOL(1)
                           ENDIF
                        ENDIF
                        IF (LM2.EQ.4) THEN
                           !p_x
                           IF (I.EQ.2) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) + SUM*MAGDIPOL(3)
                           ENDIF
                           IF (I.EQ.3) THEN
                              AGRAD(LM1+M1-1,LM0+M0-1) = AGRAD(LM1+M1-1,LM0+M0-1) - SUM*MAGDIPOL(2)
                           ENDIF
                        ENDIF
                        LM2MAX=MAX(LM2, LM2MAX)
                     ENDIF
                     ENDDO ! I

                  ENDIF

               ENDDO  ! IC1

            ENDDO ! M3
            ENDDO ! M0

            LM3=LM3+2*L3+1
            ENDDO ! L3

         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

      ! units
      AGRAD(:,:)=AGRAD(:,:)*MAGMOMTOENERGY

      ! make antisymmetric
      DO I0=1,PP%LMMAX
         DO I1=I0,PP%LMMAX
            AGRAD(I0,I1)= (AGRAD(I0,I1)-AGRAD(I1,I0))/2
            AGRAD(I1,I0)= -AGRAD(I0,I1)
         ENDDO
      ENDDO
      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM)
      DEALLOCATE(WTMP,DWAE,DWPS,TMP)

      CTMP=0
#ifdef dij_augmentation
      CALL CALC_D_CORR_JVEC(PP, CTMP)
#endif

!      NI=1
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (-AIMAG( (0._q,1._q)*(AGRAD(L3,L1)*1E3)),L3=1,MIN(8,PP%LMMAX))
!      ENDDO
!      WRITE(*,*)
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (AIMAG(-(0._q,1._q)*(CTMP(L3,L1)*1E3)),L3=1,MIN(8,PP%LMMAX))
!      ENDDO
!      WRITE(*,*)

      CALL CDIJ_FLIP(WDES, LMDIM, CDIJ, .TRUE., .FALSE.)
      CDIJ(1:PP%LMMAX, 1:PP%LMMAX, NIP, 1) = CDIJ(1:PP%LMMAX, 1:PP%LMMAX, NIP, 1)  &
     &     - (0._q,1._q)*(AGRAD + CTMP)
      CALL CDIJ_FLIP(WDES, LMDIM, CDIJ, .TRUE., .TRUE.)

!test dump DIJ contributions
!      NI=1
!      DO ISP=1,WDES%NCDIJ
!         DO L1=1,P(1)%LMMAX
!            WRITE(*,'(16(F7.3,1X))') (REAL(CDIJ(L3,L1,NIP,ISP)),L3=1,MIN(8,P(1)%LMMAX))
!         ENDDO
!         DO L1=1,P(1)%LMMAX
!            WRITE(*,'(16(F7.3,1X))') (-AIMAG(CDIJ(L3,L1,NIP,ISP)*1000),L3=1,MIN(8,P(1)%LMMAX))
!         ENDDO
!         WRITE(*,*)
!      ENDDO

      DEALLOCATE(AGRAD, CTMP)
!     STOP

    END SUBROUTINE SET_DD_MAGATOM

!***************************** SUBROUTINE CALC_D_CORR_JVEC ********************************
!
! one centre term for correction of one-center augmentation via moment
!
! \int A(r) g_l(r) Y_LM(r) dr with
!      A(r) = m_s x (r-r_s)/|r-r_s|^3 in real space
!
! units: - (-|e|)/(m_e c) AUTOA**3 mu_B hbar
!
! lacking for hamiltonian:
!    multipy by MAGMOMTOENERGY to get eV
!    multipy by -i to get proper action momentum operator
!
!    D_CORR_JVEC still lives locally
!
! Based on CALC_JPAW and SETDIJ_AVEC see comments there
!
!  P= -i hbar nabla
!
!
!***********************************************************************************


    SUBROUTINE CALC_D_CORR_JVEC(PP, CTMP)
      USE pseudo
      USE constant
      USE asa
      USE radial
      USE wave
      USE paw
      IMPLICIT NONE

      TYPE (potcar), POINTER :: PP
      INTEGER CHANNELS          ! number of channels
     !local variables
      INTEGER LMAXNABLA,K,LMAX,I1
      REAL(q)  :: FACT
      OVERLAP  :: DLM(256),SUM
      OVERLAP  :: CTMP(:,:)
      REAL(q), ALLOCATABLE :: TMP(:)
      REAL(q)  :: SUMA
      INTEGER NMAX

      FACT = SQRT(4._q*PI/3._q)

      SUM=0._q
      DO K=1,PP%R%NMAX
         SUM=SUM+PP%AUG(K,1)*PP%R%SI(K)/PP%R%R(K)**2
      ENDDO

      SUM=SUM*FACT
      SUM=SUM*MAGMOMTOENERGY

! the -i is not included in AVEC, so should not be included here.

      CTMP=0._q

! x component
      DLM=0
      DLM(3) =   MAGDIPOL(2) * SUM
      DLM(2) = - MAGDIPOL(3) * SUM

      CALL CALC_DLLMM_AVEC( CTMP, DLM, PP, 1)

! y component

      DLM=0
      DLM(4) =   MAGDIPOL(3) * SUM
      DLM(3) = - MAGDIPOL(1) * SUM

      CALL CALC_DLLMM_AVEC( CTMP, DLM, PP, 2)

! z component

      DLM=0
      DLM(2) =   MAGDIPOL(1) * SUM
      DLM(4) = - MAGDIPOL(2) * SUM

      CALL CALC_DLLMM_AVEC( CTMP, DLM, PP, 3)

! minus

      CTMP = - CTMP

    END SUBROUTINE CALC_D_CORR_JVEC

!***************************** SUBROUTINE CALC_DD_BFCONST **************************
!
! one centre term for each atom in constant magnetic field
!
!   <phi^AE_j | Bxr/2 . nabla/i |phi^AE_i> - <phi^PS_j | Bxr/2 . nabla/i |phi^PS_i>
!
!   using (Bxr).p = (rxp).B = B.(rxp)
!
!   B.(<phi^AE_j | r |Y_lm> x <Y_lm| nabla/i |phi^AE_i> - ... )
!
! LGAUGE implicitely assumed, and constant magnetic field assumed as well
! for these cases,  CALC_DD_BFCONST yields exactly the same contribution to
! the Hamiltonian as SETDIJ_AVEC
! matter of fact this routine is more restrictive in the sense that
! only constant B fields are supported
!
! the routine is only called if dij_augmentation is not set
!
!***********************************************************************************

    SUBROUTINE SET_DD_BFCONST(WDES, T_INFO, P, LMDIM, CDIJ)
      USE pseudo
      USE poscar
      USE constant
      USE asa
      USE radial
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)         :: WDES          ! wave function descriptor
      TYPE (type_info)       :: T_INFO        ! type information
      TYPE (potcar), TARGET  :: P(:)          ! pseudopotential header
      INTEGER                :: LMDIM
      OVERLAP                :: CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      TYPE (potcar), POINTER :: PP
      INTEGER                :: NI,NIP,NT

      INTEGER CHANNELS          ! number of channels
      INTEGER NMAX,I
      INTEGER I0,I1
      INTEGER L0,L1,L2
      INTEGER M0,M1,M2
      INTEGER LM0,LM1,LM2,LM2MAX,LM2_IND
      INTEGER LMAXNABLA,LMAX,K,IDP
      INTEGER II1,II2,II3,LL,M
      INTEGER LRHO,LMRHO,MRHO,L2MAX,LCHECK
      REAL(q) :: SUMM
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: JPAWRAD(:,:,:)
      COMPLEX(q), ALLOCATABLE :: D_TMP(:,:)
      COMPLEX(q) :: TMP2(3), TMP3

      IF (.NOT. ORBITALMAG) RETURN

      ions: DO NI=1,T_INFO%NIONS        ! index of ion

      NIP=NI_LOCAL(NI, WDES%COMM_INB)   ! local index of ion
      IF (NIP==0) CYCLE                 ! not stored locally return
      NT=T_INFO%ITYP(NI)                ! type of ion

      PP=> P(NT)
      IF ( .NOT. ASSOCIATED(PP%QPAW)) CYCLE

      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX))

      CHANNELS = PP%LMAX
      LMAX=MAXL1(PP)
      LMAXNABLA=MAXL1(PP)+1   ! nabla Y_lm has components up to LMAX+1
      ALLOCATE(JPAWRAD(PP%R%NMAX,(LMAXNABLA+LMAX+1)*(LMAXNABLA+LMAX+1),3))        ! be ware
      ALLOCATE(D_TMP(PP%LMMAX,PP%LMMAX))

      ALLOCATE(YLM_NABLA_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAXNABLA+2)*(2*LMAXNABLA+2),0:3))
      ALLOCATE(YLM_X_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q

      CALL SETYLM_NABLA_YLM(2*LMAX+1,YLM_NABLA_YLM,YLM_X_YLM)

      D_TMP=0._q

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS

         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

         DO M1=1,2*L1+1
         DO M0=1,2*L0+1

            JPAWRAD=0._q
            LM2MAX=0
            L2MAX=0

            LM2=1
            DO L2=0,LMAX+1
            DO M2=1,2*L2+1

               LM2_IND=LM2+M2-1
               LCHECK=L2*L2+M2
               IF (LM2_IND.NE.LCHECK) THEN
                  WRITE (0,*) 'error stop'
                  STOP
               ENDIF

               IF (LM2_IND <=0 .OR. LM2_IND > SIZE(JPAWRAD,2)) THEN
                  WRITE(0,*) 'internal error in SET_DD_BFCONST: index 2 into JPAWRAD exceeds bounds'
                  STOP
               ENDIF
               LM2MAX=MAX(LM2_IND, LM2MAX)
               L2MAX=MAX(L2, L2MAX)

               DO I=1,3

                  IF (ABS(YLM_X_YLM(LM2_IND,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM2_IND,L1*L1+M1,I))>1E-8) THEN
                     TMP=0._q
                     DO K=1,PP%R%NMAX
                        TMP(K) = (                                                               &
        &   PP%WAE(K,I0) * ( DWAE(K) * YLM_X_YLM(LM2_IND,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM2_IND,L1*L1+M1,I)/PP%R%R(K) * PP%WAE(K,I1) )  &
        & -(PP%WPS(K,I0) * ( DWPS(K) * YLM_X_YLM(LM2_IND,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM2_IND,L1*L1+M1,I)/PP%R%R(K) * PP%WPS(K,I1)))  &
        &                  )
                     ENDDO
                     DO K=1,PP%R%NMAX
                        JPAWRAD(K,LM2_IND,I)=JPAWRAD(K,LM2_IND,I)+ TMP(K)
                     ENDDO
                  ENDIF

               ENDDO  ! I

            ENDDO ! M2
            LM2=LM2+2*L2+1
            ENDDO ! L2

            TMP2=0
            DO LM2_IND=1,LM2MAX
               DO I=1,3
     
                  IF (LM2_IND <1 .OR. LM2_IND > SIZE(JPAWRAD,2)) THEN
                     WRITE(0,*) 'internal error SET_DD_BFCONST',LM2,SIZE(JPAWRAD,2)
                     STOP
                  ENDIF

                  TMP=0._q
                  SUMM=0._q
                  DO K=1,PP%R%NMAX
                     IF (I.EQ.1) TMP(K)=(YLM_X_YLM(L0*L0+M0,LM2_IND,2)*JPAWRAD(K,LM2_IND,3)  &
                                        -YLM_X_YLM(L0*L0+M0,LM2_IND,3)*JPAWRAD(K,LM2_IND,2)) *PP%R%R(K)
                     IF (I.EQ.2) TMP(K)=(YLM_X_YLM(L0*L0+M0,LM2_IND,3)*JPAWRAD(K,LM2_IND,1)  &
                                        -YLM_X_YLM(L0*L0+M0,LM2_IND,1)*JPAWRAD(K,LM2_IND,3)) *PP%R%R(K)
                     IF (I.EQ.3) TMP(K)=(YLM_X_YLM(L0*L0+M0,LM2_IND,1)*JPAWRAD(K,LM2_IND,2)  &
                                        -YLM_X_YLM(L0*L0+M0,LM2_IND,2)*JPAWRAD(K,LM2_IND,1)) *PP%R%R(K)
                  ENDDO
         
                  CALL SIMPI(PP%R,TMP,SUMM)
                  TMP2(I)=TMP2(I)+SUMM
         
               ENDDO
            ENDDO
         
            DO I=1,3
               D_TMP(LM0+M0-1,LM1+M1-1)=D_TMP(LM0+M0-1,LM1+M1-1)+TMP2(I)*BCONST(I)/2._q*MAGMOMTOENERGY
            ENDDO

         ENDDO ! M0
         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

!test dump DIJ contributions
!      WRITE (*,*) 'NI', NI
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (REAL(D_TMP(L2,L1))*1d6,L2=1,MIN(8,PP%LMMAX))
!      ENDDO
!      WRITE (*,*)
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (-AIMAG(D_TMP(L2,L1)*1d6),L2=1,MIN(8,PP%LMMAX))
!      ENDDO
!      WRITE (*,*)

      ! make antisymmetric
!     DO I0=1,PP%LMMAX
!        DO I1=I0,PP%LMMAX
!           D_TMP(I0,I1)= (D_TMP(I0,I1)-D_TMP(I1,I0))/2
!           D_TMP(I1,I0)= -D_TMP(I0,I1)
!        ENDDO
!     ENDDO

      CALL CDIJ_FLIP(WDES, LMDIM, CDIJ, .TRUE., .FALSE.)
      CDIJ(1:PP%LMMAX, 1:PP%LMMAX, NIP, 1) = CDIJ(1:PP%LMMAX, 1:PP%LMMAX, NIP, 1) - D_TMP/(0._q,1._q)
      CALL CDIJ_FLIP(WDES, LMDIM, CDIJ, .TRUE., .TRUE.)

!     WRITE (*,'("one center magnetic moment*1E6",6E16.7)') REAL(MU,q)*1E6

!test dump DIJ contributions
!      WRITE (*,*) 'NI', NI
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (REAL(D_TMP(L2,L1)),L2=1,MIN(8,PP%LMMAX))
!      ENDDO
!      DO L1=1,PP%LMMAX
!         WRITE(*,'(16(F7.3,1X))') (-AIMAG(D_TMP(L2,L1)*1000),L2=1,MIN(8,PP%LMMAX))
!      ENDDO
!      WRITE(*,*)



      DEALLOCATE(WTMP,DWAE,DWPS,TMP)
      DEALLOCATE(JPAWRAD,D_TMP,YLM_NABLA_YLM,YLM_X_YLM)

! ---------------

      ENDDO ions

      RETURN

    END SUBROUTINE SET_DD_BFCONST


!****************************** CALC_MU_LQ *****************************************
!
! calculate the paramagnetic contribution 
!  (termed L_R Q_R)
!
!       Sum_ij \int \phi_i L  \phi_j  \rho_ij d Omega_r
!       Sum_ij \int \phi_i (r-R) x nabla  \phi_j  \rho_ij d Omega_r
!
! turns out this is identical to the paramagnetic contribution from
! CALC_MU_MAGATOM but much more elegantly coded here (mM)
!
!***********************************************************************************

    SUBROUTINE CALC_MU_LQ(PP,CRHODE,MU_LQ_ONE_CTR)
      USE prec
      USE pseudo
      USE radial
      USE constant
      USE relativistic
      IMPLICIT NONE
      TYPE (potcar), POINTER :: PP
      OVERLAP :: CRHODE(:,:)
      REAL(q) :: MU_LQ_ONE_CTR(3)
      ! local variables
      INTEGER, PARAMETER :: LMAX=3, MMAX=LMAX*2+1
      COMPLEX(q) L_OP_R(MMAX,MMAX,3,LMAX) ! fmv in SETUP_LS L_OP_R is defined by L_OP_R(2*L+1,2*L+1,3) 3 dimensions here 4
      COMPLEX(q) DUMMY(MMAX,MMAX,4) ! fmv I change the dimension of the dummy array to foollow the original definition in SETUP_LS SUBROUTINE
      COMPLEX(q), ALLOCATABLE :: CLQIJ(:,:,:)
      INTEGER CH1,CH2,LM,LMP,LL,LLP,M,MP,L,MAXIMUM_L,LDIMP

!**********************************************************************
!
! determine the maximum L quantum number (LDIMP)
!
!**********************************************************************
! extracted from LDIM_PSEUDO(LORBIT, NTYPD, P, LDIMP, LMDIMP)

      MAXIMUM_L=MAXL1(PP)  ! nabla Y_lm has components up to LMAX+1

      ! if the maximum L is larger than 2 we have to set LDIMP

      IF ( MAXIMUM_L > 2 ) THEN
         LDIMP= MAXIMUM_L+1
      ELSE
         LDIMP=3
      ENDIF

      L_OP_R=(0._q,0._q)
      DO L=1,LDIMP
         CALL SETUP_LS(L,0._q,0._q,L_OP_R(1:2*L+1,1:2*L+1,1:3,L),DUMMY(1:2*L+1,1:2*L+1,1:4))
      ENDDO


      ALLOCATE(CLQIJ(PP%LMMAX,PP%LMMAX,3))
      CLQIJ=0

      LM=1
      DO CH1=1,PP%LMAX
      LMP=1
      DO CH2=1,PP%LMAX
         LL =PP%LPS(CH1)
         LLP=PP%LPS(CH2)
         IF (LL == LLP .AND. LL>0 .AND. LL<=LMAX ) THEN
            DO M =1,2*LL+1
            DO MP=1,2*LLP+1
               CLQIJ(LMP+MP-1,LM+M-1,:)=CLQIJ(LMP+MP-1,LM+M-1,:)+L_OP_R(M,MP,:,LL)*PP%QION(CH1,CH2)
            ENDDO
            ENDDO
         ENDIF
      LMP=LMP+(2*LLP+1)
      ENDDO
      LM=LM+(2*LL+1)
      ENDDO

      DO CH1=1,PP%LMMAX
      DO CH2=1,PP%LMMAX
#ifdef realmode
         MU_LQ_ONE_CTR(:)=MU_LQ_ONE_CTR(:)-CLQIJ(CH1,CH2,:)*CRHODE(CH1,CH2)
#else
         MU_LQ_ONE_CTR(:)=MU_LQ_ONE_CTR(:)-CLQIJ(CH1,CH2,:)*GCONJG(CRHODE(CH1,CH2))
#endif
      ENDDO
      ENDDO

      DEALLOCATE(CLQIJ)

!     MU_LQ_ONE_CTR=-MU_LQ_ONE_CTR
      
      RETURN
    END SUBROUTINE CALC_MU_LQ


!****************************** CALC_MU_MAGATOM ************************************
!
! To calculate one-centre magnetic moment on atom where external moment is placed.
! First paramagnetic and diamagnetic current "moments" are calculated:
!
! COMMENT: I changed von p -> -i nabla
! COMMENT: changed von A to M x r / r^3
! para: J_LM^alpha(r) = - Aimag
!       Sum_ij \int Y_lm(r) \phi_i [ nabla r + r nabla]/2 \phi_j  \rho_ij d Omega_r
! dia:  J_LM^alpha(r) = - Sum_ij \int \phi_i  M x r / r^3 Y_LM \phi_j \rho_ij d Omega_r
!                         * MOMTOMOM
!
! these are collected into one array and the moment is calculated (see explanation
! below)
!
! p = -i \hbar nabla
!
!***********************************************************************************

    SUBROUTINE CALC_MU_MAGATOM(PP,CRHODE,MU_PARA_ONE_CTR,MU_DIA_ONE_CTR,L_DO_AUG)
      USE pseudo
      USE constant
      USE asa
      USE radial                                                                ! ?
      USE wave
      USE paw
      IMPLICIT NONE
     
      TYPE (potcar), POINTER :: PP   ! pseudopotential descriptor
      OVERLAP :: CRHODE(:,:)         ! one center occupancy matrix
      REAL(q) :: MU_PARA_ONE_CTR(3), MU_DIA_ONE_CTR(3)  ! one centre moments
      LOGICAL :: L_DO_AUG

     !local variables
      INTEGER CHANNELS          ! number of channels
      INTEGER NMAX,I,NT
      INTEGER I0,I1
      INTEGER L0,L1,L2,L3
      INTEGER M0,M1,M3
      INTEGER LM0,LM1,LM2,LM3,LM2MAX
      INTEGER ISTART1,IEND1,LMIND1,IC1
      INTEGER LMAXNABLA,LMAX,K,IDP
      INTEGER II1,II2,II3,LL,M
      INTEGER LRHO,LMRHO,MRHO,L2MAX
      REAL(q) :: CGOR
      REAL(q) :: SUM
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:),YLM_MxR_YLM(:,:,:)
!     REAL(q), ALLOCATABLE :: JPAWRAD(:,:,:)
      COMPLEX(q), ALLOCATABLE :: JPAWRAD(:,:,:)
      OVERLAP :: JLM(256,3),J0
      OVERLAP :: OSUM
      OVERLAP, ALLOCATABLE :: OTMP(:)
      COMPLEX(q) :: MU(3)

! allocate
      IF (.NOT. ORBITALMAG .OR. .NOT. ASSOCIATED(PP%JPAW)) RETURN

      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX),OTMP(NMAX))

      CHANNELS = PP%LMAX
      LMAX=MAXL1(PP)
      LMAXNABLA=MAXL1(PP)+1   ! nabla Y_lm has components up to LMAX+1
      ALLOCATE(JPAWRAD(PP%R%NMAX,(LMAXNABLA+LMAX+1)*(LMAXNABLA+LMAX+1),3))        ! be ware

! ALLOCATE and setup nabla array
!GIL  ALLOCATE(YLM_NABLA_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))
!GIL  ALLOCATE(YLM_X_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))
!GIL  ALLOCATE(YLM_MxR_YLM((LMAXNABLA+1)*(LMAXNABLA+1),(LMAXNABLA+1)*(LMAXNABLA+1),0:3))

      ALLOCATE(YLM_NABLA_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAXNABLA+2)*(2*LMAXNABLA+2),0:3))
      ALLOCATE(YLM_X_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))
      ALLOCATE(YLM_MxR_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q
      YLM_MxR_YLM=0._q

!GIL  CALL SETYLM_NABLA_YLM(LMAXNABLA,YLM_NABLA_YLM,YLM_X_YLM)
! in principle we should go up to LMAX+LMAXNABLA+1 = 2 LMAX + 2
! see below, but that does not work for f electrons
      CALL SETYLM_NABLA_YLM(2*LMAX+1,YLM_NABLA_YLM,YLM_X_YLM)

      YLM_MxR_YLM(:,:,1)=MAGDIPOL(2)*YLM_X_YLM(:,:,3)-MAGDIPOL(3)*YLM_X_YLM(:,:,2)
      YLM_MxR_YLM(:,:,2)=MAGDIPOL(3)*YLM_X_YLM(:,:,1)-MAGDIPOL(1)*YLM_X_YLM(:,:,3)
      YLM_MxR_YLM(:,:,3)=MAGDIPOL(1)*YLM_X_YLM(:,:,2)-MAGDIPOL(2)*YLM_X_YLM(:,:,1)
 
      JPAWRAD=0._q   

      DO I=1,3
         CALL CALC_RHOLM_JVEC_NOMONO(  CRHODE, JLM(:,I), PP, I)
      ENDDO

 diapara: DO IDP=1,2

! if IDP.eq.1 paramagnetic contribution
! if IDP.eq.2 diamagnetic contribution

!-----------------------------------------------------------------------
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------
      LM2MAX=0
      L2MAX=0

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

         DO M1=1,2*L1+1
            LM3=1
            DO L3=0,LMAXNABLA

            CALL YLM3LOOKUP(L0,L3,LMIND1)

            DO M0=1,2*L0+1
            DO M3=1,2*L3+1

               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               DO IC1=ISTART1,IEND1-1

                  CGOR=YLM3(IC1)
                  LM2 =JS(IC1)
                  L2  =JL(IC1)
                  IF (LM2 <=0 .OR. LM2 > SIZE(JPAWRAD,2)) THEN
                     WRITE(0,*) 'internal error in CALC_MU_MAGATOM: index 2 into JPAWRAD exceeds bounds'
                     STOP
                  ENDIF
                  LM2MAX=MAX(LM2, LM2MAX)
                  L2MAX=MAX(L2, L2MAX)

                  DO I=1,3

!-----------------------------------------------------------------------
! paramagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.1) THEN
                     IF (ABS(YLM_X_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                        TMP=0._q
                        SUM=0._q
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * ( &
        &   PP%WAE(K,I0) * ( DWAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WAE(K,I1) )  &
        & -(PP%WPS(K,I0) * ( DWPS(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WPS(K,I1)))  &
        &                  )
                        ENDDO

                       !   sum contains now
                       !   phi^AE_0(r) r nabla phi^AE_1(r) -phi^PS_0(r) r nabla phi^PS_1(r)
                        DO K=1,PP%R%NMAX
! gK: changed this seems more concise
!                       ! multipy by -i and account for negative charge
!                          JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)-TMP(K)*AIMAG(1.0_q,0.0_q)* &
!                                ((CHGCONJG(CRHODE(LM1+M1-1,LM0+M0-1))+CRHODE(LM0+M0-1,LM1+M1-1))/2))
                        ! multipy by -i and account for negative charge
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*(0.0_q,1.0_q)* &
                                 ((CHGCONJG(CRHODE(LM1+M1-1,LM0+M0-1))+CRHODE(LM0+M0-1,LM1+M1-1))/2)
                        ENDDO
                     ENDIF
                     ENDIF
!-----------------------------------------------------------------------
! diamagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.2) THEN
                     IF (ABS(YLM_MxR_YLM(LM3+M3-1,L1*L1+M1,I))>1E-10) THEN
                        TMP=0._q
                        SUM=0._q
                        !  1/|r|^3 * |r| from Y*x/y/z*Y
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR* YLM_MxR_YLM(LM3+M3-1,L1*L1+M1,I) / PP%R%R(K)**2 * ( &
                                    PP%WAE(K,I0) * PP%WAE(K,I1)  &
                                 -  PP%WPS(K,I0) * PP%WPS(K,I1)  &
                            )
                        ENDDO
!                       WRITE (*,'(3I3,2f16.10)') I1,I0,L2,PP%QPAW(I0,I1,L2)

                        DO K=1,PP%R%NMAX
! gK: changed this: PROBLEM order makes a slight difference for CH4, C2H2 ok
!                          JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM1+M1-1,LM0+M0-1)
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM0+M0-1,LM1+M1-1)
                        ENDDO
                     ENDIF
                     ENDIF
                  ENDDO  ! I

               ENDDO  ! IC1

            ENDDO ! M3
            ENDDO ! M0

            LM3=LM3+2*L3+1
            ENDDO ! L3

         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

      IF (L_DO_AUG) THEN
      IF (IDP==1) THEN
!-----------------------------------------------------------------------
! paramagnetic current augmentation contribution (matches
!  CURRENT_AUGMENTATION in us.F)
!-----------------------------------------------------------------------
#ifdef current_augmentation
      LMRHO=1
      DO LRHO=0,2*LMAX
         DO MRHO=1,2*LRHO+1
            DO I=1,3
               DO K=1,PP%R%NMAX
                  JPAWRAD(K,LMRHO+MRHO-1,I)=JPAWRAD(K,LMRHO+MRHO-1,I) +  & ! -1 (as above) x -1 for augmentation correction
                       PP%AUG(K,LRHO)*AIMAG((1.0_q,0.0_q)*JLM(LMRHO+MRHO-1,I))
               ENDDO
            ENDDO
         ENDDO
         LMRHO=LMRHO+2*LRHO+1
      ENDDO
#endif
      ENDIF
      ENDIF

! calculate moment
! \int r \times J_LM^alpha Y_LM dV
! x Y_LM = sqrt(4pi) Y_00 x Y_LM = sqrt(4pi) |r|/|r| (Y_00 x Y_LM) = sqrt(4pi) |r|/|r| YLM_X_YLM(LM=0,LM,x)

      MU=0._q

      DO LM2=1,LM2MAX
         DO I=1,3
            ! PROBLEM: please put in some guard against going over bounds
            DO K=1,PP%R%NMAX
               IF (I.EQ.1) OTMP(K)=(YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,3)-YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,2)) *PP%R%R(K)
               IF (I.EQ.2) OTMP(K)=(YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,1)-YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,3)) *PP%R%R(K)
               IF (I.EQ.3) OTMP(K)=(YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,2)-YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,1)) *PP%R%R(K)
            ENDDO

            OSUM=0._q
            DO K=1,PP%R%NMAX
               OSUM=OSUM+OTMP(K)*PP%R%SI(K)
            ENDDO
            MU(I)=MU(I)+OSUM *SQRT(4._q*PI)  !normalisation Y00

         ENDDO
      ENDDO

      WRITE (*,'("one center magnetic moment*1E6",6E16.7)') MU*1E6

      IF (IDP.EQ.1) THEN
         MU_PARA_ONE_CTR(1)=REAL(MU(1),q)
         MU_PARA_ONE_CTR(2)=REAL(MU(2),q)
         MU_PARA_ONE_CTR(3)=REAL(MU(3),q)
      ENDIF
      IF (IDP.EQ.2) THEN
         MU_DIA_ONE_CTR(1)=REAL(MU(1),q)
         MU_DIA_ONE_CTR(2)=REAL(MU(2),q)
         MU_DIA_ONE_CTR(3)=REAL(MU(3),q)
      ENDIF

! calculate dia and paramagnetic contr. seperately
        JPAWRAD=0

      ENDDO diapara


      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM, YLM_MxR_YLM)
      DEALLOCATE(WTMP,DWAE,DWPS,TMP,OTMP)
      DEALLOCATE(JPAWRAD)

    END SUBROUTINE CALC_MU_MAGATOM


!******************************* PS_BFIELD *****************************************
!
! Calculates the magnetic field at the atomic positions from J on the large grid
!
!***********************************************************************************

    SUBROUTINE PS_BFIELD(JTOT,GRIDC,LATT_CUR, T_INFO, B_OUT, B_NICS_OUT, IU0, IU6)

      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE poscar
      USE fileio

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR
      TYPE (type_info):: T_INFO        ! type information
      INTEGER            IU0, IU6
      LOGICAL, SAVE :: LNICSSTART=.TRUE.

      COMPLEX(q) JTOT(GRIDC%MPLWV,3)           !ugly
      COMPLEX(q),ALLOCATABLE ::   BNICS(:,:) 
      REAL(q)    B_OUT(3,T_INFO%NIONS)
      REAL(q)    B_NICS_OUT(3,T_INFO%NIONS)

      INTEGER     ::  N, NC, N1, N2, N3, NI, I, J, MAXI
      REAL(q)     ::  G(3), G2, ENERGY
      COMPLEX(q)  ::  CPHASE, BSUM(3), TMP(3), TMPJ(3)
      REAL(q)     ::  JMAX,AA
      INTEGER I1, I2, I3 

      REAL(q)     ::  POSX, POSY, POSZ

      IF ((.NOT.ORBITALMAG).AND.(.NOT.LCHIMAG)) RETURN

! bring JTOT to reciprocal space

      DO I=1,3
         CALL RL_ADD(JTOT(1,I),1._q/GRIDC%NPLWV,JTOT(1,I),0.0_q,JTOT(1,I),GRIDC)
         CALL FFT3D(JTOT(1,I),GRIDC,-1)
      ENDDO


      ions: DO NI=1,T_INFO%NIONS
      MAXI=0

! Biot-Savart
      BSUM=0
      N=0
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

! (10.3) Georg thesis: backtransform: C_r = sum_G exp(iGr)
        CPHASE=EXP( CITPI*(T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)+T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)+ &
                           T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)))
!       CPHASE=EXP( CITPI*(AA*GRIDC%LPCTX(N1)))
        MAXI=MAX(MAXI,GRIDC%LPCTX(N1))
       
!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G^2 exp(-i G . r_s)
!=======================================================================
        ENERGY=HSQDTM*G2

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)
           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy
           BSUM(1)=BSUM(1)+TMP(1)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BSUM(2)=BSUM(2)+TMP(2)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BSUM(3)=BSUM(3)+TMP(3)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
!         IF (ENERGY>ENCUT_AVEC) AVTOT(N,:)=0
        ENDIF
      ENDDO row
      ENDDO col

      CALLMPI( M_sum_d(GRIDC%COMM, BSUM, 6))   !BSUM is complex


      !IF (IU0>=0) WRITE (IU0,'(I4,6E16.7)') NI, BSUM*1E6  *2 ! empirical programming
      B_OUT(1,NI)=REAL(BSUM(1),q)*2._q
      B_OUT(2,NI)=REAL(BSUM(2),q)*2._q
      B_OUT(3,NI)=REAL(BSUM(3),q)*2._q
      ENDDO ions

!=======================================================================
! start the NICS
!=======================================================================

      IF (NUCIND) THEN
        IF (LNICSALL) THEN   
        ALLOCATE(BNICS(GRIDC%MPLWV,3))

         BNICS=(0._q,0._q)


! Biot-Savart
      N=0
      colb: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      rowb: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G^2 exp(-i G . r_s)
!=======================================================================

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)
           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy
           BNICS(N,1)=TMP(1)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BNICS(N,2)=TMP(2)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BNICS(N,3)=TMP(3)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
        ENDIF
      ENDDO rowb
      ENDDO colb

      BNICS=BNICS*2._q*1E6

      IF (LNICSSTART) THEN
         OPEN (UNIT=97,FILE="NICSCAR",STATUS="UNKNOWN")
         LNICSSTART = .FALSE.
      ELSE
         OPEN (UNIT=97,FILE="NICSCAR",STATUS="UNKNOWN",POSITION="APPEND")
      ENDIF

      DO I=1,3
         CALL OUTCHG(GRIDC,97,.TRUE.,BNICS(1,I))
      ENDDO

      CLOSE (97)

      DEALLOCATE(BNICS)

     ELSE  !  LNICSALL=.FALSE.
      nics: DO NI=1,NPOS_NICS
      POSX = TAUNICS(NI,1)
      POSY = TAUNICS(NI,2)
      POSZ = TAUNICS(NI,3)
      MAXI=0

! Biot-Savart
      BSUM=0
      N=0
      colc: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      rowc: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

! (10.3) Georg thesis: backtransform: C_r = sum_G exp(iGr)
        CPHASE=EXP(CITPI*(POSZ*GRIDC%LPCTZ(N3)+POSY*GRIDC%LPCTY(N2)+POSX*GRIDC%LPCTX(N1)))
        MAXI=MAX(MAXI,GRIDC%LPCTX(N1))

!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G2 exp(-i G . r_s)
!=======================================================================
        ENERGY=HSQDTM*G2

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)
           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy
           BSUM(1)=BSUM(1)+TMP(1)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA
!*MAGMOMTOENERGY
           BSUM(2)=BSUM(2)+TMP(2)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA
!*MAGMOMTOENERGY
           BSUM(3)=BSUM(3)+TMP(3)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA
!*MAGMOMTOENERGY
        ENDIF
      ENDDO rowc
      ENDDO colc

      CALLMPI( M_sum_d(GRIDC%COMM, BSUM, 6))   !BSUM is complex

!      IF (IU0>=0) WRITE (IU0,'(I4,6E16.7)') NI, BSUM*1E6  *2 ! empirical programming
      B_NICS_OUT(1,NI)=REAL(BSUM(1),q)*2._q*1E6
      B_NICS_OUT(2,NI)=REAL(BSUM(2),q)*2._q*1E6
      B_NICS_OUT(3,NI)=REAL(BSUM(3),q)*2._q*1E6


      ENDDO nics

      ENDIF ! LNICSALL IF


      ENDIF  ! NUCIND IF
!=======================================================================
! the NICS finished
!=======================================================================

! restore JTOT on in real space
      DO I=1,3
         CALL FFT3D(JTOT(1,I),GRIDC,1)
      ENDDO

   END SUBROUTINE PS_BFIELD


!******************************* PS_BFIELD2 ****************************************
!
! Calculates the magnetic field at the atomic positions from J on the large grid
!
!***********************************************************************************

    SUBROUTINE PS_BFIELD_C(JTOT,GRIDC,LATT_CUR,T_INFO,NPOSROT,POSROT,B_OUT,B_NICS_OUT,IU0,IU6)

      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE poscar
      USE fileio

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR
      TYPE (type_info):: T_INFO        ! type information
      INTEGER            NPOSROT
      INTEGER            IU0, IU6
      LOGICAL, SAVE :: LNICSSTART=.TRUE.

      COMPLEX(q) JTOT(GRIDC%MPLWV,3)
      COMPLEX(q),ALLOCATABLE :: BNICS(:,:) 
      COMPLEX(q) B_OUT(3,T_INFO%NIONS)
      REAL(q)    B_NICS_OUT(3,NPOSROT)
      REAL(q)    POSROT(NPOSROT,3)

      INTEGER     ::  N, NC, N1, N2, N3, NI, I, J, MAXI
      REAL(q)     ::  G(3), G2, ENERGY
      COMPLEX(q)  ::  CPHASE, BSUM(3), TMP(3), TMPJ(3)
      REAL(q)     ::  JMAX,AA
      INTEGER I1, I2, I3 

      REAL(q)     ::  POSX, POSY, POSZ
      REAL(q)     ::  ADEG

      IF ((.NOT.ORBITALMAG).AND.(.NOT.LCHIMAG)) RETURN

! bring JTOT to reciprocal space

      DO I=1,3
         CALL RL_ADD(JTOT(1,I),1._q/GRIDC%NPLWV,JTOT(1,I),0.0_q,JTOT(1,I),GRIDC)
         CALL FFT3D(JTOT(1,I),GRIDC,-1)
      ENDDO

      ions: DO NI=1,T_INFO%NIONS
      MAXI=0

! Biot-Savart
      BSUM=0
      N=0
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

! (10.3) Georg thesis: backtransform: C_r = sum_G exp(iGr)
        CPHASE=EXP( CITPI*(T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)+T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)+ &
                           T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)))
!       CPHASE=EXP( CITPI*(AA*GRIDC%LPCTX(N1)))
        MAXI=MAX(MAXI,GRIDC%LPCTX(N1))
       
!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G^2 exp(-i G . r_s)
!=======================================================================
        ENERGY=HSQDTM*G2

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)
           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy

           ADEG=1._q
#if defined(NGZhalf)
           IF (GRIDC%LPCTZ(N3)/=0) ADEG=2._q
#endif
#if defined(NGXhalf)
           IF (GRIDC%LPCTX(N1)/=0) ADEG=2._q
#endif

           BSUM(1)=BSUM(1)+TMP(1)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA *ADEG !*MAGMOMTOENERGY
           BSUM(2)=BSUM(2)+TMP(2)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA *ADEG !*MAGMOMTOENERGY
           BSUM(3)=BSUM(3)+TMP(3)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA *ADEG !*MAGMOMTOENERGY
!         IF (ENERGY>ENCUT_AVEC) AVTOT(N,:)=0
        ENDIF
      ENDDO row
      ENDDO col

      CALLMPI( M_sum_d(GRIDC%COMM, BSUM, 6))   !BSUM is complex


      !IF (IU0>=0) WRITE (IU0,'(I4,6E16.7)') NI, BSUM*1E6  *2 ! empirical programming
      B_OUT(1,NI)=BSUM(1)*2._q
      B_OUT(2,NI)=BSUM(2)*2._q
      B_OUT(3,NI)=BSUM(3)*2._q
      ENDDO ions

!=======================================================================
! start the NICS
!=======================================================================

      IF (NUCIND) THEN
        IF (LNICSALL) THEN   
        ALLOCATE(BNICS(GRIDC%MPLWV,3))

         BNICS=(0._q,0._q)


! Biot-Savart
      N=0
      colb: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      rowb: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G^2 exp(-i G . r_s)
!=======================================================================

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)
           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy
           BNICS(N,1)=TMP(1)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BNICS(N,2)=TMP(2)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
           BNICS(N,3)=TMP(3)/MAX(G2,1E-5_q)       *2*CITPI/LATT_CUR%OMEGA !*MAGMOMTOENERGY
        ENDIF
      ENDDO rowb
      ENDDO colb

      BNICS=BNICS*2._q*1E6

      IF (LNICSSTART) THEN
         OPEN (UNIT=97,FILE="NICSCAR",STATUS="UNKNOWN")
         LNICSSTART = .FALSE.
      ELSE
         OPEN (UNIT=97,FILE="NICSCAR",STATUS="UNKNOWN",POSITION="APPEND")
      ENDIF

      DO I=1,3
         CALL OUTCHG(GRIDC,97,.TRUE.,BNICS(1,I))
      ENDDO

      CLOSE (97)

      DEALLOCATE(BNICS)

     ELSE  !  LNICSALL=.FALSE.

      nics: DO NI=1,NPOSROT
      POSX = POSROT(NI,1)
      POSY = POSROT(NI,2)
      POSZ = POSROT(NI,3)
      MAXI=0

! Biot-Savart
      BSUM=0
      N=0
      colc: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      rowc: DO N1=1,GRIDC%RC%NROW
        N=N+1
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        G(1)=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
        G(2)=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
        G(3)=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

        G2=G(1)**2+G(2)**2+G(3)**2

! (10.3) Georg thesis: backtransform: C_r = sum_G exp(iGr)
        CPHASE=EXP(CITPI*(POSZ*GRIDC%LPCTZ(N3)+POSY*GRIDC%LPCTY(N2)+POSX*GRIDC%LPCTX(N1)))
        MAXI=MAX(MAXI,GRIDC%LPCTX(N1))

!=======================================================================
! Equ. (61) of PRB 63 245101 2001
! 4 pi i / Omega G x J / G2 exp(-i G . r_s)
!=======================================================================
        ENERGY=HSQDTM*G2

        IF ( ( (GRIDC%LPCTX(N1)/=0) .OR. (GRIDC%LPCTY(N2)/=0) .OR. &
     &         (GRIDC%LPCTZ(N3)/=0))) THEN
           ! copy EXPRO for performance reasons from lattice.F
           TMPJ(1)=JTOT(N,1)
           TMPJ(2)=JTOT(N,2)
           TMPJ(3)=JTOT(N,3)
           CALL EXPRO_CRC(TMP, G, TMPJ)

           ADEG=1._q
#if defined(NGZhalf)
           IF (GRIDC%LPCTZ(N3)/=0) ADEG=2._q
#endif
#if defined(NGXhalf)
           IF (GRIDC%LPCTX(N1)/=0) ADEG=2._q
#endif

           ! convert from magnetic moment (supplied in multiples of Bohr
           ! magneton) to energy
           BSUM(1)=BSUM(1)+TMP(1)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA*ADEG
!*MAGMOMTOENERGY
           BSUM(2)=BSUM(2)+TMP(2)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA*ADEG
!*MAGMOMTOENERGY
           BSUM(3)=BSUM(3)+TMP(3)/MAX(G2,1E-5_q)*CPHASE*2*CITPI/LATT_CUR%OMEGA*ADEG
!*MAGMOMTOENERGY
        ENDIF
      ENDDO rowc
      ENDDO colc

      CALLMPI( M_sum_d(GRIDC%COMM, BSUM, 6))   !BSUM is complex

!      IF (IU0>=0) WRITE (IU0,'(I4,6E16.7)') NI, BSUM*1E6  *2 ! empirical programming
      B_NICS_OUT(1,NI)=REAL(BSUM(1),q)*2._q*1E6
      B_NICS_OUT(2,NI)=REAL(BSUM(2),q)*2._q*1E6
      B_NICS_OUT(3,NI)=REAL(BSUM(3),q)*2._q*1E6

      ENDDO nics


      ENDIF ! LNICSALL IF

      ENDIF  ! NUCIND IF
!=======================================================================
! the NICS finished
!=======================================================================

! restore JTOT on in real space
      DO I=1,3
         CALL FFT3D(JTOT(1,I),GRIDC,1)
      ENDDO

   END SUBROUTINE PS_BFIELD_C

!****************************** CALC_B_MAGATOM *************************************
!
! To calculate one-centre contribution to magnetic field at atomic nuclei.
! First paramagnetic and diamagnetic "currents" are calculated:
!
! COMMENT: change from p -> -i nabla
! COMMENT: and  A to M x r / r^3
! para: J_LM^alpha(r) =   Sum_ij \int \phi_i ((-i nabla)^* + -i nabla )/2 Y_LM \phi_j dOmega \rho_ji
! dia:  J_LM^alpha(r) = - Sum_ij \int \phi_i  M x r / r^3   Y_LM \phi_j dOmega \rho_ji
!                         * MOMTOMOM           
!
! These are collected into one array
!
! Next the augmentation currents are subracted
!
! Finally Biot-Savart is applied to obtain the field:
!
!   B = - r x J /|r^3|
!
! A=(Bxr)/2*MAGMOMTOENERGY
!
! This routine is similar to and derived from CALC_MU_MAGATOM
! It is different though, as the augmentation current has to be subtracted in
! a "different" place.
!
!***********************************************************************************

    SUBROUTINE CALC_B_MAGATOM(PP,CRHODE,NIP,NI,B_OUT_PARA_ONE_CENTR,B_OUT_DIA_ONE_CENTR)
      USE pseudo
      USE constant
      USE asa
      USE radial                                                                ! ?
      USE wave
      USE paw
      USE poscar
      IMPLICIT NONE
     
      TYPE (potcar), POINTER :: PP   ! pseudopotential descriptor
      OVERLAP :: CRHODE(:,:)         ! one center occupancy matrix
      REAL(q) :: B_OUT_PARA_ONE_CENTR(3), B_OUT_DIA_ONE_CENTR(3)
      INTEGER :: NIP,NI

     !local variables
      INTEGER CHANNELS          ! number of channels
      INTEGER NMAX,I,NT
      INTEGER I0,I1
      INTEGER L0,L1,L2,L3
      INTEGER M0,M1,M3
      INTEGER LM0,LM1,LM2,LM3,LM2MAX
      INTEGER ISTART1,IEND1,LMIND1,IC1
      INTEGER LMAXNABLA,LMAX,K,IDP
      INTEGER II1,II2,II3,LL,M
      INTEGER LRHO,LMRHO,MRHO,L2MAX
      REAL(q) :: CGOR
      REAL(q) :: SUM
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:),YLM_BxR_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: JPAWRAD(:,:,:)
      OVERLAP :: JLM(256,3)
      OVERLAP :: OSUM
      REAL(q) :: OSUM1,OSUM2
      OVERLAP, ALLOCATABLE :: OTMP(:)
      COMPLEX(q) :: BF(3)

! allocate
      IF (.NOT. ORBITALMAG .OR. .NOT. ASSOCIATED(PP%JPAW)) RETURN

      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX),OTMP(NMAX))

      CHANNELS = PP%LMAX
      LMAX=MAXL1(PP)
      LMAXNABLA=MAXL1(PP)+1   ! nabla Y_lm has components up to LMAX+1
      ALLOCATE(JPAWRAD(PP%R%NMAX,(LMAXNABLA+LMAX+1)*(LMAXNABLA+LMAX+1),3))        ! be ware

      ALLOCATE(YLM_NABLA_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAXNABLA+2)*(2*LMAXNABLA+2),0:3))
      ALLOCATE(YLM_X_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))
      ALLOCATE(YLM_BxR_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q
      YLM_BxR_YLM=0._q

!GIL  CALL SETYLM_NABLA_YLM(LMAXNABLA,YLM_NABLA_YLM,YLM_X_YLM)
! in principle we should go up to LMAX+LMAXNABLA+1 = 2 LMAX + 2
! see below, but that does not work for f electrons
      CALL SETYLM_NABLA_YLM(2*LMAX+1,YLM_NABLA_YLM,YLM_X_YLM)

      YLM_BxR_YLM(:,:,1)=BCONST(2)*YLM_X_YLM(:,:,3)-BCONST(3)*YLM_X_YLM(:,:,2)
      YLM_BxR_YLM(:,:,2)=BCONST(3)*YLM_X_YLM(:,:,1)-BCONST(1)*YLM_X_YLM(:,:,3)
      YLM_BxR_YLM(:,:,3)=BCONST(1)*YLM_X_YLM(:,:,2)-BCONST(2)*YLM_X_YLM(:,:,1)
      YLM_BxR_YLM(:,:,:)=YLM_BxR_YLM(:,:,:)/2. !*MAGMOMTOENERGY
 
      JPAWRAD=0._q   
      B_OUT_PARA_ONE_CENTR=0._q
      B_OUT_DIA_ONE_CENTR=0._q

      DO I=1,3
         CALL CALC_RHOLM_JVEC(  CRHODE, JLM(:,I), PP, I)
      ENDDO

 diapara: DO IDP=1,2

! if IDP.eq.1 paramagnetic contribution
! if IDP.eq.2 diamagnetic contribution

!-----------------------------------------------------------------------
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------
      LM2MAX=0
      L2MAX=0

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

         DO M1=1,2*L1+1

            LM3=1
            DO L3=0,LMAXNABLA

            CALL YLM3LOOKUP(L0,L3,LMIND1)

            DO M0=1,2*L0+1
            DO M3=1,2*L3+1

               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               DO IC1=ISTART1,IEND1-1

                  CGOR=YLM3(IC1)
                  LM2 =JS(IC1)
                  L2  =JL(IC1)
                  IF (LM2 <=0 .OR. LM2 > SIZE(JPAWRAD,2)) THEN
                     WRITE(0,*) 'internal error in CALC_B_MAGATOM: index 2 into JPAWRAD exceeds bounds'
                     STOP
                  ENDIF
                  LM2MAX=MAX(LM2, LM2MAX)
                  L2MAX=MAX(L2, L2MAX)

                  DO I=1,3

!-----------------------------------------------------------------------
! paramagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.1) THEN
                     IF (ABS(YLM_X_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                        TMP=0._q
                        SUM=0._q
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * ( &
        &   PP%WAE(K,I0) * ( DWAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WAE(K,I1) )  &
        & -(PP%WPS(K,I0) * ( DWPS(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                             &
        &                        + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) * PP%WPS(K,I1)))  &
        &                  )
                        ENDDO

!                       IF (LM0+M0-1 <=0 .OR. LM0+M0-1 > SIZE(PP%JPAW,1)) THEN
!                          WRITE(0,*) 'internal error in CALC_JPAW: index 1 into JPAR exceeds bounds'
!                          STOP
!                       ENDIF
!                       IF (LM1+M1-1 <=0 .OR. LM1+M1-1 > SIZE(PP%JPAW,2)) THEN
!                          WRITE(0,*) 'internal error in CALC_JPAW: index 2 into JPAR exceeds bounds'
!                          STOP
!                       ENDIF
!                       PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) = PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) + SUM

                        !multipy by -i and make symmetric (!)
                        DO K=1,PP%R%NMAX
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)-AIMAG((1.0_q,0.0_q)* &
!                                 TMP(K)*(CRHODE(LM1+M1-1,LM0+M0-1)+CHGCONJG(CRHODE(LM0+M0-1,LM1+M1-1)))/2)
                                 (TMP(K)*(CHGCONJG(CRHODE(LM1+M1-1,LM0+M0-1))+CRHODE(LM0+M0-1,LM1+M1-1)))/2)
                        ENDDO
                     ENDIF
                     ENDIF
!-----------------------------------------------------------------------
! diamagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.2) THEN
                     IF (ABS(YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                        TMP=0._q
                        SUM=0._q
!                       |r| from Y*x/y/z*Y
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * PP%R%R(K) * ( &
                                    PP%WAE(K,I0) * PP%WAE(K,I1) *( YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I) )  &
                                 -  PP%WPS(K,I0) * PP%WPS(K,I1) *( YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I) )  &
                            )
                        ENDDO
!                       WRITE (*,'(3I3,2f16.10)') I1,I0,L2,PP%QPAW(I0,I1,L2)

                        DO K=1,PP%R%NMAX
!gK change
!                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM1+M1-1,LM0+M0-1)
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM0+M0-1,LM1+M1-1)
                        ENDDO
                     ENDIF
                     ENDIF
                  ENDDO  ! I

               ENDDO  ! IC1

            ENDDO ! M3
            ENDDO ! M0

            LM3=LM3+2*L3+1
            ENDDO ! L3

         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

      IF (IDP==1) THEN
!-----------------------------------------------------------------------
! paramagnetic current augmentation contribution (matches
!  CURRENT_AUGMENTATION in us.F)
! i.e. depends on whether 
!   j_para = e hbar/m_e sum_n Im(psi nabla psi)
! is augmented with augmentation currents or not
!-----------------------------------------------------------------------
#ifdef current_augmentation
      LMRHO=1
      DO LRHO=0,2*LMAX
         DO MRHO=1,2*LRHO+1
            DO I=1,3
               DO K=1,PP%R%NMAX
                  JPAWRAD(K,LMRHO+MRHO-1,I)=JPAWRAD(K,LMRHO+MRHO-1,I) +  & ! -1 as above -1 for augmentation correction
                       PP%AUG(K,LRHO)*AIMAG((1.0_q,0.0_q)*JLM(LMRHO+MRHO-1,I)) ! R^2 in PP%AUG ?
               ENDDO
            ENDDO
         ENDDO
         LMRHO=LMRHO+2*LRHO+1
      ENDDO
#endif
      ENDIF

! calculate moment
! \int r \times J_LM^alpha Y_LM dV
! x Y_LM = sqrt(4pi) Y_00 x Y_LM = sqrt(4pi) |r|/|r| (Y_00 x Y_LM) = sqrt(4pi) |r|/|r| YLM_X_YLM(LM=0,LM,x)

      BF=0._q

      DO LM2=1,LM2MAX
         DO I=1,3
            ! please put in some guard against going over bounds

            DO K=1,PP%R%NMAX
               IF (I.EQ.1) OTMP(K)=(YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,3)-YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,2)) *PP%R%R(K)
               IF (I.EQ.2) OTMP(K)=(YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,1)-YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,3)) *PP%R%R(K)
               IF (I.EQ.3) OTMP(K)=(YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,2)-YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,1)) *PP%R%R(K)
               OTMP(K)=OTMP(K)/PP%R%R(K)**3
            ENDDO

            OSUM=0._q
            DO K=1,PP%R%NMAX
               OSUM=OSUM+OTMP(K)*PP%R%SI(K)
            ENDDO
            BF(I)=BF(I)+OSUM *SQRT(4._q*PI) *2._q !normalisation Y00 * 2

         ENDDO
      ENDDO

!      WRITE (*,'("one center magnetic field*1E6",I3,6E16.7)') NI,REAL(BF,q)*1E6

! default here: calculate dia and paramagnetic contr. seperately
        JPAWRAD=0

! collect one-centre contributions to field
         IF (IDP.EQ.1) THEN
            B_OUT_PARA_ONE_CENTR(1)=REAL(BF(1),q)
            B_OUT_PARA_ONE_CENTR(2)=REAL(BF(2),q)
            B_OUT_PARA_ONE_CENTR(3)=REAL(BF(3),q)
         ENDIF
         IF (IDP.EQ.2) THEN
            B_OUT_DIA_ONE_CENTR(1)=REAL(BF(1),q)
            B_OUT_DIA_ONE_CENTR(2)=REAL(BF(2),q)
            B_OUT_DIA_ONE_CENTR(3)=REAL(BF(3),q)
         ENDIF

      ENDDO diapara

      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM, YLM_BxR_YLM)
      DEALLOCATE(WTMP,DWAE,DWPS,TMP,OTMP)
      DEALLOCATE(JPAWRAD)

    END SUBROUTINE CALC_B_MAGATOM


!***************************** SUBROUTINE BLOCH_CURRENT   **************************
!
! routine to evaluate total moment using a summation over states
! and the derivative of the wavefunctions with respect to k-points
! (modern theory of orbital polarization)
!
!***********************************************************************************
! #define justatest
    SUBROUTINE BLOCH_CURRENT( W, GRID_SOFT, GRIDC, GRIDUS, C_TO_US, SOFT_TO_C, P, LATT_CUR, &
          AVEC, AVTOT, CHTOT, NONLR_S, NONL_S, RPHI, RPHI_CPROJ, CDIJ, CQIJ, SV, EFERMI, &
          T_INFO, LMDIM, CRHODE, IRDMAX, IU6, IU0)
      USE wave_high
      USE lattice
      USE mgrid
      USE constant
      USE poscar
      USE us
      USE pseudo
      USE paw
      USE nonl_high
      USE hamil

      IMPLICIT NONE

      TYPE (wavespin)    W             ! wavefunction
      TYPE (grid_3d)     GRID_SOFT     ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d)     GRIDC         ! full, fine grid
      TYPE (grid_3d)     GRIDUS        ! doubled grid
      TYPE (transit)     SOFT_TO_C     !
      TYPE (transit)     C_TO_US       !
      TYPE (latt)        LATT_CUR      ! lattice
      TYPE (potcar), TARGET :: P(:)          ! pseudopotential information
      RGRID, POINTER  :: AVEC(:,:)     ! vector potential on course grid
      COMPLEX(q),POINTER ::  AVTOT(:,:)! vector potential on fine grid
      COMPLEX(q) ::       CHTOT(:,:)   ! charge density on fine grid
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct)  NONL_S
      COMPLEX(qs)     :: RPHI(:,:,:,:,:)
      GDEFS           :: RPHI_CPROJ(:,:,:,:,:)
      OVERLAP  CDIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ), &
               CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ), &
               CDIJ_TMP(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NRSPINORS*W%WDES%NRSPINORS)
      RGRID           :: SV(DIMREAL(W%WDES%GRID%MPLWV),W%WDES%NCDIJ)
      REAL(q)         :: EFERMI
      TYPE (type_info):: T_INFO        ! type information
      INTEGER         :: LMDIM         !
      OVERLAP         :: CRHODE(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      INTEGER         :: IRDMAX        ! required allocation
      INTEGER         :: IU6, IU0


    ! local
      INTEGER :: ISP, NK, N, I, J, NP, NI
      TYPE (wavedes1) :: WDES1          ! descriptor for one k-point
      TYPE (wavefuna) :: WH
      TYPE (wavefun1)    W1(0:3)          ! stores d u / d k_i
      TYPE (wavefun1)    W2(0:3)          ! stores d u / d k_i
#ifdef justatest
      TYPE (wavefun1)    WPROJ
#endif
      TYPE (potcar), POINTER :: PP
      COMPLEX(q) :: CMAG_LC(3,3)           ! orbital magnetization
      COMPLEX(q) :: CMAG_LC2(3,3)           ! orbital magnetization
      COMPLEX(q) :: CMAG_IC(3,3)        ! orbital magnetization
      COMPLEX(q) :: CMAG_CHERN(3,3) 
      COMPLEX(q) :: M_LC(3), M_IC(3), DM_BAPA(3) , CHERN(3)
      COMPLEX(q) :: CMAG_BAPA(3,3)        ! Delta M bare and para before vector product
      REAL(q) :: WEIGHT
      REAL(q) :: MU_PARA_ONE_CTR(3), MU_DIA_ONE_CTR(3)
!GIL 24 01 11      COMPLEX(q) :: MU_PARA_MIXED(3), MU_DIA_MIXED(3)
      INTEGER :: NT,NIP,ICART
      INTEGER :: LM0, L0, M0, I0, LM1, L1, M1, I1
      OVERLAP, ALLOCATABLE :: CRHODE1(:,:,:,:)
      COMPLEX(q), ALLOCATABLE :: VIJ_PARA(:,:,:), VIJ_DIA(:,:,:)
      REAL(q), POINTER :: A_ONE_CENTER(:,:,:)
      COMPLEX(q) :: MU_ONE_CTR(3,T_INFO%NIONS),E
      REAL(q)    :: DISPL(3,T_INFO%NIONS)
      INTEGER j1
      TYPE (nonlr_struct), TARGET :: NONLR_D
      TYPE (nonl_struct) NONL_D

#ifdef MPI
      IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('BLOCH_CURRENT: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif
    ! descriptor to k-point 0
      CALL SETWDES(W%WDES,WDES1,0)
    ! create wavefunction array with three entries
      CALL NEWWAVA(WH,  WDES1, 3)
     
      DO N=0,3
         CALL NEWWAV(W1(N), WDES1, .TRUE.)
         CALL NEWWAV(W2(N), WDES1, .TRUE.)
      END DO

#ifdef justatest
      CALL NEWWAV(WPROJ,WDES1,.TRUE.)
#endif

      M_LC=0
      M_IC=0
      DM_BAPA=0

      IF (NONLR_S%LREAL) THEN
!        NONLR_D=NONLR_S
         CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
         CALL NONLR_ALLOC_CRREXP(NONLR_D)
      ELSE
         CALL NONL_ALLOC_DER(NONL_S,NONL_D)
         write(*,*) 'bloch_current2: nonl_alloc_der called'
      ENDIF

!GIL 23 04 11 #ifndef current_augmentation
!GIL 23 01 11      WRITE(*,*) 'CDIJ zero'
!GIL 23 01 11      CDIJ=0
!GIL 23 04 11 #endif

      spin:  DO ISP=1,W%WDES%ISPIN

! we kijken of het werkt
      CDIJ_TMP=0
      IF (W%WDES%ISPIN==2) THEN
         CDIJ_TMP(:,:,:,1)=CDIJ(:,:,:,ISP)
      ELSE
         CDIJ_TMP(:,:,:,:)=CDIJ(:,:,:,:)
      ENDIF

      kpoint: DO NK=1,W%WDES%NKPTS
         CALL SETWDES(W%WDES,WDES1,NK)
         IF (NONLR_S%LREAL) THEN
            CALL PHASER(WDES1%GRID,LATT_CUR,NONLR_S,NK,W%WDES)
         ELSE
            CALL PHASE(W%WDES,NONL_S,NK)
         ENDIF

         DO N=1,W%WDES%NBANDS  ! loop over bands
            W1(0)%CW   =W%CW(:,N,NK,ISP)
            W1(0)%CPROJ=W%CPROJ(:,N,NK,ISP)
            CALL FFTWAV_W1(W1(0))
            W1(0)%LDO=.TRUE.
           
            DO I=1,3  ! loop over cartesian index
               W1(I)%CW   =RPHI(:,N,NK,ISP,I)
               ! store i d/ d k<beta|   | u_nk> = -d/ d (i k)<beta|   | u_nk>
               W1(I)%CPROJ=-RPHI_CPROJ(:,N,NK,ISP,I)
               CALL FFTWAV_W1(W1(I))
               W1(I)%LDO=.TRUE.
#ifdef norm_conserving
               IF (NONLR_S%LREAL ) THEN
                  ! calculate  <beta| d/ d (i k) u_nk>
                  CALL RPRO1(NONLR_S,WDES1,W1(I))                        !  W1(I)%CPROJ = <beta | d/dik u_nk>
#ifdef justatest
                  ! add derivative of projector  d <beta| / d ik  | u_nk>
                  CALL PHASERR(W%WDES%GRID,LATT_CUR,NONLR_D,NK,W%WDES,I) ! NONLR_D = <beta|r
                  WPROJ%CR=W1(0)%CR                                      ! W1(0)%CR = u_nk  (input)
                  CALL RPRO1(NONLR_D,WDES1,WPROJ)                        ! WPROJ%CPROJ = <beta|r u_nk>
                  W1(I)%CPROJ=-W1(I)%CPROJ-WPROJ%CPROJ                   ! minus total differential
#endif
               ELSE
                  CALL PROJ1(NONL_S,WDES1,W1(I))                         !  W1(I)%CPROJ = <beta | d/dik u_nk>
#ifdef justatest
                  CALL SPHER_DER(W%WDES%GRID,NONL_D,P,W%WDES,LATT_CUR,I)
                  CALL PHASE(W%WDES,NONL_D,NK)
                  WPROJ%CW=W1(0)%CW
                  CALL PROJ1(NONL_D,WDES1,WPROJ)
                  W1(I)%CPROJ=-W1(I)%CPROJ-WPROJ%CPROJ
#endif
               ENDIF
#endif
            ENDDO


            ! calculate dot product < d/dk_i  u_k | H_k  | d/dk_j  u_k >
            E=W%CELEN(N,NK,ISP)
            DO I=1,3
               DO J=1,3
                  IF (ASSOCIATED(AVEC)) THEN
                     CALL ECCP_VEC(WDES1,W1(J),W1(I),LMDIM,CDIJ(1,1,1,ISP),WDES1%GRID,SV(1,ISP),AVEC,CMAG_LC(I,J))
                  ELSE
                     CALL ECCP(WDES1,W1(J),W1(I),LMDIM,CDIJ(1,1,1,ISP),WDES1%GRID,SV(1,ISP),CMAG_LC(I,J))
                  ENDIF
!                 CMAG_IC(I,J)=W1_DOT( W1(I), W1(J), CQIJ)*E
                  CMAG_IC(I,J)=W1_DOT( W1(I), W1(J), CQIJ)*(E-2._q*EFERMI)
                  CMAG_CHERN(I,J)=W1_DOT( W1(I), W1(J), CQIJ)
               ENDDO
            ENDDO

            WEIGHT=W%WDES%RSPIN * W%WDES%WTKPT(NK) * W%FERWE(N,NK,ISP)

            CHERN(1)=CHERN(1)+(CMAG_CHERN(2,3)-CMAG_CHERN(3,2))*WEIGHT
            CHERN(2)=CHERN(2)+(CMAG_CHERN(3,1)-CMAG_CHERN(1,3))*WEIGHT
            CHERN(3)=CHERN(3)+(CMAG_CHERN(1,2)-CMAG_CHERN(2,1))*WEIGHT


            M_IC(1)=M_IC(1)+(CMAG_IC(2,3)-CMAG_IC(3,2))*WEIGHT
            M_IC(2)=M_IC(2)+(CMAG_IC(3,1)-CMAG_IC(1,3))*WEIGHT
            M_IC(3)=M_IC(3)+(CMAG_IC(1,2)-CMAG_IC(2,1))*WEIGHT

            M_LC(1)=M_LC(1)+(CMAG_LC(2,3)-CMAG_LC(3,2))*WEIGHT
            M_LC(2)=M_LC(2)+(CMAG_LC(3,1)-CMAG_LC(1,3))*WEIGHT
            M_LC(3)=M_LC(3)+(CMAG_LC(1,2)-CMAG_LC(2,1))*WEIGHT

            DO I=1,3  ! loop over cartesian index
#ifdef norm_conserving
               IF (NONLR_S%LREAL ) THEN
                  W1(I)%CR=W1(0)%CR
! add derivative of projector (testing only) d <beta| / d ik  | u_nk>
                  CALL PHASERR(W%WDES%GRID,LATT_CUR,NONLR_D,NK,W%WDES,I)
                                                      ! NONLR_D = <beta|r
                  CALL RPRO1(NONLR_D,WDES1,W1(I))
                                                      ! W1(I)%CR = u_nk  (input)
                                                      ! W1(I)%CPROJ = <beta|r u_nk>  (output)
               ELSE
                  W1(I)%CW=W1(0)%CW
                  WRITE (*,*) 'bloch_currect2: recip. space proj der'
                  CALL SPHER_DER(W%WDES%GRID,NONL_D,P,W%WDES,LATT_CUR,I)
                  CALL PHASE(W%WDES,NONL_D,NK)
                  CALL PROJ1(NONL_D,WDES1,W1(I))
               ENDIF
#endif
            ENDDO

            !beware, should fail for spin polarized calculation
            DO I=1,3
               DO J=1,3
                  CALL ECCP_NL_ALL(WDES1,W1(J),W1(I),CDIJ_TMP,CDIJ_TMP,0._q,CMAG_BAPA(I,J))
               ENDDO
            ENDDO

            DM_BAPA(1)=DM_BAPA(1)+(CMAG_BAPA(2,3)-CMAG_BAPA(3,2))*WEIGHT
            DM_BAPA(2)=DM_BAPA(2)+(CMAG_BAPA(3,1)-CMAG_BAPA(1,3))*WEIGHT
            DM_BAPA(3)=DM_BAPA(3)+(CMAG_BAPA(1,2)-CMAG_BAPA(2,1))*WEIGHT

         ENDDO
      ENDDO kpoint
      ENDDO spin

      IF (NONLR_S%LREAL) THEN
         CALL NONLR_DEALLOC_CRREXP(NONLR_D)
      ELSE
         CALL NONL_DEALLOC_DER(NONL_D)
      ENDIF


      CALLMPI( M_sum_z(W%WDES%COMM, M_IC,3))
      CALLMPI( M_sum_z(W%WDES%COMM, M_LC,3))
      CALLMPI( M_sum_z(W%WDES%COMM, DM_BAPA,3))

! one-centre terms from atom where magnetic moment is placed
      MU_PARA_ONE_CTR=0
      MU_DIA_ONE_CTR =0
      ! type of magnetic ion
      NT=T_INFO%ITYP(MAGATOM)
      PP=>P(NT)
      NIP=NI_LOCAL(MAGATOM,W%WDES%COMM_INB)
      IF (NIP/=0) THEN
         IF ((ALLOCATED(DO_LOCAL)).AND.(DO_LOCAL(NT))) THEN
            CALL CALC_MU_MAGATOM(PP, CRHODE(:,:,NIP,1),MU_PARA_ONE_CTR,MU_DIA_ONE_CTR,.FALSE.)
!                                                 this is what we want ^^^^^^^^^^^^^^ = Delta M_dia ; eq 37
         ENDIF
      ENDIF

      CALLMPI( M_sum_d(W%WDES%COMM, MU_PARA_ONE_CTR(1), 3))    ! we don't want this
      CALLMPI( M_sum_d(W%WDES%COMM, MU_DIA_ONE_CTR(1), 3))     ! we do want this though

! CDIJ contain both v_ij and f_ij, see electron.F
! the on-site f_ij with the external magnetic moment (see SET_DD_MAGAGOM) doesn't have an augmentation by construction


!GIL 24 01 11! one-centre terms from all atome
!GIL 24 01 11      ! ATTENTION: the 9 has to be replaced by something more useful on the long run
!GIL 24 01 11      ALLOCATE(A_ONE_CENTER(3,9,T_INFO%NIONS))
!GIL 24 01 11
!GIL 24 01 11      DISPL=0
!GIL 25 04 11      CALL SETDIJ_AVEC_ONE_CENTER(W%WDES, GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, W%WDES%LOVERL, &
!GIL 25 04 11           LMDIM,CDIJ,AVTOT(1,1),  NONLR_S, NONL_S, IRDMAX, DISPL, A_ONE_CENTER, SIZE(A_ONE_CENTER,2))
!GIL 24 01 11
!GIL 25 04 11      CALL ONE_CENTRE_MOMENT( &
!GIL 25 04 11           W%WDES, GRIDC, GRIDUS, C_TO_US, &
!GIL 25 04 11           LATT_CUR, P, T_INFO, &
!GIL 25 04 11           LMDIM, CRHODE, IRDMAX, DISPL,  A_ONE_CENTER, MU_ONE_CTR )
!GIL 24 01 11
!GIL 24 01 11      DEALLOCATE(A_ONE_CENTER)
!GIL 24 01 11     
!GIL 24 01 11      IF (IU0>=0) THEN
!GIL 24 01 11         WRITE(IU0,'("M_ONE_C   ",6F14.7)') MU_ONE_CTR*1E6
!GIL 24 01 11      ENDIF
!GIL 24 01 11! mixed terms

!GIL 24 01 11      MU_PARA_MIXED=0
!GIL 24 01 11      MU_DIA_MIXED =0
!GIL 24 01 11     
!GIL 24 01 11      IF (W%WDES%LOVERL) THEN
!GIL 24 01 11         ALLOCATE(CRHODE1(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ))
!GIL 24 01 11         ALLOCATE(VIJ_PARA(LMDIM,LMDIM,3))
!GIL 24 01 11         ALLOCATE(VIJ_DIA(LMDIM,LMDIM,3))
!GIL 24 01 11
!GIL 24 01 11         DO NI=1,T_INFO%NIONS
!GIL 24 01 11         NT=T_INFO%ITYP(NI)
!GIL 24 01 11         PP=>P(NT)
!GIL 24 01 11         NIP=NI_LOCAL(NI,W%WDES%COMM_INB)
!GIL 24 01 11
!GIL 24 01 11         VIJ_PARA=0
!GIL 24 01 11         VIJ_DIA=0
!GIL 24 01 11
!GIL 24 01 11         IF (NIP/=0) THEN
!GIL 24 01 11            IF ((ALLOCATED(DO_LOCAL)).AND.(DO_LOCAL(NT))) THEN
!GIL 24 01 11               IF (ASSOCIATED(PP%JPAW)) THEN
!GIL 24 01 11                  ! copy the <psi_i | nabla Y_lm(0) | psi_i>- <tilde psi_i | nabla Y_lm(0) | tilde psi_i>
!GIL 24 01 11                  ! multiply by sqrt(4 pi) to kill Y_lm(0)
!GIL 24 01 11                  DO ICART=1,3
!GIL 24 01 11                    VIJ_PARA(1:PP%LMMAX,1:PP%LMMAX,ICART)=PP%JPAW(1:PP%LMMAX,1:PP%LMMAX,1,ICART)*SQRT(4*PI)
!GIL 24 01 11                  ENDDO
!GIL 24 01 11               ENDIF
!GIL 24 01 11               IF (NI==MAGATOM) CALL CALC_VIJ_ONECTR(PP, LMDIM, VIJ_PARA, VIJ_DIA)
!GIL 24 01 11            ENDIF
!GIL 24 01 11         ENDIF
!GIL 24 01 11
!GIL 24 01 11         DO ICART=1,3
!GIL 24 01 11            CRHODE1=0
!GIL 24 01 11            CALL DEPSUM_PHI(W, W%WDES, RPHI_CPROJ, LMDIM, CRHODE1, W%WDES%LOVERL, ICART)
!GIL 24 01 11
!GIL 24 01 11            DO ISP=1,W%WDES%ISPIN
!GIL 24 01 11            LM0=1
!GIL 24 01 11            DO I0=1,PP%LMAX   !CHANNELS
!GIL 24 01 11            LM1=1
!GIL 24 01 11            DO I1=1,PP%LMAX   !CHANNELS
!GIL 24 01 11
!GIL 24 01 11               L0 = PP%LPS(I0)
!GIL 24 01 11               L1 = PP%LPS(I1)
!GIL 24 01 11
!GIL 24 01 11               DO M1=1,2*L1+1
!GIL 24 01 11               DO M0=1,2*L0+1
!GIL 24 01 11
!GIL 24 01 11                  IF ((LM1+M1-1) <0 .OR. (LM1+M1-1) > SIZE(CRHODE1,1)) THEN
!GIL 24 01 11                     WRITE(0,*) 'internal error BLOCH_CURRENT',(LM1+M1-1),SIZE(CRHODE1,1)
!GIL 24 01 11                     STOP
!GIL 24 01 11                  ENDIF
!GIL 24 01 11                  IF ((LM0+M0-1) <0 .OR. (LM0+M0-1) > SIZE(CRHODE1,2)) THEN
!GIL 24 01 11                     WRITE(0,*) 'internal error BLOCH_CURRENT',(LM0+M0-1),SIZE(CRHODE1,2)
!GIL 24 01 11                     STOP
!GIL 24 01 11                  ENDIF
!GIL 24 01 11
!GIL 24 01 11                  IF (ICART==1) THEN
!GIL 24 01 11                     ! CONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP)) = CRHODE1(LM1+M1-1,LM0+M0-1,NIP,ISP)
!GIL 24 01 11                     ! this version is required if SO is ever implemented :)
!GIL 24 01 11                     MU_PARA_MIXED(2)=MU_PARA_MIXED(2)-CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,3)
!GIL 24 01 11                     MU_PARA_MIXED(3)=MU_PARA_MIXED(3)+CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,2)
!GIL 24 01 11                     MU_DIA_MIXED(2) =MU_DIA_MIXED(2) -CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,3)
!GIL 24 01 11                     MU_DIA_MIXED(3) =MU_DIA_MIXED(3) +CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,2)
!GIL 24 01 11                  ELSE IF (ICART==2) THEN                                                                           
!GIL 24 01 11                     MU_PARA_MIXED(3)=MU_PARA_MIXED(3)-CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,1)
!GIL 24 01 11                     MU_PARA_MIXED(1)=MU_PARA_MIXED(1)+CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,3)
!GIL 24 01 11                     MU_DIA_MIXED(3) =MU_DIA_MIXED(3) -CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,1)
!GIL 24 01 11                     MU_DIA_MIXED(1) =MU_DIA_MIXED(1) +CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,3)
!GIL 24 01 11                  ELSE IF (ICART==3) THEN                                                                           
!GIL 24 01 11                     MU_PARA_MIXED(1)=MU_PARA_MIXED(1)-CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,2)
!GIL 24 01 11                     MU_PARA_MIXED(2)=MU_PARA_MIXED(2)+CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_PARA(LM0+M0-1,LM1+M1-1,1)
!GIL 24 01 11                     MU_DIA_MIXED(1) =MU_DIA_MIXED(1) -CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,2)
!GIL 24 01 11                     MU_DIA_MIXED(2) =MU_DIA_MIXED(2) +CHGCONJG(CRHODE1(LM0+M0-1,LM1+M1-1,NIP,ISP))*VIJ_DIA (LM0+M0-1,LM1+M1-1,1)
!GIL 24 01 11                  ENDIF
!GIL 24 01 11
!GIL 24 01 11               ENDDO ! M0
!GIL 24 01 11               ENDDO ! M1
!GIL 24 01 11
!GIL 24 01 11            LM1=LM1+2*L1+1
!GIL 24 01 11            ENDDO ! I1
!GIL 24 01 11            LM0=LM0+2*L0+1
!GIL 24 01 11            ENDDO ! I0
!GIL 24 01 11            ENDDO
!GIL 24 01 11         ENDDO !ICART
!GIL 24 01 11         ENDDO
!GIL 24 01 11
!GIL 24 01 11         DEALLOCATE(CRHODE1,VIJ_PARA,VIJ_DIA)
!GIL 24 01 11
!GIL 24 01 11         CALLMPI( M_sum_z(W%WDES%COMM, MU_PARA_MIXED(1), 3))
!GIL 24 01 11         CALLMPI( M_sum_z(W%WDES%COMM, MU_DIA_MIXED(1), 3))
!GIL 24 01 11
!GIL 24 01 11         ! paramagnetic term: j = - |e| Imag <psi| nabla | psi>
!GIL 24 01 11         MU_PARA_MIXED=- AIMAG(MU_PARA_MIXED)
!GIL 24 01 11         ! diamagnetic term: j = - |e|^2 A
!GIL 24 01 11         MU_DIA_MIXED = -MU_DIA_MIXED
!GIL 24 01 11      ENDIF

! one-centre end

!     M_LC=M_LC/LATT_CUR%OMEGA      we need the moment in the full cell
!     M_IC=M_IC/LATT_CUR%OMEGA      we need the moment in the full cell
      M_LC=M_LC/(2.d0*RYTOEV)/AUTOA/AUTOA
      M_IC=M_IC/(2.d0*RYTOEV)/AUTOA/AUTOA
      DM_BAPA=DM_BAPA/(2.d0*RYTOEV)/AUTOA/AUTOA*(0._q,-1._q)
      DM_BAPA=-DM_BAPA                                           ! BEWARE: WE DO NOT UNDERSTAND THIS MINUS
      M_LC=M_LC*(0._q,-1._q)
      M_IC=M_IC*(0._q,-1._q)
      CHERN=CHERN/TPI  
      CHERN=CHERN*(0._q,-1._q)
        

        J=0
        DO I=1,3
        IF(DABS(AIMAG((M_LC(I) + M_IC(I) + DM_BAPA(I) + MU_DIA_ONE_CTR(I))*1E6)).gt.1E-8) THEN
        J=1
        ENDIF
        ENDDO


        IF(J.eq.1) THEN   !print complex part only if is non null

        IF (IU6>=0) WRITE(IU6,*) "the imaginary part of total magnetization is not zero"
        IF (IU0>=0) WRITE(IU0,*) "the imaginary part of total magnetization is not zero"

        IF (.NOT.NUCIND) THEN
                IF (IU6>=0) THEN
                 WRITE(IU6,111) CHERN,M_LC*1E6, M_IC*1E6, DM_BAPA*1E6, MU_DIA_ONE_CTR*1E6,&
     &          (M_LC+M_IC+DM_BAPA+MU_DIA_ONE_CTR)*1E6
                ENDIF
                IF (IU0>=0) THEN
                 WRITE(IU0,111) CHERN,M_LC*1E6, M_IC*1E6, DM_BAPA*1E6, MU_DIA_ONE_CTR*1E6,&
     &          (M_LC+M_IC+DM_BAPA+MU_DIA_ONE_CTR)*1E6

                ENDIF

        ELSE

                IF (IU6>=0) THEN
                 WRITE(IU6,222) CHERN,M_LC*1E6,M_IC*1E6,(M_LC+M_IC)*1E6
                ENDIF

                IF (IU0>=0) THEN
                 WRITE(IU0,222) CHERN,M_LC*1E6, M_IC*1E6, (M_LC+M_IC)*1E6
                ENDIF
        ENDIF

        ENDIF     ! !print complex part only if is non null

        IF (.NOT.NUCIND) THEN
                IF (IU6>=0) THEN
                 WRITE(IU6,1111) REAL(CHERN,q),REAL(M_LC,q)*1E6_q, REAL(M_IC,q)*1E6_q, REAL(DM_BAPA,q)*1E6_q, MU_DIA_ONE_CTR*1E6_q,&
     &          (REAL(M_LC,q)+REAL(M_IC,q)+REAL(DM_BAPA,q)+MU_DIA_ONE_CTR)*1E6_q
                ENDIF
                IF (IU0>=0) THEN
                 WRITE(IU0,1111) REAL(CHERN,q),REAL(M_LC,q)*1E6_q, REAL(M_IC,q)*1E6_q, REAL(DM_BAPA,q)*1E6_q, MU_DIA_ONE_CTR*1E6_q,&
     &      (REAL(M_LC,q)+REAL(M_IC,q)+REAL(DM_BAPA,q)+MU_DIA_ONE_CTR)*1E6_q
                ENDIF

        ELSE

                IF (IU6>=0) THEN
                 WRITE(IU6,2222) REAL(CHERN,q),REAL(M_LC,q)*1E6_q,REAL(M_IC,q)*1E6_q,(REAL(M_LC,q)+REAL(M_IC,q))*1E6_q
                ENDIF

                IF (IU0>=0) THEN
                 WRITE(IU0,2222) REAL(CHERN,q),REAL(M_LC,q)*1E6_q, REAL(M_IC,q)*1E6_q,(REAL(M_LC,q)+REAL(M_IC,q))*1E6_q
                ENDIF
        ENDIF

111  FORMAT(  ' '/ &
     &        ' '/ &
     &        ' '/ &
     &        ' '/ &
     &        '  Bloch Orbital magnetization (mu_B*1e6)                                                                               '/ &
     &        '                                                         X                              Y                             Z'/ &
     &        '  ---------------------------------------------------------------------------------------------------------------------'/ &
     &        '  Chern number                 CHERN          = ',6F14.8/ &            
     &        '  ---------------------------------------------------------------------------------------------------------------------'/ &
     &        '  Local Circulation            M_LC           = ',6F14.8/ &
     &        '  Itinerant Circulation        M_IC           = ',6F14.8/ &
     &        '  bare + para                  DM_BAPA        = ',6F14.8/ &
     &        '  diamagnetic contibution      MU_DIA_ONE_CTR = ',3F14.8/ &
     &        '  ---------------------------------------------------------------------------------------------------------------------'/ &
     &        '  Total magnetization          M              = ',6F14.8/ &
     &        '                                                                        '/ &
     &        '                                                                        '/ &
     &        '                                                                        '/ &
     &        '                                                                        ')

222  FORMAT(  '                                                                        '/ &
     &        '                                                                        '/ &
     &        '                                                                        '/ &
     &        '                                                                        '/ &
     &        '  Bloch Orbital magnetization NICS (mu_B*1e6)                           '/ &         
     &        '                                                        X                              Y                             Z'/ &
     &        '  --------------------------------------------------------------------------------------------------------------------'/ &
     &        '  Chern number                 CHERN          = ',6F14.8/ &
     &        '  --------------------------------------------------------------------------------------------------------------------'/ &  
     &        '  Local Circulation            M_LC           = ',6F14.8/ &
     &        '  Itinerant Circulation        M_IC           = ',6F14.8/ &
     &        '  --------------------------------------------------------------------------------------------------------------------'/ &
     &        '  Total magnetization          M              = ',6F14.8)


1111  FORMAT( '                                                                                      '/ &
     &        '----------------------------------------------------------------------------------------------'/ &
     &        '  Bloch Orbital magnetization (mu_B/1e6)                    X               Y               Z'/ &
     &        '----------------------------------------------------------------------------------------------'/ &
     &        '  Chern number                             ',3F16.6/ &
     &        '----------------------------------------------------------------------------------------------'/ &
     &        '  Local Circulation       M_LC             ',3F16.6/ &
     &        '  Itinerant Circulation   M_IC             ',3F16.6/ &
     &        '  bare + para             DM_BAPA          ',3F16.6/ &
     &        '  dia                     MU_DIA_ONE_CTR   ',3F16.6/ &
     &        '----------------------------------------------------------------------------------------------'/ &
     &        '  Total magnetization     TOTMAG           ',3F16.6/ &
     &        '----------------------------------------------------------------------------------------------'/ &
     &        '                                                                       ')


2222  FORMAT( '                                                                                       '/ &
     &        '                                                                                       '/ &
     &        '                                                                                       '/ &
     &        '                                                                                       '/ &
     &        '  Bloch Orbital magnetization NICS (mu_B*1e6)                                          '/ & 
     &        '                                                     X                 Y              Z'/ &
     &        '  --------------------------------------------------------------------------------------------'/ &
     &        '  Chern number                 CHERN          = ',3F14.8/ &
     &        '  --------------------------------------------------------------------------------------------'/ &
     &        '  Local Circulation            M_LC           = ',3F14.8/ &
     &        '  Itinerant Circulation        M_IC           = ',3F14.8/ &
     &        '  --------------------------------------------------------------------------------------------'/ &
     &        '  Total magnetization          TOTMAG         = ',3F14.8/ &
     &        '                                                                       '/ &
     &        '                                                                       '/ &
     &        '                                                                       '/ &
     &        '                                                                       ')



!****************************************************************************


#ifdef justatest
      CALL DELWAV(WPROJ,.TRUE.)
#endif
      DO N=0,3
         CALL DELWAV(W1(N) ,.TRUE.)
         CALL DELWAV(W2(N) ,.TRUE.)
      ENDDO
      CALL DELWAVA(WH)

    END SUBROUTINE BLOCH_CURRENT


!************************ SUBROUTINE DEPSUM_PHI ************************
!
! this subroutine calculates  the first order change of the
! one center (on site) "occupancy" i.e
!
!  rho_ij =  sum_n i < d u_n /d k_icart | p_i> <p_j| u_n>
!
!***********************************************************************

      SUBROUTINE DEPSUM_PHI(W0, WDES, RPHI_CPROJ, LMDIM, CRHODE, LOVERL, ICART)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavespin)    W0
      TYPE (wavedes)     WDES
      GDEFS           :: RPHI_CPROJ(:,:,:,:,:)
      LOGICAL LOVERL
      INTEGER LMDIM
      OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      INTEGER ICART

      INTEGER ISP, NT, NK, N, ISPINOR, ISPINOR_, LMBASE, LMBASE_, NIS, &
           LMMAXC, NI, L, LP
      REAL(q) WEIGHT0, WEIGHT1


      IF (.NOT.LOVERL) RETURN

      CRHODE=0

      spin:   DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
      band:   DO N=1,WDES%NBANDS

      WEIGHT0=WDES%RSPIN*W0%FERWE(N,NK,ISP)*WDES%WTKPT(NK)/2

      spinor: DO ISPINOR =0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1

      LMBASE =ISPINOR *(WDES%NPRO/2)
      LMBASE_=ISPINOR_*(WDES%NPRO/2)

      NIS   =1
      typ:  DO NT=1,WDES%NTYP
      LMMAXC=WDES%LMMAX(NT)
      IF (LMMAXC==0) GOTO 210

      ion: DO NI=NIS,WDES%NITYP(NT)+NIS-1

!DIR$ IVDEP
!OCL NOVREC
        DO L=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
        DO LP=1,LMMAXC
           ! we need i d/ d k<beta|   | u_nk> = -d/ d (i k)<beta|   | u_nk>
           CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)=CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)- &
                WEIGHT0*RPHI_CPROJ(L+LMBASE,N,NK,ISP,ICART)*GCONJG(W0%CPROJ(LP+LMBASE_,N,NK,ISP)) &
               -WEIGHT0*W0%CPROJ(L+LMBASE,N,NK,ISP)*GCONJG(RPHI_CPROJ(LP+LMBASE_,N,NK,ISP,ICART))

        ENDDO
        ENDDO
     
      LMBASE = LMMAXC+LMBASE
      LMBASE_= LMMAXC+LMBASE_

      ENDDO ion

  210 NIS = NIS+WDES%NITYP(NT)
      ENDDO typ
      ENDDO
      ENDDO spinor

      ENDDO band
      ENDDO kpoint
      ENDDO spin
      ! sum over all bands
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_d(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ*2))
#endif

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE ONE_CENTER_MOMENT *****************
!
! this subroutine calculates  the augmentation contribution to
! the current density JTOT
! as input it requires  CRHODE(LM,LMP,ION,ISP)
! and a possible displacement vector DISPL
!
!***********************************************************************


    SUBROUTINE ONE_CENTRE_MOMENT( &
           WDES, GRIDC_, GRIDUS, C_TO_US, &
           LATT_CUR, P, T_INFO, &
           LMDIM, CRHODE, IRDMAX, DISPL,  A_ONE_CENTER , MOMENT)
      USE prec
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE constant
      USE us

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER :: GRIDC
      TYPE (transit)     C_TO_US
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES

      INTEGER   IRDMAX         ! allocation required for augmentation
      INTEGER   LMDIM
      OVERLAP   CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      LOGICAL   LADDITIONAL
      REAL(q), POINTER, OPTIONAL :: A_ONE_CENTER(:,:,:)
      REAL(q)   DISPL(3,T_INFO%NIONS)
      COMPLEX(q)       :: MOMENT(:,:)
!  work arrays
      INTEGER       :: LYMAX, LYMAXP1
      TYPE (potcar), POINTER :: PP
      OVERLAP       :: JLM(256)
      REAL(q),ALLOCATABLE ::   DIST(:),DEP(:),YLM(:,:),XS(:),YS(:),ZS(:)
      OVERLAP,ALLOCATABLE ::   SUM(:)
      INTEGER,ALLOCATABLE ::   NLI(:)
      OVERLAP,ALLOCATABLE ::   CTMP(:,:,:)
      INTEGER       :: LYDIM, NT, NI, NIP, ICART, L, M, IND, INDMAX, LMYDIM, INDYLM , LSTART
      INTEGER, EXTERNAL :: MAXL_AUG

!=======================================================================
! allocation of all required quantities
!=======================================================================
      ! LADDITIONAL uses an even finer grid for
      ! calculating the augmentation charges
      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)


! find the maximum L for augmentation charge (usually just 2 maximum l)
      LYDIM=MAXL_AUG(T_INFO%NTYP,P)+1
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE( DIST(IRDMAX),DEP(IRDMAX),SUM(IRDMAX),YLM(IRDMAX,LMYDIM), &
                NLI(IRDMAX),XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX))

      IF (LADDITIONAL) THEN
         GRIDC => GRIDUS
      ELSE
         GRIDC => GRIDC_
      ENDIF

      ALLOCATE(CTMP(LMDIM,LMDIM,T_INFO%NIONS))
      MOMENT=0
!-----------------------------------------------------------------------
! merge augmentation occupancies CRHODE from all nodes
! for simplicity I do this using M_sum_d but there are of course better
! ways to do this
!-----------------------------------------------------------------------

! hopefully CRHODE is in the representation such that the *total*
! charge is stored in channel one
      CTMP=0
      DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB)
         IF (NIP/=0) THEN
            CTMP(:,:,NI)=CRHODE(:,:,NIP,1)
         ENDIF
      ENDDO
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INB,CTMP,LMDIM*LMDIM*T_INFO%NIONS))
#else
      CALLMPI( M_sum_z(WDES%COMM_INB,CTMP,LMDIM*LMDIM*T_INFO%NIONS))
#endif
!-----------------------------------------------------------------------
! loop over cartesian index and ion
!-----------------------------------------------------------------------
   cart: DO ICART=1,3
      ion: DO NI=1,T_INFO%NIONS
      NT=T_INFO%ITYP(NI)
      PP=>PP_POINTER(P, NI, NT)
!-----------------------------------------------------------------------
! for this ion (this type of ion) no depletion charge
!-----------------------------------------------------------------------
      IF (PP%PSDMAX==0 .OR.  .NOT. ASSOCIATED(PP%JPAW) ) CYCLE
!-----------------------------------------------------------------------
! calulate the spherical harmonics (DEP is Work-arrays)
!-----------------------------------------------------------------------
      ! nabla operator increases the maximum L one-center number by one
      LYMAX  =MAXL1(PP)*2
      LYMAXP1=LYMAX+1

      CALL SETYLM_AUG(GRIDC,LATT_CUR,T_INFO%POSION(1,NI),PP%PSDMAX,NPSRNL, &
     &        LMYDIM,LYMAXP1,YLM(1,1),IRDMAX,INDMAX, &
     &        DISPL(1,NI),DISPL(2,NI), DISPL(3,NI),DIST(1),NLI(1),XS(1),YS(1),ZS(1))

      SUM=0
      JLM=0
      CALL CALC_RHOLM_JVEC( CTMP(:,:,NI) , JLM, PP, ICART)

#ifdef current_augmentation
      DO L =0,LYMAXP1
!        IF (L<=1) WRITE(*,'("JAUG",3I4,10F10.6)') ICART,NI,L,(AIMAG(JLM(L**2+M))*1E6,M=1,(L*2)+1)
         CALL SETDEP(PP%QDEP(1,1,L),PP%PSDMAX,NPSRNL, &
              LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
         DO M=1,(L*2)+1
            INDYLM =L **2  +M
            ! paramagnetic augmentation current
!DIR$ IVDEP
!OCL NOVREC
            DO IND=1,INDMAX
               SUM(IND)=SUM(IND)+DEP(IND)*YLM(IND,INDYLM)*JLM(INDYLM)*(0.0_q,1.0_q)
            ENDDO
         ENDDO
      ENDDO
#endif
!-----------------------------------------------------------------------
! here we have the total paramagnetic augmentation current
!-----------------------------------------------------------------------
      IF (ICART==1) THEN
         DO IND=1,INDMAX
            MOMENT(2,NI)=MOMENT(2,NI)+ZS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
            MOMENT(3,NI)=MOMENT(3,NI)-YS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
         ENDDO
      ELSE IF (ICART==2) THEN
         DO IND=1,INDMAX
            MOMENT(3,NI)=MOMENT(3,NI)+XS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
            MOMENT(1,NI)=MOMENT(1,NI)-ZS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
         ENDDO
      ELSE IF (ICART==3) THEN
         DO IND=1,INDMAX
            MOMENT(1,NI)=MOMENT(1,NI)+YS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
            MOMENT(2,NI)=MOMENT(2,NI)-XS(IND)*SUM(IND)*DIST(IND)*(1._q/GRIDC%NPLWV)
         ENDDO
      ENDIF
!-----------------------------------------------------------------------
      ENDDO ion
      ENDDO cart
!-----------------------------------------------------------------------
      CALLMPI( M_sum_z(GRIDC%COMM, MOMENT, SIZE(MOMENT)))

      DEALLOCATE(DIST,DEP,SUM,YLM,NLI,XS,YS,ZS)
      DEALLOCATE(CTMP)

      RETURN
    END SUBROUTINE ONE_CENTRE_MOMENT

!*********************************** CALC_VIJ **************************************
!
! To calculate one-centre integrals of mixed contributions the magnetic moment on
! the atom where external moment is placed.
!
! COMMENT:
! para:
! dia: 
!
!***********************************************************************************

    SUBROUTINE CALC_VIJ_ONECTR(PP,LMDIM,VIJ_PARA,VIJ_DIA)
      USE pseudo
      USE constant
      USE asa
      USE radial                                                                ! ?
      USE wave
      USE paw
      IMPLICIT NONE
     
      TYPE (potcar), POINTER :: PP   ! pseudopotential descriptor
      INTEGER LMDIM
      COMPLEX(q) :: VIJ_PARA(LMDIM,LMDIM,3), VIJ_DIA(LMDIM,LMDIM,3)  ! one centre velocities

     !local variables
      INTEGER NMAX,I,NT
      INTEGER I0,I1
      INTEGER L0,L1
      INTEGER M0,M1
      INTEGER LM0,LM1
      INTEGER LMRHO, LRHO, MRHO
      INTEGER LMAX,K
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:),YLM_MxR_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: TMP(:),WTMP(:),DWAE(:),DWPS(:)
      OVERLAP :: CTMP(LMDIM,LMDIM)
      REAL(q) :: DLM(256)
      REAL(q) :: SUMI
      integer :: j1

      VIJ_PARA=0
      VIJ_DIA=0

! allocate
      IF (.NOT. ORBITALMAG .OR. .NOT. ASSOCIATED(PP%JPAW)) RETURN

      NMAX=PP%R%NMAX
      ALLOCATE(TMP(NMAX))
      ALLOCATE(WTMP(NMAX))
      ALLOCATE(DWAE(NMAX))
      ALLOCATE(DWPS(NMAX))

      LMAX=MAXL1(PP)

! because of augmentation contribution 2*LMAX
      ALLOCATE(YLM_NABLA_YLM((2*LMAX+1)*(2*LMAX+1),(2*LMAX+1)*(2*LMAX+1),0:3))
      ALLOCATE(YLM_X_YLM    ((2*LMAX+1)*(2*LMAX+1),(2*LMAX+1)*(2*LMAX+1),0:3))
      ALLOCATE(YLM_MxR_YLM  ((2*LMAX+1)*(2*LMAX+1),(2*LMAX+1)*(2*LMAX+1),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM    =0._q
      YLM_MxR_YLM  =0._q

      CALL SETYLM_NABLA_YLM(2*LMAX,YLM_NABLA_YLM,YLM_X_YLM)

      ! store < Y_lm | M x r | Y_l'm' >
      YLM_MxR_YLM(:,:,1)=MAGDIPOL(2)*YLM_X_YLM(:,:,3)-MAGDIPOL(3)*YLM_X_YLM(:,:,2)
      YLM_MxR_YLM(:,:,2)=MAGDIPOL(3)*YLM_X_YLM(:,:,1)-MAGDIPOL(1)*YLM_X_YLM(:,:,3)
      YLM_MxR_YLM(:,:,3)=MAGDIPOL(1)*YLM_X_YLM(:,:,2)-MAGDIPOL(2)*YLM_X_YLM(:,:,1)

!-----------------------------------------------------------------------
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
         WTMP(:)=PP%WAE(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         DO M1=1,2*L1+1
         DO M0=1,2*L0+1

            IF ((L1*L1+M1) <0 .OR. (L1*L1+M1) > SIZE(YLM_MxR_YLM,1)) THEN
               WRITE(0,*) 'internal error CALC_VIJ_ONECTR',(LM1+M1-1),SIZE(YLM_MxR_YLM,1)
               STOP
            ENDIF
            IF ((L0*L0+M0) <0 .OR. (L0*L0+M0) > SIZE(YLM_MxR_YLM,2)) THEN
               WRITE(0,*) 'internal error CALC_VIJ_ONECTR',(LM0+M0-1),SIZE(YLM_MxR_YLM,2)
               STOP
            ENDIF
            IF ((LM1+M1-1) <0 .OR. (LM1+M1-1) > SIZE(VIJ_DIA,1)) THEN
               WRITE(0,*) 'internal error  CALC_VIJ_ONECTR',(LM1+M1-1),SIZE(VIJ_DIA,1)
               STOP
            ENDIF
            IF ((LM0+M0-1) <0 .OR. (LM0+M0-1) > SIZE(VIJ_DIA,2)) THEN
               WRITE(0,*) 'internal error  CALC_VIJ_ONECTR',(LM0+M0-1),SIZE(VIJ_DIA,2)
               STOP
            ENDIF

            DO I=1,3
!-----------------------------------------------------------------------
! paramagnetic contribution
!-----------------------------------------------------------------------
               IF ((ABS(YLM_X_YLM(L0*L0+M0,L1*L1+M1,I))>1E-8) .OR.  &
                   (ABS(YLM_NABLA_YLM(L0*L0+M0,L1*L1+M1,I))>1E-8)) THEN
                  TMP=0._q
                  DO K=1,PP%R%NMAX
                     TMP(K) =  ( YLM_NABLA_YLM(L0*L0+M0,L1*L1+M1,I)/PP%R%R(K) ) *        &
                          (PP%WAE(K,I0) * PP%WAE(K,I1) -  PP%WPS(K,I0) * PP%WPS(K,I1)) + &
                                 YLM_X_YLM(L0*L0+M0,L1*L1+M1,I) *                        &
                          (PP%WAE(K,I0) * DWAE(K)      -  PP%WPS(K,I0) * DWPS(K))
                  ENDDO
                  SUMI=0._q
                  CALL SIMPI(PP%R,TMP,SUMI)
                  VIJ_PARA(LM1+M1-1,LM0+M0-1,I)=VIJ_PARA(LM1+M1-1,LM0+M0-1,I)+SUMI
               ENDIF
!-----------------------------------------------------------------------
! diamagnetic contribution
!-----------------------------------------------------------------------
               IF (ABS(YLM_MxR_YLM(L1*L1+M1,L0*L0+M0,I))>1E-10) THEN
                  TMP=0._q
                  DO K=1,PP%R%NMAX
                     TMP(K) = 1._q/PP%R%R(K)**2 * &
                          (PP%WAE(K,I0) * PP%WAE(K,I1) -  PP%WPS(K,I0) * PP%WPS(K,I1))
                  ENDDO
                  SUMI=0._q
                  CALL SIMPI(PP%R,TMP,SUMI)
                  ! A(r) =  M x r / r^3
                  VIJ_DIA(LM1+M1-1,LM0+M0-1,I)=VIJ_DIA(LM1+M1-1,LM0+M0-1,I)+SUMI*MOMTOMOM* &
                          YLM_MxR_YLM(L1*L1+M1,L0*L0+M0,I)
               ENDIF

            ENDDO  ! I

         ENDDO ! M0
         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

! make antisymmetric
      DO I0=1,PP%LMMAX
         DO I1=I0,PP%LMMAX
            VIJ_PARA(I0,I1,:)= (VIJ_PARA(I0,I1,:)-VIJ_PARA(I1,I0,:))/2
            VIJ_PARA(I1,I0,:)= -VIJ_PARA(I0,I1,:)
         ENDDO
      ENDDO
!-----------------------------------------------------------------------
! diamagnetic augmentation contribution
! int M x r / r^3 Y_lm(r) Q(r) = int M x (vec r/r) / r^2 r^2 Y_lm(r) dr
! use instead of AE-PS simply the augmentation contribution
!-----------------------------------------------------------------------
      DO I=1,3
         DLM=0
         LMRHO=1
         DO LRHO=0,2*LMAX
            TMP=0
            DO K=1,PP%R%NMAX
               TMP(K)=PP%AUG(K,LRHO)/PP%R%R(K)**2
            ENDDO
            SUMI=0._q
            CALL SIMPI(PP%R,TMP,SUMI)
            DO MRHO=1,2*LRHO+1
               IF (LMRHO+MRHO-1>SIZE(YLM_MxR_YLM,2)) THEN
                  WRITE(*,*) 'internal error in  CALC_VIJ_ONECTR:',LMRHO+MRHO-1,LMRHO,MRHO, SIZE(YLM_MxR_YLM)
                  STOP
               ENDIF
               DLM(LMRHO+MRHO-1)=SUMI*MOMTOMOM*(YLM_MxR_YLM(1,LMRHO+MRHO-1,I)*SQRT(4*PI))
            ENDDO
            LMRHO=LMRHO+2*LRHO+1
         ENDDO
         CTMP=0
         CALL CALC_DLLMM( CTMP, DLM, PP)

         VIJ_DIA(1:LMDIM,1:LMDIM,I)=CTMP
      ENDDO

      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM, YLM_MxR_YLM, TMP, WTMP, DWAE, DWPS)

    END SUBROUTINE CALC_VIJ_ONECTR


!************************ SUBROUTINE SETDIJ_AVEC ***********************
!
! this subroutine is the interface to SETDIJ_AVEC_
! upon first call, it however does a little bit more
! if LGAUGE is selected the average vector potential in each sphere
! is determined A(R) and this vector field is then used as the
! gauge twist in the non-local projectors
!
!***********************************************************************

    SUBROUTINE SETDIJ_AVEC(WDES, GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, LOVERL, &
        LMDIM,CDIJ,AVTOT_, NONLR_S, NONL_S, IRDMAX)
      USE prec
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE constant
      USE nonl_high
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  :: GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER :: GRIDC
      TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES

      INTEGER  IRDMAX      ! allocation required for augmentation
      INTEGER  LMDIM
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      COMPLEX(q),POINTER :: AVTOT_(:,:)
      LOGICAL  LOVERL
      REAL(q)  DISPL(3,T_INFO%NIONS)
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct)  NONL_S

   !  local
      LOGICAL :: LINIT

      IF (.NOT. ORBITALMAG ) RETURN

      LINIT=.FALSE.
      IF (LGAUGE) THEN
         IF (NONLR_S%LREAL) THEN
            IF (.NOT. ASSOCIATED(NONLR_S%VKPT_SHIFT)) THEN
               ALLOCATE(NONLR_S%VKPT_SHIFT(3, T_INFO%NIONS))
               NONLR_S%VKPT_SHIFT=0
               LINIT=.TRUE.
            ENDIF
         ELSE
            WRITE(*,*) 'SETDIJ_AVEC: ERROR: when LGAUGE=.TRUE. you must set LREAL=.TRUE. as well, stopping ...'
            STOP
         ENDIF
      ENDIF
#ifdef dij_augmentation
      DISPL=0
      CALL SETDIJ_AVEC_(WDES, GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, LOVERL, &
           LMDIM,CDIJ,AVTOT_,  NONLR_S, NONL_S, IRDMAX, DISPL)
#else
      IF (LINIT) THEN
      ! we need to call SETDIJ_AVEC at least once to set up the phase twisted
      ! projectors
      ! the first call to this routine is done from main.F and CDIJ from that call is 
      ! never used
         WRITE(0,*) 'SETDIJ_AVEC is only called in initial phase'
         DISPL=0
         CALL SETDIJ_AVEC_(WDES, GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, LOVERL, &
            LMDIM,CDIJ,AVTOT_,  NONLR_S, NONL_S, IRDMAX, DISPL)
      ENDIF
#endif

      IF (LINIT) THEN
         IF (NONLR_S%LREAL) THEN
            CALL RSPHER(WDES%GRID,NONLR_S,LATT_CUR)
         ELSE
            CALL SPHER(WDES%GRID,NONL_S,P,WDES,LATT_CUR, 1)
            CALL PHASE(WDES,NONL_S,0)
         ENDIF
      ENDIF

    END SUBROUTINE SETDIJ_AVEC

!************************ SUBROUTINE DIAGSHIFT *************************
!
! Subroutine for converting shift/shielding tensor to csa parameters
!
! Definitions from J.Mason, Solid state Nuc.Magn.Res. 2, 285 (1993)
!
! Mind: converting from shielding to shift and vice versa changes
!       the sign of iso. span and skew are unaffected.
!
! Mind: initial program by Filipe. Extension to shielding by Gilles.
!       for shieldings only eta, span and skew were adapted.
!
! Mind: evidently we do have to symmetrize the tensor first, otherwise
!       we don't obtain the DALTON reference.
!
!***********************************************************************

      SUBROUTINE DIAGSHIFT(A,MODUS,ISO,SPAN,SKEW)

      USE prec

      IMPLICIT NONE

      REAL(q) A(3,3)
      CHARACTER*5 MODUS

      REAL(q) ISO,SPAN,SKEW

      INTEGER, PARAMETER :: LWORK=6
      REAL(q) EIG(3), AS(3,3)
      REAL(q) WORK(3*LWORK)
      REAL(q) DEL11,DEL22,DEL33
      REAL(q) TMP1
      INTEGER I,ID,J,II
      INTEGER IFAIL

      ISO=0._q
      SPAN=0._q
      SKEW=0._q

      IF ((MODUS.EQ.'SHIFT').OR.(MODUS.EQ.'shift')) THEN
!        WRITE (0,*) 'shift'
      ELSEIF ((MODUS.EQ.'SHIEL').OR.(MODUS.EQ.'shiel')) THEN
!        WRITE (0,*) 'shielding'
      ELSE 
         WRITE (0,*) 'ERROR IN DAIGSHIFT INPUT, STOP'
         STOP
      ENDIF

! symmetrize
      DO I=1,3
        DO J=1,3
          AS(J,I)=0.5_q*(A(J,I)+A(I,J))
!         AA(J,I)=0.5_q*(A(J,I)-A(I,J))
        ENDDO
      ENDDO

      CALL DSYEV('V','U',3,AS,3,EIG,WORK,3*LWORK,IFAIL)
      IF (IFAIL.NE.0) THEN
         WRITE (0,*) 'INTERNAL ERROR DIAGSHIFT, DSYEV, STOP'
         STOP
      ENDIF

      IF ((EIG(1).GE.EIG(2)).AND.(EIG(1).GE.EIG(3))) THEN
        DEL11=EIG(1)
        IF (EIG(2).GE.EIG(3)) THEN
          DEL22=EIG(2)
          DEL33=EIG(3)
        ELSE
          DEL22=EIG(3)
          DEL33=EIG(2)
        ENDIF
      ELSEIF ((EIG(2).GE.EIG(1)).AND.(EIG(2).GE.EIG(3))) THEN
        DEL11=EIG(2)
        IF (EIG(1).GE.EIG(3)) THEN
          DEL22=EIG(1)
          DEL33=EIG(3)
        ELSE
          DEL22=EIG(3)
          DEL33=EIG(1)
        ENDIF
      ELSEIF ((EIG(3).GE.EIG(1)).AND.(EIG(3).GE.EIG(1))) THEN
        DEL11=EIG(3)
        IF (EIG(1).GE.EIG(2)) THEN
          DEL22=EIG(1)
          DEL33=EIG(2)
        ELSE
          DEL22=EIG(2)
          DEL33=EIG(1)
        ENDIF
      ELSE
        WRITE (0,*) 'INTERNAL ERROR DIAGSHIFT, STOP'
        STOP
      ENDIF

      IF ((MODUS.EQ.'SHIFT').OR.(MODUS.EQ.'shift')) THEN
        ISO  = (DEL11+DEL22+DEL33)/3._q
        SPAN = DEL11-DEL33
        IF (ABS(DEL33 - DEL11).LE.0.0001_q) THEN
          SKEW = 1000._q
        ELSE
          SKEW = 3._q*(ISO - DEL22)/(DEL33 - DEL11)
        ENDIF
      ELSEIF ((MODUS.EQ.'SHIEL').OR.(MODUS.EQ.'shiel')) THEN
        TMP1 = DEL33
        DEL33 = DEL11
        DEL11 = TMP1
        ISO  = (DEL11+DEL22+DEL33)/3._q
        SPAN = DEL33-DEL11
        IF (ABS(DEL33 - DEL11).LE.0.0001_q) THEN
          SKEW = 1000._q
        ELSE
          SKEW = 3._q*(ISO - DEL22)/(DEL33 - DEL11)
        ENDIF
      ELSE
        WRITE (*,*) 'INTERNAL ERROR DIAGSHIFT,  STOP'
        STOP
      ENDIF

      RETURN

      END SUBROUTINE DIAGSHIFT

END MODULE morbitalmag
