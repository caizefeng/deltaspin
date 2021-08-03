#include "symbol.inc"
      MODULE locproj
      USE prec
      USE c2f_interface

      IMPLICIT NONE

      PUBLIC LPRJ_READER,LPRJ_PROALL,LPRJ_COVL,LPRJ_WRITE, &
     &       LPRJ_DEALLOC_COVL,LPRJ_LCAO,LPRJ_LDApU

      PRIVATE
!
!! on which we will project our Bloch orbitals.
!
      TYPE LPRJ_function
         ! radial part
         INTEGER radial_type    ! 1="optimal" PAW projector, 2=PAW partial wave (PS), 3=Hydrogen-like (=wannier90)
         INTEGER species
         INTEGER n
         REAL(q) za
         ! spherical part
         INTEGER l
         INTEGER m
         ! site part
         INTEGER poscar_site
         REAL(q) R(3)
         ! 
         INTEGER iproj          ! index of projector or partial wave to be projected on
         INTEGER ibase          ! first pseudopotential channel with angular moment L=l
         INTEGER nproj          ! number of projectors with same L

         REAL(q), POINTER :: optproj(:)
      END TYPE LPRJ_function

      TYPE (LPRJ_function), ALLOCATABLE, SAVE :: LPRJ_functions(:)

      REAL(q), ALLOCATABLE, SAVE :: PSPNL_SPLINE(:,:,:,:)
      REAL(q), ALLOCATABLE, SAVE :: HYDRO_SPLINE(:,:,:)

      INTEGER, PARAMETER :: LMAX=3
      INTEGER, PARAMETER :: NMAX=1000

      COMPLEX(q), ALLOCATABLE, SAVE :: LPRJ_COVL(:,:,:,:)

!
!! when one or more local functions are of locproj::lprj_function::radial_type=2
!! (pseudo partial waves). This tells VASP to call lcao::lcao_init.
!
      LOGICAL :: LOCPROJ_NEEDS_PS=.FALSE.

      CONTAINS


!******************** SUBROUTINE LPRJ_READER ***************************
!
!! Contrary to the usual behaviour, multiple entries are all taken into
!! account (instead of only the first occurence).
!!
!! locproj::lprj_reader calls the parser and allocated and fills the
!! locproj::lprj_functions data structure.
!!
!! At the end the following members of LPRJ_functions(i) will have been
!! set (for all functions i in the list):
!!
!! locproj::lprj_function::radial_type:
!!   specifies the type of radial funtion. This can be
!!   radial_type=1  "optimal" PAW projector
!!                  (set by sphpro::sphpro_fast for LORBIT=14),
!!   radial_type=2  PAW pseudo partial wave, or
!!   radial_type=3  Hydrogen-like (Slater type).
!!
!! locproj::lprj_function::poscar_site = n,
!!   where n is the number of the site in the POSCAR file, on which the local
!!   function is centered (if specified in this manner).
!!   Alternatively the position of the local function may have been
!!   specified in terms of direct coordinates of the real space lattice.
!!   In the latter case n=-1.
!!
!! locproj::lprj_function::R: 
!!   specifies the site on which the local function is centered in
!!   direct coordinates of the real space lattice.
!!
!! locproj::lprj_function::l:
!!   l-quantum number of this local function:
!!    l= 0: s,  l= 1: p,   l= 2: d,   l= 3: f,
!!    l=-1: sp, l=-2: sp2, l=-3: sp3, l=-4: sp3d, l=-5: sp3d2
!!   for radial_type/=3 only pure spherical harmonics are used
!!   (i.e., l>=0).
!!
!! locproj::lprj_function::m:
!!   orbital magnetic quantum number of this local function.
!!
!! locproj::lprj_function::species = n
!!   signifies that projectors or pseudo partial waves derived from
!!   the n-th PAW dataset in the POTCAR are to be used to define the
!!   radial part of a particular local function (n=-1 in case
!!   radial_type=3).
!!
!! locproj::lprj_function::n:
!!   in case radial_type/=3: use the "n"-th projector or pseudo partial
!!   wave of angular momentum l.
!!   For radial_type=3: use Slater-type function with n nodes.
!!   Default: locproj::lprj_function::n=1
!!
!! locproj::lprj_function::za:
!!   defines diffusivity of Slater type function (default za=1.0). 
!!   Only relevant for hydrogen-like local functions.
!!
!! locproj::lprj_function::iproj:
!!   the channel number of the PAW projector or pseudo partial wave
!!   to be used as radial part of the local function (only set for
!!   radial_type=1 or 2, otherwise locproj::lprj_function::iproj=-1).
!!   For radial_type=1 or 2 the projector or pseudo partial wave
!!   of the highest lying bound atomic state with angular moment l
!!   will be used. Strictly this is only guaranteed for radial_type=2:
!!   for radial_type=1 VASP will use linear combinations of the
!!   projectors read in from POTCAR in accordance with potcar::optproj
!!   (set in pawm::set_opt_proj in a way that tries to recover the
!!   original order, i.e., the one in potcar::atomic_e).
!!   For LORBIT=14 potcar::optproj is set in sphpro::sphpro_fast, and
!!   only a single "optimal" projector will be available. In that case
!!   the value of iproj is ignored in locproj::lprj_proall below.
!!
!! locproj::lprj_function::ibase:
!!   the channel number of the first PAW projector or pseudo partial wave
!!   with angular moment l (only set for radial_type=1 or 2).
!!
!! locproj::lprj_function::nproj:
!!   number PAW channels with angular moment l
!!   (only set for radial_type=1 or 2).
!
!***********************************************************************

      SUBROUTINE LPRJ_READER(T_INFO,P,IU0)
      USE base
      USE poscar
      USE pseudo
      USE full_kpoints
      USE pead
      USE vaspxml

      TYPE (type_info) T_INFO
      TYPE (potcar) P(T_INFO%NTYP)
      INTEGER IU0

!!      ! local variables
!!      TYPE parser_function
!!         SEQUENCE
!!         ! radial part
!!         INTEGER radial_type
!!         INTEGER species
!!         INTEGER n
!!         ! spherical part
!!         INTEGER l
!!         INTEGER m
!!         ! site part
!!         INTEGER poscar_site
!!         REAL*8  posx
!!         REAL*8  posy
!!         REAL*8  posz 
!!
!!         REAL*8  sigma
!!      END TYPE parser_function

      TYPE (parser_function), ALLOCATABLE :: functions(:)

      REAL(q) DE,DEOLD,E,Z,ZCORE,DR
      INTEGER N,I,J,IFOUND1,IFOUND2,IFOUND3,ITMP(3)
      LOGICAL LFOUND

      REAL(q), PARAMETER :: TINYE=1.E-3_q
      REAL(q), PARAMETER :: TINYO=1.E-1_q
      REAL(q), PARAMETER :: TINYR=1.E-4_q

      CALL parse_file(N,LEN_TRIM(INCAR),INCAR)

      IF (N==0) THEN
         CALL free_parser(); RETURN
      ENDIF

      ALLOCATE(functions(N))

      DO I=1,N
         CALL fill_basis_info(functions(I),I)
      ENDDO

      CALL free_parser()

      IF (ALLOCATED(LPRJ_functions)) DEALLOCATE(LPRJ_functions)
      ALLOCATE(LPRJ_functions(N))

      LPRJ_functions(:)%iproj=-1

      CALL XML_TAG("locproj")

      ! validate and copy input to LPRJ_functions
      DO I=1,N
         IF (functions(I)%poscar_site/=-1) THEN
            ! a poscar site index has to be in range [1,NIONS]
            IF (functions(I)%poscar_site<1.OR.functions(I)%poscar_site>T_INFO%NIONS) THEN
               IF (IU0>=0) WRITE(*,'(A,I4,A,I4)') 'LPRJ_READER: ERROR: site index of function:',I, &
              &   ' out of range:',functions(I)%poscar_site
               STOP
            ENDIF
            functions(I)%posx=T_INFO%POSION(1,functions(I)%poscar_site)
            functions(I)%posy=T_INFO%POSION(2,functions(I)%poscar_site)
            functions(I)%posz=T_INFO%POSION(3,functions(I)%poscar_site)
!        ELSE
!           IF (functions(I)%radial_type==1) THEN
!              IF (IU0>=0) WRITE(*,'(A,I4,A)') 'LPRJ_READER: ERROR: function:',I, &
!             &   ': no POSCAR site specified'
!              STOP
!           ENDIF
         ENDIF
         ! copy site information
         LPRJ_functions(I)%poscar_site=functions(I)%poscar_site
         LPRJ_functions(I)%R(1)=MOD(functions(I)%posx+100.5_q,1._q)-0.5_q
         LPRJ_functions(I)%R(2)=MOD(functions(I)%posy+100.5_q,1._q)-0.5_q
         LPRJ_functions(I)%R(3)=MOD(functions(I)%posz+100.5_q,1._q)-0.5_q
         LPRJ_functions(I)%nproj=-1

         IF (functions(I)%radial_type==1.OR.functions(I)%radial_type==2) THEN
            IF (functions(I)%l<0) THEN
               IF (IU0>=0) WRITE(*,'(A,I4,A,I4)') 'LPRJ_READER: ERROR: function:',I, &
              &   ' is not a simple spherical harmonic, L:',functions(I)%l
               STOP
            ENDIF
            IF (functions(I)%species==-1) THEN
               IF (functions(I)%poscar_site==-1) THEN
                  IF (IU0>=0) WRITE(*,'(A,I4,A)') 'LPRJ_READER: ERROR: both POSCAR as well as POTCAR index of function:',I, &
                 &   ' not specified'
                  STOP
               ENDIF
               functions(I)%species=T_INFO%ITYP(functions(I)%poscar_site)  
            ENDIF
            IF (functions(I)%species<1.OR.functions(I)%species>T_INFO%NTYP) THEN
               IF (IU0>=0) WRITE(*,'(A,I4,A,I4)') 'LPRJ_READER: ERROR: POTCAR index of function:',I, &
           &   ' out of range:',functions(I)%species
               STOP
            ENDIF


            IFOUND1=0; IFOUND2=0
            DO J=1,P(functions(I)%species)%LMAX
               IF (P(functions(I)%species)%LPS(J)==functions(I)%l .AND. IFOUND1==0) IFOUND1=J
               IF (P(functions(I)%species)%LPS(J)==functions(I)%l) IFOUND2=IFOUND2+1
            ENDDO
            
            IF (IFOUND2==0) THEN
               IF (IU0>=0) WRITE(*,'(A,I4,A,I4,A,I4)') 'LPRJ_READER: ERROR: function:',I, &
                    &   ': no state with L:',functions(I)%l,' on site:',functions(I)%poscar_site
               STOP
            ENDIF
            
            LPRJ_functions(I)%ibase=IFOUND1          ! first index of projectors
            LPRJ_functions(I)%nproj=IFOUND2          ! total number of projectors

            ALLOCATE(LPRJ_functions(I)%optproj(IFOUND2)); LPRJ_functions(I)%optproj=0

            IF (functions(I)%n/=-1) THEN
               ! Take the n-th projector with angular moment l (if it exists, of course).
               IFOUND1=0; LFOUND=.FALSE.
               DO J=1,P(functions(I)%species)%LMAX
                  IF (P(functions(I)%species)%LPS(J)==functions(I)%l) IFOUND1=IFOUND1+1
                  IF (IFOUND1==functions(I)%n) THEN
                     LFOUND=.TRUE.; IFOUND2=J; EXIT
                  ENDIF
               ENDDO
               IF (.NOT.LFOUND) THEN
                  IF (IU0>=0) WRITE(*,'(A,I4,A,I1,A,I1,A,I4)') 'LPRJ_READER: ERROR: function:',I, &
                 &   ': no ',functions(I)%n,"-th projector, with l=",functions(I)%l,' on site:',functions(I)%poscar_site
                  STOP
               ELSE
                  LPRJ_functions(I)%iproj=IFOUND2
               ENDIF
            ELSE
               ! Take the projector that corresponds to the highest lying bound state with
               ! angular momentum l.
               ! First identify the corresponding atomic state.
               LFOUND=.FALSE.; IFOUND1=0; E=-1.E+6_q
               DO J=1,SIZE(P(functions(I)%species)%ATOMIC_N)
                  IF (P(functions(I)%species)%ATOMIC_L(J)==functions(I)%l &
                 &   .AND.P(functions(I)%species)%ATOMIC_OCC(J)>TINYO &
                 &   .AND.P(functions(I)%species)%ATOMIC_E(J)>E) THEN
                     LFOUND=.TRUE.; IFOUND1=J; E=P(functions(I)%species)%ATOMIC_E(J)
                  ENDIF
               ENDDO
               IF (.NOT.LFOUND) THEN
                  IF (IU0>=0) WRITE(*,'(A,I4,A,I4,A,I4)') 'LPRJ_READER: ERROR: function:',I, &
              &   ': no state with L:',functions(I)%l,' on site:',functions(I)%poscar_site
                  STOP
               ENDIF
               ! search for corresponding projector
               IFOUND2=0; IFOUND3=0; DEOLD=1.E+6_q
               DO J=1,P(functions(I)%species)%LMAX
                  IF (P(functions(I)%species)%LPS(J)==functions(I)%l) IFOUND3=IFOUND3+1
                  DE=ABS(P(functions(I)%species)%ATOMIC_E(IFOUND1)-P(functions(I)%species)%E(J))
                  IF (DE<DEOLD) THEN
                     IFOUND2=J; DEOLD=DE
                     functions(I)%n=IFOUND3
                  ENDIF
               ENDDO            
               IF (DEOLD>TINYE) THEN
                  IF (IU0>=0) WRITE(*,'(A,I4,A,I4,A,I4)') 'LPRJ_READER: ERROR: function:',I, &
              &   ': no projector for highest valence state with L: ',functions(I)%l,'on site:',functions(I)%poscar_site
                  STOP
               ELSE
                  LPRJ_functions(I)%iproj=IFOUND2
               ENDIF
            ENDIF
         ENDIF

         ! this tells VASP to call LCAO_INIT
         IF (functions(I)%radial_type==2) LOCPROJ_NEEDS_PS=.TRUE.

         LPRJ_functions(I)%radial_type=functions(I)%radial_type

         LPRJ_functions(I)%species=functions(I)%species

         LPRJ_functions(I)%n=functions(I)%n
         LPRJ_functions(I)%l=functions(I)%l
         LPRJ_functions(I)%m=functions(I)%m

         LPRJ_functions(I)%za=functions(I)%sigma


         ITMP(1)=LPRJ_functions(I)%n; ITMP(2)=LPRJ_functions(I)%l; ITMP(3)=LPRJ_functions(I)%m

         CALL XML_TAG("function")
         CALL XML_TAG_INT("radial_type",LPRJ_functions(I)%radial_type)
         CALL XML_TAG_INT("species",LPRJ_functions(I)%species)
         CALL XML_VEC_INT(ITMP,"nlm")
         CALL XML_TAG_REAL("za",LPRJ_functions(I)%za)
         CALL XML_TAG_INT("poscar_site",LPRJ_functions(I)%poscar_site)
         CALL XML_VEC_REAL(LPRJ_functions(I)%R,"position")
         CALL XML_CLOSE_TAG("function")

!        IF (IU0>=0) WRITE(*,'(I4,X,"n=",I2,X,"l=",I2,X,"m=",I2,X,"rtype=",I2,X,"species=",I2,X,"za=",F14.7,X,"site=",I4,X,"R=",3F14.7,X,"ipr=",I2)') &
!       & I,LPRJ_functions(I)%n,LPRJ_functions(I)%l,LPRJ_functions(I)%m,LPRJ_functions(I)%radial_type, &
!       &   LPRJ_functions(I)%species,LPRJ_functions(I)%za,LPRJ_functions(I)%poscar_site,LPRJ_functions(I)%R, &
!       &   LPRJ_functions(I)%iproj
      ENDDO

      CALL XML_CLOSE_TAG("locproj")

      DEALLOCATE(functions)

      CALL USE_FULL_KPOINTS
      CALL PEAD_REQUEST

      RETURN
      END SUBROUTINE LPRJ_READER


!******************** SUBROUTINE LPRJ_PROALL ***************************
!
!! means of the LOCPROJ tag.
!!
!! This list of functions is stored in the locproj::lprj_functions
!! datastructure.
!
!***********************************************************************

      SUBROUTINE LPRJ_PROALL(W,WDES,GRID,P,CQIJ,LATT_CUR,T_INFO,INFO,IU6,IU0)
      USE base
      USE pseudo
      USE mgrid
      USE wave_high
      USE full_kpoints
      USE constant
      USE lattice
      USE poscar

      USE lcao, ONLY : ATOM_LCAO

      TYPE (wavespin) W
      TYPE (wavedes) WDES
      TYPE (grid_3d) GRID
      TYPE (potcar) P(:)
      TYPE (latt) LATT_CUR
      TYPE (type_info) T_INFO
      TYPE (info_struct) INFO

      OVERLAP CQIJ(:,:,:,:)

      INTEGER IU6,IU0

      ! local variables
      INTEGER IFNC,ISP,IK,ISPINOR,I,IB
      INTEGER LSTART,LSTOP,NYLM,LMBASE,LM

      INTEGER N,L,M,IPROJ,IBASE,ITYPE,ISPEC,NPROJ
      REAL(q) R(3),ZA

      COMPLEX(q) CPROJ

      REAL(q) HYBRID_ORBITAL((LMAX+1)**2)

      REAL(q), ALLOCATABLE :: SPLINE(:,:,:)
      GDEF, ALLOCATABLE :: S(:,:)

      LOGICAL LNORM,LAUG

      IF (.NOT.ALLOCATED(LPRJ_functions)) RETURN

      CALL CHECK_FULL_KPOINTS

      IF (ALLOCATED(LPRJ_COVL)) DEALLOCATE(LPRJ_COVL)
      ALLOCATE(LPRJ_COVL(WDES%NB_TOT,KPOINTS_FULL%NKPTS,MAX(WDES%ISPIN,WDES%NRSPINORS),SIZE(LPRJ_functions)))

      funcs: DO IFNC=1,SIZE(LPRJ_functions)

         N=LPRJ_functions(IFNC)%n
         L=LPRJ_functions(IFNC)%l
         M=LPRJ_functions(IFNC)%m
         R=LPRJ_functions(IFNC)%R

         IBASE=LPRJ_functions(IFNC)%ibase
         IPROJ=LPRJ_functions(IFNC)%iproj
         ISPEC=LPRJ_functions(IFNC)%species
         ITYPE=LPRJ_functions(IFNC)%radial_type
         NPROJ=LPRJ_functions(IFNC)%nproj
         
         ZA=LPRJ_functions(IFNC)%za

         ! radial types
         SELECT CASE(ITYPE)

         CASE(1)
         ! project onto PAW projector or "optimal" PAW projector (LORBIT=14)

            ! slightly paranoid tests
            IF (IPROJ-IBASE<0) THEN
               WRITE(0,*) 'internal error in: LPRJ_PROALL IBASE and IPROJ incompatible',IBASE-IPROJ
               STOP
            ENDIF
            IF (IPROJ-IBASE>=NPROJ) THEN
               WRITE(0,*) 'internal error in: LPRJ_PROALL IPROJ too large',IBASE-IPROJ,NPROJ
               STOP
            ENDIF
            ! this condition can only occur if SPHPRO_FAST has reduced the number of available
            ! vectors to 1 (LORBIT=14); then we pick the only available projector
            ! (unfortunately this information can not be made available to LPRJ_READER)
            IF (IPROJ-IBASE>=SIZE(P(ISPEC)%OPTPROJ,2)) THEN
               LPRJ_functions(IFNC)%iproj=IBASE
               LPRJ_functions(IFNC)%n=1
               IPROJ=IBASE
            ENDIF
            ! calculate non-local projectors (rec. space spline fits)
            CALL CALC_PSPNL_SPLINE(P)
            ! copy relevant one to SPLINE
            ALLOCATE(SPLINE(1:NPSNL,5,LMAX+1))
            ! copy positions from first projector
            SPLINE(1:NPSNL,1,L+1)=PSPNL_SPLINE(1:NPSNL,1,IBASE,ISPEC)
            ! initialize rest
            SPLINE(1:NPSNL,2:,L+1)=0
!           IF (IU0>=0) WRITE(*,'("using NT=",I4," L=",I4," IBASE=",I4," IPROJ=",I4,4F14.7)') & 
!                ISPEC, L, IBASE, IPROJ-IBASE+1, P(ISPEC)%OPTPROJ(1:NPROJ,IPROJ-IBASE+1,L+1)
            ! now linear combine the projectors
            DO I=1,NPROJ
               SPLINE(1:NPSNL,2:,L+1)=SPLINE(1:NPSNL,2:,L+1)+PSPNL_SPLINE(1:NPSNL,2:,IBASE+I-1,ISPEC)*P(ISPEC)%OPTPROJ(I,IPROJ-IBASE+1,L+1)
            ENDDO
            LPRJ_functions(IFNC)%optproj(1:NPROJ)=P(ISPEC)%OPTPROJ(1:NPROJ,IPROJ-IBASE+1,L+1)
            ! not S orthogonal
            LSTART=L; LSTOP=LSTART; LNORM=.FALSE.; LAUG=.FALSE.
            ! make them S orthogonal
!           LSTART=L; LSTOP=LSTART; LNORM=.TRUE.; LAUG=.TRUE.

         CASE(2)
         ! project onto PAW partial wave (pseudized)

            I=FIND_ATOM_LCAO_ENTRY(ISPEC,IPROJ)
            ! copy relevant entry of ATOM_LCAO%WFCT_G to SPLINE
            ALLOCATE(SPLINE(SIZE(ATOM_LCAO(ISPEC)%WFCT_G,1),5,LMAX+1))
            SPLINE(:,:,L+1)=ATOM_LCAO(ISPEC)%WFCT_G(:,:,I)
            LSTART=L; LSTOP=LSTART; LNORM=.TRUE.; LAUG=.TRUE.

         CASE(3)
         ! project onto Hydrogen-like functions (like in the VASP2WANNIER90 interface)

            ALLOCATE(SPLINE(NMAX,5,LMAX+1))
            ! calculate hydrogen like functions (rec. space spline fits)
            CALL CALC_HYDRO_SPLINE(N,ZA,LMAX,INFO%ENMAX)
            SPLINE=HYDRO_SPLINE
            LSTART=0; LSTOP=LMAX; LNORM=.TRUE.; LAUG=.TRUE.
         CASE DEFAULT
            IF (IU0>=0) WRITE(*,'(A,I4,X,A,I4)') 'LPRJ_PROALL: ERROR: function:',IFNC, &
           &   'is of an unknown radial type:',ITYPE
            STOP
         END SELECT

         NYLM=(LSTOP+1)**2-LSTART**2
         ALLOCATE(S(WDES%NB_TOT,NYLM))

         CALL LPRJ_ORBITAL_DEFINITIONS(L,M,HYBRID_ORBITAL)

         spinor: DO ISPINOR=1,WDES%NRSPINORS
         spin: DO ISP=1,WDES%ISPIN
         kpoints: DO IK=1,KPOINTS_FULL%NKPTS

            CALL CALC_OVERLAP_GN(LSTART,LSTOP,SPLINE,R,W,KPOINTS_FULL%VKPT(:,IK),ISP,ISPINOR,P,CQIJ,LATT_CUR,T_INFO,LNORM,LAUG,S)

            ! Make the desired linear combinations
            LMBASE=LSTART**2
            DO IB=1,WDES%NB_TOT
               CPROJ=0
               DO LM=1,NYLM
                  CPROJ=CPROJ+S(IB,LM)*HYBRID_ORBITAL(LM+LMBASE)
               ENDDO
               LPRJ_COVL(IB,IK,ISP+ISPINOR-1,IFNC)=CPROJ
            ENDDO

         ENDDO kpoints
         ENDDO spin
         ENDDO spinor

         DEALLOCATE(S,SPLINE)

      ENDDO funcs

      IF (ALLOCATED(PSPNL_SPLINE)) DEALLOCATE(PSPNL_SPLINE)
      IF (ALLOCATED(HYDRO_SPLINE)) DEALLOCATE(HYDRO_SPLINE)

! test_
!     IF (IU0>=0) THEN
!        DO ISP=1,WDES%ISPIN
!        DO IK=1,WDES%NKPTS
!           WRITE(IU0,'(A,I4)') 'kpoint:',IK
!           DO IB=1,WDES%NB_TOT
!              WRITE(IU0,'(I4,4X,16(2F14.7,2X))') IB,LPRJ_COVL(IB,IK,ISP,:) !,W%CPROJ(9*WDES%LMMAX(1)+1,IB,IK,ISP)
!              WRITE(IU0,'(I4,4X,16(2F14.7,2X))') IB,W%CPROJ(WDES%LMMAX(1)+1,IB,IK,ISP),W%CPROJ(WDES%LMMAX(1)+4,IB,IK,ISP),W%CPROJ(WDES%LMMAX(1)+5,IB,IK,ISP),W%CPROJ(WDES%LMMAX(1)+3,IB,IK,ISP)
!           ENDDO
!        ENDDO
!        ENDDO
!     ENDIF
! test_

      RETURN
      END SUBROUTINE LPRJ_PROALL


!******************** SUBROUTINE LPRJ_WRITE ****************************
!
!! functions to the PROJCAR, LOCPROJ, and vasprun.xml files.
!!
!! The information written to these files is essentially the same,
!! but only PROJCAR is explicitly constructed to be human readable.
!!
!! The first line of the LOCPROJ file lists: NS, NK, NB, and NF
!! (# of spin channels, # of k-points, # of bands, and # of local functions)
!! followed by a block of NF lines describing the local functions.
!! After this block the projections are written:
!!
!! ( IFNC,LPRJ_COVL(IB,IK,ISP,IFNC), IFNC=1,NF, IB=1,NB, IK=1,NK, ISP=1,NS )
!!
!! i.e., IFNC is the fastest index, and ISP the slowest.
!
!***********************************************************************

      SUBROUTINE LPRJ_WRITE(IU6,IU0,W)
      USE vaspxml
      USE wave
      USE full_kpoints
      INTEGER IU6,IU0
      TYPE(wavespin) :: W
      ! local variables
      INTEGER NB,NK,NS,NF
      INTEGER IB,IK,IK_IRZ,ISP,IFNC,IBASE,ISITE,ITMP(3)
      REAL(q), PARAMETER :: TINYR=1.E-4_q
      REAL(q) RTMP(2), DR
      LOGICAL, ALLOCATABLE :: LDONE(:), LDONOW(:)

     ! CHARACTER(LEN=10) :: LABEL(1:7,-5:3)= &
     !&(/"  sp3d2-1 ","  sp3d2-2 ","  sp3d2-3 ","  sp3d2-4 ","  sp3d2-5 ","  sp3d2-6 ","          ", &
     !&  "   sp3d-1 ","   sp3d-2 ","   sp3d-3 ","   sp3d-4 ","   sp3d-5 ","          ","          ", &
     !&  "   sp3-1  ","   sp3-2  ","   sp3-3  ","   sp3-4  ","          ","          ","          ", &
     !&  "   sp2-1  ","   sp2-2  ","   sp2-3  ","          ","          ","          ","          ", &
     !&  "    sp-1  ","    sp-2  ","          ","          ","          ","          ","          ", &
     !&  "      s   ","          ","          ","          ","          ","          ","          ", &
     !&  "     py   ","     pz   ","     px   ","          ","          ","          ","          ", &
     !&  "    dxy   ","    dyz   ","    dz2   ","    dxz   ","   dx2-y2 ","          ","          ", &
     !&  "fy(3x2-y2)","    fxyz  ","    fyz2  ","    fz3   ","    fxz2  "," fz(x2-y2)","fx(x2-3y2)" /)
      CHARACTER(LEN=10) :: LABEL(1:7,-5:3)
      DATA LABEL(1:7,-5) &
     &/"  sp3d2-1 ","  sp3d2-2 ","  sp3d2-3 ","  sp3d2-4 ","  sp3d2-5 ","  sp3d2-6 ","          "/
      DATA LABEL(1:7,-4) &
     &/"   sp3d-1 ","   sp3d-2 ","   sp3d-3 ","   sp3d-4 ","   sp3d-5 ","          ","          "/
      DATA LABEL(1:7,-3) &
     &/"   sp3-1  ","   sp3-2  ","   sp3-3  ","   sp3-4  ","          ","          ","          "/
      DATA LABEL(1:7,-2) &
     &/"   sp2-1  ","   sp2-2  ","   sp2-3  ","          ","          ","          ","          "/
      DATA LABEL(1:7,-1) &
     &/"    sp-1  ","    sp-2  ","          ","          ","          ","          ","          "/
      DATA LABEL(1:7, 0) &
     &/"      s   ","          ","          ","          ","          ","          ","          "/
      DATA LABEL(1:7, 1) &
     &/"     py   ","     pz   ","     px   ","          ","          ","          ","          "/
      DATA LABEL(1:7, 2) &
     &/"    dxy   ","    dyz   ","    dz2   ","    dxz   ","   dx2-y2 ","          ","          "/
      DATA LABEL(1:7, 3) &
     &/"fy(3x2-y2)","    fxyz  ","    fyz2  ","    fz3   ","    fxz2  "," fz(x2-y2)","fx(x2-3y2)"/

      CHARACTER(LEN=15) :: PRKIND(3)=(/"PAW projector  ","PS partial wave","Hydrogen-like  "/)

      CHARACTER (LEN=40) :: strcounter
 
      IF (.NOT.ALLOCATED(LPRJ_COVL)) RETURN

      NB=SIZE(LPRJ_COVL,1)
      NK=SIZE(LPRJ_COVL,2)
      NS=SIZE(LPRJ_COVL,3)
      NF=SIZE(LPRJ_COVL,4)


      IF (IU0>=0) THEN
!=======================================================================
         OPEN(UNIT=99,FILE='PROJCAR',STATUS='REPLACE')

         ALLOCATE(LDONE(NF), LDONOW(NF))
         LDONE=.FALSE.

         funcs1: DO IBASE=1,NF

            ! skip entries already printed
            samesite: IF (.NOT.LDONE(IBASE)) THEN
            
               ! print site and type
               WRITE(99,'(3X,A,I6,2X,A,3F14.7,2X,A,2X,A,A)',ADVANCE='No') &
              &  'ISITE:',LPRJ_functions(IBASE)%poscar_site,'  R= ',LPRJ_functions(IBASE)%R, &
              &   PRKIND(LPRJ_functions(IBASE)%radial_type),':'
       
               IF (LPRJ_functions(IBASE)%radial_type==3) THEN
                  WRITE(99,'(2X,A,I2,2X,A,F8.4)') &
                 &   'n=',LPRJ_functions(IBASE)%n,'za=',LPRJ_functions(IBASE)%za
               ELSE
                  WRITE(99,'(2X,A,I2)') &
                 &   'species  =',LPRJ_functions(IBASE)%species
               ENDIF
       
               ! now collect all functions on this site with a similar "recipy":
               ! for radial_type==3 this means all functions at the same site of
               ! the same type and with the same n and za. In case radial_type/=3
               ! all functions on the same site of the same type and the same POTCAR
               ! species are grouped together.
               LDONOW=.FALSE.
               DO IFNC=IBASE,NF
                  ! same site as before
                  DR=ABS(LPRJ_functions(IBASE)%R(1)-LPRJ_functions(IFNC)%R(1))+ &
                       &   ABS(LPRJ_functions(IBASE)%R(2)-LPRJ_functions(IFNC)%R(2))+ &
                       &   ABS(LPRJ_functions(IBASE)%R(3)-LPRJ_functions(IFNC)%R(3))
                  
                  IF (.NOT. LDONE(IFNC) .AND.  &
                       & DR<TINYR.AND. &
                       & LPRJ_functions(IBASE)%radial_type==LPRJ_functions(IFNC)%radial_type.AND. &
                       & LPRJ_functions(IBASE)%species==LPRJ_functions(IFNC)%species.AND. &
                       & LPRJ_functions(IBASE)%n==LPRJ_functions(IFNC)%n.AND. &
                       & LPRJ_functions(IBASE)%za==LPRJ_functions(IFNC)%za) THEN
                     LDONOW(IFNC)=.TRUE.
                  ENDIF
               ENDDO
       
               spin1: DO ISP=1,NS
               kpoints1: DO IK=1,NK
                  WRITE(99,'(/X,A,I6,2X,A,I4/)') 'k-point:',IK,'spin:',ISP
       
                  WRITE(99,'(3X,A)',ADVANCE='No') 'band'
                  DO IFNC=IBASE,NF
                     IF (LDONOW(IFNC)) &
                     WRITE(99,'(4X,A,X)',ADVANCE='No') LABEL(LPRJ_functions(IFNC)%m,LPRJ_functions(IFNC)%l)
                  ENDDO
                  WRITE(99,*)
       
                  DO IB=1,NB
                     WRITE(99,'(X,I6,X)',ADVANCE='No') IB
                     DO IFNC=IBASE,NF
                        IF (LDONOW(IFNC)) &
                        WRITE(99,'(2X,F6.3,X,F6.3)',ADVANCE='No') LPRJ_COVL(IB,IK,ISP,IFNC)
                     ENDDO
                     WRITE(99,*)
                  ENDDO
       
               ENDDO kpoints1
               ENDDO spin1
       
               LDONE=LDONE.OR.LDONOW

               WRITE(99,*)
            ENDIF samesite
         ENDDO funcs1

         DEALLOCATE(LDONE, LDONOW)
         CLOSE(99)
!=======================================================================
         OPEN(UNIT=99,FILE='LOCPROJ',STATUS='REPLACE')
         WRITE(99,'(4I6,"  # of spin, # of k-points, # of bands, # of proj" )') NS,NK,NB,NF
         DO IFNC=1,NF
            WRITE(99,'(3X,A,I6,2X,A,3F14.7,2X,A,2X,A,A)',ADVANCE='No') &
           &  'ISITE:',LPRJ_functions(IFNC)%poscar_site,'  R= ',LPRJ_functions(IFNC)%R, &
           &   PRKIND(LPRJ_functions(IFNC)%radial_type),':'
            WRITE(99,'(A)') LABEL(LPRJ_functions(IFNC)%m,LPRJ_functions(IFNC)%l)
         ENDDO
         WRITE(99,*)
         spin2: DO ISP=1,NS
            kpoints2: DO IK=1,NK
               IK_IRZ=KPOINTS_FULL%NEQUIV(IK)
               bands2: DO IB=1,NB
                  WRITE(99,'("orbital",3I6,2F20.10)') ISP,IK,IB,REAL(W%CELTOT(IB,IK_IRZ,ISP),q),W%FERTOT(IB,IK_IRZ,ISP)
                  funcs2: DO IFNC=1,NF
                     WRITE(99,'(1I6,2F20.10)') IFNC,LPRJ_COVL(IB,IK,ISP,IFNC)
                  ENDDO funcs2
                  WRITE(99,*)
               ENDDO bands2
            ENDDO kpoints2
         ENDDO spin2
         CLOSE(99)
      ENDIF

      CALL XML_TAG("locprojected")
      funcs3: DO IFNC=1,NF
         WRITE(strcounter,"(I6)") IFNC
         CALL XML_TAG("set", comment="function "//TRIM(ADJUSTL(strcounter)))

         ITMP(1)=LPRJ_functions(IFNC)%n; ITMP(2)=LPRJ_functions(IFNC)%l; ITMP(3)=LPRJ_functions(IFNC)%m

         CALL XML_TAG_INT("radial_type",LPRJ_functions(IFNC)%radial_type)
         CALL XML_TAG_INT("species",LPRJ_functions(IFNC)%species)
         CALL XML_VEC_INT(ITMP,"nlm")
         CALL XML_TAG_REAL("za",LPRJ_functions(IFNC)%za)
         CALL XML_TAG_INT("poscar_site",LPRJ_functions(IFNC)%poscar_site)
         CALL XML_VEC_REAL(LPRJ_functions(IFNC)%R,"position")
         CALL XML_TAG_STRING("label",TRIM(ADJUSTL(LABEL(LPRJ_functions(IFNC)%m,LPRJ_functions(IFNC)%l))))

         IF (LPRJ_functions(IFNC)%radial_type==1) &
            CALL XML_VEC_REAL(LPRJ_functions(IFNC)%optproj,"optproj")

         spin3: DO ISP=1,NS
            WRITE(strcounter,"(I2)") ISP
            CALL XML_TAG("set", comment="spin "//TRIM(ADJUSTL(strcounter)))
      
            kpoints3: DO IK=1,NK
               WRITE(strcounter,"(I6)") IK
               CALL XML_TAG("set", comment="kpoint "//TRIM(ADJUSTL(strcounter)))
      
               DO IB=1,NB
                  RTMP(1)=REAL(LPRJ_COVL(IB,IK,ISP,IFNC),q)
                  RTMP(2)=AIMAG(LPRJ_COVL(IB,IK,ISP,IFNC))
                  CALL XML_ROW_DATA(RTMP, FORM='(F14.7,1X)')
               ENDDO
               CALL XML_CLOSE_TAG("set")
            ENDDO kpoints3
            CALL XML_CLOSE_TAG("set")
         ENDDO spin3
         CALL XML_CLOSE_TAG("set")
      ENDDO funcs3
      CALL XML_CLOSE_TAG("locprojected")

      RETURN
      END SUBROUTINE LPRJ_WRITE


!******************** SUBROUTINE LPRJ_LDApU ****************************
!
! this is a kind of hacked in version of LDA+U
! instead of doing LDA+U directly, this bit of code writes
! a density matrix file GAMMA that can be used to correct
! the density matrix (using ICHARG=5)
!
!***********************************************************************

      SUBROUTINE LPRJ_LDApU(IO, W)
      USE base
      USE wave
      USE full_kpoints
      USE fileio
      TYPE (in_struct)   IO
      TYPE (wavespin) W
    ! local
      INTEGER ISP, NS, NK, IK, IK_IRZ, IB, IB2, NB, IFNC, IFNC2, I, NF
      REAL(q) DR
      REAL(q), PARAMETER :: TINYR=1.E-4_q
      GDEF, ALLOCATABLE :: DENSITY_MATRIX(:,:,:), U(:,:,:), GAMMA(:,:)
!#define testgamma_matrix
#ifdef testgamma_matrix
  ! LAPACK    
      REAL(q)    R(W%WDES%NB_TOT)
      INTEGER :: IFAIL
      INTEGER, PARAMETER :: LWORK=32
      GDEF       CWRK(LWORK*W%WDES%NB_TOT)
      REAL(q)    RWORK(3*W%WDES%NB_TOT)
#endif
      
      IF (.NOT.ALLOCATED(LPRJ_COVL)) RETURN
      
      NB=SIZE(LPRJ_COVL,1)
      NK=SIZE(LPRJ_COVL,2)
      NS=SIZE(LPRJ_COVL,3)
      NF=SIZE(LPRJ_COVL,4)

      ALLOCATE(DENSITY_MATRIX(NF,NF, NS), U(NF,NF, NS))
      DENSITY_MATRIX=0

      ! the first step is to determine the occupancy matrix in the
      ! space of the projected functions
      DO ISP=1,NS
         DO IK=1,NK
            IK_IRZ=KPOINTS_FULL%NEQUIV(IK)
            DO IB=1,NB
               DO IFNC=1,NF
               DO IFNC2=1,NF
                  DENSITY_MATRIX(IFNC,IFNC2,ISP)=DENSITY_MATRIX(IFNC,IFNC2,ISP)+ & 
                       LPRJ_COVL(IB,IK,ISP,IFNC)*CONJG(LPRJ_COVL(IB,IK,ISP,IFNC2))*W%FERTOT(IB,IK_IRZ,ISP)
               ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      DENSITY_MATRIX=DENSITY_MATRIX/KPOINTS_FULL%NKPTS
      ! now work on one site only
      U=0
      IFNC=1
      DO
         DO I=IFNC,NF
            ! same site as before
            DR=ABS(LPRJ_functions(IFNC)%R(1)-LPRJ_functions(I)%R(1))+ &
                 &   ABS(LPRJ_functions(IFNC)%R(2)-LPRJ_functions(I)%R(2))+ &
                 &   ABS(LPRJ_functions(IFNC)%R(3)-LPRJ_functions(I)%R(3))
            IF (DR<TINYR) THEN
               IFNC2=I
            ENDIF
         ENDDO
         ! for each site and spin calculate (1/2-n)
         DO ISP=1,NS
            U(IFNC:IFNC2,IFNC:IFNC2,ISP)=-DENSITY_MATRIX(IFNC:IFNC2,IFNC:IFNC2,ISP)
            DO I=IFNC,IFNC2
               U(I,I,ISP)=0.5_q+U(I,I,ISP)
            ENDDO
         ENDDO
         IFNC=IFNC2+1
         IF (IFNC2>=NF) EXIT
      ENDDO

    ! hard coded U
      U=U*7

#ifdef testgamma_matrix
      DO ISP=1,NS
         WRITE(*,*) 'density matrix'
         WRITE(*,'(10F8.3)') REAL(DENSITY_MATRIX(:,:,ISP),q)
      ENDDO

      DO ISP=1,NS
         WRITE(*,*) 'one-center U matrix'
         WRITE(*,'(10F8.3)') REAL(U(:,:,ISP),q)
      ENDDO
#endif
    ! finally construct the correction to the Hamiltonian and
    ! write it to the file GAMMA
      CALL OPENGAMMA
      CALL WRITEGAMMA_HEAD_HAM( W%WDES%NKPTS, W%WDES%NB_TOT, IO)

      ALLOCATE(GAMMA(W%WDES%NB_TOT,W%WDES%NB_TOT))
      
      DO ISP=1,NS
         DO IK=1,W%WDES%NKPTS
            GAMMA=0
            DO IB=1,NB
            DO IB2=1,NB
               DO IFNC=1,NF
               DO IFNC2=1,NF
                  GAMMA(IB,IB2)=GAMMA(IB,IB2)+CONJG(LPRJ_COVL(IB,IK,ISP,IFNC))*LPRJ_COVL(IB2,IK,ISP,IFNC2)* & 
                       U(IFNC,IFNC2,ISP)
               ENDDO
               ENDDO
            ENDDO
            ENDDO
            CALL WRITEGAMMA(IK, W%WDES%NB_TOT, W%WDES%NB_TOT, GAMMA, .FALSE.)
            DO IB=1,W%WDES%NB_TOT
               GAMMA(IB,IB)=GAMMA(IB,IB)+REAL(W%CELTOT(IB,IK,ISP),q)
            ENDDO
#ifdef testgamma_matrix
            IF (IK==1 .AND. ISP==1) THEN
               WRITE(*,'(8F14.7)') (REAL(GAMMA(I,I),q),I=1, W%WDES%NB_TOT)
            ENDIF
#ifdef  gammareal
            CALL DSYEV &
              ('V','U',NB,GAMMA(1,1),W%WDES%NB_TOT, &
              R,CWRK,LWORK*WDES%NB_TOT, IFAIL)
#else
            CALL ZHEEV &
              ('V','U',NB,GAMMA(1,1),W%WDES%NB_TOT, &
              R,CWRK,LWORK*W%WDES%NB_TOT, RWORK,  IFAIL)
#endif
            IF (IK==1 .AND. ISP==1) THEN
               WRITE(*,'(8F14.7)') (R(I),I=1, W%WDES%NB_TOT)
               WRITE(*,*)
            ENDIF
#endif
         ENDDO
      ENDDO
      CALL CLOSEGAMMA

      DEALLOCATE(GAMMA)
      DEALLOCATE(DENSITY_MATRIX,U)

      END SUBROUTINE LPRJ_LDApU


!******************** SUBROUTINE LPRJ_DEALLOC_COVL *********************
!      
!***********************************************************************

      SUBROUTINE LPRJ_DEALLOC_COVL()
      IF (ALLOCATED(LPRJ_COVL)) DEALLOCATE(LPRJ_COVL)
      RETURN
      END SUBROUTINE LPRJ_DEALLOC_COVL


!******************** LPRJ_LCAO ****************************************
!
!! In case locproj::lprj_lcao=.TRUE. VASP will call lcao::lcao_init
!! (from main.F).
!
!***********************************************************************

      FUNCTION LPRJ_LCAO()
      IMPLICIT NONE
      LOGICAL LPRJ_LCAO
      LPRJ_LCAO=LOCPROJ_NEEDS_PS
      END FUNCTION LPRJ_LCAO


!******************** SUBROUTINE CALC_OVERLAP_GN ***********************
!      
!! where i runs over a set of local functions at site R=POS with
!! l = [LSTART,LSTOP], and m = [1,2*l+1], and nb runs over all
!! orbitals at k-point k.
!!
!! For LNORM=.TRUE. the local functions themselves are S-normalized.
!! For LAUG=.FALSE. the augmentation part in the contraction of S(nb,i)
!! set to zero, the overlap operator is taken to be the identity
!! operator, S=1.
!
!***********************************************************************
      SUBROUTINE CALC_OVERLAP_GN( &
     &   LSTART,LSTOP,FG,POS,W,K,ISP,ISPINOR,P,CQIJ,LATT_CUR,T_INFO,LNORM,LAUG,S &
     &)
      USE pead
      USE poscar
      USE pseudo
      USE lattice
      USE full_kpoints
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE(wavespin) W
      TYPE(potcar) P(:)
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      INTEGER LSTART,LSTOP
      INTEGER ISP
      INTEGER ISPINOR
      REAL(q) K(3)
      REAL(q) FG(:,:,:)
      REAL(q) POS(3)
      OVERLAP CQIJ(:,:,:,:)
      GDEF S(:,:)
      LOGICAL LNORM,LAUG
      ! local variables
      TYPE(wavespin) WP
      TYPE(wavefuna) WK,WRYLM
      TYPE(wavedes1), TARGET :: WDESK
      TYPE(nonl_struct) NONL_S
            
      TYPE(rotation_handle), POINTER :: ROT_HANDLE

      GDEF C
      REAL(q) WSCAL
      INTEGER NK,NB,N,NYLM
      
      WP=W
      WP%WDES=>WDES_FULL_PEAD
            
      CALL CHECK_FULL_KPOINTS

      NULLIFY(ROT_HANDLE)      

      NYLM=(LSTOP+1)**2-LSTART**2

      ! search for kpoint k in BZ
      NK=KPOINT_IN_FULL_GRID(K,KPOINTS_FULL)
      CALL SETWDES(WP%WDES,WDESK,NK)
      IF (NK==KPOINTS_FULL%NEQUIV(NK)) THEN
         ! k is a kpoint in the IBZ
         WK=ELEMENTS(WP,WDESK,ISP)
      ELSE
         ! k is not a kpoint in the IBZ
         CALL NEWWAVA(WK,WDESK,WDESK%NBANDS)
         CALL PEAD_WA_ROTATE(WP,P,LATT_CUR,ISP,WK)
      ENDIF

      CALL NONL_ALLOC(NONL_S,T_INFO,P,WP%WDES,.FALSE.)
      CALL SPHER(WP%WDES%GRID,NONL_S,P,WP%WDES,LATT_CUR,1,NK)
      CALL PHASE(WP%WDES,NONL_S,NK)      

      CALL NEWWAVA(WRYLM,WDESK,NYLM); WRYLM%CW=0

      CALL CONSTRUCT_RYLM(LSTART,LSTOP,FG,LATT_CUR,POS,NONL_S,ISPINOR,WRYLM)
      ! and normalize the functions WRYLM
      IF (LNORM) THEN
         DO N=1,NYLM
            CALL CNORMN(ELEMENT(WRYLM,N),CQIJ,1,WSCAL)
         ENDDO
      ENDIF

      IF (.NOT.LAUG) WRYLM%CPROJ=0

      ! calculate overlap between Wk and WRYLM: < RYlm | S | w_{m,k1} >
      S=0    
      DO NB=1,WP%WDES%NBANDS
         DO N=1,NYLM
            C=W1_DOT(ELEMENT(WRYLM,N),ELEMENT(WK,NB),CQIJ)
            S(WDESK%NB_LOW+WDESK%NB_PAR*(NB-1),N)=C
!           WRITE(*,*) WDESK%NB_LOW+WDESK%NB_PAR*(NB-1),N,C
         ENDDO
      ENDDO

      CALLMPI( M_sum_g(WDESK%COMM_INTER,S(1,1),WDESK%NB_TOT*NYLM) )
      
      ! some deallocation to be done      
      CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

      CALL DELWAVA(WRYLM)
      IF (NK/=KPOINTS_FULL%NEQUIV(NK)) CALL DELWAVA(WK)      

      CALL NONL_DEALLOC(NONL_S)
            
      RETURN
      END SUBROUTINE CALC_OVERLAP_GN


!******************** SUBROUTINE CONSTRUCT_FUNCTION_RYlm ***************
!
!! space:
!!
!! < k+G | RYlm > = 1/sqrt(Omega) (-i)^l F(|k+G|) Y_lm(k+G) e^{-i(k+G)R}
!!
!! (see Sec. 10.5.1 of the thesis of gK). The plane wave part of these
!! local functions are stored in WRYLM%CW, and the projections onto
!! the PAW projectors in WRYLM%CPROJ.
!!
!! This is done for a set of local functions with l = [LSTART,LSTOP],
!! and m = [1, 2*l+1] (i.e., -l,..,l )
!
!***********************************************************************
      SUBROUTINE CONSTRUCT_RYLM( &
     &   LSTART,LSTOP,F,LATT_CUR,POS,NONL_S,ISPINOR,WRYLM &
     &)
      USE ini
      USE asa
      USE pead
      USE lattice
      USE constant
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE(latt) LATT_CUR
      TYPE(wavefuna) WRYLM
      TYPE (nonl_struct) NONL_S
      REAL(q) F(:,:,:)
      REAL(q) POS(3)
      INTEGER LSTART,LSTOP
      INTEGER ISPINOR
      ! local variables
      TYPE(wavefun1) W1
      INTEGER N1,N2,N3,IND,NPL,LMMAX,L,M,LM,LMBASE,IG
      REAL(q) G1,G2,G3,GKX,GKY,GKZ,FACTM,FAKT,FDER,GMAX
      REAL(q), ALLOCATABLE :: XS(:),YS(:),ZS(:),YLM(:,:)
      REAL(q), ALLOCATABLE :: G(:),FG(:)
      COMPLEX(q) CSET,CGDR
      COMPLEX(q), ALLOCATABLE :: CFAKTX(:)

      LMMAX=(LMAX+1)**2

      NPL=WRYLM%WDES1%NGVECTOR      

      IF (ISPINOR/=1.AND.(.NOT.WRYLM%WDES1%LNONCOLLINEAR)) THEN
         WRITE(*,*) 'CONSTRUCT_FUNCTION_RYlm: ERROR: ISPINOR=',ISPINOR,' but LNONCOLLINEAR=.FALSE.'
         STOP
      ENDIF

      ALLOCATE(G(NPL),FG(NPL))
      ALLOCATE(XS(NPL),YS(NPL),ZS(NPL),CFAKTX(NPL))

      ! loop over all G-vectors in the basis at this k-point
      DO IND=1,WRYLM%WDES1%NGVECTOR
         N1=MOD(WRYLM%WDES1%IGX(IND)+WRYLM%WDES1%GRID%NGX,WRYLM%WDES1%GRID%NGX)+1
         N2=MOD(WRYLM%WDES1%IGY(IND)+WRYLM%WDES1%GRID%NGY,WRYLM%WDES1%GRID%NGY)+1 
         N3=MOD(WRYLM%WDES1%IGZ(IND)+WRYLM%WDES1%GRID%NGZ,WRYLM%WDES1%GRID%NGZ)+1

         G1=(WRYLM%WDES1%GRID%LPCTX(N1)+WRYLM%WDES1%VKPT(1))
         G2=(WRYLM%WDES1%GRID%LPCTY(N2)+WRYLM%WDES1%VKPT(2))
         G3=(WRYLM%WDES1%GRID%LPCTZ(N3)+WRYLM%WDES1%VKPT(3))

         FACTM=1._q
         IF (WRYLM%WDES1%LGAMMA .AND. (N1/=1 .OR. N2/=1 .OR. N3/=1)) FACTM=SQRT(2._q)

         GKX=(G1*LATT_CUR%B(1,1)+G2*LATT_CUR%B(1,2)+G3*LATT_CUR%B(1,3))*TPI
         GKY=(G1*LATT_CUR%B(2,1)+G2*LATT_CUR%B(2,2)+G3*LATT_CUR%B(2,3))*TPI
         GKZ=(G1*LATT_CUR%B(3,1)+G2*LATT_CUR%B(3,2)+G3*LATT_CUR%B(3,3))*TPI

         G(IND)=MAX(SQRT(GKX*GKX+GKY*GKY+GKZ*GKZ),1E-10_q)

         ! phase factor e^{-i(k+G)R} where R is the origin
         ! of the localized function
         CGDR=CITPI*(G1*POS(1)+G2*POS(2)+G3*POS(3))
! test_
!        CGDR=CITPI*(WRYLM%WDES1%GRID%LPCTX(N1)*POS(1)+WRYLM%WDES1%GRID%LPCTY(N2)*POS(2)+WRYLM%WDES1%GRID%LPCTZ(N3)*POS(3))
!        CGDR=0
! test_
         CFAKTX(IND)=FACTM*EXP(-CGDR)


         XS(IND)  =GKX/G(IND)
         YS(IND)  =GKY/G(IND)
         ZS(IND)  =GKZ/G(IND)
      ENDDO

      ALLOCATE(YLM(NPL,LMMAX))
      ! get me all the Y_lm up to and including l=LMAX
      CALL SETYLM(LMAX,NPL,YLM,XS,YS,ZS)

      ! Setup the plane wave part of the desired function
      FAKT= 1/SQRT(LATT_CUR%OMEGA)
      CSET=CMPLX(0._q,-1._q,q)

      LMBASE=LSTART**2+1; LM=1
      DO L=LSTART,LSTOP
         ! get me the Bessel transform of the radial function
         GMAX=F(SIZE(F,1),1,L+1); FG=0
         DO IG=1,SIZE(G)
            IF (G(IG)>=GMAX) CYCLE
            CALL SPLVAL(G(IG),FG(IG),FDER,F(:,:,L+1),SIZE(F,1),SIZE(F,1))
         ENDDO
         DO IND=1,NPL
            DO M=1,2*L+1
               WRYLM%CW(IND+(ISPINOR-1)*NPL,LM+M-1)= &
              &   FAKT*(CSET**L)*CFAKTX(IND)*FG(IND)*YLM(IND,LMBASE+M-1)
            ENDDO
         ENDDO
         LM=LM+2*L+1; LMBASE=LMBASE+2*L+1
      ENDDO

      ! and get the projections of RYlm onto the PAW projectors
      WRYLM%CPROJ=0
      DO LM=1,SIZE(WRYLM%CPROJ,2)
         W1=ELEMENT(WRYLM,LM)
         CALL PROJ1(NONL_S,WRYLM%WDES1,W1)
      ENDDO

      DEALLOCATE(G,FG,XS,YS,ZS,CFAKTX,YLM)
      
      RETURN
      END SUBROUTINE CONSTRUCT_RYLM


!******************** SUBROUTINE CALC_PSPNL_SPLINE *********************
!      
!***********************************************************************
      SUBROUTINE CALC_PSPNL_SPLINE(P)
      USE pseudo

      TYPE (potcar) P(:)

      ! local variables
      INTEGER ITYP,IC,I

      IF (ALLOCATED(PSPNL_SPLINE)) RETURN

      ALLOCATE(PSPNL_SPLINE(0:NPSNL,5,MAXVAL(P(:)%LDIM),SIZE(P)))

      DO ITYP=1,SIZE(P)
         DO IC=1,P(ITYP)%LMAX
            DO I=0,NPSNL
               PSPNL_SPLINE(I,1,IC,ITYP)=P(ITYP)%PSMAXN/NPSNL*(I-1)
            ENDDO
            PSPNL_SPLINE(:,2,IC,ITYP)=P(ITYP)%PSPNL(:,IC)

            CALL SPLCOF(PSPNL_SPLINE(0,1,IC,ITYP),NPSNL+1,NPSNL+1,1.E+30_q)
         ENDDO
      ENDDO

      END SUBROUTINE CALC_PSPNL_SPLINE


!******************** FUNCTION FIND_ATOM_LCAO_ENTRY ********************
!      
!***********************************************************************
      FUNCTION FIND_ATOM_LCAO_ENTRY(ITYP,ICHANNEL)
      USE lcao, ONLY : ATOM_LCAO
      INTEGER FIND_ATOM_LCAO_ENTRY,ITYP,ICHANNEL
      ! local variables
      INTEGER I
      FIND_ATOM_LCAO_ENTRY=-1
      DO I=1,SIZE(ATOM_LCAO(ITYP)%L,1)
         IF (ATOM_LCAO(ITYP)%L(I,4)==ICHANNEL) THEN
            FIND_ATOM_LCAO_ENTRY=I; EXIT
         ENDIF
      ENDDO
      END FUNCTION FIND_ATOM_LCAO_ENTRY


!******************** SUBROUTINE CALC_HYDRO_SPLINE *********************
!      
!***********************************************************************
      SUBROUTINE CALC_HYDRO_SPLINE(N,ZA,LMAX,ENMAX)
      USE constant
      USE radial

      INTEGER N,LMAX
      REAL(q) ZA,ENMAX

      ! local variables
      INTEGER L
      REAL(q), ALLOCATABLE :: FTMP(:),FGTMP(:,:)

      TYPE (rgrid) R
      REAL(q) :: RSTART=0.0025_q,REND=10._q,H=0.025_q

      IF (ALLOCATED(HYDRO_SPLINE)) RETURN
      ALLOCATE(HYDRO_SPLINE(NMAX,5,LMAX+1))

      ! set up a logarithmic grid
      CALL SETRGRID(RSTART,REND,H,R)
      ALLOCATE(FTMP(R%NMAX),FGTMP(NMAX,5))
      ! construct hydrogenic radial functions
      CALL RADIAL_FUNCTION(N,R,ZA,FTMP)
      DO L=0,LMAX
         CALL BESSEL_TRANSFORM_RADIAL_FUNCTION(L,R,FTMP,REAL(SQRT(2._q*ENMAX/HSQDTM)/NMAX,KIND=q),HYDRO_SPLINE(:,:,L+1))
      ENDDO
 
      DEALLOCATE(FTMP,FGTMP)

      DEALLOCATE(R%R,R%SI); NULLIFY(R%R,R%SI)
      RETURN
      END SUBROUTINE CALC_HYDRO_SPLINE


!******************** SUBROUTINE RADIAL_FUNCTION ***********************
!      
!***********************************************************************
      SUBROUTINE RADIAL_FUNCTION( &
     & ITYP,R,ZA,FR &
     &)
      USE radial
      IMPLICIT NONE
      TYPE(rgrid) R
      INTEGER ITYP
      REAL(q) ZA
      REAL(q) FR(:)

      IF (SIZE(FR)<R%NMAX) THEN
         WRITE(*,*) 'RADIAL_FUNCTION: ERROR: FR too small:',SIZE(FR),R%NMAX
      ENDIF      
      IF (ITYP==1) THEN
         FR(:)=2._q*ZA**(3._q/2._q)*EXP(-ZA*R%R(:))
      ELSEIF (ITYP==2) THEN
         FR(:)=1._q/(2._q*SQRT(2._q))*ZA**(3._q/2._q)*(2._q-ZA*R%R(:))*EXP(-ZA*R%R(:)/2._q)
      ELSEIF (ITYP==3) THEN
         FR(:)=SQRT(4._q/27._q)*ZA**(3._q/2._q)* &
        &   (1._q-2._q/3._q*ZA*R%R(:)+2._q/27._q*ZA*ZA*R%R(:)*R%R(:))*EXP(-ZA*R%R(:)/3._q)
      ELSE
         WRITE(*,*) 'RADIAL_FUNCTION: TYPE does not exist',ITYP
      ENDIF
      
      RETURN
      END SUBROUTINE RADIAL_FUNCTION


!******************** SUBROUTINE SETRGRID ******************************
!      
!***********************************************************************
      SUBROUTINE SETRGRID( &
     & RSTART,REND,H,R &
     &)
      USE radial
      IMPLICIT NONE
      TYPE(rgrid) R
      REAL(q) RSTART,REND,H
      ! local variables
      INTEGER I
      
      I=0
      DO 
        I=I+1
        IF (RSTART*EXP(H*(I-1))>=REND) EXIT
      ENDDO
      ! make sure I is uneven
      I=I+(MODULO(I,2)+1)
      R%NMAX=I
      R%RSTART=RSTART; R%H=H; R%D=EXP(H)
      IF (ASSOCIATED(R%R)) THEN
         DEALLOCATE(R%R); NULLIFY(R%R)
      ENDIF
      ALLOCATE(R%R(I))
      DO I=1,R%NMAX
         R%R(I)=R%RSTART*EXP(H*(I-1))
      ENDDO      
      R%REND=R%R(R%NMAX)

      IF (ASSOCIATED(R%SI)) THEN
         DEALLOCATE(R%SI); NULLIFY(R%SI)
      ENDIF
      CALL SET_SIMP(R)

      RETURN
      END SUBROUTINE SETRGRID


!******************** SUBROUTINE LPRJ_ORBITAL_DEFINITIONS **************
!
! VASP sets up the YLM functions in the following order
! (see for instance SETYLM in asa.F)
! YLM(:,1)     -> s
! YLM(:,2:4)   -> p:= y, z, x
! YLM(:,5:9)   -> d:= xy, yz, z2, xz, x2
! YLM(:,10:16) -> f:= y(3x2-y2), xyz, yz2, z3, xz2, z(x2-y2), x(x2-3y2)           
!
! This routine provides a the aforementioned and adds the
! sp, sp2, sp3, sp3d and sp3d2 orbital definitions used in wannier90
! 
!***********************************************************************
      SUBROUTINE LPRJ_ORBITAL_DEFINITIONS( &
     &   L,M,HYBRID_ORBITAL &
     &)
      IMPLICIT NONE
      INTEGER L,M
      REAL(q) HYBRID_ORBITAL(:)
      ! local variables
      
      ! we should be able to deal with anything up to and including L=3
      IF (SIZE(HYBRID_ORBITAL)<16) THEN
         WRITE(*,*) 'LPRJ_ORBITAL_DEFINITIONS: ERROR: HYBRID_ORBITAL array too small', &
        &   SIZE(HYBRID_ORBITAL)
         STOP
      ENDIF
      
      HYBRID_ORBITAL=0
      
      SELECT CASE(L)
         CASE(0)
            ! s-function
            IF (M==1) THEN
               HYBRID_ORBITAL(1)=1._q
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(1)
            ! p-functions
            IF (M==1) THEN
               HYBRID_ORBITAL(2)=1._q
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(3)=1._q
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(4)=1._q
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(2)
            ! d-functions
            IF (M==1) THEN
               HYBRID_ORBITAL(5)=1._q
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(6)=1._q
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(7)=1._q
            ELSEIF (M==4) THEN
               HYBRID_ORBITAL(8)=1._q
            ELSEIF (M==5) THEN
               HYBRID_ORBITAL(9)=1._q
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(3)
            ! f-functions
            IF (M==1) THEN
               HYBRID_ORBITAL(10)=1._q
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(11)=1._q
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(12)=1._q
            ELSEIF (M==4) THEN
               HYBRID_ORBITAL(13)=1._q
            ELSEIF (M==5) THEN
               HYBRID_ORBITAL(14)=1._q
            ELSEIF (M==6) THEN
               HYBRID_ORBITAL(15)=1._q
            ELSEIF (M==7) THEN
               HYBRID_ORBITAL(16)=1._q
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(4:)
            ! nothing beyond L=3 yet
            WRITE(*,'(A,I2,A,I2,A)')  &
           &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
            STOP
         CASE(-1)
            ! sp-hybrids
            IF (M==1) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(4)= 1._q/SQRT(2._q)
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(2._q)
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(-2)
            ! sp2-hybrids
            IF (M==1) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)= 1._q/SQRT(2._q)
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)=-1._q/SQRT(2._q)
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)= 2._q/SQRT(6._q)
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(-3)
            ! sp3-hybrids
            IF (M==1) THEN
               HYBRID_ORBITAL(1)= 1._q/2._q
               HYBRID_ORBITAL(4)= 1._q/2._q
               HYBRID_ORBITAL(2)= 1._q/2._q
               HYBRID_ORBITAL(3)= 1._q/2._q
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(1)= 1._q/2._q
               HYBRID_ORBITAL(4)= 1._q/2._q
               HYBRID_ORBITAL(2)=-1._q/2._q
               HYBRID_ORBITAL(3)=-1._q/2._q
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(1)= 1._q/2._q
               HYBRID_ORBITAL(4)=-1._q/2._q
               HYBRID_ORBITAL(2)= 1._q/2._q
               HYBRID_ORBITAL(3)=-1._q/2._q
            ELSEIF (M==4) THEN
               HYBRID_ORBITAL(1)= 1._q/2._q
               HYBRID_ORBITAL(4)=-1._q/2._q
               HYBRID_ORBITAL(2)=-1._q/2._q
               HYBRID_ORBITAL(3)= 1._q/2._q
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(-4)
            ! spd3d-hybrids
            IF (M==1) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)= 1._q/SQRT(2._q)
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)=-1._q/SQRT(2._q)
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(3._q)
               HYBRID_ORBITAL(4)= 2._q/SQRT(6._q)
            ELSEIF (M==4) THEN
               HYBRID_ORBITAL(3)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)= 1._q/SQRT(2._q)
            ELSEIF (M==5) THEN
               HYBRID_ORBITAL(3)=-1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)= 1._q/SQRT(2._q)
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(-5)
            ! sp3d2-hybrids
            IF (M==1) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(4)=-1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)=-1._q/SQRT(12._q)
               HYBRID_ORBITAL(9)= 1._q/2._q
            ELSEIF (M==2) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(4)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)=-1._q/SQRT(12._q)
               HYBRID_ORBITAL(9)= 1._q/2._q
            ELSEIF (M==3) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)=-1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)=-1._q/SQRT(12._q)
               HYBRID_ORBITAL(9)=-1._q/2._q
            ELSEIF (M==4) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(2)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)=-1._q/SQRT(12._q)
               HYBRID_ORBITAL(9)=-1._q/2._q
            ELSEIF (M==5) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(3)=-1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)= 1._q/SQRT(3._q)
            ELSEIF (M==6) THEN
               HYBRID_ORBITAL(1)= 1._q/SQRT(6._q)
               HYBRID_ORBITAL(3)= 1._q/SQRT(2._q)
               HYBRID_ORBITAL(7)= 1._q/SQRT(3._q)
            ELSE
               WRITE(*,'(A,I2,A,I2,A)')  &
              &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
               STOP
            ENDIF
         CASE(:-6)
            WRITE(*,'(A,I2,A,I2,A)')  &
           &   'LPRJ_ORBITAL_DEFINITIONS: ERROR: L=',L,' M=',M,' not implemented'
            STOP
         
      END SELECT
      
      RETURN
      END SUBROUTINE LPRJ_ORBITAL_DEFINITIONS


      END MODULE locproj
