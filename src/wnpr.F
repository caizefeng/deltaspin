#include "symbol.inc"
      MODULE wnpr
      USE prec
      USE pead, ONLY : PEAD_REQUEST

      IMPLICIT NONE

      PRIVATE :: FUNCS_PER_TYPE,GET_ATOMIC_FUNCTIONS,ROTATE_ORBITALS_FULLK, &
     &           WRITE_PROJECTIONS,ROTATE_PROJECTIONS,CALCULATE_PROJECTIONS

      ! number of projected Wannier functions
      INTEGER, SAVE :: WNPR_num_wann

      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_incar_i(:)
      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_incar_l(:)

      INTEGER, SAVE :: WNPR_num_sites
      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_index(:)

      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_l(:)
      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_m(:)
      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_t(:)
      INTEGER, ALLOCATABLE, PRIVATE, SAVE :: WNPR_i(:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: WNPR_r(:,:)

      REAL(q), PRIVATE, SAVE :: window(2)
      LOGICAL, ALLOCATABLE, SAVE :: WNPR_use_band(:,:,:)

      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: Atomic_funcs(:,:,:,:)
      LOGICAL, ALLOCATABLE, PRIVATE, SAVE :: Atomic_funcs_l(:,:) 

      COMPLEX(q), ALLOCATABLE, SAVE :: WNPR_projections(:,:,:,:)

      INTEGER, PARAMETER, PRIVATE :: LMAX=3
 
      INTEGER, PARAMETER, PRIVATE :: NMAX=1000

      ! switch on the projected Wannier functions functionality
      LOGICAL, SAVE :: LWANPROJ=.FALSE.

      CONTAINS

!******************** SUBROUTINE WNPR_READER ***************************
!      
!***********************************************************************

      SUBROUTINE WNPR_READER(NIONS,IU0,IU5)
      USE base
      USE vaspxml
      USE full_kpoints
      INTEGER NIONS
      INTEGER IU0,IU5
      ! local variables
      INTEGER ITMP(NIONS)
      ! reader related
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      CALL RDATAB(LOPEN,INCAR,IU5,'WANPROJ','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LWANPROJ,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''WANPROJ'' from file INCAR.'
         LWANPROJ=.FALSE.
      ENDIF
      CALL XML_INCAR('WANPROJ','L',IDUM,RDUM,CDUM,LWANPROJ,CHARAC,N)

      ! early exit if possible
      IF (WANPROJ()) THEN
         ! switch on PEAD routines
         CALL PEAD_REQUEST
 
         ! read the ids of the atomic sites on which we rquire
         ! projected Wannier functions to be constructed
         CALL RDATAB(LOPEN,INCAR,IU5,'WANPROJ_I','=','#',';','I', &
        &            ITMP,RDUM,CDUM,LDUM,CHARAC,N,NIONS,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''WANPROJ_I'' from file INCAR.'
            LWANPROJ=.FALSE.
         ENDIF
         ! if WANPROJ_I was correctly specified in the INCAR file,
         ! then N is the number of sites on which we want to construct
         ! projected Wannier functions, and ITMP(1:N) contains their
         ! site ids 
         IF (IERR==0) THEN
            WNPR_num_sites=N
            ALLOCATE(WNPR_incar_i(N))
            WNPR_incar_i=ITMP(1:N) 
            CALL XML_INCAR_V('WANPROJ_I','I',WNPR_incar_i,RDUM,CDUM,LDUM,CHARAC,N)
         ENDIF
 
         ! if no projected Wannier functions are to be constructed
         ! then there is no need for further reading, else ...
         further: IF (WNPR_num_sites>0) THEN
 
         ! read the type of projected Wannier functions that should
         ! be constructed (one entry for each of the WNPR_num_wann sites)
         ALLOCATE(WNPR_incar_l(WNPR_num_sites))
         CALL RDATAB(LOPEN,INCAR,IU5,'WANPROJ_L','=','#',';','I', &
        &            WNPR_incar_l,RDUM,CDUM,LDUM,CHARAC,N,WNPR_num_sites,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<WNPR_num_sites))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''WANPROJ_L'' from file INCAR.'
            LWANPROJ=.FALSE.
         ENDIF
         CALL XML_INCAR_V('WANPROJ_L','I',WNPR_incar_l,RDUM,CDUM,LDUM,CHARAC,N)
 
         ! read the energy window that defines the Bloch bands that are to
         ! be used in the construction of the projected Wannier functions
         CALL RDATAB(LOPEN,INCAR,IU5,'WANPROJ_E','=','#',';','F', &
        &            IDUM,window,CDUM,LDUM,CHARAC,N,2,IERR)
         IF (((IERR/=0).AND.(IERR/=3))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''WANPROJ_E'' from file INCAR.'
         ENDIF
         ! if WANPROJ_E was not (or at least not correctly) specified
         ! then the energy window should include all bands by default:
         ! (-1E+10_q,1E+10) should do the trick
         IF ((IERR==0.AND.N<2).OR.IERR==3) THEN
            window(1)=-1E+10_q; window(2)=1E+10_q            
         ENDIF 
         CALL XML_INCAR_V('WANPROJ_E','F',IDUM,window,CDUM,LDUM,CHARAC,2)
 
         ENDIF further
  
         !
         CALL USE_FULL_KPOINTS
      ENDIF

      CLOSE(IU5)

      RETURN
      END SUBROUTINE WNPR_READER


!***********************************************************************
!
! WANPROJ
!
!***********************************************************************
      FUNCTION WANPROJ()
      IMPLICIT NONE
      LOGICAL WANPROJ
      WANPROJ=LWANPROJ
      END FUNCTION WANPROJ


!******************** SUBROUTINE WNPR_PROJECT **************************
!      
!***********************************************************************

      SUBROUTINE WNPR_PROJECT(W,WDES,T_INFO,INFO,P,CQIJ,LATT_CUR,IO)
      USE base
      USE constant
      USE poscar
      USE lattice
      USE wave_high
      USE full_kpoints
      USE pseudo
      TYPE (wavespin) W 
      TYPE (wavedes) WDES
      TYPE (type_info) T_INFO
      TYPE (info_struct) INFO
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (latt) LATT_CUR
      TYPE (in_struct) IO

      OVERLAP CQIJ(:,:,:,:)

      ! local variables
      INTEGER I,J,N,NF
      INTEGER ISP,IK,IB

      ! early exit if possible
      IF (.NOT.WANPROJ().OR.WNPR_num_sites==0) RETURN

      ! this need to be in order to proceed as well
      CALL CHECK_FULL_KPOINTS

      ! WNPR_num_sites holds the number of sites on which projected Wannier
      ! functions are to be constructed, and not the actual number of functions
      ! ... calculate the number of functions WNPR_num_wann
      NF=0
      DO I=1,WNPR_num_sites
         N=FUNCS_PER_TYPE(WNPR_incar_l(I))
         IF (N==0) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*) 'WNPR_PROJECT: ERROR: unknown type designator:',WNPR_incar_l(I)
            STOP
         ENDIF
         NF=NF+N 
      ENDDO
      ! store the number of Wannier functions
      WNPR_num_wann=NF

      ALLOCATE(WNPR_index(WNPR_num_sites+1))
      ALLOCATE(WNPR_r(3,WNPR_num_wann),WNPR_l(WNPR_num_wann),WNPR_m(WNPR_num_wann),WNPR_t(WNPR_num_wann),WNPR_i(WNPR_num_wann))
      WNPR_t=-1; WNPR_i=-1; NF=0; WNPR_index(1)=1
      DO I=1,WNPR_num_sites
         N=FUNCS_PER_TYPE(WNPR_incar_l(I))
         DO J=1,N
            NF=NF+1
            ! specify the angular part of the projection functions
            WNPR_l(NF)=WNPR_incar_l(I)
            WNPR_m(NF)=J
            ! set the position of the Wannier center
            IF (I<=SIZE(WNPR_incar_i)) THEN
               ! the position of the Wannier function may be specified
               ! through the index of an atom in the POSCAR file
               WNPR_r(1:3,NF)=T_INFO%POSION(1:3,WNPR_incar_i(I))
               ! in this case store the atomic type as well since we
               ! will use this information to generate the radial part
               ! of the projection functions
               WNPR_t(NF)=T_INFO%ITYP(WNPR_incar_i(I))
               ! and the id of the ionic site
               WNPR_i(NF)=WNPR_incar_i(I)
            ELSE
               ! or the position is defined differently (in future)
               WRITE(*,*) 'WNPR_PROJECT: alternatives not implemented yet, sorry ...'
               STOP
            ENDIF
         ENDDO
         WNPR_index(I+1)=WNPR_index(I)+N
      ENDDO

      ! determine which Bloch orbitals should be used to construct the
      ! projected Wannier functions
      ALLOCATE(WNPR_use_band(WDES%NB_TOT,KPOINTS_FULL%NKPTS,WDES%ISPIN)); WNPR_use_band=.FALSE. 
      ! use all bands that lie within the energy window
      DO ISP=1,WDES%ISPIN
         DO IK=1,KPOINTS_FULL%NKPTS
            DO IB=1,WDES%NB_TOT
               IF (REAL(W%CELTOT(IB,KPOINTS_FULL%NEQUIV(IK),ISP),q)>=window(1).AND. &
              &   REAL(W%CELTOT(IB,KPOINTS_FULL%NEQUIV(IK),ISP),q)<=window(2)) WNPR_use_band(IB,IK,ISP)=.TRUE.
            ENDDO
         ENDDO
      ENDDO

      ! setup the radial atomic functions if needed
      funcs: DO J=1,WNPR_num_wann
         ! even if we need them for only one function, we simply generate
         ! the radial part of the atomic pseudo wave functions for all
         ! atomic types and be done with it
         IF (WNPR_t(J)>0) THEN
            CALL GET_ATOMIC_FUNCTIONS(T_INFO,INFO,WDES%ENMAX,P,IO)
            EXIT funcs
         ENDIF 
      ENDDO funcs

      CALL CALCULATE_PROJECTIONS(W,WDES,T_INFO,INFO,P,CQIJ,LATT_CUR,IO)

      DEALLOCATE(Atomic_funcs,Atomic_funcs_l)

! ... write the projections to file
      IF (IO%IU6>=0) CALL WRITE_PROJECTIONS(W,99)

! ... rotate the projections so that the onsite density matrices are diagonal
!     CALL ROTATE_PROJECTIONS(W,WDES,IO%IU0)

      ! and the final cleanup
      DEALLOCATE(WNPR_r,WNPR_l,WNPR_m,WNPR_t,WNPR_i)
      DEALLOCATE(WNPR_index)

      DEALLOCATE(WNPR_projections)
      DEALLOCATE(WNPR_use_band)

      RETURN
      END SUBROUTINE WNPR_PROJECT


!******************** SUBROUTINE WNPR_ROTATE_ORBITALS ******************
!      
!***********************************************************************

      SUBROUTINE WNPR_ROTATE_ORBITALS(&
     &   WDES,W,KPOINTS,GRID,T_INFO,P,NONL_S,SYMM,LATT_CUR,LATT_INI,DYN,LMDIM,CQIJ,IO)
      USE base
      USE lattice
      USE pseudo
      USE poscar
      USE mgrid
      USE msymmetry
      USE nonl_high
      USE wave_high
      USE kpoints_change
      USE full_kpoints
      IMPLICIT NONE
      TYPE (wavedes) WDES
      TYPE (wavespin) W
      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d) GRID
      TYPE (type_info) T_INFO
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (nonl_struct) NONL_S
      TYPE (symmetry) SYMM
      TYPE (latt) LATT_CUR
      TYPE (latt) LATT_INI
      TYPE (dynamics) DYN
      TYPE (in_struct) IO

      OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      INTEGER LMDIM

      ! local variables
      TYPE (wavedes) WNPR_WDES
      TYPE (wavespin) WNPR_W

      ! early exit if possible
      IF (.NOT.WANPROJ().OR.WNPR_num_sites==0) RETURN

      ! generate wave functions on full k-point grid
      IF (SYMM%ISYM>=0.AND.(.NOT.WDES%LGAMMA)) THEN
         ! switch of symmetry
         CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
        &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,WDES%ISPIN,-1)
         ! reread k-points with LINVERSION=.FALSE. to generate full mesh
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR,.FALSE., &
        &   T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,IO%IU0)
         CALL KPAR_SYNC_ALL(WDES,W)
         CALL RE_GEN_LAYOUT(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,-1,IO%IU0)
         CALL REALLOCATE_WAVE(W,GRID,WDES,NONL_S,T_INFO,P,LATT_CUR)
      ENDIF     
 
      CALL SETWNDES(WDES,WNPR_WDES); CALL ALLOCW(WNPR_WDES,WNPR_W)
      CALL ROTATE_ORBITALS_FULLK(W,WDES,WNPR_W,WNPR_WDES)
! test_
      CALL WNPR_TEST(W,WDES,WNPR_W,WNPR_WDES,LMDIM,CQIJ,IO%IU0)
! test_

      ! restore wave functions on symmetry reduced k-point grid
      IF (SYMM%ISYM>=0.AND.(.NOT.WDES%LGAMMA)) THEN
      IF (SYMM%ISYM>0) &
         CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
              T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
              SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
              SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,-1)
#ifdef oldsym
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,IO%IU0)
#else
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,IO%IU0)
#endif
         CALL KPAR_SYNC_ALL(WDES,W)
         CALL RE_GEN_LAYOUT(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,-1,IO%IU0)
         CALL REALLOCATE_WAVE(W,GRID,WDES,NONL_S,T_INFO,P,LATT_CUR)
      ENDIF

      RETURN
      END SUBROUTINE WNPR_ROTATE_ORBITALS


!******************** SUBROUTINE WNPR_ROTATE_ORBITALS_NOSYMM ***********
!      
!***********************************************************************

      SUBROUTINE WNPR_ROTATE_ORBITALS_NOSYMM(W,WDES,WP,WDESP)
      USE wave_high
      USE full_kpoints
      USE kpoints_change
      USE dfast, ONLY : NBLK
      TYPE (wavedes) WDES,WDESP
      TYPE (wavespin) W,WP
      ! local variables
      TYPE (wavedes1) WDESK,WDESKP
      TYPE (wavefuna) WA,WAP
      INTEGER ISP,ISPINOR,IK,IKP,I,IFNC

      COMPLEX(q), ALLOCATABLE :: CBLOCK(:,:)
      GDEF, ALLOCATABLE :: GBLOCK(:,:)
      INTEGER IBLK,NBLK_ACT

      GDEF, ALLOCATABLE :: U(:,:)

      CALL CHECK_FULL_KPOINTS

      ALLOCATE(U(WDES%NB_TOT,WDESP%NB_TOT),CBLOCK(NBLK,WDES%NB_TOT),GBLOCK(NBLK,WDES%NB_TOT))

      spin: DO ISP=1,WDES%ISPIN
         kpoints: DO IK=1,WDES%NKPTS
            ! find the corresponding entry in KPOINTS_FULL_ORIG, that contains
            ! the set of k-points used in the computation of the rotation matrices
            DO IKP=1,KPOINTS_FULL_ORIG%NKPTS
               IF (LIDENTICAL_KPOINT(WDES%VKPT(:,IK),KPOINTS_FULL%VKPT(:,IKP))) EXIT 
            ENDDO
            IF (IKP>KPOINTS_FULL%NKPTS) THEN
               WRITE(*,*) 'WNPR_ROTATE_ORBITALS_NOSYMM: ERROR: no matching k-point found in KPOINTS_FULL',IK
               STOP
            ENDIF

            ! setup the rotation matrix U
            U=zero
            sites: DO IFNC=1,WNPR_num_wann
            bands: DO I=1,WDES%NB_TOT
               IF (.NOT.WNPR_use_band(I,IKP,ISP)) CYCLE bands
               DO ISPINOR=0,WDES%NRSPINORS-1
                  U(I,IFNC+WNPR_num_wann*ISPINOR)=WNPR_projections(I,IKP,MAX(ISP,ISPINOR+1),IFNC)
               ENDDO
            ENDDO bands
            ENDDO sites

            CALL SETWDES(WDES,WDESK,IK)
            WA=ELEMENTS(W,WDESK,ISP)

            CALL SETWDES(WDESP,WDESKP,IK)
            WAP=ELEMENTS(WP,WDESKP,ISP)

            ! redistribute over plane wave coefficients
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP); CALL REDISTRIBUTE_PROJ(WAP)

            WAP%CW_RED=(0._q,0._q)
            DO IBLK=0,WDESK%NPL_RED-1,NBLK
               NBLK_ACT=MIN(NBLK,WDESK%NPL_RED-IBLK)

               CBLOCK(1:NBLK_ACT,1:WDES%NB_TOT)=WA%CW_RED(IBLK+1:IBLK+NBLK_ACT,1:WDES%NB_TOT)

               CALL GGEMM(&
              &   'N','N',m_ NBLK_ACT,WDESP%NB_TOT,WDES%NB_TOT, & 
              &   one,CBLOCK(1,1),m_ NBLK,U(1,1),WDES%NB_TOT, &
              &   zero,WAP%CW_RED(IBLK+1,1),m_ WDESKP%NRPLWV_RED  &
              &   )

            ENDDO

            IF (WDESK%NPRO_RED/=0) THEN
               WAP%CPROJ_RED=zero
               DO IBLK=0,WDESK%NPRO_RED-1,NBLK
                  NBLK_ACT=MIN(NBLK,WDESK%NPRO_RED-IBLK)

                  GBLOCK(1:NBLK_ACT,1:WDES%NB_TOT)=WA%CPROJ_RED(IBLK+1:IBLK+NBLK_ACT,1:WDES%NB_TOT)

                  CALL GGEMM(&
                 &   'N','N',WDESK%NPRO_RED,WDESP%NB_TOT,WDES%NB_TOT, & 
                 &   one,GBLOCK(1,1),NBLK,U(1,1),WDES%NB_TOT, &
                 &   zero,WAP%CPROJ_RED(IBLK+1,1),WDESKP%NPROD_RED  &
                 &   )
               ENDDO
            ENDIF

            ! redistribute back over bands
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP); CALL REDISTRIBUTE_PROJ(WAP)
         ENDDO kpoints
      ENDDO spin

      DEALLOCATE(U,CBLOCK,GBLOCK)

      RETURN
      END SUBROUTINE WNPR_ROTATE_ORBITALS_NOSYMM


!******************** SUBROUTINE WNPR_TEST *****************************
!      
!***********************************************************************
#define debug
      SUBROUTINE WNPR_TEST(W,WDES,WP,WDESP,LMDIM,CQIJ,IU0)
      USE wave_high
      USE full_kpoints
      USE dfast, ONLY : NSTRIP_STANDARD_GLOBAL
      TYPE (wavedes) WDES,WDESP
      TYPE (wavespin) W,WP
      OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      INTEGER LMDIM,IU0
      ! local variables
      TYPE (wavedes1) WDESK,WDESKP
      TYPE (wavefuna) WA,WAP,WOVL
      COMPLEX(q), ALLOCATABLE :: CBLOCK(:,:)
      GDEF, ALLOCATABLE :: GBLOCK(:,:),COVL(:,:)
      INTEGER ISP,IK,NPOS
      INTEGER NSTRIP,ISTRIP,NSTRIP_ACT
#ifdef debug
      INTEGER I,J
#endif
      ALLOCATE(COVL(WDES%NB_TOT,WDESP%NB_TOT))

      CALL SETWDES(WDESP,WDESKP,0)
      CALL NEWWAVA_PROJ(WOVL,WDESKP)

      NSTRIP=NSTRIP_STANDARD_GLOBAL

      spin: DO ISP=1,WDES%ISPIN
         kpoint: DO IK=1,WDES%NKPTS

            CALL SETWDES(WDES,WDESK,IK)
            WA=ELEMENTS(W,WDESK,ISP)
           
            CALL SETWDES(WDESP,WDESKP,IK)
            WAP=ELEMENTS(WP,WDESKP,ISP)
 
            CALL OVERL(WDESKP,WDESKP%LOVERL,LMDIM,CQIJ(1,1,1,ISP),WAP%CPROJ(1,1),WOVL%CPROJ(1,1))
           
            ! redistribute over plane wave coefficients
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP); CALL REDISTRIBUTE_PROJ(WOVL)
           
            COVL=zero

!           CALL GGEMM(&
!          &   trans,'N',WDESK%NB_TOT,WDESKP%NB_TOT,m_ WDESK%NPL_RED, &
!          &   one,WA%CW_RED,m_ WDESK%NRPLWV_RED,WAP%CW_RED,m_ WDESKP%NRPLWV_RED, &
!          &   zero,COVL,WDESK%NB_TOT &
!          &   )

            ALLOCATE(CBLOCK(WDESK%NPL_RED,NSTRIP))
            DO ISTRIP=0,WDESK%NB_TOT-1,NSTRIP
               NSTRIP_ACT=MIN(NSTRIP,WDESK%NB_TOT-ISTRIP)

               CBLOCK(1:WDESK%NPL_RED,1:NSTRIP_ACT)=WA%CW_RED(1:WDESK%NPL_RED,ISTRIP+1:ISTRIP+NSTRIP_ACT)

               CALL GGEMM(&
              &   trans,'N',NSTRIP_ACT,WDESKP%NB_TOT,m_ WDESK%NPL_RED, &
              &   one,CBLOCK(1,1),m_ WDESK%NPL_RED,WAP%CW_RED,m_ WDESKP%NRPLWV_RED, &
              &   zero,COVL(ISTRIP+1,1),WDESK%NB_TOT &
              &   )

            ENDDO
            DEALLOCATE(CBLOCK)

!           IF (WDESK%NPRO_O_RED/=0) &
!           CALL GGEMM(&
!          &   trans,'N',WDESK%NB_TOT,WDESKP%NB_TOT,WDESK%NPRO_O_RED, &
!          &   one,WA%CPROJ_RED,WDESK%NPROD_RED,WOVL%CPROJ_RED,WDESKP%NPROD_RED, &
!          &   one,COVL,WDESK%NB_TOT &
!          &   )

            IF (WDESK%NPRO_O_RED/=0) THEN
               ALLOCATE(GBLOCK(WDESK%NPRO_O_RED,NSTRIP))
               DO ISTRIP=0,WDESK%NB_TOT-1,NSTRIP
                  NSTRIP_ACT=MIN(NSTRIP,WDESK%NB_TOT-ISTRIP)

                  GBLOCK(1:WDESK%NPRO_O_RED,1:NSTRIP_ACT)=WA%CPROJ_RED(1:WDESK%NPRO_O_RED,ISTRIP+1:ISTRIP+NSTRIP_ACT)

                  CALL GGEMM(&
                 &   trans,'N',NSTRIP_ACT,WDESKP%NB_TOT,WDESK%NPRO_O_RED, &
                 &   one,GBLOCK(1,1),WDESK%NPRO_O_RED,WOVL%CPROJ_RED,WDESKP%NPROD_RED, &
                 &   one,COVL(ISTRIP+1,1),WDESK%NB_TOT &
                 &   )
               ENDDO
               DEALLOCATE(GBLOCK)
            ENDIF

            CALLMPI( M_sum_g(WDES%COMM,COVL(1,1),WDES%NB_TOT*WDESP%NB_TOT))
#ifdef debug
            IF (IU0>=0) THEN
               WRITE(IU0,'(/X,A,I5,X,A,I4)') 'kpoint:',IK,'spin:',ISP
               DO I=1,WDESK%NB_TOT
                  WRITE(IU0,'(3X,A,I5)',ADVANCE='No') 'band:',I
                  DO J=1,WNPR_num_wann*WDESKP%NRSPINORS
                     WRITE(IU0,'(2X,F7.3,X,F7.3)',ADVANCE='No') COVL(I,J)
                  ENDDO
                  WRITE(IU0,*)
               ENDDO
            ENDIF
#endif
            ! redistribute back over bands
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP)
         ENDDO kpoint
      ENDDO spin

      CALL DELWAVA_PROJ(WOVL)
      DEALLOCATE(COVL)

      RETURN
      END SUBROUTINE WNPR_TEST
#undef debug

!******************** SUBROUTINE GET_ATOMIC_FUNCTIONS ******************
!      
!***********************************************************************

      SUBROUTINE GET_ATOMIC_FUNCTIONS(T_INFO,INFO,ENMAX,P,IO)
      USE base
      USE pseudo
      USE poscar
      USE constant
      USE vaspxml
      USE lcao, ONLY : ATOM_LCAO
!     USE lcao, ONLY : RCUT,ATOM_LCAO,LCAO_INIT
      USE radial, ONLY : BESSEL_TRANSFORM_RADIAL_FUNCTION
      TYPE (type_info) T_INFO
      TYPE (info_struct) INFO
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (in_struct) IO
      REAL(q) ENMAX
      ! local variables
      REAL(q), ALLOCATABLE :: FTMP(:),FGTMP(:,:),FR(:,:),TMP(:),FG(:,:,:)

      INTEGER NT,I

!     ! the routine LCAO_INIT needs RCUT to be set: normally this
!     ! is read in by LCAO_READER, but since we do not want to call
!     ! that particular routine, we set RCUT here
!     ! (to a common constant for all atomic types)
!     ALLOCATE(RCUT(T_INFO%NTYP))
!     RCUT=10._q
!     ! solve the atomic PAW equations
!     CALL LCAO_INIT(P,INFO,ENMAX,IO%IU0,IO%IU6)

      IF (ALLOCATED(Atomic_funcs)) DEALLOCATE(Atomic_funcs)
      ALLOCATE(Atomic_funcs(NMAX,5,4,T_INFO%NTYP))
      Atomic_funcs=0._q

      IF (ALLOCATED(Atomic_funcs_l)) DEALLOCATE(Atomic_funcs_l)
      ALLOCATE(Atomic_funcs_l(4,T_INFO%NTYP))
      Atomic_funcs_l=.FALSE.

      CALL XML_TAG("atomic")
      DO NT=1,T_INFO%NTYP
         ALLOCATE(FTMP(ATOM_LCAO(NT)%FR%NMAX),FGTMP(NMAX,5))
         ALLOCATE(FR(ATOM_LCAO(NT)%OCCUPIED_VALENCE_STATES+1,ATOM_LCAO(NT)%FR%NMAX),TMP(ATOM_LCAO(NT)%OCCUPIED_VALENCE_STATES+1))
         FR(1,:)=ATOM_LCAO(NT)%FR%R(:)
         DO I=1,ATOM_LCAO(NT)%OCCUPIED_VALENCE_STATES
            FTMP(:)=ATOM_LCAO(NT)%WFCT_L(:,2,I)/(ATOM_LCAO(NT)%FR%R(:)) !**(ATOM_LCAO(NT)%L(I,1)+1))
            FR(I+1,:)=ATOM_LCAO(NT)%WFCT_L(:,2,I)
            CALL BESSEL_TRANSFORM_RADIAL_FUNCTION( &
           &   ATOM_LCAO(NT)%L(I,1),ATOM_LCAO(NT)%FR,FTMP,REAL(SQRT(2._q*ENMAX/HSQDTM)/NMAX,KIND=q),FGTMP)
! test_
!           IF (.NOT.Atomic_funcs_l(ATOM_LCAO(NT)%L(I,1)+1,NT)) THEN
!              Atomic_funcs(:,:,ATOM_LCAO(NT)%L(I,1)+1,NT)=FGTMP(:,:)
!              Atomic_funcs_l(ATOM_LCAO(NT)%L(I,1)+1,NT)=.TRUE.
!           ENDIF
            Atomic_funcs(:,:,ATOM_LCAO(NT)%L(I,1)+1,NT)=FGTMP(:,:)
            Atomic_funcs_l(ATOM_LCAO(NT)%L(I,1)+1,NT)=.TRUE.
! test_
         ENDDO
         CALL XML_TAG("set", comment="type "//TRIM(ADJUSTL(ATOM_LCAO(NT)%ELEMENT)))
         DO I=1,SIZE(FR,2)
            TMP(:)=FR(:,I)
            CALL XML_ROW_DATA(TMP)
         ENDDO
         CALL XML_CLOSE_TAG("set")
         DEALLOCATE(FTMP,FGTMP,FR,TMP)
      ENDDO
      CALL XML_CLOSE_TAG("atomic")

!     DEALLOCATE(RCUT)

      RETURN
      END SUBROUTINE GET_ATOMIC_FUNCTIONS


!******************** SUBROUTINE CALCULATE_PROJECTIONS *****************
!      
!***********************************************************************

      SUBROUTINE CALCULATE_PROJECTIONS(W,WDES,T_INFO,INFO,P,CQIJ,LATT_CUR,IO)
      USE base
      USE constant
      USE wave_high
      USE full_kpoints
      USE lattice
      USE pseudo
      USE poscar
      USE radial
      USE mlwf, ONLY : SETRGRID,RADIAL_FUNCTION,WANNIER90_ORBITAL_DEFINITIONS, &
     &                 SETROTYLM,CALC_OVERLAP_GN
      TYPE (wavespin) W
      TYPE (wavedes) WDES
      TYPE (type_info) T_INFO
      TYPE (info_struct) INFO
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (latt) LATT_CUR
      TYPE (in_struct) IO

      OVERLAP CQIJ(:,:,:,:)

      ! local variables
      GDEF, ALLOCATABLE :: A(:,:),AP(:,:)
      REAL(q), ALLOCATABLE :: ROTYLM(:,:)
      REAL(q), ALLOCATABLE :: HYBRID_ORBITAL(:)

      COMPLEX(q) CPROJ

      INTEGER IFNC,L,ISP,ISPINOR,IK,IB,M,N,NP

      REAL(q), ALLOCATABLE :: FTMP(:),FGTMP(:,:),FG(:,:,:)

      TYPE (rgrid) R
      REAL(q) :: RSTART=0.0025_q,REND=10._q,H=0.025_q

      REAL(q) :: xaxis(3)=(/1._q,0._q,0._q/),zaxis(3)=(/0._q,0._q,1._q/)

      LOGICAL LDONE

      CALL CHECK_FULL_KPOINTS

      IF (ALLOCATED(WNPR_projections)) DEALLOCATE(WNPR_projections)
      ALLOCATE(WNPR_projections(WDES%NB_TOT,KPOINTS_FULL%NKPTS,MAX(WDES%ISPIN,WDES%NRSPINORS),WNPR_num_wann))

      ALLOCATE(A(WDES%NB_TOT,(LMAX+1)**2),AP(WDES%NB_TOT,(LMAX+1)**2))
      ALLOCATE(ROTYLM((LMAX+1)**2,(LMAX+1)**2),HYBRID_ORBITAL((LMAX+1)**2))

      ! just a precaution, in case the compiler does not
      ! nullify pointers by default
      R%R=>NULL(); R%SI=>NULL()

      sites: DO IFNC=1,WNPR_num_wann
         ! allocate space to hold the radial functions
         ! (interpolating tables in reciprocal space)
         ALLOCATE(FG(NMAX,5,LMAX+1))
         ! loop over all l-quantum numbers
         DO L=0,LMAX
            LDONE=.FALSE.
            ! do we use the atomic radial function?
            IF (WNPR_t(IFNC)>0) THEN 
               ! if so, is it available?
               IF (Atomic_funcs_l(L+1,WNPR_t(IFNC))) THEN
                  FG(:,:,L+1)=Atomic_funcs(:,:,L+1,WNPR_t(IFNC))
                  LDONE=.TRUE.
               ENDIF
            ENDIF
            ! if we do not take the atomic radial function or if that function
            ! is not available, then we use hydrogenic radial functions
            IF (.NOT.LDONE) THEN
               ! set up a radial logarithmic grid
               CALL SETRGRID(RSTART,REND,H,R)
               ALLOCATE(FTMP(R%NMAX),FGTMP(NMAX,5))
               ! construct a hydrogenic radial function
               CALL RADIAL_FUNCTION(1,R,1._q,FTMP)
               CALL BESSEL_TRANSFORM_RADIAL_FUNCTION(L,R,FTMP,REAL(SQRT(2._q*INFO%ENMAX/HSQDTM)/NMAX,KIND=q),FGTMP)
               FG(:,:,L+1)=FGTMP(:,:)
               DEALLOCATE(FTMP,FGTMP)
            ENDIF
         ENDDO
         IF (ASSOCIATED(R%R)) THEN
            DEALLOCATE(R%R,R%SI); NULLIFY(R%R,R%SI)
         ENDIF

         ! we will use the angular orbital definitions of wannier90
         CALL WANNIER90_ORBITAL_DEFINITIONS(WNPR_l(IFNC),WNPR_m(IFNC),HYBRID_ORBITAL)

         ! setup the Ylm rotation matrix in accordance with xaxis and zaxis
         ! (assumed to be 1,0,0 and 0,0,1 for all functions for now)
         CALL SETROTYLM(xaxis,zaxis,LMAX,ROTYLM)

         spin: DO ISP=1,WDES%ISPIN
         kpoint: DO IK=1,KPOINTS_FULL%NKPTS
         spinor: DO ISPINOR=1,WDES%NRSPINORS

            CALL CALC_OVERLAP_GN( &
           &   LMAX,FG,WNPR_r(:,IFNC),W,KPOINTS_FULL%VKPT(:,IK),ISP,ISPINOR,P,CQIJ,LATT_CUR,T_INFO,A)

            ! and rotate them in accordance with zaxis and xaxis
            AP=0
            DO M=1,WDES%NB_TOT
            DO N=1,(LMAX+1)**2
               DO NP=1,(LMAX+1)**2
                  AP(M,N)=AP(M,N)+A(M,NP)*ROTYLM(N,NP)
               ENDDO
            ENDDO   
            ENDDO
#ifdef debug
            IF (IO%IU6>=0) THEN
               WRITE(*,'(I4,3F14.7)') IK,KPOINTS_FULL%VKPT(1:3,IK)
               DO M=1,WDES%NB_TOT
                  WRITE(*,'(4(2F14.7,2X))') (A(M,N),N=1,4)
               ENDDO
               WRITE(*,*)
               DO M=1,WDES%NB_TOT
                  WRITE(*,'(4(2F14.7,2X))') (AP(M,N),N=1,4)
               ENDDO
               WRITE(*,*)
            ENDIF
#endif
            ! Make the desired linear combinations
            DO M=1,WDES%NB_TOT
               CPROJ=(0._q,0._q)
               DO N=1,(LMAX+1)**2
                  CPROJ=CPROJ+AP(M,N)*HYBRID_ORBITAL(N)
               ENDDO
               WNPR_projections(M,IK,ISP+ISPINOR-1,IFNC)=CPROJ
            ENDDO

         ENDDO spinor
         ENDDO kpoint
         ENDDO spin

         DEALLOCATE(FG)

      ENDDO sites

      DEALLOCATE(A,AP,ROTYLM,HYBRID_ORBITAL)

      RETURN
      END SUBROUTINE CALCULATE_PROJECTIONS


!******************** SUBROUTINE ROTATE_PROJECTIONS ********************
!      
!***********************************************************************

      SUBROUTINE ROTATE_PROJECTIONS(W,WDES,IU0)
      USE wave_high
      USE full_kpoints
      TYPE (wavespin) W
      TYPE (wavedes) WDES
      INTEGER IU0
      ! local variables
      COMPLEX(q), ALLOCATABLE :: Wij(:,:,:),WORK(:),WTMP(:,:,:),WTMP2(:,:,:,:)
      REAL(q), ALLOCATABLE :: LAMBDA(:),RWORK(:) 
      INTEGER ISITE,ISP,IK,IK_,IB
      INTEGER I,J,Ibase,Inext,N,NP
      INTEGER ISPINOR,ISPINOR_
      INTEGER INFO


      CALL CHECK_FULL_KPOINTS

!-----Calculate the onsite density matrices, diagonalize them,
!     and rotate the WNPR_projections accordingly
      sites: DO ISITE=1,WNPR_num_sites
         Ibase=WNPR_index(ISITE)
         Inext=WNPR_index(ISITE+1)

         N=Inext-Ibase; NP=N*WDES%NRSPINORS

         ALLOCATE(Wij(NP,NP,WDES%ISPIN))
         ALLOCATE(WORK(2*NP-1),RWORK(3*NP-2),LAMBDA(NP))

         Wij=(0._q,0._q)
         spin1: DO ISP=1,WDES%ISPIN
            kpoint: DO IK=1,KPOINTS_FULL%NKPTS
               IK_=KPOINTS_FULL%NEQUIV(IK)
               bands: DO IB=1,WDES%NB_TOT
                  IF (.NOT.WNPR_use_band(IB,IK,ISP)) CYCLE bands
                  DO J=1,N
                  DO I=1,N
                     DO ISPINOR_=0,WDES%NRSPINORS-1
                     DO ISPINOR =0,WDES%NRSPINORS-1
                        Wij(I+N*ISPINOR,J+N*ISPINOR_,ISP)=Wij(I+N*ISPINOR,J+N*ISPINOR_,ISP)+ &
                       &   W%FERTOT(IB,IK_,ISP)*WNPR_projections(IB,IK,MAX(ISP,ISPINOR+1),I+Ibase-1)*CONJG(WNPR_projections(IB,IK,MAX(ISP,ISPINOR_+1),J+Ibase-1))
                     ENDDO
                     ENDDO
                  ENDDO
                  ENDDO
               ENDDO bands
            ENDDO kpoint

            Wij=Wij/KPOINTS_FULL%NKPTS*WDES%RSPIN

         ENDDO spin1

         ALLOCATE(WTMP(WDES%NB_TOT,KPOINTS_FULL%NKPTS,NP))
         spin2: DO ISP=1,WDES%ISPIN
            ! diagonalize Wij
            CALL ZHEEV('V','U',NP,Wij(1,1,ISP),NP,LAMBDA,WORK,2*NP-1,RWORK,INFO)

!           IF (.NOT.WDES%LNONCOLLINEAR) THEN 
!              ! nonmagnetic or collinear magnetic
!              WTMP=(0._q,0._q)
!              DO J=1,N
!                 DO I=1,N
!                    WTMP(:,:,J)=WTMP(:,:,J)+WNPR_projections(:,:,ISP,I+Ibase-1)*CONJG(Wij(I,J,ISP))
!                 ENDDO
!              ENDDO
!              ! and backfill
!              DO J=1,N
!                 WNPR_projections(:,:,ISP,J+Ibase-1)=WTMP(:,:,J)
!              ENDDO
!           ELSE
!              ! noncollinear magnetic
!              WTMP=(0._q,0._q)
!              DO J=1,NP
!                 DO I=1,N
!                    DO ISPINOR=0,WDES%NRSPINORS-1
!                       WTMP(:,:,J)=WTMP(:,:,J)+WNPR_projections(:,:,ISPINOR+1,I+Ibase-1)*CONJG(Wij(I+N*ISPINOR,J,ISP))
!                    ENDDO
!                 ENDDO
!              ENDDO
!              ! and backfill
!              DO J=1,N
!                 DO ISPINOR_=0,WDES%NRSPINOR-1
!                    WNPR_projections(:,:,ISPINOR_+1,J+Ibase-1)=WTMP(:,:,J+N*ISPINOR_)    
!                 ENDDO
!              ENDDO    
!           ENDIF       

            WTMP=(0._q,0._q)
            DO J=1,NP
               DO I=1,N
                  DO ISPINOR=0,WDES%NRSPINORS-1
                     WTMP(:,:,J)=WTMP(:,:,J)+WNPR_projections(:,:,MAX(ISP,ISPINOR+1),I+Ibase-1)*CONJG(Wij(I+N*ISPINOR,J,ISP))
                  ENDDO
               ENDDO
            ENDDO
            ! and backfill
            DO J=1,N
               DO ISPINOR_=0,WDES%NRSPINORS-1
                  WNPR_projections(:,:,MAX(ISP,ISPINOR_+1),J+Ibase-1)=WTMP(:,:,J+N*ISPINOR_)    
               ENDDO
            ENDDO 
#define debug
#ifdef debug
            IF (IU0>=0) THEN
               ! write the site number
               IF (ISP/=2) THEN
                  WRITE(IU0,'(4X,A,I5)',ADVANCE='No') 'site:',ISITE
                  IF (WNPR_i(WNPR_index(ISITE))>0) THEN
                     ! and if it is an atomic site, the atomic id
                     WRITE(IU0,'(2X,A,I5)') 'atom id:',WNPR_i(WNPR_index(ISITE))
                  ELSE
                     WRITE(IU0,*)
                  ENDIF
               ENDIF
               IF (WDES%ISPIN==2) WRITE(IU0,'(/4X,A,I5)') 'spin:',ISP
               DO J=NP,1,-1
                  WRITE(IU0,'(X,A,F14.7)',ADVANCE='No') 'o =',LAMBDA(J)
                  DO I=1,NP
                     WRITE(IU0,'(2X,F8.4,X,F8.4)',ADVANCE='No') Wij(I,J,ISP)
                  ENDDO
                  WRITE(IU0,*)
               ENDDO
            ENDIF
#endif
         ENDDO spin2

#ifdef debug
         Wij=(0._q,0._q)
         DO ISP=1,WDES%ISPIN
            DO IK=1,KPOINTS_FULL%NKPTS
               IK_=KPOINTS_FULL%NEQUIV(IK)
               DO IB=1,WDES%NB_TOT
                  IF (.NOT.WNPR_use_band(IB,IK,ISP)) CYCLE
                  DO J=1,N
                  DO I=1,N
                     DO ISPINOR_=0,WDES%NRSPINORS-1
                     DO ISPINOR =0,WDES%NRSPINORS-1
                        Wij(I+N*ISPINOR,J+N*ISPINOR_,ISP)=Wij(I+N*ISPINOR,J+N*ISPINOR_,ISP)+ &
                       &   W%FERTOT(IB,IK_,ISP)*WNPR_projections(IB,IK,MAX(ISP,ISPINOR+1),I+Ibase-1)*CONJG(WNPR_projections(IB,IK,MAX(ISP,ISPINOR_+1),J+Ibase-1))
                     ENDDO
                     ENDDO
                  ENDDO
                  ENDDO
               ENDDO
            ENDDO
            Wij=Wij/KPOINTS_FULL%NKPTS*WDES%RSPIN
         ENDDO

         IF (IU0>=0) THEN
            DO ISP=1,WDES%ISPIN
               IF (WDES%ISPIN==2) WRITE(IU0,'(/4X,A,I5)') 'spin:',ISP
               DO J=NP,1,-1
                  WRITE(IU0,'(18X)',ADVANCE='No')
                  DO I=NP,1,-1
                     WRITE(IU0,'(2X,F8.4,X,F8.4)',ADVANCE='No') Wij(I,J,ISP)
                  ENDDO
                  WRITE(IU0,*)
               ENDDO
            ENDDO
         ENDIF
#endif
         DEALLOCATE(Wij,WTMP,WORK,RWORK,LAMBDA)
      ENDDO sites

      RETURN
      END SUBROUTINE ROTATE_PROJECTIONS


!******************** SUBROUTINE ROTATE_ORBITALS_FULLK *****************
!      
!***********************************************************************

      SUBROUTINE ROTATE_ORBITALS_FULLK(W,WDES,WP,WDESP)
      USE wave_high
      USE full_kpoints
      USE kpoints_change
      USE dfast, ONLY : NBLK
      TYPE (wavedes) WDES,WDESP
      TYPE (wavespin) W,WP
      ! local variables
      TYPE (wavedes1) WDESK,WDESKP
      TYPE (wavefuna) WA,WAP
      INTEGER ISP,ISPINOR,IK,IKP,I,IFNC

      COMPLEX(q), ALLOCATABLE :: CBLOCK(:,:)
      GDEF, ALLOCATABLE :: GBLOCK(:,:)
      INTEGER IBLK,NBLK_ACT

      GDEF, ALLOCATABLE :: U(:,:)

      CALL CHECK_FULL_KPOINTS

      ALLOCATE(U(WDES%NB_TOT,WDESP%NB_TOT),CBLOCK(NBLK,WDES%NB_TOT),GBLOCK(NBLK,WDES%NB_TOT))

      spin: DO ISP=1,WDES%ISPIN
         kpoints: DO IK=1,KPOINTS_FULL%NKPTS
            ! find the corresponding entry in KPOINTS_FULL_ORIG, that contains
            ! the set of k-points used in the computation of the rotation matrices
            DO IKP=1,KPOINTS_FULL_ORIG%NKPTS
               IF (LIDENTICAL_KPOINT(KPOINTS_FULL%VKPT(:,IK),KPOINTS_FULL_ORIG%VKPT(:,IKP))) EXIT 
            ENDDO
            IF (IKP>KPOINTS_FULL_ORIG%NKPTS) THEN
               WRITE(*,*) 'ROTATE_ORBITALS_FULLK: ERROR: no matching k-point found in KPOINTS_FULL_ORIG',IK
               STOP
            ENDIF

            ! setup the rotation matrix U
            U=zero
            sites: DO IFNC=1,WNPR_num_wann
            bands: DO I=1,WDES%NB_TOT
               IF (.NOT.WNPR_use_band(I,IKP,ISP)) CYCLE bands
               DO ISPINOR=0,WDES%NRSPINORS-1
                  U(I,IFNC+WNPR_num_wann*ISPINOR)=WNPR_projections(I,IKP,MAX(ISP,ISPINOR+1),IFNC)
               ENDDO
            ENDDO bands
            ENDDO sites

            CALL SETWDES(WDES,WDESK,IK)
            WA=ELEMENTS(W,WDESK,ISP)

            CALL SETWDES(WDESP,WDESKP,IK)
            WAP=ELEMENTS(WP,WDESKP,ISP)

            ! redistribute over plane wave coefficients
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP); CALL REDISTRIBUTE_PROJ(WAP)
!           IF (WDES%DO_REDIS) THEN
!              CALL REDIS_PROJ(WDESK, WDES%NBANDS, W%CPROJ(1,1,IK,ISP))
!              CALL REDIS_PW  (WDESK, WDES%NBANDS, W%CW   (1,1,IK,ISP))
!
!              CALL REDIS_PROJ(WDESKP, WDESP%NBANDS, WP%CPROJ(1,1,IK,ISP))
!              CALL REDIS_PW  (WDESKP, WDESP%NBANDS, WP%CW   (1,1,IK,ISP))
!           ENDIF

!           CALL GGEMM(&
!          &   'N','N',m_ WDESK%NPL_RED,WDESP%NB_TOT,WDES%NB_TOT, & 
!          &   one,WA%CW_RED,m_ WDESK%NRPLWV_RED,U(1,1),WDES%NB_TOT, &
!          &   zero,WAP%CW_RED,m_ WDESKP%NRPLWV_RED  &
!          &   )

            WAP%CW_RED=(0._q,0._q)
            DO IBLK=0,WDESK%NPL_RED-1,NBLK
               NBLK_ACT=MIN(NBLK,WDESK%NPL_RED-IBLK)

               CBLOCK(1:NBLK_ACT,1:WDES%NB_TOT)=WA%CW_RED(IBLK+1:IBLK+NBLK_ACT,1:WDES%NB_TOT)

               CALL GGEMM(&
              &   'N','N',m_ NBLK_ACT,WDESP%NB_TOT,WDES%NB_TOT, & 
              &   one,CBLOCK(1,1),m_ NBLK,U(1,1),WDES%NB_TOT, &
              &   zero,WAP%CW_RED(IBLK+1,1),m_ WDESKP%NRPLWV_RED  &
              &   )

            ENDDO

!           IF (WDESK%NPRO_RED/=0) &
!           CALL GGEMM(&
!          &   'N','N',WDESK%NPRO_RED,WDESP%NB_TOT,WDES%NB_TOT, & 
!          &   one,WA%CPROJ_RED,WDESK%NPROD_RED,U(1,1),WDES%NB_TOT, &
!          &   zero,WAP%CPROJ_RED,WDESKP%NPROD_RED  &
!          &   )

            IF (WDESK%NPRO_RED/=0) THEN
               WAP%CPROJ_RED=zero
               DO IBLK=0,WDESK%NPRO_RED-1,NBLK
                  NBLK_ACT=MIN(NBLK,WDESK%NPRO_RED-IBLK)

                  GBLOCK(1:NBLK_ACT,1:WDES%NB_TOT)=WA%CPROJ_RED(IBLK+1:IBLK+NBLK_ACT,1:WDES%NB_TOT)

                  CALL GGEMM(&
                 &   'N','N',WDESK%NPRO_RED,WDESP%NB_TOT,WDES%NB_TOT, & 
                 &   one,GBLOCK(1,1),NBLK,U(1,1),WDES%NB_TOT, &
                 &   zero,WAP%CPROJ_RED(IBLK+1,1),WDESKP%NPROD_RED  &
                 &   )
               ENDDO
            ENDIF

            ! redistribute back over bands
            CALL REDISTRIBUTE_PW(WA) ; CALL REDISTRIBUTE_PROJ(WA)
            CALL REDISTRIBUTE_PW(WAP); CALL REDISTRIBUTE_PROJ(WAP)
!           IF (WDES%DO_REDIS) THEN
!              CALL REDIS_PROJ(WDESK, WDES%NBANDS, W%CPROJ(1,1,IK,ISP))
!              CALL REDIS_PW  (WDESK, WDES%NBANDS, W%CW   (1,1,IK,ISP))
!
!              CALL REDIS_PROJ(WDESKP, WDESP%NBANDS, WP%CPROJ(1,1,IK,ISP))
!              CALL REDIS_PW  (WDESKP, WDESP%NBANDS, WP%CW   (1,1,IK,ISP))
!           ENDIF
         ENDDO kpoints
      ENDDO spin

      DEALLOCATE(U,CBLOCK,GBLOCK)

      RETURN
      END SUBROUTINE ROTATE_ORBITALS_FULLK


!******************** SUBROUTINE ***************************************
!      
!***********************************************************************

      SUBROUTINE SETWNDES(WDES,WNDES)
      USE wave
      TYPE (wavedes) WDES
      TYPE (wavedes) WNDES
      ! local variables

      ! start by copying WDES
      WNDES=WDES

      ! the total number of bands must be >= WNPR_num_wann,
      ! and it must be dividable by WNWDES%NB_PAR
      WNDES%NB_TOT=((WNPR_num_wann*WNDES%NRSPINORS+WNDES%NB_PAR-1)/WNDES%NB_PAR)*WNDES%NB_PAR
      WNDES%NBANDS=WNDES%NB_TOT/WNDES%NB_PAR

      ! set KPOINT related quantities
      NULLIFY(WNDES%VKPT,WNDES%WTKPT)
      WNDES%NKDIM=WDES%NKDIM
      WNDES%NKPTS=WDES%NKPTS
      WNDES%NKPTS_FOR_GEN_LAYOUT=WDES%NKPTS_FOR_GEN_LAYOUT
      ALLOCATE(WNDES%VKPT(3,WNDES%NKPTS),WNDES%WTKPT(WNDES%NKPTS))
      WNDES%VKPT =WDES%VKPT
      WNDES%WTKPT=WDES%WTKPT

      ! hardcopy of the pointers of WDES to WNWDES
      NULLIFY(WNDES%NPLWKP,WNDES%NGVECTOR,WNDES%NGVECTOR_POS,WNDES%NPLWKP_TOT, &
     &   WNDES%NINDPW,WNDES%AT_GAMMA,WNDES%PL_INDEX,WNDES%PL_COL,WNDES%DATAKE, &
     &   WNDES%IGX,WNDES%IGY,WNDES%IGZ)

      CALL ALLOCWDES(WNDES,LEXTEND=.TRUE.)

      WNDES%NPLWKP      =WDES%NPLWKP
      WNDES%NGVECTOR    =WDES%NGVECTOR
      WNDES%NGVECTOR_POS=WDES%NGVECTOR_POS
      WNDES%NPLWKP_TOT  =WDES%NPLWKP_TOT
      WNDES%NINDPW      =WDES%NINDPW
      WNDES%AT_GAMMA    =WDES%AT_GAMMA
      WNDES%PL_INDEX    =WDES%PL_INDEX
      WNDES%PL_COL      =WDES%PL_COL
      WNDES%DATAKE      =WDES%DATAKE
      WNDES%IGX         =WDES%IGX
      WNDES%IGY         =WDES%IGY
      WNDES%IGZ         =WDES%IGZ

      RETURN
      END SUBROUTINE SETWNDES


!******************** FUNCTION FUNCS_PER_TYPE **************************
!      
!***********************************************************************

      FUNCTION FUNCS_PER_TYPE(ITYPE)
      INTEGER ITYPE,FUNCS_PER_TYPE
      SELECT CASE(ITYPE)
         CASE(0)
            ! s-functions
            FUNCS_PER_TYPE=1
         CASE(1)
            ! p-functions
            FUNCS_PER_TYPE=3
         CASE(2)
            ! d-functions
            FUNCS_PER_TYPE=5
         CASE(3)
            ! f-functions
            FUNCS_PER_TYPE=7
         CASE(4:)
            ! unknown types
            FUNCS_PER_TYPE=0
         CASE(-1)
            ! sp-functions
            FUNCS_PER_TYPE=2
         CASE(-2)
            ! sp2-functions
            FUNCS_PER_TYPE=3
         CASE(-3)
            ! sp3-functions
            FUNCS_PER_TYPE=4
         CASE(-4)
            ! sp3d-functions
            FUNCS_PER_TYPE=5
         CASE(-5)
            ! sp3d2-functions
            FUNCS_PER_TYPE=1
         CASE(:-6)
            ! unknown types
            FUNCS_PER_TYPE=0
      END SELECT
      END FUNCTION FUNCS_PER_TYPE


!******************** SUBROUTINE WRITE_PROJECTIONS *********************
! 
!***********************************************************************

      SUBROUTINE WRITE_PROJECTIONS(W,IUNIT)
      USE main_mpi
      USE full_kpoints
      USE wave_high
      TYPE (wavespin) W
      INTEGER IUNIT
      ! local variables
      INTEGER ISP,IK,IB,ISITE,IFNC,ISP_
      CHARACTER(LEN=115) :: HEADER(-5:3)
      ! initialize table headers
      DATA HEADER(0) /'       s                                                                                                           '/
      DATA HEADER(1) /'       pz               px               py                                                                        '/
      DATA HEADER(2) /'       dz2              dxz              dyz            dx2-y2             dxy                                     '/
      DATA HEADER(3) /'       fz3              fxz2             fyz2          fz(x2-y2)           fxyz         fx(x2-3yz)       fy(3x2-y2)'/
      DATA HEADER(-1)/'       s+px             s-px                                                                                       '/
      DATA HEADER(-2)/'      s-px+py          s-px-py           s+px                                                                      '/
      DATA HEADER(-3)/'    s+px+py+pz       s+px-py-pz       s-px+py-pz      s-px-py+pz                                                   '/
      DATA HEADER(-4)/'      sp3d-1           sp3d-2           sp3d-3           sp3d-4           sp3d-5                                   '/
      DATA HEADER(-5)/'      sp3d2-1          sp3d2-2          sp3d2-3          sp3d2-4          sp3d2-5          sp3d2-6                 '/
!                      012345678912345XX012345678912345XX012345678912345XX012345678912345XX012345678912345XX012345678912345XX0123456789123

      CALL CHECK_FULL_KPOINTS

      OPEN(UNIT=IUNIT,FILE=DIR_APP(1:DIR_LEN)//'WANCAR',STATUS='UNKNOWN')

      sites: DO ISITE=1,WNPR_num_sites 
         ! write the site number
         WRITE(IUNIT,'(4X,A,I5)',ADVANCE='No') 'site:',ISITE
         IF (WNPR_i(WNPR_index(ISITE))>0) THEN
            ! and if it is an atomic site, the atomic id
            WRITE(IUNIT,'(2X,A,I5)') 'atom id:',WNPR_i(WNPR_index(ISITE))
         ELSE
            WRITE(IUNIT,*)
         ENDIF
         
         spin: DO ISP=1,SIZE(WNPR_projections,3)
            ISP_=ISP; IF (W%WDES%LNONCOLLINEAR) ISP_=1
            ! write the spin channel
            WRITE(IUNIT,'(/4X,A,I5)') 'spin:',ISP
            kpoint: DO IK=1,SIZE(WNPR_projections,2)
               ! let us restrict this loop to the IBZ
               IF (KPOINTS_FULL%NEQUIV(IK)/=IK) EXIT kpoint
               ! write the kpoint information
               WRITE(IUNIT,'(/X,A,I5,X,A,3F11.7,4X,A,F11.7)') 'k-point:',IK,':',KPOINTS_FULL%VKPT(:,IK),'weight =',KPOINTS_FULL%WTKPT(IK)
               ! and the header for the band x projections table
               WRITE(IUNIT,'(2X,A,2X,A)') 'band     energy  ',HEADER(WNPR_l(WNPR_index(ISITE)))
               bands: DO IB=1,SIZE(WNPR_projections,1) 
                  ! the band index
                  WRITE(IUNIT,'(X,I4,F14.7)',ADVANCE='No') IB,REAL(W%CELTOT(IB,IK,ISP_),q)
                  ! WNPR_index(ISITE) holds the index of the first projection on site ISITE
                  ! in arrays like WNPR_projections
                  DO IFNC=WNPR_index(ISITE),WNPR_index(ISITE+1)-1
                     ! and the projections < p_(nf+ifnc) | \Psi_(ib,ik,isp) > 
                     WRITE(IUNIT,'(2X,F7.3,X,F7.3)',ADVANCE='No') CONJG(WNPR_projections(IB,IK,ISP,IFNC))
                  ENDDO
                  WRITE(IUNIT,*)
               ENDDO bands
            ENDDO kpoint
         ENDDO spin
         IF (ISITE<WNPR_num_sites) WRITE(IUNIT,'(/)')
      ENDDO sites

      CLOSE(IUNIT)

      RETURN
      END SUBROUTINE WRITE_PROJECTIONS

      END MODULE wnpr
