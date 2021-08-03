#include "symbol.inc"
MODULE dmft 
  USE prec
  USE fock
  USE chi_base
  USE wpot
  USE lattice
  USE xi
  USE crpa 
  USE mlwf
  USE constant
  IMPLICIT NONE

!**********************************************************************
!
! The routines in this module compute
!
! \int_BZ dk1 dk2 dk3 dk4
!  < w_n1 w'_n2 | W | w_n4 w'_n3 > =
!
! \int_BZ dk1 dk2 dk3 dk4
!  \int_d3r d3r' w*_k1,n1(r) w_k2+q,n4(r) w*_k2,n2(r') w_k1-q,n3(r') W(r',r)
!
! where W is the bare or a screened Coulomb interaction
!
! Caveat: the difference vectors between any two k-points must be
! included in the k-point set. This requires to use Gamma centered
! meshes.
!
! These routines are similar to the stuff in the local_field.F and
! ump2.F routines. 
!              
!**********************************************************************

  ! 
  REAL(q),ALLOCATABLE :: WPOS(:,:)
  ! two electron four orbital integrals in a Wannier basis
  GDEF, ALLOCATABLE :: TWOE_4WANNIER(:,:,:,:,:,:,:)
 
  ! some parameter from ADD_XI routine, needed for PIJKL
  REAL(q), PARAMETER :: PIJKL_EMPTY_THRESHHOLD=0.00001
  REAL(q), PARAMETER :: DEGENERACY_THRESHOLD=1E-2_q
  LOGICAL, PRIVATE, SAVE :: LWPOT
  LOGICAL, PRIVATE, SAVE :: LVPOT
  ! 
  ! wannier states for DMFT matrix elements
  ! 
  INTEGER, PRIVATE, SAVE :: NWLOW              
  INTEGER, PRIVATE, SAVE :: NWHIGH              
  INTEGER, PRIVATE, SAVE :: NWTOT
  !print all matrix elements calculated
  REAL(q),PRIVATE, SAVE :: VCUTOFF !high cutoff of V
  
#ifdef VASP2WANNIER90
  CONTAINS 
!*********************************************************************
!
! High level scheduler:
!   1) switch off symmetry 
!   2) run WANNIER90 
!   3) construct Wannier functions
!   4) compute matrix elements
!
!*********************************************************************

   SUBROUTINE DMFT_MATRIX_ELEMENTS( &
  &   WDES,W,KPOINTS,GRID,T_INFO,INFO,P,NONL_S,NONLR_S,SYMM,LATT_CUR,CQIJ,LMDIM,IO)
     USE base
     USE lattice
     USE pseudo
     USE poscar
     USE mgrid
     USE msymmetry
     USE nonl_high
     USE wave_high
     USE full_kpoints
     USE kpoints_change
     USE choleski
     USE pead, ONLY : PEAD_RESETUP_WDES
     USE mlwf
     USE wnpr
     IMPLICIT NONE
     TYPE(wavedes) WDES
     TYPE(wavespin) W
     TYPE(kpoints_struct) KPOINTS
     TYPE(grid_3d) GRID
     TYPE(type_info) T_INFO
     TYPE (info_struct) INFO
     TYPE(potcar) P(T_INFO%NTYP)
     TYPE(nonl_struct) NONL_S
     TYPE (nonlr_struct) NONLR_S
     TYPE(symmetry) SYMM
     TYPE(latt) LATT_CUR
     TYPE(in_struct) IO
     INTEGER LMDIM
     OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
     ! local variables
     TYPE (wavedes), POINTER :: WGW
     TYPE (grid_3d), POINTER :: GRIDWGW
     GDEF, ALLOCATABLE :: WIJKL(:,:,:,:,:,:)
     GDEF, ALLOCATABLE :: CIJKL(:,:,:,:,:,:)   !correction 
     GDEF, ALLOCATABLE :: PIJKL(:,:,:,:)
     GDEF, ALLOCATABLE :: UIJKL(:,:,:,:,:,:)
     COMPLEX(q), SAVE :: COMEGA !freuqency point for four orbital integrals
     INTEGER N1,N2,N3,N4,I,NX,NY,NZ,NWU_TOT
     INTEGER M1,M2,M3,M4, N, ISP1,ISP2, COUNTER 
     CHARACTER(5) :: EXTENSION   !file extension for spin polarized calculations

     ! switch off all symmetry
     IF (SYMM%ISYM>=0.AND. .NOT.WDES%LGAMMA) THEN
        ! switch of symmetry
        CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
       &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,WDES%ISPIN,IO%IU6)
        ! reread k-points with LINVERSION=.FALSE. to generate full mesh
        CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR,.FALSE., &
       &   T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
        CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_CUR,-1, IO%IU0)
        CALL PEAD_RESETUP_WDES(WDES, GRID, KPOINTS, LATT_CUR, LATT_CUR, IO)
        CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR)
        CALL RESETUP_FOCK( WDES, LATT_CUR)
        CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)
     ENDIF
 
     !we need the projection matrices U_matrix and U_matrix_opt in the following
     !so if they are not stored, determine them now
     CALL READ_WANPROJ(WDES,IO, .TRUE.)
     IF ( .NOT. LUFROMFILE ) THEN
     IF ((L2E4W .OR. LNLRPA)) THEN
         !if we are working in the Bloch basis Wannier functions are Bloch functions 
         IF ( LUSEBLOCH ) THEN
            ! if the interaction parameter are computed in the Bloch basis 
            ! we do not need Wannier functions
            ! number of basis elements is then the total # of bands 
            DMFT_num_wann = WDES%NB_TOT
         ELSEIF ( LUSEMLWF ) THEN
            CALL MLWF_WANNIER90(WDES,W,P,CQIJ,T_INFO,LATT_CUR,INFO,IO)
            DMFT_num_wann = MLWF_num_wann
         ELSEIF ( LUSELCAO ) THEN
!            IF ( LUSEORTH_LCAOS ) THEN
!                CALL WNPR_PROJECT_ORTH(W,WDES,T_INFO,INFO, P,CQIJ,LATT_CUR,IO)
!            ELSE
                CALL WNPR_PROJECT(W,WDES,T_INFO,INFO,P,CQIJ,LATT_CUR,IO)
!            ENDIF 
            DMFT_num_wann = WNPR_num_wann
         ENDIF 
     ENDIF 
     ENDIF 

     !a projection is necessary only if LCAOs or MLWFs are used 
     IF ( .NOT. LUSEBLOCH ) CALL PROJECT_ORBITALS_NOSYMM(WDES,W,IO%IU0)
    
     !set Target states, i.e. range of bands
     !for which matrix elements will be caluclated 
     CALL SET_TARGET_STATES(IO)
     IF ( VCUTOFF < 0 ) VCUTOFF =  1.1_q*W%WDES%ENMAX 

     !in the first run we compute V exactly 
     !using Gaunt coeffiecients and Slater integrals
     LMAXMP2=4
     LMAX_FOCKAE=-1

     ! generate descriptor for response function
     ALLOCATE(WGW, GRIDWGW)
     WGW=WDES_FOCK
     WGW%NKPTS=KPOINTS_FULL%NKPTS
     WGW%NKDIM=KPOINTS_FULL%NKPTS
     WGW%NKPTS_FOR_GEN_LAYOUT=KPOINTS_FULL%NKPTS
     WGW%VKPT =>KPOINTS_FULL%VKPT
     WGW%WTKPT=>KPOINTS_FULL%WTKPT
     WGW%ENMAX=VCUTOFF 

     IF ( LVPOT ) THEN
        IF ( IO%IU0>=0 ) WRITE(IO%IU0, '(" ")')
        IF ( IO%IU0>=0 ) WRITE(IO%IU0, '(" Calculation of V for high cutoff:",F10.2)')&
           WGW%ENMAX
        IF ( IO%IU6>=0 ) WRITE(IO%IU6, '(" Calculation of V for high cutoff:",F10.2)')&
           WGW%ENMAX
     ENDIF 

     ! GRIDWGW is identical to GRID_FOCK, except for GRIDWGW%FFTSCA
     GRIDWGW=GRID_FOCK
     IF (IO%IU6>=0) THEN 
        WRITE(IO%IU6,*) 'Basis sets for responsefunctions:'
        WRITE(IO%IU6,*) '================================='
     ENDIF

     IF (WGW%LGAMMA) THEN
        ! gamma only data layout with wavefunction stored as real in real space
        CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
        GRIDWGW%LREAL=.TRUE.
     ELSE
        CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
     ENDIF
     CALL GEN_INDEX (GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B,IO%IU6,-1, .TRUE.)
     !  init FFT (required if real to complex FFT is used)
     CALL FFTINI(WGW%NINDPW(1,1), WGW%NGVECTOR(1), WGW%NKPTS, WGW%NGDIM, GRIDWGW)
     ! allocate W_ijkl
     ALLOCATE(WIJKL(NWTOT,NWTOT,NWTOT,NWTOT,WDES%ISPIN,WDES%ISPIN))
     IF ( LCRPA .OR. LNLRPA .OR. L2E4W ) THEN
        ALLOCATE(CIJKL(NWTOT,NWTOT,NWTOT,NWTOT,WDES%ISPIN,WDES%ISPIN))
     ENDIF

! change the frequency grid point if necessary
 IF (OMEGAMAX>0.AND.NOMEGA==1) COMEGA=CMPLX(OMEGAMAX,0.0_q)

!=======================================================================
! should go in seperate routine
!=======================================================================
    IF (L2E4W_ALL) THEN
       I=0
       IF(.NOT.ALLOCATED(WPOS)) ALLOCATE(WPOS(3,KPOINTS_FULL%NKPTS))
       DO NX=1,KPOINTS%NKPX
          DO NY=1,KPOINTS%NKPY
             DO NZ=1,KPOINTS%NKPZ
                I=I+1
                WPOS(1,I)=NX-1
                WPOS(2,I)=NY-1
                WPOS(3,I)=NZ-1
             ENDDO
          ENDDO
       ENDDO
       IF (I/=KPOINTS_FULL%NKPTS) THEN
          WRITE(0,*) 'internal error in DMFT_MATRIX_ELEMENTS: something is at odds with the KPOINTS',I,KPOINTS_FULL%NKPTS
          STOP
       ENDIF
    ENDIF

!*******************************************************************
! bare Coulomb kernel
!         (r)             (r)
!          |---------------|
!       < w_I w_K | V | w_J w_L > =
!               |---------------|
!              (r')            (r')
!      \int_BZ dk1 dk2 dk3 dk4
!       \int_d3r d3r' w_k3,L(r') w*_k2,K(r') w_k4,J(r) w*_k1,I(r) / | r - r'|  
!      
!   corresponds to following diagram
!     I               K
!     -->--\     /-->--
!           |~~~|               
!     --<--/     \--<--
!     J               L
!
!    Note that the index convention in CALCULATE_WIJKL_TWOCENTRE for VIJKL
!    is different, namely
! 
!    V(J,L,K,I)
!
!*******************************************************************
     WIJKL=0
     CIJKL=0

     IF ( LVPOT ) THEN
        !print out diagram
        CALL DRAW_DIAGRAMS(.TRUE.,.FALSE.,.FALSE.,.FALSE.,IO%IU0 ) 
        CALL CALCULATE_WIJKL_TWOCENTRE( &
           WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NWLOW,NWHIGH,COMEGA,LMAXMP2,IO,CIJKL)
     ENDIF 

     !reduce cutoff and compute V using LMAXFOCKE = 4 and LMAXMP2 = -1
     LMAXMP2=-1
     LMAX_FOCKAE=4
     !first reallocate WGW properly
     DEALLOCATE( WGW, GRIDWGW)
     ! generate descriptor for response function
     ALLOCATE(WGW, GRIDWGW)
     WGW=WDES_FOCK
     WGW%NKPTS=KPOINTS_FULL%NKPTS
     WGW%NKDIM=KPOINTS_FULL%NKPTS
     WGW%NKPTS_FOR_GEN_LAYOUT=KPOINTS_FULL%NKPTS
     WGW%VKPT =>KPOINTS_FULL%VKPT
     WGW%WTKPT=>KPOINTS_FULL%WTKPT
     WGW%ENMAX=ENCUTGW

     IF ( LVPOT ) THEN
        !compute V with ENCUTGW and Gaunt coefficients
        IF ( IO%IU0>=0 ) WRITE(IO%IU0, '(" ")')
        IF ( IO%IU0>=0 ) WRITE(IO%IU0, '(" Calculation of V for small cutoff:",F10.2)')&
           WGW%ENMAX
        IF ( IO%IU6>=0 ) WRITE(IO%IU6, '(" Calculation of V for small cutoff:",F10.2)')&
           WGW%ENMAX
     ENDIF

     ! GRIDWGW is identical to GRID_FOCK, except for GRIDWGW%FFTSCA
     GRIDWGW=GRID_FOCK
     IF (IO%IU6>=0) THEN 
        WRITE(IO%IU6,*) 'Basis sets for responsefunctions:'
        WRITE(IO%IU6,*) '================================='
     ENDIF

     IF (WGW%LGAMMA) THEN
        ! gamma only data layout with wavefunction stored as real in real space
        CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
        GRIDWGW%LREAL=.TRUE.
     ELSE
        CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
     ENDIF
     CALL GEN_INDEX (GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B,IO%IU6,-1, .TRUE.)
     !  init FFT (required if real to complex FFT is used)
     CALL FFTINI(WGW%NINDPW(1,1), WGW%NGVECTOR(1), WGW%NKPTS, WGW%NGDIM, GRIDWGW)

     IF ( LVPOT ) THEN
        CALL CALCULATE_WIJKL_TWOCENTRE( &
           WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NWLOW,NWHIGH,COMEGA,LMAXMP2,IO,WIJKL)

        !write matrix elements of bare interaction to VIJKL using MP2
        CALL WRITE_DMFT_MATRIX_ELEMENTS(IO,WDES%ISPIN,.FALSE.,CIJKL,WIJKL)

        !save correction to CIJKL
        CIJKL = CIJKL - WIJKL  
     ENDIF 
     
     IF (.NOT.LWPOT) THEN
         IF (IO%IU0>=0) WRITE(IO%IU0,*)' Skipping calculation of WIJKL,PIJKL and UIJKL'
       GOTO  1
     ENDIF

!************************************************************************
! sreened Coulomb kernel
!         (r)             (r)
!          |---------------|
!       < w_I w_K | W | w_J w_L > =
!               |---------------|
!              (r')            (r')
!      \int_BZ dk1 dk2 dk3 dk4
!       \int_d3r d3r' w_k3,L(r') w*_k2,K(r') w_k4,J(r) w*_k1,I(r) W(r,r')  
!      
!   corresponds to following diagram
!     I               K
!     -->--\  W  /-->--
!           |===|               
!     --<--/     \--<--
!     J               L
!
!  This time the unscreened interaction UIJKL in the four-point basis is
!  calculated too via the Dyson-like equation:
!
!  U_IJKL = W_IJKL - U_IJK'L' * P_K'L'I'J' * W_I'J'KL
!
!  This corresponds to following diagrams:
!
! I         K   I         K   I        K'    I'       K  
! ->-\ U /->-   ->-\ W /->-   ->-\ U /->----->-\ W /->-
!     |-|     =     |=|     -     |-|     P     |=|     
! -<-/   \-<-   -<-/   \-<-   -<-/   \-<-----<-/   \-<-   
! J         L   J         L   J        L'    J'       L    
!
!*************************************************************************
! #ifdef use_shmem
!      IF ( IO%IU0>=0 ) WRITE(IO%IU0,*) ' Currently shared memory for calculation of UIJKL '//&
!           'not supported '
!      STOP
! #endif
     !print diragram
     CALL DRAW_DIAGRAMS(.FALSE.,LWPOT,LCRPA,LNLRPA, IO%IU0 ) 

     CALL INIT_WPOT_HANDLE( WPOTH, WGW, KPOINTS_FULL%NKPTS, IO%IU6, IO%IU0, 1, 1, 1 )
     WIJKL=0

     IF ( LNLRPA .OR. L2E4W ) THEN
        ALLOCATE(UIJKL(NWTOT,NWTOT,NWTOT,NWTOT,WDES%ISPIN,WDES%ISPIN))
        ALLOCATE(PIJKL(NWTOT,NWTOT,NWTOT,NWTOT))
        PIJKL=0
        UIJKL=0
        CALL CALCULATE_WIJKL_TWOCENTRE( &
           WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NWLOW,NWHIGH,COMEGA,-1,IO,WIJKL,PIJKL)

        !unscreening 
        CALL REMOVE_PIJKL_FROM_WIJKL(WDES, NWTOT,PIJKL,WIJKL,UIJKL)

        !add the correction to WIJKL and UIJKL 
        WIJKL = WIJKL + CIJKL
        UIJKL = UIJKL + CIJKL
  
        !write matrix elements
        CALL WRITE_DMFT_MATRIX_ELEMENTS(IO,WDES%ISPIN,.TRUE.,WIJKL,UIJKL,PIJKL)
    
        ! deallocation
        DEALLOCATE(UIJKL,PIJKL)

     ELSE
        CALL CALCULATE_WIJKL_TWOCENTRE( &
           WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NWLOW,NWHIGH,COMEGA,-1,IO,WIJKL)

        !add the correction to WIJKL and UIJKL 
        WIJKL = WIJKL + CIJKL

        !write matrix elements
        CALL WRITE_DMFT_MATRIX_ELEMENTS(IO,WDES%ISPIN,.TRUE.,WIJKL)
     ENDIF 
 

1    CONTINUE
     
     DEALLOCATE(WGW,GRIDWGW)
     DEALLOCATE( WIJKL , CIJKL ) 
     RETURN
   END SUBROUTINE DMFT_MATRIX_ELEMENTS

!*********************************************************************
!
! main calculational procedure
!
!*********************************************************************

  SUBROUTINE CALCULATE_WIJKL( &
 &  WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NB_START,NB_STOP,LMAX,WIJKL,IO)

    USE base
    USE pseudo
    USE mpimy
    USE mkpoints
    USE constant
    USE poscar
    USE pot
    USE pawm
    USE wave_high
    USE ini
    USE radial
    USE kpoints_change
    USE full_kpoints
    USE constant
    IMPLICIT NONE
    TYPE (wavedes) WDES
    TYPE (wavespin) W
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt) LATT_CUR
    TYPE (type_info) T_INFO
    TYPE (potcar) P(T_INFO%NTYP)
    TYPE (in_struct) IO
    TYPE (wavedes) WGW           ! descriptor for basis set of response function
    INTEGER NB_START,NB_STOP     ! global band indeces
    INTEGER LMAX                 ! maximum L index for one center terms
    GDEF, OPTIONAL :: WIJKL(:,:,:,:)

! local variables
    GDEF, ALLOCATABLE :: TWOe4o(:,:,:,:),TWOe4o_ADD(:,:,:,:)
    TYPE (wavespin) WHF
    TYPE (wavefun1), ALLOCATABLE :: W1(:), W2(:), W3(:), W4(:)
    TYPE (wavefun1) :: WTMP
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2, WDESK3, WDESK4
#define collect_all_wavefunctions
#ifdef collect_all_wavefunctions
    TYPE (wavefun1), ALLOCATABLE :: W_all(:,:,:)
    TYPE (wavedes1), TARGET :: WDESK
#endif
    INTEGER :: IERR              ! error indicator
    INTEGER :: NSTRIP            ! block size
    INTEGER :: NGLB              ! total number of bands 
    INTEGER :: N, NP
    INTEGER :: K1, K2, K3, K4, K4_BASE, K4_COLLECT, K4_LOCAL, K2_LOCAL, K2_COLLECT, K3_IN_FULL_ORIG
    REAL(q) :: SCALE
    REAL(q) :: NFFTW             ! number of FFTs for  wavefunctions
    REAL(q) :: NFLOAT4O, NFFT4O  ! number of BLAS3 operations in 4 orbital routines

    INTEGER :: ISP1, ISP2
    INTEGER :: R1, R2, R3, INDEX, N1, N2, N3, N4
    COMPLEX :: CPHASE
    ! one-center terms
    TYPE (one_center_handle), POINTER :: H => NULL()
    INTEGER :: NCPU
    
    CHARACTER (LEN=1) :: SP(2)=(/ "u", "d" /)

!=======================================================================
! preparation of four-orbital related quantities
!=======================================================================
    IF (LMAX>=0) THEN
       CALL SET_UP_ONE_CENTER_H(WDES, P, T_INFO, LMAX, H)
    ENDIF

    CALL CHECK_FULL_KPOINTS

    CALL START_TIMING("WIJKL+")

    ! switch to WDES_FOCK
    WHF=W; WHF%WDES => WDES_FOCK
!=======================================================================

130 FORMAT (5X, //, &
         &'----------------------------------------------------', &
         &'----------------------------------------------------'//)
    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK2,0)
    CALL SETWDES(WHF%WDES,WDESK3,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    ! total number of bands requested
    NGLB=NB_STOP-NB_START+1

#ifdef collect_all_wavefunctions
    ALLOCATE(W_all(NGLB,WDES%NKPTS,WDES%ISPIN))
    ! collect all wave functions
    DO ISP1=1,WDES%ISPIN
       DO K1=1,WDES%NKPTS
          CALL SETWDES(WHF%WDES,WDESK,K1)
          DO N=1,NGLB
             CALL NEWWAV(W_all(N,K1,ISP1),WDESK,.FALSE.) 
          ENDDO
          CALL W1_GATHER_GLB_NOCR( WHF, NB_START, NB_STOP, ISP1, W_all(:,K1,ISP1) )
       ENDDO
    ENDDO
#endif

    ! allocate storage for W1, W2, W3, and W4
    ALLOCATE(W1(NGLB),W2(NGLB),W3(NGLB),W4(NGLB))
    DO N=1,NGLB 
       CALL NEWWAV(W1(N),WDESK1,.TRUE.)
       CALL NEWWAV(W2(N),WDESK2,.TRUE.)
       CALL NEWWAV(W3(N),WDESK3,.TRUE.)
       CALL NEWWAV(W4(N),WDESK4,.TRUE.)
    ENDDO
    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,4I5)') 'allocating two-electron 4 orbital integral table', &
   & NGLB,NGLB,NGLB,NGLB

    ! TWOe4o(n4, n3, n2, n1) = 
    !  \int dk1 dk2 dk3 dk4 < w_k1,n1 w'_k2,n2 | W | w_k2+q,n4 w'_k1-q,n3 >
    ALLOCATE(TWOe4o(NGLB,NGLB,NGLB,NGLB), TWOe4o_ADD(NGLB,NGLB,NGLB,NGLB))
    TWOe4o=0._q

    IF (L2E4W_ALL) THEN
       CALL  TWOE_4WANNIER_ALLOCATE( WDES, NGLB )
    ENDIF

    SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
    IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2
    NFLOAT4O=0 ; NFFT4O=0
!==========================================================================
!  outer loop construct for two electron four orbital integrals
!  these loops go over the indices k1, k2, 
!  and possibly blocks of the band indices n1 and n2
!==========================================================================
    CALL START_TIMING("G")
!==========================================================================
    sp1: DO ISP1=1,WDES%ISPIN
    sp2: DO ISP2=1,WDES%ISPIN
    IF (ISP1>ISP2) CYCLE sp2
!==========================================================================
    ! loop over all q-points 
    DO K3=1, WDES%NKPTS
       IF (IO%IU0>=0) WRITE(IO%IU0,*)
       IF (IO%IU0>=0) THEN
          IF (WDES%ISPIN==1) THEN
             WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') K3,KPOINTS%VKPT(:,K3)
          ELSE
             WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ",A1,A1,", ")') K3,KPOINTS%VKPT(:,K3),SP(ISP1),SP(ISP2)
          ENDIF
       ENDIF

       ! collect bands [NB_START,NB_STOP] (global index) for K3=K1-Q
       CALL SETWDES(WHF%WDES,WDESK3,K3)
#ifndef collect_all_wavefunctions
       CALL W1_GATHER_DISTR( WHF, NB_START, NB_STOP, ISP2, W3)
#else
       CALL SETWDES(WHF%WDES,WDESK,K3)
       DO N=1,NGLB
          CALL W1_COPY(W_all(N,K3,ISP2),W3(N))
          CALL FFTWAV_W1(W3(N))
       ENDDO
#endif
       k4loop: DO K4_BASE=1, ((WDES%NKPTS+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR,WDES%NB_PAR
          ! distribute the K4 in a round robin fashion over cores
          K4=-1        ! determine which k-point treated locally (K4)
          DO K4_COLLECT=K4_BASE,MIN(K4_BASE+WDES%NB_PAR-1,WDES%NKPTS)
             K4_LOCAL=K4_COLLECT-K4_BASE+1

#ifndef collect_all_wavefunctions
             CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)
             CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP1, W4, K4_LOCAL)
#endif
             IF (K4_LOCAL==WDES%NB_LOW) K4=K4_COLLECT
          ENDDO
#ifdef collect_all_wavefunctions
          IF (K4>=1) THEN
             CALL SETWDES(WHF%WDES,WDESK,K4)
             CALL SETWDES(WHF%WDES,WDESK4,K4)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K4,ISP1),W4(N))
                CALL FFTWAV_W1(W4(N))
             ENDDO
          ENDIF
#endif
!==========================================================================
!  inner loop for four orbital integrals
!==========================================================================
!         DO K1=1,KPOINTS_ORIG%NKPTS
          DO K1=1,WDES%NKPTS
             TWOe4o_ADD=0
             CALL GWPROGRESS(IO%IU0, K4, WDES%NKPTS, K1, WDES%NKPTS)

             ! generate the proper descriptor for W1 wavefunctions
             CALL SETWDES(WHF%WDES,WDESK1,K1)
#ifndef collect_all_wavefunctions
             CALL W1_GATHER_GLB( WHF, NB_START, NB_STOP, ISP1, W1)
#else
             CALL SETWDES(WHF%WDES,WDESK,K1)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K1,ISP1),W1(N))
                CALL FFTWAV_W1(W1(N))
             ENDDO
#endif
             ! collect bands for k-point K2=K3+K4-K1
             ! depending on whether K4 is distributed among nodes or not
             ! K2 is also accordingly assembled
             K2=-1
             DO K4_COLLECT=K4_BASE, WDES%NKPTS ! loop from present K2 upto WDES%NKPTS
                K2_COLLECT=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K4_COLLECT)-WHF%WDES%VKPT(:,K1),KPOINTS_FULL)

                K2_LOCAL=K4_COLLECT-K4_BASE+1   ! new k-point to be included
#ifndef collect_all_wavefunctions
                ! generate the proper descriptor for W2 wavefunctions
                CALL SETWDES(WHF%WDES,WDESK2,K2_COLLECT)
                CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP2, W2, K2_LOCAL)
#endif
                IF (K2_LOCAL==WDES%NB_LOW) K2=K2_COLLECT
             ENDDO
#ifdef collect_all_wavefunctions
             IF (K2>=1) THEN
                CALL SETWDES(WHF%WDES,WDESK,K2)
                CALL SETWDES(WHF%WDES,WDESK2,K2)
                DO N=1,NGLB
                   CALL W1_COPY(W_all(N,K2,ISP2),W2(N))
                   CALL FFTWAV_W1(W2(N))
                ENDDO
             ENDIF
#endif
             IF (K2>=1) THEN
                CALL SETWDES(WHF%WDES,WDESK2,K2)
                CALL SETWDES(WHF%WDES,WDESK4,K4)
                ! here the work is done
                CALL TWOELECTRON4O_WIJKL( &
                     !  < w_k1,n1 w_k2,n2 | W | w_k4,n4 w_k3,n3 > = TWOe4o(n4,n3,n2,n1)
                     WHF, H, LATT_CUR, WGW, FSG_STORE(1), &       ! FSG = FSG_STORE(1) or 0._q (convergence correction)?
                     W1, K1, W2, K2, W3, K3, W4, K4, & 
                     NB_START, NGLB, NFLOAT4O, NFFT4O, TWOe4o_ADD)
             ENDIF
             TWOe4o_ADD=SCALE*SCALE*SCALE*TWOe4o_ADD
             TWOe4o=TWOe4o+TWOe4o_ADD

             IF (L2E4W_ALL) THEN
                CALL TWOE_4WANNIER_SET( WGW, WDES, TWOe4o_ADD, K1, K2, K3, K4, NGLB)
             ENDIF
          ENDDO ! K1
!==========================================================================
! close outer loop
!==========================================================================
       ENDDO k4loop
       CALL STOP_TIMING("G",IO%IU6,"WIJKL")
    ENDDO ! K3
!==========================================================================
    ENDDO sp2
    ENDDO sp1
!==========================================================================
    CALL STOP_TIMING("WIJKL+",IO%IU6,XMLTAG='total')

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT4O, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT4O  , 1))

    CALLMPI( M_sum_g(WGW%COMM_INTER, TWOe4o, NGLB*NGLB*NGLB*NGLB))

    IF (PRESENT(WIJKL)) THEN
       IF (SIZE(WIJKL,1)>=NGLB.AND.SIZE(WIJKL,2)>=NGLB.AND. &
      &     SIZE(WIJKL,3)>=NGLB.AND.SIZE(WIJKL,4)>=NGLB) &
      &   WIJKL(1:NGLB,1:NGLB,1:NGLB,1:NGLB)=TWOe4o
    ENDIF

    DEALLOCATE(TWOe4o_ADD,TWOe4o)
!==========================================================================
! deallocation
!==========================================================================
    DO N=1,NGLB
       CALL DELWAV(W4(N) ,.TRUE.)
       CALL DELWAV(W3(N) ,.TRUE.)
       CALL DELWAV(W2(N) ,.TRUE.)
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1,W2,W3,W4)

    IF (ASSOCIATED(H)) CALL DEALLOCATE_ONE_CENTER_H(H)

#ifdef collect_all_wavefunctions
    DO ISP1=1,WDES%ISPIN
    DO K1=1,WDES%NKPTS
       DO N=1,NGLB
          CALL DELWAV(W_all(N,K1,ISP1),.FALSE.) 
       ENDDO
    ENDDO; ENDDO
    DEALLOCATE(W_all)
#endif
!==========================================================================
! finally  write out the full 4 orbital integrals
!==========================================================================
    IF(L2E4W_ALL)THEN
       CALL TWOE_4WANNIER_WRITE( WGW, WDES, NGLB)
       CALL TWOE_4WANNIER_DEALLOCATE
    END IF
    RETURN
  END SUBROUTINE CALCULATE_WIJKL

!*********************************************************************
!
! this routine allocated the 4 orbital 3 centre integrals
!
!*********************************************************************

  SUBROUTINE TWOE_4WANNIER_ALLOCATE( WDES, NGLB )
    TYPE(wavedes) WDES
    INTEGER NGLB
    INTEGER NCPU
#ifdef MPI
    NCPU=WDES%COMM_INTER%NCPU
#else
    NCPU=1
#endif
    IF(.NOT.ALLOCATED(TWOE_4WANNIER)) &
         ALLOCATE(TWOE_4WANNIER((WDES%NKPTS+NCPU-1)/NCPU,WDES%NKPTS,WDES%NKPTS,NGLB,NGLB,NGLB,NGLB))
    TWOE_4WANNIER=0

  END SUBROUTINE TWOE_4WANNIER_ALLOCATE

  SUBROUTINE TWOE_4WANNIER_DEALLOCATE()
    DEALLOCATE(TWOE_4WANNIER)
  END SUBROUTINE TWOE_4WANNIER_DEALLOCATE


!*********************************************************************
!
! this routine broadcasts the local 4 orbital integrals
! to other nodes
!
!*********************************************************************

  SUBROUTINE TWOE_4WANNIER_SET( WGW, WDES, TWOe4o, K1, K2, K3, K4, NGLB)
    USE constant
    TYPE (wavedes) WGW           ! descriptor for basis set of response function
    TYPE (wavedes) WDES          ! descriptor for basis set of response function
    GDEF :: TWOe4o(:,:,:,:)
    INTEGER :: K1, K2, K3, K4, NGLB
  ! local
    INTEGER :: K_BCAST(1:4)
    INTEGER N
    COMPLEX(q) :: CPHASE
    INTEGER :: R1, R2, R3, INDEX
    GDEF, ALLOCATABLE :: TWOe4o_BCAST(:,:,:,:)


#ifdef MPI
    ALLOCATE(TWOe4o_BCAST(NGLB,NGLB,NGLB,NGLB))

    DO N=1,WGW%COMM_INTER%NCPU
       IF (N==WGW%COMM_INTER%NODE_ME) THEN
          TWOe4o_BCAST=TWOe4o
          K_BCAST(1)=K1
          K_BCAST(2)=K2
          K_BCAST(3)=K3
          K_BCAST(4)=K4
       ENDIF
       CALLMPI( M_bcast_i_from(WGW%COMM_INTER, K_BCAST, 4, N ))
#ifdef gammareal
       CALLMPI( M_bcast_d_from(WGW%COMM_INTER, TWOe4o_BCAST, SIZE(TWOe4o_BCAST), N ))
#else
       CALLMPI( M_bcast_z_from(WGW%COMM_INTER, TWOe4o_BCAST, SIZE(TWOe4o_BCAST), N ))
#endif
       !  < w_k1,n1 w_k2,n2 | W | w_k4,n4 w_k3,n3 > = TWOe4o(n4,n3,n2,n1)
       ! R4 = 0
       !                                           +---------------------+
       ! TWOE_4WANNIER(R1,R2,R3,n4,n3,n2,n1) = < w_n1,R1 w_n2,R2 | W | w_n4,0 w_n3,R3 >
       !                                                    +-------------------+
       !
       INDEX=0
       DO R1=WGW%COMM_INTER%NODE_ME, WDES%NKPTS, WGW%COMM_INTER%NCPU
          INDEX=INDEX+1
          DO R2=1,WDES%NKPTS
             DO R3=1,WDES%NKPTS
                CPHASE= EXP(CITPI* (+SUM(WDES%VKPT(1:3,K_BCAST(1))*WPOS(1:3,R1)) & 
                     +SUM(WDES%VKPT(1:3,K_BCAST(2))*WPOS(1:3,R2)) &
                     -SUM(WDES%VKPT(1:3,K_BCAST(3))*WPOS(1:3,R3))))
                TWOE_4WANNIER(INDEX,R2,R3,:,:,:,:)=TWOE_4WANNIER(INDEX,R2,R3,:,:,:,:)+TWOe4o_BCAST*CPHASE
             ENDDO
          ENDDO
       ENDDO
       
    ENDDO
    DEALLOCATE(TWOe4o_BCAST)
#else
    !  < w_k1,n1 w_k2,n2 | W | w_k4,n4 w_k3,n3 > = TWOe4o(n4,n3,n2,n1)
    ! R4 = 0
    !                                           +---------------------+
    ! TWOE_4WANNIER(R1,R2,R3,n4,n3,n2,n1) = < w_n1,R1 w_n2,R2 | W | w_n4,0 w_n3,R3 >
    !                                                    +-------------------+
    !
    INDEX=0
    DO R1=1, WDES%NKPTS
       INDEX=INDEX+1
       DO R2=1,WDES%NKPTS
          DO R3=1,WDES%NKPTS
             CPHASE= EXP(CITPI* (+SUM(WDES%VKPT(1:3,K1)*WPOS(1:3,R1)) & 
                  +SUM(WDES%VKPT(1:3,K2)*WPOS(1:3,R2)) &
                  -SUM(WDES%VKPT(1:3,K3)*WPOS(1:3,R3))))
             TWOE_4WANNIER(INDEX,R2,R3,:,:,:,:)=TWOE_4WANNIER(INDEX,R2,R3,:,:,:,:)+TWOe4o*CPHASE
          ENDDO
       ENDDO
    ENDDO
#endif
    
  END SUBROUTINE TWOE_4WANNIER_SET


!****************** SUBROUTINE TWOE_4WANNIER_WRITE ********************
!
! this routine writes the 3 centre Wannier integrals to a file
!
!**********************************************************************

  SUBROUTINE TWOE_4WANNIER_WRITE( WGW, WDES, NGLB)
    TYPE (wavedes) WGW           ! descriptor for basis set of response function
    TYPE (wavedes) WDES          ! descriptor for basis set of response function
    INTEGER :: NGLB
 ! local
    INTEGER :: R1, R2, R3, N1, N2, N3, N4, INDEX

    CHARACTER (LEN=16) :: FILENAME_NODE
    GDEF, ALLOCATABLE :: E2W4_ALL(:,:,:,:,:,:,:)

       ! EVERY node writes part of VRSTijkl
       ! filename
#ifdef MPI
       WRITE(FILENAME_NODE,'(''VRSTijkl.'',I3.3)') WGW%COMM_INTER%NODE_ME
#else
       WRITE(FILENAME_NODE,'(''VRSTijkl.'',I3.3)') 1
#endif
       OPEN(UNIT=101,FILE=FILENAME_NODE,STATUS='REPLACE')
        !
       ! write out positions of Ri
       WRITE(101,'(''# R1'')')
#ifdef MPI
       DO R1=WGW%COMM_INTER%NODE_ME, WDES%NKPTS, WGW%COMM_INTER%NCPU
          WRITE(101,'(I4,3F10.6)') R1,WPOS(:,R1)
       END DO
#else
       DO R1=1, WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R1,WPOS(:,R1)
       END DO
#endif
       WRITE(101,'(''# R2'')')
       DO R2=1,WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R2,WPOS(:,R2)
       END DO
       WRITE(101,'(''# R3'')')
       DO R3=1,WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R3,WPOS(:,R3)
       END DO
       !
       INDEX=0
#ifdef MPI
       DO R1=WGW%COMM_INTER%NODE_ME, WDES%NKPTS, WGW%COMM_INTER%NCPU
#else
       DO R1=1, WDES%NKPTS
#endif
          INDEX=INDEX+1
          DO R2=1,WDES%NKPTS
             DO R3=1,WDES%NKPTS
                !
                DO N1=1,NGLB 
                   DO N2=1,NGLB
                      DO N3=1,NGLB 
                         DO N4=1,NGLB
                            !
                            !                                           +---------------------+
                            ! TWOE_4WANNIER(R1,R2,R3,n4,n3,n2,n1) = < w_n1,R1 w_n2,R2 | W | w_n4,0 w_n3,R3 >
                            !                                                    +-------------------+
                            !
                            WRITE(101,'(4i4,4i4,2F20.10)') R1,0,R2,R3,N1,N2,N4,N3,TWOE_4WANNIER(INDEX,R2,R3,N4,N3,N2,N1)
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
                !
             END DO
          END DO
       END DO
       !
       CLOSE(101)

       !===========================================================================
       ! write out entire matrix in one file
       ! only for test purposes, otherwise probably too large
       ALLOCATE(E2W4_ALL(WDES%NKPTS,WDES%NKPTS,WDES%NKPTS,NGLB,NGLB,NGLB,NGLB))
       E2W4_ALL=0
       INDEX=0
#ifdef MPI
       DO R1=WGW%COMM_INTER%NODE_ME, WDES%NKPTS, WGW%COMM_INTER%NCPU
#else
       DO R1=1, WDES%NKPTS
#endif
          INDEX=INDEX+1
          E2W4_ALL(R1,:,:,:,:,:,:)=TWOE_4WANNIER(INDEX,:,:,:,:,:,:)
       END DO
       CALLMPI( M_sum_g(WGW%COMM_INTER, E2W4_ALL, SIZE(E2W4_ALL)) )

       OPEN(UNIT=101,FILE='VRSTijkl',STATUS='REPLACE')
       !
       ! write out positions of Ri
       WRITE(101,'(''# R1'')')
       DO R1=1,WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R1,WPOS(:,R1)
       END DO
       WRITE(101,'(''# R2'')')
       DO R2=1,WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R2,WPOS(:,R2)
       END DO
       WRITE(101,'(''# R3'')')
       DO R3=1,WDES%NKPTS
          WRITE(101,'(I4,3F10.6)') R3,WPOS(:,R3)
       END DO
       !
       DO R1=1,WDES%NKPTS
          DO R2=1,WDES%NKPTS
             DO R3=1,WDES%NKPTS
                !
                DO N1=1,NGLB 
                   DO N2=1,NGLB
                      DO N3=1,NGLB 
                         DO N4=1,NGLB
                            !
                            !                                           +---------------------+
                            ! TWOE_4WANNIER(R1,R2,R3,n4,n3,n2,n1) = < w_n1,R1 w_n2,R2 | W | w_n4,0 w_n3,R3 >
                            !                                                    +-------------------+
                            !
                            WRITE(101,'(4i4,4i4,2F20.10)') R1,1,R2,R3,N1,N2,N4,N3,E2W4_ALL(R1,R2,R3,N4,N3,N2,N1)
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
                !
             END DO
          END DO
       END DO
       DEALLOCATE(E2W4_ALL)
       CLOSE(101)

     END SUBROUTINE


!****************** SUBROUTINE TWOELECTRON4O_WIJKL ********************
! calculate
!    (r)                   (r)
!     |---------------------|
!  < w_k1,n1 w_k2,n2 | W | w_k4,n4 w_k3,n3 > =
!             |---------------------|
!            (r')                  (r')
!
! int_d3r d3r' w_k3,n3(r') w*_k2,n2(r') w_k4,n4(r) w*_k1,n1(r) W(r,r')
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
!
!**********************************************************************

  SUBROUTINE TWOELECTRON4O_WIJKL( WHF, H, LATT_CUR, WGW, FSG, &
       W1, K1, W2, K2, W3, K3, W4, K4, &
       NB_START, NGLB, NFLOAT, NFFT, TWOELECTRON4O)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
!#define do_fly_epsilon
#ifdef do_fly_epsilon
    USE chi_base
#endif
    IMPLICIT NONE

! passed variables
    TYPE (wavespin) WHF
    TYPE (one_center_handle), POINTER :: H
    TYPE (latt) LATT_CUR
    TYPE (wavedes)       WGW           ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4
    INTEGER :: NB_START, NGLB ! first and last band as well as number of bands
    REAL(q) :: FSG                  ! singularity correction
    REAL(q) :: NFLOAT, NFFT
    GDEF :: TWOELECTRON4O(:,:,:,:)

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)  ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)  ! charge 
    GDEF      , ALLOCATABLE :: CRHO14(:,:)    ! one center charge
    GDEF      , ALLOCATABLE :: CRHO23(:,:)    ! one center charge
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    LOGICAL :: LFULL                      ! read W(G+q,G'+q) or just the diagonal part
#ifdef do_fly_epsilon
    INTEGER NQ_IN_FULL_ORIG,NQ_IN_IRZ
#endif

    ! set the descriptor for wavefunctions at the point k1-k4
    ! the product w*_k4,n4(r) w_k1,n1(r) corresponds to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

    ! the same should hold for k3-k2 (=q)
    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'TWOELECTRON4O_WIJKL: internal error: q-point changed',NQ,NQ_
       STOP
    ENDIF
! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP =WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NGLB* NGLB ), &
         GCHG23(NP, NGLB* NGLB ), &
         CHAM(NGLB, NGLB, NGLB,  NGLB))

    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) &
   &   ALLOCATE(CRHO14(H%TOTAL_ENTRIES, NGLB* NGLB ), CRHO23(H%TOTAL_ENTRIES, NGLB* NGLB ))

! set the right potential in POTFAK
    CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
! or possibly read screened potential from file
    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)
!==========================================================================
!   w*_k4,n4(r) w_k1,n1(r)
!==========================================================================

    ! average electrostatic potential for k=k' and n=n'
    ! k1+k4-q might be any reciprocal lattice vector G
    ! in that case the result is shifted by G with respect to the 
    ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
    ! to cure the problem
    CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
    ! set phase factor in fast_aug
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4))
 
    GCHG14=0 ; IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) CRHO14=0
       
    DO N4=1,NGLB
       NB4_INTO_TOT=NB_START-1+N4

       DO N1=1,NGLB
          NB1_INTO_TOT=NB_START-1+N1
          
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N1), W4(N4), GWORK(1), &
                  H, CRHO14(1,N1+NGLB*(N4-1)), CRHOLM,  SIZE(CRHOLM))
          ELSE
             CALL FOCK_CHARGE_NOINT( W1(N1), W4(N4), GWORK(1), CRHOLM,  SIZE(CRHOLM))
          ENDIF
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
          
          CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
               GWORK(1),GCHG14(1,N1+NGLB*(N4-1)),WGWQ%GRID,.FALSE.)
          NFFT=NFFT+1
          
          IF (.NOT.LFULL) THEN
             ! multiply with potential factor
             CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+NGLB*(N4-1)), POTFAK(1))
          ENDIF
       ENDDO
    ENDDO
    
    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
       DO N1=1, NGLB*NGLB, NGLB*NGLB
          N2=MIN(NGLB*NGLB, NGLB*NGLB-N1+1)
          ! use CRHO23 as temporary work array
          CALL APPLY_ONE_CENTER_H( WHF%WDES, H, CRHO14(:,N1:), CRHO23(:,:), N2)
       ENDDO
    ENDIF
!==========================================================================
!     w_k4,n4(r) w*_k1,n1(r) W(r,r')
!==========================================================================
    IF (LFULL) THEN
       IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
          CALL DGEMM('N','N', 2*NP, NGLB*NGLB, 2*NP, 1.0_q, &
               WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1), 2*SIZE(GCHG14,1), &
               0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
       ELSE
          CALL ZGEMM('T','N', NP, NGLB*NGLB, NP, (1.0_q,0.0_q), &
                  WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1), SIZE(GCHG14,1), &
                  (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
       ENDIF
       CALL ZCOPY( SIZE(GCHG14,1)*NGLB*NGLB,  GCHG23(1,1), 1,  GCHG14(1,1), 1)
    ENDIF
    
    ! set the phase factors q=k3-k2
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2))
    CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE, LPHASE)
!==========================================================================
!     w_k3,n3(r') w*_k2,n2(r')
!==========================================================================
    GCHG23=0 ; IF (ASSOCIATED(H)) CRHO23=0

    DO N3=1,NGLB
       NB3_INTO_TOT=NB_START-1+N3
       
       DO N2=1,NGLB
          NB2_INTO_TOT=NB_START-1+N2
          
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             CALL FOCK_CHARGE_ONE_CENTER_NOINT( W3(N3), W2(N2), GWORK(1), & 
                  H, CRHO23(1,N2+NGLB*(N3-1)), CRHOLM,  SIZE(CRHOLM))
          ELSE
             CALL FOCK_CHARGE_NOINT( W3(N3), W2(N2), GWORK(1), CRHOLM,  SIZE(CRHOLM))
          ENDIF
          
          ! apply phase factor e^iGr if required
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
          
          ! now we have w_k3,n3(r') w'*_k2,n2(r')
          ! extract using FFT
          CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
               GWORK(1),GCHG23(1,N2+NGLB*(N3-1)),WGWQ%GRID,.FALSE.)
          NFFT=NFFT+1
          
          GCHG23(1:NP,N2+NGLB*(N3-1))=GCHG23(1:NP,N2+NGLB*(N3-1))*(1.0_q/GRIDHF%NPLWV)
       ENDDO
    ENDDO
!==========================================================================
!    w_k3,n3(r') w*_k2,n2(r')   x     W(r',r) w_k4,n4(r)  w*_k1,n1(r)
!         CHG23                               CHG14
! N.B.: we have actually constructed 
!       GCHG14 = W(r',r) w*_k4,n4(r) w_k1,n1(r),
!       but GCHG14 is conjugated in the GGEMM routine.
!==========================================================================
    CHAM=0

    CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, m_ NP, -one, &
         GCHG14(1,1), m_ SIZE(GCHG14,1), GCHG23(1,1), m_ SIZE(GCHG23,1), &
         zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
    NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*NP*8
    
    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
       CALL APPLY_PHASE_ONE_CENTER(WHF%WDES, H, CRHO23(:,:NGLB*NGLB), & 
            WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4))
       
       CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, H%TOTAL_ENTRIES, -one, &
            CRHO14(1,1), SIZE(CRHO14,1), CRHO23(1,1), SIZE(CRHO23,1), &
            one , CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
       NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*H%TOTAL_ENTRIES*8
    ENDIF
!==========================================================================
!   store in TWOELECTRON4O
!==========================================================================
    DO N1=1,NGLB
       NB1_INTO_TOT=NB_START-1+N1
       DO N2=1,NGLB
          NB2_INTO_TOT=NB_START-1+N2
          DO N3=1,NGLB
             NB3_INTO_TOT=NB_START-1+N3
             DO N4=1,NGLB
                NB4_INTO_TOT=NB_START-1+N4
                ! WRITE(*,*) N1,NB1_INTO_TOT,N2,NB2_INTO_TOT,N3,NB3_INTO_TOT,N4,NB4_INTO_TOT
                ! WRITE(*,'(4F14.7)') CHAM(N1, N4, N2, N3)
                ! TWOELECTRON4O(n4, n3, n2, n1) = < w_k1,n1 w'_k2,n2 | W | w_k2+q,n4 w'_k1-q,n3 >
                ! TWOELECTRON4O(N4,N3,N2,N1)=TWOELECTRON4O(N4,N3,N2,N1)+CHAM(N1,N4,N2,N3)
                TWOELECTRON4O(N4,N3,N2,N1)=-CHAM(N1,N4,N2,N3)
             ENDDO ! N3
          ENDDO ! N4
       ENDDO ! N2
    ENDDO ! N1
    
    ! deallocation
    DEALLOCATE(GCHG14, GCHG23, CHAM, CRHOLM)
    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) DEALLOCATE(CRHO14, CRHO23)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

! test
#ifdef do_fly_epsilon
    IF (ASSOCIATED(WPOTH)) THEN
       ! k-point in full grid
       NQ_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(KPOINTS_FULL%VKPT(:,NQ),KPOINTS_FULL_ORIG)
       ! corresponding k-point in IRZ
       NQ_IN_IRZ=KPOINTS_FULL_ORIG%NEQUIV(NQ_IN_FULL_ORIG)
       ! deallocate response functions outside IRZ
       IF (NQ_IN_IRZ/=NQ) THEN
          CALL DEALLOCATE_RESPONSEFUN(WPOTH%WPOT(NQ))
          WPOTH%LSET(NQ)=.FALSE.
       ENDIF
    ENDIF
#endif
! test
  END SUBROUTINE TWOELECTRON4O_WIJKL



!*********************************************************************
!
! faster routine that restricted to two centre integrals
! this version parallelizes over NQ 
! it is fairly efficient
!
!*********************************************************************

  SUBROUTINE CALCULATE_WIJKL_TWOCENTRE( &
       WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NB_START,NB_STOP,COMEGA,LMAX,IO,WIJKL,PIJKL)

    USE base
    USE pseudo
    USE mpimy
    USE mkpoints
    USE constant
    USE poscar
    USE pot
    USE pawm
    USE wave_high
    USE ini
    USE radial
    USE kpoints_change
    USE full_kpoints
    USE constant
    IMPLICIT NONE
    TYPE (wavedes) WDES
    TYPE (wavespin) W
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt) LATT_CUR
    TYPE (type_info) T_INFO
    TYPE (potcar) P(T_INFO%NTYP)
    TYPE (in_struct) IO
    TYPE (wavedes) WGW           ! descriptor for basis set of response function
    INTEGER NB_START,NB_STOP     ! global band indeces
    COMPLEX(q) COMEGA            ! frequency grid point for PIJKL
    INTEGER LMAX                 ! maximum L index for one center terms
    GDEF, OPTIONAL :: WIJKL(:,:,:,:,:,:)
    GDEF, OPTIONAL :: PIJKL(:,:,:,:)

! local variables
    GDEF, ALLOCATABLE :: TWOe4o(:,:,:,:,:,:)
    GDEF, ALLOCATABLE :: POL_ADD(:,:,:,:)
    TYPE (wavespin) WHF
    TYPE (wavefun1), ALLOCATABLE :: W1(:), W4(:)
    TYPE (wavefun1) :: WTMP
    TYPE (wavedes1), TARGET :: WDESK1, WDESK4
#define collect_all_wavefunctions
#ifdef collect_all_wavefunctions
    TYPE (wavefun1), ALLOCATABLE :: W_all(:,:,:)
    TYPE (wavedes1), TARGET :: WDESK
#endif
    INTEGER :: IERR              ! error indicator
    INTEGER :: NSTRIP            ! block size
    INTEGER :: NGLB              ! total number of bands 
    INTEGER :: N, NP
    INTEGER :: NPOS1, NSTRIP1    ! base index and width of the n1 block
    INTEGER :: K1, K4, NQ_BASE, K4_COLLECT, NQ_LOCAL, NQ, NQ_COLLECT
    TYPE (wavedes1)    WGWQ
    REAL(q) :: SCALE
    REAL(q) :: NFFTW             ! number of FFTs for  wavefunctions
    REAL(q) :: NFLOAT, NFFT      ! number of BLAS3 operations in 4 orbital routines

    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)  ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)  ! charge 
    GDEF      , ALLOCATABLE :: CRHO14(:,:)    ! one center charge
    GDEF      , ALLOCATABLE :: CRHO23(:,:)    ! one center charge
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL :: LFULL
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    INTEGER :: ISP1, ISP2
    INTEGER :: R1, N1, N2, N3, N4
    COMPLEX :: CPHASE
    CHARACTER (LEN=16) :: FILENAME_NODE
    GDEF, ALLOCATABLE  :: TWOE_4WANNIER(:,:,:,:,:,:,:)
    ! one-center terms
    TYPE (one_center_handle), POINTER :: H => NULL()
    !for PIJKL
    INTEGER I,J,K,IP,IPP,NW,NB,NKP,NB1,NB2
    INTEGER IP_,IPP_
    COMPLEX(q) :: CTMP 
    GDEF, ALLOCATABLE :: UK1(:,:),UK4(:,:)

     
!=======================================================================
! preparation of four-orbital related quantities
!=======================================================================
    IF (LMAX>=0) THEN
       CALL SET_UP_ONE_CENTER_H(WDES, P, T_INFO, LMAX, H)
    ENDIF

    CALL CHECK_FULL_KPOINTS

    CALL START_TIMING("WIJKL+")

    ! switch to WDES_FOCK
    WHF=W; WHF%WDES => WDES_FOCK
!=======================================================================

130 FORMAT (5X, //, &
         &'----------------------------------------------------', &
         &'----------------------------------------------------'//)
    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    ! total number of bands requested
    NGLB=NB_STOP-NB_START+1

#ifdef collect_all_wavefunctions
    ALLOCATE(W_all(NGLB,WDES%NKPTS,WDES%ISPIN))
    ! collect all wave functions
    DO ISP1=1,WDES%ISPIN
       DO K1=1,WDES%NKPTS
          CALL SETWDES(WHF%WDES,WDESK,K1)
          DO N=1,NGLB
             CALL NEWWAV(W_all(N,K1,ISP1),WDESK,.FALSE.) 
          ENDDO
          CALL W1_GATHER_GLB_NOCR( WHF, NB_START, NB_STOP, ISP1, W_all(:,K1,ISP1) )
       ENDDO
    ENDDO
#endif

    ! allocate storage for W1, W4
    ALLOCATE(W1(NGLB),W4(NGLB))
    DO N=1,NGLB 
       CALL NEWWAV(W1(N),WDESK1,.TRUE.)
       CALL NEWWAV(W4(N),WDESK4,.TRUE.)
    ENDDO

    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,4I5)') 'allocating two-electron 4 orbital integral table', &
   & NGLB,NGLB,NGLB,NGLB

    ! TWOe4o(n4, n3, n2, n1) = 
    !  \int dk1 dk2 dk3 dk4 < w_k1,n1 w'_k2,n2 | W | w_k2+q,n4 w'_k1-q,n3 >
    ALLOCATE(TWOe4o(NGLB,NGLB,NGLB,NGLB,WDES%ISPIN,WDES%ISPIN))
    TWOe4o=0._q

    IF (L2E4W_ALL) THEN
       IF(.NOT.ALLOCATED(TWOE_4WANNIER)) &
            ALLOCATE(TWOE_4WANNIER(WDES%NKPTS,NGLB,NGLB,NGLB,NGLB,WDES%ISPIN,WDES%ISPIN))
       TWOE_4WANNIER=0
    ENDIF

    SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
    IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2

    NFLOAT=0 ; NFFT=0

!calculate PIJKL
    IF (PRESENT(PIJKL) ) THEN 
       ALLOCATE(POL_ADD(NGLB,NGLB,NGLB,NGLB)) 
       POL_ADD=0
    ENDIF
       IF (IO%IU0>=0.AND.ASSOCIATED(WPOTH).AND. .NOT.ASSOCIATED(H)) THEN
           WRITE(*,*)
           IF ( PRESENT( PIJKL ) ) THEN
              WRITE(*,'(" Caluclating WIJKL and PIJKL  for w=",(2F10.4))')REAL(COMEGA),AIMAG(COMEGA)
           ELSE
              WRITE(*,'(" Caluclating UIJKL for w=",(2F10.4))')REAL(COMEGA),AIMAG(COMEGA)
           ENDIF 
           WRITE(*,*)
       ENDIF


!==========================================================================
! loop of q vectors
!==========================================================================
    CALL START_TIMING("G")
!==========================================================================
    ! loop over all q-point blocks block size is NB_PAR (= number of cores)
    ! these are going to be distributed over the nodes when
    ! executed in parallel
    nqloop: DO NQ_BASE=1,((WDES%NKPTS+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR,WDES%NB_PAR
       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,*)
          WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') NQ_BASE,KPOINTS%VKPT(:,NQ_BASE)
       ENDIF
          
       ! k-point treated on current node
       NQ=NQ_BASE+WDES%NB_LOW-1
       IF (NQ>WDES%NKPTS) NQ=-1

       LFULL=.FALSE.
       IF (NQ>=0) THEN
          CALL SETWDES(WGW, WGWQ, NQ)
          NP =WGWQ%NGVECTOR

          ! set the right potential in POTFAK
          CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG_STORE(1), POTFAK ) ! FSG = FSG_STORE(1) or 0._q (convergence correction)?
          ! or possibly read screened potential from file
          IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)

          ALLOCATE( GCHG14(NP, NGLB* NGLB ), GCHG23(NP, NGLB* NGLB ), CHAM(NGLB, NGLB, NGLB, NGLB))
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             ALLOCATE(CRHO14(H%TOTAL_ENTRIES, NGLB* NGLB ), CRHO23(H%TOTAL_ENTRIES, NGLB* NGLB )) 
          ELSE
             ALLOCATE(CRHO14(1, NGLB* NGLB ), CRHO23(1, NGLB* NGLB )) 
          ENDIF

       ENDIF

       DO ISP1=1,WDES%ISPIN

       IF (NQ>=0) THEN
          GCHG14=0 ; CRHO14=0
       ENDIF

       DO K1=1, WDES%NKPTS
          CALL GWPROGRESS(IO%IU0, NQ, WDES%NKPTS, K1, WDES%NKPTS)
!==========================================================================
! maybe most difficult
! collect bands [NB_START,NB_STOP] (global index) at K1 and K4
! K4 is distributed over cores
!==========================================================================
          CALL SETWDES(WHF%WDES,WDESK1,K1)
#ifndef collect_all_wavefunctions
          CALL W1_GATHER_DISTR( WHF, NB_START, NB_STOP, ISP1, W1)
#else
          CALL SETWDES(WHF%WDES,WDESK,K1)
          DO N=1,NGLB
             CALL W1_COPY(W_all(N,K1,ISP1),W1(N))
             CALL FFTWAV_W1(W1(N))
          ENDDO
#endif
          K4=-1                  ! determine which k-point treated locally
          ! loop over block
          DO NQ_COLLECT=NQ_BASE,MIN(NQ_BASE+WDES%NB_PAR-1,WDES%NKPTS)
             NQ_LOCAL=NQ_COLLECT-NQ_BASE+1

             K4_COLLECT=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ_COLLECT),KPOINTS_FULL)
#ifndef collect_all_wavefunctions
             CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)
             CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP1, W4, NQ_LOCAL)
#endif
             ! round robin distribution (my cpu id = NQ_LOCAL)
             IF (NQ_LOCAL==WDES%NB_LOW) K4=K4_COLLECT
          ENDDO
#ifdef collect_all_wavefunctions
          IF (K4>=0) THEN
             CALL SETWDES(WHF%WDES,WDESK,K4)
             CALL SETWDES(WHF%WDES,WDESK4,K4)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K4,ISP1),W4(N))
                CALL FFTWAV_W1(W4(N))
             ENDDO
          ENDIF
#endif
!==========================================================================
! calculate density corresponding to  w_k4,n4(r) w*_k1,n1(r)
! into CHG14
!==========================================================================
          IF (NQ>=0) THEN
             CALL SETWDES(WHF%WDES,WDESK4,K4)
             CALL ONEELECTRON_CHARGE( &
                  WHF,H,LATT_CUR, WGW, &
                  W1, K1, W4, K4, NB_START, NGLB, NFFT, &
                  GCHG14, CRHO14)
                    
             !=============================================================
             ! calculation of PIJKL
             !=============================================================
             IF ( PRESENT(PIJKL) .AND. .NOT. LCRPA ) THEN
                CALL CALCULATE_PIJKL(WHF,WDES,LATT_CUR,KPOINTS%ISMEAR,KPOINTS%SIGMA,COMEGA,&
                                     K1,K4,ISP1,NB_START,NB_STOP,POL_ADD,IO)
             ENDIF
          ENDIF
       ENDDO !  k1

       IF (NQ>=0) THEN
          DO N=1,NGLB*NGLB
             GCHG14(1:NP,N)=GCHG14(1:NP,N)*SQRT((1.0_q/GRIDHF%NPLWV))
          ENDDO
!==========================================================================
! multiply with W and store results in CHG23, CRHO23
!==========================================================================
! apply W potential on centre densities
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             CALL APPLY_ONE_CENTER_H_NOOVERWRT( WHF%WDES, H, CRHO14(:,:), CRHO23(:,:), NGLB*NGLB)
          ENDIF

          IF (LFULL) THEN
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                CALL DGEMM('N','N', 2*NP, NGLB*NGLB, 2*NP, 1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, NGLB*NGLB, NP, (1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
             ENDIF
          ELSE
             CALL ZCOPY( SIZE(GCHG14,1)*NGLB*NGLB,  GCHG14(1,1), 1, GCHG23(1,1), 1)
             ! multiply with potential factor
             DO N=1,NGLB*NGLB
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG23(1,N), POTFAK(1))
             ENDDO
          ENDIF
!==========================================================================
!  w_k4,n4(r')  w*_k1,n1(r')   W_(r',r) sum_k1 w_k4,n4(r)  w*_k1,n1(r)
!           CHG14                              CHG23
!
! N.B.: we have actually constructed GCHG23  = W(r',r) w*_k4,n4(r) w_k1,n1(r)
!       but GCHG23 is conjugated in the GGEMM routine
!==========================================================================
          CHAM=0
          CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, m_ NP, one, &
               GCHG23(1,1), m_ SIZE(GCHG23,1), GCHG14(1,1), m_ SIZE(GCHG14,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*NP*8
          
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, H%TOTAL_ENTRIES, one, &
                  CRHO23(1,1), SIZE(CRHO23,1), CRHO14(1,1), SIZE(CRHO14,1), &
                  one , CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
             NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*H%TOTAL_ENTRIES*8
          ENDIF
          DO N2=1,NGLB
             DO N3=1,NGLB
                DO N4=1,NGLB
                   DO N1=1,NGLB
                      TWOe4o(N1,N4,N3,N2,ISP1,ISP1)=TWOe4o(N1,N4,N3,N2,ISP1,ISP1)+CHAM(N1,N4,N3,N2)*SCALE*SCALE*SCALE
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
          
          IF (L2E4W_ALL) THEN
             DO R1=1,WDES%NKPTS
                CPHASE= EXP(CITPI* (+SUM(WDES%VKPT(:,K1)*WPOS(:,R1)-SUM(WDES%VKPT(:,K4)*WPOS(:,R1)))))
                
                DO N1=1,NGLB
                   DO N2=1,NGLB
                      DO N3=1,NGLB
                         DO N4=1,NGLB
                            TWOE_4WANNIER(R1,N4,N3,N2,N1,ISP1,ISP1)=TWOE_4WANNIER(R1,N4,N3,N2,N1,ISP1,ISP1)+CHAM(N1,N4,N3,N2)*SCALE*SCALE*SCALE*CPHASE
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDIF
       ENDIF
!==========================================================================
!       loop over second spin not neccessary, because of symmetry
!        DO ISP2 = 1, WDES%ISPIN 
       IF (WDES%ISPIN==2 .AND. ISP1==1) THEN
!          WRITE(*,*) 'internal error in CALCULATE_WIJKL_TWOCENTRE: spin polarized version is not tested'
          ISP2=2
          IF (NQ>=0) THEN
             GCHG14=0; CRHO14=0
          ENDIF

          DO K1=1, WDES%NKPTS
             CALL SETWDES(WHF%WDES,WDESK1,K1)
#ifndef collect_all_wavefunctions
             CALL W1_GATHER_DISTR( WHF, NB_START, NB_STOP, ISP2, W1)
#else
             CALL SETWDES(WHF%WDES,WDESK,K1)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K1,ISP2),W1(N))
                CALL FFTWAV_W1(W1(N))
             ENDDO
#endif
             K4=-1                  ! determine which k-point treated locally
             ! loop over block
             DO NQ_COLLECT=NQ_BASE,MIN(NQ_BASE+WDES%NB_PAR-1,WDES%NKPTS)
                NQ_LOCAL=NQ_COLLECT-NQ_BASE+1
                
                K4_COLLECT=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ_COLLECT),KPOINTS_FULL)
#ifndef collect_all_wavefunctions
                CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)
                CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP2, W4, NQ_LOCAL)
#endif
             ! round robin distribution (my cpu id = NQ_LOCAL)
                IF (NQ_LOCAL==WDES%NB_LOW) K4=K4_COLLECT
             ENDDO
#ifdef collect_all_wavefunctions
             IF (K4>=1) THEN
                CALL SETWDES(WHF%WDES,WDESK,K4)
                CALL SETWDES(WHF%WDES,WDESK4,K4)
                DO N=1,NGLB
                   CALL W1_COPY(W_all(N,K4,ISP2),W4(N))
                   CALL FFTWAV_W1(W4(N))
                ENDDO
             ENDIF
#endif
!==========================================================================
! calculate density corresponding to  w_k4,n4(r) w*_k1,n1(r)
! into CHG14
!==========================================================================
             IF (NQ>=0) THEN
                CALL SETWDES(WHF%WDES,WDESK4,K4)
                CALL ONEELECTRON_CHARGE( &
                     WHF,H,LATT_CUR,WGW, &
                     W1, K1, W4, K4, NB_START, NGLB, NFFT, &
                     GCHG14, CRHO14)
             ENDIF
          ENDDO

          IF (NQ>=0) THEN
             DO N=1,NGLB*NGLB
                GCHG14(1:NP,N)=GCHG14(1:NP,N)*SQRT((1.0_q/GRIDHF%NPLWV))
             ENDDO

             CHAM=0
             CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, m_ NP, one, &
                  GCHG23(1,1), m_ SIZE(GCHG23,1), GCHG14(1,1), m_ SIZE(GCHG14,1), &
                  zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
             NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*NP*8
             
             IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
                CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB, H%TOTAL_ENTRIES, one, &
                     CRHO23(1,1), SIZE(CRHO23,1), CRHO14(1,1), SIZE(CRHO14,1), &
                     one , CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
                NFLOAT=NFLOAT+NGLB*NGLB*NGLB*NGLB*H%TOTAL_ENTRIES*8
             ENDIF
             DO N3=1,NGLB
                DO N2=1,NGLB
                   DO N4=1,NGLB
                      DO N1=1,NGLB
                         TWOe4o(N1,N4,N2,N3,ISP1,ISP2)=TWOe4o(N1,N4,N2,N3,ISP1,ISP2)+CHAM(N1,N4,N2,N3)*SCALE*SCALE*SCALE
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
             
             IF (L2E4W_ALL) THEN
                DO R1=1,WDES%NKPTS
                   CPHASE= EXP(CITPI* (+SUM(WDES%VKPT(:,K1)*WPOS(:,R1))-SUM(WDES%VKPT(:,K4)*WPOS(:,R1))))
                   
                   DO N1=1,NGLB
                      DO N2=1,NGLB
                         DO N3=1,NGLB
                            DO N4=1,NGLB
                               TWOE_4WANNIER(R1,N4,N3,N2,N1,ISP1,ISP2)=TWOE_4WANNIER(R1,N4,N3,N2,N1,ISP1,ISP2)+CHAM(N1,N4,N3,N2)*SCALE*SCALE*SCALE*CPHASE
                            ENDDO
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
             ENDIF
          ENDIF

       ENDIF ! ISP=2
!       ENDDO !isp2

    ENDDO ! spin

    IF (NQ>=0) THEN
       DEALLOCATE( GCHG14, GCHG23, CRHO14, CRHO23, CHAM)
    ENDIF
    ENDDO nqloop
!==========================================================================
    CALL STOP_TIMING("WIJKL+",IO%IU6,XMLTAG='total')

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT  , 1))

    CALLMPI( M_sum_g(WGW%COMM_INTER, TWOe4o, SIZE(TWOe4o)))

    IF (L2E4W_ALL) THEN
      CALLMPI( M_sum_g(WGW%COMM_INTER, TWOE_4WANNIER, SIZE(TWOE_4WANNIER) ))
    ENDIF

    IF (PRESENT(WIJKL)) THEN
       IF (SIZE(WIJKL,1)>=NGLB.AND.SIZE(WIJKL,2)>=NGLB.AND. &
      &     SIZE(WIJKL,3)>=NGLB.AND.SIZE(WIJKL,4)>=NGLB) &
      &   WIJKL(1:NGLB,1:NGLB,1:NGLB,1:NGLB,1:WDES%ISPIN,1:WDES%ISPIN)=&
         TWOe4o(:,:,:,:,1:WDES%ISPIN,1:WDES%ISPIN)
    ENDIF

    IF ( PRESENT (PIJKL) ) THEN
      CALLMPI( M_sum_g(WGW%COMM_INTER, POL_ADD, SIZE(POL_ADD)))
       IF (SIZE(PIJKL,1)>=NGLB.AND.SIZE(PIJKL,2)>=NGLB.AND. &
      &     SIZE(PIJKL,3)>=NGLB.AND.SIZE(PIJKL,4)>=NGLB) &
      &   PIJKL(1:NGLB,1:NGLB,1:NGLB,1:NGLB)=POL_ADD(:,:,:,:)
    ENDIF
    
   

!==========================================================================
! deallocation
!==========================================================================
    DEALLOCATE(TWOe4o)
    IF ( PRESENT(PIJKL) )DEALLOCATE(POL_ADD) 
    IF (L2E4W_ALL) DEALLOCATE(TWOE_4WANNIER)
   DO N=1,NGLB
       CALL DELWAV(W4(N) ,.TRUE.)
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1,W4)


    IF (ASSOCIATED(H)) CALL DEALLOCATE_ONE_CENTER_H(H)
#ifdef collect_all_wavefunctions
    DO ISP1=1,WDES%ISPIN
    DO K1=1,WDES%NKPTS
       DO N=1,NGLB
          CALL DELWAV(W_all(N,K1,ISP1),.FALSE.) 
       ENDDO
    ENDDO
    ENDDO
    DEALLOCATE(W_all)
#endif

    RETURN
  END SUBROUTINE CALCULATE_WIJKL_TWOCENTRE



!*********************************************************************
!
! faster routine that allows to calculate four centre integrals
! it stores the "intermediates" 
! this version parallelizes over NQ and is very efficient,
! however, it might require quite some memory to store
! the intermediadates
! also presently the Wannier transformations are dead slow
!  TWOE_4WANNIER_SET might need an updated
! to gain more speed
!
!*********************************************************************

  SUBROUTINE CALCULATE_WIJKL_FOURCENTRE( &
       WDES,W,KPOINTS,LATT_CUR,T_INFO,P,WGW,NB_START,NB_STOP,LMAX,WIJKL,IO)

    USE base
    USE pseudo
    USE mpimy
    USE mkpoints
    USE constant
    USE poscar
    USE pot
    USE pawm
    USE wave_high
    USE ini
    USE radial
    USE kpoints_change
    USE full_kpoints
    USE constant
    IMPLICIT NONE
    TYPE (wavedes) WDES
    TYPE (wavespin) W
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt) LATT_CUR
    TYPE (type_info) T_INFO
    TYPE (potcar) P(T_INFO%NTYP)
    TYPE (in_struct) IO
    TYPE (wavedes) WGW           ! descriptor for basis set of response function
    INTEGER NB_START,NB_STOP     ! global band indeces
    INTEGER LMAX                 ! maximum L index for one center terms
    GDEF, OPTIONAL :: WIJKL(:,:,:,:)

! local variables
    GDEF, ALLOCATABLE :: TWOe4o(:,:,:,:,:,:)
    TYPE (wavespin) WHF
    TYPE (wavefun1), ALLOCATABLE :: W1(:), W4(:)
    TYPE (wavefun1) :: WTMP
    TYPE (wavedes1), TARGET :: WDESK1, WDESK4
#define collect_all_wavefunctions
#ifdef collect_all_wavefunctions
    TYPE (wavefun1), ALLOCATABLE :: W_all(:,:,:)
    TYPE (wavedes1), TARGET :: WDESK
#endif
    INTEGER :: IERR              ! error indicator
    INTEGER :: NSTRIP            ! block size
    INTEGER :: NGLB              ! total number of bands 
    INTEGER :: N, NP
    INTEGER :: NPOS1, NSTRIP1    ! base index and width of the n1 block
    INTEGER :: K1, K4, NQ_BASE, K4_COLLECT, NQ_LOCAL, NQ, NQ_COLLECT
    INTEGER :: K4_STORE(WDES%NKPTS)
    TYPE (wavedes1)    WGWQ
    REAL(q) :: SCALE
    REAL(q) :: NFFTW             ! number of FFTs for  wavefunctions
    REAL(q) :: NFLOAT, NFFT      ! number of BLAS3 operations in 4 orbital routines

    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:,:)  ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:,:)  ! charge 
    GDEF      , ALLOCATABLE :: CRHO14(:,:,:)  ! one center charge
    GDEF      , ALLOCATABLE :: CRHO23(:,:,:)  ! one center charge
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:,:),TWOe4o_ADD(:,:,:,:)
    LOGICAL :: LFULL
    REAL(q) :: POTFAK(GRIDHF%MPLWV)           ! 1/(G+dk)**2 (G)
    INTEGER :: ISP1, ISP2
    INTEGER :: R1, N1, N2, N3, N4
    COMPLEX :: CPHASE
    CHARACTER (LEN=16) :: FILENAME_NODE
    ! one-center terms
    TYPE (one_center_handle), POINTER :: H => NULL()

!=======================================================================
! preparation of four-orbital related quantities
!=======================================================================
    IF (LMAX>=0) THEN
       CALL SET_UP_ONE_CENTER_H(WDES, P, T_INFO, LMAX, H)
    ENDIF

    CALL CHECK_FULL_KPOINTS

    CALL START_TIMING("WIJKL+")

    ! switch to WDES_FOCK
    WHF=W; WHF%WDES => WDES_FOCK
!=======================================================================

130 FORMAT (5X, //, &
         &'----------------------------------------------------', &
         &'----------------------------------------------------'//)
    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    ! total number of bands requested
    NGLB=NB_STOP-NB_START+1

#ifdef collect_all_wavefunctions
    ALLOCATE(W_all(NGLB,WDES%NKPTS,WDES%ISPIN))
    ! collect all wave functions
    DO ISP1=1,WDES%ISPIN
       DO K1=1,WDES%NKPTS
          CALL SETWDES(WHF%WDES,WDESK,K1)
          DO N=1,NGLB
             CALL NEWWAV(W_all(N,K1,ISP1),WDESK,.FALSE.) 
          ENDDO
          CALL W1_GATHER_GLB_NOCR( WHF, NB_START, NB_STOP, ISP1, W_all(:,K1,ISP1) )
       ENDDO
    ENDDO
#endif

    ! allocate storage for W1, W4
    ALLOCATE(W1(NGLB),W4(NGLB))
    DO N=1,NGLB 
       CALL NEWWAV(W1(N),WDESK1,.TRUE.)
       CALL NEWWAV(W4(N),WDESK4,.TRUE.)
    ENDDO

    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,4I5)') 'allocating two-electron 4 orbital integral table', &
   & NGLB,NGLB,NGLB,NGLB

    ! TWOe4o(n4, n3, n2, n1) = 
    !  \int dk1 dk2 dk3 dk4 < w_k1,n1 w'_k2,n2 | W | w_k2+q,n4 w'_k1-q,n3 >
    ALLOCATE(TWOe4o(NGLB,NGLB,NGLB,NGLB,WDES%ISPIN,WDES%ISPIN),CHAM(NGLB, NGLB, NGLB, NGLB, WDES%NKPTS ))
    TWOe4o=0._q

    IF (L2E4W_ALL) THEN
       CALL TWOE_4WANNIER_ALLOCATE( WDES, NGLB)
       ALLOCATE(TWOe4o_ADD(NGLB,NGLB,NGLB,NGLB))
    ENDIF

    SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
    IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2

    NFLOAT=0 ; NFFT=0
!==========================================================================
! loop of q vectors
!==========================================================================
    CALL START_TIMING("G")
!==========================================================================
    ! loop over all q-point blocks block size is NB_PAR (= number of cores)
    ! these are going to be distributed over the nodes when
    ! executed in parallel
    nqloop: DO NQ_BASE=1,((WDES%NKPTS+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR,WDES%NB_PAR
       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,*)
          WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') NQ_BASE,KPOINTS%VKPT(:,NQ_BASE)
       ENDIF

       ! k-point treated on current node
       NQ=NQ_BASE+WDES%NB_LOW-1
       IF (NQ>WDES%NKPTS) NQ=-1

       LFULL=.FALSE.
       IF (NQ>=0) THEN
          CALL SETWDES(WGW, WGWQ, NQ)
          NP =WGWQ%NGVECTOR
          ! set the right potential in POTFAK
          CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG_STORE(1), POTFAK ) ! FSG = FSG_STORE(1) or 0._q (convergence correction)?
          ! or possibly read screened potential from file
          IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)

          ALLOCATE( GCHG14(NP, NGLB* NGLB, WDES%NKPTS ), GCHG23(NP, NGLB* NGLB, WDES%NKPTS ))
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             ALLOCATE(CRHO14(H%TOTAL_ENTRIES, NGLB* NGLB, WDES%NKPTS ), CRHO23(H%TOTAL_ENTRIES, NGLB* NGLB, WDES%NKPTS )) 
          ELSE
             ALLOCATE(CRHO14(1, NGLB* NGLB, WDES%NKPTS ), CRHO23(1, NGLB* NGLB, WDES%NKPTS )) 
          ENDIF
       ENDIF


       DO ISP1=1,WDES%ISPIN

       IF (NQ>=0) THEN
          GCHG14=0 ; CRHO14=0
       ENDIF

       DO K1=1, WDES%NKPTS
          CALL GWPROGRESS(IO%IU0, NQ, WDES%NKPTS, K1, WDES%NKPTS)
!==========================================================================
! maybe most difficult
! collect bands [NB_START,NB_STOP] (global index) at K1 and K4
! K4 is distributed over cores
!==========================================================================
          CALL SETWDES(WHF%WDES,WDESK1,K1)
#ifndef collect_all_wavefunctions
          CALL W1_GATHER_DISTR( WHF, NB_START, NB_STOP, ISP1, W1)
#else
          CALL SETWDES(WHF%WDES,WDESK,K1)
          DO N=1,NGLB
             CALL W1_COPY(W_all(N,K1,ISP1),W1(N))
             CALL FFTWAV_W1(W1(N))
          ENDDO
#endif
          K4=-1                  ! determine which k-point treated locally
          ! loop over block
          DO NQ_COLLECT=NQ_BASE,MIN(NQ_BASE+WDES%NB_PAR-1,WDES%NKPTS)
             NQ_LOCAL=NQ_COLLECT-NQ_BASE+1

             K4_COLLECT=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ_COLLECT),KPOINTS_FULL)
#ifndef collect_all_wavefunctions
             CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)
             CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP1, W4, NQ_LOCAL)
#endif
             ! round robin distribution (my cpu id = NQ_LOCAL)
             IF (NQ_LOCAL==WDES%NB_LOW) K4=K4_COLLECT
          ENDDO
          K4_STORE(K1)=K4
#ifdef collect_all_wavefunctions
          IF (K4>=0) THEN
             CALL SETWDES(WHF%WDES,WDESK,K4)
             CALL SETWDES(WHF%WDES,WDESK4,K4)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K4,ISP1),W4(N))
                CALL FFTWAV_W1(W4(N))
             ENDDO
          ENDIF
#endif
!==========================================================================
! calculate density corresponding to  w_k4,n4(r) w*_k1,n1(r)
! into CHG14
!==========================================================================
          IF (NQ>=0) THEN
             CALL SETWDES(WHF%WDES,WDESK4,K4)
             CALL ONEELECTRON_CHARGE( &
                  WHF,H,LATT_CUR, WGW, &
                  W1, K1, W4, K4, NB_START, NGLB, NFFT, &
                  GCHG14(:,:,K1), CRHO14(:,:,K1))
          ENDIF
       ENDDO !  k1

       IF (NQ>=0) THEN
          DO K1=1,WDES%NKPTS
             DO N=1,NGLB*NGLB
                GCHG14(1:NP,N,K1)=GCHG14(1:NP,N,K1)*SQRT((1.0_q/GRIDHF%NPLWV))
             ENDDO
          ENDDO
!==========================================================================
! multiply with W and store results in CHG23, CRHO23
!==========================================================================
! apply W potential on centre densities
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
          DO K1=1,WDES%NKPTS
             CALL APPLY_ONE_CENTER_H_NOOVERWRT( WHF%WDES, H, CRHO14(:,:,K1), CRHO23(:,:,K1), NGLB*NGLB*WDES%NKPTS)
          ENDDO
          ENDIF

          IF (LFULL) THEN
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                CALL DGEMM('N','N', 2*NP, NGLB*NGLB*WDES%NKPTS, 2*NP, 1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1,1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, NGLB*NGLB*WDES%NKPTS, NP, (1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,1,1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1,1), SIZE(GCHG23,1) )
             ENDIF
          ELSE
             CALL ZCOPY( SIZE(GCHG14,1)*NGLB*NGLB*WDES%NKPTS,  GCHG14(1,1,1), 1, GCHG23(1,1,1), 1)
             ! multiply with potential factor
             DO K1=1,WDES%NKPTS
             DO N=1,NGLB*NGLB
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG23(1,N,K1), POTFAK(1))
             ENDDO
             ENDDO
          ENDIF
       ENDIF
!==========================================================================
!  w_k4,n4(r')  w*_k1,n1(r')   W_(r',r) sum_k1 w_k4,n4(r)  w*_k1,n1(r)
!           CHG14                              CHG23
!
! N.B.: we have actually constructed GCHG23  = W(r',r) w*_k4,n4(r) w_k1,n1(r)
!       but GCHG23 is conjugated in the GGEMM routine
!==========================================================================
       DO K1=1,WDES%NKPTS
          CHAM=0
          IF (NQ>=0) THEN
          CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB*WDES%NKPTS, m_ NP, one, &
               GCHG23(1,1,K1), m_ SIZE(GCHG23,1), GCHG14(1,1,1), m_ SIZE(GCHG14,1), &
               zero, CHAM, SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+(NGLB*NGLB)**2*WDES%NKPTS*NP*8
!         it is straightforward and simple to calculate all elements of CHAM in one go
!         simply add one more (last) index to CHAM, but this might require huge amounts of memory
!          CALL GGEMM(trans,'N',  NGLB*NGLB*WDES%NKPTS, NGLB*NGLB*WDES%NKPTS, m_ NP, one, &
!               GCHG23(1,1,1), m_ SIZE(GCHG23,1), GCHG14(1,1,K1), m_ SIZE(GCHG14,1), &
!               zero, CHAM, SIZE(CHAM,1)*SIZE(CHAM,2)*SIZE(CHAM,3))
!          NFLOAT=NFLOAT+(NGLB*NGLB*WDES%NKPTS)**2*NP*8
          
          IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
             CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB*WDES%NKPTS, H%TOTAL_ENTRIES, one, &
                  CRHO23(1,1,K1), SIZE(CRHO23,1), CRHO14(1,1,1), SIZE(CRHO14,1), &
                  one , CHAM, SIZE(CHAM,1)*SIZE(CHAM,2))
             NFLOAT=NFLOAT+(NGLB*NGLB)**2*WDES%NKPTS*H%TOTAL_ENTRIES*8
          ENDIF

          CHAM=CHAM*SCALE*SCALE*SCALE

          DO K4=1,WDES%NKPTS
          DO N1=1,NGLB
             DO N2=1,NGLB
                DO N3=1,NGLB
                   DO N4=1,NGLB
                      TWOe4o(N4,N3,N2,N1,ISP1,ISP1)=TWOe4o(N4,N3,N2,N1,ISP1,ISP1)+CHAM(N1,N4,N3,N2,K4)
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
          ENDDO
          ENDIF
          
          IF (L2E4W_ALL) THEN
             DO K4=1,WDES%NKPTS
                DO N1=1,NGLB
                   DO N2=1,NGLB
                      DO N3=1,NGLB
                         DO N4=1,NGLB
                            TWOe4o_ADD(N4,N3,N2,N1)=CHAM(N1,N4,N3,N2,K4)
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
                ! index order for second pair (K4) get by trial and error
                CALL TWOE_4WANNIER_SET( WGW, WDES, TWOe4o_ADD, K1, K4_STORE(K4), K4, K4_STORE(K1), NGLB)
             ENDDO
          ENDIF
       ENDDO
!==========================================================================
       IF (WDES%ISPIN==2 .AND. ISP1==1) THEN
          WRITE(*,*) 'internal error in CALCULATE_WIJKL_TWOCENTRE: spin polarized version is not tested'

          IF (NQ>=0) THEN
             GCHG14=0; CRHO14=0
          ENDIF

          DO K1=1, WDES%NKPTS
             ISP2=2
             CALL SETWDES(WHF%WDES,WDESK1,K1)
#ifndef collect_all_wavefunctions
             CALL W1_GATHER_DISTR( WHF, NB_START, NB_STOP, ISP2, W1)
#else
             CALL SETWDES(WHF%WDES,WDESK,K1)
             DO N=1,NGLB
                CALL W1_COPY(W_all(N,K1,ISP2),W1(N))
                CALL FFTWAV_W1(W1(N))
             ENDDO
#endif
             K4=-1                  ! determine which k-point treated locally
             ! loop over block
             DO NQ_COLLECT=NQ_BASE,MIN(NQ_BASE+WDES%NB_PAR-1,WDES%NKPTS)
                NQ_LOCAL=NQ_COLLECT-NQ_BASE+1
                
                K4_COLLECT=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ_COLLECT),KPOINTS_FULL)
#ifndef collect_all_wavefunctions
                CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)
                CALL W1_GATHER_KSEL( WHF, NB_START, NB_STOP, ISP2, W4, NQ_LOCAL)
#endif
             ! round robin distribution (my cpu id = NQ_LOCAL)
                IF (NQ_LOCAL==WDES%NB_LOW) K4=K4_COLLECT
             ENDDO
#ifdef collect_all_wavefunctions
             IF (K4>=1) THEN
                CALL SETWDES(WHF%WDES,WDESK,K4)
                CALL SETWDES(WHF%WDES,WDESK4,K4)
                DO N=1,NGLB
                   CALL W1_COPY(W_all(N,K4,ISP2),W4(N))
                   CALL FFTWAV_W1(W4(N))
                ENDDO
             ENDIF
#endif
!==========================================================================
! calculate density corresponding to  w_k4,n4(r) w*_k1,n1(r)
! into CHG14
!==========================================================================
             IF (NQ>=0) THEN
                CALL SETWDES(WHF%WDES,WDESK4,K4)
                CALL ONEELECTRON_CHARGE( &
                     WHF,H,LATT_CUR,WGW, &
                     W1, K1, W4, K4, NB_START, NGLB, NFFT, &
                     GCHG14(:,:,K1), CRHO14(:,:,K1))
             ENDIF
          ENDDO

          IF (NQ>=0) THEN
             DO K1=1,WDES%NKPTS
                DO N=1,NGLB*NGLB
                   GCHG14(1:NP,N,K1)=GCHG14(1:NP,N,K1)*SQRT((1.0_q/GRIDHF%NPLWV))
                ENDDO
             ENDDO
          ENDIF
          CHAM=0
             
          DO K1=1,WDES%NKPTS
             IF (NQ>=0) THEN
             CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB*WDES%NKPTS, m_ NP, one, &
                  GCHG23(1,1,K1), m_ SIZE(GCHG23,1), GCHG14(1,1,1), m_ SIZE(GCHG14,1), &
                  zero, CHAM, SIZE(CHAM,1)*SIZE(CHAM,2))
             NFLOAT=NFLOAT+(NGLB*NGLB)**2*WDES%NKPTS*NP*8
             
             IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
                CALL GGEMM(trans,'N',  NGLB*NGLB, NGLB*NGLB*WDES%NKPTS, H%TOTAL_ENTRIES, one, &
                     CRHO23(1,1,K1), SIZE(CRHO23,1), CRHO14(1,1,1), SIZE(CRHO14,1), &
                     one , CHAM, SIZE(CHAM,1)*SIZE(CHAM,2))
                NFLOAT=NFLOAT+(NGLB*NGLB)**2*WDES%NKPTS*H%TOTAL_ENTRIES*8
             ENDIF

             CHAM=CHAM*SCALE*SCALE*SCALE
             
             DO K4=1,WDES%NKPTS
             DO N1=1,NGLB
                DO N2=1,NGLB
                   DO N3=1,NGLB
                      DO N4=1,NGLB
                         TWOe4o(N4,N3,N2,N1,ISP1,ISP2)=TWOe4o(N4,N3,N2,N1,ISP1,ISP2)+CHAM(N1,N4,N3,N2,K4)
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
             ENDDO
             ENDIF
             IF (L2E4W_ALL) THEN
                
             ENDIF
          ENDDO

       END IF ! ISP=2

    ENDDO ! spin

    IF (NQ>=0) THEN
       DEALLOCATE( GCHG14, GCHG23, CRHO14, CRHO23 )
    ENDIF

    ENDDO nqloop
!==========================================================================
    CALL STOP_TIMING("WIJKL+",IO%IU6,XMLTAG='total')

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT  , 1))

    CALLMPI( M_sum_g(WGW%COMM_INTER, TWOe4o, SIZE(TWOe4o)))

    IF (PRESENT(WIJKL)) THEN
       IF (SIZE(WIJKL,1)>=NGLB.AND.SIZE(WIJKL,2)>=NGLB.AND. &
      &     SIZE(WIJKL,3)>=NGLB.AND.SIZE(WIJKL,4)>=NGLB) &
      &   WIJKL(1:NGLB,1:NGLB,1:NGLB,1:NGLB)=TWOe4o(:,:,:,:,1,1)
    ENDIF
!==========================================================================
! deallocation
!==========================================================================
    DEALLOCATE(TWOe4o, CHAM)
    DO N=1,NGLB
       CALL DELWAV(W4(N) ,.TRUE.)
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1,W4)

    IF (ASSOCIATED(H)) CALL DEALLOCATE_ONE_CENTER_H(H)
#ifdef collect_all_wavefunctions
    DO ISP1=1,WDES%ISPIN
    DO K1=1,WDES%NKPTS
       DO N=1,NGLB
          CALL DELWAV(W_all(N,K1,ISP1),.FALSE.) 
       ENDDO
    ENDDO
    ENDDO
    DEALLOCATE(W_all)
#endif

    IF (L2E4W_ALL) THEN
       CALL TWOE_4WANNIER_WRITE( WGW, WDES, NGLB)
       CALL TWOE_4WANNIER_DEALLOCATE
       DEALLOCATE(TWOe4o_ADD)
    ENDIF
       

    RETURN
  END SUBROUTINE CALCULATE_WIJKL_FOURCENTRE




!****************** SUBROUTINE ONEELECTRON_CHARGE      ****************
!
! int_d3r d3r' w_k3,n3(r') w*_k2,n2(r') \sum_k1 w_k4,n4(r) w*_k1,n1(r) W(r,r')
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NGLB] and n2=[NPOS2,NPOS2+NGLB]
!
!**********************************************************************

  SUBROUTINE ONEELECTRON_CHARGE( WHF, H, LATT_CUR, WGW, &
       W1, K1, W4, K4, NB_START, NGLB, NFFT, &
       GCHG14, CRHO14 )

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    IMPLICIT NONE

! passed variables
    TYPE (wavespin) WHF
    TYPE (one_center_handle), POINTER :: H
    TYPE (latt) LATT_CUR
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W4(:)
    INTEGER K1, K4
    INTEGER :: NB_START, NGLB
    REAL(q) :: NFFT
    COMPLEX(q) :: GCHG14(:,:)    ! charge 
    GDEF       :: CRHO14(:,:)    ! one center charge

! local variables
    INTEGER N1, N4, NB1_INTO_TOT, NB4_INTO_TOT
    INTEGER NP
    TYPE (wavedes1)    WGWQ
    INTEGER NQ
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:), CRHO_TMP(:,:) ! augmentation occupancy matrix
    LOGICAL LPHASE

    ! set the descriptor for wavefunctions at the point k1-k4
    ! the product w*_k4,n4(r) w_k1,n1(r) corresponds to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP =WGWQ%NGVECTOR

    IF (NP/= SIZE(GCHG14,1))THEN
       WRITE(*,*) 'internal error ONEELECTRON_CHARGE: NP is not correct', NQ, K1, K4, NP, SIZE(GCHG14,1)
       STOP
    ENDIF

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS))
    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
       ALLOCATE(CRHO_TMP(H%TOTAL_ENTRIES,1))
    ENDIF

!==========================================================================
!   w*_k4,n4(r) w_k1,n1(r)
!==========================================================================
    ! average electrostatic potential for k=k' and n=n'
    ! k1+k4-q might be any reciprocal lattice vector G
    ! in that case the result is shifted by G with respect to the 
    ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
    ! to cure the problem
    CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
    ! set phase factor in fast_aug
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4))
 
    DO N4=1,NGLB
       NB4_INTO_TOT=N4+NB_START-1
       
       DO N1=1,NGLB
          NB1_INTO_TOT=N1+NB_START-1
          
          IF (ASSOCIATED(H).AND. .NOT.ASSOCIATED(WPOTH)) THEN
             CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N1), W4(N4), GWORK(1), &
                  H, CRHO_TMP, CRHOLM,  SIZE(CRHOLM))
             IF (LPHASE) THEN
                CALL APPLY_PHASE_ONE_CENTER(WHF%WDES, H, CRHO_TMP, & 
                     WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ))
             ENDIF
             CRHO14(:,N1+NGLB*(N4-1))= CRHO14(:,N1+NGLB*(N4-1))+CRHO_TMP(:,1)
          ELSE
             CALL FOCK_CHARGE_NOINT( W1(N1), W4(N4), GWORK(1), CRHOLM,  SIZE(CRHOLM))
          ENDIF
          IF (LPHASE) THEN
             CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
          ENDIF
          CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+NGLB*(N4-1)),WGWQ%GRID,.TRUE.)
          NFFT=NFFT+1
       ENDDO
    ENDDO
    
    ! deallocation
    DEALLOCATE(CRHOLM)
    IF (ASSOCIATED(H).AND..NOT.ASSOCIATED(WPOTH)) THEN
       DEALLOCATE(CRHO_TMP)
    ENDIF

  END SUBROUTINE ONEELECTRON_CHARGE

!****************** SUBROUTINE CALCULATE_PIJKL*****************************
! This routine sets up 
!
! POL_ADD(i,j,k,l)=
!        sum_nn' (f_1-f_4)
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1- e_4- i delta    -w+ e_1- e_4+ i delta
!        x U_ni(k1)U_n'k(k4)U*_nj(k1)U*_n'l(k4)
! 
! k4=k1-q
! n.........band of k-point k1
! n'........band of k-point k4
! i,j,k,l...Wannier indices
!**********************************************************************

 SUBROUTINE CALCULATE_PIJKL(W,WDES,LATT_CUR,ISMEAR,SIGMA,COMEGA,K1,K4,ISP1,&
                            NB_START,NB_STOP,POL_ADD,IO)
    USE prec 
    USE base 
    USE wave_high
    USE full_kpoints
    IMPLICIT NONE
    TYPE (wavedes) WDES 
    TYPE (wavespin) W
    TYPE (latt) LATT_CUR
    TYPE (in_struct) IO
    GDEF POL_ADD(:,:,:,:)
    INTEGER K1,K4,ISP1
    INTEGER NB_START, NB_STOP
    INTEGER ISMEAR 
    REAL(q) SIGMA 
    COMPLEX(q) COMEGA
    !local variables
    INTEGER NW, NB, NKP, I,J,K,IP,IPP
    INTEGER N1,N2,N3,N4
    INTEGER M1,M2,M3,M4
    INTEGER NB1,NB2
    COMPLEX(q) :: WEIGHT,CTMP
    GDEF, ALLOCATABLE :: UK1(:,:),UK4(:,:)  

    CALL CHECK_FULL_KPOINTS

    ALLOCATE(UK1(WDES%NB_TOT,WDES%NB_TOT),UK4(WDES%NB_TOT,WDES%NB_TOT))
    UK1=zero
    UK4=zero
    CALL DETERMINE_WANNIER_PROJECTOR(WDES,K1,ISP1,UK1,IO%IU0)
    CALL DETERMINE_WANNIER_PROJECTOR(WDES,K4,ISP1,UK4,IO%IU0) 

   !---------------------------------------------------------------------------
   ! we need to sum over Bloch bands. In chi_base.F 2 runs over unoccupied
   ! 1 runs over occupied bands only. This has to be done in the same way here.
   !---------------------------------------------------------------------------
   DO NB2=FIRST_EMPTY_PIJKL( W, K4, ISP1),W%WDES%NB_TOT  
   DO NB1=1,W%WDES%NB_TOT
    IF (EMPTY_PIJKL_ORBITAL(W%FERTOT(NB1,K1,ISP1))) CYCLE
     !skip if excitation energy goes below threshold
     IF (REAL(W%CELTOT(NB2, K4, ISP1)-W%CELTOT(NB1,K1,ISP1),q)>DEGENERACY_THRESHOLD) THEN
        DO N4=NB_START,NB_STOP  
           DO N3=NB_START,NB_STOP  
              DO N2=NB_START,NB_STOP  
                 DO N1=NB_START,NB_STOP  
                     M1=N1-NB_START+1             
                     M2=N2-NB_START+1             
                     M3=N3-NB_START+1             
                     M4=N4-NB_START+1             
                     WEIGHT=0.0_q
                     CALL SET_PIJKL_WEIGHT(WEIGHT,REAL(W%CELTOT(NB2,K4,ISP1)-&
                        W%CELTOT(NB1,K1,ISP1),q),SHIFT,COMEGA,ISMEAR,SIGMA)
                     WEIGHT=W%WDES%RSPIN*W%WDES%WTKPT(K1)*&
                        (W%FERTOT(NB2,K4,ISP1)-W%FERTOT(NB1, K1, ISP1))*WEIGHT
                     WEIGHT=WEIGHT/W%WDES%NKPTS*UK1(NB1,N1)*UK4(NB2,N2)*&
                        GCONJG(UK1(NB1,N3))*GCONJG(UK4(NB2,N4))
                     POL_ADD(M1,M2,M3,M4)=POL_ADD(M1,M2,M3,M4)+WEIGHT
          
                 ENDDO 
              ENDDO
           ENDDO
        ENDDO
     ENDIF
   ENDDO 
   ENDDO 
    
   DEALLOCATE(UK1,UK4)

 END SUBROUTINE CALCULATE_PIJKL 


!****************** SUBROUTINE SET_PIJKL_WEIGHT************************
! This routine caclulates the Weight
!
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1- e_2- i delta    -w+ e_1- e_2+ i delta
! 
! completely analogously to the chi_base.F routine SET_XI_WEIGHT
! but with additional Wannier rotation matrices U_nj(k1)
!**********************************************************************
  SUBROUTINE SET_PIJKL_WEIGHT(WEIGHT, DECEL, SHIFT, COMEGA, ISMEAR, SIGMA)
    COMPLEX(q) :: WEIGHT
    REAL(q)    :: DECEL, SHIFT
    COMPLEX(q) :: COMEGA
    INTEGER    :: ISMEAR
    REAL(q)    :: SIGMA
   ! local
    REAL(q)    :: SFUN, DFUN, E, EPSDOS
    REAL(q)    :: DEMAX=5, DESTEP
    INTEGER    :: NE, NEDOS



   ! imaginary frequency or ISMEAR < 0 use standard formula
   ! this seems to work best, even for metals (if Hartree-Fock corrections
   ! are applied as well)
    IF ((ISMEAR <0 .AND. ISMEAR/=-6) .OR. (REAL(COMEGA)==0 .AND. AIMAG(COMEGA)/=0)) THEN
!#define resonant
#ifdef timeordered
       WEIGHT=(1/( COMEGA+DECEL+CMPLX(0, SHIFT,q)) &
              +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#elif defined(resonant)
       WEIGHT=1/( COMEGA+DECEL-CMPLX(0, SHIFT,q))
#else
       WEIGHT=(1/( COMEGA+DECEL-CMPLX(0, SHIFT,q)) &
              +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#endif
   ! NOT USED (see above)
   ! frequency lying along imaginary axis COMEGA is small
    ELSE IF (REAL(COMEGA)==0 .AND. AIMAG(COMEGA)/=0) THEN

       ! frequency close to transition level
       ! problem: the weight is still discontinous using this switch
       ! but it is hopefully only in the sub sub meV regime

       ! for the time being this is switched off since
       ! it does not seem to be reliable
       
       IF (ABS(COMEGA+DECEL)<DEMAX*SIGMA*4 .AND. .FALSE.) THEN
          ! this step width yields results that are converged to 1E-7 usually
          DESTEP=MIN(AIMAG(COMEGA)/4,SIGMA/2)
          NEDOS =DEMAX*SIGMA/DESTEP
       
          WEIGHT=0
          !Methfessel-Paxton Dirac delta
          DO NE=-NEDOS,NEDOS-1
             E=DECEL+DESTEP*NE
             CALL DELSTP(ISMEAR,(E-DECEL)/SIGMA,DFUN,SFUN)
             EPSDOS=DFUN*DESTEP/SIGMA
#ifdef timeordered
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)) &
                  +1/(-COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)))
#else
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E-CMPLX(0, MAX(SHIFT,DESTEP),q)) &
                  +1/(-COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)))
#endif
         ENDDO
       ELSE
     ! single point formula (same as above)
#ifdef timeordered
          WEIGHT=(1/( COMEGA+DECEL+CMPLX(0, SHIFT,q)) &
                 +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#else
          WEIGHT=(1/( COMEGA+DECEL-CMPLX(0, SHIFT,q)) &
                 +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#endif
       ENDIF
    ELSE
   ! frequency lying along real axis
   ! this is hardly ever used, because usually the spectral method is applied (LSPECTRAL)
   ! The spectral method has a built in broadening given by the distance between
   ! frequency points.
   !
   ! this routine attempts some "broadening" of the transition energies
   ! by replacing the delta function by a sum of Hermite polynomials
   ! (Methfessel Paxton function)
   !
   ! SHIFT must be set to a sensible value, other wise, routine will
   ! apply a "sensible" backup for the shift
       ! frequency close to transition level
       ! problem: the integral is still discontinous using this switch
       IF (ABS(COMEGA+DECEL)<DEMAX*SIGMA*4) THEN
          DESTEP=SIGMA/4
          NEDOS =DEMAX*SIGMA/DESTEP

          WEIGHT=0
          !Methfessel-Paxton Dirac delta
          DO NE=-NEDOS,NEDOS-1
             E=DECEL+DESTEP*NE
             CALL DELSTP(ISMEAR,(E-DECEL)/SIGMA,DFUN,SFUN)
             EPSDOS=DFUN*DESTEP/SIGMA
#ifdef timeordered
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)) &
                  +1/(-COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)))
#else
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E-CMPLX(0, MAX(SHIFT,DESTEP),q)) &
                  +1/(-COMEGA+E+CMPLX(0, MAX(SHIFT,DESTEP),q)))
#endif
           ENDDO
       ELSE
     ! single point formula
#ifdef timeordered
          WEIGHT=(1/( COMEGA+DECEL+CMPLX(0, SHIFT,q)) &
                 +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#else
          WEIGHT=(1/( COMEGA+DECEL-CMPLX(0, SHIFT,q)) &
                 +1/(-COMEGA+DECEL+CMPLX(0, SHIFT,q)))
#endif
       ENDIF
    ENDIF
  END SUBROUTINE SET_PIJKL_WEIGHT

!***********************************************************************
!  Function EMPTY_PIJKL_ORBITAL returns .TRUE. if the orbital
!  is considered to be empty in the calculation of PIJKL
!  returns true only if the orbital occupancy is smaller than
!  PIJKL_EMPTY_THRESHHOLD
!***********************************************************************

  FUNCTION EMPTY_PIJKL_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL EMPTY_PIJKL_ORBITAL
    REAL(q) :: F
    IF (ABS(F) <PIJKL_EMPTY_THRESHHOLD) THEN
       EMPTY_PIJKL_ORBITAL=.TRUE.
    ELSE
       EMPTY_PIJKL_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION EMPTY_PIJKL_ORBITAL

!***********************************************************************
!  Function FILLED_PIJKL_ORBITAL returns .TRUE. if the orbital
!  is considered to be filled in the PIJKL
!  FILLED_PIJKL_ORBITAL's are skipped when an unoccupied band is required
!***********************************************************************

  FUNCTION FILLED_PIJKL_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL FILLED_PIJKL_ORBITAL
    REAL(q) :: F
    IF (ABS(F-1) <PIJKL_EMPTY_THRESHHOLD) THEN
       FILLED_PIJKL_ORBITAL=.TRUE.
    ELSE
       FILLED_PIJKL_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION FILLED_PIJKL_ORBITAL

!***********************************************************************
!  FIRST_EMPTY_PIJKL returns the first emtpy band in FERTOT
!***********************************************************************
  FUNCTION FIRST_EMPTY_PIJKL( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_PIJKL
    INTEGER K1, ISP
  ! local
    INTEGER NB
    DO NB=1,W%WDES%NB_TOT
       IF ( ABS(W%FERTOT( NB, K1, ISP)-1)>PIJKL_EMPTY_THRESHHOLD ) EXIT
    ENDDO
    
    FIRST_EMPTY_PIJKL=NB
     
  END FUNCTION FIRST_EMPTY_PIJKL

!***********************************************************************
!  LAST_EMPTY_PIJKL returns the last emtpy band
!  this uses the auxilary array which is set to 0 for bands
!  that do not need to be considered 
!***********************************************************************
  FUNCTION LAST_EMPTY_PIJKL( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_EMPTY_PIJKL
    INTEGER K1, ISP
  ! local
    INTEGER NB
    DO NB=W%WDES%NB_TOT,1,-1
       IF ( W%AUX( NB, K1, ISP)/=0 ) EXIT
    ENDDO
    
    LAST_EMPTY_PIJKL=NB
  END FUNCTION LAST_EMPTY_PIJKL


!***********************************************************************
! REMOVE_PIJKL_FROM_WIJKL returns the unscreened potential UIJKL which
! contains only the off-centre screening contributions.
!***********************************************************************
 SUBROUTINE REMOVE_PIJKL_FROM_WIJKL(WDES,BANDS_TOT,P,W,U)
    USE base
    USE wave_high
    USE prec
    IMPLICIT NONE
    TYPE (wavedes) WDES 
    INTEGER I,J,K,L,BANDS_TOT
    GDEF P(:,:,:,:)
    GDEF W(:,:,:,:,:,:)
    GDEF U(:,:,:,:,:,:)
    GDEF,ALLOCATABLE :: UCOMP(:,:)
    GDEF,ALLOCATABLE :: WCOMP(:,:)
    GDEF,ALLOCATABLE :: PCOMP(:,:)
    INTEGER NW_START,NW_STOP
    INTEGER INF,NWI,NWF,NWTOTSUB
    INTEGER I_,J_,L_,K_, COUNTER
    INTEGER ISP1, ISP2
    INTEGER,ALLOCATABLE :: IPIV(:)
    COMPLEX(q),ALLOCATABLE :: WORK(:),WTARGET(:)

 NWI=1
 NWF=NWTOT
 NWTOTSUB = NWTOT

 IF ( NWTOTSUB > BANDS_TOT ) THEN
   WRITE(*,*)' ERROR in REMOVE_PIJKL_FROM_WIJKL: submaxtrix is too large',BANDS_TOT,NWTOTSUB
   NWTOTSUB=BANDS_TOT
 ENDIF

 !define compound indices 
 ALLOCATE(UCOMP(NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB));UCOMP=zero
 ALLOCATE(WCOMP(NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB));WCOMP=zero
 ALLOCATE(PCOMP(NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB));PCOMP=zero

 !allocate IPIV WORK, needed for LAPACK inversion routines 
  ALLOCATE(IPIV(NWTOTSUB*NWTOTSUB))
  ALLOCATE(WORK(NWTOTSUB*NWTOTSUB))

!loop over spin degrees
DO ISP1 =1, WDES%ISPIN
DO ISP2 =ISP1, WDES%ISPIN
 !define compound quantities
 DO L=1,NWTOTSUB
  DO K=1,NWTOTSUB
    DO J=1,NWTOTSUB
     DO I=1,NWTOTSUB
       L_=L    !pick appropriate component in old array 
       K_=K
       J_=J
       I_=I
       IF ( NTARGET_STATES(I)==1 .OR. &
            NTARGET_STATES(J)==1 .OR. &
            NTARGET_STATES(K)==1 .OR. &
            NTARGET_STATES(L)==1 ) THEN

          PCOMP(I+NWTOTSUB*(J-1),K+NWTOTSUB*(L-1))=P(I_,J_,K_,L_)
          WCOMP(I+NWTOTSUB*(J-1),K+NWTOTSUB*(L-1))=W(I_,J_,K_,L_,ISP1,ISP2)
       ENDIF 
    ENDDO 
   ENDDO 
      !define unit matrix
      UCOMP(K+NWTOTSUB*(L-1),K+NWTOTSUB*(L-1))=1.0_q
  ENDDO 
 ENDDO 

!-------------------------------------------------------------- 
! Solve the Unscreening equation W=U+UPW => U=(1+PW)^-1 .W
!--------------------------------------------------------------
      !UCOMP contains now 1+P.W
      CALL ZGEMM('N','N',NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB,&
      &NWTOTSUB*NWTOTSUB, (1.0_Q,0.0_Q),PCOMP,SIZE(PCOMP,1),&
      &WCOMP,SIZE(WCOMP,1),(1.0_Q,0.0_Q),UCOMP,SIZE(UCOMP,1))   

      !LU decomposition of (1 + P.W)
      CALL ZGETRF(NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB, &
            UCOMP,SIZE(UCOMP,1), IPIV, INF )
      IF (INF/=0) THEN
         WRITE(0,*) 'ZGETRF returns',INF
         STOP
      ENDIF

     !invert matrix (1 + W.P)
      CALL ZGETRI(NWTOTSUB*NWTOTSUB,UCOMP, SIZE(UCOMP,1), IPIV, & 
           WORK, SIZE(WORK,1), INF )
      IF (INF/=0) THEN
         WRITE(0,*) 'error : ZGETRI returns',INF
         STOP
      ENDIF

     PCOMP=UCOMP
     
     !U=(1 + P.W)^-1.W 
     CALL ZGEMM('N','N',NWTOTSUB*NWTOTSUB,NWTOTSUB*NWTOTSUB,&
     &NWTOTSUB*NWTOTSUB, (1.0_Q,0.0_Q),PCOMP,SIZE(PCOMP,1),&
     &WCOMP,SIZE(WCOMP,1),(0.0_Q,0.0_Q),UCOMP,SIZE(UCOMP,1))           

!-------------------------------------------------------------- 
! UCOMP contains now the local unscreened effective potential
!--------------------------------------------------------------
  U=zero
  !save unscreened potential into U
    DO L=1,NWTOTSUB
     DO K=1,NWTOTSUB
      DO J=1,NWTOTSUB
       DO I=1,NWTOTSUB
         U(I,J,K,L,ISP1,ISP2)=UCOMP(I+NWTOTSUB*(J-1),K+NWTOTSUB*(L-1))
       ENDDO 
      ENDDO 
     ENDDO 
    ENDDO 
 ENDDO
 ENDDO


 DEALLOCATE(UCOMP,WCOMP,PCOMP)

 ENDSUBROUTINE REMOVE_PIJKL_FROM_WIJKL


!***********************************************************************
! Helper routine, draws some diagrams onto the screen and in the OUTCAR
!***********************************************************************
  SUBROUTINE DRAW_DIAGRAMS(LV,LW,LC,LNL,IU0)
     LOGICAL LV, LW, LC, LNL
     INTEGER :: IU0 
    
   !only master node is going to do something    
   IF ( IU0>=0 ) THEN
   
     !for VIJKL
     IF ( LV ) THEN
        WRITE(IU0,*)"                                "
        IF ( LUSEBLOCH ) THEN
        WRITE(IU0,*)"Calculation of Coulomb interaction in Bloch basis:"
        ELSE
        WRITE(IU0,*)"Calculation of Coulomb interaction in Wannier basis:"
        ENDIF 
        WRITE(IU0,*)"       I                     K  "
        WRITE(IU0,*)"       --->---\   V   /--->---  "
        WRITE(IU0,*)"               |~~~~~|          "
        WRITE(IU0,*)"       ---<---/       \---<---  "
        WRITE(IU0,*)"       J                     L  "
        WRITE(IU0,*)"                                "
     ENDIF

     !for UIJKL NON-LOCAL 
     IF ( LW ) THEN
        WRITE(IU0,*)"                                "
        IF ( LUSEBLOCH ) THEN
        WRITE(IU0,*)"Calculation of screened interaction in Bloch basis:"
        ELSE
        WRITE(IU0,*)"Calculation of screened interaction in Wannier basis:"
        ENDIF 
        IF ( LC ) &
        WRITE(IU0,*)"in the constrained random phase approximation:"
        IF ( LNL) &
        WRITE(IU0,*)"in the non-local random phase approximation:"
        WRITE(IU0,*)"       I                     K  "
        WRITE(IU0,*)"       --->---\   U   /--->---  "
        IF ( .NOT.LNL ) &
        WRITE(IU0,*)"               |=====|          "
        IF ( LNL) &
        WRITE(IU0,*)"               |-----|          "
        WRITE(IU0,*)"       ---<---/       \---<---  "
        WRITE(IU0,*)"       J                     L  "
        WRITE(IU0,*)"                                "
     ENDIF
 

   ENDIF
  

 
   RETURN
  ENDSUBROUTINE DRAW_DIAGRAMS


!***********************************************************************
! Helper routine, reads some INCAR flags, needed to set NWLOW and NWHIGH
! which determine, which matrix elements have to be calculated
!***********************************************************************
  SUBROUTINE SET_TARGET_STATES(IO)
     USE base
     TYPE(in_struct) IO
     INTEGER IDUM,IERR
     REAL(q) RDUM
     COMPLEX(q) CDUM
     LOGICAL LDUM
     CHARACTER (1) :: CHARAC
  
     INTEGER COUNTER,I,N, NSTATES
     INTEGER, ALLOCATABLE :: WTARGET(:)
    
     ! Do we need to calculate all elements?
     OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
      !Calculate unscreened potential, true by default
      LVPOT=.TRUE.
      CALL RDATAB(.FALSE.,INCAR,IO%IU5,'LVPOT','=','#',';','L', &
      &            IDUM,RDUM,CDUM,LVPOT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LVPOT'' from file INCAR.'
         LVPOT=.TRUE.
         ENDIF

      !Calculate screened potential, true by default
      LWPOT=.TRUE.
      CALL RDATAB(.FALSE.,INCAR,IO%IU5,'LWPOT','=','#',';','L', &
      &            IDUM,RDUM,CDUM,LWPOT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LWPOT'' from file INCAR.'
         LWPOT=.TRUE.
         ENDIF

      NWLOW=-1
      CALL RDATAB(.FALSE.,INCAR,IO%IU5,'NWLOW','=','#',';','I', &
          &            NWLOW,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''NWLOW'' from file INCAR.'
         NWLOW=-1
      ENDIF

      NWHIGH=-1
      CALL RDATAB(.FALSE.,INCAR,IO%IU5,'NWHIGH','=','#',';','I', &
          &            NWHIGH,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''NWHIGH'' from file INCAR.'
         NWHIGH=-1
      ENDIF

      VCUTOFF=-30
      CALL RDATAB(.FALSE.,INCAR,IO%IU5,'VCUTOFF','=','#',';','F', &
           &            IDUM,VCUTOFF,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''VCUTOFF'' from file INCAR.'
      ENDIF

     CLOSE(IO%IU5)

     !----------------------------------------------------------------------
     ! Important: 
     ! if we are in the CRPA or NLRPA mode,we need to switch on HFCALC and
     ! set AEXX = 1, otherwise the POTFAK for the bare Coulomb interaction 
     ! will not be calculated and the result for VIJKL = 0
     IF ( LCRPA .OR. LNLRPA .OR. L2E4W) THEN
       LHFCALC = .TRUE.
       AEXX  = 1.0
     ENDIF

     IF (DMFT_num_wann<SIZE(NTARGET_STATES)) NTARGET_STATES(DMFT_num_wann+1:)=0
  
     !set default values for NWLOW and NWHIGH
     NSTATES=SUM(NTARGET_STATES(1:DMFT_num_wann))
     IF (NWLOW < 0 .AND. NWHIGH < 0 )THEN
        !full RPA (all states)
        IF( NSTATES == 0 )THEN
           NWLOW = 1
           NWHIGH = DMFT_num_wann
        !crpa (all states)
        ELSEIF( NSTATES == DMFT_num_wann)THEN
           NWLOW = 1
           NWHIGH = DMFT_num_wann
        !some states
        ELSEIF(NSTATES > 0 .AND. NSTATES < DMFT_num_wann )THEN
           ALLOCATE(WTARGET(DMFT_num_wann))
           WTARGET=0
           COUNTER=0
           DO I = 1, DMFT_num_wann
              IF ( NTARGET_STATES (I) == 1 ) THEN
                 COUNTER = COUNTER + 1
                 WTARGET(COUNTER) = I
              ENDIF
           ENDDO
           NWLOW  = WTARGET(1)
           NWHIGH = WTARGET(COUNTER)

           DEALLOCATE(WTARGET)
        ENDIF 

     ELSEIF( NWLOW < 0 .AND. NWHIGH > 0 )THEN
        NWLOW = 1
        NWHIGH = MIN(DMFT_num_wann, NWHIGH)
      
     !wrong set set default
     ELSEIF( NWHIGH > NWLOW .AND. NWLOW >0 )THEN
        NWHIGH = MIN(DMFT_num_wann, NWHIGH)
        IF (SUM(NTARGET_STATES(1:DMFT_num_wann)) == 0 ) THEN
           NTARGET_STATES(1:NWLOW-1) = 0 
           NTARGET_STATES(NWHIGH:DMFT_num_wann) = 0
           NTARGET_STATES(NWLOW:NWHIGH) = 1
        ENDIF
     ELSE
        IF( IO%IU0>=0) WRITE( IO%IU0,'(A/,3I4)')'Warning in DMFT_MATRIX_ELEMENTS,'//&
           ' choice of target bands incompatible',NWLOW,NWHIGH,DMFT_num_wann
        NWLOW = 1 
        NWHIGH = DMFT_num_wann
     ENDIF 

     !set NWLOW and NWHIGH for BLOCH basis
     IF ( LUSEBLOCH .AND. ( NWLOW < 0 .AND. NWHIGH <0 ) ) THEN
        NWLOW = DMFT_num_wann
        NWHIGH = 1 
        DO I = 1, DMFT_num_wann
           IF ( NTARGET_STATES(I)==1)THEN
              NWLOW = MIN( NWLOW,I)
              NWHIGH = MAX( NWHIGH,I)
           ENDIF 
        ENDDO   
     ENDIF 
     
     !number of Wannier bands considered
     NWTOT = NWHIGH - NWLOW + 1
     
     IF ( IO%IU0>=0 ) THEN
        WRITE(IO%IU0,*)" " 
        IF ( LWPOT ) WRITE(IO%IU0,*)"Will calculate bare V and effective interaction U"
        IF ( .NOT.LWPOT ) WRITE(IO%IU0,*)"Will calculate only bare interaction V"
        IF ( LUSEBLOCH ) THEN
           WRITE(IO%IU0,'(" for the Bloch bands: ",I2," - ",I2)') NWLOW, NWHIGH
        ELSE
           WRITE(IO%IU0,'(" for the Wannier states: ",I2," - ",I2)') NWLOW, NWHIGH
        ENDIF
        WRITE(IO%IU0,*)" " 
     ENDIF
     RETURN
  END SUBROUTINE SET_TARGET_STATES

!***********************************************************************
! Writes The matrix elements into the files VIJKL, WIJKL,PIJKL, UIJKL
! and some selectes values into the OUTCAR file
!***********************************************************************
  SUBROUTINE WRITE_DMFT_MATRIX_ELEMENTS(IO,ISPIN,LSCREENED,WIJKL,UIJKL,PIJKL)
    USE base
    IMPLICIT NONE
    TYPE(in_struct) IO
    GDEF ::  WIJKL(:,:,:,:,:,:)  !either bare or screened interaction
    LOGICAL :: LSCREENED     !write either screened or bare interaction
    INTEGER :: ISPIN         !number of spin degrees of freedom
    GDEF, OPTIONAL ::  UIJKL(:,:,:,:,:,:)
    GDEF, OPTIONAL ::  PIJKL(:,:,:,:)
 
    INTEGER N1,N2,N3,N4
    INTEGER ISP1,ISP2
    INTEGER :: COUNTER,J,I, NWI,NWF,NWTOTSUB
    INTEGER,ALLOCATABLE :: WTARGET(:)
    CHARACTER(5) :: EXTENSION
    COMPLEX(q) :: CTMP1,CTMP2,CTMP3
    LOGICAL :: LFOCK
    
    NWI=1
    NWF=NWTOT
    NWTOTSUB=NWTOT  

    !do we have to write VIJKL obtained with FOCKAE=4 settings?
    IF ( PRESENT( UIJKL ) .AND. .NOT.(PRESENT(PIJKL)) ) THEN
       LFOCK = .TRUE.
    ELSE
       LFOCK = .FALSE.
    ENDIF     
       
    !first of all determine optimal subspace of Wannier subspace
    !only those elements are relevant   
    DO ISP1 = 1, ISPIN 
       ! there are maximally three linear independent spin polarizations 
       DO ISP2 = ISP1, ISPIN
          !define extension
          IF ( ISP1 ==1 .AND. ISP2 == 1 ) THEN
             EXTENSION='up.up'
          ELSEIF (ISP1==1 .AND. ISP2 == 2 ) THEN
             EXTENSION='up.dn'
          ELSEIF (ISP1==2 .AND. ISP2 == 2 ) THEN
             EXTENSION='dn.dn'
          ENDIF
 
          !---------------------------------------------------------------
          !write the bare coulomb matrix, if we don't have to write 
          !the screenend interaction and the polarizability
          IF (.NOT.LSCREENED) THEN
          !---------------------------------------------------------------
             IF (IO%IU6>=0) THEN
                !open a VIJKL file
                IF ( ISPIN == 1 ) THEN
#ifdef debug
                   OPEN(UNIT=100,FILE='RE.VIJ',STATUS='REPLACE')
                   OPEN(UNIT=101,FILE='IM.VIJ',STATUS='REPLACE')
#endif
                   OPEN(UNIT=99,FILE='VIJKL',STATUS='REPLACE')
                ELSE
#ifdef debug
                   OPEN(UNIT=100,FILE='RE.VIJ.'//EXTENSION,STATUS='REPLACE')
                   OPEN(UNIT=101,FILE='IM.VIJ.'//EXTENSION,STATUS='REPLACE')
#endif
                   OPEN(UNIT=99,FILE='VIJKL.'//EXTENSION,STATUS='REPLACE')
                ENDIF
                !print all matrix elements into VIJKL
                DO N4=1,NWTOT
                   DO N3=1,NWTOT
                      DO N2=1,NWTOT
                         DO N1=1,NWTOT
                            WRITE(99,'(4i4,2F20.10)') N1,N2,N3,N4,WIJKL(N1,N2,N3,N4,ISP1,ISP2 )
                         ENDDO 
                      ENDDO
                    ENDDO
                ENDDO
      
                CLOSE(99)

                WRITE(IO%IU6,*)
                WRITE(IO%IU6,'(" spin components:",2I3)') ISP1,ISP2
                ! Write WIJKL(I,I,J,J) to OUTCAR
                IF ( LUSEBLOCH ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj:'
                ELSE IF ( LUSELCAO ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj between LCAOs:'
                ELSE
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj between MLWFs:'
                ENDIF
                WRITE(IO%IU6,998,ADVANCE="No")
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,999,ADVANCE="No") N1
                ENDDO
                WRITE(IO%IU6,*)
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,997,ADVANCE="No") N1 
                   DO N2=1,NWTOTSUB
                      WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N1,N2,N2,ISP1,ISP2))
                   ENDDO
                   WRITE(IO%IU6,*)
#ifdef debug
                   WRITE(100,1010) (REAL(WIJKL(N1,N1,N3,N3,ISP1,ISP2),q),N3=1,NWTOTSUB)
                   WRITE(101,1010) (IMAG(WIJKL(N1,N1,N3,N3,ISP1,ISP2)),N3=1,NWTOTSUB)
#endif
                ENDDO 
                ! Write WIJKL(I,J,J,I) to OUTCAR
                IF ( LUSEBLOCH ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji:'
                ELSE IF ( LUSELCAO ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji between LCAOs:'
                ELSE
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji between MLWFs:'
                ENDIF
                WRITE(IO%IU6,998,ADVANCE="No")
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,999,ADVANCE="No") N1
                ENDDO
                WRITE(IO%IU6,*)
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,997,ADVANCE="No") N1 
                   DO N2=1,NWTOTSUB
                      WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N2,N2,N1,ISP1,ISP2))
                   ENDDO
                   WRITE(IO%IU6,*)
                ENDDO 
                ! Write WIJKL(I,J,I,J) to OUTCAR
                IF ( LUSEBLOCH ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij:'
                ELSE IF ( LUSELCAO ) THEN
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij between LCAOs:'
                ELSE
                   WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij between MLWFs:'
                ENDIF
                WRITE(IO%IU6,998,ADVANCE="No")
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,999,ADVANCE="No") N1
                ENDDO
                WRITE(IO%IU6,*)
                DO N1=1,NWTOTSUB
                   WRITE(IO%IU6,997,ADVANCE="No") N1 
                   DO N2=1,NWTOTSUB
                      WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N2,N1,N2,ISP1,ISP2))
                   ENDDO
                   WRITE(IO%IU6,*)
                ENDDO 
                CLOSE(100)
                CLOSE(101)

                ! Write Hubbard Kanamori parameter
                WRITE(IO%IU6,'(/A)') ' averaged bare interaction'
                CTMP1 = 0
                CTMP2 = 0
                CTMP3 = 0
                DO N1 = 1, NWTOTSUB
                   CTMP1 = CTMP1 + WIJKL(N1,N1,N1,N1,ISP1,ISP2)
                   DO N2 = 1, NWTOTSUB
                        IF ( N1 /= N2 ) THEN
                           CTMP2 = CTMP2 + WIJKL(N1,N1,N2,N2,ISP1,ISP2)
                           CTMP3 = CTMP3 + WIJKL(N1,N2,N1,N2,ISP1,ISP2)
                        ENDIF
                   ENDDO
                ENDDO
                CTMP1 = CTMP1/REAL(NWTOTSUB,q)
                IF ( NWTOTSUB > 1 ) THEN
                   CTMP2 = CTMP2/REAL(NWTOTSUB*(NWTOTSUB-1),q)
                   CTMP3 = CTMP3/REAL(NWTOTSUB*(NWTOTSUB-1),q) 
                   WRITE(IO%IU6,'(" bare Hubbard U =",2F10.4)')CTMP1
                   WRITE(IO%IU6,'(" bare Hubbard u =",2F10.4)')CTMP2
                   WRITE(IO%IU6,'(" bare Hubbard J =",2F10.4)')CTMP3
                ELSE
                   WRITE(IO%IU6,'(" bare Hubbard U =",2F10.4)')CTMP1
                ENDIF 
                WRITE(IO%IU6,*)

                !print low cutoff V
                IF ( LFOCK ) THEN
                   WRITE(IO%IU6,*)
                   WRITE(IO%IU6,'(" low cutoff result for V_ijkl ")')
                   WRITE(IO%IU6,'(" spin components:",2I3)') ISP1,ISP2
                   ! Write WIJKL(I,I,J,J) to OUTCAR
                   IF ( LUSEBLOCH ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj:'
                   ELSE IF ( LUSELCAO ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj between LCAOs:'
                   ELSE
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_iijj between MLWFs:'
                   ENDIF
                   WRITE(IO%IU6,998,ADVANCE="No")
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,999,ADVANCE="No") N1
                   ENDDO
                   WRITE(IO%IU6,*)
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,997,ADVANCE="No") N1 
                      DO N2=1,NWTOTSUB
                         WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N1,N2,N2,ISP1,ISP2))
                      ENDDO
                      WRITE(IO%IU6,*)
                   ENDDO 
                   ! Write WIJKL(I,J,J,I) to OUTCAR
                   IF ( LUSEBLOCH ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji:'
                   ELSE IF ( LUSELCAO ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji between LCAOs:'
                   ELSE
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijji between MLWFs:'
                   ENDIF
                   WRITE(IO%IU6,998,ADVANCE="No")
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,999,ADVANCE="No") N1
                   ENDDO
                   WRITE(IO%IU6,*)
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,997,ADVANCE="No") N1 
                      DO N2=1,NWTOTSUB
                         WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N2,N2,N1,ISP1,ISP2))
                      ENDDO
                      WRITE(IO%IU6,*)
                   ENDDO 
                   ! Write WIJKL(I,J,I,J) to OUTCAR
                   IF ( LUSEBLOCH ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij:'
                   ELSE IF ( LUSELCAO ) THEN
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij between LCAOs:'
                   ELSE
                      WRITE(IO%IU6,'(/A)') ' bare Coulomb repulsion V_ijij between MLWFs:'
                   ENDIF
                   WRITE(IO%IU6,998,ADVANCE="No")
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,999,ADVANCE="No") N1
                   ENDDO
                   WRITE(IO%IU6,*)
                   DO N1=1,NWTOTSUB
                      WRITE(IO%IU6,997,ADVANCE="No") N1 
                      DO N2=1,NWTOTSUB
                         WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N2,N1,N2,ISP1,ISP2))
                      ENDDO
                      WRITE(IO%IU6,*)
                   ENDDO 
                   ! Write Hubbard Kanamori parameter
                   WRITE(IO%IU6,'(/A)') ' cutoff correction to averaged interaction'
                   CTMP1 = 0
                   CTMP2 = 0
                   CTMP3 = 0
                   DO N1 = 1, NWTOTSUB
                      CTMP1 = CTMP1 + WIJKL(N1,N1,N1,N1,ISP1,ISP2)-UIJKL(N1,N1,N1,N1,ISP1,ISP2)
                      DO N2 = 1, NWTOTSUB
                           IF ( N1 /= N2 ) THEN
                              CTMP2 = CTMP2 + WIJKL(N1,N1,N2,N2,ISP1,ISP2) - &
                                              UIJKL(N1,N1,N2,N2,ISP1,ISP2)
                              CTMP3 = CTMP3 + WIJKL(N1,N2,N1,N2,ISP1,ISP2) - &
                                              UIJKL(N1,N2,N1,N2,ISP1,ISP2)
                           ENDIF
                      ENDDO
                   ENDDO
                   CTMP1 = CTMP1/REAL(NWTOTSUB,q)
                   IF ( NWTOTSUB > 1 ) THEN
                      CTMP2 = CTMP2/REAL(NWTOTSUB*(NWTOTSUB-1),q)
                      CTMP3 = CTMP3/REAL(NWTOTSUB*(NWTOTSUB-1),q) 
                      WRITE(IO%IU6,'(" correction of U =",2F10.4)')CTMP1
                      WRITE(IO%IU6,'(" correction of u =",2F10.4)')CTMP2
                      !WRITE(IO%IU6,'(" correction of U",A z",2F10.4)')CTMP2
                      WRITE(IO%IU6,'(" correction of J =",2F10.4)')CTMP3
                   ELSE
                      WRITE(IO%IU6,'(" correction of U =",2F10.4)')CTMP1
                   ENDIF 
                   WRITE(IO%IU6,*)
                ENDIF 
         
             ENDIF
         
          !---------------------------------------------------------------
          ELSE  
          ! write the screened interaction 
          !---------------------------------------------------------------
              IF (IO%IU6>=0) THEN
                 !if cRPA we need to open UIJKL instead of WIJKL
                 IF ( ISPIN == 1 ) THEN
                    IF ( LCRPA ) THEN
                       OPEN(UNIT=99,FILE='UIJKL',STATUS='REPLACE')
                    ELSEIF ( LNLRPA .OR. L2E4W ) THEN 
                       OPEN(UNIT=99,FILE='WIJKL',STATUS='REPLACE')
                    ENDIF
                 ELSE
                    IF ( LCRPA ) THEN
                       OPEN(UNIT=99,FILE='UIJKL.'//EXTENSION,STATUS='REPLACE')
                    ELSEIF ( LNLRPA .OR. L2E4W ) THEN 
                       OPEN(UNIT=99,FILE='WIJKL.'//EXTENSION,STATUS='REPLACE')
                    ENDIF
                 ENDIF
                 DO N4=1,NWTOT
                  DO N3=1,NWTOT
                    DO N2=1,NWTOT
                     DO N1=1,NWTOT
                        WRITE(99,'(4i4,2F20.10)') N1,N2,N3,N4,WIJKL(N1,N2,N3,N4,ISP1,ISP2 )
                     ENDDO 
                    ENDDO
                   ENDDO
                 ENDDO
                 CLOSE(99)
  
                 WRITE(IO%IU6,130)
                 WRITE(IO%IU6,*)
                 WRITE(IO%IU6,'(" spin components:",2I3)') ISP1,ISP2
                 ! Write WIJKL(I,I,J,J) or UIJKL to OUTCAR
                 IF ( LCRPA ) THEN
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_iijj:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_iijj between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_iijj between MLWFs:'
                    ENDIF
                 ELSE
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_iijj:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_iijj between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_iijj between MLWFs:'
                    ENDIF
                 ENDIF

                 WRITE(IO%IU6,998,ADVANCE="No")
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,999,ADVANCE="No") N1
                 ENDDO
                 WRITE(IO%IU6,*)
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,997,ADVANCE="No") N1 
                    DO N2=1,NWTOTSUB
                       WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N1,N2,N2,ISP1,ISP2))
                    ENDDO
                    WRITE(IO%IU6,*)
                 ENDDO 
                 ! Write WIJKL(I,J,J,I) to OUTCAR
                 IF ( LCRPA ) THEN
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijji:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijji between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijji between MLWFs:'
                    ENDIF
                 ELSE
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijji:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijji between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijji between MLWFs:'
                    ENDIF
                 ENDIF
                 WRITE(IO%IU6,998,ADVANCE="No")
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,999,ADVANCE="No") N1
                 ENDDO
                 WRITE(IO%IU6,*)
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,997,ADVANCE="No") N1 
                    DO N2=1,NWTOTSUB
                       WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N2,N2,N1,ISP1,ISP2))
                    ENDDO
                    WRITE(IO%IU6,*)
                 ENDDO 
                 ! Write WIJKL(I,J,I,J) to OUTCAR
                 IF ( LCRPA ) THEN
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijij:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijij between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion U_ijij between MLWFs:'
                    ENDIF
                 ELSE
                    IF ( LUSEBLOCH ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijij:'
                    ELSE IF ( LUSELCAO ) THEN
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijij between LCAOs:'
                    ELSE
                       WRITE(IO%IU6,'(/A)') ' screened Coulomb repulsion W_ijij between MLWFs:'
                    ENDIF
                 ENDIF
                 WRITE(IO%IU6,998,ADVANCE="No")
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,999,ADVANCE="No") N1
                 ENDDO
                 WRITE(IO%IU6,*)
                 DO N1=1,NWTOTSUB
                    WRITE(IO%IU6,997,ADVANCE="No") N1 
                    DO N2=1,NWTOTSUB
                       WRITE(IO%IU6,1000,ADVANCE="No") REAL(WIJKL(N1,N2,N1,N2,ISP1,ISP2))
                    ENDDO
                    WRITE(IO%IU6,*)
                 ENDDO 
           
                 ! Write Hubbard Kanamori parameter
                 WRITE(IO%IU6,'(/A)') ' averaged interaction parameter'
                 CTMP1 = 0
                 CTMP2 = 0
                 CTMP3 = 0
                 DO N1 = 1, NWTOTSUB
                    CTMP1 = CTMP1 + WIJKL(N1,N1,N1,N1,ISP1,ISP2)
                    DO N2 = 1, NWTOTSUB
                       IF ( N1 /= N2 ) THEN
                          CTMP2 = CTMP2 + WIJKL(N1,N1,N2,N2,ISP1,ISP2)
                          CTMP3 = CTMP3 + WIJKL(N1,N2,N1,N2,ISP1,ISP2)
                       ENDIF
                    ENDDO
                 ENDDO
                 CTMP1 = CTMP1/REAL(NWTOTSUB,q)
                 IF ( NWTOTSUB > 1 ) THEN
                    CTMP2 = CTMP2/REAL(NWTOTSUB*(NWTOTSUB-1),q)
                    CTMP3 = CTMP3/REAL(NWTOTSUB*(NWTOTSUB-1),q) 
                    WRITE(IO%IU6,'(" screened Hubbard U =",2F10.4)')CTMP1
                    WRITE(IO%IU6,'(" screened Hubbard u =",2F10.4)')CTMP2
                    WRITE(IO%IU6,'(" screened Hubbard J =",2F10.4)')CTMP3
                 ELSE
                    WRITE(IO%IU6,'(" screened Hubbard U =",2F10.4)')CTMP1
                 ENDIF 
                 WRITE(IO%IU6,*)

                 IF ( PRESENT(PIJKL) .AND. PRESENT(UIJKL) ) THEN
                    IF ( ISP1==1 .AND. ISP2 == 1) OPEN(UNIT=98,FILE='PIJKL',STATUS='REPLACE')
                    IF ( ISPIN == 1 ) THEN 
                       OPEN(UNIT=97,FILE='UIJKL',STATUS='REPLACE')
                    ELSE
                       OPEN(UNIT=97,FILE='UIJKL.'//EXTENSION,STATUS='REPLACE')
                    ENDIF
                     DO N4=1,NWTOTSUB 
                      DO N3=1,NWTOTSUB
                        DO N2=1,NWTOTSUB
                         DO N1=1,NWTOTSUB
                            IF( ISP1 ==1.AND.ISP2==1) WRITE(98,'(4i4,2F20.10)') N1,N2,N3,N4,PIJKL(N1,N2,N3,N4)
                            WRITE(97,'(4i4,2F20.10)') N1,N2,N3,N4,UIJKL(N1,N2,N3,N4,ISP1,ISP2)
                         ENDDO 
                        ENDDO
                       ENDDO
                      ENDDO
      
                    CLOSE(97)
                    IF ( ISP1 ==1 .AND. ISP2 == 1 ) THEN 
                       CLOSE(98)     
      
                       WRITE(IO%IU6,130)
                       ! Write PIJKL(I,I,J,J) to OUTCAR
                       IF ( LUSEBLOCH ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_iijj:'
                       ELSE IF ( LUSELCAO ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_iijj in LCAO Basis:'
                       ELSE
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_iijj in MLWF Basis:'
                       ENDIF
                       WRITE(IO%IU6,998,ADVANCE="No")
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,999,ADVANCE="No") N1
                       ENDDO
                       WRITE(IO%IU6,*)
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,997,ADVANCE="No") N1 
                          DO N2=1,NWTOTSUB
                             WRITE(IO%IU6,1000,ADVANCE="No") REAL(PIJKL(N1,N1,N2,N2))
                          ENDDO
                          WRITE(IO%IU6,*)
                       ENDDO 
                       ! Write PIJKL(I,J,I,J) to OUTCAR
                       IF ( LUSEBLOCH ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijji:'
                       ELSE IF ( LUSELCAO ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijji in LCAO Basis:'
                       ELSE
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijji in MLWF Basis:'
                       ENDIF
                       WRITE(IO%IU6,998,ADVANCE="No")
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,999,ADVANCE="No") N1
                       ENDDO
                       WRITE(IO%IU6,*)
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,997,ADVANCE="No") N1 
                          DO N2=1,NWTOTSUB
                             WRITE(IO%IU6,1000,ADVANCE="No") REAL(PIJKL(N1,N2,N2,N1))
                          ENDDO
                          WRITE(IO%IU6,*)
                       ENDDO 
                       ! Write PIJKL(I,J,I,J) to OUTCAR
                       IF ( LUSEBLOCH ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijij:'
                       ELSE IF ( LUSELCAO ) THEN
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijij in LCAO Basis:'
                       ELSE
                          WRITE(IO%IU6,'(/A)') ' particle hole propagator P_ijij in MLWF Basis:'
                       ENDIF
                       WRITE(IO%IU6,998,ADVANCE="No")
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,999,ADVANCE="No") N1
                       ENDDO
                       WRITE(IO%IU6,*)
                       DO N1=1,NWTOTSUB
                          WRITE(IO%IU6,997,ADVANCE="No") N1 
                          DO N2=1,NWTOTSUB
                             WRITE(IO%IU6,1000,ADVANCE="No") REAL(PIJKL(N1,N2,N1,N2))
                          ENDDO
                          WRITE(IO%IU6,*)
                       ENDDO 
                    ENDIF
       
                    WRITE(IO%IU6,130)
                    WRITE(IO%IU6,*)
                    WRITE(IO%IU6,'(" spin components:",2I3)') ISP1,ISP2
                    ! Write UIJKL(I,I,J,J) to OUTCAR
                    WRITE(IO%IU6,'(/A)') ' non-local screened Coulomb repulsion U_iijj between target states:'
                    WRITE(IO%IU6,998,ADVANCE="No")
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,999,ADVANCE="No") N1
                    ENDDO
                    WRITE(IO%IU6,*)
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,997,ADVANCE="No") N1 
                       DO N2=1,NWTOTSUB
                          WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N1,N2,N2,ISP1,ISP2))
                       ENDDO
                       WRITE(IO%IU6,*)
                    ENDDO 
                    ! Write UIJKL(I,I,J,J) to OUTCAR
                    WRITE(IO%IU6,'(/A)') ' non-local screened Coulomb repulsion U_ijji between target states:'
                    WRITE(IO%IU6,998,ADVANCE="No")
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,999,ADVANCE="No") N1
                    ENDDO
                    WRITE(IO%IU6,*)
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,997,ADVANCE="No") N1 
                       DO N2=1,NWTOTSUB
                          WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N2,N2,N1,ISP1,ISP2))
                       ENDDO
                       WRITE(IO%IU6,*)
                    ENDDO 
                    ! Write UIJKL(I,I,J,J) to OUTCAR
                    WRITE(IO%IU6,'(/A)') ' non-local screened Coulomb repulsion U_ijij between target states:'
                    WRITE(IO%IU6,998,ADVANCE="No")
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,999,ADVANCE="No") N1
                    ENDDO
                    WRITE(IO%IU6,*)
                    DO N1=1,NWTOTSUB
                       WRITE(IO%IU6,997,ADVANCE="No") N1 
                       DO N2=1,NWTOTSUB
                          WRITE(IO%IU6,1000,ADVANCE="No") REAL(UIJKL(N1,N2,N1,N2,ISP1,ISP2))
                       ENDDO
                       WRITE(IO%IU6,*)
                    ENDDO 

                    ! Write Hubbard Kanamori parameter
                    WRITE(IO%IU6,'(/A)') ' averaged non-local interaction '
                    CTMP1 = 0
                    CTMP2 = 0
                    CTMP3 = 0
                    DO N1 = 1, NWTOTSUB
                       CTMP1 = CTMP1 + UIJKL(N1,N1,N1,N1,ISP1,ISP2)
                       DO N2 = 1, NWTOTSUB
                         IF ( N1 /= N2 ) THEN
                            CTMP2 = CTMP2 + UIJKL(N1,N1,N2,N2,ISP1,ISP2)
                            CTMP3 = CTMP3 + UIJKL(N1,N2,N1,N2,ISP1,ISP2)
                         ENDIF
                       ENDDO
                    ENDDO
                    CTMP1 = CTMP1/REAL(NWTOTSUB,q)
                    IF ( NWTOTSUB > 1 ) THEN
                       CTMP2 = CTMP2/REAL(NWTOTSUB*(NWTOTSUB-1),q)
                       CTMP3 = CTMP3/REAL(NWTOTSUB*(NWTOTSUB-1),q) 
                       WRITE(IO%IU6,'(" non-local screened Hubbard U =",2F10.4)')CTMP1
                       WRITE(IO%IU6,'(" non-local screened Hubbard u =",2F10.4)')CTMP2
                       WRITE(IO%IU6,'(" non-local screened Hubbard J =",2F10.4)')CTMP3
                    ELSE
                       WRITE(IO%IU6,'(" non-local screened Hubbard U =",2F10.4)')CTMP1
                    ENDIF 
                    WRITE(IO%IU6,*)
                 ENDIF
              ENDIF
             
          !---------------------------------------------------------------
           ENDIF
          !---------------------------------------------------------------
       ENDDO 
     ENDDO

130 FORMAT (5X, //, &
         &'----------------------------------------------------', &
         &'----------------------------------------------------'//)
!column distance for interaction matrices
 997 FORMAT(I5)
 998 FORMAT(3X)
 999 FORMAT(3X,I4,3X)
!format for entries
1000 FORMAT(F10.4)
1010 FORMAT(20F20.10)
  ENDSUBROUTINE WRITE_DMFT_MATRIX_ELEMENTS

!******************** SUBROUTINE PROJECT_ORBITALS_NOSYMM ***************
! 
! projects a wavefunction onto the chosen localized basis 
!
!***********************************************************************
   SUBROUTINE PROJECT_ORBITALS_NOSYMM(WDES,W, IU0)
      USE dfast
      USE wave_high
      USE full_kpoints
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      INTEGER IU0
      ! local variables
      TYPE(wavedes1) WDES1
      TYPE(wavefuna) WA
      INTEGER I,J,K,IP,IPP
      INTEGER ISP,NK,NKP,NW,NB,NBEXCL,NBwin
      COMPLEX(q) CTMP
      GDEF, ALLOCATABLE :: U(:,:)
 
      CALL CHECK_FULL_KPOINTS

      ALLOCATE(U(WDES%NB_TOT,WDES%NB_TOT))

      spin: DO ISP=1,WDES%ISPIN
         kpoints: DO NK=1,WDES%NKPTS
            
            CALL DETERMINE_WANNIER_PROJECTOR( WDES, NK, ISP, U, IU0 ) 
             
            CALL SETWDES(WDES,WDES1,NK)
            WA=ELEMENTS(W,WDES1,ISP)

            ! redistribute over plane wave coefficients
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF

            ! build linear combinations of the wave functions in
            ! accordance with the effective rotation matrix U
            CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,U(1,1), &
           &     WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), &
           &     WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,WDES%NB_TOT, &
           &     WA%CW_RED,WA%CPROJ_RED)

            ! redistribute back over bands
            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, WDES%NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDIF
         ENDDO kpoints
      ENDDO spin

      DEALLOCATE(U)

      RETURN
   END SUBROUTINE PROJECT_ORBITALS_NOSYMM

#endif

END MODULE dmft
