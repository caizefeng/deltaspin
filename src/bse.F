#include "symbol.inc"

! use time propagation code of Friedhelm Bechstedts group (added by C. Roedl and jF)
!#define use_bse_te
!furth Currently, bse_te.F code cannot be compiled with -DwNGZhalf because this
!      part is fully complex, so we need to switch off use_bse_te in this case
#ifdef gammareal
#undef use_bse_te
#endif
#ifndef MPI
#undef use_bse_te
#endif
   
MODULE bse
  USE local_field
  USE scala
  USE wave_high
#ifdef use_bse_te
  USE bse_te
#endif
  IMPLICIT NONE


!**********************************************************************
!
! this subroutine solves the BSE equation 
! presently in the Tamm Dancoff approximation
!
! it is based on the local_field subroutine and relies on the 
! routines implemented therein
! the required integrals are
! comment to publication Sander:
!  k1,n1=  i
!  k2,n2=  j
!  k3,n3=  a
!  k4,n4=  b
!
! The  direct contribution (A = <e,h |W| e',h'>)
! (two electrons at the same spatial coordinate, and two holes)
!
!     int_d3r d3r' c*_k1+q,n3(r') c'_k2+q,n4(r') v'*_k2,n2(r)   v_k1,n1(r)  W(r',r)
!
! and a second  exchange like contribution (B= <e,h | h',e'>)
! (lacking in the Tamm Dancoff approximation)
!    int_d3r d3r' c_k1-q,n3(r')  v'*_k2,n2(r') c'_k2+q,n4(r)  v*_k1,n1(r)  W(r',r)
!
! and a term stemming from the variation of the Hartree potential
!
!    int_d3r d3r' v'*_k2,n2(r') c'_k2+q,n4(r') c*_k1+q,n3(r) v_k1,n1(r) v(r',r)
!
!
! Note that the notation A and B refers to the quantum chemists notation,
! where the full TDFT-TDHF response function is often written as
!
!        K_ai,bj    K_ai,jb         A_ai,bj   B_ai,bj
!                                =  
!        K_ia,bj    K_ia,jb         B*_ai,bj  A*_ai,bj
!
! v states are restricted to valence band states 
! whereas c are conduction band states
! the sums are restricted to NBANDSLF bands below and above the Fermi-level

! loops over n1/n2/n3 and n4 are done in blocks of NSTRIP/NLOC
!  where NSTRIP is the number of local bands, and NLOC the total number
!
! outline of the structure of the routine:
!
!   allocate array W1 for merged block n1
!   allocate array W2 for merged block n2
!
!   allocate array W3 for merged block n3
!   allocate array W4 for merged block n4
!   allocate charge array (WKAPPA) of size merged block n1 x NBANDSLF
!     (this costs possibly a lot of memory and restricts n1)
!
!   allocate charge array of size merged WA block n4 x merged block n1
!
!   loop over all k1-points
!     loop overal all v1-states in blocks (preferably all n1)
!        gather wavefunction in the current block
!        loop overall k2-points
!           loop overal all v2-states in blocks (preferably all n2)
!              gather wavefunction in the current block
!
!              calculate F(n1,n3,n2,n4) = c*_k1+q,n3  c'_k2+q,n4(r') W(r',r) v'*_k2,n2(r), v_k1,n1(r)
!
! 
! ANTIRES = 0 corresponds to the Tamm Dancoff approximation, only direct terms
! ANTIRES = 1 applies a fancy approximation that includes
!             resonant-antiresonant coupling with w=0 (zero frequency) exact
! ANTIRES = 2 full version
!
!
!              
!**********************************************************************

! the routine has plenty of flags to determine the precise
! behaviour
!
! the flag single_prec_bse allows to select single precision storage of BSE 
! matrix and thus saves a factor 2 in storage
! this is currently the default
! if you want to use double precision define double_prec_bse in the precompiler step
!
#define double_prec_bse
#ifndef double_prec_bse
#define single_prec_bse
#endif

!
! the flag determine whether ZHEEVX is used
! ZHEEVX is much faster than ZHEEV, but requires twice as much memory 
! the flag MUST be set if scaLAPACK should be used
! otherwise the ZHEEV is used as fallback

#define USE_ZHEEVX


! this is already defined in scala.F, but private there
#ifdef scaLAPACK
      INTEGER,PARAMETER,PRIVATE :: BLOCK_CYCLIC_2D=1, DT_=1, &
                           CSRC_ =8, CTXT_=2, DLEN_=9, LLD_=9, &
                           MB_=5, M_=3, NB_=6, N_=4, RSRC_=7
#endif

  TYPE banddesc
    INTEGER :: VBMIN, VBMAX      ! band index for the VB minimum and VB maximum
    INTEGER :: CBMIN, CBMAX      ! band index for the CB minimum and CB maximum
    INTEGER :: CBMIN4, CBMAX4    ! band index for the CB minimum and CB maximum for n4
                                 ! this might be reduced for parallelization over bands
    INTEGER :: NGLB              ! block size for conduction bands = total number of conduction bands
    INTEGER :: NGLB4             ! block size for conduction bands for fourth index (N4 usually)
                                 ! [ N4 is distributed over nodes when band parallelization is used
                                 !   in this case NGLB4 is equal NGLB/ number of nodes]
  END TYPE banddesc

  TYPE bse_matrix_index
    INTEGER :: NCV
    INTEGER, ALLOCATABLE ::  N1(:)  ! valence band index
    INTEGER, ALLOCATABLE ::  N3(:)  ! conduction band index
    INTEGER, ALLOCATABLE ::  NK(:)  ! k-point index
    INTEGER, ALLOCATABLE ::  ISP(:) ! spin index
    INTEGER, ALLOCATABLE ::  INDEX(:,:,:,:) ! yield index into Hamilton matrix for given valence, conduction, k-point and spin
  END TYPE bse_matrix_index


    INTEGER,SAVE :: NBSEEIG          ! Number of BSE eigenfunctions written out in BSEEIGEN file
                                     ! starting from the lowest exciton, ie. LAMBDA=1,NBSEEIG.
CONTAINS

  SUBROUTINE CALCULATE_BSE( &
       IBSE,P,WDES,NONLR_S,NONL_S,W,LATT_CUR,LATT_INI,  &
       T_INFO,DYN,INFO,IO,KPOINTS,SYMM,GRID,LMDIM,CQIJ, &
       WGW, SHIFT, & 
       NBANDSO,NBANDSV, OMEGAMAX_IN, NQPOINT, KPOINT_BSE, ISP_IN, & 
       LHARTREE, LADDER, LTRIPLET, ANTIRES, LGWLF, LFXC, NEDOS_IN, NELMGW )

    USE base
    USE pseudo
    USE nonl_high
    USE msymmetry
    USE mpimy
    USE mgrid
    USE mkpoints
    USE constant
    USE poscar
    USE pot
    USE pawm
    USE kpoints_change
    USE full_kpoints
    USE mlr_optic
    USE dfast
    USE choleski
    USE ini
    IMPLICIT NONE
    ! structures
    INTEGER              IBSE
    TYPE (type_info)     T_INFO
    TYPE (potcar)        P(T_INFO%NTYP)
    TYPE (wavedes)       WDES
    TYPE (nonlr_struct)  NONLR_S
    TYPE (nonl_struct)   NONL_S
    TYPE (wavespin)      W
    TYPE (latt)          LATT_CUR, LATT_INI
    TYPE (dynamics)      DYN
    TYPE (info_struct)   INFO
    TYPE (in_struct)     IO
    TYPE (kpoints_struct) KPOINTS
    TYPE (symmetry)      SYMM
    TYPE (grid_3d)       GRID       ! grid for wavefunctions
    INTEGER  LMDIM
    OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    REAL(q)              SHIFT      ! complex frequency shift for broadening
    INTEGER              NBANDSO    ! number of occupied states included in BSE
    INTEGER              NBANDSV    ! number of virtual states included in BSE
    REAL(q)              OMEGAMAX_IN! maximum considered transition energy in BSE
    INTEGER              NQPOINT    ! q-point for which excitations are calculated (usually optical e.g. q=0)
    INTEGER              KPOINT_BSE(3) ! reciprocal lattice vector G (g=G+q) at which response is evaluated
    INTEGER              ISP_IN     ! spin index to be considered, if this is a negative number
                                    ! negative number: both spin channels are treated simultaneously (only valid for ISPIN=2)
                                    ! ISP_IN=1 : treat only spin channel 1 (and pretend the system is non-magnetic)
                                    ! ISP_IN=2 : treat only spin channel 2 (and pretend the system is non-magnetic)
    LOGICAL              LHARTREE   ! include Hartree effects
    LOGICAL              LTRIPLET   ! calculate triplet
    LOGICAL              LADDER     ! local field effects beyond RPA (ladder diagrams)
    INTEGER              ANTIRES    ! how to treat antiresonant part
    ! presently implemented 0 = Tamm-Dankoff
    ! presently implemented 1 -> correct w=0 properties 
    LOGICAL              LGWLF      ! use screened W from GW for exchange interactions
    LOGICAL              LFXC       ! include DFT xc kernel
    INTEGER              NEDOS_IN   ! number grid points in DOS (read from INCAR)
    INTEGER              NELMGW     ! number of timesteps in solving the BSE equation using time propagation
! local
    REAL(q)              OMEGAMAX   ! maximum considered transition energy in BSE
    INTEGER              NEDOS      ! number grid points in DOS
    TYPE(banddesc) :: BD(WDES%ISPIN)! descriptor for describing which bands are included, spin up
    TYPE(banddesc) :: B1,B2         ! current descriptor for the spin loops
    TYPE(bse_matrix_index) :: BSE_INDEX
    GDEF, ALLOCATABLE :: TWOELECTRON3O(:,:,:,:)   ! resonant-resonant coupling
    GDEF, ALLOCATABLE :: BTWOELECTRON3O(:,:,:,:)  ! resonant-antiresonant coupling 
    TYPE (wavespin) WHF
    TYPE (wavefun1),ALLOCATABLE :: W1(:), W2(:), W3(:), W4(:)
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2, WDESK3, WDESK4
    INTEGER :: NSTRIP               ! block size used internally in TWOELECTRON4O routines
    INTEGER :: NSTRIPV              ! block size for valence bands (valence bands are not distributed over nodes)
    INTEGER :: N 
    INTEGER :: NPOS1, NSTRIP1       ! base index and width of the n1 block
    INTEGER :: NPOS2, NSTRIP2       ! base index and width of the n1 block
    INTEGER :: K1, K2, K3, K4, K2_LOCAL, K2_COLLECT, K2_DONE, K4_LOCAL
    INTEGER :: ALLOC_VALENCE_BANDS, ALLOC_CONDUCTION_BANDS, ALLOC_KPOINTS
    INTEGER :: NCBD, NCBD4          ! max # of conduction bands over different spin orientations
    REAL(q) :: NFFTW
    REAL(q) :: NFLOAT4O, NFFT4O     ! number of BLAS3 operations in 4 orbital routines
    LOGICAL :: LEX_INTERPOLATED     ! use interpolation of the exchange like contribution (B)
    LOGICAL :: LscaLAPACKaware      ! distribute data using BLACS
    INTEGER :: NCV, IFAIL
    INTEGER :: ISP, ISP2, ISP_LOW, ISP_HIGH
    LOGICAL :: W1EQUALW2, LKPOINT_PARALLEL
    REAL(q), PARAMETER :: G2ZERO=1E-12_q
    REAL(q) :: AHARTREE
    LOGICAL :: LFULL
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
#ifdef single_prec_bse
    GDEFS, ALLOCATABLE :: AMAT(:,:), BMAT(:,:), COVL(:,:)
    GDEFS, ALLOCATABLE :: AMAT_SCALA(:), BMAT_SCALA(:), DIAG_ASCALA(:)
#else
    GDEF, ALLOCATABLE :: AMAT(:,:), BMAT(:,:), COVL(:,:)
    GDEF, ALLOCATABLE :: AMAT_SCALA(:), BMAT_SCALA(:), DIAG_ASCALA(:)
#endif

    GDEF              :: TMP
    REAL(q), ALLOCATABLE ::  R(:), RB(:)
    COMPLEX(q) :: CSUM_A           ! sum of diagonal components of matrix A
! scaLAPACK
    INTEGER :: NP,NQ,NPROW,NPCOL,MYROW,MYCOL
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER :: nd

    MULTIPLY_FRACTIONAL_OCC=.TRUE.
!test
!    WRITE(*,*) 'conjugate 20 occ'
!    W%CW(:,1:4,1:20,1)=W%CW(:,1:4,1:20,1)*(0.0_q,1.0_q)
!    W%CPROJ(:,1:4,1:20,1)=W%CPROJ(:,1:4,1:20,1)*(0.0_q,1.0_q)
!testend


#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('CALCULATE_BSE: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#endif
    ! smaller values are dangerous and are usually not what the user desires
    NEDOS=MAX(NEDOS_IN,1000)

    IF (ISP_IN<0 .AND. WDES%ISPIN==1) THEN
       WRITE(0,*) 'internal error in CALCULATE_BSE: ISP is negative but ISPIN=1'
       STOP
    ENDIF

    IF (WDES%LGAMMA .AND. (.NOT. WGW%GRID%REAL2CPLX .OR. .NOT. WGW%GRID%LREAL)) THEN
       WRITE(0,*) 'internal error in CALCULATE_BSE: WGW must apply real to complex FFT'
       STOP
    ENDIF
!=======================================================================
! preparation (literal copy from local_field.F
!=======================================================================
    ! read in WPOT files
    ! safer here, since it can happen that not all nodes call GET_WPOT simultaneously
    ! (e.g. k-point parallelization)
    ! this might cause problems with SHMEM calls
    IF (LGWLF .AND. .NOT. ASSOCIATED(WPOTH)) THEN
       CALL INIT_WPOT_HANDLE( WPOTH, WGW, KPOINTS_FULL%NKPTS, IO%IU6, IO%IU0, 1, 1, 1 )
       DO K1=1,KPOINTS_FULL%NKPTS
          IF (W%WDES%WTKPT(K1)==0) CYCLE
! NKREDLF not presently used by bse.F
!          IF (SKIP_THIS_KPOINT_IN_LF(W%WDES%VKPT(:,K1), NKREDLFX, NKREDLFY, NKREDLFZ)) CYCLE
          K2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K1)-W%WDES%VKPT(:,1),KPOINTS_FULL)
          IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, K2, POTFAK, LFULL)
       ENDDO
    ENDIF

    WHF=W      ! use temporarily another WDES
    WHF%WDES => WDES_FOCK

!  switch off NKRED
    NKREDX=1
    NKREDY=1
    NKREDZ=1

! determine wether the k1+k2 is found in the full k-point grid or not
    DO K1=1,KPOINTS_FULL%NKPTS
       IF (LIDENTICAL_KPOINT(WHF%WDES%VKPT(:,1)+WHF%WDES%VKPT(:,2),KPOINTS_FULL%VKPT(:,K1))) EXIT
    ENDDO
   
    IF (K1==KPOINTS_FULL%NKPTS+1 .AND. ANTIRES>=1) THEN
       LEX_INTERPOLATED=.TRUE.      ! not found use LEX_INTERPOLATED (slower)
    ELSE
       LEX_INTERPOLATED=.FALSE.     ! found, LEX_INTERPOLATED not required
    ENDIF

! presently the conjugated version is not tested or even coded up
    IF (ANTIRES>=1 .AND. .NOT. WDES%LORBITALREAL) THEN
       CALL VTUTOR('E','BSE ANTIRES', &
            0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
       CALL VTUTOR('E','BSE ANTIRES', &
            0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
    ENDIF
! set the wavefunction descriptors
    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK2,0)
    CALL SETWDES(WHF%WDES,WDESK3,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    IF (IO%IU6>=0) THEN
      WRITE(IO%IU6,'("NQ=",I4,3F10.4,", ")') NQPOINT,WHF%WDES%VKPT(:,NQPOINT)
    ENDIF

    W1EQUALW2=.FALSE.

    DO ISP=1,WDES%ISPIN
       CALL SET_BAND_PARAMETERS(W, ISP, NBANDSO, NBANDSV, & 
            BD(ISP)%VBMAX, BD(ISP)%VBMIN, BD(ISP)%CBMIN, BD(ISP)%CBMAX, & 
            NSTRIPV, NSTRIP, BD(ISP)%NGLB, BD(ISP)%CBMIN4, BD(ISP)%CBMAX4, BD(ISP)%NGLB4, W1EQUALW2,  LKPOINT_PARALLEL)
    ENDDO

    IF (ISP_IN>0) THEN
       ! reset NSTRIP and NSTRIPV to the optimal values
       CALL SET_BAND_PARAMETERS(W, ISP_IN, NBANDSO, NBANDSV, & 
            BD(ISP_IN)%VBMAX, BD(ISP_IN)%VBMIN, BD(ISP_IN)%CBMIN, BD(ISP_IN)%CBMAX, & 
            NSTRIPV, NSTRIP, BD(ISP_IN)%NGLB, & 
            BD(ISP_IN)%CBMIN4, BD(ISP_IN)%CBMAX4, BD(ISP_IN)%NGLB4, W1EQUALW2, LKPOINT_PARALLEL)

       ALLOC_VALENCE_BANDS=BD(ISP_IN)%VBMAX-BD(ISP_IN)%VBMIN+1
       ALLOC_CONDUCTION_BANDS=BD(ISP_IN)%NGLB
       ISP_LOW =ISP_IN
       ISP_HIGH=ISP_IN
    ELSE
       ALLOC_VALENCE_BANDS=MAX(BD(1)%VBMAX-BD(1)%VBMIN+1,BD(2)%VBMAX-BD(2)%VBMIN+1)
       ALLOC_CONDUCTION_BANDS=MAX(BD(1)%NGLB,BD(2)%NGLB)
       ISP_LOW =1
       ISP_HIGH=2

       ! TODO this requires a little bit more consideration
       ! for spin polarized calculations at the gamma point this increases the memory demand
       W1EQUALW2=.FALSE.
       IF (BD(1)%NGLB/=BD(2)%NGLB .OR. BD(1)%NGLB4 /= BD(2)%NGLB4) THEN
          CALL VTUTOR('E','BSE spin', &
               &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
!          CALL VTUTOR('S','BSE spin', &
!               &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
!          STOP
       ENDIF
    ENDIF

    IF (ABS(WHF%WDES%VKPT(1,NQPOINT))>1E-6_q .OR. ABS(WHF%WDES%VKPT(2,NQPOINT))>1E-6_q .OR. ABS(WHF%WDES%VKPT(3,NQPOINT))>1E-6_q) THEN
       ! allocate space for all k-points in the full wedge
       CALL  ALLOCATE_CDER_BETWEEN_STATES_Q( WHF%WDES%NB_TOT, MAX(BD(ISP_LOW)%VBMAX, BD(ISP_HIGH)%VBMAX), & 
            WHF%WDES%NKPTS , WHF%WDES%ISPIN)
    ENDIF


    IF (LKPOINT_PARALLEL) THEN
       IF (IO%IU6>=0) WRITE(IO%IU6,*) 'parallelization over k-points'
    ELSE
       IF (IO%IU6>=0) WRITE(IO%IU6,*) 'parallelization over bands'
    ENDIF

    ! allocate the orbitals: these contain the states in the conduction or valence bands 
    ALLOCATE(W1 (ALLOC_VALENCE_BANDS))
    ALLOCATE(W2 (ALLOC_VALENCE_BANDS))
    ALLOCATE(W3 (ALLOC_CONDUCTION_BANDS))
    ALLOCATE(W4 (ALLOC_CONDUCTION_BANDS))

    DO N=1,ALLOC_VALENCE_BANDS
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL NEWWAV(W2(N) , WDESK2,.TRUE.)
    ENDDO

    DO N=1,ALLOC_CONDUCTION_BANDS
       CALL NEWWAV(W3(N) , WDESK3,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL NEWWAV(W4(N) , WDESK4,.TRUE.)
    ENDDO

    ISP=ISP_LOW
    IF (ANTIRES >=2) THEN
       nd=2*ndata
    ELSE
       nd=ndata
    ENDIF

    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,4I4)') ' allocating two-electron 4 orbital integral table',NSTRIPV, BD(ISP)%NGLB, NSTRIPV, BD(ISP)%NGLB4
    CALL REGISTER_ALLOCATE(8._q*nd* SIZE(TWOELECTRON3O) , "bse")

    NCBD=MAX( BD(ISP_LOW)%NGLB,BD(ISP_HIGH)%NGLB) ; NCBD4=MAX( BD(ISP_LOW)%NGLB4,BD(ISP_HIGH)%NGLB4)
    ALLOCATE(TWOELECTRON3O( NSTRIPV, NCBD, NSTRIPV, NCBD4))
    IF (ANTIRES>=2) ALLOCATE(BTWOELECTRON3O( NSTRIPV, NCBD, NSTRIPV, NCBD4))

    OMEGAMAX=OMEGAMAX_IN
    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(/A,/2(A,I5,2X,A,I5/))')' Bands included in the BSE', & 
            ' VB(min)=',BD(ISP)%VBMIN,'VB(max)=',BD(ISP)%VBMAX,' CB(min)=',BD(ISP)%CBMIN,'CB(max)=',BD(ISP)%CBMAX

       IF (OMEGAMAX>0) THEN
          WRITE(IO%IU6,'(A,F10.5)') ' electron-hole pairs beyond OMEGAMAX are removed OMEGAMAX=',OMEGAMAX
       ENDIF

       IF (LADDER .AND. (AEXX /=0 .OR. L_MODEL_HF .OR. LGWLF) ) THEN
          IF (LGWLF) THEN
             WRITE(IO%IU6,'(A,3F14.7)')' W is read from the files WXXXX.tmp or WFULLXXXX.tmp if present'
          ELSE
             WRITE(IO%IU6,'(A,F14.7,A,F14.7)')' parameters for screened Coulomb W: AEXX=',AEXX,' HFSCREEN=',HFSCREEN
          ENDIF
          IF (.NOT. LHARTREE .OR. LTRIPLET) THEN
             WRITE(IO%IU6,'(A)') ' only ladders are included (Hartree term=RPA part is switched off)'
          ENDIF
       ELSE
          IF (.NOT. LHARTREE .OR. LTRIPLET) THEN
             WRITE(IO%IU6,'(A)') ' IP particle spectrum'
          ELSE
             WRITE(IO%IU6,'(A)')' simple RPA calculation, excitonic effects (ladders) are not included'
          ENDIF
       ENDIF
       IF (LHARTREE .AND. LFXC .AND. .NOT. LGWLF) THEN
          WRITE(IO%IU6,'(A)')' xc-kernel from LOCAL part of DFT is included (LFXC=.TRUE.)'
       ENDIF
    ENDIF

    NFFTW = 0
    NFLOAT4O=0 ; NFFT4O=0
!==========================================================================
! few BSE specific things
!==========================================================================
    ALLOC_KPOINTS=0
    DO K1=1,WDES%NKPTS
       IF (WHF%WDES%WTKPT(K1)==0) CYCLE
       ALLOC_KPOINTS=ALLOC_KPOINTS+1
    ENDDO

    CALL SET_BSE_MATRIX_INDEX( ISP_LOW, ISP_HIGH, ALLOC_KPOINTS, BD, BSE_INDEX, WHF, OMEGAMAX, NCV, NQPOINT )

    IF (NCV ==0) THEN
       CALL VTUTOR('E','BSE NCV', &
            0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
       CALL VTUTOR('S','BSE NCV', &
            0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
    ENDIF

    LscaLAPACKaware=.FALSE.
#ifdef USE_ZHEEVX
    IF (LscaLAPACK) THEN
       LscaLAPACKaware=.TRUE.
    ENDIF
#endif
    IF (IBSE>0) LscaLAPACKaware=.FALSE.
!==========================================================================
! intialize scaLAPACK (usually done during the matrix diagonalization)
! but if the scaLAPACKaware routines are used, we need to do
! it here and now
!==========================================================================
    IF ( IBSE==0) THEN
    IF ( LscaLAPACKaware) THEN
#ifdef scaLAPACK
       CALL INIT_scala( WDES%COMM, NCV)
       CALL BLACS_GRIDINFO(DESCSTD(CTXT_), NPROW, NPCOL,MYROW,MYCOL)
       NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
       NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)
#ifdef single_prec_bse
       IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE (scaLAPACK) single prec attempting allocation of',1.0_q/1E9*NP*NQ*4*nd ,' Gbyte  rank=',NCV
       IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE (scaLAPACK) single prec attempting allocation of',1.0_q/1E9*NP*NQ*4*nd ,' Gbyte  rank=',NCV
       CALL REGISTER_ALLOCATE(4._q*nd* MAX(NP*NQ,1) , "bse")
#else
       IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE (scaLAPACK) attempting allocation of',1.0_q/1E9*NP*NQ*8*nd ,' Gbyte  rank=',NCV
       IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE (scaLAPACK) attempting allocation of',1.0_q/1E9*NP*NQ*8*nd ,' Gbyte  rank=',NCV
       CALL REGISTER_ALLOCATE(8._q*nd* MAX(NP*NQ,1) , "bse")
#endif
       ALLOCATE( AMAT( 1, 1), AMAT_SCALA( MAX(NP*NQ,1)), R(NCV), DIAG_ASCALA(NCV))
       IF (ANTIRES>=2) ALLOCATE( BMAT( 1, 1), BMAT_SCALA( MAX(NP*NQ,1)), RB(NCV))
#endif
    ELSE
#ifdef single_prec_bse
       IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE single prec attempting allocation of',1.0_q/1E9*NCV*NCV*4*nd ,' Gbyte  rank=',NCV
       IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE single prec  attempting allocation of',1.0_q/1E9*NCV*NCV*4*nd ,' Gbyte  rank=',NCV
       CALL REGISTER_ALLOCATE(4._q*nd* NCV*NCV , "bse")
#else
       IF (IO%IU0>=0) WRITE(IO%IU0,'(A,F8.3,A,I7)') ' BSE attempting allocation of',1.0_q/1E9*NCV*NCV*8*nd,' Gbyte  rank=',NCV
       IF (IO%IU6>=0) WRITE(IO%IU6,'(A,F8.3,A,I7)') ' BSE attempting allocation of',1.0_q/1E9*NCV*NCV*8*nd,' Gbyte  rank=',NCV
       CALL REGISTER_ALLOCATE(8._q*nd* NCV*NCV , "bse")
#endif

#ifdef USE_ZHEEVX
       ALLOCATE( AMAT( NCV, NCV), R(NCV),  AMAT_SCALA(1), DIAG_ASCALA(NCV))
       IF (ANTIRES>=2) ALLOCATE( BMAT( NCV, NCV), BMAT_SCALA(1), RB(NCV))
#else 
       ALLOCATE( AMAT( NCV, NCV), R(NCV), AMAT_SCALA(1), DIAG_ASCALA(NCV))
       IF (ANTIRES>=2) ALLOCATE( BMAT( NCV, NCV), BMAT_SCALA(1), RB(NCV))
#endif
    ENDIF
    ENDIF
#ifdef use_bse_te
    IF (IBSE>0) THEN
       ALLOCATE( AMAT( 1, 1), AMAT_SCALA(1), R(NCV))
       IF (ANTIRES>=2) ALLOCATE( BMAT( 1, 1), BMAT_SCALA( 1), RB(NCV))
       CALL TW4O_STORE_STRIP_PREPARE(WHF, IO, NCV,  LKPOINT_PARALLEL, WDES%NKPTS) 
    ENDIF
#endif

    CALL DUMP_ALLOCATE(IO%IU6)
    IF (IO%LOPEN) CALL WFORCE(IO%IU6)

    IF ( LscaLAPACKaware) THEN
       AMAT_SCALA=0
       IF (ANTIRES>=2) BMAT_SCALA=0
    ELSE
       AMAT=0
       IF (ANTIRES>=2) BMAT=0
    ENDIF

    IF (IO%IU0>=0) WRITE(IO%IU0,*) 'BSE setting up matrix'
    CALL START_TIMING("G")

    ! Tamm-Dankoff with correct properties at large w

    IF (ISP_IN<0 .OR. WHF%WDES%LNONCOLLINEAR ) THEN
       ! spin polarized calculation weight for Hartree term is one
       AHARTREE=1.0_q
    ELSE
       ! non spin polarized: weight for Hartree is two
       AHARTREE=2.0_q
    ENDIF

    ! correct properties at w=0
    ! at w=0  resonant-resonant coupling = antires-antires=resonant-antires
    ! so Hartree needs to be counted twice (this however spoils high frequency behaviour)
    IF (ANTIRES==1) AHARTREE=AHARTREE*2.0_q
!==========================================================================
! outer loop body for first set of pair states
!==========================================================================
    DO ISP=ISP_LOW,ISP_HIGH
    B1=BD(ISP)
    DO K1=1,WDES%NKPTS
       ! generate the proper descriptor for W1 wavefunctions
       CALL SETWDES(WHF%WDES,WDESK1,K1)
       IF (WHF%WDES%WTKPT(K1)==0) CYCLE

       K3 =KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,NQPOINT),KPOINTS_FULL)

       ! skip if there are no occupied bands
       IF (B1%VBMAX==0) CYCLE

       CALL W1_GATHER_GLB( WHF, B1%VBMIN, B1%VBMAX, ISP, W1)
       NFFTW=NFFTW+B1%VBMAX-B1%VBMIN+1

       ! generate the proper descriptor for W3 wavefunctions at k-point K3
       CALL SETWDES(WHF%WDES,WDESK3 ,K3)

       CALL W1_GATHER_GLB( WHF, B1%CBMIN, B1%CBMAX, ISP, W3)
       NFFTW=NFFTW+B1%CBMAX-B1%CBMIN+1

       
       IF (ABS(WHF%WDES%VKPT(1,NQPOINT))>1E-6_q .OR. ABS(WHF%WDES%VKPT(2,NQPOINT))>1E-6_q .OR. ABS(WHF%WDES%VKPT(3,NQPOINT))>1E-6_q) THEN
          ! set transition matrix elements
          CALL ONEELECTRON_TRANS(WHF, P, LATT_CUR, ISP, WGW, & 
               W1, K1, B1%VBMIN, B1%VBMAX, & 
               W3, K3, B1%CBMIN, B1%CBMAX, KPOINT_BSE, NFFTW )
       ENDIF


       IF (NSTRIPV == 0) NSTRIPV = 1 

       DO NPOS1=B1%VBMIN, B1%VBMAX, NSTRIPV
          CALL GWPROGRESS(IO%IU0, K1, WDES%NKPTS, NPOS1-B1%VBMIN+1, B1%VBMAX-B1%VBMIN+1 )
          NSTRIP1=MIN(B1%VBMAX+1-NPOS1,NSTRIPV)

!==========================================================================
! inner loop body for second set of pair states
!==========================================================================
          DO ISP2=ISP,ISP_HIGH
          B2=BD(ISP2)
          K2=K1-1
          IF (ISP2/=ISP) K2=0
          DO
             K2=K2+1
             ! distribute the bands over k-points in a round robin fashion
             K2_DONE =0                   ! counts the number of k-points that have been collected
             K2_LOCAL=-1                  ! determine which k-point treated locally
             DO K2_COLLECT=K2, WDES%NKPTS ! loop from present K2 up to number of k-points in wavefunction descriptor
                ! generate the proper descriptor for W2 wavefunctions
                CALL SETWDES(WHF%WDES,WDESK2,K2_COLLECT)

                ! skip bands that are empty
                IF (WHF%WDES%WTKPT(K2_COLLECT)==0) CYCLE

                K2_DONE=K2_DONE+1  ! new k-point to be included

                K4=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K2_COLLECT)+WHF%WDES%VKPT(:,NQPOINT),KPOINTS_FULL)

                IF (LKPOINT_PARALLEL) THEN
                   IF ( B2%VBMAX-B2%VBMIN >0) &
                        CALL W1_GATHER_KSEL( WHF, B2%VBMIN, B2%VBMAX, ISP2, W2, K2_DONE)
                   NFFTW=NFFTW+(B2%CBMAX-B2%CBMIN+1)/WDES%NB_PAR
                ELSE
                   IF (W1EQUALW2) THEN
                      W2=W1
                   ELSE
                      IF ( B2%VBMAX-B2%VBMIN >0) &
                           CALL W1_GATHER_GLB( WHF, B2%VBMIN, B2%VBMAX, ISP2, W2)
                      NFFTW=NFFTW+B2%VBMAX-B2%VBMIN+1
                   ENDIF
                ENDIF

                CALL SETWDES(WHF%WDES,WDESK4,K4)

                IF (LKPOINT_PARALLEL) THEN
                   IF ( B2%VBMAX-B2%VBMIN >0) &
                        CALL W1_GATHER_KSEL( WHF, B2%CBMIN, B2%CBMAX, ISP2, W4, K2_DONE)
                   NFFTW=NFFTW+(B2%CBMAX-B2%CBMIN+1)/WDES%NB_PAR
                ELSE
                   IF (W1EQUALW2) THEN
                      W4=W3
                   ELSE
                      IF ( B2%VBMAX-B2%VBMIN >0) &
                           CALL W1_GATHER_GLB( WHF, B2%CBMIN, B2%CBMAX, ISP2, W4)
                      NFFTW=NFFTW+B2%CBMAX-B2%CBMIN+1
                   ENDIF
                ENDIF

                IF (K2_DONE==WDES%NB_LOW .OR. .NOT. LKPOINT_PARALLEL) THEN
                   K2_LOCAL=K2_COLLECT
                   K4_LOCAL=K4
                ENDIF
                IF (K2_DONE==WDES%NB_PAR .OR. .NOT. LKPOINT_PARALLEL) EXIT
             ENDDO
!==========================================================================
!  set up BSE matrix
!==========================================================================
             ! at this point each CPU holds the k-points corresponding to K2_LOCAL 
             ! in W2 (and corresponding to K2-NQ in W4)
             ! set K2 and K4 again
             ! (Mind the condition K2>=1, some nodes might skip this loop altogheter)
             K2=K2_LOCAL
             K4=K4_LOCAL
             IF (K2>=1) THEN
                CALL SETWDES(WHF%WDES,WDESK2,K2)
                CALL SETWDES(WHF%WDES,WDESK4,K4)
             ENDIF
             DO NPOS2=B2%VBMIN, B2%VBMAX, NSTRIPV
                NSTRIP2=MIN(B2%VBMAX+1-NPOS2,NSTRIPV)

                IF (K2>=1) THEN

                   TWOELECTRON3O=0
                   IF (ISP2==ISP) THEN
                      IF (LHARTREE .AND. .NOT. LTRIPLET) THEN
                           CALL TWOELECTRON4O_ACC_HARTREE(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
                           LFXC .AND. .NOT. LGWLF, .FALSE. , AHARTREE,  &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                           CALL TWOELECTRON4O_ACC_HEAD(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
                           LFXC .AND. .NOT. LGWLF, .FALSE. , AHARTREE,  &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                      ENDIF
                      ! copy Hartree-term to resonant-antiresonant part
                      IF (ANTIRES>=2) BTWOELECTRON3O=TWOELECTRON3O
                      
                      IF (LADDER .AND. (AEXX /=0 .OR. L_MODEL_HF .OR. LGWLF) ) THEN
                         ! A contributions
                         CALL TWOELECTRON4O_ACC_DIRECT(WHF, P, LATT_CUR, ISP, WGW, DELTA_COND,  &
                              W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                              TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                         
                         ! B contributions folded down
                         IF (ANTIRES==1 .AND. .NOT. LEX_INTERPOLATED) &
                              CALL TWOELECTRON4O_ACC_EX(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                              W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                              TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                         ! B contributions stored seperately in BTWOELECTRON3O
                         IF (ANTIRES>=2 .AND. .NOT. LEX_INTERPOLATED) &
                              CALL TWOELECTRON4O_ACC_EX(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                              W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                              BTWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                         
                         ! B contributions folded down
                         IF (ANTIRES==1 .AND. LEX_INTERPOLATED) &
                              CALL TWOELECTRON4O_ACC_EX_INTER(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                              W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                              TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, & 
                              NFLOAT4O, NFFT4O, NSTRIP, LATT_CUR%B)
                         ! B contributions stored seperately in BTWOELECTRON3O
                         IF (ANTIRES>=2 .AND. LEX_INTERPOLATED) &
                              CALL TWOELECTRON4O_ACC_EX_INTER(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                              W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                              BTWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, & 
                              NFLOAT4O, NFFT4O, NSTRIP, LATT_CUR%B)
                      ENDIF
                   ELSE
                      ! spin-up spin-down coupling only via Hartree term
                      IF (LHARTREE .AND. .NOT. LTRIPLET) &
                           CALL TWOELECTRON4O_ACC_HARTREE_SFLIP(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
                           LFXC .AND. .NOT. LGWLF, .FALSE. ,AHARTREE,  &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, B1%CBMIN, B1%CBMAX, B1%CBMIN4, B1%CBMAX4, B1%VBMIN, & 
                           B2%CBMIN, B2%CBMIN4, B2%CBMAX4, B2%VBMIN, NFLOAT4O, NFFT4O, NSTRIP)
                      ! copy Hartree-term to resonant-antiresonant part
                      IF (ANTIRES>=2) BTWOELECTRON3O=TWOELECTRON3O
                   ENDIF
                ENDIF

                IF (IBSE==0) THEN
                IF (LscaLAPACKaware) THEN
#ifdef scaLAPACK
                   CALL TWOELECTRON4O_STORE_SCALA(WHF, ISP, ISP2, ISP_LOW, &
                        K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
                        TWOELECTRON3O, B1, B2, AMAT_SCALA, NCV, BSE_INDEX,1)
                   IF (ANTIRES>=2) &
                   CALL TWOELECTRON4O_STORE_SCALA(WHF, ISP, ISP2, ISP_LOW, &
                        K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
                        BTWOELECTRON3O, B1, B2, BMAT_SCALA, NCV, BSE_INDEX,0)
#endif
                ELSE IF (K2>=1) THEN
                   CALL TWOELECTRON4O_STORE(WHF, ISP, ISP2, ISP_LOW,  &
                        K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
                        TWOELECTRON3O, B1, B2, AMAT, BSE_INDEX,1)
                   IF (ANTIRES>=2) &
                   CALL TWOELECTRON4O_STORE(WHF, ISP, ISP2, ISP_LOW,  &
                        K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
                        BTWOELECTRON3O, B1, B2, BMAT, BSE_INDEX,0)
                ENDIF
                ENDIF
#ifdef use_bse_te
                IF (IBSE>0) THEN
                   CALL TW4O_STORE_STRIP(WHF, ISP, ISP2, ISP_LOW, &
                        K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
                        TWOELECTRON3O, B1, B2, BSE_INDEX )
                ENDIF
#endif
             ENDDO
!==========================================================================
!  end of loop body
!==========================================================================
             K2=K2_COLLECT
             IF (K2>=WDES%NKPTS) EXIT
          ENDDO
          ENDDO
       ENDDO
    ENDDO
    ENDDO

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT4O, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT4O  , 1))

10  FORMAT(" BLAS level 3 operations / number of FFT's:"/ &
         " number of FFTs for wave wavefunctions          ",F10.0," fft"/ & 
         " number of operations in four-orbital integrals ",F10.2," Gflops, ",F10.0," fft")

    IF (IO%IU6>0) THEN
       WRITE(IO%IU6,10) NFFTW,NFLOAT4O/1E9, NFFT4O
    ENDIF
    ! ok finished get rid of WPOTH
    IF (ASSOCIATED(WPOTH)) CALL DESTROY_WPOT_HANDLE(WPOTH)
!==========================================================================
! deallocation
!==========================================================================
    DO N=1,ALLOC_CONDUCTION_BANDS
       CALL DELWAV(W3(N) ,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL DELWAV(W4(N) ,.TRUE.)
    ENDDO

    DO N=1,ALLOC_VALENCE_BANDS
      CALL DELWAV(W1(N) ,.TRUE.)
      IF (.NOT. W1EQUALW2) CALL DELWAV(W2(N) ,.TRUE.)
    ENDDO

    CALL DEREGISTER_ALLOCATE(8._q*nd* SIZE(TWOELECTRON3O) , "bse")
    DEALLOCATE(TWOELECTRON3O)
    IF (ANTIRES>=2) DEALLOCATE(BTWOELECTRON3O)
    DEALLOCATE(W1,W2,W3,W4)

    CALL DEALLOCW_CW(WHF)
!==========================================================================
! BSE  redistribution of matrix elements
!==========================================================================
    CALL STOP_TIMING("G",IO%IU6,'BSESET')

    IF (IO%IU0>=0) WRITE(IO%IU0,*) 'BSE redistributing all elements'

    IF ((IBSE==0).AND.(.NOT. LscaLAPACKaware)) THEN
#ifdef single_prec_bse
#ifdef gammareal
       CALLMPI( M_sum_single(WGW%COMM_INTER, AMAT  , SIZE(AMAT)))
       IF (ANTIRES>=2) THEN 
          CALLMPI( M_sum_single(WGW%COMM_INTER, BMAT  , SIZE(BMAT)))
       ENDIF
#else 
       CALLMPI( M_sum_single(WGW%COMM_INTER, AMAT  , 2*SIZE(AMAT)))
       IF (ANTIRES>=2) THEN 
          CALLMPI( M_sum_single(WGW%COMM_INTER, BMAT  , 2*SIZE(BMAT)))
       ENDIF
#endif
#else
       CALLMPI( M_sum_g(WGW%COMM_INTER, AMAT  , SIZE(AMAT)))
       IF (ANTIRES>=2) THEN 
          CALLMPI( M_sum_g(WGW%COMM_INTER, BMAT  , SIZE(BMAT)))
       ENDIF
#endif
    ENDIF

    CALL STOP_TIMING("G",IO%IU6,'BSESUM')

!    IF (WDES%COMM%NODE_ME==WDES%COMM%IONODE) THEN
!    WRITE(*,*)
!    WRITE(*,'(32F8.3)') REAL(AMAT(1:32,1:32),q)
!#ifndef gammareal
!    WRITE(*,*)
!    WRITE(*,'(32F8.3)') AIMAG(AMAT(1:32,1:32))
!#endif
!    ENDIF

    IF (IO%IU0>=0) WRITE(IO%IU0,*) 
    CALL DUMP_ALLOCATE(IO%IU6)
    IF (IO%LOPEN) CALL WFORCE(IO%IU6)
!==========================================================================
! at this point the BSE matrix has been set up and can be diagonalized
! here is the full (non TDA) version
!==========================================================================
   ! get the sum of diagonal elements of A 
    CSUM_A=(0.0_q,0.0_q)
#ifdef scaLAPACK
    IF (LscaLAPACKaware) THEN
       CALL TWOELECTRON4O_DIAG_SCALA( AMAT_SCALA, NCV, CSUM_A, DIAG_ASCALA, WHF%WDES%COMM)
    ELSE
#endif
       CALL TWOELECTRON4O_DIAG( AMAT, NCV, CSUM_A, DIAG_ASCALA)
#ifdef scaLAPACK
    ENDIF
#endif

    ! store A-B in A, and A+B in B
    IF (IBSE==0) THEN
    IF (ANTIRES>=2) THEN
       IF (LscaLAPACKaware) THEN
          DO K1=1,SIZE(AMAT_SCALA)
             TMP=AMAT_SCALA(K1)
             AMAT_SCALA(K1)=TMP+BMAT_SCALA(K1)
             BMAT_SCALA(K1)=TMP-BMAT_SCALA(K1)
          ENDDO
       ELSE
          DO K1=1,NCV
             DO K2=K1,NCV
                TMP=AMAT(K1,K2)
                AMAT(K1,K2)=TMP+BMAT(K1,K2)
                BMAT(K1,K2)=TMP-BMAT(K1,K2)
                ! for AMAT the lower triangle needs to be properly set up
                ! since below we multiply the matrix from the left and right
                AMAT(K2,K1)=GCONJG(AMAT(K1,K2))
                ! strictly speaking it suffices to set the upper triangle of B
                ! just to be sure, set the lower triangle as well
                BMAT(K2,K1)=GCONJG(BMAT(K1,K2))
             ENDDO
          ENDDO
       ENDIF
       ! diagonalize A-B
       CALL DIAG_BSE_MATRIX( NCV, BMAT, BMAT_SCALA, RB, LscaLAPACKaware, WDES%COMM, IO%IU0 )

       ! calculate   (A-B)^1/2 (A+B) (A-B)^1/2
       IF ( LscaLAPACKaware) THEN
#ifdef scaLAPACK
          CALL TDA_BSE_SCA( NCV, AMAT_SCALA, BMAT_SCALA, RB )
#endif
       ELSE
          CALL TDA_BSE( NCV, AMAT, BMAT, RB )
       ENDIF
       ! diagonalize (A-B)^1/2 (A+B) (A-B)^1/2
       CALL DIAG_BSE_MATRIX( NCV, AMAT, AMAT_SCALA, R, LscaLAPACKaware, WDES%COMM, IO%IU0 )
       ! eigenvalues of original problem
       R=SQRT(R)

       IF (IO%IU0>=0 ) THEN
          WRITE(IO%IU0,'(A,2F20.10)') ' plasmon correlation energy ', 0.5*(SUM(R)-REAL(CSUM_A,q))*KPOINTS_ORIG%WTKPT(NQPOINT)
          WRITE(IO%IU6,'(A,2F20.10)') ' plasmon correlation energy ', 0.5*(SUM(R)-REAL(CSUM_A,q))*KPOINTS_ORIG%WTKPT(NQPOINT)
       ENDIF

       ! left multiply by (A-B)^1/2 to obtain eigenvectors 
       IF ( LscaLAPACKaware) THEN
#ifdef scaLAPACK
          CALL TDA_BSE_EIGENVECTORS_SCA( NCV, BMAT_SCALA, RB, AMAT_SCALA, R )
#endif
       ELSE
          CALL TDA_BSE_EIGENVECTORS( NCV, BMAT, RB, AMAT, R )
       ENDIF
!==========================================================================
! use standard algorithms
!==========================================================================
    ELSE !!this happens when ANTIRES == 0 (TDA)
       CALL DIAG_BSE_MATRIX( NCV, AMAT, AMAT_SCALA, R, LscaLAPACKaware, WDES%COMM, IO%IU0 )
    ENDIF
    ENDIF

!==========================================================================
! BSEFATBANDS
!==========================================================================
    IF (IO%IU0>=0) &
       WRITE(IO%IU0,"(A12,I6,A17)") 'Writing out ',NBSEEIG,' BSE eigenvectors'

    IF ( LscaLAPACKaware) THEN
#ifdef scaLAPACK
       CALL WRITE_BSE_SCALA(WHF, LATT_CUR, ISP_LOW, ISP_HIGH, &
         BD, BSE_INDEX, R, AMAT_SCALA, NCV, NBSEEIG, IO%IU6)
#endif
    ELSE
!MB TODO: write out bse eigenvectors when not scalapack!
!      CALL WRITE_BSE(WHF,NCV,BSE_INDEX,EIGVECT,R,NBSEEIG)
    ENDIF

#ifdef use_bse_te
!==========================================================================
! use time evolution algorithm 
!==========================================================================
    IF (IBSE>0) THEN
       CALL TW4O_CHECK_STRIP(WHF)
       CALL INIT_BSE_OSZI_STRENGTH(WHF, LATT_CUR, ISP_LOW, ISP_HIGH, BD, BSE_INDEX)
       CALL CALCULATE_BSE_TIME_EVOLUTION(SHIFT,LATT_CUR%OMEGA, &
            WDES%ISPIN, WDES%NKPTS, NCV, WDES%WTKPT, OMEGAMAX, LADDER .OR. (LHARTREE .AND. .NOT. LTRIPLET), ANTIRES==2, NELMGW, IO)
    ENDIF
#endif
!==========================================================================
! calculate oszillator strength
!==========================================================================
    CALL STOP_TIMING("G",IO%IU6,'BSEDIAG')

    IF (IBSE==0) THEN
    ISP=1
    IF (IO%IU0>=0) WRITE(IO%IU0,*) 'BSE calculating oscillator strength'

    IF ( LscaLAPACKaware) THEN
#ifdef scaLAPACK
       CALL CALCULATE_BSE_OSZI_SCALA(WHF, NQPOINT, KPOINT_BSE, LATT_CUR, ISP_LOW, ISP_HIGH, & 
            SHIFT, BD, BSE_INDEX, R, AMAT_SCALA, NCV, ANTIRES, NEDOS, OMEGAMAX, IO%IU6)
#endif
    ELSE
       CALL CALCULATE_BSE_OSZI_STRENGTH(WHF, NQPOINT, KPOINT_BSE, LATT_CUR, ISP_LOW, ISP_HIGH, & 
            SHIFT, BD, BSE_INDEX, R, AMAT, ANTIRES, NEDOS, OMEGAMAX, IO%IU6)
    ENDIF
    CALL STOP_TIMING("G",IO%IU6,'BSEOSZI')
    ENDIF

    CALL DEALLOCATE_CDER_BETWEEN_STATES_Q
    CALL DEALLOCATE_BSE_MATRIX_INDEX( BSE_INDEX )

    IF ( LscaLAPACKaware) THEN
       DEALLOCATE( AMAT_SCALA, R)
       IF (ANTIRES>=2) DEALLOCATE( BMAT_SCALA, RB)
    ELSE
       DEALLOCATE( AMAT, R, AMAT_SCALA)
       IF (ANTIRES>=2) DEALLOCATE( BMAT, BMAT_SCALA, RB, DIAG_ASCALA)
    ENDIF

    RETURN

  END SUBROUTINE CALCULATE_BSE


!****************** SUBROUTINE  TWOELECTRON4O_STORE *******************
!
! this subroutine stores the two electron 4 orbital integrals
! in the BSE / TDFT Hamilton matrix AMAT
! this version of the routine is not scaLAPACK "aware" and
! requires that the full BSE matrix is stored on every node
! 
!**********************************************************************

  SUBROUTINE TWOELECTRON4O_STORE(WHF, ISP, ISP2, ISP_LOW, &
       K1, NPOS1, NSTRIP1, K2, NPOS2, NSTRIP2, K3, K4, &
       TWOELECTRON3O, B1, B2, AMAT, BSE_INDEX, IDIAG)

    USE constant
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    INTEGER ISP, ISP2, ISP_LOW
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    TYPE(banddesc) :: B1,B2         ! current descriptor for the spin loops
    GDEF ::  TWOELECTRON3O(:,:,:,:)
! actual dim TWOELECTRON3O( NSTRIPV, NGLB, NSTRIPV, NGLB4))
#ifdef single_prec_bse
    GDEFS :: AMAT(:,:)
#else
    GDEF :: AMAT(:,:)
#endif
  ! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NB3, NB4
    INTEGER NCV13, NCV24
    TYPE(bse_matrix_index) :: BSE_INDEX
    INTEGER :: IDIAG     ! add eigenvalues

    DO N1=1,NSTRIP1
       DO N2=1,NSTRIP2
          NB1_INTO_TOT=NPOS1-1+N1
          NB2_INTO_TOT=NPOS2-1+N2
          DO N4=1,B2%CBMAX4-B2%CBMIN4+1
             NB4_INTO_TOT=B2%CBMIN4-1+N4
             NB4         =N4
#define check_index
#ifdef check_index
             IF (N1 > SIZE(TWOELECTRON3O,1)) THEN
                WRITE(*,*)'internal error in TWOELECTRON4O_STORE: out of bounds 1 ',N1
                STOP
             ENDIF

             IF (N2 > SIZE(TWOELECTRON3O,3)) THEN
                WRITE(*,*)'internal error in TWOELECTRON4O_STORE: out of bounds 3 ',N2
                STOP
             ENDIF

             IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                WRITE(*,*)'internal error in TWOELECTRON4O_STORE: out of bounds 4 ',NB4
                STOP
             ENDIF
#endif
           ! NCV24=NB2_INTO_TOT-B1%VBMIN+1 +(B1%VBMAX-B1%VBMIN+1)*(NB4_INTO_TOT-B1%CBMIN+(B1%CBMAX-B1%CBMIN+1)*(K2-1))
             NCV24=BSE_INDEX%INDEX(NB2_INTO_TOT-B2%VBMIN+1, NB4_INTO_TOT-B2%CBMIN+1, K2, ISP2-ISP_LOW+1)
             IF (NCV24==0) CYCLE ! cycle if this pair is not included
#ifdef check_index
             IF ( NCV24 > SIZE(AMAT,2) ) THEN
                WRITE(*,*) NB2_INTO_TOT-B2%VBMIN+1, B2%VBMAX-B2%VBMIN+1,NB4_INTO_TOT-B2%CBMIN+1,(B2%CBMAX-B2%CBMIN+1),(K2-1)
                WRITE(*,*) 'internal error in  TWOELECTRON4O_STORE: NCV exceeds AMAT size-boundaries',NCV13,NCV24,SIZE(AMAT,1),SIZE(AMAT,2) 
                STOP
             ENDIF
#endif
             DO N3=1,B1%CBMAX-B1%CBMIN+1
                NB3_INTO_TOT=B1%CBMIN-1+N3
                NB3         =N3
#ifdef check_index
                IF ( NB3> SIZE(TWOELECTRON3O,2)) THEN
                   WRITE(*,*)'internal error in TWOELECTRON4O_STORE: out of bounds 2 ',NB3_INTO_TOT,NB3
                   STOP
                ENDIF
#endif
                ! build index NCV13 and NCV24
                ! NCV13=NB1_INTO_TOT-B1%VBMIN+1 +(B1%VBMAX-B1%VBMIN+1)*(NB3_INTO_TOT-B1%CBMIN+(B1%CBMAX-B1%CBMIN+1)*(K1-1))
                NCV13=BSE_INDEX%INDEX(NB1_INTO_TOT-B1%VBMIN+1, NB3_INTO_TOT-B1%CBMIN+1, K1, ISP -ISP_LOW+1)
                IF (NCV13==0) CYCLE ! cycle if this pair is not included
                ! check whether NCV13 is smaller or equal than SIZE(AMAT,1)
#ifdef check_index
                IF ( NCV13 > SIZE(AMAT,1) ) THEN
                   WRITE(*,*) NB2_INTO_TOT-B2%VBMIN+1, B2%VBMAX-B2%VBMIN+1,NB4_INTO_TOT-B2%CBMIN+1,(B2%CBMAX-B2%CBMIN+1),(K2-1)
                   WRITE(*,*) 'internal error in  TWOELECTRON4O_STORE: NCV exceeds AMAT size-boundaries',NCV13,NCV24,SIZE(AMAT,1),SIZE(AMAT,2) 
                   STOP
                ENDIF
                IF (AMAT(NCV13,NCV24) /= 0) THEN
                   WRITE(*,*) 'internal error in  TWOELECTRON4O_STORE: AMAT was already written',NCV13,NCV24
                   WRITE(*,*) ' attempt of overwriting points towards inconsistent indexing'
                   WRITE(*,*) ' correct the formulas for NCV13 and NCV24 in bse.F'
                   STOP
                ENDIF
#endif
                AMAT(NCV13,NCV24)=TWOELECTRON3O( N1, NB3 , N2, NB4)*WHF%WDES%WTKPT(1)

                ! diagonal components 
                ! subtract eigenvalue difference for the A matrix (IDIAG is 0.0 when BMAT is passed on)
                IF (K1==K2 .AND. NB1_INTO_TOT==NB2_INTO_TOT .AND. NB3_INTO_TOT==NB4_INTO_TOT .AND.  ISP== ISP2) THEN
                   IF (NCV13/= NCV24) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_STORE: not a diagonal element',NCV13, NCV24
                      WRITE(*,*) N1,N2,N3,N4,K1,K2
                      STOP
                   ENDIF
                   AMAT(NCV13, NCV24) = AMAT(NCV13, NCV24)-IDIAG*(WHF%CELTOT(NB1_INTO_TOT,K1, ISP)-WHF%CELTOT(NB3_INTO_TOT,K3, ISP))
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE TWOELECTRON4O_STORE


!****************** SUBROUTINE  TWOELECTRON4O_STORE_SCALA  ************
!
! this subroutine stores the two electron 4 orbital integrals
! in the BSE / TDFT Hamilton matrix AMAT
! this version is scaLAPACK "aware" and matter of fact only
! available of scalapack is used 
! 
!**********************************************************************

#ifdef scaLAPACK
  SUBROUTINE TWOELECTRON4O_STORE_SCALA(WHF, ISP, ISP2, ISP_LOW, &
       K1, NPOS1, NSTRIP1, K2_, NPOS2, NSTRIP2, K3, K4_, &
       TWOELECTRON3O, B1, B2, AMAT, NCV, BSE_INDEX, IDIAG)

    USE constant
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    INTEGER ISP, ISP2, ISP_LOW
    INTEGER K1, K2_, K3, K4_, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    TYPE(banddesc) :: B1,B2         ! current descriptor for the spin loops
    GDEF ::  TWOELECTRON3O(:,:,:,:)
! actual dim TWOELECTRON3O( NSTRIPV, NGLB, NSTRIPV, NGLB4))
    TYPE(bse_matrix_index) :: BSE_INDEX
    INTEGER :: IDIAG     ! add eigenvalues

#ifdef single_prec_bse
    GDEFS :: AMAT(:)
#else
    GDEF :: AMAT(:)
#endif
    INTEGER :: NCV
  ! local variables
    GDEF, ALLOCATABLE ::  TWOELECTRON3O_LOCAL(:,:,:,:)
    INTEGER N1, N2, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NB3, NB4
    INTEGER NCV13, NCV24
    INTEGER K2, K4, CBMIN4, CBMAX4
    INTEGER N
    INTEGER ITWOELECTRON3O(4)
! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP, NQ
    INTEGER I1RES, J1RES, IROW, JCOL
    INTEGER I1, I2, J1, J2
    INTEGER IFAIL

! allocate work array into which data is subsequently merged from all nodes
    ITWOELECTRON3O=SHAPE(TWOELECTRON3O)
    CALL M_max_i(WHF%WDES%COMM, ITWOELECTRON3O, 4 )

    ALLOCATE(TWOELECTRON3O_LOCAL(ITWOELECTRON3O(1),ITWOELECTRON3O(2), & 
                                 ITWOELECTRON3O(3),ITWOELECTRON3O(4)))

!==========================================================================
! loop over all nodes and broadcast their TWOELECTRO3O array
! to other nodes also broadcast the storage position (B2%CBMIN4, B2%CBMAX4)
! and the local k-point indices
!==========================================================================
cp: DO N=1,WHF%WDES%COMM%NCPU

! broadcast K2, K4 and all the other integers to all nodes
    IF (N==WHF%WDES%COMM%NODE_ME) THEN
       K2=K2_
       K4=K4_
       CBMIN4=B2%CBMIN4
       CBMAX4=B2%CBMAX4
    ENDIF
    CALL M_bcast_i_from(WHF%WDES%COMM, K2, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, K4, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, CBMIN4, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, CBMAX4, 1, n)

! if K2 is properly set distribute the matrix elements
    IF (K2>=1) THEN

    TWOELECTRON3O_LOCAL=0
    IF (N==WHF%WDES%COMM%NODE_ME) THEN
       TWOELECTRON3O_LOCAL(1:SIZE(TWOELECTRON3O,1),1:SIZE(TWOELECTRON3O,2),1:SIZE(TWOELECTRON3O,3),1:SIZE(TWOELECTRON3O,4))=TWOELECTRON3O
    ENDIF
#ifdef gammareal 
    CALL M_bcast_d_from(WHF%WDES%COMM, TWOELECTRON3O_LOCAL, SIZE(TWOELECTRON3O_LOCAL), N)
#else
    CALL M_bcast_z_from(WHF%WDES%COMM, TWOELECTRON3O_LOCAL, SIZE(TWOELECTRON3O_LOCAL), N)
#endif
!==========================================================================
! now distribute the matrix elements among nodes
!==========================================================================
    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)

    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

    ! loop over all columns of the global matrix
    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
        ! global column index NCV24
          NCV24=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2
        ! spin test
          IF (ISP2 /= BSE_INDEX%ISP(NCV24)) CYCLE
        ! k-point test, CYCLE if not identical
        ! IF (K2 /= 1+(NCV24-1)/((B1%VBMAX-B1%VBMIN+1)*(B1%CBMAX-B1%CBMIN+1))) CYCLE
          IF (K2 /= BSE_INDEX%NK(NCV24)) CYCLE

        ! test on NB2
        ! NB2_INTO_TOT=MOD(NCV24-1,B1%VBMAX-B1%VBMIN+1)+B1%VBMIN
          NB2_INTO_TOT=BSE_INDEX%N1(NCV24)
          N2=NB2_INTO_TOT-NPOS2+1
          IF (N2<1 .OR. N2 > NSTRIP2) CYCLE

        ! test on NB4_INTO_TOT
        ! NB4_INTO_TOT=MOD((NCV24-1)/(B1%VBMAX-B1%VBMIN+1),(B1%CBMAX-B1%CBMIN+1))+B1%CBMIN
          NB4_INTO_TOT=BSE_INDEX%N3(NCV24)
          IF (NB4_INTO_TOT<CBMIN4 .OR. NB4_INTO_TOT >CBMAX4) CYCLE

          NB4=NB4_INTO_TOT-CBMIN4+1

#ifdef check_index
          ! internal consistency test (could be removed)
        ! IF (NCV24/=NB2_INTO_TOT-B2%VBMIN+1 +(B2%VBMAX-B2%VBMIN+1)*(NB4_INTO_TOT-B2%CBMIN+(B2%CBMAX-B2%CBMIN+1)*(K2-1))) THEN
          IF (NCV24/=BSE_INDEX%INDEX(NB2_INTO_TOT-B2%VBMIN+1, NB4_INTO_TOT-B2%CBMIN+1, K2, ISP2-ISP_LOW+1)) THEN
             WRITE(*,*) 'internal error 1 in TWOELECTRON4O_STORE_SCALA:',NCV24, NB2_INTO_TOT-B2%VBMIN+1,  NB4_INTO_TOT-B2%CBMIN, K2, ISP2, BSE_INDEX%INDEX(NB2_INTO_TOT-B2%VBMIN+1, NB4_INTO_TOT-B2%CBMIN+1, K2, ISP2-ISP_LOW+1),BSE_INDEX%ISP(NCV24)
             STOP
          ENDIF
#endif
          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1

                NCV13=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
              ! spin test
                IF (ISP /= BSE_INDEX%ISP(NCV13)) CYCLE
              ! k-point test
              ! IF (K1 /= 1+(NCV13-1)/((B1%VBMAX-B1%VBMIN+1)*(B1%CBMAX-B1%CBMIN+1))) CYCLE
                IF (K1 /= BSE_INDEX%NK(NCV13)) CYCLE

              ! test on NB1
              ! NB1_INTO_TOT=MOD(NCV13-1,B1%VBMAX-B1%VBMIN+1)+B1%VBMIN
                NB1_INTO_TOT=BSE_INDEX%N1(NCV13)
                N1=NB1_INTO_TOT-NPOS1+1
                IF (N1<1 .OR. N1 > NSTRIP1) CYCLE

              ! test on NB3
              ! NB3_INTO_TOT=MOD((NCV13-1)/(B1%VBMAX-B1%VBMIN+1),(B1%CBMAX-B1%CBMIN+1))+B1%CBMIN
                NB3_INTO_TOT=BSE_INDEX%N3(NCV13)
                NB3=NB3_INTO_TOT-B1%CBMIN+1

#ifdef check_index
                ! internal consistency test (could be removed)
              ! IF (NCV13/=NB1_INTO_TOT-B1%VBMIN+1 +(B1%VBMAX-B1%VBMIN+1)*(NB3_INTO_TOT-B1%CBMIN+(B1%CBMAX-B1%CBMIN+1)*(K1-1))) THEN
                IF (NCV13/=BSE_INDEX%INDEX(NB1_INTO_TOT-B1%VBMIN+1, NB3_INTO_TOT-B1%CBMIN+1, K1, ISP -ISP_LOW+1)) THEN
                   WRITE(*,*) 'internal error 2 in TWOELECTRON4O_STORE_SCALA:',NCV13, NB1_INTO_TOT-B1%VBMIN+1,  NB3_INTO_TOT, K1, ISP, BSE_INDEX%INDEX(NB1_INTO_TOT-B1%VBMIN+1, NB3_INTO_TOT-B1%CBMIN+1, K1, ISP -ISP_LOW+1), BSE_INDEX%ISP(NCV13)
                   STOP
                ENDIF

                IF ( N1>SIZE(TWOELECTRON3O_LOCAL,1) .OR. NB3>SIZE(TWOELECTRON3O_LOCAL,2) & 
                 .OR.N2>SIZE(TWOELECTRON3O_LOCAL,3) .OR. NB4>SIZE(TWOELECTRON3O_LOCAL,4)) THEN
                   WRITE(*,*) 'internal error 3 in TWOELECTRON4O_STORE_SCALA:',N1,SIZE(TWOELECTRON3O_LOCAL,1),NB3,SIZE(TWOELECTRON3O_LOCAL,2), N2,SIZE(TWOELECTRON3O_LOCAL,3),NB4,SIZE(TWOELECTRON3O_LOCAL,4),CBMAX4
                   STOP
                ENDIF
                IF ( IROW+(JCOL-1)*DESCSTD(LLD_)>SIZE(AMAT) ) THEN
                   WRITE(*,*) 'internal error 4 in TWOELECTRON4O_STORE_SCALA:', IROW+(JCOL-1)*DESCSTD(LLD_),SIZE(AMAT)
                   STOP
                ENDIF
#endif
                ! finally if all test have a green light store the element in AMAT
                AMAT(IROW+(JCOL-1)*DESCSTD(LLD_))= &
                     TWOELECTRON3O_LOCAL( N1, NB3 , N2, NB4)*WHF%WDES%WTKPT(1)

                ! add diagonal element
                IF (NCV13== NCV24) THEN
                   AMAT(IROW+(JCOL-1)*DESCSTD(LLD_)) = AMAT(IROW+(JCOL-1)*DESCSTD(LLD_)) & 
                        -IDIAG*(WHF%CELTOT(NB1_INTO_TOT,K1, ISP)-WHF%CELTOT(NB3_INTO_TOT,K3, ISP))
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    !==========================================================================
    ! now fill in conjugated elements in lower triangle of matrix
    !==========================================================================

#define lower_triangle
#ifdef lower_triangle
    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
          ! global column index NCV24 species now the row in the conjugated matrix
          NCV24=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

          ! spin test
          IF (ISP /= BSE_INDEX%ISP(NCV24)) CYCLE
          ! k-point test
          IF (K1 /= BSE_INDEX%NK(NCV24)) CYCLE
          
          ! test on NB1
          NB1_INTO_TOT=BSE_INDEX%N1(NCV24)
          N1=NB1_INTO_TOT-NPOS1+1
          IF (N1<1 .OR. N1 > NSTRIP1) CYCLE
          
          ! test on NB3
          NB3_INTO_TOT=BSE_INDEX%N3(NCV24)

          NB3=NB3_INTO_TOT-B1%CBMIN+1

          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
              ! global row index NCV13 species the column position in the conjugated matrix
                NCV13=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2

                ! spin test
                IF (ISP2 /= BSE_INDEX%ISP(NCV13)) CYCLE
                ! k-point test, CYCLE if not identical
                IF (K2 /= BSE_INDEX%NK(NCV13)) CYCLE

                ! test on NB2
                NB2_INTO_TOT=BSE_INDEX%N1(NCV13)
                N2=NB2_INTO_TOT-NPOS2+1
                IF (N2<1 .OR. N2 > NSTRIP2) CYCLE
                
                ! test on NB4_INTO_TOT
                NB4_INTO_TOT=BSE_INDEX%N3(NCV13)
                IF (NB4_INTO_TOT<CBMIN4 .OR. NB4_INTO_TOT >CBMAX4) CYCLE

                NB4=NB4_INTO_TOT-CBMIN4+1

                IF (NCV13> NCV24) THEN
                   AMAT(IROW+(JCOL-1)*DESCSTD(LLD_))= &
                        GCONJG(TWOELECTRON3O_LOCAL(N1, NB3, N2, NB4 ))*WHF%WDES%WTKPT(1)
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO !! loop over column global matrix??
#endif

    ENDIF
    ENDDO cp

    DEALLOCATE(TWOELECTRON3O_LOCAL)
  END SUBROUTINE TWOELECTRON4O_STORE_SCALA
#endif

!****************** SUBROUTINE  TWOELECTRON4O_DIAG *******************
!
! this subroutine returns the sum of the diagonal elements of matrix A
! when this is not stored in a scaLAPACK aware way. Note that only the 
! spin component 1 is considered and the number of bands included goes
! up to NSTRIPV (which is typically 64; it is however set in 
! SET_BAND_PARAMETRS)
! 
!**********************************************************************

  SUBROUTINE TWOELECTRON4O_DIAG(AMAT, NCV, CSUM, IDIAG )

    USE constant
    IMPLICIT NONE
    INTEGER NCV

#ifdef single_prec_bse
    GDEFS :: AMAT(:,:)
#else
    GDEF :: AMAT(:,:)
#endif
    COMPLEX(q) :: CSUM
    GDEF :: IDIAG(:)
    INTEGER I,J

    CSUM=0.0_q
    DO I=1,NCV
       CSUM=CSUM+AMAT(I,I)
       IDIAG(I)=AMAT(I,I)
    ENDDO


  END SUBROUTINE TWOELECTRON4O_DIAG


!****************** SUBROUTINE  TWOELECTRON4O_DIAG_SCALA  *************
!
! this subroutine determines the sum of the diagonal elements 
! of a matrix A
! routine should be called using
!  CALL TWOELECTRON4O_DIAG_SCALA( AMAT, NCV, SUM, WHF%WDES%COMM)
! 
!**********************************************************************

#ifdef scaLAPACK
  SUBROUTINE TWOELECTRON4O_DIAG_SCALA( AMAT, NCV, CSUM, IDIAG, MY_COMM )

    USE constant
    IMPLICIT NONE

#ifdef single_prec_bse
    GDEFS :: AMAT(:), IDIAG(:)      ! matrix and its diagonal elements
#else
    GDEF :: AMAT(:), IDIAG(:)       ! matrix and its diagonal elements
#endif
    INTEGER :: NCV            ! dimension of global matrix
    COMPLEX(q) :: CSUM        ! sum of diagonal components
    TYPE(communic) :: MY_COMM ! communicator to globally sum the diagonal components

  ! local variables
    INTEGER NCV13, NCV24
  ! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP, NQ
    INTEGER IROW, JCOL
    INTEGER I1, I2, J1, J2, J1RES, I1RES

    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW, NPCOL, MYROW, MYCOL)

    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

    ! loop over all columns of the global matrix
    JCOL=0
    ! loop over block along column index
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       ! loop over elements in this block
       DO J2=1,J1RES
          JCOL=JCOL+1

          ! original global column index (if matrix where stored sequentially)
          NCV24=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

           IROW=0
         ! loop over blocks along row index
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
            ! loop over elements in this block
             DO I2=1,I1RES
                IROW=IROW+1
                ! original global row index (if matrix where stored sequentially)
                NCV13=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2

                IF (NCV13 == NCV24) THEN
                   CSUM= CSUM + AMAT(IROW+(JCOL-1)*DESCSTD(LLD_))
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    CALLMPI( M_sum_z(MY_COMM, CSUM, 1))

  END SUBROUTINE TWOELECTRON4O_DIAG_SCALA


#endif


#ifdef use_bse_te
!****************** SUBROUTINE TW4O_STORE_STRIP ***********************
!
! store matrix elements as stripes on target nodes
! certainly not the most clever routine since data are sent
! everywhere and than the target nodes pick their required data
!
!**********************************************************************

  SUBROUTINE TW4O_STORE_STRIP(WHF, ISP, ISP2, ISP_LOW, &
       K1, NPOS1, NSTRIP1, K2_, NPOS2, NSTRIP2, K3, K4_, &
       TWOELECTRON3O, B1, B2, BSE_INDEX )

    USE constant
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    INTEGER ISP, ISP2, ISP_LOW
    INTEGER K1, K2_, K3, K4_, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF ::  TWOELECTRON3O(:,:,:,:)
! actual dim TWOELECTRON3O( NSTRIPV, NGLB, NSTRIPV, NGLB4))
    TYPE(banddesc) :: B1,B2         ! current descriptor for the spin loops
    TYPE(bse_matrix_index) :: BSE_INDEX
! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NB3, NB4
    INTEGER NCV13, NCV24
    INTEGER NCV13_LOCAL, NCV24_LOCAL
    INTEGER K2, K4, CBMIN4, CBMAX4
    INTEGER ITWOELECTRON3O(4)
    INTEGER N

    GDEF, ALLOCATABLE ::  TWOELECTRON3O_LOCAL(:,:,:,:)


! allocate work array into which data is subsequently merged from all nodes
    ITWOELECTRON3O=SHAPE(TWOELECTRON3O)
    CALL M_max_i(WHF%WDES%COMM, ITWOELECTRON3O, 4 )

    ALLOCATE(TWOELECTRON3O_LOCAL(ITWOELECTRON3O(1),ITWOELECTRON3O(2), & 
                                 ITWOELECTRON3O(3),ITWOELECTRON3O(4)))
!==========================================================================
! loop over all nodes and broadcast their TWOELECTRO3O array
! to other nodes also broadcast the storage position (B2%CBMIN4, B2%CBMAX4)
! and the local k-point indices
!==========================================================================
cp: DO N=1,WHF%WDES%COMM%NCPU

! broadcast K2, K4 and all the other integers to all nodes
    IF (N==WHF%WDES%COMM%NODE_ME) THEN
       K2=K2_
       K4=K4_
       CBMIN4=B2%CBMIN4
       CBMAX4=B2%CBMAX4
    ENDIF
    CALL M_bcast_i_from(WHF%WDES%COMM, K2, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, K4, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, CBMIN4, 1, n)
    CALL M_bcast_i_from(WHF%WDES%COMM, CBMAX4, 1, n)

! if K2 is properly set distribute the matrix elements
    IF (K2>=1) THEN

    TWOELECTRON3O_LOCAL=0
    IF (N==WHF%WDES%COMM%NODE_ME) THEN
       TWOELECTRON3O_LOCAL(1:SIZE(TWOELECTRON3O,1),1:SIZE(TWOELECTRON3O,2),1:SIZE(TWOELECTRON3O,3),1:SIZE(TWOELECTRON3O,4))=TWOELECTRON3O
    ENDIF
#ifdef gammareal 
    CALL M_bcast_d_from(WHF%WDES%COMM, TWOELECTRON3O_LOCAL, SIZE(TWOELECTRON3O_LOCAL), n)
#else
    CALL M_bcast_z_from(WHF%WDES%COMM, TWOELECTRON3O_LOCAL, SIZE(TWOELECTRON3O_LOCAL), n)
#endif
!==========================================================================
! now store the matrix elements on the local nodes
! simply walk through all received data and decide whether they
! are stored locally (stupid but fairly simple)
!==========================================================================
    DO N1=1,NSTRIP1
       DO N2=1,NSTRIP2
          NB1_INTO_TOT=NPOS1-1+N1
          NB2_INTO_TOT=NPOS2-1+N2
          DO N4=1,CBMAX4-CBMIN4+1
             NB4_INTO_TOT=CBMIN4-1+N4
             NB4         =N4
#ifdef check_index
             IF (NB4 > SIZE(TWOELECTRON3O_LOCAL,4)) THEN
                WRITE(*,*)'internal error in TW4O_STORE_STRIP: out of bounds 4 ',NB4
                STOP
             ENDIF
#endif
             NCV24=BSE_INDEX%INDEX(NB2_INTO_TOT-B2%VBMIN+1, NB4_INTO_TOT-B2%CBMIN+1, K2, ISP2-ISP_LOW+1)
             IF (NCV24==0) CYCLE ! cycle if this pair is not included
             NCV24_LOCAL=NCV24-FIRST_ROW_INDEX+1
#ifdef check_index
                ! check whether NCV24 is smaller or equal than SIZE(BSEMATRIX,2)
                IF ( NCV24 > SIZE(BSEMATRIX,2)) THEN
                   WRITE(*,*) 'internal error in TW4O_STORE_STRIP: NCV exceeds BSEMATRIX size-boundaries',NCV13,NCV24,SIZE(BSEMATRIX,1),SIZE(BSEMATRIX,2) 
                   STOP
                ENDIF
#endif
             DO N3=1,B1%CBMAX-B1%CBMIN+1
                NB3_INTO_TOT=B1%CBMIN-1+N3
                NB3         =N3
#ifdef check_index
                IF ( NB3> SIZE(TWOELECTRON3O_LOCAL,2)) THEN
                   WRITE(*,*)'internal error in TW4O_STORE_STRIP: out of bounds 2 ',NB3_INTO_TOT,NB3
                   STOP
                ENDIF
#endif
                ! build index NCV13 and NCV24
                NCV13=BSE_INDEX%INDEX(NB1_INTO_TOT-B1%VBMIN+1, NB3_INTO_TOT-B1%CBMIN+1, K1, ISP -ISP_LOW+1)
                IF (NCV13==0) CYCLE ! cycle if this pair is not included
                NCV13_LOCAL=NCV13-FIRST_ROW_INDEX+1
#ifdef check_index
                ! check whether NCV24 or  NCV13 is smaller or equal than SIZE(BSEMATRIX,2)
                IF ( NCV13 > SIZE(BSEMATRIX,2) ) THEN
                   WRITE(*,*) 'internal error in TW4O_STORE_STRIP: NCV exceeds BSEMATRIX size-boundaries',NCV13,NCV24,SIZE(BSEMATRIX,1),SIZE(BSEMATRIX,2) 
                   STOP
                ENDIF
#endif
                IF ( NCV24_LOCAL>0 .AND. NCV24_LOCAL<=SIZE(BSEMATRIX,1)) THEN
                   BSEMATRIX(NCV24_LOCAL,NCV13)=TWOELECTRON3O_LOCAL( N1, NB3 , N2, NB4)*WHF%WDES%WTKPT(1)
                   ! diagonal components subtract eigenvalue difference
                   IF (NCV13== NCV24) THEN
                      BSEMATRIX( NCV24_LOCAL, NCV13 ) = BSEMATRIX( NCV24_LOCAL, NCV13 )-(WHF%CELTOT(NB1_INTO_TOT,K1, ISP)-WHF%CELTOT(NB3_INTO_TOT,K3, ISP))
                   ENDIF
                ENDIF
                IF ( NCV13_LOCAL>0 .AND. NCV13_LOCAL<=SIZE(BSEMATRIX,1)) THEN
                   BSEMATRIX(NCV13_LOCAL,NCV24)=CONJG(TWOELECTRON3O_LOCAL( N1, NB3 , N2, NB4)*WHF%WDES%WTKPT(1))
                   ! diagonal components subtract eigenvalue difference
                   IF (NCV13== NCV24) THEN
                      BSEMATRIX( NCV13_LOCAL, NCV24 ) = BSEMATRIX( NCV13_LOCAL, NCV24 )-(WHF%CELTOT(NB1_INTO_TOT,K1, ISP)-WHF%CELTOT(NB3_INTO_TOT,K3, ISP))
! store matrix in transposed form, this allows to change the DGEMM calls in bse_te.F GGEMM('N', -> GGEMM('T',
! also upon allocation of matrix the dimensions must be exchanged!
!                      BSEMATRIX( NCV24, NCV13_LOCAL ) = BSEMATRIX( NCV24, NCV13_LOCAL )-(WHF%CELTOT(NB1_INTO_TOT,K1, ISP)-WHF%CELTOT(NB3_INTO_TOT,K3, ISP))
                   ENDIF
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ENDIF
  ENDDO cp
  DEALLOCATE(TWOELECTRON3O_LOCAL)

  END SUBROUTINE TW4O_STORE_STRIP
#endif

!**********************************************************************
!
! calculate the BSE oscillator strength
! AMAT stores the eigenvectors of the BSE Hamiltonian
! the BSE Hamiltonian is calculated as
! Hamiltonian(I, J) = < c_k3,n3 v'_k2,n2 | v+W | v_k1,n1  c'_k4,n4 >
! with I = (k1,n1, k3, n3)   J= (k2,n2, k4, n4)
!
! after diagonalization AMAT stores U(I, lambda) with
! Hamiltonian(I, J)= U Lambda U+ =
!        \sum_lambda U(I, lambda) lambda U(J, lambda)*
! (this has been carefully checked by dumping matrix elements)
!
! the oscillator strength is given by
! \sum_lambda \sum_I <v_k1,n1|O|c_k3,n3> U(I, lambda)  1/ (w- lambda )
!             [sum_J U( J, lambda) <v_k2,n2|O|c_k4,n4>]*
! where O is an arbitrary operator
! To obtain the polarizabilty at a wavevector q=k3-k1 
!  < c_k3,n3| e^i(k3-k1 ) | v_k1,n1 >
!
!**********************************************************************

  SUBROUTINE CALCULATE_BSE_OSZI_STRENGTH(WHF, NQPOINT, KPOINT_BSE, LATT_CUR, ISP_LOW, ISP_HIGH,  &
       SHIFT, BD, BSE_INDEX, R, AMAT, ANTIRES, NOMEGA_DIM, OMEGAMAX, IU6)

    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE mymath
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    INTEGER :: NQPOINT       ! q-point considered
    INTEGER :: KPOINT_BSE(3) ! G-point considered
    TYPE (latt) LATT_CUR
    INTEGER ISP_LOW, ISP_HIGH
    REAL(q) SHIFT   ! complex shift for broadening
    TYPE(banddesc) :: BD(:)
    TYPE(bse_matrix_index) :: BSE_INDEX
    REAL(q) :: R(:)
#ifdef single_prec_bse
    GDEFS :: AMAT(:,:)
#else
    GDEF  :: AMAT(:,:)
#endif
    INTEGER :: ANTIRES
    INTEGER :: NOMEGA_DIM
    REAL(q) :: OMEGAMAX
    INTEGER :: IU6
    ! local 
    INTEGER :: NCV13
    INTEGER K1, K3
    INTEGER N1, N3, ISP
    INTEGER NB1_INTO_TOT, NB3_INTO_TOT
    COMPLEX(q) :: WEIGHT
    INTEGER I, J, NK1_IN_KPOINTS_FULL_ORIG, INFO
    INTEGER     :: NCV
    INTEGER     :: LAMBDA, NOMEGA
    GDEF        :: CDER_BETWEEN_STATE(3)
    COMPLEX(q)  :: CTRANS(3)
    COMPLEX(q)  :: CTRANS_SQUARE(3,3)
    COMPLEX(q)  :: CTRANS_SQUARE_ANTIRES(3,3)
    REAL(q)     :: DKX, DKY, DKZ
    COMPLEX(q)  :: EPS(3,3,NOMEGA_DIM)
    REAL(q)     :: OMEGA(NOMEGA_DIM),DOMEGA
    REAL(q)     :: EPS_REAL(NOMEGA_DIM, 3,3), EPS_IMAG(NOMEGA_DIM, 3, 3)
    REAL(q)     :: R_AND_INTENSITY(2,SIZE(R))
    REAL(q)     :: RSPIN

!  three cases need to be considered
!  non-collinear RSPIN=1.0
    IF (WHF%WDES%LNONCOLLINEAR) THEN
       RSPIN=1.0_q
    ELSE
!  collinear non spin polarized: RSPIN=2.0
!  collinear spin polarized:     RSPIN=1.0 for  ISP_LOW=1 and ISP_HIGH=2
!  I believe the case ISP_HIGH=ISP_LOW for collinear spin polarized makes little sense
!  but it is not used right now anyway
       RSPIN=2._q/(ISP_HIGH-ISP_LOW+1)
    ENDIF

    NCV=SIZE(AMAT,1)

    DOMEGA=R(NCV)/(NOMEGA_DIM-1)
    IF (OMEGAMAX>0) THEN
       DOMEGA=OMEGAMAX/(NOMEGA_DIM-1)
    ENDIF

    EPS = 0
    lambda_bse: DO LAMBDA=1,NCV
       CTRANS=0

       DO ISP=ISP_LOW,ISP_HIGH
       DO K1=1,WHF%WDES%NKPTS
          IF (WHF%WDES%WTKPT(K1)==0) CYCLE
          K3 =KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,NQPOINT),KPOINTS_FULL)

          ! determined the index of this k-point in the original full k-point grid
          NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)

          DO N3=1,BD(ISP)%CBMAX-BD(ISP)%CBMIN+1
             NB3_INTO_TOT=N3+BD(ISP)%CBMIN-1
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE
             DO N1=1,BD(ISP)%VBMAX-BD(ISP)%VBMIN+1
                NB1_INTO_TOT=N1+BD(ISP)%VBMIN-1
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

                WEIGHT=(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)-WHF%FERTOT(NB3_INTO_TOT, K3, ISP))

                ! CDER_BETWEEN_STATE =  <v_k1,n1|- i d/dq_j c_k1+q,n3> =  - <v_k1,n1| r | c_k1,n3>  for q->0
                ! or   <v_k1,n1| e-iqr  |c_k1+q,n3>  at finite q
                ! right now we have to used  <c_k1+q,n3| e iqr  |v_k1,n1> which is very odd
                ! see ONEELECTRON_TRANS in local_field.F
                CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB3_INTO_TOT)
                ! sort  fort.78 | uniq >sort.78.sort
                !WRITE(78,'(4I4,2F14.7)')  K1, K3, NB1_INTO_TOT, NB3_INTO_TOT, CDER_BETWEEN_STATE(1)

              !  NCV13=NB1_INTO_TOT-BD(ISP)%VBMIN+1 +(BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)*(NB3_INTO_TOT-BD(ISP)%CBMIN+(BD(ISP)%CBMAX-BD(ISP)%CBMIN+1)*(K1-1))

                NCV13=BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)
                IF (NCV13==0) CYCLE ! cycle if this pair is not included                
                CTRANS=CTRANS+CDER_BETWEEN_STATE*AMAT(NCV13,LAMBDA)*WEIGHT
             ENDDO  ! enddo for N3-loop
          ENDDO     ! enddo for N1-loop
       ENDDO  ! enddo K1 loop
       ENDDO  ! enddo ISP loop
       DO I=1,3
          DO J=1,3
             CTRANS_SQUARE(I,J)=CTRANS(I)*CONJG(CTRANS(J))
          ENDDO
       ENDDO

       R_AND_INTENSITY(1,LAMBDA)= R(LAMBDA)
       R_AND_INTENSITY(2,LAMBDA)= (REAL(CTRANS_SQUARE(1,1),q)+REAL(CTRANS_SQUARE(2,2),q)+REAL(CTRANS_SQUARE(3,3),q))*1000

       ! add to reducible polarizability X(w)
       IF (ANTIRES<0) THEN
          ! only antiresonant part (helpful for debugging)
          DO NOMEGA=1,NOMEGA_DIM
             EPS(:,:,NOMEGA)=EPS(:,:,NOMEGA)+RSPIN*WHF%WDES%WTKPT(1)*CTRANS_SQUARE* & 
                  (1/((NOMEGA-1)*DOMEGA-R(LAMBDA)+CMPLX(0,SHIFT,q)))
          ENDDO
       ELSE
          DO NOMEGA=1,NOMEGA_DIM
             EPS(:,:,NOMEGA)=EPS(:,:,NOMEGA)+RSPIN*WHF%WDES%WTKPT(1)*CTRANS_SQUARE* & 
                  (1/( (NOMEGA-1)*DOMEGA-R(LAMBDA)+CMPLX(0,SHIFT,q)) &
                  +1/(-(NOMEGA-1)*DOMEGA-R(LAMBDA)-CMPLX(0,SHIFT,q)))
          ENDDO
       ENDIF
    ENDDO lambda_bse

    CALL XML_VECARRAY("opticaltransitions")
    CALL XML_ARRAY_REAL(R_AND_INTENSITY,"(F10.3,' ')")
    CALL XML_CLOSE_TAG("varray")

    ! multiply by Coloumb kernel 4 pi e^2 / q^2 (in a.u.)
    ! to obtain v X^red 
    ! for q->0 the 1/q^2 is accounted for in the CDER_BETWEEN_STATES 
    EPS=EPS*EDEPS/LATT_CUR%OMEGA
    
    IF (ABS(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))>1E-6_q .OR. & 
        ABS(WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))>1E-6_q .OR. & 
        ABS(WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))>1E-6_q) THEN       
       ! divide by q^2 at finite q
       DKX=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(1,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(1,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(1,3)
       DKY=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(2,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(2,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(2,3)
       DKZ=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(3,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(3,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(3,3)
       
       EPS=EPS/((DKX**2+DKY**2+DKZ**2)*TPI**2)
       ! eps_mac = ( 1 + v X^red(w))^-1
       DO NOMEGA=1,NOMEGA_DIM
          EPS(1,1,NOMEGA)=(1+EPS(1,1,NOMEGA))
          EPS(2,2,NOMEGA)=(1+EPS(2,2,NOMEGA))
          EPS(3,3,NOMEGA)=(1+EPS(3,3,NOMEGA))
          CALL INVERSE_Z(EPS(1,1,NOMEGA),3)
       ENDDO
    ELSE
       IF (LHEAD_DIRECT) THEN
          WRITE(*,*) 'doing head direct'
          DO NOMEGA=1,NOMEGA_DIM
             EPS(1,1,NOMEGA)=1/(1+EPS(1,1,NOMEGA))
             EPS(2,2,NOMEGA)=EPS(1,1,NOMEGA)
             EPS(3,3,NOMEGA)=EPS(1,1,NOMEGA)
          ENDDO
       ELSE
          ! at q=0, we have calculated EPS= v X'^red, where X'^red is the
          ! the reducible polarizability with the amputed Coulomb kernel
          ! (component G->0 removed)
          ! epsilon_mac = 1 - v X'^red (Onida et al. Rev. Mod. Phys. 74, 601)
          DO NOMEGA=1,NOMEGA_DIM
             EPS(1,1,NOMEGA)=1-EPS(1,1,NOMEGA)
             EPS(2,2,NOMEGA)=1-EPS(2,2,NOMEGA)
             EPS(3,3,NOMEGA)=1-EPS(3,3,NOMEGA)
          ENDDO
       ENDIF
    ENDIF

    DO NOMEGA=1,NOMEGA_DIM
       OMEGA(NOMEGA)=(NOMEGA-1)*DOMEGA
       EPS_REAL(NOMEGA,:,:)=REAL(EPS(:,:,NOMEGA),q)
       EPS_IMAG(NOMEGA,:,:)=AIMAG(EPS(:,:,NOMEGA))
    ENDDO

    CALL XML_EPSILON_W(DOMEGA, EPS_REAL, EPS_IMAG, NOMEGA_DIM )

  END SUBROUTINE CALCULATE_BSE_OSZI_STRENGTH

#ifdef scaLAPACK
!**********************************************************************
!
! calculate the BSE oszillator strength
! scaLAPACK aware version
! see comments above
!
!**********************************************************************


  SUBROUTINE CALCULATE_BSE_OSZI_SCALA(WHF, NQPOINT, KPOINT_BSE, LATT_CUR, ISP_LOW, ISP_HIGH,  &
       SHIFT, BD, BSE_INDEX, R, AMAT, NCV, ANTIRES, NOMEGA_DIM, OMEGAMAX, IU6)

    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE mymath
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    INTEGER :: NQPOINT        ! q-point considered
    INTEGER :: KPOINT_BSE(3)  ! G-vector considered
    TYPE (latt) LATT_CUR
    INTEGER ISP_LOW, ISP_HIGH
    REAL(q) SHIFT   ! complex shift for broadening
    TYPE(banddesc) :: BD(:)
    TYPE(bse_matrix_index) :: BSE_INDEX
    REAL(q) :: R(:)
#ifdef single_prec_bse
    GDEFS :: AMAT(:)
#else
    GDEF  :: AMAT(:)
#endif
    INTEGER :: NCV
    INTEGER :: ANTIRES
    INTEGER :: NOMEGA_DIM
    REAL(q) :: OMEGAMAX
    INTEGER :: IU6
 ! local 
    INTEGER :: NCV13
    INTEGER K1, K1_LOOKED_UP, K3, ISP
    INTEGER N1, N3
    INTEGER NB1_INTO_TOT, NB3_INTO_TOT
    COMPLEX(q) :: WEIGHT
    INTEGER I, J, NK1_IN_KPOINTS_FULL_ORIG
    INTEGER     :: LAMBDA, NOMEGA
    GDEF        :: CDER_BETWEEN_STATE(3)
    COMPLEX(q)  :: CTRANS(3)
    COMPLEX(q)  :: CTRANS_SQUARE(3,3)
    REAL(q)     :: DKX, DKY, DKZ
    COMPLEX(q)  :: EPS(3,3,NOMEGA_DIM)
    REAL(q)     :: EPS_REAL(NOMEGA_DIM, 3,3), EPS_IMAG(NOMEGA_DIM, 3, 3)
    REAL(q)     :: OMEGA(NOMEGA_DIM), DOMEGA
    REAL(q)     :: R_AND_INTENSITY(2,SIZE(R))
! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP,NQ
    INTEGER I1RES, J1RES, IROW, JCOL
    INTEGER I1, I2, J1, J2, NLAMBDA, INFO
    REAL(q)     :: RSPIN

!  three cases need to be considered
!  non-collinear RSPIN=1.0
    IF (WHF%WDES%LNONCOLLINEAR) THEN
       RSPIN=1.0_q
    ELSE
!  collinear non spin polarized: RSPIN=2.0
!  collinear spin polarized:     RSPIN=1.0 for  ISP_LOW=1 and ISP_HIGH=2
!  I believe the case ISP_HIGH=ISP_LOW for collinear spin polarized makes little sense
!  but it is not used right now anyway
       RSPIN=2._q/(ISP_HIGH-ISP_LOW+1)
    ENDIF

    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)
    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

    EPS = 0
    K1_LOOKED_UP=-1
    NLAMBDA=0

    DOMEGA=R(NCV)/(NOMEGA_DIM-1)
    IF (OMEGAMAX>0) THEN
       DOMEGA=OMEGAMAX/(NOMEGA_DIM-1)
    ENDIF

!   loop over column indices (eigenvectors)
    DO LAMBDA=1, NCV
       ! local storage index in block (J1) and local index of block (J2)
       J1=   ((LAMBDA-DESCSTD(NB_)*MYCOL-1)/(NPCOL*DESCSTD(NB_)))*DESCSTD(NB_)+1
       J2=MOD((LAMBDA-DESCSTD(NB_)*MYCOL-1) ,NPCOL*DESCSTD(NB_))+1

       JCOL=J1+J2-1
       IF (J2 <= 0 .OR. J2 > DESCSTD(NB_)) THEN
          JCOL  =-1
       ELSE
          NLAMBDA=NLAMBDA+1
          IF (LAMBDA /= DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2) THEN
             WRITE(*,*) 'internal error in CALCULATE_BSE_OSZI_SCALA:', LAMBDA, DESCSTD(NB_)*MYCOL, NPCOL*(J1-1), J2
             STOP
          ENDIF
       ENDIF

       IF (JCOL==-1) THEN
          ! no data on local node for eigenvector/eigenvalue pair LAMBDA_DONE
          ! simply clear CTRANS
          CTRANS=0
       ELSE
          ! local data then calculate contribution to CTRANS
          CTRANS=0
          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                
                NCV13=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
                ISP= BSE_INDEX%ISP(NCV13)
              ! K1 = 1+(NCV13-1)/((BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)*(BD(ISP)%CBMAX-BD(ISP)%CBMIN+1))
                K1 = BSE_INDEX%NK(NCV13)
                IF (WHF%WDES%WTKPT(K1)==0) CYCLE
                
                IF (K1/= K1_LOOKED_UP) THEN
                   NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)
                   K1_LOOKED_UP=K1
                   K3 =KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,NQPOINT),KPOINTS_FULL)
                ENDIF
                      
              ! NB1_INTO_TOT=MOD(NCV13-1,BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)+BD(ISP)%VBMIN
                NB1_INTO_TOT=BSE_INDEX%N1(NCV13)
                N1=NB1_INTO_TOT-BD(ISP)%VBMIN+1
                
              ! NB3_INTO_TOT=MOD((NCV13-1)/(BD(ISP)%VBMAX-BD(ISP)%VBMIN+1),(BD(ISP)%CBMAX-BD(ISP)%CBMIN+1))+BD(ISP)%CBMIN
                NB3_INTO_TOT=BSE_INDEX%N3(NCV13)
                N3=NB3_INTO_TOT-BD(ISP)%CBMIN+1
                      
              !  IF (NCV13/=NB1_INTO_TOT-BD(ISP)%VBMIN+1 +(BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)*(NB3_INTO_TOT-BD(ISP)%CBMIN+(BD(ISP)%CBMAX-BD(ISP)%CBMIN+1)*(K1-1))) THEN
                IF (NCV13/=BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)) THEN
                   WRITE(*,*) 'internal error 1 in CALCULATE_BSE_OSZI_SCALA:',NCV13, NB1_INTO_TOT-BD(ISP)%VBMIN+1,  NB3_INTO_TOT, K1, ISP, BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)
                   STOP
                ENDIF
                IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))  CYCLE
                
                WEIGHT=(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)-WHF%FERTOT(NB3_INTO_TOT, K3, ISP))

                ! CDER_BETWEEN_STATE =  <v_k1,n1|- i d/dq_j c_k1+q,n3> =  - <v_k1,n1| r | c_k1,n3>  for q->0
                ! or   <v_k1,n1| e-iqr  |c_k1+q,n3>  at finite q
                CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB3_INTO_TOT)
                CTRANS=CTRANS+CDER_BETWEEN_STATE*AMAT(IROW+(JCOL-1)*DESCSTD(LLD_))*WEIGHT
             ENDDO
          ENDDO
       ENDIF
       
       ! now sum CTRANS over all nodes for the current LAMBDA
       CALLMPI( M_sum_z(WHF%WDES%COMM, CTRANS, 3))
       
       DO I=1,3
          DO J=1,3
             CTRANS_SQUARE(I,J)=CTRANS(I)*CONJG(CTRANS(J))
          ENDDO
       ENDDO

       R_AND_INTENSITY(1,LAMBDA)= R(LAMBDA)
       R_AND_INTENSITY(2,LAMBDA)= (REAL(CTRANS_SQUARE(1,1),q)+REAL(CTRANS_SQUARE(2,2),q)+REAL(CTRANS_SQUARE(3,3),q))*1000

       ! add to reducible polarizability X(w)
       IF (ANTIRES<0) THEN
          ! only antiresonant part (helpful for debugging)
          DO NOMEGA=1,NOMEGA_DIM
             EPS(:,:,NOMEGA)=EPS(:,:,NOMEGA)+RSPIN*WHF%WDES%WTKPT(1)*CTRANS_SQUARE* & 
                  (1/((NOMEGA-1)*DOMEGA-R(LAMBDA)+CMPLX(0,SHIFT,q)))
          ENDDO
       ELSE
          DO NOMEGA=1,NOMEGA_DIM
             EPS(:,:,NOMEGA)=EPS(:,:,NOMEGA)+RSPIN*WHF%WDES%WTKPT(1)*CTRANS_SQUARE* & 
                  (1/( (NOMEGA-1)*DOMEGA-R(LAMBDA)+CMPLX(0,SHIFT,q)) &
                  +1/(-(NOMEGA-1)*DOMEGA-R(LAMBDA)-CMPLX(0,SHIFT,q)))
          ENDDO
       ENDIF
    ENDDO

    CALL XML_VECARRAY("opticaltransitions")
    CALL XML_ARRAY_REAL(R_AND_INTENSITY,"(F10.3,' ')")
    CALL XML_CLOSE_TAG("varray")

    ! multiply by Coloumb kernel 4 pi e^2 / q^2 (in a.u.)
    ! to obtain v X^red 
    ! for q->0 the 1/q^2 is accounted for in the CDER_BETWEEN_STATES at q=0
    EPS=EPS*EDEPS/LATT_CUR%OMEGA

    IF (ABS(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))>1E-6_q .OR. & 
        ABS(WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))>1E-6_q .OR. & 
        ABS(WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))>1E-6_q) THEN       
       ! divide by q^2 at finite q
       DKX=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(1,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(1,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(1,3)
       DKY=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(2,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(2,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(2,3)
       DKZ=(WHF%WDES%VKPT(1,NQPOINT)+KPOINT_BSE(1))*LATT_CUR%B(3,1)+ &
           (WHF%WDES%VKPT(2,NQPOINT)+KPOINT_BSE(2))*LATT_CUR%B(3,2)+ &
           (WHF%WDES%VKPT(3,NQPOINT)+KPOINT_BSE(3))*LATT_CUR%B(3,3)
       
       EPS=EPS/((DKX**2+DKY**2+DKZ**2)*TPI**2)
       ! eps_mac = ( 1 + v X^red(w))^-1
       DO NOMEGA=1,NOMEGA_DIM
          EPS(1,1,NOMEGA)=(1+EPS(1,1,NOMEGA))
          EPS(2,2,NOMEGA)=(1+EPS(2,2,NOMEGA))
          EPS(3,3,NOMEGA)=(1+EPS(3,3,NOMEGA))
          CALL INVERSE_Z(EPS(1,1,NOMEGA),3)
       ENDDO
    ELSE
       IF (LHEAD_DIRECT) THEN
          WRITE(*,*) 'doing head direct'
          DO NOMEGA=1,NOMEGA_DIM
             EPS(1,1,NOMEGA)=1/(1+EPS(1,1,NOMEGA))
             EPS(2,2,NOMEGA)=EPS(1,1,NOMEGA)
             EPS(3,3,NOMEGA)=EPS(1,1,NOMEGA)
          ENDDO
       ELSE
          ! at q=0, we have calculated EPS= v X'^red, where X'^red is the
          ! the reducible polarizability with the amputed Coulomb kernel
          ! (component G->0 removed)
          ! epsilon_mac = 1 - v X'^red (Onida et al. Rev. Mod. Phys. 74, 601)
          DO NOMEGA=1,NOMEGA_DIM
             EPS(1,1,NOMEGA)=1-EPS(1,1,NOMEGA)
             EPS(2,2,NOMEGA)=1-EPS(2,2,NOMEGA)
             EPS(3,3,NOMEGA)=1-EPS(3,3,NOMEGA)
          ENDDO
       ENDIF
    ENDIF

    DO NOMEGA=1,NOMEGA_DIM
       OMEGA(NOMEGA)=(NOMEGA-1)*DOMEGA
       EPS_REAL(NOMEGA,:,:)=REAL(EPS(:,:,NOMEGA),q)
       EPS_IMAG(NOMEGA,:,:)=AIMAG(EPS(:,:,NOMEGA))
    ENDDO

    CALL XML_EPSILON_W(DOMEGA, EPS_REAL, EPS_IMAG, NOMEGA_DIM )

  END SUBROUTINE CALCULATE_BSE_OSZI_SCALA
#endif


#ifdef scaLAPACK
!**********************************************************************
!
! Write out BSE eigenfunctions and eigenvalues
! scaLAPACK aware version
! Based on CALCULATE_BSE_OSZI_SCALA
! 			M.Bokdam April  2016
!**********************************************************************

  SUBROUTINE WRITE_BSE_SCALA(WHF, LATT_CUR, ISP_LOW, ISP_HIGH,  &
       BD, BSE_INDEX, R, AMAT, NCV, NBSEEIG, IU6)

    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    TYPE (latt) LATT_CUR
    INTEGER ISP_LOW, ISP_HIGH
    TYPE(banddesc) :: BD(:)
    TYPE(bse_matrix_index) :: BSE_INDEX
    REAL(q) :: R(:)
#ifdef single_prec_bse
    GDEFS :: AMAT(:)
#else
    GDEF  :: AMAT(:)
#endif
    INTEGER :: NCV
    INTEGER :: ANTIRES
    INTEGER :: IU6
 ! local 
    INTEGER :: NCV13
    INTEGER K1, ISP
    INTEGER N1, N3
    INTEGER K2,K3,K4,NPOS1,NPOS2 !MB
    INTEGER NB1_INTO_TOT, NB3_INTO_TOT
    INTEGER I, J
    INTEGER     :: LAMBDA
!MB
    COMPLEX(q), ALLOCATABLE :: EIGVECT(:)
    INTEGER :: NBSEEIG

! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP,NQ
    INTEGER I1RES, J1RES, IROW, JCOL
    INTEGER I1, I2, J1, J2

    IF (NBSEEIG.GE.NCV) THEN
       IF (IU6>=0) THEN
          WRITE(*,*) "Warning: NBSEEIG cannot be larger than the number of BSE eigenstates."
          WRITE(*,*) "NBSEEIG is set to the total number of BSE eigenstates."
       ENDIF
       NBSEEIG=NCV
    ENDIF

!Large NBSEEIG value can overrun the memory of a local node.    
    ALLOCATE(EIGVECT(NBSEEIG*NCV)) 
    EIGVECT=0.0_q

    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)
    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

!   loop over column indices (eigenvectors)
    DO LAMBDA=1, NBSEEIG
       ! local storage index in block (J1) and local index of block (J2)
       J1=   ((LAMBDA-DESCSTD(NB_)*MYCOL-1)/(NPCOL*DESCSTD(NB_)))*DESCSTD(NB_)+1
       J2=MOD((LAMBDA-DESCSTD(NB_)*MYCOL-1) ,NPCOL*DESCSTD(NB_))+1

       JCOL=J1+J2-1
       IF (J2 <= 0 .OR. J2 > DESCSTD(NB_)) THEN
          JCOL  =-1
       ELSE
          IF (LAMBDA /= DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2) THEN
             WRITE(*,*) 'internal error in WRITE_BSE_SCALA:', LAMBDA, DESCSTD(NB_)*MYCOL, NPCOL*(J1-1), J2
             STOP
          ENDIF
       ENDIF

       IF (JCOL==-1) THEN
          ! no data on local node for eigenvector/eigenvalue pair LAMBDA_DONE
       ELSE
          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                
                NCV13=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
                ISP= BSE_INDEX%ISP(NCV13)
                K1 = BSE_INDEX%NK(NCV13)

                IF (WHF%WDES%WTKPT(K1)==0) CYCLE
                
                NB1_INTO_TOT=BSE_INDEX%N1(NCV13)
                N1=NB1_INTO_TOT-BD(ISP)%VBMIN+1
                
                NB3_INTO_TOT=BSE_INDEX%N3(NCV13)
                N3=NB3_INTO_TOT-BD(ISP)%CBMIN+1
                      
                IF (NCV13/=BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)) THEN
                   IF (IU6>=0) WRITE(*,*) 'internal error in WRITE_BSE_SCALA:',NCV13, NB1_INTO_TOT-BD(ISP)%VBMIN+1,  NB3_INTO_TOT, K1, ISP, BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)
                   STOP
                ENDIF

                IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K1,ISP))) CYCLE
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))  CYCLE
                
                EIGVECT((LAMBDA-1)*NCV+NCV13)=AMAT(IROW+(JCOL-1)*DESCSTD(LLD_))
             ENDDO
          ENDDO
       ENDIF
    ENDDO

!Gather all parts of the eigenvectors stored on the different nodes
    CALLMPI( M_sum_z(WHF%WDES%COMM, EIGVECT, NBSEEIG*NCV))

    IF (IU6>=0) CALL WRITE_BSE(WHF,NCV,BSE_INDEX,EIGVECT,R,NBSEEIG)

    DEALLOCATE(EIGVECT)
 
  END SUBROUTINE WRITE_BSE_SCALA
#endif

!**********************************************************************
!
! Write out BSE eigenfunctions and eigenvalues
! 			M.Bokdam Nov 2014
!**********************************************************************

  SUBROUTINE WRITE_BSE(WHF,NCV,BSE_INDEX,EIGVECT,R,NBSEEIG)

    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    TYPE (latt) LATT_CUR
    TYPE(bse_matrix_index) :: BSE_INDEX
    INTEGER :: NCV
    INTEGER :: NBSEEIG
    COMPLEX(q) :: EIGVECT(:)
    REAL(q) :: R(:)
 ! local 
    INTEGER K1,K2,K3,K4,NPOS1,NPOS2,MINL 
    INTEGER I, J
    INTEGER     :: LAMBDA, ISP
    REAL(q) :: RIP(NCV),MINV 

      K1=1
      DO NPOS2=1,SIZE(R)
        K2=BSE_INDEX%NK(NPOS2)
        K3=BSE_INDEX%N1(NPOS2)
        K4=BSE_INDEX%N3(NPOS2)
        ISP=BSE_INDEX%ISP(NPOS2)
        RIP(NPOS2)=REAL(WHF%CELTOT(K4,K2,ISP)-WHF%CELTOT(K3,K2,ISP))
        K1=K1+1
      ENDDO
!MB very inefficient SORTING routine, but ok for now
      DO NPOS1=1,NBSEEIG
        MINV=MINVAL(RIP(NPOS1:NCV)) !locate
        MINL=INT(MINLOC(RIP(NPOS1:NCV),1))+NPOS1-1
        RIP(MINL)=RIP(NPOS1)        !swap
        RIP(NPOS1)=MINV
      ENDDO

      OPEN(unit=78, file='BSEFATBAND', status="UNKNOWN")
      WRITE(78,'(2I18)') SIZE(R),NBSEEIG
      K1=1
      DO NPOS1=1,NBSEEIG
        WRITE(78,'(I6,A14,F14.8,A20,F14.8)') NPOS1,"BSE eigenvalue:",R(NPOS1), "IP-eigenvalue:",RIP(NPOS1)
        DO NPOS2=1,SIZE(R)
          K2=BSE_INDEX%NK(NPOS2)
          K3=BSE_INDEX%N1(NPOS2)
          K4=BSE_INDEX%N3(NPOS2)
          ISP=BSE_INDEX%ISP(NPOS2)
          WRITE(78,'(3F9.5,3F14.7,2I6,F14.6,A4,F12.6)') WHF%WDES%VKPT(:,K2),REAL(WHF%CELTOT(K3,K2,ISP)),REAL(WHF%CELTOT(K4,K2,ISP)),ABS(EIGVECT(K1))/WHF%WDES%WTKPT(K2),K3,K4,REAL(EIGVECT(K1)),"+i* ",AIMAG(EIGVECT(K1))
          K1=K1+1
        ENDDO
      ENDDO
      CLOSE(78)
      
  END SUBROUTINE WRITE_BSE

      

#ifdef use_bse_te
!****************** SUBROUTINE INIT_BSE_OSZI_STRENGTH *****************
!
! initialize the BSE oscillator strenght
! this is done quite analogously to the calculation of the 
! determination of the CALCULATE_BSE_OSZI_STRENGTH routine
!
!**********************************************************************

  SUBROUTINE INIT_BSE_OSZI_STRENGTH(WHF, LATT_CUR, ISP_LOW, ISP_HIGH, BD, BSE_INDEX )
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

    ! passed variables
    TYPE (wavespin) WHF
    TYPE (latt) LATT_CUR
    INTEGER :: ISP_LOW, ISP_HIGH
    TYPE(banddesc) :: BD(:)
    TYPE(bse_matrix_index) :: BSE_INDEX
    ! local
    INTEGER ISP
    INTEGER :: NCV13
    INTEGER K1
    INTEGER N1, N3
    INTEGER NB1_INTO_TOT, NB3_INTO_TOT
    REAL(q) :: WEIGHT
    INTEGER I, J, NK1_IN_KPOINTS_FULL_ORIG
    GDEF       :: CDER_BETWEEN_STATE(3)

    DO ISP=ISP_LOW,ISP_HIGH
    DO K1=1,WHF%WDES%NKPTS
       IF (WHF%WDES%WTKPT(K1)==0) CYCLE

       ! determined the index of this k-point in the original full k-point grid
       NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)
       
       DO N3=1,BD(ISP)%CBMAX-BD(ISP)%CBMIN+1
          NB3_INTO_TOT=N3+BD(ISP)%CBMIN-1
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K1,ISP))) CYCLE
          DO N1=1,BD(ISP)%VBMAX-BD(ISP)%VBMIN+1
             NB1_INTO_TOT=N1+BD(ISP)%VBMIN-1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE
             
             WEIGHT=(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)-WHF%FERTOT(NB3_INTO_TOT, K1, ISP))
             ! G-> 0 limit, consider all three directions
             CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB3_INTO_TOT)
             
             ! CDER_BETWEEN_STATE =  <v_k1,n1|- i d/dq_j | c_k1+q,n3>
             ! we need \int r v*_k1,n1(r) c_k1+q,n3(r) = i CDER_BETWEEN_STATE
           ! NCV13=NB1_INTO_TOT-BD(ISP)%VBMIN+1 +(BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)*(NB3_INTO_TOT-BD(ISP)%CBMIN+(BD(ISP)%CBMAX-BD(ISP)%CBMIN+1)*(K1-1))
             NCV13=BSE_INDEX%INDEX(NB1_INTO_TOT-BD(ISP)%VBMIN+1, NB3_INTO_TOT-BD(ISP)%CBMIN+1, K1, ISP -ISP_LOW+1)
             IF (NCV13==0) CYCLE ! cycle if this pair is not included
!      OPTMAT (on Transmatrix)
!      was first of all multiplied with EC-EV in order to obtain the true matrix
!      element (for consistency with the old momentum-operator implementation)
!      but was then divided again by this factor upon reading in get_exciton.F;
!      Only a factor AUTOA/(2*HSQDTM) for conversion into atomic units should be
!      left here! Furthermore the k-point weight is also misplaced here!
             CDER_BETWEEN_STATE=CDER_BETWEEN_STATE*WEIGHT*AUTOA/(2._q*HSQDTM)
             OPTMAT(NCV13,:)=CDER_BETWEEN_STATE
             LDAGW(NCV13)=WHF%CELTOT(NB3_INTO_TOT,K1,ISP)-WHF%CELTOT(NB1_INTO_TOT, K1, ISP)
          ENDDO  ! enddo for N3-loop
       ENDDO     ! enddo for N1-loop
    ENDDO  ! enddo for K1-loop
    ENDDO  ! enddo spin loop

  END SUBROUTINE INIT_BSE_OSZI_STRENGTH
#endif

!**********************************************************************
!
! small subroutine that determine
! which band window is condsidered in the BSE
! In fact the window is usually different for up and
! down spin making things really difficult
!
!**********************************************************************

  SUBROUTINE SET_BAND_PARAMETERS(W, ISP, NBANDSO, NBANDSV, & 
       VBMAX, VBMIN, CBMIN, CBMAX, & 
       NSTRIPV, NSTRIP, NGLB, CBMIN4, CBMAX4, NGLB4, W1EQUALW2,  LKPOINT_PARALLEL, FIRST_EMPTYSTATE, LAST_OCCSTATE ) 
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER :: ISP, NBANDSO, NBANDSV
    INTEGER :: VBMAX, VBMIN, CBMIN, CBMAX
    INTEGER,OPTIONAL :: NSTRIPV, NSTRIP, NGLB, CBMIN4, CBMAX4, NGLB4
    INTEGER,OPTIONAL, INTENT(IN) :: FIRST_EMPTYSTATE, LAST_OCCSTATE
    LOGICAL,OPTIONAL :: W1EQUALW2,  LKPOINT_PARALLEL
  ! local
    INTEGER :: N, NK
    
    INTEGER LAST_FILLED, FIRST_EMPTY
 
    LAST_FILLED=0
    FIRST_EMPTY=W%WDES%NB_TOT
    DO NK=1,W%WDES%NKPTS
       IF (W%WDES%WTKPT(NK)==0) CYCLE   
       LAST_FILLED=MAX(LAST_FILLED, LAST_FILLED_XI_NOMOD(W,NK,ISP))
       FIRST_EMPTY=MIN(FIRST_EMPTY, FIRST_EMPTY_XI_NOMOD(W,NK,ISP))
    ENDDO
    IF (PRESENT (FIRST_EMPTYSTATE)) THEN
        FIRST_EMPTY=FIRST_EMPTYSTATE
    ENDIF
    IF (PRESENT(LAST_OCCSTATE)) THEN
       LAST_FILLED=LAST_OCCSTATE
    ENDIF

    ! determine VBMIN and VBMAX
    VBMAX=LAST_FILLED
    VBMIN=MAX(LAST_FILLED-NBANDSO+1,1)
    IF (VBMIN > VBMAX) VBMIN=VBMAX

    CBMIN=FIRST_EMPTY
    CBMAX=MIN(FIRST_EMPTY+NBANDSV-1,W%WDES%NB_TOT)

    ! return if optional arguments are missing
    IF (.NOT. PRESENT(NSTRIPV)) RETURN

    ! NSTRIPV determines the blocking for the valence band states
    ! it can be made smaller than NBANDSO
    ! seems to work in the cases I have tested
    NSTRIPV=VBMAX-VBMIN+1
    NSTRIPV=MIN(NSTRIPV,64)

! more k-points than cores, always parallelization over k-points
! or number of k-points larger than number of conduction bands/4
#ifdef MPI
    IF (W%WDES%NKPTS >= W%WDES%NB_PAR*2 .OR. W%WDES%NKPTS*4 >= (CBMAX-CBMIN)) THEN
       LKPOINT_PARALLEL=.TRUE.
       W1EQUALW2=.FALSE.
    ELSE
       LKPOINT_PARALLEL=.FALSE.
       W1EQUALW2=.FALSE.
       IF (W%WDES%NKPTS==1) W1EQUALW2=.TRUE.
    ENDIF
#else
    LKPOINT_PARALLEL=.FALSE.
    W1EQUALW2=.FALSE.
    IF (W%WDES%NKPTS==1) W1EQUALW2=.TRUE.
#endif      
       
    IF (LKPOINT_PARALLEL) THEN
       CBMIN4=CBMIN
       CBMAX4=CBMAX
    ELSE
       ! loops over the fourth index are distributed over nodes
       ! determine data distribution
       NGLB4=(CBMAX-CBMIN)/W%WDES%NB_PAR+1
       CBMIN4=CBMIN
       DO N=1,W%WDES%NB_PAR
          CBMAX4=MIN(CBMIN4+NGLB4-1,CBMAX)
          IF (N==W%WDES%NB_LOW) EXIT
          CBMIN4=CBMAX4+1
       ENDDO
    ENDIF

    NGLB =CBMAX -CBMIN +1
    NGLB4=CBMAX4-CBMIN4+1

    ! NSTRIP determines the blocking for the conduction band states
    ! applied only internally in the TWOELECTRON4O_ACC
    ! it limits the matrix size in BLAS level three, but values around 32-64 should
    ! be fine for maximum performance
    NSTRIP= MIN(NGLB,64)

  END SUBROUTINE SET_BAND_PARAMETERS

!**********************************************************************
!
! subroutine which determines the index arrays that allow
! to determine the indices N1(valence),N2(conduction),NK,ISP
! from the global index into the BSE matrix
! also an index array is set up to determine the global index
! from the band indices, k-point and spin
!
!**********************************************************************

  SUBROUTINE SET_BSE_MATRIX_INDEX( ISP_LOW, ISP_HIGH, ALLOC_KPOINTS, BD, BSE_INDEX, W, OMEGAMAX_IN, NCV, NQPOINT)
       
    INTEGER :: ISP_LOW, ISP_HIGH, ALLOC_KPOINTS
    TYPE (banddesc) :: BD(:)
    TYPE (bse_matrix_index) :: BSE_INDEX

    INTEGER :: ALLOC_VALENCE_BANDS, ALLOC_CONDUCTION_BANDS, ALLOC_SPIN
    INTEGER :: N1, N3, K1, K3
    INTEGER :: ISP, INDEX
    TYPE (wavespin) W
    REAL(q) :: OMEGAMAX_IN    ! maximum energy difference between conduction and valence band to be considered in BSE
    REAL(q) :: OMEGAMAX_ACT   ! actual maximum energy difference between conduction and valence band
    INTEGER :: NCV            ! number of valence conduction band pairs
    INTEGER :: NQPOINT        ! q point at which exciton is calculated 

    REAL(q) :: OMEGAMAX

    IF (OMEGAMAX_IN>0) THEN
       OMEGAMAX=OMEGAMAX_IN
    ELSE
       OMEGAMAX=1E10
    ENDIF

    OMEGAMAX_ACT=0

    ALLOC_SPIN=ISP_HIGH-ISP_LOW+1
    ALLOC_VALENCE_BANDS=0
    ALLOC_CONDUCTION_BANDS=0

    DO ISP=ISP_LOW,ISP_HIGH
       ALLOC_VALENCE_BANDS   =MAX(ALLOC_VALENCE_BANDS   ,BD(ISP)%VBMAX-BD(ISP)%VBMIN+1)
       ALLOC_CONDUCTION_BANDS=MAX(ALLOC_CONDUCTION_BANDS,BD(ISP)%CBMAX-BD(ISP)%CBMIN+1)
    ENDDO
    
    ALLOCATE(BSE_INDEX%INDEX(ALLOC_VALENCE_BANDS,ALLOC_CONDUCTION_BANDS,ALLOC_KPOINTS,ALLOC_SPIN))
    ALLOCATE(BSE_INDEX%N1(ALLOC_VALENCE_BANDS*ALLOC_CONDUCTION_BANDS*ALLOC_KPOINTS*ALLOC_SPIN))
    ALLOCATE(BSE_INDEX%N3(ALLOC_VALENCE_BANDS*ALLOC_CONDUCTION_BANDS*ALLOC_KPOINTS*ALLOC_SPIN))
    ALLOCATE(BSE_INDEX%NK(ALLOC_VALENCE_BANDS*ALLOC_CONDUCTION_BANDS*ALLOC_KPOINTS*ALLOC_SPIN))
    ALLOCATE(BSE_INDEX%ISP(ALLOC_VALENCE_BANDS*ALLOC_CONDUCTION_BANDS*ALLOC_KPOINTS*ALLOC_SPIN))
    
    BSE_INDEX%INDEX=0
    BSE_INDEX%N1=0
    BSE_INDEX%N3=0
    BSE_INDEX%NK=0
    BSE_INDEX%ISP=0
! first spin component always U
    INDEX=0
    DO ISP=ISP_LOW,ISP_HIGH
       ! K1 is the k-point index for the valence band
       DO K1=1, ALLOC_KPOINTS
          ! K3 is the k-point index for the conduction band state (unoccupied)
          K3 =KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K1)+W%WDES%VKPT(:,NQPOINT),KPOINTS_FULL)

          DO N3=BD(ISP)%CBMIN, BD(ISP)%CBMAX
             ! safe guard, it might be that the number of bands (NB_TOT) exceeds the
             ! number of plane wave coefficients stored in NB_TOTK(NK,ISP)
             ! skip those useless bands
             IF (N3> W%WDES%NB_TOTK(K3,ISP)) CYCLE

             ! if occupancy is larger than 0.5 the band is occupied and we do not treat it as unoccupied state
             IF (FILLED_XI_ORBITAL(W%FERTOT(N3,K3,ISP))) CYCLE

             DO N1=BD(ISP)%VBMIN, BD(ISP)%VBMAX
                ! if occupancy of the valence band is smaller than 0.5 we skip the orbital as well
                IF (EMPTY_XI_ORBITAL(W%FERTOT(N1,K1,ISP))) CYCLE
                !! another safeguard: if the transition from k1 to k3 has energy difference smaller or eq to 0.0 then cycle
                IF (REAL(W%CELTOT(N3,K3,ISP)-W%CELTOT(N1,K1,ISP),q) <= 0.0_q) CYCLE
                
                 
 
                OMEGAMAX_ACT=MAX(OMEGAMAX_ACT, REAL(W%CELTOT(N3,K3,ISP)-W%CELTOT(N1,K1,ISP),q))
                IF (REAL(W%CELTOT(N3,K3,ISP)-W%CELTOT(N1,K1,ISP),q) < OMEGAMAX ) THEN
                   INDEX=INDEX+1
                   BSE_INDEX%INDEX(N1-BD(ISP)%VBMIN+1,N3-BD(ISP)%CBMIN+1,K1,ISP-ISP_LOW+1)=INDEX
                   BSE_INDEX%N1(INDEX) =  N1
                   BSE_INDEX%N3(INDEX) =  N3
                   BSE_INDEX%NK(INDEX) =  K1
                   BSE_INDEX%ISP(INDEX)= ISP

                ELSE
                   BSE_INDEX%INDEX(N1-BD(ISP)%VBMIN+1,N3-BD(ISP)%CBMIN+1,K1,ISP-ISP_LOW+1)=0
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDDO
!    WRITE(77,'(4I16)') BSE_INDEX%INDEX
!    WRITE(78,'(4I16)') BSE_INDEX%N1
!    WRITE(79,'(4I16)') BSE_INDEX%N3
!    WRITE(80,'(4I16)') BSE_INDEX%NK
!    WRITE(81,'(4I16)') BSE_INDEX%ISP

! number of pairs equals final index
    NCV=INDEX
    BSE_INDEX%NCV=NCV

    IF (OMEGAMAX_IN<=0) THEN
       ! be shure that the highest transition is never thrown away
       ! add a small safety measure
       OMEGAMAX_IN=OMEGAMAX_ACT*(1._q+1E-5_q)
    ENDIF

  END SUBROUTINE SET_BSE_MATRIX_INDEX


!**********************************************************************
!
! deallocate the BSE_INDEX array
!
!**********************************************************************

  SUBROUTINE DEALLOCATE_BSE_MATRIX_INDEX( BSE_INDEX )
    TYPE (bse_matrix_index) :: BSE_INDEX

    DEALLOCATE(BSE_INDEX%INDEX)
    DEALLOCATE(BSE_INDEX%N1)
    DEALLOCATE(BSE_INDEX%N3)
    DEALLOCATE(BSE_INDEX%NK)
    DEALLOCATE(BSE_INDEX%ISP)
    
  END SUBROUTINE DEALLOCATE_BSE_MATRIX_INDEX

!**********************************************************************
!
! diagonalize BSE matrix using one of the many possible calls
!
!**********************************************************************

  SUBROUTINE DIAG_BSE_MATRIX( NCV, AMAT, AMAT_SCALA, R, LscaLAPACKaware, MY_COMM, IU0 )
    USE wave_high
    INTEGER :: NCV
#ifdef single_prec_bse
    GDEFS :: AMAT(:,:)
    GDEFS :: AMAT_SCALA(:)
#else
    GDEF  :: AMAT(:,:)
    GDEF  :: AMAT_SCALA(:)
#endif
    REAL(q) ::  R(:)
    LOGICAL :: LscaLAPACKaware      ! distribute data using BLACS
    TYPE(communic) :: MY_COMM ! communicator to globally sum the diagonal components
    INTEGER :: IU0
! local
    REAL(q) ::  ABSTOL
    REAL(q) :: VL, VU

    INTEGER, PARAMETER :: LWORK=32
#ifdef single_prec_bse
    REAL(qs), ALLOCATABLE ::  RWORK(:)
    GDEFS, ALLOCATABLE    ::  CWRK(:)
    REAL(qs) :: R_tmp(SIZE(R))
#else
    REAL(q), ALLOCATABLE ::  RWORK(:)
    GDEF, ALLOCATABLE    ::  CWRK(:)
#endif
#ifdef USE_ZHEEVX
    INTEGER, ALLOCATABLE :: IWORK(:), INFOZ(:)
    INTEGER :: IL, IU, NB_CALC
#endif
#ifdef single_prec_bse
    GDEFS, ALLOCATABLE :: COVL(:,:)
#else
    GDEF, ALLOCATABLE :: COVL(:,:)
#endif
    INTEGER :: IFAIL

!==========================================================================
! simplest version ZHEEV
! TODO add single precision version
! only issue is that R is always double prec. 
! need a temporary single prec R_single
!==========================================================================
#ifndef USE_ZHEEVX
    ALLOCATE( CWRK(NCV*LWORK), RWORK(3*NCV))
#ifdef  gammareal
#ifdef  single_prec_bse
    IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (SSYEV)'
    CALL SSYEV &
         ('V','U',NCV, AMAT(1,1), NCV, &
         R_TMP, CWRK, LWORK*NCV, IFAIL)
    R=R_TMP
#else
    IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (DSYEV)'
    CALL DSYEV &
         ('V','U',NCV, AMAT(1,1), NCV, &
         R, CWRK, LWORK*NCV, IFAIL)
#endif
#else
#ifdef  single_prec_bse
    IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (CHEEV)'
    CALL CHEEV &
         ('V','U',NCV, AMAT(1,1), NCV, &
         R_TMP, CWRK, LWORK*NCV, RWORK,  IFAIL)
    R=R_TMP
#else
    IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (ZHEEV)'
    CALL ZHEEV &
         ('V','U',NCV, AMAT(1,1), NCV, &
         R, CWRK, LWORK*NCV, RWORK,  IFAIL)
#endif
#endif
    DEALLOCATE( CWRK, RWORK )
#else
!==========================================================================
! versions using ZHEEVX and corresponding LAPACK routines
!==========================================================================
!==========================================================================
! version using scaLAPACK matrix already distributed
!==========================================================================
    ABSTOL=1E-6_q
    IF ( LscaLAPACKaware) THEN
       IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (scaLAPACKaware)'
#ifdef single_prec_bse
       CALL pSSYEX_CHEEVX_single(MY_COMM, AMAT, R,  NCV, NCV, ABSTOL, AMAT_SCALA)
#else 
       CALL pDSSYEX_ZHEEVX(      MY_COMM, AMAT, R,  NCV, NCV, ABSTOL, AMAT_SCALA)
#endif
    ELSE IF ( LscaLAPACK) THEN
       IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (scaLAPACK)'
#ifdef single_prec_bse
       CALL pSSYEX_CHEEVX( MY_COMM, AMAT, R,  NCV, NCV, ABSTOL)
#else
       CALL pDSSYEX_ZHEEVX(MY_COMM, AMAT, R,  NCV, NCV, ABSTOL)
#endif

#ifdef single_prec_bse
#ifdef gammareal
       CALLMPI( M_sum_single(MY_COMM, AMAT(1,1)  , SIZE(AMAT)))
#else 
       CALLMPI( M_sum_single(MY_COMM, AMAT(1,1)  , 2*SIZE(AMAT)))
#endif
#else
       CALLMPI( M_sum_g(MY_COMM, AMAT(1,1)  , SIZE(AMAT)))
#endif
    ELSE
!==========================================================================
! version using LAPACK ZHEEVX
! TODO add single precision version
!==========================================================================
       IF (IU0>=0) WRITE(IU0,*) 'BSE diagonalizing matrix (ZHEEVX)'
       VL=0 ; VU=0 ; IL=0 ; IU=0
       ALLOCATE(COVL(NCV,NCV), CWRK(NCV*LWORK), RWORK(7*NCV), IWORK(5*NCV), INFOZ(NCV))
#ifdef  gammareal
#ifdef single_prec_bse
       CALL SSYEVX( 'V', 'A', 'U', NCV, AMAT(1,1) , NCV, VL, VU, IL, IU, &
            ABSTOL , NB_CALC , R_TMP, COVL(1,1), NCV, CWRK, &
            LWORK*NCV, RWORK, IWORK, INFOZ, IFAIL )
       R=R_TMP
#else
       CALL DSYEVX( 'V', 'A', 'U', NCV, AMAT(1,1) , NCV, VL, VU, IL, IU, &
            ABSTOL , NB_CALC , R, COVL(1,1), NCV, CWRK, &
            LWORK*NCV, RWORK, IWORK, INFOZ, IFAIL )         
#endif
#else
#ifdef single_prec_bse
       CALL CHEEVX( 'V', 'A', 'U', NCV, AMAT(1,1) , NCV, VL, VU, IL, IU, &
            ABSTOL , NB_CALC , R_TMP, COVL(1,1), NCV, CWRK, &
            LWORK*NCV, RWORK, IWORK, INFOZ, IFAIL )
       R=R_TMP
#else
       CALL ZHEEVX( 'V', 'A', 'U', NCV, AMAT(1,1) , NCV, VL, VU, IL, IU, &
            ABSTOL , NB_CALC , R, COVL(1,1), NCV, CWRK, &
            LWORK*NCV, RWORK, IWORK, INFOZ, IFAIL )
#endif
#endif
       AMAT=COVL
       DEALLOCATE(COVL, CWRK, RWORK, IWORK, INFOZ)
    ENDIF
#endif

#ifdef USE_ZHEEVX
#else
#endif

  END SUBROUTINE DIAG_BSE_MATRIX


!**********************************************************************
!
! left and right multiply the matrix A by (A-B)^(1/2)
! the matrix A is supplied in A
! the matrix A-B is supplied as U R U+
!
! the transformed  matrix is therefore given by
! C = U R U+ A U R U+
! to save storage 4 matrix matrix products are calculated
! only one temporary matrix is used
!
!**********************************************************************

! whooping many combinations
#ifdef single_prec_bse
#ifdef gammareal
#define GGEMMbse SGEMM
#define onebse   1._qs
#define zerobse  0._qs
#else
#define GGEMMbse CGEMM
#define onebse   (1._qs,0._qs)
#define zerobse  (0._qs,0._qs)
#endif
#else
#ifdef gammareal
#define GGEMMbse DGEMM
#define onebse   1._q
#define zerobse  0._q
#else
#define GGEMMbse ZGEMM
#define onebse   (1._q,0._q)
#define zerobse  (0._q,0._q)
#endif
#endif


  SUBROUTINE TDA_BSE( NCV, A, U, RB )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:)
#ifdef single_prec_bse
    GDEFS :: A(:,:), U(:,:)
    GDEFS,ALLOCATABLE :: T(:,:)
#else
    GDEF  :: A(:,:), U(:,:)
    GDEF,ALLOCATABLE :: T(:,:)
#endif
    INTEGER :: I, J
    REAL(q) :: SR(SIZE(RB))

    ALLOCATE(T(NCV,NCV))

    !  U+ (A+B) -> T
    CALL GGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, A(1,1), NCV, zerobse, T(1,1), NCV)
    !  [U+ (A+B)] U -> A
    CALL GGEMMbse( 'N','N', NCV, NCV, NCV, onebse, & 
         T(1,1), NCV, U(1,1), NCV, zerobse, A(1,1), NCV)
    ! multiply with sqrt of eigenvalues from left and right
    SR=SQRT(RB)
    DO J=1,NCV
       DO I=1,NCV
          A(I,J)=A(I,J)*SR(I)*SR(J)
       ENDDO
    ENDDO
    !  U A -> T
    CALL GGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, A(1,1), NCV, zerobse, T(1,1), NCV)
    !  [U A ] U+ -> A
    CALL GGEMMbse( 'N',trans, NCV, NCV, NCV, onebse, & 
         T(1,1), NCV, U(1,1), NCV, zerobse, A(1,1), NCV)

    DEALLOCATE(T)

  END SUBROUTINE TDA_BSE

!**********************************************************************
!
! determine the eigenvectors of the original problem e.g.
!  A  B    X     1  0   X
!             =             w
!  B  A    Y     0  -1  Y
!
! it is fairly easy to see that
!  (X+Y) = (A-B)^( 1/2) Z R^(-1/2)
!  (X-Y) = (A-B)^(-1/2) Z R^1/2
!
! where Z are the eigenvectors of the 
! squared problem [(A-B)^(1/2) (A+B) (A-B)^(1/2) Z = Z R]
! and R the eigenvalues of the original problem
! furthermore one can show the
!  (X+Y) (X-Y)+  = 1
! and 
!  X Y+  - Y X+  = 0
! and 
!  X X+ - Y Y+   = 1
!
! input:
! Z=z (eigenvectors of squared problem) and R
! A-B is passed as A-B  = U RB U+
!
! return:
! Z = X+Y contains usually most of the intensity
! the second routine returns X and Y but needs additional work space
!
!**********************************************************************
    
  SUBROUTINE TDA_BSE_EIGENVECTORS( NCV, U, RB, Z,  R )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:), R(:)
#ifdef single_prec_bse
    GDEFS :: Z(:,:), U(:,:)
    GDEFS,ALLOCATABLE :: T(:,:)
#else
    GDEF  :: Z(:,:), U(:,:)
    GDEF,ALLOCATABLE :: T(:,:)
#endif
    INTEGER :: I, J
    REAL(q) :: SRB(SIZE(R)),SR(SIZE(R))

    ALLOCATE(T(NCV,NCV))

    !  U+ Z -> T2
    CALL GGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, Z(1,1), NCV, zerobse, T(1,1), NCV)
    ! T1= SQRT(RB) U+ Z 1/SQRT(R)
    SRB=SQRT(RB)
    SR =1/SQRT(R)
    DO J=1,NCV
       DO I=1,NCV
          T(I,J)=SRB(I)*T(I,J)*SR(J)
       ENDDO
    ENDDO

    !  store Z =(X+Y)= U T1 -> A
    CALL GGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, T(1,1), NCV, zerobse, Z(1,1), NCV)
    U=0

    DEALLOCATE(T)
      
  END SUBROUTINE TDA_BSE_EIGENVECTORS
      
  SUBROUTINE TDA_BSE_EIGENVECTORS_FULL( NCV, U, RB, Z,  R )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:), R(:)
#ifdef single_prec_bse
    GDEFS :: Z(:,:), U(:,:)
    GDEFS,ALLOCATABLE :: T1(:,:),T2(:,:)
#else
    GDEF  :: Z(:,:), U(:,:)
    GDEF,ALLOCATABLE :: T1(:,:),T2(:,:)
#endif
    INTEGER :: I, J
    REAL(q) :: SRB(SIZE(R)),SR(SIZE(R))

    ALLOCATE(T1(NCV,NCV),T2(NCV,NCV))

    !  U+ Z -> T2
    CALL GGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, Z(1,1), NCV, zerobse, T2(1,1), NCV)
    ! T1= SQRT(RB) U+ Z 1/SQRT(R)
    SRB=SQRT(RB)
    SR =1/SQRT(R)
    DO I=1,NCV
       DO J=1,NCV
          T1(I,J)=SRB(I)*T2(I,J)*SR(J)
       ENDDO
    ENDDO

    ! T2= 1/SQRT(RB) U+ Z SQRT(R)
    SRB=1/SQRT(RB)
    SR =SQRT(R)
    DO J=1,NCV
       DO I=1,NCV
          T2(I,J)=SRB(I)*T2(I,J)*SR(J)
       ENDDO
    ENDDO
    !  Z =(X+Y)= U T1 -> A
    CALL GGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, T1(1,1), NCV, zerobse, Z(1,1), NCV)

    !  T1=(X-Y)= U T2 -> A
    CALL GGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1,1), NCV, T2(1,1), NCV, zerobse, T1(1,1), NCV)

    !  test multiply Z x U should be equal unity matrix
    ! U=T1
    ! CALL GGEMMbse( 'N', trans, NCV, NCV, NCV, onebse, & 
    !     Z(1,1), NCV, T1(1,1), NCV, zerobse, T2(1,1), NCV)
    !DO I=1,NCV
    !   WRITE(77,'(10E20.10)') T2(:,I)
    !ENDDO

    ! U = Y
    U=(Z-T1)/2
    ! Z = X
    Z=(Z+T1)/2

    ! test calculate X X+ - Y Y+  should be equal unity matrix
    ! CALL GGEMMbse( 'N', trans, NCV, NCV, NCV, onebse, & 
    !   Z(1,1), NCV, Z(1,1), NCV, zerobse, T1(1,1), NCV)
    ! CALL GGEMMbse( 'N', trans, NCV, NCV, NCV, -onebse, & 
    !   U(1,1), NCV, U(1,1), NCV, onebse, T1(1,1), NCV)
    ! DO I=1,NCV
    !    WRITE(77,'(10E20.10)') T1(:,I)
    ! ENDDO
    DEALLOCATE(T1,T2)
      
  END SUBROUTINE TDA_BSE_EIGENVECTORS_FULL


#ifdef scaLAPACK
!**********************************************************************
!
! same as TDA_BSE but distributed version using SCALAPACK
!
!**********************************************************************

! whooping many combinations
#ifdef single_prec_bse
#ifdef gammareal
#define GPGEMMbse PSGEMM
#else
#define GPGEMMbse PCGEMM
#endif
#else
#ifdef gammareal
#define GPGEMMbse PDGEMM
#else
#define GPGEMMbse PZGEMM
#endif
#endif


  SUBROUTINE TDA_BSE_SCA( NCV, A, U, RB )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:)
#ifdef single_prec_bse
    GDEFS :: A(:), U(:)
    GDEFS,ALLOCATABLE :: T(:)
#else
    GDEF  :: A(:), U(:)
    GDEF,ALLOCATABLE :: T(:)
#endif
    INTEGER :: I, J
    REAL(q) :: SR(SIZE(RB))
! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP, NQ
    INTEGER I1RES,J1RES,IROW,JCOL
    INTEGER I1,I2,J1,J2
    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)

    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

    ALLOCATE(T(SIZE(A)))

    !  U+ (A+B) -> T
    CALL GPGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, A(1), 1, 1, DESCSTD, zerobse, T(1), 1, 1, DESCSTD)
    !  [U+ (A+B)] U -> A
    CALL GPGEMMbse( 'N','N', NCV, NCV, NCV, onebse, & 
         T(1), 1, 1, DESCSTD, U(1), 1, 1, DESCSTD, zerobse, A(1), 1, 1, DESCSTD)
    ! multiply with sqrt of eigenvalues from left and right
    SR=SQRT(ABS(RB))

    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
          J=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                I=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
          ! A(I,J)=A(I,J)*SR(I)*SR(J)
                A(IROW+(JCOL-1)*DESCSTD(LLD_))=A(IROW+(JCOL-1)*DESCSTD(LLD_))*SR(I)*SR(J)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    !  U A -> T
    CALL GPGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, A(1), 1, 1, DESCSTD, zerobse, T(1), 1, 1, DESCSTD)
    !  [U A ] U+ -> A
    CALL GPGEMMbse( 'N',trans, NCV, NCV, NCV, onebse, & 
         T(1), 1, 1, DESCSTD, U(1), 1, 1, DESCSTD, zerobse, A(1), 1, 1, DESCSTD)

    DEALLOCATE(T)

  END SUBROUTINE TDA_BSE_SCA

!**********************************************************************
!
! same as TDA_BSE_EIGENVECTORS but distributed version using SCALAPACK
!
!**********************************************************************


  SUBROUTINE TDA_BSE_EIGENVECTORS_SCA( NCV, U, RB, Z,  R )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:), R(:)
#ifdef single_prec_bse
    GDEFS :: Z(:), U(:)
    GDEFS,ALLOCATABLE :: T(:)
#else
    GDEF  :: Z(:), U(:)
    GDEF,ALLOCATABLE :: T(:)
#endif
    INTEGER :: I, J
    REAL(q) :: SRB(SIZE(R)),SR(SIZE(R))
! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP, NQ
    INTEGER I1RES,J1RES,IROW,JCOL
    INTEGER I1,I2,J1,J2
    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)

    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)

    ALLOCATE(T(SIZE(Z)))

    !  U+ Z -> T2
    CALL GPGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, Z(1), 1, 1, DESCSTD, zerobse, T(1), 1, 1, DESCSTD)
    ! T1= SQRT(RB) U+ Z 1/SQRT(R)
    SRB=SQRT(RB)
    SR =1/SQRT(R)

    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
          J=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                I=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
          ! T(I,J)=SRB(I)*T(I,J)*SR(J)
                T(IROW+(JCOL-1)*DESCSTD(LLD_))=T(IROW+(JCOL-1)*DESCSTD(LLD_))*SRB(I)*SR(J)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    !  Z =(X+Y)= U T1 -> A
    CALL GPGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, T(1), 1, 1, DESCSTD, zerobse, Z(1), 1, 1, DESCSTD)
    U=0

    DEALLOCATE(T)
      
  END SUBROUTINE TDA_BSE_EIGENVECTORS_SCA

  SUBROUTINE TDA_BSE_EIGENVECTORS_FULL_SCA( NCV, U, RB, Z,  R )
    USE wave_high
    INTEGER :: NCV
    REAL(q) :: RB(:), R(:)
#ifdef single_prec_bse
    GDEFS :: Z(:), U(:)
    GDEFS,ALLOCATABLE :: T1(:),T2(:)
#else
    GDEF  :: Z(:), U(:)
    GDEF,ALLOCATABLE :: T1(:),T2(:)
#endif
    INTEGER :: I, J
    REAL(q) :: SRB(SIZE(R)),SR(SIZE(R))
! BLACS variables
    INTEGER, EXTERNAL ::     NUMROC
    INTEGER MYROW, MYCOL, NPROW, NPCOL, NP, NQ
    INTEGER I1RES,J1RES,IROW,JCOL
    INTEGER I1,I2,J1,J2
    CALL BLACS_GRIDINFO(DESCSTD(CTXT_),NPROW,NPCOL,MYROW,MYCOL)

    NP = NUMROC(NCV,DESCSTD(MB_),MYROW,0,NPROW)
    NQ = NUMROC(NCV,DESCSTD(NB_),MYCOL,0,NPCOL)


    ALLOCATE(T1(SIZE(Z)),T2(SIZE(Z)))

    !  U+ Z -> T2
    CALL GPGEMMbse( trans,'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, Z(1), 1, 1, DESCSTD, zerobse, T2(1), 1, 1, DESCSTD)
    ! T1= SQRT(RB) U+ Z 1/SQRT(R)
    SRB=SQRT(RB)
    SR =1/SQRT(R)

    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
          J=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                I=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
          ! T1(I,J)=SRB(I)*T2(I,J)*SR(J)
                T1(IROW+(JCOL-1)*DESCSTD(LLD_))=T2(IROW+(JCOL-1)*DESCSTD(LLD_))*SRB(I)*SR(J)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ! T2= 1/SQRT(RB) U+ Z SQRT(R)
    SRB=1/SQRT(RB)
    SR =SQRT(R)

    JCOL=0
    DO J1=1,NQ,DESCSTD(NB_)
       J1RES=MIN(DESCSTD(NB_),NQ-J1+1)
       DO J2=1,J1RES
          JCOL=JCOL+1
          J=DESCSTD(NB_)*MYCOL+NPCOL*(J1-1)+J2

          IROW=0
          DO I1=1,NP,DESCSTD(MB_)
             I1RES=MIN(DESCSTD(MB_),NP-I1+1)
             DO I2=1,I1RES
                IROW=IROW+1
                I=DESCSTD(MB_)*MYROW+NPROW*(I1-1)+I2
                T2(IROW+(JCOL-1)*DESCSTD(LLD_))=T2(IROW+(JCOL-1)*DESCSTD(LLD_))*SRB(I)*SR(J)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    !  Z =(X+Y)= U T1 -> A
    CALL GPGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, T1(1), 1, 1, DESCSTD, zerobse, Z(1), 1, 1, DESCSTD)

    !  T1=(X-Y)= U T2 -> A
    CALL GPGEMMbse( 'N', 'N', NCV, NCV, NCV, onebse, & 
         U(1), 1, 1, DESCSTD, T2(1), 1, 1, DESCSTD, zerobse, T1(1), 1, 1, DESCSTD)

    !  test multiply Z x U should be equal unity matrix
    ! U=T1
    ! CALL GPGEMMbse( 'N', trans, NCV, NCV, NCV, onebse, & 
    !     Z(1), 1, 1, DESCSTD, T1(1), 1, 1, DESCSTD, zerobse, T2(1), 1, 1, DESCSTD)
    !DO I=1,NCV
    !   WRITE(77,'(10E20.10)') T2(:,I)
    !ENDDO

    ! U = Y
    U=(Z-T1)/2
    ! Z = X
    Z=(Z+T1)/2

    ! test calculate X X+ - Y Y+  should be equal unity matrix
    ! CALL GPGEMMbse( 'N', trans, NCV, NCV, NCV, onebse, & 
    !   Z(1), 1, 1, DESCSTD, Z(1), 1, 1, DESCSTD, zerobse, T1(1), 1, 1, DESCSTD)
    ! CALL GPGEMMbse( 'N', trans, NCV, NCV, NCV, -onebse, & 
    !   U(1), 1, 1, DESCSTD, U(1), 1, 1, DESCSTD, onebse, T1(1), 1, 1, DESCSTD)
    ! DO I=1,NCV
    !    WRITE(77,'(10E20.10)') T1(:,I)
    ! ENDDO
    DEALLOCATE(T1,T2)
      
  END SUBROUTINE TDA_BSE_EIGENVECTORS_FULL_SCA


#endif
      
END MODULE bse
