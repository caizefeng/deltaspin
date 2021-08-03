!#define dotiming
#include "symbol.inc"
#define USE_ZHEEVX
#define GPU_FAST_TRANSFERT

MODULE subrot
  USE prec
  USE pead
  USE dfast
  USE dfast_gpu
  USE hamil_high
  USE hamil_gpu
CONTAINS
!************************ SUBROUTINE EDDIAG ****************************
! RCS:  $Id: subrot.F,v 1.10 2003/06/27 13:22:23 kresse Exp kresse $
!
! this subroutine calculates the electronic eigenvalues and
! optionally performes a  sub-space diagonalisation
! i.e. unitary transforms the wavefunctions so that the Hamiltonian
!  becomes diagonal in the subspace spanned by the wavefunctions
! IFLAG:
!  0 only eigenvalues (without  diagonalisation no sub-space matrix)
!  1 only eigenvalues and sub-space matrix (no diagonalisation)
!  2 eigenvalues using diagonalisation of sub-space matrix
!    do not rotate wavefunctions
! 12 eigenvalues using diagonalisation of sub-space matrix
!    no kinetic energy
!  3 eigenvalues and sub-space diagonalisation rotate wavefunctions
! 13 (for 13 no Jacoby algorithm is allowed)
!  4 eigenvalues and sub-space diagonalisation rotate wavefunctions
!    using Loewdin pertubation theory (conserves ordering)
!  5 eigenvalues and sub-space diagonalisation + orthogonalization
!    unfortunately this option turns out to slow down the 
!    convergence of IALGO=48
!
! notes on the matrix CHAM used internally
! it is defined as 
!  CHAM(n2,n1)   = < phi_n2 | H | phi_n1 >
! however the returned matrix CHAMHF is defined as the complex conjugated 
!  CHAMHF(n2,n1) = < phi_n2 | H | phi_n1 >* = < phi_n1 | H | phi_n2 >
! since this is what rot.F uses
!
!  written by gK
!  last update Sep 24 2004 massive cleanup
!  optional arguments:
!  NBANDS_MAX   maximum number of bands for which eigenvalues are
!               calculated (only supported for IFLAG=0)
!  TODO: Martijn new arguments
!
!
! GPU part : HACENE Mohamed
!***********************************************************************

  SUBROUTINE EDDIAG(HAMILTONIAN, &
       GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
       LMDIM,CDIJ,CQIJ,IFLAG,SV,T_INFO,P,IU0,EXHF, & 
       CHAMHF,LFIRST,LLAST,NKSTART,NKSTOP,NBANDS_MAX,EXHF_ACFDT)
    USE iso_c_binding
    USE cuda_interface
    USE prec
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE constant
    USE jacobi
    USE scala
    USE main_mpi
    USE fock
    USE pseudo

    USE gpu_data
      
    USE ini
    USE sym_grad

    USE c2f_interface, ONLY : VTIME

    IMPLICIT NONE
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (symmetry) :: SYMM
    INTEGER LMDIM
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    INTEGER            IFLAG            ! determines mode of diagonalisation
    RGRID, TARGET ::   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    INTEGER IU0, IU6
    REAL(q) EXHF
   INTEGER ierror
! if CHAMHF is present, CHAMHF is added to the subspace matrix at each iteration
! in addition calculation of Fock part is bypassed
    GDEF, OPTIONAL :: CHAMHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
! if LFIRST is supplied CHAMHF is set to CONJG(CHAMHF)-CHAM
! if LLAST  is supplied CHAMHF is set to CONJG(CHAM)
    LOGICAL, OPTIONAL :: LFIRST, LLAST
    INTEGER, OPTIONAL :: NKSTART, NKSTOP     ! start k-point
    INTEGER, OPTIONAL :: NBANDS_MAX  ! maximum band index
    LOGICAL :: LSCAAWARE_LOCAL
    REAL(q), OPTIONAL :: EXHF_ACFDT
    
! local
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*WDES%NB_TOT)
    REAL(q)    R(WDES%NB_TOT)
#ifndef USE_ZHEEVX
    REAL(q)    RWORK(3*WDES%NB_TOT)
#else
    REAL(q)    RWORK(7*WDES%NB_TOT), ABSTOL, VL, VU
    INTEGER IWORK(5*WDES%NB_TOT), INFO(WDES%NB_TOT),IL, IU, NB_CALC
#endif
    INTEGER     USED_MAGMA  !Indicates if we used MAGMA, this prevents a memory copy
    ! work arrays (do max of 16 strips simultaneously)
  
    ! 3 per loop  , 2 loops, so 6 total
    !#define NSTREAMBUFFERS  6   
    INTEGER , PARAMETER ::  NSTREAMBUFFERS=6 
    TYPE (wavedes1)    WDES1          ! descriptor for one k-point
    TYPE (wavefun1)    W1             ! current wavefunction
    TYPE (wavefun1)    W1ARRAY(NSTREAMBUFFERS)
    TYPE (wavefun1)    WHAM1          ! store Hamiltonian times wavefunction
    TYPE (wavefuna)    WA             ! array to store wavefunction
    TYPE (wavespin)    WFOCK          ! array to store the Fock (exchange contribution)
    TYPE (wavefuna)    WNONL          ! array to hold non local part D * wave function character
    TYPE (wavefuna)    WOVL           ! array to hold non local part Q * wave function character
    TYPE (wavefuna)    WHAM           ! array to store accelerations for a selected block
    TYPE (wavefuna)    WHAM_PING           ! array to store accelerations for a selected block
    TYPE (wavefuna)    WHAM_PONG           ! array to store accelerations for a selected block
    COMPLEX(q)         CDCHF          ! HF double counting energy
    COMPLEX(q)         COMPLEX_SIZE
    INTEGER :: ICALL=0                ! number of calls
    GDEF,ALLOCATABLE,TARGET::  CHAM(:,:),COVL(:,:) ! Hamiltonian and overlap matrix
    TYPE (wavefun1)    WTMP(NSTRIP_STANDARD)

    ! array for asyncronous data redistribution
    TYPE (REDIS_PW_CTR),POINTER :: H_PW1, H_PW2
    TYPE (REDIS_PW_CTR),POINTER :: H_PWNVA1, H_PWNVA2, H_PWNVB1, H_PWNVB2
    TYPE (REDIS_PW_CTR)         :: H_PWNV1, H_PWNV2
    INTEGER MPLWV,NRSPINORS
    !Use GPU 
    TYPE(gpu_type) GPU

    INTEGER(c_intptr_t) :: GPU_CHARRAY(NSTREAMBUFFERS)
    INTEGER :: STREAM, W1_COUNT, OLD_W1_COUNT, OLD_NPOS, OLD_NSTRIP_ACT
    INTEGER :: STREAM2, STREAM3, II, LOOP_COUNTER
    INTEGER(c_intptr_t) :: GPU_CWWARRAY(NSTREAMBUFFERS)
    INTEGER(c_intptr_t) :: GPU_CRARRAY(NSTREAMBUFFERS)
    INTEGER(c_intptr_t) GPU_CWORK(NSTREAMBUFFERS),GPU_CWORK_ALL
    INTEGER :: maxNGVECTOR
    
    ! the following flag decides whether we are allowed to redistribute
    ! the wavefunctions during the calculation of the accelerations
    ! this fails i.e. for Hartree Fock calculations

    LOGICAL, EXTERNAL :: USEFOCK_CONTRIBUTION

    INTEGER :: NB_TOT, NBANDS, NSTRIP, ISP, NK, N, NP, NPOS, NSTRIP_ACT, &
         NPOS_RED, NSTRIP_RED, IFAIL, MY_NKSTART,i,j, MY_NKSTOP, NSIM_LOCAL
    COMPLEX(q)  CW_RED_OLD(WDES%NB_TOT)
    REAL(q) TV,TV0,TC,TC0,TV1,TC1,TimeTrans,TimeFFT,TimeORTH,TimeECCP,TimeHamil,TimeOther,TV2,TC2
    REAL(q) TVJB1, TVJB2, TVJB3, TVJB4
    COMPLEX(q) fakec
#ifdef TAU_PROF
    integer, dimension (2,18) :: profile
    save profile
        

    call TAU_PROFILE_TIMER(profile(1,1),'EDDIAG')
    call TAU_PROFILE_START(profile(1,1))
    call TAU_PROFILE_TIMER(profile(1,2),'EDDIAG::INIT_BLOCK')
    call TAU_PROFILE_TIMER(profile(1,3),'EDDIAG::S_K_LOOP')
    call TAU_PROFILE_TIMER(profile(1,4),'EDDIAG::END_BLOCK')
    call TAU_PROFILE_TIMER(profile(1,5),'EDDIAG::S_K_LOOP::ALLOCS')
    call TAU_PROFILE_TIMER(profile(1,6),'EDDIAG::S_K_LOOP::IFLAG0')
    call TAU_PROFILE_TIMER(profile(1,7),'EDDIAG::S_K_LOOP::REDIS')
    call TAU_PROFILE_TIMER(profile(1,8),'EDDIAG::S_K_LOOP::GPUMEMCPY')
    call TAU_PROFILE_TIMER(profile(1,9),'EDDIAG::S_K_LOOP::STRIPLOOP')
    call TAU_PROFILE_TIMER(profile(1,10),'EDDIAG::S_K_LOOP::GPUMEMFREE')
    call TAU_PROFILE_TIMER(profile(1,11),'EDDIAG::S_K_LOOP::IFLAG5')
    call TAU_PROFILE_TIMER(profile(1,12),'EDDIAG::S_K_LOOP::IFLAG4')
    call TAU_PROFILE_TIMER(profile(1,13),'EDDIAG::S_K_LOOP::LINCOM')
    call TAU_PROFILE_TIMER(profile(1,14),'EDDIAG::S_K_LOOP::ENDREDIS')
    call TAU_PROFILE_TIMER(profile(1,15),'EDDIAG::S_K_LOOP::STREAMSYNC')
    call TAU_PROFILE_TIMER(profile(1,16),'EDDIAG::S_K_LOOP::DATARECV')
    call TAU_PROFILE_TIMER(profile(1,17),'EDDIAG::S_K_LOOP::MEMCPY')
    call TAU_PROFILE_TIMER(profile(1,18),'EDDIAG::S_K_LOOP::ZGMEMSTART')
#endif

    
    TimeTrans = 0
    TimeFFT   = 0
    TimeORTH  = 0
    TimeECCP  = 0
    TimeHamil = 0
    TimeOther = 0
#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*) "######################################################################"
 write(*,*) "                          *** EDDIAG ***"
#ifdef MPI
 endif
#endif
#endif     
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,2))
#endif
    LscaAWARE_LOCAL=LSCAAWARE.AND.(.NOT.(IFLAG==4))
   
    USED_MAGMA = 0

!#define timing
#ifdef timing
    IU6=8
    IF (IU0<0) IU6=-1
    CALL START_TIMING("S")
#endif
    IF (PRESENT(CHAMHF) .OR. IFLAG==1) LSCAAWARE_LOCAL=.FALSE.
#ifdef timing
    LSCAAWARE_LOCAL=.FALSE.
#endif

nv_profile_start(NVP_EDDIAG)
!=======================================================================
! decide whether the wavefunctions can be redistributed
! during the calculation of the H phi
! this is not possible if Hartree Fock exchange is used
!=======================================================================
    CDCHF=0         ! double counting HF
    IF (PRESENT(EXHF_ACFDT)) EXHF_ACFDT=0
    ICALL=ICALL+1

    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS
    IF (PRESENT(NBANDS_MAX) .AND. IFLAG == 0) NBANDS=NBANDS_MAX

    NSTRIP=NSTRIP_STANDARD
    ! allocate work space
    ALLOCATE(W1%CR(GRID%MPLWV*WDES%NRSPINORS))

    IF (.NOT. LscaAWARE_LOCAL) THEN
       ALLOCATE(CHAM(NB_TOT,NB_TOT))
       CALL cublas_alloc_safety(NB_TOT*NB_TOT,int(c_sizeof(CHAM(1,1)),c_size_t),GPU%COVL)
    ELSE
       CALL INIT_scala(WDES%COMM_KIN, NB_TOT)
       ALLOCATE(CHAM(SCALA_NP(),SCALA_NQ()))
    ENDIF

    CALL SETWDES(WDES,WDES1,0)

    CALL NEWWAVA(WHAM_PING, WDES1, NSTRIP)
    CALL NEWWAVA(WHAM_PONG, WDES1, NSTRIP)
    WHAM = WHAM_PING
    CALL NEWWAVA_PROJ(WNONL, WDES1)

    IF (IFLAG==5) THEN
       ALLOCATE(COVL(NB_TOT,NB_TOT))
       CALL NEWWAVA_PROJ(WOVL, WDES1)
    ENDIF

    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ELSE
       MY_NKSTART=1
    ENDIF
    IF (PRESENT(NKSTOP)) THEN
       MY_NKSTOP=NKSTOP
    ELSE
       MY_NKSTOP=WDES%NKPTS
    ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,2))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,3))
#endif

    CALL VTIME(TV0,TC0)
    !print *,"WDES%ISPIN",WDES%ISPIN,"WDES%NKPTS",WDES%NKPTS
!=======================================================================
! start with HF part and store results WFOCK
!=======================================================================
    IF (USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF))) THEN

       CALL ALLOCW(WDES,WFOCK)
       NSIM_LOCAL=(W%WDES%NSIM*2+W%WDES%NB_PAR-1)/W%WDES%NB_PAR
       IF (LPEAD_NO_SCF()) THEN
          DO N=1,NSIM_LOCAL
             CALL NEWWAV(WTMP(N) , WDES1, .FALSE.)
          ENDDO
       ENDIF

       DO ISP=1,WDES%ISPIN
       DO NK=MY_NKSTART,MY_NKSTOP
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL SETWDES(WDES,WDES1,NK)

          IF (NONLR_S%LREAL) THEN
             CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
          ELSE
             CALL PHASE(WDES,NONL_S,NK)
          ENDIF

          DO NPOS=1,NBANDS,NSIM_LOCAL
             NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSIM_LOCAL)
             IF (LPEAD_NO_SCF()) THEN
             ! this is a little bit tricky/dirty
             ! if local field effects are not taken into account in the pead,
             ! (i.e. Hamiltonian constructed from initial wavefunctions)
             ! W_STORE (original wavefunctions) must be passed to FOCK_ACC and WTMP 
             ! (wavefunction on which the action is calculated)
             ! must be supplied as an additional argument
                DO N=NPOS,NPOS+NSTRIP_ACT-1
                   NP=N-NPOS+1
                   CALL W1_COPY(ELEMENT(W,WDES1,N,ISP),WTMP(NP))
                ENDDO
                CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_STORE,   &
                     NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                     WFOCK%CW(:,NPOS:,NK,ISP), P, CQIJ(1,1,1,1), CDCHF, WTMP, LSYMGRAD=LSYMGRAD)
             ELSE
                IF (PRESENT(EXHF_ACFDT)) THEN
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W,  &
                        NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                        WFOCK%CW(:,NPOS:,NK,ISP), P, CQIJ(1,1,1,1), CDCHF,  LSYMGRAD=LSYMGRAD, EXHF_ACFDT=EXHF_ACFDT )
                ELSE
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W,  &
                        NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                        WFOCK%CW(:,NPOS:,NK,ISP), P, CQIJ(1,1,1,1), CDCHF,  LSYMGRAD=LSYMGRAD)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       ENDDO
       IF (LSYMGRAD) &
            CALL APPLY_SMALL_SPACE_GROUP_OP( W, WFOCK, NONLR_S, NONL_S,P, T_INFO%NIONS, LATT_CUR, SYMM, CQIJ, .FALSE. , -1, MY_NKSTART)
    ENDIF
!=======================================================================
    MPLWV = GRID%MPLWV
    do II=1,NSTREAMBUFFERS
       call cublas_alloc_safety(WDES%NRPLWV,int(c_sizeof(COMPLEX_SIZE),c_size_t),GPU_CWWARRAY(II))
       call cublas_alloc_safety(MPLWV*WDES%NRSPINORS,int(c_sizeof(COMPLEX_SIZE),c_size_t),GPU_CRARRAY(II))
    enddo
    GPU%CWW = GPU_CWWARRAY(1)
    GPU%CR = GPU_CRARRAY(1)

!(sm): NGVECTORS carries the length of (e.g.) NINDPW per k-point. To allocate it
!      only once, we used the maximum length employed for all k-points
    maxNGVECTOR = MAXVAL(WDES%NGVECTOR)

    call cublas_alloc_safety (maxNGVECTOR,int(c_sizeof(WDES%NINDPW(1,1)),c_size_t),GPU%NINDPW)
    call cublas_alloc_safety (WDES%NGDIM*WDES%NRSPINORS,int(c_sizeof(WDES%DATAKE(1,1,1)),c_size_t),GPU%DATAKE)
    call cublas_alloc_safety (DIMREAL(MPLWV)*WDES%NRSPINORS*WDES%NRSPINORS,int(c_sizeof(SV(1,1)),c_size_t),GPU%SV)

!(sm): the following arrays are only required on the GPU if IFLAG /= 0
    IF ( IFLAG /= 0 ) THEN
!(sm): Find maximum required size to allocate device memory before entering the
!      spin/kpoint-loop. Originally this was done within the loop
       NSTRIP_RED = 0
       NPOS_RED   = 0
       DO NPOS=1,NBANDS,NSTRIP
               NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)
               if(NSTRIP_ACT > NSTRIP_RED) NSTRIP_RED = NSTRIP_ACT
               if(WDES%NB_PAR > NPOS_RED) NPOS_RED = WDES%NB_PAR
       ENDDO
       call cublas_alloc_safety(WDES%NRPLWV_RED*NSTRIP_RED*NPOS_RED,int(c_sizeof(WHAM%CW(1,1)),c_size_t),GPU%CFW)
       call cublas_alloc_safety(WDES%NPROD_RED*NSTRIP_RED*NPOS_RED, int(c_sizeof(WNONL%CPROJ_RED(1,1)),c_size_t),GPU%CPROW)
       call cublas_alloc_safety(WDES%NRPLWV_RED*NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_size_t),GPU%CPTWFP)
       call cublas_alloc_safety(WDES%NPROD_RED*NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_size_t),GPU%CPROJ)
       call cublas_alloc_safety(maxNGVECTOR,int(c_sizeof(COMPLEX_SIZE),c_size_t),GPU%CH)
       call cublas_alloc_safety(WDES%GRID%MPLWV*WDES%NRSPINORS*NSTREAMBUFFERS,int(c_sizeof(COMPLEX_SIZE),c_size_t),GPU_CWORK_ALL)
       DO II=1,NSTREAMBUFFERS
          call cublas_alloc_safety(WDES%NRPLWV,int(c_sizeof(COMPLEX_SIZE),c_size_t),GPU_CHARRAY(II))
          GPU_CWORK(II) = GPU_CWORK_ALL + (II-1)*WDES%GRID%MPLWV*WDES%NRSPINORS*sizeof(fakec)
       ENDDO
    ENDIF

    spin:  DO ISP=1,WDES%ISPIN
    kpoint: DO NK=MY_NKSTART,MY_NKSTOP
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
       IF (LscaAWARE_LOCAL) CALL INIT_scala(WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))

       CALL SETWDES(WDES,WDES1,NK)
!=======================================================================
!  IFLAG=0 calculate eigenvalues  only
!=======================================================================
       MPLWV     = WDES1%GRID%MPLWV
       NRSPINORS = WDES1%NRSPINORS  
       
       CALL VTIME(TV1,TC1)

       call cublas_Set_Vector(WDES1%NGVECTOR,int(c_sizeof(WDES1%NINDPW(1)),c_int),c_loc(WDES1%NINDPW),1,GPU%NINDPW,1)
       call cublas_Set_Vector (WDES1%NGDIM*NRSPINORS,int(c_sizeof(WDES1%DATAKE(1,1)),c_int),c_loc(WDES1%DATAKE),1,GPU%DATAKE,1)
       call cublas_Set_Matrix (DIMREAL(MPLWV),NRSPINORS*NRSPINORS,int(c_sizeof(SV(1,1)),c_int),c_loc(SV(1,ISP)),&
                       & DIMREAL(MPLWV),GPU%SV,DIMREAL(MPLWV)) 
       
       CALL VTIME(TV,TC)   
       TimeTrans = TimeTrans + TC - TC1  
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,5))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,6))
#endif
       IF (IFLAG==0) THEN
          W%CELEN(:,NK,ISP)=0
          DO N=1,NBANDS
             ! transform wavefunction to real space
             ! and calculate eigenvalues calling ECCP, no redistribution !
             CALL SETWAV(W, W1, WDES1, N, ISP) ! allocation for W1%CR done above
             !CALL FFTWAV_W1(W1)
             CALL VTIME(TV1,TC1)
             call cublas_Set_Vector(WDES1%NRPLWV,int(c_sizeof(COMPLEX_SIZE),c_int),c_loc(W1%CW),1,GPU%CWW,1)
             CALL VTIME(TV,TC)   
             TimeTrans = TimeTrans + TC - TC1
             
             CALL VTIME(TV1,TC1)
             !CALL FFTWAV_W1_GPU_X(W1)
             CALL FFTWAV_W1_GPU(W1,GPU%CWW, GPU%CR, GPU%NINDPW)
             CALL THREADSYNCHRONIZE ()
             CALL VTIME(TV,TC)
             TimeFFT = TimeFFT + TC - TC1
            
!            IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
!               CALL ECCP_VEC(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),HAMILTONIAN%AVEC, W1%CELEN)
             IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                CALL ECCP_TAU(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),W1%CELEN)    
             ELSE
             CALL VTIME(TV1,TC1)
             !CALL ECCP(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W1%CELEN)
             CALL ECCP_GPU(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),W1%CELEN,&
                           GPU%CR,GPU%CR,GPU%SV,0,GPU%CWW,GPU%CWW,GPU%DATAKE)
             ! JBNV, the below line was what was originally there, but I think the shift in this case is wrong
             ! thats why it is replaced by a shift of 0. This seems to solve a crash we noticed when ISPIN > 1 
             ! TODO verify that this is indeed the fix and works for all other tests. If so delete these comments
             !GPU%CR,GPU%CR,GPU%SV,IDX(1,ISP,DIMREAL(WDES1%GRID%MPLWV)),GPU%CWW,GPU%CWW,GPU%DATAKE)
             CALL THREADSYNCHRONIZE ()
             CALL VTIME(TV,TC)
             TimeECCP = TimeECCP + TC - TC1
             ENDIF
             IF (USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF))) THEN
                W1%CELEN=W1%CELEN+ &
                     W1_DOT( ELEMENT( W, WDES1, N, ISP), ELEMENT (WFOCK, WDES1, N, ISP))
             ENDIF

             W%CELEN(N,NK,ISP)=W1%CELEN
          ENDDO
 CALL VTIME(TV,TC)
 TimeOther = TimeOther + TC - TC1
! test
          CALL PEAD_EIGENVALUES(W,NK,ISP)
! test
          CYCLE kpoint
       ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,6))
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,7))
#endif

       WA=ELEMENTS(W, WDES1, ISP)
!=======================================================================
!  IFLAG /= 0 calculate Hamiltonian CHAM
!=======================================================================
       !  caclulate D |cfin_n> (D = non local strength of PP)
       IF (WDES%DO_REDIS .AND. LASYNC) THEN
          CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PW1)
          CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PW2)
          DO NPOS=1,NSTRIP
             CALL REDIS_PW_START(WDES, WA%CW(1,NPOS), NPOS, H_PW1)
          ENDDO
       ENDIF

       CALL OVERL(WDES1, .TRUE.,LMDIM,CDIJ(1,1,1,ISP), WA%CPROJ(1,1),WNONL%CPROJ(1,1))
       DO N=1,NBANDS
          CALL PEAD_ACC_ADD_CPROJ(WNONL%CPROJ(:,N),N,NK,ISP)
       ENDDO

       IF (IFLAG==5) THEN
          CALL OVERL(WDES1, WDES1%LOVERL,LMDIM,CQIJ(1,1,1,ISP), WA%CPROJ(1,1),WOVL%CPROJ(1,1))
       ENDIF

       ! redistribute the wavefunction characters
       CALL REDISTRIBUTE_PROJ(WA)
       CALL REDISTRIBUTE_PROJ(WNONL)
       IF (IFLAG==5) CALL REDISTRIBUTE_PROJ(WOVL)

       CHAM=0
 CALL VTIME(TV,TC)
 TimeOther = TimeOther + TC - TC1
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,7))
#endif




#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,8))
#endif



  ! Buffers for asynchronous MPI communication
  CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PWNVA1)
  CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PWNVA2)
  CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PWNVB1)
  CALL REDIS_PW_ALLOC(WDES, NSTRIP, H_PWNVB2)

  call cuda_memset(GPU%COVL,0,NB_TOT*NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t))
  !CALL gpu_initz(-1,NB_TOT*NB_TOT,NB_TOT*NB_TOT,GPU%COVL,0,zero)
 
  ! copy from host to device
  call cublas_Set_Matrix(WDES1%NRPLWV_RED,NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),c_loc(WA%CW_RED),WDES1%NRPLWV_RED,GPU%CPTWFP,WDES1%NRPLWV_RED)
  call cublas_Set_Matrix(WDES1%NPROD_RED,NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_int),c_loc(WA%CPROJ_RED),WDES1%NPROD_RED,GPU%CPROJ,WDES1%NPROD_RED)
          
       W1_COUNT = 0
       OLD_NPOS = 0
       STREAM   = 1
       !strip: DO NPOS=1,NBANDS,NSTRIP
       ! Note the +NSTRIP, we require one more iteration to finish
       ! the MPI communication of the last set
       strip: DO NPOS=1,NBANDS+NSTRIP,NSTRIP
       
#ifdef DEBUG_AND_WATCH
          call percent(NPOS,NBANDS)
#endif

          NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

          STREAM = MOD(STREAM,2) + 1
          
          if(MOD(W1_COUNT,2) == 0) then
            WHAM    = WHAM_PING
            H_PWNV1 = H_PWNVA1
            H_PWNV2 = H_PWNVA2
          else
            WHAM    = WHAM_PONG
            H_PWNV1 = H_PWNVB1
            H_PWNV2 = H_PWNVB2            
          endif

          if(NSTRIP_ACT > 0) then
          !  calculate V_{local} |phi> + T | phi >
          !  for a block containing NSTRIP wavefunctions
          ! set Fock contribution
          IF (USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF))) THEN
             WHAM%CW(:,1:NSTRIP_ACT)=WFOCK%CW(:,NPOS:NPOS+NSTRIP_ACT-1,NK,ISP)
          ENDIF

#ifdef MPI
                  IF (WDES%COMM_INTER%NCPU > 1) THEN
                    do i=1,NB_TOT
                      CW_RED_OLD(i) = WA%CW_RED(1,i)
                    enddo
                  ENDIF
#endif
            
            DO N=NPOS,NPOS+NSTRIP_ACT-1             
              IF (WDES%DO_REDIS)  CALL REDIS_PW_START(WDES, WA%CW(1,N), N, H_PWNV1)    !Testing this one
            ENDDO

            LOOP_COUNTER = 1
            DO N=NPOS,NPOS+NSTRIP_ACT-1
              NP=N-NPOS+1
              STREAM3 = (STREAM-1)*3 + MOD(LOOP_COUNTER,3) + 1

             !CALL SETWAV(W, W1, WDES1, N, ISP)
             call SETWAV(W,W1ARRAY(STREAM3), WDES1, N, ISP)
             call cuda_memcpyhtod(STREAM3,GPU_CHARRAY(STREAM3),c_loc(WHAM%CW(1,NP)),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
             call cuda_memcpyhtod(STREAM3,GPU_CWWARRAY(STREAM3),c_loc(W1ARRAY(STREAM3)%CW),WDES1%NRPLWV,int(c_sizeof(COMPLEX_SIZE),c_size_t))

            !CALL FFTWAV_W1(W1)
             CALL FFTWAV_W1_GPU_STREAM(STREAM3,W1ARRAY(STREAM3),GPU_CWWARRAY(STREAM3), GPU_CRARRAY(STREAM3),GPU%NINDPW)

             IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                 write(*,*) 'VASP ERROR: Not implemented on the GPU yet!'
                 call cuda_device_reset()
                 stop
              !CALL HAMILT_LOCAL_TAU(W1, SV, LATT_CUR, HAMILTONIAN%MU, ISP, WHAM%CW(:,NP), &
              ! &   USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF)), IFLAG/=12) 
             ELSE
              CALL HAMILT_LOCAL_GPU(STREAM3, W1ARRAY(STREAM3), GPU%SV, 1, USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF)),IFLAG/=12,GPU%DATAKE,GPU_CWWARRAY(STREAM3), GPU_CHARRAY(STREAM3), GPU_CRARRAY(STREAM3), GPU_CWORK(STREAM3), GPU%NINDPW)
                !CALL HAMILT_LOCAL(W1, SV, ISP, WHAM%CW(:,NP), USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF)), IFLAG/=12)
             ENDIF
             call cuda_memcpydtoh(STREAM3,c_loc(WHAM%CW(1,NP)),GPU_CHARRAY(STREAM3),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))

            !This is needed if we are using CUDA 5.0 ...
            !Uncomment the next line in case you are using CUDA 5.0 to force
            !synchronization and no overlapping streams
            !if(MOD(LOOP_COUNTER,3)+1 == 1)    call THREADSYNCHRONIZE()         
              
            LOOP_COUNTER = LOOP_COUNTER + 1
          ENDDO
          endif ! if N_ACT > 1 


          if(W1_COUNT > 0) then
            !Note we only get here after the first iteration, when there is
            !actually data to be received             
            
            if(NSTRIP_ACT <= 0) then
              ! Get the previous stream since current did not do anything
              STREAM = MOD(STREAM,2) + 1
            endif
            
            STREAM2 = NSTREAMBUFFERS + 1 !The ZGEMMS can be run in a different independent stream
            
              !Switch to previous storage
              if(MOD(W1_COUNT+1,2) == 0) then
                WHAM    = WHAM_PING
                H_PWNV1 = H_PWNVA1
                H_PWNV2 = H_PWNVA2
              else
                WHAM    = WHAM_PONG
                H_PWNV1 = H_PWNVB1
                H_PWNV2 = H_PWNVB2            
              endif
     

             !Wait for the MPI communication to complete
              DO N=OLD_NPOS,OLD_NPOS+OLD_NSTRIP_ACT-1
                  NP=N-OLD_NPOS+1
                  IF (WDES%DO_REDIS) CALL REDIS_PW_STOP (WDES, WA%CW(1,N), N,    H_PWNV1)
                  IF (WDES%DO_REDIS) CALL REDIS_PW_STOP (WDES, WHAM%CW(1,NP), N, H_PWNV2)
              ENDDO     

              NPOS_RED  =(OLD_NPOS-1)*WDES%NB_PAR+1
              NSTRIP_RED=OLD_NSTRIP_ACT*WDES%NB_PAR
              call cublas_Set_Matrix_Async(STREAM2, WDES1%NRPLWV_RED,NSTRIP_RED,int(c_sizeof(WHAM%CW(1,1)),c_int),c_loc(WHAM%CW),WDES1%NRPLWV_RED,GPU%CFW,WDES1%NRPLWV_RED)   
              call cublas_Set_Matrix_Async(STREAM2, WDES1%NPROD_RED,NSTRIP_RED,int(c_sizeof(WNONL%CPROJ_RED(1,1)),c_int),c_loc(WNONL%CPROJ_RED(1,NPOS_RED)),WDES1%NPROD_RED,GPU%CPROW,WDES1%NPROD_RED)
 
             !Needed to perform MPI (not used in sequential mode)
#ifdef MPI
#ifdef GPU_FAST_TRANSFERT
                    IF (WDES%COMM_INTER%NCPU > 1) THEN
                      j=0
                      do i=1,NB_TOT
                          ! Not the best way...
                          if (CW_RED_OLD(i) /= WA%CW_RED(1,i)) then
                            j=j+1
                          else
                            if (j /= 0) then
                              call cublas_set_matrix_async(STREAM2, &
                                    WDES1%NRPLWV_RED,j, &
                                    int(c_sizeof(WA%CW_RED(1,1)),c_int), &
                                    c_loc(WA%CW_RED(1,i-j)), &
                                    WDES1%NRPLWV_RED, &
                                    GPU%CPTWFP+IDX(1,i-j,WDES1%NRPLWV_RED)*int(c_sizeof(WA%CW_RED(1,1)),c_size_t), &
                                    WDES1%NRPLWV_RED)
                              j=0
                            endif
                          endif
                      enddo
                      if (j /= 0) then
                         call cublas_set_matrix_async(STREAM2, &
                             WDES1%NRPLWV_RED,j, &
                             int(c_sizeof(WA%CW_RED(1,1)),c_int), &
                             c_loc(WA%CW_RED(1,i-j)), &
                             WDES1%NRPLWV_RED, &
                             GPU%CPTWFP+IDX(1,i-j,WDES1%NRPLWV_RED)*int(c_sizeof(WA%CW_RED(1,1)),c_size_t), &
                             WDES1%NRPLWV_RED)
                      !print *, "row",i-j,"to",i-1
                      endif          
                    ENDIF
#else
                    call cublas_set_matrix_async(STREAM2,WDES1%NRPLWV_RED,NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),c_loc(WA%CW_RED),WDES1%NRPLWV_RED,GPU%CPTWFP,WDES1%NRPLWV_RED)
#endif
#endif

          ! redistribute wavefunctions
          ! after this redistributed up to and including 1...NPOS+NSTRIP_ACT
          !IF (WDES%DO_REDIS) THEN
          !   IF (LASYNC) THEN
          !      DO N=NPOS,NPOS+NSTRIP_ACT-1
          !         NP=N-NPOS+1
          !         CALL REDIS_PW_STOP (WDES, WA%CW(1,N), N, H_PW1)
          !         IF (N+NSTRIP<=NBANDS) &
          !              CALL REDIS_PW_START(WDES, WA%CW(1,N+NSTRIP), N+NSTRIP, H_PW1)
          !         CALL REDIS_PW_STOP (WDES, WHAM%CW(1,NP), N, H_PW2)
          !      ENDDO
          !   ELSE
          !      CALL REDISTRIBUTE_PW( ELEMENTS( WA, NPOS, NPOS-1+NSTRIP_ACT))
          !      CALL REDISTRIBUTE_PW( ELEMENTS( WHAM, 1, NSTRIP_ACT))
          !   ENDIF
          !ENDIF
          !NPOS_RED  =(NPOS-1)*WDES%NB_PAR+1
          !NSTRIP_RED=NSTRIP_ACT*WDES%NB_PAR
          IF (.NOT. LscaAWARE_LOCAL) THEN
             !CALL ORTH1('U', &
             !  WA%CW_RED(1,1),WHAM%CW(1,1),WA%CPROJ_RED(1,1), &
             !  WNONL%CPROJ_RED(1,NPOS_RED),NB_TOT, &
             !  NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1))
             !CALL ORTH1_GPU('U', &
              CALL ORTH1_GPU_STREAM(STREAM2, 'U', &
                    GPU%CPTWFP,GPU%CFW,GPU%CPROJ, &
                    GPU%CPROW, NB_TOT, &
                    NPOS_RED, NSTRIP_RED, WDES1%NPL_RED, WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,GPU%COVL)
             !CALL THREADSYNCHRONIZE ()            
             !############################## GPU ##############################
         ELSE
             write(*,*) "ORTH1_DISTRI Not Yet Implemented !!!"
             call exit()
             !CALL ORTH1_DISTRI('U', &
             !  WA%CW_RED(1,1),WHAM%CW(1,1),WA%CPROJ_RED(1,1), &
             !  WNONL%CPROJ_RED(1,NPOS_RED),NB_TOT, &
             !  NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1), & 
             !  WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))
          ENDIF

#ifdef TAU_PROF
              call nvtx_range_pop()
              call TAU_PROFILE_STOP(profile(1,18))
#endif     
          endif !W1 count > 1

         if(NSTRIP_ACT > 0) then
#if 1
              !Sync moved to the end for large models the GPU is sufficiently busy that waiting takes relative long
              !Note we need to modify the buffers again incase they got changed by the zgemm start part here above
              STREAM3 = (STREAM-1)*3 + MOD(1,3) + 1
              call cuda_streamsynchronize(STREAM3)
              STREAM3 = (STREAM-1)*3 + MOD(2,3) + 1
              call cuda_streamsynchronize(STREAM3)
              STREAM3 = (STREAM-1)*3 + MOD(3,3) + 1
              call cuda_streamsynchronize(STREAM3)
              
              if(MOD(W1_COUNT,2) == 0) then
                WHAM    = WHAM_PING
                H_PWNV1 = H_PWNVA1
                H_PWNV2 = H_PWNVA2
              else
                WHAM    = WHAM_PONG
                H_PWNV1 = H_PWNVB1
                H_PWNV2 = H_PWNVB2            
              endif
              DO N=NPOS,NPOS+NSTRIP_ACT-1
                NP=N-NPOS+1      
                CALL PEAD_ACC_ADD_PW(WHAM%CW(:,NP),N,NK,ISP)
                IF (WDES%DO_REDIS)  CALL REDIS_PW_START(WDES, WHAM%CW(1,NP), N, H_PWNV2) !Works
              ENDDO         

#ifdef TAU_PROF
              call TAU_PROFILE_STOP(profile(1,15))
#endif                           
#endif
           endif ! if(NSTRIP_ACT > 0)
          W1_COUNT       = W1_COUNT + 1         
          OLD_NPOS       = NPOS
          OLD_NSTRIP_ACT = NSTRIP_ACT

       ENDDO strip

   call THREADSYNCHRONIZE() ! Make sure all previous async kernels and memory copies have finished
#ifdef timing
       CALL STOP_TIMING("S",IU6,"SETUP")
#endif
      
      CALL REDIS_PW_DEALLOC(H_PWNVA1)
      CALL REDIS_PW_DEALLOC(H_PWNVA2)       
      CALL REDIS_PW_DEALLOC(H_PWNVB1)
      CALL REDIS_PW_DEALLOC(H_PWNVB2)       

       call cublas_Get_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_int),GPU%COVL,NB_TOT,c_loc(CHAM),NB_TOT)  

#ifdef DEBUG_AND_WATCH    
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*) " "
 write(*,*) "       Function Name        ","      Time (s)       ","      percentage"
 write(*,*) "TIME TRANSFER...........=",TimeTrans,nint(TimeTrans/(TC-TC2)*100)
 write(*,*) "TIME FFT................=",TimeFFT,nint(TimeFFT/(TC-TC2)*100)
 write(*,*) "TIME ORTH...............=",TimeORTH,nint(TimeORTH/(TC-TC2)*100)
 write(*,*) "TIME ECCP...............=",TimeECCP,nint(TimeECCP/(TC-TC2)*100)
 write(*,*) "TIME HAMIL..............=",TimeHamil,nint(TimeHamil/(TC-TC2)*100)
 write(*,*) "OTHER...................=",TimeOther,nint(TimeOther/(TC-TC2)*100)
 write(*,*) "Total percent...........=",nint((TimeTrans+TimeFFT+TimeHamil+TimeOther+TimeORTH+TimeECCP)/(TC-TC2)*100)
 write(*,*) "Total time............................................=",TC-TC2
#ifdef MPI
 endif
#endif
#endif
       TimeTrans = 0
       TimeFFT   = 0
       TimeORTH  = 0
       TimeECCP  = 0
       TimeHamil = 0
       TimeOther = 0
       IF (WDES%DO_REDIS .AND. LASYNC) THEN
          CALL REDIS_PW_DEALLOC(H_PW1)
          CALL REDIS_PW_DEALLOC(H_PW2)
       ENDIF

       statmg
       IF (.NOT. LscaAWARE_LOCAL) THEN
          CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM(1,1),NB_TOT*NB_TOT))
          ! add lower triangle
          DO N=1,NB_TOT
             DO NP=N+1,NB_TOT
                CHAM(NP,N)=GCONJG(CHAM(N,NP))
             ENDDO
          ENDDO
       ENDIF

       IF (PRESENT(CHAMHF).AND.PRESENT(LFIRST)) THEN
          IF (LFIRST) THEN
             CHAMHF(:,:,NK,ISP)=GCONJG(CHAMHF(:,:,NK,ISP))-CHAM(:,:)
          ENDIF
          CHAM(:,:)=CHAM(:,:)+CHAMHF(:,:,NK,ISP)
       ENDIF
#ifdef debug
       IF (IU0>=0) CALL DUMP_HAM( "Hamilton matrix subrot",WDES, CHAM)
#endif
!-----------------------------------------------------------------------
! calculate the overlap matrix
!-----------------------------------------------------------------------
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,11))
#endif
       IF (IFLAG==5) THEN
          !COVL=(0._q,0._q)
          write(*,*) "option IFLAG == 5, Not yet on GPU"
          stop
          call cuda_memset(GPU%COVL,0,NB_TOT*NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t))
          !CALL gpu_initz(-1,NB_TOT*NB_TOT,NB_TOT*NB_TOT,GPU%COVL,0,zero)

          DO NPOS=1,NB_TOT-NSTRIP_STANDARD_GLOBAL,NSTRIP_STANDARD_GLOBAL
             CALL ORTH1('U',WA%CW_RED(1,1),WA%CW_RED(1,NPOS),WA%CPROJ_RED(1,1), &
                  WOVL%CPROJ_RED(1,NPOS),NB_TOT, &
                  NPOS,NSTRIP_STANDARD_GLOBAL,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
          ENDDO

          CALL ORTH1('U',WA%CW_RED(1,1),WA%CW_RED(1,NPOS),WA%CPROJ_RED(1,1), &
               WOVL%CPROJ_RED(1,NPOS),NB_TOT, &
               NPOS,NB_TOT-NPOS+1,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
          CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "Overlap matrix",WDES, COVL)
#endif
       ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,11))
#endif
#ifdef timing
       CALL STOP_TIMING("S",IU6,"GLBSUM")
#endif
!=======================================================================
! IFLAG =2
!  simply copy eigenvalues
!=======================================================================
       IF (IFLAG==12) THEN
          IF (IU0>=0) CALL DUMP_HAM( "Hamilton matrix",WDES, CHAM)
          STOP
       ENDIF

       IF (.NOT. LscaAWARE_LOCAL) THEN
          DO N=1,WDES%NB_TOTK(NK,ISP)
             W%CELTOT(N,NK,ISP)=CHAM(N,N)
          ENDDO
       ENDIF
!=======================================================================
! IFLAG =4 use Loewdin perturbation to get rotation matrix
! this preserves the ordering of the eigenvalues
! MIND: does not work for real matrices
!=======================================================================
       IF (IFLAG==4) THEN
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,12))
#endif
          CALL LOEWDIN_DIAG(WDES%NB_TOTK(NK,ISP), NB_TOT, CHAM)
          CALL ORSP(WDES%NB_TOTK(NK,ISP), NB_TOT, NB_TOT, CHAM)
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,12))
#endif
       ELSE
!=======================================================================
! IFLAG > 1 and IFLAG <4
! diagonalization of CHAM
! we have lots of choices for the parallel version
! this  makes things rather complicated
! to allow for reasonable simple programming, once the diagonalisation
! has been done I jump to line 100
!=======================================================================
          IF (IFLAG==1) GOTO 1000

#ifndef gammareal
          DO N=1,NB_TOT
             IF (.NOT. LscaAWARE_LOCAL) THEN
                IF (ABS(AIMAG(CHAM(N,N)))>1E-2_q .AND. IU0>=0) THEN
                   WRITE(IU0,'(A,I5,E14.3)')'WARNING in EDDIAG: sub space matrix is not hermitian',N,AIMAG(CHAM(N,N))
                ENDIF
                CHAM(N,N)= REAL( CHAM(N,N) ,KIND=q)
             ELSE
                CALL BG_CHANGE_DIAGONALE(WDES%NB_TOTK(NK,ISP),CHAM(1,1),IU0)
             ENDIF
          ENDDO
#endif

          !
          ! parallel versions
          ! if fast Jacobi method exists use it (T3D, T3E only)
          ! use the first line in that case
          stotmg

          IFAIL=0

#if defined(MPI)

          IF ( LJACOBI .AND. IFLAG /=13 ) THEN
             IF (IU0>=0) WRITE(IU0,*)'jacoby called'
             CALL jacDSSYEV(WDES%COMM_KIN, CHAM(1,1), R, NB_TOT)
             CALLMPI( M_sum_g(WDES%COMM_KIN, CHAM(1,1),NB_TOT*NB_TOT))
             CALLMPI( M_sum_g(WDES%COMM_KIN, R , NB_TOT))

             GOTO 100
          ENDIF

          ! use scaLAPACK if available in parallel version
          IF ( LscaLAPACK .AND. IFLAG /=5 ) THEN
             IF (.NOT. LscaAWARE_LOCAL) THEN
                CALL pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM(1,1), R,  NB_TOT, WDES%NB_TOTK(NK,ISP))
                CALLMPI( M_sum_g(WDES%COMM_KIN, CHAM(1,1),NB_TOT*NB_TOT))
             ELSE
                CALL BG_pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM(1,1), R,  WDES%NB_TOTK(NK,ISP))
             ENDIF
             stotmg
             GOTO 100
          ENDIF

#endif

CALL VTIME(TV1,TC1)
#ifdef  gammareal
! gammareal
          IF (IFLAG == 5) THEN
#ifdef USE_MAGMA
             USED_MAGMA=1
             write(*,*) "WARNING: have not tested magma_dsygvd yet!"
             call magma_dsygvd(1,'V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT,COVL(1,1),NB_TOT,R,IFAIL)
             CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t),COVL,NB_TOT,GPU%COVL,NB_TOT)
#else
             CALL DSYGV &
                  (1,'V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT,COVL(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, IFAIL)
#endif
          ELSE
#ifndef USE_ZHEEVX
#ifdef USE_MAGMA
             USED_MAGMA=1
            call cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_size_t),CHAM,NB_TOT,GPU%COVL,NB_TOT)
             write(*,*) "WARNING: have not tested magma_dsyevd_gpu yet!"
             call magma_dsyevd_gpu('V','U',WDES%NB_TOTK(NK,ISP),GPU%COVL,NB_TOT,R,IFAIL)
#else
             CALL DSYEV &
                  ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, IFAIL)
#endif
#else
             ABSTOL=1E-10_q
             VL=0 ; VU=0 ; IL=0 ; IU=0
#ifdef USE_MAGMA
             USED_MAGMA=1
            call cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_size_t),CHAM,NB_TOT,GPU%COVL,NB_TOT)
             write(*,*) "WARNING: have not tested magma_dsyevdx_gpu yet!"
             call magma_dsyevdx_gpu('V','A','U',WDES%NB_TOTK(NK,ISP),GPU%COVL,NB_TOT,VL,VU,IL,IU,NB_CALC,R,IFAIL)
#else
             ALLOCATE(COVL(NB_TOT,NB_TOT))
             CALL DSYEVX( 'V', 'A', 'U', WDES%NB_TOTK(NK,ISP), CHAM(1,1) , NB_TOT, VL, VU, IL, IU, &
                  ABSTOL , NB_CALC , R, COVL(1,1), NB_TOT, CWRK, &
                  LWORK*NB_TOT, RWORK, IWORK, INFO, IFAIL )         
             CHAM=COVL
#endif
             DEALLOCATE(COVL)
#endif
          ENDIF
#else
! not gammareal
          IF (IFLAG == 5) THEN
#ifdef USE_MAGMA
             USED_MAGMA=1
             write(*,*) "WARNING: have not tested magma_zheevd yet!"
             call magma_zhegvd(1,'V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT,COVL(1,1),NB_TOT,R,IFAIL)
             CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(COVL(1,1)),c_size_t),COVL,NB_TOT,GPU%COVL,NB_TOT)
#else
             CALL ZHEGV &
                  (1,'V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT,COVL(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, RWORK,  IFAIL)
#endif
          ELSE
#ifndef USE_ZHEEVX
#ifdef USE_MAGMA
             USED_MAGMA=1
             CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_size_t),CHAM,NB_TOT,GPU%COVL,NB_TOT)
             write(*,*) "WARNING: have not tested magma_zheevd_gpu yet!"
             call magma_zheevd_gpu('V','U',WDES%NB_TOTK(NK,ISP),GPU%COVL,NB_TOT,R,IFAIL)
#else
             CALL ZHEEV &
                  ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, RWORK,  IFAIL)
#endif
#else
             ABSTOL=1E-10_q
             VL=0 ; VU=0 ; IL=0 ; IU=0
#ifndef USE_MAGMA
nv_profile_start(NVP_EDDIAG_ZHEEVD)
             ALLOCATE(COVL(NB_TOT,NB_TOT))
             CALL ZHEEVX( 'V', 'A', 'U', WDES%NB_TOTK(NK,ISP), CHAM(1,1) , NB_TOT, VL, VU, IL, IU, &
                  ABSTOL , NB_CALC , R, COVL(1,1), NB_TOT, CWRK, &
                  LWORK*NB_TOT, RWORK, IWORK, INFO, IFAIL )         
             CHAM=COVL
             DEALLOCATE(COVL)
nv_profile_stop(NVP_EDDIAG_ZHEEVD)
#else
             USED_MAGMA = 1
nv_profile_start(NVP_EDDIAG_MEMCPY)
             CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_size_t),CHAM,NB_TOT,GPU%COVL,NB_TOT)
nv_profile_stop(NVP_EDDIAG_MEMCPY)
nv_profile_start(NVP_EDDIAG_ZHEEVD)
             !call magma_zheevd_gpu('V','U',WDES%NB_TOTK(NK,ISP),GPU%COVL,NB_TOT,R,IFAIL)
             call magma_zheevdx_gpu('V','A','U',WDES%NB_TOTK(NK,ISP),GPU%COVL,NB_TOT,VL,VU,IL,IU,NB_CALC,R,IFAIL)
nv_profile_stop(NVP_EDDIAG_ZHEEVD)
#endif
#endif
          ENDIF
#endif
#ifdef DEBUG_AND_WATCH
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*) "LAPACK TIME...........................................=",TC-TC1
#ifdef MPI
 endif
#endif
#endif
          ! T3D uses a global sum which does not guarantee to give the same results on all nodes
          ! the following line is required to make the code waterproof (we had problems)
          ! since we now use a propritary sum (see mpi.F) we should not require
          ! this broadcast anymore
          ! stotmg
          ! CALLMPI( M_bcast_g(WDES%COMM, CHAM(1,1), NB_TOT*NB_TOT))

100       CONTINUE
          stotmg

          IF (IFAIL/=0) THEN
             WRITE(0,*) 'ERROR in EDDIAG: call to ZHEEV/ZHEEVX/DSYEV/DSYEVX failed! '// &
                  &              'error code was ',IFAIL
             STOP
          ENDIF

          DO N=1,WDES%NB_TOTK(NK,ISP)
             W%CELTOT(N,NK,ISP)=R(N)
          ENDDO
       ENDIF
#ifdef timing
       CALL STOP_TIMING("S",IU6,"DIAG")
#endif
!=======================================================================
! IFLAG > 2
! rotate wavefunctions
!=======================================================================
       IF (IFLAG==2) GOTO 1000
#ifdef TAU_PROF
    call TAU_PROFILE_START(profile(1,13))
#endif

       if (USED_MAGMA == 0) then
          !No need to copy if data is already on GPU because of magma call instead of lapack
          CALL cublas_Set_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_int),c_loc(CHAM),NB_TOT,GPU%COVL,NB_TOT)
       endif
       IF (.NOT. LscaAWARE_LOCAL) THEN
          !CALL LINCOM('F',WA%CW_RED(:,:),WA%CPROJ_RED(:,:),CHAM(1,1), &
            !WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
            !WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
            !WA%CW_RED(:,:),WA%CPROJ_RED(:,:))
          CALL LINCOM_GPU('F',GPU%CPTWFP,GPU%CPROJ,GPU%COVL, &
            WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
            WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
            GPU%CPTWFP,GPU%CPROJ)
          CALL THREADSYNCHRONIZE ()
       ELSE
          write(*,*) "LINCOM_DISTRI_GPU Not Yet Implemented !!!"
          CALL exit()
          !CALL LINCOM_DISTRI('F',WA%CW_RED(1,1),WA%CPROJ_RED(1,1),CHAM(1,1), &
          !  WDES%NB_TOTK(NK,ISP), & 
          !  WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
          !  WDES%COMM_KIN, NBLK )
       ENDIF
       CALL cublas_Get_Matrix(WDES1%NRPLWV_RED,NB_TOT,int(c_sizeof(WA%CW_RED(1,1)),c_int),GPU%CPTWFP,WDES1%NRPLWV_RED,c_loc(WA%CW_RED),WDES1%NRPLWV_RED)
       IF (WDES1%NPRO_RED /= 0) THEN
          CALL cublas_Get_Matrix(WDES1%NPROD_RED,NB_TOT,int(c_sizeof(WA%CPROJ_RED(1,1)),c_int),GPU%CPROJ,WDES1%NPROD_RED,c_loc(WA%CPROJ_RED),WDES1%NPROD_RED)
       ENDIF
       CALL cublas_Get_Matrix(NB_TOT,NB_TOT,int(c_sizeof(CHAM(1,1)),c_int),GPU%COVL,NB_TOT,c_loc(CHAM),NB_TOT)
       
 CALL VTIME(TV,TC)  
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,13))
#endif

#ifdef DEBUG_AND_WATCH 
#ifdef MPI
 if (WDES%COMM%NODE_ME == 1) then
#endif
 write(*,*) "LINCOM_GPU Time.......................................=",TC-TC1
 write(*,*) "                        # # # # # # # # # # #                         "
#ifdef MPI
 endif
#endif
#endif

#ifdef timing
       CALL STOP_TIMING("S",IU6,"LINCOM")
#endif

1000   CONTINUE
       
       !  back redistribution over bands
       IF (WDES%DO_REDIS) THEN
          CALL REDISTRIBUTE_PROJ( ELEMENTS( W, WDES1, ISP))
          IF (LASYNC) THEN
             W%OVER_BAND=.TRUE.
          ELSE
             CALL REDISTRIBUTE_PW( ELEMENTS( W, WDES1, ISP))
          ENDIF
          DWRITE "redis ok"
       ENDIF

       ! return Hamiltonian (IFLAG==1) or rotation matrix (IFLAG=3,4)
       IF (PRESENT(LLAST)) THEN       
          IF (LLAST) THEN
             IF (IFLAG==1) CHAM=GCONJG(CHAM)
             CHAMHF(:,:,NK,ISP)=CHAM(:,:)
          ENDIF
       ENDIF
#ifdef timing
       CALL STOP_TIMING("S",IU6,"DISTRI")
#endif
!###########################################################################
! 
      !call cublas_Free(GPU%CR)
      !call cublas_Free(GPU%CVR)
      !call cublas_Free(GPU%SV)
!
!###########################################################################     

    ENDDO kpoint
    ENDDO spin


    do II=1,NSTREAMBUFFERS
       call cublas_free(GPU_CWWARRAY(II))
       call cublas_free(GPU_CRARRAY(II))
    enddo

    call cublas_free(GPU%NINDPW)
    call cublas_free(GPU%DATAKE)
    call cublas_free(GPU%SV)

    IF ( IFLAG /= 0 ) THEN
    do II=1,NSTREAMBUFFERS
       call cublas_free(GPU_CHARRAY(II))
    enddo
    call cublas_free(GPU%CFW)
    call cublas_free(GPU%CPROW)
    call cublas_free(GPU%CPTWFP)
    call cublas_free(GPU%CPROJ)
    call cublas_free(GPU_CWORK_ALL)
    call cublas_free(GPU%CH)
    ENDIF


#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,3))
    call TAU_PROFILE_START(profile(1,4))
#endif


    CALLMPI( M_sum_z(WDES%COMM_KIN,CDCHF,1))
    CALLMPI( M_sum_z(WDES%COMM_KINTER,CDCHF,1))
    EXHF=CDCHF

    IF (PRESENT(EXHF_ACFDT)) THEN
       CALLMPI( M_sum_d(WDES%COMM_KIN,EXHF_ACFDT,1))
       CALLMPI( M_sum_d(WDES%COMM_KINTER,EXHF_ACFDT,1))
    ENDIF

    ! need to correct CELTOT at this point so that it is correct on all nodes
    IF (IFLAG==0) CALL MRG_CEL(WDES,W)
#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_CELTOT(WDES,W)
    ENDIF

    IF (IFLAG>=3.AND.(LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_ALL(WDES,W)
    ENDIF
#endif

    ! deallocation ...
    DEALLOCATE(CHAM,W1%CR)
    IF (.NOT. LscaAWARE_LOCAL) &
      CALL cublas_free(GPU%COVL)
    
    !CALL DELWAVA(WHAM)
    CALL DELWAVA(WHAM_PING)
    CALL DELWAVA(WHAM_PONG)
    CALL DELWAVA_PROJ(WNONL)
    IF (USEFOCK_CONTRIBUTION().AND.(.NOT.PRESENT(CHAMHF))) THEN 
       CALL DEALLOCW(WFOCK)
       IF (LPEAD_NO_SCF()) THEN
          DO N=1,NSIM_LOCAL
             CALL DELWAV(WTMP(N), .FALSE.)
          ENDDO
       ENDIF
    ENDIF

    IF (IFLAG==5) THEN
       DEALLOCATE(COVL)
       CALL DELWAVA_PROJ(WOVL)
    ENDIF

    IF (PRESENT(CHAMHF).AND.PRESENT(LFIRST)) THEN
       LFIRST=.FALSE.
    ENDIF
    CALL VTIME(TV,TC)

nv_profile_stop(NVP_EDDIAG)

#ifdef DEBUG_AND_WATCH
#ifdef MPI
    if (WDES%COMM%NODE_ME == 1) then
#endif
      write(*,*) "Time EDDIAG...........................................=",TC-TC0
      write(*,*) "######################################################################"
#ifdef MPI
    endif
#endif
#endif
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,4))
    call TAU_PROFILE_STOP(profile(1,1))
#endif
    RETURN

  END SUBROUTINE EDDIAG

!***********************************************************************
!
!  for a selected k-points and spin component
! ) calculate onsite (one centre) contribution to the Hamilton matrix
!  if LOVERL is .TRUE.
!
!    CCORR(m,n) += <psi_k,n| beta_i> D_ij <beta_j | psi_k,m> 
!
! ) if the local potential SV is passed  down the Hamilton matrix
!
!    CCORR(m,n) += <psi_k,n|  SV | psi_k,m> 
!
!  is added as well
!  the calling routine must initialize CCORR to zero
!  
!***********************************************************************

  SUBROUTINE ONE_CENTER_BETWEEN_STATES(HAMILTONIAN, LATT_CUR, LOVERL, WDES, W, NK, ISP, LMDIM, &
       CDIJC, CCORR, SV)
    USE prec
    USE wave_high
    USE dfast
    USE lattice
    USE hamil_high

    TYPE (ham_handle)  HAMILTONIAN
    TYPE (latt)        LATT_CUR
    LOGICAL LOVERL
    INTEGER NK, ISP, LMDIM
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    OVERLAP CDIJC(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)  ! one centre correction
    GDEF ::  CCORR(WDES%NB_TOT,WDES%NB_TOT)
    RGRID, OPTIONAL  :: SV(DIMREAL(WDES%GRID%MPLWV),WDES%NCDIJ)   ! local potential
  ! local
    GDEF   , POINTER :: CPROJ_RED(:,:)
    INTEGER NPOS
    INTEGER NSTRIP, NSTRIP_ACT, NPOS_RED, NSTRIP_RED
    TYPE (wavefuna)    WNONL          ! array to hold non local part D * wave function character
    TYPE (wavefuna)    WOVL           ! array to hold non local part
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1             ! current wavefunction
    TYPE (wavefuna)    WHAM           ! store Hamiltonian times wavefunction
    TYPE (wavefuna)    WA             ! array to store wavefunction

    NSTRIP=NSTRIP_STANDARD
!-----------------------------------------------------------------------
! non local part
!-----------------------------------------------------------------------
    IF (LOVERL) THEN
       CALL SETWDES(WDES,WDES1,NK)

       CALL NEWWAVA_PROJ(WNONL, WDES1)
       IF (WDES%DO_REDIS) THEN
          CALL NEWWAVA_PROJ(WOVL, WDES1)
          CALL WA_COPY_CPROJ(ELEMENTS(W, WDES1, ISP), WOVL)
       ELSE
          WOVL=ELEMENTS(W, WDES1, ISP)
       ENDIF
       
       CALL OVERL(WDES1, .TRUE., LMDIM, CDIJC(1,1,1,ISP), WOVL%CPROJ(1,1), WNONL%CPROJ(1,1))
       
       IF (WDES%DO_REDIS) THEN
          CALL REDIS_PROJ(WDES1, WDES%NBANDS, WNONL%CPROJ(1,1))
          CALL REDIS_PROJ(WDES1, WDES%NBANDS, WOVL%CPROJ (1,1))
       ENDIF
       
       DO NPOS=1,WDES%NBANDS,NSTRIP
          NSTRIP_ACT=MIN(WDES%NBANDS+1-NPOS,NSTRIP)
          NPOS_RED  =(NPOS-1)*WDES%NB_PAR+1
          NSTRIP_RED=NSTRIP_ACT*WDES%NB_PAR
          
          CALL ORTH1('U', &
               W%CW(1,1,NK,ISP),W%CW(1,1,NK,ISP),WOVL%CPROJ(1,1), &
               WNONL%CPROJ_RED(1,NPOS_RED),WDES%NB_TOT, &
               NPOS_RED, NSTRIP_RED, 0,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CCORR(1,1))
!             attention              - 

          
       ENDDO
       CALL DELWAVA_PROJ(WNONL)
       IF (WDES%DO_REDIS) CALL DELWAVA_PROJ(WOVL)
    ENDIF
!-----------------------------------------------------------------------
! local part
!-----------------------------------------------------------------------
    IF (PRESENT(SV)) THEN
       ! allocate work space
       ALLOCATE(W1%CR(WDES%GRID%MPLWV*WDES%NRSPINORS))

       CALL SETWDES(WDES,WDES1,NK)
       CALL NEWWAVA(WHAM, WDES1, NSTRIP)

       WA=ELEMENTS(W, WDES1, ISP)

       strip: DO NPOS=1,WDES%NBANDS,NSTRIP
          NSTRIP_ACT=MIN(WDES%NBANDS+1-NPOS,NSTRIP)

          !  calculate V_{local} |phi> + T | phi >
          !  for a block containing NSTRIP wavefunctions
          DO N=NPOS,NPOS+NSTRIP_ACT-1
             NP=N-NPOS+1

             CALL SETWAV(W, W1, WDES1, N, ISP)
             CALL FFTWAV_W1_GPU_X(W1)
             !CALL FFTWAV_W1(W1)
             IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                CALL HAMILT_LOCAL_TAU(W1, SV, LATT_CUR, HAMILTONIAN%MU, ISP,  WHAM%CW(:,NP), .FALSE., .FALSE.)
             ELSE
                CALL HAMILT_LOCAL(W1, SV, ISP,  WHAM%CW(:,NP), .FALSE., .FALSE.)
             ENDIF
          ENDDO
          ! redistribute wavefunctions
          ! after this redistributed up to and including 1...NPOS+NSTRIP_ACT
          IF (WDES%DO_REDIS) THEN
             CALL REDISTRIBUTE_PW( ELEMENTS( WA, NPOS, NPOS-1+NSTRIP_ACT))
             CALL REDISTRIBUTE_PW( ELEMENTS( WHAM, 1, NSTRIP_ACT))
          ENDIF

          NPOS_RED  =(NPOS-1)*WDES%NB_PAR+1
          NSTRIP_RED=NSTRIP_ACT*WDES%NB_PAR

          CALL ORTH1('U', &
               WA%CW_RED(1,1),WHAM%CW(1,1),WA%CPROJ_RED(1,1), &
               WA%CPROJ_RED(1,NPOS_RED),WDES%NB_TOT, &
               NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CCORR(1,1))
          !           attention                   ---

       ENDDO strip

       IF (WDES%DO_REDIS) CALL REDISTRIBUTE_PW( ELEMENTS( W, WDES1, ISP))

       ! deallocation ...
       DEALLOCATE(W1%CR)
       CALL DELWAVA(WHAM)

    ENDIF

    CALLMPI( M_sum_g(WDES%COMM_KIN,CCORR(1,1),WDES%NB_TOT*WDES%NB_TOT))

  END SUBROUTINE ONE_CENTER_BETWEEN_STATES


!***********************************************************************
!
! read in the file GAMMA and add the density matrix to the
! diagonal density matrix F (one-electron occupancies)
! then diagonalize the resulting matrix and rotate the
! one electron orbitals
!  
!***********************************************************************

  SUBROUTINE ADD_GAMMA_FROM_FILE( WDES, W, KPOINTS, NELECT, NUP_DOWN, LWRITE, IO )
    USE prec
    USE wave_high
    USE mkpoints
    USE dfast
    USE base
    USE fileio
    IMPLICIT NONE
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    REAL (q)  NELECT, NUP_DOWN        ! number of electrons, and total spin
    TYPE (kpoints_struct) KPOINTS     ! k-points structure
    LOGICAL :: LWRITE                 ! write final eigenvalues to OUTCAR
    TYPE (in_struct)   IO
  ! local
    INTEGER NK, ISP, NB, LMDIM
    TYPE (wavedes1)    WDES1
    TYPE (wavefuna)    WA             ! array to store wavefunction
    GDEF, ALLOCATABLE :: CHAM(:,:)
    REAL(q), ALLOCATABLE :: CELTOT(:,:,:)
    INTEGER :: IMODE                  ! add to Hamiltonian (IMODE=2) or density matrix (IMODE=1)
  ! LAPACK    
    REAL(q)    R(WDES%NB_TOT)
    INTEGER :: IFAIL
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*WDES%NB_TOT)
    REAL(q)    RWORK(3*WDES%NB_TOT)

    CALL OPENGAMMA
    CALL READGAMMA_HEAD( IMODE, WDES%NKPTS , WDES%NB_TOT, IO)

    ALLOCATE(CHAM(WDES%NB_TOT, WDES%NB_TOT))

  ! for IMODE = 2 we save the CELTOT locally
    IF (IMODE==2) THEN
       ALLOCATE(CELTOT(SIZE(W%CELTOT,1),SIZE(W%CELTOT,2),SIZE(W%CELTOT,3)))
       CELTOT=W%CELTOT
    ENDIF


    DO ISP=1,WDES%ISPIN
    DO NK=1,WDES%NKPTS

       CHAM=0
       CALL READGAMMA(NK, WDES%NB_TOTK(NK,ISP), SIZE(CHAM,1), CHAM,  IO)
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
       ! add one-particle occupancies
       IF (IMODE==1) THEN
          DO NB=1, WDES%NB_TOTK(NK,ISP)
             CHAM(NB,NB)=CHAM(NB,NB)+W%FERTOT(NB,NK,ISP)
          ENDDO
       ! add one-particle eigenvalues
       ELSE
          DO NB=1, WDES%NB_TOTK(NK,ISP)
             CHAM(NB,NB)=CHAM(NB,NB)+W%CELTOT(NB,NK,ISP)
          ENDDO
       ENDIF
#ifdef debug
       IF (IO%IU0>=0) CALL DUMP_HAM( "Gamma",WDES, CHAM)
#endif
       ! diagonalize the one-particle matrix (or Hamiltonian matrix)
       ! do not use the X routines, iterative diagonalization is not save
       ! for density matrix
       IFAIL=0
       
       ! change sign of density matrix to sort occupied states as lowest states
       IF (IMODE==1) CHAM=-CHAM
#ifdef testgamma_matrix
       IF (NK==1 .AND. ISP==1) THEN
          WRITE(*,'(8F14.7)') (REAL(CHAM(NB,NB),q),NB=1, W%WDES%NB_TOT)
       ENDIF
#endif
#ifdef  gammareal
       CALL DSYEV &
            ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),WDES%NB_TOT, &
            R,CWRK,LWORK*WDES%NB_TOT, IFAIL)
#else
       CALL ZHEEV &
            ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),WDES%NB_TOT, &
            R,CWRK,LWORK*WDES%NB_TOT, RWORK,  IFAIL)
#endif
#ifdef debug
       IF (IO%IU0>=0) CALL DUMP_HAM( "U",WDES, CHAM)
#endif
       ! change sign of eigenvalues
       IF (IMODE==1) R=-R

       IF (IFAIL/=0) THEN
          WRITE(0,*) 'ERROR in ADD_GAMMA_FROM_FILE: call to ZHEEV/ DSYEV failed! '// &
               &              'error code was ',IFAIL
          STOP
       ENDIF

       CALL SETWDES(WDES,WDES1,NK)

       WA=ELEMENTS(W, WDES1, ISP)

       !  distribution over plane wave coefficients
       IF (WDES%DO_REDIS) CALL REDISTRIBUTE_PROJ( ELEMENTS( W, WDES1, ISP))
       IF (WDES%DO_REDIS) CALL REDISTRIBUTE_PW( ELEMENTS( W, WDES1, ISP))

       CALL LINCOM('F',WA%CW_RED(:,:),WA%CPROJ_RED(:,:),CHAM(1,1), &
            WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
            WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,WDES%NB_TOT, &
            WA%CW_RED(:,:),WA%CPROJ_RED(:,:))

       !  back redistribution over bands
       IF (WDES%DO_REDIS) CALL REDISTRIBUTE_PROJ( ELEMENTS( W, WDES1, ISP))
       IF (WDES%DO_REDIS) CALL REDISTRIBUTE_PW( ELEMENTS( W, WDES1, ISP))

       ! updated one-electron occupancies
       IF (IMODE==1) THEN
          W%FERTOT(1:WDES%NB_TOTK(NK,ISP),NK,ISP)=R(1:WDES%NB_TOTK(NK,ISP))
       ELSE
          ! copy R to CELTOT
          W%CELTOT(1:WDES%NB_TOTK(NK,ISP),NK,ISP)=R(1:WDES%NB_TOTK(NK,ISP))
       ENDIF
    ENDDO
    ENDDO

    CALL CLOSEGAMMA

    ! just in case sync everything back to all nodes if KPAR is used
    CALL KPAR_SYNC_ALL(WDES,W)

    ! IMODE==2, update one-electron occupancies now
    IF (IMODE==2) THEN
       ! update fermi-weights
#ifdef  testgamma_matrix
       WRITE(*,'(8F14.7)') REAL(W%CELTOT(:,1,1),q)
#endif       
       CALL DENSTA_SIMPLE(W, KPOINTS, NELECT, NUP_DOWN )
       IF (LWRITE) THEN
          IF (IO%IU6>=0) THEN
             WRITE(IO%IU6,*) 'eigenvalues after inclusion of GAMMA file'
          ENDIF
          CALL WRITE_EIGENVAL( W%WDES, W, IO%IU6)
       ENDIF
       ! restore old one-electron eigenvalues
       W%CELTOT=CELTOT
    ENDIF

    DEALLOCATE(CHAM)


  END SUBROUTINE ADD_GAMMA_FROM_FILE


!************************ SUBROUTINE EDDIAG_EXACT **********************
!
! this subroutine performs a full diagonalization of the Hamiltonian
! right now it is stupidly implemented since
! the number of bands is increased for all k-points
! than EDDIAG is called
! and finally the number of bands is set back to the original
! value
! this requires a lot of storage but is still convenient for 
! GW and RPA calculations
!
!***********************************************************************

  SUBROUTINE EDDIAG_EXACT(HAMILTONIAN, &
       GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
       LMDIM,CDIJ,CQIJ,IFLAG,SV,T_INFO,P,IU0,IU6,EXHF,EXHF_ACFDT)
    USE prec
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE main_mpi
    USE pseudo
    USE poscar
    USE ini
    USE choleski
    USE fock
    USE scala
    IMPLICIT NONE
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (symmetry) ::   SYMM      
    INTEGER LMDIM
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    INTEGER            IFLAG            ! determines mode of diagonalisation
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    INTEGER IU0, IU6
    REAL(q) EXHF
    REAL(q) EXHF_ACFDT
  ! local
    INTEGER NB_TOT    ! maximum number of plane wave coefficients = number of bands
    TYPE (wavedes)     WDES_TMP
    TYPE (wavespin)    W_TMP
    INTEGER NK, DEGREES_OF_FREEDOM

  ! just make sure that data distribution is over bands
    CALL REDIS_PW_OVER_BANDS(WDES, W)
  ! are all bands calculated anyway 
    DEGREES_OF_FREEDOM=MAXVAL(WDES%NPLWKP_TOT)
    IF (WDES%LGAMMA) THEN
       DEGREES_OF_FREEDOM=DEGREES_OF_FREEDOM*2-1
    ENDIF

    IF (DEGREES_OF_FREEDOM<=WDES%NB_TOT) THEN
       IFLAG=3
       CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
            LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IU0,EXHF,EXHF_ACFDT=EXHF_ACFDT)
    ELSE

       NB_TOT=((DEGREES_OF_FREEDOM+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR
    
       WDES_TMP=WDES
       WDES_TMP%NB_TOT=NB_TOT
       WDES_TMP%NBANDS=NB_TOT/WDES%NB_PAR
       CALL INIT_SCALAAWARE( WDES_TMP%NB_TOT, WDES_TMP%NRPLWV, WDES_TMP%COMM_KIN )
       
       NULLIFY(WDES_TMP%NB_TOTK)
       ALLOCATE(WDES_TMP%NB_TOTK(WDES%NKDIM,2))
! set the maximum number of bands k-point dependent
       DO NK=1,WDES_TMP%NKPTS
          IF (WDES_TMP%LGAMMA) THEN
             WDES_TMP%NB_TOTK(NK,:)=MIN(WDES_TMP%NB_TOT,WDES_TMP%NPLWKP_TOT(NK)*2-1)
          ELSE
             WDES_TMP%NB_TOTK(NK,:)=MIN(WDES_TMP%NB_TOT,WDES_TMP%NPLWKP_TOT(NK))
          ENDIF
       ENDDO
       CALL RESETUP_FOCK_WDES(WDES_TMP, LATT_CUR, LATT_CUR, -1)

       CALL ALLOCW(WDES_TMP,W_TMP)

       CALL DUMP_ALLOCATE(IU6)

       W_TMP%FERTOT=0
       W_TMP%CELTOT=0
    ! copy data back to work array
       W_TMP%CW(:,1:WDES%NBANDS,:,:)    =W%CW(:,1:WDES%NBANDS,:,:)
       W_TMP%CPROJ(:,1:WDES%NBANDS,:,:) =W%CPROJ(:,1:WDES%NBANDS,:,:)
       W_TMP%CELTOT(1:WDES%NB_TOT,:,:)=W%CELTOT(1:WDES%NB_TOT,:,:)
       W_TMP%FERTOT(1:WDES%NB_TOT,:,:)=W%FERTOT(1:WDES%NB_TOT,:,:)

    ! random initialization beyond WDES%NBANDS
       CALL WFINIT(WDES_TMP, W_TMP, 1E10_q, WDES%NB_TOT+1) ! ENINI=1E10 not cutoff restriction
    ! get characters   
       CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W_TMP)
    ! orthogonalization
       CALL ORTHCH(WDES_TMP,W_TMP, WDES%LOVERL, LMDIM,CQIJ)
    ! and diagonalization
       IFLAG=3
       CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W_TMP,WDES_TMP,SYMM, &
            LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IU0,EXHF,EXHF_ACFDT=EXHF_ACFDT)
    ! copy data back to original array
       W%CW(:,1:WDES%NBANDS,:,:)    =W_TMP%CW(:,1:WDES%NBANDS,:,:)
       W%CPROJ(:,1:WDES%NBANDS,:,:) =W_TMP%CPROJ(:,1:WDES%NBANDS,:,:)
       W%CELTOT(1:WDES%NB_TOT,:,:)=W_TMP%CELTOT(1:WDES%NB_TOT,:,:)
       W%FERTOT(1:WDES%NB_TOT,:,:)=W_TMP%FERTOT(1:WDES%NB_TOT,:,:)
!       
       CALL DEALLOCW(W_TMP)
       DEALLOCATE(WDES_TMP%NB_TOTK)

       CALL RESETUP_FOCK_WDES(WDES, LATT_CUR, LATT_CUR, -1)

    ENDIF

  END SUBROUTINE EDDIAG_EXACT


!*************** SUBROUTINE EDDIAG_EXACT_UPDATE_NBANDS *****************
!
! this subroutine performs a full diagonalization of the Hamiltonian
! and keeps the band number to the large number so determined
!
!***********************************************************************

  SUBROUTINE EDDIAG_EXACT_UPDATE_NBANDS(HAMILTONIAN,KINEDEN, &
       GRID,GRID_SOFT,GRIDC,GRIDB,GRIDUS,C_TO_US,SOFT_TO_C,B_TO_C,E, &
       CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
       IRDMAX,CRHODE,MIX,N_MIX_PAW,RHOLM,RHOLM_LAST,CHDEN, &
       LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
       LMDIM,CDIJ,CQIJ,SV,T_INFO, DYN, P, IO, INFO, &
       XCSIF, EWSIF, TSIF, EWIFOR, TIFOR, PRESS, TOTEN, KPOINTS, NBANDSEXACT )

    USE prec
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE main_mpi
    USE pseudo
    USE poscar
    USE ini
    USE choleski
    USE fock
    USE scala
    USE setexm
    USE meta
    USE us
    USE pawm
    IMPLICIT NONE
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (tau_handle)  KINEDEN
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (symmetry)    SYMM
    INTEGER LMDIM
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    TYPE (type_info)   T_INFO
    TYPE (kpoints_struct) :: KPOINTS
    TYPE (dynamics)    DYN
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDB      ! grid for Broyden mixer
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q)  CHTOTL(GRIDC%MPLWV,WDES%NCDIJ)! old charge-density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    INTEGER            IRDMAX, N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)
    REAL(q)  RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (energy)      E
    TYPE (in_struct)   IO
    TYPE (mixing)      MIX
    TYPE (info_struct) INFO
    REAL(q)   XCSIF(3,3)                      ! stress stemming from XC
    REAL(q)   EWSIF(3,3)                      ! stress from Ewald contribution
    REAL(q)   TSIF(3,3)                       ! total stress (set by routine)
    REAL(q)   EWIFOR(3,T_INFO%NIOND)          ! ewald force
    REAL(q)   TIFOR(3,T_INFO%NIOND)           ! total force (set by routine)
    REAL(q)   PRESS                           ! external pressure
    REAL(q)   TOTEN
    INTEGER   NBANDSEXACT                     ! bands to be used in exact diagonalization
  ! local
    INTEGER NB_TOT            ! maximum number of plane wave coefficients = number of bands
    TYPE (wavespin)    W_TMP
    INTEGER ISP, NK, NB, DEGREES_OF_FREEDOM
    INTEGER            IFLAG  ! determines mode of diagonalisation
    INTEGER NB_TOT_OLD, NBANDS_OLD
    LOGICAL :: LCORR_TMP
    INTEGER,ALLOCATABLE :: NB_TOTK(:,:)

  ! just make sure that data distribution is over bands
    CALL REDIS_PW_OVER_BANDS(WDES, W)

    ! recalculate charge density and  kinetic energy density
    CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
         GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
         LATT_CUR, P, SYMM, T_INFO, &
         CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

    CALL SET_KINEDEN(GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM, &
         T_INFO%NIONS,W,WDES,KINEDEN)      

  ! calculate local potential
    CALL UPDATE_POT

    LCORR_TMP=INFO%LCORR
    INFO%LCORR=.FALSE.

  ! we include the contributions \sum_ij H^HF_ij <phi_i| dp/dR Q |phi_j> + c.c,
  ! where i and j are both occupied,
  ! in the routine FORNL2 and FORNLR2 in rpa_force
  ! by setting CELTOT to 0, these contributions are bypassed here
    W%CELTOT=0
  ! to get the correct forces, the orbitals need to diagonalized the HF 
  ! Hamiltonian (see above)
  ! note that LREMOVE_DRIFT is .false. i.e. no poking with forces
    IF (IO%IU6>=0) WRITE(IO%IU6,*) "HF-forces evaluated using KS orbitals"
    CALL FORCE_AND_STRESS( &
         KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
         T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
         GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
         CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
         CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
         CHDEN,SV, &
         LMDIM, IRDMAX, .TRUE., &
         DYN%ISIF/=0, DYN%ISIF/=0,.FALSE.,  &
         XCSIF, EWSIF, TSIF, EWIFOR, TIFOR, PRESS, TOTEN, KPOINTS )

    ! force routine uses NONLR_S, so reset it
    IF (INFO%LREAL) THEN
       CALL RSPHER(GRID,NONLR_S,LATT_CUR)
    ENDIF

    INFO%LCORR=LCORR_TMP

    PROFILING_START('eddiag_exact_update_nbands')

  ! the first step is to restore the original XC-functional (as read from POTCAR/INCAR)
  ! in order to use the appropriate Hamiltonian
    CALL POP_XC_TYPE
    IF (WDES%LNONCOLLINEAR .OR. INFO%ISPIN == 2) THEN
       CALL SETUP_LDA_XC(2,-1,-1,IO%IDIOT)
    ELSE
       CALL SETUP_LDA_XC(1,-1,-1,IO%IDIOT)
    ENDIF
  ! set exact exchange according to LDAX
    AEXX=1.0-LDAX
  ! now update the PAW one-center terms to current functional
    CALL SET_PAW_ATOM_POT( P , T_INFO, WDES%LOVERL, LMDIM, INFO%EALLAT, INFO%LMETAGGA, IO%IU6 )
    ! and restore the convergence corrections
    DO NK=1,WDES%NKPTS
       FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
    ENDDO

    CALL UPDATE_POT

    DEGREES_OF_FREEDOM=MAXVAL(WDES%NPLWKP_TOT)
    IF (WDES%LGAMMA) THEN
       DEGREES_OF_FREEDOM=DEGREES_OF_FREEDOM*2-1
    ENDIF

    IF (DEGREES_OF_FREEDOM<=WDES%NB_TOT) THEN
  ! smaller than already included bands
  ! restore DFT eigenvalue
       IFLAG=3
       CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
            LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,E%EXHF,EXHF_ACFDT=E%EXHF_ACFDT)
    ELSE
       NB_TOT_OLD=WDES%NB_TOT
       NBANDS_OLD=WDES%NBANDS
       NB_TOT=((DEGREES_OF_FREEDOM+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR
    
       WDES%NB_TOT=NB_TOT
       WDES%NBANDS=NB_TOT/WDES%NB_PAR
       CALL INIT_SCALAAWARE( WDES%NB_TOT, WDES%NRPLWV, WDES%COMM_KIN )
       
! set the maximum number of bands k-point dependent
       DO NK=1,WDES%NKPTS
          IF (WDES%LGAMMA) THEN
             WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK)*2-1)
          ELSE
             WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK))
          ENDIF
       ENDDO
       CALL RESETUP_FOCK_WDES(WDES, LATT_CUR, LATT_CUR, -1)

    ! copy W to new W_TMP    
       CALL ALLOCW(WDES, W_TMP)
       W_TMP%FERTOT(:,:,:)=0
       W_TMP%CELTOT(:,:,:)=0
    ! copy data back to work array
       W_TMP%CW(:,1:NBANDS_OLD,:,:)    =W%CW(:,1:NBANDS_OLD,:,:)
       W_TMP%CPROJ(:,1:NBANDS_OLD,:,:) =W%CPROJ(:,1:NBANDS_OLD,:,:)
       W_TMP%CELTOT(1:NB_TOT_OLD,:,:)=W%CELTOT(1:NB_TOT_OLD,:,:)
       W_TMP%FERTOT(1:NB_TOT_OLD,:,:)=W%FERTOT(1:NB_TOT_OLD,:,:)
       CALL DEALLOCW(W)
    ! random initialization beyond WDES%NBANDS
       CALL WFINIT(WDES, W_TMP, 1E10_q, NB_TOT_OLD+1) ! ENINI=1E10 not cutoff restriction

    ! get characters
       CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W_TMP)
    ! orthogonalization
       CALL ORTHCH(WDES,W_TMP, WDES%LOVERL, LMDIM,CQIJ)
    ! and diagonalization
       IFLAG=3
       CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W_TMP,WDES,SYMM, &
            LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,E%EXHF,EXHF_ACFDT=E%EXHF_ACFDT)
    ! set W=W_TMP
       W=W_TMP
    ! now if NBANDSEXACT is set, decrease the number of bands to NBANDSEXACT
       IF (NBANDSEXACT>=0 .AND. NBANDSEXACT< NB_TOT) THEN
          NBANDSEXACT=MAX(NB_TOT_OLD, NBANDSEXACT)  ! make sure user did not use stupid values
          NB_TOT=((NBANDSEXACT+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR
    
          WDES%NB_TOT=NB_TOT
          WDES%NBANDS=NB_TOT/WDES%NB_PAR
          CALL INIT_SCALAAWARE( WDES%NB_TOT, WDES%NRPLWV, WDES%COMM_KIN )

          ! set the maximum number of bands k-point dependent
          DO NK=1,WDES%NKPTS
             IF (WDES%LGAMMA) THEN
                WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK)*2-1)
             ELSE
                WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK))
             ENDIF
          ENDDO
          CALL RESETUP_FOCK_WDES(WDES, LATT_CUR, LATT_CUR, -1)
          
          CALL ALLOCW(WDES, W_TMP)
          ! copy data to W_TMP
          W_TMP%CW(:,1:WDES%NBANDS,:,:)    =W%CW(:,1:WDES%NBANDS,:,:)
          W_TMP%CPROJ(:,1:WDES%NBANDS,:,:) =W%CPROJ(:,1:WDES%NBANDS,:,:)
          W_TMP%CELTOT(1:WDES%NB_TOT,:,:)=W%CELTOT(1:WDES%NB_TOT,:,:)
          W_TMP%FERTOT(1:WDES%NB_TOT,:,:)=W%FERTOT(1:WDES%NB_TOT,:,:)
          CALL DEALLOCW(W)
          W=W_TMP
       ENDIF
       CALL DUMP_ALLOCATE(IO%IU6)
    ENDIF

    ! now we restore the HF xc-correlation functional
    CALL PUSH_XC_TYPE_FOR_GW
    IF (WDES%LNONCOLLINEAR .OR. INFO%ISPIN == 2) THEN
       CALL SETUP_LDA_XC(2,-1,-1,IO%IDIOT)
    ELSE
       CALL SETUP_LDA_XC(1,-1,-1,IO%IDIOT)
    ENDIF
    ! now update the PAW one-center terms to current functional
    CALL SET_PAW_ATOM_POT( P , T_INFO, WDES%LOVERL, LMDIM, INFO%EALLAT, INFO%LMETAGGA, IO%IU6 )

    ! and restore the convergence corrections
    DO NK=1,WDES%NKPTS
       FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
    ENDDO

    E%EBANDSTR=BANDSTRUCTURE_ENERGY(WDES, W)

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,7240) " Exact diagonalization of KS Hamiltonian yields:", & 
            E%PSCENC,E%TEWEN,E%DENC,E%EXHF,E%XCENC,E%PAWPS,E%PAWAE, &
            E%EENTROPY,E%EBANDSTR,INFO%EALLAT,'  free energy    TOTEN  = ', &
            E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF
    ENDIF

7240 FORMAT(/ &
              A,/ &
              ' Free energy of the ion-electron system (eV)', / &
     &        '  ---------------------------------------------------'/ &
     &        '  alpha Z        PSCENC = ',F18.8/ &
     &        '  Ewald energy   TEWEN  = ',F18.8/ &
     &        '  -Hartree energ DENC   = ',F18.8/ &
     &        '  -exchange      EXHF   = ',F18.8/ &
     &        '  -V(xc)+E(xc)   XCENC  = ',F18.8/ &
     &        '  PAW double counting   = ',2F18.8/ &
     &        '  entropy T*S    EENTRO = ',F18.8/ &
     &        '  eigenvalues    EBANDS = ',F18.8/ &
     &        '  atomic energy  EATOM  = ',F18.8/ &
     &        '  ---------------------------------------------------'/ &
     &        A,F18.8,' eV' )

    CALL WRITE_EIGENVAL_NBANDS( WDES, W, IO%IU6, WDES%NB_TOT)

    PROFILING_STOP('eddiag_exact_update_nbands')

  CONTAINS 

    SUBROUTINE UPDATE_POT
      USE pot
      USE pawm
      USE morbitalmag
      USE us
      REAL(q) :: XCSIF(3,3)
      INTEGER :: IRDMAA

      CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
           INFO,P,T_INFO,E,LATT_CUR, &
           CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
      
      CALL POTLOK_METAGGA(KINEDEN, &
           GRID,GRIDC,GRID_SOFT,WDES%COMM_INTER,WDES,INFO,P,T_INFO,E,LATT_CUR, &
           CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT_TO_C,XCSIF)
      
      CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C,  WDES%COMM_INTER, & 
           LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)
      
      
      CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)
      
      CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)
      
      CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)
      
      CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
           WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
           E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
      
      CALL UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IO%IU6)
      
    END SUBROUTINE UPDATE_POT

  END SUBROUTINE EDDIAG_EXACT_UPDATE_NBANDS

!*************** SUBROUTINE UPDATE_CDIJ *****************
!
!  Update CDIJ to the current functional that is applied
!
!***********************************************************************

SUBROUTINE UPDATE_CDIJ(HAMILTONIAN,KINEDEN,GRID,GRID_SOFT, &
           GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,E,CHTOT,DENCOR, &
           CVTOT,CSTRF,IRDMAX,CRHODE,N_MIX_PAW,RHOLM,CHDEN, &
           LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM,LMDIM,CDIJ, &
           CQIJ,SV,T_INFO,P,IO,INFO,XCSIF)

    USE prec
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE main_mpi
    USE pseudo
    USE poscar
    USE ini
    USE choleski
    USE fock
    USE scala
    USE setexm
    USE meta
    USE us
    USE pawm
    IMPLICIT NONE
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (tau_handle)  KINEDEN
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (symmetry)    SYMM
    INTEGER LMDIM
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    TYPE (type_info)   T_INFO
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    INTEGER            IRDMAX, N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (energy)      E
    TYPE (in_struct)   IO
    TYPE (info_struct) INFO
    REAL(q)   XCSIF(3,3)                      ! stress stemming from XC
    INTEGER NK

  ! the first step is to restore the original XC-functional (as read from POTCAR/INCAR)
  ! in order to use the appropriate Hamiltonian
    CALL POP_XC_TYPE
    IF (WDES%LNONCOLLINEAR .OR. INFO%ISPIN == 2) THEN
       CALL SETUP_LDA_XC(2,-1,-1,IO%IDIOT)
    ELSE
       CALL SETUP_LDA_XC(1,-1,-1,IO%IDIOT)
    ENDIF
  ! set exact exchange according to LDAX
    AEXX=1.0-LDAX
  ! now update the PAW one-center terms to current functional
    CALL SET_PAW_ATOM_POT( P , T_INFO, WDES%LOVERL, LMDIM, INFO%EALLAT, INFO%LMETAGGA, IO%IU6 )
    ! and restore the convergence corrections
    DO NK=1,WDES%NKPTS
       FSG_STORE(NK)=SET_FSG(GRIDHF, LATT_CUR, NK)
    ENDDO

  ! just make sure that data distribution is over bands
    CALL REDIS_PW_OVER_BANDS(WDES, W)

    ! recalculate charge density and  kinetic energy density
    CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
         GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
         LATT_CUR, P, SYMM, T_INFO, &
         CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

    CALL SET_KINEDEN(GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM, &
         T_INFO%NIONS,W,WDES,KINEDEN)      
    CALL UPDATE_POT

  CONTAINS 

    SUBROUTINE UPDATE_POT
      USE pot
      USE pawm
      USE morbitalmag
      USE us
      REAL(q) :: XCSIF(3,3)
      INTEGER :: IRDMAA

      CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
           INFO,P,T_INFO,E,LATT_CUR, &
           CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
      
      CALL POTLOK_METAGGA(KINEDEN, &
           GRID,GRIDC,GRID_SOFT,WDES%COMM_INTER,WDES,INFO,P,T_INFO,E,LATT_CUR, &
           CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT_TO_C,XCSIF)
      
      CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C,  WDES%COMM_INTER, & 
           LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)
      
      
      CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)
      
      CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)
      
      CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)
      
      CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
           WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
           E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
      
      CALL UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IO%IU6)
      
    END SUBROUTINE UPDATE_POT

  END SUBROUTINE UPDATE_CDIJ

END MODULE subrot


!***********************************************************************
!
! dump a "Hamilton matrix" between the calculated states
!
!***********************************************************************

  
  SUBROUTINE DUMP_HAM( STRING, WDES, CHAM )
    USE wave
    CHARACTER (LEN=*) :: STRING
    TYPE (wavedes)     WDES
    GDEF ::  CHAM(WDES%NB_TOT,WDES%NB_TOT)
    INTEGER N1, N2, NPL2
    INTEGER NB_TOT

    NB_TOT=WDES%NB_TOT

    WRITE(*,*) STRING

    NPL2=MIN(12,NB_TOT)
    DO N1=1,NPL2
       WRITE(*,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#ifndef gammareal
    DO N1=1,NPL2
       WRITE(6,2)N1,(AIMAG( CHAM(N1,N2)),N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#endif

1   FORMAT(1I2,3X,40F9.5)
!1   FORMAT(1I2,3X,40F14.9)
2   FORMAT(1I2,3X,40F9.5)

  END SUBROUTINE DUMP_HAM


!***********************************************************************
!
! dump a "Hamilton matrix" between the calculated states
! single precision version
!
!***********************************************************************

  
  SUBROUTINE DUMP_HAM_SINGLE( STRING, WDES, CHAM)
    USE wave
    CHARACTER (LEN=*) :: STRING
    TYPE (wavedes)     WDES
    GDEFS ::  CHAM(WDES%NB_TOT,WDES%NB_TOT)
    INTEGER N1, N2, NPL2
    INTEGER NB_TOT

    NB_TOT=WDES%NB_TOT

    WRITE(*,*) STRING
    NPL2=MIN(10,NB_TOT)
    DO N1=1,NPL2
       WRITE(*,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#ifndef gammareal
    DO N1=1,NPL2
       WRITE(6,2)N1,(AIMAG( CHAM(N1,N2)),N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#endif
1   FORMAT(1I2,3X,40F9.5)
2   FORMAT(1I2,3X,40F9.5)
!2   FORMAT(1I2,3X,40E9.1)

  END SUBROUTINE DUMP_HAM_SINGLE


!***********************************************************************
!
! dump a "Hamilton matrix" between the calculated states
!
!***********************************************************************

  
  SUBROUTINE DUMP_HAM_SELECTED( STRING, WDES, CHAM, NDIM, NBANDS)
    USE wave
    CHARACTER (LEN=*) :: STRING
    TYPE (wavedes)     WDES
    INTEGER NDIM
    INTEGER NBANDS
    GDEF ::  CHAM(NDIM,NDIM)
  ! local
    INTEGER N1, N2, NPL2
    INTEGER NB_TOT

    NB_TOT=WDES%NB_TOT

    WRITE(*,*) STRING
    NPL2=MIN(12,NBANDS)
    DO N1=1,NPL2
       WRITE(*,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#ifndef gammareal
    DO N1=1,NPL2
       WRITE(6,2)N1,(AIMAG( CHAM(N1,N2)),N2=1,NPL2)
    ENDDO
    WRITE(*,*)
#endif
1   FORMAT(1I2,3X,24F9.5)
2   FORMAT(1I2,3X,24F9.5)

  END SUBROUTINE DUMP_HAM_SELECTED


!=======================================================================
!
! small routine to dump a distributed matrix
! the descriptor in DESCA must properly describe the matrix
! since RECON_SLICE calls check 
!
!=======================================================================

#if defined(MPI) && defined(scaLAPACK)
  SUBROUTINE DUMP_HAM_DISTRI( STRING, WDES, CHAM_DISTRI, NB_TOT, DESCA, IU)
    USE wave
    USE scala
    IMPLICIT NONE
    CHARACTER (LEN=*) :: STRING            ! string to dump
    TYPE (wavedes)    :: WDES              ! wave function descriptor
    GDEF              :: CHAM_DISTRI(*)    ! distributed matrix
    INTEGER           :: NB_TOT
    INTEGER           :: DESCA(*)          ! distributed matrix descriptor array
    INTEGER           :: IU                ! unit to write to (not dump for IU<0)
  ! local
    INTEGER, PARAMETER :: NDUMP=16
    GDEF,ALLOCATABLE  :: CHAM(:,:)
    INTEGER N1, N2
    INTEGER COLUMN_HIGH, COLUMN_LOW

    COLUMN_LOW=1

    COLUMN_HIGH=MIN(COLUMN_LOW+NDUMP-1,NB_TOT)

    ALLOCATE(CHAM(NB_TOT, COLUMN_HIGH-COLUMN_LOW+1))

    CALL RECON_SLICE(CHAM, NB_TOT, NB_TOT, CHAM_DISTRI,  DESCA, COLUMN_LOW, COLUMN_HIGH)

    CALLMPI( M_sum_g(WDES%COMM_KIN, CHAM(1,1), SIZE(CHAM)))

    IF (IU>=0) THEN
    WRITE(IU,*) STRING
    DO N1=1,NDUMP
       WRITE(IU,1)N1+COLUMN_LOW-1,(REAL( CHAM(N1,N2+COLUMN_LOW-1) ,KIND=q) ,N2=1,NDUMP)
    ENDDO
    WRITE(IU,*)
#ifndef gammareal
    DO N1=1,NDUMP
       WRITE(IU,2)N1+COLUMN_LOW-1,(AIMAG( CHAM(N1,N2+COLUMN_LOW-1)),N2=1,NDUMP)
    ENDDO
    WRITE(IU,*)
#endif
    ENDIF

    DEALLOCATE(CHAM)
!1   FORMAT(1I2,3X,40F9.5)
!2   FORMAT(1I2,3X,40F9.5)
1   FORMAT(1I2,3X,40F7.4)
2   FORMAT(1I2,3X,40F7.4)
!1   FORMAT(1I2,3X,40F14.9)

  END SUBROUTINE DUMP_HAM_DISTRI
#endif

!************************ SUBROUTINE ORSP   ****************************
!
! this subroutine perfomes a gram-schmidt orthogonalistion of a set
! of vectors (all elements on local node)
! the subroutine uses BLAS 3 calls
!
!***********************************************************************

  SUBROUTINE ORSP(NBANDS, NPL, NRPLWV, CPTWFP)
    USE prec
    IMPLICIT NONE

    INTEGER NBANDS
    INTEGER NPL
    INTEGER NRPLWV
    GDEF CPTWFP(NRPLWV,NBANDS)
  ! local
    GDEF CPRO(NBANDS)
    COMPLEX(q), EXTERNAL :: ZDOTC
    REAL(q), EXTERNAL ::  DDOT
    INTEGER I, N
    REAL(q) WFMAG

    IF (NBANDS> NRPLWV) THEN
       WRITE(*,*) 'internal error in ORSP: leading dimension of matrix too small'
       STOP
    ENDIF

    CPRO=0
    DO N=1,NBANDS

       ! normalise the vector

       WFMAG=GDOTC(NPL,CPTWFP(1,N),1,CPTWFP(1,N),1)
       CALL GDSCAL(NPL,1/SQRT(WFMAG),CPTWFP(1,N),1)

       ! now orthogonalise all higher vectors to the
       ! present vector

       IF (NBANDS/=N ) THEN
          CALL GGEMV( trans, NPL , NBANDS-N ,one , CPTWFP(1,N+1), &
               &             NRPLWV, CPTWFP(1,N), 1 , zero ,  CPRO, 1)

          DO I=1,NBANDS
             CPRO(I)=GCONJG(CPRO(I))
          ENDDO

          CALL GGEMM( 'N', 'T' , NPL , NBANDS-N , 1 , -one , &
               &             CPTWFP(1,N), NRPLWV , CPRO , NBANDS , &
               &             one , CPTWFP(1,N+1) , NRPLWV )
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ORSP


!***********************************************************************
!
! use Loewdin perturbation to determine a rotation matrix
! this preserves the ordering of the eigenvalues
! MIND: does not work for real matrices
!
!***********************************************************************

  SUBROUTINE LOEWDIN_DIAG(NB_TOT, NBDIM, CHAM)
    USE prec
    IMPLICIT NONE
    INTEGER NB_TOT, NBDIM
    GDEF :: CHAM(NBDIM, NB_TOT)
  ! local
    REAL(q), PARAMETER  :: DIFMAX=0.001_q
    REAL(q) DIFCEL
    INTEGER N1, N2
    COMPLEX(q) :: CROT
    REAL(q) :: FAKT

    IF (NB_TOT>NBDIM) THEN
       WRITE(*,*) 'internal error in LOEWDIN_DIAG: leading dimension of matrix too small'
       STOP
    ENDIF


    DO N2=1,NB_TOT
       DO N1=1,N2-1
          DIFCEL= REAL( CHAM(N2,N2)-CHAM(N1,N1) ,KIND=q)
          IF (ABS(DIFCEL)<DIFMAX) THEN
             CROT  =0
          ELSE
             CROT  =GCONJG(CHAM(N1,N2))/DIFCEL
             IF (ABS(CROT)>0.1_q) THEN
                FAKT= 0.1_q/ABS(CROT)
                CROT  = CROT*FAKT
             ENDIF
          ENDIF
          CHAM(N2,N1) =-CROT
          CHAM(N1,N2) =-GCONJG(CROT)
       ENDDO
    ENDDO
    DO N1=1,NB_TOT
       CHAM(N1,N1)=1
    ENDDO
  END SUBROUTINE LOEWDIN_DIAG


!*******************************************************************
!  calculate the matrix elements of a local potential
!           
!  CHAM(i,j)= <psi_i,k| V |psi_j,k> = int psi_i,k*(r) V(r) psi_j,k(r)
!
! between states
! the argument CVTOT must be in real space
! the result is retured in CHAM
!
!*******************************************************************

  SUBROUTINE LOCAL_BETWEEN_STATES( HAMILTONIAN, W, LATT_CUR, P, T_INFO, IRDMAX, LMDIM, &
       GRID_SOFT, GRIDC, GRIDUS, SOFT_TO_C, C_TO_US, CVTOT, CHAM)
    USE prec
    USE wave_high
    USE lattice
    USE poscar
    USE pseudo
    USE pot
    USE pawm
    USE subrot
    USE hamil_high
    USE us
    
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (wavespin)    W
    
    INTEGER  IRDMAX      ! allocation required for augmentation
    TYPE (latt)        LATT_CUR
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (grid_3d)     GRIDC                  ! grid for potentials/charge
    TYPE (grid_3d)     GRID_SOFT              ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDUS                 ! grid for augmentation
    TYPE (transit)     SOFT_TO_C              ! index table between GRID_SOFT and GRIDC
    TYPE (transit)     C_TO_US                ! index table between GRID_SOFT and GRIDC
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,W%WDES%NCDIJ) ! local potential
    INTEGER LMDIM
    GDEF       CHAM(W%WDES%NB_TOT,W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN)
  ! local
    INTEGER ISP, NK
    RGRID ::   SV(DIMREAL(W%WDES%GRID%MPLWV),W%WDES%NCDIJ)   ! local potential
    OVERLAP :: CDIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    OVERLAP :: CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    INTEGER IRDMAA
    REAL(q)  DISPL(3,T_INFO%NIONS)

#ifdef MPI
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('LOCAL_BETWEEN_STATES: KPAR>1 not tested (but seems ok), sorry.') )
!PK Trivial but callers must be adapted
       STOP
    END IF
#endif

    DISPL=0

    ! get  the non local strenght parameters
    CALL SETDIJ_(W%WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, W%WDES%LOVERL, &
         LMDIM, CDIJ, CQIJ, CVTOT, .FALSE., IRDMAA, IRDMAX, DISPL)

    ! transform CVTOT to reciprocal space (required by SET_SV)
    DO ISP=1,W%WDES%NCDIJ
       CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
    ENDDO

    ! now set SV from CVTOT
    CALL SET_SV( W%WDES%GRID, GRIDC, GRID_SOFT, W%WDES%COMM_INTER, SOFT_TO_C, W%WDES%NCDIJ, SV, CVTOT)
    
    CHAM=0

    DO ISP=1,W%WDES%NCDIJ
       DO NK=1,W%WDES%NKPTS
          CALL ONE_CENTER_BETWEEN_STATES( HAMILTONIAN, LATT_CUR, W%WDES%LOVERL, W%WDES, W, NK, ISP, LMDIM, &
               CDIJ, CHAM(1,1,NK,ISP), SV)
!         CALL DUMP_HAM( "Hamiltonian", W%WDES, CHAM(1,1,NK,ISP))

       ENDDO
    ENDDO
       

  END SUBROUTINE LOCAL_BETWEEN_STATES
