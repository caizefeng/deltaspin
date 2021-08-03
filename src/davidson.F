#include "symbol.inc"
#define USE_ZHEEVX

MODULE david
CONTAINS
!************************ SUBROUTINE EDDDAV *****************************
! RCS:  $Id: davidson.F,v 1.5 2003/06/27 13:22:15 kresse Exp kresse $
!
! this subroutine performes a Davidson like optimsation of the
! wavefunctions i.e. it the expectation value
!     < phi | H |  phi >
! for NSIM bands in parallel
! 
! different preconditioners can be chosen using INFO%IALGO
!  INFO%IALGO   determine type of preconditioning and the algorithm
!    6    rms-minimization          +  TAP preconditioning
!    7    rms-minimization          +  no preconditioning
!    8    precond rms-minimization  +  TAP preconditioning
!    9    precond rms-minimization  +  Jacobi like preconditioning
!    (TAP Teter Alan Payne)
!  WEIMIN  treshhold for total energy minimisation
!    is the fermiweight of a band < WEIMIN,
!    minimisation will break after a maximum of two iterations
!  EBREAK  absolut break condition
!    intra-band minimisation is stopped if DE is < EBREAK
!
!***********************************************************************

  SUBROUTINE EDDAV(HAMILTONIAN, P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
       LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV, EXHF, IU6, IU0, LDELAY, LSUBROTI, LEMPTY, LHF, NKSTART, EXHF_ACFDT)
    USE prec
    USE wave_high
    USE base
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE constant
    USE scala
    USE fock
    USE pseudo
    USE dfast
    USE pead
    IMPLICIT NONE
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (potcar)      P(:)
    TYPE (grid_3d)     GRID          ! descriptor for FFT grids
    TYPE (info_struct) INFO          ! INFO structure of VASP
    TYPE (latt)        LATT_CUR      !  
    TYPE (nonlr_struct) NONLR_S      ! descriptor for non local part of PP (real space)
    TYPE (nonl_struct) NONL_S        ! descriptor for non local part of PP (reciprocal space)
    TYPE (wavespin)    W             ! array for wavefunction
    TYPE (wavedes)     WDES          ! descriptor for wavefunction
    INTEGER NSIM                     ! simultaneously optimised bands
    INTEGER LMDIM                    ! dimension of arrays CQIJ and CDIJ
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    REAL (q)           RMS           ! on return: norm of residual vector summed over all bands
    REAL (q)           DESUM         ! on return: change of eigenvalues
    INTEGER            ICOUEV        ! number of intraband eigen value minimisations
    RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    INTEGER            IU6           ! stdout
    INTEGER            IU0           ! sterr
    LOGICAL LDELAY                   ! delay phase (not used) 
    LOGICAL LSUBROTI                 ! perform subspace rotation
    LOGICAL LEMPTY                   ! optimize emtpy bands only
    LOGICAL, OPTIONAL :: LHF         ! calculate HF contributions
    INTEGER, OPTIONAL :: NKSTART     ! start k-point
    REAL(q),OPTIONAL :: EXHF_ACFDT
 ! local work arrays
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    TYPE (wavefuna)    WA            ! wavefunction array, points to W section for one k and spin
    TYPE (wavefuna), TARGET :: WOPT  ! subset of wavefunctions currently optimized
    TYPE (wavefuna), TARGET :: WHAM  ! action of H on these wavefunctions
    TYPE (wavefuna), TARGET :: WS    ! stores CQIJ * wavefunctions
    TYPE (wavefuna), TARGET :: WHF   ! HF part of Hamiltonian
    TYPE (wavespin)    W_ORIG        ! stores the original not redistributed wavefunctions
                                     ! required for HF
    GDEF, ALLOCATABLE :: CORTHO(:,:) ! stores <psi_i| S |psi_k>
    TYPE (wavefun1), TARGET :: W1(NSIM) ! wavefunction currently added to subspace

    REAL(q),ALLOCATABLE:: PRECON(:,:)! preconditioning matrix for each
    GDEF,ALLOCATABLE:: CHAM(:,:),COVL(:,:),CEIG(:,:),COVL_(:,:)
    REAL(q) :: EVALUE_INI(NSIM)      ! eigenvalue of that band at the beginning
    REAL(q) :: EVALUE_GLBL(NSIM)     ! same as eigenvalue but global
    REAL(q) :: EVALUE_INI_GLBL(NSIM) ! same as eigenvalue but global
    REAL(q) :: DEIT                  ! relative break criterion for that band
    INTEGER :: IT(NSIM)              ! current iteration for this band
    REAL(q) :: TRIAL(NSIM)           ! trial step for each band
    LOGICAL :: LSTOP                 ! optimisation finished
    LOGICAL :: LSUBROT               ! usually LSUBROTI
    LOGICAL :: DO_REDIS              ! redistribution of wavefunctions required
    ! nbands times nbands hamilton matrix and overlap matrix
    GDEF,ALLOCATABLE,TARGET::  CHAM_ALL(:,:),COVL_ALL(:,:)

    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER  :: LWORK=32
    GDEF,ALLOCATABLE    :: CWRK(:)
    REAL(q),ALLOCATABLE ::  R(:),RWORK(:)
    INTEGER :: NB_MAX, MY_NKSTART
    LOGICAL :: LHFCALC_DAV
#ifndef USE_ZHEEVX
    INTEGER,PARAMETER :: IRWORK=3
#else
    INTEGER,PARAMETER :: IRWORK=7
    INTEGER,ALLOCATABLE :: IWORK(:), MINFO(:)
#endif
    REAL (q)   :: ABSTOL=1E-10_q, VL, VU, RCOND
    INTEGER    :: IL, IU, NB_CALC
    ! more local variables
    INTEGER :: NODE_ME, NODE_MEI, IONODE, NCPU, NSIM_LOCAL, NSIM_, NSIM_LOCAL_, &
         NSUBD, NITER, I, NBANDS, NB_TOT, ISP, NK, NB_START, &
         NB_DONE, NP, N, ITER, NPP, M, MM,  IDUMP, &
         ISPINOR, N1, N2, NPOS_RED, IFAIL, II, NITER_NOW, IPREC
    REAL(q) :: SLOCAL, DE_ATT, EKIN, FAKT, X, X2, FNORM, FPRE_, DECEL, DEMAX, WSCAL
    COMPLEX(q) :: CPT, CDCHF
    REAL(q) :: EXHF

    PROFILING_START('eddav')

!=======================================================================
! initialise the required variables for MPI
!=======================================================================
#ifdef MPI
    NODE_ME=WDES%COMM_KIN%NODE_ME
    IONODE =WDES%COMM_KIN%IONODE
    NODE_MEI=WDES%COMM_INTER%NODE_ME ! number of groups (each group holds one band)
    NCPU   =WDES%COMM_INTER%NCPU     ! number of groups (each group holds one band)
#else
    NODE_ME=1
    NODE_MEI=1
    IONODE =1
    NCPU=1
#endif
!=======================================================================
! number of bands treated simultaneously this must be a multiple of NCPU
!=======================================================================
    NSIM_LOCAL=NSIM/NCPU  ! number of bands optimised on this node
    IF (NSIM_LOCAL*NCPU /= NSIM) THEN
       WRITE(*,*) 'internal ERROR in EDDAV NSIM is not correct',NSIM
       STOP
    ENDIF

    LSUBROT=LSUBROTI
    IF (NSIM>=WDES%NB_TOT) THEN
       LSUBROT=.FALSE.
    ENDIF

    NITER =MAX(INFO%NDAV+1,2) ! maximum number of iterations

    ! at least one optimisation step
    NSUBD =NITER*NSIM         ! maximum size of the subspace
    NB_MAX=NSUBD

    IF (LSUBROT) NB_MAX=MAX(NB_MAX,WDES%NB_TOT)

    CALL SETWDES(WDES,WDES1,0)

    CALL NEWWAVA(WOPT, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA(WHAM, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA_PROJ(WS, WDES1, NSIM_LOCAL)

    ALLOCATE(PRECON(WDES%NRPLWV,NSIM_LOCAL), &
         &        CHAM(NSUBD,NSUBD),COVL(NSUBD,NSUBD),CEIG(NSUBD,NSUBD),COVL_(NSUBD,NSUBD), &
         &        CORTHO(WDES%NB_TOT,NSIM),R(NB_MAX),RWORK(NB_MAX*IRWORK),CWRK(LWORK*WDES%NB_TOT))

    LHFCALC_DAV=.FALSE.
    IF (PRESENT(LHF)) THEN
       IF (LHF .AND.  LHFCALC) THEN
          LHFCALC_DAV=.TRUE.
          CALL NEWWAVA(WHF, WDES1, NSIM_LOCAL)
          CALL ALLOCW(WDES,W_ORIG)
          W_ORIG%CW       =W%CW
          W_ORIG%CPROJ    =W%CPROJ
          W_ORIG%CELTOT   =W%CELTOT
          W_ORIG%FERTOT   =W%FERTOT
          W_ORIG%OVER_BAND=W%OVER_BAND
       ENDIF
    ENDIF

    DESUM =0
    RMS   =0
    ICOUEV=0

    ! average local potential
    SLOCAL=0
    DO I=1,GRID%RL%NP
       SLOCAL=SLOCAL+SV(I,1)
    ENDDO

    CALLMPI( M_sum_d(WDES%COMM_INB, SLOCAL, 1))
    SLOCAL=SLOCAL/GRID%NPLWV

    DO I=1,NSIM_LOCAL
       CALL NEWWAV(W1(I), WDES1, .TRUE.)
    ENDDO

    COVL=0
    CHAM=0
    EXHF=0

    IF (PRESENT(EXHF_ACFDT)) EXHF_ACFDT=0
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS

    ! allocate array for the subspace diagonalisation
    IF (LSUBROT) THEN
       IF (.NOT. LscaAWARE) THEN
          ALLOCATE(CHAM_ALL(NB_TOT,NB_TOT))
       ELSE
          CALL INIT_scala(WDES%COMM_KIN, NB_TOT)
          ALLOCATE(CHAM_ALL(SCALA_NP(),SCALA_NQ()))
       ENDIF
    ENDIF
#ifdef  USE_ZHEEVX
    IF (LSUBROT) ALLOCATE(IWORK(5*WDES%NB_TOT),MINFO(WDES%NB_TOT))
#endif
    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ELSE
       MY_NKSTART=1
    ENDIF
!=======================================================================
    spin:    DO ISP=1,WDES%ISPIN
    kpoints: DO NK=MY_NKSTART,WDES%NKPTS
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
       CALL SETWDES(WDES,WDES1,NK)
       IF (LscaAWARE) CALL INIT_scala(WDES%COMM_KIN, WDES%NB_TOTK(NK,ISP))

       WA=ELEMENTS(W, WDES1, ISP)

       DE_ATT=ABS(W%CELTOT(WDES%NB_TOTK(NK,ISP),NK,ISP)-W%CELTOT(1,NK,ISP))/4

       IF (INFO%LREAL) THEN
          CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
       ELSE
          CALL PHASE(WDES,NONL_S,NK)
       ENDIF

       ! redistribute over plane wave coefficients
       IF (WDES%DO_REDIS) THEN
          CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
          CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
       ENDIF


       IF (LSUBROT .AND. .NOT. LscaAWARE) CHAM_ALL=0

       W1%NB=0       ! empty the list of bands, which are optimized currently
       NB_DONE=0     ! index of bands already optimised
!=======================================================================
       IF (LEMPTY) THEN
          IF (WDES%WTKPT(NK)/=0) THEN
             DO NB_DONE=NBANDS,1,-1
                IF (ABS(W%FERWE(NB_DONE,NK,ISP)) > INFO%WEIMIN) EXIT
             ENDDO
             WRITE(*,*) 'perform empty band optimisation starting at',NB_DONE
          ENDIF
       ENDIF

       bands: DO
!***********************************************************************
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!  please mind, that optimization is done in chunks of NCPU bands
!  presently, we have more functionality in the scheduler 
!  then actually used
!
!***********************************************************************
          newband: DO NP=1,NSIM_LOCAL
             IF (W1(NP)%NB==0 .AND.  NB_DONE < NBANDS ) THEN
                NB_DONE=NB_DONE+1
                N     =NB_DONE
                W1(NP)%NB=NB_DONE

                ITER=1
                NPP=(ITER-1)*NSIM_LOCAL+NP

                ! W1(NP) = WA(N)
                CALL W1_COPY( ELEMENT( WA, N), W1(NP))
                CALL W1_COPY( W1(NP), ELEMENT( WOPT, NPP) )

                ! now redistribute W1 over bands
                IF (WDES%DO_REDIS) THEN
                   CALL REDIS_PW(WDES1, 1, W1(NP)%CW(1))
                   CALL REDIS_PROJ(WDES1, 1, W1(NP)%CPROJ(1))
                ENDIF

                IDUMP=0
#ifdef debug
                IDUMP=2
#endif
#ifdef MPI
                IF (NODE_ME /= IONODE) IDUMP=0
#endif
                IF (IDUMP==2) WRITE(*,'(I3,1X)',ADVANCE='NO') N

                ! start with FFT and  exact evaluation of the eigenenergy 

                IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') REAL( W%CELEN(N,NK,ISP) ,KIND=q)

                CALL FFTWAV_W1(W1(NP))
                IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
                   CALL ECCP_VEC(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),HAMILTONIAN%AVEC,W%CELEN(N,NK,ISP))
                ELSEIF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                   CALL ECCP_TAU(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),W%CELEN(N,NK,ISP))
                ELSE
                   CALL ECCP(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W%CELEN(N,NK,ISP))
                ENDIF

                ! propagate calculated eigenvalues to all nodes

                EVALUE_INI(NP)=W%CELEN(N,NK,ISP)
                IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') REAL( W%CELEN(N,NK,ISP) ,KIND=q)
                ! calculate the preconditioning matrix
                CALL  TRUNCATE_HIGH_FREQUENCY_W1( W1(NP), LDELAY, INFO%ENINI)
                IPREC=INFO%IALGO
                IF (LDELAY) IPREC=8
                CALL SETUP_PRECOND( W1(NP), IPREC, IDUMP, PRECON(1,NP), & 
                     EVALUE_INI(NP)-SLOCAL, DE_ATT )
                IT(NP)  =0
                !=======================================================================
             ENDIF
          ENDDO newband

!=======================================================================
! increase iteration counter and check whether list is empty
!=======================================================================
          LSTOP=.TRUE.
          W1%LDO  =.FALSE.
          NSIM_LOCAL_=0
          DO NP=1,NSIM_LOCAL
             N=W1(NP)%NB
             IF ( N /= 0 ) THEN
                LSTOP  =.FALSE.
                W1(NP)%LDO=.TRUE.     ! band not finished yet
                IT(NP) =IT(NP)+1      ! increase iteration count
                NSIM_LOCAL_=NSIM_LOCAL_+1
             ENDIF
          ENDDO
          IF (LSTOP) EXIT bands

          ! right now all bands are consecutively ordered in W1 or WOPT%CW
          ! but it can happen that we treat less band in the last round
          NSIM_=NSIM_LOCAL_ * NCPU
!=======================================================================
! now calculate the HF part and update CELEN accordingly
!=======================================================================
          IF (LHFCALC_DAV) THEN
             IF (IT(1)==1) THEN
             ! first iteration
             ! update eigenvalues and d.c.
                CDCHF=0
                IF (PRESENT(EXHF_ACFDT)) THEN
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                        NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                        WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1, EXHF_ACFDT=EXHF_ACFDT)
                ELSE
                   CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                        NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                        WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1)
                ENDIF
                ! add Fock contribution to eigenvalues
                DO NP=1,NSIM_LOCAL
                   N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                   W%CELEN(N,NK,ISP)=W%CELEN(N,NK,ISP)+ &
                        W1_DOT( ELEMENT( W_ORIG, WDES1, N, ISP), ELEMENT (WHF, NP))
                ENDDO
                ! d.c. corrections
                EXHF=EXHF+CDCHF
             ELSE
                CDCHF=0
                CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_ORIG,   &
                     NONLR_S, NONL_S, NK, ISP, W1(1)%NB, W1(NSIM_LOCAL_)%NB-W1(1)%NB+1, &
                     WHF%CW(:,:), P, CQIJ(1,1,1,1), CDCHF, W1)
             ENDIF
          ENDIF
!=======================================================================
! finally distribute the eigenvalues onto all nodes
!=======================================================================
          IF (IT(1)==1) THEN
             EVALUE_GLBL=0
             DO NP=1,NSIM_LOCAL
                N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                EVALUE_INI(NP)=W%CELEN(N,NK,ISP)
                EVALUE_GLBL((NP-1)*NCPU+NODE_MEI)=W%CELEN(N,NK,ISP)
             ENDDO
             CALLMPI( M_sum_d(WDES%COMM_INTER,EVALUE_GLBL,NSIM_))
             EVALUE_INI_GLBL= EVALUE_GLBL
          ENDIF
!***********************************************************************
!
! intra-band optimisation
! first calculate (H - epsilon)  psi 
!
!***********************************************************************
!gK here I can shift the eigenvalues to zero
! this must not change the Hamilton matrix 
!      EVALUE_INI=0
!      EVALUE_INI_GLBL=0

          ITER=IT(1)
          NPP=(ITER-1)*NSIM_LOCAL_+1   ! storage position in WOPT%CW, WHAM%CW
          !  store H | psi > in WHAM%CW
          IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
             CALL HAMILTMU_VEC(WDES1, W1, NONLR_S, NONL_S, EVALUE_INI, &
             &     CDIJ, CQIJ, SV, HAMILTONIAN%AVEC, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ELSEIF (ASSOCIATED(HAMILTONIAN%MU)) THEN
             CALL HAMILTMU_TAU(WDES1, W1, NONLR_S, NONL_S, EVALUE_INI, &
             &     CDIJ, CQIJ, SV, LATT_CUR, HAMILTONIAN%MU, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ELSE
             CALL HAMILTMU(WDES1, W1(1:NSIM_LOCAL_), NONLR_S, NONL_S, EVALUE_INI, &
             &     CDIJ, CQIJ, SV, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ENDIF
          IF (LHFCALC_DAV) THEN
             ! add Fock contribution to (H -epsilon ) psi
             DO NP=1,NSIM_LOCAL_
                IF (.NOT. W1(NP)%LDO) CYCLE
                CALL W1_DAXPY( ELEMENT(WHF,NP),  1.0_q, ELEMENT(WHAM, NPP+NP-1))
             ENDDO
          ENDIF

          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE

             IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                WDES1%NBANDS=1    ! WDES1%NBANDS is used only here to fake OVERL
                CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
                IF (WDES%DO_REDIS) THEN
                   CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
                ENDIF
             ENDIF

             NPP=(ITER-1)*NSIM_LOCAL_+NP

             CALL TRUNCATE_HIGH_FREQUENCY_W1( ELEMENT(WHAM, NPP), LDELAY, INFO%ENINI)
             CALL PW_NORM_WITH_METRIC_W1( ELEMENT(WHAM, NPP), FNORM, FPRE_, PRECON(1,NP))

             CALL W1_COPY( ELEMENT( WHAM, NPP), W1(NP))

             IF (IDUMP==2) WRITE(*,'(E9.2,"R")',ADVANCE='NO') SQRT(ABS(FNORM))
             IF (ITER==1) THEN
                RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)* &
                     &      SQRT(ABS(FNORM))/NB_TOT*WDES%NRSPINORS
             ENDIF

             ! rearrange WHAM%CW
             IF (WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, WHAM%CW(1,NPP))
             ENDIF
          ENDDO
!***********************************************************************
!
! update the elements of the Hamilton matrix and of the overlap matrix
! in the space spanned by the present wave functions
!
!***********************************************************************
          LSTOP=.FALSE.

          ! calulcate CQIJ * W1%CPROJ (required for overlap)
          ! get the index into the redistributed array
          ITER=IT(1)                  ! iter is right now the same for each band

          NPOS_RED=(ITER-1)*NSIM_+1   ! storage position in redistributed WOPT%CW, WHAM%CW

          CHAM(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
          COVL(:,NPOS_RED:NPOS_RED+NSIM_-1)=0

          CALL ORTH1('U', &
               WOPT%CW_RED(1,1),WHAM%CW_RED(1,NPOS_RED),WOPT%CPROJ_RED(1,1), &
               WS%CPROJ_RED(1,1),NSUBD, &
               NPOS_RED, NSIM_, WDES1%NPL_RED, 0 ,WDES%NRPLWV_RED,WDES%NPROD_RED,CHAM(1,1))

          CALL ORTH1('U', &
               WOPT%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED),WOPT%CPROJ_RED(1,1), &
               WS%CPROJ_RED(1,1),NSUBD, &
               NPOS_RED, NSIM_, WDES1%NPL_RED, WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL(1,1))

          CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,NPOS_RED),NSUBD*NSIM_))
          CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM(1,NPOS_RED),NSUBD*NSIM_))

          ! add remaining elements to COVL
          DO M=1,NSIM_*(ITER-1)
             DO I=1,NSIM_
                COVL(NPOS_RED-1+I,M)=GCONJG(COVL(M,NPOS_RED-1+I))
             ENDDO
          ENDDO
          ! correct CHAM by subtraction of epsilon COVL
          DO M=1,NSIM_*ITER
             DO I=1,NSIM_
                CHAM(M,NPOS_RED-1+I)=CHAM(M,NPOS_RED-1+I)+COVL(M,NPOS_RED-1+I)*EVALUE_INI_GLBL(I)
             ENDDO
          ENDDO

#ifndef gammareal
          DO N1=1,NSIM_*ITER
             IF (ABS(AIMAG(CHAM(N1,N1)))/MAX(1._q,ABS(REAL(CHAM(N1,N1), q)))>1E-5_q) THEN
                WRITE(*,*)'WARNING: Sub-Space-Matrix is not hermitian in DAV ',N1, &
                     AIMAG(CHAM(N1,N1))
             ENDIF
             CHAM(N1,N1)= REAL( CHAM(N1,N1) ,KIND=q)
          ENDDO
#endif

          ! solve eigenvalue-problem and calculate lowest eigenvector
          ! this eigenvector corresponds to a minimal residuum
          ! CHAM(n1,n2) U(n2,1) = E(1) S(n1,n2)  U(n2,1)

          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "Hamiltonian", WDES, CHAM, SIZE(CHAM,1), ITER*NSIM_)
             CALL DUMP_HAM_SELECTED( "overlap", WDES, COVL, SIZE(COVL,1), ITER*NSIM_)
             io_end
          ENDIF

          IF (NODE_ME==IONODE) THEN
             ! estimate L1 norm of the inverse of the overlap matrix
             ! which is essentially the condition number (since L1 norm of COVL is 1)
             COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
#ifdef gammareal
             CALL DPOTRF &
#else
             CALL ZPOTRF &
#endif
             & ('U',ITER*NSIM_,COVL_,NSUBD,IFAIL)
             IF (IFAIL==0) THEN
#ifdef gammareal
                CALL DPOCON &
#else 
                CALL ZPOCON &
#endif
                &  ( 'U', ITER*NSIM_,COVL_,NSUBD, 1.0_q, RCOND, CWRK, RWORK, IFAIL )
             ENDIF

             ! ok if that is less than 1E-13 to 1E-14 stop
             ! to have some headroom I set it to 1E-13 (1E-14 worked in my tests however)
             IF (ABS(RCOND)<1E-13 .AND. IFAIL==0 ) IFAIL=1
             IF (IFAIL==0) THEN
                CEIG (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
                COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
#ifdef gammareal
                CALL DSYGV &
                &  (1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R, &
                &           CWRK(1),LWORK*NB_TOT,IFAIL)
#else 
                CALL ZHEGV &
                &  (1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R, &
                &           CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
#ifdef debug
                WRITE(*,*) ITER, NSIM_, IFAIL
                WRITE(*,'(8F12.5)') R(1:ITER*NSIM_)
#endif
             ENDIF
          ENDIF

          ! communicate IFAIL to all nodes
          CALLMPI( M_bcast_i(WDES%COMM_KIN, IFAIL, 1))

          IF (IFAIL/=0) THEN
             IF (ITER>=3) THEN
             ITER=ITER-1
             CEIG (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
             COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
#ifdef gammareal
             CALL DSYGV &
               &  (1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R, &
               &           CWRK(1),LWORK*NB_TOT,IFAIL)
#else
             CALL ZHEGV &
               &  (1,'V','U',ITER*NSIM_,CEIG,NSUBD,COVL_,NSUBD,R, &
               &           CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
             ENDIF
             IF (IFAIL/=0) THEN
                IF (IU6>=0) &
                WRITE(IU6,219) IFAIL,ITER,ITER*NSIM_
                IF (IU0>=0) &
                WRITE(IU0,219) IFAIL,ITER,ITER*NSIM_
                STOP
             ENDIF
             ! force stop now (can get only worse)
             LSTOP=.TRUE.
             IT(:)=ITER
          ENDIF
219       FORMAT('Error EDDDAV: Call to ZHEGV failed. Returncode =',I4,I2,I4)

          ! now broadcase to make sure all nodes use identical CEIG
          CALLMPI( M_bcast_g(WDES%COMM_KIN, CEIG, NSUBD*ITER*NSIM_))
          CALLMPI( M_bcast_d(WDES%COMM_KIN, R, ITER*NSIM_))

          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "reconstructed Hamiltonian", WDES, CEIG, SIZE(CEIG,1), ITER*NSIM_)
             io_end
          ENDIF
!-----------------------------------------------------------------------
! update energies and calculate total energy change
!-----------------------------------------------------------------------
          II=0
          DEMAX=0
          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
             DO NPOS_RED=(N-1)*NCPU+1,N*NCPU
                II=II+1
                W%CELTOT(NPOS_RED,NK,ISP)=R(II) ! update CELTOT array
                DECEL =R(II)-EVALUE_GLBL(II)    ! change in eigenenergy

                ! if the change in the eigenenergy is very small 
                DEMAX=MAX(DEMAX, ABS(DECEL))

                IF (IDUMP==2)  WRITE(*,'(E10.2,2H |)',ADVANCE='NO') DECEL
                DESUM =DESUM +WDES%RSPIN*WDES%WTKPT(NK)*W%FERTOT(NPOS_RED,NK,ISP)*DECEL
                EVALUE_GLBL(II)         =R(II)  ! update 

             ENDDO
          ENDDO
!-----------------------------------------------------------------------
! possibly break the optimisation
! and new eigenenergy
!-----------------------------------------------------------------------
          ITER=IT(1)

          ! break if absolute change in eigenenergy is small
          ! -------------------------------------------------
          IF (ITER>1 .AND. DEMAX < INFO%EBREAK) LSTOP=.TRUE.
          ! relative break criterion 
          ! -------------------------------------------------
          IF (ITER==2) DEIT=DEMAX*INFO%DEPER
          IF (ITER>2 .AND. DEMAX < DEIT) LSTOP=.TRUE.

          NITER_NOW=NITER

          ! sufficient iterations done
          ! -------------------------------------------------
          IF (ITER >= NITER_NOW) LSTOP=.TRUE.
          IF (ITER >= NITER)     LSTOP=.TRUE.  ! certainly stop if storage requires this
!=======================================================================
! if stopping is selected store the optimised wave function back
!=======================================================================
          IF (LSTOP) THEN
             IF (IDUMP==2)  WRITE(*,*)

             NPOS_RED=(W1(1)%NB-1)*NCPU

             IF (WDES1%NPL_RED/=0) &
                  CALL GGEMM('N', 'N', m_ WDES1%NPL_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CW_RED(1,1), m_ WDES%NRPLWV_RED, CEIG(1,1), NSUBD,  &
                  &               zero, WA%CW_RED(1,NPOS_RED+1), m_ WDES%NRPLWV_RED)

             IF (WDES1%NPRO_RED/=0) &
                  CALL GGEMM('N', 'N', WDES1%NPRO_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CPROJ_RED(1,1), WDES%NPROD_RED, CEIG(1,1), NSUBD,  &
                  &               zero, WA%CPROJ_RED(1,NPOS_RED+1), WDES%NPROD_RED)

! idential F90 version
!      DO II=1,NSIM_
!        CW_RED(1:NPL,NPOS_RED+II)     = 0
!        WA%CPROJ_RED(1:NPRO,NPOS_RED+II) = 0
!        DO I=1,NSIM_*ITER
!           DO M=1,NPL
!              CW_RED   (M,NPOS_RED+II)=CW_RED   (M,NPOS_RED+II)+CEIG(I,II)*WOPT%CW_RED(M,I)
!           ENDDO
!           DO M=1,NPRO
!              WA%CPROJ_RED(M,NPOS_RED+II)=WA%CPROJ_RED(M,NPOS_RED+II)+CEIG(I,II)*WOPT%CPROJ_RED(M,I)
!           ENDDO
!        ENDDO
!     ENDDO


             IF (LSUBROT) THEN
                ! store in the corresponding (H - epsilon S)  in WOPT%CW_RED

                DO II=1,NSIM_
                   WOPT%CW_RED(1:WDES1%NPL_RED,II)      =0
                   DO I=1,NSIM_*ITER
!DIR$ SIMD
                      DO M=1,WDES1%NPL_RED
                         WOPT%CW_RED   (M,II)=WOPT%CW_RED   (M,II)+CEIG(I,II)*WHAM%CW_RED(M,I)
                      ENDDO
                   ENDDO
                ENDDO
                ! calculate epsilon COVL
                NPOS_RED =(W1(1)%NB-1)*NCPU+1

                IF (.NOT. LscaAWARE) THEN
                   CALL ORTH1('U', &
                     WA%CW_RED(1,1),WOPT%CW_RED(1,1),WA%CPROJ_RED(1,1), &
                     WA%CPROJ_RED(1,1),NB_TOT, &
                     NPOS_RED, NSIM_, WDES1%NPL_RED,0 ,WDES%NRPLWV_RED,WDES%NPROD_RED,CHAM_ALL(1,1))

                ! correct the small NSIM_ times NSIM_ block
                ! which is incorrect since we have calculate H - S epsilon psi
                ! and not H psi and since  our psi are not orthogonal to each other
                ! this block is however anyway diagonal with the elements R(I)
                   CHAM_ALL(NPOS_RED:NPOS_RED+NSIM_-1,NPOS_RED:NPOS_RED+NSIM_-1)=0
                   IF (NODE_ME==IONODE) THEN
                      DO I=1,NSIM_
                         CHAM_ALL(NPOS_RED-1+I,NPOS_RED-1+I)=R(I)
                      ENDDO
                   ENDIF
                ELSE
                   CALL ORTH1_DISTRI_DAVIDSON('U', &
                     WA%CW_RED(1,1),WOPT%CW_RED(1,1),WA%CPROJ_RED(1,1), &
                     WA%CPROJ_RED(1,1),NB_TOT, &
                     NPOS_RED, NSIM_, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM_ALL(1,1), & 
                     R(1),WDES%COMM_KIN,WDES%NB_TOTK(NK,ISP))
                ENDIF
             ENDIF
             W1%NB=0

             CYCLE bands
          ENDIF
!***********************************************************************
!
! next step increase interaction count
!
!***********************************************************************
          ICOUEV=ICOUEV+NSIM_
!-----------------------------------------------------------------------
! preconditioning of calculated residual vectors
!-----------------------------------------------------------------------
          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP)+1; IF (.NOT. W1(NP)%LDO) CYCLE
             CALL APPLY_PRECOND( W1(NP), W1(NP), PRECON(1,NP))
          ENDDO
!-----------------------------------------------------------------------
! calculate the wavefunction character of these vectors and redistribute
!-----------------------------------------------------------------------
          IF ( INFO%LREAL ) THEN
             DO NP=1,NSIM_LOCAL_
                N=W1(NP)%NB; ITER=IT(NP)+1; IF (.NOT. W1(NP)%LDO) CYCLE
                CALL FFTWAV_W1(W1(NP))
             ENDDO
          ENDIF
          CALL W1_PROJALL(WDES1, W1, NONLR_S, NONL_S, NSIM_LOCAL_)
          
          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
             CALL CNORMN(W1(NP),CQIJ, ISP, WSCAL)

             NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW

             CALL W1_COPY(W1(NP), ELEMENT(WOPT, NPP))
             
             IF (WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, WOPT%CW(1,NPP))
                CALL REDIS_PROJ(WDES1, 1, WOPT%CPROJ(1,NPP))
             ENDIF
             
             IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                WDES1%NBANDS=1    ! is used this only here not quite clean
                CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
                IF (WDES%DO_REDIS) THEN
                   CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
                ENDIF
             ENDIF
          ENDDO
!-----------------------------------------------------------------------
! overlap and orthogonalisation
!-----------------------------------------------------------------------
          NPOS_RED=ITER*NSIM_+1   ! storage position in WOPT%CW_RED, WHAM%CW_RED
          
          CORTHO=0
          
          CALL ORTH1('L', &
               WA%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
               WS%CPROJ_RED(1,1),NB_TOT, &
               1, NSIM_, WDES1%NPL_RED, WDES1%NPRO_O_RED ,WDES%NRPLWV_RED,WDES%NPROD_RED,CORTHO(1,1))

          CALLMPI( M_sum_g(WDES%COMM_KIN, CORTHO(1,1), NB_TOT*NSIM_))
          
          IF (.FALSE.) THEN
             io_begin
             CALL DUMP_HAM_SELECTED( "overlap", WDES, CORTHO, SIZE(CORTHO,1), NB_TOT)
             io_end
          ENDIF
          
          IF (WDES1%NPL_RED /=0 ) &
               CALL GGEMM( 'N', 'N' , m_ WDES1%NPL_RED , NSIM_ , NB_TOT , -one , &
               WA%CW_RED(1,1), m_ WDES%NRPLWV_RED , CORTHO(1,1) , NB_TOT , &
               one , WOPT%CW_RED(1,NPOS_RED) , m_ WDES%NRPLWV_RED )
          
          IF (WDES1%NPRO_RED /= 0) &
               CALL GGEMM( 'N', 'N' ,  WDES1%NPRO_RED , NSIM_ , NB_TOT  , -one , &
               WA%CPROJ_RED(1,1) ,  WDES%NPROD_RED , CORTHO(1,1) , NB_TOT , &
               one , WOPT%CPROJ_RED(1,NPOS_RED) ,  WDES%NPROD_RED  )
          
!-----------------------------------------------------------------------
! now store the results back in W1, and perform an FFT to real space
!-----------------------------------------------------------------------
          DO NP=1,NSIM_LOCAL_
             
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
             NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW
             
             ! W1(NP) = WOPT(NNP)
             CALL W1_COPY( ELEMENT( WOPT, NPP), W1(NP))

             ! distribute W1 over bands
             IF (WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, W1(NP)%CW(1))
                CALL REDIS_PROJ(WDES1, 1, W1(NP)%CPROJ(1))
             ENDIF
                                ! normalize the wave function
             CALL FFTWAV_W1(W1(NP))
          ENDDO
!=======================================================================
! move onto the next bands
!=======================================================================
       ENDDO bands
      
!***********************************************************************
!
! last step perform the sub space rotation
!
!***********************************************************************
       subr: IF(LSUBROT) THEN
          ! sum subspace matrix over all nodes
          IF (.NOT. LscaAWARE) THEN
             CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM_ALL(1,1),NB_TOT*NB_TOT))
             ! add lower triangle
             DO N=1,NB_TOT
                DO NP=N+1,NB_TOT
                   CHAM_ALL(NP,N)=GCONJG(CHAM_ALL(N,NP))
                ENDDO
             ENDDO
          ENDIF

#ifdef debug
          io_begin
          CALL DUMP_HAM_SELECTED( "final HAM", WDES, CHAM_ALL, SIZE(CHAM_ALL,1), NB_TOT)
          io_end
#endif

#ifndef gammareal
          DO N1=1,NB_TOT
             IF (.NOT. LscaAWARE) THEN
             
                IF (ABS(AIMAG(CHAM_ALL(N1,N1)))>1E-2_q .AND. IU0>=0) THEN
                   WRITE(IU0,*)'WARNING: Sub-Space-Matrix is not hermitian subr', &
                        &              AIMAG(CHAM_ALL(N1,N1)),N1
                ENDIF
                CHAM_ALL(N1,N1)= REAL( CHAM_ALL(N1,N1) ,KIND=q)
             ELSE
                CALL BG_CHANGE_DIAGONALE(WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),IU0)
             ENDIF
          ENDDO
#endif

          ! here we support only scaLAPACK and LAPACK 

#if defined(MPI)
          ! use scaLAPACK if available in parallel version
          IF ( LscaLAPACK .AND. .NOT. LscaAWARE) THEN
             CALL pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM_ALL(1,1), R,  NB_TOT, WDES%NB_TOTK(NK,ISP))
             CALLMPI( M_sum_g(WDES%COMM_KIN, CHAM_ALL(1,1),NB_TOT*NB_TOT))
             GOTO 1000
          ELSE IF ( LscaLAPACK ) THEN
             CALL BG_pDSSYEX_ZHEEVX(WDES%COMM_KIN, CHAM_ALL(1,1), R,  WDES%NB_TOTK(NK,ISP))
             GOTO 1000
          ENDIF
#endif
          !
          !  seriel codes
          !
#ifdef  gammareal

#ifndef USE_ZHEEVX
          CALL DSYEV &
               ('V','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT, &
               R,CWRK,LWORK*NB_TOT, IFAIL)
#else
          ABSTOL=1E-10_q
          VL=0 ; VU=0 ; IL=0 ; IU=0
          ALLOCATE(COVL_ALL(NB_TOT,NB_TOT))

          CALL DSYEVX( 'V', 'A', 'U', WDES%NB_TOTK(NK,ISP), CHAM_ALL(1,1) , NB_TOT, VL, VU, IL, IU, &
               ABSTOL , NB_CALC , R, COVL_ALL(1,1), NB_TOT, CWRK, &
               LWORK*NB_TOT, RWORK, IWORK, MINFO, IFAIL )         
          CHAM_ALL=COVL_ALL
          DEALLOCATE(COVL_ALL)
#endif
#else
#ifndef USE_ZHEEVX
          CALL ZHEEV &
               ('V','U',WDES%NB_TOTK(NK,ISP),CHAM_ALL(1,1),NB_TOT, &
               R,CWRK,LWORK*NB_TOT, RWORK,  IFAIL)
#else
          ABSTOL=1E-10_q
          VL=0 ; VU=0 ; IL=0 ; IU=0
          ALLOCATE(COVL_ALL(NB_TOT,NB_TOT))
          CALL ZHEEVX( 'V', 'A', 'U', WDES%NB_TOTK(NK,ISP), CHAM_ALL(1,1) , NB_TOT, VL, VU, IL, IU, &
               ABSTOL , NB_CALC , R, COVL_ALL(1,1), NB_TOT, CWRK, &
               LWORK*NB_TOT, RWORK, IWORK, MINFO, IFAIL )         
          CHAM_ALL=COVL_ALL
          DEALLOCATE(COVL_ALL)
#endif
#endif
          ! T3D uses a global sum which does not guarantee to give the same results on all nodes
          ! the following line is required to make the code waterproof (we had problems)
          ! since we now use a propritary sum (see mpi.F) we should not require
          ! this broadcast anymore
          !CALLMPI( M_bcast_g(WDES%COMM, CHAM_ALL(1,1), NB_TOT*NB_TOT))

1000      CONTINUE

          IF (IFAIL/=0) THEN
             WRITE(*,*) 'ERROR EDDIAG: Call to routine ZHEEV failed! '// &
                  &              'Error code was ',IFAIL
             WRITE(*,*) ' try to use ALGO = Exact if you use many bands (exact diagonalization)'
             STOP
          ENDIF

          DO N=1,WDES%NB_TOTK(NK,ISP)
             W%CELTOT(N,NK,ISP)=R(N)
          ENDDO

          IF ( .NOT. LscaAWARE) THEN
             CALL LINCOM('F',WA%CW_RED,WA%CPROJ_RED,CHAM_ALL(1,1), &
                  WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
                  WDES1%NPL_RED,WDES1%NPRO_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,NB_TOT, &
                  WA%CW_RED,WA%CPROJ_RED)
          ELSE
             CALL LINCOM_DISTRI('F',WA%CW_RED(1,1),WA%CPROJ_RED(1,1),CHAM_ALL(1,1), &
                  WDES%NB_TOTK(NK,ISP), & 
                  WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
                  WDES%COMM_KIN, NBLK)
          ENDIF
          DWRITE "lincom ok"

       ENDIF subr

       IF (WDES%DO_REDIS) THEN
          CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
          CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
       ENDIF
    END DO kpoints
    ENDDO spin

#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_CELTOT(WDES,W)
    ENDIF

    IF ((LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
    ENDIF
#endif

    CALLMPI( M_sum_d(WDES%COMM_KINTER, DESUM, 1))
    CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV, 1))

    ! RMS was only calculate for the band treated locally (sum over all nodes)
    CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))
    CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))

    DEALLOCATE(PRECON, CHAM, COVL, CEIG, COVL_, CORTHO, R, RWORK, CWRK)
    IF (LSUBROT) DEALLOCATE(CHAM_ALL)
#ifdef  USE_ZHEEVX
    IF (LSUBROT) DEALLOCATE(IWORK,MINFO)
#endif
    DO I=1,NSIM_LOCAL
       CALL DELWAV(W1(I), .TRUE.)
    ENDDO

    CALL DELWAVA(WOPT)
    CALL DELWAVA(WHAM)
    CALL DELWAVA_PROJ(WS)

    IF (LHFCALC_DAV) THEN
       CALL DELWAVA(WHF)
       CALL DEALLOCW(W_ORIG)
       CALLMPI( M_sum_d(WDES1%COMM_KIN,EXHF,1))
       CALLMPI( M_sum_d(WDES1%COMM_KINTER,EXHF,1))
       IF (PRESENT(EXHF_ACFDT)) THEN
          CALLMPI( M_sum_d(WDES1%COMM_KIN,EXHF_ACFDT,1))
          CALLMPI( M_sum_d(WDES1%COMM_KINTER,EXHF_ACFDT,1))
       ENDIF

    ENDIF

    PROFILING_STOP('eddav')

    RETURN
  END SUBROUTINE EDDAV
END MODULE david
