#include "symbol.inc"
#define USE_ZHEEVX

MODULE david_inner
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

  SUBROUTINE EDDAV_INNER(HAMILTONIAN, P, GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, NSIM, &
       LMDIM,CDIJ,CQIJ, RMS,DESUM,ICOUEV, SV, EXHF, IU6, IU0, &
       LEMPTY, NKSTART, EREF)
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
    LOGICAL LEMPTY                   ! optimize emtpy bands only
    INTEGER, OPTIONAL :: NKSTART     ! start k-point
    REAL(q), OPTIONAL :: EREF
 ! local work arrays
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    TYPE (wavefuna)    WA            ! wavefunction array, points to W section for one k and spin
    TYPE (wavefuna)    WOPT          ! subset of wavefunctions currently optimized
    TYPE (wavefuna)    WHAM          ! action of H on these wavefunctions
    TYPE (wavefuna)    WHELP          ! 
    TYPE (wavefuna)    WS            ! stores CQIJ * wavefunctions
    TYPE (wavefuna)    WS1            ! stores CQIJ * wavefunctions
    TYPE (wavefuna)    WLAST         ! stores last optimized wavefunctions for restart
    GDEF, ALLOCATABLE      ::   CORTHO(:,:) ! stores <psi_i| S |psi_k>
    TYPE (wavefun1)    W1(NSIM)      ! wavefunction currently added to subspace
                                     ! redistributed plane wave coefficients
    REAL(q),ALLOCATABLE:: PRECON(:,:)! preconditioning matrix for each
    GDEF,ALLOCATABLE:: CHAM(:,:),COVL(:,:),CEIGVEC(:,:),CEIGVEC_LAST(:,:),COVL_(:,:),CTMP(:,:),COVL_HARM(:,:),&
         ALPHA(:),ALPHAR(:),ALPHAI(:),BETA(:),RV(:,:),LV(:,:)
    REAL(q) :: EVALUE_IN_S(NSIM)     ! value used when calculating  E S| phi>
    REAL(q) :: EVALUE_IN_S_GLBL(NSIM)! same as eigenvalue available on all nodes
    REAL(q) :: EVALUE_INI(NSIM)      ! initial estimate for eigenvalue <phi | H | phi>
    REAL(q) :: EVALUE_INI_GLBL(NSIM) ! same as above but available on all nodes
    COMPLEX(q) :: CEVALUE_IN_S(NSIM) ! presently not used
    REAL(q) :: EVALUE_GLBL(NSIM)     ! same as eigenvalue but available on all nodes
    REAL(q) :: MAXR                 
    REAL(q) :: DEIT                  ! relative break criterion for that band
    INTEGER :: IT(NSIM)              ! current iteration for this band
    LOGICAL :: LSTOP                 ! optimisation finished
    LOGICAL :: DO_REDIS              ! redistribution of wavefunctions required

    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER  :: LWORK=8192
    GDEF,ALLOCATABLE    :: CWRK(:)
    REAL(q),ALLOCATABLE ::  R(:),RR(:),RWORK(:),RESIDUAL(:)
    COMPLEX(q),ALLOCATABLE :: CE(:)
    INTEGER :: NB_MAX, MY_NKSTART
!gK TODO
!    INTEGER,PARAMETER :: IRWORK=3 ! should be ok
    INTEGER,PARAMETER :: IRWORK=7
    REAL (q)   :: ABSTOL=1E-10_q, VL, VU 
    INTEGER    :: IL, IU, NB_CALC
    ! more local variables
    INTEGER :: NODE_ME, NODE_MEI, IONODE, NCPU, NSIM_LOCAL, NSIM_, NSIM_LOCAL_, &
         NSUBD, NITER, I, J, NBANDS, NB_TOT, ISP, NK, NB_START, &
         NB_DONE, NP, N, ITER, NPP, M, MM,  IDUMP, &
         ISPINOR, N1, N2, NPOS_RED, IFAIL, II, NITER_NOW, IPREC, NDIM, NDIM_
    REAL(q) :: SLOCAL, DE_ATT, EKIN, FAKT, X, X2, FNORM, FNORMALL,FPRE_, DECEL, DEMAX, WSCAL
    COMPLEX(q) :: CPT, CDCHF
    REAL(q) :: EXHF
    INTEGER :: LOWER, UPPER
    GDEF,ALLOCATABLE :: WORK1(:)
    REAL(q),ALLOCATABLE :: RWORK1(:)
    INTEGER :: LWORK1
    INTEGER, ALLOCATABLE :: KEY(:)
    LOGICAL :: LHARMONIC
!=======================================================================
! initialise the required variables for MPI
!=======================================================================

#ifdef MPI
    NODE_ME=WDES%COMM%NODE_ME
    IONODE =WDES%COMM%IONODE
    NODE_MEI=WDES%COMM_INTER%NODE_ME ! number of groups (each group holds one band)
    NCPU   =WDES%COMM_INTER%NCPU     ! number of groups (each group holds one band)
    IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('EDDAV_INNER: KPAR>1 not implemented, sorry.') )
!PK Just copy the recipe over from davidson.F
       STOP
    END IF
#else
    NODE_ME=1
    NODE_MEI=1
    IONODE =1
    NCPU=1
#endif

    IF(INFO%IHARMONIC==1) LHARMONIC=.TRUE.
!=======================================================================
! number of bands treated simultaneously this must be a multiple of NCPU
!=======================================================================
    NSIM_LOCAL=NSIM/NCPU  ! number of bands optimised on this node
    IF (NSIM_LOCAL*NCPU /= NSIM) THEN
       WRITE(*,*) 'internal ERROR in EDDAV NSIM is not correct',NSIM
       STOP
    ENDIF

    ! maximum number of iterations
    NITER=MAX(2,INFO%NDAV+1)

    ! at least one optimisation step
    NSUBD =NITER*NSIM         ! maximum size of the subspace
    NB_MAX=NSUBD

    CALL SETWDES(WDES,WDES1,0)

    CALL NEWWAVA(WLAST, WDES1, WDES%NB_TOT)
    CALL NEWWAVA(WOPT, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA(WHELP, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA(WHAM, WDES1, NSIM_LOCAL*NITER)
    CALL NEWWAVA_PROJ(WS, WDES1, NSIM_LOCAL)
    CALL NEWWAVA_PROJ(WS1, WDES1, NSIM_LOCAL*NITER)

    ALLOCATE(PRECON(WDES%NRPLWV,NSIM_LOCAL), &
         CHAM(NSUBD,NSUBD),COVL(NSUBD,NSUBD),COVL_HARM(NSUBD,NSUBD),CEIGVEC(NSUBD,NSUBD),CEIGVEC_LAST(NSUBD,NSUBD),COVL_(NSUBD,NSUBD), &
         CORTHO(WDES%NB_TOT,NSIM),R(NB_MAX),RR(NB_MAX),RWORK(NB_MAX*IRWORK),CWRK(LWORK*WDES%NB_TOT),&
         ALPHA(NSUBD),ALPHAR(NSUBD),ALPHAI(NSUBD),BETA(NSUBD),RV(NSUBD,NSUBD),LV(1,1) )
    LWORK1=8*NSUBD+16
    ALLOCATE(RESIDUAL(NSIM),KEY(NSUBD),WORK1(LWORK1),RWORK1(LWORK1))

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
       CALL NEWWAV(W1(I) , WDES1,.TRUE.)
    ENDDO

    COVL=0
    CHAM=0
    EXHF=0
    COVL_HARM=0
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS

    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ELSE
       MY_NKSTART=1
    ENDIF

!=======================================================================
    spin:    DO ISP=1,WDES%ISPIN
    kpoints: DO NK=MY_NKSTART,WDES%NKPTS
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
                CALL CNORMN(W1(NP), CQIJ, ISP, WSCAL)
                IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
                   CALL ECCP_VEC(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),HAMILTONIAN%AVEC,W%CELEN(N,NK,ISP))
                ELSEIF (ASSOCIATED(HAMILTONIAN%MU)) THEN
                   CALL ECCP_TAU(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),W%CELEN(N,NK,ISP))
                ELSE
                   CALL ECCP(WDES1,W1(NP),W1(NP),LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W%CELEN(N,NK,ISP))
                ENDIF
                IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') REAL( W%CELEN(N,NK,ISP) ,KIND=q)
                ! calculate the preconditioning matrix
                IPREC=INFO%IALGO
                CALL SETUP_PRECOND( W1(NP), IPREC, IDUMP, PRECON(1,NP), & 
                     REAL(W%CELEN(N,NK,ISP),q)-SLOCAL, DE_ATT )
                IT(NP)=0

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
! finally distribute the eigenvalues onto all nodes (first iteration)
!=======================================================================
          IF (IT(1)==1) THEN
            EVALUE_IN_S_GLBL=0
             DO NP=1,NSIM_LOCAL
                N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                EVALUE_IN_S(NP)=W%CELEN(N,NK,ISP)
                EVALUE_IN_S_GLBL((NP-1)*NCPU+NODE_MEI)=W%CELEN(N,NK,ISP)
             ENDDO
             CALLMPI( M_sum_d(WDES%COMM_INTER,EVALUE_IN_S_GLBL,NSIM_))
             EVALUE_GLBL= EVALUE_IN_S_GLBL

             EVALUE_INI=EVALUE_IN_S
             EVALUE_INI_GLBL=EVALUE_IN_S_GLBL
             IF(PRESENT(EREF))THEN
                EVALUE_IN_S=EREF
                EVALUE_IN_S_GLBL=EREF
             END IF
          ENDIF
!***********************************************************************
!
! the Davidson scheme implemented here builds up the
! iterative space setting out from a set of initial vectors
! {psi^n_0| n=lower_band...upper_band} and adds the sets
! psi^n_1 = P (H - S epsilon_n)  psi^n_0
! psi^n_2 = P (H - S epsilon_n)  psi^n_1
! psi^n_3 = P (H - S epsilon_n)  psi^n_2 etc.
!
! psi^n_0 is the initial guess orbital and stored in W1
! this is not ideal for restarting, since the three step recurrence
! scheme consisting of
!   w^n_iter-1, w^n_iter, P (H - S epsilon_n)  w^n_iter
! will result in optimal performance (identical to CG) for quadratic
! problems.
! Here w_^n_iter is the best vector in the iteration iter
!
!***********************************************************************
!
! calculate (H - S epsilon)  psi^n_iter
! and store result in WHAM and subsequently in back into W1
! overwriting psi^n_iter
!
!***********************************************************************
#define ortho_first
          ! first calculate the non-local part of the overlap Q acting on psi^n_iter
          IF (INFO%LOVERL .AND. WDES%NPROD>0) THEN
             DO NP=1,NSIM_LOCAL_
                ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
                WDES1%NBANDS=1  ! WDES1%NBANDS is used only here to fake OVERL
                CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
                NPP=(ITER-1)*NSIM_LOCAL_+NP
                CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS1%CPROJ(1,NPP))
                IF (WDES%DO_REDIS) THEN
                   CALL REDIS_PROJ(WDES1, 1, WS%CPROJ(1,NP))
                   CALL REDIS_PROJ(WDES1, 1, WS1%CPROJ(1,NPP))
                ENDIF
             ENDDO
          ENDIF

          ITER=IT(1)
          NPP=(ITER-1)*NSIM_LOCAL_+1   ! storage position in WOPT%CW, WHAM%CW

          !  store H | psi > in WHAM%CW
          IF (ASSOCIATED(HAMILTONIAN%AVEC)) THEN
             CALL HAMILTMU_VEC(WDES1, W1, NONLR_S, NONL_S, EVALUE_IN_S, &
             &     CDIJ, CQIJ, SV, HAMILTONIAN%AVEC, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ELSEIF (ASSOCIATED(HAMILTONIAN%MU)) THEN
             CALL HAMILTMU_TAU(WDES1, W1, NONLR_S, NONL_S, EVALUE_IN_S, &
             &     CDIJ, CQIJ, SV, LATT_CUR, HAMILTONIAN%MU, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ELSE
             CALL HAMILTMU(WDES1, W1, NONLR_S, NONL_S, EVALUE_IN_S, &
             &     CDIJ, CQIJ, SV, ISP, ELEMENTS(WHAM, NPP, NPP+NSIM_LOCAL_-1))
          ENDIF

!gJ-test: calculate residual
          IF(ITER==1)THEN
             CALL HAMILTMU(WDES1,W1, NONLR_S, NONL_S, EVALUE_INI, &
                  &     CDIJ, CQIJ, SV, ISP, ELEMENTS(WHELP, 1, NSIM_LOCAL_))
             RESIDUAL=0
             DO NP=1,NSIM_LOCAL_
                N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
                CALL PW_NORM_WITH_METRIC_W1( ELEMENT(WHELP,NP), RESIDUAL( (NP-1)*NCPU+NODE_MEI ), FPRE_, PRECON(1,NP) )
                RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)* &
                     &      SQRT(ABS(RESIDUAL(((NP-1)*NCPU+NODE_MEI))))/NB_TOT*WDES%NRSPINORS
             END DO
             CALLMPI( M_sum_d(WDES%COMM_INTER,RESIDUAL,NSIM_) )
             II=0
             DO NP=1,NSIM_LOCAL_
                N=W1(NP)%NB; IF (.NOT. W1(NP)%LDO) CYCLE
                DO NPOS_RED=(N-1)*NCPU+1,N*NCPU
                   II=II+1
                   W%AUXTOT(NPOS_RED,NK,ISP)=SQRT(ABS(RESIDUAL(II)))
                ENDDO
             ENDDO   
!gK If we know the true residual add it here rather than below
          END IF

          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE

             NPP=(ITER-1)*NSIM_LOCAL_+NP

             CALL PW_NORM_WITH_METRIC_W1( ELEMENT(WHAM, NPP), FNORM, FPRE_, PRECON(1,NP))

             ! now copy the residual vector (H-epsilon) psi  back to W1
             CALL W1_COPY( ELEMENT( WHAM, NPP), W1(NP))

             IF (IDUMP==2) WRITE(*,'(E9.2,"R")',ADVANCE='NO') SQRT(ABS(FNORM))
!gK: removed fermi-weight, and moved to calculations of true norm of residual above
!             IF (ITER==1) THEN
!                RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)* &
!                     &      SQRT(ABS(FNORM))/NB_TOT*WDES%NRSPINORS
!             ENDIF

             IF(LHARMONIC)THEN
                IF ( INFO%LREAL ) CALL FFTWAV_W1(W1(NP))
             END IF

             ! distribute WHAM%CW such that the vectors are distributed over
             ! nodes 
             ! this allows for parallel scalar products used below
             IF (WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, WHAM%CW(1,NPP))
             ENDIF
          ENDDO


          IF(LHARMONIC)THEN
             CALL W1_PROJALL(WDES1, W1, NONLR_S, NONL_S, NSIM_LOCAL_)
             DO NP=1,NSIM_LOCAL_
                ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
                NPP=(ITER-1)*NSIM_LOCAL_+NP
                
                CALL W1_COPY_CPROJ( W1(NP), ELEMENT( WHAM, NPP))
                IF (WDES%DO_REDIS) THEN
                   CALL REDIS_PROJ(WDES1, 1, WHAM%CPROJ(1,NPP))
                ENDIF
             ENDDO
          END IF
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
          COVL_HARM(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
          IF (LHARMONIC) THEN
             ! diagonalize W+ H V = V+ H+H V, the squared Hamiltonian 
             CALL ORTH1('U', &
                  WHAM%CW_RED(1,1),WHAM%CW_RED(1,NPOS_RED), &
                  WHAM%CPROJ_RED(1,1),WS%CPROJ_RED(1,1),NSUBD, & ! note WS%CPROJ_RED(1,1) is not used 
                  NPOS_RED, NSIM_, WDES1%NPL_RED, 0 ,WDES%NRPLWV_RED,WDES%NPROD_RED,CHAM(1,1))
          ELSE
             CALL ORTH1('U', &
                  WOPT%CW_RED(1,1),WHAM%CW_RED(1,NPOS_RED), & 
                  WOPT%CPROJ_RED(1,1),WS%CPROJ_RED(1,1),NSUBD, & ! WS%CPROJ_RED(1,1) is not used here
                  NPOS_RED, NSIM_, WDES1%NPL_RED, 0 ,WDES%NRPLWV_RED,WDES%NPROD_RED,CHAM(1,1))
          ENDIF
          CALLMPI( M_sum_g(WDES%COMM,CHAM(1,NPOS_RED),NSUBD*NSIM_))

          IF(LHARMONIC)THEN
             CALL ORTH3( &
                  WHAM%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED), & 
                  WHAM%CPROJ_RED(1,1),WS1%CPROJ_RED(1,NPOS_RED), &
                  (ITER-1)*NSIM_, NSIM_, NSUBD, NPOS_RED, &
                  WDES1%NPL_RED, WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL_HARM(1,1))
             CALLMPI( M_sum_g(WDES%COMM,COVL_HARM(1,NPOS_RED),NSUBD*NSIM_) )
             COVL_HARM=GCONJG(TRANSPOSE(COVL_HARM))
             CALL ORTH3( &
                  WOPT%CW_RED(1,1), WHAM%CW_RED(1,NPOS_RED), &
                  WS1%CPROJ_RED(1,1), WHAM%CPROJ_RED(1,NPOS_RED), &
                  ITER*NSIM_, NSIM_, NSUBD, NPOS_RED, &
                  WDES1%NPL_RED, WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL_HARM(1,1))
             CALLMPI( M_sum_g(WDES%COMM,COVL_HARM(1,NPOS_RED),NSUBD*NSIM_) )
             COVL_HARM=GCONJG(TRANSPOSE(COVL_HARM))            
          END IF

!gJ: It seems that despite orthogonalization the overlap matrix is not
!exactly unity and this seems to cause instability ? => keep this for now
!!$#ifdef ortho_first
!!$#else
          CALL ORTH1('U', &
               WOPT%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED), & 
               WOPT%CPROJ_RED(1,1),WS%CPROJ_RED(1,1),NSUBD, &
               NPOS_RED, NSIM_, WDES1%NPL_RED, WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL(1,1))
          CALLMPI( M_sum_g(WDES%COMM,COVL(1,NPOS_RED),NSUBD*NSIM_))
          ! add remaining elements to COVL
          DO M=1,NSIM_*(ITER-1)
             DO I=1,NSIM_
                COVL(NPOS_RED-1+I,M)=GCONJG(COVL(M,NPOS_RED-1+I))
             ENDDO
          ENDDO
!!$#endif

          IF(LHARMONIC)THEN
             ! add remaining elements to CHAM
             DO M=1,NSIM_*(ITER-1)
                DO I=1,NSIM_
                   CHAM(NPOS_RED-1+I,M)=GCONJG(CHAM(M,NPOS_RED-1+I))
                ENDDO
             ENDDO
          END IF

          IF (.NOT.LHARMONIC) THEN
            ! correct CHAM by subtraction of epsilon COVL
            DO M=1,NSIM_*ITER
               DO I=1,NSIM_
                  CHAM(M,NPOS_RED-1+I)=CHAM(M,NPOS_RED-1+I)+COVL(M,NPOS_RED-1+I)*EVALUE_IN_S_GLBL(I)
               ENDDO
            ENDDO
          ENDIF
#ifdef debug
          io_begin
          CALL DUMP_HAM_SELECTED( "Hamiltonian", WDES, CHAM, SIZE(CHAM,1), ITER*NSIM_)
          CALL DUMP_HAM_SELECTED( "overlap", WDES, COVL, SIZE(COVL,1), ITER*NSIM_)
          CALL DUMP_HAM_SELECTED( "overlap HARM", WDES, COVL_HARM, SIZE(COVL_HARM,1), ITER*NSIM_)
          io_end
#endif
#ifndef gammareal
          DO N1=1,NSIM_*ITER
             IF (ABS(AIMAG(CHAM(N1,N1)))/MAX(1._q,ABS(REAL(CHAM(N1,N1))))>1E-5_q) THEN
                WRITE(*,*)'WARNING: Sub-Space-Matrix is not hermitian in DAV ',N1, &
                     AIMAG(CHAM(N1,N1))
             ENDIF
             CHAM(N1,N1)= REAL( CHAM(N1,N1) ,KIND=q)
          ENDDO
#endif

          ! solve eigenvalue-problem and calculate lowest eigenvector
          ! CHAM(n1,n2) U(n2,1) = E(1) S(n1,n2)  U(n2,1)
          CEIGVEC = 0
          CEIGVEC (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
          COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)

          IF(LHARMONIC)THEN
             ! solve generalized non-symmetric eigenvalue-problem
             ! alpha(i) CHAM(n1,n2) U(n2,i) = beta(i) COVL_HARM(n1,n2)  U(n2,i)
             NDIM=ITER*NSIM_
             COVL_(1:NDIM,1:NDIM) = COVL_HARM(1:NDIM,1:NDIM)
             
             ! note: LV is not used
#ifdef gammareal
             call dggev('N','V',NDIM,CEIGVEC,SIZE(CEIGVEC,1),COVL_,SIZE(COVL_,1),&
                  ALPHAR,ALPHAI,BETA,LV,1,RV,SIZE(RV,1),WORK1,LWORK1,IFAIL)
             IF (IFAIL/=0) THEN
                WRITE(*,*) 'internal ERROR in EDDAV_NEW: Call to routine DGGEV failed! '// &
                     &              'Error code was ',IFAIL
                STOP
             ENDIF
             ! calculate eigenvalues
             J=0
             DO I=1,NDIM
                ! skip if division by zero
                IF(ABS(BETA(I))<1.E-14_q)CYCLE
                ! stop if imaginary part /= 0
!!$!                      IF (IU6>=0)WRITE(IU6,*) 'alphai(',I,')=',ALPHAI(I)
!!$!                      IF (IU0>=0)WRITE(IU0,*) 'alphai(',I,')=',ALPHAI(I)
!!$!                   STOP
                J=J+1
                R(J)=SIGN(SQRT(ALPHAR(I)**2+ALPHAI(I)**2),ALPHAR(I))/BETA(I)
                CEIGVEC(1:NDIM,J)=RV(1:NDIM,I)
             END DO
             NDIM_=J
#else
             call zggev('N','V',NDIM,CEIGVEC,SIZE(CEIGVEC,1),COVL_,SIZE(COVL_,1), &
                  ALPHA,BETA,LV,1,RV,SIZE(RV,1),WORK1,LWORK1,RWORK1,IFAIL)
             ! calculate eigenvalues
             DO I=1,NDIM
                ! skip if division by zero
                IF(ABS(BETA(I))<1.E-14_q)CYCLE
                ! stop if imaginary part /= 0
                IF(ABS(AIMAG(ALPHA(I)/BETA(I)))>1.E-6_q)THEN
                   IF (IU6>=0)WRITE(IU6,*) 'complex eigenvalue(',I,')=',ALPHA(I)/BETA(I)
                   IF (IU0>=0)WRITE(IU0,*) 'complex eigenvalue(',I,')=',ALPHA(I)/BETA(I)
                   STOP
                END IF
                R(I)=ALPHA(I)/BETA(I)
             END DO
#endif

             ! sort eigenvalues by increasing absolute value
             ! R = (eps - eps_ref)
             ! the eigenvalue with largest absolute value is chosen as solution
             RV(1:NDIM,1:NDIM_)=CEIGVEC(1:NDIM,1:NDIM_)
             RR(1:NDIM_)=R(1:NDIM_)
             R(1:NDIM_)=ABS(RR(1:NDIM_))
             KEY(1:NDIM_)=(/ (I,I=1,NDIM_) /)
#ifdef scaLAPACK
             call dlasrt2('I',NDIM_,R(1),KEY(1),IFAIL)
#endif
             CEIGVEC=0
             DO I=1,NDIM_
                CEIGVEC(1:NDIM,I)=RV(1:NDIM,KEY(I))
                R(I)=RR(KEY(I))
             END DO
             R(1:NSIM_)=R(1:NSIM_)+EVALUE_IN_S_GLBL(1:NSIM_)
             RR=R
#ifdef debug
          io_begin
          CALL DUMP_HAM_SELECTED( "Eigenvectors", WDES, CEIGVEC, SIZE(CEIGVEC,1), ITER*NSIM_)
          io_end
#endif
          ELSE

!!$#ifdef ortho_first
!!$#ifdef gammareal
!!$#else
!!$#endif
!!$#else
#ifdef gammareal
             CALL DSYGV &
                  (1,'V','U',ITER*NSIM_,CEIGVEC,NSUBD,COVL_,NSUBD,R, &
                  CWRK(1),LWORK*NB_TOT,IFAIL)
#else
             CALL ZHEGV &
                  (1,'V','U',ITER*NSIM_,CEIGVEC,NSUBD,COVL_,NSUBD,R, &
                  CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
!!$#endif
          END IF

#ifdef debug
          io_begin
          WRITE(*,'(8F12.5)') R(1:ITER*NSIM_)
          CALL DUMP_HAM_SELECTED( "Eigenvectors", WDES, CEIGVEC, SIZE(CEIGVEC,1), ITER*NSIM_)
          io_end
#endif
          CALLMPI( M_bcast_g(WDES%COMM, CEIGVEC, NSUBD*ITER*NSIM_))


       IF (IFAIL/=0) THEN
          IF (ITER>=3) THEN
             ! if iteration failed go back one step
             ! and diagonalize again
             ITER=ITER-1
             CEIGVEC (1:ITER*NSIM_,1:ITER*NSIM_) = CHAM(1:ITER*NSIM_,1:ITER*NSIM_)
             COVL_(1:ITER*NSIM_,1:ITER*NSIM_) = COVL(1:ITER*NSIM_,1:ITER*NSIM_)
!!$#ifdef ortho_first
!!$#ifdef gammareal
!!$#else
!!$#endif
!!$#else
#ifdef gammareal
             CALL DSYGV &
                  (1,'V','U',(ITER)*NSIM_,CEIGVEC,NSUBD,COVL_,NSUBD,R, &
                  CWRK(1),LWORK*NB_TOT,IFAIL)
#else
             CALL ZHEGV &
                  (1,'V','U',(ITER)*NSIM_,CEIGVEC,NSUBD,COVL_,NSUBD,R, &
                  CWRK(1),LWORK*NB_TOT,RWORK,IFAIL)
#endif
!!$#endif

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


#ifdef debug
       io_begin
       CALL DUMP_HAM_SELECTED( "reconstructed Hamiltonian", WDES, CEIGVEC, SIZE(CEIGVEC,1), ITER*NSIM_)
       io_end
#endif
!-----------------------------------------------------------------------
! update energies and calculate total energy change
!-----------------------------------------------------------------------
       II=0
       DEMAX=0
       DO NP=1,NSIM_LOCAL_
          N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
          DO NPOS_RED=(N-1)*NCPU+1,N*NCPU
             II=II+1
             
             ! EVALUE_GLBL corresponds to stored < phi| H | phi>
             ! R contains new updated eigenvalue
             DECEL =R(II)-EVALUE_GLBL(II)    ! change in eigenenergy
             EVALUE_GLBL(II)=R(II)           ! update 
             ! determine maximum change
             DEMAX=MAX(DEMAX, ABS(DECEL))
             IF (IDUMP==2)  WRITE(*,'(E10.2,2H |)',ADVANCE='NO') DECEL
             
             ! change of eigenvalues is from squared problem 
             ! not useful if LFOLD is true
!             IF (.NOT. LHARMONIC) THEN 
!gK: can we use this even in harmonic case ??
                W%CELTOT(NPOS_RED,NK,ISP)=R(II) ! update CELTOT array
!                no weighting with weights
!                   DESUM =DESUM +WDES%RSPIN*WDES%WTKPT(NK)*FERTOT(NPOS_RED,NK,ISP)*DECEL
                DESUM =DESUM +WDES%RSPIN*WDES%WTKPT(NK)*DECEL
!             ENDIF
             
          ENDDO
       ENDDO

!-----------------------------------------------------------------------
! possibly break the optimisation
! and new eigenenergy
!-----------------------------------------------------------------------
          ITER=IT(1)
          ! break if absolute change in eigenenergy is small
          ! -------------------------------------------------
          MAXR=MAXVAL(SQRT(ABS(RESIDUAL(1:NSIM_))))
          IF(MAXR<INFO%DEPER)LSTOP=.TRUE.
          IF (ITER>1 .AND. DEMAX < INFO%EBREAK) LSTOP=.TRUE.
          ! relative break criterion 
          ! -------------------------------------------------
          ! for inner eigenvalue problems this is not really
          ! particularly relevant
          ! IF (ITER==2) DEIT=DEMAX*INFO%DEPER
          ! IF (ITER >2 .AND. DEMAX < DEIT) LSTOP=.TRUE.

          NITER_NOW=NITER
          ! sufficient iterations done
          ! -------------------------------------------------
          IF (ITER >= NITER_NOW) LSTOP=.TRUE.
          IF (ITER >= NITER)     LSTOP=.TRUE.  ! certainly stop if storage requires this

          !-----------------------------------------------------------------
          ! If LHARMONIC: before we write back the eigenvectors, they should
          ! be orthogonalized wrt. COVL
          ! also before reboot orthogonalization is necessary!
          ! (since the eigenvectors of the generalized eigenvalue problem 
          ! are orthogonal wrt. COVL_HARM !)
          IF(LHARMONIC)THEN
             ! Cholesky factorization of overlap matrix between eigenvectors
             NDIM=0
             NDIM=MAX(NDIM,1)

             COVL_=0
             COVL_(1:ITER*NSIM_,1:NDIM*NSIM_)=MATMUL( COVL(1:ITER*NSIM_,1:ITER*NSIM_),CEIGVEC(1:ITER*NSIM_,1:NDIM*NSIM_) )
             COVL_(1:NDIM*NSIM_,1:NDIM*NSIM_)=MATMUL( GCONJG(TRANSPOSE(CEIGVEC(1:ITER*NSIM_,1:NDIM*NSIM_))),COVL_(1:ITER*NSIM_,1:NDIM*NSIM_) )
#ifdef gammareal
             CALL DPOTRF &
#else
             CALL ZPOTRF &
#endif
                ('U',NDIM*NSIM_, COVL_(1,1), SIZE(COVL_,1),IFAIL)
             DO I=2,NDIM*NSIM_
                DO M=1,I-1
                   COVL_(I,M)=0
                END DO
             END DO
             IF (IFAIL/=0) THEN
                WRITE(*,*) 'EDDAV LAPACK: Routine ZPOTRF failed!',IFAIL
                STOP
             ENDIF
#ifdef gammareal
             CALL DTRTRI &
#else
             CALL ZTRTRI &
#endif
                ('U','N',NDIM*NSIM_,COVL_(1,1), SIZE(COVL_,1),IFAIL)
             IF (IFAIL/=0) THEN
                WRITE(*,*) 'EDDAV LAPACK: Routine ZTRTRI failed!',IFAIL
                STOP
             ENDIF
             ! the orthogonalized vectors are given by
             ! d_jx = U_kj^(-1) c_kx = U_kj^(-1) CEIGVEC(x,k) = CEIGVEC(x,k)U_kj^(-1)
             CEIGVEC(1:ITER*NSIM_,1:NDIM*NSIM_)=MATMUL(CEIGVEC(1:ITER*NSIM_,1:NDIM*NSIM_),COVL_(1:NDIM*NSIM_,1:NDIM*NSIM_))

          END IF

          !=======================================================================
          ! if stopping is selected store the optimised orbitals back
          !=======================================================================
          stop: IF (LSTOP) THEN
             IF (IDUMP==2)  WRITE(*,*)

             NPOS_RED=(W1(1)%NB-1)*NCPU

             IF (WDES1%NPL_RED/=0) &
                  CALL GGEMM('N', 'N', m_ WDES1%NPL_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CW_RED(1,1), m_ WDES%NRPLWV_RED, CEIGVEC(1,1), NSUBD,  &
                  &               zero, WA%CW_RED(1,NPOS_RED+1), m_ WDES%NRPLWV_RED)

             IF (WDES1%NPRO_RED/=0) &
                  CALL GGEMM('N', 'N', WDES1%NPRO_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CPROJ_RED(1,1), WDES%NPROD_RED, CEIGVEC(1,1), NSUBD,  &
                  &               zero, WA%CPROJ_RED(1,NPOS_RED+1), WDES%NPROD_RED)

             ! calculate the best vector of the previous iteration
             ! stores back the NSIM_ lowest eigenvectors; NSIM_*ITER is the dimension of the current optimization subspace
             IF (WDES1%NPL_RED/=0) &
                  CALL GGEMM('N', 'N', m_ WDES1%NPL_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CW_RED(1,1), m_ WDES%NRPLWV_RED, CEIGVEC_LAST(1,1), NSUBD,  &
                  &               zero, WLAST%CW_RED(1,NPOS_RED+1), m_ WDES%NRPLWV_RED)


             IF (WDES1%NPRO_RED/=0) &
                  CALL GGEMM('N', 'N', WDES1%NPRO_RED , NSIM_, NSIM_*ITER, one, &
                  &               WOPT%CPROJ_RED(1,1), WDES%NPROD_RED, CEIGVEC_LAST(1,1), NSUBD,  &
                  &               zero, WLAST%CPROJ_RED(1,NPOS_RED+1), WDES%NPROD_RED)


             ! idential F90 version
             !      DO II=1,NSIM_
             !        CW_RED(1:NPL,NPOS_RED+II)     = 0
             !        WA%CPROJ_RED(1:NPRO,NPOS_RED+II) = 0
             !        DO I=1,NSIM_*ITER
             !           DO M=1,NPL
             !              CW_RED   (M,NPOS_RED+II)=CW_RED   (M,NPOS_RED+II)+CEIGVEC(I,II)*WOPT%CW_RED(M,I)
             !           ENDDO
             !           DO M=1,NPRO
             !              WA%CPROJ_RED(M,NPOS_RED+II)=WA%CPROJ_RED(M,NPOS_RED+II)+CEIGVEC(I,II)*WOPT%CPROJ_RED(M,I)
             !           ENDDO
             !        ENDDO
             !     ENDDO

             W1%NB=0
             CYCLE bands
          ENDIF stop

!***********************************************************************
!
! next step increase iteration count
!
!***********************************************************************
          ICOUEV=ICOUEV+NSIM_
! now store best eigenvector in CEIGVEC_LAST
          CEIGVEC_LAST(:,:)=CEIGVEC(:,:)

#define ortho_before_precond
#ifdef ortho_before_precond
!-----------------------------------------------------------------------
! first store the new W1 orbitals to WOPT
! and redistribute over plane wave coefficients
!-----------------------------------------------------------------------
          DO NP=1,NSIM_LOCAL_
             N=W1(NP)%NB; ITER=IT(NP); IF (.NOT. W1(NP)%LDO) CYCLE
             CALL CNORMN(W1(NP), CQIJ, ISP, WSCAL)
             NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW

             CALL W1_COPY(W1(NP), ELEMENT(WOPT, NPP))
             
             IF (WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, WOPT%CW(1,NPP))
                CALL REDIS_PROJ(WDES1, 1, WOPT%CPROJ(1,NPP))
             ENDIF
          ENDDO
!-----------------------------------------------------------------------
! overlap and orthogonalisation to all other bands 
!
! 1- \sum_i S |phi_i> <phi_i|
!
! or 32 other bands for folded (squared) problem
!-----------------------------------------------------------------------
             ! This orthogonalizes the NSIM_ vectors currently added to the search space WOPT
             ! to 32 or all bands. It starts in ITER==1 with the 2nd slot in WOPT. The 1st slot is 
             ! skipped because it contains the optimized bands themselves, and they must not be 
             ! orthogonalized to themselves.
             !
             LOWER=0
             UPPER=NB_TOT+1
             
             IF (UPPER-LOWER>1) THEN
                
                NPOS_RED=ITER*NSIM_+1   ! storage position in WOPT%CW_RED, WHAM%CW_RED
                
                CORTHO=0
                CALL ORTH1('L', &
                     WA%CW_RED(1,LOWER+1),WOPT%CW_RED(1,NPOS_RED), & 
                     WA%CPROJ_RED(1,LOWER+1), WS%CPROJ_RED(1,1),UPPER-LOWER-1, &
                     1, NSIM_, WDES1%NPL_RED, 0 ,WDES%NRPLWV_RED,WDES%NPROD_RED,CORTHO(1,1))
                
                CALLMPI( M_sum_g(WDES%COMM, CORTHO(1,1), (UPPER-LOWER-1)*NSIM_))
                
                
                IF (WDES1%NPL_RED /=0 ) &
                     CALL GGEMM( 'N', 'N' , m_ WDES1%NPL_RED , NSIM_ , UPPER-LOWER-1 , -one , &
                     WA%CW_RED(1,LOWER+1), m_ WDES%NRPLWV_RED , CORTHO(1,1) , UPPER-LOWER-1 , &
                     one , WOPT%CW_RED(1,NPOS_RED) , m_ WDES%NRPLWV_RED ) ! WOPT%CPROJ(1,ITER*NSIM_LOCAL_+1)

!gK: no need to calculate WOPT%CPROJ_RED before hand, since we overwrite it here                
                WOPT%CPROJ_RED(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
                IF (WDES1%NPRO_RED /= 0) &
                     CALL GGEMM( 'N', 'N' ,  WDES1%NPRO_RED , NSIM_ , UPPER-LOWER-1  , -one , &
                     WA%CPROJ_RED(1,LOWER+1) ,  WDES%NPROD_RED , CORTHO(1,1) , UPPER-LOWER-1 , &
                     one , WOPT%CPROJ_RED(1,NPOS_RED) ,  WDES%NPROD_RED  )
             ENDIF

!-----------------------------------------------------------------------
! subtract 
! \sum_kl |beta_k> Q_kl <beta_l| \sum_i phi_i><phi_i| R_j> 
! in  WOPT%CPROJ_RED  - \sum_i <beta_l| phi_i><phi_i| R_j>  is stored
!-----------------------------------------------------------------------
          DO NP=1,NSIM_LOCAL_
             IF (.NOT. W1(NP)%LDO) CYCLE
             
             NPP=ITER*NSIM_LOCAL_+NP ! storage position in WOPT%CW, WHAM%CW
             ! redistribute data over bands
             CALL W1_COPY( ELEMENT(WOPT, NPP), W1(NP))
             IF( WDES%DO_REDIS) THEN
                CALL REDIS_PW(WDES1, 1, W1(NP)%CW)
                CALL REDIS_PROJ(WDES1, 1,  W1(NP)%CPROJ)
             ENDIF
             ! multiply each band with Q_kl and store results in 
             ! W1%CPROJ(j,k) =  - Q_kl <beta_l| \sum_i phi_i><phi_i| R_j> 
             IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                WDES1%NBANDS=1
                CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
             ENDIF
             ! subtract  |beta_k> W1%CPROJ(j,k)
             IF (INFO%LREAL) THEN
                W1(NP)%CR(:)=0
                CALL RACC0(NONLR_S,WDES1, WS%CPROJ(1,NP) ,W1(NP)%CR(1))
                DO ISPINOR=0,WDES1%NRSPINORS-1
                   CALL FFTEXT(WDES1%NGVECTOR,WDES1%NINDPW, W1(NP)%CR(1+ISPINOR*WDES1%GRID%MPLWV), W1(NP)%CW(1+ISPINOR*WDES1%NGVECTOR),GRID,.TRUE.)
                ENDDO
             ELSE
                CALL VNLAC0(NONL_S,WDES1, WS%CPROJ(1,NP), W1(NP)%CW(1+ISPINOR*WDES1%NGVECTOR))
             ENDIF
          ENDDO
#endif
!-----------------------------------------------------------------------
! preconditioning of calculated orthogonalized residual vector
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
!gK: scheduled for removal (not really required)
             CALL CNORMN(W1(NP), CQIJ, ISP, WSCAL)

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
! overlap and orthogonalisation to all other bands (again)
!-----------------------------------------------------------------------
             ! This orthogonalizes the NSIM_ vectors currently added to the search space WOPT
             ! to all bands. It starts in ITER==1 with the 2nd slot in WOPT. The 1st slot is 
             ! skipped because it contains the optimized bands themselves, and they must not be 
             ! orthogonalized to themselves.
             !
             LOWER=0
             UPPER=NB_TOT+1
             
             IF (UPPER-LOWER>1) THEN
                
                NPOS_RED=ITER*NSIM_+1   ! storage position in WOPT%CW_RED, WHAM%CW_RED
                
                CORTHO=0
                CALL ORTH1('L', &
                     WA%CW_RED(1,LOWER+1),WOPT%CW_RED(1,NPOS_RED), & 
                     WA%CPROJ_RED(1,LOWER+1), WS%CPROJ_RED(1,1),UPPER-LOWER-1, &
                     1, NSIM_, WDES1%NPL_RED, WDES1%NPRO_O_RED ,WDES%NRPLWV_RED,WDES%NPROD_RED,CORTHO(1,1))
                
                CALLMPI( M_sum_g(WDES%COMM, CORTHO(1,1), (UPPER-LOWER-1)*NSIM_))
                
                IF (WDES1%NPL_RED /=0 ) &
                     CALL GGEMM( 'N', 'N' , m_ WDES1%NPL_RED , NSIM_ , UPPER-LOWER-1 , -one , &
                     WA%CW_RED(1,LOWER+1), m_ WDES%NRPLWV_RED , CORTHO(1,1) , UPPER-LOWER-1 , &
                     one , WOPT%CW_RED(1,NPOS_RED) , m_ WDES%NRPLWV_RED )
                
                IF (WDES1%NPRO_RED /= 0) &
                     CALL GGEMM( 'N', 'N' ,  WDES1%NPRO_RED , NSIM_ , UPPER-LOWER-1  , -one , &
                     WA%CPROJ_RED(1,LOWER+1) ,  WDES%NPROD_RED , CORTHO(1,1) , UPPER-LOWER-1 , &
                     one , WOPT%CPROJ_RED(1,NPOS_RED) ,  WDES%NPROD_RED  )
                
             ENDIF
!-----------------------------------------------------------------------
! orthogonalization to subset of bands in the iterated set
! (required to improve stability)
!-----------------------------------------------------------------------
#ifdef ortho_first
          IF (ITER /= NITER) THEN
             ! recalculate CQIJ x projected character for the updated bands
             IF (INFO%LOVERL .AND. WDES%NPROD>0 ) THEN
                DO NP=1,NSIM_LOCAL_
                   IF (.NOT. W1(NP)%LDO) CYCLE
                   
                   NPP=ITER*NSIM_LOCAL_+NP   ! storage position in WOPT%CW, WHAM%CW
                   CALL W1_COPY( ELEMENT( WOPT, NPP), W1(NP))
                   
                   IF( WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, 1,  W1(NP)%CPROJ(1))
                   WDES1%NBANDS=1
                   CALL OVERL(WDES1, INFO%LOVERL,LMDIM,CQIJ(1,1,1,1), W1(NP)%CPROJ(1),WS%CPROJ(1,NP))
                   IF( WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, 1,  WS%CPROJ(1,NP))
                   
                ENDDO
             ENDIF

             NPOS_RED=ITER*NSIM_+1   ! storage position in redistributed WOPT%CW, WHAM%CW
             
             ! this calculates the in-product with the vectors stored in the iterative subspace
             ! (and with itself)
             ! store results in COVL(:,NPOS_RED:NPOS_RED+NSIM_-1)
             
             COVL(:,NPOS_RED:NPOS_RED+NSIM_-1)=0
#define Sortho
#ifdef Sortho
!gK: ok little extra cost
!     numerically more stable in DGGEV (I saw some problems)
! also I believe the subtraction of COVL below is only correct
! if S is taken into account or have I overlooked something ??
! I think you get an odd mixture of with S and without
             CALL ORTH1('U', &
                  WOPT%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED), & 
                  WOPT%CPROJ_RED(1,1),WS%CPROJ_RED(1,1),NSUBD, &
                  NPOS_RED, NSIM_, WDES1%NPL_RED, WDES1%NPRO_O_RED,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL(1,1))
#else
             ! do not use WS%CPROJ !
             CALL ORTH1('U', &
                  WOPT%CW_RED(1,1),WOPT%CW_RED(1,NPOS_RED), & 
                  WOPT%CPROJ_RED(1,1),WS%CPROJ_RED(1,1),NSUBD, &
                  NPOS_RED, NSIM_, WDES1%NPL_RED, 0,WDES%NRPLWV_RED,WDES%NPROD_RED,COVL(1,1))
#endif
             CALLMPI( M_sum_g(WDES%COMM,COVL(1,NPOS_RED),NSUBD*NSIM_))

#ifdef debug
             io_begin
             CALL DUMP_HAM_SELECTED( "overlap1", WDES, COVL(1,1), SIZE(COVL,1), (ITER+1)*NSIM_)
             io_end
#endif
             
             IF (WDES1%NPL_RED /=0 ) &
                  CALL GGEMM( 'N', 'N' , m_ WDES1%NPL_RED , NSIM_ , ITER*NSIM_ , -one , &
                  WOPT%CW_RED(1,1), m_ WDES%NRPLWV_RED , COVL(1,NPOS_RED) , SIZE(COVL,1) , &
                  one , WOPT%CW_RED(1,NPOS_RED) , m_ WDES%NRPLWV_RED )
             
             IF (WDES1%NPRO_RED /= 0) &
                  CALL GGEMM( 'N', 'N' ,  WDES1%NPRO_RED , NSIM_ , ITER*NSIM_  , -one , &
                  WOPT%CPROJ_RED(1,1) ,  WDES%NPROD_RED , COVL(1,NPOS_RED) , SIZE(COVL,1) , &
                  one , WOPT%CPROJ_RED(1,NPOS_RED) ,  WDES%NPROD_RED  )
             
             ! correct COVL for subtraction of other vectors
             DO I=1,NSIM_
                DO II=1,NSIM_
                   DO M=1,ITER*NSIM_
                      COVL(NPOS_RED-1+I,NPOS_RED-1+II)=COVL(NPOS_RED-1+I,NPOS_RED-1+II)-GCONJG(COVL(M,NPOS_RED-1+I))*COVL(M,NPOS_RED-1+II)
                   ENDDO
                ENDDO
             ENDDO
             ! set COVL in lower part to zero
             COVL(1:NPOS_RED-1,NPOS_RED:NPOS_RED+NSIM_-1)=0

#ifdef debug
             io_begin
             CALL DUMP_HAM_SELECTED( "overlap updated", WDES, COVL(1,1), SIZE(COVL,1), (ITER+1)*NSIM_)
             io_end
#endif

             ! LU factorization of remaining overlap matrix between new added vectors
             ! stored in COVL(NPOS_RED:,NPOS_RED:)
#ifdef gammareal
             CALL DPOTRF &
#else
             CALL ZPOTRF &
#endif
                ('U',NSIM_, COVL(NPOS_RED,NPOS_RED), SIZE(COVL,1),IFAIL)
             IF (IFAIL/=0) THEN
                WRITE(*,*) 'EDDAV LAPACK: Routine ZPOTRF failed!',IFAIL,NPOS_RED
                STOP
             ENDIF

#ifdef gammareal
             CALL DTRTRI &
#else
             CALL ZTRTRI &
#endif
                  ('U','N',NSIM_,COVL(NPOS_RED,NPOS_RED), SIZE(COVL,1),IFAIL)
             IF (IFAIL/=0) THEN
                WRITE(*,*) 'EDDAV LAPACK: Routine ZTRTRI failed!',IFAIL,NPOS_RED
                STOP
             ENDIF

#ifdef debug
             io_begin
             CALL DUMP_HAM_SELECTED( "transform", WDES, COVL(1,1), SIZE(COVL,1), (ITER+1)*NSIM_)
             io_end
#endif
             ! transform added orbitals among each other to make them orthonormal
             CALL LINCOM('U',  WOPT%CW_RED(1:,NPOS_RED:), WOPT%CPROJ_RED(1:,NPOS_RED:), COVL(NPOS_RED,NPOS_RED), &
                  NSIM_,NSIM_, WDES1%NPL_RED, WDES1%NPRO_RED, WDES1%NRPLWV_RED, WDES1%NPROD_RED, SIZE(COVL,1), &
                  WOPT%CW_RED(1:,1:), WOPT%CPROJ_RED(1:,NPOS_RED:))
             
          ENDIF
#endif
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
!             WRITE(*,'(8F10.4)') W1(NP)%CPROJ(1:4)
!             W1(NP)%CW=W1(NP)%CW*0.1
             CALL FFTWAV_W1(W1(NP))
          ENDDO

!=======================================================================
! move on to the next bands
!=======================================================================
       ENDDO bands
       IF (WDES%DO_REDIS) THEN
          CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
          CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
       ENDIF
    END DO kpoints
    ENDDO spin

    ! RMS was only calculate for the band treated locally (sum over all nodes)
    CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))

!gK: can you check that everything is deallocated 
    DEALLOCATE(PRECON, CHAM, COVL, CEIGVEC, COVL_, COVL_HARM, CORTHO, R, RWORK, CWRK)
    DEALLOCATE(RR,ALPHAR,ALPHAI,ALPHA,BETA,RV,LV,RWORK1,WORK1,KEY,RESIDUAL)
    DO I=1,NSIM_LOCAL
       CALL DELWAV(W1(I), .TRUE.)
    ENDDO

    CALL DELWAVA(WOPT)
    CALL DELWAVA(WHAM)
    CALL DELWAVA(WLAST)
    CALL DELWAVA(WHELP)
    CALL DELWAVA_PROJ(WS)
    CALL DELWAVA_PROJ(WS1)

    RETURN
  END SUBROUTINE EDDAV_INNER


!************************ SUBROUTINE  **************************
!
! analyze the overlap matrix
! output: CHAM is overwritten by the eigenvectors
!
!***********************************************************************

  SUBROUTINE PROJOVL(CHAM, COVL, W, IU6 )
    USE prec
    IMPLICIT NONE
    !
    GDEF :: CHAM(:,:),COVL(:,:)
    REAL(q) :: W(:)
    INTEGER :: IU6
    ! local
    REAL(q), PARAMETER :: THR=1.0E-14_q
    INTEGER :: NDIM,IFAIL,LWORK,SDIM,J(1),I,K
    GDEF, ALLOCATABLE ::  LHAM(:,:),LOVL(:,:),WORK(:),TMAT(:,:)
    REAL(q), ALLOCATABLE :: RWORK(:),R(:)

    NDIM = SIZE(CHAM,1)
    LWORK=5*NDIM
    ALLOCATE(LHAM(NDIM,NDIM),LOVL(NDIM,NDIM), &
         WORK(LWORK),RWORK(LWORK),R(NDIM))
    LOVL=COVL
!=======================================================================
! diagononalize the overlap matrix
!=======================================================================
#ifdef  gammareal
    CALL DSYEV &
         ('V','U',NDIM,LOVL,NDIM,R,RWORK,LWORK,IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in PROJOVL: Call to routine DSYEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
#else
    CALL ZHEEV &
         ('V','U',NDIM,LOVL,NDIM,R,WORK,LWORK,RWORK,IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in PROJOVL: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
#endif

!    IF(IU6>=0)WRITE(IU6,'("S=",500E15.4)') R

    J=MINLOC(R,ABS(R)>THR*R(NDIM))
    ! SDIM = effective rank of COVL
    SDIM=NDIM-J(1)+1
    ALLOCATE(TMAT(NDIM,SDIM))
    TMAT=LOVL(:,J(1):NDIM)

    ! diagonal overlap
    LOVL=0
    DO I=1,SDIM
       LOVL(I,I)=R(J(1)+I-1)
    END DO

    ! make hermitian
    DO I=1,NDIM
       DO K=1,I-1
          CHAM(I,K)=GCONJG(CHAM(K,I))
       ENDDO
    ENDDO
    ! project the eigensystem on the non-null subspace
    ! H v = lambda S v
    ! H v = lambda TDT+ v
    ! (T+ H T) (T+ v) = lambda D (T+ v)
    LHAM(:,1:SDIM)=MATMUL(CHAM,TMAT)
    LHAM(1:SDIM,1:SDIM)=MATMUL(TRANSPOSE(GCONJG(TMAT)),LHAM)
    ! solve the projected generalized eigensystem
#ifdef gammareal
    CALL DSYGV &
         (1,'V','U',SDIM,LHAM,NDIM,LOVL,NDIM,W, &
         WORK,LWORK,IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in PROJOVL: Call to routine DSYGV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
#else
    CALL ZHEGV &
         (1,'V','U',SDIM,LHAM,NDIM,LOVL,NDIM,W, &
         WORK,LWORK,RWORK,IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in PROJOVL: Call to routine ZHEGV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
#endif

    ! transform back eigenvectors
    CHAM(1:NDIM,1:SDIM)=MATMUL(TMAT,LHAM(1:SDIM,1:SDIM))
!    WRITE(*,'("CHAM=",20E15.4)') CHAM(1:NDIM,1)
!    WRITE(*,'("W=",20E15.4)') W(1:SDIM)
    

!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
!!$
!!$

    DEALLOCATE(TMAT,LHAM,LOVL,WORK,RWORK,R)

  END SUBROUTINE PROJOVL


END MODULE david_inner
