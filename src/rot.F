#include "symbol.inc"
      MODULE rot
      USE prec
      CONTAINS
!************************ SUBROUTINE EDWAV *****************************
! RCS:  $Id: rot.F,v 1.2 2001/02/20 14:45:00 kresse Exp $
!
! subroutine for optimizing all bands simultaneous
! the subroutine calculates a search direction into which the
! total energy is minimized, than a trial step into this direction
! is done.
! On the next call the new exact energy must be supplied by the calling
! routine, EDWAV then tries to determine an optimum step length
! and steps into the minimum.
! IFLAG determines the status of the routine:
!  0   trial - steepest descent step
!  1   trial - conjugate gradient step (IALGO can overwrite this mode to 0)
!  2   go to the  to minimum
!      TOTEN must contain the old energy TOTEN2 the energy after the step
!  10  increase trial step (mainly for checking accuracy of routine)
! on exit
! >0   main routine performs another energy-evaluation
!  0   main routine ends the loop
!
! IALGO determines the specific algorithm
!  3   damped MD with damping term automatically determined
!      by the given timestep
!  4   damped MD (velocity quench or quickmin)
!  5   steepest descent without optimizing the stepsize
!  7   steepest descent with optimization of stepsize
!  8   conjugated gradient
!  9   conjugate gradient with gradient information only
!
! IROT determines if subspace rotation and/or wavefunctions are optimized
!  1   only orbital coefficients are optimized
!  2   only rotations in the set of calculated orbitals (sub space rotation)
!  3   orbital coefficient optimization and subspace rotation
!  4   orbital coefficient optimization and subspace rotation
!      the Hamilton matrix in the subspace is determined by a self-consistent cycle 
!      (subrot_scf)
!  5   orbitals optimized, exact subspace rotation in the subspace spanned by 
!      the calculated orbitals. 
!      The rotation matrix is determined by scf iterations in the subset of 
!      calculated orbitals  (often unstable)
!  6   optimize pseudo "Hamiltonian" matrix eta  instead of subspace rotation
!  7   optimize pseudo "Hamiltonian" matrix eta  instead of subspace rotation
!      the Hamilton matrix in the subspace is determined by a self-consistent cycle 
!
! ICEL determines how partial occupancies are optimized
!  0   no optimization
!  1   standard optimization (optimize a new set of support variables
!       that are stored in W_F%CELTOT)
!  2   use pseudo Hamiltonian matrix eta to optimize eigenvalues
!      in this case W_F%CELTOT is returned and occupancies need to be 
!      calculated from W_F%CELTOT (similar to ICEL=1)
! IRET determines which algorithm has been used
!  0   steepest descent
!  1   conjugate gradient
!  3   damped
!  4   quick-min
!  
!
! DESUM1 expected energy change energy change
!
! gK: 26.02.2013 bug fix for NC potentials (in two places)
!
!***********************************************************************

 SUBROUTINE EDWAV(HAMILTONIAN,KINEDEN,&
      INFO,IFLAG,IROT,ICEL,IRET, IO,BTRIAL, EFERMI, &
      ORT, RMS, E, TOTEN, TOTEN2, DESUM1, &
      GRID, KPOINTS, LATT_CUR, NONLR_S, NONL_S, T_INFO, P, W,WDES,W_F,W_G, &
      LMDIM, NSIM, CQIJ,CDIJ,CHAM, CHF, SV, &
      SYMM,GRID_SOFT,GRIDC,GRIDB,GRIDUS, &
      C_TO_US,B_TO_C,SOFT_TO_C,DENCOR,CSTRF, &
      MIX,N_MIX_PAW,IRDMAX,CHTOT,RHOLM)
   USE ini
   USE prec
   USE wave
   USE wave_mpi
   USE wave_high
   USE lattice
   USE mpimy
   USE mgrid
      
   USE nonl_high
   USE poscar
   USE pseudo
   USE hamil_high
   USE constant
   USE mkpoints
   USE choleski
   USE fock
   USE hamil
   USE base
   USE lattice
   USE sym_grad
   USE subrotscf
   USE pead
   USE meta
   IMPLICIT COMPLEX(q) (C)
   IMPLICIT REAL(q) (A-B,D-H,O-Z)
      
   TYPE (ham_handle)  HAMILTONIAN
   TYPE (tau_handle)  KINEDEN
   TYPE (grid_3d)     GRID
   TYPE (latt)        LATT_CUR
   TYPE (type_info)   T_INFO
   TYPE (potcar)      P(T_INFO%NTYP)
   TYPE (nonlr_struct)NONLR_S
   TYPE (nonl_struct) NONL_S
   TYPE (wavespin)    W     ! current wavefunctions
   TYPE (wavespin)    W_F   ! last search direction
   TYPE (wavespin)    W_G   ! current gradient
   ! W_F%CELEN is special, and stores variational eigenvalues
   ! from which partial occupancies are set
   TYPE (wavefun1)    W1(NSIM),WSEARCH
   TYPE (wavedes)     WDES
   TYPE (wavedes1)    WDES1
   TYPE (kpoints_struct) KPOINTS
   TYPE (info_struct) INFO
   TYPE (in_struct)   IO
   TYPE (mixing)      MIX
   TYPE (symmetry)    SYMM      
   TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
   TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
   TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
   TYPE (grid_3d)     GRIDB      ! Broyden grid
   TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
   TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
   TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC      
   TYPE (energy)      E
   
   RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
   COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor
   COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
   REAL(q)     RHOLM(N_MIX_PAW,WDES%NCDIJ)   !  paw sphere charge density
   
   LOGICAL LREAL,LOVERL
   LOGICAL LCHCON                  ! update of charge or not
   REAL(q) EFERMI                  ! Fermi-level
   REAL(q) WEIMIN                  ! minimum occupancy for empty state
   INTEGER NSIM                    ! simultaneously optimised bands
   INTEGER LMDIM                   ! dimension of arrays CQIJ and CDIJ

   RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
   OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
   !-----Arrays for gradient
   GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)  ! sub space Hamiltionian in space spanned by orbitals
   GDEF CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)   ! search direction in the space spanned by orbitals
   GDEF CROT, CB_
   INTEGER,PARAMETER :: NSTEP=10
   REAL(q) :: STEP(NSTEP),ENRGY(NSTEP)
! work array
   COMPLEX(q) CGRAD(WDES%NRPLWV)
   GDEF,ALLOCATABLE,TARGET :: CPROW(:,:)
   COMPLEX(q),ALLOCATABLE,TARGET:: CF(:,:)    ! stores the plane wave coefficients of the gradient vector
   GDEF,ALLOCATABLE,TARGET::   CPROF(:,:)     ! stores the projected coefficients corresponding to CF
   COMPLEX(q),ALLOCATABLE,TARGET:: CFOLD(:,:) ! stores the plane wave coefficients of the previous search
   GDEF,ALLOCATABLE :: CHAMP(:,:,:,:)         ! stores the preconditioned gradient
   TYPE (wavefuna)    WA             ! pointer to W%CW(:,:;NK,ISP)
   TYPE (wavefuna)    WA_F           ! pointer to W_F%CW(:,:;NK,ISP)
   COMPLEX(q), POINTER :: CF_RED(:,:)
   GDEF   , POINTER ::    CPROW_RED(:,:),CPROF_RED(:,:)
   GDEF, ALLOCATABLE :: COVL(:,:), CUNI(:,:), CEIDB(:,:)

   ! ILOEW=1 uses an approximate rotation matrix R based on Loewdin perturbation theory
   ! ILOEW=2 uses  e^iR to build a rotation matrix for a finite step
   ! usually there is not much difference between both calles
   INTEGER,PARAMETER :: ILOEW=2
   
   ! this parameter determines whether line searches are performed
   ! with high precision using the gradient along the search direction
   LOGICAL,PARAMETER :: ITERATE_PREC=.FALSE.
   
   ! the search direction can be renormalized 
   ! such that a unit step into the prec. gradient direction leads directly
   ! to the minimum for each band
   ! this is useful if the charge is not updated (LCHCON)
   ! but it seems make calculations faster in any case
   ! if MAXRENORM is 0, no renormalization is performed
   REAL(q) :: MAXRENORM=0.1
   
   ! scaling factor for pseudo-Hamiltonian
   ! note: this is not applied for the sophisticated algorithm where the
   ! pseudo Hamiltonian is determined by subrot
   REAL(q) :: KAPPA=0.4

   INTEGER,SAVE :: ICOUNT=-1, NCOUNT
   REAL(q),SAVE :: GNORML=0, GNORM=0, AVERAG=0, BSTEP=0, FRICTION
   REAL(q),SAVE :: DESUM0, GAMMA, GSL, STRIAL1, STRIAL2
   
   ! these two arrays are required to store the redistributed wavefunctions
   REAL(q)    :: CELTOT(WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN)
   COMPLEX(q) :: CDCHF
 
   PROFILING_START('edwav')

   IALGO=INFO%IALGO; LOVERL=INFO%LOVERL; LREAL=INFO%LREAL
   WEIMIN=INFO%WEIMIN; LCHCON=INFO%LCHCON
   
   CALL PEAD_REINIT_EDWAV(ICOUNT,GNORML,GNORM,AVERAG,BSTEP)
!=======================================================================
! initialise the required variables for MPI
!=======================================================================
#ifdef MPI
   NODE_ME =WDES%COMM_KIN%NODE_ME
   IONODE  =WDES%COMM_KIN%IONODE
   NODE_MEI=WDES%COMM_INTER%NODE_ME ! number of groups (each group holds one band)
   NCPU    =WDES%COMM_INTER%NCPU     ! number of groups (each group holds one band)
#else
   NODE_ME =1
   NODE_MEI=1
   IONODE  =1
   NCPU    =1
#endif

!=======================================================================
! limit scaling
! unfortunately level reordering might spoil somewhat for the new algorithm
!=======================================================================
   CALL MRG_AUX(WDES,W)
   WSCAL=SUM(W%AUXTOT)/SIZE(W%AUXTOT)
   ! W%AUXTOT=W%AUXTOT/WSCAL ; WSCAL = 1  ! this line makes the average scaling 1
   W%AUXTOT=MIN(4._q*WSCAL  ,W%AUXTOT)
   W%AUXTOT=MAX(1/4._q*WSCAL,W%AUXTOT)
!=======================================================================
! number of bands treated simultaneously this must be a multiple of NCPU
!=======================================================================
   NSIM_LOCAL=NSIM/NCPU  ! number of bands optimised on this node
   IF (NSIM_LOCAL*NCPU /= NSIM) THEN
      WRITE(*,*) 'internal ERROR in EDWAV: NSIM is not correct',NSIM
      STOP
   ENDIF
   
!    initialized average step size to trial step size
   IF (AVERAG==0) AVERAG=BTRIAL
   BTRIAL=AVERAG
   IDUMP=1
#ifdef MPI
   IF (NODE_ME/=IONODE) IDUMP=0
#endif

   CALL SETWDES(WDES,WDES1,0)
   DO I=1,NSIM_LOCAL
      CALL NEWWAV(W1(I) , WDES1, .TRUE.)
   ENDDO
   CALL NEWWAV(WSEARCH, WDES1, .TRUE.)
   ALLOCATE(COVL(WDES%NB_TOT,WDES%NB_TOT),CUNI(WDES%NB_TOT,WDES%NB_TOT), &
        CEIDB(WDES%NB_TOT,WDES%NB_TOT))

   NB_TOT=WDES%NB_TOT
   NBANDS=WDES%NBANDS
!*********************************************************************
!   TRIAL STEP (or PREDICTOR STEP)
! ) calculate the gradient vector and store
!   preconditioned gradient in W_G
! ) calculate change in energy if step into direction W_G is done DESUM
! ) calculate norm of gradient-vector GNORM
! ) test orthonormality with old direction ORT
! ) do trial step
!***********************************************************************
   main: IF (IFLAG< 2) THEN
      E%EXHF_ACFDT=0

      ALLOCATE(CPROW(WDES%NPROD,NBANDS))
      ALLOCATE(CF(WDES%NRPLWV,NSIM_LOCAL),CFOLD(WDES%NRPLWV,NSIM_LOCAL),CPROF(WDES%NPROD,NSIM_LOCAL))

      NCOUNT=0

      ORT1  =0
      ORT2  =0
      ORTCEL=0
      DE1   =0
      DE2   =0
      DECEL =0
      CHAM=0
      CDCHF=0
!=======================================================================
! first loop
! calculate gradient (for wavefunctions)
!=======================================================================
      CALL PEAD_ACC_CALC_ALL(W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO)

!=======================================================================
! start with HF part and store results W_G
!=======================================================================
      W_G%CW=0
      IF (LHFCALC) THEN
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES1,NK)

      IF (NONLR_S%LREAL) THEN
        CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
      ELSE
        CALL PHASE(WDES,NONL_S,NK)
      ENDIF
      NBANDSK=CEILING(REAL(W%WDES%NB_TOTK(NK,ISP))/REAL(W%WDES%NB_PAR))
      DO NN=1,NBANDSK,NSIM_LOCAL
         NNMAX=MIN(NN+NSIM_LOCAL-1,NBANDSK)
         IF (LPEAD_NO_SCF()) THEN
            DO N=NN,NNMAX
               NP=N-NN+1; CALL W1_COPY(ELEMENT(W,WDES1,N,ISP),W1(NP))
            ENDDO
            CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W_STORE, &
                    NONLR_S, NONL_S, NK, ISP, NN, NNMAX-NN+1, &
                    W_G%CW(:,NN:,NK,ISP), P, CQIJ(1,1,1,1), CDCHF, W1, &
                    LSYMGRAD=LSYMGRAD, EXHF_ACFDT= E%EXHF_ACFDT)
         ELSE
            CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W, &
                    NONLR_S, NONL_S, NK, ISP, NN, NNMAX-NN+1, &
                    W_G%CW(:,NN:,NK,ISP), P, CQIJ(1,1,1,1), CDCHF, &
                    LSYMGRAD=LSYMGRAD, EXHF_ACFDT= E%EXHF_ACFDT)
         ENDIF
      ENDDO
      ENDDO
      ENDDO

      CALLMPI( M_sum_z(WDES1%COMM_KIN,CDCHF,1))
      CALLMPI( M_sum_z(WDES%COMM_KINTER, CDCHF, 1))
!      WRITE(*,*) '-1/2 Fock energy is ',CDCHF
      E%EXHF=CDCHF
      CALLMPI( M_sum_d(WDES1%COMM_KIN, E%EXHF_ACFDT ,1))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, E%EXHF_ACFDT, 1))
      IF (LSYMGRAD) THEN
        CALL APPLY_SMALL_SPACE_GROUP_OP( W, W_G, NONLR_S, NONL_S,P, T_INFO%NIONS, LATT_CUR, SYMM, CQIJ, .FALSE. , -1)
      ENDIF
      ENDIF

!=======================================================================
! DFT part optimized to yield minimum number of operations
!=======================================================================
spin:   DO ISP=1,WDES%ISPIN
kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

      CALL SETWDES(WDES,WDES1,NK)

      WA=ELEMENTS(W, WDES1, ISP)
      WA_F=ELEMENTS(W_F, WDES1, ISP)
 
      IF (WDES%DO_REDIS) THEN
        CALL SET_GPOINTER(CPROW_RED, WDES1%NPROD_RED, NB_TOT, CPROW(1,1))
        CALL SET_WPOINTER(CF_RED,   WDES1%NRPLWV_RED, NSIM, CF(1,1))
        CALL SET_GPOINTER(CPROF_RED, WDES1%NPROD_RED, NSIM, CPROF(1,1))
      ELSE
        CPROW_RED => CPROW(:,:)
        CF_RED    => CF(:,:)
        CPROF_RED => CPROF(:,:)
      ENDIF

      NGVECTOR=WDES1%NGVECTOR
      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,ISP), W%CPROJ(1,1,NK,ISP),CPROW(1,1))

!-----------------------------------------------------------------------
! avoid conjugation of those bands which do not contribute to energy
!-----------------------------------------------------------------------
      ! wave functions coefficients are redistributed in W_F
      DO N1=1,NB_TOT
         IF (ABS(W%FERTOT(N1,NK,ISP))<WEIMIN .AND. IALGO>=8) THEN
            DO M=1,WDES1%NPL_RED
               WA_F%CW_RED(M,N1)=0
            ENDDO
         ENDIF
      ENDDO
      ! same for character (which is distributed differently on the nodes)
      DO N=1,NBANDS
         IF (ABS(W%FERWE(N,NK,ISP))<WEIMIN .AND. IALGO>=8) THEN
            DO NPRO_=1,WDES%NPRO
               WA_F%CPROJ(NPRO_,N)=0
            ENDDO
         ENDIF
      ENDDO
!-----------------------------------------------------------------------
! calculate the phase-factor on the compressed grid (only non local)
!-----------------------------------------------------------------------
      IF (LREAL) THEN
        CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
      ELSE
        CALL PHASE(WDES,NONL_S,NK)
      ENDIF

      NBANDSK=CEILING(REAL(W%WDES%NB_TOTK(NK,ISP))/REAL(W%WDES%NB_PAR))

      ! redistribute over plane wave coefficients
      IF (WDES%DO_REDIS) THEN
        IF (LOVERL) CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW(1,1,NK,ISP))
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
      ENDIF
!=======================================================================
  bands: DO NN=1,NBANDSK,NSIM_LOCAL
!=======================================================================
      CF=0
      CPROF=0
      CFOLD=0

      NNMAX=MIN(NN+NSIM_LOCAL-1,NBANDSK)
      NSIM_=NNMAX-NN+1

      DO N=NN,NNMAX ! each step in the loop copies NSIM_LOCAL/NSIM bands !
         NP=N-NN+1
         CF(:,NP)=W_G%CW(:,N,NK,ISP)
         CALL SETWDES(WDES, WDES1, NK)
         ! use as source the wavefunctions distributed over bands (if possible)
         DO M=1,WDES%NRPLWV
            W1(NP)%CW(M)=W%CW(M,N,NK,ISP)
            CFOLD(M,NP) =W_F%CW(M,N,NK,ISP)
         ENDDO
         DO M=1,WDES%NPROD
            W1(NP)%CPROJ(M)=W%CPROJ(M,N,NK,ISP)
         ENDDO
         ! redistribute over bands
         IF (WDES%DO_REDIS) THEN
            CALL REDIS_PW(WDES1, 1, W1(NP)%CW(1))
            CALL REDIS_PROJ(WDES1, 1, W1(NP)%CPROJ(1))
            CALL REDIS_PW(WDES1, 1, CFOLD(1,NP))
         ENDIF

         DO ISPINOR=0,WDES%NRSPINORS-1
            CALL FFTWAV(NGVECTOR, WDES%NINDPW(1,NK),W1(NP)%CR(1+ISPINOR*GRID%MPLWV),W1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
         ENDDO

         IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
            CALL HAMILT_LOCAL_TAU(W1(NP), SV, LATT_CUR, HAMILTONIAN%MU, ISP, CF(:,NP), LHFCALC)     
! test
!           CALL ECCP_TAU(WDES1,W1(NP) ,W1(NP) ,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),CE)
!           CALL HAMILT_LOCAL(W1(NP), SV, ISP, CF(:,NP), LHFCALC)
! test
         ELSE
            CALL HAMILT_LOCAL(W1(NP), SV, ISP, CF(:,NP), LHFCALC)
         ENDIF

         WDES1%NBANDS=1    ! WDES1%NBANDS is used only here to fake OVERL
         CALL OVERL(WDES1, .TRUE., LMDIM, CDIJ(1,1,1,ISP), W1(NP)%CPROJ(1), CPROF(1,NP))

         CALL PEAD_ACC_ADD(CF(:,NP),CPROF(:,NP),N,NK,ISP)

         IF ((W%WDES%NB_LOW+(N-1)*W%WDES%NB_PAR)>W%WDES%NB_TOTK(NK,ISP)) THEN
            CF(:,NP)=0
            CPROF(:,NP)=0
            W1(NP)%CW=0
            W1(NP)%CR=0
            W1(NP)%CPROJ=0
         ENDIF

         ! redistribute over plane wave coefficients
         IF (WDES%DO_REDIS) THEN
            CALL REDIS_PW(WDES1, 1, CF(1,NP))
            CALL REDIS_PROJ(WDES1, 1, CPROF(1,NP))
         ENDIF
      ENDDO

! calculate the lower triangle of the Hamilton matrix
! in the subspace spanned by the present set of wavefunctions

      NPOS_RED  =(NN-1)*NCPU+1
      NSTRIP_RED=NSIM_*NCPU

      CALL ORTH1('L', &
        WA%CW_RED(1,1),CF(1,1),WA%CPROJ_RED(1,1), &
        CPROF(1,1),NB_TOT, &
        NPOS_RED, NSTRIP_RED, WDES1%NPL_RED, WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1,NK,ISP))

      CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM(1,NPOS_RED,NK,ISP),NB_TOT*NSTRIP_RED))

!     fill in upper triangle
!     in the parallel version part of the upper triangle has been destroyed
!     and needs to be refilled
      DO N1=NPOS_RED,NPOS_RED+NSTRIP_RED-1
         DO N2=N1+1,NB_TOT
            CHAM(N1,N2,NK,ISP)=GCONJG(CHAM(N2,N1,NK,ISP))
         ENDDO
         W%CELTOT(N1,NK,ISP)=CHAM(N1,N1,NK,ISP)
         DO N2=1,N1-1
            CHAM(N2,N1,NK,ISP)=GCONJG(CHAM(N1,N2,NK,ISP))
         ENDDO
      ENDDO

      ! set all matrix elements beyond NB_TOTK to zero
      IF (W%WDES%NB_TOTK(NK,ISP)<NB_TOT) THEN
         CHAM(:,W%WDES%NB_TOTK(NK,ISP)+1:NB_TOT,NK,ISP)=0
         CHAM(W%WDES%NB_TOTK(NK,ISP)+1:NB_TOT,:,NK,ISP)=0
      ENDIF

!
! subtract from H phi the components parallel to the subspace
! spanned by the other orbitals in the set
      IF ( WDES1%NPL_RED/=0) &
! subtract - < phi_n1 | H_local  | phi_n2 >  | phi_n2>
      CALL GGEMM('N', 'N', m_  WDES1%NPL_RED , NSTRIP_RED, NB_TOT, -one, &
     &               WA%CW_RED(1,1), m_ WDES1%NRPLWV_RED, CHAM(1,NPOS_RED,NK,ISP), NB_TOT,  &
     &               one, CF(1,1), m_ WDES1%NRPLWV_RED)

! subtract < phi_n1 |  H_local | phi_n2 > Q_ij  < p_j| phi_n2 > from orbital character
      IF (LOVERL) THEN
         CALL GGEMM('N', 'N',  WDES1%NPRO_RED, NSTRIP_RED, NB_TOT, -one, &
     &               CPROW_RED(1,1), WDES1%NPROD_RED, CHAM(1,NPOS_RED,NK,ISP), NB_TOT,  &
     &               one, CPROF(1,1), WDES1%NPROD_RED)
      ENDIF

      ! redistribute the gradient over bands
      DO N=NN,NNMAX
         NP=N-NN+1
         IF (WDES%DO_REDIS) THEN
            CALL REDIS_PW(WDES1, 1, CF(1,NP))
            CALL REDIS_PROJ(WDES1, 1, CPROF(1,NP))
         ENDIF
      ENDDO

    innerband: DO N=NN,NNMAX
      IF ( (W%WDES%NB_LOW+(N-1)*W%WDES%NB_PAR)>W%WDES%NB_TOTK(NK,ISP)) CYCLE

      NP=N-NN+1
!-----------------------------------------------------------------------
! add the non local contribution to CF
!-----------------------------------------------------------------------
      IF (LREAL) THEN
         WSEARCH%CR(:)=0
         CALL RACC0(NONLR_S,WDES1,CPROF(1,NP),WSEARCH%CR(1))
         DO ISPINOR=0,WDES1%NRSPINORS-1
            CALL FFTEXT(NGVECTOR,WDES%NINDPW(1,NK),WSEARCH%CR(1+ISPINOR*WDES1%GRID%MPLWV),CF(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
         ENDDO
      ELSE
        CALL VNLAC0(NONL_S,WDES1,CPROF(1,NP),CF(1,NP))
      ENDIF

      WEIGHT =2*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP) ! times two to include the c.c.
!-----------------------------------------------------------------------
! perform preconditioning of gradient
! CF is the current "gradient" WSEARCH the direction to go
!-----------------------------------------------------------------------
      WSEARCH%CW=CF(:,NP)
      CALL SIMPLE_PRECOND( WSEARCH )      
!----------------------------------------------------------------------
!  calculate the character of the wavefunctions
!----------------------------------------------------------------------
      DO ISPINOR=0,WDES1%NRSPINORS-1
         CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),WSEARCH%CR(1+ISPINOR*WDES1%GRID%MPLWV),WSEARCH%CW(1+ISPINOR*NGVECTOR),GRID)
      ENDDO
      IF (LREAL) THEN
         CALL RPRO1(NONLR_S,WDES1,WSEARCH)
      ELSE
         CALL PROJ1(NONL_S,WDES1,WSEARCH)
      ENDIF

!-----------------------------------------------------------------------
! following statements allow to renormalize the trial step
! such that a unit step into the direction of the precond gradient
! steps directly into the minimum
!-----------------------------------------------------------------------
      ! project out current wave function

      CALL PROJCN(WSEARCH,W1(NP),CQIJ, ISP, CSCPD)
      IF (MAXRENORM/=0.AND..NOT.LPEAD_EFIELD_SWITCHED_ON()) THEN     
!     IF (MAXRENORM/=0) THEN     
         ! renormalize the wave function
         CALL CNORMN(WSEARCH,CQIJ, ISP, WSCAL)

         ! recalculate the wavefunction in real space
         CALL CNORMN_REAL(WSEARCH, W1(NP), ISP, WSCAL, CSCPD )

         ! determine 2x2 sub space Hamilton matrix and diagonalize it
         ! use local contributions only
         CB_=0
         DO M=1,WDES1%NPL
            CB_ =CB_+ CF(M,NP)*CONJG(WSEARCH%CW(M))
         ENDDO
         CALLMPI( M_sum_g(WDES%COMM_INB, CB_, 1))

         IF (ASSOCIATED(HAMILTONIAN%MU)) THEN
            CALL ECCP_TAU(WDES1,W1(NP) ,W1(NP) ,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),CE)
            CALL ECCP_TAU(WDES1,WSEARCH,WSEARCH,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),CA)
            CALL ECCP_TAU(WDES1,W1(NP) ,WSEARCH,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP),LATT_CUR,HAMILTONIAN%MU(:,ISP),CB)
         ELSE
            CALL ECCP(WDES1,W1(NP) ,W1(NP),  LMDIM,CDIJ(1,1,1,ISP),GRID, SV(1,ISP), CE)
            CALL ECCP(WDES1,WSEARCH,WSEARCH, LMDIM,CDIJ(1,1,1,ISP),GRID, SV(1,ISP), CA)
            CALL ECCP(WDES1,W1(NP) ,WSEARCH, LMDIM,CDIJ(1,1,1,ISP),GRID, SV(1,ISP), CB)
         ENDIF

         ! determine optimal step size
         DIFCEL =CA-CE
         CROT   =CB/DIFCEL

         THETA=.5_q*ATAN(2*ABS(CROT))
         
         CROT =  SIN(THETA)*CROT/MAX(ABS(CROT),1E-8_q)

         ! here the real important trick
         ! CROT*WSCAL is a kind of preconditioner by which we should scale
         ! the gradient, but only the average is sensible 
         ! in each single step the quantity is pretty useless and spoiles
         ! the conjugate gradient cycle

         ! maximum allowed change given by MAXRENORM
         W%AUX(N,NK,ISP)=MIN(W%AUX(N,NK,ISP)*(1+MAXRENORM),MAX(W%AUX(N,NK,ISP)*(1-MAXRENORM), &
                         W%AUX(N,NK,ISP)*(1-MAXRENORM)+REAL(CROT,q)*WSCAL*MAXRENORM))

         ! now scale the search direction by this value
         WSCAL=W%AUX(N,NK,ISP)/WSCAL
         CALL W1_DSCAL( WSEARCH, WSCAL)
         IF (IDUMP>=3) WRITE(*,'(1F9.4,"scal")',ADVANCE='NO') W%AUX(N,NK,ISP)

      ENDIF
!-----------------------------------------------------------------------
! calculate expected energy-change into search direction
! and store the (preconditioned) gradient in W_G%CW
!-----------------------------------------------------------------------
      ! gradient should be orthogonal to present wavefunction
      DORT=0
      DO M=1,WDES1%NPL
         DORT  =DORT+ CF(M,NP)*CONJG(W1(NP)%CW(M))
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM_INB, DORT, 1))
      IF (ABS(DORT)>1E-4) THEN
         WRITE(*,*) 'EDWAV: internal error, the gradient is not orthogonal',NK,NP,DORT !,WDES%COMM%NODE_ME
         STOP
      ENDIF

      ! orthogonality of gradient to previous search direction
      DORT=0
      DO M=1,WDES1%NPL
         DORT  =DORT+ CF(M,NP)*CONJG(CFOLD(M,NP))
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM_INB, DORT, 1))
      IF (IDUMP>=3) WRITE(*,'(F9.4,"O")',ADVANCE='NO') DORT
      ORT1  =ORT1+ DORT *WEIGHT *WDES%RSPIN

      ! energy change for a unit step along gradient
      IF (IROT==1 .OR. IROT>=3 ) THEN
         DE=0
         DO M=1,WDES1%NPL
            DE =DE+ CF(M,NP)*CONJG(WSEARCH%CW(M))
         ENDDO
         CALLMPI( M_sum_d(WDES%COMM_INB, DE, 1))
         IF (DE<0) THEN
            WRITE(*,'("ups2",I4,8F14.7)') N,REAL(W%CELEN(N,NK,ISP))-REAL(CA),CB_,CROT,THETA,DE
         ENDIF
         IF (IDUMP>=3)  WRITE(*,'(E14.5,"E")',ADVANCE='NO') DE

         DE1   =DE1   +DE*WEIGHT *WDES%RSPIN
         
         W_G%CPTWFP(:,N,NK,ISP)=0
         DO M=1,WDES1%NPL
            W_G%CW(M,N,NK,ISP)=WSEARCH%CW(M)
         ENDDO
         
         W_G%CPROJ(:,N,NK,ISP)=0
         DO NPRO_=1,WDES%NPRO
            W_G%CPROJ(NPRO_,N,NK,ISP)=WSEARCH%CPROJ(NPRO_)
         ENDDO

      ENDIF
      IF (IDUMP>=3)  WRITE(*,*) 
!=======================================================================
    ENDDO innerband
  ENDDO bands
!=======================================================================

      IF (WDES%DO_REDIS) THEN
! redistribute CPROJ over  bands 
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))        
! gradient distributed over plane wave coefficients
        CALL REDIS_PW  (WDES1, NBANDS, W_G%CW   (1,1,NK,ISP))
      ENDIF

      ! set CELTOT too huge eigenvalues for bands beyond NB_TOTK
      ! occupancies zero
      IF (W%WDES%NB_TOTK(NK,ISP)<NB_TOT) THEN
         DO N=W%WDES%NB_TOTK(NK,ISP)+1,NB_TOT
            W%CELTOT(N,NK,ISP)=10000._q
            W%FERTOT(N,NK,ISP)=0._q
            CHAM(N,N,NK,ISP)=10000._q
         ENDDO
      ENDIF
!=======================================================================
ENDDO kpoint
ENDDO spin
!=======================================================================

!      WRITE(*,*) '-1/2 Fock energy is ',CDCHF
      
      CALLMPI( M_sum_d(WDES%COMM_INTER, ORT1, 1))
      CALLMPI( M_sum_d(WDES%COMM_INTER, DE1, 1))

      CALLMPI( M_sum_d(WDES%COMM_KINTER, ORT1, 1))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, DE1, 1))

      IF (WDES%DO_REDIS) THEN
         W%OVER_BAND=.TRUE.
         W_G%OVER_BAND=.TRUE.
         W_F%OVER_BAND=.TRUE.
      ENDIF
! at this point W and W_G are distributed over plane wave coefficients
! orthogonalize the search direction to current orbitals
!#define none
!#ifdef none
      DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            CALL SETWDES(WDES,WDES1,NK)
         ! single node and unefficient version
            !DO N1=1,NB_TOT
            !   CALL ORTHON(NK,W,ELEMENT(W_G,WDES1,N1,ISP),CQIJ,ISP)
            !ENDDO
            CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,ISP), W_G%CPROJ(1,1,NK,ISP),CPROW(1,1))

            IF (WDES%DO_REDIS) THEN
               IF (LOVERL) CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
               CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PROJ(WDES1, NBANDS, W_G%CPROJ(1,1,NK,ISP))
            ENDIF

            COVL=0
            CALL ORTH2( &
                 W%CW(1,1,NK,ISP),W_G%CW(1,1,NK,ISP),W%CPROJ(1,1,NK,ISP), &
                 CPROW(1,1),NB_TOT, &
                 1, NB_TOT, WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
            
            CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))

            IF ((W%WDES%NB_TOTK(NK,ISP)+1)<=NB_TOT) &
           &    COVL(:,W%WDES%NB_TOTK(NK,ISP)+1:)=0; COVL(W%WDES%NB_TOTK(NK,ISP)+1:,:)=0

            IF (WDES1%NPL_RED /=0 ) &
            CALL GGEMM('N', 'N', m_ WDES1%NPL_RED , NB_TOT, NB_TOT, -one, &
                 W%CW(1,1,NK,ISP), m_ WDES1%NRPLWV_RED, COVL(1,1), NB_TOT,  &
                 one, W_G%CW(1,1,NK,ISP), m_ WDES1%NRPLWV_RED)

            IF ( WDES1%NPRO_RED /=0) &
            CALL GGEMM('N', 'N', WDES1%NPRO_RED , NB_TOT, NB_TOT, -one, &
                 W%CPROJ(1,1,NK,ISP), WDES1%NPROD_RED, COVL(1,1), NB_TOT,  &
                 one, W_G%CPROJ(1,1,NK,ISP), WDES1%NPROD_RED)

            IF (WDES%DO_REDIS) THEN
               CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
               CALL REDIS_PROJ(WDES1, NBANDS, W_G%CPROJ(1,1,NK,ISP))
            ENDIF
         ENDDO
      ENDDO
!#endif
      DEALLOCATE(CPROW)
      DEALLOCATE(CF,CFOLD,CPROF)

!-----------------------------------------------------------------------
! conjugate CHAM
!  CHAM(n2,n1) = < phi_n2 | H | phi_n1 >* = < phi_n1 | H | phi_n2 >
! why this is done, I do not know. Historic reasons, but really
! this is a mess
!-----------------------------------------------------------------------
      CHAM=GCONJG(CHAM)

      IF (IDUMP>3) THEN
        DO ISP=1,WDES%ISPIN
           DO NK=1,WDES%NKPTS
#ifdef MPI
              IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

              CALL DUMP_HAM( "Hamilton matrix",WDES, CHAM(1,1,NK,ISP))
           ENDDO
        ENDDO
      ENDIF
!=======================================================================
! calculate energy-change due to subspacerotation
! and the orthonormality to the last rotation
! the gradient is  simply  i [H,F]
! ORT   = i [H,F] * CHF (CHF== last rotation matrix)
! DE2   = i [H,F] * new rotation matrix
!=======================================================================
      IF (IROT==4 .OR. IROT==5 .OR. IROT==7) THEN
         ALLOCATE(CHAMP(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
         CHAMP=CHAM

         CALL SUBROT_SCF( &
              HAMILTONIAN,KINEDEN, &
              P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
              T_INFO,INFO,IO,MIX,KPOINTS,SYMM, &
              GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,E, &
              DENCOR,CSTRF,N_MIX_PAW,LMDIM,IRDMAX, &
              CHTOT,RHOLM,CHAMP, IROT==4 .OR. IROT==7)
         IF (IROT==5) THEN
            DO ISP=1,WDES%ISPIN
            DO NK=1,WDES%NKPTS
#ifdef MPI
               IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

               CALL SETWDES(WDES,WDES1,NK)
               IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_G%CPROJ(1,1,NK,ISP))
               CALL LINCOM('F',W_G%CW(:,:,NK,ISP),W_G%CPROJ(:,:,NK,ISP),CHAMP(1,1,NK,ISP), &
                    W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
                    WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
                    W_G%CW(:,:,NK,ISP),W_G%CPROJ(:,:,NK,ISP))
               IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_G%CPROJ(1,1,NK,ISP))

               IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
               CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CHAMP(1,1,NK,ISP), &
                    W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
                    WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
                    W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
               IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
            ENDDO
            ENDDO
            ! store diagonals for later use (update of fermi-weights)
            IF (ICEL/=0) THEN
               WRITE(*,*) 'internal error in EDWAV: for IROT=5, ICEL must be 0'
               STOP
            ENDIF
         ELSE
            ! store diagonals for later use (update of fermi-weights)
            DO ISP=1,WDES%ISPIN
               DO NK=1,WDES%NKPTS
#ifdef MPI
                  IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
                  DO N=1,WDES%NB_TOT
                     CELTOT(N,NK,ISP)=CHAMP(N,N,NK,ISP)
                  ENDDO
               ENDDO
            ENDDO
            ! scale the rotation matrix by the mean value of 1/BTRIAL and 1
            ! this moves the typical rotation frequencies into the middle of the spectrum
            IF (IROT==4 .OR. IROT==5) THEN
               IF (ILOEW==1 .OR. ILOEW==2) &
               CALL ROTINI(WDES, W, W_F, CHAM, CHF, CEIDB, WEIMIN, ORT2, DE2, &
                    !  .TRUE. indicates here that the rotation matrix is not
                    ! conditioned by ROTINI
                    .TRUE., CHAMP, ((1/BTRIAL+1)/2))
            ELSE
               CALL ETAINI(WDES, W, W_F, CHAM, CHF, CEIDB, KPOINTS, EFERMI, WEIMIN, & 
                    ((1/BTRIAL+1)/2), ORT2, DE2, ORTCEL, DECEL, &
                    .TRUE., CHAMP)
            ENDIF
         ENDIF
         DEALLOCATE(CHAMP)
      ELSE IF (IROT==2 .OR. IROT==3) THEN
         IF (ILOEW==1 .OR. ILOEW==2) THEN
            CALL ROTINI(WDES, W, W_F, CHAM, CHF, CEIDB, WEIMIN, ORT2, DE2, &
                 LCHCON )
         ENDIF
      ELSE IF (IROT==6) THEN
         CALL ETAINI(WDES, W, W_F, CHAM, CHF, CEIDB, KPOINTS, EFERMI, WEIMIN, & 
              KAPPA, ORT2, DE2, ORTCEL, DECEL, &
              LCHCON )
      ENDIF
!=======================================================================
!
! calculate gradient for fermi-weights
! actually the fermi-weights are calculated from W_F%CELTOT
! and W_F%CELTOT is optimized in the course of the calculations
! the prec. gradient for W_F%CELTOT is stored in W_G%FERTOT
! and the search direction in W_F%FERTOT
!
!=======================================================================
      IF (KPOINTS%ISMEAR>=-1 .AND. KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
         EFERMI_SHIFT=0
         WSUM =0

!      depsilon_Fermi =
!     [\sum_n df_n/d(epsilon_n-epsilon_Fermi) x d epsilon_n ]/ [\sum_n df_n/d(epsilon_n-epsilon_Fermi)]
         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,W%WDES%NB_TOTK(NK,ISP)
               X=  (EFERMI- REAL( W_F%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               WSUM =WSUM +DFUN* WEIGHT
               EFERMI_SHIFT=EFERMI_SHIFT+DFUN*(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))/KPOINTS%SIGMA*WEIGHT
            ENDDO
         ENDDO
         ENDDO

         CALLMPI( M_sum_d(WDES%COMM_KINTER,WSUM,1) )
         CALLMPI( M_sum_d(WDES%COMM_KINTER,EFERMI_SHIFT,1) )

      ! shift is the first order change in the Fermi-energy
         IF (ABS(WSUM)>1E-10) THEN
            EFERMI_SHIFT=EFERMI_SHIFT/WSUM
         ELSE
            EFERMI_SHIFT=0
         ENDIF
         WMIN=WSUM/WDES%NB_TOT*WEIMIN

         DECEL_in=DECEL
         ORTCEL_in=ORTCEL
         DECEL=0
         ORTCEL=0

         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,W%WDES%NB_TOTK(NK,ISP)
               X=  (EFERMI- REAL( W_F%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               DER = -DFUN/KPOINTS%SIGMA
              ! avoid conjugation of those elements which do not contribute to energy
              ! for CG algorithm
               IF (ABS(DFUN)<WMIN .AND. IALGO>=8) THEN
                  W_F%FERTOT(N,NK,ISP)=0
               ENDIF
               DFUN= DFUN*((W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))/KPOINTS%SIGMA-EFERMI_SHIFT)

               IF (IROT==4) THEN
             ! for IROT = 4 we simply set the direction such that the dummy eigenvalues
             ! move into the direction of the optimized
             ! sub space Hamiltonian (stored few lines above)
                  SFUN= (W_F%CELTOT(N,NK,ISP)-CELTOT(N,NK,ISP))
               ELSE IF (.NOT.LCHCON) THEN
             ! precond. gradient, damp response somewhat
                  SFUN= (W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))/2
               ELSE
             ! preconditioned gradient equal gradient
                  SFUN=(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))
               ENDIF

               DECEL  =DECEL  +DFUN* SFUN*WEIGHT
               ORTCEL =ORTCEL +DFUN* W_F%FERTOT(N,NK,ISP)*WEIGHT
               W_G%FERTOT(N,NK,ISP)= SFUN
            ENDDO
         ENDDO
         ENDDO

         CALLMPI( M_sum_d(WDES%COMM_KINTER,DECEL,1) )
         CALLMPI( M_sum_d(WDES%COMM_KINTER,ORTCEL,1) )
         DECEL=DECEL+DECEL_in
         ORTCEL=ORTCEL+ORTCEL_in

      ELSE IF (KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
         DECEL  =0
         ORTCEL =0
         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,W%WDES%NB_TOTK(NK,ISP)
               W_G%FERTOT(N,NK,ISP)=(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))
               IF (.NOT.LCHCON) THEN
                  W_G%FERTOT(N,NK,ISP)=(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))/2
               ENDIF
            ENDDO
         ENDDO
         ENDDO
      ENDIF
! set selected components of the gradient to zero 
! in order to test things
!test
!      W_G%CW=0;  W_G%CPROJ=0; DE1=0 ; ORT1=0

!      DO N1=1,WDES%NBANDS
!         CHAM(N1,N1,:,:)=0
!      ENDDO
!      W_G%FERTOT=0; DECEL=0; ORTCEL=0
      
!      DO N1=1,WDES%NBANDS ; DO N2=N1+1,WDES%NBANDS
!         CHAM(N1,N2,:,:)=0 ; CHAM(N2,N1,:,:)=0
!      ENDDO ; ENDDO
!      DE2=0; ORT2=0 ! CHAM=0

      GNORM =DECEL  +DE1  +DE2
      ORT   =ORTCEL +ORT1 +ORT2
!=======================================================================
! perform CORRECTOR step taking into account the new gradient only
!=======================================================================
 cor: IF (IALGO==9 .AND.IFLAG/=0 .AND. MOD(ICOUNT,2)==0) THEN
      
      BCSTEP= BTRIAL*ORT/(GSL-ORT)
      BSTEP = BTRIAL+BCSTEP

      AVERAG=AVERAG*0.95_q+BSTEP*0.05_q

      IF (IO%IU0>=0) &
      WRITE(IO%IU0,12,ADVANCE='NO') BCSTEP,GSL,ORT
   12 FORMAT('cor-step=',F10.5,' , ',F13.6,F13.6)
      ICOUNT=ICOUNT+1
      DESUM1= -GSL*(BCSTEP+BTRIAL)/2

! step for wavefunction

      IF (IROT==1 .OR. IROT>=3) THEN
        DO ISP=1,WDES%ISPIN
        DO NK=1,WDES%NKPTS
#ifdef MPI
           IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        DO N=1,WDES%NBANDS
          DO M=1,WDES%NRPLWV
            W%CPTWFP(M,N,NK,ISP)=W%CPTWFP(M,N,NK,ISP)   -W_F%CW(M,N,NK,ISP)      *BCSTEP
          ENDDO

          DO NPRO_=1,WDES%NPROD
            W%CPROJ(NPRO_,N,NK,ISP)=W%CPROJ(NPRO_,N,NK,ISP)-W_F%CPROJ(NPRO_,N,NK,ISP)*BCSTEP
          ENDDO
        ENDDO
        ENDDO
        ENDDO
      ENDIF

! step for suspace-matrix
      IF (IROT==2 .OR. IROT==3 .OR. IROT==4) THEN
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)
#ifdef testrot
       CEIDB(:,:)=CHF(:,:,NK,ISP)
#endif
        IF (ILOEW==1) THEN
           CALL ROT1(NB_TOT,CHF(1,1,NK,ISP),BCSTEP,CUNI,CEIDB,COVL)
        ELSE IF (ILOEW==2) THEN
           CALL ROT2(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),BCSTEP,CUNI,CEIDB,COVL)
        ENDIF

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
      ENDDO
      ENDDO
      ELSE IF (IROT==6 .OR. IROT==7) THEN

      CALL MRG_AUX(WDES,W)

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)

        CALL ROTETA(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),W_F%CELTOT(:,NK,ISP),BCSTEP,CUNI,CEIDB,COVL,W%AUXTOT(:,NK,ISP))

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        
        ! rotate W_F accordingly so that conjugation works out in next step (orbital change)
        ! must check whether ILOEW=2 can swap bands (if not one might comment this call out)
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
      ENDDO
      ENDDO
      ENDIF

      CALL ORTHCH(W%WDES,W, LOVERL, LMDIM,CQIJ)

! step for fermi-weights

      IF (KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
        DO ISP=1,WDES%ISPIN
        DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        DO N =1,W%WDES%NB_TOTK(NK,ISP)
          W_F%CELTOT(N,NK,ISP)=W_F%CELTOT(N,NK,ISP)-BCSTEP*W_F%FERTOT(N,NK,ISP)
        ENDDO
        ENDDO
        ENDDO
      ENDIF

      IFLAG=0
      ORT=0
      GOTO 1000

 ENDIF cor
!=======================================================================
! performe trial step (possibly with conjugation of search directions)
! and calculate change in energy up to first order
! the conjugate gradient optimization is restarted
! ) after 6 steps
! ) based on a 10 %  criterion
!=======================================================================
      IF ( IALGO==3 .AND. ICOUNT ==-1) THEN
         ! ideally I would like to use the following term
         ! for integrating the equation of motion
         ! VEL=((1-FRICTION/2)*VEL+2*F*TIMESTEP)/(1+FRICTION/2)
         ! but this is not what is implemented
         ! hence we rescale the timestep (BTRIAL) and set GAMMA properly
         !
         ! FRICTION is proportional to sqrt(min eigenvalue(P H) / max eigenvalue(P H))
         ! The maximum eigenvalue is proportional to the inverse step size.
         ! The minimum eigenvalue is 1 for insulators and around 2 for metals
         ! since the steps on the eigenvalues are damped by 2.
         ! some overrelaxation also never harms
         ! For non collinear calculations, however very soft modes exist
         ! and one might need to "over-relax" e.g. FRICTION=SQRT(BTRIAL*100)
         IF (ICEL==0) THEN
            FRICTION=SQRT(BTRIAL)
         ELSE
            FRICTION=SQRT(BTRIAL*2)
         ENDIF

         ! in principle BTRIAL needs to be rescaled
         ! but empirically that does not improve convergence, hence stick to the original timestep
         ! BTRIAL=BTRIAL/(1+FRICTION/2)*2
      ENDIF

      IF ( IALGO==3 .AND. IFLAG==1 .AND. ICOUNT /=-1) THEN
         GAMMA=(1-FRICTION/2)/(1+FRICTION/2)
         IRET    =3
      ELSE IF ( IALGO==4 .AND. IFLAG==1 .AND. ICOUNT /=-1) THEN
         GAMMA=1
         ! DESUM0 expected first order energy change
         ! when larger 0 we are starting to move uphill
         ! hence the velocities are quenched to zero
         DESUM0=GAMMA*ORT
         ICOUNT=ICOUNT+1
         IF (DESUM0<0) THEN
            ICOUNT=0
            GAMMA=0
         ENDIF
         IRET  =4
      ! reset every 30 steps
      ELSE IF ( (IALGO <8) .OR.  IFLAG/=1 .OR. ICOUNT == 30 &
     &   ) THEN
        GAMMA =0
        ICOUNT=0
        IRET  =0
      ELSE
! Fletcher-Reeves
        GAMMA=GNORM /GNORML
        ICOUNT=ICOUNT+1
        IRET  =1
        ! previous line minization was so bad that ORT*GAMMA is larger
        ! then it is useless to conjugate
        ! (ORT should be essentially zero)
        IF ( ABS(GAMMA*ORT) > ABS(GNORM)/2) THEN
          GAMMA=0
          ICOUNT=0
        ENDIF
      ENDIF
!test
!      GAMMA=0

      IF (IDUMP>=1.AND. IO%IU0>=0) &
     &  WRITE(IO%IU0,20) GAMMA,DE1,DE2,DECEL,ORT1,ORT2,ORTCEL
      GNORML=GNORM
      RMS   =GNORM

  20  FORMAT(' gam=',F6.3, &
           &  ' g(H,U,f)= ',3E10.3,' ort(H,U,f) =',3E10.3)
!-----------------------------------------------------------------------
!  DESUM0 is expected first order energy-change into the trial direction
!  the trial direction is  f(N) = p(N) + GAMMA f(N-1)
!  (g(N) gradient, p(N) preconditioned gradient, f(N) search direction)
!  because ORT= g(N) f(N-1) and GNORM = g(N) p(N)
!  DESUM is given by DESUM = g(N) * f(N) = GNORM+GAMMA* ORT
!-----------------------------------------------------------------------
      DESUM0=GNORM+GAMMA*ORT
      GSL   =GNORM+GAMMA*ORT
!-----------------------------------------------------------------------
! conjugate directions
!-----------------------------------------------------------------------
      conjug: DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      IF (IROT==1 .OR. IROT>=3) THEN
      DO N=1,WDES%NBANDS
        DO M=1,WDES%NRPLWV
           W_F%CW(M,N,NK,ISP)      =W_G%CW(M,N,NK,ISP)    +W_F%CW(M,N,NK,ISP)*GAMMA
        ENDDO
        DO NPRO_=1,WDES%NPROD
          W_F%CPROJ(NPRO_,N,NK,ISP)=W_G%CPROJ(NPRO_,N,NK,ISP)+W_F%CPROJ(NPRO_,N,NK,ISP)*GAMMA
        ENDDO
      ENDDO
      ENDIF

      IF (IROT==2 .OR. IROT>=3) THEN
        DO N2=1,NB_TOT
        DO N1=1,NB_TOT
          CHF(N1,N2,NK,ISP)=CHAM(N1,N2,NK,ISP)+GAMMA*CHF(N1,N2,NK,ISP)
        ENDDO
        ENDDO
      ENDIF

      IF (KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
        DO N =1,W%WDES%NB_TOTK(NK,ISP)
          W_F%FERTOT(N,NK,ISP)=W_G%FERTOT(N,NK,ISP)+W_F%FERTOT(N,NK,ISP)*GAMMA
        ENDDO
      ENDIF

      ENDDO
      ENDDO conjug
!-----------------------------------------------------------------------
! trial step
!-----------------------------------------------------------------------
      BSTEP=BTRIAL
      IF (IROT==1 .OR. IROT>=3) THEN
        DO ISP=1,WDES%ISPIN
        DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        DO N=1,WDES%NBANDS
          DO M=1,WDES%NRPLWV
#ifdef restore_W
             W_G%CW(M,N,NK,ISP)    =W%CW(M,N,NK,ISP)
#endif
             W%CW(M,N,NK,ISP)=W%CW(M,N,NK,ISP)-W_F%CW(M,N,NK,ISP)*BSTEP
          ENDDO
          DO NPRO_=1,WDES%NPROD
#ifdef restore_W
            W_G%CPROJ(NPRO_,N,NK,ISP)=W%CPROJ(NPRO_,N,NK,ISP)
#endif
            W%CPROJ(NPRO_,N,NK,ISP)=W%CPROJ(NPRO_,N,NK,ISP)-W_F%CPROJ(NPRO_,N,NK,ISP)*BSTEP
          ENDDO

        ENDDO
        ENDDO
        ENDDO
      ENDIF

! trial step for subspace-matrix

      IF (IROT==2 .OR. IROT==3 .OR. IROT==4) THEN
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)
#ifdef testrot
       CEIDB(:,:)=CHF(:,:,NK,ISP)
#endif
        IF (ILOEW==1) THEN
           CALL ROT1(NB_TOT,CHF(1,1,NK,ISP),BSTEP,CUNI,CEIDB,COVL)
        ELSE IF (ILOEW==2) THEN
           CALL ROT2(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),BSTEP,CUNI,CEIDB,COVL)
        ENDIF

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
! strange why was this documented out
! must check whether ILOEW=2 can swap bands (if not one might comment this call out)
#ifndef restore_W
         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
         CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
              W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
              WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
              W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
#endif
      ENDDO
      ENDDO
      ELSE IF (IROT==6 .OR. IROT==7) THEN

      CALL MRG_AUX(WDES,W)

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)

        CALL ROTETA(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),W_F%CELTOT(:,NK,ISP),BSTEP,CUNI,CEIDB,COVL,W%AUXTOT(:,NK,ISP))

         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
         CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
              W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
              WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
              W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))

#ifndef restore_W
         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
         CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
              W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
              WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
              W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
         IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
#endif
      ENDDO
      ENDDO
      ENDIF

      CALL ORTHCH(W%WDES,W, LOVERL, LMDIM,CQIJ)
!
! fermi-weights
! update fermi-weights W%FERTOT

      IF (KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         DO N =1,W%WDES%NB_TOTK(NK,ISP)
            W_F%CELTOT(N,NK,ISP)=W_F%CELTOT(N,NK,ISP)-BSTEP*W_F%FERTOT(N,NK,ISP)
         ENDDO
         ENDDO
         ENDDO
      ENDIF
!-----------------------------------------------------------------------
! set IFLAG and return to main routine
! DESUM1 is the first order energy change
!-----------------------------------------------------------------------
      IF (IALGO<=6 .OR. IALGO==9) THEN
        IFLAG=0
      ELSE
        IFLAG=1
      ENDIF
      IF (IFLAG==0) DESUM1 =-DESUM0*BSTEP
      GOTO 1000
!***********************************************************************
! CORRECTOR STEP
! IFLAG=2  goto the minimum
! ) calculate the remaining distance we have to go
! ) and step to the minimum
! IFLAG>2  increase trial step
!***********************************************************************
   ELSE IF (IFLAG>=2) THEN main
#ifdef MPI
      IF ((LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
      END IF
#endif
      ! redistribute data first
      IF (NCPU /= 1) THEN

         DO ISP=1,WDES%ISPIN
            DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            CALL SETWDES(WDES,WDES1,NK)            
            CALL REDIS_PW(WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
            ENDDO
         ENDDO
      
      W%OVER_BAND=.TRUE.
      ENDIF

! store step and energy for later use
      NCOUNT=NCOUNT+1
      STEP(NCOUNT)  =BSTEP
      ENRGY(NCOUNT)=TOTEN2

! calculate position of minimum using a quadratic interpolation
      A1= TOTEN2-TOTEN
      A2=-DESUM0*BSTEP

      IF (IDUMP>=2 .AND. IO%IU0>=0) WRITE(IO%IU0,11,ADVANCE='NO') BSTEP,A1,A2
   11 FORMAT('step=',F10.5,' , ',F13.8,F13.8)

      BOPT  = -A2/(A1-A2)/2
      DESUM1= (A2+(A1-A2)*BOPT)*BOPT
!-----------------------------------------------------------------------
! IFLAG==2
!-----------------------------------------------------------------------
      IF (IFLAG==2) THEN

!---- first trial step, search near minimum
        IF (NCOUNT==1) THEN
           IF (IDUMP>=1 .AND.IO%IU0>=0 ) &
           WRITE(IO%IU0,10) GAMMA,BTRIAL,BOPT*BSTEP,AVERAG
   10 FORMAT(' gam=',F6.3,' trial=',F6.3,'  step=',F6.3,' mean=',F6.3)

! INTERATE_PREC requires at least 3 energy calculations
!    and seeks the minimum from these 3 energies
! apply this if BOPT gets unreasonably large (indicating 
! non-quadratic behaviour or some other trouble)
           IF (.NOT. ITERATE_PREC .AND. BOPT<5) THEN
              IFLAG=0
           ELSE
              ! maximum step size is 5 times the trial step
              ! there we will have data points at 4, 8 and 12 times the original step size
              BOPT=MIN(BOPT,5._q)
           ENDIF

          ! the additional steps are STRIAL1-STRIAL2 and STRIAL1 and STRIAL1+STRIAL2          
          STRIAL1=BSTEP*BOPT
          STRIAL2=BSTEP*BOPT*0.3_q

          IF (IFLAG==0) STRIAL2=0
          IF (BOPT<0)  THEN
             IF (IO%IU0>=0) &
             WRITE(IO%IU0,*)'BOPT < 0, no correction made'
             GOTO 1000
          ELSE
             BCSTEP=(STRIAL1-STRIAL2)-BSTEP
             BSTEP =(STRIAL1-STRIAL2)
          ENDIF
!---- second trial step, increase BSTEP by STRIAL2

        ELSE IF (NCOUNT==2 .OR. NCOUNT==3) THEN
           BCSTEP=STRIAL2
           BSTEP =BSTEP+STRIAL2

!---- linear approximation between two search steps near minimum

        ELSE IF (NCOUNT==4) THEN
! abort loop
           IFLAG=0
           BOPT= STRIAL1+STRIAL2*(ENRGY(2)- ENRGY(4)) &
     &         / (2*ENRGY(2)-4*ENRGY(3)+2*ENRGY(4))
           A1  = (-ENRGY(2)+ENRGY(4))/2/STRIAL2
           A2  = ( ENRGY(2)-2*ENRGY(3)+ENRGY(4))/2/STRIAL2**2
           DESUM1=A1*(BOPT-STRIAL1)+A2*(BOPT-STRIAL1)**2
           ! never step much outside the considered range of values
           BOPT=MAX(BSTEP,MIN(STRIAL1+STRIAL2,BOPT))

           IF (IO%IU0>=0) &
           WRITE(IO%IU0,10) GAMMA,BSTEP,BOPT,AVERAG
           BCSTEP=BOPT-BSTEP
           BSTEP =BOPT
        ENDIF
!-----------------------------------------------------------------------
! IFLAG>2
!-----------------------------------------------------------------------
      ELSE
!---- increase trial step by BTRIAL
        BCSTEP=BTRIAL
        BSTEP =BSTEP+BTRIAL
      ENDIF
      AVERAG=AVERAG*0.95_q+BSTEP*0.05_q

      NK=1
!-----------------------------------------------------------------------
! goto minimum (or continue trial step)
!-----------------------------------------------------------------------
      IF (IROT==1 .OR. IROT>=3) THEN

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      DO N=1,WDES%NBANDS
        DO M=1,WDES%NRPLWV
#ifdef restore_W
          W%CPTWFP(M,N,NK,ISP)=W_G%CW(M,N,NK,ISP)   -W_F%CW(M,N,NK,ISP)      *BSTEP
#else
          W%CPTWFP(M,N,NK,ISP)=W%CW(M,N,NK,ISP)   -W_F%CW(M,N,NK,ISP)      *BCSTEP
#endif
        ENDDO
        DO NPRO_=1,WDES%NPROD
#ifdef restore_W
          W%CPROJ(NPRO_,N,NK,ISP)=W_G%CPROJ(NPRO_,N,NK,ISP)-W_F%CPROJ(NPRO_,N,NK,ISP)*BSTEP
#else
          W%CPROJ(NPRO_,N,NK,ISP)=W%CPROJ(NPRO_,N,NK,ISP)-W_F%CPROJ(NPRO_,N,NK,ISP)*BCSTEP
#endif
        ENDDO

      ENDDO
      ENDDO
      ENDDO

      ENDIF

! step for subpace-matrix

      IF (IROT==2 .OR. IROT==3 .OR. IROT==4) THEN
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)
#ifdef testrot
       CEIDB(:,:)=CHF(:,:,NK,ISP)
#endif
        IF (ILOEW==1) THEN
#ifdef restore_W
          CALL ROT1(NB_TOT,CHF(1,1,NK,ISP),BSTEP,CUNI,CEIDB,COVL)
#else
          CALL ROT1(NB_TOT,CHF(1,1,NK,ISP),BCSTEP,CUNI,CEIDB,COVL)
#endif
        ELSE IF (ILOEW==2) THEN
#ifdef restore_W
          CALL ROT2(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),BSTEP,CUNI,CEIDB,COVL)
#else
          CALL ROT2(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),BCSTEP,CUNI,CEIDB,COVL)
#endif
        ENDIF

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
      ENDDO
      ENDDO

      ELSE IF (IROT==6 .OR. IROT==7) THEN

      CALL MRG_AUX(WDES,W)

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        CALL SETWDES(WDES,WDES1,NK)
         ! CHF   on entry: search direction for the pseudo Hamiltonian
         !       on exit:  search direction for the pseudo Hamiltonian U+ CHF U
         ! BSTEP step size
         ! CEIDB unitary rotation matrix U
         ! W_F%CELTOT on entry: previous pseudo Hamiltonian (diagonal eigenvalues)
         !            on exit:  new diagonal components of pseudo Hamiltonian 
        CALL ROTETA(WDES%COMM_KIN,NB_TOT,CHF(1,1,NK,ISP),W_F%CELTOT(:,NK,ISP),BCSTEP,CUNI,CEIDB,COVL,W%AUXTOT(:,NK,ISP))

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))

        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
        CALL LINCOM('F',W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP),CEIDB(1,1), &
             W%WDES%NB_TOTK(NK,ISP),W%WDES%NB_TOTK(NK,ISP), &
             WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
             W_F%CW(:,:,NK,ISP),W_F%CPROJ(:,:,NK,ISP))
        IF (WDES%DO_REDIS) CALL REDIS_PROJ(WDES1, NBANDS, W_F%CPROJ(1,1,NK,ISP))
      ENDDO
      ENDDO
      ENDIF

      CALL ORTHCH(W%WDES,W, LOVERL, LMDIM,CQIJ)

! step for fermi-weights

      IF (KPOINTS%SIGMA>0 .AND. ICEL == 1) THEN
        DO ISP=1,WDES%ISPIN
        DO NK=1,WDES%NKPTS
#ifdef MPI
        IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
        DO N =1,W%WDES%NB_TOTK(NK,ISP)
            W_F%CELTOT(N,NK,ISP)=W_F%CELTOT(N,NK,ISP)-BCSTEP*W_F%FERTOT(N,NK,ISP)
        ENDDO
        ENDDO
        ENDDO
      ENDIF

!-----------------------------------------------------------------------
   ENDIF main

1000 CONTINUE
   CALL REDIS_PW_OVER_BANDS(WDES, W)

   DO I=1,NSIM_LOCAL
      CALL DELWAV(W1(I), .TRUE.)
   ENDDO
   CALL DELWAV(WSEARCH, .TRUE.)
   DEALLOCATE(COVL, CUNI, CEIDB)

#ifdef MPI
   IF ((LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
      CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
      CALL MRG_FERWE(WDES,W)
   END IF
#endif   

   PROFILING_STOP('edwav')

   RETURN
 END SUBROUTINE EDWAV

!************************ SUBROUTINE ROTDIA ****************************
!
! this subroutine calculates the rotation matrix  for one k-point
! the commutator CHF=[F,H] must be supplied by the calling routine
! on return the rotation matrix is stored in   CEIDB
! the rotation matrix is
!    -i d [F,H]       -i d E +                     +
!   e            = U e      U    with [F,H] = U E U
!
! i.e. U is the matrix which diagonolizes [F,H] and E is the diagonal
!      matrix containing the eigenvalues
!  routine is no longer used
!***********************************************************************

      SUBROUTINE ROTDIA(NBANDS,NK,CHF,BSTEP,W,NW,CUNI,CEIDB,CTMP)
      USE prec

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      DIMENSION CHF(NBANDS,NBANDS)
      DIMENSION CTMP(NBANDS,NBANDS),CUNI(NBANDS,NBANDS), &
     &          CEIDB(NBANDS,NBANDS)

      DIMENSION HFEIG(NBANDS),W(NW)

!=======================================================================
! Diagononalize the resulting Matrix (search direction)
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
        CUNI(N1,N2)=CHF(N1,N2)
      ENDDO
      ENDDO

      CALL ZHEEV('V','U',NBANDS,CUNI,NBANDS,HFEIG,CTMP,NBANDS*NBANDS, &
     &         W,  IFAIL)
      IF (IFAIL/=0) THEN
         WRITE(*,*) 'ERROR ROTDIA: Call to routine ZHEEV failed! '// &
     &              'Error code was ',IFAIL
         STOP
      ENDIF
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CTMP(N1,N2)=CUNI(N1,N2)*EXP(-BSTEP*(0._q,1._q)*HFEIG(N2))
      ENDDO
      ENDDO
      CALL ZGEMM( 'N', 'C', NBANDS, NBANDS, NBANDS, (1.,0.), CTMP, &
     &             NBANDS, CUNI, NBANDS, (0._q,0._q), CEIDB, NBANDS)
!=======================================================================
! rotate the wavefunction (with CEIDB): warning about LINCOM! We must
! take the transposed of CEIDB because MATMUL uses the transposed ...
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CTMP(N1,N2)=CEIDB(N2,N1)
      ENDDO
      ENDDO

      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CEIDB(N2,N1)=CTMP(N2,N1)
      ENDDO
      ENDDO
      RETURN
      END SUBROUTINE


!************************ SUBROUTINE ROTINI    *************************
!
! this subroutine calculates a preconditioned rotation matrix CHF
! the unconditioned rotation matrix is given by
!
!   H(i,j) <- H(i,j) / epsilon(i)-epsilon(j)
! 
!   W               current wavefunctions and eigenvalues
!   W_F%CELEN       from W_F%CELEN the occupancies are calculated
!   CHAM on entry:  Hamiltonian between all states
!        on exit:   first order rotation matrix 
!   
!   CHF  on entry:  previous rotation matrix
!        on exit:   previous rotation matrix with some elements possibly
!                   zero (no conjugation of those elements)
!   CHAMP on entry: preconditioned Hamilton matrix
!   ORT2            orthogonality of current gradient to previous rotation
!                   matrix
!   DE2             expected energy change along present rotation matrix
!   WEIMIN          threshhold for empty bands
!   CEIDB           work array
!   SCALE           scale step length by supplied factor
!                   if CHAMP is supplied SCALE is required
!
!***********************************************************************

      SUBROUTINE ROTINI(WDES, W, W_F, CHAM, CHF, CEIDB, WEIMIN, ORT2, DE2,  & 
           LCHCON, CHAMP, SCALE)
      USE prec
      USE wave
      USE constant
      USE poscar
      USE pseudo
      USE mgrid
      USE lattice
      USE nonl_high
      USE augfast
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W, W_F

      GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN) 
      GDEF CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
      GDEF CEIDB(WDES%NB_TOT,WDES%NB_TOT)
      GDEF, OPTIONAL :: CHAMP(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN) 
      REAL(q), OPTIONAL :: SCALE

      REAL(q) WEIMIN
      REAL(q) ORT2, DE2

      LOGICAL        :: LCHCON   ! charge updated
! local
      INTEGER ISP, NK, N1, N2
      REAL(q) WEIGHT, DIFFER, DIFCEL, THETA
      REAL(q), PARAMETER ::  DIFMAX=1E-8   ! threshhold for degeneragy
      GDEF CROT, CEXPRE
      REAL(q) :: EH
      REAL(q) :: ORT2_in,DE2_in
      ORT2_in=ORT2
      DE2_in=DE2
      ORT2=0.
      DE2=0.

!=======================================================================
! calculate charge of each band
!=======================================================================
      IF (.NOT. LCHCON) THEN
         EH=0.35
      ELSE
         EH=0
      ENDIF
!=======================================================================
! calculate off symmetry elements of rotation matrix
!=======================================================================
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
         CEIDB=0
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         WEIGHT =WDES%WTKPT(NK)
         
         DO N2=1,WDES%NB_TOT
            DO N1=1,WDES%NB_TOT
              DIFFER=     (W%FERTOT(N1,NK,ISP)-W%FERTOT(N2,NK,ISP))
              ! gradient
              CEXPRE=WDES%RSPIN*CHAM(N2,N1,NK,ISP)*DIFFER
              IF (PRESENT(CHAMP)) THEN
                DIFCEL= REAL( CHAMP(N2,N2,NK,ISP)-CHAMP(N1,N1,NK,ISP) ,KIND=q)
                CROT  = CHAMP(N1,N2,NK,ISP)*SCALE
              ELSE
                DIFCEL= REAL( CHAM(N2,N2,NK,ISP)-CHAM(N1,N1,NK,ISP) ,KIND=q)
                CROT  = CHAM(N1,N2,NK,ISP)
              ENDIF
              ! avoid rotation for degenerated pairs
              IF ( (ABS(DIFCEL)<DIFMAX)  &
              ! no rotation if occupancies are incorrect (very important !!!)
              ! this can lead to components with an energy increase along the search direction
              ! which the CG algorithm does not like
                .OR. (DIFCEL* REAL( W_F%CELTOT(N2,NK,ISP)-W_F%CELTOT(N1,NK,ISP) ,KIND=q) <DIFMAX)  &
              ) THEN
                CROT  =0
              ELSE
                ! EH is used to calculate a diagonal preconditioning matrix
                ! EH is propotional to 4 times the second derivative 
                ! of Hartree energy for rotation by an angle THETA
                ! for fixed potential the second derivative is 2*DIFCEL
                IF (DIFCEL>0) THEN 
                   DIFCEL=DIFCEL+2*EH*ABS(2*W%FERTOT(N1,NK,ISP)-2*W%FERTOT(N2,NK,ISP))
                ELSE
                   DIFCEL=DIFCEL-2*EH*ABS(2*W%FERTOT(N1,NK,ISP)-2*W%FERTOT(N2,NK,ISP))
                ENDIF
                ! Loewdin perturbation theory 
                ! the position of the minimum is given  by first deriv./ second deriv.
                CROT = CROT/DIFCEL
                ! smooth cutoff function for rotation matrix
                THETA=ATAN(ABS(CROT*4))/4
                CROT=SIN(THETA)*CROT/MAX(ABS(CROT),1E-8_q)
             ENDIF
             ! avoid conjugation of those elements which do not contribute to energy
             ! these are all elements that have similar fermi-weights
             ! since rotation in that sub-space leaves the energy invariant
             IF (ABS(DIFFER)<WEIMIN) THEN
                CHF(N1,N2,NK,ISP)=0
             ENDIF
             CEIDB(N1,N2)=CROT
             ! gradient time previous search direction
             ORT2  =ORT2  +CEXPRE*CHF(N1,N2,NK,ISP)*WEIGHT
             ! gradient times present precond. gradient
             DE2   =DE2   +CEXPRE*CROT*WEIGHT
           ENDDO
         ENDDO

         DO N2=1,WDES%NB_TOT
            DO N1=1,WDES%NB_TOT
               CHAM(N1,N2,NK,ISP)=CEIDB(N1,N2)
            ENDDO
         ENDDO
!         CALL DUMP_HAM( "rotation matrix",WDES, CHAM(1,1,NK,ISP))

      ENDDO
      ENDDO

      CALLMPI (M_sum_d(WDES%COMM_KINTER,ORT2,1) )
      CALLMPI (M_sum_d(WDES%COMM_KINTER,DE2,1) )
      ORT2=ORT2+ORT2_in
      DE2=DE2+DE2_in

      END SUBROUTINE ROTINI

!************************ SUBROUTINE ROT1  ****************************
!
! this subroutine calculates a "Jacobi like" rotation matrix
!  CEIDB = 1 - CHF* BSTEP
! for one k-point, where CHF is the Loewdin-rotation matrix
! which must be supplied by the calling routine
!  (i.e. H(n1,n2) /  (H(n1,n1)-H(n2,n2))
! for a larger step CHF is not unitary
! a better approximation is obtained by diagonalisation of each 2x2 subblock
!  H (n1, n1)             H(n1, n2)
!  H (n2, n1)             H(n2, n2)               
! which yields sin(2 atan ( H(n1,n2) / 2(H(n1,n1)-H(n2,n2))))
! for the off symmetry elements of the rotation matrix
!***********************************************************************

      SUBROUTINE ROT1   (NBANDS,CHF,BSTEP,CUNI,CEIDB,CTMP)
      USE prec

      IMPLICIT NONE
      INTEGER NBANDS
      GDEF CHF(NBANDS,NBANDS)
      GDEF CTMP(NBANDS,NBANDS),CUNI(NBANDS,NBANDS), &
     &     CEIDB(NBANDS,NBANDS)
      REAL(q) BSTEP
   ! local
      INTEGER N1,N2
      REAL(q) HFEIG(NBANDS),THETA
      COMPLEX(q) CH

      DO N1=1,NBANDS
         DO N2=1,NBANDS
            CH= CHF(N1,N2)
            THETA=.5_q*ATAN(2*ABS(CH)*(-BSTEP))
! store in the  transposed of CEIDB because MATMUL uses the transposed ...
            IF (ABS(CH)/=0) THEN
               CEIDB(N2,N1)=SIN(THETA)*CH/MAX(ABS(CH),1E-8_q)
            ELSE
               CEIDB(N2,N1)=0
            ENDIF
         ENDDO
      ENDDO

      DO N1=1,NBANDS
         CEIDB(N1,N1)=1
      ENDDO

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE ROT2   ****************************
!
! this subroutine calculates the rotation matrix
!    - CHF d          - E d  +                   +
!   e            = U e      U    with CHF = U E U
!
! for one k-point, the Loewdin-perturbation CHF matrix must be supplied
! the resulting matrix is unitary
!
! i.e. U is the matrix which diagonolizes CHF   and E is the diagonal
!      matrix containing the eigenvalues
! up to first order the result from ROT1 and ROT2 are identical
!
!***********************************************************************

      SUBROUTINE ROT2(COMM, NBANDS,CHF,BSTEP,CUNI,CEIDB,CTMP)
      USE mpimy
      USE prec
      USE dfast
      USE scala

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE(communic) :: COMM
      GDEF CHF(NBANDS,NBANDS)
      GDEF CTMP(NBANDS,NBANDS),CUNI(NBANDS,NBANDS), &
     &     CEIDB(NBANDS,NBANDS)

      DIMENSION HFEIG(NBANDS),W(3*NBANDS)

      TYPE (parallel_gemm), POINTER ::  PGEMM_HANDLE
      
!      CALL SETUP_PARALLEL_GEMM(COMM, NBANDS, PGEMM_HANDLE)
!=======================================================================
! Diagononalize the resulting Matrix (search direction)
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
        CUNI(N1,N2)=CHF(N1,N2)
      ENDDO
      ENDDO


#if defined(MPI)
      ! scaLAPACK if available in parallel version
      ! unfortunately the matrix seems to be unsuitable for scaLAPACK
      ! since it can be very close to 0
      ! so avoid calling scaLAPACK routines
      IF ( LscaLAPACK .AND. .FALSE. ) THEN
         CALL pDSSYEX_ZHEEVX(COMM, CUNI, HFEIG, NBANDS , NBANDS)
         CALLMPI( M_sum_g(COMM,CUNI, NBANDS*NBANDS))
      ELSE
#endif
#ifdef  gammareal
         CALL DSYEV &
              ('V','U',NBANDS,CUNI,NBANDS,HFEIG,CTMP,NBANDS*NBANDS, &
              IFAIL)
#else
         CALL ZHEEV &
              ('V','U',NBANDS,CUNI,NBANDS,HFEIG,CTMP,NBANDS*NBANDS, &
              W,  IFAIL)
#endif
         IF (IFAIL/=0) THEN
            WRITE(*,*) 'ERROR ROTDIA: Call to routine ZHEEV failed! '// &
                 &              'Error code was ',IFAIL
            STOP
         ENDIF
#if defined(MPI)
      ENDIF
#endif
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CTMP(N1,N2)=CUNI(N1,N2)*EXP(-BSTEP*HFEIG(N2))
      ENDDO
      ENDDO

! currently there is an issue since PARALLEL_GGEMM does not work
! if second matrix is transposed
!      CALL PARALLEL_GGEMM(PGEMM_HANDLE, 'N',trans, NBANDS, NBANDS, NBANDS, one, CTMP, &
!     &             NBANDS, CUNI, NBANDS, zero, CEIDB, NBANDS)
      CALL GGEMM('N',trans, NBANDS, NBANDS, NBANDS, one, CTMP, &
     &             NBANDS, CUNI, NBANDS, zero, CEIDB, NBANDS)
!=======================================================================
! rotate the wavefunction (with CEIDB): warning about MATMUL! We must
! take the transposed of CEIDB because MATMUL uses the transposed ...
!=======================================================================
      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CTMP(N1,N2)=CEIDB(N2,N1)
      ENDDO
      ENDDO
      DO N1=1,NBANDS
      DO N2=1,NBANDS
         CEIDB(N2,N1)=CTMP(N2,N1)
      ENDDO
      ENDDO

!      CALL RELEASE_PARALLEL_GEMM(PGEMM_HANDLE)

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE ETAINI    *************************
!
! this subroutine calculates the preconditioned step for the pseudo
! Hamiltonian (delta eta) from CHAM
! see FBN: Freysoldt, Boeck, Neugebauer, Phys. Rev. B 79, 241103 (2009)
! note that VASP stores search direction respectively the negative 
! gradient
!
!   W%CELTOT        diagonal components of <phi_n | H | phi_n > 
!                   most likely identical to diagonal of CHAM
!   W_F%CELTOT      diagonal components of current pseudo Hamiltonian
!                   (epsilon in FBN)
!   CHAM on entry:  Hamiltonian matrix <phi_n | H | phi_m >
!        on exit:   search direction for pseudo Hamiltonian eta
!   CHF  on entry:  previous search direction for pseudo Hamiltonian eta
!        on exit:   previous search direction for pseudo Hamiltonian eta
!
!   CHAMP on entry: preconditioned Hamilton matrix (optional)
!   ORT2            orthogonality of current gradient to previous rotation
!                   matrix
!   DE2             expected energy change along present rotation matrix
!   WEIMIN          threshhold for empty bands
!   CEIDB           work array
!
!***********************************************************************

      SUBROUTINE ETAINI(WDES, W, W_F, CHAM, CHF, CEIDB, KPOINTS, EFERMI, WEIMIN, & 
           KAPPA, ORT2, DE2, ORTCEL, DECEL, &
           LCHCON, CHAMP)
      USE prec
      USE wave
      USE constant
      USE poscar
      USE pseudo
      USE mgrid
      USE lattice
      USE nonl_high
      USE augfast
      USE mkpoints
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W, W_F

      GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN) 
      GDEF CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
      GDEF CEIDB(WDES%NB_TOT,WDES%NB_TOT)
      GDEF, OPTIONAL :: CHAMP(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN) 
      TYPE (kpoints_struct) KPOINTS
      REAL(q)        :: EFERMI   ! chemical potential, Fermi energy
      REAL(q)        :: KAPPA    ! scaling factor
      REAL(q) WEIMIN
      REAL(q) ORT2, DE2
      REAL(q) ORTCEL, DECEL

      LOGICAL        :: LCHCON   ! charge updated
! local
      INTEGER ISP, NK, N, N1, N2, NC
      REAL(q) WEIGHT, DIFFER, DIFCEL, THETA
      REAL(q), PARAMETER ::  DIFMAX=1E-8   ! threshhold for degeneragy
      GDEF CROT, CGRAD
      REAL(q) :: EH, WSUM, DFUN, SFUN, X, EFERMI_SHIFT, WMIN, DER
      REAL(q) :: ORT2_in,DE2_in,DECEL_in,ORTCEL_in

!=======================================================================
! "gradient" for diagonal components of the pseudo Hamiltonian
! store result in the diagonal of CHAM
! these are identical to the gradient for the eigenvalues in
! the old algorithm
!=======================================================================
      EFERMI_SHIFT=0
      WSUM =0

!      depsilon_Fermi =  
!     [\sum_n df_n/d(epsilon_n-epsilon_Fermi) x d epsilon_n ]/ [\sum_n df_n/d(epsilon_n-epsilon_Fermi)]
      DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,W%WDES%NB_TOTK(NK,ISP)
               X=  (EFERMI- REAL( W_F%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               WSUM =WSUM +DFUN/KPOINTS%SIGMA* WEIGHT
               EFERMI_SHIFT=EFERMI_SHIFT+DFUN*(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))/KPOINTS%SIGMA*WEIGHT
            ENDDO
         ENDDO
      ENDDO

      CALLMPI(M_sum_d(WDES%COMM_KINTER,WSUM,1))
      CALLMPI(M_sum_d(WDES%COMM_KINTER,EFERMI_SHIFT,1))

      ! shift is the first order change in the Fermi-energy
      IF (ABS(WSUM)>1E-10) THEN
         EFERMI_SHIFT=EFERMI_SHIFT/WSUM
      ELSE
         EFERMI_SHIFT=0
      ENDIF
      WMIN=WSUM/WDES%NB_TOT*WEIMIN

      DECEL_in=DECEL
      ORTCEL_in=ORTCEL
      DECEL=0.
      ORTCEL=0.

      DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,W%WDES%NB_TOTK(NK,ISP)
               X=  (EFERMI- REAL( W_F%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               DER = -DFUN/KPOINTS%SIGMA

               ! avoid conjugation of those elements which do not contribute to energy
               IF (ABS(DFUN)<WMIN) THEN
                   CHF(N,N,NK,ISP)=0
               ENDIF

               DFUN= DFUN*((W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP)-EFERMI_SHIFT)/KPOINTS%SIGMA)

               IF (PRESENT(CHAMP)) THEN
                  SFUN= (W_F%CELTOT(N,NK,ISP)-CHAMP(N,N,NK,ISP))*KAPPA
               ELSE
                  ! preconditioned gradient equal gradient
                  SFUN=(W_F%CELTOT(N,NK,ISP)-W%CELTOT(N,NK,ISP))*KAPPA
               ENDIF

               ! first order change for for move along diagonal components
               DECEL  =DECEL  +DFUN* SFUN*WEIGHT
               ORTCEL =ORTCEL +DFUN* CHF(N,N,NK,ISP)*WEIGHT
               CHAM(N,N,NK,ISP)=SFUN
            ENDDO
         ENDDO
      ENDDO

      CALLMPI(M_sum_d(WDES%COMM_KINTER,DECEL,1))
      CALLMPI(M_sum_d(WDES%COMM_KINTER,ORTCEL,1))

      DECEL=DECEL+DECEL_in
      ORTCEL=ORTCEL+ORTCEL_in

!=======================================================================
! calculate off diagonal elements of rotation matrix
!=======================================================================
      ORT2_in=ORT2
      DE2_in=DE2
      ORT2=0.
      DE2=0.

      EH=0
!     IF (.NOT. LCHCON) THEN
!        EH=0.35
!     ELSE
!        EH=0
!     ENDIF

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CEIDB=0
         WEIGHT =WDES%WTKPT(NK)
         
         NC=0
         DO N2=1,WDES%NB_TOT
            DO N1=1,WDES%NB_TOT
              IF (N1/=N2) THEN
                 NC=NC+1
                 DIFFER=     (W%FERTOT(N1,NK,ISP)-W%FERTOT(N2,NK,ISP))
                 ! gradient
                 CGRAD=WDES%RSPIN*CHAM(N2,N1,NK,ISP)*DIFFER

                 DIFCEL= REAL( W_F%CELTOT(N2,NK,ISP)-W_F%CELTOT(N1,NK,ISP) ,KIND=q)
                 IF (ABS(DIFCEL)>DIFMAX) THEN
                    CGRAD= CGRAD/DIFCEL
                 ELSE
                    CGRAD=0
                 ENDIF
                 
                 ! rotation matrix supplied by calling routine in CHAMP
                 IF (PRESENT(CHAMP)) THEN
                    CROT  = CHAMP(N1,N2,NK,ISP)
                 ELSE
                    CROT  = CHAM(N1,N2,NK,ISP)
                 ENDIF

                 ! slow down rotation if occupancies are widely different and energies close
                 ! (not applied right now since EH=0)
                 IF (ABS(DIFCEL)>DIFMAX) THEN
                    IF (DIFCEL>0) THEN 
                       CROT=CROT*(DIFCEL/(DIFCEL+2*EH*ABS(2*W%FERTOT(N1,NK,ISP)-2*W%FERTOT(N2,NK,ISP))))
                    ELSE
                       CROT=CROT*(DIFCEL/(DIFCEL-2*EH*ABS(2*W%FERTOT(N1,NK,ISP)-2*W%FERTOT(N2,NK,ISP))))
                    ENDIF
                 ENDIF
                 
                 CROT=CROT*KAPPA
                 
                 ! avoid conjugation of those elements which do not contribute to energy
                 ! these are all elements that have similar fermi-weights
                 ! since rotation in that sub-space leaves the energy invariant
                 IF (ABS(DIFFER)<WEIMIN) THEN
                    CHF(N1,N2,NK,ISP)=0
                 ENDIF
                 CEIDB(N1,N2)=-CROT
                 ! gradient times previous search direction
                 ORT2  =ORT2  -CGRAD*CHF(N1,N2,NK,ISP)*WEIGHT
                 ! gradient times present precond. gradient
                 DE2   =DE2   -CGRAD*CEIDB(N1,N2)*WEIGHT
             ENDIF
           ENDDO
         ENDDO

         ! store negative search direction
         DO N2=1,WDES%NB_TOT
            DO N1=1,WDES%NB_TOT
               IF (N1/=N2) THEN
                  CHAM(N1,N2,NK,ISP)=CEIDB(N1,N2)
               ENDIF
            ENDDO
         ENDDO
!         CALL DUMP_HAM( "rotation matrix",WDES, CHAM(1,1,NK,ISP))

      ENDDO
      ENDDO

      CALLMPI (M_sum_d(WDES%COMM_KINTER,ORT2,1) )
      CALLMPI (M_sum_d(WDES%COMM_KINTER,DE2,1) )

      ORT2=ORT2+ORT2_in
      DE2=DE2+DE2_in

    END SUBROUTINE ETAINI


!************************ SUBROUTINE ROTETA ****************************
!
! this subroutine diagonalizes the rotation matrix eta [Equ. (32) in
! see FBN: Freysoldt, Boeck, Neugebauer, Phys. Rev. B 79, 241103 (2009)]
!
!   epsilon of pseudo Hamiltonian + search direction * stepsize
!
!   CELTOT + CHF * BSTEP  = U E U+
!
! see FBN: Freysoldt, Boeck, Neugebauer, Phys. Rev. B 79, 241103 (2009)
! 
! CHF    on entry: search direction for the pseudo Hamiltonian
!        on exit:  unitarily transformed 
!                  search direction for the pseudo Hamiltonian U+ CHF U
! BSTEP  step size
! CEIDB  on exit: unitary rotation matrix U
!        this rotation must be applied to the orbitals 
! CELTOT on entry: previous pseudo Hamiltonian (diagonal eigenvalues)
!        on exit:  new diagonal components of pseudo Hamiltonian 
!                  after unitary rotation
!
! CUNI and CTMP are auxilary matrices
!
!***********************************************************************

      SUBROUTINE ROTETA(COMM,NBANDS,CHF,CELTOT,BSTEP,CUNI,CEIDB,CTMP,AUXTOT)
      USE mpimy
      USE prec
      USE scala
      USE dfast

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE(communic) :: COMM
      GDEF CHF(NBANDS,NBANDS)
      GDEF CTMP(NBANDS,NBANDS),CUNI(NBANDS,NBANDS), &
     &     CEIDB(NBANDS,NBANDS)
      COMPLEX(q) :: CELTOT(NBANDS)
      REAL(q) :: HFEIG(NBANDS)
      REAL(q),OPTIONAL :: AUXTOT(NBANDS)  ! auxiliary array that needs to be rotated
    ! local
      INTEGER N1, N2, IFAIL
      REAL(q) :: W(3*NBANDS)
      TYPE (parallel_gemm), POINTER ::  PGEMM_HANDLE
      
      CALL SETUP_PARALLEL_GEMM(COMM, NBANDS, PGEMM_HANDLE)
!=======================================================================
! diagonalize the resulting matrix (search direction)
!
! first conjugate since all Hamilton related quantities are complex
! conjugated w.r.t what VASP uses in rot.F
!  CHF(n2,n1) proto < phi_n2 | H | phi_n1 >* = < phi_n1 | H | phi_n2 >
!=======================================================================
      CHF=GCONJG(CHF)

      CUNI=0
      DO N1=1,NBANDS
         DO N2=1,NBANDS
            CUNI(N1,N2)=-CHF(N1,N2)*BSTEP
         ENDDO
         CUNI(N1,N1)=CELTOT(N1)-CHF(N1,N1)*BSTEP
      ENDDO
#if defined(MPI)
      ! scaLAPACK if available in parallel version
      IF ( LscaLAPACK ) THEN
         CALL pDSSYEX_ZHEEVX(COMM, CUNI, HFEIG, NBANDS , NBANDS)
         CALLMPI( M_sum_g(COMM,CUNI, NBANDS*NBANDS))
      ELSE
#endif
#ifdef  gammareal
         CALL DSYEV &
              ('V','U',NBANDS,CUNI,NBANDS,HFEIG,CTMP,NBANDS*NBANDS, &
              IFAIL)
#else
         CALL ZHEEV &
              ('V','U',NBANDS,CUNI,NBANDS,HFEIG,CTMP,NBANDS*NBANDS, &
              W,  IFAIL)
#endif
         IF (IFAIL/=0) THEN
            WRITE(*,*) 'ERROR ROTDIA: Call to routine ZHEEV failed! '// &
                 &              'Error code was ',IFAIL
            STOP
         ENDIF
#if defined(MPI)
      ENDIF
#endif
      ! store current (updated) eigenvalues back in CELTOT
      CELTOT(:)=HFEIG
!=======================================================================
! transform AUXTOT (conditioning step with for each orbital)
!=======================================================================
      IF (PRESENT(AUXTOT)) THEN
      ! CTMP = AUXTOT * U
         DO N1=1,NBANDS
            DO N2=1,NBANDS
               CTMP(N1,N2)=AUXTOT(N1)*CUNI(N1,N2)
            ENDDO
         ENDDO
      ! CEIDB = U+ CTMP
         CALL PARALLEL_GGEMM( PGEMM_HANDLE, trans, 'N', NBANDS, NBANDS, NBANDS, one, CUNI, &
         &             NBANDS, CTMP, NBANDS, zero, CEIDB, NBANDS)
      ! store diagonals back to AUXTOT
         DO N1=1,NBANDS
            AUXTOT(N1)=CEIDB(N1,N1)
         ENDDO
      ENDIF
!=======================================================================
! transform CHF (equ. (21) of FBN)
!=======================================================================
      ! CEIDB = U+ CHF
      CALL PARALLEL_GGEMM(PGEMM_HANDLE, trans, 'N', NBANDS, NBANDS, NBANDS, one, CUNI, &
           &             NBANDS, CHF, NBANDS, zero, CEIDB, NBANDS)

     ! CHF = (U+ CHF ) U
      CALL PARALLEL_GGEMM(PGEMM_HANDLE, 'N', 'N', NBANDS, NBANDS, NBANDS, one, CEIDB, &
           &             NBANDS, CUNI, NBANDS, zero, CHF, NBANDS)
!=======================================================================
! rotate the wavefunction (with CEIDB)
!=======================================================================
      CEIDB=CUNI
      CHF=GCONJG(CHF)

      CALL RELEASE_PARALLEL_GEMM(PGEMM_HANDLE)

      RETURN
      END SUBROUTINE

      END MODULE
