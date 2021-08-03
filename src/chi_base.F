#include "symbol.inc"

MODULE chi_base
!*********************************************************************
!
! This module implements the irreducible polarizability matrix
! and the dynamical selfenergy
!
!
! The irreducible polarizability matrix is here defined as
!
!  xi_q(r',r,w) =  \sum_1,2 u_1(r) u*_2(r) u*_1(r') u_2(r') (f_1-f_2)
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1 -e_2- i delta    -w+ e_1- e_2+ i delta
!
! (retarded or causal, all poles above real axis in W
!  this makes xi hermitian for w=0 and q=0
!  Imag xi(-w) = -Imag xi(w), xi(-w)= xi^+(w)
! using
! #define timeordered 
! this can be changed, however, resulting in non Hermitian
! polarizabilities at w=0 since Imag xi(-w) = Imag xi(w)
!
! the use of the retarded polarizability does not
! not make the results of the code incorrect, since
! xi(-w) is never stored, and whenever xi(-w) is required,
! the proper time ordered version is internally constructed xi(-w)=xi(w)
!
! Here 1 is restricted to occupied and 2 to unoccupied orbitals
! both indices include a k-point and a band index:
!   1 = k_1,n_1
!   2 = k_2,n_2
! and 
!   k_2 = k_1-q
! u is the cell periodic part of the wavefunction 
!  phi_1(r) = e ikr u_1(r)
!
! the response function describes the response of the charge rho(r) to 
! a perturbation V(r')
!
! rho(r) = xi(r',r) V(r')
!
! the indices in xi are interchanged compared to the paper 
! Equ. (8) Shishkin, Kresse, PRB 74, 035101
!
! anyhow for F90 this kind of indexing is preferable
!
! the response function observes
!
! xi_0(r',r,w) = xi_0(r',r,-w)^*
!
! i.e. the imaginary part is changes sign upon inversion of w. 
!
! note-conjugation:
! The VASP code possesses an important internal inconsistency:
! after calculating the matrix elements
!  < u_1 u_2 | W(w) | u_2 u_1> 
! the matrix elements are conjugated 
! this moves the poles below the real axis in the stored matrix
! elements such as SCREENED_TWO_ELECTRON_INTEGRALS
! 
!*********************************************************************
  USE prec
  USE fock
  USE dfast
  USE twoelectron4o
  USE mlr_optic
  USE screened_2e
  USE crpa 

  IMPLICIT NONE
  !
  ! XI_EMPTY_THRESHHOLD sets a threshhold for emtpy orbitals
  ! 0.5 implies that the bands are divided into empty and occupied bands
  ! smaller values imply that more bands are included in the sum over
  ! occupied states and unoccupied states
  ! for metals a value of 0.001 is required to get values that are
  ! converged to 1 meV
  ! maybe we need to be even stricter for very high accuracy !?
  ! 
  REAL(q), PARAMETER :: XI_EMPTY_THRESHHOLD=0.00001

  ! threshhold wether a wavevector G^2 is zero or not
  ! comparisons are done using ABS(GX**2+GY**2+GZ**2)<G2ZERO
  REAL(q), PARAMETER :: G2ZERO=1E-12_q
  !
  ! structure to store the irreducable polarizability tensor
  !
  !
  TYPE responsefunction
     INTEGER NOMEGA                      ! number of frequencies of the responsefunctions
     REAL(q)    :: SHIFT                 ! complex shift used in evaluation
     COMPLEX(q), POINTER:: COMEGA(:)     ! frequencies
     REAL(q), POINTER   :: OMEGA(:)      ! absolute frequencies
     INTEGER            :: NOMEGA_LOW    ! low frequency in global OMEGA array
     INTEGER            :: NOMEGA_HIGH   ! highest frequency in global OMEGA array
     INTEGER            :: NQ            ! q-vectors in WDES at which response function is required
     REAL(q)            :: VKPT(3)       ! q-vectors at which response function is required
#ifndef PGI16
     COMPLEX(q),POINTER, CONTIGUOUS :: RESPONSEFUN(:,:,:)  ! responsefunctions xi_q(G',G,w)
#else
     COMPLEX(q),POINTER             :: RESPONSEFUN(:,:,:)
#endif
     REAL(q),POINTER, CONTIGUOUS    :: RESPONSER  (:,:,:)  ! real valued responsefunction (LREALSTORE)
     COMPLEX(q),POINTER, CONTIGUOUS :: RESPONSEONE(:,:)    ! one center responsefunction  xi_q(kappa,nu,w)
     REAL(q),POINTER, CONTIGUOUS    :: RESPONSEONER(:,:)   ! real valued one center responsefunction (LREALSTORE)
     INTEGER            :: NP            ! number of G vectors
     INTEGER            :: NP1           ! number of complex words along first dimension
     INTEGER            :: NP2           ! number of complex words along second dimension  
     COMPLEX(q),POINTER :: WING(:,:,:)   ! q->0 wings of responsefunctions of (:,1:3,NOMEGA) divided by 1/q
                                         ! for each cartesian direction one column 
                                         ! WING(:,IDIR,NOMEGA)=RESPONSEFUN(:,1,NOMEGA)
     COMPLEX(q),POINTER :: CWING(:,:,:)  ! q->0 wings of responsefunctions of (:,1:3,NOMEGA) divided by 1/q
                                         ! for each cartesian direction one column
                                         ! CWING(:,IDIR,NOMEGA)=RESPONSEFUN(1,:,NOMEGA)
     REAL(q),POINTER    :: WINGR(:,:,:)  ! real valued wing (LREALSTORE)
     REAL(q),POINTER    :: CWINGR(:,:,:) ! real valued wing (LREALSTORE)
     COMPLEX(q),POINTER :: HEAD(:,:,:)   ! q->0 head of responsefunctions of (1:3,1:3,NOMEGA) divided by 1/q
     LOGICAL            :: LGAMMA        ! special treatment of head and wing required
     LOGICAL            :: LREAL         ! special mode using sin consine transforms
                                         ! applicable if complex to real FFTs are used
                                         ! usually equivalent to WGW%LGAMMA
     LOGICAL            :: LREALSTORE    ! special mode using real valued response function
                                         ! reduces storage requirement by a factor 2
                                         ! but works only along imaginary axis or for spectral rep
                                         ! or for w=0
     LOGICAL            :: LSHMEM        ! shmem is used for this response function (always set)
     INTEGER            :: SHMEMLOCK     ! id of semaphore for shm write operations
     INTEGER            :: SHMID         ! id of shmem
     LOGICAL            :: LLEAD         ! lead node, all operations need to be done on this node (always set)
     TYPE(communic), POINTER :: COMM_SHMEM    ! shmem communicator
     TYPE(communic), POINTER :: COMM_INTER_SHMEM  ! communicator between different shmem arrays
     ! cache structure 
     INTEGER            :: MAXCACHE      ! maximum number of cache lines
     COMPLEX(q),POINTER, CONTIGUOUS :: STRONE (:,:,:)              ! for one center terms
     COMPLEX(q),POINTER, CONTIGUOUS :: STORAGE(:,:,:) ! to be added to responsefunction (:,NCACHE_DIM,NOMEGA)
     COMPLEX(q),POINTER :: WEIGHT (:,:  ) ! weight when added to responsefunction
     INTEGER,   POINTER :: NCACHE (:    ) ! how many elements are stored in cache bin (NOMEGA)
     INTEGER,   POINTER :: STPOS  (:,:,:) ! storage postion of integral in screened two electron table
     REAL(q),   POINTER :: W_INTER(:,:,:) ! weights for interpolation of selfenergy and derivative


  END TYPE responsefunction

  INTERFACE
     SUBROUTINE SET_RESPONSER(CW_P, N1, N2, N3, CW)
       USE prec
       INTEGER N1, N2, N3
       REAL(q), POINTER :: CW_P(:,:,:)
       COMPLEX(q), TARGET  :: CW
     END SUBROUTINE SET_RESPONSER

     SUBROUTINE SET_RESPONSE_ONE(CW_P, N1, N2, CW)
       USE prec
       INTEGER N1, N2
       REAL(q), POINTER :: CW_P(:,:)
       COMPLEX(q), TARGET  :: CW
     END SUBROUTINE SET_RESPONSE_ONE
  END INTERFACE
  
  CONTAINS

!************************ SUBROUTINE ADD_XI ****************************
!
! Add the contributions from k-points NK1 and bands NB1...NB1+NSTRIP1
! to xi
!
!  xi_q(r',r,w) =  \sum_1,2 u_1(r) u*_2(r) u*_1(r') u_2(r') (f_1-f_2)
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1 -e_2- i delta    -w+ e_1- e_2+ i delta
!
! 
! using rho(r) = xi(r',r) V(r') and 
!       rho(r)= sum_G exp(iGr) rho(G)  and V(r)= sum_G exp(iGr) V(G)
!
! we obtain
!
! rho(G) = xi(G',G) V(G')
!
! with
!
! xi(G'+q,G+q,w)= RESPONSEFUN(G',G,w) =
!
!         = 1/N^2 sum_rr' e iG'r' xi_q(r',r) e -iGr (f_1-f_2)
!
!         = \sum_1,2  1/N (\sum_r  u_1(r)  u*_2(r)  e -iGr )
!                     1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')*
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1- e_2- i delta    -w+ e_1- e_2+ i delta
!
! = <u_k1-q,n2 | e-iGr | u_k1,n1> < u_k1,n1 | e iG'r' | u_k1-q,n2> f(e_2-e_1)
!
! xi(r',r,w) = \sum_q  e-i(G'+q)r' xi(G'+q,G+q,w)  e i(G+q) r
!
! Equ. (8) Shishkin, Kresse, PRB 74, 035101
!
! where G and G' have been interchanged in Xi
! 1 is restricted to occupied and 2 to unoccupied orbitals
! both indices include a k-point and a band index:
!   1 = k1,n1
!   2 = k2,n2
! and 
!   k2 = k1-q
! u is the cell periodic part of the wavefunction
! the calling routine must loop over all k-points NK1 and all bands
!
! if the derivatives of the orbitals with respect to q is supplied
! i.e. 
!    CDER_BETWEEN_STATES =  <u_m|- i d/dq_j | u_n> = - <u_m| r_j |u_n>
! than the head and the wing of the dielectric matrix are properly
! calculated as well
!
! COMMENTS on Gamma only version:
! NP is set to the number of complex data resulting from the FFT
!   if the real to complex FFT is used, the data points are interpreted
!   as the coeficient of a cosine and sine transform hence there are
! 2*NP real data points
! after the multiplication with the complex weight 1/(w+ e_1 -e_2- i delta)
! one has 2*NP complex data points (in total 4*NP real data points)
! 
!
!**********************************************************************

  SUBROUTINE ADD_XI(LMDIM, LATT_CUR, W, WDESQ, &
       H, P, ISP, ISMEAR, SIGMA, &
       NK1, NB1, NSTRIP1, NSTRIP_TOTAL, CHI, NBANDSGWLOW, NOPER, NFLOAT )
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE fock
    USE subrot_cluster
    USE crpa 
    USE c2f_interface, ONLY : LOCKSEM,UNLOCKSEM
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    ! descriptor for wavefunctions at the point NQ
    ! the conjugated charge  (u_k1n1(r') u*_k1+q n2(r'))*=u*_k1n1(r') u_k1+q n2(r')
    ! is transformed corresponding to a wavevector q
    TYPE (wavedes1) WDESQ
    TYPE (wavespin) W
    TYPE (one_center_handle), POINTER :: H
    TYPE (potcar)      P(:)
    INTEGER ISP            ! spin index
    INTEGER ISMEAR         ! smearing mode
    REAL(q) SIGMA          ! smearing width
    INTEGER NK1            ! k-point
    INTEGER NB1            ! first band index
    INTEGER NSTRIP1        ! block to be done NK,[NB1,NB1+NSTRIP1]
    INTEGER NSTRIP_TOTAL   ! maximum number of pairs done simultaneously
    TYPE (responsefunction) CHI
    INTEGER NBANDSGWLOW    ! lowest band included in response function
    INTEGER NOPER          ! number of updates
    REAL(q) NFLOAT         ! number of floating point operations
! local variables
    TYPE (wavedes1) :: WDESK1, WDESK2 
    TYPE (wavefun1) :: W2
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    COMPLEX(q) :: WEIGHT
    INTEGER ISPINOR
    INTEGER N, NK2, N2, NB2, NOMEGA, NOMEGA_LOOP, NGLB, NB1_INTO_TOT, NP, NSTRIP2, NSTRIP2_ACT, NB2_INTO_TOT
    INTEGER I, J, NK1_IN_KPOINTS_FULL_ORIG
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
!gKreal
!   REAL(q) :: TMPR(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
    COMPLEX(q), ALLOCATABLE :: GCHG(:,:,:)      ! charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:,:,:)     ! charge weighted by 1/(w-e1-e2)
    COMPLEX(q), ALLOCATABLE :: CWORK(:)         ! work array for FFT
    GDEF, ALLOCATABLE :: CRHO(:,:,:)            ! one center charge
    GDEF, ALLOCATABLE :: CRHOW(:,:,:)           ! one center charge
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    INTEGER :: ierror
    INTEGER :: N2_IND
    TYPE (wavespin) WHF
    LOGICAL :: LPHASE
    GDEF :: CDER_BETWEEN_STATE(3)

    PROFILING_START('add_xi')

    !weights need to be allocated for Friedrich's version of CRPA
    IF ( LWEIGHTED .AND. ( .NOT.ALLOCATED(CRPA_WEIGHTS) ) ) THEN
       WRITE(*,*)'internal error in VASP: ADD_XI weighting without weights does not work'
       STOP
    ENDIF 
    IF ( LDISENTANGLE .AND. ( .NOT.ALLOCATED(UCRPA) ) ) THEN
       WRITE(*,*)'internal error in VASP: ADD_XI disentanglement without UCRPA will not work'
       STOP
    ENDIF 

! use temporarily another WDES
    H=>NULL()
    WHF=W
    WHF%WDES => WDES_FOCK
    CALL CHECK_FULL_KPOINTS

    ! determined the index of k2=k1-q
    NK2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1)-CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WHF%WDES,WDESK2,NK2)

    IF (W%WDES%WTKPT(NK1)==0 .OR. W%WDES%WTKPT(NK2)==0) THEN
       IF (W%WDES%WTKPT(NK1)/=0 .OR. W%WDES%WTKPT(NK2)/=0) THEN
          WRITE(*,*)'internal error in CALCULATE_XI: one k-point weight is zero'
          STOP
       ENDIF
       PROFILING_STOP('add_xi')
       RETURN
    ENDIF

    IF (WDESQ%LGAMMA .NEQV. CHI%LREAL) THEN
       WRITE(0,*) 'chi_base.F: internal error WDESQ%LGAMMA .NEQV. CHI%LREAL'
       STOP
    ENDIF

    ! determined the index of this k-point in the original full k-point grid
    NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1),KPOINTS_FULL_ORIG)

    ! set the phase factors to k1-k2
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, KPOINTS_FULL%VKPT(:,NK1)-KPOINTS_FULL%VKPT(:,NK2))

    ! unfortunately k1-k2-q might be any reciprocal lattice vector G
    ! we would like to calculate
    !  u*_k1-q u_k1 but calculate u*_k1-q-G u_k1    (G=k1-k2-q)
    ! since u*_k1-q=  e+iGr u*_k1-q-G (mind u is the cell periodic part)
    ! we must shift the result by e+iGr 
    ! CPHASE stores the required "shift" e+iGr
    CALL SETPHASE(W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:), &
         GRIDHF,CPHASE(1),LPHASE)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=0
    DO N=1,NSTRIP1*W%WDES%NB_PAR
       NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
       IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,NK1,ISP))) THEN
          CYCLE
       ELSE
          NGLB=MAX(NGLB,N)
       END IF
    ENDDO
    NSTRIP2=MAX(NSTRIP_TOTAL/NGLB,1)

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG( SIZE(CHI%RESPONSEFUN,1),NGLB, NSTRIP2), &
         GWORK(SIZE(CHI%RESPONSEFUN,1),NGLB, NSTRIP2), &
         CWORK(MAX(GRIDHF%MPLWV,WDESQ%GRID%MPLWV)))
    IF (ASSOCIATED(H)) THEN
       IF (CHI%LREAL .AND. .NOT. CHI%LREALSTORE) THEN
          ! CRHOW needs to store complex entries but is defined as real
          ALLOCATE(CRHO(H%TOTAL_ENTRIES, NGLB, NSTRIP2),CRHOW(2*H%TOTAL_ENTRIES, NGLB, NSTRIP2))
       ELSE
          ALLOCATE(CRHO(H%TOTAL_ENTRIES, NGLB, NSTRIP2),CRHOW(H%TOTAL_ENTRIES, NGLB, NSTRIP2))
       ENDIF
    ENDIF
    CALL NEWWAV(W2, WDESK2, .TRUE.)
    CALL SETWDES(WHF%WDES, WDESK1, NK1)
    ALLOCATE(W1(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
    ENDDO
!==========================================================================
! fourier transform the occupied bands at NK1
! to real space, then gather into W1
!==========================================================================
    CALL W1_GATHER_N( WHF, NB1, NB1+NSTRIP1-1, ISP, W1, NGLB)

    NP=WDESQ%NGVECTOR

    ! loop over all unoccupied orbitals in blocks of NSTRIP2
    mband: DO N2=FIRST_EMPTY_XI_LOCAL( WHF, NK2, ISP), LAST_EMPTY_XI_LOCAL( WHF, NK2, ISP), NSTRIP2
       ! determine actual block size (avoid to go beyond NBANDS)
       NSTRIP2_ACT=MIN(WHF%WDES%NBANDS+1-N2,NSTRIP2)
       GCHG=0
       IF (ASSOCIATED(H)) CRHO=0
       ! loop over current block of unoccupied orbitals
       DO NB2=N2,N2+NSTRIP2_ACT-1
          N2_IND=NB2-N2+1
          NB2_INTO_TOT=(NB2-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
          IF (FILLED_XI_ORBITAL((WHF%FERWE( NB2, NK2, ISP))) .OR. WHF%AUX(NB2, NK2, ISP)==0) THEN
             CYCLE
          ENDIF

          CALL W1_COPY(ELEMENT(WHF, WDESK2, NB2, ISP), W2)
          CALL FFTWAV_W1(W2)
!-----------------------------------------------------------------------------
! calculate the charge u_1(r) u_2*(r) 
!-----------------------------------------------------------------------------
          ! loop over all occupied bands in the current block
          nband: DO N=1,NGLB
             ! determine index into the FERTOT (fermi-weight) array
             NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N

             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,NK1,ISP)).OR. NB1_INTO_TOT<NBANDSGWLOW) THEN
                CYCLE
             ENDIF

             IF (ASSOCIATED(H)) THEN
                CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N), W2, CWORK(1), &
                     H, CRHO(1,N,N2_IND  ), CRHOLM(1),  SIZE(CRHOLM))
             ELSE
                CALL FOCK_CHARGE_NOINT( W1(N), W2, CWORK(1), CRHOLM(1), SIZE(CRHOLM))
             ENDIF
             ! apply phase factor e^iGr if required
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), CWORK(1), CWORK(1))
!-----------------------------------------------------------------------------
! FFT of charge to the reciprocal space
!-----------------------------------------------------------------------------
             ! extract data using wave function FFT (impose cutoff at the same time)
             CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                  CWORK(1), &
                  GWORK(1,N,N2_IND  ),WDESQ%GRID,.FALSE.)
             ! divide by number of grid points and conjugate
             IF (CHI%LREAL) THEN
                ! GCHG = (\sum_r' u_1(r') u*_2(r') e-iG'r')
                ! interpreted as cosine and -sin transform
                GCHG(1:NP,N,N2_IND  )=GWORK(1:NP,N,N2_IND  )*(1.0_q/GRIDHF%NPLWV)
             ELSE
                ! GCHG = (\sum_r' u_1(r') u*_2(r') e-iG'r')*
                GCHG(1:NP,N,N2_IND  )=CONJG(GWORK(1:NP,N,N2_IND  ))*(1.0_q/GRIDHF%NPLWV)
             ENDIF
          ENDDO nband
       ENDDO
       IF (ASSOCIATED(H)) THEN
          CALL APPLY_PHASE_ONE_CENTER_NOINT(WHF%WDES, H, CRHO(1,1,1), NGLB*NSTRIP2_ACT, & 
               W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:))
          CRHO=GCONJG(CRHO)
       ENDIF

       DO NOMEGA_LOOP=1,CHI%NOMEGA
          NOMEGA=NOMEGA_LOOP
#ifdef use_shmem
          ! offset NOMEGA by CHI%COMM_SHMEM%NODE_ME to avoid writing
          ! onto the same frequency segment
          IF (CHI%LSHMEM) NOMEGA=MOD(NOMEGA_LOOP-1+CHI%COMM_SHMEM%NODE_ME-1,CHI%NOMEGA)+1
#endif
!-----------------------------------------------------------------------------
! set GWORK to GCHG times
!                      1                            1
! f(e_2-e_1)=(  ------------------------ + ----------------------- )
!               w - e_2 + e_1 - i delta   -w - e_2 + e_1 + i delta
!
! in this equation 2 has to be restricted to unoccupied and
! 1 to occupied bands, since the interchange of two band indices is
! note on the sign in the second term
! in principle both +i delta or -i delta will yield exactly the same
! result for positive frequencies (and only this is calculated)
! but: for w-> 0 a positive sign guarantees that the function is
!      real, which I find preferable (gK)
!-----------------------------------------------------------------------------
          GWORK=0
          DO N=1,NGLB
             ! determine index into the FERTOT (fermi-weight) array
             NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,NK1,ISP)).OR. NB1_INTO_TOT<NBANDSGWLOW) THEN
                CYCLE
             ENDIF

             DO NB2=N2,N2+NSTRIP2_ACT-1
                N2_IND=NB2-N2+1
                IF (FILLED_XI_ORBITAL((W%FERWE( NB2, NK2, ISP)))) CYCLE

                ! global band index into FERTOT is given by NB2_INTO_TOT
                NB2_INTO_TOT=(NB2-1)*W%WDES%NB_PAR+W%WDES%NB_LOW

                ! wrong order skip
                WEIGHT=0
                ! so what to we do about degenerate pairs
                ! let's decide for the time being to skip them as well
                IF (REAL(WHF%CELTOT(NB2_INTO_TOT, NK2, ISP)-WHF%CELTOT(NB1_INTO_TOT,NK1,ISP),q)>DEGENERACY_THRESHOLD) THEN
                   CALL SET_XI_WEIGHT( WEIGHT, REAL(WHF%CELTOT(NB1_INTO_TOT,NK1,ISP)-WHF%CELEN(NB2, NK2, ISP),q), CHI%SHIFT, CHI%COMEGA(NOMEGA), ISMEAR, SIGMA)
                   WEIGHT=WHF%WDES%RSPIN* WHF%WDES%WTKPT(NK1)* (WHF%FERTOT(NB1_INTO_TOT,NK1,ISP)-WHF%FERWE(NB2, NK2, ISP))*WEIGHT
                   WEIGHT=WEIGHT*WHF%AUX(NB2, NK2, ISP)
                ENDIF

                !used by Friedrich et al in Phys. Rev. B 83, 121101
                !removes only diagonal part of correlated polarizability
                IF ( ALLOCATED( CRPA_WEIGHTS) ) THEN
                   WEIGHT = WEIGHT*( 1._q - CRPA_WEIGHTS( NB1_INTO_TOT,NK1,ISP )*&
                        CRPA_WEIGHTS( NB2_INTO_TOT,NK2,ISP ) )
                ENDIF
                !removes disentangled states completely
                IF ( ALLOCATED( LCRPABAND)) THEN
                  IF ( LCRPABAND(ISP,NK1,NB1_INTO_TOT) .AND.  LCRPABAND(ISP,NK2,NB2_INTO_TOT) ) &
                     WEIGHT = 0 
                ENDIF 

                !remove all bands between NCRPALOW and NCRPAHIGH
                IF ( NB1_INTO_TOT >= NCRPALOW .AND. NB1_INTO_TOT <= NCRPAHIGH .AND. &
                    NB2_INTO_TOT >= NCRPALOW .AND. NB2_INTO_TOT <= NCRPAHIGH ) THEN
                   WEIGHT = 0 
                ENDIF

                IF ( WEIGHT == 0 ) CYCLE
                IF ( LSUBTRACT_CHIC ) WEIGHT = -WEIGHT

!#define new_version
#ifdef new_version
                CALL ADD_RESPONSEFUNCTION_CACHE(CHI, GCHG(:,N,N2_IND  ), WEIGHT, NOMEGA, NP )
#endif
                IF (CHI%LREALSTORE) THEN
                   ! multiply real valued vector with 2 NP data points by real weight
                   CALL DAXPY( 2*NP, WEIGHT, GCHG(1,N,N2_IND  ), 1,  GWORK(1,N,N2_IND  ), 1)
                   IF (ASSOCIATED(H)) &
                        CALL DAXPY( H%TOTAL_ENTRIES, WEIGHT, CRHO(1,N,N2_IND  ), 1,  CRHOW(1,N,N2_IND  ), 1)
                ELSE IF (CHI%LREAL) THEN
                   ! multiply real valued vector with 2 NP data points by complex weight
                   ! resulting in 2 NP complex data points
                   CALL ZDAXPY( 2*NP, WEIGHT, GCHG(1,N,N2_IND  ), 1,  GWORK(1,N,N2_IND  ), 1)

                   ! multiply CRHO with complex weight and store in CRHOW (allocated as real but twice as large)
                   IF (ASSOCIATED(H)) &
                        CALL ZDAXPY( H%TOTAL_ENTRIES, WEIGHT, CRHO(1,N,N2_IND  ), 1,  CRHOW(1,N,N2_IND  ), 1)
                ELSE
                   CALL ZAXPY( NP, WEIGHT, GCHG(1,N,N2_IND  ), 1,  GWORK(1,N,N2_IND  ), 1)
                   IF (ASSOCIATED(H)) &
                        CALL ZAXPY( H%TOTAL_ENTRIES, WEIGHT, CRHO(1,N,N2_IND  ), 1,  CRHOW(1,N,N2_IND  ), 1)
                ENDIF

                ! determine head and wing
                ! CDER_BETWEEN_STATE =  <u_k1,n1|- i d/dq_j | u_k1+q,n2>
                IF (ABS(SUM(CHI%VKPT*CHI%VKPT))<G2ZERO) THEN

                !compute correlated head and wings if necessary  
                IF (ALLOCATED(UCRPA).AND.((LCHIC.AND.LSUBTRACT_CHIC).OR.ALLOCATED(LCRPABAND)) )THEN !rotate correlated long-wave limit
                   CALL  CDER_CRPA(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG,& 
                      ISP, NB1_INTO_TOT, NB2_INTO_TOT, UCRPA, NK1)
                ELSE 
                   CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, &
                      NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB2_INTO_TOT)
                ENDIF
                   DO I=1,3
                   DO J=1,3
                      CHI%HEAD(J,I,NOMEGA)= CHI%HEAD(J,I,NOMEGA)+REAL(GCONJG(CDER_BETWEEN_STATE(J))*CDER_BETWEEN_STATE(I),q)*WEIGHT
                   ENDDO
                   ENDDO
                ! the wing stores the limit G -> 0 (second index towards 0)
                ! accumulate  1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')*  (\int -ir u_k1,n1(r) u*_k1-q,n2(r)) 
                !             1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')* [(\int  u*_k1,n1(r) d/d q u_k1+q,n2(r))*] *
                ! CDER_BETWEEN_STATE(I) holds <u*_k1,n1 | d/d(iq) u_k1+q,n2>
                ! multiply by i 
                   IF (CHI%LREAL .AND. CHI%LREALSTORE) THEN
                      ! real valued array with 2 NP entries = NP complex entries
                      DO I=1,3
                         CHI%WING(1:NP,I,NOMEGA) =CHI%WING(1:NP,I,NOMEGA) +GWORK(1:NP,N,N2_IND  )*CDER_BETWEEN_STATE(I)
                         CHI%CWING(1:NP,I,NOMEGA)=CHI%CWING(1:NP,I,NOMEGA)+GWORK(1:NP,N,N2_IND  )*CDER_BETWEEN_STATE(I)
                      ENDDO
                   ELSE IF (CHI%LREAL) THEN
                      ! complex array with 2 NP entries
                      DO I=1,3
                         CHI%WING(1:2*NP,I,NOMEGA) =CHI%WING(1:2*NP,I,NOMEGA) +GWORK(1:2*NP,N,N2_IND  )*CDER_BETWEEN_STATE(I)
                         CHI%CWING(1:2*NP,I,NOMEGA)=CHI%CWING(1:2*NP,I,NOMEGA)+GWORK(1:2*NP,N,N2_IND  )*CDER_BETWEEN_STATE(I)
                      ENDDO
                   ELSE
                      ! complex array with NP entries
                      DO I=1,3
                         CHI%WING(1:NP,I,NOMEGA) =CHI%WING(1:NP,I,NOMEGA) +GCHG(1:NP,N,N2_IND  )*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I))*WEIGHT
                         CHI%CWING(1:NP,I,NOMEGA)=CHI%CWING(1:NP,I,NOMEGA)+CONJG(GCHG(1:NP,N,N2_IND  )*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I)))*WEIGHT
                      ENDDO
                   ENDIF
                ENDIF
             ENDDO
          ENDDO
!-----------------------------------------------------------------------------
! accumulate data 
! RESPONSEFUN(G',G)   =    Xi(G',G) =     \sum_1,2 
!   [\sum_r' (u_1(r') u*_2(r') e -iG'r')* f(e_2-e_1)] \sum_r u_1(r) u*_2(r) e -iGr 
!
! = <u_k1-q,n2 | e -iGr | u_k1,n1> < u_k1,n1 | e iG'r' | u_k1-q,n2> f(e_2-e_1) 
!                                    
!-----------------------------------------------------------------------------
#ifndef new_version
#ifdef use_shmem
          IF (CHI%LSHMEM) CALL LOCKSEM( CHI%SHMEMLOCK, NOMEGA)
#endif
          IF (CHI%LREALSTORE) THEN
             ! GWORK has 2*NP real word
             ! GCHG  has 2*NP real words -> resulting matrix has 2 NP x 2 NP real entries
             CALL DGEMM('N','T', 2*NP, 2*NP, NGLB*NSTRIP2_ACT, 1.0_q, &
                  GWORK(1,1,1), 2*SIZE(GWORK,1), GCHG(1,1,1), 2*SIZE(GCHG,1), &
                  1.0_q, CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1))

             NFLOAT=NFLOAT+8._q*NGLB*NSTRIP2_ACT*NP*NP
             NOPER=NOPER+NGLB*NSTRIP2_ACT
          ELSE IF (CHI%LREAL) THEN
             ! GWORK has 2*NP complex word -> 4*NP real words
             ! GCHG  has 2*NP real words -> resulting matrix has 2 NP x 2 NP complex entries
             CALL DGEMM('N','T', 4*NP, 2*NP, NGLB*NSTRIP2_ACT, 1.0_q, &
                  GWORK(1,1,1), 2*SIZE(GWORK,1), GCHG(1,1,1), 2*SIZE(GCHG,1), &
                  1.0_q, CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1))

             NFLOAT=NFLOAT+16._q*NGLB*NSTRIP2_ACT*NP*NP
             NOPER=NOPER+NGLB*NSTRIP2_ACT
          ELSE
             CALL ZGEMM('N','C', NP, NP, NGLB*NSTRIP2_ACT, (1.0_q,0.0_q), &
                  GWORK(1,1,1), SIZE(GWORK,1), GCHG(1,1,1), SIZE(GCHG,1), &
                  (1.0_q,0.0_q), CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1))
             NFLOAT=NFLOAT+8._q*NGLB*NSTRIP2_ACT*NP*NP
             NOPER=NOPER+NGLB*NSTRIP2_ACT
          ENDIF
#ifdef use_shmem
          IF (CHI%LSHMEM) CALL UNLOCKSEM( CHI%SHMEMLOCK, NOMEGA)
#endif
          IF (ASSOCIATED(H)) THEN
             CALL GEMM_ONE_CENTER(CHI%LREALSTORE, CHI%LREAL, WHF%WDES, & 
                  H, CRHO(1,1,1), CRHOW(1,1,1), NGLB*NSTRIP2_ACT, & 
                  CHI%RESPONSEONE(:,NOMEGA), CHI%RESPONSEONER(:,NOMEGA))
          ENDIF
#endif
       ENDDO
    ENDDO mband

!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
    DEALLOCATE(CRHOLM, GCHG, GWORK, CWORK)
    IF (ASSOCIATED(H)) THEN
       DEALLOCATE(CRHO,CRHOW)
    ENDIF
    CALL DELWAV(W2,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1)

    PROFILING_STOP('add_xi')
#undef debug
  END SUBROUTINE ADD_XI

!-----------------------------------------------------------------------------
!
! helper routine performs broadening (smearing) of transition energy
! (at least if broadening is selected in the INCAR file)
!
!-----------------------------------------------------------------------------

  SUBROUTINE SET_XI_WEIGHT(WEIGHT, DECEL, SHIFT, COMEGA, ISMEAR, SIGMA)
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
    IF (ISMEAR <0 .OR. (REAL(COMEGA)==0 .AND. AIMAG(COMEGA)/=0)) THEN
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
          DO NE=-NEDOS,NEDOS
             E=DECEL+DESTEP*NE
             CALL DELSTP(ISMEAR,(E-DECEL)/SIGMA,DFUN,SFUN)
             EPSDOS=DFUN*DESTEP/SIGMA
#ifdef timeordered
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E+CMPLX(0, SHIFT,q)) &
                  +1/(-COMEGA+E+CMPLX(0, SHIFT,q)))
#else
             WEIGHT=WEIGHT+EPSDOS* &
                  (1/( COMEGA+E-CMPLX(0, SHIFT,q)) &
                  +1/(-COMEGA+E+CMPLX(0, SHIFT,q)))
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
  END SUBROUTINE SET_XI_WEIGHT


!************************ SUBROUTINE ADD_XI_SPECTRAL ******************
!
! add the contributions from k-points NK1 and bands NB1...NB1+NSTRIP1
! to the spectral function
!
! Xi(2)(G'+q,G+q,w) =  delta(w+e_1-e_2) (f(e_1) -f(e_2))
!  <u_k1-q,n2 | e-iGr | u_k1,n1> < u_k1,n1 | e iG'r' | u_k1-q,n2>
!
! =        1/N (\sum_r  u_1(r)  u*_2(r)  -iGr )
!          1/N (\sum_r' u_1(r') u*_2(r') -iG'r')*
!
! 1 is restricted to occupied and 2 to unoccupied orbitals
! both indices include a k-point and a band index:
!   1 = k1,n1
!   2 = k2,n2
! and 
!   k2 = k1-q
! 
! This is Equ. (16) Shishkin, Kresse, PRB 74, 035101
! where G and G' have been interchanged in Xi
! 
!
!**********************************************************************

  SUBROUTINE ADD_XI_SPECTRAL( LMDIM, LATT_CUR, W, WDESQ, &
       H, P, ISP, NK1, NB1, NSTRIP1,  &
       CHI, OMEGA, NBANDSGWLOW, NOPER, NFLOAT) 
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE fock
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    ! descriptor for wavefunctions at the point NQ
    ! the conjugated charge  (u_k1n1(r') u*_k1+q n2(r'))*=u*_k1n1(r') u_k1+q n2(r')
    ! is transformed corresponding to a wavevector q
    TYPE (wavedes1) WDESQ
    TYPE (one_center_handle), POINTER :: H
    TYPE (potcar)      P(:)
    INTEGER ISP            ! spin index
    INTEGER NK1            ! k-point
    INTEGER NB1            ! first band index
    INTEGER NSTRIP1        ! block to be done NK,[NB1,NB1+NSTRIP1]
    TYPE (responsefunction) CHI
    REAL(q) OMEGA(:)       ! all frequencies
    INTEGER NBANDSGWLOW    ! lowest band included in response function
    INTEGER NOPER          ! number of updates
    REAL(q) NFLOAT         ! number of floating point operations
! local variables
    TYPE (wavedes1) :: WDESK1, WDESK2
    TYPE (wavefun1) :: W2
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    REAL(q) :: WEIGHT, WEIGHT1, WEIGHT2
    INTEGER ISPINOR
    INTEGER N, NK2, NB2, NGLB, NB1_INTO_TOT, NP, NB2_INTO_TOT
    INTEGER I,J,NK1_IN_KPOINTS_FULL_ORIG
    INTEGER NOMEGA1, NOMEGA2, MOMEGA1, MOMEGA2
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
    COMPLEX(q), ALLOCATABLE :: GCHG(:)      ! charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:)     ! work array
    GDEF, ALLOCATABLE :: CRHO(:)            ! one center charge
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)          ! augmentation occupancy matrix
    INTEGER :: ierror
    TYPE (wavespin) WHF
    LOGICAL :: LPHASE
    GDEF :: CDER_BETWEEN_STATE(3)
    INTEGER :: NODE_ME, IONODE

    PROFILING_START('add_xi_spectral')

    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=W%WDES%COMM_KIN%NODE_ME
    IONODE =W%WDES%COMM_KIN%IONODE
#endif
    
    H=>NULL()
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    CALL CHECK_FULL_KPOINTS

    ! determined the index of k2= k1+q
    NK2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1)-CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WHF%WDES,WDESK2,NK2)

    IF (W%WDES%WTKPT(NK1)==0 .OR. W%WDES%WTKPT(NK2)==0) THEN
       IF (W%WDES%WTKPT(NK1)/=0 .OR. W%WDES%WTKPT(NK2)/=0) THEN
          WRITE(*,*)'internal error in ADD_XI_SPECTRAL: one k-point weight is zero'
          STOP
       ENDIF
       PROFILING_STOP('add_xi_spectral')
       RETURN
    ENDIF

    IF (WDESQ%LGAMMA .NEQV. CHI%LREAL) THEN
       WRITE(0,*) 'chi_base.F: internal error WDESQ%LGAMMA .NEQV. CHI%LREAL'
       STOP
    ENDIF

    ! determined the index of this k-point in the original full k-point grid
    NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1),KPOINTS_FULL_ORIG)

    ! set the phase factors q'=k1-k2
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, KPOINTS_FULL%VKPT(:,NK1)-KPOINTS_FULL%VKPT(:,NK2))

    ! unfortunately k1-k2-q might be any reciprocal lattice vector G
    ! we would like to calculate
    !  u*_k1-q u_k1 but calculate u*_k1-q-G u_k1    (G=k1-k2-q)
    ! since u*_k1-q=  e+iGr u*_k1-q-G (mind u is the cell periodic part)
    ! we must shift the result by e+iGr 
    ! CPHASE stores the required "shift" e+iGr
    CALL SETPHASE(W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:), &
         GRIDHF,CPHASE(1),LPHASE)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIP1*W%WDES%NB_PAR

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG(MAX(GRIDHF%MPLWV,WDESQ%GRID%MPLWV)), GWORK(GRIDHF%MPLWV))
    IF (ASSOCIATED(H)) THEN
       ALLOCATE(CRHO(H%TOTAL_ENTRIES))
    ENDIF

    CALL SETWDES(WHF%WDES, WDESK1, NK1)
    ALLOCATE(W1(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
    ENDDO
!==========================================================================
! fourier transform the occupied bands at NK1
! to real space, then gather into W1
!==========================================================================
    CALL W1_GATHER( WHF, NB1, NB1+NSTRIP1-1, ISP, W1)

    NP=WDESQ%NGVECTOR

    CALL NEWWAV(W2,WDESK2,.TRUE.)
    mband: DO NB2=FIRST_EMPTY_XI_LOCAL( WHF, NK2, ISP), LAST_EMPTY_XI_LOCAL( WHF, NK2, ISP)
       ! NB2 is only the local index of the unoccupied band
       ! the global band index into FERTOT is given by
       NB2_INTO_TOT=(NB2-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
       IF (FILLED_XI_ORBITAL((WHF%FERWE( NB2, NK2, ISP))) .OR. WHF%AUX(NB2, NK2, ISP)==0 ) THEN
          CYCLE
       ENDIF

       CALL W1_COPY(ELEMENT(WHF, WDESK2, NB2, ISP), W2)
       CALL FFTWAV_W1(W2)
!-----------------------------------------------------------------------------
! calculate the charge u_1(r) u_2*(r) 
!-----------------------------------------------------------------------------
       ! loop over all occupied bands in the current block
       DO N=1,NGLB
          ! determine index into the FERTOT (fermi-weight) array
          NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,NK1,ISP)).OR. NB1_INTO_TOT<NBANDSGWLOW) THEN
             CYCLE
          ENDIF
          ! wrong order skip
          IF (REAL(WHF%CELTOT(NB2_INTO_TOT, NK2, ISP)-WHF%CELTOT(NB1_INTO_TOT,NK1,ISP),q)<=0) CYCLE

          WEIGHT=WHF%WDES%RSPIN* WHF%WDES%WTKPT(NK1)*(WHF%FERTOT(NB1_INTO_TOT,NK1,ISP)-WHF%FERTOT(NB2_INTO_TOT,NK2,ISP))
          WEIGHT=WEIGHT*WHF%AUX(NB2, NK2, ISP)
          IF (WEIGHT==0) CYCLE

          CALL DETERMINE_SLOT(REAL(WHF%CELEN(NB2, NK2, ISP)- WHF%CELTOT(NB1_INTO_TOT,NK1,ISP),q), & 
               OMEGA, MOMEGA1, MOMEGA2, WEIGHT1, WEIGHT2)
          CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA1, NOMEGA1)
          CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA2, NOMEGA2)
          IF (NOMEGA1==-1 .AND. NOMEGA2==-1) CYCLE

          IF (ASSOCIATED(H)) THEN
             CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N), W2, GCHG(1), &
                  H, CRHO(1), CRHOLM(1),  SIZE(CRHOLM))
             CALL APPLY_PHASE_ONE_CENTER_NOINT(WHF%WDES, H, CRHO(1), 1, & 
                  W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:))
             CRHO=GCONJG(CRHO)
          ELSE
             ! CALL FOCK_CHARGE( W1(N), W2, GCHG, CRHOLM)
             CALL FOCK_CHARGE_NOINT( W1(N), W2, GCHG(1), CRHOLM(1), SIZE(CRHOLM))
          ENDIF

          ! apply phase factor e^iGr if required
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GCHG(1), GCHG(1) )
!-----------------------------------------------------------------------------
! FFT of charge the reciprocal space
!-----------------------------------------------------------------------------
          ! extract data using wave function FFT (impose cutoff at the same time)
          ! reduces the amount of data
          CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
               GCHG (1),GWORK(1),WDESQ%GRID,.FALSE.)

          ! divide by number of grid points and conjugate
          IF (CHI%LREAL) THEN
          ! GCHG = (\sum_r' u_1(r') u*_2(r') e-iG'r')*
          ! interpreted as cosine and -sin transform
             GCHG(1:NP)=GWORK(1:NP)*(1.0_q/GRIDHF%NPLWV)
          ELSE
          ! GCHG = (\sum_r' u_1(r') u*_2(r') e-iG'r')*
             GCHG(1:NP)=CONJG(GWORK(1:NP))*(1.0_q/GRIDHF%NPLWV)
          ENDIF
!-----------------------------------------------------------------------------
! set cache lines at frequency w = e_1- e_2
!         1/N (\sum_r  u_1(r)  u*_2(r)  -iGr )
!         1/N (\sum_r' u_1(r') u*_2(r') -iG'r')*
!-----------------------------------------------------------------------------
          IF (NOMEGA1>0) THEN
             CALL ADD_RESPONSEFUNCTION_CACHE(CHI, GCHG(:), CMPLX(WEIGHT1*WEIGHT,0,q), NOMEGA1, NP)
             IF (CHI%LREAL.AND. .NOT. CHI%LREALSTORE) THEN
                NFLOAT=NFLOAT+16._q*NP*NP
             ELSE
                NFLOAT=NFLOAT+8._q*NP*NP
             ENDIF
             NOPER=NOPER+1
          ENDIF

          IF (NOMEGA2>0) THEN
             CALL ADD_RESPONSEFUNCTION_CACHE(CHI, GCHG(:), CMPLX(WEIGHT2*WEIGHT,0,q), NOMEGA2, NP)
             IF (CHI%LREAL .AND. .NOT. CHI%LREALSTORE) THEN
                NFLOAT=NFLOAT+16._q*NP*NP
             ELSE
                NFLOAT=NFLOAT+8._q*NP*NP
             ENDIF
             NOPER=NOPER+1
          ENDIF

          ! determine head and wing
          ! CDER_BETWEEN_STATE =  <u_k1,n1|- i d/dq_j | u_k1+q,n2>
          IF (ABS(SUM(CHI%VKPT*CHI%VKPT))<G2ZERO) THEN
             CALL CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB2_INTO_TOT)
             IF (NOMEGA1>0) THEN
                DO I=1,3
                   DO J=1,3
                      CHI%HEAD(J,I,NOMEGA1)= CHI%HEAD(J,I,NOMEGA1)+GCONJG(CDER_BETWEEN_STATE(J))*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT1
                   ENDDO
                ENDDO
                ! the wing stores the limit G -> 0 (second index towards 0)
                ! accumulate  1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')*  (\int -ir u_k1,n1(r) u*_k1-q,n2(r)) 
                !             1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')* [(\int u*_k1,n1(r) d/d q u_k1+q,n2(r))*] *
                ! CDER_BETWEEN_STATE(I) holds <u*_k1,n1 | d/d(iq) u_k1+q,n2>
                ! multiply by i
                IF (CHI%LREALSTORE) THEN
                   ! real valued array with 2 NP entries = NP complex entries
                   DO I=1,3
                      CHI%WING(1:NP,I,NOMEGA1) =CHI%WING(1:NP,I,NOMEGA1) +GCHG(1:NP)*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT1
                      CHI%CWING(1:NP,I,NOMEGA1)=CHI%CWING(1:NP,I,NOMEGA1)+GCHG(1:NP)*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT1
                   ENDDO
                ELSE IF (CHI%LREAL) THEN
                   ! complex array with 2 NP entries
                   GWORK(1:2*NP)=0
                   CALL ZDAXPY( 2*NP, CMPLX(WEIGHT*WEIGHT1,0,q), GCHG(1), 1,  GWORK(1), 1)
                   DO I=1,3
                      CHI%WING(1:2*NP,I,NOMEGA1) =CHI%WING(1:2*NP,I,NOMEGA1) +GWORK(1:2*NP)*CDER_BETWEEN_STATE(I)
                      CHI%CWING(1:2*NP,I,NOMEGA1)=CHI%CWING(1:2*NP,I,NOMEGA1)+GWORK(1:2*NP)*CDER_BETWEEN_STATE(I)
                   ENDDO
                ELSE
                   ! complex array with NP entries
                   DO I=1,3
                      CHI%WING(1:NP,I,NOMEGA1) =CHI%WING(1:NP,I,NOMEGA1) +GCHG(1:NP)*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I))*WEIGHT*WEIGHT1
                      CHI%CWING(1:NP,I,NOMEGA1)=CHI%CWING(1:NP,I,NOMEGA1)+CONJG(GCHG(1:NP)*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I)))*WEIGHT*WEIGHT1
                   ENDDO
                ENDIF
             ENDIF
             IF (NOMEGA2>0) THEN
                DO I=1,3
                   DO J=1,3
                      CHI%HEAD(J,I,NOMEGA2)= CHI%HEAD(J,I,NOMEGA2)+GCONJG(CDER_BETWEEN_STATE(J))*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT2
                   ENDDO
                ENDDO
                IF (CHI%LREALSTORE) THEN
                   DO I=1,3
                      CHI%WING(1:NP,I,NOMEGA2) =CHI%WING(1:NP,I,NOMEGA2) +GCHG(1:NP)*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT2
                      CHI%CWING(1:NP,I,NOMEGA2)=CHI%CWING(1:NP,I,NOMEGA2)+GCHG(1:NP)*CDER_BETWEEN_STATE(I)*WEIGHT*WEIGHT2
                   ENDDO
                ELSE IF (CHI%LREAL) THEN
                   GWORK(1:2*NP)=0
                   CALL ZDAXPY( 2*NP, CMPLX(WEIGHT*WEIGHT2,0,q), GCHG(1), 1,  GWORK(1), 1)
                   DO I=1,3
                      CHI%WING(1:2*NP,I,NOMEGA2) =CHI%WING(1:2*NP,I,NOMEGA2) +GWORK(1:2*NP)*CDER_BETWEEN_STATE(I)
                      CHI%CWING(1:2*NP,I,NOMEGA2)=CHI%CWING(1:2*NP,I,NOMEGA2)+GWORK(1:2*NP)*CDER_BETWEEN_STATE(I)
                   ENDDO
                ELSE
                   DO I=1,3
                      CHI%WING(1:NP,I,NOMEGA2) =CHI%WING(1:NP,I,NOMEGA2) +GCHG(1:NP)*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I))*WEIGHT*WEIGHT2
                      CHI%CWING(1:NP,I,NOMEGA2)=CHI%CWING(1:NP,I,NOMEGA2)+CONJG(GCHG(1:NP)*(0.0_q,1.0_q)*GCONJG(CDER_BETWEEN_STATE(I)))*WEIGHT*WEIGHT2
                   ENDDO
                ENDIF
             ENDIF
          ENDIF
       ENDDO
    ENDDO mband
     CALL DELWAV(W2,.TRUE.)
    DEALLOCATE(CRHOLM, GCHG, GWORK)
    IF (ASSOCIATED(H)) THEN
       DEALLOCATE(CRHO)
    ENDIF

    DO N=1,NGLB
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1)

    PROFILING_STOP('add_xi_spectral')
#undef debug
  END SUBROUTINE ADD_XI_SPECTRAL

!***********************************************************************
!
! calculate the dynamically screened two electron matrix elements
!  < u_1 u_2 | W(w) | u_2 u_1>  = 
!  A_1,2(omega)=  
!      sum_G G' 1/N  \sum_r' u_1(r') u*_2(r') e -iG'r'
!               1/N (\sum_r  u_1(r)  u*_2(r)  e -iGr  )*
!               RESPONSEFUN(G',G,omega)
!      sum_G G' 1/N  \sum_r' psi_1(r') psi*_2(r') e -i(G'+q)r'
!               1/N (\sum_r  psi_1(r)  psi*_2(r)  e -i(G +q)r  )*
!               W(G'+q,G+q,omega)
!   1 = k1,n1
!   2 = k2,n2
! and 
!   k2 = k1 -q
! psi_k(r') =  u_k(r') e+ ikr   [ u is the cell periodic part ]
!
! this is equivalent to equation (19)  Shishkin, Kresse, PRB 74, 035101
! mind that G and G' are interchanged in RESPONSEFUN 
! compared to the publication
! 
!***********************************************************************


  SUBROUTINE SCREENED_TWO_ELECTRON_INTEGRAL( LMDIM, LATT_CUR, W, WDESQ, NQ, & 
       H, P, ISP, S2E, SCREENED_TWO_ELECTRON_INTEGRALS, &
       NK1, NB1, NSTRIP1, NSTRIP_TOTAL, CHI, NOPER, NFLOAT ) 
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE fock
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (wavedes1) WDESQ
    INTEGER NQ
    TYPE (one_center_handle), POINTER :: H
    TYPE (potcar)      P(:)
    TYPE (screened_2e_handle) S2E
    COMPLEX(qs) :: SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:,:,:)
    INTEGER ISP            ! spin index
    INTEGER NK1            ! k-point
    INTEGER NB1            ! first band index
    INTEGER NSTRIP1        ! block to be done NK,[NB1,NB1+NSTRIP1]
    INTEGER NSTRIP_TOTAL   ! maximum number of pairs done simultaneously
    TYPE (responsefunction) CHI
    INTEGER NOPER          ! number of update 
    REAL(q) NFLOAT         ! number of floating point operations
! local variables
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2 
    TYPE (wavefun1) :: W2
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    INTEGER ISPINOR
    INTEGER N, NK2, N2, NB2, NOMEGA, NGLB, NB1_INTO_TOT, NP, NSTRIP2, NSTRIP2_ACT, NB2_INTO_TOT, N2_IND
    INTEGER I, J
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
    COMPLEX(q), ALLOCATABLE :: GCHG(:,:,:)      ! charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:,:,:)     ! charge weighted by 1/(w-e1-e2)
    COMPLEX(q), ALLOCATABLE :: CWORK(:)         ! work array for FFT
    GDEF, ALLOCATABLE :: CRHO(:,:,:)            ! one center charge
    GDEF, ALLOCATABLE :: CRHOW(:,:,:)           ! one center charge
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)              ! augmentation occupancy matrix
    COMPLEX(q) :: Z
    INTEGER :: ierror
    TYPE (wavespin) WHF
    LOGICAL :: LPHASE
    COMPLEX(q), EXTERNAL ::  ZDOTC, ZDDOTC
    REAL(q), EXTERNAL :: DDOT
!==========================================================================
! initialise variables
!==========================================================================
    H=>NULL()
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    ! set the descriptor for wavefunctions at the point NQ
    ! the conjugated charge  u_k1n1(r') u*_k1-q n2(r') is transformed
    ! corresponding to the wavevector q
    
    IF (.NOT. REQUIRED_SCREENED_2E(S2E, NQ, NK1)) RETURN

    ! determined the index of k2= k1-q
    NK2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1)-CHI%VKPT(:),KPOINTS_FULL)
    IF (W%WDES%WTKPT(NK1)==0 .OR. W%WDES%WTKPT(NK2)==0) THEN
       IF (W%WDES%WTKPT(NK1)/=0 .OR. W%WDES%WTKPT(NK2)/=0) THEN
          WRITE(*,*)'internal error in SCREENED_TWO_ELECTRON_INTEGRAL: one k-point weight is zero'
          STOP
       ENDIF
       RETURN
    ENDIF
    CALL SETWDES(WHF%WDES,WDESK2,NK2)

    IF (WDESQ%LGAMMA .NEQV. CHI%LREAL) THEN
       WRITE(0,*) 'chi_base.F: internal error WDESQ%LGAMMA .NEQV. CHI%LREAL'
       STOP
    ENDIF

    CALL CHECK_FULL_KPOINTS

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIP1*W%WDES%NB_PAR
    NSTRIP2=MAX(NSTRIP_TOTAL/NGLB,1)

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG( SIZE(CHI%RESPONSEFUN,1),NGLB, NSTRIP2), &
         GWORK(SIZE(CHI%RESPONSEFUN,1),NGLB, NSTRIP2), &
         CWORK(MAX(GRIDHF%MPLWV,WDESQ%GRID%MPLWV)))
    IF (ASSOCIATED(H)) THEN
       IF (CHI%LREAL .AND. .NOT. CHI%LREALSTORE) THEN
          ! CRHOW needs to store complex entries but is defined as real
          ALLOCATE(CRHO(H%TOTAL_ENTRIES, NGLB, NSTRIP2),CRHOW(2*H%TOTAL_ENTRIES, NGLB, NSTRIP2))
       ELSE
          ALLOCATE(CRHO(H%TOTAL_ENTRIES, NGLB, NSTRIP2),CRHOW(H%TOTAL_ENTRIES, NGLB, NSTRIP2))
       ENDIF
    ENDIF
         
    CALL NEWWAV(W2, WDESK2, .TRUE.)

    CALL SETWDES(WHF%WDES, WDESK1, NK1)
    ALLOCATE(W1(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
    ENDDO
!==========================================================================
! fourier transform the occupied bands at NK1
! to real space, then gather to W1
!==========================================================================
    CALL W1_GATHER( WHF, NB1, NB1+NSTRIP1-1, ISP, W1)

    ! set the phase factors q'=k1-k2
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, KPOINTS_FULL%VKPT(:,NK1)-KPOINTS_FULL%VKPT(:,NK2))

    ! unfortunately q'=k1-k2-q might be any reciprocal lattice vector G
    ! we would like to calculate
    !  u*_k1-q u_k1 but calculate u*_k1-q-G u_k1    (G=k1-k2-q)
    ! since u*_k1-q=  e+iGr u*_k1-q-G (mind u is the cell periodic part)
    ! we must shift the result by e+iGr 
    ! CPHASE stores the required "shift" e+iGr
    CALL SETPHASE(W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:), &
         GRIDHF,CPHASE(1),LPHASE)

    NP=WDESQ%NGVECTOR

    ! loop over all unoccupied bands in blocks of NSTRIP2
    mband: DO N2=1,WHF%WDES%NBANDS,NSTRIP2
       ! determine actual block size (avoid to go beyond NBANDS)
       NSTRIP2_ACT=MIN(WHF%WDES%NBANDS+1-N2,NSTRIP2)

       ! loop over current block 
       DO NB2=N2,N2+NSTRIP2_ACT-1
          N2_IND=NB2-N2+1
          CALL W1_COPY(ELEMENT(WHF, WDESK2, NB2, ISP), W2)
          CALL FFTWAV_W1(W2)
!==========================================================================
! calculate the charge u_1(r) u_2*(r) 
!==========================================================================
          ! loop over all occupied bands in the current block
          nband: DO N=1,NGLB
             ! determine index into the FERTOT (fermi-weight) array
             NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N

             IF (ASSOCIATED(H)) THEN
                CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N), W2, CWORK(1), &
                     H, CRHO(1,N,N2_IND  ), CRHOLM(1),  SIZE(CRHOLM))
             ELSE
                ! CALL FOCK_CHARGE( W1(N), W2, GCHG(:,N,NB2-N2+1), CRHOLM)
                CALL FOCK_CHARGE_NOINT( W1(N), W2, CWORK(1), CRHOLM(1), SIZE(CRHOLM))
             ENDIF

             ! apply phase factor e^iGr if required
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), CWORK(1), CWORK(1) )
!==========================================================================
! FFT of charge to the reciprocal space
!==========================================================================
             ! extract data using wave function FFT: sum_r e -iGr 1/N
             CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                  CWORK(1), GWORK(1,N,N2_IND  ),WDESQ%GRID,.FALSE.)
             ! factor 1/N
             GCHG(1:NP,N,N2_IND  )=GWORK(1:NP,N,N2_IND  )*(1.0_q/GRIDHF%NPLWV)
          ENDDO nband
       ENDDO

       IF (ASSOCIATED(H)) THEN
          CALL APPLY_PHASE_ONE_CENTER_NOINT(WHF%WDES, H, CRHO(1,1,1), NGLB*NSTRIP2_ACT, & 
               W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:))
       ENDIF
!==========================================================================
!   GCHG(G)=  sum_G'  RESPONSEFUN(G',G)
!                1/N  \sum_r' u_1(r') u*_2(r') -iG'r'
!==========================================================================
       DO NOMEGA=1,CHI%NOMEGA
          IF (CHI%LREALSTORE) THEN
             CALL DGEMM('N','N', 2*NP, NGLB*NSTRIP2_ACT, 2*NP, 1.0_q, &
                  CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1), GCHG(1,1,1), 2*SIZE(GCHG,1), &
                  0.0_q, GWORK(1,1,1), 2*SIZE(GWORK,1) )

             NOPER=NOPER+1
             NFLOAT=NFLOAT+8._q*NP*NP
          ELSE IF (CHI%LREAL) THEN
             CALL DGEMM('N','N', 4*NP, NGLB*NSTRIP2_ACT, 2*NP, 1.0_q, &
                  CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1), GCHG(1,1,1), 2*SIZE(GCHG,1), &
                  0.0_q, GWORK(1,1,1), 2*SIZE(GWORK,1) )

             NOPER=NOPER+1
             NFLOAT=NFLOAT+16._q*NP*NP
          ELSE
             ! W_q^T u_k1  u^*_k2 with q= k1-k2
             ! generally this is how the kernel needs to be applied
             CALL ZGEMM('T','N', NP, NGLB*NSTRIP2_ACT, NP, (1.0_q,0.0_q), &
                  CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1), GCHG(1,1,1), SIZE(GCHG,1), &
                  (0.0_q,0.0_q), GWORK(1,1,1), SIZE(GWORK,1) )

             NOPER=NOPER+1
             NFLOAT=NFLOAT+8._q*NP*NP
          ENDIF
          !  sum_G   GWORK(G)*  1/N (\sum_r  u_1(r)  u*_2(r)  -iGr  )
          !  actually this is the complex conjugated of the equation given above
          !  for the real part this does not matter, but it inverts the imaginary part
          !  as elaborated in "note-conjugation"
          DO N=1,NGLB
             DO NB2=N2,N2+NSTRIP2_ACT-1
                N2_IND=NB2-N2+1

                NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
                NB2_INTO_TOT=(NB2-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
                IF (CHI%LREALSTORE) THEN
                   Z=DDOT(  2*NP, GWORK(1,N,N2_IND  ), 1, GCHG(1,N,N2_IND  ), 1)
                ELSE IF (CHI%LREAL) THEN
                   Z=ZDDOTC(2*NP, GWORK(1,N,N2_IND  ), 1, GCHG(1,N,N2_IND  ), 1)
                ELSE
                   Z=ZDOTC(   NP, GWORK(1,N,N2_IND  ), 1, GCHG(1,N,N2_IND  ), 1)
                ENDIF
                CALL ENTER_IN_SCREENED_2E( S2E, NQ, NK1, NB1_INTO_TOT, NB2, &
                     SCREENED_TWO_ELECTRON_INTEGRALS(:, :, :, :, ISP, NOMEGA), Z )
             ENDDO
          ENDDO
       ENDDO
    ENDDO mband

    DEALLOCATE(CRHOLM, GCHG, GWORK, CWORK)
    IF (ASSOCIATED(H)) THEN
       DEALLOCATE(CRHO,CRHOW)
    ENDIF

    CALL DELWAV(W2,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO

    DEALLOCATE(W1)
  END SUBROUTINE SCREENED_TWO_ELECTRON_INTEGRAL


!***********************************************************************
!
! calculate the dynamically screened two electron matrix elements
!
! ISIGN=0)
!  identical to previous routine, but
!  the routine caches the required calculations in a CHI%STORAGE
!  array and delays calculations until sufficient data have accumulated
!  this makes the coding somewhat simpler
!
! ISIGN=1 or -1)
! assumes that the spectral representation of W is given
! the difference to the previous routine is that
! the screened two electron integrals are only calculated for 
! a small energy interval around e_i - e_j
!
! for comments see above
! for the selfconsistent GW, also 
!  V_1,2(omega)=  
!      sum_G' 1/N  \sum_r' u_1(r') u*_2(r') -iG'r'
!               RESPONSEFUN(G',G,omega)
! and 
!      sum_G   1/N (\sum_r u_1(r)  u*_2(r)  -iGr  )*
! is calculated and stored
! 
!***********************************************************************


  SUBROUTINE SCREENED_TWO_ELECTRON_CACHED( LMDIM, LATT_CUR, W, WDESQ, NQ,  &
       H, P, ISP, S2E, WCACHE, SCREENED_TWO_ELECTRON_INTEGRALS, &
       NK1, NB1, NSTRIP1, CHI, OMEGA, NOPER, NFLOAT, ISIGN ) 
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE fock
    USE wave_cacher
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (wavedes1) WDESQ
    INTEGER NQ
    TYPE (one_center_handle), POINTER :: H
    TYPE (potcar)      P(:)
    TYPE (screened_2e_handle) S2E
    TYPE (wave_cache), POINTER :: WCACHE
    COMPLEX(qs) :: SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:,:,:)
    INTEGER ISP            ! spin index
    INTEGER NK1            ! k-point
    INTEGER NB1            ! first band index
    INTEGER NSTRIP1        ! block to be done NK,[NB1,NB1+NSTRIP1]
    TYPE (responsefunction) CHI
    REAL(q) OMEGA(:)       ! global frequencies
    INTEGER NOPER          ! number of updates
    REAL(q) NFLOAT         ! number of floating point operations
    INTEGER ISIGN          ! W supplied for positive or negative frequencies
! local variables
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2
    TYPE (wavefun1) :: W2
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    INTEGER ISPINOR
    INTEGER N, NK2, NB2, NOMEGA, NGLB, NB1_INTO_TOT, NP
    INTEGER :: NOMEGA1, NOMEGA2, MOMEGA1, MOMEGA2
    REAL(q) :: EDIFF
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
    COMPLEX(q), ALLOCATABLE :: GCHG(:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:)   ! temporary
    GDEF, ALLOCATABLE       :: CRHO(:)    ! one center charge density
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    INTEGER :: ierror
    TYPE (wavespin) WHF
    LOGICAL :: LPHASE, LCYCLE
    COMPLEX(q), EXTERNAL ::  ZDOTC
    REAL(q) :: W_INTER(2,2)               ! weights for interpolation of selfenergy
    INTEGER :: IERR

    PROFILING_START('screened_two_electron_cached')

!==========================================================================
! initialise variables
!==========================================================================
    H=>NULL()
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    IF (ASSOCIATED(WCACHE) .AND. ISIGN==0 .AND. CHI%NOMEGA>1) THEN
       WRITE(0,*) 'SCREENED_TWO_ELECTRON_CACHED: internal error more than two frequencies and WCACHE associated'
       STOP
    ENDIF

    IF (.NOT. REQUIRED_SCREENED_2E(S2E, NQ, NK1)) THEN
       PROFILING_STOP('screened_two_electron_cached')
       RETURN
    ENDIF

    IF (WDESQ%LGAMMA .NEQV. CHI%LREAL) THEN
       WRITE(0,*) 'SCREENED_TWO_ELECTRON_CACHED: internal error WDESQ%LGAMMA .NEQV. CHI%LREAL'
       STOP
    ENDIF

    ! determined the index of k2= k1-q
    NK2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,NK1)-CHI%VKPT(:),KPOINTS_FULL)
    IF (W%WDES%WTKPT(NK1)==0 .OR. W%WDES%WTKPT(NK2)==0) THEN
       IF (W%WDES%WTKPT(NK1)/=0 .OR. W%WDES%WTKPT(NK2)/=0) THEN
          WRITE(*,*)'internal error in SCREENED_TWO_ELECTRON_INTEGRAL_CACHED: one k-point weight is zero'
          STOP
       ENDIF
       PROFILING_STOP('screened_two_electron_cached')
       RETURN
    ENDIF
    CALL SETWDES(WHF%WDES,WDESK2,NK2)


    CALL CHECK_FULL_KPOINTS

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIP1*W%WDES%NB_PAR

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG(MAX(GRIDHF%MPLWV,WDESQ%GRID%MPLWV)), GWORK( GRIDHF%MPLWV))
    IF (ASSOCIATED(H)) THEN
       ALLOCATE(CRHO(H%TOTAL_ENTRIES))
    ENDIF
         
    CALL SETWDES(WHF%WDES, WDESK1, NK1)
    ALLOCATE(W1(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
    ENDDO
!==========================================================================
! fourier transform the occupied bands at NK1
! to real space, then gather to W1
!==========================================================================
    CALL W1_GATHER( WHF, NB1, NB1+NSTRIP1-1, ISP, W1)

    ! set the phase factors q'=k1-k2
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, KPOINTS_FULL%VKPT(:,NK1)-KPOINTS_FULL%VKPT(:,NK2))

    ! unfortunately q'=k1-k2-q might be any reciprocal lattice vector G
    ! we would like to calculate
    !  u*_k1-q u_k1 but calculate u*_k1-q-G u_k1    (G=k1-k2-q)
    ! since u*_k1-q=  e+iGr u*_k1-q-G (mind u is the cell periodic part)
    ! we must shift the result by e+iGr 
    ! CPHASE stores the required "shift" e+iGr
    ! magic number for no phase factor is zero (phase factor is always e^(i x) hence never 0
    CPHASE(1)=0
    CALL SETPHASE(W%WDES%VKPT(:,NK1)-W%WDES%VKPT(:,NK2)-CHI%VKPT(:), &
         GRIDHF,CPHASE(1),LPHASE)

    CALL STORE_PHASE( WCACHE, NK2, LPHASE, CPHASE)
       
    NP=WDESQ%NGVECTOR
    CALL NEWWAV(W2, WDESK2, .TRUE.)
    mband: DO NB2=1,WHF%WDES%NBANDS
       IF (WHF%AUX(NB2, NK2, ISP)==0 ) CYCLE  ! bands that should not be included are bypassed

       CALL W1_COPY(ELEMENT(WHF, WDESK2, NB2, ISP), W2)
       CALL FFTWAV_W1(W2)
!==========================================================================
! calculate the charge u_1(r) u_2*(r) 
!==========================================================================
       ! loop over all bands for self energy is evaluated
       nband: DO N=1,NGLB
          ! determine index into the FERTOT (fermi-weight) array
          NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
          ! attention LGWNO resets AUXTOT to save compute time 
          ! force bypass for orbitals that are not properly set using NB_TOTK and not AUXTOT
          IF (NB1_INTO_TOT > W%WDES%NB_TOTK(NK1, ISP)) CYCLE 

          EDIFF = REAL(WHF%CELTOT(NB1_INTO_TOT,NK1,ISP)-WHF%CELEN(NB2, NK2, ISP),q)
          IF (ISIGN/=0) THEN
#ifdef oldversion
          ! this version "assumes" a sharp delta like function for the Fermi-distribution

          ! NB2 filled band
             IF( W%FERWE(NB2, NK2, ISP) > 0.5) THEN
                ! filled band, NB1 above or equal NB2,  we need positive frequencies only
                !  i.e. ISIGN must be +1
                IF (EDIFF>=0  .AND. ISIGN<0) CYCLE
                ! filled band, NB1 below or equal NB2,  need negative frequencies only
                !  i.e. ISIGN must be -1
                IF (EDIFF<0 .AND. ISIGN>0) CYCLE
             ELSE
                ! empty band, NB1 above or equal NB2,  we need negative frequencies only
                !  i.e. ISIGN must be -1
                IF (EDIFF>=0 .AND. ISIGN>0) CYCLE
                ! empty band, NB1 below or equal NB2, we need positive frequencies only
                !  i.e. ISIGN must be +1
                IF (EDIFF<0 .AND. ISIGN<0) CYCLE
             ENDIF
             CALL DETERMINE_SLOT_INTER(EDIFF,OMEGA, MOMEGA1, MOMEGA2, W_INTER)
             CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA1, NOMEGA1)
             CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA2, NOMEGA2)
             IF (NOMEGA1==-1 .AND. NOMEGA2==-1) CYCLE
#else
             ! new version seems to be much more stable than old version
             LCYCLE= .TRUE.
             ! NB2 filled band (all bands that do not have strictly 0 occupancy)
             IF( ABS(W%FERWE(NB2, NK2, ISP)) > XI_EMPTY_THRESHHOLD) THEN
                !          IF( ABS(W%FERWE(NB2, NK2, ISP)) > 0.5) THEN
                ! NB2 filled band, NB1 above or equal NB2,  we need positive frequencies only
                !  i.e. ISIGN must be +1
                
                ! NB2 filled band, NB1 below or equal NB2,  need negative frequencies only
                !  i.e. ISIGN must be -1
!                IF ((EDIFF>=-1E-10_q .AND. ISIGN>0).OR. (EDIFF<-1E-10_q  .AND. ISIGN<0)) THEN
                IF ((EDIFF>=0 .AND. ISIGN>0).OR. (EDIFF<0 .AND. ISIGN<0)) THEN
                   LCYCLE=.FALSE.
                   CALL DETERMINE_SLOT_INTER_WEIGHT(EDIFF,OMEGA, MOMEGA1, MOMEGA2, W_INTER, W%FERWE(NB2, NK2, ISP) , IERR)
                   IF (IERR==1) THEN
                      WRITE(*,*) 'SCREENED_TWO_ELECTRON_CACHED involved bands',REAL(WHF%CELTOT(NB1_INTO_TOT,NK1,ISP),q) , REAL(WHF%CELEN(NB2, NK2, ISP),q), NB1_INTO_TOT,NK1,NB2, NK2,ISP
                      STOP
                   ENDIF
                ENDIF
             ENDIF
             ! NB2 empty band (all bands that do not have strictly 1 occupancy)
             IF ( ABS(1-W%FERWE(NB2, NK2, ISP)) > XI_EMPTY_THRESHHOLD) THEN
                !          IF ( W%FERWE(NB2, NK2, ISP) <= 0.5) THEN
                ! NB2 empty band, NB1 above or equal NB2,  we need negative frequencies only
                !  i.e. ISIGN must be -1
                
                ! NB2 empty band, NB1 below or equal NB2, we need positive frequencies only
                !  i.e. ISIGN must be +1
                ! IF (.NOT.(EDIFF>=0 .AND. ISIGN>0) .AND. .NOT. (EDIFF<0 .AND. ISIGN<0)) THEN
!                IF ((EDIFF<-1E-10_q .AND. ISIGN>0) .OR. (EDIFF>=-1E-10_q .AND. ISIGN<0)) THEN
                IF ((EDIFF<0 .AND. ISIGN>0) .OR. (EDIFF>=0 .AND. ISIGN<0)) THEN
                   IF (.NOT. LCYCLE) THEN
                      WRITE(*,*) 'internal error in SCREENED_TWO_ELECTRON_CACHED: double use of entries'
                      STOP
                   ENDIF
                   LCYCLE=.FALSE.
                   CALL DETERMINE_SLOT_INTER_WEIGHT(EDIFF,OMEGA, MOMEGA1, MOMEGA2, W_INTER, 1-W%FERWE(NB2, NK2, ISP) , IERR)
                   IF (IERR==1) THEN
                      WRITE(*,*) 'SCREENED_TWO_ELECTRON_CACHED involved bands',REAL(WHF%CELTOT(NB1_INTO_TOT,NK1,ISP),q) , REAL(WHF%CELEN(NB2, NK2, ISP),q), NB1_INTO_TOT,NK1,NB2, NK2,ISP
                      STOP
                   ENDIF
                ENDIF
             ENDIF
             IF (LCYCLE) CYCLE

             CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA1, NOMEGA1)
             CALL DETERMINE_LOCAL_SLOT( CHI, MOMEGA2, NOMEGA2)
             IF (NOMEGA1==-1 .AND. NOMEGA2==-1) CYCLE
#endif
          ELSE
          ! COHSEX
             W_INTER=0
             ! COH plus SEX in first entry (W-v) (0.5-f)
             W_INTER(1,1)= -W%FERWE(NB2, NK2, ISP)+0.5_q
             ! SEX in second entry
             W_INTER(1,2)= -W%FERWE(NB2, NK2, ISP)
          ENDIF

          ! get charge
          IF (ASSOCIATED(H)) THEN
             CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(N), W2, GCHG(1), &
                  H, CRHO(1), CRHOLM(1),  SIZE(CRHOLM))
          ELSE
             ! CALL FOCK_CHARGE( W1(N), W2, GCHG, CRHOLM)
             CALL FOCK_CHARGE_NOINT( W1(N), W2, GCHG(1), CRHOLM(1), SIZE(CRHOLM))
          ENDIF

          ! apply phase factor e^iGr if required
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GCHG(1), GCHG(1) )

          ! extract data using wave function FFT (impose cutoff at the same time)
!==========================================================================
!   GWORK(G')=  1/N  \sum_r' u_1(r') u*_2(r') -iG'r'
!==========================================================================
          CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
               GCHG (1),GWORK(1),WDESQ%GRID,.FALSE.)
          GWORK(1:NP)=GWORK(1:NP)*(1.0_q/GRIDHF%NPLWV)

          IF (ISIGN==0) THEN
             DO NOMEGA=1,CHI%NOMEGA
                CALL STORE_CACHER(WCACHE, NB2, NK2, ISP, W2)

                CALL ADD_RESPONSEFUNCTION_INT(WDESQ, CHI, GWORK(1:NP), NOMEGA, NP, & 
                     S2E, SCREENED_TWO_ELECTRON_INTEGRALS, WCACHE, &
                     NQ, NK1, NB1_INTO_TOT, NB2, ISP, NOMEGA, NK2 , W_INTER(NOMEGA,:))
                IF (CHI%LREAL) THEN
                   NFLOAT=NFLOAT+16._q*NP*NP
                   NOPER=NOPER+1
                ELSE
                   NFLOAT=NFLOAT+8._q*NP*NP
                   NOPER=NOPER+1
                ENDIF
             ENDDO
          ELSE
             IF (NOMEGA2>0) THEN
                CALL STORE_CACHER(WCACHE, NB2, NK2, ISP, W2)

                ! use W_INTER to determine final contributions to selfenergy
                ! and its derivative
                CALL ADD_RESPONSEFUNCTION_INT(WDESQ, CHI, GWORK(1:NP), NOMEGA2, NP, & 
                     S2E, SCREENED_TWO_ELECTRON_INTEGRALS, WCACHE, & 
                     NQ, NK1, NB1_INTO_TOT, NB2, ISP, -1, NK2 , W_INTER(2,:))
                IF (CHI%LREAL) THEN
                   NFLOAT=NFLOAT+16._q*NP*NP
                   NOPER=NOPER+1
                ELSE
                   NFLOAT=NFLOAT+8._q*NP*NP
                   NOPER=NOPER+1
                ENDIF
             ENDIF
             
             IF (NOMEGA1>0) THEN
                CALL STORE_CACHER(WCACHE, NB2, NK2, ISP, W2)

                CALL ADD_RESPONSEFUNCTION_INT(WDESQ, CHI, GWORK(1:NP), NOMEGA1, NP,  & 
                     S2E, SCREENED_TWO_ELECTRON_INTEGRALS, WCACHE, & 
                     NQ, NK1, NB1_INTO_TOT, NB2, ISP, -1, NK2 , W_INTER(1,:))
                IF (CHI%LREAL) THEN
                   NFLOAT=NFLOAT+16._q*NP*NP
                   NOPER=NOPER+1
                ELSE
                   NFLOAT=NFLOAT+8._q*NP*NP
                   NOPER=NOPER+1
                ENDIF
             ENDIF
          ENDIF

       ENDDO nband
    ENDDO mband
    CALL DELWAV(W2,.TRUE.)
    DEALLOCATE(CRHOLM, GCHG, GWORK)
    IF (ASSOCIATED(H)) THEN
       DEALLOCATE(CRHO)
    ENDIF
    DO N=1,NGLB
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO

    DEALLOCATE(W1)

    PROFILING_STOP('screened_two_electron_cached')

  END SUBROUTINE SCREENED_TWO_ELECTRON_CACHED


!************************ SUBROUTINE OEP_GW ****************************
!
! this subroutine realizes an OEP method for the RPA correlation
!
! it can use either COH-SEX or an hermitian approximation to
! the self-energy ala Schilfgaarde and Kotani
! 
! for COHSEX (select by NOMEGA = 1 in the INCAR files)
! WACC1  stores the entirely correlation contribution
! WACC2  stores the SEX contribution only (not used in fact)
! 
! for conventional scheme
! WACC1  is the action of the selfenergy operator on each band (at E_QP)
! WACC2  is the action of the derivative of the selfenergy operator 
!        on each band (at E_QP)
!
! it is also possible to use this routine for EXXOEP calculations
! but requires to manually edit a few lines
! search for WACC1%CW below
! this part has been tested several times, and works
! (do not forget the increase NBANDS to very large values)
!
!***********************************************************************

  SUBROUTINE OEP_GW( NBANDSGW_, WDES, LATT_CUR, NONLR_S, NONL_S, W, WACC1, WACC2, LEXX, &
      LMDIM, CDIJ, CQIJ, SV, T_INFO, P, STEP, LGW0, IU0, IU6, &
      ISMEAR, SIGMA, SYMM, LCOHSEX, &
      INFO, WDESQ, CHI0, GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
      CHTOT, DENCOR, CVTOT, CSTRF, IRDMAX, CRHODE, N_MIX_PAW, AMIX, RHOLM )

      
    USE prec
    USE mgrid
    USE wave_high
    USE msymmetry
    USE lattice
    USE nonl_high
    USE hamil
    USE main_mpi
    USE fock
    USE pseudo
    USE dfast
    USE mlr_optic
    USE pawm
    USE pot
    USE us
!    USE fock
    IMPLICIT NONE

    INTEGER  NBANDSGW_
    TYPE (wavedes)     WDES
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W, WACC1, WACC2
    INTEGER LMDIM                                   ! leading dimension
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! 
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! 
    RGRID   SV(DIMREAL(WDES%GRID%MPLWV),WDES%NCDIJ) ! local potential
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    REAL(q) STEP
    LOGICAL LGW0
    INTEGER IU0, IU6
    INTEGER ISMEAR                ! smearing mode
    REAL(q) SIGMA                 ! smearing width
    TYPE (symmetry)    SYMM
    LOGICAL            LCOHSEX    ! apply COHSEX
    LOGICAL            LEXX       ! use exact exchange only
    TYPE (info_struct) INFO
    TYPE (wavedes1)    WDESQ      ! descriptor for gamma point (passed from main routine)
    TYPE (responsefunction) CHI0  ! response function at Gamma point
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge density
    RGRID       DENCOR(GRIDC%RL%NP)
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
    INTEGER     IRDMAX
    OVERLAP     CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    INTEGER     N_MIX_PAW
    REAL(q)     AMIX
    REAL(q)     RHOLM(N_MIX_PAW,WDES%NCDIJ)
! local
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*WDES%NB_TOT)
!    REAL(q)    RWORK(3*WDES%NB_TOT) ! sufficient for ZHEGV
    REAL(q)    RWORK(8*WDES%NB_TOT)  ! required for ZGGEV
    COMPLEX(q) CDCHF
! local variables
    GDEF, ALLOCATABLE :: CHAM(:,:) ! subspace rotation
    GDEF, ALLOCATABLE :: COVL(:,:) ! overlap 
    GDEF, ALLOCATABLE :: CTMP(:,:) ! temporary
    INTEGER ISP, NK, NSTRIP, NSTRIP_ACT, NSTRIP_RED
    INTEGER NPOS, NPOS_RED, N, N2, NP, NB_TOT, NBANDS, NBANDSGW, IFAIL
    TYPE (wavedes1)    WDES1           ! descriptor for one k-point
    TYPE (wavefun1)    W1              ! current wavefunction
    TYPE (wavefuna)    WA              ! array to store wavefunction
    TYPE (wavefuna)    WHAM            ! array to store accelerations for a selected block
    TYPE (wavefuna)    WHAM_HARTREE_ION! array to store accelerations for a selected block
    TYPE (wavefuna)    ACC2            ! array to store overlap 
    TYPE (wavefuna)    WNONL           ! array to hold non local part D * wave function character
    TYPE (wavefuna)    WNONL_HARTREE_ION! array to hold non local part D * wave function character
    TYPE (energy)      E
    REAL(q) :: XCSIF(3,3)
    INTEGER :: IRDMAA
    OVERLAP :: CDIJ1(LMDIM,LMDIM,WDES%NIONS, WDES%NCDIJ), CDIJ_HARTREE_ION(LMDIM,LMDIM,WDES%NIONS, WDES%NCDIJ) 
    RGRID   :: SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ), SV_HARTREE_ION(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
    COMPLEX(q) :: CVTOT1(GRIDC%MPLWV,WDES%NCDIJ)
    COMPLEX(q) :: GCHG(CHI0%NP2, WDES%NCDIJ) , GCHG_(CHI0%NP2)
    COMPLEX(q) :: CHI_WORK(CHI0%NP2, CHI0%NP2, WDES%NCDIJ)
    INTEGER    :: IPIV(CHI0%NP2)
    COMPLEX(q) :: CWORK(GRID%MPLWV)
    REAL(q) :: Z(WDES%NB_TOT), SIGMAQP(WDES%NB_TOT), CELNEW(WDES%NB_TOT)
#ifdef MPI
    INTEGER :: NK2
    REAL(q) :: QP_OUT(NBANDSGW_,5)
#endif

! evaluate Hamiltonian at QP energy
! see above
    LOGICAL :: LUPDATEQP=.FALSE.
! see electron_OEP.F or comments below
    LOGICAL :: LKINETIC=.FALSE.
! the iteration count must be 1 right now, since the Hartree potential is not updated
    INTEGER IDUM, IERR ; REAL(q) :: RDUM ; COMPLEX(q) :: CDUM ; CHARACTER (LEN=1) :: CHARAC 
    REAL(q) :: DISPL(3,T_INFO%NIONS)
    REAL(q) :: ESHIFT(WDES%NCDIJ), EMAX(WDES%NCDIJ), ESHIFT_TOP(WDES%NCDIJ)
! recalculate response function locally
    LOGICAL :: LCHI0=.TRUE.

    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS
    NSTRIP=NSTRIP_STANDARD
    CALL SETWDES(WDES,WDES1,0)
    CALL NEWWAVA(WHAM, WDES1, NSTRIP)
    CALL NEWWAVA(WHAM_HARTREE_ION, WDES1, NSTRIP)
    CALL NEWWAVA_PROJ(WNONL, WDES1)
    CALL NEWWAVA_PROJ(WNONL_HARTREE_ION, WDES1)
    CALL NEWWAVA(WA, WDES1, NBANDS)

    ALLOCATE(  W1%CR(WDES%GRID%MPLWV*WDES%NRSPINORS))
    ALLOCATE( CHAM(WDES%NB_TOT, WDES%NB_TOT), & 
              COVL(WDES%NB_TOT, WDES%NB_TOT), & 
              CTMP(WDES%NB_TOT, WDES%NB_TOT))
    ! force full fock contribution
    HFSCREEN=0.0
    AEXX    =1.0

!=======================================================================
! determine local Hartee and ionic potential
!=======================================================================
    !  calculate potential
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES,  &
         INFO,P,T_INFO,E,LATT_CUR, &
         CHTOT,CSTRF,CVTOT1,DENCOR,SV_HARTREE_ION, SOFT_TO_C,XCSIF)
    
    ! add the one center augmentation related terms  
    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ_HARTREE_ION,CQIJ,CVTOT1,IRDMAA,IRDMAX)

    ! finally add one center terms
    ! including exact exchange contributions (matter of fact)
    ! (this will be used to calculate the full Hamiltonian including the selfenergy)
    LHFCALC_FORCE=.TRUE.
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ_HARTREE_ION(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
         E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    LHFCALC_FORCE=.FALSE.
!=======================================================================
! determine OEP potential
!=======================================================================
    ! determine local potential (excludes OEP potential)
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E,LATT_CUR, &
         CHTOT,CSTRF,CVTOT1, DENCOR, SV1, SOFT_TO_C,XCSIF)
    
    ! determine CDIJ1 from CVTOT1
    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)
    
    ! enforce calculation of HF one center contributions -> CDIJ1
    LHFCALC_FORCE=.TRUE.
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
         E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    LHFCALC_FORCE=.FALSE.
    
    IF (.NOT. LKINETIC)  THEN
       ! determine potential -V_OEP by calculating the difference of current local potential
       ! and ionic and Hartree contribution to local potential stored in CVTOT1, CDIJ1, and SV1
       CDIJ1 =CDIJ1-CDIJ
       CVTOT1=CVTOT1-CVTOT
       SV1   =SV1-SV
    ELSE
       ! since (T + V_OEP+ V_ion + V_H -epsilon0 S) phi=0 -> 
       ! (V^NL_x - V_OEP) phi = ((V^NL_x + T) + V_ion + V_H - epsilon0 S) phi
       ! one could also use the full original Hamiltonian
    ENDIF

!=======================================================================
! this is a helpfull code to test specific charge perturbation
! note: LKINETIC must be set to .FALSE. for this to work
#ifdef  test_specific_charge
!=======================================================================
! first determine the current local OEP potential from SV1
    ISP=1
    ! works only if WDESQ%GRID is equal GRID_FOCK
    ! so that does not work
    IF (GRID%NGX /= WDESQ%GRID%NGX .OR. GRID%NGY /= WDESQ%GRID%NGY .OR. GRID%NGZ /= WDESQ%GRID%NGZ) THEN
       WRITE(*,*) 'internal error in OEP_GW: non commensurate arrays'
       STOP
    ENDIF

    ! ok here is a problem the grid on which the potential is defined
    ! GRID_SOFT and GRID are identical so good so fine
    CALL RL_COPY_RGRID_WAVE(SV1(1,1), CWORK(1), 1.0_q, GRID)
    
    ! solution is to use a wavefunction FFT to get them in reciprocal space
    ! FFTWAV
    ! FFT back to real space using WDES_FOCK (and thus GRID_FOCK)
    ! but I need to know where the Gamma point is etc. etc.
    CALL FFTEXT(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
         CWORK(1), GCHG(1,ISP), WDESQ%GRID, .FALSE.)
    
    GCHG(:,ISP)=GCHG(:,ISP)*(1.0_q/WDESQ%GRID%NPLWV)

! impose a certain potential
!    GCHG = 0
!    GCHG(2,ISP) =(0.0,1.0)
!    GCHG(3,ISP) =(0.0,-1.0)
!    GCHG(2,ISP) =1.0
!    GCHG(3,ISP) =1.0
    NP=WDESQ%NGVECTOR
    IF (WQ%LGAMMA) NP=NP*2

    WRITE(*,*) 'This vector should be compared to result from OEP_CHARGE'
    WRITE(*,'(10F8.3)') MATMUL(TRANSPOSE(CHI0%RESPONSEFUN(1:NP,1:NP,1)), GCHG(1:NP,ISP))

! bring this potential to SV1, CVTOT1 and CDIJ1
    CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
         CWORK(1), GCHG(1,ISP), WDESQ%GRID)
    
    ! copy wave function based array (COMPLEX or REAL) to RGRID
    CALL RL_COPY_WAVE_RGRID( CWORK(1), SV1(1,1), 1.0_q, GRID)
    
    CALL FFT_RC_SCALE(SV1, CWORK, GRID_SOFT)
    
    CVTOT1=0
    CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, CWORK, CVTOT1)
    CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV1, CVTOT1)
    
    ! determine CDIJ1 from CVTOT1
    DISPL=0
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,.FALSE.,IRDMAA,IRDMAX,DISPL)
    
! finally bring the current potential to the small cutoff sphere (double check !)
! as given by ENCUTGW

    ! ok here is a problem the grid on which the potential is defined
    ! GRID_SOFT and GRID are identical so good so fine
    CALL RL_COPY_RGRID_WAVE(SV1(1,1), CWORK(1), 1.0_q, GRID)
    
    ! but the WDESQ%GRID is usually GRID_FOCK
    ! so that does not work
    IF (GRID%NGX /= WDESQ%GRID%NGX .OR. GRID%NGY /= WDESQ%GRID%NGY .OR. GRID%NGZ /= WDESQ%GRID%NGZ) THEN
       WRITE(*,*) 'internal error in OEP_GW: non commensurate arrays'
       STOP
    ENDIF
    ! solution is to use a wavefunction FFT to get them in reciprocal space
    ! FFTWAV
    ! FFT back to real space using WDES_FOCK (and thus GRID_FOCK)
    ! but I need to know where the Gamma point is etc. etc.
    CALL FFTEXT(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
         CWORK(1), GCHG(1,ISP), WDESQ%GRID, .FALSE.)
    
    GCHG=GCHG*(1.0_q/WDESQ%GRID%NPLWV)
    WRITE(*,*) 'imposing the following local test potential'
    WRITE(*,*) 'we would like to recover this potential    '
    WRITE(*,'(10F8.3)') GCHG
    
!    CALL WRT_RL_LINE(6, GRID_SOFT, SV1)
!    CALL WRT_RL_LINE(6, GRIDC, CVTOT1)
#endif
    NBANDSGW =WDES%NB_TOT

!=======================================================================
    GCHG=0
    CHI0%LGAMMA=.TRUE.
    CALLMPI( M_bcast_z(WDES%COMM_INTER, CHI0%COMEGA(1), 1))
    CALLMPI( M_bcast_z(WDES%COMM_KINTER, CHI0%COMEGA(1), 1))

    ESHIFT=0
    EMAX  =-1000
    ESHIFT_TOP=0


sp1: DO ISP=1,WDES%ISPIN

    IF (IU6>=0) WRITE(IU6,'(//A,I2)') ' QP shifts <psi_nk| G(iteration)W_0 |psi_nk>:'


    IF (ABS(CHI0%COMEGA(1)-0)/=0) THEN
       WRITE(*,*)' internal error in OEP_GW: CHI0 stores the wrong frequency', CHI0%COMEGA
       STOP
    ENDIF

    IF (LCHI0) CALL CLEAR_RESPONSE( CHI0 )

    DO NK=1,WDES%NKPTS
       ! here parallelization is over all NK
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
! construct the selfenergy matrix and its derivative
!=======================================================================
       CALL SETWDES(WDES,WDES1,NK)

       IF (NONLR_S%LREAL) THEN
          CALL PHASER(WDES%GRID,LATT_CUR,NONLR_S,NK,WDES)
       ELSE
          CALL PHASE(WDES,NONL_S,NK)
       ENDIF
       
       CALL WA_COPY(ELEMENTS(W, WDES1, ISP), WA)
       ACC2=ELEMENTS(WACC2, WDES1, ISP)
       CALL REDISTRIBUTE_PW( ACC2)
!      CALL REDISTRIBUTE_PROJ(ACC2)  ! ACC2%CPROJ is not used

       CALL OVERL(WDES1, .TRUE.,LMDIM,CDIJ1(1,1,1,ISP), WA%CPROJ(1,1),WNONL%CPROJ(1,1))
       CALL OVERL(WDES1, .TRUE.,LMDIM,CDIJ_HARTREE_ION(1,1,1,ISP), WA%CPROJ(1,1),WNONL_HARTREE_ION%CPROJ(1,1))

       CALL REDISTRIBUTE_PW( WA )
       CALL REDISTRIBUTE_PROJ(WA)
       CALL REDISTRIBUTE_PROJ(WNONL)
       CALL REDISTRIBUTE_PROJ(WNONL_HARTREE_ION)

       CHAM=0
       CTMP=0
       COVL=0
       strip: DO NPOS=1,NBANDSGW/WDES%NB_PAR,NSTRIP
          NSTRIP_ACT=MIN(NBANDSGW/WDES%NB_PAR+1-NPOS,NSTRIP)
             
          !  calculate V_{local} |phi> + T | phi >
          !  for a block containing NSTRIP wavefunctions
             
          ! calculate HF contribution on current stripe of bands
          ! (using the full set of k-points)
          CALL FOCK_ACC(WDES%GRID, LMDIM, LATT_CUR, W,  &
               NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
               WHAM%CW(:,:), P, CQIJ(1,1,1,1), CDCHF )
          
          DO N=NPOS,NPOS+NSTRIP_ACT-1
             NP=N-NPOS+1
             CALL SETWAV(W, W1, WDES1, N, ISP)
             CALL FFTWAV_W1(W1)
             WHAM_HARTREE_ION%CW(:,NP)=WHAM%CW(:,NP)
             CALL HAMILT_LOCAL(W1, SV1, ISP, WHAM%CW(:,NP),.TRUE.,LKINETIC)
             CALL HAMILT_LOCAL(W1, SV_HARTREE_ION, ISP, WHAM_HARTREE_ION%CW(:,NP),.TRUE.)
             IF (N<=NBANDSGW_/WDES%NB_PAR) THEN
             ! add correlation
             ! possible to remove this part by commenting WACC1%CW out
                IF (.NOT. LEXX) THEN
                   WHAM%CW(:,NP)=WHAM%CW(:,NP)+ WACC1%CW(:,N,NK,ISP)
                   WHAM_HARTREE_ION%CW(:,NP)=WHAM_HARTREE_ION%CW(:,NP)+ WACC1%CW(:,N,NK,ISP)
                ENDIF
             ENDIF
          ENDDO
          ! redistribute wavefunctions
          ! after this redistributed up to and including 1...NPOS+NSTRIP_ACT
          CALL REDISTRIBUTE_PW( ELEMENTS( WHAM, 1, NSTRIP_ACT))
          CALL REDISTRIBUTE_PW( ELEMENTS( WHAM_HARTREE_ION, 1, NSTRIP_ACT))
          
          NPOS_RED  =(NPOS-1)*WDES%NB_PAR+1
          NSTRIP_RED=NSTRIP_ACT*WDES%NB_PAR
          ! CHAM(2,1) = <u_2 | T+ V_local + sigma | u_1 >
          CALL ORTH2( &
               WA%CW_RED(1,1),WHAM%CW(1,1),WA%CPROJ_RED(1,1), &
               WNONL%CPROJ_RED(1,NPOS_RED),NB_TOT, &
               NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1))
          ! CTMP(2,1) = <u_2 | T+ V_local + sigma | u_1 >
          CALL ORTH2( &
               WA%CW_RED(1,1),WHAM_HARTREE_ION%CW(1,1),WA%CPROJ_RED(1,1), &
               WNONL_HARTREE_ION%CPROJ_RED(1,NPOS_RED),NB_TOT, &
               NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CTMP(1,1))
          ! COVL(2,1) = <u_2 | d sigma / d E | u_1 >
          CALL ORTH2( &
               WA%CW_RED(1,1),ACC2%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
               ACC2%CPROJ_RED(1,NPOS_RED),NB_TOT, &
               NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
       ENDDO strip

       CALLMPI( M_sum_g(WDES%COMM_KIN,CHAM(1,1),NB_TOT*NB_TOT))
       CALLMPI( M_sum_g(WDES%COMM_KIN,CTMP(1,1),NB_TOT*NB_TOT))
       CALLMPI( M_sum_g(WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))
       
       ! now add matrix elements in upper triangle beyond NBANDSGW_
       DO N=NBANDSGW_+1,NBANDSGW
          DO NP=1,NBANDSGW_
             CHAM(NP,N)=GCONJG(CHAM(N,NP))
          ENDDO
       ENDDO
       
       IF ( LCOHSEX ) THEN
          DO N=1,NBANDSGW_
             SIGMAQP(N)=CTMP(N,N)
             Z(N)=1
             CELNEW(N)=REAL(W%CELTOT(N,NK,ISP),q)+Z(N)*(SIGMAQP(N)-REAL(W%CELTOT(N,NK,ISP)))
             ! top of conduction band
             ! store shift
             IF (W%FERTOT(N,NK,ISP)>0.4 .AND. REAL(W%CELTOT(N,NK,ISP),q)>EMAX(ISP)) THEN
                EMAX(ISP)=W%CELTOT(N,NK,ISP)
                ESHIFT_TOP(ISP)=(SIGMAQP(N)-W%CELTOT(N,NK,ISP))
             ENDIF
             
             ESHIFT(ISP)=ESHIFT(ISP)+(SIGMAQP(N)-W%CELTOT(N,NK,ISP))*WDES%WTKPT(NK)*WDES%RSPIN*W%FERTOT(N,NK,ISP)
          ENDDO
          
          ! dump QP energies
          IF (IU6>=0) THEN
             WRITE(IU6,1) NK, WDES%VKPT(:,NK)
             DO N=1,NBANDSGW_
                WRITE(IU6,10) & 
                     N, REAL(W%CELTOT(N,NK,ISP),q), CELNEW(N), SIGMAQP(N),Z(N), &
                     W%FERTOT(N,NK,ISP)*W%WDES%RSPIN
             ENDDO
          ENDIF
#ifdef MPI
          IF (WDES%COMM_KIN%NODE_ME.eq.1) then
!i'm the communication node
            IF (WDES%COMM_KINTER%NODE_ME.eq.1) then
!sitting on the output node
              DO NK2=2,WDES%COMM_KINTER%NCPU
                CALLMPI ( M_recv_d (WDES%COMM_KINTER, NK2, QP_OUT, 5*NBANDSGW_))
                WRITE(IU6,1) NK2+NK-1, WDES%VKPT(:,NK2+NK-1)
                DO N=1,NBANDSGW_
                  WRITE(IU6,10) N, QP_OUT(N,:)
                ENDDO
              ENDDO
            ELSE
!sitting on a node with kpt NK
              QP_OUT=0._q
              QP_OUT(:,1)=REAL(W%CELTOT(:,NK,ISP),q)
              QP_OUT(:,2)=CELNEW(:)
              QP_OUT(:,3)=SIGMAQP(:)
              QP_OUT(:,4)=Z(:)
              QP_OUT(:,5)=W%FERTOT(:,NK,ISP)*W%WDES%RSPIN
              CALLMPI ( M_send_d (WDES%COMM_KINTER, 1, QP_OUT, 5*NBANDSGW_))
            ENDIF
          ENDIF
#endif

       ELSE
          DO N=1,NBANDSGW_
             SIGMAQP(N)=CTMP(N,N)
             Z(N)=1/(1+REAL(COVL(N,N),q))
             CELNEW(N)=REAL(W%CELTOT(N,NK,ISP),q)+Z(N)*(SIGMAQP(N)-REAL(W%CELTOT(N,NK,ISP)))
             ! top of conduction band
             ! store shift
             IF (W%FERTOT(N,NK,ISP)>0.4 .AND. REAL(W%CELTOT(N,NK,ISP),q)>EMAX(ISP)) THEN
                EMAX(ISP)=W%CELTOT(N,NK,ISP)
                ESHIFT_TOP(ISP)=(SIGMAQP(N)-W%CELTOT(N,NK,ISP))
             ENDIF
             
             ESHIFT(ISP)=ESHIFT(ISP)+(SIGMAQP(N)-W%CELTOT(N,NK,ISP))*WDES%WTKPT(NK)*WDES%RSPIN*W%FERTOT(N,NK,ISP)
          ENDDO
          
          ! dump QP energies
          ! in the k-point parallel version only the QP shifts from the k-points on the output node are written
          ! that means 1 3 5 7 or 1 5 or 1 for 8 NKPTS and KPAR 2, 4, and 8 respectively
          IF (IU6>=0) THEN
!send if KINTER!=0 and KIN=0
!receive if KINTER=0 and KIN=0 (node attached to OUTCAR) 
             WRITE(IU6,1) NK, WDES%VKPT(:,NK)
             DO N=1,NBANDSGW_
               WRITE(IU6,10) & 
                     N, REAL(W%CELTOT(N,NK,ISP),q), CELNEW(N), SIGMAQP(N),Z(N), &
                     W%FERTOT(N,NK,ISP)*W%WDES%RSPIN
             ENDDO
          ENDIF
#ifdef MPI
!write the QP energies from other k-points, WDES%VKPT(:,NK) stored locally, the other values need to be communicated 
!loop over other k-points and print if received
          IF (WDES%COMM_KIN%NODE_ME.eq.1) then
!i'm the communication node
            IF (WDES%COMM_KINTER%NODE_ME.eq.1) then
!sitting on the output node
!this loop will not be performed if KPAR=1(=WDES%COMM_KINTER%NCPU)
              DO NK2=2,WDES%COMM_KINTER%NCPU
                CALLMPI ( M_recv_d (WDES%COMM_KINTER, NK2, QP_OUT, 5*NBANDSGW_))
                WRITE(IU6,1) NK2+NK-1, WDES%VKPT(:,NK2+NK-1)
                DO N=1,NBANDSGW_
                  WRITE(IU6,10) N, QP_OUT(N,:)
                ENDDO
              ENDDO
            ELSE
!sitting on a node with kpt NK
              QP_OUT=0._q 
              QP_OUT(:,1)=REAL(W%CELTOT(:,NK,ISP),q)
              QP_OUT(:,2)=CELNEW(:)
              QP_OUT(:,3)=SIGMAQP(:)
              QP_OUT(:,4)=Z(:)
              QP_OUT(:,5)=W%FERTOT(:,NK,ISP)*W%WDES%RSPIN
              CALLMPI ( M_send_d (WDES%COMM_KINTER, 1, QP_OUT, 5*NBANDSGW_))
            ENDIF
         ENDIF
#endif
       ENDIF

       IF (LUPDATEQP) THEN
          DO  N=1,NBANDSGW_
             CHAM(1:NBANDSGW_,N)=CHAM(1:NBANDSGW_,N)+(W%CELTOT(N,NK,ISP)-CELNEW(N))*COVL(1:NBANDSGW_,N)
          ENDDO
       ENDIF
       
       DO  N=1,NBANDSGW
          DO  NP=N,NBANDSGW
             CHAM(N,NP)=(CHAM(N,NP)+GCONJG(CHAM(NP,N)))/2
             CHAM(NP,N)=GCONJG(CHAM(N,NP))
          ENDDO
       ENDDO

       CALL OEP_CHARGE(GRID, LMDIM, LATT_CUR, W, WDESQ, &
            ISP, NK, (NSTRIP+W%WDES%NB_PAR-1)/W%WDES%NB_PAR, CHI0, CHAM , ISMEAR, SIGMA, GCHG(:,ISP), LCHI0) 

    ENDDO
!=======================================================================
! at this point we have to solve 
!  rho = Xi V 
! for the potential V
!=======================================================================
    IF (LCHI0) THEN
       CALL CLEAN_RESPONSEFUNCTION_CACHE(CHI0, 1, CHI0%NOMEGA)
       CALL DO_CHI_SUM(CHI0, WDESQ, .TRUE. )
    ENDIF
    CALLMPI( M_sum_z(WDESQ%COMM_INTER, GCHG(:,ISP), SIZE(GCHG,1)))
    CALLMPI( M_sum_z(WDESQ%COMM_KINTER, GCHG(:,ISP), SIZE(GCHG,1)))

! store response function in CHI_WORK
    CALL SET_MAT_FROM_RESPONSE( CHI_WORK(:,:,ISP), CHI0, 1, 0)

#ifdef  test_specific_charge
    WRITE(*,*) 'Result from OEP_CHARGE'
    WRITE(*,'(10F8.3)') GCHG(:,ISP)
#endif

   ENDDO sp1

   ! store potential shift (average occupied bands)
    ESHIFT=ESHIFT/INFO%NELECT
   ! align top of valence band
    ESHIFT=ESHIFT_TOP

    IF (IU0 >0) THEN
       WRITE(IU0,"('potential shift',F14.5)") ESHIFT
    ENDIF
    IF (IU6>=0) THEN
       WRITE(IU6,"('potential shift',F14.5)") ESHIFT
    ENDIF
!=======================================================================
! pain in the neck, if calculated locally the response function 
! does not have the full symmetry
! for the time being, I simply symmetrize the charge density GCHG
! this is not quite exact
!=======================================================================
    DO ISP=1,WDES%ISPIN
       CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
            CWORK(1), GCHG(1,ISP), WDESQ%GRID)
       ! copy wave function based array (COMPLEX) to RGRID
       CALL RL_COPY_WAVE_RGRID( CWORK(1), SV(1,ISP), 1.0_q, GRID)
    
       CALL FFT_RC_SCALE(SV(1,ISP), CWORK(1), GRID_SOFT)
       CALL RC_ADD(CWORK(1),1.0_q,CWORK(1),0.0_q,SV(1,ISP),GRID_SOFT)
    ENDDO

    ! change storage convention to (total, magnetization)
    CALL RC_FLIP(SV,GRID,WDES%NCDIJ,.FALSE.)

    IF (SYMM%ISYM==2) THEN
       IF (WDES%LNONCOLLINEAR) THEN
          IF (IU0>=0) THEN
             WRITE(IU0,*) 'internal error in OEP_GW: presently non collinear calculations are not supported'
          ENDIF
          STOP
       ELSE
          DO ISP=1,WDES%ISPIN
             CALL RHOSYM(SV(1,ISP),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
          ENDDO
       ENDIF
    ENDIF

    ! change storage convention back
    CALL RC_FLIP(SV,GRID,WDES%NCDIJ,.TRUE.)

    DO ISP=1,WDES%ISPIN
       ! go to real space
       CALL FFT3D( SV(1,ISP), GRID_SOFT, 1 )
       ! correct data on all nodes
#ifdef realmode
       CALLMPI( M_bcast_d(COMM_INTER, SV(1,ISP), GRID%RL%NP))
       CALLMPI( M_bcast_d(COMM_KINTER, SV(1,ISP), GRID%RL%NP))
#else
       CALLMPI( M_bcast_z(COMM_INTER, SV(1,ISP), GRID%RL%NP))
       CALLMPI( M_bcast_z(COMM_KINTER, SV(1,ISP), GRID%RL%NP))
#endif
       ! go back to GCHG
       CALL RL_COPY_RGRID_WAVE(SV(1,ISP), CWORK(1), 1.0_q, GRID)
       CALL FFTEXT(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
            CWORK(1), GCHG(1,ISP), WDESQ%GRID, .FALSE.)
    
       GCHG(:,ISP)=GCHG(:,ISP)*(1.0_q/WDESQ%GRID%NPLWV)
    ENDDO
!=======================================================================
!  solve  rho = Xi V  for the potential V
!=======================================================================
    DO ISP=1,WDES%ISPIN

       NP=WDESQ%NGVECTOR
       IF (WDESQ%LGAMMA) NP=NP*2
       CHI_WORK(1,:,ISP)=0
       CHI_WORK(:,1,ISP)=0
       CHI_WORK(1,1,ISP)=1
       IF (CHI0%LREAL) THEN
          CHI_WORK(2,:,ISP)=0
          CHI_WORK(:,2,ISP)=0
          CHI_WORK(2,2,ISP)=1
       ENDIF

       ! force it to be Hermitian (should be anyhow)
       DO N=1,NP
          DO N2=N,NP
             CHI_WORK(N,N2,ISP)=(CHI_WORK(N,N2,ISP)+CONJG(CHI_WORK(N2,N,ISP)))/2
             CHI_WORK(N2,N,ISP)=CONJG(CHI_WORK(N,N2,ISP))
          ENDDO
       ENDDO
       ! standard inversion
       !    CALL ZGETRF( NP, NP, CHI_WORK(1,1,ISP), SIZE(CHI_WORK,1), IPIV, IERR )
       !    CALL ZGETRS('T', NP, 1, CHI_WORK(1,1,ISP), SIZE(CHI_WORK,1), IPIV, GCHG(1,ISP), SIZE(GCHG,1), IERR )

       ! inversion using singular value decomposition, eigenvalues smaller 1E-6 are disposed
       CALL CHI_INVERT(CHI_WORK(:,:,ISP), NP, 1.E-4_q , IU6)

       IF (CHI0%LREAL) THEN
          GCHG_=0
          ! real valued vector with NP data points -> NP complex  data points
          CALL DAXPY( NP, 1.0_q, GCHG(1,ISP), 1,  GCHG_(1), 2)
          GCHG_(1:NP)=MATMUL(GCHG_(1:NP),CHI_WORK(1:NP,1:NP,ISP) )
          
          GCHG(:,ISP)=0
          ! NP complex  data points -> NP real points
          CALL DAXPY( NP, 1.0_q, GCHG_(1), 2, GCHG(1,ISP), 1)
       ELSE
          GCHG(1:NP,ISP)=MATMUL(GCHG(1:NP,ISP),CHI_WORK(1:NP,1:NP,ISP) )
       ENDIF
       
       GCHG(1,ISP)=ESHIFT(ISP)

       ! just in case broadcast to all nodes
       CALLMPI( M_bcast_z(WDES%COMM_INTER, GCHG(1,ISP), NP))
       CALLMPI( M_bcast_z(WDES%COMM_KINTER, GCHG(1,ISP), NP))
    ENDDO
!=======================================================================
! symmetrize the final potential
! since this can not be done using GCHG, go to potential on GRID_SOFT
! symmetrize and go back
!=======================================================================
    DO ISP=1,WDES%ISPIN
       CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
            CWORK(1), GCHG(1,ISP), WDESQ%GRID)
    
       ! copy wave function based array (COMPLEX) to RGRID
       CALL RL_COPY_WAVE_RGRID( CWORK(1), SV1(1,ISP), 1.0_q, GRID)
    
       CALL FFT_RC_SCALE(SV1(1,ISP), CWORK(1), GRID_SOFT)
       CALL RC_ADD(CWORK(1),1.0_q,CWORK(1),0.0_q,SV1(1,ISP),GRID_SOFT)
    ENDDO

    ! change storage convention to (total, magnetization)
    CALL RC_FLIP(SV1,GRID,WDES%NCDIJ,.FALSE.)

    IF (SYMM%ISYM==2) THEN
       IF (WDES%LNONCOLLINEAR) THEN
          IF (IU0>=0) THEN
             WRITE(IU0,*) 'internal error in OEP_GW: presently non collinear calculations are not supported'
          ENDIF
          STOP
       ELSE
          DO ISP=1,WDES%ISPIN
             CALL RHOSYM(SV1(1,ISP),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
          ENDDO
       ENDIF
    ENDIF

    ! change storage convention back
    CALL RC_FLIP(SV1,GRID,WDES%NCDIJ,.TRUE.)

    DO ISP=1,WDES%ISPIN
       ! store SV  (update potential)
       SV(:,ISP)=SV1(:,ISP)*AMIX
       ! go to real space
       CALL FFT3D( SV1(1,ISP), GRID_SOFT, 1 )
       ! correct data on all nodes
#ifdef realmode
       CALLMPI( M_bcast_d(COMM_INTER, SV1(1,ISP), GRID%RL%NP))
       CALLMPI( M_bcast_d(COMM_KINTER, SV1(1,ISP), GRID%RL%NP))
#else
       CALLMPI( M_bcast_z(COMM_INTER, SV1(1,ISP), GRID%RL%NP))
       CALLMPI( M_bcast_z(COMM_KINTER, SV1(1,ISP), GRID%RL%NP))
#endif
       ! go back to GCHG
       CALL RL_COPY_RGRID_WAVE(SV1(1,ISP), CWORK(1), 1.0_q, GRID)

       CALL FFTEXT(WDESQ%NGVECTOR, WDESQ%NINDPW(1), &
            CWORK(1), GCHG(1,ISP), WDESQ%GRID, .FALSE.)
    
       GCHG(:,ISP)=GCHG(:,ISP)*(1.0_q/WDESQ%GRID%NPLWV)
#ifdef  test_specific_charge
       WRITE(*,*) 'found the following potential correction after multiplication with X1-1'
       WRITE(*,'(10F8.3)') GCHG(:,ISP)
#endif

    ENDDO
!=======================================================================
! now update CVTOT, SV, and CDIJ
!=======================================================================
!    IF (IU6>=0) THEN
!       WRITE(IU6,*) 'old potential'
!       CALL WRT_RL_LINE(IU6, GRIDC, CVTOT)
!    ENDIF
    ! first determine the one-center contributions
    CDIJ1=CDIJ
    ! determine CDIJ from CVTOT
    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT,IRDMAA,IRDMAX)
    ! subtract the plane wave part from the CDIJ that were read from the file 
    ! to obtain the OEP contribution only
    CDIJ1=CDIJ-CDIJ1
     
    ! add correction SV to CVTOT
    DO ISP=1,WDES%NCDIJ
       CALL FFT_RC_SCALE(CVTOT(1,ISP),CVTOT(1,ISP),GRIDC)
       CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C, SV(1,ISP), CVTOT(1,ISP))
    ENDDO
    
    ! now set SV from CVTOT
    CALL SET_SV( GRID, GRIDC, GRID_SOFT, WDES%COMM_INTER, SOFT_TO_C, WDES%NCDIJ, SV, CVTOT)
    
    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

    ! add one center OEP contributions
    CDIJ=CDIJ+CDIJ1

!    IF (IU6>=0) THEN
!       WRITE(IU6,*) 'new potential'
!       CALL WRT_RL_LINE(IU6, GRIDC, CVTOT)
!    ENDIF

    CALL DELWAVA(WHAM)
    CALL DELWAVA(WHAM_HARTREE_ION)
    CALL DELWAVA_PROJ(WNONL)
    CALL DELWAVA_PROJ(WNONL_HARTREE_ION)
    CALL DELWAVA(WA)
    DEALLOCATE(W1%CR)
    DEALLOCATE(CHAM, COVL, CTMP)
 1  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
          &         "  band No.  KS-energies   G0W0  sigma(KS)+T+V_ion+V_H  Z         occupation"/)
  
10  FORMAT((3X,I4,3X,7(F10.4,3X)))

  END SUBROUTINE OEP_GW


!************************ SUBROUTINE CHI_INVERT   **********************
!
! this subroutine calculates a matrix to the power -1
! but kills the very low frequency components
!
!***********************************************************************

  SUBROUTINE CHI_INVERT(CUNI, NBANDS, THRESHHOLD, IU6 )
    INTEGER NBANDS
    COMPLEX(q) CUNI(:,:)
    REAL (q) :: THRESHHOLD
    INTEGER ::  IU6
    ! local
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    COMPLEX(q), ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)
    
    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))
!=======================================================================
! diagononalize the matrix
!=======================================================================
    CALL ZHEEV &
         &         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         &           W,  IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in CHI_INVERT: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    IF (IU6>=0) WRITE(IU6,'(A)') 'eigenvalues of the response function'
    IF (IU6>=0) WRITE(IU6,'(10F12.8)') HFEIG
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
    DO N1=1,NBANDS
       DO N2=1,NBANDS
          IF (ABS(HFEIG(N2))<THRESHHOLD) THEN
             CTMP(N1,N2)=0
          ELSE
             CTMP(N1,N2)=CUNI(N1,N2)/HFEIG(N2)
          ENDIF
       ENDDO
    ENDDO
    CALL ZGEMM( 'N','C', NBANDS, NBANDS, NBANDS, (1.0_q, 0.0_q), CTMP, NBANDS, &
         &               CUNI(1,1), NDIM, (0.0_q, 0.0_q), CEIDB, NBANDS)
   
    CUNI=0
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE CHI_INVERT

!************************ SUBROUTINE OEP_CHARGE ************************
!
! construct the OEP charge from this specific band
! this is done in an analogous way as in the ADD_XI routine
! 
! VASP uses the convention:
! rho(G) = sum_G' RESPONSEFUN_q(G',G,w) V(G')
!
! with
! RESPONSEFUN(G',G,w) =
!
!         = 1/N^2 sum_rr' e iG'r' xi_q(r',r) e -iGr 
!
!         = \sum_1,2  1/N (\sum_r  u_1(r)  u*_2(r)  e -iGr )
!                     1/N (\sum_r' u_1(r') u*_2(r') e -iG'r')*
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1- e_2- i delta    -w+ e_1- e_2- i delta
!
! This yields sum_G' RESPONSEFUN_q(G',G,w) V(G')
!
! rho(G)  = 
!         = \sum_1,2  1/N (\sum_r  u_1(r)  u*_2(r)  e -iGr )
!                     1/N (\sum_r' u*_1(r') u_2(r') V(r') 
!                      1                      1
!        x (  -------------------- + ----------------------- )
!             w+ e_1- e_2- i delta    -w+ e_1- e_2- i delta
! 
!***********************************************************************

  SUBROUTINE OEP_CHARGE(GRID, LMDIM, LATT_CUR, W, WDESQ, &
       ISP, NK, NSTRIP, CHI0, CHAM, ISMEAR, SIGMA, CHG_ACCUMULATED, LCHI0 ) 
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE fock
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (grid_3d) GRID
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (wavedes1) WDESQ
    INTEGER ISP            ! spin index
    INTEGER NK             ! k-point
    INTEGER NSTRIP         ! blocking to be applied
    TYPE (responsefunction) CHI0
    GDEF       :: CHAM(:,:)
    INTEGER ISMEAR         ! smearing mode
    REAL(q) SIGMA          ! smearing width
    COMPLEX(q) :: CHG_ACCUMULATED(:)
    LOGICAL    :: LCHI0
! local variables
    TYPE (wavedes1) :: WDESK
    TYPE (wavefun1) :: W2
    TYPE (wavefun1),ALLOCATABLE :: W1(:)
    INTEGER NB1, N, NB2, NGLB, NB1_INTO_TOT, NP, NSTRIP_ACT, NB2_INTO_TOT
    INTEGER LAST_FILLED
    COMPLEX(q) :: WEIGHT
    COMPLEX(q), ALLOCATABLE :: GCHG(:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:)   ! charge weighted by 1/(w-e1-e2)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    COMPLEX(q) :: CSUM
    TYPE (wavespin) WHF
    INTEGER, EXTERNAL :: ONE_CENTER_NMAX_FOCKAE
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

    ! set the phase factors (zero here)
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, (/ 0.0_q, 0.0_q , 0.0_q /) )

    CALL SETWDES(WHF%WDES,WDESK,NK)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG( MAX(GRIDHF%MPLWV,WDESQ%GRID%MPLWV)),GWORK(GRIDHF%MPLWV))
         
    CALL NEWWAV(W2, WDESK, .TRUE.)

    CALL SETWDES(WHF%WDES, WDESK, NK)
    NGLB=NSTRIP*WHF%WDES%NB_PAR

    ALLOCATE(W1(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(W1(N) , WDESK,.TRUE.)
    ENDDO
!==========================================================================
! fourier transform the occupied bands at NK
! to real space, then gather into W1
!==========================================================================
    LAST_FILLED=LAST_FILLED_XI(WHF, NK, ISP)/ WHF%WDES%NB_PAR
    DO NB1=1,LAST_FILLED , NSTRIP

    NSTRIP_ACT=MIN(LAST_FILLED+1-NB1,NSTRIP)

    CALL W1_GATHER_N( WHF, NB1, NB1+NSTRIP_ACT-1, ISP, W1, NGLB)

    NGLB=NSTRIP_ACT*WHF%WDES%NB_PAR
    NP=WDESQ%NGVECTOR

    ! loop over all unoccupied orbitals
    CSUM=0
    mband: DO NB2=FIRST_EMPTY_XI_LOCAL( WHF, NK, ISP), LAST_EMPTY_XI_LOCAL( WHF, NK, ISP)
       NB2_INTO_TOT=(NB2-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
       IF (FILLED_XI_ORBITAL((WHF%FERWE( NB2, NK, ISP))) .OR. WHF%AUX(NB2, NK, ISP)==0 ) THEN
          CYCLE
       ENDIF

       CALL W1_COPY(ELEMENT(WHF, WDESK, NB2, ISP), W2)
       CALL FFTWAV_W1(W2)
!-----------------------------------------------------------------------------
! calculate the charge u_1(r) u_2*(r) 
!-----------------------------------------------------------------------------
       ! loop over all occupied bands in the current block
       nband: DO N=1,NGLB
          ! determine index into the FERTOT (fermi-weight) array
          NB1_INTO_TOT=(NB1-1)*W%WDES%NB_PAR+N
          
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,NK,ISP))) THEN
             CYCLE
          ENDIF
         
          ! here is a real issue:
          ! the calculated response function takes into account LMAXFOCKAE
          ! so we should include it here as well to get a charge density that include LMAXFOCKAE
          ! we then solve for the potential V = Xi rho
          ! however the local potential is later used without LMAXFOCKAE to
          ! calculate <phi_i | V | phi_j >
          ! this is clearly not consistent and results in some error
          ! the best solution is to avoid LMAXFOCKAE here *and* in the
          ! response function
          IF ( ONE_CENTER_NMAX_FOCKAE() > 0) THEN
          ! this issue is now solved since AE augmentation is handled properly
          ! by SETDIJ and SET_DD_PAW
             CALL FOCK_CHARGE_NOINT( W1(N), W2, GCHG(1), CRHOLM(1), SIZE(CRHOLM))
          ELSE
          ! ok if we do not use the 
             CALL FOCK_CHARGE_NOINT_NOAE( W1(N), W2, GCHG(1), CRHOLM(1), SIZE(CRHOLM))
          ENDIF
!-----------------------------------------------------------------------------
! FFT of charge to the reciprocal space
!-----------------------------------------------------------------------------
          ! extract data using wave function FFT (impose cutoff at the same time)
          CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
               GCHG (1), GWORK(1),WDESQ%GRID,.FALSE.)
          ! divide by number of grid points
          GWORK(1:NP)=GWORK(1:NP)*(1.0_q/GRIDHF%NPLWV)

          IF (CHI0%LREAL) THEN
             GCHG(1:NP)=GWORK(1:NP)
          ELSE
             GCHG(1:NP)=CONJG(GWORK(1:NP))
          ENDIF

          WEIGHT=0
          IF (REAL(WHF%CELTOT(NB2_INTO_TOT, NK, ISP)-WHF%CELTOT(NB1_INTO_TOT,NK,ISP),q)>0) THEN
                   
!             CALL SET_XI_WEIGHT( WEIGHT, REAL(WHF%CELTOT(NB1_INTO_TOT,NK,ISP)-WHF%CELEN(NB2, NK, ISP),q), CHI0%SHIFT, CHI0%COMEGA(1), ISMEAR, SIGMA)
             ! the WEIGHT should be real, otherwise the conjugation below will be not working quite correctly
             WEIGHT = 2/ REAL(WHF%CELTOT(NB1_INTO_TOT,NK,ISP)-WHF%CELEN(NB2, NK, ISP),q)
             WEIGHT=WHF%WDES%RSPIN* WHF%WDES%WTKPT(NK)* (WHF%FERTOT(NB1_INTO_TOT,NK,ISP)-WHF%FERWE(NB2, NK, ISP))*WEIGHT
             WEIGHT=WEIGHT*WHF%AUX(NB2, NK, ISP)
          ENDIF
          ! CHAM(1,2) = <u_1| V | u_2 > =  1/N \sum_r' u*_1(r') u_2(r') V(r')

          CHG_ACCUMULATED(1:NP)=CHG_ACCUMULATED(1:NP)+GWORK(1:NP)*CHAM(NB1_INTO_TOT,NB2_INTO_TOT)*WEIGHT

          IF (LCHI0) CALL ADD_RESPONSEFUNCTION_CACHE(CHI0, GCHG(:), WEIGHT, 1, NP )

       ENDDO nband
    ENDDO mband
    !-----------------------------------------------------------------------------
    ENDDO

    DEALLOCATE(CRHOLM, GCHG, GWORK)
    CALL DELWAV(W2,.TRUE.)

    DO N=1,NGLB
       CALL DELWAV(W1(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(W1)

#undef debug
  END SUBROUTINE OEP_CHARGE


!*********************************************************************
!
! helper routine:
! allocates or deallocates a response function array and
! the cache substructures
! 
! LREALSTORE selects a special storage mode usually
!            this is only selected for ACFDT and Gamma point
!            only calculations
!            or for the potential handles in the wpot
!
!*********************************************************************

  SUBROUTINE ALLOCATE_RESPONSEFUN( CHI, NALLOC, LGAMMA, LREALSTORE, NOMEGA_CHI)
    USE ini
    TYPE (responsefunction) :: CHI
    LOGICAL :: LGAMMA       ! special mode using sin and cosine transforms
    LOGICAL :: LREALSTORE   ! special mode using real valued responsefunctions
    INTEGER, INTENT(IN)  :: NALLOC, NOMEGA_CHI
  ! local
    INTEGER :: N1, N2

    N1=NALLOC
    N2=NALLOC
    IF (LGAMMA .AND. .NOT. LREALSTORE) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! this gives 2*NALLOC words in each direction
    ! the response function is complex
    ! this is usually used for real frequencies
       N1=NALLOC*2
       N2=NALLOC*2
    ELSE IF (LGAMMA) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! the response function is stored as a real valued function
    ! if a complex array is allocted its first dimension can equal NALLOC
    ! since one complex number allows to store two real numbers
    ! this is usually used for complex frequencies (or at w=0)
       N1=NALLOC
       N2=NALLOC*2
    ENDIF
    CHI%LREAL=LGAMMA
    CHI%LREALSTORE=LREALSTORE
    CHI%NP1=N1
    CHI%NP2=N2

    ALLOCATE(CHI%RESPONSEFUN(N1, N2, NOMEGA_CHI), & 
         CHI%WING(N1, 1:3, NOMEGA_CHI), &
         CHI%CWING(N1, 1:3, NOMEGA_CHI), &
         CHI%HEAD(1:3, 1:3, NOMEGA_CHI), &
         CHI%COMEGA(NOMEGA_CHI),CHI%OMEGA(NOMEGA_CHI))

    CALL REGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN), "response")

    ! tricky stuff make the RESPONSER entry equivalent to the RESPONSEFUN
    ! entry the number of real values is twice as large as the number of 
    ! complex entries in the first direction
    CALL SET_RESPONSER( CHI%RESPONSER, 2*N1, N2, NOMEGA_CHI, CHI%RESPONSEFUN(1,1,1))
    CALL SET_RESPONSER( CHI%WINGR,     2*N1, 3,  NOMEGA_CHI, CHI%WING(1,1,1))
    CALL SET_RESPONSER( CHI%CWINGR,    2*N1, 3,  NOMEGA_CHI, CHI%CWING(1,1,1))

    NULLIFY(CHI%STORAGE, CHI%WEIGHT, CHI%STPOS, CHI%W_INTER, CHI%NCACHE)
    CHI%MAXCACHE=0
    CHI%NOMEGA=NOMEGA_CHI
    CHI%LSHMEM=.FALSE.
    CHI%LLEAD =.TRUE.
    NULLIFY(CHI%COMM_SHMEM)

  END SUBROUTINE ALLOCATE_RESPONSEFUN


  SUBROUTINE ALLOCATE_RESPONSEFUN_SHMEM( CHI, NALLOC, LGAMMA, LREALSTORE, NOMEGA_CHI, COMM_SHMEM, IU0, IU6, LSEM )
    USE ini
    USE c2f_interface
    TYPE (responsefunction) :: CHI
    LOGICAL :: LGAMMA       ! special mode using sin and cosine transforms
    LOGICAL :: LREALSTORE   ! special mode using real valued responsefunctions
    INTEGER, INTENT(IN)  :: NALLOC, NOMEGA_CHI
    TYPE (communic), POINTER :: COMM_SHMEM
    INTEGER IU0, IU6        ! error io handles
    LOGICAL, OPTIONAL :: LSEM 
  ! local
    INTEGER :: N1, N2
    LOGICAL :: LSEM_
#ifdef use_shmem
    INTEGER ::  ierror
    TYPE(c_ptr)       :: address
    INTEGER(c_size_t) :: MEM
#endif
    ! default is to allocate semamphores
    LSEM_=.TRUE.
    ! if optional argument is present, set LSEM to value supplied by caller
    IF (PRESENT (LSEM)) THEN
       LSEM_=LSEM
    ENDIF
 
    N1=NALLOC
    N2=NALLOC
    IF (LGAMMA .AND. .NOT. LREALSTORE) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! this gives 2*NALLOC words in each direction
    ! the response function is complex
    ! this is usually used for real frequencies
       N1=NALLOC*2
       N2=NALLOC*2
    ELSE IF (LGAMMA) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! the response function is stored as a real valued function
    ! if a complex array is allocted its first dimension can equal NALLOC
    ! since one complex number allows to store two real numbers
    ! this is usually used for complex frequencies (or at w=0)
       N1=NALLOC
       N2=NALLOC*2
    ENDIF
    CHI%LREAL=LGAMMA
    CHI%LREALSTORE=LREALSTORE
    CHI%NP1=N1
    CHI%NP2=N2


    ALLOCATE(CHI%WING(N1, 1:3, NOMEGA_CHI), &
         CHI%CWING(N1, 1:3, NOMEGA_CHI), &
         CHI%HEAD(1:3, 1:3, NOMEGA_CHI), &
         CHI%COMEGA(NOMEGA_CHI),CHI%OMEGA(NOMEGA_CHI))

#ifdef use_shmem
    NULLIFY(CHI%COMM_INTER_SHMEM)

    CHI%LSHMEM=.FALSE.
    IF (ASSOCIATED(COMM_SHMEM)) THEN
       IF (COMM_SHMEM%NCPU>1) THEN
          CHI%LSHMEM=.TRUE.
       ENDIF
    ENDIF

    IF (CHI%LSHMEM) THEN
       CALLMPI( M_barrier(COMM_SHMEM) )
       
       ! set SHMEMLOCK to -1 indicating a non initialized system V semaphore
       CHI%SHMEMLOCK=-1
       IF (COMM_SHMEM%NODE_ME==1 .AND. LSEM_) THEN
          CALL GETSEM( NOMEGA_CHI, CHI%SHMEMLOCK )
       ENDIF
       
       CALLMPI( M_bcast_i(COMM_SHMEM,CHI%SHMEMLOCK,1))

       MEM=NOMEGA_CHI
       MEM=(MEM*16*N1)*N2
       IF (COMM_SHMEM%NODE_ME==1) THEN
          CALL GETSHMEM_ERROR(int(MAX(1_c_size_t,MEM),c_size_t), CHI%SHMID)
          CHI%LLEAD =.TRUE.
       ELSE
          CHI%LLEAD =.FALSE.
       ENDIF
       CALLMPI( M_bcast_i(COMM_SHMEM, CHI%SHMID, 1))

       IF (CHI%SHMID==-1) THEN
          ! somewhat akward all root shmem nodes do the dump to stderr to make sure 
          ! to user catches the error (required if machines are heterogeneous)
          IF (COMM_SHMEM%NODE_ME==1) THEN
             CALL VTUTOR('S','SHMEM ERROR', &
             &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,0,3)
          ENDIF
          CALL VTUTOR('S','SHMEM ERROR', &
          &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)          
          
          STOP
       ENDIF

       call attachshmem(CHI%SHMID, address)
       call c_f_pointer(address,CHI%RESPONSEFUN,[N1, N2, NOMEGA_CHI])       

       ! just to be sure wait for all "assignments to be finished"
       CALLMPI( M_barrier(COMM_SHMEM) )

       ! tricky we destroy the shmem segment here: 
       ! this works, since the memory is attached, and actual
       ! deletion does not occur until the detachment occurs
       ! this seems to work, at least in Linux.
       ! If this fails comment out the three lines below
       ! and comment in the second DESTROYSHMEM(CHI%SHMID) below
       ! in the routine DEALLOCATE_RESPONSEFUN
       IF (CHI%LLEAD) THEN
          CALL DESTROYSHMEM(CHI%SHMID)
       ENDIF

       ! register allocation per core
       CALL REGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN)/COMM_SHMEM%NCPU, "response")
       CHI%LSHMEM=.TRUE.
       CHI%COMM_SHMEM=>COMM_SHMEM
    ELSE
#endif
       ALLOCATE(CHI%RESPONSEFUN(N1, N2, NOMEGA_CHI))
       CALL REGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN), "response")
       CHI%LSHMEM=.FALSE.
       CHI%LLEAD =.TRUE.
       NULLIFY(CHI%COMM_SHMEM)
#ifdef use_shmem
    ENDIF
#endif
    ! tricky stuff make the RESPONSER entry equivalent to the RESPONSEFUN
    ! entry the number of real values is twice as large as the number of 
    ! complex entries in the first direction
    CALL SET_RESPONSER( CHI%RESPONSER, 2*N1, N2, NOMEGA_CHI, CHI%RESPONSEFUN(1,1,1))
    CALL SET_RESPONSER( CHI%WINGR,     2*N1, 3,  NOMEGA_CHI, CHI%WING(1,1,1))
    CALL SET_RESPONSER( CHI%CWINGR,    2*N1, 3,  NOMEGA_CHI, CHI%CWING(1,1,1))

    NULLIFY(CHI%STORAGE, CHI%WEIGHT, CHI%STPOS, CHI%W_INTER, CHI%NCACHE)
    CHI%MAXCACHE=0
    CHI%NOMEGA=NOMEGA_CHI

  END SUBROUTINE ALLOCATE_RESPONSEFUN_SHMEM

!
! determined memory for response function
!

  FUNCTION  RESPONSEFUN_MEMORY( NALLOC, LGAMMA, LREALSTORE )
    USE ini
    LOGICAL :: LGAMMA       ! special mode using sin and cosine transforms
    LOGICAL :: LREALSTORE   ! special mode using real valued responsefunctions
    INTEGER, INTENT(IN)  :: NALLOC
    REAL(q) :: RESPONSEFUN_MEMORY
  ! local
    INTEGER :: N1, N2

    N1=NALLOC
    N2=NALLOC
    IF (LGAMMA .AND. .NOT. LREALSTORE) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! this gives 2*NALLOC words in each direction
    ! the response function is complex
    ! this is usually used for real frequencies
       N1=NALLOC*2
       N2=NALLOC*2
    ELSE IF (LGAMMA) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! the response function is stored as a real valued function
    ! if a complex array is allocted its first dimension can equal NALLOC
    ! since one complex number allows to store two real numbers
    ! this is usually used for complex frequencies (or at w=0)
       N1=NALLOC
       N2=NALLOC*2
    ENDIF

    RESPONSEFUN_MEMORY=1E-6_q*N1*N2*16
    
  END FUNCTION RESPONSEFUN_MEMORY


  FUNCTION  RESPONSEFUN_MEMORY_CACHED( NALLOC, LGAMMA, LREALSTORE,  MAXCACHE, COMM_SHMEM )
    USE ini
    LOGICAL :: LGAMMA       ! special mode using sin and cosine transforms
    LOGICAL :: LREALSTORE   ! special mode using real valued responsefunctions
    INTEGER :: MAXCACHE
    TYPE (communic), POINTER :: COMM_SHMEM
    INTEGER, INTENT(IN)  :: NALLOC
    REAL(q) :: RESPONSEFUN_MEMORY_CACHED
  ! local
    INTEGER :: N1, N2

    N1=NALLOC
    N2=NALLOC
    IF (LGAMMA .AND. .NOT. LREALSTORE) THEN
       N1=NALLOC*2
       N2=NALLOC*2
    ELSE IF (LGAMMA) THEN
       N1=NALLOC
       N2=NALLOC*2
    ENDIF

    RESPONSEFUN_MEMORY_CACHED=1E-6_q*N1*N2*16
#ifdef use_shmem
    IF (ASSOCIATED(COMM_SHMEM)) THEN
       RESPONSEFUN_MEMORY_CACHED=1E-6_q*N1*N2*16/COMM_SHMEM%NCPU
    ENDIF
#endif
    RESPONSEFUN_MEMORY_CACHED=RESPONSEFUN_MEMORY_CACHED+1E-6_q*NALLOC*MAXCACHE*16
    
  END FUNCTION RESPONSEFUN_MEMORY_CACHED


!
! this version allows to distribute data of nodes, for instance
! along the row or column index used by chi_GG.F
!


  SUBROUTINE ALLOCATE_RESPONSEFUN_DISTRI( CHI, NALLOC_ROW, NALLOC_COL, LGAMMA, LREALSTORE, NOMEGA_CHI)
    USE ini
    TYPE (responsefunction) :: CHI
    LOGICAL :: LGAMMA       ! special mode using sin and cosine transforms
    LOGICAL :: LREALSTORE   ! special mode using real valued responsefunctions
    INTEGER, INTENT(IN)  :: NALLOC_ROW, NALLOC_COL, NOMEGA_CHI
  ! local
    INTEGER :: N1, N2

    N1=NALLOC_ROW
    N2=NALLOC_COL
    IF (LGAMMA .AND. .NOT. LREALSTORE) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! this gives 2*NALLOC words in each direction
    ! the response function is complex
    ! this is usually used for real frequencies
       N1=NALLOC_ROW*2
       N2=NALLOC_COL*2
    ELSE IF (LGAMMA) THEN
    ! real valued charge densities (gamma only version)
    ! the complex coefficients are interpreted as sin and cosine transforms
    ! the response function is stored as a real valued function
    ! if a complex array is allocted its first dimension can equal NALLOC
    ! since one complex number allows to store two real numbers
    ! this is usually used for complex frequencies (or at w=0)
       N1=NALLOC_ROW
       N2=NALLOC_COL*2
    ENDIF
    CHI%LREAL=LGAMMA
    CHI%LREALSTORE=LREALSTORE
    CHI%NP1=N1
    CHI%NP2=N2

    ALLOCATE(CHI%RESPONSEFUN(N1, N2, NOMEGA_CHI), & 
         CHI%WING(N1, 1:3, NOMEGA_CHI), &
         CHI%CWING(N1, 1:3, NOMEGA_CHI), &
         CHI%HEAD(1:3, 1:3, NOMEGA_CHI), &
         CHI%COMEGA(NOMEGA_CHI),CHI%OMEGA(NOMEGA_CHI))

    CALL REGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN), "response")

    ! tricky stuff make the RESPONSER entry equivalent to the RESPONSEFUN
    ! entry the number of real values is twice as large as the number of 
    ! complex entries in the first direction
    CALL SET_RESPONSER( CHI%RESPONSER, 2*N1, N2, NOMEGA_CHI, CHI%RESPONSEFUN(1,1,1))
    CALL SET_RESPONSER( CHI%WINGR,     2*N1, 3,  NOMEGA_CHI, CHI%WING(1,1,1))
    CALL SET_RESPONSER( CHI%CWINGR,    2*N1, 3,  NOMEGA_CHI, CHI%CWING(1,1,1))

    NULLIFY(CHI%STORAGE, CHI%WEIGHT, CHI%STPOS, CHI%W_INTER, CHI%NCACHE)
    CHI%MAXCACHE=0
    CHI%NOMEGA=NOMEGA_CHI
    CHI%LSHMEM=.FALSE.
    CHI%LLEAD =.TRUE.
    NULLIFY(CHI%COMM_SHMEM)

  END SUBROUTINE ALLOCATE_RESPONSEFUN_DISTRI

!
!
  SUBROUTINE ALLOCATE_RESPONSEFUN_CACHE( CHI, MAXCACHE)
    USE ini
    TYPE (responsefunction) :: CHI
    INTEGER, INTENT(IN)  :: MAXCACHE

    INTEGER :: NALLOC, NOMEGA_CHI


    NALLOC= SIZE(CHI%RESPONSEFUN,1)
    ! try to determine the original NGDIM (undo what was done in ALLOCATE_RESPONSEFUN)
    IF (CHI%LREAL .AND. .NOT. CHI%LREALSTORE) THEN
       NALLOC=NALLOC/2
    ELSE IF (CHI%LREAL) THEN
       ! here I am not sure, I think it is not allowed to halve the value in this case
       ! leave as is
    ENDIF
    NOMEGA_CHI= SIZE(CHI%RESPONSEFUN,3)

    ALLOCATE( &
         CHI%NCACHE(NOMEGA_CHI), &
         CHI%STORAGE( NALLOC, MAXCACHE, NOMEGA_CHI), &
         CHI%WEIGHT(MAXCACHE, NOMEGA_CHI), &
         CHI%STPOS (MAXCACHE, 7, NOMEGA_CHI), &
         CHI%W_INTER (MAXCACHE, 2, NOMEGA_CHI))
    CHI%MAXCACHE=MAXCACHE
    CHI%NCACHE = 0  ! initialize cache to zero

    CALL REGISTER_ALLOCATE(16._q*SIZE(CHI%STORAGE), "response")


  END SUBROUTINE ALLOCATE_RESPONSEFUN_CACHE

!
!
  SUBROUTINE DEALLOCATE_RESPONSEFUN( CHI)
    USE ini
    USE c2f_interface
    TYPE (responsefunction) :: CHI
!   local
    INTEGER :: ierror

    IF (.NOT. ASSOCIATED(CHI%RESPONSEFUN)) RETURN
#ifdef use_shmem
    IF (CHI%LSHMEM) THEN
       CALL DEREGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN)/CHI%COMM_SHMEM%NCPU, "response")
       CALL DETACHSHMEM(c_loc(CHI%RESPONSEFUN(1,1,1)))

       ! to be sure no one is still using the shmem segment
       ! note we have already destroyed the shmem above 
       ! if SHMEMLOCK is -1, the semaphore was not acutally allocated
       CALLMPI( M_barrier(CHI%COMM_SHMEM))
       IF (CHI%LLEAD .AND. CHI%SHMEMLOCK/=-1) THEN
       !   CALL DESTROYSHMEM(CHI%SHMID)
          CALL DESTROYSEM(CHI%SHMEMLOCK )
       ENDIF
       CALLMPI( M_barrier(CHI%COMM_SHMEM))
       NULLIFY(CHI%RESPONSEFUN, CHI%COMM_SHMEM)

       IF (ASSOCIATED(CHI%COMM_INTER_SHMEM)) THEN
         CALLMPI( MPI_COMM_FREE(CHI%COMM_INTER_SHMEM%MPI_COMM, ierror))
         DEALLOCATE(CHI%COMM_INTER_SHMEM)
         NULLIFY(CHI%COMM_INTER_SHMEM)
       ENDIF
    ELSE
#endif
    CALL DEREGISTER_ALLOCATE(16._q*SIZE(CHI%RESPONSEFUN), "response")
    DEALLOCATE(CHI%RESPONSEFUN)
#ifdef use_shmem
    ENDIF
#endif

    DEALLOCATE(CHI%WING, CHI%CWING, CHI%HEAD, CHI%COMEGA, CHI%OMEGA )
    NULLIFY(CHI%RESPONSEFUN, CHI%WING, CHI%CWING, CHI%HEAD)

    IF (ASSOCIATED(CHI%STORAGE)) THEN
       CALL DEREGISTER_ALLOCATE(16._q*SIZE(CHI%STORAGE), "response")
       DEALLOCATE(CHI%STORAGE, CHI%WEIGHT, CHI%STPOS, CHI%W_INTER, CHI%NCACHE)
       NULLIFY(CHI%STORAGE, CHI%WEIGHT, CHI%STPOS, CHI%W_INTER)
    END IF

  END SUBROUTINE DEALLOCATE_RESPONSEFUN


!***********************************************************************
!
! determine local slot if the global storage index is known
!
!***********************************************************************

  SUBROUTINE DETERMINE_LOCAL_SLOT( CHI, NOMEGA, NOMEGA_LOCAL )
    TYPE (responsefunction) CHI
    INTEGER :: NOMEGA               ! global storage position in the frequency array
    INTEGER :: NOMEGA_LOCAL         ! local storage position in CHI

    IF (NOMEGA>= CHI%NOMEGA_LOW .AND. NOMEGA<= CHI%NOMEGA_HIGH) THEN
       NOMEGA_LOCAL=NOMEGA-CHI%NOMEGA_LOW+1
    ELSE
       NOMEGA_LOCAL=-1
    ENDIF

  END SUBROUTINE DETERMINE_LOCAL_SLOT


!***********************************************************************
!
! the ADD_RESPONSEFUNCTION_CACHE routine copies CHARGE and WEIGHT
! to the CHI%CACHE and CHI%WEIGHT at the storage elements NOMEGA
! if the cache line is full CLEAN_RESPONSEFUNCTION_CACHE is first
! called to clean the cache and then the entry is added
!
!
!***********************************************************************

  SUBROUTINE ADD_RESPONSEFUNCTION_CACHE(CHI, GCHG, WEIGHT, NOMEGA, NP)
    USE constant
    TYPE (responsefunction) CHI
    COMPLEX(q) :: GCHG(:)

    COMPLEX(q) :: WEIGHT
    INTEGER :: NOMEGA
    INTEGER :: NP


    PROFILING_START('add_responsefunction_cache')

    IF (.NOT. ASSOCIATED(CHI%NCACHE) ) THEN
       WRITE(0,*) 'internal error in ADD_RESPONSEFUNCTION_CACHE: cache structure not allocated'
       WRITE(0,*) ' search in chi.F and comment in ALLOCATE_RESPONSEFUN_CACHE'
       STOP
    ENDIF


    IF (CHI%NCACHE(NOMEGA)/=0) THEN
       IF (NP/=CHI%NP) THEN
          WRITE(0,*) 'internal error in ADD_RESPONSEFUNCTION_CACHE: NP changed',NP, CHI%NP
          STOP
       ENDIF
    ENDIF
    CHI%NP=NP

    IF (CHI%NCACHE(NOMEGA)>= SIZE(CHI%WEIGHT,1)) THEN
       CALL CLEAN_RESPONSEFUNCTION_CACHE( CHI, NOMEGA )
    ENDIF

    CHI%NCACHE(NOMEGA)=CHI%NCACHE(NOMEGA)+1

    CALL ZCOPY( NP, GCHG(1), 1, CHI%STORAGE(1,CHI%NCACHE(NOMEGA),NOMEGA), 1)
    CHI%WEIGHT(CHI%NCACHE(NOMEGA), NOMEGA)= WEIGHT

    PROFILING_STOP('add_responsefunction_cache')

  END SUBROUTINE ADD_RESPONSEFUNCTION_CACHE


!***********************************************************************
!
! the CLEAN_RESPONSEFUNCTION_CACHE routine updates the CHI%RESPONSEFUN
! using the remaining data stored in the CHI%CACHE
! 
!
!*************************************************************************

  SUBROUTINE CLEAN_RESPONSEFUNCTION_CACHE( CHI, NOMEGA1, NOMEGA2)
    USE constant
    USE c2f_interface, ONLY : LOCKSEM,UNLOCKSEM
    TYPE (responsefunction) CHI
    INTEGER    :: NOMEGA1
    INTEGER, OPTIONAL :: NOMEGA2
    COMPLEX(q) :: GWORK(SIZE(CHI%RESPONSEFUN,1),CHI%MAXCACHE)
    INTEGER NOMEGA, NOMEGA2_ACTUAL, I, N
 

    PROFILING_START('clean_responsefunction_cache')

    IF (.NOT. ASSOCIATED(CHI%NCACHE)) THEN
       PROFILING_STOP('clean_responsefunction_cache')
       RETURN
    ENDIF


    IF (PRESENT(NOMEGA2)) THEN
       NOMEGA2_ACTUAL=NOMEGA2
    ELSE
       NOMEGA2_ACTUAL=NOMEGA1
    ENDIF

    DO NOMEGA=NOMEGA1,NOMEGA2_ACTUAL
       IF (CHI%NCACHE(NOMEGA)>0) THEN
          DO N=1,CHI%NCACHE(NOMEGA)
             IF (CHI%LREALSTORE) THEN
                ! multiply real valued vector with NP data points by real weight
                GWORK(1:CHI%NP,N)=0
                CALL DAXPY( CHI%NP, CHI%WEIGHT(N,NOMEGA), CHI%STORAGE(1,N,NOMEGA), 1,  GWORK(1,N),1)
             ELSE IF (CHI%LREAL) THEN
                ! multiply real valued vector with 2 NP data points by complex weight
                GWORK(1:2*CHI%NP,N)=0
                CALL ZDAXPY( 2*CHI%NP, CHI%WEIGHT(N,NOMEGA), CHI%STORAGE(1,N,NOMEGA), 1,  GWORK(1,N),1)
             ELSE
                ! multiply complex valued vector with NP data points by complex weight
                GWORK(1:CHI%NP,N)=0
                CALL ZAXPY( CHI%NP, CHI%WEIGHT(N,NOMEGA), CHI%STORAGE(1,N,NOMEGA), 1,  GWORK(1,N),1)
             ENDIF
          ENDDO
          
#ifdef use_shmem
          IF (CHI%LSHMEM) CALL LOCKSEM( CHI%SHMEMLOCK, NOMEGA)
#endif
          IF (CHI%LREALSTORE) THEN
             CALL DGEMM('N','T', 2*CHI%NP, 2*CHI%NP, CHI%NCACHE(NOMEGA), 1.0_q, &
                  GWORK(1,1), 2*SIZE(GWORK,1), CHI%STORAGE(1,1,NOMEGA), 2*SIZE(CHI%STORAGE,1), &
                  1.0_q, CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1))
          ELSE IF (CHI%LREAL) THEN
             CALL DGEMM('N','T', 4*CHI%NP, 2*CHI%NP, CHI%NCACHE(NOMEGA), 1.0_q, &
                  GWORK(1,1), 2*SIZE(GWORK,1), CHI%STORAGE(1,1,NOMEGA), 2*SIZE(CHI%STORAGE,1), &
                  1.0_q, CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1))
          ELSE
             CALL ZGEMM('N','C', CHI%NP, CHI%NP, CHI%NCACHE(NOMEGA), (1.0_q, 0.0_q), &
                  GWORK(1,1), SIZE(GWORK,1), CHI%STORAGE(1,1,NOMEGA), SIZE(CHI%STORAGE,1), &
                  (1.0_q,0.0_q), CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1))
          ENDIF
#ifdef use_shmem
          IF (CHI%LSHMEM) CALL UNLOCKSEM( CHI%SHMEMLOCK, NOMEGA)
#endif
       ENDIF
       CHI%NCACHE(NOMEGA)=0
    ENDDO

    PROFILING_STOP('clean_responsefunction_cache')

  END SUBROUTINE CLEAN_RESPONSEFUNCTION_CACHE


!***********************************************************************
!
! the ADD_RESPONSEFUNCTION_INT routine copies the overlap charge density
!    1/N  \sum_r' u_1(r') u*_2(r') -iG'r'
! and a set of seven integers into the CACHE structure
! these integers usually correspond to the k and band indices
! the following integers determine where the final integral needs to be
! stored (in SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:,:,:)
!  I1    NQ    index of difference between k-points q= K1- K2
!  I2    NK1   index of k-point K1
!  I3    NB1   band index N1 
!  I4    NB2   band index N2
!  I5    ISP   spin index (applies to both state 1 and 2 in the RPA)
!  I6          frequency point where the final result is stored
!              in SCREENED_TWO_ELECTRON_INTEGRALS !
!  I7    NK2   index of k-point K2
! 
!
!***********************************************************************

  SUBROUTINE ADD_RESPONSEFUNCTION_INT(WDESQ, CHI, GCHG, NOMEGA, NP, & 
        S2E, SCREENED_TWO_ELECTRON_INTEGRALS, WCACHE, I1, I2, I3, I4, I5, I6, I7 , &
        W_INTER )
    USE constant
    USE wave_cacher
    TYPE (wavedes1) WDESQ
    TYPE (responsefunction) CHI
    COMPLEX(q) :: GCHG(:)
    INTEGER :: NOMEGA
    INTEGER :: NP
    TYPE (screened_2e_handle) S2E
    COMPLEX(qs) :: SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:,:,:)
    TYPE (wave_cache), POINTER :: WCACHE

    INTEGER :: I1, I2,  I3,  I4,  I5,  I6,     I7
    !          NQ, NK1, NB1, NB2, ISP, NOMEGA, NK2
    REAL(q) :: W_INTER(2)


    PROFILING_START('add_responsefunction_int')

    IF (.NOT.ASSOCIATED(CHI%NCACHE)) THEN
       WRITE(0,*) 'internal error in ADD_RESPONSEFUNCTION_INT: cache not allocated'
       WRITE(0,*) ' search in chi.F and comment in ALLOCATE_RESPONSEFUN_CACHE'
       STOP
    ENDIF

    IF (CHI%NCACHE(NOMEGA)/=0) THEN
       IF (NP/=CHI%NP) THEN
          WRITE(0,*) 'internal error in ADD_RESPONSEFUNCTION_CACHE_INT: NP changed',NP, CHI%NP
          STOP
       ENDIF
    ENDIF
    CHI%NP=NP

    IF (CHI%NCACHE(NOMEGA)>= SIZE(CHI%WEIGHT,1)) THEN
       CALL CLEAN_RESPONSEFUNCTION_INT( WDESQ, CHI, S2E, SCREENED_TWO_ELECTRON_INTEGRALS, WCACHE, NOMEGA )
    ENDIF

    CHI%NCACHE(NOMEGA)=CHI%NCACHE(NOMEGA)+1

    CALL ZCOPY( NP, GCHG(1), 1,CHI%STORAGE(1,CHI%NCACHE(NOMEGA),NOMEGA), 1)
    CHI%STPOS(CHI%NCACHE(NOMEGA), 1, NOMEGA)= I1  ! NQ
    CHI%STPOS(CHI%NCACHE(NOMEGA), 2, NOMEGA)= I2  ! NK1
    CHI%STPOS(CHI%NCACHE(NOMEGA), 3, NOMEGA)= I3  ! NB1_INTO_TOT
    CHI%STPOS(CHI%NCACHE(NOMEGA), 4, NOMEGA)= I4  ! NB2 
    CHI%STPOS(CHI%NCACHE(NOMEGA), 5, NOMEGA)= I5  ! ISP
    CHI%STPOS(CHI%NCACHE(NOMEGA), 6, NOMEGA)= I6  ! NOMEGA "frequency" index where the final result ist stored
    CHI%STPOS(CHI%NCACHE(NOMEGA), 7, NOMEGA)= I7  ! NK2
    CHI%W_INTER(CHI%NCACHE(NOMEGA), 1:2, NOMEGA)= W_INTER

    PROFILING_STOP('add_responsefunction_int')

  END SUBROUTINE ADD_RESPONSEFUNCTION_INT

!***********************************************************************
!
! the CLEAN_RESPONSEFUNCTION_INT routine updates the screened
! two electron tables using the data stored in the CHI%CACHE
! the cache lines between NOMEGA1 and NOMEGA2 are
! "cleaned" e.g. the required integrals are calculated
! and stored in SCREENED_TWO_ELECTRON_INTEGRALS
! if NOMEGA2 is missing only the cache line NOMEGA1 is cleared
! 
!
!*************************************************************************

  SUBROUTINE CLEAN_RESPONSEFUNCTION_INT( WDESQ, CHI, S2E, SCREENED_TWO_ELECTRON_INTEGRALS , WCACHE, NOMEGA1, NOMEGA2)
    USE constant
    USE wave_cacher
    TYPE (wavedes1) :: WDESQ  
    TYPE (responsefunction) CHI
    INTEGER    :: NOMEGA1
    INTEGER, OPTIONAL :: NOMEGA2
  ! local
    COMPLEX(q) :: GWORK(SIZE(CHI%RESPONSEFUN,1),CHI%MAXCACHE)
    INTEGER    :: NOMEGA, NOMEGA2_ACTUAL
    INTEGER I, N
    TYPE (screened_2e_handle) S2E
    COMPLEX(qs) :: SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:,:,:)
    TYPE (wave_cache), POINTER :: WCACHE
    COMPLEX(q), EXTERNAL ::  ZDOTC, ZDDOTC
    REAL(q), EXTERNAL :: DDOT
    COMPLEX(q) :: Z
    INTEGER  :: NQ, NK1, K1_IN_IRZ, NK2, K2_IN_IRZ, NB1, NB2, ISP, NOMEGA_IN_SCREENED_TWO_E, K2_STORE_INDEX


    PROFILING_START('clean_responsefunction_int')

    IF (.NOT. ASSOCIATED(CHI%NCACHE)) THEN
       PROFILING_STOP('clean_responsefunction_int')
       RETURN
    ENDIF

    IF (PRESENT(NOMEGA2)) THEN
       NOMEGA2_ACTUAL=NOMEGA2
    ELSE
       NOMEGA2_ACTUAL=NOMEGA1
    ENDIF

    DO NOMEGA=NOMEGA1,NOMEGA2_ACTUAL
       IF (CHI%NCACHE(NOMEGA)>0) THEN
          IF (CHI%LREALSTORE) THEN
!      sum_G' RESPONSEFUN(G',G,omega) 1/N  \sum_r' u_1(r') u*_2(r') -iG'r'

             CALL DGEMM('N','N', 2*CHI%NP, CHI%NCACHE(NOMEGA), 2*CHI%NP, 1.0_q, &
                  CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1), & 
                  CHI%STORAGE(1,1,NOMEGA), 2*SIZE(CHI%STORAGE,1), &
                  0.0_q, GWORK(1,1), 2*SIZE(GWORK,1) )
          ELSE IF (CHI%LREAL) THEN
             CALL DGEMM('N','N', 4*CHI%NP, CHI%NCACHE(NOMEGA), 2*CHI%NP, 1.0_q, &
                  CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1), & 
                  CHI%STORAGE(1,1,NOMEGA), 2*SIZE(CHI%STORAGE,1), &
                  0.0_q, GWORK(1,1), 2*SIZE(GWORK,1) )
          ELSE
             CALL ZGEMM('T','N', CHI%NP, CHI%NCACHE(NOMEGA), CHI%NP, (1.0_q,0.0_q), &
                  CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1), & 
                  CHI%STORAGE(1,1,NOMEGA), SIZE(CHI%STORAGE,1), &
                  (0.0_q,0.0_q), GWORK(1,1), SIZE(GWORK,1) )
          ENDIF
          DO N=1,CHI%NCACHE(NOMEGA)
             NQ =CHI%STPOS(N, 1, NOMEGA)
             NK1=CHI%STPOS(N, 2, NOMEGA)
             NB1=CHI%STPOS(N, 3, NOMEGA)
             NK2=CHI%STPOS(N, 7, NOMEGA)
             NB2=CHI%STPOS(N, 4, NOMEGA)
             ISP=CHI%STPOS(N, 5, NOMEGA)
             K1_IN_IRZ=S2E%K1_IN_IRZ(NQ, NK1)
             K2_IN_IRZ=S2E%K2_IN_IRZ(NQ, NK1)
! using these lines various contributions can be excluded (debugging)
!             IF (NB1/=1 .OR. NB2 /= 1 ) THEN
!                CALL STORE_GW_ACC( WCACHE, WDESQ, GWORK(:,N), CHI%STORAGE(:,N,NOMEGA), (0.0_q.0.0_q), & 
!                     NB1=NB1, NK1=NK1, NB2=NB2, NK2=NK2, ISP=ISP, &
!                     W_INTER=CHI%W_INTER(N, :, NOMEGA)*0)
!                CYCLE
!             ENDIF
!  multiply by  1/N (\sum_r  u_1(r)  u*_2(r)  -iGr  )*
             IF (CHI%LREALSTORE) THEN
                Z=DDOT(  2*CHI%NP, GWORK(1,N), 1, CHI%STORAGE(1,N,NOMEGA), 1)
             ELSE IF (CHI%LREAL) THEN
                Z=ZDDOTC(2*CHI%NP, GWORK(1,N), 1, CHI%STORAGE(1,N,NOMEGA), 1)
             ELSE
                Z=ZDOTC(   CHI%NP, GWORK(1,N), 1, CHI%STORAGE(1,N,NOMEGA), 1)
             ENDIF
             
             ! GW for eigenvalues only
             NOMEGA_IN_SCREENED_TWO_E=CHI%STPOS(N, 6, NOMEGA)
             ! if NOMEGA_IN_SCREENED_TWO_E is set store it in 
             ! corresponding storage position in SCREENED_TWO_ELECTRON_INTEGRALS
             IF (NOMEGA_IN_SCREENED_TWO_E>=0) THEN
             !   CALL ENTER_IN_SCREENED_2E( S2E, & 
             !        NQ, NK1, NB1, NB2, & 
             !        SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:, ISP,NOMEGA_IN_SCREENED_TWO_E), &
             !        Z )
                K2_STORE_INDEX=S2E%K2_STORE_INDEX(K1_IN_IRZ, K2_IN_IRZ)
                SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, NOMEGA_IN_SCREENED_TWO_E)=Z+ &
                     SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, NOMEGA_IN_SCREENED_TWO_E)
             ELSE
             ! if not we are using the spectral method and only sum
             ! the contributions to the self energy and its derivative
             ! the weights are stored in W_INTER
             !   CALL ENTER_IN_SCREENED_2E( S2E, & 
             !        NQ, NK1, NB1, NB2, & 
             !        SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:, ISP,1), &
             !        Z*CHI%W_INTER(N, 1, NOMEGA) )
             !   CALL ENTER_IN_SCREENED_2E( S2E, & 
             !        NQ, NK1, NB1, NB2, & 
             !        SCREENED_TWO_ELECTRON_INTEGRALS(:,:,:,:, ISP,2), &
             !        Z*CHI%W_INTER(N, 2, NOMEGA) )
             ! equivalent inlined code
                K2_STORE_INDEX=S2E%K2_STORE_INDEX(K1_IN_IRZ, K2_IN_IRZ)
                SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, 1)=Z*CHI%W_INTER(N, 1, NOMEGA)+ &
                     SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, 1)
                SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, 2)=Z*CHI%W_INTER(N, 2, NOMEGA)+ &
                     SCREENED_TWO_ELECTRON_INTEGRALS(NB1, K1_IN_IRZ, NB2, K2_STORE_INDEX, ISP, 2)

             ENDIF
             
             ! for scQPGW store 
             ! sum_G 1/N  u_2(r) 
             !   e^iG r sum_G' 1/N (\sum_r  u_1(r') u*_2(r') -iG'r' ) RESPONSEFUN(G',G,omega)
             CALL STORE_GW_ACC( WCACHE, WDESQ, GWORK(:,N), CHI%STORAGE(:,N,NOMEGA), Z, & 
                  NB1=NB1, NK1=NK1, NB2=NB2, NK2=NK2, ISP=ISP, W_INTER=CHI%W_INTER(N,:,NOMEGA)*S2E%WTKPT(K1_IN_IRZ, K2_IN_IRZ))
          ENDDO
       ENDIF
       CHI%NCACHE(NOMEGA)=0
    ENDDO

    PROFILING_STOP('clean_responsefunction_int')

  END SUBROUTINE CLEAN_RESPONSEFUNCTION_INT

!*********************************************************************
!
! helper routine:
! progress counter (roughly 20 dots per step)
! band index N (maximum value NMAX) and k-point index K (maximum value
! KMAX) must be supplied
!
!*********************************************************************

  SUBROUTINE GWPROGRESS(IU0, K, KMAX, N, NMAX)
    INTEGER, INTENT(IN) :: IU0, K, KMAX, N, NMAX
    REAL(q) :: FRAC
    REAL(q), SAVE :: FRAC_DONE=1
    INTEGER :: FRAC_INT=0
    CHARACTER (LEN=1) S

    FRAC=(FLOAT(K-1)*NMAX+FLOAT(N))/KMAX/NMAX

    IF (FRAC<FRAC_DONE) THEN
       FRAC_DONE=-1/20._q
       FRAC_INT=0
    ELSE IF ( FRAC_DONE+1/20._q<FRAC ) THEN
       DO
          IF ( FRAC_INT>=20 .OR. IU0<0 ) EXIT
          IF ( FRAC_DONE+1/20._q >=FRAC) EXIT
          SELECT CASE(FRAC_INT)
            CASE (0)  ; S='|'
            CASE (10)  ; S='|'
            CASE DEFAULT ; S='.'
          END SELECT
          WRITE(IU0,'(A1)',ADVANCE='NO') S
          FRAC_INT=FRAC_INT+1
          FRAC_DONE=FRAC_DONE+1/20._q
       ENDDO
    ENDIF
  END SUBROUTINE GWPROGRESS
 
  
!*********************************************************************
!
! This routine calculates the transformation tables (the weights), 
! used to obtain the real and imaginary part of the polarizability
!
!   int dw' X(w')(1/(w-w'- i eta)- (1/( w + w' -+ i eta) =
!   int dw' X(w')(1/(w-w'- i eta)+ (1/(-w - w' +- i eta) =
! 
! Equ.  (18)  Shishkin, Kresse, PRB 74, 035101
! We usually use the retarded or causal response function to
! obtain the real part (negative sign in first equation)
! (see comments on top of the file)
! as before all poles are above the real axis
!
! the "proper" time ordered version can be selected by 
! #define timeordered
!
! here a finite element basis is used to calculate the
! transformation matrix:
! triangular finite elements are introduced at w'
! (Equ. (17) Shishkin, Kresse, PRB 74, 035101 )
! the contributions are Hilbert transformed 
! note that the retarded causal version looses all weight at w'=0
! whereas the time ordered version does not
!
!*********************************************************************    
  
  
  SUBROUTINE SETUP_KRAM_KRON_TABLE(OMEGA,OMEGAR,SHIFT,TABLE,I1)
    REAL(q) :: OMEGAR, SHIFT,DELTA,ISIGN 
    REAL(q) :: SHIFT1,SHIFT2
    REAL(q) :: OMEGA(:)
    INTEGER :: I1, I_R, K
    COMPLEX(q) :: TABLE(:)

    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus,WEIGHT

    REAL(q) :: X, XP

    COMPLEX(q) :: Integral(4)
    COMPLEX(q) :: AA, BB
    INTEGER :: L

    ISIGN = 1
    SHIFT1 = SHIFT*ISIGN
    SHIFT2 = abs(SHIFT)

    OMEGAQ = OMEGA(I1)

    XP = OMEGAR

    Integral = (0.0_q,0.0_q)
    DO K=1,4
       IF (I1>1) THEN
          WEIGHT=1
          IF (I1==SIZE(OMEGA)) WEIGHT=2 ! to avoid loosing spectral weight at last point
          ! is counted twice (triangle between OMEGAQ_minus and QMEGAQ)

          OMEGAQ_minus = OMEGA(I1-1)
          IF (K<=2) THEN
             IF (K==1) THEN
                X  = OMEGAQ_minus
             ELSE 
                X  = OMEGAQ
             ENDIF
             AA = (0.0_q,0.0_q)

             ISIGN = -1
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

             ISIGN = -1
#ifdef timeordered
             ISIGN = 1
#endif
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA + (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 

             Integral(K) = -OMEGAQ_minus/(OMEGAQ - OMEGAQ_minus)*AA*WEIGHT

             BB = (0.0_q,0.0_q)

             ISIGN = -1
             SHIFT1 =SHIFT*ISIGN
             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)

             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  


             ISIGN = -1
#ifdef timeordered
             ISIGN = 1
#endif
             SHIFT1 = SHIFT*ISIGN
             BB = BB - (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB + (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)

             BB = BB + (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB - X + XP*LOG((X+XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)  

             Integral(K) = Integral(K)+1.0/(OMEGAQ - OMEGAQ_minus)*BB*WEIGHT
          ENDIF
       ENDIF

       IF (I1<SIZE(OMEGA)) THEN
          OMEGAQ_plus  = OMEGA(I1+1)
          IF (K>2) THEN
             IF (K==3) THEN
                X  = OMEGAQ
             ELSE 
                X  = OMEGAQ_plus
             ENDIF

             AA = (0.0_q,0.0_q)

             ISIGN = -1
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

             ISIGN = -1
#ifdef timeordered
             ISIGN = 1
#endif
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA + (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 

             Integral(K) = OMEGAQ_plus/(OMEGAQ_plus - OMEGAQ)*AA

             ISIGN = -1
             SHIFT1 = SHIFT*ISIGN
             BB = (0.0_q,0.0_q)
             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)
             
             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  
             

             ISIGN = -1
#ifdef timeordered
             ISIGN = 1
#endif
             SHIFT1 = SHIFT*ISIGN
             BB = BB - (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB + (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)
             
             BB = BB + (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB - X + XP*LOG((X+XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)   
             
             Integral(K) = Integral(K) - 1.0/(OMEGAQ_plus - OMEGAQ)*BB
          ENDIF
       ENDIF
    ENDDO
       

    TABLE(I1) = Integral(2)-Integral(1)+Integral(4)-Integral(3)
  END SUBROUTINE SETUP_KRAM_KRON_TABLE

!*********************************************************************
!
! This routine calculates transformation tables (the weights), 
! used to obtain the real and imaginary part of the polarizability
! considering the resonant (positive frequency) part only
! 
!   int dw' X(w')(1/(w-w'- i eta)
!
!*********************************************************************    
  
  SUBROUTINE SETUP_KRAM_KRON_TABLE_RES(OMEGA,OMEGAR,SHIFT,TABLE,I1)
    REAL(q) :: OMEGAR, SHIFT,DELTA,ISIGN 
    REAL(q) :: SHIFT1,SHIFT2
    REAL(q) :: OMEGA(:)
    INTEGER :: I1, I_R, K
    COMPLEX(q) :: TABLE(:)

    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus,WEIGHT

    REAL(q) :: X, XP

    COMPLEX(q) :: Integral(4)
    COMPLEX(q) :: AA, BB
    INTEGER :: L

    ISIGN = 1
    SHIFT1 = SHIFT*ISIGN
    SHIFT2 = abs(SHIFT)

    OMEGAQ = OMEGA(I1)

    XP = OMEGAR

    Integral = (0.0_q,0.0_q)
    DO K=1,4
       IF (I1>1) THEN
          WEIGHT=1
          IF (I1==SIZE(OMEGA)) WEIGHT=2 ! to avoid loosing spectral weight last point
          ! is counted twice (triangle between OMEGAQ_minus and QMEGAQ)

          OMEGAQ_minus = OMEGA(I1-1)
          IF (K<=2) THEN
             IF (K==1) THEN
                X  = OMEGAQ_minus
             ELSE 
                X  = OMEGAQ
             ENDIF
             AA = (0.0_q,0.0_q)

             ISIGN = -1
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

!             ISIGN = 1
!             SHIFT1 = SHIFT*ISIGN
!             AA = AA - 0.5*LOG((X+XP)**2+SHIFT1**2)
!             AA = AA + (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 

             Integral(K) = -OMEGAQ_minus/(OMEGAQ - OMEGAQ_minus)*AA*WEIGHT

             BB = (0.0_q,0.0_q)

             ISIGN = -1.0
             SHIFT1 = SHIFT*ISIGN
             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)

             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  


!             ISIGN = 1
!             SHIFT1 = SHIFT*ISIGN
!             BB = BB - (XP/2)*LOG((X+XP)**2+SHIFT1**2)
!             BB = BB + (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)
!
!             BB = BB + (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
!             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2
!
!             BB = BB - X + XP*LOG((X+XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
!             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)  

             Integral(K) = Integral(K)+1.0/(OMEGAQ - OMEGAQ_minus)*BB*WEIGHT
          ENDIF
       ENDIF

       
       IF (I1<SIZE(OMEGA)) THEN
          OMEGAQ_plus  = OMEGA(I1+1)
          IF (K>2) THEN
             IF (K==3) THEN
                X  = OMEGAQ
             ELSE 
                X  = OMEGAQ_plus
             ENDIF

             AA = (0.0_q,0.0_q)
             
             ISIGN = -1.0
             SHIFT1 = SHIFT*ISIGN
             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

!          ISIGN = 1.0
!          SHIFT1 = SHIFT*ISIGN
!          AA = AA - 0.5*LOG((X+XP)**2+SHIFT1**2)
!          AA = AA + (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 


             Integral(K) = OMEGAQ_plus/(OMEGAQ_plus - OMEGAQ)*AA

             ISIGN = -1.0
             SHIFT1 = SHIFT*ISIGN
             BB = (0.0_q,0.0_q)
             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)

             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  

!          ISIGN = 1.0
!          SHIFT1 = SHIFT*ISIGN
!          BB = BB - (XP/2)*LOG((X+XP)**2+SHIFT1**2)
!          BB = BB + (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)
!
!          BB = BB + (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
!          BB = BB - (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2
!
!          BB = BB - X + XP*LOG((X+XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
!          BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)   

             Integral(K) = Integral(K) - 1.0/(OMEGAQ_plus - OMEGAQ)*BB
          ENDIF
       ENDIF
    ENDDO

    TABLE(I1) = Integral(2)-Integral(1)+Integral(4)-Integral(3)
  END SUBROUTINE SETUP_KRAM_KRON_TABLE_RES

#ifdef numerical_kramkron
!*********************************************************************
!
! set up the tables for the Kramers Kronig relation of the 
! response function using numerical quadrature (for testing only)
! this calculated the time ordered transformation
!
!*********************************************************************  
  
  SUBROUTINE SETUP_KRAM_KRON_TABLE_NUM1(OMEGA,OMEGAR,SHIFT,TABLE,I)
    REAL(q) :: OMEGAR, SHIFT, ISIGN
    REAL(q) :: OMEGA(:)
    INTEGER :: I, I_R, K
    COMPLEX(q) :: TABLE(:)
    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus
    REAL(q) :: X, XP
    COMPLEX(q) :: FF
    REAL(q), PARAMETER :: STEP=0.001

    OMEGAQ = OMEGA(I)
    IF (I>1) THEN
       OMEGAQ_minus = OMEGA(I-1)
    ELSE
       OMEGAQ_minus =-OMEGA(2)
    ENDIF
    OMEGAQ_plus  = OMEGA(I+1)

    ISIGN =1.0
    XP = OMEGAR

    TABLE(I)=0
    IF (I>1) THEN
       DO X = OMEGAQ_minus, OMEGAQ, STEP
          FF = (X - OMEGAQ_minus)/(OMEGAQ - OMEGAQ_minus)
          TABLE(I) = TABLE(I) +FF/(-X+XP-ISIGN*CMPLX(0,SHIFT,q))  - &
               FF/(X+XP+ISIGN*CMPLX(0,SHIFT,q))
       ENDDO
    ENDIF
    DO X = OMEGAQ, OMEGAQ_plus, STEP
       FF = (OMEGAQ_plus - X)/(OMEGAQ_plus - OMEGAQ)
       TABLE(I) = TABLE(I) +FF/(-X+XP-ISIGN*CMPLX(0,SHIFT,q)) - &
            FF/(X+XP+ISIGN*CMPLX(0,SHIFT,q))
    ENDDO

    TABLE(I) = STEP*TABLE(I)
    !    IF (I==1) TABLE(I)=TABLE(I)*2

  END SUBROUTINE SETUP_KRAM_KRON_TABLE_NUM1


  SUBROUTINE SETUP_KRAM_KRON_TABLE_NUM2(OMEGA,OMEGAR,SHIFT,TABLE,I)
    REAL(q) :: OMEGAR, SHIFT, ISIGN
    REAL(q) :: OMEGA(:)
    INTEGER :: I, I_R, K
    COMPLEX(q) :: TABLE(:)
    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus
    REAL(q) :: X, XP
    COMPLEX(q) :: FF
    REAL(q), PARAMETER :: STEP=0.001

    OMEGAQ = OMEGA(I)
    IF (I>1) THEN
       OMEGAQ_minus = OMEGA(I-1)
    ELSE
       OMEGAQ_minus =-OMEGA(2)
    ENDIF
    OMEGAQ_plus  = OMEGA(I+1)

    ISIGN =1.0
    XP = OMEGAR

    TABLE(I)=0
    IF (I>1) THEN
       DO X = OMEGAQ_minus, OMEGAQ, STEP
          FF = (X - OMEGAQ_minus)/(OMEGAQ - OMEGAQ_minus)
          TABLE(I) = TABLE(I) +FF/(-X+XP-ISIGN*CMPLX(0,SHIFT,q))  + &
               FF/(-X-XP-ISIGN*CMPLX(0,SHIFT,q))
       ENDDO
    ENDIF
    DO X = OMEGAQ, OMEGAQ_plus, STEP
       FF = (OMEGAQ_plus - X)/(OMEGAQ_plus - OMEGAQ)
       TABLE(I) = TABLE(I) +FF/(-X+XP-ISIGN*CMPLX(0,SHIFT,q)) + &
            FF/(-X-XP-ISIGN*CMPLX(0,SHIFT,q))
    ENDDO

    TABLE(I) = STEP*TABLE(I)
    !    IF (I==1) TABLE(I)=TABLE(I)*2

  END SUBROUTINE SETUP_KRAM_KRON_TABLE_NUM2
#endif


!*********************************************************************
!
! allocate and set table for Kramers Kronig transformation of
! of the frequency dependent dielectric quantities
!
!   int dw' X(w')(1/(w-w'- i eta)- (1/( w + w' -+ i eta) =
! 
! Equ.  (18)  Shishkin, Kresse, PRB 74, 035101
! (minus sign in first equation -> causal or retarded, + -> timeordered)
!
! when an input function f(2) on the grid is multiplied by the table
! the real part of the input function f(2) (spectrum of function)
! is stored into the imaginary part of the result and multiplied by 2 pi
! the real part of the resulting function f(1) is the Hilbert transform
! of the spectral input function f(2)
!
!  f(ir) = sum_iw TABLE(iw, ir) f(iw)
!
!*********************************************************************

  SUBROUTINE CHI_KRAM_KRON_TABLE( TABLE, OMEGA, SHIFT)
    USE constant
    COMPLEX(q), POINTER :: TABLE(:,:)   ! table for Kramers Kroning 
    REAL(q)    :: OMEGA(:)     ! frequencies
    REAL(q)    :: SHIFT        ! complex shift to be used
  ! local
    INTEGER :: NOMEGA, I_R, IW
    REAL(q) :: SCALE
    COMPLEX(q) :: SUMI

    NOMEGA=SIZE(OMEGA)
    ALLOCATE(  TABLE(NOMEGA, NOMEGA))

    TABLE  = 0

#ifdef numerical_kramkron
    DO IW=1,NOMEGA-1
       DO I_R=1,NOMEGA
          CALL SETUP_KRAM_KRON_TABLE_NUM1(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW)
          CALL SETUP_KRAM_KRON_TABLE_NUM2(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW)
#else
    DO IW=1,NOMEGA
       DO I_R=1,NOMEGA
          CALL SETUP_KRAM_KRON_TABLE(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW)
#endif
       ENDDO
!#define debug
#ifdef debug
       SUMI=0
       DO I_R=1,NOMEGA
          SUMI=SUMI+TABLE(IW,I_R)/PI
       ENDDO
       WRITE(100+IW,'(3F8.4)') (OMEGA(I_R),TABLE(IW,I_R)/PI,I_R=1,NOMEGA)
       WRITE(*,'(F14.7,I4,2F14.7)') OMEGA(IW),IW,SUMI
#endif
    ENDDO

  END SUBROUTINE CHI_KRAM_KRON_TABLE  


!*********************************************************************
!
! allocate and set table for Kramers Kronig transformation of
! of the frequency dependent dielectric quantities 
! resonant part only
!
!*********************************************************************

  SUBROUTINE CHI_KRAM_KRON_RES_TABLE( TABLE, OMEGA, SHIFT)

    COMPLEX(q), POINTER :: TABLE(:,:)   ! table for Kramers Kroning 
    REAL(q)    :: OMEGA(:)     ! frequencies
    REAL(q)    :: SHIFT        ! complex shift to be used
  ! local
    INTEGER :: NOMEGA, I_R, IW

    NOMEGA=SIZE(OMEGA)
    ALLOCATE(  TABLE(NOMEGA, NOMEGA))

    TABLE  = 0
    DO IW=1,NOMEGA
       DO I_R=1,NOMEGA
          CALL SETUP_KRAM_KRON_TABLE_RES(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW) 
       ENDDO
    ENDDO
  END SUBROUTINE CHI_KRAM_KRON_RES_TABLE

!*********************************************************************
!
! set up the tables for the Hilbert transform 
!
!  sigma(w) = i/(2 pi) int d w' W(w')/( w-w' +- i eta) 
!
! where W(w')=W(-w') (as for time ordered functions)
! using analytical integration and a finite element basis
! Equ. (22) in Shishkin, Kresse, PRB 74, 035101
!
! the routine calculates the transformation table only
! for positive frequencies w; for negative frequencies
! the following relation applies
!  sigma(-w, eta) =  -sigma(+w,-eta) 
! since W(w') = W(-w')
!
!*********************************************************************

  SUBROUTINE SETUP_HILBERT_TABLE(OMEGA,OMEGAR,SHIFT,TABLE,I,ISIGN)
    USE constant
    REAL(q) :: OMEGAR, SHIFT,DELTA
    INTEGER :: ISIGN
    REAL(q) :: SHIFT1,SHIFT2
    REAL(q) :: OMEGA(:)
    INTEGER :: I, I_R, K
    COMPLEX(q) :: TABLE(:)

    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus,WEIGHT

    REAL(q) :: X, XP

    COMPLEX(q) :: Integral(4)
    COMPLEX(q) :: AA, BB
    INTEGER :: L


    SHIFT1 = SHIFT*ISIGN
    SHIFT2 = abs(SHIFT)

    OMEGAQ = OMEGA(I)

    XP = OMEGAR

    Integral = (0.0_q,0.0_q)
    DO K=1,4
       IF (I>1) THEN
          WEIGHT=1
          IF (I==SIZE(OMEGA)) WEIGHT=2 ! to avoid loosing spectral weight last point
          ! is counted twice (triangle between OMEGAQ_minus and QMEGAQ)

          OMEGAQ_minus = OMEGA(I-1)
          IF (K<=2) THEN
             IF (K==1) THEN
                X  = OMEGAQ_minus
             ELSE 
                X  = OMEGAQ
             ENDIF
             AA = (0.0_q,0.0_q)

             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2


             AA = AA + 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 

             Integral(K) = -OMEGAQ_minus/(OMEGAQ - OMEGAQ_minus)*AA*WEIGHT

             BB = (0.0_q,0.0_q)



             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)

             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  



             BB = BB + (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)

             BB = BB - (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB + X - XP*LOG((X+XP)**2+SHIFT1**2)+2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)  

             Integral(K) = Integral(K)+1.0/(OMEGAQ - OMEGAQ_minus)*BB*WEIGHT
          ENDIF
       ENDIF


       IF (I<SIZE(OMEGA)) THEN
          OMEGAQ_plus  = OMEGA(I+1)

          IF (K>2) THEN
             IF (K==3) THEN
                X  = OMEGAQ
             ELSE 
                X  = OMEGAQ_plus
             ENDIF

             AA = (0.0_q,0.0_q)

             AA = AA - 0.5*LOG((X-XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2
             
             AA = AA + 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 
             

             Integral(K) = OMEGAQ_plus/(OMEGAQ_plus - OMEGAQ)*AA

             BB = (0.0_q,0.0_q)
             
             BB = BB + (XP/2)*LOG((X-XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X-XP)**2+SHIFT1**2)
             
             BB = BB + (XP**2)/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB - (0.0_q,1.0_q)*XP*ATAN((X-XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB - X -XP*LOG((X-XP)**2+SHIFT1**2)-2*XP**2/SHIFT2*ATAN((X-XP)/SHIFT2)
             BB = BB + ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X-XP)/SHIFT2)  
             

             BB = BB + (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)

             BB = BB - (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB + X - XP*LOG((X+XP)**2+SHIFT1**2)+2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)   
             
             Integral(K) = Integral(K) - 1.0/(OMEGAQ_plus - OMEGAQ)*BB
          ENDIF
       ENDIF
    ENDDO

    TABLE(I) = Integral(2)-Integral(1)+Integral(4)-Integral(3)

    ! scale by prefactor
    TABLE(I) = TABLE(I)*CMPLX(0,0.5_q/PI,q)
  END SUBROUTINE SETUP_HILBERT_TABLE


!*********************************************************************
!
! set up the tables for the Hilbert transform using spectral 
! representation
!
! sigma(w) = i/(2 pi) int dw' Imag(W(w'))/(w-w' +- i eta)
!
! the Hilbert transformation is applied to W(G,G') at a point where
! the poles are above the real axis (see note-conjugation)
! 
! one can then show analytically that 
!  for ISIGN = +1:  Imag sigma(w) = Imag(W(w) Theta(-w))
! i.e. Imag sigma(w) has components only a negative w
!  for ISIGN = -1:  Imag sigma(w) = Imag(W(w) Theta(w))
! The real part of sigma is given by the Hilbert transformation
! of Imag sigma(w)
!
! sigma(w) = -1/(pi) int dw' Imag(sigma(w'))/(w-w'+ i eta)
!
! see also INTEGRATE_W_2E_SPECTRAL
! for ISIGN = +1 the routine calculates the transformation
! matrix to positive frequencies +w, whereas for ISIGN = -1
! it calculates it for negative frequencies -w (and positive eta);
! the relation sigma(+w,-eta)= -sigma(-w, eta)
! is used to obtain the result at negative eta and positive w
! 
! tests:
!  x=TABLE(4,:); wm=MATMUL(x,TABLE_POT_MIN); wp=MATMUL(x,TABLE_POT_PLUS)
! and compare with the tables created by this routine
!  x=AIMAG(TABLE(4,:));  wm=MATMUL(x,TABLE_POT_MIN) ; wp=MATMUL(x,TABLE_POT_PLUS)
! must be identical
!*********************************************************************

  SUBROUTINE SETUP_HILBERT_TABLE_SPECTRAL(OMEGA,OMEGAR,SHIFT,TABLE,I,ISIGN)
    USE constant
    REAL(q) :: OMEGAR, SHIFT,DELTA
    INTEGER :: ISIGN 
    REAL(q) :: SHIFT1,SHIFT2
    REAL(q) :: OMEGA(:)
    INTEGER :: I, I_R, K
    COMPLEX(q) :: TABLE(:)

    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus,WEIGHT

    REAL(q) :: X, XP

    COMPLEX(q) :: Integral(4)
    COMPLEX(q) :: AA, BB
    INTEGER :: L

    SHIFT1 = abs(SHIFT)
    SHIFT2 = abs(SHIFT)

    OMEGAQ = OMEGA(I)
    ! calculate result a negative frequency XP=-w'
    XP = OMEGAR*ISIGN
      
    Integral = (0.0_q,0.0_q)
    DO K=1,4
       IF (I>1) THEN
          WEIGHT=1
          IF (I==SIZE(OMEGA)) WEIGHT=2 ! to avoid loosing spectral weight last point
          ! is counted twice (triangle between OMEGAQ_minus and QMEGAQ)

          OMEGAQ_minus = OMEGA(I-1)
          IF (K<=2) THEN
             IF (K==1) THEN
                X  = OMEGAQ_minus
             ELSE 
                X  = OMEGAQ
             ENDIF
             AA = (0.0_q,0.0_q)

             AA = AA + 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 

             Integral(K) = -OMEGAQ_minus/(OMEGAQ - OMEGAQ_minus)*AA*WEIGHT

             BB = (0.0_q,0.0_q)

             BB = BB + (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)

             BB = BB - (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2

             BB = BB + X - XP*LOG((X+XP)**2+SHIFT1**2)+2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)  

             Integral(K) = Integral(K)+1.0/(OMEGAQ - OMEGAQ_minus)*BB*WEIGHT
          ENDIF
       ENDIF


       IF (I<SIZE(OMEGA)) THEN
          OMEGAQ_plus  = OMEGA(I+1)

          IF (K>2) THEN
             IF (K==3) THEN
                X  = OMEGAQ
             ELSE 
                X  = OMEGAQ_plus
             ENDIF

             AA = (0.0_q,0.0_q)

             AA = AA + 0.5*LOG((X+XP)**2+SHIFT1**2)
             AA = AA - (0.0_q,1.0_q)*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2 
             

             Integral(K) = OMEGAQ_plus/(OMEGAQ_plus - OMEGAQ)*AA

             BB = (0.0_q,0.0_q)
             
             BB = BB + (XP/2)*LOG((X+XP)**2+SHIFT1**2)
             BB = BB - (0.0_q,1.0_q)*SHIFT1*0.5*LOG((X+XP)**2+SHIFT1**2)

             BB = BB - (XP**2)/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB + (0.0_q,1.0_q)*XP*ATAN((X+XP)/SHIFT2)*SHIFT1/SHIFT2
             
             BB = BB + X - XP*LOG((X+XP)**2+SHIFT1**2)+2*XP**2/SHIFT2*ATAN((X+XP)/SHIFT2)
             BB = BB - ((XP**2+SHIFT1**2)/SHIFT2)*ATAN((X+XP)/SHIFT2)   
             
             Integral(K) = Integral(K) - 1.0/(OMEGAQ_plus - OMEGAQ)*BB
          ENDIF
       ENDIF
    ENDDO

    TABLE(I) = Integral(2)-Integral(1)+Integral(4)-Integral(3)

    ! scale by 1/PI and invert sign for negative ISIGN
    !  sigma(+w,ISIGN=-1) = -sigma(-w,ISIGN=1)
    TABLE(I) = -TABLE(I)/PI*ISIGN

  END SUBROUTINE SETUP_HILBERT_TABLE_SPECTRAL

#ifdef numerical_kramkron

!*********************************************************************
!
! set up the tables for the Hilbert transform 
!
!  i/(2 pi) int dw' W(w')/(w-w'+ i delta sign)
!
! using numerical integration
!
!*********************************************************************  
  
  SUBROUTINE SETUP_HILBERT_TABLE_NUM(OMEGA,OMEGAR,SHIFT,TABLE,I,EPSILON,ISIGN)
    USE constant
    REAL(q) :: OMEGAR, EPSILON, SHIFT
    INTEGER :: ISIGN
    REAL(q) :: OMEGA(:)
    INTEGER :: I, I_R, K
    COMPLEX(q) :: TABLE(:)
    REAL(q) :: OMEGAQ,OMEGAQ_minus,OMEGAQ_plus
    REAL(q) :: X, XP
    COMPLEX(q) :: FF
    REAL(q), PARAMETER :: STEP=0.01

    OMEGAQ = OMEGA(I)
    IF (I>1) THEN
       OMEGAQ_minus = OMEGA(I-1)
    ELSE
       OMEGAQ_minus =-OMEGA(2)
    ENDIF
    OMEGAQ_plus  = OMEGA(I+1)

    XP = OMEGAR-EPSILON

    TABLE(I)=0
    IF (I>1) THEN
       DO X = OMEGAQ_minus, OMEGAQ, STEP
          FF = (X - OMEGAQ_minus)/(OMEGAQ - OMEGAQ_minus)
          TABLE(I) = TABLE(I) +FF/(X+XP+ISIGN*CMPLX(0,SHIFT,q)) +  &
               FF/(-X+XP+ISIGN*CMPLX(0,SHIFT,q))
       ENDDO
    ENDIF
    DO X = OMEGAQ, OMEGAQ_plus, STEP
       FF = (OMEGAQ_plus - X)/(OMEGAQ_plus - OMEGAQ)
       TABLE(I) = TABLE(I) +FF/(X+XP+ISIGN*CMPLX(0,SHIFT,q)) +  &
            FF/(-X+XP+ISIGN*CMPLX(0,SHIFT,q))
    ENDDO

    TABLE(I) = STEP*TABLE(I)*CMPLX(0,0.5_q/PI,q)

  END SUBROUTINE SETUP_HILBERT_TABLE_NUM
#endif

  
!*********************************************************************
!
! set up the tables for the Hilbert transform 
! of the screened potential (either for positive or negative
! frequencies, depending on ISIGN)
!
!  i/(2 pi) int dw' W(w')/(w-w'+ i delta sign) [e (id w')]
!
! Equ. (22) in Shishkin, Kresse, PRB 74, 035101
! see SETUP_HILBERT_TABLE for details
!
! sigma(ir) = sum_iw  W (iw) TABLE(iw, ir)
!
!*********************************************************************

  SUBROUTINE POT_HILBERT_TABLE( TABLE, OMEGA, SHIFT, ISIGN)
    USE constant
    COMPLEX(q), POINTER :: TABLE(:,:)   ! table for Kramers Kroning 
    REAL(q)    :: OMEGA(:)     ! frequencies
    REAL(q)    :: SHIFT        ! complex shift to be used
    INTEGER    :: ISIGN        ! transformation to positive or negative frequencies
  ! local
    INTEGER :: NOMEGA, I_R, IW
    REAL(q) :: SCALE
    COMPLEX(q) :: SUMI

    NOMEGA=SIZE(OMEGA)
    ALLOCATE(  TABLE(NOMEGA, NOMEGA))

    TABLE  = 0
#ifdef numerical_kramkron
    DO IW=1,NOMEGA-1
       DO I_R=1,NOMEGA
          CALL SETUP_HILBERT_TABLE_NUM(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW, 0.0_q, ISIGN) 
#else
    DO IW=1,NOMEGA
       DO I_R=1,NOMEGA
          CALL SETUP_HILBERT_TABLE(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW, ISIGN) 
#endif
       ENDDO
#ifdef debug
       SUMI=0
       DO I_R=1,NOMEGA
          IF (I_R>1) THEN
             SUMI=SUMI+TABLE(IW,I_R)
          ENDIF
       ENDDO
       WRITE(100+IW,'(2F8.4)') (OMEGA(I_R),IMAG(TABLE(IW,I_R)),I_R=1,NOMEGA)
       WRITE(100+IW,*)
       WRITE(100+IW,'(2F8.4)') (OMEGA(I_R),REAL(TABLE(IW,I_R)),I_R=1,NOMEGA)

       WRITE(*,'(F14.7,I4,2F14.7)') OMEGA(IW),IW,SUMI
#endif
    ENDDO

  END SUBROUTINE POT_HILBERT_TABLE


!*********************************************************************
!
! set up the tables for the Hilbert transform 
! of the screened potential (use LSPECTRALGW =.TRUE. in the INCAR)
!
!  i/(2 pi) int dw' Imag(W(w'))/(w-w'+ i delta sign) [e (id w')]
!
! For this version, the spectral function Imag(W) needs to be supplied
! to the routine performing the Hilbert transformation
! see SETUP_HILBERT_TABLE_SPECTRAL for details
! this version is in principle an "improvement" of the version
! above, although QP shifts are not improved at all
!
!*********************************************************************

  SUBROUTINE POT_HILBERT_TABLE_SPECTRAL( TABLE, OMEGA, SHIFT, ISIGN)
    USE constant
    COMPLEX(q), POINTER :: TABLE(:,:)   ! table for Kramers Kroning 
    REAL(q)    :: OMEGA(:)     ! frequencies
    REAL(q)    :: SHIFT        ! complex shift to be used
    INTEGER    :: ISIGN        ! transformation to positive or negative frequencies
  ! local
    INTEGER :: NOMEGA, I_R, IW
    REAL(q) :: SCALE
    COMPLEX(q) :: SUMI

    NOMEGA=SIZE(OMEGA)
    ALLOCATE(  TABLE(NOMEGA, NOMEGA))

    TABLE  = 0
    DO IW=1,NOMEGA
       DO I_R=1,NOMEGA
          CALL SETUP_HILBERT_TABLE_SPECTRAL(OMEGA, OMEGA(I_R), SHIFT, TABLE(:,I_R), IW, ISIGN) 
       ENDDO
    ENDDO

  END SUBROUTINE POT_HILBERT_TABLE_SPECTRAL

!*********************************************************************
!
! deallocate a table generated for a Kramers Kroning transformation
!
!*********************************************************************

  SUBROUTINE DEALLOCATE_KRAM_KRON_TABLE( TABLE)
    
    COMPLEX(q), POINTER :: TABLE(:,:)   ! table for Kramers Kroning 

    IF (ASSOCIATED(TABLE)) DEALLOCATE(TABLE)

  END SUBROUTINE DEALLOCATE_KRAM_KRON_TABLE

!*********************************************************************
!
! add Drude like term according to the supplied
! WPLASMON term
!
!*********************************************************************

  SUBROUTINE ADD_DRUDE_IMAG(WPOT, WGW, WPLASMON , COMEGA, VOLUME, LKSUM)
    USE constant
    USE wave
    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (wavedes) WGW
    REAL(q) :: WPLASMON(3,3)
    COMPLEX(q) :: COMEGA(:)
    REAL(q) :: VOLUME
    LOGICAL :: LKSUM
  ! local
    REAL(q)    :: DELTA, SCALE, SUM(3,3), NCORE
    INTEGER    :: NOMEGA

    IF (WPOT%LGAMMA) THEN
       SUM=0
       DO NOMEGA=2,WPOT%NOMEGA-1
          DELTA=REAL(WPOT%COMEGA(NOMEGA+1)-WPOT%COMEGA(NOMEGA-1),q)/2
          SUM=SUM+WPOT%HEAD(:,:,NOMEGA)*WPOT%COMEGA(NOMEGA)*DELTA
       ENDDO
!      wp^2 = 2/pi \ Imag eps(w) w dw
!      the Imag eps(w)  is obtained by multiplication with EDEPS/VOLUME
!      factor pi is lacking in our definition of the spectral function
       IF (LKSUM) THEN
          CALLMPI( M_sum_d(WGW%COMM, SUM(1,1), 9))
       ELSE
          CALLMPI( M_sum_d(WGW%COMM_INTER, SUM(1,1), 9))
       ENDIF

       ! count the number of cores that hold contributions to each frequency
       NCORE=1
       IF (LKSUM) THEN
          CALLMPI( M_sum_d(WGW%COMM, NCORE, 1))
       ELSE
          CALLMPI( M_sum_d(WGW%COMM_INTER, NCORE, 1))
       ENDIF

!       WRITE(*,*)
!       WRITE(*,'(3F14.7)') SUM*PI*EDEPS/VOLUME*2/PI
!       WRITE(*,'(3F14.7)') WPLASMON

       IF (WPOT%NOMEGA_LOW<=2 .AND. 2<=WPOT%NOMEGA_HIGH) THEN
          NOMEGA=2-WPOT%NOMEGA_LOW +1
          IF (ABS(WPOT%COMEGA(NOMEGA)-COMEGA(2))>1E-6) THEN
             WRITE(*,*) 'internal error in ADD_DRUDE_TERM: incorrect slot'
             STOP
          ENDIF
          DELTA=REAL(COMEGA(3)-COMEGA(1),q)/2
          WPOT%HEAD(:,:,NOMEGA)=WPOT%HEAD(:,:,NOMEGA)+WPLASMON/DELTA /WPOT%OMEGA(NOMEGA) & 
               /(EDEPS/VOLUME*2)/NCORE
!          WRITE(*,'(6F14.7)') WPOT%HEAD(:,:,1)
       ENDIF
       SUM=0
       DO NOMEGA=2,WPOT%NOMEGA-1
          DELTA=REAL(WPOT%COMEGA(NOMEGA+1)-WPOT%COMEGA(NOMEGA-1),q)/2
          SUM=SUM+WPOT%HEAD(:,:,NOMEGA)*WPOT%COMEGA(NOMEGA)*DELTA
       ENDDO
       IF (LKSUM) THEN
          CALLMPI( M_sum_d(WGW%COMM, SUM(1,1), 9))
       ELSE
          CALLMPI( M_sum_d(WGW%COMM_INTER, SUM(1,1), 9))
       ENDIF
!       WRITE(*,'(3F14.7)') SUM*EDEPS*PI/VOLUME*2/PI

    ENDIF
  END SUBROUTINE ADD_DRUDE_IMAG

  SUBROUTINE ADD_DRUDE_REAL(WPOT, WGW, WPLASMON , COMEGA, VOLUME, CSHIFT, LKSUM)
    USE constant

    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (wavedes) WGW
    REAL(q) :: WPLASMON(3,3)
    COMPLEX(q) :: COMEGA(:)
    REAL(q) :: VOLUME, CSHIFT
    LOGICAL :: LKSUM
   ! local
    REAL(q)    :: DAMP           
    REAL(q)    :: DELTA, SCALE
    COMPLEX(q) :: SUM(3,3)
    INTEGER    :: NOMEGA
   
   ! these lines are for metals not well tested so use with care
    IF (WPOT%LGAMMA .AND. ABS(WPLASMON(1,1))+ABS(WPLASMON(2,2))+ABS(WPLASMON(3,3))>1.0 ) THEN
       DAMP = 8*CSHIFT
       SUM=0
       DO NOMEGA=2,WPOT%NOMEGA-1
          DELTA=REAL(WPOT%COMEGA(NOMEGA+1)-WPOT%COMEGA(NOMEGA-1),q)/2
          SUM=SUM+WPOT%HEAD(:,:,NOMEGA)*WPOT%COMEGA(NOMEGA)*DELTA
       ENDDO
       IF (LKSUM) THEN
          CALLMPI( M_sum_z(WGW%COMM, SUM, 9))
       ELSE
          CALLMPI( M_sum_z(WGW%COMM_INTER, SUM, 9))
       ENDIF
       
!       wp^2 = 2/pi \ Imag eps(w) w dw
!       the Imag eps(w)  is obtained by multiplication with EDEPS/VOLUME
!       WRITE(*,*)
!       WRITE(*,'(6F14.7)') SUM*EDEPS/VOLUME*2/PI
!       WRITE(*,'(3F14.7)') WPLASMON
!       WRITE(*,*) WPOT%NOMEGA
       DO NOMEGA=1,WPOT%NOMEGA
          IF (ABS(WPOT%COMEGA(NOMEGA))>1E-6) THEN
             WPOT%HEAD(:,:,NOMEGA)=WPOT%HEAD(:,:,NOMEGA)&
             +(WPLASMON/((WPOT%COMEGA(NOMEGA))*(WPOT%COMEGA(NOMEGA)-CMPLX(0,DAMP,q))))/(EDEPS/VOLUME)
          ELSE
             !metalic singularity of head
             WPOT%HEAD(:,:,NOMEGA)=0
             WPOT%HEAD(1,1,NOMEGA)=-1000
             WPOT%HEAD(2,2,NOMEGA)=-1000
             WPOT%HEAD(3,3,NOMEGA)=-1000
          ENDIF
       ENDDO
       SUM=0
       DO NOMEGA=2,WPOT%NOMEGA-1
          DELTA=REAL(WPOT%COMEGA(NOMEGA+1)-WPOT%COMEGA(NOMEGA-1),q)/2
          SUM=SUM+WPOT%HEAD(:,:,NOMEGA)*WPOT%COMEGA(NOMEGA)*DELTA
       ENDDO
       IF (LKSUM) THEN
          CALLMPI( M_sum_z(WGW%COMM, SUM, 9))
       ELSE
          CALLMPI( M_sum_z(WGW%COMM_INTER, SUM, 9))
       ENDIF
          
!       wp^2 = 2/pi \ Imag eps(w) w dw
!       the Imag eps(w)  is obtained by multiplication with EDEPS/VOLUME
!       WRITE(*,*) "------------------------------------"
!       WRITE(*,'(6F14.7)') SUM*EDEPS/VOLUME*2/PI
    ENDIF
  END SUBROUTINE ADD_DRUDE_REAL

!*********************************************************************
!
! This routine performes the Kramers-Kronig transform of the screened 
! potential using the precalculated tables TABLE
!
!   int dw' X(w')(1/(w-w'- i eta)- (1/( w + w' -+ i eta) =
!
! X(ir) = sum_iw TABLE(iw, ir) Spectral_function_of X(iw)
!
!*********************************************************************

  SUBROUTINE DO_CHI_KRAM_KRON_TABLE(WPOT, CHI, WGW, TABLE )
    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (responsefunction) CHI         ! full response function
    TYPE (wavedes), POINTER :: WGW      ! communicator for WPOT
    COMPLEX(q) :: TABLE(:,:)            ! table for the Kramers Kronig transformation
  ! local
    INTEGER :: NALLOC1, NALLOC2, I_R, IW

#ifdef use_shmem
    IF (CHI%LSHMEM) THEN
        WRITE(*,*) 'internal error in DO_CHI_KRAM_KRON_TABLE: the Hilbert transform is not yet support for a shmem result array'
        STOP
    ENDIF
#endif

    NALLOC1=SIZE(CHI%RESPONSEFUN,1)
    NALLOC2=SIZE(CHI%RESPONSEFUN,2)

    IF (CHI%LREALSTORE .OR. WPOT%LREALSTORE) THEN
       WRITE(*,*) 'internal error in DO_CHI_KRAM_KRON_TABLE: LREALSTORE is not supported yet'
       STOP
    ENDIF

    ! CHI%RESPONSEFUN(:,:,I_R) += sum_IW (WPOT%RESPONSEFUN(:,:,IW))*TABLE(IW,I_R)
    CALL ZGEMM( 'N', 'N' , NALLOC1*NALLOC2 , CHI%NOMEGA , WPOT%NOMEGA  , & 
         (1.0_q,0.0_q) , WPOT%RESPONSEFUN(1,1,1), NALLOC1*NALLOC2 , & 
         TABLE(WPOT%NOMEGA_LOW, CHI%NOMEGA_LOW) , SIZE(TABLE,1) , &
         (1.0_q, 0.0_q) , CHI%RESPONSEFUN(1,1,1) , NALLOC1*NALLOC2 )
    
    ! CHI%WING(:,:,I_R) += sum_IW (WPOT%WING(:,:,IW))*TABLE(IW,I_R)
    CALL ZGEMM( 'N', 'N' , NALLOC1*3 , CHI%NOMEGA , WPOT%NOMEGA  , & 
         (1.0_q,0.0_q) , WPOT%WING(1,1,1), NALLOC1*3 , & 
         TABLE(WPOT%NOMEGA_LOW, CHI%NOMEGA_LOW) , SIZE(TABLE,1) , &
         (1.0_q, 0.0_q) , CHI%WING(1,1,1) , NALLOC1*3 )

    CALL ZGEMM( 'N', 'N' , NALLOC1*3 , CHI%NOMEGA , WPOT%NOMEGA  , & 
         (1.0_q,0.0_q) , WPOT%CWING(1,1,1), NALLOC1*3 , & 
         TABLE(WPOT%NOMEGA_LOW, CHI%NOMEGA_LOW) , SIZE(TABLE,1) , &
         (1.0_q, 0.0_q) , CHI%CWING(1,1,1) , NALLOC1*3 )
    
    ! CHI%HEAD(:,:,I_R) += sum_IW (WPOT%HEAD(:,:,IW))*TABLE(IW,I_R)
    CALL ZGEMM( 'N', 'N' , 3*3 , CHI%NOMEGA , WPOT%NOMEGA  , &
         (1.0_q,0.0_q) , WPOT%HEAD(1,1,1), 3*3 , & 
         TABLE(WPOT%NOMEGA_LOW, CHI%NOMEGA_LOW) , SIZE(TABLE,1) , &
         (1.0_q, 0.0_q) , CHI%HEAD(1,1,1) , 3*3 )
    
  END SUBROUTINE DO_CHI_KRAM_KRON_TABLE


!*********************************************************************
!
! This routine sums the WPOT array over all nodes
! if the outer NQ loop is parallelized over k-points
! only summation over COMM_INTER (distribution over bands)
! needs to be done
! if the inner K loop is parallelized over k-points
! summation over COMM_INTER COMM_KINTER needs to be done
!
!*********************************************************************

  SUBROUTINE DO_CHI_SUM(WPOT, WGWQ, LSUMK )
    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (wavedes1)          WGWQ       ! communicator for WPOT
    LOGICAL :: LSUMK
  ! local
    INTEGER :: NALLOC1, NALLOC2, NOMEGA
    INTEGER :: ierror

    NALLOC1=SIZE(WPOT%RESPONSEFUN,1)
    NALLOC2=SIZE(WPOT%RESPONSEFUN,2)

#ifdef use_shmem
    IF (WPOT%LSHMEM) THEN
       ! first barrier so that all calculations are certainly done on all shmem node
       CALLMPI( M_barrier(WPOT%COMM_SHMEM) )

       IF (WPOT%LLEAD) THEN
          ! sum results over all "root" cores
          DO  NOMEGA=1,WPOT%NOMEGA
            CALLMPI( M_sum_z(WPOT%COMM_INTER_SHMEM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
          ENDDO
       ENDIF

       ! second barrier so that all nodes wait until all data are properly merged
       CALLMPI( M_barrier(WPOT%COMM_SHMEM))

       IF (LSUMK) THEN
          CALLMPI( M_sum_z(WGWQ%COMM, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
          CALLMPI( M_sum_z(WGWQ%COMM, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
          CALLMPI( M_sum_z(WGWQ%COMM, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
       ELSE
          CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
          CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
          CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
       ENDIF
    ELSE
#endif
    ! sum result over all nodes
    IF (LSUMK) THEN
       CALLMPI( M_sum_z(WGWQ%COMM, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
       DO  NOMEGA=1,WPOT%NOMEGA
          CALLMPI( M_sum_z(WGWQ%COMM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
       ENDDO
       CALLMPI( M_sum_z(WGWQ%COMM, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGWQ%COMM, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
    ELSE
       CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
       DO  NOMEGA=1,WPOT%NOMEGA
          CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
       ENDDO
       CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGWQ%COMM_INTER, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
    ENDIF
#ifdef use_shmem
    ENDIF
#endif


  END SUBROUTINE DO_CHI_SUM


!*********************************************************************
!
! This routine sums the WPOT array and stores the
! frequencies handled locally in the CHI array
! if the outer NQ loop is parallelized over k-points
! only summation over COMM_INTER (distribution over bands)
! needs to be done
! if the inner K loop is parallelized over k-points
! summation over COMM_INTER COMM_KINTER needs to be done
!
!*********************************************************************

  SUBROUTINE SCATTER_FREQU(WPOT, CHI, WGWQ, LSUMK )
    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (responsefunction) CHI         ! full response function
    TYPE (wavedes1)         WGWQ        ! communicator for WPOT
    LOGICAL :: LSUMK
  ! local
    INTEGER :: NALLOC1, NALLOC2, I_WPOT, I_CHI

#ifdef use_shmem
    IF (CHI%LSHMEM) THEN
        WRITE(*,*) 'internal error in SCATTER_FREQU: the Hilbert transform is not yet support for a shmem result array'
        STOP
    ENDIF
#endif
    CALL DO_CHI_SUM(WPOT, WGWQ, LSUMK )

    NALLOC1=SIZE(CHI%RESPONSEFUN,1)
    NALLOC2=SIZE(CHI%RESPONSEFUN,2)

    ! now scatter results to nodes

    DO I_WPOT=1, WPOT%NOMEGA
       DO I_CHI=1, CHI%NOMEGA
          IF (CHI%COMEGA(I_CHI)==WPOT%COMEGA(I_WPOT)) THEN

             ! CHI%RESPONSEFUN = WPOT%RESPONSEFUN
             CALL ZCOPY( NALLOC1*NALLOC2, WPOT%RESPONSEFUN(1,1,I_WPOT), 1,CHI%RESPONSEFUN(1,1,I_CHI), 1)
             ! CHI%WING = WPOT%WING
             CALL ZCOPY( 3*NALLOC1, WPOT%WING(1,1,I_WPOT), 1,CHI%WING(1,1,I_CHI), 1)

             ! CHI%CWING = WPOT%CWING
             CALL ZCOPY( 3*NALLOC1, WPOT%CWING(1,1,I_WPOT), 1,CHI%CWING(1,1,I_CHI), 1)

             ! CHI%HEAD = WPOT%HEAD
             CALL ZCOPY( 3*3, WPOT%HEAD(1,1,I_WPOT), 1,CHI%HEAD(1,1,I_CHI), 1)

          ENDIF
       ENDDO
    ENDDO
  END SUBROUTINE SCATTER_FREQU



!*********************************************************************
!
! This routine stores the wings and heads into the responsefun array
! at a selected frequency and for a selected direction
!
!*********************************************************************

  SUBROUTINE WING_FROM_BODY( CHI, IDIR, NOMEGA)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    INTEGER :: IDIR
    INTEGER, OPTIONAL :: NOMEGA

    IF (.NOT.CHI%LGAMMA) THEN
       WRITE(*,*) 'internal error in WING_FROM_BODY: LGAMMA is not set'
       STOP
    ENDIF

    IF (CHI%LREALSTORE) THEN
       IF (PRESENT(NOMEGA)) THEN
          CHI%WINGR(:,IDIR,NOMEGA)  =CHI%RESPONSER(:,1,NOMEGA)
          CHI%CWINGR(:,IDIR,NOMEGA) =CHI%RESPONSER(1,:,NOMEGA)
          CHI%HEAD(IDIR,IDIR,NOMEGA)=CHI%RESPONSER(1,1,NOMEGA)
       ELSE
          CHI%WINGR(:,IDIR,1)  =CHI%RESPONSER(:,1,1)
          CHI%CWINGR(:,IDIR,1) =CHI%RESPONSER(1,:,1)
          CHI%HEAD(IDIR,IDIR,1)=CHI%RESPONSER(1,1,1)
       ENDIF
    ELSE
       IF (PRESENT(NOMEGA)) THEN
          CHI%WING(:,IDIR,NOMEGA)   =CHI%RESPONSEFUN(:,1,NOMEGA)
          CHI%CWING(:,IDIR,NOMEGA)  =CHI%RESPONSEFUN(1,:,NOMEGA)
          CHI%HEAD(IDIR,IDIR,NOMEGA)=CHI%RESPONSEFUN(1,1,NOMEGA)
       ELSE
          CHI%WING(:,IDIR,1)   =CHI%RESPONSEFUN(:,1,1)
          CHI%CWING(:,IDIR,1)  =CHI%RESPONSEFUN(1,:,1)
          CHI%HEAD(IDIR,IDIR,1)=CHI%RESPONSEFUN(1,1,1)
       ENDIF
    ENDIF
  END SUBROUTINE WING_FROM_BODY

  SUBROUTINE BODY_FROM_WING( CHI, IDIR, NOMEGA)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    INTEGER :: IDIR
    INTEGER, OPTIONAL :: NOMEGA

    IF (.NOT.CHI%LGAMMA) THEN
       WRITE(*,*) 'internal error in BODY_FROM_WING: LGAMMA is not set'
       STOP
    ENDIF
    
    IF (CHI%LREALSTORE) THEN
       IF (PRESENT(NOMEGA)) THEN
          CHI%RESPONSER(:,1,NOMEGA)=CHI%WINGR(:,IDIR,NOMEGA)
          CHI%RESPONSER(1,:,NOMEGA)=CHI%CWINGR(:,IDIR,NOMEGA)
          CHI%RESPONSER(1,1,NOMEGA)=CHI%HEAD(IDIR,IDIR,NOMEGA)
       ELSE
          CHI%RESPONSER(:,1,1)=CHI%WINGR(:,IDIR,1)
          CHI%RESPONSER(1,:,1)=CHI%CWINGR(:,IDIR,1)
          CHI%RESPONSER(1,1,1)=CHI%HEAD(IDIR,IDIR,1)
       ENDIF
    ELSE
       IF (PRESENT(NOMEGA)) THEN
          CHI%RESPONSEFUN(:,1,NOMEGA)=CHI%WING(:,IDIR,NOMEGA)
          CHI%RESPONSEFUN(1,:,NOMEGA)=CHI%CWING(:,IDIR,NOMEGA)
          CHI%RESPONSEFUN(1,1,NOMEGA)=CHI%HEAD(IDIR,IDIR,NOMEGA)
       ELSE
          CHI%RESPONSEFUN(:,1,1)=CHI%WING(:,IDIR,1)
          CHI%RESPONSEFUN(1,:,1)=CHI%CWING(:,IDIR,1)
          CHI%RESPONSEFUN(1,1,1)=CHI%HEAD(IDIR,IDIR,1)
       ENDIF
    ENDIF
  END SUBROUTINE BODY_FROM_WING

!*********************************************************************
!
! This subroutine sets the k-point entry in a response function
! and accordingly the flag LGAMMA
!
!*********************************************************************

  SUBROUTINE SET_RESPONSE_KPOINT(CHI, VKPT, NQ)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    REAL(q) :: VKPT(:)
    INTEGER :: NQ

    CHI%VKPT=VKPT
    CHI%NQ  =NQ

    IF (ABS(SUM(CHI%VKPT*CHI%VKPT))>G2ZERO ) THEN
       CHI%LGAMMA=.FALSE.
    ELSE
       CHI%LGAMMA=.TRUE.
    ENDIF
    
  END SUBROUTINE SET_RESPONSE_KPOINT

!*********************************************************************
!
! This subroutine copies a response function matrix
!
!*********************************************************************

  SUBROUTINE SET_MAT_FROM_RESPONSE( MAT, CHI, NOMEGA, IDIR)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    COMPLEX(q) :: MAT(:,:)
    INTEGER :: NOMEGA
    INTEGER, OPTIONAL :: IDIR

    IF (CHI%LGAMMA) THEN
       IF (.NOT.PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_MAT_RESPONSE_FROM: IDIR not specified'
          STOP
       ENDIF
    ELSE
       IF (PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_MAT_RESPONSE_FROM: IDIR must not be specified'
          STOP
       ENDIF
    ENDIF

    IF (CHI%LREALSTORE) THEN
       MAT=CHI%RESPONSER(:,:,NOMEGA)
       IF (PRESENT(IDIR)) THEN
          IF (IDIR>0 ) THEN
             MAT(:,1)=CHI%WINGR(:,IDIR,NOMEGA)
             MAT(1,:)=CHI%CWINGR(:,IDIR,NOMEGA)
             MAT(1,1)=CHI%HEAD(IDIR,IDIR,NOMEGA)
          ENDIF
       ENDIF
    ELSE
       MAT=CHI%RESPONSEFUN(:,:,NOMEGA)
       IF (PRESENT(IDIR)) THEN
          IF (IDIR>0 ) THEN
             MAT(:,1)=CHI%WING(:,IDIR,NOMEGA)
             MAT(1,:)=CHI%CWING(:,IDIR,NOMEGA)
             MAT(1,1)=CHI%HEAD(IDIR,IDIR,NOMEGA)
          ENDIF
       ENDIF
    ENDIF
       
  END SUBROUTINE SET_MAT_FROM_RESPONSE

  SUBROUTINE SET_RESPONSE_FROM_MAT( MAT, CHI, NOMEGA, IDIR)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    COMPLEX(q) :: MAT(:,:)
    INTEGER :: NOMEGA
    INTEGER, OPTIONAL :: IDIR

    IF (CHI%LGAMMA) THEN
       IF (.NOT.PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_RESPONSE_FROM_MAT: IDIR not specified'
          STOP
       ENDIF
    ELSE
       IF (PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_RESPONSE_FROM_MAT: IDIR must not be specified'
          STOP
       ENDIF
    ENDIF

    IF (CHI%LREALSTORE) THEN
       CHI%RESPONSER(:,:,NOMEGA)=MAT

       IF (PRESENT(IDIR)) THEN 
          IF ( IDIR>0 ) THEN
             CHI%WINGR(:,IDIR,NOMEGA)    =MAT(:,1)
             CHI%CWINGR(:,IDIR,NOMEGA)   =MAT(1,:)
             CHI%HEAD(IDIR,IDIR,NOMEGA)  =MAT(1,1)
          ENDIF
       ENDIF
    ELSE
       CHI%RESPONSEFUN(:,:,NOMEGA)=MAT

       IF (PRESENT(IDIR)) THEN 
          IF ( IDIR>0 ) THEN
             CHI%WING(:,IDIR,NOMEGA)    =MAT(:,1)
             CHI%CWING(:,IDIR,NOMEGA)   =MAT(1,:)
             CHI%HEAD(IDIR,IDIR,NOMEGA) =MAT(1,1)
          ENDIF
       ENDIF
    ENDIF
  END SUBROUTINE SET_RESPONSE_FROM_MAT


  SUBROUTINE SET_WING_FROM_MAT( MAT, CHI, NOMEGA, IDIR)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    COMPLEX(q) :: MAT(:,:)
    INTEGER :: NOMEGA
    INTEGER, OPTIONAL :: IDIR

    IF (CHI%LGAMMA) THEN
       IF (.NOT.PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_RESPONSE_FROM_MAT: IDIR not specified'
          STOP
       ENDIF
    ELSE
       IF (PRESENT(IDIR)) THEN
          WRITE(*,*) 'internal error in SET_RESPONSE_FROM_MAT: IDIR must not be specified'
          STOP
       ENDIF
    ENDIF

    IF (CHI%LREALSTORE) THEN
       IF (PRESENT(IDIR)) THEN
          CHI%WINGR(:,IDIR,NOMEGA)    =MAT(:,1)
          CHI%CWINGR(:,IDIR,NOMEGA)   =MAT(1,:)
          CHI%HEAD(IDIR,IDIR,NOMEGA) =MAT(1,1)
       ENDIF
    ELSE
       IF (PRESENT(IDIR)) THEN
          CHI%WING(:,IDIR,NOMEGA)    =MAT(:,1)
          CHI%CWING(:,IDIR,NOMEGA)   =MAT(1,:)
          CHI%HEAD(IDIR,IDIR,NOMEGA) =MAT(1,1)
       ENDIF
    ENDIF
   
  END SUBROUTINE SET_WING_FROM_MAT


!**********************************************************************
!
! multiply a response function from right by MAT
!   CHI MAT -> CHI
!
!**********************************************************************

  SUBROUTINE MATMUL_RIGHT( CHI, MAT, WDESQ)
    USE dfast
    COMPLEX(q) :: CHI(:,:)
    COMPLEX(q) :: MAT(:,:)
    TYPE (wavedes1) :: WDESQ
  ! local
    COMPLEX(q) :: CBLOCK(NBLK, SIZE(CHI,1))
    INTEGER NP, IBLOCK, ILEN, N1, M, IADDPL
   
    NP= WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2

    DO IBLOCK=0,NP-1,NBLK
       ILEN=MIN(NBLK,NP-IBLOCK)
       IADDPL=MIN(IBLOCK,NP-1)
       ILEN=MAX(ILEN,0)
         
       DO N1=1,NP
          DO M=1,ILEN
             CBLOCK(M,N1)=CHI(M+IADDPL,N1)
          ENDDO
       ENDDO
       
       CALL ZGEMM('N', 'N', ILEN, NP, NP, (1.0_q,0.0_q), &
            CBLOCK(1,1), NBLK, &
            MAT(1,1), SIZE(MAT,1), &
            (0.0_q,0.0_q), CHI(IADDPL+1,1), SIZE(CHI,1))
    ENDDO
    
  END SUBROUTINE MATMUL_RIGHT

!**********************************************************************
!
! multiply a response function from left by MAT
!   MAT CHI -> CHI
!
!**********************************************************************

  SUBROUTINE MATMUL_LEFT( CHI, MAT, WDESQ)
    USE dfast
    COMPLEX(q) :: CHI(:,:)
    COMPLEX(q) :: MAT(:,:)
    TYPE (wavedes1) :: WDESQ
  ! local
    COMPLEX(q) :: CBLOCK(SIZE(CHI,1), NBLK)
    INTEGER NP, IBLOCK, ILEN, N1, M, IADDPL

    NP= WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2
    
    DO IBLOCK=0,NP-1,NBLK
       ILEN=MIN(NBLK,NP-IBLOCK)
       IADDPL=MIN(IBLOCK,NP-1)
       ILEN=MAX(ILEN,0)
         
       DO N1=1,ILEN
          DO M=1,NP
             CBLOCK(M,N1)=CHI(M,N1+IADDPL)
          ENDDO
       ENDDO
       
       CALL ZGEMM('N', 'N', NP, ILEN, NP, (1.0_q,0.0_q), &
            MAT(1,1), SIZE(MAT,1), &
            CBLOCK(1,1), SIZE(CHI,1), &
            (0.0_q,0.0_q), CHI(1,IADDPL+1), SIZE(CHI,1))
    ENDDO
    
  END SUBROUTINE MATMUL_LEFT


!**********************************************************************
!
! multiply two response function matrices and store result in RES
!   RES = CHI MAT
!
!**********************************************************************

  SUBROUTINE MATMUL_RESPONSE( RES, CHI, MAT, WDESQ)
    USE dfast
    COMPLEX(q) :: RES(:,:)
    COMPLEX(q) :: CHI(:,:)
    COMPLEX(q) :: MAT(:,:)
    TYPE (wavedes1) :: WDESQ
  ! local
    INTEGER NP
   
    NP= WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2
    
    CALL ZGEMM('N', 'N', NP, NP, NP, (1.0_q,0.0_q), &
         CHI(1,1), SIZE(CHI,1), &
         MAT(1,1), SIZE(MAT,1), &
         (0.0_q,0.0_q), RES(1,1), SIZE(RES,1))
    
  END SUBROUTINE MATMUL_RESPONSE

!*********************************************************************
!
! Dump the responsefunction or the screened potential to IU
! 
! for NWRITE <3, only the head of the response function array is 
!     written doe IU
!
! if LINVERS is passed, the supplies head is inverted 
! FAKT is a scaling constant used to scale the head
!      (usually EDEPS/LATT_CUR%OMEGA)
! ADD  is a constant that is added (usually 1.0, or 0.0).
!
! for instance if the head stores the IP contribution
!
! epsilon = 1 -EDEPS/LATT_CUR%OMEGA Xi
! 
!
!*********************************************************************

  SUBROUTINE DUMP_RESPONSE( CHI, WDESQ, NOMEGA, STRING, IU, NWRITE, LSUMK, LINVERS, FAKT, ADD)
    TYPE (responsefunction), INTENT(IN) :: CHI
    CHARACTER (LEN=*), INTENT(IN) :: STRING
    INTEGER, INTENT(IN) :: IU
    INTEGER, INTENT(IN) :: NWRITE
    LOGICAL, INTENT(IN) :: LSUMK
    LOGICAL, OPTIONAL :: LINVERS
    TYPE (wavedes1) WDESQ
    REAL(q), OPTIONAL :: FAKT, ADD
  ! local
    INTEGER NP, N, I, ISP, NOMEGA, MING
    COMPLEX(q) :: HEAD(3,3,NOMEGA), COMEGA(NOMEGA), RESPONSEFUN(10,10,NOMEGA)
    REAL(q)    :: DOMEGA(NOMEGA), EPS_REAL(3,3,NOMEGA), EPS_IMAG(3,3,NOMEGA)
    COMPLEX(q) :: WING(10,3,NOMEGA), CWING(10,3,NOMEGA)
    INTEGER :: SET(NOMEGA), INDEX
    INTEGER :: IPIV(3), INFO
    INTEGER, PARAMETER :: NWORK=10
    COMPLEX(q) :: WORK(NWORK*3)

    IF (.NOT.CHI%LGAMMA .AND. NWRITE <3) RETURN

    HEAD=0
    SET =0
    COMEGA=0
    WING =0
    CWING=0
    RESPONSEFUN=0

    MING=WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) MING=MING*2
    MING=MIN(10,MING)
    
    DO N=CHI%NOMEGA_LOW, CHI%NOMEGA_HIGH
       IF (N<=NOMEGA) THEN
          HEAD(:,:,N)             =CHI%HEAD (:,:,N-CHI%NOMEGA_LOW+1)
          IF (CHI%LREALSTORE) THEN
             RESPONSEFUN(1:MING,1:MING,N)=CHI%RESPONSER(1:MING,1:MING,N-CHI%NOMEGA_LOW+1)
             WING (1:MING,1:3,N)     =CHI%WINGR (1:MING,1:3,N-CHI%NOMEGA_LOW+1)
             CWING(1:MING,1:3,N)     =CHI%CWINGR(1:MING,1:3,N-CHI%NOMEGA_LOW+1)
          ELSE
             RESPONSEFUN(1:MING,1:MING,N)=CHI%RESPONSEFUN(1:MING,1:MING,N-CHI%NOMEGA_LOW+1)
             WING (1:MING,1:3,N)     =CHI%WING (1:MING,1:3,N-CHI%NOMEGA_LOW+1)
             CWING(1:MING,1:3,N)     =CHI%CWING(1:MING,1:3,N-CHI%NOMEGA_LOW+1)
          ENDIF
          COMEGA(N)=CHI%COMEGA(N-CHI%NOMEGA_LOW+1)
          SET (N)=1
       ENDIF
    ENDDO

    IF (LSUMK) THEN
       CALLMPI( M_sum_z(WDESQ%COMM, RESPONSEFUN(1,1,1), SIZE(RESPONSEFUN)))
       CALLMPI( M_sum_z(WDESQ%COMM, HEAD(1,1,1), SIZE(HEAD)))
       CALLMPI( M_sum_z(WDESQ%COMM, WING(1,1,1), SIZE(WING)))
       CALLMPI( M_sum_z(WDESQ%COMM, CWING(1,1,1), SIZE(CWING)))
       CALLMPI( M_sum_z(WDESQ%COMM, COMEGA(1), SIZE(COMEGA)))
       CALLMPI( M_sum_i(WDESQ%COMM, SET(1), SIZE(SET)))
    ELSE
       CALLMPI( M_sum_z(WDESQ%COMM_INTER, RESPONSEFUN(1,1,1), SIZE(RESPONSEFUN)))
       CALLMPI( M_sum_z(WDESQ%COMM_INTER, HEAD(1,1,1), SIZE(HEAD)))
       CALLMPI( M_sum_z(WDESQ%COMM_INTER, WING(1,1,1), SIZE(WING)))
       CALLMPI( M_sum_z(WDESQ%COMM_INTER, CWING(1,1,1), SIZE(CWING)))
       CALLMPI( M_sum_z(WDESQ%COMM_INTER, COMEGA(1), SIZE(COMEGA)))
       CALLMPI( M_sum_i(WDESQ%COMM_INTER, SET(1), SIZE(SET)))
    ENDIF

    ! conjugate here to make the imaginary part positive
    ! in fact this places the poles below the real axis
    ! (see also "note-conjugation")
    HEAD=CONJG(HEAD)

    IF (PRESENT(FAKT).AND. PRESENT(ADD)) THEN
       HEAD=HEAD*FAKT
       RESPONSEFUN=RESPONSEFUN*FAKT
       HEAD(1,1,:)=HEAD(1,1,:)+ADD
       HEAD(2,2,:)=HEAD(2,2,:)+ADD
       HEAD(3,3,:)=HEAD(3,3,:)+ADD
    ENDIF

    IF (IU>=0) THEN
       WRITE(IU,1100) STRING
       INDEX=0
       DO N=1,NOMEGA
          IF (SET(N)>0) THEN
             WRITE(IU,'(" w=",2F10.3)')  COMEGA(N)/SET(N)
             IF (CHI%LGAMMA) THEN
                IF (PRESENT(LINVERS)) THEN
                   IF (LINVERS) THEN
                   !   CALL ZGETRF( 3, 3, HEAD(1,1,N), SIZE(HEAD,1), IPIV, INFO )
                   !   CALL ZGETRI( 3, HEAD(1,1,N), SIZE(HEAD,1), IPIV, WORK, 3*NWORK, INFO )
                   HEAD(1,1,N)=1/HEAD(1,1,N)
                   HEAD(2,2,N)=1/HEAD(2,2,N)
                   HEAD(3,3,N)=1/HEAD(3,3,N)
                   ENDIF
                ENDIF

                WRITE(IU,'(3(5X,3(2F10.4,2X)/))') HEAD(:,:,N)
                WRITE(IU,'(3X,F10.3,4X,2F10.3,A)')  ABS(COMEGA(N)), (HEAD(1,1,N)+ HEAD(2,2,N)+ HEAD(3,3,N))/3, " dielectric  constant"

                INDEX=INDEX+1
                DOMEGA(INDEX)=ABS(COMEGA(N))
                EPS_REAL(:,:,INDEX)=REAL(HEAD(:,:,N),q)
                EPS_IMAG(:,:,INDEX)=AIMAG(HEAD(:,:,N))
             ENDIF
             IF (NWRITE>=3) THEN
                WRITE(IU,'( " real part")')
                WRITE(IU,'(10F10.5)') REAL(RESPONSEFUN(1:10,1:10,N),q)
                IF (.NOT. CHI%LREALSTORE) THEN
                   WRITE(IU,'( " imag part")')
                   WRITE(IU,'(10F10.5)') AIMAG(RESPONSEFUN(1:10,1:10,N))
                ENDIF
                ! test
                IF (CHI%LGAMMA) THEN
                   WRITE(IU,'( " real part wings")') 
                   WRITE(IU,'(10F10.5)') REAL(WING(1:10,1:3,N),q)
                   IF (.NOT. CHI%LREALSTORE) THEN
                      WRITE(IU,'( " imag part wings")') 
                      WRITE(IU,'(10F10.5)') AIMAG(WING(1:10,1:3,N))
                   ENDIF
                   WRITE(IU,'( " real part cwings")') 
                   WRITE(IU,'(10F10.5)') REAL(CWING(1:10,1:3,N),q)
                   IF (.NOT. CHI%LREALSTORE) THEN
                      WRITE(IU,'( " imag part cwings")') 
                      WRITE(IU,'(10F10.5)') AIMAG(CWING(1:10,1:3,N))
                   ENDIF
                ENDIF
             ENDIF
             WRITE(IU,*)
          ENDIF
       ENDDO
       IF (CHI%LGAMMA) THEN
          CALL XML_EPSILON_E(DOMEGA, EPS_REAL, EPS_IMAG, INDEX, STRING )
       ENDIF
    ENDIF
1100 FORMAT(/" ",A/, &
             " -------------------------------------")

  END SUBROUTINE DUMP_RESPONSE

!**********************************************************************
!
! add Hartree local field effects to CHI_WORK
!  CHI_WORK = CHIWORK + v Xi 
!
!**********************************************************************

  SUBROUTINE XI_LOCAL_FIELD_HARTREE( CHI_WORK, CHI, NOMEGA, WGWQ )
    COMPLEX(q) :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    IF (CHI%LREALSTORE) THEN
       DO NI=1,NP
          CHI_WORK(NI,1:NP)=CHI_WORK(NI,1:NP)+ & 
               WGWQ%DATAKE(NI,1)*CHI%RESPONSER(NI,1:NP,NOMEGA)
       ENDDO
    ELSE
       DO NI=1,NP
          CHI_WORK(NI,1:NP)=CHI_WORK(NI,1:NP)+ & 
               WGWQ%DATAKE(NI,1)*CHI%RESPONSEFUN(NI,1:NP,NOMEGA)
       ENDDO
    ENDIF
  END SUBROUTINE XI_LOCAL_FIELD_HARTREE

! transposed version
!  CHI_WORK = CHIWORK + Xi v

  SUBROUTINE XI_LOCAL_FIELD_HARTREE_T( CHI_WORK, CHI, NOMEGA, WGWQ )
    COMPLEX(q) :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    IF (CHI%LREALSTORE) THEN
       DO NI=1,NP
          CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)+ & 
               CHI%RESPONSER(1:NP,NI,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDDO
    ELSE
       DO NI=1,NP
          CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)+ & 
               CHI%RESPONSEFUN(1:NP,NI,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDDO
    ENDIF
  END SUBROUTINE XI_LOCAL_FIELD_HARTREE_T

! transposed version
!  CHI_WORK = CHIWORK + Xi v

  SUBROUTINE XI_LOCAL_FIELD_HARTREE_T_MAT( CHI_WORK, RESPONSEFUN, WGWQ )
    COMPLEX(q) :: CHI_WORK(:,:)
    COMPLEX(q) :: RESPONSEFUN(:,:)
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    DO NI=1,NP
       CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)+ & 
            RESPONSEFUN(1:NP,NI)*WGWQ%DATAKE(NI,1)
    ENDDO
       
  END SUBROUTINE XI_LOCAL_FIELD_HARTREE_T_MAT

!
! version that excludes the q=0 component
!
  SUBROUTINE XI_HARTREEBAR_T( CHI_WORK, CHI, NOMEGA, WGWQ )
    COMPLEX(q) :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2
    
    IF (CHI%LREALSTORE) THEN
       DO NI=2,NP
          CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)+ & 
               CHI%RESPONSER(1:NP,NI,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDDO
    ELSE
       DO NI=2,NP
          CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)+ & 
               CHI%RESPONSEFUN(1:NP,NI,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDDO
    ENDIF
  END SUBROUTINE XI_HARTREEBAR_T

  SUBROUTINE XI_LOCAL_FIELD_HARTREE_SYM( CHI_WORK, CHI, NOMEGA, WGWQ )
    IMPLICIT NONE
    COMPLEX(q) :: CHI_WORK(:,:)
    TYPE (responsefunction) :: CHI
    INTEGER :: NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP
    REAL(q) :: POTFAK

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    IF (CHI%LREALSTORE) THEN
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSER(1:NP,1:NP,NOMEGA)
    ELSE
       CHI_WORK(1:NP,1:NP)=CHI%RESPONSEFUN(1:NP,1:NP,NOMEGA)
    ENDIF

    DO NI=1,NP
       ! get v^ 1/2
       POTFAK=SQRT(WGWQ%DATAKE(NI,1))
       CHI_WORK(1:NP,NI)=CHI_WORK(1:NP,NI)*POTFAK
       CHI_WORK(NI,1:NP)=CHI_WORK(NI,1:NP)*POTFAK
    ENDDO

  END SUBROUTINE XI_LOCAL_FIELD_HARTREE_SYM


!**********************************************************************
!
! multiply Xi by the potential operator v 
!  v Xi 
! usually the response function stores at this point epsilon^-1
!
!**********************************************************************

  SUBROUTINE XI_HARTREE( CHI, NOMEGA, WGWQ )
    TYPE (responsefunction) CHI
    INTEGER NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    DO NI=1,NP
       CHI%RESPONSEFUN(NI,1:NP,NOMEGA)=WGWQ%DATAKE(NI,1)*CHI%RESPONSEFUN(NI,1:NP,NOMEGA)
       IF (CHI%LGAMMA) THEN
          CHI%WING(NI,1,NOMEGA)=CHI%WING(NI,1,NOMEGA)*WGWQ%DATAKE(NI,1)
          CHI%WING(NI,2,NOMEGA)=CHI%WING(NI,2,NOMEGA)*WGWQ%DATAKE(NI,1)
          CHI%WING(NI,3,NOMEGA)=CHI%WING(NI,3,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDIF
    ENDDO
    CHI%CWING(:,:,NOMEGA) = CHI%CWING(:,:,NOMEGA)*WGWQ%DATAKE(1,1)
       
  END SUBROUTINE XI_HARTREE

! multiply Xi by the potential operator v 
!  Xi v

  SUBROUTINE XI_HARTREE_T( CHI, NOMEGA, WGWQ )
    TYPE (responsefunction) CHI
    INTEGER NOMEGA
    TYPE (wavedes1) :: WGWQ
  ! local
    INTEGER    NI, NI2, NP

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    DO NI=1,NP
       CHI%RESPONSEFUN(1:NP,NI,NOMEGA)=CHI%RESPONSEFUN(1:NP,NI,NOMEGA)*WGWQ%DATAKE(NI,1)
       IF (CHI%LGAMMA) THEN
          CHI%CWING(NI,1,NOMEGA)=CHI%CWING(NI,1,NOMEGA)*WGWQ%DATAKE(NI,1)
          CHI%CWING(NI,2,NOMEGA)=CHI%CWING(NI,2,NOMEGA)*WGWQ%DATAKE(NI,1)
          CHI%CWING(NI,3,NOMEGA)=CHI%CWING(NI,3,NOMEGA)*WGWQ%DATAKE(NI,1)
       ENDIF
    ENDDO

    ! clear off diagonal elements
!    WRITE(0,*) 'XI_HARTREE_T: cleares off diagonal'
!    DO NI=1,NP
!       CHI%RESPONSEFUN(1:NI-1, NI,NOMEGA)=0
!       CHI%RESPONSEFUN(NI+1:NP,NI,NOMEGA)=0
!    ENDDO

    CHI%WING(:,:,NOMEGA) = CHI%WING(:,:,NOMEGA)*WGWQ%DATAKE(1,1)

  END SUBROUTINE XI_HARTREE_T
      
!*********************************************************************
!
! simple error dump of all entries in the response function
!
!*********************************************************************

  SUBROUTINE DUMP_CHI( LOG, CHI, NOMEGA, NP)
    IMPLICIT NONE
    CHARACTER (LEN=*) LOG
    TYPE (responsefunction) CHI         ! full response function
    INTEGER NOMEGA
    INTEGER NP, NDUMP

    NDUMP=MIN(10,NP)

    WRITE(*,*) "------------------------------------------------------------------"
    WRITE(*,'(A)')  LOG
    IF (CHI%LREALSTORE) THEN
       WRITE(*,'(10F10.5)') REAL(CHI%RESPONSER(1:NDUMP,1:NDUMP,1),q)
       
       IF (CHI%LGAMMA) THEN
          WRITE(*,'( " real part wings")') 
          WRITE(*,'(10F10.5)') REAL(CHI%WINGR(1:NDUMP,1:3,1),q)
          WRITE(*,'( " real part cwings")') 
          WRITE(*,'(10F10.5)') REAL(CHI%CWINGR(1:NDUMP,1:3,1),q)
          WRITE(*,'( " head")') 
          WRITE(*,'(3(5X,3(2F10.5,2X)/))') CHI%HEAD(:,:,1)
       ENDIF
    ELSE
       WRITE(*,'(10F10.5)') REAL(CHI%RESPONSEFUN(1:NDUMP,1:NDUMP,1),q)
       WRITE(*,*)
       WRITE(*,'(10F10.5)') AIMAG(CHI%RESPONSEFUN(1:NDUMP,1:NDUMP,1))
       
       WRITE(*,'( " real part wings")') 
       IF (CHI%LGAMMA) THEN
          WRITE(*,'(10F10.5)') REAL(CHI%WING(1:NDUMP,1:3,1),q)
          WRITE(*,'( " imag part wings")') 
          WRITE(*,'(10F10.5)') AIMAG(CHI%WING(1:NDUMP,1:3,1))
          WRITE(*,'( " real part cwings")') 
          WRITE(*,'(10F10.5)') REAL(CHI%CWING(1:NDUMP,1:3,1),q)
          WRITE(*,'( " imag part cwings")') 
          WRITE(*,'(10F10.5)') AIMAG(CHI%CWING(1:NDUMP,1:3,1))
          WRITE(*,'( " head")') 
          WRITE(*,'(3(5X,3(2F10.5,2X)/))') CHI%HEAD(:,:,1)
       ENDIF
    ENDIF
       
  END SUBROUTINE DUMP_CHI

  SUBROUTINE DUMP_MAT( LOG, E, NP)
    IMPLICIT NONE
    CHARACTER (LEN=*) LOG
    COMPLEX(q) :: E(:,:)
    INTEGER NOMEGA
    INTEGER NP, NDUMP

    NDUMP=MIN(10,NP)

    WRITE(*,*) "------------------------------------------------------------------"
    WRITE(*,'(A)')  LOG
    WRITE(*,'(10F10.5)') REAL(E(1:NDUMP,1:NDUMP),q)
    WRITE(*,*)
    WRITE(*,'(10F10.5)') AIMAG(E(1:NDUMP,1:NDUMP))
    
  END SUBROUTINE DUMP_MAT


!********************** SUBROUTINE DUMP_RESPONSE_G    *****************
!
! writes the diagonal part of the response function to vasprun.xml
! 
!**********************************************************************

  SUBROUTINE DUMP_RESPONSE_G( CHI, WDESQ, LATT_CUR, STRING)
    USE constant
    USE mgrid
    USE lattice
    USE wave 

    IMPLICIT NONE
    TYPE (responsefunction) CHI
    TYPE (wavedes1) WDESQ
    TYPE (latt) LATT_CUR
    ! local
    INTEGER    NP, NI, NI_, NQ_IN_WGW, NG
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE
    COMPLEX(q) :: TMP
    CHARACTER (LEN=*) :: STRING

    REAL(q), ALLOCATABLE :: A(:,:)

    IF (CHI%NOMEGA_LOW/=1) RETURN

    SCALE=TPI**2

    DKX=(CHI%VKPT(1))*LATT_CUR%B(1,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(1,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(CHI%VKPT(1))*LATT_CUR%B(2,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(2,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(CHI%VKPT(1))*LATT_CUR%B(3,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(3,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(3,3)

    NP=WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2

    ALLOCATE(A(2,NP))
    A=0

    DO NI=1,WDESQ%NGVECTOR
       NI_=NI
       IF (WDESQ%LGAMMA) NI_=(NI-1)/2+1

       GX=(WDESQ%IGX(NI_)*LATT_CUR%B(1,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WDESQ%IGX(NI_)*LATT_CUR%B(2,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WDESQ%IGX(NI_)*LATT_CUR%B(3,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

       A(1,NI)=SQRT(GSQU*SCALE)
       IF (CHI%LREALSTORE) THEN
          A(2,NI)=CHI%RESPONSER  (NI,NI,1)
       ELSE
          A(2,NI)=CHI%RESPONSEFUN(NI,NI,1)
       ENDIF
    ENDDO

    IF (CHI%LGAMMA) THEN
       A(2,1)=(CHI%HEAD(1,1,1)+CHI%HEAD(2,2,1)+CHI%HEAD(3,3,1))/3
    ENDIF
      
    CALL XML_VECARRAY(STRING)
    CALL XML_ARRAY_REAL(A(:,1:NP))
    CALL XML_CLOSE_TAG

    DEALLOCATE(A)

  END SUBROUTINE DUMP_RESPONSE_G

!
! dump it at all frequencies
!

  SUBROUTINE DUMP_RESPONSE_G_OMEGA( CHI, WDESQ, LATT_CUR, STRING)
    USE constant
    USE mgrid
    USE lattice
    USE wave 

    IMPLICIT NONE
    TYPE (responsefunction) CHI
    TYPE (wavedes1) WDESQ
    TYPE (latt) LATT_CUR
    ! local
    INTEGER    NP, NI, NI_, NQ_IN_WGW, NG
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE
    COMPLEX(q) :: TMP
    CHARACTER (LEN=*) :: STRING

    INTEGER :: NOMEGA
    REAL(q), ALLOCATABLE :: A(:,:)
    CHARACTER (LEN=8) :: i_char

    IF (CHI%NOMEGA_LOW/=1) RETURN
    SCALE=TPI**2
    DKX=(CHI%VKPT(1))*LATT_CUR%B(1,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(1,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(CHI%VKPT(1))*LATT_CUR%B(2,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(2,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(CHI%VKPT(1))*LATT_CUR%B(3,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(3,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(3,3)

    NP=WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2

    ALLOCATE(A(2,NP))

    DO NOMEGA=CHI%NOMEGA_LOW, CHI%NOMEGA_HIGH

    A=0

    DO NI=1,WDESQ%NGVECTOR
       NI_=NI
       IF (WDESQ%LGAMMA) NI_=(NI-1)/2+1

       GX=(WDESQ%IGX(NI_)*LATT_CUR%B(1,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WDESQ%IGX(NI_)*LATT_CUR%B(2,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WDESQ%IGX(NI_)*LATT_CUR%B(3,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

       A(1,NI)=SQRT(GSQU*SCALE)
       IF (CHI%LREALSTORE) THEN
          A(2,NI)=CHI%RESPONSER  (NI,NI,NOMEGA)
       ELSE
          A(2,NI)=CHI%RESPONSEFUN(NI,NI,NOMEGA)
       ENDIF
    ENDDO

    IF (CHI%LGAMMA) THEN
       A(2,1)=(CHI%HEAD(1,1,NOMEGA)+CHI%HEAD(2,2,NOMEGA)+CHI%HEAD(3,3,NOMEGA))/3
    ENDIF

    WRITE(i_char, '(I8)') NOMEGA

    CALL XML_VECARRAY(STRING// TRIM(ADJUSTL(i_char)))
    CALL XML_ARRAY_REAL(A(:,1:NP))
    CALL XML_CLOSE_TAG

    ENDDO

    DEALLOCATE(A)

  END SUBROUTINE DUMP_RESPONSE_G_OMEGA



!********************** SUBROUTINE DUMP_RESPONSE_G_MIC    *************
!
! writes the diagonal part of the response function to vasprun.xml
! this version is for the polarizability matrix 
! and requires a multiplications by 4 pi e^2 and addition of 1
! 
!**********************************************************************

  SUBROUTINE DUMP_RESPONSE_G_MIC( CHI, WDESQ, LATT_CUR, STRING)
    USE constant
    USE mgrid
    USE lattice
    USE wave 

    IMPLICIT NONE
    TYPE (responsefunction) CHI
    TYPE (wavedes1) WDESQ
    TYPE (latt) LATT_CUR
    ! local
    INTEGER    NP, NI, NI_, NQ_IN_WGW, NG
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE
    COMPLEX(q) :: TMP
    CHARACTER (LEN=*) :: STRING

    REAL(q), ALLOCATABLE :: A(:,:)

    IF (CHI%NOMEGA_LOW/=1) RETURN

    SCALE=TPI**2

    DKX=(CHI%VKPT(1))*LATT_CUR%B(1,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(1,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(CHI%VKPT(1))*LATT_CUR%B(2,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(2,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(CHI%VKPT(1))*LATT_CUR%B(3,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(3,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(3,3)

    NP=WDESQ%NGVECTOR
    IF (WDESQ%LGAMMA) NP=NP*2

    ALLOCATE(A(2,NP))

    A=0

    DO NI=1,NP
       NI_=NI
       IF (WDESQ%LGAMMA) NI_=(NI-1)/2+1
       GX=(WDESQ%IGX(NI_)*LATT_CUR%B(1,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WDESQ%IGX(NI_)*LATT_CUR%B(2,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WDESQ%IGX(NI_)*LATT_CUR%B(3,1)+WDESQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WDESQ%IGZ(NI_)*LATT_CUR%B(3,3))

       GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2

       A(1,NI)=SQRT(GSQU*SCALE)
       IF (CHI%LREALSTORE) THEN
          A(2,NI)=1-CHI%RESPONSER  (NI,NI,1)*WDESQ%DATAKE(NI, 1)
       ELSE
          A(2,NI)=1-CHI%RESPONSEFUN(NI,NI,1)*WDESQ%DATAKE(NI, 1)
       ENDIF
    ENDDO
    IF (CHI%LGAMMA) THEN
       A(2,1)=1-(CHI%HEAD(1,1,1)+CHI%HEAD(2,2,1)+CHI%HEAD(3,3,1))*WDESQ%DATAKE(1, 1)/3
    ENDIF

    CALL XML_VECARRAY(STRING)
    CALL XML_ARRAY_REAL(A(:,1:NP))
    CALL XML_CLOSE_TAG

    DEALLOCATE(A)

  END SUBROUTINE DUMP_RESPONSE_G_MIC

!*********************************************************************
!
! This routine clears a resonsefunction array
!
!*********************************************************************

  SUBROUTINE CLEAR_RESPONSE( CHI )
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
! shmem TODO: maybe lock write, but really should not matter
    CHI%RESPONSEFUN=0
    CHI%HEAD=0
    CHI%WING=0
    CHI%CWING=0

  END SUBROUTINE CLEAR_RESPONSE

!
! this version should be used, if SHMEM allocation was used
! it first waits for all cores to come to this point
! before clearing the data on the root node, than a second barrier
! is used
!

  SUBROUTINE CLEAR_RESPONSE_SHMEM( CHI )
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    TYPE (wavedes), POINTER :: WGW      ! communicator for CHI
    
#ifdef use_shmem
    IF (CHI%LSHMEM) THEN
       ! wait for all nodes to get here so that RESPONSEFUN is no longer used
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )      
       ! clear on root node only
       IF (CHI%LLEAD) CHI%RESPONSEFUN=0
       ! wait on all nodes for zero setting to finish
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
    ELSE
#endif
       CHI%RESPONSEFUN=0
#ifdef use_shmem
    ENDIF
#endif
    CHI%HEAD=0
    CHI%WING=0
    CHI%CWING=0

  END SUBROUTINE CLEAR_RESPONSE_SHMEM


!*********************************************************************
!
! Tiny helper routine to scale responsefunction array by
! some real number
!
!*********************************************************************

  SUBROUTINE SCALE_RESPONSE( CHI, SCALE)
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    REAL (q) :: SCALE
    
#ifdef use_shmem
    INTEGER ierror
    IF (CHI%LSHMEM) THEN
       ! wait for all nodes to get here so that RESPONSEFUN is no longer used
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
       ! clear on root node only
       IF (CHI%LLEAD) CHI%RESPONSEFUN=CHI%RESPONSEFUN*SCALE
       ! wait on all nodes for zero setting to finish
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
    ELSE
#endif
       CHI%RESPONSEFUN=CHI%RESPONSEFUN*SCALE
#ifdef use_shmem
    ENDIF
#endif
    CHI%HEAD=CHI%HEAD*SCALE
    CHI%WING=CHI%WING*SCALE
    CHI%CWING=CHI%CWING*SCALE
      
  END SUBROUTINE SCALE_RESPONSE


  SUBROUTINE SCALE_RESPONSE_RESPONSE( CHI, SCALE )
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    REAL (q) :: SCALE
    
#ifdef use_shmem
    INTEGER ierror
    IF (CHI%LSHMEM) THEN
       ! wait for all nodes to get here so that RESPONSEFUN is no longer used
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
       ! clear on root node only
       IF (CHI%LLEAD) CHI%RESPONSEFUN=CHI%RESPONSEFUN*SCALE
       ! wait on all nodes for zero setting to finish
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
    ELSE
#endif
       CHI%RESPONSEFUN=CHI%RESPONSEFUN*SCALE
#ifdef use_shmem
    ENDIF
#endif
      
  END SUBROUTINE SCALE_RESPONSE_RESPONSE

!
! scale the element (1,1) of a responsefunction array
!

  SUBROUTINE SCALE_RESPONSE_0( CHI, SCALE )
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    REAL (q) :: SCALE
    
#ifdef use_shmem
    INTEGER ierror
    IF (CHI%LSHMEM) THEN
       ! wait for all nodes to get here so that RESPONSEFUN is no longer used
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
       ! clear on root node only
       IF (CHI%LLEAD) CHI%RESPONSEFUN(1,1,:)=CHI%RESPONSEFUN(1,1,:)*SCALE
       ! wait on all nodes for zero setting to finish
       CALLMPI( M_barrier(CHI%COMM_SHMEM) )
    ELSE
#endif
       CHI%RESPONSEFUN(1,1,:)=CHI%RESPONSEFUN(1,1,:)*SCALE
#ifdef use_shmem
    ENDIF
#endif
      
  END SUBROUTINE SCALE_RESPONSE_0


!*********************************************************************
!
! This routine performes Hilbert transform of the screened 
! potential using the precalculated tables TABLE
! essentially  Equ. (22) in Shishkin, Kresse, PRB 74, 035101
!
!  i/(2 pi) int dw' W(w')/(w'+w+ i eta)
!
! the TABLE is set up by SETUP_HILBERT_TABLE
! if the inner K loop is parallelized over k-points
! summation over COMM_KINTER needs to be done
!
! sigma(ir) = sum_iw TABLE(iw, ir) W (iw)
!
!*********************************************************************

  SUBROUTINE DO_POT_HILBERT_TABLE(CHI, WPOT, WGW, TABLE, LSUMK)
    USE c2f_interface, ONLY : LOCKSEM,UNLOCKSEM
    IMPLICIT NONE
    TYPE (responsefunction) CHI        ! screened potential
    TYPE (responsefunction) WPOT        ! transformed potential
    TYPE (wavedes), POINTER :: WGW      ! communicator for WPOT
    COMPLEX(q) :: TABLE(:,:)            ! table for the Kramers Kronig transformation
    LOGICAL LSUMK
  ! local
    INTEGER :: NALLOC1, NALLOC2, NOMEGA
#ifdef use_shmem
    INTEGER :: ierror
#endif

    IF (CHI%LREALSTORE .OR. WPOT%LREALSTORE) THEN
       WRITE(*,*) 'internal error in DO_POT_HILBERT_TABLE: LREALSTORE is not supported yet'
       STOP
    ENDIF

    ! first clear the result array
    ! clear response function on all nodes (includes a barrier for shmem)
    ! barrier all nodes using same SHMEM segment
    CALL CLEAR_RESPONSE_SHMEM( WPOT )

    NALLOC1=SIZE(CHI%RESPONSEFUN,1)
    NALLOC2=SIZE(CHI%RESPONSEFUN,1)

    ! WPOT%RESPONSEFUN(:,:,:)= SUM_I CHI%RESPONSEFUN(:,:,I)*TABLE(I,:)

#ifdef use_shmem
    ! lock the entire array and perform update
    ! this lock is only safe if a barrier over all shmem nodes is done before and after
    IF (WPOT%LSHMEM) CALL LOCKSEM( WPOT%SHMEMLOCK, 0)
#endif
    CALL ZGEMM( 'N', 'N' , NALLOC1*NALLOC2 , WPOT%NOMEGA , CHI%NOMEGA  , & 
         (1.0_q,0.0_q) , CHI%RESPONSEFUN(1,1,1), NALLOC1*NALLOC2 , & 
         TABLE(CHI%NOMEGA_LOW, WPOT%NOMEGA_LOW) , SIZE(TABLE,1) , &
         (1.0_q, 0.0_q) , WPOT%RESPONSEFUN(1,1,1) , NALLOC1*NALLOC2 )
#ifdef use_shmem
    IF (WPOT%LSHMEM) CALL UNLOCKSEM( WPOT%SHMEMLOCK, 0)
#endif

#ifdef use_shmem
    IF (WPOT%LSHMEM) THEN
       ! first barrier so that all calculations are certainly done on a shmem node
       CALLMPI( M_barrier(WPOT%COMM_SHMEM) )

       IF (WPOT%LLEAD) THEN
          ! sum results over all nodes on "root" node of the shmem cluster
          DO NOMEGA=1,WPOT%NOMEGA
            CALLMPI( M_sum_z(WPOT%COMM_INTER_SHMEM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
          ENDDO
       ENDIF

       ! second barrier so that all nodes wait until all data are properly merged
       CALLMPI( M_barrier(WPOT%COMM_SHMEM) )
       
    ELSE
#endif
    ! sum result over all nodes
    IF (LSUMK) THEN
       DO NOMEGA=1,WPOT%NOMEGA
         CALLMPI( M_sum_z(WGW%COMM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
       ENDDO
    ELSE
       DO NOMEGA=1,WPOT%NOMEGA
          CALLMPI( M_sum_z(WGW%COMM_INTER, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
       ENDDO
    ENDIF
#ifdef use_shmem
    ENDIF
#endif
    
  END SUBROUTINE DO_POT_HILBERT_TABLE


!*********************************************************************
!
! This routine merges the CHI array from all nodes and stores the 
! results in WPOT
! care is taken to copy the correct frequency slots from CHI to 
! WPOT
!
!*********************************************************************

  SUBROUTINE MERGE_FREQU(CHI, WPOT, WGW, LSUMK )
    USE c2f_interface, ONLY : LOCKSEM,UNLOCKSEM
    IMPLICIT NONE
    TYPE (responsefunction) CHI         ! full response function
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (wavedes), POINTER :: WGW      ! communicator for WPOT
    LOGICAL LSUMK
  ! local
    INTEGER :: NALLOC1, NALLOC2, I_WPOT, I_CHI, NOMEGA
#ifdef use_shmem
    INTEGER :: ierror
#endif

    NALLOC1=SIZE(CHI%RESPONSEFUN,1)
    NALLOC2=SIZE(CHI%RESPONSEFUN,2)

    ! clear response function on all nodes (includes a barrier for shmem)
    ! barrier all nodes using same SHMEM segment
    CALL CLEAR_RESPONSE_SHMEM( WPOT )

    DO I_WPOT=1, WPOT%NOMEGA
       DO I_CHI=1, CHI%NOMEGA
          IF (CHI%COMEGA(I_CHI)==WPOT%COMEGA(I_WPOT)) THEN
             ! WPOT%RESPONSEFUN = CHI%RESPONSEFUN
! I believe the lock in not necessary, since the never two nodes write to one destination
#ifdef use_shmem
             IF (WPOT%LSHMEM) CALL LOCKSEM( WPOT%SHMEMLOCK, I_WPOT)
#endif
             CALL ZCOPY( NALLOC1*NALLOC2, CHI%RESPONSEFUN(1,1,I_CHI), 1,WPOT%RESPONSEFUN(1,1,I_WPOT), 1)
#ifdef use_shmem
             IF (WPOT%LSHMEM) CALL UNLOCKSEM( WPOT%SHMEMLOCK, I_WPOT)
#endif
             CALL ZCOPY( 3*NALLOC1, CHI%WING(1,1,I_CHI), 1,WPOT%WING(1,1,I_WPOT), 1)
             CALL ZCOPY( 3*NALLOC1, CHI%CWING(1,1,I_CHI), 1,WPOT%CWING(1,1,I_WPOT), 1)
             CALL ZCOPY( 3*3, CHI%HEAD(1,1,I_CHI), 1,WPOT%HEAD(1,1,I_WPOT), 1)
          ENDIF
       ENDDO
    ENDDO

#ifdef use_shmem
    IF (WPOT%LSHMEM) THEN
       ! barrier all nodes using same SHMEM segment
       CALLMPI( M_barrier(WPOT%COMM_SHMEM) )
       IF (WPOT%LLEAD) THEN
          ! sum results over all nodes on "root" node of the shmem cluster
          DO NOMEGA=1,WPOT%NOMEGA
            CALLMPI( M_sum_z(WPOT%COMM_INTER_SHMEM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
          ENDDO
       ENDIF

       ! second barrier so that all nodes wait until all data are properly merged
       CALLMPI( M_barrier(WPOT%COMM_SHMEM) )
    ELSE
#endif
    ! normal non shmem version
       IF (LSUMK) THEN
          DO NOMEGA=1,WPOT%NOMEGA
            CALLMPI( M_sum_z(WGW%COMM, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
          ENDDO
       ELSE
          DO NOMEGA=1,WPOT%NOMEGA
            CALLMPI( M_sum_z(WGW%COMM_INTER, WPOT%RESPONSEFUN(1,1,NOMEGA), NALLOC1*NALLOC2))
          ENDDO
       ENDIF
#ifdef use_shmem
    ENDIF
#endif

    ! sum WING and HEAD over all nodes
    ! could be replaced by more efficient scatter
    IF (LSUMK) THEN
       CALLMPI( M_sum_z(WGW%COMM, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGW%COMM, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGW%COMM, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
    ELSE
       CALLMPI( M_sum_z(WGW%COMM_INTER, WPOT%WING(1,1,1)       , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGW%COMM_INTER, WPOT%CWING(1,1,1)      , NALLOC1*3*WPOT%NOMEGA))
       CALLMPI( M_sum_z(WGW%COMM_INTER, WPOT%HEAD(1,1,1)       , 3*3*WPOT%NOMEGA))
    ENDIF
!    WRITE(*,*) ' done MERGE_FREQU'

  END SUBROUTINE MERGE_FREQU

  SUBROUTINE COMM_INTER_SHMEM( WPOT, WGW, LSUMK )
    IMPLICIT NONE
    TYPE (responsefunction) WPOT        ! spectral rep. of response function
    TYPE (wavedes), POINTER :: WGW      ! communicator for WPOT
    LOGICAL LSUMK
  ! local
#ifdef use_shmem
    INTEGER :: ierror

    IF (WPOT%LSHMEM) THEN
       IF (ASSOCIATED(WPOT%COMM_INTER_SHMEM)) THEN
         CALLMPI( MPI_COMM_FREE(WPOT%COMM_INTER_SHMEM%MPI_COMM, ierror))
       ELSE
         ALLOCATE(WPOT%COMM_INTER_SHMEM)
       ENDIF
       IF (LSUMK) THEN
          CALL MPI_COMM_SPLIT(WGW%COMM%MPI_COMM, WPOT%COMM_SHMEM%NODE_ME, WGW%COMM%NODE_ME, WPOT%COMM_INTER_SHMEM%MPI_COMM , ierror)
       ELSE
          CALLMPI( MPI_COMM_SPLIT(WGW%COMM_INTER%MPI_COMM, WPOT%COMM_SHMEM%NODE_ME, WGW%COMM_INTER%NODE_ME, WPOT%COMM_INTER_SHMEM%MPI_COMM, ierror))
       ENDIF

       CALLMPI( M_initc(  WPOT%COMM_INTER_SHMEM))

       IF ( ierror /= MPI_success ) THEN
          WRITE(*,*) 'MERGE_FREQU: Error in  MPI_COMM_SPLIT',ierror
          STOP
       ENDIF
    ENDIF
#endif
  END SUBROUTINE
  

!***********************************************************************
!  Function EMPTY_XI_ORBITAL returns .TRUE. if the orbital
!  is considered to be empty in the calculation of XI
!  returns true only if the orbital occupancy is smaller than
!  XI_EMPTY_THRESHHOLD
!***********************************************************************

  FUNCTION EMPTY_XI_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL EMPTY_XI_ORBITAL
    REAL(q) :: F
    IF (ABS(F) <XI_EMPTY_THRESHHOLD) THEN
       EMPTY_XI_ORBITAL=.TRUE.
    ELSE
       EMPTY_XI_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION EMPTY_XI_ORBITAL

!***********************************************************************
!  Function FILLED_XI_ORBITAL returns .TRUE. if the orbital
!  is considered to be filled in the XI
!  FILLED_XI_ORBITAL's are skipped when an unoccupied band is required
!***********************************************************************

  FUNCTION FILLED_XI_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL FILLED_XI_ORBITAL
    REAL(q) :: F
    IF (ABS(F-1) <XI_EMPTY_THRESHHOLD) THEN
       FILLED_XI_ORBITAL=.TRUE.
    ELSE
       FILLED_XI_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION FILLED_XI_ORBITAL

!***********************************************************************
!  function LAST_FILLED_XI returns the last filled band
!  modulo the number of bands done in parallel
!***********************************************************************

  FUNCTION LAST_FILLED_XI( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_XI
    INTEGER K1, ISP
  ! local
    INTEGER NB

    DO NB=W%WDES%NB_TOT, 1, -1
       ! if non  empty band is required
       IF (ABS(W%FERTOT( NB, K1, ISP))>XI_EMPTY_THRESHHOLD ) EXIT
    ENDDO

    ! round to next larger value modulo W%WDES%NB_PAR
    LAST_FILLED_XI=((NB+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)*W%WDES%NB_PAR
  END FUNCTION LAST_FILLED_XI


  ! this version is without the modulo
  FUNCTION LAST_FILLED_XI_NOMOD( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_XI_NOMOD
    INTEGER K1, ISP
  ! local
    INTEGER NB

    DO NB=W%WDES%NB_TOT, 1, -1
       ! if non  empty band is required
       IF (ABS(W%FERTOT( NB, K1, ISP))>XI_EMPTY_THRESHHOLD ) EXIT
    ENDDO
    LAST_FILLED_XI_NOMOD=NB
  END FUNCTION LAST_FILLED_XI_NOMOD



!***********************************************************************
!  FIRST_EMPTY_XI_LOCAL returns the first local emtpy band in 
!  FERWE
!***********************************************************************


  FUNCTION FIRST_EMPTY_XI_LOCAL( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_XI_LOCAL
    INTEGER K1, ISP
  ! local
    INTEGER NB
    DO NB=1,W%WDES%NBANDS
       IF ( ABS(W%FERWE( NB, K1, ISP)-1)>XI_EMPTY_THRESHHOLD ) EXIT
    ENDDO
    
    FIRST_EMPTY_XI_LOCAL=NB
     
  END FUNCTION FIRST_EMPTY_XI_LOCAL


!***********************************************************************
!  LAST_EMPTY_XI_LOCAL returns the last local emtpy band
!  this uses the auxilary array which is set to 0 for bands
!  that do not need to be considered 
!***********************************************************************

  FUNCTION LAST_EMPTY_XI_LOCAL( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_EMPTY_XI_LOCAL
    INTEGER K1, ISP
  ! local
    INTEGER NB
    DO NB=W%WDES%NBANDS,1,-1
       IF ( W%AUX( NB, K1, ISP)/=0 ) EXIT
    ENDDO
    
    LAST_EMPTY_XI_LOCAL=NB
  END FUNCTION LAST_EMPTY_XI_LOCAL

!***********************************************************************
!  function FIRST_EMPTY_XI returns the first empty band
!  modulo the number of bands done in parallel
!***********************************************************************

  FUNCTION FIRST_EMPTY_XI( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_XI
    INTEGER K1, ISP
  ! local
    INTEGER NB

    DO NB=1,W%WDES%NB_TOT
       IF ( ABS(W%FERTOT( NB, K1, ISP)-1)>XI_EMPTY_THRESHHOLD ) EXIT
    ENDDO

    ! round to next smaller value
    FIRST_EMPTY_XI=((NB-1)/W%WDES%NB_PAR+1)*W%WDES%NB_PAR
    
  END FUNCTION FIRST_EMPTY_XI

  FUNCTION FIRST_EMPTY_XI_NOMOD( W, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_XI_NOMOD
    INTEGER K1, ISP
  ! local
    INTEGER NB
    
    DO NB=1,W%WDES%NB_TOT
       IF ( ABS(W%FERTOT( NB, K1, ISP)-1)>XI_EMPTY_THRESHHOLD ) EXIT
    ENDDO

    ! round to next smaller value
    FIRST_EMPTY_XI_NOMOD=NB
    
  END FUNCTION FIRST_EMPTY_XI_NOMOD

END MODULE chi_base


!***********************************************************************
!
! small subroutine to multiply a real vector by a complex number 
!
!***********************************************************************

SUBROUTINE ZDAXPY( NP, A, IN, INSTRIDE, OUT, OUTSTRIDE)
  USE prec
  INTEGER NP              ! number of grid points
  COMPLEX(q) A            ! complex number
  REAL(q)  :: IN(NP)      ! input vector (real)
  INTEGER  :: INSTRIDE    ! stride for input vector
  REAL(q)  :: OUT(2*NP)   ! output vector declared real
  INTEGER  :: OUTSTRIDE

  CALL DAXPY( NP, REAL(A,q), IN(1), INSTRIDE,  OUT(1), 2*OUTSTRIDE)
  CALL DAXPY( NP, AIMAG(A) , IN(1), INSTRIDE,  OUT(2), 2*OUTSTRIDE)

END SUBROUTINE ZDAXPY

!***********************************************************************
!
! small subroutine to calculate the inproduct between a complex vector 
! and a real vector
!
!***********************************************************************

FUNCTION ZDDOTC( NP, ZIN, ZINSTRIDE, AIN, AINSTRIDE)
  USE prec
  COMPLEX(q) ZDDOTC
  INTEGER NP              ! number of grid points
  REAL(q)  :: ZIN(2*NP)   ! input vector (complex) declared real
  INTEGER  :: ZINSTRIDE   ! stride for input vector
  REAL(q)  :: AIN(NP)     ! real input vector
  INTEGER  :: AINSTRIDE
  REAL(q)  :: A, B
  REAL(q), EXTERNAL :: DDOT

  
  A=DDOT(NP, ZIN(1), 2*ZINSTRIDE, AIN(1), AINSTRIDE)
  B=DDOT(NP, ZIN(2), 2*ZINSTRIDE, AIN(1), AINSTRIDE)

  ZDDOTC=CMPLX(A,-B,q)

END FUNCTION ZDDOTC


!***********************************************************************
!
! small subroutine to copy in real space an RGRID array (local potential)
! to a COMPLEX (wave function like array)
!
!***********************************************************************

SUBROUTINE RL_COPY_RGRID_WAVE( A, C, SCALE, GRID )
  USE mgrid

  TYPE (grid_3d) GRID
  REAL(q)    :: SCALE
  RGRID      :: A(GRID%RL%NP)
#ifdef gammareal
  REAL(q) :: C(GRID%RL%NP)
#else
  COMPLEX(q) :: C(GRID%RL%NP)
#endif
  INTEGER    :: N, NP

  NP=GRID%RL%NP
  !
  !   C=A*SCALE
  !
  DO N=1,NP
     C(N)=A(N)
  ENDDO

END SUBROUTINE RL_COPY_RGRID_WAVE


SUBROUTINE RL_COPY_WAVE_RGRID( C, A, SCALE, GRID )
  USE mgrid

  TYPE (grid_3d) GRID
  REAL(q) :: SCALE
#ifdef gammareal
  REAL(q) :: C(GRID%RL%NP)
#else
  COMPLEX(q) :: C(GRID%RL%NP)
#endif
  RGRID      :: A(GRID%RL%NP)
  INTEGER    :: N, NP

  NP=GRID%RL%NP
  !
  !   A=C*SCALE
  !
  DO N=1,NP
     A(N)=C(N)
  ENDDO

END SUBROUTINE RL_COPY_WAVE_RGRID

!=======================================================================
!
! this routine returns a pointer to an SEQUENCED F77 like array
! with a given storage convention
! 1. dimension is N1, second one N2 and third N3
!
!=======================================================================

SUBROUTINE SET_RESPONSER(CW_P, N1, N2, N3, CW)
  USE prec
  IMPLICIT NONE

  INTEGER N1,N2,N3
  REAL(q), POINTER :: CW_P(:,:,:)
  REAL(q), TARGET  :: CW(N1,N2,N3)
  CW_P => CW(:,:,:)
  
END SUBROUTINE SET_RESPONSER


SUBROUTINE SET_RESPONSE_ONE(CW_P, N1, N2,  CW)
  USE prec
  IMPLICIT NONE

  INTEGER N1,N2
  REAL(q), POINTER :: CW_P(:,:)
  REAL(q), TARGET  :: CW(N1,N2)
  CW_P => CW(:,:)
  
END SUBROUTINE SET_RESPONSE_ONE
