!#define debug
#include "symbol.inc"
MODULE hamil_lrf
  USE prec
  USE hamil_lr
CONTAINS

!************************ SUBROUTINE LRF_COMMUTATOR *********************
!
! this subroutine evaluates
!        |xi_nk>=  ([H(0), r] - e(0) [S(0),r]- e(1) S(0)) | phi(0)_nk>
!               =-i (d/dk [H_k - e(0) S_k- d e_k/d k S(0)]) | phi(0)_nk >
! where H_k (S_k, e_k) is the Hamiltonian (overlap, eigenenergy) in 
! dependency of k
! more specifically:
!    |xi>  = - hbar^2/ m_e nabla 
!                 + sum_ij | p_i> D(1)_ij <p_j | phi > 
!                 + sum_ij | p_i> D(0)_ij - e(0) Q(0)_ij < p_j r| phi >
!                 - sum_ij |r p_i>D(0)_ij - e(0) Q(0)_ij < p_j  | phi >
!                 - sum_ij | p_i> e(1) Q(0)_ij <p_j | phi >
!                 - e(1) |phi>
!
! the last two terms involving e(1) are added to make the xi orthogonal to phi_0 
!  (<xi | phi_0> = 0)
! D(1) is usually 0, but could be set to
!  - hbar^2/ m_e < psi_i| nabla |psi_j > - < tilde psi_i| nabla |tilde psi_j >
! for LNABLA ==.TRUE., terms involving <p_i r| are then ommited
! IDIR supplies the cartesian index of r (r_i)
!
! for LI=.TRUE.,  (d/dk [H_k - e(0) S_k] - d e_k/d k S(0)]) | phi(0)_nk >
! is returned
! in this case the change of the eigenvalues is real, whereas in the
! previous case it is complex
!
! set upon return:
! <G | xi>        is returned in WXI%CW
! <p_j r| phi >   is returned in WXI%CPROJ,
! e(1)            is returned in WXI%CELEN
!
!
!***********************************************************************

  SUBROUTINE LRF_COMMUTATOR(GRID,INFO,LATT_CUR, &
       NONLR_S, NONLR_D, NONL_S, NONL_D, W0, WXI, WDES, &
        LMDIM, CDIJ, CDIJ0, CQIJ, LNABLA, LI, IDIR, CSHIFT, RMS, ICOUEV, LXI_SET)
    USE prec
      
    USE wave
    USE wave_high
    USE base
    USE lattice
    USE mpimy
    USE mgrid
    
    USE nonl_high
    USE hamil
    USE constant
    USE wave_mpi

    USE openmp, ONLY : omp_nthreads_hi
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    TYPE (info_struct) INFO
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S, NONLR_D
    TYPE (nonl_struct) NONL_S, NONL_D
    TYPE (wavespin)    W0, WXI, W1
    TYPE (wavedes)     WDES

    INTEGER LMDIM  
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CDIJ0(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    REAL(q) RMS        ! magnitude of the residual vector
    INTEGER ICOUEV     ! number of H | phi> evaluations
    LOGICAL LNABLA
    LOGICAL LI
    INTEGER IDIR
    REAL(q) CSHIFT
    LOGICAL, OPTIONAL :: LXI_SET
! local work arrays and structures
    TYPE (wavedes1)  WDES1                     ! descriptor for one k-point
    TYPE (wavefun1), TARGET :: W0_1(WDES%NSIM) ! current wavefunction
    TYPE (wavefun1)  WTMP(WDES%NSIM)           ! see below

    COMPLEX(q),ALLOCATABLE:: CF(:,:)
    GDEF, TARGET, ALLOCATABLE ::  CPROJ(:,:)

    INTEGER :: NSIM                  ! number of bands treated simultaneously
    INTEGER :: NODE_ME, IONODE
    INTEGER :: NP, ISP, NK, NPL, NGVECTOR, NB_DONE, N, IDUMP, ISPINOR, LD, M, MM
    REAL(q) :: FNORM
    COMPLEX(q) :: ORTH
    INTEGER :: NB(WDES%NSIM)         ! contains a list of bands currently optimized
    REAL(q) :: EVALUE(WDES%NSIM)     ! eigenvalue during optimization
    LOGICAL :: LDO(WDES%NSIM)        ! band finished
    LOGICAL :: LSTOP,LNEWB
    COMPLEX(q) ::C
    COMPLEX(q) ::CI                ! imaginary

    PROFILING_START('lrf_commutator')

    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=WDES%COMM%NODE_ME
    IONODE =WDES%COMM%IONODE
#endif

    IF (LI) THEN
       CI=(0.0_q,1.0_q)
#ifdef gammareal
       WRITE(0,*) 'internal error in LRF_COMMUTATOR: for the gamma-point code, LI must be set to .FALSE.'
       STOP
#endif
    ELSE
       CI=(1.0_q,0.0_q)
    ENDIF
    
!=======================================================================
!  INITIALISATION:
! maximum  number of bands treated simultaneously
!=======================================================================
      RMS   =0
      ICOUEV=0
      NSIM=WDES%NSIM

      ALLOCATE(CF(WDES%NRPLWV,NSIM),CPROJ(WDES%NPROD,NSIM))
      LD=WDES%NRPLWV

      DO NP=1,NSIM
         ALLOCATE(W0_1(NP)%CR(GRID%MPLWV*WDES%NRSPINORS))
      ENDDO

      WXI%CPROJ=0
!=======================================================================
      spin:    DO ISP=1,WDES%ISPIN
      kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID) 

      NPL=WDES1%NPL
      NGVECTOR=WDES1%NGVECTOR

      IF (INFO%LREAL) THEN
        CALL PHASER (GRID,LATT_CUR,NONLR_S,NK,WDES)
        CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK,WDES,IDIR)
      ELSE
        CALL PHASE(WDES,NONL_S,NK)
        NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
      ENDIF

!=======================================================================
      NB_DONE=0                   ! index of the bands allready optimised
      bands: DO
        NB=0                      ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
        IDUMP=0
        newband: DO NP=1,NSIM
        LNEWB=.FALSE.
        IF (NB_DONE<WXI%WDES%NBANDS) THEN
           NB_DONE=NB_DONE+1; LNEWB=.TRUE.
        ENDIF
        IF (LNEWB) THEN
           N     =NB_DONE
           NB(NP)=NB_DONE
           ICOUEV=ICOUEV+1

           CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)
#ifdef debug
           IDUMP=2
#endif
#ifdef MPI
           IF (NODE_ME /= IONODE) IDUMP=0
#endif
           ! fft to real space
           DO ISPINOR=0,WDES%NRSPINORS-1
              CALL FFTWAV(NGVECTOR, WDES%NINDPW(1,NK),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
           ENDDO
           ! WTMP is identical to W0_1, except for CPROJ entry
           ! which will contain the derivative of the projectors
           ! w.r.t. k after the call HAMILMU_LRF
           WTMP(NP)=W0_1(NP)
           WTMP(NP)%CPROJ => CPROJ(:,NP)

           EVALUE(NP)=W0_1(NP)%CELEN
        ENDIF
        ENDDO newband
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
        LSTOP=.TRUE.
        LDO  =.FALSE.
        DO NP=1,NSIM
           IF ( NB(NP) /= 0 ) THEN
              LSTOP  =.FALSE.
              LDO(NP)=.TRUE.     ! band not finished yet
           ENDIF
        ENDDO
        IF (LSTOP) EXIT bands
!=======================================================================
! determine gradient and store it
!=======================================================================
        
        DO NP=1,NSIM
           N=NB(NP); IF (.NOT. LDO(NP)) CYCLE
           IF (PRESENT (LXI_SET)) THEN
              CF(:,NP)=WXI%CW(:,N,NK,ISP)
           ELSE
              CF(:,NP)=0
           ENDIF
        ENDDO

        !  store H | psi > temporarily
        !  to have uniform stride for result array
        CALL HAMILTMU_COMMUTATOR(WDES1,W0_1,WTMP,NONLR_S,NONLR_D,NONL_S,NONL_D, &
                & GRID,  INFO%LREAL, EVALUE, LATT_CUR, &
                & LMDIM,CDIJ(1,1,1,ISP),CDIJ0(1,1,1,ISP),CQIJ(1,1,1,ISP), &
                & CF(1,1),LD, NSIM, LDO, WDES%VKPT(1:3,NK), LNABLA, IDIR, CSHIFT)

        i2: DO NP=1,NSIM
           N=NB(NP); IF (.NOT. LDO(NP)) CYCLE i2
           ORTH =0
           FNORM=0
           DO ISPINOR=0,WDES%NRSPINORS-1
           DO M=1,NGVECTOR
              MM=M+ISPINOR*NGVECTOR
              CF(MM,NP)=(CF(MM,NP)-W0_1(NP)%CELEN*W0_1(NP)%CW(MM))*CI
              IF (WDES%LSPIRAL.AND.WDES%DATAKE(M,ISPINOR+1,NK)>INFO%ENINI) CF(MM,NP)=0
              FNORM =FNORM+CF(MM,NP)*CONJG(CF(MM,NP))
              ORTH  =ORTH +GREAL(CF(MM,NP)*CONJG(W0_1(NP)%CW(MM)))
              WXI%CW(MM,N,NK,ISP)=CF(MM,NP)
           ENDDO
           WXI%CELEN(N,NK,ISP)  =W0_1(NP)%CELEN*CI
           WXI%CPROJ(:,N,NK,ISP)=WTMP(NP)%CPROJ*CI

           ENDDO
           CALLMPI( M_sum_s(WDES%COMM_INB, 1, FNORM, 0._q, 0._q, 0._q))
           CALLMPI( M_sum_z(WDES%COMM_INB, ORTH, 1))

           IF (ABS(ORTH)>1E-4) THEN
              WRITE(0,*)'LRF_COMMUTATOR internal error: the vector H(1)-e(1) S(1) |phi(0)> is not orthogonal to |phi(0)>',ORTH
!             STOP
           ENDIF

           IF (IDUMP==2) WRITE(*,'(I3,E11.4,"R ",2E11.4,"O ",2E11.4,"E ",2E14.7)') N,SQRT(ABS(FNORM)),ORTH,WXI%CELEN(N,NK,ISP),WXI%FERWE(N,NK,ISP)
           RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W0%FERWE(N,NK,ISP)*SQRT(ABS(FNORM))/WDES%NB_TOT

!=======================================================================
! move onto the next block of bands
!=======================================================================
        ENDDO i2
!=======================================================================
      ENDDO bands
      ENDDO kpoints
      ENDDO spin
!=======================================================================
      CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))

      CALLMPI( M_sum_i(WDES%COMM_INTER, ICOUEV ,1))
      CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV ,1))

      DO NP=1,NSIM
         DEALLOCATE(W0_1(NP)%CR)
      ENDDO
      DEALLOCATE(CF, CPROJ)

      PROFILING_STOP('lrf_commutator')

      RETURN
    END SUBROUTINE LRF_COMMUTATOR


!************************* SUBROUTINE HAMILTMU_COMMUTATOR *************
!
! this subroutine evaluates |xi> at a selected k-point
!    |xi>  = - hbar^2/ m_e nabla 
!                 + sum_ij | p_i> D(1)_ij <p_j | phi > 
!                 + sum_ij | p_i> D(0)_ij - e(0) Q(0)_ij < p_j r| phi >
!                 - sum_ij |r p_i>D(0)_ij - e(0) Q(0)_ij < p_j  | phi >
!                 - sum_ij | p_i> e(1) Q(0)_ij <p_j | phi >
!                [- e(1) |phi> (see NOTE below)]
!          = -i d/dk [H(k) - e(0) S(k)] | phi >
!
! the  wavefunctions must be supplied in reciprocal space C and real
! space CR
!
! D(1) change of strength parameter with k    (usually zero)
! D(0) is the original strength               (CDIJ0)
! e(0) is the zero order eigenvalue           (EVALUE0)
! e(1) is the first order change of the eigen energy
!
! e(1) is evaluated during the calculation of |xi>:
! e(1) = <phi| - hbar^2/ m_e nabla  | phi > + sum_ij <phi | p_i> D(1)_ij <p_j | phi >
!        +  sum_ij <phi | p_i > D(0)_ij - e(0) Q(0)_ij <p_j r| phi > -
!        -  sum_ij <phi |r p_i >D(0)_ij - e(0) Q(0)_ij <p_j | phi >
!
! IDIR supplies the cartesian index of (r_i)
! for LNABLA ==.TRUE., terms involving <p_i r| are ommited
!
! NOTE: the calling routine has to subtract e(1) |phi> to obtain the
! correct vector xi
!
!
!***********************************************************************

    SUBROUTINE HAMILTMU_COMMUTATOR( &
         WDES1,W0_1,WTMP,NONLR_S,NONLR_D,NONL_S,NONL_D, &
         GRID, LREAL, EVALUE0, LATT_CUR, &
         LMDIM,CDIJ,CDIJ0,CQIJ, &
         CH,LD, NSIM, LDO, VKPT, LNABLA, IDIR, CSHIFT)
      USE prec
      USE constant
      USE mpimy
      USE mgrid
      USE wave
      
      USE nonl_high
      USE lattice
      USE hamil

      USE openmp, ONLY : omp_nthreads_hamiltmu
      IMPLICIT NONE


      INTEGER NSIM,NP,LD
      INTEGER LMDIM, NGVECTOR, ISPINOR, ISPINOR_, MM, MM_ 
      TYPE (grid_3d)     GRID
      TYPE (nonlr_struct) NONLR_S, NONLR_D
      TYPE (nonl_struct)  NONL_S, NONL_D
      TYPE (wavefun1)    W0_1(NSIM)
      TYPE (wavefun1)    WTMP(NSIM)
      TYPE (wavedes1)    WDES1
      TYPE (latt)        LATT_CUR

      OVERLAP    CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
                 CQIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
                 CDIJ0(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
      COMPLEX(q) CH(LD,NSIM)
      REAL(q)    EVALUE0(NSIM)
      COMPLEX(q) EVALUE0_(NSIM)
      COMPLEX(q) EVALUE1(NSIM)
      LOGICAL LREAL
      LOGICAL LDO(NSIM)
      LOGICAL LNABLA
      INTEGER IDIR
      REAL(q) VKPT(3)
      REAL(q) CSHIFT
! local variables
      REAL(q) RINPLW; INTEGER M
      COMPLEX(q), ALLOCATABLE :: CWORK1(:,:)
      REAL(q)      ::    G1,G2,G3,GC(LD)
      COMPLEX(q)   ::    CE
      INTEGER      ::    LMBASE, LMBASE_, NIS, LMMAXC, NI, L, LP, NT
      REAL(q)      ::    WEIGHT


      PROFILING_START('hamiltmu_commutator')

      EVALUE1=0

      ALLOCATE(CWORK1(GRID%MPLWV*WDES1%NRSPINORS,NSIM)) 
      RINPLW=1._q/GRID%NPLWV
      NGVECTOR=WDES1%NGVECTOR
      EVALUE0_=EVALUE0+CMPLX(0.0_q,CSHIFT,q)
!=======================================================================
! calculate the local contribution (result in CWORK1)
!=======================================================================
      CE=0
      DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            CWORK1(:,NP) =0
         ENDIF
      ENDDO
!=======================================================================
! add term   [- hbar^2/ 2 m_e laplace , r] = - hbar^2/ m_e nabla 
!
! using  phi(r) = sum_G C_G  e^ i(G+k)r
!
! the term becomes  -i hbar^2/ m_e (G+k) C_G which is the change of the 
! kinetic energy operator with respect to k times -i
!   -i nabla_k ( hbar^2/ 2 m_e (G+k)^2)
!
!=======================================================================
      DO M=1,NGVECTOR
         G1=WDES1%IGX(M)+VKPT(1)
         G2=WDES1%IGY(M)+VKPT(2)
         G3=WDES1%IGZ(M)+VKPT(3)
         GC(M)=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))*(TPI*(2*HSQDTM))
      ENDDO
      DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            CE=0
            DO ISPINOR =0,WDES1%NRSPINORS-1
               DO M=1,NGVECTOR
                  MM=M+ISPINOR*NGVECTOR
                  CH(MM,NP)=CH(MM,NP)+(W0_1(NP)%CW(MM)*GC(M))*(0._q,-1._q)
                  CE=CE+CONJG(W0_1(NP)%CW(MM))*CH(MM,NP)
               ENDDO
            ENDDO
            CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))
            W0_1(NP)%CELEN=GREAL(CE)
         ENDIF
      ENDDO
!=======================================================================
! non-local contribution in real-space
!=======================================================================
      IF (LREAL) THEN
         IF (.NOT. LNABLA) THEN
            ! contribution - r_idir | p_i > D_ij - e(0) Q_ij < p_j |
            IF (CSHIFT==0) THEN
               CALL RACCMU(NONLR_D,WDES1,W0_1, LMDIM,CDIJ0,CQIJ,EVALUE0,CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
            ELSE
               CALL RACCMU_C(NONLR_D,WDES1,W0_1, LMDIM,CDIJ0,CQIJ,CONJG(EVALUE0_),CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
            ENDIF
            DO NP=1,NSIM
               IF ( LDO(NP) ) CWORK1(:,NP)=-CWORK1(:,NP)
            ENDDO

            ! contribution | p_i > D_ij - e(0) Q_ij <  p_j | r_idir
            CALL RPROMU(NONLR_D,WDES1,WTMP, NSIM, LDO)
            IF (CSHIFT==0) THEN
               CALL RACCMU(NONLR_S,WDES1,WTMP, LMDIM,CDIJ0,CQIJ,EVALUE0,CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
            ELSE
               CALL RACCMU_C(NONLR_S,WDES1,WTMP, LMDIM,CDIJ0,CQIJ,EVALUE0_,CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
            ENDIF
         ELSE
            DO NP=1,NSIM
               IF ( LDO(NP) ) THEN
                  CWORK1(:,NP)=0
                  WTMP(NP)%CPROJ=0
               ENDIF
            ENDDO
         ENDIF

         ! non local contributions to e(1)
         DO NP=1,NSIM
            IF ( LDO(NP) ) THEN
               CALL ECCP_NL_ALL(WDES1,W0_1(NP),WTMP(NP),CDIJ0,CQIJ,EVALUE0(NP),CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN-CE
               CALL ECCP_NL_ALL(WDES1,WTMP(NP),W0_1(NP),CDIJ0,CQIJ,EVALUE0(NP),CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
               CALL ECCP_NL_ALL(WDES1,W0_1(NP),W0_1(NP),CDIJ,CQIJ,0.0_q,CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
               ! should be purely imaginary
               W0_1(NP)%CELEN=CMPLX(0.0_q,AIMAG(W0_1(NP)%CELEN),q)
               EVALUE1(NP)=W0_1(NP)%CELEN
            ENDIF
         ENDDO

         ! contribution | p_i > d D_ij / d k - e(1) Q_ij < p_j |
         CALL RACCMU_C(NONLR_S,WDES1,W0_1, LMDIM,CDIJ,CQIJ,EVALUE1,CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
!=======================================================================
! calculate the non local contribution in reciprocal space
!=======================================================================
      ELSE
         DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            IF (.NOT. LNABLA) THEN
               ! contribution - | -i d p_i/ d k> D_ij - epsilon Q_ij < p_j |
               CH(:,NP)=-CH(:,NP)
               IF (CSHIFT==0) THEN
                  CALL VNLACC_ADD(NONL_D,W0_1(NP),CDIJ0,CQIJ,1,EVALUE0(NP),CH(:,NP))
               ELSE
                  CALL VNLACC_ADD_C(NONL_D,W0_1(NP),CDIJ0,CQIJ,1,CONJG(EVALUE0_(NP)),CH(:,NP))
               ENDIF
               CH(:,NP)=-CH(:,NP)
               ! contribution | p_i > D_ij - epsilon Q_ij < -i d p_j/ d k |
               CALL PROJ1(NONL_D,WDES1, WTMP(NP))
               IF (CSHIFT==0) THEN
                  CALL VNLACC_ADD(NONL_S,WTMP(NP),CDIJ0,CQIJ,1,EVALUE0(NP),CH(:,NP))
               ELSE
                  CALL VNLACC_ADD_C(NONL_S,WTMP(NP),CDIJ0,CQIJ,1,EVALUE0_(NP),CH(:,NP))
               ENDIF
               ! non local contributions to e(1)
               CALL ECCP_NL_ALL(WDES1,W0_1(NP),WTMP(NP),CDIJ0,CQIJ,EVALUE0(NP),CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN-CE
               CALL ECCP_NL_ALL(WDES1,WTMP(NP),W0_1(NP),CDIJ0,CQIJ,EVALUE0(NP),CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
            ELSE
               WTMP(NP)%CPROJ=0
            ENDIF

            CALL ECCP_NL_ALL(WDES1,W0_1(NP),W0_1(NP),CDIJ,CQIJ,0.0_q,CE)
            W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE

            ! should be purely imaginary
            W0_1(NP)%CELEN=CMPLX(0.0_q,AIMAG(W0_1(NP)%CELEN),q)
            EVALUE1(NP)=W0_1(NP)%CELEN

            ! contribution | p_i > d D_ij - e(1) Q_ij < p_j |
            CALL VNLACC_ADD_C(NONL_S,W0_1(NP),CDIJ,CQIJ,1,EVALUE1(NP),CH(:,NP))
         ENDIF
         ENDDO
      ENDIF
      DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            DO ISPINOR=0,WDES1%NRSPINORS-1
               CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
            ENDDO
         ENDIF
      ENDDO

      DEALLOCATE(CWORK1)

      PROFILING_STOP('hamiltmu_commutator')

      RETURN
    END SUBROUTINE HAMILTMU_COMMUTATOR



!************************ SUBROUTINE LRF_HAMIL **************************
!
! this subroutine evaluates 
!
!    |xi>  =  (H_0(1) -e(0) S(1)) |phi> + (H_sc(1) - e(1) S(0)) | phi>
!
! for a general perturbation, where H_0(1) -e(0) S(1) |phi> is supplied
! as input an the array in WXI
!
! H_sc(1), e(1) are the first order change of the
! Hamiltonian, eigenenergies and overlap, respectively
! More specifically for the PAW Hamiltonian this becomes:
!
!    |xi>  = V_sc(1) + sum_ij | p_i> D_sc(1)_ij <p_j | phi > 
!                    - sum_ij | p_i> e(1) Q(0)_ij <p_j | phi >
!                    - e(1) |phi> 
!                    + |xi(0)> + sum_i xi(0)_i | p_i>
!
! xi(0)_i is usually the change of the wave function character with respect
! to i k (see LRF_COMMUTATOR)
! roughly speeking this is essentially
!
!  xi(0)_i= -i d/dk <p_i |phi>= -i(<p_i | d/dk phi>+< d/dk p_i | phi>)
!                               - tau < p_i | phi_nk> 
! see comments at the bottom of LRF_RPHI
!
!
!***********************************************************************

  SUBROUTINE LRF_HAMIL(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W0, WXI, WDES, &
        LMDIM, CDIJ, CQIJ, SV, RMS, ICOUEV)
    USE prec
      
    USE wave
    USE wave_high
    USE base
    USE lattice
    USE mpimy
    USE mgrid
    
    USE nonl_high
    USE hamil
    USE constant
    USE wave_mpi

    USE openmp, ONLY : omp_nthreads_hi
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    TYPE (info_struct) INFO
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W0, WXI
    TYPE (wavedes)     WDES

    RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
    INTEGER LMDIM
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    REAL(q) RMS        ! magnitude of the residual vector
    INTEGER ICOUEV     ! number of H | phi> evaluations
! local work arrays and structures
    TYPE (wavedes1)  WDES1                     ! descriptor for one k-point
    TYPE (wavefun1), TARGET :: W0_1(WDES%NSIM) ! current wavefunction
    TYPE (wavefun1)  WXI0(WDES%NSIM)           ! see below

    COMPLEX(q),ALLOCATABLE:: CF(:,:)
    INTEGER :: NSIM                  ! number of bands treated simultaneously
    INTEGER :: NODE_ME, IONODE
    INTEGER :: NP, ISP, NK, NPL, NGVECTOR, NB_DONE, N, IDUMP, ISPINOR, LD, M, MM
    REAL(q) :: FNORM
    COMPLEX(q) :: ORTH
    INTEGER :: NB(WDES%NSIM)         ! contains a list of bands currently optimized
    REAL(q) :: EVALUE(WDES%NSIM)     ! eigenvalue during optimization
    LOGICAL :: LDO(WDES%NSIM)        ! band finished
    LOGICAL :: LSTOP,LNEWB
    COMPLEX(q) ::C

      PROFILING_START('lrf_hamil')

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif
!=======================================================================
!  INITIALISATION:
! maximum  number of bands treated simultaneously
!=======================================================================
      RMS   =0
      ICOUEV=0
      NSIM=WDES%NSIM

      ALLOCATE(CF(WDES%NRPLWV,NSIM))
      LD=WDES%NRPLWV

      CALL SETWDES(WDES, WDES1, 0)

      DO NP=1,NSIM
         CALL NEWWAV(WXI0(NP), WDES1, .FALSE.)
         ALLOCATE(W0_1(NP)%CR(GRID%MPLWV*WDES%NRSPINORS))
      ENDDO

!=======================================================================
      spin:    DO ISP=1,WDES%ISPIN
      kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES1,NK)

      NPL=WDES1%NPL
      NGVECTOR=WDES1%NGVECTOR

      IF (INFO%LREAL) THEN
        CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
      ELSE
        CALL PHASE(WDES,NONL_S,NK)
      ENDIF
!=======================================================================
      NB_DONE=0                 ! index of the bands allready optimised
      bands: DO
        NB=0                      ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
        IDUMP=0
        newband: DO NP=1,NSIM
        LNEWB=.FALSE.
        IF (NB_DONE<WXI%WDES%NBANDS ) THEN
           NB_DONE=NB_DONE+1; LNEWB=.TRUE.
        ENDIF
        IF (LNEWB) THEN
           N     =NB_DONE
           NB(NP)=NB_DONE
           ICOUEV=ICOUEV+1

           CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

           WXI0(NP)%CW   =WXI%CW(:,N,NK,ISP)
           WXI0(NP)%CPROJ=WXI%CPROJ(:,N,NK,ISP)
           
#ifdef debug
           IDUMP=2
#endif
#ifdef MPI
           IF (NODE_ME /= IONODE) IDUMP=0
#endif
           ! fft to real space
           DO ISPINOR=0,WDES%NRSPINORS-1
              CALL FFTWAV(NGVECTOR, WDES%NINDPW(1,NK),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
           ENDDO

           EVALUE(NP)=W0_1(NP)%CELEN
        ENDIF
        ENDDO newband
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
        LSTOP=.TRUE.
        LDO  =.FALSE.
        DO NP=1,NSIM
           IF ( NB(NP) /= 0 ) THEN
              LSTOP  =.FALSE.
              LDO(NP)=.TRUE.     ! band not finished yet
           ENDIF
        ENDDO
        IF (LSTOP) EXIT bands

!=======================================================================
! determine gradient and store it
!=======================================================================
        !  store H | psi > temporarily
        !  to have uniform stride for result array
        CALL HAMILTMU_LRF(WDES1,W0_1,WXI0,NONLR_S,NONL_S, &
                & GRID,  INFO%LREAL, EVALUE, LATT_CUR, &
                & LMDIM,CDIJ(1,1,1,ISP),CQIJ(1,1,1,ISP), &
                & SV(1,ISP), CF(1,1),LD, NSIM, LDO)

        i2: DO NP=1,NSIM
           N=NB(NP); IF (.NOT. LDO(NP)) CYCLE i2

           ORTH =0
           FNORM=0
           DO ISPINOR=0,WDES%NRSPINORS-1
           DO M=1,NGVECTOR
              MM=M+ISPINOR*NGVECTOR
              CF(MM,NP)=CF(MM,NP)-W0_1(NP)%CELEN*W0_1(NP)%CW(MM)
              IF (WDES%LSPIRAL.AND.WDES%DATAKE(M,ISPINOR+1,NK)>INFO%ENINI) CF(MM,NP)=0
              FNORM =FNORM+CF(MM,NP)*CONJG(CF(MM,NP))
              ORTH  =ORTH +GREAL(CF(MM,NP)*CONJG(W0_1(NP)%CW(MM)))
              WXI%CW(MM,N,NK,ISP)=CF(MM,NP)
           ENDDO
! the projector functions do not change when an external field is applied
           WXI%CPROJ(:,N,NK,ISP)=0
           WXI%CELEN(N,NK,ISP)=W0_1(NP)%CELEN
           ENDDO
           CALLMPI( M_sum_s(WDES%COMM_INB, 1, FNORM, 0._q, 0._q, 0._q))
           CALLMPI( M_sum_z(WDES%COMM_INB, ORTH, 1))

           ! at some point I need to double check this why are the
           ! gradients sometimes not orthogonal in OEP methods 
           IF (ABS(ORTH)>1E-1) THEN
              WRITE(0,*)'LRF_HAMIL internal error: the vector H(1)-e(1) S(1) |phi(0)> is not orthogonal to |phi(0)>',ORTH,W0_1(NP)%CELEN
              STOP
           ENDIF

           IF (IDUMP==2) WRITE(*,'(I3,E11.4,"R ",2E11.4,"O ",2E14.7)') N,SQRT(ABS(FNORM)),ORTH,WXI%CELEN(N,NK,ISP)
           RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W0%FERWE(N,NK,ISP)*SQRT(ABS(FNORM))/WDES%NB_TOT

!=======================================================================
! move onto the next block of bands
!=======================================================================
        ENDDO i2
!=======================================================================
      ENDDO bands
      ENDDO kpoints
      ENDDO spin
!=======================================================================
      CALLMPI( M_sum_d(WDES%COMM_INTER, RMS, 1))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))

      CALLMPI( M_sum_i(WDES%COMM_INTER, ICOUEV ,1))
      CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV ,1))

      DO NP=1,NSIM
         CALL DELWAV(WXI0(NP) ,.FALSE.)
         DEALLOCATE(W0_1(NP)%CR)
      ENDDO
      DEALLOCATE(CF)

      PROFILING_STOP('lrf_hamil')

      RETURN
    END SUBROUTINE LRF_HAMIL


!************************* SUBROUTINE HAMILTMU_LRF ********************
!
! this subroutine calculates the first order change of H
! acting onto a set of wavefuntions
! the  wavefunction must be given in reciprocal space C and real
! space CR
! CH contains the result
!    |xi>  = V_sc(1) + sum_ij | p_i> D_sc(1)_ij <p_j | phi > 
!                    - sum_ij | p_i> e(1) Q(0)_ij <p_j | phi >
!                    - e(1) |phi> 
!                    + |xi(0)> + sum_i xi(0)_i | p_i>
!
! V(1) is the first order change of the local potential  (SV)
! D(1) is the first order change of the PAW strength     (CDIJ)
! e(1) is the first order change of the eigen energy
!
! e(1) is evaluated during the calculation of |xi>:
! e(1) = <phi| xi(0) > +  sum_i <phi |p_i > xi(0)_i +
!                  <phi|V_sc(1)|phi> +  <phi | p_i> D_sc(1)_ij <p_j | phi >
! 
! NOTE: the calling routine has to subtract e(1) |phi> to obtain the
! correct vector xi
!
!
!***********************************************************************

    SUBROUTINE HAMILTMU_LRF( &
         WDES1,W0_1,WXI0,NONLR_S,NONL_S, &
         GRID, LREAL, EVALUE0, LATT_CUR, &
         LMDIM,CDIJ,CQIJ, &
         SV,CH,LD, NSIM, LDO)
      USE prec

      USE mpimy
      USE mgrid
      USE wave
      
      USE nonl_high
      USE lattice
      USE hamil

      USE openmp, ONLY : omp_nthreads_hamiltmu
      IMPLICIT NONE


      INTEGER NSIM,NP,LD
      INTEGER LMDIM, NGVECTOR, ISPINOR, ISPINOR_, MM, MM_ 
      TYPE (grid_3d)     GRID
      TYPE (nonlr_struct) NONLR_S,NONLR_ION, NONLR_IOND
      TYPE (nonl_struct) NONL_S
      TYPE (wavefun1)    W0_1(NSIM)
      TYPE (wavefun1)    WXI0(NSIM)
      TYPE (wavedes1)    WDES1
      TYPE (latt)        LATT_CUR

      RGRID      SV(DIMREAL(GRID%MPLWV),WDES1%NRSPINORS*WDES1%NRSPINORS) ! local potential
      OVERLAP    CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
                 CQIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
      COMPLEX(q) CH(LD,NSIM)
      REAL(q)    EVALUE0(NSIM)
      COMPLEX(q) EVALUE1(NSIM)
      LOGICAL LREAL
      LOGICAL LDO(NSIM)
! local variables
      REAL(q) RINPLW; INTEGER M
      COMPLEX(q),ALLOCATABLE :: CWORK1(:,:)
      COMPLEX(q)   ::    CE
      REAL(q)      ::    WEIGHT


      PROFILING_START('hamiltmu_lrf')

      EVALUE1=0

      ALLOCATE(CWORK1(GRID%MPLWV*WDES1%NRSPINORS,NSIM)) 
      RINPLW=1._q/GRID%NPLWV
      NGVECTOR=WDES1%NGVECTOR
!=======================================================================
! calculate the local contribution (result in CWORK1)
!=======================================================================
      DO NP=1,NSIM
        IF ( LDO(NP) ) THEN
        CE=0
        CWORK1(:,NP)  = 0
        DO ISPINOR =0,WDES1%NRSPINORS-1
        DO ISPINOR_=0,WDES1%NRSPINORS-1
           DO M=1,GRID%RL%NP
              MM =M+ISPINOR *GRID%MPLWV
              MM_=M+ISPINOR_*GRID%MPLWV
              CWORK1(MM,NP)=  CWORK1(MM,NP)+(SV(M,1+ISPINOR_+2*ISPINOR) *W0_1(NP)%CR(MM_)*RINPLW)
              CE=CE + CONJG(W0_1(NP)%CR(MM)) *(SV(M,1+ISPINOR_+2*ISPINOR) *W0_1(NP)%CR(MM_)*RINPLW)
           ENDDO
        ENDDO
        ENDDO
        CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))
        W0_1(NP)%CELEN=CE
        ENDIF
      ENDDO
!=======================================================================
! add term xi_0
!=======================================================================
      DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            CE=0
            DO ISPINOR =0,WDES1%NRSPINORS-1
               DO M=1,NGVECTOR
                  MM=M+ISPINOR*NGVECTOR
                  CH(MM,NP)=WXI0(NP)%CW(MM)
                  CE=CE+CONJG(W0_1(NP)%CW(MM))*CH(MM,NP)
               ENDDO
            ENDDO
            DO M=1,WDES1%NPRO
               CE=CE+GCONJG(W0_1(NP)%CPROJ(M))*WXI0(NP)%CPROJ(M)
            ENDDO
            CALLMPI( M_sum_z(WDES1%COMM_INB, CE, 1))
            W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
         ENDIF
      ENDDO
!=======================================================================
! non-local contribution in real-space
!=======================================================================
      IF (LREAL) THEN
         ! non local contributions to e(1)
         DO NP=1,NSIM
            IF ( LDO(NP) ) THEN
               CALL ECCP_NL_ALL(WDES1,W0_1(NP),W0_1(NP),CDIJ,CQIJ,0.0_q,CE)
               W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
               EVALUE1(NP)=W0_1(NP)%CELEN
            ENDIF
         ENDDO
         ! contribution | p_i > D_sc(1)_ij - e(1) Q_ij < p_j |
         CALL RACCMU_C(NONLR_S,WDES1,W0_1, LMDIM,CDIJ,CQIJ,EVALUE1,CWORK1,GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
         DO NP=1,NSIM
            IF ( LDO(NP) ) THEN
            ! contribution | p_i > cxi_i
            CALL RACC0(NONLR_S,WDES1,WXI0(NP)%CPROJ(1),CWORK1(1,NP))
            DO ISPINOR=0,WDES1%NRSPINORS-1
               CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
            ENDDO
            ENDIF
         ENDDO
!=======================================================================
! calculate the non local contribution in reciprocal space
!=======================================================================
      ELSE
         DO NP=1,NSIM
         IF ( LDO(NP) ) THEN
            ! non local contributions to e(1)
            CALL ECCP_NL_ALL(WDES1,W0_1(NP),W0_1(NP),CDIJ,CQIJ,0.0_q,CE)
            W0_1(NP)%CELEN=W0_1(NP)%CELEN+CE
            EVALUE1(NP)=W0_1(NP)%CELEN
            ! contribution | p_i > cxi_i
            CALL VNLAC0(NONL_S,WDES1,WXI0(NP)%CPROJ(1),CH(1,NP))
            ! contribution | p_i > D_(sc)_ij - e(1) Q_ij < p_j |
            CALL VNLACC_ADD_C(NONL_S,W0_1(NP),CDIJ,CQIJ,1,EVALUE1(NP),CH(:,NP))

            DO ISPINOR=0,WDES1%NRSPINORS-1
               CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
            ENDDO
        ENDIF
        ENDDO
      ENDIF
      DEALLOCATE(CWORK1)

      PROFILING_STOP('hamiltmu_lrf')

      RETURN
    END SUBROUTINE HAMILTMU_LRF

  END MODULE hamil_lrf
