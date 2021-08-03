!
! ) test CHDEN array versus CHTOT in main.F in particular for 
!   magnetization in y direction
!   o correct
! ) UNCLEAR:
! ) factor 2 in core contribution

#include "symbol.inc"

!***********************************************************************
!
! this module implements the base local Hartree Fock and KLI routines
! on  the plane wave grid
!
! for details on the KLI method see
!   Krieger, Li, and Iafrate, Phys. Rev. A46, 5453 (1992)
! the local Hartree Fock method was introduced in
!   Sala and Goerling, J. Chem. Phys. 115, 5718 (2001)
!
! Both methods require the calculation of the Slater energy density
! \sum_a f_a psi_a(r) V_x(r,r') psi_a(r')
! (where V_x is the exact exchange potential) and a correction term.
! The correction term differs in the KLI and LHF method
! for insulators the KLI method yields a rotationally invariant
! correction term (i.e. unitary transformation between the orbitals
! will not change the result), whereas in the KLI method the correction
! term depends on the "ordering" of the orbitals.
! However for metals the LHF method does not apply (as densities, or
! total energies are not invariant under unitary transformations of
! the occupied orbitals).
! In any case, the correction terms have very little influence on the
! results, and hence it probably does not matter very much which method
! is applied
!
!
! gK 06.2004
!
!***********************************************************************

#define LMAXFOCKAE

MODULE pwkli
  USE prec
  IMPLICIT none
  REAL(q), PARAMETER :: MIN_CHARGE=0.5E-4

CONTAINS

!************************ SUBROUTINE PW_SLATER_POTENTIAL ***************
!
! This subroutine calculates the Slater energy density SV and the
! charge density CHDEN on the plane wave grid described by the
! grid structure GRID
!
! it is essentially a copy of the FOCK_ACC routine plus the required
! modifications to average the energy density over all filled states
! most modifications  are bracketed by comment !KLI !KLIend
!
!***********************************************************************

  SUBROUTINE PW_SLATER_POTENTIAL(GRID, LMDIM, LATT_CUR, W, &
       NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIPN, &
       CH, SV, CHDEN, P, CDCHF)
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE pseudo
    USE fock
    USE hamil
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (grid_3d) GRID
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(:)
    INTEGER NK,ISP,NPOS,NSTRIPN
    COMPLEX(q)       :: CH(W%WDES%NRPLWV,NSTRIPN) ! accelerations in rec. space
    COMPLEX(q) CDCHF                            ! double counting correction
    RGRID   SV(:, :)
    RGRID   CHDEN(:,:)

! local variables
    TYPE (wavedes1), TARGET :: WDESK, WDESQ, WDESQ_IRZ
    TYPE (wavefun1) :: W1, WQ
    TYPE (wavefun1),ALLOCATABLE :: WIN(:)
    REAL(q) :: WEIGHT
    REAL(q) :: FSG                            ! singularity correction
    INTEGER ISPINOR, ISPINOR_
    INTEGER N, N_, MQ, NP, NGLB, MM, ISP_IRZ
    INTEGER NQ
    INTEGER NODE_ME, NODE_ME_I, IONODE, NCPU
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
    GDEF    :: GWORK(m_ GRID_FOCK%MPLWV,W%WDES%NRSPINORS) ! fock pot in real sp
    GDEF    :: GCHG(m_ GRID_FOCK%MPLWV,W%WDES%NRSPINORS*W%WDES%NRSPINORS)
    COMPLEX(q),ALLOCATABLE :: CXI(:,:)         ! acc. in real space
    GDEF,      ALLOCATABLE :: CKAPPA(:,:)      ! stores NL accelerations
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:)    ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! 1/(G+dk)**2 (G)
    REAL(q) :: CHARGE
    INTEGER :: ierror
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    TYPE (wavespin) WHF
    COMPLEX(q) :: CWORK(W%WDES%GRID%MPLWV*W%WDES%NRSPINORS)
#ifdef dotiming
      usetmg
      REAL(q) :: TVini,TCini,TVfft,TCfft,TVchg,TCchg,TVdep,TCdep,TVaug,TCaug, &
                 TVpot,TCpot,TVacc,TCacc,TVrpo,TCrpo,TVovl,TCovl,TVend,TCend,TVwait,TCwait,TVsum,TCsum
      TVini=0; TCini=0; TVfft=0; TCfft=0; TVchg=0; TCchg=0; TVdep=0; TCdep=0; TVaug=0; TCaug=0;
      TVpot=0; TCpot=0; TVacc=0; TCacc=0; TVrpo=0; TCrpo=0; TVovl=0; TCovl=0; TVend=0; TCend=0
      TVwait=0;TCwait=0; TVsum=0; TCsum=0
#endif

! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK

#ifdef MPI
    NODE_ME=WHF%WDES%COMM%NODE_ME
    NODE_ME_I=WHF%WDES%COMM_INTER%NODE_ME
    IONODE=WHF%WDES%COMM%IONODE
    NCPU=WHF%WDES%COMM_INTER%NCPU              ! number of band groups
    IF (WHF%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('PW_SLATER_POTENTIAL: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#else
    NODE_ME=0
    NODE_ME_I=1
    IONODE=0
    NCPU=1
#endif

    CALL CHECK_FULL_KPOINTS

! determine whether redistribution is required
    IF (NCPU /= 1) THEN                    ! more than one band-group
       DO_REDIS=.TRUE.
    ELSE 
       DO_REDIS=.FALSE.
    ENDIF

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIPN*NCPU


    ALLOCATE( &
         CXI(GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,NGLB), &
         CKAPPA(WHF%WDES%NPROD,NGLB),                   &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS), &
         CDLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         POTFAK(GRIDHF%MPLWV))

    CALL SETWDES(WHF%WDES,WDESQ,0)
    CALL NEWWAV(WQ , WDESQ,.TRUE.)

    CALL SETWDES(WHF%WDES,WDESK,NK)
    ALLOCATE(WIN(NGLB))
    DO N=1,NGLB
       CALL NEWWAV(WIN(N) , WDESK,.TRUE.)
    ENDDO

! average electrostatic potential for k=k' and n=n'
    FSG=FSG_STORE(NK)
!==========================================================================
! initialise variables
!==========================================================================
    CDIJ=0; CDLM=0; CXI=0; CKAPPA=0; CH=0

    CALLMPI(MPI_barrier(WHF%WDES%COMM%MPI_COMM,ierror))
    statmg
!==========================================================================
! fourier transform the bands for which the HF exchange need to be calculated
! to real space, then distribute the WIN to all nodes
!==========================================================================
    DO N=NPOS,NPOS+NSTRIPN-1
       CALL W1_COPY( ELEMENT( WHF, WDESK, N, ISP), WIN((N-NPOS)*NCPU+NODE_ME_I) )
       CALL FFTWAV_W1( WIN((N-NPOS)*NCPU+NODE_ME_I))
    ENDDO

    ! distribute WIN to all nodes
#ifdef MPI
    IF (DO_REDIS) THEN
       DO N=1,NGLB
          CALL M_bcast_z_from(WDESK%COMM_INTER,WIN(N)%CR(1), &
               GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS,MOD(N-1,NCPU)+1)
#ifndef gammareal
          IF (WHF%WDES%LOVERL) CALL M_bcast_z_from(WDESK%COMM_INTER,WIN(N)%CPROJ(1), &
               WHF%WDES%NPROD,MOD(N-1,NCPU)+1)
#else
          IF (WHF%WDES%LOVERL) CALL M_bcast_d_from(WDESK%COMM_INTER,WIN(N)%CPROJ(1), &
               WHF%WDES%NPROD,MOD(N-1,NCPU)+1)
#endif
       ENDDO
    ENDIF
#endif
    addtmg(TVini,TCini)
!==========================================================================
!  loop over all q-points (index NQ)
!  sum_nq phi_nq mq (r') \int phi_nq mq(r) phi_nk mk(r) / (r-r') d3r 
!==========================================================================
    qpoints: DO NQ=1,KPOINTS_FULL%NKPTS
       IF( SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,NQ)-WHF%WDES%VKPT(:,NK))) CYCLE

       CALL SETWDES(WHF%WDES,WDESQ,NQ)
       CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))
!new       
       ISP_IRZ=ISP
       IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
          ISP_IRZ=3-ISP
       ENDIF
!newend

   ! set POTFAK for this q and k point
       CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)

   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,WHF%WDES%NBANDS
          IF (ABS(WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE mband

          IF (NQ<=WHF%WDES%NKPTS) THEN
             CALL W1_COPY(ELEMENT(WHF, WDESQ, MQ, ISP), WQ)
             CALL FFTWAV_W1(WQ)
          ELSE
#ifndef gammareal
           ! 
           ! symmetry must be considered if the wavefunctions for this
           ! k-point NQ (containing all k-points in the entire BZ) 
           ! are not stored in W
           !
             LSHIFT=.FALSE.
             IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                 (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.

             CALL W1_ROTATE_AND_FFT(WQ, ELEMENT(WHF, WDESQ_IRZ, MQ, ISP_IRZ), &
                  ROT_HANDLE, P, LATT_CUR, LSHIFT)
#endif
          ENDIF
          addtmga(TVfft,TCfft)
!-----------------------------------------------------------------------------
! calculate fock potential and add to accelerations
!-----------------------------------------------------------------------------
         ! calculate charge phi_q nq(r) phi_k nk(r)
          nband: DO N=1,NGLB
!KLImod
             CALL PW_CHARGE_CMPLX(WDESK, GCHG(1,1), SIZE(GCHG,1), WIN(N)%CR(1), WQ%CR(1))
!KLIend
             addtmga(TVchg,TCchg)

             ! add augmentation part to charge (if required)
             IF (WHF%WDES%LOVERL) THEN
#ifdef LMAXFOCKAE
                CALL DEPSUM_TWO_BANDS_RHOLM_FULL(WIN(N)%CPROJ(:),WQ%CPROJ(:), WDESK, AUG_DES, &
                      TRANS_MATRIX_FOCK, CRHOLM,WHF%WDES%LOVERL )
#else
                CALL DEPSUM_TWO_BANDS_RHOLM_FULL_NOAE(WIN(N)%CPROJ(:),WQ%CPROJ(:), WDESK, AUG_DES, &
                      TRANS_MATRIX_FOCK, CRHOLM,WHF%WDES%LOVERL, FAST_AUG_FOCK%LMAX)
#endif
                addtmga(TVdep,TCdep)
                
                AUG_DES%RINPL=1._q ! multiplicator used by RACC0
!KLImod
                DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
                   CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES,CRHOLM(1+ISPINOR*AUG_DES%NPRO), GCHG(1,1+ISPINOR))
                ENDDO
!KLIend
                addtmga(TVaug,TCaug)
             ENDIF
!KLIadd
             DO NP=1,GRIDHF%RL%NP
                GWORK(NP,1)=GCHG(NP,1)
             ENDDO
             IF (W%WDES%NRSPINORS==2) THEN
                DO NP=1,GRIDHF%RL%NP
                   MM=NP+GRIDHF%MPLWV
                   GWORK(NP,1)=GWORK(NP,1)+GCHG(NP,4)
                ENDDO
             ENDIF
!KLIend
      ! fft to reciprocal space
             CALL FFT3D(GWORK(1,1),GRIDHF,-1)
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL APPLY_GFAC(GRIDHF, GWORK(1,1), POTFAK(1))
      ! back to real space to get  \int phi_q(r) phi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1,1),GRIDHF,1)

      ! singularity correction for isolated molecules could be added here 
      ! (monopole-quadrupole correction)
      ! .... to be filled out by GK ....

             addtmga(TVpot,TCpot)
!KLIadd
! now multiply by the complex conjugated charge to obtain the Slater energy density
             WEIGHT=W%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)* &
                 W%FERTOT((NPOS-1)*NCPU+N,NK,ISP)*W%WDES%WTKPT(NK)*W%WDES%RSPIN
             DO ISPINOR =0,W%WDES%NRSPINORS-1
             DO ISPINOR_=0,W%WDES%NRSPINORS-1
                DO NP=1,GRIDHF%RL%NP
                   SV(NP,ISP+ISPINOR_+2*ISPINOR)=SV(NP,ISP+ISPINOR_+2*ISPINOR)- &
                        GCONJG(GCHG(NP,1+ISPINOR_+2*ISPINOR))*GWORK(NP,1)*WEIGHT*AEXX
                ENDDO
             ENDDO
             ENDDO
!KLIend

      ! add to acceleration xi in real space
             WEIGHT=WHF%FERWE(MQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)/GRIDHF%NPLWV
             CALL VHAMIL_TRACE(WDESK, GRID_FOCK, GWORK(1,1), WQ%CR(1), CXI(1,N), WEIGHT)
             addtmga(TVacc,TCacc)
                
             IF (WHF%WDES%LOVERL) THEN
! add to acceleration kappa
   ! calculate D_LM
                ! build the descriptor for RPRO1
                W1%CPROJ => CDLM(:)
                AUG_DES%RINPL=WEIGHT ! multiplicator for RPRO1
                CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, W1, GWORK(:,1))
                IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
                addtmga(TVrpo,TCrpo)
   ! transform D_LM -> D_lml'm'
#ifdef LMAXFOCKAE
                CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ,CDLM )
#else
                CALL CALC_DLLMM_TRANS_NOAE(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ,CDLM, FAST_AUG_FOCK%LMAX)
#endif
   ! add D_lml'm' to kappa_lm_N (sum over l'm')
                CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1), WQ%CPROJ(1), CKAPPA(1,N),.TRUE.)
                addtmga(TVovl,TCovl)
             ENDIF
          ENDDO nband
       ENDDO mband
! fourier transform local accelerations xi (only own bands)
    ENDDO qpoints
!-----------------------------------------------------------------------------
! end of main fock loop                                         
!-----------------------------------------------------------------------------
    addtmg(TVwait,TCwait)
    CALLMPI(MPI_barrier(WHF%WDES%COMM%MPI_COMM,ierror))
! collect CXI and CKAPPA
#ifdef MPI
    IF (DO_REDIS) THEN
       CALL M_sum_z(WDESK%COMM_INTER,CXI(1,1),GRID_FOCK%MPLWV*WHF%WDES%NRSPINORS*NGLB)
       CALL M_sum_g(WDESK%COMM_INTER,CKAPPA(1,1),WHF%WDES%NPROD*NGLB)
    END IF
#endif
    addtmg(TVsum,TCsum)

!KLI
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, (/0.0_q, 0.0_q, 0.0_q/))
!KLIend

! generate the descriptor for the original WDES
    CALL SETWDES(W%WDES,WDESQ,NK)

! fourier transform local accelerations xi (only own bands)
    fft_back:DO N=NODE_ME_I,NGLB,NCPU
       N_=(N-1)/NCPU+1
! add CKAPPA to CXI (full acceleration on band N now in CXI)
       IF (WHF%WDES%LOVERL) THEN

          IF (NONLR_S%LREAL) THEN
             CWORK=0
             CALL RACC0(NONLR_S, WDESQ, CKAPPA(1,N), CWORK(1))
             DO ISPINOR=0,WDESQ%NRSPINORS-1
                CALL FFTEXT(WDESQ%NGVECTOR,WDESQ%NINDPW(1), &
                     CWORK(1+ISPINOR*WDESQ%GRID%MPLWV), &
                     CH(1+ISPINOR*WDESQ%NGVECTOR,N_),WDESQ%GRID,.TRUE.)
             ENDDO
          ELSE
             CALL VNLAC0(NONL_S, WDESK, CKAPPA(1,N), CH(1,N_))
          ENDIF
       ENDIF

       ! double counting hence subtract half the self energy
       WEIGHT=WHF%FERWE(N_+NPOS-1,NK,ISP)*WHF%WDES%WTKPT(NK)*0.5_q*WHF%WDES%RSPIN

       DO ISPINOR=0,WDESK%NRSPINORS-1
          CALL FFTEXT(WDESK%NGVECTOR,WDESK%NINDPW(1), &
               CXI(1+ISPINOR*GRID_FOCK%MPLWV,N), &
               CH(1+ISPINOR*WDESK%NGVECTOR,N_),GRID_FOCK,.TRUE.)

          DO NP=1,WDESK%NGVECTOR
             MM=NP+ISPINOR*WDESK%NGVECTOR
             CH(MM,N_)=-CH(MM,N_)*AEXX
             CDCHF=CDCHF-CONJG(CH(MM,N_))*WHF%CW(MM,N_+NPOS-1,NK,ISP)*WEIGHT
            ENDDO
       ENDDO
!KLI
!
!  calculate total charge density on coarse grid (CHDEN) this includes
!  the normal augmentation contributions that are also use in us.F (FOCK_AE is not included)
!
       IF (W%FERTOT((NPOS-1)*NCPU+N,NK,ISP)/=W%FERWE(N_+NPOS-1,NK,ISP)) THEN
          WRITE(0,*) 'internal error in  PW_SLATER_POTENTIAL: index into FERTOT incorrect'
          STOP
       ENDIF

       CALL PW_CHARGE_CMPLX(WDESK, GCHG(1,1), SIZE(GCHG,1), WIN(N)%CR(1), WIN(N)%CR(1))
      ! add augmentation part to charge (if required)
       IF (W%WDES%LOVERL) THEN
#ifdef LMAXFOCKAE
          CALL DEPSUM_TWO_BANDS_RHOLM_FULL(WIN(N)%CPROJ(:),WIN(N)%CPROJ(:), WDESK, AUG_DES, &
               TRANS_MATRIX_FOCK, CRHOLM, W%WDES%LOVERL )
#else
          CALL DEPSUM_TWO_BANDS_RHOLM_FULL_NOAE(WIN(N)%CPROJ(:),WIN(N)%CPROJ(:), WDESK, AUG_DES, &
               TRANS_MATRIX_FOCK, CRHOLM, W%WDES%LOVERL, FAST_AUG_FOCK%LMAX)
#endif
          
          AUG_DES%RINPL=1._q ! multiplicator used by RACC0
          DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
             CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, &
                  CRHOLM(1+ISPINOR*AUG_DES%NPRO), GCHG(1,1+ISPINOR))
          ENDDO
       ENDIF
       WEIGHT=WHF%FERWE(N_+NPOS-1,NK,ISP)*WHF%WDES%WTKPT(NK)*WHF%WDES%RSPIN

       DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
          DO NP=1,GRIDHF%RL%NP
             CHDEN(NP,ISP+ISPINOR)=CHDEN(NP,ISP+ISPINOR)+GCHG(NP,1+ISPINOR)*WEIGHT
          ENDDO
       ENDDO
!KLIend

    ENDDO fft_back

    addtmg(TVend,TCend)
#ifdef dotiming
    IF (NODE_ME==IONODE) WRITE(*,*)
    wrttmg("initialisation  (FFT of orbitals)        TVini ",TVini,TCini)
    wrttmg("FFT of occupied orbitals                 TVfft ",TVfft,TCfft)
    wrttmg("charge construction                      TVchg ",TVchg,TCchg)
    wrttmg("calculate onsite occupancies             TVdep ",TVdep,TCdep)
    wrttmg("augmentation with charge                 TVaug ",TVaug,TCaug)
    wrttmg("potential construction                   TVpot ",TVpot,TCpot)
    wrttmg("add acceleration                         TVacc ",TVacc,TCacc)
    wrttmg("project potential Q(r) V(r)              TVrpo ",TVrpo,TCrpo)
    wrttmg("construct non local accelerations        TVovl ",TVovl,TCovl)
    wrttmg("wait                                     TVwait",TVwait,TCwait)
    wrttmg("summation                                TVsum ",TVsum,TCsum)
    wrttmg("finalisation                             TVend ",TVend,TCend)
#endif

    DEALLOCATE(CXI,CKAPPA,CRHOLM,CDIJ,CDLM,POTFAK)
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(WQ,.TRUE.)
    DO N=1,NGLB
       CALL DELWAV(WIN(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(WIN)

#undef GRID_FOCK
#undef debug
  END SUBROUTINE PW_SLATER_POTENTIAL

  
!************************ SUBROUTINE PW_LHF_CORR_POTENTIAL ************
!
! this subroutine calculates the "correction" energy density on the 
! plane wave grid
!
! e(r) = V_lhf(r) rho(r) 
!  =\sum_nm psi_k,n*(r) psi_k,m*(r) <psi_k,n| V^old_lhf - V_x | psi_k,m>
!           * sqrt(f_k,n) * sqrt(f_k,m)
!
! V^old_lhf is the local echange potential in the previous step
! V_x       is the exact (non local) exchange potential V_x(r,r')
! f_k,n     is the occupancy of the single electron state
! 
!  <psi_k,n| V_x | psi_k,m> is supplied by the calling routine in CCORR
!           CCORR(m,n) = - < psi_m | V_x | psi_n >
!
! in principle, the correction term is only correct, if, f_k,n is 1 or 0
! otherwise Goerlings correction term is not valid (i.e. metals)
! this follows from the fact that the density matrix
!  rho = \sum_a f_a |psi_a> <psi_a|
! is not idempotent for metals
!  rho * rho = \sum_a f_a |psi_a> <psi_a| \sum_b f_b |psi_b> <psi_b|
!            = \sum_a f_a^2 |psi_a> <psi_a|
! as a backup the sqrt(f_k,n) is used, which is valid in isulators, and
! probably a good "approximation" for metals
! in principle however, for metals only the KLI correction should be used
!
!***********************************************************************

  SUBROUTINE PW_LHF_CORR_POTENTIAL(GRID,LMDIM, LATT_CUR, W, &
       NONLR_S, NONL_S, NK, ISP, NSTRIP, &
       SV_OLD, SV, CCORR , CCORR_LOCAL, CRHODE_CORR)

    USE prec
    USE sym_prec
    
    USE nonl_high
    USE mpimy
    USE mgrid
    USE wave
    USE wave_mpi
    USE lattice
    USE constant
    USE mdipol
    USE full_kpoints
    USE fock
    IMPLICIT NONE

! passed variables
    TYPE (grid_3d) GRID
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    INTEGER NK, ISP
    RGRID   SV(:,:)
    RGRID   SV_OLD(:, :)
    GDEF    CCORR(:,:), CCORR_LOCAL(:,:)
    OVERLAP CRHODE_CORR(:,:,:,:)
! local variables

! charge and potential
    GDEF GWORK(m_ GRID%MPLWV,W%WDES%NRSPINORS*W%WDES%NRSPINORS) ! fock pot in real sp
    TYPE (wavedes1) WDESK
    TYPE (wavefun1) W1, WQ
    REAL(q) :: WEIGHT
    INTEGER ISPINOR, ISPINOR_, NPL
    INTEGER N, N_, MQ, MM, MM_, NP, NB_LOC, NB_TOT
    INTEGER NODE_ME, NODE_ME_I, IONODE, NCPU
    INTEGER NGLB
    LOGICAL DO_REDIS, LSHIFT
    CHARACTER(1) CHARAC
    COMPLEX(q),ALLOCATABLE,TARGET :: CWRN(:,:) ! wave functions to be acc.
    COMPLEX(q),TARGET,ALLOCATABLE :: CWRM(:)   ! M-wavefunctions
    GDEF,      ALLOCATABLE :: CPROJK(:,:)      ! stores wave character of curr. strip
    GDEF,TARGET,ALLOCATABLE:: CPROJM(:)        ! stores the wave character of ; wavefunctions
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF ::    CLOCAL, CSUM
    INTEGER :: NN, NPOS, NSTRIPN, NSTRIP

#ifdef MPI
    NODE_ME=W%WDES%COMM%NODE_ME
    NODE_ME_I=W%WDES%COMM_INTER%NODE_ME
    IONODE=W%WDES%COMM%IONODE
    NCPU=W%WDES%COMM_INTER%NCPU              ! number of band groups
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('PW_LHF_CORR_POTENTIAL: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#else
    NODE_ME=0
    NODE_ME_I=1
    IONODE=0
    NCPU=1
#endif

    CALL CHECK_FULL_KPOINTS
! determine whether redistribution is required
    IF (NCPU /= 1) THEN                    ! more than one band-group
       DO_REDIS=.TRUE.
    ELSE 
       DO_REDIS=.FALSE.
    ENDIF

    CCORR_LOCAL=0
! set some more variables
    NB_TOT=W%WDES%NB_TOT
    NB_LOC=W%WDES%NBANDS

    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, (/0.0_q, 0.0_q, 0.0_q/))

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NGLB=NSTRIP*NCPU

    ALLOCATE(CWRN(GRID%MPLWV*W%WDES%NRSPINORS,NGLB), &
         CWRM(GRID%MPLWV*W%WDES%NRSPINORS), &
         CPROJK(W%WDES%NPROD,NGLB), &
         CRHOLM(AUG_DES%NPRO*W%WDES%NRSPINORS*W%WDES%NRSPINORS), &
         CPROJM(W%WDES%NPROD))

strip:   DO NPOS=1,W%WDES%NBANDS,NSTRIP
!==========================================================================

   NSTRIPN=MIN(W%WDES%NBANDS+1-NPOS,NSTRIP)
   NGLB=NSTRIPN*NCPU

!==========================================================================
! initialise variables
!==========================================================================
    CWRN=0
! set 1-kpoint wavedescriptor
    CALL SETWDES(W%WDES,WDESK,NK); CALL SETWGRID_OLD(WDESK,GRID)
    NPL=WDESK%NGVECTOR

!==========================================================================
! fourier transform the bands to be accelerated to real space (CWRN)
! then distribute the CWRN array to all nodes
!==========================================================================
    fftn: DO N=NPOS,NPOS+NSTRIPN-1
       DO ISPINOR=0,WDESK%NRSPINORS-1
          CALL FFTWAV(NPL,WDESK%NINDPW(1), &
               CWRN(1+ISPINOR*GRID%MPLWV,(N-NPOS)*NCPU+NODE_ME_I), &
               W%CPTWFP(1+ISPINOR*NPL,N,NK,ISP),GRID)
       ENDDO
    ENDDO fftn
! if LOVERL copy the projectors into CPROJ array
    IF (W%WDES%LOVERL) THEN
       DO N=NPOS,NPOS+NSTRIPN-1
          CPROJK(1:W%WDES%NPROD,(N-NPOS)*NCPU+NODE_ME_I)= &
               W%CPROJ(1:W%WDES%NPROD,N,NK,ISP)
       ENDDO
    ENDIF
! distribute CWRN and CPROJK to all nodes
#ifdef MPI
    IF (DO_REDIS) THEN
       DO N=1,NGLB
          CALL M_bcast_z_from(WDESK%COMM_INTER,CWRN(:,N), &
               GRID%MPLWV*W%WDES%NRSPINORS,MOD(N-1,NCPU)+1)
#ifndef gammareal
          IF (W%WDES%LOVERL) CALL M_bcast_z_from(WDESK%COMM_INTER,CPROJK(:,N), &
               W%WDES%NPROD,MOD(N-1,NCPU)+1)
#else
          IF (W%WDES%LOVERL) CALL M_bcast_d_from(WDESK%COMM_INTER,CPROJK(:,N), &
               W%WDES%NPROD,MOD(N-1,NCPU)+1)
#endif
       ENDDO
    ENDIF
#endif
   ! loop over bands mq (occupied bands for present q-point on the local CPU)
       mband: DO MQ=1,NB_LOC
          ! speed up when band is empty 
          IF (ABS(W%FERWE(MQ,NK,ISP))<=1E-10) THEN
             CCORR_LOCAL((MQ-1)*NCPU+W%WDES%NB_LOW,:)=0
             CYCLE mband 
          ENDIF

          IF (W%FERTOT((MQ-1)*W%WDES%NB_PAR+W%WDES%NB_LOW,NK,ISP)/=W%FERWE(MQ,NK,ISP)) THEN
             WRITE(0,*) 'internal error in PW_LHF_CORR_POTENTIAL: index into full array incorrect',(MQ-1)*NCPU+W%WDES%NB_LOW,MQ
             STOP
          ENDIF

   ! wavefunction of current band (NQ,MQ) in real space (WQ%CR) and character (WQ%CPROJ)
          ! in some cases the wavefunction is already calculated
          ! this gains a little performance 

          IF ((MQ>=NPOS .AND. MQ<NPOS+NSTRIPN)) THEN
             WQ%CR   =>CWRN(:,(MQ-NPOS)*NCPU+NODE_ME_I)
             WQ%CPROJ=>W%CPROJ(:,MQ,Nk,ISP)
          ELSE
           !
           ! fft of wavefunction to real space CWRM
           ! 
             DO ISPINOR=0,WDESK%NRSPINORS-1
                CALL FFTWAV(W%WDES%NGVECTOR(NK),W%WDES%NINDPW(1,NK), &
                     CWRM(1+ISPINOR*GRID%MPLWV), &
                     W%CPTWFP(1+ISPINOR*W%WDES%NGVECTOR(NK), &
                     MQ,KPOINTS_FULL%NEQUIV(NK),ISP),GRID)
             ENDDO
             WQ%CPROJ=>W%CPROJ(:,MQ,NK,ISP)
             WQ%CR   =>CWRM
          ENDIF
!-----------------------------------------------------------------------------
! calculate charge density
!  GWORK = psi_m*(r) psi_n(r) 
!-----------------------------------------------------------------------------
          ! calculate charge psi_k,n(r) psi_k,m*(r)
          nband: DO N=1,NGLB
             spinor: DO ISPINOR=0,W%WDES%NRSPINORS-1
             DO ISPINOR_=0,W%WDES%NRSPINORS-1
                DO NP=1,GRID%RL%NP
                   MM =NP+ISPINOR *GRID%MPLWV
                   MM_=NP+ISPINOR_*GRID%MPLWV
                   GWORK(NP,1+ISPINOR_+ISPINOR*2)=CONJG(WQ%CR(MM_))*CWRN(MM,N)
                ENDDO
             ENDDO
             ENDDO spinor

             ! add augmentation part to charge (if required)
             IF (W%WDES%LOVERL) THEN
#ifdef LMAXFOCKAE
                CALL DEPSUM_TWO_BANDS_RHOLM_FULL(CPROJK(:,N),WQ%CPROJ(:), WDESK, AUG_DES, &
                      TRANS_MATRIX_FOCK, CRHOLM,W%WDES%LOVERL )
#else
                CALL DEPSUM_TWO_BANDS_RHOLM_FULL_NOAE(CPROJK(:,N),WQ%CPROJ(:), WDESK, AUG_DES, &
                      TRANS_MATRIX_FOCK, CRHOLM,W%WDES%LOVERL, FAST_AUG_FOCK%LMAX)
#endif
                AUG_DES%RINPL=1._q ! multiplicator used by RACC0
                DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
                   CALL RACC0_HF(FAST_AUG_FOCK, AUG_DES, &
                        CRHOLM(1+ISPINOR*AUG_DES%NPRO), GWORK(1,1+ISPINOR))
                ENDDO
             ENDIF

             !  determine CLOCAL = < psi_m | V_local | psi_n >
             CLOCAL=0
             DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
                DO NP=1,GRID%RL%NP
                   ! alternatively the spin order could be reversed in the GWORK
                   CLOCAL=CLOCAL+GCONJG(GWORK(NP,1+ISPINOR))*SV_OLD(NP,ISP+ISPINOR)
                ENDDO
             ENDDO
             CLOCAL=CLOCAL/GRID%NPLWV

             CCORR_LOCAL((MQ-1)*NCPU+W%WDES%NB_LOW,(NPOS-1)*NCPU+N)=CLOCAL

             !  subtract HF term CCORR(m,n) = - < psi_m | H_x | psi_n >
             CSUM=CLOCAL-CCORR((MQ-1)*NCPU+W%WDES%NB_LOW,(NPOS-1)*NCPU+N)

             IF ((MQ-1)*NCPU+W%WDES%NB_LOW==(NPOS-1)*NCPU+N) THEN
                WEIGHT=W%FERWE(MQ,NK,ISP)*W%WDES%WTKPT(NK)*W%WDES%RSPIN
             ELSE
                WEIGHT=SQRT(ABS(W%FERWE(MQ,NK,ISP)*W%FERTOT((NPOS-1)*NCPU+N,NK,ISP)))* &
                     W%WDES%WTKPT(NK)*W%WDES%RSPIN
             ENDIF
             ! the final correction to the potential is real 
             ! since for each complex contribution from bands n,n' a conjugated contribution
             ! from bands n'n is added
             DO ISPINOR=0,W%WDES%NRSPINORS*W%WDES%NRSPINORS-1
                DO NP=1,GRID%RL%NP
                   SV(NP,ISP+ISPINOR)=SV(NP,ISP+ISPINOR)+ &
                        GCONJG(GWORK(NP,1+ISPINOR))*CSUM*WEIGHT
                ENDDO
             ENDDO
             ! the final augmentation contributions are real as well
             CALL DEPSUM_TWO_BANDS( CPROJK(:,N), WQ%CPROJ(:), WDESK, ISP, &
                      CRHODE_CORR(:,:,:,:), CSUM*WEIGHT, W%WDES%LOVERL )


          ENDDO nband
       ENDDO mband
    ENDDO strip
!-----------------------------------------------------------------------------

    CALLMPI( M_sum_g(W%WDES%COMM_INTER,CCORR_LOCAL(1,1),W%WDES%NB_TOT*W%WDES%NB_TOT))

    DEALLOCATE(CWRN,CWRM,CPROJK,CRHOLM,CPROJM)
#undef debug
  END SUBROUTINE PW_LHF_CORR_POTENTIAL


!************************ SUBROUTINE LHF_PW ****************************
!
! determine the plane wave part of the local exchange potential in the
! LHF approximation
! ) first calculate the Slater exchange energy density and the local
!   density
! ) iterate the local exchange potential until selfconsistency is 
!   reached
! if LSV is set SV is considered to be already set, and the iteration
! towards selfconsistency is contined from the start guess
!
! TODO: 
! non collinear case 
!  ) charge conservation (correction to potential)
!  ) symmetrisation
!  ) CHDEN_CORE can be calculated by CORE_LOCAL routine 
!
!***********************************************************************

  SUBROUTINE LHF(HAMILTONIAN,GRID, GRID_SOFT, LATT_CUR, SYMM, T_INFO, INFO, &
            NONLR_S, NONL_S, W, WDES, P, &
            LMDIM, LOVERL, LREAL, LCORE_, POT, CRHODE, CDIJ,  &
            LSV, IU0, EXHF, SWITCH_TO_OEP )
    USE prec
    USE base
    USE wave_mpi
    USE wave
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE subrot
    USE nonl_high
    USE hamil_high
    USE constant
    USE jacobi
    USE scala
    USE main_mpi
    USE fock
    USE poscar
    USE pseudo
    USE pawkli
    USE paw
    USE charge
    USE gridq
    IMPLICIT NONE

    TYPE (ham_handle)  HAMILTONIAN
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SOFT
    TYPE (latt)        LATT_CUR
    TYPE (symmetry)    SYMM
    TYPE (type_info)   T_INFO
    TYPE (info_struct) INFO
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (potcar),TARGET::  P(T_INFO%NTYP)
    INTEGER LMDIM
    LOGICAL LREAL, LOVERL
    LOGICAL LCORE_                                     ! partial core
    TYPE (GRIDQUANT) :: POT                           ! final potential
    OVERLAP,TARGET :: CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! one center occupancy matrix
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)   ! one centre strenght parameters of potentials
    LOGICAL LSV                                       ! is SV set to an initial guess
    INTEGER IU0                                       ! f90 io unit for error dumps
    REAL(q) EXHF                                      ! double counting correction
! if SWITCH_TO_OEP is selected the on site terms 
    LOGICAL, OPTIONAL :: SWITCH_TO_OEP
    TYPE (GRIDQUANT) :: ACCUMULATE1 , ACCUMULATE2
    TYPE (GRIDQUANT) :: POT_SLATER, CHG, CHG_CORE, POT_VEL, POT_NEW
    TYPE (GRIDQUANT), SAVE :: POT_CORR
    COMPLEX(q) CSTRF(GRID_SOFT%MPLWV,T_INFO%NTYP)
  
    OVERLAP CDIJC(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)  ! one centre correction
    OVERLAP CRHODE_CORR(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)! one centre occupancies to correct potential
    INTEGER,SAVE ::  SV_DIM=0
    
! local variables
    LOGICAL   DO_REDIS
    TYPE (wavedes1)    WDES1          ! descriptor for one k-point
    TYPE (wavefun1)    W1             ! current wavefunction
    INTEGER NODE_ME, IONODE, NCPU
    INTEGER NRPLWV_RED, NPROD_RED
    INTEGER NB_TOT, NBANDS, NGVECTOR, NSTRIP, NSTRIP_RED, NSTRIP_ACT
    INTEGER NPOS, NPOS_RED
    INTEGER ISP, ISPINOR, NK, NPL, NPRO, NPRO_O
    INTEGER N1, N2, NPL2
    INTEGER ITER, N, NP, NITER
    GDEF,ALLOCATABLE,TARGET::  CSLATER(:,:,:,:), CCORR(:,:), CCORR_LOCAL(:,:)
    COMPLEX(q),ALLOCATABLE,TARGET::   CH(:,:)
! redistributed plane wave coefficients
    COMPLEX(q), POINTER :: CW_RED(:,:),CH_RED(:,:)
    GDEF   , POINTER :: CPROJ_RED(:,:),CPROW_RED(:,:)
    COMPLEX(q), POINTER :: CW_(:,:)
    COMPLEX(q) ::   CDCHF_HARTREE_FOCK, CDCHF, CDCHF_
    REAL(q) :: CDCHF_ONE_CENTRE,SUM
    OVERLAP :: D_CORE(T_INFO%NIONS,WDES%NCDIJ)
    REAL(q), EXTERNAL :: RHO0
    REAL(q) :: RHO0_SHIFT
    LOGICAL, EXTERNAL :: USEFOCK_ONECENTER
    ! this flag can be used to switch on site contributions off regardless of USEFOCK_ONECENTER
    LOGICAL :: USEFOCKONSITE_=.TRUE.
    INTEGER M, MM
    !
    ! friction parameter is used to accelerate convergence in the
    ! correction term for the LHF potential
    ! see below for details
    REAL(q) :: FRICTION=1.0
    COMPLEX(q) :: DELTA_SV
    LOGICAL :: LCORE
    LCORE=LCORE_

!test
!    LCORE=.FALSE.
!    WRITE(*,*) 'lcore set to',LCORE
!test


!   go to spin up down /spinor representation for CRHODE
    CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .TRUE.)

    CALL ALLOCATE_GRID_QUANTITY(POT_SLATER , GRID_SOFT, WDES%NCDIJ)
    CALL ALLOCATE_GRID_QUANTITY(CHG , GRID_SOFT, WDES%NCDIJ)
    CALL ALLOCATE_GRID_QUANTITY(CHG_CORE , GRID_SOFT, 1)
    CALL ALLOCATE_GRID_QUANTITY(POT_VEL , GRID_SOFT, WDES%NCDIJ)
    CALL ALLOCATE_GRID_QUANTITY(POT_NEW , GRID_SOFT, WDES%NCDIJ)
! the following quantities are for accumulation in real space
! when loops over all bands
    CALL ALLOCATE_GRID_QUANTITY_FORCE_RL(ACCUMULATE2, GRID, GRID_SOFT, WDES%NCDIJ)
    CALL ALLOCATE_GRID_QUANTITY_FORCE_RL(ACCUMULATE1, GRID, GRID_SOFT, WDES%NCDIJ)

!-----------------------------------------------------------------------
! allocate local work arrays
!-----------------------------------------------------------------------
    IF (SV_DIM/=GRID_SOFT%MPLWV) THEN
       IF (SV_DIM/=0) THEN
          CALL DEALLOCATE_GRID_QUANTITY(POT_CORR)
       ENDIF
       SV_DIM=GRID_SOFT%MPLWV
       CALL ALLOCATE_GRID_QUANTITY(POT_CORR , GRID_SOFT, WDES%NCDIJ)
       POT_CORR=0.0_q
    ENDIF

! if LSV is set, POT_CORR, the correction to the potential from the previous call
! is reused and only few iterations are performed
    IF (.NOT.LSV) THEN
       POT_CORR=0.0_q
       NITER=20 
       FRICTION= 2.0
    ELSE
       NITER=2
       FRICTION= 2.0
    ENDIF
!-----------------------------------------------------------------------
! pseudo core charge on coarse grid
!-----------------------------------------------------------------------
    CALL STUFAK(GRID_SOFT,T_INFO,CSTRF)

    CHG_CORE=0.0_q;  CHG_CORE%REALSPACE=.TRUE.

    IF (LCORE) CALL RHOPAR(GRID_SOFT,T_INFO,INFO,LATT_CUR,P,CSTRF,CHG_CORE%RG,-1)
    IF (WDES%ISPIN==2 .OR. WDES%LNONCOLLINEAR ) CHG_CORE=CHG_CORE*(0.5_q)

    CDCHF=0
    CDCHF_ONE_CENTRE=0

#ifdef MPI
    NODE_ME=WDES%COMM%NODE_ME
    IONODE =WDES%COMM%IONODE
    NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
    IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('LHF: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#else
    NODE_ME=0
    IONODE =0
    NCPU=1
#endif
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
    IF (NCPU /= 1) THEN
       DO_REDIS=.TRUE.
       NRPLWV_RED=WDES%NRPLWV/NCPU
       NPROD_RED =WDES%NPROD /NCPU
    ELSE
       DO_REDIS=.FALSE.
       NRPLWV_RED=WDES%NRPLWV
       NPROD_RED =WDES%NPROD
    ENDIF
    
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS
    
    ! set NSTRIP between [1 and 32]
    NSTRIP=NSTRIP_STANDARD
    
    ! allocate work space
    ALLOCATE( &
         &     CSLATER(NB_TOT,NB_TOT,WDES%NKPTS,WDES%ISPIN),CCORR(NB_TOT, NB_TOT), &
         &     CCORR_LOCAL(NB_TOT, NB_TOT),CH(WDES%NRPLWV,NSTRIP))

    CSLATER=0
    
    IF (DO_REDIS) THEN
       ALLOCATE(CW_(WDES%NRPLWV,WDES%NBANDS))
    ENDIF
!=======================================================================
!
!  first step calculate the Slater energy density 
!   e(r)      = \sum_kn m psi*_k,n(r) V_x(r,r') psi_k,n(r')
!  and the Fock matrix enclosed in between all states
!  CSLATER(m,n) = <psi_k,n| V_x | psi_k,m> 
!
!=======================================================================
    ACCUMULATE1=0.0_q; ACCUMULATE1%REALSPACE=.TRUE.
    ACCUMULATE2=0.0_q; ACCUMULATE2%REALSPACE=.TRUE.

    spin:  DO ISP=1,WDES%ISPIN
    kpoint: DO NK=1,WDES%NKPTS
       IF (W%WDES%LOVERL) THEN
          IF (NONLR_S%LREAL) THEN
             CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,W%WDES)
          ELSE
             CALL PHASE(W%WDES,NONL_S,NK)
          ENDIF
       ENDIF


      IF (DO_REDIS) THEN
         CW_   = W%CW(:,:,NK,ISP)
      ELSE
         CW_   => W%CW(:,:,NK,ISP)
      ENDIF
      CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

      ! get pointers for redistributed wavefunctions
      IF (DO_REDIS) THEN
        CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, CW_(1,1))
        CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
        CALL SET_WPOINTER(CH_RED,   NRPLWV_RED, NCPU*NSTRIP, CH(1,1))
      ELSE
        CW_RED    => W%CW(:,:,NK,ISP)
        CPROJ_RED => W%CPROJ(:,:,NK,ISP)
        CH_RED    => CH(:,:)
      ENDIF

      ! set number of wavefunctions after redistribution
      NPL = WDES1%NPL     ! number of plane waves/node after data redistribution
      NPRO= WDES1%NPRO    ! number of projected wavef. after data redistribution
      NPRO_O=0

      CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

      NGVECTOR=WDES1%NGVECTOR

      strip: DO NPOS=1,NBANDS,NSTRIP
         NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

         ! calculate accelerations on current strip of bands
         ! from the HF Hamiltonian
         CALL PW_SLATER_POTENTIAL(GRID, LMDIM, LATT_CUR, W,   &
              NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
              CH, ACCUMULATE1%RG, ACCUMULATE2%RG, P, CDCHF )
        
         ! redistribute wavefunctions
         ! CW_ is then redistributed up to and including 1...NPOS+NSTRIP_ACT
         IF (DO_REDIS) THEN
            CALL REDIS_PW(WDES1, NSTRIP_ACT, CW_(1,NPOS))
            CALL REDIS_PW(WDES1, NSTRIP_ACT, CH(1,1))
         ENDIF

         NPOS_RED  =(NPOS-1)*NCPU+1
         NSTRIP_RED=NSTRIP_ACT*NCPU

         CALL ORTH1('U', &
              CW_RED(1,1),CH_RED(1,1),CPROJ_RED(1,1), &
              CPROJ_RED(1,NPOS_RED),NB_TOT, &
              NPOS_RED, NSTRIP_RED, NPL,NPRO_O,NRPLWV_RED,NPROD_RED,CSLATER(1,1,NK,ISP))

      ENDDO strip

      CALLMPI( M_sum_g(WDES%COMM,CSLATER(1,1,NK,ISP),NB_TOT*NB_TOT))
    ENDDO kpoint
    ENDDO spin
!=======================================================================
!
! now symmetrize the resulting energy density
! and calculate the local potential
!
!=======================================================================
    CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF,1))
    CDCHF_HARTREE_FOCK=CDCHF

    ! merge charge from all bands
    CALL SUMRL_GQ( ACCUMULATE2, CHG, WDES%COMM_INTER)
    CALL SUMRL_GQ( ACCUMULATE1, POT_SLATER, WDES%COMM_INTER)

    CALL CHARGE_SYM_REAL_SPACE( WDES, GRID_SOFT, CHG%RG, SYMM, T_INFO, LATT_CUR)

    CDCHF=0
    DO ISP=1,WDES%NCDIJ,MAX(1,WDES%NCDIJ-1)
       DO NP=1,GRID_SOFT%RL%NP
          CDCHF=CDCHF  -POT_SLATER%RG(NP,ISP)*(0.5_q/GRID_SOFT%NPLWV)
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF,1))

    IF (ABS(REAL(CDCHF_HARTREE_FOCK,q)-REAL(CDCHF,q))>1E-2) THEN
       IF (IU0>=0) WRITE(IU0,*) 'internal error in LHF: Hartree-Fock energy and Slater energy differ', &
            REAL(CDCHF_HARTREE_FOCK,q),REAL(CDCHF,q)
       STOP
    ENDIF

    POT=0.0_q ; POT%REALSPACE = .TRUE.
    CALL POTENTIAL_FROM_ENERGY(GRID_SOFT, POT_SLATER%RG, POT_CORR%RG, POT%RG, CHG%RG, CHG_CORE%RG, WDES%NCDIJ)
    ! symmetrize potential
    CALL CHARGE_SYM_REAL_SPACE( WDES, GRID_SOFT, POT%RG, SYMM, T_INFO, LATT_CUR)
#ifdef debug
    DO ISP=1,WDES%NCDIJ
       CALL WRT_RL_LINE(IU0, GRID_SOFT, CHG%RG(1,ISP))
    ENDDO
    DO ISP=1,WDES%NCDIJ
       CALL WRT_RL_LINE(IU0, GRID_SOFT, POT_SLATER%RG(1,ISP))
    ENDDO
    DO ISP=1,WDES%NCDIJ
       CALL WRT_RL_LINE(IU0, GRID_SOFT, POT%RG(1,ISP))
    ENDDO
#endif

    CDCHF_=0
    DO ISP=1,WDES%NCDIJ
       DO NP=1,GRID_SOFT%RL%NP
          CDCHF_=CDCHF_-POT%RG(NP,ISP)*(0.5_q/GRID_SOFT%NPLWV)*CHGCONJG(CHG%RG(NP,ISP))
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF_,1))

    CDCHF=0
    DO ISP=1,WDES%NCDIJ,MAX(1,WDES%NCDIJ-1)
       DO NP=1,GRID_SOFT%RL%NP
          CDCHF =CDCHF -POT%RG(NP,ISP)*(0.5_q/GRID_SOFT%NPLWV)*CHGCONJG(CHG_CORE%RG(NP,1))
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF,1))
    CDCHF=CDCHF+CDCHF_

    IF (ABS(REAL(CDCHF_HARTREE_FOCK,q)-REAL(CDCHF,q))>2E-2) THEN
       IF (IU0>=0) WRITE(IU0,*) 'internal error in LHF: Hartree-Fock energy and Slater energy differ after division by potential', &
            REAL(CDCHF_HARTREE_FOCK,q),REAL(CDCHF,q)
       STOP
    ENDIF
    IF (IU0>=0) WRITE(IU0,'(A,4F14.7)') '-Hartree energy',REAL(CDCHF,q),REAL(CDCHF_,q)
!=======================================================================
!
!  second step iterate the corrections to the potential
!  at each k-point and add the corrections to the energy density
!  
!=======================================================================
    POT_VEL=0.0_q
!-----------------------------------------------------------------------
!  calculate the onsite strength paramters
!   D_ij= - <psi^ps_i| V_x + V_local | psi^ps_j> +
!           <psi_i   | V_x + V_local | psi_j> 
!-----------------------------------------------------------------------
    CALL CORE_LOCAL(GRID_SOFT, P, T_INFO, LATT_CUR, WDES%NCDIJ, POT%RG, D_CORE)

    IF (USEFOCK_ONECENTER() .AND. USEFOCKONSITE_) THEN
       CALL  SET_DD_LHF(WDES, P , T_INFO, LOVERL, LSV, LCORE, LMDIM, & 
            CDIJC, CDIJ, D_CORE, CRHODE, CDCHF_ONE_CENTRE)
       IF (IU0>=0) WRITE(IU0,'(A,4F14.7)') '-one centre Hartree energy',CDCHF_ONE_CENTRE
    ELSE
       CDCHF_ONE_CENTRE=0
       CDIJC=0
       CDIJ =0
    ENDIF
!=======================================================================
    DO ITER=1,NITER
!=======================================================================

       ACCUMULATE1=0.0_q
       CRHODE_CORR=0

       ! broadcase data from root node to all nodes
       CALL BROAD_CAST_GQ( POT, ACCUMULATE2, WDES%COMM_INTER)

       spin2:  DO ISP=1,WDES%ISPIN
       kpoint2: DO NK=1,WDES%NKPTS
#ifdef debug
          CALL DUMP_DLLMM( "TOTAL-ONE_CENTER",CDIJC(:,:,1,ISP), P(1))
          CALL DUMP_DLLMM( "LOCAL-ONE-CENTER",CDIJ(:,:,1,ISP), P(1))
#endif
!-----------------------------------------------------------------------
!  calculate onsite correction to the strength matrix
!  CCORR(m,n) = <psi_k,n| beta_i> D_ij <beta_j | psi_k,m> 
!-----------------------------------------------------------------------
          CCORR=0
          IF (USEFOCK_ONECENTER() .AND. USEFOCKONSITE_) THEN
             CALL ONE_CENTER_BETWEEN_STATES( HAMILTONIAN, LATT_CUR, LOVERL, WDES, W, NK, ISP, & 
                  LMDIM, CDIJC, CCORR)
             CCORR= CSLATER(:,:,NK,ISP)+CCORR
#ifdef debug
             CALL DUMP_HAM( "on site corrections to correction",WDES, CCORR)
#endif
          ELSE
             CCORR= CSLATER(:,:,NK,ISP)
          ENDIF
!-----------------------------------------------------------------------
! determine correction term on the plane wave grid
! for present k-point
!-----------------------------------------------------------------------
          CALL PW_LHF_CORR_POTENTIAL(GRID, LMDIM, LATT_CUR, W, &
               NONLR_S, NONL_S, NK, ISP, NSTRIP, &
               ACCUMULATE2%RG, ACCUMULATE1%RG, CCORR, CCORR_LOCAL, CRHODE_CORR )
#ifdef debug
          CALL DUMP_HAM( "total correction",WDES, CCORR_LOCAL-CCORR)
          CALL DUMP_HAM( "Slater",WDES, CSLATER(:,:,NK,ISP))
          CALL DUMP_HAM( "pw local Hamiltonmatrix minus Slater",WDES, CCORR- CSLATER(:,:,NK,ISP))
          SUM=0
          DO N=1,WDES%NB_TOT
             SUM=SUM+(CCORR(N,N)- CSLATER(N,N,NK,ISP))*W%FERTOT(N,NK,ISP)
          ENDDO
          WRITE(*,*) 'sum of diagonal elements',SUM
#endif
          CCORR= CCORR_LOCAL

       ENDDO kpoint2
       ENDDO spin2

       CALL SUMRL_GQ( ACCUMULATE1, POT_CORR, WDES%COMM_INTER)
#ifdef realmode
       CALLMPI( M_sum_d(WDES%COMM_INTER, CRHODE_CORR, SIZE(CRHODE_CORR)))
#else
       CALLMPI( M_sum_z(WDES%COMM_INTER, CRHODE_CORR, SIZE(CRHODE_CORR)))
#endif
#ifdef debug
       DO ISP=1,WDES%NCDIJ
          CALL WRT_RL_LINE(IU0, GRID_SOFT, POT_CORR%RG(1,ISP))
       ENDDO
       
       DO ISP=1,WDES%NCDIJ
          CALL DUMP_DLLMM( "ONE-CENTRE-CORRECTION OCCUPANCIS",CRHODE_CORR(:,:,1,ISP), P(1))
       ENDDO
#endif
!-----------------------------------------------------------------------
! now add correction and Slater term on the plane wave grid
!    the average energy correction term (POT_CORR) is 0
!    -  \int dr \sum_ab  psi_a(r) psi*_b(r) < psi_a | V_x | psi_b>
!    +  \int dr \sum_ab  psi_a(r) psi*_b(r) 
!          \int dr' psi*_a(r') psi_b(r') \sum_c psi*_c(r') (V_x psi_c)(r') /
!                                        \sum_d psi*_d(r') psi_d(r') =
!   =- \sum_a  < psi_a | V_x | psi_a> 
!    +  \int dr' \sum_c  psi*_c(r') (V_x psi_c)(r') = 0
!-----------------------------------------------------------------------
       DO ISP=1,WDES%NCDIJ
          
          CALL FFT_RC_SCALE(POT_CORR%RG(1,ISP),POT_CORR%RG(1,ISP),GRID_SOFT)
          RHO0_SHIFT=ABS(RHO0(GRID_SOFT, POT_CORR%RG(1,ISP)))
          IF (RHO0_SHIFT>=1E-2) THEN
             IF (IU0>=0) WRITE(IU0,*) 'internal error in LHF: energy density shift observed',ISP,RHO0_SHIFT
          ENDIF
          CALL SET_RHO0(GRID_SOFT, POT_CORR%RG(1,ISP), 0.0_q)
          CALL FFT3D(POT_CORR%RG(1,ISP),GRID_SOFT,1)
       ENDDO
!-----------------------------------------------------------------------
! update plane wave potential
!-----------------------------------------------------------------------
       CALL POTENTIAL_FROM_ENERGY(GRID_SOFT, POT_SLATER%RG, POT_CORR%RG, POT_NEW%RG, CHG%RG, CHG_CORE%RG, WDES%NCDIJ)

       CDCHF_=0
       DO ISP=1,WDES%NCDIJ
          DO NP=1,GRID_SOFT%RL%NP
             ! difference between new and previous potential
             DELTA_SV=POT_NEW%RG(NP,ISP)-POT%RG(NP,ISP)
             ! 
             ! damped molecular dynamics for the potential
             ! for FRICTION=2 the next potential equals exactly the new potential
             ! for smaller FRICTION parameters a damped equation of motion is used
             ! the update of the one center terms is only correct for
             ! FRICTION=2
             POT_VEL%RG(NP,ISP)=((1-FRICTION/2)*POT_VEL%RG(NP,ISP)+2*DELTA_SV)/(1+FRICTION/2)
             POT%RG(NP,ISP)=POT%RG(NP,ISP)+POT_VEL%RG(NP,ISP)
             
             !*********************************************************
             ! simple steepest descent update
             !*********************************************************
             POT%RG(NP,ISP)=POT_NEW%RG(NP,ISP)
             
             CDCHF_=CDCHF_-POT%RG(NP,ISP)*CHGCONJG(CHG%RG(NP,ISP))*(0.5_q/GRID_SOFT%NPLWV)
          ENDDO
       ENDDO
       CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF_,1))
       
       CDCHF=0
       DO ISP=1,WDES%NCDIJ,MAX(1,WDES%NCDIJ-1)
          DO NP=1,GRID_SOFT%RL%NP
             CDCHF =CDCHF -POT%RG(NP,ISP)*(0.5_q/GRID_SOFT%NPLWV)*CHGCONJG(CHG_CORE%RG(NP,1))
          ENDDO
       ENDDO
       CALLMPI( M_sum_z(WDES%COMM_INTER,CDCHF,1))
       CDCHF=CDCHF+CDCHF_
       
       ! symmetrize potential
       CALL CHARGE_SYM_REAL_SPACE( WDES, GRID_SOFT, POT%RG, SYMM, T_INFO, LATT_CUR)
       IF (IU0>=0) WRITE(IU0,'(A,4F14.7)') '-plane wave Hartree energy from locpot',REAL(CDCHF,q),REAL(CDCHF_,q)
!-----------------------------------------------------------------------
! update on site potentials
!-----------------------------------------------------------------------
       CALL CORE_LOCAL(GRID_SOFT, P, T_INFO, LATT_CUR, WDES%NCDIJ, POT%RG, D_CORE)

       IF (USEFOCK_ONECENTER() .AND. USEFOCKONSITE_) THEN
          CALL  SET_DD_LHF(WDES, P , T_INFO, LOVERL, .TRUE., LCORE, LMDIM, &
               CDIJC, CDIJ, D_CORE, CRHODE,  CDCHF_ONE_CENTRE, CRHODE_CORR )

          IF (IU0>=0) WRITE(IU0,'(A,4F14.7)') '-one centre Hartree energy from locpot',REAL(CDCHF_ONE_CENTRE,q)
       ENDIF
    ENDDO
    
! back to total charge/ magnetisation representation
    CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .FALSE.)

#ifdef debug
    DO ISP=1,WDES%NCDIJ
       CALL WRT_RL_LINE(IU0, GRID_SOFT, POT%RG(1,ISP))
    ENDDO
    DO ISP=1,WDES%NCDIJ
       CALL DUMP_DLLMM( "LOCAL",CDIJ(:,:,1,ISP), P(1))
    ENDDO
#endif
    DO ISP=1,WDES%NCDIJ
       CALL FFT_RC_SCALE(POT%RG(1,ISP),POT%RG(1,ISP),GRID_SOFT)
       CALL SETUNB(POT%RG(1,ISP),GRID_SOFT)
    ENDDO

    DEALLOCATE(CSLATER, CCORR, CCORR_LOCAL, CH)

    IF (DO_REDIS) THEN
       DEALLOCATE(CW_)
    ENDIF

    EXHF=(2*CDCHF_-CDCHF) +CDCHF_ONE_CENTRE

    IF (PRESENT(SWITCH_TO_OEP)) THEN
       IF (SWITCH_TO_OEP) THEN
          CALL LHF_TO_OEP(WDES, P , T_INFO, LOVERL, LCORE, LMDIM, &
               D_CORE, CRHODE )
       ENDIF
    ENDIF

    CALL DEALLOCATE_GRID_QUANTITY(POT_SLATER)
    CALL DEALLOCATE_GRID_QUANTITY(CHG )
    CALL DEALLOCATE_GRID_QUANTITY(CHG_CORE )
    CALL DEALLOCATE_GRID_QUANTITY(POT_VEL )
    CALL DEALLOCATE_GRID_QUANTITY(POT_NEW )
    CALL DEALLOCATE_GRID_QUANTITY(ACCUMULATE1)
    CALL DEALLOCATE_GRID_QUANTITY(ACCUMULATE2)
    RETURN
  END SUBROUTINE LHF

 
!************************ SUBROUTINE CORE_LOCAL ************************
!
! this subroutine calculates the local potential times the
! partial core for each ions
!
!***********************************************************************

    SUBROUTINE CORE_LOCAL(GRIDC, P, T_INFO, LATT_CUR, NCDIJ, SV, D_CORE)
      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT NONE

      INTEGER NCDIJ
      
      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR

      RGRID   SV(DIMREAL(GRIDC%MPLWV),NCDIJ) ! local exchange potential
      COMPLEX(q) SV_REC(GRIDC%MPLWV,NCDIJ) ! Slater energy density
      OVERLAP    D_CORE(T_INFO%NIONS,NCDIJ)
      ! local
      INTEGER ISP, NIS, NT, NIADD, N, N1, N2, N3, NC, NADDR, NI, NGP, N1P, NG
      REAL(q) :: ARGSC, PSGMA2, GX, GY, GZ, G, ARG, REM, V1, V2, V3, V4, &
           T0, T1, T2, T3, G1, G2, G3, FACTM
      COMPLEX(q) :: CX, CEXPF, CE
      REAL(q) :: WORK(GRIDC%RC%NP)

      D_CORE=0
      
      ! charge density in real space
      DO ISP=1,NCDIJ
         CALL FFT_RC_SCALE(SV(1,ISP),SV_REC(1,ISP),GRIDC)
      ENDDO

      NIS=1
      typ: DO NT=1,T_INFO%NTYP

       NIADD=T_INFO%NITYP(NT)
       IF (ASSOCIATED(P(NT)%PSPCOR)) THEN

       ! interpolate the pseudopotential on the grid of reciprocal
       ! lattice-vectors

       ARGSC=NPSPTS/P(NT)%PSGMAX
       PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS*3

       DO N=1,GRIDC%RC%NP
          N1= MOD((N-1),GRIDC%RC%NROW) +1
          NC= (N-1)/GRIDC%RC%NROW+1
          N2= GRIDC%RC%I2(NC)
          N3= GRIDC%RC%I3(NC)

          ! calculate the magnitude of the reciprocal lattice vector
          GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
          GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
          GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

          G=SQRT(GX**2+GY**2+GZ**2)*TPI

          IF (G/=0 .AND. G<PSGMA2) THEN
             ! convert the magnitude of the reciprocal latice vector to a position
             ! in the charge-dens. array  and interpolate the atomic-chargedensity
             ARG=(G*ARGSC)+1
             NADDR=MAX(INT(ARG),2)
             REM=ARG-NADDR
             V1=P(NT)%PSPCOR(NADDR-1)
             V2=P(NT)%PSPCOR(NADDR)
             V3=P(NT)%PSPCOR(NADDR+1)
             V4=P(NT)%PSPCOR(NADDR+2)
             T0=V2
             T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
             T2=(V1+V3-(2*V2))/2._q
             T3=(V4-V1+(3*(V2-V3)))/.6_q
             WORK(N)=(T0+REM*(T1+REM*(T2+REM*T3)))
          ELSE
             WORK(N)=P(NT)%PSPCOR(1)
          ENDIF
      ENDDO

      ! determine inproduct between the partial core charge
      ! and potential

      ion: DO NI=NIS,NIADD+NIS-1
         CX =EXP(-CITPI*T_INFO%POSION(1,NI))
         G1 =T_INFO%POSION(1,NI)*xmin(GRIDC%NGX)

         DO NC=1,GRIDC%RC%NCOL
           NGP=(NC-1)*GRIDC%RC%NROW+1

           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           CE=EXP(-CITPI*(G3+G2+G1))

           DO N1P=0,GRIDC%RC%NROW-1
              SETN1fromN1P
              NG=NGP+N1
              N1=N1+1
              
              SETFACT1
              SETFACT
              CEXPF=CE
              CE=CE*CX

              DO ISP=1,NCDIJ
                 D_CORE(NI,ISP)=D_CORE(NI,ISP)+WORK(NG)*MULFACT (SV_REC(NG,ISP)*CONJG(CEXPF))
              ENDDO
           ENDDO
         ENDDO
      ENDDO ion
      ENDIF

      NIS=NIS+NIADD
      ENDDO typ

#ifdef realmode
      CALLMPI( M_sum_d(GRIDC%COMM, D_CORE,SIZE(D_CORE)))
#else
      CALLMPI( M_sum_z(GRIDC%COMM, D_CORE,SIZE(D_CORE)))
#endif

      ! division by two since we want the up and down core charge only
      DO ISP=1,NCDIJ
         D_CORE(:,ISP)=D_CORE(:,ISP)/2
      ENDDO
      RETURN
    END SUBROUTINE CORE_LOCAL

!************************ SUBROUTINE CORE_LOCAL ************************
!
! this subroutine symmetrizes a charge density or potential
! supplied in real space or reciprocal space
! the result is returned in real space after symmetrization
!
!***********************************************************************

    
    SUBROUTINE CHARGE_SYM_REAL_SPACE( WDES, GRID_SOFT, CHDEN, SYMM, T_INFO, LATT_CUR)
      USE prec
      USE base
      USE mgrid
      USE wave
      USE poscar
      USE lattice
      
      TYPE (grid_3d)     GRID_SOFT
      TYPE (wavedes)     WDES
      TYPE (symmetry)    SYMM
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      RGRID  CHDEN(DIMREAL(GRID_SOFT%MPLWV),WDES%NCDIJ)! charge density
   ! local
      INTEGER ISP

      IF (SYMM%ISYM==1 .OR. SYMM%ISYM==2) THEN
         ! FFT to reciprocal space
         DO ISP=1,WDES%NCDIJ
            CALL FFT_RC_SCALE(CHDEN(1,ISP),CHDEN(1,ISP),GRID_SOFT)
         ENDDO

         ! (total charge, magnetisation) storage convention as required by RHOSYM
         CALL RC_FLIP(CHDEN,GRID_SOFT,WDES%NCDIJ,.FALSE.)

         IF (WDES%LNONCOLLINEAR) THEN
            CALL RHOSYM(CHDEN(1,1),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,1)
            ! Marsman: Insert symmetrization of vector field
            IF (.NOT.WDES%LSPIRAL) &
                 &   CALL SYMFIELD(CHDEN(1,2),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
         ELSE
            DO ISP=1,WDES%ISPIN
               CALL RHOSYM(CHDEN(1,ISP),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
            ENDDO
         ENDIF

         CALL RC_FLIP(CHDEN,GRID_SOFT,WDES%NCDIJ,.TRUE.)

         ! back to real space 
         DO ISP=1,WDES%NCDIJ
            CALL FFT3D(CHDEN(1,ISP),GRID_SOFT,1)
         ENDDO
      ENDIF
      
    END SUBROUTINE CHARGE_SYM_REAL_SPACE


!***********************************************************************
!
! this subroutine calculates the local potential from the 
! local energy density and the supplied charge density
! a Hermitian potential with the property
!
!   V_sigma sigma' rho*_sigma',sigma'' = e_sigma,sigma''
!
! is determined in the non collinear case
! 
!***********************************************************************
      

    SUBROUTINE POTENTIAL_FROM_ENERGY(GRID, SV_SLATER, SV_CORR, SV, CHDEN, CHDEN_CORE, NCDIJ)

      USE prec
      USE mgrid
      IMPLICIT NONE

      TYPE (grid_3d)     GRID
      INTEGER NCDIJ
      RGRID   SV_SLATER(:,:)
      RGRID   SV_CORR(:,:)
      RGRID   SV(:,:)
      RGRID   CHDEN(:,:)
      RGRID   CHDEN_CORE(:,:)
      RGRID,ALLOCATABLE ::   CHDEN_TOTAL(:,:)

      INTEGER NP, ISP
      REAL(q) :: MAG, PROJ_E, E_UP, E_DWN, RHO_UP, RHO_DWN, V_UP, V_DWN

      IF (NCDIJ==1 .OR. NCDIJ==2) THEN

         DO ISP=1,NCDIJ
            DO NP=1,GRID%RL%NP
               IF (ABS(CHDEN(NP,ISP)+CHDEN_CORE(NP,1))>=MIN_CHARGE) THEN
                  SV(NP,ISP)=REAL(SV_SLATER(NP,ISP)+SV_CORR(NP,ISP),q) & 
                       /REAL(CHDEN(NP,ISP)+CHDEN_CORE(NP,1),q)
               ELSE
                  SV(NP,ISP)=0
               ENDIF
            ENDDO
         ENDDO
      ELSE
         ALLOCATE(CHDEN_TOTAL(DIMREAL(GRID%MPLWV),NCDIJ))
         DO ISP=1,NCDIJ
            ! either charge or energy density must be conjugated
            ! the final potential should be parallel to the local magnetisation
            ! axis, which is achived by conjugating the energy density
            DO NP=1,GRID%RL%NP
               CHDEN_TOTAL(NP,ISP)=CHDEN(NP,ISP)
               SV(NP,ISP)=CHGCONJG(SV_SLATER(NP,ISP)+SV_CORR(NP,ISP))
            ENDDO
         ENDDO
         DO ISP=1,NCDIJ,NCDIJ-1
            DO NP=1,GRID%RL%NP
               CHDEN_TOTAL(NP,ISP)=CHDEN_TOTAL(NP,ISP)+CHDEN_CORE(NP,1)
            ENDDO
         ENDDO
         ! storage convention to (total, magnetisation)
         CALL RL_FLIP(CHDEN_TOTAL,GRID,NCDIJ,.FALSE.)
         CALL RL_FLIP(SV,GRID,NCDIJ,.FALSE.)

         ! project the energy density onto the local magnetisation axis
         DO NP=1,GRID%RL%NP
            MAG= REAL(CHDEN_TOTAL(NP,2),q)*REAL(CHDEN_TOTAL(NP,2),q)+ & 
                 REAL(CHDEN_TOTAL(NP,3),q)*REAL(CHDEN_TOTAL(NP,3),q)+ &
                 REAL(CHDEN_TOTAL(NP,4),q)*REAL(CHDEN_TOTAL(NP,4),q)
            PROJ_E=(REAL(CHDEN_TOTAL(NP,2),q)*REAL(SV(NP,2),q)+ &
                 REAL(CHDEN_TOTAL(NP,3),q)*REAL(SV(NP,3),q)+ &
                 REAL(CHDEN_TOTAL(NP,4),q)*REAL(SV(NP,4),q))*(1._q/SQRT(MAG))

            E_UP =SV(NP,1)-PROJ_E
            E_DWN=SV(NP,1)+PROJ_E

            RHO_UP =CHDEN_TOTAL(NP,1)-SQRT(MAG)
            RHO_DWN=CHDEN_TOTAL(NP,1)+SQRT(MAG)

            IF (ABS(RHO_UP)>=MIN_CHARGE) THEN
               V_UP=E_UP/RHO_UP
            ELSE
               V_UP=0
            ENDIF
            IF (ABS(RHO_DWN)>=MIN_CHARGE) THEN
               V_DWN=E_DWN/RHO_DWN
            ENDIF

            SV(NP,1)=V_UP+V_DWN
            SV(NP,2)=REAL(CHDEN_TOTAL(NP,2),q)*(V_DWN-V_UP)*(1._q/SQRT(MAG))
            SV(NP,3)=REAL(CHDEN_TOTAL(NP,3),q)*(V_DWN-V_UP)*(1._q/SQRT(MAG))
            SV(NP,4)=REAL(CHDEN_TOTAL(NP,4),q)*(V_DWN-V_UP)*(1._q/SQRT(MAG))

         ENDDO
!         DO ISP=1,NCDIJ
!            CALL WRT_RL_LINE(6, GRID, SV(1,ISP))
!         ENDDO

         CALL RL_FLIP(SV,GRID,NCDIJ,.TRUE.)

         DEALLOCATE(CHDEN_TOTAL)
      ENDIF
      
    END SUBROUTINE POTENTIAL_FROM_ENERGY


END MODULE pwkli


!************************ SUBROUTINE SETDIJ_FOCK ***********************
!
! this subroutine does essentially the same as SETDIJ put projects
! the total potential onto the additional FOCK augmentation charges
!
! spiral code, displacements as well as support for LDIAGONAL have
! been removed
!
!***********************************************************************


    SUBROUTINE SETDIJ_FOCKAE(WDES, GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, LOVERL, &
        LMDIM,CDIJ,CQIJ, CVTOT_, IRDMAA, IRDMAX )
      USE prec
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE constant
      USE augfast
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  :: GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER :: GRIDC
      TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES

      INTEGER  IRDMAX      ! allocation required for augmentation
      INTEGER  IRDMAA      ! actual maximum augmentation index
      INTEGER  LMDIM
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
               CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      RGRID,TARGET :: CVTOT_(DIMREAL(GRIDC_%MPLWV),WDES%NCDIJ)
      LOGICAL  LOVERL
!  work arrays
      LOGICAL LADDITIONAL
      REAL(q)   DLM(256)
      REAL(q)   ,ALLOCATABLE ::   DIST(:),DEP(:),POT(:),YLM(:,:)
      INTEGER,ALLOCATABLE ::   NLI(:)
      RGRID,POINTER :: CVTOT(:),CWORK(:)
      REAL(q) QVEC(3),QR
      REAL(q),ALLOCATABLE :: XS(:),YS(:),ZS(:)
      TYPE (potcar), POINTER :: PP
      INTEGER :: LYDIM, LMYDIM, ISP, ISP_INC, NI, NIP, LOW, NT, LYMAX, MPLOW, & 
                 LL, LLP, LMP, LM, N, INDMAX, LDEP_INDEX, L, LP, M, MP, & 
                 INDYLM, INDPYL, IND, QDEP_FOCK_INDEX
      INTEGER, EXTERNAL :: MAXL_AUG, MAXL1
#ifdef MPI
      ! mind: in the MPI version CTMP holds all elements
      ! to achieve good load balancing CDIJ is calculated locally on all nodes,
      ! merged, and finally distributed among nodes
      OVERLAP,ALLOCATABLE :: CTMP(:,:,:,:)
      ALLOCATE(CTMP(LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
#else
#define CTMP CDIJ
#endif
      ! LADDITIONAL uses an even finer grid for
      ! calculating the augmentation charges
      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)
      CTMP  =0
      IRDMAA=0
      
 overl: IF (LOVERL) THEN
      ! storage convention to (total,magnetization)
      CALL RL_FLIP(CVTOT_(1,1),GRIDC_,WDES%NCDIJ,.FALSE.)

      LYDIM=MAXL_AUG(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE( DIST(IRDMAX),DEP(IRDMAX),POT(IRDMAX), &
     &          YLM(IRDMAX,LMYDIM),NLI(IRDMAX))
      IF (LADDITIONAL) THEN
         ALLOCATE(CVTOT(DIMREAL(GRIDUS%MPLWV)),CWORK(DIMREAL(GRIDC_%MPLWV)))
      ENDIF
      
      ALLOCATE(XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX))
      IF (WDES%LSPIRAL) THEN
      ! Take QSPIRAL from direct to cartesian coordinates
         QVEC(1)=WDES%QSPIRAL(1)*LATT_CUR%B(1,1)+WDES%QSPIRAL(2)*LATT_CUR%B(1,2)+WDES%QSPIRAL(3)*LATT_CUR%B(1,3)
         QVEC(2)=WDES%QSPIRAL(1)*LATT_CUR%B(2,1)+WDES%QSPIRAL(2)*LATT_CUR%B(2,2)+WDES%QSPIRAL(3)*LATT_CUR%B(2,3)
         QVEC(3)=WDES%QSPIRAL(1)*LATT_CUR%B(3,1)+WDES%QSPIRAL(2)*LATT_CUR%B(3,2)+WDES%QSPIRAL(3)*LATT_CUR%B(3,3)
         WRITE(0,*) 'internal error in SETDIJ_FOCK: LSPIRAL is not supported'
         STOP
      ENDIF
 
 spin:DO ISP=1,WDES%NCDIJ

      IF (LADDITIONAL) THEN
         RINPL=1._q/GRIDC_%NPLWV
         CALL RL_ADD(CVTOT_(1,ISP),RINPL,CWORK,0.0_q,CWORK,GRIDC_)
         CALL FFT3D(CWORK(1),GRIDC_,-1)

         CVTOT=0
         CALL CPB_GRID(GRIDUS,GRIDC_,C_TO_US,CWORK(1),CVTOT(1))
         CALL FFT3D(CVTOT(1),GRIDUS,1)
         GRIDC => GRIDUS
      ELSE
         CVTOT => CVTOT_(:,ISP)
         GRIDC => GRIDC_
      ENDIF
      RINPL=1._q/GRIDC%NPLWV

!=======================================================================
! loop over all ions
!=======================================================================

      ion: DO NI=1,T_INFO%NIONS
      NT=T_INFO%ITYP(NI)
      PP=>PP_POINTER(P, NI, NT)

      ! for this ion (this type of ion) no depletion charge
      IF (PP%PSDMAX==0) CYCLE
!-----------------------------------------------------------------------
! calculate the spherical harmonics YLM and the distance DIST  between
! grid-points and central atom (DEP and POT are work-arrays)
!-----------------------------------------------------------------------
      LYMAX=MAXL1(PP)
      IF ( ASSOCIATED(PP%QPAW) ) THEN
         LYMAX=LYMAX*2
      ENDIF

      CALL SETYLM_AUG(GRIDC,LATT_CUR,T_INFO%POSION(1,NI),PP%PSDMAX,NPSRNL, &
     &        LMYDIM,LYMAX,YLM(1,1),IRDMAX,INDMAX, &
     &        0.0_q, 0.0_q, 0.0_q,DIST(1),NLI(1),XS(1),YS(1),ZS(1))

      IRDMAA=MAX(IRDMAA,INDMAX)

      DO N=1,INDMAX
         POT(N)=CVTOT(NLI(N))
      ENDDO
      IF (ASSOCIATED(PP%QPAW_FOCK)) THEN
      NMAX_FOCKAE=SIZE(PP%QPAW_FOCK,4)
      LMAX_FOCKAE=SIZE(PP%QPAW_FOCK,3)-1
!=======================================================================
! PAW approach
! calculate first the integral int V Y(L,M) Q(L)
! (Q(L) are the L dependent compensation charges in the PAW method)
! around one atom
! then transform to  L,L',M,M' use Clebsch-Gordan coefficients
!=======================================================================
      IF (.NOT.WDES%LSPIRAL .OR. ISP==1 .OR. ISP==4) THEN
! no phase factor for total charge and m_z or if LSPIRAL=.FALSE.
      DLM=0
      ! proper indexing of QDEP_FOCK is nasty see fast_aug.F
      QDEP_FOCK_INDEX=SIZE(PP%QDEP_FOCK,3)-NMAX_FOCKAE*(LMAX_FOCKAE+1)-1

      DO NAE=1,NMAX_FOCKAE
      DO L  =0,LMAX_FOCKAE
         QDEP_FOCK_INDEX=QDEP_FOCK_INDEX+1

         CALL SETDEP(PP%QDEP_FOCK(1,1,QDEP_FOCK_INDEX),PP%PSDMAX,NPSRNL, &
              LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
         DO M=1,(L*2)+1
            INDYLM =L*L  +M
            SUM=0
            SUMN=0
!DIR$ IVDEP
!OCL NOVREC
            DO IND=1,INDMAX
               SUM=SUM+POT(IND)*DEP(IND)*YLM(IND,INDYLM)
               SUMN=SUMN+DEP(IND)*YLM(IND,INDYLM)
            ENDDO
            DLM(INDYLM)=SUM*RINPL
         ENDDO
       !  IF (L<=1) WRITE(0,'("DLM",I2,10F7.4)') L,(1E6*DLM(L**2+M),M=1,(L*2)+1)
      ENDDO
      CALL CALC_DLLMM_FOCK( CTMP(:,:,NI,ISP), DLM, PP, NAE)
      ENDDO

      IF (QDEP_FOCK_INDEX /= SIZE(PP%QDEP_FOCK,3)-1) THEN
         WRITE(*,*) 'internal error in SETDIJ_FOCKAE: size of QDEP_FOCK seems to be incorrect'
         STOP
      ENDIF
      ENDIF
      ENDIF
!=======================================================================
      ENDDO ion
!-----------------------------------------------------------------------
      ENDDO spin

      DEALLOCATE(DIST,DEP,POT,YLM,NLI,XS,YS,ZS)
      IF (LADDITIONAL) DEALLOCATE(CVTOT,CWORK)
! reduce CTMP
#ifdef realmode
      CALLMPI( M_sum_d(GRIDC%COMM, CTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_d(GRIDC%COMM, CTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ*2))
#endif
      ! back to spinor representation
      CALL RL_FLIP(CVTOT_(1,1),GRIDC_,WDES%NCDIJ,.TRUE.)

      ENDIF overl
!-----------------------------------------------------------------------
! now set up CQIJ and add diagonal part to CDIJ
! find blocks with same quantum number L
! the routine is somewhat complicated
! only terms with same quantum numbers L L' and M M' are non-zero
!-----------------------------------------------------------------------
      ion2: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion2
#ifdef MPI
        DO ISP=1,WDES%NCDIJ
           CDIJ(:,:,NIP,ISP)=CTMP(:,:,NI,ISP)
        ENDDO
#endif
        IF (T_INFO%VCA(NT)/=1.0) THEN
           DO ISP=1,WDES%NCDIJ
              CDIJ(:,:,NIP,ISP)=CDIJ(:,:,NIP,ISP)*T_INFO%VCA(NT)
           ENDDO
        ENDIF

      ENDDO ion2

      ! go to spin up and down presentation for CDIJ
      CALL US_FLIP(WDES, LMDIM, CDIJ, .TRUE., .TRUE.)

#ifdef MPI
      DEALLOCATE(CTMP)
#else
#undef CTMP
#endif

      RETURN
    END SUBROUTINE SETDIJ_FOCKAE

