#include "symbol.inc"
!*********************************************************************
!
! this module implements routines to symmetrize the gradient
! this is a by far non trivial operation, since symmetry caused
! degeneracies between the wavefunctions make the symmetrization
! rather involved. 
! Also the distribution over bands makes things more difficult than
! one might expect-
!
! The general algorithm works along the following lines
! Perform a loop over every k-point
!   Determine the "small" point/space group for this k-point
!   i.e. all symmetry operations S that leave the k-point invariant
!   
!   Next apply all of these symmetry operations S to each wavefunction
!   phi and to the gradient and build a new symmetrized gradient
!   for band j according to
!
!   |g(sym)_j> = \sum_i,S   < phi_j| S|phi_i> S|g_i> / n_S
!
!   where S is a symmetry operation and n_S the number of operations
!   and the loop is over all bands i.
!   The matrix < phi | S |phi> is always symmetric but possibly complex.
!   In principle < phi | S |phi> should have only non zero elements
!   for degenerate pairs, but that is not so easy to implement
!   in the parallel version.
!
! caveats:
! ) the routine uses a lot of storage (three times a set for a single
!    kpoint)
! ) it must be started with very well converged wavefunctions 
!   only, since the symmetry of the original set phi is CONSERVED.
!   If phi does not posses the right symmetry, convergence will be
!   fucked up
!
! written by gK 
!
!*********************************************************************

MODULE sym_grad
  USE prec
  USE base
  USE mkpoints
  USE full_kpoints
  USE kpoints_change
  IMPLICIT NONE

  TYPE (skpoints_trans) :: KPOINTS_TRANS_SG

CONTAINS

!*********************************************************************
!
! determine the small space group, which is the subset of
! symmetry operations that leave the supplied k point NK
! invariant
! a structure (kpoints_trans) is returned which contains
! all the required information to rotate orbitals
!
!*********************************************************************

  SUBROUTINE SMALL_SPACE_GROUP( NQ, WDES, NIONS, ROTMAP, MAGROT,  ISYM, IU6)
    USE wave
    USE constant
#ifndef oldsym
    USE spinsym
#endif
    INTEGER NQ           ! preset q-point
! passed structures and variables
    TYPE (wavedes) :: WDES                ! wave function descriptor
    INTEGER ISYM
    INTEGER NIONS
    INTEGER ROTMAP(:,:,:)
    REAL(q) MAGROT(:,:)
    INTEGER :: IU6
! external routine
    LOGICAL,EXTERNAL :: SGREQL
! common symmetry variables
    INTEGER ISYMOP, NROT, IGRPOP, NROTK, INVMAP, NPCELL
    REAL(q) GTRANS, AP
    COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
! local variables
    REAL(q) :: TINY=1E-6, V(3), VR(3), ROP(3,3)
    INTEGER :: INVERS(9)=(/-1,0,0,0,-1,0,0,0,-1/),IOP(3,3)
    INTEGER NOP
    LOGICAL LINV
    INTEGER, ALLOCATABLE :: IGRIDIND(:,:,:)
    INTEGER              :: NI
    INTEGER              :: NKPTS_START
    INTEGER              :: NG1I,NG2I,NG3I,NG1,NG2,NG3
    INTEGER              :: NGX,NGY,NGZ
    REAL(q)              :: PHASE

    INTEGER SMALL_GROUP_OP(48)
    INTEGER ITRANS(3,48)
    LOGICAL SMALL_GROUP_LINV(48)
    INTEGER NOP_SMALL_GROUP

!=======================================================================
! loop over all point group operations and check whether 
! symmetry leaves the k-point invariant
!=======================================================================
    NOP_SMALL_GROUP=0
    VR(1)=WDES%VKPT(1,NQ)
    VR(2)=WDES%VKPT(2,NQ)
    VR(3)=WDES%VKPT(3,NQ)
    LINV=.FALSE.
    DO NOP=1,NROTK
       ! test existence of inversion op
       IF (SGREQL(IGRPOP(1,1,NOP),INVERS)) LINV=.TRUE.
       ! copy symmetry op to real array
       ROP=IGRPOP(:,:,NOP)

       ! generate new k-point S k
       V(1)=VR(1)*ROP(1,1)+VR(2)*ROP(2,1)+VR(3)*ROP(3,1)
       V(2)=VR(1)*ROP(1,2)+VR(2)*ROP(2,2)+VR(3)*ROP(3,2)
       V(3)=VR(1)*ROP(1,3)+VR(2)*ROP(2,3)+VR(3)*ROP(3,3)
       ! compare with original k-point
       IF ( ( ABS(MOD(VR(1)-V(1)+6.5,1._q)-0.5_q)<TINY) .AND. &
            ( ABS(MOD(VR(2)-V(2)+6.5,1._q)-0.5_q)<TINY) .AND. &
            ( ABS(MOD(VR(3)-V(3)+6.5,1._q)-0.5_q)<TINY)) THEN
          NOP_SMALL_GROUP=NOP_SMALL_GROUP+1
          ! round to next integer
          ITRANS(:,NOP_SMALL_GROUP)=CEILING(VR(:)-V(:)-0.5)
          SMALL_GROUP_OP (NOP_SMALL_GROUP)=NOP
          SMALL_GROUP_LINV(NOP_SMALL_GROUP)=.FALSE.
       ENDIF
    END DO
    ! same now applying inversion symmetry
#ifndef oldsym
    IF (.NOT. LINV .AND. ISYM>=0 .AND. .NOT.WDES%LNONCOLLINEAR) THEN
#else
    IF (.NOT. LINV .AND. ISYM>=0) THEN
#endif
       DO NOP=1,NROTK
          ! apply inversion symmetry to form to get IOP
          CALL SGRPRD(INVERS,IGRPOP(1,1,NOP),IOP(1,1))
          ! copy symmetry op to real array
          ROP=IOP
          V(1)=VR(1)*ROP(1,1)+VR(2)*ROP(2,1)+VR(3)*ROP(3,1)
          V(2)=VR(1)*ROP(1,2)+VR(2)*ROP(2,2)+VR(3)*ROP(3,2)
          V(3)=VR(1)*ROP(1,3)+VR(2)*ROP(2,3)+VR(3)*ROP(3,3)
          ! compare with original k-point
          IF ( ( ABS(MOD(VR(1)-V(1)+6.5,1._q)-0.5_q)<TINY) .AND. &
               ( ABS(MOD(VR(2)-V(2)+6.5,1._q)-0.5_q)<TINY) .AND. &
               ( ABS(MOD(VR(3)-V(3)+6.5,1._q)-0.5_q)<TINY)) THEN
             NOP_SMALL_GROUP=NOP_SMALL_GROUP+1
             ITRANS(:,NOP_SMALL_GROUP)=CEILING(VR(:)-V(:)-0.5)
             SMALL_GROUP_OP (NOP_SMALL_GROUP)=NOP
             SMALL_GROUP_LINV(NOP_SMALL_GROUP)=.TRUE.
          ENDIF
       END DO
    ENDIF
    IF (IU6>=0) WRITE(*,*) 'NQ=',NQ,' operations',NOP_SMALL_GROUP
!=======================================================================
! now create required structures to rotate orbitals
!=======================================================================

    NGX=WDES%GRID%NGX
    NGY=WDES%GRID%NGY
    NGZ=WDES%GRID%NGZ

    IF (ASSOCIATED(KPOINTS_TRANS_SG%NK_OLD)) THEN
       CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS_SG)
    ENDIF
    
    ! allocate 3d index array and KPOINTS_TRANS_SG structure
    ALLOCATE(IGRIDIND(NGX,NGY,NGZ),&
         KPOINTS_TRANS_SG%CPHASE (WDES%NGDIM,NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%NINDPW(WDES%NGDIM,NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%NK_OLD(NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%ISYMOP(3,3,NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%LSHIFT(NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%LSHIFT_G(NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%LINV(NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%ROTMAP(NIONS,NOP_SMALL_GROUP), &
         KPOINTS_TRANS_SG%SPINFLIP(NOP_SMALL_GROUP))
#ifndef oldsym
    ALLOCATE(KPOINTS_TRANS_SG%RSSYMOP(2,2,NOP_SMALL_GROUP))
#endif
    KPOINTS_TRANS_SG%CPHASE=(1._q,0._q)
    KPOINTS_TRANS_SG%LSHIFT=.FALSE.
          

    ! first we set up an 3d-array, that stores the index into
    ! the compact plane wave array
    IGRIDIND=0
    DO NI=1,WDES%NGVECTOR(NQ)
       NG1=WDES%IGX(NI,NQ)
       NG2=WDES%IGY(NI,NQ)
       NG3=WDES%IGZ(NI,NQ)
       IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))=NI
    ENDDO

!loop over all entries in kpoints_trans
    DO NOP=1, NOP_SMALL_GROUP

       KPOINTS_TRANS_SG%NK_OLD(NOP)=NQ  ! both k-points are equivalent
       KPOINTS_TRANS_SG%LINV(NOP)  =SMALL_GROUP_LINV(NOP)
       KPOINTS_TRANS_SG%ISYMOP(:,:,NOP)=ISYMOP(:,:,SMALL_GROUP_OP(NOP))
       KPOINTS_TRANS_SG%ROTMAP(:,NOP)  =ROTMAP(:,SMALL_GROUP_OP(NOP),1)
       KPOINTS_TRANS_SG%SPINFLIP(NOP)  =0
#ifndef oldsym
       KPOINTS_TRANS_SG%RSSYMOP(:,:,NOP)=RSSYMOP(:,:,SMALL_GROUP_OP(NOP))
#endif

       IF ( SIZE(MAGROT)>1) THEN
          IF( MAGROT(SMALL_GROUP_OP(NOP),1)==-1) THEN
             KPOINTS_TRANS_SG%SPINFLIP(NOP)  =1
             WRITE(*,*) 'SMALL_SPACE_GROUP: spinflip detected (yet untested)'
          ENDIF
       ENDIF
       IF (SMALL_GROUP_LINV(NOP)) THEN
          CALL SGRPRD(INVERS,IGRPOP(1,1,SMALL_GROUP_OP(NOP)),IOP(1,1))
          ROP=IOP
       ELSE
          ROP=IGRPOP(:,:,SMALL_GROUP_OP(NOP))
       ENDIF

       ! phase shift (required for space group operations)
       IF ((ABS(GTRANS(1,SMALL_GROUP_OP(NOP)))>TINY) .OR. &
           (ABS(GTRANS(2,SMALL_GROUP_OP(NOP)))>TINY) .OR. &
           (ABS(GTRANS(3,SMALL_GROUP_OP(NOP)))>TINY)) THEN
          KPOINTS_TRANS_SG%LSHIFT(NOP)=.TRUE.
       ELSE
          KPOINTS_TRANS_SG%LSHIFT(NOP)=.FALSE.
       ENDIF
       IF (ITRANS(1,NOP)/=0 .OR. ITRANS(2,NOP)/=0 .OR. ITRANS(3,NOP)/=0) THEN
          KPOINTS_TRANS_SG%LSHIFT_G(NOP)=.TRUE.
       ELSE
          KPOINTS_TRANS_SG%LSHIFT_G(NOP)=.FALSE.
       ENDIF
       !  loop over all G vetors
       DO NI=1,WDES%NGVECTOR(NQ)
          NG1I=WDES%IGX(NI,NQ)
          NG2I=WDES%IGY(NI,NQ)
          NG3I=WDES%IGZ(NI,NQ)
          ! generate S G
          NG1=NG1I*ROP(1,1)+NG2I*ROP(2,1)+NG3I*ROP(3,1)-ITRANS(1,NOP)
          NG2=NG1I*ROP(1,2)+NG2I*ROP(2,2)+NG3I*ROP(3,2)-ITRANS(2,NOP)
          NG3=NG1I*ROP(1,3)+NG2I*ROP(2,3)+NG3I*ROP(3,3)-ITRANS(3,NOP)
          
          ! KPOINTS_TRANS_SG%NINDPW(NI,NOP) stores the index
          ! of G' = S G, where S is the symmetry operation leaving the k point invariant
          KPOINTS_TRANS_SG%NINDPW(NI,NOP)=IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))
          IF (KPOINTS_TRANS_SG%NINDPW(NI,NOP)==0) THEN
1234         FORMAT("internal error in SMALL_SPACE_GROUP: G vector not found ",12I5)
             WRITE(*,1234) NQ,NOP,NI,NG1I,NG2I,NG3I,NG1,NG2,NG3,MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ)
             WRITE(*,'(3F14.7)') ROP
             STOP
          ENDIF
            
          ! calculate phase shift related to primitive translations
          IF ( KPOINTS_TRANS_SG%LSHIFT(NOP)) THEN
             PHASE=-TPI*(REAL(NG1,KIND=q)*GTRANS(1,SMALL_GROUP_OP(NOP))+ &
                         REAL(NG2,KIND=q)*GTRANS(2,SMALL_GROUP_OP(NOP))+ &
                         REAL(NG3,KIND=q)*GTRANS(3,SMALL_GROUP_OP(NOP)))
             KPOINTS_TRANS_SG%CPHASE(NI,NOP)=CMPLX(COS(PHASE),SIN(PHASE),KIND=q)
          ENDIF
       ENDDO
    ENDDO
    DEALLOCATE(IGRIDIND)
  END SUBROUTINE SMALL_SPACE_GROUP


!*********************************************************************
!
! apply all symmetry operations that leave the k-point invariant
! to a set of orbitals and project onto original set
! reset the gradient such that it has the same symmetry
! as the supplied wavefunctions W
! this is a rather involved operation and its use is only
! recommended for Hamiltonians that are expensive to evaluate
! 
!*********************************************************************


  SUBROUTINE APPLY_SMALL_SPACE_GROUP_OP( W, WACC, NONLR_S, NONL_S, & 
       P, NIONS, LATT_CUR, SYMM, CQIJ, LPROJ, IU6, NKSTART)
    USE dfast
    USE wave_high
    USE nonl_high
    USE pseudo
    USE lattice
#ifndef oldsym
    USE spinsym
#endif
    TYPE (wavespin) :: W, WACC
    TYPE (nonlr_struct)NONLR_S      ! descriptor for non local part of PP (real space)
    TYPE (nonl_struct) NONL_S       ! descriptor for non local part of PP (reciprocal space)
    TYPE (potcar)   :: P(:)
    INTEGER         :: NIONS
    TYPE (latt)     :: LATT_CUR
    TYPE (symmetry) ::   SYMM      
    OVERLAP, OPTIONAL :: CQIJ(:,:,:,:)
    INTEGER :: IU6
    LOGICAL :: LPROJ                ! recalculate CPROJ for gradient search direction as well
    INTEGER, OPTIONAL :: NKSTART
  ! local
    TYPE (wavedes1) :: WDES1
    TYPE (wavefun1) :: W_NEW
    TYPE( rotation_handle), POINTER :: ROT_HANDLE
    INTEGER :: ISP, NB, NK, NOP, ISPINOR, ISP_OLD
    INTEGER :: NIS, NPRO, NT, NI, NIP, NPROP, ITER
    INTEGER :: MY_NKSTART
  ! 
    TYPE (wavefuna)    WS           ! wavefunction or acceleration after application of symmetry
    TYPE (wavefuna)    WA           ! original wavefunction  
    TYPE (wavefuna)    WACC_NEW     ! new accelerations
    INTEGER :: NPOS, NSTRIP, NBANDS, NB_TOT
    GDEF    :: CHAM(W%WDES%NB_TOT,W%WDES%NB_TOT)
    GDEF    :: SCALE

    GDEF :: S
    REAL(q) :: SUM

! presently gamma only version is not supported
    IF (W%WDES%LGAMMA .OR. SYMM%ISYM <=0) RETURN

    ! NULLIFY to ensure IF(ASSOCIATED()) statements later on will work
    NULLIFY (ROT_HANDLE)
    NULLIFY (KPOINTS_TRANS_SG%NK_OLD)

    NK=WDES1%NK
    NBANDS=W%WDES%NBANDS
    NB_TOT=W%WDES%NB_TOT

    CALL SETWDES(W%WDES, WDES1, 0)

    CALL NEWWAV(W_NEW , WDES1, .TRUE.)
    CALL NEWWAVA(WA, WDES1, NBANDS)
    CALL NEWWAVA(WS, WDES1, NBANDS)
    CALL NEWWAVA(WACC_NEW, WDES1, NBANDS)

    MY_NKSTART=1
    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ENDIF

!=======================================================================
! loop over all spins k-points and symmetry operations
!=======================================================================
spn: DO ISP=1, W%WDES%ISPIN
kpt: DO NK=MY_NKSTART, W%WDES%NKPTS
    CALL SETWDES(W%WDES, WDES1, NK)

#ifdef MPI
    IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
    ! determine symmetry operations for small point/space group
    CALL SMALL_SPACE_GROUP( NK, W%WDES, NIONS, SYMM%ROTMAP, SYMM%MAGROT, SYMM%ISYM, IU6)

    ! set up phase factors
    IF (NONLR_S%LREAL) THEN
       CALL PHASER(W%WDES%GRID,LATT_CUR,NONLR_S,NK,W%WDES)
    ELSE
       CALL PHASE(W%WDES,NONL_S,NK)
    ENDIF

    WACC_NEW%CW=0
    WACC_NEW%CPROJ=0

    DO NOP=1,SIZE(KPOINTS_TRANS_SG%NK_OLD)
       ISP_OLD=ISP
       IF (KPOINTS_TRANS_SG%SPINFLIP(NOP)==1) THEN
          ISP_OLD=3-ISP
       ENDIF

       CALL GENERATE_ROT_HANDLE(ROT_HANDLE, KPOINTS_TRANS_SG, P, LATT_CUR, NOP, WDES1 )
!=======================================================================
! apply symmetry operation to all bands at this k-point
! and collect result in WS
!=======================================================================
          DO NB=1, NBANDS

             DO ISPINOR=0,WDES1%NRSPINORS-1
                CALL ROTATE_WAVE(WDES1%NGVECTOR, &
                     W_NEW%CW(1+ISPINOR*WDES1%NGVECTOR), & 
                     W%CW(1+ISPINOR*WDES1%NGVECTOR,NB,NK,ISP_OLD), &
                     KPOINTS_TRANS_SG%CPHASE(1,NOP), KPOINTS_TRANS_SG%NINDPW(1,NOP),  &
                     KPOINTS_TRANS_SG%LINV(NOP), KPOINTS_TRANS_SG%LSHIFT(NOP))

                NIS=1
                NPRO =ISPINOR *(WDES1%NPRO/2)+1
                DO NT=1, WDES1%NTYP
                   DO NI=NIS, WDES1%NITYP(NT)+NIS-1
                      NIP=KPOINTS_TRANS_SG%ROTMAP(NI,NOP)
                      NPROP=ROT_HANDLE%NPRO_NI(NIP)+ISPINOR *(WDES1%NPRO/2)
                      CALL ROTATE_VECTOR( KPOINTS_TRANS_SG%LINV(NOP),W%CPROJ(NPROP,NB,NK,ISP_OLD),& 
                           W_NEW%CPROJ(NPRO),ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,ROT_HANDLE%SL(1,1,0),P(NT))
                      NPRO= WDES1%LMMAX(NT)+NPRO
                   ENDDO
                   NIS=NIS+WDES1%NITYP(NT)
                ENDDO
             ENDDO
#ifndef oldsym
             CALL ROTATE_WAVE_SPIN_RECIP(W%WDES, KPOINTS_TRANS_SG%RSSYMOP(:,:,NOP), NK, KPOINTS_TRANS_SG%SPINFLIP(NOP), W_NEW%CW)
             CALL ROTATE_WAVE_CHARACTER_SPIN(W_NEW%WDES1, W_NEW%CPROJ, KPOINTS_TRANS_SG%RSSYMOP(:,:,NOP))
#endif
             ! ok here I still have a bug
             ! if the rotated wavefunction is shifted by a reciprocal lattice vector
             ! the projected wave function characters are wrong
             ! need to correct them by simply recalculating them
             IF (KPOINTS_TRANS_SG%LSHIFT_G(NOP)) THEN
                IF (NONLR_S%LREAL) THEN
                   CALL FFTWAV_W1(W_NEW)
                   CALL RPRO1(NONLR_S, WDES1, W_NEW)
                ELSE
                   CALL PROJ1(NONL_S, WDES1, W_NEW)
                ENDIF
             ENDIF
             WS%CPROJ(:,NB)=W_NEW%CPROJ
             WS%CW   (:,NB)=W_NEW%CW
          ENDDO
!=======================================================================
! now calculate inproduct between original set and symmetry
! transformed set
!=======================================================================
          ! copy original wavefunctions
          CALL WA_COPY(ELEMENTS(W, WDES1, ISP), WA)
          ! Q times wavefunction character
          CALL OVERL(WDES1, WDES1%LOVERL,SIZE(CQIJ,1),CQIJ(1,1,1,ISP),  W%CPROJ(1,1,NK,ISP), WA%CPROJ(1,1))
          ! redistribute everything
          IF (W%WDES%DO_REDIS) THEN
             CALL REDIS_PW  (WDES1, NBANDS, WA%CW   (1,1))
             CALL REDIS_PROJ(WDES1, NBANDS, WA%CPROJ(1,1))
             CALL REDIS_PW  (WDES1, NBANDS, WS%CW   (1,1))
             CALL REDIS_PROJ(WDES1, NBANDS, WS%CPROJ(1,1))
          ENDIF
          
          CHAM=0
          NSTRIP=NSTRIP_STANDARD_GLOBAL
          DO NPOS=1, NB_TOT-NSTRIP,NSTRIP
             CALL ORTH2( &
                  WS%CW(1,1)   ,WA%CW_RED(1,NPOS), & 
                  WS%CPROJ(1,1),WA%CPROJ_RED(1,NPOS), NB_TOT, &
                  NPOS,NSTRIP, & 
                  WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1))
          ENDDO

          CALL ORTH2( &
               WS%CW(1,1),   WA%CW_RED(1,NPOS), & 
               WS%CPROJ(1,1),WA%CPROJ_RED(1,NPOS), NB_TOT, &
               NPOS,NB_TOT-NPOS+1, & 
               WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1))
          
          CALLMPI( M_sum_g(W%WDES%COMM_INTER,CHAM(1,1),NB_TOT*NB_TOT))
!=======================================================================
! apply symmetry operation to all accelerations at this k-point
! and collect result in WS
!=======================================================================
          DO NB=1, NBANDS
             
             DO ISPINOR=0,WDES1%NRSPINORS-1

                CALL ROTATE_WAVE(WDES1%NGVECTOR, &
                     W_NEW%CW(1+ISPINOR*WDES1%NGVECTOR), & 
                     WACC%CW(1+ISPINOR*WDES1%NGVECTOR,NB,NK,ISP_OLD), &
                     KPOINTS_TRANS_SG%CPHASE(1,NOP), KPOINTS_TRANS_SG%NINDPW(1,NOP),  &
                     KPOINTS_TRANS_SG%LINV(NOP), KPOINTS_TRANS_SG%LSHIFT(NOP))

                IF (LPROJ) THEN
                NIS=1
                NPRO =ISPINOR *(WDES1%NPRO/2)+1
                DO NT=1, WDES1%NTYP
                   DO NI=NIS, WDES1%NITYP(NT)+NIS-1
                      NIP=KPOINTS_TRANS_SG%ROTMAP(NI,NOP)
                      NPROP=ROT_HANDLE%NPRO_NI(NIP)+ISPINOR *(WDES1%NPRO/2)
                      CALL ROTATE_VECTOR( KPOINTS_TRANS_SG%LINV(NOP),WACC%CPROJ(NPROP,NB,NK,ISP_OLD),& 
                           W_NEW%CPROJ(NPRO),ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,ROT_HANDLE%SL(1,1,0),P(NT))
                      NPRO= WDES1%LMMAX(NT)+NPRO
                   ENDDO
                   NIS=NIS+WDES1%NITYP(NT)
                ENDDO
                ENDIF
             ENDDO
#ifndef oldsym
             CALL ROTATE_WAVE_SPIN_RECIP(W%WDES, KPOINTS_TRANS_SG%RSSYMOP(:,:,NOP), NK, KPOINTS_TRANS_SG%SPINFLIP(NOP), W_NEW%CW)
             IF (LPROJ) CALL ROTATE_WAVE_CHARACTER_SPIN(W_NEW%WDES1, W_NEW%CPROJ, KPOINTS_TRANS_SG%RSSYMOP(:,:,NOP))
#endif
             ! ok here I still have a bug
             ! if the rotated wavefunction is shifted by a reciprocal lattice vector
             ! the projected wave function characters are wrong
             ! need to correct them by simply recalculating them
             IF (LPROJ) THEN
             IF (KPOINTS_TRANS_SG%LSHIFT_G(NOP)) THEN
                IF (NONLR_S%LREAL) THEN
                   CALL FFTWAV_W1(W_NEW)
                   CALL RPRO1(NONLR_S, WDES1, W_NEW)
                ELSE
                   CALL PROJ1(NONL_S, WDES1, W_NEW)
                ENDIF
             ENDIF
             ENDIF

             IF (LPROJ) WS%CPROJ(:,NB)=W_NEW%CPROJ
             WS%CW   (:,NB)=W_NEW%CW
          ENDDO
!=======================================================================
! add appropriate linear combination
!=======================================================================
          ! redistribute everything
          IF (W%WDES%DO_REDIS) THEN
             CALL REDIS_PW  (WDES1, NBANDS, WS%CW   (1,1))
             IF (LPROJ) CALL REDIS_PROJ(WDES1, NBANDS, WS%CPROJ(1,1))
          ENDIF
          ! divide by number of symmetry operations
          SCALE=1./SIZE(KPOINTS_TRANS_SG%NK_OLD)
          IF (WDES1%NPL_RED/=0) &
               CALL GGEMM('N', 'N', m_ WDES1%NPL_RED, NB_TOT, NB_TOT, SCALE, &
               &               WS%CW(1,1), m_ WDES1%NRPLWV_RED, CHAM(1,1), &
               &               NB_TOT, one, WACC_NEW%CW(1,1), m_ WDES1%NRPLWV_RED)
          IF (WDES1%NPRO_RED/=0 .AND. LPROJ ) &
               CALL GGEMM('N', 'N', WDES1%NPRO_RED, NB_TOT, NB_TOT, SCALE, &
               &               WS%CPROJ(1,1), WDES1%NPROD_RED, CHAM(1,1), &
               &               NB_TOT, one, WACC_NEW%CPROJ(1,1), WDES1%NPROD_RED)
    ENDDO
    ! redistribute back
    IF (W%WDES%DO_REDIS) THEN
       CALL REDIS_PW  (WDES1, NBANDS, WACC_NEW%CW(1,1))
       IF (LPROJ) CALL REDIS_PROJ(WDES1, NBANDS, WACC_NEW%CPROJ(1,1))
    ENDIF

!    WRITE(*,'(8F14.7)') WACC_NEW%CW(1:4,2), WACC%CW(1:4,2,NK,ISP)
!    WRITE(*,'(8F14.7)') WACC_NEW%CPROJ(1:4,2), WACC%CPROJ(1:4,2,NK,ISP)
    ! copy back to original accelerations
    CALL WA_COPY(WACC_NEW, ELEMENTS(WACC, WDES1, ISP))
    ENDDO kpt
    ENDDO spn

    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
    CALL DELWAV(W_NEW,.TRUE.)
    CALL DELWAVA(WA)
    CALL DELWAVA(WS)
    CALL DELWAVA(WACC_NEW)
    
    IF (ASSOCIATED(KPOINTS_TRANS_SG%NK_OLD)) THEN
       CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS_SG)
    ENDIF

    CALL KPAR_SYNC_WAVEFUNCTIONS(W%WDES,WACC)
    
  END SUBROUTINE APPLY_SMALL_SPACE_GROUP_OP

END MODULE sym_grad
