#include "symbol.inc"
!*********************************************************************
!
! this module implements all routines required to go from
! one k-points mesh to another k-point mesh 
! it is required to allow VASP to dynamically reallocate the
! wavefunction array when the number of k-points is changed
!
!*********************************************************************

MODULE kpoints_change
  USE prec
  USE vaspxml
  USE mkpoints
  USE full_kpoints
  IMPLICIT NONE

! pointer to the k-points generated initially
! (it is possible the k-points change during a run, if the
!  e.g. the symmetry is lowered)
  TYPE (kpoints_struct),SAVE,POINTER :: KPOINTS_ORIG

! pointer to the full k-point grid generated initially
  TYPE (skpoints_full),SAVE,POINTER :: KPOINTS_FULL_ORIG


!
!  structure to store all quantities that are required to go
!  from one k-point grid in the IRZ to another one 
!  i.e. if symmetry is reduced
!
  TYPE skpoints_trans
     INTEGER NKPTS                   ! not used
     LOGICAL,POINTER   :: LINV(:)    ! mirror point?
     LOGICAL,POINTER   :: LSHIFT(:)  ! phase shift required
     LOGICAL,POINTER   :: LSHIFT_G(:)! G vectors shifted
     COMPLEX(q),POINTER:: CPHASE(:,:)! phase shift for PW components 
     INTEGER,POINTER   :: NK_OLD(:)  ! index of original k-point in the IRZ
     INTEGER,POINTER   :: NINDPW(:,:)! index of plane wave component after application of symmetry
     INTEGER,POINTER  ::ISYMOP(:,:,:)! rotation matrix from k-point in the IRZ to full k-point
     INTEGER,POINTER   :: ROTMAP(:,:)! map indexing atoms that are taken into each other when 
                                     ! the symmetry operation is applied
     INTEGER,POINTER   :: SPINFLIP(:)! spin flip required
#ifndef oldsym
     COMPLEX(q),POINTER :: RSSYMOP(:,:,:) ! Spin rotation matrix in real space
#endif
  END TYPE skpoints_trans

CONTAINS

!*********************************************************************
!
! set the KPOINTS_ORIG and KPOINTS_FULL_ORIG pointer properly
! (called only once during initialisation)
!
!*********************************************************************

  SUBROUTINE SETUP_ORIG_KPOINTS

    KPOINTS_ORIG=>KPOINTS_

    IF (LFULL_KPOINTS) THEN
       KPOINTS_FULL_ORIG=>KPOINTS_FULL
    ENDIF

  END SUBROUTINE SETUP_ORIG_KPOINTS

!*********************************************************************
!
!  RE_READ_KPOINTS rereads the k-points structure from the KPOINTS
!  file and generates a new set of k-points (KPOINTS)
!  and reorders the k-points in KPOINTS
!  such that the first KPOINTS_ORIG%NKPTS k-points in KPOINTS
!  are identical to KPOINTS_ORIG
!  furthermore the static entries KPOINTS_ and KPOINTS_FULL are
!  reset to the new setup
!  it is also possible to add new k-points to the list 
!  these are supplied in VKPT_ADD
!
!*********************************************************************

  SUBROUTINE RE_READ_KPOINTS(KPOINTS, LATT_CUR, LINVERSION, & 
      NIONS, ROTMAP, MAGROT, ISYM, IU6, IU0, VKPT_ADD)

    USE prec
    USE lattice
    USE mkpoints
    USE full_kpoints
    IMPLICIT NONE
    
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt)        LATT_CUR
    LOGICAL LINVERSION
    INTEGER NIONS
    INTEGER ROTMAP(:,:,:)
    REAL(q) MAGROT(:,:)
    INTEGER ISYM
   
    INTEGER IU0,IU6 ! units for output
    REAL(q), OPTIONAL :: VKPT_ADD(:,:)
  ! local
    TYPE (kpoints_struct) KPOINTS_RD
    INTEGER  NKPTS, NKPTS_RD, NT, NK, NKP, NK_FULL, N, I
    INTEGER, PARAMETER :: MAGIC=-1234
    REAL(q) :: EMIN, EMAX

    IF(.NOT. ASSOCIATED(KPOINTS_FULL_ORIG)) THEN
       IF (IU0>=0) WRITE(IU0,*) 'internal error in RE_READ_KPOINTS: KPOINTS_FULL_ORIG not associated'
       IF (IU0>=0) WRITE(IU0,*) '        call routine USE_FULL_KPOINTS first'
       STOP
    ENDIF
!=======================================================================
! generate new k-point list
!=======================================================================
    ! first copy (some entries are read from INCAR and we need
    ! to set them up properly)
    KPOINTS_RD=KPOINTS_ORIG
    NULLIFY(KPOINTS_RD%VKPT,KPOINTS_RD%WTKPT,KPOINTS_RD%IDTET)

    CALL RD_KPOINTS(KPOINTS_RD,LATT_CUR,LINVERSION,ISYM<0,IU6,IU0)
#ifdef debug
    DO N=1,KPOINTS_RD%NTET
       WRITE(*,'(1I5,"   ",6I5)') N,KPOINTS_RD%IDTET(:,N)
    ENDDO
#endif


    NKPTS_RD=KPOINTS_RD%NKPTS
    NT=MAX(KPOINTS_RD%NTET,1)

    ! copy all entries in from KPOINTS_RD to KPOINTS (except for EMIN and EMAX)
    EMIN=KPOINTS%EMIN
    EMAX=KPOINTS%EMAX
    KPOINTS=KPOINTS_RD
    KPOINTS%EMIN=EMIN
    KPOINTS%EMAX=EMAX
      

    ! increase k-point number by the number of additional k-points
    NKPTS=KPOINTS_RD%NKPTS
    IF (PRESENT(VKPT_ADD)) THEN
       NKPTS        =NKPTS        +SIZE(VKPT_ADD,2)
       KPOINTS%NKDIM=KPOINTS%NKDIM+SIZE(VKPT_ADD,2)
       KPOINTS%NKPTS=KPOINTS%NKPTS+SIZE(VKPT_ADD,2)
    ENDIF

    NULLIFY(KPOINTS%VKPT,KPOINTS%WTKPT,KPOINTS%IDTET)
    ALLOCATE(KPOINTS%VKPT(3,NKPTS),KPOINTS%WTKPT(NKPTS),KPOINTS%IDTET(0:4,NT))

    ! copy the tetrahedron weights from KPOINTS to KPOINTS
    KPOINTS%IDTET(0,:)=KPOINTS_RD%IDTET(0,:)
!=======================================================================
! search for old k-points in the new k-points structure
! and store them in the correct order
!=======================================================================
    DO NK=1,KPOINTS_ORIG%NKPTS
       DO NKP=1,NKPTS_RD
          IF (LIDENTICAL_KPOINT(KPOINTS_ORIG%VKPT(:,NK),KPOINTS_RD%VKPT(:,NKP))) THEN
             EXIT
          ENDIF
       ENDDO
       IF (NKP>NKPTS_RD) THEN
          WRITE(0,*) 'internal ERROR in RE_READ_KPOINTS_RD: the new k-point set for the reduced symmetry case'
          WRITE(0,*) '   does not contain all original k-points. Try to switch off symmetry'
          STOP
       ENDIF
       ! store k-point in KPOINTS structure
#ifdef debug
       WRITE(*,'(A,2I5,6F9.5)')  &
            'k-point restored',NK,NKP,KPOINTS_ORIG%VKPT(:,NK),KPOINTS_RD%VKPT(:,NKP)
#endif
       ! reuse old k-point, since the new one might be shifted by a unit vector
       KPOINTS%VKPT(:,NK)=KPOINTS_ORIG%VKPT(:,NK)
       KPOINTS%WTKPT(NK) =KPOINTS_RD%WTKPT(NKP)
       ! clear this k-point from the list
       KPOINTS_RD%WTKPT(NKP)=MAGIC

       ! update tetrahedron list
       DO I=1,4
          DO N=1,NT
             IF (KPOINTS_RD%IDTET(I,N)==NKP) THEN
                KPOINTS%IDTET(I,N)=NK
                KPOINTS_RD%IDTET(I,N)=0
             ENDIF
          ENDDO
       ENDDO
    ENDDO
!=======================================================================
! now search for remaining new k-points in the full k-points list 
! and add them to the new kpoint list
! the weights are used to identify k-points that have been inserted
! into the new list
!=======================================================================

    NKP=0
    DO NK=KPOINTS_ORIG%NKPTS+1, NKPTS_RD
       ! search k-points that have not yet been stored in the KPOINTS structure
       DO NKP=NKP+1,NKPTS_RD
          IF (KPOINTS_RD%WTKPT(NKP).NE.MAGIC) EXIT
       ENDDO
       IF (NKP>NKPTS_RD) THEN
          WRITE(0,*)'internal ERROR in RE_READ_KPOINTS: KPOINTS list is empty'
          STOP
       ENDIF
       NK_FULL=KPOINT_IN_FULL_GRID(KPOINTS_RD%VKPT(:,NKP),KPOINTS_FULL_ORIG)

       ! store k-point in KPOINTS structure
#ifdef debug
       WRITE(*,'(A,2I5,6F9.5)')  &
            'k-point restored',NK,NKP,KPOINTS_RD%VKPT(:,NKP)
#endif
       ! use k-point from the full list to avoid periodic boundary problems
       KPOINTS%VKPT(:,NK)=KPOINTS_FULL_ORIG%VKPT(:,NK_FULL)
       KPOINTS%WTKPT(NK) =KPOINTS_RD%WTKPT(NKP)
       KPOINTS_RD%WTKPT(NKP)=MAGIC

       ! update tetrahedron list
       DO I=1,4
          DO N=1,KPOINTS_RD%NTET
             IF (KPOINTS_RD%IDTET(I,N)==NKP) THEN
                KPOINTS%IDTET(I,N)=NK
                KPOINTS_RD%IDTET(I,N)=0
             ENDIF
          ENDDO
       ENDDO
    ENDDO 
!=======================================================================
! consistency checks and
! copy some entries from KPOINTS_ORIG to KPOINTS
!=======================================================================
    DO N=1,KPOINTS_RD%NTET
       DO I=1,4
          IF (KPOINTS_RD%IDTET(I,N).NE.0) THEN
             WRITE(0,*)'internal ERROR in RE_READ_KPOINTS: tetrahedron list screwed up!',I,N,KPOINTS_RD%IDTET(I,N)
             STOP
          ENDIF
       ENDDO
    ENDDO

    ! deallocate the temporary KPOINTS structure
    DEALLOCATE(KPOINTS_RD%VKPT,KPOINTS_RD%WTKPT,KPOINTS_RD%IDTET)
!=======================================================================
! finally add additional k-points
!=======================================================================
    DO NK=NKPTS_RD+1,NKPTS
       KPOINTS%VKPT(:,NK)=VKPT_ADD(:,NK-NKPTS_RD)
       KPOINTS%WTKPT(NK)=0
    ENDDO
#ifdef debug
    DO NK=1,KPOINTS_ORIG%NKPTS
       WRITE(*,'(1I5,8F9.5)') NK,KPOINTS_ORIG%VKPT(:,NK),KPOINTS%VKPT(:,NK), &
            KPOINTS_ORIG%WTKPT(NK),KPOINTS%WTKPT(NK)
    ENDDO
    DO NK=KPOINTS_ORIG%NKPTS+1,KPOINTS%NKPTS
       WRITE(*,'(1I5,8F9.5)') NK,KPOINTS%VKPT(:,NK),KPOINTS%WTKPT(NK)
    ENDDO
    DO N=1,KPOINTS%NTET
       WRITE(*,'(1I5,"   ",6I5)') N,KPOINTS%IDTET(:,N)
    ENDDO
#endif
!=======================================================================
! now the delicate part
! we need to reset the static KPOINTS_ and the static KPOINTS_FULL
! structures
! it is however possible to free the respective structures,
! if and only if they are not equivalent to the KPOINTS_ORIG and
! KPOINTS_FULL_ORIG structure (this is the case only from the second
! time this routine is called)
! a clever garbage collector could do this but I am not sure about
! the F90 capabilities in this respect 
!=======================================================================
    IF (.NOT.ASSOCIATED(KPOINTS_,KPOINTS_ORIG)) THEN
       CALL DEALLOC_KPOINTS_STATIC
    ENDIF
    IF (.NOT.ASSOCIATED(KPOINTS_FULL,KPOINTS_FULL_ORIG)) THEN
       CALL DEALLOC_FULL_KPOINTS
    ENDIF
! set the KPOINTS_ structure 
    CALL  SETUP_KPOINTS_STATIC(KPOINTS)
! at this point we can recalculate the full set of kpoints
    CALL SETUP_FULL_KPOINTS(KPOINTS,LATT_CUR, NIONS, ROTMAP, MAGROT, ISYM, LINVERSION, -1,IU0)
    IF (KPOINTS_FULL%NKPTS_NON_ZERO  /= KPOINTS_FULL_ORIG%NKPTS_NON_ZERO) THEN
       IF (IU0>=0) THEN
          WRITE(IU0,*) 'internal ERROR in RE_READ_KPOINTS: the total number of non zero k-points in the full Brillouine zone has changed',KPOINTS_FULL%NKPTS_NON_ZERO, KPOINTS_FULL_ORIG%NKPTS_NON_ZERO
          WRITE(IU0,*) 'trying to continue but this does not look good at all'
       ENDIF
    ENDIF

  END SUBROUTINE RE_READ_KPOINTS


!*********************************************************************
!
! RE_GEN_LAYOUT regenerates the data layout and checks whether 
! the data layout is identical to the original one
!
!*********************************************************************
  
  SUBROUTINE RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_INI, IU6, IU0)
    USE prec
    USE mgrid
    USE wave
    USE constant
    USE base
    USE mkpoints
    USE full_kpoints
    USE fock
    USE lattice

    TYPE (grid_3d), TARGET :: GRID
    TYPE (wavedes)     WDES_NEW
    TYPE (wavedes)     WDES
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt)        LATT_CUR
    TYPE (latt)        LATT_INI
    INTEGER IU6, IU0
  ! local
    TYPE (grid_3d)     GRID_NEW

! Gamma-only version: quick return
#ifdef gammareal
    RETURN
#endif

  ! first check that we use over band distribution only
  ! if not, full stop now
#ifdef MPI
    IF (WDES%COMM_INB%NCPU /= 1) THEN
         CALL VTUTOR('E','NPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
         IU6,3)
         CALL VTUTOR('S','NPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
         IU0,3)
    ENDIF
#endif
  ! copy the loop counters and dimensions from GRID to GRID_NEW
    GRID_NEW=GRID
  ! copy all date from WDES to WDES_NEW
    WDES_NEW=WDES
    CALL INIT_KPOINT_WDES(WDES_NEW, KPOINTS)

  ! now reset NKDIM and NKPTS_FOR_GEN_LAYOUT
    CALL SET_FULL_KPOINTS(WDES_NEW%NKPTS_FOR_GEN_LAYOUT,WDES_NEW%VKPT)
    CALL SET_FOCK_KPOINTS(WDES_NEW%NKDIM)

    CALL GEN_LAYOUT(GRID_NEW, WDES_NEW, LATT_CUR%B, LATT_INI%B, IU6,.TRUE.)
    CALL GEN_INDEX (GRID_NEW, WDES_NEW, LATT_CUR%B, LATT_INI%B, -1, -1, .TRUE.)
    ! check new reciprocal grid for consistency
    ! with old grid
    ! if this is inconsistent the FFT or the retrieval of
    ! wavefunctions from the grid to the wavefunction array would fail to work
    CALL CHECK_GEN_LAYOUT_GRID( GRID,  GRID_NEW)
    CALL CHECK_GEN_LAYOUT( WDES, WDES_NEW, KPOINTS_ORIG%NKPTS)

    ! grid is no longer required
    CALL DEALLOC_GRD(GRID_NEW)
    WDES_NEW%GRID=>GRID
    ! overwrite the WDES by WDES_NEW
    CALL COPYWDES(WDES, WDES_NEW ,.TRUE.)
    ! update the WDES_FOCK
    CALL RESETUP_FOCK_WDES( WDES, LATT_CUR, LATT_INI, IU6)
    
  END SUBROUTINE RE_GEN_LAYOUT

!*********************************************************************
!
! RE_GEN_LAYOUT regenerates the data layout and checks whether 
! the data layout is identical to the original one
!
!*********************************************************************
  
  SUBROUTINE GEN_LAYOUT_FOR_INTERPOLATION( GRID, WDES, WDES_INTER, KPOINTS_INTER, LATT_CUR, LATT_INI, IU6, IU0)
    USE prec
    USE mgrid
    USE wave
    USE constant
    USE base
    USE mkpoints
    USE full_kpoints
    USE lattice

    TYPE (grid_3d), TARGET :: GRID
    TYPE (wavedes)     WDES_INTER
    TYPE (wavedes)     WDES
    TYPE (kpoints_struct) KPOINTS_INTER
    TYPE (latt)        LATT_CUR
    TYPE (latt)        LATT_INI
    INTEGER IU6, IU0
  ! local
    TYPE (grid_3d)     GRID_NEW
    INTEGER IND, N2, N3, NK
  ! first check that we use over band distribution only
  ! if not, full stop now
#ifdef MPI
    IF (WDES%COMM_INB%NCPU /= 1) THEN
         CALL VTUTOR('E','NPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
         IU6,3)
         CALL VTUTOR('S','NPAR',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
         IU0,3)
    ENDIF
#endif
  ! copy the loop counters and dimensions from GRID to GRID_NEW
    GRID_NEW=GRID
  ! copy all data and references from WDES to WDES_INTER
    WDES_INTER=WDES
    CALL INIT_KPOINT_WDES(WDES_INTER, KPOINTS_INTER)

    CALL GEN_LAYOUT(GRID_NEW, WDES_INTER, LATT_CUR%B, LATT_INI%B, IU6,.TRUE.)
    CALL GEN_INDEX (GRID_NEW, WDES_INTER, LATT_CUR%B, LATT_INI%B, -1, -1, .TRUE.)
    ! check new reciprocal grid for consistency
    ! with old grid
    ! if this is inconsistent the FFT or the retrieval of
    ! wavefunctions from the grid to the wavefunction array would fail to work
    CALL CHECK_GEN_LAYOUT_GRID( GRID,  GRID_NEW)
    ! grid is no longer required
    CALL DEALLOC_GRD(GRID_NEW)
    WDES_INTER%GRID=>GRID
    
  END SUBROUTINE GEN_LAYOUT_FOR_INTERPOLATION


!*********************************************************************
!
! REALLOCATE_WAVE reallocates the wavefunction array and 
! calculates the wavefunctions in the entire BZ from the 
! wavefunctions in the IRBZ (KPOINTS_ORIG%NKPTS)
! WDES must already contain the new number of k-points
!
!*********************************************************************

  SUBROUTINE REALLOCATE_WAVE(W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR, KPOINTS_TRANS )
    USE prec
    USE wave
    USE mgrid
    USE full_kpoints
    USE poscar
    USE pseudo
    USE lattice
    USE nonl_high
    USE pawsym
#ifndef oldsym
    USE spinsym
#endif
    TYPE (wavespin)    W          ! wavefunction
    TYPE (grid_3d)     GRID
    TYPE (wavedes)     WDES
    TYPE (nonl_struct) NONL_S
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (latt)        LATT_CUR
    TYPE (skpoints_trans),TARGET,OPTIONAL :: KPOINTS_TRANS
    
! local
    TYPE (skpoints_trans),TARGET :: KPOINTS_TRANS_
    TYPE (skpoints_trans),POINTER:: PKPOINTS_TRANS
    TYPE (wavespin)    W_NEW      ! new wavefunction array
    TYPE (wavefun)     WUP
    TYPE (wavefun)     WDW
    TYPE (wavedes1)    WDES1
    INTEGER ISP, NK, ISPINOR, N, NK_OLD, ISP_OLD
! 
    TYPE( rotation_handle), POINTER :: ROT_HANDLE => NULL()
    INTEGER NI, NT, NPRO, NIP, NPROP, NIS

! first calculate the index array and the phase factor to determine
! the transform from the old to the new k-points
!
    IF (PRESENT(KPOINTS_TRANS)) THEN
       PKPOINTS_TRANS=>KPOINTS_TRANS
    ELSE
       PKPOINTS_TRANS=>KPOINTS_TRANS_
    ENDIF
    
    CALL GENERATE_KPOINTS_TRANS(GRID, KPOINTS_ORIG%NKPTS, WDES, KPOINTS_FULL_ORIG, PKPOINTS_TRANS)
!   gamma only: nothing usefull can be done here
#ifndef gammareal
!
! now allocate the new wavefunction array and zero it
!
    CALL ALLOCW(WDES,W_NEW,WUP,WDW)
!
! and perform the transformation of the wavefunction coefficients
!
    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          NK_OLD=PKPOINTS_TRANS%NK_OLD(NK)
          ISP_OLD=ISP
          IF (PKPOINTS_TRANS%SPINFLIP(NK)==1) THEN
             ISP_OLD=3-ISP
          ENDIF

          DO N=1,WDES%NBANDS
             DO ISPINOR=0,WDES%NRSPINORS-1
                CALL ROTATE_WAVE(WDES%NGVECTOR(NK), &
                     W_NEW%CW(1+ISPINOR*WDES%NGVECTOR(NK),N,NK,ISP), & 
                     W%CW(1+ISPINOR*WDES%NGVECTOR(NK_OLD),N,NK_OLD,ISP_OLD), &
                     PKPOINTS_TRANS%CPHASE(1,NK), PKPOINTS_TRANS%NINDPW(1,NK),  &
                     PKPOINTS_TRANS%LINV(NK), PKPOINTS_TRANS%LSHIFT(NK))
             ENDDO
#ifndef oldsym
             !SMAT=PKPOINTS_TRANS%RSSYMOP(:,:,NK)
             CALL ROTATE_WAVE_SPIN_RECIP(WDES, PKPOINTS_TRANS%RSSYMOP(:,:,NK), &
                  NK, PKPOINTS_TRANS%SPINFLIP(NK), W_NEW%CW(:,N,NK,ISP))
#endif       
          ENDDO
       ENDDO
    ENDDO

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS          
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL SETWDES(WDES,WDES1,NK)
          CALL GENERATE_ROT_HANDLE(ROT_HANDLE, PKPOINTS_TRANS, P, LATT_CUR, NK, WDES1 )

          NK_OLD=PKPOINTS_TRANS%NK_OLD(NK)
          ISP_OLD=ISP
          IF (PKPOINTS_TRANS%SPINFLIP(NK)==1) THEN
             ISP_OLD=3-ISP
          ENDIF

          DO N=1,WDES%NBANDS
             DO ISPINOR=0,WDES%NRSPINORS-1
                NIS=1
                NPRO =ISPINOR *(WDES%NPRO/2)+1
                DO NT=1, WDES%NTYP
                   DO NI=NIS, WDES%NITYP(NT)+NIS-1
                      NIP=PKPOINTS_TRANS%ROTMAP(NI,NK)
                      NPROP=ROT_HANDLE%NPRO_NI(NIP)+ISPINOR *(WDES%NPRO/2)
                      CALL ROTATE_VECTOR( PKPOINTS_TRANS%LINV(NK),W%CPROJ(NPROP,N,NK_OLD,ISP_OLD),W_NEW%CPROJ(NPRO,N,NK,ISP),ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,ROT_HANDLE%SL(1,1,0),P(NT))
                      NPRO= WDES%LMMAX(NT)+NPRO
                   ENDDO
                   NIS=NIS+WDES%NITYP(NT)
                ENDDO
             ENDDO
#ifndef oldsym
             CALL ROTATE_WAVE_CHARACTER_SPIN(WDES1, W_NEW%CPROJ(:,N,NK,ISP), PKPOINTS_TRANS%RSSYMOP(:,:,NK))
#endif
          ENDDO
       ENDDO
    ENDDO
!
! restore the eigenvalues and weights
! do this on all nodes
!
    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          NK_OLD=PKPOINTS_TRANS%NK_OLD(NK)
          ISP_OLD=ISP
          IF (PKPOINTS_TRANS%SPINFLIP(NK)==1) THEN
             ISP_OLD=3-ISP
          ENDIF
          DO N=1,WDES%NB_TOT
             W_NEW%CELTOT(N,NK,ISP)=W%CELTOT(N,NK_OLD,ISP_OLD)
             W_NEW%FERTOT(N,NK,ISP)=W%FERTOT(N,NK_OLD,ISP_OLD)
             W_NEW%AUXTOT(N,NK,ISP)=W%AUXTOT(N,NK_OLD,ISP_OLD)
          ENDDO
       ENDDO
    ENDDO
    
    CALL DEALLOCW(W)
    W=W_NEW
#endif
! sync orbitals to all nodes
    CALL KPAR_SYNC_ALL(WDES,W)
!
! reallocate reciprocal space projectors (k-point dependent)
!
    IF (NONL_S%LRECIP) THEN
       CALL NONL_DEALLOC(NONL_S)
       CALL NONL_ALLOC (NONL_S,T_INFO,P,WDES, .FALSE.)

       CALL SPHER(GRID, NONL_S, P, WDES, LATT_CUR, 1)
       CALL PHASE(WDES,NONL_S,0)
    ENDIF
    IF (.NOT.PRESENT(KPOINTS_TRANS)) THEN
       CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS_)
    ENDIF
    CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

  END SUBROUTINE REALLOCATE_WAVE

!*********************************************************************
!
! CONTRACT_WAVE contracts contributions to the gradient 
! from all k-points into the IRBZ
! this routine is called from the GW subroutine 
!
!*********************************************************************

  SUBROUTINE CONTRACT_WAVE(W, GRID, WDES)
    USE prec
    USE wave
    USE mgrid
    USE full_kpoints
    USE fock
    USE poscar
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE pawsym
#ifndef oldsym
    use spinsym
#endif
    TYPE (wavespin)    W          ! wavefunction
    TYPE (grid_3d)     GRID
    TYPE (wavedes)     WDES
    
! local
    TYPE (skpoints_trans),TARGET :: KPOINTS_TRANS
    INTEGER ISP, NK, ISPINOR, N, NK_IRZ, ISP_IRZ
#ifndef oldsym
    COMPLEX(q) :: SMAT(2,2)
#endif
! 
!    IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
!       CALLMPI( M_stop('CONTRACT_WAVE: KPAR>1 not implemented, sorry.') )
!       STOP
!    END IF

    CALL GENERATE_KPOINTS_TRANS(GRID, KPOINTS_ORIG%NKPTS, WDES, KPOINTS_FULL_ORIG, KPOINTS_TRANS)
!
! and perform the transformation of the wavefunction coefficients
!
    DO ISP=1,WDES%ISPIN
       DO NK=KPOINTS_ORIG%NKPTS+1,WDES%NKPTS
          NK_IRZ=KPOINTS_TRANS%NK_OLD(NK)
          ISP_IRZ=ISP
          IF (KPOINTS_TRANS%SPINFLIP(NK)==1) THEN
             ISP_IRZ=3-ISP
          ENDIF    
          DO N=1,WDES%NBANDS
#ifndef oldsym
             SMAT=CONJG(KPOINTS_TRANS%RSSYMOP(:,:,NK))
             SMAT=TRANSPOSE(SMAT)
             CALL ROTATE_WAVE_SPIN_RECIP(W%WDES, SMAT, NK_IRZ, &
                  KPOINTS_TRANS%SPINFLIP(NK), W%CW(:,N,NK,ISP))
#endif
             DO ISPINOR=0,WDES%NRSPINORS-1
                CALL ROTATE_WAVE_BACK(WDES%NGVECTOR(NK), &
                     W%CW(1+ISPINOR*WDES%NGVECTOR(NK),N,NK,ISP), & 
                     W%CW(1+ISPINOR*WDES%NGVECTOR(NK_IRZ),N,NK_IRZ,ISP_IRZ), &
                     KPOINTS_TRANS%CPHASE(1,NK), KPOINTS_TRANS%NINDPW(1,NK),  &
                     KPOINTS_TRANS%LINV(NK), KPOINTS_TRANS%LSHIFT(NK))
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS)

  END SUBROUTINE CONTRACT_WAVE


!********************** SUBROUTINE GENERATE_KPOINTS_TRANS  *************
!
! GENERATE_KPOINTS_TRANS is similar to the routine SET_INDPW_FULL
! it determines the  index and the phase factors for rotating wavefunctions
! from one k point to another
! the WDES must contain a proper pointer to the new set of wavefunctions
!
! the subroutine essentially copies all required entries from 
! KPOINTS_F to KPOINTS_TRANS and generates a complete description how
! to go from the k-points in the original IRBZ to another k-points
!
!***********************************************************************

    SUBROUTINE GENERATE_KPOINTS_TRANS(GRID, NKPTS_ORIG, WDES, KPOINTS_F, KPOINTS_TRANS)
      USE prec
      USE mgrid
      USE wave
      USE mpimy
      USE constant
      USE sym_prec
      USE main_mpi
      IMPLICIT NONE

      TYPE (grid_3d) :: GRID
      INTEGER NKPTS_ORIG                    ! number of k-points in the IRZof the original (high symmetry) structure
      TYPE (wavedes) :: WDES                ! wave function descriptor
      TYPE (skpoints_full) :: KPOINTS_F     ! k-points in the wull BZ 
      TYPE (skpoints_trans) KPOINTS_TRANS   ! result: transformation matrix
   ! local
      INTEGER, ALLOCATABLE :: IGRIDIND(:,:,:)
      INTEGER              :: NI,NK,NKI,NK_FULL
      INTEGER              :: NKPTS_START
      INTEGER              :: NG1I,NG2I,NG3I,NG1,NG2,NG3
      INTEGER              :: NGX,NGY,NGZ,NGVECI
      REAL(q)              :: PHASE
      INTEGER              :: NIONS

! allocate arrays and set some values to zero
      IF (WDES%NKPTS < NKPTS_ORIG) THEN
         WRITE(0,*) 'internal error in GENERATE_KPOINTS_TRANS: not sufficient k-points ',WDES%NKPTS,NKPTS_ORIG
         STOP
      ENDIF

      NGX=GRID%NGX
      NGY=GRID%NGY
      NGZ=GRID%NGZ

      NKPTS_START=NKPTS_ORIG
! test even rotate original wavefunctions (of course nothing should happen or change)
      NKPTS_START=0
      NIONS=SIZE(KPOINTS_F%ROTMAP,1)
      ALLOCATE(IGRIDIND(NGX,NGY,NGZ),&
               KPOINTS_TRANS%CPHASE (WDES%NGDIM,WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%NINDPW(WDES%NGDIM,WDES%NKPTS -NKPTS_START), &
               KPOINTS_TRANS%NK_OLD(WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%ISYMOP(3,3,WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%LSHIFT(WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%LSHIFT_G(WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%LINV(WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%ROTMAP(NIONS,WDES%NKPTS-NKPTS_START), &
               KPOINTS_TRANS%SPINFLIP(WDES%NKPTS-NKPTS_START))

#ifndef oldsym
      ALLOCATE(KPOINTS_TRANS%RSSYMOP(2,2,WDES%NKPTS -NKPTS_START))
#endif
      KPOINTS_TRANS%CPHASE=(1._q,0._q)
      KPOINTS_TRANS%LSHIFT=.FALSE.

! now index the wavefunctions at the new k-point
! take G-index from old k-point, rotate G-vec, enter new index
      DO NK=NKPTS_START+1,WDES%NKPTS

         ! first we set up an 3d-array, that stores the index into
         ! the compact plane wave array at the new k-point
         IGRIDIND=0
         DO NI=1,WDES%NGVECTOR(NK)
            NG1=WDES%IGX(NI,NK)
            NG2=WDES%IGY(NI,NK)
            NG3=WDES%IGZ(NI,NK)
            IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))=NI
         ENDDO

         ! determine index of this k-point in KPOINTS_F
         NK_FULL=KPOINT_IN_FULL_GRID(WDES%VKPT(:,NK),KPOINTS_F)
         ! index in the IRBZ
         NKI=KPOINTS_F%NEQUIV(NK_FULL)

         ! copy entries from KPOINTS_F
         KPOINTS_TRANS%NK_OLD(NK)=NKI
         KPOINTS_TRANS%LINV(NK)  =KPOINTS_F%LINV(NK_FULL)
         KPOINTS_TRANS%ISYMOP(:,:,NK)=KPOINTS_F%ISYMOP(:,:,NK_FULL)
         KPOINTS_TRANS%ROTMAP(:,NK)  =KPOINTS_F%ROTMAP(:,NK_FULL)
         KPOINTS_TRANS%SPINFLIP(NK)  =KPOINTS_F%SPINFLIP(NK_FULL)
#ifndef oldsym
         KPOINTS_TRANS%RSSYMOP(:,:,NK)= KPOINTS_F%RSSYMOP(:,:,NK_FULL)
#endif
         ! number of G vectors for original k in IRBZ must be identical
         NGVECI=WDES%NGVECTOR(NKI)
         IF (NGVECI /= WDES%NGVECTOR(NK)) THEN
            WRITE(0,*) "internal error in GENERATE_KPOINTS_TRANS: number of G-vector changed in star ",& 
                 NGVECI,WDES%NGVECTOR(NK)
            STOP
         ENDIF
         
         ! phase shift (required for space group operations)
         IF ((ABS(KPOINTS_F%TRANS(1,NK_FULL))>TINY) .OR. &
             (ABS(KPOINTS_F%TRANS(2,NK_FULL))>TINY) .OR. &
             (ABS(KPOINTS_F%TRANS(3,NK_FULL))>TINY)) THEN
            KPOINTS_TRANS%LSHIFT(NK-NKPTS_START)=.TRUE.
         ELSE
            KPOINTS_TRANS%LSHIFT(NK-NKPTS_START)=.FALSE.
         ENDIF
         
         ! now loop over all G vectors of the original k point in the IRBZ
         DO NI=1,NGVECI
            NG1I=WDES%IGX(NI,NKI)
            NG2I=WDES%IGY(NI,NKI)
            NG3I=WDES%IGZ(NI,NKI)
            ! corresponding vector G'=S G
            ! S is the symmetry operation taking k_irz into k' = S k_irz
            NG1=NG1I*KPOINTS_F%IROTOP(1,1,NK_FULL)+NG2I*KPOINTS_F%IROTOP(2,1,NK_FULL)+NG3I*KPOINTS_F%IROTOP(3,1,NK_FULL)
            NG2=NG1I*KPOINTS_F%IROTOP(1,2,NK_FULL)+NG2I*KPOINTS_F%IROTOP(2,2,NK_FULL)+NG3I*KPOINTS_F%IROTOP(3,2,NK_FULL)
            NG3=NG1I*KPOINTS_F%IROTOP(1,3,NK_FULL)+NG2I*KPOINTS_F%IROTOP(2,3,NK_FULL)+NG3I*KPOINTS_F%IROTOP(3,3,NK_FULL)

            ! KPOINTS_TRANS%NINDPW(NI,NK) stores the index of G' = S G
            KPOINTS_TRANS%NINDPW(NI,NK-NKPTS_START)=IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))

            IF (KPOINTS_TRANS%NINDPW(NI,NK-NKPTS_START)==0) THEN
1234           FORMAT("internal error in GENERATE_KPOINTS_TRANS: G vector not found ",8I5," mkpoints_change.F")
               WRITE(*,1234) NK,NG1,NG2,NG3,NG1I,NG2I,NG3I,NI
               STOP
            ENDIF
            ! k point was already in IRBZ
            IF (NK<=NKPTS_ORIG) THEN
               IF (KPOINTS_TRANS%NINDPW(NI,NK-NKPTS_START)/=NI) THEN
                  WRITE(*,*) "internal error in GENERATE_KPOINTS_TRANS: a G-vector in the IRBZ changed position",NI,KPOINTS_TRANS%NINDPW(NI,NK-NKPTS_START)
                  STOP
               ENDIF
            ENDIF

            IF (KPOINTS_TRANS%LSHIFT(NK-NKPTS_START)) THEN
               PHASE=-TPI*(NG1*KPOINTS_F%TRANS(1,NK_FULL)+ &
                           NG2*KPOINTS_F%TRANS(2,NK_FULL)+ &
                           NG3*KPOINTS_F%TRANS(3,NK_FULL))
               KPOINTS_TRANS%CPHASE(NI,NK-NKPTS_START)=CMPLX(COS(PHASE),SIN(PHASE),KIND=q)
            ENDIF
         ENDDO
      ENDDO
      DEALLOCATE(IGRIDIND)

    END SUBROUTINE GENERATE_KPOINTS_TRANS


    SUBROUTINE DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS)
      IMPLICIT NONE
      TYPE (skpoints_trans) KPOINTS_TRANS

!   gamma only: nothing usefull can be done here

      DEALLOCATE(&
               KPOINTS_TRANS%CPHASE, &
               KPOINTS_TRANS%NINDPW, &
               KPOINTS_TRANS%NK_OLD, &
               KPOINTS_TRANS%ISYMOP, &
               KPOINTS_TRANS%LSHIFT, &
               KPOINTS_TRANS%LSHIFT_G, &
               KPOINTS_TRANS%LINV, &
               KPOINTS_TRANS%ROTMAP, &
               KPOINTS_TRANS%SPINFLIP)
#ifndef oldsym
       DEALLOCATE(KPOINTS_TRANS%RSSYMOP)
#endif
    END SUBROUTINE DEALLOCATE_KPOINTS_TRANS

!********************** SUBROUTINE KPOINTS_TRANS_Q       *************
!
! KPOINTS_TRANS_Q generates a structure that allows to symmetrize
! two dimensional arrays such as the potential or the dielectric matrix
!  W_q(G,G') = W(G+q,G'+q)
! the calling routine must supply the q-point NQ and all symmetry
! operations
! the routine then determine which symmetry operations leave the k-point
! unchanged 
!
!***********************************************************************

    SUBROUTINE KPOINTS_TRANS_Q(GRID, WDES, KPOINTS_TRANS, NQ, ROTMAP, MAGROT )
      USE prec
      USE mgrid
      USE wave
      USE mpimy
      USE constant
      USE sym_prec
      USE main_mpi
#ifndef oldsym
      USE spinsym
#endif
      IMPLICIT NONE

      TYPE (grid_3d) :: GRID
      TYPE (wavedes) :: WDES                ! wave function descriptor
      TYPE (skpoints_trans) KPOINTS_TRANS   ! result: transformation matrix
      INTEGER :: NQ 
      INTEGER ROTMAP(:,:,:)
      REAL(q) MAGROT(:,:)
! common symmetry variables
      INTEGER ISYMOP, NROT, IGRPOP, NROTK, INVMAP, NPCELL
      REAL(q) GTRANS, AP
      COMMON /SYMM/ ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &                            GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      INTEGER INVERS(9)
      DATA INVERS /-1,0,0,0,-1,0,0,0,-1/
    ! local
      LOGICAL LINV
! external routine
      LOGICAL,EXTERNAL :: SGREQL
      TYPE (skpoints_full) K_TMP
      REAL(q) V(3), VR(3), VT(3), ROP(3,3)
      INTEGER :: NOP, NEQUIV

      INTEGER, ALLOCATABLE :: IGRIDIND(:,:,:)
      INTEGER              :: NI,NK
      INTEGER              :: NG1I,NG2I,NG3I,NG1,NG2,NG3
      INTEGER              :: NGX,NGY,NGZ
      REAL(q)              :: PHASE
      INTEGER              :: NIONS

      NIONS=SIZE(ROTMAP,1)

      ALLOCATE(K_TMP%VKPT(3,NROTK),K_TMP%WTKPT(NROTK), K_TMP%TRANS(3,NROTK), K_TMP%LSHIFT(NROTK), & 
           K_TMP%SPINFLIP(NROTK), &
           K_TMP%NEQUIV(NROTK),K_TMP%IROTOP(3,3,NROTK),K_TMP%ISYMOP(3,3,NROTK), &
           K_TMP%LINV(NROTK),K_TMP%NOP(NROTK),K_TMP%ROTMAP(NIONS,NROTK))
#ifndef oldsym
      ALLOCATE(K_TMP%RSSYMOP(2,2,NROTK))
#endif
      NEQUIV=0

      DO NOP=1,NROTK
! test existence of inversion op
         IF (SGREQL(IGRPOP(1,1,NOP),INVERS)) LINV=.TRUE.
! copy symmetry op to real array
         ROP=IGRPOP(:,:,NOP)
! make new k-point and shift (for testing) it to 1st BZ
         VR(1)=WDES%VKPT(1,NQ)
         VR(2)=WDES%VKPT(2,NQ)
         VR(3)=WDES%VKPT(3,NQ) 
         V(1)=VR(1)*ROP(1,1)+VR(2)*ROP(2,1)+VR(3)*ROP(3,1)
         V(2)=VR(1)*ROP(1,2)+VR(2)*ROP(2,2)+VR(3)*ROP(3,2)
         V(3)=VR(1)*ROP(1,3)+VR(2)*ROP(2,3)+VR(3)*ROP(3,3)
! bring the point to the primitive cell
         VT(1)=MOD(V(1)+6.5_q,1._q)-0.5_q
         VT(2)=MOD(V(2)+6.5_q,1._q)-0.5_q
         VT(3)=MOD(V(3)+6.5_q,1._q)-0.5_q
! test against current k-point
         IF(( ABS(MOD(VT(1)-VR(1)+6.5,1._q)-0.5_q)<TINY) .AND. &
            ( ABS(MOD(VT(2)-VR(2)+6.5,1._q)-0.5_q)<TINY) .AND. &
            ( ABS(MOD(VT(3)-VR(3)+6.5,1._q)-0.5_q)<TINY)) THEN
            NEQUIV=NEQUIV+1
            ! store the difference vector between rotated and original k-point
            K_TMP%VKPT(1,NEQUIV)=NINT(V(1)-VR(1))
            K_TMP%VKPT(2,NEQUIV)=NINT(V(2)-VR(2))
            K_TMP%VKPT(3,NEQUIV)=NINT(V(3)-VR(3))
            K_TMP%NEQUIV(NEQUIV)=NQ
            K_TMP%LINV(NEQUIV)=.FALSE.
            K_TMP%IROTOP(:,:,NEQUIV)=INT(ROP(:,:))
            K_TMP%ISYMOP(:,:,NEQUIV)=ISYMOP(:,:,NOP)
            K_TMP%TRANS(:,NEQUIV)=GTRANS(:,NOP)
            K_TMP%SPINFLIP(NEQUIV) =0
            IF ( SIZE(MAGROT)>1) THEN
               IF( MAGROT(NOP,1)==-1) THEN
                  K_TMP%SPINFLIP(NEQUIV) =1
               ENDIF
            ENDIF
            K_TMP%ROTMAP(:,NEQUIV)=ROTMAP(:,NOP,1)
            K_TMP%NOP(NEQUIV)=NOP
#ifndef oldsym
            K_TMP%RSSYMOP(:,:,NEQUIV)=RSSYMOP(:,:,NOP)
#endif
         ENDIF
      ENDDO
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Here one could add the additional use of time reversal symmetry
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     IF (.NOT. LINV .AND. ISYM>=0) THEN
!     ENDIF

      NGX=GRID%NGX
      NGY=GRID%NGY
      NGZ=GRID%NGZ

      ! use KPOINTS_TRANS%NKPTS to store the number of symmetry ops
      ! that leave this k-point unchanged
      KPOINTS_TRANS%NKPTS=NEQUIV

      ALLOCATE(IGRIDIND(NGX,NGY,NGZ),&
               KPOINTS_TRANS%CPHASE(WDES%NGVECTOR(NQ),NEQUIV), &
               KPOINTS_TRANS%NINDPW(WDES%NGVECTOR(NQ),NEQUIV), &
               KPOINTS_TRANS%NK_OLD(NEQUIV), &
               KPOINTS_TRANS%ISYMOP(3,3,NEQUIV), &
               KPOINTS_TRANS%LSHIFT(NEQUIV), &
               KPOINTS_TRANS%LSHIFT_G(NEQUIV), &
               KPOINTS_TRANS%LINV(NEQUIV), &
               KPOINTS_TRANS%ROTMAP(NIONS,NEQUIV), &
               KPOINTS_TRANS%SPINFLIP(NEQUIV))
#ifndef oldsym
      ALLOCATE(KPOINTS_TRANS%RSSYMOP(2,2,NEQUIV))
#endif
      KPOINTS_TRANS%CPHASE=(1._q,0._q)
      KPOINTS_TRANS%LSHIFT=.FALSE.

      ! first we set up an 3d-array, that stores the index into
      ! the compact plane wave array
      IGRIDIND=0
      DO NI=1,WDES%NGVECTOR(NQ)
         NG1=WDES%IGX(NI,NQ)
         NG2=WDES%IGY(NI,NQ)
         NG3=WDES%IGZ(NI,NQ)
         IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))=NI
      ENDDO

! now index the wavefunctions at the new k-point
! take G-index from old k-point, rotate G-vec, enter new index
      DO NK=1,NEQUIV
         ! copy entries from K_TMP
         KPOINTS_TRANS%NK_OLD(NK)=NQ
         KPOINTS_TRANS%LINV(NK)  =K_TMP%LINV(NK)
         KPOINTS_TRANS%ISYMOP(:,:,NK)=K_TMP%ISYMOP(:,:,NK)
         KPOINTS_TRANS%ROTMAP(:,NK)  =K_TMP%ROTMAP(:,NK)
         KPOINTS_TRANS%SPINFLIP(NK)  =K_TMP%SPINFLIP(NK)
#ifndef oldsym
         KPOINTS_TRANS%RSSYMOP(:,:,NK)= K_TMP%RSSYMOP(:,:,NK)
#endif
         ! phase shift (required for space group operations)
         IF ((ABS(K_TMP%TRANS(1,NK))>TINY) .OR. &
             (ABS(K_TMP%TRANS(2,NK))>TINY) .OR. &
             (ABS(K_TMP%TRANS(3,NK))>TINY)) THEN
            KPOINTS_TRANS%LSHIFT(NK)=.TRUE.
         ELSE
            KPOINTS_TRANS%LSHIFT(NK)=.FALSE.
         ENDIF
         
         ! now loop over all G vectors of the original k point in the IRBZ
         DO NI=1,WDES%NGVECTOR(NQ)
            NG1I=WDES%IGX(NI,NQ)
            NG2I=WDES%IGY(NI,NQ)
            NG3I=WDES%IGZ(NI,NQ)
            ! corresponding vector G'=S G
            ! S is the symmetry operation taking G+k_irz into G' = S (G+k_irz) - k_irz
            NG1=NG1I*K_TMP%IROTOP(1,1,NK)+NG2I*K_TMP%IROTOP(2,1,NK)+NG3I*K_TMP%IROTOP(3,1,NK)+INT(K_TMP%VKPT(1,NK))
            NG2=NG1I*K_TMP%IROTOP(1,2,NK)+NG2I*K_TMP%IROTOP(2,2,NK)+NG3I*K_TMP%IROTOP(3,2,NK)+INT(K_TMP%VKPT(2,NK))
            NG3=NG1I*K_TMP%IROTOP(1,3,NK)+NG2I*K_TMP%IROTOP(2,3,NK)+NG3I*K_TMP%IROTOP(3,3,NK)+INT(K_TMP%VKPT(3,NK))

            ! KPOINTS_TRANS%NINDPW(NI,NK) stores the index of G' = S G
            KPOINTS_TRANS%NINDPW(NI,NK)=IGRIDIND(MP(NG1,NGX),MP(NG2,NGY),MP(NG3,NGZ))

            IF (KPOINTS_TRANS%NINDPW(NI,NK)==0) THEN
1234           FORMAT("internal error in KPOINTS_TRANS_Q: G vector not found ",10I5," mkpoints_change.F")
               WRITE(*,1234) NK,NG1,NG2,NG3,NG1I,NG2I,NG3I,NI,NQ,NROTK
               STOP
            ENDIF

            IF (KPOINTS_TRANS%LSHIFT(NK)) THEN
               PHASE=-TPI*(NG1*K_TMP%TRANS(1,NK)+ &
                           NG2*K_TMP%TRANS(2,NK)+ &
                           NG3*K_TMP%TRANS(3,NK))
               KPOINTS_TRANS%CPHASE(NI,NK)=CMPLX(COS(PHASE),SIN(PHASE),KIND=q)
            ENDIF
         ENDDO
      ENDDO
      DEALLOCATE(IGRIDIND)

      DEALLOCATE(K_TMP%VKPT, K_TMP%TRANS, K_TMP%LSHIFT, K_TMP%SPINFLIP, &
           K_TMP%NEQUIV, K_TMP%IROTOP, K_TMP%ISYMOP, &
           K_TMP%LINV, K_TMP%NOP, K_TMP%ROTMAP)
#ifndef oldsym
      DEALLOCATE(K_TMP%RSSYMOP)
#endif
    END SUBROUTINE KPOINTS_TRANS_Q


    SUBROUTINE SYMMETRIZE_WPOT( WDES1, KPOINTS_TRANS, WPOT)
      USE wave
      TYPE (wavedes1) :: WDES1               ! wave function descriptor
      TYPE (skpoints_trans) KPOINTS_TRANS   ! result: transformation matrix
      COMPLEX(q) :: WPOT(:,:)
    ! local
      INTEGER NP, NK
      COMPLEX(q),ALLOCATABLE :: W_ROT(:,:),WSUM(:,:)
     
      
      NP=WDES1%NGVECTOR

      ALLOCATE(W_ROT(SIZE(WPOT,1), NP),WSUM(SIZE(WPOT,1), NP))

      WSUM=0
      DO NK=1,KPOINTS_TRANS%NKPTS
         
         CALL ROTATE_WPOT(NP,W_ROT(1,1), WPOT(1,1), SIZE(WPOT,1 ), & 
              KPOINTS_TRANS%CPHASE(1,NK), KPOINTS_TRANS%NINDPW(1,NK),  &
              KPOINTS_TRANS%LINV(NK), KPOINTS_TRANS%LSHIFT(NK))
#ifndef oldsym
!        CALL ROTATE_WPOT_SPIN()
#endif
         WSUM=WSUM+W_ROT
      END DO

!      IF (WDES1%COMM%NODE_ME==WDES1%COMM%IONODE) THEN
!         WRITE(*,'(20F10.5)') WPOT(1:10,1:10)
!      ENDIF

      WPOT(1:NP,1:NP)=WSUM(1:NP,1:NP)/KPOINTS_TRANS%NKPTS

!      IF (WDES1%COMM%NODE_ME==WDES1%COMM%IONODE) THEN
!         WRITE(*,'(20F10.5)') WPOT(1:10,1:10)
!      ENDIF

      DEALLOCATE(W_ROT,WSUM)

    END SUBROUTINE SYMMETRIZE_WPOT


!***********************************************************************
!
! this subroutine generates a rotation handle using
! a KPOINTS_TRANS structure
! it is similar to the routine in mkpoints_full, but that version
! is restricted to the HF case and does not use a KPOINTS_TRANS
! structure
!
!***********************************************************************

  SUBROUTINE GENERATE_ROT_HANDLE(ROT_HANDLE, KPOINTS_TRANS, P, LATT_CUR, NK, WDES1 )
    USE wave_high
    USE pseudo
    USE lattice
    USE nonl_high
    USE pawsym

    TYPE( rotation_handle), POINTER :: ROT_HANDLE  ! handle for rotation
    TYPE (skpoints_trans) :: KPOINTS_TRANS         ! "transition" structure 
    TYPE (potcar)      P(:)                        ! pseudo potential descriptors
    TYPE (latt)        LATT_CUR                    ! lattice
    INTEGER            NK                          ! operation (k-points) in KPOINTS_TRANS
    TYPE (wavedes1)    WDES1                       ! WDES only required to get NPRO, NPROD, NIONS etc.
! local

    INTEGER :: NPRO, NIS, NT, NI
    INTEGER, EXTERNAL :: MAXL


    IF (.NOT. ASSOCIATED(ROT_HANDLE)) THEN
       ALLOCATE (ROT_HANDLE)
       
       ROT_HANDLE%LDIM=MAXL(SIZE(P),P(1))          ! maximum l quantum number
       ROT_HANDLE%MMAX=(2*ROT_HANDLE%LDIM+1)       ! maximum m quantum number
       ALLOCATE (ROT_HANDLE%SL(ROT_HANDLE%MMAX,ROT_HANDLE%MMAX,0:ROT_HANDLE%LDIM), ROT_HANDLE%NPRO_NI(WDES1%NIONS))

       ! setup index array NPRO_NI into the CPROJ array
       NPRO=1
       NIS=1
       DO NT=1, WDES1%NTYP
          DO NI=NIS, WDES1%NITYP(NT)+NIS-1
             ROT_HANDLE%NPRO_NI(NI)=NPRO
             NPRO= WDES1%LMMAX(NT)+NPRO
          ENDDO
          NIS=NIS+WDES1%NITYP(NT)
       ENDDO
       ROT_HANDLE%NK=-1
    ENDIF

    IF (ROT_HANDLE%NK/= NK) THEN
       ROT_HANDLE%NK = NK
       CALL SETUP_SYM_LL(ROT_HANDLE%MMAX,ROT_HANDLE%LDIM,KPOINTS_TRANS%ISYMOP(:,:,NK), & 
            ROT_HANDLE%SL,LATT_CUR%A,LATT_CUR%B)
    ENDIF
    
  END SUBROUTINE GENERATE_ROT_HANDLE
    
END MODULE kpoints_change
