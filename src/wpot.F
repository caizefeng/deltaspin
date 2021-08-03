#include "symbol.inc"
!*********************************************************************
!
! This module reads and writes the screened potential
!
! Constructing the potential at q points that are not in 
! the IRZ is managed through a handle wpothandle
!
!*********************************************************************

MODULE wpot
  USE chi_base
  USE wave_high
  USE kpoints_change

  TYPE wpothandle
     LOGICAL, POINTER :: LSET(:)
     INTEGER :: IU0, IU6
     TYPE (wavedes), POINTER ::       WGW
     TYPE (skpoints_trans),POINTER :: KPOINTS_TRANS
     COMPLEX(q), POINTER :: C(:,:)
     TYPE (responsefunction), POINTER :: WPOT(:)
     INTEGER :: NKREDLFX, NKREDLFY, NKREDLFZ
  END TYPE wpothandle

  CONTAINS

!*********************************************************************
!
! initialize the potential handle
!
!*********************************************************************

  SUBROUTINE INIT_WPOT_HANDLE(WH, WGW, NQ , IU6, IU0, NKREDLFX, NKREDLFY, NKREDLFZ)
    USE kpoints_change
    IMPLICIT NONE
    TYPE(wpothandle), POINTER :: WH
    TYPE (wavedes), TARGET :: WGW
    INTEGER :: NQ
    INTEGER :: IU6, IU0

    INTEGER :: NKREDLFX, NKREDLFY, NKREDLFZ
  ! local
    INTEGER :: NK

    ALLOCATE(WH)
    ALLOCATE(WH%LSET(NQ), WH%C(WGW%NGDIM, NQ), WH%KPOINTS_TRANS)
    NULLIFY(WH%WPOT)
    WH%WGW=> WGW
    WH%IU6=IU6
    WH%IU0=IU0
    WH%C=0
    WH%LSET=.FALSE.
    WH%NKREDLFX=NKREDLFX
    WH%NKREDLFY=NKREDLFY
    WH%NKREDLFZ=NKREDLFZ
    CALL GENERATE_KPOINTS_TRANS(WGW%GRID, KPOINTS_ORIG%NKPTS, WGW, KPOINTS_FULL_ORIG, WH%KPOINTS_TRANS)

  ! allocate the potential handle
    ALLOCATE(WH%WPOT(KPOINTS_FULL_ORIG%NKPTS))
    DO NK=1,KPOINTS_FULL_ORIG%NKPTS
       NULLIFY(WH%WPOT(NK)%RESPONSEFUN)
    ENDDO
    
  END SUBROUTINE INIT_WPOT_HANDLE

!*********************************************************************
!
! destroy the potential handle
!
!*********************************************************************

  SUBROUTINE DESTROY_WPOT_HANDLE(WH)
    IMPLICIT NONE
    TYPE(wpothandle), POINTER :: WH
  ! local
    INTEGER :: NK

    IF (.NOT. ASSOCIATED(WH)) RETURN

    CALL DEALLOCATE_KPOINTS_TRANS(WH%KPOINTS_TRANS)
    IF (ASSOCIATED(WH%WPOT)) THEN
       ! deallocate the potential handle
       DO NK=1,KPOINTS_FULL_ORIG%NKPTS
          IF (ASSOCIATED(WH%WPOT(NK)%RESPONSEFUN)) THEN
             CALL DEALLOCATE_RESPONSEFUN(WH%WPOT(NK) )
          ENDIF
       ENDDO
       DEALLOCATE(WH%WPOT)
       NULLIFY(WH%WPOT)
    ENDIF

    DEALLOCATE(WH%LSET, WH%C, WH%KPOINTS_TRANS)
    DEALLOCATE(WH)

  END SUBROUTINE DESTROY_WPOT_HANDLE

!*********************************************************************
!
! determine screened two electron potential 
! at an arbitrary q point supplied by the calling routine
! some comments:
! WPOT is of the type of a response function array
! a the gamma point the entries 
! are interpreted as sin and cosine transforms
! and the potential is stored as a real valued function
! in RESPONSER
!
!*********************************************************************

  SUBROUTINE GET_WPOT( WH, NQ_IN_FULL, POTFAK, LFULL )
    USE sym_prec
    USE lattice
    IMPLICIT NONE
    TYPE(wpothandle), POINTER :: WH
    INTEGER :: NQ_IN_FULL
    REAL(q) :: POTFAK(WH%WGW%NGDIM)
  ! local
    INTEGER :: NQ, NQ_IN_FULL_ORIG
    COMPLEX(q) :: C(WH%WGW%NGDIM)
    LOGICAL :: LFULL
    INTEGER :: IERR, I, NP, ierror
    TYPE (wavedes1) WGWQ
    
    CALL SETWDES(WH%WGW, WGWQ, NQ_IN_FULL)

    IF (.NOT. WH%LSET(NQ_IN_FULL)) THEN
       IF (.NOT. ASSOCIATED(WH%WPOT)) THEN
          WRITE(*,*) 'internal error in GET_WPOT: the WH%WPOT handle is not allocated'
          STOP
       ENDIF

       ! map this k-point to corresponding k-point in the KPOINTS_FULL_ORIG
       NQ_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(KPOINTS_FULL%VKPT(:,NQ_IN_FULL),KPOINTS_FULL_ORIG)
       ! corresponding k-point in IRZ
       NQ=KPOINTS_FULL_ORIG%NEQUIV(NQ_IN_FULL_ORIG)
       IF (ABS(KPOINTS_FULL_ORIG%VKPT(1,NQ_IN_FULL_ORIG)-KPOINTS_FULL%VKPT(1,NQ_IN_FULL))>TINY .OR. & 
           ABS(KPOINTS_FULL_ORIG%VKPT(2,NQ_IN_FULL_ORIG)-KPOINTS_FULL%VKPT(2,NQ_IN_FULL))>TINY .OR. &
           ABS(KPOINTS_FULL_ORIG%VKPT(3,NQ_IN_FULL_ORIG)-KPOINTS_FULL%VKPT(3,NQ_IN_FULL))>TINY) THEN
          WRITE(*,*) 'internal error in GET_WPOT shift: ',KPOINTS_FULL_ORIG%VKPT(:,NQ_IN_FULL_ORIG)-KPOINTS_FULL%VKPT(:,NQ_IN_FULL)
          STOP
       ENDIF

       IF (WH%WGW%NGVECTOR(NQ) /=WH%WGW%NGVECTOR(NQ_IN_FULL)) THEN
          WRITE(*,*) 'internal error in GET_WPOT G-vector: ',WH%WGW%NGVECTOR(NQ), WH%WGW%NGVECTOR(NQ_IN_FULL)
          STOP
       ENDIF

       LFULL=.FALSE.
       NP=WGWQ%NGVECTOR

       ! allocate response function array at the k-point in the IRZ and read the potential
       IF (.NOT. ASSOCIATED(WH%WPOT(NQ)%RESPONSEFUN)) THEN

          ! allocate response function array for gamma-point only WH%WPOT%LREALSTORE=.TRUE.
          ! if the file W????.tmp is read than LFULL is set to .FALSE. and WH%WPOT(NQ) is 
          ! deallocated before leaving the routine
#ifdef use_shmem
          CALL ALLOCATE_RESPONSEFUN_SHMEM(WH%WPOT(NQ), WH%WGW%NGDIM, WH%WGW%LGAMMA, WH%WGW%LGAMMA, 1, & 
               WH%WGW%COMM_SHMEM, WH%IU0, WH%IU6, LSEM=.FALSE.)
#else
          CALL ALLOCATE_RESPONSEFUN(WH%WPOT(NQ), WH%WGW%NGDIM, WH%WGW%LGAMMA, WH%WGW%LGAMMA, 1)
#endif

          CALL READ_WPOT(WH%WPOT(NQ), WGWQ, NQ, LFULL, WH%IU0, IERR)
          ! test set all off diagonal components to zero
          ! DO I=1,NP
          !   WH%WPOT(NQ)%RESPONSEFUN(1:I-1,I,1)=0
          !   WH%WPOT(NQ)%RESPONSEFUN(I+1:,I,1)=0
          !   WH%WPOT(NQ)%RESPONSEFUN(I,I,1)=REAL(WH%WPOT(NQ)%RESPONSEFUN(I,I,1))
          ! ENDDO

          ! divide by number of grid points to get proper integration weight
          ! ok, this is really "stupid" and not very clean
          CALL SCALE_RESPONSE_RESPONSE(WH%WPOT(NQ), 1.0_q/GRIDHF%NPLWV)

          ! scale convergence corrections by NKREDLFX
          ! this is only correct if reduction in x y and z is identical
          ! obviously this will require some carefull reconsideration
          ! probably I need the full epsilon with a full recalculation ...
          IF ( WH%NKREDLFX /= WH%NKREDLFY .OR. WH%NKREDLFX /= WH%NKREDLFZ) THEN
             WRITE(*,*) 'internal error in GET_WPOT G-vector: NKREDLF must be identical in all directions'
             STOP
          ENDIF
          IF (  ABS(SUM(WGWQ%VKPT*WGWQ%VKPT))<=G2ZERO ) THEN
! scale the "head" (1,1) element of the response function array
! in principle the first version is the correct version
! empirically the first version is more accurate for LiF and the second one for C
             CALL SCALE_RESPONSE_0(WH%WPOT(NQ), 1.0_q/WH%NKREDLFX/WH%NKREDLFX) 
!            CALL SCALE_RESPONSE_0(WH%WPOT(NQ), 1.0_q/WH%NKREDLFX/WH%NKREDLFY/WH%NKREDLFZ)
          ENDIF

          IF (IERR/=0) THEN
             CALL VTUTOR('E','W missing', &
                  &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.FALSE.,1,WH%IU0,3)
             CALL VTUTOR('E','W missing', &
                  &               0.0_q,1,NQ,1,(0.0_q,0.0_q),1,.FALSE.,1,WH%IU6,3)
             STOP
          ENDIF
       ELSE
          ! potential at k-point in IRZ already associated, hence full potential is
          ! available
          LFULL=.TRUE.
       ENDIF
       !==========================================================================
       ! create potential at corresponding point in full BZ
       !==========================================================================
       IF (LFULL) THEN
          IF (.NOT. ASSOCIATED(WH%WPOT(NQ_IN_FULL)%RESPONSEFUN)) THEN
#ifdef use_shmem
             CALL ALLOCATE_RESPONSEFUN_SHMEM(WH%WPOT(NQ_IN_FULL), WH%WGW%NGDIM, WH%WGW%LGAMMA, WH%WGW%LGAMMA, 1, & 
                  WH%WGW%COMM_SHMEM, WH%IU0, WH%IU6, LSEM=.FALSE. )
#else
             CALL ALLOCATE_RESPONSEFUN(WH%WPOT(NQ_IN_FULL), WH%WGW%NGDIM, WH%WGW%LGAMMA, WH%WGW%LGAMMA, 1)
#endif

             IF (WH%WPOT(NQ_IN_FULL)%LLEAD) &
             CALL ROTATE_WPOT(NP,WH%WPOT(NQ_IN_FULL)%RESPONSEFUN(1,1,1), WH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE( WH%WPOT(NQ)%RESPONSEFUN,1 ), & 
                     WH%KPOINTS_TRANS%CPHASE(1,NQ_IN_FULL), WH%KPOINTS_TRANS%NINDPW(1,NQ_IN_FULL),  &
                     WH%KPOINTS_TRANS%LINV(NQ_IN_FULL), WH%KPOINTS_TRANS%LSHIFT(NQ_IN_FULL))
#ifdef use_shmem
             IF (WH%WPOT(NQ_IN_FULL)%LSHMEM) THEN
                CALLMPI( M_barrier(WH%WPOT(NQ_IN_FULL)%COMM_SHMEM) )
             ENDIF
#endif

          ENDIF
! here one can dump the potential to a file to compare it to e.g. calculation
! without symmetry
!          WRITE(77,'(I10,3F8.3)') NQ_IN_FULL, KPOINTS_FULL%VKPT(:,NQ_IN_FULL)
!          DO I=1,NP
!             WRITE(77,'(32F8.3)') WH%WPOT(NQ_IN_FULL)%RESPONSEFUN(1:NP,I,1)*GRIDHF%NPLWV
!          ENDDO
       ENDIF
       !==========================================================================
       ! this part is only required if the full potential has not been read in
       ! it generates the diagonal part of the potential
       ! it can be executed anyhow, if one desires to set POTFAK correctly
       !==========================================================================
       IF (.NOT. LFULL .OR. .TRUE.) THEN
          IF (WH%WPOT(NQ)%LREALSTORE) THEN
             ! the real value in WH%C stores the potential factor for the cosine transforms
             ! whereas the imag value in WH%C stores that for the sine transform
             DO I=1,NP
                C(I)=CMPLX(WH%WPOT(NQ)%RESPONSER  ((I-1)*2+1,(I-1)*2+1,1), WH%WPOT(NQ)%RESPONSER  ((I-1)*2+2,(I-1)*2+2,1),q)
             ENDDO
             ! at this point, the real value in WH%C stores the potential factor for the cosine transforms
             ! whereas the imag value in WH%C stores that for the sin transform
             ! but only the real part is passed back in POTFAK
             ! so better average over both components (except for G=0)
             DO I=2,NP
                C(I)=(WH%WPOT(NQ)%RESPONSER  ((I-1)*2+1,(I-1)*2+1,1)+WH%WPOT(NQ)%RESPONSER  ((I-1)*2+2,(I-1)*2+2,1))/2
             ENDDO
          ELSE
             DO I=1,NP
                C(I)=WH%WPOT(NQ)%RESPONSEFUN(I,I,1)
             ENDDO
          ENDIF
          
          ! re-index to new k-point
          DO I=1,WGWQ%NGVECTOR
             WH%C(WH%KPOINTS_TRANS%NINDPW(I,NQ_IN_FULL),NQ_IN_FULL)=C(I)
          ENDDO

          ! deallocate response function at the k-point in the IRZ
          IF (.NOT. LFULL) CALL DEALLOCATE_RESPONSEFUN( WH%WPOT(NQ))
       ENDIF
    ENDIF

    WH%LSET(NQ_IN_FULL)=.TRUE.
    POTFAK(1:WGWQ%NGVECTOR)=WH%C(1:WGWQ%NGVECTOR, NQ_IN_FULL)

    IF (ASSOCIATED(WH%WPOT(NQ_IN_FULL)%RESPONSEFUN)) THEN
       LFULL=.TRUE.
    ELSE
       LFULL=.FALSE.
    ENDIF

  END SUBROUTINE GET_WPOT

!*********************************************************************
!
! remove the G=0 component from the stored WPOT and return it
! to the calling routine
!
!*********************************************************************

  FUNCTION GET_WPOT_GZERO(WH)
    IMPLICIT NONE
    TYPE(wpothandle), POINTER :: WH
    REAL(q) :: GET_WPOT_GZERO
  ! local
    INTEGER :: NK

    IF (.NOT. ASSOCIATED(WH)) RETURN

       ! deallocate the potential handle
    DO NK=1,KPOINTS_FULL_ORIG%NKPTS
       IF (  ABS(SUM(KPOINTS_FULL_ORIG%VKPT(:,NK)*KPOINTS_FULL_ORIG%VKPT(:,NK)))<=G2ZERO ) THEN
          IF (ASSOCIATED(WH%WPOT)) THEN
             GET_WPOT_GZERO=WH%WPOT(NK)%RESPONSEFUN(1,1,1)
             WH%WPOT(NK)%RESPONSEFUN(1,1,1)=0
          ELSE IF (ASSOCIATED(WH%C)) THEN
             GET_WPOT_GZERO=WH%C(1, 1)
             WH%C(1, 1)=0
          ENDIF
       ENDIF
    ENDDO


    ! ok this scaling is really odd, but thats used in SET_GFAC_WAVEFUN
    GET_WPOT_GZERO=GET_WPOT_GZERO*GRIDHF%NPLWV &
           *KPOINTS_FULL_ORIG%WTKPT(1)*WH%NKREDLFX*WH%NKREDLFY*WH%NKREDLFZ

  END FUNCTION GET_WPOT_GZERO

!*********************************************************************
!
! write the screened potential to a file
! a slight complication is caused by the possibly subtracted
! bare Hartree Fock potential for q/=0 the bare kernel is stored
! in DATAKE
! the bare singularity correction at q=0 is passed by the calling 
! routine (FSG0)
!
!*********************************************************************

  SUBROUTINE WRITE_WPOT(WPOT, WGWQ, FSG0, NQ, LFOCK_SUBTRACT, FILEBASE_ )

    IMPLICIT NONE
    TYPE (responsefunction) :: WPOT  ! screened two electron potential-HF
    TYPE (wavedes1)         :: WGWQ  ! basis set descriptor
    REAL (q)                :: FSG0  ! HF singularity correction
    INTEGER              NQ          ! q-point for which local field correction are required
    LOGICAL :: LFOCK_SUBTRACT
    CHARACTER(*), OPTIONAL :: FILEBASE_
  ! local
    INTEGER              IERR        ! error status
    CHARACTER (4) :: APP
    INTEGER       :: IU=72, NOMEGA, I, NP
    COMPLEX(q)    :: W( WPOT%NP2, WPOT%NP2)
    CHARACTER(32) :: FILEBASE

    IERR=0
    IF (WPOT%NOMEGA_LOW==1) THEN
       ! WRITE(*,*) 'node ',WGWQ%COMM_INTER%NODE_ME,' writing ',NQ
       NOMEGA=1
       WRITE (APP  , "(4I1)") MOD(NQ/1000,10),MOD(NQ/100,10),MOD(NQ/10,10), MOD(NQ,10)
       FILEBASE="W"
       IF(PRESENT(FILEBASE_)) FILEBASE=TRIM(FILEBASE_)
       OPEN( UNIT=IU, FILE=TRIM(FILEBASE)//APP//".tmp", IOSTAT=IERR, FORM='UNFORMATTED')

       NP=WGWQ%NGVECTOR
       IF (WGWQ%LGAMMA) NP=NP*2
       IF (NP>WPOT%NP2) THEN
          WRITE(0,*)'internal error in WRITE_WPOT: NP>WPOT%NP2 ',NP,WPOT%NP2
          STOP
       ENDIF

       IF (IERR==0) WRITE(IU) NP, 0
       IF (IERR==0) WRITE(IU) WPOT%HEAD(:,:,NOMEGA)
       IF (WPOT%LREALSTORE) THEN
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%WINGR(1:NP,:,NOMEGA)
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%CWINGR(1:NP,:,NOMEGA)
       ELSE
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%WING(1:NP,:,NOMEGA)
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%CWING(1:NP,:,NOMEGA)
       ENDIF

       IF (WPOT%LREALSTORE) THEN
          W=WPOT%RESPONSER  (:,:,NOMEGA)
       ELSE
          W=WPOT%RESPONSEFUN(:,:,NOMEGA)
       ENDIF


       IF (LFOCK_SUBTRACT) THEN
          IF (WPOT%LGAMMA) THEN
             DO I=2,NP
                W(I,I)=W(I,I)+WGWQ%DATAKE(I,1)
             ENDDO
             W(1,1)=W(1,1)+FSG0
          ELSE
             DO I=1,NP
                W(I,I)=W(I,I)+WGWQ%DATAKE(I,1)
             ENDDO
          ENDIF
       ENDIF
       IF (IERR==0) THEN
          IF (WPOT%LREAL) THEN
             WRITE(IU, IOSTAT=IERR) (REAL(W(I,I),q),I=1, NP)
          ELSE
             WRITE(IU, IOSTAT=IERR) (W(I,I),I=1, NP)
          ENDIF
       ENDIF

       CLOSE(IU)
    ENDIF
    CALLMPI( M_sum_i(WGWQ%COMM_INTER, IERR, 1))

  END SUBROUTINE WRITE_WPOT

!*********************************************************************
!
! write the screened potential to a file
! this version writes the entire matrix WPOT(G,G')
!
!*********************************************************************

  SUBROUTINE WRITE_WPOT_FULL(WPOT, WGWQ, FSG0, NQ, LFOCK_SUBTRACT ) 

    IMPLICIT NONE
    TYPE (responsefunction) :: WPOT  ! screened two electron potential-HF
    TYPE (wavedes1)         :: WGWQ  ! basis set descriptor
    REAL (q)                :: FSG0  ! HF singularity correction
    INTEGER              NQ          ! q-point for which local field correction are required
    LOGICAL :: LFOCK_SUBTRACT
  ! local
    INTEGER              IERR        ! error status
    CHARACTER (4) :: APP
    INTEGER       :: IU=72, NOMEGA, I, NP
    COMPLEX(q)    :: W( WPOT%NP2, WPOT%NP2)

    IERR=0
! write full potential at every k-point
    IF (WPOT%NOMEGA_LOW==1) THEN
       NOMEGA=1
       WRITE (APP  , "(4I1)") MOD(NQ/1000,10),MOD(NQ/100,10),MOD(NQ/10,10), MOD(NQ,10)
       OPEN( UNIT=IU, FILE="WFULL"//APP//".tmp", IOSTAT=IERR, FORM='UNFORMATTED')

       NP=WGWQ%NGVECTOR
       IF (WGWQ%LGAMMA) NP=NP*2
       IF (NP>WPOT%NP2) THEN
          WRITE(0,*)'internal error in WRITE_WPOT: NP>WPOT%NP2 ',NP,WPOT%NP2
          STOP
       ENDIF

       IF (IERR==0) WRITE(IU) NP, NP
       IF (IERR==0) WRITE(IU) WPOT%HEAD(:,:,NOMEGA)
       IF (WPOT%LREALSTORE) THEN
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%WINGR(1:NP,:,NOMEGA)
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%CWINGR(1:NP,:,NOMEGA)
       ELSE
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%WING(1:NP,:,NOMEGA)
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) WPOT%CWING(1:NP,:,NOMEGA)
       ENDIF

       IF (WPOT%LREALSTORE) THEN
          W=WPOT%RESPONSER  (:,:,NOMEGA)
       ELSE
          W=WPOT%RESPONSEFUN(:,:,NOMEGA)
       ENDIF

       IF (LFOCK_SUBTRACT) THEN
          IF (WPOT%LGAMMA) THEN
             DO I=2,NP
                W(I,I)=W(I,I)+WGWQ%DATAKE(I,1)
             ENDDO
             W(1,1)=W(1,1)+FSG0
          ELSE
             DO I=1,NP
                W(I,I)=W(I,I)+WGWQ%DATAKE(I,1)
             ENDDO
          ENDIF
       ENDIF
       IF (WPOT%LREAL) THEN
          ! WPOT%LREAL indicates that we are using the gamma point only version
          ! in this case the response function is necessarily real valued
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) REAL(W(1:NP,1:NP),q)
       ELSE
          IF (IERR==0) WRITE(IU, IOSTAT=IERR) W(1:NP,1:NP)
       ENDIF
       CLOSE(IU)
    ENDIF
    CALLMPI( M_sum_i(WGWQ%COMM_INTER, IERR, 1))

  END SUBROUTINE WRITE_WPOT_FULL

!*********************************************************************
!
! read screened potential from the file
! this version always reads from WFULLXXXX.tmp but is capable
! to read both WXXXX.tmp 
! so if you want to read WXXXX.tmp copy it to WFULLXXXX.tmp
!
!*********************************************************************

  SUBROUTINE READ_WPOT(WPOT, WGWQ, NQ, LFULL, IU0, IERR)

    IMPLICIT NONE
    TYPE (responsefunction) :: WPOT  ! screened two electron potential
    TYPE (wavedes1)         :: WGWQ  ! basis set descriptor
    INTEGER              NQ         ! q-point for which local field correction are required
    LOGICAL              LFULL      ! all elements read
    INTEGER              IU0        ! stderr 
    INTEGER              IERR       ! error status
  ! local
    CHARACTER (4) :: APP
    INTEGER       :: N1, N2, NP
    INTEGER :: IU=72, NOMEGA, I, NALLOC
    LOGICAL :: LREAD
    INTEGER :: ierror

    IERR=0
    NOMEGA=1
    WRITE (APP  , "(4I1)") MOD(NQ/1000,10),MOD(NQ/100,10),MOD(NQ/10,10), MOD(NQ,10)

    OPEN( UNIT=IU, FILE="WFULL"//APP//".tmp", STATUS="OLD", IOSTAT=IERR, FORM='UNFORMATTED')
    IF (IERR/=0) THEN
       IF (IU0>=0) WRITE(IU0,*) 'reading now W'//APP//".tmp"
       OPEN( UNIT=IU, FILE="W"//APP//".tmp", STATUS="OLD", IOSTAT=IERR, FORM='UNFORMATTED')
    ELSE
       IF (IU0>=0) WRITE(IU0,*) 'reading now WFULL'//APP//".tmp"
    ENDIF

    NP=WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2
    IF (NP>WPOT%NP2) THEN
       WRITE(0,*)'internal error in READ_WPOT: NP>WPOT%NP2 ',NP,WPOT%NP2
       STOP
    ENDIF

    IF (IERR==0) READ(IU) N1, N2
    ! there are N1 data on the actual file
    ! either these match WPOT%NP2 (old version)
    ! or they match NP
    ! in both cases we can read the file without problems, since
    ! all arrays are allocted as WPOT%NP2 and NP is smaller than WPOT%NP2

    IF ( N1/= WPOT%NP2 .AND. N1/= NP) THEN
       IERR=1
       IF (IU0>=0) WRITE(IU0,'(" ",A,2I10)') "N1 differs from file",N1,WPOT%NP2
       IF (N1<=WPOT%NP2) THEN
          IF (IU0>=0) WRITE(IU0,'(" ",A,2I10)') "trying to continue anyhow"
          IERR=0
       ENDIF
    ENDIF

    IF (IERR==0) READ(IU) WPOT%HEAD(:,:,NOMEGA)
    IF (WPOT%LREALSTORE) THEN
       IF (IERR==0) READ(IU, IOSTAT=IERR) WPOT%WINGR(1:N1,:,NOMEGA)
       IF (IERR==0) READ(IU, IOSTAT=IERR) WPOT%CWINGR(1:N1,:,NOMEGA)
    ELSE
        IF (IERR==0) READ(IU, IOSTAT=IERR) WPOT%WING (1:N1,:,NOMEGA)
        IF (IERR==0) READ(IU, IOSTAT=IERR) WPOT%CWING(1:N1,:,NOMEGA)
    ENDIF

    IF (IERR==0) THEN
       IF (N2==0) THEN
          LFULL=.FALSE.
       ELSE
          LFULL=.TRUE.
       ENDIF
    ENDIF

    IF (IERR==0 .AND. WPOT%LLEAD) THEN
       IF (N2==0) THEN
          ! read only diagonal components
          WPOT%RESPONSEFUN=0
          IF (WPOT%LREALSTORE) THEN
             READ(IU, IOSTAT=IERR) (WPOT%RESPONSER  (I,I,NOMEGA),I=1,N1)
          ELSE
             READ(IU, IOSTAT=IERR) (WPOT%RESPONSEFUN(I,I,NOMEGA),I=1,N1)
          ENDIF
          IF (IERR/=0 .AND. IU0>=0) THEN
             WRITE(IU0,'(" ",A)') "error upon READ of WXXXX.tmp file"
          ENDIF
       ELSE
          ! read all components
          IF (WPOT%LREALSTORE) THEN
             WPOT%RESPONSER(:,:,NOMEGA)=0
             READ(IU, IOSTAT=IERR)  WPOT%RESPONSER  (1:N1,1:N1,NOMEGA)
          ELSE
             WPOT%RESPONSEFUN(:,:,NOMEGA)=0
             READ(IU, IOSTAT=IERR)  WPOT%RESPONSEFUN(1:N1,1:N1,NOMEGA)
          ENDIF
          IF (IERR/=0 .AND. IU0>=0) THEN
             WRITE(IU0,'(" ",A)') "error upon READ of WXXXX.tmp file"
          ENDIF
       ENDIF
    ENDIF

#ifdef use_shmem
    ! force the other cores to wait for root core
    IF (WPOT%LSHMEM)  THEN 
       ! communicate IERR to all shmem nodes
       CALLMPI( M_bcast_i(WPOT%COMM_SHMEM, IERR, 1))
       CALLMPI( M_barrier(WPOT%COMM_SHMEM ) )
    ENDIF
#endif
    CLOSE(IU)

  END SUBROUTINE READ_WPOT

END MODULE wpot
