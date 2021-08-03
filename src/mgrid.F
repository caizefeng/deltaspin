#include "symbol.inc"
! RCS:  $Id: mgrid.F,v 1.5 2003/06/27 13:22:19 kresse Exp kresse $

MODULE mgrid
  USE prec
  USE mpimy

!***********************************************************************
! this module sets up the GRID structure
! the GRID structure contains (among other things) the communication
! patterns for the 3d FFT.
!***********************************************************************


  ! If LPLANE_WISE is set, the data are distributed in real and reciprocal
  ! space plane by plane i.e. one processor holds all elements of
  ! a plane with a specific x index
  ! this  reduces the communication in the FFT considerably
  ! the default for LPLANE_WISE can be set in this file (see below),
  ! or using the flag LPLANE in the INCAR reader

#ifdef plane_wise
  LOGICAL :: LPLANE_WISE=.TRUE.
#else
  LOGICAL :: LPLANE_WISE=.FALSE.
#endif

  ! compatibility modus to vasp.4.4
  ! the flag determine among other things whether the charge at unbalanced
  ! lattice vectors NGX/2+1 are zeroed or not

  LOGICAL LCOMPAT  

  TYPE layout
     INTEGER NP                   ! local number of grid points
     INTEGER NALLOC               ! allocation required
     INTEGER NFAST                ! which index is fast (1-x, 2-y, 3-z)
     INTEGER NCOL                 ! number of columns in the grid
     INTEGER NROW                 ! number of elements in each column
     INTEGER,POINTER :: I2(:)     ! y/z/x-index of each column
     INTEGER,POINTER :: I3(:)     ! z/x/y-index of each column
     INTEGER,POINTER :: INDEX(:,:)! column index for each yz, zx or xy pair
  END TYPE layout

  TYPE grid_map
     INTEGER,POINTER :: N(:)      ! number of elements send by each node
     INTEGER,POINTER :: PTR(:)    ! sum_j=1,I N(j)
     INTEGER,POINTER :: RMT(:)    ! remote address for send  (shmem t3d)
     INTEGER,POINTER :: TBL(:)    ! address of each element
     ! inverse transformation (i.e. receiver information)
     INTEGER,POINTER :: NI(:)
     INTEGER,POINTER :: PTRI(:)
     INTEGER,POINTER :: TBLI(:)
     INTEGER,POINTER :: RMTI(:)
     LOGICAL LOCAL                ! all information is local
     LOGICAL LOCAL_COPY           ! no data redistribution required
  END TYPE grid_map

  TYPE grid_3d
     !only  GRID
     INTEGER NGX,NGY,NGZ          ! number of grid points in x,y,z
     INTEGER NGX_rd,NGY_rd,NGZ_rd ! in the complex mode the _rd values
     ! are equal to NGX, NGY, NGZ
     ! if a real to complex FFT is used only half of the data
     ! are stored in one direction and the corresponding NG?_rd is 
     ! set to (NG?+1)/2
     INTEGER NPLWV                ! total number of grid points NGX*NGY*NGZ
     INTEGER MPLWV                ! allocation in complex words required to do in place FFT's
     INTEGER NGPTAR(3)            ! equivalent to /(NGX,NGY,NGZ/)
     INTEGER,POINTER :: LPCTX(:)  ! loop counters in x
     INTEGER,POINTER :: LPCTY(:)  ! loop counters in y
     INTEGER,POINTER :: LPCTZ(:)  ! loop counters in z
     ! loop counters, in which the unbalanced contribution is zeroed
     INTEGER,POINTER :: LPCTX_(:) ! loop counters in x
     INTEGER,POINTER :: LPCTY_(:) ! loop counters in y
     INTEGER,POINTER :: LPCTZ_(:) ! loop counters in z
     ! reciprocal space layout (x is always  fast index)
     TYPE(layout)    :: RC
     ! intermediate layout (y is always  fast index, used only in parallel version)
     TYPE(layout)    :: IN
     ! real space layout   (x or z is the fast index)
     TYPE(layout)    :: RL
     ! real space layout for FFT  (x or z is the fast index)
     ! this structure is usually equivalent to RL (and hence points to RL)
     ! only if the serial version is used for the FFT of wavefunctions,
     ! the structure differs from RL for the GRID_SOFT structure
     ! in this case, the FFT has z as fast index as required for the parallel FFT
     ! but the RL structure has x as fast index to be compatible to the FFT
     ! of wavefunctions
     TYPE(layout), POINTER  :: RL_FFT
     ! information only required for real space representation
     INTEGER NGZ_complex          ! number of grid points for z fast
     ! mapping for parallel version
     TYPE(grid_map)  :: RC_IN     ! recip -> intermediate
     TYPE(grid_map)  :: IN_RL     ! intermediate -> real space
     TYPE(communic), POINTER :: COMM,COMM_KIN,COMM_KINTER
     LOGICAL         :: LREAL     ! are data stored as complex or real numbers in real space
     LOGICAL         :: REAL2CPLX ! real to complex FFT
     REAL(q), POINTER:: FFTSCA(:,:) ! scaling factors for real to complex fft (wavefunction fft) in reciprocal space (only defined for PW in cutoff-sphere)
     REAL(q), POINTER:: FFTWEIGHT(:)! for real space version, weight of each coefficient in reciprocal space for all coefficients
     ! plane waves with N1=0 (NX=0) within the cutoff sphere correspond to two 
     ! G vectors on the 3D FFT grid
     ! IND_IN_SPHERE stores the PW indices (within the sphere) with N1=0
     ! NINDPWCONJG stores the position where the conjugated coefficient needs to be stored
     REAL(q), POINTER:: NINDPWCONJG(:)
     REAL(q), POINTER:: IND_IN_SPHERE(:)
  END TYPE grid_3d
  ! comments:
  ! REAL2CPLX determines wether a real to complex FFT is use
  ! LREAL     determines wether the data in real space are 
  !           stored in real a complex array
  !
  ! the decision whether a serial of parallel FFT is performed
  ! is presently decided by the GRID%RL%NFAST tag
  ! if GRID%RL%NFAST =1 -> serial FFT
  ! if GRID%RL%NFAST =3 -> parallel FFT

  !
  ! transition table used to go from a large to a small grid
  ! or vice versa
  !
  TYPE transit
     INTEGER,POINTER :: IND1(:)   ! fast index transition table
     INTEGER,POINTER :: INDCOL(:) ! column to column transition table
  END TYPE transit

CONTAINS


!****************  SUBROUTINE NULLIFY_GRD ******************************
!
!  nullify all entries in the GRID structure
!
!***********************************************************************

  SUBROUTINE  NULLIFY_GRD(GRID)
    IMPLICIT NONE

    TYPE (grid_3d), TARGET :: GRID

    GRID%RL%NALLOC=0
    GRID%RC%NALLOC=0
    GRID%IN%NALLOC=0

    NULLIFY(GRID%RC%I2)
    NULLIFY(GRID%RC%I3)
    NULLIFY(GRID%RC%INDEX)  ! not used
    NULLIFY(GRID%IN%I2)
    NULLIFY(GRID%IN%I3)
    NULLIFY(GRID%IN%INDEX)  ! not used
    NULLIFY(GRID%RL%I2)
    NULLIFY(GRID%RL%I3)
    NULLIFY(GRID%RL%INDEX)
! on the long run this routine should also 
! nullify the following arrays
!    NULLIFY(GRID%IND_IN_SPHERE)
!    NULLIFY(GRID%NINDPWCONJG)
!    NULLIFY(GRID%FFTSCA)

! FFT weight is initialized by GEN_GRID_HF
! which is only called once from SETUP_FOCK
    NULLIFY(GRID%FFTWEIGHT)

    GRID%RL_FFT=>GRID%RL

  END SUBROUTINE NULLIFY_GRD


!****************  SUBROUTINE INIGRD ***********************************
!
! allocate and initialize loop counters in the grid structure
! this is a sort of minimal setup 
!  (see next subroutine for details how LPCTX,Y,Z is set)
!
!***********************************************************************
  
  SUBROUTINE INILGRD(NGX,NGY,NGZ,GRID)
    USE base
    IMPLICIT NONE

    INTEGER NGX,NGY,NGZ
    TYPE (grid_3d) GRID

    GRID%NGX=NGX
    GRID%NGY=NGY
    GRID%NGZ=NGZ
    GRID%NPLWV =NGX*NGY*NGZ
    GRID%MPLWV =0
    ALLOCATE(GRID%LPCTX(NGX), GRID%LPCTY(NGY), GRID%LPCTZ(NGZ))
    ALLOCATE(GRID%LPCTX_(NGX),GRID%LPCTY_(NGY),GRID%LPCTZ_(NGZ))
    GRID%NGPTAR(1)=NGX
    GRID%NGPTAR(2)=NGY
    GRID%NGPTAR(3)=NGZ
    CALL INILPC(NGX,NGY,NGZ,GRID%LPCTX(1:NGX), GRID%LPCTY(1:NGY), GRID%LPCTZ(1:NGZ))
    CALL INILPC(NGX,NGY,NGZ,GRID%LPCTX_(1:NGX),GRID%LPCTY_(1:NGY),GRID%LPCTZ_(1:NGZ))
    GRID%LPCTX_(NGX/2+1) = 0
    GRID%LPCTY_(NGY/2+1) = 0
    GRID%LPCTZ_(NGZ/2+1) = 0
    RETURN

  END  SUBROUTINE INILGRD


!****************  SUBROUTINE INILPC ***********************************
!
! initialize the loop counters LPCTX,LPCTY,LPCTZ etc that
! label the number of the reciprocal lattice vectors in the x,y,z
! directions, respectively. for the x direction the reciprocal lattice
! vectors corresponding to the first,second,...,ngxth elements in all
! of the reciprocal lattice arrays are 0,1,..,(NGX/2),-((NGX/2-1),..,-1
! times the x reciprocal lattice vector
! only called by INI
!
!***********************************************************************

  SUBROUTINE INILPC (NGX,NGY,NGZ,LPCTX,LPCTY,LPCTZ)
    IMPLICIT NONE
    INTEGER NGX, NGY, NGZ
    INTEGER LPCTX(NGX),LPCTY(NGY),LPCTZ(NGZ)
  ! local
    INTEGER NX, NY, NZ

    DO 100 NX=1,(NGX/2)+1
       LPCTX(NX)=NX-1
100 ENDDO
    DO 110 NX=(NGX/2)+2,NGX
       LPCTX(NX)=NX-1-NGX
110 ENDDO
    DO 120 NY=1,(NGY/2)+1
       LPCTY(NY)=NY-1
120 ENDDO
    DO 130 NY=(NGY/2)+2,NGY
       LPCTY(NY)=NY-1-NGY
130 ENDDO
    DO 140 NZ=1,(NGZ/2)+1
       LPCTZ(NZ)=NZ-1
140 ENDDO
    DO 150 NZ=(NGZ/2)+2,NGZ
       LPCTZ(NZ)=NZ-1-NGZ
150 ENDDO
    RETURN
  END SUBROUTINE INILPC

!****************  SUBROUTINE DEALLOC_GRD ******************************
!
!  deallocate a GRID structure
!
!***********************************************************************

  SUBROUTINE  DEALLOC_GRD(GRID)
    IMPLICIT NONE
    TYPE (grid_3d) GRID

    IF (ASSOCIATED(GRID%RC%I2)) DEALLOCATE(GRID%RC%I2)
    IF (ASSOCIATED(GRID%RC%I3)) DEALLOCATE(GRID%RC%I3)
    IF (ASSOCIATED(GRID%RC%INDEX)) DEALLOCATE(GRID%RC%INDEX) ! not used
    IF (ASSOCIATED(GRID%IN%I2)) DEALLOCATE(GRID%IN%I2)
    IF (ASSOCIATED(GRID%IN%I3)) DEALLOCATE(GRID%IN%I3)
    IF (ASSOCIATED(GRID%IN%INDEX)) DEALLOCATE(GRID%IN%INDEX) ! not used
    IF (ASSOCIATED(GRID%RL%I2)) DEALLOCATE(GRID%RL%I2)
    IF (ASSOCIATED(GRID%RL%I3)) DEALLOCATE(GRID%RL%I3)
    IF (ASSOCIATED(GRID%RL%INDEX)) DEALLOCATE(GRID%RL%INDEX)

! on the long run this routine should deallocate the following arrays
! attention: sometimes GRID_TMP=GRID is used, and then DEALLOC_GRD is called
! furthermore FFTINI (which initializes these fields) is not called in all
! required instances
!   IF (ASSOCIATED(GRID%IND_IN_SPHERE)) DEALLOCATE(GRID%IND_IN_SPHERE)
!   IF (ASSOCIATED(GRID%NINDPWCONJG))   DEALLOCATE(GRID%NINDPWCONJG)
!   IF (ASSOCIATED(GRID%FFTSCA)) DEALLOATE(GRID%FFTSCA)

  END SUBROUTINE DEALLOC_GRD


!***********************************************************************
!  Generate layout for a 3d grid (optionally a subgrid of GRID)
!
! setup data distribution for a subgrid contained in a supergrid
! it is guaranteed that the new (small) grid has a data distribution that
! ensures that if one column (in reciprocal space) on the large grid
! (supergrid) is on proc x the corresponding column is also on proc x 
! on the small grid (subgrid)
! if a subgrid is generated a transition table
!  for going from one to the other grid is also created
!
! if the 3d grid has to be used also in real space (FFTs) 
! LREAL must be set to .TRUE.
! grids in real space have the same date distribution,
! if and only if the dimension of the grids are the same
!
!***********************************************************************

!***********************************************************************
!
!  GRID possibly real data layout (realmode) in the direct space
!  and complex in the reciprocal space (depends on the selection in 
!   the makefile and symbol.inc)
!
!***********************************************************************

  SUBROUTINE GEN_RC_GRID(GRID)
    IMPLICIT NONE

    TYPE (transit)     TRANS
    TYPE (grid_3d)     GRID

    GRID%NGX_rd=xm(GRID%NGX)
    GRID%NGY_rd=GRID%NGY
    GRID%NGZ_rd=zm(GRID%NGZ)
#ifdef realmode
    GRID%LREAL =.TRUE.
    GRID%REAL2CPLX =.TRUE.
#else
    GRID%LREAL =.FALSE.
    GRID%REAL2CPLX =.FALSE.
#endif
    CALL GEN_SUB_GRID_(GRID, GRID, TRANS,.TRUE.,.FALSE.)

  END SUBROUTINE GEN_RC_GRID


!***********************************************************************
!
!  GRID with  complex data layout in direct space
!  and complex layout in the reciprocal space
!
!***********************************************************************

  SUBROUTINE GEN_GRID(GRID)

    IMPLICIT NONE

    TYPE (transit)     TRANS
    TYPE (grid_3d)     GRID

    GRID%NGX_rd=GRID%NGX
    GRID%NGY_rd=GRID%NGY
    GRID%NGZ_rd=GRID%NGZ

    GRID%LREAL=.FALSE.
    GRID%REAL2CPLX =.FALSE.

    CALL GEN_SUB_GRID_(GRID,GRID, TRANS,.TRUE.,.FALSE.)

  END SUBROUTINE GEN_GRID

!***********************************************************************
!
!  GRID_SUB with a possibly real data layout in direct space
!  and complex layout in the reciprocal space
!  GRID_SUB is possibly a subgrid of GRID
!
!***********************************************************************

  SUBROUTINE GEN_RC_SUB_GRID(GRID_SUB, GRID, TRANS, LREAL, LSUB)
    IMPLICIT NONE

    LOGICAL LREAL,LSUB
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SUB
    TYPE (transit)     TRANS

    GRID_SUB%NGX_rd=xm(GRID_SUB%NGX)
    GRID_SUB%NGY_rd=GRID_SUB%NGY
    GRID_SUB%NGZ_rd=zm(GRID_SUB%NGZ)

#ifdef realmode
    GRID_SUB%LREAL =.TRUE.
    GRID_SUB%REAL2CPLX =.TRUE.
#else
    GRID_SUB%LREAL =.FALSE.
    GRID_SUB%REAL2CPLX =.FALSE.
#endif      

    CALL GEN_SUB_GRID_(GRID_SUB, GRID, TRANS, LREAL, LSUB)
  END SUBROUTINE GEN_RC_SUB_GRID

!***********************************************************************
!
!  GRID_SUB with complex data layout in direct space
!  and complex layout in the reciprocal space
!  GRID_SUB is possibly a subgrid of GRID
!
!***********************************************************************

  SUBROUTINE GEN_SUB_GRID(GRID_SUB, GRID, TRANS, LREAL, LSUB)
    IMPLICIT NONE

    LOGICAL LREAL,LSUB
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SUB
    TYPE (transit)     TRANS

    GRID_SUB%NGX_rd=GRID_SUB%NGX
    GRID_SUB%NGY_rd=GRID_SUB%NGY
    GRID_SUB%NGZ_rd=GRID_SUB%NGZ

    GRID_SUB%LREAL=.FALSE.
    GRID_SUB%REAL2CPLX =.FALSE.

    CALL GEN_SUB_GRID_(GRID_SUB, GRID, TRANS, LREAL, LSUB)
  END SUBROUTINE GEN_SUB_GRID

!***********************************************************************
!
!  main routine for the generation of a GRID structure
!
!***********************************************************************

  SUBROUTINE GEN_SUB_GRID_(GRID_SUB,GRID, TRANS, LREAL, LSUB)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    LOGICAL LREAL,LSUB
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SUB
    TYPE (transit)     TRANS

    CALL NULLIFY_GRD(GRID_SUB)
#ifdef MPI
!-----------------------------------------------------------------------
! parallel version sub grid
!-----------------------------------------------------------------------
    NODE_ME=GRID_SUB%COMM%NODE_ME
    IF (LSUB) THEN
       GRID_SUB%RC%NFAST= 1
       GRID_SUB%RC%NROW = GRID_SUB%NGX_rd
       ALLOCATE(GRID_SUB%RC%I2( GRID%RC%NCOL )) ! always save
       ALLOCATE(GRID_SUB%RC%I3( GRID%RC%NCOL ))

       NCOL=GRID%RC%NCOL

       IND=0
       DO NC=1,NCOL
          N2=GRID%RC%I2(NC) ; L2=GRID%LPCTY(N2)
          N3=GRID%RC%I3(NC) ; L3=GRID%LPCTZ(N3)
          !     is that one in the sub grid
          CALL SRCH_COL2(L2,L3,GRID_SUB,N2P,N3P)
          IF (N2P/=0 .AND. N3P/=0 ) THEN
             IND=IND+1
             GRID_SUB%RC%I2(IND)=N2P
             GRID_SUB%RC%I3(IND)=N3P
          ENDIF
       ENDDO
       GRID_SUB%RC%NCOL = IND
       IF (GRID_SUB%RC%NCOL >GRID%RC%NCOL) THEN
          WRITE(*,*) 'GEN_GRID_SUB: internal error',GRID_SUB%RC%NCOL
          STOP
       ENDIF
       GRID_SUB%RC%NP    = GRID_SUB%RC%NCOL*GRID_SUB%RC%NROW
       GRID_SUB%RC%NALLOC= GRID_SUB%RC%NCOL*GRID_SUB%RC%NROW
    ELSE
       CALL REC_STDLAY(GRID_SUB)
    ENDIF

    IF (LREAL) THEN
       CALL INTER_STDLAY(GRID_SUB)
       CALL REAL_STDLAY (GRID_SUB, LPARALLEL=.TRUE.)
    ENDIF

#else
!-----------------------------------------------------------------------
! no problems with conventional version
! all grid points are contained within 3d box
!-----------------------------------------------------------------------
    GRID_SUB%RC%NFAST= 1
    GRID_SUB%RC%NCOL = GRID_SUB%NGZ_rd*GRID_SUB%NGY
    GRID_SUB%RC%NROW = GRID_SUB%NGX_rd
    ALLOCATE(GRID_SUB%RC%I2( GRID_SUB%RC%NCOL ))
    ALLOCATE(GRID_SUB%RC%I3( GRID_SUB%RC%NCOL ))
    IND=1
    DO N3=1,GRID_SUB%NGZ_rd
       DO N2=1,GRID_SUB%NGY
          GRID_SUB%RC%I2(IND)=N2
          GRID_SUB%RC%I3(IND)=N3
          IND=IND+1
       ENDDO
    ENDDO
    GRID_SUB%RC%NP    = GRID_SUB%RC%NCOL*GRID_SUB%RC%NROW
    GRID_SUB%RC%NALLOC= GRID_SUB%RC%NCOL*GRID_SUB%RC%NROW

    IF (LREAL) THEN
       CALL REAL_STDLAY (GRID_SUB, LPARALLEL=.FALSE.)
    ENDIF
#endif
    ! create transition table for fast index (always x) quite simple
    IF (LSUB) THEN
       CALL GEN_TRANS(GRID_SUB, GRID, TRANS)
    ENDIF
    GRID_SUB%MPLWV=__align_DIM(MAX(GRID_SUB%RC%NALLOC ,GRID_SUB%IN%NALLOC , GRID_SUB%RL%NALLOC))
    DWRITE 'grid set up return',NODE_ME,GRID_SUB%NGX,GRID_SUB%NGY,GRID_SUB%NGZ,GRID_SUB%MPLWV


  END SUBROUTINE GEN_SUB_GRID_

  !
  ! helper subroutine to search a specific column in a sub grid
  ! must succeed
  !
  SUBROUTINE SRCH_COL1(L2S,L3S,GRID,NC2)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

    NCOL=GRID%RC%NCOL
    DO NC=1,NCOL
       N2=GRID%RC%I2(NC) ; L2=GRID%LPCTY(N2)
       N3=GRID%RC%I3(NC) ; L3=GRID%LPCTZ(N3)
       IF (L2==L2S .AND. L3==L3S) THEN
          NC2=NC
          RETURN
       ENDIF
    ENDDO
    WRITE(*,*)'SRCH_COL1: internal error',L2S,L3S
    WRITE(*,'(20I6)') GRID%LPCTY
    WRITE(*,*)
    WRITE(*,'(20I6)') GRID%LPCTZ
    STOP
  END SUBROUTINE SRCH_COL1

  !
  ! helper subroutine to check whether one specific column is in a
  ! sub grid: on sucess N2P and N3P is set
  !
  SUBROUTINE SRCH_COL2(L2,L3,GRID,N2,N3)
    USE prec
    USE mpimy

    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

    DO N3=1,GRID%NGZ_rd
       DO N2=1,GRID%NGY
          IF (L2==GRID%LPCTY(N2) .AND. L3==GRID%LPCTZ(N3)) RETURN
       ENDDO
    ENDDO
    N2=0
    N3=0

  END SUBROUTINE SRCH_COL2

!***********************************************************************
!
! generate a transition table in reciprocal space between 
! to grid structures
!
!***********************************************************************

  SUBROUTINE GEN_TRANS(GRID_SUB, GRID, TRANS)
    USE prec
    IMPLICIT NONE
    TYPE (grid_3d)     GRID
    TYPE (grid_3d)     GRID_SUB
    TYPE (transit)     TRANS

    INTEGER NGX, N1, NCOL, NC, N2, N3, L2, L3, NCP
    NGX=GRID_SUB%NGX

    IF (NGX>GRID%NGX) THEN
       WRITE(*,*) 'internal error in GEN_TRANS: GRID_SUB is not a subgrid of GRID'
       STOP
    ENDIF

    ALLOCATE (TRANS%IND1(NGX))
    
    DO N1=1,(NGX/2)+1
       TRANS%IND1(N1)=N1
    ENDDO
    DO N1=(NGX/2)+2,NGX
       TRANS%IND1(N1)=N1-NGX+GRID%NGX
    ENDDO
    !
    ! create transition table for columns
    NCOL=GRID_SUB%RC%NCOL
    ALLOCATE (TRANS%INDCOL(NCOL))
    
    DO NC=1,NCOL
       N2=GRID_SUB%RC%I2(NC) ; L2=GRID_SUB%LPCTY(N2)
       N3=GRID_SUB%RC%I3(NC) ; L3=GRID_SUB%LPCTZ(N3)
       CALL SRCH_COL1(L2,L3,GRID,NCP)
       TRANS%INDCOL(NC)=NCP
    ENDDO

  END SUBROUTINE GEN_TRANS

!***********************************************************************
!
!  GRIDHF describes the FFT grids for overlap densities 
!    (phi_k(r)  phi_k'(r)) in VASP
!  in principle this could be done with GRID_FOCK but for 
!  historic reasons there is a seperate grid for densities
!
!  Actually there is a subtle reason for this for the *Gamma*
!  point only version:
!  VASP stores the orbitals in real space always as complex arrays
!  (the imaginary part then being strictly zero)
!  GRIDHF, however, assumes for the Gamma point version that
!  the overlap density is stored in a real valued array
!  usually declared as GDEF (see e.g. PW_CHARGE_TRACE)
!
!  Concerning the routine:
!  in the parallel version, the serial FFT is applied if the GRID
!  structure uses a serial FFT 
!  otherwise the parallel FFT is applied
!
!***********************************************************************

  SUBROUTINE GEN_GRID_HF(GRIDHF, GRID)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRIDHF   ! 3d FFT grid for charges and potentials
    TYPE (grid_3d)     GRID     ! 3d FFT grid for orbitals
    TYPE (transit)     TRANS

    CALL NULLIFY_GRD(GRIDHF)

#ifdef gammareal
#ifndef realmode
    WRITE(0,*) 'internal error in FOCK_ACC: GRIDHF is not real'
    WRITE(0,*) ' the gamma only version in combination with Hartree-Fock requires'
    WRITE(0,*) ' to compile the code using -DNGXhalf or -DNGZhalf (mpiversion)'
    STOP
#endif
#endif

#ifdef MPI
!-----------------------------------------------------------------------
! serial FFT for orbitals (GRID%RL%NFAST ==1) despite compiled for parallel
! processing, this is the difficult case
!-----------------------------------------------------------------------
    IF (GRID%RL%NFAST==1) THEN
       ! now check that only one node contains all orbital data
       IF (GRID%COMM%NCPU/=1) THEN
          WRITE(0,*) 'internal ERROR in SET_RL_GRID: GRID contains more than one CPU', GRID%COMM%NCPU
          STOP
       ENDIF
       ! now check that the data layout is strictly sequential in real space
       IND=0
       DO N3=1,GRID%NGZ
          DO N2=1,GRID%NGY
             IND=IND+1
             IF (GRID%RL%I2(IND)/=N2 .OR. GRID%RL%I3(IND)/=N3) THEN
                WRITE(0,*) 'internal ERROR in GEN_GRID_HF: orbital GRID not sequential', IND,& 
                     GRID%RL%I2(IND),N2,GRID%RL%I3(IND),N3
                STOP
             ENDIF
          ENDDO
       ENDDO
#if defined(gammareal)
       GRIDHF%NGX_rd=GRIDHF%NGX/2+1
       GRIDHF%NGY_rd=GRIDHF%NGY
       GRIDHF%NGZ_rd=GRIDHF%NGZ
       GRIDHF%LREAL =.TRUE.
       GRIDHF%REAL2CPLX =.TRUE.
#else
       GRIDHF%NGX_rd=GRIDHF%NGX
       GRIDHF%NGY_rd=GRIDHF%NGY
       GRIDHF%NGZ_rd=GRIDHF%NGZ
       GRIDHF%LREAL=.FALSE.
       GRIDHF%REAL2CPLX =.FALSE.
#endif
       GRIDHF%RC%NFAST= 1
       GRIDHF%RC%NCOL = GRIDHF%NGZ*GRIDHF%NGY
       GRIDHF%RC%NROW = GRIDHF%NGX_rd
       ALLOCATE(GRIDHF%RC%I2( GRIDHF%RC%NCOL ))
       ALLOCATE(GRIDHF%RC%I3( GRIDHF%RC%NCOL ))
       IND=1
       DO N3=1,GRIDHF%NGZ_rd
          DO N2=1,GRIDHF%NGY
             GRIDHF%RC%I2(IND)=N2
             GRIDHF%RC%I3(IND)=N3
             IND=IND+1
          ENDDO
       ENDDO
       GRIDHF%RC%NP    = GRIDHF%RC%NCOL*GRIDHF%RC%NROW
       GRIDHF%RC%NALLOC= GRIDHF%RC%NCOL*GRIDHF%RC%NROW

       CALL REAL_STDLAY (GRIDHF, LPARALLEL=.FALSE.)
    ELSE
#endif
!-----------------------------------------------------------------------
! conventional version: data distribution in GRIDHF
! ) parallel in MPI
! ) serial non-MPI 
!-----------------------------------------------------------------------
       GRIDHF%NGX_rd=GRIDHF%NGX
       GRIDHF%NGY_rd=GRIDHF%NGY
       GRIDHF%NGZ_rd=GRIDHF%NGZ
! gamma-point only the exchange potential is real
#ifdef gammareal
       GRIDHF%LREAL =.TRUE.
       GRIDHF%REAL2CPLX =.TRUE.
#ifdef MPI
       GRIDHF%NGZ_rd=GRIDHF%NGZ/2+1
#else
       GRIDHF%NGX_rd=GRIDHF%NGX/2+1
#endif
#else
       GRIDHF%LREAL =.FALSE.
       GRIDHF%REAL2CPLX =.FALSE.
#endif
       
       CALL GEN_SUB_GRID_(GRIDHF, GRIDHF, TRANS,.TRUE.,.FALSE.)
! allign GRIDHF with GRID in real space
       CALL SET_RL_GRID(GRIDHF,GRID)
#ifdef MPI
    ENDIF
#endif

    GRIDHF%MPLWV=__align_DIM(MAX(GRIDHF%RC%NALLOC ,GRIDHF%IN%NALLOC , GRIDHF%RL%NALLOC))

    CALL FFTWEIGHT_INI(GRIDHF)

    DWRITE 'grid set up return',NODE_ME,GRIDHF%NGX,GRIDHF%NGY,GRIDHF%NGZ,GRIDHF%MPLWV
  END SUBROUTINE GEN_GRID_HF


!***********************************************************************
!
!  generate a GRIDpadded structure that has a 'similar data layout' as an
!  original GRID structure
!  'similar data layout' here simply means:
!  o if GRID uses a parallel data layout GRIDpadded does as well
!  o if GRID uses a serial data layout GRID does as well
!
!***********************************************************************

  SUBROUTINE GEN_GRIDpadded(GRIDpadded, GRID)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRIDpadded
    TYPE (grid_3d)     GRID
    TYPE (transit)     TRANS

    CALL NULLIFY_GRD(GRIDpadded)

#ifdef MPI
!-----------------------------------------------------------------------
! serial version, if GRID%RL%NFAST ==1, despite compiled for parallel
! processing, this is the difficult special case
!-----------------------------------------------------------------------
    IF (GRID%RL%NFAST==1) THEN
#if defined(gammareal)
       GRIDpadded%NGX_rd=GRIDpadded%NGX/2+1
       GRIDpadded%NGY_rd=GRIDpadded%NGY
       GRIDpadded%NGZ_rd=GRIDpadded%NGZ
       GRIDpadded%LREAL =.TRUE.
       GRIDpadded%REAL2CPLX =.TRUE.
#else
       GRIDpadded%NGX_rd=GRIDpadded%NGX
       GRIDpadded%NGY_rd=GRIDpadded%NGY
       GRIDpadded%NGZ_rd=GRIDpadded%NGZ
       GRIDpadded%LREAL=.FALSE.
       GRIDpadded%REAL2CPLX =.FALSE.
#endif
       GRIDpadded%RC%NFAST= 1
       GRIDpadded%RC%NCOL = GRIDpadded%NGZ*GRIDpadded%NGY
       GRIDpadded%RC%NROW = GRIDpadded%NGX_rd
       ALLOCATE(GRIDpadded%RC%I2( GRIDpadded%RC%NCOL ))
       ALLOCATE(GRIDpadded%RC%I3( GRIDpadded%RC%NCOL ))
       IND=1
       DO N3=1,GRIDpadded%NGZ_rd
          DO N2=1,GRIDpadded%NGY
             GRIDpadded%RC%I2(IND)=N2
             GRIDpadded%RC%I3(IND)=N3
             IND=IND+1
          ENDDO
       ENDDO
       GRIDpadded%RC%NP    = GRIDpadded%RC%NCOL*GRIDpadded%RC%NROW
       GRIDpadded%RC%NALLOC= GRIDpadded%RC%NCOL*GRIDpadded%RC%NROW

       CALL REAL_STDLAY (GRIDpadded, LPARALLEL=.FALSE.)
    ELSE
#endif
!-----------------------------------------------------------------------
! standard layout data distribution in GRIDpadded
! e.g. parallel for MPI, serial for non-MPI)
!-----------------------------------------------------------------------
       GRIDpadded%NGX_rd=GRIDpadded%NGX
       GRIDpadded%NGY_rd=GRIDpadded%NGY
       GRIDpadded%NGZ_rd=GRIDpadded%NGZ
! gamma-point only the exchange potential is real
#ifdef gammareal
       GRIDpadded%LREAL =.TRUE.
       GRIDpadded%REAL2CPLX =.TRUE.
#ifdef MPI
       GRIDpadded%NGZ_rd=GRIDpadded%NGZ/2+1
#else
       GRIDpadded%NGX_rd=GRIDpadded%NGX/2+1
#endif
#else
       GRIDpadded%LREAL =.FALSE.
       GRIDpadded%REAL2CPLX =.FALSE.
#endif
       
       CALL GEN_SUB_GRID_(GRIDpadded, GRIDpadded, TRANS,.TRUE.,.FALSE.)
! standard data layout for GRIDpadded in reciprocal space
       CALL REAL_STDLAY (GRIDpadded, LPARALLEL=(GRID%RL%NFAST/=1))
#ifdef MPI
    ENDIF
#endif

    GRIDpadded%MPLWV=__align_DIM(MAX(GRIDpadded%RC%NALLOC ,GRIDpadded%IN%NALLOC , GRIDpadded%RL%NALLOC))
    DWRITE 'grid set up return',NODE_ME,GRIDpadded%NGX,GRIDpadded%NGY,GRIDpadded%NGZ,GRIDpadded%MPLWV

  END SUBROUTINE GEN_GRIDpadded

!*************************SUBROUTINE REC_STDLAY   **********************
!
! setup reciprocal layout for FFT only parallel version
! columns are distributed among procressors in a round robin fashion
!
!***********************************************************************

  SUBROUTINE REC_STDLAY(GRID)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID
#ifdef MPI
    NCOL_TOT= GRID%NGZ_rd*GRID%NGY
    NCOL_MAX=(NCOL_TOT+GRID%COMM%NCPU-1)/GRID%COMM%NCPU
    ALLOCATE(GRID%RC%I2( NCOL_MAX ))
    ALLOCATE(GRID%RC%I3( NCOL_MAX ))

    GRID%RC%NFAST= 1
    GRID%RC%NCOL = 0
    GRID%RC%NROW = GRID%NGX_rd
    ! distribute columns among proc
    IND=0
    DO N3=1,GRID%NGZ_rd
       DO N2=1,GRID%NGY
          NODE_TARGET =MOD(IND,GRID%COMM%NCPU)+1
          IND=IND+1
          IF (NODE_TARGET == GRID%COMM%NODE_ME) THEN
             N=GRID%RC%NCOL
             N=N+1
             GRID%RC%NCOL=N
             GRID%RC%I2(N)=N2 ! I2 contains y index
             GRID%RC%I3(N)=N3 ! I3      the z index
          ENDIF
       ENDDO
    ENDDO
    IF (GRID%RC%NCOL >NCOL_MAX) THEN
       WRITE(*,*) 'REC_STDLAY: internal error',NCOL_MAX,GRID%RC%NCOL
       STOP
    ENDIF
    GRID%RC%NP    = GRID%RC%NCOL*GRID%RC%NROW
    GRID%RC%NALLOC= GRID%RC%NCOL*GRID%RC%NROW
#else
    WRITE(*,*)'internal ERROR: REC_STDLAY called in serial mode'
    STOP
#endif
    RETURN
  END SUBROUTINE REC_STDLAY

!*************************SUBROUTINE INTER_STDLAY **********************
!
! setup intermediate layout for FFT only parallel version
! columns are distributed among procressors in a round robin fashion
!
!***********************************************************************

  SUBROUTINE INTER_STDLAY(GRID)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID
#ifdef MPI

    NCOL_TOT= GRID%NGX_rd*GRID%NGZ_rd
    NCOL_MAX=(NCOL_TOT+GRID%COMM%NCPU-1)/GRID%COMM%NCPU
    IF (LPLANE_WISE) THEN
       NCOL_MAX=GRID%NGZ_rd*((GRID%NGX_rd+GRID%COMM%NCPU-1)/GRID%COMM%NCPU)
    ENDIF


    ALLOCATE(GRID%IN%I2( NCOL_MAX ))
    ALLOCATE(GRID%IN%I3( NCOL_MAX ))

    GRID%IN%NFAST= 2
    GRID%IN%NCOL = 0
    GRID%IN%NROW = GRID%NGY
    ! distribute planes among proc
    IND=0
    DO N3=1,GRID%NGX_rd
       DO N2=1,GRID%NGZ_rd
          IF( LPLANE_WISE ) THEN
             NODE_TARGET=MOD(N3-1,GRID%COMM%NCPU)+1
          ELSE
             NODE_TARGET=MOD(IND,GRID%COMM%NCPU)+1
          ENDIF

          IND=IND+1
          IF (NODE_TARGET == GRID%COMM%NODE_ME) THEN
             N=GRID%IN%NCOL
             N=N+1
             GRID%IN%NCOL=N
             GRID%IN%I2(N)=N2 ! I2 contains z index
             GRID%IN%I3(N)=N3 ! I3      the x index
          ENDIF
       ENDDO
    ENDDO
    IF (GRID%IN%NCOL >NCOL_MAX) THEN
       WRITE(*,*) 'INTER_STDLAY: internal error',NCOL_MAX,GRID%IN%NCOL
       STOP
    ENDIF
    GRID%IN%NP    = GRID%IN%NCOL*GRID%IN%NROW
    GRID%IN%NALLOC= GRID%IN%NCOL*GRID%IN%NROW
#else
    WRITE(*,*)'internal ERROR: INTER_STDLAY was called in serial mode'
    STOP
#endif
  END SUBROUTINE INTER_STDLAY

!*************************SUBROUTINE REAL_STDLAY ***********************
!
! setup standard layout for real space
! this is quite simple
! all points are distributed among procressors in a round robin fashion
!
!***********************************************************************


  SUBROUTINE REAL_STDLAY(GRID, LPARALLEL)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID
    LOGICAL LPARALLEL

    IF (LPARALLEL) THEN
#ifdef MPI
!-----------------------------------------------------------------------
! version for parallel computers
! (currently zy planes are on one proc, z is fast index)
!-----------------------------------------------------------------------
       NCOL_TOT= GRID%NGY*GRID%NGX
       NCOL_MAX=(NCOL_TOT+GRID%COMM%NCPU-1)/GRID%COMM%NCPU
       IF (LPLANE_WISE) THEN
          NCOL_MAX=GRID%NGY*((GRID%NGX+GRID%COMM%NCPU-1)/GRID%COMM%NCPU)
       ENDIF

       ALLOCATE(GRID%RL%I2( NCOL_MAX ))
       ALLOCATE(GRID%RL%I3( NCOL_MAX ))
       ALLOCATE(GRID%RL%INDEX(0:GRID%NGX-1,0:GRID%NGY-1))

       GRID%RL%NFAST= 3
       GRID%RL%NCOL = 0
       GRID%RL%NROW = GRID%NGZ
       GRID%RL%INDEX= 0
       ! distribute planes among proc
       IND=0
       DO N3=1,GRID%NGY
          DO N2=1,GRID%NGX
             IF( LPLANE_WISE ) THEN
                NODE_TARGET=MOD(N2-1,GRID%COMM%NCPU)+1
             ELSE
                NODE_TARGET=MOD(IND,GRID%COMM%NCPU)+1
             ENDIF

             IND=IND+1
             IF (NODE_TARGET == GRID%COMM%NODE_ME) THEN
                N=GRID%RL%NCOL
                N=N+1
                GRID%RL%INDEX(N2-1,N3-1)=N
                GRID%RL%NCOL=N
                GRID%RL%I2(N)=N2 ! I2 contains x index
                GRID%RL%I3(N)=N3 ! I3      the y index
             ENDIF
          ENDDO
       ENDDO
       IF (GRID%RL%NCOL >NCOL_MAX) THEN
          WRITE(*,*) 'REAL_STDLAY: internal error',NCOL_MAX,GRID%RL%NCOL
          STOP
       ENDIF
       GRID%RL%NP    = GRID%RL%NCOL*GRID%RL%NROW
       IF (GRID%LREAL) THEN
          GRID%RL%NALLOC=NCOL_MAX*GRID%NGZ_rd
       ELSE
          GRID%RL%NALLOC=NCOL_MAX*GRID%NGZ
       ENDIF
#else
       WRITE(*,*) 'internal error: REAL_STDLAY was called with LPARALLEL=.TRUE.'
       WRITE(*,*) '  in serial version'
       STOP
#endif
!-----------------------------------------------------------------------
! conventional version (really simple)
! include all points x fast
!-----------------------------------------------------------------------
    ELSE
       GRID%RL%NFAST= 1
       GRID%RL%NCOL = GRID%NGZ*GRID%NGY
       GRID%RL%NROW = GRID%NGX

       ALLOCATE(GRID%RL%INDEX(0:GRID%NGY-1,0:GRID%NGZ-1))
       ALLOCATE(GRID%RL%I2( GRID%RL%NCOL ))
       ALLOCATE(GRID%RL%I3( GRID%RL%NCOL ))

       GRID%RL%INDEX=0

       IND=0
       DO N3=1,GRID%NGZ
          DO N2=1,GRID%NGY
             IND=IND+1
             GRID%RL%INDEX(N2-1,N3-1)=IND
             GRID%RL%I2(IND)=N2
             GRID%RL%I3(IND)=N3
          ENDDO
       ENDDO

       GRID%RL%NP    = GRID%RL%NCOL*GRID%RL%NROW
       IF (GRID%LREAL) THEN
          GRID%RL%NALLOC= GRID%RL%NCOL*GRID%RL%NROW/2
       ELSE
          GRID%RL%NALLOC= GRID%RL%NCOL*GRID%RL%NROW
       ENDIF
    ENDIF
  END SUBROUTINE REAL_STDLAY


!************************************************************************
!
! copy real space layout from one GRID_SRC to another GRID_DEST
! it might be that the GRID_DEST    contains more nodes in this
! case remaining nodes in GRID_DEST will have no columns in real space
! for instance consider the following topology (see M_divide in mpi.F)
! wave1           0 (0,0)        1 (0,1)
! wave2           2 (1,0)        3 (1,1)
! wave3           4 (2,0)        5 (2,1)
! wave4           6 (3,0)        7 (3,1)
! GRID_DEST%COMM contains all nodes 0-7, whereas GRID_SRC%COMM is an
! cartesian sub-communicator (in-band-group)
! (which communicate along first row, i.e proc 0 and 1).
! in real space node 0 and 1 will store same columns as GRID_SRC
! all other nodes will have no data in real space
!
!***********************************************************************

  SUBROUTINE  SET_RL_GRID(GRID_DEST,GRID_SRC)
    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (grid_3d)     GRID_DEST,GRID_SRC
#ifdef MPI
    ! deallocate old RL space data prior to copy
    DEALLOCATE(GRID_DEST%RL%I2, GRID_DEST%RL%I3, GRID_DEST%RL%INDEX)
    NULLIFY(   GRID_DEST%RL%I2, GRID_DEST%RL%I3, GRID_DEST%RL%INDEX)
    !
    IF (GRID_DEST%NGX /= GRID_SRC%NGX .OR. GRID_DEST%NGY /= GRID_SRC%NGY .OR. GRID_DEST%NGZ /= GRID_SRC%NGZ) THEN
       WRITE(0,*) 'internal ERROR in SET_RL_GRID: GRID_DEST is not conform to GRID_SRC'
       STOP
    ENDIF
    !
    ! complicated case serial FFT is used for GRID_SRC, but GRID_DST
    ! must apply parallel FFT
    ! in this case the RL_FFT structure is setup to the standard parallel data layout 
    ! on the root node (and no data on other nodes)
    ! the FFT routine must change from the serial data layout to the 
    ! parallel data layout before calling the real-> complex FFT
    ! and after calling the complex -> real FFT the data data layout must
    ! be swapped again
    !
    IF (GRID_SRC%RL%NFAST ==1) THEN
       ! now check that only one node contains all data
       IF (GRID_SRC%COMM%NCPU/=1) THEN
          WRITE(0,*) 'internal ERROR in SET_RL_GRID: GRID_SRC contains more than one CPU', GRID_SRC%COMM%NCPU
          STOP
       ENDIF
       ! now check that the data layout is strictly sequential in real space
       IND=0
       DO N3=1,GRID_SRC%NGZ
          DO N2=1,GRID_SRC%NGY
             IND=IND+1
             IF (GRID_SRC%RL%I2(IND)/=N2 .OR. GRID_SRC%RL%I3(IND)/=N3) THEN
                WRITE(0,*) 'internal ERROR in SET_RL_GRID: GRID_SRC not sequential', IND,& 
                     GRID_SRC%RL%I2(IND),N2,GRID_SRC%RL%I3(IND),N3
                STOP
             ENDIF
          ENDDO
       ENDDO

       ! allocate RL_FFT structure
       NULLIFY(GRID_DEST%RL_FFT)
       ALLOCATE(GRID_DEST%RL_FFT)
       CALL REAL_STDLAY_ONE_NODE(GRID_DEST)
    ELSE
       IF (GRID_SRC%RL%NFAST/=3) THEN
          WRITE(0,*) 'internal ERROR in SET_RL_GRID: GRID_SRC has a strange NFAST index', GRID_SRC%RL%NFAST
          STOP
       ENDIF
    ENDIF
    !
    ! now copy the SRC data layout to the GRID_DEST%RL
    !
    GRID_DEST%RL%NFAST= GRID_SRC%RL%NFAST

    IF (GRID_DEST%COMM%NODE_ME <= GRID_SRC%COMM%NCPU) THEN
       GRID_DEST%RL%NFAST= GRID_SRC%RL%NFAST
       GRID_DEST%RL%NCOL = GRID_SRC%RL%NCOL

       ! required memory in FFT is subtle
       IF (GRID_DEST%RL%NFAST==1) THEN
          ! first case maximum of FFT requirements (RL_FFT)
          !   and NGX*NGY*NGZ
          IF (GRID_DEST%LREAL) THEN
             GRID_DEST%RL%NALLOC=GRID_DEST%RL%NCOL*GRID_DEST%NGX/2
          ELSE
             GRID_DEST%RL%NALLOC=GRID_DEST%RL%NCOL*GRID_DEST%NGX
          ENDIF
          GRID_DEST%RL%NALLOC=MAX(GRID_DEST%RL%NALLOC,GRID_DEST%RL_FFT%NALLOC)
       ELSE
          ! second case: standard
          GRID_DEST%RL%NALLOC=GRID_DEST%RL%NCOL*GRID_DEST%NGZ_rd
       ENDIF

       GRID_DEST%RL%NROW = GRID_SRC%RL%NROW
       GRID_DEST%RL%NP    =GRID_SRC%RL%NP
       GRID_DEST%RL%INDEX=>GRID_SRC%RL%INDEX
       GRID_DEST%RL%I2   =>GRID_SRC%RL%I2
       GRID_DEST%RL%I3   =>GRID_SRC%RL%I3
    ELSE
       GRID_DEST%RL%NCOL  =0
       GRID_DEST%RL%NROW  =0
       GRID_DEST%RL%NALLOC=0
       GRID_DEST%RL%NP    =0
       NULLIFY(GRID_DEST%RL%I2, GRID_DEST%RL%I3, GRID_DEST%RL%INDEX)
    ENDIF
    GRID_DEST%MPLWV=__align_DIM(MAX(GRID_DEST%RC%NALLOC ,GRID_DEST%IN%NALLOC , GRID_DEST%RL%NALLOC))

#else
    ! nothing
#endif
  END SUBROUTINE SET_RL_GRID


  SUBROUTINE  SET_RL_GRID_OLD(GRID_DEST,GRID_SRC)
    USE prec

    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (grid_3d)     GRID_DEST,GRID_SRC
#ifdef MPI
    DEALLOCATE(GRID_DEST%RL%I2,GRID_DEST%RL%I3,GRID_DEST%RL%INDEX)
    IF (GRID_DEST%NGX /= GRID_SRC%NGX .OR. GRID_DEST%NGY /= GRID_SRC%NGY .OR. GRID_DEST%NGZ /= GRID_SRC%NGZ) THEN
       WRITE(0,*) 'ERROR in SET_RL_GRID: GRID_DEST is not conform to GRID_SRC'
       STOP
    ENDIF
    IF (GRID_DEST%COMM%NODE_ME <= GRID_SRC%COMM%NCPU) THEN
       GRID_DEST%RL%NFAST= GRID_SRC%RL%NFAST
       GRID_DEST%RL%NCOL = GRID_SRC%RL%NCOL
       GRID_DEST%RL%NALLOC=GRID_DEST%RL%NCOL*GRID_DEST%NGZ_rd
       GRID_DEST%RL%NROW = GRID_SRC%RL%NROW
       GRID_DEST%RL%NP    =GRID_SRC%RL%NP
       GRID_DEST%RL%INDEX=>GRID_SRC%RL%INDEX
       GRID_DEST%RL%I2   =>GRID_SRC%RL%I2
       GRID_DEST%RL%I3   =>GRID_SRC%RL%I3
    ELSE
       GRID_DEST%RL%NFAST =3
       GRID_DEST%RL%NCOL  =0
       GRID_DEST%RL%NROW  =0
       GRID_DEST%RL%NALLOC=0
       GRID_DEST%RL%NP    =0
       NULLIFY(GRID_DEST%RL%I2); NULLIFY(GRID_DEST%RL%I3)
       NULLIFY(GRID_DEST%RL%INDEX)
    ENDIF
    GRID_DEST%MPLWV=__align_DIM(MAX(GRID_DEST%RC%NALLOC ,GRID_DEST%IN%NALLOC , GRID_DEST%RL%NALLOC))
#else
    ! nothing
#endif
  END SUBROUTINE SET_RL_GRID_OLD


!-----------------------------------------------------------------------
! helper routine to generate a standard parallel data layout
! on the root node
!-----------------------------------------------------------------------

  SUBROUTINE REAL_STDLAY_ONE_NODE(GRID)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

#ifdef MPI
    IF (GRID%COMM%NODE_ME==1) THEN
       NCOL_TOT= GRID%NGY*GRID%NGX
       NCOL_MAX= NCOL_TOT

       ALLOCATE(GRID%RL_FFT%I2( NCOL_MAX ))
       ALLOCATE(GRID%RL_FFT%I3( NCOL_MAX ))
       ALLOCATE(GRID%RL_FFT%INDEX(0:GRID%NGX-1,0:GRID%NGY-1))

       GRID%RL_FFT%NFAST= 3
       GRID%RL_FFT%NROW = GRID%NGZ
       GRID%RL_FFT%INDEX= 0
       ! distribute planes among proc
       IND=0
       DO N3=1,GRID%NGY
          DO N2=1,GRID%NGX
             NODE_TARGET=1
             IND=IND+1
             GRID%RL_FFT%INDEX(N2-1,N3-1)=IND
             GRID%RL_FFT%NCOL=IND
             GRID%RL_FFT%I2(IND)=N2 ! I2 contains x index
             GRID%RL_FFT%I3(IND)=N3 ! I3      the y index
          ENDDO
       ENDDO
       GRID%RL_FFT%NP    = GRID%RL_FFT%NCOL*GRID%RL_FFT%NROW
       IF (GRID%LREAL) THEN
          GRID%RL_FFT%NALLOC=NCOL_MAX*(GRID%NGZ/2+1)
       ELSE
          GRID%RL_FFT%NALLOC=NCOL_MAX*GRID%NGZ
       ENDIF
    ELSE
       ! other nodes contain no data
       GRID%RL_FFT%NFAST =3
       GRID%RL_FFT%NCOL  =0
       GRID%RL_FFT%NROW  =0
       GRID%RL_FFT%NALLOC=0
       GRID%RL_FFT%NP    =0
       NULLIFY(GRID%RL_FFT%I2, GRID%RL_FFT%I3, GRID%RL_FFT%INDEX)
    ENDIF
#else
    WRITE(*,*) 'internal error: REAL_STDLAY_ONE_NODE was called in serial version'
    STOP
#endif
  END SUBROUTINE REAL_STDLAY_ONE_NODE

END MODULE
!************************* SUBROUTINE FFTWEIGHT_INI ********************
!
! FFTWEIGHT_INI set's up a "metric" for evaluating scalar products
! on a grid such as
!   sum_G rho_1(G) rho_2(G)
! this is required for the gamma only version since only
! coefficients g_x (serial) or g_z (parallel) >=0 are stored
! Hence planes with g_x>0 (or g_z>0) need to be weighted twice
! (presently only used for GRIDHF in fock.F)
!
!***********************************************************************

    SUBROUTINE  FFTWEIGHT_INI(GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT NONE
      TYPE (grid_3d)  GRID
    ! local
      INTEGER NX, NY, NZ, IND, N1, NC, N2, N3

      IF (GRID%REAL2CPLX) THEN
         NULLIFY(GRID%FFTWEIGHT)
         ALLOCATE(GRID%FFTWEIGHT(GRID%RC%NP))

         IF (GRID%RL%NFAST==1) THEN
            ! serial version
            DO IND=1,GRID%RC%NP
               N1= MOD((IND-1),GRID%RC%NROW)+1
               NC= (IND-1)/GRID%RC%NROW+1
               N2= GRID%RC%I2(NC)
               N3= GRID%RC%I3(NC)
               
               ! g_x=0 plane is stored fully
               IF (N1==1) THEN
                  GRID%FFTWEIGHT(IND)=1._q
               ! only g_x>0 is stored; g_x < 0 is not stored
               ELSE
                  GRID%FFTWEIGHT(IND)=2._q
               ENDIF
            ENDDO
         ELSE
            ! parallel version
            DO IND=1,GRID%RC%NP
               N1= MOD((IND-1),GRID%RC%NROW)+1
               NC= (IND-1)/GRID%RC%NROW+1
               N2= GRID%RC%I2(NC)
               N3= GRID%RC%I3(NC)
               ! g_z=0 plane is stored fully
               IF (N3==1) THEN
                  GRID%FFTWEIGHT(IND)=1._q
               ! only g_z>0 is stored; g_x < 0 is not stored
               ELSE
                  GRID%FFTWEIGHT(IND)=2._q
               ENDIF
            ENDDO
         ENDIF
      ENDIF
      RETURN
    END SUBROUTINE FFTWEIGHT_INI


!*************************SUBROUTINE RC_FLIP ***************************
!
! rearranges the storage mode for spin components of charge density:
! 
! given rho_up and rho_down on input the quantities (rho_up+rho_down)
! and (rho_up-rho_down) = total charge and magnetization are returned
! and also the reverse operation is possible if setting LBACK=.TRUE.
! RC_FLIP  operates in reciprocal space (half gird)
! RL_FLIP  operates in real space       (possibly real arrays)
!
! for the non collinear version ISPIN must be set to 4 by the caller
!
! given the total 2x2 density matrix (CHTOT = p00, p11, p01, p10 ) 
! the  charge density (C00) and the magnetisation density 
! (CX, CY, CY) are calculated 
! If LBACK is defined, and given the charge density (C00) and the 
! magnetisation density (CX, CY, CY) the total 2x2 density matrix 
! (CHTOT = p00, p11, p01, p10 ) is calculated
!
!***********************************************************************

!
!  version for operating in reciprocal space (half grid version)
!
  SUBROUTINE RC_FLIP(CHTOT, GRID,ISPIN,LBACK)
    USE mgrid

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    LOGICAL LBACK
    COMPLEX(q) CHTOT(GRID%MPLWV,ISPIN)

    IF (ISPIN==2) THEN

       FAC=1._q
       IF (LBACK) FAC=0.5_q
       DO K=1,GRID%RC%NP
          CQU=CHTOT(K,1)
          CQD=CHTOT(K,2)
          CHTOT(K,1)=FAC*(CQU+CQD)
          CHTOT(K,2)=FAC*(CQU-CQD)
       ENDDO
    ELSE IF ( ISPIN==4 .AND. .NOT. LBACK) THEN
       DO K=1,GRID%RC%NP
          C00=CHTOT(K,1)
          C01=CHTOT(K,2)
          C10=CHTOT(K,3)
          C11=CHTOT(K,4)

          CHTOT(K,1)= C00+C11             
          CHTOT(K,2)= C01+C10             
          CHTOT(K,3)=(C01-C10)*(0._q,1._q)
          CHTOT(K,4)= C00-C11             
       ENDDO
    ELSE IF ( ISPIN==4 .AND. LBACK) THEN
       FAC=0.5_q
       DO K=1,GRID%RC%NP
          C00=CHTOT(K,1)
          CX =CHTOT(K,2)
          CY =CHTOT(K,3)
          CZ =CHTOT(K,4)

          CHTOT(K,1)= (C00+CZ)*FAC           
          CHTOT(K,2)= (CX-CY*(0._q,1._q))*FAC
          CHTOT(K,3)= (CX+CY*(0._q,1._q))*FAC
          CHTOT(K,4)= (C00-CZ)*FAC           
       ENDDO
    ENDIF

  END SUBROUTINE

!
! version for potentials
! differs by a factor two
!
  SUBROUTINE RC_FLIP_POTENTIAL(CHTOT, GRID,ISPIN,LBACK)
    USE mgrid

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    LOGICAL LBACK
    COMPLEX(q) CHTOT(GRID%MPLWV,ISPIN)

    IF (ISPIN==2) THEN

       FAC=0.5_q
       IF (LBACK) FAC=1.0_q
       DO K=1,GRID%RC%NP
          CQU=CHTOT(K,1)
          CQD=CHTOT(K,2)
          CHTOT(K,1)=FAC*(CQU+CQD)
          CHTOT(K,2)=FAC*(CQU-CQD)
       ENDDO
    ELSE IF ( ISPIN==4 .AND. .NOT. LBACK) THEN
       FAC=0.5_q
       DO K=1,GRID%RC%NP
          C00=CHTOT(K,1)
          C01=CHTOT(K,2)
          C10=CHTOT(K,3)
          C11=CHTOT(K,4)

          CHTOT(K,1)=(C00+C11)*FAC
          CHTOT(K,2)=(C01+C10)*FAC
          CHTOT(K,3)=((C01-C10)*(0._q,1._q))*FAC
          CHTOT(K,4)=(C00-C11)*FAC
       ENDDO
    ELSE IF ( ISPIN==4 .AND. LBACK) THEN
       FAC=1.0_q
       DO K=1,GRID%RC%NP
          C00=CHTOT(K,1)
          CX =CHTOT(K,2)
          CY =CHTOT(K,3)
          CZ =CHTOT(K,4)

          CHTOT(K,1)= (C00+CZ)*FAC           
          CHTOT(K,2)= (CX-CY*(0._q,1._q))*FAC
          CHTOT(K,3)= (CX+CY*(0._q,1._q))*FAC
          CHTOT(K,4)= (C00-CZ)*FAC           
       ENDDO
    ENDIF

  END SUBROUTINE

!
!  version for operating in real space (possibly real arrays)
!
  SUBROUTINE RL_FLIP(CHTOTR, GRID,ISPIN,LBACK)
    USE mgrid

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    LOGICAL LBACK

    RGRID CHTOTR(DIMREAL(GRID%MPLWV),ISPIN)

    IF (ISPIN==2) THEN
       FAC=1._q
       IF (LBACK) FAC=0.5_q
       DO K=1,GRID%RL%NP
          CQU=CHTOTR(K,1)
          CQD=CHTOTR(K,2)
          CHTOTR(K,1)=FAC*(CQU+CQD)
          CHTOTR(K,2)=FAC*(CQU-CQD)
       ENDDO
    ELSE IF ( ISPIN==4 .AND. .NOT. LBACK) THEN
       DO K=1,GRID%RL%NP
          C00=CHTOTR(K,1)
          C01=CHTOTR(K,2)
          C10=CHTOTR(K,3)
          C11=CHTOTR(K,4)

          CHTOTR(K,1)= C00+C11             
          CHTOTR(K,2)= C01+C10             
          CHTOTR(K,3)=(C01-C10)*(0._q,1._q)
          CHTOTR(K,4)= C00-C11             
       ENDDO
    ELSE IF ( ISPIN==4 .AND. LBACK) THEN
       FAC=0.5_q
       DO K=1,GRID%RL%NP
          C00=CHTOTR(K,1)
          CX =CHTOTR(K,2)
          CY =CHTOTR(K,3)
          CZ =CHTOTR(K,4)

          CHTOTR(K,1)= (C00+CZ)*FAC           
          CHTOTR(K,2)= (CX-CY*(0._q,1._q))*FAC
          CHTOTR(K,3)= (CX+CY*(0._q,1._q))*FAC
          CHTOTR(K,4)= (C00-CZ)*FAC           
       ENDDO
    ENDIF
  END SUBROUTINE RL_FLIP


!************************ SUBROUTINE MRG_GRID_RL ***********************
!
! helper routine to merge a grid-array from all nodes to the
! local node (real space version)
! result is always real and has standard serial layout
!
!***********************************************************************

  SUBROUTINE MRG_GRID_RL(GRID, C, C_LOCAL)
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
#ifdef MPI
    RGRID   C_LOCAL(GRID%RL%NROW,GRID%RL%NCOL)
    REAL(q) C(GRID%NGX,GRID%NGY,GRID%NGZ)
    INTEGER NC,NX,NY,NZ

    C=0
    DO NC=1,GRID%RL%NCOL
       NX= GRID%RL%I2(NC)
       NY= GRID%RL%I3(NC)
       DO NZ=1,GRID%NGZ
          C(NX,NY,NZ)=C_LOCAL(NZ,NC)
       ENDDO
    ENDDO

    CALLMPI( M_sum_d(GRID%COMM, C(1,1,1), GRID%NPLWV))
#else
    RGRID    C_LOCAL(GRID%RL%NP)
    REAL(q)  C(GRID%RL%NP)

    C(1:GRID%RL%NP)=C_LOCAL(1:GRID%RL%NP)
#endif

  END SUBROUTINE MRG_GRID_RL

!************************ SUBROUTINE MRG_GRID_RL_PLANE *****************
!
! helper routine to merge a plane of a grid-array from all nodes to the
! local node (real space version)
! result is always real and has standard serial layout
!
!***********************************************************************

  SUBROUTINE MRG_GRID_RL_PLANE(GRID, C, C_LOCAL, NZ)
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
#ifdef MPI
    RGRID   C_LOCAL(GRID%RL%NROW,GRID%RL%NCOL)
    REAL(q) C(GRID%NGX,GRID%NGY)
    INTEGER NC,NX,NY,NZ

    C=0
    IF (GRID%RL%NFAST==1) THEN
       DO NC=1,GRID%RL%NCOL
          IF (GRID%RL%I3(NC)/=NZ) CYCLE
          NY=GRID%RL%I2(NC)
          C(:,NY)=C_LOCAL(:,NC)
       ENDDO
    ELSEIF (GRID%RL%NFAST==3) THEN
       DO NC=1,GRID%RL%NCOL
          NX= GRID%RL%I2(NC)
          NY= GRID%RL%I3(NC)
          C(NX,NY)=C_LOCAL(NZ,NC)
       ENDDO
    ENDIF

    CALLMPI( M_sum_d(GRID%COMM, C(1,1), GRID%NGX*GRID%NGY))
#else
    INTEGER NZ
    RGRID    C_LOCAL(GRID%NGX*GRID%NGY,NZ)
    REAL(q)  C(GRID%NGX*GRID%NGY)

    C=C_LOCAL(:,NZ)
#endif

  END SUBROUTINE MRG_GRID_RL_PLANE

!************************ SUBROUTINE MRG_GRID_RC_PLANE *****************
!
! helper routine to merge a plane of a grid-array from all nodes to the
! local node (reciprocal space version)
! result is always complex and has standard serial layout
!
!***********************************************************************

  SUBROUTINE MRG_GRID_RC_PLANE(GRID, C, C_LOCAL, NX)
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID

    COMPLEX(q) C_LOCAL(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) C(GRID%NGZ_rd,GRID%NGY_rd)
    INTEGER NC,NX,NY,NZ

    C=0
    DO NC=1,GRID%RC%NCOL
       NY= GRID%RC%I2(NC)
       NZ= GRID%RC%I3(NC)
       C(NZ,NY)=C_LOCAL(NX,NC)
    ENDDO

    CALLMPI( M_sum_d(GRID%COMM, C(1,1), 2*GRID%NGY_rd*GRID%NGZ_rd))

  END SUBROUTINE MRG_GRID_RC_PLANE

!************************ SUBROUTINE DIS_GRID_RL ***********************
!
! helper routine to distribute an array from io-node to the
! local nodes (real space version)
! input must be always real and must have  standard serial layout
!
!***********************************************************************

  SUBROUTINE DIS_GRID_RL(GRID, C, C_LOCAL, LBROADCAST )
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
    LOGICAL LBROADCAST
#ifdef MPI
    RGRID  C_LOCAL(GRID%RL%NROW,GRID%RL%NCOL)
    REAL(q)   C(GRID%NGX,GRID%NGY,GRID%NGZ)
    INTEGER NC,NX,NY,NZ

    ! broadcast C to all nodes from IONODE
    IF (LBROADCAST) &
         CALLMPI( M_bcast_d(GRID%COMM, C(1,1,1), GRID%NPLWV))
    ! and pick up data
    DO NC=1,GRID%RL%NCOL
       NX= GRID%RL%I2(NC)
       NY= GRID%RL%I3(NC)
       DO NZ=1,GRID%NGZ
          C_LOCAL(NZ,NC)=C(NX,NY,NZ)
       ENDDO
    ENDDO
#else
    RGRID C_LOCAL(GRID%RL%NP)
    REAL(q)  C(GRID%RL%NP)

    C_LOCAL(1:GRID%RL%NP)=C(1:GRID%RL%NP)
#endif

  END SUBROUTINE DIS_GRID_RL

!************************ SUBROUTINE DIS_GRID_RL ***********************
!
! helper routine to distribute a plane of a grid based array from io-node 
! to the local nodes (real space version)
! input must be always real and must have  standard serial layout
!
!***********************************************************************

  SUBROUTINE DIS_GRID_RL_PLANE(GRID, C, C_LOCAL, LBROADCAST , NZ  )
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
    LOGICAL LBROADCAST
#ifdef MPI
    RGRID  C_LOCAL(GRID%RL%NROW,GRID%RL%NCOL)
    REAL(q)   C(GRID%NGX,GRID%NGY)
    INTEGER NC,NX,NY,NZ

    ! broadcast C to all nodes from IONODE
    IF (LBROADCAST) &
         CALLMPI( M_bcast_d(GRID%COMM, C(1,1), GRID%NGX*GRID%NGY))
    ! and pick up data
    DO NC=1,GRID%RL%NCOL
       NX= GRID%RL%I2(NC)
       NY= GRID%RL%I3(NC)
       C_LOCAL(NZ,NC)=C(NX,NY)
    ENDDO
#else
    INTEGER NZ
    RGRID    C_LOCAL(GRID%NGX*GRID%NGY,NZ)
    REAL(q)  C(GRID%NGX*GRID%NGY)

    C_LOCAL(:,NZ)=C
#endif

  END SUBROUTINE DIS_GRID_RL_PLANE

!************************ SUBROUTINE MRG_GRID_RC ***********************
!
! helper routine to merge a grid-array from all nodes to the
! local node (reciprocal space version)
! result is always complex and has standard serial layout
!
!***********************************************************************

  SUBROUTINE MRG_GRID_RC(GRID, C, C_LOCAL)
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
#ifdef MPI
    COMPLEX(q) C_LOCAL(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) C(GRID%NGX_rd,GRID%NGY,GRID%NGZ_rd)
    INTEGER NC,NX,NY,NZ,NPLWV

    C=0
    DO NC=1,GRID%RC%NCOL
       NY= GRID%RC%I2(NC)
       NZ= GRID%RC%I3(NC)
       DO NX=1,GRID%NGX_rd
          C(NX,NY,NZ)=C_LOCAL(NX,NC)
       ENDDO
    ENDDO

    NPLWV=GRID%NGX_rd*GRID%NGY*GRID%NGZ_rd
    CALLMPI( M_sum_z(GRID%COMM, C(1,1,1), NPLWV))
#else
    COMPLEX(q) C_LOCAL(GRID%RC%NP)
    COMPLEX(q) C(GRID%RC%NP)

    C(1:GRID%RC%NP)=C_LOCAL(1:GRID%RC%NP)
#endif

  END SUBROUTINE MRG_GRID_RC

!************************ SUBROUTINE DIS_GRID_RC ***********************
!
! helper routine to distribute an array from io-node to the
! local nodes (reciprocal space version)
! input must be always real and must have  standard serial layout
!
!***********************************************************************

  SUBROUTINE DIS_GRID_RC(GRID, C, C_LOCAL, LBROADCAST )
    USE mgrid
    IMPLICIT NONE

    TYPE (grid_3d) GRID
    LOGICAL LBROADCAST
#ifdef MPI
    COMPLEX(q) C_LOCAL(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) C(GRID%NGX_rd,GRID%NGY,GRID%NGZ_rd)
    INTEGER NC,NX,NY,NZ,NPLWV
    NPLWV=GRID%NGX_rd*GRID%NGY*GRID%NGZ_rd

    ! broadcast C to all nodes from IONODE
    IF (LBROADCAST) &
         CALLMPI( M_bcast_z(GRID%COMM, C(1,1,1), NPLWV))
    ! and pick up data
    DO NC=1,GRID%RC%NCOL
       NY= GRID%RC%I2(NC)
       NZ= GRID%RC%I3(NC)
       DO NX=1,GRID%NGX_rd
          C_LOCAL(NX,NC)=C(NX,NY,NZ)
       ENDDO
    ENDDO
#else
    COMPLEX(q) C_LOCAL(GRID%RC%NP)
    COMPLEX(q) C(GRID%RC%NP)

    C_LOCAL(1:GRID%RC%NP)=C(1:GRID%RC%NP)
#endif

  END SUBROUTINE DIS_GRID_RC



!************************ SUBROUTINE RL_ADD  ***************************
!
!  This modul contains a helper routine to perform the operation
!     C = A * SCALE1 + B * SCALE2
!  in REAL(q) SPACE
!  depending on the mode this is done using real or complex
!  arrays
!  C might be equal to A
!  B might be equal to A or C or if SCALE2 is 0
! !! there is no need to optimized these routines                    !!
!
!***********************************************************************

  SUBROUTINE RL_ADD(A,SCALE1,B,SCALE2,C,GRID)
    USE mgrid

    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d) GRID
    RGRID  C(GRID%RL%NP),A(GRID%RL%NP),B(GRID%RL%NP)

    NP=GRID%RL%NP
    !
    !   C=A*SCALE1
    !
    !  )  case SCALE1=0
    IF (SCALE1==0)  THEN
       DO 100 N=1,NP
          C(N)=0
100    ENDDO
       !  )  case SCALE1=1
    ELSE IF (SCALE1==1)  THEN
       DO 200 N=1,NP
          C(N)=A(N)
200    ENDDO
       !  )  else
    ELSE
       DO 300 N=1,NP
          C(N)=A(N)*SCALE1
300    ENDDO
    ENDIF

    !
    !   C=C+ B*SCALE2
    !
    IF (SCALE2==0) THEN
    ELSE IF (SCALE2==1) THEN
       DO 400 N=1,NP
          C(N)=C(N)+B(N)
400    ENDDO
    ELSE
       DO 500 N=1,NP
          C(N)=C(N)+B(N)*SCALE2
500    ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RL_ADD



!************************ SUBROUTINE RC_ADD  ***************************
!
!  This modul contains a helper routine to perform the operation
!     C = A * SCALE1 + B * SCALE2
!  in RECIPROCAL SPACE
!  C might be equal to A
!
!***********************************************************************

  SUBROUTINE RC_ADD(A,SCALE1,B,SCALE2,C,GRID)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d) GRID
    COMPLEX(q) C(GRID%RC%NP),A(GRID%RC%NP),B(GRID%RC%NP)
    NP=GRID%RC%NP
    !
    !   C=A*SCALE1
    !
    !  )  case SCALE1=0
    IF (SCALE1==0)  THEN
       DO 100 N=1,NP
          C(N)=0
100    ENDDO
       !  )  case SCALE1=1
    ELSE IF (SCALE1==1)  THEN
       DO 200 N=1,NP
          C(N)=A(N)
200    ENDDO
       !  )  else
    ELSE
       DO 300 N=1,NP
          C(N)=A(N)*SCALE1
300    ENDDO
    ENDIF

    !
    !   C=C+ B*SCALE2
    !
    IF (SCALE2==0) THEN
    ELSE IF (SCALE2==1) THEN
       DO 400 N=1,NP
          C(N)=C(N)+B(N)
400    ENDDO
    ELSE
       DO 500 N=1,NP
          C(N)=C(N)+B(N)*SCALE2
500    ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RC_ADD

!************************ SUBROUTINE RLR_MUL  **************************
!  This modul contains a helper routine to perform the operation
!     C = A * B * SCALE
!  in REAL SPACE, where only the real part of B is used
!                              
!  depending on the mode this is done using real or complex
!  arrays C and A
!   C might be equal to A
!
! !! there is no need to optimized these routines                    !!
!***********************************************************************

  SUBROUTINE RLR_MUL(A,B,SCALE,C,GRID)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d) GRID
    RGRID B(GRID%RL%NP)
    RGRID C(GRID%RL%NP),A(GRID%RL%NP)

    NPLWV=GRID%RL%NP
    !
    !   C=0
    !
    !  )  case SCALE=0
    IF (SCALE==0)  THEN
       DO N=1,NPLWV
          C(N)=0
       ENDDO
       !  )  case SCALE=1
    ELSE IF (SCALE==1)  THEN
       DO N=1,NPLWV
          C(N)=A(N)*B(N)
       ENDDO
       !  )  else
    ELSE
       DO N=1,NPLWV
          C(N)=A(N)*B(N)*SCALE
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RLR_MUL

!***********************************************************************
!
!  This subroutine adds an array defined on the small grid
!  GRID_SOFT to  an array defined on the fine grid GRID
!  both arrays are in the half grid mode
!
!***********************************************************************

  SUBROUTINE ADD_GRID(GRID,GRID_SOFT,SOFT_TO_C,CADD,CSUM)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID,GRID_SOFT
    TYPE (transit)     SOFT_TO_C

    COMPLEX(q) CSUM(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) CADD(GRID_SOFT%RC%NROW,GRID_SOFT%RC%NCOL)

    col: DO NC=1,GRID_SOFT%RC%NCOL
       row: DO N1=1,GRID_SOFT%RC%NROW
          CSUM(SOFT_TO_C%IND1(N1),SOFT_TO_C%INDCOL(NC))= &
               CSUM(SOFT_TO_C%IND1(N1),SOFT_TO_C%INDCOL(NC))+ CADD(N1,NC)
       ENDDO row
    ENDDO col
    RETURN
  END SUBROUTINE ADD_GRID

!***********************************************************************
!
!  This subroutine copies an array defined on the small grid
!  GRID_SOFT to  an array defined on the fine grid GRID
!  both arrays are in the half grid mode
!
!***********************************************************************

  SUBROUTINE CPB_GRID(GRID,GRID_SOFT,SOFT_TO_C,CSRC,CDEST)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID,GRID_SOFT
    TYPE (transit)     SOFT_TO_C

    COMPLEX(q) CDEST(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) CSRC(GRID_SOFT%RC%NROW,GRID_SOFT%RC%NCOL)

    col: DO NC=1,GRID_SOFT%RC%NCOL
       row: DO N1=1,GRID_SOFT%RC%NROW
          CDEST(SOFT_TO_C%IND1(N1),SOFT_TO_C%INDCOL(NC))=CSRC(N1,NC)
       ENDDO row
    ENDDO col
    RETURN
  END SUBROUTINE CPB_GRID

!***********************************************************************
!
!  This subroutine brings an array defined on the fine grid
!  to an array defined on the wavefunction grid
!  both arrays are stored in the half grid mode
!
!***********************************************************************

  SUBROUTINE CP_GRID(GRID,GRID_SOFT,SOFT_TO_C,CSRC,CDEST)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID,GRID_SOFT
    TYPE (transit)     SOFT_TO_C

    COMPLEX(q) CSRC(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) CDEST(GRID_SOFT%RC%NROW,GRID_SOFT%RC%NCOL)

    col: DO NC=1,GRID_SOFT%RC%NCOL
       row: DO N1=1,GRID_SOFT%RC%NROW
          CDEST(N1,NC)=CSRC(SOFT_TO_C%IND1(N1),SOFT_TO_C%INDCOL(NC))
       ENDDO row
    ENDDO col
    RETURN
  END SUBROUTINE CP_GRID

!***********************************************************************
!
!  This subroutine adds an array defined on the fine grid
!  to an array defined on the wavefunctions grid
!  both arays are stored in the half grid mode
!
!***********************************************************************

  SUBROUTINE CP_ADD_GRID(GRID,GRID_SOFT,SOFT_TO_C,CSRC,CDEST)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID,GRID_SOFT
    TYPE (transit)     SOFT_TO_C

    COMPLEX(q) CSRC(GRID%RC%NROW,GRID%RC%NCOL)
    COMPLEX(q) CDEST(GRID_SOFT%RC%NROW,GRID_SOFT%RC%NCOL)

    col: DO NC=1,GRID_SOFT%RC%NCOL
       row: DO N1=1,GRID_SOFT%RC%NROW
          CDEST(N1,NC)=CDEST(N1,NC)+ CSRC(SOFT_TO_C%IND1(N1),SOFT_TO_C%INDCOL(NC))
       ENDDO row
    ENDDO col
    RETURN
  END SUBROUTINE CP_ADD_GRID

!***********************SUBROUTINE SETUNB ******************************
!
! set contribution of unbalanced lattic-vectors in an array to zero
! if wrap-arround errors are ommited the contribution is zero
!  operates on the reduced half grid
!**********************************************************************

  SUBROUTINE SETUNB(C,GRID)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

    COMPLEX(q) C(GRID%RC%NROW,GRID%RC%NCOL)

    N2= (GRID%NGY/2)+1
    N3= (GRID%NGZ/2)+1

    DO NC=1,GRID%RC%NCOL
       IF (GRID%RC%I2(NC)==N2) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
       IF (GRID%RC%I3(NC)==N3) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
    ENDDO

    N1= (GRID%NGX/2)+1
    DO NC=1,GRID%RC%NCOL
       C(N1,NC)=0
    ENDDO

    RETURN
  END SUBROUTINE SETUNB


 SUBROUTINE SETUNB_REAL(C,GRID)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

    REAL(q) C(GRID%RC%NROW,GRID%RC%NCOL)

    N2= (GRID%NGY/2)+1
    N3= (GRID%NGZ/2)+1

    DO NC=1,GRID%RC%NCOL
       IF (GRID%RC%I2(NC)==N2) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
       IF (GRID%RC%I3(NC)==N3) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
    ENDDO

    N1= (GRID%NGX/2)+1
    DO NC=1,GRID%RC%NCOL
       C(N1,NC)=0
    ENDDO

    RETURN
  END SUBROUTINE SETUNB_REAL


!***********************SUBROUTINE SETUNB ******************************
!
! set contribution of unbalanced lattic-vectors in an array to zero
! if wrap-arround errors are ommited the contribution is zero
!  operates on the reduced half grid
!
!**********************************************************************

  SUBROUTINE SETUNB_COMPAT(C,GRID)
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID

    COMPLEX(q) C(GRID%RC%NROW,GRID%RC%NCOL)

    IF (.NOT. LCOMPAT) RETURN

    N2= (GRID%NGY/2)+1
    N3= (GRID%NGZ/2)+1

    DO NC=1,GRID%RC%NCOL
       IF (GRID%RC%I2(NC)==N2) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
       IF (GRID%RC%I3(NC)==N3) THEN
          DO N1=1,GRID%RC%NROW
             C(N1,NC)=0
          ENDDO
       ENDIF
    ENDDO

    N1= (GRID%NGX/2)+1
    DO NC=1,GRID%RC%NCOL
       C(N1,NC)=0
    ENDDO

    RETURN
  END SUBROUTINE SETUNB_COMPAT

!***********************************************************************
!
!  OPSYNC does nothing
!  just returns
!  this function is called (for instance) from xcgrad.F
!  to avoid that the compiler is too clever
!
!***********************************************************************

  SUBROUTINE OPSYNC(CSRC,CDEST,NPLWV)
    USE prec
    IMPLICIT NONE
    INTEGER NPLWV
    RGRID CSRC(NPLWV),CDEST(NPLWV)

    RETURN
  END SUBROUTINE OPSYNC

!***********************************************************************
!
!  sum all points on the grid and dump onto screen
!
!***********************************************************************

  SUBROUTINE SUM_RC(STRING,C,GRID)
    USE mgrid
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    COMPLEX(q) C(GRID%RC%NROW,GRID%RC%NCOL)
    CHARACTER (LEN=*) STRING

    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=GRID%COMM%NODE_ME
    IONODE =GRID%COMM%IONODE
#endif
    CSUM=0
    DO NC=1,GRID%RC%NCOL
       DO N1=1,GRID%RC%NROW
          CSUM=CSUM+C(N1,NC)
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(GRID%COMM, CSUM,1))
    io_begin
    WRITE(*,'("SUM_RC:",A,2E20.10)') STRING,CSUM
    io_end
    RETURN
  END SUBROUTINE SUM_RC

!***********************************************************************
!
!  sum all points on the grid and dump onto screen
!
!***********************************************************************

  SUBROUTINE SUM_IN(STRING,C,GRID)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    COMPLEX(q) C(GRID%IN%NCOL,GRID%IN%NROW)
    CHARACTER (LEN=*) STRING

    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=GRID%COMM%NODE_ME
    IONODE =GRID%COMM%IONODE
#endif
    CSUM=0
    DO NC=1,GRID%IN%NCOL
       DO N1=1,GRID%IN%NROW
          CSUM=CSUM+C(NC,N1)
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(GRID%COMM, CSUM,1))
    io_begin
    WRITE(*,'("SUM_IN:",A,2E20.10)') STRING,CSUM
    io_end
    RETURN
  END SUBROUTINE SUM_IN

!***********************************************************************
!
!  sum all points on the real grid and dump onto screen
!
!***********************************************************************

  SUBROUTINE SUM_RL(STRING,C,GRID)
    USE prec
    USE mpimy
    USE mgrid
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    RGRID C(GRID%RL%NROW,GRID%RL%NCOL)
    CHARACTER (LEN=*) STRING

    NODE_ME=0
    IONODE =0
#ifdef MPI
    NODE_ME=GRID%COMM%NODE_ME
    IONODE =GRID%COMM%IONODE
#endif
    CSUM=0
    DO NC=1,GRID%RL%NCOL
       DO N1=1,GRID%RL%NROW
          CSUM=CSUM+C(N1,NC)
       ENDDO
    ENDDO
    CALLMPI( M_sum_z(GRID%COMM, CSUM,1))
    io_begin
    WRITE(*,'("SUM_RL:",A,4E20.10)') STRING,CSUM/GRID%NPLWV,CSUM
    io_end
    RETURN
  END SUBROUTINE SUM_RL


!*************************SUBROUTINE INIDAT ****************************
!  subroutine to initialize a workarray to some non zero data
!***********************************************************************

  SUBROUTINE INIDAT(NPLWV,CWORK)
    USE prec
    COMPLEX(q)  CWORK(NPLWV)

    DO M=1,NPLWV
       CWORK(M)=(0.1_q,0.2_q)
    ENDDO
    RETURN
  END SUBROUTINE INIDAT

!*************************SUBROUTINE INIDATR ***************************
!  subroutine to initialize a workarray to some non zero data
!***********************************************************************

  SUBROUTINE INIDATR(NPLWV,CWORK)
    USE prec
    GDEF  CWORK(NPLWV)

    DO M=1,NPLWV
       CWORK(M)=(0.1_q,0.2_q)
    ENDDO
    RETURN
  END SUBROUTINE INIDATR

!***********************SUBROUTINE WRT_RC_LINE ************************
!
!  dump the contents of an array (defined in reciprocal space)
!  along three lines to an specified unit
!  parallel version currently no support
!
!**********************************************************************

  SUBROUTINE WRT_RC_LINE(IU,GRID,CHDEN)
    USE prec
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (grid_3d) GRID
    COMPLEX(q) CHDEN(GRID%NGX_rd,GRID%NGY,GRID%NGZ_rd)
#ifndef MPI
    NGX=GRID%NGX
    NGY=GRID%NGY
    NGZ=GRID%NGZ

    WRITE(IU,'(2Hx ,10F10.4)') (CHDEN(NX,1,1) ,NX=1,(NGX/2)+1)
    WRITE(IU,'(2Hy ,10F10.4)') (CHDEN(1,NY,1) ,NY=1,(NGY/2)+1)
    WRITE(IU,'(2Hz ,10F10.4)') (CHDEN(1,1,NZ) ,NZ=1,(NGZ/2)+1)
#endif
    RETURN
  END SUBROUTINE WRT_RC_LINE

!***********************SUBROUTINE WRT_RC_LINE ************************
!
!  dump the contents of an array (defined in real space)
!  along three lines to an specified unit
!  parallel version currently no support
!
!**********************************************************************

  SUBROUTINE WRT_RL_LINE(IU,GRID,CHDEN)
    USE prec
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (grid_3d) GRID
    REAL(q) CHDEN(GRID%NGX,GRID%NGY,GRID%NGZ)

    IF (GRID%LREAL) THEN
       NGX=GRID%NGX
       NGY=GRID%NGY
       NGZ=GRID%NGZ
       IF (IU>=0) THEN
          WRITE(IU,*) 'real'
          WRITE(IU,'(2Hx ,10F10.4)') (REAL( CHDEN(NX,1,1) ,KIND=q) ,NX=1,NGX/2+1)
          WRITE(IU,'(2Hy ,10F10.4)') (REAL( CHDEN(1,NY,1) ,KIND=q) ,NY=1,NGY/2+1)
          WRITE(IU,'(2Hz ,10F10.4)') (REAL( CHDEN(1,1,NZ) ,KIND=q) ,NZ=1,NGZ/2+1)
       ENDIF
    ELSE
       CALL WRT_RL_LINE_C(IU,GRID,CHDEN)
    ENDIF
  END SUBROUTINE WRT_RL_LINE

  SUBROUTINE WRT_RL_LINE_C(IU,GRID,CHDEN)
    USE prec
    USE mgrid
    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (grid_3d) GRID
    COMPLEX(q) CHDEN(GRID%NGX,GRID%NGY,GRID%NGZ)

    NGX=GRID%NGX
    NGY=GRID%NGY
    NGZ=GRID%NGZ
    WRITE(IU,*) 'real'
    WRITE(IU,'(2Hx ,10F10.4)') (REAL( CHDEN(NX,1,1) ,KIND=q) ,NX=1,NGX)
    WRITE(IU,'(2Hy ,10F10.4)') (REAL( CHDEN(1,NY,1) ,KIND=q) ,NY=1,NGY)
    WRITE(IU,'(2Hz ,10F10.4)') (REAL( CHDEN(1,1,NZ) ,KIND=q) ,NZ=1,NGZ)
    
    WRITE(IU,*) 'imag'
    WRITE(IU,'(2Hx ,10F10.4)') (AIMAG( CHDEN(NX,1,1)) ,NX=1,NGX)
    WRITE(IU,'(2Hy ,10F10.4)') (AIMAG( CHDEN(1,NY,1)) ,NY=1,NGY)
    WRITE(IU,'(2Hz ,10F10.4)') (AIMAG( CHDEN(1,1,NZ)) ,NZ=1,NGZ)
  END SUBROUTINE WRT_RL_LINE_C
