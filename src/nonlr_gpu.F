!#define nonlr_single
#include "symbol.inc"
!***********************************************************************
!
!  this module contains all the routines required to support 
!  real space presentation of the non local projection operators
!  on an equally spaced grid
!
! GPU part : HACENE Mohamed
!***********************************************************************
MODULE nonlr_struct_def
  USE iso_c_binding
  USE prec

   !  structure required to support non local projection operators in real space

  TYPE nonlr_proj
     REAL(q), POINTER :: PSPRNL(:,:,:)
     INTEGER, POINTER :: LPS(:)
  END TYPE nonlr_proj

  TYPE nonlr_struct
     !only NONLR_S
     LOGICAL LREAL                 ! structure set up ?
     INTEGER NTYP                  ! number of types
     INTEGER NIONS                 ! number of ions
     INTEGER SELECTED_ION          ! allows to generate a projector for a single ion
     INTEGER IRMAX                 ! maximum number points in sphere
     INTEGER IRALLOC               ! size for allocation =IRMAX*LMDIM*NIONS
     INTEGER NK                    ! kpoint for which CRREXP is set up
     INTEGER, POINTER :: NITYP(:)  ! number of ions for each type
     INTEGER, POINTER :: ITYP(:)   ! type for each ion
     INTEGER, POINTER :: LMAX(:)   ! max l-quantum number for each type
     INTEGER, POINTER :: LMMAX(:)  ! number lmn-quantum numbers for each type
     INTEGER, POINTER ::CHANNELS(:)! number of ln-quantum for each type
     REAL(q), POINTER :: PSRMAX(:) ! real space cutoff
     REAL(q), POINTER :: RSMOOTH(:)! radius for smoothing the projectors around each point
     INTEGER, POINTER :: NLIMAX(:) ! maximum index for each ion
     INTEGER, POINTER :: NLI(:,:)  ! index for gridpoints
     INTEGER(c_intptr_t)       :: NLI_GPU
     REAL(qn),POINTER :: RPROJ(:)  ! projectors on real space grid
     REAL(q), POINTER :: POSION(:,:)! positions (required for setup) usually => T_INFO%POSION
     REAL(q), POINTER :: VKPT_SHIFT(:,:)
                                   ! k-point shift for each ion
     TYPE(nonlr_proj), POINTER :: BETA(:) ! a set of structures containing pointers to 
     ! the parameters for the non local projection operators
     LOGICAL LSPIRAL               ! do we want to calculate spin spirals
     ! these arrays allow loops over all ions to be restructured from
     ! nested "over-types" + "over-ions-of-type" to straightforward
     ! loops "over-all-ions" (used in many subroutines).
     INTEGER, POINTER :: LMBASE(:)
     INTEGER, POINTER :: NLIBASE(:)

     COMPLEX(q),POINTER :: CRREXP(:,:,:) ! phase factor exp (i k (R(ion)-r(grid)))
     ! device arrays in RACC0MU_GPU and RPROMU_GPU
     INTEGER(c_intptr_t) :: CRREXP_GPU      ! device pointer
     INTEGER(c_intptr_t) :: RPROJ_GPU       ! device pointer
     INTEGER(c_intptr_t) :: CUDA_NLIMAX     ! device pointer
     ! local work arrays on device
     INTEGER(c_intptr_t) :: CPROJ_GPU
     INTEGER(c_intptr_t) :: LDO_INDEX_GPU
     INTEGER(c_intptr_t) :: WORK_GPU
     INTEGER(c_intptr_t) :: TMP_GPU
     ! host arrays local on host
#ifdef USE_PINNED_MEMORY
     GDEF,POINTER   :: CPROJ(:,:)
     type(C_PTR) :: CPROJ_PTR
     GDEF :: fake
     INTEGER,POINTER :: LDO_INDEX(:)
     type(C_PTR) :: LDO_INDEX_PTR
#else
     GDEF,ALLOCATABLE   :: CPROJ(:,:)
     INTEGER,ALLOCATABLE :: LDO_INDEX(:)
#endif

#if USE_PINNED_MEMORY
    type(C_PTR) :: RPROJ_PTR
    type(C_PTR) :: CRREXP_PTR
    type(C_PTR) :: NLI_PTR     
#endif

  END TYPE nonlr_struct

  TYPE smoothing_handle
     INTEGER :: N                  ! number of grid points
     REAL(q), POINTER :: WEIGHT(:) ! weight of each grid point
     REAL(q), POINTER :: X1(:), X2(:), X3(:) ! positions of additional grid points in fractional coordinates
  END TYPE smoothing_handle
END MODULE nonlr_struct_def

MODULE nonlr
  USE prec
  USE wave
  USE mgrid
  USE nonlr_struct_def

  INTERFACE
     SUBROUTINE RACC0(NONLR_S, WDES1, CPROJ_LOC, CRACC)
       USE nonlr_struct_def
       USE wave
       TYPE (nonlr_struct) NONLR_S
       TYPE (wavedes1)     WDES1
       COMPLEX(q) CRACC
       GDEF       CPROJ_LOC
     END SUBROUTINE RACC0
  END INTERFACE

  INTERFACE
     SUBROUTINE RACC0_HF(NONLR_S, WDES1, CPROJ_LOC, CRACC)
       USE nonlr_struct_def
       USE wave
       TYPE (nonlr_struct) NONLR_S
       TYPE (wavedes1)     WDES1
       GDEF   CRACC
       GDEF   CPROJ_LOC
     END SUBROUTINE RACC0_HF
  END INTERFACE

  INTERFACE
     SUBROUTINE RACC0MU(NONLR_S, WDES1, CPROJ_LOC, CRACC, LD, NSIM, LDO)
       USE nonlr_struct_def
       USE wave
       TYPE (nonlr_struct) NONLR_S
       TYPE (wavedes1)     WDES1
       COMPLEX(q) CRACC
       INTEGER    LD
       GDEF       CPROJ_LOC
       INTEGER    NSIM
       LOGICAL LDO(*)
     END SUBROUTINE RACC0MU
  END INTERFACE

  INTERFACE
     SUBROUTINE RACC0MU_HF(NONLR_S, WDES1, CPROJ_LOC, LD1, CRACC, LD2, NSIM)
       USE nonlr_struct_def
       USE wave
       TYPE (nonlr_struct) NONLR_S
       TYPE (wavedes1)     WDES1
       GDEF       CRACC
       INTEGER    LD1,LD2
       GDEF       CPROJ_LOC
       INTEGER    NSIM
     END SUBROUTINE RACC0MU_HF
  END INTERFACE

CONTAINS


!****************** subroutine NONLR_SETUP ****************************
!
! NONLR_SETUP is the base initialisation routine
! o it sets the number of types and the number of ions
! o it links the positions descriptors to the T_INFO structure
! o it links the tables to the pseudpotential structure
!
! before the data structure can be used in actual calculations
! the following calls must be made:
! ) REAL_OPTLAY  determine the number of grid point in the
!                real space cutoff spheres
! ) NONLR_ALLOC  allocate the required tables 
! ) SPHERE       set the tables for the fast evaluation of the
!                non local projetors
! 
!***********************************************************************

  SUBROUTINE  NONLR_SETUP(NONLR_S,T_INFO,P,LREAL,LSPIRAL,COMM)
    USE pseudo
    USE poscar
    USE mpimy
    IMPLICIT NONE


    TYPE (nonlr_struct) NONLR_S
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    LOGICAL LREAL
    LOGICAL LSPIRAL
    TYPE (communic), TARGET, OPTIONAL :: COMM
    INTEGER, EXTERNAL :: MAXL1
    ! local var
    INTEGER NT


    NONLR_S%LREAL  =LREAL
    NONLR_S%NK     =0
    NONLR_S%NTYP   =T_INFO%NTYP
    NONLR_S%NIONS  =T_INFO%NIONS
    NONLR_S%IRALLOC=-1
    NONLR_S%IRMAX  =-1
    NONLR_S%NITYP  =>T_INFO%NITYP
    NONLR_S%ITYP   =>T_INFO%ITYP
    NONLR_S%POSION =>T_INFO%POSION
    NONLR_S%LSPIRAL=LSPIRAL

    ALLOCATE(NONLR_S%LMAX  (NONLR_S%NTYP), &
         NONLR_S%LMMAX (NONLR_S%NTYP), &
         NONLR_S%CHANNELS(NONLR_S%NTYP), &
         NONLR_S%PSRMAX(NONLR_S%NTYP), &
         NONLR_S%RSMOOTH(NONLR_S%NTYP), &
         NONLR_S%BETA  (NONLR_S%NTYP))

    NULLIFY(NONLR_S%NLIMAX, NONLR_S%NLI, NONLR_S%RPROJ, NONLR_S%CRREXP, NONLR_S%VKPT_SHIFT)

    DO NT=1,T_INFO%NTYP
       NONLR_S%LMAX(NT)    = MAXL1(P(NT))
       NONLR_S%LMMAX(NT)   =P(NT)%LMMAX
       NONLR_S%CHANNELS(NT)=P(NT)%LMAX
       NONLR_S%PSRMAX(NT)  =P(NT)%PSRMAX
       NONLR_S%RSMOOTH(NT) =0
       NONLR_S%BETA(NT)%PSPRNL=>P(NT)%PSPRNL
       NONLR_S%BETA(NT)%LPS   =>P(NT)%LPS
    ENDDO

    NULLIFY(NONLR_S%LMBASE,NONLR_S%NLIBASE)

    NONLR_S%SELECTED_ION=-1

    RETURN
  END SUBROUTINE NONLR_SETUP


!****************** subroutine NONLR_ALLOC *****************************
!
! allocate required work arrays
! this function can be called only after NONLR_SETUP
! since it requires the number of data points in the real space cutoff
! spheres (NONLR_S%IRMAX and NONLR_S%IRALLOC)
!
!***********************************************************************

  SUBROUTINE  NONLR_ALLOC(NONLR_S)
    USE iso_c_binding
    USE cuda_interface
    USE pseudo
    USE ini
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    INTEGER NIONS,IRMAX
    INTEGER :: ret
    INTEGER :: fakei
    COMPLEX(q) :: fakec
    REAL(q) :: faker

    NIONS = NONLR_S%NIONS
    IRMAX = NONLR_S%IRMAX

    IF (NONLR_S%LREAL) THEN
       IF (NONLR_S%IRMAX<0.OR. NONLR_S%IRALLOC<0) THEN
          WRITE(0,*) 'internal ERROR in NONLR_ALLOC: IRMAX or IRALLOC is not set',NONLR_S%IRMAX,NONLR_S%IRALLOC
          WRITE(0,*) '  call NONLR_SETUP before  NONLR_ALLOC'
          STOP
       ENDIF
#if USE_PINNED_MEMORY
       ! allocate pinned memory
       call nvpinnedmalloc(NONLR_S%RPROJ_PTR,NONLR_S%IRALLOC*int(c_sizeof(faker),c_size_t))
       call nvpinnedmalloc(NONLR_S%NLI_PTR,IRMAX*NIONS*int(c_sizeof(fakei),c_size_t))
       call c_f_pointer (NONLR_S%RPROJ_PTR,NONLR_S%RPROJ,(/NONLR_S%IRALLOC/)) 
       call c_f_pointer (NONLR_S%NLI_PTR,NONLR_S%NLI,(/IRMAX,NIONS/))

       ALLOCATE(NONLR_S%NLIMAX(NIONS))
#else
       ! allocate host memory
       ALLOCATE(NONLR_S%NLIMAX(NIONS), &
            NONLR_S%NLI   (IRMAX,NIONS), &
            NONLR_S%RPROJ (NONLR_S%IRALLOC))
#endif
       ! allocate device memory
       call cublas_alloc_safety(IRMAX*NIONS,int(c_sizeof(fakei),c_size_t),NONLR_S%NLI_GPU)
       call cublas_alloc_safety(NONLR_S%IRALLOC,int(c_sizeof(faker),c_size_t),NONLR_S%RPROJ_GPU)

       CALL REGISTER_ALLOCATE(8._q*SIZE(NONLR_S%RPROJ), "nonlr-proj")

#ifndef gammareal
       IF (.NOT.NONLR_S%LSPIRAL ) THEN
#if USE_PINNED_MEMORY
          ! allocate pinned memory
          call nvpinnedmalloc(NONLR_S%CRREXP_PTR,IRMAX*NIONS*1*int(c_sizeof(fakec),c_size_t))
          call c_f_pointer (NONLR_S%CRREXP_PTR,NONLR_S%CRREXP,(/IRMAX,NIONS,1/))
          NONLR_S%CRREXP = 0;
#else
          ALLOCATE(NONLR_S%CRREXP(IRMAX,NIONS,1))
#endif
          call cublas_alloc_safety(IRMAX*NIONS*1,int(c_sizeof(fakec),c_size_t),NONLR_S%CRREXP_GPU)
         
          CALL REGISTER_ALLOCATE(16._q*SIZE(NONLR_S%CRREXP), "nonlr-proj")
       ELSE
#if USE_PINNED_MEMORY
          call nvpinnedmalloc(NONLR_S%CRREXP_PTR,IRMAX*NIONS*2*int(c_sizeof(fakec),c_size_t))
          call c_f_pointer (NONLR_S%CRREXP_PTR,NONLR_S%CRREXP,(/IRMAX,NIONS,2/))          
#else 
          ALLOCATE(NONLR_S%CRREXP(IRMAX,NIONS,2))
#endif
          CALL REGISTER_ALLOCATE(16._q*SIZE(NONLR_S%CRREXP), "nonlr-proj")
       ENDIF
#endif
       ALLOCATE(NONLR_S%LMBASE(NIONS+1),NONLR_S%NLIBASE(NIONS+1))
    ENDIF
    RETURN
  END SUBROUTINE NONLR_ALLOC

!****************** subroutine NONLR_DEALLOC ***************************
!
! deallocate all work arrays
! but leave the links to the pseudopotentials and ions open
! such that the projectors can be reinitialized by a call to 
! NONLR_SETUP, NONLR_ALLOC and SPHERE
!
!***********************************************************************
  ! DDNVIDIA NONLR_DEALLOC implementation
  SUBROUTINE  NONLR_DEALLOC(NONLR_S)
    USE iso_c_binding
    USE cuda_interface
    USE ini
    IMPLICIT NONE

    COMPLEX(q) :: fakec
    INTEGER :: fakei
    COMPLEX(q) :: faker
#if USE_PINNED_MEMORY
    INTEGER :: ret
#endif

      TYPE (nonlr_struct) NONLR_S

      IF (NONLR_S%LREAL) THEN
         CALL DEREGISTER_ALLOCATE(8._q*SIZE(NONLR_S%RPROJ), "nonlr-proj")

#if USE_PINNED_MEMORY
         call nvpinnedfree(NONLR_S%NLI_PTR)
         call nvpinnedfree(NONLR_S%RPROJ_PTR)
#else 
         DEALLOCATE(NONLR_S%NLIMAX, &
             NONLR_S%NLI   , &
             NONLR_S%RPROJ)
         
        call cublas_free(NONLR_S%NLI_GPU)
        ! call cublas_free(NONLR_S%RPROJ_GPU)

#endif
       ! free device memory
       call cublas_free(NONLR_S%RPROJ_GPU)

       IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(NONLR_S%CRREXP), "nonlr-proj")
#if USE_PINNED_MEMORY
          call nvpinnedfree(NONLR_S%CRREXP_PTR)
#else
          DEALLOCATE(NONLR_S%CRREXP)
#endif
          call cublas_free(NONLR_S%CRREXP_GPU)
       ENDIF
    ENDIF

    NULLIFY(NONLR_S%NLIMAX, NONLR_S%NLI, NONLR_S%RPROJ,NONLR_S%CRREXP )
    RETURN
  END SUBROUTINE NONLR_DEALLOC

!****************** subroutine NONLR_DESTROY ***************************
!
! destroy the  links to the pseudopotentials and release all resources
! allocated by the NONLR_S descriptor
! the subroutine performs all the operations performed by 
! NONLR_DEALLOC
! and destroys all other links as well
!
!***********************************************************************

  SUBROUTINE  NONLR_DESTROY(NONLR_S)
      USE iso_c_binding
      USE cuda_interface
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    !! TODO: destroy also ,NONLR_S%CRREXP_GPU ,NONLR_S%NLI_GPU, ,NONLR_S%RPROJ

    IF (NONLR_S%LREAL) THEN
       IF (ASSOCIATED(NONLR_S%NLIMAX)) DEALLOCATE(NONLR_S%NLIMAX)
#if USE_PINNED_MEMORY
       IF (ASSOCIATED(NONLR_S%NLI)) then
          call nvpinnedfree(NONLR_S%NLI_PTR)
       end IF
       IF (ASSOCIATED(NONLR_S%RPROJ))  call nvpinnedfree(NONLR_S%RPROJ_PTR)
       IF (ASSOCIATED(NONLR_S%CRREXP)) call nvpinnedfree(NONLR_S%CRREXP_PTR)
#else
       IF (ASSOCIATED(NONLR_S%NLI))    DEALLOCATE(NONLR_S%NLI)
       IF (ASSOCIATED(NONLR_S%RPROJ))  DEALLOCATE(NONLR_S%RPROJ)
       IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
          DEALLOCATE(NONLR_S%CRREXP)
       ENDIF
#endif
       IF (ASSOCIATED(NONLR_S%NLIMAX)) DEALLOCATE(NONLR_S%NLIMAX)
    ENDIF
    NULLIFY(NONLR_S%NLIMAX, NONLR_S%NLI, NONLR_S%RPROJ,NONLR_S%CRREXP)

    DEALLOCATE(NONLR_S%LMAX, &
         NONLR_S%LMMAX, &
         NONLR_S%CHANNELS, &
         NONLR_S%PSRMAX, &
         NONLR_S%RSMOOTH, &
         NONLR_S%BETA)

    DEALLOCATE(NONLR_S%LMBASE, &
         NONLR_S%NLIBASE)

    RETURN
  END SUBROUTINE NONLR_DESTROY

!****************** SUBROUTINE NONLR_ASSIGN ****************************
!
! This routine accomplishes the same thing as the assignment:
!
!   NONLR_LHS = NONLR_RHS
!
! would, but allows for the use of "fast"-memory (in the sense that
! the fm_* arrays are excluded from the assignment).
!
!***********************************************************************

  SUBROUTINE NONLR_ASSIGN(NONLR_LHS,NONLR_RHS)
    IMPLICIT NONE
    TYPE (nonlr_struct) NONLR_LHS,NONLR_RHS

    NONLR_LHS%LREAL       = NONLR_RHS%LREAL
    NONLR_LHS%NTYP        = NONLR_RHS%NTYP
    NONLR_LHS%NIONS       = NONLR_RHS%NIONS
    NONLR_LHS%SELECTED_ION= NONLR_RHS%SELECTED_ION
    NONLR_LHS%IRMAX       = NONLR_RHS%IRMAX
    NONLR_LHS%IRALLOC     = NONLR_RHS%IRALLOC
    NONLR_LHS%NK          = NONLR_RHS%NK
    NONLR_LHS%LSPIRAL     = NONLR_RHS%LSPIRAL

    NONLR_LHS%NITYP       =>NONLR_RHS%NITYP
    NONLR_LHS%ITYP        =>NONLR_RHS%ITYP
    NONLR_LHS%LMAX        =>NONLR_RHS%LMAX
    NONLR_LHS%LMMAX       =>NONLR_RHS%LMMAX
    NONLR_LHS%CHANNELS    =>NONLR_RHS%CHANNELS
    NONLR_LHS%PSRMAX      =>NONLR_RHS%PSRMAX
    NONLR_LHS%RSMOOTH     =>NONLR_RHS%RSMOOTH
    NONLR_LHS%POSION      =>NONLR_RHS%POSION
    NONLR_LHS%VKPT_SHIFT  =>NONLR_RHS%VKPT_SHIFT
    NONLR_LHS%BETA        =>NONLR_RHS%BETA

    NONLR_LHS%NLIMAX      =>NONLR_RHS%NLIMAX

    NONLR_LHS%LMBASE      =>NONLR_RHS%LMBASE
    NONLR_LHS%NLIBASE     =>NONLR_RHS%NLIBASE

    NONLR_LHS%NLI         =>NONLR_RHS%NLI
    NONLR_LHS%RPROJ       =>NONLR_RHS%RPROJ
    NONLR_LHS%CRREXP      =>NONLR_RHS%CRREXP

    RETURN
  END SUBROUTINE NONLR_ASSIGN

!****************** subroutine REAL_OPTLAY *****************************
!
! determine 
! NONLR_S%IRMAX and NONLR_S%IRALLOC in the non local PP structure
!
! for the parallel version the subroutine also optimizes
! the layout (i.e. data distribution) of the data points (columns)
! in real space such that the total number of grid points
! is the same on all nodes (this requires an update of the GRID structure)
! if LNOREDIS is set the data layout is, however,
! not allowed to change in the GRID structure
!
! NOTE: if the data layout is updated in the GRID structure the parallel
! fast Fourier transformation must be reinitialised
! furthermore  the routine signals to the calling routine whether 
! a reallocation of the work arrays in NONLR_S is required
! via NONLR_ALLOC. This is done be checking the current setting
! of NONLR_S%IRMAX and NONLR_S%IRALLOC
!
!***********************************************************************

  SUBROUTINE REAL_OPTLAY(GRID,LATT_CUR,NONLR_S,LNOREDIS, &
       LREALLOCATE,IU6,IU0)

    USE lattice
    USE constant
    USE pseudo

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)      GRID
    TYPE (latt)         LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    ! local work arrays
    INTEGER, ALLOCATABLE :: USED_ROWS(:,:) ! counts how many elements
    ! must be allocated for one row
    INTEGER, ALLOCATABLE :: REDISTRIBUTION_INDEX(:)
    LOGICAL  LNOREDIS   ! no redistribution allowed
    LOGICAL  LREALLOCATE

    LREALLOCATE=.FALSE.

    IF (.NOT. NONLR_S%LREAL) RETURN
!=======================================================================
! loop over all ions
!=======================================================================
    NLIIND=0
    IRMAX =0
    NIS=1

    IF (GRID%RL%NFAST==3) THEN
       ALLOCATE(USED_ROWS(GRID%NGX,GRID%NGY), &
            REDISTRIBUTION_INDEX(GRID%NGX*GRID%NGY) )
       USED_ROWS=0
       IRALLOC=0  ! number of real space proj on local node
    ENDIF

    type: DO NT=1,NONLR_S%NTYP
       IF (NONLR_S%LMMAX(NT)==0) GOTO 600
       LMMAXC=NONLR_S%LMMAX(NT)
       ions: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1

          IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) CYCLE
          ARGSC=NPSRNL/NONLR_S%PSRMAX(NT)

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
          F1=1._q/GRID%NGX
          F2=1._q/GRID%NGY
          F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
!sh
!          D1= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(1)*GRID%NGX
!          D2= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(2)*GRID%NGY
!          D3= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(3)*GRID%NGZ
          D1= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(1)*GRID%NGX
          D2= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(2)*GRID%NGY
          D3= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(3)*GRID%NGZ

          N3LOW= INT(NONLR_S%POSION(3,NI)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
          N2LOW= INT(NONLR_S%POSION(2,NI)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
          N1LOW= INT(NONLR_S%POSION(1,NI)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

          N3HI = INT(NONLR_S%POSION(3,NI)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
          N2HI = INT(NONLR_S%POSION(2,NI)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
          N1HI = INT(NONLR_S%POSION(1,NI)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
          IF (GRID%RL%NFAST==3) THEN
          IND=1

          DO N2=N2LOW,N2HI
             X2=(N2*F2-NONLR_S%POSION(2,NI))
             N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

             DO N1=N1LOW,N1HI
                X1=(N1*F1-NONLR_S%POSION(1,NI))
                N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

                DO N3=N3LOW,N3HI
                   X3=(N3*F3-NONLR_S%POSION(3,NI))

                   X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                   Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                   Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                   D=SQRT(X*X+Y*Y+Z*Z)
                   ARG=(D*ARGSC)+1
                   NADDR=INT(ARG)

!sh                IF (NADDR<NPSRNL) THEN
                   IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN

                      IND=IND+1
                      USED_ROWS(N1P+1,N2P+1)=USED_ROWS(N1P+1,N2P+1)+LMMAXC
                      ! if on local processor add to IRALLOC
                      IF (GRID%RL%INDEX(N1P,N2P) /=0) IRALLOC=IRALLOC+LMMAXC
                   ENDIF
                ENDDO
             ENDDO
          ENDDO
          ELSE
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
          IND=1
          DO N3=N3LOW,N3HI
             X3=(N3*F3-NONLR_S%POSION(3,NI))
             N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

             DO N2=N2LOW,N2HI
                X2=(N2*F2-NONLR_S%POSION(2,NI))
                N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                NCOL=GRID%RL%INDEX(N2P,N3P)

                DO N1=N1LOW,N1HI
                   X1=(N1*F1-NONLR_S%POSION(1,NI))

                   X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                   Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                   Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                   D=SQRT(X*X+Y*Y+Z*Z)
                   ARG=(D*ARGSC)+1
                   NADDR=INT(ARG)

!sh                IF (NADDR<NPSRNL) THEN
                   IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                      N1P=MOD(N1+10*GRID%NGX,GRID%NGX)
                      NCHECK=N1P+(NCOL-1)*GRID%NGX+1
                      IF (NCHECK /= 1+N1P+GRID%NGX*(N2P+GRID%NGY* N3P)) THEN
                         WRITE(*,*)'REAL_OPT: internal ERROR:',N1P,N2P,N3P, NCOL
                         STOP
                      ENDIF
                      IND=IND+1
                   ENDIF
                ENDDO
             ENDDO
          ENDDO
          ENDIF

          INDMAX=IND-1
          IRMAX =MAX(IRMAX,INDMAX)
          NLIIND=NONLR_S%LMMAX(NT)*INDMAX+NLIIND
       ENDDO ions
600    NIS = NIS+NONLR_S%NITYP(NT)
    ENDDO type
!=======================================================================
! now redistribute rows in MPI version
!=======================================================================
#ifdef MPI
    IF (GRID%RL%NFAST==3) THEN
       ! first check whether everything is ok
       NLISUM=SUM(USED_ROWS)
       IF (NLISUM /= NLIIND) THEN
          WRITE(*,*)'REAL_OPTLAY: internal error (1)',NLISUM,NLIIND
          STOP
       ENDIF
       ! setup redistribution index
       NCOL_TOT=GRID%NGY*GRID%NGX
       DO I=1,NCOL_TOT
          REDISTRIBUTION_INDEX(I)=I
       ENDDO
       IF (.NOT. LNOREDIS .AND. GRID%COMM%NCPU>1 ) THEN
          WRITE(*,*) 'resort distribution'
          CALL SORT_REDIS(NCOL_TOT,USED_ROWS(1,1),REDISTRIBUTION_INDEX(1))
          CALL REAL_OPTLAY_GRID(GRID,     REDISTRIBUTION_INDEX,USED_ROWS(1,1),IRALLOC)
       ENDIF
       IF (IRMAX >  NONLR_S%IRMAX) THEN
          ! IRMAX  is the maximum global number, could be improved !!!!
          NONLR_S%IRMAX   =IRMAX  *1.1
          LREALLOCATE=.TRUE.
       ENDIF
       IF( IRALLOC > NONLR_S%IRALLOC) THEN
          ! more safety on parallel machines increase by 20 %
          NONLR_S%IRALLOC =IRALLOC*1.2
          LREALLOCATE=.TRUE.
       ENDIF

       IALLOC_MAX=  IRALLOC
       IALLOC_MIN= -IRALLOC
       CALLMPI (M_max_i( GRID%COMM, IALLOC_MAX, 1))
       CALLMPI (M_max_i( GRID%COMM, IALLOC_MIN, 1))
       CALLMPI (M_sum_i( GRID%COMM, IRALLOC, 1))
       IALLOC_MIN=-IALLOC_MIN
       AKBYTES=1024/8  ! conversion from 8 byte words to kbytes

       IF (.NOT. LNOREDIS .AND. GRID%COMM%NCPU>1  .AND. IU6>=0) &
            WRITE(IU6,*)'redistribution in real space done'
       IF (.NOT. LNOREDIS .AND. GRID%COMM%NCPU>1 .AND. IU0>=0) &
            WRITE(IU6,*)'redistribution in real space done'
       IF (IU6>=0) &
            WRITE(IU6,1) IRALLOC/AKBYTES,IALLOC_MAX/AKBYTES,IALLOC_MIN/AKBYTES
1      FORMAT(/' real space projection operators:'/ &
            '  total allocation   :',F14.2,' KBytes'/ &
            '  max/ min on nodes  :',2F14.2/)

       IF (NLISUM /= NLIIND) THEN
          WRITE(*,*)'REAL_OPTLAY: internal error (2)',IRALLOC,NLIIND
          STOP
       ENDIF
       DEALLOCATE(USED_ROWS,REDISTRIBUTION_INDEX)
    ELSE
#endif
!=======================================================================
! serial version
!=======================================================================
! to avoid too often reallocation increase values by 10 %
       IF (IRMAX >  NONLR_S%IRMAX) THEN
          NONLR_S%IRMAX   =IRMAX*1.1
          LREALLOCATE=.TRUE.
       ENDIF

       IF( NLIIND > NONLR_S%IRALLOC) THEN
          NONLR_S%IRALLOC =NLIIND*1.1
          LREALLOCATE=.TRUE.
       ENDIF
#ifdef MPI
    ENDIF
#endif
  END SUBROUTINE REAL_OPTLAY


#ifdef MPI
!
! step through all columns and distribute them onto proc.
! in the manner 1 ... NCPU - NCPU ... 1 - 1 ... NCPU - etc.
!
  SUBROUTINE REAL_OPTLAY_GRID(GRID,REDISTRIBUTION_INDEX,USED_ROWS,IRALLOC)
    IMPLICIT REAL(q) (A-H,O-Z)

    TYPE (grid_3d)     GRID
    INTEGER REDISTRIBUTION_INDEX(GRID%NGX*GRID%NGY), &
         USED_ROWS(GRID%NGX*GRID%NGY)
    LOGICAL LUP

    GRID%RL%INDEX= 0
    GRID%RL%NCOL = 0

    NODE_TARGET=0  ! NODE onto which column has to go
    LUP=.TRUE.     ! determines whether NODE_TARGET is increased or decreased
    IRALLOC=0

    NCOL_TOT=GRID%NGY*GRID%NGX
    DO NCOL=1,NCOL_TOT
       IND_REDIS=REDISTRIBUTION_INDEX(NCOL)
       N2=MOD(IND_REDIS-1,GRID%NGX)+1         ! x index (is fast)
       N3=   (IND_REDIS-1)/GRID%NGX+1         ! y index
       IF (LUP) THEN
          IF (NODE_TARGET == GRID%COMM%NCPU) THEN
             LUP=.FALSE.
          ELSE
             NODE_TARGET=NODE_TARGET+1
          ENDIF
       ELSE
          IF (NODE_TARGET == 1) THEN
             LUP=.TRUE.
          ELSE
             NODE_TARGET=NODE_TARGET-1
          ENDIF
       ENDIF

       IND_ON_CPU=(NCOL-1)/GRID%COMM%NCPU+1

       ! element on local node
       ! set up required elements
       IF (NODE_TARGET == GRID%COMM%NODE_ME) THEN
          GRID%RL%NCOL=GRID%RL%NCOL+1
          IF (IND_ON_CPU /= GRID%RL%NCOL) THEN
             WRITE(*,*)'REAL_OPTLAY: internal error(3) ',GRID%COMM%NODE_ME,IND_ON_CPU,GRID%RL%NCOL
             STOP
          ENDIF
          GRID%RL%INDEX(N2-1,N3-1)=IND_ON_CPU
          GRID%RL%I2(IND_ON_CPU)=N2 ! I2 contains x index
          GRID%RL%I3(IND_ON_CPU)=N3 ! I3      the y index
          IRALLOC=IRALLOC+USED_ROWS(NCOL)
       ENDIF
    ENDDO

  END SUBROUTINE REAL_OPTLAY_GRID
#endif


!****************** subroutine RSPHER  *********************************
!
!  subroutine RSPHER calculates the sperical harmonics multiplied
!  by the radial projector function in real space
!  the result is the real space projection operator NONLR_S
!    RPROJ = 1/Omega ^(1/2) Xi(r-R(N)) Y_lm(r-R(N) exp(i k r-R(N))
!
!  all ions can be displaced by a constant shift to allow
!  the evaluation of the first derivative of the projector functions
!
!  RSPHER is the simple interface
!  whereas RSPHER_ALL allows for finite difference calculations
!
!  IDISPL:
!     0 set projector function
!     1 use finite differences to calculate the derivative of
!       the projector function with respect to the specified displacement
!  LOMEGA: use 1/volume scaling (required for FAST_AUG)
!
!***********************************************************************

  SUBROUTINE RSPHER(GRID,NONLR_S, LATT_CUR )
    USE mpimy
    USE lattice
    USE constant

    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (wavedes)     WDES
    INTEGER NK
    REAL(q)   DISPL(3,NONLR_S%NIONS)

    DISPL=0
    CALL RSPHER_ALL(GRID,NONLR_S, LATT_CUR, LATT_CUR, LATT_CUR, &
         DISPL,DISPL, 0)
    RETURN
  END SUBROUTINE RSPHER

  SUBROUTINE RSPHER_ALL(GRID,NONLR_S,LATT_FIN1, LATT_FIN2, LATT_CUR, &
       DISPL1, DISPL2, IDISPL, LOMEGA)
  USE pseudo
  USE mpimy
  USE lattice
  USE constant
  USE asa
  IMPLICIT COMPLEX(q) (C)
  IMPLICIT REAL(q) (A-B,D-H,O-Z)

  TYPE (nonlr_struct) NONLR_S
  TYPE (grid_3d)     GRID
  TYPE (latt)        LATT_CUR,LATT_FIN1,LATT_FIN2,LATT_FIN
  INTEGER NK
  INTEGER IDISPL      ! 0 no finite differences, 1 finite differences
  LOGICAL, OPTIONAL :: LOMEGA
  ! work arrays
  REAL(q),ALLOCATABLE :: DIST(:),XS(:),YS(:),ZS(:),VPS(:),YLM(:,:),VYLM(:)
  REAL(q) :: DISPL1(3,NONLR_S%NIONS),DISPL2(3,NONLR_S%NIONS)
  REAL(q) :: DISPL(3)
  TYPE (smoothing_handle) :: SH
  INTEGER :: ISH


  LYDIM=MAXVAL(NONLR_S%LMAX)
  LMYDIM=(LYDIM+1)**2          ! number of lm pairs

  LMMAX =MAXVAL(NONLR_S%LMMAX) ! number of nlm indices in the non local potential
  IRMAX=NONLR_S%IRMAX

  ALLOCATE(DIST(IRMAX),XS(IRMAX),YS(IRMAX),ZS(IRMAX),VPS(IRMAX),YLM(IRMAX,LMYDIM), &
       VYLM(IRMAX*LMMAX))

  CALL RSPHER_SMOOTH( SH, NONLR_S , GRID, LATT_CUR )

!sh added statments
  NONLR_S%RPROJ=0
  smooth: DO ISH=1,SH%N
!=======================================================================
! loop over all ions
!=======================================================================
    NLIIND=0
    NIS=1

    type: DO NT=1,NONLR_S%NTYP
       ! DDNVIDIA max value of INDMAX for an atom type
       MAX_INDMAX=0
       IF (NONLR_S%LMMAX(NT)==0) GOTO 600
       ions: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1

          IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) THEN
             NONLR_S%NLIMAX(NI)=0
             CYCLE
          ENDIF

          ARGSC=NPSRNL/NONLR_S%PSRMAX(NT)
!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
          F1=1._q/GRID%NGX
          F2=1._q/GRID%NGY
          F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
!sh
!          D1= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(1)*GRID%NGX
!          D2= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(2)*GRID%NGY
!          D3= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(3)*GRID%NGZ
          D1= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(1)*GRID%NGX
          D2= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(2)*GRID%NGY
          D3= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(3)*GRID%NGZ

          N3LOW= INT(NONLR_S%POSION(3,NI)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
          N2LOW= INT(NONLR_S%POSION(2,NI)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
          N1LOW= INT(NONLR_S%POSION(1,NI)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

          N3HI = INT(NONLR_S%POSION(3,NI)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
          N2HI = INT(NONLR_S%POSION(2,NI)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
          N1HI = INT(NONLR_S%POSION(1,NI)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX

          VYLM= 0

          dis: DO IDIS=-ABS(IDISPL),ABS(IDISPL),2

             IF (IDIS==-1) THEN
                LATT_FIN=LATT_FIN1
                DISPL=DISPL1(:,NI)
             ELSE IF (IDIS==1) THEN
                LATT_FIN=LATT_FIN2
                DISPL=DISPL2(:,NI)
             ELSE
                LATT_FIN=LATT_CUR
                DISPL=0
             ENDIF
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
             IF (GRID%RL%NFAST==3) THEN
             IND=1

             DO N2=N2LOW,N2HI
                X2=(N2*F2-NONLR_S%POSION(2,NI))
                N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                DO N1=N1LOW,N1HI
                   X1=(N1*F1-NONLR_S%POSION(1,NI))
                   N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

                   NCOL=GRID%RL%INDEX(N1P,N2P)
                   IF (NCOL==0) CYCLE ! not on local node go on
                   IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
                      WRITE(*,*)'RSPHER: internal ERROR:', &
                           GRID%RL%I2(NCOL),N1P+1, GRID%RL%I3(NCOL),N2P+1
                      STOP
                   ENDIF
!OCL SCALAR
                   DO N3=N3LOW,N3HI
                      X3=(N3*F3-NONLR_S%POSION(3,NI))

                      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(XC*XC+YC*YC+ZC*ZC)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)
!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         X= (X1+SH%X1(ISH))*LATT_FIN%A(1,1)+(X2+SH%X2(ISH))*LATT_FIN%A(1,2)+(X3+SH%X3(ISH))*LATT_FIN%A(1,3)
                         Y= (X1+SH%X1(ISH))*LATT_FIN%A(2,1)+(X2+SH%X2(ISH))*LATT_FIN%A(2,2)+(X3+SH%X3(ISH))*LATT_FIN%A(2,3)
                         Z= (X1+SH%X1(ISH))*LATT_FIN%A(3,1)+(X2+SH%X2(ISH))*LATT_FIN%A(3,2)+(X3+SH%X3(ISH))*LATT_FIN%A(3,3)
!sh end
                         N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)
                         NONLR_S%NLI (IND,NI) =1+N3P+ GRID%NGZ*(NCOL-1)

                         ZZ=Z-DISPL(3)
                         YY=Y-DISPL(2)
                         XX=X-DISPL(1)
                         ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
                         ! was done using the well known formula  | R+d | = | R | + d . R/|R|
                         ! this improves the stability of finite differences considerable
                         !IF (D<1E-4_q) THEN
                         !  DIST(IND)=1E-4_q
                         !ELSE
                         !  DIST(IND)=MAX(D-IDIS*(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
                         !ENDIF
                         DIST(IND)=MAX(SQRT(XX*XX+YY*YY+ZZ*ZZ),1E-10_q)

                         XS(IND)  =XX/DIST(IND)
                         YS(IND)  =YY/DIST(IND)
                         ZS(IND)  =ZZ/DIST(IND)
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ELSE
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
             IND=1
             DO N3=N3LOW,N3HI
                X3=(N3*F3-NONLR_S%POSION(3,NI))
                N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

                DO N2=N2LOW,N2HI
                   X2=(N2*F2-NONLR_S%POSION(2,NI))
                   N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                   NCOL=GRID%RL%INDEX(N2P,N3P)

                   DO N1=N1LOW,N1HI
                      X1=(N1*F1-NONLR_S%POSION(1,NI))

                      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(XC*XC+YC*YC+ZC*ZC)
!sh
!                      ARG=(D*ARGSC)+1
!                      NADDR=INT(ARG)
!                      IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         X= (X1+SH%X1(ISH))*LATT_FIN%A(1,1)+(X2+SH%X2(ISH))*LATT_FIN%A(1,2)+(X3+SH%X3(ISH))*LATT_FIN%A(1,3)
                         Y= (X1+SH%X1(ISH))*LATT_FIN%A(2,1)+(X2+SH%X2(ISH))*LATT_FIN%A(2,2)+(X3+SH%X3(ISH))*LATT_FIN%A(2,3)
                         Z= (X1+SH%X1(ISH))*LATT_FIN%A(3,1)+(X2+SH%X2(ISH))*LATT_FIN%A(3,2)+(X3+SH%X3(ISH))*LATT_FIN%A(3,3)
!shend

                         N1P=MOD(N1+10*GRID%NGX,GRID%NGX)
                         NONLR_S%NLI (IND,NI) =N1P+(NCOL-1)*GRID%NGX+1
                         IF (NONLR_S%NLI (IND,NI) /= 1+N1P+GRID%NGX*(N2P+GRID%NGY* N3P)) THEN
                            WRITE(*,*)'RSHPER internal ERROR:',N1P,N2P,N3P, NCOL
                            STOP
                         ENDIF

                         ZZ=Z-DISPL(3)
                         YY=Y-DISPL(2)
                         XX=X-DISPL(1)
                         ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
                         ! was done using the well known formula  | R+d | = | R | + d . R/|R|
                         ! this improves the stability of finite differences considerable
                         !IF (D<1E-4_q) THEN
                         !  DIST(IND)=1E-4_q
                         !ELSE
                         !  DIST(IND)=MAX(D-IDIS*(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
                         !ENDIF
                         DIST(IND)=MAX(SQRT(XX*XX+YY*YY+ZZ*ZZ),1E-10_q)

                         XS(IND)  =XX/DIST(IND)
                         YS(IND)  =YY/DIST(IND)
                         ZS(IND)  =ZZ/DIST(IND)
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ENDIF
!-----------------------------------------------------------------------
!  compare maximum index with INDMAX
!-----------------------------------------------------------------------
             INDMAX=IND-1
             IF (INDMAX>NONLR_S%IRMAX) THEN
                WRITE(*,*)'internal ERROR: RSPHER:  NONLR_S%IRMAX must be increased to', &
                     &            INT(INDMAX*1.1_q)
                STOP
             ENDIF
             NONLR_S%NLIMAX(NI)=INDMAX
!=======================================================================
! now calculate the tables containing the spherical harmonics
! multiplied by the pseudopotential
!=======================================================================
             LYDIM=NONLR_S%LMAX(NT)
             CALL SETYLM(LYDIM,INDMAX,YLM,XS,YS,ZS)

             LMIND=1
             l_loop: DO L=1,NONLR_S%CHANNELS(NT)
!-----------------------------------------------------------------------
! interpolate the non-local pseudopotentials
! and multiply by (LATT_CUR%OMEGA)^(1/2)
! interpolation is done here using spline-fits this inproves the
! numerical stability of the forces the MIN operation takes care
! that the index is between  1 and NPSRNL
!-----------------------------------------------------------------------
                FAKT= SQRT(LATT_FIN%OMEGA)
                IF (PRESENT (LOMEGA)) THEN
                   IF (LOMEGA) FAKT=LATT_FIN%OMEGA
                ENDIF                   
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   I  =MIN(INT(DIST(IND)*ARGSC)+1,NPSRNL-1)
                   REM=DIST(IND)-NONLR_S%BETA(NT)%PSPRNL(I,1,L)
                   VPS(IND)=(NONLR_S%BETA(NT)%PSPRNL(I,2,L)+REM*(NONLR_S%BETA(NT)%PSPRNL(I,3,L)+ &
                        &         REM*(NONLR_S%BETA(NT)%PSPRNL(I,4,L)+REM*NONLR_S%BETA(NT)%PSPRNL(I,5,L))))*FAKT
                ENDDO

                LL=NONLR_S%BETA(NT)%LPS(L)
                MMAX=2*LL

                ! invert sign for first displacement
                IF (IDIS==-1) THEN
                   DO IND=1,INDMAX
                      VPS(IND)=-VPS(IND)
                   ENDDO
                ENDIF

                LMBASE=LL**2+1

                DO LM=0,MMAX
                   DO IND=1,INDMAX
                      IBAS = (LMIND-1+LM)*INDMAX
                      VYLM(IBAS+IND)=VYLM(IBAS+IND)+VPS(IND)*YLM(IND,LM+LMBASE)
                   ENDDO
                ENDDO

                LMIND=LMIND+MMAX+1
             ENDDO l_loop

             IF (LMIND-1/=NONLR_S%LMMAX(NT)) THEN
                WRITE(*,*)'internal ERROR: RSPHER:  NONLR_S%LMMAX is wrong',LMIND-1,NONLR_S%LMMAX(NT)
                STOP
             ENDIF

             IF ( NONLR_S%LMMAX(NT)*INDMAX+NLIIND >= NONLR_S%IRALLOC) THEN
                WRITE(*,*)'internal ERROR RSPHER:', &
                     'running out of buffer ',NLIIND,INDMAX,NONLR_S%LMMAX(NT),NT,NONLR_S%IRALLOC
#ifdef MPI
                     CALL M_stop("nonlr.F:Out of buffer RSPHER")
#endif

                STOP
             ENDIF

          ENDDO dis
!-----------------------------------------------------------------------
! finally store the coefficients
!-----------------------------------------------------------------------
! DDNVIDIA update max INDMAX
          MAX_INDMAX=MAX(MAX_INDMAX,INDMAX)

          DO LMIND=1,NONLR_S%LMMAX(NT)
             DO IND=1,INDMAX
                IBAS = (LMIND-1)*INDMAX
!sh
                NONLR_S%RPROJ(IND+IBAS+NLIIND)=NONLR_S%RPROJ(IND+IBAS+NLIIND)+VYLM(IND+IBAS)*SH%WEIGHT(ISH)
             ENDDO
          ENDDO

          NLIIND= NONLR_S%LMMAX(NT)*INDMAX+NLIIND
!=======================================================================
! end of loop over ions
!=======================================================================
       ENDDO ions
600    NIS = NIS+NONLR_S%NITYP(NT)
    ENDDO type
  ENDDO smooth

!=======================================================================
! Here we set LMBASE and NLIBASE. These arrays of offsets allow
! us to contract nested loops over types+ions_of_type to single loops
! over all ions (used lateron in for instance RPROMU_HF).
!=======================================================================
    NONLR_S%NLIBASE(:)=0
    NONLR_S%LMBASE(1)=0
    DO NI=1,NONLR_S%NIONS
       NT=NONLR_S%ITYP(NI)
       NONLR_S%LMBASE(NI+1)=NONLR_S%LMBASE(NI)+NONLR_S%LMMAX(NT)
       IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) THEN
          NONLR_S%NLIBASE(NI+1)=NONLR_S%NLIBASE(NI)
       ELSE
          NONLR_S%NLIBASE(NI+1)=NONLR_S%NLIBASE(NI)+NONLR_S%LMMAX(NT)*NONLR_S%NLIMAX(NI)
       ENDIF
    ENDDO

  DEALLOCATE(DIST,XS,YS,ZS,VPS,YLM,VYLM)
  
  CALL RSPHER_SMOOTH_DEALLOCATE( SH )

END SUBROUTINE RSPHER_ALL

!****************** subroutine RSPHER_SMOOTH ***************************
!
! subroutine to smooth the real space projector functions
! using a simple real space method
!
!***********************************************************************


SUBROUTINE RSPHER_SMOOTH( SH, NONLR_S, GRID, LATT_CUR)
  USE lattice
  USE constant
  IMPLICIT NONE
  TYPE (smoothing_handle) SH
  TYPE (nonlr_struct) NONLR_S
  TYPE (grid_3d)     GRID
  TYPE (latt)        LATT_CUR
  
  INTEGER :: N1, N2, N3, I1, I2, I3, NT
  INTEGER, PARAMETER :: IREFINE=4
  REAL(q) :: F1, F2, F3, X, Y, Z, D
  INTEGER, PARAMETER :: NMAX=2000
  REAL(q) :: X1(NMAX), X2(NMAX), X3(NMAX), WEIGHT(NMAX)
  REAL(q) :: RSMOOTH

! currently only one fixed value for 
!   RSMOOTH is allowed
! check that they are all the same

  RSMOOTH=NONLR_S%RSMOOTH(1)

  DO NT=1,SIZE(NONLR_S%RSMOOTH)
     IF (RSMOOTH/=NONLR_S%RSMOOTH(NT)) THEN
        WRITE(*,*)'RSPHER_SMOOTH: internal error only one value for RSMOOTH allowed',RSMOOTH,NONLR_S%RSMOOTH(NT),NT
        STOP
     ENDIF
  ENDDO

  IF (RSMOOTH==0) THEN
     SH%N=1

     ALLOCATE(SH%WEIGHT(SH%N))
     ALLOCATE(SH%X1(SH%N), SH%X2(SH%N), SH%X3(SH%N))

     SH%N=1
     SH%X1=0
     SH%X2=0
     SH%X3=0
     SH%WEIGHT=1
  ELSE
     N1= RSMOOTH*LATT_CUR%BNORM(1)*IREFINE*GRID%NGX
     N2= RSMOOTH*LATT_CUR%BNORM(2)*IREFINE*GRID%NGY
     N3= RSMOOTH*LATT_CUR%BNORM(3)*IREFINE*GRID%NGZ

     SH%N=0

     DO I1=-N1,N1
        F1=REAL(I1,q)/GRID%NGX/IREFINE
        DO I2=-N2,N2
           F2=REAL(I2,q)/GRID%NGY/IREFINE
           DO I3=-N3,N3
              F3=REAL(I3,q)/GRID%NGZ/IREFINE
              X= F1*LATT_CUR%A(1,1)+F2*LATT_CUR%A(1,2)+F3*LATT_CUR%A(1,3)
              Y= F1*LATT_CUR%A(2,1)+F2*LATT_CUR%A(2,2)+F3*LATT_CUR%A(2,3)
              Z= F1*LATT_CUR%A(3,1)+F2*LATT_CUR%A(3,2)+F3*LATT_CUR%A(3,3)
              
              D=SQRT(X*X+Y*Y+Z*Z)
              IF (D<RSMOOTH) THEN
                 SH%N=SH%N+1
                 IF( (SH%N) > NMAX) THEN
                    WRITE(0,*) 'internal error in RSPHER_SMOOTH: increase NMAX' 
                    STOP
                 ENDIF
                 X1(SH%N)=F1
                 X2(SH%N)=F2
                 X3(SH%N)=F3
!                 WEIGHT(SH%N)= (COS(D/RSMOOTH*PI)+1)/2
                 WEIGHT(SH%N)= EXP(-4*(D/RSMOOTH)**2)
!                 WRITE(*,'(5F14.7)') D,F1,F2,F3

              ENDIF
           ENDDO
        ENDDO
     ENDDO
     ALLOCATE(SH%WEIGHT(SH%N))
     ALLOCATE(SH%X1(SH%N), SH%X2(SH%N), SH%X3(SH%N))

     SH%X1=X1(1:SH%N)
     SH%X2=X2(1:SH%N)
     SH%X3=X3(1:SH%N)
     SH%WEIGHT=WEIGHT(1:SH%N)/SUM(WEIGHT(1:SH%N))
     DO I1=1,SH%N
        X= SH%X1(I1)*LATT_CUR%A(1,1)+SH%X2(I1)*LATT_CUR%A(1,2)+SH%X3(I1)*LATT_CUR%A(1,3)
        Y= SH%X1(I1)*LATT_CUR%A(2,1)+SH%X2(I1)*LATT_CUR%A(2,2)+SH%X3(I1)*LATT_CUR%A(2,3)
        Z= SH%X1(I1)*LATT_CUR%A(3,1)+SH%X2(I1)*LATT_CUR%A(3,2)+SH%X3(I1)*LATT_CUR%A(3,3)
        D=SQRT(X*X+Y*Y+Z*Z)
!        WRITE(*,'(5F14.7)') D,WEIGHT(I1),SH%X1(I1),SH%X2(I1),SH%X3(I1)
     ENDDO
     WRITE(*,*) 'grid is refined by ',IREFINE,SH%N
  END IF

END SUBROUTINE RSPHER_SMOOTH

SUBROUTINE RSPHER_SMOOTH_DEALLOCATE( SH )

  TYPE (smoothing_handle) :: SH

  DEALLOCATE(SH%WEIGHT)
  DEALLOCATE(SH%X1, SH%X2, SH%X3)
  

END SUBROUTINE RSPHER_SMOOTH_DEALLOCATE


!****************** subroutine PHASER  *********************************
!
! subroutine PHASER
! recalculates the phase factor for the real-space projectors
!
!***********************************************************************

  SUBROUTINE PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
    USE lattice
    USE mpimy
    USE constant
    USE pseudo
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES

    NONLR_S%NK=NK

#ifdef gammareal
    RETURN
#else
!-----------------------------------------------------------------------
! k-point in kartesian coordiantes
!-----------------------------------------------------------------------
    VKX= WDES%VKPT(1,NK)*LATT_CUR%B(1,1)+WDES%VKPT(2,NK)*LATT_CUR%B(1,2)+WDES%VKPT(3,NK)*LATT_CUR%B(1,3)
    VKY= WDES%VKPT(1,NK)*LATT_CUR%B(2,1)+WDES%VKPT(2,NK)*LATT_CUR%B(2,2)+WDES%VKPT(3,NK)*LATT_CUR%B(2,3)
    VKZ= WDES%VKPT(1,NK)*LATT_CUR%B(3,1)+WDES%VKPT(2,NK)*LATT_CUR%B(3,2)+WDES%VKPT(3,NK)*LATT_CUR%B(3,3)

!-----------------------------------------------------------------------
! spin spiral propagation vector in cartesian coordinates
! is simply zero when LSPIRAL=.FALSE.
!-----------------------------------------------------------------------
    QX= (WDES%QSPIRAL(1)*LATT_CUR%B(1,1)+WDES%QSPIRAL(2)*LATT_CUR%B(1,2)+WDES%QSPIRAL(3)*LATT_CUR%B(1,3))/2
    QY= (WDES%QSPIRAL(1)*LATT_CUR%B(2,1)+WDES%QSPIRAL(2)*LATT_CUR%B(2,2)+WDES%QSPIRAL(3)*LATT_CUR%B(2,3))/2
    QZ= (WDES%QSPIRAL(1)*LATT_CUR%B(3,1)+WDES%QSPIRAL(2)*LATT_CUR%B(3,2)+WDES%QSPIRAL(3)*LATT_CUR%B(3,3))/2

!=======================================================================
! Loop over NSPINORS: here only in case of spin spirals NRSPINOR=2
!=======================================================================
    IF (NONLR_S%LSPIRAL) THEN 
       NSPINORS=2
    ELSE
       NSPINORS=1
    ENDIF

    spinor: DO ISPINOR=1,NSPINORS
!=======================================================================
! loop over all ions
!=======================================================================
       NIS=1

!OCL SCALAR
       type: DO NT=1,NONLR_S%NTYP
          IF (NONLR_S%LMMAX(NT)==0) GOTO 600
          ions: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1

             IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) CYCLE

             IF (ASSOCIATED(NONLR_S%VKPT_SHIFT)) THEN
                VKX= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(1,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(1,2)+ & 
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(1,3)
                VKY= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(2,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(2,2)+ & 
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(2,3)
                VKZ= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(3,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(3,2)+ &
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(3,3)
             ENDIF
!-----------------------------------------------------------------------
! check some quantities
!-----------------------------------------------------------------------
             ARGSC=NPSRNL/NONLR_S%PSRMAX(NT)

             F1=1._q/GRID%NGX
             F2=1._q/GRID%NGY
             F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
!sh
!            D1= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(1)*GRID%NGX
!            D2= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(2)*GRID%NGY
!            D3= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(3)*GRID%NGZ
             D1= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(1)*GRID%NGX
             D2= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(2)*GRID%NGY
             D3= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(3)*GRID%NGZ

             N3LOW= INT(NONLR_S%POSION(3,NI)*GRID%NGZ-D3+GRID%NGZ+.99_q)-GRID%NGZ
             N2LOW= INT(NONLR_S%POSION(2,NI)*GRID%NGY-D2+GRID%NGY+.99_q)-GRID%NGY
             N1LOW= INT(NONLR_S%POSION(1,NI)*GRID%NGX-D1+GRID%NGX+.99_q)-GRID%NGX

             N3HI = INT(NONLR_S%POSION(3,NI)*GRID%NGZ+D3)
             N2HI = INT(NONLR_S%POSION(2,NI)*GRID%NGY+D2)
             N1HI = INT(NONLR_S%POSION(1,NI)*GRID%NGX+D1)
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
             IF (GRID%RL%NFAST==3) THEN
             IND=1

             DO N2=N2LOW,N2HI
                X2=(N2*F2-NONLR_S%POSION(2,NI))
                N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                DO N1=N1LOW,N1HI
                   X1=(N1*F1-NONLR_S%POSION(1,NI))
                   N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

                   NCOL=GRID%RL%INDEX(N1P,N2P)
                   IF (NCOL==0) CYCLE
                   IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
                      WRITE(*,*)'internal ERROR PHASER:', &
                           GRID%RL%I2(NCOL),N1P+1, GRID%RL%I3(NCOL),N2P+1
                      STOP
                   ENDIF

                   DO N3=N3LOW,N3HI
                      X3=(N3*F3-NONLR_S%POSION(3,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X*(VKX-QX)+Y*(VKY-QY)+Z*(VKZ-QZ)))
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ELSE
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
             IND=1
             DO N3=N3LOW,N3HI
                X3=(N3*F3-NONLR_S%POSION(3,NI))
                N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

                DO N2=N2LOW,N2HI
                   X2=(N2*F2-NONLR_S%POSION(2,NI))
                   N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                   DO N1=N1LOW,N1HI
                      X1=(N1*F1-NONLR_S%POSION(1,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X*(VKX-QX)+Y*(VKY-QY)+Z*(VKZ-QZ)))
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ENDIF
!=======================================================================
! end of loop over ions
!=======================================================================
          ENDDO ions
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO type

       ! conjugate phase alteration for spin down: -q/2 -> q/2
       QX=-QX
       QY=-QY
       QZ=-QZ
    ENDDO spinor
#endif
    RETURN
  END SUBROUTINE PHASER

!****************** subroutine PHASERR  ********************************
!
! recalculates the phase factor e^(ik r-R_i) times x,y,z
! the cartesian direction, selecting x, y or z,  is supplied by an 
! index IDIR
!
!***********************************************************************

  SUBROUTINE PHASERR(GRID,LATT_CUR,NONLR_S,NK,WDES,IDIR)
    USE lattice
    USE mpimy
    USE constant
    USE pseudo
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES
    INTEGER IDIR

    REAL (q) XX(3)

    NONLR_S%NK=NK

    IF (.NOT. ASSOCIATED(NONLR_S%CRREXP)) THEN
       WRITE(0,*) 'internal error in PHASERR: CRREXP is not allocated'
       STOP
    ENDIF
!-----------------------------------------------------------------------
! k-point in kartesian coordiantes
!-----------------------------------------------------------------------
    VKX= WDES%VKPT(1,NK)*LATT_CUR%B(1,1)+WDES%VKPT(2,NK)*LATT_CUR%B(1,2)+WDES%VKPT(3,NK)*LATT_CUR%B(1,3)
    VKY= WDES%VKPT(1,NK)*LATT_CUR%B(2,1)+WDES%VKPT(2,NK)*LATT_CUR%B(2,2)+WDES%VKPT(3,NK)*LATT_CUR%B(2,3)
    VKZ= WDES%VKPT(1,NK)*LATT_CUR%B(3,1)+WDES%VKPT(2,NK)*LATT_CUR%B(3,2)+WDES%VKPT(3,NK)*LATT_CUR%B(3,3)

!-----------------------------------------------------------------------
! spin spiral propagation vector in cartesian coordinates
! is simply zero when LSPIRAL=.FALSE.
!-----------------------------------------------------------------------
    QX= (WDES%QSPIRAL(1)*LATT_CUR%B(1,1)+WDES%QSPIRAL(2)*LATT_CUR%B(1,2)+WDES%QSPIRAL(3)*LATT_CUR%B(1,3))/2
    QY= (WDES%QSPIRAL(1)*LATT_CUR%B(2,1)+WDES%QSPIRAL(2)*LATT_CUR%B(2,2)+WDES%QSPIRAL(3)*LATT_CUR%B(2,3))/2
    QZ= (WDES%QSPIRAL(1)*LATT_CUR%B(3,1)+WDES%QSPIRAL(2)*LATT_CUR%B(3,2)+WDES%QSPIRAL(3)*LATT_CUR%B(3,3))/2

!=======================================================================
! Loop over NSPINORS: here only in case of spin spirals NRSPINOR=2
!=======================================================================
    IF (NONLR_S%LSPIRAL) THEN 
       NSPINORS=2
    ELSE
       NSPINORS=1
    ENDIF

    spinor: DO ISPINOR=1,NSPINORS
       NIS=1

!OCL SCALAR
       type: DO NT=1,NONLR_S%NTYP
          IF (NONLR_S%LMMAX(NT)==0) GOTO 600
          ions: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1

             IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) CYCLE
             ARGSC=NPSRNL/NONLR_S%PSRMAX(NT)

             IF (ASSOCIATED(NONLR_S%VKPT_SHIFT)) THEN
                VKX= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(1,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(1,2)+ & 
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(1,3)
                VKY= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(2,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(2,2)+ & 
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(2,3)
                VKZ= (WDES%VKPT(1,NK)+NONLR_S%VKPT_SHIFT(1,NI))*LATT_CUR%B(3,1)+ & 
                     (WDES%VKPT(2,NK)+NONLR_S%VKPT_SHIFT(2,NI))*LATT_CUR%B(3,2)+ &
                     (WDES%VKPT(3,NK)+NONLR_S%VKPT_SHIFT(3,NI))*LATT_CUR%B(3,3)
             ENDIF
!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
             F1=1._q/GRID%NGX
             F2=1._q/GRID%NGY
             F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
!sh
!            D1= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(1)*GRID%NGX
!            D2= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(2)*GRID%NGY
!            D3= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(3)*GRID%NGZ
             D1= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(1)*GRID%NGX
             D2= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(2)*GRID%NGY
             D3= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(3)*GRID%NGZ

             N3LOW= INT(NONLR_S%POSION(3,NI)*GRID%NGZ-D3+GRID%NGZ+.99_q)-GRID%NGZ
             N2LOW= INT(NONLR_S%POSION(2,NI)*GRID%NGY-D2+GRID%NGY+.99_q)-GRID%NGY
             N1LOW= INT(NONLR_S%POSION(1,NI)*GRID%NGX-D1+GRID%NGX+.99_q)-GRID%NGX

             N3HI = INT(NONLR_S%POSION(3,NI)*GRID%NGZ+D3)
             N2HI = INT(NONLR_S%POSION(2,NI)*GRID%NGY+D2)
             N1HI = INT(NONLR_S%POSION(1,NI)*GRID%NGX+D1)

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
             IF (GRID%RL%NFAST==3) THEN
             IND=1

             DO N2=N2LOW,N2HI
                X2=(N2*F2-NONLR_S%POSION(2,NI))
                N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                DO N1=N1LOW,N1HI
                   X1=(N1*F1-NONLR_S%POSION(1,NI))
                   N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

                   NCOL=GRID%RL%INDEX(N1P,N2P)
                   IF (NCOL==0) CYCLE
                   IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
                      WRITE(*,*)'internal ERROR PHASER:', &
                           GRID%RL%I2(NCOL),N1P+1, GRID%RL%I3(NCOL),N2P+1
                      STOP
                   ENDIF

                   DO N3=N3LOW,N3HI
                      X3=(N3*F3-NONLR_S%POSION(3,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)
                      XX(1)=X
                      XX(2)=Y
                      XX(3)=Z

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X*(VKX-QX)+Y*(VKY-QY)+Z*(VKZ-QZ)))*XX(IDIR)
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ELSE
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
             IND=1
             DO N3=N3LOW,N3HI
                X3=(N3*F3-NONLR_S%POSION(3,NI))
                N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

                DO N2=N2LOW,N2HI
                   X2=(N2*F2-NONLR_S%POSION(2,NI))
                   N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                   DO N1=N1LOW,N1HI
                      X1=(N1*F1-NONLR_S%POSION(1,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)
                      XX(1)=X
                      XX(2)=Y
                      XX(3)=Z

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X*(VKX-QX)+Y*(VKY-QY)+Z*(VKZ-QZ)))*XX(IDIR)
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ENDIF
!=======================================================================
! end of loop over ions
!=======================================================================
          ENDDO ions
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO type

       ! conjugate phase alteration for spin down: -q/2 -> q/2
       QX=-QX
       QY=-QY
       QZ=-QZ
    ENDDO spinor

    RETURN
  END SUBROUTINE PHASERR

!****************** subroutine PHASER_HF  ******************************
!
! subroutine PHASER_HF
! recalculates the phase factor for the real-space projectors
! for this version the k-point coordinate is explicitly supplied
!
!***********************************************************************

  SUBROUTINE PHASER_HF(GRID,LATT_CUR,NONLR_S,VK)
    USE lattice
    USE mpimy
    USE constant
    USE pseudo
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    REAL(q)            VK(3)

#ifdef gammareal
    RETURN
#else
!=======================================================================
! Loop over NSPINORS: here only in case of spin spirals NRSPINOR=2
!=======================================================================
    IF (NONLR_S%LSPIRAL) THEN 
       NSPINORS=2
    ELSE
       NSPINORS=1
    ENDIF

    spinor: DO ISPINOR=1,NSPINORS

!=======================================================================
! loop over all ions
!=======================================================================
       NIS=1

!OCL SCALAR
       type: DO NT=1,NONLR_S%NTYP
          IF (NONLR_S%LMMAX(NT)==0) GOTO 600
          ions: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1

             IF (NONLR_S%SELECTED_ION>0 .AND. NONLR_S%SELECTED_ION/=NI) CYCLE
             ARGSC=NPSRNL/NONLR_S%PSRMAX(NT)

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
             F1=1._q/GRID%NGX
             F2=1._q/GRID%NGY
             F3=1._q/GRID%NGZ
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
!sh
!            D1= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(1)*GRID%NGX
!            D2= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(2)*GRID%NGY
!            D3= NONLR_S%PSRMAX(NT)*LATT_CUR%BNORM(3)*GRID%NGZ
             D1= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(1)*GRID%NGX
             D2= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(2)*GRID%NGY
             D3= (NONLR_S%PSRMAX(NT)+NONLR_S%RSMOOTH(NT))*LATT_CUR%BNORM(3)*GRID%NGZ

             N3LOW= INT(NONLR_S%POSION(3,NI)*GRID%NGZ-D3+GRID%NGZ+.99_q)-GRID%NGZ
             N2LOW= INT(NONLR_S%POSION(2,NI)*GRID%NGY-D2+GRID%NGY+.99_q)-GRID%NGY
             N1LOW= INT(NONLR_S%POSION(1,NI)*GRID%NGX-D1+GRID%NGX+.99_q)-GRID%NGX

             N3HI = INT(NONLR_S%POSION(3,NI)*GRID%NGZ+D3)
             N2HI = INT(NONLR_S%POSION(2,NI)*GRID%NGY+D2)
             N1HI = INT(NONLR_S%POSION(1,NI)*GRID%NGX+D1)

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
             IF (GRID%RL%NFAST==3) THEN
             IND=1

             DO N2=N2LOW,N2HI
                X2=(N2*F2-NONLR_S%POSION(2,NI))
                N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                DO N1=N1LOW,N1HI
                   X1=(N1*F1-NONLR_S%POSION(1,NI))
                   N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

                   NCOL=GRID%RL%INDEX(N1P,N2P)
                   IF (NCOL==0) CYCLE
                   IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
                      WRITE(*,*)'internal ERROR PHASER:', &
                           GRID%RL%I2(NCOL),N1P+1, GRID%RL%I3(NCOL),N2P+1
                      STOP
                   ENDIF

                   DO N3=N3LOW,N3HI
                      X3=(N3*F3-NONLR_S%POSION(3,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X1*VK(1)+X2*VK(2)+X3*VK(3)))
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ELSE
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
             IND=1
             DO N3=N3LOW,N3HI
                X3=(N3*F3-NONLR_S%POSION(3,NI))
                N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

                DO N2=N2LOW,N2HI
                   X2=(N2*F2-NONLR_S%POSION(2,NI))
                   N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

                   DO N1=N1LOW,N1HI
                      X1=(N1*F1-NONLR_S%POSION(1,NI))

                      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
                      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
                      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

                      D=SQRT(X*X+Y*Y+Z*Z)
                      ARG=(D*ARGSC)+1
                      NADDR=INT(ARG)

!sh                   IF (NADDR<NPSRNL) THEN
                      IF (D<(NPSRNL-1)/ARGSC+NONLR_S%RSMOOTH(NT)) THEN
                         NONLR_S%CRREXP(IND,NI,ISPINOR)=EXP(CITPI*(X1*VK(1)+X2*VK(2)+X3*VK(3)))
                         IND=IND+1
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             ENDIF
!=======================================================================
! end of loop over ions
!=======================================================================
          ENDDO ions
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO type

    ENDDO spinor
#endif
    RETURN
  END SUBROUTINE PHASER_HF

!***********************************************************************
!
! generate a non local projector  for a single ion
! reuse the data structure of the full projector whereever that
! is possible
! NOTE: deallocation must go through  NONLR_DEALLOC_SINGLE_ION
! since otherwise the original data structure is destroyed
! the first projector contains the conventional projector, whereas the second
! one is the derivative of the projector
!
!***********************************************************************

  SUBROUTINE  NONLR_SET_SINGLE_ION(GRID,LATT_CUR, NONLR_S, NONLR_ION, NONLR_IOND, ION, IDIR)
    USE lattice
    IMPLICIT NONE

    TYPE (grid_3d)      GRID
    TYPE (latt)         LATT_CUR
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonlr_struct) NONLR_ION  ! descriptor for single ion projector
    TYPE (nonlr_struct) NONLR_IOND ! derivative of projector
    LOGICAL LREALLOCATE
    INTEGER ION, IDIR
    REAL(q)  DISPL(3,NONLR_S%NIONS)
    REAL(q), PARAMETER :: DIS=fd_displacement
    REAL(q) faker

    NONLR_ION%LREAL=.FALSE.
    NONLR_IOND%LREAL=.FALSE.
    IF (.NOT. NONLR_S%LREAL) RETURN

    ! copy entire descriptor
    NONLR_ION=NONLR_S
    ! select one specific ion
    NONLR_ION%SELECTED_ION=ION
    ! determine IRALLOC
    NONLR_ION%IRALLOC=-1
    CALL REAL_OPTLAY(GRID,LATT_CUR,NONLR_ION,.TRUE.,LREALLOCATE,-1,-1)

    ! allocate required arrays
    ALLOCATE(NONLR_ION%NLIMAX(NONLR_ION%NIONS), &
         NONLR_ION%RPROJ (NONLR_ION%IRALLOC))

     ! TODO: add back if hamil_lr.F ported to GPU..
!    ALLOCATE(NONLR_ION%NLIMAX(NONLR_ION%NIONS))
!#if USE_PINNED_MEMORY
!    call nvpinnedmalloc(NONLR_ION%RPROJ_PTR,NONLR_ION%IRALLOC*int(c_sizeof(faker),c_size_t))
!    call c_f_pointer(NONLR_ION%RPROJ_PTR,NONLR_ION%RPROJ,(/NONLR_ION%IRALLOC/))
!#else
!    ALLOCATE(NONLR_ION%RPROJ(NONLR_ION%IRALLOC))
!#endif

    ! copy entire descriptor
    NONLR_IOND=NONLR_S
    ! select one specific ion
    NONLR_IOND%SELECTED_ION=ION
    ! determine IRALLOC
    NONLR_IOND%IRALLOC=-1
    CALL REAL_OPTLAY(GRID,LATT_CUR,NONLR_IOND,.TRUE.,LREALLOCATE,-1,-1)

    ! allocate required arrays
    ALLOCATE(NONLR_IOND%NLIMAX(NONLR_IOND%NIONS), &
         NONLR_IOND%RPROJ (NONLR_IOND%IRALLOC))

     ! TODO: add back if hamil_lr.F ported to GPU..
!    ALLOCATE(NONLR_IOND%NLIMAX(NONLR_IOND%NIONS))
!#if USE_PINNED_MEMORY
!    call nvpinnedmalloc(NONLR_IOND%RPROJ_PTR,NONLR_IOND%IRALLOC*int(c_sizeof(faker),c_size_t))
!    call
!    c_f_pointer(NONLR_IOND%RPROJ_PTR,NONLR_IOND%RPROJ,(/NONLR_IOND%IRALLOC/))
!#else
!    ALLOCATE(NONLR_IOND%RPROJ(NONLR_IOND%IRALLOC))
!#endif

    ! setup single ion projector
    DISPL=0
    CALL RSPHER_ALL(GRID,NONLR_ION,LATT_CUR,LATT_CUR,LATT_CUR, DISPL,DISPL,0)

    ! calculate first derivative of projector
    DISPL(IDIR,:)=DIS
    NONLR_IOND%RPROJ=0
    CALL RSPHER_ALL(GRID,NONLR_IOND,LATT_CUR,LATT_CUR,LATT_CUR, -DISPL,DISPL,1)
    NONLR_IOND%RPROJ= NONLR_IOND%RPROJ*(0.5_q/DIS)

  END SUBROUTINE NONLR_SET_SINGLE_ION


  SUBROUTINE  NONLR_DEALLOC_SINGLE_ION(NONLR_S)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S

    IF (NONLR_S%LREAL) THEN
       ! TODO: add back if hamil_lr.F ported to GPU..
!      #if USE_PINNED_MEMORY
!        call nvpinnedfree(NONLR_S%RPROJ_PTR)
!      #else
!        DEALLOCATE(NONLR_S%RPROJ)
!      #endif
      
      DEALLOCATE(NONLR_S%RPROJ)
      DEALLOCATE(NONLR_S%NLIMAX)     
    ENDIF

    NULLIFY(NONLR_S%NLIMAX, NONLR_S%NLI, NONLR_S%RPROJ,NONLR_S%CRREXP)
    RETURN
  END SUBROUTINE NONLR_DEALLOC_SINGLE_ION

!****************** subroutine NONLR_ALLOC_CRREXP **********************
!
! allocate the CRREXP array and deallocate it
!
!***********************************************************************

  SUBROUTINE  NONLR_ALLOC_CRREXP(NONLR_S)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    INTEGER NIONS,IRMAX

    NIONS = NONLR_S%NIONS
    IRMAX = NONLR_S%IRMAX

    NULLIFY(NONLR_S%CRREXP)
    IF (.NOT.NONLR_S%LSPIRAL ) THEN
       ALLOCATE(NONLR_S%CRREXP(IRMAX,NIONS,1))
    ELSE
       ALLOCATE(NONLR_S%CRREXP(IRMAX,NIONS,2))
    ENDIF
    RETURN
  END SUBROUTINE NONLR_ALLOC_CRREXP

  SUBROUTINE  NONLR_DEALLOC_CRREXP(NONLR_S)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S

    IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
       DEALLOCATE(NONLR_S%CRREXP)
    ENDIF
    RETURN
  END SUBROUTINE NONLR_DEALLOC_CRREXP



!****************** subroutine RPRO1    ******************************
!
! this subroutine calculates the scalar product of one wavefunction with
! all projector functions in real space
! thesis gK Equ. (10.36)
!
!*********************************************************************


  SUBROUTINE RPRO1(NONLR_S, WDES1, W1)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1

  ! local
    INTEGER :: IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, &
         NI, INDMAX, IND, LM
    REAL(q) :: RP, SUMR, SUMI
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
    COMPLEX(q) CTMP
#ifdef avoidalloc
    REAL(qn) :: WORK(NONLR_S%IRMAX*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE   :: CPROJ(:)

    ALLOCATE(WORK(NONLR_S%IRMAX*2),TMP(101,2),CPROJ(WDES1%NPRO_TOT))
#endif

    IF (WDES1%NK /= NONLR_S%NK) THEN
       WRITE(*,*) 'internal error in RPRO1: PHASE not properly set up',WDES1%NK, NONLR_S%NK
       STOP
    ENDIF

    CPROJ=0
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
!=======================================================================
!  extract the relevant points for this ion
!=======================================================================
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
             IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
#ifdef gammareal
                   WORK(IND)      = REAL( W1%CR(IP) ,KIND=qn)
#else
                   CTMP=  W1%CR(IP)*NONLR_S%CRREXP(IND,NI,ISPIRAL)
                   WORK(IND)      = REAL( CTMP ,KIND=qn)
                   WORK(IND+NONLR_S%IRMAX)=AIMAG(CTMP)
#endif
                ENDDO
#else
                CALL CRREXP_MUL_WAVE( INDMAX, NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                           W1%CR(ISPINOR*WDES1%GRID%MPLWV+1),WORK(1),WORK(NONLR_S%IRMAX+1))
#endif
             ELSE
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   WORK(IND)      = REAL( W1%CR(IP) ,KIND=qn)
                ENDDO
             ENDIF
!=======================================================================
! loop over composite indexes L,M
!=======================================================================
#if defined(do_loops) || defined(pro_loop)
             DO LM=1,LMMAXC
                SUMR=0
                SUMI=0
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   RP   =  NONLR_S%RPROJ(IND+ (LM-1)*INDMAX+NLIIND)
                   SUMR = SUMR + WORK(IND)       * RP
#ifndef gammareal
                   SUMI = SUMI + WORK(IND+NONLR_S%IRMAX) * RP
#endif
                ENDDO
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
             ENDDO
#else
             CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, WORK(1) , 1 , ZERO ,  TMP(1,1), 1)
#ifndef gammareal
             CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, WORK(1+NONLR_S%IRMAX) , 1 , ZERO ,  TMP(1,2), 1)
#endif

             l_loop: DO LM=1,LMMAXC
#ifdef gammareal
                CPROJ(LM+LMBASE)=TMP(LM,1)*WDES1%RINPL
#else
                SUMR=TMP(LM,1)
                SUMI=TMP(LM,2)
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
#endif
             ENDDO l_loop
#endif

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion

600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor
    ! distribute the projected wavefunctions to nodes
    CALL DIS_PROJ(WDES1,CPROJ(1),W1%CPROJ(1))

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RPRO1

!****************** subroutine RPRO1_HF    ***************************
!
! this subroutine calculates the scalar product of one wavefunction with
! all projector functions in real space
! thesis gK Equ. (10.36)
! cannot use RPRO1 because of possible type real of GCR
!
!*********************************************************************

  SUBROUTINE RPRO1_HF(NONLR_S,WDES1,W1,GCR)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1
    GDEF :: GCR(*)

    INTEGER :: IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, &
         NI, INDMAX, IND, LM
    REAL(q) :: RP, SUMR, SUMI
    COMPLEX(q) CTMP 
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
#ifdef avoidalloc
    REAL(qn) :: WORK(NONLR_S%IRMAX*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE   :: CPROJ(:)

    ALLOCATE(WORK(NONLR_S%IRMAX*2),TMP(101,2),CPROJ(WDES1%NPRO_TOT))
#endif

    CPROJ=0
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
!=======================================================================
!  extract the relevant points for this ion
!=======================================================================
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
             IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
                CALL CRREXP_MUL_GWAVE( INDMAX, NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                           GCR(ISPINOR*WDES1%GRID%MPLWV+1),WORK(1),WORK(NONLR_S%IRMAX+1))
             ELSE
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   WORK(IND)      = REAL( GCR(IP) ,KIND=qn)
                ENDDO
             ENDIF
!=======================================================================
! loop over composite indexes L,M
!=======================================================================
#if defined(do_loops) || defined(pro_loop)
             DO LM=1,LMMAXC
                SUMR=0
                SUMI=0
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   RP   =  NONLR_S%RPROJ(IND+ (LM-1)*INDMAX+NLIIND)
                   SUMR = SUMR + WORK(IND)       * RP
#ifndef gammareal
                   SUMI = SUMI + WORK(IND+NONLR_S%IRMAX) * RP
#endif
                ENDDO
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
             ENDDO
#else
             CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, WORK(1) , 1 , ZERO ,  TMP(1,1), 1)
#ifndef gammareal
             CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, WORK(1+NONLR_S%IRMAX) , 1 , ZERO ,  TMP(1,2), 1)
#endif

             l_loop: DO LM=1,LMMAXC
#ifdef gammareal
                CPROJ(LM+LMBASE)=TMP(LM,1)*WDES1%RINPL
#else
                SUMR=TMP(LM,1)
                SUMI=TMP(LM,2)
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
#endif
             ENDDO l_loop
#endif

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion

600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor
    ! distribute the projected wavefunctions to nodes
    CALL DIS_PROJ(WDES1,CPROJ(1),W1%CPROJ(1))

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RPRO1_HF

!****************** subroutine RPROMU   ******************************
!
!  this subroutine  calculates the projection of a set of
!  bands onto the
!  real space projection operators
!
!*********************************************************************


  SUBROUTINE RPROMU(NONLR_S, WDES1, W1, NSIM, LDO)
    IMPLICIT NONE

    INTEGER NSIM
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(NSIM)
    LOGICAL            LDO(NSIM)
  ! local
    INTEGER :: NP, IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, &
         NI, INDMAX, IND, IND0, NPFILL, LM
    REAL(q) :: SUMR, SUMI
    COMPLEX(q) CTMP 
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
    INTEGER, PARAMETER :: NLM=101
#ifdef avoidalloc
    REAL(qn) :: WORK(ndata*NONLR_S%IRMAX*NSIM),TMP(NLM, ndata*NSIM)
    GDEF    :: CPROJ(WDES1%NPRO_TOT,NSIM)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE   :: CPROJ(:,:)

    ALLOCATE(WORK(ndata*NONLR_S%IRMAX*NSIM),TMP(NLM, ndata*NSIM), &
         CPROJ(WDES1%NPRO_TOT,NSIM))
#endif
    IF (WDES1%NK /= NONLR_S%NK) THEN
       WRITE(*,*) 'internal error in RPROMU: PHASE not properly set up',WDES1%NK, NONLR_S%NK
       STOP
    ENDIF

    CPROJ=0
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
!=======================================================================
!  extract the relevant points for this ion
!=======================================================================
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
             IND0=0
             NPFILL=0

             DO NP=1,NSIM

                IF (LDO(NP)) THEN
                   IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         CTMP=  W1(NP)%CR(IP)*NONLR_S%CRREXP(IND,NI,ISPIRAL)
                         WORK(IND+IND0) = REAL( CTMP ,KIND=q)
                         WORK(IND+(NONLR_S%IRMAX+IND0))=AIMAG(CTMP)
                      ENDDO
#else
                      CALL CRREXP_MUL_WAVE( INDMAX, NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                           W1(NP)%CR(ISPINOR*WDES1%GRID%MPLWV+1),WORK(IND0+1),WORK(IND0+(NONLR_S%IRMAX+1)))
#endif
                   ELSE
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         WORK(IND+IND0) = REAL( W1(NP)%CR(IP) ,KIND=qn)
                      ENDDO
                   ENDIF

                   IND0=IND0+ndata * NONLR_S%IRMAX
                   NPFILL=NPFILL+1
                ENDIF

             ENDDO

!=======================================================================
! loop over composite indexes L,M
!=======================================================================
#ifdef RPROMU_DGEMV
             DO NP=0,ndata*NPFILL-1
                CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                     INDMAX, WORK(1+NONLR_S%IRMAX*NP) , 1 , ZERO ,  TMP(1,NP+1), 1)
             ENDDO
#else
             CALL DGEMMn( 'T', 'N' , LMMAXC,  ndata*NPFILL, INDMAX, ONE, &
                  NONLR_S%RPROJ(1+NLIIND), INDMAX, WORK(1), NONLR_S%IRMAX, &
                  ZERO,  TMP(1,1), NLM )
#endif
             IND0=0
             DO NP=1,NSIM
                IF (LDO(NP)) THEN
                   l_loop: DO LM=1,LMMAXC
#ifdef gammareal
                      CPROJ(LM+LMBASE,NP)=TMP(LM,1+IND0)*WDES1%RINPL
#else
                      SUMR=TMP(LM,1+IND0)
                      SUMI=TMP(LM,2+IND0)
                      CPROJ(LM+LMBASE,NP)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
#endif
                   ENDDO l_loop
                   IND0=IND0+ndata
                ENDIF
             ENDDO

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion


600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ

       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

    ! distribute the projected wavefunctions to nodes
    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL DIS_PROJ(WDES1,CPROJ(1,NP),W1(NP)%CPROJ(1))
       ENDIF
    ENDDO

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RPROMU

!****************** subroutine RPROMU_HF *****************************
!
! essentially a copy of the previous routine 
! -  with W1(NP)%CR(i) -> GCR(i,NP)
! -  phase factor check removed
! -  LDO array removed
!
!*********************************************************************


  SUBROUTINE RPROMU_HF(NONLR_S, WDES1, W1, NSIM, GCR, LD)
    IMPLICIT NONE

    INTEGER NSIM
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(NSIM)
    INTEGER :: LD  ! leading dimension of GCR
    GDEF :: GCR(LD, NSIM)
  ! local
    INTEGER :: NP, IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, &
         NI, INDMAX, IND, IND0, NPFILL, LM
    REAL(q) :: SUMR, SUMI
    COMPLEX(q) CTMP 
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
    INTEGER, PARAMETER :: NLM=101
#ifdef avoidalloc
    REAL(qn) :: WORK(ndata*NONLR_S%IRMAX*NSIM),TMP(NLM, ndata*NSIM)
    GDEF    :: CPROJ(WDES1%NPRO_TOT,NSIM)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE   :: CPROJ(:,:)

    ALLOCATE(WORK(ndata*NONLR_S%IRMAX*NSIM),TMP(NLM, ndata*NSIM), &
         CPROJ(WDES1%NPRO_TOT,NSIM))
#endif

    CPROJ=0
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
!=======================================================================
!  extract the relevant points for this ion
!=======================================================================
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
             IND0=0
             NPFILL=0

             DO NP=1,NSIM

                   IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
                      CALL CRREXP_MUL_GWAVE( INDMAX, NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                           GCR(ISPINOR*WDES1%GRID%MPLWV+1,NP),WORK(IND0+1),WORK(IND0+(NONLR_S%IRMAX+1)))
                   ELSE
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         WORK(IND+IND0) = REAL( GCR(IP, NP) ,KIND=qn)
                      ENDDO
                   ENDIF

                   IND0=IND0+ndata * NONLR_S%IRMAX
                   NPFILL=NPFILL+1

             ENDDO

!=======================================================================
! loop over composite indexes L,M
!=======================================================================
#ifdef RPROMU_DGEMV
             DO NP=0,ndata*NPFILL-1
                CALL DGEMVn( 'T' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                     INDMAX, WORK(1+NONLR_S%IRMAX*NP) , 1 , ZERO ,  TMP(1,NP+1), 1)
             ENDDO
#else
             CALL DGEMMn( 'T', 'N' , LMMAXC,  ndata*NPFILL, INDMAX, ONE, &
                  NONLR_S%RPROJ(1+NLIIND), INDMAX, WORK(1), NONLR_S%IRMAX, &
                  ZERO,  TMP(1,1), NLM )
#endif
             IND0=0
             DO NP=1,NSIM
                   l_loop: DO LM=1,LMMAXC
#ifdef gammareal
                      CPROJ(LM+LMBASE,NP)=TMP(LM,1+IND0)*WDES1%RINPL
#else
                      SUMR=TMP(LM,1+IND0)
                      SUMI=TMP(LM,2+IND0)
                      CPROJ(LM+LMBASE,NP)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *WDES1%RINPL)
#endif
                   ENDDO l_loop
                   IND0=IND0+ndata
             ENDDO

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion


600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ

       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

    ! distribute the projected wavefunctions to nodes
    DO NP=1,NSIM
          CALL DIS_PROJ(WDES1,CPROJ(1,NP),W1(NP)%CPROJ(1))
    ENDDO

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RPROMU_HF


!****************** subroutine RPROMU_GPU ****************************
!
! this subroutine is a GPU version of RPROMU, it parallelizes over
! inner dimensions and NSIM, and batches/streams over ions with
! the BATCH_SIZE parameter
!
!*********************************************************************
  SUBROUTINE RPROMU_GPU(NONLR_S, WDES1, W1, NSIM, LDO, GPU_CR, Transfert, BATCH_SIZE)
    USE iso_c_binding
    USE cuda_interface
    USE gpu_data
    IMPLICIT NONE

    INTEGER NSIM
    TYPE (nonlr_struct), TARGET :: NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(NSIM)
    LOGICAL            LDO(NSIM)
    INTEGER            BATCH_SIZE
  ! local
    INTEGER :: NP, IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, &
         NI, INDMAX, IND, IND0, NPFILL, LM, C1, C2, C3, N1, N2, J, NPNEXT, I
    INTEGER :: LMBASE_SAVE,NLIIND_SAVE,NIB,STREAM_IDX
    REAL(q) :: SUMR, SUMI, Transfert,TV0,TC0,TV,TC
    COMPLEX(q) CTMP
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
    INTEGER, PARAMETER :: NLM=101
    INTEGER(c_intptr_t)          :: GPU_CR(:)
    integer             :: LDO_IND_SIZE
!    integer(8)          :: GPU_LDO_IND
    GDEF                :: fake
    integer             :: fakei
    REAL(qn)            :: fakern
    COMPLEX(q)          :: fakec
    logical             :: FIRST_PASS
    REAL(qn), DIMENSION(ndata*NONLR_S%IRMAX*NSIM) :: GPU_TEST
    REAL(qn), DIMENSION(ndata*NONLR_S%IRMAX*NSIM) :: GPU_TEST_BLOCK
    INTEGER(c_intptr_t) :: GPU_CR1_ARRAY(NSIM)
    integer    :: copy_gpu_cr1_array
    integer    :: copy_gpu_M_array

nv_profile_start(NVP_RPROMU)
    C1=size(NONLR_S%CRREXP,1)
    C2=size(NONLR_S%CRREXP,2)
    C3=size(NONLR_S%CRREXP,3)
    N1=size(NONLR_S%NLI,1)
    N2=size(NONLR_S%NLI,2)

    IF (WDES1%NK /= NONLR_S%NK) THEN
       WRITE(*,*) 'internal error in RPROMU: PHASE not properly set up',WDES1%NK, NONLR_S%NK
       STOP
    ENDIF

    NPFILL=0
    GPU_CR1_ARRAY=0;
    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          NPFILL=NPFILL+1
          NONLR_S%LDO_INDEX(NPFILL)=NP-1
          GPU_CR1_ARRAY(NP)=GPU_CR(NP)
       END IF
    END DO
    LDO_IND_SIZE = NPFILL

    FIRST_PASS=.true.
nv_profile_start(NVP_RPROMU_MEMCPY)
    CALL cublas_Set_Vector(LDO_IND_SIZE,int(c_sizeof(fakei),c_int),c_loc(NONLR_S%LDO_INDEX),1,NONLR_S%LDO_INDEX_GPU,1)
nv_profile_stop(NVP_RPROMU_MEMCPY)

!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0
    ISPIRAL = 1
    copy_gpu_cr1_array = 1
    copy_gpu_M_array = 1

nv_profile_start(NVP_RPROMU_COMPUTE)
nv_profile_start(NVP_GEMM)
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600
          ionbloc: DO NIB=NIS,NONLR_S%NITYP(NT)+NIS-1,BATCH_SIZE
             LMBASE_SAVE = LMBASE
             IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
             ! for a batch of ions,
             call gpu_crrexp_mul_wave_st_multi( &
                0, &
                NONLR_S%NLIMAX, &
                NONLR_S%NIONS,&
                copy_gpu_M_array, &
                GPU_CR1_ARRAY, &
                copy_gpu_cr1_array,&
                ISPINOR*WDES1%GRID%MPLWV, &
                NONLR_S%WORK_GPU, &
                ndata*NSIM*NONLR_S%IRMAX,&
                NONLR_S%IRMAX, &
                ndata, &
                NSIM, &
                NONLR_S%NLI_GPU, &
                N1, &
                NONLR_S%CRREXP_GPU,&
                ISPIRAL, &
                C1, &
                C2, &
                NIB, &
                MIN(NONLR_S%NITYP(NT)+NIS-1,NIB+BATCH_SIZE-1))
             copy_gpu_cr1_array = 0
             copy_gpu_M_array = 0
             END IF
             call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPROMU_GPU!"//C_NULL_CHAR)

!=======================================================================
! loop over composite indexes L,M
!=======================================================================
             DO NI=NIB,MIN(NONLR_S%NITYP(NT)+NIS-1,NIB+BATCH_SIZE-1)
                INDMAX=NONLR_S%NLIMAX(NI)
                IF (INDMAX /= 0) THEN
                ! for each ion,
                call cuda_gemmsmallmn(NI-NIB,'T','N' ,LMMAXC,ndata*LDO_IND_SIZE,INDMAX,&
                     ONE,NONLR_S%RPROJ_GPU,NLIIND,NONLR_S%NLIMAX(NI),&
                     NONLR_S%WORK_GPU,(NI-NIB)*ndata*NSIM*NONLR_S%IRMAX,NONLR_S%IRMAX,&
                     ZERO,NONLR_S%TMP_GPU,(NI-NIB)*NLM*ndata*NSIM,NLM)
                ENDIF
                LMBASE=LMMAXC+LMBASE
                NLIIND=LMMAXC*INDMAX+NLIIND
             END DO
             call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPROMU_GPU!"//C_NULL_CHAR)

             ! for a batch of ions,
             NI=MIN(NONLR_S%NITYP(NT)+NIS-1,NIB+BATCH_SIZE-1)-NIB;
             call cuda_calccproj(LMMAXC,NI+1,NONLR_S%CPROJ_GPU,LMBASE_SAVE,WDES1%NPRO_TOT,&
                  NONLR_S%TMP_GPU,NLM*ndata*NSIM,WDES1%RINPL,NSIM,NLM,&
                  NONLR_S%LDO_INDEX_GPU,LDO_IND_SIZE,ndata)
             call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPROMU_GPU!"//C_NULL_CHAR)
          END DO ionbloc
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_RPROMU_COMPUTE)

    CALL CUBLAS_RESET_ST() ! set default stream for cublas to zero

nv_profile_start(NVP_RPROMU_MEMCPY)
#ifdef RPROMU_CPROJ_OVERLAP
    IF(LDO_IND_SIZE >= 1) THEN
       NP = NONLR_S%LDO_INDEX(1)+1 ! LDO_IND use C indexing
       call cuda_memcpydtohshift(NULL_STREAM,c_loc(NONLR_S%CPROJ(1,NP)),0,NONLR_S%CPROJ_GPU,IDX(1,NP,WDES1%NPRO_TOT),WDES1%NPRO_TOT,int(c_sizeof(fake),c_size_t))
    END IF
    DO J=1,LDO_IND_SIZE-1
       NP = NONLR_S%LDO_INDEX(J)+1
       NPNEXT = NONLR_S%LDO_INDEX(J+1)+1
       call cuda_memcpydtohshift(0,c_loc(NONLR_S%CPROJ(1,NPNEXT)),0,NONLR_S%CPROJ_GPU,IDX(1,NPNEXT,WDES1%NPRO_TOT),WDES1%NPRO_TOT,int(c_sizeof(fake),c_size_t))
       CALL DIS_PROJ(WDES1,NONLR_S%CPROJ(1,NP),W1(NP)%CPROJ(1))
       CALL CUDA_ALL_STREAM_SYNCHRONIZE () ! wait for NPNEXT transfert to finish
    END DO
    IF(LDO_IND_SIZE >= 1) THEN
       NP = NONLR_S%LDO_INDEX(LDO_IND_SIZE)+1
       CALL DIS_PROJ(WDES1,NONLR_S%CPROJ(1,NP),W1(NP)%CPROJ(1))
    END IF
#else
    call cublas_Get_Vector(WDES1%NPRO_TOT*NSIM,int(c_sizeof(fake),c_int),NONLR_S%CPROJ_GPU,1,c_loc(NONLR_S%CPROJ),1)
    ! distribute the projected wavefunctions to nodes
    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL DIS_PROJ(WDES1,NONLR_S%CPROJ(1,NP),W1(NP)%CPROJ(1))
       ENDIF
    ENDDO
#endif
nv_profile_stop(NVP_RPROMU_MEMCPY)
nv_profile_stop(NVP_RPROMU)
    RETURN
  END SUBROUTINE RPROMU_GPU

!****************** subroutine RPRO     ******************************
!
!  this subroutine  calculates the projection of all bands onto the
!  real space projection operators doing a set of
!  bands at the same time
!  
!*********************************************************************


  SUBROUTINE RPRO(NONLR_S,WDES,W,GRID,NK)
    IMPLICIT NONE
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES
    TYPE (wavedes1)    WDES1
    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID
    INTEGER NK

    CALL RPRO_ISP_GPU(NONLR_S,WDES,W,GRID,0,NK)
  END SUBROUTINE RPRO

  SUBROUTINE RPRO_ISP_GPU(NONLR_S,WDES,W,GRID,ISP_SWITCH,NK)
    USE iso_c_binding
    USE cuda_interface
    USE wave_high
    USE gpu_data
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID
    INTEGER ISP_SWITCH       ! 0 all spin components, 1 or 2 only selected
    INTEGER NK
    ! device pointers
    INTEGER(c_intptr_t) GPU_NINDPW
    INTEGER(c_intptr_t) GPU_W1_CW(WDES%NSIM),GPU_W1_CR(WDES%NSIM)
    INTEGER(c_intptr_t) GPU_W1_CW_ALL,GPU_W1_CR_ALL
    ! local variables
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)  W1(WDES%NSIM)
    INTEGER :: NSIM, NT, N, NPL, NGVECTOR, ISP_START, ISP_END, ISP, NUP, NN, NNP, ISPINOR
    INTEGER SID
    INTEGER fakei
    COMPLEX(q) fakec
    REAL(q) faker
    REAL(q) TimeTransfert
    INTEGER, SAVE :: NCALL=0

    !IF(NCALL>=3) THEN
     !call cuda_device_reset()
     !stop
    !ENDIF

nv_profile_start(NVP_RPRO)
    NSIM = WDES%NSIM
    DO NT=1,NONLR_S%NTYP
       IF (NONLR_S%LMMAX(NT)/=0) GOTO 300
    ENDDO
    ! shortcut for NC potentials
    RETURN

300 CONTINUE
    DO N=1,NSIM
       ALLOCATE(W1(N)%CR(GRID%MPLWV*WDES%NRSPINORS))
    ENDDO

    ! setup descriptor
    CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

    NPL=WDES%NPLWKP(NK)
    NGVECTOR=WDES%NGVECTOR(NK)

nv_profile_start(NVP_RPRO_MALLOC)
    ! allocate device arrays
    call cublas_Alloc_safety(NGVECTOR,int(c_sizeof(fakei),c_size_t),GPU_NINDPW)
    call cublas_Alloc_safety(WDES1%NRPLWV*NSIM,int(c_sizeof(fakec),c_size_t),GPU_W1_CW_ALL)
    call cublas_Alloc_safety(WDES1%GRID%MPLWV*WDES1%NRSPINORS*NSIM,int(c_sizeof(fakec),c_size_t),GPU_W1_CR_ALL)
nv_profile_stop(NVP_RPRO_MALLOC)

    ! set device pointers
    DO N=1,NSIM
        GPU_W1_CW(N) = GPU_W1_CW_ALL + (N-1)*WDES1%NRPLWV*int(c_sizeof(fakec),c_size_t)
        GPU_W1_CR(N) = GPU_W1_CR_ALL + (N-1)*WDES1%GRID%MPLWV*WDES1%NRSPINORS*int(c_sizeof(fakec),c_size_t)
    ENDDO

nv_profile_start(NVP_RPRO_MEMCPY)
    ! copy from host to device
    call cuda_memcpyhtod(-1,GPU_NINDPW,c_loc(WDES%NINDPW(1,NK)),NGVECTOR,int(c_sizeof(fakei),c_size_t))
    call INIT_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
    call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPRO_ISP!"//C_NULL_CHAR)
nv_profile_stop(NVP_RPRO_MEMCPY)

    IF (ISP_SWITCH==1 .OR. ISP_SWITCH==2) THEN
       ISP_START=ISP_SWITCH
       ISP_END  =ISP_SWITCH
    ELSE
       ISP_START=1
       ISP_END  =WDES%ISPIN
    ENDIF

nv_profile_start(NVP_RPRO_COMPUTE)
    DO ISP=ISP_START,ISP_END
       DO N=1,WDES%NBANDS,NSIM
          NUP=MIN(N+NSIM-1,WDES%NBANDS)
          SID=0
          DO NN=N,NUP
             NNP=NN-N+1
             CALL SETWAV(W,W1(NNP),WDES1,NN,ISP)
             ! copy W1%CW, from host to device
             call cuda_memcpyhtod(SID,GPU_W1_CW(NNP),c_loc(W1(NNP)%CW),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
             ! fft to real space
             CALL FFTWAV_W1_GPU_STREAM(SID,W1(NNP),GPU_W1_CW(NNP),GPU_W1_CR(NNP),GPU_NINDPW)
             SID=SID+1
             IF(SID>=NV_NUM_STREAMS) SID=0
             !call cuda_prints(1,NNP,"cw",2,WDES1%NRPLWV,GPU_W1_CW(NNP))
             !call cuda_prints(1,NNP,"cr",2,WDES1%GRID%MPLWV*WDES1%NRSPINORS,GPU_W1_CR(NNP))
          ENDDO
          call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPRO_ISP!"//C_NULL_CHAR)
          !call cuda_prints(NCALL,I,"cw",'c',WDES1%NRPLWV,GPU_W1_CW(1))
          !call cuda_prints(NCALL,I,"cr",'c',WDES1%GRID%MPLWV*WDES1%NRSPINORS,GPU_W1_CR)
          !IF (NSIM/=1) THEN
             CALL RPROMU_GPU(NONLR_S,WDES1,W1,NUP-N+1,W1%LDO,GPU_W1_CR,TimeTransfert,NV_NUM_BATCHES)
          !ELSE
          !   ! copy W1%CW, from device to host
          !   call cuda_memcpydtoh(-1,W1(1)%CW,GPU_W1_CW(1),WDES1%NRPLWV,int(c_sizeof(fakec),c_size_t))
          !   call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPRO_ISP!"//C_NULL_CHAR)
          !   CALL RPRO1(NONLR_S,WDES1,W1(1))
          !ENDIF
       ENDDO
    ENDDO
nv_profile_stop(NVP_RPRO_COMPUTE)

    DO N=1,NSIM
       DEALLOCATE(W1(N)%CR)
    ENDDO

nv_profile_start(NVP_RPRO_MALLOC)
    ! free device arrays
    call cublas_free(GPU_NINDPW)
    call cublas_free(GPU_W1_CW_ALL)
    call cublas_free(GPU_W1_CR_ALL)
    call DESTROY_NONLR_GPU(NONLR_S,WDES1,NSIM,NV_NUM_BATCHES)
    call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RPRO_ISP!"//C_NULL_CHAR)
nv_profile_stop(NVP_RPRO_MALLOC)
nv_profile_stop(NVP_RPRO)
    RETURN
  END SUBROUTINE RPRO_ISP_GPU

  SUBROUTINE RPRO_ISP(NONLR_S,WDES,W,GRID,ISP_SWITCH,NK)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID
    INTEGER ISP_SWITCH       ! 0 all spin components, 1 or 2 only selected
    INTEGER NK
  ! local variables
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)  W1(WDES%NSIM)
    LOGICAL ::       LDO(WDES%NSIM)
    INTEGER :: NSIM, NT, N, NPL, NGVECTOR, ISP_START, ISP_END, ISP, NUP, NN, NNP, ISPINOR

    LDO=.TRUE.
    NSIM = WDES%NSIM
    DO NT=1,NONLR_S%NTYP
       IF (NONLR_S%LMMAX(NT)/=0) GOTO 300
    ENDDO
    ! shortcut for NC potentials
    RETURN

300 CONTINUE
    DO N=1,NSIM
       ALLOCATE(W1(N)%CR(GRID%MPLWV*WDES%NRSPINORS))
    ENDDO

    ! setup descriptor
    CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

    NPL=WDES%NPLWKP(NK)
    NGVECTOR=WDES%NGVECTOR(NK)

    IF (ISP_SWITCH==1 .OR. ISP_SWITCH==2) THEN
       ISP_START=ISP_SWITCH
       ISP_END  =ISP_SWITCH
    ELSE
       ISP_START=1
       ISP_END  =WDES%ISPIN
    ENDIF

    DO ISP=ISP_START,ISP_END
       DO N=1,WDES%NBANDS,NSIM
          NUP=MIN(N+NSIM-1,WDES%NBANDS)
          DO NN=N,NUP
             NNP=NN-N+1
             CALL SETWAV(W,W1(NNP),WDES1,NN,ISP)
             DO ISPINOR=0,WDES%NRSPINORS-1
                CALL FFTWAV(NGVECTOR,WDES%NINDPW(1,NK),W1(NNP)%CR(1+ISPINOR*WDES1%GRID%MPLWV),W1(NNP)%CW(1+ISPINOR*NGVECTOR),GRID)
             ENDDO
          ENDDO
          IF (NSIM/=1) THEN
             CALL RPROMU(NONLR_S,WDES1,W1,NUP-N+1,LDO)
          ELSE
             CALL RPRO1(NONLR_S,WDES1,W1(1))
          ENDIF
       ENDDO
    ENDDO

    DO N=1,NSIM
       DEALLOCATE(W1(N)%CR)
    ENDDO

    RETURN
  END SUBROUTINE RPRO_ISP

!****************** subroutine RACCT    ******************************
!
!  this subroutine  calculates the non local part of the gradient for
!  all bands.
!  it is only for performance testing
!
!*********************************************************************

  SUBROUTINE RACCT(NONLR_S,WDES,W,GRID,CDIJ,CQIJ,ISP,LMDIM, NK)
    USE mpimy
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes)     WDES
    TYPE (wavedes1)    WDES1
    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    ! local variables
    TYPE (wavefun1)  W1(WDES%NSIM)
    LOGICAL ::       LDO(WDES%NSIM)
    COMPLEX(q),ALLOCATABLE :: CWORK(:,:),CWORK2(:)
    REAL(q) :: EVALUE(WDES%NSIM)

    LDO=.TRUE.
    NSIM = WDES%NSIM
    DO NT=1,NONLR_S%NTYP
       IF (NONLR_S%LMMAX(NT)/=0) GOTO 300
    ENDDO
    RETURN

300 CONTINUE
    ALLOCATE(CWORK(GRID%MPLWV,NSIM),CWORK2(GRID%MPLWV))


    ! setup descriptor
    CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

    NPL=WDES%NPLWKP(NK)
    NGVECTOR=WDES%NGVECTOR(NK)

    DO ISP=1,WDES%ISPIN
       DO N=1,WDES%NBANDS,NSIM
          NUP=MIN(N+NSIM-1,WDES%NBANDS)
          CWORK=0
          DO NN=N,NUP
             NNP=NN-N+1
             CALL SETWAV(W,W1(NNP),WDES1,NN,ISP)
             EVALUE(NNP)=W%CELEN(N,1,ISP)
          ENDDO
          CALL RACCMU(NONLR_S,WDES1,W1, LMDIM,CDIJ(1,1,1,ISP),CQIJ(1,1,1,ISP),EVALUE,CWORK(1,1), &
               WDES1%GRID%MPLWV*WDES1%NRSPINORS, NSIM, LDO)
          DO NN=N,NUP
             NNP=NN-N+1
             DO  ISPINOR=0,WDES%NRSPINORS-1
                CALL FFTEXT(NGVECTOR,WDES%NINDPW(1,NK),CWORK(1+ISPINOR*GRID%MPLWV,NNP),CWORK2(1+ISPINOR*NGVECTOR),GRID,.FALSE.)
             ENDDO
          ENDDO


       ENDDO
    ENDDO
    DEALLOCATE(CWORK,CWORK2)

    RETURN
  END SUBROUTINE RACCT

!****************** subroutine RLACC    ******************************
!
!  subroutine for calculating the non local contribution of
!  the Hamiltonian, using real space projection scheme
!  the result of the wavefunction projected on the projection operatores
!  must be given in CPROJ
!  the result is added to  CRACC
!                !!!!!
!*********************************************************************

  SUBROUTINE RACC(NONLR_S, W1, CDIJ, CQIJ, ISP, EVALUE,  CRACC)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavefun1)    W1
    COMPLEX(q)  CRACC(:)
    OVERLAP  CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER  ISP
  ! work arrays
    GDEF :: CRESUL(W1%WDES1%NPROD)

    CALL OVERL1(W1%WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL RACC0(NONLR_S, W1%WDES1, CRESUL(1), CRACC(1))

    RETURN
  END SUBROUTINE RACC



!****************** subroutine RACCMU  ******************************
!
!  subroutine for calculating the non local contribution of
!  the Hamiltonian, using real space projection scheme
!  for a set of bands simultaneously
!  the result is added to  CRACC
!                -----
!*********************************************************************

!
! scheduled for removal
!


  SUBROUTINE RACCMU(NONLR_S,WDES1,W1, &
       &     LMDIM,CDIJ,CQIJ,EVALUE, CRACC,LD, NSIM, LDO)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavedes)    WDES
    TYPE (wavefun1)    W1(NSIM)

    COMPLEX(q) CRACC(LD, NSIM)
    OVERLAP    CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
         CQIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
    REAL(q)    EVALUE(NSIM)
    LOGICAL    LDO(NSIM)
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,NSIM)

    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL OVERL1(WDES1, LMDIM,CDIJ,CQIJ, EVALUE(NP), W1(NP)%CPROJ(1),CRESUL(1,NP))
       ENDIF
    ENDDO
    IF (NSIM/=1) THEN
       CALL RACC0MU(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1),LD, NSIM,LDO)
    ELSE
       CALL RACC0(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1))
    ENDIF

    RETURN
  END SUBROUTINE RACCMU


  SUBROUTINE RACCMU_(NONLR_S, WDES1, W1, CDIJ, CQIJ, ISP, EVALUE, CRACC)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(:)
    OVERLAP CDIJ(:,:,:,:), CQIJ(:,:,:,:)
    INTEGER :: ISP
    REAL(q)    EVALUE(:)
    COMPLEX(q) CRACC(:,:)
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,SIZE(W1))
    INTEGER :: NP

    DO NP=1,SIZE(W1)
       IF (W1(NP)%LDO) THEN
          CALL OVERL1(WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), EVALUE(NP), W1(NP)%CPROJ(1), CRESUL(1,NP))
       ENDIF
    ENDDO
    IF (SIZE(W1)/=1) THEN
       CALL RACC0MU(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1),SIZE(CRACC,1), SIZE(W1),W1%LDO)
    ELSE
       CALL RACC0(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1))
    ENDIF

    RETURN
  END SUBROUTINE RACCMU_

  SUBROUTINE RACCMU_GPU(NONLR_S, WDES1, W1, CDIJ, CQIJ, ISP, EVALUE, GPU_CRACC_ALL, TIMETRANS)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(:)
    OVERLAP CDIJ(:,:,:,:), CQIJ(:,:,:,:)
    INTEGER :: ISP
    REAL(q)    EVALUE(:)
    INTEGER(c_intptr_t) GPU_CRACC_ALL
	REAL(q) TIMETRANS
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,SIZE(W1))
    INTEGER :: NP
! DDNVIDIA local variables
   !INTEGER :: BATCH_SIZE

    DO NP=1,SIZE(W1)
       IF (W1(NP)%LDO) THEN
          CALL OVERL1(WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), EVALUE(NP), W1(NP)%CPROJ(1), CRESUL(1,NP))
       ENDIF
    ENDDO
    CALL RACC0MU_GPU(NONLR_S,WDES1,CRESUL(1,1),GPU_CRACC_ALL,SIZE(W1),W1%LDO,NV_NUM_BATCHES)
    RETURN
  END SUBROUTINE RACCMU_GPU


!
! same as before but for complex EVALUE
! scheduled for removal
!
  SUBROUTINE RACCMU_C(NONLR_S,WDES1,W1, &
       &     LMDIM,CDIJ,CQIJ,CEVALUE, CRACC,LD, NSIM, LDO)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavedes)    WDES
    TYPE (wavefun1)    W1(NSIM)

    COMPLEX(q) CRACC(LD, NSIM)
    OVERLAP    CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
         CQIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
    COMPLEX(q) CEVALUE(NSIM)
    LOGICAL    LDO(NSIM)
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,NSIM)

    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL OVERL1_C(WDES1, LMDIM,CDIJ,CQIJ, CEVALUE(NP), W1(NP)%CPROJ(1),CRESUL(1,NP))
       ENDIF
    ENDDO
    IF (NSIM/=1) THEN
       CALL RACC0MU(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1),LD, NSIM,LDO)
    ELSE
       CALL RACC0(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1))
    ENDIF

    RETURN
  END SUBROUTINE RACCMU_C

  SUBROUTINE RACCMU_C_(NONLR_S, WDES1, W1, &
       &     CDIJ, CQIJ, ISP, CEVALUE, CRACC)
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1(:)
    OVERLAP CDIJ(:,:,:,:), CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CEVALUE(:)
    COMPLEX(q) CRACC(:,:)
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,SIZE(W1))
    INTEGER :: NP

    DO NP=1,SIZE(W1)
       IF (W1(NP)%LDO) THEN
          CALL OVERL1_C(WDES1, SIZE(CDIJ,1),CDIJ(1,1,1,ISP),CQIJ(1,1,1,ISP), CEVALUE(NP), W1(NP)%CPROJ(1),CRESUL(1,NP))
       ENDIF
    ENDDO
    IF (SIZE(W1)/=1) THEN
       CALL RACC0MU(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1),SIZE(CRACC,1), SIZE(W1),W1%LDO)
    ELSE
       CALL RACC0(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1))
    ENDIF

    RETURN
  END SUBROUTINE RACCMU_C_

! here CDIJ and QCIJ are always complex
  SUBROUTINE RACCMU_CCDIJ(NONLR_S,WDES1,W1, &
       &     LMDIM,CDIJ,CQIJ,EVALUE, CRACC,LD, NSIM, LDO)
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)    WDES1
    TYPE (wavedes)    WDES
    TYPE (wavefun1)    W1(NSIM)

    COMPLEX(q) CRACC(LD, NSIM)
    COMPLEX(q) CDIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS), &
         CQIJ(LMDIM,LMDIM,WDES1%NIONS,WDES1%NRSPINORS*WDES1%NRSPINORS)
    REAL(q)    EVALUE(NSIM)
    LOGICAL    LDO(NSIM)
    ! work arrays
    GDEF :: CRESUL(WDES1%NPROD,NSIM)

    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL OVERL1_CCDIJ(WDES1, LMDIM,CDIJ,CQIJ, EVALUE(NP), W1(NP)%CPROJ(1),CRESUL(1,NP))
       ENDIF
    ENDDO
    IF (NSIM/=1) THEN
       CALL RACC0MU(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1),LD, NSIM,LDO)
    ELSE
       CALL RACC0(NONLR_S,WDES1,CRESUL(1,1),CRACC(1,1))
    ENDIF

    RETURN
  END SUBROUTINE RACCMU_CCDIJ


!****************** subroutine RNLPR     *******************************
! subroutine for calculating the non-local energy per ion
!
! E(ION,k) = SUM(BAND,L,M) Z(ION,BAND,L,M,k) CONJG( Z(ION,BAND,L,M,k))
!
!***********************************************************************

  SUBROUTINE RNLPR(GRID,NONLR_S, P, LATT_FIN1, LATT_FIN2, LATT_CUR, W, WDES, &
       &    LMDIM, CDIJ, CQIJ, ENL)
    USE pseudo
    USE mpimy
    USE lattice
    USE constant
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (potcar)      P(NONLR_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W,WTMP
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR,LATT_FIN1,LATT_FIN2
    INTEGER LMDIM
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    REAL(q) ENL(NONLR_S%NIONS)
  ! local
    INTEGER NK, ISP, N, ISPINOR, ISPINOR_, LBASE, LBASE_, NIS, NT, LMMAXC, &
         NI, L, LP, NP
    REAL(q) EVALUE, WEIGHT
    GDEF,ALLOCATABLE,TARGET :: CPROW(:,:,:,:)
    REAL(q) DISPL(3,NONLR_S%NIONS)

#ifdef TAU_PROF
    integer, dimension (2,2) :: profile
    save profile
#endif


    ALLOCATE(CPROW(WDES%NPROD,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN))
!=======================================================================
!  calculate the projection operator
!=======================================================================
    NK=1
    DISPL=0
    CALL RSPHER_ALL(GRID,NONLR_S,LATT_FIN1,LATT_FIN2,LATT_CUR,DISPL,DISPL, 1)

    ENL=0
    WTMP=W
    WTMP%CPROJ => CPROW  ! relink the CPROJ array to temporary workspace

    kpoint: DO NK=1,WDES%NKPTS

#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
       CALL PHASER(GRID,LATT_CUR,NONLR_S, NK,WDES)

#ifdef TAU_PROF
       call TAU_PROFILE_TIMER(profile(1,1),'RNLPR_RPRO')
       call TAU_PROFILE_START(profile(1,1))
#endif
       CALL RPRO(NONLR_S,WDES,WTMP,GRID,NK)

#ifdef TAU_PROF
       call TAU_PROFILE_STOP(profile(1,1))
#endif

       spin: DO ISP=1,WDES%ISPIN
!=======================================================================
!  sum up to give the non-local energy per ion
!=======================================================================

          band: DO N=1,WDES%NBANDS
             EVALUE=W%CELEN(N,NK,ISP)
             WEIGHT=WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*WDES%RSPIN

             spinor: DO ISPINOR=0,WDES%NRSPINORS-1
                DO ISPINOR_=0,WDES%NRSPINORS-1

                   LBASE =ISPINOR *WDES%NPRO/2
                   LBASE_=ISPINOR_*WDES%NPRO/2

                   NIS=1
                   typ: DO NT=1,WDES%NTYP
                      LMMAXC=WDES%LMMAX(NT)
                      IF (LMMAXC==0) GOTO 510

                      ion: DO NI=NIS,WDES%NITYP(NT)+NIS-1
!DIR$ IVDEP
!OCL NOVREC
                         DO L=1 ,LMMAXC
                            DO LP=1,LMMAXC
                               ENL(NI)=ENL(NI)+WEIGHT*W%CPROJ(LBASE_+LP,N,NK,ISP)*GCONJG(CPROW(LBASE+L,N,NK,ISP))* &
                                    (CDIJ(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)-EVALUE*CQIJ(LP,L,NI,ISP+ISPINOR_+2*ISPINOR))
                            ENDDO
                         ENDDO
                         LBASE = LMMAXC+LBASE
                         LBASE_= LMMAXC+LBASE_
                      ENDDO ion
510                   NIS = NIS+WDES%NITYP(NT)
                   ENDDO typ
                ENDDO
             ENDDO spinor

          ENDDO band
       ENDDO spin

    ENDDO kpoint

    CALLMPI( M_sum_d( WDES%COMM_KINTER, ENL(1),NONLR_S%NIONS) )
    DEALLOCATE(CPROW)
    RETURN
  END SUBROUTINE RNLPR

!****************** subroutine STRNLR    *******************************
!
!  subroutine for calculating the non-local contributions to stress,
!  use central differences
!  all components to the stress tensor are calculated
!  except if ISIF = 1
!
!  uncomment CTEST-lines if you want to test finit-differences
!
!***********************************************************************

  SUBROUTINE STRNLR(GRID,NONLR_S,P,LATT_CUR,W, &
      &    CDIJ,CQIJ, ISIF,FNLSIF)
    USE pseudo
    USE mpimy
    USE lattice
    USE constant
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    TYPE (nonlr_struct) NONLR_S
    TYPE (potcar)      P(NONLR_S%NTYP)
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    OVERLAP CDIJ(:,:,:,:)
    OVERLAP CQIJ(:,:,:,:)
    INTEGER ISIF                         ! which componets
    REAL(q) FNLSIF(3,3)                ! result stress tensor
  ! local
    TYPE (latt)        LATT_FIN1,LATT_FIN2
    INTEGER :: IDIR, JDIR, I, J, NI
  ! magnitude used to distort lattice
    REAL(q) :: DIS=fd_displacement
    REAL(q) ::  ENL(NONLR_S%NIONS)

#ifdef TAU_PROF
    integer, dimension (2,2) :: profile
    save profile
#endif

    !TEST
    !      DIS=1E-3
    ! 1000 DIS=DIS/2
    !TEST
!=======================================================================
! initialise non-local forces to zero
!=======================================================================
    FNLSIF=0
!=======================================================================
! calculate the contribution to the energy from the nonlocal
! pseudopotential for elongation of each basis-vector
!=======================================================================
    DO IDIR=1,3
       DO JDIR=1,3

          LATT_FIN1=LATT_CUR
          LATT_FIN2=LATT_CUR
          IF (ISIF==1) THEN
             !  only isotrop pressure
             DO I=1,3
                DO J=1,3
                   LATT_FIN1%A(I,J)=LATT_CUR%A(I,J)*(1+DIS/3)
                   LATT_FIN2%A(I,J)=LATT_CUR%A(I,J)*(1-DIS/3)
                ENDDO
             ENDDO
          ELSE
             !  all directions
             DO I=1,3
                LATT_FIN1%A(IDIR,I)=LATT_CUR%A(IDIR,I)+DIS*LATT_CUR%A(JDIR,I)
                LATT_FIN2%A(IDIR,I)=LATT_CUR%A(IDIR,I)-DIS*LATT_CUR%A(JDIR,I)
             ENDDO
          ENDIF
          CALL LATTIC(LATT_FIN1)
          CALL LATTIC(LATT_FIN2)

#ifdef TAU_PROF
          call TAU_PROFILE_TIMER(profile(1,1),'RNLPR')
          call TAU_PROFILE_START(profile(1,1))
#endif
          CALL RNLPR(GRID,NONLR_S,P,LATT_FIN1,LATT_FIN2,LATT_CUR,W,W%WDES, &
               &    SIZE(CDIJ,1),CDIJ,CQIJ,ENL)

#ifdef TAU_PROF
          call TAU_PROFILE_STOP(profile(1,1))
#endif

          DO NI=1,NONLR_S%NIONS
             FNLSIF(IDIR,JDIR)=FNLSIF(IDIR,JDIR)+ENL(NI)
          ENDDO
          !
          !  only isotrop pressure terminate loop
          !
          IF (ISIF==1) THEN
             FNLSIF(2,2)= FNLSIF(1,1)
             FNLSIF(3,3)= FNLSIF(1,1)
             GOTO 400 ! terminate (not very clean but who cares)
          ENDIF

       ENDDO
    ENDDO
!=======================================================================
! calculation finished  scale pressure
!=======================================================================
400 CONTINUE

    CALLMPI( M_sum_d(W%WDES%COMM_KIN, FNLSIF, 9))
    FNLSIF=FNLSIF/DIS
    !TEST
    !      WRITE(*,'(E10.3,3E14.7)')DIS,((FNLSIF(I,J),I=1,3),J=1,3)
    !      IF (DIS>1E-10) GOTO 1000
    !TEST

    RETURN
  END SUBROUTINE STRNLR


!****************** subroutine FORNLR    *******************************
!
!  subroutine for calculating the non local contribution
!  to the forces acting onto the ions (using simple central finite
!  differences)
!  uncomment CTEST-lines if you want to test finit-differences
!
!***********************************************************************

  SUBROUTINE FORNLR(GRID, NONLR_S, P, LATT_CUR, W, &
       &    CDIJ, CQIJ, DISPL0, FORNL)
    USE pseudo
    USE mpimy
    USE lattice
    USE constant

    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (potcar)      P(NONLR_S%NTYP)
    TYPE (wavespin), TARGET ::    W
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR

    REAL(q) FORNL(3,NONLR_S%NIONS)
    REAL(q) DISPL0(3,NONLR_S%NIONS)
    OVERLAP CDIJ(:,:,:,:)
    OVERLAP CQIJ(:,:,:,:)
  ! local
    TYPE (wavespin)    WTMP
    TYPE (wavedes), POINTER :: WDES
    REAL(q) ENL(NONLR_S%NIONS), EVALUE, WEIGHT
    COMPLEX(q) CE
    INTEGER IDIR, NK, ISP, N, ISPINOR, ISPINOR_, LBASE, LBASE_, NIS, NT, &
         LMMAXC, NI, L, LP, NIP
    REAL(q) DISPL1(3,NONLR_S%NIONS),DISPL2(3,NONLR_S%NIONS)
    ! allocate required work space
    GDEF,ALLOCATABLE,TARGET :: CPROW(:,:,:,:)
    GDEF,POINTER :: CPROT(:,:,:,:)
  ! magnitude used for finite differencesq
    REAL(q) :: DIS=fd_displacement

#ifdef TAU_PROF
    integer, dimension (2,2) :: profile
    save profile
#endif

    WDES=>W%WDES

    ALLOCATE(CPROW(WDES%NPROD,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN))
    !TEST
    !      DIS=1E-3
    ! 1000 DIS=DIS/2
    !TEST
!=======================================================================
! initialise non-local forces to zero
!=======================================================================
    FORNL=0
!=======================================================================
! calculate the contribution to the force from the nonlocal
! projection functions for displacement X using central (semianalytical)
! finite differences (about 9 digits precision)
!=======================================================================
    dir: DO IDIR=1,3
       ENL=0

       DISPL1=DISPL0
       DISPL1(IDIR,:)= DISPL0(IDIR,:)-DIS

       DISPL2=DISPL0
       DISPL2(IDIR,:)= DISPL0(IDIR,:)+DIS
       CALL RSPHER_ALL(GRID,NONLR_S,LATT_CUR,LATT_CUR,LATT_CUR, DISPL1, DISPL2, 1)

       WTMP=W
       WTMP%CPROJ => CPROW       ! relink the CPROJ array to temporary workspace

       kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL PHASER(GRID,LATT_CUR,NONLR_S, NK,WDES)

#ifdef TAU_PROF
          call TAU_PROFILE_TIMER(profile(1,1),'FORNLR_RPRO')
          call TAU_PROFILE_START(profile(1,1))
#endif
          CALL RPRO(NONLR_S,WDES,WTMP,GRID,NK)

#ifdef TAU_PROF
          call TAU_PROFILE_STOP(profile(1,1))
#endif
          spin: DO ISP=1,WDES%ISPIN

             band: DO N=1,WDES%NBANDS
                EVALUE=W%CELEN(N,NK,ISP)
                WEIGHT=WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*WDES%RSPIN

                spinor: DO ISPINOR=0,WDES%NRSPINORS-1
                   DO ISPINOR_=0,WDES%NRSPINORS-1

                      LBASE =ISPINOR *WDES%NPRO/2
                      LBASE_=ISPINOR_*WDES%NPRO/2

                      NIS=1
                      typ: DO NT=1,WDES%NTYP
                         LMMAXC=WDES%LMMAX(NT)
                         IF (LMMAXC==0) GOTO 510

                         DO NI=NIS,WDES%NITYP(NT)+NIS-1
                            CE=0
                            CALL ECCP_NL_(SIZE(CDIJ,1),LMMAXC,CDIJ(1,1,NI,ISP+ISPINOR_+2*ISPINOR),CQIJ(1,1,NI,ISP+ISPINOR_+2*ISPINOR), &
                                 EVALUE,W%CPROJ(LBASE_+1,N,NK,ISP),CPROW(LBASE+1,N,NK,ISP),CE)
                            ENL(NI)=ENL(NI)+CE*WEIGHT

                            LBASE = LMMAXC+LBASE
                            LBASE_= LMMAXC+LBASE_
                         ENDDO
510                      NIS = NIS+WDES%NITYP(NT)
                      ENDDO typ

                   ENDDO
                ENDDO spinor
             ENDDO band
          ENDDO spin
       ENDDO kpoint

       DO NI=1,WDES%NIONS
          NIP=NI_GLOBAL(NI, WDES%COMM_INB)
          FORNL(IDIR,NIP)=FORNL(IDIR,NIP)-ENL(NI)/DIS
       ENDDO

    ENDDO dir

    CALLMPI( M_sum_d(WDES%COMM, FORNL(1,1),NONLR_S%NIONS*3))
    !TEST
    !      WRITE(*,'(4E20.12)') DIS
    !      WRITE(*,'("n",3F15.9 )') FORNL
    !      GOTO 1000
    !TEST

    DEALLOCATE(CPROW)
    RETURN
  END SUBROUTINE FORNLR


!****************** subroutine RPROXYZ   *******************************
!
! this subroutine calculates the first order change of the 
! wave function character upon moving the ions for one selected k-point
! and spin component
! the results are stored in CPROJXYZ
! mind that either the bra or the kat can vary, therefore
! a factor two has to be included
!
!***********************************************************************

  SUBROUTINE RPROXYZ(GRID, NONLR_S, P, LATT_CUR, W, WDES, ISP, NK, CPROJXYZ)
    USE pseudo
    USE mpimy
    USE lattice
    USE constant

    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (potcar)      P(NONLR_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W,WTMP
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    INTEGER            ISP, NK
    GDEF :: CPROJXYZ(WDES%NPROD, WDES%NBANDS, 3)

    !-----some temporary arrays
    REAL(q) :: DISPL(3,NONLR_S%NIONS), DIS
    GDEF,ALLOCATABLE,TARGET :: CPROW(:,:,:,:)
    INTEGER :: IDIR

    ALLOCATE(CPROW(WDES%NPROD,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN))

    CPROW(:,:,NK,ISP)=0
    DIS=fd_displacement
    DO IDIR=1,3

       DISPL=0
       ! operator= beta(r-R-dis)
       DISPL(IDIR,:)= DIS
       ! this includes the factor two since a displacement + and - is performed
       ! by RSPHER_ALL
       !   2 d projector / d dis   = projector(R+dis) - projector(R-dis) / dis
       CALL RSPHER_ALL(GRID,NONLR_S,LATT_CUR,LATT_CUR,LATT_CUR, -DISPL,DISPL,1)

       WTMP=W
       WTMP%CPROJ => CPROW    ! relink the CPROJ array to temporary workspace

       CALL RPRO_ISP_GPU(NONLR_S,WDES,WTMP,GRID,ISP,NK)

       CPROJXYZ(:,:,IDIR)=CPROW(:,:,NK,ISP)/DIS
    ENDDO
    DISPL=0
    CALL RSPHER_ALL(GRID,NONLR_S,LATT_CUR,LATT_CUR,LATT_CUR, DISPL,DISPL,0)

    DEALLOCATE(CPROW)

    RETURN
  END SUBROUTINE RPROXYZ


!****************** subroutine RPROXYZ   *******************************
!
! this subroutine calculates the first order change of the 
! wave function character upon changing the lattice
!
!***********************************************************************

  SUBROUTINE RPROLAT_DER(GRID, NONLR_S, P, LATT_CUR, W, WDES, ISP, NK, CPROJXYZ)
    USE pseudo
    USE mpimy
    USE lattice
    USE constant

    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (potcar)      P(NONLR_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W,WTMP
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    INTEGER            ISP, NK
    GDEF :: CPROJXYZ(WDES%NPROD, WDES%NBANDS, 6)

    !-----some temporary arrays
    REAL(q) :: DISPL(3,NONLR_S%NIONS), DIS
    GDEF,ALLOCATABLE,TARGET :: CPROW(:,:,:,:)
    INTEGER :: IDIR, JDIR, IJDIR
    TYPE (latt)        LATT_FIN1,LATT_FIN2

    ALLOCATE(CPROW(WDES%NPROD,WDES%NBANDS,WDES%NKPTS,WDES%ISPIN))

    CPROW(:,:,NK,ISP)=0
    DIS=fd_displacement

    IJDIR=0
    DO IDIR=1,3
       DO JDIR=1,IDIR
          IJDIR=IJDIR+1
          LATT_FIN1=LATT_CUR
          LATT_FIN1%A(IDIR,:)=LATT_CUR%A(IDIR,:)+DIS*LATT_CUR%A(JDIR,:)

          LATT_FIN2=LATT_CUR
          LATT_FIN2%A(IDIR,:)=LATT_CUR%A(IDIR,:)-DIS*LATT_CUR%A(JDIR,:)

          CALL LATTIC(LATT_FIN1)
          CALL LATTIC(LATT_FIN2)

          DISPL=0
       ! this includes the factor two since a displacement + and - is performed
       ! by RSPHER_ALL
       !   2 d projector / d dis   = projector(R+dis) - projector(R-dis) / dis
          CALL RSPHER_ALL(GRID,NONLR_S,LATT_FIN2,LATT_FIN1,LATT_CUR, DISPL,DISPL,1)

          WTMP=W
          WTMP%CPROJ => CPROW   ! relink the CPROJ array to temporary workspace
          
          CALL RPRO_ISP_GPU(NONLR_S,WDES,WTMP,GRID,ISP,NK)

          CPROJXYZ(:,:,IJDIR)=CPROW(:,:,NK,ISP)/DIS
       ENDDO
    ENDDO
    DISPL=0
    CALL RSPHER_ALL(GRID,NONLR_S,LATT_CUR,LATT_CUR,LATT_CUR, DISPL,DISPL,0)
    
    DEALLOCATE(CPROW)

    RETURN
  END SUBROUTINE RPROLAT_DER

END MODULE nonlr


!****************** subroutine RACC0   ******************************
!
! this subroutine calculates a linear combination of
! projection operatores in real space
! the result is added to CRACC
!
!*********************************************************************

  SUBROUTINE RACC0(NONLR_S,WDES1,CPROJ_LOC,CRACC)
    USE nonlr_struct_def
    USE wave
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)     WDES1
    COMPLEX(q) CRACC(WDES1%GRID%RL%NP)
    GDEF       CPROJ_LOC(WDES1%NPROD)

  ! local
    REAL(q) RP
    INTEGER IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, NI, INDMAX, L, IND
    COMPLEX(q)          :: CTMP
    REAL(qn),PARAMETER  :: ONE=1,ZERO=0
#ifdef avoidalloc
    REAL(qn) :: WORK(NONLR_S%IRMAX*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
#else 
    REAL(qn),ALLOCATABLE :: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE    :: CPROJ(:)

    ALLOCATE(WORK(NONLR_S%IRMAX*2),TMP(101,2),CPROJ(WDES1%NPRO_TOT))
#endif

    IF (WDES1%NK /= NONLR_S%NK) THEN
       WRITE(*,*) 'internal error in RACC0: PHASE not properly set up',WDES1%NK, NONLR_S%NK
       STOP
    ENDIF

    ! merge projected wavefunctions from all nodes (if distributed over 
    !   plane wave coefficients)
    CALL MRG_PROJ(WDES1,CPROJ(1),CPROJ_LOC(1))
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO  NI=NIS,NONLR_S%NITYP(NT)+NIS-1
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
!=======================================================================
! set TMP
!=======================================================================
!DIR$ IVDEP
!OCL NOVREC
             DO L=1,LMMAXC
                CTMP= CPROJ(LMBASE+L)*WDES1%RINPL
                TMP(L,1)= REAL( CTMP ,KIND=qn)
#ifndef gammareal
                TMP(L,2)=AIMAG(CTMP)
#endif
             ENDDO
!=======================================================================
! calculate SUM(LM=1,NONLR_S%LMMAX) NONLR_S%RPROJ(K,LM) * TMP(LM)
!=======================================================================
#ifdef do_loops

             ! initialise accelerations to 0 (real &  imaginary part seperately)

!DIR$ IVDEP
!OCL NOVREC
             DO IND=1,INDMAX
                WORK(IND)=0
#ifndef gammareal
                WORK(IND+NONLR_S%IRMAX)=0
#endif
             ENDDO

             DO LM=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   RP  =  NONLR_S%RPROJ(IND+(LM-1)*INDMAX+NLIIND)
                   WORK(IND)      =WORK(IND)      +TMP(LM,1)*RP
#ifndef gammareal
                   WORK(IND+NONLR_S%IRMAX)=WORK(IND+NONLR_S%IRMAX)+TMP(LM,2)*RP
#endif
                ENDDO
             ENDDO
#else
             CALL DGEMVn( 'N' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, TMP(1,1) , 1 , ZERO , WORK(1), 1)
#ifndef gammareal
             CALL DGEMVn( 'N' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, TMP(1,2) , 1 , ZERO , WORK(1+NONLR_S%IRMAX), 1)
#endif

#endif
!=======================================================================
!  add the non local contribution to the accelerations in real space
!=======================================================================
             IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   CRACC(IP)= CRACC(IP)+ &
                        GCMPLX( WORK(IND) , WORK(IND+NONLR_S%IRMAX) ,q) *CONJG(NONLR_S%CRREXP(IND,NI,ISPIRAL))
                ENDDO
#else
                CALL CRREXP_MUL_WORK_ADD(INDMAX,NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                          WORK(1),WORK(1+NONLR_S%IRMAX),CRACC(1+ISPINOR*WDES1%GRID%MPLWV))
#endif
             ELSE
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   CRACC(IP)= CRACC(IP)+WORK(IND)
                ENDDO
             ENDIF

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RACC0


!****************** subroutine RACC0_HF   ******************************
!
! this subroutine calculates a linear combination of
! projection operatores in real space
! the only difference is the CRACC is defined as GDEF whereas it is
! defined COMPLEX in the previous version
!
!*********************************************************************

  SUBROUTINE RACC0_HF(NONLR_S, WDES1, CPROJ_LOC, CRACC)
    USE nonlr_struct_def
    USE wave
    IMPLICIT NONE
    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)     WDES1
    GDEF   CRACC(WDES1%GRID%RL%NP)
    GDEF   CPROJ_LOC(WDES1%NPROD)

    ! work array
    REAL(q) RP
    INTEGER IP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, NI, INDMAX, L, IND, LM
    COMPLEX(q)          :: CTMP
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
#ifdef avoidalloc
    REAL(qn) :: WORK(NONLR_S%IRMAX*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
#else
    REAL(qn),ALLOCATABLE :: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE    :: CPROJ(:)

    ALLOCATE(WORK(NONLR_S%IRMAX*2),TMP(101,2),CPROJ(WDES1%NPRO_TOT))
#endif

    ! merge projected wavefunctions from all nodes (if distributed over 
    !   plane wave coefficients)
    CALL MRG_PROJ(WDES1,CPROJ(1),CPROJ_LOC(1))
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO  NI=NIS,NONLR_S%NITYP(NT)+NIS-1
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
!=======================================================================
! set TMP
!=======================================================================
!DIR$ IVDEP
!OCL NOVREC
             DO L=1,LMMAXC
                CTMP= CPROJ(LMBASE+L)*WDES1%RINPL
                TMP(L,1)= REAL( CTMP ,KIND=qn)
#ifndef gammareal
                TMP(L,2)=AIMAG(CTMP)
#endif
             ENDDO
!=======================================================================
! calculate SUM(LM=1,NONLR_S%LMMAX) NONLR_S%RPROJ(K,LM) * TMP(LM)
!=======================================================================
#ifdef do_loops

             ! initialise accelerations to 0 (real &  imaginary part seperately)

!DIR$ IVDEP
!OCL NOVREC
             DO IND=1,INDMAX
                WORK(IND)=0
#ifndef gammareal
                WORK(IND+NONLR_S%IRMAX)=0
#endif
             ENDDO

             DO LM=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   RP  =  NONLR_S%RPROJ(IND+(LM-1)*INDMAX+NLIIND)
                   WORK(IND)      =WORK(IND)      +TMP(LM,1)*RP
#ifndef gammareal
                   WORK(IND+NONLR_S%IRMAX)=WORK(IND+NONLR_S%IRMAX)+TMP(LM,2)*RP
#endif
                ENDDO
             ENDDO
#else
             CALL DGEMVn( 'N' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, TMP(1,1) , 1 , ZERO , WORK(1), 1)
#ifndef gammareal
             CALL DGEMVn( 'N' , INDMAX, LMMAXC, ONE , NONLR_S%RPROJ(1+NLIIND), &
                  INDMAX, TMP(1,2) , 1 , ZERO , WORK(1+NONLR_S%IRMAX), 1)
#endif

#endif
!=======================================================================
!  add the non local contribution to the accelerations in real space
!=======================================================================
             IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   CRACC(IP)= CRACC(IP)+ &
                        GCMPLX( WORK(IND) , WORK(IND+NONLR_S%IRMAX) ,q) *CONJG(NONLR_S%CRREXP(IND,NI,ISPIRAL))
                ENDDO
#else
                CALL CRREXP_MUL_WORK_GADD(INDMAX,NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                          WORK(1),WORK(1+NONLR_S%IRMAX),CRACC(1+ISPINOR*WDES1%GRID%MPLWV))
#endif
             ELSE
!DIR$ IVDEP
!OCL NOVREC
                DO IND=1,INDMAX
                   IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                   CRACC(IP)= CRACC(IP)+WORK(IND)
                ENDDO
             ENDIF


100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ

       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RACC0_HF


!****************** subroutine RACCMU   ******************************
!
! this subroutine calculates a set of linear combination of
! projection operatores in real space
! the result is added to CRACC
!               -----
!*********************************************************************

  SUBROUTINE RACC0MU(NONLR_S, WDES1, CPROJ_LOC, CRACC, LD, NSIM, LDO)
    USE nonlr_struct_def
    USE wave
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)     WDES1
    INTEGER LD                        ! leading dimension of CRACC
    INTEGER NSIM                      ! do NSIM bands at a time
    COMPLEX(q) CRACC(LD,NSIM)         ! result in real space
    GDEF   CPROJ_LOC(WDES1%NPROD,NSIM)! wave function character
    LOGICAL LDO(NSIM)                 ! which bands are included

  ! local
    INTEGER, PARAMETER  :: NLM=101
    REAL(qn),PARAMETER  :: ONE=1,ZERO=0
    INTEGER NP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, NI, &
         INDMAX, IND0, NPFILL, L, IND, IP
    COMPLEX(q)          :: CTMP
#ifdef avoidalloc
    GDEF    :: CPROJ(WDES1%NPRO_TOT,NSIM)
    REAL(qn) :: WORK(ndata*NSIM*NONLR_S%IRMAX),TMP(NLM,ndata*2*NSIM)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE    :: CPROJ(:,:)

    ALLOCATE(WORK(ndata*NSIM*NONLR_S%IRMAX),TMP(NLM,ndata*2*NSIM), &
         CPROJ(WDES1%NPRO_TOT,NSIM))
#endif
    IF (WDES1%NK /= NONLR_S%NK) THEN
       WRITE(*,*) 'internal error in RACC0MU: PHASE not properly set up',WDES1%NK, NONLR_S%NK
       STOP
    ENDIF

    ! merge projected wavefunctions from all nodes

    DO NP=1,NSIM
       IF (LDO(NP)) THEN
          CALL MRG_PROJ(WDES1,CPROJ(1,NP),CPROJ_LOC(1,NP))
       ENDIF
    ENDDO
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
!=======================================================================
! set TMP
!=======================================================================
             IND0=0
             NPFILL=0
             DO NP=1,NSIM
                IF (LDO(NP)) THEN
!DIR$ IVDEP
!OCL NOVREC

                   DO L=1,LMMAXC
                      CTMP= CPROJ(LMBASE+L,NP)*WDES1%RINPL
                      TMP(L,1+IND0)= REAL( CTMP ,KIND=qn)
#ifndef gammareal
                      TMP(L,2+IND0)=AIMAG(CTMP)
#endif
                   ENDDO
                   IND0=IND0+ndata
                   NPFILL=NPFILL+1
                ENDIF

             ENDDO
!=======================================================================
! calculate SUM(LM=1,NONLR_S%LMMAX) NONLR_S%RPROJ(K,LM) * TMP(LM)
!=======================================================================
#ifdef RACCMU_DGEMV
             DO NP=0,ndata*NPFILL-1
                CALL DGEMVn( 'N' , INDMAX, LMMAXC,  ONE , NONLR_S%RPROJ(1+NLIIND), &
                     INDMAX,  TMP(1,1+NP) , 1 , ZERO ,  WORK(1+NONLR_S%IRMAX*NP), 1)
             ENDDO
#else
             CALL DGEMMn( 'N' , 'N', INDMAX, ndata*NPFILL, LMMAXC, ONE, &
                  NONLR_S%RPROJ(1+NLIIND), INDMAX, TMP(1,1) , NLM , &
                  ZERO , WORK(1), NONLR_S%IRMAX)
#endif
!=======================================================================
!  add the non local contribution to the accelerations
!=======================================================================
             IND0=0
             DO NP=1,NSIM
                IF (LDO(NP)) THEN
                   IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         CRACC(IP,NP)= CRACC(IP,NP)+ &
                              GCMPLX( WORK(IND+IND0), WORK(IND+(NONLR_S%IRMAX+IND0)) ,q)* &
                              CONJG(NONLR_S%CRREXP(IND,NI,ISPIRAL))
                      ENDDO
#else
                      CALL CRREXP_MUL_WORK_ADD(INDMAX,NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                          WORK(1+IND0),WORK(1+IND0+NONLR_S%IRMAX),CRACC(1+ISPINOR*WDES1%GRID%MPLWV,NP))
#endif
                   ELSE
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         CRACC(IP,NP)= CRACC(IP,NP)+WORK(IND+IND0)
                      ENDDO
                   ENDIF

                   IND0=IND0+ndata * NONLR_S%IRMAX
                ENDIF
             ENDDO

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RACC0MU

!****************** subroutine RACCMU_HF *****************************
!
! exact copy of the previous routine with CRACC defined as GDEF
! phase factor test is also removed
!
!*********************************************************************

  SUBROUTINE RACC0MU_HF(NONLR_S, WDES1, CPROJ_LOC, LD1, CRACC, LD2, NSIM)
    USE nonlr_struct_def
    USE wave
    IMPLICIT NONE

    TYPE (nonlr_struct) NONLR_S
    TYPE (wavedes1)     WDES1
    INTEGER LD1,LD2                   ! leading dimension of CPROJ_LOC and CRACC
    INTEGER NSIM                      ! do NSIM bands at a time
    GDEF   CPROJ_LOC(LD1,NSIM)        ! wave function character
    GDEF   CRACC(LD2,NSIM)            ! result in real space

  ! local
    INTEGER, PARAMETER  :: NLM=101
    REAL(qn),PARAMETER  :: ONE=1,ZERO=0
    INTEGER NP, LMBASE, ISPIRAL, ISPINOR, NLIIND, NIS, NT, LMMAXC, NI, &
         INDMAX, IND0, NPFILL, L, IND, IP
    COMPLEX(q)          :: CTMP
#ifdef avoidalloc
    GDEF    :: CPROJ(WDES1%NPRO_TOT,NSIM)
    REAL(qn) :: WORK(ndata*NSIM*NONLR_S%IRMAX),TMP(NLM,ndata*2*NSIM)
#else
    REAL(qn),ALLOCATABLE:: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE    :: CPROJ(:,:)

    ALLOCATE(WORK(ndata*NSIM*NONLR_S%IRMAX),TMP(NLM,ndata*2*NSIM), &
         CPROJ(WDES1%NPRO_TOT,NSIM))
#endif
    ! merge projected wavefunctions from all nodes

    DO NP=1,NSIM
       CALL MRG_PROJ(WDES1,CPROJ(1,NP),CPROJ_LOC(1,NP))
    ENDDO
!=======================================================================
! loop over ions
!=======================================================================
    LMBASE= 0

    ISPIRAL = 1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NLIIND= 0
       NIS=1

       typ: DO NT=1,NONLR_S%NTYP
          LMMAXC=NONLR_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONLR_S%NITYP(NT)+NIS-1
             INDMAX=NONLR_S%NLIMAX(NI)
             IF (INDMAX == 0) GOTO 100
!=======================================================================
! set TMP
!=======================================================================
             IND0=0
             NPFILL=0
             DO NP=1,NSIM
!DIR$ IVDEP
!OCL NOVREC
                   DO L=1,LMMAXC
                      CTMP= CPROJ(LMBASE+L,NP)*WDES1%RINPL
                      TMP(L,1+IND0)= REAL( CTMP ,KIND=qn)
#ifndef gammareal
                      TMP(L,2+IND0)=AIMAG(CTMP)
#endif
                   ENDDO
                   IND0=IND0+ndata
                   NPFILL=NPFILL+1
             ENDDO
!=======================================================================
! calculate SUM(LM=1,NONLR_S%LMMAX) NONLR_S%RPROJ(K,LM) * TMP(LM)
!=======================================================================
#ifdef RACCMU_DGEMV
             DO NP=0,ndata*NPFILL-1
                CALL DGEMVn( 'N' , INDMAX, LMMAXC,  ONE , NONLR_S%RPROJ(1+NLIIND), &
                     INDMAX,  TMP(1,1+NP) , 1 , ZERO ,  WORK(1+NONLR_S%IRMAX*NP), 1)
             ENDDO
#else
             CALL DGEMMn( 'N' , 'N', INDMAX, ndata*NPFILL, LMMAXC, ONE, &
                  NONLR_S%RPROJ(1+NLIIND), INDMAX, TMP(1,1) , NLM , &
                  ZERO , WORK(1), NONLR_S%IRMAX)
#endif
!=======================================================================
!  add the non local contribution to the accelerations
!=======================================================================
             IND0=0
             DO NP=1,NSIM
                   IF (ASSOCIATED(NONLR_S%CRREXP)) THEN
#ifdef CRREXP_inline
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         CRACC(IP,NP)= CRACC(IP,NP)+ &
                              GCMPLX( WORK(IND+IND0), WORK(IND+(NONLR_S%IRMAX+IND0)) ,q)* &
                              CONJG(NONLR_S%CRREXP(IND,NI,ISPIRAL))
                      ENDDO
#else
                      CALL CRREXP_MUL_WORK_GADD(INDMAX,NONLR_S%CRREXP(1,NI,ISPIRAL),NONLR_S%NLI(1,NI), &
                          WORK(1+IND0),WORK(1+IND0+NONLR_S%IRMAX),CRACC(1+ISPINOR*WDES1%GRID%MPLWV,NP))
#endif
                   ELSE
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         IP=NONLR_S%NLI(IND,NI)+ISPINOR*WDES1%GRID%MPLWV
                         CRACC(IP,NP)= CRACC(IP,NP)+WORK(IND+IND0)
                      ENDDO
                   ENDIF

                   IND0=IND0+ndata * NONLR_S%IRMAX
             ENDDO

100          LMBASE= LMMAXC+LMBASE
             NLIIND= LMMAXC*INDMAX+NLIIND
          ENDDO ion
600       NIS = NIS+NONLR_S%NITYP(NT)
       ENDDO typ
       IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif
    RETURN
  END SUBROUTINE RACC0MU_HF


!****************** subroutine INIT_NONLR_GPU ************************
!
! this subroutine copies from host to device and allocates 
! device arrays used in RACC0MU_GPU and RPROMU_GPU
!
!*********************************************************************
SUBROUTINE INIT_NONLR_GPU(NONLR_S, WDES1, NSIM, BATCH_SIZE)
    USE iso_c_binding
    USE cuda_interface
    use nonlr_struct_def
    use wave
    implicit none

    ! input parameters
    type(nonlr_struct) NONLR_S
    type(wavedes1) WDES1
    integer NSIM
    integer BATCH_SIZE

    ! local variables
    integer, parameter :: NLM=101  ! a constant in CPU code as well
    GDEF :: sizegdef
    integer sizei
    real(qn) :: sizer
    complex(q) :: sizec

    ! copy from host to device
    call cuda_memcpyhtod(0,NONLR_S%CRREXP_GPU,c_loc(NONLR_S%CRREXP),size(NONLR_S%CRREXP,1)*size(NONLR_S%CRREXP,2)*size(NONLR_S%CRREXP,3),int(c_sizeof(NONLR_S%CRREXP(1,1,1)),c_size_t))
    call cuda_memcpyhtod(0,NONLR_S%NLI_GPU,c_loc(NONLR_S%NLI),size(NONLR_S%NLI,1)*size(NONLR_S%NLI,2),int(c_sizeof(NONLR_S%NLI(1,1)),c_size_t))
    call cuda_memcpyhtod(0,NONLR_S%RPROJ_GPU,c_loc(NONLR_S%RPROJ),NONLR_S%IRALLOC,int(c_sizeof(sizer),c_size_t))

    ! allocate host arrays
#if USE_PINNED_MEMORY
    call nvpinnedmalloc(NONLR_S%CPROJ_PTR, WDES1%NPRO_TOT*NSIM*int(c_sizeof(sizegdef),c_size_t))
    call c_f_pointer(NONLR_S%CPROJ_PTR, NONLR_S%CPROJ, (/WDES1%NPRO_TOT, NSIM/))
    call nvpinnedmalloc(NONLR_S%LDO_INDEX_PTR, NSIM*int(c_sizeof(sizei),c_size_t))
    call c_f_pointer(NONLR_S%LDO_INDEX_PTR, NONLR_S%LDO_INDEX, (/NSIM/))
#else
    !ALLOCATE(CPROJ(WDES1%NPRO_TOT,NSIM),LDO_IND(NSIM))
    ALLOCATE(NONLR_S%CPROJ(WDES1%NPRO_TOT,NSIM))
    ALLOCATE(NONLR_S%LDO_INDEX(NSIM))
#endif

    ! allocate device arrays
    call cublas_alloc_safety(WDES1%NPRO_TOT*NSIM,int(c_sizeof(sizegdef),c_size_t),NONLR_S%CPROJ_GPU)
    call cublas_alloc_safety(NSIM,int(c_sizeof(sizei),c_size_t),NONLR_S%LDO_INDEX_GPU)
    call cublas_alloc_safety(ndata*NSIM*NONLR_S%IRMAX*BATCH_SIZE,int(c_sizeof(sizer),c_size_t),NONLR_S%WORK_GPU)
    call cublas_alloc_safety(NLM*ndata*NSIM*BATCH_SIZE,int(c_sizeof(sizer),c_size_t),NONLR_S%TMP_GPU)
    return
END SUBROUTINE INIT_NONLR_GPU

!****************** subroutine DESTROY_NONLR_GPU *********************
!
! this subroutine frees device arrays used in RACC0MU_GPU and RPROMU_GPU
!
!*********************************************************************
SUBROUTINE DESTROY_NONLR_GPU(NONLR_S, WDES1, NSIM, BATCH_SIZE)
    USE iso_c_binding
    USE cuda_interface
    use nonlr_struct_def
    use wave
    implicit none

    ! input parameters
    type(nonlr_struct) NONLR_S
    type(wavedes1) WDES1
    integer NSIM
    integer BATCH_SIZE

    ! local variables
    integer, parameter :: NLM=101  ! a constant in CPU code as well
    GDEF :: sizegdef
    integer sizei
    real(qn) :: sizer
    complex(q) :: sizec

    ! free host arrays
#if USE_PINNED_MEMORY
    call nvpinnedfree(NONLR_S%CPROJ_PTR)
    NULLIFY(NONLR_S%CPROJ)
    call nvpinnedfree(NONLR_S%LDO_INDEX_PTR)
    NULLIFY(NONLR_S%LDO_INDEX)
#else
    DEALLOCATE(NONLR_S%CPROJ)
    DEALLOCATE(NONLR_S%LDO_INDEX)
#endif

    ! free device arrays
    call cublas_free(NONLR_S%CPROJ_GPU)
    call cublas_free(NONLR_S%LDO_INDEX_GPU)
    call cublas_free(NONLR_S%WORK_GPU)
    call cublas_free(NONLR_S%TMP_GPU)
    return
END SUBROUTINE DESTROY_NONLR_GPU

!****************** subroutine RACC0MU_GPU ***************************
!
! this subroutine is a GPU version of RACC0MU, it parallelizes over
! inner dimensions and NSIM, and batches/streams over ions with
! the BATCH_SIZE parameter
!
!*********************************************************************
SUBROUTINE RACC0MU_GPU(NONLR_S, WDES1, CPROJ_LOC, GPU_CRACC_ALL, NSIM, LDO, BATCH_SIZE)
    USE iso_c_binding
    USE cuda_interface
    USE nonlr_struct_def
    USE wave
    IMPLICIT NONE

    ! input parameters
    TYPE (nonlr_struct), TARGET :: NONLR_S
    TYPE (wavedes1) WDES1
    GDEF CPROJ_LOC(WDES1%NPROD,NSIM) ! wave function character
    INTEGER(c_intptr_t) GPU_CRACC_ALL ! device pointer to CRACC
    INTEGER NSIM ! do NSIM bands at a time
    LOGICAL LDO(NSIM) ! which bands are included
    INTEGER :: BATCH_SIZE ! size of batches (relevant for GEMMs)
    ! local variables
    INTEGER, PARAMETER :: NLM=101
    REAL(qn),PARAMETER :: ONE=1,ZERO=0
    INTEGER :: NP,LMBASE,ISPIRAL,ISPINOR,NLIIND,NIS,NT,LMMAXC,NI,&
               INDMAX,IND0,NPFILL,L,IND,IP,NB,I,IRMAX
    INTEGER :: ION_BLOCK,CMPLX_FLAG,LDO_SIZE,PARTIAL_FLAG
    INTEGER :: shiftCRACC,shiftWORK,shiftTMP,shiftCRREXP,shiftNLI
    GDEF :: sizegdef
    INTEGER :: sizei

nv_profile_start(NVP_RACC0MU)
    IRMAX=NONLR_S%IRMAX
    IF (WDES1%NK /= NONLR_S%NK) THEN
      WRITE(*,*) 'internal error in RACC0MU: PHASE not properly set up',WDES1%NK, NONLR_S%NK
      STOP
    ENDIF
nv_profile_start(NVP_RACC0MU_MERGE)
   ! merge projected wavefunctions from all nodes
   ! store wavefunction index NP in LDO_INDEX, this is used
   ! by CUDA kernels in case some of the wavefunctions are not
   ! operational, that is, LDO=.FALSE.
    NONLR_S%LDO_INDEX=0
    NONLR_S%CPROJ=0
    NPFILL=0
    DO NP=1,NSIM
      IF (LDO(NP)) THEN
        NPFILL=NPFILL+1
        CALL MRG_PROJ(WDES1, NONLR_S%CPROJ(1,NP),CPROJ_LOC(1,NP))
         NONLR_S%LDO_INDEX(NPFILL)=NP-1 ! C indexing...
      ENDIF
    ENDDO
    LDO_SIZE=NPFILL
    ! flag is set when not all wavefunctions are operational, that is,
    ! only some wavefunctions are used in computations
    IF(LDO_SIZE==NSIM) THEN
        PARTIAL_FLAG=0
    ELSE
        PARTIAL_FLAG=1
    ENDIF
nv_profile_stop(NVP_RACC0MU_MERGE)

nv_profile_start(NVP_RACC0MU_MEMCPY)
    ! copy from host to device
    CALL cublas_Set_Matrix(WDES1%NPRO_TOT,NSIM,int(c_sizeof(sizegdef),c_int), c_loc(NONLR_S%CPROJ),WDES1%NPRO_TOT,NONLR_S%CPROJ_GPU,WDES1%NPRO_TOT)
    CALL cublas_Set_Vector(LDO_SIZE,int(c_sizeof(sizei),c_int), c_loc(NONLR_S%LDO_INDEX),1,NONLR_S%LDO_INDEX_GPU,1)
nv_profile_stop(NVP_RACC0MU_MEMCPY)

#ifndef gammareal
    CMPLX_FLAG=1
#else
    CMPLX_FLAG=0
#endif

!=======================================================================
! loop over ions
!=======================================================================
nv_profile_start(NVP_RACC0MU_COMPUTE)
nv_profile_start(NVP_GEMM)
    LMBASE=0
    ISPIRAL=1
    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
      NLIIND=0
      NIS=1

      ! loop over all atom types
      typ: DO NT=1,NONLR_S%NTYP
        LMMAXC=NONLR_S%LMMAX(NT)
        IF (LMMAXC==0) GOTO 600

        ! loop over all ions of each type
        ion: DO NB=NIS,NONLR_S%NITYP(NT)+NIS-1,BATCH_SIZE
          ! determine size of ion batches
          ION_BLOCK=BATCH_SIZE
          IF((NIS+NONLR_S%NITYP(NT)-1)<(NB+BATCH_SIZE-1)) THEN
             ION_BLOCK=NIS+NONLR_S%NITYP(NT)-NB
          ENDIF

!=======================================================================
! set TMP
!=======================================================================
          ! for a batch of ions,
          call cuda_splitcproj(ION_BLOCK,LDO_SIZE,LMMAXC,&
               NONLR_S%TMP_GPU,NLM,ndata*NSIM*NLM,&
               NONLR_S%CPROJ_GPU,LMBASE,WDES1%NPRO_TOT,&
               WDES1%RINPL,ndata,NONLR_S%LDO_INDEX_GPU,CMPLX_FLAG,PARTIAL_FLAG)
          call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RACC0MU_GPU!"//C_NULL_CHAR)

!=======================================================================
! calculate SUM(LM=1,NONLR_S%LMMAX) NONLR_S%RPROJ(K,LM) * TMP(LM)
! add the non local contribution to the accelerations
!=======================================================================
          shiftCRACC=ISPINOR*WDES1%GRID%MPLWV
          shiftWORK=ndata*NSIM*IRMAX
          shiftCRREXP=(ISPIRAL-1)*NONLR_S%NIONS*IRMAX+(NB-1)*IRMAX
          shiftNLI=(NB-1)*IRMAX
          ! for a batch of ions,
          DO NI=NB,MIN(NONLR_S%NITYP(NT)+NIS-1,NB+BATCH_SIZE-1)
              I=NI-NB
              ! for each ion,
              ! stream GEMMs
              call cublas_dgemmsh_st(I,'N','N',NONLR_S%NLIMAX(NI),ndata*LDO_SIZE,LMMAXC,ONE,&
                   NONLR_S%RPROJ_GPU, NLIIND, NONLR_S%NLIMAX(NI),&
                   NONLR_S%TMP_GPU, I*NLM*ndata*NSIM, NLM, ZERO,&
                   NONLR_S%WORK_GPU, I*ndata*NSIM*NONLR_S%IRMAX, IRMAX)
              ! stream CRACC
              IF(ASSOCIATED(NONLR_S%CRREXP)) THEN
              call cuda_calccracc_crrexp(I,LDO_SIZE,IRMAX,NONLR_S%NLIMAX(NI),&
                   GPU_CRACC_ALL, shiftCRACC, WDES1%NRSPINORS*WDES1%GRID%MPLWV,&
                   NONLR_S%WORK_GPU, I*shiftWORK, IRMAX,&
                   NONLR_S%CRREXP_GPU, shiftCRREXP+I*IRMAX,&
                   ndata, &
                   NONLR_S%NLI_GPU, &
                   shiftNLI+I*IRMAX,&
                   NONLR_S%LDO_INDEX_GPU, PARTIAL_FLAG)
              ELSE
              call cuda_calccracc(I,LDO_SIZE,IRMAX,NONLR_S%NLIMAX(NI),&
                   GPU_CRACC_ALL, shiftCRACC, WDES1%NRSPINORS*WDES1%GRID%MPLWV,&
                   NONLR_S%WORK_GPU, I*shiftWORK, IRMAX,&
                   ndata,NONLR_S%NLI_GPU, shiftNLI+I*IRMAX,&
                   NONLR_S%LDO_INDEX_GPU, PARTIAL_FLAG)
              ENDIF
              ! stride between matrices stored in RPROJ for block of ions
              NLIIND=NLIIND+LMMAXC*NONLR_S%NLIMAX(NI)
          ENDDO
          call cuda_devicesynchronize(C_CHAR_"cudaDeviceSynchronize failed in RACC0MU_GPU!"//C_NULL_CHAR)
          ! stride between lmn-vectors for block of ions
100       LMBASE=LMBASE+LMMAXC*ION_BLOCK
        ENDDO ion
600     NIS=NIS+NONLR_S%NITYP(NT)
      ENDDO typ
      IF (NONLR_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_RACC0MU_COMPUTE)

nv_profile_stop(NVP_RACC0MU)
    RETURN
  END SUBROUTINE RACC0MU_GPU

!***********************************************************************
!
! small f77 helper  routines to 
! multiply with phasefactor and divide into real and imaginary part
!
!***********************************************************************
  
  SUBROUTINE CRREXP_MUL_WAVE( INDMAX, CRREXP, NLI, CR, WORK1, WORK2)
    USE prec
    IMPLICIT NONE
    INTEGER INDMAX
    COMPLEX(q) :: CRREXP(INDMAX)
    INTEGER    :: NLI(INDMAX)
    COMPLEX(q) :: CR(*)
    REAL(qn)   :: WORK1(INDMAX), WORK2(INDMAX)
    COMPLEX(qn):: CTMP
  ! local
    INTEGER IND, IP

!DIR$ IVDEP
!OCL NOVREC
    DO IND=1,INDMAX
       IP=NLI(IND)
       CTMP=    CR(IP)*CRREXP(IND)
       WORK1(IND) = REAL( CTMP ,KIND=q)
       WORK2(IND)=  AIMAG(CTMP)
    ENDDO
  END SUBROUTINE CRREXP_MUL_WAVE


  SUBROUTINE CRREXP_MUL_GWAVE( INDMAX, CRREXP, NLI, CR, WORK1, WORK2)
    USE prec
    IMPLICIT NONE
    INTEGER INDMAX
    COMPLEX(q) :: CRREXP(INDMAX)
    INTEGER    :: NLI(INDMAX)
    GDEF       :: CR(*)
    REAL(qn)   :: WORK1(INDMAX), WORK2(INDMAX)
    COMPLEX(qn):: CTMP
  ! local
    INTEGER IND, IP

!DIR$ IVDEP
!OCL NOVREC
    DO IND=1,INDMAX
       IP=NLI(IND)
       CTMP=    CR(IP)*CRREXP(IND)
       WORK1(IND) = REAL( CTMP ,KIND=q)
       WORK2(IND)=  AIMAG(CTMP)
    ENDDO
  END SUBROUTINE CRREXP_MUL_GWAVE


  SUBROUTINE CRREXP_MUL_WORK_ADD( INDMAX, CRREXP, NLI, WORK1, WORK2, CR)
    USE prec
    IMPLICIT NONE
    INTEGER INDMAX
    COMPLEX(q) :: CRREXP(INDMAX)
    INTEGER    :: NLI(INDMAX)
    REAL(qn)   :: WORK1(INDMAX), WORK2(INDMAX)
    COMPLEX(q) :: CR(*)
! local
    COMPLEX(q) :: CTMP
    GDEFN :: CTMPN
    INTEGER IND,IP
!DIR$ IVDEP
!OCL NOVREC
    DO IND=1,INDMAX
       IP=NLI(IND)
       CTMPN=GCMPLX( WORK1(IND), WORK2(IND), q)
       CTMP =CONJG(CRREXP(IND))
       CR(IP)=CR(IP)+CTMPN*CTMP
    ENDDO
  END SUBROUTINE CRREXP_MUL_WORK_ADD


  SUBROUTINE CRREXP_MUL_WORK_GADD( INDMAX, CRREXP, NLI, WORK1, WORK2, CR)
    USE prec
    IMPLICIT NONE
    INTEGER INDMAX
    COMPLEX(q) :: CRREXP(INDMAX)
    INTEGER    :: NLI(INDMAX)
    REAL(qn)   :: WORK1(INDMAX), WORK2(INDMAX)
    GDEF       :: CR(*)
! local
    COMPLEX(q) :: CTMP
    GDEFN :: CTMPN
    INTEGER IND,IP
!DIR$ IVDEP
!OCL NOVREC
    DO IND=1,INDMAX
       IP=NLI(IND)
       CTMPN=GCMPLX( WORK1(IND), WORK2(IND), q)
       CTMP =CONJG(CRREXP(IND))
       CR(IP)=CR(IP)+CTMPN*CTMP
    ENDDO
  END SUBROUTINE CRREXP_MUL_WORK_GADD
