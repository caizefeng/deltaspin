!#define debug
#include "symbol.inc"
!************************************************************************
! RCS:  $Id: wave.F,v 1.6 2002/08/14 13:59:43 kresse Exp $
!
!  this module contains the routines required to setup
!  the distribution of wavefunctions over nodes and all basic routines
!  handling wavedes etc.
!
!***********************************************************************
MODULE WAVE
  USE prec
  USE mpimy
  USE mgrid

  ! the flag determines whether the serial of parallel FFT is used for
  ! wavefunctions, if one node holds all data for an entire wavefunction
  ! routines like HF require LUSE_PARALLEL_FFT=.FALSE., hence
  ! I recommend to use that option
  ! It is usually faster anyway
  LOGICAL :: LUSE_PARALLEL_FFT=.FALSE.

  !
  !  structure required to support storage of wavefunction for
  !  several kpoints and spins
  !
  TYPE wavedes
     REAL(q)    RSPIN              ! spin multiplicity
     REAL(q)    ENMAX              ! energy cutoff
     INTEGER NRSPINORS             ! number of spinors (1 for collinear, 2 for non collinear)
     INTEGER NGDIM                 ! first dimension of any array related to the plane wave basis
     INTEGER NRPLWV                ! first dimension of wavefunction array
     ! collinear:  NRPLWV=NGDIM, noncollinear:  NRPLWV=2*NGDIM
     INTEGER NRPLWV_RED            ! local number of coefficients in wave function array after data redistribution
     INTEGER NPROD                 ! first dimension of projected wave array
     INTEGER NPRO                  ! local number of elements in projected wave array
     INTEGER NPRO_TOT              ! total number of elements (summed over all nodes)
     ! NPRO, NPROD, and NPRO_TOT are all doubled in the non collinear version
     INTEGER NPROD_RED             ! dimension of projected wave array after redistribution
     INTEGER NBANDS                ! local number of bands
     INTEGER NB_TOT                ! total number bands
     INTEGER NB_PAR                ! distribution over bands (number of bands done in parallel )= WDES%COMM_INTER%NCPU
     INTEGER NSIM                  ! band blocking (mainly for seriel version)
     INTEGER NB_LOW                ! lowest band index in global
     INTEGER NKDIM                 ! total number of k-points in the entire Brillouin zone (BZ)
     ! required for HF calculations (otherwise equal to NKPTS)
     INTEGER NKPTS_FOR_GEN_LAYOUT  ! number of k-points used for the generation of the data layout
     ! this must not change when the number of k-point changes
     INTEGER NKPTS                 ! number of k-points in the irreducable wedge of the BZ (IBZ)
     INTEGER ISPIN                 ! number of spins
     INTEGER NCDIJ                 ! dimension of arrays like CDIJ, CQIJ
     INTEGER NIONS                 ! number of ions stored locally 
     INTEGER NTYP                  ! number of types stored locally
     TYPE (grid_3d), POINTER ::GRID! pointer to a grid if FFT's are required
     INTEGER,POINTER :: NPLWKP(:)  ! number of coefficients for each k-point and band per node
     INTEGER,POINTER :: NGVECTOR(:)! number of G-vectors in the basis for each k-point per node
     ! collinear: NPLWKP= NGVECTOR, noncollinear NPLWKP = 2*NGVECTOR 
     ! NGVECTOR is the same for collinear and non collinear calculations
     ! (summed over nodes, doubled in the non collinear case)
     INTEGER,POINTER :: NGVECTOR_POS(:)! sum of NGVECTOR up to (but not including) the current node
     INTEGER,POINTER :: NPLWKP_TOT(:)  ! total number of coefficients in plane wave array at each k-points
     INTEGER,POINTER :: NB_TOTK(:,:)! number of bands to be calculated for each k-point and spin
     ! possibly smaller than NB_TOT
     INTEGER         :: NCOL       ! number of columns
     INTEGER,POINTER,CONTIGUOUS :: PL_INDEX(:,:) ! index a column would have in serial version
     INTEGER,POINTER,CONTIGUOUS :: PL_COL(:,:)! number of plane wave in this column
     INTEGER,POINTER ::NPRO_POS(:) ! for each atom, start index of entries in CPROJ in serial version
     INTEGER,POINTER :: LMMAX(:)   ! total number of NLM quantum numbers for each type
     INTEGER,POINTER :: LMBASE(:)  !
     INTEGER,POINTER :: NITYP(:)   ! number of ions stored locally for each type
     INTEGER,POINTER :: ITYP(:)    ! type for each ion
     INTEGER,POINTER ::NT_GLOBAL(:)! global type index for this type 
     REAL(q),POINTER :: VKPT(:,:)  ! coordinate of k-point
     REAL(q),POINTER :: WTKPT(:)   ! symmetry weight-factor for each k-point
     INTEGER,POINTER,CONTIGUOUS :: NINDPW(:,:)! index to the FFT box for each pw comp and k-point
     INTEGER,POINTER,CONTIGUOUS :: IGX(:,:)   ! x index of each pw comp and k-point
     INTEGER,POINTER,CONTIGUOUS :: IGY(:,:)   ! y index of each pw comp and k-point
     INTEGER,POINTER,CONTIGUOUS :: IGZ(:,:)   ! z index of each pw comp and k-point
     REAL(q),POINTER,CONTIGUOUS :: DATAKE(:,:,:) ! kinetic energy for each plane wave
     ! last index labels up and down components
     ! of the spinor in case of spin spirals
     REAL(q) QSPIRAL(3)            ! propagation vector of spin spiral
     TYPE(communic),POINTER  :: COMM,COMM_INTER,COMM_INB
     TYPE(communic),POINTER  :: COMM_KINTER,COMM_KIN
     TYPE(communic),POINTER  :: COMM_SHMEM,COMM_intra_node,COMM_inter_node
     REAL(q) SAXIS(3)              ! quantisation axis of the spin operator
     LOGICAL,POINTER :: AT_GAMMA(:)! indicates that a k-point corresponds to gamma
                                   ! selects special treatment
     LOGICAL LORBITALREAL          ! special treatment at gamma
     LOGICAL LOVERL                ! overlap required
     LOGICAL DO_REDIS              ! data redistribution required
     LOGICAL LNONCOLLINEAR         ! noncollinear calculations
     LOGICAL LSORBIT               ! spin orbit coupling
     LOGICAL LGAMMA                ! gamma point only, projected wavefunction character is REAL
                                   ! this is only .TRUE. if precompiler flag gammareal is define 
     LOGICAL LSPIRAL               ! calculate spin spirals?
     LOGICAL LZEROZ                ! set m_z to zero in SET_CHARGE?
  END TYPE wavedes

  !
  ! description for one k point
  ! contains also all information required for simple calculations in real space
  !
  TYPE wavedes1
     !only WDES1
     REAL(q)    RSPIN              ! spin multiplicity
     INTEGER NRSPINORS             ! number of spinors (1 for collinear, 2 for non collinear)
     INTEGER NGDIM                 ! first dimension of any array related to the basis
     INTEGER NRPLWV                ! first dimension of wavefunction array (stores coefficients in rec. space)
     ! collinear:  NRPLWV=NGDIM, noncollinear:  NRPLWV=2*NGDIM
     INTEGER NRPLWV_RED            ! local number of electrons in wavefunctionarray after data redistribution
     INTEGER NPROD                 ! first dimension of projected wave array
     INTEGER NPRO                  ! local number of elements in projected wave array
     INTEGER NPRO_O_RED            ! local number of elements for overlap (0 for normconserving pot) after resdistribution
     INTEGER NPRO_TOT              ! total number of elements
     INTEGER NPROD_RED             ! local number of elements in projected wave array after redistribution
     ! NPRO, NPROD, and NPRO_TOT are all doubled in the non collinear version
     INTEGER NPRO_RED              ! local number of elements in projected wave array after redistribution
     INTEGER NBANDS                ! bands
     INTEGER NB_TOT                ! total number bands
     INTEGER NB_PAR                ! distribution over bands (number of bands done in parallel )= WDES%COMM_INTER%NCPU
     INTEGER NSIM                  ! band blocking (mainly for serial version)
     INTEGER NB_LOW                ! lowest band index in global 
     INTEGER NPL                   ! number of plane waves coefficients (local)
     INTEGER NPL_RED               ! number of plane waves coefficients after data redistribution
     INTEGER NGVECTOR              ! number of G-vectors in the basis (local)
     ! collinear: NGVECTOR == NPL, noncollinear 2*NGVECTOR == NPL
     INTEGER NGVECTOR_POS          ! sum of NGVECTOR up to (but not including) the current node
     INTEGER NPL_TOT               ! total number of plane waves (global)
     INTEGER NB_TOTK(2)            ! number of bands to be calculated
     INTEGER NIONS                 ! number of ions stored locally
     INTEGER NTYP                  ! number of types stored locally
     REAL(q) RINPL                 ! inverse of total number of plane waves
     TYPE (grid_3d), POINTER ::GRID! pointer to a grid if FFT's are required
     INTEGER NK                    ! k-point number (required for non-local proj.)
     ! few things which are only required in parallel version
     INTEGER         :: NCOL       ! number of columns
     INTEGER,POINTER,CONTIGUOUS :: PL_INDEX(:)! index a column would have in serial version
     INTEGER,POINTER,CONTIGUOUS :: PL_COL(:)  ! number of plane waves in this column
     INTEGER,POINTER :: NPRO_POS(:)! index CPROJ would have in serial version
     INTEGER,POINTER :: LMMAX(:)   ! total number of NLM quantum numbers for each type
     INTEGER,POINTER :: LMBASE(:)  !
     INTEGER,POINTER :: NITYP(:)   ! number of ions stored locally for each type
     INTEGER,POINTER :: ITYP(:)    ! type for each ion
     INTEGER,POINTER ::NT_GLOBAL(:)! global type index for this type 
     INTEGER,POINTER,CONTIGUOUS :: NINDPW(:)  ! index to the FFT box for each pw comp
     INTEGER,POINTER,CONTIGUOUS :: IGX(:)     ! x index of each pw comp and k-point
     INTEGER,POINTER,CONTIGUOUS :: IGY(:)     ! y index of each pw comp and k-point
     INTEGER,POINTER,CONTIGUOUS :: IGZ(:)     ! z index of each pw comp and k-point
     REAL(q),POINTER :: VKPT(:)    ! k-point
     REAL(q)         :: WTKPT      ! symmetry weight-factor for this k-point
     REAL(q),POINTER,CONTIGUOUS :: DATAKE(:,:)! kinetic energy for each plane wave
     ! last index labels up and down components
     ! of the spinor in case of spin spirals
     REAL(q) QSPIRAL(3)            ! propagation vector of spin spiral
     TYPE(communic),POINTER  :: COMM
     TYPE(communic),POINTER  :: COMM_INTER,COMM_INB
     TYPE(communic),POINTER  :: COMM_KINTER,COMM_KIN
     TYPE(communic),POINTER  :: COMM_SHMEM,COMM_intra_node,COMM_inter_node
     REAL(q) SAXIS(3)              ! quantisation axis of the spin operator
     LOGICAL AT_GAMMA              ! indicates that a k-point corresponds to gamma
                                   ! selects special treatment
     LOGICAL LORBITALREAL          ! special treatment at gamma
     LOGICAL LOVERL                ! overlap required
     LOGICAL DO_REDIS              ! data redistribution required
     LOGICAL LNONCOLLINEAR         ! allows one to turn on noncollinear calculations
     LOGICAL LSORBIT               ! spin orbit coupling
     LOGICAL LGAMMA                ! gamma point only, projected wavefunction character is REAL
     LOGICAL LSPIRAL               ! calculate spin spirals?
     LOGICAL LZEROZ                ! set m_z to zero in SET_CHARGE?
  END TYPE wavedes1

  !
  ! structure required to store a set of orbitals including band index, k-point and spin
  ! in most places in VASP structures of this type are defined as TYPE(wavespin) :: W
  !
  TYPE wavespin
     TYPE(wavedes),POINTER:: WDES         ! descriptor for orbital
!#ifndef PGI16
!     COMPLEX(q),POINTER,CONTIGUOUS:: CPTWFP(:,:,:,:) ! plane wave coefficients of orbital, (orbital) C_G,n,k,spin
!                                   ! indices: plane wave, local band index, k-point, spin
!     GDEF      ,POINTER,CONTIGUOUS:: CPROJ(:,:,:,:)  ! orbital character <p_alpha| phi_n,k,spin>
!                                   ! indices: projectors, local band index, k-point, spin
!#else
     COMPLEX(q),POINTER           :: CPTWFP(:,:,:,:)
     GDEF      ,POINTER           :: CPROJ(:,:,:,:)
!#endif
     REAL(q),   POINTER:: FERWE(:,:,:)    ! fermi-weight for each band stored on local node
                                   ! local band index, k-point, spin
     REAL(q),   POINTER:: AUX  (:,:,:)    ! auxilary
                                   ! local band index, k-point, spin
     COMPLEX(q),POINTER:: CELEN(:,:,:)    ! eigenvalues
                                   ! local band index, k-point, spin
     REAL(q),   POINTER:: FERTOT(:,:,:)   ! global array for fermi-weights
                                   ! global band index, k-point, spin
     REAL(q),   POINTER:: AUXTOT(:,:,:)   ! global array for auxilary quantities
                                   ! global band index, k-point, spin
     COMPLEX(q),POINTER:: CELTOT(:,:,:)   ! global array for eigenvalues
                                   ! global band index, k-point, spin
     LOGICAL           :: OVER_BAND       ! distribution over bands or not
  END TYPE wavespin

  !
  ! structure required to store a set of orbitals including and band index, k-point
  ! this type is hardly used in VASP, but was required during early porting to F90
  !
  TYPE wavefun
     TYPE(wavedes),POINTER:: WDES       ! descriptor for orbital
#ifndef PGI16
     COMPLEX(q),POINTER,CONTIGUOUS:: CPTWFP(:,:,:) ! plane wave coefficients of orbital (orbital) C_G,n,k
                                   ! indices: plane wave, local band index, k-point
     GDEF      ,POINTER,CONTIGUOUS:: CPROJ(:,:,:)  ! orbital character <p_alpha| phi_n,k>
                                   ! indices: projectors, local band index, k-point
     COMPLEX(q),POINTER,CONTIGUOUS:: CR(:,:,:)     ! orbital in real space
                                   ! hardly ever allocated
#else
     COMPLEX(q),POINTER           :: CPTWFP(:,:,:)
     GDEF      ,POINTER           :: CPROJ(:,:,:)
     COMPLEX(q),POINTER           :: CR(:,:,:)
#endif
     REAL(q),   POINTER:: FERWE(:,:)    ! fermi-weight for each band
                                   ! local band index, k-point
     REAL(q),   POINTER:: AUX  (:,:)    ! auxilary
                                   ! local band index, k-point
     COMPLEX(q),POINTER:: CELEN(:,:)    ! eigenvalues
                                   ! local band index, k-point
     REAL(q),   POINTER:: FERTOT(:,:)   ! global array for fermi-weights
                                   ! global band index, k-point
     REAL(q),   POINTER:: AUXTOT(:,:)   ! global array for auxilary quantities
                                   ! global band index, k-point
     COMPLEX(q),POINTER:: CELTOT(:,:)   ! global array for eigenvalues
                                   ! global band index, k-point
     LOGICAL, POINTER  :: LOPT(:,:,:)   ! optimize this orbital
     LOGICAL           :: OVER_BAND     ! distribution over bands or not
  END TYPE wavefun

  !
  ! structure required to store one orbital (no spin no bands)
  !
  TYPE wavefun1
     TYPE(wavedes1),POINTER:: WDES1   ! descriptor for orbital
#ifndef PGI16
     COMPLEX(q),POINTER,CONTIGUOUS:: CPTWFP(:)   ! plane wave coefficients of a single orbital 
     GDEF      ,POINTER,CONTIGUOUS:: CPROJ(:)    ! orbital character <p_alpha| phi>
     COMPLEX(q),POINTER,CONTIGUOUS:: CR(:)       ! orbital in real space
#else
     COMPLEX(q),POINTER           :: CPTWFP(:)
     GDEF      ,POINTER           :: CPROJ(:)
     COMPLEX(q),POINTER           :: CR(:)
#endif
     REAL(q)           :: FERWE       ! fermi-weight for each band
     REAL(q)           :: AUX         ! auxilary
     COMPLEX(q)        :: CELEN       ! eigenvalues
     INTEGER           :: NB          ! band index if it applies (not used)
     INTEGER           :: ISP         ! spin index if it applies (not used)
     LOGICAL           :: LDO         ! initialised and operational
  END TYPE wavefun1
  !
  ! structure required to store a set of orbitals (no spin and k dependency)
  !
  TYPE wavefuna
     TYPE(wavedes1),POINTER:: WDES1   ! descriptor for orbital
#ifndef PGI16
     COMPLEX(q),POINTER,CONTIGUOUS:: CPTWFP(:,:) ! plane wave coefficients of a set of orbitals
     GDEF      ,POINTER,CONTIGUOUS:: CPROJ(:,:)  ! orbital character <p_alpha| phi>
#else
     COMPLEX(q),POINTER           :: CPTWFP(:,:)
     GDEF      ,POINTER           :: CPROJ(:,:)
#endif
     COMPLEX(q),POINTER,CONTIGUOUS:: CW_RED(:,:) ! redistributed orbitals
     GDEF      ,POINTER,CONTIGUOUS:: CPROJ_RED(:,:)! orbital character redistributed
     REAL(q),   POINTER:: FERWE(:)    ! fermi-weight for each band
     REAL(q),   POINTER:: AUX  (:)    ! auxilary
     COMPLEX(q),POINTER:: CELEN(:)    ! eigenvalues
     INTEGER           :: ISP         ! spin index if apply (not used)
     INTEGER           :: FIRST_DIM   ! first dimension, if acessed as a multidimensional array
     LOGICAL           :: OVER_BAND   ! distribution over bands or not (not used)
  END TYPE wavefuna

CONTAINS

!=======================================================================
!
!  allocate the descriptor for the orbitals
!  if LEXTEND is set the kinetic energy array and
!  the index arrays IGX, Y and Z are allocated as well
!
!=======================================================================

  SUBROUTINE ALLOCWDES(WDES,LEXTEND)
    IMPLICIT NONE

    INTEGER NK
    TYPE (wavedes)  WDES
    INTEGER NRPLWV,NKPTS,NCOL,NKDIM
    LOGICAL LEXTEND

    NRPLWV=WDES%NGDIM
    NKPTS =WDES%NKPTS
    NKDIM =WDES%NKDIM   ! total number of k-points in the 1st BZ (for HF)
    NCOL  =WDES%NCOL    
    ! a few arrays are allocated with the size NKDIM
    ALLOCATE( WDES%NPLWKP(NKDIM),WDES%NGVECTOR(NKDIM),WDES%NGVECTOR_POS(NKDIM),WDES%NPLWKP_TOT(NKDIM), & 
         WDES%NINDPW(NRPLWV,NKDIM),WDES%NB_TOTK(NKDIM,2),WDES%AT_GAMMA(NKDIM))
    WDES%NPLWKP    =0
    WDES%NPLWKP_TOT=0
    WDES%NGVECTOR  =0
    WDES%NGVECTOR_POS=0
    IF (NCOL>0) THEN
       ALLOCATE(WDES%PL_INDEX(NCOL,NKPTS),WDES%PL_COL(NCOL,NKPTS))
    ELSE
       NULLIFY(WDES%PL_INDEX); NULLIFY(WDES%PL_COL)
    ENDIF

    IF (LEXTEND) THEN
       ALLOCATE(WDES%DATAKE(NRPLWV,2,NKPTS), &
            &    WDES%IGX(NRPLWV,NKDIM),WDES%IGY(NRPLWV,NKDIM),WDES%IGZ(NRPLWV,NKDIM))
    ELSE
       NULLIFY(WDES%DATAKE)
       NULLIFY(WDES%IGX); NULLIFY(WDES%IGY); NULLIFY(WDES%IGZ)
       NULLIFY(WDES%PL_INDEX); NULLIFY(WDES%PL_COL)
    END IF

  END SUBROUTINE ALLOCWDES

!=======================================================================
!  deallocate a  descriptor for the orbitals
!=======================================================================

  SUBROUTINE DEALLOCWDES(WDES,LEXTEND)
    IMPLICIT NONE
    TYPE (wavedes)  WDES
    LOGICAL LEXTEND

    DEALLOCATE( WDES%NPLWKP,WDES%NGVECTOR,WDES%NGVECTOR_POS,WDES%NPLWKP_TOT, &
         WDES%NINDPW, WDES%NB_TOTK, WDES%AT_GAMMA)

    NULLIFY( WDES%NPLWKP,WDES%NGVECTOR,WDES%NGVECTOR_POS,WDES%NPLWKP_TOT, &
         WDES%NINDPW, WDES%NB_TOTK, WDES%AT_GAMMA)

    IF (WDES%NCOL>0) THEN
       DEALLOCATE(WDES%PL_INDEX,WDES%PL_COL)
       NULLIFY(WDES%PL_INDEX); NULLIFY(WDES%PL_COL)
    ENDIF
    IF (LEXTEND) THEN
       DEALLOCATE( WDES%DATAKE,WDES%IGX,WDES%IGY,WDES%IGZ)
       NULLIFY(WDES%DATAKE, WDES%IGX, WDES%IGY, WDES%IGZ)
    END IF
  END SUBROUTINE DEALLOCWDES


!=======================================================================
!  free a descriptor and overwrite it by a new one
!=======================================================================

  SUBROUTINE COPYWDES(WDES_OLD, WDES_NEW ,LEXTEND)
    IMPLICIT NONE
    TYPE (wavedes)  WDES_OLD, WDES_NEW
    LOGICAL LEXTEND

    CALL DEALLOCWDES(WDES_OLD,LEXTEND)
    WDES_OLD=WDES_NEW

  END SUBROUTINE COPYWDES

!=======================================================================
!  initialize k-point related quantities
!=======================================================================

  SUBROUTINE INIT_KPOINT_WDES(WDES, KPOINTS )
    USE mkpoints
    IMPLICIT NONE
    TYPE (wavedes)  WDES
    TYPE (kpoints_struct) KPOINTS
    INTEGER :: NK

    WDES%NKDIM =KPOINTS%NKPTS
    WDES%NKPTS =KPOINTS%NKPTS
    WDES%NKPTS_FOR_GEN_LAYOUT =KPOINTS%NKPTS
    WDES%VKPT  =>KPOINTS%VKPT
    WDES%WTKPT =>KPOINTS%WTKPT

  END SUBROUTINE INIT_KPOINT_WDES

!=======================================================================
!  initialize the projector part of the descriptor for the
!  orbitals
!=======================================================================

  SUBROUTINE WDES_SET_NPRO(WDES,T_INFO,P,LOVERL)
    USE  poscar
    USE  pseudo

    TYPE (wavedes)  WDES
    TYPE (type_info) :: T_INFO
    TYPE (potcar)   P(T_INFO%NTYP)
    LOGICAL :: LOVERL
    ! local varibles
    INTEGER NALLOC,NPRO_TOT,NT,NI,NIS,NODE_TARGET,NPRO, &
         LMMAXC,NIONS,LASTTYP
#ifndef MPI
    WDES%NIONS = T_INFO%NIONS
    WDES%NTYP  = T_INFO%NTYP
    WDES%NITYP =>T_INFO%NITYP
    WDES%ITYP  =>T_INFO%ITYP

    ALLOCATE(WDES%LMMAX(WDES%NTYP),WDES%NT_GLOBAL(WDES%NTYP))
    DO NT=1,T_INFO%NTYP
       WDES%LMMAX(NT)=P(NT)%LMMAX
       WDES%NT_GLOBAL(NT)=NT
    ENDDO
    WDES%NPRO  =SUM(WDES%LMMAX*WDES%NITYP)
    WDES%NPRO_TOT=SUM(WDES%LMMAX*WDES%NITYP)
    WDES%NPROD =WDES%NPRO
    NULLIFY(WDES%NPRO_POS)
#else
    !-----------------------------------------------------------------------
    ! parallel version
    !-----------------------------------------------------------------------
    TYPE (communic) COMM
    ! first count number of projection operators
    COMM = WDES%COMM_INB
    NPRO_TOT=0
    DO NT=1,T_INFO%NTYP
       NPRO_TOT=NPRO_TOT+P(NT)%LMMAX*T_INFO%NITYP(NT)
    ENDDO
    ! check implementation of NI_LOCAL and NI_GLOBAL
    DO NI=1,T_INFO%NIONS
       NI_L=NI_LOCAL(NI,COMM)
       IF (NI_L /= 0) THEN
          IF (NI /= NI_GLOBAL(NI_L,COMM)) THEN
             WRITE(*,*)' internal error NI_GLOBAL, NI_LOCAL'
             STOP
          ENDIF
       ENDIF
    ENDDO
    ! number of projection operators per node
    NALLOC=(T_INFO%NIONS+COMM%NCPU-1)/COMM%NCPU

    WDES%NIONS=0
    WDES%NTYP =0
    WDES%NPRO =0
    WDES%NPRO_TOT=NPRO_TOT
    ALLOCATE(WDES%NITYP(T_INFO%NTYP))
    ALLOCATE(WDES%ITYP(T_INFO%NIONS))
    ALLOCATE(WDES%LMMAX(T_INFO%NTYP),WDES%NT_GLOBAL(T_INFO%NTYP))
    ALLOCATE(WDES%NPRO_POS(NALLOC))
    WDES%LMMAX=0
    WDES%NITYP=0
    WDES%ITYP=0
    WDES%NT_GLOBAL=0

    LASTTYP=0
    NIS    =1

    NPRO=0
    DO NT=1,T_INFO%NTYP
       LMMAXC=P(NT)%LMMAX
       IF (LMMAXC==0) GOTO 100
       DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
          ! does this element reside on local node
          IF (NI_LOCAL(NI,COMM) /=0 ) THEN
             WDES%NIONS=WDES%NIONS+1
             WDES%NPRO_POS(WDES%NIONS)=NPRO
             WDES%NPRO =WDES%NPRO +LMMAXC
             IF (NT /= LASTTYP) THEN
                WDES%NTYP=WDES%NTYP+1
                LASTTYP=NT
             ENDIF
             WDES%LMMAX(WDES%NTYP)=LMMAXC
             WDES%NT_GLOBAL(WDES%NTYP)=NT
             WDES%NITYP(WDES%NTYP)=WDES%NITYP(WDES%NTYP)+1
             WDES%ITYP(WDES%NIONS)=WDES%NTYP
          ENDIF
          NPRO= LMMAXC+NPRO
       ENDDO

100    NIS = NIS+T_INFO%NITYP(NT)
    ENDDO
    ! check whether everything is right
    NPRO=SUM(WDES%LMMAX*WDES%NITYP)
    IF (NPRO/= WDES%NPRO) THEN
       WRITE(*,*)'internal ERROR 1 in WDES_SET_NPRO:',NPRO,WDES%NPRO
       STOP
    ENDIF
    CALLMPI( M_sum_i(COMM,NPRO ,1))
    IF (NPRO/= NPRO_TOT) THEN
       WRITE(*,*)'internal ERROR 2 in WDES_SET_NPRO:',NPRO,NPRO_TOT
       STOP
    ENDIF
    ! make NPROD dividable by number of NB_PAR
    WDES%NPROD=((WDES%NPRO+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR
    ! and set it to the maximum value of all processors
    CALLMPI (M_max_i(COMM, WDES%NPROD, 1)) ! (required for  SHMALLOC)
#endif

    WDES%NPRO  =WDES%NPRO      *WDES%NRSPINORS
    WDES%NPRO_TOT=WDES%NPRO_TOT*WDES%NRSPINORS
    WDES%NPROD =WDES%NPROD*WDES%NRSPINORS
    WDES%NPROD_RED =WDES%NPROD /WDES%NB_PAR
#ifdef MPI
    IF (WDES%NB_PAR /= WDES%COMM_INTER%NCPU) THEN
       WRITE(*,*)'internal ERROR 3 in WDES_SET_NPRO:',WDES%NB_PAR, WDES%COMM_INTER%NCPU
       STOP
    ENDIF
#endif

    WDES%LOVERL=LOVERL

    ! determine the position the projections onto local ion NI
    ! in the CPROJ arrays: (given by an offset) the first projection
    ! onto the projection operators residing on ion NI has index
    ! LMBASE(NI)+1
    ALLOCATE(WDES%LMBASE(WDES%NIONS+1))
    WDES%LMBASE(1)=0
    DO NI=1,WDES%NIONS
       NT=WDES%ITYP(NI)
       WDES%LMBASE(NI+1)=WDES%LMBASE(NI)+WDES%LMMAX(NT)
    ENDDO

  END SUBROUTINE WDES_SET_NPRO


!=======================================================================
!
! this routine gives the local storage index for
! the non local overlap CQIJ , strength CDIJ matrix elements
! and for the projected orbitals
! return is 0 if the element resides on an other processor
! on entry NI is the global index
!
!=======================================================================

  FUNCTION NI_LOCAL(NI,COMM)
    IMPLICIT NONE
    TYPE (communic)  COMM
    INTEGER NI,NI_LOCAL
#ifdef MPI
    INTEGER NODE_TARGET

    NODE_TARGET=MOD(NI-1, COMM%NCPU)+1
    NI_LOCAL   =   (NI-1)/COMM%NCPU +1

    IF (NODE_TARGET /= COMM%NODE_ME) THEN
       NI_LOCAL=0
    ENDIF
#else
    !
    ! in conventional version all elements are local
    !
    NI_LOCAL=NI
#endif

    RETURN
  END FUNCTION NI_LOCAL

!=======================================================================
!
! this routine gives the global ion index for
! the non local overlap CQIJ , strength CDIJ matrix elements
! and for the projected orbitals
!
!=======================================================================

  FUNCTION NI_GLOBAL(NI,COMM)
    IMPLICIT NONE
    TYPE (communic)  COMM
    INTEGER NI, NI_GLOBAL
#ifdef MPI
    INTEGER NODE_TARGET
    NI_GLOBAL=(NI-1)*COMM%NCPU + COMM%NODE_ME
#else
    !
    ! in conventional version all elements are local
    !
    NI_GLOBAL=NI
#endif

    RETURN
  END FUNCTION NI_GLOBAL

!=======================================================================
!  set WDES for one k-point
!  this is quite simple and sometimes necessary
!=======================================================================

  SUBROUTINE CREATE_SINGLE_KPOINT_WDES(WDES_ORIG,WDES,NK)
    IMPLICIT NONE
    TYPE (wavedes)  WDES,WDES_ORIG
    INTEGER NK

    WDES=WDES_ORIG
    WDES%NKPTS=1

    WDES%NPLWKP=> WDES_ORIG%NPLWKP(NK:NK)
    WDES%NGVECTOR=> WDES_ORIG%NGVECTOR(NK:NK)
    WDES%NGVECTOR_POS=> WDES_ORIG%NGVECTOR_POS(NK:NK)
    WDES%NPLWKP_TOT=> WDES_ORIG%NPLWKP_TOT(NK:NK)
    WDES%WTKPT => WDES_ORIG%WTKPT (NK:NK)
    WDES%VKPT  => WDES_ORIG%VKPT  (:,NK:NK)
    WDES%NINDPW=> WDES_ORIG%NINDPW(:,NK:NK)
    WDES%IGX   => WDES_ORIG%IGX   (:,NK:NK)
    WDES%IGY   => WDES_ORIG%IGY   (:,NK:NK)
    WDES%IGZ   => WDES_ORIG%IGZ   (:,NK:NK)
    WDES%DATAKE=> WDES_ORIG%DATAKE(:,:,NK:NK)
    WDES%AT_GAMMA=WDES_ORIG%AT_GAMMA(NK)
    WDES%LORBITALREAL=WDES_ORIG%LORBITALREAL

    IF (WDES%NCOL>0) THEN
       WDES%PL_INDEX => WDES_ORIG%PL_INDEX(:,NK:NK)
       WDES%PL_COL   => WDES_ORIG%PL_COL (:,NK:NK)
    ENDIF

  END SUBROUTINE CREATE_SINGLE_KPOINT_WDES

!=======================================================================
!  initialize the storage for the orbitals
!=======================================================================

  SUBROUTINE ALLOCW(WDES,W,WUP,WDW)
    USE ini
    IMPLICIT NONE
    INTEGER NK
    TYPE (wavedes), TARGET ::  WDES
    TYPE (wavespin) W
    TYPE (wavefun), OPTIONAL :: WUP,WDW

    INTEGER NRPLWV,NPROD,NKPTS,NBANDS,ISPIN,NB_TOT,NB_PAR,NB_LOW

    NRPLWV=WDES%NRPLWV
    NPROD =WDES%NPROD
    NKPTS =WDES%NKPTS
    NBANDS=WDES%NBANDS
    NB_TOT=WDES%NB_TOT
    NB_LOW=WDES%NB_LOW
    NB_PAR=WDES%NB_PAR
    ISPIN =WDES%ISPIN

    ALLOCATE(W%CPTWFP(NRPLWV,NBANDS,NKPTS,ISPIN), &
         W%CPROJ (NPROD, NBANDS,NKPTS,ISPIN), &
         W%CELTOT(NB_TOT,NKPTS,ISPIN), &
         W%FERTOT(NB_TOT,NKPTS,ISPIN), &
         W%AUXTOT(NB_TOT,NKPTS,ISPIN))
    IF (WDES%LGAMMA) THEN
       CALL REGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+8._q *SIZE(W%CPROJ), "wavefun")
    ELSE
       CALL REGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+16._q*SIZE(W%CPROJ), "wavefun")
    ENDIF

    W%CPTWFP=0
    W%CPROJ =0
    W%CELTOT=0
    W%FERTOT=0
    W%AUXTOT=1
    W%FERWE => W%FERTOT(NB_LOW:NB_TOT:NB_PAR,:,:)
    W%AUX   => W%AUXTOT(NB_LOW:NB_TOT:NB_PAR,:,:)
    W%CELEN => W%CELTOT(NB_LOW:NB_TOT:NB_PAR,:,:)

    W%OVER_BAND=.FALSE.
    W%WDES  => WDES

    IF (PRESENT(WUP)) THEN
       WUP%CELTOT=> W%CELTOT(:,:,1)
       WUP%FERTOT=> W%FERTOT(:,:,1)
       WUP%AUXTOT=> W%AUXTOT(:,:,1)
       WUP%CELEN => W%CELEN(:,:,1)
       WUP%FERWE => W%FERWE(:,:,1)
       WUP%AUX   => W%AUX  (:,:,1)
       WUP%CPTWFP=> W%CPTWFP(:,:,:,1)
       WUP%CPROJ => W%CPROJ(:,:,:,1)      
       WUP%OVER_BAND=.FALSE.
       WUP%WDES  => WDES
    ENDIF
 
    IF (PRESENT(WDW) .AND. WDES%ISPIN==2) THEN
       WDW%CELTOT=> W%CELTOT(:,:,2)
       WDW%FERTOT=> W%FERTOT(:,:,2)
       WDW%AUXTOT=> W%AUXTOT(:,:,2)
       WDW%CELEN => W%CELEN(:,:,2)
       WDW%FERWE => W%FERWE(:,:,2)
       WDW%AUX   => W%AUX  (:,:,2)
       WDW%CPTWFP=> W%CPTWFP(:,:,:,2)
       WDW%CPROJ => W%CPROJ(:,:,:,2)
       WDW%OVER_BAND=.FALSE.
       WDW%WDES  => WDES
    ENDIF

  END SUBROUTINE ALLOCW


!=======================================================================
!  initialize the storage for the orbitals without 
!  orbital coefficients
!=======================================================================

  SUBROUTINE ALLOCW_NOPLANEWAVE(WDES,W)
    USE ini
    IMPLICIT NONE
    INTEGER NK
    TYPE (wavedes), TARGET ::  WDES
    TYPE (wavespin) W

    INTEGER NRPLWV,NPROD,NKPTS,NBANDS,ISPIN,NB_TOT,NB_PAR,NB_LOW

    NRPLWV=WDES%NRPLWV
    NPROD =WDES%NPROD
    NKPTS =WDES%NKPTS
    NBANDS=WDES%NBANDS
    NB_TOT=WDES%NB_TOT
    NB_LOW=WDES%NB_LOW
    NB_PAR=WDES%NB_PAR
    ISPIN =WDES%ISPIN

    ALLOCATE( &
         W%CELTOT(NB_TOT,NKPTS,ISPIN), &
         W%FERTOT(NB_TOT,NKPTS,ISPIN), &
         W%AUXTOT(NB_TOT,NKPTS,ISPIN))

    W%CELTOT=0
    W%FERTOT=0
    W%AUXTOT=1
    W%FERWE => W%FERTOT(NB_LOW:NB_TOT:NB_PAR,:,:)
    W%AUX   => W%AUXTOT(NB_LOW:NB_TOT:NB_PAR,:,:)
    W%CELEN => W%CELTOT(NB_LOW:NB_TOT:NB_PAR,:,:)

    W%OVER_BAND=.FALSE.
    W%WDES  => WDES

 END SUBROUTINE ALLOCW_NOPLANEWAVE


!=======================================================================
!  initialize the storage for the orbitals without 
!  orbital coefficients
!=======================================================================

 SUBROUTINE DEALLOCW_NOPLANEWAVE(W)
    USE ini
    IMPLICIT NONE
    INTEGER NK
    TYPE (wavedes), TARGET ::  WDES
    TYPE (wavespin) W

    DEALLOCATE( &
         W%CELTOT, &
         W%FERTOT, &
         W%AUXTOT)

  END SUBROUTINE DEALLOCW_NOPLANEWAVE

!=======================================================================
!  dealloc of orbital array
!=======================================================================

  SUBROUTINE DEALLOCW(W)
    USE ini
    IMPLICIT NONE
    TYPE (wavespin) W

    IF (ASSOCIATED(W%CPTWFP)) THEN
       IF (W%WDES%LGAMMA) THEN
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+8._q*SIZE(W%CPROJ) , "wavefun")
       ELSE
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+16._q*SIZE(W%CPROJ), "wavefun")
       ENDIF
    ENDIF

    IF (ASSOCIATED(W%CPTWFP) ) DEALLOCATE(W%CPTWFP)
    IF (ASSOCIATED(W%CPROJ) )  DEALLOCATE(W%CPROJ)
    IF (ASSOCIATED(W%CELTOT) ) DEALLOCATE(W%CELTOT)
    IF (ASSOCIATED(W%FERTOT) ) DEALLOCATE(W%FERTOT)
    IF (ASSOCIATED(W%AUXTOT) ) DEALLOCATE(W%AUXTOT)

    NULLIFY(W%CPTWFP)
    NULLIFY(W%CPROJ)
    NULLIFY(W%CELTOT)
    NULLIFY(W%FERTOT)
    NULLIFY(W%AUXTOT)

  END SUBROUTINE DEALLOCW

!=======================================================================
! dealloc of orbital array but CELTOT and FERTOT remain
! allocated
!=======================================================================

  SUBROUTINE DEALLOCW_CW(W)
    USE ini
    IMPLICIT NONE
    TYPE (wavespin) W

    IF (ASSOCIATED(W%CPTWFP)) THEN
       IF (W%WDES%LGAMMA) THEN
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+8._q*SIZE(W%CPROJ) , "wavefun")
       ELSE
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(W%CPTWFP)+16._q*SIZE(W%CPROJ), "wavefun")
       ENDIF
    ENDIF

    IF (ASSOCIATED(W%CPTWFP) ) DEALLOCATE(W%CPTWFP)
    IF (ASSOCIATED(W%CPROJ) )  DEALLOCATE(W%CPROJ)
    NULLIFY(W%CPTWFP)
    NULLIFY(W%CPROJ)

  END SUBROUTINE DEALLOCW_CW


!=======================================================================
!  initialize and descriptor for one orbital  (wavedes1) from
!  an descriptor of an array of orbitals  (wavedes)
!  (kpoint index must be supplied)
!=======================================================================

  SUBROUTINE SETWDES(WDES,WDES1,NK)
    IMPLICIT NONE
    INTEGER NK
    TYPE (wavedes)  WDES
    TYPE (wavedes1) WDES1

    WDES1%RSPIN= WDES%RSPIN
    WDES1%LNONCOLLINEAR=WDES%LNONCOLLINEAR
    WDES1%LSORBIT=WDES%LSORBIT
    WDES1%NRSPINORS=WDES%NRSPINORS
    WDES1%NRPLWV=WDES%NRPLWV
    WDES1%NRPLWV_RED=WDES%NRPLWV_RED
    WDES1%NGDIM =WDES%NGDIM
    WDES1%NPROD= WDES%NPROD
    WDES1%NPROD_RED= WDES%NPROD_RED
    WDES1%NBANDS=WDES%NBANDS
    WDES1%NB_TOT=WDES%NB_TOT
    WDES1%NB_PAR=WDES%NB_PAR
    WDES1%NB_LOW=WDES%NB_LOW
    IF (NK/=0) THEN
       IF (NK > SIZE(WDES%NGVECTOR) ) THEN
          WRITE(*,*) 'internal error in SETWDES: NK exceeds SIZE', NK, SIZE(WDES%NGVECTOR)
          STOP
       ENDIF

       WDES1%NPL   =WDES%NPLWKP(NK)
       WDES1%NGVECTOR= WDES%NGVECTOR(NK)
       WDES1%NGVECTOR_POS= WDES%NGVECTOR_POS(NK)
       WDES1%NPL_TOT= WDES%NPLWKP_TOT(NK)
       WDES1%NB_TOTK= WDES%NB_TOTK(NK,1:2)
    ENDIF
    WDES1%NPRO  =WDES%NPRO
    WDES1%NPRO_TOT = WDES%NPRO_TOT
    WDES1%LOVERL=WDES%LOVERL
    WDES1%DO_REDIS=WDES%DO_REDIS
    WDES1%NIONS =WDES%NIONS
    WDES1%NTYP  =WDES%NTYP
    WDES1%NITYP=>WDES%NITYP
    WDES1%ITYP=>WDES%ITYP
    WDES1%GRID =>WDES%GRID 
    WDES1%RINPL =1._q/WDES1%GRID%NPLWV
    WDES1%LMMAX=>WDES%LMMAX
    WDES1%LMBASE=>WDES%LMBASE
    WDES1%NT_GLOBAL=>WDES%NT_GLOBAL
    WDES1%NPRO_POS=>WDES%NPRO_POS
    IF (NK/=0) THEN
       WDES1%NINDPW=>WDES%NINDPW(:,NK)
       WDES1%IGX  =>WDES%IGX(:,NK)
       WDES1%IGY  =>WDES%IGY(:,NK)
       WDES1%IGZ  =>WDES%IGZ(:,NK)
       WDES1%VKPT =>WDES%VKPT(:,NK)
       WDES1%AT_GAMMA=WDES%AT_GAMMA(NK)
       WDES1%WTKPT = 0
       IF( NK <=WDES%NKPTS) THEN
          WDES1%DATAKE=>WDES%DATAKE(:,:,NK)
          WDES1%WTKPT= WDES%WTKPT(NK)
       ENDIF
    ENDIF
    WDES1%LORBITALREAL=WDES%LORBITALREAL      
    WDES1%LSPIRAL=WDES%LSPIRAL
    WDES1%LGAMMA=WDES%LGAMMA
    WDES1%LZEROZ=WDES%LZEROZ
    WDES1%QSPIRAL=WDES%QSPIRAL
    WDES1%NK    =NK
    WDES1%NCOL  =WDES%NCOL
    IF (WDES%NCOL>0 .AND. NK <= WDES%NKPTS .AND. NK/=0) THEN
       WDES1%PL_INDEX => WDES%PL_INDEX(:,NK)
       WDES1%PL_COL   => WDES%PL_COL (:,NK)
    ENDIF

    WDES1%COMM        => WDES%COMM
    WDES1%COMM_INTER  => WDES%COMM_INTER
    WDES1%COMM_INB    => WDES%COMM_INB
    WDES1%COMM_KINTER => WDES%COMM_KINTER
    WDES1%COMM_KIN    => WDES%COMM_KIN
#ifdef use_shmem
    WDES1%COMM_SHMEM  => WDES%COMM_SHMEM
    WDES1%COMM_intra_node => WDES%COMM_intra_node
    WDES1%COMM_inter_node => WDES%COMM_inter_node
#endif
    WDES1%NPL_RED = WDES1%NPL     ! number of plane waves/node after data redistribution
    WDES1%NPRO_RED= WDES1%NPRO    ! number of projected wavef. after data redistribution

    CALLMPI( SET_NPL_NPRO(WDES1, WDES1%NPL_RED, WDES1%NPRO_RED))
    WDES1%NPRO_O_RED=WDES1%NPRO_RED
    IF (.NOT. WDES1%LOVERL) THEN
       WDES1%NPRO_O_RED=0
    ENDIF

  END SUBROUTINE SETWDES

!=======================================================================
!  initialize the optional datas required for real space calculations
!  in a descriptor for one single orbital (wavedes1)
!  scheduled for removal
!=======================================================================

  SUBROUTINE SETWGRID_OLD(WDES1,GRID)
    IMPLICIT NONE

    TYPE (grid_3d), TARGET ::  GRID
    TYPE (wavedes1) WDES1

    WDES1%RINPL =1._q/GRID%NPLWV
    WDES1%GRID =>GRID

  END SUBROUTINE SETWGRID_OLD

!=======================================================================
!  create storage for one orbital W
!  optionally real space array is allocated
!=======================================================================

  SUBROUTINE NEWWAV(W1, WDES1, ALLOC_REAL, ISTAT)
    IMPLICIT NONE
    TYPE (wavefun1), TARGET :: W1
    TYPE (wavedes1), TARGET :: WDES1
    LOGICAL ALLOC_REAL
    INTEGER, OPTIONAL :: ISTAT
    ! local variables
    INTEGER MPLWV
    IF (PRESENT(ISTAT)) THEN
       IF (ALLOC_REAL) THEN
          MPLWV=WDES1%GRID%MPLWV*WDES1%NRSPINORS
          ALLOCATE(W1%CPTWFP(WDES1%NRPLWV),W1%CPROJ(WDES1%NPROD),W1%CR(MPLWV),STAT=ISTAT)
       ELSE
          ALLOCATE(W1%CPTWFP(WDES1%NRPLWV),W1%CPROJ(WDES1%NPROD),STAT=ISTAT)
          NULLIFY(W1%CR)
       ENDIF
    ELSE
       IF (ALLOC_REAL) THEN
          MPLWV=WDES1%GRID%MPLWV*WDES1%NRSPINORS
          ALLOCATE(W1%CPTWFP(WDES1%NRPLWV),W1%CPROJ(WDES1%NPROD),W1%CR(MPLWV))
       ELSE
          ALLOCATE(W1%CPTWFP(WDES1%NRPLWV),W1%CPROJ(WDES1%NPROD))
          NULLIFY(W1%CR)
       ENDIF
    ENDIF
    W1%FERWE=0
    W1%CELEN=0

    W1%WDES1 =>WDES1
    ! not a valid band of the Hamiltonian
    W1%NB=-1
    W1%ISP=-1
    W1%LDO=.TRUE.
  END SUBROUTINE NEWWAV

!=======================================================================
!  create parallel storage for orbitals W
!  optionally real space array is allocated
!  this version is for OPENMP creating OMP_GET_MAX_THREADS()
!  W1 instances
!=======================================================================

  SUBROUTINE NEWWAV_OMP(W1, WDES1, ALLOC_REAL )
    IMPLICIT NONE
    TYPE (wavefun1), POINTER ::  W1(:)
    TYPE (wavedes1), TARGET :: WDES1
    LOGICAL ALLOC_REAL
    INTEGER N

    ALLOCATE(W1(1))
    CALL NEWWAV(W1(1), WDES1, ALLOC_REAL)
  END SUBROUTINE NEWWAV_OMP

  SUBROUTINE DELWAV_OMP(W1, ALLOC_REAL )
    IMPLICIT NONE
    TYPE (wavefun1), POINTER ::  W1(:)
    LOGICAL ALLOC_REAL
    INTEGER N

    DO N=1,SIZE(W1,1)
      CALL DELWAV(W1(N), ALLOC_REAL )
    ENDDO
    DEALLOCATE(W1)
    NULLIFY(W1)

  END SUBROUTINE DELWAV_OMP

#ifdef use_shmem
!=======================================================================
!  create storage for one orbital W
!  optionally real space array is allocated
!  use shmem shared memory
!=======================================================================

  SUBROUTINE NEWWAV_SHMEM(W1,WDES1,ALLOC_REAL,address,shmid)
    USE c2f_interface
    IMPLICIT NONE
    TYPE (wavefun1) W1
    TYPE (wavedes1), TARGET :: WDES1
    LOGICAL ALLOC_REAL
    TYPE(c_ptr) address
    INTEGER(c_int) shmid
    ! local variables
    INTEGER(c_size_t) k
    INTEGER ierror

    IF (ALLOC_REAL) THEN
       k=WDES1%GRID%MPLWV*WDES1%NRSPINORS
       IF (WDES1%COMM_SHMEM%NODE_ME==1) CALL getshmem(INT(8*2*k,KIND=c_size_t),shmid)
       CALL M_bcast_i(WDES1%COMM_SHMEM,shmid,1)
       CALLMPI( MPI_barrier(WDES1%COMM_SHMEM%MPI_COMM,ierror) )
       call attachshmem(shmid, address)
       call c_f_pointer(address,W1%CR,[k])
       ALLOCATE(W1%CPROJ(WDES1%NPROD))
       NULLIFY(W1%CW)
    ELSE
       k=WDES1%NRPLWV
       IF (WDES1%COMM_SHMEM%NODE_ME==1) CALL getshmem(INT(8*2*k,KIND=c_size_t),shmid)
       CALL M_bcast_i(WDES1%COMM_SHMEM,shmid,1)
       CALLMPI( MPI_barrier(WDES1%COMM_SHMEM%MPI_COMM,ierror) )
       call attachshmem(shmid, address)
       call c_f_pointer(address,W1%CPTWFP,[k])
       ALLOCATE(W1%CPROJ(WDES1%NPROD))
       NULLIFY(W1%CR)
    ENDIF
    W1%FERWE=0
    W1%CELEN=0

    W1%WDES1=>WDES1
    ! not a valid band of the Hamiltonian
    W1%NB=-1
    W1%ISP=-1
    W1%LDO=.TRUE.
  END SUBROUTINE NEWWAV_SHMEM
#endif

!=======================================================================
!  create storage for one orbital W for real space storage
!=======================================================================

  SUBROUTINE NEWWAV_R(W1, WDES1)
    IMPLICIT NONE
    TYPE (wavefun1), TARGET :: W1
    TYPE (wavedes1), TARGET :: WDES1
    ! local variables
    INTEGER MPLWV

    MPLWV=WDES1%GRID%MPLWV*WDES1%NRSPINORS
    ALLOCATE(W1%CR(MPLWV))
    NULLIFY(W1%CW, W1%CPROJ)
    W1%FERWE=0
    W1%CELEN=0

    W1%WDES1 =>WDES1
    ! not a valid band of the Hamiltonian
    W1%NB=-1
    W1%ISP=-1
    W1%LDO=.TRUE.

  END SUBROUTINE NEWWAV_R

!=======================================================================
!  create storage for one orbital W for real space storage
!=======================================================================

  SUBROUTINE NEWWAV_NOCW(W1, WDES1)
    IMPLICIT NONE
    TYPE (wavefun1) W1
    TYPE (wavedes1), TARGET :: WDES1
    INTEGER MPLWV

    MPLWV=WDES1%GRID%MPLWV*WDES1%NRSPINORS
    ALLOCATE(W1%CR(MPLWV))
    ALLOCATE(W1%CPROJ(WDES1%NPROD))
    NULLIFY(W1%CW)
    W1%FERWE=0
    W1%CELEN=0

    W1%WDES1 =>WDES1
    ! not a valid band of the Hamiltonian
    W1%NB=-1
    W1%ISP=-1
    W1%LDO=.TRUE.

  END SUBROUTINE NEWWAV_NOCW

!=======================================================================
!  destroy storage for one orbital W
!  optionally real space array is deallocated
!=======================================================================

  SUBROUTINE DELWAV(W,DEALLOC_REAL,ISTAT)
    IMPLICIT NONE
    TYPE (wavefun1) W
    LOGICAL DEALLOC_REAL
    INTEGER, OPTIONAL :: ISTAT

    IF (PRESENT(ISTAT)) THEN
       IF (DEALLOC_REAL) THEN
          DEALLOCATE(W%CPTWFP,W%CPROJ,W%CR,STAT=ISTAT)
       ELSE
          DEALLOCATE(W%CPTWFP,W%CPROJ,STAT=ISTAT)
       ENDIF
    ELSE
       IF (DEALLOC_REAL.AND. ASSOCIATED(W%CR)) THEN
          DEALLOCATE(W%CPTWFP,W%CPROJ,W%CR)
       ELSE IF (DEALLOC_REAL .AND. .NOT. ASSOCIATED(W%CR)) THEN
          WRITE(*,*) 'internal error in DELWAV: should be called with DEALLOC_REAL=.FALSE.'
       ELSE
          DEALLOCATE(W%CPTWFP,W%CPROJ)
       ENDIF
    ENDIF    

  END SUBROUTINE DELWAV

#ifdef use_shmem
!=======================================================================
!  destroy storage for one orbital W
!  optionally real space array is deallocated
!  use shmem shared memory
!=======================================================================

  SUBROUTINE DELWAV_SHMEM(W1,WDES1,address,shmid)
    USE c2f_interface
    IMPLICIT NONE
    TYPE (wavefun1) W1
    TYPE (wavedes1), TARGET :: WDES1
    LOGICAL DEALLOC_REAL
    TYPE(c_ptr) address
    INTEGER(c_int) shmid
    ! local variables
    INTEGER ierror

    CALL detachshmem(address)
    IF (WDES1%COMM_SHMEM%NODE_ME==1) CALL destroyshmem(shmid)
    CALLMPI( MPI_barrier(WDES1%COMM_SHMEM%MPI_COMM,ierror) )
    DEALLOCATE(W1%CPROJ)

  END SUBROUTINE DELWAV_SHMEM
#endif

  SUBROUTINE DELWAV_R(W)
    IMPLICIT NONE
    TYPE (wavefun1) W
    DEALLOCATE(W%CR)
  END SUBROUTINE DELWAV_R

  SUBROUTINE DELWAV_NOCW(W)
    IMPLICIT NONE
    TYPE (wavefun1) W
    DEALLOCATE(W%CR,W%CPROJ)
    NULLIFY(W%CR,W%CPROJ)
  END SUBROUTINE DELWAV_NOCW

!=======================================================================
!  set one singe orbital (W1) from an array of orbitals
!  local band index and k point must be supplied
!=======================================================================

  SUBROUTINE SETWAV(W,W1,WDES1,NB,ISP)
    IMPLICIT NONE
    INTEGER NB,ISP
    TYPE (wavespin) W
    TYPE (wavefun1) W1
    TYPE (wavedes1), TARGET :: WDES1
    INTEGER NK

    NK=WDES1%NK

    W1%CPTWFP=>W%CPTWFP(:,NB,NK,ISP)
    W1%CPROJ =>W%CPROJ(:,NB,NK,ISP)
    W1%FERWE =W%FERWE(NB,NK,ISP)
    W1%CELEN =W%CELEN(NB,NK,ISP)
    W1%WDES1 => WDES1
    ! remember band index
    ! presently not used
    W1%NB =NB
    W1%ISP =ISP
    W1%LDO=.TRUE.
  END SUBROUTINE SETWAV


!=======================================================================
!  set orbital (wavefun) from an spin array of orbitals
!  spin must be supplied
!=======================================================================

  SUBROUTINE SETW_SPIN(W,W1,ISPIN)
    USE prec
    IMPLICIT NONE
    INTEGER ISPIN
    TYPE (wavespin) W
    TYPE (wavefun)  W1

    W1%CPTWFP=>W%CPTWFP(:,:,:,ISPIN)
    W1%CPROJ =>W%CPROJ (:,:,:,ISPIN)
    W1%FERWE =>W%FERWE (:,:,ISPIN)
    W1%CELEN =>W%CELEN (:,:,ISPIN)
    W1%AUX   =>W%AUX   (:,:,ISPIN)
    W1%FERTOT=>W%FERTOT(:,:,ISPIN)
    W1%AUXTOT=>W%AUXTOT(:,:,ISPIN)
    W1%CELTOT=>W%CELTOT(:,:,ISPIN)

  END SUBROUTINE SETW_SPIN

!************************* SUBROUTINE WVREAL ***************************
!
! this subroutine makes a orbital real
! it is required for the gamma point only mode
! to avoid that small non real components develop
!
!***********************************************************************

  SUBROUTINE WVREAL(WDES,GRID,W)
    IMPLICIT REAL(q) (A-H,O-Z)
    TYPE (wavespin) W
    TYPE (wavedes)  WDES
    TYPE (grid_3d)  GRID
#ifdef gammareal
    !
    ! search for position of G=0 zero element
    !
    NK  =1
    NIND=0
    NPL =WDES%NGVECTOR(NK)

    DO N=1,NPL
       IND=WDES%NINDPW(N,NK)
       N1= MOD((IND-1),GRID%RC%NROW)+1
       NC= (IND-1)/GRID%RC%NROW+1
       N2= GRID%RC%I2(NC)
       N3= GRID%RC%I3(NC)
       IF (N1==1 .AND. N2==1 .AND. N3==1) THEN
          NIND=N
          EXIT
       ENDIF
    ENDDO

    IF (NIND /= 0) THEN
       DO ISP=1,WDES%ISPIN
          DO NB=1,WDES%NBANDS
             W%CPTWFP(NIND,NB,NK,ISP)= REAL( W%CPTWFP(NIND,NB,NK,ISP) ,KIND=q)
          ENDDO
       ENDDO
       IF (WDES%LNONCOLLINEAR) THEN
          DO ISP=1,WDES%ISPIN
             DO NB=1,WDES%NBANDS
                W%CPTWFP(NIND+NPL,NB,NK,ISP)= REAL( W%CPTWFP(NIND+NPL,NB,NK,ISP) ,KIND=q)
             ENDDO
          ENDDO
       ENDIF
    ENDIF

#endif
    RETURN
  END SUBROUTINE WVREAL


!=======================================================================
!
! NB_LOCAL returns the local storage index of a band
! if bands are distributed over processors
!
!=======================================================================

  FUNCTION NB_LOCAL(NB,WDES1)
    USE prec
    IMPLICIT NONE
    INTEGER NB,NB_LOCAL
    TYPE (wavedes1)    WDES1

    IF ( MOD(NB-1,WDES1%NB_PAR)+1 == WDES1%NB_LOW) THEN
       NB_LOCAL=1+(NB-1)/WDES1%NB_PAR
    ELSE
       NB_LOCAL=0
    ENDIF

  END FUNCTION NB_LOCAL

!***************************SUBROUTINE WFINIT***************************
!
! this subroutine initializes the orbital array W
! it use always a random number generator
! to initialize the coefficients
!
!***********************************************************************

#define serial_compat
#define random

  SUBROUTINE WFINIT(WDES, W, ENINI, NBANDS)
    USE constant
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (wavespin)    W
    REAL(q)            ENINI    ! cutoff energy
    INTEGER, OPTIONAL::NBANDS   ! first band to be initialized
 !  local
    LOGICAL :: LSCALE
    TYPE (wavedes)     WDES
    TYPE (wavedes1)    WDES1
    COMPLEX (q),ALLOCATABLE ::  CW(:)
    COMPLEX (q) :: YY
    INTEGER :: ISPINOR, I, NK, FIRST_BAND
    REAL(q) :: MAX_EIGENVALUE

    IF (PRESENT(NBANDS)) THEN
      FIRST_BAND=NBANDS
      MAX_EIGENVALUE=MAXVAL(REAL(W%CELTOT,q))+2
    ELSE
      FIRST_BAND=1
      W%CELTOT=0
      W%CPTWFP=0
      MAX_EIGENVALUE=0
    ENDIF

    NALLOC = MAXVAL(WDES%NPLWKP_TOT)
    ALLOCATE(CW(NALLOC))

    ! if all band are calculated, use non weighted random numbers
    ! for the initialization; this avoids linear dependencies
    IF (MINVAL(WDES%NB_TOTK(1:WDES%NKPTS,:))< WDES%NB_TOT) THEN
       LSCALE=.FALSE.
    ELSE
       LSCALE=.TRUE.
    ENDIF

    spin:   DO I=1,WDES%ISPIN
       YR=RANE()  ! what is that ? it is here for compatibility
       kpoint: DO NK=1,WDES%NKPTS

          CALL SETWDES(WDES,WDES1,NK)

#if defined(serial_compat) && defined(MPI)
          band: DO NB_GLOBAL=FIRST_BAND,WDES%NB_TOT
             NB=NB_LOCAL(NB_GLOBAL,WDES1)
!=======================================================================
!     initialize for serial compatible version
!=======================================================================
             CW=0  ! just in case initialize CW
             NPL_GLOBAL=WDES%NPLWKP_TOT(NK)/WDES%NRSPINORS

             DO ISPINOR=0,WDES%NRSPINORS-1 ! to be included later, Testing only
                DO M=1,NPL_GLOBAL
                   YY=RANE()
                   !  at the gamma it is somethimes better to use random phase factor
                   !  (!!! but if the cell has inversion symmtry it is a bad choice !!!)
#if defined(random) || defined(gammareal)
                   IF (M/=1) YY=CMPLX(REAL(YY,q),0.2_q*RANE()-0.1_q,q)
#endif
                   MM=M+NPL_GLOBAL*ISPINOR
                   CW(MM)=YY
                ENDDO
             ENDDO
             CALL DIS_PW_BAND(WDES1,NB_GLOBAL,CW,W%CPTWFP(1,1,NK,I))
             IF (NB==0) CYCLE band  ! not on local node, than cycle

             NPL=WDES%NGVECTOR(NK)

             DO ISPINOR=0,WDES%NRSPINORS-1
                DO M=1,NPL
                   MM=M+NPL*ISPINOR
                   IF(WDES%DATAKE(M,ISPINOR+1,NK)>=ENINI) THEN
                      W%CPTWFP(MM,NB,NK,I)=0
                   ENDIF
                   WW=WDES%DATAKE(M,ISPINOR+1,NK)      
                   IF(WW<=0.01_q) WW=0.1_q
                   IF(.NOT. LSCALE) WW=1.0
                   W%CPTWFP(MM,NB,NK,I)=W%CPTWFP(MM,NB,NK,I)/WW
                ENDDO
             ENDDO
!=======================================================================
!     initialize components of orbital with random numbers
!     weighted with the inverse of the kinetic energy of the plane
!     wave (random phase factor added gK)
!=======================================================================
#else
          NPL=WDES%NGVECTOR(NK)

          band: DO NB=FIRST_BAND,WDES%NBANDS
             spinor: DO ISPINOR=0,WDES%NRSPINORS-1 ! to be included later, Testing only
                DO M=1,NPL
                   YY=RANE()
                   !  at the gamma it is somethimes better to use random phase factor
                   !  (!!! but if the cell has inversion symmtry it is a bad choice !!!)
#if defined(random) || defined(gammareal)
                   IF (M/=1) YY=CMPLX(REAL(YY,q),0.2_q*RANE()-0.1_q,q)
#endif

                   IF(WDES%DATAKE(M,ISPINOR+1,NK)>=ENINI) THEN
                      YY=0
                   ENDIF
                   WW=WDES%DATAKE(M,ISPINOR+1,NK)          
                   IF(WW<=0.01_q) WW=0.1_q
                   IF(.NOT.LSCALE) WW=1.0
                   YY=YY/WW
                   W%CPTWFP(M+ NPL*ISPINOR, NB, NK, I)=YY
                ENDDO
             ENDDO spinor
#endif
!=======================================================================
! calculate magnitude squared of orbital
!=======================================================================
             WFMAG=0
             DO M=1,WDES%NPLWKP(NK)
                CCC=W%CPTWFP(M,NB,NK,I)
                WFMAG=WFMAG+CCC*CONJG(CCC)
             ENDDO
             CALLMPI( M_sum_d(WDES%COMM_INB, WFMAG, 1))
!=======================================================================
! check that it is nonzero
!=======================================================================
             IF (WFMAG<=0.000001_q) THEN
                WRITE(6,10)
10              FORMAT('internal error in WFINIT: orbitals linearily dependent at', &
                     ' random-number initialization ')
                STOP
             ENDIF
!=======================================================================
!     normalize the orbital
!     and set CELEN to kinetic energy
!=======================================================================
             WFMINV=1._q/SQRT(WFMAG)
             DO M=1,WDES%NPLWKP(NK)
                W%CPTWFP(M,NB,NK,I)=W%CPTWFP(M,NB,NK,I)*WFMINV
             ENDDO

             SUM_=0

             DO ISPINOR=0,WDES%NRSPINORS-1
                DO M=1,NPL
                   MM=M+NPL*ISPINOR
                   SUM_=SUM_+W%CPTWFP(MM,NB,NK,I)*CONJG(W%CPTWFP(MM,NB,NK,I))*WDES%DATAKE(M,ISPINOR+1,NK)
                ENDDO
             ENDDO
             CALLMPI( M_sum_d(WDES%COMM_INB, SUM_, 1))
             ! W%CELEN is hardly used, but the occupancies are sometimes
             ! initialized from them
             ! if additional randomly initialized bands are added they most
             ! lie well above the occupied bands
             W%CELEN (NB,NK,I)=MAX(MAX_EIGENVALUE,SUM_)
          ENDDO band
       ENDDO kpoint
    ENDDO spin

    CALL WFZERO(W)
    CALL MRG_CEL(WDES,W)
#ifdef debug
#ifdef MPI
    IF (WDES%COMM%NODE_ME ==  WDES%COMM%IONODE) THEN
       WRITE(*,'(I4,F14.10)') (I,REAL(W%CELTOT(I,1,1),q),I=1,WDES%NB_TOT)
    ENDIF
#else
    WRITE(*,'(I4,F14.10)') (I,REAL(W%CELTOT(I,1,1),q),I=1,WDES%NB_TOT)
#endif
#endif

    DEALLOCATE(CW)
    RETURN
  END SUBROUTINE

!***************************SUBROUTINE WFZERO **************************
!
! this subroutine sets orbitals beyond W%WDES%NB_TOTK(NK,ISP)
! to zero
! the second subroutine sets eigenvalues to very large values
! this should avoid any problems in the DOS routines
! at least as long as they are not WDES%NB_TOTK clean
!
!***********************************************************************

  SUBROUTINE WFZERO(W)
    USE constant
    IMPLICIT NONE
    TYPE (wavespin)    W
 !  local
    INTEGER ISP, NK, NB

    DO ISP=1,W%WDES%ISPIN
       DO NK=1,W%WDES%NKPTS

          ! first local band set to zero 
          DO NB=(W%WDES%NB_TOTK(NK,ISP)-W%WDES%NB_LOW)/W%WDES%NB_PAR+2, W%WDES%NBANDS
             W%CW(:,NB,NK,ISP)=0
             W%CPROJ(:,NB,NK,ISP)=0
             W%FERWE(NB,NK,ISP)=0
             W%CELEN(NB,NK,ISP)=1E4     ! huge eigenvalue
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE WFZERO


  SUBROUTINE WFSET_HIGH_CELEN(W)
    USE constant
    IMPLICIT NONE
    TYPE (wavespin)    W
 !  local
    INTEGER ISP, NK, NB

    DO ISP=1,W%WDES%ISPIN
       DO NK=1,W%WDES%NKPTS
          DO NB=W%WDES%NB_TOTK(NK,ISP)+1, W%WDES%NB_TOT
             W%FERTOT(NB,NK,ISP)=0         ! occupancy zero
             W%CELTOT(NB,NK,ISP)=10000     ! huge eigenvalue
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE WFSET_HIGH_CELEN


!*************************SUBROUTINE GEN_LAYOUT**************************
!
! subroutine GEN_LAYOUT performs a number of tasks:
! ) determines the layout (distribution) of the columns on parallel
!   computers
!   essentially these are the structures GRID%RC, GRID%RL
!   and in the parallel version the array GRID%IN
!
! ) also the following  arrays are allocated:
!     WDES%NPLWKP  WDES%NINDPW
! ) for LSETUP=.TRUE. the kinetic energy arrays and the G-vector array
!   are additionally allocated
!     WDES%DATAKE, WDES%IGX,Y,Z
! ) setup of these arrays is performed in the routines GEN_INDEX
!
! the data layout is based on the initial reciprocal lattice vectors
! stored in BI
!
!***********************************************************************

  SUBROUTINE GEN_LAYOUT(GRID,WDES, B,BI,IU6,LSETUP,LNOGAMMA)
    USE prec
    USE mgrid
    USE constant
    USE base
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d), TARGET :: GRID
    TYPE (wavedes)     WDES
    DIMENSION B(3,3),BI(3,3) ! current lattice, and initial lattice
    LOGICAL LSETUP
    LOGICAL, OPTIONAL :: LNOGAMMA
    ! local
    LOGICAL LUP

    LOGICAL, ALLOCATABLE :: LUSE_IN(:)
    INTEGER, ALLOCATABLE :: USED_ROWS(:),IND2(:),IND3(:)
    INTEGER, ALLOCATABLE :: REDISTRIBUTION_INDEX(:)
    COMMON /WAVCUT/   IXMIN,IXMAX,IYMIN,IYMAX,IZMIN,IZMAX
    REAL(q), PARAMETER :: G2ZERO=1E-12_q

#ifdef gammareal
    WDES%LGAMMA=.TRUE.
    GRID%REAL2CPLX=.TRUE.
#else
    WDES%LGAMMA=.FALSE.
    GRID%REAL2CPLX=.FALSE.
#endif
    ! force complex to complex FFT
    IF (PRESENT(LNOGAMMA)) THEN
       IF ( LNOGAMMA) THEN
          WDES%LGAMMA=.FALSE.
          GRID%REAL2CPLX=.FALSE.
       ENDIF
    ENDIF

    WDES%GRID=> GRID

    GRID%NGX_rd=GRID%NGX
    GRID%NGY_rd=GRID%NGY
    GRID%NGZ_rd=GRID%NGZ

    ! orbitals are always complex in the direct grid in VASP
    ! hence LREAL is set to .FALSE.
    GRID%LREAL=.FALSE.

    CALL NULLIFY_GRD(GRID)

    WDES%NCOL=0
!=======================================================================
! determine the layout
! i.e. all required columns
! or (y,z) pairs which are required for the 3d-FFT grid
!=======================================================================
#ifdef MPI
    IF (GRID%COMM%NCPU/=1 .OR. LUSE_PARALLEL_FFT) THEN
       IF (WDES%LGAMMA) THEN
          GRID%NGZ_rd=GRID%NGZ/2+1
          IF (IU6>=0) THEN
             WRITE(IU6,*) 'use parallel FFT for orbitals z direction half grid'
          ENDIF
       ENDIF
!-----------------------------------------------------------------------
! parallel version
! reciprocal space first
!-----------------------------------------------------------------------

       ! determine all required grid points (usually a circle around center)

       ALLOCATE(USED_ROWS(GRID%NGY*GRID%NGZ_rd), &
            IND2(GRID%NGY*GRID%NGZ_rd),IND3(GRID%NGY*GRID%NGZ_rd), &
            REDISTRIBUTION_INDEX(GRID%NGY*GRID%NGZ_rd), &
            LUSE_IN(GRID%NGZ_rd))
       ! following arrays are required to setup the reciprocal layout
       USED_ROWS=0     ! gives the total number of points in each column
       ! contained within the cutoff sphere
       NCOL_TOT=0      ! total number of columns
       IND2=0          ! y index of each used column
       IND3=0          ! z index of each used column
       ! LUSE_IN array is used to find out which z-planes must be contained
       ! in the intermediate representation
       LUSE_IN=.FALSE.

       DO N3=1,GRID%NGZ_rd
          DO N2=1,GRID%NGY

             IUSED_MAX=0   ! maxmimal number of rows for this column

             ! loop over all k-points in the *entire* or irreducible Brillouin zone (N)
             ! NKPTS_FOR_GEN_LAYOUT is set, if HF is used, or if the number of k-points
             ! is allowed to change dynamically (linear response, finite differences)
             DO NK=1,WDES%NKPTS_FOR_GEN_LAYOUT
                G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
                G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
                IUSED=0     ! number of rows for this column and k-point
                DO N1=1,GRID%NGX_rd
                   G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
                   GIX= (G1*BI(1,1)+G2*BI(1,2)+G3*BI(1,3)) *TPI
                   GIY= (G1*BI(2,1)+G2*BI(2,2)+G3*BI(2,3)) *TPI
                   GIZ= (G1*BI(3,1)+G2*BI(3,2)+G3*BI(3,3)) *TPI
                   ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))

                   ! exclude some components for gamma-only version (C(G)=C*(-G))
                   IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                      IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                      IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE
                   ENDIF

                   IF(ENERGI<WDES%ENMAX) THEN
                      IUSED=IUSED+1
                      LUSE_IN(N3)=.TRUE.
                   ENDIF
                ENDDO
                IUSED_MAX=MAX(IUSED,IUSED_MAX)
             ENDDO
             ! if this column is used, enter it in required array
             IF (IUSED_MAX>0) THEN
                NCOL_TOT=NCOL_TOT+1        ! increase column counter
                USED_ROWS(NCOL_TOT)=IUSED  ! set number of rows in this column
                IND2(NCOL_TOT)=N2          ! set y index of this column
                IND3(NCOL_TOT)=N3          ! set z index of this column
             ENDIF
          ENDDO
       ENDDO
       ! setup redistribution array
       ! by sorting according to the number of rows in each column
       ! we try to get better load balancing
       DO I=1,NCOL_TOT
          REDISTRIBUTION_INDEX(I)=I
       ENDDO
       ! do not sort column 1 (it must store the G vector G=(0,0,0)) !
       CALL SORT_REDIS(NCOL_TOT,USED_ROWS(1),REDISTRIBUTION_INDEX(1))

       ! distribute among processors

       NCOL_MAX=(NCOL_TOT+GRID%COMM%NCPU-1)/GRID%COMM%NCPU
       ALLOCATE(GRID%RC%I2( NCOL_MAX ))
       ALLOCATE(GRID%RC%I3( NCOL_MAX ))

       GRID%RC%NFAST= 1
       GRID%RC%NROW = GRID%NGX_rd
       GRID%RC%NCOL = 0

       IND=0
       !
       ! step through all columns and distribute them onto proc.
       ! in the manner 1 ... NCPU - NCPU ... 1 - 1 ... NCPU - etc.
       !
       NODE_TARGET=0  ! NODE onto which column has to go
       LUP=.TRUE.     ! determines whether NODE_TARGET is increased or decreased

       DO NCOL=1,NCOL_TOT
          IND_REDIS=REDISTRIBUTION_INDEX(NCOL)
          N2=IND2(IND_REDIS)
          N3=IND3(IND_REDIS)
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

          IF (NODE_TARGET == GRID%COMM%NODE_ME) THEN
             GRID%RC%NCOL=GRID%RC%NCOL+1
             IF (IND_ON_CPU /= GRID%RC%NCOL) THEN
                WRITE(*,*)'GENSP: internal error(1) ',GRID%COMM%NODE_ME,IND_ON_CPU,GRID%RC%NCOL
                STOP
             ENDIF
             GRID%RC%I2(IND_ON_CPU)=N2
             GRID%RC%I3(IND_ON_CPU)=N3
          ENDIF
       ENDDO
       GRID%RC%NP    = GRID%RC%NCOL*GRID%RC%NROW
       GRID%RC%NALLOC= GRID%RC%NCOL*GRID%RC%NROW
       WDES%NCOL     = GRID%RC%NCOL
       !
       ! set up intermediate grid
       ! this usually contains a certain number of z-planes around the center
       !
       LUSE_IN=.TRUE.

       NCOL_TOT=0
       DO N3=1,GRID%NGX_rd
          DO N2=1,GRID%NGZ_rd
             IF (LUSE_IN(N2)) NCOL_TOT=NCOL_TOT+1
          ENDDO
       ENDDO

       NCOL_MAX=(NCOL_TOT+GRID%COMM%NCPU-1)/GRID%COMM%NCPU

       IF (LPLANE_WISE) THEN
          NCOL_MAX=GRID%NGZ*((GRID%NGX+GRID%COMM%NCPU-1)/GRID%COMM%NCPU)
       ENDIF

       ALLOCATE(GRID%IN%I2( NCOL_MAX ))
       ALLOCATE(GRID%IN%I3( NCOL_MAX ))

       GRID%IN%NFAST= 2
       GRID%IN%NCOL = 0
       GRID%IN%NROW = GRID%NGY

       IND=0
       DO N3=1,GRID%NGX_rd
          DO N2=1,GRID%NGZ_rd
             IF (LUSE_IN(N2)) THEN
                IF (LPLANE_WISE) THEN
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
             ENDIF
          ENDDO
       ENDDO
       IF (GRID%IN%NCOL >NCOL_MAX) THEN
          WRITE(*,*) 'GEN_LAYOUT: internal error',NCOL_MAX,GRID%IN%NCOL
          STOP
       ENDIF
       GRID%IN%NP    = GRID%IN%NCOL*GRID%IN%NROW
       GRID%IN%NALLOC= GRID%IN%NCOL*GRID%IN%NROW
       !
       ! set up real space grid, use all points
       ! in this case GRID will be conformable to GRID_SOFT in real space
       !
       CALL REAL_STDLAY(GRID, LPARALLEL=.TRUE.)

       DEALLOCATE(USED_ROWS,IND2,IND3,REDISTRIBUTION_INDEX,LUSE_IN)
    ELSE
#endif
!-----------------------------------------------------------------------
! serial version
! set reciprocal space and real space layout for non parallel computers
! always x-first (or x-fast) layout
! all grid points are used for FFT
!-----------------------------------------------------------------------
       IF (WDES%LGAMMA) THEN
          GRID%NGX_rd=GRID%NGX/2+1
          IF (IU6>=0) WRITE(IU6,*) 'use serial FFT for orbitals x direction half grid'
       ENDIF

       GRID%RC%NFAST= 1
       GRID%RC%NCOL = GRID%NGZ_rd*GRID%NGY
       WDES%NCOL    = GRID%RC%NCOL
       GRID%RC%NROW = GRID%NGX_rd
       GRID%RC%NP   = GRID%RC%NCOL*GRID%RC%NROW
       GRID%RC%NALLOC= GRID%RC%NCOL*GRID%RC%NROW
       ALLOCATE(GRID%RC%I2( GRID%RC%NCOL ))
       ALLOCATE(GRID%RC%I3( GRID%RC%NCOL ))
       IND=1
       DO N3=1,GRID%NGZ_rd
          DO N2=1,GRID%NGY
             GRID%RC%I2(IND)=N2
             GRID%RC%I3(IND)=N3
             IND=IND+1
          ENDDO
       ENDDO

       CALL REAL_STDLAY(GRID, LPARALLEL=.FALSE.)
#ifdef MPI
    ENDIF
#endif
!=======================================================================
! count number of plane wave coefficients
! and allocate required arrays
!=======================================================================
    NRPLWV=0

    DO NK=1,WDES%NKPTS
       IND=0
       DO NC=1,GRID%RC%NCOL
          N2=GRID%RC%I2(NC) ; G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
          N3=GRID%RC%I3(NC) ; G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
          DO N1=1,GRID%RC%NROW

             G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))

             GIX= (G1*BI(1,1)+G2*BI(1,2)+G3*BI(1,3)) *TPI
             GIY= (G1*BI(2,1)+G2*BI(2,2)+G3*BI(2,3)) *TPI
             GIZ= (G1*BI(3,1)+G2*BI(3,2)+G3*BI(3,3)) *TPI

             ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))

             ! exclude some components for gamma-only version (C(G)=C*(-G))
             IF (GRID%NGX/=GRID%NGX_rd) THEN
                IF (GRID%LPCTX(N1)==0 .AND. GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTX(N1)==0 .AND. GRID%LPCTY(N2)==0 .AND. GRID%LPCTZ(N3)<0) CYCLE
             ENDIF
             IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                IF (GRID%LPCTZ(N3)==0 .AND. GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTZ(N3)==0 .AND. GRID%LPCTY(N2)==0 .AND. GRID%LPCTX(N1)<0) CYCLE
             ENDIF
             IF(ENERGI<WDES%ENMAX) THEN
                IND=IND+1
             ENDIF
          ENDDO
       ENDDO
       NRPLWV=MAX(NRPLWV,IND)
    ENDDO
    ! make WDES%NRPLWV dividable by NB_PAR

#ifdef MPI
    ! ok this is tricky
    ! if symmetry operations are applied, we might have too  few plane wave coefficients
    ! in the parallel mode at k-points outside the IRZ, whenever HF or symmetry routines are used
    IF (WDES%NKDIM /= WDES%NKPTS .AND. GRID%COMM%NCPU/=1 ) THEN
!       NRPLWV=NRPLWV+8*GRID%COMM%NCPU  ! safeguard this certainly needs to be improved
! gK symmetry can usually not yield more PW coefficients than on any other core
! so max it
       CALLMPI( M_max_i(GRID%COMM, NRPLWV, 1 ))
       NRPLWV=NRPLWV+8
    ENDIF
#endif
    WDES%NRPLWV=((NRPLWV+WDES%NB_PAR-1)/WDES%NB_PAR)*WDES%NB_PAR
    WDES%NGDIM=WDES%NRPLWV
    WDES%NRPLWV = WDES%NRPLWV*WDES%NRSPINORS
    WDES%NRPLWV_RED=WDES%NRPLWV/WDES%NB_PAR

    IF (WDES%NB_PAR /=1) THEN
       WDES%DO_REDIS=.TRUE.
    ENDIF

    !    CALL  MAKE_STRIDE(WDES%NRPLWV)

    GRID%MPLWV=__align_DIM(MAX(GRID%RC%NALLOC ,GRID%IN%NALLOC , GRID%RL%NALLOC))

#ifdef MPI
    NODE_ME=GRID%COMM%NODE_ME
    IONODE =GRID%COMM%IONODE
#endif
    DWRITE 'gen_layout',NODE_ME,GRID%RC%NALLOC ,GRID%IN%NALLOC , GRID%RL%NALLOC

    CALL ALLOCWDES(WDES,LSETUP)
    WDES%NPLWKP=0
    WDES%NGVECTOR=0
    WDES%NGVECTOR_POS=0

    IF      (WDES%ISPIN==1  .AND. .NOT. WDES%LNONCOLLINEAR ) THEN
       WDES%NCDIJ=1 
    ELSE IF (WDES%ISPIN==2) THEN
       WDES%NCDIJ=2
    ELSE IF (WDES%ISPIN==1  .AND. WDES%LNONCOLLINEAR ) THEN
       WDES%NCDIJ=4 
    ELSE
       WRITE(*,*) 'internal error: can not set NCDIJ'
       STOP
    ENDIF

! set WDES%AT_GAMMA
! this flag determines whether a specific k-point corresponds to Gamma
! if this is the case the code might do some special treatment allowed
! only at Gamma (e.g. orbitals are real)
! presently this is only supported in few places
    DO NK=1,WDES%NKDIM
       IF (ABS(SUM(WDES%VKPT(:,NK)**2)) < G2ZERO .AND. .NOT. WDES%LNONCOLLINEAR) THEN
          WDES%AT_GAMMA(NK)=.TRUE.
       ELSE
          WDES%AT_GAMMA(NK)=.FALSE.
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE GEN_LAYOUT


!*************************SUBROUTINE GEN_INDEX_GAMMA ******************
! 
! the following subroutine creates an index array 
! to store orbital coefficients strictly compatible to
! the serial Gamma only version compiled using -DWNGXhalf
! this requires two arrays
!  )  one telling where to store a coefficient
!  )  and one telling whether the coefficient needs to be
!     complex conjugated 
!
!***********************************************************************


  SUBROUTINE GEN_INDEX_GAMMA(WDES1, B, ENMAX, INDEX, LCONJG )
    USE constant
    IMPLICIT NONE
    TYPE (wavedes1)    WDES1
    REAL(q) :: B(3,3)   ! reciprocal lattice
    REAL(q) :: ENMAX
    LOGICAL, POINTER :: LCONJG(:)
    INTEGER, POINTER :: INDEX(:)
  ! local
    INTEGER :: M, N1, N2, N3, NUSED, IND
    INTEGER, ALLOCATABLE  :: INDEX3D(:,:,:)
    REAL(q) :: G1, G2, G3, GIX, GIY, GIZ, ENERGI

    IF (ASSOCIATED(INDEX) .OR. ASSOCIATED(LCONJG)) THEN
       WRITE(*,*) 'internal error in GEN_INDEX_GAMMA: not null pointer passed down'
       STOP
    ENDIF

    ALLOCATE( INDEX(WDES1%NGVECTOR), LCONJG(WDES1%NGVECTOR))


    ALLOCATE( INDEX3D(0:WDES1%GRID%NGX/2, -WDES1%GRID%NGY/2:WDES1%GRID%NGY/2 , -WDES1%GRID%NGZ/2:WDES1%GRID%NGZ/2))
  ! loop over all grid points 
  ! assume reduction in x direction
    INDEX3D=0
    NUSED=0
    DO N3=1,WDES1%GRID%NGZ
       DO N2=1,WDES1%GRID%NGY
          G3=(WDES1%GRID%LPCTZ(N3)+WDES1%VKPT(3))
          G2=(WDES1%GRID%LPCTY(N2)+WDES1%VKPT(2))

          row: DO N1=1,WDES1%GRID%NGX/2+1

             G1=(WDES1%GRID%LPCTX(N1)+WDES1%VKPT(1))
             GIX= (G1*B(1,1)+G2*B(1,2)+G3*B(1,3)) *TPI
             GIY= (G1*B(2,1)+G2*B(2,2)+G3*B(2,3)) *TPI
             GIZ= (G1*B(3,1)+G2*B(3,2)+G3*B(3,3)) *TPI

             ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))
             ! exclude some components for gamma-only version (C(G)=C*(-G))

             IF (WDES1%GRID%LPCTX(N1)==0 .AND.WDES1%GRID%LPCTY(N2)<0) CYCLE
             IF (WDES1%GRID%LPCTX(N1)==0 .AND.WDES1%GRID%LPCTY(N2)==0 .AND.WDES1%GRID%LPCTZ(N3)<0) CYCLE

             IF(ENERGI<ENMAX) THEN
                NUSED=NUSED+1
                INDEX3D(WDES1%GRID%LPCTX(N1) ,WDES1%GRID%LPCTY(N2), WDES1%GRID%LPCTZ(N3))=NUSED
             ENDIF
          ENDDO row
       ENDDO
    ENDDO

    DO IND=1,WDES1%NGVECTOR
       N1=WDES1%IGX(IND)
       N2=WDES1%IGY(IND)
       N3=WDES1%IGZ(IND)
       IF (N1<0) THEN
          LCONJG(IND)=.TRUE.
          INDEX (IND)= INDEX3D(-N1 ,-N2, -N3)
       ELSE IF (N1==0) THEN
          ! try first conventional one
          INDEX (IND)= INDEX3D(N1 ,N2, N3)
          IF (INDEX (IND) >0 ) THEN
             ! ok no conjugation
             LCONJG(IND)=.FALSE.
          ELSE
             ! try inverted one
             INDEX (IND)= INDEX3D(-N1 ,-N2, -N3)
             LCONJG(IND)=.TRUE.
          ENDIF
       ELSE
          ! must be conventional one
          LCONJG(IND)=.FALSE.
          INDEX (IND)= INDEX3D(N1 ,N2, N3)
       ENDIF
       IF (INDEX(IND)==0) THEN
          WRITE(*,*) 'internal error in  GEN_INDEX_GAMMA: can not find appropriate entry in INDEX3d', N1, N2, N3, INDEX3D(N1, N2, N3), INDEX3D(-N1, -N2,-N3)
          STOP
       ENDIF
    ENDDO

    DEALLOCATE(INDEX3D)
  END SUBROUTINE GEN_INDEX_GAMMA


  SUBROUTINE FREE_INDEX_GAMMA(INDEX, LCONJG)
    LOGICAL, POINTER :: LCONJG(:)
    INTEGER, POINTER :: INDEX(:)

    DEALLOCATE(INDEX)
    DEALLOCATE(LCONJG)
    
    NULLIFY(INDEX)
    NULLIFY(LCONJG)
  END SUBROUTINE FREE_INDEX_GAMMA

END MODULE


!=======================================================================
! sorts RA in descending order, and rearanges an index array RB
! seems to be a quicksort, but I am not sure
! subroutine writen by Florian Kirchhof
!=======================================================================

  SUBROUTINE SORT_REDIS(N,RA,RB)
    INTEGER RA(N),RB(N)
    INTEGER RRA,RRB

    IF (N<=1) RETURN

    L=N/2+1
    IR=N
10  CONTINUE
    IF(L.GT.1)THEN
       L=L-1
       RRA=RA(L)
       RRB=RB(L)
    ELSE
       RRA=RA(IR)
       RRB=RB(IR)
       RA(IR)=RA(1)
       RB(IR)=RB(1)
       IR=IR-1
       IF(IR.EQ.1)THEN
          RA(1)=RRA
          RB(1)=RRB
          RETURN
       ENDIF
    ENDIF
    I=L
    J=L+L
20  IF(J.LE.IR)THEN
       IF(J.LT.IR)THEN
          IF(RA(J).GT.RA(J+1))J=J+1
       ENDIF
       IF(RRA.GT.RA(J))THEN
          RA(I)=RA(J)
          RB(I)=RB(J)
          I=J
          J=J+J
       ELSE
          J=IR+1
       ENDIF
       GO TO 20
    ENDIF
    RA(I)=RRA
    RB(I)=RRB
    GO TO 10
  END SUBROUTINE SORT_REDIS

!=======================================================================
! sorts RA in ascending order, and rearanges an index array RB
! seems to be a quicksort, by I am not sure
! subroutine writen by Florian Kirchhof
!=======================================================================

  SUBROUTINE SORT_REDIS_ASC(N,RA,RB)
    INTEGER RA(N),RB(N)
    INTEGER RRA,RRB

    IF (N<=1) RETURN

    L=N/2+1
    IR=N
10  CONTINUE
    IF(L.GT.1)THEN
       L=L-1
       RRA=RA(L)
       RRB=RB(L)
    ELSE
       RRA=RA(IR)
       RRB=RB(IR)
       RA(IR)=RA(1)
       RB(IR)=RB(1)
       IR=IR-1
       IF(IR.EQ.1)THEN
          RA(1)=RRA
          RB(1)=RRB
          RETURN
       ENDIF
    ENDIF
    I=L
    J=L+L
20  IF(J.LE.IR)THEN
       IF(J.LT.IR)THEN
          IF(RA(J).LT.RA(J+1))J=J+1
       ENDIF
       IF(RRA.LT.RA(J))THEN
          RA(I)=RA(J)
          RB(I)=RB(J)
          I=J
          J=J+J
       ELSE
          J=IR+1
       ENDIF
       GO TO 20
    ENDIF
    RA(I)=RRA
    RB(I)=RRB
    GO TO 10
  END SUBROUTINE SORT_REDIS_ASC

!*************************SUBROUTINE COUNT_ROWS ************************
!
! this subroutine counts the total number of plane waves contained
! within the cutoff sphere up to (but excluding) a certain column
! this array is required to find out which index a certain column would
! have in the serial version
! mind the index is 0 based
! the total number of plane waves is returned in NUSED
! USED_POINTS returns the number of plane wave coefficients
! up to column N1,N3
!
!***********************************************************************

  SUBROUTINE  COUNT_ROWS(GRID, WDES, BI, NK, USED_POINTS, NUSED)
    USE mgrid
    USE wave
    USE constant
    USE base
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    TYPE (wavedes)     WDES
    DIMENSION BI(3,3)

    INTEGER :: USED_POINTS(GRID%NGY,GRID%NGZ)

    NUSED=0
    DO N3=1,GRID%NGZ_rd
       DO N2=1,GRID%NGY
          G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
          G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
          USED_POINTS(N2,N3)=NUSED

          row: DO N1=1,GRID%NGX_rd

             G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
             GIX= (G1*BI(1,1)+G2*BI(1,2)+G3*BI(1,3)) *TPI
             GIY= (G1*BI(2,1)+G2*BI(2,2)+G3*BI(2,3)) *TPI
             GIZ= (G1*BI(3,1)+G2*BI(3,2)+G3*BI(3,3)) *TPI

             ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))
             ! exclude some components for gamma-only version (C(G)=C*(-G))
             IF (GRID%NGX/=GRID%NGX_rd) THEN
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTZ(N3)<0) CYCLE
             ENDIF
             IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE
             ENDIF
             IF(ENERGI<WDES%ENMAX) THEN
                NUSED=NUSED+1
             ENDIF
          ENDDO row
       ENDDO
    ENDDO
  END SUBROUTINE COUNT_ROWS

!*************************SUBROUTINE REPAD_INDEX_ARRAY  ****************
!
! this subroutine calculates two index arrays that allow
! to "restore" a plane wave array from an old cutoff and lattice
! to a new one
! this operation works only if the orbitals are stored in
! the serial layout on one single core (not parallel)
! this is usually the case when orbitals are read in
! in the Gamma point only version the orbitals are presently
! stored in a way that is always compatible to the seriel
! version (implying that the grid is reduced in the x direction
!
!***********************************************************************

  SUBROUTINE  REPAD_INDEX_ARRAY(GRID, VKPT, VKPTI, B,  BI, ENMAX, ENMAXI, & 
       NP, NPI, IND, INDI, INDMAX, IFAIL )
    USE prec
    USE mgrid
    USE constant
    USE base
    IMPLICIT NONE

    TYPE (grid_3d)     GRID  ! grid descriptor
    REAL(q) :: VKPT(3)       ! new k-point
    REAL(q) :: VKPTI(3)      ! old k-point
    REAL(q) ::  B (3,3)      ! new reciprocal lattice constant
    REAL(q) ::  BI(3,3)      ! old reciprocal lattice constant
    REAL(q) ::  ENMAX        ! new cutoff
    REAL(q) ::  ENMAXI       ! old cutoff
    INTEGER ::  NP           ! number of plane wave coefficients old
    INTEGER ::  NPI          ! number of plane wave coefficients new
    ! MIND: NP and NPI must be set by the caller
    INTEGER ::  IND(MAX(NP,NPI))  ! index array new
    INTEGER ::  INDI(MAX(NP,NPI)) ! index array old
    INTEGER ::  INDMAX       ! on return maximum index
    INTEGER ::  IFAIL        ! 0  NP and NPI were correct
    ! 1  NP was incorrect, 2 NPI was incorrect
    ! local
    INTEGER NP_,NPI_,N1,N2,N3
    INTEGER NGZ_rd, NGX_rd
    REAL(q) :: G1,G2,G3, G1I,G2I,G3I, GIX,GIY,GIZ, GX,GY,GZ, ENERGI, ENERG

    IFAIL = 0

    NP_ =0
    NPI_=0
    INDMAX=0

    NGX_rd =  GRID%NGX
    NGZ_rd =  GRID%NGZ
    ! grid reduced: the orbitals are stored compatible to seriel version
    ! hence the reduction is always along x direction
    IF (GRID%NGZ_rd /= GRID%NGZ .OR.  GRID%NGX_rd /=  GRID%NGX) THEN
       NGX_rd =  GRID%NGX/2+1
       NGZ_rd =  GRID%NGZ
    ENDIF

    DO N3=1,NGZ_rd
       DO N2=1,GRID%NGY
          G3=(GRID%LPCTZ(N3)+VKPT(3))
          G2=(GRID%LPCTY(N2)+VKPT(2))
          G3I=(GRID%LPCTZ(N3)+VKPTI(3))
          G2I=(GRID%LPCTY(N2)+VKPTI(2))

          row: DO N1=1,NGX_rd

             G1=(GRID%LPCTX(N1)+VKPT(1))
             G1I=(GRID%LPCTX(N1)+VKPTI(1))

             GIX= (G1I*BI(1,1)+G2I*BI(1,2)+G3I*BI(1,3)) *TPI
             GIY= (G1I*BI(2,1)+G2I*BI(2,2)+G3I*BI(2,3)) *TPI
             GIZ= (G1I*BI(3,1)+G2I*BI(3,2)+G3I*BI(3,3)) *TPI

             GX= (G1*B(1,1)+G2*B(1,2)+G3*B(1,3)) *TPI
             GY= (G1*B(2,1)+G2*B(2,2)+G3*B(2,3)) *TPI
             GZ= (G1*B(3,1)+G2*B(3,2)+G3*B(3,3)) *TPI

             ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))
             ENERG =HSQDTM*( (GX**2)+ (GY**2)+ (GZ**2))

             ! exclude some components for gamma-only version (C(G)=C*(-G))
             IF (GRID%NGX/=NGX_rd) THEN
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTZ(N3)<0) CYCLE
             ENDIF
             IF (GRID%NGZ/=NGZ_rd) THEN
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE
             ENDIF
             IF (ENERG <ENMAX) THEN
                NP_=NP_+1    ! increase index for new array
             ENDIF
             IF (ENERGI<ENMAXI) THEN
                NPI_=NPI_+1  ! increase index for old array
             ENDIF
             IF (ENERG<ENMAX .AND. ENERGI<ENMAXI) THEN
                INDMAX= MIN( MIN(INDMAX+1 ,NP ), NPI) 
                ! increase index, and avoid overrun
                IND(INDMAX) =NP_
                INDI(INDMAX)=NPI_
             ENDIF
          ENDDO row
       ENDDO
    ENDDO

    IF  (NP_ /= NP) THEN
       NP=NP_
       IFAIL=1
    ENDIF
    IF  (NPI_ /= NPI) THEN
       NPI=NPI_
       IFAIL=1
    ENDIF

  END SUBROUTINE REPAD_INDEX_ARRAY

  SUBROUTINE REPAD_WITH_INDEX_ARRAY(INDMAX,IND,INDI, CW, CWI)
    USE prec
    IMPLICIT NONE

    INTEGER INDMAX
    INTEGER ::  IND(INDMAX)  ! index array new
    INTEGER ::  INDI(INDMAX) ! index array old
    COMPLEX(q) :: CW(*),CWI(*)
    ! local
    INTEGER I

    DO I=1,INDMAX
       CW(IND(I))=CWI(INDI(I))
    ENDDO

  END SUBROUTINE REPAD_WITH_INDEX_ARRAY

!*************************SUBROUTINE GEN_INDEX ************************
!
! subroutine GEN_INDEX calculates the following arrays:
! ) the indexing array NINDPW for copying the plane wave coefficients 
!   from the continuous array CW to the column wise layout used for
!   the 3d-FFT
! for LSETUP=.TRUE., additionally the following array are set up:
! ) the kinetic energies of the plane wave basis states are computed
!   (WDES%DATAKE)
! ) the G vector corresponding to each plane wave basis state is stored
!   (WDES%IGX,Y,Z)
!
! ) in the parallel version, the arrays PL_INDEX and PL_COL
!   are set up and stored
!     PL_INDEX(NC,NK) stores the position of a column at
!                     which data is stored in the serial version
!     PL_COL(NC,NK)   number of data in this column
!
! the data layout is based on the initial reciprocal lattice vectors
! stored in BI
!
!***********************************************************************


  SUBROUTINE GEN_INDEX(GRID, WDES, B, BI, IU6, IU0, LSETUP)
    USE prec
    USE mpimy
    USE mgrid
    USE wave
    USE constant
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (grid_3d)     GRID
    TYPE (wavedes)     WDES
    DIMENSION B(3,3),BI(3,3) ! current lattice, and initial lattice
    LOGICAL LSETUP
! local
    INTEGER, ALLOCATABLE :: USED_POINTS(:,:)
    INTEGER :: NG_default(3)
#ifdef MPI
    INTEGER :: NGVECTOR_POS(GRID%COMM%NCPU)
!    NODE_ME= WDES%COMM%NODE_ME
!    IONODE = WDES%COMM%IONODE
#endif
!=======================================================================
! now setup the required quantities
!=======================================================================
    TESTMX=0.0_q

    IXMAX=0
    IYMAX=0
    IZMAX=0
    IXMIN=0
    IYMIN=0
    IZMIN=0

    ALLOCATE(USED_POINTS(GRID%NGY,GRID%NGZ))

    kpoint: DO NK=1,WDES%NKPTS
       NLBOXI=0
       IND=1
       CALL COUNT_ROWS(GRID,WDES,BI,NK, USED_POINTS,NUSED)

       IF (WDES%LNONCOLLINEAR) THEN
          NUSED=NUSED*WDES%NRSPINORS
       ENDIF

       col: DO NC=1,GRID%RC%NCOL
          N2=GRID%RC%I2(NC) ; G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
          N3=GRID%RC%I3(NC) ; G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
          IN_THIS_ROW=0

          row: DO N1=1,GRID%RC%NROW
             NLBOXI=NLBOXI+1

             G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
             GX= (G1*B(1,1)+G2*B(1,2)+G3*B(1,3)) *TPI
             GY= (G1*B(2,1)+G2*B(2,2)+G3*B(2,3)) *TPI
             GZ= (G1*B(3,1)+G2*B(3,2)+G3*B(3,3)) *TPI
             ENERG =HSQDTM*((GX**2)+(GY**2)+(GZ**2))
             ! kinetic energy of plane wave components of spin up part of the spinor
             GX= ((G1-WDES%QSPIRAL(1)/2)*B(1,1)+(G2-WDES%QSPIRAL(2)/2)*B(1,2)+(G3-WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
             GY= ((G1-WDES%QSPIRAL(1)/2)*B(2,1)+(G2-WDES%QSPIRAL(2)/2)*B(2,2)+(G3-WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
             GZ= ((G1-WDES%QSPIRAL(1)/2)*B(3,1)+(G2-WDES%QSPIRAL(2)/2)*B(3,2)+(G3-WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
             ENERGUP=HSQDTM*((GX**2)+(GY**2)+(GZ**2))
             ! kinetic energy of plane wave components of spin up part of the spinor
             GX= ((G1+WDES%QSPIRAL(1)/2)*B(1,1)+(G2+WDES%QSPIRAL(2)/2)*B(1,2)+(G3+WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
             GY= ((G1+WDES%QSPIRAL(1)/2)*B(2,1)+(G2+WDES%QSPIRAL(2)/2)*B(2,2)+(G3+WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
             GZ= ((G1+WDES%QSPIRAL(1)/2)*B(3,1)+(G2+WDES%QSPIRAL(2)/2)*B(3,2)+(G3+WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
             ENERGDN=HSQDTM*((GX**2)+(GY**2)+(GZ**2))
             GIX= (G1*BI(1,1)+G2*BI(1,2)+G3*BI(1,3)) *TPI
             GIY= (G1*BI(2,1)+G2*BI(2,2)+G3*BI(2,3)) *TPI
             GIZ= (G1*BI(3,1)+G2*BI(3,2)+G3*BI(3,3)) *TPI

             ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))
             TESTMX=MAX(TESTMX,ENERGI)
             !
             ! exclude some components for gamma-only version (C(G)=C*(-G))
             IF (GRID%NGX/=GRID%NGX_rd) THEN
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)<0) CYCLE row
                IF (GRID%LPCTX(N1)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTZ(N3)<0) CYCLE row
             ENDIF
             IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE row
                IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE row
             ENDIF
             ! check to see if the kinetic energy of the plane wave is less than
             ! ENMAX in which case the plane wave is included in the set of basis
             ! states for this k point
             IF(ENERGI<WDES%ENMAX) THEN
                IN_THIS_ROW=IN_THIS_ROW+1

                IXMAX=MAX(IXMAX,GRID%LPCTX(N1))
                IYMAX=MAX(IYMAX,GRID%LPCTY(N2))
                IZMAX=MAX(IZMAX,GRID%LPCTZ(N3))
                IXMIN=MIN(IXMIN,GRID%LPCTX(N1))
                IYMIN=MIN(IYMIN,GRID%LPCTY(N2))
                IZMIN=MIN(IZMIN,GRID%LPCTZ(N3))

                IF (LSETUP) THEN
                   WDES%IGX(IND,NK)=GRID%LPCTX(N1)
                   WDES%IGY(IND,NK)=GRID%LPCTY(N2)
                   WDES%IGZ(IND,NK)=GRID%LPCTZ(N3)
                   WDES%DATAKE(IND,1,NK)=ENERGUP
                   WDES%DATAKE(IND,2,NK)=ENERGDN
                ENDIF
                WDES%NINDPW(IND,NK)=NLBOXI
                IND=IND+1
             ENDIF
          ENDDO row
          IF (WDES%NCOL /= 0) THEN
             WDES%PL_INDEX(NC,NK)=USED_POINTS(N2,N3)
             WDES%PL_COL  (NC,NK)=IN_THIS_ROW
          ENDIF
       ENDDO col
!=======================================================================
! check to see if there are less than NRPLWV basis states at this kpoint
! if not stop
!=======================================================================
       IND=IND-1

       ! at this point IND is set to the number of plane wave coefficients
       ! for the current k-point
       IND=IND*WDES%NRSPINORS

       IF(WDES%NRPLWV < IND) THEN
          WRITE(*,*)'internal ERROR: GEN_INDEX: number of plane waves is too large', &
               IND,WDES%NRPLWV
          STOP
       ENDIF
       IF (WDES%NPLWKP(NK)/=0 .AND. WDES%NPLWKP(NK)/=IND) THEN
          WRITE(*,*) 'GEN_INDEX: number of plane waves is incorrect', &
               ' propably incorrect WAVECAR read in'
          STOP
       ENDIF
       WDES%NPLWKP(NK)=IND
       WDES%NPLWKP_TOT(NK)=IND
       WDES%NGVECTOR(NK)=WDES%NPLWKP(NK)/WDES%NRSPINORS

       CALLMPI( M_sum_i(GRID%COMM,WDES%NPLWKP_TOT(NK)  ,1))

       IF (WDES%NPLWKP_TOT(NK) /= NUSED) THEN
          WRITE(*,*)'internal ERROR 2: GEN_INDEX:',WDES%NPLWKP_TOT(NK),NUSED
          STOP
       ENDIF

       WDES%NGVECTOR_POS(NK)=1
#ifdef MPI
       ! NGVECTOR_POS stores the sum of NGVECTOR up to but not including
       ! the current node 
       ! this is required to efficiently merge plane wave coefficient over nodes
       NGVECTOR_POS=0
       NGVECTOR_POS(GRID%COMM%NODE_ME)=WDES%NGVECTOR(NK)

       CALLMPI( M_sum_i(GRID%COMM,NGVECTOR_POS  ,GRID%COMM%NCPU))
       
       WDES%NGVECTOR_POS(NK)=1
       DO I=1,GRID%COMM%NODE_ME-1
          WDES%NGVECTOR_POS(NK)=WDES%NGVECTOR_POS(NK)+NGVECTOR_POS(I)
       ENDDO
#endif

       IF (IU6>=0) WRITE(IU6,10)NK,WDES%VKPT(1:3,NK),WDES%NPLWKP_TOT(NK)
    ENDDO kpoint

    DEALLOCATE(USED_POINTS)
!=======================================================================
! write maximum index for each direction and give optimal values for
! NGX NGY and NGZ
!=======================================================================
10  FORMAT(' k-point ',I2,' :  ',3F7.4,'  plane waves: ',I7)

    NPLMAX=0
    DO NK=1,WDES%NKPTS
       NPLMAX=MAX( WDES%NPLWKP_TOT(NK),NPLMAX)
    ENDDO

    NPLMAX_LOC=0
    NPLMIN_LOC=-NPLMAX
    DO NK=1,WDES%NKPTS
       NPLMAX_LOC=MAX( WDES%NPLWKP(NK),NPLMAX_LOC)
       NPLMIN_LOC=MAX(-WDES%NPLWKP(NK),NPLMIN_LOC)
    ENDDO

    CALLMPI( M_max_i(GRID%COMM,NPLMAX_LOC ,1))
    CALLMPI( M_max_i(GRID%COMM,NPLMIN_LOC ,1))
    NPLMIN_LOC=-NPLMIN_LOC

    IXMIN=-IXMIN
    IYMIN=-IYMIN
    IZMIN=-IZMIN
    CALLMPI( M_max_i(GRID%COMM,IXMAX  ,1))
    CALLMPI( M_max_i(GRID%COMM,IYMAX  ,1))
    CALLMPI( M_max_i(GRID%COMM,IZMAX  ,1))
    CALLMPI( M_max_i(GRID%COMM,IXMIN  ,1))
    CALLMPI( M_max_i(GRID%COMM,IYMIN  ,1))
    CALLMPI( M_max_i(GRID%COMM,IZMIN  ,1))
    IXMIN=-IXMIN
    IYMIN=-IYMIN
    IZMIN=-IZMIN

! set the maximum number of bands k-point dependent
    DO NK=1,WDES%NKPTS
       IF (WDES%LGAMMA) THEN
          WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK)*2-1)
       ELSE
          WDES%NB_TOTK(NK,:)=MIN(WDES%NB_TOT,WDES%NPLWKP_TOT(NK))
       ENDIF
    ENDDO

    IF (IU6>=0) THEN

#ifdef MPI
       WRITE(IU6,21) NPLMAX_LOC,NPLMIN_LOC
21     FORMAT(/' maximum and minimum number of plane-waves per node : ',2I9)
#endif

       WRITE(IU6,20) NPLMAX,IXMAX,IYMAX,IZMAX,IXMIN,IYMIN,IZMIN
20     FORMAT(/' maximum number of plane-waves: ',I9/ &
            &        ' maximum index in each direction: ',/ &
            &        '   IXMAX=',I5,'   IYMAX=',I5,'   IZMAX=',I5/ &
            &        '   IXMIN=',I5,'   IYMIN=',I5,'   IZMIN=',I5/)

       ! gamma-only version searches only half the grid
       IF (IXMIN==0) IXMIN=-IXMAX
       IF (IYMIN==0) IYMIN=-IYMAX
       IF (IZMIN==0) IZMIN=-IZMAX

       ! these is the behaviour up to vasp.5.4.4, but yields potentially 
       ! non-symmetric grids, and is somewhat more stringent than the default FFT grids
       NG_default(1)=(IXMAX-IXMIN+1)*2
       NG_default(2)=(IYMAX-IYMIN+1)*2
       NG_default(3)=(IZMAX-IZMIN+1)*2

       CALL FFTCHK(NG_default)

       IF (.FALSE.) THEN
          WRITE(IU6,*)'The following grids will avoid any aliasing or wrap around errors in the Hartree energy'
          WRITE(IU6,*)' - symmetry arguments have not been applied'
          WRITE(IU6,*)' - exchange correlation energies might require even more grid points'
          WRITE(IU6,*)' - we recommend to set PREC=Normal or Accurate and rely on VASP defaults'

          IWARN=0
          IF (NG_default(1)>GRID%NGX) THEN
             IF (IU6>=0) WRITE(IU6,30)'NGX',NG_default(1)
             IWARN=1
          ELSE
             IF (IU6>=0) WRITE(IU6,31)'NGX',NG_default(1)
          ENDIF
          
          IF (NG_default(2)>GRID%NGY) THEN
             IF (IU6>=0) WRITE(IU6,30)'NGY',NG_default(2)
             IWARN=1
          ELSE
             IF (IU6>=0) WRITE(IU6,31)'NGY',NG_default(2)
          ENDIF
          
          IF (NG_default(3)>GRID%NGZ) THEN
             IF (IU6>=0) WRITE(IU6,30)'NGZ',NG_default(3)
             IWARN=1
          ELSE
             IF (IU6>=0) WRITE(IU6,31)'NGZ',NG_default(3)
          ENDIF
30     FORMAT(' WARNING: aliasing errors must be expected', &
            &       ' set ',A3,' to ',I5,' to avoid them')
31     FORMAT(' ',A3,' is ok and might be reduce to ',I5)
       ENDIF

    ENDIF

    RETURN
  END SUBROUTINE GEN_INDEX


  SUBROUTINE SET_DATAKE(WDES, B )
    USE wave
    USE constant
    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (wavedes)     WDES
    REAL(q) B(3,3)

    DO NK=1,WDES%NKPTS
      DO IND=1,WDES%NGVECTOR(NK)
         G1=(WDES%IGX(IND,NK)+WDES%VKPT(1,NK))
         G2=(WDES%IGY(IND,NK)+WDES%VKPT(2,NK))
         G3=(WDES%IGZ(IND,NK)+WDES%VKPT(3,NK))
         
         ! kinetic energy of plane wave components of spin up part of the spinor
         GX= ((G1-WDES%QSPIRAL(1)/2)*B(1,1)+(G2-WDES%QSPIRAL(2)/2)*B(1,2)+(G3-WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
         GY= ((G1-WDES%QSPIRAL(1)/2)*B(2,1)+(G2-WDES%QSPIRAL(2)/2)*B(2,2)+(G3-WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
         GZ= ((G1-WDES%QSPIRAL(1)/2)*B(3,1)+(G2-WDES%QSPIRAL(2)/2)*B(3,2)+(G3-WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
         ENERGUP=HSQDTM*((GX**2)+(GY**2)+(GZ**2))

         ! kinetic energy of plane wave components of spin up part of the spinor
         GX= ((G1+WDES%QSPIRAL(1)/2)*B(1,1)+(G2+WDES%QSPIRAL(2)/2)*B(1,2)+(G3+WDES%QSPIRAL(3)/2)*B(1,3)) *TPI
         GY= ((G1+WDES%QSPIRAL(1)/2)*B(2,1)+(G2+WDES%QSPIRAL(2)/2)*B(2,2)+(G3+WDES%QSPIRAL(3)/2)*B(2,3)) *TPI
         GZ= ((G1+WDES%QSPIRAL(1)/2)*B(3,1)+(G2+WDES%QSPIRAL(2)/2)*B(3,2)+(G3+WDES%QSPIRAL(3)/2)*B(3,3)) *TPI
         ENERGDN=HSQDTM*((GX**2)+(GY**2)+(GZ**2))
         WDES%DATAKE(IND,1,NK)=ENERGUP
         WDES%DATAKE(IND,2,NK)=ENERGDN
      ENDDO
   ENDDO
 END SUBROUTINE SET_DATAKE


      
      
!***********************************************************************
!      
! restart a spin spiral calculations from a WAVECAR
! obtained at a different q-vector or using a different value for ENINI
!      
!***********************************************************************

  SUBROUTINE CLEANWAV(WDES,W,ENINI)

    USE prec
    USE constant
    USE wave

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    TYPE (wavedes1)    WDES1

    spin:   DO I=1,WDES%ISPIN
       kpoint: DO NK=1,WDES%NKPTS

          NPL=WDES%NGVECTOR(NK)
          band: DO NB=1,WDES%NBANDS
             spinor: DO ISPINOR=0,WDES%NRSPINORS-1

                DO M=1,NPL
                   IF(WDES%DATAKE(M,ISPINOR+1,NK)>=ENINI) W%CPTWFP(M+NPL*ISPINOR,NB,NK,I)=0
                ENDDO

             ENDDO spinor
          ENDDO band
       ENDDO kpoint
    ENDDO spin

    RETURN
  END SUBROUTINE CLEANWAV
