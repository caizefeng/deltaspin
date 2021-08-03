#include "symbol.inc"
!************************************************************************
! RCS:  $Id: wave.F,v 1.6 2002/08/14 13:59:43 kresse Exp $
!
!  this module contains a subset of routines to cache wavefunctions
!  in real space and their projectors
!  the cache structure uses a reference counter to count how often
!  a wavefunction is stored in the cache.
!  Upon removal the counter is usually decreased by one, and the 
!  cache-line is cleared if the counter is zero
!  A routine to clear the cache, as well as one to check whether all
!  cache-lines have been cleared exists
!
!  Implementation:
!  A simple implementation is chosen. The cache is structured in blocks
!  where each block stores NBLK entries
!  NBLK such block exist.
!
!
!***********************************************************************

MODULE WAVE_CACHER
  USE wave_high
  USE fock
  IMPLICIT NONE
  
  TYPE wave_cache_block
     INTEGER :: NFREE                ! free entries in this block (-1 signifies not allocated)
     INTEGER, POINTER :: ICNT(:)     ! how often is entry used
     COMPLEX(q),POINTER:: CR(:,:)    ! wavefunction in real space
     GDEF      ,POINTER:: CPROJ(:,:) ! projector (complex)
  END TYPE wave_cache_block

  TYPE wave_cache
     INTEGER :: NBLK                 ! block size of cache
     ! the following arrays depend on the band index (NBANDS), k-point (NK) and spin (ISP)
     INTEGER, POINTER :: IBLK(:,:,:) ! integer indicating in which block a wavefunction is store
     INTEGER, POINTER :: IPOS(:,:,:) ! integer indicating where in the block a wavefunction is store
     ! NBLK wave_cache_block arrays
     TYPE( wave_cache_block), POINTER :: C(:)
     COMPLEX(q), POINTER :: CPHASE(:,:)! cache for phase factor at each k-point
     LOGICAL, POINTER ::  LPHASE(:)  ! phase factor required or not

     ! this handle stores simga_c phi_n
     ! 
     INTEGER :: NBLOW  ! lowest  band entered in FH
     INTEGER :: NBHIGH ! highest band entered in FH
     INTEGER :: NB_PAR ! 
     TYPE(fock_handle), POINTER:: FH(:)
  END TYPE wave_cache

  REAL(q) :: MAX_MEMORY_CACHER

  GDEF, ALLOCATABLE :: CHF(:,:,:,:) ! matrix to store search direction for subspace rotation


CONTAINS

!***********************************************************************
!
! allocate the chacher structure and initialize everything to zero
!
!
!***********************************************************************

  SUBROUTINE ALLOCATE_CACHER(WDES, WCACHE, LMDIM, NBANDSGW)
    USE openmp, ONLY : omp_nthreads
    IMPLICIT NONE
    TYPE (wavedes)     WDES
    TYPE (wave_cache), POINTER ::  WCACHE
    INTEGER :: LMDIM, NBANDSGW
  ! local
    INTEGER :: NBLK, N

    ALLOCATE(WCACHE)

    NBLK=SQRT(REAL(WDES%NBANDS* WDES%NKPTS * WDES%ISPIN-1,q))+1
    IF (NBLK*NBLK < WDES%NBANDS* WDES%NKPTS * WDES%ISPIN) THEN
       WRITE(*,*) 'internal error in ALLOCATE_CACHER: cache allocation failed'
       STOP
    ENDIF

    ! allocate WCACHE structure
    WCACHE%NBLK=NBLK
    ALLOCATE( WCACHE%C(NBLK))
    
    ! signify that nothing is allocated
    DO N=1, NBLK
       WCACHE%C(N)%NFREE=-1
    ENDDO

    ! allocate pointers
    ALLOCATE(WCACHE%IBLK(WDES%NBANDS, WDES%NKPTS, WDES%ISPIN), & 
             WCACHE%IPOS(WDES%NBANDS, WDES%NKPTS, WDES%ISPIN), & 
             WCACHE%LPHASE(WDES%NKPTS))

    WCACHE%IBLK=0
    WCACHE%IPOS=0
    WCACHE%LPHASE=.FALSE.
    NULLIFY( WCACHE%CPHASE)

    ALLOCATE(WCACHE%FH(omp_nthreads))
    CALL ALLOCATE_FOCK_HANDLE(WCACHE%FH(1),LMDIM,NBANDSGW)

    WCACHE%NBLOW =-1
    WCACHE%NBHIGH=NBANDSGW

  END SUBROUTINE ALLOCATE_CACHER


!***********************************************************************
!
! set lower and upper band index in the cache structure
! these indices are local band indices and correspond to the 
! maximum and minimum band for which sigma |phi_n> is calculated
!
!***********************************************************************

  SUBROUTINE NBLOW_CACHER(WCACHE, NBLOW, NBHIGH, NB_PAR)
    USE ini
    IMPLICIT NONE
    TYPE (wave_cache), POINTER :: WCACHE
    INTEGER :: NBLOW, NBHIGH, NB_PAR

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    WCACHE%NB_PAR = NB_PAR
    WCACHE%NBLOW  = NBLOW
    WCACHE%NBHIGH = NBHIGH

  END SUBROUTINE NBLOW_CACHER


!***********************************************************************
!
! deallocate cache structure
!
!
!***********************************************************************

  SUBROUTINE DEALLOCATE_CACHER(WCACHE)
    USE ini
    USE openmp, ONLY : omp_nthreads
    IMPLICIT NONE
    TYPE (wave_cache), POINTER :: WCACHE
    ! local
    INTEGER IBLK

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    CALL DEALLOCATE_FOCK_HANDLE(WCACHE%FH(1))
    DEALLOCATE(WCACHE%FH)

    IF (ASSOCIATED(WCACHE%CPHASE)) THEN
       DEALLOCATE(WCACHE%CPHASE)
    ENDIF

    DO IBLK=1,WCACHE%NBLK

       ! found an allocated block
       IF (WCACHE%C(IBLK)%NFREE/=-1) THEN
          CALL DEREGISTER_ALLOCATE(16._q*SIZE(WCACHE%C(IBLK)%CR), "cacher")

          DEALLOCATE(WCACHE%C(IBLK)%CR, WCACHE%C(IBLK)%CPROJ, WCACHE%C(IBLK)%ICNT)
          WCACHE%C(IBLK)%NFREE=-1
       ENDIF
    ENDDO
    DEALLOCATE(WCACHE%C)

    ! deallocate pointers
    DEALLOCATE(WCACHE%IBLK, WCACHE%IPOS, WCACHE%LPHASE)
    DEALLOCATE(WCACHE)
    NULLIFY(WCACHE)

  END SUBROUTINE DEALLOCATE_CACHER


!***********************************************************************
!
! enter a wavefunction with index NBANDS, NK, ISP in the structure
!
!
!***********************************************************************

  SUBROUTINE STORE_CACHER(WCACHE, NB, NK, ISP, W)
    IMPLICIT NONE
    TYPE (wave_cache), POINTER ::  WCACHE
    INTEGER :: NB, NK, ISP
    TYPE (wavefun1) :: W
    ! local
    INTEGER :: IBLK, IPOS

    PROFILING_START('store_cacher')

    IF (.NOT. ASSOCIATED(WCACHE)) THEN
       PROFILING_STOP('store_cacher')
       RETURN
    ENDIF
    IF (WCACHE%IBLK(NB, NK, ISP)==0) THEN

    ! new entry
       CALL FIND_NEW_CACHER( WCACHE, IBLK, IPOS, W)
       WCACHE%C(IBLK)%CR(:,IPOS)   =W%CR
       WCACHE%C(IBLK)%CPROJ(:,IPOS)=W%CPROJ
       WCACHE%C(IBLK)%ICNT(IPOS)=1
       WCACHE%C(IBLK)%NFREE=WCACHE%C(IBLK)%NFREE-1

       WCACHE%IPOS(NB, NK, ISP)=IPOS
       WCACHE%IBLK(NB, NK, ISP)=IBLK
    ELSE

    ! entry already exists increase reference counter
       IBLK=WCACHE%IBLK(NB, NK, ISP)
       IPOS=WCACHE%IPOS(NB, NK, ISP)
       WCACHE%C(IBLK)%ICNT(IPOS)=WCACHE%C(IBLK)%ICNT(IPOS)+1

    ENDIF
    PROFILING_STOP('store_cacher')

  END SUBROUTINE STORE_CACHER


!***********************************************************************
!
! store phase factor 
!
!***********************************************************************

  SUBROUTINE STORE_PHASE(WCACHE, NK,  LPHASE, CPHASE)
    IMPLICIT NONE
    TYPE (wave_cache), POINTER ::  WCACHE
    LOGICAL LPHASE
    COMPLEX(q) :: CPHASE(:)
    INTEGER :: NK

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    WCACHE%LPHASE(NK) = LPHASE
    IF (LPHASE .AND. .NOT. ASSOCIATED(WCACHE%CPHASE)) THEN
       ALLOCATE(WCACHE%CPHASE(SIZE(CPHASE), SIZE(WCACHE%LPHASE)))
    ENDIF
    IF (LPHASE) THEN
       IF ( SIZE(WCACHE%CPHASE,1) /= SIZE(CPHASE)) THEN
          WRITE(*,*) 'internal error in STORE_PHASE: size changed :',SIZE(WCACHE%CPHASE,1), SIZE(CPHASE)
          STOP
       ENDIF
       WCACHE%CPHASE(:,NK)=CPHASE(:)
    ENDIF
    
  END SUBROUTINE STORE_PHASE

!***********************************************************************
!
! store phase factor 
!
!***********************************************************************

  SUBROUTINE GET_PHASE(WCACHE, NK,  LPHASE, CPHASE)
    IMPLICIT NONE
    TYPE (wave_cache), POINTER ::  WCACHE
    LOGICAL LPHASE
    COMPLEX(q) :: CPHASE(:)
    INTEGER :: NK

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    LPHASE= WCACHE%LPHASE(NK)
    IF (LPHASE) THEN
       CPHASE(:)=WCACHE%CPHASE(:,NK)
    ENDIF
    
  END SUBROUTINE GET_PHASE

!***********************************************************************
!
! store phase factor 
!
!***********************************************************************

  SUBROUTINE GET_PHASE_CONJG(WCACHE, NK,  LPHASE, CPHASE)
    IMPLICIT NONE
    TYPE (wave_cache), POINTER ::  WCACHE
    LOGICAL LPHASE
    COMPLEX(q) :: CPHASE(:)
    INTEGER :: NK

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    LPHASE= WCACHE%LPHASE(NK)
    IF (LPHASE) THEN
       CPHASE(:)=GCONJG(WCACHE%CPHASE(:,NK))
    ENDIF
    
  END SUBROUTINE GET_PHASE_CONJG

!***********************************************************************
!
! find a new empty cache line
!
!***********************************************************************

  SUBROUTINE FIND_NEW_CACHER(WCACHE, IBLK, IPOS, W)
    USE ini
    IMPLICIT NONE
    TYPE (wave_cache)  WCACHE
    TYPE (wavefun1) :: W
    INTEGER :: IBLK, IPOS
    ! local
    INTEGER :: I, J

    DO IBLK=1,WCACHE%NBLK

       ! found an empty block
       ! search for an emtpy slot in that block
       IF (WCACHE%C(IBLK)%NFREE>0) THEN
          DO IPOS=1,WCACHE%NBLK
             IF (WCACHE%C(IBLK)%ICNT(IPOS)==0) RETURN
          ENDDO
          WRITE(*,*) 'internal error in FIND_NEW_CACHER: NFREE was not correct, no free line'
          STOP

       ! found a block that is not allocated yet
       ! allocate initialize and return

       ELSE IF ( WCACHE%C(IBLK)%NFREE<0) THEN
          WCACHE%C(IBLK)%NFREE=WCACHE%NBLK

          ALLOCATE(WCACHE%C(IBLK)%CR(SIZE(W%CR),WCACHE%NBLK), & 
                   WCACHE%C(IBLK)%CPROJ(SIZE(W%CPROJ),WCACHE%NBLK), &
                   WCACHE%C(IBLK)%ICNT(WCACHE%NBLK))
          WCACHE%C(IBLK)%ICNT=0

          CALL REGISTER_ALLOCATE(16._q*SIZE(WCACHE%C(IBLK)%CR), "cacher")
          MAX_MEMORY_CACHER=MAX(MAX_MEMORY_CACHER,SEARCH_ALLOC_MEMORY("cacher"))

          IPOS=1
          RETURN
       ENDIF
    ENDDO
    WRITE(*,*) 'internal error in FIND_NEW_CACHER: no free block'
    STOP

  END SUBROUTINE FIND_NEW_CACHER


!***********************************************************************
!
! retrieve a wavefunction with index NBANDS, NK, ISP in the structure
!
!
!***********************************************************************

  SUBROUTINE REMOVE_CACHER(WCACHE, NB, NK, ISP, W)
    USE ini
    IMPLICIT NONE
    TYPE (wave_cache), POINTER ::  WCACHE
    INTEGER :: NB, NK, ISP
    TYPE (wavefun1) :: W
    ! local
    INTEGER :: IBLK, IPOS

    PROFILING_START('remove_cacher')

    IF (.NOT. ASSOCIATED(WCACHE)) THEN
       PROFILING_STOP('remove_cacher')
       RETURN
    ENDIF
    IBLK=WCACHE%IBLK(NB, NK, ISP)
    IPOS=WCACHE%IPOS(NB, NK, ISP)
    IF (IBLK==0 .OR. IPOS==0 ) THEN
       WRITE(*,*) 'error in REMOVE_CACHER: could not find wavefunction',NB, NK, ISP
       STOP
    ELSE
       IF ( WCACHE%C(IBLK)%ICNT(IPOS)<=0) THEN
          WRITE(*,*) 'error in REMOVE_CACHER: reference count is zero',IBLK, IPOS
          STOP
       ENDIF

       W%CR   =WCACHE%C(IBLK)%CR(:,IPOS)
       W%CPROJ=WCACHE%C(IBLK)%CPROJ(:,IPOS)

       ! decrease reference count
       WCACHE%C(IBLK)%ICNT(IPOS)=WCACHE%C(IBLK)%ICNT(IPOS)-1

       ! if reference count is 0, remove from pointer table
       ! and increase the NFREE entry
       IF (WCACHE%C(IBLK)%ICNT(IPOS)==0) THEN
          WCACHE%C(IBLK)%NFREE=WCACHE%C(IBLK)%NFREE+1
          WCACHE%IBLK(NB, NK, ISP)=0
          WCACHE%IPOS(NB, NK, ISP)=0

          ! entirely free -> deallocate storage
          IF (WCACHE%C(IBLK)%NFREE==WCACHE%NBLK) THEN
             WCACHE%C(IBLK)%NFREE=-1
             CALL DEREGISTER_ALLOCATE(16._q*SIZE(WCACHE%C(IBLK)%CR), "cacher")
             DEALLOCATE(WCACHE%C(IBLK)%CR,WCACHE%C(IBLK)%CPROJ, WCACHE%C(IBLK)%ICNT)
          ENDIF
       ENDIF
    ENDIF
    PROFILING_STOP('remove_cacher')

  END SUBROUTINE REMOVE_CACHER


!***********************************************************************
!
! check whether cache structure is empty
!
!***********************************************************************

  SUBROUTINE ISEMTPY_CACHER(WCACHE)
    IMPLICIT NONE
    TYPE (wave_cache), POINTER :: WCACHE
    ! local
    INTEGER :: NBANDS, NKPTS, ISPIN
    INTEGER :: NB, K, ISP, IBLK, IPOS

    IF (.NOT. ASSOCIATED(WCACHE)) RETURN

    NBANDS=SIZE(WCACHE%IBLK,1)
    NKPTS =SIZE(WCACHE%IBLK,2)
    ISPIN =SIZE(WCACHE%IBLK,3)

    DO ISP=1,ISPIN
       DO K=1,NKPTS
          DO NB=1,NBANDS
             IBLK=WCACHE%IBLK(NB, K, ISP)
             IPOS=WCACHE%IPOS(NB, K, ISP)
             IF (IBLK/=0 .OR. IPOS/=0 ) THEN
                WRITE(*,*) 'error in ISEMPTY_CACHER: cache structure is not emtpy',NB, K, ISP
                STOP
             ENDIF
          ENDDO
       ENDDO
    ENDDO

    DO IBLK=1,WCACHE%NBLK
       ! found an empty block
       ! search for an emtpy slot in that block
       IF (WCACHE%C(IBLK)%NFREE>0 .AND. WCACHE%C(IBLK)%NFREE/=WCACHE%NBLK ) THEN
          WRITE(*,*) 'error in ISEMPTY_CACHER: cache blocks are dusty',IBLK,WCACHE%C(IBLK)%NFREE,WCACHE%NBLK
          STOP
       ELSE IF (WCACHE%C(IBLK)%NFREE>0) THEN
          DO IPOS=1,WCACHE%NBLK
             IF (WCACHE%C(IBLK)%ICNT(IPOS)/=0) THEN
                WRITE(*,*) 'error in ISEMPTY_CACHER: cache block is dusty',IBLK, IPOS, WCACHE%C(IBLK)%ICNT(IPOS)
                STOP
             ENDIF
          ENDDO          
       ENDIF
    ENDDO

  END SUBROUTINE ISEMTPY_CACHER


!***********************************************************************
!
! store self-energy times orbital into  WCACHE%FH%CXI(1,NB1_STORE,1:2)
! the non-local contribution  \int D_ij(r) .... is stored into
! WCACHE%FH%CKAPPA(:,NB1_STORE,1:2)
!
! self-energy times orbital is calculated as
!
!  a_1(r) = sum_G 1/N  u_2(r) 
!   e^iG r sum_G' 1/N (\sum_r  u_1(r') u*_2(r') -iG'r' ) RESPONSEFUN(G',G,omega)
!
!   1 = k_1,n_1
!   2 = k_2,n_2
! and 
!   k_2 = k_1 + q
!
! upon entry:
! WPSI contains
!    W_1,2(G) = 1/N sum_G' (\sum_r  u_1(r) u*_2(r) -iG'r ) RESPONSEFUN(G',G,omega)
! CHG  contains (not used)
!    CHG(G) = (\sum_r  u_1(r') u*_2(r') -iG r )
!
! the contributions are added weighted by W_INTER 
! (set in SCREENED_TWO_ELECTRON_CACHED) to the action WCACHE%FH%CXI
!
! IMPORTANT RESTRICTION:
! the phase factors for the augmentation charge (required for 
! the call RPRO1_HF) are not recalculated for efficiency reasons.
! this would require a call such as:
!  CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, & 
!      KPOINTS_FULL%VKPT(:,NK1)-KPOINTS_FULL%VKPT(:,NK2))
! but would be too time consuming
! also storing all accelerations at all k-points would be 
! too storage demanding.
!
! thus the calling routine must clear the cache between 
! subsequent calls (done by chi.F)
! this makes the code significantly slower if many k-points are used
!
!
!***********************************************************************

  SUBROUTINE STORE_GW_ACC( WCACHE, WDESQ, WPSI, CHG, Z, NB1, NK1, NB2, NK2, ISP, W_INTER)
    USE fock
    USE twoelectron4o
    IMPLICIT NONE
    TYPE (wave_cache), POINTER :: WCACHE
    TYPE (wavedes1) :: WDESQ
    COMPLEX(q) :: WPSI(:)     ! see above
    COMPLEX(q) :: CHG (:)     ! see above
    COMPLEX(q) :: Z           ! \sum_G CONJG(W_1,2(G)) CHG(G)
    INTEGER :: NB1, NK1, NB2, NK2, ISP
    REAL(q) :: W_INTER(2)     ! weight use for the addition to XI
  ! local
    TYPE (wavedes1) :: WDES1
    LOGICAL :: LPHASE
    INTEGER :: I
    COMPLEX(q) :: CPHASE(m_ GRIDHF%MPLWV)
    COMPLEX(q) :: WPSIP(WDESQ%NGVECTOR)
    GDEF    :: WPSIR(m_ MAX(GRIDHF%MPLWV, WDESQ%GRID%MPLWV))
!    GDEF    :: CHGR(m_ MAX(GRIDHF%MPLWV, WDESQ%GRID%MPLWV))
    GDEF    :: SUM
    REAL(q) :: WEIGHT
    GDEF    :: CPROJ(WDESQ%NPROD)
    COMPLEX(q), EXTERNAL ::  ZDOTC, ZDDOTC
    INTEGER :: NB1_STORE


    PROFILING_START('store_gw_acc')

    IF (.NOT. ASSOCIATED(WCACHE)) THEN
       PROFILING_STOP('store_gw_acc')
       RETURN
    ENDIF


    IF (WCACHE%NBLOW==-1) THEN
       IF (NB1 > SIZE(WCACHE%FH(1)%CXI,2)) THEN
          WRITE(*,*) 'internal error in STORE_GW_ACC: band index exceeds FH dimension',NB1,SIZE(WCACHE%FH(1)%CXI,2)
          STOP
       ENDIF
       NB1_STORE=NB1
    ELSE
       IF (NB1 < (WCACHE%NBLOW-1)*WCACHE%NB_PAR+1 ) THEN
          WRITE(*,*) 'internal error in STORE_GW_ACC: band index below NBLOW',NB1, WCACHE%NBLOW 
          STOP
       ENDIF

       IF (NB1 > WCACHE%NBHIGH*WCACHE%NB_PAR ) THEN
          WRITE(*,*) 'internal error in STORE_GW_ACC: band index exceeds NBHIGH',NB1, WCACHE%NBHIGH
          STOP
       ENDIF
       NB1_STORE=NB1-(WCACHE%NBLOW-1)*WCACHE%NB_PAR
    ENDIF
       
    IF ((GRIDHF%LREAL  .NEQV.  WDESQ%GRID%LREAL) .OR. (GRIDHF%REAL2CPLX  .NEQV. WDESQ%GRID%REAL2CPLX)) THEN
       WRITE(*,*) 'internal error in STORE_GW_ACC: incompatible grids',GRIDHF%LREAL, WDESQ%GRID%LREAL, GRIDHF%REAL2CPLX, WDESQ%GRID%REAL2CPLX
       STOP
    ENDIF
    
    ! bring screened potential to real space 
    ! sum_G e^iGr sum_G' 1/N (\sum_r  u_1(r') u*_2(r') -iG'r' )  RESPONSEFUN(G',G,omega)

    IF (WDESQ%LGAMMA) THEN
       ! first problem
       ! the vector WPSI has 2*WDESQ%NGVECTOR complex elements
       ! corresponding to c_0 cos(G_0 r), s_0 sin(G_0 r), c_1 cos(G_1 r), s_1 sin(G_1 r), etc.
       ! we can use only the real part at this point
       ! REAL(c_0, s_0, c_1, s_1, ...)
       CALL DCOPY(2*WDESQ%NGVECTOR , WPSI(1), 2, WPSIP(1), 1)
       ! this is also subtle: WDESQ%GRID is used here and not GRID_FOCK
       ! GRID_FOCK uses complex storage mode in real space, whereas WDESQ%GRID uses
       ! real storage mode in real space and thus is compatibly to the hamil.F routines
       CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), WPSIR(1), WPSIP(1), WDESQ%GRID )
    ELSE
       CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), WPSIR(1), WPSI(1), WDESQ%GRID )
    ENDIF

    ! apply phase factor e^iGr if required
    CALL GET_PHASE_CONJG(WCACHE, NK2,  LPHASE, CPHASE)
    IF (LPHASE) CALL APPLY_PHASE_GDEF( GRIDHF, CPHASE(1), WPSIR(1), WPSIR(1) )

    ! get wave function in real space from cache
    CALL REMOVE_CACHER(WCACHE, NB2, NK2, ISP, WCACHE%FH(1)%WQ)

    ! multiply it with the screened potential  u_2(r) W_1,2(r) store result in WCACHE%FH%CXI
    ! (this is the most time consuming step, any optimization must go here)
    WEIGHT=1./WDESQ%GRID%NPLWV
    CALL SETWDES(WDES_FOCK,WDES1,NK1) ! result is at NK1 as pointed out by Daniel Aberg

    CALL VHAMIL_TRACE(WDES1, GRID_FOCK, WPSIR(1), WCACHE%FH(1)%WQ%CR(1), WCACHE%FH(1)%CXI(1,NB1_STORE,1), WEIGHT*W_INTER(1))
    CALL VHAMIL_TRACE(WDES1, GRID_FOCK, WPSIR(1), WCACHE%FH(1)%WQ%CR(1), WCACHE%FH(1)%CXI(1,NB1_STORE,2), WEIGHT*W_INTER(2))

! this test here should give the same value as Z passed by the calling routine
! if this works (not quite trivial) the Hamiltonian should be ok
!    CALL FFTWAV(WDESQ%NGVECTOR, WDESQ%NINDPW(1), CHGR(1), CHG(1), WDESQ%GRID )
!    SUM=0
!    DO I=1,WDESQ%GRID%RL%NP
!       SUM=SUM+CHGR(I)*GCONJG( WPSIR(I))
!    ENDDO
!    WRITE(77,'(4F14.7)') Z
!    WRITE(78,'(4F14.7)') SUM*1./WDESQ%GRID%NPLWV
!test end

    IF (WDES_FOCK%LOVERL) THEN
       ! add to acceleration kappa
       ! calculate D_LM
       AUG_DES%RINPL=WEIGHT  ! multiplicator for RPRO1
       ! this call sets WCACHE%FH%CDLM
       CALL RPRO1_HF(FAST_AUG_FOCK, AUG_DES, WCACHE%FH(1)%W1, WPSIR(:))
       IF (WDES_FOCK%NRSPINORS==2) WCACHE%FH(1)%CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)= WCACHE%FH(1)%CDLM(1:AUG_DES%NPRO)
       ! transform D_LM -> D_lml'm'
       CALL CALC_DLLMM_TRANS(WDES_FOCK, AUG_DES, TRANS_MATRIX_FOCK,  WCACHE%FH(1)%CDIJ(:,:,:,:), WCACHE%FH(1)%CDLM(:))
       ! add D_lml'm' to kappa_lm_N (sum over l'm')
       CALL OVERL_FOCK(WDES_FOCK,  WCACHE%FH(1)%LMDIM,  WCACHE%FH(1)%CDIJ(1,1,1,1), & 
            WCACHE%FH(1)%WQ%CPROJ(1), CPROJ ,.FALSE.)
       WCACHE%FH(1)%CKAPPA(:,NB1_STORE,1)=WCACHE%FH(1)%CKAPPA(:,NB1_STORE,1)+CPROJ*W_INTER(1)
       WCACHE%FH(1)%CKAPPA(:,NB1_STORE,2)=WCACHE%FH(1)%CKAPPA(:,NB1_STORE,2)+CPROJ*W_INTER(2)
    ENDIF
    PROFILING_STOP('store_gw_acc')

  END SUBROUTINE STORE_GW_ACC

!***********************************************************************
!
! store the final acceleration steming from GW for a pair of
! k-points from WCACHE%FH%CXI(1,:,2)  and WCACHE%FH%CKAPPA(1,:,2)
! into a wave function array W%CW
!
! this implies essentially that a sum of the local and non local
! terms over all nodes is performed
!
! the non-local part sum_i p_i C_i is added in real or reciprocal
! space
! this is essentially the final part of the Fock routines
!
!***********************************************************************


  SUBROUTINE STORE_GW_ACC_FINAL( WCACHE, INDEX, W, LATT_CUR, NONLR_S, NONL_S, NK1, ISP, W0)
    USE fock
    USE twoelectron4o
    USE screened_2e
    USE nonl_high
    USE lattice
    IMPLICIT NONE
    INTEGER :: INDEX
    TYPE (wave_cache), POINTER :: WCACHE
    TYPE (wavespin)            :: W, W0
    TYPE (latt)        LATT_CUR
    INTEGER :: NK1, ISP
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
  ! local
    TYPE (wavedes1) :: WDES1, WDES1HF
    INTEGER :: N, N_, N_STORE, ISPINOR
    COMPLEX(q) :: CWORK(W%WDES%GRID%MPLWV*W%WDES%NRSPINORS)

    IF (NONLR_S%LREAL) THEN
       CALL PHASER(W%WDES%GRID,LATT_CUR,NONLR_S,NK1,W%WDES)
    ELSE
       CALL PHASE(W%WDES,NONL_S,NK1)
    ENDIF

! generate the descriptor for the original WDES
    CALL SETWDES(W%WDES,WDES1,NK1)
    CALL SETWDES(WDES_FOCK,WDES1HF,NK1)

! collect CXI and CKAPPA
!    IF (WDES1%DO_REDIS) THEN  ! why is this if here
      CALLMPI( M_sum_z(WDES1%COMM_INTER,WCACHE%FH(1)%CXI(1,1,INDEX),SIZE(WCACHE%FH(1)%CXI,1)*SIZE(WCACHE%FH(1)%CXI,2)))
      CALLMPI( M_sum_g(WDES1%COMM_INTER,WCACHE%FH(1)%CKAPPA(1,1,INDEX),SIZE(WCACHE%FH(1)%CKAPPA,1)*SIZE(WCACHE%FH(1)%CKAPPA,2)))
!    ENDIF
! fourier transform local accelerations xi (only own bands)
    fft_back:DO N_=1,SIZE(WCACHE%FH(1)%CXI,2)/W%WDES%NB_PAR
       ! global band index, N_ is the local band index
       N=(N_-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
       IF (WCACHE%NBLOW==-1) THEN
          N_STORE=N_
       ELSE
          N_STORE=N_+WCACHE%NBLOW-1
          IF (N_STORE> WCACHE%NBHIGH) CYCLE
       ENDIF
       IF (N_STORE> SIZE(W%CW,2)) THEN
          WRITE(*,*) 'internal error in STORE_GW_ACC_FINAL: N_STORE is out of range',N_STORE, SIZE(W%CW,2), WCACHE%NBLOW, WCACHE%NBHIGH
          STOP
       ENDIF

! add CKAPPA to CXI (full acceleration on band N now in CXI to W%CW
       IF (W%WDES%LOVERL) THEN
          IF (NONLR_S%LREAL) THEN
             CWORK=0
             CALL RACC0(NONLR_S, WDES1, WCACHE%FH(1)%CKAPPA(1,N,INDEX), CWORK(1))
             DO ISPINOR=0,WDES1%NRSPINORS-1
                CALL FFTEXT(WDES1%NGVECTOR,WDES1%NINDPW(1), &
                     CWORK(1+ISPINOR*WDES1%GRID%MPLWV), &
                     W%CW(1+ISPINOR*WDES1%NGVECTOR,N_STORE,NK1,ISP),WDES1%GRID,.TRUE.)
             ENDDO
          ELSE
             CALL VNLAC0(NONL_S, WDES1, WCACHE%FH(1)%CKAPPA(1,N,INDEX), W%CW(1, N_STORE, NK1, ISP))
          ENDIF
       ENDIF
       DO ISPINOR=0,WDES1%NRSPINORS-1
          CALL FFTEXT(WDES1HF%NGVECTOR,WDES1HF%NINDPW(1), &
               WCACHE%FH(1)%CXI(1+ISPINOR*GRID_FOCK%MPLWV,N,INDEX), &
               W%CW(1+ISPINOR*WDES1HF%NGVECTOR,N_STORE, NK1, ISP),GRID_FOCK,.TRUE.)
       ENDDO
       
    ENDDO fft_back
    
    WCACHE%FH(1)%CXI(:,:,INDEX)   =0
    WCACHE%FH(1)%CKAPPA(:,:,INDEX)=0

!    WRITE(*,*) 'maxmimum memory used',MAX_MEMORY_CACHER

  END SUBROUTINE STORE_GW_ACC_FINAL

!************************ SUBROUTINE EDDIAG_GW *************************
!
! this subroutine diagonalizes the linearized GW Hamiltonian
!
! linearization of selfenergy around QP energies yields the following 
! generalized eigenvalue problem
!  sigma(E_QP) + d sigma(E_QP)/dE (E-E_QP) = E
!  sigma(E_QP) - d sigma(E_QP)/dE E_QP = E (1 - d sigma(E_QP)/dE )
!
! WACC1  is the action of the selfenergy operator on each band (at E_QP)
! WACC2  is the action of the derivative of the selfenergy operator 
!        on each band (at E_QP)
!
! for COHSEX (select by NOMEGA = 1 in the INCAR files)
! WACC1  stores the entirely correlation contribution
!        (done in SCREENED_TWO_ELECTRON_CACHED 
!         via the weights W_INTER)
!        a_i(G) = sum_r e^(- i G r)
!         int dr' sum_n u_n(r)  (W(r,r')-v(r,r')) (1/2 - f_n) u*_n(r')  u_i(r')
!        1/2 term is the COH 
!       -f_n      corresponds to SEX
! WACC2  stores the SEX contribution only (used only for natural orbitals)
!        a_i(G) = sum_r e^(- i G r)
!         int dr' sum_n u_n(r')  (W(r,r')-v(r,r'))  f_n u*_n(r')  u_i(r')
!
! STEP determines the step width
! for STEP=1  a simple steepest descent step is done
!             essentially diagonalizing the Hermitian approximation
!             to the Hamiltonian
! for STEP/=1 a damped MD algorithm is used
!             following 
!             Freysoldt, Boeck, Neugebauer, Phys. Rev. B 79, 241103 (2009)
!             this is usually more stable for metals
! 
!***********************************************************************

  SUBROUTINE EDDIAG_GW( NBANDSGW, WDES, LATT_CUR, NONLR_S, NONL_S, W, WACC1, WACC2,  &
       &    LMDIM, CDIJ, CQIJ, SV, T_INFO, P, NELM, STEP, SYMM, LCOHSEX, LGWNO, &
       &    KPOINTS, WEIMIN, EFERMI, IU0, IU6 )
    USE prec
    USE wave_high
    USE lattice
    USE nonl_high
    USE hamil
    USE main_mpi
    USE fock
    USE pseudo
    USE dfast
    USE mlr_optic
    USE subrot_cluster
    USE mkpoints
    USE ini
    USE sym_grad
    IMPLICIT NONE

    INTEGER  NBANDSGW
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
    INTEGER NELM
    REAL(q) STEP
    LOGICAL LCOHSEX                                 ! use COHSEX
    LOGICAL LGWNO                                   ! determine natural orbitals for GW type calculations (see below)
    TYPE (kpoints_struct) KPOINTS
    REAL(q) WEIMIN                                  ! minimum occupancy for empty state
    REAL(q) EFERMI                                  ! Fermi-level
    INTEGER IU0, IU6
    TYPE (symmetry)    SYMM
! local variables
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER  NBANDSGW_K(WDES%NKPTS,WDES%ISPIN)     ! avoid cutting through degenerate eigenstates
    INTEGER  MAX_NBANDSGW
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*WDES%NB_TOT)
    REAL(q)    R(WDES%NB_TOT)
!    REAL(q)    RWORK(3*WDES%NB_TOT) ! sufficient for ZHEGV
    REAL(q)    RWORK(8*WDES%NB_TOT)  ! required for ZGGEV
    COMPLEX(q) CDCHF
    INTEGER ISYM
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:) ! subspace rotation
    GDEF, ALLOCATABLE :: COVL(:,:) ! overlap 
    GDEF, ALLOCATABLE :: CTMP(:,:) ! temporary
    INTEGER ISP, NK, NSTRIP, NSTRIP_ACT, NSTRIP_RED
    INTEGER NPOS, NPOS_RED, N, NP, NB_TOT, NBANDS, IFAIL
    TYPE (wavedes1)    WDES1           ! descriptor for one k-point
    TYPE (wavefun1)    W1              ! current wavefunction
    TYPE (wavefuna)    WHAM            ! array to store accelerations for a selected block
    TYPE (wavefuna)    WA              ! array to store pointer
    TYPE (wavefuna)    ACC1            ! array to store selfenergy times orbitals
    TYPE (wavefuna)    ACC2            ! array to store overlap 
    TYPE (wavefuna)    WNONL           ! array to hold non local part D * wave function character
    REAL(q) :: Z(WDES%NB_TOT), SIGMA(WDES%NB_TOT), CELNEW(WDES%NB_TOT)
    COMPLEX(q) :: ALPHA(WDES%NB_TOT), BETA(WDES%NB_TOT)
    INTEGER ::  IPIV(WDES%NB_TOT)
! variables for damped MD
    REAL(q) :: ORT2, DE2, ORTCEL, DECEL, GNORM, ORT, FRICTION, GAMMA, DESUM0
    INTEGER :: N1, N2
! cluster
    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
    TYPE (eigenf_cluster),POINTER :: PDEG_CLUSTER
    INTEGER IDUM, IERR ; REAL(q) :: RDUM ; COMPLEX(q) :: CDUM ; CHARACTER (LEN=1) :: CHARAC 
! costumize here
! LKOTANI implies using the version of Schilfgaarde and Kotani (read from INCAR)
    LOGICAL :: LKOTANI=.FALSE.

! LDYSON this flag forces the eigenstates to become as close
! as possible to the Dyson orbitals solving T + V_H + Sigma(e) | phi >= e | phi>
! this is achived by copying the lower triangle to the upper triangle
! putting more emphasize on the occupied states
    LOGICAL :: LDYSON=.FALSE.

! LADDHERM adds the hermitian elements to the matrix
! this however might cause odd diagonal eigenvalues beyond NBANDSGW, since
! these are not properly calculated and simply continously updated
    LOGICAL :: LADDHERM=.FALSE.
!
! HAMIL_HERM = .TRUE. forces the Hamilton matrix and overlap matrix to be Hermitian
! in this case the Hermitian eigenvalue problem S^(-1/2) H S^(-1/2) is solved
!
! HAMIL_HERM is .FALSE. the full non Hermitian eigenvalue problem
!  H T alpha =  S T  beta is solved (<=>  S-1 H T alpha =  T  beta )
! is solved and the eigenvectors are afterwards orthogonalized
! this is maybe the cleanest version, however often the LAPACK routine fails
! hence I prefer HAMIL_HERM=.TRUE.

    LOGICAL :: HAMIL_HERM=.TRUE.
    
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS
    NSTRIP=NSTRIP_STANDARD
    ISYM  =SYMM%ISYM

    ! force full fock contribution
    HFSCREEN=0.0
    AEXX    =1.0

    LKOTANI=.FALSE.
    LDYSON=.FALSE.
    LADDHERM=.FALSE.

    CALL RDATAB(.TRUE.,INCAR,99,'LKOTANI','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LKOTANI,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IU0>=0) WRITE(IU0,*)'Error reading item ''LKOTANI'' from file INCAR.'
    ENDIF

    CALL RDATAB(.TRUE.,INCAR,99,'LDYSON','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LDYSON,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IU0>=0) WRITE(IU0,*)'Error reading item ''LDYSON'' from file INCAR.'
    ENDIF
    IF (LDYSON) THEN
       LKOTANI=.TRUE.
       LADDHERM=.TRUE.
    ENDIF
    
    IF (LDYSON) WRITE(*,*) 'LDYSON was set'

    CALL RDATAB(.TRUE.,INCAR,99,'LADDHERM','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LADDHERM,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IU0>=0) WRITE(IU0,*)'Error reading item ''LADDHERM'' from file INCAR.'
    ENDIF

    IF (IU6>=0) WRITE(IU6,'(//A,I2)') ' QP shifts <psi_nk| G(iteration)W_0 |psi_nk>: iteration',NELM
    IF (IU0>=0) THEN
       WRITE(IU0,'(//A,I2)') ' QP shifts <psi_nk| G(iteration)W_0 |psi_nk>: iteration',NELM
       WRITE(17,'(//A,I2)') ' QP shifts <psi_nk| G(iteration)W_0 |psi_nk>: iteration',NELM
    ENDIF

!=======================================================================
! add kinetic energy, local and exact exchange term to WACC1
!=======================================================================
    CALL SETWDES(WDES,WDES1,0)
    CALL NEWWAVA(WHAM, WDES1, NSTRIP)
    ALLOCATE(  W1%CR(WDES%GRID%MPLWV*WDES%NRSPINORS))

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          CALL SETWDES(WDES,WDES1,NK)
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code

          IF (NONLR_S%LREAL) THEN
             CALL PHASER(WDES%GRID,LATT_CUR,NONLR_S,NK,WDES)
          ELSE
             CALL PHASE(WDES,NONL_S,NK)
          ENDIF

          DO NPOS=1,NBANDSGW/WDES%NB_PAR,NSTRIP
             NSTRIP_ACT=MIN(NBANDSGW/WDES%NB_PAR+1-NPOS,NSTRIP)
             
             !  calculate V_{local} |phi> + T | phi >
             !  for a block containing NSTRIP wavefunctions
             
             ! calculate accelerations on current stripe of bands
             ! (using the full set of k-points)
             ! start with the exchange term only
             CALL FOCK_ACC(WDES%GRID, LMDIM, LATT_CUR, W,  &
                  NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                  WHAM%CW(:,:), P, CQIJ(1,1,1,1), CDCHF )
             ! add local contribution (Hartree + ionic) plus kinetic energy
             ! and also add the correlation part WACC1%CW(:,N,NK,ISP)
             DO N=NPOS,NPOS+NSTRIP_ACT-1
                NP=N-NPOS+1
                CALL SETWAV(W, W1, WDES1, N, ISP)
                CALL FFTWAV_W1(W1)
                CALL HAMILT_LOCAL(W1, SV, ISP, WHAM%CW(:,NP),.TRUE.)
                ! add the correlation piece
                WACC1%CW(:,N,NK,ISP)=WHAM%CW(:,NP)+WACC1%CW(:,N,NK,ISP)
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    CALL DELWAVA(WHAM)
    DEALLOCATE(W1%CR)
#ifdef MPI
    IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
       CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,WACC1)
    ENDIF
#endif
    IF (SYMM%ISYM>=0 .AND. .NOT. WDES%LGAMMA) THEN
      CALL APPLY_SMALL_SPACE_GROUP_OP( W, WACC1, NONLR_S, NONL_S, &
         P, T_INFO%NIONS, LATT_CUR, SYMM, CQIJ, .FALSE. , IU6)

      CALL APPLY_SMALL_SPACE_GROUP_OP( W, WACC2, NONLR_S, NONL_S, &
         P, T_INFO%NIONS, LATT_CUR, SYMM, CQIJ, .FALSE. , -1)

      CALL STOP_TIMING("G",IU6,"GWSYM")
    ENDIF
!=======================================================================
! determine whether NBANDSGW_K needs to be reduced to
! avoid cutting through degenerate bands
!=======================================================================
    CALL SETWDES(WDES,WDES1,0)
    CALL NEWWAVA_PROJ(WNONL, WDES1)  ! instad of WNONL one could use WACC1

    ALLOCATE( CHAM(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN), & 
              COVL(WDES%NB_TOT, WDES%NB_TOT), & 
              CTMP(WDES%NB_TOT, WDES%NB_TOT))

    NULLIFY(DEG_CLUSTER)
    IF (ISYM>0) &
       CALL FIND_DEG_CLUSTERS(W%WDES, W, DEG_CLUSTER, LDISTRIBUTED_IN=.FALSE.)

    NBANDSGW_K=0
    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code
          CALL SETWDES(WDES,WDES1,NK)

          NBANDSGW_K(NK,ISP)=MIN(NBANDSGW,WDES%NB_TOTK(NK,ISP))
          IF (ISYM>0) THEN
             PDEG_CLUSTER=>DEG_CLUSTER(NK,ISP)%DEG_CLUSTER
             DO
                IF (.NOT.ASSOCIATED(PDEG_CLUSTER)) EXIT
                IF (PDEG_CLUSTER%BAND_START<=NBANDSGW .AND. PDEG_CLUSTER%BAND_END> NBANDSGW) THEN
                   NBANDSGW_K(NK,ISP)=PDEG_CLUSTER%BAND_START-1
                   EXIT
                ENDIF
                PDEG_CLUSTER=>PDEG_CLUSTER%NEXT
             ENDDO
          ENDIF
!=======================================================================
! construct the selfenergy matrix and its derivative
!=======================================================================
          WA=ELEMENTS(W, WDES1, ISP)
          ACC2=ELEMENTS(WACC2, WDES1, ISP)
          ACC1=ELEMENTS(WACC1, WDES1, ISP)
          CALL REDISTRIBUTE_PW( ACC2)  ! ACC1%CPROJ and ACC2&CPROJ is never used
          CALL REDISTRIBUTE_PW( ACC1)

          CALL OVERL(WDES1, .TRUE.,LMDIM,CDIJ(1,1,1,ISP), WA%CPROJ(1,1),WNONL%CPROJ(1,1))

          CALL REDISTRIBUTE_PW(WA)
          CALL REDISTRIBUTE_PROJ(WA)
          CALL REDISTRIBUTE_PROJ(WNONL)

          CHAM(:,:,NK,ISP)=0
          COVL=0
          strip: DO NPOS=1,NBANDSGW/WDES%NB_PAR,NSTRIP
             NSTRIP_ACT=MIN(NBANDSGW/WDES%NB_PAR+1-NPOS,NSTRIP)
             
             NPOS_RED  =(NPOS-1)*WDES%NB_PAR+1
             NSTRIP_RED=NSTRIP_ACT*WDES%NB_PAR
             ! CHAM(2,1) = <u_2 | T+ V_local + sigma_1 | u_1 >
             CALL ORTH2( &
                  WA%CW_RED(1,1),ACC1%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
                  WNONL%CPROJ_RED(1,NPOS_RED),NB_TOT, &
                  NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM(1,1,NK,ISP))
             ! COVL(2,1) = <u_2 | d sigma_1 / d E | u_1 >
             CALL ORTH2( &
                  WA%CW_RED(1,1),ACC2%CW_RED(1,NPOS_RED),WA%CPROJ_RED(1,1), &
                  WNONL%CPROJ_RED(1,NPOS_RED),NB_TOT, &
                  NPOS_RED, NSTRIP_RED, WDES1%NPL_RED,0,WDES1%NRPLWV_RED,WDES1%NPROD_RED,COVL(1,1))
          ENDDO strip
          ! sum matrix  over cores
          CALLMPI( M_sum_g(WDES%COMM_INTER,CHAM(1,1,NK,ISP),NB_TOT*NB_TOT))
          CALLMPI( M_sum_g(WDES%COMM_INTER,COVL(1,1),NB_TOT*NB_TOT))

          ! just in case: add the right square of the matrix beyond NBANDSGW
          ! this can be used for COHSEX
          DO N=1,NBANDSGW_K(NK,ISP)
             CHAM(N,NBANDSGW_K(NK,ISP)+1:NB_TOT,NK,ISP)=GCONJG(CHAM(NBANDSGW_K(NK,ISP)+1:NB_TOT,N,NK,ISP))
             COVL(N,NBANDSGW_K(NK,ISP)+1:NB_TOT)=GCONJG(COVL(NBANDSGW_K(NK,ISP)+1:NB_TOT,N))
          ENDDO
          ! clear "undetermined" elements
          CHAM(NBANDSGW_K(NK,ISP)+1:NB_TOT, NBANDSGW_K(NK,ISP)+1:NB_TOT, NK,ISP)=0
          COVL(NBANDSGW_K(NK,ISP)+1:NB_TOT, NBANDSGW_K(NK,ISP)+1:NB_TOT)=0

          ! fill in eigenvalues for elements not calculated
          DO N=NBANDSGW_K(NK,ISP)+1,NB_TOT
             CHAM(N,N,NK,ISP)=REAL(W%CELTOT(N,NK,ISP),q)
          ENDDO

! this might be helpfull for debugging, maybe document out SYMGRAD above
!          CALL ZHEEV &
!          ('V','L',NBANDSGW_K(NK,ISP),CHAM(1,1,NK,ISP),NB_TOT, &
!             R,CWRK,SIZE(CWRK), RWORK,  IFAIL)
!          WRITE(101,'(10E14.7)') R

          ! for COHSEX the selfenergy is Hermitian, and we can
          ! add "unknown" matrix elements
          ! this however might sometimes lead to divergence of the unoccupied states
          ! since the "remaining" block is not properly calculated
          IF (LADDHERM) THEN
             NBANDSGW_K(NK,ISP)=WDES%NB_TOTK(NK,ISP)
          ENDIF

          MAX_NBANDSGW=MAX(NBANDSGW, NBANDSGW_K(NK,ISP))

          IF ( LCOHSEX ) THEN
             DO N=1,MAX_NBANDSGW
                SIGMA(N)=CHAM(N,N,NK,ISP)
                Z(N)=1
                CELNEW(N)=REAL(W%CELTOT(N,NK,ISP),q)+Z(N)*(SIGMA(N)-REAL(W%CELTOT(N,NK,ISP)))
             ENDDO
             ! COVL holds the matrix  
             ! COVL(j,i) = \int dr dr' u_j*(r) K(r,r')  u_i(r')
             !       with K(r,r') [sum_n u_n(r)  (W(r,r')-v(r,r')) f_n u*_n(r')]
             ! we now determine a rotation matrix that optimizes the virtual orbitals in the set u_i
             ! do maximize the "spatial" overlap with K(r,r') 
             IF (LGWNO) THEN
                CHAM(:,:,NK,ISP)=-COVL  ! change sign, since diagonalization sorts ascending
                DO N=1,NBANDSGW_K(NK,ISP)
                   IF (W%FERTOT(N,NK,ISP)>0.5) THEN ! right now only insulators
                      CHAM(N,N+1:,NK,ISP)=0
                      CHAM(N+1:,N,NK,ISP)=0
                   ENDIF
                ENDDO
             ENDIF
             COVL=0
          ELSE
             DO N=1,MAX_NBANDSGW
                SIGMA(N)=CHAM(N,N,NK,ISP)
                Z(N)=1/(1+REAL(COVL(N,N),q))
                CELNEW(N)=REAL(W%CELTOT(N,NK,ISP),q)+Z(N)*(SIGMA(N)-REAL(W%CELTOT(N,NK,ISP)))
             ENDDO
          ENDIF
!=======================================================================
! set up the linearized eigenvalue problem
!  sigma(E_QP) - d sigma(E_QP)/dE E_QP = E (1 - d sigma(E_QP)/dE )
!   COVL = - <phi_j| d sigma(E_j)/dE |phi_j>
!=======================================================================
!          IF (ABS(MOD(W%WDES%VKPT(1,NK)*2+100,1._q))< 1E-6 .AND. &
!              ABS(MOD(W%WDES%VKPT(2,NK)*2+100,1._q))< 1E-6 .AND. &
!              ABS(MOD(W%WDES%VKPT(3,NK)*2+100,1._q))< 1E-6) THEN
!          ! force matrices to be real valued
!              CHAM=REAL(CHAM,q)
!              COVL=REAL(COVL,q)
!          ENDIF

     ! force overlap matrix to be hermitian
          IF (HAMIL_HERM) THEN
             DO  N=1,MAX_NBANDSGW
                DO  NP=N,MAX_NBANDSGW
                   COVL(NP,N)=(COVL(NP,N)+GCONJG(COVL(N,NP)))/2
                   COVL(N,NP)=GCONJG(COVL(NP,N))
                ENDDO
             ENDDO
          ENDIF
          DO  N=1,MAX_NBANDSGW
             IF ( LCOHSEX ) THEN
             ! nothing
             ELSE IF (LKOTANI) THEN
             ! interpolate to new eigenvalue
                CHAM(:,N,NK,ISP)=CHAM(:,N,NK,ISP)+(W%CELTOT(N,NK,ISP)-CELNEW(N))*COVL(:,N)
             ELSE
                CHAM(:,N,NK,ISP)=CHAM(:,N,NK,ISP)+ W%CELTOT(N,NK,ISP)*COVL(:,N)
             ENDIF
             COVL(N,N)=1+COVL(N,N)

          ENDDO
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "Hamilton matrix",WDES, CHAM(:,:,NK,ISP))
          IF (IU0>=0) CALL DUMP_HAM( "Overlap  matrix",WDES, COVL)
#endif

          IF ( LCOHSEX ) THEN
             ! can use the Hamilton matrix as is
          ELSE IF (.NOT. HAMIL_HERM) THEN
             ! calculate S-1 H
             ! I prefer this version to solving   H T alpha =  S T  beta
             ! it allows to damp of the off-diagonal elements of S-1 H

             ! compose into upper and lower triangular matrix
             CALL ZGETRF( NBANDSGW_K(NK,ISP),  NBANDSGW_K(NK,ISP), COVL(1,1), SIZE(COVL, 1), IPIV, IFAIL )
             IF (IFAIL/=0) THEN
                WRITE(0,*) 'error in EDDIAG_GW: ZGETRF returns',IFAIL
                STOP
             ENDIF
             ! invert matrix
             CALL ZGETRI( NBANDSGW_K(NK,ISP), COVL(1,1), SIZE(COVL,1), IPIV, & 
                   CWRK, SIZE(CWRK), IFAIL )
             IF (IFAIL/=0) THEN
                WRITE(0,*) 'error in EDDIAG_GW: ZGETRI returns',IFAIL
                STOP
             ENDIF
             ! multiply from left with S-1
             CTMP=0
             CALL GGEMM( 'N', 'N', NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), one, COVL(1,1), NB_TOT, &
                  CHAM(1,1,NK,ISP), NB_TOT, zero , CTMP(1,1), NB_TOT)
             CHAM(:,:,NK,ISP)=CTMP

             ! now set overlap matrix to unity matrix
             COVL=0
             DO  N=1,NB_TOT
                COVL(N,N)=1
             ENDDO
          ELSE IF (.NOT. LKOTANI) THEN
             IF (.FALSE.) THEN
             ! LU decomposition of overlap matrix (presently not applied)
                IFAIL=0
#ifdef gammareal
                CALL DPOTRF('U',NBANDSGW_K(NK,ISP),COVL(1,1),NB_TOT,IFAIL)
#else
                CALL ZPOTRF('U',NBANDSGW_K(NK,ISP),COVL(1,1),NB_TOT,IFAIL)
#endif
                
                IF (IFAIL/=0) THEN
                   WRITE(*,*) 'LAPACK: Routine ZPOTRF failed!',IFAIL
                   STOP
                ENDIF
                ! inversion of upper triangle
#ifdef gammareal
                CALL DTRTRI('U','N',NBANDSGW_K(NK,ISP),COVL(1,1),NB_TOT,IFAIL)
#else
                CALL ZTRTRI('U','N',NBANDSGW_K(NK,ISP),COVL(1,1),NB_TOT,IFAIL)
#endif
                IF (IFAIL/=0) THEN
                   WRITE(*,*) 'LAPACK: Routine ZTRTRI failed!',IFAIL
                   STOP
                ENDIF
                ! clear lower triangle
                DO  N=1,NB_TOT
                   COVL(N+1:NB_TOT,N)= 0
                ENDDO
             ELSE
             ! calculate S-1/2 of overlap matrix
                CALL ROTHALF(COVL, NBANDSGW_K(NK,ISP), IU6 )
             ENDIF
             ! multiply from left with conjugated inverted matrix (i.e. L)
             ! for LU case this is correct
             ! for S-1/2 case CTMP is Hermitian so CTMP^+ = CTMP
             CALL GGEMM( trans, 'N', NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), one, COVL(1,1), NB_TOT, &
                  CHAM(1,1,NK,ISP), NB_TOT, zero , CTMP(1,1), NB_TOT)

             ! multiply from right with inverted U matrix
             CHAM(:,:,NK,ISP)=0
             CALL GGEMM( 'N', 'N', NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), one, CTMP(1,1), NB_TOT, &
                  COVL(1,1), NB_TOT, zero , CHAM(1,1,NK,ISP), NB_TOT)
          ENDIF

          ! build of Hamilton matrix is finished
          ! CHAM now stores the approximate Hamilton matrix in the space spanned by the orbitals
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "final Hamilton matrix",WDES, CHAM(:,:,NK,ISP))
#endif
          IF (HAMIL_HERM .AND. .NOT. LCOHSEX) THEN
             ! force final matrix to be Hermitian again (is usually already the case)
             ! except for Kotani-Schilfgaarde
             DO  N=1,NBANDSGW_K(NK,ISP)
                ! force diagonal to be real
                CHAM(N,N,NK,ISP)=REAL(CHAM(N,N,NK,ISP),q)
                DO  NP=N,NBANDSGW_K(NK,ISP)
                   IF (LDYSON) THEN
                   CHAM(NP,N,NK,ISP)=(CHAM(NP,N,NK,ISP)*MAX(W%FERTOT(N,NK,ISP),1.E-2_q) &
                              +GCONJG(CHAM(N,NP,NK,ISP))*MAX(W%FERTOT(NP,NK,ISP),1.E-2_q))/ &
                              (MAX(W%FERTOT(N,NK,ISP),1.E-2_q)+MAX(W%FERTOT(NP,NK,ISP),1.E-2_q))
                   ELSE
                      CHAM(NP,N,NK,ISP)=(CHAM(NP,N,NK,ISP)+GCONJG(CHAM(N,NP,NK,ISP)))/2
                   ENDIF
                   ! copy lower triangle to upper triangle
                   CHAM(N,NP,NK,ISP)=GCONJG(CHAM(NP,N,NK,ISP))
                ENDDO
             ENDDO
          ENDIF
          ! build of Hermitian approximation to the Hamilton matrix is finished
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "final Hamilton matrix",WDES, CHAM(:,:,NK,ISP))
#endif
!=======================================================================
!  Diagonalize the problem
!=======================================================================
          IF (STEP==1) THEN     ! (STEP equals INFO%TIME)
             ! damp off-diagonal elements of CHAM and COVL
             ! damp also diagonal components by step width
             ! well presently no damping since this is only executed for STEP==1
             DO N=1,NBANDSGW_K(NK,ISP)
                CHAM(N,N,NK,ISP)=W%CELTOT(N,NK,ISP)+STEP*(CHAM(N,N,NK,ISP)-W%CELTOT(N,NK,ISP))
                DO NP=N+1,NBANDSGW_K(NK,ISP)
                   COVL(N,NP)=COVL(N,NP)*STEP
                   COVL(NP,N)=COVL(NP,N)*STEP
                   CHAM(N,NP,NK,ISP)=CHAM(N,NP,NK,ISP)*STEP
                   CHAM(NP,N,NK,ISP)=CHAM(NP,N,NK,ISP)*STEP
                ENDDO
             ENDDO
          ENDIF

#ifdef  gammareal
          IF (HAMIL_HERM) THEN
          ! diagonalisation of symmetric problem
          !---------------------------------------------------------------------
             COVL=CHAM(:,:,NK,ISP)  ! store CHAM for later use
             CALL DSYEV &
                  ('V','U',NBANDSGW_K(NK,ISP),CHAM(1,1,NK,ISP),NB_TOT, &
                  R,CWRK,SIZE(CWRK), IFAIL)
          ELSE
             WRITE(*,*) 'internal error in wave_cacher.F: generalised real eigenvalue solver is not implemented'
             STOP
          ENDIF
#else
          IF (HAMIL_HERM) THEN
          ! diagonalisation of Hermitian problem
          !---------------------------------------------------------------------
             COVL=CHAM(:,:,NK,ISP)  ! store CHAM for later use
             CALL ZHEEV &
                  ('V','L',NBANDSGW_K(NK,ISP),CHAM(1,1,NK,ISP),NB_TOT, &
                  R,CWRK,SIZE(CWRK), RWORK,  IFAIL)
          ELSE
          ! diagonalisation of general eigenvalue problem
          !---------------------------------------------------------------------
             ! right hand eigenvalues of alpha S^(-1) H T alpha = 1 T beta
             CTMP=0
             CALL ZGGEV( 'N', 'V', NBANDSGW_K(NK,ISP), CHAM(1,1,NK,ISP), NB_TOT, COVL(1,1), NB_TOT, ALPHA, BETA, &
                  CTMP, NB_TOT, CTMP, NB_TOT, CWRK, SIZE(CWRK), RWORK, IFAIL )
!             WRITE(*,*) 'internal error in wave_cacher.F: ZGGEV is undocumented in the distr. version'
!             WRITE(*,*) '   the routine is not compatible to vasp.4.lib'
!             STOP
  
             R=ALPHA/BETA
             CALL SORT_EIGENVAL( R, ALPHA, BETA, CTMP, NBANDSGW_K(NK,ISP))
             CHAM(:,:,NK,ISP)=CTMP

             ! calculate overlap O= T+ T
             CALL GGEMM( trans, 'N', NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), one, CHAM(1,1,NK,ISP), NB_TOT, &
                  CHAM(1,1,NK,ISP), NB_TOT, zero , COVL(1,1), NB_TOT)

             ! determine O^(-1/2)
             CALL ROTHALF(COVL, NBANDSGW_K(NK,ISP), -1 )
             
             ! multiply rotation matrix T with O^(-1/2) -> T
             CALL GGEMM( 'N', 'N', NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), NBANDSGW_K(NK,ISP), one, CHAM(1,1,NK,ISP), NB_TOT, & 
                   COVL(1,1), NB_TOT, zero , CTMP(1,1), NB_TOT)
             CHAM(:,:,NK,ISP)=CTMP

          ENDIF
#endif
          IF (IU6>=0) THEN
             WRITE(IU6,1) NK, WDES%VKPT(:,NK)
             DO N=1,MIN(NBANDSGW_K(NK,ISP),NBANDSGW)
                WRITE(IU6,10) & 
                     N, REAL(W%CELTOT(N,NK,ISP),q), R(N), CELNEW(N), SIGMA(N), Z(N), &
                     W%FERTOT(N,NK,ISP)*W%WDES%RSPIN
             ENDDO
             WRITE(IU6,*)
             DO N=NBANDSGW_K(NK,ISP)+1,NBANDSGW
                WRITE(IU6,10) & 
                     N, REAL(W%CELTOT(N,NK,ISP),q), REAL(W%CELTOT(N,NK,ISP),q), CELNEW(N), SIGMA(N),Z(N), &
                     W%FERTOT(N,NK,ISP)*W%WDES%RSPIN
             ENDDO
          ENDIF
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "eigenvectors",WDES, CHAM(:,:,NK,ISP))
#endif
       IF (STEP==1) THEN
#ifdef debug
          IF (IU0>=0) CALL DUMP_HAM( "rotation matrix",WDES, CHAM(:,:,NK,ISP))
#endif
          W%CELTOT(1:NBANDSGW_K(NK,ISP),NK,ISP)=R(1:NBANDSGW_K(NK,ISP))
       ELSE
          ! restore Hermitian approximation of Hamiltonian
          ! required for damped MD
          CHAM(:,:,NK,ISP)=COVL
       ENDIF
       ENDDO
    ENDDO
!=======================================================================
! set up rotation matrix using more sophisticated 
! damped MD algorithm 
!=======================================================================
    IF (STEP/=1) THEN
      IF (.NOT. ALLOCATED(CHF)) THEN
         ALLOCATE(CHF(WDES%NB_TOT, WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN))
         CHF=0
      ENDIF

      CHAM=GCONJG(CHAM)  ! same as in rot.F (stupid, stupid,...)

      ! set up step for pseudo Hamiltonian matrix eta and store it in CHAM
      CALL ETAINI_GW(WDES, W, NBANDSGW_K, CHAM, CHF, COVL, KPOINTS, EFERMI, WEIMIN, & 
                    ORT2, DE2, ORTCEL, DECEL )

      GNORM =DECEL  +DE2
      ORT   =ORTCEL +ORT2

      FRICTION=SQRT(STEP*2)
      GAMMA=(1-FRICTION/2)/(1+FRICTION/2)
!      GAMMA=0

      IF (IU0>=0) THEN
         WRITE(IU0,20) GAMMA,DE2,DECEL,ORT2,ORTCEL
         WRITE(17 ,20) GAMMA,DE2,DECEL,ORT2,ORTCEL
      ENDIF
         
  20  FORMAT(' gam=',F6.3, &
           &  ' g(U,f)= ',2E10.3,' ort(U,f) =',2E10.3)

      ! expected energy change along search step
      DESUM0=GNORM+GAMMA*ORT

      DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
            IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code

            CALL SETWDES(WDES,WDES1,NK)
#ifdef debug
            CALL DUMP_HAM( "CHAM matrix",WDES, CHAM(1,1,NK,ISP))
            CALL DUMP_HAM( "CHF matrix",WDES, CHF(1,1,NK,ISP))
#endif
            ! add previous search direction for pseudo Hamiltonian eta to current one (CHAM)
            ! and store for later use in CHF
            DO N2=1,NB_TOT
               DO N1=1,NB_TOT
                  CHF(N1,N2,NK,ISP)=CHAM(N1,N2,NK,ISP)+GAMMA*CHF(N1,N2,NK,ISP)
               ENDDO
            ENDDO
            
#ifdef debug
            WRITE(*,'(10F9.5)') REAL(W%CELTOT(1:10,NK,ISP))
#endif
            ! calculate the rotation matrix COVL
            CALL ROTETA_GW(NB_TOT,NBANDSGW_K(NK,ISP),CHF(1,1,NK,ISP),W%CELTOT(:,NK,ISP),STEP,CTMP,COVL)
            ! overwrite CHAM by this step direction
            CHAM(:,:,NK,ISP)=COVL
#ifdef debug
            CALL DUMP_HAM( "CHF matrix",WDES, CHF(1,1,NK,ISP))
            CALL DUMP_HAM( "rotation matrix",WDES, CHAM(1,1,NK,ISP))
            WRITE(*,'(10F9.5)') REAL(W%CELTOT(1:10,NK,ISP))
#endif
         ENDDO
      ENDDO
    ENDIF


! make sure that all nodes use identical rotation matrix
#ifdef gammareal
    CALLMPI( M_bcast_d(WDES%COMM_KINTER, CHAM, SIZE(CHAM)))
#else
    CALLMPI( M_bcast_z(WDES%COMM_KINTER, CHAM, SIZE(CHAM)))
#endif

    IF (ISYM>0) CALL FREE_DEG_CLUSTERS(W%WDES,DEG_CLUSTER)
!=======================================================================
! rotate wavefunctions and <psi | nabla | psi > finally
!=======================================================================
    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code

          CALL SETWDES(WDES,WDES1,NK)

          ! zero elements beyond NBANDSGW_K(NK,ISP)
          CHAM(NBANDSGW_K(NK,ISP)+1:NB_TOT,:,NK,ISP)=0
          CHAM(:,NBANDSGW_K(NK,ISP)+1:NB_TOT,NK,ISP)=0
          ! set diagonal beyond NBANDSGW_K(NK,ISP) to 1
          DO  N=NBANDSGW_K(NK,ISP)+1,NB_TOT
             CHAM(N,N,NK,ISP)=1
          ENDDO

          CALL LINCOM('F',W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CHAM(1,1,NK,ISP), &
               NB_TOT,NB_TOT,WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
               W%CW(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
          CALL REDISTRIBUTE_PROJ( ELEMENTS( W, WDES1, ISP))
          CALL REDISTRIBUTE_PW( ELEMENTS( W, WDES1, ISP))
       ENDDO
    ENDDO

    ! with expensive method take not risk sync now !
    CALL KPAR_SYNC_ALL(W%WDES,W)

    CALL DELWAVA_PROJ(WNONL)
    DEALLOCATE(CHAM, COVL, CTMP)

 1  FORMAT(/' k-point ',I3,' :',3X,3F10.4/ &
          &         "  band No. DFT-energies  QP-energies  QP-e(diag)   sigma(DFT)    Z            occupation"/)
10  FORMAT((3X,I4,3X,7(F10.4,3X)))

  END SUBROUTINE EDDIAG_GW


!************************ SUBROUTINE ZERO_CHF   ************************
!************************ SUBROUTINE DEALLOCATE_CHF   ******************
! 
! zero CHF
! 
!***********************************************************************

 SUBROUTINE ZERO_CHF
   IF (ALLOCATED(CHF))THEN
      CHF=0
   ENDIF
 END SUBROUTINE ZERO_CHF

 SUBROUTINE DEALLOCATE_CHF
   IF (ALLOCATED(CHF))THEN
      DEALLOCATE(CHF)
   ENDIF
 END SUBROUTINE DEALLOCATE_CHF

!************************ SUBROUTINE ROTHALF  **************************
!
! this subroutine calculates a matrix to the power -1/2
! but kills the very low frequency components
!
!***********************************************************************

  SUBROUTINE ROTHALF(CUNI, NBANDS, IU6 )
    INTEGER NBANDS
    GDEF    ::  CUNI(:,:)
    INTEGER ::  IU6
    ! local
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    GDEF, ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)

    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))
    
! force CUNI to be Hermitian (take Hermitian part)
    DO  N1=1,NBANDS
       DO  N2=N1,NBANDS
          CUNI(N1,N2)=(CUNI(N1,N2)+GCONJG(CUNI(N2,N1)))/2
          CUNI(N2,N1)=GCONJG(CUNI(N1,N2))
       ENDDO
    ENDDO
!=======================================================================
! diagononalize the matrix
!=======================================================================
#ifdef  gammareal
    CALL DSYEV &
         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         IFAIL)
#else
    CALL ZHEEV &
         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         W,  IFAIL)
#endif
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTHALF: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
    DO N2=1,NBANDS
    IF (HFEIG(N2)<1E-8) THEN
       WRITE(IU6,'(A,I5,F10.5)') ' negative eigenvalue (try to increase CSHIFT) ',N2,HFEIG(N2)
       WRITE(IU6,'(20F6.3)') CUNI(1:NBANDS,N2)
    ENDIF
    ENDDO
    
    DO N1=1,NBANDS
       DO N2=1,NBANDS
          IF (HFEIG(N2)<1E-8) THEN
             ! try to fix the negative eigenvalue somehow
             CTMP(N1,N2)=CUNI(N1,N2)*1/SQRT(ABS(HFEIG(N2)))
          ELSE
             CTMP(N1,N2)=CUNI(N1,N2)*1/SQRT(HFEIG(N2))
          ENDIF
       ENDDO
    ENDDO
    CALL GGEMM( 'N',trans, NBANDS, NBANDS, NBANDS, one, CTMP, &
         &              NBANDS, CUNI(1,1), NDIM, zero, CEIDB, NBANDS)
   
    CUNI=0 
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)
  END SUBROUTINE ROTHALF


!************************ SUBROUTINE ROTHALF  **************************
!
! this subroutine iterates X until it becomes the solution of
!  X-2 = A
! an initial value for X and the matrix a are supplied
! assuming that X0 is an approximate solution of X-2 = a
! one can Taylor expand X around X0 (X=X0+dX)
! this yields
!         -2                  -1    -2    -2        -1   -2
!  (X0+dX)   = a -> (X0*(1+ X0  dX))  = X0   (1 + X0  dX)
!                      -2      -3
!                    X0  - 2 X0  dX = A
!            
! yields              3                       2
!   dX = 1/2  (X0 - X0  A)         1/2 X (3- X A) -> X
!
! this subroutine is presently not used
!
!***********************************************************************

  SUBROUTINE REFINE_ROTHALF(X, A, NBANDS )
    INTEGER NBANDS
    GDEF    ::  X(:,:), A(:,:)
    ! local
    INTEGER :: I, II, ITER, NDIM

    GDEF, ALLOCATABLE ::  C(:,:), D(:,:)
    
    NDIM = SIZE(X,1)
    ALLOCATE(C(NBANDS,NBANDS),D(NBANDS,NBANDS))

    DO ITER=1,5
       ! C= X^2
       CALL GGEMM( 'N','N', NBANDS, NBANDS, NBANDS, one, & 
            X, NDIM, X, NDIM, zero, C, NBANDS)
       
       ! D= -X^2 A
       CALL GGEMM( 'N','N', NBANDS, NBANDS, NBANDS, -one, & 
            C, NBANDS, A, NDIM, zero, D, NBANDS)

       ! D= 3-X^2 A
       DO I=1,NBANDS
          D(I,I)=3+D(I,I)
       ENDDO
       
       ! C = 1/2 X D
       CALL GGEMM( 'N','N', NBANDS, NBANDS, NBANDS, 0.5_q*one, & 
            X, NDIM, D, NBANDS, zero, C, NBANDS)

       X=0
       X(1:NBANDS,1:NBANDS)=C(1:NBANDS,1:NBANDS)

! this construct should yield the one matrix in D (at the end of the iteration)
!       CALL GGEMM( 'N','N', NBANDS, NBANDS, NBANDS, one, & 
!            X, NDIM, A, NDIM, zero, C, NBANDS)
!       CALL GGEMM( 'N','N', NBANDS, NBANDS, NBANDS, one, &
!            C, NBANDS, X, NDIM, zero, D, NBANDS)
    ENDDO

    DEALLOCATE(C,D)

  END SUBROUTINE REFINE_ROTHALF

!************************ SUBROUTINE SORT_EIGENVAL *********************
!
! this subroutine sorts the eivenvalues and eigenvectors in COVL
!
!***********************************************************************

  SUBROUTINE SORT_EIGENVAL(RA, RALPHA, RBETA, C, N)
    INTEGER    :: N
    REAL(q)    :: RA(N)
    COMPLEX(q) :: RALPHA(N), RBETA(N)
    GDEF       :: C(:,:)
  ! local
    REAL(q)    :: RRA
    COMPLEX(q) :: RRALPHA, RRBETA
    GDEF       :: RC(N)
    INTEGER    ::  L, IR, I, J

    IF (N==0) RETURN

    L=N/2+1
    IR=N
10  CONTINUE
    IF(L.GT.1)THEN
       L=L-1
       RRA    =RA(L)
       RRALPHA=RALPHA(L)
       RRBETA =RBETA(L)
       RC(:)  =C(1:N,L)
    ELSE
       RRA    =RA(IR)
       RRALPHA=RALPHA(IR)
       RRBETA =RBETA(IR)
       RC(:)  =C(1:N,IR)

       RA(IR)    =RA(1)
       RALPHA(IR)=RALPHA(1)
       RBETA(IR) =RBETA(1)
       C(1:N,IR) =C(1:N,1)

       IR=IR-1
       IF(IR.EQ.1)THEN
          RA(1)    =RRA
          RALPHA(1)=RRALPHA
          RBETA(1) =RRBETA
          C(1:N,1) =RC(:)
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
          RA(I)    =RA(J)
          RBETA(I) =RBETA(J)
          RALPHA(I)=RALPHA(J)
          C(1:N,I) =C(1:N,J)
          I=J
          J=J+J
       ELSE
          J=IR+1
       ENDIF
       GO TO 20
    ENDIF
    RA(I)    =RRA
    RALPHA(I)=RRALPHA
    RBETA(I) =RRBETA
    C(1:N,I) =RC
    GO TO 10

  END SUBROUTINE SORT_EIGENVAL


#ifdef tttt      
#endif

!************************ SUBROUTINE ETAINI_GW    ***********************
!
! this subroutine calculates the preconditioned step for the pseudo
! Hamiltonian (delta eta) from CHAM
! see FBN: Freysoldt, Boeck, Neugebauer, Phys. Rev. B 79, 241103 (2009)
! note that VASP stores search direction respectively the negative 
! gradient
! this routine is similar to ETAINI in rot.F, but slimmed down
! for damped MD algorithms (not applicable to conjugate gradient)
!
!   CELTOT          diagonal components of current pseudo Hamiltonian eta
!                   (epsilon in FBN paper)
!   CHAM on entry:  Hamiltonian matrix <phi_n | H | phi_m >
!        on exit:   search direction for pseudo Hamiltonian eta
!   CHF  on entry:  previous search direction for pseudo Hamiltonian eta
!        on exit:   previous search direction for pseudo Hamiltonian eta
!   ORT2            orthogonality of current gradient to previous rotation
!                   matrix
!   DE2             expected energy change along present rotation matrix
!   WEIMIN          threshhold for empty bands
!   CEIDB           work array
!
! differences to ETAINI_GW W%CELTOT here corresponds to W_F%CELTOT in rot.F
! diagonals of CHAM are used instead of W
!
!***********************************************************************

      SUBROUTINE ETAINI_GW(WDES, W, NBANDSGW_K, CHAM, CHF, CEIDB, KPOINTS, EFERMI, WEIMIN, & 
           ORT2, DE2, ORTCEL, DECEL )
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
      TYPE (wavespin)    W
      INTEGER :: NBANDSGW_K(WDES%NKPTS,WDES%ISPIN)
      GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN) 
      GDEF CHF(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
      GDEF CEIDB(WDES%NB_TOT,WDES%NB_TOT)
      TYPE (kpoints_struct) KPOINTS
      REAL(q) EFERMI   ! chemical potential, Fermi energy
      REAL(q) WEIMIN
      REAL(q) ORT2, DE2
      REAL(q) ORTCEL, DECEL

! local
      INTEGER ISP, NK, N, N1, N2, NC
      REAL(q) WEIGHT, DIFFER, DIFCEL, THETA
      REAL(q), PARAMETER ::  DIFMAX=1E-8   ! threshhold for degeneragy
      GDEF CROT, CGRAD
      REAL(q) :: WSUM, DFUN, SFUN, X, EFERMI_SHIFT, WMIN, DER

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
            IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code
#ifdef MPI
!            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!     presently the routine operates on all cores
            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,NBANDSGW_K(NK,ISP)
               X=  (EFERMI- REAL( W%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               WSUM =WSUM +DFUN/KPOINTS%SIGMA* WEIGHT
               EFERMI_SHIFT=EFERMI_SHIFT+DFUN*(W%CELTOT(N,NK,ISP)-CHAM(N,N,NK,ISP))/KPOINTS%SIGMA*WEIGHT
            ENDDO
         ENDDO
      ENDDO

!      CALLMPI(M_sum_d(WDES%COMM_KINTER,WSUM,1))
!      CALLMPI(M_sum_d(WDES%COMM_KINTER,EFERMI_SHIFT,1))

      ! shift is the first order change in the Fermi-energy
      IF (ABS(WSUM)>1E-10) THEN
         EFERMI_SHIFT=EFERMI_SHIFT/WSUM
      ELSE
         EFERMI_SHIFT=0
      ENDIF
      WMIN=WSUM/NBANDSGW_K(NK,ISP)*WEIMIN

      DECEL=0.
      ORTCEL=0.

      DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
            IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code
#ifdef MPI
!            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            WEIGHT =WDES%WTKPT(NK)*WDES%RSPIN
            DO N =1,NBANDSGW_K(NK,ISP)
               X=  (EFERMI- REAL( W%CELTOT(N,NK,ISP) ,KIND=q) )/KPOINTS%SIGMA
               CALL DELSTP(KPOINTS%ISMEAR,X,DFUN,SFUN)
               DER = -DFUN/KPOINTS%SIGMA

               ! avoid conjugation of those elements which do not contribute to energy
               IF (ABS(DFUN)<WMIN) THEN
                   CHF(N,N,NK,ISP)=0
               ENDIF

               DFUN= DFUN*((W%CELTOT(N,NK,ISP)-CHAM(N,N,NK,ISP)-EFERMI_SHIFT)/KPOINTS%SIGMA)
               ! preconditioned gradient equal gradient
               SFUN=(W%CELTOT(N,NK,ISP)-CHAM(N,N,NK,ISP))

               ! first order change for move along diagonal components
               DECEL  =DECEL  +DFUN* SFUN*WEIGHT
               ORTCEL =ORTCEL +DFUN* CHF(N,N,NK,ISP)*WEIGHT
               ! store search direction in CHAM
               CHAM(N,N,NK,ISP)=SFUN
            ENDDO
         ENDDO
      ENDDO

!      CALLMPI(M_sum_d(WDES%COMM_KINTER,DECEL,1))
!      CALLMPI(M_sum_d(WDES%COMM_KINTER,ORTCEL,1))

!=======================================================================
! calculate off diagonal elements of rotation matrix
!=======================================================================
      ORT2=0.
      DE2=0.

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
          IF (WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code
#ifdef MPI
!         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CEIDB=0
         WEIGHT =WDES%WTKPT(NK)
         
         NC=0
         DO N2=1,NBANDSGW_K(NK,ISP)
            DO N1=1,NBANDSGW_K(NK,ISP)
              IF (N1/=N2) THEN
              NC=NC+1
              DIFFER=     (W%FERTOT(N1,NK,ISP)-W%FERTOT(N2,NK,ISP))
              ! gradient
              CGRAD=WDES%RSPIN*CHAM(N2,N1,NK,ISP)*DIFFER

              DIFCEL= REAL( W%CELTOT(N2,NK,ISP)-W%CELTOT(N1,NK,ISP) ,KIND=q)
              IF (ABS(DIFCEL)>DIFMAX) THEN
                 CGRAD= CGRAD/DIFCEL
              ELSE
                 CGRAD=0
              ENDIF

              CROT  = CHAM(N1,N2,NK,ISP)

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
         DO N2=1,NBANDSGW_K(NK,ISP)
            DO N1=1,NBANDSGW_K(NK,ISP)
               IF (N1/=N2) THEN
                  CHAM(N1,N2,NK,ISP)=CEIDB(N1,N2)
               ENDIF
            ENDDO
         ENDDO
!         CALL DUMP_HAM( "rotation matrix",WDES, CHAM(1,1,NK,ISP))

      ENDDO
      ENDDO

!      CALLMPI (M_sum_d(WDES%COMM_KINTER,ORT2,1) )
!      CALLMPI (M_sum_d(WDES%COMM_KINTER,DE2,1) )


    END SUBROUTINE ETAINI_GW


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
! CUNI  auxilary matrices
!
!***********************************************************************

      SUBROUTINE ROTETA_GW(NDIM,NBANDS,CHF,CELTOT,STEP,CUNI,CEIDB)
      USE prec

      IMPLICIT NONE
      INTEGER NDIM   ! leading dimension of matrices
      INTEGER NBANDS ! actual rank
      GDEF CHF(NDIM,NDIM)
      GDEF CUNI(NDIM,NDIM),CEIDB(NDIM,NDIM)
      COMPLEX(q) :: CELTOT(NDIM)
      REAL(q) :: STEP
      REAL(q) :: HFEIG(NDIM)
    ! local
      INTEGER :: N1, N2, IFAIL
      REAL(q) :: W(3*NDIM)
      INTEGER, PARAMETER :: LWORK=32
      GDEF     CWRK(LWORK*NDIM)

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
            CUNI(N1,N2)=-CHF(N1,N2)*STEP
         ENDDO
         CUNI(N1,N1)=CELTOT(N1)-CHF(N1,N1)*STEP
      ENDDO

#ifdef  gammareal
      CALL DSYEV &
     &         ('V','U',NBANDS,CUNI,NDIM,HFEIG,CWRK,LWORK*NDIM, &
     &           IFAIL)
#else
      CALL ZHEEV &
     &         ('V','U',NBANDS,CUNI,NDIM,HFEIG,CWRK,LWORK*NDIM, &
     &           W,  IFAIL)
#endif
      IF (IFAIL/=0) THEN
         WRITE(*,*) 'ERROR ROTDIA: Call to routine ZHEEV failed! '// &
     &              'Error code was ',IFAIL
         STOP
      ENDIF

      ! store current (updated) eigenvalues back in CELTOT
      CELTOT(1:NBANDS)=HFEIG(1:NBANDS)
!=======================================================================
! transform CHF (equ. (21) of FBN)
!=======================================================================
      ! CEIDB = U+ CHF
      CALL GGEMM( trans, 'N', NBANDS, NBANDS, NBANDS, one, CUNI, &
           &             NDIM, CHF, NDIM, zero, CEIDB, NDIM)

     ! CHF = (U+ CHF ) U
      CALL GGEMM( 'N', 'N', NBANDS, NBANDS, NBANDS, one, CEIDB, &
           &             NDIM, CUNI, NDIM, zero, CHF, NDIM)
!=======================================================================
! rotate the wavefunction (with CEIDB)
!=======================================================================
      CEIDB=0
      CEIDB(1:NBANDS,1:NBANDS)=CUNI(1:NBANDS,1:NBANDS)
      CHF=GCONJG(CHF)

      RETURN
      END SUBROUTINE ROTETA_GW


END MODULE WAVE_CACHER




