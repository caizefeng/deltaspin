!#define dotiming
!#define debug
#include "symbol.inc"

!***********************************************************************
!
! this subroutine finds clusters of wavefunctions that lie within
! a certain energy interval (determined by DEGENERACY_THRESHOLD)
! subroutines for determining such clusters, and for rotating
! wavefunctions within such a cluster are available
! the clusters are stored in a linked list for each k-point
!
!***********************************************************************


MODULE subrot_cluster
  USE prec
  IMPLICIT NONE
  TYPE eigenf_cluster
     INTEGER :: CLUSTER_SIZE
     INTEGER :: BAND_START                  ! index of first band
     INTEGER :: BAND_END                    ! index of last band
     GDEF,POINTER :: U(:,:)                 ! unitary rotation matrix
     GDEF,POINTER :: U_COMMULATIVE(:,:)     ! commulative rotation matrix
     TYPE(eigenf_cluster), POINTER :: next  ! pointer to next cluster
  END TYPE eigenf_cluster

  ! we need an array of pointers with dimension WDES%NKPTS,WDES%ISPIN
  ! to structures of the type eigenf_cluster
  ! but that construct does not exit in F90
  ! hence we have an additional type which just contains a single
  ! pointer, and use arrays of this type
  TYPE eigenf_cluster_pointer
     TYPE(eigenf_cluster), POINTER :: deg_cluster
  END TYPE eigenf_cluster_pointer

  !
  ! The parameter determines whether two eigenvalues
  ! are treated as degenerate or not.
  ! This is parameter plays a rather subtle role
  ! If it is chosen too large, the non degenerate eigenvalue
  ! pairs might be dedected as degenerate pairs;
  ! then the inverse iteration will try to converge them
  ! but subrot_lr will work against it.
  ! Ultimately the program will get stuck with bad convergence.
  ! On the other hand too small values might not spot all degenerate eigenvalues
  ! The most severe problems are caused by very narrow semicore bands
  ! A new method using complex shifts is implemented now in most
  ! routines and is clearly more robust
  REAL(q) :: DEGENERACY_THRESHOLD=2E-3_q

  INTEGER :: RESOLVE_DEG_NTIMES=2

CONTAINS

!************************* FIND_DEG_CLUSTERS ***************************
!
! this subroutine finds clusters of wavefunctions that lie within
! a certain energy interval (determined by DEGENERACY_THRESHOLD)
!
!***********************************************************************

  SUBROUTINE FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER, THRESHOLD, LDISTRIBUTED_IN )
    USE wave_mpi
    USE wave

    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
    REAL(q), OPTIONAL :: THRESHOLD
    LOGICAL, OPTIONAL :: LDISTRIBUTED_IN
! local
    REAL(q) :: DIFCEL
    INTEGER ISP, NK, NB_START, NB_END
    REAL(q) :: MY_THRESHOLD
    LOGICAL :: LDISTRIBUTED

    LDISTRIBUTED=.TRUE.
    IF (PRESENT(LDISTRIBUTED_IN)) THEN
       LDISTRIBUTED=LDISTRIBUTED_IN
    ENDIF

    IF (PRESENT(THRESHOLD) ) THEN
       MY_THRESHOLD=THRESHOLD
    ELSE
       MY_THRESHOLD=DEGENERACY_THRESHOLD
    ENDIF

    IF (ASSOCIATED(DEG_CLUSTER)) THEN
       CALL FREE_DEG_CLUSTERS(WDES,DEG_CLUSTER)
    ENDIF

    ALLOCATE(DEG_CLUSTER(WDES%NKPTS,WDES%ISPIN))

    IF (SIZE(DEG_CLUSTER,1) /= WDES%NKPTS) THEN
       WRITE(0,*) 'internal error in FIND_DEG_CLUSTERS: dimension one wrong',SIZE(DEG_CLUSTER,1),WDES%NKPTS
       STOP
    ENDIF
    IF (SIZE(DEG_CLUSTER,2) /= WDES%ISPIN) THEN
       WRITE(0,*) 'internal error in FIND_DEG_CLUSTERS: dimension two wrong',SIZE(DEG_CLUSTER,2),WDES%ISPIN
       STOP
    ENDIF

    spin:  DO ISP=1,WDES%ISPIN
       kpoint: DO NK=1,WDES%NKPTS
          NULLIFY(DEG_CLUSTER(NK, ISP)%DEG_CLUSTER)

#ifdef MPI
          IF (LDISTRIBUTED .AND. MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          NB_START=1
          DO
             DO NB_END=NB_START,WDES%NB_TOT-1
                DIFCEL= REAL( W%CELTOT(NB_END+1,NK,ISP)-W%CELTOT(NB_END,NK,ISP) ,KIND=q)
                IF (ABS(DIFCEL)>MY_THRESHOLD) EXIT
             ENDDO
             ! found a new cluster
             IF (NB_END>NB_START) THEN
                CALL ADD_DEG_CLUSTER(DEG_CLUSTER(NK,ISP)%DEG_CLUSTER,NB_START,NB_END)
#ifdef debug
                WRITE(*,'(A,I4,A,2I4,(10F8.4))') 'found at nk',NK,' cluster', &
                     NB_START,NB_END,W%CELTOT(NB_START:NB_END,NK,ISP)
#endif
                IF (.NOT.ASSOCIATED(DEG_CLUSTER(NK,ISP)%DEG_CLUSTER)) THEN
                   WRITE(0,*) 'internal error in FIND_DEG_CLUSTERS: association failed'
                   STOP
                ENDIF

             ENDIF
             ! goto next band, and possible exit loop
             NB_START=NB_END+1
             IF (NB_START>WDES%NB_TOT) EXIT
          ENDDO
       END DO kpoint
    ENDDO spin

  END SUBROUTINE FIND_DEG_CLUSTERS


  SUBROUTINE REINIT_DEG_CLUSTERS(WDES, DEG_CLUSTER)
    USE wave_mpi
    USE wave
    TYPE (wavedes)     WDES

    TYPE (eigenf_cluster_pointer) :: DEG_CLUSTER(:,:)

    INTEGER ISP, NK
    spin:  DO ISP=1,SIZE(DEG_CLUSTER,2)
       kpoint: DO NK=1,SIZE(DEG_CLUSTER,1)
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          CALL REINIT_DEG_CLUSTER(DEG_CLUSTER(NK, ISP)%DEG_CLUSTER)
     END DO kpoint
    ENDDO spin

  END SUBROUTINE REINIT_DEG_CLUSTERS

!************************* FREE_DEGENERATE_CLUSTERS ********************
!
! this subroutine frees the list of degeneray clusters for all
! k-points and spins
!
!***********************************************************************


  SUBROUTINE FREE_DEG_CLUSTERS(WDES, DEG_CLUSTER)
    USE wave_mpi
    USE wave
    TYPE (wavedes)     WDES

    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)

    INTEGER ISP, NK
    spin:  DO ISP=1,SIZE(DEG_CLUSTER,2)
       kpoint: DO NK=1,SIZE(DEG_CLUSTER,1)
          CALL FREE_DEG_CLUSTER(DEG_CLUSTER(NK, ISP)%DEG_CLUSTER)
     END DO kpoint
    ENDDO spin

    DEALLOCATE(DEG_CLUSTER)


  END SUBROUTINE FREE_DEG_CLUSTERS


!************************* ADD_DEG_CLUSTER *****************************
!
! ADD_DEG_CLUSTER 
!    adds an entry to the linked list of degenerated eigenfunction clusters
! CLEAN_DEG_CLUSTER 
!    frees an entire list of clusters
!
! written recursively because it is much shorter
! the problem with an iterative implementation is that there is nothing
! like a pointer to a pointer in F90, therefore we would have
! complicated IF cases for the initial node in the list
! (whereas dummy arguments to pointer pass their association status back
!  to the CALLER, in some way they behave like pointers to pointer)
!
!***********************************************************************
  
  RECURSIVE SUBROUTINE ADD_DEG_CLUSTER(root, NB_START, NB_END)
    IMPLICIT NONE
    INTEGER NB_START, NB_END
    TYPE (eigenf_cluster),POINTER :: root
    INTEGER I
    
    IF (.NOT.ASSOCIATED(root)) THEN
       ALLOCATE(root)
       ALLOCATE(root%U(NB_END-NB_START+1,NB_END-NB_START+1))
       ALLOCATE(root%U_COMMULATIVE(NB_END-NB_START+1,NB_END-NB_START+1))
       root%U_COMMULATIVE=0
       DO I=1,NB_END-NB_START+1
          root%U_COMMULATIVE(I,I)=1
       ENDDO
       
       root%BAND_START=NB_START
       root%BAND_END  =NB_END
       NULLIFY(root%NEXT)
    ELSE
       CALL ADD_DEG_CLUSTER(root%NEXT, NB_START, NB_END)
    ENDIF
  END SUBROUTINE ADD_DEG_CLUSTER

  RECURSIVE SUBROUTINE FREE_DEG_CLUSTER(root)
    TYPE (eigenf_cluster),POINTER :: root

    IF (.NOT.ASSOCIATED(root)) RETURN
    CALL FREE_DEG_CLUSTER(root%NEXT)
    IF (ASSOCIATED(root%U)) DEALLOCATE(root%U)
    IF (ASSOCIATED(root%U_COMMULATIVE)) DEALLOCATE(root%U_COMMULATIVE)
    DEALLOCATE(root)
    NULLIFY(root)
    RETURN
  END SUBROUTINE FREE_DEG_CLUSTER

  RECURSIVE SUBROUTINE REINIT_DEG_CLUSTER(root)
    TYPE (eigenf_cluster),POINTER :: root
    INTEGER I

    IF (.NOT.ASSOCIATED(root)) RETURN
    root%U=0
    root%U_COMMULATIVE=0
    DO I=1,SIZE(root%U_COMMULATIVE,1)
       root%U_COMMULATIVE(I,I)=1
    ENDDO
    CALL REINIT_DEG_CLUSTER(root%NEXT)
    RETURN
  END SUBROUTINE REINIT_DEG_CLUSTER


!************************* SETUPT_DEG_CLUSTERS *************************
!
! setup the unitary transformation matrix for a cluster of
! degenerated eigenvalue/eigenfunction pairs of the unperturbed
! problem
!
!***********************************************************************

  SUBROUTINE SETUP_DEG_CLUSTERS(DEG_CLUSTER_BASE, CHAM, CHAM_DIAGONAL, CHAM_DIAGONAL0)
    USE wave_mpi
    USE wave

    TYPE (eigenf_cluster),POINTER :: DEG_CLUSTER_BASE
    TYPE (eigenf_cluster),POINTER :: DEG_CLUSTER
    GDEF :: CHAM(:,:)
    COMPLEX(q) :: CHAM_DIAGONAL0(:)
    COMPLEX(q) :: CHAM_DIAGONAL(:)
! local
    INTEGER :: NB_START, NB_END, NB_TOT
    INTEGER, PARAMETER :: NMAX_DEG=48
    GDEF :: U(NMAX_DEG,NMAX_DEG)
! work arrays for unitary transformation
    GDEF :: CWRK(NMAX_DEG*NMAX_DEG)
    REAL(q) :: RWORK(3*NMAX_DEG)
    REAL(q) :: R(NMAX_DEG)  
    INTEGER :: NBANDS, N, NP
    INTEGER :: IFAIL
    
    NBANDS = SIZE(CHAM_DIAGONAL)
    IF (NBANDS /= SIZE(CHAM,1)) THEN
       WRITE(0,*) 'internal error in SETUP_DEG_CLUSTER: CHAM is not conform'
       STOP
    ENDIF
    DEG_CLUSTER=>DEG_CLUSTER_BASE

    DO
       IF (.NOT.ASSOCIATED(DEG_CLUSTER)) EXIT
       NB_START=DEG_CLUSTER%BAND_START
       NB_END=DEG_CLUSTER%BAND_END
       NB_TOT=NB_END-NB_START+1
       
       IF (NB_TOT>NMAX_DEG) THEN
          WRITE(0,*) 'internal error in SETUP_DEG_CLUSTERS: NB_TOT exceeds NMAX_DEG'
          WRITE(0,*) '   increase NMAX_DEG to',NB_TOT
          STOP
       ENDIF
       DO N=1,NB_TOT
          DO NP=1,NB_TOT
             U(N,NP)=CHAM(NB_START+N-1,NB_START+NP-1)
          ENDDO
       ENDDO
       DO N=1,NB_TOT
          U(N,N)=U(N,N)+CHAM_DIAGONAL(NB_START+N-1)
       ENDDO
       CHAM(NB_START:NB_END,NB_START:NB_END)=0
#ifdef debug
        DO N=1,NB_TOT
          WRITE(*,1)N,REAL( U(N,1:NB_TOT) ,KIND=q)
        ENDDO
         WRITE(*,*)
#ifndef gammareal
        DO N=1,NB_TOT
          WRITE(*,2)N,AIMAG(U(N,1:NB_TOT))
        ENDDO
        WRITE(*,*)
#endif
#endif
      1 FORMAT(1I2,3X,20F9.5)
      2 FORMAT(1I2,3X,20E9.1)

#ifdef  gammareal
       CALL DSYEV &
            ('V','U',NB_TOT,U(1,1),NMAX_DEG, &
            R,CWRK,NMAX_DEG*NMAX_DEG, IFAIL)
#else
       CALL ZHEEV &
            ('V','U',NB_TOT,U(1,1),NMAX_DEG, &
            R,CWRK,NMAX_DEG*NMAX_DEG, RWORK,  IFAIL)
#endif

       IF (IFAIL/=0) THEN
          WRITE(0,*) 'ERROR in SETUP_DEG_CLUSTERS: call to ZHEEV/DSYEV failed! '// &
               &              'error code was ',IFAIL
          STOP
       ENDIF
#ifdef debug
        DO N=1,NB_TOT
          WRITE(*,1)N,REAL( U(N,1:NB_TOT) ,KIND=q)
        ENDDO
         WRITE(*,*)
#ifndef gammareal
        DO N=1,NB_TOT
          WRITE(*,2)N,AIMAG(U(N,1:NB_TOT))
        ENDDO
        WRITE(*,*)
#endif
#endif
#undef debug
       IF (SIZE(DEG_CLUSTER%U,1)/=NB_TOT) THEN
          WRITE(0,*) 'internal ERROR in SETUP_DEG_CLUSTERS: incorrect allocation'
          STOP
       ENDIF

       DEG_CLUSTER%U=U(1:NB_TOT,1:NB_TOT)

       ! accumate the matrix in U_COMMULATIVE
       CALL GGEMM('N', 'N', NB_TOT, NB_TOT, NB_TOT, one, &
     &               DEG_CLUSTER%U_COMMULATIVE(1,1), NB_TOT, DEG_CLUSTER%U(1,1), &
     &               NB_TOT, zero, U, NMAX_DEG)

       DEG_CLUSTER%U_COMMULATIVE=U(1:NB_TOT,1:NB_TOT)

       DEG_CLUSTER=>DEG_CLUSTER%NEXT
    ENDDO

  END SUBROUTINE SETUP_DEG_CLUSTERS

!**********************  ZERO_HAM_DEG_CLUSTERS *************************
!
! zero the Hamiltonian for the sub-matrices spanned by 
! by degenerated bands of the unperturbed Hamiltonian
!
!***********************************************************************

  SUBROUTINE ZERO_HAM_DEG_CLUSTERS(DEG_CLUSTER_BASE, CHAM)
    USE wave_mpi
    USE wave

    TYPE (eigenf_cluster),POINTER :: DEG_CLUSTER_BASE
    TYPE (eigenf_cluster),POINTER :: DEG_CLUSTER
    GDEF :: CHAM(:,:)
! local
    INTEGER :: NB_START, NB_END, NB_TOT
! work arrays for unitary transformation

    DEG_CLUSTER=>DEG_CLUSTER_BASE

    DO
       IF (.NOT.ASSOCIATED(DEG_CLUSTER)) EXIT
       NB_START=DEG_CLUSTER%BAND_START
       NB_END=DEG_CLUSTER%BAND_END
       NB_TOT=NB_END-NB_START+1
       
       NB_END=MIN(NB_END, SIZE(CHAM,2))
       IF (NB_START<= SIZE(CHAM,2)) THEN
          CHAM(NB_START:NB_END,NB_START:NB_END)=0
       ENDIF
       DEG_CLUSTER=>DEG_CLUSTER%NEXT
    ENDDO

  END SUBROUTINE ZERO_HAM_DEG_CLUSTERS


!************************* SUBROT_DEG_ALL          ********************
!
! perform a sub space rotation in the space spanned by clusters 
! of eigenvalues for all k-points and spins
! the wavefunction array is redistributed over plane wave coeffiecients
! LCOMMULATIVE == .TRUE. use U_COMMULATIVE
! LCONJG       == .TRUE. use the conjugated transformation matrix
!
!***********************************************************************

  SUBROUTINE SUBROT_DEG_ALL(WDES, W, LCONJG, LCOMMULATIVE, DEG_CLUSTER )
      USE prec
      USE wave_mpi
      USE wave

      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W
      LOGICAL LCONJG                 ! use the conjugation of the stored matrix
      LOGICAL LCOMMULATIVE           ! use the commulative matrix transform
      TYPE (eigenf_cluster_pointer) :: DEG_CLUSTER(:,:)
      
! redistributed plane wave coefficients
      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      COMPLEX(q), POINTER :: CW_RED(:,:)
      GDEF   , POINTER :: CPROJ_RED(:,:)
      LOGICAL DO_REDIS
      INTEGER :: NRPLWV_RED, NPROD_RED, NPL, NPRO
      INTEGER :: NB_TOT, NBANDS, ISP, NK
      INTEGER :: IONODE, NODE_ME, NCPU

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
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

!=======================================================================
      spin:  DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         IF (ASSOCIATED(DEG_CLUSTER(NK,ISP)%DEG_CLUSTER)) THEN

         CALL SETWDES(WDES,WDES1,NK)
         IF (DO_REDIS) THEN
            CALL SET_WPOINTER(CW_RED,    NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
            CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
         ELSE
            CW_RED    => W%CW(:,:,NK,ISP)
            CPROJ_RED => W%CPROJ(:,:,NK,ISP)
         ENDIF

         !   set number of wavefunctions after redistribution
         NPL = WDES1%NPL     ! number of plane waves/node after data redistribution
         NPRO= WDES1%NPRO    ! number of projected wavef. after data redistribution
         CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

         IF (DO_REDIS) THEN
            CALL REDIS_PW  (WDES1, NBANDS, W%CW    (1,1,NK,ISP))
            CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ (1,1,NK,ISP))
         ENDIF

         CALL SUBROT_DEG_CLUSTERS(WDES, NPL, NPRO, NRPLWV_RED, NPROD_RED, &
              CW_RED, CPROJ_RED, DEG_CLUSTER(NK,ISP)%DEG_CLUSTER, LCONJG, LCOMMULATIVE)
         
         IF (DO_REDIS) THEN
            CALL REDIS_PW  (WDES1, NBANDS, W%CW    (1,1,NK,ISP))
            CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ (1,1,NK,ISP))
         ENDIF
         ENDIF
      END DO kpoint
      END DO spin
      
    END SUBROUTINE SUBROT_DEG_ALL

!************************* SUBROT_DEG_CLUSTERS *************************
!
! perform a sub space rotation in the space spanned by cluster using
! the unitary tranformation U or U+
! it is assumed that the bands are already distributed over plane wave 
! coefficients
!
!***********************************************************************

  SUBROUTINE SUBROT_DEG_CLUSTERS(WDES, NPL, NPRO, NRPLWV_RED, NPROD_RED, &
       CW_RED, CPROJ_RED, DEG_CLUSTER_, LCONJG, LCOMMULATIVE)
    USE wave_mpi
    USE wave
    USE dfast

    TYPE (wavedes)     WDES
    TYPE (eigenf_cluster),TARGET  :: DEG_CLUSTER_
    TYPE (eigenf_cluster),POINTER :: DEG_CLUSTER
    INTEGER :: NPL, NPRO, NRPLWV_RED, NPROD_RED
    COMPLEX(q) :: CW_RED(NRPLWV_RED,WDES%NB_TOT)
    GDEF       :: CPROJ_RED(NPROD_RED ,WDES%NB_TOT)
    LOGICAL :: LCONJG
    LOGICAL :: LCOMMULATIVE

! local
    INTEGER :: NB_START, NB_END, NB_TOT

    DEG_CLUSTER=> DEG_CLUSTER_

    DO
       IF (.NOT.ASSOCIATED(DEG_CLUSTER)) EXIT

       NB_START=DEG_CLUSTER%BAND_START
       NB_END=DEG_CLUSTER%BAND_END
       NB_TOT=NB_END-NB_START+1

       IF (LCOMMULATIVE) THEN
       IF (.NOT. LCONJG) THEN
       CALL LINCOM('F',CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:),DEG_CLUSTER%U_COMMULATIVE(:,:), &
             NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
             CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:))
       ELSE
       CALL LINCOM(trans,CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:),DEG_CLUSTER%U_COMMULATIVE(:,:), &
             NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
             CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:))
       ENDIF
       ELSE
       IF (.NOT. LCONJG) THEN
       CALL LINCOM('F',CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:),DEG_CLUSTER%U(:,:), &
             NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
             CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:))
       ELSE
       CALL LINCOM(trans,CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:),DEG_CLUSTER%U(:,:), &
             NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
             CW_RED(1:,NB_START:),CPROJ_RED(1:,NB_START:))
       ENDIF
       ENDIF

       DEG_CLUSTER=>DEG_CLUSTER%NEXT
    ENDDO


  END SUBROUTINE SUBROT_DEG_CLUSTERS

END MODULE subrot_cluster
