#include "symbol.inc"
!************************************************************************
! RCS:  $Id: wave_mpi.F,v 1.3 2001/04/03 10:43:18 kresse Exp $
!
!  this module contains the routines required to communicate
!  wavefunctions and projected wavefunctions
!
!  there are also two quite tricky routines SET_WPOINTER
!  which allow to generate pointer to F77-sequenced arrays
!  There is no guarantee that this will work on all computers, but
!  currently it seems to be ok
!
!***********************************************************************

  MODULE wave_mpi
    USE prec
    USE mpimy
!
! I need an interface block here because I pass the
! first element of a pointer to CW
! most F90 comiler do not allow such a construct
!
      INTERFACE
      SUBROUTINE SET_WPOINTER(CW_P, N1, N2, CW)
      USE prec
      INTEGER N1,N2
      COMPLEX(q), POINTER :: CW_P(:,:)
      COMPLEX(q), TARGET  :: CW
      END SUBROUTINE
      END INTERFACE

      INTERFACE
      SUBROUTINE SET_GPOINTER(CW_P, N1, N2, CW)
      USE prec
      INTEGER N1,N2
      GDEF, POINTER :: CW_P(:,:)
      GDEF, TARGET  :: CW
      END SUBROUTINE
      END INTERFACE


      ! each communcation package needs a unique identifier (ICOMM)
      ! this is handled by the global variable ICOMM_BASE_HANDLE
      ! which is incremented / decremented by ICOMM_INCREMENT
      ! whenever a redistribution handle is allocated

      INTEGER, PARAMETER :: ICOMM_BASE=10000, ICOMM_INCREMENT=1000
      INTEGER, SAVE :: ICOMM_BASE_HANDLE=ICOMM_BASE

      ! is assyncronous communication allowed or not

      LOGICAL :: LASYNC=.FALSE.

      TYPE REDIS_PW_CTR
         INTEGER :: NB                   ! number of bands to be redistributed
         INTEGER :: NBANDS               ! maximum band index
         INTEGER :: NEXT                 ! next vacant slot
         COMPLEX(q), POINTER :: CW(:,:)  ! storage for redistribution
         INTEGER,POINTER :: BAND(:)      ! bands that are currently redistributed
         INTEGER,POINTER :: SREQUEST(:,:)! handles for outstanding send requests
         INTEGER,POINTER :: RREQUEST(:,:)! handles for outstanding receive requests
         INTEGER :: ICOMM                ! identifier for communication
         TYPE(communic),POINTER :: COMM  ! communication handle
      END TYPE REDIS_PW_CTR


    CONTAINS
!************************ SUBROUTINE REDIS_PW_ALLOC ********************
! 
! This subroutine allocates the storage required for asyncronous
! redistribution of wave function coeffients
! the redistribution can be initiated for up to NSIM bands 
! at the same time
!
!***********************************************************************

    SUBROUTINE REDIS_PW_ALLOC(WDES, NSIM, H)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)   :: WDES
      INTEGER :: NSIM                  ! number of bands done simultaneausly
      TYPE (redis_pw_ctr),POINTER :: H ! handle
#ifdef MPI
      INTEGER NRPLWV

      NRPLWV=WDES%NRPLWV

      IF (MOD(NRPLWV,WDES%NB_PAR) /= 0) THEN
        WRITE(*,*)'REDIS_PW_ALLOC: internal error(1) ',NRPLWV,WDES%NB_PAR
        STOP
      ENDIF

      ALLOCATE(H)
      H%NB=NSIM
      H%NBANDS=WDES%NBANDS
      H%COMM=>WDES%COMM_INTER
      H%ICOMM=ICOMM_BASE_HANDLE

      ALLOCATE(H%SREQUEST(H%COMM%NCPU,NSIM), H%RREQUEST(H%COMM%NCPU,NSIM))
      ALLOCATE(H%CW(NRPLWV,NSIM),H%BAND(NSIM))
      H%BAND=0

      H%NEXT=1

      ICOMM_BASE_HANDLE=ICOMM_BASE_HANDLE+ICOMM_INCREMENT
#endif
    END SUBROUTINE REDIS_PW_ALLOC


    SUBROUTINE REDIS_PW_DEALLOC(H)
      USE prec
      USE wave
      IMPLICIT NONE

      INTEGER :: NSIM                  ! number of bands done simultaneausly
      TYPE (redis_pw_ctr),POINTER :: H ! handle
      INTEGER :: IND
#ifdef MPI
      IF (.NOT.ASSOCIATED(H)) THEN
        WRITE(*,*)'REDIS_PW_DEALLOC: internal error(1)'
        STOP
      ENDIF

      DO IND=1, H%NB
         IF ( H%BAND(IND) /= 0) THEN
            WRITE(0,*)'internal error: REDIS_PW_DEALLOC not all lines STOPED'
            STOP
         ENDIF
      ENDDO
         
      DEALLOCATE(H%CW,H%BAND)
      DEALLOCATE(H%SREQUEST, H%RREQUEST)
      DEALLOCATE(H)

      ICOMM_BASE_HANDLE=ICOMM_BASE_HANDLE-ICOMM_INCREMENT
      IF (ICOMM_BASE_HANDLE < ICOMM_BASE) THEN
        WRITE(*,*)'REDIS_PW_DEALLOC: internal error(2)',ICOMM_BASE_HANDLE,ICOMM_BASE
        STOP
      ENDIF
#endif
    END SUBROUTINE REDIS_PW_DEALLOC


  END MODULE wave_mpi

!=======================================================================
!
! this routine returns a pointer to an SEQUENCED F77 like array
! with a given storage convention
! 1. dimension is N1 second one N2
!
!=======================================================================

      SUBROUTINE SET_WPOINTER(CW_P, N1, N2, CW)
      USE prec
      IMPLICIT NONE
      INTEGER N1,N2
      COMPLEX(q), POINTER :: CW_P(:,:)
      COMPLEX(q), TARGET  :: CW(N1,N2)
      CW_P => CW(:,:)

      END SUBROUTINE

      SUBROUTINE SET_GPOINTER(CW_P, N1, N2, CW)
      USE prec
      IMPLICIT NONE
      INTEGER N1,N2
      GDEF, POINTER :: CW_P(:,:)
      GDEF, TARGET  :: CW(N1,N2)
      CW_P => CW(:,:)

      END SUBROUTINE

!=======================================================================
!  distribute projector part of wavefunctions over all nodes
!=======================================================================

      SUBROUTINE DIS_PROJ(WDES1,CPROJ,CPROJ_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      GDEF CPROJ(WDES1%NPRO_TOT)
      GDEF CPROJ_LOCAL(WDES1%NPRO)
#ifdef MPI
      INTEGER NIS,NPRO,NT,LMMAXC,NPRO_POS,L,NI,NPRO_TOT
!
! quick copy if possible
      IF (WDES1%COMM_INB%NCPU==1) THEN
         CPROJ_LOCAL(1:WDES1%NPRO)=CPROJ(1:WDES1%NPRO)
         RETURN
      ENDIF

      CALLMPI( M_sum_g(WDES1%COMM_INB, CPROJ, WDES1%NPRO_TOT))

      NIS =1
      NPRO=0
      DO NT=1,WDES1%NTYP
        LMMAXC=WDES1%LMMAX(NT)
        IF (LMMAXC==0) GOTO 100
          DO NI=NIS,WDES1%NITYP(NT)+NIS-1
          NPRO_POS=WDES1%NPRO_POS(NI)
          DO L=1,LMMAXC
             CPROJ_LOCAL(L+NPRO)=CPROJ(L+NPRO_POS)
          ENDDO
          NPRO= LMMAXC+NPRO
          ENDDO
 100    NIS = NIS+WDES1%NITYP(NT)
      ENDDO

      IF (WDES1%LNONCOLLINEAR) THEN
         
         NIS =1
         NPRO_TOT= WDES1%NPRO_TOT/2

         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC==0) GOTO 200
            DO NI=NIS,WDES1%NITYP(NT)+NIS-1
            NPRO_POS=WDES1%NPRO_POS(NI)
            DO L=1,LMMAXC
               CPROJ_LOCAL(L+NPRO)=CPROJ(L+NPRO_POS+NPRO_TOT)
            ENDDO
            NPRO= LMMAXC+NPRO
            ENDDO
 200        NIS = NIS+WDES1%NITYP(NT)
         ENDDO
      ENDIF


#else
      CPROJ_LOCAL(1:WDES1%NPRO)=CPROJ(1:WDES1%NPRO)
#endif
      END SUBROUTINE

!=======================================================================
!  merge projector part wavefunctions from all nodes
!  definitely not the most efficient implementation
!  but it works
!=======================================================================

      SUBROUTINE MRG_PROJ(WDES1,CPROJ,CPROJ_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      GDEF CPROJ(WDES1%NPRO_TOT)
      GDEF CPROJ_LOCAL(WDES1%NPRO)
#ifdef MPI
      INTEGER NIS,NPRO,NT,LMMAXC,NPRO_POS,L,NI,NPRO_TOT

!
! quick copy if possible
      IF (WDES1%COMM_INB%NCPU==1) THEN
         CPROJ(1:WDES1%NPRO)=CPROJ_LOCAL(1:WDES1%NPRO)
         RETURN
      ENDIF

      CPROJ(1:WDES1%NPRO_TOT)=0

      NIS =1
      NPRO=0
      DO NT=1,WDES1%NTYP
        LMMAXC=WDES1%LMMAX(NT)
        IF (LMMAXC==0) GOTO 100
          DO NI=NIS,WDES1%NITYP(NT)+NIS-1
          NPRO_POS=WDES1%NPRO_POS(NI)
          DO L=1,LMMAXC
            CPROJ(L+NPRO_POS)=CPROJ_LOCAL(L+NPRO)
          ENDDO;
          NPRO= LMMAXC+NPRO
          ENDDO
 100    NIS = NIS+WDES1%NITYP(NT)
      ENDDO
      
      IF (WDES1%LNONCOLLINEAR) THEN
         
         NIS =1
         NPRO_TOT= WDES1%NPRO_TOT/2

         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC==0) GOTO 200
            DO NI=NIS,WDES1%NITYP(NT)+NIS-1
            NPRO_POS=WDES1%NPRO_POS(NI)
            DO L=1,LMMAXC
               CPROJ(L+NPRO_POS+NPRO_TOT)=CPROJ_LOCAL(L+NPRO)
            ENDDO
            NPRO= LMMAXC+NPRO
            ENDDO
 200        NIS = NIS+WDES1%NITYP(NT)
         ENDDO
      ENDIF

      CALLMPI( M_sum_g(WDES1%COMM_INB,CPROJ, WDES1%NPRO_TOT))

#else
      CPROJ(1:WDES1%NPRO)=CPROJ_LOCAL(1:WDES1%NPRO)
#endif

      END SUBROUTINE

!=======================================================================
!  distribute plane wave part of one wavefunction over all COMM_INB nodes
!  the supplied wavefunction must have the standard serial layout and
!  must be defined on the node COMM_INB%IONODE
!=======================================================================

      SUBROUTINE DIS_PW(WDES1,CW,CW_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NPL)
#ifdef MPI
      INTEGER NC,IND,I
      INTEGER NPL_TOT,NPL

      CALLMPI( M_bcast_z(WDES1%COMM_INB, CW, WDES1%NPL_TOT))

#ifdef  MPI_barrier_after_bcast
      CALLMPI( MPI_barrier( WDES1%COMM_INB%MPI_COMM, I))
#endif

      IND=0
      DO NC=1,WDES1%NCOL
        DO I=1,WDES1%PL_COL(NC)
          CW_LOCAL(IND+I)=CW(WDES1%PL_INDEX(NC)+I)
        ENDDO
        IND=IND+WDES1%PL_COL(NC)
      ENDDO

      IF ( WDES1%LNONCOLLINEAR ) THEN
         IND=0
         NPL=WDES1%NGVECTOR
         NPL_TOT=WDES1%NPL_TOT / 2
         DO NC=1,WDES1%NCOL
            DO I=1,WDES1%PL_COL(NC)
               CW_LOCAL(IND+I+NPL)=CW(WDES1%PL_INDEX(NC)+I+NPL_TOT)
            ENDDO
            IND=IND+WDES1%PL_COL(NC)
         ENDDO
      ENDIF

      IF (IND*WDES1%NRSPINORS/=WDES1%NPL) THEN
        WRITE(*,*)'internal error in DIS_PW:',IND,WDES1%NPL
        STOP
      ENDIF
#else
      CW_LOCAL(1:WDES1%NPL)=CW(1:WDES1%NPL)
#endif
      END SUBROUTINE


!=======================================================================
!  merge plane wave part of wavefunctions from all COMM_INB nodes
!  using an index array 
!  only used when reading orbitals in Gamma only
!=======================================================================

      SUBROUTINE DIS_PW_GAMMA(WDES1,CW,CW_LOCAL, INDEX, LCONJG)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NPL)
      LOGICAL :: LCONJG(WDES1%NGVECTOR)
      INTEGER :: INDEX(WDES1%NGVECTOR)
#ifdef MPI
      INTEGER NC,IND,I
      INTEGER NPL_TOT,NPL

      CALLMPI( M_bcast_z(WDES1%COMM_INB, CW, WDES1%NPL_TOT))

#ifdef  MPI_barrier_after_bcast
      CALLMPI( MPI_barrier( WDES1%COMM_INB%MPI_COMM, I))
#endif

      DO IND=1,WDES1%NGVECTOR
         IF (LCONJG(IND)) THEN
            CW_LOCAL(IND)=CONJG(CW(INDEX(IND)))
         ELSE
            CW_LOCAL(IND)=CW(INDEX(IND))
         ENDIF
      ENDDO

      IF ( WDES1%LNONCOLLINEAR ) THEN
         NPL=WDES1%NGVECTOR
         NPL_TOT=WDES1%NPL_TOT / 2
         DO IND=1,WDES1%NGVECTOR
            IF (LCONJG(IND)) THEN
               CW_LOCAL(IND+NPL)=CONJG(CW(INDEX(IND)+NPL_TOT))
            ELSE
               CW_LOCAL(IND+NPL)=CW(INDEX(IND)+NPL_TOT)
            ENDIF
         ENDDO
      ENDIF

#else
      CW_LOCAL(1:WDES1%NPL)=CW(1:WDES1%NPL)
#endif
      END SUBROUTINE


!=======================================================================
! distribute one specific band to nodes
! only required after i.e. reading wavefunctions
! it is sufficient if wavefunctions are defined on master node
! but mind, that all nodes must call this communication routine
!=======================================================================

      SUBROUTINE DIS_PW_BAND(WDES1, NB, CW, CW_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE
      INTEGER NB,NB_L
      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NRPLWV,WDES1%NBANDS)
      INTEGER ierror
#ifdef MPI

! first copy wavefunction to all nodes (it would be sufficient to
!     copy it to master nodes in each inband-communicator, but
!     I am not sure who the master node is)
      CALLMPI( M_bcast_z(WDES1%COMM_KIN, CW, WDES1%NPL_TOT))
#ifdef  MPI_barrier_after_bcast
      CALLMPI( MPI_barrier( WDES1%COMM_KIN%MPI_COMM, ierror ))
#endif

      NB_L=NB_LOCAL(NB,WDES1)
      IF ( MOD(NB-1,WDES1%NB_PAR)+1 == WDES1%NB_LOW) THEN
        CALL DIS_PW(WDES1,CW,CW_LOCAL(1,NB_L))
      ENDIF
#else
      CALL DIS_PW(WDES1,CW,CW_LOCAL(1,NB))
#endif
      END SUBROUTINE

!
! new version uses an index array and works even if parallel or seriell FFT is used
!

      SUBROUTINE DIS_PW_BAND_GAMMA(WDES1, NB, CW, CW_LOCAL, INDEX, LCONJG)
      USE prec
      USE wave
      IMPLICIT NONE
      INTEGER NB,NB_L
      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NRPLWV,WDES1%NBANDS)
      LOGICAL :: LCONJG(WDES1%NGVECTOR)
      INTEGER :: INDEX(WDES1%NGVECTOR)
      INTEGER ierror
#ifdef MPI

! first copy wavefunction to all nodes (it would be sufficient to
!     copy it to master nodes in each inband-communicator, but
!     I am not sure who the master node is)
      CALLMPI( M_bcast_z(WDES1%COMM_KIN, CW, WDES1%NPL_TOT))
#ifdef  MPI_barrier_after_bcast
      CALLMPI( MPI_barrier( WDES1%COMM_KIN%MPI_COMM, ierror ))
#endif

      NB_L=NB_LOCAL(NB,WDES1)
      IF ( MOD(NB-1,WDES1%NB_PAR)+1 == WDES1%NB_LOW) THEN
         CALL DIS_PW_GAMMA(WDES1,CW,CW_LOCAL(1,NB_L), INDEX, LCONJG)
      ENDIF
#else
      ! serial version is always ok
      CALL DIS_PW(WDES1,CW,CW_LOCAL(1,NB))
#endif
      END SUBROUTINE


!=======================================================================
!  merge plane wave part of wavefunctions from all COMM_INB nodes
!=======================================================================

      SUBROUTINE MRG_PW(WDES1,CW,CW_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NPL)
#ifdef MPI
      INTEGER NC,IND,I
      INTEGER NPL_TOT,NPL

      CW=0
      IND=0
      DO NC=1,WDES1%NCOL
        DO I=1,WDES1%PL_COL(NC)
          CW(WDES1%PL_INDEX(NC)+I)=CW_LOCAL(IND+I)
        ENDDO
        IND=IND+WDES1%PL_COL(NC)
      ENDDO

      IF ( WDES1%LNONCOLLINEAR ) THEN
         IND=0
         NPL=WDES1%NGVECTOR
         NPL_TOT=WDES1%NPL_TOT / 2
         DO NC=1,WDES1%NCOL
            DO I=1,WDES1%PL_COL(NC)
               CW(WDES1%PL_INDEX(NC)+I+NPL_TOT)=CW_LOCAL(IND+I+NPL)
            ENDDO
            IND=IND+WDES1%PL_COL(NC)
         ENDDO
      ENDIF

      IF (IND*WDES1%NRSPINORS/=WDES1%NPL) THEN
        WRITE(*,*)'internal error MRG_PW:',IND,WDES1%NPL
        STOP
      ENDIF

      CALLMPI( M_sum_z(WDES1%COMM_INB, CW(1), WDES1%NPL_TOT))
#else
      CW(1:WDES1%NPL)=CW_LOCAL(1:WDES1%NPL)
#endif
      END SUBROUTINE


!=======================================================================
!  merge plane wave part of wavefunctions from all COMM_INB nodes
!  using an index array
!  only used when writing orbitals in Gamma only
!=======================================================================

      SUBROUTINE MRG_PW_GAMMA(WDES1, CW, CW_LOCAL, INDEX, LCONJG)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NPL)
      LOGICAL :: LCONJG(WDES1%NGVECTOR)
      INTEGER :: INDEX(WDES1%NGVECTOR)
#ifdef MPI
      INTEGER IND
      INTEGER NPL_TOT,NPL

      CW=0
      DO IND=1,WDES1%NGVECTOR
         IF (LCONJG(IND)) THEN
            CW(INDEX(IND))=CONJG(CW_LOCAL(IND))
         ELSE
            CW(INDEX(IND))=CW_LOCAL(IND)
         ENDIF
      ENDDO

      IF ( WDES1%LNONCOLLINEAR ) THEN
         NPL=WDES1%NGVECTOR
         NPL_TOT=WDES1%NPL_TOT / 2
         DO IND=1,WDES1%NGVECTOR
            IF (LCONJG(IND)) THEN
               CW(INDEX(IND)+NPL_TOT)=CONJG(CW_LOCAL(IND+NPL))
            ELSE
               CW(INDEX(IND)+NPL_TOT)=CW_LOCAL(IND+NPL)
            ENDIF
         ENDDO
      ENDIF

      CALLMPI( M_sum_z(WDES1%COMM_INB, CW(1), WDES1%NPL_TOT))
#else
      CW(1:WDES1%NPL)=CW_LOCAL(1:WDES1%NPL)
#endif
      END SUBROUTINE

!=======================================================================
! merge  band NB from all nodes into a local copy CW
! only required for i.e. writing wavefunctions
!=======================================================================

      SUBROUTINE MRG_PW_BAND(WDES1, NB, CW, CW_LOCAL)
      USE prec
      USE wave
      IMPLICIT NONE
      INTEGER NB,NB_L
      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NRPLWV,WDES1%NBANDS)
#ifdef MPI
      CW=0
      NB_L=NB_LOCAL(NB,WDES1)
! if wavefunction is located on this node merge it to CW
      IF ( MOD(NB-1,WDES1%NB_PAR)+1 == WDES1%NB_LOW) THEN
        CALL MRG_PW(WDES1,CW,CW_LOCAL(1,NB_L))
      ENDIF
! then merge over all nodes using inter-band communication
      CALLMPI( M_sum_z(WDES1%COMM_INTER, CW(1), WDES1%NPL_TOT))
#else
      CALL MRG_PW(WDES1,CW,CW_LOCAL(1,NB))
#endif
      END SUBROUTINE

!
! new version uses an index array and works even if parallel or seriell FFT is used
!

      SUBROUTINE MRG_PW_BAND_GAMMA(WDES1, NB, CW, CW_LOCAL, INDEX, LCONJG)
      USE prec
      USE wave
      IMPLICIT NONE
      INTEGER NB,NB_L

      TYPE (wavedes1)    WDES1
      COMPLEX(q) CW(WDES1%NPL_TOT)
      COMPLEX(q) CW_LOCAL(WDES1%NRPLWV,WDES1%NBANDS)
      LOGICAL :: LCONJG(WDES1%NGVECTOR)
      INTEGER :: INDEX(WDES1%NGVECTOR)
#ifdef MPI
      CW=0
      NB_L=NB_LOCAL(NB,WDES1)
! if wavefunction is located on this node merge it to CW
      IF ( MOD(NB-1,WDES1%NB_PAR)+1 == WDES1%NB_LOW) THEN
         CALL MRG_PW_GAMMA(WDES1,CW,CW_LOCAL(1,NB_L), INDEX, LCONJG)
      ENDIF
! then merge over all nodes using inter-band communication
      CALLMPI( M_sum_z(WDES1%COMM_INTER, CW(1), WDES1%NPL_TOT))
#else
      ! serial version is always ok
      CALL MRG_PW(WDES1,CW,CW_LOCAL(1,NB))
#endif
      END SUBROUTINE

!=======================================================================
! merge eigenvalues from all nodes
!=======================================================================

      SUBROUTINE MRG_CEL(WDES,W)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)    WDES
      TYPE (wavedes1)   WDES1
      TYPE (wavespin)   W
      INTEGER I,NK,NB_GLOBAL,NB,NCEL
#ifdef MPI
!
! first zero out all components from other nodes
!
      spin:   DO I=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
         DO NB_GLOBAL=1,WDES%NB_TOT
            W%CELTOT(NB_GLOBAL,NK,I)=0
         END DO
      ELSE
         CALL SETWDES(WDES,WDES1,NK)
         
         band:   DO NB_GLOBAL=1,WDES%NB_TOT
            NB=NB_LOCAL(NB_GLOBAL,WDES1)
            IF (NB==0) THEN
               W%CELTOT(NB_GLOBAL,NK,I)=0
            ENDIF
         ENDDO band
      ENDIF
      ENDDO kpoint
      ENDDO spin
!
! than merge CELTOT from all nodes
!
      NCEL=WDES%NB_TOT*WDES%NKPTS*WDES%ISPIN
      CALLMPI( M_sum_z( WDES%COMM_KINTER, W%CELTOT(1,1,1),NCEL))
      CALLMPI( M_sum_z( WDES%COMM_INTER, W%CELTOT(1,1,1),NCEL))
#endif
      ! finally set eigenvalues of states beyond WDES%NB_TOTK to large values
      CALL WFSET_HIGH_CELEN(W)

      END SUBROUTINE


!=======================================================================
! merge partial occupancies from all nodes
!=======================================================================

      SUBROUTINE MRG_FERWE(WDES,W)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)    WDES
      TYPE (wavedes1)   WDES1
      TYPE (wavespin)   W
      INTEGER I,NK,NB_GLOBAL,NB,NCEL
#ifdef MPI
!
! first zero out all components from other nodes
!
      spin:   DO I=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
         DO NB_GLOBAL=1,WDES%NB_TOT
            W%FERTOT(NB_GLOBAL,NK,I)=0
         END DO
      ELSE
      CALL SETWDES(WDES,WDES1,NK)

      band:   DO NB_GLOBAL=1,WDES%NB_TOT
        NB=NB_LOCAL(NB_GLOBAL,WDES1)
        IF (NB==0) THEN
          W%FERTOT(NB_GLOBAL,NK,I)=0
        ENDIF
      ENDDO band
      ENDIF
      ENDDO kpoint
      ENDDO spin
!
! than merge FERTOT from all nodes
!
      NCEL=WDES%NB_TOT*WDES%NKPTS*WDES%ISPIN
      CALLMPI( M_sum_d( WDES%COMM_KINTER, W%FERTOT(1,1,1),NCEL))
      CALLMPI( M_sum_d( WDES%COMM_INTER, W%FERTOT(1,1,1),NCEL))
#endif
      END SUBROUTINE


!=======================================================================
! merge auxilary array from all nodes
!=======================================================================

      SUBROUTINE MRG_AUX(WDES,W)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)    WDES
      TYPE (wavedes1)   WDES1
      TYPE (wavespin)   W
      INTEGER I,NK,NB_GLOBAL,NB,NCEL
#ifdef MPI
!
! first zero out all components from other nodes
!
      spin:   DO I=1,WDES%ISPIN
      kpoint: DO NK=1,WDES%NKPTS
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
         DO NB_GLOBAL=1,WDES%NB_TOT
            W%AUXTOT(NB_GLOBAL,NK,I)=0
         END DO
      ELSE
      CALL SETWDES(WDES,WDES1,NK)
      band:   DO NB_GLOBAL=1,WDES%NB_TOT
        NB=NB_LOCAL(NB_GLOBAL,WDES1)
        IF (NB==0) THEN
          W%AUXTOT(NB_GLOBAL,NK,I)=0
        ENDIF
      ENDDO band
      ENDIF
      ENDDO kpoint
      ENDDO spin
!
! than merge AUXTOT from all nodes
!
      NCEL=WDES%NB_TOT*WDES%NKPTS*WDES%ISPIN
      CALLMPI( M_sum_d( WDES%COMM_KINTER, W%AUXTOT(1,1,1),NCEL))
      CALLMPI( M_sum_d( WDES%COMM_INTER, W%AUXTOT(1,1,1),NCEL))
#endif
      END SUBROUTINE

!************************ SUBROUTINE REDIS_PW **************************
!
! redistribute plane wave coefficients from over band to over
! plane wave coefficient or vice versa
! this operation is done in place to reduce storage demands
!
! mind that if the routine is called twice the original distribution
! is obtained
!
!***********************************************************************

      SUBROUTINE REDIS_PW(WDES1, NBANDS, CW)
      USE prec
      USE wave
#ifdef CUDA_GPU
      USE cuda_interface
#endif
      IMPLICIT NONE

      TYPE (wavedes1)  WDES1
      INTEGER NBANDS
      COMPLEX(q) :: CW(WDES1%NRPLWV*NBANDS)
#if defined(T3D_SMA)
      INTEGER MALLOC_DONE
      INTEGER, EXTERNAL :: ISHM_CHECK
      COMMON /SHM/ MALLOC_DONE, PBUF
      POINTER ( PBUF, CWORK ) ;  COMPLEX(q):: CWORK(*)
#else
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
#endif
! local variables
      INTEGER :: NRPLWV,N,NB,INFO

#ifdef MPI
      ! quick return if possible
      IF (WDES1%COMM_INTER%NCPU ==1) RETURN

      NRPLWV=WDES1%NRPLWV

      IF (MOD(NRPLWV,WDES1%NB_PAR) /= 0) THEN
        WRITE(*,*)'REDIS_PW: internal error(1) ',NRPLWV,WDES1%NB_PAR
        STOP
      ENDIF

#ifdef T3D_SMA
      IF (ISHM_CHECK(NRPLWV*2) == 0) THEN
         WRITE(0,*) 'REDIS_PW: shmem workspace unsufficient'
         STOP
      ENDIF
#else
      ALLOCATE( CWORK(NRPLWV))
#endif
      DO NB=0,NBANDS-1
#ifdef CUDA_GPU
nv_profile_start(NVP_REDIS_MPI)
nv_profile_start(NVP_MPI)
#endif
         CALL M_alltoall_z(WDES1%COMM_INTER, NRPLWV, CW(NB*NRPLWV+1), CWORK(1))
#ifdef CUDA_GPU
nv_profile_stop(NVP_MPI)
nv_profile_stop(NVP_REDIS_MPI)
#endif 

#ifdef CUDA_GPU
nv_profile_start(NVP_REDIS_COPY)
#endif 
         CALL ZCOPY( NRPLWV, CWORK(1), 1, CW(NB*NRPLWV+1), 1)
#ifdef CUDA_GPU
nv_profile_stop(NVP_REDIS_COPY)
#endif
      ENDDO
#ifdef T3D_SMA
      ! nup
#else
      DEALLOCATE(CWORK)
#endif
#endif
      END SUBROUTINE

!************************ SUBROUTINE REDIS_PW_OVER_BANDS ***************
!
! redistribute all coefficients if and only if LOVER_BAND
! is .TRUE.
! after this call the bands are distributed over nodes
! and LOVER_BAND is .FALSE.
!
!***********************************************************************

      SUBROUTINE REDIS_PW_OVER_BANDS(WDES, W)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W

#ifdef MPI
      IF ( W%OVER_BAND ) THEN
         CALL REDIS_PW_ALL(WDES, W)
      ENDIF
#endif
      END SUBROUTINE REDIS_PW_OVER_BANDS

!************************ SUBROUTINE REDIS_PW_ALL **********************
!
! redistribute all plane wave coeffcients, change the order 
! W%OVER_BAND
! calling the routine twice restores the original order
!
!***********************************************************************

      SUBROUTINE REDIS_PW_ALL(WDES, W)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W

#ifdef MPI
      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      INTEGER NCPU, ISP, NK
      
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
      IF (NCPU /= 1) THEN
         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
            CALL SETWDES(WDES,WDES1,NK)
            CALL REDIS_PW  (WDES1, WDES%NBANDS, W%CW   (1,1,NK,ISP))
         ENDDO
         ENDDO

         W%OVER_BAND=.NOT. W%OVER_BAND
      ENDIF
#endif
      END SUBROUTINE REDIS_PW_ALL

!************************ SUBROUTINE REDIS_PW_START ********************
!
! redistribute plane wave coefficients from over band to over
! plane wave coefficient or vice versa asyncronously 
! one band is initiated
!
!***********************************************************************

      SUBROUTINE REDIS_PW_START(WDES, CW, BANDINDEX, H)
      USE prec
      USE wave_mpi
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)  WDES
      INTEGER BANDINDEX
      COMPLEX(q) :: CW(WDES%NRPLWV)
      TYPE(redis_pw_ctr) :: H
! local variables
      INTEGER :: NRPLWV
      INTEGER :: IND

#ifdef MPI
      IF (BANDINDEX>H%NBANDS) THEN
         WRITE(0,*) 'REDIS_PW_START: internal error, non existing band',BANDINDEX
      ENDIF
      NRPLWV=WDES%NRPLWV

      IND=H%NEXT
      H%NEXT=H%NEXT+1 ; IF (H%NEXT > H%NB) H%NEXT=1

      IF (H%BAND(IND)/=0) THEN
         WRITE(0,*)'REDIS_PW_START: internal error, slot NEXT not empty',H%NEXT
         STOP
      ENDIF
      
      H%BAND(IND)=BANDINDEX
!#define no_async
#ifdef no_async
      H%CW(1:NRPLWV,IND)=CW(1:NRPLWV)
#else
      CALL M_alltoall_d_async(H%COMM, 2*NRPLWV, CW(1), H%CW(1,IND), &
           H%ICOMM+IND, H%SREQUEST(1,IND), H%RREQUEST(1,IND) )
#endif
#endif
      END SUBROUTINE

!************************ SUBROUTINE REDIS_PW_STOP  ********************
!
! finish redistribution of  plane wave coefficients from over band to over
! plane wave coefficient
!
!***********************************************************************

      SUBROUTINE REDIS_PW_STOP(WDES, CW, BANDINDEX, H)
      USE prec
      USE wave_mpi
      USE wave
      IMPLICIT NONE

      TYPE (wavedes)  WDES
      INTEGER BANDINDEX
      COMPLEX(q) :: CW(WDES%NRPLWV)
      TYPE(redis_pw_ctr) :: H
! local variables
      INTEGER :: NRPLWV
      INTEGER :: IND,N
#ifdef MPI
      IF (BANDINDEX>H%NBANDS) THEN
         WRITE(0,*) 'REDIS_PW_STOP: internal error, non existing band',BANDINDEX
      ENDIF

      NRPLWV=WDES%NRPLWV

      ! usually the required band is in the slot NEXT, however
      ! possibly we have to search for it

      IND=H%NEXT
      IF (H%BAND(IND) /= BANDINDEX) THEN
         DO IND=1, H%NB
            IF ( H%BAND(IND) == BANDINDEX) EXIT
         ENDDO
         IND=MIN(IND,H%NB)
         IF  (H%BAND(IND) /= BANDINDEX) THEN
            WRITE(0,*)'REDIS_PW_STOP: internal error, can not find',BANDINDEX,H%BAND
            STOP
         ENDIF
      ENDIF
      ! fill in zero into the band index array
      H%BAND(IND)=0
      


#ifdef no_async
      CALL M_alltoall_d_async(H%COMM, 2*NRPLWV, H%CW(1,IND), CW(1) , &
           H%ICOMM+IND, H%SREQUEST(1,IND), H%RREQUEST(1,IND) )
      CALL M_alltoall_wait(H%COMM, H%SREQUEST(1,IND), H%RREQUEST(1,IND) )
#else
      ! wait for send and receive to finish
      CALL M_alltoall_wait(H%COMM, H%SREQUEST(1,IND), H%RREQUEST(1,IND) )
      ! copy CW 
      DO N=1,NRPLWV
         CW(N)=H%CW(N,IND)
      ENDDO
#endif
#endif
      END SUBROUTINE


!************************ SUBROUTINE REDIS_PROJ ************************
!
! redistribute projector part of wave function from over band to over
! plane wave coefficient or vice versa
! this operation is done in place to reduce storage demands
!
! mind that if the routine is called twice the original distribution
! is obtained
!
!***********************************************************************

      SUBROUTINE REDIS_PROJ(WDES1, NBANDS, CPROJ)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)  WDES1
      INTEGER NBANDS
      GDEF :: CPROJ(WDES1%NPROD*NBANDS)
#if defined(T3D_SMA)
      INTEGER MALLOC_DONE
      INTEGER, EXTERNAL :: ISHM_CHECK
      COMMON /SHM/ MALLOC_DONE, PBUF
      POINTER ( PBUF, CWORK ) ;  GDEF:: CWORK(*)
#else
      GDEF, ALLOCATABLE :: CWORK(:)
#endif

#ifdef gammareal
      INTEGER, PARAMETER :: MCOMP=1
#else
      INTEGER, PARAMETER :: MCOMP=2
#endif
! local variables
#ifdef MPI
      INTEGER :: NPRO,NPROD,N,NB,INFO

      IF (WDES1%COMM_INTER%NCPU ==1) RETURN

      NPRO =WDES1%NPRO
      NPROD=WDES1%NPROD
      IF (MOD(NPROD,WDES1%NB_PAR) /= 0) THEN
        WRITE(*,*)'REDIS_PW: internal error(1) ',NPROD,WDES1%NB_PAR
        STOP
      ENDIF
#ifdef T3D_SMA
      IF (ISHM_CHECK(NPROD*MCOMP) == 0) THEN
         WRITE(0,*) 'REDIS_PROJ: shmem workspace unsufficient'
         STOP
      ENDIF
#else
      ALLOCATE( CWORK(NPROD))
#endif
      DO NB=0,NBANDS-1
#ifdef gammareal
        CALL M_alltoall_d(WDES1%COMM_INTER, NPROD, CPROJ(NB*NPROD+1), CWORK(1))
#else
        CALL M_alltoall_z(WDES1%COMM_INTER, NPROD, CPROJ(NB*NPROD+1), CWORK(1))
#endif
        DO N=1,NPROD
          CPROJ(NB*NPROD+N)=CWORK(N)
        ENDDO
      ENDDO
#ifdef T3D_SMA
      ! nup
#else
      DEALLOCATE(CWORK)
#endif
#endif
      END SUBROUTINE


!************************ SUBROUTINE SET_NPL_NPRO **********************
!
! set the local number of plane waves on a node after redistribution
! of plane wave coefficients and projected wavefunctions
! on entry the global number of plane waves must be given
!
!***********************************************************************

      SUBROUTINE SET_NPL_NPRO(WDES1, NPL, NPRO)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavedes1)  WDES1
      INTEGER NPL,NPRO,NPL_REM,NPRO_REM,I
! local variables
      NPL_REM =NPL
      NPRO_REM=NPRO
#ifdef MPI
      NPL =WDES1%NRPLWV/WDES1%COMM_INTER%NCPU
      NPRO=WDES1%NPROD /WDES1%COMM_INTER%NCPU
      DO I=1,WDES1%COMM_INTER%NODE_ME
        NPL =MIN(NPL,NPL_REM)
        NPRO=MIN(NPRO,NPRO_REM)
        NPL_REM  =NPL_REM -NPL
        NPRO_REM =NPRO_REM-NPRO
      ENDDO
#else
      NPL=WDES1%NRPLWV
      NPRO=WDES1%NPROD
#endif

      END SUBROUTINE

!***********************************************************************
!
! Copy wavefunction coefficients from each k-point communicator to
! replicated copies in other communicators
! Should use a gather operation
!
!***********************************************************************

    SUBROUTINE KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
      USE prec
      USE wave
      USE wave_mpi
      USE mpimy
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      TYPE (wavespin)    W
      INTEGER :: ISP,K,inode

#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN

         DO ISP=1,WDES%ISPIN
            DO K=1,WDES%NKPTS
               inode=MOD(K-1,WDES%COMM_KINTER%NCPU)+1
               CALLMPI( M_bcast_z_from(WDES%COMM_KINTER, W%CW(1,1,K,ISP),SIZE(W%CW,1)*SIZE(W%CW,2),inode) )
#ifdef gammareal
               CALLMPI( M_bcast_d_from(WDES%COMM_KINTER, W%CPROJ(1,1,K,ISP),SIZE(W%CPROJ,1)*SIZE(W%CPROJ,2),inode) )
#else
               CALLMPI( M_bcast_z_from(WDES%COMM_KINTER, W%CPROJ(1,1,K,ISP),SIZE(W%CPROJ,1)*SIZE(W%CPROJ,2),inode) )
#endif
            END DO
         END DO
      ENDIF
#endif
    END SUBROUTINE KPAR_SYNC_WAVEFUNCTIONS

!***********************************************************************
!
! Copy eigenvalues from each k-point communicator to
! replicated copies in other communicators
!
!***********************************************************************

    SUBROUTINE KPAR_SYNC_CELTOT(WDES,W)
! Not the same as a MRG_CEL since we do not sum over within bands
      USE prec
      USE wave
      USE wave_mpi
      USE mpimy
      IMPLICIT NONE
      TYPE (wavedes) :: WDES
      TYPE (wavespin) :: W
      INTEGER :: ISP,K

#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
         DO ISP=1,WDES%ISPIN
            DO K=1,WDES%NKPTS
               IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                  W%CELTOT(:,K,ISP)=0.
               END IF
            END DO
         END DO
! Reduce and broadcast for simplicity. Really an allgather.
         CALLMPI( M_sum_master_z( WDES%COMM_KINTER, W%CELTOT, SIZE(W%CELTOT) ) )
         CALLMPI( M_bcast_z( WDES%COMM_KINTER, W%CELTOT, SIZE(W%CELTOT) ) )
      ENDIF
#endif

    END SUBROUTINE KPAR_SYNC_CELTOT

!***********************************************************************
!
! Copy occupancies from each k-point communicator to
! replicated copies in other communicators
!
!***********************************************************************

    SUBROUTINE KPAR_SYNC_FERTOT(WDES,W)
      USE prec
      USE wave
      USE wave_mpi
      USE mpimy
      IMPLICIT NONE
      TYPE (wavedes) :: WDES
      TYPE (wavespin) :: W
      INTEGER :: ISP,K

#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
         DO ISP=1,WDES%ISPIN
            DO K=1,WDES%NKPTS
               IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                  W%FERTOT(:,K,ISP)=0.
               END IF
            END DO
         END DO
! Reduce and broadcast for simplicity. Really an allgather.
         CALLMPI( M_sum_master_d( WDES%COMM_KINTER, W%FERTOT, SIZE(W%FERTOT) ) )
         CALLMPI( M_bcast_d( WDES%COMM_KINTER, W%FERTOT, SIZE(W%FERTOT) ) )
      ENDIF
#endif

    END SUBROUTINE KPAR_SYNC_FERTOT

!***********************************************************************
!
! Copy auxilary array from each k-point communicator to
! replicated copies in other communicators
!
!***********************************************************************

    SUBROUTINE KPAR_SYNC_AUXTOT(WDES,W)
      USE prec
      USE wave
      USE wave_mpi
      USE mpimy
      IMPLICIT NONE
      TYPE (wavedes) :: WDES
      TYPE (wavespin) :: W
      INTEGER :: ISP,K

#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
         DO ISP=1,WDES%ISPIN
            DO K=1,WDES%NKPTS
               IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                  W%AUXTOT(:,K,ISP)=0.
               END IF
            END DO
         END DO
! Reduce and broadcast for simplicity. Really an allgather.
         CALLMPI( M_sum_master_d( WDES%COMM_KINTER, W%AUXTOT, SIZE(W%AUXTOT) ) )
         CALLMPI( M_bcast_d( WDES%COMM_KINTER, W%AUXTOT, SIZE(W%AUXTOT) ) )
      ENDIF
#endif

    END SUBROUTINE KPAR_SYNC_AUXTOT

!***********************************************************************
!
! Sync orbitals, eigenvalues and occupancies
!
!***********************************************************************

    SUBROUTINE KPAR_SYNC_ALL(WDES,W)
      USE wave
      IMPLICIT NONE
      TYPE (wavedes) :: WDES
      TYPE (wavespin) :: W
#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_FERTOT(WDES,W)
         CALL KPAR_SYNC_CELTOT(WDES,W)
         CALL KPAR_SYNC_AUXTOT(WDES,W)
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
      ENDIF
#endif
    END SUBROUTINE KPAR_SYNC_ALL
