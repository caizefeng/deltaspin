#include "symbol.inc"
      MODULE elphon
      USE prec
      IMPLICIT NONE

      PUBLIC ELPH_READER, ELPH_OVERLAP_W_WP, ELPH_LWSWQ

      PRIVATE

      LOGICAL, SAVE :: LWSWQ = .FALSE.

      CONTAINS

!***********************************************************************
!******************** PUBLIC PROCEDURES ********************************
!***********************************************************************


!******************** SUBROUTINE ELPH_READER ***************************
!
! Reads from the INCAR file
!
!***********************************************************************

      SUBROUTINE ELPH_READER(IU0,IU5)
      USE base
      USE vaspxml
      IMPLICIT NONE

      INTEGER IU5,IU6,IU0
      ! local variables
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      CALL RDATAB(LOPEN,INCAR,IU5,'LWSWQ','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LWSWQ,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LWSWQ'' from file INCAR.'
         LWSWQ=.FALSE.
      ENDIF

      CALL XML_INCAR('LWSWQ','L',IDUM,RDUM,CDUM,LWSWQ,CHARAC,N)

      CLOSE(IU5)

      RETURN
      END SUBROUTINE ELPH_READER

!******************** SUBROUTINE ELPH_OVERLAP_W_WP *********************
!
! Reads the file WAVECAR.q and compute the S-overlap between the
! wavefunctions in WAVECAR.q (W_P) and the current wavefunctions (W):
!
!  C^k_mn = < W_k,m | S | W_P_k,n >
!
!***********************************************************************

      SUBROUTINE ELPH_OVERLAP_W_WP(W,LATT_CUR,LMDIM,CQIJ,NONLR_S,NONL_S,IO)
      USE base
      USE fileio
      USE lattice
      USE wave_high
      USE dfast
      USE nonl_high
      TYPE (wavespin) W
      TYPE (latt) LATT_CUR
      TYPE (in_struct) IO
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      ! local variables
      TYPE (wavespin) W_P
      TYPE (wavefuna) WA,WOVL
      TYPE (wavedes1), TARGET :: WDES1
      GDEF, ALLOCATABLE :: COVL(:,:)

      INTEGER ISP,NK,NB_TOT,NA,NB
      INTEGER NPOS, NSTRIP
      LOGICAL LWREAD

      CALL INWAV_ALTERNATIVE(IO,W%WDES,W_P,W%WDES%GRID,LATT_CUR,LWREAD,'qqq')

      CALL PROALL (W%WDES%GRID,LATT_CUR,NONLR_S,NONL_S,W_P)

      IF (.NOT.LWREAD) THEN
         IF (IO%IU0>=0) WRITE(IO%IU0,*) 'ELPH_OVERLAP_W_WP: ERROR: unable to read WAVECAR.qqq'
         CALL DEALLOCW(W_P)
         RETURN
      ENDIF

      NB_TOT=W%WDES%NB_TOT
      NSTRIP=NSTRIP_STANDARD_GLOBAL

      ALLOCATE(COVL(NB_TOT,NB_TOT))
      COVL = 0

      CALL SETWDES(W%WDES,WDES1,0)

      CALL NEWWAVA_PROJ(WOVL,WDES1)

      ! Generate output file
      IF (IO%IU0 >= 0) THEN
         OPEN(UNIT = 1447, FILE = 'WSWQ', STATUS = 'REPLACE')
      ENDIF

      spin: DO ISP=1,W%WDES%ISPIN
      kpoint: DO NK=1,W%WDES%NKPTS

         CALL SETWDES(W%WDES,WDES1,NK)

         ! let WA point to the wavefunctions at k-point NK and spin component ISP
         WA=ELEMENTS(W,WDES1,ISP)
         ! redistribute to "over-pw-components"
         CALL REDISTRIBUTE_PW(WA)
         CALL REDISTRIBUTE_PROJ(WA)

         WOVL%WDES1=>WDES1
         ! let WOVL%CW point to the pw-components of W_P at k-point NK and spin component ISP
         WOVL%CW=>W_P%CW(:,:,WDES1%NK,ISP)
         IF (WDES1%DO_REDIS) THEN
            CALL SET_WPOINTER(WOVL%CW_RED,WDES1%NRPLWV_RED,W%WDES%NB_TOT,WOVL%CW(1,1))
         ELSE
            WOVL%CW_RED=>WOVL%CW
         ENDIF
         ! redistribute to "over-pw-components
         CALL REDISTRIBUTE_PW(WOVL)
         ! compute \sum_j Q_ij < p_j | W_P >
         CALL OVERL(WDES1,WDES1%LOVERL,LMDIM,CQIJ(1,1,1,ISP),W_P%CPROJ(1,1,NK,ISP),WOVL%CPROJ(1,1))
         ! and redistribute
         CALL REDISTRIBUTE_PROJ(WOVL)

         ! compute < W_k,m | S | W_P_k,n >
         DO NPOS=1,NB_TOT-NSTRIP,NSTRIP
            CALL ORTH2(WA%CW_RED(1, 1), WOVL%CW_RED(1, NPOS), WA%CPROJ_RED(1, 1), WOVL%CPROJ_RED(1, NPOS), &
            &          NB_TOT, NPOS, NSTRIP, WDES1%NPL_RED, &
            &          WDES1%NPRO_O_RED, WDES1%NRPLWV_RED, WDES1%NPROD_RED, COVL(1, 1))
         ENDDO

         CALL ORTH2(WA%CW_RED(1, 1), WOVL%CW_RED(1, NPOS), WA%CPROJ_RED(1, 1), WOVL%CPROJ_RED(1, NPOS), &
         &          NB_TOT, NPOS, NB_TOT-NPOS+1, WDES1%NPL_RED, &
         &          WDES1%NPRO_O_RED, WDES1%NRPLWV_RED, WDES1%NPROD_RED, COVL(1, 1))

         CALLMPI( M_sum_g(W%WDES%COMM,COVL(1,1),NB_TOT*NB_TOT))

         ! back to original data distribution
         CALL REDISTRIBUTE_PROJ(WA)
         CALL REDISTRIBUTE_PW(WA)

         ! Write spin-component and k-point to file
         IF (IO%IU0 >= 0) THEN
            WRITE(1447, '(A, I1, A, I6)') '   spin=', ISP, ', kpoint=', NK
            DO NA = 1, W%WDES%NB_TOT
               DO NB = 1, W%WDES%NB_TOT
                  WRITE(1447, '(A, I6, A, I6, A) ', ADVANCE = 'No') 'i=', NB, ', j=', NA, ' : '
#ifdef  gammareal
                  WRITE(1447, '(F12.6, A, F12.6)') COVL(NB, NA), ' ', 0
#else
                  WRITE(1447, '(F12.6, A, F12.6)') REAL(COVL(NB, NA)), ' ', AIMAG(COVL(NB, NA))
#endif
               ENDDO
            ENDDO
         ENDIF

      ENDDO kpoint
      ENDDO spin

      IF (IO%IU0 >= 0) THEN
         CLOSE(1447)
      ENDIF

      NULLIFY(WOVL%CW)
      CALL DELWAVA_PROJ(WOVL)

      CALL DEALLOCW(W_P)

      DEALLOCATE(COVL)

      RETURN
      END SUBROUTINE ELPH_OVERLAP_W_WP

!***********************************************************************
!******************** PUBLIC QUERY FUNCTIONS ***************************
!***********************************************************************

      FUNCTION ELPH_LWSWQ()
      IMPLICIT NONE
      LOGICAL ELPH_LWSWQ
      ELPH_LWSWQ=LWSWQ
      END FUNCTION ELPH_LWSWQ

      END MODULE elphon
