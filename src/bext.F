#include "symbol.inc"
      MODULE bexternal
      USE prec
      IMPLICIT NONE

      PUBLIC BEXT_READER,LBEXTERNAL,BEXT,BEXT_ADDV

      PRIVATE

      REAL(q), SAVE :: BEXT(3)=0

      LOGICAL, SAVE :: LBEXT=.FALSE.

      CONTAINS

!***********************************************************************
!******************** PUBLIC PROCEDURES ********************************
!***********************************************************************

!******************** SUBROUTINE BEXT_READER ***************************
!
! Reads BEXT from the INCAR file
!
!***********************************************************************

      SUBROUTINE BEXT_READER(IU0,IU5)
      USE base
      USE vaspxml

      INTEGER       :: IU5,IU6,IU0
      ! local variables
      INTEGER       :: IDUM, N, IERR
      REAL(q)       :: RDUM
      COMPLEX(q)    :: CDUM
      LOGICAL       :: LOPEN,LDUM
      CHARACTER (1) :: CHARAC

      REAL(q)       :: BNORM2
      INTEGER       :: ISPIN,NREQ
      LOGICAL       :: LSORBIT,LNONCOLLINEAR

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

! read in flag ISPIN
      ISPIN=1
      CALL RDATAB(LOPEN,INCAR,IU5,'ISPIN','=','#',';','I', &
     &            ISPIN,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ISPIN'' from file INCAR.'
         ISPIN=1
      ENDIF
! read in flag LSORBIT
      LSORBIT=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LSORBIT','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LSORBIT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LSORBIT'' from file INCAR.'
         LSORBIT=.FALSE.
      ENDIF
! read in flag LNONCOLLINEAR
      LNONCOLLINEAR=LSORBIT
      CALL RDATAB(LOPEN,INCAR,IU5,'LNONCOLLINEAR','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LNONCOLLINEAR,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LNONCOLLINEAR'' from file INCAR.'
         LNONCOLLINEAR=.FALSE.
      ENDIF
      IF (LSORBIT) LNONCOLLINEAR=LSORBIT

      NREQ=0
      IF (LNONCOLLINEAR) THEN
         NREQ=3
      ELSEIF (ISPIN==2) THEN
         NREQ=1
      ENDIF

      IF (NREQ/=0) THEN
! read in flag BEXT
         CALL RDATAB(LOPEN,INCAR,IU5,'BEXT','=','#',';','F', &
        &            IDUM,BEXT(1:NREQ),CDUM,LDUM,CHARAC,N,NREQ,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
        &                    ((IERR==0).AND.(N<NREQ))) THEN
            IF (IU0>=0) &
               WRITE(IU0,*)'Error reading item ''BEXT'' from file INCAR.'
            BEXT=0
         ENDIF

         CALL XML_INCAR_V('BEXT','F',IDUM,BEXT,CDUM,LDUM,CHARAC,N)

         BNORM2=BEXT(1)*BEXT(1)+BEXT(2)*BEXT(2)+BEXT(3)*BEXT(3)
         IF (BNORM2>1.E-6_q) THEN
            LBEXT=.TRUE.
         ENDIF
      ENDIF

      CLOSE(IU5)

      RETURN
      END SUBROUTINE BEXT_READER


!******************** FUNCTION LBEXTERNAL ******************************
!
!***********************************************************************

      FUNCTION LBEXTERNAL()
      LOGICAL LBEXTERNAL
      LBEXTERNAL=LBEXT
      END FUNCTION LBEXTERNAL


!******************** SUBROUTINE BEXT_ADDV *****************************
!
!***********************************************************************

      SUBROUTINE BEXT_ADDV(CVTOT,GRIDC,NCDIJ)
      USE mgrid
      
      TYPE (grid_3d) GRIDC

      COMPLEX(q)  :: CVTOT(GRIDC%MPLWV,NCDIJ)
      INTEGER     :: NCDIJ
      ! local variables
      INTEGER     :: I

      DO I=2,NCDIJ
         CALL ADD2VG0(CVTOT(1,I),GRIDC,CMPLX(BEXT(I-1),0._q,KIND=q))
      ENDDO

      RETURN
      END SUBROUTINE BEXT_ADDV

      END MODULE bexternal


!******************** SUBROUTINE ADD2VG0 *******************************
!
!***********************************************************************

      SUBROUTINE ADD2VG0(CVTOT,GRIDC,C)
      USE mgrid
      TYPE (grid_3d) GRIDC
      COMPLEX(q)  :: CVTOT(GRIDC%RC%NROW,GRIDC%RC%NCOL)
      COMPLEX(q)  :: C
      ! local variables
      INTEGER N1,N2,N3,NC

      N1=1; N2=1; N3=1

      DO NC=1,GRIDC%RC%NCOL
         IF (GRIDC%RC%I2(NC)==N2 .AND. GRIDC%RC%I3(NC)==N3) THEN
            CVTOT(NC,N1)=CVTOT(NC,N1)+C
         ENDIF
      ENDDO

      RETURN
      END SUBROUTINE ADD2VG0
