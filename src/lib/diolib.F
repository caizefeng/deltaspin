!************************** IOLIB *************************************
!                                                                     *
!     Contains some miscellaneous stuff for file-handling ... .       *
!     Note: should work "as is" on 99% of all platforms, but it       *
!     might happen that for few platforms some few changes could      *
!     become necessary --> read all comments in all subroutines!      *
!                                                                     *
!**********************************************************************

#ifndef use_FLUSH
      SUBROUTINE WFORCE(IUNIT)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! On UNIX system normally output is written asynchronously (that means
! data are written 'blocked' - first all data will be written to some
! buffer and if this buffer is full the buffer contents is written on
! the physical storage medium).
! Due to this technique usually data can be lost when the system crashes
! (those data which are still in the buffer and not yet on disk or tape)
! and therefore subroutine WFORCE should be used from time to time to
! force the system to write all data on a given unit on disk or tape
! by closing and reopening the data file (and positioning the record
! pointer to the last written record so that new data following this
! subroutine call are appended to the existing file).
! If you do not need this (non-UNIX system) you can easily make a dummy
! version of this routine by setting parameter DUMMY true.
! Caution: Unit 5 is not allowed to be closed (standard input), unit 0
! (standard error) is written immediately on disk or tape (standard
! error - it could contain helpful informations and therefore no data
! should be lost on that unit) and unit 6 (standard output) may only
! be closed under some special conditions (parameter ALLOW6!): Closing
! and reopening unit 6 works only if you redirect standard output to
! file fort.6 (e.g. a.out > fort.6), otherwise WFORCE ends abnormally!
! Last advice: If you use 'preconnected units' (standard filenames -
! usually fort.$iunit) set parameter PRECON true - if not set it false.
      LOGICAL       DUMMY,ALLOW6,PRECON,FOUND
      CHARACTER*255 FNAME,LINE
      CHARACTER*16  ACC,DFORM
      INTEGER       IUNIT,LENGTH,I,LRECL
! Here you can customize some things specific to your system/own needs
! DUMMY: act as dummy routine, ALLOW6: one may also close and reopen
! unit 6 (usually stdout ...), PRECON: 'preconnected units' are used
      PARAMETER(DUMMY=.FALSE.,ALLOW6=.FALSE.,PRECON=.FALSE.)
      EQUIVALENCE (FNAME,LINE)
! Do not allow units 0,5 (stderr and stdin) and maybe also 6 (stdout);
! maybe one might need to change this for special platforms / purposes
      IF (DUMMY.OR.((IUNIT.EQ.6).AND.(.NOT.ALLOW6)).OR.(IUNIT.EQ.5) &
     &                                          .OR.(IUNIT.LE.0)) RETURN
      INQUIRE(UNIT=IUNIT,EXIST=FOUND)
      IF (.NOT.FOUND) RETURN
      IF (.NOT.PRECON) THEN
         INQUIRE(UNIT=IUNIT,NAME=FNAME)
! red alert: some problem with INQUIRE occurred ...
         IF (FNAME(1:1).EQ.' ') RETURN
         DO 100 I=1,255
            IF (FNAME(I:I).NE.' ') LENGTH=I
  100    CONTINUE
      END IF
      INQUIRE(UNIT=IUNIT,FORM=DFORM)
      CALL UPPER(DFORM)
      INQUIRE(UNIT=IUNIT,ACCESS=ACC)
      CALL UPPER(ACC)
      RECL=0
      IF (ACC(1:6).EQ.'DIRECT') INQUIRE(UNIT=IUNIT,RECL=LRECL)
      CLOSE(IUNIT)
      IF (.NOT.PRECON) THEN
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH), &
     &                      ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH),FORM=DFORM, &
     &           POSITION='APPEND')
         ENDIF
      ELSE
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FORM=DFORM,POSITION='APPEND')
         ENDIF
      END IF
  400 RETURN
      END
#else
      SUBROUTINE WFORCE(IUNIT)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! On UNIX system normally output is written asynchronously (that means
! data are written 'blocked' - first all data will be written to some
! buffer and if this buffer is full the buffer contents is written on
! the physical storage medium).
! Due to this technique usually data can be lost when the system crashes
! (those data which are still in the buffer and not yet on disk or tape)
! and therefore subroutine WFORCE should be used from time to time to
! force the system to write all data on a given unit on disk or tape
! by closing and reopening the data file (and positioning the record
! pointer to the last written record so that new data following this
! subroutine call are appended to the existing file).
! If you do not need this (non-UNIX system) you can easily make a dummy
! version of this routine by setting parameter DUMMY true.
! Caution: Unit 5 is not allowed to be closed (standard input), unit 0
! (standard error) is written immediately on disk or tape (standard
! error - it could contain helpful informations and therefore no data
! should be lost on that unit) and unit 6 (standard output) may only
! be closed under some special conditions (parameter ALLOW6!): Closing
! and reopening unit 6 works only if you redirect standard output to
! file fort.6 (e.g. a.out > fort.6), otherwise WFORCE ends abnormally!
! Last advice: If you use 'preconnected units' (standard filenames -
! usually fort.$iunit) set parameter PRECON true - if not set it false.
      LOGICAL       DUMMY,ALLOW6,PRECON,FOUND
      CHARACTER*255 FNAME,LINE
      CHARACTER*16  ACC,DFORM
      INTEGER       IUNIT,LENGTH,I,LRECL
! Here you can customize some things specific to your system/own needs
! DUMMY: act as dummy routine, ALLOW6: one may also close and reopen
! unit 6 (usually stdout ...), PRECON: 'preconnected units' are used
      PARAMETER(DUMMY=.FALSE.,ALLOW6=.FALSE.,PRECON=.FALSE.)
      EQUIVALENCE (FNAME,LINE)
! Do not allow units 0,5 (stderr and stdin) and maybe also 6 (stdout);
! maybe one might need to change this for special platforms / purposes
      IF (DUMMY.OR.((IUNIT.EQ.6).AND.(.NOT.ALLOW6)).OR.(IUNIT.EQ.5) &
     &                                          .OR.(IUNIT.LE.0)) RETURN
      INQUIRE(UNIT=IUNIT,EXIST=FOUND)
      IF (.NOT.FOUND) RETURN
      IF (.NOT.PRECON) THEN
         INQUIRE(UNIT=IUNIT,NAME=FNAME)
! red alert: some problem with INQUIRE occurred ...
         IF (FNAME(1:1).EQ.' ') RETURN
         DO 100 I=1,255
            IF (FNAME(I:I).NE.' ') LENGTH=I
  100    CONTINUE
      END IF
      INQUIRE(UNIT=IUNIT,FORM=DFORM)
      CALL UPPER(DFORM)
      INQUIRE(UNIT=IUNIT,ACCESS=ACC)
      CALL UPPER(ACC)
      RECL=0
      IF (ACC(1:6).EQ.'DIRECT') INQUIRE(UNIT=IUNIT,RECL=LRECL)
      IF (.NOT.PRECON) THEN
         IF (ACC(1:6).EQ.'DIRECT') THEN
            CLOSE(IUNIT)
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH), &
     &                      ACCESS='DIRECT',RECL=LRECL)
         ELSE
            FLUSH(IUNIT)
         ENDIF
      ELSE
         IF (ACC(1:6).EQ.'DIRECT') THEN
            CLOSE(IUNIT)
            OPEN(UNIT=IUNIT,ACCESS='DIRECT',RECL=LRECL)
         ELSE
            FLUSH(IUNIT)
         ENDIF
      END IF
  400 RETURN
      END
#endif

      SUBROUTINE REOPEN(IUNIT)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! This routine has a similar function like WFORCE. The only difference
! is that the 'record pointer' will not be positioned to the end of the
! file but to the beginning of the file (REOPEN acts like WFORCE with
! some REWIND immediately after it ---> can also replace REWIND ...).
      LOGICAL       DUMMY,ALLOW6,PRECON,FOUND
      CHARACTER*255 FNAME
      CHARACTER*16  DFORM,ACC
      INTEGER       IUNIT,LENGTH,I,LRECL
! Here you can customize some things specific to your system/own needs
! DUMMY: act as dummy routine, ALLOW6: one may also close and reopen
! unit 6 (usually stdout ...), PRECON: 'preconnected units' are used
      PARAMETER(DUMMY=.FALSE.,ALLOW6=.FALSE.,PRECON=.FALSE.)
! Do not allow units 0,5 (stderr and stdin) and maybe also 6 (stdout);
! maybe one might need to change this for special platforms / purposes
      IF (DUMMY.AND.(IUNIT.NE.0).AND.(IUNIT.NE.5).AND. &
     &         (((IUNIT.EQ.6).AND.ALLOW6).OR.(IUNIT.NE.6))) REWIND IUNIT
      IF (DUMMY.OR.((IUNIT.EQ.6).AND.(.NOT.ALLOW6)).OR.(IUNIT.EQ.5) &
     &                                          .OR.(IUNIT.LE.0)) RETURN
      INQUIRE(UNIT=IUNIT,EXIST=FOUND)
      IF (.NOT.FOUND) RETURN
      IF (.NOT.PRECON) THEN
         INQUIRE(UNIT=IUNIT,NAME=FNAME)
! red alert: some problem with INQUIRE occurred ...
         IF (FNAME(1:1).EQ.' ') RETURN
         DO 100 I=1,255
            IF (FNAME(I:I).NE.' ') LENGTH=I
  100    CONTINUE
      END IF
      INQUIRE(UNIT=IUNIT,FORM=DFORM)
      CALL UPPER(DFORM)
      INQUIRE(UNIT=IUNIT,ACCESS=ACC)
      CALL UPPER(ACC)
      RECL=0
      IF (ACC(1:6).EQ.'DIRECT') INQUIRE(UNIT=IUNIT,RECL=LRECL)
      CLOSE(IUNIT)
      IF (.NOT.PRECON) THEN
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH), &
     &                      ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH),FORM=DFORM)
         ENDIF
      ELSE
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FORM=DFORM)
         ENDIF
      END IF
      RETURN
      END


      SUBROUTINE CLEAN(IUNIT)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! This routine has a similar function like ERASE, but after removing
! the file it will be reopened (as now empty file ...)
      LOGICAL       DUMMY,ALLOW6,PRECON,FOUND
      CHARACTER*255 FNAME
      CHARACTER*16  DFORM,ACC
      INTEGER       IUNIT,LENGTH,I,LRECL
! Here you can customize some things specific to your system/own needs
! DUMMY: act as dummy routine, ALLOW6: one may also close and reopen
! unit 6 (usually stdout ...), PRECON: 'preconnected units' are used
      PARAMETER(DUMMY=.FALSE.,ALLOW6=.FALSE.,PRECON=.FALSE.)
! Do not allow units 0,5 (stderr and stdin) and maybe also 6 (stdout);
! maybe one might need to change this for special platforms / purposes
      INQUIRE(UNIT=IUNIT,ACCESS=ACC)
      CALL UPPER(ACC)

      IF (DUMMY.AND.(IUNIT.NE.0).AND.(IUNIT.NE.5).AND. &
     &         ACC(1:6).NE.'DIRECT'.AND. &
     &         (((IUNIT.EQ.6).AND.ALLOW6).OR.(IUNIT.NE.6))) REWIND IUNIT
      IF (DUMMY.OR.((IUNIT.EQ.6).AND.(.NOT.ALLOW6)).OR.(IUNIT.EQ.5) &
     &                                          .OR.(IUNIT.LE.0)) RETURN
      INQUIRE(UNIT=IUNIT,EXIST=FOUND)
      IF (.NOT.FOUND) RETURN
      IF (.NOT.PRECON) THEN
         INQUIRE(UNIT=IUNIT,NAME=FNAME)
! red alert: some problem with INQUIRE occurred ...
         IF (FNAME(1:1).EQ.' ') RETURN
         DO 100 I=1,255
            IF (FNAME(I:I).NE.' ') LENGTH=I
  100    CONTINUE
      END IF
      INQUIRE(UNIT=IUNIT,FORM=DFORM)
      CALL UPPER(DFORM)
      INQUIRE(UNIT=IUNIT,ACCESS=ACC)
      CALL UPPER(ACC)
      IF (ACC(1:6).EQ.'DIRECT') INQUIRE(UNIT=IUNIT,RECL=LRECL)
      CLOSE(IUNIT,STATUS='DELETE')
      IF (.NOT.PRECON) THEN
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH), &
     &                      ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FILE=FNAME(1:LENGTH),FORM=DFORM)
         ENDIF
      ELSE
         IF (ACC(1:6).EQ.'DIRECT') THEN
            OPEN(UNIT=IUNIT,ACCESS='DIRECT',RECL=LRECL)
         ELSE
            OPEN(UNIT=IUNIT,FORM=DFORM)
         ENDIF
      END IF
      RETURN
      END


      SUBROUTINE APPEND(IUNIT)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! Go to the end of a file so that following data will be appended to
! that file rather than destroying the old contents ... :
      CHARACTER*1 LINE
      LOGICAL     FOUND
      INTEGER     IUNIT
! Units 0,5,6 are usually reserved for stderr,stdin,stdout ...
! If your system uses other standard I/O-units change it ... !
      IF ((IUNIT.LE.0).OR.(IUNIT.EQ.5).OR.(IUNIT.EQ.6)) RETURN
      INQUIRE(UNIT=IUNIT,EXIST=FOUND)
      IF (.NOT.FOUND) RETURN
  100 CONTINUE
      READ(IUNIT,'(A)',ERR=200,END=200) LINE
      GOTO 100
  200 CONTINUE
      BACKSPACE IUNIT
      RETURN
      END


      SUBROUTINE ERASE(FILNAM,IERR)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! Remove a file ... :
      CHARACTER*(*) FILNAM
      INTEGER       IERR,NXTFRU,IUNIT
      EXTERNAL      NXTFRU
      IUNIT=NXTFRU()
      IF (IUNIT.LT.0) THEN
         IERR=3
         RETURN
      END IF
      IERR=0
      OPEN(IUNIT,FILE=FILNAM,ERR=200)
      CLOSE(IUNIT,STATUS='DELETE',ERR=300)
      RETURN
  200 IERR=1
      RETURN
  300 IERR=2
      RETURN
      END


      INTEGER FUNCTION NXTFRU()
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! Find the next free unit number ...
      LOGICAL OCCUP
      INTEGER I
      NXTFRU=-1
! Usually the standard FORTRAN range for unit numbers is 0...99;
! on some systems also unit numbers beyond 99 might be allowed:
! if this is the case and you want to make use of it change it ...
      DO 100 I=0,99
! Units 0,5,6 are usually reserved for stderr,stdin,stdout ...
! If your system uses other standard I/O-units change it ... !
         IF ((I.LE.0).OR.(I.EQ.5).OR.(I.EQ.6)) GOTO 100
         INQUIRE(UNIT=I,OPENED=OCCUP)
         IF (.NOT.OCCUP) THEN
            NXTFRU=I
            RETURN
         END IF
  100 CONTINUE
      RETURN
      END


      SUBROUTINE RDLINE(FILNAM,LINE,STRING,IERR)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! Extract line # LINE from file FILNAM and put the result into STRING
      CHARACTER*(*) STRING,FILNAM
      CHARACTER*1   DUMMY
      INTEGER       LINE,IERR,NXTFRU,IUNIT,I
      EXTERNAL NXTFRU
      STRING=' '
      IUNIT=NXTFRU()
      IF (IUNIT.LT.0) THEN
         IERR=3
         RETURN
      END IF
      IERR=0
      IF (LINE.LT.1) RETURN
      OPEN(IUNIT,FILE=FILNAM,ERR=200)
      DO 100 I=1,LINE-1
         READ(IUNIT,'(A)',ERR=300,END=300) DUMMY
  100 CONTINUE
      READ(IUNIT,'(A)',ERR=300,END=300) STRING
      CLOSE(IUNIT,ERR=200)
      RETURN
  200 IERR=1
      RETURN
  300 IERR=2
      RETURN
      END


      SUBROUTINE RDPOS(FILNAM,IU,STRING,IERR)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
! Read some file until STRING has been found ... -- warning:
! the maximum number of characters recognized here is 255!
      CHARACTER*(*) FILNAM,STRING
      CHARACTER*255 LINEIN,WORK
      INTEGER       IU,IERR,LS,LI,LENGTH,INDEX
      LOGICAL       LOPEN
      EXTERNAL      LENGTH
      IERR=0
      INQUIRE(UNIT=IU,OPENED=LOPEN)
      IF (.NOT.LOPEN) OPEN(IU,FILE=FILNAM,STATUS='UNKNOWN',ERR=100)
      REWIND IU
      WORK=STRING
! Ignore any leading blanks!
      CALL STRIP(WORK,LS,'L')
   10 CONTINUE
      READ(IU,'(A)',ERR=200,END=200) LINEIN
      LI=LENGTH(LINEIN)
! If line is shorter than search string forget it ...
      IF (LI.GE.LS) THEN
! Bingo!
         IF (INDEX(LINEIN,WORK(1:LS)).GT.0) RETURN
      ENDIF
      GOTO 10
      GOTO 200
! OPEN error ...
  100 IERR=1
      CLOSE(IU,ERR=300)
      RETURN
! READ error (or nothing found) ...
  200 IERR=2
      CLOSE(IU,ERR=300)
  300 RETURN
      END
