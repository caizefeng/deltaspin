#include "symbol.inc"
      MODULE ini
      USE prec
!**********************************************************************
!
!  this module implements a small timer utility to time
!  subroutine
!  (see START_TIMING)
!**********************************************************************

      ! this allows to set a maxmimum of 10 internested timers
      ! that should suffice for VASP
      INTEGER, PARAMETER, PRIVATE :: maxtimer=10

      INTEGER,SAVE :: used_timers=0
      CHARACTER (LEN=6), PRIVATE  :: timer_tag(maxtimer)
      REAL(q)      :: timer_vpu(maxtimer),timer_cpu(maxtimer)

      ! this allows to set a maxmimum of registered allocates
      INTEGER, PARAMETER, PRIVATE :: maxalloc=20
      INTEGER,SAVE                :: used_allocs=0
      CHARACTER (LEN=10), PRIVATE :: alloc_tag(maxalloc)
      REAL(q),PRIVATE             :: alloc_event(maxalloc)=0
      REAL(q),PRIVATE             :: alloc_total=0

      INTEGER, PRIVATE            :: used_allocs_hi
      REAL(q), PRIVATE            :: alloc_total_hi=0
      REAL(q), PRIVATE            :: alloc_event_hi(maxalloc)
      CHARACTER (LEN=10), PRIVATE :: alloc_tag_hi(maxalloc)

      INTEGER, PRIVATE ::  MINPGF,MAJPGF,ISWPS,IOOPS,IVCSW
      REAL(q), PRIVATE ::  UTIME,STIME,ETIME,RSIZM,AVSIZ,DAYTIM


      CONTAINS

!***********************************************************************
!
! timing routines
! START_TIMING(TAG) 
! registers a new timing routine with a specific name and 
! initialises the timer
!
!***********************************************************************

      SUBROUTINE START_TIMING(TAG)
      USE c2f_interface, ONLY : VTIME
      IMPLICIT NONE
      INTEGER ENTRY
      CHARACTER (LEN=*) :: TAG
      REAL(q) TV,TC

      CALL SEARCH_TIMING(TAG,ENTRY)
      IF (ENTRY==0) RETURN
    
      CALL VTIME(TV,TC)
      timer_vpu(ENTRY)=TV
      timer_cpu(ENTRY)=TC
      timer_tag(ENTRY)=TAG 

      END SUBROUTINE


      SUBROUTINE STOP_TIMING(TAG,IU,NAME,XMLTAG)
      USE vaspxml
      USE c2f_interface, ONLY : VTIME
      IMPLICIT NONE
      INTEGER ENTRY
      CHARACTER (LEN=*) :: TAG
      INTEGER :: IU
      CHARACTER (LEN=*),OPTIONAL :: XMLTAG
      CHARACTER (LEN=*),OPTIONAL :: NAME
      REAL(q) TV,TC

      CALL SEARCH_TIMING(TAG,ENTRY)
      IF (ENTRY==0) RETURN
    
      CALL VTIME(TV,TC)
      IF (PRESENT(NAME)) THEN
        IF (IU>=0) WRITE(IU,100) NAME,TV-timer_vpu(ENTRY),TC-timer_cpu(ENTRY)
      ELSE
        IF (IU>=0) WRITE(IU,100) TAG,TV-timer_vpu(ENTRY),TC-timer_cpu(ENTRY)
      ENDIF
      IF (PRESENT(XMLTAG)) &
           CALL XML_TIMING(TV-timer_vpu(ENTRY),TC-timer_cpu(ENTRY),name=XMLTAG)  


100   FORMAT(2X,A8,':  cpu time',F10.4,': real time',F10.4)

      
      timer_vpu(ENTRY)=TV
      timer_cpu(ENTRY)=TC
      timer_tag(ENTRY)=TAG 
      
      END SUBROUTINE

      SUBROUTINE SEPERATOR_TIMING(IU)

      IF (IU>0) WRITE(IU,100)
100   FORMAT(2X,'  --------------------------------------------')
      END SUBROUTINE SEPERATOR_TIMING


      SUBROUTINE SEARCH_TIMING(TAG,ENTRY)
      IMPLICIT NONE
      INTEGER ENTRY
      CHARACTER (LEN=*) :: TAG
      
      ! search for entry
      DO ENTRY=1,used_timers
         IF (timer_tag(ENTRY)==TAG) THEN
            RETURN
         ENDIF
      END DO

      IF (ENTRY>maxtimer) THEN
         ! no more entry available
         ENTRY=0
         WRITE(0,*) 'internal ERROR in SEARCH_TIMING: no more timing slot available'
      ELSE
         used_timers=used_timers+1
      ENDIF

      
      END SUBROUTINE SEARCH_TIMING

!***********************************************************************
!
! the following routines can be used to keep track of allocate
! and deallocate commands
! registered allocate calles also supply a tag
!
!***********************************************************************

      SUBROUTINE REGISTER_ALLOCATE(NALLOC, TAG)
      IMPLICIT NONE
      REAL(q) NALLOC
      CHARACTER (LEN=*), OPTIONAL :: TAG
      INTEGER ENTRY

      IF (PRESENT(TAG)) THEN
         CALL SEARCH_ALLOC(TAG, ENTRY)
         alloc_tag(ENTRY)  =TAG
         alloc_event(ENTRY)=alloc_event(ENTRY)+AINT(NALLOC/1000)
      END IF

      alloc_total=alloc_total+AINT(NALLOC/1000)

      IF (alloc_total>alloc_total_hi) THEN
         used_allocs_hi=used_allocs
         alloc_total_hi=alloc_total
         alloc_event_hi(1:used_allocs)=alloc_event(1:used_allocs)
         alloc_tag_hi(1:used_allocs)=alloc_tag(1:used_allocs)
      ENDIF

      END SUBROUTINE

      SUBROUTINE DEREGISTER_ALLOCATE(NALLOC, TAG)
      IMPLICIT NONE
      REAL(q) NALLOC
      CHARACTER (LEN=*), OPTIONAL :: TAG
      INTEGER ENTRY

      IF (PRESENT(TAG)) THEN
         CALL SEARCH_ALLOC(TAG, ENTRY)
         alloc_event(ENTRY)=alloc_event(ENTRY)-AINT(NALLOC/1000)
      END IF

      alloc_total=alloc_total-AINT(NALLOC/1000)

      END SUBROUTINE

      FUNCTION QUERRY_ALLOCATE()
      IMPLICIT NONE
      INTEGER QUERRY_ALLOCATE

      QUERRY_ALLOCATE=alloc_total
      END FUNCTION QUERRY_ALLOCATE


      SUBROUTINE DUMP_ALLOCATE(IU)
      IMPLICIT NONE
      INTEGER IU
      INTEGER ENTRY

      IF (IU>=0) THEN
      WRITE(IU,'(/1X,A,F10.0,A/A/)') 'total amount of memory used by VASP MPI-rank0',alloc_total,' kBytes', &
                                '======================================================================='

      DO ENTRY=1,used_allocs
         WRITE(IU,'(3X,A,A,F10.0,A)') alloc_tag(ENTRY),':  ',alloc_event(ENTRY),' kBytes'
      ENDDO
      WRITE(IU,*)
      ENDIF
      END SUBROUTINE DUMP_ALLOCATE


      SUBROUTINE DUMP_ALLOCATE_HI(IU)
      IMPLICIT NONE
      INTEGER IU
      INTEGER ENTRY

      IF (IU>=0) THEN
      WRITE(IU,'(/1X,A,F10.0,A/A/)') 'maximum amount of memory used by VASP MPI-rank0',alloc_total_hi,' kBytes', &
                                '======================================================================='

      DO ENTRY=1,used_allocs_hi
         WRITE(IU,'(3X,A,A,F10.0,A)') alloc_tag_hi(ENTRY),':  ',alloc_event_hi(ENTRY),' kBytes'
      ENDDO
      WRITE(IU,*)
      ENDIF
      END SUBROUTINE DUMP_ALLOCATE_HI


      SUBROUTINE DUMP_ALLOCATE_TAG(IU,TAG)
      IMPLICIT NONE
      INTEGER IU
      INTEGER ENTRY
      CHARACTER (LEN=*) :: TAG

      IF (IU>=0) THEN
      WRITE(IU,'(/1X,A,A,F10.0,A/A/)') 'memory high mark on MPI-rank0 inside ',TAG, alloc_total,' kBytes', &
                                '========================================================================'

      DO ENTRY=1,used_allocs
         WRITE(IU,'(3X,A,A,F10.0,A)') alloc_tag(ENTRY),':  ',alloc_event(ENTRY),' kBytes'
      ENDDO
      WRITE(IU,*)
      ENDIF
      END SUBROUTINE DUMP_ALLOCATE_TAG


      SUBROUTINE SEARCH_ALLOC(TAG,ENTRY)
      IMPLICIT NONE
      INTEGER ENTRY
      CHARACTER (LEN=*) :: TAG
      
      ! search for entry
      DO ENTRY=1,used_allocs
         IF (alloc_tag(ENTRY)==TAG) THEN
            RETURN
         ENDIF
      END DO

      IF (ENTRY>maxalloc) THEN
         ! no more entry available
         ENTRY=0
         WRITE(0,*) 'internal ERROR in SEARCH_ALLOC: no more registered allocation slots available'
      ELSE
         used_allocs=used_allocs+1
      ENDIF

      
      END SUBROUTINE SEARCH_ALLOC

      FUNCTION SEARCH_ALLOC_MEMORY(TAG)
      IMPLICIT NONE
      REAL(q)  SEARCH_ALLOC_MEMORY
      CHARACTER (LEN=*) :: TAG
      INTEGER ENTRY
      
      ! search for entry
      DO ENTRY=1,used_allocs
         IF (alloc_tag(ENTRY)==TAG) THEN
            SEARCH_ALLOC_MEMORY=alloc_event(ENTRY)
            RETURN
         ENDIF
      END DO

      SEARCH_ALLOC_MEMORY=0
      END FUNCTION SEARCH_ALLOC_MEMORY


!***********************************************************************
!
! dump some information on paging memory etc.
!
!***********************************************************************

      SUBROUTINE INIT_FINAL_TIMING()
      USE c2f_interface, ONLY : TIMING
      INTEGER IERR

      CALL TIMING(0,UTIME,STIME,ETIME,MINPGF,MAJPGF, &
     &            RSIZM,AVSIZ,ISWPS,IOOPS,IVCSW,IERR)
      IF (IERR/=0) ETIME=0._q

      END SUBROUTINE INIT_FINAL_TIMING

      SUBROUTINE DUMP_FINAL_TIMING(TIU6)
      USE c2f_interface, ONLY : TIMING
      INTEGER TIU6
      ! local
      INTEGER IERR
      INTEGER NODE_ME, IONODE

      CALL TIMING(0,UTIME,STIME,DAYTIM,MINPGF,MAJPGF, &
     &            RSIZM,AVSIZ,ISWPS,IOOPS,IVCSW,IERR)

      IF (TIU6>=0) THEN

      ETIME=DAYTIM-ETIME

      TOTTIM=UTIME+STIME
      WRITE(TIU6,*) ' '
      WRITE(TIU6,*) ' '
      WRITE(TIU6,'(A)') &
     &   ' General timing and accounting informations for this job:'
      WRITE(TIU6,'(A)') &
     &   ' ========================================================'
      WRITE(TIU6,*) ' '
      WRITE(TIU6,'(17X,A,F12.3)') ' Total CPU time used (sec): ',TOTTIM
      WRITE(TIU6,'(17X,A,F12.3)') '           User time (sec): ',UTIME
      WRITE(TIU6,'(17X,A,F12.3)') '         System time (sec): ',STIME
      WRITE(TIU6,'(17X,A,F12.3)') '        Elapsed time (sec): ',ETIME
      WRITE(TIU6,*) ' '
      WRITE(TIU6,'(17X,A,F12.0)') '  Maximum memory used (kb): ',RSIZM
      WRITE(TIU6,'(17X,A,F12.0)') '  Average memory used (kb): ',AVSIZ
      WRITE(TIU6,*) ' '
      WRITE(TIU6,'(17X,A,I12)')   '         Minor page faults: ',MINPGF
      WRITE(TIU6,'(17X,A,I12)')   '         Major page faults: ',MAJPGF
      WRITE(TIU6,'(17X,A,I12)')   'Voluntary context switches: ',IVCSW
      ENDIF

      END SUBROUTINE

      END MODULE

!**************** SUBROUTINE SPLCOF, SPLCOF_N0 *************************
! RCS:  $Id: ini.F,v 1.3 2002/08/14 13:59:39 kresse Exp $
!
!  Subroutine for calculating spline-coefficients
!  using the routines of the book 'numerical  recipes'
!  on input P(1,N) must contain x-values
!           P(2,N) must contain function-values
!  YP is the first derivatives at the first point
!  if >= 10^30 natural boundary-contitions (y''=0) are used
!
!  for point N always natural boundary-conditions are used in
!  SPLCOF, whereas SPLCOF_N0 assume 0 derivative at N
!  SPLCOF_NDER allows to specify a boundary condition
!  at both end points
!
!***********************************************************************

      SUBROUTINE SPLCOF(P,N,NDIM,Y1P)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(NDIM,5)
!
!     determination of spline coefficients
!     ------------------------------------
!     f = ((d*dx+c)*dx+b)*dx+a
!         between adjacent x - values
!
!     result
!     P-ARRAY
!     P(I,1) = X(I)
!     P(I,2) = A(I) = F(I)
!     P(I,3) = B(I)
!     P(I,4) = C(I)
!     P(I,5) = D(I)
!
      IF (Y1P> .99E30_q) THEN
        P(1,4)=0.0_q
        P(1,3)=0.0_q
      ELSE
        P(1,4)=-.5_q
        P(1,3)=(3._q/(P(2,1)-P(1,1)))*((P(2,2)-P(1,2))/ &
     &             (P(2,1)-P(1,1))-Y1P)
      ENDIF

      DO 20 I=2,N-1
        S=(P(I,1)-P(I-1,1))/(P(I+1,1)-P(I-1,1))
        R=S*P(I-1,4)+2._q
        P(I,4)=(S-1._q)/R
        P(I,3)=(6*((P(I+1,2)-P(I,2))/(P(I+1,1)-P(I,1))- &
     &          (P(I,2)-P(I-1,2))/(P(I,1)-P(I-1,1)))/ &
     &          (P(I+1,1)-P(I-1,1))-S*P(I-1,3))/R
   20 ENDDO

      P(N,4)=0.0_q
      P(N,3)=0.0_q
!
      DO 30 I=N-1,1,-1
        P(I,4)=P(I,4)*P(I+1,4)+P(I,3)
  30  ENDDO
!
      DO 50 I=1,N-1
        S= P(I+1,1)-P(I,1)
        R=(P(I+1,4)-P(I,4))/6
        P(I,5)=R/S
        P(I,4)=P(I,4)/2.0_q
        P(I,3)=(P(I+1,2)-P(I,2))/S-(P(I,4)+R)*S
   50 ENDDO
      RETURN
      END SUBROUTINE



      SUBROUTINE SPLCOF_N0(P,N,NDIM,Y1P)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(NDIM,5)
!
!     determination of spline coefficients
!     ------------------------------------
!     f = ((d*dx+c)*dx+b)*dx+a
!         between adjacent x - values
!
!     result
!     P-ARRAY
!     P(I,1) = X(I)
!     P(I,2) = A(I) = F(I)
!     P(I,3) = B(I)
!     P(I,4) = C(I)
!     P(I,5) = D(I)
!
      IF (Y1P> .99E30_q) THEN
        P(1,4)=0.0_q
        P(1,3)=0.0_q
      ELSE
        P(1,4)=-.5_q
        P(1,3)=(3._q/(P(2,1)-P(1,1)))*((P(2,2)-P(1,2))/ &
     &             (P(2,1)-P(1,1))-Y1P)
      ENDIF

      DO 20 I=2,N-1
        S=(P(I,1)-P(I-1,1))/(P(I+1,1)-P(I-1,1))
        R=S*P(I-1,4)+2._q
        P(I,4)=(S-1._q)/R
        P(I,3)=(6*((P(I+1,2)-P(I,2))/(P(I+1,1)-P(I,1))- &
     &          (P(I,2)-P(I-1,2))/(P(I,1)-P(I-1,1)))/ &
     &          (P(I+1,1)-P(I-1,1))-S*P(I-1,3))/R
   20 ENDDO
      YNP=0
      IF (YNP> .99E30_q) THEN
        QN=0
        UN=0
      ELSE
        QN=0.5_q
        UN=(3._q/(P(N,1)-P(N-1,1)))*(YNP-(P(N,2)-P(N-1,2))/ &
     &             (P(N,1)-P(N-1,1)))
      ENDIF
      P(N,4)=(UN-QN*P(N-1,3))/(QN*P(N-1,4)+1.)
      P(N,3)=0  ! never used
!
      DO 30 I=N-1,1,-1
        P(I,4)=P(I,4)*P(I+1,4)+P(I,3)
  30  ENDDO
!
      DO 50 I=1,N-1
        S= P(I+1,1)-P(I,1)
        R=(P(I+1,4)-P(I,4))/6
        P(I,5)=R/S
        P(I,4)=P(I,4)/2.0_q
        P(I,3)=(P(I+1,2)-P(I,2))/S-(P(I,4)+R)*S
   50 ENDDO
      RETURN
      END SUBROUTINE


      SUBROUTINE SPLCOF_NDER(P,N,NDIM,Y1P,YNP)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(NDIM,5)
!
!     determination of spline coefficients
!     ------------------------------------
!     f = ((d*dx+c)*dx+b)*dx+a
!         between adjacent x - values
!
!     result
!     P-ARRAY
!     P(I,1) = X(I)
!     P(I,2) = A(I) = F(I)
!     P(I,3) = B(I)
!     P(I,4) = C(I)
!     P(I,5) = D(I)
!
      IF (Y1P> .99E30_q) THEN
        P(1,4)=0.0_q
        P(1,3)=0.0_q
      ELSE
        P(1,4)=-.5_q
        P(1,3)=(3._q/(P(2,1)-P(1,1)))*((P(2,2)-P(1,2))/ &
     &             (P(2,1)-P(1,1))-Y1P)
      ENDIF

      DO 20 I=2,N-1
        S=(P(I,1)-P(I-1,1))/(P(I+1,1)-P(I-1,1))
        R=S*P(I-1,4)+2._q
        P(I,4)=(S-1._q)/R
        P(I,3)=(6*((P(I+1,2)-P(I,2))/(P(I+1,1)-P(I,1))- &
     &          (P(I,2)-P(I-1,2))/(P(I,1)-P(I-1,1)))/ &
     &          (P(I+1,1)-P(I-1,1))-S*P(I-1,3))/R
   20 ENDDO
      IF (YNP> .99E30_q) THEN
        QN=0
        UN=0
      ELSE
        QN=0.5_q
        UN=(3._q/(P(N,1)-P(N-1,1)))*(YNP-(P(N,2)-P(N-1,2))/ &
     &             (P(N,1)-P(N-1,1)))
      ENDIF
      P(N,4)=(UN-QN*P(N-1,3))/(QN*P(N-1,4)+1.)
      P(N,3)=0  ! never used
!
      DO 30 I=N-1,1,-1
        P(I,4)=P(I,4)*P(I+1,4)+P(I,3)
  30  ENDDO
!
      DO 50 I=1,N-1
        S= P(I+1,1)-P(I,1)
        R=(P(I+1,4)-P(I,4))/6
        P(I,5)=R/S
        P(I,4)=P(I,4)/2.0_q
        P(I,3)=(P(I+1,2)-P(I,2))/S-(P(I,4)+R)*S
   50 ENDDO
      RETURN
      END SUBROUTINE

!
!  helper routine, which copies X and Y arrays to P
!  and than performes the fit on the array Y
!
      SUBROUTINE SPLCPY(X,Y,P,NAC,NDIM,Y1P)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(NDIM,5)
      DIMENSION X(NAC)
      DIMENSION Y(NAC)
      DO 100 N=1,NAC
        P(N,1)=X(N)
        P(N,2)=Y(N)
  100 CONTINUE
      CALL SPLCOF(P,NAC,NDIM,Y1P)
      RETURN
      END SUBROUTINE
!
!  helper routine, which evaluates the spline fit at a specific
!  position
!
      SUBROUTINE SPLVAL(X,F,FDER,P,NAC,NDIM)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION P(NDIM,5)
!  interval bisectioning
      I=1
      J=NAC
      IF (X   <P(I,1)) GO TO 60
      IF (X   <P(J,1)) GO TO 70
      K=J-1
      GOTO 90
   60 K=1
      GOTO 90
   70 K=(I+J)/2
      IF(I==K) GOTO 90
      IF (X   <P(K,1)) GO TO 80
      I=K
      GOTO 70
   80 J=K
      GOTO 70
!
   90 DX=X   -P(K,1)
      F   =((P(K,5)*DX+P(K,4))*DX+P(K,3))*DX+P(K,2)
      FDER=(3.0_q*P(K,5)*DX+2.0_q*P(K,4))*DX+P(K,3)
      END SUBROUTINE


!***********************************************************************
!
! system name date and time
!
!***********************************************************************

      SUBROUTINE MY_DATE_AND_TIME(IU6)
      USE prec
      USE vaspxml
      IMPLICIT NONE
      INTEGER IU6
      CHARACTER (8)  DATE
      CHARACTER (10) TIME
#ifndef HOST
#define HOST "unknown"
#endif

      CALL DATE_AND_TIME( DATE,TIME)
      IF (IU6>=0) &
      WRITE(IU6,"(' executed on ',A20,' date ',A4,'.',A2,'.',A2,'  ',A2,':',A2,':',A2 )") & 
           HOST,DATE(1:4),DATE(5:6),DATE(7:8),TIME(1:2),TIME(3:4),TIME(5:6)

      CALL XML_TAG_STRING("platform" , HOST)
      CALL XML_TAG_STRING("date" , DATE(1:4)//" "//DATE(5:6)//" "//DATE(7:8))
      CALL XML_TAG_STRING("time" , TIME(1:2)//":"//TIME(3:4)//":"//TIME(5:6))

      END SUBROUTINE


!***********************************************************************
!
! write out current memory requirements
!
!***********************************************************************

      SUBROUTINE MEMORY_CHECK(LOOP,STR)
      USE prec
      USE c2f_interface, ONLY : TIMING
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      CHARACTER (LEN=*) STR
      REAL(q) SUM

      CALL TIMING(0,UTIME,STIME,DAYTIM,MINPGF,MAJPGF, &
     &            RSIZM,AVSIZ,ISWPS,IOOPS,IVCSW,IERR)
      IF (IERR/=0) WRITE(*,*) 'WARNING main: call to TIMING failed.'
      IF (IERR/=0) WRITE(*,*) 'WARNING main: call to TIMING failed.'
      WRITE(*,*) ' '
      WRITE(*,*) ' '
      WRITE(*,'(A)') &
     &   ' General timing and accounting informations for this job:'
      WRITE(*,'(A)') &
     &   ' ========================================================'
      WRITE(*,*) ' '
      WRITE(*,'(17X,A,F12.0)') '  Maximum memory used (kb): ',RSIZM
      WRITE(*,'(17X,A,F12.0)') '  Average memory used (kb): ',AVSIZ
      WRITE(*,*) ' '
      WRITE(*,'(17X,A,I12)')   '         Minor page faults: ',MINPGF
      WRITE(*,'(17X,A,I12)')   '         Major page faults: ',MAJPGF
      WRITE(*,'(17X,A,I12)')   'Voluntary context switches: ',IVCSW

      END SUBROUTINE
