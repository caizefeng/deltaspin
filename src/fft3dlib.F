#include "symbol.inc"
#ifndef MINLOOP
#define MINLOOP 1
#endif

#undef  ORIGINAL_CODE_BASE
#define VECTORIZE

! RCS:  $Id: fft3dlib.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
      SUBROUTINE FFTC3N(DATA,ID1,N1,ID2,N2,ID3,N3,WRK,IFAC, &
     &                       TRIG1,TRIG2,TRIG3,ISIGN,IERR,MCACHE)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

!***********************************************************************
!                                                                      *
! High-performance 3-dimensional complex Fast Fourier Transformation   *
! using special subroutine CFFTML (1-dimensional FFT of a vector of    *
! vectors with parallel processing of all one-dimensional data sets).  *
! This routine returns an unnormed output vector for inverse FFT s!    *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,TSET
      DIMENSION DATA(2*ID1,ID2,ID3),WRK(4*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./

      IERR=0
!-----------------------------------------------------------------------
! Initialisation-branch for trigonometric tables and factor tables:
!-----------------------------------------------------------------------
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<N1).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routine CFTTAB.
         CALL CFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN

!-----------------------------------------------------------------------
!  version which tries to keep as much data in cache as possible
!  doing FFT plane by plane
!  for this version MCACHE is not important except for FFT-lenght > 100
!-----------------------------------------------------------------------
      IF (MCACHE==0) THEN
       DO I=1,N3
! Transformation along first dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1), &
     &              TRIG1(1),IFAC(1,1),2,2*ID1,N1,ID2,ISIGN,MCACHE)
! Transformation along second dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1), &
     &              TRIG2(1),IFAC(1,2),2*ID1,2,N2,ID1,ISIGN,MCACHE)
       ENDDO
! Transformation along third dimension:
       ID12=ID1*ID2
       DO I=1,N2
         CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1), &
     &              TRIG3(1),IFAC(1,3),2*ID12,2,N3,ID1,ISIGN,MCACHE)
       ENDDO
      ELSE
!-----------------------------------------------------------------------
!  version which tries to do as much data simultaneaously as possible
!  caching is done within CFFTML
!-----------------------------------------------------------------------
! Transformation along first dimension:
       CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1), &
     &              TRIG1(1),IFAC(1,1),2,2*ID1,N1,ID2*ID3,ISIGN,MCACHE)
! Transformation along second dimension: must be splitted
       DO I=1,N3
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1), &
                     TRIG2(1),IFAC(1,2),2*ID1,2,N2,ID1,ISIGN,MCACHE)
       ENDDO
! Transformation along third dimension:
      ID12=ID1*ID2
      CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1),IFAC(1,3), &
                     2*ID12,2,N3,ID12,ISIGN,MCACHE)
      ENDIF
      RETURN
      END

#if defined(ORIGINAL_CODE_BASE)

      SUBROUTINE CFFTML(AR,AI,WORK,TRIGS,IFAC,INCL,INCN,L,N,ISIGN,MCACHE)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! This subroutine performs C omplex 1-D F ast F ourier T ransforms on  *
! M ultiple data sets (doing the 1-D transforms in parallel). Routine  *
! CFFTML uses the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM.  *
!                                                                      *
!***********************************************************************

      DIMENSION AR((N-1)*INCN+(L-1)*INCL+1),AI((N-1)*INCN+(L-1)*INCL+1)
      DIMENSION WORK(4*L*N),TRIGS(2*L),IFAC(19)
      DATA IWOFF /0/, MAXOLD /0/
      SAVE IWOFF,MAXOLD
! no data return immediately
      IF (N.EQ.0) RETURN
! IFAC(1) contains the number of factors
      NFAC=IFAC(1)

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),MINLOOP))
      IF (MCACHE==0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      IF ((MAXDAT*L)/=MAXOLD) THEN
        IF (IWOFF/=0) IWOFF=2*MAXDAT*L
        IF (IWOFF>(2*MAXOLD)) IWOFF=0
        MAXOLD=MAXDAT*L
      ENDIF
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      LA=1
! Special case NFAC=1:
      IF (NFAC==1) THEN
! Use routine SPASSM, nothing will be swapped (SPASSM swaps internally):
         CALL SPASSM(AR(I0),AI(I0),WORK(1),WORK(1+N0*L), &
     &                                             INCL,INCN,N0,L,ISIGN)
! At this point the final transform result is stored in AR/AI, bye ...
         GOTO 10000
      END IF
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR/AI to WORK:
               CALL FPASSM(AR(I0),AI(I0),WORK(1+IWOFF), &
     &                     WORK(1+IWOFF+N0*L),TRIGS(1), &
     &                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
           ELSE IF (K==NFAC) THEN
! The last pass swaps WORK to AR/AI:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AI(I0),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR/AI to WORK:
               CALL IPASSM(AR(I0),AI(I0),WORK(1+IWOFF), &
     &                     WORK(1+IWOFF+N0*L),TRIGS(1), &
     &                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE IF (K==NFAC) THEN
! The last pass swaps WORK to AR/AI:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AI(I0),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
10000 CONTINUE
! At this point the final transform result is stored in AR/AI, bye ...
      RETURN
      END


      SUBROUTINE FFTC3V(DATC,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC, &
     &                                     TRIG1,TRIG2,TRIG3,ISIGN,IERR)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! High-performance 3-dimensional complex Fast Fourier Transformation   *
! using the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM. This   *
! routine uses complex storage mode for input and output, real storage *
! mode on work array hopefully speeding up the whole at least for long *
! transforms with many factors (hence many operations on work array).  *
!                                                                      *
! This routine is specially optimized for vector machines -- never use *
! this routine on "small workstations"!!!! (no efficient use of cache) *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,TSET
      DIMENSION DATC(2*ID1*ID2*ID3),WORK(4*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
! Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<N1).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routine CFTTAB.
         CALL CFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN
! Extract numbers of factors and preset some variables:
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3
      IWOFF=0
      LA=1
!
!     Perform the transform passes needed to transform third dimension,
!     one pass for each factor:
!
      INCL=2*ID1*ID2
      INCN=2
      L=N3
      N=ID1*ID2
      INCLO=1
      INCNO=ID3
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC3
            IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATC to WORK:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATC to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   50    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC3
            IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATC to WORK:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATC to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   60    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'third' dimension), one pass for each factor:
!
      INCL=ID1*ID3
      INCN=1
      L=N2
      N=ID1*ID3
      INCLO=1
      INCNO=ID2
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 70 K=1,NFAC2
            IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   70    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 80 K=1,NFAC2
            IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   80    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
!
!     Perform the transform passes needed to transform first dimension
!     (which is currently 'third' dimension), one pass for each factor:
!
      INCL=ID2*ID3
      INCN=1
      L=N1
      N=ID2*ID3
      INCLO=2
      INCNO=2*ID1
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 90 K=1,NFAC1
            IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATC and rearranges order of data
! to shift the role of the three dimensions cyclically:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1), &
     &              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
   90    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 100 K=1,NFAC1
            IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATC and rearranges order of data
! to shift the role of the three dimensions cyclically:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1), &
     &              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
  100    CONTINUE
      END IF
      RETURN
      END

#endif

      SUBROUTINE FFTCRN(DATA,ID1,N1,ID2,N2,ID3,N3,WRK,IFAC, &
     &                               TRIG1,TRIG2,TRIG3,ISIGN,IHERM,IERR, &
     &                               MCACHE)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! High-performance 3-dim. real <-> hermite Fast Fourier Transformation *
! using my library subroutine RHFFTM (1-dimensional FFT of a vector of *
! vectors with parallel processing of all one-dimensional data sets).  *
! The real <-> hermite pass is made along the first dimension!         *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,FACODD,TSET
      DIMENSION DATA(ID1,ID2,ID3),WRK(2*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR,FACODD
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
!-----------------------------------------------------------------------
! Initialisation-branch for trigonometric tables and factor tables:
!-----------------------------------------------------------------------
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<(N1+2)).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (FACODD(N1)) IERR=4
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routines CFTTAB/RFTTAB.
         CALL RFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN
      N1HP1=N1/2+1
!-----------------------------------------------------------------------
!  version which tries to keep as much data in cache as possible
!  doing FFT plane by plane
!  for this version MCACHE is not important except for FFT-lenght > 100
!-----------------------------------------------------------------------
! Here the real to hermite transformations ...:
     IF (MCACHE==0) THEN
      IF (IHERM>=0) THEN
         IF (ISIGN>0) ITRANS=1
         IF (ISIGN<0) ITRANS=-2
         DO I=1,N3
! Transformation along first dimension ('real to hermite pass'):
            CALL RHFFTM(DATA(1,1,I),WRK(1),TRIG1(1), &
                               IFAC(1,1),1,ID1,N1,ID2,ITRANS,MCACHE)
! Transformation along second dimension:
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1), &
                               IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,MCACHE)
         ENDDO
! Transformation along third dimension:
         DO I=1,N2
            CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1),TRIG3(1), &
                              IFAC(1,3),ID1*ID2,2,N3,N1HP1,ISIGN,MCACHE)
         ENDDO
! Here the hermite to real transformations ...:
      ELSE
         IF (ISIGN>0) ITRANS=2
         IF (ISIGN<0) ITRANS=-1
! Transformation along third dimension:
         DO I=1,N2
            CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1),TRIG3(1), &
                              IFAC(1,3),ID1*ID2,2,N3,N1HP1,ISIGN,MCACHE)
         ENDDO
         DO I=1,N3
! Transformation along second dimension:
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1), &
                              IFAC(1,2),ID1,2,N2,N1HP1,ISIGN, MCACHE)
! Transformation along first dimension ('hermite to real pass'):
            CALL RHFFTM(DATA(1,1,I),WRK(1),TRIG1(1), &
                              IFAC(1,1),1,ID1,N1,ID2,ITRANS,MCACHE)
         ENDDO
       END IF
      ELSE
!-----------------------------------------------------------------------
!  version which tries to do as much data simultaneaously as possible
!  caching is done within CFFTML
!-----------------------------------------------------------------------
! Here the real to hermite transformations ...:
      IF (IHERM>=0) THEN
         IF (ISIGN>0) ITRANS=1
         IF (ISIGN<0) ITRANS=-2
! Transformation along first dimension ('real to hermite pass'):
         CALL RHFFTM(DATA(1,1,1),WRK(1),TRIG1(1), &
     &                        IFAC(1,1),1,ID1,N1,ID2*ID3,ITRANS,MCACHE)
         DO I=1,N3
! Transformation along second dimension: must be splitted
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1), &
     &                        IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,MCACHE)
         ENDDO
! Transformation along third dimension:
         CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1), &
     &                        IFAC(1,3),ID1*ID2,2,N3,N1HP1*ID2,ISIGN,MCACHE)
! Here the hermite to real transformations ...:
      ELSE
         IF (ISIGN>0) ITRANS=2
         IF (ISIGN<0) ITRANS=-1
! Transformation along third dimension:
         CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1), &
     &                        IFAC(1,3),ID1*ID2,2,N3,N1HP1*ID2,ISIGN,MCACHE)
         DO I=1,N3
! Transformation along second dimension: must be splitted
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1), &
     &                        IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,MCACHE)
         ENDDO
! Transformation along first dimension ('hermite to real pass'):
         CALL RHFFTM(DATA(1,1,1),WRK(1),TRIG1(1), &
     &                        IFAC(1,1),1,ID1,N1,ID2*ID3,ITRANS,MCACHE)
      END IF
      END IF
      RETURN
      END

#if defined(ORIGINAL_CODE_BASE)

      SUBROUTINE FFTR3V(DATA,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC, &
     &                               TRIG1,TRIG2,TRIG3,ISIGN,IHERM,IERR)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! High-performance 3-dimensional real <--> first dimension hermitian   *
! Fast Fourier Transformation using the TEMPERTON/FULKER FFT-kernel    *
! routines FPASSM/IPASSM. This routine uses complex storage mode for   *
! hermitian input and output and real storage mode on the work array   *
! hopefully speeding up the whole at least for long transforms with    *
! many factors (hence many operations on work array).                  *
!                                                                      *
! This routine is specially optimized for vector machines -- never use *
! this routine on "small workstations"!!!! (no efficient use of cache) *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,FACODD,TSET
      DIMENSION DATA(ID1*ID2*ID3),WORK(2*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR,FACODD
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
! Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<(N1+2)).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (FACODD(N1)) IERR=4
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routine CFTTAB.
         CALL RFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN
! Extract numbers of factors and preset some variables:
      RSIGN=ISIGN
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3/2
      IWOFF=0
      LA=1

!***********************************************************************
!                                                                      *
!     Here comes the real --> complex hermite FFT ...                  *
!                                                                      *
!***********************************************************************
      IF (IHERM>=0) THEN
!
!     Perform the transform passes needed to transform first dimension,
!     one pass for each factor:
!
         INCL=2
         INCN=ID1
         L=N1/2
         N=ID2*ID3
         INCLO=ID2*ID3
         INCNO=1
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 50 K=1,NFAC1
               IF (K==1) THEN
! The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   50       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 60 K=1,NFAC1
               IF (K==1) THEN
! The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   60       CONTINUE
         END IF
         LA=1
!
! Now we have some complex data sets containing the requested result in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps between portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
!
         NEWOFF=2*ID123-IWOFF
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123), &
     &              TRIG1(1),N,INCLO,1,INCNO,N,L,RSIGN)
         IWOFF=NEWOFF
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'first' dimension), one pass for each factor:
!
         INCL=1
         INCN=ID2
         L=N2
         N=ID1*ID3/2
         INCLO=ID1*ID3/2
         INCNO=1
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 70 K=1,NFAC2
               IF ((K==1).AND.(NFAC2==1)) THEN
! The first (and only) pass swaps between different portions of WORK and
! rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC2>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   70       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 80 K=1,NFAC2
               IF ((K==1).AND.(NFAC2==1)) THEN
! The first (and only) pass swaps between different portions of WORK and
! rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC2>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   80       CONTINUE
         END IF
         LA=1
!
!     Perform the transform passes needed to transform third dimension
!     (which is currently 'first' dimension), one pass for each factor:
!
         INCL=1
         INCN=ID3
         L=N3
         N=ID1*ID2/2
         INCLO=ID1*ID2
         INCNO=2
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 90 K=1,NFAC3
               IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps from WORK to DATA and rearranges order
! of data to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                        DATA(1),DATA(2),TRIG3(1), &
     &                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC3) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
   90       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 100 K=1,NFAC3
! The first (and only) pass swaps from WORK to DATA and rearranges order
! of data to shift the role of the three dimensions cyclically:
               IF ((K==1).AND.(NFAC3==1)) THEN
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                        DATA(1),DATA(2),TRIG3(1), &
     &                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC3) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  100       CONTINUE
         END IF
!***********************************************************************
!                                                                      *
!     Here comes the complex hermite --> real FFT ...                  *
!                                                                      *
!***********************************************************************
      ELSE
!
!     Perform the transform passes needed to transform third dimension,
!     one pass for each factor:
!
         INCL=ID1*ID2
         INCN=2
         L=N3
         N=ID1*ID2/2
         INCLO=1
         INCNO=ID3
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 150 K=1,NFAC3
               IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATA to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
                 CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  150       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 160 K=1,NFAC3
               IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATA to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  160       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'third' dimension), one pass for each factor:
!
         INCL=ID1*ID3/2
         INCN=1
         L=N2
         N=ID1*ID3/2
         INCLO=1
         INCNO=ID2
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 170 K=1,NFAC2
               IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  170       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 180 K=1,NFAC2
               IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  180       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
!
!     Perform the transform passes needed to transform first dimension
!     (which is currently 'third' dimension), one pass for each factor:
!
         INCL=ID2*ID3
         INCN=1
         L=N1/2
         N=ID2*ID3
         INCLO=2
         INCNO=ID1
!
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we do on the real -> hermite pass):
!
         NEWOFF=2*ID123-IWOFF
         CALL RCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123), &
     &              TRIG1(1),N,N,1,1,N,L,RSIGN)
         IWOFF=NEWOFF
! And now the transform:
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 190 K=1,NFAC1
               IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  190       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 200 K=1,NFAC1
               IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  200       CONTINUE
         END IF
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
      END IF

      RETURN
      END



      SUBROUTINE RHFFTM(AR,WORK,TRIGS,IFAC,INCLR,INCN,LR,N,ISIGN, &
     &                  MCACHE)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! This subroutine performs R eal or H ermitian 1-D F ast F ourier      *
! T ransforms on M ultiple data sets (doing the 1-D transforms in      *
! parallel). Routine RHFFTM uses the TEMPERTON/FULKER FFT-kernel       *
! routines FPASSM/IPASSM and post-/preprocessing routines RCOMB/HCOMB. *
! Maximum performance on vector processors will be achieved for INCN=1 *
! or at least odd values for INCN and if possible odd values for INCL. *
! Before using CFFTM the first time one must initialise trigonometric  *
! tables and factor tables by calling routine RFTTAB. The output data  *
! will overwrite the input data, the output is   n o t   normalized!   *
! Arguments:                                                           *
!   AR: array containing the real (and imaginary) parts of the data    *
!   WORK: work array of length 2*LR*N                                  *
!   LR: is the length of the transformations (lr = 2**p * 3**q * 5**r) *
!   INCLR: addressing increment between elements of one data set       *
!   N: is the number of data sets to be transformed                    *
!   INCN: is the addressing increment between the starting elements of *
!         consecutive data sets                                        *
!   ISIGN: >=0: forward tranform, <0: inverse transform                *
!   TRIGS: array of length 2*LR containing all sine and cosine factors *
!   IFAC: array of lenght 19 containing the factors contained in LR    *
!                                                                      *
! INFORMATION: Routine RHFFTM is an equivalent to routine RFFTMLT from *
!              the CRAY SCIentific LIBrary (SCILIB)! But it should be  *
!              noted that RHFFTM supports some additional features not *
!              being supported by SCILIB routine RFFTMLT!              *
!                                                                      *
!***********************************************************************

      DIMENSION AR((N-1)*INCN+(LR+1)*INCLR+1)
      DIMENSION WORK(2*LR*N),TRIGS(2*LR),IFAC(19)
      DATA IWOFF /0/, MAXOLD /0/
      SAVE IWOFF,MAXOLD
! no data return immediately
      IF (N.EQ.0) RETURN
! IFAC(1) contains the number of factors
      NFAC=IFAC(1)
! We need a complex transform of length (LR/2) with some tricky
! re-interpretation of the originally real data as complex data:
      INCL=INCLR*2
      L=LR/2

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),MINLOOP))
      IF (MCACHE==0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      IF ((MAXDAT*L)/=MAXOLD) THEN
        IF (IWOFF/=0) IWOFF=2*MAXDAT*L
        IF (IWOFF>(2*MAXOLD)) IWOFF=0
        MAXOLD=MAXDAT*L
      ENDIF
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      LA=1
! We support two storages modes: 'real' and 'complex', 'complex' refers
! to the behaviour of SCILIB-routine RFFTMLT, 'real' goes beyond the
! capabilities of RFFTMLT ... ! Mode 'real' is used for ABS(ISIGN)>=10,
! 'complex' mode else ... :
      IF (ABS(ISIGN)<10) THEN
         INCHER=INCL
         IOFHER=INCLR
      ELSE
         INCHER=INCLR
         IOFHER=(L+1)*INCLR
      END IF
! We first treat the normal case (forward tranform for real data and
! inverse transform for the hermitian conjugate data set ...). The
! opposite case (forward transform for hermitian conjugate data and
! inverse transformation of the corresponding real data sets) shall
! be selected if ISIGN is larger equal 20 or smaller equal -20 (then
! - see above - 'real' storage mode is used) or if ISIGN is in the
! range 2 ... 9 or -9 ... -2 (then complex storage mode is used) ... :
      IF ((ABS(ISIGN)>=20).OR. &
     &            ((ABS(ISIGN)>1).AND.(ABS(ISIGN)<10))) GOTO 99999
! Note that SCILIB routine RFFTMLT does not support the second case,
! it is an other extension going beyond the capabilities of RFFTMLT!
! ---> SO USE ISIGN=1 OR -1 TO GET EXACTLY THE BEHAVIOUR OF RFFTMLT !!
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN>=0) THEN
         RSIGN=1._q
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR to WORK:
               CALL FPASSM(AR(I0),AR(I0+INCLR), &
     &                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
! Now we have some complex data sets containing the requested result in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0), &
     &            AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=-1._q
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we have done during the forward FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER), &
     &              WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K==NFAC) THEN
! The last pass swaps WORK to AR:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AR(I0+INCLR), &
     &                     TRIGS(1),N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
! At this point the final transform result is stored in AR, bye ...
      GOTO 10000
99999 CONTINUE
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN<0) THEN
         RSIGN=-1._q
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 150 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR to WORK:
               CALL IPASSM(AR(I0),AR(I0+INCLR), &
     &                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  150    CONTINUE
! Now we have some complex data sets containing the requested result in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0), &
     &              AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=1._q
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we have done during the inverse FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER), &
     &              WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
! 'Forward transform' using exp(+i*phase)-factors:
         DO 160 K=1,NFAC
            IF (K==NFAC) THEN
! The last pass swaps WORK to AR:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AR(I0+INCLR),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  160    CONTINUE
      END IF
10000 CONTINUE
! At this point the final transform result is stored in AR, bye ...
      RETURN
      END

#endif


      SUBROUTINE CFTTAB(L,IFAC,TRIGS)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Serves as an interface calling the routines FTFACT and FTRIGC to get *
! the factors of L and trigonometric tables for a Fourier Transform of *
! length L. For details see routines FTFACT and FTRIGC ... .           *
! INFORMATION: Routine CFTTAB is an equivalent to routine CFTFAX from  *
!              the CRAY SCIentific LIBrary (SCILIB)!                   *
!                                                                      *
!***********************************************************************

      DIMENSION IFAC(19),TRIGS(2*L)
!
      CALL FTFACT(L,IFAC)
      CALL FTRIGC(L,TRIGS)
      RETURN
      END


      SUBROUTINE RFTTAB(L,IFAC,TRIGS)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Serves as an interface calling the routines FTFACT and FTRIGR to get *
! the factors of L and trigonometric tables for a Fourier Transform of *
! length L. For details see routines FTFACT and FTRIGR ... .           *
! INFORMATION: Routine RFTTAB is an equivalent to routine FFTFAX from  *
!              the CRAY SCIentific LIBrary (SCILIB)!                   *
!                                                                      *
!***********************************************************************

      DIMENSION IFAC(19),TRIGS(2*L)
! Length of transformation (L) must be even!!!
      IF (MOD(ABS(L),2)/=0) THEN
         IFAC(1)=-99
         RETURN
      END IF
      LH=L/2
      CALL FTFACT(LH,IFAC)
      CALL FTRIGR(L,TRIGS)
      RETURN
      END


      SUBROUTINE FTFACT(L,IFAC)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Factorization routine which extracts all factors of 4/2, 3, 5 and 7  *
! contained in L (L must be larger than 1). Currently a maximum of 18  *
! factors is allowed. On output IFAC(1) contains the number of factors *
! (or -99 in case of error), IFAC(2)...IFAC(IFAC(1)+1) all factors.    *
!                                                                      *
!***********************************************************************

      DIMENSION IFAC(19)
! L must be larger than 1!
      IF (ABS(L)<=1) THEN
         IFAC(1)=-99
         RETURN
      ENDIF
! the sign of L does not matter at all
      NN=ABS(L)
! simple case of single factor which can be treated by SPASSM
      IF (NN<=8) THEN
         IFAC(1)=1
         IFAC(2)=NN
         RETURN
      ENDIF
      N2=0
      N3=0
      N5=0
      N7=0
! get the number of factors 2, 3, 5, and 7 contained in L
   10 IF (MOD(NN,2)/=0) GOTO 20
      N2=N2+1
      NN=NN/2
      IF (NN==1) GOTO 50
      GOTO 10
   20 IF (MOD(NN,3)/=0) GOTO 30
      N3=N3+1
      NN=NN/3
      IF (NN==1) GOTO 50
      GOTO 20
   30 IF (MOD(NN,5)/=0) GOTO 40
      N5=N5+1
      NN=NN/5
      IF (NN==1) GOTO 50
      GOTO 30
   40 IF (MOD(NN,7)/=0) GOTO 100
      N7=N7+1
      NN=NN/7
      IF (NN==1) GOTO 50
      GOTO 40
   50 CONTINUE
      N6=0
      N8=0
! now calculate the number of factors of 4 we can build out of the factors of 2
      N4=N2/2
! reset the number of factors of 2 accordingly
      N2=N2-2*N4
! factors of 6 or 8 are beneficial if there are not yet too many other large
! factors present and if at least one smaller factor (3 or 4) remains as first
! factor; use only any remaining factor of 2, never split factors of 4, i.e.,
! usually something like 3,3,4 is expected to be better than 2,3,6 or 6,6 since
! any large factor is usually not very efficient, the only benefit of factors 6
! or 8 is that we get rid of a single factor of 2 which is often less efficient
! because less work is done in the loops (worst loop overhead of all factors)
! if one single factor of 2 is left then try to merge it into a factor of 6
! but only if at least one further small factor is left as first factor ...
      IF ((MIN(N2,N3)>0).AND.(MAX(N4+1,N3)>1).AND.((N5+N7)<2).AND.((N3+N4)<4)) THEN
        N6=N6+1
        N2=N2-1
        N3=N3-1
      ENDIF
! if one single factor of 2 is left then try to merge it into a factor of 8
! but only if at least one further small factor is left as first factor ...
! mind: currently only one single factor of 8 may appear (only as last factor!)
      IF ((MIN(N2,N4)>0).AND.(MAX(N3+1,N4)>1).AND.((N5+N7)<2).AND.(N4<3).AND.((N3+N4)<4)) THEN
        N8=N8+1
        N2=N2-1
        N4=N4-1
      ENDIF
! set array IFAC, starting with the smallest, ending with the largest factor
      K=1
      DO ILOOP=1,N2
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=2
      ENDDO
      DO ILOOP=1,N4
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=4
      ENDDO
      DO ILOOP=1,N3
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=3
      ENDDO
      DO ILOOP=1,N6
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=6
      ENDDO
      DO ILOOP=1,N5
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=5
      ENDDO
      DO ILOOP=1,N7
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=7
      ENDDO
! mind: currently a (single) factor of 8 may only appear as the last factor
      DO ILOOP=1,N8
        K=K+1
        IF (K>19) GOTO 100
        IFAC(K)=8
      ENDDO
! IFAC(1) contains the number of factors
      IFAC(1)=K-1
      RETURN
! Error: other factors than 2,3,5, or 7 or more than 18 factors!
  100 IFAC(1)=-99
      RETURN
      END


      SUBROUTINE FTRIGC(L,TRIGS)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Sets up trigonometric tables for Fourier Transform of length L. The  *
! complex phase factors are stored in TRIGS, with alternating storage  *
! for real (cosine) parts and imaginary (sine) parts as in a complex   *
! FORTRAN array. TRIGS contains the factors phase_k=exp(+i*PI*(k/L)).  *
!                                                                      *
!***********************************************************************

      DIMENSION TRIGS(*)
      REAL(q) PI,DELTA,ANGLE
      PARAMETER(PI=3.1415926535897932384626433832795E0_q)
!
      DELTA=PI/DBLE(ABS(L))
      DO 10 I=1,2*ABS(L)-1,2
         ANGLE=DBLE(I-1)*DELTA
         TRIGS(I)=COS(ANGLE)
         TRIGS(I+1)=SIN(ANGLE)
   10 CONTINUE
      RETURN
      END


      SUBROUTINE FTRIGR(L,TRIGS)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Sets up trigonometric tables for Fourier Transform of length L. The  *
! complex phase factors are stored in TRIGS, with alternating storage  *
! for real (cosine) parts and imaginary (sine) parts as in a complex   *
! FORTRAN array. TRIGS contains the factors phase_k=exp(+i*PI*(k/L)).  *
!                                                                      *
!***********************************************************************

      DIMENSION TRIGS(*)
      REAL(q) PI,DELTA,ANGLE
      PARAMETER(PI=3.1415926535897932384626433832795E0_q)
!
      DELTA=PI/DBLE(ABS(L)/2)
      DO 10 I=1,ABS(L)-1,2
         ANGLE=DBLE(I-1)*DELTA
         TRIGS(I)=COS(ANGLE)
         TRIGS(I+1)=SIN(ANGLE)
   10 CONTINUE
      DELTA=PI/DBLE(ABS(L))
      DO 20 I=ABS(L)+1,ABS(L)+2*((ABS(L)+2)/4)-1,2
         ANGLE=DBLE(I-ABS(L)-1)*DELTA
         TRIGS(I)=COS(ANGLE)
         TRIGS(I+1)=SIN(ANGLE)
   20 CONTINUE
      DO 30 I=ABS(L)+2*((ABS(L)+2)/4)+1,2*ABS(L)
      TRIGS(I)=0._q
   30 CONTINUE
      RETURN
      END


      FUNCTION FACERR(L)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL FACERR
!***********************************************************************
!                                                                      *
! Serves as a possible factorization pre-checker for routine CFTTAB.   *
! Checks the factorization of L. Factors 2,3,4,5,6 and 7 are allowed.  *
! Warning! You may not have more than 18 factors in routine CFTTAB!    *
!                                                                      *
!***********************************************************************

      PARAMETER(NFAC=6)
! List of allowed factors:
      INTEGER FACTOR(NFAC)
      SAVE FACTOR
      DATA FACTOR /4,2,6,5,3,7/
!
      FACERR=.FALSE.
! L must be larger than 1!
      IF (L<2) GOTO 150
      N=L
      ICOUNT=0
! Check all factors ... :
      DO 100 I=1,NFAC
   10    IF (MOD(N,FACTOR(I))==0) THEN
            ICOUNT=ICOUNT+1
            N=N/FACTOR(I)
            GOTO 10
         END IF
! At the end we should arrive at factor '1' ...
         IF (N==1) GOTO 200
  100 CONTINUE
! ... otherwise there was some factor that is not allowed!
  150 FACERR=.TRUE.
      RETURN
  200 CONTINUE
! Too much factors!
      IF (ICOUNT>18) GOTO 150
      RETURN
      END


      FUNCTION FACODD(L)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      LOGICAL FACODD
!***********************************************************************
!                                                                      *
! Checks whether L is odd (result=TRUE) of even (result=FALSE)         *
!                                                                      *
!***********************************************************************

      FACODD=(MOD(L,2)==1)
      RETURN
      END


      SUBROUTINE FTFLOP(IFAC,IFLOP,NDATA)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Estimates the number of floating point operations (IFLOP) needed for *
! a FFT of length L according to the factorization from routine FTFACT *
!                                                                      *
!***********************************************************************

      DIMENSION IFAC(19)

      L=1
! get the transform length
      DO J=2,IFAC(1)+1
         L=L*IFAC(J)
      ENDDO
! Single factor transforms according to structure of SPASSM:
      IF (L<=8) THEN
         IF (L==2) IFLOP=4*NDATA
         IF (L==3) IFLOP=18*NDATA
         IF (L==4) IFLOP=16*NDATA
         IF (L==5) IFLOP=44*NDATA
         IF (L==6) IFLOP=46*NDATA
         IF (L==7) IFLOP=74*NDATA
         IF (L==8) IFLOP=76*NDATA
         RETURN
      END IF
      IFLOP=0
! Count operations according to the structure of FPASSM/IPASSM:
      LA=1
      DO 30 I=2,IFAC(1)+1
         IFACT=IFAC(I)
! lets hope that following operation counts are somehow correct ...
         M=L/IFACT
         DO 10 J=1,LA
            IF (IFACT==2) IFLOP=IFLOP+4*NDATA
            IF (IFACT==3) IFLOP=IFLOP+16*NDATA
            IF (IFACT==4) IFLOP=IFLOP+16*NDATA
            IF (IFACT==5) IFLOP=IFLOP+40*NDATA
            IF (IFACT==6) IFLOP=IFLOP+40*NDATA
            IF (IFACT==7) IFLOP=IFLOP+68*NDATA
            IF (IFACT==8) IFLOP=IFLOP+68*NDATA
   10    CONTINUE
         IF (LA/=M) THEN
            DO 21 J=LA+1,M,LA
             DO 20 JJ=1,LA
              IF (IFACT==2) IFLOP=IFLOP+10*NDATA
              IF (IFACT==3) IFLOP=IFLOP+28*NDATA
              IF (IFACT==4) IFLOP=IFLOP+34*NDATA
              IF (IFACT==5) IFLOP=IFLOP+64*NDATA
              IF (IFACT==6) IFLOP=IFLOP+70*NDATA
              IF (IFACT==7) IFLOP=IFLOP+104*NDATA
              IF (IFACT==8) IFLOP=IFLOP+110*NDATA
   20       CONTINUE
   21      CONTINUE
         ENDIF
         LA=LA*IFACT
   30 CONTINUE
! That is all ...
      RETURN
      END


      FUNCTION FTFTOT(NGPTAR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION NGPTAR(3),IFAC(19)
! estimate the number of floating point operations for a 3D-FFT
      FTFTOT=0
      CALL FTFACT(NGPTAR(1),IFAC)
      CALL FTFLOP(IFAC,IFLOP,NGPTAR(2)*NGPTAR(3))
      FTFTOT=FTFTOT+IFLOP
      CALL FTFACT(NGPTAR(2),IFAC)
      CALL FTFLOP(IFAC,IFLOP,NGPTAR(1)*NGPTAR(3))
      FTFTOT=FTFTOT+IFLOP
      CALL FTFACT(NGPTAR(3),IFAC)
      CALL FTFLOP(IFAC,IFLOP,NGPTAR(1)*NGPTAR(2))
      FTFTOT=FTFTOT+IFLOP
      RETURN
      END



#if defined(ORIGINAL_CODE_BASE)

      SUBROUTINE FPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! FPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) RETURN
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO==1) GOTO 10
      IF (IGO==2) GOTO 50
      IF (IGO==3) GOTO 90
      IF (IGO==4) GOTO 130
      IF (IGO==5) GOTO 170
      IF (IGO==6) GOTO 210
      IF (IGO==7) GOTO 250
!      GOTO (10,50,90,130,170,210,250),IGO
! end test change doris
!
! Coding for factor 2:
!
   10 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      DO 20 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 15 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IB+I)
      D(JA+J)=B(IA+I)+B(IB+I)
      C(JB+J)=A(IA+I)-A(IB+I)
      D(JB+J)=B(IA+I)-B(IB+I)
      I=I+INC3
      J=J+INC4
   15 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   20 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 40 K=LA1,M,LA
      KB=K+K-2
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      DO 30 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 25 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IB+I)
      D(JA+J)=B(IA+I)+B(IB+I)
      C(JB+J)=C1*(A(IA+I)-A(IB+I))-S1*(B(IA+I)-B(IB+I))
      D(JB+J)=S1*(A(IA+I)-A(IB+I))+C1*(B(IA+I)-B(IB+I))
      I=I+INC3
      J=J+INC4
   25 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   30 CONTINUE
      JBASE=JBASE+JUMP
   40 CONTINUE
      RETURN
!
! Coding for factor 3:
!
   50 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      DO 60 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=(A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC3
      J=J+INC4
   55 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   60 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 80 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      DO 70 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 65 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)= &
     &    C1*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))) &
     &   -S1*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      D(JB+J)= &
     &    S1*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))) &
     &   +C1*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      C(JC+J)= &
     &    C2*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))) &
     &   -S2*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      D(JC+J)= &
     &    S2*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))) &
     &   +C2*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      I=I+INC3
      J=J+INC4
   65 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   70 CONTINUE
      JBASE=JBASE+JUMP
   80 CONTINUE
      RETURN
!
! Coding for factor 4:
!
   90 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      DO 100 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 95 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      C(JC+J)=(A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      D(JC+J)=(B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I))
      C(JB+J)=(A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I))
      C(JD+J)=(A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I))
      D(JB+J)=(B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I))
      D(JD+J)=(B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I))
      I=I+INC3
      J=J+INC4
   95 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  100 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 120 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      DO 110 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 105 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      C(JC+J)= &
     &    C2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I))) &
     &   -S2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      D(JC+J)= &
     &    S2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I))) &
     &   +C2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      C(JB+J)= &
     &    C1*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I))) &
     &   -S1*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      D(JB+J)= &
     &    S1*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I))) &
     &   +C1*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      C(JD+J)= &
     &    C3*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I))) &
     &   -S3*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      D(JD+J)= &
     &    S3*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I))) &
     &   +C3*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  105 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  110 CONTINUE
      JBASE=JBASE+JUMP
  120 CONTINUE
      RETURN
!
! Coding for factor 5:
!
  130 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      DO 140 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  135 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  140 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 160 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      DO 150 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 145 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)= &
     &    C1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   -S1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JB+J)= &
     &    S1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +C1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JE+J)= &
     &    C4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   -S4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JE+J)= &
     &    S4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +C4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JC+J)= &
     &    C2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   -S2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JC+J)= &
     &    S2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +C2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      C(JD+J)= &
     &    C3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   -S3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JD+J)= &
     &    S3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +C3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      I=I+INC3
      J=J+INC4
  145 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  150 CONTINUE
      JBASE=JBASE+JUMP
  160 CONTINUE
      RETURN
!
! Coding for factor 6:
!
  170 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      DO 180 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 175 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(ID+I))+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      C(JD+J)=(A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))
      D(JA+J)=(B(IA+I)+B(ID+I))+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      D(JD+J)=(B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I)))
      C(JB+J)= &
     &     (A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      C(JF+J)= &
     &     (A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      D(JB+J)= &
     &     (B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      D(JF+J)= &
     &     (B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      C(JC+J)= &
     &     (A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      C(JE+J)= &
     &     (A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      D(JC+J)= &
     &     (B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      D(JE+J)= &
     &     (B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  175 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  180 CONTINUE
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 200 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      DO 190 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 185 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(ID+I))+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      D(JA+J)=(B(IA+I)+B(ID+I))+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      C(JD+J)= &
     &   C3*((A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))) &
     &  -S3*((B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))))
      D(JD+J)= &
     &   S3*((A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))) &
     &  +C3*((B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))))
      C(JB+J)= &
     &  C1*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & -S1*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      D(JB+J)= &
     &  S1*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +C1*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      C(JF+J)= &
     &  C5*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & -S5*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      D(JF+J)= &
     &  S5*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +C5*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      C(JC+J)= &
     &  C2*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & -S2*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      D(JC+J)= &
     &  S2*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +C2*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      C(JE+J)= &
     &  C4*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & -S4*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      D(JE+J)= &
     &  S4*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +C4*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  185 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  190 CONTINUE
      JBASE=JBASE+JUMP
  200 CONTINUE
      RETURN
!
! Coding for factor 7:
!
  210 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      IG=IF+IINK
      JG=JF+JINK
      DO 220 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I)) &
     &               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I)) &
     &               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &               -(SPIS2*(B(IB+I)-B(IG+I)) &
     &                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &               +(SPIS2*(B(IB+I)-B(IG+I)) &
     &                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &               +(SPIS2*(A(IB+I)-A(IG+I)) &
     &                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &               -(SPIS2*(A(IB+I)-A(IG+I)) &
     &                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &               -(SPIS3*(B(IB+I)-B(IG+I)) &
     &                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &               +(SPIS3*(B(IB+I)-B(IG+I)) &
     &                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &               +(SPIS3*(A(IB+I)-A(IG+I)) &
     &                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &               -(SPIS3*(A(IB+I)-A(IG+I)) &
     &                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &               -(SPIS1*(B(IB+I)-B(IG+I)) &
     &                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &               +(SPIS1*(B(IB+I)-B(IG+I)) &
     &                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &               +(SPIS1*(A(IB+I)-A(IG+I)) &
     &                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &               -(SPIS1*(A(IB+I)-A(IG+I)) &
     &                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  215 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  220 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 240 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      KG=KF+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      C6=TRIGS(KG+1)
      S6=TRIGS(KG+2)
      DO 230 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 225 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I)) &
     &               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I)) &
     &               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)= &
     &       C1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            -(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       -S1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            +(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JB+J)= &
     &        S1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            -(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       +C1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            +(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JG+J)= &
     &        C6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            +(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       -S6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            -(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JG+J)= &
     &        S6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            +(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       +C6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            -(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JC+J)= &
     &        C2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            -(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       -S2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            +(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JC+J)= &
     &        S2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            -(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +C2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            +(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JF+J)= &
     &        C5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            +(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       -S5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            -(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JF+J)= &
     &        S5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            +(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +C5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            -(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JD+J)= &
     &        C3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            -(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       -S3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            +(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JD+J)= &
     &        S3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            -(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +C3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            +(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      C(JE+J)= &
     &        C4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            +(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       -S4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            -(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JE+J)= &
     &        S4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            +(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +C4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            -(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  225 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  230 CONTINUE
      JBASE=JBASE+JUMP
  240 CONTINUE
      RETURN
!
! Coding for factor 8:
!
  250 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      IG=IF+IINK
      JG=JF+JINK
      IH=IG+IINK
      JH=JG+JINK
      DO 260 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 255 IJK=1,LOT
      C(JA+J)=((A(IA+I)+A(IE+I))+(A(IB+I)+A(IF+I))) &
     &       +((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
      D(JA+J)=((B(IA+I)+B(IE+I))+(B(IB+I)+B(IF+I))) &
     &       +((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
      C(JE+J)=((A(IA+I)+A(IE+I))-(A(IB+I)+A(IF+I))) &
     &       +((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
      D(JE+J)=((B(IA+I)+B(IE+I))-(B(IB+I)+B(IF+I))) &
     &       +((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
      C(JB+J)=    ((A(IA+I)-A(IE+I))-(B(IC+I)-B(IG+I))) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JB+J)=    ((B(IA+I)-B(IE+I))+(A(IC+I)-A(IG+I))) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JF+J)=    ((A(IA+I)-A(IE+I))-(B(IC+I)-B(IG+I))) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JF+J)=    ((B(IA+I)-B(IE+I))+(A(IC+I)-A(IG+I))) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JC+J)=((A(IA+I)+A(IE+I))-(A(IC+I)+A(IG+I))) &
     &       -((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JC+J)=((B(IA+I)+B(IE+I))-(B(IC+I)+B(IG+I))) &
     &       +((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JG+J)=((A(IA+I)+A(IE+I))-(A(IC+I)+A(IG+I))) &
     &       +((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JG+J)=((B(IA+I)+B(IE+I))-(B(IC+I)+B(IG+I))) &
     &       -((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JD+J)=    ((A(IA+I)-A(IE+I))+(B(IC+I)-B(IG+I))) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JD+J)=    ((B(IA+I)-B(IE+I))-(A(IC+I)-A(IG+I))) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JH+J)=    ((A(IA+I)-A(IE+I))+(B(IC+I)-B(IG+I))) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JH+J)=    ((B(IA+I)-B(IE+I))-(A(IC+I)-A(IG+I))) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      I=I+INC3
      J=J+INC4
  255 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  260 CONTINUE
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 280 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      KG=KF+KB
      KH=KG+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      C6=TRIGS(KG+1)
      S6=TRIGS(KG+2)
      C7=TRIGS(KH+1)
      S7=TRIGS(KH+2)
      DO 270 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 265 IJK=1,LOT

      WRITE(*,*) 'Big sorry from FPASSM: factor 8 only coded as last factor'
      STOP

      I=I+INC3
      J=J+INC4
  265 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  270 CONTINUE
      JBASE=JBASE+JUMP
  280 CONTINUE
      RETURN
      END


      SUBROUTINE IPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! IPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Originally VPASSM did not allow to perform the inverse FFT and so    *
! one had to rearrange the data externally in some way that the FFT    *
! could be done with the original phase factors. This required some    *
! additional calculation time and was hence not optimal. IPASSM is     *
! now the 'VPASSM'-version with conjugate phase factors needed for     *
! the inverse transform which can now be performed at same perfomance  *
! rates as the forward transform!                                      *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) RETURN
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO==1) GOTO 10
      IF (IGO==2) GOTO 50
      IF (IGO==3) GOTO 90
      IF (IGO==4) GOTO 130
      IF (IGO==5) GOTO 170
      IF (IGO==6) GOTO 210
      IF (IGO==7) GOTO 250
!      GOTO (10,50,90,130,170,210,250),IGO
! end test change doris
!
!
! Coding for factor 2:
!
   10 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      DO 20 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 15 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IB+I)
      D(JA+J)=B(IA+I)+B(IB+I)
      C(JB+J)=A(IA+I)-A(IB+I)
      D(JB+J)=B(IA+I)-B(IB+I)
      I=I+INC3
      J=J+INC4
   15 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   20 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 40 K=LA1,M,LA
      KB=K+K-2
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      DO 30 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 25 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IB+I)
      D(JA+J)=B(IA+I)+B(IB+I)
      C(JB+J)=C1*(A(IA+I)-A(IB+I))+S1*(B(IA+I)-B(IB+I))
      D(JB+J)=C1*(B(IA+I)-B(IB+I))-S1*(A(IA+I)-A(IB+I))
      I=I+INC3
      J=J+INC4
   25 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   30 CONTINUE
      JBASE=JBASE+JUMP
   40 CONTINUE
      RETURN
!
! Coding for factor 3:
!
   50 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      DO 60 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=(A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC3
      J=J+INC4
   55 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   60 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 80 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      DO 70 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 65 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)= &
     &    C1*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))) &
     &   +S1*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      D(JB+J)= &
     &    C1*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))) &
     &   -S1*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I))))
      C(JC+J)= &
     &    C2*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))) &
     &   +S2*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      D(JC+J)= &
     &    C2*((B(IA+I)-0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))) &
     &   -S2*((A(IA+I)-0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I))))
      I=I+INC3
      J=J+INC4
   65 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   70 CONTINUE
      JBASE=JBASE+JUMP
   80 CONTINUE
      RETURN
!
! Coding for factor 4:
!
   90 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      DO 100 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 95 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      C(JC+J)=(A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      D(JC+J)=(B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I))
      C(JB+J)=(A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I))
      C(JD+J)=(A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I))
      D(JB+J)=(B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I))
      D(JD+J)=(B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I))
      I=I+INC3
      J=J+INC4
   95 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  100 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 120 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      DO 110 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 105 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      C(JC+J)= &
     &    C2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I))) &
     &   +S2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      D(JC+J)= &
     &    C2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I))) &
     &   -S2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I)))
      C(JB+J)= &
     &    C1*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I))) &
     &   +S1*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      D(JB+J)= &
     &    C1*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I))) &
     &   -S1*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I)))
      C(JD+J)= &
     &    C3*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I))) &
     &   +S3*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      D(JD+J)= &
     &    C3*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I))) &
     &   -S3*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I)))
      I=I+INC3
      J=J+INC4
  105 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  110 CONTINUE
      JBASE=JBASE+JUMP
  120 CONTINUE
      RETURN
!
! Coding for factor 5:
!
  130 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      DO 140 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  135 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  140 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 160 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      DO 150 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 145 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)= &
     &    C1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +S1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JB+J)= &
     &   -S1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +C1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JE+J)= &
     &    C4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +S4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JE+J)= &
     &   -S4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))) &
     &   +C4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JC+J)= &
     &    C2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +S2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JC+J)= &
     &   -S2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +C2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      C(JD+J)= &
     &    C3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +S3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JD+J)= &
     &   -S3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I))) &
     &      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))) &
     &   +C3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I))) &
     &      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      I=I+INC3
      J=J+INC4
  145 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  150 CONTINUE
      JBASE=JBASE+JUMP
  160 CONTINUE
      RETURN
!
! Coding for factor 6:
!
  170 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      DO 180 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 175 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(ID+I))+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      C(JD+J)=(A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))
      D(JA+J)=(B(IA+I)+B(ID+I))+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      D(JD+J)=(B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I)))
      C(JB+J)= &
     &     (A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      C(JF+J)= &
     &     (A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      D(JB+J)= &
     &     (B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      D(JF+J)= &
     &     (B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      C(JC+J)= &
     &     (A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      C(JE+J)= &
     &     (A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      D(JC+J)= &
     &     (B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      D(JE+J)= &
     &     (B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  175 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  180 CONTINUE
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 200 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      DO 190 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 185 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(ID+I))+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      D(JA+J)=(B(IA+I)+B(ID+I))+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      C(JD+J)= &
     &   C3*((A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))) &
     &  +S3*((B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))))
      D(JD+J)= &
     &  -S3*((A(IA+I)-A(ID+I))-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))) &
     &  +C3*((B(IA+I)-B(ID+I))-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))))
      C(JB+J)= &
     &  C1*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +S1*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      D(JB+J)= &
     & -S1*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +C1*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      C(JF+J)= &
     &  C5*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +S5*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      D(JF+J)= &
     & -S5*((A(IA+I)-A(ID+I))+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))) &
     & +C5*((B(IA+I)-B(ID+I))+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I))))
      C(JC+J)= &
     &  C2*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +S2*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      D(JC+J)= &
     & -S2*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +C2*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      C(JE+J)= &
     &  C4*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +S4*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      D(JE+J)= &
     & -S4*((A(IA+I)+A(ID+I))-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &                      -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))) &
     & +C4*((B(IA+I)+B(ID+I))-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &                      +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  185 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  190 CONTINUE
      JBASE=JBASE+JUMP
  200 CONTINUE
      RETURN
!
! Coding for factor 7:
!
  210 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      IG=IF+IINK
      JG=JF+JINK
      DO 220 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I)) &
     &               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I)) &
     &               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &               +(SPIS2*(B(IB+I)-B(IG+I)) &
     &                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &               -(SPIS2*(B(IB+I)-B(IG+I)) &
     &                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &               -(SPIS2*(A(IB+I)-A(IG+I)) &
     &                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &               +(SPIS2*(A(IB+I)-A(IG+I)) &
     &                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &               +(SPIS3*(B(IB+I)-B(IG+I)) &
     &                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &               -(SPIS3*(B(IB+I)-B(IG+I)) &
     &                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &               -(SPIS3*(A(IB+I)-A(IG+I)) &
     &                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &               +(SPIS3*(A(IB+I)-A(IG+I)) &
     &                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &               +(SPIS1*(B(IB+I)-B(IG+I)) &
     &                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &               -(SPIS1*(B(IB+I)-B(IG+I)) &
     &                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &               -(SPIS1*(A(IB+I)-A(IG+I)) &
     &                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &               +(SPIS1*(A(IB+I)-A(IG+I)) &
     &                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  215 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  220 CONTINUE
      IF (LA==M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 240 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      KG=KF+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      C6=TRIGS(KG+1)
      S6=TRIGS(KG+2)
      DO 230 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 225 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I)) &
     &               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I)) &
     &               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)= &
     &       C1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            +(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &      +S1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            -(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JB+J)= &
     &       -S1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            +(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       +C1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            -(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JG+J)= &
     &        C6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            -(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       +S6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            +(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JG+J)= &
     &       -S6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I)) &
     &             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &            -(SPIS2*(B(IB+I)-B(IG+I)) &
     &             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))) &
     &       +C6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I)) &
     &             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &            +(SPIS2*(A(IB+I)-A(IG+I)) &
     &             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JC+J)= &
     &        C2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            +(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +S2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            -(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JC+J)= &
     &       -S2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            +(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +C2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            -(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JF+J)= &
     &        C5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            -(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +S5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            +(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JF+J)= &
     &       -S5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I)) &
     &             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &            -(SPIS3*(B(IB+I)-B(IG+I)) &
     &             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))) &
     &       +C5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I)) &
     &             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &            +(SPIS3*(A(IB+I)-A(IG+I)) &
     &             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JD+J)= &
     &        C3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            +(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +S3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            -(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JD+J)= &
     &       -S3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            +(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +C3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            -(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      C(JE+J)= &
     &        C4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            -(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +S4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            +(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JE+J)= &
     &       -S4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I)) &
     &             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &            -(SPIS1*(B(IB+I)-B(IG+I)) &
     &             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))) &
     &       +C4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I)) &
     &             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &            +(SPIS1*(A(IB+I)-A(IG+I)) &
     &             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  225 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  230 CONTINUE
      JBASE=JBASE+JUMP
  240 CONTINUE
      RETURN
!
! Coding for factor 8:
!
  250 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      ID=IC+IINK
      JD=JC+JINK
      IE=ID+IINK
      JE=JD+JINK
      IF=IE+IINK
      JF=JE+JINK
      IG=IF+IINK
      JG=JF+JINK
      IH=IG+IINK
      JH=JG+JINK
      DO 260 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 255 IJK=1,LOT
      C(JA+J)=((A(IA+I)+A(IE+I))+(A(IB+I)+A(IF+I))) &
     &       +((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
      D(JA+J)=((B(IA+I)+B(IE+I))+(B(IB+I)+B(IF+I))) &
     &       +((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
      C(JE+J)=((A(IA+I)+A(IE+I))-(A(IB+I)+A(IF+I))) &
     &       +((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
      D(JE+J)=((B(IA+I)+B(IE+I))-(B(IB+I)+B(IF+I))) &
     &       +((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
      C(JB+J)=    ((A(IA+I)-A(IE+I))+(B(IC+I)-B(IG+I))) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JB+J)=    ((B(IA+I)-B(IE+I))-(A(IC+I)-A(IG+I))) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JF+J)=    ((A(IA+I)-A(IE+I))+(B(IC+I)-B(IG+I))) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JF+J)=    ((B(IA+I)-B(IE+I))-(A(IC+I)-A(IG+I))) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JC+J)=((A(IA+I)+A(IE+I))-(A(IC+I)+A(IG+I))) &
     &       +((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JC+J)=((B(IA+I)+B(IE+I))-(B(IC+I)+B(IG+I))) &
     &       -((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JG+J)=((A(IA+I)+A(IE+I))-(A(IC+I)+A(IG+I))) &
     &       -((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JG+J)=((B(IA+I)+B(IE+I))-(B(IC+I)+B(IG+I))) &
     &       +((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JD+J)=    ((A(IA+I)-A(IE+I))-(B(IC+I)-B(IG+I))) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JD+J)=    ((B(IA+I)-B(IE+I))+(A(IC+I)-A(IG+I))) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JH+J)=    ((A(IA+I)-A(IE+I))-(B(IC+I)-B(IG+I))) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JH+J)=    ((B(IA+I)-B(IE+I))+(A(IC+I)-A(IG+I))) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      I=I+INC3
      J=J+INC4
  255 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  260 CONTINUE
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 280 K=LA1,M,LA
      KB=K+K-2
      KC=KB+KB
      KD=KC+KB
      KE=KD+KB
      KF=KE+KB
      KG=KF+KB
      KH=KG+KB
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      C2=TRIGS(KC+1)
      S2=TRIGS(KC+2)
      C3=TRIGS(KD+1)
      S3=TRIGS(KD+2)
      C4=TRIGS(KE+1)
      S4=TRIGS(KE+2)
      C5=TRIGS(KF+1)
      S5=TRIGS(KF+2)
      C6=TRIGS(KG+1)
      S6=TRIGS(KG+2)
      C7=TRIGS(KH+1)
      S7=TRIGS(KH+2)
      DO 270 L=1,LA
      I=IBASE
      J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 265 IJK=1,LOT

      WRITE(*,*) 'Big sorry from IPASSM: factor 8 only coded as last factor'
      STOP

      I=I+INC3
      J=J+INC4
  265 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  270 CONTINUE
      JBASE=JBASE+JUMP
  280 CONTINUE
      RETURN
      END


      SUBROUTINE SPASSM(A,B,C,D,INC1,INC2,LOT,N,ISIGN)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! SPASSM is derived from the TEMPERTON/FULKER routine 'VPASSM' which   *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! SPASSM is the special version for 'single factor transformations'.   *
! Variables:                                                           *
!   A is the array containing the real parts of the data vector        *
!   B is the array containing the imaginary parts of the data vector   *
!   C is a work array                                                  *
!   D is a work array                                                  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the addressing increment between data sets in A and B      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   ISIGN gives the type of transformation (>=0: forward, <0: inverse) *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      IGO=N-1
      IF (IGO>7) RETURN
      IF (ISIGN<0) GOTO 1000
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO==1) GOTO 10
      IF (IGO==2) GOTO 50
      IF (IGO==3) GOTO 90
      IF (IGO==4) GOTO 130
      IF (IGO==5) GOTO 170
      IF (IGO==6) GOTO 210
      IF (IGO==7) GOTO 250
!      GOTO (10,50,90,130,170,210,250),IGO
! end test change doris
!
! Coding for factor 2:
!
   10 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      I=0
      J=0
      DO 15 IJK=1,LOT
      C(JB+J)=A(IA+I)-A(IB+I)
      D(JB+J)=B(IA+I)-B(IB+I)
      I=I+INC2
      J=J+1
   15 CONTINUE
      I=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 25 IJK=1,LOT
      A(IA+I)=A(IA+I)+A(IB+I)
      B(IA+I)=B(IA+I)+B(IB+I)
      I=I+INC2
   25 CONTINUE
      I=0
      J=0
      DO 35 IJK=1,LOT
      A(IB+I)=C(JB+J)
      B(IB+I)=D(JB+J)
      I=I+INC2
      J=J+1
   35 CONTINUE
      RETURN
!
! Coding for factor 3:
!
   50 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IB+I)+A(IC+I)
      D(JA+J)=B(IB+I)+B(IC+I)
      C(JB+J)=(0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC2
      J=J+1
   55 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 65 IJK=1,LOT
      A(IB+I)=A(IA+I)-C(JB+J)
      A(IC+I)=A(IA+I)-C(JC+J)
      B(IB+I)=B(IA+I)-D(JB+J)
      B(IC+I)=B(IA+I)-D(JC+J)
      I=I+INC2
      J=J+1
   65 CONTINUE
      I=0
      J=0
      DO 75 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
   75 CONTINUE
      RETURN
!
! Coding for factor 4:
!
   90 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 95 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IC+I)
      C(JB+J)=A(IB+I)+A(ID+I)
      D(JA+J)=B(IA+I)+B(IC+I)
      D(JB+J)=B(IB+I)+B(ID+I)
      C(JC+J)=A(IA+I)-A(IC+I)
      C(JD+J)=B(IB+I)-B(ID+I)
      D(JC+J)=B(IA+I)-B(IC+I)
      D(JD+J)=A(IB+I)-A(ID+I)
      I=I+INC2
      J=J+1
   95 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 105 IJK=1,LOT
      A(IA+I)=C(JA+J)+C(JB+J)
      A(IC+I)=C(JA+J)-C(JB+J)
      B(IA+I)=D(JA+J)+D(JB+J)
      B(IC+I)=D(JA+J)-D(JB+J)
      A(IB+I)=C(JC+J)-C(JD+J)
      A(ID+I)=C(JC+J)+C(JD+J)
      B(IB+I)=D(JC+J)+D(JD+J)
      B(ID+I)=D(JC+J)-D(JD+J)
      I=I+INC2
      J=J+1
  105 CONTINUE
      RETURN
!
! Coding for factor 5:
!
  130 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I))) &
     &  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I))) &
     &  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I))) &
     &  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I))) &
     &  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC2
      J=J+1
  135 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 145 IJK=1,LOT
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      I=I+INC2
      J=J+1
  145 CONTINUE
      I=0
      J=0
      DO 155 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
  155 CONTINUE
      RETURN
!
! Coding for factor 6:
!
  170 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 175 IJK=1,LOT
      C(JA+J)= A(ID+I)+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      C(JD+J)=-A(ID+I)-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))
      D(JA+J)= B(ID+I)+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      D(JD+J)=-B(ID+I)-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I)))
      C(JB+J)= &
     &     -A(ID+I)+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &             -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      C(JF+J)= &
     &     -A(ID+I)+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &             +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      D(JB+J)= &
     &     -B(ID+I)+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &             +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      D(JF+J)= &
     &     -B(ID+I)+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &             -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      C(JC+J)= &
     &      A(ID+I)-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &             -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      C(JE+J)= &
     &      A(ID+I)-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &             +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      D(JC+J)= &
     &      B(ID+I)-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &             +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      D(JE+J)= &
     &      B(ID+I)-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &             -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      I=I+INC2
      J=J+1
  175 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 185 IJK=1,LOT
      A(ID+I)=A(IA+I)+C(JD+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
  185 CONTINUE
      I=0
      J=0
      DO 195 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
  195 CONTINUE
      RETURN
!
! Coding for factor 7:
!
  210 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      IG=IF+INC1
      JG=JF+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IG+I)) &
     &       +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=(B(IB+I)+B(IG+I)) &
     &       +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(CPIS2*(A(IB+I)+A(IG+I)) &
     &        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &       -(SPIS2*(B(IB+I)-B(IG+I)) &
     &        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(CPIS2*(A(IB+I)+A(IG+I)) &
     &        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &       +(SPIS2*(B(IB+I)-B(IG+I)) &
     &        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(CPIS2*(B(IB+I)+B(IG+I)) &
     &        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &       +(SPIS2*(A(IB+I)-A(IG+I)) &
     &        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(CPIS2*(B(IB+I)+B(IG+I)) &
     &        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &       -(SPIS2*(A(IB+I)-A(IG+I)) &
     &        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(-CPIS3*(A(IB+I)+A(IG+I)) &
     &         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &        -(SPIS3*(B(IB+I)-B(IG+I)) &
     &         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(-CPIS3*(A(IB+I)+A(IG+I)) &
     &         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &        +(SPIS3*(B(IB+I)-B(IG+I)) &
     &         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(-CPIS3*(B(IB+I)+B(IG+I)) &
     &         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &        +(SPIS3*(A(IB+I)-A(IG+I)) &
     &         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(-CPIS3*(B(IB+I)+B(IG+I)) &
     &         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &        -(SPIS3*(A(IB+I)-A(IG+I)) &
     &         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(-CPIS1*(A(IB+I)+A(IG+I)) &
     &         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &        -(SPIS1*(B(IB+I)-B(IG+I)) &
     &         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(-CPIS1*(A(IB+I)+A(IG+I)) &
     &         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &        +(SPIS1*(B(IB+I)-B(IG+I)) &
     &         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(-CPIS1*(B(IB+I)+B(IG+I)) &
     &         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &        +(SPIS1*(A(IB+I)-A(IG+I)) &
     &         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(-CPIS1*(B(IB+I)+B(IG+I)) &
     &         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &        -(SPIS1*(A(IB+I)-A(IG+I)) &
     &         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC2
      J=J+1
  215 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 225 IJK=1,LOT
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IG+I)=A(IA+I)+C(JG+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IG+I)=B(IA+I)+D(JG+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
  225 CONTINUE
      I=0
      J=0
      DO 235 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
  235 CONTINUE
      RETURN
!
! Coding for factor 8:
!
  250 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      IG=IF+INC1
      JG=JF+LOT
      IH=IG+INC1
      JH=JG+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 255 IJK=1,LOT
      C(JA+J)=           A(IE+I)+(A(IB+I)+A(IF+I)) &
     &       +((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
      D(JA+J)=           B(IE+I)+(B(IB+I)+B(IF+I)) &
     &       +((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
      C(JE+J)=           A(IE+I)-(A(IB+I)+A(IF+I)) &
     &       +((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
      D(JE+J)=           B(IE+I)-(B(IB+I)+B(IF+I)) &
     &       +((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
      C(JB+J)=              -A(IE+I)-(B(IC+I)-B(IG+I)) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JB+J)=              -B(IE+I)+(A(IC+I)-A(IG+I)) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JF+J)=              -A(IE+I)-(B(IC+I)-B(IG+I)) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JF+J)=              -B(IE+I)+(A(IC+I)-A(IG+I)) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JC+J)=           A(IE+I)-(A(IC+I)+A(IG+I)) &
     &       -((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JC+J)=           B(IE+I)-(B(IC+I)+B(IG+I)) &
     &       +((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JG+J)=           A(IE+I)-(A(IC+I)+A(IG+I)) &
     &       +((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JG+J)=           B(IE+I)-(B(IC+I)+B(IG+I)) &
     &       -((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JD+J)=              -A(IE+I)+(B(IC+I)-B(IG+I)) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JD+J)=              -B(IE+I)-(A(IC+I)-A(IG+I)) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JH+J)=              -A(IE+I)+(B(IC+I)-B(IG+I)) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JH+J)=              -B(IE+I)-(A(IC+I)-A(IG+I)) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      I=I+INC2
      J=J+1
  255 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 265 IJK=1,LOT
      A(IH+I)=A(IA+I)+C(JH+J)
      B(IH+I)=B(IA+I)+D(JH+J)
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IG+I)=A(IA+I)+C(JG+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IG+I)=B(IA+I)+D(JG+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
  265 CONTINUE
      I=0
      J=0
      DO 275 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
  275 CONTINUE
      RETURN
 1000 CONTINUE
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO==1) GOTO 1010
      IF (IGO==2) GOTO 1050
      IF (IGO==3) GOTO 1090
      IF (IGO==4) GOTO 1130
      IF (IGO==5) GOTO 1170
      IF (IGO==6) GOTO 1210
      IF (IGO==7) GOTO 1250
!      GOTO (1010,1050,1090,1130,1170,1210,1250),IGO
! end test change doris
!
! Coding for factor 2:
!
 1010 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      I=0
      J=0
      DO 1015 IJK=1,LOT
      C(JB+J)=A(IA+I)-A(IB+I)
      D(JB+J)=B(IA+I)-B(IB+I)
      I=I+INC2
      J=J+1
 1015 CONTINUE
      I=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1025 IJK=1,LOT
      A(IA+I)=A(IA+I)+A(IB+I)
      B(IA+I)=B(IA+I)+B(IB+I)
      I=I+INC2
 1025 CONTINUE
      I=0
      J=0
      DO 1035 IJK=1,LOT
      A(IB+I)=C(JB+J)
      B(IB+I)=D(JB+J)
      I=I+INC2
      J=J+1
 1035 CONTINUE
      RETURN
!
! Coding for factor 3:
!
 1050 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1055 IJK=1,LOT
      C(JA+J)=A(IB+I)+A(IC+I)
      D(JA+J)=B(IB+I)+B(IC+I)
      C(JB+J)=(0.5_q*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(0.5_q*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(0.5_q*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(0.5_q*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC2
      J=J+1
 1055 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1065 IJK=1,LOT
      A(IB+I)=A(IA+I)-C(JB+J)
      A(IC+I)=A(IA+I)-C(JC+J)
      B(IB+I)=B(IA+I)-D(JB+J)
      B(IC+I)=B(IA+I)-D(JC+J)
      I=I+INC2
      J=J+1
 1065 CONTINUE
      I=0
      J=0
      DO 1075 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
 1075 CONTINUE
      RETURN
!
! Coding for factor 4:
!
 1090 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1095 IJK=1,LOT
      C(JA+J)=A(IA+I)+A(IC+I)
      C(JB+J)=A(IB+I)+A(ID+I)
      D(JA+J)=B(IA+I)+B(IC+I)
      D(JB+J)=B(IB+I)+B(ID+I)
      C(JC+J)=A(IA+I)-A(IC+I)
      C(JD+J)=B(IB+I)-B(ID+I)
      D(JC+J)=B(IA+I)-B(IC+I)
      D(JD+J)=A(IB+I)-A(ID+I)
      I=I+INC2
      J=J+1
 1095 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1105 IJK=1,LOT
      A(IA+I)=C(JA+J)+C(JB+J)
      A(IC+I)=C(JA+J)-C(JB+J)
      B(IA+I)=D(JA+J)+D(JB+J)
      B(IC+I)=D(JA+J)-D(JB+J)
      A(IB+I)=C(JC+J)+C(JD+J)
      A(ID+I)=C(JC+J)-C(JD+J)
      B(IB+I)=D(JC+J)-D(JD+J)
      B(ID+I)=D(JC+J)+D(JD+J)
      I=I+INC2
      J=J+1
 1105 CONTINUE
      RETURN
!
! Coding for factor 5:
!
 1130 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1135 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I))) &
     &  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I))) &
     &  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I))) &
     &  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I))) &
     &  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I))) &
     &  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I))) &
     &  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC2
      J=J+1
 1135 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1145 IJK=1,LOT
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      I=I+INC2
      J=J+1
 1145 CONTINUE
      I=0
      J=0
      DO 1155 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
 1155 CONTINUE
      RETURN
!
! Coding for factor 6:
!
 1170 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1175 IJK=1,LOT
      C(JA+J)= A(ID+I)+((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I)))
      C(JD+J)=-A(ID+I)-((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I)))
      D(JA+J)= B(ID+I)+((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I)))
      D(JD+J)=-B(ID+I)-((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I)))
      C(JB+J)= &
     &     -A(ID+I)+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &             +SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      C(JF+J)= &
     &     -A(ID+I)+0.5_q*((A(IB+I)+A(IF+I))-(A(IC+I)+A(IE+I))) &
     &             -SIN60*((B(IB+I)-B(IF+I))+(B(IC+I)-B(IE+I)))
      D(JB+J)= &
     &     -B(ID+I)+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &             -SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      D(JF+J)= &
     &     -B(ID+I)+0.5_q*((B(IB+I)+B(IF+I))-(B(IC+I)+B(IE+I))) &
     &             +SIN60*((A(IB+I)-A(IF+I))+(A(IC+I)-A(IE+I)))
      C(JC+J)= &
     &      A(ID+I)-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &             +SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      C(JE+J)= &
     &      A(ID+I)-0.5_q*((A(IB+I)+A(IF+I))+(A(IC+I)+A(IE+I))) &
     &             -SIN60*((B(IB+I)-B(IF+I))-(B(IC+I)-B(IE+I)))
      D(JC+J)= &
     &      B(ID+I)-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &             -SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      D(JE+J)= &
     &      B(ID+I)-0.5_q*((B(IB+I)+B(IF+I))+(B(IC+I)+B(IE+I))) &
     &             +SIN60*((A(IB+I)-A(IF+I))-(A(IC+I)-A(IE+I)))
      I=I+INC2
      J=J+1
 1175 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1185 IJK=1,LOT
      A(ID+I)=A(IA+I)+C(JD+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
 1185 CONTINUE
      I=0
      J=0
      DO 1195 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
 1195 CONTINUE
      RETURN
!
! Coding for factor 7:
!
 1210 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      IG=IF+INC1
      JG=JF+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1215 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IG+I)) &
     &       +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=(B(IB+I)+B(IG+I)) &
     &       +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(CPIS2*(A(IB+I)+A(IG+I)) &
     &        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &       +(SPIS2*(B(IB+I)-B(IG+I)) &
     &        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(CPIS2*(A(IB+I)+A(IG+I)) &
     &        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I))) &
     &       -(SPIS2*(B(IB+I)-B(IG+I)) &
     &        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(CPIS2*(B(IB+I)+B(IG+I)) &
     &        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &       -(SPIS2*(A(IB+I)-A(IG+I)) &
     &        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(CPIS2*(B(IB+I)+B(IG+I)) &
     &        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I))) &
     &       +(SPIS2*(A(IB+I)-A(IG+I)) &
     &        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(-CPIS3*(A(IB+I)+A(IG+I)) &
     &         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &        +(SPIS3*(B(IB+I)-B(IG+I)) &
     &         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(-CPIS3*(A(IB+I)+A(IG+I)) &
     &         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I))) &
     &        -(SPIS3*(B(IB+I)-B(IG+I)) &
     &         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(-CPIS3*(B(IB+I)+B(IG+I)) &
     &         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &        -(SPIS3*(A(IB+I)-A(IG+I)) &
     &         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(-CPIS3*(B(IB+I)+B(IG+I)) &
     &         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I))) &
     &        +(SPIS3*(A(IB+I)-A(IG+I)) &
     &         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(-CPIS1*(A(IB+I)+A(IG+I)) &
     &         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &        +(SPIS1*(B(IB+I)-B(IG+I)) &
     &         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(-CPIS1*(A(IB+I)+A(IG+I)) &
     &         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I))) &
     &        -(SPIS1*(B(IB+I)-B(IG+I)) &
     &         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(-CPIS1*(B(IB+I)+B(IG+I)) &
     &         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &        -(SPIS1*(A(IB+I)-A(IG+I)) &
     &         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(-CPIS1*(B(IB+I)+B(IG+I)) &
     &         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I))) &
     &        +(SPIS1*(A(IB+I)-A(IG+I)) &
     &         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC2
      J=J+1
 1215 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1225 IJK=1,LOT
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IG+I)=A(IA+I)+C(JG+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IG+I)=B(IA+I)+D(JG+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
 1225 CONTINUE
      I=0
      J=0
      DO 1235 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
 1235 CONTINUE
      RETURN
!
! Coding for factor 8:
!
 1250 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      ID=IC+INC1
      JD=JC+LOT
      IE=ID+INC1
      JE=JD+LOT
      IF=IE+INC1
      JF=JE+LOT
      IG=IF+INC1
      JG=JF+LOT
      IH=IG+INC1
      JH=JG+LOT
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1255 IJK=1,LOT
      C(JA+J)=           A(IE+I)+(A(IB+I)+A(IF+I)) &
     &       +((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
      D(JA+J)=           B(IE+I)+(B(IB+I)+B(IF+I)) &
     &       +((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
      C(JE+J)=           A(IE+I)-(A(IB+I)+A(IF+I)) &
     &       +((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
      D(JE+J)=           B(IE+I)-(B(IB+I)+B(IF+I)) &
     &       +((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
      C(JB+J)=              -A(IE+I)+(B(IC+I)-B(IG+I)) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JB+J)=              -B(IE+I)-(A(IC+I)-A(IG+I)) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JF+J)=              -A(IE+I)+(B(IC+I)-B(IG+I)) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           +((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JF+J)=              -B(IE+I)-(A(IC+I)-A(IG+I)) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           -((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JC+J)=           A(IE+I)-(A(IC+I)+A(IG+I)) &
     &       +((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JC+J)=           B(IE+I)-(B(IC+I)+B(IG+I)) &
     &       -((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JG+J)=           A(IE+I)-(A(IC+I)+A(IG+I)) &
     &       -((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
      D(JG+J)=           B(IE+I)-(B(IC+I)+B(IG+I)) &
     &       +((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
      C(JD+J)=              -A(IE+I)-(B(IC+I)-B(IG+I)) &
     &    -SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JD+J)=              -B(IE+I)+(A(IC+I)-A(IG+I)) &
     &    -SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      C(JH+J)=              -A(IE+I)-(B(IC+I)-B(IG+I)) &
     &    +SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I))) &
     &           -((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
      D(JH+J)=              -B(IE+I)+(A(IC+I)-A(IG+I)) &
     &    +SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I))) &
     &           +((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
      I=I+INC2
      J=J+1
 1255 CONTINUE
      I=0
      J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 1265 IJK=1,LOT
      A(IH+I)=A(IA+I)+C(JH+J)
      B(IH+I)=B(IA+I)+D(JH+J)
      A(IB+I)=A(IA+I)+C(JB+J)
      A(IG+I)=A(IA+I)+C(JG+J)
      B(IB+I)=B(IA+I)+D(JB+J)
      B(IG+I)=B(IA+I)+D(JG+J)
      A(IC+I)=A(IA+I)+C(JC+J)
      A(IF+I)=A(IA+I)+C(JF+J)
      B(IC+I)=B(IA+I)+D(JC+J)
      B(IF+I)=B(IA+I)+D(JF+J)
      A(ID+I)=A(IA+I)+C(JD+J)
      A(IE+I)=A(IA+I)+C(JE+J)
      B(ID+I)=B(IA+I)+D(JD+J)
      B(IE+I)=B(IA+I)+D(JE+J)
      I=I+INC2
      J=J+1
 1265 CONTINUE
      I=0
      J=0
      DO 1275 IJK=1,LOT
      A(IA+I)=A(IA+I)+C(JA+J)
      B(IA+I)=B(IA+I)+D(JA+J)
      I=I+INC2
      J=J+1
 1275 CONTINUE
      RETURN
      END

#else

!=======================================================================
! CODE MODIFICATIONS --- EXPERIMENTAL (no warranty)
!=======================================================================
!
! this part contains subroutine definitions
!   + FPASSM[_1_1]
!   + IPASSM[_1_1]
!   + SPASSM[_1_1]
!   + CFFTML
!   + FFTC3V
!   + FFTR3V
!   + RHFFTM
!
!=======================================================================
! WHO:           Florian Wende (FW)
! INSTITUTION:   Zuse Institute Berlin (ZIB)
! MAIL:          wende.at.zib.de
! CONTEXT:       IPCC
! LAST MODIFIED: Sep 17, 2015
!
! **************** CHANGE HISTORY (BEGIN) ******************************
!
! June 02, 2015: * removed almost all GOTO and labeled LOOP constructs
!                * introduce local scalar variables for loading from
!                  A(:) and B(:) fields (seems to help vectorization):
!                  using these local scalars instead of accessing A and
!                  B reduces the number of (non-unit-stride/indexed)
!                  loads (see Intel optimization report) 
!                  => formulas are more compact
!                * OpenMP4 SIMD directives for loop vectorization
!                * introduced timing events (timer_fw module)
!
!                CORRECTNESS: checked (restart: OK, full run: OK)
!
! Sep 17, 2015:  * added optimizations of SPASSM routine
!                * introduced specializations of [F,I,S]PASSM routines
!                  for INC1=INC2=1
!
! **************** CHANGE HISTORY (END) ********************************
!
! NOTES:
!    * to enabled OpenMP4 SIMD statements, VECTORIZE needs to be defined
!    * compile this file with "-(f)openmp" flag
!=======================================================================

      SUBROUTINE FPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! FPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************
      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)

      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) THEN 
         RETURN
      ENDIF
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
! Coding for factor 2:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               C(JA+J)=TEMP1+TEMP3
               D(JA+J)=TEMP2+TEMP4
               C(JB+J)=TEMP1-TEMP3
               D(JB+J)=TEMP2-TEMP4
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  C(JA+J)=TEMP1+TEMP3
                  D(JA+J)=TEMP2+TEMP4
                  C(JB+J)=C1*(TEMP1-TEMP3)-S1*(TEMP2-TEMP4)
                  D(JB+J)=S1*(TEMP1-TEMP3)+C1*(TEMP2-TEMP4)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
! Coding for factor 3:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP5)
               D(JA+J)=TEMP2+(TEMP4+TEMP6)
               TEMP7=SIN60*(TEMP3-TEMP5)
               TEMP8=SIN60*(TEMP4-TEMP6)
               C(JB+J)=TEMP1-0.5_q*(TEMP3+TEMP5)-TEMP8
               C(JC+J)=TEMP1-0.5_q*(TEMP3+TEMP5)+TEMP8
               D(JB+J)=TEMP2-0.5_q*(TEMP4+TEMP6)+TEMP7
               D(JC+J)=TEMP2-0.5_q*(TEMP4+TEMP6)-TEMP7
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP5)
                  D(JA+J)=TEMP2+(TEMP4+TEMP6)
                  TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
                  TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
                  TEMP9=SIN60*(TEMP4-TEMP6)
                  TEMP10=SIN60*(TEMP3-TEMP5)
                  C(JB+J)=C1*(TEMP7-TEMP9)-S1*(TEMP8+TEMP10)
                  D(JB+J)=S1*(TEMP7-TEMP9)+C1*(TEMP8+TEMP10)
                  C(JC+J)=C2*(TEMP7+TEMP9)-S2*(TEMP8-TEMP10)
                  D(JC+J)=S2*(TEMP7+TEMP9)+C2*(TEMP8-TEMP10)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
! Coding for factor 4:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
               C(JC+J)=(TEMP1+TEMP5)-(TEMP3+TEMP7)
               D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
               D(JC+J)=(TEMP2+TEMP6)-(TEMP4+TEMP8)
               C(JB+J)=(TEMP1-TEMP5)-(TEMP4-TEMP8)
               C(JD+J)=(TEMP1-TEMP5)+(TEMP4-TEMP8)
               D(JB+J)=(TEMP2-TEMP6)+(TEMP3-TEMP7)
               D(JD+J)=(TEMP2-TEMP6)-(TEMP3-TEMP7)
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
                  D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
                  TEMP9=(TEMP1+TEMP5)-(TEMP3+TEMP7)
                  TEMP10=(TEMP2+TEMP6)-(TEMP4+TEMP8)
                  C(JC+J)=C2*TEMP9-S2*TEMP10
                  D(JC+J)=S2*TEMP9+C2*TEMP10
                  TEMP11=(TEMP1-TEMP5)-(TEMP4-TEMP8)
                  TEMP12=(TEMP2-TEMP6)+(TEMP3-TEMP7)
                  C(JB+J)=C1*TEMP11-S1*TEMP12
                  D(JB+J)=S1*TEMP11+C1*TEMP12
                  TEMP13=(TEMP1-TEMP5)+(TEMP4-TEMP8)
                  TEMP14=(TEMP2-TEMP6)-(TEMP3-TEMP7)
                  C(JD+J)=C3*TEMP13-S3*TEMP14
                  D(JD+J)=S3*TEMP13+C3*TEMP14
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN         
      ENDIF
      
      IF (IGO.EQ.4) THEN
! Coding for factor 5:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
               D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
               TEMP11=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
               TEMP12=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
               C(JB+J)=TEMP11-TEMP12
               C(JE+J)=TEMP11+TEMP12
               TEMP13=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
               TEMP14=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
               D(JB+J)=TEMP13+TEMP14
               D(JE+J)=TEMP13-TEMP14
               TEMP15=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
               TEMP16=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
               C(JC+J)=TEMP15-TEMP16
               C(JD+J)=TEMP15+TEMP16
               TEMP17=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
               TEMP18=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
               D(JC+J)=TEMP17+TEMP18
               D(JD+J)=TEMP17-TEMP18
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
                  D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
                  TEMP11=(TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7))-(SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8))
                  TEMP12=(TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8))+(SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7))
                  C(JB+J)=C1*TEMP11-S1*TEMP12
                  D(JB+J)=S1*TEMP11+C1*TEMP12
                  TEMP13=(TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7))+(SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8))
                  TEMP14=(TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8))-(SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7))
                  C(JE+J)=C4*TEMP13-S4*TEMP14
                  D(JE+J)=S4*TEMP13+C4*TEMP14
                  TEMP15=(TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7))-(SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8))
                  TEMP16=(TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8))+(SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7))
                  C(JC+J)=C2*TEMP15-S2*TEMP16
                  D(JC+J)=S2*TEMP15+C2*TEMP16
                  TEMP17=(TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7))+(SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8))
                  TEMP18=(TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8))-(SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7))
                  C(JD+J)=C3*TEMP17-S3*TEMP18
                  D(JD+J)=S3*TEMP17+C3*TEMP18
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
         
      IF (IGO.EQ.5) THEN
! Coding for factor 6:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
               C(JD+J)=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
               D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
               D(JD+J)=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP13=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))
               TEMP14=SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
               C(JB+J)=TEMP13-TEMP14
               C(JF+J)=TEMP13+TEMP14
               TEMP15=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP16=SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
               D(JB+J)=TEMP15+TEMP16
               D(JF+J)=TEMP15-TEMP16
               TEMP17=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))
               TEMP18=SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
               C(JC+J)=TEMP17-TEMP18
               C(JE+J)=TEMP17+TEMP18
               TEMP19=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))
               TEMP20=SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
               D(JC+J)=TEMP19+TEMP20
               D(JE+J)=TEMP19-TEMP20
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
                  D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
                  C(JD+J)=C3*((TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9)))-S3*((TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10)))
                  D(JD+J)=S3*((TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9)))+C3*((TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10)))
                  TEMP13=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP14=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JB+J)=C1*TEMP13-S1*TEMP14
                  D(JB+J)=S1*TEMP13+C1*TEMP14
                  TEMP15=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP16=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JF+J)=C5*TEMP15-S5*TEMP16
                  D(JF+J)=S5*TEMP15+C5*TEMP16
                  TEMP17=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP18=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JC+J)=C2*TEMP17-S2*TEMP18
                  D(JC+J)=S2*TEMP17+C2*TEMP18
                  TEMP19=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP20=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JE+J)=C4*TEMP19-S4*TEMP20
                  D(JE+J)=S4*TEMP19+C4*TEMP20
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
! Coding for factor 7:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
               D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
               TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
               TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
               C(JB+J)=TEMP15-TEMP16
               C(JG+J)=TEMP15+TEMP16
               TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
               TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
               D(JB+J)=TEMP17+TEMP18
               D(JG+J)=TEMP17-TEMP18
               TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
               TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
               C(JC+J)=TEMP19-TEMP20
               C(JF+J)=TEMP19+TEMP20
               TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
               TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
               D(JC+J)=TEMP21+TEMP22
               D(JF+J)=TEMP21-TEMP22
               TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
               TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
               C(JD+J)=TEMP23-TEMP24
               C(JE+J)=TEMP23+TEMP24
               TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
               TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
               D(JD+J)=TEMP25+TEMP26
               D(JE+J)=TEMP25-TEMP26
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  TEMP13=A(IG+I)
                  TEMP14=B(IG+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
                  D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
                  TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
                  TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
                  TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
                  TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
                  C(JB+J)=C1*(TEMP15-TEMP16)-S1*(TEMP17+TEMP18)
                  D(JB+J)=S1*(TEMP15-TEMP16)+C1*(TEMP17+TEMP18)
                  C(JG+J)=C6*(TEMP15+TEMP16)-S6*(TEMP17-TEMP18)
                  D(JG+J)=S6*(TEMP15+TEMP16)+C6*(TEMP17-TEMP18)
                  TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
                  TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
                  TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
                  TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
                  C(JC+J)=C2*(TEMP19-TEMP20)-S2*(TEMP21+TEMP22)
                  D(JC+J)=S2*(TEMP19-TEMP20)+C2*(TEMP21+TEMP22)
                  C(JF+J)=C5*(TEMP19+TEMP20)-S5*(TEMP21-TEMP22)
                  D(JF+J)=S5*(TEMP19+TEMP20)+C5*(TEMP21-TEMP22)
                  TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
                  TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
                  TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
                  TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
                  C(JD+J)=C3*(TEMP23-TEMP24)-S3*(TEMP25+TEMP26)
                  D(JD+J)=S3*(TEMP23-TEMP24)+C3*(TEMP25+TEMP26)
                  C(JE+J)=C4*(TEMP23+TEMP24)-S4*(TEMP25-TEMP26)
                  D(JE+J)=S4*(TEMP23+TEMP24)+C4*(TEMP25-TEMP26)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.7) THEN 
! Coding for factor 8:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         IH=IG+IINK
         JH=JG+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               TEMP15=A(IH+I)
               TEMP16=B(IH+I)
               C(JA+J)=((TEMP1+TEMP9)+(TEMP3+TEMP11))+((TEMP5+TEMP13)+(TEMP7+TEMP15))
               D(JA+J)=((TEMP2+TEMP10)+(TEMP4+TEMP12))+((TEMP6+TEMP14)+(TEMP8+TEMP16))
               C(JE+J)=((TEMP1+TEMP9)-(TEMP3+TEMP11))+((TEMP5+TEMP13)-(TEMP7+TEMP15))
               D(JE+J)=((TEMP2+TEMP10)-(TEMP4+TEMP12))+((TEMP6+TEMP14)-(TEMP8+TEMP16))
               C(JC+J)=((TEMP1+TEMP9)-(TEMP5+TEMP13))-((TEMP4+TEMP12)-(TEMP8+TEMP16))
               D(JC+J)=((TEMP2+TEMP10)-(TEMP6+TEMP14))+((TEMP3+TEMP11)-(TEMP7+TEMP15))
               C(JG+J)=((TEMP1+TEMP9)-(TEMP5+TEMP13))+((TEMP4+TEMP12)-(TEMP8+TEMP16))
               D(JG+J)=((TEMP2+TEMP10)-(TEMP6+TEMP14))-((TEMP3+TEMP11)-(TEMP7+TEMP15))
               TEMP17=SQR2H*((TEMP3-TEMP11)-(TEMP7-TEMP15))
               TEMP18=SQR2H*((TEMP4-TEMP12)+(TEMP8-TEMP16))
               TEMP19=SQR2H*((TEMP4-TEMP12)-(TEMP8-TEMP16))
               TEMP20=SQR2H*((TEMP3-TEMP11)+(TEMP7-TEMP15))
               C(JB+J)=((TEMP1-TEMP9)-(TEMP6-TEMP14))+TEMP17-TEMP18
               D(JB+J)=((TEMP2-TEMP10)+(TEMP5-TEMP13))+TEMP19+TEMP20
               C(JF+J)=((TEMP1-TEMP9)-(TEMP6-TEMP14))-TEMP17+TEMP18
               D(JF+J)=((TEMP2-TEMP10)+(TEMP5-TEMP13))-TEMP19-TEMP20
               C(JD+J)=((TEMP1-TEMP9)+(TEMP6-TEMP14))-TEMP17-TEMP18
               D(JD+J)=((TEMP2-TEMP10)-(TEMP5-TEMP13))-TEMP19+TEMP20
               C(JH+J)=((TEMP1-TEMP9)+(TEMP6-TEMP14))+TEMP17+TEMP18
               D(JH+J)=((TEMP2-TEMP10)-(TEMP5-TEMP13))+TEMP19-TEMP20
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            KH=KG+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            C7=TRIGS(KH+1)
            S7=TRIGS(KH+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
               DO IJK=1,LOT
                  WRITE(*,*) 'Big sorry from FPASSM: factor 8 only coded as last factor'
                  STOP       
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      RETURN
      END SUBROUTINE FPASSM

      SUBROUTINE FPASSM_1_1(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! FPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************
      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)

      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) THEN 
         RETURN
      ENDIF
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
! Coding for factor 2:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               C(JA+J)=TEMP1+TEMP3
               D(JA+J)=TEMP2+TEMP4
               C(JB+J)=TEMP1-TEMP3
               D(JB+J)=TEMP2-TEMP4
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  C(JA+J)=TEMP1+TEMP3
                  D(JA+J)=TEMP2+TEMP4
                  C(JB+J)=C1*(TEMP1-TEMP3)-S1*(TEMP2-TEMP4)
                  D(JB+J)=S1*(TEMP1-TEMP3)+C1*(TEMP2-TEMP4)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
! Coding for factor 3:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP5)
               D(JA+J)=TEMP2+(TEMP4+TEMP6)
               TEMP7=SIN60*(TEMP3-TEMP5)
               TEMP8=SIN60*(TEMP4-TEMP6)
               C(JB+J)=TEMP1-0.5_q*(TEMP3+TEMP5)-TEMP8
               C(JC+J)=TEMP1-0.5_q*(TEMP3+TEMP5)+TEMP8
               D(JB+J)=TEMP2-0.5_q*(TEMP4+TEMP6)+TEMP7
               D(JC+J)=TEMP2-0.5_q*(TEMP4+TEMP6)-TEMP7
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP5)
                  D(JA+J)=TEMP2+(TEMP4+TEMP6)
                  TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
                  TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
                  TEMP9=SIN60*(TEMP4-TEMP6)
                  TEMP10=SIN60*(TEMP3-TEMP5)
                  C(JB+J)=C1*(TEMP7-TEMP9)-S1*(TEMP8+TEMP10)
                  D(JB+J)=S1*(TEMP7-TEMP9)+C1*(TEMP8+TEMP10)
                  C(JC+J)=C2*(TEMP7+TEMP9)-S2*(TEMP8-TEMP10)
                  D(JC+J)=S2*(TEMP7+TEMP9)+C2*(TEMP8-TEMP10)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
! Coding for factor 4:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
               C(JC+J)=(TEMP1+TEMP5)-(TEMP3+TEMP7)
               D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
               D(JC+J)=(TEMP2+TEMP6)-(TEMP4+TEMP8)
               C(JB+J)=(TEMP1-TEMP5)-(TEMP4-TEMP8)
               C(JD+J)=(TEMP1-TEMP5)+(TEMP4-TEMP8)
               D(JB+J)=(TEMP2-TEMP6)+(TEMP3-TEMP7)
               D(JD+J)=(TEMP2-TEMP6)-(TEMP3-TEMP7)
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
                  D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
                  TEMP9=(TEMP1+TEMP5)-(TEMP3+TEMP7)
                  TEMP10=(TEMP2+TEMP6)-(TEMP4+TEMP8)
                  C(JC+J)=C2*TEMP9-S2*TEMP10
                  D(JC+J)=S2*TEMP9+C2*TEMP10
                  TEMP11=(TEMP1-TEMP5)-(TEMP4-TEMP8)
                  TEMP12=(TEMP2-TEMP6)+(TEMP3-TEMP7)
                  C(JB+J)=C1*TEMP11-S1*TEMP12
                  D(JB+J)=S1*TEMP11+C1*TEMP12
                  TEMP13=(TEMP1-TEMP5)+(TEMP4-TEMP8)
                  TEMP14=(TEMP2-TEMP6)-(TEMP3-TEMP7)
                  C(JD+J)=C3*TEMP13-S3*TEMP14
                  D(JD+J)=S3*TEMP13+C3*TEMP14
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN         
      ENDIF
      
      IF (IGO.EQ.4) THEN
! Coding for factor 5:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
               D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
               TEMP11=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
               TEMP12=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
               C(JB+J)=TEMP11-TEMP12
               C(JE+J)=TEMP11+TEMP12
               TEMP13=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
               TEMP14=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
               D(JB+J)=TEMP13+TEMP14
               D(JE+J)=TEMP13-TEMP14
               TEMP15=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
               TEMP16=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
               C(JC+J)=TEMP15-TEMP16
               C(JD+J)=TEMP15+TEMP16
               TEMP17=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
               TEMP18=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
               D(JC+J)=TEMP17+TEMP18
               D(JD+J)=TEMP17-TEMP18
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
                  D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
                  TEMP11=(TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7))-(SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8))
                  TEMP12=(TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8))+(SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7))
                  C(JB+J)=C1*TEMP11-S1*TEMP12
                  D(JB+J)=S1*TEMP11+C1*TEMP12
                  TEMP13=(TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7))+(SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8))
                  TEMP14=(TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8))-(SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7))
                  C(JE+J)=C4*TEMP13-S4*TEMP14
                  D(JE+J)=S4*TEMP13+C4*TEMP14
                  TEMP15=(TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7))-(SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8))
                  TEMP16=(TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8))+(SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7))
                  C(JC+J)=C2*TEMP15-S2*TEMP16
                  D(JC+J)=S2*TEMP15+C2*TEMP16
                  TEMP17=(TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7))+(SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8))
                  TEMP18=(TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8))-(SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7))
                  C(JD+J)=C3*TEMP17-S3*TEMP18
                  D(JD+J)=S3*TEMP17+C3*TEMP18
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
         
      IF (IGO.EQ.5) THEN
! Coding for factor 6:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
               C(JD+J)=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
               D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
               D(JD+J)=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP13=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))
               TEMP14=SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
               C(JB+J)=TEMP13-TEMP14
               C(JF+J)=TEMP13+TEMP14
               TEMP15=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP16=SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
               D(JB+J)=TEMP15+TEMP16
               D(JF+J)=TEMP15-TEMP16
               TEMP17=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))
               TEMP18=SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
               C(JC+J)=TEMP17-TEMP18
               C(JE+J)=TEMP17+TEMP18
               TEMP19=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))
               TEMP20=SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
               D(JC+J)=TEMP19+TEMP20
               D(JE+J)=TEMP19-TEMP20
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
                  D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
                  C(JD+J)=C3*((TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9)))-S3*((TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10)))
                  D(JD+J)=S3*((TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9)))+C3*((TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10)))
                  TEMP13=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP14=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JB+J)=C1*TEMP13-S1*TEMP14
                  D(JB+J)=S1*TEMP13+C1*TEMP14
                  TEMP15=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP16=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JF+J)=C5*TEMP15-S5*TEMP16
                  D(JF+J)=S5*TEMP15+C5*TEMP16
                  TEMP17=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP18=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JC+J)=C2*TEMP17-S2*TEMP18
                  D(JC+J)=S2*TEMP17+C2*TEMP18
                  TEMP19=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP20=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JE+J)=C4*TEMP19-S4*TEMP20
                  D(JE+J)=S4*TEMP19+C4*TEMP20
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
! Coding for factor 7:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
               D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
               TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
               TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
               C(JB+J)=TEMP15-TEMP16
               C(JG+J)=TEMP15+TEMP16
               TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
               TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
               D(JB+J)=TEMP17+TEMP18
               D(JG+J)=TEMP17-TEMP18
               TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
               TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
               C(JC+J)=TEMP19-TEMP20
               C(JF+J)=TEMP19+TEMP20
               TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
               TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
               D(JC+J)=TEMP21+TEMP22
               D(JF+J)=TEMP21-TEMP22
               TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
               TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
               C(JD+J)=TEMP23-TEMP24
               C(JE+J)=TEMP23+TEMP24
               TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
               TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
               D(JD+J)=TEMP25+TEMP26
               D(JE+J)=TEMP25-TEMP26
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  TEMP13=A(IG+I)
                  TEMP14=B(IG+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
                  D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
                  TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
                  TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
                  TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
                  TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
                  C(JB+J)=C1*(TEMP15-TEMP16)-S1*(TEMP17+TEMP18)
                  D(JB+J)=S1*(TEMP15-TEMP16)+C1*(TEMP17+TEMP18)
                  C(JG+J)=C6*(TEMP15+TEMP16)-S6*(TEMP17-TEMP18)
                  D(JG+J)=S6*(TEMP15+TEMP16)+C6*(TEMP17-TEMP18)
                  TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
                  TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
                  TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
                  TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
                  C(JC+J)=C2*(TEMP19-TEMP20)-S2*(TEMP21+TEMP22)
                  D(JC+J)=S2*(TEMP19-TEMP20)+C2*(TEMP21+TEMP22)
                  C(JF+J)=C5*(TEMP19+TEMP20)-S5*(TEMP21-TEMP22)
                  D(JF+J)=S5*(TEMP19+TEMP20)+C5*(TEMP21-TEMP22)
                  TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
                  TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
                  TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
                  TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
                  C(JD+J)=C3*(TEMP23-TEMP24)-S3*(TEMP25+TEMP26)
                  D(JD+J)=S3*(TEMP23-TEMP24)+C3*(TEMP25+TEMP26)
                  C(JE+J)=C4*(TEMP23+TEMP24)-S4*(TEMP25-TEMP26)
                  D(JE+J)=S4*(TEMP23+TEMP24)+C4*(TEMP25-TEMP26)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.7) THEN 
! Coding for factor 8:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         IH=IG+IINK
         JH=JG+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               TEMP15=A(IH+I)
               TEMP16=B(IH+I)
               C(JA+J)=((TEMP1+TEMP9)+(TEMP3+TEMP11))+((TEMP5+TEMP13)+(TEMP7+TEMP15))
               D(JA+J)=((TEMP2+TEMP10)+(TEMP4+TEMP12))+((TEMP6+TEMP14)+(TEMP8+TEMP16))
               C(JE+J)=((TEMP1+TEMP9)-(TEMP3+TEMP11))+((TEMP5+TEMP13)-(TEMP7+TEMP15))
               D(JE+J)=((TEMP2+TEMP10)-(TEMP4+TEMP12))+((TEMP6+TEMP14)-(TEMP8+TEMP16))
               C(JC+J)=((TEMP1+TEMP9)-(TEMP5+TEMP13))-((TEMP4+TEMP12)-(TEMP8+TEMP16))
               D(JC+J)=((TEMP2+TEMP10)-(TEMP6+TEMP14))+((TEMP3+TEMP11)-(TEMP7+TEMP15))
               C(JG+J)=((TEMP1+TEMP9)-(TEMP5+TEMP13))+((TEMP4+TEMP12)-(TEMP8+TEMP16))
               D(JG+J)=((TEMP2+TEMP10)-(TEMP6+TEMP14))-((TEMP3+TEMP11)-(TEMP7+TEMP15))
               TEMP17=SQR2H*((TEMP3-TEMP11)-(TEMP7-TEMP15))
               TEMP18=SQR2H*((TEMP4-TEMP12)+(TEMP8-TEMP16))
               TEMP19=SQR2H*((TEMP4-TEMP12)-(TEMP8-TEMP16))
               TEMP20=SQR2H*((TEMP3-TEMP11)+(TEMP7-TEMP15))
               C(JB+J)=((TEMP1-TEMP9)-(TEMP6-TEMP14))+TEMP17-TEMP18
               D(JB+J)=((TEMP2-TEMP10)+(TEMP5-TEMP13))+TEMP19+TEMP20
               C(JF+J)=((TEMP1-TEMP9)-(TEMP6-TEMP14))-TEMP17+TEMP18
               D(JF+J)=((TEMP2-TEMP10)+(TEMP5-TEMP13))-TEMP19-TEMP20
               C(JD+J)=((TEMP1-TEMP9)+(TEMP6-TEMP14))-TEMP17-TEMP18
               D(JD+J)=((TEMP2-TEMP10)-(TEMP5-TEMP13))-TEMP19+TEMP20
               C(JH+J)=((TEMP1-TEMP9)+(TEMP6-TEMP14))+TEMP17+TEMP18
               D(JH+J)=((TEMP2-TEMP10)-(TEMP5-TEMP13))+TEMP19-TEMP20
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            KH=KG+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            C7=TRIGS(KH+1)
            S7=TRIGS(KH+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
               DO IJK=1,LOT
                  WRITE(*,*) 'Big sorry from FPASSM: factor 8 only coded as last factor'
                  STOP       
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      RETURN
      END SUBROUTINE FPASSM_1_1

      SUBROUTINE IPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! IPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Originally VPASSM did not allow to perform the inverse FFT and so    *
! one had to rearrange the data externally in some way that the FFT    *
! could be done with the original phase factors. This required some    *
! additional calculation time and was hence not optimal. IPASSM is     *
! now the 'VPASSM'-version with conjugate phase factors needed for     *
! the inverse transform which can now be performed at same perfomance  *
! rates as the forward transform!                                      *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************
      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) THEN
         RETURN
      ENDIF
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
! Coding for factor 2:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               C(JA+J)=TEMP1+TEMP3
               D(JA+J)=TEMP2+TEMP4
               C(JB+J)=TEMP1-TEMP3
               D(JB+J)=TEMP2-TEMP4
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  C(JA+J)=TEMP1+TEMP3
                  D(JA+J)=TEMP2+TEMP4
                  C(JB+J)=C1*(TEMP1-TEMP3)+S1*(TEMP2-TEMP4)
                  D(JB+J)=C1*(TEMP2-TEMP4)-S1*(TEMP1-TEMP3)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
! Coding for factor 3:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP5)
               D(JA+J)=TEMP2+(TEMP4+TEMP6)
               TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
               TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
               TEMP9=SIN60*(TEMP4-TEMP6)
               TEMP10=SIN60*(TEMP3-TEMP5)
               C(JB+J)=TEMP7+TEMP9
               C(JC+J)=TEMP7-TEMP9
               D(JB+J)=TEMP8-TEMP10
               D(JC+J)=TEMP8+TEMP10
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP5)
                  D(JA+J)=TEMP2+(TEMP4+TEMP6)
                  TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
                  TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
                  TEMP9=SIN60*(TEMP3-TEMP5)
                  TEMP10=SIN60*(TEMP4-TEMP6)
                  C(JB+J)=C1*(TEMP7+TEMP10)+S1*(TEMP8-TEMP9)
                  D(JB+J)=C1*(TEMP8-TEMP9)-S1*(TEMP7+TEMP10)
                  C(JC+J)=C2*(TEMP7-TEMP10)+S2*(TEMP8+TEMP9)
                  D(JC+J)=C2*(TEMP8+TEMP9)-S2*(TEMP7-TEMP10)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
! Coding for factor 4:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
               C(JC+J)=(TEMP1+TEMP5)-(TEMP3+TEMP7)
               D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
               D(JC+J)=(TEMP2+TEMP6)-(TEMP4+TEMP8)
               C(JB+J)=(TEMP1-TEMP5)+(TEMP4-TEMP8)
               C(JD+J)=(TEMP1-TEMP5)-(TEMP4-TEMP8)
               D(JB+J)=(TEMP2-TEMP6)-(TEMP3-TEMP7)
               D(JD+J)=(TEMP2-TEMP6)+(TEMP3-TEMP7)
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
                  D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
                  C(JC+J)=C2*((TEMP1+TEMP5)-(TEMP3+TEMP7))+S2*((TEMP2+TEMP6)-(TEMP4+TEMP8))
                  D(JC+J)=C2*((TEMP2+TEMP6)-(TEMP4+TEMP8))-S2*((TEMP1+TEMP5)-(TEMP3+TEMP7))
                  C(JB+J)=C1*((TEMP1-TEMP5)+(TEMP4-TEMP8))+S1*((TEMP2-TEMP6)-(TEMP3-TEMP7))
                  D(JB+J)=C1*((TEMP2-TEMP6)-(TEMP3-TEMP7))-S1*((TEMP1-TEMP5)+(TEMP4-TEMP8))
                  C(JD+J)=C3*((TEMP1-TEMP5)-(TEMP4-TEMP8))+S3*((TEMP2-TEMP6)+(TEMP3-TEMP7))
                  D(JD+J)=C3*((TEMP2-TEMP6)+(TEMP3-TEMP7))-S3*((TEMP1-TEMP5)-(TEMP4-TEMP8))
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
! Coding for factor 5:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
               D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
               TEMP11=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
               TEMP12=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
               C(JB+J)=TEMP11+TEMP12
               C(JE+J)=TEMP11-TEMP12
               TEMP13=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
               TEMP14=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
               D(JB+J)=TEMP13-TEMP14
               D(JE+J)=TEMP13+TEMP14
               TEMP15=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
               TEMP16=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
               C(JC+J)=TEMP15+TEMP16
               C(JD+J)=TEMP15-TEMP16
               TEMP17=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
               TEMP18=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
               D(JC+J)=TEMP17-TEMP18
               D(JD+J)=TEMP17+TEMP18
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
                  D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
                  TEMP15=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
                  TEMP16=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
                  TEMP17=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
                  TEMP18=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
                  C(JB+J)=C1*(TEMP15+TEMP16)+S1*(TEMP17-TEMP18)
                  D(JB+J)=-S1*(TEMP15+TEMP16)+C1*(TEMP17-TEMP18)
                  C(JE+J)=C4*(TEMP15-TEMP16)+S4*(TEMP17+TEMP18)
                  D(JE+J)=-S4*(TEMP15-TEMP16)+C4*(TEMP17+TEMP18)
                  TEMP19=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
                  TEMP20=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
                  TEMP21=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
                  TEMP22=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
                  C(JC+J)=C2*(TEMP19+TEMP20)+S2*(TEMP21-TEMP22)
                  D(JC+J)=-S2*(TEMP19+TEMP20)+C2*(TEMP21-TEMP22)
                  C(JD+J)=C3*(TEMP19-TEMP20)+S3*(TEMP21+TEMP22)
                  D(JD+J)=-S3*(TEMP19-TEMP20)+C3*(TEMP21+TEMP22)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      
      IF (IGO.EQ.5) THEN
! Coding for factor 6:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
               C(JD+J)=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
               D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
               D(JD+J)=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP13=SIN60*(TEMP3-TEMP11)
               TEMP14=SIN60*(TEMP4-TEMP12)
               TEMP15=SIN60*(TEMP5-TEMP9)
               TEMP16=SIN60*(TEMP6-TEMP10)
               C(JB+J)=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+TEMP14+TEMP16
               C(JF+J)=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-TEMP14-TEMP16
               D(JB+J)=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-TEMP13-TEMP15
               D(JF+J)=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+TEMP13+TEMP15
               C(JC+J)=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+TEMP14-TEMP16
               C(JE+J)=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-TEMP14+TEMP16
               D(JC+J)=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-TEMP13+TEMP15
               D(JE+J)=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+TEMP13-TEMP15
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
                  D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
                  TEMP13=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
                  TEMP14=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
                  C(JD+J)=C3*TEMP13+S3*TEMP14
                  D(JD+J)=-S3*TEMP13+C3*TEMP14
                  TEMP15=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP16=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JB+J)=C1*TEMP15+S1*TEMP16
                  D(JB+J)=-S1*TEMP15+C1*TEMP16
                  TEMP17=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP18=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JF+J)=C5*TEMP17+S5*TEMP18
                  D(JF+J)=-S5*TEMP17+C5*TEMP18
                  TEMP19=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP20=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JC+J)=C2*TEMP19+S2*TEMP20
                  D(JC+J)=-S2*TEMP19+C2*TEMP20
                  TEMP21=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP22=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JE+J)=C4*TEMP21+S4*TEMP22
                  D(JE+J)=-S4*TEMP21+C4*TEMP22
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
! Coding for factor 7:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
               D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
               TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
               TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
               C(JB+J)=TEMP15+TEMP16
               C(JG+J)=TEMP15-TEMP16
               TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
               TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
               D(JB+J)=TEMP17-TEMP18
               D(JG+J)=TEMP17+TEMP18
               TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
               TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
               C(JC+J)=TEMP19+TEMP20
               C(JF+J)=TEMP19-TEMP20
               TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
               TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
               D(JC+J)=TEMP21-TEMP22
               D(JF+J)=TEMP21+TEMP22
               TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
               TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
               C(JD+J)=TEMP23+TEMP24
               C(JE+J)=TEMP23-TEMP24
               TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
               TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
               D(JD+J)=TEMP25-TEMP26
               D(JE+J)=TEMP25+TEMP26
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  TEMP13=A(IG+I)
                  TEMP14=B(IG+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
                  D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
                  TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
                  TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
                  TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
                  TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
                  C(JB+J)=C1*(TEMP15+TEMP16)+S1*(TEMP17-TEMP18)
                  D(JB+J)=-S1*(TEMP15+TEMP16)+C1*(TEMP17-TEMP18)
                  C(JG+J)=C6*(TEMP15-TEMP16)+S6*(TEMP17+TEMP18)
                  D(JG+J)=-S6*(TEMP15-TEMP16)+C6*(TEMP17+TEMP18)
                  TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
                  TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
                  TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
                  TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
                  C(JC+J)=C2*(TEMP19+TEMP20)+S2*(TEMP21-TEMP22)
                  D(JC+J)=-S2*(TEMP19+TEMP20)+C2*(TEMP21-TEMP22)
                  C(JF+J)=C5*(TEMP19-TEMP20)+S5*(TEMP21+TEMP22)
                  D(JF+J)=-S5*(TEMP19-TEMP20)+C5*(TEMP21+TEMP22)
                  TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
                  TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
                  TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
                  TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
                  C(JD+J)=C3*(TEMP23+TEMP24)+S3*(TEMP25-TEMP26)
                  D(JD+J)=-S3*(TEMP23+TEMP24)+C3*(TEMP25-TEMP26)
                  C(JE+J)=C4*(TEMP23-TEMP24)+S4*(TEMP25+TEMP26)
                  D(JE+J)=-S4*(TEMP23-TEMP24)+C4*(TEMP25+TEMP26)
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      
      IF (IGO.EQ.7) THEN
! Coding for factor 8:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         IH=IG+IINK
         JH=JG+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT               
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               TEMP15=A(IH+I)
               TEMP16=B(IH+I)
               TEMP17=TEMP1+TEMP9
               TEMP18=TEMP2+TEMP10
               TEMP19=TEMP3+TEMP11
               TEMP20=TEMP4+TEMP12
               TEMP21=TEMP5+TEMP13
               TEMP22=TEMP6+TEMP14
               TEMP23=TEMP7+TEMP15
               TEMP24=TEMP8+TEMP16
               C(JA+J)=(TEMP17+TEMP19)+(TEMP21+TEMP23)
               D(JA+J)=(TEMP18+TEMP20)+(TEMP22+TEMP24)
               C(JE+J)=(TEMP17-TEMP19)+(TEMP21-TEMP23)
               D(JE+J)=(TEMP18-TEMP20)+(TEMP22-TEMP24)
               C(JC+J)=(TEMP17-TEMP21)+(TEMP20-TEMP24)
               D(JC+J)=(TEMP18-TEMP22)-(TEMP19-TEMP23)
               C(JG+J)=(TEMP17-TEMP21)-(TEMP20-TEMP24)
               D(JG+J)=(TEMP18-TEMP22)+(TEMP19-TEMP23)
               TEMP25=TEMP1-TEMP9
               TEMP26=TEMP2-TEMP10
               TEMP27=TEMP6-TEMP14
               TEMP28=TEMP5-TEMP13
               TEMP29=(TEMP3-TEMP11)-(TEMP7-TEMP15)
               TEMP30=(TEMP4-TEMP12)-(TEMP8-TEMP16)
               TEMP31=(TEMP4-TEMP12)+(TEMP8-TEMP16)
               TEMP32=(TEMP3-TEMP11)+(TEMP7-TEMP15)
               C(JB+J)=(TEMP25+TEMP27)+SQR2H*(TEMP29+TEMP31)
               D(JB+J)=(TEMP26-TEMP28)+SQR2H*(TEMP30-TEMP32)
               C(JF+J)=(TEMP25+TEMP27)-SQR2H*(TEMP29+TEMP31)
               D(JF+J)=(TEMP26-TEMP28)-SQR2H*(TEMP30-TEMP32)
               C(JD+J)=(TEMP25-TEMP27)-SQR2H*(TEMP29-TEMP31)
               D(JD+J)=(TEMP26+TEMP28)-SQR2H*(TEMP30+TEMP32)
               C(JH+J)=(TEMP25-TEMP27)+SQR2H*(TEMP29-TEMP31)
               D(JH+J)=(TEMP26+TEMP28)+SQR2H*(TEMP30+TEMP32)
               I=I+INC3
               J=J+INC4
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            KH=KG+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            C7=TRIGS(KH+1)
            S7=TRIGS(KH+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
               DO IJK=1,LOT
                  WRITE(*,*) 'Big sorry from IPASSM: factor 8 only coded as last factor'
                  STOP
                  I=I+INC3
                  J=J+INC4
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      RETURN
      END SUBROUTINE IPASSM

      SUBROUTINE IPASSM_1_1(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! IPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! Originally VPASSM did not allow to perform the inverse FFT and so    *
! one had to rearrange the data externally in some way that the FFT    *
! could be done with the original phase factors. This required some    *
! additional calculation time and was hence not optimal. IPASSM is     *
! now the 'VPASSM'-version with conjugate phase factors needed for     *
! the inverse transform which can now be performed at same perfomance  *
! rates as the forward transform!                                      *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   IFAC is the current factor of N for which the pass is performed    *
!   LA is the product of all previous factors of all previous passes   *
!                                                                      *
!***********************************************************************
      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO>7) THEN
         RETURN
      ENDIF
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
! Coding for factor 2:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               C(JA+J)=TEMP1+TEMP3
               D(JA+J)=TEMP2+TEMP4
               C(JB+J)=TEMP1-TEMP3
               D(JB+J)=TEMP2-TEMP4
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  C(JA+J)=TEMP1+TEMP3
                  D(JA+J)=TEMP2+TEMP4
                  C(JB+J)=C1*(TEMP1-TEMP3)+S1*(TEMP2-TEMP4)
                  D(JB+J)=C1*(TEMP2-TEMP4)-S1*(TEMP1-TEMP3)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
! Coding for factor 3:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP5)
               D(JA+J)=TEMP2+(TEMP4+TEMP6)
               TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
               TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
               TEMP9=SIN60*(TEMP4-TEMP6)
               TEMP10=SIN60*(TEMP3-TEMP5)
               C(JB+J)=TEMP7+TEMP9
               C(JC+J)=TEMP7-TEMP9
               D(JB+J)=TEMP8-TEMP10
               D(JC+J)=TEMP8+TEMP10
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP5)
                  D(JA+J)=TEMP2+(TEMP4+TEMP6)
                  TEMP7=TEMP1-0.5_q*(TEMP3+TEMP5)
                  TEMP8=TEMP2-0.5_q*(TEMP4+TEMP6)
                  TEMP9=SIN60*(TEMP3-TEMP5)
                  TEMP10=SIN60*(TEMP4-TEMP6)
                  C(JB+J)=C1*(TEMP7+TEMP10)+S1*(TEMP8-TEMP9)
                  D(JB+J)=C1*(TEMP8-TEMP9)-S1*(TEMP7+TEMP10)
                  C(JC+J)=C2*(TEMP7-TEMP10)+S2*(TEMP8+TEMP9)
                  D(JC+J)=C2*(TEMP8+TEMP9)-S2*(TEMP7-TEMP10)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
! Coding for factor 4:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
               C(JC+J)=(TEMP1+TEMP5)-(TEMP3+TEMP7)
               D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
               D(JC+J)=(TEMP2+TEMP6)-(TEMP4+TEMP8)
               C(JB+J)=(TEMP1-TEMP5)+(TEMP4-TEMP8)
               C(JD+J)=(TEMP1-TEMP5)-(TEMP4-TEMP8)
               D(JB+J)=(TEMP2-TEMP6)-(TEMP3-TEMP7)
               D(JD+J)=(TEMP2-TEMP6)+(TEMP3-TEMP7)
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  C(JA+J)=(TEMP1+TEMP5)+(TEMP3+TEMP7)
                  D(JA+J)=(TEMP2+TEMP6)+(TEMP4+TEMP8)
                  C(JC+J)=C2*((TEMP1+TEMP5)-(TEMP3+TEMP7))+S2*((TEMP2+TEMP6)-(TEMP4+TEMP8))
                  D(JC+J)=C2*((TEMP2+TEMP6)-(TEMP4+TEMP8))-S2*((TEMP1+TEMP5)-(TEMP3+TEMP7))
                  C(JB+J)=C1*((TEMP1-TEMP5)+(TEMP4-TEMP8))+S1*((TEMP2-TEMP6)-(TEMP3-TEMP7))
                  D(JB+J)=C1*((TEMP2-TEMP6)-(TEMP3-TEMP7))-S1*((TEMP1-TEMP5)+(TEMP4-TEMP8))
                  C(JD+J)=C3*((TEMP1-TEMP5)-(TEMP4-TEMP8))+S3*((TEMP2-TEMP6)+(TEMP3-TEMP7))
                  D(JD+J)=C3*((TEMP2-TEMP6)+(TEMP3-TEMP7))-S3*((TEMP1-TEMP5)-(TEMP4-TEMP8))
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
! Coding for factor 5:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
               D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
               TEMP11=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
               TEMP12=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
               C(JB+J)=TEMP11+TEMP12
               C(JE+J)=TEMP11-TEMP12
               TEMP13=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
               TEMP14=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
               D(JB+J)=TEMP13-TEMP14
               D(JE+J)=TEMP13+TEMP14
               TEMP15=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
               TEMP16=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
               C(JC+J)=TEMP15+TEMP16
               C(JD+J)=TEMP15-TEMP16
               TEMP17=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
               TEMP18=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
               D(JC+J)=TEMP17-TEMP18
               D(JD+J)=TEMP17+TEMP18
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP9)+(TEMP5+TEMP7)
                  D(JA+J)=TEMP2+(TEMP4+TEMP10)+(TEMP6+TEMP8)
                  TEMP15=TEMP1+COS72*(TEMP3+TEMP9)-COS36*(TEMP5+TEMP7)
                  TEMP16=SIN72*(TEMP4-TEMP10)+SIN36*(TEMP6-TEMP8)
                  TEMP17=TEMP2+COS72*(TEMP4+TEMP10)-COS36*(TEMP6+TEMP8)
                  TEMP18=SIN72*(TEMP3-TEMP9)+SIN36*(TEMP5-TEMP7)
                  C(JB+J)=C1*(TEMP15+TEMP16)+S1*(TEMP17-TEMP18)
                  D(JB+J)=-S1*(TEMP15+TEMP16)+C1*(TEMP17-TEMP18)
                  C(JE+J)=C4*(TEMP15-TEMP16)+S4*(TEMP17+TEMP18)
                  D(JE+J)=-S4*(TEMP15-TEMP16)+C4*(TEMP17+TEMP18)
                  TEMP19=TEMP1-COS36*(TEMP3+TEMP9)+COS72*(TEMP5+TEMP7)
                  TEMP20=SIN36*(TEMP4-TEMP10)-SIN72*(TEMP6-TEMP8)
                  TEMP21=TEMP2-COS36*(TEMP4+TEMP10)+COS72*(TEMP6+TEMP8)
                  TEMP22=SIN36*(TEMP3-TEMP9)-SIN72*(TEMP5-TEMP7)
                  C(JC+J)=C2*(TEMP19+TEMP20)+S2*(TEMP21-TEMP22)
                  D(JC+J)=-S2*(TEMP19+TEMP20)+C2*(TEMP21-TEMP22)
                  C(JD+J)=C3*(TEMP19-TEMP20)+S3*(TEMP21+TEMP22)
                  D(JD+J)=-S3*(TEMP19-TEMP20)+C3*(TEMP21+TEMP22)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      
      IF (IGO.EQ.5) THEN
! Coding for factor 6:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
               C(JD+J)=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
               D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
               D(JD+J)=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
               TEMP13=SIN60*(TEMP3-TEMP11)
               TEMP14=SIN60*(TEMP4-TEMP12)
               TEMP15=SIN60*(TEMP5-TEMP9)
               TEMP16=SIN60*(TEMP6-TEMP10)
               C(JB+J)=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+TEMP14+TEMP16
               C(JF+J)=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-TEMP14-TEMP16
               D(JB+J)=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-TEMP13-TEMP15
               D(JF+J)=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+TEMP13+TEMP15
               C(JC+J)=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+TEMP14-TEMP16
               C(JE+J)=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-TEMP14+TEMP16
               D(JC+J)=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-TEMP13+TEMP15
               D(JE+J)=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+TEMP13-TEMP15
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  C(JA+J)=(TEMP1+TEMP7)+((TEMP3+TEMP11)+(TEMP5+TEMP9))
                  D(JA+J)=(TEMP2+TEMP8)+((TEMP4+TEMP12)+(TEMP6+TEMP10))
                  TEMP13=(TEMP1-TEMP7)-((TEMP3+TEMP11)-(TEMP5+TEMP9))
                  TEMP14=(TEMP2-TEMP8)-((TEMP4+TEMP12)-(TEMP6+TEMP10))
                  C(JD+J)=C3*TEMP13+S3*TEMP14
                  D(JD+J)=-S3*TEMP13+C3*TEMP14
                  TEMP15=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP16=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JB+J)=C1*TEMP15+S1*TEMP16
                  D(JB+J)=-S1*TEMP15+C1*TEMP16
                  TEMP17=(TEMP1-TEMP7)+0.5_q*((TEMP3+TEMP11)-(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)+(TEMP6-TEMP10))
                  TEMP18=(TEMP2-TEMP8)+0.5_q*((TEMP4+TEMP12)-(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)+(TEMP5-TEMP9))
                  C(JF+J)=C5*TEMP17+S5*TEMP18
                  D(JF+J)=-S5*TEMP17+C5*TEMP18
                  TEMP19=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))+SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP20=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))-SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JC+J)=C2*TEMP19+S2*TEMP20
                  D(JC+J)=-S2*TEMP19+C2*TEMP20
                  TEMP21=(TEMP1+TEMP7)-0.5_q*((TEMP3+TEMP11)+(TEMP5+TEMP9))-SIN60*((TEMP4-TEMP12)-(TEMP6-TEMP10))
                  TEMP22=(TEMP2+TEMP8)-0.5_q*((TEMP4+TEMP12)+(TEMP6+TEMP10))+SIN60*((TEMP3-TEMP11)-(TEMP5-TEMP9))
                  C(JE+J)=C4*TEMP21+S4*TEMP22
                  D(JE+J)=-S4*TEMP21+C4*TEMP22
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
! Coding for factor 7:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
               D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
               TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
               TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
               C(JB+J)=TEMP15+TEMP16
               C(JG+J)=TEMP15-TEMP16
               TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
               TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
               D(JB+J)=TEMP17-TEMP18
               D(JG+J)=TEMP17+TEMP18
               TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
               TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
               C(JC+J)=TEMP19+TEMP20
               C(JF+J)=TEMP19-TEMP20
               TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
               TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
               D(JC+J)=TEMP21-TEMP22
               D(JF+J)=TEMP21+TEMP22
               TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
               TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
               C(JD+J)=TEMP23+TEMP24
               C(JE+J)=TEMP23-TEMP24
               TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
               TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
               D(JD+J)=TEMP25-TEMP26
               D(JE+J)=TEMP25+TEMP26
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
               DO IJK=1,LOT
                  TEMP1=A(IA+I)
                  TEMP2=B(IA+I)
                  TEMP3=A(IB+I)
                  TEMP4=B(IB+I)
                  TEMP5=A(IC+I)
                  TEMP6=B(IC+I)
                  TEMP7=A(ID+I)
                  TEMP8=B(ID+I)
                  TEMP9=A(IE+I)
                  TEMP10=B(IE+I)
                  TEMP11=A(IF+I)
                  TEMP12=B(IF+I)
                  TEMP13=A(IG+I)
                  TEMP14=B(IG+I)
                  C(JA+J)=TEMP1+(TEMP3+TEMP13)+(TEMP5+TEMP11)+(TEMP7+TEMP9)
                  D(JA+J)=TEMP2+(TEMP4+TEMP14)+(TEMP6+TEMP12)+(TEMP8+TEMP10)
                  TEMP15=TEMP1+CPIS2*(TEMP3+TEMP13)-CPIS3*(TEMP5+TEMP11)-CPIS1*(TEMP7+TEMP9)
                  TEMP16=SPIS2*(TEMP4-TEMP14)+SPIS3*(TEMP6-TEMP12)+SPIS1*(TEMP8-TEMP10)
                  TEMP17=TEMP2+CPIS2*(TEMP4+TEMP14)-CPIS3*(TEMP6+TEMP12)-CPIS1*(TEMP8+TEMP10)
                  TEMP18=SPIS2*(TEMP3-TEMP13)+SPIS3*(TEMP5-TEMP11)+SPIS1*(TEMP7-TEMP9)
                  C(JB+J)=C1*(TEMP15+TEMP16)+S1*(TEMP17-TEMP18)
                  D(JB+J)=-S1*(TEMP15+TEMP16)+C1*(TEMP17-TEMP18)
                  C(JG+J)=C6*(TEMP15-TEMP16)+S6*(TEMP17+TEMP18)
                  D(JG+J)=-S6*(TEMP15-TEMP16)+C6*(TEMP17+TEMP18)
                  TEMP19=TEMP1-CPIS3*(TEMP3+TEMP13)-CPIS1*(TEMP5+TEMP11)+CPIS2*(TEMP7+TEMP9)
                  TEMP20=SPIS3*(TEMP4-TEMP14)-SPIS1*(TEMP6-TEMP12)-SPIS2*(TEMP8-TEMP10)
                  TEMP21=TEMP2-CPIS3*(TEMP4+TEMP14)-CPIS1*(TEMP6+TEMP12)+CPIS2*(TEMP8+TEMP10)
                  TEMP22=SPIS3*(TEMP3-TEMP13)-SPIS1*(TEMP5-TEMP11)-SPIS2*(TEMP7-TEMP9)
                  C(JC+J)=C2*(TEMP19+TEMP20)+S2*(TEMP21-TEMP22)
                  D(JC+J)=-S2*(TEMP19+TEMP20)+C2*(TEMP21-TEMP22)
                  C(JF+J)=C5*(TEMP19-TEMP20)+S5*(TEMP21+TEMP22)
                  D(JF+J)=-S5*(TEMP19-TEMP20)+C5*(TEMP21+TEMP22)
                  TEMP23=TEMP1-CPIS1*(TEMP3+TEMP13)+CPIS2*(TEMP5+TEMP11)-CPIS3*(TEMP7+TEMP9)
                  TEMP24=SPIS1*(TEMP4-TEMP14)-SPIS2*(TEMP6-TEMP12)+SPIS3*(TEMP8-TEMP10)
                  TEMP25=TEMP2-CPIS1*(TEMP4+TEMP14)+CPIS2*(TEMP6+TEMP12)-CPIS3*(TEMP8+TEMP10)
                  TEMP26=SPIS1*(TEMP3-TEMP13)-SPIS2*(TEMP5-TEMP11)+SPIS3*(TEMP7-TEMP9)
                  C(JD+J)=C3*(TEMP23+TEMP24)+S3*(TEMP25-TEMP26)
                  D(JD+J)=-S3*(TEMP23+TEMP24)+C3*(TEMP25-TEMP26)
                  C(JE+J)=C4*(TEMP23-TEMP24)+S4*(TEMP25+TEMP26)
                  D(JE+J)=-S4*(TEMP23-TEMP24)+C4*(TEMP25+TEMP26)
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      
      IF (IGO.EQ.7) THEN
! Coding for factor 8:
         IA=1
         JA=1
         IB=IA+IINK
         JB=JA+JINK
         IC=IB+IINK
         JC=JB+JINK
         ID=IC+IINK
         JD=JC+JINK
         IE=ID+IINK
         JE=JD+JINK
         IF=IE+IINK
         JF=JE+JINK
         IG=IF+IINK
         JG=JF+JINK
         IH=IG+IINK
         JH=JG+JINK
         DO L=1,LA
            I=IBASE
            J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
            DO IJK=1,LOT               
               TEMP1=A(IA+I)
               TEMP2=B(IA+I)
               TEMP3=A(IB+I)
               TEMP4=B(IB+I)
               TEMP5=A(IC+I)
               TEMP6=B(IC+I)
               TEMP7=A(ID+I)
               TEMP8=B(ID+I)
               TEMP9=A(IE+I)
               TEMP10=B(IE+I)
               TEMP11=A(IF+I)
               TEMP12=B(IF+I)
               TEMP13=A(IG+I)
               TEMP14=B(IG+I)
               TEMP15=A(IH+I)
               TEMP16=B(IH+I)
               TEMP17=TEMP1+TEMP9
               TEMP18=TEMP2+TEMP10
               TEMP19=TEMP3+TEMP11
               TEMP20=TEMP4+TEMP12
               TEMP21=TEMP5+TEMP13
               TEMP22=TEMP6+TEMP14
               TEMP23=TEMP7+TEMP15
               TEMP24=TEMP8+TEMP16
               C(JA+J)=(TEMP17+TEMP19)+(TEMP21+TEMP23)
               D(JA+J)=(TEMP18+TEMP20)+(TEMP22+TEMP24)
               C(JE+J)=(TEMP17-TEMP19)+(TEMP21-TEMP23)
               D(JE+J)=(TEMP18-TEMP20)+(TEMP22-TEMP24)
               C(JC+J)=(TEMP17-TEMP21)+(TEMP20-TEMP24)
               D(JC+J)=(TEMP18-TEMP22)-(TEMP19-TEMP23)
               C(JG+J)=(TEMP17-TEMP21)-(TEMP20-TEMP24)
               D(JG+J)=(TEMP18-TEMP22)+(TEMP19-TEMP23)
               TEMP25=TEMP1-TEMP9
               TEMP26=TEMP2-TEMP10
               TEMP27=TEMP6-TEMP14
               TEMP28=TEMP5-TEMP13
               TEMP29=(TEMP3-TEMP11)-(TEMP7-TEMP15)
               TEMP30=(TEMP4-TEMP12)-(TEMP8-TEMP16)
               TEMP31=(TEMP4-TEMP12)+(TEMP8-TEMP16)
               TEMP32=(TEMP3-TEMP11)+(TEMP7-TEMP15)
               C(JB+J)=(TEMP25+TEMP27)+SQR2H*(TEMP29+TEMP31)
               D(JB+J)=(TEMP26-TEMP28)+SQR2H*(TEMP30-TEMP32)
               C(JF+J)=(TEMP25+TEMP27)-SQR2H*(TEMP29+TEMP31)
               D(JF+J)=(TEMP26-TEMP28)-SQR2H*(TEMP30-TEMP32)
               C(JD+J)=(TEMP25-TEMP27)-SQR2H*(TEMP29-TEMP31)
               D(JD+J)=(TEMP26+TEMP28)-SQR2H*(TEMP30+TEMP32)
               C(JH+J)=(TEMP25-TEMP27)+SQR2H*(TEMP29-TEMP31)
               D(JH+J)=(TEMP26+TEMP28)+SQR2H*(TEMP30+TEMP32)
               I=I+1
               J=J+1
            ENDDO
            IBASE=IBASE+INC1
            JBASE=JBASE+INC2
         ENDDO
         IF (LA.EQ.M) THEN
            RETURN
         ENDIF
         LA1=LA+1
         JBASE=JBASE+JUMP
         DO K=LA1,M,LA
            KB=K+K-2
            KC=KB+KB
            KD=KC+KB
            KE=KD+KB
            KF=KE+KB
            KG=KF+KB
            KH=KG+KB
            C1=TRIGS(KB+1)
            S1=TRIGS(KB+2)
            C2=TRIGS(KC+1)
            S2=TRIGS(KC+2)
            C3=TRIGS(KD+1)
            S3=TRIGS(KD+2)
            C4=TRIGS(KE+1)
            S4=TRIGS(KE+2)
            C5=TRIGS(KF+1)
            S5=TRIGS(KF+2)
            C6=TRIGS(KG+1)
            S6=TRIGS(KG+2)
            C7=TRIGS(KH+1)
            S7=TRIGS(KH+2)
            DO L=1,LA
               I=IBASE
               J=JBASE
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
               DO IJK=1,LOT
                  WRITE(*,*) 'Big sorry from IPASSM: factor 8 only coded as last factor'
                  STOP
                  I=I+1
                  J=J+1
               ENDDO
               IBASE=IBASE+INC1
               JBASE=JBASE+INC2
            ENDDO
            JBASE=JBASE+JUMP
         ENDDO
         RETURN
      ENDIF
      RETURN
      END SUBROUTINE IPASSM_1_1

      SUBROUTINE SPASSM(A,B,C,D,INC1,INC2,LOT,N,ISIGN)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! SPASSM is derived from the TEMPERTON/FULKER routine 'VPASSM' which   *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! SPASSM is the special version for 'single factor transformations'.   *
! Variables:                                                           *
!   A is the array containing the real parts of the data vector        *
!   B is the array containing the imaginary parts of the data vector   *
!   C is a work array                                                  *
!   D is a work array                                                  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the addressing increment between data sets in A and B      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   ISIGN gives the type of transformation (>=0: forward, <0: inverse) *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      IGO=N-1
      IF (IGO>7) THEN 
         RETURN
      ENDIF

      IF (ISIGN<0) GOTO 1000
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
!
! Coding for factor 2:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            C(JB+J)=TEMP1-TEMP3
            D(JB+J)=TEMP2-TEMP4
            A(IA+I)=TEMP1+TEMP3
            B(IA+I)=TEMP2+TEMP4
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IB+I)=C(JB+J)
            B(IB+I)=D(JB+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
!
! Coding for factor 3:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=TEMP1+TEMP3
            TEMP6=TEMP2+TEMP4
            TEMP7=SIN60*(TEMP1-TEMP3)
            TEMP8=SIN60*(TEMP2-TEMP4)
            C(JA+J)=TEMP5
            D(JA+J)=TEMP6
            C(JB+J)=0.5_q*TEMP5+TEMP8
            C(JC+J)=0.5_q*TEMP5-TEMP8
            D(JB+J)=0.5_q*TEMP6-TEMP7
            D(JC+J)=0.5_q*TEMP6+TEMP7
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)-C(JB+J)
            A(IC+I)=A(IA+I)-C(JC+J)
            B(IB+I)=B(IA+I)-D(JB+J)
            B(IC+I)=B(IA+I)-D(JC+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
!
! Coding for factor 4:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            TEMP5=A(IC+I)
            TEMP6=B(IC+I)
            TEMP7=A(ID+I)
            TEMP8=B(ID+I)
            C(JA+J)=TEMP1+TEMP5
            C(JB+J)=TEMP3+TEMP7
            D(JA+J)=TEMP2+TEMP6
            D(JB+J)=TEMP4+TEMP8
            C(JC+J)=TEMP1-TEMP5
            C(JD+J)=TEMP4-TEMP8
            D(JC+J)=TEMP2-TEMP6
            D(JD+J)=TEMP3-TEMP7
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=C(JA+J)
            TEMP2=D(JA+J)
            TEMP3=C(JB+J)
            TEMP4=D(JB+J)
            TEMP5=C(JC+J)
            TEMP6=D(JC+J)
            TEMP7=C(JD+J)
            TEMP8=D(JD+J)
            A(IA+I)=TEMP1+TEMP3
            A(IC+I)=TEMP1-TEMP3
            B(IA+I)=TEMP2+TEMP4
            B(IC+I)=TEMP2-TEMP4
            A(IB+I)=TEMP5-TEMP7
            A(ID+I)=TEMP5+TEMP7
            B(IB+I)=TEMP6+TEMP8
            B(ID+I)=TEMP6-TEMP8
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
!
! Coding for factor 5:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            C(JA+J)=(TEMP1+TEMP7)+(TEMP3+TEMP5)
            D(JA+J)=(TEMP2+TEMP8)+(TEMP4+TEMP6)
            TEMP9=COS72*(TEMP1+TEMP7)-COS36*(TEMP3+TEMP5)
            TEMP10=SIN72*(TEMP2-TEMP8)+SIN36*(TEMP4-TEMP6)
            C(JB+J)=TEMP9-TEMP10
            C(JE+J)=TEMP9+TEMP10
            TEMP11=COS72*(TEMP2+TEMP8)-COS36*(TEMP4+TEMP6)
            TEMP12=SIN72*(TEMP1-TEMP7)+SIN36*(TEMP3-TEMP5)
            D(JB+J)=TEMP11+TEMP12
            D(JE+J)=TEMP11-TEMP12
            TEMP13=COS72*(TEMP3+TEMP5)-COS36*(TEMP1+TEMP7)
            TEMP14=SIN36*(TEMP2-TEMP8)-SIN72*(TEMP4-TEMP6)
            C(JC+J)=TEMP13-TEMP14
            C(JD+J)=TEMP13+TEMP14
            TEMP15=COS72*(TEMP4+TEMP6)-COS36*(TEMP2+TEMP8)
            TEMP16=SIN36*(TEMP1-TEMP7)-SIN72*(TEMP3-TEMP5)
            D(JC+J)=TEMP15+TEMP16
            D(JD+J)=TEMP15-TEMP16
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.5) THEN
!
! Coding for factor 6:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            C(JA+J)= TEMP5+((TEMP1+TEMP9)+(TEMP3+TEMP7))
            C(JD+J)=-TEMP5-((TEMP1+TEMP9)-(TEMP3+TEMP7))
            D(JA+J)= TEMP6+((TEMP2+TEMP10)+(TEMP4+TEMP8))
            D(JD+J)=-TEMP6-((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP11=-TEMP5+0.5_q*((TEMP1+TEMP9)-(TEMP3+TEMP7))
            TEMP12=SIN60*((TEMP2-TEMP10)+(TEMP4-TEMP8))
            C(JB+J)=TEMP11-TEMP12
            C(JF+J)=TEMP11+TEMP12
            TEMP13=-TEMP6+0.5_q*((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP14=SIN60*((TEMP1-TEMP9)+(TEMP3-TEMP7))
            D(JB+J)=TEMP13+TEMP14
            D(JF+J)=TEMP13-TEMP14
            TEMP15=TEMP5-0.5_q*((TEMP1+TEMP9)+(TEMP3+TEMP7))
            TEMP16=SIN60*((TEMP2-TEMP10)-(TEMP4-TEMP8))
            C(JC+J)=TEMP15-TEMP16
            C(JE+J)=TEMP15+TEMP16
            TEMP17=TEMP6-0.5_q*((TEMP2+TEMP10)+(TEMP4+TEMP8))
            TEMP18=SIN60*((TEMP1-TEMP9)-(TEMP3-TEMP7))
            D(JC+J)=TEMP17+TEMP18
            D(JE+J)=TEMP17-TEMP18
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(ID+I)=A(IA+I)+C(JD+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
!
! Coding for factor 7:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         IG=IF+INC1
         JG=JF+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            C(JA+J)=(TEMP1+TEMP11)+(TEMP3+TEMP9)+(TEMP5+TEMP7)
            D(JA+J)=(TEMP2+TEMP12)+(TEMP4+TEMP10)+(TEMP6+TEMP8)
            TEMP13=CPIS2*(TEMP1+TEMP11)-CPIS3*(TEMP3+TEMP9)-CPIS1*(TEMP5+TEMP7)
            TEMP14=SPIS2*(TEMP2-TEMP12)+SPIS3*(TEMP4-TEMP10)+SPIS1*(TEMP6-TEMP8)
            C(JB+J)=TEMP13-TEMP14
            C(JG+J)=TEMP13+TEMP14
            TEMP15=CPIS2*(TEMP2+TEMP12)-CPIS3*(TEMP4+TEMP10)-CPIS1*(TEMP6+TEMP8)
            TEMP16=SPIS2*(TEMP1-TEMP11)+SPIS3*(TEMP3-TEMP9)+SPIS1*(TEMP5-TEMP7)
            D(JB+J)=TEMP15+TEMP16
            D(JG+J)=TEMP15-TEMP16
            TEMP17=-CPIS3*(TEMP1+TEMP11)-CPIS1*(TEMP3+TEMP9)+CPIS2*(TEMP5+TEMP7)
            TEMP18=SPIS3*(TEMP2-TEMP12)-SPIS1*(TEMP4-TEMP10)-SPIS2*(TEMP6-TEMP8)
            C(JC+J)=TEMP17-TEMP18
            C(JF+J)=TEMP17+TEMP18
            TEMP19=-CPIS3*(TEMP2+TEMP12)-CPIS1*(TEMP4+TEMP10)+CPIS2*(TEMP6+TEMP8)
            TEMP20=SPIS3*(TEMP1-TEMP11)-SPIS1*(TEMP3-TEMP9)-SPIS2*(TEMP5-TEMP7)
            D(JC+J)=TEMP19+TEMP20
            D(JF+J)=TEMP19-TEMP20
            TEMP21=-CPIS1*(TEMP1+TEMP11)+CPIS2*(TEMP3+TEMP9)-CPIS3*(TEMP5+TEMP7)
            TEMP22=SPIS1*(TEMP2-TEMP12)-SPIS2*(TEMP4-TEMP10)+SPIS3*(TEMP6-TEMP8)
            C(JD+J)=TEMP21-TEMP22
            C(JE+J)=TEMP21+TEMP22
            TEMP23=-CPIS1*(TEMP2+TEMP12)+CPIS2*(TEMP4+TEMP10)-CPIS3*(TEMP6+TEMP8)
            TEMP24=SPIS1*(TEMP1-TEMP11)-SPIS2*(TEMP3-TEMP9)+SPIS3*(TEMP5-TEMP7)
            D(JD+J)=TEMP23+TEMP24
            D(JE+J)=TEMP23-TEMP24
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.7) THEN
!
! Coding for factor 8:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         IG=IF+INC1
         JG=JF+LOT
         IH=IG+INC1
         JH=JG+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            TEMP13=A(IH+I)
            TEMP14=B(IH+I)
            C(JA+J)=TEMP7+(TEMP1+TEMP9)+((TEMP3+TEMP11)+(TEMP5+TEMP13))
            D(JA+J)=TEMP8+(TEMP2+TEMP10)+((TEMP4+TEMP12)+(TEMP6+TEMP14))
            C(JE+J)=TEMP7-(TEMP1+TEMP9)+((TEMP3+TEMP11)-(TEMP5+TEMP13))
            D(JE+J)=TEMP8-(TEMP2+TEMP10)+((TEMP4+TEMP12)-(TEMP6+TEMP14))
            TEMP15=SQR2H*(((TEMP1-TEMP9)-(TEMP5-TEMP13))-((TEMP2-TEMP10)+(TEMP6-TEMP14)))
            TEMP16=SQR2H*(((TEMP2-TEMP10)-(TEMP6-TEMP14))+((TEMP1-TEMP9)+(TEMP5-TEMP13)))
            C(JB+J)=-TEMP7-(TEMP4-TEMP12)+TEMP15
            D(JB+J)=-TEMP8+(TEMP3-TEMP11)+TEMP16
            C(JF+J)=-TEMP7-(TEMP4-TEMP12)-TEMP15
            D(JF+J)=-TEMP8+(TEMP3-TEMP11)-TEMP16
            C(JC+J)=TEMP7-(TEMP3+TEMP11)-((TEMP2+TEMP10)-(TEMP6+TEMP14))
            D(JC+J)=TEMP8-(TEMP4+TEMP12)+((TEMP1+TEMP9)-(TEMP5+TEMP13))
            C(JG+J)=TEMP7-(TEMP3+TEMP11)+((TEMP2+TEMP10)-(TEMP6+TEMP14))
            D(JG+J)=TEMP8-(TEMP4+TEMP12)-((TEMP1+TEMP9)-(TEMP5+TEMP13))
            TEMP17=SQR2H*(((TEMP1-TEMP9)-(TEMP5-TEMP13))+((TEMP2-TEMP10)+(TEMP6-TEMP14)))
            TEMP18=SQR2H*(((TEMP2-TEMP10)-(TEMP6-TEMP14))-((TEMP1-TEMP9)+(TEMP5-TEMP13)))
            C(JD+J)=-TEMP7+(TEMP4-TEMP12)-TEMP17
            D(JD+J)=-TEMP8-(TEMP3-TEMP11)-TEMP18
            C(JH+J)=-TEMP7+(TEMP4-TEMP12)+TEMP17
            D(JH+J)=-TEMP8-(TEMP3-TEMP11)+TEMP18
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IH+I)=A(IA+I)+C(JH+J)
            B(IH+I)=B(IA+I)+D(JH+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

 1000 CONTINUE
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
!
! Coding for factor 2:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            C(JB+J)=TEMP1-TEMP3
            D(JB+J)=TEMP2-TEMP4
            A(IA+I)=TEMP1+TEMP3
            B(IA+I)=TEMP2+TEMP4
            I=I+INC2
            J=J+1
         ENDDO
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IB+I)=C(JB+J)
            B(IB+I)=D(JB+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.2) THEN
!
! Coding for factor 3:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=TEMP1+TEMP3
            TEMP6=TEMP2+TEMP4
            C(JA+J)=TEMP5
            D(JA+J)=TEMP6
            TEMP7=SIN60*(TEMP1-TEMP3)
            TEMP8=SIN60*(TEMP2-TEMP4)
            C(JB+J)=0.5_q*TEMP5-TEMP8
            C(JC+J)=0.5_q*TEMP5+TEMP8
            D(JB+J)=0.5_q*TEMP6+TEMP7
            D(JC+J)=0.5_q*TEMP6-TEMP7
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)-C(JB+J)
            A(IC+I)=A(IA+I)-C(JC+J)
            B(IB+I)=B(IA+I)-D(JB+J)
            B(IC+I)=B(IA+I)-D(JC+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
!
! Coding for factor 4:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            TEMP5=A(IC+I)
            TEMP6=B(IC+I)
            TEMP7=A(ID+I)
            TEMP8=B(ID+I)
            C(JA+J)=TEMP1+TEMP5
            C(JB+J)=TEMP3+TEMP7
            D(JA+J)=TEMP2+TEMP6
            D(JB+J)=TEMP4+TEMP8
            C(JC+J)=TEMP1-TEMP5
            C(JD+J)=TEMP4-TEMP8
            D(JC+J)=TEMP2-TEMP6
            D(JD+J)=TEMP3-TEMP7
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=C(JA+J)+C(JB+J)
            A(IC+I)=C(JA+J)-C(JB+J)
            B(IA+I)=D(JA+J)+D(JB+J)
            B(IC+I)=D(JA+J)-D(JB+J)
            A(IB+I)=C(JC+J)+C(JD+J)
            A(ID+I)=C(JC+J)-C(JD+J)
            B(IB+I)=D(JC+J)-D(JD+J)
            B(ID+I)=D(JC+J)+D(JD+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
!
! Coding for factor 5:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            C(JA+J)=(TEMP1+TEMP7)+(TEMP3+TEMP5)
            D(JA+J)=(TEMP2+TEMP8)+(TEMP4+TEMP6)
            TEMP9=COS72*(TEMP1+TEMP7)-COS36*(TEMP3+TEMP5)
            TEMP10=SIN72*(TEMP2-TEMP8)+SIN36*(TEMP4-TEMP6)
            C(JB+J)=TEMP9+TEMP10
            C(JE+J)=TEMP9-TEMP10
            TEMP11=COS72*(TEMP2+TEMP8)-COS36*(TEMP4+TEMP6)
            TEMP12=SIN72*(TEMP1-TEMP7)+SIN36*(TEMP3-TEMP5)
            D(JB+J)=TEMP11-TEMP12
            D(JE+J)=TEMP11+TEMP12
            TEMP13=COS72*(TEMP3+TEMP5)-COS36*(TEMP1+TEMP7)
            TEMP14=SIN36*(TEMP2-TEMP8)-SIN72*(TEMP4-TEMP6)
            C(JC+J)=TEMP13+TEMP14
            C(JD+J)=TEMP13-TEMP14
            TEMP15=COS72*(TEMP4+TEMP6)-COS36*(TEMP2+TEMP8)
            TEMP16=SIN36*(TEMP1-TEMP7)-SIN72*(TEMP3-TEMP5)
            D(JC+J)=TEMP15-TEMP16
            D(JD+J)=TEMP15+TEMP16
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.5) THEN
!
! Coding for factor 6:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            C(JA+J)= TEMP5+((TEMP1+TEMP9)+(TEMP3+TEMP7))
            C(JD+J)=-TEMP5-((TEMP1+TEMP9)-(TEMP3+TEMP7))
            D(JA+J)= TEMP6+((TEMP2+TEMP10)+(TEMP4+TEMP8))
            D(JD+J)=-TEMP6-((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP11=-TEMP5+0.5_q*((TEMP1+TEMP9)-(TEMP3+TEMP7))
            TEMP12=SIN60*((TEMP2-TEMP10)+(TEMP4-TEMP8))
            C(JB+J)=TEMP11+TEMP12
            C(JF+J)=TEMP11-TEMP12
            TEMP13=-TEMP6+0.5_q*((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP14=SIN60*((TEMP1-TEMP9)+(TEMP3-TEMP7))
            D(JB+J)=TEMP13-TEMP14
            D(JF+J)=TEMP13+TEMP14
            TEMP15=TEMP5-0.5_q*((TEMP1+TEMP9)+(TEMP3+TEMP7))
            TEMP16=SIN60*((TEMP2-TEMP10)-(TEMP4-TEMP8))
            C(JC+J)=TEMP15+TEMP16
            C(JE+J)=TEMP15-TEMP16
            TEMP17=TEMP6-0.5_q*((TEMP2+TEMP10)+(TEMP4+TEMP8))
            TEMP18=SIN60*((TEMP1-TEMP9)-(TEMP3-TEMP7))
            D(JC+J)=TEMP17-TEMP18
            D(JE+J)=TEMP17+TEMP18
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(ID+I)=A(IA+I)+C(JD+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.6) THEN
!
! Coding for factor 7:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         IG=IF+INC1
         JG=JF+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            C(JA+J)=(TEMP1+TEMP11)+(TEMP3+TEMP9)+(TEMP5+TEMP7)
            D(JA+J)=(TEMP2+TEMP12)+(TEMP4+TEMP10)+(TEMP6+TEMP8)
            TEMP13=CPIS2*(TEMP1+TEMP11)-CPIS3*(TEMP3+TEMP9)-CPIS1*(TEMP5+TEMP7)
            TEMP14=SPIS2*(TEMP2-TEMP12)+SPIS3*(TEMP4-TEMP10)+SPIS1*(TEMP6-TEMP8)
            C(JB+J)=TEMP13+TEMP14
            C(JG+J)=TEMP13-TEMP14
            TEMP15=CPIS2*(TEMP2+TEMP12)-CPIS3*(TEMP4+TEMP10)-CPIS1*(TEMP6+TEMP8)
            TEMP16=SPIS2*(TEMP1-TEMP11)+SPIS3*(TEMP3-TEMP9)+SPIS1*(TEMP5-TEMP7)
            D(JB+J)=TEMP15-TEMP16
            D(JG+J)=TEMP15+TEMP16
            TEMP17=-CPIS3*(TEMP1+TEMP11)-CPIS1*(TEMP3+TEMP9)+CPIS2*(TEMP5+TEMP7)
            TEMP18=SPIS3*(TEMP2-TEMP12)-SPIS1*(TEMP4-TEMP10)-SPIS2*(TEMP6-TEMP8)
            C(JC+J)=TEMP17+TEMP18
            C(JF+J)=TEMP17-TEMP18
            TEMP19=-CPIS3*(TEMP2+TEMP12)-CPIS1*(TEMP4+TEMP10)+CPIS2*(TEMP6+TEMP8)
            TEMP20=SPIS3*(TEMP1-TEMP11)-SPIS1*(TEMP3-TEMP9)-SPIS2*(TEMP5-TEMP7)
            D(JC+J)=TEMP19-TEMP20
            D(JF+J)=TEMP19+TEMP20
            TEMP21=-CPIS1*(TEMP1+TEMP11)+CPIS2*(TEMP3+TEMP9)-CPIS3*(TEMP5+TEMP7)
            TEMP22=SPIS1*(TEMP2-TEMP12)-SPIS2*(TEMP4-TEMP10)+SPIS3*(TEMP6-TEMP8)
            C(JD+J)=TEMP21+TEMP22
            C(JE+J)=TEMP21-TEMP22
            TEMP23=-CPIS1*(TEMP2+TEMP12)+CPIS2*(TEMP4+TEMP10)-CPIS3*(TEMP6+TEMP8)
            TEMP24=SPIS1*(TEMP1-TEMP11)-SPIS2*(TEMP3-TEMP9)+SPIS3*(TEMP5-TEMP7)
            D(JD+J)=TEMP23-TEMP24
            D(JE+J)=TEMP23+TEMP24
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.7) THEN
!
! Coding for factor 8:
!
         IA=1
         JA=1
         IB=IA+INC1
         JB=JA+LOT
         IC=IB+INC1
         JC=JB+LOT
         ID=IC+INC1
         JD=JC+LOT
         IE=ID+INC1
         JE=JD+LOT
         IF=IE+INC1
         JF=JE+LOT
         IG=IF+INC1
         JG=JF+LOT
         IH=IG+INC1
         JH=JG+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            TEMP13=A(IH+I)
            TEMP14=B(IH+I)
            C(JA+J)=A(IE+I)+(A(IB+I)+A(IF+I))+((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
            D(JA+J)=B(IE+I)+(B(IB+I)+B(IF+I))+((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
            C(JE+J)=A(IE+I)-(A(IB+I)+A(IF+I))+((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
            D(JE+J)=B(IE+I)-(B(IB+I)+B(IF+I))+((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
            TEMP15=SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I)))+((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
            TEMP16=SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I)))-((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
            C(JB+J)=-A(IE+I)+(B(IC+I)-B(IG+I))+TEMP15
            D(JB+J)=-B(IE+I)-(A(IC+I)-A(IG+I))+TEMP16
            C(JF+J)=-A(IE+I)+(B(IC+I)-B(IG+I))-TEMP15
            D(JF+J)=-B(IE+I)-(A(IC+I)-A(IG+I))-TEMP16
            C(JC+J)=A(IE+I)-(A(IC+I)+A(IG+I))+((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
            D(JC+J)=B(IE+I)-(B(IC+I)+B(IG+I))-((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
            C(JG+J)=A(IE+I)-(A(IC+I)+A(IG+I))-((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
            D(JG+J)=B(IE+I)-(B(IC+I)+B(IG+I))+((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
            TEMP17=SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I)))-((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
            TEMP18=SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I)))+((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
            C(JD+J)=-A(IE+I)-(B(IC+I)-B(IG+I))-TEMP17
            D(JD+J)=-B(IE+I)+(A(IC+I)-A(IG+I))-TEMP18
            C(JH+J)=-A(IE+I)-(B(IC+I)-B(IG+I))+TEMP17
            D(JH+J)=-B(IE+I)+(A(IC+I)-A(IG+I))+TEMP18
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IH+I)=A(IA+I)+C(JH+J)
            B(IH+I)=B(IA+I)+D(JH+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+INC2
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+INC2
            J=J+1
         ENDDO
      ENDIF
      RETURN
      END SUBROUTINE SPASSM

      SUBROUTINE SPASSM_1_1(A,B,C,D,INC1,INC2,LOT,N,ISIGN)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! SPASSM is derived from the TEMPERTON/FULKER routine 'VPASSM' which   *
! performs one pass through data as part of a multiple complex Fast    *
! Fourier Transformation in one dimension on multiple 1-D data sets.   *
! SPASSM is the special version for 'single factor transformations'.   *
! Variables:                                                           *
!   A is the array containing the real parts of the data vector        *
!   B is the array containing the imaginary parts of the data vector   *
!   C is a work array                                                  *
!   D is a work array                                                  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the addressing increment between data sets in A and B      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   ISIGN gives the type of transformation (>=0: forward, <0: inverse) *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907_q)
      PARAMETER(COS36=0.80901699437494742410229341718282_q)
      PARAMETER(SIN60=0.86602540378443864676372317075294_q)
      PARAMETER(SIN72=0.95105651629515357211643933337938_q)
      PARAMETER(COS72=0.30901699437494742410229341718282_q)
      PARAMETER(SQR2H=0.70710678118654752440084436210485_q)
      PARAMETER(CPIS1=0.90096886790241912623610231950745_q)
      PARAMETER(CPIS2=0.62348980185873353052500488400424_q)
      PARAMETER(CPIS3=0.22252093395631440428890256449679_q)
      PARAMETER(SPIS1=0.43388373911755812047576833284836_q)
      PARAMETER(SPIS2=0.78183148246802980870844452667406_q)
      PARAMETER(SPIS3=0.97492791218182360701813168299393_q)
!
      IGO=N-1
      IF (IGO>7) THEN 
         RETURN
      ENDIF

      IF (ISIGN<0) GOTO 1000
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
!
! Coding for factor 2:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            C(JB+J)=TEMP1-TEMP3
            D(JB+J)=TEMP2-TEMP4
            A(IA+I)=TEMP1+TEMP3
            B(IA+I)=TEMP2+TEMP4
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IB+I)=C(JB+J)
            B(IB+I)=D(JB+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.2) THEN
!
! Coding for factor 3:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=TEMP1+TEMP3
            TEMP6=TEMP2+TEMP4
            TEMP7=SIN60*(TEMP1-TEMP3)
            TEMP8=SIN60*(TEMP2-TEMP4)
            C(JA+J)=TEMP5
            D(JA+J)=TEMP6
            C(JB+J)=0.5_q*TEMP5+TEMP8
            C(JC+J)=0.5_q*TEMP5-TEMP8
            D(JB+J)=0.5_q*TEMP6-TEMP7
            D(JC+J)=0.5_q*TEMP6+TEMP7
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)-C(JB+J)
            A(IC+I)=A(IA+I)-C(JC+J)
            B(IB+I)=B(IA+I)-D(JB+J)
            B(IC+I)=B(IA+I)-D(JC+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
!
! Coding for factor 4:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            TEMP5=A(IC+I)
            TEMP6=B(IC+I)
            TEMP7=A(ID+I)
            TEMP8=B(ID+I)
            C(JA+J)=TEMP1+TEMP5
            C(JB+J)=TEMP3+TEMP7
            D(JA+J)=TEMP2+TEMP6
            D(JB+J)=TEMP4+TEMP8
            C(JC+J)=TEMP1-TEMP5
            C(JD+J)=TEMP4-TEMP8
            D(JC+J)=TEMP2-TEMP6
            D(JD+J)=TEMP3-TEMP7
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=C(JA+J)
            TEMP2=D(JA+J)
            TEMP3=C(JB+J)
            TEMP4=D(JB+J)
            TEMP5=C(JC+J)
            TEMP6=D(JC+J)
            TEMP7=C(JD+J)
            TEMP8=D(JD+J)
            A(IA+I)=TEMP1+TEMP3
            A(IC+I)=TEMP1-TEMP3
            B(IA+I)=TEMP2+TEMP4
            B(IC+I)=TEMP2-TEMP4
            A(IB+I)=TEMP5-TEMP7
            A(ID+I)=TEMP5+TEMP7
            B(IB+I)=TEMP6+TEMP8
            B(ID+I)=TEMP6-TEMP8
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
!
! Coding for factor 5:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            C(JA+J)=(TEMP1+TEMP7)+(TEMP3+TEMP5)
            D(JA+J)=(TEMP2+TEMP8)+(TEMP4+TEMP6)
            TEMP9=COS72*(TEMP1+TEMP7)-COS36*(TEMP3+TEMP5)
            TEMP10=SIN72*(TEMP2-TEMP8)+SIN36*(TEMP4-TEMP6)
            C(JB+J)=TEMP9-TEMP10
            C(JE+J)=TEMP9+TEMP10
            TEMP11=COS72*(TEMP2+TEMP8)-COS36*(TEMP4+TEMP6)
            TEMP12=SIN72*(TEMP1-TEMP7)+SIN36*(TEMP3-TEMP5)
            D(JB+J)=TEMP11+TEMP12
            D(JE+J)=TEMP11-TEMP12
            TEMP13=COS72*(TEMP3+TEMP5)-COS36*(TEMP1+TEMP7)
            TEMP14=SIN36*(TEMP2-TEMP8)-SIN72*(TEMP4-TEMP6)
            C(JC+J)=TEMP13-TEMP14
            C(JD+J)=TEMP13+TEMP14
            TEMP15=COS72*(TEMP4+TEMP6)-COS36*(TEMP2+TEMP8)
            TEMP16=SIN36*(TEMP1-TEMP7)-SIN72*(TEMP3-TEMP5)
            D(JC+J)=TEMP15+TEMP16
            D(JD+J)=TEMP15-TEMP16
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.5) THEN
!
! Coding for factor 6:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            C(JA+J)= TEMP5+((TEMP1+TEMP9)+(TEMP3+TEMP7))
            C(JD+J)=-TEMP5-((TEMP1+TEMP9)-(TEMP3+TEMP7))
            D(JA+J)= TEMP6+((TEMP2+TEMP10)+(TEMP4+TEMP8))
            D(JD+J)=-TEMP6-((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP11=-TEMP5+0.5_q*((TEMP1+TEMP9)-(TEMP3+TEMP7))
            TEMP12=SIN60*((TEMP2-TEMP10)+(TEMP4-TEMP8))
            C(JB+J)=TEMP11-TEMP12
            C(JF+J)=TEMP11+TEMP12
            TEMP13=-TEMP6+0.5_q*((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP14=SIN60*((TEMP1-TEMP9)+(TEMP3-TEMP7))
            D(JB+J)=TEMP13+TEMP14
            D(JF+J)=TEMP13-TEMP14
            TEMP15=TEMP5-0.5_q*((TEMP1+TEMP9)+(TEMP3+TEMP7))
            TEMP16=SIN60*((TEMP2-TEMP10)-(TEMP4-TEMP8))
            C(JC+J)=TEMP15-TEMP16
            C(JE+J)=TEMP15+TEMP16
            TEMP17=TEMP6-0.5_q*((TEMP2+TEMP10)+(TEMP4+TEMP8))
            TEMP18=SIN60*((TEMP1-TEMP9)-(TEMP3-TEMP7))
            D(JC+J)=TEMP17+TEMP18
            D(JE+J)=TEMP17-TEMP18
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(ID+I)=A(IA+I)+C(JD+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.6) THEN
!
! Coding for factor 7:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         IG=IF+1
         JG=JF+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            C(JA+J)=(TEMP1+TEMP11)+(TEMP3+TEMP9)+(TEMP5+TEMP7)
            D(JA+J)=(TEMP2+TEMP12)+(TEMP4+TEMP10)+(TEMP6+TEMP8)
            TEMP13=CPIS2*(TEMP1+TEMP11)-CPIS3*(TEMP3+TEMP9)-CPIS1*(TEMP5+TEMP7)
            TEMP14=SPIS2*(TEMP2-TEMP12)+SPIS3*(TEMP4-TEMP10)+SPIS1*(TEMP6-TEMP8)
            C(JB+J)=TEMP13-TEMP14
            C(JG+J)=TEMP13+TEMP14
            TEMP15=CPIS2*(TEMP2+TEMP12)-CPIS3*(TEMP4+TEMP10)-CPIS1*(TEMP6+TEMP8)
            TEMP16=SPIS2*(TEMP1-TEMP11)+SPIS3*(TEMP3-TEMP9)+SPIS1*(TEMP5-TEMP7)
            D(JB+J)=TEMP15+TEMP16
            D(JG+J)=TEMP15-TEMP16
            TEMP17=-CPIS3*(TEMP1+TEMP11)-CPIS1*(TEMP3+TEMP9)+CPIS2*(TEMP5+TEMP7)
            TEMP18=SPIS3*(TEMP2-TEMP12)-SPIS1*(TEMP4-TEMP10)-SPIS2*(TEMP6-TEMP8)
            C(JC+J)=TEMP17-TEMP18
            C(JF+J)=TEMP17+TEMP18
            TEMP19=-CPIS3*(TEMP2+TEMP12)-CPIS1*(TEMP4+TEMP10)+CPIS2*(TEMP6+TEMP8)
            TEMP20=SPIS3*(TEMP1-TEMP11)-SPIS1*(TEMP3-TEMP9)-SPIS2*(TEMP5-TEMP7)
            D(JC+J)=TEMP19+TEMP20
            D(JF+J)=TEMP19-TEMP20
            TEMP21=-CPIS1*(TEMP1+TEMP11)+CPIS2*(TEMP3+TEMP9)-CPIS3*(TEMP5+TEMP7)
            TEMP22=SPIS1*(TEMP2-TEMP12)-SPIS2*(TEMP4-TEMP10)+SPIS3*(TEMP6-TEMP8)
            C(JD+J)=TEMP21-TEMP22
            C(JE+J)=TEMP21+TEMP22
            TEMP23=-CPIS1*(TEMP2+TEMP12)+CPIS2*(TEMP4+TEMP10)-CPIS3*(TEMP6+TEMP8)
            TEMP24=SPIS1*(TEMP1-TEMP11)-SPIS2*(TEMP3-TEMP9)+SPIS3*(TEMP5-TEMP7)
            D(JD+J)=TEMP23+TEMP24
            D(JE+J)=TEMP23-TEMP24
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.7) THEN
!
! Coding for factor 8:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         IG=IF+1
         JG=JF+LOT
         IH=IG+1
         JH=JG+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            TEMP13=A(IH+I)
            TEMP14=B(IH+I)
            C(JA+J)=TEMP7+(TEMP1+TEMP9)+((TEMP3+TEMP11)+(TEMP5+TEMP13))
            D(JA+J)=TEMP8+(TEMP2+TEMP10)+((TEMP4+TEMP12)+(TEMP6+TEMP14))
            C(JE+J)=TEMP7-(TEMP1+TEMP9)+((TEMP3+TEMP11)-(TEMP5+TEMP13))
            D(JE+J)=TEMP8-(TEMP2+TEMP10)+((TEMP4+TEMP12)-(TEMP6+TEMP14))
            TEMP15=SQR2H*(((TEMP1-TEMP9)-(TEMP5-TEMP13))-((TEMP2-TEMP10)+(TEMP6-TEMP14)))
            TEMP16=SQR2H*(((TEMP2-TEMP10)-(TEMP6-TEMP14))+((TEMP1-TEMP9)+(TEMP5-TEMP13)))
            C(JB+J)=-TEMP7-(TEMP4-TEMP12)+TEMP15
            D(JB+J)=-TEMP8+(TEMP3-TEMP11)+TEMP16
            C(JF+J)=-TEMP7-(TEMP4-TEMP12)-TEMP15
            D(JF+J)=-TEMP8+(TEMP3-TEMP11)-TEMP16
            C(JC+J)=TEMP7-(TEMP3+TEMP11)-((TEMP2+TEMP10)-(TEMP6+TEMP14))
            D(JC+J)=TEMP8-(TEMP4+TEMP12)+((TEMP1+TEMP9)-(TEMP5+TEMP13))
            C(JG+J)=TEMP7-(TEMP3+TEMP11)+((TEMP2+TEMP10)-(TEMP6+TEMP14))
            D(JG+J)=TEMP8-(TEMP4+TEMP12)-((TEMP1+TEMP9)-(TEMP5+TEMP13))
            TEMP17=SQR2H*(((TEMP1-TEMP9)-(TEMP5-TEMP13))+((TEMP2-TEMP10)+(TEMP6-TEMP14)))
            TEMP18=SQR2H*(((TEMP2-TEMP10)-(TEMP6-TEMP14))-((TEMP1-TEMP9)+(TEMP5-TEMP13)))
            C(JD+J)=-TEMP7+(TEMP4-TEMP12)-TEMP17
            D(JD+J)=-TEMP8-(TEMP3-TEMP11)-TEMP18
            C(JH+J)=-TEMP7+(TEMP4-TEMP12)+TEMP17
            D(JH+J)=-TEMP8-(TEMP3-TEMP11)+TEMP18
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IH+I)=A(IA+I)+C(JH+J)
            B(IH+I)=B(IA+I)+D(JH+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

 1000 CONTINUE
! test change doris: compatible with ifort>=9.1, -e95 compiler option
      IF (IGO.EQ.1) THEN
!
! Coding for factor 2:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            C(JB+J)=TEMP1-TEMP3
            D(JB+J)=TEMP2-TEMP4
            A(IA+I)=TEMP1+TEMP3
            B(IA+I)=TEMP2+TEMP4
            I=I+1
            J=J+1
         ENDDO
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IB+I)=C(JB+J)
            B(IB+I)=D(JB+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.2) THEN
!
! Coding for factor 3:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=TEMP1+TEMP3
            TEMP6=TEMP2+TEMP4
            C(JA+J)=TEMP5
            D(JA+J)=TEMP6
            TEMP7=SIN60*(TEMP1-TEMP3)
            TEMP8=SIN60*(TEMP2-TEMP4)
            C(JB+J)=0.5_q*TEMP5-TEMP8
            C(JC+J)=0.5_q*TEMP5+TEMP8
            D(JB+J)=0.5_q*TEMP6+TEMP7
            D(JC+J)=0.5_q*TEMP6-TEMP7
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)-C(JB+J)
            A(IC+I)=A(IA+I)-C(JC+J)
            B(IB+I)=B(IA+I)-D(JB+J)
            B(IC+I)=B(IA+I)-D(JC+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.3) THEN
!
! Coding for factor 4:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IA+I)
            TEMP2=B(IA+I)
            TEMP3=A(IB+I)
            TEMP4=B(IB+I)
            TEMP5=A(IC+I)
            TEMP6=B(IC+I)
            TEMP7=A(ID+I)
            TEMP8=B(ID+I)
            C(JA+J)=TEMP1+TEMP5
            C(JB+J)=TEMP3+TEMP7
            D(JA+J)=TEMP2+TEMP6
            D(JB+J)=TEMP4+TEMP8
            C(JC+J)=TEMP1-TEMP5
            C(JD+J)=TEMP4-TEMP8
            D(JC+J)=TEMP2-TEMP6
            D(JD+J)=TEMP3-TEMP7
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=C(JA+J)+C(JB+J)
            A(IC+I)=C(JA+J)-C(JB+J)
            B(IA+I)=D(JA+J)+D(JB+J)
            B(IC+I)=D(JA+J)-D(JB+J)
            A(IB+I)=C(JC+J)+C(JD+J)
            A(ID+I)=C(JC+J)-C(JD+J)
            B(IB+I)=D(JC+J)-D(JD+J)
            B(ID+I)=D(JC+J)+D(JD+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.4) THEN
!
! Coding for factor 5:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            C(JA+J)=(TEMP1+TEMP7)+(TEMP3+TEMP5)
            D(JA+J)=(TEMP2+TEMP8)+(TEMP4+TEMP6)
            TEMP9=COS72*(TEMP1+TEMP7)-COS36*(TEMP3+TEMP5)
            TEMP10=SIN72*(TEMP2-TEMP8)+SIN36*(TEMP4-TEMP6)
            C(JB+J)=TEMP9+TEMP10
            C(JE+J)=TEMP9-TEMP10
            TEMP11=COS72*(TEMP2+TEMP8)-COS36*(TEMP4+TEMP6)
            TEMP12=SIN72*(TEMP1-TEMP7)+SIN36*(TEMP3-TEMP5)
            D(JB+J)=TEMP11-TEMP12
            D(JE+J)=TEMP11+TEMP12
            TEMP13=COS72*(TEMP3+TEMP5)-COS36*(TEMP1+TEMP7)
            TEMP14=SIN36*(TEMP2-TEMP8)-SIN72*(TEMP4-TEMP6)
            C(JC+J)=TEMP13+TEMP14
            C(JD+J)=TEMP13-TEMP14
            TEMP15=COS72*(TEMP4+TEMP6)-COS36*(TEMP2+TEMP8)
            TEMP16=SIN36*(TEMP1-TEMP7)-SIN72*(TEMP3-TEMP5)
            D(JC+J)=TEMP15-TEMP16
            D(JD+J)=TEMP15+TEMP16
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.5) THEN
!
! Coding for factor 6:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            C(JA+J)= TEMP5+((TEMP1+TEMP9)+(TEMP3+TEMP7))
            C(JD+J)=-TEMP5-((TEMP1+TEMP9)-(TEMP3+TEMP7))
            D(JA+J)= TEMP6+((TEMP2+TEMP10)+(TEMP4+TEMP8))
            D(JD+J)=-TEMP6-((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP11=-TEMP5+0.5_q*((TEMP1+TEMP9)-(TEMP3+TEMP7))
            TEMP12=SIN60*((TEMP2-TEMP10)+(TEMP4-TEMP8))
            C(JB+J)=TEMP11+TEMP12
            C(JF+J)=TEMP11-TEMP12
            TEMP13=-TEMP6+0.5_q*((TEMP2+TEMP10)-(TEMP4+TEMP8))
            TEMP14=SIN60*((TEMP1-TEMP9)+(TEMP3-TEMP7))
            D(JB+J)=TEMP13-TEMP14
            D(JF+J)=TEMP13+TEMP14
            TEMP15=TEMP5-0.5_q*((TEMP1+TEMP9)+(TEMP3+TEMP7))
            TEMP16=SIN60*((TEMP2-TEMP10)-(TEMP4-TEMP8))
            C(JC+J)=TEMP15+TEMP16
            C(JE+J)=TEMP15-TEMP16
            TEMP17=TEMP6-0.5_q*((TEMP2+TEMP10)+(TEMP4+TEMP8))
            TEMP18=SIN60*((TEMP1-TEMP9)-(TEMP3-TEMP7))
            D(JC+J)=TEMP17-TEMP18
            D(JE+J)=TEMP17+TEMP18
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(ID+I)=A(IA+I)+C(JD+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN         
      ENDIF

      IF (IGO.EQ.6) THEN
!
! Coding for factor 7:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         IG=IF+1
         JG=JF+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            C(JA+J)=(TEMP1+TEMP11)+(TEMP3+TEMP9)+(TEMP5+TEMP7)
            D(JA+J)=(TEMP2+TEMP12)+(TEMP4+TEMP10)+(TEMP6+TEMP8)
            TEMP13=CPIS2*(TEMP1+TEMP11)-CPIS3*(TEMP3+TEMP9)-CPIS1*(TEMP5+TEMP7)
            TEMP14=SPIS2*(TEMP2-TEMP12)+SPIS3*(TEMP4-TEMP10)+SPIS1*(TEMP6-TEMP8)
            C(JB+J)=TEMP13+TEMP14
            C(JG+J)=TEMP13-TEMP14
            TEMP15=CPIS2*(TEMP2+TEMP12)-CPIS3*(TEMP4+TEMP10)-CPIS1*(TEMP6+TEMP8)
            TEMP16=SPIS2*(TEMP1-TEMP11)+SPIS3*(TEMP3-TEMP9)+SPIS1*(TEMP5-TEMP7)
            D(JB+J)=TEMP15-TEMP16
            D(JG+J)=TEMP15+TEMP16
            TEMP17=-CPIS3*(TEMP1+TEMP11)-CPIS1*(TEMP3+TEMP9)+CPIS2*(TEMP5+TEMP7)
            TEMP18=SPIS3*(TEMP2-TEMP12)-SPIS1*(TEMP4-TEMP10)-SPIS2*(TEMP6-TEMP8)
            C(JC+J)=TEMP17+TEMP18
            C(JF+J)=TEMP17-TEMP18
            TEMP19=-CPIS3*(TEMP2+TEMP12)-CPIS1*(TEMP4+TEMP10)+CPIS2*(TEMP6+TEMP8)
            TEMP20=SPIS3*(TEMP1-TEMP11)-SPIS1*(TEMP3-TEMP9)-SPIS2*(TEMP5-TEMP7)
            D(JC+J)=TEMP19-TEMP20
            D(JF+J)=TEMP19+TEMP20
            TEMP21=-CPIS1*(TEMP1+TEMP11)+CPIS2*(TEMP3+TEMP9)-CPIS3*(TEMP5+TEMP7)
            TEMP22=SPIS1*(TEMP2-TEMP12)-SPIS2*(TEMP4-TEMP10)+SPIS3*(TEMP6-TEMP8)
            C(JD+J)=TEMP21+TEMP22
            C(JE+J)=TEMP21-TEMP22
            TEMP23=-CPIS1*(TEMP2+TEMP12)+CPIS2*(TEMP4+TEMP10)-CPIS3*(TEMP6+TEMP8)
            TEMP24=SPIS1*(TEMP1-TEMP11)-SPIS2*(TEMP3-TEMP9)+SPIS3*(TEMP5-TEMP7)
            D(JD+J)=TEMP23-TEMP24
            D(JE+J)=TEMP23+TEMP24
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
         RETURN
      ENDIF

      IF (IGO.EQ.7) THEN
!
! Coding for factor 8:
!
         IA=1
         JA=1
         IB=IA+1
         JB=JA+LOT
         IC=IB+1
         JC=JB+LOT
         ID=IC+1
         JD=JC+LOT
         IE=ID+1
         JE=JD+LOT
         IF=IE+1
         JF=JE+LOT
         IG=IF+1
         JG=JF+LOT
         IH=IG+1
         JH=JG+LOT
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            TEMP1=A(IB+I)
            TEMP2=B(IB+I)
            TEMP3=A(IC+I)
            TEMP4=B(IC+I)
            TEMP5=A(ID+I)
            TEMP6=B(ID+I)
            TEMP7=A(IE+I)
            TEMP8=B(IE+I)
            TEMP9=A(IF+I)
            TEMP10=B(IF+I)
            TEMP11=A(IG+I)
            TEMP12=B(IG+I)
            TEMP13=A(IH+I)
            TEMP14=B(IH+I)
            C(JA+J)=A(IE+I)+(A(IB+I)+A(IF+I))+((A(IC+I)+A(IG+I))+(A(ID+I)+A(IH+I)))
            D(JA+J)=B(IE+I)+(B(IB+I)+B(IF+I))+((B(IC+I)+B(IG+I))+(B(ID+I)+B(IH+I)))
            C(JE+J)=A(IE+I)-(A(IB+I)+A(IF+I))+((A(IC+I)+A(IG+I))-(A(ID+I)+A(IH+I)))
            D(JE+J)=B(IE+I)-(B(IB+I)+B(IF+I))+((B(IC+I)+B(IG+I))-(B(ID+I)+B(IH+I)))
            TEMP15=SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I)))+((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
            TEMP16=SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I)))-((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
            C(JB+J)=-A(IE+I)+(B(IC+I)-B(IG+I))+TEMP15
            D(JB+J)=-B(IE+I)-(A(IC+I)-A(IG+I))+TEMP16
            C(JF+J)=-A(IE+I)+(B(IC+I)-B(IG+I))-TEMP15
            D(JF+J)=-B(IE+I)-(A(IC+I)-A(IG+I))-TEMP16
            C(JC+J)=A(IE+I)-(A(IC+I)+A(IG+I))+((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
            D(JC+J)=B(IE+I)-(B(IC+I)+B(IG+I))-((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
            C(JG+J)=A(IE+I)-(A(IC+I)+A(IG+I))-((B(IB+I)+B(IF+I))-(B(ID+I)+B(IH+I)))
            D(JG+J)=B(IE+I)-(B(IC+I)+B(IG+I))+((A(IB+I)+A(IF+I))-(A(ID+I)+A(IH+I)))
            TEMP17=SQR2H*(((A(IB+I)-A(IF+I))-(A(ID+I)-A(IH+I)))-((B(IB+I)-B(IF+I))+(B(ID+I)-B(IH+I))))
            TEMP18=SQR2H*(((B(IB+I)-B(IF+I))-(B(ID+I)-B(IH+I)))+((A(IB+I)-A(IF+I))+(A(ID+I)-A(IH+I))))
            C(JD+J)=-A(IE+I)-(B(IC+I)-B(IG+I))-TEMP17
            D(JD+J)=-B(IE+I)+(A(IC+I)-A(IG+I))-TEMP18
            C(JH+J)=-A(IE+I)-(B(IC+I)-B(IG+I))+TEMP17
            D(JH+J)=-B(IE+I)+(A(IC+I)-A(IG+I))+TEMP18
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
         DO IJK=1,LOT
            A(IH+I)=A(IA+I)+C(JH+J)
            B(IH+I)=B(IA+I)+D(JH+J)
            A(IB+I)=A(IA+I)+C(JB+J)
            A(IG+I)=A(IA+I)+C(JG+J)
            B(IB+I)=B(IA+I)+D(JB+J)
            B(IG+I)=B(IA+I)+D(JG+J)
            A(IC+I)=A(IA+I)+C(JC+J)
            A(IF+I)=A(IA+I)+C(JF+J)
            B(IC+I)=B(IA+I)+D(JC+J)
            B(IF+I)=B(IA+I)+D(JF+J)
            A(ID+I)=A(IA+I)+C(JD+J)
            A(IE+I)=A(IA+I)+C(JE+J)
            B(ID+I)=B(IA+I)+D(JD+J)
            B(IE+I)=B(IA+I)+D(JE+J)
            I=I+1
            J=J+1
         ENDDO
         I=0
         J=0
#if defined(VECTORIZE)
#else
!DIR$ NOVECTOR
#endif
         DO IJK=1,LOT
            A(IA+I)=A(IA+I)+C(JA+J)
            B(IA+I)=B(IA+I)+D(JA+J)
            I=I+1
            J=J+1
         ENDDO
      ENDIF
      RETURN
      END SUBROUTINE SPASSM_1_1

      SUBROUTINE CFFTML(AR,AI,WORK,TRIGS,IFAC,INCL,INCN,L,N,ISIGN,MCACHE)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! This subroutine performs C omplex 1-D F ast F ourier T ransforms on  *
! M ultiple data sets (doing the 1-D transforms in parallel). Routine  *
! CFFTML uses the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM.  *
!                                                                      *
!***********************************************************************

      DIMENSION AR((N-1)*INCN+(L-1)*INCL+1),AI((N-1)*INCN+(L-1)*INCL+1)
      DIMENSION WORK(4*L*N),TRIGS(2*L),IFAC(19)
      DATA IWOFF /0/, MAXOLD /0/
      SAVE IWOFF,MAXOLD
! no data return immediately
      IF (N.EQ.0) RETURN
! IFAC(1) contains the number of factors
      NFAC=IFAC(1)

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),MINLOOP))
      IF (MCACHE==0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      IF ((MAXDAT*L)/=MAXOLD) THEN
        IF (IWOFF/=0) IWOFF=2*MAXDAT*L
        IF (IWOFF>(2*MAXOLD)) IWOFF=0
        MAXOLD=MAXDAT*L
      ENDIF
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      LA=1
! Special case NFAC=1:
      IF (NFAC.EQ.1) THEN
! Use routine SPASSM, nothing will be swapped (SPASSM swaps internally):
         IF ((INCL.EQ.1).AND.(INCN.EQ.1)) THEN
            CALL SPASSM_1_1(AR(I0),AI(I0),WORK(1),WORK(1+N0*L), &
     &                      1,1,N0,L,ISIGN)
         ELSE
            CALL SPASSM(AR(I0),AI(I0),WORK(1),WORK(1+N0*L), &
     &                  INCL,INCN,N0,L,ISIGN)
         ENDIF
! At this point the final transform result is stored in AR/AI, bye ...
         GOTO 10000
      END IF
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR/AI to WORK:
               CALL FPASSM(AR(I0),AI(I0),WORK(1+IWOFF), &
     &                     WORK(1+IWOFF+N0*L),TRIGS(1), &
     &                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
           ELSE IF (K==NFAC) THEN
! The last pass swaps WORK to AR/AI:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AI(I0),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR/AI to WORK:
               CALL IPASSM(AR(I0),AI(I0),WORK(1+IWOFF), &
     &                     WORK(1+IWOFF+N0*L),TRIGS(1), &
     &                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE IF (K==NFAC) THEN
! The last pass swaps WORK to AR/AI:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AI(I0),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
10000 CONTINUE
! At this point the final transform result is stored in AR/AI, bye ...
      RETURN
      END

      SUBROUTINE FFTC3V(DATC,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC, &
     &                                     TRIG1,TRIG2,TRIG3,ISIGN,IERR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! High-performance 3-dimensional complex Fast Fourier Transformation   *
! using the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM. This   *
! routine uses complex storage mode for input and output, real storage *
! mode on work array hopefully speeding up the whole at least for long *
! transforms with many factors (hence many operations on work array).  *
!                                                                      *
! This routine is specially optimized for vector machines -- never use *
! this routine on "small workstations"!!!! (no efficient use of cache) *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,TSET
      DIMENSION DATC(2*ID1*ID2*ID3),WORK(4*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./

      IERR=0
! Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<N1).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routine CFTTAB.
         CALL CFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN
! Extract numbers of factors and preset some variables:
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3
      IWOFF=0
      LA=1
!
!     Perform the transform passes needed to transform third dimension,
!     one pass for each factor:
!
      INCL=2*ID1*ID2
      INCN=2
      L=N3
      N=ID1*ID2
      INCLO=1
      INCNO=ID3
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC3
            IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATC to WORK:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATC to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   50    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC3
            IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATC to WORK:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATC to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   60    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'third' dimension), one pass for each factor:
!
      INCL=ID1*ID3
      INCN=1
      L=N2
      N=ID1*ID3
      INCLO=1
      INCNO=ID2
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 70 K=1,NFAC2
            IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   70    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 80 K=1,NFAC2
            IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   80    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
!
!     Perform the transform passes needed to transform first dimension
!     (which is currently 'third' dimension), one pass for each factor:
!
      INCL=ID2*ID3
      INCN=1
      L=N1
      N=ID2*ID3
      INCLO=2
      INCNO=2*ID1
      IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
         DO 90 K=1,NFAC1
            IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATC and rearranges order of data
! to shift the role of the three dimensions cyclically:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1), &
     &              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
   90    CONTINUE
      ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 100 K=1,NFAC1
            IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATC and rearranges order of data
! to shift the role of the three dimensions cyclically:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1), &
     &              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                         N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
  100    CONTINUE
      END IF
      RETURN
      END

      SUBROUTINE FFTR3V(DATA,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC, &
     &                               TRIG1,TRIG2,TRIG3,ISIGN,IHERM,IERR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! High-performance 3-dimensional real <--> first dimension hermitian   *
! Fast Fourier Transformation using the TEMPERTON/FULKER FFT-kernel    *
! routines FPASSM/IPASSM. This routine uses complex storage mode for   *
! hermitian input and output and real storage mode on the work array   *
! hopefully speeding up the whole at least for long transforms with    *
! many factors (hence many operations on work array).                  *
!                                                                      *
! This routine is specially optimized for vector machines -- never use *
! this routine on "small workstations"!!!! (no efficient use of cache) *
!                                                                      *
!***********************************************************************

      LOGICAL FACERR,FACODD,TSET
      DIMENSION DATA(ID1*ID2*ID3),WORK(2*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR,FACODD
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./

      IERR=0
! Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN==0) THEN
! First check errors:
         IF ((ID1<(N1+2)).OR.(ID2<N2).OR.(ID3<N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (FACODD(N1)) IERR=4
         IF (IERR/=0) RETURN
! Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
! Initialisation of the tables using routine CFTTAB.
         CALL RFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
! Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1/=N1SET).OR.(N2/=N2SET) &
     &                                         .OR.(N3/=N3SET)) IERR=3
      IF (IERR/=0) RETURN
! Extract numbers of factors and preset some variables:
      RSIGN=ISIGN
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3/2
      IWOFF=0
      LA=1

!***********************************************************************
!                                                                      *
!     Here comes the real --> complex hermite FFT ...                  *
!                                                                      *
!***********************************************************************
      IF (IHERM>=0) THEN
!
!     Perform the transform passes needed to transform first dimension,
!     one pass for each factor:
!
         INCL=2
         INCN=ID1
         L=N1/2
         N=ID2*ID3
         INCLO=ID2*ID3
         INCNO=1
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 50 K=1,NFAC1
               IF (K==1) THEN
! The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   50       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 60 K=1,NFAC1
               IF (K==1) THEN
! The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   60       CONTINUE
         END IF
         LA=1
!
! Now we have some complex data sets containing the requested result in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps between portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
!
         NEWOFF=2*ID123-IWOFF
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123), &
     &              TRIG1(1),N,INCLO,1,INCNO,N,L,RSIGN)
         IWOFF=NEWOFF
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'first' dimension), one pass for each factor:
!
         INCL=1
         INCN=ID2
         L=N2
         N=ID1*ID3/2
         INCLO=ID1*ID3/2
         INCNO=1
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 70 K=1,NFAC2
               IF ((K==1).AND.(NFAC2==1)) THEN
! The first (and only) pass swaps between different portions of WORK and
! rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC2>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   70       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 80 K=1,NFAC2
               IF ((K==1).AND.(NFAC2==1)) THEN
! The first (and only) pass swaps between different portions of WORK and
! rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC2>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   80       CONTINUE
         END IF
         LA=1
!
!     Perform the transform passes needed to transform third dimension
!     (which is currently 'first' dimension), one pass for each factor:
!
         INCL=1
         INCN=ID3
         L=N3
         N=ID1*ID2/2
         INCLO=ID1*ID2
         INCNO=2
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 90 K=1,NFAC3
               IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps from WORK to DATA and rearranges order
! of data to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                        DATA(1),DATA(2),TRIG3(1), &
     &                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC3) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
   90       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 100 K=1,NFAC3
! The first (and only) pass swaps from WORK to DATA and rearranges order
! of data to shift the role of the three dimensions cyclically:
               IF ((K==1).AND.(NFAC3==1)) THEN
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                        DATA(1),DATA(2),TRIG3(1), &
     &                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                                 INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K==NFAC3) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  100       CONTINUE
         END IF
!***********************************************************************
!                                                                      *
!     Here comes the complex hermite --> real FFT ...                  *
!                                                                      *
!***********************************************************************
      ELSE
!
!     Perform the transform passes needed to transform third dimension,
!     one pass for each factor:
!
         INCL=ID1*ID2
         INCN=2
         L=N3
         N=ID1*ID2/2
         INCLO=1
         INCNO=ID3
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 150 K=1,NFAC3
               IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATA to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
                 CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  150       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 160 K=1,NFAC3
               IF ((K==1).AND.(NFAC3>1)) THEN
! The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==1).AND.(NFAC3==1)) THEN
! The first (and only) pass swaps DATA to WORK and rearranges order of
! data to shift the role of the three dimensions cyclically:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123), &
     &                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K==NFAC3).AND.(NFAC3>1)) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  160       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
!
!     Perform the transform passes needed to transform second dimension,
!     (which is currently 'third' dimension), one pass for each factor:
!
         INCL=ID1*ID3/2
         INCN=1
         L=N2
         N=ID1*ID3/2
         INCLO=1
         INCNO=ID2
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 170 K=1,NFAC2
               IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  170       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 180 K=1,NFAC2
               IF (K==NFAC2) THEN
! The last pass swaps between different portions of WORK and rearranges
! order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  180       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
!
!     Perform the transform passes needed to transform first dimension
!     (which is currently 'third' dimension), one pass for each factor:
!
         INCL=ID2*ID3
         INCN=1
         L=N1/2
         N=ID2*ID3
         INCLO=2
         INCNO=ID1
!
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we do on the real -> hermite pass):
!
         NEWOFF=2*ID123-IWOFF
         CALL RCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123), &
     &              TRIG1(1),N,N,1,1,N,L,RSIGN)
         IWOFF=NEWOFF
! And now the transform:
         IF (ISIGN>=0) THEN
! 'Forward transform' using exp(+i*phase)-factors:
            DO 190 K=1,NFAC1
               IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  190       CONTINUE
         ELSE
! 'Inverse transform' using exp(-i*phase)-factors:
            DO 200 K=1,NFAC1
               IF (K==NFAC1) THEN
! The last pass swaps between WORK and DATA and rearranges order of data
! to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1), &
     &              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
! All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+ID123), &
     &                            WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1), &
     &                            N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  200       CONTINUE
         END IF
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
      END IF
      RETURN
      END

      SUBROUTINE RHFFTM(AR,WORK,TRIGS,IFAC,INCLR,INCN,LR,N,ISIGN, &
     &                  MCACHE)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! This subroutine performs R eal or H ermitian 1-D F ast F ourier      *
! T ransforms on M ultiple data sets (doing the 1-D transforms in      *
! parallel). Routine RHFFTM uses the TEMPERTON/FULKER FFT-kernel       *
! routines FPASSM/IPASSM and post-/preprocessing routines RCOMB/HCOMB. *
! Maximum performance on vector processors will be achieved for INCN=1 *
! or at least odd values for INCN and if possible odd values for INCL. *
! Before using CFFTM the first time one must initialise trigonometric  *
! tables and factor tables by calling routine RFTTAB. The output data  *
! will overwrite the input data, the output is   n o t   normalized!   *
! Arguments:                                                           *
!   AR: array containing the real (and imaginary) parts of the data    *
!   WORK: work array of length 2*LR*N                                  *
!   LR: is the length of the transformations (lr = 2**p * 3**q * 5**r) *
!   INCLR: addressing increment between elements of one data set       *
!   N: is the number of data sets to be transformed                    *
!   INCN: is the addressing increment between the starting elements of *
!         consecutive data sets                                        *
!   ISIGN: >=0: forward tranform, <0: inverse transform                *
!   TRIGS: array of length 2*LR containing all sine and cosine factors *
!   IFAC: array of lenght 19 containing the factors contained in LR    *
!                                                                      *
! INFORMATION: Routine RHFFTM is an equivalent to routine RFFTMLT from *
!              the CRAY SCIentific LIBrary (SCILIB)! But it should be  *
!              noted that RHFFTM supports some additional features not *
!              being supported by SCILIB routine RFFTMLT!              *
!                                                                      *
!***********************************************************************

      DIMENSION AR((N-1)*INCN+(LR+1)*INCLR+1)
      DIMENSION WORK(2*LR*N),TRIGS(2*LR),IFAC(19)
      DATA IWOFF /0/, MAXOLD /0/
      SAVE IWOFF,MAXOLD
! no data return immediately
      IF (N.EQ.0) RETURN
! IFAC(1) contains the number of factors
      NFAC=IFAC(1)
! We need a complex transform of length (LR/2) with some tricky
! re-interpretation of the originally real data as complex data:
      INCL=INCLR*2
      L=LR/2

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),MINLOOP))
      IF (MCACHE==0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      IF ((MAXDAT*L)/=MAXOLD) THEN
        IF (IWOFF/=0) IWOFF=2*MAXDAT*L
        IF (IWOFF>(2*MAXOLD)) IWOFF=0
        MAXOLD=MAXDAT*L
      ENDIF
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      LA=1
! We support two storages modes: 'real' and 'complex', 'complex' refers
! to the behaviour of SCILIB-routine RFFTMLT, 'real' goes beyond the
! capabilities of RFFTMLT ... ! Mode 'real' is used for ABS(ISIGN)>=10,
! 'complex' mode else ... :
      IF (ABS(ISIGN)<10) THEN
         INCHER=INCL
         IOFHER=INCLR
      ELSE
         INCHER=INCLR
         IOFHER=(L+1)*INCLR
      END IF
! We first treat the normal case (forward tranform for real data and
! inverse transform for the hermitian conjugate data set ...). The
! opposite case (forward transform for hermitian conjugate data and
! inverse transformation of the corresponding real data sets) shall
! be selected if ISIGN is larger equal 20 or smaller equal -20 (then
! - see above - 'real' storage mode is used) or if ISIGN is in the
! range 2 ... 9 or -9 ... -2 (then complex storage mode is used) ... :
      IF ((ABS(ISIGN)>=20).OR. &
     &            ((ABS(ISIGN)>1).AND.(ABS(ISIGN)<10))) GOTO 99999
! Note that SCILIB routine RFFTMLT does not support the second case,
! it is an other extension going beyond the capabilities of RFFTMLT!
! ---> SO USE ISIGN=1 OR -1 TO GET EXACTLY THE BEHAVIOUR OF RFFTMLT !!
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN>=0) THEN
         RSIGN=1._q
! 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR to WORK:
               CALL FPASSM(AR(I0),AR(I0+INCLR), &
     &                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
! Now we have some complex data sets containing the requested result in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0), &
     &            AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=-1._q
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we have done during the forward FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER), &
     &              WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K==NFAC) THEN
! The last pass swaps WORK to AR:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AR(I0+INCLR), &
     &                     TRIGS(1),N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
! At this point the final transform result is stored in AR, bye ...
      GOTO 10000
99999 CONTINUE
!
!     Perform the transform passes, one pass for each factor:
!
      IF (ISIGN<0) THEN
         RSIGN=-1._q
! 'Inverse transform' using exp(-i*phase)-factors:
         DO 150 K=1,NFAC
            IF (K==1) THEN
! The first pass swaps AR to WORK:
               CALL IPASSM(AR(I0),AR(I0+INCLR), &
     &                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL IPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  150    CONTINUE
! Now we have some complex data sets containing the requested resulgt in
! some 'schizophrenic' way which requires a further post-processing. The
! post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0), &
     &              AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=1._q
! We have to rearrange the hermitian conjugate sequence in some way
! which is suitable to get the real data by some complex transform of
! half length (as inversion of what we have done during the inverse FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER), &
     &              WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
! 'Forward transform' using exp(+i*phase)-factors:
         DO 160 K=1,NFAC
            IF (K==NFAC) THEN
! The last pass swaps WORK to AR:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                     AR(I0),AR(I0+INCLR),TRIGS(1), &
     &                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
! All other passes swap between different portions of WORK:
               NEWOFF=2*MAXDAT*L-IWOFF
               CALL FPASSM_1_1(WORK(1+IWOFF),WORK(1+IWOFF+N0*L), &
     &                         WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L), &
     &                         TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  160    CONTINUE
      END IF
10000 CONTINUE
! At this point the final transform result is stored in AR, bye ...
      RETURN
      END

!=======================================================================
! END CODE MODIFICATIONS --- EXPERIMENTAL (no warranty)
!=======================================================================
#endif


      SUBROUTINE HCOMB(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,RSIGN)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! HCOMB is the post-processing routine performing the final data       *
! arrangement  for a real to hermitian conjugate transform.            *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   RSIGN is the type of transformation (1: forward, -1: inverse)      *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
!
      IBASEL=1
      JBASEL=1
      IBASEH=1+(N-1)*INC1
      JBASEH=1+N*INC2
! First (0'th) and last (N'th) data point treated seperately here:
      IL=1
      JL=1
      JH=JBASEH
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO 10 IJK=1,LOT
         C(JL)=A(IL)+B(IL)
         D(JL)=0._q
         C(JH)=A(IL)-B(IL)
         D(JH)=0._q
         IL=IL+INC3
         JL=JL+INC4
         JH=JH+INC4
   10 CONTINUE
! Now we treat the n'th and the N-n'th data point simultaneously:
      IBASEL=1+INC1
      JBASEL=1+INC2
      JBASEH=JBASEH-INC2
      DO 100 K=2,N-1,2
         CO=0.5_q*TRIGS(2*N+K+1)
         SI=0.5_q*TRIGS(2*N+K+2)*RSIGN
         IL=IBASEL
         JL=JBASEL
         IH=IBASEH
         JH=JBASEH
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
         DO 50 IJK=1,LOT
            C(JL)=0.5_q*(A(IL)+A(IH))+CO*(B(IL)+B(IH))+SI*(A(IL)-A(IH))
            D(JL)=0.5_q*(B(IL)-B(IH))+SI*(B(IL)+B(IH))-CO*(A(IL)-A(IH))
            C(JH)=0.5_q*(A(IL)+A(IH))-CO*(B(IL)+B(IH))-SI*(A(IL)-A(IH))
            D(JH)=0.5_q*(B(IH)-B(IL))+SI*(B(IL)+B(IH))-CO*(A(IL)-A(IH))
            IL=IL+INC3
            JL=JL+INC4
            IH=IH+INC3
            JH=JH+INC4
   50    CONTINUE
         IBASEL=IBASEL+INC1
         JBASEL=JBASEL+INC2
         IBASEH=IBASEH-INC1
         JBASEH=JBASEH-INC2
  100 CONTINUE
! Finally if N was even there remains the 'central' data point! We must
! Treat this point separately because otherwise vectorisation of the
! previous loop (DO 50 IJK=1,LOT) would not be allowed!
      IF (MOD(N,2)==0) THEN
         IC=IBASEL
         JC=JBASEL
         DO 1000 IJK=1,LOT
            C(JC)=A(IC)
            D(JC)=RSIGN*B(IC)
            IC=IC+INC3
            JC=JC+INC4
 1000    CONTINUE
      END IF
      RETURN
      END


      SUBROUTINE RCOMB(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,RSIGN)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! RCOMB is the preprocessing routine preparing the data arrangement    *
! for a hermitian conjugate to real transform.                         *
! Variables:                                                           *
!   A is the array containing the real parts of the input vector       *
!   B is the array containing the imaginary parts of the input vector  *
!   C is the array containing the real parts of the output vector      *
!   D is the array containing the imaginary parts of the output vector *
!   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
!   INC1 is the increment between elements of one data set in A and B  *
!   INC2 is the increment between elements of one data set in C and D  *
!   INC3 is the addressing increment between data sets in A and B      *
!   INC4 is the addressing increment between data sets in C and D      *
!   LOT is the number of data sets                                     *
!   N is the length of one data set (length of the FFT)                *
!   RSIGN is the type of transformation (1: forward, -1: inverse)      *
!                                                                      *
!***********************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
!
      IBASEL=1
      JBASEL=1
      IBASEH=1+N*INC1
      JBASEH=1+(N-1)*INC2
! First (0'th) and last (N'th) data point treated seperately here:
      IL=1
      JL=1
      IH=IBASEH
      DO 10 IJK=1,LOT
         C(JL)=A(IL)+A(IH)
         D(JL)=A(IL)-A(IH)
         IL=IL+INC3
         JL=JL+INC4
         IH=IH+INC3
   10 CONTINUE
! Now we treat the n'th and the N-n'th data point simultaneously:
      IBASEL=1+INC1
      JBASEL=1+INC2
      IBASEH=IBASEH-INC1
      DO 100 K=2,N-1,2
         CO=TRIGS(2*N+K+1)
         SI=TRIGS(2*N+K+2)*RSIGN
         IL=IBASEL
         JL=JBASEL
         IH=IBASEH
         JH=JBASEH
! Enforce vectorization to get maximum performance (you may vectorize
! this loop but compiler says there   c o u l d   be some recurrence)!
! First directive is for Crays, second for CONVEXes, supply directive
! needed for your own machine if it is not a Cray or a CONVEX ... .
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
         DO 50 IJK=1,LOT
            C(JL)=(A(IL)+A(IH))-CO*(B(IL)+B(IH))-SI*(A(IL)-A(IH))
            D(JL)=(B(IL)-B(IH))-SI*(B(IL)+B(IH))+CO*(A(IL)-A(IH))
            C(JH)=(A(IL)+A(IH))+CO*(B(IL)+B(IH))+SI*(A(IL)-A(IH))
            D(JH)=(B(IH)-B(IL))-SI*(B(IL)+B(IH))+CO*(A(IL)-A(IH))
            IL=IL+INC3
            JL=JL+INC4
            IH=IH+INC3
            JH=JH+INC4
   50    CONTINUE
         IBASEL=IBASEL+INC1
         JBASEL=JBASEL+INC2
         IBASEH=IBASEH-INC1
         JBASEH=JBASEH-INC2
  100 CONTINUE
! Finally if N was even there remains the 'central' data point! We must
! Treat this point separately because otherwise vectorisation of the
! previous loop (DO 50 IJK=1,LOT) would not be allowed!
      IF (MOD(N,2)==0) THEN
         IC=IBASEL
         JC=JBASEL
         DO 1000 IJK=1,LOT
            C(JC)=2._q*A(IC)
            D(JC)=(-2._q)*RSIGN*B(IC)
            IC=IC+INC3
            JC=JC+INC4
 1000    CONTINUE
      END IF
      RETURN
      END


      SUBROUTINE D1ZERO(N1,ID1,LOT,DATA)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
!***********************************************************************
!                                                                      *
! Routine D1ZERO fills the part of an array with dimensions (ID1,LOT)  *
! with zeros which is given by addresses (I,J) with N1<I<=ID1 and with *
! 1<=J<=LOT. This is necessary for some routines (FFT3DC, FFTC3D, ...) *
! to avoid trouble with undefined contents of work arrays (which leads *
! to no errors[!!] - but probably to floating point errors ... [??]).  *
! Variables:                                                           *
! DATA is the data array                                               *
! N1 is the length of the transforms along the first direction         *
! ID1 is the leading dimension of array DATA                           *
! LOT is the number of data sets ("second dimension")                  *
!                                                                      *
!***********************************************************************

      DIMENSION DATA(ID1*LOT)
      IF (N1>=ID1) RETURN
      DO 100 I1=N1+1,ID1
         IA=I1
         DO 200 IJK=1,LOT
            DATA(IA)=0._q
            IA=IA+ID1
  200    CONTINUE
  100 CONTINUE
      RETURN
      END

!=======================================================================
!
!   this routine returns true a NIN is a legal value for the FFT
!
!=======================================================================


      LOGICAL FUNCTION FFTCHK_FURTH(NIN)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (NFACT=4)
      DIMENSION IFACT(NFACT),NCOUNT(NFACT)
      DATA      IFACT /2,3,5,7/
      N=NIN
      DO 100 I=1,NFACT
        NCOUNT(I)=0
  120   NEXT=N/IFACT(I)
        IF (NEXT*IFACT(I)==N) THEN
          N=NEXT
          NCOUNT(I)=NCOUNT(I)+1
          GOTO 120
        ENDIF
  100 CONTINUE
      IF (N==1 .AND. (NCOUNT(1)/=0)) &
     &  THEN
        FFTCHK_FURTH=.TRUE.
      ELSE
        FFTCHK_FURTH=.FALSE.
      ENDIF
      RETURN
      END
