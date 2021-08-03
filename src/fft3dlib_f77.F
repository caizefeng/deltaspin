      SUBROUTINE FFTC3N(DATA,ID1,N1,ID2,N2,ID3,N3,WRK,IFAC,
     C                                     TRIG1,TRIG2,TRIG3,ISIGN,IERR,
     C                                     MCACHE)

************************************************************************
*                                                                      *
* High-performance 3-dimensional complex Fast Fourier Transformation   *
* using special subroutine CFFTML (1-dimensional FFT of a vector of    *
* vectors with parallel processing of all one-dimensional data sets).  *
* This routine returns an unnormed output vector for inverse FFT s!    *
*                                                                      *
************************************************************************

      LOGICAL FACERR,TSET,LPSECT
      DIMENSION DATA(2*ID1,ID2,ID3),WRK(4*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      COMMON /PARALL/ LPSECT
      EXTERNAL FACERR
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
      NCPU=1
* Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN.EQ.0) THEN
* First check errors:
         IF ((ID1.LT.N1).OR.(ID2.LT.N2).OR.(ID3.LT.N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (IERR.NE.0) RETURN
* Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
* Initialisation of the tables using routine CFTTAB.
         IF (NCPU.GT.1) THEN
            CALL CFTTAB(N1,IFAC(1,1),TRIG1(1))
            CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
            CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         ELSE
            CALL CFTTAB(-N1,IFAC(1,1),TRIG1(1))
            CALL CFTTAB(-N2,IFAC(1,2),TRIG2(1))
            CALL CFTTAB(-N3,IFAC(1,3),TRIG3(1))
         ENDIF
         RETURN
      END IF
* Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1.NE.N1SET).OR.(N2.NE.N2SET)
     C                                         .OR.(N3.NE.N3SET)) IERR=3
      IF (IERR.NE.0) RETURN
C-----------------------------------------------------------------------
C  parallel version
C  do the job plane by plane only for the third dimension
C  data are moved around processors
C-----------------------------------------------------------------------
      IF ((.NOT.LPSECT).AND.(NCPU.GT.1)) THEN
C$DOACROSS LOCAL(I),SHARE(DATA,WRK),MP_SCHEDTYPE=SIMPLE
       DO 1000 I=1,N3
         MYTHREAD=0
* Transformation along first dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(4*MYTHREAD*ID1*ID2+1),
     C                          TRIG1(1),IFAC(1,1),2,2*ID1,N1,ID2,ISIGN,
     C                          MCACHE)
* Transformation along second dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(4*MYTHREAD*ID2*ID1+1),
     C                          TRIG2(1),IFAC(1,2),2*ID1,2,N2,ID1,ISIGN,
     C                          MCACHE)
 1000  CONTINUE
* Transformation along third dimension:
       ID12=ID1*ID2
C$DOACROSS LOCAL(I),SHARE(DATA,WRK),MP_SCHEDTYPE=SIMPLE
       DO 2000 I=1,N2
         MYTHREAD=0
         CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(4*MYTHREAD*ID1*ID3+1),
     C                         TRIG3(1),IFAC(1,3),2*ID12,2,N3,ID1,ISIGN,
     C                         MCACHE)
 2000  CONTINUE
      ELSE IF (MCACHE.EQ.0) THEN
C-----------------------------------------------------------------------
C  version which tries to keep as much data in cache as possible
C  doing FFT plane by plane
C  for this version MCACHE is not important except for FFT-lenght > 100
C-----------------------------------------------------------------------
       DO 1100 I=1,N3
* Transformation along first dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),
     &                          TRIG1(1),IFAC(1,1),2,2*ID1,N1,ID2,ISIGN,
     &                          MCACHE)
* Transformation along second dimension:
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),
     &                          TRIG2(1),IFAC(1,2),2*ID1,2,N2,ID1,ISIGN,
     &                          MCACHE)
 1100  CONTINUE
* Transformation along third dimension:
       ID12=ID1*ID2
       DO 2100 I=1,N2
         CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1),
     &                         TRIG3(1),IFAC(1,3),2*ID12,2,N3,ID1,ISIGN,
     &                          MCACHE)
 2100 CONTINUE
      ELSE
C-----------------------------------------------------------------------
C  version which tries to do as much data simultaneaously as possible
C  caching is done within CFFTML
C-----------------------------------------------------------------------
* Transformation along first dimension:
       CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),
     &                     TRIG1(1),IFAC(1,1),2,2*ID1,N1,ID2*ID3,ISIGN,
     &                     MCACHE)
* Transformation along second dimension: must be splitted
       DO 2200 I=1,N3
         CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),
     &                     TRIG2(1),IFAC(1,2),2*ID1,2,N2,ID1,ISIGN,
     &                     MCACHE)
 2200 CONTINUE
* Transformation along third dimension:
      ID12=ID1*ID2
      CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1),IFAC(1,3),
     &                                           2*ID12,2,N3,ID12,ISIGN,
     &                                           MCACHE)
      ENDIF
      RETURN
      END


      SUBROUTINE CFFTML(AR,AI,WORK,TRIGS,IFAC,INCL,INCN,L,N,ISIGN,
     &                  MCACHE)

************************************************************************
*                                                                      *
* This subroutine performs C omplex 1-D F ast F ourier T ransforms on  *
* M ultiple data sets (doing the 1-D transforms in parallel). Routine  *
* CFFTML uses the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM.  *
*                                                                      *
************************************************************************

      DIMENSION AR((N-1)*INCN+(L-1)*INCL+1),AI((N-1)*INCN+(L-1)*INCL+1)
      DIMENSION WORK(4*L*N),TRIGS(2*L),IFAC(19)

* IFAC(1) contains the number of factors
      NFAC=IFAC(1)

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),1))
* The following is somehow "empirical": for two factor we use effectively
* only half of WORK --> MAXDAT from above is "optimal", for three factors
* 2/3 of it would be better, for more factors = longer transforms some
* few cache misses add usually a lower penalty than the 10000-loop which
* will be longer and longer with shorter and shorter inner loop (balance
* between loop-startup time and memory access time ... --> ???????? :-)
      IF (NFAC.EQ.3) MAXDAT=MAXDAT/1.5
      IF (MCACHE.EQ.0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      IWOFF=0
      LA=1
* Special case NFAC=1:
      IF (NFAC.EQ.1) THEN
* Use routine SPASSM, nothing will be swapped (SPASSM swaps internally):
         CALL SPASSM(AR(I0),AI(I0),WORK(1),WORK(1+N0*L),
     C                                             INCL,INCN,N0,L,ISIGN)
* At this point the final transform result is stored in AR/AI, bye ...
         GOTO 10000
      END IF
*
*     Perform the transform passes, one pass for each factor:
*
      IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K.EQ.1) THEN
* The first pass swaps AR/AI to WORK:
               CALL FPASSM(AR(I0),AI(I0),WORK(1+IWOFF),
     C                     WORK(1+IWOFF+N0*L),TRIGS(1),
     C                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
           ELSE IF (K.EQ.NFAC) THEN
* The last pass swaps WORK to AR/AI:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     AR(I0),AI(I0),TRIGS(1),
     C                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
      ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K.EQ.1) THEN
* The first pass swaps AR/AI to WORK:
               CALL IPASSM(AR(I0),AI(I0),WORK(1+IWOFF),
     C                     WORK(1+IWOFF+N0*L),TRIGS(1),
     C                     INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE IF (K.EQ.NFAC) THEN
* The last pass swaps WORK to AR/AI:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     AR(I0),AI(I0),TRIGS(1),
     C                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
10000 CONTINUE
* At this point the final transform result is stored in AR/AI, bye ...
      RETURN
      END


      SUBROUTINE FFTC3V(DATC,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC,
     C                                     TRIG1,TRIG2,TRIG3,ISIGN,IERR)

************************************************************************
*                                                                      *
* High-performance 3-dimensional complex Fast Fourier Transformation   *
* using the TEMPERTON/FULKER FFT-kernel routines FPASSM/IPASSM. This   *
* routine uses complex storage mode for input and output, real storage *
* mode on work array hopefully speeding up the whole at least for long *
* transforms with many factors (hence many operations on work array).  *
*                                                                      *
* This routine is specially optimized for vector machines -- never use *
* this routine on "small workstations"!!!! (no efficient use of cache) *
*                                                                      *
************************************************************************

      LOGICAL FACERR,TSET
      DIMENSION DATC(2*ID1*ID2*ID3),WORK(4*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
* Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN.EQ.0) THEN
* First check errors:
         IF ((ID1.LT.N1).OR.(ID2.LT.N2).OR.(ID3.LT.N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (IERR.NE.0) RETURN
* Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
* Initialisation of the tables using routine CFTTAB.
         CALL CFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
* Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1.NE.N1SET).OR.(N2.NE.N2SET)
     C                                         .OR.(N3.NE.N3SET)) IERR=3
      IF (IERR.NE.0) RETURN
* Extract numbers of factors and preset some variables:
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3
      IWOFF=0
      LA=1
*
*     Perform the transform passes needed to transform third dimension,
*     one pass for each factor:
*
      INCL=2*ID1*ID2
      INCN=2
      L=N3
      N=ID1*ID2
      INCLO=1
      INCNO=ID3
      IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC3
            IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps DATC to WORK:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123),
     C                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
* The first (and only) pass swaps DATC to WORK and rearranges order of
* data to shift the role of the three dimensions cyclically:
               CALL FPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123),
     C                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K.EQ.NFAC3).AND.(NFAC3.GT.1)) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   50    CONTINUE
      ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC3
            IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps DATC to WORK:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123),
     C                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
* The first (and only) pass swaps DATC to WORK and rearranges order of
* data to shift the role of the three dimensions cyclically:
               CALL IPASSM(DATC(1),DATC(2),WORK(1),WORK(1+ID123),
     C                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
            ELSE IF ((K.EQ.NFAC3).AND.(NFAC3.GT.1)) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,3)
   60    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
*
*     Perform the transform passes needed to transform second dimension,
*     (which is currently 'third' dimension), one pass for each factor:
*
      INCL=ID1*ID3
      INCN=1
      L=N2
      N=ID1*ID3
      INCLO=1
      INCNO=ID2
      IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
         DO 70 K=1,NFAC2
            IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   70    CONTINUE
      ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 80 K=1,NFAC2
            IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,2)
   80    CONTINUE
      END IF
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
      CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
      LA=1
*
*     Perform the transform passes needed to transform first dimension
*     (which is currently 'third' dimension), one pass for each factor:
*
      INCL=ID2*ID3
      INCN=1
      L=N1
      N=ID2*ID3
      INCLO=2
      INCNO=2*ID1
      IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
         DO 90 K=1,NFAC1
            IF (K.EQ.NFAC1) THEN
* The last pass swaps between WORK and DATC and rearranges order of data
* to shift the role of the three dimensions cyclically:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1),
     C              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
   90    CONTINUE
      ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 100 K=1,NFAC1
            IF (K.EQ.NFAC1) THEN
* The last pass swaps between WORK and DATC and rearranges order of data
* to shift the role of the three dimensions cyclically:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATC(1),
     C              DATC(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*ID123-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1,1)
  100    CONTINUE
      END IF
      RETURN
      END


      SUBROUTINE FFTCRN(DATA,ID1,N1,ID2,N2,ID3,N3,WRK,IFAC,
     C                               TRIG1,TRIG2,TRIG3,ISIGN,IHERM,IERR,
     C                               MCACHE)

************************************************************************
*                                                                      *
* High-performance 3-dim. real <-> hermite Fast Fourier Transformation *
* using my library subroutine RHFFTM (1-dimensional FFT of a vector of *
* vectors with parallel processing of all one-dimensional data sets).  *
* The real <-> hermite pass is made along the first dimension!         *
*                                                                      *
************************************************************************

      LOGICAL FACERR,FACODD,TSET
      DIMENSION DATA(ID1,ID2,ID3),WRK(2*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR,FACODD
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
* Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN.EQ.0) THEN
* First check errors:
         IF ((ID1.LT.(N1+2)).OR.(ID2.LT.N2).OR.(ID3.LT.N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (FACODD(N1)) IERR=4
         IF (IERR.NE.0) RETURN
* Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
* Initialisation of the tables using routines CFTTAB/RFTTAB.
         CALL RFTTAB(-N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(-N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(-N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
* Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1.NE.N1SET).OR.(N2.NE.N2SET)
     C                                         .OR.(N3.NE.N3SET)) IERR=3
      IF (IERR.NE.0) RETURN
      N1HP1=N1/2+1
      IF (MCACHE.EQ.0) THEN
C-----------------------------------------------------------------------
C  version which tries to keep as much data in cache as possible
C  doing FFT plane by plane
C  for this version MCACHE is not important except for FFT-lenght > 100
C-----------------------------------------------------------------------
* Here the real to hermite transformations ...:
      IF (IHERM.GE.0) THEN
         IF (ISIGN.GT.0) ITRANS=1
         IF (ISIGN.LT.0) ITRANS=-2
         DO 1000 I=1,N3
* Transformation along first dimension ('real to hermite pass'):
            CALL RHFFTM(DATA(1,1,I),WRK(1),TRIG1(1),
     C                                    IFAC(1,1),1,ID1,N1,ID2,ITRANS,
     C                                    MCACHE)
* Transformation along second dimension:
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1),
     C                                   IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,
     C                                   MCACHE)
 1000    CONTINUE
* Transformation along third dimension:
         DO 1500 I=1,N2
            CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1),TRIG3(1),
     C                               IFAC(1,3),ID1*ID2,2,N3,N1HP1,ISIGN,
     C                               MCACHE)
 1500    CONTINUE
* Here the hermite to real transformations ...:
      ELSE
         IF (ISIGN.GT.0) ITRANS=2
         IF (ISIGN.LT.0) ITRANS=-1
* Transformation along third dimension:
         DO 2500 I=1,N2
            CALL CFFTML(DATA(1,I,1),DATA(2,I,1),WRK(1),TRIG3(1),
     C                               IFAC(1,3),ID1*ID2,2,N3,N1HP1,ISIGN,
     C                               MCACHE)
 2500    CONTINUE
         DO 2000 I=1,N3
* Transformation along second dimension:
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1),
     C                                   IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,
     C                                   MCACHE)
* Transformation along first dimension ('hermite to real pass'):
            CALL RHFFTM(DATA(1,1,I),WRK(1),TRIG1(1),
     C                                    IFAC(1,1),1,ID1,N1,ID2,ITRANS,
     C                                    MCACHE)
 2000    CONTINUE
       END IF
      ELSE
C-----------------------------------------------------------------------
C  version which tries to do as much data simultaneaously as possible
C  caching is done within CFFTML
C-----------------------------------------------------------------------
* Here the real to hermite transformations ...:
      IF (IHERM.GE.0) THEN
         IF (ISIGN.GT.0) ITRANS=1
         IF (ISIGN.LT.0) ITRANS=-2
* Transformation along first dimension ('real to hermite pass'):
         CALL RHFFTM(DATA(1,1,1),WRK(1),TRIG1(1),
     C                                IFAC(1,1),1,ID1,N1,ID2*ID3,ITRANS,
     C                                MCACHE)
         DO 100 I=1,N3
* Transformation along second dimension: must be splitted
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1),
     C                                   IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,
     C                                   MCACHE)
 100     CONTINUE
* Transformation along third dimension:
         CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1),
     C                           IFAC(1,3),ID1*ID2,2,N3,N1HP1*ID2,ISIGN,
     C                           MCACHE)
* Here the hermite to real transformations ...:
      ELSE
         IF (ISIGN.GT.0) ITRANS=2
         IF (ISIGN.LT.0) ITRANS=-1
* Transformation along third dimension:
         CALL CFFTML(DATA(1,1,1),DATA(2,1,1),WRK(1),TRIG3(1),
     C                           IFAC(1,3),ID1*ID2,2,N3,N1HP1*ID2,ISIGN,
     C                           MCACHE)
         DO 200 I=1,N3
* Transformation along second dimension: must be splitted
            CALL CFFTML(DATA(1,1,I),DATA(2,1,I),WRK(1),TRIG2(1),
     C                                   IFAC(1,2),ID1,2,N2,N1HP1,ISIGN,
     C                                   MCACHE)
 200     CONTINUE
* Transformation along first dimension ('hermite to real pass'):
         CALL RHFFTM(DATA(1,1,1),WRK(1),TRIG1(1),
     C                                IFAC(1,1),1,ID1,N1,ID2*ID3,ITRANS,
     C                                MCACHE)
      END IF
      END IF
      RETURN
      END

      SUBROUTINE FFTR3V(DATA,ID1,N1,ID2,N2,ID3,N3,WORK,IFAC,
     C                               TRIG1,TRIG2,TRIG3,ISIGN,IHERM,IERR)

************************************************************************
*                                                                      *
* High-performance 3-dimensional real <--> first dimension hermitian   *
* Fast Fourier Transformation using the TEMPERTON/FULKER FFT-kernel    *
* routines FPASSM/IPASSM. This routine uses complex storage mode for   *
* hermitian input and output and real storage mode on the work array   *
* hopefully speeding up the whole at least for long transforms with    *
* many factors (hence many operations on work array).                  *
*                                                                      *
* This routine is specially optimized for vector machines -- never use *
* this routine on "small workstations"!!!! (no efficient use of cache) *
*                                                                      *
************************************************************************

      LOGICAL FACERR,FACODD,TSET
      DIMENSION DATA(ID1*ID2*ID3),WORK(2*ID1*ID2*ID3)
      DIMENSION TRIG1(2*ID1),TRIG2(2*ID2),TRIG3(2*ID3),IFAC(19,3)
      EXTERNAL FACERR,FACODD
      SAVE TSET,N1SET,N2SET,N3SET
      DATA TSET /.FALSE./
      IERR=0
* Initialisation-branch for trigonometric tables and factor tables:
      IF (ISIGN.EQ.0) THEN
* First check errors:
         IF ((ID1.LT.(N1+2)).OR.(ID2.LT.N2).OR.(ID3.LT.N3)) IERR=1
         IF ((FACERR(N1)).OR.(FACERR(N2)).OR.(FACERR(N3))) IERR=2
         IF (FACODD(N1)) IERR=4
         IF (IERR.NE.0) RETURN
* Remember THAT we have intialized and WHAT we have initialized!
         TSET=.TRUE.
         N1SET=N1
         N2SET=N2
         N3SET=N3
* Initialisation of the tables using routine CFTTAB.
         CALL RFTTAB(N1,IFAC(1,1),TRIG1(1))
         CALL CFTTAB(N2,IFAC(1,2),TRIG2(1))
         CALL CFTTAB(N3,IFAC(1,3),TRIG3(1))
         RETURN
      END IF
* Not initialised or wrong initialised!
      IF ((.NOT.TSET).OR.(N1.NE.N1SET).OR.(N2.NE.N2SET)
     C                                         .OR.(N3.NE.N3SET)) IERR=3
      IF (IERR.NE.0) RETURN
* Extract numbers of factors and preset some variables:
      RSIGN=ISIGN
      NFAC1=IFAC(1,1)
      NFAC2=IFAC(1,2)
      NFAC3=IFAC(1,3)
      ID123=ID1*ID2*ID3/2
      IWOFF=0
      LA=1

************************************************************************
*                                                                      *
*     Here comes the real --> complex hermite FFT ...                  *
*                                                                      *
************************************************************************
      IF (IHERM.GE.0) THEN
*
*     Perform the transform passes needed to transform first dimension,
*     one pass for each factor:
*
         INCL=2
         INCN=ID1
         L=N1/2
         N=ID2*ID3
         INCLO=ID2*ID3
         INCNO=1
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 50 K=1,NFAC1
               IF (K.EQ.1) THEN
* The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   50       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 60 K=1,NFAC1
               IF (K.EQ.1) THEN
* The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                        TRIG1(1),INCL,N,INCN,1,N,L,IFAC(K+1,1),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
   60       CONTINUE
         END IF
         LA=1
*
* Now we have some complex data sets containing the requested result in
* some 'schizophrenic' way which requires a further post-processing. The
* post-processing phase swaps between portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
*
         NEWOFF=2*ID123-IWOFF
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),
     C              TRIG1(1),N,INCLO,1,INCNO,N,L,RSIGN)
         IWOFF=NEWOFF
*
*     Perform the transform passes needed to transform second dimension,
*     (which is currently 'first' dimension), one pass for each factor:
*
         INCL=1
         INCN=ID2
         L=N2
         N=ID1*ID3/2
         INCLO=ID1*ID3/2
         INCNO=1
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 70 K=1,NFAC2
               IF ((K.EQ.1).AND.(NFAC2.EQ.1)) THEN
* The first (and only) pass swaps between different portions of WORK and
* rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K.EQ.1).AND.(NFAC2.GT.1)) THEN
* The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   70       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 80 K=1,NFAC2
               IF ((K.EQ.1).AND.(NFAC2.EQ.1)) THEN
* The first (and only) pass swaps between different portions of WORK and
* rearranges data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                         INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K.EQ.1).AND.(NFAC2.GT.1)) THEN
* The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                 INCL,N,INCN,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
   80       CONTINUE
         END IF
         LA=1
*
*     Perform the transform passes needed to transform third dimension
*     (which is currently 'first' dimension), one pass for each factor:
*
         INCL=1
         INCN=ID3
         L=N3
         N=ID1*ID2/2
         INCLO=ID1*ID2
         INCNO=2
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 90 K=1,NFAC3
               IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
* The first (and only) pass swaps from WORK to DATA and rearranges order
* of data to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                        DATA(1),DATA(2),TRIG3(1),
     C                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                 INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K.EQ.NFAC3) THEN
* The last pass swaps between WORK and DATA and rearranges order of data
* to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1),
     C              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
   90       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 100 K=1,NFAC3
* The first (and only) pass swaps from WORK to DATA and rearranges order
* of data to shift the role of the three dimensions cyclically:
               IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                        DATA(1),DATA(2),TRIG3(1),
     C                        INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                           INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE IF (K.EQ.NFAC3) THEN
* The last pass swaps between WORK and DATA and rearranges order of data
* to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1),
     C              DATA(2),TRIG3(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  100       CONTINUE
         END IF
************************************************************************
*                                                                      *
*     Here comes the complex hermite --> real FFT ...                  *
*                                                                      *
************************************************************************
      ELSE
*
*     Perform the transform passes needed to transform third dimension,
*     one pass for each factor:
*
         INCL=ID1*ID2
         INCN=2
         L=N3
         N=ID1*ID2/2
         INCLO=1
         INCNO=ID3
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 150 K=1,NFAC3
               IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps DATA to WORK:
                  CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
* The first (and only) pass swaps DATA to WORK and rearranges order of
* data to shift the role of the three dimensions cyclically:
                 CALL FPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K.EQ.NFAC3).AND.(NFAC3.GT.1)) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  150       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 160 K=1,NFAC3
               IF ((K.EQ.1).AND.(NFAC3.GT.1)) THEN
* The first pass swaps DATA to WORK:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                        TRIG3(1),INCL,N,INCN,1,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K.EQ.1).AND.(NFAC3.EQ.1)) THEN
* The first (and only) pass swaps DATA to WORK and rearranges order of
* data to shift the role of the three dimensions cyclically:
                  CALL IPASSM(DATA(1),DATA(2),WORK(1),WORK(1+ID123),
     C                TRIG3(1),INCL,INCLO,INCN,INCNO,N,L,IFAC(K+1,3),LA)
               ELSE IF ((K.EQ.NFAC3).AND.(NFAC3.GT.1)) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG3(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,3),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,3)
  160       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
*
*     Perform the transform passes needed to transform second dimension,
*     (which is currently 'third' dimension), one pass for each factor:
*
         INCL=ID1*ID3/2
         INCN=1
         L=N2
         N=ID1*ID3/2
         INCLO=1
         INCNO=ID2
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 170 K=1,NFAC2
               IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  170       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 180 K=1,NFAC2
               IF (K.EQ.NFAC2) THEN
* The last pass swaps between different portions of WORK and rearranges
* order of data to shift the role of the three dimensions cyclically:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                               N,INCLO,1,INCNO,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG2(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,2),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,2)
  180       CONTINUE
         END IF
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF))
         CALL D1ZERO(L,INCNO,N,WORK(1+IWOFF+ID123))
         LA=1
*
*     Perform the transform passes needed to transform first dimension
*     (which is currently 'third' dimension), one pass for each factor:
*
         INCL=ID2*ID3
         INCN=1
         L=N1/2
         N=ID2*ID3
         INCLO=2
         INCNO=ID1
*
* We have to rearrange the hermitian conjugate sequence in some way
* which is suitable to get the real data by some complex transform of
* half length (as inversion of what we do on the real -> hermite pass):
*
         NEWOFF=2*ID123-IWOFF
         CALL RCOMB(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C              WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),
     C              TRIG1(1),N,N,1,1,N,L,RSIGN)
         IWOFF=NEWOFF
* And now the transform:
         IF (ISIGN.GE.0) THEN
* 'Forward transform' using exp(+i*phase)-factors:
            DO 190 K=1,NFAC1
               IF (K.EQ.NFAC1) THEN
* The last pass swaps between WORK and DATA and rearranges order of data
* to shift the role of the three dimensions cyclically:
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1),
     C              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  190       CONTINUE
         ELSE
* 'Inverse transform' using exp(-i*phase)-factors:
            DO 200 K=1,NFAC1
               IF (K.EQ.NFAC1) THEN
* The last pass swaps between WORK and DATA and rearranges order of data
* to shift the role of the three dimensions cyclically:
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),DATA(1),
     C              DATA(2),TRIG1(1),N,INCLO,1,INCNO,N,L,IFAC(K+1,1),LA)
               ELSE
* All other passes swap between different portions of WORK:
                  NEWOFF=2*ID123-IWOFF
                  CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+ID123),
     C                   WORK(1+NEWOFF),WORK(1+NEWOFF+ID123),TRIG1(1),
     C                                       N,N,1,1,N,L,IFAC(K+1,1),LA)
                  IWOFF=NEWOFF
               END IF
               LA=LA*IFAC(K+1,1)
  200       CONTINUE
         END IF
         CALL D1ZERO(N1,ID1,(ID2*ID3),DATA(1))
      END IF

      RETURN
      END



      SUBROUTINE RHFFTM(AR,WORK,TRIGS,IFAC,INCLR,INCN,LR,N,ISIGN,
     &                  MCACHE)

************************************************************************
*                                                                      *
* This subroutine performs R eal or H ermitian 1-D F ast F ourier      *
* T ransforms on M ultiple data sets (doing the 1-D transforms in      *
* parallel). Routine RHFFTM uses the TEMPERTON/FULKER FFT-kernel       *
* routines FPASSM/IPASSM and post-/preprocessing routines RCOMB/HCOMB. *
* Maximum performance on vector processors will be achieved for INCN=1 *
* or at least odd values for INCN and if possible odd values for INCL. *
* Before using CFFTM the first time one must initialise trigonometric  *
* tables and factor tables by calling routine RFTTAB. The output data  *
* will overwrite the input data, the output is   n o t   normalized!   *
* Arguments:                                                           *
*   AR: array containing the real (and imaginary) parts of the data    *
*   WORK: work array of length 2*LR*N                                  *
*   LR: is the length of the transformations (lr = 2**p * 3**q * 5**r) *
*   INCLR: addressing increment between elements of one data set       *
*   N: is the number of data sets to be transformed                    *
*   INCN: is the addressing increment between the starting elements of *
*         consecutive data sets                                        *
*   ISIGN: >=0: forward tranform, <0: inverse transform                *
*   TRIGS: array of length 2*LR containing all sine and cosine factors *
*   IFAC: array of lenght 19 containing the factors contained in LR    *
*                                                                      *
* INFORMATION: Routine RHFFTM is an equivalent to routine RFFTMLT from *
*              the CRAY SCIentific LIBrary (SCILIB)! But it should be  *
*              noted that RHFFTM supports some additional features not *
*              being supported by SCILIB routine RFFTMLT!              *
*                                                                      *
************************************************************************

      DIMENSION AR((N-1)*INCN+(LR+1)*INCLR+1)
      DIMENSION WORK(2*LR*N),TRIGS(2*LR),IFAC(19)

* IFAC(1) contains the number of factors
      NFAC=IFAC(1)
* We need a complex transform of length (LR/2) with some tricky
* re-interpretation of the originally real data as complex data:
      INCL=INCLR*2
      L=LR/2

      MAXDAT=MIN(N,MAX(((MCACHE-32-2*L)/(4*L)),1))
* The following is somehow "empirical": for two factor we use effectively
* only half of WORK --> MAXDAT from above is "optimal", for three factors
* 2/3 of it would be better, for more factors = longer transforms some
* few cache misses add usually a lower penalty than the 10000-loop which
* will be longer and longer with shorter and shorter inner loop (balance
* between loop-startup time and memory access time ... --> ???????? :-)
      IF (NFAC.EQ.3) MAXDAT=MAXDAT/1.5
      IF (MCACHE.EQ.0) MAXDAT=N
      NBLOCK=(N+MAXDAT-1)/MAXDAT
      MAXDAT=(N+NBLOCK-1)/NBLOCK
      DO 10000 IBLOCK=0,NBLOCK-1
      I0=IBLOCK*MAXDAT*INCN+1
      N0=MIN(MAXDAT,N-IBLOCK*MAXDAT)

      IWOFF=0
      LA=1
* We support two storages modes: 'real' and 'complex', 'complex' refers
* to the behaviour of SCILIB-routine RFFTMLT, 'real' goes beyond the
* capabilities of RFFTMLT ... ! Mode 'real' is used for ABS(ISIGN)>=10,
* 'complex' mode else ... :
      IF (ABS(ISIGN).LT.10) THEN
         INCHER=INCL
         IOFHER=INCLR
      ELSE
         INCHER=INCLR
         IOFHER=(L+1)*INCLR
      END IF
* We first treat the normal case (forward tranform for real data and
* inverse transform for the hermitian conjugate data set ...). The
* opposite case (forward transform for hermitian conjugate data and
* inverse transformation of the corresponding real data sets) shall
* be selected if ISIGN is larger equal 20 or smaller equal -20 (then
* - see above - 'real' storage mode is used) or if ISIGN is in the
* range 2 ... 9 or -9 ... -2 (then complex storage mode is used) ... :
      IF ((ABS(ISIGN).GE.20).OR.
     C            ((ABS(ISIGN).GT.1).AND.(ABS(ISIGN).LT.10))) GOTO 99999
* Note that SCILIB routine RFFTMLT does not support the second case,
* it is an other extension going beyond the capabilities of RFFTMLT!
* ---> SO USE ISIGN=1 OR -1 TO GET EXACTLY THE BEHAVIOUR OF RFFTMLT !!
*
*     Perform the transform passes, one pass for each factor:
*
      IF (ISIGN.GE.0) THEN
         RSIGN=1.
* 'Forward transform' using exp(+i*phase)-factors:
         DO 50 K=1,NFAC
            IF (K.EQ.1) THEN
* The first pass swaps AR to WORK:
               CALL FPASSM(AR(I0),AR(I0+INCLR),
     C                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   50    CONTINUE
* Now we have some complex data sets containing the requested result in
* some 'schizophrenic' way which requires a further post-processing. The
* post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0),
     C            AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=-1.
* We have to rearrange the hermitian conjugate sequence in some way
* which is suitable to get the real data by some complex transform of
* half length (as inversion of what we have done during the forward FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER),
     C              WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 60 K=1,NFAC
            IF (K.EQ.NFAC) THEN
* The last pass swaps WORK to AR:
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     AR(I0),AR(I0+INCLR),
     C                     TRIGS(1),N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
   60    CONTINUE
      END IF
* At this point the final transform result is stored in AR, bye ...
      GOTO 10000
99999 CONTINUE
*
*     Perform the transform passes, one pass for each factor:
*
      IF (ISIGN.LT.0) THEN
         RSIGN=-1.
* 'Inverse transform' using exp(-i*phase)-factors:
         DO 150 K=1,NFAC
            IF (K.EQ.1) THEN
* The first pass swaps AR to WORK:
               CALL IPASSM(AR(I0),AR(I0+INCLR),
     C                     WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     TRIGS(1),INCL,N0,INCN,1,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL IPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  150    CONTINUE
* Now we have some complex data sets containing the requested result in
* some 'schizophrenic' way which requires a further post-processing. The
* post-processing phase swaps WORK back to AR ... :
         CALL HCOMB(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),AR(I0),
     C              AR(I0+IOFHER),TRIGS(1),N0,INCHER,1,INCN,N0,L,RSIGN)
      ELSE
         RSIGN=1.
* We have to rearrange the hermitian conjugate sequence in some way
* which is suitable to get the real data by some complex transform of
* half length (as inversion of what we have done during the inverse FFT:
         CALL RCOMB(AR(I0),AR(I0+IOFHER),
     C              WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C              TRIGS(1),INCHER,N0,INCN,1,N0,L,RSIGN)
* 'Forward transform' using exp(+i*phase)-factors:
         DO 160 K=1,NFAC
            IF (K.EQ.NFAC) THEN
* The last pass swaps WORK to AR:
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     AR(I0),AR(I0+INCLR),TRIGS(1),
     C                     N0,INCL,1,INCN,N0,L,IFAC(K+1),LA)
            ELSE
* All other passes swap between different portions of WORK:
               NEWOFF=2*N0*L-IWOFF
               CALL FPASSM(WORK(1+IWOFF),WORK(1+IWOFF+N0*L),
     C                     WORK(1+NEWOFF),WORK(1+NEWOFF+N0*L),
     C                     TRIGS(1),N0,N0,1,1,N0,L,IFAC(K+1),LA)
               IWOFF=NEWOFF
            END IF
            LA=LA*IFAC(K+1)
  160    CONTINUE
      END IF
10000 CONTINUE
* At this point the final transform result is stored in AR, bye ...
      RETURN
      END


      SUBROUTINE CFTTAB(L,IFAC,TRIGS)

************************************************************************
*                                                                      *
* Serves as an interface calling the routines FTFACT and FTRIGC to get *
* the factors of L and trigonometric tables for a Fourier Transform of *
* length L. For details see routines FTFACT and FTRIGC ... .           *
* INFORMATION: Routine CFTTAB is an equivalent to routine CFTFAX from  *
*              the CRAY SCIentific LIBrary (SCILIB)!                   *
*                                                                      *
************************************************************************

      DIMENSION IFAC(19),TRIGS(2*L)
*
      CALL FTFACT(L,IFAC)
      CALL FTRIGC(L,TRIGS)
      RETURN
      END


      SUBROUTINE RFTTAB(L,IFAC,TRIGS)

************************************************************************
*                                                                      *
* Serves as an interface calling the routines FTFACT and FTRIGR to get *
* the factors of L and trigonometric tables for a Fourier Transform of *
* length L. For details see routines FTFACT and FTRIGR ... .           *
* INFORMATION: Routine RFTTAB is an equivalent to routine FFTFAX from  *
*              the CRAY SCIentific LIBrary (SCILIB)!                   *
*                                                                      *
************************************************************************

      DIMENSION IFAC(19),TRIGS(2*L)
* Length of transformation (L) must be even!!!
      IF (MOD(ABS(L),2).NE.0) THEN
         IFAC(1)=-99
         RETURN
      END IF
      LH=L/2
      CALL FTFACT(LH,IFAC)
      CALL FTRIGR(L,TRIGS)
      RETURN
      END


      SUBROUTINE FTFACT(L,IFAC)

************************************************************************
*                                                                      *
* Factorization routine that first extracts all factors of 4, then all *
* factors of 2, all factors of 3, all factors of 5 and (this is some   *
* extension!) all factors of 7 (the allowed maximum number of factors  *
* is 18 factors). On output IFAC(1) contains the number of factors of  *
* L (or -99 in case of error), IFAC(2)...IFAC(IFAC(1)+1) all factors.  *
* One nice feature has been included in this routine which deviates a  *
* little from the conventions used in SCILIB routine CFTFAX: If L<0 we *
* factorize ABS(L) and take some other order of the factors: Instead   *
* of taking first all factors 4, then all factors 2, 3, 5 we rearrange *
* the factors so that the first factor is preferrably 2, 4 if at least *
* two factors 4 are present or if no factor 3 is present or 3 and the  *
* last factor is preferably (2), 4 or 3 ... ! This has a quite simple  *
* reason: The routines CFFTM etc. use for the inner passes which swap  *
* between portions of the WORK array the highest efficient layout for  *
* the data arrangement which garuantees highest performances. But the  *
* first and the last pass are not necessarily as efficient as the all  *
* other passes (depending on the increments ...), hence we must assume *
* that they are 'low performance passes'. Our choice for the factors   *
* tries to minimize the number of operations to be used for these two  *
* passes and hence to minimize the 'slowdown' due to these passes.     *
*                                                                      *
************************************************************************

* The following parameter decides whether to use factors 4 or not!
* The number of necessary floating point operations is always larger
* if we use only factors 2, but on some machines factors 2 might have
* the advantage that less work is done in the last pass what could
* reduce the slowdown if this is a low performance pass -- but it is
* hard to decide whether this advantage compensates the disadvantage
* of a slightly increased operation count ... . I think using factors 4
* by default should in most cases still be the best (test if you like!).
* This makes of course only sense if reordering is used (L<0) ... !
* For non-vector machines I always recommend usage of NOFOUR=0 ... !?
      PARAMETER(NOFOUR=0,LSPLIT=0)
      LOGICAL LTHREE,LFOUR
      DIMENSION IFAC(19)
*
      IORD=0
* L must be larger than 1!
      IF (ABS(L).LE.1) THEN
         IFAC(1)=-99
         RETURN
      ENDIF
* Negative L means reordering of the factors (makes only sense if more
* than one factor is contained in L, single factor is trivial) ... :
      IF (L.LT.-5) IORD=1
      NN=ABS(L)
      K=1
* No ordering, factors 4 always allowed ('original mode')
      IF (IORD.EQ.0) THEN
* Test for factors of 4:
   10    IF (MOD(NN,4).NE.0) GOTO 20
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=4
         NN=NN/4
         IF (NN.EQ.1) GOTO 90
         GOTO 10
* Test for extra factor of 2
   20    IF (MOD(NN,2).NE.0) GOTO 30
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=2
         NN=NN/2
         IF (NN.EQ.1) GOTO 90
* Test for factors of 3
   30    IF (MOD(NN,3).NE.0) GOTO 40
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=3
         NN=NN/3
         IF (NN.EQ.1) GOTO 90
         GOTO 30
* Test for factors of 7
   40    IF (MOD(NN,7).NE.0) GOTO 45
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=7
         NN=NN/7
         IF (NN.EQ.1) GOTO 90
         GOTO 40
* Test for factors of 5
   45    IF (MOD(NN,5).NE.0) GOTO 100
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=5
         NN=NN/5
         IF (NN.EQ.1) GOTO 90
         GOTO 45
      ELSE
         RN=NN
         LTHREE=MOD(NN,3).EQ.0
         LFOUR =ABS((LOG(RN)/LOG(4.))-NINT(LOG(RN)/LOG(4.))).LT.1E-6
* In some few cases it is helpful to split one factor 4 in 2 factors 2
* (placing one factor 2 at the beginning and one at the end ...)
         IF ((NN.GT.16).AND.LFOUR) THEN
            K=K+1
            IFAC(K)=2
            NN=NN/2
         ENDIF
* Some factor 4 at the beginning seems generally to be helpful??
         IF ((K.EQ.1).AND.(MOD(NN,4).EQ.0).AND.(NN.GE.8)
     C                                    .AND.(.NOT.LTHREE)) THEN
            K=K+1
            IFAC(K)=4
            NN=NN/4
         ENDIF
* Search factors of 3
   50    IF (MOD(NN,3).NE.0) GOTO 60
         K=K+1
         IFAC(K)=3
         NN=NN/3
         IF (NN.EQ.1) GOTO 90
         GOTO 50
* Search factors of 4
   60    IF (MOD(NN,4).NE.0) GOTO 70
         K=K+1
         IFAC(K)=4
         NN=NN/4
         IF (NN.EQ.1) GOTO 90
         GOTO 60
* Search factors of 7
   70    IF (MOD(NN,7).NE.0) GOTO 75
         K=K+1
         IF (K.GT.19) GOTO 100
         IFAC(K)=7
         NN=NN/7
         IF (NN.EQ.1) GOTO 90
         GOTO 70
* Search factors of 5
   75    IF (MOD(NN,5).NE.0) GOTO 80
         K=K+1
         IFAC(K)=5
         NN=NN/5
         IF (NN.EQ.1) GOTO 90
         GOTO 75
* Search any remaining factor 2
   80    IF (MOD(NN,2).NE.0) GOTO 100
         K=K+1
         IFAC(K)=2
         NN=NN/2
* In few cases it is better to place it a the beginning?
         IF ((IFAC(2).EQ.3).AND.(K.GT.4)) THEN
            DO 85 J=K,3,-1
   85       IFAC(J)=IFAC(J-1)
            IFAC(2)=2
         ENDIF
         IF (NN.EQ.1) GOTO 90
         GOTO 80
      END IF
* IFAC(1) now contains number of factors
   90 IFAC(1)=K-1
      RETURN
* Error: other factors than 2,3,4 or 5 or more than 18 factors!
  100 IFAC(1)=-99
      RETURN
      END


      SUBROUTINE FTRIGC(L,TRIGS)

************************************************************************
*                                                                      *
* Sets up trigonometric tables for Fourier Transform of length L. The  *
* complex phase factors are stored in TRIGS, with alternating storage  *
* for real (cosine) parts and imaginary (sine) parts as in a complex   *
* FORTRAN array. TRIGS contains the factors phase_k=exp(+i*PI*(k/L)).  *
*                                                                      *
************************************************************************

      DIMENSION TRIGS(*)
      DOUBLE PRECISION PI,DELTA,ANGLE
      PARAMETER(PI=3.1415926535897932384626433832795D0)
*
      DELTA=PI/DBLE(ABS(L))
      DO 10 I=1,2*ABS(L)-1,2
         ANGLE=DBLE(I-1)*DELTA
         TRIGS(I)=COS(ANGLE)
         TRIGS(I+1)=SIN(ANGLE)
   10 CONTINUE
      RETURN
      END


      SUBROUTINE FTRIGR(L,TRIGS)

************************************************************************
*                                                                      *
* Sets up trigonometric tables for Fourier Transform of length L. The  *
* complex phase factors are stored in TRIGS, with alternating storage  *
* for real (cosine) parts and imaginary (sine) parts as in a complex   *
* FORTRAN array. TRIGS contains the factors phase_k=exp(+i*PI*(k/L)).  *
*                                                                      *
************************************************************************

      DIMENSION TRIGS(*)
      DOUBLE PRECISION PI,DELTA,ANGLE
      PARAMETER(PI=3.1415926535897932384626433832795D0)
*
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
   30 TRIGS(I)=0.
      RETURN
      END


      LOGICAL FUNCTION FACERR(L)

************************************************************************
*                                                                      *
* Serves as a possible factorization pre-checker for routine CFTTAB.   *
* Checks the factorization of L. Factors 2,3,4,5 and 7 are allowed.    *
* Warning! You may not have more than 18 factors in routine CFTTAB!    *
*                                                                      *
************************************************************************

      PARAMETER(NFAC=5)
* List of allowed factors:
      INTEGER FACTOR(NFAC)
      SAVE FACTOR
      DATA FACTOR /4,2,3,5,7/
*
      FACERR=.FALSE.
* L must be larger than 1!
      IF (L.LT.2) GOTO 150
      N=L
      ICOUNT=0
* Check all factors ... :
      DO 100 I=1,NFAC
   10    IF (MOD(N,FACTOR(I)).EQ.0) THEN
            ICOUNT=ICOUNT+1
            N=N/FACTOR(I)
            GOTO 10
         END IF
* At the end we should arrive at factor '1' ...
         IF (N.EQ.1) GOTO 200
  100 CONTINUE
* ... otherwise there was some factor that is not allowed!
  150 FACERR=.TRUE.
      RETURN
  200 CONTINUE
* Too much factors!
      IF (ICOUNT.GT.18) GOTO 150
      RETURN
      END


      LOGICAL FUNCTION FACODD(L)

************************************************************************
*                                                                      *
* Checks whether L is odd (result=TRUE) of even (result=FALSE)         *
*                                                                      *
************************************************************************

      FACODD=(MOD(L,2).EQ.1)
      RETURN
      END


      SUBROUTINE FTFLOP(L,IFLOP)

************************************************************************
*                                                                      *
* Estimates the number of floating point operations (IFLOP) needed for *
* a FFT of length L according to the factorization from routine FTFACT *
*                                                                      *
************************************************************************

      DIMENSION IFAC(19)

* Single factor transforms according to structure of SPASSM:
      IF (L.LE.5) THEN
         IF (L.EQ.2) IFLOP=4
         IF (L.EQ.3) IFLOP=16
         IF (L.EQ.4) IFLOP=16
         IF (L.EQ.5) IFLOP=36
         RETURN
      END IF
* Get the factorization:
      CALL FTFACT(L,IFAC)
      IFLOP=0
* Count operations according to the structure of FPASSM/IPASSM:
      LA=1
      DO 30 I=2,IFAC(1)+1
         IFACT=IFAC(I)
         M=L/IFACT
         DO 10 J=1,LA
            IF (IFACT.EQ.2) IFLOP=IFLOP+4
            IF (IFACT.EQ.3) IFLOP=IFLOP+16
            IF (IFACT.EQ.4) IFLOP=IFLOP+16
            IF (IFACT.EQ.5) IFLOP=IFLOP+36
   10    CONTINUE
         IF (LA.NE.M) THEN
            DO 20 J=LA+1,M,LA
             DO 20 JJ=1,LA
              IF (IFACT.EQ.2) IFLOP=IFLOP+10
              IF (IFACT.EQ.3) IFLOP=IFLOP+28
              IF (IFACT.EQ.4) IFLOP=IFLOP+34
              IF (IFACT.EQ.5) IFLOP=IFLOP+52
   20       CONTINUE
         ENDIF
         LA=LA*IFACT
   30 CONTINUE
* That is all ...
      RETURN
      END


      SUBROUTINE FPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)

************************************************************************
*                                                                      *
* FPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
* performs one pass through data as part of a multiple complex Fast    *
* Fourier Transformation in one dimension on multiple 1-D data sets.   *
* Variables:                                                           *
*   A is the array containing the real parts of the input vector       *
*   B is the array containing the imaginary parts of the input vector  *
*   C is the array containing the real parts of the output vector      *
*   D is the array containing the imaginary parts of the output vector *
*   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
*   INC1 is the increment between elements of one data set in A and B  *
*   INC2 is the increment between elements of one data set in C and D  *
*   INC3 is the addressing increment between data sets in A and B      *
*   INC4 is the addressing increment between data sets in C and D      *
*   LOT is the number of data sets                                     *
*   N is the length of one data set (length of the FFT)                *
*   IFAC is the current factor of N for which the pass is performed    *
*   LA is the product of all previous factors of all previous passes   *
*                                                                      *
************************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907)
      PARAMETER(COS36=0.80901699437494742410229341718282)
      PARAMETER(SIN60=0.86602540378443864676372317075294)
      PARAMETER(SIN72=0.95105651629515357211643933337938)
      PARAMETER(COS72=0.30901699437494742410229341718282)
      PARAMETER(SQR2H=0.70710678118654752440084436210485)
      PARAMETER(CPIS1=0.90096886790241912623610231950745)
      PARAMETER(CPIS2=0.62348980185873353052500488400424)
      PARAMETER(CPIS3=0.22252093395631440428890256449679)
      PARAMETER(SPIS1=0.43388373911755812047576833284836)
      PARAMETER(SPIS2=0.78183148246802980870844452667406)
      PARAMETER(SPIS3=0.97492791218182360701813168299393)
*
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO.GT.7) RETURN
      GOTO (10,50,90,130,170,210,250),IGO
*
* Coding for factor 2:
*
   10 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      DO 20 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
!OCL VECTOR,NOVREC
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
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 40 K=LA1,M,LA
      KB=K+K-2
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      DO 30 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 3:
*
   50 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      DO 60 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=(A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC3
      J=J+INC4
   55 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   60 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 65 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=
     C    C1*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I))))
     C   -S1*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      D(JB+J)=
     C    S1*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I))))
     C   +C1*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      C(JC+J)=
     C    C2*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I))))
     C   -S2*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      D(JC+J)=
     C    S2*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I))))
     C   +C2*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      I=I+INC3
      J=J+INC4
   65 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   70 CONTINUE
      JBASE=JBASE+JUMP
   80 CONTINUE
      RETURN
*
* Coding for factor 4:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 105 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      C(JC+J)=
     C    C2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I)))
     C   -S2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      D(JC+J)=
     C    S2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I)))
     C   +C2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      C(JB+J)=
     C    C1*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I)))
     C   -S1*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      D(JB+J)=
     C    S1*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I)))
     C   +C1*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      C(JD+J)=
     C    C3*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I)))
     C   -S3*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      D(JD+J)=
     C    S3*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I)))
     C   +C3*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  105 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  110 CONTINUE
      JBASE=JBASE+JUMP
  120 CONTINUE
      RETURN
*
* Coding for factor 5:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  135 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  140 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 145 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=
     C    C1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   -S1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JB+J)=
     C    S1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +C1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JE+J)=
     C    C4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   -S4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JE+J)=
     C    S4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +C4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JC+J)=
     C    C2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   -S2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JC+J)=
     C    S2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +C2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      C(JD+J)=
     C    C3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   -S3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JD+J)=
     C    S3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +C3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      I=I+INC3
      J=J+INC4
  145 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  150 CONTINUE
      JBASE=JBASE+JUMP
  160 CONTINUE
      RETURN
*
* Coding for factor 6:
*
  170 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from FPASSM: factor 6 not yet coded ...'
      STOP

      RETURN
*
* Coding for factor 7:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I))
     C               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I))
     C               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C               -(SPIS2*(B(IB+I)-B(IG+I))
     C                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C               +(SPIS2*(B(IB+I)-B(IG+I))
     C                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C               +(SPIS2*(A(IB+I)-A(IG+I))
     C                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C               -(SPIS2*(A(IB+I)-A(IG+I))
     C                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C               -(SPIS3*(B(IB+I)-B(IG+I))
     C                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C               +(SPIS3*(B(IB+I)-B(IG+I))
     C                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C               +(SPIS3*(A(IB+I)-A(IG+I))
     C                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C               -(SPIS3*(A(IB+I)-A(IG+I))
     C                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C               -(SPIS1*(B(IB+I)-B(IG+I))
     C                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C               +(SPIS1*(B(IB+I)-B(IG+I))
     C                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C               +(SPIS1*(A(IB+I)-A(IG+I))
     C                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C               -(SPIS1*(A(IB+I)-A(IG+I))
     C                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  215 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  220 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 225 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I))
     C               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I))
     C               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=
     C       C1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            -(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       -S1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            +(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JB+J)=
     C        S1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            -(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       +C1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            +(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JG+J)=
     C        C6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            +(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       -S6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            -(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JG+J)=
     C        S6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            +(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       +C6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            -(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JC+J)=
     C        C2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            -(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       -S2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            +(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JC+J)=
     C        S2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            -(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +C2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            +(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JF+J)=
     C        C5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            +(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       -S5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            -(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JF+J)=
     C        S5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            +(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +C5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            -(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JD+J)=
     C        C3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            -(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       -S3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            +(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JD+J)=
     C        S3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            -(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +C3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            +(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      C(JE+J)=
     C        C4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            +(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       -S4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            -(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JE+J)=
     C        S4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            +(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +C4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            -(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  225 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  230 CONTINUE
      JBASE=JBASE+JUMP
  240 CONTINUE
      RETURN
*
* Coding for factor 8:
*
  250 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from FPASSM: factor 8 not yet coded ...'
      STOP

      RETURN
      END


      SUBROUTINE IPASSM(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,IFAC,LA)

************************************************************************
*                                                                      *
* IPASSM is essentially the TEMPERTON/FULKER routine 'VPASSM' which    *
* performs one pass through data as part of a multiple complex Fast    *
* Fourier Transformation in one dimension on multiple 1-D data sets.   *
* Originally VPASSM did not allow to perform the inverse FFT and so    *
* one had to rearrange the data externally in some way that the FFT    *
* could be done with the original phase factors. This required some    *
* additional calculation time and was hence not optimal. IPASSM is     *
* now the 'VPASSM'-version with conjugate phase factors needed for     *
* the inverse transform which can now be performed at same perfomance  *
* rates as the forward transform!                                      *
* Variables:                                                           *
*   A is the array containing the real parts of the input vector       *
*   B is the array containing the imaginary parts of the input vector  *
*   C is the array containing the real parts of the output vector      *
*   D is the array containing the imaginary parts of the output vector *
*   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
*   INC1 is the increment between elements of one data set in A and B  *
*   INC2 is the increment between elements of one data set in C and D  *
*   INC3 is the addressing increment between data sets in A and B      *
*   INC4 is the addressing increment between data sets in C and D      *
*   LOT is the number of data sets                                     *
*   N is the length of one data set (length of the FFT)                *
*   IFAC is the current factor of N for which the pass is performed    *
*   LA is the product of all previous factors of all previous passes   *
*                                                                      *
************************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907)
      PARAMETER(COS36=0.80901699437494742410229341718282)
      PARAMETER(SIN60=0.86602540378443864676372317075294)
      PARAMETER(SIN72=0.95105651629515357211643933337938)
      PARAMETER(COS72=0.30901699437494742410229341718282)
      PARAMETER(SQR2H=0.70710678118654752440084436210485)
      PARAMETER(CPIS1=0.90096886790241912623610231950745)
      PARAMETER(CPIS2=0.62348980185873353052500488400424)
      PARAMETER(CPIS3=0.22252093395631440428890256449679)
      PARAMETER(SPIS1=0.43388373911755812047576833284836)
      PARAMETER(SPIS2=0.78183148246802980870844452667406)
      PARAMETER(SPIS3=0.97492791218182360701813168299393)
*
      M=N/IFAC
      IINK=M*INC1
      JINK=LA*INC2
      JUMP=(IFAC-1)*JINK
      IBASE=0
      JBASE=0
      IGO=IFAC-1
      IF (IGO.GT.7) RETURN
      GOTO (10,50,90,130,170,210,250),IGO
*
* Coding for factor 2:
*
   10 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      DO 20 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
      IF (LA.EQ.M) RETURN
      LA1=LA+1
      JBASE=JBASE+JUMP
      DO 40 K=LA1,M,LA
      KB=K+K-2
      C1=TRIGS(KB+1)
      S1=TRIGS(KB+2)
      DO 30 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 3:
*
   50 IA=1
      JA=1
      IB=IA+IINK
      JB=JA+JINK
      IC=IB+IINK
      JC=JB+JINK
      DO 60 L=1,LA
      I=IBASE
      J=JBASE
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=(A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC3
      J=J+INC4
   55 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   60 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 65 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IC+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IC+I))
      C(JB+J)=
     C    C1*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I))))
     C   +S1*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
      D(JB+J)=
     C    C1*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I))))
     C   -S1*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I))))
      C(JC+J)=
     C    C2*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I))))
     C   +S2*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
      D(JC+J)=
     C    C2*((B(IA+I)-0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I))))
     C   -S2*((A(IA+I)-0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I))))
      I=I+INC3
      J=J+INC4
   65 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
   70 CONTINUE
      JBASE=JBASE+JUMP
   80 CONTINUE
      RETURN
*
* Coding for factor 4:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 105 IJK=1,LOT
      C(JA+J)=(A(IA+I)+A(IC+I))+(A(IB+I)+A(ID+I))
      D(JA+J)=(B(IA+I)+B(IC+I))+(B(IB+I)+B(ID+I))
      C(JC+J)=
     C    C2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I)))
     C   +S2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
      D(JC+J)=
     C    C2*((B(IA+I)+B(IC+I))-(B(IB+I)+B(ID+I)))
     C   -S2*((A(IA+I)+A(IC+I))-(A(IB+I)+A(ID+I)))
      C(JB+J)=
     C    C1*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I)))
     C   +S1*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
      D(JB+J)=
     C    C1*((B(IA+I)-B(IC+I))-(A(IB+I)-A(ID+I)))
     C   -S1*((A(IA+I)-A(IC+I))+(B(IB+I)-B(ID+I)))
      C(JD+J)=
     C    C3*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I)))
     C   +S3*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
      D(JD+J)=
     C    C3*((B(IA+I)-B(IC+I))+(A(IB+I)-A(ID+I)))
     C   -S3*((A(IA+I)-A(IC+I))-(B(IB+I)-B(ID+I)))
      I=I+INC3
      J=J+INC4
  105 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  110 CONTINUE
      JBASE=JBASE+JUMP
  120 CONTINUE
      RETURN
*
* Coding for factor 5:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC3
      J=J+INC4
  135 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  140 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 145 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=
     C    C1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +S1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JB+J)=
     C   -S1*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +C1*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JE+J)=
     C    C4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +S4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      D(JE+J)=
     C   -S4*((A(IA+I)+COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C      -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I))))
     C   +C4*((B(IA+I)+COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C      +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I))))
      C(JC+J)=
     C    C2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +S2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JC+J)=
     C   -S2*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +C2*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      C(JD+J)=
     C    C3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +S3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      D(JD+J)=
     C   -S3*((A(IA+I)-COS36*(A(IB+I)+A(IE+I))+COS72*(A(IC+I)+A(ID+I)))
     C      -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I))))
     C   +C3*((B(IA+I)-COS36*(B(IB+I)+B(IE+I))+COS72*(B(IC+I)+B(ID+I)))
     C      +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I))))
      I=I+INC3
      J=J+INC4
  145 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  150 CONTINUE
      JBASE=JBASE+JUMP
  160 CONTINUE
      RETURN
*
* Coding for factor 6:
*
  170 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from IPASSM: factor 6 not yet coded ...'
      STOP

      RETURN
*
* Coding for factor 7:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I))
     C               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I))
     C               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C               +(SPIS2*(B(IB+I)-B(IG+I))
     C                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C                -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C               -(SPIS2*(B(IB+I)-B(IG+I))
     C                +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C               -(SPIS2*(A(IB+I)-A(IG+I))
     C                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C                -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C               +(SPIS2*(A(IB+I)-A(IG+I))
     C                +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C               +(SPIS3*(B(IB+I)-B(IG+I))
     C                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C                -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C               -(SPIS3*(B(IB+I)-B(IG+I))
     C                -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C               -(SPIS3*(A(IB+I)-A(IG+I))
     C                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C                -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C               +(SPIS3*(A(IB+I)-A(IG+I))
     C                -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C               +(SPIS1*(B(IB+I)-B(IG+I))
     C                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C                +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C               -(SPIS1*(B(IB+I)-B(IG+I))
     C                -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C               -(SPIS1*(A(IB+I)-A(IG+I))
     C                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C                +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C               +(SPIS1*(A(IB+I)-A(IG+I))
     C                -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC3
      J=J+INC4
  215 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  220 CONTINUE
      IF (LA.EQ.M) RETURN
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 225 IJK=1,LOT
      C(JA+J)=A(IA+I)+(A(IB+I)+A(IG+I))
     C               +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=B(IA+I)+(B(IB+I)+B(IG+I))
     C               +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=
     C       C1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            +(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C      +S1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            -(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JB+J)=
     C       -S1*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            +(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       +C1*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            -(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JG+J)=
     C        C6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            -(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       +S6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            +(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      D(JG+J)=
     C       -S6*((A(IA+I)+CPIS2*(A(IB+I)+A(IG+I))
     C             -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C            -(SPIS2*(B(IB+I)-B(IG+I))
     C             +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I))))
     C       +C6*((B(IA+I)+CPIS2*(B(IB+I)+B(IG+I))
     C             -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C            +(SPIS2*(A(IB+I)-A(IG+I))
     C             +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I))))
      C(JC+J)=
     C        C2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            +(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +S2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            -(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JC+J)=
     C       -S2*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            +(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +C2*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            -(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JF+J)=
     C        C5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            -(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +S5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            +(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      D(JF+J)=
     C       -S5*((A(IA+I)-CPIS3*(A(IB+I)+A(IG+I))
     C             -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C            -(SPIS3*(B(IB+I)-B(IG+I))
     C             -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I))))
     C       +C5*((B(IA+I)-CPIS3*(B(IB+I)+B(IG+I))
     C             -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C            +(SPIS3*(A(IB+I)-A(IG+I))
     C             -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I))))
      C(JD+J)=
     C        C3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            +(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +S3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            -(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JD+J)=
     C       -S3*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            +(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +C3*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            -(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      C(JE+J)=
     C        C4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            -(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +S4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            +(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      D(JE+J)=
     C       -S4*((A(IA+I)-CPIS1*(A(IB+I)+A(IG+I))
     C             +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C            -(SPIS1*(B(IB+I)-B(IG+I))
     C             -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I))))
     C       +C4*((B(IA+I)-CPIS1*(B(IB+I)+B(IG+I))
     C             +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C            +(SPIS1*(A(IB+I)-A(IG+I))
     C             -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I))))
      I=I+INC3
      J=J+INC4
  225 CONTINUE
      IBASE=IBASE+INC1
      JBASE=JBASE+INC2
  230 CONTINUE
      JBASE=JBASE+JUMP
  240 CONTINUE
      RETURN
*
* Coding for factor 8:
*
  250 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from IPASSM: factor 8 not yet coded ...'
      STOP

      RETURN
      END


      SUBROUTINE SPASSM(A,B,C,D,INC1,INC2,LOT,N,ISIGN)

************************************************************************
*                                                                      *
* SPASSM is derived from the TEMPERTON/FULKER routine 'VPASSM' which   *
* performs one pass through data as part of a multiple complex Fast    *
* Fourier Transformation in one dimension on multiple 1-D data sets.   *
* SPASSM is the special version for 'single factor transformations'.   *
* Variables:                                                           *
*   A is the array containing the real parts of the data vector        *
*   B is the array containing the imaginary parts of the data vector   *
*   C is a work array                                                  *
*   D is a work array                                                  *
*   INC1 is the increment between elements of one data set in A and B  *
*   INC2 is the addressing increment between data sets in A and B      *
*   LOT is the number of data sets                                     *
*   N is the length of one data set (length of the FFT)                *
*   ISIGN gives the type of transformation (>=0: forward, <0: inverse) *
*                                                                      *
************************************************************************

      DIMENSION A(*),B(*),C(*),D(*)
      PARAMETER(SIN36=0.58778525229247312916870595463907)
      PARAMETER(COS36=0.80901699437494742410229341718282)
      PARAMETER(SIN60=0.86602540378443864676372317075294)
      PARAMETER(SIN72=0.95105651629515357211643933337938)
      PARAMETER(COS72=0.30901699437494742410229341718282)
      PARAMETER(SQR2H=0.70710678118654752440084436210485)
      PARAMETER(CPIS1=0.90096886790241912623610231950745)
      PARAMETER(CPIS2=0.62348980185873353052500488400424)
      PARAMETER(CPIS3=0.22252093395631440428890256449679)
      PARAMETER(SPIS1=0.43388373911755812047576833284836)
      PARAMETER(SPIS2=0.78183148246802980870844452667406)
      PARAMETER(SPIS3=0.97492791218182360701813168299393)
*
      IGO=N-1
      IF (IGO.GT.7) RETURN
      IF (ISIGN.LT.0) GOTO 1000
      GOTO (10,50,90,130,170,210,250),IGO
*
* Coding for factor 2:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 3:
*
   50 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 55 IJK=1,LOT
      C(JA+J)=A(IB+I)+A(IC+I)
      D(JA+J)=B(IB+I)+B(IC+I)
      C(JB+J)=(0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC2
      J=J+1
   55 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .

CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 4:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 5:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 135 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I)))
     C  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I)))
     C  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I)))
     C  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I)))
     C  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC2
      J=J+1
  135 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 6:
*
  170 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from SPASSM: factor 6 not yet coded ...'
      STOP

      RETURN
*
* Coding for factor 7:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 215 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IG+I))
     C       +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=(B(IB+I)+B(IG+I))
     C       +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(CPIS2*(A(IB+I)+A(IG+I))
     C        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C       -(SPIS2*(B(IB+I)-B(IG+I))
     C        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(CPIS2*(A(IB+I)+A(IG+I))
     C        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C       +(SPIS2*(B(IB+I)-B(IG+I))
     C        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(CPIS2*(B(IB+I)+B(IG+I))
     C        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C       +(SPIS2*(A(IB+I)-A(IG+I))
     C        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(CPIS2*(B(IB+I)+B(IG+I))
     C        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C       -(SPIS2*(A(IB+I)-A(IG+I))
     C        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(-CPIS3*(A(IB+I)+A(IG+I))
     C         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C        -(SPIS3*(B(IB+I)-B(IG+I))
     C         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(-CPIS3*(A(IB+I)+A(IG+I))
     C         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C        +(SPIS3*(B(IB+I)-B(IG+I))
     C         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(-CPIS3*(B(IB+I)+B(IG+I))
     C         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C        +(SPIS3*(A(IB+I)-A(IG+I))
     C         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(-CPIS3*(B(IB+I)+B(IG+I))
     C         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C        -(SPIS3*(A(IB+I)-A(IG+I))
     C         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(-CPIS1*(A(IB+I)+A(IG+I))
     C         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C        -(SPIS1*(B(IB+I)-B(IG+I))
     C         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(-CPIS1*(A(IB+I)+A(IG+I))
     C         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C        +(SPIS1*(B(IB+I)-B(IG+I))
     C         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(-CPIS1*(B(IB+I)+B(IG+I))
     C         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C        +(SPIS1*(A(IB+I)-A(IG+I))
     C         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(-CPIS1*(B(IB+I)+B(IG+I))
     C         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C        -(SPIS1*(A(IB+I)-A(IG+I))
     C         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC2
      J=J+1
  215 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 8:
*
  250 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from SPASSM: factor 8 not yet coded ...'
      STOP

      RETURN
 1000 CONTINUE
      GOTO (1010,1050,1090,1130,1170,1210,1250),IGO
*
* Coding for factor 2:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 3:
*
 1050 IA=1
      JA=1
      IB=IA+INC1
      JB=JA+LOT
      IC=IB+INC1
      JC=JB+LOT
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 1055 IJK=1,LOT
      C(JA+J)=A(IB+I)+A(IC+I)
      D(JA+J)=B(IB+I)+B(IC+I)
      C(JB+J)=(0.5*(A(IB+I)+A(IC+I)))-(SIN60*(B(IB+I)-B(IC+I)))
      C(JC+J)=(0.5*(A(IB+I)+A(IC+I)))+(SIN60*(B(IB+I)-B(IC+I)))
      D(JB+J)=(0.5*(B(IB+I)+B(IC+I)))+(SIN60*(A(IB+I)-A(IC+I)))
      D(JC+J)=(0.5*(B(IB+I)+B(IC+I)))-(SIN60*(A(IB+I)-A(IC+I)))
      I=I+INC2
      J=J+1
 1055 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 4:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 5:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 1135 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IE+I))+(A(IC+I)+A(ID+I))
      D(JA+J)=(B(IB+I)+B(IE+I))+(B(IC+I)+B(ID+I))
      C(JB+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  +(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      C(JE+J)=(COS72*(A(IB+I)+A(IE+I))-COS36*(A(IC+I)+A(ID+I)))
     C  -(SIN72*(B(IB+I)-B(IE+I))+SIN36*(B(IC+I)-B(ID+I)))
      D(JB+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  -(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      D(JE+J)=(COS72*(B(IB+I)+B(IE+I))-COS36*(B(IC+I)+B(ID+I)))
     C  +(SIN72*(A(IB+I)-A(IE+I))+SIN36*(A(IC+I)-A(ID+I)))
      C(JC+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I)))
     C  +(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      C(JD+J)=(COS72*(A(IC+I)+A(ID+I))-COS36*(A(IB+I)+A(IE+I)))
     C  -(SIN36*(B(IB+I)-B(IE+I))-SIN72*(B(IC+I)-B(ID+I)))
      D(JC+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I)))
     C  -(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      D(JD+J)=(COS72*(B(IC+I)+B(ID+I))-COS36*(B(IB+I)+B(IE+I)))
     C  +(SIN36*(A(IB+I)-A(IE+I))-SIN72*(A(IC+I)-A(ID+I)))
      I=I+INC2
      J=J+1
 1135 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 6:
*
 1170 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from SPASSM: factor 6 not yet coded ...'
      STOP

      RETURN
*
* Coding for factor 7:
*
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 1215 IJK=1,LOT
      C(JA+J)=(A(IB+I)+A(IG+I))
     C       +(A(IC+I)+A(IF+I))+(A(ID+I)+A(IE+I))
      D(JA+J)=(B(IB+I)+B(IG+I))
     C       +(B(IC+I)+B(IF+I))+(B(ID+I)+B(IE+I))
      C(JB+J)=(CPIS2*(A(IB+I)+A(IG+I))
     C        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C       +(SPIS2*(B(IB+I)-B(IG+I))
     C        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      C(JG+J)=(CPIS2*(A(IB+I)+A(IG+I))
     C        -CPIS3*(A(IC+I)+A(IF+I))-CPIS1*(A(ID+I)+A(IE+I)))
     C       -(SPIS2*(B(IB+I)-B(IG+I))
     C        +SPIS3*(B(IC+I)-B(IF+I))+SPIS1*(B(ID+I)-B(IE+I)))
      D(JB+J)=(CPIS2*(B(IB+I)+B(IG+I))
     C        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C       -(SPIS2*(A(IB+I)-A(IG+I))
     C        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      D(JG+J)=(CPIS2*(B(IB+I)+B(IG+I))
     C        -CPIS3*(B(IC+I)+B(IF+I))-CPIS1*(B(ID+I)+B(IE+I)))
     C       +(SPIS2*(A(IB+I)-A(IG+I))
     C        +SPIS3*(A(IC+I)-A(IF+I))+SPIS1*(A(ID+I)-A(IE+I)))
      C(JC+J)=(-CPIS3*(A(IB+I)+A(IG+I))
     C         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C        +(SPIS3*(B(IB+I)-B(IG+I))
     C         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      C(JF+J)=(-CPIS3*(A(IB+I)+A(IG+I))
     C         -CPIS1*(A(IC+I)+A(IF+I))+CPIS2*(A(ID+I)+A(IE+I)))
     C        -(SPIS3*(B(IB+I)-B(IG+I))
     C         -SPIS1*(B(IC+I)-B(IF+I))-SPIS2*(B(ID+I)-B(IE+I)))
      D(JC+J)=(-CPIS3*(B(IB+I)+B(IG+I))
     C         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C        -(SPIS3*(A(IB+I)-A(IG+I))
     C         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      D(JF+J)=(-CPIS3*(B(IB+I)+B(IG+I))
     C         -CPIS1*(B(IC+I)+B(IF+I))+CPIS2*(B(ID+I)+B(IE+I)))
     C        +(SPIS3*(A(IB+I)-A(IG+I))
     C         -SPIS1*(A(IC+I)-A(IF+I))-SPIS2*(A(ID+I)-A(IE+I)))
      C(JD+J)=(-CPIS1*(A(IB+I)+A(IG+I))
     C         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C        +(SPIS1*(B(IB+I)-B(IG+I))
     C         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      C(JE+J)=(-CPIS1*(A(IB+I)+A(IG+I))
     C         +CPIS2*(A(IC+I)+A(IF+I))-CPIS3*(A(ID+I)+A(IE+I)))
     C        -(SPIS1*(B(IB+I)-B(IG+I))
     C         -SPIS2*(B(IC+I)-B(IF+I))+SPIS3*(B(ID+I)-B(IE+I)))
      D(JD+J)=(-CPIS1*(B(IB+I)+B(IG+I))
     C         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C        -(SPIS1*(A(IB+I)-A(IG+I))
     C         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      D(JE+J)=(-CPIS1*(B(IB+I)+B(IG+I))
     C         +CPIS2*(B(IC+I)+B(IF+I))-CPIS3*(B(ID+I)+B(IE+I)))
     C        +(SPIS1*(A(IB+I)-A(IG+I))
     C         -SPIS2*(A(IC+I)-A(IF+I))+SPIS3*(A(ID+I)-A(IE+I)))
      I=I+INC2
      J=J+1
 1215 CONTINUE
      I=0
      J=0
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
*
* Coding for factor 8:
*
 1250 IA=1
      JA=1

      WRITE(*,*) 'Big sorry from SPASSM: factor 8 not yet coded ...'
      STOP

      RETURN
      END


      SUBROUTINE HCOMB(A,B,C,D,TRIGS,INC1,INC2,INC3,INC4,LOT,N,RSIGN)

************************************************************************
*                                                                      *
* HCOMB is the post-processing routine performing the final data       *
* arrangement  for a real to hermitian conjugate transform.            *
* Variables:                                                           *
*   A is the array containing the real parts of the input vector       *
*   B is the array containing the imaginary parts of the input vector  *
*   C is the array containing the real parts of the output vector      *
*   D is the array containing the imaginary parts of the output vector *
*   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
*   INC1 is the increment between elements of one data set in A and B  *
*   INC2 is the increment between elements of one data set in C and D  *
*   INC3 is the addressing increment between data sets in A and B      *
*   INC4 is the addressing increment between data sets in C and D      *
*   LOT is the number of data sets                                     *
*   N is the length of one data set (length of the FFT)                *
*   RSIGN is the type of transformation (1: forward, -1: inverse)      *
*                                                                      *
************************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
*
      IBASEL=1
      JBASEL=1
      IBASEH=1+(N-1)*INC1
      JBASEH=1+N*INC2
* First (0'th) and last (N'th) data point treated seperately here:
      IL=1
      JL=1
      JH=JBASEH
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
      DO 10 IJK=1,LOT
         C(JL)=A(IL)+B(IL)
         D(JL)=0.
         C(JH)=A(IL)-B(IL)
         D(JH)=0.
         IL=IL+INC3
         JL=JL+INC4
         JH=JH+INC4
   10 CONTINUE
* Now we treat the n'th and the N-n'th data point simultaneously:
      IBASEL=1+INC1
      JBASEL=1+INC2
      JBASEH=JBASEH-INC2
      DO 100 K=2,N-1,2
         CO=0.5*TRIGS(2*N+K+1)
         SI=0.5*TRIGS(2*N+K+2)*RSIGN
         IL=IBASEL
         JL=JBASEL
         IH=IBASEH
         JH=JBASEH
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
         DO 50 IJK=1,LOT
            C(JL)=0.5*(A(IL)+A(IH))+CO*(B(IL)+B(IH))+SI*(A(IL)-A(IH))
            D(JL)=0.5*(B(IL)-B(IH))+SI*(B(IL)+B(IH))-CO*(A(IL)-A(IH))
            C(JH)=0.5*(A(IL)+A(IH))-CO*(B(IL)+B(IH))-SI*(A(IL)-A(IH))
            D(JH)=0.5*(B(IH)-B(IL))+SI*(B(IL)+B(IH))-CO*(A(IL)-A(IH))
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
* Finally if N was even there remains the 'central' data point! We must
* Treat this point separately because otherwise vectorisation of the
* previous loop (DO 50 IJK=1,LOT) would not be allowed!
      IF (MOD(N,2).EQ.0) THEN
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

************************************************************************
*                                                                      *
* RCOMB is the preprocessing routine preparing the data arrangement    *
* for a hermitian conjugate to real transform.                         *
* Variables:                                                           *
*   A is the array containing the real parts of the input vector       *
*   B is the array containing the imaginary parts of the input vector  *
*   C is the array containing the real parts of the output vector      *
*   D is the array containing the imaginary parts of the output vector *
*   TRIGS is a precalculated table of sines and cosines (from CFTTAB)  *
*   INC1 is the increment between elements of one data set in A and B  *
*   INC2 is the increment between elements of one data set in C and D  *
*   INC3 is the addressing increment between data sets in A and B      *
*   INC4 is the addressing increment between data sets in C and D      *
*   LOT is the number of data sets                                     *
*   N is the length of one data set (length of the FFT)                *
*   RSIGN is the type of transformation (1: forward, -1: inverse)      *
*                                                                      *
************************************************************************

      DIMENSION A(*),B(*),C(*),D(*),TRIGS(*)
*
      IBASEL=1
      JBASEL=1
      IBASEH=1+N*INC1
      JBASEH=1+(N-1)*INC2
* First (0'th) and last (N'th) data point treated seperately here:
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
* Now we treat the n'th and the N-n'th data point simultaneously:
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
* Enforce vectorization to get maximum performance (you may vectorize
* this loop but compiler says there   c o u l d   be some recurrence)!
* First directive is for Crays, second for CONVEXes, supply directive
* needed for your own machine if it is not a Cray or a CONVEX ... .
CDIR$ IVDEP
*vdir nodep
C$DIR FORCE_VECTOR
*VOCL LOOP,NOVREC
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
* Finally if N was even there remains the 'central' data point! We must
* Treat this point separately because otherwise vectorisation of the
* previous loop (DO 50 IJK=1,LOT) would not be allowed!
      IF (MOD(N,2).EQ.0) THEN
         IC=IBASEL
         JC=JBASEL
         DO 1000 IJK=1,LOT
            C(JC)=2.*A(IC)
            D(JC)=(-2.)*RSIGN*B(IC)
            IC=IC+INC3
            JC=JC+INC4
 1000    CONTINUE
      END IF
      RETURN
      END


      SUBROUTINE D1ZERO(N1,ID1,LOT,DATA)

************************************************************************
*                                                                      *
* Routine D1ZERO fills the part of an array with dimensions (ID1,LOT)  *
* with zeros which is given by addresses (I,J) with N1<I<=ID1 and with *
* 1<=J<=LOT. This is necessary for some routines (FFT3DC, FFTC3D, ...) *
* to avoid trouble with undefined contents of work arrays (which leads *
* to no errors[!!] - but probably to floating point errors ... [??]).  *
* Variables:                                                           *
* DATA is the data array                                               *
* N1 is the length of the transforms along the first direction         *
* ID1 is the leading dimension of array DATA                           *
* LOT is the number of data sets ("second dimension")                  *
*                                                                      *
************************************************************************

      DIMENSION DATA(ID1*LOT)
      IF (N1.GE.ID1) RETURN
      DO 100 I1=N1+1,ID1
         IA=I1
         DO 200 IJK=1,LOT
            DATA(IA)=0.
            IA=IA+ID1
  200    CONTINUE
  100 CONTINUE
      RETURN
      END


*=======================================================================
*
*   this routine returns true a NIN is a legal value for the FFT
*
*=======================================================================


      LOGICAL FUNCTION FFTCHK_FURTH(NIN)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (NFACT=4)
      DIMENSION IFACT(NFACT),NCOUNT(NFACT)
      DATA      IFACT /2,3,5,7/
      N=NIN
      DO 100 I=1,NFACT
        NCOUNT(I)=0
  120   NEXT=N/IFACT(I)
        IF (NEXT*IFACT(I).EQ.N) THEN
          N=NEXT
          NCOUNT(I)=NCOUNT(I)+1
          GOTO 120
        ENDIF
  100 CONTINUE
      IF (N.EQ.1 .AND. (NCOUNT(1).NE.0)) 
     &  THEN
        FFTCHK_FURTH=.TRUE.
      ELSE
        FFTCHK_FURTH=.FALSE.
      ENDIF
      RETURN
      END
