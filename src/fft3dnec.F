#include "symbol.inc"
!===============================================================================
! RCS:  $Id: fft3dnec.F,v 1.2 2001/04/12 17:33:45 kresse Exp $
!
! this modul implements the following FFT routines (which are called by VASP)
!   FFTWAV
!   FFTEXT
!   FFT3RC
!
! the basic fft routines
!   FFTBAS complex <-> complex
!   FFTBRC complex <-> real
! are missing
! usually this modul should be included in the main fft fortran file
! using the statement
!  #include "fft3dsimple.F"
! the main fft  file should contain the FFTBAS and (optionally FFTBRC) routine;
! but it is also possible to copy this file, and to implement FFTBAS
! (and FFTBRC) directly in the copy.
! On some machines it might be a good idea to optimize all routines
! in this modul, according to the functionallity of the basic FFT routine
! available on the specific machine
! (especially the gamma point only version which is fastest if an
!  real to complex FFT exists requires some optimization)
!
!===============================================================================
!===============================================================================
! template for  basic complex 3-d fast fourier transformation routine
! should give you some idea how the basic complex 3-d fft should be implemented
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!===============================================================================



    SUBROUTINE FFTBAS_(C,GRID,ISN)
      USE prec
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d) GRID
      COMPLEX(q) C(0:1)
      WRITE(*,*)"FFTBAS: is not implemented"
      STOP
      RETURN
    END SUBROUTINE FFTBAS_

!===============================================================================
! template for  basic real to complex 3-d fast fourier transformation routine
!   C is used as input and output array
!   in real space C is defined as
!    REAL(q)    C(1:N(1)+2  ,1:N(2),1:N(3)
!   in reciprocal space C is defined as
!    COMPLEX(q) C(1:N(1)/2+1,1:N(2),1:N(3)
!   this conforms to the ESSL and to the CRAY routines
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!===============================================================================

    SUBROUTINE FFTBRC_(C,GRID,ISN)
      USE prec
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d) GRID
      COMPLEX(q) C(0:1)
      WRITE(*,*)"FFTBRC: is not implemented"
      STOP
      RETURN
    END SUBROUTINE FFTBRC_

!===============================================================================
!     3-d fast fourier transformation for wavefunctions
!  this routine must be called only via FFTWAV and FFTEXT
!  if this is not the case results are unpredictable
!  any scaling should be done on the reduced linear grid in the routines
!  FFTWAV and FFTEXT, this save time
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)    complex to real
!     -1  r->q   vq= sum(r) vr exp(-iqr)    real to complex
!
!===============================================================================

      SUBROUTINE FFT3D(C,GRID,ISN)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)  GRID
#ifdef wNGZhalf
!-------------------------------------------------------------------------------
!  real mode version wWGZhalf
!  this version requires only a complex 3d-FFT
!  the inversion through the midpoint is avoided by a trick:
!   the elements with z!=0 have to be multiplied by a factor 2
!   before calling FFT3D
!   in FFT3D the FFT is performed and the result is made real
!   the multiplication with the scaling factor is performed in FFTWAV
!-------------------------------------------------------------------------------
      COMPLEX(q) C(0:GRID%NGPTAR(1)*GRID%NGPTAR(2)*GRID%NGPTAR(3)-1)
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      CALL FFTBAS(C,NX,NY,NZ,ISN)

      IF (ISN==1) THEN
        DO M=0,GRID%NGPTAR(1)*GRID%NGPTAR(2)*GRID%NGPTAR(3)-1
            C(M)= REAL( C(M) ,KIND=q)
        ENDDO
      ENDIF
#elif defined(wNGXhalf)
!-------------------------------------------------------------------------------
!  real mode version wWGXhalf
!  this version requires a real 3d-FFT
!-------------------------------------------------------------------------------
      REAL(q) C(1)

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      INC3X=(NX/2+1)*NY
      INC2X= NX/2+1

      IF (ISN==1) THEN
        CALL FFTBRC(C,NX,NY,NZ,ISN)
!       concat  x-lines (go from stride NX+2 to NX)
!DIR$ IVDEP
!OCL NOVREC
        DO IL=NY*NZ-1,0,-1
        NDEST=IL*NX*2
        NSRC =IL*(NX+2)
!DIR$ IVDEP
!OCL NOVREC
          DO  NXX=NX,1,-1
            C(NDEST+NXX*2-1)=C(NSRC+NXX)
            C(NDEST+NXX*2)=0
          ENDDO
        ENDDO
      ELSE
!     r->q FFT
!       x-lines (go from complex stride NX to real stride NX+2)
!DIR$ IVDEP
!OCL NOVREC
        DO IL=0,NY*NZ-1
        NSRC =IL*NX*2
        NDEST=IL*(NX+2)
!DIR$ IVDEP
!OCL NOVREC
          DO NXX=1,NX
            C(NDEST+NXX)=C(NSRC+NXX*2-1)
          ENDDO
        ENDDO
        CALL FFTBRC(C,NX,NY,NZ,ISN)
      ENDIF
      RETURN
#else
!-------------------------------------------------------------------------------
!  complex version
!-------------------------------------------------------------------------------
      COMPLEX(q) C(0:1)
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      CALL FFTBAS(C,NX,NY,NZ,ISN)
      RETURN
#endif
      END

!************************* SUBROUTINE FFTINI ***************************
!
!  if necessary this routine performes initialization
!  for FFTWAV and FFTEXT
!  usually this is only necessary for the Gamma point only
!  1-kpoint version
!
!***********************************************************************

      SUBROUTINE  FFTINI(NINDPW,NPLWKP,NKPTS,NRPLW,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d)  GRID
      DIMENSION NPLWKP(NKPTS)
      DIMENSION NINDPW(NRPLW,NKPTS)

#if defined(wNGZhalf) ||  defined(wNGXhalf)

      IF (NKPTS>1) THEN
        WRITE(*,*)'FFT3D: real version works only for 1 k-point'
        STOP
      ENDIF

      NK=1
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      NPL=NPLWKP(NK)
      NULLIFY(GRID%FFTSCA)
      ALLOCATE(GRID%FFTSCA(NPL,2))

      DO N=1,NPL
        IND=NINDPW(N,NKPTS)
        N1= MOD((IND-1),GRID%RC%NROW)+1
        NC= (IND-1)/GRID%RC%NROW+1
        N2= GRID%RC%I2(NC)
        N3= GRID%RC%I3(NC)

        FACTM=SQRT(2._q)
        IF (N==1) FACTM=1
        GRID%FFTSCA(N,1)= FACTM
#ifdef  wNGZhalf
        GRID%FFTSCA(N,2)= FACTM
#else
        GRID%FFTSCA(N,2)= 1/FACTM
        IF (N1==1) GRID%FFTSCA(N,2)= FACTM
#endif
      ENDDO

      RETURN
!-------------------------------------------------------------------------------
!  complex version
!  no initialization necessary
!-------------------------------------------------------------------------------
#else
      RETURN
#endif
      END


!************************* SUBROUTINE FFTWAV ***************************
!  this subroutine transforms a wavefunction C defined  within  the
!  cutoff-sphere to real space CR
! MIND:
! for the real version (gamma point only) it is assumed
! that the wavefunctions at NGX != 0 (wNGXhalf) or NGZ != 0 (wNGZhalf)
! are multiplied by a factor sqrt(2) on the linear grid
! this factor has to be removed before the FFT transformation !
! (scaling with   FFTSCA(M,2))
!***********************************************************************
      SUBROUTINE FFTWAV(NPL,NINDPW,CR,C,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRID
      DIMENSION C(NPL),CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)

      DO M=1,GRID%NPLWV
        CR(M)=(0.0_q,0.0_q)
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO M=1,NPL
#if defined(wNGZhalf) ||  defined(wNGXhalf)
        CR(NINDPW(M))=C(M)*GRID%FFTSCA(M,2)
#else
        CR(NINDPW(M))=C(M)
#endif
      ENDDO
! here you can make what you want - safety first ...
      CALL FFT3D(CR,GRID,1)
! ... or 'fastness' first (but often it is not   so   much faster ...):
!      CALL FFTQ2Q(CR,GRID%NGPTAR(1),1)
      RETURN
      END

!************************* SUBROUTINE FFTEXT ***************************
!
! this subroutine performes a FFT to reciprocal space and extracts data
! from the FFT-mesh
! MIND:
! for the real version (gamma point only) it is assumed
! that the wavefunctions at NGX != 0 (wNGXhalf) or NGZ != 0 (wNGZhalf)
! are multiplied by a factor sqrt(2) on the linear grid
! this factor has to be applied after the FFT transformation !
!  (scaling with   FFTSCA(M))
!***********************************************************************
      SUBROUTINE FFTEXT(NPL,NINDPW,CR,C,GRID,LADD)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      DIMENSION C(NPL),CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)
      LOGICAL   LADD

      CALL FFT3D(CR,GRID,-1)

      IF (LADD) THEN
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
#if defined(wNGZhalf) ||  defined(wNGXhalf)
          C(M)=C(M)+CR(NINDPW(M))*GRID%FFTSCA(M,1)
#else
          C(M)=C(M)+CR(NINDPW(M))
#endif
        ENDDO
      ELSE
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
#if defined(wNGZhalf) ||  defined(wNGXhalf)
          C(M)=CR(NINDPW(M))*GRID%FFTSCA(M,1)
#else
          C(M)=CR(NINDPW(M))
#endif
        ENDDO
      ENDIF
      RETURN
      END


!===============================================================================
!    3-d fast fourier transform (possibly real to complex and vice versa)
!    for chardensities and potentials
!     +1  q->r   vr= sum(q) vq exp(+iqr)    (might be complex to real)
!     -1  r->q   vq= sum(r) vr exp(-iqr)    (might be real to complex)
!
!===============================================================================


      SUBROUTINE FFT3D(C,GRID,ISN)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)   GRID
#ifdef realmode
#ifdef NGXhalf
!-------------------------------------------------------------------------------
!  real mode version NGXhalf
!-------------------------------------------------------------------------------
      REAL(q) C(1)

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      INC3X=(NX/2+1)*NY
      INC2X= NX/2+1
!     q->r FFT
      IF (ISN==1) THEN
        CALL FFTBRC(C,NX,NY,NZ,ISN)
!       concat  x-lines (go from stride NX+2 to NX)
!DIR$ IVDEP
!OCL NOVREC
        DO IL=0,NY*NZ-1
        NDEST=IL*NX
        NSRC =IL*(NX+2)
!DIR$ IVDEP
!OCL NOVREC
          DO NXX=1,NX
            C(NDEST+NXX)=C(NSRC+NXX)
          ENDDO
        ENDDO
      ELSE
!     r->q FFT
!       x-lines (go from stride NX to NX+2)
!DIR$ IVDEP
!OCL NOVREC
        DO IL=NY*NZ-1,0,-1
        NSRC =IL*NX
        NDEST=IL*(NX+2)
!DIR$ IVDEP
!OCL NOVREC
          DO NXX=NX,1,-1
            C(NDEST+NXX)=C(NSRC+NXX)
          ENDDO
        ENDDO
        CALL FFTBRC(C,NX,NY,NZ,ISN)
      ENDIF
      RETURN
#elif defined(NGZhalf)
!-------------------------------------------------------------------------------
!  real mode version NGZhalf
!  this version requires an work array and uses a complex 3d-FFT
!  the inversion through the midpoint  is ommited by multiplying the Z!=0
!  elements with a factor 2
!-------------------------------------------------------------------------------
      REAL(q) C(1)
      REAL(q),ALLOCATABLE:: DWORK(:)

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      NXYZ=NX*NY*NZ
      ALLOCATE(DWORK(NXYZ*2))

!     q->r FFT
      IF (ISN==1) THEN
        DO I=1,NX*NY*(NZ+2)
          DWORK(I)=C(I)
        ENDDO
        DO I=NX*NY*(NZ+2)+1,2*NX*NY*NZ
          DWORK(I)=0
        ENDDO
        CALL MULZ(DWORK,NX,NY,NZ,2.0_q)
        CALL FFTBAS(DWORK,NX,NY,NZ,ISN)

        DO I=1,NXYZ
!          C(I)=CWORK(I)
          C(I)=DWORK(I*2-1)
        ENDDO
      ELSE
!     r->q FFT
        DO I=1,NXYZ
!          CWORK(I)=C(I)
          DWORK(I*2-1)= REAL( C(I) ,KIND=q)
          DWORK(I*2)  =0
        ENDDO
        CALL FFTBAS(DWORK,NX,NY,NZ,ISN)
        DO I=1,NX*NY*(NZ+2)
          C(I)=DWORK(I)
        ENDDO
      ENDIF
      DEALLOCATE(DWORK)
      RETURN
#else
      WRITE(*,*)'Version not implemented'
      STOP
#endif
#else
!-------------------------------------------------------------------------------
!  complex mode version
!-------------------------------------------------------------------------------
      COMPLEX(q) C(0:1)
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
      CALL FFTBAS(C,NX,NY,NZ,ISN)
#endif
      RETURN
      END

!************************ SUBROUTINE MULZ    ***************************
!
!  this subroutine multiplies the Z!=0 components by a factor FACT
!  or divides
!
!***********************************************************************

      SUBROUTINE MULZ(C,NGX,NGY,NGZ,FACT)
      USE prec
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:NGX-1,0:NGY-1,0:NGZ-1)

      DO N3=1,NGZ/2-1
       DO N2=0,NGY-1
        DO N1=0,NGX-1
          C(N1,N2,N3)= C(N1,N2,N3)*FACT
      ENDDO; ENDDO; ENDDO

      RETURN
      END






!===============================================================================
! RCS:  $Id: fft3dnec.F,v 1.2 2001/04/12 17:33:45 kresse Exp $
!
!   basic complex 3-d fast fourier transform using CRAY FFT3D
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!===============================================================================


      SUBROUTINE FFTBAS(C,NX,NY,NZ,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)

      COMPLEX*16 WK(NX*NY*NZ)
      INTEGER IFAX(60)
      DOUBLE PRECISION TRIGS(2*(NX+NY+NZ))

      SCALE=1.0_q

      IF(ISN.EQ.1) THEN
      ISW=-1
      ELSE
      ISW=+1
      ENDIF

      CALL LZFC3FB(NX,NY,NZ,C,NX,NY,NZ,ISW,IFAX,TRIGS,WK,IERR)

      RETURN
      END

!===============================================================================
!   basic real to complex 3-d fast fourier transformation routine using CRAY
!   routines
!   C is used as input and output array
!   in real space C is defined as
!    REAL(q)    C(1:N(1)+2  ,1:N(2),1:N(3)
!   in reciprocal space C is defined as
!    COMPLEX(q) C(1:N(1)/2+1,1:N(2),1:N(3)
!   this conforms to the ESSL and to the CRAY routines
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!===============================================================================


      SUBROUTINE FFTBRC(RC,NX,NY,NZ,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
!     COMPLEX(q) C(NX/2+1,NY,NZ)
      DOUBLE PRECISION RC(NX+2,NY,NZ)
      PARAMETER (NMAXM=64)

      DOUBLE PRECISION TRIGS,WK
      INTEGER IFAX
      dimension TRIGS(NX+2*(NY+NZ)),WK((NX+2)*NY*NZ)
      dimension IFAX(60)


      INC3X=(NX/2+1)*NY
      INC2X= NX/2+1
      SCALE=1.0_q
      IF(ISN.EQ.1) THEN
      ISW=-1
      ELSE
      ISW=+1
      ENDIF

      CALL LDFR3FB(NX,NY,NZ,RC,INC2X*2,NY,NZ,ISW,IFAX,TRIGS,WK,IERR)

      RETURN
      END




!=======================================================================
!   this routine returns the next correct setting for the
!   three dimensional FFT
!   I have not checked whether the CRAY supports prime factors 2,3 and 5
!   because I had no documentation on hand when wrote the routine
!=======================================================================

      SUBROUTINE FFTCHK(NFFT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION NFFT(3)
      LOGICAL FFTCH1

      DO 100 IND=1,3
  200 CONTINUE
        IF (FFTCH1(NFFT(IND))) GOTO 100
        NFFT(IND)=NFFT(IND)+1
        GOTO 200
  100 CONTINUE
      END

      LOGICAL FUNCTION FFTCH1(NIN)
      USE prec
      LOGICAL, EXTERNAL :: FFTCHK_FURTH
      FFTCH1=FFTCHK_FURTH(NIN)
      END FUNCTION
