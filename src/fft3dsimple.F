#include "symbol.inc"
#undef FFTMAKEPLAN
#undef FFT3D
#undef FFT3DR 
#undef FFTCHK
#undef FFT3RC
#undef FFTEXT
#undef FFTWAV
#undef FFTINI

!===============================================================================
! RCS:  $Id: fft3dsimple.F,v 1.2 2002/08/14 13:59:38 kresse Exp $
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


!************************* SUBROUTINE FFTINI ***************************
!
!  if necessary this routine performes initialization
!  for FFTWAV and FFTEXT
!  usually this is only necessary for the Gamma point only
!  single  k-point version
!
!   FFTSCA(.,1) is the scaling factor for extracting the wavefunction
!               from the FFT grid (FFTEXT)
!   FFTSCA(.,2) is the scaling factor for puting the wavefunction on
!               the grid
!
!***********************************************************************

    SUBROUTINE  FFTINI(NINDPW,NPLWKP,NKPTS,NRPLW,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT NONE

      TYPE (grid_3d)  GRID
      INTEGER :: NRPLW               ! maximum number of plane wave coefficients
      INTEGER :: NKPTS               ! number of k-points
      INTEGER :: NINDPW(NRPLW,NKPTS) ! index array from coefficients in sphere to 3D grid
      INTEGER :: NPLWKP(NKPTS)       ! actual number of plane wave coefficients for each k-points
    ! local
      INTEGER :: NK, NX, NY, NZ, NPL, N, IND, N1, N2, N3, NC
      INTEGER :: N2INV, N3INV, NUMBER_OF_CONJG, NCP
      REAL(q) :: FACTM

      IF (GRID%REAL2CPLX) THEN

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
#define inversion_through_mid_point
#ifdef inversion_through_mid_point
         NUMBER_OF_CONJG=0
         DO N=1,NPL
            IND=NINDPW(N,NKPTS)
            N1= MOD((IND-1),GRID%RC%NROW)+1
            NC= (IND-1)/GRID%RC%NROW+1
            N2= GRID%RC%I2(NC)
            N3= GRID%RC%I3(NC)
            IF (N1==1 .AND. N/=1) THEN
               NUMBER_OF_CONJG=NUMBER_OF_CONJG+1
               ! invert second and third index
               N2INV=MOD(-GRID%LPCTY(N2)+GRID%NGY,GRID%NGY)+1
               N3INV=MOD(-GRID%LPCTZ(N3)+GRID%NGZ,GRID%NGZ)+1
               ! now determine the corresponding column index
               NCP=N2INV+(N3INV-1)*GRID%NGY
               ! check whether correct
               IF ( GRID%LPCTY(N2)+GRID%LPCTY(GRID%RC%I2(NCP)) /=0 .OR.  GRID%LPCTZ(N3)+GRID%LPCTZ(GRID%RC%I3(NCP))/=0) THEN
                  WRITE(0,*) 'internal error in FFTINI: could not determine conjugated coefficient',GRID%LPCTY(N2),GRID%LPCTY(GRID%RC%I2(NCP)),GRID%LPCTZ(N3)+GRID%LPCTZ(GRID%RC%I3(NCP))
                  STOP
               ENDIF
            ENDIF
         ENDDO
         NULLIFY(GRID%NINDPWCONJG, GRID%IND_IN_SPHERE)
         ! allocate the required index arrays
         ALLOCATE(GRID%NINDPWCONJG(NUMBER_OF_CONJG), GRID%IND_IN_SPHERE(NUMBER_OF_CONJG))
#endif
         NUMBER_OF_CONJG=0
         DO N=1,NPL
            IND=NINDPW(N,NKPTS)
            N1= MOD((IND-1),GRID%RC%NROW)+1
            NC= (IND-1)/GRID%RC%NROW+1
            N2= GRID%RC%I2(NC)
            N3= GRID%RC%I3(NC)
            FACTM=SQRT(2._q)
            IF (N==1) FACTM=1
            ! in the real version the coefficients stored in the compressed mode (sphere)
            ! are multiplied by a factor sqrt(2) compared to complex version (except at Gamma)
            ! this allows to use DGEMM calls to calculate inproducts
            GRID%FFTSCA(N,1)= FACTM
            GRID%FFTSCA(N,2)= 1/FACTM
#ifdef inversion_through_mid_point
            IF (N1==1 .AND. N/=1) THEN
               NUMBER_OF_CONJG=NUMBER_OF_CONJG+1
               ! invert second and third index
               N2INV=MOD(-GRID%LPCTY(N2)+GRID%NGY,GRID%NGY)+1
               N3INV=MOD(-GRID%LPCTZ(N3)+GRID%NGZ,GRID%NGZ)+1
               ! now determine the corresponding column index
               NCP=N2INV+(N3INV-1)*GRID%NGY
               ! store original index in the compressed storage mode (sphere)
               GRID%IND_IN_SPHERE(NUMBER_OF_CONJG)=N
               ! store index to the 3D grid
               GRID%NINDPWCONJG(NUMBER_OF_CONJG)=N1+(NCP-1)*GRID%RC%NROW
            ENDIF
#else
            ! in the simple version, for the index nx=0
            ! only half of the FFT components are set on the 3D grid by FFTWAV
            ! however the complex to real FFT's (reciprocal space to real space)
            ! expects that the entire x=0 plane is set
            ! multiplying in a factor 2/sqrt(2) yields the correct result for many FFT versions
            IF (N1==1) GRID%FFTSCA(N,2)=FACTM
#endif
         ENDDO
      ENDIF

    END SUBROUTINE FFTINI


!************************* SUBROUTINE FFTWAV ***************************
!  this subroutine transforms a wavefunction C defined  within  the
!  cutoff-sphere to real space CR
! MIND:
! for the real version (gamma point only) it is assumed
! that the wavefunctions at NGX != 0 (wNGXhalf)
! are multiplied by a factor sqrt(2) on the reduced plane wave grid
! this factor has to be removed before the FFT transformation !
! (scaling with   FFTSCA(M,2))
!
!***********************************************************************

    SUBROUTINE FFTWAV(NPL,NINDPW,CR,C,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRID
      COMPLEX(q) :: C(NPL), CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)

      PROFILING_START('fftwav')

      ! zero all elements on the grid gK 021206
!DIR$ IVDEP
!OCL NOVREC
      DO M=1,GRID%NGX_rd*GRID%NGY_rd*GRID%NGZ_rd
         CR(M)=(0.0_q,0.0_q)
      ENDDO
      ! old version to zero all elements
!     DO M=1,GRID%NPLWV
!        CR(M)=(0.0_q,0.0_q)
!     ENDDO
      ! now fill in non zero elements from
      ! within the radial cutoff sphere
      IF (GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            CR(NINDPW(M))=C(M)*GRID%FFTSCA(M,2)
         ENDDO
#ifdef inversion_through_mid_point
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,SIZE(GRID%IND_IN_SPHERE)
            CR(GRID%NINDPWCONJG(M))=CONJG(C(GRID%IND_IN_SPHERE(M)))*GRID%FFTSCA(GRID%IND_IN_SPHERE(M),2)
         ENDDO
#endif
      ELSE
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            CR(NINDPW(M))=C(M)
         ENDDO
      ENDIF
! here you can make what you want - safety first ...
      CALL FFT3D(CR,GRID,1)
! ... or 'fastness' first (but often it is not   so   much faster ...):
!      CALL FFTQ2Q(CR,GRID%NGPTAR(1),1)

      PROFILING_STOP('fftwav')

      RETURN
    END SUBROUTINE FFTWAV

!************************* SUBROUTINE FFTEXT ***************************
!
! this subroutine performes a FFT to reciprocal space and extracts data
! from the FFT-mesh
! MIND:
! for the real version (gamma point only) it is assumed
! that the wavefunctions at NGX != 0
! are multiplied by a factor sqrt(2) on the linear grid
! this factor has to be applied after the FFT transformation !
!  (scaling with   FFTSCA(M))
!
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

      PROFILING_START('fftext')

      CALL FFT3D(CR,GRID,-1)

      IF (LADD .AND. GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            C(M)=C(M)+CR(NINDPW(M))*GRID%FFTSCA(M,1)
         ENDDO
      ELSE IF (LADD .AND. .NOT. GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            C(M)=C(M)+CR(NINDPW(M))
         ENDDO
      ELSE IF (GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
          C(M)=CR(NINDPW(M))*GRID%FFTSCA(M,1)
        ENDDO
     ELSE
!DIR$ IVDEP
!OCL NOVREC
        DO M=1,NPL
          C(M)=CR(NINDPW(M))
        ENDDO
      ENDIF

      PROFILING_STOP('fftext')

      RETURN
    END SUBROUTINE FFTEXT

!===============================================================================
!
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
      REAL(q) C(1)

      PROFILING_START('fft3d')

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)

!-------------------------------------------------------------------------------
!  complex to complex version
!-------------------------------------------------------------------------------
      IF (.NOT. GRID%REAL2CPLX .AND. .NOT. GRID%LREAL ) THEN
         IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 1 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF
         CALL FFTBAS(C,GRID,ISN)
!-------------------------------------------------------------------------------
!  complex to complex version, but with a real array in real space
!-------------------------------------------------------------------------------
      ELSE IF (.NOT. GRID%REAL2CPLX .AND. GRID%LREAL) THEN
         IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 2 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF

         !     q->r FFT
         IF (ISN==1) THEN
            CALL FFTBAS(C,GRID,ISN)

            !  go from complex stride 2 to 1
!DIR$ IVDEP
!OCL NOVREC
            DO IL=0,NX*NY*NZ-1
               NDEST=IL+1
               NSRC =IL*2+1
               C(NDEST)=C(NSRC)
            ENDDO
         ELSE

            !     r->q FFT
            !  go from stride 1 to stride 2
!DIR$ IVDEP
!OCL NOVREC
            DO IL=NX*NY*NZ-1,0,-1
               NSRC =IL+1
               NDEST=IL*2+1
               C(NDEST)=C(NSRC)
               C(NDEST+1)=0
            ENDDO
            CALL FFTBAS(C,GRID,ISN)
         ENDIF
!-------------------------------------------------------------------------------
!  real to complex FFT  only half grid mode in X direction supported
!  data are stored as real array in real space
!-------------------------------------------------------------------------------
      ELSE IF (GRID%LREAL) THEN
         IF (.NOT. (NX/2+1==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 3 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF

         !  in real space the first dimension in VASP is NGX (REAL data)
         !  but the FFT required NGX+2 (real data)
         !  therefore some data movement is required

         INC3X=(NX/2+1)*NY
         INC2X= NX/2+1

         !     q->r FFT
         IF (ISN==1) THEN
            CALL FFTBRC(C,GRID,ISN)
            !  concat  x-lines (go from stride NX+2 to NX)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=1,NY*NZ-1
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
            !     x-lines (go from stride NX to NX+2)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=NY*NZ-1,1,-1
               NSRC =IL*NX
               NDEST=IL*(NX+2)
! ifc10.1 has troubles with vectorizing this statment
!!DIR$ IVDEP
!!OCL NOVREC
               DO NXX=NX,1,-1
                  C(NDEST+NXX)=C(NSRC+NXX)
               ENDDO
            ENDDO
            CALL FFTBRC(C,GRID,ISN)
         ENDIF
!-------------------------------------------------------------------------------
! same as above (real to complex FFT) but this time the data layout
! is complex in real space
!-------------------------------------------------------------------------------
      ELSE
         IF (.NOT. (NX/2+1==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 4 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF
         INC3X=(NX/2+1)*NY
         INC2X= NX/2+1

         IF (ISN==1) THEN
             CALL FFTBRC(C,GRID,ISN)
            ! concat  x-lines (go from "real" stride NX+2 to complex stride NX)
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
            !     x-lines (go from complex stride NX to real stride NX+2)
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
            CALL FFTBRC(C,GRID,ISN)
         ENDIF
      ENDIF
      
      PROFILING_STOP('fft3d')

      RETURN
    END SUBROUTINE FFT3D

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
      ENDDO
      ENDDO 
      ENDDO

      RETURN
    END SUBROUTINE MULZ






