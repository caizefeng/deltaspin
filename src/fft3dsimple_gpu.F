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

! GPU part : HACENE Mohamed
    module GPU_FFT_DATA
        use iso_c_binding
        implicit none
        INTEGER(c_intptr_t) ::     GPU_FFTSCA
    end module GPU_FFT_DATA

    SUBROUTINE FFTBAS_(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
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

    SUBROUTINE FFTBRC_(C,N,ISN)
      USE prec
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      COMPLEX(q) C(0:1)
      DIMENSION N(3)
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
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE mpimy
      USE mgrid

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d)  GRID
      DIMENSION NPLWKP(NKPTS)
      DIMENSION NINDPW(NRPLW,NKPTS)

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
         
         DO N=1,NPL
            IND=NINDPW(N,NKPTS)
            N1= MOD((IND-1),GRID%RC%NROW)+1
            NC= (IND-1)/GRID%RC%NROW+1
            N2= GRID%RC%I2(NC)
            N3= GRID%RC%I3(NC)
            
            IF (N1/=MOD((IND-1),NX/2+1)+1) THEN
               WRITE(*,*) 'internal error in FFTINI: old and new version are different'
               STOP
            ENDIF
            
            FACTM=SQRT(2._q)
            IF (N==1) FACTM=1
            GRID%FFTSCA(N,1)= FACTM
            GRID%FFTSCA(N,2)= 1/FACTM
            ! this statment is required
            ! because for x==0 only half of the FFT components are set
            ! upon calling FFTWAV but the real to complex FFT's
            ! usually assume that the entire x=0 plane is set
            IF (N1==1) GRID%FFTSCA(N,2)= FACTM
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
#ifndef CUFFT_MIN
    SUBROUTINE FFTWAV(NPL,NINDPW,CR,C,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRID
      COMPLEX(q) :: C(NPL), CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)

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
      !write(*,*) "zc=",GRID%NPLWV
      !do ii=1,GRID%NPLWV
      !  write(*,*) CR(ii)
      !enddo
      !call exit()
      
    END SUBROUTINE FFTWAV
#endif

! DDNVIDIA FFTWAV CUDA streams implementation    
    SUBROUTINE FFTWAV_GPU(SID,NPL,GPU_NINDPW,GPU_CR,shiftCR,GPU_C,shiftC,GRID)
      USE iso_c_binding
      USE cuda_interface
      USE GPU_FFT_DATA
      USE prec
      USE mpimy
      USE mgrid
      
      IMPLICIT NONE

      !inputs
      INTEGER           SID
      INTEGER           NPL
      INTEGER(c_intptr_t)        GPU_NINDPW
      INTEGER(c_intptr_t) ::     GPU_CR
      INTEGER           shiftCR
      INTEGER(c_intptr_t) ::     GPU_C
      INTEGER           shiftC
      TYPE (grid_3d)    GRID
      !locals
      COMPLEX(q) fakec
      REAL(q) faker
      INTEGER   :: REAL2CPLX_INT

      !TODO(ca): fixed a bug where GPU_FFTSCA was completely uninitialized;
      !however, check whether we can reuse fftsca to avoid creating, copying,
      !and deleting it in every call
      !NOTE: don't forget the free on the bottom of this subroutine
      if(GRID%REAL2CPLX) then
          if(associated(GRID%FFTSCA) .eqv. .False.) then
              print *, "GRID%FFTSCA is not associated! Something went wrong"
              stop
          end if
          call cublas_alloc_safety(NPL, int(c_sizeof(faker),c_size_t), GPU_FFTSCA)
          call cuda_memcpyhtod(SID, GPU_FFTSCA, c_loc(GRID%FFTSCA),NPL, int(c_sizeof(faker),c_size_t))
      endif

      ! zero all elements on the grid gK 021206
      call cuda_memsetasync(SID,GPU_CR+shiftCR*int(c_sizeof(fakec),c_size_t),0,GRID%NGX_rd*GRID%NGY_rd*GRID%NGZ_rd,c_sizeof(fakec))
      !CALL gpu_initz(SID,GRID%NGX_rd*GRID%NGY_rd*GRID%NGZ_rd,GRID%NPLWV,GPU_CR,shiftCR,zero)
      
      !write(*,*) "FFTWAV_GPU..."
      ! now fill in non zero elements from
      ! within the radial cutoff sphere
#if 1
      if(GRID%REAL2CPLX) then
          REAL2CPLX_INT = 1
      else
          REAL2CPLX_INT = 0
      endif
      !NOTE: GPU_FFTSCA is only read if REAL2CPLX_INT is true; otherwise we
      !are passing an invalid (null) pointer
      call cuda_fftwav(SID,NPL,GPU_CR+shiftCR*int(c_sizeof(fakec),c_size_t),GPU_C+shiftC*c_sizeof(fakec),GPU_NINDPW,GPU_FFTSCA,REAL2CPLX_INT)
#else
      IF(GRID%REAL2CPLX) 
         CALL gpu_crcsr(SID,NPL,GPU_CR,shiftCR,GPU_C,shiftC,GPU_FFTSCA,IDX(1,2,NPL),GPU_NINDPW)
      ELSE
         CALL gpu_permut(SID,NPL,GPU_CR,shiftCR,GPU_C,shiftC,GPU_NINDPW,0)
      ENDIF
#endif

      !call cuda_print("cr.dat",'c',NPL,GPU_CR)
      !call cuda_device_reset()
      !stop

      CALL FFT3D_GPU(SID,GPU_CR,shiftCR,GRID,1)
      if(GRID%REAL2CPLX) then
        call cublas_free(GPU_FFTSCA)
      endif
    END SUBROUTINE FFTWAV_GPU

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


#ifndef CUFFT_MIN
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
      RETURN
    END SUBROUTINE FFTEXT
#endif

! DDNVIDIA FFTEXT CUDA streams implementation
    SUBROUTINE FFTEXT_GPU(SID,NPL,GPU_NINDPW,GPU_CR,SHIFT_CR,GPU_C,SHIFT_C,GRID,LADD)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT NONE

      ! inputs
      INTEGER           SID
      INTEGER           NPL
      INTEGER(c_intptr_t)        GPU_NINDPW
      INTEGER(c_intptr_t)        GPU_CR
      INTEGER           SHIFT_CR
      INTEGER(c_intptr_t)        GPU_C
      INTEGER(c_intptr_t)        GPU_FFTSCA
      INTEGER           SHIFT_C
      TYPE (grid_3d)    GRID
      LOGICAL           LADD
      INTEGER           LADD_INT
      INTEGER           REAL2CPLX_INT
      ! locals
      COMPLEX(q) fakec

      CALL FFT3D_GPU(SID,GPU_CR,SHIFT_CR,GRID,-1)

      !write(*,*) "FFTEXT_GPU..."
      !write(*,*) "LADD=",LADD
#if 1
      if(LADD) then
        LADD_INT = 1
      else
        LADD_INT = 0
      endif
      if(GRID%REAL2CPLX) then
          REAL2CPLX_INT = 1
      else
          REAL2CPLX_INT = 0
      endif
      !write(*,*) "calling cuda_fftext..."
      !TODO(ca): I think this subroutine contains a bug: GPU_FFTSCA is not
      !initialized anywhere!
      call cuda_fftext(SID,NPL,GPU_C+SHIFT_C*int(c_sizeof(fakec),c_size_t),GPU_CR+SHIFT_CR*c_sizeof(fakec),GPU_NINDPW,GPU_FFTSCA,LADD_INT,REAL2CPLX_INT)
#else
      IF (LADD .AND. GRID%REAL2CPLX) THEN
         call GPU_LADD_REAL2CPLX(SID,NPL,GPU_C,SHIFT_C,GPU_CR,SHIFT_CR,GPU_FFTSCA,0,GPU_NINDPW)
      ELSE IF (LADD .AND. .NOT. GRID%REAL2CPLX) THEN
         call GPU_LADD_NOTREAL2CPLX(SID,NPL,GPU_C,SHIFT_C,GPU_CR,SHIFT_CR,GPU_NINDPW)
      ELSE IF (GRID%REAL2CPLX) THEN
         call GPU_REAL2CPLX(SID,NPL,GPU_C,SHIFT_C,GPU_CR,SHIFT_CR,GPU_FFTSCA,0,GPU_NINDPW)
      ELSE
         call gpu_permut(SID,NPL,GPU_C,SHIFT_C,GPU_CR,SHIFT_CR,GPU_NINDPW,1)
      ENDIF
#endif
      !call cuda_print("c.dat",'c',NPL,GPU_C)
      !call cuda_device_reset()
      !stop

      RETURN
    END SUBROUTINE FFTEXT_GPU

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
         CALL FFTBAS(C,GRID%NGPTAR,ISN)
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
            CALL FFTBAS(C,GRID%NGPTAR(1),ISN)

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
            CALL FFTBAS(C,GRID%NGPTAR(1),ISN)
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
            CALL FFTBRC(C,GRID%NGPTAR(1),ISN)

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
            CALL FFTBRC(C,GRID%NGPTAR(1),ISN)
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
            CALL FFTBRC(C,GRID%NGPTAR(1),ISN)
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
            CALL FFTBRC(C,GRID%NGPTAR(1),ISN)
         ENDIF
      ENDIF
      
    END SUBROUTINE FFT3D
 
#if 0
!! Added by Arash
    SUBROUTINE FFT3D_PIPELINED_PIN(GRID,ISN,NP,GPU_CR,shift)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)   GRID
      INTEGER(c_intptr_t) :: GPU_CR
      INTEGER shift

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)

!-------------------------------------------------------------------------------
!  complex to complex version
!-------------------------------------------------------------------------------
         IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd)) THEN
            WRITE(0,*) 'internal error 1 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF

         CALL FFTBASPIPELINED_PIN(GRID%NGPTAR,ISN,NP,GPU_CR,shift)
    END SUBROUTINE FFT3D_PIPELINED_PIN
#endif

    SUBROUTINE FFT3D_GPU(SID,GPU_C,shiftC,GRID,ISN)
      USE prec
      USE mpimy
      USE mgrid
      USE iso_c_binding
      USE cuda_interface
      integer(c_intptr_t) GPU_C
      INTEGER shiftC
      TYPE (grid_3d) GRID
      INTEGER SID

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)
!-------------------------------------------------------------------------------
!  complex to complex version
!-------------------------------------------------------------------------------
      IF(.NOT. GRID%REAL2CPLX .AND. .NOT. GRID%LREAL) THEN
         IF(.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 1 in FFT3D: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF

         call cufft_execz2z( &
                SID, &
                NX, &
                NY, &
                NZ, &
                GPU_C, &
                shiftC, &
                GPU_C, &
                shiftC, &
                ISN)
!-------------------------------------------------------------------------------
!  complex to complex version, but with a real array in real space
!-------------------------------------------------------------------------------
      ELSE IF(.NOT. GRID%REAL2CPLX .AND. GRID%LREAL) THEN
         WRITE(*,*) "FFT3D_GPU ERROR: complex to complex with real array not yet implemented on GPU"
         STOP
!-------------------------------------------------------------------------------
!  real to complex FFT  only half grid mode in X direction supported
!  data are stored as real array in real space
!-------------------------------------------------------------------------------
      ELSE
         WRITE(*,*) "FFT3D_GPU ERROR: complex to real not yet implemented on GPU"
         STOP
      ENDIF
    END SUBROUTINE FFT3D_GPU

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






