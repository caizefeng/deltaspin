#include "symbol.inc"
#ifndef CACHE_SIZE
#define CACHE_SIZE 8000
#endif
#ifndef MINLOOP
#define MINLOOP 1
#endif

#undef FFTMAKEPLAN
#undef FFT3D 
#undef FFT3DR 
#undef FFTCHK
#undef FFT3RC
#undef FFTEXT
#undef FFTWAV
#undef FFTINI

!===============================================================================
!
!  FFTW requires a plan therefore this new calling interface is provided
!  which calls the FFTBAS and FFTBRC routine for generating these
!  plans
!
!
!===============================================================================

      SUBROUTINE FFTMAKEPLAN_MPI(A,GRID)
      USE prec
      USE mgrid
      USE openmp, ONLY : omp_dfftw_init_threads

      include 'fftw3.f'

      TYPE (grid_3d) GRID
      REAL(q) A(*)


!serFFT
      IF (GRID%RL%NFAST==1) THEN
         CALL FFTMAKEPLAN(A,GRID)
         RETURN
      ENDIF
!serFFTend


      CALL FFTBAS_PLAN_MPI(A,GRID, 1,FFTW_MEASURE)
      CALL FFTBAS_PLAN_MPI(A,GRID,-1,FFTW_MEASURE)

      END SUBROUTINE

!-----------------------------------------------------------------------
! RCS:  $Id: fftmpi.F,v 1.3 2002/08/14 13:59:38 kresse Exp $
!
!   3-d parallel fast fourier transformation using fftw
!   written by Georg Kresse
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
! 
!   the FFTBAS_PLAN routine performs both the complex to complex, and
!   complex to real FFT
!
!   the FFTBAS routine is the calling interface for the 
!    complex, complex FFT 
!   whereas FFTBRC is the calling interface for complex to real FFT
!  
!=======================================================================

!
!  this subroutine calls the FFTBAS_PLAN routine  with FFTW_ESTIMATE
!

    SUBROUTINE FFTBAS_MPI(A,GRID,ISIGN)
      USE prec
      USE mgrid

      include 'fftw3.f'

      TYPE (grid_3d) GRID
      REAL(q) A(*)
      INTEGER ISIGN   !  direction of fft

      CALL FFTBAS_PLAN_MPI(A,GRID,ISIGN,FFTW_ESTIMATE)

    END SUBROUTINE


!
    SUBROUTINE FFTBAS_PLAN_MPI(A,GRID,ISIGN,IPLAN)
      USE prec
      USE smart_allocate
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      include 'fftw3.f'

      TYPE (grid_3d) GRID
      REAL(q) A(*)
      INTEGER ISIGN   !  direction of fft
      INTEGER IPLAN   !  make a plan (/=FFTW_ESTIMATE)
      COMPLEX(q),POINTER,SAVE ::  RCVBUF(:),SNDBUF(:)
      INTEGER(8) :: planx, plany, planz


      PROFILING_START('fftbas_plan_mpi')

!$!=======================================================================
!$!  initialise openMP FFT, has to be done here and not in main.F in
!$!  in general, since the FFTs are called outside and inside openMP
!$!  parallel regions.
!$!=======================================================================
!=======================================================================
! initialization
!=======================================================================
      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)

      CALL SMART_ALLOCATE_COMPLEX(RCVBUF,GRID%MPLWV)
      CALL SMART_ALLOCATE_COMPLEX(SNDBUF,GRID%MPLWV)

      IDX=NX
      IDY=NY
      IDZ=NZ

      IF (ISIGN==1) THEN
         CALL dfftw_plan_many_dft(planx, 1, NX , GRID%RC%NCOL, &
                             A(1), NX, 1 , IDX, &
                             A(1), NX, 1 , IDX, &
                             FFTW_BACKWARD, IPLAN)
         CALL dfftw_plan_many_dft(plany, 1, NY , GRID%IN%NCOL, &
                             A(1), NY, GRID%IN%NCOL, 1 , &
                             A(1), NY, GRID%IN%NCOL, 1 , &
                             FFTW_BACKWARD, IPLAN)
         IF (NZ/2+1==GRID%NGZ_rd) THEN
!           WRITE(*,*) 'detected real to complex'
           CALL dfftw_plan_many_dft_c2r(planz, 1, NZ , GRID%RL_FFT%NCOL, &
                             A(1), NZ, 1, (IDZ+2)/2 , &
                             A(1), NZ, 1, IDZ+2 , &
                             IPLAN)
         ELSE
!           WRITE(*,*) 'complex to complex'
           CALL dfftw_plan_many_dft(planz, 1, NZ , GRID%RL_FFT%NCOL, &
                             A(1), NZ, 1, IDZ , &
                             A(1), NZ, 1, IDZ , &
                             FFTW_BACKWARD, IPLAN)
         ENDIF
      ELSE
         IF (NZ/2+1==GRID%NGZ_rd) THEN
!           WRITE(*,*) 'detected inverse real to complex'
           CALL dfftw_plan_many_dft_r2c(planz, 1, NZ , GRID%RL_FFT%NCOL, &
                             A(1), NZ, 1, IDZ+2 , &
                             A(1), NZ, 1, (IDZ+2)/2 , &
                             IPLAN)
         ELSE
!           WRITE(*,*) 'detected inverse complex to complex'
           CALL dfftw_plan_many_dft(planz, 1, NZ , GRID%RL_FFT%NCOL, &
                             A(1), NZ, 1, IDZ , &
                             A(1), NZ, 1, IDZ , &
                             FFTW_FORWARD, IPLAN)
         ENDIF
         CALL dfftw_plan_many_dft(plany, 1, NY , GRID%IN%NCOL, &
                             A(1), NY, GRID%IN%NCOL, 1 , &
                             A(1), NY, GRID%IN%NCOL, 1 , &
                             FFTW_FORWARD, IPLAN)
         CALL dfftw_plan_many_dft(planx, 1, NX , GRID%RC%NCOL, &
                             A(1), NX, 1 , IDX, &
                             A(1), NX, 1 , IDX, &
                             FFTW_FORWARD, IPLAN)
      ENDIF
!=======================================================================
! do the transformation forward (q->r)
!=======================================================================
       IF (ISIGN ==1 .AND. IPLAN==FFTW_ESTIMATE) THEN
! transformation along first dimension:
         PROFILING_START('dfftw_execute')
         CALL dfftw_execute_dft(planx,A(1),A(1))
         PROFILING_STOP('dfftw_execute')
         CALL MAP_FORWARD(A(1), GRID%IN%NALLOC, SNDBUF(1), RCVBUF(1), GRID%RC_IN, GRID%COMM)
! transformation along second dimension:
         PROFILING_START('dfftw_execute')
         CALL dfftw_execute_dft(plany,A(1),A(1))
         PROFILING_STOP('dfftw_execute')
         CALL MAP_FORWARD(A(1), GRID%RL_FFT%NALLOC, SNDBUF(1), RCVBUF(1), GRID%IN_RL, GRID%COMM)
! transformation along third dimension:
         PROFILING_START('dfftw_execute')
         IF (GRID%RL_FFT%NCOL>0) THEN
            IF (NZ/2+1==GRID%NGZ_rd) THEN
               CALL dfftw_execute_dft_c2r(planz,A(1),A(1))
            ELSE
               CALL dfftw_execute_dft(planz,A(1),A(1))
            ENDIF
         ENDIF
         PROFILING_STOP('dfftw_execute')
!=======================================================================
! do the transformation backward (r->q)
!=======================================================================
       ELSE IF(IPLAN==FFTW_ESTIMATE) THEN
! transformation along third dimension:
         PROFILING_START('dfftw_execute')
         IF (GRID%RL_FFT%NCOL>0) THEN
            IF (NZ/2+1==GRID%NGZ_rd) THEN
               CALL dfftw_execute_dft_r2c(planz,A(1),A(1))
            ELSE
               CALL dfftw_execute_dft(planz,A(1),A(1))
            ENDIF
         ENDIF
         PROFILING_STOP('dfftw_execute')
         CALL MAP_BACKWARD(A(1), GRID%IN%NALLOC, SNDBUF(1), RCVBUF(1), GRID%IN_RL, GRID%COMM)
! transformation along second dimension:
         PROFILING_START('dfftw_execute')
         CALL  dfftw_execute_dft(plany,A(1),A(1))
         PROFILING_STOP('dfftw_execute')
         CALL MAP_BACKWARD(A(1), GRID%RC%NALLOC, SNDBUF(1), RCVBUF(1), GRID%RC_IN, GRID%COMM)
! transformation along first dimension:
         PROFILING_START('dfftw_execute')
         CALL dfftw_execute_dft(planx,A(1),A(1))
         PROFILING_STOP('dfftw_execute')
      ENDIF

      call dfftw_destroy_plan(planx)
      call dfftw_destroy_plan(plany)
      call dfftw_destroy_plan(planz)

      PROFILING_STOP('fftbas_plan_mpi')

      RETURN
    END SUBROUTINE FFTBAS_PLAN_MPI


!=======================================================================
!   3-d parallel real to complex fast fourier transformation using 
!   fftw-kernels
!   communication routines and set of communication routines
!   in fftmpi_map.F written by Georg Kresse
!
!     +1  q->r   vr= sum(q) vq exp(+iqr)
!     -1  r->q   vq= sum(r) vr exp(-iqr)
!
!=======================================================================

!
!  this subroutine calls the FFTBAS_PLAN_MPI routine  with FFTW_ESTIMATE
!  the FFTBAS_PLAN_MPI routine detects automatically real to complex
!  transforms and handles them accordingly
!
    SUBROUTINE FFTBRC_MPI(A,GRID,ISIGN)
      USE prec
      USE mgrid
      TYPE (grid_3d) GRID
      REAL(q) A(*)
      INTEGER ISIGN   !  direction of fft
      include 'fftw3.f'

      CALL FFTBAS_PLAN_MPI(A,GRID,ISIGN,FFTW_ESTIMATE)

    END SUBROUTINE FFTBRC_MPI


!***********************************************************************
! FROM HERE ON THE ROUTINES ARE IDENTICAL TO fftmpi.F
!***********************************************************************


!************************* SUBROUTINE FFTINI ***************************
!
!  if necessary this routine performes initialization
!  for FFTWAV and FFTEXT
!  usually this is only necessary for the Gamma point only
!  1-kpoint version
!
!   FFTSCA(.,1) is the scaling factor for extracting the wavefunction
!               from the FFT grid (FFTEXT)
!   FFTSCA(.,2) is the scaling factor for puting the wavefunction on
!               the grid
!***********************************************************************

    SUBROUTINE  FFTINI_MPI(NINDPW,NPLWKP,NKPTS,NRPLW,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d)  GRID
      DIMENSION NPLWKP(NKPTS)
      DIMENSION NINDPW(NRPLW,NKPTS)

      IF (GRID%REAL2CPLX) THEN
         IF (GRID%RL%NFAST==1) THEN
            CALL FFTINI(NINDPW,NPLWKP,NKPTS,NRPLW,GRID)
            RETURN
         ENDIF
         
         IF (NKPTS>1) THEN
            WRITE(*,*)'FFT3D: real version works only for 1 k-point'
            STOP
         ENDIF
         
         NK=1
         NPL=NPLWKP(NK)
         NULLIFY(GRID%FFTSCA)
         ALLOCATE(GRID%FFTSCA(NPL,2))
         
         DO N=1,NPL
            IND=NINDPW(N,NK)
            N1= MOD((IND-1),GRID%RC%NROW)+1
            NC= (IND-1)/GRID%RC%NROW+1
            N2= GRID%RC%I2(NC)
            N3= GRID%RC%I3(NC)
            
            FACTM=SQRT(2._q)
            IF (N1==1 .AND. N2==1 .AND. N3==1) FACTM=1
            GRID%FFTSCA(N,1)= FACTM
            GRID%FFTSCA(N,2)= 1/FACTM
            ! this statment is required
            ! because for z==0 only half of the FFT components are set
            ! upon calling FFTWAV
            IF (N3==1) GRID%FFTSCA(N,2)=FACTM
         ENDDO
      END IF
      RETURN
    END SUBROUTINE FFTINI_MPI

!************************* SUBROUTINE FFTWAV ***************************
!
!  this subroutine transforms a wavefunction C defined  within  the
!  cutoff-sphere to real space CR
! MIND:
! for the real version (gamma point only) it is assumed
! that the wavefunctions at NGZ != 0
! are multiplied by a factor sqrt(2) on the linear grid
! this factor has to be removed before the FFT transformation !
! (scaling with   FFTSCA(M,2))
!
!
!***********************************************************************

    SUBROUTINE FFTWAV_MPI(NPL,NINDPW,CR,C,GRID)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRID
      COMPLEX(q):: C(NPL), CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)

      PROFILING_START('fftwav_mpi')

      IF (GRID%RL%NFAST==1) THEN
         CALL FFTWAV(NPL,NINDPW,CR,C,GRID)
         PROFILING_STOP('fftwav_mpi')
         RETURN
      ENDIF

      IF (GRID%LREAL) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,GRID%RL%NCOL*GRID%NGZ/2
            CR(M)=0.0_q
         ENDDO
      ELSE
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,GRID%RL%NCOL*GRID%NGZ
            CR(M)=0.0_q
         ENDDO
      ENDIF

      IF (GRID%REAL2CPLX) THEN
!DIR$ IVDEP
!OCL NOVREC
         DO M=1,NPL
            CR(NINDPW(M))=C(M)*GRID%FFTSCA(M,2)
         ENDDO
      ELSE
         DO M=1,NPL
            CR(NINDPW(M))=C(M)
         ENDDO
      ENDIF
      CALL FFT3D_MPI(CR,GRID,1)

      PROFILING_STOP('fftwav_mpi')

      RETURN
    END SUBROUTINE FFTWAV_MPI

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
!
!***********************************************************************

    SUBROUTINE FFTEXT_MPI(NPL,NINDPW,CR,C,GRID,LADD)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      DIMENSION C(NPL),CR(GRID%NPLWV)
      DIMENSION NINDPW(NPL)
      LOGICAL   LADD

      PROFILING_START('fftext_mpi')

      CALL FFT3D_MPI(CR,GRID,-1)

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

      PROFILING_STOP('fftext_mpi')

      RETURN
    END SUBROUTINE FFTEXT_MPI


!===============================================================================
!
!    3-d fast fourier transform (possibly real to complex and vice versa)
!    for chardensities and potentials
!     +1  q->r   vr= sum(q) vq exp(+iqr)    (might be complex to real)
!     -1  r->q   vq= sum(r) vr exp(-iqr)    (might be real to complex)
!
!===============================================================================

    SUBROUTINE FFT3D_MPI(C,GRID,ISN)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      
      TYPE (grid_3d)   GRID
      REAL(q) C(*)
      
      PROFILING_START('fft3d_mpi')

      NX=GRID%NGPTAR(1)
      NY=GRID%NGPTAR(2)
      NZ=GRID%NGPTAR(3)

!-------------------------------------------------------------------------------
! use serial FFT
!-------------------------------------------------------------------------------
      IF (GRID%RL%NFAST==1 .AND. GRID%RL_FFT%NFAST==1) THEN
         CALL FFT3D(C, GRID, ISN)
      ELSE IF (GRID%RL%NFAST==1) THEN
!-------------------------------------------------------------------------------
! parallel FFT with serial data layout  (GRID%RL%NFAST==1)
!-------------------------------------------------------------------------------
       ! complex to complex case
         IF (.NOT. GRID%REAL2CPLX) THEN
            IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
               WRITE(0,*) 'internal error 1 in FFT3D_MPI: something not properly set',GRID%LREAL, GRID%REAL2CPLX
               WRITE(0,*) NX, NY, NZ
               WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
               STOP
            ENDIF
            
            !     q->r FFT
            IF (ISN==1) THEN
               CALL FFTBAS_MPI(C,GRID,ISN)
               IF (GRID%COMM%NODE_ME==1) THEN
                  CALL FFTPAR_TO_SER(GRID%NGX, GRID%NGY, GRID%NGZ, C)
               ENDIF
            ELSE
               IF (GRID%COMM%NODE_ME==1) THEN
                  CALL FFTSER_TO_PAR(GRID%NGX, GRID%NGY, GRID%NGZ, C)
               ENDIF
               CALL FFTBAS_MPI(C,GRID,ISN)
            ENDIF
            
            ! complex to real case
         ELSE IF (GRID%LREAL) THEN
            IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ/2+1==GRID%NGZ_rd) ) THEN
               WRITE(0,*) 'internal error 2 in FFT3D_MPI: something not properly set',GRID%LREAL, GRID%REAL2CPLX
               WRITE(0,*) NX, NY, NZ
               WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
               STOP
            ENDIF

            !  in real space the first dimension in VASP is NGZ (REAL data)
            !  but the FFT requires NGZ+2 (real data)
            !  therefore some data movement is required

            NZ=GRID%NGPTAR(3)

            !     q->r FFT
            IF (ISN==1) THEN
               CALL FFTBRC_MPI(C,GRID,ISN)
!DIR$ IVDEP
!OCL NOVREC
               DO IL=1,GRID%RL_FFT%NCOL-1
                  NDEST=IL* NZ
                  NSRC =IL*(NZ+2)
!DIR$ IVDEP
!OCL NOVREC
                  DO NZZ=1,NZ
                     C(NDEST+NZZ)=C(NSRC+NZZ)
                  ENDDO
               ENDDO
               IF (GRID%COMM%NODE_ME==1) THEN
                  CALL FFTPAR_TO_SER_REAL(GRID%NGX, GRID%NGY, GRID%NGZ, C)
               ENDIF
            ELSE
               IF (GRID%COMM%NODE_ME==1) THEN
                  CALL FFTSER_TO_PAR_REAL(GRID%NGX, GRID%NGY, GRID%NGZ, C)
               ENDIF
               !     r->q FFT
               !       x-lines (go from stride NZ to NZ+2)
!DIR$ IVDEP
!OCL NOVREC
               DO IL=GRID%RL_FFT%NCOL-1,1,-1
                  NSRC =IL*NZ
                  NDEST=IL*(NZ+2)
! ifc10.1 has troubles with vectorizing this statment
!!DIR$ IVDEP
!!OCL NOVREC
                  DO NZZ=NZ,1,-1
                     C(NDEST+NZZ)=C(NSRC+NZZ)
                  ENDDO
               ENDDO
               CALL FFTBRC_MPI(C,GRID,ISN)
            ENDIF
         ELSE
            WRITE(0,*) 'ERROR in FFT3D_MPI: this version does not support the required half grid mode'
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF
!-------------------------------------------------------------------------------
!  complex parallel FFT
!-------------------------------------------------------------------------------
      ELSE IF (.NOT. GRID%REAL2CPLX) THEN
         IF (.NOT. (NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 3 in FFT3D_MPI: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF
         CALL FFTBAS_MPI(C,GRID,ISN)
!-------------------------------------------------------------------------------
!  real to complex parallel FFT
!-------------------------------------------------------------------------------
      ELSE IF (GRID%LREAL) THEN
         IF (.NOT.(NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ/2+1==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 4 in FFT3D_MPI: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF
         
         !  in real space the first dimension in VASP is NGZ (REAL data)
         !  but the FFT requires NGZ+2 (real data)
         !  therefore some data movement is required
         
         !     q->r FFT
         IF (ISN==1) THEN
            CALL FFTBRC_MPI(C,GRID,ISN)
            
            !  concat  z-lines (go from stride NZ+2 to NZ)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=1,GRID%RL%NCOL-1
               NDEST=IL* NZ
               NSRC =IL*(NZ+2)
!DIR$ IVDEP
!OCL NOVREC
               DO NZZ=1,NZ
                  C(NDEST+NZZ)=C(NSRC+NZZ)
               ENDDO
            ENDDO
         ELSE

            !     r->q FFT
            !     z-lines (go from stride NZ to NZ+2)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=GRID%RL%NCOL-1,1,-1
               NSRC =IL*NZ
               NDEST=IL*(NZ+2)
! ifc10.1 has troubles with vectorization of this loop
!!DIR$ IVDEP
!!OCL NOVREC
               DO NZZ=NZ,1,-1
                  C(NDEST+NZZ)=C(NSRC+NZZ)
               ENDDO
            ENDDO
            CALL FFTBRC_MPI(C,GRID,ISN)
         ENDIF
!-------------------------------------------------------------------------------
!  real to complex parallel FFT with complex storage layout in real space
!-------------------------------------------------------------------------------
      ELSE
         IF (.NOT.(NX==GRID%NGX_rd .AND. NY==GRID%NGY_rd .AND. NZ/2+1==GRID%NGZ_rd) ) THEN
            WRITE(0,*) 'internal error 5 in FFT3D_MPI: something not properly set',GRID%LREAL, GRID%REAL2CPLX
            WRITE(0,*) NX, NY, NZ
            WRITE(0,*) GRID%NGX_rd, GRID%NGY_rd, GRID%NGZ_rd
            STOP
         ENDIF

         !     q->r FFT
         IF (ISN==1) THEN
            CALL FFTBRC_MPI(C,GRID,ISN)
            !       concat  z-lines (go from stride NZ+2 to NZ)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=GRID%RL%NCOL-1,0,-1
               NDEST=IL* NZ*2
               NSRC =IL*(NZ+2)
!!DIR$ IVDEP
!!OCL NOVREC
               DO  NZZ=NZ,1,-1
                  C(NDEST+NZZ*2-1)=C(NSRC+NZZ)
                  C(NDEST+NZZ*2)=0
               ENDDO
            ENDDO
         ELSE
            !     r->q FFT
            !       z-lines (go from complex stride NZ to real stride NZ+2)
!DIR$ IVDEP
!OCL NOVREC
            DO IL=0,GRID%RL%NCOL-1
               NSRC =IL* NZ*2
               NDEST=IL*(NZ+2)
!DIR$ IVDEP
!OCL NOVREC
               DO NZZ=1,NZ
                  C(NDEST+NZZ)=C(NSRC+NZZ*2-1)
               ENDDO
            ENDDO
            CALL FFTBRC_MPI(C,GRID,ISN)
         ENDIF
      ENDIF

      PROFILING_STOP('fft3d_mpi')

    END SUBROUTINE FFT3D_MPI


!=======================================================================
!   this routine returns the next correct setting for the
!   three dimensional FFT
!=======================================================================

    SUBROUTINE FFTCHK_MPI(NFFT)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION NFFT(3)
      LOGICAL FFTCH1_MPI

      DO IND=1,3
200      CONTINUE
         IF (FFTCH1_MPI(NFFT(IND))) CYCLE
         NFFT(IND)=NFFT(IND)+1
         GOTO 200
100   ENDDO
    END SUBROUTINE FFTCHK_MPI
    
    LOGICAL FUNCTION FFTCH1_MPI(NIN)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (NFACT=4)
      DIMENSION IFACT(NFACT),NCOUNT(NFACT)
      DATA      IFACT /2,3,5,7/
      N=NIN
      DO 100 I=1,NFACT
         NCOUNT(I)=0
120      NEXT=N/IFACT(I)
         IF (NEXT*IFACT(I)==N) THEN
            N=NEXT
            NCOUNT(I)=NCOUNT(I)+1
            GOTO 120
         ENDIF
100   ENDDO
      IF (N==1 .AND. (NCOUNT(1)/=0)) &
           &  THEN
         FFTCH1_MPI=.TRUE.
      ELSE
         FFTCH1_MPI=.FALSE.
      ENDIF
      RETURN
    END FUNCTION FFTCH1_MPI

!=======================================================================
! 
! change data layout from parallel to serial data layout
! and vice versa for complex and real arrays
! operates usually in real space
!
!=======================================================================


    SUBROUTINE FFTPAR_TO_SER(NGX, NGY, NGZ, CORIG)
      USE prec

      INTEGER NGX, NGY, NGZ
      COMPLEX(q) :: CORIG(NGX*NGY*NGZ)
    ! local
      INTEGER IX, IY, IZ
      COMPLEX(q) :: C(NGX*NGY*NGZ)
      
      
      DO IX=0,NGX-1
         DO IY=0,NGY-1
!DIR$ IVDEP
!OCL NOVREC
            DO IZ=0,NGZ-1
               ! C(IX,IY,IZ)=CORIG(IZ,IX,IY)
               C(1+IX+NGX*(IY+NGY*IZ))=CORIG(1+IZ+NGZ*(IX+NGX*IY))
            ENDDO
         ENDDO
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO IX=1,NGX*NGY*NGZ
         CORIG(IX)=C(IX)
      ENDDO

    END SUBROUTINE FFTPAR_TO_SER

    SUBROUTINE FFTPAR_TO_SER_REAL(NGX, NGY, NGZ, CORIG)
      USE prec

      INTEGER NGX, NGY, NGZ
      REAL(q) :: CORIG(NGX*NGY*NGZ)
    ! local
      INTEGER IX, IY, IZ
      REAL(q) :: C(NGX*NGY*NGZ)

      
      DO IX=0,NGX-1
         DO IY=0,NGY-1
!DIR$ IVDEP
!OCL NOVREC
            DO IZ=0,NGZ-1
               ! C(IX,IY,IZ)=CORIG(IZ,IX,IY)
               C(1+IX+NGX*(IY+NGY*IZ))=CORIG(1+IZ+NGZ*(IX+NGX*IY))
            ENDDO
         ENDDO
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO IX=1,NGX*NGY*NGZ
         CORIG(IX)=C(IX)
      ENDDO

    END SUBROUTINE FFTPAR_TO_SER_REAL


    SUBROUTINE FFTSER_TO_PAR(NGX, NGY, NGZ, CORIG)
      USE prec

      INTEGER NGX, NGY, NGZ
      COMPLEX(q) :: CORIG(NGX*NGY*NGZ)
    ! local
      INTEGER IX, IY, IZ
      COMPLEX(q) :: C(NGX*NGY*NGZ)

      
      DO IX=0,NGX-1
         DO IY=0,NGY-1
!DIR$ IVDEP
!OCL NOVREC
            DO IZ=0,NGZ-1
               ! C(IZ,IX,IY)=CORIG(IX,IY,IZ)
               C(1+IZ+NGZ*(IX+NGX*IY))=CORIG(1+IX+NGX*(IY+NGY*IZ))
            ENDDO
         ENDDO
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO IX=1,NGX*NGY*NGZ
         CORIG(IX)=C(IX)
      ENDDO

    END SUBROUTINE FFTSER_TO_PAR

    SUBROUTINE FFTSER_TO_PAR_REAL(NGX, NGY, NGZ, CORIG)
      USE prec

      INTEGER NGX, NGY, NGZ
      REAL(q) :: CORIG(NGX*NGY*NGZ)
    ! local
      INTEGER IX, IY, IZ
      REAL(q) :: C(NGX*NGY*NGZ)

      
      DO IX=0,NGX-1
         DO IY=0,NGY-1
!DIR$ IVDEP
!OCL NOVREC
            DO IZ=0,NGZ-1
               ! C(IZ,IX,IY)=CORIG(IX,IY,IZ)
               C(1+IZ+NGZ*(IX+NGX*IY))=CORIG(1+IX+NGX*(IY+NGY*IZ))
            ENDDO
         ENDDO
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO IX=1,NGX*NGY*NGZ
         CORIG(IX)=C(IX)
      ENDDO

    END SUBROUTINE FFTSER_TO_PAR_REAL
