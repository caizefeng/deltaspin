#include "symbol.inc"
!*****************************************************************
! This module contains routines to calculate the fourier transform
! of non-equally spaced grid points. The idea behind this is to 
! apply Chebyshev's minimax theorem and applying the corresponding
! Remez algorithm. 
!
! Remarks:
!       ) exp grid (ITYPE=1) integrates exp(-xt) and provides
!         solid time grid for RPA and GW calculations
!       ) cos grid (ITYPE=2) integrates (x/(x^2+w^2))^2 and 
!         is the optimum grid for RPA frequency integration
!       ) sin grid (ITYPE=3) integrates (w/(x^2+w^2))^2 and
!         is the optimum grid for the imaginary part of the 
!         non-interacting green's function (useless)
!       ) abs grid (ITYPE=4) integrates 1/(x^2+w^2)^2 and 
!         is good for Fourier interpolation 
!       ) gre grid (ITYPE=5) integrates  w/(x^2+w^2)^2 
!
! TODO:
! introduced FTCOSINV and FTSININV to store inverse transformations
! but the calling routines don't use this yet
!
!*****************************************************************

MODULE minimax
   USE prec
#ifdef qd_emulate
   USE qdmodule
#endif
   USE separable_leastsq, ONLY : TOREAL, DOT_PRODUCT_MPR

   PRIVATE

   PUBLIC :: imag_grid_handle, SET_IMAG_GRID_HANDLE, &
             ALLOCATE_IMAG_GRID_HANDLE,DEALLOCATE_IMAG_GRID_HANDLE, &
             ADD_POINT_IMAG_GRID_HANDLE,INVERT_REAL_MATRIX

   !parameter for Remez algorithm
   INTEGER, SAVE      :: NMINALT = 50
   INTEGER, SAVE      :: NMAXALT = 1500
   INTEGER, SAVE      :: NALTCON = 200
   !tweak minimization interval ?
   LOGICAL, PRIVATE, SAVE   :: LTWEAK=.TRUE.
   !number of iteration steps in sloppy Remez algorithm
   INTEGER, PRIVATE, SAVE   :: MAXITER_FT=100

   !some private parameter 
   !maximum iterations in Newton-Raphson algorithm
   INTEGER, PRIVATE, PARAMETER :: MAXNEWTON = 300
   !maximum iterations in Newton-Root finder
   INTEGER, PRIVATE, PARAMETER :: MAXNEWT = 100
   !maximum iterations in Remez algorithm
   INTEGER, PRIVATE, SAVE :: MAXREMEZ = 100
   !maximum bisections for FINDMAX and FINDMAX_LINEAR
   INTEGER, PRIVATE, PARAMETER :: MAXBISEC=5000
   !parameter for LSQ search algorithm 
   INTEGER, PRIVATE, SAVE  :: ITMAXLSQ = 1500
   INTEGER, PRIVATE, SAVE  :: NDATALSQ = 100
   INTEGER, PARAMETER  :: IGUESS_MAX = 10
#ifdef qd_emulate
#define PIQ      qdpi()
#define ACC      qdreal('1.E-16')
#define ACC2     qdreal('1.E-12')
#define VERYTINY qdreal('1.E-20')
#define DAMPNEWT qdreal('0.35')
#define RANDLSQ  qdreal('50.')
#else
   QDPREAL ,PARAMETER :: PIQ=3.1415926535897932384626433832795028841971693993751_qd
   !accuracy in minimization routines
   QDPREAL , PRIVATE, PARAMETER :: ACC=1.E-16_qd  
   !truncation threshold for SVD and accuracy in FT routine
   QDPREAL , PRIVATE, PARAMETER :: ACC2=1.E-12_qd   
   !tiny number for QDLUDCMP 
   QDPREAL , PRIVATE, PARAMETER :: VERYTINY=1.E-20_qd
   !damping factor for Newton-Raphson root finder
   QDPREAL , PRIVATE, PARAMETER :: DAMPNEWT=0.35_qd
   !parameter for LSQ search algorithm 
   QDPREAL , PARAMETER :: RANDLSQ =  50._qd
#endif
   ! define imaginary grid object
   TYPE imag_grid_handle
      !number of grid points
      INTEGER :: NOMEGA
      !minimization interval 
      REAL(q) :: A, B
      !defines the grid type
      INTEGER :: GRIDTYPE
      !using MM grids 
      LOGICAL :: LMMGRIDS
  
      ! for ACFDT calls
      !default imaginary frequencies (typically cos transform related)
      REAL(q), POINTER :: OMEGA(:)
      !integration frequency weights (typically  transform related)
      REAL(q), POINTER :: OMEGA_WEIGHT(:)
      !frequency grid error 
      REAL(q) :: OMEGA_ERROR
      
      ! ACFDTR(K) calls need time grids and FTCOS matrix 
      REAL(q), POINTER :: TAU(:)
      !tau integration weights, needed only for LTDMP2 calculations
      REAL(q), POINTER :: TAU_WEIGHT(:)
      !time error
      REAL(q) :: TAU_ERROR
      !cosine transformation matrix 
      REAL(q), POINTER :: FTCOS(:,:)
      !inverse cosine transformation matrix
      REAL(q), POINTER :: FTCOSINV(:,:)
      !transformation error 
      REAL(q), POINTER :: COS_ERROR(:)

      ! GW0R(K) calls need also 
      !additional imaginary frequencies (sin transform related)
      REAL(q), POINTER :: OMEGA_SIN(:)
      !additional imaginary frequencies (sin transform related)
      REAL(q), POINTER :: OMEGA_SIN_WEIGHT(:)
      !sin error
      REAL(q) :: OMEGA_SIN_ERROR
      !SIN transformation matrix
      REAL(q), POINTER :: FTSIN(:,:)
      !inverse cosine transformation matrix
      REAL(q), POINTER :: FTSININV(:,:)
      !transformation error 
      REAL(q), POINTER :: SIN_ERROR(:)

      !for integration of odd terms such as w G(iw)Sigma(iw)G(iw) terms
      !additional imaginary frequencies (sin transform related)
      REAL(q), POINTER :: OMEGA_SIN_INT(:)
      !additional imaginary frequencies (sin transform related)
      REAL(q), POINTER :: OMEGA_SIN_INT_WEIGHT(:)
      !SIN integration transformation matrix
      REAL(q), POINTER :: FTSININT(:,:)

   END TYPE imag_grid_handle
   

CONTAINS

!---------------------------------------------------------------
! function which should be interpolated
! with the minimax contition
!---------------------------------------------------------------

   FUNCTION INTERPOL(X)
      USE prec
      IMPLICIT NONE
      QDPREAL  X
      QDPREAL  INTERPOL
       
      IF (ABS(X)<QDPP(1.E-9)) THEN
         WRITE(*,*) ' INTERPOL: X too close to 0, returning 0',X
         INTERPOL=0
      ELSE
         INTERPOL=1/X
         ENDIF
   END FUNCTION INTERPOL

!----------------------LOGPARTITION-----------------------------
! logarithmic partition of inter a,b 
!---------------------------------------------------------------

   FUNCTION LOGPARTITION(I,N,A,B)
      USE prec 
      IMPLICIT NONE
      QDPREAL  LOGPARTITION
      QDPREAL  A,B
      INTEGER I,N
      
      IF ( N<=1 ) THEN
         WRITE(*,*)'ERROR in LOGPARTITION, too less partitions chosen',N
         STOP
      ENDIF

      IF (I<1) THEN
         WRITE(*,*)'ERROR in LOGPARTITION, I range must be between 1 and N',I,N
         STOP
      ENDIF
      
      LOGPARTITION=(B*LOG(QDPD(2))*LOG(QDPD(2-I+N))+LOG(QDPD(1+N))*((A-B)*LOG(QDPD(2))-A*LOG(QDPD(2-I+N))))/&
         ((LOG(QDPD(2))-LOG(QDPD(1+N)))*LOG(QDPD(2-I+N)))
 
     RETURN
  ENDFUNCTION LOGPARTITION 

!****************************************************************************
! Main routine, caluclates imaginary time and frequency grid
! and corresponding forward Fouier transformation matrix,i.e.
! transformation from imaginary time to imanginary frequency  
! ocptionally the sin transform is calculated
!****************************************************************************

   SUBROUTINE SET_IMAG_GRID_HANDLE(X1, X2, NTAU, TYP, LFREQ, LTIME, LSIN, GRIDS, LCRPA, IO)
      USE prec 
      USE base
      IMPLICIT NONE
      REAL(q)             :: X1, X2            !energy interval 
      INTEGER             :: NTAU              !number of time points
      INTEGER             :: TYP               !type of quadrature (lsq or mm)
      LOGICAL             :: LFREQ, LTIME      !flags
      LOGICAL             :: LSIN
      TYPE(imag_grid_handle) GRIDS             !grid handle
      LOGICAL             :: LCRPA             !time grids needs to be scaled differently in case of CRPA
      TYPE(in_struct)     :: IO                !in_struct 
      !local
      REAL(q)             :: TAU(NTAU)         !imaginary times
      REAL(q)             :: TAUWEIGHT(NTAU)   !imaginary time weight
      INTEGER             :: NNU               !number of frequency points
      REAL(q)             :: NU_COS(NTAU)       !imaginary frequencies (cos related)
      REAL(q)             :: NU_COSWEIGHT(NTAU) !imaginary frequency weight (cos related)
      REAL(q)             :: FTCOS(NTAU,NTAU)   !Fourier transformation matrix (cos related)

      REAL(q)             :: NU_SIN(NTAU)       !imaginary frequencies  (sin related)
      REAL(q)             :: NU_SINWEIGHT(NTAU) !imaginary frequency weight (sin related)
      REAL(q)             :: FTSIN(NTAU,NTAU)   !Fourier transformation matrix (sin related)
      QDPREAL ,ALLOCATABLE:: TIMES(:)
      QDPREAL ,ALLOCATABLE:: FREQ(:)
      QDPREAL ,ALLOCATABLE:: FREQ_SIN(:) 
      QDPREAL ,ALLOCATABLE:: FREQ_SIN_INT(:)
      QDPREAL             :: TRANSFORMATION(NTAU,NTAU)
      REAL(q)             :: FTSININT(NTAU,NTAU)
      REAL(q)             :: R,A,B
      INTEGER             :: I,J
      REAL(q)             :: ERR, ERRS(NTAU)
      REAL(q)             :: RMAX
      INTEGER             :: INU
      INTEGER             :: INO
      INTEGER             :: ICOS, ISIN
      LOGICAL             :: LMM = .TRUE. 

      !call minimax reader
      CALL MINIMAX_READER(IO%IU5,IO%IU0)
   
      INU=IO%IU6
      INO=IO%IU0
      NNU=NTAU
      !check if Sin transform should be calculated 
      ALLOCATE(FREQ_SIN(2*NNU))
      ALLOCATE(FREQ_SIN_INT(2*NNU))
      ALLOCATE(FREQ(2*NNU))
      ALLOCATE(TIMES(2*NNU))
   
      DO I = 1, 2*NNU
         FREQ(I) = 0
         FREQ_SIN(I) = 0
         FREQ_SIN_INT(I) = 0
         TIMES(I) = 0
         IF ( I <= NTAU ) ERRS(I) = 0
      ENDDO

      ! first determine R
      R=X2/X1
      !find maximum R, depending on NNU
      CALL FIND_RNMAX(R,NNU)  
      A=MAX(0.01_q,X1)
      B=R*A

      !allocate grid handle
      CALL ALLOCATE_IMAG_GRID_HANDLE(GRIDS, NNU)

      !find out which grids to compute
      IF ( TYP == 141 ) THEN  
         ! ABS grid
         ICOS=4
         ISIN=4
         LMM = .TRUE. 
      ELSE IF ( TYP == 142 ) THEN
         ! SIN grid
         ICOS=3
         ISIN=3
         LMM = .TRUE. 
      ELSE IF (TYP == 143 ) THEN 
         ! COS + SIN grid
         ICOS=2
         ISIN=3
         LMM = .TRUE. 
      ELSE IF (TYP == 144 ) THEN 
         ! x/(x^2+w^2)^3 grid 
         ICOS=5
         ISIN=5
         LMM = .TRUE. 
      ELSE IF (TYP == 145 ) THEN 
         ! COS grid + additional sin integration (can be applied for G0W0)
         ICOS=2
         ISIN=3
         LMM = .TRUE. 
      !least square girds only
      ELSE IF ( TYP == 150 ) THEN  
         ! COS grid
         ICOS=2
         ISIN=2
         LMM=.FALSE.
      ELSE IF ( TYP == 151 ) THEN  
         ! ABS grid
         ICOS=4
         ISIN=4
         LMM=.FALSE.
      ELSE IF ( TYP == 152 ) THEN
         ! SIN grid
         ICOS=3
         ISIN=3
         LMM=.FALSE.
      ELSE IF (TYP == 153 ) THEN 
         ! COS + SIN grid
         ICOS=2
         ISIN=3
         LMM=.FALSE.
      ELSE IF (TYP == 154 ) THEN 
         ! x/(x^2+w^2)^3 grid 
         ICOS=5
         ISIN=5
         LMM=.FALSE.
      ELSE
         ! COS grid
         ICOS=2
         ISIN=2
         LMM=.TRUE.
      ENDIF

      !set grid type and minimization interval in handle
      GRIDS%LMMGRIDS = LMM
      GRIDS%A = A
      GRIDS%B = B

      IF ( NNU > 32) THEN
         NNU=24
         NTAU=24          !default value for high accuracy
         IF ( INU >= 0 ) WRITE(*,'(" Number of grid points forced to ",I4)') NNU
      ENDIF
   
      !determine, which coefficients to use
      IF (INU>=0) WRITE(INU,'(" Interval for error minimization: [",E10.3,",",E10.3,"]")')A,B
      IF (INU>=0) WRITE(*,'(" Interval for error minimization: [",E10.3,",",E10.3,"]")')A,B
      !decide if we start from a higher RMAX(N) and converge to actual R
      CALL RMAX2CURRENT_R(R,NNU,RMAX,IO%IU0)
      IF ( .NOT.LTWEAK) THEN
         R=RMAX
         B=RMAX*A
         IF (INU>=0) WRITE(INU,'(" Used interval: [",E10.3,",",E10.3,"]")')A,B
         IF (INU>=0) WRITE(*,'(" Used interval: [",E10.3,",",E10.3,"]")')A,B
      ENDIF

      !frequency grid
      IF ( LFREQ ) THEN
         !cos-nu points 
         !lsq coefficients may have to be found by converging from RMAX to R
         IF ( LTWEAK ) THEN
            CALL CONVERGE_FROM_RMAX2R_LSQ(ICOS,RMAX,R,NNU,FREQ,IO)
         ELSE 
            CALL NONLINEAR_LSQ_FIT(ICOS,QDPD(R),NNU,FREQ,IO)
         ENDIF 
#ifdef print_erf
         CALL PRINT_ERROR_FUNCTION(NUCOS_ERROR_FUNCTION,NNU,FREQ,QDPP(1.),QDPP(150.),5000,.FALSE.,1)
#endif
         !this calculates optimal nu (cos related) points
         IF ( LMM ) CALL REMEZ_NL(ICOS,NNU,QDPD(R),ACC,FREQ,INU,INO, ERR,IO%NWRITE)  
#ifdef print_erf
         CALL PRINT_ERROR_FUNCTION(NUCOS_ERROR_FUNCTION,NNU,FREQ,QDPP(1.),QDPP(150.),5000,.FALSE.,2)
#endif
         !the scaled coefficients are obtained by multipliciation of A
         GRIDS%OMEGA_ERROR=ERR/A
         DO I=1,NNU
            NU_COS(I)=ABS(FREQ(I))*A
            NU_COSWEIGHT(I)=FREQ(I+NNU)*A
            GRIDS%OMEGA(I)=NU_COS(I)
            GRIDS%OMEGA_WEIGHT(I)=NU_COSWEIGHT(I)
         ENDDO

         !sin-nu points, if desired
         IF (LSIN) THEN
            !sin only necessary if ISIN>=3
            IF ( ISIN==3 ) THEN
               !lsq coefficients may have to be found by converging from RMAX to R
               IF ( LTWEAK ) THEN
                  CALL CONVERGE_FROM_RMAX2R_LSQ(ISIN,RMAX,R,NNU,FREQ_SIN,IO)
               ELSE 
                  CALL NONLINEAR_LSQ_FIT(ISIN,QDPD(R),NNU,FREQ_SIN,IO)
               ENDIF 
               !this calculates optimal nu (cos related) points
               IF ( LMM ) CALL REMEZ_NL(ISIN,NNU,QDPD(R),ACC,FREQ_SIN,INU,INO, ERR,IO%NWRITE)  

               !hack for sin integration
               IF ( TYP == 145 ) THEN
                  DO I = 1, NNU*2
                     FREQ_SIN_INT(I)=FREQ_SIN(I)
                  ENDDO
                  DO I=1,NNU
                     NU_SIN(I)=ABS(FREQ_SIN_INT(I))*A
                     NU_SINWEIGHT(I)=FREQ_SIN_INT(I+NNU)*A
                     GRIDS%OMEGA_SIN_INT(I)=NU_SIN(I)
                     GRIDS%OMEGA_SIN_INT_WEIGHT(I)=NU_SINWEIGHT(I)
                  ENDDO
                  IF ( IO%IU0 >=0 ) WRITE(*,*)'sin integration quadrature determined' 
                  !cos points still used 
                  IF ( IO%IU0 >=0 ) WRITE(*,*)'Using cos frequencies for sin transformation' 
                  DO I = 1, NNU*2
                     FREQ_SIN(I)=FREQ(I)
                  ENDDO
               ENDIF 
            ELSE
               IF ( IO%IU0 >=0 ) WRITE(*,*)'Using cos frequencies for sin transformation' 
               !otherwise use cos points
               DO I = 1, NNU*2
                   FREQ_SIN(I)=FREQ(I)
               ENDDO
            ENDIF
            !the scaled coefficients are obtained by multipliciation of A
            GRIDS%OMEGA_SIN_ERROR=ERR/A
            DO I=1,NNU
               NU_SIN(I)=ABS(FREQ_SIN(I))*A
               NU_SINWEIGHT(I)=FREQ_SIN(I+NNU)*A
               GRIDS%OMEGA_SIN(I)=NU_SIN(I)
               GRIDS%OMEGA_SIN_WEIGHT(I)=NU_SINWEIGHT(I)
            ENDDO
         ENDIF
      ENDIF
  
      !time grid
      IF ( LTIME  ) THEN
         IF ( LTWEAK ) THEN
            CALL CONVERGE_FROM_RMAX2R_LSQ(1,RMAX,R,NTAU,TIMES,IO)
         ELSE
            CALL NONLINEAR_LSQ_FIT(1,QDPD(R),NTAU,TIMES,IO)
         ENDIF
#ifdef print_erf
         CALL PRINT_ERROR_FUNCTION(TAU_ERROR_FUNCTION,NNU,TIMES,QDPP(1.),QDPP(150.),5000,.FALSE.,3)
#endif
         IF ( LMM ) CALL REMEZ_NL(1,NTAU,QDPD(R),ACC,TIMES,INU,INO,ERR,IO%NWRITE)  
#ifdef print_erf
         CALL PRINT_ERROR_FUNCTION(TAU_ERROR_FUNCTION,NNU,TIMES,QDPP(1.),QDPP(150.),5000,.FALSE.,4)
#endif
         !scaled time points obtained by division by 2A 
         !no scaling for CRPA, because \chi(tau) is integrated only for CRPA calculations
         IF ( .NOT. LCRPA ) THEN
            DO I = 1, 2*NTAU
               TIMES( I ) =TIMES( I )/2    
            ENDDO
         ENDIF
         GRIDS%TAU_ERROR=ERR/A
         DO I=1,NTAU
            TAU(I)=ABS(TIMES(I))/A
            TAUWEIGHT(I)=TIMES(I+NTAU)/A
            GRIDS%TAU(I)=TAU(I)
            GRIDS%TAU_WEIGHT(I)=TAUWEIGHT(I)
         ENDDO
      ENDIF

      
      !----------------------------------------------------------
      ! cos fourier matrix necessary for ACFDTRK, or ACFDTR
      IF (LTIME .AND. LFREQ ) THEN
         !finally the cos tranformation matrix (no sloppy Remez algorithm)
         CALL CALC_MINIMAX_FT(.TRUE., FREQ, NNU, TIMES, NTAU,R , MAXITER_FT,TRANSFORMATION,IO,ERRS,A)
         !scale onto proper interval
         DO I=1,NNU
            DO J=1,NTAU
               FTCOS(I,J)=TRANSFORMATION(I,J)*(COS(FREQ(I)*TIMES(J))/A)
            ENDDO
            GRIDS%COS_ERROR(I)=ERRS(I)
         ENDDO
         GRIDS%FTCOS=FTCOS

         !----------------------------------------------------------
         ! GW0R(K) calls need sin transformation matrix as well 
         IF (LSIN) THEN
            !sine grid
            CALL CALC_MINIMAX_FT(.FALSE., FREQ_SIN, NNU, TIMES, NTAU, R, MAXITER_FT,TRANSFORMATION,IO,ERRS,A)
            !scale onto proper interval
            DO I=1,NNU
               DO J=1,NTAU
                  FTSIN(I,J)=TRANSFORMATION(I,J)*(SIN(FREQ_SIN(I)*TIMES(J))/A)
               ENDDO
               GRIDS%SIN_ERROR(I)=ERRS(I)
            ENDDO
            GRIDS%FTSIN=FTSIN

            !hack for sin integration
            IF ( TYP == 145 ) THEN
               !sine grid
               CALL CALC_MINIMAX_FT(.FALSE., FREQ_SIN_INT, NNU, TIMES, NTAU, R, MAXITER_FT,TRANSFORMATION,IO,ERRS,A)
               !scale onto proper interval
               DO I=1,NNU
                  DO J=1,NTAU
                     FTSININT(I,J)=TRANSFORMATION(I,J)*(SIN(FREQ_SIN_INT(I)*TIMES(J))/A)
                  ENDDO
               ENDDO
               GRIDS%FTSININT=FTSININT
            ENDIF
         ENDIF

      ENDIF

      DEALLOCATE(FREQ_SIN)
      DEALLOCATE(FREQ_SIN_INT)

   ENDSUBROUTINE SET_IMAG_GRID_HANDLE

!****************************************************************************
! Remez algorithm for calculating the minimax approximation
! for a given function INTERPOL in the intervall A,B by a 
! sum of weightend nonlinear rationals in NU
! This gives the optimal frequency or time grid
!
! COEFF(2*N) ... on entry : guess for minimax coefficients in [1,R]
! COEFF(2*N) ... on exit  : exact minimax coefficients in [1,R]
! if ITYPE=1 ... exp time grid 
! if ITYPE=3 ... sin related coefficients are calculated
! if ITYPE=4 ... abs(=cos^2+sin^2) related coefficients are calculated
! if ITYPE=5 ... gre(=1/x cos ) related coefficients are calculated
! otherwise  ... cos related coefficients are calculated
! 
!****************************************************************************

   SUBROUTINE REMEZ_NL(ITYPE,N,R,XACC,COEFF,INU,INO,ERROR,NWRITE,LCONV)
      USE prec
      USE ini
      IMPLICIT NONE
      INTEGER          :: ITYPE      !kind of error function?
      INTEGER          :: N          !order of interpolation
      QDPREAL          :: R          !right interval boundary 
      QDPREAL          :: XACC       !desired accuracy
      QDPREAL          :: COEFF(2*N) !coefficients 
      INTEGER          :: INU        !writing unit
      INTEGER          :: INO        !writing unit
      INTEGER          :: NWRITE     !verbosity level
      REAL(q)          :: ERROR      !maximum error
      LOGICAL,OPTIONAL :: LCONV
      !local
      INTEGER               :: I,INFO
      INTEGER               :: ITER
      QDPREAL               :: B,A
      QDPREAL               :: LAMB(2*N+1),LAMBTEMP(2*N+1)
      QDPREAL               :: MAXERROR2,MAXERROR1
      QDPREAL               :: X0(2*N+1)
      LOGICAL               :: LCONV_
  
      LCONV_=.FALSE.
      IF ( PRESENT(LCONV))LCONV_=LCONV
   
      IF ( SIZE(COEFF) < 2*N ) THEN
         WRITE(*,*)'Error in REMEZ_NL, Sizes incompatible:',SIZE(COEFF),2*N
         STOP
      ENDIF
   
      !this is the minimization interval 
      A=1
      B=R
      
      !write preamble
      IF ( INU >=0 ) THEN 
         WRITE(INU,196)
         WRITE(INU,197)
         IF (ITYPE==1) THEN
            WRITE(INU,201)
         ELSEIF (ITYPE==3) THEN
            WRITE(INU,203)
         ELSEIF(ITYPE==4) THEN
            WRITE(INU,204)
         ELSEIF(ITYPE==5) THEN
            WRITE(INU,205)
         ELSE
            WRITE(INU,202)
         ENDIF 

         IF ( NWRITE >2 ) THEN 
            WRITE(*,196)
            WRITE(*,197)
            IF (ITYPE==1) THEN
               WRITE(*,201)
            ELSEIF (ITYPE==3) THEN
               WRITE(*,203)
            ELSEIF(ITYPE==4) THEN
               WRITE(*,204)
            ELSEIF(ITYPE==5) THEN
               WRITE(*,205)
            ELSE
               WRITE(*,202)
            ENDIF 
         ENDIF 
      ENDIF 
     
      ! this is the starting guess, obtained from varpro
      DO I=1, 2*N
         LAMB(I)=COEFF(I)
      ENDDO  
   
      ! The minimum of the L_\infty norm of the error function is:
      IF (ITYPE==1) THEN
         LAMB(2*N+1)=FINDMAX(N,LAMB,TAU_ERROR_FUNCTION,A,B)   
      ELSEIF (ITYPE==3) THEN
         LAMB(2*N+1)=FINDMAX(N,LAMB,NUSIN_ERROR_FUNCTION,A,B)   
      ELSEIF(ITYPE==4) THEN
         LAMB(2*N+1)=FINDMAX(N,LAMB,NUABS_ERROR_FUNCTION,A,B)   
      ELSEIF(ITYPE==5) THEN
         LAMB(2*N+1)=FINDMAX(N,LAMB,NUGRE_ERROR_FUNCTION,A,B)   
      ELSE
         LAMB(2*N+1)=FINDMAX(N,LAMB,NUCOS_ERROR_FUNCTION,A,B)   
      ENDIF
   
      !==========================================================================
      remez: DO ITER=1, MAXREMEZ  !start the remez algorithm
      !==========================================================================
       
         IF(ITYPE==1) THEN 
            MAXERROR1=FINDMAX(N,LAMB,TAU_ERROR_FUNCTION,A,B)
         ELSEIF(ITYPE==3) THEN 
            MAXERROR1=FINDMAX(N,LAMB,NUSIN_ERROR_FUNCTION,A,B)
         ELSEIF(ITYPE==4) THEN
            MAXERROR1=FINDMAX(N,LAMB,NUABS_ERROR_FUNCTION,A,B)
         ELSEIF(ITYPE==5) THEN
            MAXERROR1=FINDMAX(N,LAMB,NUGRE_ERROR_FUNCTION,A,B)
         ELSE
            MAXERROR1=FINDMAX(N,LAMB,NUCOS_ERROR_FUNCTION,A,B)
         ENDIF 

         !----------------------------------------------------------------------
         ! find an alternant of error function ( all local extrema)
         IF ( N < 27 ) THEN
            IF (ITYPE==1) THEN
               CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,NMINALT,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF (ITYPE==3) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,NMINALT,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF(ITYPE==4) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMINALT,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF(ITYPE==5) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,NMINALT,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSE
               CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMINALT,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ENDIF
         ELSE
            IF (ITYPE==1) THEN
               CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,2*NALTCON,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF (ITYPE==3) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,2*NALTCON,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF(ITYPE==4) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,2*NALTCON,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSEIF(ITYPE==5) THEN
               CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,2*NALTCON,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ELSE
               CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,2*NALTCON,INFO)
               IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
            ENDIF
         ENDIF 
         IF (INFO <=-1 ) THEN
           IF (LCONV_)THEN
               IF (ITYPE==1) THEN
                  CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,NALTCON,INFO)
                  IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
               ELSEIF (ITYPE==3) THEN
                  CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,NALTCON,INFO)
                  IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
               ELSEIF(ITYPE==4) THEN
                  CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,NALTCON,INFO)
                  IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
               ELSEIF(ITYPE==5) THEN
                  CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,NALTCON,INFO)
                  IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
               ELSE
                  CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,NALTCON,INFO)
                  IF (INFO<=-1) CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,A,B,LAMB,X0,NMAXALT,INFO)
               ENDIF
               CALL VTUTOR('E','ALTERNANT', &
               &               1._q,1,NMAXALT,1,(0.0_q,0.0_q),1,.TRUE.,1,INO,3)
               CALL VTUTOR('S','ALTERNANT', &
               &               1._q,1,NMAXALT,1,(0.0_q,0.0_q),1,.TRUE.,1,INU,3)
               STOP
            ELSE
               CALL VTUTOR('E','ALTERNANT', &
               &               1._q,1,NMAXALT,1,(0.0_q,0.0_q),1,.TRUE.,1,INO,3)
               CALL VTUTOR('S','ALTERNANT', &
               &               1._q,1,NMAXALT,1,(0.0_q,0.0_q),1,.TRUE.,1,INU,3)
               STOP
            ENDIF
         ENDIF 
         !----------------------------------------------------------------------
         
         !---------------------------------------------------------------------- 
         ! solve non linear system for new coefficients
         LAMBTEMP(1:2*N+1)=LAMB(1:2*N+1)
         IF (ITYPE==1)THEN
            CALL SOLVE_NONLS(TAU_ERROR_FUNCTION_GRAD,N,LAMBTEMP,X0,XACC,INFO)
         ELSEIF (ITYPE==3)THEN
            CALL SOLVE_NONLS(NUSIN_ERROR_FUNCTION_GRAD,N,LAMBTEMP,X0,XACC,INFO)
         ELSEIF(ITYPE==4)THEN
            CALL SOLVE_NONLS(NUABS_ERROR_FUNCTION_GRAD,N,LAMBTEMP,X0,XACC,INFO)
         ELSEIF(ITYPE==5)THEN
            CALL SOLVE_NONLS(NUGRE_ERROR_FUNCTION_GRAD,N,LAMBTEMP,X0,XACC,INFO)
         ELSE
            CALL SOLVE_NONLS(NUCOS_ERROR_FUNCTION_GRAD,N,LAMBTEMP,X0,XACC,INFO)
         ENDIF
         !----------------------------------------------------------------------
      
         !check sucess
         IF (INFO < 0 ) THEN 
            IF (INU>=0) WRITE(*,*)'ERROR: no solution found for ITER=',ITER
            STOP
         ELSEIF ( INFO == 0 ) THEN 
            LAMB(1:2*N+1)=LAMBTEMP(1:2*N+1)
         ENDIF 
           
         !now LAMB contains the new set of coefficients
         !calculate error difference w.r.t. to previous step
         IF (ITYPE==1) THEN
            MAXERROR2=FINDMAX(N,LAMB,TAU_ERROR_FUNCTION,A,B)
         ELSEIF (ITYPE==3) THEN
            MAXERROR2=FINDMAX(N,LAMB,NUSIN_ERROR_FUNCTION,A,B)
         ELSEIF (ITYPE==4) THEN
            MAXERROR2=FINDMAX(N,LAMB,NUABS_ERROR_FUNCTION,A,B)
         ELSEIF (ITYPE==5) THEN
            MAXERROR2=FINDMAX(N,LAMB,NUGRE_ERROR_FUNCTION,A,B)
         ELSE
            MAXERROR2=FINDMAX(N,LAMB,NUCOS_ERROR_FUNCTION,A,B)
         ENDIF
   
         IF (INU>=0) THEN
            WRITE(INU,198)ITER,TOREAL(MAXERROR1),TOREAL(ABS(MAXERROR2-MAXERROR1))
            IF(NWRITE>2) WRITE(*,198)ITER,TOREAL(MAXERROR1),TOREAL(ABS(MAXERROR2-MAXERROR1))
         ENDIF
         !----------------------------------------------------------------------
         ! if accuracy reached, get out! 
         !----------------------------------------------------------------------
         IF ( ABS(MAXERROR2-MAXERROR1) < XACC ) THEN
            EXIT remez
         ELSE
            CYCLE remez
         ENDIF

   
      !=========================================================================
      ENDDO remez
      !=========================================================================
   
      !sort coefficients
      COEFF(1:2*N)=LAMB(1:2*N)
      CALL PIKSRT(N,COEFF)         
      !determine the maximum error
      IF (ITYPE==1) THEN
         ERROR=FINDMAX(N,LAMB,TAU_ERROR_FUNCTION,A,B)
      ELSEIF (ITYPE==3) THEN
         ERROR=FINDMAX(N,LAMB,NUSIN_ERROR_FUNCTION,A,B)
      ELSEIF(ITYPE==4) THEN
         ERROR=FINDMAX(N,LAMB,NUABS_ERROR_FUNCTION,A,B)
      ELSEIF(ITYPE==5) THEN
         ERROR=FINDMAX(N,LAMB,NUGRE_ERROR_FUNCTION,A,B)
      ELSE
         ERROR=FINDMAX(N,LAMB,NUCOS_ERROR_FUNCTION,A,B)
      ENDIF
       
      IF ( ITER > MAXREMEZ ) THEN
      !at this stage the Remez algorithm did not converge after MAXREMEZ steps.
      !write an appropriate Warning, but don't terminate the program
         IF( INU >=0 ) WRITE(INU,199)MAXREMEZ,ERROR
         IF( INU >=0 ) WRITE(*,199)MAXREMEZ,ERROR
      ELSE
      ! success!   
         IF ( INU >=0 ) THEN
            WRITE(INU,200)ITER,ERROR
            IF (ITYPE==1) THEN     !exp grid
               WRITE(*,211)N,ERROR 
            ELSEIF (ITYPE==3) THEN !sin grid
               WRITE(*,213)N,ERROR
            ELSEIF (ITYPE==4) THEN !abs grid
               WRITE(*,214)N,ERROR
            ELSEIF (ITYPE==5) THEN !gre grid
               WRITE(*,215)N,ERROR
            ELSE                   !cos grid
               WRITE(*,212)N,ERROR
            ENDIF

            IF ( NWRITE>2 ) THEN
               WRITE(*,200)ITER,ERROR
               WRITE(INU,*)'Minimax coefficients:'
               DO I = 1, N
                  WRITE(INU,300)TOREAL(COEFF(I)),TOREAL(COEFF(I+N))
               ENDDO
               IF(NWRITE>2)WRITE(*,196)
            ENDIF
         ENDIF
      ENDIF
      IF(INU>=0)WRITE(INU,196)
      RETURN
   
196   FORMAT('  ',76('-'))
197   FORMAT(' |',15(' '),'Calculation minimax points for imaginary axes',16(' '),'|')
198   FORMAT(' |','  Remez',I4,':    maximum error= ',E15.7,',    dE= ',E15.7'      |')
199   FORMAT('  WARNING: Remez not finished after ',I2,' steps, maximum error:',F14.8,/,&
             '  Calculated quadrature is not the minimax approximation! ')
200   FORMAT(' |  finished after ',I2,' steps, maximum error:',E20.10,15(' '),'|')

201   FORMAT(' |',26(' '),'(for exp transformation)',26(' '),'|')
202   FORMAT(' |',26(' '),'(for cos transformation)',26(' '),'|')
203   FORMAT(' |',26(' '),'(for sin transformation)',26(' '),'|')
204   FORMAT(' |',26(' '),'(for abs transformation)',26(' '),'|')
205   FORMAT(' |',26(' '),'(for gre transformation)',26(' '),'|')

211   FORMAT(' MM quadrature (exp) found (',I2,'th order), error=',E20.10)
212   FORMAT(' MM quadrature (cos) found (',I2,'th order), error=',E20.10)
213   FORMAT(' MM quadrature (sin) found (',I2,'th order), error=',E20.10)
214   FORMAT(' MM quadrature (abs) found (',I2,'th order), error=',E20.10)
215   FORMAT(' MM quadrature (gre) found (',I2,'th order), error=',E20.10)
300   FORMAT( 2F40.32)

   ENDSUBROUTINE REMEZ_NL

!****************************************************************************
! Performs a linear least square fit of the form 
!
! \sum g(i,k) Cos(\nu_k \tau_i) Exp(-x \tau_i) = x/(x^2+\nu_k^2)
! if  LCOS=.TRUE. or 
!
! \sum g(i,k) Sin(\nu_k \tau_i) Exp(-x \tau_i) = \nu_k/(x^2+\nu_k^2)
! if LCOS=.FALSE.
! 
! The arrays NU and TAU containing the points \a_k and \tau_i 
! must be given in addition to the interval (X1,X2) in which X may vary
! On output GAM(NNU,NTAU) is the optimal forward Fourier matrix for 
! the RPA polarizability, i.e.
! transformation of imaginary time to imaginary frequency domain
!****************************************************************************

   SUBROUTINE CALC_MINIMAX_FT( LCOS, NU, NNU, TAU, NTAU, RD, MAXITER, GAM,IO,ERRORS,FACT)
      USE prec
      USE base, ONLY: in_struct
      IMPLICIT NONE
      LOGICAL          :: LCOS               !cos or sin transform?
      QDPREAL          :: NU(NNU),TAU(NTAU)  !frequency and time points       
      INTEGER          :: NNU, NTAU          !number of points, respectively
      REAL(q)          :: RD                 !minimization interval
      INTEGER          :: MAXITER            !maximal number of Remez iterations
      QDPREAL          :: GAM(NNU,NTAU)      !Transformation matrix
      TYPE(in_struct)  :: IO
      REAL(q)          :: ERRORS(NNU)        !resulting errors for each frequency
      REAL(q),OPTIONAL :: FACT               !factor to multiply error 
      !local
      INTEGER,  PARAMETER   :: NDATA=100      !number of data points
      INTEGER               :: O,T,J, I       !loop variables 
      REAL(q)               :: X1,X2          !minimization interval
      QDPREAL               :: X(NDATA)       !data points 
      QDPREAL , ALLOCATABLE :: A(:,:)         !design matrix of fitting problem
      QDPREAL , ALLOCATABLE :: U(:,:)         !left singluar vectors
      QDPREAL , ALLOCATABLE :: S(:)           !singular values
      QDPREAL , ALLOCATABLE :: VT(:,:)        !right singular vectors
      QDPREAL , ALLOCATABLE :: VAR(:)         !variances
      QDPREAL               :: B(NDATA)       !function values
      QDPREAL               :: VTJ(NDATA)     !right singular vectors
      QDPREAL , ALLOCATABLE :: G(:)           !temporary fourier matrix
      QDPREAL               :: W              !weight 
      QDPREAL               :: R              !interval length
      QDPREAL               :: MAD            !mean absolute deviation
      INTEGER               :: SIGNS(NDATA)   !signs of error for each data point 
      QDPREAL               :: XI(NDATA)      !alternant
      INTEGER               :: ITER      !loop variables           
      INTEGER               :: NEXTREMA       !number of consideres extrema
      QDPREAL               :: F(NDATA,NTAU+1)!matrix of minimax coefficients
      QDPREAL , ALLOCATABLE :: GTEMP(:)       !temporary coefficients
      QDPREAL               :: MAXERROR1      !some errors
      QDPREAL               :: MAXERROR2      !some errors
      QDPREAL               :: VARTOT1,VARTOT2!auxilary 
#ifdef debug
      INTEGER,  PARAMETER   :: NPRINT=5000
      REAL(q)               :: XX(NPRINT),ERROR
#endif
      INTEGER               :: INU,IU0,NWRITE !for output
   

      INU=IO%IU6
      IU0=IO%IU0
      NWRITE=IO%NWRITE
      !dump preamble  
      IF (INU>=0) THEN              
         WRITE(INU,100)
         IF (LCOS) THEN
            WRITE(INU,101) 
         ELSE
            WRITE(INU,102) 
         ENDIF 
         IF ( NWRITE>2) THEN
            WRITE(*,100)
            IF (LCOS) THEN
               WRITE(*,101) 
            ELSE
               WRITE(*,102) 
            ENDIF 
         ENDIF 
      ENDIF
      MAXERROR2=0

      !use conventional Chebyshev nodes as Data points
      X1=1
      X2=RD
      !minimization interval length 
      R=RD
      !use exponentially distributed Chebyshev nodes
      !decreases the maximum error for x->R by a factor of ~ 0.75  
      IF ( ABS(X1-1._q) > ACC2 ) THEN
         WRITE(*,*)'ERROR in CALC_MINIMAX_FT, X1/=1',X1
         STOP
      ENDIF 
      DO I=1, NDATA
         X(I)=LOG(R)+LOG(R)*COS(((2*(2*NDATA-I)+1)*PIQ)/(4*NDATA))
         X(I)=EXP(X(I))
      ENDDO  
   
      !allocate memory for design matrix A and fourier vector for frequency
      ALLOCATE( A(NTAU, NDATA) ) 
      ALLOCATE( G(NTAU+1),GTEMP(NTAU+1) ) 
      ALLOCATE( VAR( NTAU+1)) 
      A=0
      G=0
      GTEMP=0
      VAR  =0
   
      !---------------------------------------------------
      frequency: DO O=1,NNU                   !nu loop 
      !---------------------------------------------------
         time: DO T=1,NTAU                    !tau loop
            IF (LCOS) THEN
               data_cos: DO J=1,NDATA         !data points
                  !set up design matrix A and vector B
                  A(T,J)=COS( NU(O)*TAU(T) )*EXP(-X(J)*TAU(T))
   
                  !B contains frequency dependence of polarizability 
                  B(J)=X(J)/(X(J)*X(J)+NU(O)*NU(O))
               ENDDO data_cos
            ELSE 
               data_sin: DO J=1,NDATA         !data points
                  !set up design matrix A and vector B
                  A(T,J)=SIN( NU(O)*TAU(T) )*EXP(-X(J)*TAU(T))
                  !B contains frequency dependence of polarizability 
                  B(J)=NU(O)/(X(J)*X(J)+NU(O)*NU(O))
               ENDDO data_sin
            ENDIF
         ENDDO time

   !     WRITE(*,'(12F18.10)')B(1:NDATA)
    
         !the optimal fourier matrix transforming from 
         !imaginary time points tau to imaginary frequency 
         !nu is the solution of a linear fiting problem 
         !The least square solution is simply the solution 
         !vector a of A^T . A . a = A^T .B         
         !we solve this using SVD of A=U.S.V^T
         ALLOCATE( U(NTAU, NTAU) ) ; U =0
         ALLOCATE( S(NDATA) )      ; S =0
         ALLOCATE(VT(NDATA,NDATA)) ; VT=0
       
         !compute SVD of A
         CALL CALC_SVD( A, NTAU, NDATA, U, S, VT)
         !caluclate least square coefficients
         !initialize solution vector
         G=0
         VAR=0
         VARTOT1=0
         DO T=1,NTAU
            DO J=1,NTAU  
               !kill large inverse singular values
               IF (ABS(S(J)) < ACC2 ) THEN
                  W=0
               ELSE
                  W=1/S(J)
               ENDIF
               !(taken from NR chapter 14 for 
               !linear least square fiting problems)
               DO I = 1, NDATA 
                  VTJ( I ) = VT( J , I )
               ENDDO
               G(T)=G(T)+W*DOT_PRODUCT_MPR(VTJ,B)*U(T,J)
               !variances
               VAR(T)= VAR(T) + W*W*U(T,J)*U(T,J)
            ENDDO
            VARTOT1 = VARTOT1 + VAR(T)
         ENDDO
   
         !save least square coefficients 
         DO I = 1, NTAU 
            GTEMP( I )=G( I )
         ENDDO
         !find the maximum error for either the cos or sin transform
         IF (LCOS) THEN
            MAXERROR1=FINDMAX_LINEAR(NTAU,GTEMP,TAU_NU_COS_ERROR,X1,X2,NU(O),TAU)
         ELSE
            MAXERROR1=FINDMAX_LINEAR(NTAU,GTEMP,TAU_NU_SIN_ERROR,X1,X2,NU(O),TAU)
         ENDIF
         !clean memory for S, V, U
         DEALLOCATE( VT, S, U )
   
         !now G is least square solution vector of fitting problem
         !next we need to minimize the error function of fitting problem
         !using the Remez algorithm. 
         !This is almost trivial, since the problem is linear!
         !However in general the alternant of the error function has more than
         !NTAU+1 points, this means we acutally have to solve an overdetermined
         !system of equations. This is done best using the SVD of the coupling 
         !matrix 

      
         !save X to XI, this is advantageous if we use the MAD and not 
         !the sloppy REMEZ algorithm described above
         NEXTREMA=NDATA 
         DO I = 1, NDATA 
            !XI(1:NDATA) = X(1:NDATA) 
            XI( I ) =  X( I ) 
         ENDDO
         !==========================================================================
         remez:DO ITER=1,MAXITER
         !==========================================================================
            IF (LCOS) THEN
               MAXERROR1=FINDMAX_LINEAR(NTAU,GTEMP,TAU_NU_COS_ERROR,X1,X2,NU(O),TAU)
               CALL FIND_MAD(TAU_NU_COS_ERROR,NTAU,TAU,NU(O),GTEMP,X,NDATA,MAD,SIGNS)  
            ELSE
               MAXERROR1=FINDMAX_LINEAR(NTAU,GTEMP,TAU_NU_SIN_ERROR,X1,X2,NU(O),TAU)
               CALL FIND_MAD(TAU_NU_SIN_ERROR,NTAU,TAU,NU(O),GTEMP,X,NDATA,MAD,SIGNS)  
            ENDIF
       
            !build Coefficient matrix and r.h.s
            DO J=1,NEXTREMA
               DO T=1,NTAU+1
                  IF (T<=NTAU ) THEN
                     IF (LCOS) THEN
                        F(J,T)=COS(NU(O)*TAU(T))*EXP(-XI(J)*TAU(T))
                     ELSE
                        F(J,T)=SIN(NU(O)*TAU(T))*EXP(-XI(J)*TAU(T))
                     ENDIF
                  ELSE
                     !F(J,T)=(-QDPP(1.))**J
                     F(J,T)=SIGNS(J)*MAD
                  ENDIF
               ENDDO   
               !b is now vector of r.h.s at extrema Xi
               IF (LCOS) THEN
                  B(J)=XI(J)/(XI(J)*XI(J)+NU(O)*NU(O))   
               ELSE
                  B(J)=NU(O)/(XI(J)*XI(J)+NU(O)*NU(O))   
               ENDIF
               !WRITE(*,'(11F10.6," | ",F10.6)')F(J,1:NTAU+1),B(J)
            ENDDO
            !allocate memory for U,S,VT
            ALLOCATE( U(NEXTREMA, NEXTREMA) ) 
            ALLOCATE( S(NTAU+1) )           
            ALLOCATE(VT(NTAU+1,NTAU+1))     
            U= 0
            S= 0
            VT=0
   
            !solve overdetermined system using SVD of coupling matrix F
            CALL CALC_SVD( F(1:NEXTREMA,1:NTAU+1), NEXTREMA, NTAU+1, U, S, VT)
        
            !calculate new coefficients
            G=0
            VAR=0
            VARTOT2=0
            DO T=1,NTAU+1
               DO J=1,NTAU+1  
                  !kill large inverse singular values
                  IF (ABS(S(J)) < ACC2) THEN
                     W=0
                  ELSE
                     W=1/S(J)
                  ENDIF
                  G(T)=G(T)+W*DOT_PRODUCT_MPR(U(1:NEXTREMA,J),B(1:NEXTREMA))*VT(J,T)
                  !variances
                  VAR(T)= VAR(T) + W*W*VT(J,T)*VT(J,T)
               ENDDO
               VARTOT2=VARTOT2 + VAR(T)
            ENDDO
   
            !kill U,S,VT
            DEALLOCATE(U,S,VT)
       
            !check convergence
            IF (LCOS) THEN
               MAXERROR2=FINDMAX_LINEAR(NTAU,G,TAU_NU_COS_ERROR,X1,X2,NU(O),TAU)
            ELSE
               MAXERROR2=FINDMAX_LINEAR(NTAU,G,TAU_NU_SIN_ERROR,X1,X2,NU(O),TAU)
            ENDIF
            !breaking condition
            IF ( ABS(MAXERROR2-MAXERROR1)>ACC ) THEN 
               GTEMP(1:NTAU+1)=G(1:NTAU+1)
            ELSEIF ( ABS(MAXERROR2-MAXERROR1) < ACC .AND. ITER <  5 ) THEN
               GTEMP(1:NTAU+1)=G(1:NTAU+1)
            ELSEIF ( ABS(MAXERROR2-MAXERROR1) < ACC .AND. ITER >= 5 ) THEN
               EXIT remez
            ENDIF       
         
         !==========================================================================
          ENDDO remez
         !==========================================================================
   
         IF (INU>=0) THEN
            IF ( MAXITER >= 0 ) THEN 
               IF ( PRESENT(FACT) ) THEN
                  WRITE(INU,103)O,FACT*TOREAL(NU(O)),TOREAL(MAXERROR1)/FACT,ITER
                  IF(NWRITE>2)WRITE(*,103)O,FACT*TOREAL(NU(O)),TOREAL(MAXERROR1)/FACT,ITER
               ELSE
                  WRITE(INU,103)O,TOREAL(NU(O)),TOREAL(MAXERROR1),ITER
                  IF(NWRITE>2)WRITE(*,103)O,TOREAL(NU(O)),TOREAL(MAXERROR1),ITER 
               ENDIF
            ENDIF
         ENDIF

         !save error of current frequency point O
         IF ( PRESENT(FACT) ) THEN
            ERRORS(O)=MAXERROR2/FACT
         ELSE
            ERRORS(O)=MAXERROR2
         ENDIF
   
         !save fourier matrix
         DO T=1,NTAU
            GAM(O,T)=G(T)
         ENDDO
   
         !print error function
#ifdef debug
         IF (INU>=0 .AND. LCOS ) THEN
             DO T=1,NPRINT
                XX(T)=X1+((T-1)*(X2-X1))/NPRINT
                ERROR=0._q
                DO J=1,NTAU
                   ERROR = ERROR + G(J)*COS( NU(O)*TAU(J) )*EXP( - XX(T)*TAU(J)) 
                ENDDO       
                ERROR = ERROR - XX(T)/( XX(T)*XX(T)+NU(O)*NU(O) )
                WRITE(200+O,'(2F20.10)')XX(T),ERROR
             ENDDO 
         ELSEIF( INU>=0 .AND. (.NOT.LCOS)) THEN
             DO T=1,NPRINT
                XX(T)=X1+((T-1)*(X2-X1))/NPRINT
                ERROR=0._q
                DO J=1,NTAU
                   ERROR = ERROR + G(J)*SIN( NU(O)*TAU(J) )*EXP( - XX(T)*TAU(J)) 
                ENDDO       
                ERROR = ERROR - NU(O)/( XX(T)*XX(T)+NU(O)*NU(O) )
                WRITE(300+O,'(2F20.10)')XX(T),ERROR
             ENDDO 
         ENDIF
#endif 

      !---------------------------------------------------
      ENDDO frequency
      !---------------------------------------------------
 
      IF (INU>=0) THEN
         WRITE(INU,110)
         IF(NWRITE>2)THEN
             WRITE(*,110)
             WRITE(*,111)
             DO O=1,NNU
                DO T=1,NTAU
                   WRITE(*,112)O,T,TOREAL(GAM(O,T))
                ENDDO
             ENDDO
         ENDIF
      ENDIF
      DEALLOCATE(A,G,GTEMP,VAR)
      RETURN
    
100   FORMAT('  ',76('-'))
101   FORMAT(' |',18(' '),'Calculation of cos transformation matrix',18(' '),'|')
102   FORMAT(' |',18(' '),'Calculation of sin transformation matrix',18(' '),'|')
103   FORMAT(' |    nu_',I2,'=',E15.7,' ERR= ',E15.7,' finished after ',I3,' steps     |')
110   FORMAT('  ',76('-'))
111   FORMAT(' transformation matrix t_ij (unscaled, i/j...freq/time index)' )
112   FORMAT(2I3,F45.32)
   
   ENDSUBROUTINE CALC_MINIMAX_FT

!****************************************************************************
! Gauss-Chebyshev quadrature for the interval [X1,X2] 
! quadruple version 
!****************************************************************************

   SUBROUTINE GAUSS_CHEBYSHEV1Q(X1,X2,X,W,N)
      USE prec
      IMPLICIT NONE
      INTEGER N
      QDPREAL  X1,X2
      QDPREAL  X(N),W(N)
      !local variables
      QDPREAL  XX(N),WW(N)
      INTEGER I,M
       
      M=N 
      DO I=1,M
         !for [-1,1] interval
         XX(I)=COS(((2*I-1)*PIQ)/(2*M))
         WW(I)=PIQ/M
         !scaled onto [X1,X2]
         X(I)=(X1+X2)/2 + (X2-X1)/2*XX(I)
         W(I)=WW(I)*X1
      ENDDO
   
      RETURN
   ENDSUBROUTINE GAUSS_CHEBYSHEV1Q

!****************************************************************************
! sorts coefficients C small size N (<100) in ascending order (pick sort)
!****************************************************************************

   SUBROUTINE PIKSRT(N,C)
      USE prec
      INTEGER N,J, I 
      QDPREAL  ARR(N) 
      QDPREAL  A,C(2*N)
    
      IF (N>100) THEN 
        WRITE(*,*)' Warning ing PIKSRT: This will take long since N=',N
      ENDIF
     
      !abscissas
      ARR(1:N)=C(1:N)
      DO J=2, N
         A=ARR(J)
         DO I=J-1,1,-1
            IF (ARR(I)<=A) GOTO 10
            ARR(I+1)=ARR(I)
         ENDDO
         I=0
10       ARR(I+1)=A
      ENDDO
      !store back to original array
      C(1:N)=ARR(1:N)
   
      !weights
      ARR(1:N)=C(N+1:2*N)
      DO J=2, N
         A=ARR(J)
         DO I=J-1,1,-1
            IF (ARR(I)<=A) GOTO 11
            ARR(I+1)=ARR(I)
         ENDDO
         I=0
11       ARR(I+1)=A
      ENDDO
      !store back to original array 
      C(N+1:2*N)=ARR(1:N)
   
      RETURN
   ENDSUBROUTINE PIKSRT

!****************************************************************************
! This routine calculates the Singular value decomposition A=U.S.V^T  
! of a given matrix  A(M,N) 
!****************************************************************************

   SUBROUTINE CALC_SVD( AA, M, N, UU, SS, VVT )
      USE prec
      IMPLICIT NONE
      QDPREAL   :: AA(M,N)  !matrix to be decomposed
      INTEGER   :: M,N      !number of rows and columns
      QDPREAL   :: UU(M,M)  !left singular eigenvectors
      QDPREAL   :: SS(N)    !signular values
      QDPREAL   :: VVT(N,N) !right singular eigenvectors
      !local
      INTEGER, PARAMETER   :: LWMAX=20000
      INTEGER              :: LDA, LDU, LDVT
      INTEGER              :: INFO, LWORK 
      INTEGER              :: I,J
      INTEGER              :: IWORK( 8*N )
      REAL(q), ALLOCATABLE ::  WORK(:)  
      !work with double precision
      REAL(q) A(M,N), U(M,M),S(N),VT(N,N)  

      !first define leading dimensions
      LDA = M
      LDU = M
      LDVT = N

      ALLOCATE( WORK( LWMAX ) ); WORK=0

      !save quadruple precision into double precision variables
      DO I=1,M
         DO J=1,N
            A(I,J)=AA(I,J)
         ENDDO
      ENDDO
      
      DO J=1,M
         DO I=1,M
            U(J,I)=UU(J,I)
         ENDDO
      ENDDO
       
      DO J=1,N
         DO I=1,N
            VT(J,I)=VVT(J,I)
         ENDDO
         S(J)=SS(J)
      ENDDO

      !Query the optimal workspace
      LWORK = -1
      CALL DGESDD( 'Singular vectors', M, N, A, LDA, S, U, LDU, VT,&
                  LDVT, WORK, LWORK, IWORK, INFO )
      LWORK = MIN( LWMAX, INT( WORK( 1 ) ) )

      !Compute SVD
      CALL DGESDD( 'Singular vectors', M, N, A, LDA, S, U, LDU, VT,&
                  LDVT, WORK, LWORK, IWORK, INFO )


      !check for convergence
      IF( INFO.GT.0 ) THEN
         WRITE(*,*)'CAL_SVD Error: DGESDD failed to converge.',INFO
         STOP
      ENDIF

      DEALLOCATE(WORK ) 

      !save double precision into quadruple precision variables
      DO J=1,N
         DO I=1,M
            AA(I,J)=A(I,J)
         ENDDO
         SS(J)=S(J)
      ENDDO
      
      DO J=1,M
         DO I=1,M
            UU(J,I)=U(J,I)
         ENDDO
         !IF(N==11)   WRITE(*,'(19F10.6)')U(J,1:M)
      ENDDO
       
      DO J=1,N
         DO I=1,N
            VVT(J,I)=VT(J,I)
         ENDDO
      ENDDO

      RETURN
   ENDSUBROUTINE CALC_SVD

!***********************SOLVE_NONLS_FREQ*************************************
!    This routine solve the non linear system 
!
!         y(x_i,L_i)+(-1)^i-1 L_{2N+1} = INTERPOL(x_i)
!
! on entry:  X(1:2*N+1) .... extrema of nonlinear error function
!            L(1:2*N)   .... coefficients of fit
!            L(2*N+1)   .... Maximum of errorfunction in [A,B]
!          
! on exit:   L(1:2*N+1) .... new coefficients          
!****************************************************************************

   SUBROUTINE SOLVE_NONLS(NLFITERGRAD,N,L,X,ACCURACY,INF)
      USE prec
      IMPLICIT NONE
      EXTERNAL               :: NLFITERGRAD !error function and its gradient
      INTEGER                :: N           !order of error function
      QDPREAL ,INTENT(INOUT) :: L(2*N+1)    !coefficients of error function
      QDPREAL                :: X(2*N+1)    !Alternant of error function
      QDPREAL                :: ACCURACY    !accuracy 
      INTEGER                :: INF         !information about convergence
      !local
      INTEGER              :: NL            !number of variables
      INTEGER              :: ITER          !iteration loop variable 
      INTEGER              :: I,K 
      INTEGER              :: INFO          !for QDLUDCMP
      INTEGER,ALLOCATABLE  :: IPIV(:)       !for QDLUDCMP
      QDPREAL              :: F             !error value at X
      QDPREAL              :: DF(1:2*N)     !derivative of error at X
      QDPREAL              :: VARIANCE      !variance of current solution to previous one
      QDPREAL              :: D             
      QDPREAL ,ALLOCATABLE :: J(:,:)        !Jacobian
      QDPREAL ,ALLOCATABLE :: LAMB(:)       !current solution
      QDPREAL ,ALLOCATABLE :: Y(:)          !r.h.s. of system of equations
      QDPREAL ,ALLOCATABLE :: DY(:)         !new direction
    
      !allocate some stuff with size of total number of cofficients
      NL=2*N+1   
      ALLOCATE ( IPIV( NL ), J( NL, NL ), LAMB( NL ), Y( NL ), DY( NL ) )
    
      DO I = 1, NL
         LAMB(I)=L(I)
      ENDDO
   
      !---------------------------------------------------------------------------
      conv: DO ITER=1,MAXNEWTON
      !---------------------------------------------------------------------------
         ! we solve the nonlinear system iteratively using the Newton-Raphson 
         ! method. For this we need the Jacobian w.r.t. the parameter L
         DO I=1,NL
            !error function and its gradient
            CALL NLFITERGRAD(N,LAMB,X(I),F,DF)
            !IF(ITER==1)WRITE(*,'(I4,F11.6,"|",F20.10,"|",10F20.10)')I,X(I),F,DF(1:MIN(NL-1,10))
            !r.h.s.
            Y(I)=-(F-LAMB(NL)*(-1)**(I))   
            !l.h.s.
            DO K=1,NL-1
            J(I,K)=DF(K)
            ENDDO
            J(I,NL)=-(-1)**(I)
         ENDDO
         ! LU decomposition
         CALL QDLUDCMP(J,NL,NL,IPIV,D,INFO)
         IF (INFO /= 0 ) THEN
            WRITE(*,*)'ERROR in SOLVE_NONLS_FREQ: QDLUDCMP failed with:',INFO
            STOP
         ENDIF 
      
         !numerically more stable to use DGETRS than 
         CALL QDLUBKSB( J, NL, NL, IPIV, Y)
      
         IF (ITER>0) THEN    !damp here
            DO I=1, NL
               !weights must be positive!
               IF ( (LAMB(I)+DAMPNEWT*Y(I)<0) .AND. (I>N .AND. I/=NL) ) THEN 
                  WRITE(*,*)ITER,'Error in SOLVE_NONLS, weight is negative:',I
                  WRITE(*,120)NMAXALT
                  STOP
               ELSE
                  LAMB(I)=LAMB(I)+DAMPNEWT*Y(I)
               ENDIF
            ENDDO  
         ELSE                
            DO I=1, NL
               IF ( (LAMB(I)+Y(I)<0) .AND. (I>N .AND. I/=NL) ) THEN  !weights must be positive!
                  WRITE(*,*)ITER,'Error in SOLVE_NONLS_FREQ, weight is negative:',I
                  WRITE(*,120)NMAXALT
                  STOP
               ELSE
                  LAMB(I)=LAMB(I)+Y(I)
               ENDIF
            ENDDO  
         ENDIF
         
         VARIANCE=SQRT(DOT_PRODUCT_MPR(Y,Y))
          
         IF (VARIANCE<ACCURACY) THEN
            EXIT conv       !solution found
         ENDIF    
      
      !---------------------------------------------------------------------------
      ENDDO  conv 
      !---------------------------------------------------------------------------

      IF ( ITER > MAXNEWTON ) THEN
         WRITE(*,100)MAXNEWTON
         STOP
      ELSE 
         !solution found
         !store back to original array 
         L(1:NL)=LAMB(1:NL)
         INF=0                   
      ENDIF 

      DEALLOCATE( IPIV, J, LAMB, Y, DY ) 
      
120   FORMAT('             Hint: try to increase sample points for alternant NMAXALT.',/,&
             '                   Used parameter for this run:',I10)
100   FORMAT('Error in SOLVE_NONLS: Newton-Raphson not converged after',I4,&
             ' iterations')   
   
   ENDSUBROUTINE SOLVE_NONLS

!****************************************************************************
! calculates error function:
!  
!  \sum_i \gamma_i COS(\nu \tau_i)Exp(-x\tau_i) -x/(x^2-\nu^2)
!
! this is need for the optimal fourier transformation from \tau into
! \nu domain  
! 
!****************************************************************************

   SUBROUTINE TAU_NU_COS_ERROR( N, L, X, Z, ZP,NU,TAU )
      USE prec
      IMPLICIT NONE
      INTEGER   :: N      ! order of error function
      QDPREAL   :: L(N)   ! parameters
      QDPREAL   :: X      ! argument X of error
      QDPREAL   :: Z      ! function value at X
      QDPREAL   :: ZP(2)  ! derivatives of Y w.r.t. X
      QDPREAL   :: NU     ! imaginary frequency \nu
      QDPREAL   :: TAU(N) ! imaginary time points  
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)

      IF ( SIZE(L) < N) THEN
         WRITE(*,*)'Error in TAU_NU_COS_ERROR: Size of L incompatible:',SIZE(L),N
         STOP
      ENDIF

      !initialize
      ZTMP=X/(X*X+NU*NU)
 
      M=SIZE( ZP ) 
      ALLOCATE( ZTMPP( M ) ) 
      ! derivative of 1/X
      ZTMPP(1)=(-X*X+NU*NU)/((X*X+NU*NU)**2)
      DO J=2,M
      ZTMPP(J)=2*(X*X*X-3*X*NU*NU)/((X*X+NU*NU)**3)
      ENDDO
     
      !error function
      DO I=1,N 
         ZTMP = ZTMP- L(I)*COS(NU*TAU(I))*EXP(-X*TAU(I) )
      ENDDO
     
      ! Jth derivative 
      DO J=1,M
         DO I=1,N
            ZTMPP(J) = ZTMPP(J) - L(I)*COS(NU*TAU(I))*EXP( -TAU(I)*X )*( -TAU(I) )**J 
         ENDDO
      ENDDO
     
      Z=-ZTMP
      DO I = 1, M
         ZP( I )=-ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)

      RETURN
   ENDSUBROUTINE TAU_NU_COS_ERROR

!****************************************************************************
! calculates error function:
!  
!  \sum_i \gamma_i SIN(\nu \tau_i)Exp(-x\tau_i) -\nu/(x^2-\nu^2)
!
! this is need for the optimal fourier transformation from \tau into
! \nu domain  
! 
!****************************************************************************

   SUBROUTINE TAU_NU_SIN_ERROR( N, L, X, Z, ZP,NU,TAU )
      USE prec
      IMPLICIT NONE
      INTEGER         :: N          ! order of error function
      QDPREAL         :: L(N)       ! parameter of error function
      QDPREAL         :: X          ! argument X of error function
      QDPREAL         :: Z          ! error at X
      QDPREAL         :: ZP(2)      ! derivatives of error w.r.t. X
      QDPREAL         :: NU         ! imaginary frequency \nu
      QDPREAL         :: TAU(N)     ! imaginary time points  
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)

      IF ( SIZE(L) < N) THEN
         WRITE(*,*)'Error in TAU_NU_SIN_ERROR: Size of L incompatible:',SIZE(L),N
         STOP
      ENDIF

      !initialize
      ZTMP=NU/(X*X+NU*NU)
      M=SIZE( ZP ) 
      ALLOCATE( ZTMPP( M ) ) 

      ! derivative of 1/X
      ZTMPP(1)=(-2*X*NU)/((X*X+NU*NU)**2)
      DO J=2,M
         ZTMPP(J)=-2*NU*(-3*X*X+NU*NU)/((X*X+NU*NU)**3)
      ENDDO
   
      !error function
      DO I=1,N 
          ZTMP = ZTMP- L(I)*SIN(NU*TAU(I))*EXP(-X*TAU(I) )
      ENDDO
   
      !Jth derivative
      DO J=1,M
         DO I=1,N
            ZTMPP(J) = ZTMPP(J) - L(I)*SIN(NU*TAU(I))*EXP( -TAU(I)*X )*( -TAU(I) )**J 
         ENDDO
      ENDDO
   
      Z=-ZTMP
      DO I = 1, M
         ZP( I )=-ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE TAU_NU_SIN_ERROR

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X)-\sum_I=1^N \lambda(I+N)Exp(-\lambda(I)X)
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............Y(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of Y(X,\lambda) w.r.t. X
!        
! if INFO  
!****************************************************************************

  SUBROUTINE TAU_ERROR_FUNCTION( N, L, X, Z, ZP )
     USE prec
     IMPLICIT NONE
     INTEGER            :: N        ! order of error
     QDPREAL            :: L(2*N)   ! parameter 
     QDPREAL            :: X       ! argument X of error
     QDPREAL            :: Z        ! eror at X
     QDPREAL            :: ZP(2)    ! derivatives of Y w.r.t. X
     !local
     INTEGER I,J,M
     QDPREAL  ZTMP
     QDPREAL ,ALLOCATABLE :: ZTMPP(:)
  
     IF ( SIZE(L) < 2*N) THEN
        WRITE(*,*)'Error in TAU_ERROR_FUNCTION: Size of L incompatible:',SIZE(L),2*N
        STOP
     ENDIF
     
     !initialize
     ZTMP=INTERPOL(X)
     M=SIZE( ZP ) 
     ALLOCATE( ZTMPP( M ) ) 
     ! derivative of 1/X
     ZTMPP(1)=-INTERPOL(X)/X

     DO J=2,M
         ZTMPP(J)=(-J)*ZTMPP(J-1)/X
     ENDDO
  
     DO I=1,N 
         ZTMP = ZTMP - L(I+N)*EXP(-L(I)*X )
     ENDDO
  
     ! Jth derivative 
     DO J=1,M
        DO I=1,N
           ! save threshold 
           ZTMPP(J) = ZTMPP(J) - L(I+N)*EXP( -L(I)*X )*( -L(I) )**J
        ENDDO
     ENDDO
     
     Z=ZTMP
     DO I = 1, M 
        ZP( I )=ZTMPP( I )
     ENDDO
     DEALLOCATE(ZTMPP)
  
     RETURN
  ENDSUBROUTINE TAU_ERROR_FUNCTION  

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - 1/Pi \sum_I=1^N \lambda(I+N) 4 X^2/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. X
!****************************************************************************

   SUBROUTINE NUCOS_ERROR_FUNCTION(N,L,X,Z,ZP)
      USE PREC
      IMPLICIT NONE
      INTEGER        :: N       ! order of error
      QDPREAL        :: L(2*N)  ! parameter array
      QDPREAL        :: X       ! argument X of error
      QDPREAL        :: Z       ! error at X
      QDPREAL        :: ZP(2)   ! derivatives of error w.r.t. X
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
      
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUCOS_ERROR_FUNCTION: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF

      ZTMP=INTERPOL(X)
      M=SIZE( ZP ) 
      ALLOCATE( ZTMPP( M ) ) 
      ! derivative of 1/X
      ZTMPP(1)=-INTERPOL(X)/X
      DO J=2,M
         ZTMPP(J)=(-J)*ZTMPP(J-1)/X
      ENDDO
     
      DO I=1,N
         !error
         ZTMP = ZTMP - L(I+N)*( 4*(X**2) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         ! 1st derivative 
         ZTMPP(1)= ZTMPP(1)+((16*(X**3)*L(I+N))/((X*X+L(I)*L(I))**3) )/PIQ-&
                            (( 8*X*L(I+N)/((X*X+L(I)*L(I))**2)) )/PIQ
         ! 2nd derivative 
         ZTMPP(2)= ZTMPP(2)-((96*(X**4)*L(I+N) )/( ( X*X + L(I)*L(I) )**4)/PIQ ) + &
                            ((80*(X**2)*L(I+N) )/( ( X*X + L(I)*L(I) )**3)/PIQ ) -&
                            (( 8*L(I+N) )/( ( X*X + L(I)*L(I) )**2 )/PIQ )
      ENDDO
      
      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUCOS_ERROR_FUNCTION

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X)/x - 1/Pi \sum_I=1^N \lambda(I+N) 4 X/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. X
!****************************************************************************

   SUBROUTINE NUGRE_ERROR_FUNCTION(N,L,X,Z,ZP)
      USE PREC
      IMPLICIT NONE
      INTEGER        :: N       ! order of error
      QDPREAL        :: L(2*N)  ! parameter array
      QDPREAL        :: X       ! argument X of error
      QDPREAL        :: Z       ! error at X
      QDPREAL        :: ZP(2)   ! derivatives of error w.r.t. X
      !local
      INTEGER I
      QDPREAL  ZTMP
      QDPREAL  ZTMPP(2)
      
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUGRE_ERROR_FUNCTION: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF

      ZTMP=1/X**2
      ! derivative of 1/X**2
      ZTMPP(1)=-2/X**3
      ZTMPP(2)= 6/X**4

      DO I=1,N
         !error
         ZTMP = ZTMP - L(I+N)*(4*X/PIQ)/((X*X+L(I)*L(I))**2)
         ! 1st derivative 
         ZTMPP(1)= ZTMPP(1)-(L(I+N)*(L(I)**2-3*X**2)/((X**2+L(I)**2)**3)*4)/PIQ
         ! 2nd derivative 
         ZTMPP(2)= ZTMPP(2)-(48*L(I+N)/PIQ)*(X**3-L(I)**2*X)/(( X*X+L(I)*L(I))**4)
      ENDDO
      

      Z=ZTMP
      ZP(1)=ZTMPP(1)
      ZP(2)=ZTMPP(2)
      RETURN
   ENDSUBROUTINE NUGRE_ERROR_FUNCTION

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - 1/Pi \sum_I=1^N \lambda(I+N) 4 X^2/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. LAMBDA
! (used for solving non-linear fiting problem)
!****************************************************************************

   SUBROUTINE NUCOS_ERROR_FUNCTION_GRAD(N,L,X,Z,ZP)
      USE prec
      IMPLICIT NONE
      INTEGER                :: N        ! order of fit
      QDPREAL                :: L(2*N+1) ! parameter array
      QDPREAL                :: X        ! argument X of Y
      QDPREAL                :: Z        ! fit at X
      QDPREAL ,INTENT(INOUT) :: ZP(2*N)  ! derivatives of Y w.r.t. L
      !local
      INTEGER I,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
    
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUCOS_ERROR_FUNCTION_GRAD: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
    
      IF (SIZE(ZP) < 2*N) THEN
         WRITE(*,*)'Error in NUCOS_ERROR_FUNCTION_GRAD, Size of ZP incompatible:',SIZE(ZP),2*N
         STOP
      ENDIF
     
      !initialize 
      ZTMP=INTERPOL(X)
      M=2*N 
      ALLOCATE( ZTMPP( M ) ) 
    
      DO I=1,N
         ZTMP      = ZTMP - L(I+N)*( 4*(X**2) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         ZTMPP(I)  = L(I)*L(I+N)*( 16*(X**2) )/( ( X*X+L(I)*L(I) )**3 )/PIQ
         ZTMPP(I+N)= -( 4*(X**2) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
      ENDDO
    
      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
     
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUCOS_ERROR_FUNCTION_GRAD
   
!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - 1/Pi \sum_I=1^N \lambda(I+N) 4 \lambda(I)/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. X
!****************************************************************************

   SUBROUTINE NUSIN_ERROR_FUNCTION(N,L,X,Z,ZP)
      USE PREC
      IMPLICIT NONE
      INTEGER        :: N       ! order of error
      QDPREAL        :: L(2*N)  ! parameter array
      QDPREAL        :: X       ! argument X of error
      QDPREAL        :: Z       ! error at X
      QDPREAL        :: ZP(2)   ! derivatives of error w.r.t. X
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
     
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUSIN_ERROR_FUNCTION: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
                                    
                 
      ZTMP=INTERPOL(X)
      M=SIZE( ZP ) 
      ALLOCATE( ZTMPP( M ) ) 
      ! derivative of 1/X
      ZTMPP(1)=-INTERPOL(X)/X
      DO J=2,M
         ZTMPP(J)=(-J)*ZTMPP(J-1)/X
      ENDDO
     
      DO I=1,N
         ZTMP = ZTMP - L(I+N)*( 4*L(I)*L(I) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         ! 1st derivative 
         ZTMPP(1) = ZTMPP(1)+((L(I)*L(I)*16*X*L(I+N))/((X*X+L(I)*L(I))**3))/PIQ
         ! 2nd derivative 
         ZTMPP(2) = ZTMPP(2)-((L(I)*L(I)*96*X*X*L(I+N))/((X*X+L(I)*L(I))**4))/PIQ  &
                            +((L(I)*L(I)*16*L(I+N))/((X*X+L(I)*L(I))**3))/PIQ 
      ENDDO
     
      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUSIN_ERROR_FUNCTION

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X)/X - 1/Pi \sum_I=1^N \lambda(I+N) 4 X/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. LAMBDA
! (used for solving non-linear fiting problem)
!****************************************************************************

   SUBROUTINE NUGRE_ERROR_FUNCTION_GRAD(N,L,X,Z,ZP)
      USE prec
      IMPLICIT NONE
      INTEGER                :: N        ! order of fit
      QDPREAL                :: L(2*N+1) ! parameter array
      QDPREAL                :: X        ! argument X of Y
      QDPREAL                :: Z        ! fit at X
      QDPREAL ,INTENT(INOUT) :: ZP(2*N)  ! derivatives of Y w.r.t. L
      !local
      INTEGER I,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
    
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUGRE_ERROR_FUNCTION_GRAD: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
    
      IF (SIZE(ZP) < 2*N) THEN
         WRITE(*,*)'Error in NUGRE_ERROR_FUNCTION_GRAD, Size of ZP incompatible:',SIZE(ZP),2*N
         STOP
      ENDIF
     
      !initialize 
      ZTMP=1/X**2
      M=2*N 
      ALLOCATE( ZTMPP( M ) ) 
    
      DO I=1,N
         ZTMP      = ZTMP - L(I+N)*( 4*X )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         ZTMPP(I)  = L(I)*L(I+N)*( 16*X )/( ( X*X+L(I)*L(I) )**3 )/PIQ
         ZTMPP(I+N)= -( 4*X )/( ( X*X+L(I)*L(I) )**2 )/PIQ
      ENDDO
    
      Z=ZTMP
      DO I = 1, M
         ZP( I )=ZTMPP( I )
      ENDDO    
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUGRE_ERROR_FUNCTION_GRAD

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - 2/Pi \sum_I=1^N \lambda(I+N)  1/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. X
!****************************************************************************

   SUBROUTINE NUABS_ERROR_FUNCTION(N,L,X,Z,ZP)
      USE PREC
      IMPLICIT NONE
      INTEGER        :: N       ! order of error
      QDPREAL        :: L(2*N)  ! parameter array
      QDPREAL        :: X       ! argument X of error
      QDPREAL        :: Z       ! error at X
      QDPREAL        :: ZP(2)   ! derivatives of error w.r.t. X
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
   
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUABS_ERROR_FUNCTION: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
                 
      ZTMP=INTERPOL(X)
      M=SIZE( ZP ) 
      ALLOCATE( ZTMPP( M ) ) 
      ! derivative of 1/X
      ZTMPP(1)=-INTERPOL(X)/X
      DO J=2,M
        ZTMPP(J)=(-J)*ZTMPP(J-1)/X
      ENDDO
   
      DO I=1,N
         ZTMP = ZTMP - L(I+N)*(2/PIQ)/( X*X+L(I)*L(I))
         ! 1st derivative 
         ZTMPP(1) = ZTMPP(1) + (2/PIQ)*( 2*X*L(I+N))/((X*X+L(I)*L(I))**2)
         ! 2nd derivative 
         ZTMPP(2) = ZTMPP(2) - (2/PIQ)*L(I+N)*(6*X**2-2*L(I)**2)/((X*X+L(I)*L(I))**3)
      ENDDO

      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUABS_ERROR_FUNCTION

!****************************************************************************
!  Calculates the nonlinear error function at X 
!     
!   E(X,\lambda)=INTERPOL(X) - 2/Pi \sum_I=1^N \lambda(I+N) 1 \lambda(I)/( X^2+\lambda(I)^2)
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. LAMBDA
! (used for solving non-linear fiting problem)
!****************************************************************************

   SUBROUTINE NUABS_ERROR_FUNCTION_GRAD(N,L,X,Z,ZP)
      USE prec
      IMPLICIT NONE
      INTEGER                :: N       ! order of error function 
      QDPREAL                :: L(2*N+1)! parameter array
      QDPREAL                :: X       ! argument X of Y
      QDPREAL                :: Z       ! fit at X
      QDPREAL ,INTENT(INOUT) :: ZP(2*N) ! derivatives of Y w.r.t. L
      !local
      INTEGER                :: I,M
      QDPREAL                :: ZTMP
      QDPREAL ,ALLOCATABLE   :: ZTMPP(:)
   
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUABS_ERROR_FUNCTION_GRAD: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
   
      IF (SIZE(ZP) < 2*N) THEN
         WRITE(*,*)'Error in NUABS_ERROR_FUNCTION_GRAD, Size of ZP incompatible:',SIZE(ZP),2*N
         STOP
      ENDIF
     
      !initialize 
      ZTMP=INTERPOL(X)
      M=2*N 
      ALLOCATE( ZTMPP( M ) ) 
   
      DO I=1,N
         !function value   
         ZTMP=ZTMP-(2/PIQ)*L(I+N)/(X**2+L(I)**2)
         !nonlinear coeficients derivative
         ZTMPP(I)=(2/PIQ)*L(I+N)*(2*L(I))/(( X*X+L(I)*L(I) )**2)
         !linear coefficients derivative
         ZTMPP(I+N)=-(2/PIQ)/(X**2+L(I)**2)
      ENDDO
   
      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
     
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUABS_ERROR_FUNCTION_GRAD

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - 1/Pi \sum_I=1^N \lambda(I+N) 4 \lambda(I)/( X^2+\lambda(I)^2)^2
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. LAMBDA
! (used for solving non-linear fiting problem)
!****************************************************************************

   SUBROUTINE NUSIN_ERROR_FUNCTION_GRAD(N,L,X,Z,ZP)
      IMPLICIT NONE
      INTEGER                :: N       ! order of error function 
      QDPREAL                :: L(2*N+1)! parameter array
      QDPREAL                :: X       ! argument X of Y
      QDPREAL                :: Z       ! fit at X
      QDPREAL ,INTENT(INOUT) :: ZP(2*N) ! derivatives of Y w.r.t. L
      !local
      INTEGER                :: I,M
      QDPREAL                :: ZTMP
      QDPREAL ,ALLOCATABLE   :: ZTMPP(:)

      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in NUSIN_ERROR_FUNCTION_GRAD: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF

      IF (SIZE(ZP) < 2*N) THEN
         WRITE(*,*)'Error in NUSIN_ERROR_FUNCTION_GRAD, Size of ZP incompatible:',SIZE(ZP),2*N
         STOP
      ENDIF
 
      !initialize 
      ZTMP=INTERPOL(X)
      M=2*N 
      ALLOCATE( ZTMPP( M ) ) 

      DO I=1,N
         ZTMP      = ZTMP - L(I+N)*( 4*(L(I)**2) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         !derivative w.r.t. abscissa
         ZTMPP(I)  =        L(I+N)*(16*(L(I)**3) )/( ( X*X+L(I)*L(I) )**3 )/PIQ &
                          - L(I+N)*( 8*(L(I)   ) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
         !dervative w.r.t. weight
         ZTMPP(I+N)= -( 4*(L(I)**2) )/( ( X*X+L(I)*L(I) )**2 )/PIQ
      ENDDO

      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      RETURN
   ENDSUBROUTINE NUSIN_ERROR_FUNCTION_GRAD

!****************************************************************************
!  Calculates the nonlinear error function at X
!     
!   E(X,\lambda)=INTERPOL(X) - \sum_I=1^N \lambda(I+N)Exp(-\lambda(I)X)
!  
! entry: N...Order
!        L...parameter array \lambda
!  
! exit : Z...............E(X,\lambda)
!        ZP(1:SIZE(ZP))..Derivatives up to SIZE(ZP) of E(X,\lambda) 
!                        w.r.t. LAMB
!****************************************************************************

   SUBROUTINE TAU_ERROR_FUNCTION_GRAD( N, L, X, Z, ZP )
      USE prec
      IMPLICIT NONE
      INTEGER                :: N       ! order of fit
      QDPREAL                :: L(2*N)  ! parameter array
      QDPREAL                :: X       ! argument X of Y
      QDPREAL                :: Z       ! fit at X
      QDPREAL ,INTENT(INOUT) :: ZP(2*N) ! derivatives of Y w.r.t. L
      !local
      INTEGER I,J,M
      QDPREAL  ZTMP
      QDPREAL ,ALLOCATABLE :: ZTMPP(:)
     
      IF ( SIZE(L) < 2*N) THEN
         WRITE(*,*)'Error in TAU_ERROR_FUNCTION_GRAD: Size of L incompatible:',SIZE(L),2*N
         STOP
      ENDIF
      
      IF (SIZE(ZP) < 2*N) THEN
         WRITE(*,*)'Error in TAU_ERROR_FUNCTION_GRAD, Size of ZP incompatible:',SIZE(ZP),2*N
         STOP
      ENDIF
   
      !initialize
      ZTMP=INTERPOL(X)
      M=2*N
      ALLOCATE( ZTMPP( M ) )
      DO J=1,N
         ZTMPP(J)=0
      ENDDO

      DO J=1,N
         !error function 
         ZTMP = ZTMP - L(J+N)*EXP(-L(J)*X )
         ! derivative w.r.t. to exponent 
         ZTMPP(J) = X*L(J+N)*EXP( -L(J)*X )
         ! derivative w.r.t. to weight
         ZTMPP(J+N) = -EXP( -L(J)*X )
      ENDDO
      
      Z=ZTMP
      DO I = 1, M 
         ZP( I )=ZTMPP( I )
      ENDDO
      DEALLOCATE(ZTMPP)
      
      RETURN
   ENDSUBROUTINE TAU_ERROR_FUNCTION_GRAD  
   
!****************************************************************************
! finds the mean absolute deviation 
!****************************************************************************

   SUBROUTINE FIND_MAD(ERRORFUNC,N,TAU,NU,C,X,NDATA,MAD,SIGNS)
      USE prec
      USE m_unirnk
      IMPLICIT NONE
      EXTERNAL :: ERRORFUNC
      INTEGER  :: N                           ! fit order
      QDPREAL  :: TAU(N)                      ! imaginary time points
      QDPREAL  :: NU                          ! imaninary frequency point
      QDPREAL  :: C(N)
      QDPREAL  :: X(NDATA)                    ! data points
      INTEGER  :: SIGNS(NDATA)                ! tabulated signs of error for each data point
      INTEGER  :: NDATA                       ! number of data points
      QDPREAL  :: MAD                         ! mean absolute deviatation
      !local
      INTEGER :: I
      REAL(q) :: FI(NDATA)
      INTEGER :: LX(NDATA),K
      QDPREAL :: FA,DF1(2)
      
      !first step is to tabulate signs of error function for  
      !each data point X(I) and its value
      DO I = 1 , NDATA
         CALL ERRORFUNC(N,C,X(I),FA,DF1,NU,TAU)
         IF ( FA > 0 ) SIGNS(I)=1
         IF ( FA < 0 ) SIGNS(I)=-1
         FI(I) = ABS(FA)   
      ENDDO 
       
      !now sort the array FI
      CALL UNIRNK( FI, LX, K ) 
       
      !caluclate Median of FI,
      !for odd  K
      IF ( MOD ( K , 2 ) /= 0  ) THEN
         !middle value is at K/2+1
         MAD = FI ( LX (  K/2 + 1  ) )  
      !for even K
      ELSE
         MAD = ( FI ( LX ( K/2 ) ) +  FI ( LX (  K/2 + 1  ) ) )/2 
      ENDIF
   
      !finally the MAD 
      DO I =1 , NDATA
        FI ( I ) = ABS ( FI(I) - MAD )
      ENDDO

      !now sort the array FI
      CALL UNIRNK( FI, LX, K ) 
    
      !caluclate Median deviation of FI,
      !for odd  K
      IF ( MOD ( K , 2 ) /= 0  ) THEN
         !middle value is at K/2+1
         MAD = FI ( LX (  K/2 + 1  ) )  
      !for even K
      ELSE
         MAD = ( FI ( LX ( K/2 ) ) +  FI ( LX (  K/2 + 1  ) ) )/2 
      ENDIF

      RETURN                                     !exit if all extrema are found
   ENDSUBROUTINE FIND_MAD

!****************************************************************************
! find alternant using extrema 
!****************************************************************************

   SUBROUTINE FIND_ALTERNANT_EXTREMA(NLFIT,N,X1,X2,C,X,NDAT,INFO)
      USE prec
      USE m_unirnk
      USE ini
      IMPLICIT NONE
      EXTERNAL :: NLFIT
      INTEGER  :: N                      ! fit order
      QDPREAL  :: C(2*N)
      QDPREAL  :: X1,X2                  ! coefficients and interval
      QDPREAL  :: X(2*N+1)               ! extrema
      INTEGER :: NDAT
      INTEGER :: INFO
      !local
      INTEGER  :: I,J,K
      QDPREAL ,ALLOCATABLE :: XI(:),WI(:),X0(:),CTMP(:)
      REAL(q),ALLOCATABLE  :: X00(:)
      INTEGER,ALLOCATABLE  :: LX(:)
      INTEGER  :: INF,ISTART,IFINAL,NDATA
#ifdef qd_emulate
      QDPREAL :: R
      R=QDPP(1.E+6)
#else
      QDPREAL ,PARAMETER :: R=1.E+6_qd !8 significant digits 
#endif
   
      INFO = 0

      !first of all find abscissas being in [x1,x2]
      DO I=1, N
        IF ( X1-C(I) < 0) EXIT
      ENDDO 
      ALLOCATE( CTMP(N) ) 
      IF ( I > N/2 ) THEN
         ! store inverse nodes in reverse order for exp error
         ! to subdivide the error interval  
         DO I = 1, N 
            CTMP( N-I+1 ) = 1+1/C( I ) 
         ENDDO
      ELSEIF( I > 1 ) THEN
         DO I = 1, N 
            CTMP( I  ) = 1+C( I ) 
         ENDDO
      ELSE
         DO I = 1, N 
            CTMP( I  ) = C( I ) 
         ENDDO
      ENDIF
      DO I=1, N
        IF ( X1-CTMP(I) < 0) EXIT
      ENDDO 
      ISTART=I

      DO I=1,N
        IF (X2-CTMP(I) < 0) EXIT
      ENDDO
      
      IF ( I > N ) THEN
         IFINAL=N
      ELSE
         IFINAL=I
      ENDIF
   

      NDATA=(IFINAL-ISTART)*NDAT+2*NDAT
      ALLOCATE(XI(NDATA),WI(NDATA),X0(NDATA))
      ALLOCATE(X00(NDATA))
      ALLOCATE(LX(NDATA))
     
      !setup sample nodes 
      K=0
      DO I=1,NDAT
         K=K+1
         XI(K)=X1+((I-1)*(CTMP(ISTART)-X1))/(NDAT-1)
      ENDDO
      DO I=ISTART,IFINAL-1
         DO J=1,NDAT
            K=K+1
            XI(K)=CTMP(I)+((J-1)*(CTMP(I+1)-CTMP(I)))/(NDAT-1)
         ENDDO
      ENDDO
      DO I=1,NDAT
         K=K+1
         XI(K)=CTMP(IFINAL)+((I-1)*(X2-CTMP(IFINAL)))/(NDAT-1)
      ENDDO
      DEALLOCATE(CTMP)
   
      !----------------------------------------------
      node: DO I = 1, NDATA
      !----------------------------------------------
         CALL RTSAFENL2(NLFIT,N,C,QDPP(1.E-16),X0(I),INF,XI(I))
         IF ( INF == -1 .OR. INF == 1) THEN   
            X0(I)=-100
         ENDIF
   
         !root must not be negative
         IF (X0(I)<0) THEN
            X0(I)=-100
         ENDIF
         
         !root must not be out of interval  
         IF (X0(I)>0 .AND. (X0(I)<X1 .OR. X0(I)>X2) )THEN
            X0(I)=-100
         ENDIF 
       
         !backup for comparision
         X00(I)=X0(I)
      !----------------------------------------------
      ENDDO node
      !----------------------------------------------

      !remove double entries
      CALL UNIRNK(X00,LX,K)

   
      !write warning  if not enough points found
      IF ( K < 2*N-1  ) THEN 
         INFO = -1
         RETURN
      ENDIF
   
      !most probably the first entry of X0 is -100 
      !however if this is not the case we need to use
      !this point as first alternant point after X1
      IF ( .NOT. ( X00( LX(1) ) < 0) ) THEN
         X(2) = X0( LX(1) )
         !save the remaining points
         DO I=2,2*N
            X( I + 1 ) = X0( LX(I) )
         ENDDO
      ELSE
         DO I=2, 2*N
            X( I ) = X0( LX(I) ) 
         ENDDO
      ENDIF
   
      !CALL PRINT_ERROR_FUNCTION(NUCOS_ERROR_FUNCTION,N,C,X1,X2,1000,.TRUE.)
      !add the end points
      K=K+2
      X(1)=X1  
      X(2*N+1)=X2
   
      !A last check is necessary 
      !for high order N~20, still two or more points can coincide
      DO I=1,2*N+1
         DO J=1, 2*N+1
            IF ( I==J ) CYCLE
            IF ( ABS(X(I)-X(J)) < (1/R) ) THEN
               INFO = -I
               IF ( NDATA >1000 )WRITE(*,'(I5,2I3,3F45.32)')NDATA,I,J,TOREAL(X(I)),TOREAL(X(J))
            ENDIF  
         ENDDO
      ENDDO

      IF ( INFO <= -1 ) RETURN
   
      INFO = 0
      RETURN                                     !exit if all extrema are found
   ENDSUBROUTINE FIND_ALTERNANT_EXTREMA

!****************************RTSAFENL2***************************************
! simple root finder for nonlinear error function
!****************************************************************************

   SUBROUTINE RTSAFENL2(NLFIT,N,C,XACC,ROOT,INFO,GUESS)
      USE prec
      EXTERNAL  NLFIT  !error function
      INTEGER   N,INFO, I
      QDPREAL   XACC   !accuracy
      QDPREAL   C(2*N)
      QDPREAL   ROOT
      QDPREAL   GUESS
      QDPREAL   F,FP(2),DX,Z
   
      Z=GUESS
   
      DO I=1,MAXNEWT
         CALL NLFIT(N,C,Z,F,FP)
         DX=FP(1)/FP(2)
         Z=Z-DX
         IF (ABS(DX)<XACC) THEN
            INFO=0
            ROOT = Z
            RETURN
         ENDIF
      ENDDO
      
      INFO=1
      RETURN
   ENDSUBROUTINE RTSAFENL2

!********************************************************
! Calculated the maximum of L1 norm in the interval A,B
! for a linear error function 
!********************************************************

   FUNCTION FINDMAX_LINEAR(N,L,FIT,X1,X2,NU,TAU)
      USE prec
      IMPLICIT NONE
      INTEGER N
      QDPREAL  L(1:N)
      QDPREAL  FINDMAX_LINEAR
      EXTERNAL FIT
      REAL(q)  X1,X2
      QDPREAL  NU
      QDPREAL  TAU(N)
      !localz
      INTEGER I
      QDPREAL  E,ET,FP_(2)
      QDPREAL  X,A,B
     
      IF (SIZE(L)<N) THEN 
         WRITE(*,*)' Error in FINDMAX_LINEAR: Sizes incompatible in L',SIZE(L),N
         STOP
      ENDIF
   
      A=QDPD(X1)
      B=QDPD(X2)
   
      CALL FIT(N,L,A,ET,FP_,NU,TAU)
      ET=ABS(ET)
      DO I=1,MAXBISEC
         X=(A+B)/2+(B-A)/2*COS(((2*(2*MAXBISEC-I)+1)*PIQ)/(4*MAXBISEC))
         CALL FIT(N,L,X,E,FP_,NU,TAU)
         ET=MAX(ABS(ET),ABS(E))
      ENDDO
   
      FINDMAX_LINEAR=ET
      RETURN
   ENDFUNCTION FINDMAX_LINEAR

!********************************************************
! prints the nonlinear error function and its first two
! derivatives w.r.t. argument X
!********************************************************

  SUBROUTINE PRINT_ERROR_FUNCTION(NLFIT,N,L,A,B,IMAX,LLOG,IT)
     USE prec
     IMPLICIT NONE
     EXTERNAL NLFIT
     INTEGER N,IMAX
     QDPREAL  L(2*N)
     QDPREAL  A,B
     LOGICAL,INTENT(IN),OPTIONAL :: LLOG
     INTEGER,INTENT(IN),OPTIONAL :: IT 
     INTEGER I,IFIN
     QDPREAL  X
     QDPREAL  F,DF(2)
     LOGICAL LLN
     CHARACTER(LEN=4) :: APP 
  
     IFIN=IMAX
    
     IF (PRESENT(LLOG)) THEN
        LLN=LLOG
     ELSE
        LLN=.FALSE.
     ENDIF
     
     IF (PRESENT(IT)) THEN
        WRITE(APP,'(I4)')IT
        APP=ADJUSTL(APP)
        OPEN(UNIT=90,FILE='erf'//TRIM(APP)//'.dat',ACTION='WRITE',STATUS='REPLACE')
     ELSE
        OPEN(UNIT=90,FILE='erf.dat',ACTION='WRITE',STATUS='REPLACE')
     ENDIF
  
     !choose abscissas logarithmic
     IF (LLN) THEN
        DO I=1,IFIN
           !we use a logarithmic partition 
           X=LOGPARTITION(I,IFIN,A,B)
           CALL NLFIT(N,L,X,F,DF)
           WRITE(90,'(4F40.30)')X,F,DF(1),DF(2)
        ENDDO
     ELSE
        DO I=1,IFIN
           !equidistant abscissas
           X=A+((I-1)*(B-A))/IFIN
           CALL NLFIT(N,L,X,F,DF)
           WRITE(90,'(4F40.30)')X,F,DF(1),DF(2)
        ENDDO
     ENDIF
    
     CLOSE(90)
     RETURN
  ENDSUBROUTINE PRINT_ERROR_FUNCTION

!********************************************************
! Calculates the maximum of the L1 norm in the interval 
! A,B of the exp error function 
!********************************************************

   FUNCTION FINDMAX(N,L,NLFIT,AA,BB)
      USE prec
      INTEGER N
      QDPREAL  L(1:2*N)
      QDPREAL  FINDMAX
      EXTERNAL NLFIT
      QDPREAL  AA,BB
      !localz
      INTEGER I
      QDPREAL  E,ET,FP_(2)
      QDPREAL  X,XT,A,B
      INTEGER  ISTART
     
      IF (SIZE(L)<2*N) THEN 
         WRITE(*,*)' Error in FINDMAX: Sizes incompatible in L',SIZE(L),2*N
         STOP
      ENDIF

      !smart sampling of minimization interval:
      !first of all find first abscissa in [x1,x2]
      DO I=1, N
        IF ( AA-L(I) < 0 ) EXIT
      ENDDO 
      ISTART=I
      
      IF (ISTART+1 <= N)THEN       
         A=AA
         B=MIN(L(ISTART),BB)
      ELSE
         A=AA
         B=2*L(ISTART)
      ENDIF

      !obtain the error at A
      CALL NLFIT(N,L,A,ET,FP_)
   
      ET=ABS(ET)
      DO I=1,MAXBISEC
         X=A+((I-1)*(B-A))/(MAXBISEC-1) 
         CALL NLFIT(N,L,X,E,FP_)
         E=ABS(E)
         IF ( ET < E ) THEN
            ET=E
            XT=X
         ENDIF
      ENDDO
   
      FINDMAX=ET
      RETURN
   ENDFUNCTION FINDMAX

!*********************************************************
!
! LU decomposition for quadruple precision  (NR)
!
! Given a matrix a(1:n,1:n) , with physical dimension 
! np by np , this routine replaces it by the 
! LU decomposition of a rowwise permutation of itself. 
! a and n are input. a is output, arranged as in 
! equation (2.3.14) of NR above; indx(1:n) is an output vector 
! that records the row permutation effected by the partial 
! pivoting; d is output as ±1 depending on whether the number 
! of row interchanges was even or odd, respectively. 
! This routine is used in combination with qdlubksb to solve 
! linear equations
!
!*********************************************************

   SUBROUTINE QDLUDCMP(A,N,NP,INDX,D, INFO)
      IMPLICIT NONE
      QDPREAL  :: A(NP,NP)
      INTEGER  :: N
      INTEGER  :: NP
      INTEGER  :: INDX(N)
      QDPREAL  :: D
      INTEGER  :: INFO 
      !local 
      INTEGER, PARAMETER  :: NMAX=500
      INTEGER I,IMAX,J,K
      QDPREAL  AAMAX,DUM,SUM,VV(NMAX) 
 
      !vv stores the implicit scaling of each row.
      D=1
        
      !no row interchanges yet.
      !loop over rows to get the implicit scaling information.
      DO I=1,N
         AAMAX=0
         DO J=1,N
            IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
         ENDDO
         IF ( AAMAX .EQ. 0 ) THEN
            !singular matrix in qdludcmp no nonzero largest element.
            INFO=-I 
            RETURN
         ENDIF  
         !save the scaling.
         VV(I)=1/AAMAX
      ENDDO

      !this is the loop over columns of crout’s method.
      DO J=1,N
         !this is equation (2.3.12) except for i = j.
         DO I=1,J-1
            SUM=A(I,J)
            DO K=1,I-1
               SUM=SUM-A(I,K)*A(K,J)
            ENDDO
            A(I,J)=SUM
         ENDDO
         !initialize for the search for largest pivot element.
         AAMAX=0

         !this is i = j of equation (2.3.12) and i = j + 1 . . . n
         !of equation (2.3.13).
         DO I=J,N
            SUM=A(I,J)
            DO K=1,J-1
               SUM=SUM-A(I,K)*A(K,J)
            ENDDO
            A(I,J)=SUM
            !figure of merit for the pivot.
            DUM=VV(I)*ABS(SUM)
            !is it better than the best so far?
            IF (DUM.GE.AAMAX) THEN
               IMAX=I
               AAMAX=DUM
            ENDIF
         ENDDO

         !do we need to interchange rows?
         IF (J.NE.IMAX)THEN
            !YES, DO SO...
            DO K=1,N
               DUM=A(IMAX,K)
               A(IMAX,K)=A(J,K)
               A(J,K)=DUM
            ENDDO
            !...and change the parity of d.
            D=-D
            !also interchange the scale factor.
            VV(IMAX)=VV(J)
         ENDIF

         INDX(J)=IMAX
         !if the pivot element is zero the matrix is singular (at least to the precision of the al-
         !gorithm). for some applications on singular matrices, it is desirable to substitute tiny
         !for zero.
         IF( A(J,J) .EQ. 0 ) A(J,J)=VERYTINY

         !now, finally, divide by the pivot element.
         IF(J.NE.N)THEN
            DUM=1/A(J,J)
            DO I=J+1,N
                A(I,J)=A(I,J)*DUM
            ENDDO
         ENDIF
      ENDDO
 
      INFO=0
      RETURN
      ENDSUBROUTINE QDLUDCMP
 
!*********************************************************
! quadruple precision of DGETRS
!
! Solves the set of n linear equations A · X = B. Here a is 
! input, not as the matrix A but rather as its LU 
! decomposition, determined by the routine ludcmp. indx is 
! input as the permutation vector returned by ludcmp. 
! b(1:n) is input as the right-hand side vector B, and 
! returns with the solution vector X. a , n , np, and 
! indx are not modified by this routine and can be left 
! in place for successive calls with different right-hand 
! sides b. This routine takes into account the possibility 
! that b will begin with many zero elements, so it is 
! efficient for use in matrix inversion.
!*********************************************************

   SUBROUTINE QDLUBKSB(A,N,NP,INDX,B)
      INTEGER  N
      INTEGER  NP
      INTEGER  INDX(N)
      QDPREAL  A(NP,NP)
      QDPREAL  B(N)
      !local 
      INTEGER I,II,J,LL
      QDPREAL  SUM

      II=0
      !when ii is set to a positive value, it will become the in-
      !dex of the first nonvanishing element of b. we now do
      !the forward substitution, equation (2.3.6). the only new
      !wrinkle is to unscramble the permutation as we go.
      DO I=1,N
         LL=INDX(I)
         SUM=B(LL)
         B(LL)=B(I)
         IF (II.NE.0)THEN
            DO J=II,I-1
               SUM=SUM-A(I,J)*B(J)
            ENDDO
         !a nonzero element was encountered, so from now on we will
         !have to do the sums in the loop above.
         ELSEIF (SUM.NE.0) THEN
            II=I
         ENDIF
         B(I)=SUM
      ENDDO

      !now we do the backsubstitution, equation (2.3.7).
      DO I=N,1,-1
         SUM=B(I)
         DO J=I+1,N
            SUM=SUM-A(I,J)*B(J)
         ENDDO 
         !store a component of the solution vector x.
         B(I)=SUM/A(I,I)
      ENDDO
      !all done!
      RETURN
   ENDSUBROUTINE QDLUBKSB
        
!============================================================================
! Least Square Routines

!****************************************************************************
! calling sequence for varpro. 
! performs a nonlinear least square fit of order NORDER starting with the 
! first order 
!****************************************************************************

   SUBROUTINE NONLINEAR_LSQ_FIT(ITYPE,R,NORDER,C,IO)
      USE base
      USE separable_leastsq
      IMPLICIT NONE
      INTEGER         :: ITYPE      !base function selection
      QDPREAL         :: R          !interval 
      INTEGER         :: NORDER     !order of fit
      QDPREAL         :: C(2*NORDER)!coefficients 
      TYPE(in_struct) :: IO         !in_struct 
      !local
      INTEGER, PARAMETER :: NMAX=300
      QDPREAL   :: WEIGHTSCALE     !scaling factor for weights 
      QDPREAL ,ALLOCATABLE :: Y(:), T(:,:),A(:,:), W(:)
      QDPREAL ,ALLOCATABLE :: ALF(:), BETA(:)
      QDPREAL ,ALLOCATABLE :: ALFINIT(:), BETAINIT(:)
      INTEGER   :: I, IERR,  IPRINT, IV, J, L, N, NL, P
      INTEGER   :: IGUESS
      LOGICAL   :: LRETRY
      INTEGER       :: NSTART
      INTEGER       :: INU

      !standard output is OUTCAR
      INU=IO%IU6 


      IF ( IO%NWRITE ==1 ) THEN
         !surpress output of varpro, not recomended!! 
         IPRINT = -1 
      ELSEIF (IO%NWRITE == 2 .OR. IO%NWRITE == 3  ) THEN
         !normal output of varpro
         IPRINT = 0 
      ELSE
         !print complete information
         IPRINT = 1
      ENDIF 
 
      ! determine weighting factor for different error functions
      IF ( ITYPE == 1 ) THEN
         ! time error function 
         WEIGHTSCALE=1
      ELSEIF( ITYPE==2 .OR. ITYPE==3 ) THEN   
         !cos error function 
         WEIGHTSCALE=PIQ/4
      ELSEIF( ITYPE == 4 ) THEN
         !abs error function 
         WEIGHTSCALE=PIQ/2
      ELSEIF( ITYPE == 5 ) THEN
         !gre error function 
         WEIGHTSCALE=PIQ/4
      ELSE
         IF ( INU>=0) WRITE(*,*)'ERROR in NONLINEAR_LSQ_FIT, type of error function is unkown',ITYPE
         STOP
      ENDIF 

      ALLOCATE(Y(NMAX),T(NMAX,1),A(NMAX,2*AMAX+2),W(NMAX))
      ALLOCATE(ALF(AMAX), BETA(BMAX))
      ALLOCATE(ALFINIT(AMAX), BETAINIT(BMAX))
 
      !initialize saved coefficients 
      DO I = 1, AMAX
         ALFINIT(I) =0 
         ALF(I) =0
      ENDDO
      DO I = 1, BMAX
        BETAINIT(I)=0 
        BETA(I)=0
      ENDDO
      DO I = 1, NMAX 
         Y(I) = 0
         W(I) = 0
         DO J= 1, 2*AMAX+2 
            A(I,J) = 0
         ENDDO
      ENDDO

      IV=1  !# of linear independent variables 
      !starting order
      NSTART=1

      !write to OUTCAR only if IO%NWRITE is higher as default
      IF (INU>=0 .AND. IO%NWRITE > 2 ) THEN
         L=NL
         P=NL
         WRITE (INU,110) NDATALSQ, L, NL, P, IV
         IF ( ITYPE == 2 ) THEN
            WRITE(INU,997)NORDER
         ELSEIF(ITYPE==3) THEN
            WRITE(INU,998)NORDER
         ELSEIF(ITYPE==4) THEN
            WRITE(INU,999)NORDER
         ELSEIF(ITYPE==5) THEN
            WRITE(INU,996)NORDER
         ELSE
            WRITE(INU,996)NORDER
         ENDIF
      ENDIF

      !exponentially dirstributed Chebysev nodes yield smaller errors for x->1
      DO I =1, NDATALSQ
         T(I,1)=LOG(R)+LOG(R)*COS(((2*(2*NDATALSQ-I)+1)*PIQ)/(4*NDATALSQ))
         T(I,1)=EXP(T(I,1))
         W(I) = 1 !WEIGHTS ARE SET TO 1
         IF ( T(I,1) > ACC2 ) THEN
            Y(I)=1/(T(I,1))
         ELSE
            IF ( INU>=0) WRITE(INU,*)'ERROR IN NONLINEAR_LSQ_FIT: POINT',I,' TOO SMALL',TOREAL(T(I,1))
            STOP
         ENDIF 
         IF ( ITYPE==5) THEN 
            Y(I)=Y(I)/(T(I,1))
         ENDIF
         !print sample points only for IO%NWRITE >2
         IF ( INU>=0 .AND.  IO%NWRITE > 2 ) WRITE (INU,210) I, (TOREAL(T(I,J)), J=1, IV), TOREAL(Y(I))
      ENDDO
 
      LRETRY=.FALSE.
      DO N =NSTART, NORDER
         NL=N  !# of current non-linear parameters
         L=NL  !# of linear parameters
         P=NL  !# of non-vanishing non-linear partial derivatives
         !use as starting gues coefficients of previous order
         IF ( N>1)THEN
            ALF =0
            BETA=0
            DO I = 1, N-1
               ALF( I ) = ALFINIT( I )
               BETA( I ) = BETAINIT( I )
            ENDDO
         ENDIF
         ! we need initial guesses for the parameters 
         guess: DO IGUESS =1, IGUESS_MAX
            ALF(N) =((IGUESS-1)*RANDLSQ)/(IGUESS_MAX-1)
            BETA(N)=((IGUESS-1)*RANDLSQ)/(IGUESS_MAX-1)

            !print initial guess of non-linear coefficients 
            IF ( N<NORDER .AND. INU>=0 .AND. IO%NWRITE>2 ) &
               WRITE (INU,130) (TOREAL(ALF( I )), I = 1, N )
            IF ( INU>=0 .AND. IO%NWRITE>2) WRITE (INU,140)
            CALL VARPRO(N,N,NDATALSQ,NMAX,L+P+2,IV,T,Y,W,A,IPRINT,ALF,BETA,ITMAXLSQ,&
               ITYPE,IERR,INU,IO%NWRITE)
            !varpro converged
            IF (IERR>0) THEN
               IF (LRETRY) THEN
                  IF ( IO%NWRITE>2 .AND. INU>=0 ) WRITE(INU,101)IGUESS
                  LRETRY=.FALSE.
               ENDIF 
               IF ( IO%NWRITE>2 .AND. INU>=0 ) WRITE(INU,102)NL,IERR
               IF ( IO%NWRITE>2 .AND. INU>=0 ) WRITE(*,102)NL,IERR
               !converged -> save coefficients for next higher order
               ALFINIT =0
               BETAINIT=0
               DO I = 1, N
                  ALFINIT( I ) = ALF( I )
                  BETAINIT( I ) = BETA( I )
               ENDDO
               EXIT guess
            !varpro not converged 
            ELSE
               IF (INU>=0 .AND. IO%NWRITE>2 ) WRITE (INU,'("                IERR =",2I7)') &
                  IERR,IGUESS
               IF ( .NOT. LRETRY ) THEN
                  IF (INU>=0 .AND. IO%NWRITE>2 ) WRITE(INU,103)N
               ENDIF 
               LRETRY=.TRUE.
            ENDIF  
         ENDDO guess

         !truly converged, if less iterations than IGUESS_MAX needed 
         IF ( IGUESS <= IGUESS_MAX) THEN
            !optionally, dump coefficients to OUTCAR
            IF ( (N==NORDER .AND. IO%NWRITE>2) .AND. INU>=0) THEN
               IF (ITYPE==1)THEN
                  WRITE(INU,115)N
               ELSEIF(ITYPE==2)THEN
                  WRITE(INU,116)N
               ELSEIF(ITYPE==3)THEN
                  WRITE(INU,117)N
               ELSEIF(ITYPE==4)THEN
                  WRITE(INU,118)N
               ELSEIF(ITYPE==5)THEN
                  WRITE(INU,114)N
               ENDIF
               WRITE(INU,106)(TOREAL(BETA(I)),TOREAL(ALF(I)),I=1,N)
            ENDIF
            !save coefficients 
            DO I=1,N
               C(I)=ABS(ALF(I))
               C(I+N)=BETA(I)*WEIGHTSCALE
            ENDDO
         ELSE
            IF (INU>=0) WRITE(*,1001) 
            STOP
         ENDIF
      ENDDO

   IF ( INU>=0 .AND. IO%NWRITE>1 ) WRITE(*,1000)NORDER

110 FORMAT ('1   NON-LINEAR LEAST SQUARES PROBLEM'/  &
    ' NUMBER OF OBSERVATIONS =',i5, '   NUMBER OF LINEAR PARAMETERS =',i4/  &
    ' NUMBER OF NONLINEAR PARAMETERS =',i4,  &
    ' NUMBER OF NONVANISHING PARTIAL DERIVATIVES =',i4/  &
    ' NUMBER OF INDEPENDENT VARIABLES =',i4/ '    I    T(I)            Y(I)')
130 FORMAT ('0 INITIAL NONLINEAR PARAMETERS'/(4E20.7))
140 FORMAT ('0',50('*'))
210 FORMAT (i5,7E16.7)

101 FORMAT(/,'      success after ',I4,' attempts.')
102 FORMAT('      Order',I3' done after ',I6,' iterations')
103 FORMAT('          using random numbers for order',I3,' .')
115 FORMAT(' LSQ quadrature (exp) found (',I2,'th order):')
116 FORMAT(' LSQ quadrature (cos) found (',I2,'th order):')
117 FORMAT(' LSQ quadrature (sin) found (',I2,'th order):')
118 FORMAT(' LSQ quadrature (abs) found (',I2,'th order):')
114 FORMAT(' LSQ quadrature (gre) found (',I2,'th order):')
106 FORMAT(2F45.32)

996  FORMAT(/,' ',68('*'),/,'  Looking for exp least square quadrature of order ',I3)
997  FORMAT(/,' ',68('*'),/,'  Looking for cos least square quadrature of order ',I3)
998  FORMAT(/,' ',68('*'),/,'  Looking for sin least square quadrature of order ',I3)
999  FORMAT(/,' ',68('*'),/,'  Looking for abs least square quadrature of order ',I3)
1000 FORMAT(' Found non-linear least square fit of order ',I2,&
            ', set NWRITE>2 for more details')
1001 FORMAT(/,'       maximum number of iterations reached, calculation terminated!',/,&
            '       Increasing the minimization interval might help!')
ENDSUBROUTINE NONLINEAR_LSQ_FIT

!****************************************************************************
!
! Find LSQ grids by converging from RMAX to R
!
!****************************************************************************

   SUBROUTINE CONVERGE_FROM_RMAX2R_LSQ(ITYPE,RMAX,R,N,COF,IO)
      USE base
      USE separable_leastsq
      INTEGER   :: ITYPE           !base function selection
      REAL(q)   :: RMAX            !maximum interval 
      REAL(q)   :: R               !desired interval 
      INTEGER   :: N               !order of fit
      QDPREAL   :: COF(:)          !coefficients
      TYPE(in_struct) :: IO         !in_struct 
      !local
      QDPREAL   :: WEIGHTSCALE     !scaling factor for weights 
      QDPREAL ,ALLOCATABLE :: C(:)
      QDPREAL ,ALLOCATABLE :: ALF(:), BETA(:)
      QDPREAL ,ALLOCATABLE :: Y(:), T(:,:),A(:,:), W(:)
      INTEGER, PARAMETER :: NMAX=300
      INTEGER   :: ITER
      QDPREAL   :: X1,X2
      QDPREAL,ALLOCATABLE :: X0(:)
      INTEGER   :: INFO
      INTEGER   :: I, L, P, NL, IERR,IPRINT, IV
      INTEGER   :: INU             !writing output

      ALLOCATE( X0(2*N+1) )
      DO I = 1, 2*N+1
         X0(I)= 0
      ENDDO

      INU=IO%IU6

      X1=1
      !initialize current minimization interval
      X2=QDPD(RMAX)

      IF ( SIZE(COF) /= 2*N ) THEN
         IF(INU>=0)WRITE(INU,*)'Error in CONVERGE_FROM_RMAX2R_LSQ, Size of COF inconsistent',&
            SIZE(COF),2*N
         STOP
      ENDIF
      ALLOCATE(C(2*N))
      DO I = 1 , 2*N
         C(I)=COF(I) 
      ENDDO

      !first find optimum coefficients
      CALL NONLINEAR_LSQ_FIT(ITYPE,X2,N,C,IO)

      !now approach desired R from above 
      IF ( INU>=0 )WRITE(INU,'(A,F10.2,A,F10.2)')' Approaching R=',R,' from ',TOREAL(X2)
      IF ( INU>=0 )WRITE(*,'(A,F10.2,A,F10.2)')' Approaching R=',R,' from ',TOREAL(X2)

      ! determine weighting factor for different error functions
      IF ( ITYPE == 1 ) THEN
         ! time error function 
         WEIGHTSCALE=1
      ELSEIF( ITYPE==2 .OR. ITYPE==3 ) THEN   
         !cos error function 
         WEIGHTSCALE=PIQ/4
      ELSEIF( ITYPE == 4 ) THEN
         !abs error function 
         WEIGHTSCALE=PIQ/2
      ELSEIF( ITYPE == 5 ) THEN
         !abs error function 
         WEIGHTSCALE=PIQ/4
      ELSE
         IF ( INU>=0) WRITE(*,*)'ERROR in NONLINEAR_LSQ_FIT, type of error function is unkown',ITYPE
         STOP
      ENDIF 
   
      !use these variables for VARPRO
      ALLOCATE(ALF(AMAX), BETA(BMAX))
      ALF= 0      
      BETA=0  
      DO I = 1, N
         ALF( I ) = C( I ) 
         BETA( I ) = C( N+I )/WEIGHTSCALE
      ENDDO

      !initialize sample points  and weights
      ALLOCATE(Y(NMAX),T(NMAX,1),A(NMAX,2*AMAX+2),W(NMAX))
      !print info
      IF ( IO%NWRITE ==1 ) THEN
         !surpress output of varpro, not recomended!! 
         IPRINT = -1 
      ELSEIF (IO%NWRITE == 2 .OR. IO%NWRITE == 3  ) THEN
         !normal output of varpro
         IPRINT = 0 
      ELSE
         !print complete information
         IPRINT = 1
      ENDIF 

      NL=N  !# of current non-linear parameters
      L=NL  !# of linear parameters
      P=NL  !# of non-vanishing non-linear partial derivatives
      IV=1  !# of linear independent variables 

      DO I =1, NDATALSQ
         T(I,1)=LOG(X2)+LOG(X2)*COS(((2*(2*NDATALSQ-I)+1)*PIQ)/(4*NDATALSQ))
         T(I,1)=EXP(T(I,1))
         W(I) = 1 !WEIGHTS ARE SET TO 1
         IF ( T(I,1) > ACC ) THEN
            Y(I)=1/(T(I,1))
         ELSE
            IF (INU>=0)WRITE(INU,*)'Error in CONVERGE_FROM_RMAX2R_LSQ: ',I,' too small',T(I,1)
            STOP
         ENDIF 
         IF ( ITYPE==5) Y(I)=Y(I)/T(I,1) 
      ENDDO

      converge: DO ITER = 1, MAXNEWTON
         ! find an alternant of error function ( all local extrema)
         CALL VARPRO(N,N,NDATALSQ,NMAX,L+P+2,IV,T,Y,W,A,IPRINT,ALF,BETA,ITMAXLSQ,&
              ITYPE,IERR,INU,IO%NWRITE)
         IF (IERR<0) THEN
            WRITE(*,*)'ERROR in CONVERGE_FROM_RMAX2R_LSQ',ITER,IERR
            STOP
         ENDIF
         
         !store new coefficients
         DO I = 1, N
            C( I )=ALF( I )
            C( N+I )=BETA( I )*WEIGHTSCALE         
         ENDDO
         IF (ITYPE==3) THEN
            CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,X1,X2,C,X0,NMINALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,X1,X2,C,X0,NMAXALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUSIN_ERROR_FUNCTION,N,X1,X2,C,X0,NALTCON,INFO)
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(INU,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUSIN_ERROR_FUNCTION,X1,X2))
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(*,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUSIN_ERROR_FUNCTION,X1,X2))
         ELSEIF(ITYPE==4) THEN
            CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,X1,X2,C,X0,NMINALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,X1,X2,C,X0,NMAXALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUABS_ERROR_FUNCTION,N,X1,X2,C,X0,NALTCON,INFO)
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(INU,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUABS_ERROR_FUNCTION,X1,X2))
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(*,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUABS_ERROR_FUNCTION,X1,X2))
         ELSEIF(ITYPE==2) THEN
            CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,X1,X2,C,X0,NMINALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,X1,X2,C,X0,NMAXALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUCOS_ERROR_FUNCTION,N,X1,X2,C,X0,NALTCON,INFO)
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(INU,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUCOS_ERROR_FUNCTION,X1,X2))
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(*,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUCOS_ERROR_FUNCTION,X1,X2))
         ELSEIF(ITYPE==5) THEN
            CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,X1,X2,C,X0,NMINALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,X1,X2,C,X0,NMAXALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(NUGRE_ERROR_FUNCTION,N,X1,X2,C,X0,NALTCON,INFO)
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(INU,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUGRE_ERROR_FUNCTION,X1,X2))
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(*,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,NUGRE_ERROR_FUNCTION,X1,X2))
         ELSE
            CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,X1,X2,C,X0,NMINALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,X1,X2,C,X0,NMAXALT,INFO)
            IF (INFO<=-1)CALL FIND_ALTERNANT_EXTREMA(TAU_ERROR_FUNCTION,N,X1,X2,C,X0,NALTCON,INFO)
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(INU,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,TAU_ERROR_FUNCTION,X1,X2))
            IF (INU>=0 .AND. IO%NWRITE>2)WRITE(*,101)&
               TOREAL(X2),IERR,TOREAL(FINDMAX(N,C,TAU_ERROR_FUNCTION,X1,X2))
         ENDIF
         IF (INFO <=-1 ) THEN
            IF (INU>=0 ) WRITE(*,100)NALTCON
            IF (INU>=0 ) WRITE(*,'(F45.32)')(TOREAL(X0(I)),I=1,2*N)
            STOP
         ENDIF 

         !if R is reached close enough, do a final minimization for optimum R and get out
         IF ( X0(2*N) < R ) THEN
            IF ( INU>=0 .AND. IO%NWRITE>2) THEN
               IF (ITYPE==1)THEN
                  WRITE(INU,115)N
               ELSEIF(ITYPE==2)THEN
                  WRITE(INU,116)N
               ELSEIF(ITYPE==3)THEN
                  WRITE(INU,117)N
               ELSEIF(ITYPE==4)THEN
                  WRITE(INU,118)N
               ELSEIF(ITYPE==5)THEN
                  WRITE(INU,119)N
               ENDIF
               WRITE(INU,106)(TOREAL(BETA(I)),TOREAL(ALF(I)),I=1,N)
            ENDIF
            EXIT converge
         ELSE
            !set new minimization interval 
            X2=X0(2*N)+QDPP(0.01)
         ENDIF 

         DO I =1, NDATALSQ
            T(I,1)=LOG(X2)+LOG(X2)*COS(((2*(2*NDATALSQ-I)+1)*PIQ)/(4*NDATALSQ))
            T(I,1)=EXP(T(I,1))
            W(I) = 1 !WEIGHTS ARE SET TO 1
            IF ( T(I,1) > ACC ) THEN
               Y(I)=1/(T(I,1))
            ELSE
               IF (INU>=0)WRITE(INU,*)'Error in CONVERGE_FROM_RMAX2R_LSQ: ',I,' too small',T(I,1)
               STOP
            ENDIF 
            IF ( ITYPE==5) Y(I)=Y(I)/T(I,1) 
         ENDDO

      ENDDO converge 

      IF ( ITER >  MAXNEWTON ) THEN
         IF ( INU>=0) WRITE(INU,*)'ERROR in CONVERGE_FROM_RMAX2R_LSQ, could not reach desired R'
         STOP
      ELSE
         DO I = 1, 2*N
            COF( I )=C( I ) 
         ENDDO
      ENDIF 
      DEALLOCATE(C)

100 FORMAT('ERROR in CONVERGE_FROM_RMAX2R_LSQ: no alternant found!',/,&
           'Hint: try to increase sample points for alternant NMAXALT.',/,&
           '      Used parameter for this run:',I10)
101 FORMAT(' R=',F10.2,' done after',I5,' iterations, Error=',E14.6)
115 FORMAT(' LSQ quadrature (exp) found (',I2,'th order):')
116 FORMAT(' LSQ quadrature (cos) found (',I2,'th order):')
117 FORMAT(' LSQ quadrature (sin) found (',I2,'th order):')
118 FORMAT(' LSQ quadrature (abs) found (',I2,'th order):')
119 FORMAT(' LSQ quadrature (gre) found (',I2,'th order):')
106 FORMAT(2F45.32)
   ENDSUBROUTINE CONVERGE_FROM_RMAX2R_LSQ

!****************************************************************************
!
! For each N there's a maximum interval [1,RNMAX]. 
! Coefficients of [1,R>RNMAX] coincide with the coefficients of [1,RNMAX].
! So we minimize on [1,RNMAX]
!
!****************************************************************************

   SUBROUTINE FIND_RNMAX(R,N)
      REAL(q) :: R            !current minimization interval 
      INTEGER :: N            !order 
      !local
      REAL(q) :: RNMAX(32)    !maximum of R for each N
      !maximum of interval for each order N
      DATA RNMAX(1:32)/&      
      8.667,41.54, 146.8,436.0,1153,2807,6373,13749,&
      28387,56502,1.089E+5,2.042E+6,3.737E+5,6.691E+5,1.175E+6,2.027E+6,&
      3.440E+6,5.753E+6,9.491E+6,1.546E+7,2.491E+7,3.969E+7,6.258E+7,9.776E+7,&
      1.513E+8,2.325E+8,3.540E+8,5.353E+8,8.036E+8,1.198E+9,1.775E+9,2.614E+9/

      !For a given N all R>R(N)_max yield identical MM and LSQ coefficients
      !So we restrict the current interval, if necessary, to RNMAX  
      R = MIN(R,RNMAX(N))

      !for a tiny band gap below 0.1 (metal) switch to RNMAX
      !IF ( A<0.1 ) THEN
      !   IF ( IU0 >=0 ) WRITE(IU0,*)' Switching to metallic minimization intervals'
      !   R=RNMAX(N)
      !ENDIF  

   ENDSUBROUTINE FIND_RNMAX

!****************************************************************************
!decide if we start from a higher RMAX(N) and converge to actual R
!****************************************************************************

   SUBROUTINE RMAX2CURRENT_R(R,N,RMAX,IU0)
      REAL(q) :: R            !current minimization interval 
      INTEGER :: N            !order 
      REAL(q) :: RMAX         
      INTEGER :: IU0          
      !local
      REAL(q) :: CHECKPT(9)   !subdivisions 
      INTEGER :: IVAL(9)      !supported orders of each subdivision
      INTEGER :: I, J         
      !interval intersection (found empirically)
      DATA CHECKPT(1:9)/&
      25._q, 50._q, 106._q, 221._q, 1005._q ,&     
      10005._q, 20005._q, 50005._q, 100000._q/      
      !maximum number of grid points supported for intersection
      DATA IVAL(1:9)/&
      12, 14, 16, 20, 24, &
      28, 32, 32, 32/

      !adjust RMAX according to R, use the next larger interval 
      !find correct RMAX 
      DO I = 1, SIZE(CHECKPT)
         IF (R-CHECKPT(I)<0) EXIT
      ENDDO
   
      !if the interval is not supported, write a warning  
      IF ( I>SIZE(CHECKPT) )THEN
         IF ( IU0 >=0 ) WRITE(IU0,'(A,2E14.6)')&
            ' Error in RMAX2CURRENT_R: interval too large',R,CHECKPT(SIZE(CHECKPT))
         STOP
      ELSE
         !otherwise use the next larger interval as starting point for calculation
         RMAX=CHECKPT(I)
      ENDIF

      ! check if chosen RMAX supports given N,
      IF ( N > IVAL(I) .AND. I<SIZE(CHECKPT) ) THEN
         ! if not go to next larger interval, which supports this N
         rmax_n: DO J=I+1, SIZE(CHECKPT)
            IF ( N<= IVAL(J) ) THEN
               RMAX=CHECKPT(J)
               EXIT rmax_n
            ENDIF
         ENDDO rmax_n
      ENDIF 
    
   ENDSUBROUTINE RMAX2CURRENT_R

!============================================================================


!**********************************************************************
!
! helper routine : inverts a real matrix using LU decomposition
! 
!**********************************************************************
 
   SUBROUTINE INVERT_REAL_MATRIX( MATRIX, IU0 ) 
      IMPLICIT NONE
      REAL(q)      :: MATRIX(:,:)
      INTEGER      :: IU0
      !local 
      INTEGER             :: N
      REAL(q),ALLOCATABLE :: A(:,:)
      INTEGER,ALLOCATABLE :: IPIV(:)
      REAL(q),ALLOCATABLE :: WORK(:)
      INTEGER             :: IWORK
      INTEGER             :: INFO
    
      !obtain size of matrix  
      N=SIZE(MATRIX,1) 

      !consistency check
      IF( N/= SIZE(MATRIX,2)) THEN
         IF ( IU0>=0 ) WRITE(IU0,'(A,2I5)') 'internal error in VASP: INVERT_REAL_MATRIX reports',&
            'matrix not square',N,SIZE(MATRIX,2)
         STOP
      ENDIF
 
      !store MATRIX to A
      ALLOCATE(A(N,N))
      A(1:N,1:N)=MATRIX(1:N,1:N)
     
      !allocation of IPIV
      ALLOCATE(IPIV(N))
      
      !LU decomposition
      CALL DGETRF(N,N,A(1,1),N,IPIV,INFO)
      IF ( INFO/=0) THEN
         IF ( IU0>=0 ) WRITE(IU0,'(A,2I5)') 'internal error in VASP: INVERT_REAL_MATRIX reports',&
            'DGETRF reports:',INFO
         STOP  
      ENDIF
  
      !obtain optimal working array sizes 
      ALLOCATE(WORK(1))
      CALL DGETRI(N,A(1,1),N,IPIV,WORK,-1,INFO)
      IWORK=WORK(1)
      IF ( IWORK <= 0 )  THEN
         IF ( IU0>=0 ) WRITE(IU0,'(A,2I5)') 'internal error in VASP: INVERT_REAL_MATRIX reports',&
            'illegal size from DGETRI',IWORK
         STOP
      ENDIF
      !allocate working array with optimal size 
      DEALLOCATE(WORK)
      ALLOCATE(WORK(IWORK))
        
      !actual inversion comes now
      CALL DGETRI(N,A(1,1),N,IPIV,WORK,IWORK,INFO)
      IF ( INFO/=0) THEN
         IF ( IU0>=0 ) WRITE(IU0,'(A,2I5)') 'internal error in VASP: INVERT_REAL_MATRIX reports',&
            'DGETRI reports:',INFO
         STOP  
      ENDIF

      !store result into MATRIX, deallocate and return
      MATRIX(1:N,1:N)=A(1:N,1:N)
      DEALLOCATE(A,WORK,IPIV)
 
      RETURN      
   ENDSUBROUTINE INVERT_REAL_MATRIX


!*********************************************************************
!allocates grid handle
!*********************************************************************

   SUBROUTINE ALLOCATE_IMAG_GRID_HANDLE(GRIDS, NOMEGA)
      TYPE(imag_grid_handle) :: GRIDS     !handle 
      INTEGER                :: NOMEGA    !number of frequency points
      
      !set number of grid points
      GRIDS%NOMEGA = NOMEGA

      !nullify pointers
      NULLIFY(GRIDS%OMEGA)
      NULLIFY(GRIDS%OMEGA_WEIGHT)
      NULLIFY(GRIDS%OMEGA_SIN)
      NULLIFY(GRIDS%OMEGA_SIN_WEIGHT)
      NULLIFY(GRIDS%OMEGA_SIN_INT)
      NULLIFY(GRIDS%OMEGA_SIN_INT_WEIGHT)
      NULLIFY(GRIDS%TAU)
      NULLIFY(GRIDS%TAU_WEIGHT)
 
      !allocate the handle elements
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA) ) ALLOCATE( GRIDS%OMEGA(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA_WEIGHT) ) ALLOCATE( GRIDS%OMEGA_WEIGHT(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA_SIN) ) ALLOCATE( GRIDS%OMEGA_SIN(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA_SIN_WEIGHT) ) ALLOCATE( GRIDS%OMEGA_SIN_WEIGHT(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA_SIN_INT) ) ALLOCATE( GRIDS%OMEGA_SIN_INT(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%OMEGA_SIN_INT_WEIGHT) ) ALLOCATE( GRIDS%OMEGA_SIN_INT_WEIGHT(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%TAU) ) ALLOCATE( GRIDS%TAU(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%TAU_WEIGHT) ) ALLOCATE( GRIDS%TAU_WEIGHT(NOMEGA) ) 
      GRIDS%OMEGA=0
      GRIDS%OMEGA_WEIGHT=0
      GRIDS%OMEGA_SIN=0
      GRIDS%OMEGA_SIN_WEIGHT=0
      GRIDS%OMEGA_SIN_INT=0
      GRIDS%OMEGA_SIN_INT_WEIGHT=0
      GRIDS%TAU=0
      GRIDS%TAU_WEIGHT=0

      !Fourier transform
      NULLIFY(GRIDS%FTCOS)
      NULLIFY(GRIDS%FTCOSINV)
      NULLIFY(GRIDS%FTSIN)
      NULLIFY(GRIDS%FTSININV)
      NULLIFY(GRIDS%FTSININT)
      IF( .NOT. ASSOCIATED(GRIDS%FTCOS) ) ALLOCATE( GRIDS%FTCOS(NOMEGA,NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%FTSININT) ) ALLOCATE( GRIDS%FTSININT(NOMEGA,NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%FTSIN) ) ALLOCATE( GRIDS%FTSIN(NOMEGA,NOMEGA) ) 
      GRIDS%FTCOS=0
      GRIDS%FTSIN=0
      GRIDS%FTSININT=0
    
      NULLIFY(GRIDS%COS_ERROR)
      NULLIFY(GRIDS%SIN_ERROR)
      IF( .NOT. ASSOCIATED(GRIDS%COS_ERROR) ) ALLOCATE( GRIDS%COS_ERROR(NOMEGA) ) 
      IF( .NOT. ASSOCIATED(GRIDS%SIN_ERROR) ) ALLOCATE( GRIDS%SIN_ERROR(NOMEGA) ) 
      GRIDS%COS_ERROR=0
      GRIDS%SIN_ERROR=0
   END SUBROUTINE ALLOCATE_IMAG_GRID_HANDLE


!*********************************************************************
! add one time and frequency point
!*********************************************************************

   SUBROUTINE ADD_POINT_IMAG_GRID_HANDLE(GRIDS )
      TYPE(imag_grid_handle) :: GRIDS     !handle 
      TYPE(imag_grid_handle) :: GRIDS_NEW !handle 
    ! local
      INTEGER                :: NO        !number of frequency points
      
      !set number of grid points
      NO=GRIDS%NOMEGA
      
      CALL  ALLOCATE_IMAG_GRID_HANDLE(GRIDS_NEW, NO+1)

      ! copy content over
      GRIDS_NEW%OMEGA           (1:NO)=GRIDS%OMEGA           (1:NO)
      GRIDS_NEW%OMEGA_WEIGHT    (1:NO)=GRIDS%OMEGA_WEIGHT    (1:NO)
      GRIDS_NEW%OMEGA_SIN       (1:NO)=GRIDS%OMEGA_SIN       (1:NO)
      GRIDS_NEW%OMEGA_SIN_WEIGHT(1:NO)=GRIDS%OMEGA_SIN_WEIGHT(1:NO)
      GRIDS_NEW%TAU             (1:NO)=GRIDS%TAU             (1:NO)
      GRIDS_NEW%TAU_WEIGHT      (1:NO)=GRIDS%TAU_WEIGHT      (1:NO)
      GRIDS_NEW%FTCOS      (1:NO,1:NO)=GRIDS%FTCOS      (1:NO,1:NO)
      GRIDS_NEW%FTSIN      (1:NO,1:NO)=GRIDS%FTSIN      (1:NO,1:NO)

      CALL DEALLOCATE_IMAG_GRID_HANDLE(GRIDS)

      GRIDS=GRIDS_NEW
    
    END SUBROUTINE ADD_POINT_IMAG_GRID_HANDLE

!*********************************************************************
!deallocates grid handle
!*********************************************************************
   SUBROUTINE DEALLOCATE_IMAG_GRID_HANDLE(GRIDS)
      TYPE(imag_grid_handle) :: GRIDS     !handle 
      
      !allocate the handle elements
     IF(ASSOCIATED(GRIDS%OMEGA) ) DEALLOCATE( GRIDS%OMEGA ) 
     IF(ASSOCIATED(GRIDS%OMEGA_WEIGHT) ) DEALLOCATE( GRIDS%OMEGA_WEIGHT ) 
     IF(ASSOCIATED(GRIDS%OMEGA_SIN) ) DEALLOCATE( GRIDS%OMEGA_SIN ) 
     IF(ASSOCIATED(GRIDS%OMEGA_SIN_WEIGHT) ) DEALLOCATE( GRIDS%OMEGA_SIN_WEIGHT ) 
     IF(ASSOCIATED(GRIDS%OMEGA_SIN_INT) ) DEALLOCATE( GRIDS%OMEGA_SIN_INT ) 
     IF(ASSOCIATED(GRIDS%OMEGA_SIN_INT_WEIGHT) ) DEALLOCATE( GRIDS%OMEGA_SIN_INT_WEIGHT ) 
     IF(ASSOCIATED(GRIDS%TAU) ) DEALLOCATE( GRIDS%TAU ) 
     IF(ASSOCIATED(GRIDS%TAU_WEIGHT) ) DEALLOCATE( GRIDS%TAU_WEIGHT ) 
     IF(ASSOCIATED(GRIDS%FTCOS) ) DEALLOCATE( GRIDS%FTCOS ) 
     IF(ASSOCIATED(GRIDS%FTCOSINV) ) DEALLOCATE( GRIDS%FTCOSINV ) 
     IF(ASSOCIATED(GRIDS%FTSIN) ) DEALLOCATE( GRIDS%FTSIN ) 
     IF(ASSOCIATED(GRIDS%FTSININV) ) DEALLOCATE( GRIDS%FTSININV ) 
     IF(ASSOCIATED(GRIDS%FTSININT) ) DEALLOCATE( GRIDS%FTSININT ) 
     IF(ASSOCIATED(GRIDS%COS_ERROR) ) DEALLOCATE( GRIDS%COS_ERROR ) 
     IF(ASSOCIATED(GRIDS%SIN_ERROR) ) DEALLOCATE( GRIDS%SIN_ERROR ) 
   END SUBROUTINE DEALLOCATE_IMAG_GRID_HANDLE

!*********************************************************************
!
! reader for some flags
!
!*********************************************************************

   SUBROUTINE MINIMAX_READER(IU5, IU0)
      USE vaspxml
      USE base
      IMPLICIT NONE
      INTEGER IU5,  IU0
      ! local
      INTEGER  N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN, LDUM
      CHARACTER (1) :: CHARAC
  
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
      ! maxiterations in sloppy remez algorithm (default =0)
      MAXITER_FT=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'MAXITER_FT','=','#',';','I', &
           &            MAXITER_FT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''MAXITER_FT'' from file INCAR.'
         MAXITER_FT=-1
      ENDIF
      IF ( MAXITER_FT <0 ) MAXITER_FT = 0
      CALL XML_INCAR('MAXITER_FT','I',MAXITER_FT,RDUM,CDUM,LDUM,CHARAC,N)

      !maximum number of sample points for alternant in Remez algorithms
      !default =1500
      NMAXALT=1500
      CALL RDATAB(LOPEN,INCAR,IU5,'NMAXALT','=','#',';','I', &
           &            NMAXALT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NMAXALT'' from file INCAR.'
         NMAXALT=1500
      ENDIF
      CALL XML_INCAR('NMAXALT','I',NMAXALT,RDUM,CDUM,LDUM,CHARAC,N)

      !maximum number of iteration steps in LSQ search algorithm
      !default =1500
      ITMAXLSQ=1500
      CALL RDATAB(LOPEN,INCAR,IU5,'ITMAXLSQ','=','#',';','I', &
           &            ITMAXLSQ,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''ITMAXLSQ'' from file INCAR.'
         ITMAXLSQ=1500
      ENDIF
      CALL XML_INCAR('ITMAXLSQ','I',ITMAXLSQ,RDUM,CDUM,LDUM,CHARAC,N)

      !number of sampling points for LSQ search algorithm
      !default =100
      NDATALSQ=100
      CALL RDATAB(LOPEN,INCAR,IU5,'NDATALSQ','=','#',';','I', &
           &            NDATALSQ,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''NDATALSQ'' from file INCAR.'
         NDATALSQ=100
      ENDIF
      CALL XML_INCAR('NDATALSQ','I',NDATALSQ,RDUM,CDUM,LDUM,CHARAC,N)

      CLOSE(IU5)
   END SUBROUTINE MINIMAX_READER


ENDMODULE minimax
