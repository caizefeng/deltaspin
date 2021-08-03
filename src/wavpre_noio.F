#include "symbol.inc"
#define LNEW
#define USE_ZHEEVX
      MODULE mwavpre_noio
      USE prec
      CONTAINS
!*************************** WAVPRE_NOIO *****************************
! RCS:  $Id: wavpre_noio.F,v 1.4 2001/04/05 10:34:15 kresse Exp $
!
! this subroutine performes a prediction of the wavefunctions using
! the subspace-allignmt scheme proposed by  T. A. Arias et.al.
! in addition the chargedensity is speratly predicted
! arrays are used for storing the old wavefunctions and old
! postition of the ions
! following flags are important
!
! INIPRE  0 standard operation (perform extrapolation)
!         1 new startup (only POSION must be correct)
!         2 should never happen (start from TMPCAR for io-version)
!         3 restart with new ionic positions
!         4 extrapolate charge using atomic charge-densities
!           and extrapolate wavefunctions
!         5 extrapolate charge only
! POSION  must contain the new positions
! CSTRF   must contain the *old* structure-factor
!         on return it holds the new structure-factor if IPRE<0
! CPTWFP  must contain the old wavefunctions
! IU      File Unit to use
! ICMPLX  size of COMPLEX(q) item
! IPRE    prediction of wavefunctions and eigenvalues performed
!           0  nothing done
!          -1  charge from overlapping atoms
!         <-1  wavefuntions predicted charge from overlapping atoms
!         > 1  wavefuntions and charge predicted
!
! short description of arrays for no I/O version:
!   W_1 ... previous-wavefunctions
!   W_2 ... previous-change in wf.
!   CHTOT_1 ... last charge
!   CHTOT_2 ... last change in charge
!   RHOLM_1 ... last paw onsite charge terms
!   RHOLM_2 ... last change in there terms
!   POS_1, POS_2, POS_3 ... last position and last changes in positions
!   ICALL_STORE ... Nr. of calls for each band
!
! LOPEN only for compatibility
!
! 1998.02: improved charge extrapolation added by Dario Alfe (Feb 1998)
! 2012.03.14: Dario Alfe parallelization for matrix multiplication 
!
!*********************************************************************

      SUBROUTINE WAVPRE_NOIO(GRIDC,P,PRED,T_INFO,W,WDES,LATT_CUR,LOPEN, &
         CHTOT,RHOLM,NPAW, CSTRF, KINEDEN, LMDIM,CQIJ,LOVERL,IU0)
      USE prec
      USE base
      USE poscar
      USE wave
      USE wave_mpi
      USE pseudo
      USE lattice
      USE mpimy
      USE mgrid
      USE charge
      USE wave_high
      USE choleski
      USE scala
      USE jacobi
      USE dfast
      USE meta

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (prediction)  PRED
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES,WDES_RED
      TYPE (wavedes1)    WDES1
      TYPE (tau_handle)  KINEDEN

      LOGICAL LOVERL,DO_REDIS
      LOGICAL LOPEN

      OVERLAP CQIJ(LMDIM,LMDIM,T_INFO%NIOND,WDES%ISPIN)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%ISPIN)
      REAL(q)    RHOLM(NPAW,WDES%ISPIN)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

      COMPLEX(q) ZDOTC

! this variables have to be saved
      INTEGER,SAVE :: ICALLS
      REAL(q),ALLOCATABLE,SAVE :: POS_1(:,:), POS_2(:,:), POS_3(:,:)
! work arrays not important for performance
      TYPE (wavefun)  ::    WTMP
      REAL(q) R(WDES%NB_TOT)
#ifndef USE_ZHEEVX
      REAL(q)    RWORK(3*WDES%NB_TOT)
#else
      REAL(q)    RWORK(7*WDES%NB_TOT) ; INTEGER IWORK(5*WDES%NB_TOT), INFO(WDES%NB_TOT)
      PARAMETER  (LWORK=32)
      GDEF,ALLOCATABLE :: CTMP(:)
#endif
      REAL(q) POS_3L(3,T_INFO%NIONS)
! work arrays  (CW=CPTWFL)
      TYPE (wavespin),SAVE ::    W_1,W_2
      COMPLEX(q),ALLOCATABLE,SAVE :: CHTOT_1(:,:),CHTOT_2(:,:)
      REAL(q),ALLOCATABLE,SAVE    :: RHOLM_1(:,:),RHOLM_2(:,:)
      COMPLEX(q),ALLOCATABLE,SAVE :: TAU_1(:,:)  ,TAU_2(:,:)
      INTEGER,ALLOCATABLE,SAVE :: ICALL_STORE(:,:,:)
      COMPLEX(q),ALLOCATABLE :: CD0(:),CD1(:),CW(:)
      GDEF,ALLOCATABLE,TARGET :: CPROW(:,:),CPRO0(:),CPRO1(:),CPROL(:)
      GDEF,ALLOCATABLE :: CA(:,:),CAP(:,:),CU(:,:)
! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:)
      GDEF   , POINTER :: CPROW_RED(:,:),CPROJ_RED(:,:)
#ifdef LNEW
      LOGICAL :: new_version = .TRUE.
#else
      LOGICAL :: new_version = .FALSE.
#endif
      LOGICAL :: symplectic=.FALSE.

#ifdef MPI
      INTEGER, ALLOCATABLE, SAVE :: NCOL(:), OFFSET(:), NCTOT(:), OFFDATA(:)
      ! NCOL    is the number of columns on each node
      ! OFFSET  is the first column index on each node
      ! NCTOT   total number of data on each node (=NBTOT*NCOL)
      ! OFFDATA offset for data in matrix on each node (=NBTOT*OFFSET)
#ifdef gammareal
      MPIDATA = MPI_double_precision
#else
      MPIDATA = MPI_double_complex
#endif
#endif


!***********************************************************************
! determine whether redistribution is required
!***********************************************************************
#ifdef MPI
      NCPU=WDES%COMM_INTER%NCPU   ! number of cores involved in band dis.
#else
      NCPU=1
#endif
      IF (NCPU /= 1) THEN

        DO_REDIS=.TRUE.
        NRPLWV_RED=WDES%NRPLWV/NCPU
        NPROD_RED =WDES%NPROD /NCPU

      ELSE

        DO_REDIS=.FALSE.
        NRPLWV_RED=WDES%NRPLWV
        NPROD_RED =WDES%NPROD

      ENDIF
      NB_TOT=WDES%NB_TOT
      NBANDS=WDES%NBANDS

!***********************************************************************
! PRED%INIPRE 2,>5 not allowed anymore
!***********************************************************************

      IF (PRED%INIPRE==2 .OR.PRED%INIPRE>5 .OR.PRED%INIPRE<0) THEN
        WRITE(*,*) 'internal ERROR: WAVPRE_NOIO: INIPRE=',PRED%INIPRE
        STOP
      ENDIF

!***********************************************************************
! PRED%INIPRE==1 or 3 ... perform initialisation of arrays
!***********************************************************************
      IF (PRED%INIPRE==1 .OR.PRED%INIPRE==3) THEN
!-----------------------------------------------------------------------
! allocate arrays
! for  INIPRE==3 (RESTART) the arrays are already allocated
!-----------------------------------------------------------------------
        IF (PRED%INIPRE==1) THEN
        ! create a descriptor for the redistributed wavefunctions
          WDES_RED=WDES
          WDES_RED%NBANDS=NB_TOT
          WDES_RED%NRPLWV=NRPLWV_RED
          WDES_RED%NPROD =NPROD_RED
        ! allocate the wavefunctions arrays
          CALL ALLOCW(WDES_RED,W_1,WTMP,WTMP)
          CALL ALLOCW(WDES_RED,W_2,WTMP,WTMP)
        ! allocate all other arrays
          ALLOCATE(CHTOT_1(GRIDC%MPLWV,WDES%NCDIJ))
          ALLOCATE(CHTOT_2(GRIDC%MPLWV,WDES%NCDIJ))
          ALLOCATE(RHOLM_1(NPAW,WDES%NCDIJ))
          ALLOCATE(RHOLM_2(NPAW,WDES%NCDIJ))
          IF (LDO_METAGGA()) THEN
             ALLOCATE(TAU_1(GRIDC%MPLWV,WDES%NCDIJ))
             ALLOCATE(TAU_2(GRIDC%MPLWV,WDES%NCDIJ))
          ENDIF
          ALLOCATE(ICALL_STORE(NB_TOT,WDES%NKPTS,WDES%ISPIN))
          ALLOCATE(POS_1(3,T_INFO%NIONS))
          ALLOCATE(POS_2(3,T_INFO%NIONS))
          ALLOCATE(POS_3(3,T_INFO%NIONS))
!-----------------------------------------------------------------------
! give them some data
!-----------------------------------------------------------------------
          ICALL_STORE=0
          W_1%CW   =0
          W_1%CPROJ=0
          W_1%CELTOT=0
          W_2%CW=0
          W_2%CPROJ=0
          W_2%CELTOT=0
          CHTOT_1=CHTOT
          CHTOT_2=CHTOT
          RHOLM_1=RHOLM
          RHOLM_2=RHOLM
          IF (LDO_METAGGA()) THEN
             TAU_1=KINEDEN%TAU
             TAU_2=KINEDEN%TAU
          ENDIF
#ifdef MPI
          NPROC=WDES%COMM_KIN%NCPU ! total number of cores per k point
          ALLOCATE( NCOL(NPROC), OFFSET(NPROC), NCTOT(NPROC), OFFDATA(NPROC) )
          CALL DISTRIBUTE_COLUMNS(NB_TOT, NPROC, NCOL)
          OFFSET(1) = 0
          OFFDATA(1) = 0
          NCTOT(1) = NCOL(1) * NB_TOT
          DO I=2,NPROC
             OFFSET(I)  =OFFSET(I-1) + NCOL(I-1)
             NCTOT(I)   = NB_TOT * NCOL(I)
             OFFDATA(I) = NB_TOT * OFFSET(I)
          ENDDO
#endif
          IF (IU0>=0) &
          WRITE(IU0,*)'prediction of wavefunctions initialized - no I/O'
        ENDIF
!-----------------------------------------------------------------------
! store new ionic positions and set ICALLS to 0 (very important)
!-----------------------------------------------------------------------
        ICALLS=0
        POS_1=T_INFO%POSION
        POS_2=0
        POS_3=0
! that is it for this time
        GOTO 3000
!-----ENDIF (PRED%INIPRE==1 .OR.PRED%INIPRE==3)
      ENDIF

!***********************************************************************
! PRED%INIPRE==5 forget about the wavefunctions and simply extrapolate charge
!***********************************************************************
      IF (PRED%INIPRE==5) GOTO 2000

!***********************************************************************
! PRED%INIPRE==0 or 4
!***********************************************************************
!-----------------------------------------------------------------------
! init
!-----------------------------------------------------------------------
      IWARN=0

      ALLOCATE(CD0(WDES%NRPLWV),CD1(WDES%NRPLWV),CW(WDES%NRPLWV), &
               CPRO0(WDES%NPROD),CPRO1(WDES%NPROD),CPROL(WDES%NPROD), &
               CPROW(WDES%NPROD,WDES%NBANDS), &
               CA(NB_TOT,NB_TOT),CAP(NB_TOT,NB_TOT), &
               CU(NB_TOT,NB_TOT))
!***********************************************************************
!                  *****    FIRST STEP:    *****
!  calculate the extrapolation parameters
!***********************************************************************

!=======================================================================
! calculate extrapolation parameters ALPHA,BETA
!=======================================================================
      ICALLS=ICALLS+1

      POS_3L=POS_3
      POS_3=POS_2
      DO N=1,T_INFO%NIONS
        POS_2(1,N)=MOD(T_INFO%POSION(1,N)-POS_1(1,N)+1.5_q,1._q)-.5_q
        POS_2(2,N)=MOD(T_INFO%POSION(2,N)-POS_1(2,N)+1.5_q,1._q)-.5_q
        POS_2(3,N)=MOD(T_INFO%POSION(3,N)-POS_1(3,N)+1.5_q,1._q)-.5_q
      ENDDO
      POS_1=T_INFO%POSION

      PRED%ALPHA=0
      PRED%BETA =0
!-----------------------------------------------------------------------
! after two calls linear extrapolation
! PRED%ALPHA=1 , PRED%BETA=0
!-----------------------------------------------------------------------
      IF (ICALLS==2) THEN
        CALL CLCD_NOIO(T_INFO%NIONS,POS_3,POS_3,A0)
        CALL CLCD_NOIO(T_INFO%NIONS,POS_2,POS_3,A1)

        PRED%ALPHA=A1/A0
        PRED%BETA =0
      ELSE IF (ICALLS>=3) THEN
!-----------------------------------------------------------------------
! after 3 calls fit new ionic positions to an linearcombinition
! of  old ionic postitions
!
! calculate all necessary quantities i.e.
! A0 = || D(t(N+1)) || , A2 = || D(t(N)) , B2= || D(t(N-1)) ||
! A1 = -2 D(t(N)) * D(t(N+1)), B1= -2 D(t(N-1)) * D(t(N+1)),
! AB =  2 D(t(N)) * D(t(N-1))
!-----------------------------------------------------------------------
      CALL CLCD_NOIO(T_INFO%NIONS,POS_2,POS_2,A0)
      CALL CLCD_NOIO(T_INFO%NIONS,POS_3,POS_3,A2)
      CALL CLCD_NOIO(T_INFO%NIONS,POS_3L,POS_3L,B2)
      CALL CLCD_NOIO(T_INFO%NIONS,POS_2,POS_3,A1)
      CALL CLCD_NOIO(T_INFO%NIONS,POS_2,POS_3L,B1)
      CALL CLCD_NOIO(T_INFO%NIONS,POS_3,POS_3L,AB)

      A1=-2*A1
      B1=-2*B1
      AB= 2*AB

!-----------------------------------------------------------------------
!  if D(t(N+1)) == D(t(N)) == D(t(N-1))
!  <-> |   D(t(N+1))-D(t(N)) |  =0
!  than set    PRED%ALPHA to 2, and PRED%BETA to -1
!-----------------------------------------------------------------------
      IF( ABS(A0+A2+A1)/A0 < 1E-4_q .AND. ABS(A0+B2+B1)/A0 < 1E-4_q) &
     &THEN
        PRED%ALPHA= 2
        PRED%BETA= -1
        IF (IU0>=0) WRITE(IU0,*)'positions are collinear'
      ELSE

!-----------------------------------------------------------------------
! we must minimise following function
! A0+ PRED%ALPHA**2 A2 + PRED%ALPHA A1 + PRED%BETA**2 B2 + PRED%BETA B1 + PRED%ALPHA PRED%BETA AB
!
! PRED%ALPHA is approximatly 2, PRED%BETA is -1 (not astonishing)
!-----------------------------------------------------------------------
      PRED%ALPHA= -(2*B2*A1-AB*B1)/(4*A2*B2-AB*AB)
      PRED%BETA = -(2*A2*B1-AB*A1)/(4*A2*B2-AB*AB)
!      FMIN = A0+ PRED%ALPHA**2 * A2 + PRED%ALPHA * A1 + PRED%BETA**2*  B2 + PRED%BETA* B1
!     &        + PRED%ALPHA*  PRED%BETA*  AB
!     IF (IU0>=0) &
!      WRITE(IU0,'(8E10.3)')PRED%ALPHA,PRED%BETA,A0,A2,B2,A1,B1,AB
      ENDIF

!-----ENDIF (ICALLS>=3)
      ENDIF

!      IF (IU0>=0) WRITE(IU0,*) 'linear extrapolation only in wavpre_noio.F'
!      PRED%ALPHA=1
!      PRED%BETA =0

!=======================================================================
! Loop over k-Points and spins
!=======================================================================
      PRED%IPRE=ICALLS
 spin:   DO ISP=1,WDES%ISPIN
 kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES1,NK)

!   get pointers for redistributed wavefunctions
!   I can not guarantee that this works with all f90 compilers
!   please see comments in wave_mpi.F
      IF (NCPU /= 1) THEN
        CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROW_RED, NPROD_RED, NB_TOT, CPROW(1,1))
      ELSE  !furth
        CW_RED    => W%CW(:,:,NK,ISP)
        CPROJ_RED => W%CPROJ(:,:,NK,ISP)
        CPROW_RED => CPROW(:,:)
      ENDIF  !furth

!   set number of wavefunctions after redistribution
      NPL = WDES1%NPL     ! number of plane waves/node after redis
      NPRO= WDES1%NPRO    ! number of projected wavef. after redis

      CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

      NPRO_O=NPRO
      IF (.NOT. LOVERL) NPRO_O=0
!***********************************************************************
!                  *****   SECOND STEP:    *****
! calculate the sub-space-allignment matrix
! notation: N  current wavefunction
!           Mp old wavefunction
!***********************************************************************
!-----------------------------------------------------------------------
! read in the old wavefunction and calculate U(N,M) = <N,Mp>
!-----------------------------------------------------------------------
      CU=0
!  caclulate Q |cfin_n>
      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,1),W%CPROJ(1,1,NK,ISP),CPROW(1,1))
    ! redistribute everything

      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
      ENDIF
!-----------------------------------------------------------------------
! calculate the hermititian matrix AP(N,K)= CONJG(U(N,M))*U(K,M)
!-----------------------------------------------------------------------
 if1: IF (ICALLS>=2) THEN

! calculate the matrix U(N,M) = <N,Mp>

      CALL ORTH2( &
     &  CW_RED(1,1),W_1%CW(1,1,NK,ISP),CPROW_RED(1,1),W_1%CPROJ(1,1,NK,ISP), &
     &  NB_TOT,1,NB_TOT,NPL,NPRO_O,NRPLWV_RED,NPROD_RED,CU(1,1))

      CALLMPI( M_sum_g(WDES%COMM_KIN,CU(1,1),NB_TOT*NB_TOT))

#ifndef MPI
! old serial version
      CALL GGEMM( trans, 'N', NB_TOT, NB_TOT, NB_TOT, one, CU, NB_TOT, &
     &            CU, NB_TOT, zero, CAP, NB_TOT)
#else
      ALLOCATE(CTMP(NCTOT(WDES%COMM_KIN%NODE_ME)))

      CTMP=0._q
      CALL GGEMM( trans, 'N', NB_TOT, NCOL(WDES%COMM_KIN%NODE_ME), NB_TOT, one, CU, NB_TOT, &
     &            CU(1,1+OFFSET(WDES%COMM_KIN%NODE_ME)), NB_TOT, zero, CTMP, NB_TOT )

      CAP=0._q
      CALL MPI_allgatherv (CTMP, NCTOT(WDES%COMM_KIN%NODE_ME), MPIDATA, &
         & CAP, NCTOT, OFFDATA, MPIDATA, WDES%COMM_KIN%MPI_COMM, INFO)

      DEALLOCATE(CTMP)
#endif
#ifdef debug
      io_begin
        NPL2=MIN(10,NB_TOT)
        DO  N1=1,NPL2
          WRITE(6,1)N1,(REAL( CU(N1,N2) ,KIND=q) ,N2=1,NPL2)
        ENDDO
        WRITE(6,*)
#ifndef gammareal
        DO N1=1,NPL2
          WRITE(6,2)N1,(AIMAG(CU(N1,N2)),N2=1,NPL2)
        ENDDO
        WRITE(6,*)
#endif
      1 FORMAT(1I2,3X,20F9.5)
      2 FORMAT(1I2,3X,20E9.1)
      io_end
#endif
!-----------------------------------------------------------------------
! Diagonalize the resulting Matrix
! calling LAPACK the result is
!    MATRIX(N,K)* EIGENV(K,M) = COS^2(THETA)* EIGENV(N,M)
! descending order of eigenvalues as mentioned in paper is nonsense
!-----------------------------------------------------------------------
#ifndef gammareal
      DO N1=1,NB_TOT
        IF (ABS(AIMAG(CAP(N1,N1)))>1E-2_q) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'WARNING: Sub-Space-Matrix is not hermitian wav',AIMAG(CAP(N1,N1))
        ENDIF
        CAP(N1,N1)= REAL( CAP(N1,N1) ,KIND=q)
      ENDDO
#endif
      IFAIL=0
!
! parallel versions currently only gamma point only
!
#if defined(MPI)
      ! Jacobi if available in parallel version
      IF ( LJACOBI ) THEN
          CALL jacDSSYEV(WDES%COMM_KIN, CAP, R, NB_TOT)
          CALLMPI( M_sum_g(WDES%COMM_KIN, CAP, NB_TOT*NB_TOT))
          CALLMPI( M_sum_g(WDES%COMM_KIN, R , NB_TOT))

          GOTO 100
      ENDIF

      ! scaLAPACK if available in parallel version
      IF ( LscaLAPACK ) THEN
        CALL pDSSYEX_ZHEEVX(WDES%COMM_KIN, CAP, R, NB_TOT, NB_TOT)
        CALLMPI( M_sum_g(WDES%COMM_KIN,CAP, NB_TOT*NB_TOT))

        GOTO 100
      ENDIF
#endif
!
!  seriell codes
!
#ifdef  gammareal
#ifndef USE_ZHEEVX
      CALL DSYEV &
     &    ('V','U',NB_TOT,CAP,NB_TOT,R, &
     &           CA,NB_TOT*NB_TOT,IFAIL)
#else

      ABSTOL=1E-10_q
      VL=0 ; VU=0 ; IL=0 ; IU=0
      ALLOCATE(CTMP(LWORK*NB_TOT))
      CALL DSYEVX( 'V', 'A', 'U', NB_TOT, CAP , NB_TOT, VL, VU, IL, IU, &
                            ABSTOL , NB_CALC , R, CA, NB_TOT, CTMP, &
                            NB_TOT*LWORK, RWORK, IWORK, INFO, IFAIL )         
      DEALLOCATE(CTMP)
      CAP=CA

#endif
#else
#ifndef USE_ZHEEVX
      CALL ZHEEV &
          ('V','U',NB_TOT,CAP,NB_TOT,R, &
                 CA,NB_TOT*NB_TOT,RWORK,IFAIL)
#else

      ABSTOL=1E-10_q
      VL=0 ; VU=0 ; IL=0 ; IU=0
      ALLOCATE(CTMP(LWORK*NB_TOT))
      CALL ZHEEVX( 'V', 'A', 'U', NB_TOT, CAP , NB_TOT, VL, VU, IL, IU, &
                            ABSTOL , NB_CALC , R, CA, NB_TOT, CTMP, &
                            NB_TOT*LWORK, RWORK, IWORK, INFO, IFAIL )         
      DEALLOCATE(CTMP)
      CAP=CA

#endif
#endif

 100  CONTINUE

      IF (IFAIL/=0) THEN
        WRITE(*,*) 'ERROR WAVPRE_NOIO: Call to routine ZHEEV failed!', IFAIL
        STOP
      ENDIF

! Store the matrix in A(N,K)=CONJG(EIGENV(K,N))
      DO N1=1,NB_TOT
      DO N2=1,NB_TOT
        CA(N1,N2)= GCONJG(CAP(N2,N1))
      ENDDO
      ENDDO

! calculate AP(N,M)=A(N,K)*CONJG(U(M,K))
#ifndef MPI
! old serial version
      CALL GGEMM( 'N', trans, NB_TOT, NB_TOT, NB_TOT, one, CA, NB_TOT, &
     &            CU, NB_TOT, zero , CAP, NB_TOT)
#else
      ALLOCATE(CTMP(NCTOT(WDES%COMM_KIN%NODE_ME)))

      CU = TRANSPOSE( GCONJG(CU) ); CTMP=0._q
      CALL GGEMM( 'N','N', NB_TOT, NCOL(WDES%COMM_KIN%NODE_ME), NB_TOT, one, CA, NB_TOT, &
      &            CU(1,1+OFFSET(WDES%COMM_KIN%NODE_ME)), NB_TOT, zero, CTMP, NB_TOT )

      CAP=0._q
      CALL MPI_allgatherv (CTMP, NCTOT(WDES%COMM_KIN%NODE_ME), MPIDATA, &
      & CAP, NCTOT, OFFDATA, MPIDATA, WDES%COMM_KIN%MPI_COMM, INFO)

      DEALLOCATE(CTMP)
#endif
!-----------------------------------------------------------------------
! AP(N,K)=Diag(1/-Sqrt(Eigenvalue))*U2(N,K)
!-----------------------------------------------------------------------
      DO N1=1,NB_TOT
      ESQR=1/SQRT(R(N1))
      DO N2=1,NB_TOT
        CAP(N1,N2)=CAP(N1,N2)*ESQR
      ENDDO
      ENDDO

! CU(N,M)=CONJG(CAP(K,N))*CA(K,M)
#ifndef MPI
! old serial version
      CALL GGEMM(trans, 'N', NB_TOT, NB_TOT, NB_TOT, one, CAP, NB_TOT, &
     &             CA, NB_TOT, zero , CU, NB_TOT)
#else
      ALLOCATE(CTMP(NCTOT(WDES%COMM_KIN%NODE_ME)))

      CTMP=0._q
      CALL GGEMM( trans, 'N', NB_TOT, NCOL(WDES%COMM_KIN%NODE_ME), NB_TOT, one, CAP, NB_TOT, &
     &            CA(1,1+OFFSET(WDES%COMM_KIN%NODE_ME)), NB_TOT, zero, CTMP, NB_TOT )

      CU=0._q
      CALL MPI_allgatherv (CTMP, NCTOT(WDES%COMM_KIN%NODE_ME), MPIDATA, &
         & CU, NCTOT, OFFDATA, MPIDATA, WDES%COMM_KIN%MPI_COMM, INFO)

      DEALLOCATE(CTMP)
#endif
!=======================================================================
! set CAP to the invers transformation
!=======================================================================
      DO N2=1,NB_TOT
      DO N1=1,NB_TOT
         CAP(N1,N2)= GCONJG(CU(N2,N1))
      ENDDO
      ENDDO

  ENDIF if1
!***********************************************************************
!                  *****   INIPRE = 0 or 4  *****
! align wavefunctions
! and perform extrapolation
!***********************************************************************
      IF (ICALLS>=2) THEN
! rotate old wavefunction CW by multiplying with CAP
      CALL LINCOM('F',W_1%CW(:,:,NK,ISP),W_1%CPROJ(:,:,NK,ISP),CAP(1,1), &
     &      NB_TOT,NB_TOT, NPL,NPRO,NRPLWV_RED,NPROD_RED, NB_TOT, &
     &      W_1%CW(:,:,NK,ISP),W_1%CPROJ(:,:,NK,ISP))

! rotate last change of wavefunction by CAP
      CALL LINCOM('F',W_2%CW(:,:,NK,ISP),W_2%CPROJ(:,:,NK,ISP),CAP(1,1), &
     &      NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED, NB_TOT, &
     &      W_2%CW(:,:,NK,ISP),W_2%CPROJ(:,:,NK,ISP))

      ENDIF
!=======================================================================
! extrapolation of wavefunctions
! loop over all bands
!=======================================================================
!-----------------------------------------------------------------------
! read in old wafefunction CW and last change in Wavefunction CD1
! and caclulate Q |MP>  Q (overlap-matrix) and store in CPRO0
!-----------------------------------------------------------------------
      band: DO N=1,WDES%NB_TOT
      ICALL              = ICALL_STORE(N,NK,ISP)
      CELEL              = W_1%CELTOT(N,NK,ISP)
      CW(1:NPL)          = W_1%CW(1:NPL,N,NK,ISP)
      CPROL(1:NPRO)      = W_1%CPROJ(1:NPRO,N,NK,ISP)
      CDCEL1             = W_2%CELTOT(N,NK,ISP)
      CD1(1:NPL)         = W_2%CW(1:NPL,N,NK,ISP)
      CPRO1(1:NPRO)      = W_2%CPROJ(1:NPRO,N,NK,ISP)

! ICALL must not be smaller then ICALLS
! after a reinitilization ICALLS is set to 0, but ICALL is not reset

      ICALL=MIN(ICALL,ICALLS-1)
!-----------------------------------------------------------------------
! test orthogonality of CPTWFP CW
! if they are orthogonal band crossing occured, reset ICALL to 1
!-----------------------------------------------------------------------
      WFMAG  = ZDOTC(NPL,CW_RED(1,N),1,CW(1),1)
      WFMAGL = ZDOTC(NPL,CW_RED(1,N),1,CW_RED(1,N),1)
      IF (LOVERL) THEN
      WFMAG =WFMAG + GDOTC( NPRO_O,CPROW_RED(1,N),1,CPROL(1),1)
      WFMAGL=WFMAGL+ GDOTC( NPRO_O,CPROJ_RED(1,N),1,CPROW_RED(1,N),1)
      ENDIF

      CALLMPI( M_sum_d(WDES%COMM_KIN, WFMAG, 1))
      CALLMPI( M_sum_d(WDES%COMM_KIN, WFMAGL, 1))

      DWRITE0 N,ICALL,WFMAG,WFMAGL

      IF (ABS(WFMAGL-1._q) > 1E-3 .AND. ICALL >=1 ) THEN
        IWARN=IWARN+1
      ENDIF
      WFMAG=WFMAG/WFMAGL

      CDCEL0       =0
      CD0(1:NPL)   =0
      CPRO0(1:NPRO)=0

      IF (ICALL/=0 .AND.ABS(WFMAG)>0.90_q) THEN
        ICALL=ICALL+1
        DO M=1,NPL
          CD0(M)  =CW_RED(M,N)-CW(M)
        ENDDO
        DO M=1,NPRO
          CPRO0(M)=CPROJ_RED(M,N)-CPROL (M)
        ENDDO

        CDCEL0=  W%CELTOT(N,NK,ISP)-CELEL
      ELSEIF (ICALL==0) THEN
        ICALL=1
      ELSE
        ICALL=1
        IF (IU0>=0) WRITE(IU0,'(A,I4,F8.4)')'Information: wavefunction orthogonal band ',N,WFMAG
      ENDIF

      ICALL_STORE(N,NK,ISP)      = ICALL
      W_1%CELTOT(N,NK,ISP)       = W%CELTOT(N,NK,ISP)
      W_1%CW(1:NPL,N,NK,ISP)     = CW_RED(1:NPL,N)
      W_1%CPROJ(1:NPRO,N,NK,ISP) = CPROJ_RED(1:NPRO,N)
      W_2%CELTOT(N,NK,ISP)       = CDCEL0
      W_2%CW(1:NPL,N,NK,ISP)     = CD0(1:NPL)
      W_2%CPROJ(1:NPRO,N,NK,ISP) = CPRO0(1:NPRO)

!-----------------------------------------------------------------------
! extrapolate  now
!-----------------------------------------------------------------------
      IF (ICALL>=3) THEN
        DO M=1,NPL
          CW_RED(M,N)=CW_RED(M,N)+PRED%ALPHA*CD0(M)+PRED%BETA*CD1(M)
        ENDDO
        DO M=1,NPRO
          CPROJ_RED(M,N)=CPROJ_RED(M,N)+ &
     &                       PRED%ALPHA*CPRO0(M)+PRED%BETA*CPRO1(M)
        ENDDO

         W%CELTOT(N,NK,ISP)= &
     &      W%CELTOT(N,NK,ISP)+PRED%ALPHA*CDCEL0+PRED%BETA*CDCEL1
      ELSE IF (ICALL>=2) THEN
        DO M=1,NPL
          CW_RED(M,N)=CW_RED(M,N)+CD0(M)
        ENDDO
        DO M=1,NPRO
          CPROJ_RED (M,N)=CPROJ_RED (M,N)+CPRO0(M)
        ENDDO
        W%CELTOT(N,NK,ISP)=W%CELTOT(N,NK,ISP)+CDCEL0
      ENDIF

!-----------------------------------------------------------------------
      ENDDO band
!-----------------------------------------------------------------------
      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
      ENDIF

  ENDDO kpoint
  ENDDO spin

!-----------------------------------------------------------------------
      IF (IWARN/=0) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'WARNING: WAVPRE_NOIO: old wavefunctions are not orthogonal'
      ENDIF

!***********************************************************************
!   ***           extrapolate charge-density            ***
!
! PRED%INIPRE = 4,5  use atomic charge-densities instead of prediction
! Dario Alfe implemented a slightly improved algorithm
! which extrapolates the difference between the current and the
! superpostition of atomic charge densities (i.e. bond charge)
! n_bond(t)= n(t) - n_at(t)
! with a quadratic extrapolation
!***********************************************************************
      IF (PRED%INIPRE==4) ICALLS=MIN(1,ICALLS)

      DEALLOCATE(CD0,CD1,CW,CPRO0,CPRO1,CPROL,CPROW,CA,CAP,CU)
      ALLOCATE(CD0(GRIDC%MPLWV))

!---- subtract atomic charge-density for old positions
      IF ( new_version ) THEN
         CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
         DO N=1,GRIDC%RC%NP
            CHTOT(N,1)= CHTOT(N,1)- CD0(N)
         ENDDO
      ENDIF

      IF (symplectic) THEN
         DO ISP=1,WDES%NCDIJ
            DO N=1,GRIDC%RC%NP
               ! tilde rho(n-2)
               CH1=CHTOT_2(N,ISP)
               ! tilde rho(n-1)
               CH0=CHTOT_1(N,ISP)
               ! linear symplectic predictor
               IF (ICALLS>=2) CHTOT(N,ISP)=2*CHTOT(N,ISP)-CH0
               ! shift slot by one and store predicted charge
               CHTOT_2(N,ISP)=CH0
               CHTOT_1(N,ISP)=CHTOT(N,ISP)
            ENDDO

            IF (LDO_METAGGA()) THEN
               DO N=1,GRIDC%RC%NP
                  ! tilde rho(n-2)
                  CH1=TAU_2(N,ISP)
                  ! tilde rho(n-1)
                  CH0=TAU_1(N,ISP)
                  ! linear symplectic predictor
                  IF (ICALLS>=2) KINEDEN%TAU(N,ISP)=2*KINEDEN%TAU(N,ISP)-CH0
                  ! shift slot by one and store predicted charge
                  TAU_2(N,ISP)=CH0
                  TAU_1(N,ISP)=KINEDEN%TAU(N,ISP)
               ENDDO
            ENDIF

            DO N=1,NPAW
               ! tilde rho(n-2)
               CH1=RHOLM_2(N,ISP)
               ! tilde rho(n-1)
               CH0=RHOLM_1(N,ISP)
               ! linear symplectic predictor
               IF (ICALLS>=2) RHOLM(N,ISP)=2*RHOLM(N,ISP)-CH0
               ! shift slot by one and store predicted charge
               RHOLM_2(N,ISP)=CH0
               RHOLM_1(N,ISP)=RHOLM(N,ISP)
            ENDDO
         ENDDO
      ELSE
         DO ISP=1,WDES%NCDIJ
            DO N=1,GRIDC%RC%NP
! save last difference
               CH1=CHTOT_2(N,ISP)
! build difference to current charge density
               CH0=CHTOT(N,ISP)-CHTOT_1(N,ISP)
               CHTOT_1(N,ISP)=CHTOT(N,ISP)
               CHTOT_2(N,ISP)=CH0
               IF (ICALLS>=2) CHTOT(N,ISP)=CHTOT(N,ISP)+PRED%ALPHA*CH0+PRED%BETA*CH1
            ENDDO

            IF (LDO_METAGGA()) THEN
               DO N=1,GRIDC%RC%NP
! save last difference
                  CH1=TAU_2(N,ISP)
! build difference to current charge density
                  CH0=KINEDEN%TAU(N,ISP)-TAU_1(N,ISP)
                  TAU_1(N,ISP)=KINEDEN%TAU(N,ISP)
                  TAU_2(N,ISP)=CH0
                  IF (ICALLS>=2) KINEDEN%TAU(N,ISP)=KINEDEN%TAU(N,ISP)+PRED%ALPHA*CH0+PRED%BETA*CH1
               ENDDO
            ENDIF
            DO N=1,NPAW
!  save last difference
               CH1=RHOLM_2(N,ISP)
!  build difference to current charge density
               CH0=RHOLM(N,ISP)-RHOLM_1(N,ISP)
               RHOLM_1(N,ISP)=RHOLM(N,ISP)
               RHOLM_2(N,ISP)=CH0
               IF (ICALLS>=2) RHOLM(N,ISP)=RHOLM(N,ISP)+PRED%ALPHA*CH0+PRED%BETA*CH1
            ENDDO
         ENDDO
      ENDIF
!---- add atomic charge-denisty for new positions
      IF ( new_version ) THEN
         CALL STUFAK(GRIDC,T_INFO,CSTRF)
         CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
         DO N=1,GRIDC%RC%NP
            CHTOT (N,1)= CHTOT(N,1)+ CD0(N)
         ENDDO

      ENDIF

      DEALLOCATE(CD0)
      IF ( new_version ) THEN
! indicate calling routine that structure factor was recalculated
         PRED%IPRE=-PRED%IPRE
         IF (PRED%IPRE==0  ) PRED%IPRE=-1
! and return
         GOTO 3000
      ENDIF
      IF ( ICALLS>=2 ) GOTO 3000

!***********************************************************************
!
! not enough information for the prediction (or PRED%INIPRE=4,5)
! try to get a better  start-charge-density by subtracting the
! charge-density of overlapping  atoms for the old position and adding
! the charge-density corresponding  to the new positions
!
!***********************************************************************

 2000 CONTINUE

!FURTH: Warning!!! Currently no extrapolation of magnetization!!!
      PRED%IPRE=-PRED%IPRE
      IF (PRED%INIPRE==5) PRED%IPRE=-1
      IF (PRED%IPRE==0  ) PRED%IPRE=-1

      ALLOCATE(CD0(GRIDC%MPLWV))

!---- subtract atomic charge-denisty for old positions
      CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
      DO N=1,GRIDC%RC%NP
        CHTOT(N,1)= CHTOT(N,1)- CD0(N)
      ENDDO
!---- add atomic charge-denisty for new positions

      CALL STUFAK(GRIDC,T_INFO,CSTRF)
      CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
      DO N=1,GRIDC%RC%NP
        CHTOT (N,1)= CHTOT(N,1)+ CD0(N)
      ENDDO

      DEALLOCATE(CD0)

 3000 RETURN
      END SUBROUTINE


!*********************************************************************
! routine which calculates the inproduct of two 3d-vectors
!*********************************************************************

      SUBROUTINE CLCD_NOIO(NIONS,A,B,SUM)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      DIMENSION A(3,NIONS)
      DIMENSION B(3,NIONS)

      SUM=0
      DO 100 N=1,NIONS
        SUM=SUM+A(1,N)*B(1,N)+A(2,N)*B(2,N)+A(3,N)*B(3,N)
  100 CONTINUE
      RETURN
      END SUBROUTINE
!***********************************************************************
!
! distribute columns over nodes
! determines how many colums are stored on each node
!
!***********************************************************************
      SUBROUTINE  DISTRIBUTE_COLUMNS( N, NPROC, NCOL )
!
! calculate number of columns per core
!
      IMPLICIT NONE

      INTEGER :: N_PER_PROC, REMAINDER, ME1, NPROC, N, NCOL(NPROC)
      INTEGER :: NCOL_MIN=32
      INTEGER :: NPROC_MAX

      NCOL = 0

      ! only NPROC_MAX cores take part in the communication
      ! otherwise communication gets too expensive
      NPROC_MAX=MIN(MAX(1,N/NCOL_MIN),NPROC)
      
      N_PER_PROC = N/NPROC_MAX
      REMAINDER   = N - NPROC_MAX * N_PER_PROC
      DO ME1 = 1,NPROC_MAX
         IF( ME1 <= REMAINDER )THEN
           NCOL(ME1) = N_PER_PROC + 1
         ELSE
           NCOL(ME1) = N_PER_PROC
         ENDIF
      ENDDO

      RETURN
      END SUBROUTINE

      END MODULE
