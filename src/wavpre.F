#include "symbol.inc"
#define LNEW

      MODULE mwavpre
      USE prec
      CONTAINS
!*************************** WAVPRE **********************************
! RCS:  $Id: wavpre.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
! this subroutine performes a prediction of the wavefunctions using
! the subspace-allignmt scheme proposed by  T. A. Arias et.al.
! in addition the chargedensity is speratly predicted
! a temporary file is used for storing the old wavefunctions and old
! postition of the ions
! following flags are important
!
! INIPRE  0 standard operation (performe extrapolation)
!         1 new startup (only POSION must be correct)
!         2 restart with existing FILE
!         3 restart with new ionic positions
!         4 extrapolate charge using atomic charge-densities
!           and extrapolate wavefunctions
!         5 extrapolate charge only, do not use a extra-file
!        10 allign the wavefunctions on the file to CPTWFP
!           CPTWFP is used as a work-array and is destroyed
! POSION  must contain the new positions
! CSTRF   must contain the !old! structure-factor
!         on return it holds the new structure-factor if IPRE<0
! CPTWFP  must contain the old wavefunctions
! IU      File Unit to use
! ICMPLX  size of COMPLEX(q) item
! IPRE    prediction of wavefunctions and eigenvalues performed
!           0  nothing done
!          -1  charge from overlapping atoms
!         <-1  wavefuntions predicted charge from overlapping atoms
!         > 1  wavefuntions and charge predicted
!         negative flags indicate that the structure factor
!         was recalculated
!
! short description of TMPCAR:
!  record                   contents            band k-point
!   1                   important parameters
!   2                   positions of ions
!   3                   previous-wavefunctions   1    1
!   4                   previous-change in wf.   1    1
!   5                   previous-wavefunctions   2    1
!   6                   previous-change in wf.   2    1
!                      ....
!   last records contain last charge and last  change in charge
!   in a blocked mode
!*********************************************************************

      SUBROUTINE WAVPRE(GRIDC,P,PRED,T_INFO,W,WDES,LATT_CUR,LOPEN, &
         CHTOT,RHOLM,NPAW, CSTRF, LMDIM,CQIJ,LOVERL,IU0)
      USE prec
      USE base
      USE poscar
      USE wave
      USE pseudo
      USE lattice
      USE mpimy
      USE mgrid
      USE charge
      USE wave_high
      USE choleski
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE( prediction)  PRED
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES1

      LOGICAL LOVERL
      LOGICAL LOPEN

      OVERLAP CQIJ(LMDIM,LMDIM,T_INFO%NIOND,WDES%ISPIN)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%ISPIN)
      REAL(q)    RHOLM(NPAW,WDES%ISPIN)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

      COMPLEX(q) ZDOTC
      CHARACTER (5) SZ
! work arrays not important for performance
      REAL(q) R(WDES%NBANDS),RWORK(3*WDES%NBANDS)
      REAL(q) D0(3,T_INFO%NIONS),D1(3,T_INFO%NIONS)
      REAL(q) D2(3,T_INFO%NIONS),POSIOL(3,T_INFO%NIONS)
! work arrays
      COMPLEX(q),ALLOCATABLE :: CD0(:),CD1(:),CPTWFL(:)
      REAL(q),ALLOCATABLE    :: RHO0(:),RHO1(:)
      GDEF,ALLOCATABLE :: CPRO0(:),CPRO1(:),CPROL(:)
      GDEF,ALLOCATABLE :: CA(:,:),CAP(:,:),CU(:,:)
#ifdef LNEW
      LOGICAL :: new_version = .TRUE.
#else
      LOGICAL :: new_version = .FALSE.
#endif
!R.S
      integer piudir

#ifdef MPI
      WRITE(IU0,*)'WAVPRE: it is not possible to use external file'
      STOP
#endif

!R.S
      piudir = PRED%IUDIR
      MPL=MAX(GRIDC%MPLWV,WDES%NRPLWV)
      NB =WDES%NBANDS
      ALLOCATE(CD0(MPL),CD1(MPL),CPTWFL(WDES%NRPLWV), &
               CPRO0(WDES%NPRO),CPRO1(WDES%NPRO),CPROL(WDES%NPRO), &
               CA(NB,NB),CAP(NB,NB),CU(NB,NB),RHO0(NPAW),RHO1(NPAW))

      IWARN=0
!=======================================================================
! PRED%INIPRE =5 forget about the wavefunctions and simply extrapolate charge
!=======================================================================
      IF (PRED%INIPRE==5) GOTO 2000
!***********************************************************************
! performe initialisation
!***********************************************************************
      IF (PRED%INIPRE==1 .OR.PRED%INIPRE==2 .OR.PRED%INIPRE==3) THEN

!-----------------------------------------------------------------------
! find the maximum number of plane-waves
! and the necessary record-length
!-----------------------------------------------------------------------
      NPLMAX=0
      DO NK=1,WDES%NKPTS
        NPLMAX=MAX(NPLMAX,WDES%NPLWKP(NK))
      ENDDO
      NPLMAX= NPLMAX+WDES%NPRO

      IRECL=MAX(T_INFO%NIONS*9*PRED%ICMPLX/2,PRED%ICMPLX*(NPLMAX+3))
!-----------------------------------------------------------------------
! some crazy operating-systems restrict the number of bytes/record
! in this case it must be possible to span several records with a
! single write-statement, IBLK ist set to the number of spanned-records
!-----------------------------------------------------------------------
      IBLK=1
!-----------------------------------------------------------------------
! open file and write starting parameters
! if INIPRE==3 (RESTART) file is allready open
!-----------------------------------------------------------------------
        IF (PRED%INIPRE/=3) THEN
        IF (LOPEN) THEN
        OPEN(PRED%IUDIR,FILE='TMPCAR',ACCESS='DIRECT', &
     &    FORM='UNFORMATTED',RECL=IRECL)
        ELSE
        OPEN(PRED%IUDIR,ACCESS='DIRECT',FORM='UNFORMATTED',RECL=IRECL)
        ENDIF
        ENDIF

!=======================================================================
! PRED%INIPRE=1
! create a new File
!=======================================================================
      IF (PRED%INIPRE==1) THEN
!-----------------------------------------------------------------------
! write out header
! set ICALLS to 0 ( on position of ions on file is correct)
!-----------------------------------------------------------------------
      ICALLS=0
      SZ='SETUP'
      D0=0
      D1=0
      WRITE(PIUDIR,REC=1) SZ,ICALLS,NPLMAX,IRECL,IBLK,T_INFO%NIONS
      WRITE(PIUDIR,REC=2) ((T_INFO%POSION(I,N),D0(I,N),D1(I,N),I=1,3),N=1,T_INFO%NIONS)
!-----------------------------------------------------------------------
! initialise wavefunctions
! (in principle only space is allocated the contents of W%CPTWFP is
!    not important)
!-----------------------------------------------------------------------
        IREC=3
        ICALL=0

        CD0=0
        CPRO0=0

        DO ISP=1,WDES%ISPIN
        DO NK=1,WDES%NKPTS
          NPL=WDES%NPLWKP(NK)
          CDCEL0=0
          DO N=1,WDES%NBANDS
          WRITE(PIUDIR,REC=IREC)  ICALL,W%CELEN(N,NK,ISP),CDCEL0, &
     &                           (W%CPTWFP(M,N,NK,ISP),M=1,NPL), &
     &                           (W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
          WRITE(PIUDIR,REC=IREC+IBLK)ICALL,W%CELEN(N,NK,ISP),CDCEL0, &
     &        (CD0(M)        ,M=1,NPL),(CPRO0(M)     ,M=1,WDES%NPRO)
          IREC=IREC+IBLK*2
          ENDDO
        ENDDO
        ENDDO
!-----------------------------------------------------------------------
! write charge-density use a blocked mode
!-----------------------------------------------------------------------
        DO ISP=1,WDES%ISPIN
        DO NBASE=0,GRIDC%RC%NP,NPLMAX/2
          ILEN=MIN(NPLMAX/2,GRIDC%RC%NP-NBASE)
          WRITE(PIUDIR,REC=IREC) (CHTOT(M+NBASE,ISP),CD0(M+NBASE),M=1,ILEN)
          IREC=IREC+1
        ENDDO
        ENDDO

      RHO0=0

        IF (NPAW > 0) THEN
        DO ISP=1,WDES%ISPIN
        DO NBASE=0,NPAW,NPLMAX
          ILEN=MIN(NPLMAX,NPAW-NBASE)
          WRITE(PIUDIR,REC=IREC) (RHOLM(M+NBASE,ISP),RHO0(M+NBASE),M=1,ILEN)
          IREC=IREC+1
        ENDDO
        ENDDO
        ENDIF

       WRITE(IU0,*)'prediction of wavefunctions initialized'
       GOTO 3000
!=======================================================================
! PRED%INIPRE=2
! check the consistency of a existing  file TMPCAR
!=======================================================================
      ELSE IF (PRED%INIPRE==2) THEN
        READ(PIUDIR,REC=1) SZ,ICALLS,NPLMAC,IRECLC,IBLK,NIONSC
        IF (SZ/='SETUP'.OR.NPLMAC/=NPLMAX &
     &      .OR.NIONSC/=T_INFO%NIONS.OR.IRECLC/=IRECL) THEN
        WRITE(IU0,*)'ERROR: WAVPRE : File TMPCAR is not correct', &
     &                   ' set NFIRST to 2'
        STOP
        ENDIF
!=======================================================================
! PRED%INIPRE=3
! reinitialize: store new ionic positions
!=======================================================================
      ELSE
      ICALLS=0
      SZ='SETUP'
      WRITE(PIUDIR,REC=1) SZ,ICALLS,NPLMAX,IRECL,IBLK,T_INFO%NIONS
      WRITE(PIUDIR,REC=2) ((T_INFO%POSION(I,N),D0(I,N),D1(I,N),I=1,3),N=1,T_INFO%NIONS)
      ENDIF
      GOTO 3000
      ENDIF
!***********************************************************************
!                  *****    FIRST STEP:    *****
!  read header of file TMPCAR
!  calculate the extrapolation parameters
!***********************************************************************
      READ (PIUDIR,REC=1) SZ,ICALLS,NPLMAX,IRECL,IBLK,NIONSC
      IF (PRED%INIPRE/=10) ICALLS=ICALLS+1
      WRITE(PIUDIR,REC=1) SZ,ICALLS,NPLMAX,IRECL,IBLK,NIONSC

!=======================================================================
! calculate extrapolation parameters ALPHA,BETA
!=======================================================================
      IF (PRED%INIPRE/=10)  THEN

      READ(PIUDIR,REC=2) ((POSIOL(I,N),D1(I,N),D2(I,N),I=1,3),N=1,T_INFO%NIONS)

      DO N=1,T_INFO%NIONS
        D0(1,N)=MOD(T_INFO%POSION(1,N)-POSIOL(1,N)+1.5_q,1._q)-.5_q
        D0(2,N)=MOD(T_INFO%POSION(2,N)-POSIOL(2,N)+1.5_q,1._q)-.5_q
        D0(3,N)=MOD(T_INFO%POSION(3,N)-POSIOL(3,N)+1.5_q,1._q)-.5_q
      ENDDO
      WRITE(PIUDIR,REC=2) ((T_INFO%POSION(I,N),D0(I,N),D1(I,N),I=1,3),N=1,T_INFO%NIONS)

      PRED%ALPHA=0
      PRED%BETA =0
!-----------------------------------------------------------------------
! after two calls linear extrapolation
! PRED%ALPHA=1 , PRED%BETA=0
!-----------------------------------------------------------------------
      IF (ICALLS==2) THEN
        CALL CLCD(T_INFO%NIONS,D1,D1,A0)
        CALL CLCD(T_INFO%NIONS,D0,D1,A1)

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
      CALL CLCD(T_INFO%NIONS,D0,D0,A0)
      CALL CLCD(T_INFO%NIONS,D1,D1,A2)
      CALL CLCD(T_INFO%NIONS,D2,D2,B2)
      CALL CLCD(T_INFO%NIONS,D0,D1,A1)
      CALL CLCD(T_INFO%NIONS,D0,D2,B1)
      CALL CLCD(T_INFO%NIONS,D1,D2,AB)

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
        WRITE(IU0,*)'positions are collinear'
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
!      WRITE(IU0,'(8E10.3)')PRED%ALPHA,PRED%BETA,A0,A2,B2,A1,B1,AB
      ENDIF

!-----ENDIF (ICALLS>=3)
      ENDIF
!---- ENDIF (PRED%INIPRE/=10)
      ENDIF
!=======================================================================
! Loop over k-Points and spins
!=======================================================================
      PRED%IPRE=0
      IREC=3
 spin:   DO ISP=1,WDES%ISPIN
 kpoint: DO NK=1,WDES%NKPTS

      CALL SETWDES(WDES,WDES1,NK)
      WDES1%NBANDS=1

      NPL=WDES%NPLWKP(NK)
      KREC=IREC
!***********************************************************************
!                  *****   SECOND STEP:    *****
! calculate the sub-space-allignment matrix
! notation: N  current wavefunction
!           Mp old wavefunction
!***********************************************************************
!-----------------------------------------------------------------------
! read in the old wavefunction and calculate U(N,M) = <N,Mp>
!-----------------------------------------------------------------------
 if1: IF (ICALLS>=2) THEN

      PRED%IPRE=ICALLS
      IREC=KREC

      CU=0

      band_build: DO N=1,WDES%NBANDS
      READ(PIUDIR,REC=IREC) ICALL,CELEL,CDCEL1, &
     &       (CPTWFL(M), M=1,NPL), (CPROL (M),M=1,WDES%NPRO)

! caclulate Q |Mp>  Q (overlap-matrix) and store in CPRO0

      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,1), CPROL(1),CPRO0(1))
! calculate the matrix U(N,M) = <N,Mp>
      NPRO=WDES%NPRO
      IF (.NOT.LOVERL) NPRO=0
      CALL ORTH2( &
     &  W%CPTWFP(1,1,NK,ISP),CPTWFL(1),W%CPROJ(1,1,NK,ISP),CPRO0(1),WDES%NBANDS, &
     &  N,1,NPL,NPRO,WDES%NRPLWV,WDES%NPROD,CU(1,1))
       IREC=IREC+IBLK*2

      ENDDO band_build
!-----------------------------------------------------------------------
! calculate the hermititian matrix AP(N,K)= CONJG(U(N,M))*U(K,M)
!-----------------------------------------------------------------------
      CALL GGEMM( trans, 'N', WDES%NBANDS, WDES%NBANDS, WDES%NBANDS, one, CU, WDES%NBANDS, &
     &            CU, WDES%NBANDS, zero, CAP, WDES%NBANDS)
!-----------------------------------------------------------------------
! Diagonolize the resulting Matrix
! calling LAPACK the result is
!    MATRIX(N,K)* EIGENV(K,M) = COS^2(THETA)* EIGENV(N,M)
! descending order of eigenvalues as mentioned in paper is nonsense
!-----------------------------------------------------------------------
#ifndef gammareal
      DO N1=1,WDES%NBANDS
        IF (ABS(AIMAG(CAP(N1,N1)))>1E-2_q) THEN
          WRITE(IU0,*)'WARNING: Sub-Space-Matrix is not hermitian wav', &
     &              AIMAG(CAP(N1,N1))
        ENDIF
        CAP(N1,N1)= REAL( CAP(N1,N1) ,KIND=q)
      ENDDO
#endif
#ifdef  gammareal
      CALL DSYEV &
     &    ('V','U',WDES%NBANDS,CAP,WDES%NBANDS,R, &
     &           CA,WDES%NBANDS*WDES%NBANDS,IFAIL)
#else
      CALL ZHEEV &
     &    ('V','U',WDES%NBANDS,CAP,WDES%NBANDS,R, &
     &           CA,WDES%NBANDS*WDES%NBANDS,RWORK,IFAIL)
#endif
      IF (IFAIL/=0) THEN
         WRITE(IU0,*) 'ERROR WAVPRE: Call to routine ZHEEV failed! '// &
     &              'Error code was ',IFAIL
         STOP
      ENDIF

! Store the matrix in A(N,K)=CONJG(EIGENV(K,N))
      DO N1=1,WDES%NBANDS
      DO N2=1,WDES%NBANDS
        CA(N1,N2)= GCONJG(CAP(N2,N1))
      ENDDO
      ENDDO

! caclulate AP(N,M)=A(N,K)*CONJG(U(M,K))
      CALL GGEMM( 'N', trans, WDES%NBANDS, WDES%NBANDS, WDES%NBANDS, one, CA, WDES%NBANDS, &
     &            CU, WDES%NBANDS, zero , CAP, WDES%NBANDS)
!-----------------------------------------------------------------------
! AP(N,K)=Diag(1/-Sqrt(Eigenvalue))*U2(N,K)
!-----------------------------------------------------------------------
      DO N1=1,WDES%NBANDS
      ESQR=1/SQRT(R(N1))
      DO N2=1,WDES%NBANDS
        CAP(N1,N2)=CAP(N1,N2)*ESQR
      ENDDO
      ENDDO

! CU(N,M)=CONJG(CAP(K,N))*CA(K,M)

      CALL GGEMM(trans, 'N', WDES%NBANDS, WDES%NBANDS, WDES%NBANDS, one, CAP, WDES%NBANDS, &
     &             CA, WDES%NBANDS, zero , CU, WDES%NBANDS)
!=======================================================================
! set CAP to the invers transformation
!=======================================================================
      DO N2=1,WDES%NBANDS
      DO N1=1,WDES%NBANDS
        CAP(N1,N2)= GCONJG(CU(N2,N1))
      ENDDO
      ENDDO

  ENDIF if1
!***********************************************************************
!                  *****   INIPRE = 10      *****
! rotate the wavfunctions on the file so that they are
! alligned with the current wavefunctions
! the contents of CPTWFP is destroyed during this operation
!***********************************************************************
  pre: IF (PRED%INIPRE==10) THEN

    if2: IF (ICALLS>=2) THEN

! read in old wavefunction CPTWFL and rotate it by multiplying with CAP

      IREC=KREC
      DO N=1,WDES%NBANDS
      READ(PIUDIR,REC=IREC)      ICALL,CELEL,CDCEL1, &
     &  (W%CPTWFP(M,N,NK,ISP),M=1,NPL),(W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
      IREC=IREC+IBLK*2
      ENDDO

      CALL LINCOM('F',W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CAP(1,1), &
     &      WDES%NBANDS,WDES%NBANDS,NPL,WDES%NPRO,WDES%NRPLWV,WDES%NPROD,WDES%NBANDS, &
     &      W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))


      IREC=KREC
      DO N=1,WDES%NBANDS
      WRITE(PIUDIR,REC=IREC)      ICALL,CELEL,CDCEL1, &
     &  (W%CPTWFP(M,N,NK,ISP),M=1,NPL),(W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
      IREC=IREC+IBLK*2
      ENDDO

! read in last change of wavefunction and rotate by CAP

      IREC=KREC
      DO N=1,WDES%NBANDS

      READ(PIUDIR,REC=IREC+IBLK) ICALL,CELEL,CDCEL1, &
     &  (W%CPTWFP(M,N,NK,ISP),M=1,NPL),(W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
      IREC=IREC+IBLK*2
      ENDDO

      CALL LINCOM('F',W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CAP(1,1), &
     &      WDES%NBANDS,WDES%NBANDS,NPL,WDES%NPRO,WDES%NRPLWV,WDES%NPROD,WDES%NBANDS, &
     &      W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))


      IREC=KREC
      DO N=1,WDES%NBANDS
      WRITE(PIUDIR,REC=IREC+IBLK) ICALL,CELEL,CDCEL1, &
     &  (W%CPTWFP(M,N,NK,ISP),M=1,NPL),(W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
      IREC=IREC+IBLK*2
      ENDDO

   ENDIF if2

!---- ELSE (PRED%INIPRE==10)
  ELSE pre
!***********************************************************************
!                  *****   INIPRE = 0 or 4  *****
! align current wavefunctions with those on the file,
! and performe extrapolation
!***********************************************************************
      IF (ICALLS>=2) THEN
      CALL LINCOM('F',W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CU(1,1), &
           WDES%NBANDS,WDES%NBANDS,NPL,WDES%NPRO,WDES%NRPLWV,WDES%NPROD,WDES%NBANDS, &
           W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
      ENDIF
!=======================================================================
! extrapolation of wavefunctions
! loop over all bands
!=======================================================================
      IREC=KREC
!-----------------------------------------------------------------------
! read in old wafefunction CPTWFL and last change in Wavefunction CD1
! and caclulate Q |MP>  Q (overlap-matrix) and store in CPRO0
!-----------------------------------------------------------------------
      band: DO N=1,WDES%NBANDS
      READ(PIUDIR,REC=IREC)      ICALL,CELEL,CDCEL1, &
              (CPTWFL(M),M=1,NPL),(CPROL(M),M=1,WDES%NPRO)

      READ(PIUDIR,REC=IREC+IBLK) ICALL,CELEL,CDCEL1, &
              (CD1(M)   ,M=1,NPL),(CPRO1(M),M=1,WDES%NPRO)

! ICALL must not be smaller then ICALLS
! after a reinitilization ICALLS is set to 0, but no ICALL is reset

      ICALL=MIN(ICALL,ICALLS-1)
      CALL OVERL(WDES1, LOVERL,LMDIM,CQIJ(1,1,1,1), CPROL(1),CPRO0(1))
!-----------------------------------------------------------------------
! test orthogonality of CPTWFP CPTWFL
! if they are orthogonal band crossing occured, reset ICALL to 1
!-----------------------------------------------------------------------
      WFMAG = REAL( ZDOTC(NPL,W%CPTWFP(1,N,NK,ISP),1,CPTWFL,1) ,KIND=q)
      WFMAGL= REAL( ZDOTC(NPL,CPTWFL            ,1,CPTWFL,1) ,KIND=q)
      WFMAGLL=WFMAGL
      IF (LOVERL) THEN
      WFMAG =WFMAG + REAL( GDOTC( WDES%NPRO,W%CPROJ(1,N,NK,ISP),1,CPRO0,1) ,KIND=q)
      WFMAGL=WFMAGL+ REAL( GDOTC( WDES%NPRO,CPROL            ,1,CPRO0,1) ,KIND=q)
      ENDIF
      IF (ABS(WFMAGL-1._q) > 1E-3_q) THEN
        IWARN=IWARN+1
      ENDIF
      WFMAG=WFMAG/WFMAGL

      IF (ICALL/=0 .AND.ABS(WFMAG)>0.90_q) THEN
        ICALL=ICALL+1
        DO M=1,NPL
          CD0(M)  =W%CPTWFP(M,N,NK,ISP)-CPTWFL(M)
        ENDDO
        DO M=1,WDES%NPRO
          CPRO0(M)=W%CPROJ (M,N,NK,ISP)-CPROL (M)
        ENDDO

        CDCEL0=  W%CELEN(N,NK,ISP)-CELEL
      ELSEIF (ICALL==0) THEN
        ICALL=1
      ELSE
        ICALL=1
        WRITE(IU0,'(A,I4,F8.4)')'Information: wave function orthogonal band ',N,WFMAG
      ENDIF

      WRITE(PIUDIR,REC=IREC)      ICALL,W%CELEN(N,NK,ISP),CDCEL0, &
     &                       (W%CPTWFP(M,N,NK,ISP),M=1,NPL), &
     &                       (W%CPROJ(M,N,NK,ISP),M=1,WDES%NPRO)
      WRITE(PIUDIR,REC=IREC+IBLK) ICALL,W%CELEN(N,NK,ISP),CDCEL0, &
     &    (CD0(M)        ,M=1,NPL),(CPRO0(M)     ,M=1,WDES%NPRO)

!-----------------------------------------------------------------------
! extrapolate  now
!-----------------------------------------------------------------------
      IF (ICALL>=3) THEN
        DO M=1,NPL
          W%CPTWFP(M,N,NK,ISP)=W%CPTWFP(M,N,NK,ISP)+PRED%ALPHA*CD0(M)+PRED%BETA*CD1(M)
        ENDDO
        DO M=1,WDES%NPRO
          W%CPROJ (M,N,NK,ISP)=W%CPROJ (M,N,NK,ISP)+ &
     &                       PRED%ALPHA*CPRO0(M)+PRED%BETA*CPRO1(M)
        ENDDO

         W%CELEN(N,NK,ISP)= &
     &      W%CELEN(N,NK,ISP)+PRED%ALPHA*CDCEL0+PRED%BETA*CDCEL1
      ELSE IF (ICALL>=2) THEN
        DO M=1,NPL
          W%CPTWFP(M,N,NK,ISP)=W%CPTWFP(M,N,NK,ISP)+CD0(M)
        ENDDO
        DO M=1,WDES%NPRO
          W%CPROJ (M,N,NK,ISP)=W%CPROJ (M,N,NK,ISP)+CPRO0(M)
        ENDDO
        W%CELEN(N,NK,ISP)=W%CELEN(N,NK,ISP)+CDCEL0
      ENDIF

      IREC=IREC+IBLK*2
!-----------------------------------------------------------------------
      ENDDO band
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
! backward transformation (I found that this is not necessary, gK )
!-----------------------------------------------------------------------
!      IF (ICALLS>=2) THEN
!      CALL LINCOM('F',W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP),CAP,
!     &      WDES%NBANDS,WDES%NBANDS,NPL,WDES%NPRO,WDES%NRPLWV,WDES%NPROD,WDES%NBANDS,
!     &      W%CPTWFP(:,:,NK,ISP),W%CPROJ(:,:,NK,ISP))
!      ENDIF
  ENDIF pre
!-----------------------------------------------------------------------
  ENDDO kpoint
  ENDDO spin
!-----------------------------------------------------------------------

      IF (IWARN/=0) THEN
        WRITE(IU0,*)'WARNING: WAVPRE: wavefunctions on file TMPCAR', &
     &            ' are not orthogonal'
      ENDIF

      IF (PRED%INIPRE==10) GOTO 3000

!***********************************************************************
!   ***           extrapolate charge-density            ***
!
! PRED%INIPRE = 4,5  use atomic charge-densities instead of prediction
!***********************************************************************

      IF (PRED%INIPRE==4) ICALLS=MIN(1,ICALLS)

!---- subtract atomic charge-denisty for old positions
      IF ( new_version ) THEN
         CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
         DO N=1,GRIDC%RC%NP
            CHTOT(N,1)= CHTOT(N,1)- CD0(N)
         ENDDO
      ENDIF

      DO 740 ISP=1,WDES%ISPIN

      ICREC=IREC
!--- last charge density and last difference
!   (remind: we are using a blocked storage-mode)
      DO NBASE=0,GRIDC%RC%NP,NPLMAX/2
        ILEN=MIN(NPLMAX/2,GRIDC%RC%NP-NBASE)
        READ(PIUDIR,REC=IREC) (CD0(M+NBASE),CD1(M+NBASE),M=1,ILEN)
        IREC=IREC+1
      ENDDO

!  build difference to current charge density
      DO M=1,GRIDC%RC%NP
        CD0(M)=CHTOT(M,ISP)-CD0(M)
      ENDDO

! write current charge density and difference
      IREC=ICREC
      DO NBASE=0,GRIDC%RC%NP,NPLMAX/2
        ILEN=MIN(NPLMAX/2,GRIDC%RC%NP-NBASE)
        WRITE(PIUDIR,REC=IREC) (CHTOT(M+NBASE,ISP),CD0(M+NBASE),M=1,ILEN)
        IREC=IREC+1
      ENDDO

      IF (ICALLS>=2) THEN
        DO M=1,GRIDC%RC%NP
          CHTOT(M,ISP)=CHTOT(M,ISP)+PRED%ALPHA*CD0(M)+PRED%BETA*CD1(M)
        ENDDO
      ENDIF

  740 CONTINUE

      IF (NPAW>0) THEN
      DO 750 ISP=1,WDES%ISPIN

      ICREC=IREC
!--- last RHOLM and last difference
       DO NBASE=0,NPAW,NPLMAX
          ILEN=MIN(NPLMAX,NPAW-NBASE)
          READ(PIUDIR,REC=IREC) (RHO0(M+NBASE),RHO1(M+NBASE),M=1,ILEN)
          IREC=IREC+1
      ENDDO

!  build difference to current charge density
      DO M=1,NPAW
        RHO0(M)=RHOLM(M,ISP)-RHO0(M)
      ENDDO

! write current charge density and difference
      IREC=ICREC
       DO NBASE=0,NPAW,NPLMAX
          ILEN=MIN(NPLMAX,NPAW-NBASE)
          WRITE(PIUDIR,REC=IREC) (RHOLM(M+NBASE,ISP),RHO0(M+NBASE),M=1,ILEN)
          IREC=IREC+1
      ENDDO

      IF (ICALLS>=2) THEN
        DO M=1,NPAW
          RHOLM(M,ISP)=RHOLM(M,ISP)+PRED%ALPHA*RHO0(M)+PRED%BETA*RHO1(M)
        ENDDO
      ENDIF

  750 CONTINUE
      ENDIF

!---- add atomic charge-denisty for new positions
      IF ( new_version ) THEN
         CALL STUFAK(GRIDC,T_INFO,CSTRF)
         CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD0)
         DO N=1,GRIDC%RC%NP
            CHTOT (N,1)= CHTOT(N,1)+ CD0(N)
         ENDDO
      ENDIF

      IF ( new_version ) THEN
! indicate calling routine that structure factor was recalculated
         PRED%IPRE=-PRED%IPRE
         IF (PRED%IPRE==0  ) PRED%IPRE=-1
! and return
         GOTO 3000
      ENDIF
      IF (ICALLS>=2) GOTO 3000

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

 3000 CONTINUE

      DEALLOCATE(CD0,CD1,CPTWFL,CPRO0,CPRO1,CPROL,CA,CAP,CU,RHO0,RHO1)

      RETURN
      END SUBROUTINE


!*********************************************************************
! routine which calculates the inproduct of two 3d-vectors
!*********************************************************************

      SUBROUTINE CLCD(NIONS,A,B,SUM)
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
      END MODULE






